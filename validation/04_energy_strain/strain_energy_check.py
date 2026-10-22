#!/usr/bin/env python3
"""
strain_energy_check.py - strain energy of monolayer, bilayer and bulk hBN (AA' stacking) under
in-plane strain with a NEP potential in GPUMD, compared with DFT from the literature.

Usage
    python strain_energy_check.py nep_best.txt                    # one potential
    python strain_energy_check.py nep_v1.txt nep_best.txt         # compare several
    python strain_energy_check.py nep_best.txt --jobs 8           # GPUMD runs at a time per GPU
    python strain_energy_check.py nep_best.txt --force            # ignore the cache

Method (static, 0 K)
  1. Each structure is built in an orthorhombic cell (x = zigzag, y = armchair direction) and
     relaxed with atoms and box together ('minimize fire ... 1'). This is the zero of strain:
     every strain is relative to the potential's own equilibrium lattice.
  2. The relaxed cell is strained by the engineering strain eps (a -> a (1 + eps)):
       biaxial    x and y
       zigzag     x only   (uniaxial strain along the zigzag direction)
       armchair   y only   (uniaxial strain along the armchair direction)
     The other in-plane direction and the cell height stay fixed (pure strain, as in the DFT
     reference below). The atoms are relaxed at fixed cell ('minimize fire ...').
  3. Strain energy per atom: dE(eps) = [E(eps) - E(0)] / N, with E(0) from the same fixed-cell
     relaxation of the unstrained cell.

Literature
  - monolayer: Peng, Ji & De, Comput. Mater. Sci. 56, 11 (2012), PBE. Their energy-strain
    curves (Fig. 2) are not tabulated; they are rebuilt from the fifth-order continuum fit to
    their DFT data (elastic constants of Table 1), converted from Lagrangian to engineering
    strain. Same strain states as here; atoms relaxed in-plane only.
  - bulk: harmonic limit dE = (1/2) V_atom sum C_ij eps_i eps_j from the elastic constants of
    experiment (Bosak et al. 2006) and DFT (Thiemann et al. 2020, PBE+D3); valid at small
    strain only.
  - bilayer: no published energy-strain curve found.
  - your own DFT: CONFIG["dft_energies"], a CSV file (see below).

Parallel runs
  Each structure gets its own GPU (with CUDA_VISIBLE_DEVICES from Slurm): with 3 GPUs,
  monolayer, bilayer and bulk run fully in parallel; with 2, monolayer and bulk share GPU 0;
  with 1, all share it. On each GPU up to --jobs runs (default 4) run at the same time.

Output (in strain_energy_check/)
    strain_energy_<names>.png / .pdf       dE vs strain, one panel per system
    strain_energy_<names>.csv              all energies
    <potential>/<system>/                  GPUMD runs (relax/, <mode>_<strain>/)
    <potential>/results.json               cached result

Needs numpy and matplotlib.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np

# =============================================================================
# CONFIG - edit once
# =============================================================================
CONFIG = {
    "gpumd": "/scratch/work/fakhran1/GPUMD/src/gpumd",

    "a0": 2.504,                     # starting lattice (A); relaxed before straining
    "c0": 6.66,                      # bulk
    "d0": 3.33,                      # bilayer interlayer distance (start)
    "vacuum": 20.0,                  # vacuum above and below the monolayer and the bilayer (A)

    # engineering strains, eps = da / a
    "strains": [-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10],
    "modes": ["biaxial", "zigzag", "armchair"],

    # relaxation of the unstrained structure (atoms + box)
    "box_force_tol": 1e-5,
    "box_max_steps": 100000,
    # relaxation of the atoms in every strained cell (fixed box)
    "minimize": "fire",
    "force_tol": 1e-5,
    "max_steps": 100000,
    "dump_line": "dump_xyz 1 relaxed.xyz",

    "nep_cutoff": 6.0,               # largest radial cutoff of your NEP (A)

    # optional: your DFT energies, CSV with the columns
    #   system,mode,strain,dE      (strain in %, dE in meV/atom; system: monolayer|bilayer|bulk;
    #                               mode: biaxial|zigzag|armchair)
    "dft_energies": None,
    "dft_label": "DFT (this work)",
}
OUT = "strain_energy_check"
# =============================================================================

SYSTEMS = ["monolayer", "bilayer", "bulk"]
SQ3 = np.sqrt(3.0)
EV = 1.602176634e-19                 # J
GPA_A3_TO_MEV = 1e9 * 1e-30 / EV * 1e3

# Peng, Ji & De, Comput. Mater. Sci. 56, 11 (2012), Table 1 (N/m). TOEC and FFOEC are negative,
# FOEC positive (as stated in the paper). Lagrangian strain, a0 = 2.512 A.
PENG = "Peng, Ji & De, Comput. Mater. Sci. 56, 11 (2012), PBE"
PENG_A0 = 2.512
PC = dict(C11=293.2, C12=66.1, C111=-2513.6, C112=-425.0, C222=-2284.2,
          C1111=16547.0, C1112=2609.0, C1122=2215.0, C2222=12288.0,
          C11111=-65265.0, C11112=-8454.0, C11122=-28556.0, C12222=-36955.0, C22222=-100469.0)

# bulk elastic constants (GPa) and the lattice they belong to (a, c in A)
BULK_CIJ = {
    "exp": (811.0, 169.0, 2.504, 6.6612, "experiment (Bosak et al. 2006), harmonic"),
    "dft": (881.8, 195.6, 2.51, 6.61, "DFT PBE+D3 (Thiemann et al. 2020), harmonic"),
}


# ----------------------------------------------------------------------------- literature
def peng_energy(mode, eps):
    """monolayer strain energy (meV/atom) at engineering strain eps, from the continuum fit of
    Peng et al. (their Eqs. 7, 10 and 14, integrated over the Lagrangian strain)"""
    c = PC
    eta = np.asarray(eps) + 0.5 * np.asarray(eps) ** 2
    if mode == "zigzag":        # Eq. (7)
        phi = c["C11"] / 2 * eta ** 2 + c["C111"] / 6 * eta ** 3 + c["C1111"] / 24 * eta ** 4 \
              + c["C11111"] / 120 * eta ** 5
    elif mode == "armchair":    # Eq. (10)
        phi = c["C11"] / 2 * eta ** 2 + c["C222"] / 6 * eta ** 3 + c["C2222"] / 24 * eta ** 4 \
              + c["C22222"] / 120 * eta ** 5
    else:                       # Eq. (14); both stresses do work: dPhi = 2 Sigma d(eta)
        A = 2 * c["C111"] - c["C222"] + 3 * c["C112"]
        B = 1.5 * c["C1111"] + 4 * c["C1112"] + 3 * c["C1122"] - 0.5 * c["C2222"]
        D = 3 * c["C11111"] + 10 * c["C11112"] - 5 * c["C12222"] + 10 * c["C11122"] - 2 * c["C22222"]
        phi = (c["C11"] + c["C12"]) * eta ** 2 + A / 3 * eta ** 3 + B / 12 * eta ** 4 + D / 60 * eta ** 5
    area = SQ3 / 4 * PENG_A0 ** 2 * 1e-20          # reference area per atom (m^2)
    return phi * area / EV * 1e3                   # N/m * m^2 = J -> meV


def bulk_harmonic(mode, eps, src):
    """harmonic strain energy (meV/atom) of bulk hBN at fixed c from C11, C12"""
    C11, C12, a, c, _ = BULK_CIJ[src]
    v_atom = SQ3 / 2 * a * a * c / 4
    eps = np.asarray(eps)
    w = (C11 + C12) * eps ** 2 if mode == "biaxial" else 0.5 * C11 * eps ** 2
    return w * v_atom * GPA_A3_TO_MEV


# ----------------------------------------------------------------------------- structures
def reps(system):
    """supercell repeats; every periodic length must stay above 2 x cutoff at the largest
    compression"""
    rc = 2 * CONFIG["nep_cutoff"] + 0.5
    shrink = 0.97 * (1 + min(min(CONFIG["strains"]), 0.0))
    nx = int(np.ceil(rc / (shrink * CONFIG["a0"])))
    ny = int(np.ceil(rc / (shrink * CONFIG["a0"] * SQ3)))
    nz = int(np.ceil(rc / (0.95 * CONFIG["c0"]))) if system == "bulk" else 1
    return nx, ny, nz


def build(system):
    """orthorhombic AA' cell: x = zigzag, y = armchair; 4 atoms per layer per rectangle"""
    a, c, d = CONFIG["a0"], CONFIG["c0"], CONFIG["d0"]
    nx, ny, nz = reps(system)
    base = np.array([[0, 0], [0, a / SQ3], [a / 2, a * SQ3 / 2], [a / 2, a * SQ3 / 2 + a / SQ3]])
    names = ["B", "N", "B", "N"]
    if system == "bulk":
        zs = [c * (m + 0.25) for m in range(nz)] + [c * (m + 0.75) for m in range(nz)]
        flips = [False] * nz + [True] * nz
        Lz = nz * c
    else:
        nl = 1 if system == "monolayer" else 2
        zs = [CONFIG["vacuum"] + k * d for k in range(nl)]
        flips = [k % 2 == 1 for k in range(nl)]
        Lz = 2 * CONFIG["vacuum"] + (nl - 1) * d
    sp, pos = [], []
    for z, flip in zip(zs, flips):
        for i in range(nx):
            for j in range(ny):
                for (x, y), e in zip(base, names):
                    sp.append(("N" if e == "B" else "B") if flip else e)
                    pos.append([x + i * a, y + j * a * SQ3, z])
    lat = np.diag([nx * a, ny * a * SQ3, Lz])
    return sp, np.array(pos), lat


def write_xyz(path, sp, pos, lat):
    with open(path, "w") as f:
        f.write(f"{len(sp)}\n")
        f.write('Lattice="' + " ".join(f"{x:.10f}" for x in np.asarray(lat).flatten()) +
                '" pbc="T T T" Properties=species:S:1:pos:R:3\n')
        for s, p in zip(sp, pos):
            f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def read_last_xyz(path):
    """species, positions and lattice of the last frame of an extended-XYZ file"""
    lines = open(path).read().splitlines()
    k, last = 0, None
    while k < len(lines):
        if not lines[k].strip():
            k += 1
            continue
        n = int(lines[k].split()[0])
        last = (k, n)
        k += n + 2
    k, n = last
    head = lines[k + 1]
    low = head.lower()
    lat = np.array(head[low.index("lattice=") + 8:].split('"')[1].split(), float).reshape(3, 3)
    props = head[low.index("properties=") + 11:].split()[0].split(":") if "properties=" in low \
        else "species:S:1:pos:R:3".split(":")
    col, i_sp, i_pos = 0, 0, 1
    for name, typ, cnt in zip(props[0::3], props[1::3], props[2::3]):
        if name.lower() == "species":
            i_sp = col
        if name.lower() == "pos":
            i_pos = col
        col += int(cnt)
    rows = [lines[k + 2 + q].split() for q in range(n)]
    return [r[i_sp] for r in rows], np.array([r[i_pos:i_pos + 3] for r in rows], float), lat


# ----------------------------------------------------------------------------- GPUMD
TAIL = """velocity 1e-6
ensemble nve
time_step 1e-6
dump_thermo 1
{dump}run 1
"""


def gpus():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return [g for g in v.split(",") if g.strip()] or [None]


def run_gpumd(workdir, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    proc = subprocess.run([CONFIG["gpumd"]], cwd=workdir, capture_output=True, text=True, env=env)
    open(os.path.join(workdir, "gpumd.log"), "w").write(proc.stdout + proc.stderr)
    thermo = os.path.join(workdir, "thermo.out")
    if proc.returncode != 0 or not os.path.isfile(thermo):
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        raise RuntimeError(f"GPUMD failed in {workdir}:\n  " + "\n  ".join(tail))
    rows = [l for l in open(thermo).read().splitlines() if l.strip()]
    return float(rows[-1].split()[2])               # potential energy (eV)


class GpuPool:
    """one semaphore per GPU: at most `per_gpu` GPUMD runs at the same time on each GPU"""
    def __init__(self, per_gpu):
        self.gpus = gpus()
        self.per_gpu = max(1, per_gpu)
        self.sem = {g: threading.Semaphore(self.per_gpu) for g in self.gpus}

    def run(self, dirs, gpu):
        """run the folders on one GPU, up to per_gpu at a time; returns their energies"""
        def one(d):
            with self.sem[gpu]:
                return run_gpumd(d, gpu)
        with ThreadPoolExecutor(max_workers=self.per_gpu) as ex:
            return list(ex.map(one, dirs))


def prepare(workdir, pot, sp, pos, lat, box):
    """box=True: atoms + box relaxation (writes relaxed.xyz); False: atoms only, fixed box"""
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    write_xyz(os.path.join(workdir, "model.xyz"), sp, pos, lat)
    if box:
        head = f"potential {pot}\nminimize fire {CONFIG['box_force_tol']} {CONFIG['box_max_steps']} 1\n"
        dump = CONFIG["dump_line"].strip() + "\n"
    else:
        head = f"potential {pot}\nminimize {CONFIG['minimize']} {CONFIG['force_tol']} {CONFIG['max_steps']}\n"
        dump = ""
    open(os.path.join(workdir, "run.in"), "w").write(head + TAIL.format(dump=dump))
    return workdir


# ----------------------------------------------------------------------------- per potential
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def compute(pot, per_gpu, force):
    tag = os.path.splitext(os.path.basename(pot))[0]
    root = os.path.join(OUT, tag)
    res_file = os.path.join(root, "results.json")
    keys = ["a0", "c0", "d0", "vacuum", "strains", "modes", "box_force_tol", "box_max_steps",
            "minimize", "force_tol", "max_steps", "nep_cutoff"]
    key = json.loads(json.dumps({"v": 1, "sha": sha(pot), "config": {k: CONFIG[k] for k in keys}}))
    if not force and os.path.isfile(res_file):
        old = json.load(open(res_file))
        if old.get("key") == key:
            print(f"{tag}: cached ({res_file})")
            return old
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    potabs = os.path.abspath(pot)
    logf = open(os.path.join(root, "log.txt"), "w")

    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    log(f"=== {tag} ===   (GPU(s) {', '.join(str(g) for g in gpus())}; up to {per_gpu} GPUMD runs at a time per GPU)")
    out = {"key": key, "potential": os.path.basename(pot), "systems": {}}

    # Each structure runs on its own GPU (round-robin if there are fewer GPUs than structures),
    # so monolayer, bilayer and bulk proceed in parallel and independently of each other.
    pool = GpuPool(per_gpu)
    gpu_of = {s: pool.gpus[i % len(pool.gpus)] for i, s in enumerate(SYSTEMS)}
    for s in SYSTEMS:
        log(f"{s:<10} -> GPU {gpu_of[s]}")
    scale = {"biaxial": (1, 1), "zigzag": (1, 0), "armchair": (0, 1)}

    def one_system(s):
        gpu = gpu_of[s]
        # --- 1. relax the unstrained structure (atoms + box)
        d = prepare(os.path.join(root, s, "relax"), potabs, *build(s), box=True)
        pool.run([d], gpu)
        f = os.path.join(d, "relaxed.xyz")
        if not os.path.isfile(f):
            raise RuntimeError(f"{f} not found - check CONFIG['dump_line'] for your GPUMD version")
        sp, pos, lat = read_last_xyz(f)
        nx, ny, nz = reps(s)
        a = 0.5 * (np.linalg.norm(lat[0]) / nx + np.linalg.norm(lat[1]) / (ny * SQ3))
        extra = ""
        if s == "bulk":
            extra = f", c = {np.linalg.norm(lat[2]) / nz:.4f} A"
        if s == "bilayer":
            z = np.sort(pos[:, 2])
            gap = np.argmax(np.diff(z))
            extra = f", d = {z[gap + 1:].mean() - z[:gap + 1].mean():.4f} A"
        res = {"a": a, "natoms": len(sp), "E": {}}
        log(f"{s:<10} relaxed: a = {a:.4f} A{extra}   ({len(sp)} atoms, {nx}x{ny}x{nz} rectangular cells)")

        # --- 2. strained cells (atoms relaxed at fixed box), all on this structure's GPU
        todo = [("none", 0.0)] + [(m, e) for m in CONFIG["modes"] for e in CONFIG["strains"] if e != 0]
        dirs = []
        for mode, eps in todo:
            fx, fy = (1, 1) if mode == "none" else scale[mode]
            F = np.array([1 + eps * fx, 1 + eps * fy, 1.0])
            name = "unstrained" if mode == "none" else f"{mode}_{eps:+.4f}"
            dirs.append(prepare(os.path.join(root, s, name), potabs, sp, pos * F, lat * F, box=False))
        energies = pool.run(dirs, gpu)
        E0 = energies[0]
        for (mode, eps), e in zip(todo, energies):
            for m in (CONFIG["modes"] if mode == "none" else [mode]):
                res["E"].setdefault(m, {})[f"{eps:+.4f}"] = (e - E0) / len(sp) * 1e3   # meV/atom
        log(f"{s:<10} done ({len(dirs)} strained/unstrained cells)")
        return res

    with ThreadPoolExecutor(max_workers=len(SYSTEMS)) as ex:
        for s, res in zip(SYSTEMS, ex.map(one_system, SYSTEMS)):
            out["systems"][s] = res
    for s in SYSTEMS:
        for m in CONFIG["modes"]:
            d = out["systems"][s]["E"][m]
            log(f"{s:<10} {m:<9} " + "  ".join(f"{100 * float(k):+.1f}%: {d[k]:.1f}"
                                                 for k in sorted(d, key=float)) + "  meV/atom")
    json.dump(out, open(res_file, "w"))
    logf.close()
    return out


# ----------------------------------------------------------------------------- your DFT
def user_dft():
    path = CONFIG["dft_energies"]
    res = {}
    if not path:
        return res
    if not os.path.isfile(path):
        print(f"DFT energies not found: {path}")
        return res
    for line in open(path):
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4 or line.startswith("#") or p[0] == "system":
            continue
        res.setdefault(p[0], {}).setdefault(p[1], []).append((float(p[2]), float(p[3])))
    return res


# ----------------------------------------------------------------------------- table + CSV
def report(results, dft, tag):
    strains = sorted(CONFIG["strains"])
    rows = [["potential", "system", "mode", "strain_percent", "dE_meV_per_atom"]]
    lines = ["Strain energy dE = [E(eps) - E(0)] / N (meV/atom), NEP vs literature", ""]
    for s in SYSTEMS:
        lines.append(f"=== {s} ===")
        head = f"  {'strain':>7}"
        cols = []
        for m in CONFIG["modes"]:
            for r in results:
                name = "NEP" if len(results) == 1 else os.path.splitext(r["potential"])[0][:8]
                cols.append((f"{m}/{name}", lambda e, r=r, m=m:
                             r["systems"][s]["E"][m].get(f"{e:+.4f}")))
            if s == "monolayer":
                cols.append((f"{m}/Peng", lambda e, m=m: float(peng_energy(m, e))))
            if s == "bulk":
                cols.append((f"{m}/harm", lambda e, m=m: float(bulk_harmonic(m, e, "exp"))))
        head += "".join(f"{n:>17}" for n, _ in cols)
        lines += [head, "  " + "-" * (len(head) - 2)]
        for e in strains:
            row = f"  {100 * e:+6.1f}%"
            for _, fn in cols:
                v = fn(e)
                row += f"{v:>17.1f}" if v is not None else f"{'-':>17}"
            lines.append(row)
        lines.append("")
        for r in results:
            for m in CONFIG["modes"]:
                for k, v in sorted(r["systems"][s]["E"][m].items(), key=lambda t: float(t[0])):
                    rows.append([r["potential"], s, m, f"{100 * float(k):.2f}", f"{v:.4f}"])
    lines += ["Notes",
              "  Strain is relative to the potential's own equilibrium lattice; the other in-plane",
              "  direction and the cell height are fixed. Atoms are relaxed at every strain.",
              "  Peng: " + PENG + ";",
              "  continuum fit to their DFT data (Table 1), converted to engineering strain;",
              "  flat sheet (atoms relaxed in-plane only).",
              "  harm: bulk, (1/2) V_atom C_ij eps_i eps_j with the experimental C11, C12 (Bosak et al.",
              "  2006); small-strain limit only."]
    print("\n" + "\n".join(lines))
    with open(os.path.join(OUT, f"strain_energy_{tag}.csv"), "w") as f:
        f.write("\n".join(",".join(r) for r in rows) + "\n")


# ----------------------------------------------------------------------------- plot
def plot(results, dft, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    INK, INK2, MUTED, GRID, SURF = "#1f1f1d", "#4a4a46", "#8a8984", "#e6e5e0", "#ffffff"
    MODE_COL = {"biaxial": "#c92a2a", "zigzag": "#2b8a3e", "armchair": "#1c7ed6"}
    MARKERS = ["o", "s", "^", "D", "v", "P"]
    plt.rcParams.update({
        "font.size": 10.5, "font.family": "sans-serif",
        "axes.edgecolor": INK2, "axes.linewidth": 0.9, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    })
    titles = {"monolayer": "Monolayer", "bilayer": "Bilayer (AA′)", "bulk": "Bulk (AA′)"}
    lo, hi = min(CONFIG["strains"]), max(CONFIG["strains"])
    fine = np.linspace(lo, hi, 201)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4), dpi=220, sharey=True, facecolor=SURF)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.76, bottom=0.11, wspace=0.06)
    for ax, s in zip(axes, SYSTEMS):
        ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=1)
        for m in CONFIG["modes"]:
            col = MODE_COL[m]
            if s == "monolayer":
                ax.plot(100 * fine, peng_energy(m, fine), color=col, lw=1.3, ls=(0, (5, 2.5)), zorder=2)
            if s == "bulk":
                ax.plot(100 * fine, bulk_harmonic(m, fine, "exp"), color=col, lw=1.0, ls=(0, (1, 1.6)),
                        zorder=2)
            for j, r in enumerate(results):
                d = r["systems"][s]["E"][m]
                ks = sorted(d, key=float)
                ax.plot([100 * float(x) for x in ks], [d[x] for x in ks], color=col,
                        lw=1.6 if j == 0 else 1.1, ls="-" if j == 0 else (0, (3, 1.5)),
                        marker=MARKERS[j % len(MARKERS)], ms=5, mec=SURF, mew=0.8, zorder=4)
            if s in dft and m in dft[s]:
                pts = sorted(dft[s][m])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="", marker="s", ms=6,
                        mfc="none", mec=col, mew=1.3, zorder=5)
        ax.set_title(titles[s], loc="left", fontsize=12.5, fontweight="bold", color=INK)
        ax.set_xlabel("strain (%)", fontsize=11)
        ax.set_xlim(100 * lo * 1.05, 100 * hi * 1.05)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.grid(color=GRID, lw=0.6, zorder=0)
    axes[0].set_ylabel("strain energy  ΔE (meV/atom)", fontsize=11)
    axes[0].set_ylim(bottom=min(-0.03 * axes[0].get_ylim()[1], axes[0].get_ylim()[0]))

    handles = [Line2D([], [], color=MODE_COL[m], lw=2.2, label={"biaxial": "biaxial",
               "zigzag": "uniaxial, zigzag", "armchair": "uniaxial, armchair"}[m]) for m in CONFIG["modes"]]
    handles += [Line2D([], [], color=INK, lw=1.6 if j == 0 else 1.1, ls="-" if j == 0 else (0, (3, 1.5)),
                       marker=MARKERS[j % len(MARKERS)], ms=5, label=f"NEP  {r['potential']}")
                for j, r in enumerate(results)]
    handles += [Line2D([], [], color=INK2, lw=1.3, ls=(0, (5, 2.5)), label="DFT, Peng et al. 2012 (monolayer)"),
                Line2D([], [], color=INK2, lw=1.0, ls=(0, (1, 1.6)), label="harmonic, exp. C$_{ij}$ (bulk)")]
    if dft:
        handles.append(Line2D([], [], ls="", marker="s", ms=6, mfc="none", mec=INK2, mew=1.3,
                              label=CONFIG["dft_label"]))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.825), ncol=min(len(handles), 7),
               frameon=False,
               fontsize=9, handletextpad=0.5, columnspacing=1.6)
    fig.text(0.065, 0.975, "hBN under in-plane strain", fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.065, 0.93, "static strain energy per atom (0 K), atoms relaxed; strain relative to the "
             "potential's equilibrium lattice; other in-plane direction and cell height fixed",
             fontsize=9.5, color=INK2, va="top")
    fig.savefig(os.path.join(OUT, f"strain_energy_{tag}.png"), facecolor=SURF)
    fig.savefig(os.path.join(OUT, f"strain_energy_{tag}.pdf"), facecolor=SURF)
    plt.close(fig)
    print(f"plot -> {OUT}/strain_energy_{tag}.png / .pdf")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("potentials", nargs="+", help="NEP potential file(s)")
    ap.add_argument("--jobs", type=int, default=4,
                    help="GPUMD runs at the same time on each GPU (default: 4)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    if not (os.path.isfile(CONFIG["gpumd"]) and os.access(CONFIG["gpumd"], os.X_OK)):
        sys.exit(f"GPUMD executable '{CONFIG['gpumd']}' not found - set CONFIG['gpumd']")
    for p in args.potentials:
        if not os.path.isfile(p):
            sys.exit(f"potential file not found: {p}")
    bad = [m for m in CONFIG["modes"] if m not in ("biaxial", "zigzag", "armchair")]
    if bad:
        sys.exit(f"unknown strain mode(s) {bad}; use biaxial, zigzag, armchair")
    os.makedirs(OUT, exist_ok=True)

    results = [compute(p, args.jobs, args.force) for p in args.potentials]
    dft = user_dft()
    tag = "_".join(os.path.splitext(os.path.basename(p))[0] for p in args.potentials)
    report(results, dft, tag)
    plot(results, dft, tag)


if __name__ == "__main__":
    main()