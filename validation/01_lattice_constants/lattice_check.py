#!/usr/bin/env python3
"""
lattice_check.py - lattice parameters of hBN from monolayer through bilayer, trilayer and
tetralayer to bulk with a NEP potential in GPUMD, compared with experiment and DFT.

Usage
    python lattice_check.py nep_best.txt                    # one potential
    python lattice_check.py nep_v1.txt nep_best.txt         # compare several
    python lattice_check.py nep_best.txt --method scan      # energy scans instead (cross-check)
    python lattice_check.py nep_best.txt --jobs 8           # GPUMD runs in parallel
    python lattice_check.py nep_best.txt --force            # ignore the cache

Systems: monolayer, the few-layer stacks in CONFIG["layers"] (bi-, tri-, tetralayer) and
bulk hBN, all AA' stacking. Results: in-plane a for all, interlayer distance d for the
stacks and bulk, c for bulk. All are the potential's static 0 K minimum.

Two methods
  --method relax (default)  one GPUMD run per structure: 'minimize fire ... 1' relaxes the
                            atoms AND the box together (like LAMMPS fix box/relax), then a,
                            c and d are read from relaxed.xyz. Needs a GPUMD version with
                            box relaxation in 'minimize' (FIRE only).
  --method scan             energy scans: monolayer a (single points), stacks a (atoms
                            relaxed at each a), bulk a and c alternately. Each scan is
                            fitted with a 4th-order polynomial; the window moves if the
                            minimum is at its edge. ~100 runs; use it to validate 'relax'.

Compared with
    experiment          room-temperature X-ray values (see REFERENCES below)
    DFT, literature     PBE+D3 (see REFERENCES below)
    DFT, yours          optional: relaxed VASP structures (CONTCAR) in CONFIG["dft_structures"],
                        or numbers in CONFIG["dft_values"]

Output (in lattice_check/)
    summary_<names>.txt / .csv     table of all values and deviations
    <potential>/<method>/...       every GPUMD run (model.xyz, run.in, thermo.out, relaxed.xyz)
    <potential>/<method>/results.json   cached result (reused while the potential and
                                   settings are unchanged)
"""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# =============================================================================
# CONFIG - edit once
# =============================================================================
CONFIG = {
    "gpumd": "/scratch/work/fakhran1/GPUMD/src/gpumd",

    # starting guesses (A); the script finds the potential's own values around them
    "a0": 2.504,                     # hBN in-plane lattice constant
    "c0": 6.66,                      # bulk AA' c (two layers per cell)
    "d0": 3.33,                      # starting interlayer distance of few-layer stacks
    "vacuum": 20.0,                  # vacuum above/below monolayer and few-layer stacks
    "layers": (2, 3, 4),             # few-layer stacks between monolayer and bulk (bi, tri, tetra)

    # your own DFT results, optional. Relaxed structures (CONTCAR) are read automatically:
    # a from the cell / B-N bonds, the interlayer distance from the layer heights.
    "dft_structures": {
        "monolayer": None,           # e.g. "/scratch/work/fakhran1/hBN/DFT/relax/mono/CONTCAR"
        "bilayer": None,
        "trilayer": None,
        "tetralayer": None,
        "bulk": None,
    },
    # ... or give numbers directly (they override the structures), e.g.
    #   "dft_values": {"bulk.a": 2.512, "bulk.c": 6.58, "monolayer.a": 2.514},
    "dft_values": {},
    "dft_label": "DFT (yours)",      # e.g. "PBE-D3(BJ), this work"

    # --method relax: atoms + box relaxed together with FIRE
    "box_force_tol": 1e-5,           # eV/A - tight, interlayer forces and stresses are small
    "box_max_steps": 100000,
    # --method scan: atoms relaxed at fixed cell
    "minimize": "sd",                # 'sd' (always in GPUMD) or 'fire' (newer GPUMD)
    "force_tol": 1e-5,               # eV/A
    "max_steps": 200000,
    # line added to run.in to save the relaxed structure (needed for the interlayer
    # distance). Newer GPUMD: dump_xyz, older: dump_exyz 1 0 0
    "dump_line": "dump_xyz 1 relaxed.xyz",

    # largest radial cutoff of your NEP (A): every cell is made thicker than 2 x cutoff
    "nep_cutoff": 6.0,
}
OUT = "lattice_check"
# =============================================================================

# ----------------------------------------------------------------------------- references
# (value, source) or (value, source, uncertainty). Experimental values are at room temperature; NEP gives the static 0 K
# minimum without zero-point motion, so expect NEP/DFT a and c to sit slightly below them.
REFERENCES = {
    "exp": {
        "bulk.a":    (2.5040,  "Lynch & Drickamer, J. Chem. Phys. 44, 181 (1966), 297 K"),
        "bulk.c":    (6.6612,  "Lynch & Drickamer, J. Chem. Phys. 44, 181 (1966), 297 K"),
        "bulk.d":    (3.3306,  "derived: c/2 from the experimental c above"),
        "bilayer.d": (3.2500, "Warner et al., ACS Nano 4, 1299 (2010) (TEM)", 0.10),
    },
    "dft_lit": {
        "monolayer.a": (2.510, "Thiemann et al., J. Phys. Chem. C 124, 22278 (2020)"),
        "bilayer.a":   (2.510, "Thiemann et al., J. Phys. Chem. C 124, 22278 (2020)"),
        "bilayer.d":   (3.360, "Thiemann et al., J. Phys. Chem. C 124, 22278 (2020)"),
        "bulk.a":      (2.510, "Thiemann et al., J. Phys. Chem. C 124, 22278 (2020)"),
        "bulk.c":      (6.610, "Thiemann et al., J. Phys. Chem. C 124, 22278 (2020)"),
        "bulk.d":      (3.305, "derived: c/2 from the c above"),
    },
}
QUANT = {   # key: (label, unit)
    "monolayer.a": ("monolayer hBN  a", "A"),
    "bilayer.a": ("bilayer hBN    a", "A"), "bilayer.d": ("bilayer hBN    d", "A"),
    "trilayer.a": ("trilayer hBN   a", "A"), "trilayer.d": ("trilayer hBN   d", "A"),
    "tetralayer.a": ("tetralayer hBN a", "A"), "tetralayer.d": ("tetralayer hBN d", "A"),
    "bulk.a": ("bulk hBN       a", "A"), "bulk.c": ("bulk hBN       c", "A"),
    "bulk.d": ("bulk hBN       d", "A"),
}
LAYER_NAME = {1: "monolayer", 2: "bilayer", 3: "trilayer", 4: "tetralayer", 5: "pentalayer"}
SQ3 = np.sqrt(3.0)


def stack_name(n):
    return LAYER_NAME.get(n, f"{n}-layer")


def qlabel(key):
    if key in QUANT:
        return QUANT[key]
    sysname, q = key.split(".")
    return (f"{sysname:<14} {q}", "A")


# ----------------------------------------------------------------------------- structures
def write_xyz(path, sp, pos, lat):
    with open(path, "w") as f:
        f.write(f"{len(sp)}\n")
        f.write('Lattice="' + " ".join(f"{x:.8f}" for x in np.asarray(lat).flatten()) +
                '" pbc="T T T" Properties=species:S:1:pos:R:3\n')
        for s, p in zip(sp, pos):
            f.write(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")


def n_rep(length):
    return int(np.ceil((2 * CONFIG["nep_cutoff"] + 0.5) / length))


def hbn_reps(a=None, c=None):
    """number of rectangular cells along x, y (and bulk periods along z) used by hbn_layers.
    Fixed for the whole run: sized for the smallest a (0.90 a0) and c (0.80 c0) a scan may
    reach, so the supercell never changes size in the middle of a scan."""
    a_min, c_min = 0.90 * CONFIG["a0"], 0.80 * CONFIG["c0"]
    return n_rep(a_min), n_rep(a_min * SQ3), (max(1, n_rep(c_min)) if c is not None else 1)


def hbn_layers(a, n_layers, d=None, c=None):
    """Rectangular hBN cell, AA' stacking (B over N). c given: bulk with period c (two layers
    per period); otherwise a stack of n_layers separated by d with vacuum on both sides."""
    base = [("B", 0.0, 0.0), ("N", 0.0, 1 / 3), ("B", 0.5, 0.5), ("N", 0.5, 5 / 6)]
    nx, ny, nz = hbn_reps(a, c)
    Lx, Ly = nx * a, ny * a * SQ3
    if c is not None:
        n_layers, dz, Lz, z0 = 2 * nz, c / 2, nz * c, c / 4
    else:
        dz = d if d is not None else CONFIG["d0"]
        Lz = (n_layers - 1) * dz + 2 * CONFIG["vacuum"]
        Lz = max(Lz, 2 * CONFIG["nep_cutoff"] + 0.5)
        z0 = (Lz - (n_layers - 1) * dz) / 2
    sp, pos = [], []
    for l in range(n_layers):
        swap = l % 2 == 1
        for i in range(nx):
            for j in range(ny):
                for el, fx, fy in base:
                    if swap:
                        el = "N" if el == "B" else "B"
                    sp.append(el)
                    pos.append(((i + fx) * a, (j + fy) * a * SQ3, z0 + l * dz))
    return sp, np.array(pos), np.diag([Lx, Ly, Lz])


def read_structure(path):
    """POSCAR/CONTCAR or extxyz -> (species, cartesian positions, 3x3 lattice)."""
    lines = open(path).read().splitlines()
    if "lattice=" in lines[1].lower():
        n = int(lines[0])
        low = lines[1]
        i = low.lower().index("lattice=")
        lat = np.array(low[i + 8:].split('"')[1].split(), float).reshape(3, 3)
        sp = [l.split()[0] for l in lines[2:2 + n]]
        pos = np.array([l.split()[1:4] for l in lines[2:2 + n]], float)
        return sp, pos, lat
    scale = float(lines[1].split()[0])
    lat = np.array([l.split()[:3] for l in lines[2:5]], float) * scale
    names = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    i = 7
    if lines[i].strip().lower().startswith("s"):
        i += 1
    cart = lines[i].strip().lower()[0] in "ck"
    n = sum(counts)
    xyz = np.array([l.split()[:3] for l in lines[i + 1:i + 1 + n]], float)
    pos = xyz * scale if cart else xyz @ lat
    sp = sum([[e] * c for e, c in zip(names, counts)], [])
    return sp, pos, lat


def read_last_xyz(path):
    """last frame of an extxyz file written by GPUMD -> (species, positions, lattice)"""
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
    i = low.index("lattice=")
    lat = np.array(head[i + 8:].split('"')[1].split(), float).reshape(3, 3)
    props = "species:S:1:pos:R:3"
    if "properties=" in low:
        j = low.index("properties=")
        props = head[j + 11:].split()[0]
    f = props.split(":")
    col, i_sp, i_pos = 0, 0, 1
    for name, typ, cnt in zip(f[0::3], f[1::3], f[2::3]):
        if name.lower() == "species":
            i_sp = col
        if name.lower() == "pos":
            i_pos = col
        col += int(cnt)
    rows = [lines[k + 2 + m].split() for m in range(n)]
    sp = [r[i_sp] for r in rows]
    pos = np.array([r[i_pos:i_pos + 3] for r in rows], float)
    return sp, pos, lat


# ----------------------------------------------------------------------------- geometry analysis
def interlayer(pos, lat, periodic):
    """mean distance between atomic layers along z (layers = z clusters, gap > 1 A)"""
    z = np.sort(pos[:, 2])
    cuts = np.where(np.diff(z) > 1.0)[0]
    groups = np.split(z, cuts + 1)
    zs = np.array([g.mean() for g in groups])
    if len(zs) < 2 and not periodic:
        return None
    gaps = list(np.diff(zs))
    if periodic:
        gaps.append(zs[0] + lat[2, 2] - zs[-1])
    return float(np.mean(gaps))


def inplane_a(sp, pos, lat):
    """hBN a: from a hexagonal primitive cell directly, otherwise sqrt(3) x median B-N bond"""
    a1, a2 = np.linalg.norm(lat[0]), np.linalg.norm(lat[1])
    cosg = lat[0] @ lat[1] / (a1 * a2)
    if abs(a1 - a2) < 1e-3 and abs(abs(cosg) - 0.5) < 1e-3 and a1 < 2.7:
        return float(a1)
    sp = np.array(sp)
    B, N = pos[sp == "B"], pos[sp == "N"]
    inv = np.linalg.inv(lat)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]) @ lat
    dmin = []
    for b in B:
        d = N - b
        d = d - np.round(d @ inv) @ lat
        dd = np.linalg.norm(d[:, None, :] + shifts[None], axis=2).min()
        dmin.append(dd)
    return float(np.median(dmin) * SQ3)


# ----------------------------------------------------------------------------- GPUMD
SINGLE = """potential {pot}
velocity 1e-6
ensemble nve
time_step 1e-6
dump_thermo 1
run 1
"""
RELAXBOX = """potential {pot}
minimize fire {ftol} {nmax} 1
velocity 1e-6
ensemble nve
time_step 1e-6
dump_thermo 1
{dump}run 1
"""
RELAX = """potential {pot}
minimize {method} {ftol} {nmax}
velocity 1e-6
ensemble nve
time_step 1e-6
dump_thermo 1
{dump}run 1
"""


def run_gpumd(workdir):
    proc = subprocess.run([CONFIG["gpumd"]], cwd=workdir, capture_output=True, text=True)
    open(os.path.join(workdir, "gpumd.log"), "w").write(proc.stdout + proc.stderr)
    thermo = os.path.join(workdir, "thermo.out")
    if proc.returncode != 0 or not os.path.isfile(thermo):
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        raise RuntimeError(f"GPUMD failed in {workdir}:\n  " + "\n  ".join(tail))
    rows = [l for l in open(thermo).read().splitlines() if l.strip()]
    return float(rows[-1].split()[2])


def prepare(workdir, pot, sp, pos, lat, relax):
    """relax: False = single point, True = atoms only (fixed box), "box" = atoms + box"""
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    write_xyz(os.path.join(workdir, "model.xyz"), sp, pos, lat)
    dump = CONFIG["dump_line"].strip() + "\n" if CONFIG["dump_line"].strip() else ""
    if relax == "box":
        txt = RELAXBOX.format(pot=pot, ftol=CONFIG["box_force_tol"],
                              nmax=CONFIG["box_max_steps"], dump=dump)
    elif relax:
        txt = RELAX.format(pot=pot, method=CONFIG["minimize"], ftol=CONFIG["force_tol"],
                           nmax=CONFIG["max_steps"], dump=dump)
    else:
        txt = SINGLE.format(pot=pot)
    open(os.path.join(workdir, "run.in"), "w").write(txt)
    return workdir


def run_many(dirs, jobs):
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(run_gpumd, dirs))


def relaxed(workdir):
    f = os.path.join(workdir, "relaxed.xyz")
    if not os.path.isfile(f):
        sys.exit(f"{f} not found - check CONFIG['dump_line'] for your GPUMD version "
                 f"(newer: 'dump_xyz 1 relaxed.xyz', older: 'dump_exyz 1 0 0' + rename)")
    return read_last_xyz(f)


def fit_min(x, E):
    p = np.polyfit(x, E, 4)
    xx = np.linspace(x.min(), x.max(), 20001)
    i = int(np.argmin(np.polyval(p, xx)))
    edge = i < 400 or i > len(xx) - 400
    return float(xx[i]), float(np.polyval(p, xx[i])), edge


def scan(base, label, window, builder, pot, jobs, log, bounds, relax=False, npts=11):
    """1-D energy scan of one parameter; the window moves if the minimum is at its edge.
    builder(x) -> (species, positions, lattice). Returns (x_min, E_min) or (None, None)."""
    lo, hi = window
    txt = os.path.join(base, f"{label}_scan.txt")
    for attempt in range(5):
        xs = np.linspace(lo, hi, npts)
        dirs = [prepare(os.path.join(base, f"{label}_{attempt}_{i:02d}"), pot, *builder(x), relax=relax)
                for i, x in enumerate(xs)]
        E = np.array(run_many(dirs, jobs))
        nat = np.array([len(builder(x)[0]) for x in xs])      # per point, to be safe
        epa = E / nat
        with open(txt, "a") as f:
            for x, e in zip(xs, epa):
                f.write(f"{x:.6f} {e:.8f}\n")
        x0, e0, edge = fit_min(xs, epa)
        if not edge:
            return x0, e0
        w = hi - lo
        lo, hi = max(bounds[0], x0 - w / 2), min(bounds[1], x0 + w / 2)
        if hi - lo < 0.3 * w:
            break
    log(f"  WARNING: no energy minimum for {label} within {bounds[0]:.4f}-{bounds[1]:.4f} - "
        f"see {txt}. The potential may not bind here (for layers: no interlayer attraction).")
    return None, None


# ----------------------------------------------------------------------------- per potential
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def compute(pot, jobs, force, method):
    tag = os.path.splitext(os.path.basename(pot))[0]
    root = os.path.join(OUT, tag, method)
    res_file = os.path.join(root, "results.json")
    keys = ["a0", "c0", "d0", "vacuum", "layers", "nep_cutoff"]
    keys += (["box_force_tol", "box_max_steps"] if method == "relax"
             else ["minimize", "force_tol", "max_steps"])
    key = {"v": 4, "method": method, "sha": sha(pot), "config": {k: CONFIG[k] for k in keys}}
    key = json.loads(json.dumps(key))
    if not force and os.path.isfile(res_file):
        old = json.load(open(res_file))
        if old.get("key") == key:
            print(f"{tag} ({method}): cached ({res_file})")
            return old
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    potabs = os.path.abspath(pot)
    logf = open(os.path.join(root, "log.txt"), "w")

    def log(msg):
        print(msg); logf.write(msg + "\n"); logf.flush()

    log(f"=== {tag}  (method: {method}) ===")
    vals, energy = (relax_all if method == "relax" else scan_all)(root, potabs, jobs, log)
    out = {"key": key, "potential": os.path.basename(pot), "method": method,
           "values": vals, "energy_per_atom": energy}
    json.dump(out, open(res_file, "w"), indent=1)
    logf.close()
    return out


def relax_all(root, potabs, jobs, log):
    """one GPUMD run per structure: atoms and box relaxed together (minimize fire ... 1)"""
    a0, c0 = CONFIG["a0"], CONFIG["c0"]
    items = [("monolayer", 1, None)] + [(stack_name(n), n, None) for n in CONFIG["layers"]] + \
            [("bulk", 2, c0)]
    dirs = [prepare(os.path.join(root, name), potabs, *hbn_layers(a0, n, c=c), relax="box")
            for name, n, c in items]
    log(f"relaxing {len(dirs)} structures (atoms + box) ...")
    E = run_many(dirs, jobs)
    vals, energy = {}, {}
    for (name, n, c), d, e in zip(items, dirs, E):
        sp, pos, lat = relaxed(d)
        nx, ny, nz = hbn_reps(a0, c)
        ax = np.linalg.norm(lat[0]) / nx
        ay = np.linalg.norm(lat[1]) / (ny * SQ3)
        a = 0.5 * (ax + ay)
        shear = np.abs(lat - np.diag(np.diag(lat))).max()
        if abs(ax - ay) > 1e-3 or shear > 1e-3:
            log(f"  NOTE {name}: box not hexagonal after relaxation (a_x = {ax:.4f}, "
                f"a_y = {ay:.4f} A, largest off-diagonal {shear:.4f} A) - reporting the mean a")
        vals[f"{name}.a"] = float(a)
        energy[name] = e / len(sp)
        if c is not None:
            vals["bulk.c"] = float(np.linalg.norm(lat[2]) / nz)
            vals["bulk.d"] = interlayer(pos, lat, periodic=True)
            label = "bulk (AA')"
            log(f"{label:<14} a = {a:.4f} A, c = {vals['bulk.c']:.4f} A, d = {vals['bulk.d']:.4f} A")
        elif n > 1:
            vals[f"{name}.d"] = interlayer(pos, lat, periodic=False)
            log(f"{name:<14} a = {a:.4f} A, d = {vals[f'{name}.d']:.4f} A")
        else:
            log(f"{name:<14} a = {a:.4f} A")
    return vals, energy


def scan_all(root, potabs, jobs, log):
    """energy scans (about 100 GPUMD runs); see the module docstring"""
    a0, c0 = CONFIG["a0"], CONFIG["c0"]
    abounds = (0.90 * a0, 1.10 * a0)
    vals, energy = {}, {}

    # --- monolayer: a (flat sheet, no relaxation needed)
    d = os.path.join(root, "monolayer"); os.makedirs(d)
    a, e = scan(d, "a", (0.98 * a0, 1.02 * a0), lambda x: hbn_layers(x, 1), potabs, jobs, log, abounds)
    vals["monolayer.a"], energy["monolayer"] = a, e
    log(f"monolayer      a = {a:.4f} A" if a else "monolayer      a = not found")

    # --- few-layer stacks: a with relaxed atoms, then the spacing at that a
    for n in CONFIG["layers"]:
        name = stack_name(n)
        d = os.path.join(root, name); os.makedirs(d)
        a, e = scan(d, "a", (0.98 * a0, 1.02 * a0), lambda x, n=n: hbn_layers(x, n), potabs, jobs,
                    log, abounds, relax=True)
        vals[f"{name}.a"], energy[name] = a, e
        if a is None:
            continue
        fin = prepare(os.path.join(d, "final"), potabs, *hbn_layers(a, n), relax=True)
        run_gpumd(fin)
        sp, pos, lat = relaxed(fin)
        vals[f"{name}.d"] = interlayer(pos, lat, periodic=False)
        log(f"{name:<14} a = {a:.4f} A, d = {vals[f'{name}.d']:.4f} A")

    # --- bulk AA': alternate a and c (atoms stay on their symmetric sites)
    d = os.path.join(root, "bulk"); os.makedirs(d)
    a_b, c_b = a0, c0
    for it in range(3):
        x, e = scan(d, f"a{it}", (0.98 * a_b, 1.02 * a_b), lambda x: hbn_layers(x, 2, c=c_b),
                    potabs, jobs, log, abounds)
        a_b = x if x is not None else a_b
        x, e = scan(d, f"c{it}", (0.94 * c_b, 1.06 * c_b), lambda x: hbn_layers(a_b, 2, c=x),
                    potabs, jobs, log, (0.80 * c0, 1.40 * c0))
        if x is None:
            c_b = None
            break
        converged = abs(x - c_b) < 2e-3
        c_b = x
        if converged and it > 0:
            break
    vals["bulk.a"], vals["bulk.c"] = a_b, c_b
    vals["bulk.d"] = c_b / 2 if c_b else None
    energy["bulk"] = e
    log(f"bulk (AA')     a = {a_b:.4f} A, c = {c_b:.4f} A, d = {c_b / 2:.4f} A" if c_b
        else f"bulk (AA')     a = {a_b:.4f} A, c = not found")

    return vals, energy


# ----------------------------------------------------------------------------- user DFT
def user_dft():
    vals = {}
    for name, path in CONFIG["dft_structures"].items():
        if not path:
            continue
        if not os.path.isfile(path):
            print(f"DFT structure for {name} not found: {path}")
            continue
        sp, pos, lat = read_structure(path)
        vals[f"{name}.a"] = inplane_a(sp, pos, lat)
        if name == "bulk":
            dd = interlayer(pos, lat, periodic=True)
            vals["bulk.d"], vals["bulk.c"] = dd, 2 * dd
        elif name != "monolayer":
            vals[f"{name}.d"] = interlayer(pos, lat, periodic=False)
    vals.update(CONFIG["dft_values"])
    return vals


# ----------------------------------------------------------------------------- report
def keys_in_order(results, dft):
    keys = ["monolayer.a"]
    for n in CONFIG["layers"]:
        keys += [f"{stack_name(n)}.a", f"{stack_name(n)}.d"]
    keys += ["bulk.a", "bulk.c", "bulk.d"]
    have = set(dft) | set(REFERENCES["exp"]) | set(REFERENCES["dft_lit"])
    for r in results:
        have |= {k for k, v in r["values"].items() if v is not None}
    return [k for k in keys if k in have]


def pct(x, ref):
    return 100 * (x - ref) / ref if (x is not None and ref) else None


def report(results, dft, tag):
    exp, lit = REFERENCES["exp"], REFERENCES["dft_lit"]
    names = [r["potential"] for r in results]
    keys = keys_in_order(results, dft)
    fmt = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"
    w = max(10, max(len(n) for n in names))
    how = {"relax": "atoms + box relaxed (minimize fire)", "scan": "energy scans"}[results[0]["method"]]
    lines = [f"Lattice parameters: NEP (GPUMD, 0 K static; {how}) vs DFT and experiment", ""]
    head = f"{'quantity':<22}{'unit':<5}" + "".join(f"{n:>{w + 2}}" for n in names) + \
           f"{CONFIG['dft_label'][:12]:>14}{'DFT (lit.)':>12}{'exp.':>14}"
    for n in names:
        head += f"{'vs exp':>9}"
    if dft:
        for n in names:
            head += f"{'vs DFT':>9}"
    lines += [head, "-" * len(head)]
    rows = []
    for k in keys:
        lab, unit = qlabel(k)
        nd = 4
        nep = [r["values"].get(k) for r in results]
        e, l, u = exp.get(k, (None,))[0], lit.get(k, (None,))[0], dft.get(k)
        err = exp[k][2] if k in exp and len(exp[k]) > 2 else None
        e_txt = fmt(e, nd) if err is None else f"{e:.2f}±{err:.2f}"
        s = f"{lab:<22}{unit:<5}" + "".join(f"{fmt(v, nd):>{w + 2}}" for v in nep)
        s += f"{fmt(u, nd):>14}{fmt(l, nd):>12}{e_txt:>14}"
        for v in nep:
            p = pct(v, e)
            s += f"{'-' if p is None else f'{p:+.2f}%':>9}"
        if dft:
            for v in nep:
                p = pct(v, u)
                s += f"{'-' if p is None else f'{p:+.2f}%':>9}"
        lines.append(s)
        rows.append([k, unit] + nep + [u, l, e, err])
    lines += ["", "d = interlayer distance (AA' stacking).",
              "Experimental values are at room temperature; NEP and DFT are static 0 K values",
              "without zero-point motion, so slightly smaller a and c are expected (~0.1-0.5%).", "",
              "References"]
    seen = set()
    for src in ("exp", "dft_lit"):
        for k, (v, ref, *_) in REFERENCES[src].items():
            if ref not in seen and k in keys:
                seen.add(ref)
                lines.append(f"  [{'exp' if src == 'exp' else 'DFT'}] {ref}")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(OUT, f"summary_{tag}.txt"), "w").write(txt + "\n")
    with open(os.path.join(OUT, f"summary_{tag}.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["quantity", "unit"] + names + [CONFIG["dft_label"], "DFT_literature", "experiment",
                                                    "experiment_uncertainty"])
        wr.writerows(rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("potentials", nargs="+", help="NEP potential file(s)")
    ap.add_argument("--jobs", type=int, default=4, help="GPUMD runs in parallel (default 4)")
    ap.add_argument("--method", choices=("relax", "scan"), default="relax",
                    help="relax: atoms + box relaxed in one GPUMD run per structure (default); "
                         "scan: energy scans, about 100 runs (cross-check)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    if not (os.path.isfile(CONFIG["gpumd"]) and os.access(CONFIG["gpumd"], os.X_OK)):
        sys.exit(f"GPUMD executable '{CONFIG['gpumd']}' not found - set CONFIG['gpumd']")
    for p in args.potentials:
        if not os.path.isfile(p):
            sys.exit(f"potential file not found: {p}")
    os.makedirs(OUT, exist_ok=True)

    results = [compute(p, args.jobs, args.force, args.method) for p in args.potentials]
    dft = user_dft()
    tag = "_".join(os.path.splitext(os.path.basename(p))[0] for p in args.potentials)
    tag += f"_{args.method}"
    report(results, dft, tag)


if __name__ == "__main__":
    main()