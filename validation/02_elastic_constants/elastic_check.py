#!/usr/bin/env python3
"""
elastic_check.py - elastic constants of bulk and monolayer hBN with a NEP potential in GPUMD,
compared with experiment and DFT.

Usage
    python elastic_check.py nep_best.txt                    # one potential
    python elastic_check.py nep_v1.txt nep_best.txt         # compare several
    python elastic_check.py nep_best.txt --jobs 16          # GPUMD runs at the same time
    python elastic_check.py nep_best.txt --force            # ignore the cache

Method (static, 0 K, relaxed-ion)
  1. The reference structure is relaxed: atoms and box together ('minimize fire ... 1').
  2. Each strain component j is applied with +-0.5 % and +-1 % (CONFIG "strains"); shear
     strains are engineering strains (gamma = 2 eps_ij). For every strained cell the atoms
     are relaxed at fixed cell, and GPUMD's stress tensor is read from thermo.out.
  3. C_ij = d sigma_i / d eps_j from a fit through the 5 points (quadratic fit, linear term,
     so small non-linearity does not bias the result).
  4. Check: the diagonal C_jj are recomputed from the energy curvature of the same runs,
     d2E/d eps_j2 / V. Stress and energy routes must agree; if not, the log says so.

  Speed: the relaxed reference doubles as the zero-strain point of every fit (bulk: 6 x 4,
  monolayer: 3 x 4 strained runs + 2 references = 38 runs). Bulk and monolayer run together,
  and the runs are spread over all GPUs Slurm gives the job (CUDA_VISIBLE_DEVICES), 4 per
  GPU at a time by default (--jobs).

  bulk hBN (AA')   strains 1..6 -> C11, C12, C13, C33, C44, C66 (GPa), Reuss bulk modulus,
                   and the hexagonal check C66 = (C11 - C12)/2
  monolayer hBN    in-plane strains 1, 2, 6 -> C11, C12, C66 in N/m (2D values: the 3D
                   stress of the cell times the cell height), 2D Young's modulus
                   Y = (C11^2 - C12^2)/C11 and Poisson ratio nu = C12/C11

Compared with
    experiment          Bosak et al. 2006 (bulk), Falin et al. 2017 (monolayer)
    DFT, literature     bulk (Thiemann et al. 2020), monolayer (Peng et al.)
    DFT, yours          optional: numbers in CONFIG["dft_values"], or VASP OUTCARs of an
                        IBRION = 6 / ISIF = 3 run in CONFIG["dft_outcar"] (the "TOTAL ELASTIC
                        MODULI" block, which includes ionic relaxation, is read)

Output (in elastic_check/)
    summary_<names>.txt / .csv        all constants and deviations (plus full 6x6 matrices)
    <potential>/bulk|monolayer/...    every GPUMD run (model.xyz, run.in, thermo.out, ...)
    <potential>/results.json          cached result
"""
import argparse, csv, hashlib, json, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import numpy as np

# =============================================================================
# CONFIG - edit once
# =============================================================================
CONFIG = {
    "gpumd": "/scratch/work/fakhran1/GPUMD/src/gpumd",

    # starting structure (A); it is relaxed (atoms + box) before straining
    "a0": 2.504,
    "c0": 6.66,
    "vacuum": 20.0,                  # vacuum above and below the monolayer

    "strains": (0.005, 0.01),        # applied as -0.01, -0.005, 0, +0.005, +0.01

    # relaxations
    "box_force_tol": 1e-5,           # reference: atoms + box (FIRE)
    "box_max_steps": 100000,
    "minimize": "fire",              # strained cells: atoms only ('fire' or 'sd')
    "force_tol": 1e-5,               # eV/A - tight, the stress depends on it
    "max_steps": 100000,
    "dump_line": "dump_xyz 1 relaxed.xyz",   # needed to read the relaxed reference

    # largest radial cutoff of your NEP (A): every cell is made thicker than 2 x cutoff
    "nep_cutoff": 6.0,

    # your own DFT results, optional
    #   numbers:  {"bulk.C11": 850, "bulk.C33": 30, "monolayer.C11": 290, ...}
    #   (bulk in GPa, monolayer in N/m; keys as in the summary table)
    "dft_values": {},
    #   or VASP OUTCARs (IBRION = 6, ISIF = 3). For the monolayer the 3D moduli are
    #   converted to N/m with the cell height read from the same OUTCAR.
    "dft_outcar": {
        "bulk": None,                # e.g. "/scratch/work/fakhran1/hBN/DFT/elastic/bulk/OUTCAR"
        "monolayer": None,
    },
    "dft_label": "DFT (this work)",
}
OUT = "elastic_check"
# =============================================================================

# ----------------------------------------------------------------------------- references
# (value, source[, uncertainty]); bulk in GPa, monolayer in N/m
BOSAK = "Bosak et al., Phys. Rev. B 73, 041402(R) (2006)"
THIEMANN = "Thiemann et al., J. Phys. Chem. C 124, 22278 (2020)"
PENG = "Peng et al., Comp. Mater. Sci. 56 (2012)"
FALIN = "Falin et al., Nat. Commun. 8, 15815 (2017)"
REFERENCES = {
    "exp": {
        "bulk.C11": (811.0, BOSAK, 12.0),
        "bulk.C12": (169.0, BOSAK, 24.0),
        "bulk.C13": (0.0, BOSAK, 3.0),
        "bulk.C33": (27.0, BOSAK, 0.5),
        "bulk.C44": (7.7, BOSAK, 0.5),
        "bulk.C66": (321.0, "derived: (C11 - C12)/2 from the Bosak et al. values"),
        "bulk.B_R": (25.6, BOSAK, 8.0),
        "monolayer.Y": (289.0, FALIN, 24.0),
    },
    "dft_lit": {
        "bulk.C11": (881.8, THIEMANN),
        "bulk.C12": (195.6, THIEMANN),
        "bulk.C13": (0.0, THIEMANN),
        "bulk.C33": (28.9, THIEMANN),
        "bulk.C44": (3.3, THIEMANN),
        "bulk.C66": (343.1, "derived: (C11 - C12)/2 from the Thiemann et al. values"),
        "bulk.B_R": (27.4, THIEMANN),
        "monolayer.C11": (293.2, PENG),
        "monolayer.C12": (66.1, PENG),
        "monolayer.Y": (278.3, PENG),
        "monolayer.nu": (0.2254, PENG),
    },
}
ROWS = [  # key, label, unit
    ("bulk.C11", "bulk      C11", "GPa"), ("bulk.C12", "bulk      C12", "GPa"),
    ("bulk.C13", "bulk      C13", "GPa"), ("bulk.C33", "bulk      C33", "GPa"),
    ("bulk.C44", "bulk      C44", "GPa"), ("bulk.C66", "bulk      C66", "GPa"),
    ("bulk.C66_check", "bulk      (C11-C12)/2", "GPa"),
    ("bulk.B_R", "bulk      B (Reuss)", "GPa"),
    ("monolayer.C11", "monolayer C11", "N/m"), ("monolayer.C12", "monolayer C12", "N/m"),
    ("monolayer.C66", "monolayer C66", "N/m"),
    ("monolayer.C66_check", "monolayer (C11-C12)/2", "N/m"),
    ("monolayer.Y", "monolayer Y (2D)", "N/m"), ("monolayer.nu", "monolayer Poisson nu", "-"),
]
VOIGT = ["xx", "yy", "zz", "yz", "xz", "xy"]
EV_A3_TO_GPA = 160.21766
SQ3 = np.sqrt(3.0)


# ----------------------------------------------------------------------------- structures
def write_xyz(path, sp, pos, lat):
    with open(path, "w") as f:
        f.write(f"{len(sp)}\n")
        f.write('Lattice="' + " ".join(f"{x:.10f}" for x in np.asarray(lat).flatten()) +
                '" pbc="T T T" Properties=species:S:1:pos:R:3\n')
        for s, p in zip(sp, pos):
            f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def n_rep(length):
    return int(np.ceil((2 * CONFIG["nep_cutoff"] + 0.5) / length))


def hbn(bulk):
    """rectangular hBN supercell, AA' stacking (B over N), thicker than 2 x cutoff with room
    for -10 % strain; bulk: two layers per period c, monolayer: sheet in vacuum"""
    a, c = CONFIG["a0"], CONFIG["c0"]
    nx, ny = n_rep(0.9 * a), n_rep(0.9 * a * SQ3)
    base = [("B", 0.0, 0.0), ("N", 0.0, 1 / 3), ("B", 0.5, 0.5), ("N", 0.5, 5 / 6)]
    if bulk:
        nz = max(1, n_rep(0.9 * c))
        n_layers, dz, Lz, z0 = 2 * nz, c / 2, nz * c, c / 4
    else:
        n_layers, dz = 1, 0.0
        Lz = max(2 * CONFIG["vacuum"], 2 * CONFIG["nep_cutoff"] + 0.5)
        z0 = Lz / 2
    sp, pos = [], []
    for l in range(n_layers):
        for i in range(nx):
            for j in range(ny):
                for el, fx, fy in base:
                    if l % 2 == 1:
                        el = "N" if el == "B" else "B"
                    sp.append(el)
                    pos.append(((i + fx) * a, (j + fy) * a * SQ3, z0 + l * dz))
    return sp, np.array(pos), np.diag([nx * a, ny * a * SQ3, Lz])


def read_last_xyz(path):
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
    return [r[i_sp] for r in rows], np.array([r[i_pos:i_pos + 3] for r in rows], float), lat


def strain_matrix(j, delta):
    """deformation I + eps for Voigt component j (0..5); shear = engineering strain"""
    e = np.zeros((3, 3))
    if j < 3:
        e[j, j] = delta
    else:
        p, q = {3: (1, 2), 4: (0, 2), 5: (0, 1)}[j]
        e[p, q] = e[q, p] = delta / 2
    return np.eye(3) + e


# ----------------------------------------------------------------------------- GPUMD
TAIL = """velocity 1e-6
ensemble nve
time_step 1e-6
dump_thermo 1
{dump}run 1
"""


def gpus():
    """GPUs given to this job by Slurm (CUDA_VISIBLE_DEVICES), or [None] = leave it to GPUMD"""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return [g for g in v.split(",") if g.strip()] or [None]


def run_gpumd(workdir, gpu=None):
    """returns the last thermo.out row: T K U Pxx Pyy Pzz Pyz Pxz Pxy ..."""
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu                    # pin this run to one GPU
    proc = subprocess.run([CONFIG["gpumd"]], cwd=workdir, capture_output=True, text=True, env=env)
    open(os.path.join(workdir, "gpumd.log"), "w").write(proc.stdout + proc.stderr)
    thermo = os.path.join(workdir, "thermo.out")
    if proc.returncode != 0 or not os.path.isfile(thermo):
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        raise RuntimeError(f"GPUMD failed in {workdir}:\n  " + "\n  ".join(tail))
    rows = [l for l in open(thermo).read().splitlines() if l.strip()]
    row = np.array(rows[-1].split(), float)
    if len(row) < 9:
        raise RuntimeError(f"{thermo}: expected the pressure tensor in columns 4-9")
    return row


def prepare(workdir, pot, sp, pos, lat, box):
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    write_xyz(os.path.join(workdir, "model.xyz"), sp, pos, lat)
    dump = CONFIG["dump_line"].strip() + "\n" if CONFIG["dump_line"].strip() else ""
    if box:
        head = f"potential {pot}\nminimize fire {CONFIG['box_force_tol']} {CONFIG['box_max_steps']} 1\n"
    else:
        head = (f"potential {pot}\nminimize {CONFIG['minimize']} {CONFIG['force_tol']} "
                f"{CONFIG['max_steps']}\n")
    open(os.path.join(workdir, "run.in"), "w").write(head + TAIL.format(dump=dump))
    return workdir


def run_many(dirs, jobs):
    """run all folders, at most 'jobs' at a time, spread round-robin over the job's GPUs"""
    pool = Queue()
    g = gpus()
    for k in range(jobs):
        pool.put(g[k % len(g)])

    def one(d):
        gpu = pool.get()
        try:
            return run_gpumd(d, gpu)
        finally:
            pool.put(gpu)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(one, dirs))


# ----------------------------------------------------------------------------- strains and fits
def components(bulk):
    """strain directions: all six for bulk, the three in-plane ones (xx, yy, xy) for the
    monolayer. The symmetry-equivalent pairs (xx/yy, yz/xz) are averaged, and their spread is
    a check of the calculation."""
    return [0, 1, 2, 3, 4, 5] if bulk else [0, 1, 5]


def fit_system(name, comps, deltas, rows, row0, V, log):
    """stresses of the strained runs -> 6x6 stiffness of the cell (GPa), NaN where not computed.
    row0 is the thermo row of the relaxed reference, i.e. the zero-strain point of every fit."""
    C = np.full((6, 6), np.nan)
    C_energy = {}
    sigma0 = -row0[3:9]                                    # stress = - pressure (GPa)
    for j in comps:
        x = np.array([0.0] + [dl for dl in deltas])
        r = [row0] + [rows[(j, dl)] for dl in deltas]
        sig = -np.array([q[3:9] for q in r])
        E = np.array([q[2] for q in r])
        order = np.argsort(x)
        x, sig, E = x[order], sig[order], E[order]
        for i in range(6):
            C[i, j] = np.polyfit(x, sig[:, i], 2)[1]
        C_energy[j] = 2 * np.polyfit(x, E, 2)[0] / V * EV_A3_TO_GPA
    # sign convention guard: diagonal stiffness must be positive
    if np.mean([C[j, j] for j in comps if j < 3]) < 0:
        log(f"  NOTE ({name}): GPUMD pressure sign is opposite to the assumed convention - flipping")
        C, sigma0 = -C, -sigma0
    log(f"{name}: residual stress of the reference (GPa): " + " ".join(f"{v:+.3f}" for v in sigma0))
    for j in comps:
        cs, ce = C[j, j], C_energy[j]
        flag = "" if abs(cs - ce) <= max(0.02 * abs(cs), 0.5) else "   <-- stress and energy disagree"
        log(f"  C{j + 1}{j + 1}: from stress {cs:9.2f} GPa, from energy curvature {ce:9.2f} GPa{flag}")
    return C


def symmetrize(C):
    """(C + C^T)/2 where both entries exist, otherwise whichever exists"""
    A, B = C, C.T
    out = np.where(np.isnan(A), B, np.where(np.isnan(B), A, 0.5 * (A + B)))
    return out


def nanmean(*v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def to_list(M):
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in M]


# ----------------------------------------------------------------------------- per potential
def compute(pot, jobs, force):
    tag = os.path.splitext(os.path.basename(pot))[0]
    root = os.path.join(OUT, tag)
    res_file = os.path.join(root, "results.json")
    keys = ["a0", "c0", "vacuum", "strains", "box_force_tol", "box_max_steps", "minimize",
            "force_tol", "max_steps", "nep_cutoff"]
    key = json.loads(json.dumps({"v": 3,
                                 "sha": hashlib.sha256(open(pot, "rb").read()).hexdigest()[:16],
                                 "config": {k: CONFIG[k] for k in keys}}))
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

    log(f"=== {tag} ===   ({jobs} GPUMD runs at a time on GPU(s) {', '.join(str(g) for g in gpus())})")
    systems = [("bulk", True), ("monolayer", False)]

    # --- 1. relaxed references (atoms + box), both systems at once
    refs = [prepare(os.path.join(root, name, "reference"), potabs, *hbn(bulk), box=True)
            for name, bulk in systems]
    ref_rows = dict(zip([n for n, _ in systems], run_many(refs, jobs)))
    ref = {}
    for (name, bulk), d in zip(systems, refs):
        f = os.path.join(d, "relaxed.xyz")
        if not os.path.isfile(f):
            sys.exit(f"{f} not found - check CONFIG['dump_line'] for your GPUMD version")
        sp, pos0, lat0 = read_last_xyz(f)
        ref[name] = (sp, pos0, lat0, abs(np.linalg.det(lat0)))

    # --- 2. every strained cell of both systems in one parallel batch
    deltas = sorted([-s for s in CONFIG["strains"]] + list(CONFIG["strains"]))
    dirs, meta = [], []
    for name, bulk in systems:
        sp, pos0, lat0, V = ref[name]
        for j in components(bulk):
            for dl in deltas:
                F = strain_matrix(j, dl)
                dirs.append(prepare(os.path.join(root, name, f"e{j + 1}_{dl:+.4f}"), potabs,
                                    sp, pos0 @ F.T, lat0 @ F.T, box=False))
                meta.append((name, j, dl))
    log(f"{len(dirs)} strained runs (bulk {len(ref['bulk'][0])} atoms, monolayer "
        f"{len(ref['monolayer'][0])} atoms) ...")
    rows = run_many(dirs, jobs)

    # --- 3. fits
    vals, mats = {}, {}
    for name, bulk in systems:
        sp, pos0, lat0, V = ref[name]
        mine = {(j, dl): r for (n, j, dl), r in zip(meta, rows) if n == name}
        C = symmetrize(fit_system(name, components(bulk), deltas, mine, ref_rows[name], V, log))
        if bulk:
            vals.update(hexagonal_bulk(C))
            mats["bulk"] = to_list(C)
        else:
            C2 = C * np.linalg.norm(lat0[2]) * 0.1          # GPa * A -> N/m
            vals.update(hexagonal_2d(C2))
            mats["monolayer_N_per_m"] = to_list(C2)

    out = {"key": key, "potential": os.path.basename(pot), "values": vals, "matrices": mats}
    json.dump(out, open(res_file, "w"), indent=1)
    logf.close()
    return out


def hexagonal_bulk(C):
    c11 = nanmean(C[0, 0], C[1, 1])
    c12 = C[0, 1]
    c13 = nanmean(C[0, 2], C[1, 2])
    c33 = C[2, 2]
    c44 = nanmean(C[3, 3], C[4, 4])
    c66 = C[5, 5]
    b_r = ((c11 + c12) * c33 - 2 * c13 ** 2) / (c11 + c12 + 2 * c33 - 4 * c13)
    return {"bulk.C11": c11, "bulk.C12": c12, "bulk.C13": c13, "bulk.C33": c33, "bulk.C44": c44,
            "bulk.C66": c66, "bulk.C66_check": 0.5 * (c11 - c12), "bulk.B_R": b_r}


def hexagonal_2d(C):
    c11 = nanmean(C[0, 0], C[1, 1])
    c12 = C[0, 1]
    c66 = C[5, 5]
    return {"monolayer.C11": c11, "monolayer.C12": c12, "monolayer.C66": c66,
            "monolayer.C66_check": 0.5 * (c11 - c12),
            "monolayer.Y": (c11 ** 2 - c12 ** 2) / c11, "monolayer.nu": c12 / c11}


# ----------------------------------------------------------------------------- user DFT
def read_vasp_outcar(path):
    """'TOTAL ELASTIC MODULI (kBar)' of a VASP IBRION=6 run -> 6x6 in GPa (Voigt xx yy zz yz xz xy),
    and the last lattice c length (A)"""
    txt = open(path, errors="ignore").read()
    blocks = txt.split("TOTAL ELASTIC MODULI (kBar)")
    if len(blocks) < 2:
        raise ValueError(f"{path}: no 'TOTAL ELASTIC MODULI' block (IBRION = 6 and ISIF = 3 needed)")
    lines = blocks[-1].splitlines()
    rows = {}
    for l in lines[1:12]:
        p = l.split()
        if len(p) == 7 and p[0] in ("XX", "YY", "ZZ", "XY", "YZ", "ZX"):
            rows[p[0]] = [float(v) for v in p[1:]]
    order = ["XX", "YY", "ZZ", "XY", "YZ", "ZX"]          # VASP order
    idx = {"XX": 0, "YY": 1, "ZZ": 2, "YZ": 3, "ZX": 4, "XY": 5}  # -> Voigt
    C = np.zeros((6, 6))
    for r in order:
        for cpos, cname in enumerate(order):
            C[idx[r], idx[cname]] = rows[r][cpos] / 10.0  # kBar -> GPa
    lat = re.findall(r"direct lattice vectors.*\n(.*)\n(.*)\n(.*)", txt)
    c_len = None
    if lat:
        v = np.array([l.split()[:3] for l in lat[-1]], float)
        c_len = float(np.linalg.norm(v[2]))
    return C, c_len


def user_dft():
    vals = {}
    for name, path in CONFIG["dft_outcar"].items():
        if not path:
            continue
        if not os.path.isfile(path):
            print(f"DFT OUTCAR for {name} not found: {path}")
            continue
        C, c_len = read_vasp_outcar(path)
        C = symmetrize(C)
        if name == "bulk":
            vals.update(hexagonal_bulk(C))
        else:
            vals.update(hexagonal_2d(C * c_len * 0.1))
    vals.update(CONFIG["dft_values"])
    return vals


# ----------------------------------------------------------------------------- report
def pct(x, ref):
    return 100 * (x - ref) / ref if (x is not None and ref) else None


def report(results, dft, tag):
    exp, lit = REFERENCES["exp"], REFERENCES["dft_lit"]
    names = [r["potential"] for r in results]
    w = max(10, max(len(n) for n in names))
    lines = ["Elastic constants: NEP (GPUMD, 0 K static, relaxed-ion) vs DFT and experiment",
             "bulk in GPa; monolayer as 2D values in N/m", ""]
    head = f"{'quantity':<24}{'unit':<5}" + "".join(f"{n:>{w + 2}}" for n in names) + \
           f"{CONFIG['dft_label'][:12]:>14}{'DFT (lit.)':>12}{'exp.':>14}"
    head += "".join(f"{'vs exp':>9}" for _ in names)
    if dft:
        head += "".join(f"{'vs DFT':>9}" for _ in names)
    lines += [head, "-" * len(head)]
    rows_csv = []

    def f(v, unit):
        if v is None:
            return "-"
        return f"{v:.4f}" if unit == "-" else f"{v:.1f}"

    for k, lab, unit in ROWS:
        nep = [r["values"].get(k) for r in results]
        e = exp.get(k, (None,))[0]
        err = exp[k][2] if k in exp and len(exp[k]) > 2 else None
        l, u = lit.get(k, (None,))[0], dft.get(k)
        e_txt = f(e, unit) if err is None else f"{f(e, unit)}±{err:g}"
        s = f"{lab:<24}{unit:<5}" + "".join(f"{f(v, unit):>{w + 2}}" for v in nep)
        s += f"{f(u, unit):>14}{f(l, unit):>12}{e_txt:>14}"
        small = k.endswith(("C13",))                       # ~0: percent is meaningless
        for v in nep:
            p = None if small else pct(v, e)
            s += f"{'-' if p is None else f'{p:+.1f}%':>9}"
        if dft:
            for v in nep:
                p = None if small else pct(v, u)
                s += f"{'-' if p is None else f'{p:+.1f}%':>9}"
        lines.append(s)
        rows_csv.append([k, unit] + nep + [u, l, e, err])
    lines += ["",
              "C66_check = (C11 - C12)/2 must equal C66 for hexagonal symmetry (a consistency check).",
              "B (Reuss) = [(C11+C12) C33 - 2 C13^2] / (C11 + C12 + 2 C33 - 4 C13).",
              "Monolayer 2D values = 3D cell stress x cell height; Y = (C11^2 - C12^2)/C11, nu = C12/C11.",
              "C13 is ~0, so no percent deviation is given for it.",
              "Experiment is at room temperature; NEP and DFT are static 0 K values.", ""]
    for r in results:
        lines.append(f"Full stiffness matrices, {r['potential']} (Voigt order xx yy zz yz xz xy)")
        for title, key, unit in (("bulk", "bulk", "GPa"), ("monolayer", "monolayer_N_per_m", "N/m")):
            M = r["matrices"][key]
            lines.append(f"  {title} ({unit}):")
            for row in M:
                lines.append("    " + " ".join(f"{'-':>9}" if v is None else f"{v:9.2f}" for v in row))
        lines.append("")
    lines.append("References")
    seen = set()
    for src in ("exp", "dft_lit"):
        for k, (v, ref, *_) in REFERENCES[src].items():
            if ref not in seen:
                seen.add(ref)
                lines.append(f"  [{'exp' if src == 'exp' else 'DFT'}] {ref}")
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(OUT, f"summary_{tag}.txt"), "w").write(txt + "\n")
    with open(os.path.join(OUT, f"summary_{tag}.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["quantity", "unit"] + names + [CONFIG["dft_label"], "DFT_literature",
                                                    "experiment", "experiment_uncertainty"])
        wr.writerows(rows_csv)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("potentials", nargs="+", help="NEP potential file(s)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="GPUMD runs at the same time (default: 4 per GPU of the job)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    if not (os.path.isfile(CONFIG["gpumd"]) and os.access(CONFIG["gpumd"], os.X_OK)):
        sys.exit(f"GPUMD executable '{CONFIG['gpumd']}' not found - set CONFIG['gpumd']")
    for p in args.potentials:
        if not os.path.isfile(p):
            sys.exit(f"potential file not found: {p}")
    os.makedirs(OUT, exist_ok=True)

    jobs = args.jobs or 4 * len(gpus())
    results = [compute(p, jobs, args.force) for p in args.potentials]
    dft = user_dft()
    tag = "_".join(os.path.splitext(os.path.basename(p))[0] for p in args.potentials)
    report(results, dft, tag)


if __name__ == "__main__":
    main()