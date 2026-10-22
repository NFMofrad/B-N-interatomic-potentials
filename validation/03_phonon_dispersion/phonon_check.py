#!/usr/bin/env python3
"""
phonon_check.py - phonon dispersion of bulk (AA') and monolayer hBN with a NEP potential:
forces from GPUMD, force constants and bands from phonopy, compared with experiment and DFT.

Usage
    python phonon_check.py nep_best.txt                    # one potential
    python phonon_check.py nep_v1.txt nep_best.txt         # compare several
    python phonon_check.py nep_best.txt --jobs 8           # GPUMD runs at the same time
    python phonon_check.py nep_best.txt --force            # ignore the cache

Method (harmonic, 0 K, finite displacements)
  1. The structure is relaxed with GPUMD (atoms + box, 'minimize fire ... 1') in the phonon
     supercell, and the equilibrium a (and c for bulk) is read back. All phonons are computed
     at the potential's own lattice.
  2. phonopy builds the hexagonal unit cell (bulk: 4 atoms, P6_3/mmc; monolayer: 2 atoms)
     and the displaced supercells (only 2 for bulk by symmetry).
  3. GPUMD computes the forces of each displaced supercell (single point,
     'dump_xyz 1 forces.xyz force').
  4. phonopy makes the force constants (symmetrized) and the bands along
     bulk G-M-K-G-A, monolayer G-M-K-G, plus the Gamma-point frequencies with symmetry labels.

Experiment
  - bulk Gamma point: Raman / IR frequencies (Serrano et al. 2007, Table I), matched to the
    calculated modes by symmetry label
  - bulk dispersion: 180 IXS points of Serrano et al. 2007, Fig. 2(a)
    (serrano2007_fig2a_IXS.csv, next to this script); plotted, and the deviation of every
    point from the nearest calculated branch is summarized (MAE, RMSE, max)

Optional
  - your DFT phonons on the same path: a phonopy run of VASP (phonopy_disp.yaml + FORCE_SETS,
    or phonopy_params.yaml) in CONFIG["dft_phonopy"]; drawn as dashed lines, Gamma table
    compared mode by mode. NAC (LO-TO splitting) is switched off for the comparison, because
    NEP has no long-range Coulomb term.
  - biaxial in-plane strains (CONFIG["biaxial_strains"], e.g. [-0.01, 0.01, 0.05, 0.10]):
    dispersion of the strained structures (dynamic stability).

Output (in phonon_check/)
    phonons_<names>_bulk.png / _monolayer.png / _strain_<system>.png   dispersion plots
    <potential>/<system>/                  GPUMD runs, band.yaml, phonopy_params.yaml
                                           (re-usable with the phonopy command line)
    <potential>/results.json               cached result

Needs numpy, matplotlib and phonopy (pip install --user phonopy).
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import numpy as np

# =============================================================================
# CONFIG - edit once
# =============================================================================
CONFIG = {
    "gpumd": "/scratch/work/fakhran1/GPUMD/src/gpumd",

    "a0": 2.504,                     # starting lattice (A); relaxed before the phonons
    "c0": 6.66,
    "c_monolayer": 25.0,             # cell height of the monolayer (vacuum direction)

    # phonopy supercells; None = smallest that is thicker than 2 x nep_cutoff
    # (with the defaults: bulk 6x6x2 = 288 atoms, monolayer 6x6x1 = 72 atoms)
    "supercell": {"bulk": None, "monolayer": None},
    "displacement": 0.01,            # A
    "npoints": 101,                  # q-points per path segment
    "dos_mesh": {"bulk": [24, 24, 8], "monolayer": [36, 36, 1]},   # q-mesh for the DOS
    "unit": "cm-1",                  # 'cm-1', 'THz' or 'meV'

    # relaxation of the structure (atoms + box)
    "box_force_tol": 1e-5,
    "box_max_steps": 100000,
    "dump_line": "dump_xyz 1 relaxed.xyz",
    # line that writes the forces of the displaced supercells, and the file it writes
    "force_dump_line": "dump_xyz 1 forces.xyz force",
    "force_file": "forces.xyz",

    "nep_cutoff": 6.0,               # largest radial cutoff of your NEP (A)

    # optional: biaxial in-plane strains, e.g. [-0.01, 0.01, 0.05, 0.10]
    # (bulk: c kept at its equilibrium value)
    "biaxial_strains": [-0.05, -0.01, 0.01, 0.05, 0.10],

    # optional: your DFT phonons (phonopy + VASP). Either the folder of the phonopy run
    # (with phonopy_disp.yaml and FORCE_SETS) or a phonopy_params.yaml that contains forces.
    "dft_phonopy": {"bulk": None, "monolayer": None},
    "dft_label": "DFT (this work)",

    # experimental phonon dispersion (bulk): IXS points of Serrano et al. 2007, Fig. 2(a),
    # shipped with this script; None = do not use
    "exp_dispersion_bulk": "serrano2007_fig2a_IXS.csv",
}
OUT = "phonon_check"
# =============================================================================

THZ = {"THz": 1.0, "cm-1": 33.35641, "meV": 4.135667}
UNIT_LABEL = {"THz": "THz", "cm-1": "cm$^{-1}$", "meV": "meV"}
MEV_TO_THZ = 1 / 4.135667

# experimental Gamma-point frequencies of bulk hBN (meV), Serrano et al., PRL 98, 095503 (2007),
# Table I (Raman and infrared data compiled there). E1u: TO and LO; NEP has no LO-TO splitting.
SERRANO = "Serrano et al., Phys. Rev. Lett. 98, 095503 (2007), Table I"
EXP_E1U_LO = 199.6   # meV, infrared (NEP has no LO-TO splitting)
EXP_GAMMA_BULK = [  # (label, meV, note)
    ("E2g", 6.4, "Raman (shear mode)"),
    ("A2u", 97.1, "infrared"),
    ("E2g", 169.4, "Raman"),
    ("E1u", 169.5, "infrared, TO"),
]

PATHS = {
    "bulk": ([[0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0], [0, 0, 0.5]],
             ["$\\Gamma$", "M", "K", "$\\Gamma$", "A"]),
    "monolayer": ([[0, 0, 0], [0.5, 0, 0], [1 / 3, 1 / 3, 0], [0, 0, 0]],
                  ["$\\Gamma$", "M", "K", "$\\Gamma$"]),
}
SQ3 = np.sqrt(3.0)


# ----------------------------------------------------------------------------- structures
def unit_cell(system, a, c):
    """hexagonal hBN cell: bulk AA' (B over N, P6_3/mmc, 4 atoms) or monolayer (2 atoms)"""
    from phonopy.structure.atoms import PhonopyAtoms
    if system == "bulk":
        cell = [[a, 0, 0], [-a / 2, a * SQ3 / 2, 0], [0, 0, c]]
        return PhonopyAtoms(symbols=["B", "N", "N", "B"], cell=cell,
                            scaled_positions=[[0, 0, .25], [1 / 3, 2 / 3, .25],
                                              [0, 0, .75], [1 / 3, 2 / 3, .75]])
    cell = [[a, 0, 0], [-a / 2, a * SQ3 / 2, 0], [0, 0, CONFIG["c_monolayer"]]]
    return PhonopyAtoms(symbols=["B", "N"], cell=cell,
                        scaled_positions=[[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]])


def supercell_size(system):
    s = CONFIG["supercell"].get(system)
    if s:
        return tuple(s)
    rc = 2 * CONFIG["nep_cutoff"] + 0.5
    n = int(np.ceil(rc / (0.95 * CONFIG["a0"] * SQ3 / 2)))      # in-plane width of the cell
    m = int(np.ceil(rc / (0.95 * CONFIG["c0"]))) if system == "bulk" else 1
    return (n, n, m)


def write_xyz(path, symbols, pos, lat):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write('Lattice="' + " ".join(f"{x:.10f}" for x in np.asarray(lat).flatten()) +
                '" pbc="T T T" Properties=species:S:1:pos:R:3\n')
        for s, p in zip(symbols, pos):
            f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def read_last_xyz_lattice(path):
    lines = open(path).read().splitlines()
    k, last = 0, None
    while k < len(lines):
        if not lines[k].strip():
            k += 1
            continue
        n = int(lines[k].split()[0])
        last = k
        k += n + 2
    head = lines[last + 1]
    i = head.lower().index("lattice=")
    return np.array(head[i + 8:].split('"')[1].split(), float).reshape(3, 3)


# ----------------------------------------------------------------------------- GPUMD
TAIL = """velocity 1e-6
ensemble nve
time_step 1e-6
dump_thermo 1
{extra}run 1
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
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        raise RuntimeError(f"GPUMD failed in {workdir}:\n  " + "\n  ".join(tail))
    return workdir


def run_many(dirs, jobs):
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


def prepare(workdir, pot, atoms, kind):
    """kind: 'relax' (atoms + box) or 'forces' (single point, forces written by dump_xyz)"""
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    write_xyz(os.path.join(workdir, "model.xyz"), atoms.symbols, atoms.positions, atoms.cell)
    if kind == "relax":
        head = f"potential {pot}\nminimize fire {CONFIG['box_force_tol']} {CONFIG['box_max_steps']} 1\n"
        extra = CONFIG["dump_line"].strip() + "\n"
    else:
        head = f"potential {pot}\n"
        extra = CONFIG["force_dump_line"].strip() + "\n"
    open(os.path.join(workdir, "run.in"), "w").write(head + TAIL.format(extra=extra))
    return workdir


def read_forces(workdir, natoms):
    """forces (eV/A) of the last frame of the extended-XYZ file written by dump_xyz ... force;
    the force columns are found by name in its Properties= entry"""
    f = os.path.join(workdir, CONFIG["force_file"])
    if not os.path.isfile(f):
        sys.exit(f"{f} not found - check CONFIG['force_dump_line'] / ['force_file']")
    lines = open(f).read().splitlines()
    k, last = 0, None
    while k < len(lines):
        if not lines[k].strip():
            k += 1
            continue
        n = int(lines[k].split()[0])
        last = (k, n)
        k += n + 2
    k, n = last
    if n != natoms:
        sys.exit(f"{f}: {n} atoms in the file, expected {natoms}")
    head = lines[k + 1]
    low = head.lower()
    if "properties=" not in low:
        sys.exit(f"{f}: no Properties= entry in the comment line")
    props = head[low.index("properties=") + 11:].split()[0].split(":")
    col, i_f = 0, None
    for name, typ, cnt in zip(props[0::3], props[1::3], props[2::3]):
        if name.lower().startswith("force"):
            i_f = col
        col += int(cnt)
    if i_f is None:
        sys.exit(f"{f}: no force column in '{':'.join(props)}' - add 'force' to the dump_xyz line")
    F = np.array([lines[k + 2 + m].split()[i_f:i_f + 3] for m in range(n)], float)
    return F


# ----------------------------------------------------------------------------- phonopy helpers
def band_data(ph, system):
    from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
    path, labels = PATHS[system]
    qpts, conn = get_band_qpoints_and_path_connections([path], npoints=CONFIG["npoints"])
    ph.run_band_structure(qpts, path_connections=conn, labels=labels)
    d = ph.get_band_structure_dict()
    return {"distances": [x.tolist() for x in d["distances"]],
            "frequencies": [x.tolist() for x in d["frequencies"]],   # THz
            "labels": labels}


def dos_data(ph, system):
    """phonon DOS (states per THz per unit cell) projected on B and N"""
    mesh = CONFIG["dos_mesh"][system]
    ph.run_mesh(mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    ph.run_projected_dos(freq_pitch=0.05)
    d = ph.get_projected_dos_dict()
    pd = np.array(d["projected_dos"])
    sym = list(ph.primitive.symbols)
    by = {el: pd[[i for i, e in enumerate(sym) if e == el]].sum(0).tolist() for el in sorted(set(sym))}
    return {"f": np.array(d["frequency_points"]).tolist(), "total": pd.sum(0).tolist(), **by}


def gamma_modes(ph):
    """Gamma frequencies (THz) grouped by degeneracy, with symmetry labels from phonopy"""
    out = []
    try:
        if hasattr(ph, "run_irreps"):
            ph.run_irreps([0, 0, 0], degeneracy_tolerance=1e-3)
        else:                                               # older phonopy
            ph.set_irreps([0, 0, 0], degeneracy_tolerance=1e-3)
        ir = ph.irreps
        freqs = ir.frequencies
        labels = getattr(ir, "_ir_labels", None)
        for k, group in enumerate(ir.band_indices):
            f = float(np.mean([freqs[i] for i in group]))
            lab = labels[k] if labels is not None and k < len(labels) and labels[k] else None
            if lab is None:
                lab = "acoust" if abs(f) * THZ["cm-1"] < 5 else "?"
            out.append({"label": lab, "freq": f, "deg": len(group)})
    except Exception:                                       # labels are a bonus only
        out = []
    if not out:
        ph.run_qpoints([[0, 0, 0]])
        freqs = ph.get_qpoints_dict()["frequencies"][0]
        for f in freqs:
            out.append({"label": "?", "freq": float(f), "deg": 1})
    return out


def min_frequency(bands):
    fmin, where = np.inf, None
    for seg_i, seg in enumerate(bands["frequencies"]):
        a = np.array(seg)
        k = np.unravel_index(np.argmin(a), a.shape)
        if a[k] < fmin:
            fmin, where = float(a[k]), (seg_i, k[0] / max(1, a.shape[0] - 1))
    return fmin, where


SEGMENT_INDEX = {"bulk": {"G-M": 0, "M-K": 1, "K-G": 2, "G-A": 3}}


def load_exp_dispersion(system):
    """IXS points as list of (segment index, t, frequency THz, polarization), or []"""
    if system != "bulk" or not CONFIG.get("exp_dispersion_bulk"):
        return []
    path = CONFIG["exp_dispersion_bulk"]
    if not os.path.isabs(path) and not os.path.isfile(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.isfile(path):
        print(f"experimental dispersion not found: {CONFIG['exp_dispersion_bulk']}")
        return []
    pts = []
    for line in open(path):
        if line.startswith("#") or line.startswith("segment") or not line.strip():
            continue
        seg, t, e, pol = line.strip().split(",")
        pts.append((SEGMENT_INDEX["bulk"][seg], float(t), float(e) * MEV_TO_THZ, pol))
    return pts


def bands_at(bands, seg, t):
    """all band frequencies (THz) at fractional position t along path segment 'seg'"""
    d = np.array(bands["distances"][seg])
    f = np.array(bands["frequencies"][seg])
    x = d[0] + t * (d[-1] - d[0])
    return np.array([np.interp(x, d, f[:, b]) for b in range(f.shape[1])])


def exp_deviation(bands, pts):
    """deviation of each experimental point from the nearest calculated branch (THz)"""
    out = {"all": [], "in-plane": [], "out-of-plane": []}
    for seg, t, f_exp, pol in pts:
        dev = float(np.min(np.abs(bands_at(bands, seg, t) - f_exp)))
        out["all"].append(dev)
        out.setdefault(pol, []).append(dev)
    return out


# ----------------------------------------------------------------------------- per potential
def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def phonons_at(root, potabs, system, a, c, jobs, tag):
    """displacements -> GPUMD forces -> phonopy; returns (bands, gamma, ph)"""
    from phonopy import Phonopy
    uc = unit_cell(system, a, c)
    ph = Phonopy(uc, supercell_matrix=np.diag(supercell_size(system)))
    ph.generate_displacements(distance=CONFIG["displacement"])
    scs = ph.supercells_with_displacements
    dirs = [prepare(os.path.join(root, f"{tag}disp_{i + 1:03d}"), potabs, sc, "forces")
            for i, sc in enumerate(scs)]
    run_many(dirs, jobs)
    ph.forces = np.array([read_forces(d, len(sc)) for d, sc in zip(dirs, scs)])
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    return band_data(ph, system), gamma_modes(ph), ph


def compute(pot, jobs, force):
    tag = os.path.splitext(os.path.basename(pot))[0]
    root = os.path.join(OUT, tag)
    res_file = os.path.join(root, "results.json")
    keys = ["a0", "c0", "c_monolayer", "supercell", "displacement", "npoints", "box_force_tol",
            "box_max_steps", "nep_cutoff", "biaxial_strains", "dos_mesh"]
    key = json.loads(json.dumps({"v": 2, "sha": sha(pot), "config": {k: CONFIG[k] for k in keys}}))
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
    out = {"key": key, "potential": os.path.basename(pot), "systems": {}}

    # --- 1. equilibrium lattice, both systems at once, in the phonon supercell
    lat = {}
    systems = ["bulk", "monolayer"]
    rdirs = []
    for s in systems:
        n = supercell_size(s)
        sc = unit_cell(s, CONFIG["a0"], CONFIG["c0"])
        from phonopy.structure.cells import get_supercell
        rdirs.append(prepare(os.path.join(root, s, "relax"), potabs,
                             get_supercell(sc, np.diag(n)), "relax"))
    run_many(rdirs, jobs)
    for s, d in zip(systems, rdirs):
        f = os.path.join(d, "relaxed.xyz")
        if not os.path.isfile(f):
            sys.exit(f"{f} not found - check CONFIG['dump_line'] for your GPUMD version")
        L = read_last_xyz_lattice(f)
        n = supercell_size(s)
        a = 0.5 * (np.linalg.norm(L[0]) / n[0] + np.linalg.norm(L[1]) / n[1])
        c = np.linalg.norm(L[2]) / n[2] if s == "bulk" else None
        lat[s] = (a, c)
        log(f"{s:<10} relaxed: a = {a:.4f} A" + (f", c = {c:.4f} A" if c else "") +
            f"   (supercell {n[0]}x{n[1]}x{n[2]})")

    # --- 2. phonons at equilibrium
    for s in systems:
        a, c = lat[s]
        bands, gamma, ph = phonons_at(os.path.join(root, s), potabs, s, a, c, jobs, "")
        ph.save(filename=os.path.join(root, s, "phonopy_params.yaml"))
        ph.write_yaml_band_structure(filename=os.path.join(root, s, "band.yaml"))
        fmin, where = min_frequency(bands)
        out["systems"][s] = {"a": a, "c": c, "bands": bands, "gamma": gamma, "dos": dos_data(ph, s),
                             "min_freq": fmin,
                             "min_where": where, "strains": {}}
        log(f"{s:<10} {len(ph.supercells_with_displacements)} displacements; lowest frequency on "
            f"the path {fmin * THZ['cm-1']:.1f} cm-1")

    # --- 3. biaxial strains (optional)
    for eps in CONFIG["biaxial_strains"]:
        for s in systems:
            a, c = lat[s]
            sub = os.path.join(root, s, f"strain_{eps:+.3f}")
            os.makedirs(sub, exist_ok=True)
            bands, gamma, ph = phonons_at(sub, potabs, s, a * (1 + eps), c, jobs, "")
            ph.write_yaml_band_structure(filename=os.path.join(sub, "band.yaml"))
            fmin, where = min_frequency(bands)
            out["systems"][s]["strains"][f"{eps:+.4f}"] = {"bands": bands, "gamma": gamma,
                                                           "min_freq": fmin}
            log(f"{s:<10} biaxial strain {100 * eps:+.1f} %: lowest frequency "
                f"{fmin * THZ['cm-1']:.1f} cm-1")

    json.dump(out, open(res_file, "w"))
    logf.close()
    return out


# ----------------------------------------------------------------------------- user DFT
def user_dft():
    import phonopy
    res = {}
    for s, path in CONFIG["dft_phonopy"].items():
        if not path:
            continue
        try:
            if os.path.isdir(path):
                yml = os.path.join(path, "phonopy_disp.yaml")
                fs = os.path.join(path, "FORCE_SETS")
                ph = phonopy.load(yml, force_sets_filename=fs if os.path.isfile(fs) else None,
                                  is_nac=False, log_level=0)
            else:
                fs = os.path.join(os.path.dirname(path), "FORCE_SETS")
                ph = phonopy.load(path, force_sets_filename=fs if os.path.isfile(fs) else None,
                                  is_nac=False, log_level=0)
        except Exception as e:
            print(f"DFT phonopy run for {s} could not be loaded ({path}): {e}")
            continue
        res[s] = {"bands": band_data(ph, s), "gamma": gamma_modes(ph), "dos": dos_data(ph, s)}
    return res


# ----------------------------------------------------------------------------- report
def report(results, dft, tag):
    u = CONFIG["unit"]
    k = THZ[u]
    lines = [f"Phonons of hBN: NEP (GPUMD forces + phonopy, harmonic, 0 K) vs DFT and experiment",
             f"frequencies in {u}; no LO-TO splitting in NEP (and none in the DFT comparison)", ""]
    for s in ("bulk", "monolayer"):
        lines.append(f"=== {s} ===")
        for r in results:
            d = r["systems"][s]
            lat = f"a = {d['a']:.4f} A" + (f", c = {d['c']:.4f} A" if d["c"] else "")
            lines.append(f"{r['potential']}: {lat}; lowest frequency on the path "
                         f"{d['min_freq'] * k:.1f} {u}"
                         + ("   <-- imaginary (negative) modes" if d["min_freq"] * THZ['cm-1'] < -5 else ""))
        lines.append("")
        # Gamma table: modes ordered by frequency, labels from phonopy
        cols = [(r["potential"], r["systems"][s]["gamma"]) for r in results]
        if s in dft:
            cols.append((CONFIG["dft_label"], dft[s]["gamma"]))
        head = f"  {'mode':<6}{'deg':>4}" + "".join(f"{n[:16]:>18}" for n, _ in cols)
        if s == "bulk":
            head += f"{'experiment':>14}  note"
        lines += ["  Gamma point", head, "  " + "-" * (len(head) - 2)]
        ref = cols[0][1]

        def by_label(modes):
            """index of each mode within its symmetry label, ordered by frequency"""
            cnt, keyed = {}, []
            for m in sorted(modes, key=lambda m: m["freq"]):
                n = cnt.get(m["label"], 0)
                cnt[m["label"]] = n + 1
                keyed.append(((m["label"], n), m))
            return dict(keyed)

        matched = [by_label(g) for _, g in cols]
        exp_k = {}
        for lab, mev, note in sorted(EXP_GAMMA_BULK, key=lambda t: t[1]):
            n = sum(1 for kk in exp_k if kk[0] == lab)
            exp_k[(lab, n)] = (mev, note)
        for key, m in by_label(ref).items():
            row = f"  {m['label']:<6}{m['deg']:>4}"
            for mm in matched:
                v = mm.get(key)
                row += f"{(v['freq'] * k if v else float('nan')):>18.1f}"
            if s == "bulk" and key in exp_k:
                mev, note = exp_k[key]
                if key[0] == "E1u":
                    note += f" (LO: {EXP_E1U_LO * MEV_TO_THZ * k:.1f})"
                row += f"{mev * MEV_TO_THZ * k:>14.1f}  {note}"
            lines.append(row)
        lines.append("")
        # experimental dispersion (IXS)
        pts = load_exp_dispersion(s)
        if pts:
            lines.append(f"  IXS dispersion ({len(pts)} points, Serrano et al. 2007, Fig. 2a): deviation of each "
                         f"point from the nearest calculated branch, in {u}")
            lines.append(f"  {'':<22}{'MAE':>8}{'RMSE':>8}{'max':>8}   in-plane MAE   out-of-plane MAE")
            sets = [(r["potential"], r["systems"][s]["bands"]) for r in results]
            if s in dft:
                sets.append((CONFIG["dft_label"], dft[s]["bands"]))
            for name, b in sets:
                dv = exp_deviation(b, pts)
                a = np.array(dv["all"]) * k
                ip, op = np.array(dv["in-plane"]) * k, np.array(dv["out-of-plane"]) * k
                lines.append(f"  {name[:22]:<22}{a.mean():8.1f}{np.sqrt((a ** 2).mean()):8.1f}{a.max():8.1f}"
                             f"   {ip.mean():12.1f}   {op.mean():16.1f}")
            lines.append("")
        # strains
        for r in results:
            st = r["systems"][s]["strains"]
            if not st:
                continue
            eps = sorted(st, key=float)
            lines.append(f"  biaxial strain, {r['potential']}: lowest frequency on the path")
            for e in eps:
                fm = st[e]["min_freq"] * k
                lines.append(f"    {100 * float(e):+6.1f} %   {fm:9.1f} {u}"
                             + (("   <-- imaginary modes" + ("  (compression: expected buckling of the sheet)"
                                                             if float(e) < 0 else "  <-- instability"))
                                if st[e]["min_freq"] * THZ['cm-1'] < -5 else ""))
            lines.append("")
    lines += ["Notes",
              "  Experiment (bulk, room temperature): " + SERRANO + "; dispersion: IXS points of",
              "  Fig. 2(a), extracted from the vector graphics of the paper (serrano2007_fig2a_IXS.csv).",
              "  Nearest-branch deviation: each IXS point is compared with the closest calculated",
              "  branch at the same q, so the numbers measure how well the bands pass through the data.",
              "  E1u: NEP and DFT without NAC give one E1u; compare it with the TO value.",
              "  Small negative frequencies of the ZA branch right at Gamma (a few cm-1) are a",
              "  numerical artefact of finite supercells, not an instability."]
    txt = "\n".join(lines)
    print("\n" + txt)


# ----------------------------------------------------------------------------- plots
def plot(results, dft, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    INK, INK2, MUTED, GRID, SURF, BAND = "#1f1f1d", "#4a4a46", "#8a8984", "#e6e5e0", "#ffffff", "#f7f6f2"
    NEP_COLS = ["#1f1f1d", "#d9480f", "#1baf7a", "#7048e8", "#e64980", "#0c8599"]
    DFT_COL = "#868e96"
    EXP_IP, EXP_OP = "#e03131", "#1c7ed6"            # IXS in-plane (open) / out-of-plane (filled)
    RAMAN, IR = "#2b8a3e", "#ae3ec9"
    DOS_COL = {"B": "#f08c00", "N": "#4263eb"}
    plt.rcParams.update({
        "font.size": 10.5, "font.family": "sans-serif",
        "axes.edgecolor": INK2, "axes.linewidth": 0.9, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.direction": "in", "ytick.direction": "in", "ytick.minor.visible": True,
        "ytick.major.size": 4, "ytick.minor.size": 2, "axes.titlesize": 12,
    })
    u, k = CONFIG["unit"], THZ[CONFIG["unit"]]
    # second energy axis (right): meV when plotting cm-1, otherwise cm-1
    u2 = "meV" if u != "meV" else "cm-1"
    conv = THZ[u2] / THZ[u]

    def xmap(dist, rseg):
        dist, rseg = np.array(dist), np.array(rseg)
        return rseg[0] + (dist - dist[0]) / max(dist[-1] - dist[0], 1e-12) * (rseg[-1] - rseg[0])

    def draw(ax, bands, x_ref, col, ls="-", lw=1.5, z=3, alpha=1.0):
        for seg, dist, rseg in zip(bands["frequencies"], bands["distances"], x_ref):
            ax.plot(xmap(dist, rseg), np.array(seg) * k, ls=ls, color=col, lw=lw, zorder=z,
                    alpha=alpha, solid_capstyle="round", dash_capstyle="round")

    def frame(ax, ref):
        ticks = [ref["distances"][0][0]] + [d[-1] for d in ref["distances"]]
        for t in ticks[1:-1]:
            ax.axvline(t, color=INK2, lw=0.7, zorder=1)
        ax.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ref["labels"], fontsize=12)
        ax.set_xlim(ticks[0], ticks[-1])
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_ylabel(f"Frequency ({UNIT_LABEL[u]})", fontsize=11.5)
        return ticks

    def ylim_of(*band_sets, extra=()):
        hi = max([np.max(np.concatenate([np.array(sg).ravel() for sg in b["frequencies"]])) for b in band_sets]
                 + list(extra))
        lo = min([np.min(np.concatenate([np.array(sg).ravel() for sg in b["frequencies"]])) for b in band_sets])
        return min(lo * k * 1.1, -0.02 * hi * k), hi * k * 1.06

    def dos_panel(axd, dos_sets, ylim):
        """B / N projected DOS of the first potential (filled) + total of each dataset (lines)"""
        first = dos_sets[0][1]
        f = np.array(first["f"]) * k
        scale = max(np.max(first["total"]), 1e-12)
        for el in ("B", "N"):
            if el in first:
                axd.fill_betweenx(f, 0, np.array(first[el]) / scale, color=DOS_COL[el], alpha=0.35, lw=0)
                axd.plot(np.array(first[el]) / scale, f, color=DOS_COL[el], lw=0.9)
        for name, d, col, ls in dos_sets:
            axd.plot(np.array(d["total"]) / scale, np.array(d["f"]) * k, color=col, lw=1.1, ls=ls)
        axd.set_ylim(*ylim)
        axd.set_xlim(0, 1.08 * max(np.max(np.array(d["total"]) / scale) for _, d, _, _ in dos_sets))
        axd.set_xticks([])
        axd.set_xlabel("DOS", fontsize=11, labelpad=8)
        axd.tick_params(axis="y", labelleft=False)
        axd.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 3)))
        sec = axd.secondary_yaxis("right", functions=(lambda y: y * conv, lambda y: y / conv))
        sec.set_ylabel(f"Energy ({UNIT_LABEL[u2]})", fontsize=11.5, rotation=270, labelpad=16)
        sec.tick_params(direction="in", colors=INK2, labelcolor=INK)
        sec.minorticks_on()

    # ======================================================================== dispersions
    for s in ("bulk", "monolayer"):
        ref = results[0]["systems"][s]["bands"]
        pts = load_exp_dispersion(s)
        band_sets = [r["systems"][s]["bands"] for r in results] + ([dft[s]["bands"]] if s in dft else [])
        exp_hi = [p[2] for p in pts] + ([EXP_E1U_LO * MEV_TO_THZ] if s == "bulk" else [])
        ylim = ylim_of(*band_sets, extra=exp_hi)

        two_lines = s == "bulk" and bool(pts)
        fig = plt.figure(figsize=(9.2, 6.9 if two_lines else 6.6), dpi=220, facecolor=SURF)
        gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1], wspace=0.04,
                              left=0.1, right=0.9, top=0.775 if two_lines else 0.81, bottom=0.1)
        ax = fig.add_subplot(gs[0])
        axd = fig.add_subplot(gs[1], sharey=ax)
        ticks = frame(ax, ref)
        # alternate light bands for the path segments
        for t0, t1 in list(zip(ticks[:-1], ticks[1:]))[1::2]:
            ax.axvspan(t0, t1, color=BAND, lw=0, zorder=0)

        handles = []
        if s in dft:
            draw(ax, dft[s]["bands"], ref["distances"], DFT_COL, ls=(0, (5, 2.5)), lw=1.4, z=2)
            handles.append(Line2D([], [], color=DFT_COL, ls=(0, (5, 2.5)), lw=1.4, label=CONFIG["dft_label"]))
        for j, r in enumerate(results):
            col = NEP_COLS[j % len(NEP_COLS)]
            draw(ax, r["systems"][s]["bands"], ref["distances"], col, lw=1.7 if j == 0 else 1.4, z=4)
            handles.append(Line2D([], [], color=col, lw=1.7, label=f"NEP  {r['potential']}"))
        if pts:
            for pol, fc, ec in (("in-plane", SURF, EXP_IP), ("out-of-plane", EXP_OP, EXP_OP)):
                sel = [p for p in pts if p[3] == pol]
                xs = [ref["distances"][sg][0] + t * (ref["distances"][sg][-1] - ref["distances"][sg][0])
                      for sg, t, f, _ in sel]
                ax.scatter(xs, [f * k for _, _, f, _ in sel], s=20, marker="o", facecolor=fc,
                           edgecolor=ec, lw=1.0, zorder=6)
            handles += [Line2D([], [], ls="", marker="o", ms=5.5, markerfacecolor=SURF, markeredgecolor=EXP_IP,
                               markeredgewidth=1.0, label="IXS, in-plane"),
                        Line2D([], [], ls="", marker="o", ms=5.5, color=EXP_OP, label="IXS, out-of-plane")]
        if s == "bulk":
            g = [t for t, lab in zip(ticks, ref["labels"]) if "Gamma" in lab]
            for lab, mev, note in EXP_GAMMA_BULK:
                kind_ir = "infrared" in note
                for xg in g:
                    ax.scatter([xg], [mev * MEV_TO_THZ * k], marker="D", s=34, zorder=7, clip_on=False,
                               facecolor=SURF if kind_ir else RAMAN, edgecolor=IR if kind_ir else RAMAN, lw=1.3)
            for xg in g:      # E1u LO (infrared)
                ax.scatter([xg], [EXP_E1U_LO * MEV_TO_THZ * k], marker="D", s=34, zorder=7, clip_on=False,
                           facecolor=SURF, edgecolor=IR, lw=1.3)
            handles += [Line2D([], [], ls="", marker="D", ms=5.5, color=RAMAN, label="Raman (Γ)"),
                        Line2D([], [], ls="", marker="D", ms=5.5, markerfacecolor=SURF, markeredgecolor=IR,
                               markeredgewidth=1.3, label="infrared (Γ)")]
        ax.set_ylim(*ylim)

        dos_sets = [(r["potential"], r["systems"][s]["dos"], NEP_COLS[j % len(NEP_COLS)], "-")
                    for j, r in enumerate(results) if "dos" in r["systems"][s]]
        if s in dft and "dos" in dft[s]:
            dos_sets.append((CONFIG["dft_label"], dft[s]["dos"], DFT_COL, (0, (5, 2.5))))
        if dos_sets:
            dos_panel(axd, dos_sets, ylim)
            handles += [Patch(facecolor=DOS_COL["B"], alpha=0.5, label="DOS, B"),
                        Patch(facecolor=DOS_COL["N"], alpha=0.5, label="DOS, N")]
        else:
            axd.axis("off")

        title = "Bulk hBN (AA′ stacking)" if s == "bulk" else "Monolayer hBN"
        fig.text(0.1, 0.978, title, fontsize=15, fontweight="bold", color=INK, va="top")
        sub = "phonon dispersion and density of states  ·  NEP: GPUMD forces + phonopy (harmonic, 0 K)"
        if s == "bulk" and pts:
            sub += "\nexperiment: IXS dispersion, Raman and infrared at Γ (Serrano et al., PRL 98, 095503, 2007)"
        fig.text(0.1, 0.94, sub, fontsize=9.5, color=INK2, va="top", linespacing=1.4)
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.015), ncol=min(4, len(handles)),
                  frameon=False, fontsize=8.8, handletextpad=0.5, columnspacing=1.6, borderaxespad=0.2)
        fig.savefig(os.path.join(OUT, f"phonons_{tag}_{s}.png"), facecolor=SURF)
        fig.savefig(os.path.join(OUT, f"phonons_{tag}_{s}.pdf"), facecolor=SURF)
        plt.close(fig)

        # ==================================================================== strain
        st = results[0]["systems"][s]["strains"]
        if not st:
            continue
        eps = sorted(st, key=float)
        e_all = [float(e) for e in eps] + [0.0]
        vmin, vmax = min(min(e_all), -1e-6), max(max(e_all), 1e-6)
        cmap = LinearSegmentedColormap.from_list("strain", ["#1864ab", "#74c0fc", "#212529", "#ffa8a8", "#c92a2a"])
        norm = TwoSlopeNorm(vmin=vmin * 100, vcenter=0, vmax=vmax * 100)

        fig = plt.figure(figsize=(9.2, 6.2), dpi=220, facecolor=SURF)
        gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 0.12], wspace=0.04,
                              left=0.1, right=0.9, top=0.83, bottom=0.1)
        ax = fig.add_subplot(gs[0])
        cax = fig.add_subplot(gs[1])
        ticks = frame(ax, ref)
        for t0, t1 in list(zip(ticks[:-1], ticks[1:]))[1::2]:
            ax.axvspan(t0, t1, color=BAND, lw=0, zorder=0)
        for e in eps:
            draw(ax, st[e]["bands"], ref["distances"], cmap(norm(100 * float(e))), lw=1.1, z=3, alpha=0.95)
        draw(ax, ref, ref["distances"], cmap(norm(0)), lw=1.9, z=4)
        ax.set_ylim(*ylim_of(ref, *[st[e]["bands"] for e in eps]))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("strain (%)", fontsize=10.5, labelpad=2)
        cb.outline.set_edgecolor(INK2)
        cb.set_ticks(sorted(set([100 * x for x in e_all])))
        cb.ax.yaxis.set_ticks_position("right")

        title = ("Bulk hBN (AA′)" if s == "bulk" else "Monolayer hBN") + " under biaxial strain"
        fig.text(0.1, 0.975, title, fontsize=15, fontweight="bold", color=INK, va="top")
        fig.text(0.1, 0.925, f"NEP {results[0]['potential']}  ·  phonon dispersion coloured by strain "
                 f"(black: unstrained)" + ("  ·  c fixed at equilibrium" if s == "bulk" else ""),
                 fontsize=9.5, color=INK2, va="top")
        fig.savefig(os.path.join(OUT, f"phonons_{tag}_strain_{s}.png"), facecolor=SURF)
        fig.savefig(os.path.join(OUT, f"phonons_{tag}_strain_{s}.pdf"), facecolor=SURF)
        plt.close(fig)
    print(f"plots -> {OUT}/phonons_{tag}_*.png / .pdf")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("potentials", nargs="+", help="NEP potential file(s)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="GPUMD runs at the same time (default: 4 per GPU of the job)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    try:
        import phonopy  # noqa: F401
    except ImportError:
        sys.exit("phonopy is not installed: pip install --user phonopy")
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
    plot(results, dft, tag)


if __name__ == "__main__":
    main()