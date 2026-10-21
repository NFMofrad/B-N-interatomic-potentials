#!/usr/bin/env python3
"""
dataset_check.py - energy and force check of NEP training files, with one summary figure.

    python dataset_check.py train.xyz
    python dataset_check.py train.xyz test.xyz --top 15 --out check.pdf

Prints
  - frames with a missing energy or missing forces,
  - E/atom (mean, std, min, max) overall, for frames with > 2 atoms, and per data group
    (group = first part of Config_type, e.g. "B-B" from "B-B/str-b2-dimer-012"),
  - with --outliers: frames whose E/atom is far from the rest of their group,
  - the frames with the highest E/atom,
  - |F| per atom (mean, median, p99, max) overall and per element,
  - the frames with the largest single-atom force.
Plot (one figure, one row, two panels; energy = green, bottom/left axes;
force = blue, top/right axes)
  left : by group, one row per group: E/atom of every frame (upper strip, bottom axis)
         and the force of every frame (lower strip, top axis) - by default the largest
         |F| in the frame, or its RMS / mean with --frame-stat; black tick = median,
         red = force outlier (see --frame-stat) or energy outlier (--outliers)
  right: per-atom distributions: E/atom of all frames (bottom/left axes; > 2 atoms solid,
         dimers/atoms dashed) and |F| per atom (top/right axes; B solid, N dashed)

Reads the extended XYZ directly (keys are case-insensitive, so lowercase 'energy=' /
'lattice=' / 'properties=' work - ASE misplaces or ignores those). Needs numpy + matplotlib.
"""
import argparse, os, re
import numpy as np

KV = re.compile(r'(\w+)\s*=\s*("[^"]*"|\S+)')
ENERGY_NAMES = ("energy", "ref_energy", "dft_energy", "free_energy")
FORCE_NAMES = ("forces", "force", "ref_forces", "dft_forces")


# ====================================================================== reading
def frames(path):
    with open(path) as f:
        k = -1
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            k += 1
            n = int(line.split()[0])
            comment = f.readline()
            rows = [f.readline().split() for _ in range(n)]
            kv = {key.lower(): v.strip('"') for key, v in KV.findall(comment)}
            props = kv.get("properties", "species:S:1:pos:R:3").split(":")
            col, i_sp, i_f = 0, 0, None
            for name, typ, cnt in zip(props[0::3], props[1::3], props[2::3]):
                if name.lower() == "species":
                    i_sp = col
                if name.lower() in FORCE_NAMES:
                    i_f = col
                col += int(cnt)
            e = next((float(kv[key]) for key in ENERGY_NAMES if key in kv), None)
            F = (np.array([[float(x) for x in r[i_f:i_f + 3]] for r in rows])
                 if i_f is not None else None)
            cfg = kv.get("config_type", "-")
            group = cfg.split("/")[0] if cfg != "-" else os.path.basename(path)
            yield {"index": k, "file": path, "n": n, "E": e, "F": F,
                   "sp": [r[i_sp] for r in rows], "config": cfg, "group": group}


def where(r):
    return f"frame {r['index']:6d}  {r['file']} / {r['config']}"


# ====================================================================== group names
# Config_type prefixes are often short folder names ("16_ds4_bnnts"). For the figure they
# are turned into readable labels: the ordering prefix ("16_") is dropped, "_" and "-"
# become spaces, and common abbreviations get their usual spelling. A names file
# (--names) overrides any of them.
WORDS = {
    "hbn": "hBN", "cbn": "c-BN", "wbn": "w-BN", "bn": "BN", "bnnt": "BNNT", "bnnts": "BNNTs",
    "gb": "GB", "gbs": "grain boundaries", "mlhbn": "multilayer hBN", "mlmd": "ML-MD",
    "md": "MD", "aimd": "AIMD", "dft": "DFT", "hight": "high-T", "lowt": "low-T",
    "tetra": "tetra-", "bilayer": "bilayer", "trilayer": "trilayer", "multilayer": "multilayer",
    "optimization": "relaxed", "opt": "relaxed", "snapshots": "snapshots",
}


def auto_name(g):
    s = re.sub(r"^\d+[_\-. ]+", "", g)                 # drop "16_" ordering prefix
    m = re.fullmatch(r"([A-Z][a-z]?)-([A-Z][a-z]?)", s)  # "B-N" -> "B–N dimers"
    if m:
        return f"{m.group(1)}–{m.group(2)} dimers"
    m = re.fullmatch(r"ds(\d+)", s, re.I)
    if m:
        return f"Data set {m.group(1)}"
    out = []
    for tok in re.split(r"[_\- ]+", s):
        low = tok.lower()
        if not tok or re.fullmatch(r"ds\d+", low):         # data-set tags carry no meaning
            continue
        out.append(WORDS.get(low, tok))
    text = " ".join(out).replace("tetra- ", "tetra")
    return text[:1].upper() + text[1:] if text else g


def read_names(path):
    """names file: one 'Config_type group = Display name' per line, '#' starts a comment"""
    names = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            names[k.strip()] = v.strip()
    return names


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=10, help="list this many frames in the top-N tables")
    ap.add_argument("--out", default="dataset_check.png", help="figure file (.png, .pdf, .svg ...)")
    ap.add_argument("--frame-stat", choices=("max", "rms", "mean"), default="max",
                    help="per-frame force in the by-group panel: largest |F| (default, best "
                         "for finding bad frames), RMS |F| or mean |F| (how far from "
                         "equilibrium a frame is)")
    ap.add_argument("--names", metavar="FILE",
                    help="file with 'group = Display name' lines for the figure labels")
    ap.add_argument("--raw-names", action="store_true",
                    help="label groups with their Config_type prefix exactly as in the file")
    ap.add_argument("--outliers", action="store_true",
                    help="flag frames whose E/atom is far from their group median (off by default)")
    args = ap.parse_args()

    recs = []                     # one dict per frame
    mags, elems = [], []          # per atom, frames with forces only
    for path in args.files:
        for fr in frames(path):
            if fr["F"] is not None:
                m = np.linalg.norm(fr["F"], axis=1)
                mags.append(m)
                elems += fr["sp"]
                j = int(np.argmax(m))
                fr["fmax"], fr["fmax_el"] = float(m[j]), fr["sp"][j]
                fr["frms"], fr["fmean"] = float(np.sqrt(np.mean(m ** 2))), float(m.mean())
            else:
                fr["fmax"], fr["fmax_el"], fr["frms"], fr["fmean"] = None, None, None, None
            fr["F"] = None        # keep memory low for big files
            recs.append(fr)
    if not recs:
        raise SystemExit("no frames found")

    groups = sorted({r["group"] for r in recs},
                    key=lambda g: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", g)])
    custom = read_names(args.names) if args.names else {}
    label = {g: custom.get(g, g if args.raw_names else auto_name(g)) for g in groups}
    unused = sorted(set(custom) - set(groups))
    if unused:
        print(f"names file: no group called {', '.join(unused)} (check the spelling)")
    no_e = sum(r["E"] is None for r in recs)
    no_f = sum(r["fmax"] is None for r in recs)
    print(f"{len(recs)} frames read, {len(groups)} groups")
    print(f"frames with missing energy: {no_e}")
    print(f"frames with missing forces: {no_f}")

    # ------------------------------------------------------------------ energies
    erecs = [r for r in recs if r["E"] is not None]
    if not erecs:
        raise SystemExit("no energies found - check the energy key in the comment lines")
    epa = np.array([r["E"] / r["n"] for r in erecs])
    nat = np.array([r["n"] for r in erecs])
    grp = np.array([r["group"] for r in erecs])
    bulk = nat > 2

    def estats(x):
        return (f"mean={x.mean():9.3f}  std={x.std():7.3f}  "
                f"min={x.min():9.3f}  max={x.max():9.3f}  eV/atom")
    w = max(22, max(len(g) for g in groups))      # column width for the group names
    if any(label[g] != g for g in groups):
        print("\ngroup labels used in the figure:")
        for g in groups:
            print(f"  {g:<{w}} -> {label[g]}")
    print("\n=== energy per atom ===")
    print(f"  {'all frames':<{w}} n={len(epa):6d}  {estats(epa)}")
    if bulk.any() and (~bulk).any():
        print(f"  {'> 2 atoms':<{w}} n={bulk.sum():6d}  {estats(epa[bulk])}")
    for g in groups:
        m = grp == g
        if m.any():
            print(f"  {g:<{w}} n={m.sum():6d}  {estats(epa[m])}")

    # outliers: far from the group median, judged with a robust spread (MAD).
    # 1-2 atom frames are skipped - a dimer curve is supposed to spread widely.
    out = np.zeros(len(erecs), bool)
    for g in groups:
        m = (grp == g) & bulk
        if m.sum() < 5:
            continue
        med = np.median(epa[m])
        mad = 1.4826 * np.median(np.abs(epa[m] - med))
        out |= m & (np.abs(epa - med) > max(6 * mad, 0.15))
    if args.outliers:
        print("\nframes far from the rest of their group (|E/atom - median| > max(6 MAD, 0.15 eV)):")
        if not out.any():
            print("  none")
        for i in np.flatnonzero(out)[np.argsort(-epa[out])]:
            r = erecs[i]
            med = np.median(epa[(grp == r["group"]) & bulk])
            print(f"  {epa[i]:9.3f} eV/atom (group median {med:8.3f})  n={r['n']:4d}  {where(r)}")
    else:
        out[:] = False

    print(f"\nhighest E/atom (top {args.top}):")
    for i in np.argsort(-epa)[:args.top]:
        print(f"  {epa[i]:9.3f} eV/atom  n={nat[i]:4d}  {where(erecs[i])}")

    # ------------------------------------------------------------------ forces
    frecs = sorted([r for r in recs if r["fmax"] is not None], key=lambda r: -r["fmax"])
    if mags:
        mags = np.concatenate(mags)
        elems = np.array(elems)

        def fstats(x):
            return (f"n={len(x):8d}  mean={x.mean():6.2f}  median={np.median(x):6.2f}  "
                    f"p99={np.percentile(x, 99):6.2f}  max={x.max():7.2f}  eV/A")
        print("\n=== |F| per atom ===")
        print(f"  {'all':<4}{fstats(mags)}")
        for el in sorted(set(elems)):
            print(f"  {el:<4}{fstats(mags[elems == el])}")
        print(f"\nframes with the largest single-atom force (top {args.top}):")
        for r in frecs[:args.top]:
            print(f"  {r['fmax']:9.2f} eV/A on {r['fmax_el']:<3} n={r['n']:4d}  {where(r)}")
        if args.frame_stat != "max":
            key = "f" + args.frame_stat
            name = {"rms": "RMS", "mean": "mean"}[args.frame_stat]
            print(f"\nframes with the largest {name} |F| per frame (top {args.top}):")
            for r in sorted(frecs, key=lambda r: -r[key])[:args.top]:
                print(f"  {r[key]:9.2f} eV/A  (largest {r['fmax']:7.2f})  n={r['n']:4d}  {where(r)}")
    else:
        mags, elems = np.array([]), np.array([])

    plot(groups, epa, nat, grp, out, mags, elems, frecs, len(recs), args.out, args.outliers,
         args.frame_stat, label)
    print(f"\nplot -> {args.out}")


# ====================================================================== plot
INK, INK2, MUTED, GRID, SURF, BAND = "#1f1f1d", "#4a4a46", "#8a8984", "#e9e8e3", "#ffffff", "#f5f4f0"
ECOL, ETXT = "#1baf7a", "#0d7a54"      # energy: marks / axis text
FCOL, FTXT = "#2a78d6", "#1d5aa6"      # force:  marks / axis text
HOT = "#d1495b"                        # flagged frames


def plot(groups, epa, nat, grp, out, mags, elems, frecs, nframes, path_out, show_out=False,
         frame_stat="max", label=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.scale import SymmetricalLogTransform
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator

    plt.rcParams.update({
        "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    })
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    blank = FuncFormatter(lambda v, _: "")
    ng = len(groups)
    row = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(0)

    # ---------------- helpers ----------------------------------------------------------
    def log_ticks(axis, lo, hi):
        dec = np.log10(hi / lo)
        subs = (1,) if dec > 4 else (1, 2, 5) if dec > 1.5 else (1, 1.5, 2, 3, 5, 7)
        axis.set_major_locator(LogLocator(base=10, subs=subs, numticks=30))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(blank)

    def color_axis(ax, which, col, txt):
        """colour one axis (x-bottom, x-top, y-left, y-right) by quantity"""
        side = {"bottom": "x", "top": "x", "left": "y", "right": "y"}[which]
        ax.spines[which].set_visible(True)
        ax.spines[which].set_color(col)
        ax.spines[which].set_linewidth(1.1)
        ax.tick_params(axis=side, colors=col, labelcolor=txt)
        (ax.xaxis if side == "x" else ax.yaxis).label.set_color(txt)

    def top_axis(ax):
        """a second x axis on top (shares y)"""
        t = ax.twiny()
        for sp in ("left", "right", "bottom"):
            t.spines[sp].set_visible(False)
        return t

    # energy axis: linear, or symlog when dimer walls reach far up
    emin, emax = epa.min(), epa.max()
    e_symlog = emax > 10 or emin < -20
    e_ticks = [t for t in (-50, -30, -20, -15, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10,
                           30, 100, 300, 1000, 3000) if emin - 2 <= t <= emax * 1.5 + 2]
    e_label = "Energy per atom (eV/atom)" + ("  ·  linear within ±10, log beyond" if e_symlog else "")

    def energy_axis(ax):
        if e_symlog:
            ax.set_xscale("symlog", linthresh=10, linscale=1.6)
            ax.xaxis.set_major_locator(FixedLocator(e_ticks))
            ax.xaxis.set_major_formatter(plain)
            ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlabel(e_label)

    # force per frame
    have_f = bool(frecs) and len(mags) > 0
    fname = {"max": "Largest |F|", "rms": "RMS |F|", "mean": "Mean |F|"}[frame_stat]
    if have_f:
        fmx = np.array([r["f" + frame_stat] for r in frecs])
        frow = np.array([row[r["group"]] for r in frecs])
        pos = fmx[fmx > 0]
        flo = max(pos.min() if len(pos) else 1e-3, 1e-3) * 0.8
        fhi = fmx.max() * 1.3
        p99 = np.percentile(mags, 99)
        # red threshold: largest |F| -> 2 x p99 of all per-atom |F|;
        #                RMS / mean  -> 2 x p99 of the per-frame values themselves
        if frame_stat == "max":
            thr, thr_txt = 2 * p99, f"above 2 × p99 of per-atom |F| = {2 * p99:.1f} eV/Å"
        else:
            q = np.percentile(fmx, 99)
            thr, thr_txt = 2 * q, f"above 2 × p99 of per-frame {fname} = {2 * q:.1f} eV/Å"

    # ---------------- layout -------------------------------------------------------------
    def short(g, n=30):
        return g if len(g) <= n else g[:n - 1] + "…"
    label = label or {g: g for g in groups}
    ylabels = [f"{short(label[g], 48)}  ({int(np.sum(grp == g))})" for g in groups]
    W = 15.0
    L = (0.3 + 0.072 * max(len(t) for t in ylabels)) / W          # name column
    h_ax = max(4.2, 0.42 * ng + 0.4)                                 # two strips per group
    # bottom margin: x label + the legends below both panels (left: 2 columns, right: 1)
    n_left = 3 + (2 if have_f else 0) + (1 if show_out else 0)
    n_right = (1 if (nat > 2).any() else 0) + (1 if (nat <= 2).any() else 0) + len(set(elems)) + 1
    rows = max(-(-n_left // 2), n_right)
    top, bot = 2.35, 1.15 + 0.2 * rows
    H = top + h_ax + bot
    fig = plt.figure(figsize=(W, H), dpi=200, facecolor=SURF)
    y0, hr = bot / H, h_ax / H
    axG = fig.add_axes([L, y0, 0.545 - L, hr])
    axD = fig.add_axes([0.61, y0, 0.335, hr])

    # ======================================================================================
    # left: by group - energy strip (upper, bottom axis) + force strip (lower, top axis)
    # ======================================================================================
    dE, dF = -0.2, 0.2                    # sub-row offsets (y axis is inverted)
    for i in range(0, ng, 2):
        axG.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
    for i in range(1, ng):
        axG.axhline(i - 0.5, color=GRID, lw=0.6, zorder=0)

    def median_tick(ax, x, y):
        ax.plot([x, x], [y - 0.15, y + 0.15], color=INK, lw=1.6, zorder=5, solid_capstyle="butt")

    for g in groups:
        m = grp == g
        if not m.any():
            continue
        y = row[g] + dE + rng.uniform(-0.12, 0.12, m.sum())
        normal, flag = ~out[m], out[m]
        axG.scatter(epa[m][normal], y[normal], s=7, color=ECOL, alpha=0.6, lw=0, zorder=3)
        if flag.any():
            axG.scatter(epa[m][flag], y[flag], s=16, color=HOT, lw=0, zorder=4)
        median_tick(axG, np.median(epa[m]), row[g] + dE)
    energy_axis(axG)
    axG.set_yticks(range(ng))
    axG.set_yticklabels(ylabels, fontsize=9)
    if show_out:
        for lab, g in zip(axG.get_yticklabels(), groups):
            if np.any(out & (grp == g)):
                lab.set_color(HOT)
    axG.tick_params(axis="y", length=0)
    axG.set_ylim(ng - 0.5, -0.5)                       # first group on top
    axG.spines["left"].set_visible(False)
    color_axis(axG, "bottom", ECOL, ETXT)

    axGF = top_axis(axG)
    below = np.zeros(0, bool)
    if have_f:
        below = fmx < flo / 0.8
        hot = fmx > thr
        y = frow + dF + rng.uniform(-0.12, 0.12, len(fmx))
        axGF.scatter(fmx[~hot & ~below], y[~hot & ~below], s=7, color=FCOL, alpha=0.6, lw=0, zorder=3)
        if below.any():
            axGF.scatter(np.full(below.sum(), flo), y[below], s=12, marker="<", color=MUTED,
                         lw=0, zorder=3)
        if hot.any():
            axGF.scatter(fmx[hot], y[hot], s=16, color=HOT, lw=0, zorder=4)
        for g in groups:
            mm = frow == row[g]
            if mm.any():
                median_tick(axGF, np.median(fmx[mm]), row[g] + dF)
        axGF.set_xscale("log")
        axGF.set_xlim(flo / 1.15, fhi)
        log_ticks(axGF.xaxis, flo, fhi)
        axGF.axvline(thr, color=HOT, lw=0.9, ls="--", zorder=2)
    axGF.set_ylim(ng - 0.5, -0.5)
    axGF.set_xlabel(f"{fname} in the frame (eV/Å)")
    color_axis(axGF, "top", FCOL, FTXT)

    key = [Line2D([], [], ls="", marker="o", ms=5, color=ECOL, label="energy per atom (upper strip)"),
           Line2D([], [], ls="", marker="o", ms=5, color=FCOL, label=f"{fname} per frame (lower strip)"),
           Line2D([], [], color=INK, lw=1.6, label="group median")]
    if have_f:
        key.append(Line2D([], [], color=HOT, lw=0.9, ls="--", marker="o", ms=4,
                          label=f"force {thr_txt}"))
        if below.any():
            key.append(Line2D([], [], ls="", marker="<", ms=5, color=MUTED,
                              label=f"{fname} below {flo / 0.8:g} eV/Å ({below.sum()})"))
    if show_out:
        key.append(Line2D([], [], ls="", marker="o", ms=5, color=HOT,
                          label="energy outlier (group name in red)"))
    axG.legend(handles=key, loc="upper left", bbox_to_anchor=(0.0, -0.75 / h_ax), ncol=2,
               frameon=False, fontsize=8.8, handletextpad=0.4, columnspacing=1.6,
               borderaxespad=0, labelspacing=0.4)
    axG.set_title("By group", loc="left", fontsize=12, color=INK, pad=44)

    # ======================================================================================
    # right: per-atom distributions - energy (bottom/left axes) + force (top/right axes)
    # ======================================================================================
    def step_hist(ax, x, bins, col, ls):
        h, e = np.histogram(x, bins=bins)
        frac = h / max(len(x), 1) * 100
        if ls == "-":
            ax.fill_between(e[:-1], frac, step="post", color=col, alpha=0.08, lw=0)
        ax.step(e[:-1], frac, where="post", color=col, lw=1.9, ls=ls)
        return frac.max() if len(frac) else 0

    span = max(emax - emin, 1e-6)
    if e_symlog:
        T = SymmetricalLogTransform(10, 10, 1.6)
        tlo, thi = T.transform(np.array([emin, emax]))
        ebins = T.inverted().transform(np.linspace(tlo, thi + 1e-9, 91))
    else:
        ebins = np.linspace(emin - 0.02 * span, emax + 0.02 * span, 91)
    handles, hmax = [], 0
    for m, ls, name in [(nat > 2, "-", "> 2 atoms"), (nat <= 2, "--", "dimers + atoms")]:
        if not m.any():
            continue
        hmax = max(hmax, step_hist(axD, epa[m], ebins, ECOL, ls))
        x = epa[m]
        handles.append(Line2D([], [], color=ECOL, lw=2.2, ls=ls, label=(
            f"E/atom, {name} ({m.sum():,} frames): median {np.median(x):.2f}, "
            f"range {x.min():.2f} to {x.max():.2f}")))
    energy_axis(axD)
    axD.set_xlim(ebins[0], ebins[-1])
    axD.set_ylim(0, hmax * 1.12)
    axD.set_ylabel("Share of frames per bin (%)")
    color_axis(axD, "bottom", ECOL, ETXT)
    color_axis(axD, "left", ECOL, ETXT)

    axDF = axD.figure.add_axes(axD.get_position(), label="force-dist")
    axDF.patch.set_visible(False)
    axDF.xaxis.tick_top(); axDF.xaxis.set_label_position("top")
    axDF.yaxis.tick_right(); axDF.yaxis.set_label_position("right")
    for sp in ("left", "bottom"):
        axDF.spines[sp].set_visible(False)
    if len(mags):
        els = sorted(set(elems))
        styles = ["-", "--", ":", "-."]
        lo = max(np.percentile(mags[mags > 0], 0.1), 1e-3) if np.any(mags > 0) else 1e-3
        bins = np.logspace(np.log10(lo), np.log10(mags.max() * 1.05), 70)
        fmax_h = 0
        for i, el in enumerate(els):
            xe = mags[elems == el]
            fmax_h = max(fmax_h, step_hist(axDF, np.clip(xe, lo, None), bins, FCOL, styles[i % 4]))
            handles.append(Line2D([], [], color=FCOL, lw=2.2, ls=styles[i % 4], label=(
                f"|F|, {el} ({len(xe):,} atoms): mean {xe.mean():.2f}, p99 "
                f"{np.percentile(xe, 99):.2f}, max {xe.max():.1f}")))
        axDF.set_xscale("log")
        log_ticks(axDF.xaxis, bins[0], bins[-1])
        axDF.set_xlim(bins[0], bins[-1])
        axDF.set_ylim(0, fmax_h * 1.12)
        n_low = int(np.sum(mags < lo))
        if n_low > 0.005 * len(mags):
            handles.append(Line2D([], [], ls="", label=(
                f"   first force bin also holds |F| < {lo:.3g} eV/Å "
                f"({100 * n_low / len(mags):.1f}% of atoms)")))
    axDF.set_xlabel("|F| per atom (eV/Å)")
    axDF.set_ylabel("Share of atoms per bin (%)", rotation=270, labelpad=14)
    color_axis(axDF, "top", FCOL, FTXT)
    color_axis(axDF, "right", FCOL, FTXT)
    axD.grid(True, axis="y", color=GRID, lw=0.8); axD.set_axisbelow(True)
    axD.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.75 / h_ax), ncol=1,
               frameon=False, fontsize=8.8, handlelength=2.2, borderaxespad=0, labelspacing=0.4)
    axD.set_title("Per-atom distributions", loc="left", fontsize=12, color=INK, pad=44)

    # ---------------- header ---------------------------------------------------------------
    fig.text(0.02, 1 - 0.3 / H, "Training-set check: energies and forces", fontsize=16,
             fontweight="bold", color=INK, va="top")
    sub = f"{nframes:,} frames  ·  {ng} groups"
    if show_out:
        sub += f"  ·  {int(out.sum())} energy outlier frame(s)"
    if len(mags):
        sub += (f"  ·  {len(mags):,} atoms, median |F| {np.median(mags):.2f} eV/Å, "
                f"99% below {np.percentile(mags, 99):.2f} eV/Å")
    fig.text(0.02, 1 - 0.7 / H, sub, fontsize=10, color=INK2, va="top")
    fig.text(0.02, 1 - 1.02 / H, "green = energy (bottom / left axes)   ·   "
             "blue = force (top / right axes)", fontsize=10, color=INK2, va="top")
    fig.savefig(path_out, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main()