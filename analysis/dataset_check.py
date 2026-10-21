#!/usr/bin/env python3
"""
dataset_check.py - energy and force check of NEP training files, with one summary figure.

    python dataset_check.py train.xyz
    python dataset_check.py train.xyz test.xyz --top 15 --out check.png

Prints
  - frames with a missing energy or missing forces,
  - E/atom (mean, std, min, max) overall, for frames with > 2 atoms, and per data group
    (group = first part of Config_type, e.g. "B-B" from "B-B/str-b2-dimer-012"),
  - with --outliers: frames whose E/atom is far from the rest of their group,
  - the frames with the highest E/atom,
  - |F| per atom (mean, median, p99, max) overall and per element,
  - the frames with the largest single-atom force.
Plots (one figure, 2 x 2)
  top left    : E/atom of every frame, one row per group (median marked)
  top right   : E/atom distribution, frames with > 2 atoms and dimers/atoms separately,
                min / median / max marked
  bottom left : |F| per atom distribution per element, log |F| axis, mean / p99 / max marked
  bottom right: largest |F| of every frame, one row per group (red = above 2 x p99)

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


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--top", type=int, default=10, help="list this many frames in the top-N tables")
    ap.add_argument("--out", default="dataset_check.png")
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
            else:
                fr["fmax"], fr["fmax_el"] = None, None
            fr["F"] = None        # keep memory low for big files
            recs.append(fr)
    if not recs:
        raise SystemExit("no frames found")

    groups = sorted({r["group"] for r in recs},
                    key=lambda g: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", g)])
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
    print("\n=== energy per atom ===")
    print(f"  all frames             n={len(epa):6d}  {estats(epa)}")
    if bulk.any() and (~bulk).any():
        print(f"  > 2 atoms              n={bulk.sum():6d}  {estats(epa[bulk])}")
    for g in groups:
        m = grp == g
        if m.any():
            print(f"  {g:<22} n={m.sum():6d}  {estats(epa[m])}")

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
    else:
        mags, elems = np.array([]), np.array([])

    plot(groups, epa, nat, grp, out, mags, elems, frecs, len(recs), args.out, args.outliers)
    print(f"\nplot -> {args.out}")


# ====================================================================== plot
def plot(groups, epa, nat, grp, out, mags, elems, frecs, nframes, path_out, show_out=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.scale import SymmetricalLogTransform
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator

    INK, INK2, MUTED, GRID, SURF, BAND = "#1f1f1d", "#4a4a46", "#8a8984", "#e9e8e3", "#ffffff", "#f5f4f0"
    DOT, HOT = "#3d6fb6", "#d1495b"                  # strip points / flagged points
    ELCOL = {"B": "#2a78d6", "N": "#eb6834"}
    CATCOL = {"big": "#1baf7a", "small": "#8b5cf6"}  # frames with > 2 atoms / dimers + atoms
    OTHER = ["#1baf7a", "#eda100", "#8b5cf6", "#0ea5b7"]
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

    # ---------------- layout ---------------------------------------------------------------
    #   row 1: [names | E/atom by group]   [E/atom distribution]
    #   row 2: [|F| distribution]          [names | largest |F| by group]
    def short(g, n=30):
        return g if len(g) <= n else g[:n - 1] + "…"
    ylabels = [f"{short(g)}  ({int(np.sum(grp == g))})" for g in groups]
    W = 14.0
    L = (0.3 + 0.072 * max(len(t) for t in ylabels)) / W     # width of the name column
    h_row = max(4.0, 0.27 * ng + 0.6)
    top, gap, bot = 1.75, 1.55, 0.8
    H = top + 2 * h_row + gap + bot
    fig = plt.figure(figsize=(W, H), dpi=200, facecolor=SURF)
    y1, y2, hr = (bot + h_row + gap) / H, bot / H, h_row / H
    axE = fig.add_axes([L, y1, 0.545 - L, hr])
    axEH = fig.add_axes([0.615, y1, 0.37, hr])
    axF = fig.add_axes([0.065, y2, 0.385, hr])
    axM = fig.add_axes([0.46 + L, y2, 0.985 - 0.46 - L, hr])

    def title(ax, text, pad=10):
        ax.set_title(text, loc="left", fontsize=11.5, color=INK, pad=pad)

    def log_ticks(axis, lo, hi):
        dec = np.log10(hi / lo)
        subs = (1,) if dec > 4 else (1, 2, 5) if dec > 1.5 else (1, 1.5, 2, 3, 5, 7)
        axis.set_major_locator(LogLocator(base=10, subs=subs, numticks=30))
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(blank)

    def group_rows(ax):
        for i in range(0, ng, 2):
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, lw=0, zorder=0)
        ax.set_yticks(range(ng))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(ng - 0.5, -0.5)                     # first group on top
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

    def median_tick(ax, x, r):
        ax.plot([x, x], [r - 0.38, r + 0.38], color=INK, lw=1.6, zorder=5, solid_capstyle="butt")

    def markers(ax, items, lo, hi, log=False):
        """vertical lines with labels staggered in height so they never collide"""
        for (lab, val, ls), yy in zip(items, (0.985, 0.915, 0.845)):
            ax.axvline(val, color=INK, lw=0.9, ls=ls, alpha=0.75, zorder=1)
            frac = (np.log(val / lo) / np.log(hi / lo)) if log else None
            if frac is None:
                tr = ax.transData + ax.transAxes.inverted()
                frac = tr.transform((val, 0))[0]
            right = frac > 0.75
            ax.annotate(f"{lab} {val:.2f}", xy=(val, yy), xycoords=("data", "axes fraction"),
                        xytext=(-4 if right else 4, 0), textcoords="offset points",
                        ha="right" if right else "left", va="top", fontsize=8.5, color=INK2,
                        bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none", alpha=0.9))

    def legend(ax, handles):
        ax.legend(handles=handles, loc="upper left", frameon=True, facecolor=SURF,
                  edgecolor="none", framealpha=1.0, fontsize=9, labelspacing=0.9,
                  handlelength=1.6)

    def step_hist(ax, x, bins, col):
        h, e = np.histogram(x, bins=bins)
        frac = h / max(len(x), 1) * 100
        ax.fill_between(e[:-1], frac, step="post", color=col, alpha=0.07, lw=0)
        ax.step(e[:-1], frac, where="post", color=col, lw=1.9)
        return frac.max() if len(frac) else 0

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

    # ---------------- top left: E/atom of every frame, by group --------------------------
    group_rows(axE)
    for g in groups:
        m = grp == g
        if not m.any():
            continue
        y = row[g] + rng.uniform(-0.28, 0.28, m.sum())
        normal, flag = ~out[m], out[m]
        axE.scatter(epa[m][normal], y[normal], s=7, color=DOT, alpha=0.55, lw=0, zorder=3)
        if flag.any():
            axE.scatter(epa[m][flag], y[flag], s=16, color=HOT, lw=0, zorder=4)
        median_tick(axE, np.median(epa[m]), row[g])
    if show_out:
        for lab, g in zip(axE.get_yticklabels(), groups):
            if np.any(out & (grp == g)):
                lab.set_color(HOT)
    energy_axis(axE)
    leg = [Line2D([], [], ls="", marker="o", ms=4, color=DOT, label="frame")]
    if show_out:
        leg.append(Line2D([], [], ls="", marker="o", ms=5, color=HOT, label="outlier (group name in red)"))
    leg.append(Line2D([], [], color=INK, lw=1.6, label="group median"))
    axE.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, 1.008), ncol=3, frameon=False,
               fontsize=8.8, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)

    # ---------------- top right: E/atom distribution ------------------------------------
    span = max(emax - emin, 1e-6)
    if e_symlog:
        T = SymmetricalLogTransform(10, 10, 1.6)
        tlo, thi = T.transform(np.array([emin, emax]))
        ebins = T.inverted().transform(np.linspace(tlo, thi + 1e-9, 91))
    else:
        ebins = np.linspace(emin - 0.02 * span, emax + 0.02 * span, 91)
    cats = [("big", nat > 2, "more than 2 atoms"), ("small", nat <= 2, "dimers and atoms")]
    hmax, handles = 0, []
    for key, m, name in cats:
        if not m.any():
            continue
        hmax = max(hmax, step_hist(axEH, epa[m], ebins, CATCOL[key]))
        x = epa[m]
        handles.append(Line2D([], [], color=CATCOL[key], lw=2.4, label=(
            f"{name} ({m.sum():,} frames):  median {np.median(x):.2f},  "
            f"range {x.min():.2f} to {x.max():.2f}")))
    energy_axis(axEH)
    axEH.set_xlim(ebins[0], ebins[-1])
    axEH.set_ylim(0, hmax * 1.55)
    markers(axEH, [("min", emin, ":"), ("median", float(np.median(epa)), "-"), ("max", emax, ":")],
            ebins[0], ebins[-1])
    # legend above the axes so it never hides the min / median / max labels
    axEH.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.008), frameon=False,
                fontsize=8.8, handlelength=1.6, borderaxespad=0, labelspacing=0.3)
    axEH.set_ylabel("Share of frames per bin (%)")
    axEH.grid(True, axis="y", color=GRID, lw=0.8); axEH.set_axisbelow(True)
    e_pad = max(24, 12 + 15 * len(handles))        # same title height across the row
    title(axEH, "Energy per atom, distribution", pad=e_pad)
    title(axE, "Energy per atom, every frame by group", pad=e_pad)

    # ---------------- bottom left: |F| per atom per element ------------------------------
    if len(mags):
        els = sorted(set(elems))
        color = {el: ELCOL.get(el, OTHER[i % len(OTHER)]) for i, el in enumerate(els)}
        lo = max(np.percentile(mags[mags > 0], 0.1), 1e-3) if np.any(mags > 0) else 1e-3
        bins = np.logspace(np.log10(lo), np.log10(mags.max() * 1.05), 70)
        fmax_h = 0
        for el in els:
            fmax_h = max(fmax_h, step_hist(axF, np.clip(mags[elems == el], lo, None), bins, color[el]))
        axF.set_xscale("log")
        log_ticks(axF.xaxis, bins[0], bins[-1])
        axF.set_xlim(bins[0], bins[-1])
        axF.set_ylim(0, fmax_h * 1.55)
        n_low = int(np.sum(mags < lo))
        if n_low > 0.005 * len(mags):          # only worth a note when it is visible
            first = max(100 * np.sum((elems == el) & (mags < bins[1])) / max(np.sum(elems == el), 1)
                        for el in els)
            axF.annotate(f"◂ first bin also holds every |F| < {lo:.3g} eV/Å\n   "
                         f"({100 * n_low / len(mags):.1f}% of atoms, e.g. relaxed structures)",
                         xy=(bins[1], first), xytext=(6, 4), textcoords="offset points",
                         fontsize=8.3, color=MUTED, va="bottom")
        markers(axF, [("mean", mags.mean(), "-"), ("p99", np.percentile(mags, 99), "--"),
                      ("max", mags.max(), ":")], bins[0], bins[-1], log=True)
        handles = []
        for el in els:
            xe = mags[elems == el]
            handles.append(Line2D([], [], color=color[el], lw=2.4, label=(
                f"{el}   {len(xe):,} atoms\n      mean {xe.mean():.2f} · p99 "
                f"{np.percentile(xe, 99):.2f} · max {xe.max():.1f}")))
        legend(axF, handles)
    axF.set_xlabel("|F| per atom (eV/Å)")
    axF.set_ylabel("Share of atoms per bin (%)")
    axF.grid(True, axis="y", color=GRID, lw=0.8); axF.set_axisbelow(True)

    # ---------------- bottom right: largest |F| of every frame, by group -----------------
    group_rows(axM)
    if frecs:
        fmx = np.array([r["fmax"] for r in frecs])
        frow = np.array([row[r["group"]] for r in frecs])
        pos = fmx[fmx > 0]
        flo = max(pos.min() if len(pos) else 1e-3, 1e-3) * 0.8
        fhi = fmx.max() * 1.3
        below = fmx < flo / 0.8
        y = frow + rng.uniform(-0.28, 0.28, len(fmx))
        p99 = np.percentile(mags, 99) if len(mags) else np.inf
        hot = fmx > 2 * p99
        axM.scatter(fmx[~hot & ~below], y[~hot & ~below], s=7, color=DOT, alpha=0.55, lw=0, zorder=3)
        if below.any():
            axM.scatter(np.full(below.sum(), flo), y[below], s=12, marker="<", color=MUTED,
                        lw=0, zorder=3)
        if hot.any():
            axM.scatter(fmx[hot], y[hot], s=16, color=HOT, lw=0, zorder=4)
        for g in groups:
            m = frow == row[g]
            if m.any():
                median_tick(axM, np.median(fmx[m]), row[g])
        axM.set_xscale("log")
        axM.set_xlim(flo / 1.15, fhi)
        log_ticks(axM.xaxis, flo, fhi)
        if np.isfinite(p99):
            axM.axvline(2 * p99, color=HOT, lw=0.9, ls="--", zorder=2)
        note = f"red: above 2 × p99 = {2 * p99:.1f} eV/Å  ·  black tick: group median"
        if below.any():
            note += f"\n◂ {below.sum()} frame(s) with largest |F| below {flo / 0.8:g} eV/Å"
        axM.text(0.0, 1.018, note, transform=axM.transAxes, fontsize=8.8, color=INK2,
                 va="bottom", ha="left")
    axM.set_xlabel("Largest |F| in the frame (eV/Å)")
    f_pad = 24 if not (frecs and below.any()) else 36
    title(axM, "Largest force per frame, by group", pad=f_pad)
    title(axF, "Force per atom, distribution by element", pad=f_pad)

    # ---------------- figure titles ------------------------------------------------------
    fig.text(0.065, 1 - 0.35 / H, "Training-set check: energies and forces", fontsize=16,
             fontweight="bold", color=INK, va="top")
    sub = f"{nframes:,} frames  ·  {ng} groups"
    if show_out:
        sub += f"  ·  {int(out.sum())} energy outlier frame(s)"
    if len(mags):
        sub += (f"  ·  {len(mags):,} atoms, median |F| {np.median(mags):.2f} eV/Å, "
                f"99% below {np.percentile(mags, 99):.2f} eV/Å")
    fig.text(0.065, 1 - 0.75 / H, sub, fontsize=10, color=INK2, va="top")
    fig.savefig(path_out, facecolor=SURF)


if __name__ == "__main__":
    main()