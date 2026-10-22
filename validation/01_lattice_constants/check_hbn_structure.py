#!/usr/bin/env python3
"""
check_hbn_structure.py - verify the honeycomb basis and the stacking of hBN structures.

    python check_hbn_structure.py model.xyz
    python check_hbn_structure.py lattice_check/nep_best/relax/bulk/relaxed.xyz
    python check_hbn_structure.py POSCAR CONTCAR some.xyz ...

Reads extended XYZ (GPUMD model.xyz / relaxed.xyz) or POSCAR/CONTCAR. Layers are assumed to
lie in the xy plane (z = stacking direction).

1. In-plane shells (atoms of the same layer), for every atom:
     1st shell: 3 neighbours, all of the other element, at d1 = a/sqrt(3)  (1.446 A for a = 2.504)
     2nd shell: 6 neighbours, same element, at a                          (2.504 A)
     3rd shell: 3 neighbours, other element, at 2a/sqrt(3)                 (2.891 A)
   a is derived as sqrt(3) * d1 and the shells are checked against it.
2. Layers: number of layers (atoms grouped by height) and the spacing between neighbours.
3. Stacking of each pair of neighbouring layers (including the periodic pair in bulk):
   for every atom, the closest atom of the next layer in the xy projection:
     eclipsed (in-plane offset ~0) and other element  -> AA'  (B over N; bulk hBN)
     eclipsed and same element                        -> AA   (B over B)
     half eclipsed, half over a hexagon centre        -> AB-type (Bernal-like)
   Anything else is reported as 'other / shifted' with the mean offset.
"""
import sys
import numpy as np

SQ3 = np.sqrt(3.0)
TOL = 0.05          # A, tolerance for distances and offsets


def read_structure(path):
    lines = open(path).read().splitlines()
    if "lattice=" in lines[1].lower():                          # extended XYZ
        n = int(lines[0].split()[0])
        head = lines[1]
        i = head.lower().index("lattice=")
        lat = np.array(head[i + 8:].split('"')[1].split(), float).reshape(3, 3)
        props = "species:S:1:pos:R:3"
        if "properties=" in head.lower():
            j = head.lower().index("properties=")
            props = head[j + 11:].split()[0]
        f = props.split(":")
        col, i_sp, i_pos = 0, 0, 1
        for name, typ, cnt in zip(f[0::3], f[1::3], f[2::3]):
            if name.lower() == "species":
                i_sp = col
            if name.lower() == "pos":
                i_pos = col
            col += int(cnt)
        # last frame if there are several
        k, last = 0, None
        while k < len(lines):
            if not lines[k].strip():
                k += 1
                continue
            m = int(lines[k].split()[0])
            last = (k, m)
            k += m + 2
        k, n = last
        head = lines[k + 1]
        i = head.lower().index("lattice=")
        lat = np.array(head[i + 8:].split('"')[1].split(), float).reshape(3, 3)
        rows = [lines[k + 2 + q].split() for q in range(n)]
        return np.array([r[i_sp] for r in rows]), np.array([r[i_pos:i_pos + 3] for r in rows], float), lat
    scale = float(lines[1].split()[0])                          # POSCAR
    lat = np.array([l.split()[:3] for l in lines[2:5]], float) * scale
    names, counts = lines[5].split(), [int(x) for x in lines[6].split()]
    i = 7
    if lines[i].strip().lower().startswith("s"):
        i += 1
    cart = lines[i].strip().lower()[0] in "ck"
    xyz = np.array([l.split()[:3] for l in lines[i + 1:i + 1 + sum(counts)]], float)
    pos = xyz * scale if cart else xyz @ lat
    return np.array(sum([[e] * c for e, c in zip(names, counts)], [])), pos, lat


def mic(d, lat, inv):
    return d - np.round(d @ inv) @ lat


def layers_of(pos, lat):
    """group atoms by height (gap > 1 A); returns list of index arrays sorted by z, and periodic flag"""
    z = pos[:, 2]
    order = np.argsort(z)
    zs = z[order]
    cuts = np.where(np.diff(zs) > 1.0)[0]
    groups = np.split(order, cuts + 1)
    zmean = [z[g].mean() for g in groups]
    # bulk if the gap through the periodic boundary is like the others (no vacuum)
    Lz = lat[2, 2]
    wrap = zmean[0] + Lz - zmean[-1]
    gaps = np.diff(zmean)
    periodic = len(groups) > 1 and wrap < 1.5 * max(gaps.max(), 1e-9) and wrap < 6.0
    if len(groups) == 1:
        periodic = Lz < 6.0
    return groups, np.array(zmean), periodic, wrap


def check(path):
    sp, pos, lat = read_structure(path)
    inv = np.linalg.inv(lat)
    n = len(sp)
    print(f"\n=== {path} ===")
    print(f"{n} atoms: " + ", ".join(f"{e} {np.sum(sp == e)}" for e in sorted(set(sp))) +
          f";  cell lengths {np.linalg.norm(lat, axis=1).round(4)} A")
    ok = True

    groups, zmean, periodic, wrap = layers_of(pos, lat)
    layer_id = np.empty(n, int)
    for k, g in enumerate(groups):
        layer_id[g] = k

    # ---------------- 1. in-plane shells
    d1_all, problems = [], []
    shell_stats = {1: [], 2: [], 3: []}
    hetero_ok = True
    for i in range(n):
        same = np.where(layer_id == layer_id[i])[0]
        same = same[same != i]
        d = mic(pos[same] - pos[i], lat, inv)
        r = np.linalg.norm(d, axis=1)
        o = np.argsort(r)
        r, nb = r[o], same[o]
        d1 = r[0]
        d1_all.append(d1)
        # shell 1: within TOL of d1
        s1 = nb[np.abs(r - d1) < TOL]
        if len(s1) != 3:
            problems.append(f"atom {i} ({sp[i]}): {len(s1)} nearest neighbours instead of 3")
        if np.any(sp[s1] == sp[i]):
            hetero_ok = False
            problems.append(f"atom {i} ({sp[i]}): a nearest neighbour of the same element")
        shell_stats[1].append((len(s1), d1))
        a_i = SQ3 * d1
        for sh, (target, want_n, same_el) in {2: (a_i, 6, True), 3: (2 * a_i / SQ3, 3, False)}.items():
            m = np.abs(r - target) < 2 * TOL
            cnt = int(m.sum())
            elems_ok = np.all((sp[nb[m]] == sp[i]) == same_el)
            shell_stats[sh].append((cnt, r[m].mean() if cnt else np.nan))
            if cnt != want_n or not elems_ok:
                problems.append(f"atom {i} ({sp[i]}): shell {sh} has {cnt} atoms"
                                f"{'' if elems_ok else ' with wrong elements'} (expected {want_n})")
    d1_all = np.array(d1_all)
    a = SQ3 * d1_all.mean()
    print("\n1. In-plane shells (same layer)")
    for sh, name, exp in ((1, "B-N, 3 neighbours", d1_all.mean()), (2, "same element, 6", a),
                          (3, "B-N, 3 neighbours", 2 * a / SQ3)):
        cnts = np.array([c for c, _ in shell_stats[sh]])
        dists = np.array([x for _, x in shell_stats[sh]])
        print(f"   shell {sh} ({name:<18}): distance {np.nanmean(dists):.4f} +- {np.nanstd(dists):.4f} A "
              f"(ideal {exp:.4f}), count per atom {cnts.min()}-{cnts.max()}")
    print(f"   -> a = sqrt(3) x d(B-N) = {a:.4f} A;  B-N bond spread {d1_all.std():.4f} A")
    if problems:
        ok = False
        print(f"   FAIL: {len(problems)} problem(s), first ones:")
        for p in problems[:8]:
            print("     " + p)
    else:
        print("   PASS: every atom has 3 hetero nearest neighbours, 6 like neighbours at a, "
              "3 hetero at 2a/sqrt3")

    # ---------------- 2. layers
    print("\n2. Layers")
    nl = len(groups)
    print(f"   {nl} layer(s) at z = " + ", ".join(f"{z:.3f}" for z in zmean) + " A")
    for k, g in enumerate(groups):
        el = {e: int(np.sum(sp[g] == e)) for e in sorted(set(sp))}
        buck = pos[g, 2].max() - pos[g, 2].min()
        print(f"   layer {k}: {len(g)} atoms {el}, buckling {buck:.3f} A")
        if len(set(el.values())) > 1:
            ok = False
            print(f"   FAIL: layer {k} is not stoichiometric")
    if nl > 1:
        gaps = list(np.diff(zmean))
        if periodic:
            gaps.append(wrap)
        print(f"   spacing between layers: " + ", ".join(f"{g:.4f}" for g in gaps) + " A"
              + ("  (incl. the periodic one: bulk)" if periodic else "  (slab in vacuum)"))

    # ---------------- 3. stacking
    if nl > 1:
        print("\n3. Stacking (closest atom of the next layer, in the xy projection)")
        pairs = [(k, k + 1) for k in range(nl - 1)]
        if periodic:
            pairs.append((nl - 1, 0))
        labels = []
        for k, l in pairs:
            gk, gl = groups[k], groups[l]
            off, same = [], []
            for i in gk:
                d = mic(pos[gl] - pos[i], lat, inv)
                rxy = np.linalg.norm(d[:, :2], axis=1)
                j = np.argmin(rxy)
                off.append(rxy[j])
                same.append(sp[gl[j]] == sp[i])
            off, same = np.array(off), np.array(same)
            ecl = off < TOL
            hollow = np.abs(off - a / SQ3) < 2 * TOL
            f_ecl, f_hol = ecl.mean(), hollow.mean()
            if f_ecl > 0.99 and not same[ecl].any():
                lab = "AA'  (B over N)"
            elif f_ecl > 0.99 and same[ecl].all():
                lab = "AA   (B over B, N over N)"
            elif abs(f_ecl - 0.5) < 0.02 and abs(f_hol - 0.5) < 0.02:
                kind = "B over N" if not same[ecl].any() else ("B over B" if sp[gk][ecl][0] == "B"
                                                                else "N over N")
                lab = f"AB-type (half eclipsed: {kind}; half over hexagon centres)"
            else:
                lab = f"other / shifted (mean in-plane offset {off.mean():.3f} A)"
            labels.append(lab)
            print(f"   layers {k}-{l}: {lab}   [eclipsed {100 * f_ecl:.0f} %, over hollow "
                  f"{100 * f_hol:.0f} %, max offset of eclipsed atoms {off[ecl].max() if ecl.any() else 0:.3f} A]")
        if all(l.startswith("AA'") for l in labels):
            print("   PASS: AA' stacking throughout (each B directly above an N)")
        else:
            ok = False
            print("   NOTE: not AA' everywhere - see above")
    print(f"\nRESULT: {'PASS' if ok else 'CHECK THE MESSAGES ABOVE'}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    res = [check(p) for p in sys.argv[1:]]
    sys.exit(0 if all(res) else 1)