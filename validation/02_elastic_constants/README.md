# elastic_check

Elastic constants of hexagonal boron nitride (hBN) predicted by a NEP machine-learned
potential, computed with [GPUMD](https://gpumd.org) and compared with experiment and DFT.

| System | Reported |
|---|---|
| bulk hBN (AA′ stacking) | C11, C12, C13, C33, C44, C66 (GPa), Reuss bulk modulus, and the check (C11 − C12)/2 |
| monolayer hBN | C11, C12, C66 (N/m, 2D), 2D Young's modulus Y, Poisson ratio ν, and the check (C11 − C12)/2 |

All values are static (0 K) and **relaxed-ion**: the atoms are relaxed in every strained
cell, as in experiment and in VASP's "TOTAL ELASTIC MODULI". One command computes
everything, prints a comparison table with the full stiffness matrices, and saves it as text
and CSV. Results are cached per potential, and the runs are spread over all GPUs of the job.

## Usage

```bash
sbatch run_elastic_check.sh nep_best.txt                    # one potential
sbatch run_elastic_check.sh nep_v1.txt nep_best.txt         # compare several
```

or directly on a GPU node:

```bash
python elastic_check.py nep_best.txt
python elastic_check.py nep_best.txt --jobs 16     # GPUMD runs at the same time
python elastic_check.py nep_best.txt --force       # ignore the cache
```

| Option | Default | Meaning |
|---|---|---|
| `potentials` | — | one or more NEP potential files |
| `--jobs N` | 4 per GPU | GPUMD runs at the same time |
| `--force` | off | recompute even if a cached result exists |

To compare several potentials quickly, submit one job per potential, then run
`python elastic_check.py nep_v1.txt nep_v2.txt` once all are done. That combined call only
reads the cached results and writes the comparison table.

## Method

1. **Reference.** Bulk AA′ hBN and the monolayer are relaxed with atoms and box together
   (`minimize fire ... 1`). Everything below is strained relative to this equilibrium, so the
   constants belong to the potential's own lattice.
2. **Strains.** Each strain component j (Voigt 1–6 = xx, yy, zz, yz, xz, xy) is applied at
   ±0.5 % and ±1 %. Bulk uses all six; the monolayer uses the in-plane ones (xx, yy, xy).
   Shear strains are engineering strains (γ = 2ε), the convention of C44, C55 and C66.
3. **Internal relaxation.** In every strained cell the atoms are relaxed at fixed cell. This
   matters in hBN: under shear the B and N sublattices shift against each other, which lowers
   C12 and C66.
4. **Stiffness.** GPUMD's stress tensor is read from `thermo.out` (stress = −pressure), and
   C_ij = ∂σ_i/∂ε_j is the linear coefficient of a quadratic fit through the 5 points
   (−1 %, −0.5 %, 0, +0.5 %, +1 %). The quadratic fit keeps small non-linearity from biasing
   the slope. The relaxed reference is the zero-strain point of every fit.
5. **Hexagonal constants.** The matrix is symmetrized, and symmetry-equivalent pairs are
   averaged: C11 and C22, C13 and C23, C44 and C55.

From the constants:

- **Reuss bulk modulus** B = [(C11 + C12) C33 − 2 C13²] / (C11 + C12 + 2 C33 − 4 C13)
- **Monolayer 2D values** = 3D stress of the cell × cell height (1 GPa·Å = 0.1 N/m), so they
  do not depend on a chosen layer thickness
- **2D Young's modulus** Y = (C11² − C12²)/C11, **Poisson ratio** ν = C12/C11

### Built-in checks

Printed to the screen and to `log.txt`:

- **Stress vs energy.** Every diagonal C_jj is also computed from the energy curvature of
  the same runs, C_jj = (∂²E/∂ε_j²)/V. This route uses only the energy column, not the stress,
  so the two must agree (within 2 %, or 0.5 GPa for small constants); disagreements are
  flagged.
- **Hexagonal symmetry.** C66 must equal (C11 − C12)/2; both are in the table. The
  symmetry-equivalent pairs (C11/C22, C44/C55) are printed separately in the log and should
  agree.
- **Zero entries.** The full 6×6 matrices are printed in the summary; the entries that are
  zero by symmetry (e.g. C14, C16) should come out close to zero.
- **Residual stress.** The stress of the relaxed reference is printed; it should be near zero.
- **Sign convention.** If GPUMD's pressure sign is opposite to the assumed one, the diagonal
  constants come out negative; the script detects this, flips the sign and says so.

## Comparing with your own DFT

Two ways, both optional:

- **VASP OUTCARs.** Put the paths in `CONFIG["dft_outcar"]` (`bulk`, `monolayer`), from a
  calculation with `IBRION = 6` and `ISIF = 3`. The script reads the "TOTAL ELASTIC MODULI
  (kBar)" block, which includes ionic relaxation, converts kBar to GPa and VASP's order
  (XX YY ZZ XY YZ ZX) to Voigt order. For the monolayer the values are converted to N/m with
  the cell height read from the same OUTCAR.
- **Numbers.** Type them into `CONFIG["dft_values"]`, e.g.
  `{"bulk.C33": 30.1, "monolayer.C11": 290}` (bulk in GPa, monolayer in N/m, keys as in the
  table). Numbers override values read from OUTCARs.

## Built-in reference values

| Quantity | Experiment | DFT, literature |
|---|---|---|
| bulk C11 | 811 ± 12 GPa | 881.8 GPa |
| bulk C12 | 169 ± 24 GPa | 195.6 GPa |
| bulk C13 | 0 ± 3 GPa | 0.0 GPa |
| bulk C33 | 27.0 ± 0.5 GPa | 28.9 GPa |
| bulk C44 | 7.7 ± 0.5 GPa | 3.3 GPa |
| bulk C66 | 321 GPa (derived) | 343.1 GPa (derived) |
| bulk B (Reuss) | 25.6 ± 8 GPa | 27.4 GPa |
| monolayer C11 | — | 293.2 N/m |
| monolayer C12 | — | 66.1 N/m |
| monolayer Y | 289 ± 24 N/m | 278.3 N/m |
| monolayer ν | — | 0.2254 (derived) |

Sources:

- Experiment, bulk: Bosak et al., *Phys. Rev. B* **73**, 041402(R) (2006), inelastic x-ray
  scattering. C66 is not reported there; the value is derived as (C11 − C12)/2.
- Experiment, monolayer: Falin et al., *Nat. Commun.* **8**, 15815 (2017), AFM
  nanoindentation.
- DFT, bulk: Thiemann et al., *J. Phys. Chem. C* **124**, 22278 (2020), PBE+D3. C66 is
  derived as (C11 − C12)/2.
- DFT, monolayer: Peng, Ji & De, *Comput. Mater. Sci.* **56**, 11–17 (2012), PBE, Table 1
  (C11, C12, Ys). The paper does not tabulate ν; it is derived as C12/C11. The text of the
  paper quotes Ys = 279.2 N/m and ν = 0.2176, which do not follow from its own Table 1
  (they match the earlier preprint, arXiv:1107.1453); the Table 1 values are used here.

The values and sources live in the `REFERENCES` dictionary near the top of the script. An
entry is `(value, source)` or `(value, source, uncertainty)`; the uncertainty is shown in the
table, e.g. `811.0±12`.

**When comparing:** experiment is at room temperature, while NEP and DFT are static 0 K
values. C33 and C44 depend strongly on the interlayer distance, so a potential whose bulk c
is off will shift them. Check the lattice with `lattice_check.py` first.

## Output

All output goes into `elastic_check/`:

```
elastic_check/
  summary_<potentials>.txt          comparison table, full 6x6 matrices, references
  summary_<potentials>.csv          the table for spreadsheets
  <potential>/
    log.txt                         residual stress, stress-vs-energy checks, notes
    results.json                    cached result
    bulk/reference/                 relaxed reference (relaxed.xyz)
    bulk/e1_-0.0100 ... e6_+0.0100/ 24 strained bulk runs
    monolayer/reference/
    monolayer/e1_..., e2_..., e6_.../  12 strained monolayer runs
```

Folder names give the strain: `e1_-0.0100` is component 1 (xx) at −1 %, `e6_+0.0050` is
shear xy at +0.5 %. Each folder keeps `model.xyz`, `run.in`, `thermo.out` and `gpumd.log`.

The `vs exp` and `vs DFT` columns are percent deviations of NEP from experiment and from
your own DFT. C13 is about zero, so no percent deviation is given for it.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `GPUMD executable ... not found` | set `CONFIG["gpumd"]` |
| `relaxed.xyz not found` | your GPUMD uses a different dump command; set `dump_line` |
| `expected the pressure tensor in columns 4-9` | your GPUMD's `thermo.out` has no stress columns; a newer GPUMD is needed |
| `<-- stress and energy disagree` | the stress columns are in a different order or unit in your GPUMD version, or the relaxations did not converge; send a `thermo.out` |
| very slow | check a strained folder's `gpumd.log`: if minimization runs to `max_steps`, `force_tol` is too tight for GPUMD's precision; try `force_tol = 1e-4`, `max_steps = 20000` (effect on the constants well under 1 %) |
| C44 noisy | the shear stresses are tiny; try `"strains": (0.01, 0.02)` |
| GPUs not fully used | raise `--jobs` and `--cpus-per-task` together; lower them if runs fail with out-of-memory errors |
| results did not change after editing a setting | settings that affect the calculation are part of the cache key; for anything else use `--force` |