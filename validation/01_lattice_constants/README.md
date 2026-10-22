# lattice_check

Lattice parameters of hexagonal boron nitride (hBN) predicted by a NEP machine-learned
potential, computed with [GPUMD](https://gpumd.org) and compared with experiment and DFT.

The script covers hBN from a single sheet to the bulk crystal, all in AA′ stacking (B above N):

| System | Reported |
|---|---|
| monolayer | in-plane lattice constant **a** |
| bilayer, trilayer, tetralayer | **a** and the interlayer distance **d** |
| bulk | **a**, **c** and **d = c/2** |

All values are the potential's static 0 K minimum. One command computes everything, prints a
comparison table and saves it as text and CSV. Results are cached per potential, so comparing
a new potential with an old one only runs the new one.

## Requirements

- Python 3.8+ with `numpy`
- A GPUMD build. The default method needs box relaxation in `minimize`
  (`minimize fire <tol> <steps> 1`), available in recent GPUMD versions. Older builds can
  use `--method scan`.
- A GPU node.

## Setup

Edit the `CONFIG` block at the top of `lattice_check.py` once:

| Key | Default | Meaning |
|---|---|---|
| `gpumd` | `/scratch/work/.../GPUMD/src/gpumd` | path to the GPUMD executable |
| `a0`, `c0`, `d0` | 2.504, 6.66, 3.33 Å | starting guesses for a, bulk c and the few-layer spacing |
| `vacuum` | 20.0 Å | vacuum above and below the monolayer and the stacks |
| `layers` | `(2, 3, 4)` | few-layer stacks between monolayer and bulk |
| `dft_structures` | all `None` | optional: paths to your relaxed VASP CONTCARs (see below) |
| `dft_values` | `{}` | optional: your DFT numbers, e.g. `{"bulk.c": 6.58}` |
| `dft_label` | `DFT (yours)` | column name for your DFT, e.g. `PBE-D3(BJ), this work` |
| `box_force_tol`, `box_max_steps` | 1e-5 eV/Å, 100000 | relaxation of atoms + box (`--method relax`) |
| `minimize`, `force_tol`, `max_steps` | `sd`, 1e-5 eV/Å, 200000 | relaxation at fixed box (`--method scan`) |
| `dump_line` | `dump_xyz 1 relaxed.xyz` | `run.in` line that saves the relaxed structure |
| `nep_cutoff` | 6.0 Å | largest radial cutoff of your NEP |

Every cell is built thicker than 2 × `nep_cutoff` in all directions, as GPUMD requires.

## Usage

```bash
sbatch run_lattice_check.sh nep_best.txt                    # one potential
sbatch run_lattice_check.sh nep_v1.txt nep_best.txt         # compare several
sbatch run_lattice_check.sh nep_best.txt --method scan      # energy scans (cross-check)
```

or directly on a GPU node:

```bash
python lattice_check.py nep_best.txt
python lattice_check.py nep_best.txt --jobs 8      # GPUMD runs at the same time
python lattice_check.py nep_best.txt --force       # ignore the cache
```

| Option | Default | Meaning |
|---|---|---|
| `potentials` | — | one or more NEP potential files |
| `--method` | `relax` | `relax` or `scan` (see below) |
| `--jobs N` | 4 | GPUMD runs at the same time |
| `--force` | off | recompute even if a cached result exists |

## Methods

### `--method relax` (default): one run per structure

Each structure is built at the starting guesses and relaxed with

```
minimize fire <box_force_tol> <box_max_steps> 1
```

The final `1` lets the box relax together with the atoms, like `fix box/relax` in LAMMPS or
`ISIF = 3` in VASP. Values are then read from the relaxed structure (`relaxed.xyz`):

- **a** from the relaxed box size, divided by the number of unit cells along x and y. If the
  box ends up distorted (a_x ≠ a_y, or shear), the log prints a note and the mean is reported.
- **c** (bulk) from the box height, divided by the number of stacked periods.
- **d** from the average height difference between neighbouring layers. Each layer's height
  is the mean height of its atoms, so small buckling averages out.

This takes 5 GPUMD runs in total.

### `--method scan`: energy scans

The energy is computed over a range of lattice parameters, and a 4th-order polynomial is
fitted to find the minimum. If the minimum lies at the edge of the window, the window moves
and the scan is repeated, within sensible bounds.

| System | What is scanned |
|---|---|
| monolayer | a, single points (a flat sheet has no forces to relax) |
| stacks | a, with the atoms relaxed at each point so the layers find their spacing; d is then measured in a final relaxation at the equilibrium a |
| bulk | a and c alternately (2–3 rounds); the AA′ atoms stay on symmetric sites, so no relaxation is needed |

This takes about 100 GPUMD runs. Use it to validate the relaxation method, or if your
GPUMD has no box relaxation.

## Comparing with your own DFT

Two ways, both optional:

- **Relaxed structures.** Put the paths to your relaxed VASP CONTCARs in
  `CONFIG["dft_structures"]` (`monolayer`, `bilayer`, `trilayer`, `tetralayer`, `bulk`).
  The script reads:
  - **a** directly from a hexagonal primitive cell, or as √3 × the median nearest B–N
    distance for any other cell (e.g. a rectangular supercell);
  - **d** from the layer heights, including the gap across the periodic boundary for bulk;
  - **c** = 2d for bulk.
- **Numbers.** Type them into `CONFIG["dft_values"]`, e.g.
  `{"bulk.a": 2.512, "bulk.c": 6.58, "bilayer.d": 3.36}`. Numbers override values read from
  structures.

Your own DFT is the fairest comparison, because it uses the same functional and van der Waals
correction as the training data.

## Built-in reference values

| Quantity | Experiment | DFT, literature |
|---|---|---|
| monolayer a | — | 2.510 Å |
| bilayer a | — | 2.510 Å |
| bilayer d | 3.25 ± 0.10 Å (TEM) | 3.360 Å |
| bulk a | 2.5040 Å | 2.510 Å |
| bulk c | 6.6612 Å | 6.610 Å |
| bulk d | 3.3306 Å (c/2) | 3.305 Å (c/2) |

Sources:

- Experiment, bulk a and c (297 K): Lynch & Drickamer, *J. Chem. Phys.* **44**, 181 (1966).
  The bulk d is derived as c/2, not reported in the paper.
- Experiment, bilayer d: Warner et al., *ACS Nano* **4**, 1299 (2010), an electron microscopy
  measurement on exfoliated sheets. Treat it as a rough check only.
- DFT: Thiemann et al., *J. Phys. Chem. C* **124**, 22278 (2020). The bulk d is
  derived as c/2.

The values and sources live in the `REFERENCES` dictionary near the top of the script. An
entry is `(value, source)` or `(value, source, uncertainty)`; the uncertainty is shown in the
table, e.g. `3.25±0.10`.

**When comparing:** experiment is at room temperature, while NEP and DFT give the static 0 K
minimum without zero-point motion, so NEP and DFT a and c are expected to sit slightly below
experiment. Different van der Waals corrections (e.g. D3 zero damping vs D3(BJ)) can change c
by a few hundredths of an Å.

## Output

All output goes into `lattice_check/`:

```
lattice_check/
  summary_<potentials>_<method>.txt    comparison table + references
  summary_<potentials>_<method>.csv    same table for spreadsheets
  <potential>/<method>/
    log.txt                            progress, warnings, notes
    results.json                       cached result
    monolayer/ bilayer/ ... bulk/      every GPUMD run: model.xyz, run.in,
                                       thermo.out, gpumd.log, relaxed.xyz
```

With `--method scan`, each structure folder contains one subfolder per scan point,
named `<parameter><round>_<window>_<point>`: e.g. `bulk/a0_0_05` is the bulk structure in
round 0, first window, middle point. The energy per atom of every point of a scan is collected
in `<parameter>_scan.txt` (e.g. `bulk/c0_scan.txt`), which you can plot to see the curve.

Example summary:

```
quantity              unit   nep_best.txt   DFT (yours)  DFT (lit.)          exp.   vs exp
------------------------------------------------------------------------------------------
monolayer hBN  a      A            2.5086             -      2.5100             -        -
bilayer hBN    a      A            2.5087             -      2.5100             -        -
bilayer hBN    d      A            3.4868             -      3.3600     3.25±0.10   +7.29%
...
bulk hBN       c      A            6.5472             -      6.6100        6.6612   -1.71%
```

The `vs exp` and `vs DFT` columns are percent deviations of NEP from experiment and from your
own DFT. The `vs DFT` column appears only when you have given DFT values.

## Verifying the structures

`check_hbn_structure.py` checks the honeycomb basis and the stacking of any structure
(`model.xyz`, `relaxed.xyz`, POSCAR, CONTCAR):

```bash
python check_hbn_structure.py lattice_check/nep_best/relax/bulk/relaxed.xyz
```

It confirms that every atom has 3 nearest neighbours of the other element at a/√3, and it
labels each pair of layers as AA′, AA, AB-type or shifted. Run it on the relaxed structures
to confirm the layers did not slide during relaxation.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `GPUMD executable ... not found` | set `CONFIG["gpumd"]` |
| `relaxed.xyz not found` | your GPUMD uses a different dump command; set `dump_line` (older versions: `dump_exyz 1 0 0`) |
| GPUMD fails at the `minimize` line | your GPUMD has no box relaxation; use `--method scan` |
| `WARNING: no energy minimum for ...` | the energy keeps falling across the allowed range; for layers this usually means the potential does not bind the layers. Plot the `*_scan.txt` file |
| `NOTE ...: box not hexagonal after relaxation` | the relaxed box is distorted; check the structure with `check_hbn_structure.py` |
| relax and scan disagree | see *Which one to trust* above |
| results did not change after editing a setting | settings that affect the calculation are part of the cache key; for anything else use `--force` |