# strain_energy_check

Strain energy of hexagonal boron nitride (hBN) under in-plane strain, predicted by a NEP
machine-learned potential, computed with [GPUMD](https://gpumd.org) and compared with DFT.

| System | Strain states | Reported |
|---|---|---|
| monolayer hBN | biaxial, uniaxial zigzag, uniaxial armchair | strain energy per atom vs strain, compared with DFT (Peng et al. 2012) |
| bilayer hBN (AA′ stacking) | same | strain energy per atom vs strain |
| bulk hBN (AA′ stacking) | same | strain energy per atom vs strain, compared with the harmonic limit from the experimental elastic constants |

The default strains are −10 %, −7.5 %, −5 %, −2.5 %, 0, +2.5 %, +5 %, +7.5 % and +10 %. All
values are static (0 K). One command computes everything, prints a table, and saves a plot
and a CSV file. Results are cached per potential.

## Usage

```bash
sbatch run_strain_energy_check.sh nep_best.txt                    # one potential
sbatch run_strain_energy_check.sh nep_v1.txt nep_best.txt         # compare several
```

or directly on a GPU node:

```bash
python strain_energy_check.py nep_best.txt
python strain_energy_check.py nep_best.txt --jobs 8      # GPUMD runs at a time per GPU
python strain_energy_check.py nep_best.txt --force       # ignore the cache
```

| Option | Default | Meaning |
|---|---|---|
| `potentials` | — | one or more NEP potential files |
| `--jobs N` | 4 | GPUMD runs at the same time on each GPU |
| `--force` | off | recompute even if a cached result exists |

To compare several potentials quickly, submit one job per potential, then run
`python strain_energy_check.py nep_v1.txt nep_v2.txt` once all are done. That combined call
only reads the cached results and redraws the table and plot.

## Method

1. **Unstrained structure.** Each structure is built in an orthorhombic cell with x along the
   zigzag direction and y along the armchair direction, and relaxed with atoms and box
   together (`minimize fire ... 1`). This is the zero of strain: every strain is relative to
   the potential's own equilibrium lattice.
2. **Strain.** The relaxed cell and all atom positions are scaled by (1 + ε):

   | Mode | Scaled | Fixed |
   |---|---|---|
   | `biaxial` | x and y | cell height |
   | `zigzag` | x (uniaxial strain along the zigzag direction) | y, cell height |
   | `armchair` | y (uniaxial strain along the armchair direction) | x, cell height |

   The fixed directions make this pure strain (not uniaxial stress), the same strain states
   as in the DFT reference.
3. **Relaxation.** In every strained cell the atoms are relaxed at fixed box
   (`minimize fire 1e-5 ...`). The B and N sublattices shift and the bond angles adjust, and
   in the bilayer the layer spacing relaxes too.
4. **Strain energy.**

   ΔE(ε) = [E(ε) − E(0)] / N, in meV/atom,

   with E the potential energy of the supercell from `thermo.out` and N its number of atoms.
   E(0) comes from the unstrained cell put through the same fixed-box relaxation as the
   strained ones, so both energies come from the identical procedure and ΔE(0) = 0 exactly.
   Per atom, the three structures are on the same scale.

Near zero strain the curves are parabolas set by the elastic constants:
ΔE ≈ ½ C11 ε² V_atom (uniaxial) and ΔE ≈ (C11 + C12) ε² V_atom (biaxial). Further out,
compression costs more energy than the same tension: the anharmonicity of the bonds.

**Zigzag vs armchair.** Under zigzag strain no B–N bond is parallel to the strain; under
armchair strain one bond per atom is. At small strain both give the same energy (the in-plane
stiffness of a hexagonal lattice is isotropic); the difference grows with strain and comes
from the third- and higher-order elastic constants.

## Comparing with your own DFT

Put your energies into a CSV file and its path into `CONFIG["dft_energies"]`:

```
system,mode,strain,dE
monolayer,biaxial,5,140.2
monolayer,biaxial,-5,201.7
bulk,zigzag,10,190.4
```

`system` is `monolayer`, `bilayer` or `bulk`; `mode` is `biaxial`, `zigzag` or `armchair`;
`strain` is in %, `dE` in meV/atom. The values are drawn as open squares in the colour of the
strain mode. Any subset of points works. For a fair comparison, compute them the same way:
relax the unstrained cell, scale the cell with the other directions fixed, relax the ions
(`ISIF = 2` in VASP), and divide the energy difference by the number of atoms. Your DFT is
the only reference for the bilayer, and the fairest one overall, because it uses the same
functional and van der Waals correction as the training data.

## Built-in reference values

**Monolayer:** DFT (PBE) of Peng, Ji & De, *Comput. Mater. Sci.* **56**, 11 (2012). Their
energy–strain curves (Fig. 2) are not tabulated. The script rebuilds them from their
fifth-order continuum fit to the DFT data (the 14 elastic constants of their Table 1): their
stress–strain relations (Eqs. 7, 10 and 14) are integrated over the Lagrangian strain
η = ε + ε²/2, and the energy per area is multiplied by the area per atom (a0 = 2.512 Å). As a
check, the rebuilt curves reproduce the maximum stresses quoted in the paper within about
1 % (armchair, biaxial) and 4 % (zigzag). Their sheet is kept flat (atoms relaxed in-plane
only), as here.

**Bulk:** no energy–strain curve was found in the literature. The bulk panel shows the
harmonic limit from the experimental elastic constants, C11 = 811 GPa and C12 = 169 GPa
(Bosak et al., *Phys. Rev. B* **73**, 041402(R) (2006)), with the experimental lattice. It
is valid at small strain only and checks the curvature at the bottom of the curve.

**Bilayer:** no energy–strain curve was found in the literature. Fujimoto & Saito,
*Phys. Rev. B* **94**, 245427 (2016) study bilayers under biaxial tension but report band
gaps and stacking energies.

Values at the default strains (meV/atom):

| strain | monolayer, biaxial (Peng) | monolayer, zigzag (Peng) | monolayer, armchair (Peng) | bulk, biaxial (harmonic) | bulk, uniaxial (harmonic) |
|---|---|---|---|---|---|
| −10 % | 794.2 | 297.2 | 289.5 | 553.1 | 228.9 |
| −7.5 % | 420.0 | 160.5 | 157.4 | 311.1 | 128.7 |
| −5 % | 175.1 | 68.4 | 67.5 | 138.3 | 57.2 |
| −2.5 % | 41.0 | 16.4 | 16.2 | 34.6 | 14.3 |
| +2.5 % | 35.8 | 14.9 | 15.0 | 34.6 | 14.3 |
| +5 % | 133.5 | 56.8 | 57.5 | 138.3 | 57.2 |
| +7.5 % | 280.6 | 121.8 | 123.6 | 311.1 | 128.7 |
| +10 % | 466.4 | 206.0 | 209.3 | 553.1 | 228.9 |

The constants live in `PC` (Peng) and `BULK_CIJ` (bulk) near the top of the script.

## Output

All output goes into `strain_energy_check/`:

```
strain_energy_check/
  strain_energy_<potentials>.png / .pdf      plot: one panel per structure
  strain_energy_<potentials>.csv             all energies (potential, system, mode, strain, dE)
  <potential>/
    log.txt                                  GPU assignment, relaxed lattice, energies
    results.json                             cached result
    monolayer/  bilayer/  bulk/
      relax/                                 relaxation of the unstrained cell (relaxed.xyz)
      unstrained/                            fixed-box relaxation at zero strain: E(0)
      biaxial_+0.0250/ ... armchair_-0.1000/ one folder per strained cell
```

Each folder keeps `model.xyz`, `run.in`, `thermo.out` and `gpumd.log`. The table printed to
the screen (and so in the Slurm output file) lists NEP next to the reference values for every
strain.

In the plot, colour gives the strain state (biaxial red, zigzag green, armchair blue), NEP
is drawn as solid lines with markers (one marker shape per potential), Peng et al. as dashed
lines (monolayer), the bulk harmonic limit as dotted lines, and your DFT as open squares.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `GPUMD executable ... not found` | set `CONFIG["gpumd"]` |
| `relaxed.xyz not found` | your GPUMD uses a different dump command; set `dump_line` |
| `unknown strain mode(s)` | `modes` may only contain `biaxial`, `zigzag` and `armchair` |
| `DFT energies not found` | check the path in `CONFIG["dft_energies"]` |
| a curve with a kink or a drop at large strain | the structure changed (bond breaking, stacking change); check `relaxed.xyz` of that folder, e.g. with `check_hbn_structure.py` |
| very slow | check a folder's `gpumd.log`: if minimization runs to `max_steps`, `force_tol` is too tight for GPUMD's precision; try `force_tol = 1e-4` |
| results did not change after editing a setting | settings that affect the calculation are part of the cache key; for anything else use `--force` |