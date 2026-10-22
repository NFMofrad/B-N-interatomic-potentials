# phonon_check

Phonon dispersion and density of states of hexagonal boron nitride (hBN) predicted by a NEP
machine-learned potential. Forces come from [GPUMD](https://gpumd.org), force constants and
bands from [phonopy](https://phonopy.github.io/phonopy/), and the result is compared with
experiment and DFT.

| System | Reported |
|---|---|
| bulk hBN (AA′ stacking) | dispersion along Γ–M–K–Γ–A, B/N-projected DOS, Γ-point modes with symmetry labels vs Raman/infrared, deviation from the measured (IXS) dispersion |
| monolayer hBN | dispersion along Γ–M–K–Γ, B/N-projected DOS, Γ-point modes with symmetry labels |
| both, optional | dispersion under biaxial in-plane strain (dynamic stability) |

All values are harmonic, at 0 K, and computed at the potential's own relaxed lattice. One
command computes everything, prints a summary, and saves the summary and the plots. Results
are cached per potential, so comparing a new potential with an old one only runs the new one.

## Usage

```bash
sbatch run_phonon_check.sh nep_best.txt                    # one potential
sbatch run_phonon_check.sh nep_v1.txt nep_best.txt         # compare several
```

or directly on a GPU node:

```bash
python phonon_check.py nep_best.txt
python phonon_check.py nep_best.txt --jobs 8      # GPUMD runs at the same time
python phonon_check.py nep_best.txt --force       # ignore the cache
```

| Option | Default | Meaning |
|---|---|---|
| `potentials` | — | one or more NEP potential files |
| `--jobs N` | 4 per GPU | GPUMD runs at the same time |
| `--force` | off | recompute even if a cached result exists |

To compare several potentials quickly, submit one job per potential, then run
`python phonon_check.py nep_v1.txt nep_v2.txt` once all are done. That combined call only
reads the cached results and redraws the summary and plots.

## Method

1. **Equilibrium lattice.** Bulk AA′ hBN and the monolayer are relaxed with atoms and box
   together (`minimize fire ... 1`) in the phonon supercell, and a (and c for bulk) is read
   from the relaxed box. All phonons are computed at this lattice.
2. **Displacements.** phonopy builds the hexagonal unit cell (bulk: 4 atoms, P6₃/mmc;
   monolayer: 2 atoms) and the displaced supercells.
3. **Forces.** GPUMD computes the forces of each displaced supercell in a single-point run
   and writes them with `dump_xyz ... force`. The force columns are found by name in the
   file's `Properties=` entry.
4. **Phonons.** phonopy builds the force constants (symmetrized for the acoustic sum rule),
   the bands along the path, the Γ-point modes with their symmetry labels (irreducible
   representations), and the density of states projected on B and N.

There is no LO–TO splitting: NEP has no long-range Coulomb term, so the polar optical modes
near Γ come out without it. Your DFT phonons are loaded without the non-analytical
correction (NAC) for the same reason, so the two are compared on equal terms.

### Biaxial strain (optional)

Each strain in `biaxial_strains` scales the in-plane lattice constant to a(1 + ε); for bulk,
c stays at its equilibrium value. The full phonon calculation is repeated for every strain.

- **Dynamic stability.** The lowest frequency on the path is listed for every strain.
  Imaginary (negative) modes under tension mean the structure is unstable there. Under
  compression, imaginary flexural (ZA) modes are expected: a compressed sheet buckles.

A useful range is about −1 % to +10 %. Monolayer hBN stays stable to large tension in DFT, so
imaginary modes at a few percent point to a problem with the potential, typically a lack of
strained structures in the training set.

## Comparing with your own DFT

Put your phonopy + VASP runs into `CONFIG["dft_phonopy"]`, as either

- the folder of the phonopy run, containing `phonopy_disp.yaml` and `FORCE_SETS`, or
- a `phonopy_params.yaml` that contains the forces (a `FORCE_SETS` next to it is used too).

```python
"dft_phonopy": {"bulk": "/scratch/.../dft/bulk_phonons",
                "monolayer": "/scratch/.../dft/mono_phonons/phonopy_params.yaml"},
```

The DFT bands are drawn as dashed lines on the same path, its density of states as a dashed
line, its Γ modes get a column in the summary, and its deviation from the IXS data is listed
next to NEP's. The path is given in fractional coordinates of a hexagonal cell with γ = 120°,
as the script builds it; your DFT cell should use the same setting, or K lands at a different
point. Your DFT is the fairest comparison, because it uses the same functional and van der
Waals correction as the training data.

## Built-in reference values

**Γ point, bulk** (Serrano et al., Table I, a compilation of Raman and infrared data):

| Mode | Experiment | Technique |
|---|---|---|
| E2g (shear) | 6.4 meV = 51.6 cm⁻¹ | Raman |
| A2u | 97.1 meV = 783.2 cm⁻¹ | infrared |
| E2g | 169.4 meV = 1366.3 cm⁻¹ | Raman |
| E1u (TO) | 169.5 meV = 1367.1 cm⁻¹ | infrared |
| E1u (LO) | 199.6 meV = 1609.9 cm⁻¹ | infrared |

The calculated modes are matched to these by symmetry label. Compare NEP's E1u with the TO
value: without LO–TO splitting there is only one E1u.

**Dispersion, bulk:** 180 inelastic x-ray scattering (IXS) points (115 in-plane, 65
out-of-plane polarization), stored in `serrano2007_fig2a_IXS.csv`. They were extracted from
the vector graphics of Fig. 2(a) of the paper, with the energy axis calibrated on its 5 meV
tick marks; as a check, the infrared symbols of the same figure come out at 199.61 and
97.11 meV, against 199.6 and 97.1 meV in Table I. Each point is stored with its path segment
and its fractional position along it.

Source: Serrano et al., *Phys. Rev. Lett.* **98**, 095503 (2007).

There are no built-in experimental values for the monolayer; use your DFT there.

**When comparing:** experiment is at room temperature, while NEP and DFT are harmonic 0 K
values. Expect differences of a few cm⁻¹ from temperature and anharmonicity, and more for the
low-frequency interlayer modes (the bulk shear E2g and B1g), which are very sensitive to the
interlayer distance. Check the lattice with `lattice_check.py` first.

## Output

All output goes into `phonon_check/`:

```
phonon_check/
  summary_<potentials>.txt                  lattice, Gamma table, IXS deviation, strains
  phonons_<potentials>_bulk.png              dispersion + DOS, bulk
  phonons_<potentials>_monolayer.png         dispersion + DOS, monolayer
  phonons_<potentials>_strain_bulk.png       (with biaxial_strains)
  phonons_<potentials>_strain_monolayer.png
  <potential>/
    log.txt                                 relaxed lattice, displacements, lowest frequencies
    results.json                            cached result
    bulk/  monolayer/
      relax/                                relaxation (relaxed.xyz)
      disp_001/, disp_002/                  force runs (forces.xyz)
      phonopy_params.yaml                   structure + forces, re-usable with phonopy
      band.yaml                             band structure
      strain_+0.010/ ...                    the same for every strain
```

Each GPUMD folder keeps `model.xyz`, `run.in`, `thermo.out` and `gpumd.log`.
`phonopy_params.yaml` can be loaded with phonopy's command line or Python API for anything
the script does not do (other paths, thermal properties, animations).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `GPUMD executable ... not found` | set `CONFIG["gpumd"]` |
| `phonopy is not installed` | `pip install --user phonopy` (after `module load scicomp-python-env` on Triton) |
| `relaxed.xyz not found` | your GPUMD uses a different dump command; set `dump_line` |
| `dump_force has been removed` | your `force_dump_line` still uses `dump_force`; use the default `dump_xyz 1 forces.xyz force` |
| `forces.xyz not found` or `no force column` | your GPUMD writes forces differently; set `force_dump_line` and `force_file`. The file must be extended XYZ with a force column in `Properties=` |
| `experimental dispersion not found` | put `serrano2007_fig2a_IXS.csv` next to the script or in the folder you run from |
| `?` instead of a symmetry label in the Γ table | phonopy could not assign the irreducible representation, usually because small numerical noise splits a degenerate pair; that mode then gets no experimental value. The frequencies are still correct |
| imaginary ZA modes of a few cm⁻¹ right at Γ | numerical artefact of finite supercells, not an instability |
| larger imaginary modes at zero strain | the potential is not stable for this structure; check the relaxed structure with `check_hbn_structure.py`, and try a larger `supercell` |
| DFT bands with K in the wrong place | your DFT cell uses γ = 60°; rebuild it with γ = 120° |
| results did not change after editing a setting | settings that affect the calculation are part of the cache key; for anything else use `--force` |