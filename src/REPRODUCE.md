# Reproduce paper analysis and figures

This file belongs to **[copper_leaching](https://github.com/EMSL-Computing/copper_leaching)**, the snapshot shipped with the manuscript. Development (HPC, campaign HDF5, Overleaf) is on `EMSL-Computing/multiphysics_modeling_agent`, not this tree.

All paper code: **`src/`**. Analysis products: **`accessibility_ensemble/analysis/`**.

```bash
bash scripts/reproduce_figures.sh
```

| Group | Command |
|-------|---------|
| Figs 3–7 + S2 + S5–S8 + dual panels | `python src/plot_accessibility_ensemble.py all-metrics` |
| Fig. 2 (scenario schematic) | `python src/plot_fig02_scenario_schematic.py` |
| Graphical abstract | `python src/plot_graphical_abstract.py` |
| S3 | `cd src/si_flow && bash run_tracer.sh plot --flow-only` |
| S4 | `python src/si_flow/plot_darcy_ensemble_perm.py` |
| S1 | frozen file only |

Do **not** re-run `src/screen_l_r_ensemble.py` (would regenerate geometries).
PFLOTRAN campaign HDF5 files are not in this repository.
Fig. 2 and the graphical abstract are hand-built SVG; PNG/PDF need headless Chromium (`CHROME_HEADLESS` or the Playwright cache).
