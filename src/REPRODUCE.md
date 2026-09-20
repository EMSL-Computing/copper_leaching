# Reproduce paper analysis and figures

All paper code: **`src/`**. Analysis products: **`accessibility_ensemble/analysis/`**.

```bash
bash scripts/reproduce_figures.sh
```

| Group | Command |
|-------|---------|
| Figs 3–7 + S2 + S5–S8 + dual panels | `python src/plot_accessibility_ensemble.py all-metrics` |
| S3 | `cd src/si_flow && bash run_tracer.sh plot --flow-only` |
| S4 | `python src/si_flow/plot_darcy_ensemble_perm.py` |
| S1 | frozen file only |

Do **not** re-run `src/screen_l_r_ensemble.py` (would regenerate geometries).
PFLOTRAN campaign HDF5 files are not in this repository.
