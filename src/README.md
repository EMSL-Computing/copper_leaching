# `src/` — paper code

Shipped from [copper_leaching](https://github.com/EMSL-Computing/copper_leaching) (manuscript snapshot). Not the multiphysics development remote.

```text
src/
  plot_accessibility_ensemble.py   # recovery figures (Figs 3–7, S2, S5–S8)
  paper_dataset_lib.py             # accessibility_ensemble/ paths
  run_*_campaign.py                # Set A/B drivers (need PFLOTRAN + local HDF5)
  bean_bucket_generator.py         # geometry stack (frozen for this paper)
  si_flow/                         # SI S3–S4 flow/perm validation
  REPRODUCE.md
```

```bash
python src/plot_accessibility_ensemble.py all-metrics
bash scripts/reproduce_figures.sh
cd src/si_flow && bash run_tracer.sh plot --flow-only
python src/si_flow/plot_darcy_ensemble_perm.py
```

| Do not re-run for this paper | Safe with existing analysis products |
|------------------------------|--------------------------------------|
| `screen_l_r_ensemble.py` | `plot_accessibility_ensemble.py` |
| Rewrite of the 25 networks | `si_flow/` S3/S4 plots |

S1 PNG: `accessibility_ensemble/ensemble/screening/l_r_screening.png`.
