# `src/si_flow/` — SI flow & permeability validation (Figs. S3–S4)

**Purpose:** Supporting Information figures that validate **Stokes ↔ Darcy flow** on geom437 and summarize **ensemble permeability / outlet flux** distributions.

Not the main recovery analysis (that is `src/plot_accessibility_ensemble.py`).

## Layout

| Path | Contents |
|------|----------|
| `run_tracer.py` / `run_tracer.sh` | CLI for plot / calibrate / tracer / ensemble |
| `plot_darcy_stokes.py` | SI Fig. **S3** multi-panel (pressure, speed, 1:1) |
| `plot_darcy_ensemble_perm.py` | SI Fig. **S4** \(k_{\mathrm{pore}}\), \(k_{\mathrm{eff}}\), \(Q_{\mathrm{out}}\) |
| `plot_*_map.py`, `plot_flow_direction.py` | Extra SI-style maps (connected pore, chrysocolla) |
| `config.json` | Paths relative to this directory |
| `geometry/geom437/` | Frozen pore + mineral fields |
| `reference/` | Calibrated Darcy inputs (short window) |
| `figures/` | Exported PNGs |
| `stokes_flow_cache/` | Cached Stokes fields for ensemble stats |

## Commands (from repo root)

```bash
# SI Fig. S3
cd src/si_flow && bash run_tracer.sh plot --flow-only

# SI Fig. S4
python src/si_flow/plot_darcy_ensemble_perm.py

```

## Full-stack Stokes/Darcy solvers

Live under `2DStokes_FD/` and `darcy_vs_stokes/`. This folder is the SI validation package only.
