"""Calibrated PFLOTRAN Darcy ensemble permeability and outlet-flow distributions (stage-1, n=25).

Collects pore-cell permeability k_pore, domain effective permeability k_eff, and
outlet volumetric flow Q_out from calibrated Darcy flow at Delta p = 1 Pa.
Q_out is reported in uL/min for a 1 mm chip thickness (2 mm x 8 mm footprint).

Example:
    python plot_darcy_ensemble_perm.py
    python plot_darcy_ensemble_perm.py --config config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent

def _si_flow_root() -> Path:
    for cand in (HERE, *HERE.parents):
        if (cand / "config.json").is_file() and (cand / "run_tracer.py").is_file():
            return cand
    return HERE

SI_FLOW_ROOT = _si_flow_root()
REPO_ROOT = SI_FLOW_ROOT.parents[1]  # src/si_flow -> src -> repo
_SRC = REPO_ROOT / "src"
for _p in (str(_SRC), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if str(SI_FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(SI_FLOW_ROOT))


from darcy_vs_stokes.darcy_utils import load_darcy_case
from darcy_vs_stokes.utils import effective_permeability

FIG_DPI = 300


def load_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    base = config_path.parent
    return {
        **cfg,
        "config_path": config_path.resolve(),
        "stage1_manifest": (base / cfg["stage1_manifest"]).resolve(),
        "stage1_sweep_root": (base / cfg["stage1_sweep_root"]).resolve(),
        "ensemble_perm_figure_out": (base / cfg["ensemble_perm_figure_out"]).resolve(),
        "ensemble_perm_csv_out": (base / cfg["ensemble_perm_csv_out"]).resolve(),
    }


def load_stage1_entries(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text())
    entries = []
    for row in data["entries"]:
        entries.append(
            {
                "realization_index": int(row["realization_index"]),
                "geom_seed": int(row["geom_seed"]),
                "phi": float(row["phi"]),
                "l_r_median_um": float(row["l_r_median_um"]),
            }
        )
    return sorted(entries, key=lambda e: e["realization_index"])


def case_reference_dir(cfg: dict, entry: dict) -> Path:
    idx = entry["realization_index"]
    seed = entry["geom_seed"]
    return cfg["stage1_sweep_root"] / f"real{idx:03d}_geom{seed}" / "reference"


def q_ref_to_q_out_ul_min(q_ref_m2_s: float, lz_m: float) -> float:
    """Convert 2-D outlet flux [m^2/s] to volumetric flow [uL/min]."""
    q_m3_s = abs(q_ref_m2_s) * lz_m
    return q_m3_s * 1.0e9 * 60.0


def metrics_from_darcy_reference(
    reference_dir: Path,
    *,
    dx_m: float,
    mu_pa_s: float,
    dP_pa: float,
    p_out_pa: float,
    lz_m: float,
) -> dict[str, float | str]:
    stokes_npz = reference_dir / "calibrate_k_pore_run" / "stokes_reference.npz"
    darcy_h5 = reference_dir / "simulate_darcy_run" / "darcy_calibrated.h5"
    if not stokes_npz.is_file():
        raise FileNotFoundError(stokes_npz)
    if not darcy_h5.is_file():
        raise FileNotFoundError(darcy_h5)

    cal = np.load(stokes_npz)
    mask = np.asarray(cal["mask"])
    target_time_s = float(cal["times"][-1]) if "times" in cal.files else 10.0
    darcy = load_darcy_case(
        darcy_h5,
        target_time_s,
        mask,
        dx_m,
        p_out_pa + dP_pa,
        p_out_pa,
    )
    q_ref = float(darcy["q_ref"])
    k_eff = effective_permeability(q_ref, mask, dx_m, mu_pa_s, dP_pa)
    k_pore = float(cal["k_pore"]) if "k_pore" in cal.files else float("nan")
    return {
        "q_ref_m2_s": q_ref,
        "k_eff_m2": k_eff,
        "k_pore_m2": k_pore,
        "q_out_ul_min": q_ref_to_q_out_ul_min(q_ref, lz_m),
        "source_h5": str(darcy_h5),
    }


def collect_ensemble_rows(cfg: dict) -> list[dict]:
    entries = load_stage1_entries(cfg["stage1_manifest"])
    rows: list[dict] = []
    for entry in entries:
        ref_dir = case_reference_dir(cfg, entry)
        m = metrics_from_darcy_reference(
            ref_dir,
            dx_m=float(cfg["dx_m"]),
            mu_pa_s=float(cfg["mu_Pa_s"]),
            dP_pa=float(cfg["dP_Pa"]),
            p_out_pa=float(cfg["p_out_Pa"]),
            lz_m=float(cfg["lz_m"]),
        )
        rows.append(
            {
                "geom_seed": entry["geom_seed"],
                "realization_index": entry["realization_index"],
                "phi": entry["phi"],
                "l_r_median_um": entry["l_r_median_um"],
                **m,
            }
        )
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "geom_seed",
        "realization_index",
        "phi",
        "l_r_median_um",
        "k_pore_m2",
        "k_eff_m2",
        "q_ref_m2_s",
        "q_out_ul_min",
        "source_h5",
    ]
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _filled_distribution(ax: plt.Axes, values: np.ndarray, *, color: str, xlabel: str, title: str) -> None:
    """Plot a filled KDE with rug ticks and median marker."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    med = float(np.median(values))
    pad = 0.12 * max(vmax - vmin, 1.0e-30)

    sns.kdeplot(
        values,
        ax=ax,
        color=color,
        fill=True,
        alpha=0.35,
        linewidth=1.6,
        bw_adjust=0.85,
        clip=(vmin - pad, vmax + pad),
        warn_singular=False,
    )
    sns.rugplot(values, ax=ax, color=color, alpha=0.55, height=0.04, linewidth=0.8)
    ax.axvline(med, color="0.15", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylabel("Density")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=9, pad=6)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    sns.despine(ax=ax)


def plot_distributions(rows: list[dict], out_png: Path, *, lz_m: float, dP_pa: float) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    k_pore = np.array([r["k_pore_m2"] for r in rows], dtype=float) * 1.0e10
    k_eff = np.array([r["k_eff_m2"] for r in rows], dtype=float) * 1.0e12
    q_out = np.array([r["q_out_ul_min"] for r in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), sharey=False)
    panels = (
        (axes[0], k_pore, "#2C6E8C", r"$k_{\mathrm{pore}}$ [$10^{-10}$ m$^2$]", "(a) Calibrated pore permeability"),
        (axes[1], k_eff, "#3A7D44", r"$k_{\mathrm{eff}}$ [$10^{-12}$ m$^2$]", "(b) Domain effective permeability"),
        (axes[2], q_out, "#8C4A2F", r"$Q_{\mathrm{out}}$ [$\mu$L min$^{-1}$]", "(c) Outlet volumetric flow"),
    )

    for ax, values, color, xlabel, title in panels:
        _filled_distribution(ax, values, color=color, xlabel=xlabel, title=title)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.92, wspace=0.32)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out_png}")


def run_all(cfg: dict) -> list[dict]:
    rows = collect_ensemble_rows(cfg)
    write_csv(rows, cfg["ensemble_perm_csv_out"])
    print(f"Saved {cfg['ensemble_perm_csv_out']}")
    plot_distributions(
        rows,
        cfg["ensemble_perm_figure_out"],
        lz_m=float(cfg["lz_m"]),
        dP_pa=float(cfg["dP_Pa"]),
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Darcy ensemble k_pore, k_eff, Q_out distributions")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    run_all(cfg)


if __name__ == "__main__":
    main()