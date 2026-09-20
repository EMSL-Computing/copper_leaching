"""Compare Stokes and PFLOTRAN Darcy outputs."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.stats import spearmanr
try:
    from darcy_vs_stokes.utils import effective_permeability
except ModuleNotFoundError:
    from utils import effective_permeability

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def compute_score(metrics: dict[str, float]) -> float:
    return (
        0.20 * metrics["pressure_rel_rmse"]
        + 0.40 * metrics["speed_scale_err"]
        + 0.20 * metrics["btc_rel_rmse"]
        + 0.20 * metrics["t50_rel_err"]
    )


def arrival_time_cumulative(times: np.ndarray, values: np.ndarray, frac: float) -> float:
    if len(values) == 0:
        return float("nan")
    total = float(np.trapezoid(values, times))
    if total <= 1.0e-30:
        return float("nan")
    dt = np.diff(times)
    seg = 0.5 * (values[:-1] + values[1:]) * dt
    cum = np.zeros(len(times), dtype=float)
    cum[1:] = np.cumsum(seg)
    cum /= total
    idx = np.where(cum >= frac)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(times[0])
    t0, t1 = times[i - 1], times[i]
    c0, c1 = cum[i - 1], cum[i]
    if c1 == c0:
        return float(t1)
    w = (frac - c0) / (c1 - c0)
    return float(t0 + w * (t1 - t0))


def rtd_series(times: np.ndarray, btc: np.ndarray) -> np.ndarray:
    area = float(np.trapezoid(btc, times))
    if area <= 1.0e-30:
        return np.zeros_like(btc, dtype=float)
    return np.asarray(btc, dtype=float) / area


def btc_series(case: dict[str, np.ndarray | float], q_ref: float | None = None) -> np.ndarray:
    if q_ref is None:
        q_ref = float(case["q_ref"])
    return np.asarray(case["right_flux"]) / max(float(q_ref), 1.0e-30)


def compute_metrics(
    stokes: dict[str, np.ndarray | float],
    darcy: dict[str, np.ndarray | float],
) -> dict[str, float]:
    mask = np.asarray(stokes["mask"]) == 1
    stokes_speed = np.sqrt(np.asarray(stokes["u"]) ** 2 + np.asarray(stokes["v"]) ** 2)
    darcy_speed = np.sqrt(np.asarray(darcy["u"]) ** 2 + np.asarray(darcy["v"]) ** 2)

    pressure_rmse = rmse(np.asarray(darcy["p"])[mask], np.asarray(stokes["p"])[mask])
    pressure_rel_rmse = pressure_rmse / max(float(stokes["dP"]), 1.0e-30)

    speed_rmse = rmse(darcy_speed[mask], stokes_speed[mask])
    speed_rel_rmse = speed_rmse / max(float(np.mean(stokes_speed[mask])), 1.0e-30)

    q_ref_common = max(float(stokes["q_ref"]), 1.0e-30)
    stokes_btc = btc_series(stokes, q_ref_common)
    darcy_btc = btc_series(darcy, q_ref_common)
    darcy_interp = np.interp(
        np.asarray(stokes["times"]),
        np.asarray(darcy["times"]),
        darcy_btc,
        left=darcy_btc[0],
        right=darcy_btc[-1],
    )
    btc_rmse = rmse(darcy_interp, stokes_btc)
    btc_rel_rmse = btc_rmse / max(float(np.max(stokes_btc)), 1.0e-30)
    stokes_rtd = rtd_series(np.asarray(stokes["times"]), stokes_btc)
    darcy_rtd = rtd_series(np.asarray(darcy["times"]), darcy_btc)
    darcy_rtd_interp = np.interp(
        np.asarray(stokes["times"]),
        np.asarray(darcy["times"]),
        darcy_rtd,
        left=darcy_rtd[0],
        right=darcy_rtd[-1],
    )
    rtd_rel_rmse = rmse(darcy_rtd_interp, stokes_rtd) / max(float(np.max(stokes_rtd)), 1.0e-30)
    stokes_tau_mean = float(np.trapezoid(np.asarray(stokes["times"]) * stokes_rtd, np.asarray(stokes["times"])))
    darcy_tau_mean = float(np.trapezoid(np.asarray(darcy["times"]) * darcy_rtd, np.asarray(darcy["times"])))

    stokes_mean_speed = max(float(np.mean(stokes_speed[mask])), 1.0e-30)
    darcy_pore_speed = darcy_speed[mask]
    darcy_mean_speed = float(np.mean(darcy_pore_speed))
    speed_scale_err = abs(darcy_mean_speed - stokes_mean_speed) / stokes_mean_speed

    stokes_k_eff = effective_permeability(
        float(stokes["q_ref"]),
        np.asarray(stokes["mask"]),
        float(stokes["dx"]),
        float(stokes["mu"]),
        float(stokes["dP"]),
    )
    stokes_t10 = arrival_time_cumulative(np.asarray(stokes["times"]), stokes_btc, 0.1)
    stokes_t50 = arrival_time_cumulative(np.asarray(stokes["times"]), stokes_btc, 0.5)
    darcy_t10 = arrival_time_cumulative(np.asarray(darcy["times"]), darcy_btc, 0.1)
    darcy_t50 = arrival_time_cumulative(np.asarray(darcy["times"]), darcy_btc, 0.5)

    stokes_speed_pore = stokes_speed[mask]
    rho_speed, _ = spearmanr(stokes_speed_pore, darcy_pore_speed)
    thresh = np.percentile(stokes_speed_pore, 80.0)
    fast_mask_stokes = stokes_speed_pore > thresh
    fast_mask_darcy = darcy_pore_speed > thresh
    fast_overlap = np.sum(fast_mask_stokes & fast_mask_darcy) / max(np.sum(fast_mask_stokes), 1)

    metrics = {
        "pressure_rel_rmse": pressure_rel_rmse,
        "speed_rel_rmse": speed_rel_rmse,
        "btc_rel_rmse": btc_rel_rmse,
        "rtd_rel_rmse": rtd_rel_rmse,
        "speed_scale_err": speed_scale_err,
        "stokes_k_eff": stokes_k_eff,
        "stokes_t10": stokes_t10,
        "stokes_t50": stokes_t50,
        "darcy_t10": darcy_t10,
        "darcy_t50": darcy_t50,
        "t10_rel_err": abs(darcy_t10 - stokes_t10) / max(abs(stokes_t10), 1.0e-30),
        "t50_rel_err": abs(darcy_t50 - stokes_t50) / max(abs(stokes_t50), 1.0e-30),
        "stokes_tau_mean": stokes_tau_mean,
        "darcy_tau_mean": darcy_tau_mean,
        "tau_mean_rel_err": abs(darcy_tau_mean - stokes_tau_mean) / max(abs(stokes_tau_mean), 1.0e-30),
        "speed_rank_corr": float(rho_speed),
        "fast_channel_overlap": float(fast_overlap),
    }
    metrics["score"] = compute_score(metrics)
    return metrics


def masked(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = field.astype(float).copy()
    out[mask == 0] = np.nan
    return out


def solids_overlay(mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros(mask.shape + (4,), dtype=np.float32)
    overlay[mask == 0] = [0.1, 0.1, 0.1, 1.0]
    return overlay


def format_metadata_lines(metadata: dict[str, object] | None) -> list[str]:
    if not metadata:
        return []
    preferred = [
        "label",
        "score",
        "pressure_rel_rmse",
        "speed_rel_rmse",
        "btc_rel_rmse",
        "rtd_rel_rmse",
        "speed_scale_err",
        "speed_rank_corr",
        "fast_channel_overlap",
        "stokes_k_eff",
        "stokes_tau_mean",
        "darcy_tau_mean",
        "tau_mean_rel_err",
    ]
    lines: list[str] = []
    seen: set[str] = set()

    def add_item(key: str, value: object) -> None:
        if isinstance(value, float):
            lines.append(f"{key:<18} = {value:.3e}")
        elif isinstance(value, (int, np.integer)):
            lines.append(f"{key:<18} = {int(value)}")
        elif isinstance(value, str):
            lines.append(f"{key:<18} = {value}")
        else:
            lines.append(f"{key:<18} = {value}")

    for key in preferred:
        if key in metadata:
            add_item(key, metadata[key])
            seen.add(key)

    for key, value in metadata.items():
        if key in seen or key == "title":
            continue
        if isinstance(value, (float, int, np.integer, str)):
            add_item(key, value)

    return lines


def plot_comparison(
    plot_out: Path,
    stokes: dict[str, np.ndarray | float],
    darcy: dict[str, np.ndarray | float],
    metadata: dict[str, object] | None = None,
) -> None:
    mask = np.asarray(stokes["mask"])
    overlay = solids_overlay(mask)
    extent = [0, mask.shape[1], 0, mask.shape[0]]
    kw = dict(origin="lower", extent=extent, interpolation="nearest")

    stokes_p = masked(np.asarray(stokes["p"]), mask)
    darcy_p = masked(np.asarray(darcy["p"]), mask)
    p_err = masked(np.abs(np.asarray(darcy["p"]) - np.asarray(stokes["p"])), mask)

    stokes_speed = np.sqrt(np.asarray(stokes["u"]) ** 2 + np.asarray(stokes["v"]) ** 2)
    darcy_speed = np.sqrt(np.asarray(darcy["u"]) ** 2 + np.asarray(darcy["v"]) ** 2)
    stokes_speed = masked(stokes_speed, mask)
    darcy_speed = masked(darcy_speed, mask)
    speed_err = masked(np.abs(darcy_speed - stokes_speed), mask)

    p_vmax = max(float(np.nanmax(stokes_p)), float(np.nanmax(darcy_p)), 1.0e-12)
    p_emax = max(float(np.nanpercentile(p_err[mask == 1], 99.0)), 1.0e-12)
    s_vmax = max(
        float(np.nanpercentile(stokes_speed[mask == 1], 99.0)),
        float(np.nanpercentile(darcy_speed[mask == 1], 99.0)),
        1.0e-12,
    )
    s_emax = max(float(np.nanpercentile(speed_err[mask == 1], 99.0)), 1.0e-12)

    q_ref_common = max(float(stokes["q_ref"]), 1.0e-30)
    stokes_btc = btc_series(stokes, q_ref_common)
    darcy_btc = btc_series(darcy, q_ref_common)
    btc_max = max(float(np.max(stokes_btc)), float(np.max(darcy_btc)), 1.0e-12)
    stokes_rtd = rtd_series(np.asarray(stokes["times"]), stokes_btc)
    darcy_rtd = rtd_series(np.asarray(darcy["times"]), darcy_btc)
    rtd_max = max(float(np.max(stokes_rtd)), float(np.max(darcy_rtd)), 1.0e-12)

    title = "Darcy vs 2DStokes_FD  |  pressure, speed, BTC, RTD"
    if metadata and isinstance(metadata.get("title"), str):
        title = str(metadata["title"])
    summary_lines = format_metadata_lines(metadata)
    if not summary_lines and metadata and "label" in metadata:
        summary_lines = [f"label              = {metadata['label']}"]

    fig, axs = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axs[0, 0]
    im = ax.imshow(stokes_p, cmap="RdBu_r", vmin=0.0, vmax=p_vmax, **kw)
    ax.imshow(overlay, **kw)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("Pressure [Pa]")
    ax.set_title("Stokes Pressure")
    ax.set_xlabel("X  (flow ->)")
    ax.set_ylabel("Y")

    ax = axs[0, 1]
    im = ax.imshow(darcy_p, cmap="RdBu_r", vmin=0.0, vmax=p_vmax, **kw)
    ax.imshow(overlay, **kw)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("Pressure [Pa]")
    ax.set_title("Darcy Pressure")
    ax.set_xlabel("X  (flow ->)")
    ax.set_ylabel("Y")

    ax = axs[0, 2]
    im = ax.imshow(p_err, cmap="magma", vmin=0.0, vmax=p_emax, **kw)
    ax.imshow(overlay, **kw)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("|Dp| [Pa]")
    ax.set_title("Pressure Error")
    ax.set_xlabel("X  (flow ->)")
    ax.set_ylabel("Y")

    ax = axs[1, 0]
    im = ax.imshow(stokes_speed, cmap="viridis", vmin=0.0, vmax=s_vmax, **kw)
    ax.imshow(overlay, **kw)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("Speed [m/s]")
    ax.set_title("Stokes Speed Magnitude")
    ax.set_xlabel("X  (flow ->)")
    ax.set_ylabel("Y")

    ax = axs[1, 1]
    im = ax.imshow(darcy_speed, cmap="viridis", vmin=0.0, vmax=s_vmax, **kw)
    ax.imshow(overlay, **kw)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("Speed [m/s]")
    ax.set_title("Darcy Speed Magnitude")
    ax.set_xlabel("X  (flow ->)")
    ax.set_ylabel("Y")

    ax = axs[1, 2]
    im = ax.imshow(speed_err, cmap="magma", vmin=0.0, vmax=s_emax, **kw)
    ax.imshow(overlay, **kw)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("|Du| [m/s]")
    ax.set_title("Speed Error")
    ax.set_xlabel("X  (flow ->)")
    ax.set_ylabel("Y")

    ax = axs[2, 0]
    ax.plot(stokes["times"], stokes_btc, color="royalblue", linewidth=2, label="Stokes")
    ax.plot(darcy["times"], darcy_btc, color="darkorange", linewidth=2, marker="o", markersize=3, label="Darcy")
    ax.set_ylim(0.0, btc_max)
    ax.set_title("BTC at Right Face")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("C / C0")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend()

    ax = axs[2, 1]
    ax.plot(stokes["times"], stokes_rtd, color="royalblue", linewidth=2, label="Stokes RTD")
    ax.plot(darcy["times"], darcy_rtd, color="darkorange", linewidth=2, marker="o", markersize=3, label="Darcy RTD")
    ax.set_ylim(0.0, rtd_max)
    ax.set_title("Residence-Time Distribution")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("E(t) [1/s]")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend()

    ax = axs[2, 2]
    ax.axis("off")
    if summary_lines:
        ax.text(
            0.0,
            1.0,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )

    fig.savefig(plot_out, dpi=180)
    plt.close(fig)
