"""SI figures: Darcy-Stokes flow maps + 1:1 scatters and outlet BTC (geom437).

Example:
    python plot_darcy_stokes.py
    python plot_darcy_stokes.py --flow-only
    python plot_darcy_stokes.py --btc-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

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


from darcy_vs_stokes.darcy_stokes_compare import btc_series, compute_metrics, masked, solids_overlay
from darcy_vs_stokes.darcy_utils import load_darcy_case
from darcy_vs_stokes.stokes_utils import load_stokes_reference

FIG_DPI = 300
SPEED_M_S_TO_UM_S = 1.0e6


def load_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    base = config_path.parent
    ref = (base / cfg["reference_dir"]).resolve()
    return {
        **cfg,
        "config_path": config_path.resolve(),
        "reference_dir": ref,
        "stokes_npz": ref / "calibrate_k_pore_run/stokes_reference.npz",
        "darcy_h5": ref / "simulate_darcy_run/darcy_calibrated.h5",
        "geometry_npy": (base / cfg["geometry_npy"]).resolve(),
        "geometry_h5": (base / cfg["geometry_h5"]).resolve(),
        "flow_figure_out": (base / cfg["flow_figure_out"]).resolve(),
        "btc_figure_out": (base / cfg["btc_figure_out"]).resolve(),
    }


def load_cases(cfg: dict) -> tuple[dict, dict, dict[str, float]]:
    if not cfg["stokes_npz"].is_file():
        raise FileNotFoundError(cfg["stokes_npz"])
    if not cfg["darcy_h5"].is_file():
        raise FileNotFoundError(cfg["darcy_h5"])

    stokes = load_stokes_reference(cfg["stokes_npz"])
    mask = np.asarray(stokes["mask"])
    dP = float(cfg["dP_Pa"])
    darcy = load_darcy_case(
        cfg["darcy_h5"],
        target_time_s=float(np.asarray(stokes["times"])[-1]),
        mask=mask,
        dx=float(cfg["dx_m"]),
        pressure_in=float(cfg["p_out_Pa"]) + dP,
        pressure_out=float(cfg["p_out_Pa"]),
    )
    return stokes, darcy, compute_metrics(stokes, darcy)


def _strip_map_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _fluid_pairs(stokes_field: np.ndarray, darcy_field: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fluid = mask == 1
    x = np.asarray(stokes_field, dtype=float)[fluid]
    y = np.asarray(darcy_field, dtype=float)[fluid]
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((x - y) ** 2)))


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(np.sum((x - y) ** 2))
    ss_tot = float(np.sum((x - np.mean(x)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")


def _plot_map_panel(ax: plt.Axes, field: np.ndarray, *, cmap: str, vmin: float, vmax: float, overlay: np.ndarray, kw: dict) -> plt.AxesImage:
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#1a1a1a")
    im = ax.imshow(field, cmap=cmap_obj, vmin=vmin, vmax=vmax, **kw)
    ax.imshow(overlay, **kw)
    _strip_map_axes(ax)
    return im


def _plot_one_to_one_scatter(
    ax: plt.Axes,
    stokes_vals: np.ndarray,
    darcy_vals: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    units: str,
    show_ylabel: bool = True,
) -> None:
    ax.scatter(stokes_vals, darcy_vals, s=4, alpha=0.25, color="0.35", edgecolors="none", rasterized=True)
    lo = float(min(np.min(stokes_vals), np.min(darcy_vals)))
    hi = float(max(np.max(stokes_vals), np.max(darcy_vals)))
    pad = 0.04 * max(hi - lo, 1.0e-30)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], color="crimson", lw=1.2, ls="--", zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Stokes {xlabel}", labelpad=4)
    if show_ylabel:
        ax.set_ylabel(f"Darcy {ylabel}", labelpad=6)
    else:
        ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=9, pad=6)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.tick_params(axis="both", pad=2)

    rmse = _rmse(stokes_vals, darcy_vals)
    r2 = _r2(stokes_vals, darcy_vals)
    ax.text(
        0.04, 0.96,
        f"RMSE = {rmse:.3e} {units}\n$R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )


def _add_row_colorbar(fig: plt.Figure, im: plt.AxesImage, anchor_ax: plt.Axes, label: str) -> None:
    pos = anchor_ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(label, fontsize=8)


def plot_flow_panels(
    stokes: dict,
    darcy: dict,
    out_png: Path,
    *,
    lx_m: float,
    ly_m: float,
    metrics: dict[str, float] | None = None,
) -> None:
    del metrics  # metrics shown on scatter panels
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    mask = np.asarray(stokes["mask"])
    overlay = solids_overlay(mask)
    extent = [0.0, lx_m * 1e3, 0.0, ly_m * 1e3]
    kw = dict(origin="lower", extent=extent, interpolation="nearest")

    stokes_p = masked(np.asarray(stokes["p"]), mask)
    darcy_p = masked(np.asarray(darcy["p"]), mask)
    stokes_speed = masked(np.sqrt(np.asarray(stokes["u"]) ** 2 + np.asarray(stokes["v"]) ** 2), mask) * SPEED_M_S_TO_UM_S
    darcy_speed = masked(np.sqrt(np.asarray(darcy["u"]) ** 2 + np.asarray(darcy["v"]) ** 2), mask) * SPEED_M_S_TO_UM_S

    p_vmax = max(float(np.nanmax(stokes_p)), float(np.nanmax(darcy_p)), 1.0e-12)
    fluid = mask == 1
    s_vmax = max(
        float(np.nanpercentile(stokes_speed[fluid], 99.0)),
        float(np.nanpercentile(darcy_speed[fluid], 99.0)),
        1.0e-12,
    )

    fig = plt.figure(figsize=(9, 6.3))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.65, 1.35], hspace=0.03)
    maps = outer[0].subgridspec(2, 2, hspace=0.2, wspace=0.06)
    scatters = outer[1].subgridspec(1, 2, wspace=0.10)

    ax_ps = fig.add_subplot(maps[0, 0])
    ax_pd = fig.add_subplot(maps[0, 1])
    ax_ss = fig.add_subplot(maps[1, 0])
    ax_sd = fig.add_subplot(maps[1, 1])
    ax_p_sc = fig.add_subplot(scatters[0, 0])
    ax_s_sc = fig.add_subplot(scatters[0, 1])

    for ax in (ax_ps, ax_pd):
        ax.set_anchor("S")
    for ax in (ax_ss, ax_sd):
        ax.set_anchor("N")

    im_p = _plot_map_panel(ax_ps, stokes_p, cmap="RdBu_r", vmin=0.0, vmax=p_vmax, overlay=overlay, kw=kw)
    _plot_map_panel(ax_pd, darcy_p, cmap="RdBu_r", vmin=0.0, vmax=p_vmax, overlay=overlay, kw=kw)
    im_s = _plot_map_panel(ax_ss, stokes_speed, cmap="viridis", vmin=0.0, vmax=s_vmax, overlay=overlay, kw=kw)
    _plot_map_panel(ax_sd, darcy_speed, cmap="viridis", vmin=0.0, vmax=s_vmax, overlay=overlay, kw=kw)

    ax_ps.text(0.0, 1.005, "(a) Pressure", transform=ax_ps.transAxes, ha="left", va="bottom", fontsize=9)
    ax_ss.text(0.0, 1.005, "(b) Speed magnitude", transform=ax_ss.transAxes, ha="left", va="bottom", fontsize=9)

    p_x, p_y = _fluid_pairs(stokes_p, darcy_p, mask)
    s_x, s_y = _fluid_pairs(stokes_speed, darcy_speed, mask)
    _plot_one_to_one_scatter(
        ax_p_sc, p_x, p_y,
        xlabel="pressure [Pa]", ylabel="pressure [Pa]",
        title="(c) Pressure validation", units="Pa",
    )
    _plot_one_to_one_scatter(
        ax_s_sc, s_x, s_y,
        xlabel=r"speed [$\mu$m s$^{-1}$]", ylabel=r"speed [$\mu$m s$^{-1}$]",
        title="(d) Speed validation", units=r"$\mu$m s$^{-1}$",
        show_ylabel=False,
    )
    for ax in (ax_p_sc, ax_s_sc):
        ax.set_aspect("auto")
        ax.set_box_aspect(0.8)  # height / width; smaller = wider
        
    fig.subplots_adjust(left=0.09, right=0.86, top=0.92, bottom=0.08)
    _add_row_colorbar(fig, im_p, ax_pd, "Pressure [Pa]")
    _add_row_colorbar(fig, im_s, ax_sd, r"Speed [$\mu$m s$^{-1}$]")

    axis_lbl = r"$x$ (flow, 8 mm) $\times$ $y$ (2 mm)"
    for ax, title in ((ax_ps, "Stokes reference"), (ax_pd, "Calibrated Darcy")):
        pos = ax.get_position()
        cx = 0.5 * (pos.x0 + pos.x1)
        fig.text(cx, pos.y1 + 0.018, f"{title}\n{axis_lbl}", ha="center", va="bottom", fontsize=9, linespacing=1.15)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_btc_panel(
    stokes: dict,
    darcy: dict,
    out_png: Path,
    *,
    metrics: dict[str, float] | None = None,
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    q_ref = max(float(stokes["q_ref"]), float(darcy["q_ref"]), 1.0e-30)
    stokes_btc = btc_series(stokes, q_ref)
    darcy_btc = btc_series(darcy, q_ref)
    times_s = np.asarray(stokes["times"], dtype=float)
    darcy_times = np.asarray(darcy["times"], dtype=float)
    darcy_btc_on_stokes = np.interp(
        times_s, darcy_times, darcy_btc,
        left=float(darcy_btc[0]), right=float(darcy_btc[-1]),
    )

    pulse = float(stokes["pulse_duration"]) if "pulse_duration" in stokes else np.nan
    btc_max = max(float(np.max(stokes_btc)), float(np.max(darcy_btc_on_stokes)), 1.0e-12)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if np.isfinite(pulse) and pulse > 0.0:
        ax.axvspan(0.0, pulse, color="royalblue", alpha=0.10, label="Inlet pulse")
        ax.axvline(pulse, color="royalblue", ls="--", lw=0.8, alpha=0.6)

    ax.plot(times_s, stokes_btc, color="royalblue", lw=2.0, label="Stokes")
    ax.plot(times_s, darcy_btc_on_stokes, color="darkorange", lw=2.0, ls="--", label="Calibrated Darcy")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, btc_max * 1.05)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Outlet BTC, $C/C_0$")
    ax.set_title("(a) Advective tracer breakthrough at outlet", loc="left", fontsize=10)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")

    if metrics:
        ax.text(
            0.99, 0.02, f"BTC RMSE / peak = {metrics['btc_rel_rmse']:.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35",
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_all(cfg: dict, *, flow: bool = True, btc: bool = True) -> None:
    stokes, darcy, metrics = load_cases(cfg)
    if flow:
        plot_flow_panels(stokes, darcy, cfg["flow_figure_out"], lx_m=float(cfg["lx_m"]), ly_m=float(cfg["ly_m"]), metrics=metrics)
    if btc:
        plot_btc_panel(stokes, darcy, cfg["btc_figure_out"], metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--flow-only", action="store_true")
    parser.add_argument("--btc-only", action="store_true")
    args = parser.parse_args()

    flow = btc = True
    if args.flow_only:
        btc = False
    if args.btc_only:
        flow = False

    cfg = load_config(args.config.resolve())
    plot_all(cfg, flow=flow, btc=btc)


if __name__ == "__main__":
    main()