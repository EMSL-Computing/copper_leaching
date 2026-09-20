"""Binary pore map with flow-direction arrows (geom437).

Uses the same solid/pore greys as SI Fig. S2 and Stokes-reference velocity from
the local calibration folder. Inlet-connected channels get opaque arrows;
isolated pore regions use the same arrow colour at lower opacity.

Example:
    python plot_flow_direction.py
    python plot_flow_direction.py --config config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

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


from compute_reactive_accessibility_length import connected_pore_from_inlet
from darcy_vs_stokes.stokes_utils import load_stokes_reference

FIG_DPI = 600
ARROW_COLOR = "#2e6bb8"
SOLID_COLOR = "#b8b8b8"
PORE_COLOR = "#ededed"


def load_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    base = config_path.parent
    ref = (base / cfg["reference_dir"]).resolve()
    out_key = "flow_direction_figure_out"
    return {
        **cfg,
        "config_path": config_path.resolve(),
        "reference_dir": ref,
        "stokes_npz": ref / "calibrate_k_pore_run" / "stokes_reference.npz",
        "geometry_npy": (base / cfg["geometry_npy"]).resolve(),
        "flow_direction_figure_out": (base / cfg.get(out_key, "figures/flow_direction_arrows.png")).resolve(),
        "lx_m": float(cfg["lx_m"]),
        "ly_m": float(cfg["ly_m"]),
    }


def _plot_binary_pore_map(ax: plt.Axes, pore_flow: np.ndarray) -> None:
    """Solid matrix + pore space only (display frame), vector cells."""
    categories = np.where(pore_flow, 1, 0).T.astype(np.uint8)
    colors = (SOLID_COLOR, PORE_COLOR)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], len(colors))
    ny, nx = categories.shape
    patches_by_cat: list[list[Rectangle]] = [[], []]
    for row_idx in range(ny):
        row = categories[row_idx]
        col = 0
        while col < nx:
            cat = int(row[col])
            start = col
            col += 1
            while col < nx and int(row[col]) == cat:
                col += 1
            patches_by_cat[cat].append(Rectangle((start, row_idx), col - start, 1))
    for cat, color in enumerate(colors):
        if not patches_by_cat[cat]:
            continue
        ax.add_collection(
            PatchCollection(
                patches_by_cat[cat],
                facecolor=color,
                edgecolor="none",
                antialiased=False,
                rasterized=False,
            )
        )
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal", adjustable="box")


def _subsample_mask(mask: np.ndarray, step_y: int, step_x: int) -> np.ndarray:
    sub = np.zeros_like(mask, dtype=bool)
    sub[::step_y, ::step_x] = True
    return sub


def _unit_direction(u: np.ndarray, v: np.ndarray, *, min_speed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    speed = np.hypot(u, v)
    valid = speed >= min_speed
    u_hat = np.zeros_like(u, dtype=float)
    v_hat = np.zeros_like(v, dtype=float)
    u_hat[valid] = u[valid] / speed[valid]
    v_hat[valid] = v[valid] / speed[valid]
    return u_hat, v_hat, valid


def _quiver_layer(
    ax: plt.Axes,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    active: np.ndarray,
    *,
    alpha: float,
    zorder: int,
) -> None:
    if not np.any(active):
        return
    ax.quiver(
        x_mm[active],
        y_mm[active],
        u_hat[active],
        v_hat[active],
        angles="xy",
        scale_units="xy",
        scale=28.0,
        color=ARROW_COLOR,
        alpha=alpha,
        width=0.0016,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.6,
        linewidth=0.0,
        zorder=zorder,
    )


def plot_flow_direction(cfg: dict) -> Path:
    if not cfg["stokes_npz"].is_file():
        raise FileNotFoundError(cfg["stokes_npz"])
    if not cfg["geometry_npy"].is_file():
        raise FileNotFoundError(cfg["geometry_npy"])

    pore = np.load(cfg["geometry_npy"])
    pore_flow = np.asarray(pore > 0.5, dtype=bool)
    connected_flow = connected_pore_from_inlet(pore_flow, inlet="left")
    isolated_flow = pore_flow & ~connected_flow

    stokes = load_stokes_reference(cfg["stokes_npz"])
    u = np.asarray(stokes["u"], dtype=float)
    v = np.asarray(stokes["v"], dtype=float)
    if u.shape != pore_flow.T.shape:
        raise ValueError(f"Velocity shape {u.shape} does not match display frame {pore_flow.T.shape}")

    connected = connected_flow.T
    isolated = isolated_flow.T
    pore_display = pore_flow.T

    speed = np.hypot(u, v)
    pore_speed = speed[pore_display]
    min_speed = float(np.percentile(pore_speed[pore_speed > 0.0], 5.0)) if np.any(pore_speed > 0.0) else 1.0e-30

    step_x, step_y = 5, 4
    sample = _subsample_mask(pore_display, step_y, step_x)
    u_hat, v_hat, moving = _unit_direction(u, v, min_speed=min_speed)

    lx_mm = cfg["lx_m"] * 1000.0
    ly_mm = cfg["ly_m"] * 1000.0
    ny, nx = u.shape
    x_mm = (np.arange(nx, dtype=float) + 0.5) * lx_mm / nx
    y_mm = (np.arange(ny, dtype=float) + 0.5) * ly_mm / ny
    x_grid, y_grid = np.meshgrid(x_mm, y_mm)

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8.0, 2.0), dpi=FIG_DPI)
    _plot_binary_pore_map(ax, pore_flow)

    iso_active = sample & isolated & moving
    conn_active = sample & connected & moving
    _quiver_layer(ax, x_grid, y_grid, u_hat, v_hat, iso_active, alpha=0.20, zorder=4)
    _quiver_layer(ax, x_grid, y_grid, u_hat, v_hat, conn_active, alpha=0.95, zorder=5)

    ax.set_xlim(0.0, lx_mm)
    ax.set_ylim(0.0, ly_mm)
    ax.set_xlabel(r"$x$ (flow, 8 mm)")
    ax.set_ylabel(r"$y$ (2 mm)")
    ax.tick_params(length=3, width=0.6, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")

    out_png = cfg["flow_direction_figure_out"]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot geom437 flow-direction arrows on binary pore map")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    plot_flow_direction(load_config(args.config.resolve()))


if __name__ == "__main__":
    main()