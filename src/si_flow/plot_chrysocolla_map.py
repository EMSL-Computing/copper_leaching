"""Chrysocolla inventory map for geom437 (chrysocolla_real000 panel-b style).

Places chrysocolla on 25% of solid cells (mineral seed 42, paper placement protocol)
and exports a 600 dpi figure.

Example:
    python plot_chrysocolla_map.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

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


from generate_chrysocolla_accessibility_realizations import (
    choose_chrysocolla_cells,
    neighbor_distance_from_connected_pore,
)

FIG_DPI = 600
SOLID_GREY = 0.65
PORE_WHITE = 1.0
# Match manuscript schematic_setup palette (plot_schematic_setup_layers.COLOR_CHRYS).
CHRYS_COLOR = "#3d8c40"


def load_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    base = config_path.parent
    return {
        **cfg,
        "geometry_npy": (base / cfg["geometry_npy"]).resolve(),
        "chrysocolla_figure_out": (
            base / cfg.get("chrysocolla_figure_out", "figures/chrysocolla_map.png")
        ).resolve(),
        "chrysocolla_solid_fraction": float(cfg.get("chrysocolla_solid_fraction", 0.25)),
        "chrysocolla_mineral_seed": int(cfg.get("chrysocolla_mineral_seed", 42)),
        "chrysocolla_smooth_passes": int(cfg.get("chrysocolla_smooth_passes", 4)),
        "chrysocolla_accessibility_bias": float(cfg.get("chrysocolla_accessibility_bias", 0.0)),
        "lx_m": float(cfg["lx_m"]),
        "ly_m": float(cfg["ly_m"]),
    }


def build_chrysocolla_mask(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Fixed-inventory chrysocolla on solid cells (paper placement protocol)."""
    solid = img == 0.0
    n_solid = int(np.count_nonzero(solid))
    n_chrys = int(round(cfg["chrysocolla_solid_fraction"] * n_solid))
    dist_cells, _connected = neighbor_distance_from_connected_pore(img, inlet="left")
    rng = np.random.default_rng(cfg["chrysocolla_mineral_seed"])
    return choose_chrysocolla_cells(
        img,
        dist_cells,
        n_chrys,
        rng,
        smooth_passes=cfg["chrysocolla_smooth_passes"],
        accessibility_bias=cfg["chrysocolla_accessibility_bias"],
    )


def plot_chrysocolla_map(cfg: dict) -> Path:
    if not cfg["geometry_npy"].is_file():
        raise FileNotFoundError(cfg["geometry_npy"])

    img = np.load(cfg["geometry_npy"])
    if img.ndim != 2:
        raise ValueError(f"Expected 2D geometry, got shape {img.shape}")
    chrys_mask = build_chrysocolla_mask(img, cfg)

    solid_disp = np.where(img > 0.5, PORE_WHITE, SOLID_GREY).T
    chrys_disp = np.where(chrys_mask, 1.0, np.nan).T

    lx_mm = cfg["lx_m"] * 1000.0
    ly_mm = cfg["ly_m"] * 1000.0
    ny, nx = solid_disp.shape

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8.0, 2.0), dpi=FIG_DPI)
    chrys_cmap = ListedColormap([CHRYS_COLOR])
    ax.imshow(solid_disp, origin="lower", interpolation="nearest", cmap="gray", vmin=0, vmax=1)
    ax.imshow(chrys_disp, origin="lower", interpolation="nearest", cmap=chrys_cmap, vmin=0, vmax=1)

    frac = cfg["chrysocolla_solid_fraction"]
    ax.set_xlabel(r"$x$ (flow, 8 mm)")
    ax.set_ylabel(r"$y$ (2 mm)")
    ax.set_title(
        rf"Chrysocolla inventory ({frac:.0%} solid VF, mineral seed {cfg['chrysocolla_mineral_seed']})",
        loc="left",
        fontsize=10,
        pad=6,
    )
    ax.tick_params(length=3, width=0.6, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#666666")

    xticks = np.linspace(0, nx - 1, 9)
    yticks = np.linspace(0, ny - 1, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{x * lx_mm / (nx - 1):.0f}" for x in xticks])
    ax.set_yticklabels([f"{y * ly_mm / (ny - 1):.1f}" for y in yticks])
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_aspect("equal", adjustable="box")

    out_png = cfg["chrysocolla_figure_out"]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot geom437 chrysocolla map")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    plot_chrysocolla_map(load_config(args.config.resolve()))


if __name__ == "__main__":
    main()