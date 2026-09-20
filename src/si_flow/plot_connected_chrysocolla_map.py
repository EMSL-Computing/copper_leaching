"""Combined connected-pore + chrysocolla map for geom437.

Single-panel overlay: whitish inlet-connected pores with blue outline and
schematic-green chrysocolla on solid cells (25% VF, mineral seed 42).

Example:
    python plot_connected_chrysocolla_map.py
"""

from __future__ import annotations

import argparse
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


from compute_reactive_accessibility_length import connected_pore_from_inlet
from plot_chrysocolla_map import build_chrysocolla_mask

FIG_DPI = 600
SOLID_RGB = (0.75, 0.75, 0.75)
PORE_RGB = (1.0, 1.0, 1.0)
CONNECTED_FILL_RGB = (0.95, 0.97, 1.0)
CHRYS_RGB = (0x3D / 255.0, 0x8C / 255.0, 0x40 / 255.0)
OUTLINE_COLOR = "#1868a8"


def load_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    base = config_path.parent
    return {
        **cfg,
        "geometry_npy": (base / cfg["geometry_npy"]).resolve(),
        "connected_chrysocolla_figure_out": (
            base / cfg.get(
                "connected_chrysocolla_figure_out",
                "figures/connected_chrysocolla_map.png",
            )
        ).resolve(),
        "chrysocolla_solid_fraction": float(cfg.get("chrysocolla_solid_fraction", 0.25)),
        "chrysocolla_mineral_seed": int(cfg.get("chrysocolla_mineral_seed", 42)),
        "chrysocolla_smooth_passes": int(cfg.get("chrysocolla_smooth_passes", 4)),
        "chrysocolla_accessibility_bias": float(cfg.get("chrysocolla_accessibility_bias", 0.0)),
        "lx_m": float(cfg["lx_m"]),
        "ly_m": float(cfg["ly_m"]),
    }


def _combined_rgb(
    pore_flow: np.ndarray,
    connected_flow: np.ndarray,
    chrys_mask: np.ndarray,
) -> np.ndarray:
    """Display-frame RGB: solid, chrysocolla, isolated pore, connected pore."""
    pore_d = np.asarray(pore_flow, dtype=bool).T
    conn_d = np.asarray(connected_flow, dtype=bool).T
    chrys_d = np.asarray(chrys_mask, dtype=bool).T
    rgb = np.zeros((*pore_d.shape, 3), dtype=float)
    rgb[~pore_d] = SOLID_RGB
    rgb[chrys_d & ~pore_d] = CHRYS_RGB
    rgb[pore_d] = PORE_RGB
    rgb[conn_d] = CONNECTED_FILL_RGB
    return rgb


def plot_connected_chrysocolla_map(cfg: dict) -> Path:
    if not cfg["geometry_npy"].is_file():
        raise FileNotFoundError(cfg["geometry_npy"])

    img = np.load(cfg["geometry_npy"])
    pore_flow = np.asarray(img > 0.5, dtype=bool)
    connected_flow = connected_pore_from_inlet(pore_flow, inlet="left")
    chrys_mask = build_chrysocolla_mask(img, cfg)
    rgb = _combined_rgb(pore_flow, connected_flow, chrys_mask)
    connected_disp = connected_flow.T.astype(float)

    lx_mm = cfg["lx_m"] * 1000.0
    ly_mm = cfg["ly_m"] * 1000.0
    ny, nx = rgb.shape[:2]
    frac = cfg["chrysocolla_solid_fraction"]

    sns.set_theme(style="white", context="paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8.0, 2.0), dpi=FIG_DPI)
    ax.imshow(rgb, origin="lower", interpolation="nearest", aspect="equal")
    ax.contour(
        connected_disp,
        levels=[0.5],
        colors=OUTLINE_COLOR,
        linewidths=0.65,
        zorder=3,
    )

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xlabel(r"$x$ (flow, 8 mm)")
    ax.set_ylabel(r"$y$ (2 mm)")
    ax.set_title(
        rf"Inlet-connected pore space and chrysocolla ({frac:.0%} solid VF)",
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

    out_png = cfg["connected_chrysocolla_figure_out"]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot geom437 connected pore + chrysocolla overlay")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    plot_connected_chrysocolla_map(load_config(args.config.resolve()))


if __name__ == "__main__":
    main()