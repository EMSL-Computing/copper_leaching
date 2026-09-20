"""Connected pore-space map for geom437 (chrysocolla_real000 panel-a style).

Grey solid matrix, white pore, whitish connected-pore interior, and a slightly
darker-than-reference blue outline on the inlet-connected boundary.

Example:
    python plot_connected_pore_map.py
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

FIG_DPI = 600
# Reference panel (a): grey solid 0.75, pore white 1.0, tab:blue contour (#1f77b4).
SOLID_RGB = (0.75, 0.75, 0.75)
PORE_RGB = (1.0, 1.0, 1.0)
CONNECTED_FILL_RGB = (0.95, 0.97, 1.0)
OUTLINE_COLOR = "#1868a8"


def load_config(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text())
    base = config_path.parent
    out_key = "connected_pore_figure_out"
    return {
        **cfg,
        "geometry_npy": (base / cfg["geometry_npy"]).resolve(),
        "connected_pore_figure_out": (
            base / cfg.get(out_key, "figures/connected_pore_map.png")
        ).resolve(),
        "lx_m": float(cfg["lx_m"]),
        "ly_m": float(cfg["ly_m"]),
    }


def _connected_pore_rgb(pore_flow: np.ndarray, connected_flow: np.ndarray) -> np.ndarray:
    """RGB map in display frame (reference panel-a look)."""
    pore = np.asarray(pore_flow, dtype=bool)
    connected = np.asarray(connected_flow, dtype=bool)
    pore_d = pore.T
    connected_d = connected.T
    rgb = np.zeros((*pore_d.shape, 3), dtype=float)
    rgb[~pore_d] = SOLID_RGB
    rgb[pore_d & ~connected_d] = PORE_RGB
    rgb[connected_d] = CONNECTED_FILL_RGB
    return rgb


def plot_connected_pore_map(cfg: dict) -> Path:
    if not cfg["geometry_npy"].is_file():
        raise FileNotFoundError(cfg["geometry_npy"])

    pore = np.load(cfg["geometry_npy"])
    pore_flow = np.asarray(pore > 0.5, dtype=bool)
    connected_flow = connected_pore_from_inlet(pore_flow, inlet="left")
    rgb = _connected_pore_rgb(pore_flow, connected_flow)
    connected_disp = connected_flow.T.astype(float)

    lx_mm = cfg["lx_m"] * 1000.0
    ly_mm = cfg["ly_m"] * 1000.0
    ny, nx = rgb.shape[:2]

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
    ax.set_title("Inlet-connected pore space", loc="left", fontsize=10, pad=6)
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

    out_png = cfg["connected_pore_figure_out"]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_png}")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot geom437 connected pore map")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    plot_connected_pore_map(load_config(args.config.resolve()))


if __name__ == "__main__":
    main()