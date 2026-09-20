"""Export individual schematic_setup panel PNGs for compositing in Illustrator.

Example:
    python plot_schematic_setup_layers.py --tracer-test
    python plot_schematic_setup_layers.py --geom-seed 437 --out-dir my_layers/
    python plot_schematic_setup_layers.py --composed --tracer-test  # optional montage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy import ndimage as ndi

from _repo import REPO_ROOT

_ARCHIVE_VIZ = REPO_ROOT / "archive" / "exploratory_viz"
if str(_ARCHIVE_VIZ) not in sys.path:
    sys.path.insert(0, str(_ARCHIVE_VIZ))

from animate_tracer import load_velocity
from build_pflotran_hdf5_from_npy import load_geometry_npy
from compute_reactive_accessibility_length import (
    accessible_cu_fraction,
    compute_accessibility_stats,
    connected_pore_from_inlet,
    load_mask_from_h5,
    mineral_access_distance_map,
)

HERE = REPO_ROOT
DEFAULT_MANIFEST = (
    REPO_ROOT / "archive/Chip_Designs/bean_bucket_network/stage1_realizations/stage1_manifest.json"
)
DEFAULT_OUT = REPO_ROOT / "src/si_flow/figures/schematic_layers"
DEFAULT_TRACER_GEOM_NPY = (
    REPO_ROOT / "archive/Chip_Designs/bean_bucket_network/tracer_test/geometry/network_correct.npy"
)
DEFAULT_TRACER_GEOM_H5 = (
    REPO_ROOT
    / "archive/Chip_Designs/bean_bucket_network/tracer_test/chrysocolla_realizations/chrysocolla_real000.h5"
)
DEFAULT_TRACER_DARCY_H5 = (
    REPO_ROOT
    / "archive/Chip_Designs/bean_bucket_network/tracer_test/pflotran_out/stokes_reference/simulate_darcy_run/darcy_calibrated.h5"
)
DEFAULT_TRACER_OUT = HERE / "Chip_Designs/bean_bucket_network/tracer_test/schematic_layers"
DEFAULT_DX_M = 2e-5

# GPT-inspired palette (colorblind-safe)
COLOR_PORE = np.array([0xE8, 0xE8, 0xE8], dtype=np.uint8)
COLOR_SOLID = np.array([0xB0, 0xB0, 0xB0], dtype=np.uint8)
COLOR_CONNECTED = np.array([0x7E, 0xC8, 0xC8], dtype=np.uint8)
COLOR_ISOLATED_PORE = np.array([0xF5, 0xF5, 0xF5], dtype=np.uint8)
COLOR_CHRYS = np.array([0x3D, 0x8C, 0x40], dtype=np.uint8)
COLOR_FLOWPATH = np.array([0x21, 0x66, 0xAC], dtype=np.uint8)
COLOR_LR_LINE = np.array([0x20, 0x20, 0x20], dtype=np.uint8)
COLOR_PORE_D = np.array([0x43, 0x93, 0xC3], dtype=np.uint8)
COLOR_CHRYS_DARK = np.array([0x2D, 0x6A, 0x30], dtype=np.uint8)
COLOR_CHRYS_LIGHT = np.array([0xA8, 0xD5, 0xA2], dtype=np.uint8)

DISPLAY_SMOOTH_RADIUS = 1
DISPLAY_UPSCALE = 4
DISPLAY_GAUSS_SIGMA = 0.65


def load_manifest_entry(
    manifest_path: Path,
    *,
    realization_index: int | None = None,
    geom_seed: str | int | None = None,
    candidate_id: str | None = None,
) -> dict:
    """Return one stage-1 manifest entry."""
    payload = json.loads(manifest_path.read_text())
    entries = payload["entries"]
    if realization_index is not None:
        for entry in entries:
            if int(entry["realization_index"]) == int(realization_index):
                return entry
        raise KeyError(f"realization_index={realization_index} not in {manifest_path}")
    if geom_seed is not None:
        key = str(geom_seed)
        for entry in entries:
            if str(entry["geom_seed"]) == key:
                return entry
        raise KeyError(f"geom_seed={geom_seed} not in {manifest_path}")
    if candidate_id is not None:
        for entry in entries:
            if entry["candidate_id"] == candidate_id:
                return entry
        raise KeyError(f"candidate_id={candidate_id} not in {manifest_path}")
    raise ValueError("Specify one of realization_index, geom_seed, or candidate_id")


def resolve_mineral_threshold(h5_path: Path, requested: float | None = None) -> float:
    """Pick mineral VF threshold; tracer geometry uses continuous VF (max < 0.5)."""
    if requested is not None:
        return float(requested)
    with h5py.File(h5_path, "r") as f:
        vf = np.asarray(f["Chrysocolla2_VF"], dtype=np.float64)
    if float(vf.max()) <= 0.5:
        return 0.0
    return 0.5


def flow_frame_masks(
    geom_npy: Path,
    h5_path: Path,
    *,
    mineral_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pore and chrysocolla masks in flow frame (ny, nx)."""
    img = load_geometry_npy(geom_npy)
    flow = np.asarray(img.T > 0.5, dtype=bool)
    thresh = resolve_mineral_threshold(h5_path, mineral_threshold)
    mineral = np.asarray(
        load_mask_from_h5(
            h5_path,
            dataset="Chrysocolla2_VF",
            threshold=thresh,
            shape_hint=img.shape,
        ).T,
        dtype=bool,
    )
    solid = ~flow
    if mineral.shape != flow.shape:
        raise ValueError(f"Shape mismatch flow {flow.shape} mineral {mineral.shape}")
    if np.any(mineral & flow):
        raise ValueError("Mineral mask intersects pore space")
    return flow, mineral, solid


def soften_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological open/close for schematic display only (not for metrics)."""
    if radius <= 0:
        return mask
    s = 2 * radius + 1
    struct = np.ones((s, s), dtype=bool)
    return ndi.binary_closing(ndi.binary_opening(mask, structure=struct), structure=struct)


def upscale_rgb_smooth(rgb: np.ndarray, scale: int) -> np.ndarray:
    """Bilinear upscale to reduce blockiness in schematics."""
    if scale <= 1:
        return rgb
    return ndi.zoom(rgb.astype(np.float32), (scale, scale, 1), order=1)


def _rgb_from_uint8(rgba: np.ndarray) -> np.ndarray:
    """Drop alpha; return float RGB in [0, 1]."""
    return rgba[..., :3].astype(np.float32) / 255.0


def _blend_layers(layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Alpha-composite float RGB layers (bottom to top)."""
    out = np.zeros((*layers[0][0].shape[:2], 3), dtype=np.float32)
    for rgb, alpha in layers:
        a = np.clip(alpha[..., None], 0.0, 1.0)
        out = out * (1.0 - a) + rgb * a
    return out


def smooth_field(mask: np.ndarray, sigma: float = DISPLAY_GAUSS_SIGMA) -> np.ndarray:
    """Gaussian-softened [0,1] field for anti-aliased schematic edges."""
    if sigma <= 0:
        return mask.astype(np.float32)
    return ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma)


def schematic_rgb_geometry(solid: np.ndarray, pore: np.ndarray) -> np.ndarray:
    """Anti-aliased pore/solid schematic."""
    pore_f = smooth_field(pore)
    solid_f = 1.0 - pore_f
    pore_rgb = COLOR_PORE.astype(np.float32) / 255.0
    solid_rgb = COLOR_SOLID.astype(np.float32) / 255.0
    return _blend_layers(
        [
            (np.broadcast_to(solid_rgb, (*solid.shape, 3)), solid_f),
            (np.broadcast_to(pore_rgb, (*pore.shape, 3)), pore_f),
        ]
    )


def schematic_rgb_connected(pore: np.ndarray, connected: np.ndarray) -> np.ndarray:
    """Anti-aliased inlet-connected mask schematic."""
    pore_f = smooth_field(pore)
    isolated_f = smooth_field(pore & ~connected)
    connected_f = smooth_field(connected)
    solid_f = np.clip(1.0 - pore_f, 0.0, 1.0)
    return _blend_layers(
        [
            (np.broadcast_to(COLOR_SOLID.astype(np.float32) / 255.0, (*pore.shape, 3)), solid_f),
            (np.broadcast_to(COLOR_ISOLATED_PORE.astype(np.float32) / 255.0, (*pore.shape, 3)), isolated_f),
            (np.broadcast_to(COLOR_CONNECTED.astype(np.float32) / 255.0, (*pore.shape, 3)), connected_f),
        ]
    )


def schematic_rgb_mineral(solid: np.ndarray, mineral: np.ndarray) -> np.ndarray:
    """Smoothed matrix with crisp chrysocolla voxels (schematic style)."""
    rgb = schematic_rgb_geometry(solid, ~solid)
    rgb[mineral] = COLOR_CHRYS.astype(np.float32) / 255.0
    return rgb


def imshow_schematic(ax, rgb: np.ndarray, *, extent_mm: tuple[float, float, float, float]) -> None:
    """Draw supersampled domain with mm axes."""
    rgb_hi = upscale_rgb_smooth(rgb, DISPLAY_UPSCALE)
    ax.imshow(
        rgb_hi,
        origin="lower",
        extent=extent_mm,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_xlim(extent_mm[0], extent_mm[1])
    ax.set_ylim(extent_mm[2], extent_mm[3])


def load_darcy_velocity_flow_frame(darcy_h5: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load steady Darcy velocity in flow frame (ny, nx) [m/s]."""
    vx, vy, _ = load_velocity(darcy_h5)
    if vx is None or vy is None:
        raise ValueError(f"No velocity fields in {darcy_h5}")
    return np.asarray(vx.T, dtype=np.float64), np.asarray(vy.T, dtype=np.float64)


def plot_darcy_flow_on_ax(
    ax,
    ux: np.ndarray,
    uy: np.ndarray,
    flow_mask: np.ndarray,
    extent_mm: tuple[float, float, float, float],
    *,
    color: str = "#2166AC",
    density: float = 1.15,
) -> None:
    """Overlay calibrated Darcy streamlines with direction arrows."""
    ny, nx = ux.shape
    x0, x1, y0, y1 = extent_mm
    x_mm = np.linspace(x0 + 0.5 * (x1 - x0) / nx, x1 - 0.5 * (x1 - x0) / nx, nx)
    y_mm = np.linspace(y0 + 0.5 * (y1 - y0) / ny, y1 - 0.5 * (y1 - y0) / ny, ny)
    speed = np.sqrt(ux**2 + uy**2)
    active = flow_mask & (speed > 0.0)
    if not np.any(active):
        return
    speed_thr = float(np.percentile(speed[active], 8.0))
    mask = ~flow_mask | (speed < speed_thr)
    ax.streamplot(
        x_mm,
        y_mm,
        np.ma.array(ux, mask=mask),
        np.ma.array(uy, mask=mask),
        color=color,
        linewidth=1.7,
        density=density,
        arrowsize=1.15,
        arrowstyle="->",
        zorder=5,
    )


def path_polyline_mm(path_mask: np.ndarray, extent_mm: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Centerline polyline (mm) through BFS path cells, subsampled for smooth overlay."""
    ys, xs = np.where(path_mask)
    if ys.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    step = max(1, xs.size // 80)
    xs = xs[::step]
    ys = ys[::step]
    x0, x1, y0, y1 = extent_mm
    ny, nx = path_mask.shape
    xmm = x0 + (xs + 0.5) * (x1 - x0) / nx
    ymm = y0 + (ys + 0.5) * (y1 - y0) / ny
    return xmm, ymm


def save_panel_figure(
    path: Path,
    rgb: np.ndarray,
    extent_mm: tuple[float, float, float, float],
    *,
    dpi: int = 220,
    figsize: tuple[float, float] = (8.0, 2.2),
) -> None:
    """Save one display-smoothed schematic panel with mm axes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    imshow_schematic(ax, rgb, extent_mm=extent_mm)
    ax.set_xlabel("Flow direction x [mm]")
    ax.set_ylabel("y [mm]")
    ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def save_panel_a_darcy_flow(
    path: Path,
    ctx: dict,
    *,
    dpi: int = 220,
) -> None:
    """Panel (a): smoothed geometry with calibrated Darcy streamlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = schematic_rgb_geometry(ctx["solid_disp"], ctx["pore_disp"])
    extent = ctx["extent_mm"]
    fig, ax = plt.subplots(figsize=(8.0, 2.2), dpi=dpi)
    imshow_schematic(ax, rgb, extent_mm=extent)
    if ctx.get("ux") is not None and ctx.get("uy") is not None:
        plot_darcy_flow_on_ax(ax, ctx["ux"], ctx["uy"], ctx["connected"], extent)
    ax.set_xlabel("Flow direction x [mm]")
    ax.set_ylabel("y [mm]")
    ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_rgba(
    path: Path,
    rgba: np.ndarray,
    dpi: int = 300,
    *,
    min_width_px: int = 0,
) -> None:
    """Save HxWx4 uint8 image without matplotlib axes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = rgba.shape[:2]
    if min_width_px > 0 and nx < min_width_px:
        scale = int(np.ceil(min_width_px / nx))
        rgba = np.repeat(np.repeat(rgba, scale, axis=0), scale, axis=1)
        ny, nx = rgba.shape[:2]
    fig_w = nx / dpi
    fig_h = ny / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=dpi, pad_inches=0, transparent=True)
    plt.close(fig)


def panel_a_geometry(solid: np.ndarray, pore: np.ndarray) -> np.ndarray:
    """Gray solid matrix and light pore space."""
    rgba = np.zeros((*solid.shape, 4), dtype=np.uint8)
    rgba[pore] = (*COLOR_PORE.tolist(), 255)
    rgba[solid] = (*COLOR_SOLID.tolist(), 255)
    return rgba


def panel_a_flowpath(connected: np.ndarray, path_mask: np.ndarray) -> np.ndarray:
    """Transparent layer with blue path cells on connected pores."""
    rgba = np.zeros((*connected.shape, 4), dtype=np.uint8)
    rgba[path_mask] = (*COLOR_FLOWPATH.tolist(), 220)
    return rgba


def inlet_outlet_path_mask(connected: np.ndarray) -> np.ndarray:
    """BFS path from inlet column to outlet column through connected pores."""
    ny, nx = connected.shape
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    queue: list[tuple[int, int]] = []
    for iy in range(ny):
        if connected[iy, 0]:
            queue.append((iy, 0))
            prev[(iy, 0)] = None
    if not queue:
        return np.zeros_like(connected, dtype=bool)

    goal: tuple[int, int] | None = None
    head = 0
    while head < len(queue):
        cell = queue[head]
        head += 1
        iy, ix = cell
        if ix == nx - 1:
            goal = cell
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < ny and 0 <= jx < nx and connected[jy, jx]:
                nxt = (jy, jx)
                if nxt not in prev:
                    prev[nxt] = cell
                    queue.append(nxt)

    path = np.zeros_like(connected, dtype=bool)
    if goal is None:
        return path
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path[cur] = True
        cur = prev[cur]
    return path


def panel_b_connected_mask(pore: np.ndarray, connected: np.ndarray) -> np.ndarray:
    """Cyan inlet-connected pores; pale isolated pore space."""
    rgba = np.zeros((*pore.shape, 4), dtype=np.uint8)
    isolated = pore & ~connected
    rgba[isolated] = (*COLOR_ISOLATED_PORE.tolist(), 255)
    rgba[connected] = (*COLOR_CONNECTED.tolist(), 255)
    rgba[~pore] = (*COLOR_SOLID.tolist(), 255)
    return rgba


def panel_c_mineral(solid: np.ndarray, mineral: np.ndarray) -> np.ndarray:
    """Chrysocolla voxels on gray solid."""
    rgba = np.zeros((*solid.shape, 4), dtype=np.uint8)
    rgba[solid & ~mineral] = (*COLOR_SOLID.tolist(), 255)
    rgba[mineral] = (*COLOR_CHRYS.tolist(), 255)
    return rgba


def pick_example_voxels(
    dist_um: np.ndarray,
    mineral: np.ndarray,
    dx_um: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (accessible, buried) mineral voxel indices (iy, ix) in flow frame."""
    d_min = dist_um[mineral]
    if d_min.size == 0:
        raise ValueError("No mineral voxels")
    mineral_idx = np.argwhere(mineral)
    accessible_pool = mineral_idx[d_min <= dx_um + 1e-9]
    if accessible_pool.size == 0:
        accessible = tuple(mineral_idx[int(np.argmin(d_min))])
    else:
        accessible = tuple(accessible_pool[len(accessible_pool) // 2])

    buried_pool = mineral_idx[d_min >= float(np.percentile(d_min, 90.0))]
    if buried_pool.size == 0:
        buried = tuple(mineral_idx[int(np.argmax(d_min))])
    else:
        buried = tuple(buried_pool[len(buried_pool) // 2])
    return accessible, buried


def crop_zoom(
    rgba: np.ndarray,
    center: tuple[int, int],
    half: int = 12,
) -> np.ndarray:
    """Square crop around center with edge clamping."""
    ny, nx = rgba.shape[:2]
    iy, ix = center
    y0 = max(0, iy - half)
    y1 = min(ny, iy + half + 1)
    x0 = max(0, ix - half)
    x1 = min(nx, ix + half + 1)
    return rgba[y0:y1, x0:x1].copy()


def save_panel_d_lr_example(
    out_path: Path,
    solid: np.ndarray,
    pore: np.ndarray,
    connected: np.ndarray,
    mineral: np.ndarray,
    buried: tuple[int, int],
    l_r_um: float,
) -> None:
    """Cartoon l_r line from buried chrysocolla to nearest connected pore."""
    _, indices = ndi.distance_transform_edt(~connected, return_indices=True)
    iy, ix = buried
    py, px = int(indices[0, iy, ix]), int(indices[1, iy, ix])

    half_y, half_x = 18, 28
    y0 = max(0, iy - half_y)
    y1 = min(solid.shape[0], iy + half_y + 1)
    x0 = max(0, min(ix, px) - half_x)
    x1 = min(solid.shape[1], max(ix, px) + half_x + 1)

    sub_solid = solid[y0:y1, x0:x1]
    sub_pore = pore[y0:y1, x0:x1]
    sub_conn = connected[y0:y1, x0:x1]
    sub_min = mineral[y0:y1, x0:x1]
    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[sub_solid & ~sub_min] = (*COLOR_SOLID.tolist(), 255)
    rgba[sub_pore & ~sub_conn] = (*COLOR_ISOLATED_PORE.tolist(), 255)
    rgba[sub_conn] = (*COLOR_CONNECTED.tolist(), 255)
    rgba[sub_min] = (*COLOR_CHRYS.tolist(), 255)

    fig, ax = plt.subplots(figsize=(5.5, 2.2), dpi=200)
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    ly, lx = iy - y0, ix - x0
    py_local, px_local = py - y0, px - x0
    ax.plot([lx, px_local], [ly, py_local], color="#202020", linewidth=2.0)
    ax.scatter([lx], [ly], s=55, c="#3D8C40", edgecolors="k", linewidths=0.6, zorder=5)
    ax.scatter([px_local], [py_local], s=55, c="#7EC8C8", edgecolors="k", linewidths=0.6, zorder=5)
    ax.text(
        0.5 * (lx + px_local),
        0.5 * (ly + py_local) + 2.0,
        r"$l_r$",
        fontsize=12,
        ha="center",
        va="bottom",
    )
    ax.text(
        0.02,
        0.98,
        f"median l_r = {l_r_um:.1f} um",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_panel_d_lr_median(
    out_path: Path,
    dist_um: np.ndarray,
    mineral: np.ndarray,
    l_r_um: float,
) -> None:
    """Sorted-distance strip with median tick (returns via save helper)."""
    d = np.sort(dist_um[mineral])
    n = d.size
    fig, ax = plt.subplots(figsize=(5.5, 1.2), dpi=200)
    y = np.zeros(n)
    ax.scatter(d, y, s=14, c="#4C72B0", alpha=0.65, edgecolors="none")
    ax.axvline(l_r_um, color="#D62728", linewidth=1.8, linestyle="--")
    ax.text(l_r_um, 0.12, "median = l_r", fontsize=9, color="#D62728", ha="left")
    ax.set_xlabel("distance to inlet-connected pore [um]")
    ax.set_yticks([])
    ax.set_ylim(-0.5, 0.5)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)


def panel_d_fcuacc_patch(
    pore: np.ndarray,
    connected: np.ndarray,
    mineral: np.ndarray,
    dist_um: np.ndarray,
    center: tuple[int, int],
    dx_um: float,
) -> np.ndarray:
    """5x5 neighborhood legend patch for f_Cu,acc."""
    iy, ix = center
    half = 2
    y0, y1 = iy - half, iy + half + 1
    x0, x1 = ix - half, ix + half + 1
    ny, nx = pore.shape
    pad_y0 = max(0, -y0)
    pad_x0 = max(0, -x0)
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(ny, y1)
    x1 = min(nx, x1)

    sub_p = pore[y0:y1, x0:x1]
    sub_c = connected[y0:y1, x0:x1]
    sub_m = mineral[y0:y1, x0:x1]
    sub_d = dist_um[y0:y1, x0:x1]
    h, w = sub_p.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    solid = ~sub_p
    rgba[solid & ~sub_m] = (*COLOR_SOLID.tolist(), 255)
    rgba[sub_p & ~sub_c] = (*COLOR_ISOLATED_PORE.tolist(), 255)
    rgba[sub_c] = (*COLOR_PORE_D.tolist(), 255)
    accessible_min = sub_m & (sub_d <= dx_um + 1e-9)
    buried_min = sub_m & ~accessible_min
    rgba[buried_min] = (*COLOR_CHRYS_DARK.tolist(), 255)
    rgba[accessible_min] = (*COLOR_CHRYS_LIGHT.tolist(), 255)

    # pad back to 5x5 if near edge
    if pad_y0 or pad_x0 or h < 2 * half + 1 or w < 2 * half + 1:
        out = np.full((2 * half + 1, 2 * half + 1, 4), 255, dtype=np.uint8)
        out[:, :, :3] = COLOR_SOLID
        out[pad_y0 : pad_y0 + h, pad_x0 : pad_x0 + w] = rgba
        rgba = out
    return rgba


def load_geometry_context(
    *,
    geom_npy: Path,
    geom_h5: Path,
    dx_m: float,
    darcy_h5: Path | None = None,
    entry: dict | None = None,
    source_label: str = "",
    mineral_threshold: float | None = None,
) -> dict:
    """Load masks, metrics, and optional Darcy velocity for schematic export."""
    pore, mineral, solid = flow_frame_masks(geom_npy, geom_h5, mineral_threshold=mineral_threshold)
    connected = connected_pore_from_inlet(pore, inlet="left")
    dist_map_m, _ = mineral_access_distance_map(pore, mineral, dx=dx_m, inlet="left")
    dist_um = dist_map_m * 1e6
    dx_um = dx_m * 1e6
    stats = compute_accessibility_stats(dist_map_m, mineral)
    f_cu_acc = accessible_cu_fraction(dist_map_m, mineral, dx_m)
    l_r_um = stats["l_r_median_m"] * 1e6
    accessible, buried = pick_example_voxels(dist_um, mineral, dx_um)
    path_mask = inlet_outlet_path_mask(connected)

    ny, nx = pore.shape
    lx_mm = nx * dx_m * 1e3
    ly_mm = ny * dx_m * 1e3
    extent_mm = (0.0, lx_mm, 0.0, ly_mm)

    pore_disp = soften_mask(pore, DISPLAY_SMOOTH_RADIUS)
    connected_disp = connected & pore_disp
    solid_disp = ~pore_disp

    ux = uy = None
    if darcy_h5 is not None:
        ux, uy = load_darcy_velocity_flow_frame(darcy_h5)
        if ux.shape != pore.shape:
            raise ValueError(f"Darcy velocity shape {ux.shape} != pore shape {pore.shape}")

    if not source_label:
        if entry is not None and entry.get("geom_seed") is not None:
            source_label = f"geom {entry['geom_seed']}"
        else:
            source_label = geom_npy.stem

    return {
        "entry": entry,
        "geom_npy": geom_npy,
        "h5_path": geom_h5,
        "darcy_h5": darcy_h5,
        "source_label": source_label,
        "pore": pore,
        "mineral": mineral,
        "solid": solid,
        "connected": connected,
        "dist_um": dist_um,
        "dx_um": dx_um,
        "l_r_um": l_r_um,
        "f_cu_acc": f_cu_acc,
        "accessible": accessible,
        "buried": buried,
        "path_mask": path_mask,
        "extent_mm": extent_mm,
        "pore_disp": pore_disp,
        "connected_disp": connected_disp,
        "solid_disp": solid_disp,
        "ux": ux,
        "uy": uy,
    }


def load_realization_context(
    entry: dict,
    dx_m: float,
    darcy_h5: Path | None = None,
    *,
    mineral_threshold: float | None = None,
) -> dict:
    """Load masks and pre-simulation metrics for one manifest entry."""
    return load_geometry_context(
        geom_npy=Path(entry["geom_npy"]),
        geom_h5=Path(entry["h5_path"]),
        dx_m=dx_m,
        darcy_h5=darcy_h5,
        entry=entry,
        mineral_threshold=mineral_threshold,
    )


def plot_composed_schematic(ctx: dict, out_path: Path, *, dpi: int = 200) -> None:
    """Single publication-style schematic from one realization (display-smoothed)."""
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    extent = ctx["extent_mm"]
    lx_mm = extent[1]

    rgb_a = schematic_rgb_geometry(ctx["solid_disp"], ctx["pore_disp"])
    rgb_b = schematic_rgb_connected(ctx["pore_disp"], ctx["connected_disp"])
    rgb_c = schematic_rgb_mineral(ctx["solid_disp"], ctx["mineral"])

    fig = plt.figure(figsize=(11.5, 6.8), dpi=dpi)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.55, 1.0],
        height_ratios=[1.35, 1.0, 1.0],
        wspace=0.22,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[:, 1])

    for ax, rgb, title in (
        (ax_a, rgb_a, "(a) Stochastic 2D pore network (8 x 2 mm)"),
        (ax_b, rgb_b, "(b) Inlet-connected pore mask"),
        (ax_c, rgb_c, "(c) Chrysocolla placement (uniform protocol)"),
    ):
        imshow_schematic(ax, rgb, extent_mm=extent)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Flow direction x [mm]")
        ax.set_ylabel("y [mm]")
        ax.tick_params(labelsize=8)

    if ctx.get("ux") is not None and ctx.get("uy") is not None:
        plot_darcy_flow_on_ax(ax_a, ctx["ux"], ctx["uy"], ctx["connected"], extent)
    else:
        xmm, ymm = path_polyline_mm(ctx["path_mask"], extent)
        if xmm.size:
            ax_a.plot(xmm, ymm, color="#2166AC", linewidth=2.2, alpha=0.85, zorder=5)
    ax_a.annotate(
        "",
        xy=(0.35 * lx_mm, extent[3] * 0.92),
        xytext=(0.05 * lx_mm, extent[3] * 0.92),
        arrowprops=dict(arrowstyle="-|>", color="#2166AC", lw=1.8),
    )
    ax_a.text(0.04 * lx_mm, extent[3] * 0.97, "Inlet: acid pulse", fontsize=8, color="#2166AC")
    ax_a.text(0.78 * lx_mm, extent[3] * 0.97, "Outlet", fontsize=8, color="#2166AC", ha="right")

    # Zoom inset on (c)
    acc = ctx["accessible"]
    iy, ix = acc
    x0, x1, y0, y1 = extent
    ny, nx = ctx["pore"].shape
    cx = x0 + (ix + 0.5) * (x1 - x0) / nx
    cy = y0 + (iy + 0.5) * (y1 - y0) / ny
    hw_x = 0.45 * (x1 - x0) / nx * 14
    hw_y = 0.45 * (y1 - y0) / ny * 14
    ax_c.add_patch(
        Rectangle(
            (cx - hw_x, cy - hw_y),
            2 * hw_x,
            2 * hw_y,
            fill=False,
            edgecolor="black",
            linewidth=1.0,
        )
    )

    # Panel (d): metrics
    ax_d.axis("off")
    ax_d.text(0.0, 1.0, "(d) Pre-simulation accessibility metrics", fontsize=10, fontweight="bold", va="top", transform=ax_d.transAxes)

    sub_lr = ax_d.inset_axes([0.0, 0.52, 1.0, 0.42])
    buried = ctx["buried"]
    connected = ctx["connected_disp"]
    solid, pore, mineral = ctx["solid_disp"], ctx["pore_disp"], ctx["mineral"]
    _, indices = ndi.distance_transform_edt(~connected, return_indices=True)
    iy, ix = buried
    py, px = int(indices[0, iy, ix]), int(indices[1, iy, ix])
    half_y, half_x = 18, 28
    y0 = max(0, iy - half_y)
    y1 = min(solid.shape[0], iy + half_y + 1)
    x0i = max(0, min(ix, px) - half_x)
    x1i = min(solid.shape[1], max(ix, px) + half_x + 1)
    sub_solid = solid[y0:y1, x0i:x1i]
    sub_pore = pore[y0:y1, x0i:x1i]
    sub_conn = connected[y0:y1, x0i:x1i]
    sub_min = mineral[y0:y1, x0i:x1i]
    rgba = np.zeros((y1 - y0, x1i - x0i, 4), dtype=np.uint8)
    rgba[sub_solid & ~sub_min] = (*COLOR_SOLID.tolist(), 255)
    rgba[sub_pore & ~sub_conn] = (*COLOR_ISOLATED_PORE.tolist(), 255)
    rgba[sub_conn] = (*COLOR_CONNECTED.tolist(), 255)
    rgba[sub_min] = (*COLOR_CHRYS.tolist(), 255)
    sub_lr.imshow(upscale_rgb_smooth(_rgb_from_uint8(rgba), 3), origin="lower", interpolation="bilinear")
    ly, lx = iy - y0, ix - x0i
    py_local, px_local = py - y0, px - x0i
    sub_lr.plot([lx, px_local], [ly, py_local], color="#202020", linewidth=1.8)
    sub_lr.scatter([lx], [ly], s=40, c="#3D8C40", edgecolors="k", linewidths=0.5, zorder=5)
    sub_lr.scatter([px_local], [py_local], s=40, c="#7EC8C8", edgecolors="k", linewidths=0.5, zorder=5)
    sub_lr.set_title(
        f"l_r: median chrysocolla-to-connected-pore distance ({ctx['l_r_um']:.1f} um)",
        fontsize=8,
        loc="left",
    )
    sub_lr.set_xticks([])
    sub_lr.set_yticks([])

    sub_hist = ax_d.inset_axes([0.0, 0.28, 1.0, 0.18])
    d = np.sort(ctx["dist_um"][ctx["mineral"]])
    sub_hist.scatter(d, np.zeros_like(d), s=10, c="#4C72B0", alpha=0.55, edgecolors="none")
    sub_hist.axvline(ctx["l_r_um"], color="#D62728", linewidth=1.4, linestyle="--")
    sub_hist.set_xlabel("distance [um]", fontsize=7)
    sub_hist.set_yticks([])
    sub_hist.tick_params(labelsize=7)

    sub_patch = ax_d.inset_axes([0.0, 0.02, 0.42, 0.22])
    patch = panel_d_fcuacc_patch(
        ctx["pore"],
        ctx["connected"],
        ctx["mineral"],
        ctx["dist_um"],
        ctx["accessible"],
        ctx["dx_um"],
    )
    sub_patch.imshow(upscale_rgb_smooth(_rgb_from_uint8(patch), 8), origin="lower", interpolation="nearest")
    sub_patch.set_title(
        f"f_Cu,acc = {ctx['f_cu_acc']:.2f} (1-cell neighborhood)",
        fontsize=8,
        loc="left",
    )
    sub_patch.set_xticks([])
    sub_patch.set_yticks([])

    flow_note = "Darcy flow (tracer_test)" if ctx.get("darcy_h5") else "BFS path"
    fig.text(
        0.5,
        0.02,
        (
            f"{ctx['source_label']} | dx = {ctx['dx_um']:.0f} um | "
            f"l_r = {ctx['l_r_um']:.1f} um | f_Cu,acc = {ctx['f_cu_acc']:.2f} | "
            f"{flow_note} | display-smoothed schematic"
        ),
        ha="center",
        fontsize=8,
        color="#333333",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_layers(
    ctx: dict,
    out_dir: Path,
    *,
    dpi: int = 300,
) -> dict:
    """Write individual panel PNGs and a small metadata JSON."""
    pore = ctx["pore"]
    mineral = ctx["mineral"]
    solid = ctx["solid"]
    connected = ctx["connected"]
    dist_um = ctx["dist_um"]
    dx_um = ctx["dx_um"]
    l_r_um = ctx["l_r_um"]
    f_cu_acc = ctx["f_cu_acc"]
    accessible = ctx["accessible"]
    buried = ctx["buried"]
    path_mask = ctx["path_mask"]
    extent = ctx["extent_mm"]
    entry = ctx.get("entry") or {}

    out_dir.mkdir(parents=True, exist_ok=True)

    if ctx.get("ux") is not None:
        save_panel_a_darcy_flow(out_dir / "panel_a_geometry_darcy_flow.png", ctx, dpi=min(dpi, 220))
    else:
        save_panel_figure(
            out_dir / "panel_a_geometry.png",
            schematic_rgb_geometry(ctx["solid_disp"], ctx["pore_disp"]),
            extent,
            dpi=min(dpi, 220),
        )
        save_rgba(out_dir / "panel_a_flowpath.png", panel_a_flowpath(connected, path_mask), dpi=dpi)

    save_panel_figure(
        out_dir / "panel_b_connected_mask.png",
        schematic_rgb_connected(ctx["pore_disp"], ctx["connected_disp"]),
        extent,
        dpi=min(dpi, 220),
    )
    save_panel_figure(
        out_dir / "panel_c_mineral.png",
        schematic_rgb_mineral(ctx["solid_disp"], ctx["mineral"]),
        extent,
        dpi=min(dpi, 220),
    )

    mineral_rgba = panel_c_mineral(solid, mineral)
    save_rgba(out_dir / "panel_c_zoom_accessible.png", crop_zoom(mineral_rgba, accessible, half=14), dpi=dpi, min_width_px=240)
    save_rgba(out_dir / "panel_c_zoom_buried.png", crop_zoom(mineral_rgba, buried, half=14), dpi=dpi, min_width_px=240)
    save_rgba(
        out_dir / "panel_d_fcuacc_patch.png",
        panel_d_fcuacc_patch(pore, connected, mineral, dist_um, accessible, dx_um),
        dpi=dpi,
        min_width_px=200,
    )

    save_panel_d_lr_example(
        out_dir / "panel_d_lr_example.png",
        ctx["solid_disp"],
        ctx["pore_disp"],
        ctx["connected_disp"],
        mineral,
        buried,
        l_r_um,
    )
    save_panel_d_lr_median(out_dir / "panel_d_lr_median.png", dist_um, mineral, l_r_um)

    meta = {
        "candidate_id": entry.get("candidate_id"),
        "geom_seed": str(entry.get("geom_seed", ctx.get("source_label", ""))),
        "realization_index": int(entry.get("realization_index", -1)),
        "geom_npy": str(ctx["geom_npy"]),
        "h5_path": str(ctx["h5_path"]),
        "darcy_h5": str(ctx["darcy_h5"]) if ctx.get("darcy_h5") else None,
        "l_r_median_um": float(l_r_um),
        "f_Cu_acc": float(f_cu_acc),
        "phi": float(entry.get("phi", float(np.mean(pore)))),
        "accessible_voxel_yx": [int(accessible[0]), int(accessible[1])],
        "buried_voxel_yx": [int(buried[0]), int(buried[1])],
        "dx_um": float(dx_um),
        "flow_frame_shape_yx": [int(pore.shape[0]), int(pore.shape[1])],
    }
    (out_dir / "layer_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export schematic_setup layers and/or composed figure.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--realization-index", type=int, default=None)
    parser.add_argument("--geom-seed", type=str, default=None)
    parser.add_argument("--candidate-id", type=str, default=None)
    parser.add_argument(
        "--tracer-test",
        action="store_true",
        help="Use bean_bucket_network tracer_test geometry + Darcy flow field.",
    )
    parser.add_argument("--geom-npy", type=Path, default=None, help="Explicit geometry .npy path.")
    parser.add_argument("--geom-h5", type=Path, default=None, help="Explicit geometry .h5 with mineral VF.")
    parser.add_argument("--darcy-h5", type=Path, default=None, help="PFLOTRAN Darcy output .h5 for flow arrows.")
    parser.add_argument("--dx-m", type=float, default=None, help="Cell size [m]; default 2e-5.")
    parser.add_argument(
        "--mineral-threshold",
        type=float,
        default=None,
        help="Chrysocolla VF threshold (auto 0.0 for tracer continuous VF).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--composed",
        action="store_true",
        help="Also export single multi-panel montage PNG/PDF.",
    )
    args = parser.parse_args()

    if args.tracer_test and args.out_dir == DEFAULT_OUT:
        args.out_dir = DEFAULT_TRACER_OUT

    use_explicit = args.tracer_test or args.geom_npy is not None or args.geom_h5 is not None
    if use_explicit:
        geom_npy = args.geom_npy or DEFAULT_TRACER_GEOM_NPY
        geom_h5 = args.geom_h5 or DEFAULT_TRACER_GEOM_H5
        darcy_h5 = args.darcy_h5 or (DEFAULT_TRACER_DARCY_H5 if args.tracer_test else None)
        dx_m = float(args.dx_m if args.dx_m is not None else DEFAULT_DX_M)
        source_label = "bean_bucket tracer_test geometry" if args.tracer_test else geom_npy.stem
        entry = None
    else:
        if args.realization_index is None and args.geom_seed is None and args.candidate_id is None:
            args.realization_index = 0
        payload = json.loads(args.manifest.read_text())
        dx_m = float(args.dx_m if args.dx_m is not None else payload.get("dx_m", DEFAULT_DX_M))
        entry = load_manifest_entry(
            args.manifest,
            realization_index=args.realization_index,
            geom_seed=args.geom_seed,
            candidate_id=args.candidate_id,
        )
        geom_npy = Path(entry["geom_npy"])
        geom_h5 = Path(entry["h5_path"])
        darcy_h5 = args.darcy_h5
        source_label = ""

    ctx = load_geometry_context(
        geom_npy=geom_npy,
        geom_h5=geom_h5,
        dx_m=dx_m,
        darcy_h5=darcy_h5,
        entry=entry,
        source_label=source_label,
        mineral_threshold=args.mineral_threshold,
    )
    meta = export_layers(ctx, args.out_dir, dpi=args.dpi)
    print(f"Wrote individual panels to {args.out_dir.resolve()}")
    print(
        f"source={meta.get('geom_seed', source_label)} l_r={meta['l_r_median_um']:.2f} um "
        f"f_Cu,acc={meta['f_Cu_acc']:.3f}"
    )
    if args.composed:
        composed = args.out_dir / "schematic_setup_composed"
        plot_composed_schematic(ctx, composed, dpi=min(args.dpi, 220))
        print(f"Wrote {composed.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()