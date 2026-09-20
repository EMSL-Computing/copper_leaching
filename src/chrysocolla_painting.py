"""
Chrysocolla mineral painting utilities for pore-scale geometry fields.

Provides functions to generate spatially distributed mineral volume-fraction
(VF) fields (chrysocolla, quartz) from a binary pore/solid image,
# (chrysocolla, calcite, quartz) -- calcite removed from chemistry
supporting multiple placement modes:

    - surface_concentrated : mineral concentrated at pore-solid interfaces
    - uniform              : mineral distributed evenly across all solid cells
    - matrix_concentrated  : mineral concentrated deep in the solid matrix
    - clustered_fracture_localized : spatially clustered, fracture-biased placement

Main entry points:
    paint_chrysocolla_fields() -- generate VF arrays from a pore geometry image
    plot_geometry()            -- visualise the resulting mineral fields
    save_metadata_h5()         -- persist run metadata into an HDF5 file

Example:
    from chrysocolla_painting import plot_geometry
    # plot_geometry(img, chrys, calc, qtz, "out.png")  # calcite panel commented out
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


@dataclass
class ChrysocollaMetadata:
    mode: str
    seed: int
    total_chrysocolla_vf: float
    layer_depth: int
    rind_coverage: float
    inward_decay_length: float
    cluster_smoothing_passes: int
    cluster_pore_bias: float
    selected_solid_cells: int
    total_solid_cells: int


def _neighbor_mean(arr: np.ndarray) -> np.ndarray:
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out = np.zeros_like(arr, dtype=float)
    for dx, dy in shifts:
        out += np.roll(np.roll(arr, dx, axis=0), dy, axis=1)
    return out / 4.0


def _distance_from_pore(solid_mask: np.ndarray, max_layers: int | None = None) -> np.ndarray:
    nx, ny = solid_mask.shape
    inf = nx + ny + 10
    dist = np.full((nx, ny), inf, dtype=int)
    is_pore = ~solid_mask
    dist[is_pore] = 0
    frontier = is_pore.copy()
    steps = 0
    max_steps = inf if max_layers is None else max_layers
    while frontier.any() and steps < max_steps:
        steps += 1
        nbr = np.zeros_like(frontier, dtype=bool)
        nbr[:-1, :] |= frontier[1:, :]
        nbr[1:, :] |= frontier[:-1, :]
        nbr[:, :-1] |= frontier[:, 1:]
        nbr[:, 1:] |= frontier[:, :-1]
        update = solid_mask & (dist > steps) & nbr
        if not update.any():
            break
        dist[update] = steps
        frontier = update
    return dist


def _weights_for_mode(
    img: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    layer_depth: int,
    rind_coverage: float,
    inward_decay_length: float,
    cluster_smoothing_passes: int,
    cluster_pore_bias: float,
) -> tuple[np.ndarray, int]:
    solid = img == 0.0
    dist = _distance_from_pore(solid)
    w = np.zeros_like(img, dtype=float)
    solid_dist = dist[solid]
    if solid_dist.size == 0:
        return w, 0

    if mode == "surface_concentrated":
        rind_cap = max(1, layer_depth + 1)
        eligible = solid & (dist <= rind_cap)
        if rind_coverage < 1.0:
            idx = np.where(eligible.ravel(order="F"))[0]
            keep_n = int(round(len(idx) * float(rind_coverage)))
            keep_n = max(0, min(len(idx), keep_n))
            keep = np.zeros(len(idx), dtype=bool)
            if keep_n > 0:
                keep[rng.choice(len(idx), size=keep_n, replace=False)] = True
            mask_flat = np.zeros(eligible.size, dtype=bool)
            mask_flat[idx[keep]] = True
            eligible = mask_flat.reshape(img.shape, order="F")
        decay = np.exp(-np.maximum(dist - 1, 0) / max(inward_decay_length, 1e-8))
        w[eligible] = decay[eligible]
    elif mode == "uniform":
        w[solid] = 1.0
    elif mode == "matrix_concentrated":
        dmin = int(np.min(solid_dist))
        dmax = int(np.max(solid_dist))
        threshold = min(dmax, dmin + max(1, layer_depth + 1))
        deep = solid & (dist >= threshold)
        if deep.any():
            w[deep] = (dist[deep] - threshold + 1).astype(float)
        else:
            w[solid] = dist[solid].astype(float)
    elif mode == "clustered_fracture_localized":
        noise = rng.random(img.shape)
        for _ in range(max(0, cluster_smoothing_passes)):
            noise = 0.5 * noise + 0.5 * _neighbor_mean(noise)
        pore_bias = np.exp(-np.maximum(dist - 1, 0) / max(cluster_pore_bias, 1e-8))
        score = noise * (0.2 + 0.8 * pore_bias)
        score[~solid] = -1.0
        valid = score[solid]
        frac = np.clip(rind_coverage, 0.05, 1.0)
        cutoff = np.quantile(valid, 1.0 - frac)
        selected = solid & (score >= cutoff)
        w[selected] = score[selected] - cutoff + 1.0e-6
    else:
        raise ValueError(f"Unsupported chrysocolla mode: {mode}")

    n_sel = int(np.count_nonzero(w > 0.0))
    if n_sel == 0:
        w[solid] = 1.0
        n_sel = int(np.count_nonzero(solid))
    return w, n_sel


def paint_chrysocolla_fields(
    img: np.ndarray,
    mode: str,
    seed: int,
    chrysocolla_solid_fraction: float = 0.05,
    calcite_fraction: float = 0.05,
    layer_depth: int = 2,
    rind_coverage: float = 1.0,
    inward_decay_length: float = 1.25,
    cluster_smoothing_passes: int = 5,
    cluster_pore_bias: float = 2.0,
    pore_porosity: float = 0.99,
    solid_porosity: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ChrysocollaMetadata]:
    """Generate mineral VF fields from a fixed pore geometry and placement mode."""
    rng = np.random.default_rng(seed)
    flat = img.ravel(order="F")
    solid = img == 0.0
    solid_flat = flat == 0.0
    poro = np.where(flat == 1.0, pore_porosity, solid_porosity)
    solid_frac = 1.0 - poro
    n_solid = int(np.count_nonzero(solid_flat))

    weights_2d, n_sel = _weights_for_mode(
        img=img,
        mode=mode,
        rng=rng,
        layer_depth=layer_depth,
        rind_coverage=rind_coverage,
        inward_decay_length=inward_decay_length,
        cluster_smoothing_passes=cluster_smoothing_passes,
        cluster_pore_bias=cluster_pore_bias,
    )
    w = weights_2d.ravel(order="F")
    w[~solid_flat] = 0.0

    target_total = float(chrysocolla_solid_fraction) * float(n_solid)
    if w.sum() <= 0.0:
        w[solid_flat] = 1.0
    chrys = np.zeros_like(solid_frac)
    provisional = w / w.sum()
    chrys[solid_flat] = target_total * provisional[solid_flat]

    cap = solid_frac.copy()
    active = solid_flat.copy()
    for _ in range(8):
        over = active & (chrys > cap)
        if not np.any(over):
            break
        chrys[over] = cap[over]
        rem = target_total - float(chrys.sum())
        active[over] = False
        room = np.where(active, cap - chrys, 0.0)
        room_sum = float(room.sum())
        if rem <= 0.0 or room_sum <= 0.0:
            break
        chrys += rem * (room / room_sum)
    if not np.any(chrys > cap + 1e-12):
        err = target_total - float(chrys.sum())
        if abs(err) > 1e-12:
            room = np.where(solid_flat, cap - chrys, 0.0)
            room_sum = float(room.sum())
            if room_sum > 0:
                chrys += err * (room / room_sum)

    remaining = solid_frac - chrys
    calc = np.zeros_like(chrys)
    calc[solid_flat] = calcite_fraction * remaining[solid_flat]
    quartz = np.clip(solid_frac - chrys - calc, 0.0, None)

    meta = ChrysocollaMetadata(
        mode=mode,
        seed=seed,
        total_chrysocolla_vf=float(chrys.sum()),
        layer_depth=layer_depth,
        rind_coverage=rind_coverage,
        inward_decay_length=inward_decay_length,
        cluster_smoothing_passes=cluster_smoothing_passes,
        cluster_pore_bias=cluster_pore_bias,
        selected_solid_cells=n_sel,
        total_solid_cells=n_solid,
    )
    return chrys, calc, quartz, meta


def save_metadata_h5(h5_path: Path, metadata: ChrysocollaMetadata) -> None:
    with h5_path.open("ab"):
        pass
    import h5py

    with h5py.File(h5_path, "a") as f:
        grp = f.require_group("Metadata")
        for key, value in asdict(metadata).items():
            grp.attrs[key] = value


def plot_geometry(
    img: np.ndarray,
    chrysocolla_vf_flat: np.ndarray,
    calcite_vf_flat: np.ndarray,
    quartz_vf_flat: np.ndarray,
    out_path: Path,
    interface_flat: np.ndarray | None = None,
) -> None:
    nx, ny = img.shape
    chry2d = chrysocolla_vf_flat.reshape(nx, ny, order="F").T
    calc2d = calcite_vf_flat.reshape(nx, ny, order="F").T
    qtz2d = quartz_vf_flat.reshape(nx, ny, order="F").T
    solid_mask = img.T == 0.0

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    vf_fields = [
        (chry2d, "Chrysocolla2 VF", "YlOrRd"),
        # (calc2d, "Calcite VF", "Blues"),
        (qtz2d, "Quartz VF", "Greens"),
    ]
    for ax, (field, title, cmap) in zip(axes.flat[:3], vf_fields):
        plot_field = np.where(solid_mask, field, np.nan)
        vmax = float(np.nanmax(plot_field)) if np.any(np.isfinite(plot_field)) else 1.0
        if vmax <= 0.0:
            vmax = 1.0
        im = ax.imshow(plot_field, origin="lower", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("VF")
        ax.set_title(f"{title} (mean={np.nanmean(plot_field):.4f})")
        ax.set_xlabel("X (flow ->)")
        ax.set_ylabel("Y")

    dominant = np.full_like(chry2d, np.nan, dtype=float)
    stack = np.stack([chry2d, calc2d, qtz2d], axis=0)
    dominant_idx = np.argmax(stack, axis=0)
    dominant[solid_mask] = dominant_idx[solid_mask]
    cmap_dom = matplotlib.colors.ListedColormap(["#d95f02", "#1f78b4", "#33a02c"])
    imd = axes[1, 1].imshow(dominant, origin="lower", interpolation="nearest", cmap=cmap_dom, vmin=-0.5, vmax=2.5)
    cb = plt.colorbar(imd, ax=axes[1, 1], ticks=[0, 1, 2])
    # cb.set_ticklabels(["Chrysocolla2", "Calcite", "Quartz"])
    axes[1, 1].set_title("Dominant Mineral (solid cells)")
    axes[1, 1].set_xlabel("X (flow ->)")
    axes[1, 1].set_ylabel("Y")

    fig.suptitle(f"Pore geometry {nx}x{ny} | solid cells={int(solid_mask.sum())}", fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
