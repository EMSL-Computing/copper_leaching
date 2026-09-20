"""Compute reactive accessibility length scale l_r from pore/mineral masks.

Definition:
  l_r = median Euclidean distance from reactive mineral voxels to connected pore voxels.

Inputs:
  - pore_mask: 1=pore, 0=solid
  - mineral_mask: 1=reactive mineral
  - dx: grid spacing [m]

CLI example (NPY masks):
python compute_reactive_accessibility_length.py \
  --pore-mask-npy Chip_Designs/network1/geometry/network1_correct.npy \
  --mineral-mask-h5 Chip_Designs/network1/inputs/chrysocolla_exclusive_realization/surface_concentrated.h5 \
  --mineral-dataset Chrysocolla2_VF \
  --dx 2e-5 \
  --out-prefix Chip_Designs/network1/figs/accessibility/surface_concentrated

python compute_reactive_accessibility_length.py \
  --pore-mask-npy Chip_Designs/network1/geometry/network1_correct.npy \
  --mineral-mask-h5 Chip_Designs/network1/inputs/chrysocolla_exclusive_realization/clustered_fracture_localized.h5 \
  --mineral-dataset Chrysocolla2_VF \
  --dx 2e-5 \
  --out-prefix Chip_Designs/network1/figs/accessibility/clustered_fracture_localized

  
python compute_reactive_accessibility_length.py \
  --pore-mask-npy Chip_Designs/network1/geometry/network1_correct.npy \
  --mineral-mask-h5 Chip_Designs/network1/inputs/chrysocolla_exclusive_realization/uniform.h5 \
  --mineral-dataset Chrysocolla2_VF \
  --dx 2e-5 \
  --out-prefix Chip_Designs/network1/figs/accessibility/uniform

python compute_reactive_accessibility_length.py \
  --pore-mask-npy Chip_Designs/network1/geometry/network1_correct.npy \
  --mineral-mask-h5 Chip_Designs/network1/inputs/chrysocolla_exclusive_realization/matrix_concentrated.h5 \
  --mineral-dataset Chrysocolla2_VF \
  --dx 2e-5 \
  --out-prefix Chip_Designs/network1/figs/accessibility/matrix_concentrated


Notes:
- For pore mask from geometry npy with values {0,1}, pore is interpreted as >0.5.
- For mineral mask from VF field, mineral is interpreted as >threshold (default 0.5).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi


def load_mask_from_npy(path: Path, threshold: float = 0.5) -> np.ndarray:
    """Load a boolean mask from NPY using threshold."""
    arr = np.load(path)
    return np.asarray(arr > threshold, dtype=bool)


def load_mask_from_h5(
    path: Path,
    dataset: str,
    threshold: float = 0.5,
    shape_hint: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a boolean mask from HDF5 dataset using threshold.

    Supports:
      - 2D arrays
      - 3D arrays (uses k=0 slice)
      - flattened vectors (reshaped with Fortran order using shape_hint)
    """
    with h5py.File(path, "r") as h5:
        if dataset not in h5:
            raise KeyError(f"Dataset '{dataset}' not found in {path}")
        arr = np.asarray(h5[dataset])

    if arr.ndim == 3:
        arr = arr[:, :, 0]
    elif arr.ndim == 1:
        if shape_hint is None:
            raise ValueError(
                f"Dataset '{dataset}' in {path} is 1D (size={arr.size}) but no shape_hint was provided"
            )
        nx, ny = shape_hint
        if arr.size != nx * ny:
            raise ValueError(
                f"Cannot reshape 1D dataset '{dataset}' of size {arr.size} to shape {shape_hint}"
            )
        arr = arr.reshape((nx, ny), order="F")
    elif arr.ndim != 2:
        raise ValueError(f"Unsupported dataset ndim={arr.ndim} for {dataset} in {path}")

    return np.asarray(arr > threshold, dtype=bool)


def left_right_pore_connected(pore_mask: np.ndarray) -> bool:
    """Return True if pore cells form a path from left boundary to right boundary."""
    if pore_mask.ndim != 2:
        raise ValueError("This utility expects 2D masks")
    labels, nlab = ndi.label(
        pore_mask,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    )
    if nlab == 0:
        return False
    left_labels = set(labels[:, 0].tolist()) - {0}
    right_labels = set(labels[:, -1].tolist()) - {0}
    return bool(left_labels & right_labels)


def connected_pore_from_inlet(pore_mask: np.ndarray, inlet: str = "left") -> np.ndarray:
    """Return pore voxels connected to the inlet boundary component.

    inlet options: left, right, top, bottom.
    """
    if pore_mask.ndim != 2:
        raise ValueError("This utility expects 2D masks")

    labels, nlab = ndi.label(pore_mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int))
    if nlab == 0:
        return np.zeros_like(pore_mask, dtype=bool)

    if inlet == "left":
        boundary_labels = np.unique(labels[:, 0])
    elif inlet == "right":
        boundary_labels = np.unique(labels[:, -1])
    elif inlet == "top":
        boundary_labels = np.unique(labels[0, :])
    elif inlet == "bottom":
        boundary_labels = np.unique(labels[-1, :])
    else:
        raise ValueError(f"Unsupported inlet: {inlet}")

    boundary_labels = boundary_labels[boundary_labels > 0]
    if boundary_labels.size == 0:
        return np.zeros_like(pore_mask, dtype=bool)

    return np.isin(labels, boundary_labels)


def mineral_access_distance_map(
    pore_mask: np.ndarray,
    mineral_mask: np.ndarray,
    dx: float,
    inlet: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute accessibility distance map and connected pore mask.

    Returns:
      dist_map_m: distance [m] to connected pore space for every voxel
      connected_pore: boolean mask of inlet-connected pore space
    """
    if pore_mask.shape != mineral_mask.shape:
        raise ValueError(f"Shape mismatch: pore {pore_mask.shape}, mineral {mineral_mask.shape}")

    connected_pore = connected_pore_from_inlet(pore_mask, inlet=inlet)

    # EDT gives distance to nearest zero for non-zero elements.
    # Using (~connected_pore): connected pore voxels are zeros, all others are ones.
    # Result is distance from any voxel to nearest connected pore voxel.
    dist_map_m = ndi.distance_transform_edt(~connected_pore) * float(dx)
    return dist_map_m, connected_pore


def accessible_cu_fraction(
    dist_map_m: np.ndarray,
    mineral_mask: np.ndarray,
    dx: float,
) -> float:
    """Fraction of mineral voxels within one cell of inlet-connected pore space."""
    d = dist_map_m[mineral_mask]
    if d.size == 0:
        return 0.0
    return float(np.mean(d <= float(dx)))


def compute_accessibility_stats(dist_map_m: np.ndarray, mineral_mask: np.ndarray) -> dict[str, float]:
    """Compute l_r summary statistics at mineral voxels."""
    d = dist_map_m[mineral_mask]
    if d.size == 0:
        raise ValueError("No reactive mineral voxels found in mineral_mask")

    return {
        "l_r_median_m": float(np.median(d)),
        "mean_m": float(np.mean(d)),
        "p75_m": float(np.percentile(d, 75.0)),
        "p90_m": float(np.percentile(d, 90.0)),
        "n_mineral_voxels": int(d.size),
    }


def plot_distance_map_and_hist(
    dist_map_m: np.ndarray,
    mineral_mask: np.ndarray,
    out_prefix: Path,
    title: str = "Reactive accessibility",
) -> tuple[Path, Path]:
    """Save accessibility distance map and mineral-distance histogram."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    map_png = out_prefix.with_name(out_prefix.name + "_distance_map.png")
    hist_png = out_prefix.with_name(out_prefix.name + "_distance_hist.png")

    # Distance map
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    im = ax.imshow(dist_map_m.T * 1e6, origin="lower", cmap="magma")
    ax.contour(mineral_mask.T.astype(float), levels=[0.5], colors="cyan", linewidths=0.4)
    ax.set_title(f"{title} (distance to connected pore)", loc="left")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("distance [µm]")
    fig.tight_layout()
    fig.savefig(map_png, dpi=220)
    plt.close(fig)

    # Histogram at mineral voxels
    d_um = dist_map_m[mineral_mask] * 1e6
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.hist(d_um, bins=50, color="#4C72B0", edgecolor="white", alpha=0.95)
    ax.set_title(f"{title} histogram at reactive mineral voxels", loc="left")
    ax.set_xlabel("distance to connected pore [µm]")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(hist_png, dpi=220)
    plt.close(fig)

    return map_png, hist_png


def compute_l_r(
    pore_mask: np.ndarray,
    mineral_mask: np.ndarray,
    dx: float,
    inlet: str = "left",
) -> float:
    """Callable helper returning only l_r (median distance in meters)."""
    dist_map_m, _ = mineral_access_distance_map(pore_mask, mineral_mask, dx=dx, inlet=inlet)
    stats = compute_accessibility_stats(dist_map_m, mineral_mask)
    return stats["l_r_median_m"]


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description="Compute reactive accessibility length l_r")

    pore_src = p.add_mutually_exclusive_group(required=True)
    pore_src.add_argument("--pore-mask-npy", type=Path, help="NPY file containing pore mask-like array")
    pore_src.add_argument("--pore-mask-h5", type=Path, help="HDF5 file containing pore mask dataset")
    p.add_argument("--pore-dataset", default="Porosity", help="Dataset name if --pore-mask-h5 is used")
    p.add_argument("--pore-threshold", type=float, default=0.5, help="Threshold to convert pore array to mask")

    mineral_src = p.add_mutually_exclusive_group(required=True)
    mineral_src.add_argument("--mineral-mask-npy", type=Path, help="NPY file containing mineral mask-like array")
    mineral_src.add_argument("--mineral-mask-h5", type=Path, help="HDF5 file containing mineral dataset")
    p.add_argument("--mineral-dataset", default="Chrysocolla2_VF", help="Dataset name if --mineral-mask-h5 is used")
    p.add_argument("--mineral-threshold", type=float, default=0.5, help="Threshold to convert mineral array to mask")

    p.add_argument("--dx", type=float, required=True, help="Grid spacing [m]")
    p.add_argument("--inlet", choices=["left", "right", "top", "bottom"], default="left")
    p.add_argument("--out-prefix", type=Path, required=True, help="Prefix for output plots")
    p.add_argument("--title", default="Reactive accessibility")

    args = p.parse_args()

    if args.pore_mask_npy is not None:
        pore_mask = load_mask_from_npy(args.pore_mask_npy, threshold=args.pore_threshold)
    else:
        pore_mask = load_mask_from_h5(args.pore_mask_h5, dataset=args.pore_dataset, threshold=args.pore_threshold)

    if args.mineral_mask_npy is not None:
        mineral_mask = load_mask_from_npy(args.mineral_mask_npy, threshold=args.mineral_threshold)
    else:
        mineral_mask = load_mask_from_h5(
            args.mineral_mask_h5,
            dataset=args.mineral_dataset,
            threshold=args.mineral_threshold,
            shape_hint=pore_mask.shape,
        )

    dist_map_m, connected_pore = mineral_access_distance_map(
        pore_mask=pore_mask,
        mineral_mask=mineral_mask,
        dx=args.dx,
        inlet=args.inlet,
    )

    stats = compute_accessibility_stats(dist_map_m, mineral_mask)
    map_png, hist_png = plot_distance_map_and_hist(
        dist_map_m=dist_map_m,
        mineral_mask=mineral_mask,
        out_prefix=args.out_prefix,
        title=args.title,
    )

    print(f"connected_pore_voxels: {int(np.count_nonzero(connected_pore))}")
    print(f"mineral_voxels:        {stats['n_mineral_voxels']}")
    print(f"l_r_median_m:          {stats['l_r_median_m']:.6e}")
    print(f"mean_m:                {stats['mean_m']:.6e}")
    print(f"p75_m:                 {stats['p75_m']:.6e}")
    print(f"p90_m:                 {stats['p90_m']:.6e}")
    print(f"distance_map_png:      {map_png}")
    print(f"histogram_png:         {hist_png}")


if __name__ == "__main__":
    main()
