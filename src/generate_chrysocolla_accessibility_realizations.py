"""Generate fixed-inventory chrysocolla realizations.

This script creates stochastic chrysocolla fields with the same total
chrysocolla volume fraction in every realization. Minerals are placed only in
solid cells; pores remain mineral-free.

Example usage:
    python generate_chrysocolla_accessibility_realizations.py \
      --geometry-npy Chip_Designs/network1/geometry/network1_correct.npy \
      --out-dir Chip_Designs/network1/inputs/chrysocolla_realizations \
      --n-realizations 10 \
      --chrysocolla-solid-fraction 0.20
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import matplotlib.pyplot as plt
import numpy as np
from build_pflotran_hdf5_from_npy import load_geometry_npy, write_pflotran_h5
from compute_reactive_accessibility_length import connected_pore_from_inlet
from normalize_design_to_solver import normalize_design_to_solver
from paint_mutually_exclusive_minerals import neighbor_count, top_k_mask


@dataclass
class AccessibilityMetadata:
    realization: int
    seed: int
    chrysocolla_solid_fraction: float
    solid_vf: float
    calcite_solid_fraction: float
    total_solid_cells: int
    chrysocolla_cells: int
    calcite_cells: int
    quartz_cells: int
    inlet: str
    placement_model: str
    smooth_passes: int
    accessibility_bias: float


def neighbor_distance_from_connected_pore(img: np.ndarray, inlet: str) -> tuple[np.ndarray, np.ndarray]:
    """Return 4-neighbor graph distance in cells from inlet-connected pore space."""
    pore = img == 1.0
    connected_pore = connected_pore_from_inlet(pore, inlet=inlet)
    dist = np.full(img.shape, -1, dtype=int)
    dist[connected_pore] = 0
    frontier = connected_pore.copy()
    current_dist = 0

    while np.any(frontier):
        next_frontier = (neighbor_count(frontier) > 0) & (dist < 0)
        if not np.any(next_frontier):
            break
        current_dist += 1
        dist[next_frontier] = current_dist
        frontier = next_frontier

    return dist, connected_pore


def smooth_random_field(rng: np.random.Generator, shape: tuple[int, int], passes: int) -> np.ndarray:
    """Create a spatially correlated random score field."""
    field = rng.random(shape)
    for _ in range(max(0, int(passes))):
        padded_sum = np.zeros(shape, dtype=float)
        counts = np.zeros(shape, dtype=float)
        padded_sum += field
        counts += 1.0
        padded_sum[:-1, :] += field[1:, :]
        counts[:-1, :] += 1.0
        padded_sum[1:, :] += field[:-1, :]
        counts[1:, :] += 1.0
        padded_sum[:, :-1] += field[:, 1:]
        counts[:, :-1] += 1.0
        padded_sum[:, 1:] += field[:, :-1]
        counts[:, 1:] += 1.0
        field = padded_sum / counts
    return field


def choose_chrysocolla_cells(
    img: np.ndarray,
    dist_cells: np.ndarray,
    n_chrysocolla: int,
    rng: np.random.Generator,
    smooth_passes: int,
    accessibility_bias: float,
) -> np.ndarray:
    """Select a fixed number of solid cells as chrysocolla."""
    solid = img == 0.0
    random_field = smooth_random_field(rng, img.shape, passes=smooth_passes)
    reachable_dist = np.where(dist_cells > 0, dist_cells, np.nan)
    max_dist = float(np.nanmax(reachable_dist)) if np.any(np.isfinite(reachable_dist)) else 1.0
    depth_score = np.where(dist_cells > 0, dist_cells / max(max_dist, 1.0), 0.0)
    score = random_field + float(accessibility_bias) * depth_score
    score[~solid] = -np.inf
    return top_k_mask(score, solid, n_chrysocolla)


def choose_calcite_cells(
    img: np.ndarray,
    chrys_mask: np.ndarray,
    n_calcite: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select calcite cells from non-chrysocolla solids."""
    candidates = (img == 0.0) & ~chrys_mask
    score = rng.random(img.shape)
    score[~candidates] = -np.inf
    return top_k_mask(score, candidates, n_calcite)


def make_flat_vf(mask: np.ndarray, solid_vf: float) -> np.ndarray:
    """Convert a 2D mask to PFLOTRAN Fortran-order flat volume fraction."""
    out = np.zeros(mask.size, dtype=float)
    out[mask.ravel(order="F")] = float(solid_vf)
    return out


def validate_solid_only_masks(img: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    """Fail if any mineral mask occupies pore space."""
    pore = img == 1.0
    for name, mask in masks.items():
        pore_count = int(np.count_nonzero(mask & pore))
        if pore_count:
            raise ValueError(f"{name} has {pore_count} cells in pore space; minerals must be solid-only")


def write_metadata(h5_path: Path, chrys_vf: np.ndarray, metadata: AccessibilityMetadata) -> None:
    """Append realization metadata to a PFLOTRAN H5 file."""
    with h5py.File(h5_path, "a") as h5:
        grp = h5.require_group("Metadata")
        for key, value in asdict(metadata).items():
            grp.attrs[key] = value
        grp.attrs["total_chrysocolla_vf_sum"] = float(chrys_vf.sum())


def plot_chrysocolla_realization(
    img: np.ndarray,
    chrys_mask: np.ndarray,
    connected_pore: np.ndarray,
    out_png: Path,
) -> None:
    """Write a compact diagnostic plot of chrysocolla placement."""
    overlay = np.zeros((*img.shape, 4), dtype=float)
    overlay[chrys_mask] = np.array([228 / 255, 26 / 255, 28 / 255, 1.0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].imshow(np.where(img == 1.0, 1.0, 0.75).T, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[0].contour(connected_pore.T.astype(float), levels=[0.5], colors="tab:blue", linewidths=0.5)
    axes[0].set_title("(a) Connected pore space", loc="left")

    axes[1].imshow(np.where(img == 1.0, 1.0, 0.65).T, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(np.where(chrys_mask, 1.0, np.nan).T, origin="lower", cmap="YlOrRd", vmin=0, vmax=1, alpha=1.0)
    axes[1].set_title("(b) Fixed chrysocolla inventory", loc="left")

    axes[2].imshow(np.where(img == 1.0, 1.0, 0.65).T, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(np.transpose(overlay, (1, 0, 2)), origin="lower")
    axes[2].contour(connected_pore.T.astype(float), levels=[0.5], colors="black", linewidths=0.25, alpha=0.4)
    axes[2].set_title("(c) Chrysocolla on pore geometry", loc="left")

    for ax in axes:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")

    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_realization(
    img: np.ndarray,
    out_dir: Path,
    realization: int,
    seed: int,
    chrysocolla_solid_fraction: float,
    calcite_solid_fraction: float,
    solid_vf: float,
    pore_diffusivity: float,
    solid_diffusivity: float,
    pore_porosity: float,
    solid_porosity: float,
    inlet: str,
    smooth_passes: int,
    accessibility_bias: float,
) -> None:
    """Build one fixed-inventory chrysocolla realization."""
    rng = np.random.default_rng(seed)
    solid = img == 0.0
    n_solid = int(np.count_nonzero(solid))
    n_chrysocolla = int(round(float(chrysocolla_solid_fraction) * n_solid))
    n_calcite = int(round(float(calcite_solid_fraction) * n_solid))
    dist_cells, connected_pore = neighbor_distance_from_connected_pore(img, inlet=inlet)

    chrys_mask = choose_chrysocolla_cells(
        img=img,
        dist_cells=dist_cells,
        n_chrysocolla=n_chrysocolla,
        rng=rng,
        smooth_passes=smooth_passes,
        accessibility_bias=accessibility_bias,
    )
    calc_mask = choose_calcite_cells(img, chrys_mask, n_calcite, rng)
    quartz_mask = solid & ~chrys_mask & ~calc_mask
    validate_solid_only_masks(
        img,
        {
            "chrysocolla": chrys_mask,
            "calcite": calc_mask,
        },
    )

    chrys = make_flat_vf(chrys_mask, solid_vf)
    calc = make_flat_vf(calc_mask, solid_vf)
    quartz = make_flat_vf(quartz_mask, solid_vf)
    quartz[(img == 1.0).ravel(order="F")] = 1.0 - float(pore_porosity)

    out_h5 = out_dir / f"chrysocolla_real{realization:03d}.h5"
    out_png = out_dir / f"chrysocolla_real{realization:03d}.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = write_pflotran_h5(
        img=img,
        h5_path=out_h5,
        chrysocolla_vf=chrys,
        calcite_vf=calc,
        quartz_vf=quartz,
        pore_diffusivity=pore_diffusivity,
        solid_diffusivity=solid_diffusivity,
        pore_porosity=pore_porosity,
        solid_porosity=solid_porosity,
    )

    metadata = AccessibilityMetadata(
        realization=realization,
        seed=seed,
        chrysocolla_solid_fraction=chrysocolla_solid_fraction,
        solid_vf=solid_vf,
        calcite_solid_fraction=calcite_solid_fraction,
        total_solid_cells=n_solid,
        chrysocolla_cells=int(np.count_nonzero(chrys_mask)),
        calcite_cells=int(np.count_nonzero(calc_mask)),
        quartz_cells=int(np.count_nonzero(quartz_mask)),
        inlet=inlet,
        placement_model="fixed_inventory_correlated_random",
        smooth_passes=smooth_passes,
        accessibility_bias=accessibility_bias,
    )
    write_metadata(out_h5, chrys, metadata)
    plot_chrysocolla_realization(img, chrys_mask, connected_pore, out_png)

    closure = float(np.max(np.abs(fields["porosity_flat"] + chrys + calc + quartz - 1.0)))
    print(
        f"{out_h5}: chrys_cells={metadata.chrysocolla_cells} "
        f"chrys_sum={chrys.sum():.6f} closure={closure:.3e}"
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Generate fixed-inventory chrysocolla PFLOTRAN HDF5 files."
    )
    parser.add_argument("--geometry-npy", type=Path, default=Path("Chip_Designs/network1/geometry/network1_correct.npy"))
    parser.add_argument(
        "--raw-design-npy",
        type=Path,
        default=None,
        help="Optional raw chip design NPY; normalized with normalize_design_to_solver.py before generation.",
    )
    parser.add_argument(
        "--geometry-out-dir",
        type=Path,
        default=Path("Chip_Designs/network1/geometry"),
        help="Output directory for --raw-design-npy normalization.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Chip_Designs/network1/inputs/chrysocolla_realizations"),
    )
    parser.add_argument("--n-realizations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chrysocolla-solid-fraction", type=float, default=0.20)
    parser.add_argument("--calcite-solid-fraction", type=float, default=0.0)
    parser.add_argument("--solid-vf", type=float, default=0.99)
    parser.add_argument("--pore-porosity", type=float, default=0.99)
    parser.add_argument("--solid-porosity", type=float, default=0.01)
    parser.add_argument("--pore-diffusivity", type=float, default=1.0e-9)
    parser.add_argument("--solid-diffusivity", type=float, default=1.0e-11)
    parser.add_argument("--inlet", choices=["left", "right", "top", "bottom"], default="left")
    parser.add_argument(
        "--smooth-passes",
        type=int,
        default=4,
        help="Spatial correlation of stochastic mineral placement.",
    )
    parser.add_argument(
        "--accessibility-bias",
        type=float,
        default=0.0,
        help="Positive values favor deeper solid cells; negative values favor pore-adjacent solid cells.",
    )
    args = parser.parse_args()

    if args.n_realizations < 1:
        raise ValueError("--n-realizations must be >= 1")
    if not (0.0 <= args.chrysocolla_solid_fraction <= 1.0):
        raise ValueError("--chrysocolla-solid-fraction must be between 0 and 1")
    if not (0.0 <= args.calcite_solid_fraction <= 1.0):
        raise ValueError("--calcite-solid-fraction must be between 0 and 1")
    if args.chrysocolla_solid_fraction + args.calcite_solid_fraction > 1.0:
        raise ValueError("chrysocolla + calcite solid fractions must be <= 1")

    geometry_npy = args.geometry_npy
    if args.raw_design_npy is not None:
        geometry_npy = normalize_design_to_solver(args.raw_design_npy, args.geometry_out_dir)
        print(f"Normalized raw design to solver geometry: {geometry_npy}")

    img = load_geometry_npy(geometry_npy)
    pore_fraction = float(np.mean(img == 1.0))
    solid_fraction = float(np.mean(img == 0.0))
    print(f"Geometry convention: 1=pore, 0=solid; pore_fraction={pore_fraction:.3f}, solid_fraction={solid_fraction:.3f}")
    for i in range(args.n_realizations):
        build_realization(
            img=img,
            out_dir=args.out_dir,
            realization=i,
            seed=int(args.seed + i),
            chrysocolla_solid_fraction=args.chrysocolla_solid_fraction,
            calcite_solid_fraction=args.calcite_solid_fraction,
            solid_vf=args.solid_vf,
            pore_diffusivity=args.pore_diffusivity,
            solid_diffusivity=args.solid_diffusivity,
            pore_porosity=args.pore_porosity,
            solid_porosity=args.solid_porosity,
            inlet=args.inlet,
            smooth_passes=args.smooth_passes,
            accessibility_bias=args.accessibility_bias,
        )


if __name__ == "__main__":
    main()
