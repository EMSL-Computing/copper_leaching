"""Paint mutually exclusive mineral domains for PFLOTRAN inputs.

Each solid cell is assigned one dominant mineral: Chrysocolla2, Calcite, or
Quartz. PFLOTRAN still receives mineral volume-fraction fields, but only one
mineral is nonzero in each solid cell.

 --layer-depth affects both surface_concentrated and clustered_fracture_localized in the current script. If you want to tune clustered depth without changing surface behavior, run only clustered with its own --layer-depth using --modes clustered_fracture_localized, then generate surface separately with its preferred depth.

python paint_mutually_exclusive_minerals.py --geometry-npy Chip_Designs/network1/geometry/network1_correct.npy --out-dir Chip_Designs/network1/inputs/chrysocolla_exclusive_realization --cluster-smoothing-passes 1 --cluster-pore-bias 1 --calcite-solid-fraction 0.1 --chrysocolla-solid-fraction 0.1
 

**surface_concentrated**

python paint_mutually_exclusive_minerals.py \
  --geometry-npy Chip_Designs/network1/geometry/network1_correct.npy \
  --out-dir Chip_Designs/network1/inputs/chrysocolla_exclusive_realization \
  --modes surface_concentrated \
  --seed 42 \
  --calcite-solid-fraction 0\
  --chrysocolla-solid-fraction 0.2


**clustered_fracture_localized**

python paint_mutually_exclusive_minerals.py \
    --geometry-npy Chip_Designs/network1/geometry/network1_correct.npy \
    --out-dir Chip_Designs/network1/inputs/chrysocolla_exclusive_realization \
    --modes clustered_fracture_localized \
    --cluster-pore-bias 3.0 \
    --layer-depth 1 \
    --chrysocolla-solid-fraction 0.20 \
    --calcite-solid-fraction 0 \
    --cluster-smoothing-passes 1 \
    --seed 420

    
**matrix_concentrated**

python paint_mutually_exclusive_minerals.py \
  --geometry-npy Chip_Designs/network1/geometry/network1_correct.npy \
  --out-dir Chip_Designs/network1/inputs/chrysocolla_exclusive_realization \
  --modes matrix_concentrated \
  --seed 42 \
  --calcite-solid-fraction 0\
  --chrysocolla-solid-fraction 0.2
  
"""

from __future__ import annotations

import _repo  # noqa: F401 — path bootstrap

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

from build_pflotran_hdf5_from_npy import load_geometry_npy, write_pflotran_h5
from chrysocolla_painting import _distance_from_pore, _neighbor_mean, plot_geometry


MODES = [
    "surface_concentrated",
    "uniform",
    "matrix_concentrated",
    "clustered_fracture_localized",
]


@dataclass
class ExclusiveMineralMetadata:
    mode: str
    seed: int
    painting_model: str
    chrysocolla_solid_fraction: float
    calcite_solid_fraction: float
    chrysocolla_cells: int
    calcite_cells: int
    quartz_cells: int
    total_solid_cells: int
    layer_depth: int
    cluster_smoothing_passes: int
    cluster_pore_bias: float


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    """Count 4-neighbor cells without periodic wraparound."""
    out = np.zeros(mask.shape, dtype=int)
    out[:-1, :] += mask[1:, :]
    out[1:, :] += mask[:-1, :]
    out[:, :-1] += mask[:, 1:]
    out[:, 1:] += mask[:, :-1]
    return out


def top_k_mask(score: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    """Select the k highest-scoring candidate cells."""
    idx = np.flatnonzero(candidates.ravel(order="F"))
    out_flat = np.zeros(score.size, dtype=bool)
    if k <= 0 or idx.size == 0:
        return out_flat.reshape(score.shape, order="F")
    k = min(int(k), idx.size)
    vals = score.ravel(order="F")[idx]
    chosen_local = np.argpartition(vals, -k)[-k:]
    out_flat[idx[chosen_local]] = True
    return out_flat.reshape(score.shape, order="F")


def grow_clustered_mask(
    solid: np.ndarray,
    dist: np.ndarray,
    score: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
    layer_depth: int,
    cluster_pore_bias: float,
) -> np.ndarray:
    """Grow connected chrysocolla clusters from pore-adjacent seeds."""
    n_target = min(max(0, int(n_target)), int(np.count_nonzero(solid)))
    cluster = np.zeros(solid.shape, dtype=bool)
    if n_target == 0:
        return cluster

    bias_seed_bonus = int(round(3.0 / max(cluster_pore_bias, 0.5)))
    seed_cap = max(1, min(layer_depth + 2, 1 + bias_seed_bonus))
    seed_zone = solid & (dist <= seed_cap)
    if not np.any(seed_zone):
        seed_zone = solid.copy()

    # A small number of seeds gives inward thickening instead of a rind-like shell.
    n_seeds = max(1, min(n_target, int(round(np.sqrt(n_target) / 2.0))))
    seed_mask = top_k_mask(score, seed_zone, n_seeds)
    cluster |= seed_mask

    while int(np.count_nonzero(cluster)) < n_target:
        neighbors = neighbor_count(cluster).astype(float)
        frontier = solid & ~cluster & (neighbors > 0)
        max_depth = max(1, int(round(8.0 / max(cluster_pore_bias, 0.5))))
        frontier &= dist <= max_depth
        if not np.any(frontier):
            remaining = solid & ~cluster
            if not np.any(remaining):
                break
            # If disconnected pockets remain, start a new patch in the best remaining zone.
            pick = top_k_mask(score, remaining, 1)
            cluster |= pick
            continue

        growth_score = np.full(score.shape, -np.inf, dtype=float)
        pore_pref = np.exp(-np.maximum(dist - 1, 0) * max(cluster_pore_bias, 1.0e-8))
        growth_score[frontier] = (
            1.1 * score[frontier]
            + 1.0 * neighbors[frontier]
            + 1.25 * pore_pref[frontier]
            + 0.05 * rng.random(np.count_nonzero(frontier))
        )
        need = n_target - int(np.count_nonzero(cluster))
        batch = min(max(1, need // 4), 256, need)
        add = top_k_mask(growth_score, frontier, batch)
        cluster |= add

    return cluster


def chrysocolla_score(
    img: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    layer_depth: int,
    cluster_smoothing_passes: int,
    cluster_pore_bias: float,
) -> np.ndarray:
    """Build a spatial score field for mutually exclusive chrysocolla cells."""
    solid = img == 0.0
    dist = _distance_from_pore(solid)
    noise = rng.random(img.shape)
    score = np.full(img.shape, -np.inf, dtype=float)

    if mode == "surface_concentrated":
        rind = solid & (dist <= max(1, layer_depth + 1))
        score[rind] = 1.0 / np.maximum(dist[rind], 1) + 0.05 * noise[rind]
    elif mode == "uniform":
        score[solid] = noise[solid]
    elif mode == "matrix_concentrated":
        score[solid] = dist[solid].astype(float) + 0.05 * noise[solid]
    elif mode == "clustered_fracture_localized":
        smooth = noise
        for _ in range(max(0, cluster_smoothing_passes)):
            smooth = 0.5 * smooth + 0.5 * _neighbor_mean(smooth)
        pore_bias = np.exp(-np.maximum(dist - 1, 0) * max(cluster_pore_bias, 1.0e-8))
        # Keep clustered mode near flow-accessible zones, but less rind-dominated.
        score[solid] = smooth[solid] * (0.35 + 0.65 * pore_bias[solid])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return score


def assign_exclusive_minerals(
    img: np.ndarray,
    mode: str,
    seed: int,
    chrysocolla_solid_fraction: float,
    calcite_solid_fraction: float,
    layer_depth: int,
    cluster_smoothing_passes: int,
    cluster_pore_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ExclusiveMineralMetadata]:
    """Assign mutually exclusive dominant minerals and return PFLOTRAN VFs."""
    rng = np.random.default_rng(seed)
    solid = img == 0.0
    flat_solid = solid.ravel(order="F")
    n_solid = int(np.count_nonzero(flat_solid))
    solid_vf = 0.99
    pore_vf = 0.01

    n_chrys = int(round(chrysocolla_solid_fraction * n_solid))
    n_calc = int(round(calcite_solid_fraction * n_solid))
    dist = _distance_from_pore(solid)

    score = chrysocolla_score(
        img,
        mode,
        rng,
        layer_depth,
        cluster_smoothing_passes,
        cluster_pore_bias,
    )
    if mode == "clustered_fracture_localized":
        chrys_mask = grow_clustered_mask(
            solid=solid,
            dist=dist,
            score=score,
            n_target=n_chrys,
            rng=rng,
            layer_depth=layer_depth,
            cluster_pore_bias=cluster_pore_bias,
        )
    else:
        chrys_mask = top_k_mask(score, solid, n_chrys)

    remaining = solid & ~chrys_mask
    calc_score = neighbor_count(chrys_mask).astype(float)
    calc_score += 0.25 * rng.random(img.shape)
    calc_score[~remaining] = -np.inf
    calc_mask = top_k_mask(calc_score, remaining, n_calc)

    quartz_mask = solid & ~chrys_mask & ~calc_mask
    flat = img.ravel(order="F")
    chrys = np.zeros(flat.shape, dtype=float)
    calc = np.zeros(flat.shape, dtype=float)
    quartz = np.zeros(flat.shape, dtype=float)

    chrys[chrys_mask.ravel(order="F")] = solid_vf
    calc[calc_mask.ravel(order="F")] = solid_vf
    quartz[quartz_mask.ravel(order="F")] = solid_vf
    quartz[flat == 1.0] = pore_vf

    meta = ExclusiveMineralMetadata(
        mode=mode,
        seed=seed,
        painting_model="mutually_exclusive_dominant_mineral",
        chrysocolla_solid_fraction=chrysocolla_solid_fraction,
        calcite_solid_fraction=calcite_solid_fraction,
        chrysocolla_cells=int(np.count_nonzero(chrys_mask)),
        calcite_cells=int(np.count_nonzero(calc_mask)),
        quartz_cells=int(np.count_nonzero(quartz_mask)),
        total_solid_cells=n_solid,
        layer_depth=layer_depth,
        cluster_smoothing_passes=cluster_smoothing_passes,
        cluster_pore_bias=cluster_pore_bias,
    )
    return chrys, calc, quartz, meta


def save_metadata(h5_path: Path, metadata: ExclusiveMineralMetadata) -> None:
    with h5py.File(h5_path, "a") as h5:
        grp = h5.require_group("Metadata")
        for key, value in asdict(metadata).items():
            grp.attrs[key] = value


def write_mode(
    img: np.ndarray,
    out_dir: Path,
    mode: str,
    seed: int,
    chrysocolla_solid_fraction: float,
    calcite_solid_fraction: float,
    layer_depth: int,
    cluster_smoothing_passes: int,
    cluster_pore_bias: float,
    pore_diffusivity: float,
    solid_diffusivity: float,
    pore_porosity: float,
    solid_porosity: float,
) -> None:
    chrys, calc, quartz, meta = assign_exclusive_minerals(
        img=img,
        mode=mode,
        seed=seed,
        chrysocolla_solid_fraction=chrysocolla_solid_fraction,
        calcite_solid_fraction=calcite_solid_fraction,
        layer_depth=layer_depth,
        cluster_smoothing_passes=cluster_smoothing_passes,
        cluster_pore_bias=cluster_pore_bias,
    )
    out_h5 = out_dir / f"{mode}.h5"
    out_png = out_dir / f"{mode}.png"
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    fields = write_pflotran_h5(
        img,
        out_h5,
        chrys,
        calc,
        quartz,
        pore_diffusivity=pore_diffusivity,
        solid_diffusivity=solid_diffusivity,
        pore_porosity=pore_porosity,
        solid_porosity=solid_porosity,
    )
    save_metadata(out_h5, meta)
    interface_flat = fields["interface_flat"]
    porosity = fields["porosity_flat"]
    plot_geometry(img, chrys, calc, quartz, out_png, interface_flat=interface_flat)

    closure = float(np.max(np.abs(porosity + chrys + calc + quartz - 1.0)))
    print(
        f"{mode}: chrys={meta.chrysocolla_cells} calcite={meta.calcite_cells} "
        f"quartz={meta.quartz_cells} closure={closure:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paint mutually exclusive mineral realizations.")
    parser.add_argument(
        "--geometry-npy",
        type=Path,
        default=Path("Chip_Designs/network1/geometry/network1_correct.npy"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Chip_Designs/network1/inputs/chrysocolla_exclusive_realization"),
    )
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed. Different seeds change spatial realization, not target fractions.",
    )
    parser.add_argument(
        "--chrysocolla-solid-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of solid cells assigned to chrysocolla. "
            "Higher values make clustered mode grow thicker/less sparse patches. "
            "Recommended end-members: sparse=0.05, intermediate=0.10-0.15, dense=0.20-0.30."
        ),
    )
    parser.add_argument(
        "--calcite-solid-fraction",
        type=float,
        default=0.05,
        help=(
            "Fraction of solid cells assigned to calcite after chrysocolla assignment. "
            "Recommended end-members: low=0.05, medium=0.10, high=0.20."
        ),
    )
    parser.add_argument(
        "--layer-depth",
        type=int,
        default=2,
        help=(
            "Depth control for surface mode and clustered seed zone. "
            "Larger values allow seeds farther from the pore wall. "
            "Recommended end-members: shallow=1-2, moderate=3-4, deep=5+."
        ),
    )
    parser.add_argument(
        "--cluster-smoothing-passes",
        type=int,
        default=5,
        help=(
            "Patchiness knob for clustered mode: lower -> noisier/smaller patches; "
            "higher -> smoother/larger blobs. "
            "Recommended end-members: very patchy=0-1, balanced=3-6, very smooth=8-12."
        ),
    )
    parser.add_argument(
        "--cluster-pore-bias",
        type=float,
        default=2.0,
        help=(
            "Pore-wall confinement for clustered mode:"
             "- higher --cluster-pore-bias (20) = shallower, more pore-wall-hugging clusters"
             "- lower --cluster-pore-bias (5) = deeper inward growth "
             "Recommended end-members: low bias=1-5, moderate bias=5-10, high bias=10-20."),
    )
    parser.add_argument("--pore-porosity", type=float, default=0.99)
    parser.add_argument("--solid-porosity", type=float, default=0.01)
    parser.add_argument("--pore-diffusivity", type=float, default=1.0e-9)
    parser.add_argument("--solid-diffusivity", type=float, default=1.0e-11)
    args = parser.parse_args()

    img = load_geometry_npy(args.geometry_npy)
    for i, mode in enumerate(args.modes):
        write_mode(
            img=img,
            out_dir=args.out_dir,
            mode=mode,
            seed=args.seed + i,
            chrysocolla_solid_fraction=args.chrysocolla_solid_fraction,
            calcite_solid_fraction=args.calcite_solid_fraction,
            layer_depth=args.layer_depth,
            cluster_smoothing_passes=args.cluster_smoothing_passes,
            cluster_pore_bias=args.cluster_pore_bias,
            pore_diffusivity=args.pore_diffusivity,
            solid_diffusivity=args.solid_diffusivity,
            pore_porosity=args.pore_porosity,
            solid_porosity=args.solid_porosity,
        )


if __name__ == "__main__":
    main()
