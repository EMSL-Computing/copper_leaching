"""Build PFLOTRAN geometry HDF5 from a binary pore mask `.npy`.

Workflow
--------
1) `make_grain_image.py` creates `pore_geometry.npy` (1=pore, 0=solid)
2) This script converts that `.npy` into PFLOTRAN-ready `pore_geometry.h5`

Example usage
-------------
python gen_pore_geometry.py --geometry-npy pore_geometry.npy --out pore_geometry.h5

python build_pflotran_hdf5_from_npy.py --geometry-npy pore_geometry.npy --out chrysocolla_realization/clustered_check.h5 --chrysocolla_mode clustered_fracture_localized --n_realizations 1 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from chrysocolla_painting import paint_chrysocolla_fields, plot_geometry, save_metadata_h5


def load_geometry_npy(path: Path) -> np.ndarray:
    """Load and validate a 2-D binary pore mask from `.npy`."""
    img = np.asarray(np.load(path), dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2-D geometry array, got shape {img.shape}")

    unique_vals = np.unique(img)
    if not np.all(np.isin(unique_vals, [0.0, 1.0])):
        raise ValueError(
            f"Geometry must be binary 0/1 (solid/pore). Found values: {unique_vals[:10]}"
        )
    return img


def write_pflotran_h5(
    img: np.ndarray,
    h5_path: Path,
    chrysocolla_vf: np.ndarray,
    calcite_vf: np.ndarray,
    quartz_vf: np.ndarray,
    pore_diffusivity: float,
    solid_diffusivity: float,
    pore_porosity: float,
    solid_porosity: float,
) -> dict[str, np.ndarray]:
    """Write PFLOTRAN-compatible HDF5 datasets for a 2D structured grid."""
    nx, ny = img.shape
    ncells = nx * ny

    flat = img.ravel(order="F")
    perm = np.where(flat == 1.0, 2.0e-9, 1.0e-20)
    poro = np.where(flat == 1.0, pore_porosity, solid_porosity)
    diffusivity = np.where(flat == 1.0, pore_diffusivity, solid_diffusivity)
    pore_interface = np.zeros_like(img, dtype=bool)
    solid = img == 0.0
    pore_interface[:-1, :] |= (~solid[:-1, :]) & solid[1:, :]
    pore_interface[1:, :] |= (~solid[1:, :]) & solid[:-1, :]
    pore_interface[:, :-1] |= (~solid[:, :-1]) & solid[:, 1:]
    pore_interface[:, 1:] |= (~solid[:, 1:]) & solid[:, :-1]

    cell_ids = np.arange(1, ncells + 1, dtype=np.int32)
    material_ids = np.where(flat == 1.0, 1, 2).astype(np.int32)

    interface_flat = pore_interface.ravel(order="F")
    interface_cell_ids = cell_ids[interface_flat]

    pore_solid_interface = cell_ids[flat == 0.0]

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("Cell Ids", data=cell_ids)
        f.create_dataset("Permeability", data=perm)
        f.create_dataset("Porosity", data=poro)
        f.create_dataset("Diffusivity", data=diffusivity)
        f.create_dataset("Chrysocolla2_VF", data=chrysocolla_vf)
        f.create_dataset("Calcite_VF", data=calcite_vf)
        f.create_dataset("Quartz_VF", data=quartz_vf)

        mat_grp = f.create_group("Materials")
        mat_grp.create_dataset("Cell Ids", data=cell_ids)
        mat_grp.create_dataset("Material Ids", data=material_ids)

        reg_grp = f.create_group("Regions")
        grp = reg_grp.create_group("pore_interface_cells")
        grp.create_dataset("Cell Ids", data=interface_cell_ids)
        grp2 = reg_grp.create_group("pore_solid_interface")
        grp2.create_dataset("Cell Ids", data=pore_solid_interface)

        meta_grp = f.create_group("Metadata")
        meta_grp.attrs["grid_shape"] = np.asarray([nx, ny], dtype=np.int64)

    print(
        f"Wrote {h5_path}: {ncells} cells, "
        f"pore fraction = {img.mean():.2f}, "
        f"interface cells = {int(interface_flat.sum())}"
    )
    return {
        "interface_flat": interface_flat,
        "porosity_flat": poro,
        "diffusivity_flat": diffusivity,
    }


def write_mapping_diagnostics(
    img: np.ndarray,
    porosity_flat: np.ndarray,
    diffusivity_flat: np.ndarray,
    pore_diffusivity: float,
    solid_diffusivity: float,
    out_porosity_map: Path,
    out_transport_map: Path,
) -> None:
    """Write side-by-side porosity mapping and transport-property diagnostics."""
    nx, ny = img.shape
    pore_scale_porosity = img.astype(float)
    darcy_porosity = porosity_flat.reshape((nx, ny), order="F")
    darcy_diff = diffusivity_flat.reshape((nx, ny), order="F")

    base_diff = max(float(pore_diffusivity), float(solid_diffusivity))
    pore_tort = float(pore_diffusivity) / base_diff
    solid_tort = float(solid_diffusivity) / base_diff
    darcy_tort = np.where(img == 1.0, pore_tort, solid_tort)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    pore_plot = np.where(pore_scale_porosity > 0.0, pore_scale_porosity, np.nan)
    min_pos_pore = float(np.nanmin(pore_plot)) if np.any(np.isfinite(pore_plot)) else 1.0e-6
    im0 = axes[0].imshow(
        pore_plot.T,
        origin="lower",
        cmap="viridis",
        norm=LogNorm(vmin=min_pos_pore, vmax=1.0),
    )
    axes[0].set_title("(a) Pore-scale porosity (binary, log)", loc="left")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    c0.set_label("porosity [-] (log)")

    min_pos_darcy = max(float(np.nanmin(darcy_porosity[darcy_porosity > 0.0])), 1.0e-12)
    im1 = axes[1].imshow(
        darcy_porosity.T,
        origin="lower",
        cmap="viridis",
        norm=LogNorm(vmin=min_pos_darcy, vmax=float(np.nanmax(darcy_porosity))),
    )
    axes[1].set_title("(b) Darcy-scale porosity (regularized, log)", loc="left")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    c1.set_label("porosity [-] (log)")
    fig.savefig(out_porosity_map, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    min_pos_diff = max(float(np.nanmin(darcy_diff[darcy_diff > 0.0])), 1.0e-30)
    im2 = axes[0].imshow(
        darcy_diff.T,
        origin="lower",
        cmap="magma",
        norm=LogNorm(vmin=min_pos_diff, vmax=float(np.nanmax(darcy_diff))),
    )
    axes[0].set_title("(a) Darcy-cell diffusivity (log)", loc="left")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    c2 = fig.colorbar(im2, ax=axes[0], fraction=0.046, pad=0.04)
    c2.set_label("D [m$^2$/s] (log)")

    min_pos_tort = max(float(np.nanmin(darcy_tort[darcy_tort > 0.0])), 1.0e-30)
    im3 = axes[1].imshow(
        darcy_tort.T,
        origin="lower",
        cmap="cividis",
        norm=LogNorm(vmin=min_pos_tort, vmax=float(np.nanmax(darcy_tort))),
    )
    axes[1].set_title("(b) Darcy-cell tortuosity (log)", loc="left")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    c3 = fig.colorbar(im3, ax=axes[1], fraction=0.046, pad=0.04)
    c3.set_label("tortuosity [-] (log)")

    fig.savefig(out_transport_map, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Read `.npy` geometry and write PFLOTRAN `.h5` plus diagnostics."""
    parser = argparse.ArgumentParser(description="Convert binary .npy geometry to PFLOTRAN .h5")
    parser.add_argument("--geometry-npy", type=Path, required=True, help="Input binary geometry .npy")
    parser.add_argument("--out", type=Path, default=Path("pore_geometry.h5"), help="Output PFLOTRAN HDF5")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for rind heterogeneity")
    parser.add_argument("--layer_depth", type=int, default=2, help="Chrysocolla rind thickness in solid cells")
    parser.add_argument("--rind_coverage", type=float, default=1.0, help="Fraction of rind cells that carry mineral")
    parser.add_argument("--calcite_fraction", type=float, default=0.05, help="Calcite fraction of non-chrysocolla solid volume")
    parser.add_argument("--chrysocolla_solid_fraction", type=float, default=0.05, help="Total chrysocolla as fraction of total solid volume")
    parser.add_argument("--chrysocolla_mode", type=str, default="surface_concentrated",
                        choices=["surface_concentrated", "uniform", "matrix_concentrated", "clustered_fracture_localized"])
    parser.add_argument("--n_realizations", type=int, default=1)
    parser.add_argument("--inward_decay_length", type=float, default=1.25)
    parser.add_argument("--cluster_smoothing_passes", type=int, default=5)
    parser.add_argument("--cluster_pore_bias", type=float, default=2.0)
    parser.add_argument("--pore-diffusivity", type=float, default=1.0e-9, help="Pore diffusion coefficient in PFLOTRAN cells [m^2/s]")
    parser.add_argument("--solid-diffusivity", type=float, default=1.0e-11, help="Solid diffusion coefficient in PFLOTRAN cells [m^2/s]")
    parser.add_argument("--pore-porosity", type=float, default=0.99, help="Porosity in PFLOTRAN pore cells")
    parser.add_argument("--solid-porosity", type=float, default=0.01, help="Porosity in PFLOTRAN solid cells")
    args = parser.parse_args()

    img = load_geometry_npy(args.geometry_npy)
    n_real = max(1, int(args.n_realizations))
    for i in range(n_real):
        seed_i = int(args.seed + i)
        chry_flat, calc_flat, qtz_flat, meta = paint_chrysocolla_fields(
            img=img,
            mode=args.chrysocolla_mode,
            seed=seed_i,
            chrysocolla_solid_fraction=args.chrysocolla_solid_fraction,
            calcite_fraction=args.calcite_fraction,
            layer_depth=args.layer_depth,
            rind_coverage=args.rind_coverage,
            inward_decay_length=args.inward_decay_length,
            cluster_smoothing_passes=args.cluster_smoothing_passes,
            cluster_pore_bias=args.cluster_pore_bias,
            pore_porosity=args.pore_porosity,
            solid_porosity=args.solid_porosity,
        )
        out_h5 = args.out if n_real == 1 else args.out.with_name(f"{args.out.stem}_real{i:03d}.h5")
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        out_png = out_h5.with_suffix(".png")
        fields = write_pflotran_h5(
            img,
            out_h5,
            chrysocolla_vf=chry_flat,
            calcite_vf=calc_flat,
            quartz_vf=qtz_flat,
            pore_diffusivity=args.pore_diffusivity,
            solid_diffusivity=args.solid_diffusivity,
            pore_porosity=args.pore_porosity,
            solid_porosity=args.solid_porosity,
        )
        save_metadata_h5(out_h5, meta)
        plot_geometry(img, chry_flat, calc_flat, qtz_flat, out_png, interface_flat=fields["interface_flat"])
        write_mapping_diagnostics(
            img=img,
            porosity_flat=fields["porosity_flat"],
            diffusivity_flat=fields["diffusivity_flat"],
            pore_diffusivity=args.pore_diffusivity,
            solid_diffusivity=args.solid_diffusivity,
            out_porosity_map=out_h5.with_name(f"{out_h5.stem}_porosity_mapping.png"),
            out_transport_map=out_h5.with_name(f"{out_h5.stem}_diffusivity_tortuosity.png"),
        )
        closure = np.max(np.abs(
            fields["porosity_flat"] + chry_flat + calc_flat + qtz_flat - 1.0
        ))
        total_phi = float(fields["porosity_flat"].sum())
        total_solid = float((1.0 - fields["porosity_flat"]).sum())
        total_mineral = float((chry_flat + calc_flat + qtz_flat).sum())
        print(
            f"Realization {i + 1}/{n_real}: mode={args.chrysocolla_mode}, seed={seed_i}, "
            f"total_chrysocolla={meta.total_chrysocolla_vf:.8f}, closure={closure:.3e}, "
            f"sum_phi={total_phi:.6f}, sum_solid={total_solid:.6f}, sum_mineral={total_mineral:.6f}"
        )


if __name__ == "__main__":
    main()
