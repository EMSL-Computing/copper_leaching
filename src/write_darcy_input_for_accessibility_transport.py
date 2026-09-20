"""Write PFLOTRAN input for fixed-inventory chrysocolla realizations.

This is an additive writer for HDF5 files created by
generate_chrysocolla_accessibility_realizations.py. It leaves the existing
transport writer unchanged and patches only the Chrysocolla2 kinetic/constraint
details needed by this workflow.

Example:
    python write_darcy_input_for_accessibility_transport.py \
      --base-geometry accessibility_real000.h5 \
      --out-geometry geometry_accessibility.h5 \
      --out-input mineral_dissolution.in \
      --pore-perm 1.7e-9 \
      --dx 20e-6 \
      --p-in 101326 \
      --p-out 101325
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py

from write_darcy_input_for_transport import (
    write_darcy_input_for_transport,
    read_diffusivities_from_h5
)



def require_chrysocolla_dataset(h5_path: Path) -> None:
    """Validate that the geometry has the single chrysocolla VF dataset."""
    with h5py.File(h5_path, "r") as h5:
        if "Chrysocolla2_VF" not in h5:
            raise KeyError(f"{h5_path} is missing required dataset: Chrysocolla2_VF")


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    """Replace a required template block exactly once."""
    if old not in text:
        raise ValueError(f"Could not find block to replace: {label}")
    return text.replace(old, new, 1)


def rewrite_for_single_chrysocolla(
    text: str,
    ssa_chrysocolla: float,
    rate_constant: str,
) -> str:
    """Patch the generated input while keeping a single Chrysocolla2 mineral."""
    kinetics_block = "RATE_CONSTANT 1.d-10 mol/cm^2-sec"
    single_kinetics = f"RATE_CONSTANT {rate_constant} mol/cm^2-sec"
    text = replace_exact(text, kinetics_block, single_kinetics, "RATE_CONSTANT")

    constraint_block = "Chrysocolla2  DATASET chrysocolla_vf  1.d0 cm^2/cm^3"
    single_constraint_block = f"Chrysocolla2  DATASET chrysocolla_vf  {ssa_chrysocolla:.12g}d0 cm^2/cm^3"
    text = replace_exact(text, constraint_block, single_constraint_block, "initial ssa_chrysocolla")
    return text


def write_accessibility_transport_input(
    base_geometry: Path,
    out_geometry: Path,
    out_input: Path,
    geometry_npy: Path | None,
    pore_perm: float,
    dx: float,
    p_in: float,
    p_out: float,
    final_time_s: float,
    output_period_s: float,
    pulse_duration_s: float | None,
    initial_dt_s: float,
    max_dt_s: float,
    solid_perm: float,
    dz: float | None,
    ssa_chrysocolla: float,
    rate_constant: str,
    restart_file: Path | None,
    restart_time_s: float | None,
    checkpoint_time_s: float | None,
) -> None:
    """Write geometry copy and PFLOTRAN input for one Chrysocolla2 mineral."""
    require_chrysocolla_dataset(base_geometry)
    pore_diffusivity, solid_diffusivity = read_diffusivities_from_h5(base_geometry)
    out_input.parent.mkdir(parents=True, exist_ok=True)
    out_geometry.parent.mkdir(parents=True, exist_ok=True)

    write_darcy_input_for_transport(
        pflotran_input=out_input,
        base_geom_h5=base_geometry,
        out_geom_h5=out_geometry,
        pore_perm=pore_perm,
        dx=dx,
        p_in=p_in,
        p_out=p_out,
        final_time_s=final_time_s,
        output_period_s=output_period_s,
        pulse_duration_s=pulse_duration_s,
        initial_dt_s=initial_dt_s,
        max_dt_s=max_dt_s,
        solid_perm=solid_perm,
        pore_diffusivity=pore_diffusivity,
        solid_diffusivity=solid_diffusivity,
        dz=dz,
        geometry_npy=geometry_npy,
        restart_file=restart_file,
        restart_time_s=restart_time_s,
        checkpoint_time_s=checkpoint_time_s,
    )

    text = out_input.read_text()
    text = rewrite_for_single_chrysocolla(
        text,
        ssa_chrysocolla=ssa_chrysocolla,
        rate_constant=rate_constant,
    )
    out_input.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write PFLOTRAN input for one Chrysocolla2 mineral.")
    parser.add_argument("--base-geometry", type=Path, required=True)
    parser.add_argument("--geometry-npy", type=Path, default=None)
    parser.add_argument("--out-geometry", type=Path, required=True)
    parser.add_argument("--out-input", type=Path, required=True)
    parser.add_argument("--pore-perm", type=float, required=True)
    parser.add_argument("--solid-perm", type=float, default=1.0e-20)
    parser.add_argument("--dx", type=float, default=20.0e-6)
    parser.add_argument("--dz", type=float, default=1.0e-3)
    parser.add_argument("--p-in", type=float, required=True)
    parser.add_argument("--p-out", type=float, required=True)
    parser.add_argument("--final-time-s", type=float, default=7.0 * 24.0 * 3600.0)
    parser.add_argument("--output-period-s", type=float, default=1800.0)
    parser.add_argument("--pulse-duration-s", type=float, default=None)
    parser.add_argument("--initial-dt-s", type=float, default=1.0e-3)
    parser.add_argument("--max-dt-s", type=float, default=1800.0)
    parser.add_argument("--ssa-chrysocolla", type=float, default=100.0)
    parser.add_argument("--rate-constant", default="1.d-10")
    parser.add_argument("--restart-file", type=Path, default=None)
    parser.add_argument("--restart-time-s", type=float, default=None)
    parser.add_argument("--checkpoint-time-s", type=float, default=None)
    args = parser.parse_args()

    write_accessibility_transport_input(
        base_geometry=args.base_geometry,
        out_geometry=args.out_geometry,
        out_input=args.out_input,
        geometry_npy=args.geometry_npy,
        pore_perm=args.pore_perm,
        dx=args.dx,
        p_in=args.p_in,
        p_out=args.p_out,
        final_time_s=args.final_time_s,
        output_period_s=args.output_period_s,
        pulse_duration_s=args.pulse_duration_s,
        initial_dt_s=args.initial_dt_s,
        max_dt_s=args.max_dt_s,
        solid_perm=args.solid_perm,
        dz=args.dz,
        ssa_chrysocolla=args.ssa_chrysocolla,
        rate_constant=args.rate_constant,
        restart_file=args.restart_file,
        restart_time_s=args.restart_time_s,
        checkpoint_time_s=args.checkpoint_time_s,
    )


if __name__ == "__main__":
    main()
