"""
Example:
python run_compare_Darcy_Stokes_flow/calibrate_k_pore.py --geometry-h5 pore_geometry.h5 --geometry-npy pore_geometry.npy --dx 20e-6 --final-time 100 
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
try:
    from darcy_vs_stokes.utils import effective_permeability
    from darcy_vs_stokes.darcy_utils import run_darcy, write_darcy_input_for_flow, load_darcy_case
    from darcy_vs_stokes.stokes_utils import run_stokes, load_stokes_reference
except ModuleNotFoundError:
    from utils import effective_permeability
    from darcy_utils import run_darcy, write_darcy_input_for_flow, load_darcy_case
    from stokes_utils import run_stokes, load_stokes_reference

HERE = Path(__file__).resolve().parent

def calibrate_k_pore(
    geometry_h5,
    geometry_npy,
    workdir_path,
    dx,
    mu=1e-3,
    dP=1.0,
    final_time: float = 10,
):
    """One-shot calibration: returns k_pore and the generated Stokes npz path."""
    geometry_h5 = Path(geometry_h5)
    geometry_npy = Path(geometry_npy)
    workdir = workdir_path / "calibrate_k_pore_run"
    workdir.mkdir(exist_ok=True)

    stokes_npz = workdir / "stokes_reference.npz"
    run_stokes(geometry_npy, dx, mu, dP, stokes_npz, final_time=final_time)
    stokes = load_stokes_reference(stokes_npz)
    mask = np.asarray(stokes["mask"])
    k_eff_stokes = effective_permeability(float(stokes["q_ref"]), mask, dx, mu, dP)

    k_pore_trial = 1.0e-9
    solid_perm = 1.0e-20
    pore_poro = float(np.load(geometry_npy).mean())
    solid_poro = 1.0e-2

    out_geom_h5 = workdir / "geometry_calibrate.h5"

    pflotran_input_file = workdir / "darcy_trial.in"
    p_out = 101325.0
    p_in = p_out + dP

    write_darcy_input_for_flow(
        pflotran_input_file,
        geometry_h5,
        out_geom_h5,
        k_pore_trial,
        solid_perm,
        dx,
        1.0e-3,
        p_in,
        p_out,
        pulse_duration=float(stokes["pulse_duration"]),
        final_time=final_time,
        output_period=1
    )
    run_darcy(pflotran_input_file)

    darcy = load_darcy_case(
        workdir / "darcy_trial.h5",
        target_time_s=float(np.asarray(stokes["times"])[-1]),
        mask=mask,
        dx=dx,
        pressure_in=p_in,
        pressure_out=p_out,
    )
    k_eff_darcy = effective_permeability(float(darcy["q_ref"]), mask, dx, mu, dP)

    k_pore_opt = k_pore_trial * (k_eff_stokes / k_eff_darcy)
    # Persist calibrated k_pore in the Stokes reference so skip-calibrate can reuse it.
    stokes_to_save = dict(stokes)
    stokes_to_save["k_pore"] = float(k_pore_opt)
    np.savez(stokes_npz, **stokes_to_save)
    return k_pore_opt, stokes_npz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate PFLOTRAN pore permeability by matching Stokes Q."
    )
    parser.add_argument(
        "--geometry-h5",
        type=Path,                          # was missing type entirely
        required=True,                      # was passing variable not string
        help="Pore geometry HDF5 file written by build_pflotran_hdf5_from_npy.py",
    )
    parser.add_argument(
        "--geometry-npy",
        type=Path,                          # was missing type entirely
        required=True,                      # was passing variable not string
        help="Pore geometry .npy written by en_pore_geometry.py",
    )
    parser.add_argument("--workdir", type=Path, default=HERE)
    parser.add_argument(
        "--dx",                             # was positional (missing --), now optional with default
        type=float,
        default=20.0e-6,
        help="Uniform cell size in x and y [m] (default 20e-6)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1.0e-3,
        help="Dynamic viscosity [Pa·s] (default 1e-3 for water)",
    )
    parser.add_argument(
        "--dP",
        type=float,
        default=1.0,
        help="Applied pressure drop inlet minus outlet [Pa] (default 1.0)",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=100,
        help="Absolute Stokes/Darcy run duration [s] (overrides tfinal-mult)",
    )
    args = parser.parse_args()
    k_pore, _ = calibrate_k_pore(
        args.geometry_h5,
        args.geometry_npy,
        args.workdir,
        args.dx,
        args.mu,
        args.dP,
        final_time=args.final_time,
    )
    print(f"k_pore_optimal = {k_pore:.6e} m2")

if __name__ == "__main__":
    main()
