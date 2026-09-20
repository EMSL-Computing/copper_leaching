"""Calibrate PFLOTRAN pore permeability, run Darcy, and plot Stokes comparison.

Example:
python run_compare_Darcy_Stokes_flow/simulate_darcy.py \
  --geometry-h5 pore_geometry.h5 \
  --geometry-npy pore_geometry.npy \
  --plot-out run_compare_Darcy_Stokes_flow/darcy_compare.png \
  -- final-time 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

try:
    from darcy_vs_stokes import calibrate_k_pore as clp
    from darcy_vs_stokes.darcy_stokes_compare import compute_metrics, plot_comparison
    from darcy_vs_stokes.darcy_stokes_summary import print_case_summary, print_reference_summary
    from darcy_vs_stokes.darcy_utils import run_darcy, write_darcy_input_for_flow, load_darcy_case
    from darcy_vs_stokes.stokes_utils import load_stokes_reference
except ModuleNotFoundError:
    import calibrate_k_pore as clp
    from darcy_stokes_compare import compute_metrics, plot_comparison
    from darcy_stokes_summary import print_case_summary, print_reference_summary
    from darcy_utils import run_darcy, write_darcy_input_for_flow, load_darcy_case
    from stokes_utils import load_stokes_reference

HERE = Path(__file__).resolve().parent


def run_simulation(
    geometry_h5: Path,
    geometry_npy: Path,
    plot_out: Path,
    dx: float,
    mu: float,
    dP: float,
    workdir: Path,
    reference_workdir: Path,
    skip_calibrate: bool = False,
    final_time: float =10,
    output_period: float = 1.0,
    pore_diffusivity: float = 1.0e-9,
    solid_diffusivity: float = 1.0e-11,
) -> float:
    """Calibrate k_pore, run Darcy, and write a comparison plot."""
    workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stokes_npz = reference_workdir / "calibrate_k_pore_run" / "stokes_reference.npz"
    if skip_calibrate:
        if not stokes_npz.exists():
            raise FileNotFoundError(
                f"--skip-calibrate requested but Stokes reference not found: {stokes_npz}"
            )
        stokes = load_stokes_reference(stokes_npz)
        k_pore = float(stokes["k_pore"])
        if not np.isfinite(k_pore):
            raise ValueError(
                f"--skip-calibrate requested but k_pore is missing in: {stokes_npz}"
            )
    else:
        k_pore, stokes_npz = clp.calibrate_k_pore(
            geometry_h5,
            geometry_npy,
            reference_workdir,
            dx,
            mu=mu,
            dP=dP,
            final_time=final_time,
        )
        stokes = load_stokes_reference(stokes_npz)
        stokes_to_save = dict(stokes)
        stokes_to_save["k_pore"] = float(k_pore)
        np.savez(stokes_npz, **stokes_to_save)

    print_reference_summary(stokes)

    out_geom_h5 = workdir / "geometry_calibrate.h5"
    pflotran_input_file = workdir / "darcy_calibrated.in"
    p_out = 101325.0
    p_in = p_out + dP

    write_darcy_input_for_flow(
        pflotran_input_file,
        geometry_h5,
        out_geom_h5,
        k_pore,
        1.0e-20,
        dx,
        1.0e-3,
        p_in,
        p_out,
        pulse_duration=float(stokes["pulse_duration"]),
        final_time=final_time,
        output_period=output_period,
        pore_diffusivity=pore_diffusivity,
        solid_diffusivity=solid_diffusivity,
        )
    run_darcy(pflotran_input_file)
    
    pflotran_out_h5 = pflotran_input_file.with_suffix(".h5")
    darcy = load_darcy_case(
        pflotran_out_h5,
        target_time_s=float(np.asarray(stokes["times"])[-1]),
        mask=np.asarray(stokes["mask"]),
        dx=dx,
        pressure_in=p_in,
        pressure_out=p_out,
    )

    metrics = compute_metrics(stokes, darcy)
    print_case_summary(
        pflotran_input_file.stem,
        {"label": pflotran_input_file.stem, "k_pore": float(k_pore)},
        stokes,
        darcy,
        metrics,
        {"runtime_s": time.time() - t0},
    )

    plot_comparison(
        plot_out,
        stokes,
        darcy,
        metadata={
            "title": "Darcy vs 2DStokes_FD",
            "label": pflotran_input_file.stem,
            "k_pore": float(k_pore),
            "score": float(metrics["score"]),
            "pressure_rel_rmse": float(metrics["pressure_rel_rmse"]),
            "speed_rel_rmse": float(metrics["speed_rel_rmse"]),
            "btc_rel_rmse": float(metrics["btc_rel_rmse"]),
            "speed_scale_err": float(metrics["speed_scale_err"]),
            "speed_rank_corr": float(metrics["speed_rank_corr"]),
            "fast_channel_overlap": float(metrics["fast_channel_overlap"]),
            "stokes_k_eff": float(metrics["stokes_k_eff"]),
        }, 
    )
    return k_pore


def main() -> None:
    """Run the calibrated Darcy case and produce a comparison figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-h5", type=Path, required=True)
    parser.add_argument("--geometry-npy", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, default=HERE)
    parser.add_argument("--dx", type=float, default=20.0e-6)
    parser.add_argument("--mu", type=float, default=1.0e-3)
    parser.add_argument("--dP", type=float, default=1.0)
    parser.add_argument(
        "--final-time",
        type=float,
        default=10,
        help="Absolute Stokes/Darcy run duration [s] passed through to calibration.",
    )
    parser.add_argument(
        "--output-period",
        type=float,
        default=1.0,
        help="Output snapshot period [s] for PFLOTRAN tracer snapshots (default: 1.0 s).",
    )
    parser.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="Skip calibration and load k_pore from reference .npz",
    )
    parser.add_argument(
        "--pore-diffusivity",
        type=float,
        default=1.0e-9,
        help="Pore-region tracer diffusivity [m^2/s] (default: 1e-9)",
    )
    parser.add_argument(
        "--solid-diffusivity",
        type=float,
        default=1.0e-11,
        help="Solid-region tracer diffusivity [m^2/s] (default: 1e-11)",
    )
    args = parser.parse_args()

    k_pore = run_simulation(
        args.geometry_h5,
        args.geometry_npy,
        args.workdir / "darcy_compare.png",
        args.dx,
        args.mu,
        args.dP,
        args.workdir/ "simulate_darcy_run",
        args.workdir,
        skip_calibrate=args.skip_calibrate,
        final_time=args.final_time,
        output_period=args.output_period,
        pore_diffusivity=args.pore_diffusivity,
        solid_diffusivity=args.solid_diffusivity,
    )
    print(f"k_pore_optimal = {k_pore:.6e} m2")


if __name__ == "__main__":
    main()
