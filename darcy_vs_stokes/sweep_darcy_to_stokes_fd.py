"""Sweep Darcy contrasts against 2DStokes_FD and rank best match.

Edit CONFIG below. No CLI clutter.
"""
#%%
from __future__ import annotations

import csv
import itertools
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from darcy_stokes_compare import compute_metrics, load_darcy_case, load_stokes_reference, plot_comparison
from darcy_stokes_summary import print_case_summary, print_reference_summary
from write_darcy_tracer_input import build_input_text, load_grid_shape

# Ensure stdout is unbuffered so output appears in real time when redirected
sys.stdout.reconfigure(line_buffering=True)
#%%

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def resolve_pflotran_exe() -> Path:
    """Find the PFLOTRAN executable from PATH or a local build."""
    env_path = shutil.which("pflotran")
    if env_path:
        return Path(env_path)

    local_path = Path("/home/vscode/src/pflotran/src/pflotran/pflotran")
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        "PFLOTRAN executable not found. Activate the env that provides 'pflotran' "
        "or set PATH to the build directory."
    )


PFLOTRAN_EXE = resolve_pflotran_exe()


CONFIG = {
    "geometry_h5": ROOT / "pore_geometry.h5",
    "geometry_npy": ROOT / "pore_geometry.npy",
    "stokes_npz": HERE / "stokes_fd_reference.npz",
    "workdir": HERE / "sweep_fd_runs",
    "csv_out": HERE / "darcy_stokes_fd_sweep.csv",
    "plot_out": HERE / "darcy_stokes_fd_best.png",
    "dx": 20.0e-6,
    "dz": 1.0e-3,
    "pressure_in": 201326.0,
    "pressure_out": 201325.0,
    "pulse_fraction": 0.05,
    "tfinal_mult": 50.0,
    "h5_snapshots": 400,
    "rerun_stokes": True,
    "rerun_darcy": True,
    "pore_perm_values": [1.2e-9, 1.4e-9, 1.6e-9, 1.8e-9, 2.0e-9],
    "solid_perm_values": [1.0e-20],
    "pore_poro_values": None,
    "solid_poro_values": [1.0e-2],
}


def run_cmd(args: list[str], workdir: Path) -> str:
    """Run one command and return merged stdout."""
    result = subprocess.run(
        args,
        cwd=workdir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{result.stdout}")
    return result.stdout


def run_stokes_reference() -> dict[str, np.ndarray | float]:
    """Run or load 2DStokes_FD reference."""
    geometry_npy = Path(CONFIG["geometry_npy"])
    stokes_npz = Path(CONFIG["stokes_npz"])
    if bool(CONFIG["rerun_stokes"]) or not stokes_npz.exists():
        run_cmd(
            [
                sys.executable,
                str(ROOT / "2DStokes_FD" / "stokes_solver_2D.py"),
                "-g",
                str(geometry_npy),
                "--dx",
                str(CONFIG["dx"]),
                "--results-out",
                str(stokes_npz),
                "--pulse-fraction",
                str(CONFIG["pulse_fraction"]),
                "--tfinal-mult",
                str(CONFIG["tfinal_mult"]),
                "--gif-frames",
                "0",
            ],
            ROOT,
        )

    return load_stokes_reference(stokes_npz, dx_fallback=float(CONFIG["dx"]))


def copy_geometry_with_contrast(
    base_geom_h5: Path,
    out_geom_h5: Path,
    pore_perm: float,
    solid_perm: float,
    pore_poro: float,
    solid_poro: float,
) -> None:
    """Copy geometry and overwrite Darcy material properties."""
    shutil.copyfile(base_geom_h5, out_geom_h5)
    with h5py.File(out_geom_h5, "r+") as h5:
        mids = np.asarray(h5["Materials"]["Material Ids"])
        perm = np.where(mids == 1, pore_perm, solid_perm)
        poro = np.where(mids == 1, pore_poro, solid_poro)
        h5["Permeability"][...] = perm
        h5["Porosity"][...] = poro


def write_darcy_input(
    input_path: Path,
    geom_name: str,
    final_time_s: float,
    pulse_duration_s: float,
    output_period_s: float,
    max_dt_s: float,
) -> None:
    """Write Darcy tracer input."""
    nx, ny = load_grid_shape(Path(CONFIG["geometry_h5"]))
    text = build_input_text(
        geometry_h5=Path(geom_name),
        nx=nx,
        ny=ny,
        dx=float(CONFIG["dx"]),
        dz=float(CONFIG["dz"]),
        p_in=float(CONFIG["pressure_in"]),
        p_out=float(CONFIG["pressure_out"]),
        pulse_duration=pulse_duration_s,
        final_time=final_time_s,
        output_period=output_period_s,
        initial_dt=min(1.0e-3, pulse_duration_s * 0.1),
        max_dt=max_dt_s,
        diffusion_coefficient=0.0,
        background_concentration=1.0e-20,
        tracer_on_concentration=1.0,
    )
    input_path.write_text(text)


def run_darcy_case(case_input: Path) -> None:
    """Run PFLOTRAN Darcy tracer case."""
    stem = case_input.stem
    for suffix in [".h5", "-obs-0.pft", "-mas.dat", "-chk.h5"]:
        old_path = case_input.parent / f"{stem}{suffix}"
        old_path.unlink(missing_ok=True)
    run_cmd([str(PFLOTRAN_EXE), "-pflotranin", case_input.name], case_input.parent)


#%%
def main() -> None:
    """Run Darcy sweep against 2DStokes_FD."""
    workdir = Path(CONFIG["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)

    stokes = run_stokes_reference()
    final_time_s = float(np.asarray(stokes["times"])[-1])
    t_adv = final_time_s / float(CONFIG["tfinal_mult"])
    pulse_duration_s = float(CONFIG["pulse_fraction"]) * t_adv
    max_dt_s = max(0.05 * t_adv, 1.0e-3)
    h5_output_period_s = max(
        max_dt_s,
        final_time_s / (float(CONFIG["h5_snapshots"]) + 0.5),
    )
    print_reference_summary(stokes)
    print(f"  h5_output_period = {h5_output_period_s:.4e} s (~{int(final_time_s / h5_output_period_s)} snapshots)")
    print(f"  max_dt          = {max_dt_s:.4e} s (~{int(final_time_s / max_dt_s)} solver steps)")

    base_geom_h5 = Path(CONFIG["geometry_h5"])
    pore_poro_vals = CONFIG["pore_poro_values"]
    if pore_poro_vals is None:
        pore_poro_vals = [float(np.asarray(stokes["mask"]).mean())]
        print(f"  pore_poro       = {pore_poro_vals[0]:.4f} (from geometry porosity)")

    rows = []
    best_score = float("inf")
    best_row = None
    best_darcy = None

    cases = itertools.product(
        CONFIG["pore_perm_values"],
        CONFIG["solid_perm_values"],
        pore_poro_vals,
        CONFIG["solid_poro_values"],
    )

    for pore_perm, solid_perm, pore_poro, solid_poro in cases:
        label = (
            f"pp{pore_perm:.1e}_sp{solid_perm:.1e}_"
            f"pf{pore_poro:.3g}_sf{solid_poro:.3g}"
        )
        case_geom = workdir / f"{label}_geom.h5"
        case_in = workdir / f"{label}.in"
        case_h5 = workdir / f"{label}.h5"

        copy_geometry_with_contrast(
            base_geom_h5,
            case_geom,
            pore_perm,
            solid_perm,
            pore_poro,
            solid_poro,
        )
        write_darcy_input(
            case_in, case_geom.name, final_time_s, pulse_duration_s,
            h5_output_period_s, max_dt_s,
        )
        t0 = time.time()
        if bool(CONFIG["rerun_darcy"]) or not case_h5.exists():
            run_darcy_case(case_in)
            runtime_s = time.time() - t0
        else:
            runtime_s = 0.0
        darcy = load_darcy_case(
            case_h5,
            final_time_s,
            np.asarray(stokes["mask"]),
            float(CONFIG["dx"]),
            float(CONFIG["pressure_in"]),
            float(CONFIG["pressure_out"]),
        )
        metrics = compute_metrics(stokes, darcy)
        row = {
            "label": label,
            "pore_perm": pore_perm,
            "solid_perm": solid_perm,
            "pore_poro": pore_poro,
            "solid_poro": solid_poro,
            **metrics,
        }
        rows.append(row)

        print_case_summary(
            label,
            row,
            stokes,
            darcy,
            metrics,
            {"runtime_s": runtime_s},
        )

        if metrics["score"] < best_score:
            best_score = metrics["score"]
            best_row = row
            best_darcy = darcy

    rows.sort(key=lambda item: float(item["score"]))
    with Path(CONFIG["csv_out"]).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if best_row is not None and best_darcy is not None:
        plot_comparison(
            Path(CONFIG["plot_out"]),
            stokes,
            best_darcy,
            metadata={
                "title": "Darcy vs 2DStokes_FD  |  best sweep case",
                "label": best_row["label"],
                "score": float(best_row["score"]),
                "pressure_rel_rmse": float(best_row["pressure_rel_rmse"]),
                "speed_rel_rmse": float(best_row["speed_rel_rmse"]),
                "btc_rel_rmse": float(best_row["btc_rel_rmse"]),
                "speed_scale_err": float(best_row["speed_scale_err"]),
                "speed_rank_corr": float(best_row["speed_rank_corr"]),
                "fast_channel_overlap": float(best_row["fast_channel_overlap"]),
                "stokes_k_eff": float(best_row["stokes_k_eff"]),
            },
        )
        print(f"Best case: {best_row['label']}")
        print(f"Score: {best_row['score']:.6e}")

    print(f"Wrote {CONFIG['csv_out']}")
    print(f"Wrote {CONFIG['plot_out']}")


if __name__ == "__main__":
    main()
