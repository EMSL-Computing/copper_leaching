"""Darcy-Stokes SI workflow: geom437 validation and ensemble Stokes stats.

Example:
    python run_tracer.py plot
    python run_tracer.py ensemble
    python run_tracer.py run
    python run_tracer.py run --long
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _si_flow_root() -> Path:
    for cand in (HERE, *HERE.parents):
        if (cand / "config.json").is_file() and (cand / "run_tracer.py").is_file():
            return cand
    return HERE

SI_FLOW_ROOT = _si_flow_root()
REPO_ROOT = SI_FLOW_ROOT.parents[1]  # src/si_flow -> src -> repo
DEFAULT_CONFIG = SI_FLOW_ROOT / "config.json"
_SRC = REPO_ROOT / "src"
for _p in (str(_SRC), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Imports resolve within si_flow/ (HERE on path via SI_FLOW_ROOT)
if str(SI_FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(SI_FLOW_ROOT))

from plot_darcy_stokes import load_config, plot_all
from plot_darcy_ensemble_perm import load_config as load_ensemble_config, run_all as run_ensemble_all
from plot_chrysocolla_map import load_config as load_chrys_config, plot_chrysocolla_map
from plot_connected_chrysocolla_map import (
    load_config as load_conn_chrys_config,
    plot_connected_chrysocolla_map,
)
from plot_connected_pore_map import load_config as load_conn_pore_config, plot_connected_pore_map
from plot_flow_direction import load_config as load_flow_dir_config, plot_flow_direction


def log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_cmd(args: list[str]) -> None:
    log(f"RUN: {' '.join(args)}")
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def run_short(cfg: dict) -> None:
    """Calibrate k_pore and run short Darcy+tracer into reference_dir."""
    log(f"Short calibration run ({cfg['final_time_s']} s) -> {cfg['reference_dir']}")
    run_cmd(_simulate_darcy_cmd(cfg, workdir=cfg["reference_dir"], final_time=cfg["final_time_s"], output_period=cfg["output_period_s"]))


def run_long(cfg: dict) -> None:
    """Long Stokes tracer + PFLOTRAN Darcy into tracer_long_dir."""
    if not cfg["stokes_npz"].is_file():
        raise FileNotFoundError(f"Run short calibration first: missing {cfg['stokes_npz']}")

    long_dir = cfg["tracer_long_dir"]
    cal_dir = long_dir / "calibrate_k_pore_run"
    cal_dir.mkdir(parents=True, exist_ok=True)
    ref_link = cal_dir / "stokes_reference.npz"
    ref_link.unlink(missing_ok=True)
    ref_link.symlink_to(cfg["stokes_npz"])

    stokes_long = long_dir / "stokes_tracer.npz"
    if not stokes_long.is_file():
        log(f"Long Stokes tracer ({cfg['tracer_final_time_s']} s)")
        from darcy_vs_stokes.stokes_utils import run_stokes

        run_stokes(
            cfg["geometry_npy"],
            float(cfg["dx_m"]),
            float(cfg["mu_Pa_s"]),
            float(cfg["dP_Pa"]),
            stokes_long,
            final_time=float(cfg["tracer_final_time_s"]),
        )

    log(f"Long Darcy tracer ({cfg['tracer_final_time_s']} s) -> {long_dir}")
    run_cmd(
        _simulate_darcy_cmd(
            cfg,
            workdir=long_dir,
            final_time=cfg["tracer_final_time_s"],
            output_period=cfg["tracer_output_period_s"],
            skip_calibrate=True,
        )
    )


def _simulate_darcy_cmd(
    cfg: dict,
    *,
    workdir: Path,
    final_time: float,
    output_period: float,
    skip_calibrate: bool = False,
) -> list[str]:
    args = [
        sys.executable,
        "darcy_vs_stokes/simulate_darcy.py",
        "--geometry-h5", str(cfg["geometry_h5"]),
        "--geometry-npy", str(cfg["geometry_npy"]),
        "--workdir", str(workdir),
        "--dP", str(cfg["dP_Pa"]),
        "--final-time", str(final_time),
        "--output-period", str(output_period),
        "--pore-diffusivity", str(cfg["pore_diffusivity"]),
        "--solid-diffusivity", str(cfg["solid_diffusivity"]),
    ]
    if skip_calibrate:
        args.append("--skip-calibrate")
    return args


def _darcy_h5_for_cfg(cfg: dict, *, long: bool) -> Path:
    if long:
        return cfg["tracer_long_dir"] / "simulate_darcy_run" / "darcy_calibrated.h5"
    return cfg["reference_dir"] / "simulate_darcy_run" / "darcy_calibrated.h5"


def run_animate(cfg: dict, *, long: bool = False) -> None:
    """Build tracer GIF from calibrated Darcy HDF5 output."""
    h5_path = _darcy_h5_for_cfg(cfg, long=long)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Missing Darcy tracer output: {h5_path}")

    out_gif = cfg["tracer_animation_out"]
    if long:
        out_gif = cfg["tracer_long_dir"] / "darcy_tracer_animation.gif"

    run_cmd(
        [
            sys.executable,
            "animate_tracer.py",
            str(h5_path),
            "--geometry-npy",
            str(cfg["geometry_npy"]),
            "--geometry-h5",
            str(cfg["geometry_h5"]),
            "-o",
            str(out_gif),
            "--field",
            "Total_Tracer [M]",
            "--fps",
            str(cfg["tracer_animation_fps"]),
            "--pore-diffusivity",
            str(cfg["pore_diffusivity"]),
            "--solid-diffusivity",
            str(cfg["solid_diffusivity"]),
        ]
    )


def _resolve_cfg(config_path: Path) -> dict:
    cfg = load_config(config_path)
    base = config_path.parent
    cfg["tracer_long_dir"] = (base / cfg["tracer_long_dir"]).resolve()
    cfg["tracer_animation_out"] = (base / cfg["tracer_animation_out"]).resolve()
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Darcy-Stokes geom437 SI workflow")
    parser.add_argument(
        "command",
        choices=(
            "plot",
            "run",
            "animate",
            "tracer",
            "ensemble",
            "flow-direction",
            "connected-pore",
            "chrysocolla-map",
            "connected-chrysocolla",
        ),
        default="plot",
        nargs="?",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--long", action="store_true", help="With run/animate/tracer: use 7200 s tracer_long/ outputs")
    parser.add_argument(
        "--skip-calibrate",
        action="store_true",
        help="With run/tracer: reuse existing k_pore from reference/",
    )

    parser.add_argument("--flow-only", action="store_true")
    parser.add_argument("--btc-only", action="store_true")
    args = parser.parse_args()

    cfg = _resolve_cfg(args.config.resolve())

    if args.command == "plot":
        flow = btc = True
        if args.flow_only:
            btc = False
        if args.btc_only:
            flow = False
        plot_all(cfg, flow=flow, btc=btc)
    elif args.command == "ensemble":
        ens_cfg = load_ensemble_config(args.config.resolve())
        run_ensemble_all(ens_cfg)
    elif args.command == "run":
        if args.long:
            run_long(cfg)
        elif args.skip_calibrate:
            log(f"Darcy tracer ({cfg['final_time_s']} s, skip calibrate) -> {cfg['reference_dir']}")
            run_cmd(
                _simulate_darcy_cmd(
                    cfg,
                    workdir=cfg["reference_dir"],
                    final_time=cfg["final_time_s"],
                    output_period=cfg["output_period_s"],
                    skip_calibrate=True,
                )
            )
        else:
            run_short(cfg)
    elif args.command == "animate":
        run_animate(cfg, long=args.long)
    elif args.command == "tracer":
        if args.long:
            run_long(cfg)
        elif args.skip_calibrate:
            log(f"Darcy tracer ({cfg['final_time_s']} s, skip calibrate) -> {cfg['reference_dir']}")
            run_cmd(
                _simulate_darcy_cmd(
                    cfg,
                    workdir=cfg["reference_dir"],
                    final_time=cfg["final_time_s"],
                    output_period=cfg["output_period_s"],
                    skip_calibrate=True,
                )
            )
        else:
            run_short(cfg)
        run_animate(cfg, long=args.long)
    elif args.command == "flow-direction":
        plot_flow_direction(load_flow_dir_config(args.config.resolve()))
    elif args.command == "connected-pore":
        plot_connected_pore_map(load_conn_pore_config(args.config.resolve()))
    elif args.command == "chrysocolla-map":
        plot_chrysocolla_map(load_chrys_config(args.config.resolve()))
    elif args.command == "connected-chrysocolla":
        plot_connected_chrysocolla_map(load_conn_chrys_config(args.config.resolve()))


if __name__ == "__main__":
    main()