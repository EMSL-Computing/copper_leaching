"""Tools for Darcy vs 2DStokes_FD comparison.

"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
import sys

import numpy as np

try:
    from darcy_vs_stokes.utils import run_cmd
except ModuleNotFoundError:
    from utils import run_cmd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run_stokes(
    geometry_npy: Path,
    dx: float,
    mu: float,
    dP: float,
    out_npz: Path,
    final_time: float | None = None,
    tfinal_mult: float = 3.0,
) -> None:
    """Run the Stokes reference with an absolute duration when provided."""
    cmd = [
        sys.executable,
        str(ROOT / "2DStokes_FD" / "stokes_solver_2D.py"),
        "-g",
        str(geometry_npy),
        "--dx",
        str(dx),
        "--mu",
        str(mu),
        "--dP",
        str(dP),
        "--gif-frames",
        "0",
        "--results-out",
        str(out_npz),
    ]
    if final_time is None:
        cmd.extend(["--tfinal-mult", str(tfinal_mult)])
    else:
        cmd.extend(["--final-time", str(final_time)])
    run_cmd(cmd, ROOT)


def load_stokes_reference(npz_path: Path) -> dict[str, np.ndarray | float]:
    data = np.load(npz_path)
    return {
        "mask": data["mask"],
        "u": data["u"],
        "v": data["v"],
        "p": data["p"],
        "q_cols": data["q_cols"] if "q_cols" in data.files else np.array([]),
        "q_ref": float(data["q_ref"]),
        "times": data["times"],
        "right_flux": data["right_flux"],
        "total_mass": data["total_mass"] if "total_mass" in data.files else np.array([]),
        "dx": float(data["dx"]) if "dx" in data.files else np.nan,
        "mu": float(data["mu"]) if "mu" in data.files else 1.0e-3,
        "dP": float(data["dP"]) if "dP" in data.files else 1.0,
        "dt": float(data["dt"]) if "dt" in data.files else float("nan"),
        "nsteps": int(data["nsteps"]) if "nsteps" in data.files else -1,
        "save_stride": int(data["save_stride"]) if "save_stride" in data.files else -1,
        "pulse_duration": float(data["pulse_duration"]) if "pulse_duration" in data.files else float("nan"),
        "t_adv": float(data["t_adv"]) if "t_adv" in data.files else float("nan"),
        "k_pore": float(data["k_pore"]) if "k_pore" in data.files else float("nan"),
    }
