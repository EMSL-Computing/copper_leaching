"""Run a 3x3 dP x rate pilot on one bean_bucket accessibility realization.

Cases: dP = {0.001, 0.1, 1} Pa x rate = {1e-10, 5e-10, 1e-9} mol/cm^2-s

Example:
    python run_dp_rate_pilot_sweep.py all
    python run_dp_rate_pilot_sweep.py case --index 0
    python run_dp_rate_pilot_sweep.py summarize # Run summarize only (reads finished mineral_dissolution.h5 files, no PFLOTRAN)

SLURM (parallel cases + summarize):
    sbatch run_dp_rate_pilot_sweep.slurm
    sbatch --dependency=afterok:<array_jobid> run_dp_rate_pilot_summarize.slurm

Post-processing (automatic):
  Per case: plot_accessibility_mineral_dissolution.py -> dp_*_rate_*/figs/ (depletion + porosity snapshots)
  After all cases: summarize -> pilot_results.csv + dp_rate_pilot_heatmaps.png + dp_rate_pilot_regime.png

What to look for:
  - pilot_results.csv / heatmap (a): domain Cu recovery [%] — kinetic + transport control
  - heatmap (b): PV swept at 48 h — is advection sufficient (Theta ~ 1)?
  - dP=0.001 row: expect ~0 recovery at all rates (sweep-limited)
  - dP=0.1 vs 1: transport contrast; rate columns should separate most at dP=1
  - Per-case figs: inlet depletion vs outlet BTC; check acid reaches outlet via diffusion
  - reference/reference.log: k_pore calibration; shared across all cases

# Manual postprocessing:

python - <<'PY'
import subprocess, sys
from run_dp_rate_pilot_sweep import DP_VALUES, RATE_VALUES, LX, LY, LZ, workdir_for_case, resolve_pilot_paths

paths = resolve_pilot_paths()
for dp in DP_VALUES:
    for rate in RATE_VALUES:
        workdir = workdir_for_case(paths, dp, rate)
        prefix = workdir.name
        h5 = workdir / "mineral_dissolution.h5"
        subprocess.run([
            sys.executable, "plot_accessibility_mineral_dissolution.py",
            str(h5),
            "--out-dir", str(workdir / "figs"),
            "--prefix", prefix,
            "--dP", str(dp),
            "--lx", str(LX), "--ly", str(LY), "--lz", str(LZ),
        ], check=True)
PY

# for just on case:
python plot_accessibility_mineral_dissolution.py \
  Chip_Designs/bean_bucket_network/sweeps/dp_rate_pilot/chrysocolla_real000/dp_1p0_rate_1em09/mineral_dissolution.h5 \
  --out-dir Chip_Designs/bean_bucket_network/sweeps/dp_rate_pilot/chrysocolla_real000/dp_1p0_rate_1em09/figs \
  --prefix dp_1p0_rate_1em09 \
  --dP 1.0 --lx 0.008 --ly 0.002 --lz 0.001

"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_recovery_metrics import (
    initial_cu_inventory,
    outlet_inlet_flux_metrics,
    parse_time_groups,
    pick_field,
)
from prepare_accessibility_geometry import (
    AccessibilityGeometryConfig,
    prepare_accessibility_geometry,
)
from plot_accessibility_mineral_dissolution import chrysocolla_inventory_series
from write_darcy_input_for_accessibility_transport import write_accessibility_transport_input

from _repo import REPO_ROOT, SRC_DIR

HERE = REPO_ROOT

DP_VALUES = (0.001, 0.1, 1.0)
RATE_VALUES = (1e-10, 5e-10, 1e-9)
NETWORK_NAME = "bean_bucket_network"
REALIZATION_INDEX = 0

REFERENCE_DP = 1.0
REFERENCE_DURATION_S = 10.0
FINAL_TIME_S = 3600 * 24 * 10
OUTPUT_PERIOD_S = 3600.0
PULSE_DURATION_S = 3600 * 48
P_OUT = 101325.0
SSA_CHRYSOCOLLA = 100.0
DX = 20e-6
DZ = 1e-3
LX, LY, LZ = 8e-3, 2e-3, 1e-3


@dataclass(frozen=True)
class PilotPaths:
    network_dir: Path
    geom_npy: Path
    geom_h5: Path
    out_root: Path

    @classmethod
    def from_geometry(cls, geom_paths, *, realization_index: int = REALIZATION_INDEX) -> PilotPaths:
        case_name = f"chrysocolla_real{realization_index:03d}"
        return cls(
            network_dir=geom_paths.network_dir,
            geom_npy=geom_paths.geom_npy,
            geom_h5=geom_paths.realization_h5(realization_index),
            out_root=geom_paths.network_dir / "sweeps" / "dp_rate_pilot" / case_name,
        )


def default_geometry_config() -> AccessibilityGeometryConfig:
    return AccessibilityGeometryConfig(
        network_name=NETWORK_NAME,
        geometry_source="existing",
        n_realizations=REALIZATION_INDEX + 1,
        seed=42,
        chrysocolla_solid_fraction=0.50,
        pore_diffusivity=1.0e-9,
        solid_diffusivity=1.0e-11,
    )


def resolve_pilot_paths() -> PilotPaths:
    geom_paths = prepare_accessibility_geometry(default_geometry_config())
    paths = PilotPaths.from_geometry(geom_paths)
    log(f"Using geometry: {paths.geom_h5}")
    return paths


def log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_cmd(args: list[str], *, cwd: Path | None = None, log_path: Path | None = None) -> None:
    log(f"RUN: {' '.join(args)}")
    if log_path is None:
        subprocess.run(args, cwd=cwd or HERE, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        subprocess.run(args, cwd=cwd or HERE, check=True, stdout=log_file, stderr=subprocess.STDOUT)
    log(f"Saved log: {log_path}")


def tag_dp(dp: float) -> str:
    return str(dp).replace("-", "neg_").replace(".", "p")


def tag_rate(rate: float) -> str:
    return f"{rate:.0e}".replace("+", "").replace("-", "m")


def pflotran_rate(rate: float) -> str:
    return f"{rate:.1e}".replace("e", "d")


def map_case_index(index: int) -> tuple[float, float]:
    n_rates = len(RATE_VALUES)
    dp = DP_VALUES[index // n_rates]
    rate = RATE_VALUES[index % n_rates]
    return dp, rate


def resolve_pflotran_exe() -> Path:
    exe = shutil.which("pflotran")
    if exe:
        return Path(exe)
    pflotran_dir = os.environ.get("PFLOTRAN_DIR")
    if pflotran_dir:
        candidate = Path(pflotran_dir) / "pflotran"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("PFLOTRAN executable not found")


def reference_npz(paths: PilotPaths) -> Path:
    return paths.out_root / "reference" / "calibrate_k_pore_run" / "stokes_reference.npz"


def ensure_reference(paths: PilotPaths) -> Path:
    """Calibrate k_pore once (geometry property; independent of sweep dP)."""
    ref_root = paths.out_root / "reference"
    ref_npz = reference_npz(paths)
    ref_root.mkdir(parents=True, exist_ok=True)

    if reference_is_valid(ref_npz):
        log(f"Reusing k_pore reference: {ref_npz}")
        return ref_npz
    if ref_npz.is_file():
        log(f"Invalid k_pore reference; recalibrating: {ref_npz}")

    lock_path = ref_root / ".calibrate.lock"
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Waiting for in-flight k_pore calibration")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not reference_is_valid(ref_npz):
                raise RuntimeError(f"Calibration finished but missing valid {ref_npz}")
            return ref_npz

        if reference_is_valid(ref_npz):
            log("Reference appeared while acquiring calibration lock")
            return ref_npz

        log(f"Calibrating k_pore (reference dP={REFERENCE_DP:g} Pa)")
        run_cmd(
            [
                sys.executable,
                "darcy_vs_stokes/simulate_darcy.py",
                "--geometry-h5",
                str(paths.geom_h5),
                "--geometry-npy",
                str(paths.geom_npy),
                "--workdir",
                str(ref_root),
                "--final-time",
                str(REFERENCE_DURATION_S),
                "--dP",
                str(REFERENCE_DP),
                "--output-period",
                "1",
            ],
            log_path=ref_root / "reference.log",
        )
        if not reference_is_valid(ref_npz):
            raise RuntimeError(f"Calibration failed to produce valid {ref_npz}")
        return ref_npz


def load_k_pore(ref_npz: Path) -> float:
    data = np.load(ref_npz, allow_pickle=False)
    if "k_pore" not in data.files:
        raise KeyError(f"missing k_pore in {ref_npz}")
    k_pore = float(data["k_pore"])
    if not np.isfinite(k_pore):
        raise ValueError(f"non-finite k_pore in {ref_npz}: {k_pore}")
    return k_pore


def reference_is_valid(ref_npz: Path) -> bool:
    if not ref_npz.is_file():
        return False
    try:
        k_pore = load_k_pore(ref_npz)
    except (KeyError, ValueError, OSError):
        return False
    return k_pore > 0.0


def workdir_for_case(paths: PilotPaths, dp: float, rate: float) -> Path:
    return paths.out_root / f"dp_{tag_dp(dp)}_rate_{tag_rate(rate)}"


def run_one_case(paths: PilotPaths, dp: float, rate: float, pflotran_exe: Path) -> None:
    ref_npz = ensure_reference(paths)
    k_pore = load_k_pore(ref_npz)
    p_in = P_OUT + dp
    workdir = workdir_for_case(paths, dp, rate)
    prefix = workdir.name
    fig_dir = workdir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log(f"Case dP={dp:g} Pa, rate={rate:.0e} mol/cm^2-s (k_pore={k_pore:.6e}) -> {workdir}")
    write_accessibility_transport_input(
        base_geometry=paths.geom_h5,
        out_geometry=workdir / "geometry_accessibility.h5",
        out_input=workdir / "mineral_dissolution.in",
        geometry_npy=paths.geom_npy,
        pore_perm=k_pore,
        dx=DX,
        p_in=p_in,
        p_out=P_OUT,
        final_time_s=float(FINAL_TIME_S),
        output_period_s=OUTPUT_PERIOD_S,
        pulse_duration_s=float(PULSE_DURATION_S),
        initial_dt_s=1.0,
        max_dt_s=OUTPUT_PERIOD_S,
        solid_perm=1.0e-20,
        dz=DZ,
        ssa_chrysocolla=SSA_CHRYSOCOLLA,
        rate_constant=pflotran_rate(rate),
        restart_file=None,
        restart_time_s=None,
        checkpoint_time_s=None,
    )

    run_cmd([str(pflotran_exe), "-pflotranin", str(workdir / "mineral_dissolution.in")])
    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "plot_accessibility_mineral_dissolution.py"),
            str(workdir / "mineral_dissolution.h5"),
            "--out-dir",
            str(fig_dir),
            "--prefix",
            prefix,
            "--dP",
            str(dp),
        ]
    )


def plot_pilot_regime_figure(
    rows: list[tuple[float, float, float, float]],
    out_png: Path,
) -> None:
    """Regime figure: recovery vs Theta (a) and recovery vs rate at dP=1 Pa (b)."""
    if not rows:
        return

    rate_colors = {r: c for r, c in zip(RATE_VALUES, sns.color_palette("colorblind", n_colors=len(RATE_VALUES)))}
    dp_markers = {0.001: "o", 0.1: "s", 1.0: "^"}

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True, sharey=True)

    for rate in RATE_VALUES:
        pts = sorted((theta, rec, dp) for r, dp, rec, theta in rows if r == rate)
        if not pts:
            continue
        thetas, recs, dps = zip(*pts)
        axes[0].plot(thetas, recs, "-", color=rate_colors[rate], lw=1.6, zorder=1)
        for theta, rec, dp in pts:
            axes[0].scatter(
                theta,
                rec,
                c=[rate_colors[rate]],
                marker=dp_markers.get(dp, "o"),
                s=70,
                edgecolors="black",
                linewidths=0.4,
                zorder=2,
            )

    rate_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=rate_colors[r], markersize=8, label=f"{r:.0e}")
        for r in RATE_VALUES
    ]
    dp_handles = [
        plt.Line2D([0], [0], marker=dp_markers[d], color="gray", linestyle="None", markersize=8, label=f"{d:g} Pa")
        for d in DP_VALUES
    ]
    axes[0].set_xlabel(r"$\Theta$ at 48 h (PV swept / $V_{\mathrm{bulk}}$)")
    axes[0].set_ylabel("Domain Cu recovery [% initial Cu]")
    axes[0].set_title("(a) Transport exposure vs recovery", loc="left")
    leg1 = axes[0].legend(handles=rate_handles, title="rate [mol/cm$^2$/s]", loc="upper left", fontsize=8)
    axes[0].add_artist(leg1)
    axes[0].legend(handles=dp_handles, title="dP", loc="lower right", fontsize=8)

    dp_colors = {d: c for d, c in zip(DP_VALUES, sns.color_palette("dark", n_colors=len(DP_VALUES)))}
    for dp in DP_VALUES:
        pts = sorted((rate, rec) for rate, d, rec, _ in rows if d == dp)
        if not pts:
            continue
        rates, recs = zip(*pts)
        axes[1].plot(rates, recs, "o-", color=dp_colors[dp], lw=1.6, ms=8, label=f"{dp:g} Pa")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Rate constant [mol/cm$^2$/s]")
    # axes[1].set_ylabel("Domain Cu recovery [% initial Cu]")
    axes[1].set_title("(b) Recovery vs rate by dP", loc="left")
    axes[1].legend(title="dP", loc="upper left", fontsize=8)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def summarize_pilot(paths: PilotPaths) -> None:
    log("Building pilot summary table and heatmaps")
    v_bulk = LX * LY * LZ
    pulse_s = float(PULSE_DURATION_S)

    rows: list[tuple[float, float, float, float]] = []
    recovery_pct = np.full((len(RATE_VALUES), len(DP_VALUES)), np.nan)
    pv_pulse = np.full_like(recovery_pct, np.nan)

    for i, rate in enumerate(RATE_VALUES):
        for j, dp in enumerate(DP_VALUES):
            workdir = workdir_for_case(paths, dp, rate)
            h5_path = workdir / "mineral_dissolution.h5"
            geom_path = workdir / "geometry_accessibility.h5"
            if not h5_path.is_file():
                log(f"WARNING missing {h5_path}")
                continue

            with h5py.File(h5_path, "r") as h5:
                time_entries = parse_time_groups(h5)
                g0 = h5[time_entries[0][1]]
                cu_key = pick_field(g0, "Total_Cu++", required=True)
                h_key = pick_field(g0, "Free_H+", required=True)
                series = chrysocolla_inventory_series(h5, time_entries, lx=LX, ly=LY, lz=LZ)
                rec = float(series["total_recovery_pct"][-1])
                _, pore_vol = initial_cu_inventory(
                    h5,
                    time_entries[0][1],
                    pick_field(g0, "Chrysocolla2_VF", required=True),
                    str(geom_path),
                    LX,
                    LY,
                    LZ,
                )
                flux = outlet_inlet_flux_metrics(h5, time_entries, cu_key, h_key, LY, LZ)

                t_s = flux["t_s"]
                q = flux["Qout"]
                dt = np.diff(t_s, prepend=0.0)
                pv = np.cumsum(q * dt) / v_bulk
                ip = int(np.argmin(np.abs(t_s - pulse_s)))
                theta = float(pv[ip])

            recovery_pct[i, j] = rec
            pv_pulse[i, j] = theta
            rows.append((rate, dp, rec, theta))
            log(f"dP={dp:g} rate={rate:.0e}: recovery={rec:.3f}% PV@48h={theta:.6f}")

    paths.out_root.mkdir(parents=True, exist_ok=True)
    csv_path = paths.out_root / "pilot_results.csv"
    with csv_path.open("w") as f:
        f.write("rate_constant_mol_cm2_s,dP_Pa,recovery_pct,PV_at_pulse_end\n")
        for rate, dp, rec, theta in rows:
            f.write(f"{rate},{dp},{rec},{theta}\n")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
    xlabels = [str(d) for d in DP_VALUES]
    ylabels = [f"{r:.0e}" for r in RATE_VALUES]

    sns.heatmap(
        recovery_pct,
        annot=True,
        fmt=".2f",
        cmap="magma",
        xticklabels=xlabels,
        yticklabels=ylabels,
        ax=axes[0],
        cbar_kws={"label": "Domain recovery [% initial Cu]"},
    )
    axes[0].set_xlabel("dP [Pa]")
    axes[0].set_ylabel("RATE_CONSTANT [mol/cm^2/s]")
    axes[0].set_title("(a) Domain Cu recovery", loc="left")

    sns.heatmap(
        pv_pulse,
        annot=True,
        fmt=".4f",
        cmap="viridis",
        xticklabels=xlabels,
        yticklabels=ylabels,
        ax=axes[1],
        cbar_kws={"label": "PV swept at 48 h"},
    )
    axes[1].set_xlabel("dP [Pa]")
    axes[1].set_ylabel("RATE_CONSTANT [mol/cm^2/s]")
    axes[1].set_title("(b) Theta at pulse end", loc="left")

    fig_path = paths.out_root / "dp_rate_pilot_heatmaps.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    log(f"Saved {fig_path}")

    regime_path = paths.out_root / "dp_rate_pilot_regime.png"
    plot_pilot_regime_figure(rows, regime_path)
    log(f"Saved {regime_path}")
    log(f"Saved {csv_path}")


def validate_paths(paths: PilotPaths) -> None:
    if not paths.geom_h5.is_file():
        raise FileNotFoundError(f"Missing realization geometry: {paths.geom_h5}")
    if not paths.geom_npy.is_file():
        raise FileNotFoundError(f"Missing solver geometry: {paths.geom_npy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bean_bucket dP x rate pilot sweep.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("all", help="Run all cases sequentially and summarize.")
    sub.add_parser("summarize", help="Build CSV and heatmaps from finished cases.")

    case_parser = sub.add_parser("case", help="Run one case by index (0-8) for SLURM array.")
    case_parser.add_argument("--index", type=int, required=True)

    args = parser.parse_args()
    paths = resolve_pilot_paths()
    validate_paths(paths)
    paths.out_root.mkdir(parents=True, exist_ok=True)
    pflotran_exe = resolve_pflotran_exe()

    n_cases = len(DP_VALUES) * len(RATE_VALUES)

    if args.command == "case":
        if args.index < 0 or args.index >= n_cases:
            raise SystemExit(f"case index must be in [0, {n_cases - 1}]")
        dp, rate = map_case_index(args.index)
        log(f"Pilot case {args.index}: dP={dp:g} rate={rate:.0e}")
        run_one_case(paths, dp, rate, pflotran_exe)
    elif args.command == "summarize":
        summarize_pilot(paths)
    elif args.command == "all":
        for idx in range(n_cases):
            dp, rate = map_case_index(idx)
            run_one_case(paths, dp, rate, pflotran_exe)
        summarize_pilot(paths)

    log(f"Done ({args.command})")


if __name__ == "__main__":
    main()