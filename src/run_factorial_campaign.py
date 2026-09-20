"""Run factorial dP x rate campaign on accessibility_ensemble/ dataset.

Prereq: python migrate_paper_dataset.py

Example:
    python run_factorial_campaign.py info
    python run_factorial_campaign.py pending
    python run_factorial_campaign.py case --index 0 --skip-plots
    python run_factorial_campaign.py summarize
"""

from __future__ import annotations

import argparse
import fcntl
import sys
from datetime import datetime
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import numpy as np

from paper_dataset_lib import (
    DATASET_ROOT,
    FACTORIAL_ROOT,
    FactorialCase,
    k_pore_npz_calibrated,
    k_pore_npz_candidates,
    load_factorial_cases,
    write_factorial_manifest,
)
from run_dp_rate_pilot_sweep import (
    DX,
    DZ,
    FINAL_TIME_S,
    LX,
    LY,
    LZ,
    OUTPUT_PERIOD_S,
    P_OUT,
    PULSE_DURATION_S,
    REFERENCE_DP,
    REFERENCE_DURATION_S,
    SSA_CHRYSOCOLLA,
    load_k_pore,
    pflotran_rate,
    reference_is_valid,
    resolve_pflotran_exe,
    run_cmd,
)
from write_darcy_input_for_accessibility_transport import write_accessibility_transport_input

from _repo import REPO_ROOT, SRC_DIR

HERE = REPO_ROOT


def log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def resolve_k_pore_npz(geom_seed: int) -> Path | None:
    for candidate in k_pore_npz_candidates(geom_seed):
        if reference_is_valid(candidate):
            return candidate
    return None


def ensure_k_pore(case: FactorialCase) -> Path:
    ref_dir = DATASET_ROOT / "ensemble" / "shared" / "k_pore" / f"geom{case.geom_seed}"
    ref_dir.mkdir(parents=True, exist_ok=True)
    existing = resolve_k_pore_npz(case.geom_seed)
    if existing is not None:
        return existing

    lock_path = ref_dir / ".calibrate.lock"
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            existing = resolve_k_pore_npz(case.geom_seed)
            if existing is None:
                raise RuntimeError(f"Missing valid k_pore for geom{case.geom_seed}")
            return existing
        existing = resolve_k_pore_npz(case.geom_seed)
        if existing is not None:
            return existing
        log(f"Calibrating k_pore geom{case.geom_seed}")
        run_cmd(
            [
                sys.executable,
                "darcy_vs_stokes/simulate_darcy.py",
                "--geometry-h5",
                str(case.mineral_h5),
                "--geometry-npy",
                str(case.pore_solid_npy),
                "--workdir",
                str(ref_dir),
                "--final-time",
                str(REFERENCE_DURATION_S),
                "--dP",
                str(REFERENCE_DP),
                "--output-period",
                "1",
            ],
            log_path=ref_dir / "reference.log",
        )
        ref_npz = k_pore_npz_calibrated(case.geom_seed)
        if not reference_is_valid(ref_npz):
            raise RuntimeError(f"Calibration failed: {ref_npz}")
        return ref_npz


def run_case(case: FactorialCase, pflotran_exe: Path, *, skip_plots: bool) -> None:
    if case.has_output():
        log(f"SKIP [{case.index:03d}] already complete: {case.case_dir()}")
        return
    ref_npz = ensure_k_pore(case)
    k_pore = load_k_pore(ref_npz)
    workdir = case.case_dir()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "figs").mkdir(exist_ok=True)
    log(
        f"RUN [{case.index:03d}] geom={case.geom_seed} l_r={case.l_r_median_um:.1f}um "
        f"dP={case.dP_Pa:g} rate={case.rate_mol_cm2_s:.0e}"
    )
    write_accessibility_transport_input(
        base_geometry=case.mineral_h5,
        out_geometry=workdir / "geometry_accessibility.h5",
        out_input=workdir / "mineral_dissolution.in",
        geometry_npy=case.pore_solid_npy,
        pore_perm=k_pore,
        dx=DX,
        p_in=P_OUT + case.dP_Pa,
        p_out=P_OUT,
        final_time_s=float(FINAL_TIME_S),
        output_period_s=OUTPUT_PERIOD_S,
        pulse_duration_s=float(PULSE_DURATION_S),
        initial_dt_s=1.0,
        max_dt_s=OUTPUT_PERIOD_S,
        solid_perm=1.0e-20,
        dz=DZ,
        ssa_chrysocolla=SSA_CHRYSOCOLLA,
        rate_constant=pflotran_rate(case.rate_mol_cm2_s),
        restart_file=None,
        restart_time_s=None,
        checkpoint_time_s=None,
    )
    run_cmd([str(pflotran_exe), "-pflotranin", str(workdir / "mineral_dissolution.in")])
    if skip_plots:
        return
    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "plot_accessibility_mineral_dissolution.py"),
            str(workdir / "mineral_dissolution.h5"),
            "--out-dir",
            str(workdir / "figs"),
            "--prefix",
            workdir.name,
            "--dP",
            str(case.dP_Pa),
            "--lx",
            str(LX),
            "--ly",
            str(LY),
            "--lz",
            str(LZ),
        ]
    )


def summarize() -> Path:
    import h5py

    from plot_accessibility_mineral_dissolution import chrysocolla_inventory_series
    from plot_recovery_metrics import outlet_inlet_flux_metrics, parse_time_groups, pick_field

    cases = load_factorial_cases()
    rows = []
    v_bulk = LX * LY * LZ
    pulse_s = float(PULSE_DURATION_S)
    for case in cases:
        row = {
            "case_index": case.index,
            "geom_seed": case.geom_seed,
            "candidate_id": case.candidate_id,
            "phi": case.phi,
            "l_r_median_um": case.l_r_median_um,
            "dP_Pa": case.dP_Pa,
            "rate_mol_cm2_s": case.rate_mol_cm2_s,
            "recovery_pct": np.nan,
            "theta_48h": np.nan,
            "k_pore": np.nan,
            "complete": int(case.has_output()),
        }
        h5_path = case.case_dir() / "mineral_dissolution.h5"
        ref = resolve_k_pore_npz(case.geom_seed)
        if ref is not None:
            row["k_pore"] = load_k_pore(ref)
        if h5_path.is_file():
            with h5py.File(h5_path, "r") as h5:
                time_entries = parse_time_groups(h5)
                g0 = h5[time_entries[0][1]]
                cu_key = pick_field(g0, "Total_Cu++", required=True)
                h_key = pick_field(g0, "Free_H+", required=True)
                series = chrysocolla_inventory_series(h5, time_entries, lx=LX, ly=LY, lz=LZ)
                flux = outlet_inlet_flux_metrics(h5, time_entries, cu_key, h_key, LY, LZ)
                t_s = flux["t_s"]
                q = flux["Qout"]
                dt = np.diff(t_s, prepend=0.0)
                pv = np.cumsum(q * dt) / v_bulk
                ip = int(np.argmin(np.abs(t_s - pulse_s)))
                row["recovery_pct"] = float(series["total_recovery_pct"][-1])
                row["theta_48h"] = float(pv[ip])
        rows.append(row)
    csv_path = FACTORIAL_ROOT / "results.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")
    write_factorial_manifest(cases)
    log(f"Saved {csv_path} ({sum(r['complete'] for r in rows)}/{len(rows)} complete)")
    return csv_path


def cmd_info() -> None:
    cases = load_factorial_cases()
    complete = sum(1 for c in cases if c.has_output())
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Cases: {len(cases)}  complete: {complete}  pending: {len(cases) - complete}")
    print(f"SLURM --array=0-{len(cases) - 1}")
    for c in cases[:5]:
        flag = "ok" if c.has_output() else "pending"
        print(
            f"  [{c.index:03d}] geom={c.geom_seed} dP={c.dP_Pa:g} "
            f"rate={c.rate_mol_cm2_s:.0e} [{flag}]"
        )
    if len(cases) > 5:
        print("  ...")


def cmd_pending() -> None:
    for c in load_factorial_cases():
        if not c.has_output():
            print(c.index)


def preflight() -> None:
    """Validate prerequisites for all pending factorial cases."""
    cases = load_factorial_cases()
    pending = [c for c in cases if not c.has_output()]
    errors: list[str] = []
    for case in pending:
        label = f"[{case.index:03d}] geom{case.geom_seed} dP={case.dP_Pa:g} rate={case.rate_mol_cm2_s:.0e}"
        if not case.mineral_h5.is_file():
            errors.append(f"{label}: missing {case.mineral_h5}")
        if not case.pore_solid_npy.is_file():
            errors.append(f"{label}: missing {case.pore_solid_npy}")
        if resolve_k_pore_npz(case.geom_seed) is None:
            errors.append(f"{label}: missing k_pore")
        try:
            case.case_dir().mkdir(parents=True, exist_ok=True)
            probe = case.case_dir() / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            errors.append(f"{label}: not writable ({exc})")
    print(f"Pending: {len(pending)}")
    print(f"Errors: {len(errors)}")
    if errors:
        for line in errors[:20]:
            print(line)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more")
        raise SystemExit(1)
    try:
        resolve_pflotran_exe()
        print("PFLOTRAN: ok")
    except RuntimeError as exc:
        raise SystemExit(f"PFLOTRAN: {exc}") from exc
    print("Preflight passed.")


def smoke_case(index: int, *, final_time_s: float = 60.0) -> None:
    """Short PFLOTRAN run in _smoke/ (does not touch production case_dir)."""
    cases = load_factorial_cases()
    case = cases[index]
    ref_npz = ensure_k_pore(case)
    k_pore = load_k_pore(ref_npz)
    workdir = FACTORIAL_ROOT / "_smoke" / f"index{index:03d}_geom{case.geom_seed}"
    if workdir.exists():
        import shutil

        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log(
        f"SMOKE [{index:03d}] geom={case.geom_seed} dP={case.dP_Pa:g} "
        f"rate={case.rate_mol_cm2_s:.0e} final_time={final_time_s:g}s -> {workdir}"
    )
    write_accessibility_transport_input(
        base_geometry=case.mineral_h5,
        out_geometry=workdir / "geometry_accessibility.h5",
        out_input=workdir / "mineral_dissolution.in",
        geometry_npy=case.pore_solid_npy,
        pore_perm=k_pore,
        dx=DX,
        p_in=P_OUT + case.dP_Pa,
        p_out=P_OUT,
        final_time_s=final_time_s,
        output_period_s=min(OUTPUT_PERIOD_S, final_time_s),
        pulse_duration_s=min(float(PULSE_DURATION_S), final_time_s),
        initial_dt_s=1.0,
        max_dt_s=min(OUTPUT_PERIOD_S, final_time_s),
        solid_perm=1.0e-20,
        dz=DZ,
        ssa_chrysocolla=SSA_CHRYSOCOLLA,
        rate_constant=pflotran_rate(case.rate_mol_cm2_s),
        restart_file=None,
        restart_time_s=None,
        checkpoint_time_s=None,
    )
    pflotran_exe = resolve_pflotran_exe()
    run_cmd([str(pflotran_exe), "-pflotranin", str(workdir / "mineral_dissolution.in")])
    h5 = workdir / "mineral_dissolution.h5"
    if not h5.is_file():
        raise RuntimeError(f"Smoke failed: missing {h5}")
    prod = case.case_dir()
    if (prod / "mineral_dissolution.h5").is_file():
        raise RuntimeError(f"Smoke must not write production path: {prod}")
    log(f"Smoke OK: {h5} ({h5.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Factorial campaign on accessibility_ensemble/")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    sub.add_parser("pending")
    sub.add_parser("preflight")
    sub.add_parser("summarize")
    p_smoke = sub.add_parser("smoke", help="60s PFLOTRAN in _smoke/ only.")
    p_smoke.add_argument("--index", type=int, required=True)
    p_smoke.add_argument("--final-time-s", type=float, default=60.0)
    p_case = sub.add_parser("case")
    p_case.add_argument("--index", type=int, required=True)
    p_case.add_argument("--skip-plots", action="store_true")
    p_case.add_argument("--force", action="store_true", help="Re-run even if output exists.")
    args = parser.parse_args()

    if args.command == "info":
        cmd_info()
        return
    if args.command == "pending":
        cmd_pending()
        return
    if args.command == "preflight":
        preflight()
        return
    if args.command == "smoke":
        smoke_case(args.index, final_time_s=args.final_time_s)
        return
    if args.command == "summarize":
        summarize()
        return

    cases = load_factorial_cases()
    case = cases[args.index]
    if case.has_output() and not args.force:
        log(f"Already complete: {case.case_dir()}")
        return
    run_case(case, resolve_pflotran_exe(), skip_plots=args.skip_plots)
    write_factorial_manifest(cases)
    log("Done")


if __name__ == "__main__":
    main()