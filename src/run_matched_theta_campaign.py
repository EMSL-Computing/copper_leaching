"""Matched-Theta campaign on accessibility_ensemble/ (no Chip_Designs paths).

Per-geom dP holds Theta @ 48 h ~ constant; rate varies by branch (4a, 4b, 4c, ...).

Example:
    python run_matched_theta_campaign.py plan-branch --branch 4c --rate 5e-10 --from-branch 4a
    python run_matched_theta_campaign.py info --branch 4c
    python run_matched_theta_campaign.py preflight --branch 4c
    python run_matched_theta_campaign.py smoke --branch 4c --index 0
    python run_matched_theta_campaign.py case --branch 4c --index 0 --skip-plots
    sbatch run_matched_theta_campaign.slurm
"""

from __future__ import annotations

import argparse
import fcntl
import sys
from datetime import datetime
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import numpy as np

from paper_dataset_lib import (
    DATASET_ROOT,
    MATCHED_THETA_ANCHOR_DP,
    MATCHED_THETA_ANCHOR_RATE,
    MATCHED_THETA_ROOT,
    MatchedThetaCase,
    clamp_matched_theta_dp,
    k_pore_npz_calibrated,
    k_pore_npz_candidates,
    load_factorial_cases,
    load_matched_theta_cases,
    matched_theta_manifest_path,
    write_matched_theta_manifest,
)
from plot_accessibility_mineral_dissolution import chrysocolla_inventory_series
from plot_recovery_metrics import outlet_inlet_flux_metrics, parse_time_groups, pick_field
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
V_BULK = LX * LY * LZ
PULSE_S = float(PULSE_DURATION_S)
SMOKE_ROOT = REPO_ROOT / "archive" / "smoke_tests" / "_smoke_matched_theta"


def log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def resolve_k_pore_npz(geom_seed: int) -> Path | None:
    for candidate in k_pore_npz_candidates(geom_seed):
        if reference_is_valid(candidate):
            return candidate
    return None


def read_theta_48h(h5_path: Path) -> float:
    with h5py.File(h5_path, "r") as h5:
        time_entries = parse_time_groups(h5)
        g0 = h5[time_entries[0][1]]
        cu_key = pick_field(g0, "Total_Cu++", required=True)
        h_key = pick_field(g0, "Free_H+", required=True)
        flux = outlet_inlet_flux_metrics(h5, time_entries, cu_key, h_key, LY, LZ)
        t_s = flux["t_s"]
        q = flux["Qout"]
        dt = np.diff(t_s, prepend=0.0)
        pv = np.cumsum(q * dt) / V_BULK
        ip = int(np.argmin(np.abs(t_s - PULSE_S)))
        return float(pv[ip])


def ensure_k_pore(case: MatchedThetaCase) -> Path:
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


def run_case(case: MatchedThetaCase, pflotran_exe: Path, *, skip_plots: bool) -> None:
    if case.has_output():
        log(f"SKIP [{case.index:03d}] already complete: {case.case_dir()}")
        return
    ref_npz = ensure_k_pore(case)
    k_pore = load_k_pore(ref_npz)
    workdir = case.case_dir()
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "figs").mkdir(exist_ok=True)
    log(
        f"RUN [{case.index:03d}] branch={case.branch} geom={case.geom_seed} "
        f"l_r={case.l_r_median_um:.1f}um dP={case.dP_Pa:g} rate={case.rate_mol_cm2_s:.0e}"
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
            f"geom{case.geom_seed}_{case.branch}",
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


def plan_anchor(*, branch: str, rate: float, theta_target: float | None) -> Path:
    """Plan matched-Theta dP from factorial anchor (dP=1 Pa, k_r=1e-9) thetas."""
    anchor = [
        c
        for c in load_factorial_cases()
        if abs(c.dP_Pa - MATCHED_THETA_ANCHOR_DP) < 1e-12
        and abs(c.rate_mol_cm2_s - MATCHED_THETA_ANCHOR_RATE) < 1e-15
        and c.has_output()
    ]
    if len(anchor) < 2:
        raise SystemExit(
            "Need >=2 complete factorial anchor cases at dP=1 Pa, k_r=1e-9. "
            "Finish factorial anchor column first or use plan-branch --from-branch 4a."
        )
    anchor = sorted(anchor, key=lambda c: c.realization_index)
    thetas = [read_theta_48h(c.case_dir() / "mineral_dissolution.h5") for c in anchor]
    target = float(np.median(thetas)) if theta_target is None else float(theta_target)

    cases: list[MatchedThetaCase] = []
    for i, ac in enumerate(anchor):
        theta_ref = thetas[i]
        dP_new = clamp_matched_theta_dp(MATCHED_THETA_ANCHOR_DP * target / max(theta_ref, 1e-12))
        cases.append(
            MatchedThetaCase(
                index=i,
                branch=branch,
                geom_seed=ac.geom_seed,
                realization_index=ac.realization_index,
                candidate_id=ac.candidate_id,
                phi=ac.phi,
                l_r_median_um=ac.l_r_median_um,
                dP_Pa=dP_new,
                rate_mol_cm2_s=rate,
                theta_target_48h=target,
                pore_solid_npy=ac.pore_solid_npy,
                mineral_h5=ac.mineral_h5,
            )
        )
    path = write_matched_theta_manifest(
        branch,
        cases,
        extra={
            "source": "run_matched_theta_campaign.plan_anchor",
            "theta_target_48h": target,
            "anchor_dp_Pa": MATCHED_THETA_ANCHOR_DP,
            "anchor_rate_mol_cm2_s": MATCHED_THETA_ANCHOR_RATE,
            "note": "dP per geom scales anchor Theta to target median; rate fixed per branch.",
        },
    )
    log(f"Wrote {path} ({len(cases)} cases, Theta target={target:.3f} PV)")
    return path


def plan_branch(*, branch: str, rate: float, from_branch: str) -> Path:
    """New branch: reuse dP from an existing matched-Theta manifest, set rate."""
    base = load_matched_theta_cases(from_branch)
    cases: list[MatchedThetaCase] = []
    for i, bc in enumerate(base):
        cases.append(
            MatchedThetaCase(
                index=i,
                branch=branch,
                geom_seed=bc.geom_seed,
                realization_index=bc.realization_index,
                candidate_id=bc.candidate_id,
                phi=bc.phi,
                l_r_median_um=bc.l_r_median_um,
                dP_Pa=bc.dP_Pa,
                rate_mol_cm2_s=rate,
                theta_target_48h=bc.theta_target_48h,
                pore_solid_npy=bc.pore_solid_npy,
                mineral_h5=bc.mineral_h5,
            )
        )
    path = write_matched_theta_manifest(
        branch,
        cases,
        extra={
            "source": "run_matched_theta_campaign.plan_branch",
            "from_branch": from_branch,
            "rate_mol_cm2_s": rate,
            "note": "Reuses per-geom dP from from_branch; Theta match is flow-only.",
        },
    )
    log(f"Wrote {path} ({len(cases)} cases from branch {from_branch}, rate={rate:.0e})")
    return path


def summarize(branch: str) -> Path:
    cases = load_matched_theta_cases(branch)
    rows = []
    for case in cases:
        row = {
            "branch": branch,
            "case_index": case.index,
            "geom_seed": case.geom_seed,
            "candidate_id": case.candidate_id,
            "phi": case.phi,
            "l_r_median_um": case.l_r_median_um,
            "dP_Pa": case.dP_Pa,
            "rate_mol_cm2_s": case.rate_mol_cm2_s,
            "theta_target_48h": case.theta_target_48h,
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
                series = chrysocolla_inventory_series(h5, time_entries, lx=LX, ly=LY, lz=LZ)
                row["recovery_pct"] = float(series["total_recovery_pct"][-1])
                row["theta_48h"] = read_theta_48h(h5_path)
        rows.append(row)
    csv_path = MATCHED_THETA_ROOT / f"results_{branch}.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")
    write_matched_theta_manifest(branch, cases)
    log(f"Saved {csv_path} ({sum(r['complete'] for r in rows)}/{len(rows)} complete)")
    return csv_path


def cmd_info(branch: str) -> None:
    cases = load_matched_theta_cases(branch)
    complete = sum(1 for c in cases if c.has_output())
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Branch: {branch}  cases: {len(cases)}  complete: {complete}  pending: {len(cases) - complete}")
    print(f"Manifest: {matched_theta_manifest_path(branch)}")
    print(f"SLURM --array=0-{len(cases) - 1}")
    for c in cases[:5]:
        flag = "ok" if c.has_output() else "pending"
        print(
            f"  [{c.index:03d}] geom={c.geom_seed} dP={c.dP_Pa:g} "
            f"rate={c.rate_mol_cm2_s:.0e} [{flag}]"
        )
    if len(cases) > 5:
        print("  ...")


def cmd_pending(branch: str) -> None:
    for c in load_matched_theta_cases(branch):
        if not c.has_output():
            print(c.index)


def preflight(branch: str) -> None:
    cases = load_matched_theta_cases(branch)
    pending = [c for c in cases if not c.has_output()]
    errors: list[str] = []
    for case in pending:
        label = (
            f"[{case.index:03d}] geom{case.geom_seed} branch={branch} "
            f"dP={case.dP_Pa:g} rate={case.rate_mol_cm2_s:.0e}"
        )
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
    print(f"Branch: {branch}")
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


def smoke_case(branch: str, index: int, *, final_time_s: float = 60.0) -> None:
    cases = load_matched_theta_cases(branch)
    case = cases[index]
    ref_npz = ensure_k_pore(case)
    k_pore = load_k_pore(ref_npz)
    workdir = SMOKE_ROOT / f"{branch}_idx{index}"
    if workdir.exists():
        import shutil

        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    log(f"Smoke branch={branch} index={index} -> {workdir}")
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
    parser = argparse.ArgumentParser(description="Matched-Theta campaign on accessibility_ensemble/")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan_anchor = sub.add_parser("plan-anchor", help="Plan branch from factorial anchor thetas (dP=1, k_r=1e-9).")
    p_plan_anchor.add_argument("--branch", required=True)
    p_plan_anchor.add_argument("--rate", type=float, required=True)
    p_plan_anchor.add_argument("--theta-target", type=float, default=None)

    p_plan_branch = sub.add_parser("plan-branch", help="Plan branch by cloning dP from an existing branch.")
    p_plan_branch.add_argument("--branch", required=True)
    p_plan_branch.add_argument("--rate", type=float, required=True)
    p_plan_branch.add_argument("--from-branch", required=True)

    for name in ("info", "pending", "preflight", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--branch", required=True)

    p_smoke = sub.add_parser("smoke", help="60s PFLOTRAN in _smoke_matched_theta/ only.")
    p_smoke.add_argument("--branch", required=True)
    p_smoke.add_argument("--index", type=int, required=True)
    p_smoke.add_argument("--final-time-s", type=float, default=60.0)

    p_case = sub.add_parser("case")
    p_case.add_argument("--branch", required=True)
    p_case.add_argument("--index", type=int, required=True)
    p_case.add_argument("--skip-plots", action="store_true")
    p_case.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.command == "plan-anchor":
        plan_anchor(branch=args.branch, rate=args.rate, theta_target=args.theta_target)
        return
    if args.command == "plan-branch":
        plan_branch(branch=args.branch, rate=args.rate, from_branch=args.from_branch)
        return
    if args.command == "info":
        cmd_info(args.branch)
        return
    if args.command == "pending":
        cmd_pending(args.branch)
        return
    if args.command == "preflight":
        preflight(args.branch)
        return
    if args.command == "smoke":
        smoke_case(args.branch, args.index, final_time_s=args.final_time_s)
        return
    if args.command == "summarize":
        summarize(args.branch)
        return

    cases = load_matched_theta_cases(args.branch)
    case = cases[args.index]
    if case.has_output() and not args.force:
        log(f"Already complete: {case.case_dir()}")
        return
    run_case(case, resolve_pflotran_exe(), skip_plots=args.skip_plots)
    write_matched_theta_manifest(args.branch, cases)
    log("Done")


if __name__ == "__main__":
    main()