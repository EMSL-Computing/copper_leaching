"""Ensemble transport stages 1-6 + 4b (PFLOTRAN) and scaling analysis.

Prereq: screen_l_r_ensemble.py stage1-screen && stage1-select --n 25

Example:
    python run_ensemble_transport.py stage1 info
    python run_ensemble_transport.py stage1 case --index 0 --skip-plots
    python run_ensemble_transport.py stage3 plan
    python run_ensemble_transport.py plan-overnight
    python run_ensemble_transport.py stage4a_theta_match case --index 0 --skip-plots
    python run_ensemble_transport.py stage4b_theta_match_slow plan
    python run_ensemble_transport.py analyze
    python run_ensemble_transport.py analyze --figures-only

SLURM: run_ensemble_stage{1,2,3,4a,4b,5,6}.slurm  or  bash submit_overnight.sh
"""

from __future__ import annotations

import argparse
import sys

import _repo  # noqa: F401 — path bootstrap

from ensemble_transport_lib import (
    STAGE_SPECS,
    analyze_stages,
    load_stage_cases,
    log,
    plan_overnight_campaign,
    plan_stage4a_theta_match,
    plan_stage4b_theta_match_slow,
    plan_stage5_theta_swap,
    plan_stage6_rate_mid,
    run_one_case,
    summarize_stage,
    write_stage3_manifest,
)
from run_dp_rate_pilot_sweep import resolve_pflotran_exe

STAGES = (
    "stage1",
    "stage2",
    "stage3",
    "stage4a_theta_match",
    "stage4b_theta_match_slow",
    "stage5_theta_swap",
    "stage6_rate_mid",
)
PLAN_STAGES = {
    "stage3": write_stage3_manifest,
    "stage4a_theta_match": plan_stage4a_theta_match,
    "stage4b_theta_match_slow": plan_stage4b_theta_match_slow,
    "stage5_theta_swap": plan_stage5_theta_swap,
    "stage6_rate_mid": plan_stage6_rate_mid,
}


def cmd_info(stage: str) -> None:
    cases = load_stage_cases(stage)
    spec = STAGE_SPECS[stage]
    print(f"Stage: {stage}")
    print(f"Manifest: {spec.manifest_path}")
    print(f"Cases: {len(cases)}")
    print(f"Default dP={spec.dP_Pa:g} Pa  rate={spec.rate_mol_cm2_s:.0e} (per-case may vary)")
    print(f"Out: {spec.out_root}")
    print(f"SLURM --array=0-{len(cases) - 1}")
    for c in cases:
        print(
            f"  [{c.index:02d}] geom={c.geom_seed} l_r={c.l_r_median_um:.1f}um "
            f"dP={c.dP_Pa:g} rate={c.rate_mol_cm2_s:.0e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble transport stages 1-6 + 4b + analyze.")
    sub = parser.add_subparsers(dest="stage_cmd", required=True)

    for stage in STAGES:
        p = sub.add_parser(stage, help=f"Run {stage} PFLOTRAN ensemble.")
        p_sub = p.add_subparsers(dest="command", required=True)
        p_sub.add_parser("info", help="Print case count and SLURM array range.")
        p_case = p_sub.add_parser("case", help="Run one array index.")
        p_case.add_argument("--index", type=int, required=True)
        p_case.add_argument("--skip-plots", action="store_true", help="Skip postprocess (for SLURM).")
        p_sub.add_parser("summarize", help=f"Build {stage}_results.csv.")
        if stage in PLAN_STAGES:
            p_sub.add_parser("plan", help=f"Build manifest for {stage}.")

    sub.add_parser("plan-overnight", help="Build stage4a/4b/5/6 manifests for overnight campaign.")
    p_analyze = sub.add_parser("analyze", help="Combine finished stage CSVs into scaling figures.")
    p_analyze.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip anchor postprocess (faster; scaling PNGs/CSVs only).",
    )

    args = parser.parse_args()

    if args.stage_cmd == "analyze":
        analyze_stages(figures_only=args.figures_only)
        log("Done (analyze)")
        return

    if args.stage_cmd == "plan-overnight":
        plan_overnight_campaign()
        for stage in (
            "stage4a_theta_match",
            "stage4b_theta_match_slow",
            "stage5_theta_swap",
            "stage6_rate_mid",
        ):
            cmd_info(stage)
        return

    stage = args.stage_cmd
    spec = STAGE_SPECS[stage]

    if args.command == "info":
        cmd_info(stage)
        return

    if args.command == "plan":
        if stage not in PLAN_STAGES:
            raise SystemExit(f"{stage} has no plan command")
        PLAN_STAGES[stage]()
        cmd_info(stage)
        return

    cases = load_stage_cases(stage)
    spec.out_root.mkdir(parents=True, exist_ok=True)
    pflotran_exe = resolve_pflotran_exe()

    if args.command == "case":
        if args.index < 0 or args.index >= len(cases):
            raise SystemExit(f"index must be in [0, {len(cases) - 1}]")
        run_one_case(cases[args.index], spec.out_root, pflotran_exe, skip_plots=args.skip_plots)
    elif args.command == "summarize":
        summarize_stage(stage)

    log(f"Done ({stage} {args.command})")


if __name__ == "__main__":
    main()