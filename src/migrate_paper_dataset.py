"""One-time copy/rename of paper dataset into accessibility_ensemble/.

Example:
    python migrate_paper_dataset.py
    python migrate_paper_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

from paper_dataset_lib import (
    ANALYSIS_ROOT,
    CASE_DIRS,
    CASE_FILES,
    DATASET_ROOT,
    DP_VALUES,
    FACTORIAL_ROOT,
    K_PORE_ROOT,
    LEGACY_CHRYSO_DIR,
    LEGACY_NETWORK,
    LEGACY_SWEEPS,
    MATCHED_THETA_ROOT,
    PILOT_ROOT,
    REALIZATIONS_MANIFEST,
    RATE_VALUES,
    SCREENING_DIR,
    build_realizations_manifest,
    copy_case_files,
    factorial_case_dir,
    k_pore_npz,
    load_legacy_stage1_entries,
    matched_theta_case_dir,
    pilot_case_dir,
    plan_factorial_cases,
    realization_dir,
    tag_dp,
    tag_rate,
    write_factorial_manifest,
)

LEGACY_STAGE4_MANIFEST = LEGACY_NETWORK / "stage4_theta_match_realizations" / "stage4_manifest.json"
LEGACY_STAGE4B_MANIFEST = LEGACY_NETWORK / "stage4b_theta_match_slow_realizations" / "stage4b_manifest.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def copy_file(src: Path, dst: Path, *, dry_run: bool) -> bool:
    if not src.is_file():
        return False
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return False
    if dry_run:
        log(f"  [dry-run] copy {src} -> {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def migrate_realizations(*, dry_run: bool) -> int:
    entries = load_legacy_stage1_entries()
    n = 0
    for entry in entries:
        geom_seed = int(entry["geom_seed"])
        rdir = realization_dir(geom_seed)
        src_h5 = Path(entry["h5_path"])
        src_npy = Path(entry["geom_npy"])
        dst_h5 = rdir / "mineral.h5"
        dst_npy = rdir / "pore_solid.npy"
        if dry_run:
            log(f"[realizations] geom{geom_seed}: mineral.h5, pore_solid.npy")
            n += 2
            continue
        rdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_h5, dst_h5)
        shutil.copy2(src_npy, dst_npy)
        n += 2
    manifest = build_realizations_manifest(entries)
    if dry_run:
        log(f"[realizations] manifest.json ({len(entries)} entries)")
    else:
        REALIZATIONS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        REALIZATIONS_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return n


def migrate_k_pore(*, dry_run: bool) -> int:
    entries = load_legacy_stage1_entries()
    n = 0
    for entry in entries:
        geom_seed = int(entry["geom_seed"])
        idx = int(entry["realization_index"])
        src = (
            LEGACY_SWEEPS
            / "ensemble_stage1"
            / f"real{idx:03d}_geom{geom_seed}"
            / "reference"
            / "calibrate_k_pore_run"
            / "stokes_reference.npz"
        )
        dst = k_pore_npz(geom_seed)
        if copy_file(src, dst, dry_run=dry_run):
            n += 1
        ref_log = src.parent.parent / "reference.log"
        dst_log = dst.parent / "reference.log"
        if copy_file(ref_log, dst_log, dry_run=dry_run):
            n += 1
    return n


def ingest_factorial_from_src(
    src_dir: Path,
    geom_seed: int,
    dp: float,
    rate: float,
    *,
    dry_run: bool,
    label: str,
) -> bool:
    dst_dir = factorial_case_dir(geom_seed, dp, rate)
    h5 = src_dir / "mineral_dissolution.h5"
    if not h5.is_file():
        return False
    if dry_run:
        log(f"[factorial] {label} -> {dst_dir.relative_to(DATASET_ROOT)}")
        return True
    copied = copy_case_files(src_dir, dst_dir)
    log(f"[factorial] {label} -> {dst_dir.relative_to(DATASET_ROOT)} ({', '.join(copied) or 'skip'})")
    return True


def migrate_factorial_legacy(*, dry_run: bool) -> dict[str, int]:
    stats = {"stage1": 0, "stage2": 0, "stage3": 0, "stage6": 0, "pilot": 0}
    entries = load_legacy_stage1_entries()
    by_geom_idx = {int(e["geom_seed"]): int(e["realization_index"]) for e in entries}

    for entry in entries:
        geom_seed = int(entry["geom_seed"])
        idx = int(entry["realization_index"])
        src = LEGACY_SWEEPS / "ensemble_stage1" / f"real{idx:03d}_geom{geom_seed}"
        if ingest_factorial_from_src(src, geom_seed, 1.0, 1e-9, dry_run=dry_run, label="stage1"):
            stats["stage1"] += 1
        src2 = LEGACY_SWEEPS / "ensemble_stage2" / f"real{idx:03d}_geom{geom_seed}"
        if ingest_factorial_from_src(src2, geom_seed, 0.1, 1e-9, dry_run=dry_run, label="stage2"):
            stats["stage2"] += 1
        src6 = (
            LEGACY_SWEEPS
            / "ensemble_stage6_rate_mid"
            / f"real{idx:03d}_geom{geom_seed}_dp{tag_dp(1.0)}_rate{tag_rate(5e-10)}"
        )
        if ingest_factorial_from_src(src6, geom_seed, 1.0, 5e-10, dry_run=dry_run, label="stage6"):
            stats["stage6"] += 1

    stage3_manifest = LEGACY_NETWORK / "stage3_realizations" / "stage3_manifest.json"
    if stage3_manifest.is_file():
        for entry in json.loads(stage3_manifest.read_text())["entries"]:
            geom_seed = int(entry["geom_seed"])
            dp = float(entry["dP_Pa"])
            rate = float(entry["rate_mol_cm2_s"])
            idx = int(entry["stage3_index"])
            src = (
                LEGACY_SWEEPS
                / "ensemble_stage3"
                / f"real{idx:03d}_geom{geom_seed}_dp{tag_dp(dp)}_rate{tag_rate(rate)}"
            )
            if ingest_factorial_from_src(src, geom_seed, dp, rate, dry_run=dry_run, label="stage3"):
                stats["stage3"] += 1

    pilot_root = LEGACY_SWEEPS / "dp_rate_pilot" / "chrysocolla_real000"
    pilot_geom = 437
    if pilot_geom in by_geom_idx:
        for dp in DP_VALUES:
            for rate in RATE_VALUES:
                src = pilot_root / f"dp_{tag_dp(dp)}_rate_{tag_rate(rate)}"
                dst_h5 = factorial_case_dir(pilot_geom, dp, rate) / "mineral_dissolution.h5"
                if dst_h5.is_file():
                    continue
                if ingest_factorial_from_src(
                    src, pilot_geom, dp, rate, dry_run=dry_run, label="pilot"
                ):
                    stats["pilot"] += 1
    return stats


def migrate_matched_theta(*, dry_run: bool) -> dict[str, int]:
    stats = {"4a": 0, "4b": 0}
    for branch, manifest_path, sweep_dir in (
        ("4a", LEGACY_STAGE4_MANIFEST, LEGACY_SWEEPS / "ensemble_stage4_theta_match"),
        ("4b", LEGACY_STAGE4B_MANIFEST, LEGACY_SWEEPS / "ensemble_stage4b_theta_match_slow"),
    ):
        if not manifest_path.is_file():
            log(f"[matched_theta] skip {branch}: missing {manifest_path}")
            continue
        payload = json.loads(manifest_path.read_text())
        dst_manifest = MATCHED_THETA_ROOT / f"manifest_{branch}.json"
        if dry_run:
            log(f"[matched_theta] write {dst_manifest.name}")
        else:
            MATCHED_THETA_ROOT.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_path, dst_manifest)

        for i, entry in enumerate(payload["entries"]):
            geom_seed = int(entry["geom_seed"])
            dp = float(entry["dP_Pa"])
            rate = float(entry["rate_mol_cm2_s"])
            src = sweep_dir / f"real{i:03d}_geom{geom_seed}_dp{tag_dp(dp)}_rate{tag_rate(rate)}"
            dst_dir = matched_theta_case_dir(geom_seed, branch)
            if not (src / "mineral_dissolution.h5").is_file():
                log(f"[matched_theta] missing {branch} geom{geom_seed}: {src}")
                continue
            if dry_run:
                log(f"[matched_theta] {branch} geom{geom_seed} <- {src.name}")
                stats[branch] += 1
                continue
            copy_case_files(src, dst_dir)
            log(f"[matched_theta] {branch} geom{geom_seed} -> {dst_dir.relative_to(DATASET_ROOT)}")
            stats[branch] += 1
    return stats


def migrate_pilot_campaign(*, dry_run: bool) -> int:
    """Copy single-geom pilot into campaigns/pilot_dp_rate (geom437)."""
    pilot_geom = 437
    src_root = LEGACY_SWEEPS / "dp_rate_pilot" / "chrysocolla_real000"
    n = 0
    for dp in DP_VALUES:
        for rate in RATE_VALUES:
            src = src_root / f"dp_{tag_dp(dp)}_rate_{tag_rate(rate)}"
            dst_dir = pilot_case_dir(pilot_geom, dp, rate)
            if not (src / "mineral_dissolution.h5").is_file():
                continue
            if dry_run:
                log(f"[pilot] geom{pilot_geom} dp={dp} rate={rate:.0e}")
                n += 1
                continue
            copy_case_files(src, dst_dir)
            n += 1
    ref_src = src_root / "reference"
    ref_dst = PILOT_ROOT / "cases" / f"geom{pilot_geom}" / "reference"
    if ref_src.is_dir() and not dry_run:
        if ref_dst.exists():
            shutil.rmtree(ref_dst)
        shutil.copytree(ref_src, ref_dst)
    return n


def migrate_screening(*, dry_run: bool) -> int:
    src_root = LEGACY_NETWORK / "ensemble_screening"
    files = [
        (src_root / "stage1_selected.json", SCREENING_DIR / "stage1_selected.json"),
        (
            src_root / "stage1_scale_geometry" / "screening_results.csv",
            SCREENING_DIR / "screening_results.csv",
        ),
        (
            src_root / "stage1_scale_geometry" / "l_r_screening.png",
            SCREENING_DIR / "l_r_screening.png",
        ),
    ]
    n = 0
    for src, dst in files:
        if copy_file(src, dst, dry_run=dry_run):
            n += 1
    return n


def migrate_analysis(*, dry_run: bool) -> int:
    src_root = LEGACY_SWEEPS / "scaling_analysis"
    if not src_root.is_dir():
        return 0
    n = 0
    for src in src_root.iterdir():
        if not src.is_file():
            continue
        if src.suffix.lower() not in {".png", ".csv", ".txt"}:
            continue
        dst = ANALYSIS_ROOT / src.name
        if copy_file(src, dst, dry_run=dry_run):
            n += 1
    return n


def write_ingest_report(
    factorial_stats: dict[str, int],
    matched_stats: dict[str, int],
    *,
    dry_run: bool,
) -> Path:
    cases = plan_factorial_cases()
    if not dry_run:
        write_factorial_manifest(cases)
    complete = sum(1 for c in cases if c.has_output())
    pending = len(cases) - complete
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(DATASET_ROOT),
        "dry_run": dry_run,
        "factorial": {
            "n_total": len(cases),
            "n_complete": complete,
            "n_pending": pending,
            "ingest_sources": factorial_stats,
        },
        "matched_theta": matched_stats,
        "pending_by_dp_rate": {},
    }
    for dp in DP_VALUES:
        for rate in RATE_VALUES:
            sub = [c for c in cases if c.dP_Pa == dp and c.rate_mol_cm2_s == rate]
            key = f"dp{tag_dp(dp)}_rate{tag_rate(rate)}"
            report["pending_by_dp_rate"][key] = {
                "complete": sum(1 for c in sub if c.has_output()),
                "pending": sum(1 for c in sub if not c.has_output()),
            }
    out = DATASET_ROOT / "ingest_report.json"
    if not dry_run:
        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"Factorial complete: {complete}/{len(cases)}  pending: {pending}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate paper dataset to accessibility_ensemble/")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only.")
    args = parser.parse_args()

    log(f"Dataset root: {DATASET_ROOT}")
    if not args.dry_run:
        for path in (
            DATASET_ROOT,
            FACTORIAL_ROOT / "cases",
            MATCHED_THETA_ROOT / "cases",
            PILOT_ROOT / "cases",
            K_PORE_ROOT,
            SCREENING_DIR,
            ANALYSIS_ROOT,
        ):
            path.mkdir(parents=True, exist_ok=True)

    log("=== realizations ===")
    migrate_realizations(dry_run=args.dry_run)
    log("=== k_pore ===")
    migrate_k_pore(dry_run=args.dry_run)
    log("=== factorial ingest ===")
    fstats = migrate_factorial_legacy(dry_run=args.dry_run)
    log(f"  sources: {fstats}")
    log("=== matched theta ===")
    mstats = migrate_matched_theta(dry_run=args.dry_run)
    log(f"  {mstats}")
    log("=== pilot ===")
    migrate_pilot_campaign(dry_run=args.dry_run)
    log("=== screening ===")
    migrate_screening(dry_run=args.dry_run)
    log("=== analysis ===")
    migrate_analysis(dry_run=args.dry_run)
    write_ingest_report(fstats, mstats, dry_run=args.dry_run)
    log("Done.")


if __name__ == "__main__":
    main()