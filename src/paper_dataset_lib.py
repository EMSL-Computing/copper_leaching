"""Paper dataset layout under accessibility_ensemble/ (repo root, outside Chip_Designs).

Example:
    python migrate_paper_dataset.py
    python run_factorial_campaign.py info
    python run_matched_theta_campaign.py plan-branch --branch 4c --rate 5e-10 --from-branch 4a
    python run_matched_theta_campaign.py case --branch 4c --index 0
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

from _repo import REPO_ROOT

# Historical tree (archived). Paper campaigns use DATASET_ROOT only.
LEGACY_NETWORK = REPO_ROOT / "archive" / "Chip_Designs" / "bean_bucket_network"
DATASET_ROOT = REPO_ROOT / "accessibility_ensemble"
HERE = REPO_ROOT  # legacy alias: paths are relative to monorepo root

DP_VALUES = (0.001, 0.1, 1.0)
RATE_VALUES = (1e-10, 5e-10, 1e-9)

REALIZATIONS_MANIFEST = DATASET_ROOT / "ensemble" / "realizations" / "manifest.json"
SCREENING_DIR = DATASET_ROOT / "ensemble" / "screening"
K_PORE_ROOT = DATASET_ROOT / "ensemble" / "shared" / "k_pore"
FACTORIAL_ROOT = DATASET_ROOT / "campaigns" / "factorial_dp_rate"
FACTORIAL_MANIFEST = FACTORIAL_ROOT / "manifest.json"
MATCHED_THETA_ROOT = DATASET_ROOT / "campaigns" / "matched_theta"
MATCHED_THETA_DP_CLAMP = (0.05, 5.0)
MATCHED_THETA_ANCHOR_DP = 1.0
MATCHED_THETA_ANCHOR_RATE = 1e-9
PILOT_ROOT = DATASET_ROOT / "campaigns" / "pilot_dp_rate"
ANALYSIS_ROOT = DATASET_ROOT / "analysis"
ANALYSIS_TABLES = ANALYSIS_ROOT / "tables"
ANALYSIS_SHARED = ANALYSIS_ROOT / "shared"
ANALYSIS_SOLID_PHASE = ANALYSIS_ROOT / "solid_phase"
ANALYSIS_OUTLET = ANALYSIS_ROOT / "outlet"
ANALYSIS_COMBINED = ANALYSIS_ROOT / "combined"
RECOVERY_METRICS = ("solid_phase", "outlet")

CASE_FILES = (
    "mineral_dissolution.h5",
    "mineral_dissolution.in",
    "geometry_accessibility.h5",
    "mineral_dissolution-obs-0.pft",
)
CASE_DIRS = ("figs",)

FINAL_TIME_S = 3600.0 * 24.0 * 10.0
MIN_COMPLETE_H5_BYTES = 50_000_000


def case_h5_complete(h5_path: Path, *, final_time_s: float = FINAL_TIME_S) -> bool:
    """True only for finished PFLOTRAN runs (guards against partial/crashed outputs)."""
    if not h5_path.is_file():
        return False
    if h5_path.stat().st_size < MIN_COMPLETE_H5_BYTES:
        return False
    try:
        import h5py

        from plot_recovery_metrics import parse_time_groups

        with h5py.File(h5_path, "r") as h5:
            times = parse_time_groups(h5)
        if len(times) < 2:
            return False
        return float(times[-1][0]) >= 0.95 * float(final_time_s)
    except (OSError, KeyError, ValueError):
        return False

LEGACY_STAGE1_MANIFEST = LEGACY_NETWORK / "stage1_realizations" / "stage1_manifest.json"
LEGACY_CHRYSO_DIR = LEGACY_NETWORK / "chrysocolla_realizations"
LEGACY_SWEEPS = LEGACY_NETWORK / "sweeps"


def tag_dp(dp: float) -> str:
    return str(dp).replace("-", "neg_").replace(".", "p")


def tag_rate(rate: float) -> str:
    return f"{rate:.0e}".replace("+", "").replace("-", "m")


def realization_dir(geom_seed: int | str) -> Path:
    return DATASET_ROOT / "ensemble" / "realizations" / f"geom{int(geom_seed)}"


def k_pore_dir(geom_seed: int | str) -> Path:
    return K_PORE_ROOT / f"geom{int(geom_seed)}"


def k_pore_npz(geom_seed: int | str) -> Path:
    """Preferred k_pore file (flat migrated layout)."""
    return k_pore_dir(geom_seed) / "stokes_reference.npz"


def k_pore_npz_calibrated(geom_seed: int | str) -> Path:
    """Path written by simulate_darcy calibration."""
    return k_pore_dir(geom_seed) / "calibrate_k_pore_run" / "stokes_reference.npz"


def k_pore_npz_candidates(geom_seed: int | str) -> tuple[Path, ...]:
    return (k_pore_npz(geom_seed), k_pore_npz_calibrated(geom_seed))


def resolve_k_pore_npz(geom_seed: int | str) -> Path | None:
    from run_dp_rate_pilot_sweep import reference_is_valid

    for candidate in k_pore_npz_candidates(geom_seed):
        if reference_is_valid(candidate):
            return candidate
    return None


def load_realization_entries() -> list[dict]:
    payload = json.loads(REALIZATIONS_MANIFEST.read_text())
    return payload["entries"]


def factorial_case_dir(geom_seed: int | str, dp: float, rate: float) -> Path:
    return (
        FACTORIAL_ROOT
        / "cases"
        / f"geom{int(geom_seed)}"
        / f"dp{tag_dp(dp)}"
        / f"rate{tag_rate(rate)}"
    )


def matched_theta_manifest_path(branch: str) -> Path:
    return MATCHED_THETA_ROOT / f"manifest_{branch}.json"


def matched_theta_case_dir(geom_seed: int | str, branch: str) -> Path:
    branch = str(branch).strip()
    if not branch:
        raise ValueError("branch must be non-empty (e.g. 4a, 4b, 4c)")
    return MATCHED_THETA_ROOT / "cases" / f"geom{int(geom_seed)}" / branch


def clamp_matched_theta_dp(dP: float) -> float:
    lo, hi = MATCHED_THETA_DP_CLAMP
    return float(max(lo, min(hi, float(dP))))


def pilot_case_dir(geom_seed: int | str, dp: float, rate: float) -> Path:
    return (
        PILOT_ROOT
        / "cases"
        / f"geom{int(geom_seed)}"
        / f"dp{tag_dp(dp)}"
        / f"rate{tag_rate(rate)}"
    )


def copy_case_files(src_dir: Path, dst_dir: Path, *, force: bool = False) -> list[str]:
    """Copy paper-relevant PFLOTRAN outputs from src_dir to dst_dir."""
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in CASE_FILES:
        src = src_dir / name
        if not src.is_file():
            continue
        dst = dst_dir / name
        if dst.is_file() and not force and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)
        copied.append(name)
    for name in CASE_DIRS:
        src = src_dir / name
        if not src.is_dir():
            continue
        dst = dst_dir / name
        if dst.exists() and not force:
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        copied.append(f"{name}/")
    return copied


@dataclass(frozen=True)
class FactorialCase:
    index: int
    geom_seed: int
    realization_index: int
    candidate_id: str
    phi: float
    l_r_median_um: float
    dP_Pa: float
    rate_mol_cm2_s: float
    pore_solid_npy: Path
    mineral_h5: Path

    def case_dir(self) -> Path:
        return factorial_case_dir(self.geom_seed, self.dP_Pa, self.rate_mol_cm2_s)

    def has_output(self) -> bool:
        return case_h5_complete(self.case_dir() / "mineral_dissolution.h5")


@dataclass(frozen=True)
class MatchedThetaCase:
    index: int
    branch: str
    geom_seed: int
    realization_index: int
    candidate_id: str
    phi: float
    l_r_median_um: float
    dP_Pa: float
    rate_mol_cm2_s: float
    theta_target_48h: float
    pore_solid_npy: Path
    mineral_h5: Path

    def case_dir(self) -> Path:
        return matched_theta_case_dir(self.geom_seed, self.branch)

    def has_output(self) -> bool:
        return case_h5_complete(self.case_dir() / "mineral_dissolution.h5")


def _matched_theta_case_from_entry(entry: dict, *, branch: str, fallback_index: int = 0) -> MatchedThetaCase:
    geom_seed = int(entry["geom_seed"])
    rdir = realization_dir(geom_seed)
    case_index = entry.get("case_index", entry.get("stage4_index", fallback_index))
    return MatchedThetaCase(
        index=int(case_index),
        branch=branch,
        geom_seed=geom_seed,
        realization_index=int(entry["realization_index"]),
        candidate_id=str(entry["candidate_id"]),
        phi=float(entry["phi"]),
        l_r_median_um=float(entry["l_r_median_um"]),
        dP_Pa=float(entry["dP_Pa"]),
        rate_mol_cm2_s=float(entry["rate_mol_cm2_s"]),
        theta_target_48h=float(entry.get("theta_target_48h", float("nan"))),
        pore_solid_npy=rdir / "pore_solid.npy",
        mineral_h5=rdir / "mineral.h5",
    )


def write_matched_theta_manifest(branch: str, cases: list[MatchedThetaCase], *, extra: dict | None = None) -> Path:
    MATCHED_THETA_ROOT.mkdir(parents=True, exist_ok=True)
    path = matched_theta_manifest_path(branch)
    payload: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "matched_theta",
        "branch": branch,
        "n_cases": len(cases),
        "entries": [
            {
                "case_index": c.index,
                "geom_seed": c.geom_seed,
                "realization_index": c.realization_index,
                "candidate_id": c.candidate_id,
                "phi": c.phi,
                "l_r_median_um": c.l_r_median_um,
                "dP_Pa": c.dP_Pa,
                "rate_mol_cm2_s": c.rate_mol_cm2_s,
                "theta_target_48h": c.theta_target_48h,
                "case_dir": str(c.case_dir()),
                "complete": c.has_output(),
            }
            for c in cases
        ],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_matched_theta_cases(branch: str) -> list[MatchedThetaCase]:
    path = matched_theta_manifest_path(branch)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python run_matched_theta_campaign.py plan-branch --branch {branch} ..."
        )
    payload = json.loads(path.read_text())
    cases = [
        _matched_theta_case_from_entry(entry, branch=branch, fallback_index=i)
        for i, entry in enumerate(payload["entries"])
    ]
    return cases


def load_legacy_stage1_entries() -> list[dict]:
    payload = json.loads(LEGACY_STAGE1_MANIFEST.read_text())
    return payload["entries"]


def build_realizations_manifest(entries: list[dict]) -> dict:
    manifest_entries = []
    for entry in entries:
        geom_seed = int(entry["geom_seed"])
        rdir = realization_dir(geom_seed)
        manifest_entries.append(
            {
                "geom_seed": geom_seed,
                "realization_index": int(entry["realization_index"]),
                "candidate_id": entry["candidate_id"],
                "phi": float(entry["phi"]),
                "l_r_median_um": float(entry["l_r_median_um"]),
                "mineral_seed": int(entry.get("mineral_seed", 42)),
                "pore_solid_npy": str(rdir / "pore_solid.npy"),
                "mineral_h5": str(rdir / "mineral.h5"),
            }
        )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_realizations": len(manifest_entries),
        "dx_m": 2e-5,
        "chrysocolla_solid_fraction": 0.5,
        "ssa_cm2_cm3": 100.0,
        "entries": manifest_entries,
    }


def plan_factorial_cases() -> list[FactorialCase]:
    entries = load_legacy_stage1_entries()
    cases: list[FactorialCase] = []
    idx = 0
    for entry in entries:
        geom_seed = int(entry["geom_seed"])
        rdir = realization_dir(geom_seed)
        for dp in DP_VALUES:
            for rate in RATE_VALUES:
                cases.append(
                    FactorialCase(
                        index=idx,
                        geom_seed=geom_seed,
                        realization_index=int(entry["realization_index"]),
                        candidate_id=str(entry["candidate_id"]),
                        phi=float(entry["phi"]),
                        l_r_median_um=float(entry["l_r_median_um"]),
                        dP_Pa=float(dp),
                        rate_mol_cm2_s=float(rate),
                        pore_solid_npy=rdir / "pore_solid.npy",
                        mineral_h5=rdir / "mineral.h5",
                    )
                )
                idx += 1
    return cases


def write_factorial_manifest(cases: list[FactorialCase]) -> Path:
    FACTORIAL_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "factorial_dp_rate",
        "dP_Pa_values": list(DP_VALUES),
        "rate_mol_cm2_s_values": list(RATE_VALUES),
        "n_cases": len(cases),
        "entries": [
            {
                "case_index": c.index,
                "geom_seed": c.geom_seed,
                "realization_index": c.realization_index,
                "candidate_id": c.candidate_id,
                "phi": c.phi,
                "l_r_median_um": c.l_r_median_um,
                "dP_Pa": c.dP_Pa,
                "rate_mol_cm2_s": c.rate_mol_cm2_s,
                "case_dir": str(c.case_dir()),
                "complete": c.has_output(),
            }
            for c in cases
        ],
    }
    FACTORIAL_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    return FACTORIAL_MANIFEST


def load_factorial_cases() -> list[FactorialCase]:
    if not FACTORIAL_MANIFEST.is_file():
        cases = plan_factorial_cases()
        write_factorial_manifest(cases)
    payload = json.loads(FACTORIAL_MANIFEST.read_text())
    cases = []
    for entry in payload["entries"]:
        geom_seed = int(entry["geom_seed"])
        rdir = realization_dir(geom_seed)
        cases.append(
            FactorialCase(
                index=int(entry["case_index"]),
                geom_seed=geom_seed,
                realization_index=int(entry["realization_index"]),
                candidate_id=str(entry["candidate_id"]),
                phi=float(entry["phi"]),
                l_r_median_um=float(entry["l_r_median_um"]),
                dP_Pa=float(entry["dP_Pa"]),
                rate_mol_cm2_s=float(entry["rate_mol_cm2_s"]),
                pore_solid_npy=rdir / "pore_solid.npy",
                mineral_h5=rdir / "mineral.h5",
            )
        )
    return cases