"""Shared helpers for ensemble transport stages 1-6, stage 4b, and scaling analysis.

Outputs include scaling_stages_overview.png, scaling_collapse.png,
scaling_sweep_efficiency.png, scaling_pe_da_regime.png (all stages + stage 4a/4b panel),
stage4_matched_theta_recovery_vs_l_r.png (matched-Theta 4a+4b with decay guides), and anchor case figures.

Example:
    python run_ensemble_transport.py analyze
    python run_ensemble_transport.py analyze --figures-only
"""

from __future__ import annotations

import csv
import fcntl
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_accessibility_mineral_dissolution import chrysocolla_inventory_series
from plot_figure_labels import LBL_CU_RECOVERY, LBL_LR_MEDIAN, LBL_THETA_48H
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
    run_cmd,
    tag_dp,
    tag_rate,
)
from compute_reactive_accessibility_length import connected_pore_from_inlet
from write_darcy_input_for_accessibility_transport import write_accessibility_transport_input

from _repo import REPO_ROOT, SRC_DIR

HERE = REPO_ROOT
NETWORK_DIR = REPO_ROOT / "archive" / "Chip_Designs" / "bean_bucket_network"
STAGE1_MANIFEST = NETWORK_DIR / "stage1_realizations" / "stage1_manifest.json"
STAGE3_MANIFEST = NETWORK_DIR / "stage3_realizations" / "stage3_manifest.json"
STAGE4_MANIFEST = NETWORK_DIR / "stage4_theta_match_realizations" / "stage4_manifest.json"
STAGE4B_MANIFEST = NETWORK_DIR / "stage4b_theta_match_slow_realizations" / "stage4b_manifest.json"
STAGE5_MANIFEST = NETWORK_DIR / "stage5_theta_swap_realizations" / "stage5_manifest.json"
STAGE6_MANIFEST = NETWORK_DIR / "stage6_rate_mid_realizations" / "stage6_manifest.json"
SWEEPS_ROOT = NETWORK_DIR / "sweeps"

STAGE3_N_GEOMS = 6
STAGE3_RATE = 1e-10
STAGE3_DPS = (1.0, 0.1)
STAGE4_DP_CLAMP = (0.05, 5.0)
STAGE5_SWAP_PAIRS = (
    (437, 423),
    (484, 422),
    (462, 507),
    (441, 509),
    (464, 417),
    (425, 424),
)
STAGE4B_RATE = 1e-10
STAGE6_RATE = 5e-10
STAGE1_SELECTED = NETWORK_DIR / "ensemble_screening" / "stage1_selected.json"
COLLAPSE_LAMBDA_M = 60.0e-6
PORE_DIFFUSIVITY_M2S = 1.0e-9
FLUID_VISCOSITY_PA_S = 1.0e-3
SSA_CHRYSOCOLLA_M2_M3 = 100.0 * 1.0e4
_STOKES_UX_REF_CACHE: dict[int, tuple[float, float]] = {}
_DARCY_UX_CACHE: dict[str, float] = {}
MANIFEST_STAGES = (
    "stage3",
    "stage4a_theta_match",
    "stage4b_theta_match_slow",
    "stage5_theta_swap",
    "stage6_rate_mid",
)
STAGE_LEGACY_ALIASES = {"stage4_theta_match": "stage4a_theta_match"}
STAGE_RESULTS_LEGACY = {"stage4a_theta_match": ("stage4_theta_match", "stage4a_theta_match")}


@dataclass(frozen=True)
class EnsembleCase:
    index: int
    geom_seed: int
    geom_npy: Path
    geom_h5: Path
    l_r_median_um: float
    phi: float
    dP_Pa: float
    rate_mol_cm2_s: float
    candidate_id: str
    stage: str

    def workdir(self, out_root: Path) -> Path:
        rate_tag = tag_rate(self.rate_mol_cm2_s)
        if self.stage in MANIFEST_STAGES:
            return (
                out_root
                / f"real{self.index:03d}_geom{self.geom_seed}_dp{tag_dp(self.dP_Pa)}_rate{rate_tag}"
            )
        return out_root / f"real{self.index:03d}_geom{self.geom_seed}"

    def reference_dir(self, out_root: Path) -> Path:
        return self.workdir(out_root) / "reference"


@dataclass(frozen=True)
class StageSpec:
    name: str
    dP_Pa: float
    rate_mol_cm2_s: float
    out_root: Path
    manifest_path: Path


STAGE_SPECS = {
    "stage1": StageSpec("stage1", 1.0, 1e-9, SWEEPS_ROOT / "ensemble_stage1", STAGE1_MANIFEST),
    "stage2": StageSpec("stage2", 0.1, 1e-9, SWEEPS_ROOT / "ensemble_stage2", STAGE1_MANIFEST),
    "stage3": StageSpec("stage3", 1.0, 1e-9, SWEEPS_ROOT / "ensemble_stage3", STAGE3_MANIFEST),
    "stage4a_theta_match": StageSpec(
        "stage4a_theta_match", 1.0, 1e-9, SWEEPS_ROOT / "ensemble_stage4_theta_match", STAGE4_MANIFEST
    ),
    "stage4b_theta_match_slow": StageSpec(
        "stage4b_theta_match_slow",
        1.0,
        STAGE4B_RATE,
        SWEEPS_ROOT / "ensemble_stage4b_theta_match_slow",
        STAGE4B_MANIFEST,
    ),
    "stage5_theta_swap": StageSpec(
        "stage5_theta_swap", 1.0, 1e-9, SWEEPS_ROOT / "ensemble_stage5_theta_swap", STAGE5_MANIFEST
    ),
    "stage6_rate_mid": StageSpec(
        "stage6_rate_mid", 1.0, STAGE6_RATE, SWEEPS_ROOT / "ensemble_stage6_rate_mid", STAGE6_MANIFEST
    ),
}


def log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def normalize_stage_name(stage: str) -> str:
    """Map legacy stage ids (e.g. stage4_theta_match) to current names."""
    return STAGE_LEGACY_ALIASES.get(stage, stage)


def stage_results_csv(stage: str) -> Path:
    """Resolve per-stage results CSV, including legacy basenames for stage 4a."""
    spec = STAGE_SPECS[normalize_stage_name(stage)]
    stage = spec.name
    candidates = STAGE_RESULTS_LEGACY.get(stage, (stage,))
    for basename in candidates:
        path = spec.out_root / f"{basename}_results.csv"
        if path.is_file():
            return path
    return spec.out_root / f"{stage}_results.csv"


def load_stage1_entries(path: Path = STAGE1_MANIFEST) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run:\n"
            "  python screen_l_r_ensemble.py stage1-screen\n"
            "  python screen_l_r_ensemble.py stage1-select --n 25"
        )
    return json.loads(path.read_text())["entries"]


def entry_to_case(entry: dict, *, stage: str, dP_Pa: float, rate: float, index: int) -> EnsembleCase:
    geom_npy = Path(entry["geom_npy"])
    geom_h5 = Path(entry["h5_path"])
    if not geom_npy.is_file():
        raise FileNotFoundError(f"Missing geometry: {geom_npy}")
    if not geom_h5.is_file():
        raise FileNotFoundError(f"Missing H5: {geom_h5}")
    return EnsembleCase(
        index=index,
        geom_seed=int(entry["geom_seed"]),
        geom_npy=geom_npy,
        geom_h5=geom_h5,
        l_r_median_um=float(entry["l_r_median_um"]),
        phi=float(entry["phi"]),
        dP_Pa=dP_Pa,
        rate_mol_cm2_s=rate,
        candidate_id=str(entry.get("candidate_id", "")),
        stage=stage,
    )


def load_stage_cases(stage: str) -> list[EnsembleCase]:
    spec = STAGE_SPECS[stage]
    if stage in ("stage1", "stage2"):
        entries = load_stage1_entries(spec.manifest_path)
        return [
            entry_to_case(
                e,
                stage=stage,
                dP_Pa=spec.dP_Pa,
                rate=spec.rate_mol_cm2_s,
                index=int(e["realization_index"]),
            )
            for e in entries
        ]
    if stage in MANIFEST_STAGES:
        if not spec.manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing {spec.manifest_path}. Run: python run_ensemble_transport.py {stage} plan"
            )
        payload = json.loads(spec.manifest_path.read_text())
        cases = []
        for i, entry in enumerate(payload["entries"]):
            cases.append(
                entry_to_case(
                    entry,
                    stage=stage,
                    dP_Pa=float(entry["dP_Pa"]),
                    rate=float(entry["rate_mol_cm2_s"]),
                    index=i,
                )
            )
        return cases
    raise ValueError(f"Unknown stage: {stage}")


def load_stage1_results() -> list[dict]:
    """Load finished stage-1 metrics CSV (required for overnight planning)."""
    csv_path = STAGE_SPECS["stage1"].out_root / "stage1_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}. Run stage1 summarize first.")
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def stage1_entry_by_geom(geom_seed: int) -> dict:
    for entry in load_stage1_entries():
        if int(entry["geom_seed"]) == int(geom_seed):
            return entry
    raise KeyError(f"geom_seed {geom_seed} not in stage1 manifest")


def clamp_dp(dP: float) -> float:
    return float(np.clip(dP, STAGE4_DP_CLAMP[0], STAGE4_DP_CLAMP[1]))


def write_manifest(path: Path, *, source: str, entries: list[dict], extra: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "entries": entries,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Wrote {path} ({len(entries)} cases)")
    return path


def plan_stage4a_theta_match(theta_target: float | None = None) -> Path:
    """Per-geometry dP so Theta @ 48h ~ constant (median stage-1 Theta), rate=1e-9."""
    rows = load_stage1_results()
    thetas = [float(r["theta_48h"]) for r in rows if np.isfinite(float(r["theta_48h"]))]
    if not thetas:
        raise ValueError("No finite theta_48h in stage1_results.csv")
    target = float(np.median(thetas)) if theta_target is None else float(theta_target)
    entries = []
    for row in sorted(rows, key=lambda r: int(r["case_index"])):
        geom_seed = int(row["geom_seed"])
        base = stage1_entry_by_geom(geom_seed)
        theta_ref = float(row["theta_48h"])
        dP_ref = float(row["dP_Pa"])
        dP_new = clamp_dp(dP_ref * target / max(theta_ref, 1e-12))
        entries.append(
            {
                **base,
                "stage4_index": len(entries),
                "dP_Pa": dP_new,
                "rate_mol_cm2_s": 1e-9,
                "theta_target_48h": target,
                "theta_ref_stage1": theta_ref,
                "dP_ref_stage1": dP_ref,
            }
        )
    return write_manifest(
        STAGE4_MANIFEST,
        source="ensemble_transport_lib.plan_stage4a_theta_match",
        entries=entries,
        extra={"theta_target_48h": target, "dP_Pa_clamp": list(STAGE4_DP_CLAMP)},
    )


def plan_stage4_theta_match(theta_target: float | None = None) -> Path:
    """Alias for plan_stage4a_theta_match (legacy CLI name)."""
    return plan_stage4a_theta_match(theta_target)


def plan_stage4b_theta_match_slow() -> Path:
    """Matched-Theta slow kinetics: reuse stage-4a per-geom dP, rate=1e-10."""
    if not STAGE4_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing {STAGE4_MANIFEST}. Run: python run_ensemble_transport.py stage4a_theta_match plan"
        )
    payload = json.loads(STAGE4_MANIFEST.read_text())
    entries = []
    for entry in payload["entries"]:
        entries.append(
            {
                **entry,
                "stage4b_index": len(entries),
                "stage4a_index": int(entry.get("stage4_index", len(entries))),
                "dP_Pa": float(entry["dP_Pa"]),
                "rate_mol_cm2_s": STAGE4B_RATE,
            }
        )
    return write_manifest(
        STAGE4B_MANIFEST,
        source="ensemble_transport_lib.plan_stage4b_theta_match_slow",
        entries=entries,
        extra={
            "rate_mol_cm2_s": STAGE4B_RATE,
            "stage4a_manifest": str(STAGE4_MANIFEST),
            "note": "Reuses stage-4a dP_Pa; Theta is flow-only.",
        },
    )


def plan_stage5_theta_swap() -> Path:
    """Swap exposure between low/high l_r pairs (tune dP to partner's stage-1 Theta)."""
    rows = {int(r["geom_seed"]): r for r in load_stage1_results()}
    entries = []
    for g_a, g_b in STAGE5_SWAP_PAIRS:
        if g_a not in rows or g_b not in rows:
            raise KeyError(f"Missing stage1 result for pair ({g_a}, {g_b})")
        theta_a = float(rows[g_a]["theta_48h"])
        theta_b = float(rows[g_b]["theta_48h"])
        for geom_seed, partner, theta_target in (
            (g_a, g_b, theta_b),
            (g_b, g_a, theta_a),
        ):
            row = rows[geom_seed]
            base = stage1_entry_by_geom(geom_seed)
            theta_ref = float(row["theta_48h"])
            dP_new = clamp_dp(float(row["dP_Pa"]) * theta_target / max(theta_ref, 1e-12))
            entries.append(
                {
                    **base,
                    "stage5_index": len(entries),
                    "dP_Pa": dP_new,
                    "rate_mol_cm2_s": 1e-9,
                    "swap_partner_geom": int(partner),
                    "theta_target_48h": float(theta_target),
                    "theta_ref_stage1": theta_ref,
                }
            )
    return write_manifest(
        STAGE5_MANIFEST,
        source="ensemble_transport_lib.plan_stage5_theta_swap",
        entries=entries,
        extra={"pairs": [list(p) for p in STAGE5_SWAP_PAIRS]},
    )


def plan_stage6_rate_mid() -> Path:
    """Mid-rate kinetic sweep: dP=1 Pa, rate=5e-10 on full stage-1 ensemble."""
    rows = load_stage1_results()
    entries = []
    for row in sorted(rows, key=lambda r: int(r["case_index"])):
        geom_seed = int(row["geom_seed"])
        base = stage1_entry_by_geom(geom_seed)
        entries.append(
            {
                **base,
                "stage6_index": len(entries),
                "dP_Pa": 1.0,
                "rate_mol_cm2_s": STAGE6_RATE,
            }
        )
    return write_manifest(
        STAGE6_MANIFEST,
        source="ensemble_transport_lib.plan_stage6_rate_mid",
        entries=entries,
        extra={"rate_mol_cm2_s": STAGE6_RATE},
    )


def plan_overnight_campaign() -> None:
    """Build all overnight manifests (stages 4a–6)."""
    plan_stage4a_theta_match()
    plan_stage4b_theta_match_slow()
    plan_stage5_theta_swap()
    plan_stage6_rate_mid()


def reuse_reference_from_stage1(case: EnsembleCase) -> Path | None:
    """Copy stage-1 k_pore calibration when the same geom_seed was already run."""
    try:
        entry = stage1_entry_by_geom(case.geom_seed)
    except KeyError:
        return None
    idx = int(entry["realization_index"])
    src = (
        STAGE_SPECS["stage1"].out_root
        / f"real{idx:03d}_geom{case.geom_seed}"
        / "reference"
        / "calibrate_k_pore_run"
        / "stokes_reference.npz"
    )
    if not reference_is_valid(src):
        return None
    return src


def plan_stage3_subset(n_geoms: int = STAGE3_N_GEOMS) -> list[dict]:
    """Pick low→high l_r geoms from Stage-1 manifest (no PFLOTRAN results needed)."""
    entries = sorted(load_stage1_entries(), key=lambda e: float(e["l_r_median_um"]))
    if len(entries) < n_geoms:
        raise ValueError(f"Need at least {n_geoms} Stage-1 entries, got {len(entries)}")
    quantiles = np.linspace(0.0, 1.0, n_geoms)
    picked: list[dict] = []
    used: set[int] = set()
    for q in quantiles:
        target = float(np.quantile([float(e["l_r_median_um"]) for e in entries], q))
        idx = min(
            range(len(entries)),
            key=lambda i: (abs(float(entries[i]["l_r_median_um"]) - target), i),
        )
        if idx in used:
            continue
        used.add(idx)
        picked.append(entries[idx])
    while len(picked) < n_geoms:
        for i, e in enumerate(entries):
            if i not in used:
                used.add(i)
                picked.append(e)
                break
        else:
            break
    return picked[:n_geoms]


def write_stage3_manifest(n_geoms: int = STAGE3_N_GEOMS) -> Path:
    geoms = plan_stage3_subset(n_geoms)
    entries = []
    idx = 0
    for geom in geoms:
        for dP in STAGE3_DPS:
            entries.append(
                {
                    **geom,
                    "stage3_index": idx,
                    "dP_Pa": dP,
                    "rate_mol_cm2_s": STAGE3_RATE,
                }
            )
            idx += 1
    STAGE3_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ensemble_transport_lib.plan_stage3_subset",
        "n_geoms": n_geoms,
        "rate_mol_cm2_s": STAGE3_RATE,
        "dP_Pa_values": list(STAGE3_DPS),
        "entries": entries,
    }
    STAGE3_MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Wrote {STAGE3_MANIFEST} ({len(entries)} cases)")
    return STAGE3_MANIFEST


def reference_npz(case: EnsembleCase, out_root: Path) -> Path:
    return case.reference_dir(out_root) / "calibrate_k_pore_run" / "stokes_reference.npz"


def ensure_reference(case: EnsembleCase, out_root: Path) -> Path:
    ref_dir = case.reference_dir(out_root)
    ref_npz = reference_npz(case, out_root)
    ref_dir.mkdir(parents=True, exist_ok=True)
    if case.stage in MANIFEST_STAGES and not reference_is_valid(ref_npz):
        src = reuse_reference_from_stage1(case)
        if src is not None:
            dst_dir = ref_npz.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, ref_npz)
            log(f"Reused stage1 k_pore for geom={case.geom_seed}: {ref_npz}")
    if reference_is_valid(ref_npz):
        log(f"Reusing k_pore: {ref_npz}")
        return ref_npz

    lock_path = ref_dir / ".calibrate.lock"
    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not reference_is_valid(ref_npz):
                raise RuntimeError(f"Missing valid {ref_npz}")
            return ref_npz
        if reference_is_valid(ref_npz):
            return ref_npz
        log(f"Calibrating k_pore geom_seed={case.geom_seed}")
        run_cmd(
            [
                sys.executable,
                "darcy_vs_stokes/simulate_darcy.py",
                "--geometry-h5",
                str(case.geom_h5),
                "--geometry-npy",
                str(case.geom_npy),
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
        if not reference_is_valid(ref_npz):
            raise RuntimeError(f"Calibration failed: {ref_npz}")
        return ref_npz


def run_one_case(
    case: EnsembleCase,
    out_root: Path,
    pflotran_exe: Path,
    *,
    skip_plots: bool = False,
) -> None:
    ref_npz = ensure_reference(case, out_root)
    k_pore = load_k_pore(ref_npz)
    workdir = case.workdir(out_root)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "figs").mkdir(exist_ok=True)

    log(
        f"{case.stage} [{case.index:03d}] geom={case.geom_seed} "
        f"l_r={case.l_r_median_um:.1f}µm dP={case.dP_Pa:g} rate={case.rate_mol_cm2_s:.0e}"
    )
    write_accessibility_transport_input(
        base_geometry=case.geom_h5,
        out_geometry=workdir / "geometry_accessibility.h5",
        out_input=workdir / "mineral_dissolution.in",
        geometry_npy=case.geom_npy,
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


def extract_metrics(case: EnsembleCase, out_root: Path) -> dict:
    v_bulk = LX * LY * LZ
    pulse_s = float(PULSE_DURATION_S)
    row = {
        "stage": case.stage,
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
    }
    ref = reference_npz(case, out_root)
    if ref.is_file() and reference_is_valid(ref):
        row["k_pore"] = load_k_pore(ref)
    h5_path = case.workdir(out_root) / "mineral_dissolution.h5"
    if not h5_path.is_file():
        return row
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
    return row


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    """Save with padding so colorbars and long titles are not clipped."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _summarize_plot_title(stage: str, rows: list[dict]) -> str:
    """Build figure title from per-case manifest values (not StageSpec defaults)."""
    dps = [float(r["dP_Pa"]) for r in rows if np.isfinite(float(r.get("dP_Pa", np.nan)))]
    rates = [
        float(r["rate_mol_cm2_s"])
        for r in rows
        if np.isfinite(float(r.get("rate_mol_cm2_s", np.nan)))
    ]
    if not dps:
        dp_part = "dP=?"
    elif len(set(round(d, 6) for d in dps)) == 1:
        dp_part = f"dP={dps[0]:g} Pa"
    else:
        dp_part = f"per-geom dP {min(dps):.2f}-{max(dps):.2f} Pa"

    rate_part = f"rate={rates[0]:.0e}" if rates and len(set(rates)) == 1 else "mixed rate"

    extra = ""
    if stage == "stage4a_theta_match":
        thetas = [float(r["theta_48h"]) for r in rows if np.isfinite(float(r.get("theta_48h", np.nan)))]
        if thetas:
            extra = f", theta_48h ~ {float(np.median(thetas)):.2f} PV"
        else:
            extra = ", theta matched"
    elif stage == "stage4b_theta_match_slow":
        extra = ", theta matched (4a dP)"
    elif stage == "stage5_theta_swap":
        extra = ", theta-swap pairs"

    return f"{stage}: recovery vs $l_r$ ({dp_part}{extra}, {rate_part})"


def summarize_stage(stage: str) -> Path:
    spec = STAGE_SPECS[stage]
    cases = load_stage_cases(stage)
    rows = [extract_metrics(c, spec.out_root) for c in cases]
    spec.out_root.mkdir(parents=True, exist_ok=True)
    csv_path = spec.out_root / f"{stage}_results.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")
    log(f"Saved {csv_path}")

    valid = [r for r in rows if np.isfinite(r["recovery_pct"])]
    if len(valid) < 2:
        return csv_path

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    k_pore = np.array([float(r["k_pore"]) for r in valid])
    sc = ax.scatter(
        [r["l_r_median_um"] for r in valid],
        [r["recovery_pct"] for r in valid],
        s=50,
        c=k_pore * 1e10,
        cmap="viridis",
        edgecolors="k",
        linewidths=0.3,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"$k_{\mathrm{pore}}$ [$10^{-10}$ m$^2$]")
    ax.set_xlabel(LBL_LR_MEDIAN)
    ax.set_ylabel(LBL_CU_RECOVERY)
    ax.set_title(_summarize_plot_title(stage, valid), loc="left")
    png = spec.out_root / f"{stage}_recovery_vs_l_r.png"
    _save_figure(fig, png)
    log(f"Saved {png}")
    return csv_path


def load_csv_rows(path: Path) -> list[dict]:
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    return [dict(zip(header, line.split(","))) for line in lines[1:]]


def load_accessible_cu_lookup() -> dict[str, float]:
    """Map candidate_id and geom_seed -> accessible Cu fraction from screening."""
    if not STAGE1_SELECTED.is_file():
        return {}
    lookup: dict[str, float] = {}
    for row in json.loads(STAGE1_SELECTED.read_text()):
        frac = float(row["accessible_cu_frac"])
        lookup[str(row["candidate_id"])] = frac
        lookup[str(row["geom_seed"])] = frac
    return lookup


def stokes_reference_npz_for_geom(geom_seed: int) -> Path:
    """Stage-1 Stokes reference reused by stages 4-6 for the same geom_seed."""
    entry = stage1_entry_by_geom(geom_seed)
    idx = int(entry["realization_index"])
    return (
        STAGE_SPECS["stage1"].out_root
        / f"real{idx:03d}_geom{geom_seed}"
        / "reference"
        / "calibrate_k_pore_run"
        / "stokes_reference.npz"
    )


def _stokes_ux_ref_at_dp(geom_seed: int) -> tuple[float, float]:
    """Mean |u_x| on inlet-connected pore cells at the Stokes reference dP."""
    if geom_seed in _STOKES_UX_REF_CACHE:
        return _STOKES_UX_REF_CACHE[geom_seed]
    ref_npz = stokes_reference_npz_for_geom(geom_seed)
    if not ref_npz.is_file():
        raise FileNotFoundError(ref_npz)
    entry = stage1_entry_by_geom(geom_seed)
    pore_flow = np.asarray(np.load(entry["geom_npy"]).T > 0.5, dtype=bool)
    connected = connected_pore_from_inlet(pore_flow, inlet="left")
    data = np.load(ref_npz)
    u = np.asarray(data["u"], dtype=float)
    fluid = np.asarray(data["mask"], dtype=float) > 0.0
    dP_ref = float(data["dP"]) if "dP" in data.files else 1.0
    sel = connected & fluid
    if not np.any(sel):
        raise ValueError(f"No inlet-connected fluid cells for geom_seed={geom_seed}")
    u_ref = float(np.mean(np.abs(u[sel])))
    if u_ref <= 0.0:
        raise ValueError(f"Non-positive Stokes u_ref for geom_seed={geom_seed}")
    _STOKES_UX_REF_CACHE[geom_seed] = (u_ref, dP_ref)
    return u_ref, dP_ref


def velocity_from_stokes_connected_m_s(geom_seed: int, dP_Pa: float) -> float:
    """Scale mean |u_x| on inlet-connected pores linearly with dP (Stokes reference)."""
    u_ref, dP_ref = _stokes_ux_ref_at_dp(geom_seed)
    return u_ref * float(dP_Pa) / max(dP_ref, 1e-15)


def case_h5_path_from_row(row: dict) -> Path | None:
    """Resolve per-case mineral_dissolution.h5 from a results CSV row."""
    stage = str(row["stage"])
    if stage not in STAGE_SPECS:
        return None
    spec = STAGE_SPECS[stage]
    case_index = int(row["case_index"])
    geom_seed = int(row["geom_seed"])
    if stage in MANIFEST_STAGES:
        dP = float(row["dP_Pa"])
        rate = float(row["rate_mol_cm2_s"])
        workdir = (
            spec.out_root
            / f"real{case_index:03d}_geom{geom_seed}_dp{tag_dp(dP)}_rate{tag_rate(rate)}"
        )
    else:
        workdir = spec.out_root / f"real{case_index:03d}_geom{geom_seed}"
    h5_path = workdir / "mineral_dissolution.h5"
    return h5_path if h5_path.is_file() else None


def _pore_mask_for_pflotran_grid(geom_npy: Path, nx: int, ny: int) -> np.ndarray:
    """Pore mask on PFLOTRAN's native (nx, ny) grid (not the transposed l_r frame)."""
    geom = np.load(geom_npy)
    pore = np.asarray(geom > 0.5, dtype=bool)
    if pore.shape == (nx, ny):
        return pore
    pore_t = np.asarray(geom.T > 0.5, dtype=bool)
    if pore_t.shape == (nx, ny):
        return pore_t
    raise ValueError(f"geom {geom_npy} shape {geom.shape} incompatible with H5 ({nx}, {ny})")


def darcy_mean_ux_connected_from_h5(h5_path: Path, geom_npy: Path) -> float:
    """Mean |u_x| on inlet-connected pore cells from PFLOTRAN Darcy output."""
    cache_key = f"{h5_path.resolve()}:{geom_npy.resolve()}"
    if cache_key in _DARCY_UX_CACHE:
        return _DARCY_UX_CACHE[cache_key]
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if not geom_npy.is_file():
        raise FileNotFoundError(geom_npy)

    ux = None
    mat = None
    with h5py.File(h5_path, "r") as h5:
        for t, gname in parse_time_groups(h5):
            if t <= 0.0:
                continue
            g = h5[gname]
            ux_key = None
            for k in g.keys():
                if "Liquid X-Velocity" in k:
                    ux_key = k
                    break
            if ux_key is None:
                continue
            arr = np.asarray(g[ux_key], dtype=float)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if "m_per_h" in ux_key:
                arr = arr / 3600.0
            if float(np.max(np.abs(arr))) <= 0.0:
                continue
            ux = arr
            mat = np.asarray(g["Material_ID"], dtype=int)
            if mat.ndim == 3:
                mat = mat[:, :, 0]
            break
        if ux is None:
            raise ValueError(f"No nonzero Darcy u_x in {h5_path}")

    pore = _pore_mask_for_pflotran_grid(geom_npy, ux.shape[0], ux.shape[1])
    connected = connected_pore_from_inlet(pore, inlet="left")
    fluid = mat == 1
    sel = connected & fluid
    if not np.any(sel):
        raise ValueError(f"No inlet-connected pore cells for {h5_path}")
    u_mean = float(np.mean(np.abs(ux[sel])))
    if u_mean <= 0.0:
        raise ValueError(f"Non-positive Darcy u_x for {h5_path}")
    _DARCY_UX_CACHE[cache_key] = u_mean
    return u_mean


def velocity_from_theta_m_s(theta_48h: float) -> float:
    """Outlet-equivalent superficial velocity: u = Theta * V_bulk / (t_pulse * A_out)."""
    v_bulk = LX * LY * LZ
    a_out = LY * LZ
    return float(theta_48h) * v_bulk / (float(PULSE_DURATION_S) * a_out)


def characteristic_velocity_m_s(k_pore: float, dP_Pa: float) -> float:
    """Legacy Darcy estimate (k_pore proxy only)."""
    return float(k_pore) * float(dP_Pa) / (FLUID_VISCOSITY_PA_S * LX)


def compute_pe_da_metrics(row: dict) -> dict[str, float]:
    """Accessibility-aware Pe_r and Da_r using PFLOTRAN Darcy |u_x| on inlet-connected pores."""
    l_r_m = float(row["l_r_median_um"]) * 1e-6
    geom_seed = int(row["geom_seed"])
    dP_pa = float(row["dP_Pa"])
    theta = float(row.get("theta_48h", np.nan))
    u_method = "darcy_mean_ux_connected"
    u = np.nan
    h5_path = case_h5_path_from_row(row)
    if h5_path is not None:
        try:
            geom_npy = Path(stage1_entry_by_geom(geom_seed)["geom_npy"])
            u = darcy_mean_ux_connected_from_h5(h5_path, geom_npy)
        except (KeyError, FileNotFoundError, ValueError, OSError):
            u = np.nan
    if not np.isfinite(u) or u <= 0.0:
        u_method = "stokes_mean_ux_connected"
        try:
            u = velocity_from_stokes_connected_m_s(geom_seed, dP_pa)
        except (KeyError, FileNotFoundError, ValueError, OSError):
            u = np.nan
    if not np.isfinite(u) or u <= 0.0:
        if np.isfinite(theta) and theta > 0.0:
            u = velocity_from_theta_m_s(theta)
            u_method = "theta_outlet"
        else:
            u = max(characteristic_velocity_m_s(float(row["k_pore"]), dP_pa), 1e-15)
            u_method = "k_pore_proxy"
    u = max(float(u), 1e-15)
    u_theta = velocity_from_theta_m_s(theta) if np.isfinite(theta) and theta > 0 else np.nan
    pe_r = u * l_r_m / PORE_DIFFUSIVITY_M2S
    k_r = float(row["rate_mol_cm2_s"]) * 1e4
    f_acc = float(row.get("accessible_cu_frac", np.nan))
    f_use = f_acc if np.isfinite(f_acc) else 1.0
    da_r = k_r * SSA_CHRYSOCOLLA_M2_M3 * f_use * l_r_m / u
    return {
        "u_ms": u,
        "u_theta_ms": u_theta,
        "u_velocity_method": u_method,
        "Pe_r": pe_r,
        "Da_r": da_r,
    }


def enrich_rows(rows: list[dict], cu_lookup: dict[str, float]) -> list[dict]:
    """Add accessibility collapse metrics to result rows."""
    out = []
    for row in rows:
        r = dict(row)
        try:
            theta = float(r["theta_48h"])
            l_r_m = float(r["l_r_median_um"]) * 1e-6
            rec = float(r["recovery_pct"])
        except (KeyError, ValueError, TypeError):
            out.append(r)
            continue
        cu_frac = cu_lookup.get(str(r.get("candidate_id", "")), np.nan)
        if not np.isfinite(cu_frac):
            cu_frac = cu_lookup.get(str(r.get("geom_seed", "")), np.nan)
        r["accessible_cu_frac"] = cu_frac
        r["theta_exp_lr"] = theta * np.exp(-l_r_m / COLLAPSE_LAMBDA_M)
        if np.isfinite(cu_frac):
            r["theta_cu_access"] = theta * cu_frac
        else:
            r["theta_cu_access"] = np.nan
        r["recovery_pct_f"] = rec
        r["theta_48h_f"] = theta
        r["l_r_median_um_f"] = float(r["l_r_median_um"])
        pe_da = compute_pe_da_metrics(r)
        r["u_ms"] = pe_da["u_ms"]
        r["u_theta_ms"] = pe_da["u_theta_ms"]
        r["u_velocity_method"] = pe_da["u_velocity_method"]
        r["Pe_r"] = pe_da["Pe_r"]
        r["Da_r"] = pe_da["Da_r"]
        out.append(r)
    return out


ANALYZE_STAGES = (
    "stage1",
    "stage2",
    "stage3",
    "stage4a_theta_match",
    "stage4b_theta_match_slow",
    "stage5_theta_swap",
    "stage6_rate_mid",
)


def _series_style() -> list[tuple[str, float | None, str, str]]:
    return [
        ("stage1", None, "stage1 dP=1 rate=1e-9", "o"),
        ("stage2", None, "stage2 dP=0.1 rate=1e-9", "s"),
        ("stage3", 1.0, "stage3 dP=1 rate=1e-10", "^"),
        ("stage3", 0.1, "stage3 dP=0.1 rate=1e-10", "v"),
        ("stage4a_theta_match", None, "stage4a theta-match rate=1e-9", "D"),
        ("stage4b_theta_match_slow", None, "stage4b theta-match rate=1e-10", "X"),
        ("stage5_theta_swap", None, "stage5 theta-swap rate=1e-9", "P"),
        ("stage6_rate_mid", None, "stage6 dP=1 rate=5e-10", "h"),
    ]


def _filter_rows(rows: list[dict], stage: str, *, dP: float | None = None) -> list[dict]:
    out = []
    for r in rows:
        if normalize_stage_name(str(r.get("stage", ""))) != normalize_stage_name(stage):
            continue
        try:
            float(r["recovery_pct"])
            if dP is not None and abs(float(r["dP_Pa"]) - dP) > 1e-9:
                continue
        except (KeyError, ValueError, TypeError):
            continue
        out.append(r)
    return out


def plot_collapse_figure(rows: list[dict], out_png: Path) -> None:
    """Collapse test: recovery vs accessibility-weighted exposure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5))
    styles = _series_style()

    for stage, dP, label, marker in styles:
        sub = _filter_rows(rows, stage, dP=dP)
        if not sub:
            continue
        axes[0].scatter(
            [float(r["theta_exp_lr"]) for r in sub],
            [float(r["recovery_pct"]) for r in sub],
            s=50,
            marker=marker,
            label=label,
        )
        axes[1].scatter(
            [float(r["theta_cu_access"]) for r in sub if np.isfinite(float(r["theta_cu_access"]))],
            [float(r["recovery_pct"]) for r in sub if np.isfinite(float(r["theta_cu_access"]))],
            s=50,
            marker=marker,
            label=label,
        )

    lam_um = COLLAPSE_LAMBDA_M * 1e6
    axes[0].set_xlabel(rf"$\Theta \exp(-l_r/\lambda)$, $\lambda={lam_um:.0f}$ µm")
    axes[0].set_ylabel(LBL_CU_RECOVERY)
    axes[0].set_title("(a) Exponential accessibility collapse", loc="left")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel(r"$\Theta \times f_{\mathrm{Cu,acc}}$")
    axes[1].set_ylabel(LBL_CU_RECOVERY)
    axes[1].set_title("(b) Accessible-inventory collapse", loc="left")
    axes[1].legend(fontsize=7)

    _save_figure(fig, out_png)
    log(f"Saved {out_png}")


def write_qc_summary(rows: list[dict], out_path: Path) -> None:
    """Write short QC stats for stage-1 ensemble at dP=1."""
    s1 = _filter_rows(rows, "stage1")
    if len(s1) < 2:
        return
    lr = np.array([float(r["l_r_median_um"]) for r in s1])
    rec = np.array([float(r["recovery_pct"]) for r in s1])
    theta = np.array([float(r["theta_48h"]) for r in s1])
    tca = np.array([float(r["theta_cu_access"]) for r in s1 if np.isfinite(float(r["theta_cu_access"]))])
    lines = [
        "Ensemble QC summary (stage1, dP=1 Pa, rate=1e-9)",
        f"n_cases: {len(s1)}",
        f"l_r_median_um: min={lr.min():.1f} max={lr.max():.1f} span={lr.max()/lr.min():.2f}x",
        f"recovery_pct: min={rec.min():.2f} max={rec.max():.2f}",
        f"theta_48h: min={theta.min():.2f} max={theta.max():.2f}",
        f"corr(recovery, l_r): {np.corrcoef(rec, lr)[0,1]:.3f}",
        f"corr(recovery, theta): {np.corrcoef(rec, theta)[0,1]:.3f}",
    ]
    if len(tca) == len(rec):
        lines.append(f"corr(recovery, theta_cu_access): {np.corrcoef(rec, tca)[0,1]:.3f}")
    s2 = _filter_rows(rows, "stage2")
    if s2:
        rec2 = np.array([float(r["recovery_pct"]) for r in s2])
        lines.append(f"stage2 recovery_pct: min={rec2.min():.2f} max={rec2.max():.2f}")
    out_path.write_text("\n".join(lines) + "\n")
    log(f"Saved {out_path}")


def plot_anchor_cases(out_root: Path) -> None:
    """Postprocess low/mid/high l_r stage-1 anchors."""
    cases = load_stage_cases("stage1")
    if not cases:
        return
    ranked = sorted(cases, key=lambda c: c.l_r_median_um)
    picks = [("low", ranked[0]), ("mid", ranked[len(ranked) // 2]), ("high", ranked[-1])]
    anchor_dir = out_root / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    spec = STAGE_SPECS["stage1"]
    for tag, case in picks:
        workdir = case.workdir(spec.out_root)
        h5 = workdir / "mineral_dissolution.h5"
        if not h5.is_file():
            log(f"SKIP anchor {tag}: missing {h5}")
            continue
        fig_dir = anchor_dir / f"{tag}_geom{case.geom_seed}"
        fig_dir.mkdir(parents=True, exist_ok=True)
        log(f"Anchor {tag}: geom={case.geom_seed} l_r={case.l_r_median_um:.1f} µm")
        run_cmd(
            [
                sys.executable,
                str(SRC_DIR / "plot_accessibility_mineral_dissolution.py"),
                str(h5),
                "--out-dir",
                str(fig_dir),
                "--prefix",
                f"{tag}_geom{case.geom_seed}",
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


def _pe_da_series_meta() -> list[tuple[str, float | None, str, str, str]]:
    """Stage series: (stage, dP, legend, marker, short label)."""
    return [
        ("stage1", None, "stage1 dP=1 rate=1e-9", "o", "stg1"),
        ("stage2", None, "stage2 dP=0.1 rate=1e-9", "s", "stg2"),
        ("stage3", 1.0, "stage3 dP=1 rate=1e-10", "^", "stg3"),
        ("stage3", 0.1, "stage3 dP=0.1 rate=1e-10", "v", "stg3'"),
        ("stage4a_theta_match", None, "stage4a theta-match rate=1e-9", "D", "stg4a"),
        ("stage4b_theta_match_slow", None, "stage4b theta-match rate=1e-10", "X", "stg4b"),
        ("stage5_theta_swap", None, "stage5 theta-swap rate=1e-9", "P", "stg5"),
        ("stage6_rate_mid", None, "stage6 dP=1 rate=5e-10", "h", "stg6"),
    ]


def _pe_da_series_colors() -> list[str]:
    """Colorblind-friendly discrete colors per stage series."""
    return ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#66CCEE", "#BBBB44", "#888888"]


def _log_ellipse_curve(
    pe: np.ndarray, da: np.ndarray, *, n_std: float = 1.4, n_pts: int = 72
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Closed curve for a log-space centroid ellipse (Pe, Da arrays)."""
    lpe = np.log10(pe)
    lda = np.log10(da)
    cx, cy = float(np.mean(lpe)), float(np.mean(lda))
    ax_std = max(float(np.std(lpe)), 0.04)
    ay_std = max(float(np.std(lda)), 0.04)
    t = np.linspace(0.0, 2.0 * np.pi, n_pts)
    x = 10 ** (cx + n_std * ax_std * np.cos(t))
    y = 10 ** (cy + n_std * ay_std * np.sin(t))
    return x, y, cx, cy


def plot_pe_da_regime_figure(rows: list[dict], out_png: Path) -> None:
    """Pe_r-Da_r regime map: (a) all stages; (b) matched-Theta stages 4a and 4b."""
    valid = []
    for row in rows:
        try:
            pe = float(row["Pe_r"])
            da = float(row["Da_r"])
            float(row["recovery_pct"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (np.isfinite(pe) and np.isfinite(da) and pe > 0 and da > 0):
            continue
        valid.append(row)
    if len(valid) < 2:
        log(f"SKIP {out_png.name}: insufficient Pe_r/Da_r rows")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, (ax, ax_s4) = plt.subplots(
        1,
        2,
        figsize=(11.0, 5.6),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )
    colors = _pe_da_series_colors()
    overlap_stages = {"stage1", "stage4a_theta_match", "stage5_theta_swap", "stage6_rate_mid"}
    overlap_pe: list[float] = []
    overlap_da: list[float] = []

    for i, (stage, dP, label, marker, short) in enumerate(_pe_da_series_meta()):
        sub = _filter_rows(valid, stage, dP=dP)
        if not sub:
            continue
        pe = np.array([float(r["Pe_r"]) for r in sub])
        da = np.array([float(r["Da_r"]) for r in sub])
        sizes = [22.0 + 4.0 * float(r["recovery_pct"]) for r in sub]
        color = colors[i % len(colors)]
        ax.scatter(
            pe,
            da,
            s=sizes,
            c=color,
            marker=marker,
            edgecolors="0.2",
            linewidths=0.4,
            alpha=0.88,
            label=label,
            zorder=3,
        )
        ex, ey, lcx, lcy = _log_ellipse_curve(pe, da)
        ax.fill(ex, ey, color=color, alpha=0.12, zorder=1)
        ax.plot(ex, ey, color=color, lw=0.9, alpha=0.55, zorder=2)
        ax.text(
            10**lcx,
            10**lcy,
            short,
            fontsize=7,
            ha="center",
            va="center",
            color="0.15",
            fontweight="bold",
            zorder=4,
        )
        if stage in overlap_stages:
            overlap_pe.extend(pe.tolist())
            overlap_da.extend(da.tolist())

    if overlap_pe:
        op = np.array(overlap_pe)
        od = np.array(overlap_da)
        _, _, lcx, lcy = _log_ellipse_curve(op, od, n_std=1.1)
        ax.text(
            10 ** (lcx + 0.12),
            10 ** (lcy + 0.18),
            "stg1/4a/5/6\noverlap",
            fontsize=6.5,
            ha="center",
            va="bottom",
            color="0.35",
            style="italic",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.8", "alpha": 0.9},
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Pe_r = u\,l_r/D$")
    ax.set_ylabel(r"$Da_r = k_r\,SSA\,f_{\mathrm{Cu,acc}}\,l_r/u$")
    ax.set_title(
        r"(a) All stages: color = stage" + "\n"
        r"marker size $\propto$ recovery; $u$ = Darcy mean$|u_x|$ (inlet-connected)",
        loc="left",
        fontsize=8.5,
        pad=12,
    )
    ax.text(0.02, 0.02, "ellipses: stage centroids", transform=ax.transAxes, fontsize=6, color="0.4")
    ax.legend(fontsize=6, loc="upper right", framealpha=0.92, ncol=1)

    # Panel (b): matched-Theta stages 4a and 4b, colored by l_r
    s4a = _filter_rows(valid, "stage4a_theta_match")
    s4b = _filter_rows(valid, "stage4b_theta_match_slow")
    s4_all = s4a + s4b
    if len(s4_all) >= 2:
        lr_all = np.array([float(r["l_r_median_um"]) for r in s4_all])
        lr_norm = plt.Normalize(vmin=float(lr_all.min()), vmax=float(lr_all.max()))
        cmap = plt.get_cmap("viridis")
        pe_all: list[float] = []
        da_all: list[float] = []
        sc = None
        for stage, marker, label in (
            ("stage4a_theta_match", "D", "stage4a rate=1e-9"),
            ("stage4b_theta_match_slow", "X", "stage4b rate=1e-10"),
        ):
            sub = _filter_rows(valid, stage)
            if len(sub) < 1:
                continue
            pe = np.array([float(r["Pe_r"]) for r in sub])
            da = np.array([float(r["Da_r"]) for r in sub])
            lr = np.array([float(r["l_r_median_um"]) for r in sub])
            rec = np.array([float(r["recovery_pct"]) for r in sub])
            pe_all.extend(pe.tolist())
            da_all.extend(da.tolist())
            sc = ax_s4.scatter(
                pe,
                da,
                c=lr,
                cmap=cmap,
                norm=lr_norm,
                marker=marker,
                s=28.0 + 4.0 * rec,
                edgecolors="0.2",
                linewidths=0.35,
                label=label,
                zorder=3,
            )
        ax_s4.set_xscale("log")
        ax_s4.set_yscale("log")
        pe_arr = np.array(pe_all)
        da_arr = np.array(da_all)
        pad_pe = 10 ** (0.15 * (np.log10(pe_arr.max()) - np.log10(pe_arr.min()) + 0.05))
        pad_da = 10 ** (0.15 * (np.log10(da_arr.max()) - np.log10(da_arr.min()) + 0.05))
        ax_s4.set_xlim(pe_arr.min() / pad_pe, pe_arr.max() * pad_pe)
        ax_s4.set_ylim(da_arr.min() / pad_da, da_arr.max() * pad_da)
        ax_s4.set_xlabel(r"$Pe_r = u\,l_r/D$")
        ax_s4.set_ylabel(r"$Da_r = k_r\,SSA\,f_{\mathrm{Cu,acc}}\,l_r/u$")
        ax_s4.set_title(
            r"(b) Matched-$\Theta$ stg4a ($k_r=10^{-9}$) and 4b ($10^{-10}$)" + "\n"
            r"color = $l_r$; marker size $\propto$ recovery",
            loc="left",
            fontsize=8.5,
            pad=12,
        )
        ax_s4.legend(fontsize=6, loc="lower right", framealpha=0.92)
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax_s4, shrink=0.88, pad=0.06)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(r"$l_r$ [µm]", fontsize=8)
    else:
        ax_s4.set_visible(False)

    _save_figure(fig, out_png)
    log(f"Saved {out_png}")


def plot_sweep_efficiency_figure(rows: list[dict], out_png: Path) -> None:
    """Stage-1: Theta vs k_pore at fixed dP; recovery as color (sweep exceeds k scaling)."""
    sub = _filter_rows(rows, "stage1")
    if len(sub) < 2:
        return
    k = np.array([float(r["k_pore"]) for r in sub])
    theta = np.array([float(r["theta_48h"]) for r in sub])
    rec = np.array([float(r["recovery_pct"]) for r in sub])
    seeds = [int(r["geom_seed"]) for r in sub]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(5.8, 4.2), layout="constrained")
    sc = ax.scatter(k * 1e10, theta, c=rec, s=70, cmap="viridis", edgecolor="0.2", linewidth=0.4)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Recovery [%]")

    slope = float(np.median(theta / k))
    k_line = np.linspace(k.min(), k.max(), 50)
    ax.plot(k_line * 1e10, slope * k_line, "k--", lw=1.2, alpha=0.7, label=r"median $\Theta/k_{\mathrm{pore}}$")

    for outlier in (np.argmin(theta / k), np.argmax(theta / k)):
        ax.annotate(
            f"geom {seeds[outlier]}",
            (k[outlier] * 1e10, theta[outlier]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel(r"$k_{\mathrm{pore}}$ [$10^{-10}$ m$^2$]")
    ax.set_ylabel(r"$\Theta$ @ 48 h")
    ax.set_title(
        rf"(a) $\Theta$ span {theta.max()/theta.min():.1f}$\times$ vs "
        rf"$k_{{\mathrm{{pore}}}}$ span {k.max()/k.min():.1f}$\times$ (stage 1)",
        loc="left",
    )
    ax.legend(fontsize=7, loc="upper left")
    _save_figure(fig, out_png)
    log(f"Saved {out_png}")


def _fit_recovery_decay(l_r_um: np.ndarray, recovery_pct: np.ndarray) -> tuple[float, float] | None:
    """Fit R = R0 * exp(-l_r/lambda) with l_r in um; return (R0, lambda_um) or None."""
    if len(l_r_um) < 3:
        return None
    try:
        from scipy.optimize import curve_fit

        def model(lr: np.ndarray, r0: float, lam: float) -> np.ndarray:
            return r0 * np.exp(-lr / lam)

        popt, _ = curve_fit(
            model,
            l_r_um,
            recovery_pct,
            p0=(float(recovery_pct.max()), 80.0),
            bounds=([0.0, 1.0], [100.0, 500.0]),
            maxfev=8000,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return None


def plot_stage4_matched_theta_recovery_figure(rows: list[dict], out_png: Path) -> None:
    """Single-panel R vs l_r for matched-Theta stages 4a and 4b with illustrative decay guides."""
    s4a = _filter_rows(rows, "stage4a_theta_match")
    s4b = _filter_rows(rows, "stage4b_theta_match_slow")
    if len(s4a) < 2 and len(s4b) < 2:
        log(f"SKIP {out_png.name}: insufficient stage 4a/4b rows")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), layout="constrained")

    series = (
        ("stage4a_theta_match", s4a, "o", "#2166ac", r"$k_r = 10^{-9}$ mol cm$^{-2}$ s$^{-1}$"),
        ("stage4b_theta_match_slow", s4b, "s", "#d95f02", r"$k_r = 10^{-10}$ mol cm$^{-2}$ s$^{-1}$"),
    )
    lr_line = np.linspace(40.0, 120.0, 200)
    legend_handles: list = []
    stats_lines: list[str] = []

    for _stage, sub, marker, color, label in series:
        if len(sub) < 2:
            continue
        lr = np.array([float(r["l_r_median_um"]) for r in sub])
        rec = np.array([float(r["recovery_pct"]) for r in sub])
        thetas = [float(r["theta_48h"]) for r in sub if np.isfinite(float(r.get("theta_48h", np.nan)))]
        sc = ax.scatter(
            lr,
            rec,
            s=58,
            marker=marker,
            c=color,
            edgecolors="0.15",
            linewidths=0.4,
            alpha=0.92,
            label=label,
            zorder=3,
        )
        legend_handles.append(sc)
        corr = float(np.corrcoef(rec, lr)[0, 1])
        fit = _fit_recovery_decay(lr, rec)
        if fit is not None:
            r0, lam = fit
            ax.plot(
                lr_line,
                r0 * np.exp(-lr_line / lam),
                "--",
                color=color,
                lw=1.6,
                alpha=0.85,
                zorder=2,
            )
            stats_lines.append(
                f"{label}: corr(R, l_r)={corr:.2f}; guide R=R0 exp(-l_r/{lam:.0f} um), R0={r0:.1f}%"
            )
        else:
            stats_lines.append(f"{label}: corr(R, l_r)={corr:.2f}")

    theta_med = float(np.median([float(r["theta_48h"]) for r in s4a + s4b]))
    ax.set_xlabel(LBL_LR_MEDIAN)
    ax.set_ylabel(LBL_CU_RECOVERY)
    ax.set_title(
        rf"Matched cumulative pore volumes flushed at 48 h ($\Theta \approx {theta_med:.1f}$ PV)",
        loc="left",
        fontsize=10.5,
    )
    ax.set_xlim(38, 122)
    ax.set_ylim(bottom=0)
    ax.legend(handles=legend_handles, frameon=True, fontsize=8.5, loc="upper right")
    # ax.text(
    #     0.02,
    #     0.02,
    #     "\n".join(stats_lines),
    #     transform=ax.transAxes,
    #     fontsize=7.2,
    #     va="bottom",
    #     ha="left",
    #     bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "alpha": 0.95},
    # )
    _save_figure(fig, out_png)
    log(f"Saved {out_png}")


def _load_stage_result_rows(stage: str) -> list[dict]:
    csv_path = stage_results_csv(stage)
    if not csv_path.is_file():
        log(f"SKIP missing {csv_path}")
        return []
    rows = load_csv_rows(csv_path)
    for row in rows:
        if "stage" in row:
            row["stage"] = normalize_stage_name(str(row["stage"]))
    return rows


def analyze_stages(*, figures_only: bool = False) -> None:
    """Combine stage 1-6 result CSVs into scaling overview and collapse figures."""
    all_rows: list[dict] = []
    for stage in ANALYZE_STAGES:
        all_rows.extend(_load_stage_result_rows(stage))

    if not all_rows:
        raise FileNotFoundError("No stage results CSVs found. Run summarize after PFLOTRAN.")

    out_dir = SWEEPS_ROOT / "scaling_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = out_dir / "all_stages_results.csv"
    fields = list(all_rows[0].keys())
    with combined.open("w") as f:
        f.write(",".join(fields) + "\n")
        for row in all_rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")
    log(f"Saved {combined} ({len(all_rows)} rows)")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), layout="constrained")

    for stage, dP, label, marker in _series_style():
        sub = _filter_rows(all_rows, stage, dP=dP)
        if not sub:
            continue
        axes[0].scatter(
            [float(r["l_r_median_um"]) for r in sub],
            [float(r["recovery_pct"]) for r in sub],
            s=45,
            marker=marker,
            label=label,
        )
        axes[1].scatter(
            [float(r["theta_48h"]) for r in sub],
            [float(r["recovery_pct"]) for r in sub],
            s=55,
            marker=marker,
            label=label,
        )
    axes[0].set_xlabel(LBL_LR_MEDIAN)
    axes[0].set_ylabel(LBL_CU_RECOVERY)
    axes[0].set_title(rf"(a) {LBL_CU_RECOVERY} vs $l_r$", loc="left")
    axes[0].legend(fontsize=6, ncol=2, loc="upper right")

    axes[1].set_xlabel(LBL_THETA_48H)
    axes[1].set_ylabel(LBL_CU_RECOVERY)
    axes[1].set_title(rf"(b) {LBL_CU_RECOVERY} vs $\Theta$", loc="left")
    axes[1].legend(fontsize=6, ncol=2, loc="upper left")

    png = out_dir / "scaling_stages_overview.png"
    _save_figure(fig, png)
    log(f"Saved {png}")

    cu_lookup = load_accessible_cu_lookup()
    enriched = enrich_rows(all_rows, cu_lookup)
    enriched_csv = out_dir / "all_stages_enriched.csv"
    if enriched:
        efields = list(enriched[0].keys())
        with enriched_csv.open("w") as f:
            f.write(",".join(efields) + "\n")
            for row in enriched:
                f.write(",".join(str(row[k]) for k in efields) + "\n")
        log(f"Saved {enriched_csv}")

    plot_stage4_matched_theta_recovery_figure(
        all_rows, out_dir / "stage4_matched_theta_recovery_vs_l_r.png"
    )
    plot_collapse_figure(enriched, out_dir / "scaling_collapse.png")
    plot_sweep_efficiency_figure(enriched, out_dir / "scaling_sweep_efficiency.png")
    plot_pe_da_regime_figure(enriched, out_dir / "scaling_pe_da_regime.png")
    write_qc_summary(enriched, out_dir / "qc_summary.txt")
    if figures_only:
        log("SKIP anchor postprocess (--figures-only)")
    else:
        plot_anchor_cases(out_dir)