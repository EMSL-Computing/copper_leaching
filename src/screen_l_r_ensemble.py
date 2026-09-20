"""Screen bean-bucket realizations for l_r coverage before PFLOTRAN runs.

Compares geometry-only, mineral-only, and combined seed strategies, then
selects 5-10 candidates spanning l_r at similar porosity.

Example:
    python screen_l_r_ensemble.py pilot
    python screen_l_r_ensemble.py scale --strategy geometry
    python screen_l_r_ensemble.py select
    python screen_l_r_ensemble.py stage1-screen
    python screen_l_r_ensemble.py stage1-select --n 25
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import _repo  # noqa: F401 — path bootstrap

from build_pflotran_hdf5_from_npy import load_geometry_npy
from compute_reactive_accessibility_length import (
    accessible_cu_fraction,
    compute_accessibility_stats,
    connected_pore_from_inlet,
    left_right_pore_connected,
    load_mask_from_h5,
    mineral_access_distance_map,
)
from normalize_design_to_solver import normalize_design_to_solver
from prepare_accessibility_geometry import build_one_mineral_realization

from _repo import REPO_ROOT, SRC_DIR

HERE = REPO_ROOT
# Historical paths only. Paper screening figure is frozen at
# accessibility_ensemble/ensemble/screening/ — do NOT re-run this script for the draft
# (would regenerate geometries). Tree lives under archive/Chip_Designs/.
NETWORK_NAME = "bean_bucket_network"
SCREEN_ROOT = REPO_ROOT / "archive" / "Chip_Designs" / NETWORK_NAME / "ensemble_screening"
FIXED_GEOM = REPO_ROOT / "archive" / "Chip_Designs" / NETWORK_NAME / "geometry" / "network_correct.npy"
REALIZATIONS_DIR = REPO_ROOT / "archive" / "Chip_Designs" / NETWORK_NAME / "chrysocolla_realizations"

DX = 20e-6
INLET = "left"
PHI_MIN = 0.06
PHI_MAX = 0.30
CHRYSOCOLLA_SOLID_FRACTION = 0.50
MINERAL_SEED_BASE = 42
GEOM_SEED_BASE = 400
PILOT_N = 15
SCALE_N_GEOM = 25
SCALE_N_MINERAL = 25
SELECT_N_TARGET = 8
STAGE1_SCREEN_N = 75
STAGE1_SELECT_N = 25
STAGE1_MANIFEST = (
    REPO_ROOT / "archive" / "Chip_Designs" / NETWORK_NAME / "stage1_realizations" / "stage1_manifest.json"
)

STRATEGY_LABELS = {
    "geometry": "A geometry",
    "mineral": "B mineral",
    "combined": "C combined",
}

CSV_FIELDS = [
    "candidate_id",
    "strategy",
    "geom_seed",
    "mineral_seed",
    "phi",
    "l_r_median_um",
    "l_r_p75_um",
    "l_r_p90_um",
    "accessible_cu_frac",
    "connected_pore_frac",
    "left_right_connected",
    "n_mineral_voxels",
    "qc_pass",
    "qc_reason",
    "geom_npy",
    "h5_path",
]


@dataclass(frozen=True)
class CandidateSpec:
    strategy: str
    geom_seed: int | None
    mineral_seed: int
    candidate_id: str


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def pore_mask_from_geom(geom_npy: Path) -> np.ndarray:
    """Return boolean pore mask in PFLOTRAN/Stokes flow frame (Ny, Nx); X = flow."""
    img = load_geometry_npy(geom_npy)
    return np.asarray(img.T > 0.5, dtype=bool)


def geometry_qc(pore_mask: np.ndarray) -> tuple[bool, str, float, float, bool]:
    """Check porosity, inlet connectivity, and left-to-right through-connection."""
    phi = float(np.mean(pore_mask))
    connected = connected_pore_from_inlet(pore_mask, inlet=INLET)
    lr_connected = left_right_pore_connected(pore_mask)
    pore_count = int(np.count_nonzero(pore_mask))
    connected_count = int(np.count_nonzero(connected))
    if pore_count == 0:
        return False, "no_pore_voxels", phi, 0.0, False
    connected_frac = connected_count / pore_count
    if connected_count == 0:
        return False, "no_inlet_connectivity", phi, connected_frac, lr_connected
    if not lr_connected:
        return False, "no_left_right_connection", phi, connected_frac, lr_connected
    if phi < PHI_MIN or phi > PHI_MAX:
        return False, f"phi_out_of_range_{phi:.4f}", phi, connected_frac, lr_connected
    return True, "ok", phi, connected_frac, lr_connected


def generate_bean_bucket_geom(geom_seed: int, out_dir: Path) -> Path | None:
    """Generate and normalize one bean-bucket geometry; return None if QC fails."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_npy = out_dir / "network.npy"
    if not raw_npy.is_file():
        subprocess.run(
            [
                sys.executable,
                str(SRC_DIR / "bean_bucket_generator.py"),
                "--height",
                "400",
                "--width",
                "100",
                "--seed",
                str(geom_seed),
                "--out",
                str(raw_npy),
            ],
            cwd=str(REPO_ROOT),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    geom_npy = out_dir / "network_correct.npy"
    if not geom_npy.is_file():
        normalize_design_to_solver(raw_npy, out_dir)
        geom_npy = out_dir / f"{raw_npy.stem}_correct.npy"
    pore_mask = pore_mask_from_geom(geom_npy)
    passed, reason, _, _, _ = geometry_qc(pore_mask)
    if not passed:
        log(f"geom_seed={geom_seed} rejected: {reason}")
        return None
    return geom_npy


def compute_candidate_metrics(geom_npy: Path, h5_path: Path) -> dict[str, float | int]:
    """Compute accessibility metrics for one candidate."""
    img = load_geometry_npy(geom_npy)
    pore_mask = np.asarray(img.T > 0.5, dtype=bool)
    mineral_mask = np.asarray(
        load_mask_from_h5(
            h5_path,
            dataset="Chrysocolla2_VF",
            threshold=0.5,
            shape_hint=img.shape,
        ).T,
        dtype=bool,
    )
    dist_map_m, connected_pore = mineral_access_distance_map(
        pore_mask=pore_mask,
        mineral_mask=mineral_mask,
        dx=DX,
        inlet=INLET,
    )
    stats = compute_accessibility_stats(dist_map_m, mineral_mask)
    pore_count = int(np.count_nonzero(pore_mask))
    connected_frac = float(np.count_nonzero(connected_pore)) / max(pore_count, 1)
    return {
        "l_r_median_um": stats["l_r_median_m"] * 1e6,
        "l_r_p75_um": stats["p75_m"] * 1e6,
        "l_r_p90_um": stats["p90_m"] * 1e6,
        "accessible_cu_frac": accessible_cu_fraction(dist_map_m, mineral_mask, DX),
        "connected_pore_frac": connected_frac,
        "n_mineral_voxels": stats["n_mineral_voxels"],
    }


def pilot_specs() -> list[CandidateSpec]:
    """Build pilot candidate list: 15 per strategy."""
    specs: list[CandidateSpec] = []
    for i in range(PILOT_N):
        geom_seed = GEOM_SEED_BASE + i
        specs.append(
            CandidateSpec(
                strategy="geometry",
                geom_seed=geom_seed,
                mineral_seed=MINERAL_SEED_BASE,
                candidate_id=f"geom_{geom_seed}_min_{MINERAL_SEED_BASE}",
            )
        )
    for i in range(PILOT_N):
        mineral_seed = MINERAL_SEED_BASE + i
        specs.append(
            CandidateSpec(
                strategy="mineral",
                geom_seed=None,
                mineral_seed=mineral_seed,
                candidate_id=f"fixed_geom_min_{mineral_seed}",
            )
        )
    geom_seeds = [GEOM_SEED_BASE + 10 * i for i in range(5)]
    mineral_seeds = [MINERAL_SEED_BASE + 100 * i for i in range(3)]
    for geom_seed in geom_seeds:
        for mineral_seed in mineral_seeds:
            specs.append(
                CandidateSpec(
                    strategy="combined",
                    geom_seed=geom_seed,
                    mineral_seed=mineral_seed,
                    candidate_id=f"geom_{geom_seed}_min_{mineral_seed}",
                )
            )
    return specs


def scale_specs(strategy: str) -> list[CandidateSpec]:
    """Build scaled candidate list for one strategy."""
    if strategy == "geometry":
        return [
            CandidateSpec(
                strategy="geometry",
                geom_seed=GEOM_SEED_BASE + i,
                mineral_seed=MINERAL_SEED_BASE,
                candidate_id=f"geom_{GEOM_SEED_BASE + i}_min_{MINERAL_SEED_BASE}",
            )
            for i in range(SCALE_N_GEOM)
        ]
    if strategy == "mineral":
        return [
            CandidateSpec(
                strategy="mineral",
                geom_seed=None,
                mineral_seed=MINERAL_SEED_BASE + i,
                candidate_id=f"fixed_geom_min_{MINERAL_SEED_BASE + i}",
            )
            for i in range(SCALE_N_MINERAL)
        ]
    if strategy == "combined":
        specs: list[CandidateSpec] = []
        geom_seeds = [GEOM_SEED_BASE + 5 * i for i in range(10)]
        mineral_seeds = [MINERAL_SEED_BASE + 50 * i for i in range(3)]
        for geom_seed in geom_seeds:
            for mineral_seed in mineral_seeds:
                specs.append(
                    CandidateSpec(
                        strategy="combined",
                        geom_seed=geom_seed,
                        mineral_seed=mineral_seed,
                        candidate_id=f"geom_{geom_seed}_min_{mineral_seed}",
                    )
                )
        return specs
    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_candidate(spec: CandidateSpec, work_root: Path) -> dict:
    """Generate geometry + mineral, compute metrics, return one CSV row."""
    cand_dir = work_root / spec.candidate_id
    cand_dir.mkdir(parents=True, exist_ok=True)

    if spec.geom_seed is None:
        geom_npy = FIXED_GEOM
        if not geom_npy.is_file():
            raise FileNotFoundError(f"Fixed geometry missing: {geom_npy}")
        pore_mask = pore_mask_from_geom(geom_npy)
        qc_pass, qc_reason, phi, connected_frac, lr_connected = geometry_qc(pore_mask)
    else:
        geom_dir = cand_dir / "geometry"
        geom_npy = generate_bean_bucket_geom(spec.geom_seed, geom_dir)
        if geom_npy is None:
            return {
                "candidate_id": spec.candidate_id,
                "strategy": spec.strategy,
                "geom_seed": spec.geom_seed,
                "mineral_seed": spec.mineral_seed,
                "phi": "",
                "l_r_median_um": "",
                "l_r_p75_um": "",
                "l_r_p90_um": "",
                "accessible_cu_frac": "",
                "connected_pore_frac": "",
                "left_right_connected": "",
                "n_mineral_voxels": "",
                "qc_pass": False,
                "qc_reason": "geom_qc_failed",
                "geom_npy": "",
                "h5_path": "",
            }
        pore_mask = pore_mask_from_geom(geom_npy)
        qc_pass, qc_reason, phi, connected_frac, lr_connected = geometry_qc(pore_mask)

    h5_path = cand_dir / "chrysocolla_real000.h5"
    if not h5_path.is_file():
        build_one_mineral_realization(
            geom_npy,
            cand_dir,
            mineral_seed=spec.mineral_seed,
            chrysocolla_solid_fraction=CHRYSOCOLLA_SOLID_FRACTION,
        )

    if qc_pass:
        metrics = compute_candidate_metrics(geom_npy, h5_path)
    else:
        metrics = {
            "l_r_median_um": "",
            "l_r_p75_um": "",
            "l_r_p90_um": "",
            "accessible_cu_frac": "",
            "connected_pore_frac": connected_frac,
            "n_mineral_voxels": "",
        }

    return {
        "candidate_id": spec.candidate_id,
        "strategy": spec.strategy,
        "geom_seed": spec.geom_seed if spec.geom_seed is not None else "",
        "mineral_seed": spec.mineral_seed,
        "phi": phi,
        "qc_pass": qc_pass,
        "qc_reason": qc_reason,
        "left_right_connected": lr_connected,
        "geom_npy": str(geom_npy),
        "h5_path": str(h5_path),
        **metrics,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_to_passed(rows: list[dict]) -> list[dict]:
    return [r for r in rows if str(r.get("qc_pass", "")).lower() in {"true", "1"}]


def strategy_score(rows: list[dict]) -> dict[str, float]:
    """Score each strategy: l_r span ratio penalized by phi spread."""
    scores: dict[str, float] = {}
    for strategy in STRATEGY_LABELS:
        passed = [r for r in rows_to_passed(rows) if r["strategy"] == strategy]
        if len(passed) < 2:
            scores[strategy] = 0.0
            continue
        l_r = np.array([float(r["l_r_median_um"]) for r in passed])
        phi = np.array([float(r["phi"]) for r in passed])
        span = float(l_r.max() / max(l_r.min(), 1e-12))
        phi_penalty = 1.0 / (1.0 + 10.0 * float(phi.std()))
        scores[strategy] = span * phi_penalty
    return scores


def pick_best_strategy(rows: list[dict]) -> str:
    scores = strategy_score(rows)
    return max(scores, key=scores.get)


def run_screen(specs: list[CandidateSpec], work_root: Path, csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    for i, spec in enumerate(specs):
        log(f"[{i + 1}/{len(specs)}] {spec.candidate_id}")
        rows.append(evaluate_candidate(spec, work_root))
    write_csv(csv_path, rows)
    log(f"Wrote {csv_path}")
    return rows


def plot_pilot_compare(rows: list[dict], out_png: Path) -> None:
    """Four-panel comparison of pilot strategies."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    passed = rows_to_passed(rows)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))

    ax = axes[0, 0]
    for strategy, label in STRATEGY_LABELS.items():
        vals = sorted(float(r["l_r_median_um"]) for r in passed if r["strategy"] == strategy)
        if not vals:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, marker="o", ms=3, label=label)
    ax.set_xlabel(r"$l_r$ median [µm]")
    ax.set_ylabel("CDF")
    ax.set_title("(a) l_r CDF by strategy", loc="left")
    ax.legend(frameon=True, fontsize=8)

    ax = axes[0, 1]
    for strategy, label in STRATEGY_LABELS.items():
        sub = [r for r in passed if r["strategy"] == strategy]
        if not sub:
            continue
        ax.scatter(
            [float(r["phi"]) for r in sub],
            [float(r["l_r_median_um"]) for r in sub],
            s=28,
            alpha=0.85,
            label=label,
        )
    ax.set_xlabel("porosity φ [-]")
    ax.set_ylabel(r"$l_r$ median [µm]")
    ax.set_title("(b) l_r vs φ", loc="left")
    ax.legend(frameon=True, fontsize=8)

    ax = axes[1, 0]
    for strategy, label in STRATEGY_LABELS.items():
        sub = [r for r in passed if r["strategy"] == strategy]
        if not sub:
            continue
        ax.scatter(
            [float(r["l_r_median_um"]) for r in sub],
            [float(r["accessible_cu_frac"]) for r in sub],
            s=28,
            alpha=0.85,
            label=label,
        )
    ax.set_xlabel(r"$l_r$ median [µm]")
    ax.set_ylabel("accessible Cu fraction [-]")
    ax.set_title("(c) accessible Cu vs l_r", loc="left")

    ax = axes[1, 1]
    ax.axis("off")
    scores = strategy_score(rows)
    lines = ["Strategy summary (QC passed / total):"]
    for strategy, label in STRATEGY_LABELS.items():
        total = sum(1 for r in rows if r["strategy"] == strategy)
        n_pass = sum(1 for r in passed if r["strategy"] == strategy)
        sub = [r for r in passed if r["strategy"] == strategy]
        if sub:
            l_r = [float(r["l_r_median_um"]) for r in sub]
            phi = [float(r["phi"]) for r in sub]
            span = max(l_r) / max(min(l_r), 1e-12)
            lines.append(
                f"{label}: {n_pass}/{total}, l_r span={span:.2f}×, "
                f"φ={min(phi):.3f}–{max(phi):.3f}, score={scores[strategy]:.3f}"
            )
        else:
            lines.append(f"{label}: {n_pass}/{total}, score=0")
    best = pick_best_strategy(rows)
    lines.append(f"Recommended: {STRATEGY_LABELS[best]} ({best})")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")
    ax.set_title("(d) Pilot summary", loc="left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {out_png}")


def plot_scale_screen(rows: list[dict], selected_ids: set[str], out_png: Path) -> None:
    """Histogram + scatter for scaled screening with selected points."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    passed = rows_to_passed(rows)
    l_r = np.array([float(r["l_r_median_um"]) for r in passed])
    phi = np.array([float(r["phi"]) for r in passed])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    ax = axes[0]
    ax.hist(l_r, bins=15, color="#4C72B0", edgecolor="white", alpha=0.9)
    for cid in selected_ids:
        sub = next(r for r in passed if r["candidate_id"] == cid)
        ax.axvline(float(sub["l_r_median_um"]), color="crimson", ls="--", lw=1.2)
    ax.set_xlabel(r"$l_r$ median [µm]")
    ax.set_ylabel("count")
    ax.text(0.03, 0.97, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    ax = axes[1]
    colors = ["crimson" if r["candidate_id"] in selected_ids else "#4C72B0" for r in passed]
    ax.scatter(phi, l_r, c=colors, s=36, alpha=0.9, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("porosity φ [-]")
    ax.set_ylabel(r"$l_r$ median [µm]")
    ax.text(0.03, 0.97, "(b)", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {out_png}")


def _pick_nearest_l_r(
    passed: list[dict],
    target_l_r: float,
    exclude_ids: set[str],
) -> dict | None:
    """Return QC-passed row with l_r closest to target, skipping exclude_ids."""
    candidates = [r for r in passed if r["candidate_id"] not in exclude_ids]
    if not candidates:
        return None
    return min(candidates, key=lambda r: abs(float(r["l_r_median_um"]) - target_l_r))


def select_candidates(rows: list[dict], n_target: int = SELECT_N_TARGET) -> list[dict]:
    """Pick ensemble spanning global l_r extremes, quantile anchors, and phi strata.

    Priority:
      1) global min and max l_r (forced extremes)
      2) global l_r quantile anchors
      3) one median-l_r pick per phi tertile
      4) fill remaining slots by maximizing l_r spacing (farthest-point)
    """
    passed = rows_to_passed(rows)
    if not passed:
        return []

    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    def add(row: dict | None) -> None:
        if row is None:
            return
        cid = row["candidate_id"]
        if cid not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(cid)

    l_r_vals = [float(r["l_r_median_um"]) for r in passed]

    # 1) Global extremes (always, when n_target >= 2)
    add(min(passed, key=lambda r: float(r["l_r_median_um"])))
    add(max(passed, key=lambda r: float(r["l_r_median_um"])))

    # 2) Global l_r quantile anchors (interior; endpoints covered above)
    n_quantile_anchors = min(9, max(0, n_target - 2))
    if n_quantile_anchors > 0:
        quantiles = np.linspace(0.0, 1.0, n_quantile_anchors + 2)[1:-1]
        for q in quantiles:
            add(_pick_nearest_l_r(passed, float(np.quantile(l_r_vals, q)), chosen_ids))

    # 3) Phi tertiles: one l_r-median pick per bin
    phi = np.array([float(r["phi"]) for r in passed])
    phi_bins = np.quantile(phi, [0.0, 1 / 3, 2 / 3, 1.0])
    phi_bins[0] -= 1e-6
    phi_bins[-1] += 1e-6
    for b in range(3):
        mask = (phi >= phi_bins[b]) & (phi < phi_bins[b + 1])
        if not np.any(mask):
            continue
        bin_rows = [passed[i] for i in np.where(mask)[0]]
        med_target = float(np.median([float(r["l_r_median_um"]) for r in bin_rows]))
        add(_pick_nearest_l_r(bin_rows, med_target, chosen_ids))

    # 4) Farthest-point fill in l_r to reach n_target
    while len(chosen) < n_target:
        best_row: dict | None = None
        best_gap = -1.0
        for row in passed:
            if row["candidate_id"] in chosen_ids:
                continue
            lr = float(row["l_r_median_um"])
            gap = min(abs(lr - float(c["l_r_median_um"])) for c in chosen)
            if gap > best_gap:
                best_gap = gap
                best_row = row
        if best_row is None:
            break
        add(best_row)

    return chosen[:n_target]


def promote_selected(selected: list[dict], manifest_path: Path) -> None:
    """Copy selected HDF5 + geometry sidecars into chrysocolla_realizations."""
    REALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx, row in enumerate(selected):
        src_h5 = Path(row["h5_path"])
        src_geom = Path(row["geom_npy"])
        dst_h5 = REALIZATIONS_DIR / f"chrysocolla_real{idx:03d}.h5"
        shutil.copy2(src_h5, dst_h5)
        dst_geom = REALIZATIONS_DIR / f"chrysocolla_real{idx:03d}_geom.npy"
        if src_geom.is_file():
            shutil.copy2(src_geom, dst_geom)
        entries.append(
            {
                "realization_index": idx,
                "candidate_id": row["candidate_id"],
                "strategy": row["strategy"],
                "geom_seed": row["geom_seed"],
                "mineral_seed": int(row["mineral_seed"]),
                "phi": float(row["phi"]),
                "l_r_median_um": float(row["l_r_median_um"]),
                "h5_path": str(dst_h5),
                "geom_npy": str(dst_geom) if dst_geom.is_file() else str(src_geom),
            }
        )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "screen_l_r_ensemble.py",
        "dx_m": DX,
        "chrysocolla_solid_fraction": CHRYSOCOLLA_SOLID_FRACTION,
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"Wrote {manifest_path}")


def cmd_pilot(_: argparse.Namespace) -> None:
    specs = pilot_specs()
    rows = run_screen(specs, SCREEN_ROOT / "pilot", SCREEN_ROOT / "pilot" / "screening_results.csv")
    plot_pilot_compare(rows, SCREEN_ROOT / "pilot_compare.png")
    best = pick_best_strategy(rows)
    (SCREEN_ROOT / "best_strategy.json").write_text(
        json.dumps({"strategy": best, "scores": strategy_score(rows)}, indent=2) + "\n"
    )
    log(f"Best strategy: {best}")


def cmd_scale(args: argparse.Namespace) -> None:
    strategy = args.strategy
    if strategy is None:
        best_path = SCREEN_ROOT / "best_strategy.json"
        if not best_path.is_file():
            raise FileNotFoundError("Run pilot first or pass --strategy")
        strategy = json.loads(best_path.read_text())["strategy"]
    specs = scale_specs(strategy)
    work_root = SCREEN_ROOT / f"scale_{strategy}"
    csv_path = work_root / "screening_results.csv"
    rows = run_screen(specs, work_root, csv_path)
    selected = select_candidates(rows)
    selected_ids = {r["candidate_id"] for r in selected}
    plot_scale_screen(rows, selected_ids, work_root / "l_r_screening.png")
    (work_root / "selected_preview.json").write_text(
        json.dumps([{k: r[k] for k in CSV_FIELDS if k in r} for r in selected], indent=2) + "\n"
    )


def stage1_screen_specs() -> list[CandidateSpec]:
    """Stage 1: screen STAGE1_SCREEN_N geometry seeds from GEOM_SEED_BASE."""
    return [
        CandidateSpec(
            strategy="geometry",
            geom_seed=GEOM_SEED_BASE + i,
            mineral_seed=MINERAL_SEED_BASE,
            candidate_id=f"geom_{GEOM_SEED_BASE + i}_min_{MINERAL_SEED_BASE}",
        )
        for i in range(STAGE1_SCREEN_N)
    ]


def cmd_stage1_screen(_: argparse.Namespace) -> None:
    specs = stage1_screen_specs()
    work_root = SCREEN_ROOT / "stage1_scale_geometry"
    csv_path = work_root / "screening_results.csv"
    rows = run_screen(specs, work_root, csv_path)
    passed = rows_to_passed(rows)
    if passed:
        l_r = [float(r["l_r_median_um"]) for r in passed]
        print(f"Stage1 screen: {len(passed)}/{len(rows)} passed L-R QC; l_r {min(l_r):.1f}–{max(l_r):.1f} µm")
    plot_scale_screen(rows, set(), work_root / "l_r_screening_preview.png")


def cmd_stage1_select(args: argparse.Namespace) -> None:
    csv_path = SCREEN_ROOT / "stage1_scale_geometry" / "screening_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Run stage1-screen first: {csv_path}")
    rows = read_csv(csv_path)
    selected = select_candidates(rows, n_target=args.n)
    out_json = SCREEN_ROOT / "stage1_selected.json"
    out_json.write_text(json.dumps(selected, indent=2) + "\n")
    promote_selected(selected, STAGE1_MANIFEST)
    plot_scale_screen(rows, {r["candidate_id"] for r in selected}, SCREEN_ROOT / "stage1_scale_geometry" / "l_r_screening.png")
    log(f"Stage 1: selected {len(selected)} -> {STAGE1_MANIFEST}")


def cmd_select(args: argparse.Namespace) -> None:
    strategy = args.strategy
    if strategy is None:
        best_path = SCREEN_ROOT / "best_strategy.json"
        strategy = json.loads(best_path.read_text())["strategy"] if best_path.is_file() else "geometry"
    csv_path = SCREEN_ROOT / f"scale_{strategy}" / "screening_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing scale results: {csv_path}")
    rows = read_csv(csv_path)
    selected = select_candidates(rows, n_target=args.n)
    out_json = SCREEN_ROOT / "selected_realizations.json"
    out_json.write_text(json.dumps(selected, indent=2) + "\n")
    promote_selected(selected, REALIZATIONS_DIR / "ensemble_manifest.json")
    plot_scale_screen(rows, {r["candidate_id"] for r in selected}, SCREEN_ROOT / f"scale_{strategy}" / "l_r_screening.png")
    log(f"Selected {len(selected)} realizations -> {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen l_r ensemble candidates (no PFLOTRAN).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pilot", help="Run A/B/C pilot screens (N=15 each).")

    p_scale = sub.add_parser("scale", help="Scale winning strategy.")
    p_scale.add_argument("--strategy", choices=list(STRATEGY_LABELS), default=None)

    p_select = sub.add_parser("select", help="Select and promote realizations for Phase 2.")
    p_select.add_argument("--strategy", choices=list(STRATEGY_LABELS), default=None)
    p_select.add_argument("--n", type=int, default=SELECT_N_TARGET)

    sub.add_parser("stage1-screen", help=f"Screen {STAGE1_SCREEN_N} geometry seeds (L-R QC).")

    p_s1 = sub.add_parser("stage1-select", help="Select N Stage-1 transport realizations.")
    p_s1.add_argument("--n", type=int, default=STAGE1_SELECT_N)

    args = parser.parse_args()
    if args.command == "pilot":
        cmd_pilot(args)
    elif args.command == "scale":
        cmd_scale(args)
    elif args.command == "select":
        cmd_select(args)
    elif args.command == "stage1-screen":
        cmd_stage1_screen(args)
    elif args.command == "stage1-select":
        cmd_stage1_select(args)


if __name__ == "__main__":
    main()