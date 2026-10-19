"""Paper figures from accessibility_ensemble/ (no Chip_Designs paths).

Builds metric tables under analysis/tables/ and writes recovery-dependent PNGs to
analysis/solid_phase/ (domain mobilization) or analysis/outlet/ (net outlet collection).
Shared geometry and outlet-BTC figures go to analysis/shared/.

Example:
    python plot_accessibility_ensemble.py tables
    python plot_accessibility_ensemble.py --metric outlet paper
    python plot_accessibility_ensemble.py all-metrics --refresh
    python plot_accessibility_ensemble.py geom-panels
    python plot_accessibility_ensemble.py outlet-btc

    paper (--metric solid_phase|outlet): manuscript figure set (violin, recovery-lr,
    collapse, matched-theta, pe-da).
    all-metrics: tables + shared figures + full figure set for both recovery metrics.
    factorial-violin writes factorial_recovery_violin_hue.png (R vs dP, hue = k_r).
    collapse-compare writes tables/collapse_compare_<metric>.csv; LaTeX SI table for solid_phase.
    geom-panels writes shared/geom_panels_ab_grid.png (220 dpi), _600dpi.png, and .svg (Fig. S2).
    porosity-panels writes shared/porosity_init_final_four_geoms.png (SI: initial/final porosity).
    outlet-btc writes shared/factorial_outlet_cu_btc_grid.png.
    combined-figures writes dual-metric manuscript panels under analysis/combined/.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from compute_reactive_accessibility_length import connected_pore_from_inlet, load_mask_from_h5
from ensemble_transport_lib import (
    darcy_mean_ux_connected_from_h5,
    velocity_from_theta_m_s,
)
from paper_dataset_lib import (
    ANALYSIS_COMBINED,
    ANALYSIS_ROOT,
    ANALYSIS_SHARED,
    ANALYSIS_TABLES,
    DP_VALUES,
    FACTORIAL_ROOT,
    MATCHED_THETA_ROOT,
    RATE_VALUES,
    REALIZATIONS_MANIFEST,
    RECOVERY_METRICS,
    SCREENING_DIR,
    FactorialCase,
    case_h5_complete,
    load_factorial_cases,
    load_realization_entries,
    matched_theta_case_dir,
    realization_dir,
    resolve_k_pore_npz,
    tag_dp,
    tag_rate,
)
from plot_figure_labels import (
    LBL_CU_RECOVERY_OUTLET,
    LBL_CU_RECOVERY_SOLID,
    LBL_DA_R,
    LBL_DELTA_P,
    LBL_K_R,
    LBL_LR,
    LBL_LR_MEDIAN,
    LBL_PE_R,
    LBL_THETA_48H,
    LBL_THETA_FCU,
    MANUSCRIPT_AXIS_FS,
    MANUSCRIPT_LEGEND_FS,
    MANUSCRIPT_PANEL_FS,
    MANUSCRIPT_TICK_FS,
    add_panel_label,
    anchor_title,
    dp_rate_title,
    format_k_r_latex,
    recovery_axis_label,
)
from plot_accessibility_mineral_dissolution import (
    chrysocolla_inventory_series,
    porosity_snapshots,
)
from plot_recovery_metrics import (
    cumulative_metrics,
    initial_cu_inventory,
    outlet_inlet_flux_metrics,
    parse_time_groups,
    pick_field,
)
from run_dp_rate_pilot_sweep import DX, DZ, FINAL_TIME_S, LX, LY, LZ, PULSE_DURATION_S, load_k_pore

from _repo import REPO_ROOT, SRC_DIR

HERE = REPO_ROOT
V_BULK = LX * LY * LZ
PULSE_S = float(PULSE_DURATION_S)

FACTORIAL_CSV = ANALYSIS_TABLES / "factorial_metrics.csv"
MATCHED_CSV = ANALYSIS_TABLES / "matched_theta_metrics.csv"
FACTORIAL_ENRICHED_CSV = ANALYSIS_TABLES / "factorial_enriched.csv"
SI_TEX = REPO_ROOT / "manuscript/draft/Supporting_information.tex"
COLLAPSE_COMPARE_BEGIN = "% BEGIN collapse_compare_table"
COLLAPSE_COMPARE_END = "% END collapse_compare_table"

_ACTIVE_METRIC = "solid_phase"


def set_active_metric(metric: str) -> None:
    """Select solid-phase (domain) or outlet recovery for figure outputs."""
    global _ACTIVE_METRIC
    if metric not in RECOVERY_METRICS:
        raise ValueError(f"metric must be one of {RECOVERY_METRICS}, got {metric!r}")
    _ACTIVE_METRIC = metric


def figures_dir(metric: str | None = None) -> Path:
    return ANALYSIS_ROOT / (metric or _ACTIVE_METRIC)


def collapse_compare_csv(metric: str | None = None) -> Path:
    return ANALYSIS_TABLES / f"collapse_compare_{metric or _ACTIVE_METRIC}.csv"


def recovery_label() -> str:
    return recovery_axis_label(_ACTIVE_METRIC)


def apply_recovery_metric(rows: list[dict], metric: str | None = None) -> list[dict]:
    """Map solid_recovery_pct or outlet_recovery_pct onto recovery_pct for plotting."""
    col = "outlet_recovery_pct" if (metric or _ACTIVE_METRIC) == "outlet" else "solid_recovery_pct"
    out: list[dict] = []
    for row in rows:
        nr = dict(row)
        raw = nr.get(col, nr.get("recovery_pct", np.nan))
        try:
            nr["recovery_pct"] = float(raw)
        except (TypeError, ValueError):
            nr["recovery_pct"] = np.nan
        out.append(nr)
    return out


def ensure_analysis_dirs() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_TABLES.mkdir(parents=True, exist_ok=True)
    ANALYSIS_SHARED.mkdir(parents=True, exist_ok=True)
    ANALYSIS_COMBINED.mkdir(parents=True, exist_ok=True)
    for metric in RECOVERY_METRICS:
        (ANALYSIS_ROOT / metric).mkdir(parents=True, exist_ok=True)

PORE_DIFFUSIVITY_M2S = 1.0e-9
SSA_CHRYSOCOLLA_M2_M3 = 100.0 * 1.0e4
H_INLET_MOL_M3 = 0.1 * 1.0e3  # inlet acid [H+] at pH 1; reference concentration for Da_r
FLUID_VISCOSITY_PA_S = 1.0e-3
COLLAPSE_LAMBDA_M = 60.0e-6
MATCHED_THETA_K_REF = 1e-9
_STOKES_UX_PAPER_CACHE: dict[int, tuple[float, float]] = {}

def log(msg: str) -> None:
    print(msg, flush=True)


def save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int = 180,
    extra_formats: tuple[str, ...] = (),
    extra_png_dpis: tuple[int, ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    stem = path.with_suffix("")
    for fmt in extra_formats:
        extra_path = stem.with_suffix(f".{fmt}")
        fig.savefig(extra_path, bbox_inches="tight", pad_inches=0.12)
        log(f"Saved {extra_path}")
    for extra_dpi in extra_png_dpis:
        extra_png = stem.with_name(f"{stem.name}_{extra_dpi}dpi.png")
        fig.savefig(extra_png, dpi=extra_dpi, bbox_inches="tight", pad_inches=0.12)
        log(f"Saved {extra_png}")
    plt.close(fig)
    log(f"Saved {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        log(f"SKIP empty table: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log(f"Saved {path} ({len(rows)} rows)")


def metrics_from_h5(h5_path: Path, *, geom_seed: int) -> dict:
    row = {
        "solid_recovery_pct": np.nan,
        "outlet_recovery_pct": np.nan,
        "theta_48h": np.nan,
        "k_pore": np.nan,
    }
    ref = resolve_k_pore_npz(geom_seed)
    if ref is not None:
        row["k_pore"] = load_k_pore(ref)
    if not case_h5_complete(h5_path):
        return row
    geom_path = h5_path.parent / "geometry_accessibility.h5"
    with h5py.File(h5_path, "r") as h5:
        time_entries = parse_time_groups(h5)
        g0 = h5[time_entries[0][1]]
        cu_key = pick_field(g0, "Total_Cu++", required=True)
        h_key = pick_field(g0, "Free_H+", required=True)
        vf_key = pick_field(g0, "Chrysocolla2_VF", required=True)
        series = chrysocolla_inventory_series(h5, time_entries, lx=LX, ly=LY, lz=LZ)
        flux = outlet_inlet_flux_metrics(h5, time_entries, cu_key, h_key, LY, LZ)
        t_s = flux["t_s"]
        q = flux["Qout"]
        dt = np.diff(t_s, prepend=0.0)
        pv = np.cumsum(q * dt) / V_BULK
        ip = int(np.argmin(np.abs(t_s - PULSE_S)))
        row["solid_recovery_pct"] = float(series["total_recovery_pct"][-1])
        row["theta_48h"] = float(pv[ip])
        if geom_path.is_file():
            cu_init, pore_vol = initial_cu_inventory(
                h5,
                time_entries[0][1],
                vf_key,
                str(geom_path),
                LX,
                LY,
                LZ,
            )
            cum = cumulative_metrics(flux, cu_init, pore_vol)
            row["outlet_recovery_pct"] = float(cum["frac_rec"][-1]) * 100.0
    return row


def build_factorial_table(*, refresh: bool = False) -> list[dict]:
    if FACTORIAL_CSV.is_file() and not refresh:
        with FACTORIAL_CSV.open() as f:
            return list(csv.DictReader(f))

    rows: list[dict] = []
    for case in load_factorial_cases():
        h5 = case.case_dir() / "mineral_dissolution.h5"
        base = metrics_from_h5(h5, geom_seed=case.geom_seed)
        rows.append(
            {
                "campaign": "factorial",
                "case_index": case.index,
                "geom_seed": case.geom_seed,
                "candidate_id": case.candidate_id,
                "phi": case.phi,
                "l_r_median_um": case.l_r_median_um,
                "dP_Pa": case.dP_Pa,
                "rate_mol_cm2_s": case.rate_mol_cm2_s,
                "complete": int(case.has_output()),
                "case_dir": str(case.case_dir()),
                **base,
            }
        )
    write_csv(FACTORIAL_CSV, rows)
    return rows


def build_matched_theta_table(*, refresh: bool = False) -> list[dict]:
    if MATCHED_CSV.is_file() and not refresh:
        with MATCHED_CSV.open() as f:
            return list(csv.DictReader(f))

    rows: list[dict] = []
    for manifest in sorted(MATCHED_THETA_ROOT.glob("manifest_*.json")):
        branch = manifest.stem.removeprefix("manifest_")
        payload = json.loads(manifest.read_text())
        entries = payload.get("entries", [])
        if not entries:
            log(f"SKIP empty {manifest}")
            continue
        for entry in entries:
            geom_seed = int(entry["geom_seed"])
            case_dir = matched_theta_case_dir(geom_seed, branch)
            h5 = case_dir / "mineral_dissolution.h5"
            base = metrics_from_h5(h5, geom_seed=geom_seed)
            rows.append(
                {
                    "campaign": f"matched_theta_{branch}",
                    "branch": branch,
                    "geom_seed": geom_seed,
                    "candidate_id": entry.get("candidate_id", ""),
                    "phi": float(entry["phi"]),
                    "l_r_median_um": float(entry["l_r_median_um"]),
                    "dP_Pa": float(entry["dP_Pa"]),
                    "rate_mol_cm2_s": float(entry["rate_mol_cm2_s"]),
                    "theta_target_48h": float(entry.get("theta_target_48h", np.nan)),
                    "complete": int(case_h5_complete(h5)),
                    "case_dir": str(case_dir),
                    **base,
                }
            )
    write_csv(MATCHED_CSV, rows)
    return rows


def load_accessible_cu_lookup() -> dict[str, float]:
    """geom_seed / candidate_id -> accessible Cu fraction from ensemble screening."""
    path = SCREENING_DIR / "stage1_selected.json"
    if not path.is_file():
        return {}
    lookup: dict[str, float] = {}
    for row in json.loads(path.read_text()):
        frac = float(row["accessible_cu_frac"])
        lookup[str(row["candidate_id"])] = frac
        lookup[str(row["geom_seed"])] = frac
    return lookup


def _geom_npy_path(geom_seed: int) -> Path:
    return realization_dir(geom_seed) / "pore_solid.npy"


def _mineral_h5_path(geom_seed: int) -> Path:
    return realization_dir(geom_seed) / "mineral.h5"


def load_realization_masks(geom_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pore image, chrysocolla mask, and inlet-connected pore mask (flow frame)."""
    pore = np.load(_geom_npy_path(geom_seed))
    pore_mask = np.asarray(pore > 0.5, dtype=bool)
    mineral_h5 = _mineral_h5_path(geom_seed)
    chrys_mask = load_mask_from_h5(
        mineral_h5,
        dataset="Chrysocolla2_VF",
        shape_hint=tuple(int(x) for x in pore.shape),
    )
    connected = connected_pore_from_inlet(pore_mask, inlet="left")
    return pore, chrys_mask, connected


def default_geom_panel_seeds(n: int = 4) -> list[int]:
    """Pick n realizations evenly spaced in l_r rank (default four rows)."""
    entries = load_realization_entries()
    if not entries:
        raise FileNotFoundError(f"No realization entries in {REALIZATIONS_MANIFEST}")
    ranked = sorted(entries, key=lambda row: float(row["l_r_median_um"]))
    if n >= len(ranked):
        return [int(row["geom_seed"]) for row in ranked]
    idx = np.linspace(0, len(ranked) - 1, n, dtype=int)
    return [int(ranked[int(i)]["geom_seed"]) for i in idx]


def _manifest_lr_um(geom_seed: int) -> float:
    for row in load_realization_entries():
        if int(row["geom_seed"]) == int(geom_seed):
            return float(row["l_r_median_um"])
    return float("nan")


def plot_geom_panels_ab(
    pore: np.ndarray,
    chrys_mask: np.ndarray,
    connected_pore: np.ndarray,
    out_png: Path,
    *,
    row_title: str = "",
    panel_prefix: str = "",
) -> None:
    """Two-panel pore geometry figure: (a) connected pore, (b) chrysocolla inventory."""
    pore_disp = np.where(pore > 0.5, 1.0, 0.75).T
    connected_disp = connected_pore.T.astype(float)
    chrys_disp = np.where(chrys_mask, 1.0, np.nan).T

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True)
    axes[0].imshow(pore_disp, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[0].contour(connected_disp, levels=[0.5], colors="tab:blue", linewidths=0.6)
    axes[0].set_title(f"{panel_prefix}(a) Connected pore space", loc="left")

    axes[1].imshow(np.where(pore > 0.5, 1.0, 0.65).T, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(
        chrys_disp,
        origin="lower",
        cmap=ListedColormap([_GEOM_PANEL_CHRYS_COLOR]),
        vmin=0,
        vmax=1,
        alpha=1.0,
    )
    axes[1].set_title(f"{panel_prefix}(b) Fixed chrysocolla inventory", loc="left")

    for ax in axes:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    if row_title:
        fig.suptitle(row_title, fontsize=10, y=1.02)
    save_figure(fig, out_png, dpi=220)


def _rgba_hex(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _a = rgba
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


# Category colors for Fig. S2 (geom_panels_ab_grid); matched to prior imshow styling.
_GEOM_PANEL_CONNECTED_COLORS = ("#b8b8b8", "#ededed", "#2e6bb8")  # solid, pore, connected water
_GEOM_PANEL_CHRYS_COLOR = "#3d8c40"  # schematic_setup / manuscript palette
_GEOM_PANEL_MINERAL_COLORS = (
    "#a6a6a6",
    "#ffffff",
    _GEOM_PANEL_CHRYS_COLOR,
)  # solid matrix, pore, chrysocolla


def _connected_pore_category(pore: np.ndarray, connected: np.ndarray) -> np.ndarray:
    """Integer category map in display frame: 0 solid, 1 pore, 2 connected water."""
    pore_mask = np.asarray(pore > 0.5, dtype=bool)
    conn_mask = np.asarray(connected, dtype=bool)
    categories = np.zeros(pore.shape, dtype=np.uint8)
    categories[pore_mask & ~conn_mask] = 1
    categories[conn_mask] = 2
    return categories.T


def _chrysocolla_category(pore: np.ndarray, chrys_mask: np.ndarray) -> np.ndarray:
    """Integer category map in display frame: 0 solid, 1 pore, 2 chrysocolla."""
    pore_mask = np.asarray(pore > 0.5, dtype=bool)
    chrys = np.asarray(chrys_mask, dtype=bool)
    categories = np.zeros(pore.shape, dtype=np.uint8)
    categories[pore_mask] = 1
    categories[chrys & ~pore_mask] = 2
    return categories.T


def _plot_category_cell_map(ax: plt.Axes, categories: np.ndarray, colors: tuple[str, ...]) -> None:
    """Render pore maps as merged vector rectangles (no raster imshow)."""
    ny, nx = categories.shape
    patches_by_cat: list[list[Rectangle]] = [[] for _ in colors]
    for row_idx in range(ny):
        row = categories[row_idx]
        col = 0
        while col < nx:
            cat = int(row[col])
            start = col
            col += 1
            while col < nx and int(row[col]) == cat:
                col += 1
            patches_by_cat[cat].append(Rectangle((start, row_idx), col - start, 1))

    for cat, color in enumerate(colors):
        if not patches_by_cat[cat]:
            continue
        collection = PatchCollection(
            patches_by_cat[cat],
            facecolor=color,
            edgecolor="none",
            antialiased=False,
            rasterized=False,
        )
        ax.add_collection(collection)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_aspect("equal", adjustable="box")


def _strip_map_axes(ax: plt.Axes) -> None:
    """Remove ticks and spines for compact pore-map panels."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_geom_panels_grid(geom_seeds: list[int], out_png: Path) -> None:
    """Stacked a-b panels: shared column titles, tight rows, water-filled connected pore."""
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    n = len(geom_seeds)
    fig, axes = plt.subplots(n, 2, figsize=(9.2, 1.75 * n))
    if n == 1:
        axes = np.asarray([axes])

    for row_idx, geom_seed in enumerate(geom_seeds):
        pore, chrys_mask, connected = load_realization_masks(geom_seed)

        ax0, ax1 = axes[row_idx]
        _plot_category_cell_map(
            ax0,
            _connected_pore_category(pore, connected),
            _GEOM_PANEL_CONNECTED_COLORS,
        )
        _plot_category_cell_map(
            ax1,
            _chrysocolla_category(pore, chrys_mask),
            _GEOM_PANEL_MINERAL_COLORS,
        )

        _strip_map_axes(ax0)
        _strip_map_axes(ax1)
        lr_um = _manifest_lr_um(geom_seed)
        row_title = f"({chr(97 + row_idx)}) geom {geom_seed}"
        if np.isfinite(lr_um):
            row_title += f", $l_r$ = {lr_um:.0f} $\\mu$m"
        plt.sca(ax0)
        plt.title(row_title, loc="left", pad=3, fontsize=9)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.03, hspace=0.06, wspace=0.05)
    axis_lbl = r"$x$ (flow, 8 mm) $\times$ $y$ (2 mm)"
    col_titles = (
        f"Connected pore space\n{axis_lbl}",
        f"Fixed chrysocolla inventory\n{axis_lbl}",
    )
    for col_idx, title in enumerate(col_titles):
        pos = axes[0, col_idx].get_position()
        cx = 0.5 * (pos.x0 + pos.x1)
        fig.text(cx, pos.y1 + 0.030, title, ha="center", va="bottom", fontsize=10, linespacing=1.25)

    save_figure(fig, out_png, dpi=220, extra_formats=("svg",), extra_png_dpis=(600,))


def _stokes_u_ref_paper(geom_seed: int) -> tuple[float, float]:
    """Mean |u_x| on inlet-connected pores from paper-dataset Stokes reference."""
    if geom_seed in _STOKES_UX_PAPER_CACHE:
        return _STOKES_UX_PAPER_CACHE[geom_seed]
    npz_path = resolve_k_pore_npz(geom_seed)
    if npz_path is None:
        raise FileNotFoundError(f"no Stokes reference for geom {geom_seed}")
    geom_npy = _geom_npy_path(geom_seed)
    if not geom_npy.is_file():
        raise FileNotFoundError(geom_npy)
    pore = np.load(geom_npy)
    pore_flow = np.asarray(pore.T > 0.5, dtype=bool)
    connected = connected_pore_from_inlet(pore_flow, inlet="left")
    data = np.load(npz_path)
    u = np.asarray(data["u"], dtype=float)
    fluid = np.asarray(data["mask"], dtype=float) > 0.0
    dP_ref = float(data["dP"]) if "dP" in data.files else 1.0
    sel = connected & fluid
    if not np.any(sel):
        raise ValueError(f"no inlet-connected fluid cells for geom {geom_seed}")
    u_ref = float(np.mean(np.abs(u[sel])))
    if u_ref <= 0.0:
        raise ValueError(f"non-positive Stokes u_ref for geom {geom_seed}")
    _STOKES_UX_PAPER_CACHE[geom_seed] = (u_ref, dP_ref)
    return u_ref, dP_ref


def velocity_from_stokes_paper(geom_seed: int, dP_Pa: float) -> float:
    u_ref, dP_ref = _stokes_u_ref_paper(geom_seed)
    return u_ref * float(dP_Pa) / max(dP_ref, 1e-15)


def characteristic_velocity_m_s(k_pore: float, dP_Pa: float) -> float:
    return float(k_pore) * float(dP_Pa) / (FLUID_VISCOSITY_PA_S * LX)


def compute_pe_da_metrics(row: dict, *, cu_lookup: dict[str, float]) -> dict[str, float]:
    """Accessibility-aware Pe_r and Da_r for accessibility_ensemble campaign rows."""
    l_r_m = float(row["l_r_median_um"]) * 1e-6
    geom_seed = int(row["geom_seed"])
    dP_pa = float(row["dP_Pa"])
    theta = float(row.get("theta_48h", np.nan))
    cu_frac = cu_lookup.get(str(row.get("candidate_id", "")), np.nan)
    if not np.isfinite(cu_frac):
        cu_frac = cu_lookup.get(str(geom_seed), np.nan)
    f_use = float(cu_frac) if np.isfinite(cu_frac) else 1.0

    h5_path = Path(str(row["case_dir"])) / "mineral_dissolution.h5"
    geom_npy = _geom_npy_path(geom_seed)
    u_method = "darcy_mean_ux_connected"
    u = np.nan
    if case_h5_complete(h5_path):
        try:
            u = darcy_mean_ux_connected_from_h5(h5_path, geom_npy)
        except (FileNotFoundError, ValueError, OSError):
            u = np.nan
    if not np.isfinite(u) or u <= 0.0:
        u_method = "stokes_mean_ux_connected"
        try:
            u = velocity_from_stokes_paper(geom_seed, dP_pa)
        except (FileNotFoundError, ValueError, OSError):
            u = np.nan
    if not np.isfinite(u) or u <= 0.0:
        if np.isfinite(theta) and theta > 0.0:
            u = velocity_from_theta_m_s(theta)
            u_method = "theta_outlet"
        else:
            k_pore = float(row.get("k_pore", np.nan))
            u = max(characteristic_velocity_m_s(k_pore, dP_pa), 1e-15) if np.isfinite(k_pore) else 1e-15
            u_method = "k_pore_proxy"
    u = max(float(u), 1e-15)
    pe_r = u * l_r_m / PORE_DIFFUSIVITY_M2S
    k_r = float(row["rate_mol_cm2_s"]) * 1e4
    da_r = k_r * SSA_CHRYSOCOLLA_M2_M3 * f_use * l_r_m / (u * H_INLET_MOL_M3)
    return {
        "accessible_cu_frac": f_use,
        "u_ms": u,
        "u_velocity_method": u_method,
        "Pe_r": pe_r,
        "Da_r": da_r,
    }


def enrich_rows(rows: list[dict], *, cu_lookup: dict[str, float]) -> list[dict]:
    """Attach Pe_r, Da_r, and accessible_cu_frac to complete campaign rows."""
    out: list[dict] = []
    for row in rows:
        r = dict(row)
        if int(float(r.get("complete", 0))) != 1:
            out.append(r)
            continue
        try:
            float(r["recovery_pct"])
            float(r["l_r_median_um"])
            float(r["dP_Pa"])
            float(r["rate_mol_cm2_s"])
        except (KeyError, ValueError, TypeError):
            out.append(r)
            continue
        r.update(compute_pe_da_metrics(r, cu_lookup=cu_lookup))
        out.append(r)
    return out


def _complete_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if int(float(r.get("complete", 0))) != 1:
            continue
        if not np.isfinite(float(r.get("recovery_pct", np.nan))):
            continue
        out.append(r)
    return out


def plot_factorial_heatmaps(rows: list[dict]) -> None:
    """Median Cu Recovery and theta across geoms for each (dP, rate) cell."""
    valid = _complete_rows(rows)
    if not valid:
        log("SKIP factorial_heatmaps: no complete rows")
        return

    rec_grid = np.full((len(RATE_VALUES), len(DP_VALUES)), np.nan)
    theta_grid = np.full_like(rec_grid, np.nan)
    n_grid = np.zeros_like(rec_grid, dtype=int)

    for i, rate in enumerate(RATE_VALUES):
        for j, dp in enumerate(DP_VALUES):
            sub = [
                r
                for r in valid
                if abs(float(r["dP_Pa"]) - dp) < 1e-12
                and abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15
            ]
            if not sub:
                continue
            rec = [float(r["recovery_pct"]) for r in sub]
            theta = [float(r["theta_48h"]) for r in sub if np.isfinite(float(r["theta_48h"]))]
            rec_grid[i, j] = float(np.median(rec))
            if theta:
                theta_grid[i, j] = float(np.median(theta))
            n_grid[i, j] = len(sub)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), layout="constrained")
    xlabels = [str(d) for d in DP_VALUES]
    ylabels = [format_k_r_latex(r) for r in RATE_VALUES]

    sns.heatmap(
        rec_grid,
        annot=True,
        fmt=".1f",
        cmap="magma",
        xticklabels=xlabels,
        yticklabels=ylabels,
        ax=axes[0],
        cbar_kws={"label": f"Median {recovery_label()}"},
    )
    axes[0].set_xlabel(LBL_DELTA_P)
    axes[0].set_ylabel(LBL_K_R)
    axes[0].set_title(f"(a) Median {recovery_label()}", loc="left")

    sns.heatmap(
        theta_grid,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=xlabels,
        yticklabels=ylabels,
        ax=axes[1],
        cbar_kws={"label": rf"Median {LBL_THETA_48H}"},
    )
    axes[1].set_xlabel(LBL_DELTA_P)
    axes[1].set_ylabel(LBL_K_R)
    axes[1].set_title(rf"(b) Median {LBL_THETA_48H}", loc="left")

    save_figure(fig, figures_dir() / "factorial_heatmaps.png")


def _factorial_conditions() -> list[tuple[float, float, str]]:
    """Nine (dP, k_r) cells in column order: dP blocks, rates within each block."""
    out: list[tuple[float, float, str]] = []
    for dp in DP_VALUES:
        for rate in RATE_VALUES:
            kr = format_k_r_latex(rate).strip("$")
            out.append((dp, rate, rf"$\Delta p={dp:g}$\n$k_r={kr}$"))
    return out


def _build_geom_condition_matrices(
    rows: list[dict],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Return geoms sorted by l_r, l_r vector, and recovery/theta matrices (n_geom x 9)."""
    geom_lr: dict[int, float] = {}
    for r in rows:
        geom_lr[int(r["geom_seed"])] = float(r["l_r_median_um"])

    complete = {
        (int(r["geom_seed"]), float(r["dP_Pa"]), float(r["rate_mol_cm2_s"])): r
        for r in _complete_rows(rows)
    }

    geoms = sorted(geom_lr.keys(), key=lambda g: geom_lr[g])
    conditions = _factorial_conditions()
    rec = np.full((len(geoms), len(conditions)), np.nan)
    theta = np.full_like(rec, np.nan)

    for i, geom in enumerate(geoms):
        for j, (dp, rate, _) in enumerate(conditions):
            row = complete.get((geom, dp, rate))
            if row is None:
                continue
            rec[i, j] = float(row["recovery_pct"])
            th = float(row.get("theta_48h", np.nan))
            if np.isfinite(th):
                theta[i, j] = th

    lr = np.array([geom_lr[g] for g in geoms], dtype=float)
    return geoms, lr, rec, theta


def plot_factorial_ensemble_heatmaps(rows: list[dict]) -> None:
    """Realization x condition recovery heatmap; rows sorted by l_r, cols = 9 factorial cells."""
    _geoms, lr, rec, _theta = _build_geom_condition_matrices(rows)
    if not np.any(np.isfinite(rec)):
        log("SKIP factorial_ensemble_heatmaps: no complete rows")
        return

    conditions = _factorial_conditions()
    col_labels = [label for _, _, label in conditions]
    lr_col = lr[:, np.newaxis]
    lr_annot = np.array([[f"{v:.0f}"] for v in lr], dtype=object)

    sns.set_theme(style="white", context="paper", font_scale=0.9)
    fig_h = max(5.5, 0.22 * len(lr) + 1.8)
    fig = plt.figure(figsize=(12.0, fig_h))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.14, 1.0, 0.06], wspace=0.06)

    ax_lr = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        lr_col,
        ax=ax_lr,
        cmap="viridis_r",
        annot=lr_annot,
        fmt="",
        cbar=False,
        xticklabels=[LBL_LR],
        yticklabels=False,
        linewidths=0.2,
        linecolor="0.92",
        annot_kws={"fontsize": 7},
    )

    ax_hm = fig.add_subplot(gs[0, 1])
    hm = sns.heatmap(
        rec,
        mask=~np.isfinite(rec),
        ax=ax_hm,
        cmap="magma",
        xticklabels=col_labels,
        yticklabels=False,
        linewidths=0.2,
        linecolor="0.92",
        cbar=False,
    )
    ax_hm.set_xlabel(rf"{LBL_DELTA_P} / {LBL_K_R}")
    ax_hm.set_title(rf"(a) {recovery_label()} by realization (rows sorted by $l_r$)", loc="left")

    ax_cb = fig.add_subplot(gs[0, 2])
    fig.colorbar(hm.collections[0], cax=ax_cb, label=recovery_label())

    save_figure(fig, figures_dir() / "factorial_ensemble_heatmaps.png")


def plot_factorial_recovery_grid(rows: list[dict]) -> None:
    """3x3 grid: recovery vs l_r for each (dP, rate)."""
    valid = _complete_rows(rows)
    if len(valid) < 3:
        log("SKIP factorial_recovery_grid: insufficient data")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 9.5), layout="constrained", sharex=True, sharey=True)
    panel = 0
    for i, rate in enumerate(RATE_VALUES):
        for j, dp in enumerate(DP_VALUES):
            ax = axes[i, j]
            sub = [
                r
                for r in valid
                if abs(float(r["dP_Pa"]) - dp) < 1e-12
                and abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15
            ]
            if sub:
                lr = [float(r["l_r_median_um"]) for r in sub]
                rec = [float(r["recovery_pct"]) for r in sub]
                ax.scatter(lr, rec, s=36, c="#2166ac", edgecolors="0.2", linewidths=0.3)
            ax.set_title(dp_rate_title(dp, rate), loc="left", fontsize=8)
            ax.text(0.03, 0.97, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=9, va="top")
            panel += 1
    fig.supxlabel(LBL_LR_MEDIAN)
    fig.supylabel(recovery_label())
    save_figure(fig, figures_dir() / "factorial_recovery_vs_l_r_grid.png")


def _factorial_cell_recovery(
    rows: list[dict], *, dp: float, rate: float
) -> list[float]:
    """Recovery [%] for one factorial (dP, k_r) cell across all complete realizations."""
    out: list[float] = []
    for r in rows:
        if abs(float(r["dP_Pa"]) - dp) > 1e-12:
            continue
        if abs(float(r["rate_mol_cm2_s"]) - rate) > 1e-15:
            continue
        rec = float(r["recovery_pct"])
        if np.isfinite(rec):
            out.append(rec)
    return out


def _shared_recovery_ylim(values: list[float], *, log_scale: bool = False) -> tuple[float, float]:
    """Shared y-limits with modest padding for factorial recovery panels."""
    arr = np.clip(np.asarray(values, dtype=float), 0.5, None)
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if log_scale:
        log_span = np.log10(hi) - np.log10(lo)
        pad = max(0.06, 0.10 * log_span)
        return 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)
    pad = max(0.8, 0.12 * (hi - lo))
    y0 = max(0.0, np.floor((lo - pad) * 2.0) / 2.0)
    y1 = np.ceil((hi + pad) * 2.0) / 2.0
    return float(y0), float(y1)


def _draw_recovery_violin(
    ax: plt.Axes,
    values: list[float],
    *,
    color,
    width: float = 0.10,
) -> None:
    """Narrow symmetric violin (KDE only) centered in a heatmap-style cell."""
    y = np.clip(np.asarray(values, dtype=float), 0.5, None)
    if y.size < 2:
        return

    iqr = max(float(np.percentile(y, 75) - np.percentile(y, 25)), 0.4)
    yi = np.linspace(y.min() - 0.25 * iqr, y.max() + 0.25 * iqr, 72)
    try:
        from scipy.stats import gaussian_kde

        dens = gaussian_kde(y, bw_method=0.45)(yi)
    except Exception:
        dens = np.ones_like(yi)
    dens = dens / max(float(dens.max()), 1e-9)
    ax.fill_betweenx(yi, -width * dens, width * dens, color=color, alpha=0.92, linewidth=0, zorder=2)
    ax.plot(-width * dens, yi, color="0.15", alpha=0.35, lw=0.6, zorder=3)
    ax.plot(width * dens, yi, color="0.15", alpha=0.35, lw=0.6, zorder=3)


def plot_factorial_recovery_violin_grid(rows: list[dict]) -> None:
    """Compact 3x3 violin grid sized like one factorial_heatmaps panel (a)."""
    valid = _complete_rows(rows)
    if len(valid) < 9:
        log("SKIP factorial_recovery_violin_grid: insufficient data")
        return

    all_rec = [float(r["recovery_pct"]) for r in valid]
    y_lo, y_hi = _shared_recovery_ylim(all_rec, log_scale=True)
    violin_color = "#4c72b0"

    sns.set_theme(style="white", context="paper", font_scale=0.92)
    fig = plt.figure(figsize=(4.5, 4.0))
    gs = fig.add_gridspec(
        3,
        3,
        wspace=0.0,
        hspace=0.0,
        left=0.16,
        right=0.96,
        top=0.88,
        bottom=0.14,
    )
    axes = np.empty((3, 3), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    for j in range(1, 3):
        axes[0, j] = fig.add_subplot(gs[0, j], sharey=axes[0, 0])
    for i in range(1, 3):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, j], sharey=axes[0, 0])

    for i, rate in enumerate(RATE_VALUES):
        for j, dp in enumerate(DP_VALUES):
            ax = axes[i, j]
            recs = _factorial_cell_recovery(valid, dp=dp, rate=rate)
            if len(recs) < 2:
                ax.text(0.5, 0.5, "n<2", ha="center", va="center", transform=ax.transAxes, color="0.45", fontsize=6)
            else:
                _draw_recovery_violin(ax, recs, color=violin_color, width=0.10)

            ax.set_xlim(-0.18, 0.18)
            ax.set_ylim(y_lo, y_hi)
            ax.set_yscale("log")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="y", left=False, labelleft=False, right=False, labelright=False)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("0.75")

            if i == 2:
                ax.set_xticks([0.0])
                ax.set_xticklabels([f"{dp:g}"], fontsize=7)
            if j == 0:
                kr = format_k_r_latex(rate)
                ax.set_ylabel(kr, fontsize=7, rotation=0, ha="right", va="center", labelpad=1)

    ax_ref = axes[0, 2]
    ax_ref.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=6))
    ax_ref.yaxis.set_minor_locator(plt.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax_ref.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_ref.yaxis.tick_right()
    ax_ref.tick_params(axis="y", which="both", labelright=True, labelleft=False, right=True, length=2, pad=1, labelsize=7)
    ax_ref.set_ylabel(recovery_label(), fontsize=7.5, rotation=270, labelpad=10)
    ax_ref.yaxis.set_label_position("right")

    axes[2, 1].set_xlabel(LBL_DELTA_P, fontsize=8)
    fig.text(0.03, 0.5, LBL_K_R, rotation=90, va="center", ha="center", fontsize=8)
    fig.suptitle(f"(a) {recovery_label()}", x=0.16, ha="left", fontsize=9, y=0.97)

    save_figure(fig, figures_dir() / "factorial_recovery_violin_grid.png", dpi=300)


def _factorial_violin_hue_on_ax(
    ax: plt.Axes,
    rows: list[dict],
    *,
    panel_label: str,
    ylabel: str,
    y_lim: tuple[float, float],
    show_legend: bool,
    axis_fs: float = 12.0,
    tick_fs: float = 11.0,
    panel_fs: float = 12.0,
) -> None:
    """Set~A factorial recovery violins on one axis."""
    valid = _complete_rows(rows)
    if len(valid) < 9:
        ax.set_visible(False)
        return

    dp_vals = [float(r["dP_Pa"]) for r in valid]
    rec_vals = [max(float(r["recovery_pct"]), 0.5) for r in valid]
    rate_vals = [float(r["rate_mol_cm2_s"]) for r in valid]
    y_lo, y_hi = y_lim

    palette = ["#4e79a7", "#e15759", "#59a14f"]
    rate_palette = dict(zip(RATE_VALUES, palette))
    edge_palette = ["#2f4f6f", "#8b2f2f", "#2f6b35"]
    rate_labels = {rate: format_k_r_latex(rate) for rate in RATE_VALUES}
    y_tick_candidates = [0.5, 1, 2, 5, 10, 15, 20]

    sns.violinplot(
        x=dp_vals,
        y=rec_vals,
        hue=rate_vals,
        order=list(DP_VALUES),
        hue_order=list(RATE_VALUES),
        palette=palette,
        ax=ax,
        inner=None,
        cut=0,
        density_norm="width",
        linewidth=0.0,
        saturation=1.0,
        dodge=True,
        bw_adjust=0.9,
        legend=False,
        zorder=2,
    )
    for i, body in enumerate(ax.collections[: len(RATE_VALUES) * len(DP_VALUES)]):
        color_i = i % len(RATE_VALUES)
        body.set_facecolor(palette[color_i])
        body.set_edgecolor(edge_palette[color_i])
        body.set_alpha(0.55)
        body.set_linewidth(1.0)
        body.set_zorder(2)

    sns.stripplot(
        x=dp_vals,
        y=rec_vals,
        hue=rate_vals,
        order=list(DP_VALUES),
        hue_order=list(RATE_VALUES),
        palette=palette,
        ax=ax,
        dodge=True,
        jitter=0.14,
        size=2.6,
        alpha=0.45,
        linewidth=0.35,
        edgecolor="white",
        legend=False,
        zorder=3,
    )

    for x_sep in (0.5, 1.5):
        ax.axvline(x_sep, color="0.72", ls="--", lw=0.85, zorder=1)

    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    yticks = [t for t in y_tick_candidates if y_lo <= t <= y_hi]
    ax.set_yticks(yticks)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g"))
    ax.grid(axis="y", which="major", color="0.9", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(LBL_DELTA_P, fontsize=axis_fs, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=axis_fs, labelpad=8)
    ax.set_xticks(range(len(DP_VALUES)))
    ax.set_xticklabels([f"{dp:g}" for dp in DP_VALUES], fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    if panel_label:
        ax.text(
            0.03,
            0.97,
            panel_label,
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            va="top",
            ha="left",
        )

    if show_legend:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=rate_palette[rate],
                markeredgecolor=edge_palette[i],
                markeredgewidth=0.9,
                markersize=8,
                label=rate_labels[rate],
            )
            for i, rate in enumerate(RATE_VALUES)
        ]
        leg = ax.legend(
            handles=handles,
            title=LBL_K_R,
            frameon=True,
            framealpha=0.92,
            fontsize=MANUSCRIPT_LEGEND_FS if axis_fs >= MANUSCRIPT_AXIS_FS else tick_fs,
            title_fontsize=axis_fs,
            loc="lower right",
            borderaxespad=0.4,
        )
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(axis_fs)
    else:
        auto_leg = ax.get_legend()
        if auto_leg is not None:
            auto_leg.remove()


def plot_factorial_recovery_violin_hue(rows: list[dict]) -> None:
    """Recovery vs Delta p with sns.violinplot; hue encodes k_r (factorial, n=225)."""
    valid = _complete_rows(rows)
    if len(valid) < 9:
        log("SKIP factorial_recovery_violin_hue: insufficient data")
        return

    rec_vals = [max(float(r["recovery_pct"]), 0.5) for r in valid]
    y_lim = _shared_recovery_ylim(rec_vals, log_scale=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(5.0, 4.2), layout="constrained")
    _factorial_violin_hue_on_ax(
        ax,
        rows,
        panel_label="",
        ylabel=recovery_label(),
        y_lim=y_lim,
        show_legend=True,
    )
    save_figure(fig, figures_dir() / "factorial_recovery_violin_hue.png", dpi=300)


def plot_dual_factorial_recovery_violin_hue(factorial_rows: list[dict]) -> None:
    """Two-panel Set~A violins: solid-phase and outlet recovery on shared log axis."""
    solid = apply_recovery_metric(factorial_rows, "solid_phase")
    outlet = apply_recovery_metric(factorial_rows, "outlet")
    if len(_complete_rows(solid)) < 9 or len(_complete_rows(outlet)) < 9:
        log("SKIP dual factorial_recovery_violin_hue: insufficient data")
        return

    rec_vals = [
        max(float(r["recovery_pct"]), 0.5)
        for r in solid + outlet
        if int(float(r.get("complete", 0))) == 1
        and np.isfinite(float(r.get("recovery_pct", np.nan)))
    ]
    y_lim = _shared_recovery_ylim(rec_vals, log_scale=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=False)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(y_lim)
    _factorial_violin_hue_on_ax(
        axes[0],
        solid,
        panel_label="(a)",
        ylabel=recovery_axis_label("solid_phase"),
        y_lim=y_lim,
        show_legend=True,
        axis_fs=MANUSCRIPT_AXIS_FS,
        tick_fs=MANUSCRIPT_TICK_FS,
        panel_fs=MANUSCRIPT_PANEL_FS,
    )
    _factorial_violin_hue_on_ax(
        axes[1],
        outlet,
        panel_label="(b)",
        ylabel=recovery_axis_label("outlet"),
        y_lim=y_lim,
        show_legend=False,
        axis_fs=MANUSCRIPT_AXIS_FS,
        tick_fs=MANUSCRIPT_TICK_FS,
        panel_fs=MANUSCRIPT_PANEL_FS,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.95, wspace=0.22)
    save_figure(fig, ANALYSIS_COMBINED / "factorial_recovery_violin_hue_dual.png", dpi=300)


def plot_factorial_recovery_pe_da_3d(rows: list[dict]) -> None:
    """3D scatter: recovery on z-axis in accessibility-based Pe_r-Da_r space (factorial only)."""
    cu_lookup = load_accessible_cu_lookup()
    enriched = enrich_rows(rows, cu_lookup=cu_lookup)
    valid = [
        r
        for r in enriched
        if int(float(r.get("complete", 0))) == 1
        and np.isfinite(float(r.get("Pe_r", np.nan)))
        and np.isfinite(float(r.get("Da_r", np.nan)))
        and np.isfinite(float(r.get("recovery_pct", np.nan)))
        and float(r["Pe_r"]) > 0
        and float(r["Da_r"]) > 0
    ]
    if len(valid) < 5:
        log("SKIP factorial_recovery_pe_da_3d: insufficient Pe_r/Da_r rows")
        return

    rate_colors = {r: c for r, c in zip(RATE_VALUES, sns.color_palette("colorblind", n_colors=3))}
    dp_markers = {0.001: "o", 0.1: "s", 1.0: "^"}

    pe = np.array([float(r["Pe_r"]) for r in valid])
    da = np.array([float(r["Da_r"]) for r in valid])
    rec = np.array([float(r["recovery_pct"]) for r in valid])

    sns.set_theme(style="white", context="paper", font_scale=1.05)
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    for rate in RATE_VALUES:
        for dp in DP_VALUES:
            mask = [
                abs(float(r["dP_Pa"]) - dp) < 1e-12
                and abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15
                for r in valid
            ]
            if not any(mask):
                continue
            idx = np.where(mask)[0]
            ax.scatter(
                np.log10(pe[idx]),
                np.log10(da[idx]),
                rec[idx],
                c=[rate_colors[rate]],
                marker=dp_markers[dp],
                s=36,
                edgecolors="0.2",
                linewidths=0.35,
                alpha=0.9,
                depthshade=True,
            )

    z_lo = float(rec.min()) - 0.5
    z_hi = float(rec.max()) + 0.5
    ax.set_xlim(np.log10(pe.min()) - 0.08, np.log10(pe.max()) + 0.08)
    ax.set_ylim(np.log10(da.min()) - 0.08, np.log10(da.max()) + 0.08)
    ax.set_zlim(z_lo, z_hi)
    ax.set_xlabel(r"$\log_{10}\,Pe_r$")
    ax.set_ylabel(r"$\log_{10}\,Da_r$")
    ax.set_zlabel(recovery_label())
    ax.set_title(
        r"Factorial $\Delta p \times k_r$ sweep ($n=225$): "
        + recovery_label()
        + r" in $Pe_r$--$Da_r$ space",
        loc="left",
        fontsize=9,
        pad=14,
    )
    ax.view_init(elev=24, azim=-58)

    from matplotlib.lines import Line2D

    rate_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=rate_colors[rate],
            markersize=7,
            label=format_k_r_latex(rate).strip("$"),
        )
        for rate in RATE_VALUES
    ]
    dp_handles = [
        Line2D(
            [0],
            [0],
            marker=dp_markers[dp],
            color="0.2",
            linestyle="None",
            markersize=7,
            label=rf"$\Delta p={dp:g}$ Pa",
        )
        for dp in DP_VALUES
    ]
    leg1 = ax.legend(handles=rate_handles, title=r"$k_r$", loc="upper left", fontsize=7, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=dp_handles, title=r"$\Delta p$", loc="lower left", fontsize=7, framealpha=0.9)

    save_figure(fig, figures_dir() / "factorial_recovery_pe_da_3d.png", dpi=300)


def _pe_da_valid_rows(rows: list[dict]) -> list[dict]:
    """Complete campaign rows with finite Pe_r, Da_r, and recovery."""
    cu_lookup = load_accessible_cu_lookup()
    enriched = enrich_rows(rows, cu_lookup=cu_lookup)
    return [
        r
        for r in enriched
        if int(float(r.get("complete", 0))) == 1
        and np.isfinite(float(r.get("Pe_r", np.nan)))
        and np.isfinite(float(r.get("Da_r", np.nan)))
        and np.isfinite(float(r.get("recovery_pct", np.nan)))
        and float(r["Pe_r"]) > 0
        and float(r["Da_r"]) > 0
    ]


def plot_factorial_recovery_pe_da_3d_plotly(rows: list[dict]) -> None:
    """Interactive 3D scatter: Pe_r, Da_r, Cu recovery (factorial, n=225)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        log("SKIP factorial_recovery_pe_da_3d_plotly: plotly not installed")
        return

    valid = _pe_da_valid_rows(rows)
    if len(valid) < 5:
        log("SKIP factorial_recovery_pe_da_3d_plotly: insufficient Pe_r/Da_r rows")
        return

    import matplotlib.colors as mcolors

    rate_colors = dict(
        zip(
            RATE_VALUES,
            [mcolors.to_hex(c) for c in sns.color_palette("colorblind", n_colors=3)],
        )
    )
    dp_symbols = {0.001: "circle", 0.1: "square", 1.0: "diamond"}

    traces: list = []
    for rate in RATE_VALUES:
        for dp in DP_VALUES:
            sub = [
                r
                for r in valid
                if abs(float(r["dP_Pa"]) - dp) < 1e-12
                and abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15
            ]
            if not sub:
                continue
            kr = format_k_r_latex(rate).strip("$")
            traces.append(
                go.Scatter3d(
                    x=[float(r["Pe_r"]) for r in sub],
                    y=[float(r["Da_r"]) for r in sub],
                    z=[float(r["recovery_pct"]) for r in sub],
                    mode="markers",
                    name=rf"Δp={dp:g} Pa, k_r={kr}",
                    marker=dict(
                        size=5.5,
                        color=rate_colors[rate],
                        symbol=dp_symbols[dp],
                        line=dict(width=0.4, color="rgba(30,30,30,0.55)"),
                        opacity=0.9,
                    ),
                    text=[
                        (
                            f"geom {int(r['geom_seed'])}<br>"
                            f"Δp = {float(r['dP_Pa']):g} Pa<br>"
                            f"k_r = {float(r['rate_mol_cm2_s']):.1e} mol cm⁻² s⁻¹<br>"
                            f"l_r = {float(r['l_r_median_um']):.1f} μm<br>"
                            f"Pe_r = {float(r['Pe_r']):.3g}<br>"
                            f"Da_r = {float(r['Da_r']):.3g}<br>"
                            f"R = {float(r['recovery_pct']):.2f} %"
                        )
                        for r in sub
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    pe_vals = np.array([float(r["Pe_r"]) for r in valid])
    da_vals = np.array([float(r["Da_r"]) for r in valid])
    rec_vals = np.array([float(r["recovery_pct"]) for r in valid])
    pe_lim = (float(pe_vals.min()) / 1.25, float(pe_vals.max()) * 1.25)
    da_lim = (float(da_vals.min()) / 1.25, float(da_vals.max()) * 1.25)

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        width=920,
        height=760,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"Cu recovery in Pe_r–Da_r space (n={len(valid)})", x=0.03, xanchor="left"),
        scene=dict(
            xaxis=dict(
                title="Pe_r",
                type="log",
                range=[np.log10(pe_lim[0]), np.log10(pe_lim[1])],
            ),
            yaxis=dict(
                title="Da_r",
                type="log",
                range=[np.log10(da_lim[0]), np.log10(da_lim[1])],
            ),
            zaxis=dict(
                title="Cu Recovery [%]",
                range=[float(rec_vals.min()) - 0.8, float(rec_vals.max()) + 0.8],
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.55),
        ),
        legend=dict(
            title="Δp, k_r",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            font=dict(size=10),
        ),
    )

    z_floor = float(rec_vals.min()) - 0.8
    fig.add_trace(
        go.Scatter3d(
            x=[1.0, 1.0],
            y=[da_lim[0], da_lim[1]],
            z=[z_floor, z_floor],
            mode="lines",
            line=dict(color="rgba(120,120,120,0.5)", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[pe_lim[0], pe_lim[1]],
            y=[1.0, 1.0],
            z=[z_floor, z_floor],
            mode="lines",
            line=dict(color="rgba(120,120,120,0.5)", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    out_html = figures_dir() / "factorial_recovery_pe_da_3d.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    log(f"Saved {out_html}")


def load_outlet_cu_btc(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Time [h] and flux-weighted outlet Cu [mol/L] from a finished mineral_dissolution.h5."""
    with h5py.File(h5_path, "r") as h5:
        time_entries = parse_time_groups(h5)
        g0 = h5[time_entries[0][1]]
        cu_key = pick_field(g0, "Total_Cu++", required=True)
        h_key = pick_field(g0, "Free_H+", required=True)
        flux = outlet_inlet_flux_metrics(h5, time_entries, cu_key, h_key, LY, LZ)
    t_h = flux["t_s"] / 3600.0
    cu_mol_l = np.clip(flux["cu_out_molm3"] / 1000.0, 1.0e-16, None)
    return t_h, cu_mol_l


def plot_factorial_outlet_cu_btc_grid() -> None:
    """3x3 grid of flux-weighted outlet Cu BTC; 25 lines per panel colored by l_r."""
    cases = load_factorial_cases()
    complete = [
        c
        for c in cases
        if case_h5_complete(c.case_dir() / "mineral_dissolution.h5")
    ]
    if len(complete) < 3:
        log("SKIP factorial_outlet_cu_btc_grid: insufficient complete cases")
        return

    lr_all = np.array([float(c.l_r_median_um) for c in complete])
    lr_norm = plt.Normalize(vmin=float(lr_all.min()), vmax=float(lr_all.max()))
    cmap = plt.get_cmap("viridis")
    run_h = FINAL_TIME_S / 3600.0

    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.3), layout="constrained", sharex=True)
    panel = 0
    for i, rate in enumerate(RATE_VALUES):
        for j, dp in enumerate(DP_VALUES):
            ax = axes[i, j]
            cell = [
                c
                for c in complete
                if abs(c.dP_Pa - dp) < 1e-12
                and abs(c.rate_mol_cm2_s - rate) < 1e-15
            ]
            cell.sort(key=lambda c: c.l_r_median_um)
            for case in cell:
                h5 = case.case_dir() / "mineral_dissolution.h5"
                try:
                    t_h, cu = load_outlet_cu_btc(h5)
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    log(f"WARN BTC skip geom{case.geom_seed} dP={dp:g} rate={rate:.0e}: {exc}")
                    continue
                color = cmap(lr_norm(float(case.l_r_median_um)))
                ax.semilogy(t_h, cu, color=color, lw=1.1, alpha=0.9)
            ax.set_xlim(0.0, run_h)
            ax.set_title(dp_rate_title(dp, rate), loc="left", fontsize=8)
            ax.text(0.03, 0.97, f"({chr(97 + panel)})", transform=ax.transAxes, fontsize=9, va="top")
            panel += 1

    fig.supxlabel("Time [h]")
    fig.supylabel(
        r"Flux-weighted outlet Cu $\bar{C}_{Cu,out}^{fw}$ [mol L$^{-1}$]"
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=lr_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label(LBL_LR_MEDIAN, fontsize=9)
    save_figure(fig, ANALYSIS_SHARED / "factorial_outlet_cu_btc_grid.png")


def plot_anchor_recovery(rows: list[dict]) -> None:
    """Q1 anchor slice: dP=1 Pa, k_r=1e-9."""
    sub = _complete_rows(
        [
            r
            for r in rows
            if abs(float(r["dP_Pa"]) - 1.0) < 1e-12 and abs(float(r["rate_mol_cm2_s"]) - 1e-9) < 1e-15
        ]
    )
    if len(sub) < 2:
        log("SKIP anchor_recovery: need >=2 complete rows at dP=1, k_r=1e-9")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(6.2, 4.5), layout="constrained")
    lr = np.array([float(r["l_r_median_um"]) for r in sub])
    rec = np.array([float(r["recovery_pct"]) for r in sub])
    k = np.array([float(r["k_pore"]) for r in sub if np.isfinite(float(r.get("k_pore", np.nan)))])
    sc = ax.scatter(
        lr,
        rec,
        s=55,
        c=k * 1e10 if len(k) == len(sub) else "#2166ac",
        cmap="viridis" if len(k) == len(sub) else None,
        edgecolors="0.2",
        linewidths=0.35,
    )
    if len(k) == len(sub):
        cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label(r"$k_{\mathrm{pore}}$ [$10^{-10}$ m$^2$]")
    ax.set_xlabel(LBL_LR_MEDIAN)
    ax.set_ylabel(recovery_label())
    ax.set_title(anchor_title(1.0, 1e-9), loc="left")
    save_figure(fig, figures_dir() / "anchor_recovery_vs_l_r.png")


def plot_factorial_regime(rows: list[dict]) -> None:
    """Recovery vs theta (transport) and recovery vs rate, all complete factorial cases."""
    valid = _complete_rows(rows)
    if len(valid) < 5:
        log("SKIP factorial_regime: insufficient data")
        return

    rate_colors = {r: c for r, c in zip(RATE_VALUES, sns.color_palette("colorblind", n_colors=len(RATE_VALUES)))}
    dp_markers = {0.001: "o", 0.1: "s", 1.0: "^"}

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained", sharey=True)

    for rate in RATE_VALUES:
        pts = [
            (float(r["theta_48h"]), float(r["recovery_pct"]), float(r["dP_Pa"]))
            for r in valid
            if abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15 and np.isfinite(float(r["theta_48h"]))
        ]
        if not pts:
            continue
        for theta, rec, dp in pts:
            axes[0].scatter(
                theta,
                rec,
                c=[rate_colors[rate]],
                marker=dp_markers.get(dp, "o"),
                s=45,
                edgecolors="0.2",
                linewidths=0.3,
                alpha=0.85,
            )

    axes[0].set_xlabel(LBL_THETA_48H)
    axes[0].set_ylabel(recovery_label())
    axes[0].set_title(rf"(a) Transport exposure vs {recovery_label()}", loc="left")

    dp_colors = {d: c for d, c in zip(DP_VALUES, sns.color_palette("dark", n_colors=len(DP_VALUES)))}
    for dp in DP_VALUES:
        pts = sorted(
            (float(r["rate_mol_cm2_s"]), float(r["recovery_pct"]))
            for r in valid
            if abs(float(r["dP_Pa"]) - dp) < 1e-12
        )
        if not pts:
            continue
        rates, recs = zip(*pts)
        axes[1].plot(
            rates,
            recs,
            "o",
            color=dp_colors[dp],
            ms=5,
            alpha=0.55,
            label=rf"$\Delta p={dp:g}$ Pa",
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(LBL_K_R)
    axes[1].set_title(rf"(b) {recovery_label()} vs $k_r$ by $\Delta p$", loc="left")
    axes[1].legend(title=r"$\Delta p$", fontsize=7, loc="upper left")

    save_figure(fig, figures_dir() / "factorial_regime.png")


def _pe_da_reference_lines(ax: plt.Axes) -> None:
    """Faint Pe_r=1 and Da_r=1 guides on log-log Pe-Da axes."""
    ax.axvline(1.0, color="0.55", ls="--", lw=0.9, alpha=0.75, zorder=1)
    ax.axhline(1.0, color="0.55", ls="--", lw=0.9, alpha=0.75, zorder=1)


def plot_pe_da_regime(factorial_rows: list[dict], matched_rows: list[dict]) -> None:
    """Two-panel Pe_r-Da_r figure: (a) factorial 3x3 sweep; (b) matched-Theta 4a/4b/4c."""
    cu_lookup = load_accessible_cu_lookup()
    fac = enrich_rows(factorial_rows, cu_lookup=cu_lookup)
    mat = enrich_rows(matched_rows, cu_lookup=cu_lookup)
    fac_valid = [
        r
        for r in fac
        if int(float(r.get("complete", 0))) == 1
        and np.isfinite(float(r.get("Pe_r", np.nan)))
        and np.isfinite(float(r.get("Da_r", np.nan)))
        and float(r["Pe_r"]) > 0
        and float(r["Da_r"]) > 0
    ]
    mat_valid = [
        r
        for r in mat
        if int(float(r.get("complete", 0))) == 1
        and np.isfinite(float(r.get("Pe_r", np.nan)))
        and np.isfinite(float(r.get("Da_r", np.nan)))
        and float(r["Pe_r"]) > 0
        and float(r["Da_r"]) > 0
    ]
    if len(fac_valid) < 2 and len(mat_valid) < 2:
        log("SKIP factorial_pe_da_regime: insufficient Pe_r/Da_r rows")
        return

    write_csv(FACTORIAL_ENRICHED_CSV, fac_valid)

    rate_colors = {r: c for r, c in zip(RATE_VALUES, sns.color_palette("colorblind", n_colors=3))}
    dp_markers = {0.001: "o", 0.1: "s", 1.0: "^"}

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, (ax_fac, ax_mat) = plt.subplots(
        1,
        2,
        figsize=(12.0, 6.8),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )

    if len(fac_valid) >= 2:
        for rate in RATE_VALUES:
            for dp in DP_VALUES:
                sub = [
                    r
                    for r in fac_valid
                    if abs(float(r["dP_Pa"]) - dp) < 1e-12
                    and abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15
                ]
                if not sub:
                    continue
                pe = np.array([float(r["Pe_r"]) for r in sub])
                da = np.array([float(r["Da_r"]) for r in sub])
                rec = np.array([float(r["recovery_pct"]) for r in sub])
                ax_fac.scatter(
                    pe,
                    da,
                    s=24.0 + 4.0 * rec,
                    c=[rate_colors[rate]],
                    marker=dp_markers[dp],
                    edgecolors="0.2",
                    linewidths=0.35,
                    alpha=0.88,
                    label=dp_rate_title(dp, rate).replace(" Pa,", ","),
                    zorder=3,
                )
        _pe_da_reference_lines(ax_fac)
        ax_fac.set_xscale("log")
        ax_fac.set_yscale("log")
        ax_fac.set_xlabel(LBL_PE_R, fontsize=MANUSCRIPT_AXIS_FS)
        ax_fac.set_ylabel(LBL_DA_R, fontsize=MANUSCRIPT_AXIS_FS)
        ax_fac.tick_params(axis="both", labelsize=MANUSCRIPT_TICK_FS)
        add_panel_label(ax_fac, "(a)", fontsize=MANUSCRIPT_PANEL_FS)
        ax_fac.legend(
            title="Set A",
            fontsize=MANUSCRIPT_LEGEND_FS - 4.0,
            title_fontsize=MANUSCRIPT_LEGEND_FS - 3.0,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=True,
            framealpha=0.95,
            ncol=3,
            borderaxespad=0.0,
            handletextpad=0.4,
            columnspacing=1.0,
        )
    else:
        ax_fac.set_visible(False)

    if len(mat_valid) >= 2:
        branch_rates = {
            b: float(next(r["rate_mol_cm2_s"] for r in mat_valid if r.get("branch") == b))
            for b in ("4a", "4b", "4c")
            if any(r.get("branch") == b for r in mat_valid)
        }
        matched_branches = sorted(branch_rates, key=lambda b: branch_rates[b], reverse=True)
        matched_markers = {"4a": "D", "4b": "X", "4c": "P"}
        lr_all = np.array([float(r["l_r_median_um"]) for r in mat_valid if r.get("branch") in matched_branches])
        lr_norm = plt.Normalize(vmin=float(lr_all.min()), vmax=float(lr_all.max()))
        cmap = plt.get_cmap("viridis")
        pe_all: list[float] = []
        da_all: list[float] = []
        sc = None
        for branch in matched_branches:
            sub = [r for r in mat_valid if r.get("branch") == branch]
            if not sub:
                continue
            rate = float(sub[0]["rate_mol_cm2_s"])
            kr = format_k_r_latex(rate).strip("$")
            label = rf"$k_r={kr}$"
            pe = np.array([float(r["Pe_r"]) for r in sub])
            da = np.array([float(r["Da_r"]) for r in sub])
            rec = np.array([float(r["recovery_pct"]) for r in sub])
            pe_all.extend(pe.tolist())
            da_all.extend(da.tolist())
            sc = ax_mat.scatter(
                pe,
                da,
                c=[float(r["l_r_median_um"]) for r in sub],
                cmap=cmap,
                norm=lr_norm,
                marker=matched_markers.get(branch, "o"),
                s=28.0 + 4.0 * rec,
                edgecolors="0.2",
                linewidths=0.35,
                label=label,
                zorder=3,
            )
        _pe_da_reference_lines(ax_mat)
        ax_mat.set_xscale("log")
        ax_mat.set_yscale("log")
        pe_arr = np.array(pe_all)
        da_arr = np.array(da_all)
        pad_pe = 10 ** (0.15 * (np.log10(pe_arr.max()) - np.log10(pe_arr.min()) + 0.05))
        pad_da = 10 ** (0.15 * (np.log10(da_arr.max()) - np.log10(da_arr.min()) + 0.05))
        ax_mat.set_xlim(pe_arr.min() / pad_pe, pe_arr.max() * pad_pe)
        ax_mat.set_ylim(da_arr.min() / pad_da, da_arr.max() * pad_da)
        ax_mat.set_xlabel(LBL_PE_R, fontsize=MANUSCRIPT_AXIS_FS)
        ax_mat.set_ylabel(LBL_DA_R, fontsize=MANUSCRIPT_AXIS_FS)
        ax_mat.tick_params(axis="both", labelsize=MANUSCRIPT_TICK_FS)
        add_panel_label(ax_mat, "(b)", fontsize=MANUSCRIPT_PANEL_FS)
        ax_mat.legend(
            title="Set B",
            fontsize=MANUSCRIPT_LEGEND_FS - 4.0,
            title_fontsize=MANUSCRIPT_LEGEND_FS - 3.0,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=True,
            framealpha=0.95,
            ncol=1,
            borderaxespad=0.0,
        )
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax_mat, shrink=0.88, pad=0.06)
            cbar.ax.tick_params(labelsize=MANUSCRIPT_TICK_FS)
            cbar.set_label(LBL_LR_MEDIAN, fontsize=MANUSCRIPT_LEGEND_FS)
    else:
        ax_mat.set_visible(False)

    save_figure(fig, figures_dir() / "factorial_pe_da_regime.png")


def _all_collapse_rows(factorial_rows: list[dict], matched_rows: list[dict]) -> list[dict]:
    """All complete factorial and matched-Theta rows from accessibility_ensemble/."""
    return _complete_rows(factorial_rows) + _complete_rows(matched_rows)


def enrich_for_collapse(rows: list[dict], *, cu_lookup: dict[str, float]) -> list[dict]:
    """Add Theta*exp(-l_r/lambda) and Theta*f_Cu,acc without legacy case_index paths."""
    out: list[dict] = []
    for row in rows:
        r = dict(row)
        try:
            theta = float(r["theta_48h"])
            l_r_m = float(r["l_r_median_um"]) * 1e-6
        except (KeyError, ValueError, TypeError):
            out.append(r)
            continue
        cu_frac = cu_lookup.get(str(r.get("candidate_id", "")), np.nan)
        if not np.isfinite(cu_frac):
            cu_frac = cu_lookup.get(str(r.get("geom_seed", "")), np.nan)
        r["accessible_cu_frac"] = cu_frac
        r["theta_exp_lr"] = theta * np.exp(-l_r_m / COLLAPSE_LAMBDA_M)
        r["theta_cu_access"] = theta * cu_frac if np.isfinite(cu_frac) else np.nan
        out.append(r)
    return out


def _matched_theta_collapse_label(branch: str, rate: float) -> str:
    kr = format_k_r_latex(rate).strip("$")
    return rf"$k_r={kr}$"


def _set_ab_legend(ax, handles, labels, *, fontsize: float) -> None:
    """Legend with bold Set A / Set B headers; Set B ordered by decreasing k_r."""
    if not handles:
        return
    set_a = [(h, lab) for h, lab in zip(handles, labels) if r"\Delta p" in lab]
    set_b = [(h, lab) for h, lab in zip(handles, labels) if r"\Delta p" not in lab]
    desired_b = [
        rf"$k_r={format_k_r_latex(r).strip('$')}$" for r in sorted(RATE_VALUES, reverse=True)
    ]
    b_map = {lab: h for h, lab in set_b}
    set_b_sorted = [(b_map[lab], lab) for lab in desired_b if lab in b_map]
    blank = Line2D([], [], linestyle="None", marker="", color="none")
    grouped_h = [blank] + [h for h, _ in set_a] + [blank] + [h for h, _ in set_b_sorted]
    grouped_l = (
        ["Set A"]
        + [lab.replace(" Pa,", ",") for _, lab in set_a]
        + ["Set B"]
        + [lab for _, lab in set_b_sorted]
    )
    leg = ax.legend(
        grouped_h,
        grouped_l,
        fontsize=fontsize,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.92,
        ncol=1,
        borderaxespad=0.0,
        handletextpad=0.5,
    )
    for text in leg.get_texts():
        if text.get_text() in ("Set A", "Set B"):
            text.set_fontweight("bold")


def _collapse_recovery_model(
    theta_cu_access: np.ndarray,
    k_r: np.ndarray | float,
    *,
    amplitude_pct: float,
    alpha: float,
    gamma: float,
    k_ref: float = MATCHED_THETA_K_REF,
) -> np.ndarray:
    """R(%) = A * (k_r/k_ref)^alpha * (Theta*f_Cu,acc)^gamma."""
    k_arr = np.asarray(k_r, dtype=float)
    exposure = np.maximum(theta_cu_access, 1e-12)
    return amplitude_pct * (k_arr / k_ref) ** alpha * exposure**gamma


def _fit_collapse_unified_arrays(
    exposure: np.ndarray,
    kr: np.ndarray,
    rec: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Fit R = A (k_r/k_ref)^alpha exposure^gamma. Returns (A, alpha, gamma, rmse)."""
    if len(rec) < 5:
        return None
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    exposure = np.asarray(exposure, dtype=float)
    rec = np.asarray(rec, dtype=float)
    kr = np.asarray(kr, dtype=float)
    mask = np.isfinite(exposure) & np.isfinite(rec) & np.isfinite(kr) & (exposure > 0.0) & (rec > 0.0)
    if int(mask.sum()) < 5:
        return None
    exposure = exposure[mask]
    rec = rec[mask]
    kr = kr[mask]

    def model(x: tuple[np.ndarray, np.ndarray], log_a: float, alpha: float, gamma: float) -> np.ndarray:
        exp_arr, k_r = x
        return _collapse_recovery_model(
            exp_arr,
            k_r,
            amplitude_pct=np.exp(log_a),
            alpha=alpha,
            gamma=gamma,
        )

    try:
        popt, _ = curve_fit(
            model,
            (exposure, kr),
            rec,
            p0=[np.log(8.0), 0.6, 0.4],
            bounds=([-2.0, 0.0, 0.0], [4.0, 2.0, 2.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None

    log_a, alpha, gamma = (float(popt[0]), float(popt[1]), float(popt[2]))
    amplitude = float(np.exp(log_a))
    pred = model((exposure, kr), log_a, alpha, gamma)
    rmse = float(np.sqrt(np.mean((rec - pred) ** 2)))
    return amplitude, alpha, gamma, rmse


def _fit_collapse_unified_model(
    valid: list[dict],
) -> tuple[float, float, float, float] | None:
    """Fit pooled collapse law on theta_cu_access. Returns (A, alpha, gamma, rmse)."""
    exposure = np.array([float(r["theta_cu_access"]) for r in valid])
    rec = np.array([float(r["recovery_pct"]) for r in valid])
    kr = np.array([float(r["rate_mol_cm2_s"]) for r in valid])
    return _fit_collapse_unified_arrays(exposure, kr, rec)


def _coefficient_of_determination(y: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 = 1 - SS_res / SS_tot."""
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan
    x_m = x[mask]
    y_m = y[mask]
    if float(np.std(x_m)) <= 0.0 or float(np.std(y_m)) <= 0.0:
        return np.nan
    return float(np.corrcoef(x_m, y_m)[0, 1])


def _fit_simple_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float] | None:
    """Fit y = A * x^gamma for x > 0. Returns (A, gamma, r2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if int(mask.sum()) < 5:
        return None
    x_m = x[mask]
    y_m = y[mask]
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    def model(xv: np.ndarray, log_a: float, gamma: float) -> np.ndarray:
        return np.exp(log_a) * np.maximum(xv, 1e-15) ** gamma

    try:
        popt, _ = curve_fit(
            model,
            x_m,
            y_m,
            p0=[np.log(np.median(y_m)), 0.4],
            bounds=([-6.0, -4.0], [6.0, 4.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    log_a, gamma = float(popt[0]), float(popt[1])
    amplitude = float(np.exp(log_a))
    pred = model(x_m, log_a, gamma)
    r2 = _coefficient_of_determination(y_m, pred)
    return amplitude, gamma, r2


def _collapse_compare_rows(valid: list[dict]) -> list[dict]:
    """Build SI comparison rows for organizer predictors of R."""
    rec = np.array([float(r["recovery_pct"]) for r in valid])
    kr = np.array([float(r["rate_mol_cm2_s"]) for r in valid])
    lam_um = int(round(COLLAPSE_LAMBDA_M * 1e6))
    theta_exp_label = rf"$\Thetaacc \exp(-l_r/{lam_um}~\mu\mathrm{{m}})$"
    metrics: list[tuple[str, str, np.ndarray, str]] = [
        (
            "full_model_kr",
            r"$\Thetaacc f_{\mathrm{Cu,acc}}$ + $k_r$",
            np.array([float(r["theta_cu_access"]) for r in valid]),
            "kr",
        ),
        (
            "theta_exp_lr_kr",
            f"{theta_exp_label} + $k_r$",
            np.array([float(r["theta_exp_lr"]) for r in valid]),
            "kr",
        ),
        (
            "theta_kr",
            r"$\Thetaacc$ + $k_r$",
            np.array([float(r["theta_48h"]) for r in valid]),
            "kr",
        ),
        ("theta", r"$\Thetaacc$ alone", np.array([float(r["theta_48h"]) for r in valid]), "simple"),
        (
            "theta_cu_access",
            r"$\Thetaacc f_{\mathrm{Cu,acc}}$",
            np.array([float(r["theta_cu_access"]) for r in valid]),
            "simple",
        ),
        (
            "theta_exp_lr",
            theta_exp_label,
            np.array([float(r["theta_exp_lr"]) for r in valid]),
            "simple",
        ),
        (
            "f_cu_acc",
            r"$f_{\mathrm{Cu,acc}}$ alone",
            np.array([float(r["accessible_cu_frac"]) for r in valid]),
            "simple",
        ),
        (
            "l_r",
            r"$l_r$ alone",
            np.array([float(r["l_r_median_um"]) for r in valid]),
            "simple",
        ),
        ("Pe_r", r"$Pe_r$", np.array([float(r["Pe_r"]) for r in valid]), "corr_only"),
    ]

    out: list[dict] = []
    for key, label, exposure, fit_mode in metrics:
        corr = _pearson_corr(exposure, rec)
        r2 = np.nan
        if fit_mode == "kr":
            fit = _fit_collapse_unified_arrays(exposure, kr, rec)
            if fit is not None:
                amplitude, alpha, gamma, _ = fit
                pred = _collapse_recovery_model(
                    exposure,
                    kr,
                    amplitude_pct=amplitude,
                    alpha=alpha,
                    gamma=gamma,
                )
                r2 = _coefficient_of_determination(rec, pred)
        elif fit_mode == "simple":
            fit = _fit_simple_power_law(exposure, rec)
            if fit is not None:
                _, _, r2 = fit
        out.append(
            {
                "metric_key": key,
                "metric_label": label,
                "n": len(valid),
                "corr_r": corr,
                "power_law_r2": r2,
                "uses_kr": int(fit_mode == "kr"),
            }
        )
    return out


def _format_corr_tex(corr: float) -> str:
    if not np.isfinite(corr):
        return "---"
    return f"{corr:.3f}"


def _format_r2_tex(r2: float, *, bold: bool = False) -> str:
    if not np.isfinite(r2):
        return "---"
    if r2 < 0.005:
        text = "$\\approx 0$"
    else:
        text = f"{r2:.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def write_collapse_compare_table(rows: list[dict]) -> None:
    """Write per-metric collapse_compare CSV; LaTeX SI table for solid_phase only."""
    csv_path = collapse_compare_csv()
    write_csv(csv_path, rows)

    if _ACTIVE_METRIC != "solid_phase":
        return

    lines = [
        "  \\begin{tabular}{lcc}",
        "    \\hline",
        "    Organizer & $\\mathrm{corr}(R,\\cdot)$ & Fit $R^2$ \\\\",
        "    \\hline",
    ]
    for row in rows:
        label = str(row["metric_label"])
        corr_tex = _format_corr_tex(float(row["corr_r"]))
        is_full = str(row["metric_key"]) == "full_model_kr"
        r2_tex = _format_r2_tex(float(row["power_law_r2"]), bold=is_full)
        lines.append(f"    {label} & {corr_tex} & {r2_tex} \\\\")
    lines.extend(["    \\hline", "  \\end{tabular}"])

    table_tex = "\n".join(lines) + "\n"
    si_text = SI_TEX.read_text(encoding="utf-8")
    if COLLAPSE_COMPARE_BEGIN not in si_text or COLLAPSE_COMPARE_END not in si_text:
        raise RuntimeError(
            f"Missing collapse_compare markers in {SI_TEX}; "
            f"expected {COLLAPSE_COMPARE_BEGIN!r} and {COLLAPSE_COMPARE_END!r}"
        )
    before, rest = si_text.split(COLLAPSE_COMPARE_BEGIN, 1)
    _, after = rest.split(COLLAPSE_COMPARE_END, 1)
    updated = (
        before
        + COLLAPSE_COMPARE_BEGIN
        + " (auto-generated by plot_accessibility_ensemble.py)\n"
        + table_tex
        + COLLAPSE_COMPARE_END
        + after
    )
    SI_TEX.write_text(updated, encoding="utf-8")
    log(f"Updated collapse_compare table in {SI_TEX}")


def build_collapse_compare_table(rows: list[dict]) -> list[dict]:
    """Enrich pooled rows and return exposure-metric comparison records."""
    enriched = enrich_for_collapse(rows, cu_lookup=load_accessible_cu_lookup())
    enriched = enrich_rows(enriched, cu_lookup=load_accessible_cu_lookup())
    valid = [
        r
        for r in enriched
        if np.isfinite(float(r.get("theta_cu_access", np.nan)))
        and np.isfinite(float(r.get("recovery_pct", np.nan)))
        and np.isfinite(float(r.get("accessible_cu_frac", np.nan)))
        and np.isfinite(float(r.get("Pe_r", np.nan)))
        and float(r.get("Pe_r", 0.0)) > 0.0
    ]
    if len(valid) < 5:
        log("SKIP collapse-compare: insufficient complete rows")
        return []
    return _collapse_compare_rows(valid)


def _collapse_panel_on_ax(
    ax: plt.Axes,
    rows: list[dict],
    *,
    panel_label: str,
    ylabel: str,
    recovery_symbol: str,
    show_legend: bool,
    show_fit_annot: bool = True,
    axis_fs: float = 12.0,
    tick_fs: float = 11.0,
    title_fs: float = 10.0,
    annot_fs: float = 9.0,
) -> None:
    """One collapse panel: recovery vs Theta*f_Cu,acc with pooled power-law fit."""
    enriched = enrich_for_collapse(rows, cu_lookup=load_accessible_cu_lookup())
    valid = [
        r
        for r in enriched
        if np.isfinite(float(r.get("theta_cu_access", np.nan)))
        and np.isfinite(float(r.get("recovery_pct", np.nan)))
    ]
    if len(valid) < 2:
        ax.set_visible(False)
        return

    rate_colors = {r: c for r, c in zip(RATE_VALUES, sns.color_palette("colorblind", n_colors=3))}
    dp_markers = {0.001: "o", 0.1: "s", 1.0: "^"}
    matched_markers = {"4a": "D", "4b": "X", "4c": "P"}
    matched_colors = {
        branch: color
        for branch, color in zip(
            ("4a", "4b", "4c"),
            sns.color_palette("dark", n_colors=3),
        )
    }

    unified_fit = _fit_collapse_unified_model(valid)
    x_all = np.array([float(r["theta_cu_access"]) for r in valid])

    for rate in RATE_VALUES:
        for dp in DP_VALUES:
            sub = [
                r
                for r in valid
                if r.get("branch") is None
                and abs(float(r["dP_Pa"]) - dp) < 1e-12
                and abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15
            ]
            if not sub:
                continue
            color = rate_colors[rate]
            ax.scatter(
                [float(r["theta_cu_access"]) for r in sub],
                [float(r["recovery_pct"]) for r in sub],
                s=42,
                color=color,
                marker=dp_markers[dp],
                edgecolors="0.2",
                linewidths=0.35,
                alpha=0.88,
                label=dp_rate_title(dp, rate),
                zorder=3,
            )
            if unified_fit is not None:
                x_line = np.geomspace(
                    max(1e-4, float(x_all.min()) * 0.8),
                    float(x_all.max()) * 1.05,
                    120,
                )
                amplitude, alpha, gamma, _rmse = unified_fit
                ax.plot(
                    x_line,
                    _collapse_recovery_model(x_line, rate, amplitude_pct=amplitude, alpha=alpha, gamma=gamma),
                    "--",
                    color=color,
                    lw=1.5,
                    alpha=0.88,
                    zorder=2,
                )

    for branch in sorted({str(r["branch"]) for r in valid if r.get("branch")}):
        sub = [r for r in valid if r.get("branch") == branch]
        if not sub:
            continue
        rate = float(sub[0]["rate_mol_cm2_s"])
        color = matched_colors.get(branch, "#555555")
        ax.scatter(
            [float(r["theta_cu_access"]) for r in sub],
            [float(r["recovery_pct"]) for r in sub],
            s=46,
            color=color,
            marker=matched_markers.get(branch, "h"),
            edgecolors="0.2",
            linewidths=0.35,
            alpha=0.92,
            label=_matched_theta_collapse_label(branch, rate),
            zorder=4,
        )
        if unified_fit is not None:
            x_line = np.geomspace(
                max(1e-4, float(x_all.min()) * 0.8),
                float(x_all.max()) * 1.05,
                120,
            )
            amplitude, alpha, gamma, _rmse = unified_fit
            ax.plot(
                x_line,
                _collapse_recovery_model(x_line, rate, amplitude_pct=amplitude, alpha=alpha, gamma=gamma),
                "--",
                color=color,
                lw=1.6,
                alpha=0.9,
                zorder=2,
            )

    k_ref_tex = format_k_r_latex(MATCHED_THETA_K_REF).strip("$")
    add_panel_label(ax, panel_label, fontsize=MANUSCRIPT_PANEL_FS)
    ax.set_xlabel(LBL_THETA_FCU, fontsize=axis_fs)
    ax.set_ylabel(ylabel, fontsize=axis_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0.0)
    if show_fit_annot and unified_fit is not None:
        amplitude, alpha, gamma, rmse = unified_fit
        exposure = np.array([float(r["theta_cu_access"]) for r in valid])
        rec = np.array([float(r["recovery_pct"]) for r in valid])
        kr = np.array([float(r["rate_mol_cm2_s"]) for r in valid])
        pred = _collapse_recovery_model(exposure, kr, amplitude_pct=amplitude, alpha=alpha, gamma=gamma)
        r2 = 1.0 - float(np.sum((rec - pred) ** 2) / np.sum((rec - rec.mean()) ** 2))
        ax.text(
            0.02,
            0.98,
            rf"$\alpha = {alpha:.2f}$, $\gamma = {gamma:.2f}$"
            + "\n"
            + rf"$A = {amplitude:.1f}$% at $k_r = {k_ref_tex}$"
            + "\n"
            + rf"RMSE $= {rmse:.2f}$%, $R^2 = {r2:.3f}$",
            transform=ax.transAxes,
            fontsize=annot_fs,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.75", "alpha": 0.95},
        )
    if show_legend:
        ax.legend(
            fontsize=MANUSCRIPT_LEGEND_FS,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            framealpha=0.92,
            ncol=1,
            borderaxespad=0.0,
        )


def plot_theta_cu_access_collapse(rows: list[dict]) -> None:
    """Single-panel recovery vs Theta*f_Cu,acc with pooled power-law fit."""
    if len(rows) < 2:
        log("SKIP collapse: insufficient rows")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    sym = r"R_{\mathrm{s}}" if _ACTIVE_METRIC == "solid_phase" else r"R_{\mathrm{out}}"
    _collapse_panel_on_ax(
        ax,
        rows,
        panel_label="(a)",
        ylabel=recovery_label(),
        recovery_symbol=sym,
        show_legend=True,
    )
    save_figure(fig, figures_dir() / "collapse.png")


def plot_dual_theta_cu_access_collapse(factorial_rows: list[dict], matched_rows: list[dict]) -> None:
    """Two-panel collapse organizer for solid-phase and outlet recovery."""
    solid_rows = _all_collapse_rows(
        apply_recovery_metric(factorial_rows, "solid_phase"),
        apply_recovery_metric(matched_rows, "solid_phase"),
    )
    outlet_rows = _all_collapse_rows(
        apply_recovery_metric(factorial_rows, "outlet"),
        apply_recovery_metric(matched_rows, "outlet"),
    )
    if len(solid_rows) < 2 or len(outlet_rows) < 2:
        log("SKIP dual collapse: insufficient rows")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8))
    panel_kw = dict(
        axis_fs=MANUSCRIPT_AXIS_FS + 2.0,
        tick_fs=MANUSCRIPT_TICK_FS + 2.0,
        title_fs=MANUSCRIPT_TICK_FS,
        annot_fs=MANUSCRIPT_TICK_FS,
    )
    _collapse_panel_on_ax(
        axes[0],
        solid_rows,
        panel_label="(a)",
        ylabel=recovery_axis_label("solid_phase"),
        recovery_symbol=r"R_{\mathrm{s}}",
        show_legend=False,
        show_fit_annot=False,
        **panel_kw,
    )
    _collapse_panel_on_ax(
        axes[1],
        outlet_rows,
        panel_label="(b)",
        ylabel=recovery_axis_label("outlet"),
        recovery_symbol=r"R_{\mathrm{out}}",
        show_legend=False,
        show_fit_annot=False,
        **panel_kw,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    _set_ab_legend(axes[1], handles, labels, fontsize=MANUSCRIPT_LEGEND_FS)
    fig.subplots_adjust(left=0.07, right=0.82, bottom=0.14, top=0.96, wspace=0.28)
    save_figure(fig, ANALYSIS_COMBINED / "collapse_dual.png", dpi=300)


def plot_recovery_vs_l_r_all(factorial_rows: list[dict], matched_rows: list[dict]) -> None:
    """Single-panel R vs l_r for all factorial and matched-Theta cases."""
    valid = _all_collapse_rows(factorial_rows, matched_rows)
    if len(valid) < 2:
        log("SKIP recovery_vs_l_r_all: insufficient complete rows")
        return

    rate_colors = {r: c for r, c in zip(RATE_VALUES, sns.color_palette("colorblind", n_colors=3))}
    dp_markers = {0.001: "o", 0.1: "s", 1.0: "^"}
    matched_markers = {"4a": "D", "4b": "X", "4c": "P"}
    matched_colors = {
        branch: color
        for branch, color in zip(
            ("4a", "4b", "4c"),
            sns.color_palette("dark", n_colors=3),
        )
    }

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(8.2, 4.7), layout="constrained")

    for rate in RATE_VALUES:
        for dp in DP_VALUES:
            sub = [
                r
                for r in valid
                if r.get("branch") is None
                and abs(float(r["dP_Pa"]) - dp) < 1e-12
                and abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15
            ]
            if not sub:
                continue
            ax.scatter(
                [float(r["l_r_median_um"]) for r in sub],
                [float(r["recovery_pct"]) for r in sub],
                s=42,
                c=[rate_colors[rate]],
                marker=dp_markers[dp],
                edgecolors="0.2",
                linewidths=0.35,
                alpha=0.88,
                label=dp_rate_title(dp, rate),
                zorder=3,
            )

    for branch in sorted({str(r["branch"]) for r in valid if r.get("branch")}):
        sub = [r for r in valid if r.get("branch") == branch]
        if not sub:
            continue
        rate = float(sub[0]["rate_mol_cm2_s"])
        ax.scatter(
            [float(r["l_r_median_um"]) for r in sub],
            [float(r["recovery_pct"]) for r in sub],
            s=46,
            c=[matched_colors.get(branch, "#555555")],
            marker=matched_markers.get(branch, "h"),
            edgecolors="0.2",
            linewidths=0.35,
            alpha=0.92,
            label=_matched_theta_collapse_label(branch, rate),
            zorder=4,
        )

    ax.set_xlabel(LBL_LR_MEDIAN, fontsize=MANUSCRIPT_AXIS_FS)
    ax.set_ylabel(recovery_label(), fontsize=MANUSCRIPT_AXIS_FS)
    ax.tick_params(axis="both", labelsize=MANUSCRIPT_TICK_FS)
    ax.set_ylim(bottom=0.0)
    handles, labels = ax.get_legend_handles_labels()
    _set_ab_legend(ax, handles, labels, fontsize=MANUSCRIPT_LEGEND_FS - 3.0)
    save_figure(fig, figures_dir() / "recovery_vs_l_r_all.png")


def plot_rs_vs_rout(factorial_rows: list[dict], matched_rows: list[dict]) -> None:
    """Ensemble scatter of outlet vs solid-phase recovery (Set A + Set B)."""
    rows = list(factorial_rows) + list(matched_rows)
    valid = []
    for r in rows:
        if int(float(r.get("complete", 0))) != 1:
            continue
        try:
            rs = float(r["solid_recovery_pct"])
            rout = float(r["outlet_recovery_pct"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(rs) and np.isfinite(rout):
            valid.append(r)
    if len(valid) < 2:
        log("SKIP rs_vs_rout: insufficient complete rows")
        return

    rate_colors = {r: c for r, c in zip(RATE_VALUES, sns.color_palette("colorblind", n_colors=3))}
    dp_markers = {0.001: "o", 0.1: "s", 1.0: "^"}
    matched_markers = {"4a": "D", "4b": "X", "4c": "P"}
    matched_colors = {
        branch: color
        for branch, color in zip(("4a", "4b", "4c"), sns.color_palette("dark", n_colors=3))
    }

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(8.0, 5.2), layout="constrained")

    for rate in RATE_VALUES:
        for dp in DP_VALUES:
            sub = [
                r
                for r in valid
                if r.get("branch") is None
                and abs(float(r["dP_Pa"]) - dp) < 1e-12
                and abs(float(r["rate_mol_cm2_s"]) - rate) < 1e-15
            ]
            if not sub:
                continue
            ax.scatter(
                [float(r["solid_recovery_pct"]) for r in sub],
                [float(r["outlet_recovery_pct"]) for r in sub],
                s=42,
                c=[rate_colors[rate]],
                marker=dp_markers[dp],
                edgecolors="0.2",
                linewidths=0.35,
                alpha=0.88,
                label=dp_rate_title(dp, rate),
                zorder=3,
            )

    for branch in sorted({str(r["branch"]) for r in valid if r.get("branch")}):
        sub = [r for r in valid if r.get("branch") == branch]
        if not sub:
            continue
        rate = float(sub[0]["rate_mol_cm2_s"])
        ax.scatter(
            [float(r["solid_recovery_pct"]) for r in sub],
            [float(r["outlet_recovery_pct"]) for r in sub],
            s=46,
            c=[matched_colors.get(branch, "#555555")],
            marker=matched_markers.get(branch, "h"),
            edgecolors="0.2",
            linewidths=0.35,
            alpha=0.92,
            label=_matched_theta_collapse_label(branch, rate),
            zorder=4,
        )

    lim = max(
        max(float(r["solid_recovery_pct"]) for r in valid),
        max(float(r["outlet_recovery_pct"]) for r in valid),
    )
    pad = max(1.0, 0.05 * lim)
    ax.plot([0.0, lim + pad], [0.0, lim + pad], color="0.55", lw=1.0, ls="--", zorder=1)
    ax.set_xlim(0.0, lim + pad)
    ax.set_ylim(0.0, lim + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(LBL_CU_RECOVERY_SOLID, fontsize=MANUSCRIPT_AXIS_FS)
    ax.set_ylabel(LBL_CU_RECOVERY_OUTLET, fontsize=MANUSCRIPT_AXIS_FS)
    ax.tick_params(axis="both", labelsize=MANUSCRIPT_TICK_FS)
    handles, labels = ax.get_legend_handles_labels()
    _set_ab_legend(ax, handles, labels, fontsize=MANUSCRIPT_LEGEND_FS - 3.0)
    out = ANALYSIS_COMBINED / "rs_vs_rout_ensemble.png"
    save_figure(fig, out)
    draft = REPO_ROOT / "manuscript/draft/figs/rs_vs_rout_ensemble.png"
    if draft.parent.is_dir():
        import shutil

        shutil.copy2(out, draft)


def _matched_theta_recovery_model(
    l_r_um: np.ndarray,
    k_r: np.ndarray | float,
    *,
    amplitude_pct: float,
    alpha: float,
    lambda_um: float,
    k_ref: float = MATCHED_THETA_K_REF,
) -> np.ndarray:
    """R(%) = A * (k_r/k_ref)^alpha * exp(-l_r/lambda) with l_r in um."""
    k_arr = np.asarray(k_r, dtype=float)
    return amplitude_pct * (k_arr / k_ref) ** alpha * np.exp(-l_r_um / lambda_um)


def _fit_matched_theta_unified_model(
    valid: list[dict],
) -> tuple[float, float, float, float] | None:
    """Fit R = A*(k_r/k_ref)^alpha*exp(-l_r/lambda). Returns (A, alpha, lambda_um, rmse)."""
    if len(valid) < 5:
        return None
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    lr = np.array([float(r["l_r_median_um"]) for r in valid])
    rec = np.array([float(r["recovery_pct"]) for r in valid])
    kr = np.array([float(r["rate_mol_cm2_s"]) for r in valid])

    def model(x: tuple[np.ndarray, np.ndarray], log_a: float, alpha: float, lam: float) -> np.ndarray:
        l_r_um, k_r = x
        return _matched_theta_recovery_model(
            l_r_um,
            k_r,
            amplitude_pct=np.exp(log_a),
            alpha=alpha,
            lambda_um=lam,
        )

    try:
        popt, _ = curve_fit(
            model,
            (lr, kr),
            rec,
            p0=[np.log(20.0), 0.6, 130.0],
            bounds=([-2.0, 0.0, 20.0], [4.0, 2.0, 500.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None

    log_a, alpha, lam = (float(popt[0]), float(popt[1]), float(popt[2]))
    amplitude = float(np.exp(log_a))
    pred = model((lr, kr), log_a, alpha, lam)
    rmse = float(np.sqrt(np.mean((rec - pred) ** 2)))
    return amplitude, alpha, lam, rmse


def plot_matched_theta(rows: list[dict]) -> None:
    """Matched-Theta R vs l_r with unified R = A*(k_r/k_ref)^alpha*exp(-l_r/lambda)."""
    valid = _complete_rows(rows)
    branches = sorted({str(r["branch"]) for r in valid if r.get("branch")})
    if len(branches) < 1:
        log("SKIP matched_theta_recovery_vs_l_r: no complete matched-theta rows")
        return

    markers = {"4a": "o", "4b": "s", "4c": "D"}
    colors = {
        branch: color
        for branch, color in zip(("4a", "4b", "4c"), sns.color_palette("dark", n_colors=3))
    }
    lr_line = np.linspace(40.0, 120.0, 200)
    unified_fit = _fit_matched_theta_unified_model(valid)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, ax = plt.subplots(figsize=(7.6, 5.4), layout="constrained")
    legend_handles: list = []
    legend_rates: list[float] = []
    theta_vals: list[float] = []
    rec_all: list[float] = []

    for branch in branches:
        sub = [r for r in valid if str(r.get("branch")) == branch]
        if len(sub) < 2:
            log(f"SKIP matched-theta branch {branch}: <2 complete rows")
            continue
        lr = np.array([float(r["l_r_median_um"]) for r in sub])
        rec = np.array([float(r["recovery_pct"]) for r in sub])
        rate = float(sub[0]["rate_mol_cm2_s"])
        kr = format_k_r_latex(rate).strip("$")
        label = rf"$k_r={kr}$"
        color = colors.get(branch, "#555555")
        sc = ax.scatter(
            lr,
            rec,
            s=58,
            marker=markers.get(branch, "o"),
            color=color,
            edgecolors="0.15",
            linewidths=0.4,
            alpha=0.92,
            label=label,
            zorder=3,
        )
        legend_handles.append(sc)
        legend_rates.append(rate)
        rec_all.extend(rec.tolist())
        theta_vals.extend(
            float(r["theta_48h"])
            for r in sub
            if np.isfinite(float(r.get("theta_48h", np.nan)))
        )
        if unified_fit is not None:
            amplitude, alpha, lam_um, _rmse = unified_fit
            ax.plot(
                lr_line,
                _matched_theta_recovery_model(
                    lr_line,
                    rate,
                    amplitude_pct=amplitude,
                    alpha=alpha,
                    lambda_um=lam_um,
                ),
                "--",
                color=color,
                lw=1.8,
                alpha=0.9,
                zorder=2,
            )

    if not legend_handles:
        log("SKIP matched_theta_recovery_vs_l_r: insufficient branch data")
        plt.close(fig)
        return

    ax.set_xlabel(LBL_LR_MEDIAN, fontsize=MANUSCRIPT_AXIS_FS)
    ax.set_ylabel(recovery_label(), fontsize=MANUSCRIPT_AXIS_FS)
    ax.tick_params(axis="both", labelsize=MANUSCRIPT_TICK_FS)
    ax.set_xlim(38, 122)
    ax.set_ylim(bottom=0.0)
    order = sorted(range(len(legend_rates)), key=lambda i: legend_rates[i], reverse=True)
    ax.legend(
        handles=[legend_handles[i] for i in order],
        frameon=True,
        fontsize=MANUSCRIPT_LEGEND_FS - 1.0,
        loc="upper right",
    )
    if unified_fit is not None:
        amplitude, alpha, lam_um, rmse = unified_fit
        log(
            f"matched-theta fit A={amplitude:.2f}% alpha={alpha:.3f} "
            f"lambda={lam_um:.1f} um RMSE={rmse:.3f}%"
        )
    save_figure(fig, figures_dir() / "matched_theta_recovery_vs_l_r.png")


def cmd_geom_panels(geom_seeds: list[int] | None = None) -> None:
    """Write one 4x2 grid figure (four geoms, connected pore + chrysocolla per row)."""
    seeds = geom_seeds or default_geom_panel_seeds(4)
    if len(seeds) != 4:
        raise SystemExit(f"geom-panels expects exactly 4 geometry seeds, got {len(seeds)}")
    out_png = ANALYSIS_SHARED / "geom_panels_ab_grid.png"
    plot_geom_panels_grid(seeds, out_png)
    log(f"Geometry panels -> {out_png}")


def cmd_porosity_panels(
    geom_seeds: list[int] | None = None,
    *,
    dp: float = 1.0,
    rate: float = 1.0e-9,
) -> None:
    """SI grid: initial/final porosity for four SI example geometries (reference protocol)."""
    seeds = geom_seeds or default_geom_panel_seeds(4)
    if len(seeds) != 4:
        raise SystemExit(f"porosity-panels expects exactly 4 geometry seeds, got {len(seeds)}")

    cases = load_factorial_cases()
    rows: list[tuple[int, float, np.ndarray, np.ndarray, float, float]] = []
    for seed in seeds:
        case = next(
            (
                c
                for c in cases
                if c.geom_seed == int(seed)
                and abs(c.dP_Pa - dp) < 1e-12
                and abs(c.rate_mol_cm2_s - rate) < 1e-15
            ),
            None,
        )
        if case is None:
            raise SystemExit(f"No factorial case for geom={seed} dP={dp} rate={rate}")
        h5_path = case.case_dir() / "mineral_dissolution.h5"
        if not case_h5_complete(h5_path):
            raise SystemExit(f"Incomplete or missing: {h5_path}")
        with h5py.File(h5_path, "r") as h5:
            time_entries = parse_time_groups(h5)
            snaps = porosity_snapshots(h5, time_entries)
        if len(snaps) < 2:
            raise SystemExit(f"Need initial/final porosity in {h5_path}")
        por0 = np.asarray(snaps[0][1], dtype=float)
        porf = np.asarray(snaps[1][1], dtype=float)
        lr = _manifest_lr_um(seed)
        rows.append((int(seed), lr, por0, porf, float(np.mean(por0)), float(np.mean(porf))))

    phi_floor = 1.0e-2
    vmin = phi_floor
    vmax = 1.0
    norm = LogNorm(vmin=vmin, vmax=vmax)
    extent = (0.0, 8.0, 0.0, 2.0)  # Lx x Ly [mm]
    panel_letters = list("abcdefgh")

    def _as_yx(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        a = np.squeeze(a)
        if a.ndim != 2:
            raise ValueError(f"porosity field rank {a.ndim}")
        if a.shape[0] > a.shape[1]:
            return a.T
        return a

    def _log_safe(arr: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(arr, dtype=float), phi_floor, None)

    from mpl_toolkits.axes_grid1 import ImageGrid

    # ImageGrid: tight axes_pad + dedicated cbar axis (no overlap with panels)
    # Height leaves room for titles between rows (do not smush titles into axes)
    fig = plt.figure(figsize=(8.0, 5.6))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(4, 2),
        axes_pad=(0.14, 0.28),  # (w, h) inches — extra vertical pad for titles
        share_all=True,
        aspect=True,
        label_mode="L",
        cbar_mode="single",
        cbar_location="right",
        cbar_size="2.2%",
        cbar_pad=0.15,
    )

    im = None
    panel_i = 0
    for i, (seed, lr, por0, porf, m0, mf) in enumerate(rows):
        for j, (arr, tag, mean_phi) in enumerate(
            ((por0, "initial", m0), (porf, "final (10 d)", mf))
        ):
            ax = grid[panel_i]
            a_plot = _log_safe(_as_yx(arr))
            im = ax.imshow(
                a_plot,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="RdBu_r",
                norm=norm,
                interpolation="nearest",
            )
            letter = panel_letters[panel_i]
            panel_i += 1
            ax.set_title(
                rf"({letter})  geom{seed}  {tag}  "
                rf"$\langle\phi\rangle$={mean_phi:.3f}",
                fontsize=8,
                loc="left",
                pad=3.0,
            )
            ax.tick_params(labelsize=7, pad=1)
            if i == 3:
                ax.set_xlabel("$x$ [mm]", fontsize=8, labelpad=1)
            if j == 0:
                ax.set_ylabel("$y$ [mm]", fontsize=8, labelpad=1)

    cbar = grid.cbar_axes[0].colorbar(im)
    cbar.set_label(r"porosity [-] ($\log_{10}$ scale)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    tick_vals = np.array(
        [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    )
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(
        [
            r"$10^{-2}$",
            r"$2\times10^{-2}$",
            r"$3\times10^{-2}$",
            r"$5\times10^{-2}$",
            r"$7\times10^{-2}$",
            r"$10^{-1}$",
            r"$2\times10^{-1}$",
            r"$3\times10^{-1}$",
            r"$5\times10^{-1}$",
            r"$7\times10^{-1}$",
            r"$10^{0}$",
        ]
    )
    cbar.ax.minorticks_off()

    out_png = ANALYSIS_SHARED / "porosity_init_final_four_geoms.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    log(f"Porosity init/final panels -> {out_png}")


def plot_case_figures(geom_seed: int, dp: float, rate: float) -> None:
    """Per-case dissolution maps via plot_accessibility_mineral_dissolution.py."""
    cases = load_factorial_cases()
    case = next(
        (
            c
            for c in cases
            if c.geom_seed == int(geom_seed)
            and abs(c.dP_Pa - dp) < 1e-12
            and abs(c.rate_mol_cm2_s - rate) < 1e-15
        ),
        None,
    )
    if case is None:
        raise SystemExit(f"No factorial case for geom={geom_seed} dP={dp} rate={rate}")
    h5 = case.case_dir() / "mineral_dissolution.h5"
    if not case_h5_complete(h5):
        raise SystemExit(f"Incomplete or missing: {h5}")
    figs = case.case_dir() / "figs"
    prefix = f"geom{geom_seed}_dp{tag_dp(dp)}_rate{tag_rate(rate)}"
    subprocess.run(
        [
            sys.executable,
            str(SRC_DIR / "plot_accessibility_mineral_dissolution.py"),
            str(h5),
            "--out-dir",
            str(figs),
            "--prefix",
            prefix,
            "--dP",
            str(dp),
            "--lx",
            str(LX),
            "--ly",
            str(LY),
            "--lz",
            str(LZ),
        ],
        check=True,
    )
    log(f"Case figs -> {figs}")


def cmd_tables(*, refresh: bool) -> None:
    build_factorial_table(refresh=refresh)
    build_matched_theta_table(refresh=refresh)


def cmd_factorial(*, refresh: bool) -> None:
    rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    plot_factorial_heatmaps(rows)
    plot_factorial_ensemble_heatmaps(rows)
    plot_factorial_recovery_grid(rows)
    plot_factorial_regime(rows)


def cmd_factorial_violin(*, refresh: bool) -> None:
    rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    plot_factorial_recovery_violin_hue(rows)


def cmd_factorial_pe_da_3d(*, refresh: bool) -> None:
    rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    plot_factorial_recovery_pe_da_3d(rows)


def cmd_factorial_pe_da_3d_plotly(*, refresh: bool) -> None:
    rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    plot_factorial_recovery_pe_da_3d_plotly(rows)


def cmd_outlet_btc() -> None:
    plot_factorial_outlet_cu_btc_grid()


def cmd_anchor(*, refresh: bool) -> None:
    rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    plot_anchor_recovery(rows)


def cmd_matched_theta(*, refresh: bool) -> None:
    rows = apply_recovery_metric(build_matched_theta_table(refresh=refresh))
    plot_matched_theta(rows)


def cmd_pe_da(*, refresh: bool) -> None:
    factorial_rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    matched_rows = apply_recovery_metric(build_matched_theta_table(refresh=refresh))
    plot_pe_da_regime(factorial_rows, matched_rows)


def cmd_collapse(*, refresh: bool) -> None:
    factorial_rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    matched_rows = apply_recovery_metric(build_matched_theta_table(refresh=refresh))
    rows = _all_collapse_rows(factorial_rows, matched_rows)
    if len(rows) < 2:
        log("SKIP collapse: insufficient complete rows")
        return
    plot_theta_cu_access_collapse(rows)


def cmd_collapse_compare(*, refresh: bool) -> None:
    factorial_rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    matched_rows = apply_recovery_metric(build_matched_theta_table(refresh=refresh))
    rows = _all_collapse_rows(factorial_rows, matched_rows)
    compare_rows = build_collapse_compare_table(rows)
    if not compare_rows:
        return
    write_collapse_compare_table(compare_rows)


def cmd_recovery_lr(*, refresh: bool) -> None:
    factorial_rows = apply_recovery_metric(build_factorial_table(refresh=refresh))
    matched_rows = apply_recovery_metric(build_matched_theta_table(refresh=refresh))
    plot_recovery_vs_l_r_all(factorial_rows, matched_rows)


def cmd_rs_rout(*, refresh: bool) -> None:
    factorial_rows = build_factorial_table(refresh=refresh)
    matched_rows = build_matched_theta_table(refresh=refresh)
    plot_rs_vs_rout(factorial_rows, matched_rows)


def cmd_paper(*, refresh: bool) -> None:
    """Manuscript figure set for the active --metric."""
    cmd_factorial_violin(refresh=refresh)
    cmd_recovery_lr(refresh=refresh)
    cmd_collapse(refresh=refresh)
    cmd_matched_theta(refresh=refresh)
    cmd_pe_da(refresh=refresh)


def _cmd_metric_figures(*, refresh: bool) -> None:
    """Full recovery-dependent figure set (no tables, shared, or case figs)."""
    cmd_factorial(refresh=refresh)
    cmd_factorial_violin(refresh=refresh)
    cmd_factorial_pe_da_3d(refresh=refresh)
    cmd_anchor(refresh=refresh)
    cmd_matched_theta(refresh=refresh)
    cmd_collapse(refresh=refresh)
    cmd_collapse_compare(refresh=refresh)
    cmd_recovery_lr(refresh=refresh)
    cmd_pe_da(refresh=refresh)


def cmd_all(*, refresh: bool) -> None:
    cmd_tables(refresh=refresh)
    _cmd_metric_figures(refresh=refresh)
    cmd_geom_panels(None)
    cmd_outlet_btc()


def cmd_all_metrics(*, refresh: bool) -> None:
    """Tables once, shared figures once, then full sets for solid_phase and outlet."""
    cmd_tables(refresh=refresh)
    cmd_geom_panels(None)
    cmd_outlet_btc()
    for metric in RECOVERY_METRICS:
        set_active_metric(metric)
        log(f"=== recovery metric: {metric} -> {figures_dir()} ===")
        _cmd_metric_figures(refresh=refresh)
    # Main-text dual panels (Fig. 3 violin dual, Fig. 5 collapse dual).
    cmd_combined_figures(refresh=refresh)


def cmd_combined_figures(*, refresh: bool) -> None:
    """Dual-metric manuscript panels under analysis/combined/."""
    factorial_rows = build_factorial_table(refresh=refresh)
    matched_rows = build_matched_theta_table(refresh=refresh)
    plot_dual_factorial_recovery_violin_hue(factorial_rows)
    plot_dual_theta_cu_access_collapse(factorial_rows, matched_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot accessibility_ensemble campaign results.")
    parser.add_argument("--refresh", action="store_true", help="Re-scan H5 files (ignore cached CSV).")
    parser.add_argument(
        "--metric",
        choices=RECOVERY_METRICS,
        default="solid_phase",
        help="Recovery definition for figure outputs (default: solid_phase).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tables", help="Write factorial_metrics.csv and matched_theta_metrics.csv.")
    sub.add_parser(
        "factorial",
        help="Factorial heatmaps, 3x3 grids, and regime figure.",
    )
    sub.add_parser(
        "outlet-btc",
        help="3x3 flux-weighted outlet Cu BTC (25 lines per panel, color = l_r).",
    )
    sub.add_parser("anchor", help="Cu recovery vs l_r at Delta p=1 Pa, k_r=1e-9.")
    sub.add_parser("matched-theta", help="Matched-theta 4a vs 4b figure.")
    sub.add_parser(
        "collapse",
        help="R vs Theta*f_Cu,acc for all factorial and matched-theta cases.",
    )
    sub.add_parser(
        "collapse-compare",
        help="SI table comparing exposure metrics for predicting R (no figure).",
    )
    sub.add_parser(
        "recovery-lr",
        help="R vs l_r for all factorial and matched-theta cases.",
    )
    sub.add_parser(
        "rs-rout",
        help="Ensemble scatter of R_out versus R_s (Set A + Set B).",
    )
    sub.add_parser("pe-da", help="Pe_r-Da_r regime figure (factorial + matched-theta).")
    sub.add_parser(
        "factorial-violin",
        help="Violin plot of R vs dP with hue = k_r (factorial ensemble).",
    )
    sub.add_parser(
        "factorial-pe-da-3d",
        help="3D R vs log10(Pe_r) and log10(Da_r) for factorial sweep.",
    )
    sub.add_parser(
        "factorial-pe-da-3d-plotly",
        help="Interactive Plotly 3D Pe_r-Da_r-R scatter for factorial sweep.",
    )
    sub.add_parser("all", help="tables + shared + full figure set for --metric.")
    sub.add_parser("paper", help="Manuscript figure set for --metric.")
    sub.add_parser(
        "all-metrics",
        help="tables + shared + full figure sets for solid_phase and outlet.",
    )
    sub.add_parser(
        "combined-figures",
        help="Dual-metric manuscript panels (violin + collapse) under analysis/combined/.",
    )
    p_geom = sub.add_parser(
        "geom-panels",
        help="Pore geometry panels (a-b): connected pore + chrysocolla inventory.",
    )
    p_geom.add_argument(
        "--geoms",
        default="",
        help="Comma-separated four geometry seeds (default: four l_r ranks from manifest).",
    )
    p_por = sub.add_parser(
        "porosity-panels",
        help="SI: initial/final porosity for four SI example geometries (reference protocol).",
    )
    p_por.add_argument(
        "--geoms",
        default="",
        help="Comma-separated four geometry seeds (default: same ranks as geom-panels).",
    )
    p_por.add_argument("--dp", type=float, default=1.0, help="Factorial dP [Pa] (default 1).")
    p_por.add_argument(
        "--rate",
        type=float,
        default=1.0e-9,
        help="Factorial rate [mol/cm2-s] (default 1e-9).",
    )
    p_case = sub.add_parser("case-figs", help="Per-case dissolution maps for one factorial cell.")
    p_case.add_argument("--geom", type=int, required=True)
    p_case.add_argument("--dp", type=float, required=True)
    p_case.add_argument("--rate", type=float, required=True)
    args = parser.parse_args()

    ensure_analysis_dirs()
    if args.command != "all-metrics":
        set_active_metric(args.metric)
    if args.command == "tables":
        cmd_tables(refresh=args.refresh)
    elif args.command == "factorial":
        cmd_factorial(refresh=args.refresh)
    elif args.command == "outlet-btc":
        cmd_outlet_btc()
    elif args.command == "anchor":
        cmd_anchor(refresh=args.refresh)
    elif args.command == "matched-theta":
        cmd_matched_theta(refresh=args.refresh)
    elif args.command == "collapse":
        cmd_collapse(refresh=args.refresh)
    elif args.command == "collapse-compare":
        cmd_collapse_compare(refresh=args.refresh)
    elif args.command == "recovery-lr":
        cmd_recovery_lr(refresh=args.refresh)
    elif args.command == "rs-rout":
        cmd_rs_rout(refresh=args.refresh)
    elif args.command == "pe-da":
        cmd_pe_da(refresh=args.refresh)
    elif args.command == "factorial-violin":
        cmd_factorial_violin(refresh=args.refresh)
    elif args.command == "factorial-pe-da-3d":
        cmd_factorial_pe_da_3d(refresh=args.refresh)
    elif args.command == "factorial-pe-da-3d-plotly":
        cmd_factorial_pe_da_3d_plotly(refresh=args.refresh)
    elif args.command == "all":
        cmd_all(refresh=args.refresh)
    elif args.command == "paper":
        cmd_paper(refresh=args.refresh)
    elif args.command == "all-metrics":
        cmd_all_metrics(refresh=args.refresh)
    elif args.command == "combined-figures":
        cmd_combined_figures(refresh=args.refresh)
    elif args.command == "geom-panels":
        seeds = (
            [int(x.strip()) for x in args.geoms.split(",") if x.strip()]
            if args.geoms.strip()
            else None
        )
        cmd_geom_panels(seeds)
    elif args.command == "porosity-panels":
        seeds = (
            [int(x.strip()) for x in args.geoms.split(",") if x.strip()]
            if args.geoms.strip()
            else None
        )
        cmd_porosity_panels(seeds, dp=args.dp, rate=args.rate)
    elif args.command == "case-figs":
        plot_case_figures(args.geom, args.dp, args.rate)


if __name__ == "__main__":
    main()