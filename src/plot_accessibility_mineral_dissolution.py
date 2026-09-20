"""Plot chrysocolla dissolution from PFLOTRAN HDF5 output.

Outputs:
  <prefix>_inventory.png
  <prefix>_total_recovery.png
  <prefix>_depletion_snapshots.png : Chrysocolla depletion (VF0 - VF); t=0, 8-48 h, then 48-240 h
  <prefix>_porosity_snapshots.png : Initial and final porosity (log10 color scale)
  <prefix>_mineral_inventory.md : Initial/final mineral, flow, porosity, and permeability inventory

Example:
    python plot_accessibility_mineral_dissolution.py \
    Chip_Designs/multi_realization_slurm/pflotran_out_15June_dP_1_pulse_8_sim_1week/chrysocolla_real000/mineral_dissolution.h5 \
      --out-dir Chip_Designs/multi_realization_slurm/pflotran_out_15June_dP_1_pulse_8_sim_1week/chrysocolla_real000/figs \
      --prefix chrysocolla_real000

    python plot_accessibility_mineral_dissolution.py \
    Chip_Designs/bean_bucket_network/pflotran_out_dp_0.001/chrysocolla_real000/mineral_dissolution.h5 \
      --out-dir Chip_Designs/bean_bucket_network/pflotran_out_dp_0.001/chrysocolla_real000/figs \
      --prefix chrysocolla_real000 --dP 0.001 

"""

from __future__ import annotations

import argparse
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plot_recovery_metrics import (
    CHRYSOCOLLA2_CU_STOICH,
    CHRYSOCOLLA2_MOLAR_VOLUME_CM3_MOL,
    cumulative_metrics,
    load_edges,
    outlet_inlet_flux_metrics,
    parse_time_groups,
    safe_cell_widths,
)

matplotlib.use("Agg")


CHRYSOCOLLA_FIELD = "Chrysocolla2_VF"
POROSITY_FIELD = "Porosity"
WATER_MU_PA_S = 1.0e-3


def pick_field(group: h5py.Group, prefix: str) -> str:
    """Find a PFLOTRAN field by exact prefix."""
    for key in group.keys():
        if key.startswith(prefix):
            return key
    raise KeyError(f"Field starting with {prefix!r} not found in {group.name}")


def select_snapshot_indices(n: int, count: int = 4) -> list[int]:
    """Return count uniformly spaced indices including first and last (repo convention)."""
    if n <= 0:
        return []
    if n <= count:
        return list(range(n))
    idx = np.linspace(0, n - 1, count)
    return sorted(set(int(round(i)) for i in idx))


def snapshot_target_times_h(
    pulse_duration_s: float,
    final_time_s: float,
    pulse_interval_h: float = 8.0,
    pulse_snapshots: int = 6,
    post_pulse_snapshots: int = 5,
) -> list[float]:
    """12-panel schedule: t=0, pulse_snapshots during pulse, post_pulse_snapshots over remainder."""
    pulse_h = pulse_duration_s / 3600.0
    final_h = final_time_s / 3600.0
    pulse_times = [
        t
        for t in np.arange(pulse_interval_h, pulse_h + 0.5 * pulse_interval_h, pulse_interval_h)
    ][:pulse_snapshots]
    times = [0.0, *pulse_times]
    if post_pulse_snapshots > 0 and final_h > pulse_h:
        post_times = np.linspace(pulse_h, final_h, post_pulse_snapshots + 1)[1:].tolist()
        times.extend(post_times)
    return times


def nearest_time_indices(time_entries: list[tuple[float, str]], target_times_h: list[float]) -> list[int]:
    """Map target snapshot times [h] to nearest PFLOTRAN output indices."""
    times_s = np.asarray([t for t, _ in time_entries])
    return [int(np.argmin(np.abs(times_s - t_h * 3600.0))) for t_h in target_times_h]


def cell_volumes(h5: h5py.File, nx: int, ny: int, lx: float, ly: float, lz: float) -> np.ndarray:
    """Return cell volume array [m3]."""
    x_edges, y_edges, z_edges = load_edges(h5)
    dx = safe_cell_widths(x_edges, nx, lx, axis_label="X")
    dy = safe_cell_widths(y_edges, ny, ly, axis_label="Y")
    dz = np.sum(safe_cell_widths(z_edges, 1, lz, axis_label="Z"))
    return dx[:, None] * dy[None, :] * dz


def chrysocolla_inventory_series(
    h5: h5py.File,
    time_entries: list[tuple[float, str]],
    lx: float,
    ly: float,
    lz: float,
) -> dict[str, np.ndarray]:
    """Compute Cu remaining and dissolved for Chrysocolla2."""
    first_group = h5[time_entries[0][1]]
    first_key = pick_field(first_group, CHRYSOCOLLA_FIELD)
    nx, ny = np.asarray(first_group[first_key][:, :, 0]).shape
    cell_vol = cell_volumes(h5, nx, ny, lx=lx, ly=ly, lz=lz)
    molar_volume_m3_mol = CHRYSOCOLLA2_MOLAR_VOLUME_CM3_MOL * 1.0e-6

    remaining = []
    times = []
    for t_s, group_name in time_entries:
        group = h5[group_name]
        times.append(t_s)
        field = pick_field(group, CHRYSOCOLLA_FIELD)
        vf = np.asarray(group[field][:, :, 0])
        mineral_mol = vf * cell_vol / molar_volume_m3_mol
        cu_mol = float(np.sum(CHRYSOCOLLA2_CU_STOICH * mineral_mol))
        remaining.append(cu_mol)

    result: dict[str, np.ndarray] = {"t_s": np.asarray(times)}
    total_remaining = np.asarray(remaining)
    total_initial = float(total_remaining[0])
    result["total_remaining_mol"] = total_remaining
    result["total_dissolved_mol"] = total_initial - total_remaining
    result["total_recovery_pct"] = 100.0 * result["total_dissolved_mol"] / max(total_initial, 1.0e-30)
    return result


def mean_field(group: h5py.Group, field_prefix: str) -> float:
    """Return domain mean for a PFLOTRAN cell field."""
    key = pick_field(group, field_prefix)
    data = np.asarray(group[key])
    if data.ndim == 3:
        data = data[:, :, 0]
    return float(np.nanmean(data))


def domain_lengths(h5: h5py.File, lx: float, ly: float, lz: float) -> tuple[float, float, float]:
    """Read domain lengths from coordinates, with CLI values as fallback."""
    x_edges, y_edges, z_edges = load_edges(h5)
    lengths = []
    for edges, fallback in ((x_edges, lx), (y_edges, ly), (z_edges, lz)):
        if len(edges) >= 2 and np.isfinite(edges[-1] - edges[0]) and edges[-1] > edges[0]:
            lengths.append(float(edges[-1] - edges[0]))
        else:
            lengths.append(float(fallback))
    return tuple(lengths)


def effective_permeability_from_q(q_m3_s: float, lx: float, ly: float, lz: float, dP: float) -> float:
    """Compute whole-domain effective permeability from Darcy's law."""
    area = ly * lz
    darcy_velocity = abs(q_m3_s) / area
    return WATER_MU_PA_S * darcy_velocity * lx / dP


def collect_inventory_summary(
    h5: h5py.File,
    h5_file: Path,
    time_entries: list[tuple[float, str]],
    series: dict[str, np.ndarray],
    lx: float,
    ly: float,
    lz: float,
    dP: float,
) -> dict[str, float | str]:
    """Collect one-row mineral and hydraulic inventory metrics."""
    first_group = h5[time_entries[0][1]]
    last_group = h5[time_entries[-1][1]]
    domain_lx, domain_ly, domain_lz = domain_lengths(h5, lx, ly, lz)
    cu_key = pick_field(first_group, "Total_Cu++")
    h_key = pick_field(first_group, "Free_H+")
    flux = outlet_inlet_flux_metrics(h5, time_entries, cu_key, h_key, ly=domain_ly, lz=domain_lz)
    collected = cumulative_metrics(flux, float(series["total_remaining_mol"][0]), 1.0)["cu_out_cum"]

    if len(flux["Qout"]):
        nonzero = np.flatnonzero(np.abs(flux["Qout"]) > 0.0)
        flow0_idx = int(nonzero[0]) if len(nonzero) else 0
    else:
        flow0_idx = 0
    q0 = float(flux["Qout"][flow0_idx]) if len(flux["Qout"]) else float("nan")
    qf = float(flux["Qout"][-1]) if len(flux["Qout"]) else float("nan")
    return {
        "h5_file": str(h5_file),
        "initial_time_h": float(time_entries[0][0] / 3600.0),
        "final_time_h": float(time_entries[-1][0] / 3600.0),
        "initial_flow_time_h": float(flux["t_s"][flow0_idx] / 3600.0) if len(flux["t_s"]) else float("nan"),
        "initial_chrysocolla_cu_mol": float(series["total_remaining_mol"][0]),
        "dissolved_chrysocolla_cu_mol": float(series["total_dissolved_mol"][-1]),
        "remaining_chrysocolla_cu_mol": float(series["total_remaining_mol"][-1]),
        "total_recovery_pct": float(series["total_recovery_pct"][-1]),
        "total_collected_outlet_cu_mol": float(collected[-1]) if len(collected) else float("nan"),
        "initial_avg_porosity": mean_field(first_group, "Porosity"),
        "final_avg_porosity": mean_field(last_group, "Porosity"),
        "initial_avg_permeability_m2": mean_field(first_group, "Permeability"),
        "final_avg_permeability_m2": mean_field(last_group, "Permeability"),
        "initial_outlet_flow_uL_min": q0 * 60.0e9,
        "final_outlet_flow_uL_min": qf * 60.0e9,
        "initial_effective_permeability_m2": effective_permeability_from_q(q0, domain_lx, domain_ly, domain_lz, dP),
        "final_effective_permeability_m2": effective_permeability_from_q(qf, domain_lx, domain_ly, domain_lz, dP),
        "pressure_drop_pa": dP,
        "domain_lx_m": domain_lx,
        "domain_ly_m": domain_ly,
        "domain_lz_m": domain_lz,
    }


def format_value(value: float | str) -> str:
    """Format report values consistently."""
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6e}"


def write_inventory_markdown(summary: dict[str, float | str], out_md: Path) -> None:
    """Write a compact Markdown inventory report."""
    rows = [
        ("Initial chrysocolla Cu", "mol Cu", "initial_chrysocolla_cu_mol"),
        ("Dissolved chrysocolla Cu", "mol Cu", "dissolved_chrysocolla_cu_mol"),
        ("Remaining chrysocolla Cu", "mol Cu", "remaining_chrysocolla_cu_mol"),
        ("Domain Cu recovery", "%", "total_recovery_pct"),
        ("Total Cu collected at outlet", "mol Cu", "total_collected_outlet_cu_mol"),
        ("Initial average porosity", "-", "initial_avg_porosity"),
        ("Final average porosity", "-", "final_avg_porosity"),
        ("Initial average permeability", "m2", "initial_avg_permeability_m2"),
        ("Final average permeability", "m2", "final_avg_permeability_m2"),
        ("Initial outlet flow rate", "micro L/min", "initial_outlet_flow_uL_min"),
        ("Final outlet flow rate", "micro L/min", "final_outlet_flow_uL_min"),
        ("Initial effective permeability", "m2", "initial_effective_permeability_m2"),
        ("Final effective permeability", "m2", "final_effective_permeability_m2"),
    ]
    lines = [
        "# Mineral Inventory",
        "",
        f"- PFLOTRAN output: `{summary['h5_file']}`",
        f"- Time window: {format_value(summary['initial_time_h'])} to {format_value(summary['final_time_h'])} h",
        f"- Initial hydraulic sample time: {format_value(summary['initial_flow_time_h'])} h",
        f"- Domain length used: {format_value(summary['domain_lx_m'])} x {format_value(summary['domain_ly_m'])} x {format_value(summary['domain_lz_m'])} m",
        f"- Pressure drop used for effective permeability: {format_value(summary['pressure_drop_pa'])} Pa",
        "",
        "| Metric | Value | Unit |",
        "|---|---:|---|",
    ]
    for label, unit, key in rows:
        lines.append(f"| {label} | {format_value(summary[key])} | {unit} |")
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- Chrysocolla inventory is reported as Cu moles from Chrysocolla2 volume fraction and PFLOTRAN database molar volume.",
            "- Dissolved chrysocolla Cu = initial chrysocolla Cu - final chrysocolla Cu.",
            "- Total Cu collected at outlet is flux-integrated outlet Cu, not corrected by inlet Cu.",
            "- Effective permeability uses k_eff = mu * (Q_out / A) * L / dP with mu = 1.0e-3 Pa s.",
            "- Domain Cu recovery = (Dissolved chrysocolla Cu / Initial chrysocolla Cu) * 100%.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n")


def snapshot_schedule_kwargs(
    pulse_duration_s: float,
    final_time_s: float,
    pulse_interval_h: float,
    pulse_snapshots: int,
    post_pulse_snapshots: int,
) -> dict[str, float | int]:
    """Bundle args for snapshot_target_times_h."""
    return {
        "pulse_duration_s": pulse_duration_s,
        "final_time_s": final_time_s,
        "pulse_interval_h": pulse_interval_h,
        "pulse_snapshots": pulse_snapshots,
        "post_pulse_snapshots": post_pulse_snapshots,
    }


def chrysocolla_depletion_snapshots(
    h5: h5py.File,
    time_entries: list[tuple[float, str]],
    pulse_duration_s: float,
    pulse_interval_h: float = 8.0,
    pulse_snapshots: int = 6,
    post_pulse_snapshots: int = 5,
) -> list[tuple[float, np.ndarray]]:
    """Return [(t_h, delta_VF)] depletion snapshots on 12-panel schedule."""
    first = h5[time_entries[0][1]]
    key0 = pick_field(first, CHRYSOCOLLA_FIELD)
    vf0 = np.asarray(first[key0][:, :, 0])
    schedule = snapshot_schedule_kwargs(
        pulse_duration_s,
        float(time_entries[-1][0]),
        pulse_interval_h,
        pulse_snapshots,
        post_pulse_snapshots,
    )
    target_times_h = snapshot_target_times_h(**schedule)
    idcs = nearest_time_indices(time_entries, target_times_h)
    snaps: list[tuple[float, np.ndarray]] = []
    for ii in idcs:
        t_s, gname = time_entries[ii]
        g = h5[gname]
        k = pick_field(g, CHRYSOCOLLA_FIELD)
        vf = np.asarray(g[k][:, :, 0])
        delta = vf0 - vf
        snaps.append((t_s / 3600.0, delta))
    return snaps


def porosity_snapshots(
    h5: h5py.File,
    time_entries: list[tuple[float, str]],
) -> list[tuple[float, np.ndarray]]:
    """Return initial and final [(t_h, porosity)] snapshots."""
    snaps: list[tuple[float, np.ndarray]] = []
    for ii in (0, len(time_entries) - 1):
        t_s, gname = time_entries[ii]
        g = h5[gname]
        k = pick_field(g, POROSITY_FIELD)
        por = np.asarray(g[k][:, :, 0])
        snaps.append((t_s / 3600.0, por))
    return snaps


def plot_inventory(series: dict[str, np.ndarray], out_png: Path) -> None:
    """Plot remaining and dissolved Cu."""
    t_h = series["t_s"] / 3600.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(t_h, series["total_remaining_mol"] * 1.0e6, color="black", lw=1.5, label="total")
    axes[1].plot(t_h, series["total_dissolved_mol"] * 1.0e6, color="black", lw=1.5, label="total")
    axes[0].set_title("(a) Cu remaining in chrysocolla", loc="left")
    axes[1].set_title("(b) Cu dissolved from chrysocolla", loc="left")
    axes[0].set_ylabel("Cu [umol]")
    axes[1].set_ylabel("Cu [umol]")
    for ax in axes:
        ax.set_xlabel("Time [h]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_recovery(series: dict[str, np.ndarray], out_png: Path) -> None:
    """Plot domain Cu recovery from dissolved chrysocolla inventory."""
    t_h = series["t_s"] / 3600.0
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    ax.plot(t_h, series["total_recovery_pct"], color="black", lw=1.8)
    ax.set_title("Domain Cu recovery", loc="left")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Recovery [% initial solid Cu]")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_spatial_snapshots(
    snaps: list[tuple[float, np.ndarray]],
    out_png: Path,
    *,
    cmap: str,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
    ncols: int | None = None,
    nrows: int | None = None,
    figsize: tuple[float, float] | None = None,
    aspect: str = "equal",
    extent_mm: tuple[float, float, float, float] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_axis_ticks: bool = False,
) -> None:
    """Plot spatial snapshots with one shared colorbar.

    extent_mm: optional (x0, x1, y0, y1) in mm for imshow; preserves true domain aspect.
    """
    import math

    n = len(snaps)
    if n == 0:
        return

    if ncols is None and nrows is None:
        ncols = 2
        nrows = math.ceil(n / ncols)
    elif ncols is None:
        ncols = math.ceil(n / nrows)
    elif nrows is None:
        nrows = math.ceil(n / ncols)

    if figsize is None:
        figsize = (4.5 * ncols, 4.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    active_axes = [axes[i] for i in range(n)]

    stacked = np.stack([arr for _, arr in snaps])
    if vmax is None:
        vmax = float(np.nanmax(stacked))
    if vmin is None:
        vmin = float(np.nanmin(stacked))
    if log_scale:
        positive = stacked[stacked > 0.0]
        vmin = float(np.nanmin(positive)) if positive.size else 1.0e-6
        vmin = max(vmin, 1.0e-6)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0e-12
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = None
    for i, (t_h, arr) in enumerate(snaps):
        ax = axes[i]
        im = ax.imshow(
            arr.T,
            origin="lower",
            cmap=cmap,
            norm=norm,
            aspect=aspect,
            interpolation="none",
            extent=extent_mm,
        )
        ax.set_title(f"t = {t_h:.1f} h", loc="center", fontsize=12, pad=1.2)
        if extent_mm is None or not show_axis_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if extent_mm is not None:
            if xlabel and (i // ncols == nrows - 1 or nrows == 1):
                ax.set_xlabel(xlabel)
            if ylabel and (i % ncols == 0):
                ax.set_ylabel(ylabel)
        ax.set_frame_on(extent_mm is None)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    use_tight = show_axis_ticks and extent_mm is not None
    if use_tight:
        fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
        cbar_shrink, cbar_pad, cbar_aspect = 0.95, 0.02, 22.0
    elif ncols == 1:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02, wspace=0.12, hspace=0.1)
        cbar_shrink, cbar_pad, cbar_aspect = 0.88, 0.03, 18.0
    else:
        fig.subplots_adjust(left=0.01, right=0.990, top=0.95, bottom=0.02, wspace=0.12, hspace=0.1)
        cbar_shrink, cbar_pad, cbar_aspect = 0.82, 0.02, 22.0

    if im is not None:
        fig.colorbar(
            im,
            ax=active_axes,
            shrink=cbar_shrink,
            pad=cbar_pad,
            aspect=cbar_aspect,
            label=colorbar_label,
        )

    save_kwargs = {"dpi": 300, "pad_inches": 0.08 if use_tight else 0.03}
    if use_tight:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(out_png, **save_kwargs)
    plt.close(fig)


def plot_depletion_snapshots(
    snaps: list[tuple[float, np.ndarray]],
    out_png: Path,
    *,
    extent_mm: tuple[float, float, float, float] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_axis_ticks: bool = False,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot chrysocolla depletion snapshots in two columns."""
    plot_spatial_snapshots(
        snaps,
        out_png,
        cmap="magma",
        colorbar_label="depletion (VF0 - VF) [m³/m³ bulk]",
        vmin=0.0,
        ncols=2,
        figsize=figsize or (9.5, 8.0),
        extent_mm=extent_mm,
        xlabel=xlabel,
        ylabel=ylabel,
        show_axis_ticks=show_axis_ticks,
    )


def plot_porosity_snapshots(
    snaps: list[tuple[float, np.ndarray]],
    out_png: Path,
    *,
    extent_mm: tuple[float, float, float, float] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_axis_ticks: bool = False,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot initial/final porosity in two rows on a log10 color scale."""
    plot_spatial_snapshots(
        snaps,
        out_png,
        cmap="RdBu_r",
        colorbar_label="porosity [-] (log10 scale)",
        log_scale=True,
        ncols=1,
        nrows=2,
        figsize=figsize or (10, 5),
        aspect="equal",
        extent_mm=extent_mm,
        xlabel=xlabel,
        ylabel=ylabel,
        show_axis_ticks=show_axis_ticks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Chrysocolla2 dissolution.")
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="chrysocolla")
    parser.add_argument("--inventory-md", type=Path, default=None)
    parser.add_argument("--dP", type=float, default=1.0)
    parser.add_argument("--pulse-duration-s", type=float, default=48.0 * 3600.0)
    parser.add_argument("--pulse-snapshot-interval-h", type=float, default=8.0)
    parser.add_argument("--pulse-snapshots", type=int, default=6)
    parser.add_argument("--post-pulse-snapshots", type=int, default=5)
    parser.add_argument("--lx", type=float, default=1.0e-3)
    parser.add_argument("--ly", type=float, default=1.0e-3)
    parser.add_argument("--lz", type=float, default=1.0e-3)
    args = parser.parse_args()

    inventory_md = args.inventory_md or args.out_dir / f"{args.prefix}_mineral_inventory.md"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5_file, "r") as h5:
        time_entries = parse_time_groups(h5)
        if not time_entries:
            raise RuntimeError(f"No PFLOTRAN time groups found in {args.h5_file}")
        series = chrysocolla_inventory_series(h5, time_entries, lx=args.lx, ly=args.ly, lz=args.lz)
        schedule_kwargs = {
            "pulse_duration_s": args.pulse_duration_s,
            "pulse_interval_h": args.pulse_snapshot_interval_h,
            "pulse_snapshots": args.pulse_snapshots,
            "post_pulse_snapshots": args.post_pulse_snapshots,
        }
        depletion_snaps = chrysocolla_depletion_snapshots(h5, time_entries, **schedule_kwargs)
        porosity_snaps = porosity_snapshots(h5, time_entries)
        summary = collect_inventory_summary(
            h5,
            args.h5_file,
            time_entries,
            series,
            lx=args.lx,
            ly=args.ly,
            lz=args.lz,
            dP=args.dP,
        )

    plot_inventory(series, args.out_dir / f"{args.prefix}_inventory.png")
    plot_recovery(series, args.out_dir / f"{args.prefix}_total_recovery.png")
    plot_depletion_snapshots(depletion_snaps, args.out_dir / f"{args.prefix}_depletion_snapshots.png")
    plot_porosity_snapshots(porosity_snaps, args.out_dir / f"{args.prefix}_porosity_snapshots.png")
    inventory_md.parent.mkdir(parents=True, exist_ok=True)
    write_inventory_markdown(summary, inventory_md)
    print(f"Final domain Cu recovery: {series['total_recovery_pct'][-1]:.3f}%")
    print(f"Wrote mineral inventory: {inventory_md}")


if __name__ == "__main__":
    main()
