"""Generate Cu leaching recovery-focused figures from PFLOTRAN HDF5 output.

Outputs:
  1) <prefix>_btc.png      : Outlet flux-averaged Cu and free-acid pH vs time
  2) <prefix>_recovery.png : Cumulative Cu, recovery fraction, acid efficiency
  3) <prefix>_chrysocolla_cu.png : Cu dissolved from and remaining in Chrysocolla2

Example:
python plot_recovery_metrics.py cu_leaching_pore.h5 --prefix cu_leach --geometry-h5 pore_geometry.h5

"""

import argparse
import re
from pathlib import Path
from typing import Any, Optional

import _repo  # noqa: F401 — path bootstrap

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")


DEFAULT_LX_M = 1.0e-3
DEFAULT_LY_M = 1.0e-3
DEFAULT_LZ_M = 1.0e-3
DEFAULT_GEOMETRY_H5 = "pore_geometry.h5"

# Chrysocolla2 database values from database/hanford.dat.
# PFLOTRAN stores mineral VF as volume fraction, so convert to moles using the
# mineral molar volume rather than an approximate bulk density.
CHRYSOCOLLA2_MOLAR_VOLUME_CM3_MOL = 73.1900
CHRYSOCOLLA2_CU_STOICH = 1.0


FALLBACK_WARNED = set()

def format_axis(ax: Any) -> None:
    """Show full axis numbers (avoid offset notation like 1e-5 + x)."""
    for axis in (ax.xaxis, ax.yaxis):
        formatter = axis.get_major_formatter()
        if isinstance(formatter, ScalarFormatter):
            formatter.set_useOffset(False)
            formatter.set_scientific(False)


def parse_time_groups(h5: Any) -> list[tuple[float, str]]:
    """Read PFLOTRAN time groups and convert times to seconds.

    Returns a sorted list of tuples: (time_in_seconds, hdf5_group_name).
    """
    pat = re.compile(r"Time:\s+([\d.Ee+-]+)\s+(\w+)")
    out = []
    for name in h5.keys():
        m = pat.match(name)
        if not m:
            continue
        t = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "h":
            t *= 3600.0
        elif unit == "d":
            t *= 86400.0
        elif unit == "s":
            pass
        else:
            continue
        out.append((t, name))
    out.sort(key=lambda x: x[0])
    return out


def to_mol_per_m3(c: np.ndarray, field_name: str) -> np.ndarray:
    """Convert concentration arrays to mol/m^3 using field-name units.

    If a field uses [M] (mol/L), multiply by 1000.
    """
    if "[M]" in field_name:
        return c * 1000.0
    return c


def load_edges(h5: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X/Y/Z coordinate edge vectors from PFLOTRAN HDF5."""
    g = h5["Coordinates"]
    x = np.asarray(g["X [m]"])
    y = np.asarray(g["Y [m]"])
    z = np.asarray(g["Z [m]"])
    return x, y, z


def safe_cell_widths(
    edges: Optional[np.ndarray],
    n_fallback: int,
    length_fallback: float,
    axis_label: str = "axis",
) -> np.ndarray:
    """Return cell widths from coordinate edges with safe fallback.

    Conceptually: we need cell widths to compute face areas/volumes.
    If coordinate edges are invalid, use a uniform grid with given length.
    """
    warn_key = (axis_label, n_fallback, float(length_fallback))

    if edges is not None and len(edges) >= 2:
        d = np.diff(edges)
        if np.all(d > 0.0) and np.sum(d) > 0.0:
            return d
        if warn_key not in FALLBACK_WARNED:
            print(
                f"Fallback used for {axis_label}: invalid coordinate edges; "
                f"using uniform spacing with n={n_fallback}, L={length_fallback:.6e} m"
            )
            FALLBACK_WARNED.add(warn_key)
    else:
        if warn_key not in FALLBACK_WARNED:
            print(
                f"Fallback used for {axis_label}: missing/short coordinate edges; "
                f"using uniform spacing with n={n_fallback}, L={length_fallback:.6e} m"
            )
            FALLBACK_WARNED.add(warn_key)

    return np.full(n_fallback, length_fallback / n_fallback)


def reshape_cell_data(a: np.ndarray, nx: int, ny: int, name: str) -> np.ndarray:
    """Reshape 1D/2D cell arrays into (nx, ny)."""
    if a.shape == (nx, ny):
        return a
    if a.ndim == 1 and a.size == nx * ny:
        return a.reshape(nx, ny)
    raise ValueError(f"Cannot reshape {name} with shape {a.shape} to ({nx}, {ny})")


def pick_field(group: Any, preferred_name: str, required: bool = False) -> Optional[str]:
    """Pick the requested field name, or a close prefix match.

    This helps when PFLOTRAN adds unit suffixes that differ slightly.
    """
    if preferred_name in group:
        return preferred_name

    matches = [name for name in group.keys() if name.startswith(preferred_name.split(" [")[0])]
    if matches:
        return matches[0]

    if required:
        raise KeyError(f"Required field not found: {preferred_name}")
    return None


def load_porosity_field(geometry_h5: str, nx: int, ny: int) -> np.ndarray:
    """Load spatial porosity from geometry HDF5 for pore-volume integration."""
    with h5py.File(geometry_h5, "r") as h5:
        if "Porosity" not in h5:
            raise KeyError(f"Porosity dataset not found in {geometry_h5}")
        porosity = np.asarray(h5["Porosity"])
        return reshape_cell_data(porosity, nx, ny, "Porosity")


def outlet_inlet_flux_metrics(
    h5: Any,
    time_entries: list[tuple[float, str]],
    cu_key: str,
    h_key: str,
    ly: float,
    lz: float,
) -> dict[str, np.ndarray]:
    """Compute boundary flow and chemistry metrics used by KPIs.

    Conceptually: this function turns raw spatial fields into time series at the
    inlet/outlet (flow rates, flow-weighted Cu and H+, and derived outlet pH).
    """
    _, y_edges, z_edges = load_edges(h5)

    t_s = []
    q_out = []
    q_in = []
    cu_out = []
    cu_in = []
    ph_out = []
    h_out = []
    h_in = []

    outlet_side = None

    for t, time_key in time_entries:
        g = h5[time_key]
        cu_field = pick_field(g, cu_key, required=True)
        h_field = pick_field(g, h_key, required=True)

        if cu_field is None or h_field is None:
            continue

        if cu_field is None:
            continue

        vx_name = None
        for k in g.keys():
            if "Liquid X-Flux Velocities" in k:
                vx_name = k
                break
        if vx_name is None:
            for k in g.keys():
                if "Liquid X-Velocity" in k:
                    vx_name = k
                    break
        if vx_name is None:
            continue

        vx = g[vx_name][:, :, 0]
        nx, ny = vx.shape
        dy = safe_cell_widths(y_edges, ny, ly, axis_label="Y")
        dz_arr = safe_cell_widths(z_edges, 1, lz, axis_label="Z")
        dz = np.sum(dz_arr)
        area_y = dy * dz
        if "m_per_h" in vx_name:
            vx = vx / 3600.0
        face_flux = "flux" in vx_name.lower()

        cu = to_mol_per_m3(g[cu_field][:, :, 0], cu_field)
        hplus = to_mol_per_m3(g[h_field][:, :, 0], h_field)

        # Infer flow direction once from boundary-normal outflow magnitudes.
        if outlet_side is None:
            if face_flux:
                east_out_guess = np.sum(np.maximum(vx[-1, :], 0.0) * area_y) if vx.shape[0] > 0 else 0.0
                west_out_guess = np.sum(np.maximum(-vx[0, :], 0.0) * area_y) if vx.shape[0] > 0 else 0.0
            else:
                east_out_guess = np.sum(np.maximum(vx[-1, :], 0.0) * area_y)
                west_out_guess = np.sum(np.maximum(-vx[0, :], 0.0) * area_y)
            outlet_side = "east" if east_out_guess >= west_out_guess else "west"

        if outlet_side == "east":
            if face_flux:
                qn_out = np.maximum(vx[-1, :], 0.0)
                qn_in = np.maximum(vx[0, :], 0.0)
                cu_face = cu[-1, :]
                cu_in_face = cu[0, :]
                h_face = hplus[-1, :] if hplus is not None else None
                h_in_face = hplus[0, :] if hplus is not None else None
            else:
                qn_out = np.maximum(vx[-1, :], 0.0)
                qn_in = np.maximum(vx[0, :], 0.0)
                cu_face = cu[-1, :]
                cu_in_face = cu[0, :]
                h_face = hplus[-1, :] if hplus is not None else None
                h_in_face = hplus[0, :] if hplus is not None else None
        else:
            if face_flux:
                qn_out = np.maximum(-vx[0, :], 0.0)
                qn_in = np.maximum(-vx[-1, :], 0.0)
                cu_face = cu[0, :]
                cu_in_face = cu[-1, :]
                h_face = hplus[0, :] if hplus is not None else None
                h_in_face = hplus[-1, :] if hplus is not None else None
            else:
                qn_out = np.maximum(-vx[0, :], 0.0)
                qn_in = np.maximum(-vx[-1, :], 0.0)
                cu_face = cu[0, :]
                cu_in_face = cu[-1, :]
                h_face = hplus[0, :] if hplus is not None else None
                h_in_face = hplus[-1, :] if hplus is not None else None

        Qout = np.sum(qn_out * area_y)
        Qin = np.sum(qn_in * area_y)

        if Qout > 0.0:
            c_out = np.sum(qn_out * area_y * cu_face) / Qout
            h_out_val = np.sum(qn_out * area_y * h_face) / Qout if h_face is not None else np.nan
        else:
            c_out = np.nan
            h_out_val = np.nan

        c_in_val = np.sum(qn_in * area_y * cu_in_face) / Qin if Qin > 0.0 else np.nan
        h_in_val = np.sum(qn_in * area_y * h_in_face) / Qin if (h_in_face is not None and Qin > 0.0) else np.nan
        if np.isfinite(h_out_val) and h_out_val > 0.0:
            ph_out_val = 3.0 - np.log10(h_out_val)
        else:
            ph_out_val = np.nan

        t_s.append(t)
        q_out.append(Qout)
        q_in.append(Qin)
        cu_out.append(c_out)
        cu_in.append(c_in_val)
        ph_out.append(ph_out_val)
        h_out.append(h_out_val)
        h_in.append(h_in_val)

    return {
        "t_s": np.array(t_s),
        "Qout": np.array(q_out),
        "Qin": np.array(q_in),
        "cu_out_molm3": np.array(cu_out),
        "cu_in_molm3": np.array(cu_in),
        "ph_out": np.array(ph_out),
        "h_out_molm3": np.array(h_out),
        "h_in_molm3": np.array(h_in),
    }


def initial_cu_inventory(
    h5: Any,
    first_group: str,
    vf_key: str,
    geometry_h5: str,
    lx: float,
    ly: float,
    lz: float,
) -> tuple[float, float]:
    """Estimate initial solid Cu inventory and total pore volume.

    Non-expert view:
    - Initial Cu in rock comes from Chrysocolla volume fraction and the
      mineral molar volume from the PFLOTRAN database.
    - Pore volume comes from spatial porosity in the geometry file.
    """
    g = h5[first_group]
    vf_field = pick_field(g, vf_key, required=True)
    vf = g[vf_field][:, :, 0]
    nx, ny = vf.shape

    x_edges, y_edges, z_edges = load_edges(h5)
    dx = safe_cell_widths(x_edges, nx, lx, axis_label="X")
    dy = safe_cell_widths(y_edges, ny, ly, axis_label="Y")
    dz = np.sum(safe_cell_widths(z_edges, 1, lz, axis_label="Z"))
    cell_vol = dx[:, None] * dy[None, :] * dz

    molar_volume_m3_mol = CHRYSOCOLLA2_MOLAR_VOLUME_CM3_MOL * 1.0e-6
    n_chr = vf * cell_vol / molar_volume_m3_mol
    cu_solid = np.sum(CHRYSOCOLLA2_CU_STOICH * n_chr)

    porosity = load_porosity_field(geometry_h5, nx, ny)
    pore_vol = np.sum(porosity * cell_vol)
    return cu_solid, pore_vol


def chrysocolla_cu_series(
    h5: Any,
    time_entries: list[tuple[float, str]],
    vf_key: str,
    lx: float,
    ly: float,
    lz: float,
) -> dict[str, np.ndarray]:
    """Build time series of Cu remaining in, and dissolved from, Chrysocolla2."""
    _, first_group = time_entries[0]
    vf0_field = pick_field(h5[first_group], vf_key, required=True)
    vf0 = h5[first_group][vf0_field][:, :, 0]
    nx, ny = vf0.shape

    x_edges, y_edges, z_edges = load_edges(h5)
    dx = safe_cell_widths(x_edges, nx, lx, axis_label="X")
    dy = safe_cell_widths(y_edges, ny, ly, axis_label="Y")
    dz = np.sum(safe_cell_widths(z_edges, 1, lz, axis_label="Z"))
    cell_vol = dx[:, None] * dy[None, :] * dz

    t_s = []
    cu_remaining = []
    for t, time_key in time_entries:
        g = h5[time_key]
        vf_field = pick_field(g, vf_key, required=True)
        vf = g[vf_field][:, :, 0]
        molar_volume_m3_mol = CHRYSOCOLLA2_MOLAR_VOLUME_CM3_MOL * 1.0e-6
        n_chr = vf * cell_vol / molar_volume_m3_mol
        cu_remaining.append(np.sum(CHRYSOCOLLA2_CU_STOICH * n_chr))
        t_s.append(t)

    cu_remaining = np.array(cu_remaining)
    cu_dissolved = cu_remaining[0] - cu_remaining
    return {
        "t_s": np.array(t_s),
        "cu_remaining": cu_remaining,
        "cu_dissolved": cu_dissolved,
    }


def total_mineral_and_porosity_series(
    h5: Any,
    time_entries: list[tuple[float, str]],
    lx: float,
    ly: float,
    lz: float,
) -> dict[str, np.ndarray]:
    """Build time series of total mineral dissolution and domain porosity."""
    _, first_group = time_entries[0]
    g0 = h5[first_group]

    mineral_fields = [k for k in g0.keys() if "_VF [m^3 mnrl_m^3 bulk]" in k]
    if not mineral_fields:
        raise KeyError("No mineral volume-fraction fields found in time groups")

    first_vf = np.asarray(g0[mineral_fields[0]][:, :, 0])
    nx, ny = first_vf.shape
    x_edges, y_edges, z_edges = load_edges(h5)
    dx = safe_cell_widths(x_edges, nx, lx, axis_label="X")
    dy = safe_cell_widths(y_edges, ny, ly, axis_label="Y")
    dz = np.sum(safe_cell_widths(z_edges, 1, lz, axis_label="Z"))
    cell_vol = dx[:, None] * dy[None, :] * dz

    mineral_vol0 = None
    initial_total_mineral_bulk_m3 = None
    t_s = []
    mineral_dissolved_m3 = []
    porosity_mean = []
    pore_volume_m3 = []
    solid_volume_m3 = []
    domain_volume_m3 = float(np.sum(cell_vol))

    for t, time_key in time_entries:
        g = h5[time_key]

        mineral_vol = np.zeros((nx, ny), dtype=float)
        for field in mineral_fields:
            mineral_vol += np.asarray(g[field][:, :, 0])

        if mineral_vol0 is None:
            mineral_vol0 = mineral_vol.copy()
            initial_total_mineral_bulk_m3 = np.sum(mineral_vol0 * cell_vol)

        dissolved_bulk_m3 = np.sum((mineral_vol0 - mineral_vol) * cell_vol)

        poro_field = pick_field(g, "Porosity", required=True)
        poro = np.asarray(g[poro_field][:, :, 0])
        pore_vol = float(np.sum(poro * cell_vol))
        solid_vol = domain_volume_m3 - pore_vol

        t_s.append(t)
        mineral_dissolved_m3.append(dissolved_bulk_m3)
        porosity_mean.append(pore_vol / domain_volume_m3)
        pore_volume_m3.append(pore_vol)
        solid_volume_m3.append(solid_vol)

    return {
        "t_s": np.array(t_s),
        "mineral_dissolved_m3": np.array(mineral_dissolved_m3),
        "porosity_mean": np.array(porosity_mean),
        "pore_volume_m3": np.array(pore_volume_m3),
        "solid_volume_m3": np.array(solid_volume_m3),
        "domain_volume_m3": domain_volume_m3,
        "initial_total_mineral_bulk_m3": float(initial_total_mineral_bulk_m3),
    }


def mineral_inventory_series(
    h5: Any,
    time_entries: list[tuple[float, str]],
    lx: float,
    ly: float,
    lz: float,
) -> dict[str, Any]:
    """Build per-mineral volume inventory time series and mass-balance checks."""
    _, first_group = time_entries[0]
    g0 = h5[first_group]
    mineral_fields = [k for k in g0.keys() if "_VF [m^3 mnrl_m^3 bulk]" in k]
    if not mineral_fields:
        raise KeyError("No mineral VF fields found for inventory plotting")

    mineral_names = [k.split("_VF")[0] for k in mineral_fields]
    first_vf = np.asarray(g0[mineral_fields[0]][:, :, 0])
    nx, ny = first_vf.shape

    x_edges, y_edges, z_edges = load_edges(h5)
    dx = safe_cell_widths(x_edges, nx, lx, axis_label="X")
    dy = safe_cell_widths(y_edges, ny, ly, axis_label="Y")
    dz = np.sum(safe_cell_widths(z_edges, 1, lz, axis_label="Z"))
    cell_vol = dx[:, None] * dy[None, :] * dz
    domain_vol = float(np.sum(cell_vol))

    t_s = []
    vol = np.zeros((len(time_entries), len(mineral_fields)), dtype=float)
    solid_vol = np.zeros(len(time_entries), dtype=float)
    por_mean = np.zeros(len(time_entries), dtype=float)

    for i, (t, gname) in enumerate(time_entries):
        g = h5[gname]
        for j, field in enumerate(mineral_fields):
            vf = np.asarray(g[field][:, :, 0])
            vol[i, j] = float(np.sum(vf * cell_vol))

        poro_field = pick_field(g, "Porosity", required=True)
        por = np.asarray(g[poro_field][:, :, 0])
        pore_vol = float(np.sum(por * cell_vol))
        solid_vol[i] = domain_vol - pore_vol
        por_mean[i] = pore_vol / domain_vol
        t_s.append(t)

    vol0 = vol[0, :]
    dissolved = vol0[None, :] - vol
    dissolved_total = np.sum(dissolved, axis=1)
    solid_loss = solid_vol[0] - solid_vol
    residual = solid_loss - dissolved_total

    return {
        "t_s": np.array(t_s),
        "mineral_names": mineral_names,
        "vol_m3": vol,
        "dissolved_m3": dissolved,
        "solid_vol_m3": solid_vol,
        "porosity_mean": por_mean,
        "solid_loss_m3": solid_loss,
        "dissolved_total_m3": dissolved_total,
        "residual_m3": residual,
    }


def cumulative_metrics(
    series: dict[str, np.ndarray],
    cu_init_solid_mol: float,
    pore_vol_m3: float,
) -> dict[str, np.ndarray]:
    """Integrate cumulative Cu and acid KPIs from time-series boundary data.

    Non-expert view:
    - Integrate outlet flux to get Cu leaving the model.
    - Subtract inlet Cu flux to get net recovered Cu.
    - Integrate acid in/out to get net acid consumed and efficiency metrics.
    """
    t = series["t_s"]
    c_cu = series["cu_out_molm3"]
    c_cu_in = series.get("cu_in_molm3", None)
    qout = series["Qout"]
    qin = series["Qin"]
    c_h_out = series["h_out_molm3"]
    c_h_in = series.get("h_in_molm3", None)

    n = len(t)
    cu_cum = np.zeros(n)
    cu_inj_cum = np.zeros(n)
    h_inj_cum = np.zeros(n)
    h_out_cum = np.zeros(n)

    for i in range(1, n):
        dt = t[i] - t[i - 1]

        cu_flux_1 = qout[i - 1] * np.nan_to_num(c_cu[i - 1], nan=0.0)
        cu_flux_2 = qout[i] * np.nan_to_num(c_cu[i], nan=0.0)
        cu_cum[i] = cu_cum[i - 1] + 0.5 * (cu_flux_1 + cu_flux_2) * dt

        if c_cu_in is not None:
            cu_in_flux_1 = qin[i - 1] * np.nan_to_num(c_cu_in[i - 1], nan=0.0)
            cu_in_flux_2 = qin[i] * np.nan_to_num(c_cu_in[i], nan=0.0)
            cu_inj_cum[i] = cu_inj_cum[i - 1] + 0.5 * (cu_in_flux_1 + cu_in_flux_2) * dt
        else:
            cu_inj_cum[i] = np.nan

        if c_h_in is not None:
            h_in_flux_1 = qin[i - 1] * np.nan_to_num(c_h_in[i - 1], nan=0.0)
            h_in_flux_2 = qin[i] * np.nan_to_num(c_h_in[i], nan=0.0)
            h_inj_cum[i] = h_inj_cum[i - 1] + 0.5 * (h_in_flux_1 + h_in_flux_2) * dt
        else:
            h_inj_cum[i] = np.nan

        if c_h_out is not None:
            h_out_flux_1 = qout[i - 1] * np.nan_to_num(c_h_out[i - 1], nan=0.0)
            h_out_flux_2 = qout[i] * np.nan_to_num(c_h_out[i], nan=0.0)
            h_out_cum[i] = h_out_cum[i - 1] + 0.5 * (h_out_flux_1 + h_out_flux_2) * dt
        else:
            h_out_cum[i] = np.nan

    cu_net = cu_cum - cu_inj_cum
    frac_rec = cu_net / max(cu_init_solid_mol, 1.0e-30)
    pv = np.zeros(n)
    if pore_vol_m3 > 0.0:
        qin_avg = 0.5 * (qin[:-1] + qin[1:])
        pv[1:] = np.cumsum(qin_avg * np.diff(t) / pore_vol_m3)

    acid_eff_inj = cu_net / np.where(h_inj_cum > 1.0e-30, h_inj_cum, np.nan)
    h_cons = h_inj_cum - h_out_cum
    acid_eff_cons = cu_net / np.where(h_cons > 1.0e-30, h_cons, np.nan)

    return {
        "pv": pv,
        "cu_cum": cu_net,
        "cu_out_cum": cu_cum,
        "cu_inj_cum": cu_inj_cum,
        "h_inj_cum": h_inj_cum,
        "h_cons": h_cons,
        "frac_rec": frac_rec,
        "acid_eff_inj": acid_eff_inj,
        "acid_eff_cons": acid_eff_cons,
    }


def plot_btc(t_h: np.ndarray, cu_out_molm3: np.ndarray, ph_out_derived: np.ndarray, out_png: str) -> None:
    """Plot outlet breakthrough-style curves (Cu and acidity/pH proxy)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.semilogy(t_h, np.clip(cu_out_molm3 / 1000.0, 1.0e-16, None), "-o", ms=3)
    ax1.set_ylabel("Outlet Cu [mol/L]")
    ax1.grid(True, alpha=0.3)
    format_axis(ax1)
    ax1.set_title(
        "Where does reactive fluid go?\n"
        r"$\bar{C}_{Cu,out}^{fw}(t)=\frac{\int_{A_{out}} q_n C_{Cu}\,dA}{\int_{A_{out}} q_n\,dA}$",
        fontsize=11,
    )

    ax2.plot(t_h, ph_out_derived, "-o", ms=3, color="tab:red")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel("Outlet pH [-]")
    ax2.grid(True, alpha=0.3)
    format_axis(ax2)
    ax2.set_title(
        "How much acid was spent to get it?\n"
        r"$pH_{out}^{derived}(t)=-\log_{10}\!\left(\bar{C}_{H^+,out}^{fw}(t)\,[mol/L]\right)$",
        fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_recovery(metrics: dict[str, np.ndarray], out_png: str) -> None:
    """Plot cumulative net Cu recovery and recovery fraction vs pore volumes."""
    pv = metrics["pv"]
    cu_cum = metrics["cu_cum"]
    frac_rec = metrics["frac_rec"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].plot(pv, cu_cum * 1.0e6, "-o", ms=3)
    axes[0].set_xlabel("Pore volumes injected [-]")
    axes[0].set_ylabel("Cumulative Cu recovered [umol]")
    axes[0].grid(True, alpha=0.3)
    format_axis(axes[0])
    axes[0].set_title(
        "How much of that dissolved metal is recovered at the outlet?\n"
        r"$M_{Cu}^{rec}(t)=\int_0^t [Q_{out}\bar{C}_{Cu,out}^{fw}-Q_{in}\bar{C}_{Cu,in}^{fw}]\,d\tau$",
        fontsize=10,
    )

    axes[1].plot(pv, frac_rec * 100.0, "-o", ms=3, color="tab:green")
    axes[1].set_xlabel("Pore volumes injected [-]")
    axes[1].set_ylabel("Recovery [% of initial solid Cu]")
    axes[1].grid(True, alpha=0.3)
    format_axis(axes[1])
    axes[1].set_title(
        "How much metal is dissolved?\n"
        r"$Recovery(t)=\frac{M_{Cu}^{rec}(t)}{M_{Cu,solid,0}} \times 100$",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_acid_usage_time(t_h: np.ndarray, metrics: dict[str, np.ndarray], out_png: str) -> None:
    """Plot acid consumption and acid-efficiency metrics over time."""
    h_cons = metrics["h_cons"]
    eff_inj = metrics["acid_eff_inj"]
    eff_cons = metrics["acid_eff_cons"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)

    axes[0].plot(t_h, h_cons * 1.0e3, "-o", ms=3)
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("Net acid consumed [mmol H+]")
    axes[0].grid(True, alpha=0.3)
    format_axis(axes[0])
    axes[0].set_title(
        "How much acid was spent to get it?\n"
        r"$M_{H^+}^{cons}(t)=M_{H^+}^{inj}(t)-M_{H^+}^{out}(t)$",
        fontsize=10,
    )

    axes[1].plot(t_h, eff_inj, "-o", ms=3, color="tab:blue")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("Cu / acid injected [mol/mol]")
    axes[1].grid(True, alpha=0.3)
    format_axis(axes[1])
    axes[1].set_title(
        "How efficiently does injected acid recover Cu?\n"
        r"$\eta_{inj}(t)=\frac{M_{Cu}^{rec}(t)}{M_{H^+}^{inj}(t)}$",
        fontsize=10,
    )

    axes[2].plot(t_h, eff_cons, "-o", ms=3, color="tab:purple")
    axes[2].set_xlabel("Time [h]")
    axes[2].set_ylabel("Cu / acid consumed [mol/mol]")
    axes[2].grid(True, alpha=0.3)
    format_axis(axes[2])
    axes[2].set_title(
        "How efficiently is consumed acid converted to Cu?\n"
        r"$\eta_{cons}(t)=\frac{M_{Cu}^{rec}(t)}{M_{H^+}^{cons}(t)}$",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_chrysocolla_cu(t_h: np.ndarray, mineral_metrics: dict[str, np.ndarray], out_png: str) -> None:
    """Plot Chrysocolla-specific Cu dissolution and remaining inventory."""
    cu_diss = mineral_metrics["cu_dissolved"] * 1.0e6
    cu_rem = mineral_metrics["cu_remaining"] * 1.0e6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    axes[0].plot(t_h, cu_diss, "-o", ms=3, color="tab:blue")
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("Total Cu dissolved from Chrysocolla2 [umol]")
    axes[0].grid(True, alpha=0.3)
    format_axis(axes[0])
    axes[0].set_title("How much Cu is released from Chrysocolla2?", fontsize=10)

    axes[1].plot(t_h, cu_rem, "-o", ms=3, color="tab:orange")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("Cu remaining in Chrysocolla2 [umol]")
    axes[1].grid(True, alpha=0.3)
    format_axis(axes[1])
    axes[1].set_title("How much Cu remains in Chrysocolla2?", fontsize=10)

    y_min = np.min(cu_rem)
    y_max = np.max(cu_rem)
    pad = max(1.0e-6, 0.05 * max(y_max - y_min, 1.0e-12))
    axes[1].set_ylim(y_min - pad, y_max + pad)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_mineral_dissolution_porosity(t_h: np.ndarray, series: dict[str, np.ndarray], out_png: str) -> None:
    """Plot total mineral dissolution and mean domain porosity over time."""
    dissolved = series["mineral_dissolved_m3"]
    initial_total = series["initial_total_mineral_bulk_m3"]
    phi = series["porosity_mean"]
    dissolved_pct = 100.0 * dissolved / max(initial_total, 1.0e-30)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    axes[0].plot(t_h, dissolved_pct, "-o", ms=3, color="tab:brown")
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("Total mineral dissolved [% of initial]")
    axes[0].grid(True, alpha=0.3)
    format_axis(axes[0])
    axes[0].set_title("Total mineral dissolution")

    axes[1].plot(t_h, phi, "-o", ms=3, color="tab:green")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("Domain mean porosity [-]")
    axes[1].grid(True, alpha=0.3)
    format_axis(axes[1])
    axes[1].set_title("Domain porosity evolution")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_mineral_inventory(t_h: np.ndarray, inv: dict[str, Any], out_png: str) -> None:
    """Plot 2x2 mineral inventory dashboard for dissolution and porosity attribution."""
    names = inv["mineral_names"]
    vol = inv["vol_m3"]
    diss = inv["dissolved_m3"]
    solid = inv["solid_vol_m3"]
    por = inv["porosity_mean"]
    residual = inv["residual_m3"]

    colors = ["tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # A: remaining mineral inventory by phase
    axes[0, 0].stackplot(
        t_h,
        *[vol[:, i] * 1.0e9 for i in range(vol.shape[1])],
        labels=names,
        colors=colors[: len(names)],
        alpha=0.85,
    )
    axes[0, 0].set_ylabel("Remaining mineral volume [mm^3]")
    axes[0, 0].set_title("Mineral inventory by phase")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="best")

    # B: dissolved fraction per phase
    for i, n in enumerate(names):
        init_i = max(vol[0, i], 1.0e-30)
        axes[0, 1].plot(t_h, 100.0 * diss[:, i] / init_i, "-o", ms=2.5, label=n, color=colors[i % len(colors)])
    axes[0, 1].set_ylabel("Dissolved [% of phase initial]")
    axes[0, 1].set_title("Per-phase dissolution")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=8, loc="best")

    # C: solid volume and porosity
    axc = axes[1, 0]
    axc2 = axc.twinx()
    axc.plot(t_h, solid * 1.0e9, "-o", ms=2.5, color="tab:brown", label="Solid volume")
    axc2.plot(t_h, por, "-o", ms=2.5, color="tab:green", label="Mean porosity")
    axc.set_ylabel("Solid volume [mm^3]", color="tab:brown")
    axc2.set_ylabel("Mean porosity [-]", color="tab:green")
    axc.tick_params(axis="y", colors="tab:brown")
    axc2.tick_params(axis="y", colors="tab:green")
    axc.set_title("Solid shrinkage and porosity increase")
    axc.grid(True, alpha=0.25)

    # D: volume-balance residual check
    axes[1, 1].plot(t_h, residual * 1.0e12, "-o", ms=2.5, color="black")
    axes[1, 1].axhline(0.0, color="gray", lw=1)
    axes[1, 1].set_ylabel("Solid-loss minus dissolved [nL]")
    axes[1, 1].set_title("Mass-balance residual (should be near 0)")
    axes[1, 1].grid(True, alpha=0.25)

    axes[1, 0].set_xlabel("Time [h]")
    axes[1, 1].set_xlabel("Time [h]")

    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"Saved {out_png}")


def main() -> None:
    """Read PFLOTRAN outputs, compute KPIs, and write all figures."""
    p = argparse.ArgumentParser(description="Recovery KPI plots from PFLOTRAN HDF5")
    p.add_argument("h5_file", nargs="?", default="cu_leaching_pore.h5")
    p.add_argument("--prefix", default="cu_recovery")
    p.add_argument("--fig-dir", type=Path, default=".", help="Output directory for figures")
    p.add_argument("--cu-field", default="Total_Cu++ [M]")
    p.add_argument("--h-field", default="Free_H+ [M]")
    p.add_argument("--vf-field", default="Chrysocolla2_VF [m^3 mnrl_m^3 bulk]")
    p.add_argument("--geometry-file", default=DEFAULT_GEOMETRY_H5)
    p.add_argument("--lx", type=float, default=DEFAULT_LX_M, help="Fallback domain length in x [m]")
    p.add_argument("--ly", type=float, default=DEFAULT_LY_M, help="Fallback domain length in y [m]")
    p.add_argument("--lz", type=float, default=DEFAULT_LZ_M, help="Fallback domain thickness in z [m]")
    args = p.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5_file, "r") as h5:
        time_entries = parse_time_groups(h5)
        if not time_entries:
            print("No PFLOTRAN time groups found.")
            return

        mineral_metrics = chrysocolla_cu_series(
            h5,
            time_entries,
            args.vf_field,
            args.lx,
            args.ly,
            args.lz,
        )

        mineral_porosity_series = total_mineral_and_porosity_series(
            h5,
            time_entries,
            args.lx,
            args.ly,
            args.lz,
        )

        inventory = mineral_inventory_series(
            h5,
            time_entries,
            args.lx,
            args.ly,
            args.lz,
        )

        series = outlet_inlet_flux_metrics(
            h5,
            time_entries,
            args.cu_field,
            args.h_field,
            args.ly,
            args.lz,
        )

        if len(series["t_s"]) < 2:
            print("Not enough snapshots with velocity/Cu for recovery metrics.")
            return

        cu_init, pore_vol = initial_cu_inventory(
            h5,
            time_entries[0][1],
            args.vf_field,
            args.geometry_file,
            args.lx,
            args.ly,
            args.lz,
        )

    metrics = cumulative_metrics(series, cu_init, pore_vol)

    t_h = series["t_s"] / 3600.0
    plot_btc(t_h, series["cu_out_molm3"], series["ph_out"], str(args.fig_dir / f"{args.prefix}_btc.png"))
    plot_recovery(metrics, str(args.fig_dir / f"{args.prefix}_recovery.png"))
    plot_acid_usage_time(t_h, metrics, str(args.fig_dir / f"{args.prefix}_acid_time.png"))
    plot_chrysocolla_cu(mineral_metrics["t_s"] / 3600.0, mineral_metrics, str(args.fig_dir / f"{args.prefix}_chrysocolla_cu.png"))
    plot_mineral_dissolution_porosity(
        mineral_porosity_series["t_s"] / 3600.0,
        mineral_porosity_series,
        str(args.fig_dir / f"{args.prefix}_mineral_porosity.png"),
    )
    plot_mineral_inventory(
        inventory["t_s"] / 3600.0,
        inventory,
        str(args.fig_dir / f"{args.prefix}_mineral_inventory.png"),
    )

    print(f"Initial solid Cu estimate: {cu_init:.6e} mol")
    print(f"Estimated pore volume:      {pore_vol:.6e} m^3")
    init_solid = mineral_porosity_series["solid_volume_m3"][0]
    final_solid = mineral_porosity_series["solid_volume_m3"][-1]
    solid_loss = init_solid - final_solid
    init_mineral = mineral_porosity_series["initial_total_mineral_bulk_m3"]
    mineral_loss = mineral_porosity_series["mineral_dissolved_m3"][-1]
    print(f"Initial total solid volume: {init_solid:.6e} m^3")
    print(f"Final total solid volume:   {final_solid:.6e} m^3")
    print(f"Net solid volume loss:      {solid_loss:.6e} m^3 ({100.0*solid_loss/max(init_solid,1e-30):.2f}%)")
    print(f"Initial mineral volume:     {init_mineral:.6e} m^3")
    print(f"Net mineral volume loss:    {mineral_loss:.6e} m^3 ({100.0*mineral_loss/max(init_mineral,1e-30):.2f}%)")


if __name__ == "__main__":
    main()
