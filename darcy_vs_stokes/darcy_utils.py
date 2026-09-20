"""Tools for Darcy vs 2DStokes_FD comparison.

"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import h5py
import numpy as np
try:
    from darcy_vs_stokes.write_darcy_tracer_input import build_input_text, load_grid_shape
    from darcy_vs_stokes.utils import run_cmd
except ModuleNotFoundError:
    from write_darcy_tracer_input import build_input_text, load_grid_shape
    from utils import run_cmd


def resolve_pflotran_exe() -> Path:
    exe = shutil.which("pflotran")
    if exe:
        return Path(exe)
    fallback = Path("/home/vscode/src/pflotran/src/pflotran/pflotran")
    if fallback.exists():
        return fallback
    raise FileNotFoundError("pflotran not found")


def get_time_groups(h5: h5py.File) -> list[tuple[float, str]]:
    pat = re.compile(r"Time:\s+([\d.Ee+-]+)\s+(\w+)")
    groups = []
    for name in h5.keys():
        m = pat.match(name)
        if not m:
            continue
        t = float(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("h"):
            t *= 3600.0
        elif unit.startswith("d"):
            t *= 86400.0
        groups.append((t, name))
    groups.sort(key=lambda x: x[0])
    return groups



def find_key(keys: list[str], tokens: list[str]) -> str | None:
    for key in keys:
        low = key.lower().replace("_", " ").replace("-", " ")
        if all(tok in low for tok in tokens):
            return key
    return None


def to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[:, :, 0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def match_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    if arr.T.shape == target_shape:
        return arr.T
    raise ValueError(f"Shape mismatch: {arr.shape} vs {target_shape}")


def choose_orientation(
    p_raw: np.ndarray,
    arrays: list[np.ndarray],
    mask: np.ndarray,
    dP: float,
) -> tuple[np.ndarray, ...]:
    choices = []
    for use_t in (False, True):
        p = p_raw.T if use_t else p_raw
        arrs = [a.T if use_t else a for a in arrays]
        if p.shape != mask.shape or any(a.shape != mask.shape for a in arrs):
            continue
        left = p[:, 0][mask[:, 0] == 1]
        right = p[:, -1][mask[:, -1] == 1]
        if left.size == 0 or right.size == 0:
            score = float("inf")
        else:
            score = abs(float(left.mean()) - dP) + abs(float(right.mean()))
        choices.append((score, p, *arrs))
    if not choices:
        raise ValueError(
            f"Unable to orient Darcy fields: p={p_raw.shape}, mask={mask.shape}, "
            f"arrays={[a.shape for a in arrays]}"
        )
    choices.sort(key=lambda x: x[0])
    return choices[0][1:]

def copy_geometry_with_contrast(
    base_geom_h5: Path,
    out_geom_h5: Path,
    pore_perm: float,
    solid_perm: float,
) -> None:
    """Copy geometry H5 and overwrite only Permeability. Porosity and mineral VFs are preserved as-built."""
    shutil.copyfile(base_geom_h5, out_geom_h5)
    with h5py.File(out_geom_h5, "r+") as h5:
        mids = np.asarray(h5["Materials"]["Material Ids"])
        h5["Permeability"][...] = np.where(mids == 1, pore_perm, solid_perm)


def write_darcy_input_for_flow(
    pflotran_input: Path,
    base_geom_h5: Path,
    out_geom_h5: Path,
    pore_perm: float,
    solid_perm: float = 1.0e-20,
    dx: float = 20.0e-6,
    dz: float = 1.0e-3,
    p_in: float = 101326.0,
    p_out: float = 101325.0,
    initial_dt: float = 1.0e-3,
    pulse_duration: float = 1.0e-3,
    final_time: float = 10.0,
    output_period: float = 1.0,
    pore_diffusivity: float = 1.0e-9,
    solid_diffusivity: float = 1.0e-11,
) -> None:
    """Write PFLOTRAN Darcy+tracer input for flow comparisons."""
    copy_geometry_with_contrast(
        base_geom_h5,
        out_geom_h5,
        pore_perm=pore_perm,
        solid_perm=solid_perm,
    )

    nx, ny = load_grid_shape(base_geom_h5)
    text = build_input_text(
        geometry_h5=Path(out_geom_h5),
        nx=nx,
        ny=ny,
        dx=dx,
        dz=dz,
        p_in=p_in,
        p_out=p_out,
        pulse_duration=pulse_duration,
        final_time=final_time,
        output_period=output_period,
        initial_dt=initial_dt,
        max_dt=output_period,
        pore_diffusivity=pore_diffusivity,
        solid_diffusivity=solid_diffusivity,
        background_concentration=1.0e-20,
        tracer_on_concentration=1.0,
    )
    pflotran_input.write_text(text)


def run_darcy(case_input: Path) -> None:
    stem = case_input.stem
    for suffix in [".h5", "-obs-0.pft", "-mas.dat", "-chk.h5"]:
        (case_input.parent / f"{stem}{suffix}").unlink(missing_ok=True)
    PFLOTRAN_EXE = resolve_pflotran_exe()
    run_cmd([str(PFLOTRAN_EXE), "-pflotranin", case_input.name], case_input.parent)



def load_darcy_case(
    h5_path: Path,
    target_time_s: float,
    mask: np.ndarray,
    dx: float,
    pressure_in: float,
    pressure_out: float,
) -> dict[str, np.ndarray | float]:
    times = []
    right_flux = []
    flow_flux = []
    final_u = None
    final_v = None
    final_p = None
    best_dt = float("inf")
    outlet_mask = mask[:, -1] == 1

    with h5py.File(h5_path, "r") as h5:
        groups = get_time_groups(h5)
        for tval, gname in groups:
            group = h5[gname]
            keys = list(group.keys())
            kx = find_key(keys, ["liquid", "x", "velocity"])
            ky = find_key(keys, ["liquid", "y", "velocity"])
            kp = find_key(keys, ["liquid", "pressure"])
            kc = find_key(keys, ["total", "tracer"])
            if kx is None or ky is None or kp is None or kc is None:
                raise KeyError(f"Missing Darcy datasets in {h5_path}")

            u_raw = match_shape(to_2d(group[kx]), mask.shape)
            v_raw = match_shape(to_2d(group[ky]), mask.shape)
            p_raw = match_shape(to_2d(group[kp]), mask.shape)
            c_raw = match_shape(to_2d(group[kc]), mask.shape)
            p, u, v, c = choose_orientation(
                p_raw - pressure_out,
                [u_raw, v_raw, c_raw],
                mask,
                pressure_in - pressure_out,
            )

            u_out = np.maximum(u[:, -1], 0.0)
            q_out = float(np.sum(u_out[outlet_mask]) * dx)
            rf_out = float(np.sum((u_out * c[:, -1])[outlet_mask]) * dx)
            times.append(tval)
            flow_flux.append(q_out)
            right_flux.append(rf_out)

            dt = abs(tval - target_time_s)
            if dt < best_dt:
                best_dt = dt
                final_u = u.copy()
                final_v = v.copy()
                final_p = p.copy()

    flow_flux = np.asarray(flow_flux)
    times = np.asarray(times)
    right_flux = np.asarray(right_flux)
    t_cutoff = times[-1] * 0.8
    steady = times >= t_cutoff
    if np.any(steady):
        q_ref = float(np.mean(flow_flux[steady]))
    else:
        q_ref = float(flow_flux[-1])
    q_ref = max(q_ref, 1.0e-30)

    if final_u is None or final_v is None or final_p is None:
        raise RuntimeError(f"No Darcy solution fields found in {h5_path}")

    final_u_arr = np.asarray(final_u)
    q_cols = np.asarray(
        [float(np.sum(final_u_arr[:, j][mask[:, j] == 1]) * dx) for j in range(final_u_arr.shape[1])]
    )
    q_in = float(q_cols[0])
    q_out = float(q_cols[-1])
    q_mean = float(np.mean(q_cols))
    q_std = float(np.std(q_cols))
    q_var = q_std / max(abs(q_mean), 1.0e-30) * 100.0
    L = np.asarray(mask).shape[1] * dx
    U_D = q_ref / max(np.asarray(mask).shape[0] * dx, 1.0e-30)

    return {
        "times": times,
        "final_time": float(times[-1]),
        "right_flux": right_flux,
        "q_ref": q_ref,
        "q_cols": q_cols,
        "q_in": q_in,
        "q_out": q_out,
        "q_mean": q_mean,
        "q_std": q_std,
        "q_var": q_var,
        "u_D": U_D,
        "L": L,
        "u": final_u_arr,
        "v": np.asarray(final_v),
        "p": np.asarray(final_p),
    }




def load_darcy_snapshot(h5_path: Path, target_time_s: float | None = None) -> dict[str, np.ndarray | float]:
    """Load Darcy velocity and pressure snapshot."""
    with h5py.File(h5_path, "r") as h5:
        if target_time_s is None:
            t_s, gname = get_time_groups(h5)[-1]
        else:
            gname, t_s = nearest_group(h5, target_time_s)
        g = h5[gname]
        keys = list(g.keys())
        kx = find_key(keys, ["liquid", "x", "velocity"])
        ky = find_key(keys, ["liquid", "y", "velocity"])
        kp = find_key(keys, ["liquid", "pressure"])
        if kx is None or ky is None or kp is None:
            raise KeyError(f"Missing Darcy fields in {h5_path}")
        u = to_2d(g[kx])
        v = to_2d(g[ky])
        p = to_2d(g[kp])
    return {"time": t_s, "u": u, "v": v, "p": p}


def parse_obs_curve(obs_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse PFLOTRAN obs file."""
    tvals = []
    cvals = []
    lines = obs_path.read_text().splitlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        tvals.append(float(parts[0]))
        cvals.append(float(parts[-1]))
    return np.asarray(tvals), np.asarray(cvals)


def arrival_time(times: np.ndarray, values: np.ndarray, frac: float) -> float:
    """Arrival time at frac of max."""
    if len(values) == 0:
        return float("nan")
    vmax = float(np.max(values))
    if vmax <= 1.0e-30:
        return float("nan")
    target = frac * vmax
    idx = np.where(values >= target)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(times[0])
    t0, t1 = times[i - 1], times[i]
    v0, v1 = values[i - 1], values[i]
    if v1 == v0:
        return float(t1)
    w = (target - v0) / (v1 - v0)
    return float(t0 + w * (t1 - t0))


def compute_case_metrics(
    stokes: dict[str, np.ndarray | float],
    darcy: dict[str, np.ndarray | float],
    dx: float,
    darcy_obs: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute comparison metrics."""
    mask = np.asarray(stokes["mask"]) == 1
    su = np.asarray(stokes["u"])
    sv = np.asarray(stokes["v"])
    du = np.asarray(darcy["u"])
    dv = np.asarray(darcy["v"])
    sp = np.sqrt(su * su + sv * sv)
    dp = np.sqrt(du * du + dv * dv)

    speed_rmse = float(np.sqrt(np.mean((dp[mask] - sp[mask]) ** 2)))
    u_rmse = float(np.sqrt(np.mean((du[mask] - su[mask]) ** 2)))
    v_rmse = float(np.sqrt(np.mean((dv[mask] - sv[mask]) ** 2)))
    speed_rel_rmse = speed_rmse / max(float(np.mean(sp[mask])), 1.0e-30)

    q_ref = float(stokes["q_ref"])
    darcy_flux = float(np.sum(du[:, -1][mask[:, -1]]) * dx)
    stokes_flux = q_ref
    flux_rel_err = abs(darcy_flux - stokes_flux) / max(abs(stokes_flux), 1.0e-30)

    metrics = {
        "speed_rmse": speed_rmse,
        "speed_rel_rmse": speed_rel_rmse,
        "u_rmse": u_rmse,
        "v_rmse": v_rmse,
        "stokes_flux": stokes_flux,
        "darcy_flux": darcy_flux,
        "flux_rel_err": flux_rel_err,
        "darcy_time_s": float(darcy["time"]),
    }

    if darcy_obs is not None:
        t_obs, c_obs = darcy_obs
        s_times = np.asarray(stokes["times"])
        s_btc = np.asarray(stokes["right_flux"]) / max(abs(q_ref), 1.0e-30)
        metrics["stokes_t10"] = arrival_time(s_times, s_btc, 0.1)
        metrics["stokes_t50"] = arrival_time(s_times, s_btc, 0.5)
        metrics["darcy_t10"] = arrival_time(t_obs, c_obs, 0.1)
        metrics["darcy_t50"] = arrival_time(t_obs, c_obs, 0.5)
        metrics["t10_rel_err"] = abs(metrics["darcy_t10"] - metrics["stokes_t10"]) / max(
            abs(metrics["stokes_t10"]), 1.0e-30
        )
        metrics["t50_rel_err"] = abs(metrics["darcy_t50"] - metrics["stokes_t50"]) / max(
            abs(metrics["stokes_t50"]), 1.0e-30
        )
    return metrics
