from __future__ import annotations

import numpy as np

try:
    from darcy_vs_stokes.darcy_stokes_compare import arrival_time_cumulative, btc_series
    from darcy_vs_stokes.utils import effective_permeability
except ModuleNotFoundError:
    from darcy_stokes_compare import arrival_time_cumulative, btc_series
    from utils import effective_permeability


def summarize_flow(
    label: str,
    mask: np.ndarray,
    p: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    q_cols: np.ndarray,
    q_ref: float,
    mu: float,
    dP: float,
    dx: float,
) -> dict[str, float]:
    pore = mask == 1
    speed = np.sqrt(u**2 + v**2)
    q_cols = np.asarray(q_cols, dtype=float)
    q_in = float(q_cols[0]) if q_cols.size else float("nan")
    q_out = float(q_cols[-1]) if q_cols.size else float("nan")
    q_mean = float(np.mean(q_cols)) if q_cols.size else float("nan")
    q_std = float(np.std(q_cols)) if q_cols.size else float("nan")
    q_var = q_std / max(abs(q_mean), 1.0e-30) * 100.0
    L = mask.shape[1] * dx
    U_D = q_ref / max(mask.shape[0] * dx, 1.0e-30)
    k_eff = effective_permeability(q_ref, mask, dx, mu, dP)
    avg_h = mask.sum(axis=0).mean() * dx
    k_pois = avg_h**2 / 12.0
    U_pois = k_pois / float(mu) * dP / max(L, 1.0e-30)

    print(f"\n[{label}] Flow Diagnostics")
    print(f"  Porosity              = {mask.mean():.4f}")
    print(f"  Pressure range        = [{p[pore].min():.4f}, {p[pore].max():.4f}] Pa")
    print(f"  Q inlet   (face=1)    = {q_in:.6e}  m²/s")
    print(f"  Q outlet  (face=N-1)  = {q_out:.6e}  m²/s")
    print(f"  Q mean (all columns)  = {q_mean:.6e}  m²/s")
    print(f"  Q std dev             = {q_std:.6e}  m²/s")
    print(f"  Flux variation        = {q_var:.4f} %")
    print(f"  Darcy velocity U_D    = {U_D:.4e}  m/s")
    print(f"  Effective permeability= {k_eff:.4e}  m²")
    print(f"  Poiseuille reference  = {U_pois:.4e}  m/s")
    print(f"  Max pore speed        = {speed[pore].max():.4e}  m/s")
    print(f"  BTC reference flow    = {q_ref:.6e}  m²/s")
    return {
        "q_in": q_in,
        "q_out": q_out,
        "q_mean": q_mean,
        "q_std": q_std,
        "q_var": q_var,
        "u_D": U_D,
        "k_eff": k_eff,
        "U_pois": U_pois,
        "max_speed": float(speed[pore].max()),
    }


def print_reference_summary(stokes: dict[str, np.ndarray | float]) -> dict[str, float]:
    mask = np.asarray(stokes["mask"])
    print("=" * 58)
    print("  Darcy vs 2DStokes_FD sweep")
    print("=" * 58)
    print("\n[1] Geometry …")
    print(f"    Grid     = {mask.shape[0]} × {mask.shape[1]} (rows × cols)")
    print(f"    Porosity = {mask.mean():.4f}")
    print(
        f"    Domain   = {mask.shape[1] * float(stokes['dx']) * 1e6:.1f} × "
        f"{mask.shape[0] * float(stokes['dx']) * 1e6:.1f} µm  (X × Y)"
    )
    print("\n[2] Stokes reference …")
    print(f"  t_adv          = {float(stokes['t_adv']):.4e} s")
    print(f"  pulse_duration = {float(stokes['pulse_duration']):.4e} s")
    print(f"  dt             = {float(stokes['dt']):.4e} s")
    print(f"  nsteps         = {int(stokes['nsteps'])}")
    print(f"  gif stride     = {int(stokes['save_stride'])}")
    print("\n[3] Diagnostics …")
    ref = summarize_flow(
        "Stokes",
        mask,
        np.asarray(stokes["p"]),
        np.asarray(stokes["u"]),
        np.asarray(stokes["v"]),
        np.asarray(stokes["q_cols"]),
        float(stokes["q_ref"]),
        float(stokes["mu"]),
        float(stokes["dP"]),
        float(stokes["dx"]),
    )
    print("\n[4] Tracer transport …")
    stokes_injected = float(stokes["q_ref"]) * float(stokes["pulse_duration"])
    stokes_outlet = float(np.trapezoid(np.asarray(stokes["right_flux"]), np.asarray(stokes["times"])))
    if "total_mass" in stokes:
        stokes_remaining = float(np.asarray(stokes["total_mass"])[-1])
        print(f"  Final tracer mass  = {stokes_remaining:.6e}")
    else:
        stokes_remaining = float("nan")
        print("  Final tracer mass  = n/a")
    print(f"  Injected (q_ref*pulse) = {stokes_injected:.6e}")
    print(f"  Outlet integral q*C dt = {stokes_outlet:.6e}")
    if np.isfinite(stokes_remaining):
        bal = stokes_injected - (stokes_outlet + stokes_remaining)
        rel = bal / max(abs(stokes_injected), 1.0e-30)
        print(f"  Balance residual       = {bal:.6e} ({rel:.3%})")
    stokes_btc = btc_series(stokes, float(stokes["q_ref"]))
    print(f"  BTC t10 (cumul)   = {arrival_time_cumulative(np.asarray(stokes['times']), stokes_btc, 0.1):.6e} s")
    print(f"  BTC t50 (cumul)   = {arrival_time_cumulative(np.asarray(stokes['times']), stokes_btc, 0.5):.6e} s")
    return ref


def print_case_summary(
    label: str,
    row: dict[str, float],
    stokes: dict[str, np.ndarray | float],
    darcy: dict[str, np.ndarray | float],
    metrics: dict[str, float],
    timing: dict[str, float],
) -> None:
    print(f"\n[{label}] Darcy case")
    for key in ["pore_perm", "solid_perm", "pore_poro", "solid_poro"]:
        if key in row:
            print(f"  {key.replace('_', ' '):<10} = {row[key]:.3e}")
    if "runtime_s" in timing:
        print(f"  run time   = {timing['runtime_s']:.2f} s")
    print(f"  sim time   = {float(darcy['final_time']):.6e} s")
    stokes_final = float(np.asarray(stokes["times"])[-1])
    if abs(float(darcy["final_time"]) - stokes_final) > 1.0e-6 * max(stokes_final, 1.0):
        print(f"  WARNING    = Darcy final time != Stokes final time ({stokes_final:.6e} s)")
    summarize_flow(
        "Darcy",
        np.asarray(stokes["mask"]),
        np.asarray(darcy["p"]),
        np.asarray(darcy["u"]),
        np.asarray(darcy["v"]),
        np.asarray(darcy["q_cols"]),
        float(darcy["q_ref"]),
        float(stokes["mu"]),
        float(stokes["dP"]),
        float(stokes["dx"]),
    )
    print("  Comparison vs Stokes")
    print(f"    pressure rel RMSE = {metrics['pressure_rel_rmse']:.3e}")
    print(f"    speed rel RMSE    = {metrics['speed_rel_rmse']:.3e}")
    print(f"    BTC rel RMSE      = {metrics['btc_rel_rmse']:.3e}")
    print(f"    speed scale err   = {metrics['speed_scale_err']:.3e}")
    print(f"    t10 rel err       = {metrics['t10_rel_err']:.3e}")
    print(f"    t50 rel err       = {metrics['t50_rel_err']:.3e}")
    print(f"    speed rank corr   = {metrics['speed_rank_corr']:.3f}")
    print(f"    fast overlap      = {metrics['fast_channel_overlap']:.3f}")
    print(f"    stokes k_eff      = {metrics['stokes_k_eff']:.3e}")
    print(f"    total score       = {metrics['score']:.3e}")
    print("  Reactive Transport Readiness")
    print(f"    Q match err          = {metrics['speed_scale_err']*100:.1f}%")
    print(f"    Speed rank corr      = {metrics['speed_rank_corr']:.3f}")
    print(f"    Fast channel overlap = {metrics['fast_channel_overlap']*100:.1f}%")
    print(f"    BTC shape err        = {metrics['btc_rel_rmse']*100:.1f}%")
    print(f"    t50 timing err       = {metrics['t50_rel_err']*100:.1f}%")
    print("  Tracer transport")
    q_ref_common = max(float(stokes["q_ref"]), 1.0e-30)
    darcy_btc = btc_series(darcy, q_ref_common)
    pulse_duration = float(stokes["pulse_duration"])
    stokes_outlet_mass = float(np.trapezoid(np.asarray(stokes["right_flux"]), np.asarray(stokes["times"])))
    darcy_outlet_mass = float(np.trapezoid(np.asarray(darcy["right_flux"]), np.asarray(darcy["times"])))
    stokes_injected = float(stokes["q_ref"]) * pulse_duration
    darcy_injected = float(darcy["q_ref"]) * pulse_duration
    print(f"    BTC t10           = {metrics['darcy_t10']:.6e} s")
    print(f"    BTC t50           = {metrics['darcy_t50']:.6e} s")
    print(f"    Stokes injected    = {stokes_injected:.6e}")
    print(f"    Darcy injected     = {darcy_injected:.6e}")
    print(f"    Stokes outlet q*Cdt= {stokes_outlet_mass:.6e}")
    print(f"    Darcy outlet q*Cdt = {darcy_outlet_mass:.6e}")
    print(f"    Stokes out/inj     = {stokes_outlet_mass / max(stokes_injected, 1.0e-30):.6f}")
    print(f"    Darcy out/inj      = {darcy_outlet_mass / max(darcy_injected, 1.0e-30):.6f}")
    print(f"    Max BTC            = {float(np.max(darcy_btc)):.6e}")
