"""
2-D Stokes Flow — Staggered MAC-Grid Coupled Solver
====================================================
Solves the FULL coupled system on a staggered MAC grid:

  Grid layout (i↓ row, j→ col):
    p  : cell centres      (Ny,    Nx   )
    u  : vertical faces    (Ny,    Nx+1 )   u[i,j] = left face of cell(i,j)
    v  : horizontal faces  (Ny+1,  Nx   )   v[i,j] = top  face of cell(i,j)

  DOF order:  [ u  Ny*(Nx+1),  v  (Ny+1)*Nx,  p  Ny*Nx ]

  Momentum (active face — both neighbours fluid, not domain edge):
    μ·Lap(u)[i,j] = (p[i,j] − p[i,j−1]) / dx
    μ·Lap(v)[i,j] = (p[i,j] − p[i−1,j]) / dx

  Continuity (interior fluid cell, j ≠ 0, Nx−1):
    (u[i,j+1] − u[i,j]) / dx + (v[i+1,j] − v[i,j]) / dx = 0

  Pressure BCs:
    p = dP  at fluid inlet  cells (j = 0)
    p = 0   at fluid outlet cells (j = Nx−1)

  Velocity BCs:
    u = v = 0  at domain edges and faces bordering solid cells

Why this works:
  • Staggered grid eliminates the checkerboard (∇·u=0 uses ADJACENT u/v DOFs,
    not skip-one central differences).
  • Continuity is in the linear system, not a post-solve check.
  • u, v, p solved simultaneously — no decoupling.
"""

import argparse
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import label
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
# 1. GEOMETRY
# ══════════════════════════════════════════════════════════════════════

def load_geometry(path):
    """Load a pore mask from the .npy file saved by build_pflotran_hdf5_from_npy.py.

    Returns
    -------
    mask : np.ndarray, shape (Ny, Nx), float64
        1.0 = pore, 0.0 = solid.  Transposed from the (nx, ny) storage
        convention so that solver columns (j) run along the flow (X) direction.
    """
    if not path.endswith('.npy'):
        raise ValueError(
            f"Expected a .npy file from build_pflotran_hdf5_from_npy.py, got: '{path}'")
    img  = np.load(path)                    # shape (nx, ny), 1=pore
    mask = img.T.astype(np.float64)         # → (ny, nx): rows=Y, cols=X
    print(f"  Loaded geometry: {img.shape[0]}×{img.shape[1]} "
          f"(porosity={img.mean():.3f})")
    return mask


def make_pore_mask(N=64):
    try:
        from skimage.draw import disk
        def draw_disk(mask, cy, cx, r):
            rr, cc = disk((cy, cx), r, shape=mask.shape)
            mask[rr, cc] = 0
    except ImportError:
        def draw_disk(mask, cy, cx, r):
            for di in range(-r-1, r+2):
                for dj in range(-r-1, r+2):
                    if di*di + dj*dj <= r*r:
                        ii, jj = cy+di, cx+dj
                        if 0 <= ii < N and 0 <= jj < N:
                            mask[ii, jj] = 0

    mask = np.ones((N, N), dtype=np.uint8)
    grains = [
        (55,50,9),(50,42,7),(44,54,8),(52,33,5),
        (32,30,7),(23,18,8),(37,20,5),(28,47,7),
        (18,50,7),(13,28,5),(20,4,4),(36,8,5),
        (55,10,4),(45,14,3),(8,20,3),(5,46,3),
        (30,56,4),(10,55,5),(55,28,4),(60,40,4),
        (25,38,5),(15,38,4),(42,38,4),(3,35,3),(2,18,3),
    ]
    for (cy, cx, r) in grains:
        draw_disk(mask, cy, cx, r)
    return mask.astype(np.float64)


def keep_percolating(mask):
    labeled, n = label(mask)
    cands = []
    for cid in range(1, n+1):
        c = (labeled == cid)
        if c[:, 0].any() and c[:, -1].any():
            cands.append((c.sum(), cid))
    if not cands:
        return mask
    cands.sort(reverse=True)
    new_mask = (labeled == cands[0][1]).astype(np.float64)
    print(f"  Percolating pore: {int(new_mask.sum())} cells "
          f"(removed {int(mask.sum()-new_mask.sum())} isolated)")
    return new_mask


# ══════════════════════════════════════════════════════════════════════
# 2. STAGGERED MAC STOKES SOLVER
# ══════════════════════════════════════════════════════════════════════

def solve_stokes_mac(mask, dP=10.0, mu=1e-3, dx=1e-6):
    """
    Staggered MAC-grid Stokes solver — fully coupled direct solve.

    A face is *active* (carries a momentum equation) iff:
      - it is not on the domain boundary, AND
      - BOTH cells it borders are fluid.
    Otherwise the face gets u=0 or v=0 (no-slip / no-penetration).

    The continuity equation uses ADJACENT u/v face DOFs
    (not skip-one central differences), eliminating checkerboard.
    """
    Ny, Nx = mask.shape

    # DOF counts
    Nu   = Ny * (Nx + 1)      # u-faces: Ny rows × (Nx+1) cols
    Nv   = (Ny + 1) * Nx      # v-faces: (Ny+1) rows × Nx cols
    Np   = Ny * Nx             # p-cells
    NTOT = Nu + Nv + Np

    U_OFF = 0
    V_OFF = Nu
    P_OFF = Nu + Nv

    def iu(i, j): return U_OFF + i*(Nx+1) + j   # j ∈ [0, Nx]
    def iv(i, j): return V_OFF + i*Nx + j        # i ∈ [0, Ny]
    def ip(i, j): return P_OFF + i*Nx + j

    # ── face activity masks ───────────────────────────────────────────
    # u-face (i,j): connects cells (i,j-1) and (i,j); active for j∈[1,Nx-1]
    u_act = np.zeros((Ny, Nx+1), dtype=bool)
    u_act[:, 1:Nx] = (mask[:, :Nx-1] == 1) & (mask[:, 1:] == 1)

    # v-face (i,j): connects cells (i-1,j) and (i,j); active for i∈[1,Ny-1]
    v_act = np.zeros((Ny+1, Nx), dtype=bool)
    v_act[1:Ny, :] = (mask[:Ny-1, :] == 1) & (mask[1:, :] == 1)

    A = lil_matrix((NTOT, NTOT), dtype=np.float64)
    b = np.zeros(NTOT)

    c1 = mu / dx**2   # viscous stencil coefficient
    c2 = 1.0 / dx     # pressure gradient / continuity coefficient

    # ── U-MOMENTUM  (Ny × (Nx+1) faces) ─────────────────────────────
    # μ·Lap(u)[i,j] = (p[i,j] − p[i,j−1]) / dx
    for i in range(Ny):
        for j in range(Nx + 1):
            ku = iu(i, j)
            if not u_act[i, j]:
                A[ku, ku] = 1.0          # u = 0 (BC)
            else:
                A[ku, ku] = -4.0 * c1
                for ni, nj in [(i, j-1), (i, j+1), (i-1, j), (i+1, j)]:
                    if 0 <= ni < Ny and 0 <= nj <= Nx and u_act[ni, nj]:
                        A[ku, iu(ni, nj)] += c1
                # pressure gradient: μ·Lap(u) − (p_R − p_L)/dx = 0
                A[ku, ip(i, j  )] += -c2
                A[ku, ip(i, j-1)] +=  c2

    # ── V-MOMENTUM  ((Ny+1) × Nx faces) ─────────────────────────────
    # μ·Lap(v)[i,j] = (p[i,j] − p[i−1,j]) / dx
    for i in range(Ny + 1):
        for j in range(Nx):
            kv = iv(i, j)
            if not v_act[i, j]:
                A[kv, kv] = 1.0          # v = 0 (BC)
            else:
                A[kv, kv] = -4.0 * c1
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                    if 0 <= ni <= Ny and 0 <= nj < Nx and v_act[ni, nj]:
                        A[kv, iv(ni, nj)] += c1
                # pressure gradient (y: i increases downward)
                A[kv, ip(i,   j)] += -c2
                A[kv, ip(i-1, j)] +=  c2

    # ── PRESSURE / CONTINUITY  (Ny × Nx cells) ───────────────────────
    # (u[i,j+1] − u[i,j])/dx + (v[i+1,j] − v[i,j])/dx = 0
    for i in range(Ny):
        for j in range(Nx):
            kp = ip(i, j)
            if mask[i, j] == 0:
                A[kp, kp] = 1.0                  # solid: dummy p = 0
            elif j == 0:
                A[kp, kp] = 1.0; b[kp] = dP      # inlet  Dirichlet
            elif j == Nx - 1:
                A[kp, kp] = 1.0; b[kp] = 0.0     # outlet Dirichlet
            else:
                # continuity uses ADJACENT face DOFs (no skip-one)
                A[kp, iu(i, j+1)] +=  c2
                A[kp, iu(i, j  )] += -c2
                A[kp, iv(i+1, j)] +=  c2
                A[kp, iv(i,   j)] += -c2

    # ── fix only genuinely empty rows (isolated DOFs) ────────────────
    # NOTE: continuity rows have zero diagonal by design (saddle-point
    # system); only rows with zero ROW NORM are truly broken.
    A_csr    = csr_matrix(A)
    row_norms = np.asarray(np.abs(A_csr).sum(axis=1)).ravel()
    bad       = np.where(row_norms == 0)[0]
    if bad.size:
        print(f"  Fixing {bad.size} empty rows …")
        Al = A_csr.tolil()
        for k in bad:
            Al[k, k] = 1.0
        A_csr = csr_matrix(Al)

    print(f"  System: {NTOT} DOFs,  {A_csr.nnz} non-zeros")
    print("  Solving …")
    sol = spsolve(A_csr, b)
    sol = np.nan_to_num(sol, nan=0.0, posinf=0.0, neginf=0.0)

    # ── unpack staggered fields ───────────────────────────────────────
    u_face = sol[U_OFF:V_OFF].reshape(Ny, Nx+1)
    v_face = sol[V_OFF:P_OFF].reshape(Ny+1, Nx)
    p      = sol[P_OFF:     ].reshape(Ny, Nx)

    # interpolate to cell centres for output / visualisation
    u = 0.5 * (u_face[:, :-1] + u_face[:, 1:]) * mask
    v = 0.5 * (v_face[:-1, :] + v_face[1:, :]) * mask

    pore = mask == 1
    print(f"  p ∈ [{p[pore].min():.4f}, {p[pore].max():.4f}] Pa")
    print(f"  u ∈ [{u[pore].min():.4e}, {u[pore].max():.4e}] m/s")
    print(f"  v ∈ [{v[pore].min():.4e}, {v[pore].max():.4e}] m/s")

    return p, u, v, u_face, v_face


# ══════════════════════════════════════════════════════════════════════
# 3. DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════

def diagnostics(mask, u, v, u_face, p, mu, dP, dx):
    Ny, Nx = mask.shape
    pore   = mask == 1
    speed  = np.sqrt(u**2 + v**2)

    # flux through each active x-face cross-section
    q_cols = np.sum(u_face[:, 1:Nx], axis=0) * dx
    q_mean = q_cols.mean()
    q_std  = q_cols.std()
    q_var  = q_std / max(abs(q_mean), 1e-30) * 100

    q_in = float(np.sum(u_face[:, 1])) * dx
    q_out = float(np.sum(u_face[:, -2])) * dx
    q_ref = q_out if abs(q_out) > 1e-30 else q_in

    L     = Nx * dx
    U_D   = q_ref / (Ny * dx)
    k_eff = mu * abs(U_D) * L / max(abs(dP), 1e-30)

    avg_h   = mask.sum(axis=0).mean() * dx
    k_pois  = avg_h**2 / 12.0
    U_pois  = k_pois / mu * dP / L

    print("\n─── Flow Diagnostics ──────────────────────────────────────")
    print(f"  Porosity              = {mask.mean():.4f}")
    print(f"  Pressure range        = [{p[pore].min():.4f},"
          f" {p[pore].max():.4f}] Pa")
    print(f"  Q inlet   (face=1)    = {q_in:.6e}  m²/s")
    print(f"  Q outlet  (face=N-1)  = {q_out:.6e}  m²/s")
    print(f"  Q mean (all columns)  = {q_mean:.6e}  m²/s")
    print(f"  Q std dev             = {q_std:.6e}  m²/s")
    print(f"  Flux variation        = {q_var:.4f} %  "
          f"({'✓ good' if q_var < 5 else '✗ check BCs'})")
    print(f"  Darcy velocity U_D    = {U_D:.4e}  m/s")
    print(f"  Effective permeability= {k_eff:.4e}  m²")
    print(f"  Poiseuille reference  = {U_pois:.4e}  m/s")
    print(f"  Max pore speed        = {speed[pore].max():.4e}  m/s")
    print("───────────────────────────────────────────────────────────")
    print(f"  BTC reference flow    = {q_ref:.6e}  m²/s")
    return q_cols, q_mean, q_ref


# ══════════════════════════════════════════════════════════════════════
# 4. TRACER TRANSPORT
# ══════════════════════════════════════════════════════════════════════

def run_tracer(u_face, v_face, mask, dx_phys=1e-6, n_steps=1000, cfl=0.35,
               pulse_duration=1.0, save_stride=1):
    """Advance tracer with an inlet pulse active only for ``pulse_duration``."""
    Ny, Nx  = mask.shape
    u_ctr = 0.5 * (u_face[:, :-1] + u_face[:, 1:]) * mask
    v_ctr = 0.5 * (v_face[:-1, :] + v_face[1:, :]) * mask
    max_spd = np.sqrt(u_ctr**2 + v_ctr**2)[mask==1].max()

    if max_spd < 1e-30:
        print("  Zero velocity — skipping.")
        z = np.zeros(n_steps)
        return z, z, np.zeros((Ny, Nx)), [], np.zeros(0), np.zeros(n_steps)

    dt = cfl * dx_phys / max_spd
    print(f"  max_speed = {max_spd:.4e} m/s  |  "
          f"dt = {dt:.4e} s  |  "
          f"T_total = {dt*n_steps:.4e} s")

    C          = np.zeros((Ny, Nx))
    times      = np.zeros(n_steps)
    right_flux = np.zeros(n_steps)
    total_mass = np.zeros(n_steps)
    tracer_frames = []
    frame_times   = []

    for step in range(n_steps):
        t_now = step * dt
        c_in = 1.0 if t_now < pulse_duration else 0.0

        Fx = np.zeros((Ny, Nx + 1))
        up_x = np.maximum(u_face[:, 1:Nx], 0.0)
        un_x = np.minimum(u_face[:, 1:Nx], 0.0)
        Fx[:, 1:Nx] = up_x * C[:, :-1] + un_x * C[:, 1:]

        left_pore = mask[:, 0] == 1
        Fx[left_pore, 0] = np.maximum(u_face[left_pore, 1], 0.0) * c_in

        right_pore = mask[:, -1] == 1
        Fx[right_pore, -1] = np.maximum(u_face[right_pore, -2], 0.0) * C[right_pore, -1]

        Fy = np.zeros((Ny + 1, Nx))
        up_y = np.maximum(v_face[1:Ny, :], 0.0)
        un_y = np.minimum(v_face[1:Ny, :], 0.0)
        Fy[1:Ny, :] = up_y * C[:-1, :] + un_y * C[1:, :]

        divx = (Fx[:, 1:] - Fx[:, :-1]) / dx_phys
        divy = (Fy[1:, :] - Fy[:-1, :]) / dx_phys

        C = np.clip(C - dt*(divx+divy), 0.0, 1.0) * mask

        right_flux[step] = float(np.sum(Fx[:, -1])) * dx_phys
        total_mass[step] = float(np.sum(C * mask)) * dx_phys * dx_phys
        times[step]      = t_now
        if step % save_stride == 0 or step == n_steps - 1:
            tracer_frames.append(C.copy())
            frame_times.append(t_now)

    return times, right_flux, C, tracer_frames, np.array(frame_times), total_mass


def estimate_tracer_settings(
    u,
    v,
    mask,
    dx_phys,
    cfl,
    pulse_fraction,
    final_time,
    gif_frames,
    tfinal_mult=None,
):
    """Pick tracer timing from an absolute run duration."""
    _, Nx = mask.shape
    speed = np.sqrt(u**2 + v**2)
    pore_speed = speed[mask == 1]
    max_speed = pore_speed.max()

    if max_speed < 1e-30:
        return 1.0, 1, 1, 1.0, 1.0

    dt = cfl * dx_phys / max_speed
    t_adv = (Nx * dx_phys) / max_speed
    pulse_duration = pulse_fraction * t_adv
    if final_time is None:
        if tfinal_mult is None:
            tfinal_mult = 3.0
        t_final = tfinal_mult * t_adv
    else:
        t_final = float(final_time)
    n_steps = max(1, int(np.ceil(t_final / dt)))
    save_stride = max(1, int(np.ceil(n_steps / max(gif_frames, 1))))
    return dt, n_steps, save_stride, pulse_duration, t_adv


def save_results(path, mask, u, v, p, q_cols, times, right_flux, C_final,
                 tracer_frames, frame_times, total_mass, dP, q_mean, q_ref,
                 mu, dx, dt, nsteps, save_stride, pulse_duration, t_adv):
    """Save solver outputs for later plotting."""
    np.savez_compressed(
        path,
        mask=mask,
        u=u,
        v=v,
        p=p,
        q_cols=q_cols,
        times=times,
        right_flux=right_flux,
        C_final=C_final,
        tracer_frames=np.stack(tracer_frames) if tracer_frames else np.zeros(
            (0, *mask.shape), dtype=np.float64),
        frame_times=frame_times,
        total_mass=total_mass,
        dP=np.array(dP),
        q_mean=np.array(q_mean),
        q_ref=np.array(q_ref),
        mu=np.array(mu),
        dx=np.array(dx),
        dt=np.array(dt),
        nsteps=np.array(nsteps),
        save_stride=np.array(save_stride),
        pulse_duration=np.array(pulse_duration),
        t_adv=np.array(t_adv),
    )
    print(f"  Results saved → {path}")


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='2-D Stokes pore-scale solver (staggered MAC grid)')
    parser.add_argument(
        '--geometry', '-g', default=None,
        help='Path to geometry file produced by build_pflotran_hdf5_from_npy.py '
             '(.npy or .h5).  Omit to use the built-in synthetic geometry.')
    parser.add_argument('--mu',  type=float, default=1e-3,
                        help='Dynamic viscosity [Pa·s] (default 1e-3)')
    parser.add_argument('--dP',  type=float, default=1.0,
                        help='Pressure drop inlet→outlet [Pa] (default 1)')
    parser.add_argument('--dx',  type=float, default=1e-6,
                        help='Cell size [m] (default 1e-6)')
    parser.add_argument('--nsteps', type=int, default=None,
                        help='Tracer time-steps override (default auto)')
    parser.add_argument('--pulse-fraction', type=float, default=0.10,
                        help='Pulse length as fraction of advective time')
    parser.add_argument('--final-time', type=float, default=None,
                        help='Absolute tracer run duration [s] (overrides --tfinal-mult)')
    parser.add_argument('--tfinal-mult', type=float, default=3.0,
                        help='Fallback total tracer run as multiple of advective time')
    parser.add_argument('--gif-frames', type=int, default=25,
                        help='Target number of frames in tracer GIF')
    parser.add_argument('--results-out', default='stokes_results.npz',
                        help='Path to saved solver results (.npz)')
    args = parser.parse_args()

    mu = args.mu
    dP = args.dP
    dx = args.dx

    print("=" * 58)
    print("  2-D Stokes Pore-Scale Solver (Staggered MAC)")
    print("=" * 58)

    print("\n[1] Geometry …")
    if args.geometry is not None:
        mask = load_geometry(args.geometry)
        mask = keep_percolating(mask)
    else:
        N    = 64
        mask = make_pore_mask(N=N)
        mask = keep_percolating(mask)
    Ny, Nx = mask.shape
    print(f"    Grid     = {Ny} × {Nx} (rows × cols)")
    print(f"    Porosity = {mask.mean():.4f}")
    print(f"    Domain   = {Nx*dx*1e6:.1f} × {Ny*dx*1e6:.1f} µm  (X × Y)")

    print("\n[2] Solving staggered MAC Stokes system …")
    p, u, v, u_face, v_face = solve_stokes_mac(mask, dP=dP, mu=mu, dx=dx)

    print("\n[3] Diagnostics …")
    q_cols, q_mean, q_ref = diagnostics(mask, u, v, u_face, p, mu, dP, dx)

    print("\n[4] Tracer transport …")
    dt_est, nsteps_auto, stride, pulse_duration, t_adv = estimate_tracer_settings(
        u, v, mask, dx, 0.35, args.pulse_fraction, args.final_time,
        args.gif_frames, tfinal_mult=args.tfinal_mult)
    nsteps = args.nsteps if args.nsteps is not None else nsteps_auto
    if args.nsteps is not None:
        stride = max(1, int(np.ceil(nsteps / max(args.gif_frames, 1))))
    print(f"  t_adv          = {t_adv:.4e} s")
    print(f"  pulse_duration = {pulse_duration:.4e} s")
    print(f"  dt             = {dt_est:.4e} s")
    print(f"  nsteps         = {nsteps}")
    print(f"  gif stride     = {stride}")
    times, right_flux, C_final, tracer_frames, frame_times, total_mass = run_tracer(
        u_face, v_face, mask, dx_phys=dx, n_steps=nsteps, cfl=0.35,
        pulse_duration=pulse_duration, save_stride=stride)

    outlet_cumulative = float(np.trapezoid(right_flux, times))
    remaining_mass = float(total_mass[-1]) if len(total_mass) else 0.0
    print(f"  Final tracer mass  = {remaining_mass:.6e}")
    print(f"  Outlet mass total  = {outlet_cumulative:.6e}")

    print("\n[5] Saving results …")
    save_results(
        args.results_out, mask, u, v, p, q_cols, times, right_flux, C_final,
        tracer_frames, frame_times, total_mass, dP, q_mean, q_ref, mu, dx,
        dt_est, nsteps, stride, pulse_duration, t_adv)
    print("Done ✓")


if __name__ == "__main__":
    main()
