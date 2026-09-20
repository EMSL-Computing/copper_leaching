"""
usage:

python plot_stokes_solver_2D.py --results stokes_fd_reference.npz
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize


def visualise(mask, u, v, p, q_cols, times,
              right_flux, C_final, dP, q_mean, q_ref, out_path):
    Ny, Nx = mask.shape
    X, Y   = np.meshgrid(np.arange(Nx), np.arange(Ny))
    ext    = [0, Nx, 0, Ny]

    so = np.zeros((Ny, Nx, 4), dtype=np.float32)
    so[mask == 0] = [0.1, 0.1, 0.1, 1.0]
    kw = dict(origin='lower', extent=ext, interpolation='nearest')

    def mf(f):
        r = f.copy().astype(float)
        r[mask == 0] = np.nan
        return r

    fig, axs = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    fig.suptitle(
        "2-D Stokes Flow  –  Pore-Scale  (Coupled FD Solver)   | "
        r"$\mu\nabla^2\mathbf{u} = \nabla p$  │  "
        r"$\nabla\cdot\mathbf{u} = 0$  │  "
        f"ΔP = {dP:.0f} Pa  │  μ = 1 mPa·s  │  dx = 1 µm",
        fontsize=13, fontweight='bold')

    ax = axs[0, 0]
    im = ax.imshow(mf(p), cmap='RdBu_r', vmin=0, vmax=dP, **kw)
    ax.imshow(so, origin='lower', extent=ext)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("Pressure [Pa]")
    ax.set_title("Pressure Field  $p$", fontsize=11)
    ax.set_xlabel("X  (flow →)")
    ax.set_ylabel("Y")

    ax = axs[0, 1]
    speed = np.sqrt(u**2 + v**2)
    vmax_sp = np.percentile(speed[mask == 1], 99)
    im = ax.imshow(mf(speed), cmap='RdBu_r', vmin=0, vmax=vmax_sp, **kw)
    ax.imshow(so, origin='lower', extent=ext)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("Speed [m/s]")
    sk = 3
    Usk = u[::sk, ::sk].copy()
    Vsk = v[::sk, ::sk].copy()
    spsk = np.sqrt(Usk**2 + Vsk**2)
    Usk[spsk < 1e-20] = 0
    Vsk[spsk < 1e-20] = 0
    try:
        ax.streamplot(X[::sk, ::sk], Y[::sk, ::sk], Usk, Vsk,
                      color='white', linewidth=0.6,
                      density=1.2, arrowsize=0.7, minlength=0.3)
    except Exception as e:
        print(f"  streamplot skipped: {e}")
    ax.set_title("Velocity Magnitude + Streamlines", fontsize=11)
    ax.set_xlabel("X  (flow →)")
    ax.set_ylabel("Y")

    ax = axs[0, 2]
    vlim = np.percentile(np.abs(u[mask == 1]), 99)
    im = ax.imshow(mf(u), cmap='seismic', vmin=-vlim, vmax=vlim, **kw)
    ax.imshow(so, origin='lower', extent=ext)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("u [m/s]")
    ax.set_title("x-Velocity  $u$", fontsize=11)
    ax.set_xlabel("X  (flow →)")
    ax.set_ylabel("Y")

    ax = axs[1, 0]
    im = ax.imshow(mf(C_final), cmap='plasma', vmin=0, vmax=1, **kw)
    ax.imshow(so, origin='lower', extent=ext)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("C / C₀")
    ax.set_title(f"Tracer  (t = {times[-1]:.3e} s)", fontsize=11)
    ax.set_xlabel("X  (flow →)")
    ax.set_ylabel("Y")

    ax = axs[1, 1]
    jj = np.arange(len(q_cols))
    ax.plot(jj, q_cols, color='steelblue', linewidth=1.5)
    ax.axhline(q_mean, color='orange', linewidth=1.5, linestyle='--',
               label=f'mean = {q_mean:.3e} m²/s')
    ax.fill_between(jj, q_cols, alpha=0.2, color='steelblue')
    ax.set_title("Column Flux Q(j)  —  Mass Conservation\n"
                 "(flat = perfect ∇·u = 0)", fontsize=10)
    ax.set_xlabel("Face index j")
    ax.set_ylabel("Q [m²/s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=':')

    ax = axs[1, 2]
    Qss = abs(q_ref) if abs(q_ref) > 1e-30 else 1.0
    btc = right_flux / Qss
    ax.plot(times, btc, color='royalblue', linewidth=2, label='BTC  C/C₀')
    ax.fill_between(times, btc, alpha=0.18, color='royalblue')

    for frac, col, ls in [(0.5, 'red', '--'), (0.9, 'darkorange', ':')]:
        if btc.max() >= frac:
            t_br = times[np.argmax(btc >= frac)]
            ax.axvline(t_br, color=col, linestyle=ls,
                       linewidth=1.4,
                       label=f't_{int(frac * 100)} = {t_br:.3e} s')

    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("C / C₀", fontsize=10)
    ax.set_title("Breakthrough Curve\n"
                 "(right-face integrated tracer flux)", fontsize=10)
    ax.set_ylim(0.0, max(float(btc.max()), 1e-12))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.30, linestyle=':')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Figure saved → {out_path}")


def animate_tracer(mask, tracer_frames, frame_times, out_path="tracer.gif"):
    """Save a simple tracer GIF."""
    if len(tracer_frames) == 0:
        print("  No tracer frames — skipping GIF.")
        return

    Ny, Nx = mask.shape
    ext = [0, Nx, 0, Ny]
    solids = np.zeros((Ny, Nx, 4), dtype=np.float32)
    solids[mask == 0] = [0.1, 0.1, 0.1, 1.0]
    log_floor = -8.0

    def log_field(f):
        r = np.full(f.shape, np.nan, dtype=float)
        pore = mask == 1
        r[pore] = np.log10(np.clip(f[pore], 10**log_floor, None))
        return r

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(log_field(tracer_frames[0]), cmap='plasma',
                   norm=Normalize(vmin=log_floor, vmax=0.0),
                   origin='lower', extent=ext, interpolation='nearest')
    ax.imshow(solids, origin='lower', extent=ext)
    plt.colorbar(im, ax=ax, shrink=0.88).set_label("log10(C / C₀)")
    title = ax.set_title(f"Tracer  (t = {frame_times[0]:.2f} s)")
    ax.set_xlabel("X  (flow →)")
    ax.set_ylabel("Y")

    def update(frame_idx):
        im.set_data(log_field(tracer_frames[frame_idx]))
        title.set_text(f"Tracer  (t = {frame_times[frame_idx]:.2f} s)")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(tracer_frames),
                         interval=80, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=4), dpi=180)
    plt.close(fig)
    print(f"  Tracer GIF saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot saved 2-D Stokes pore-scale results')
    parser.add_argument('--results', default='stokes_results.npz',
                        help='Path to saved solver results (.npz)')
    parser.add_argument('--out-dir', default='.',
                        help='Directory to save figures')
    parser.add_argument('--no-gif', action='store_true',
                        help='Skip tracer GIF generation')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.results)

    mask = data['mask']
    u = data['u']
    v = data['v']
    p = data['p']
    q_cols = data['q_cols']
    times = data['times']
    right_flux = data['right_flux']
    C_final = data['C_final']
    tracer_frames = data['tracer_frames']
    frame_times = data['frame_times']
    dP = float(data['dP'])
    q_mean = float(data['q_mean'])
    q_ref = float(data['q_ref']) if 'q_ref' in data.files else q_mean

    if not args.no_gif:
        print("\n[1] Tracer GIF …")
        animate_tracer(mask, tracer_frames, frame_times,
                       out_path=str(out_dir / "tracer.gif"))

    print("\n[2] Plotting …")
    visualise(mask, u, v, p, q_cols, times, right_flux, C_final, dP, q_mean,
              q_ref, out_dir / "stokes_pore_flow.png")


if __name__ == "__main__":
    main()
