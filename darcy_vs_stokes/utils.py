"""Tools for Darcy vs 2DStokes_FD comparison.

"""

from pathlib import Path
import h5py
import numpy as np
import subprocess


def run_cmd(args: list[str], workdir: Path) -> None:
    result = subprocess.run(
        args,
        cwd=workdir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)
 

def effective_permeability(q_ref: float, mask: np.ndarray, dx: float, mu: float, dP: float) -> float:
    """Compute effective permeability from Darcy's law applied to the whole domain.

    Uses the bulk relation k_eff = mu * U_D * L / dP, where U_D = Q / (Ny * dx)
    is the superficial Darcy velocity and L = Nx * dx is the domain length in the
    flow direction [2].

    Args:
        q_ref: Total volumetric outlet flux [m2/s] (2-D, per unit depth).
               Equal to sum of outlet face velocities times dx [2].
        mask:  2-D integer array, shape (Ny, Nx). Pore cells = 1, solid = 0.
               Ny is the cross-section width; Nx is the flow-direction length.
        dx:    Uniform cell size [m] in both x and y.
        mu:    Dynamic viscosity [Pa·s].
        dP:    Applied pressure drop inlet minus outlet [Pa].

    Returns:
        k_eff: Effective permeability [m2] of the domain treated as a
               homogeneous Darcy medium. This is a whole-domain quantity that
               already accounts for solid blockage and tortuosity. It is NOT
               the local pore-cell permeability k_pore used in PFLOTRAN [4].
               To obtain k_pore, scale by the geometry factor:
                   k_pore = k_eff * (k_pore_trial / k_eff_trial)
               or equivalently:
                   k_pore = k_pore_trial * (q_ref_stokes / q_ref_darcy_trial)
    """
    L = mask.shape[1] * dx
    U_D = q_ref / (mask.shape[0] * dx)
    return mu * abs(U_D) * L / dP

