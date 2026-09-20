"""Write darcy_tracer.in from build_pflotran_hdf5_from_npy.py outputs.

python write_darcy_tracer_input.py --geometry pore_geometry.h5 --out darcy_tracer.in --dx 20e-6

Supports explicit pore/solid diffusivities so the generated input uses the correct
base DIFFUSION_COEFFICIENT and per-material TORTUOSITY values.

Example:
python darcy_vs_stokes/write_darcy_tracer_input.py \
  --geometry Chip_Designs/network1/geometry/network1_correct.h5 \
  --out darcy_long_tracer.in \
  --pore-diffusivity 1e-9 --solid-diffusivity 1e-11 \
  --final-time 604800 --pulse-duration 172800
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def fmt_pf(x: float) -> str:
    """Format float in PFLOTRAN-style scientific notation."""
    s = f"{x:.12e}"
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    sign = exp[0]
    digits = exp[1:].lstrip("0") or "0"
    return f"{mant}d{sign}{digits}"


def load_grid_shape(geometry_h5: Path, geometry_npy: Path | None = None) -> tuple[int, int]:
    """Read nx, ny from explicit/sidecar .npy, HDF5 metadata, or square cell count."""
    geometry_npy = geometry_npy or geometry_h5.with_suffix(".npy")
    if geometry_npy.exists():
        img = np.load(geometry_npy)
        if img.ndim != 2:
            raise ValueError(f"Expected 2-D geometry array in '{geometry_npy}', got shape {img.shape}")
        with h5py.File(geometry_h5, "r") as h5:
            if "Cell Ids" in h5:
                ncells = int(np.asarray(h5["Cell Ids"]).size)
                if img.size != ncells:
                    raise ValueError(
                        f"Geometry shape from '{geometry_npy}' has {img.size} cells, "
                        f"but '{geometry_h5}' has {ncells} cells."
                    )
        nx, ny = img.shape
        return nx, ny

    with h5py.File(geometry_h5, "r") as h5:
        if "Metadata" in h5 and "grid_shape" in h5["Metadata"].attrs:
            nx, ny = [int(v) for v in h5["Metadata"].attrs["grid_shape"]]
            return nx, ny
        if "Cell Ids" not in h5:
            raise KeyError(
                f"'{geometry_h5}' is missing required dataset 'Cell Ids' for shape inference"
            )
        ncells = int(np.asarray(h5["Cell Ids"]).size)

    # Fallback for chip-design geometries: infer square grid from flattened cell count.
    nside = int(round(ncells**0.5))
    if nside * nside == ncells:
        return nside, nside

    raise FileNotFoundError(
        f"Need sidecar grid file '{geometry_npy}' (or a square cell count for fallback); "
        f"cannot infer unique (nx, ny) from ncells={ncells}."
    )


def build_input_text(
    geometry_h5: Path,
    nx: int,
    ny: int,
    dx: float,
    dz: float,
    p_in: float,
    p_out: float,
    pulse_duration: float,
    final_time: float,
    output_period: float,
    initial_dt: float,
    max_dt: float,
    diffusion_coefficient: float = None,  # legacy
    pore_diffusivity: float = 1.0e-9,
    solid_diffusivity: float = 1.0e-11,
    background_concentration: float = 1.0e-10,
    tracer_on_concentration: float = 1.0,
) -> str:
    """Build PFLOTRAN tracer input text.
    If pore_diffusivity/solid_diffusivity are used (new path), computes
    base DIFFUSION_COEFFICIENT and per-material TORTUOSITY.
    Falls back to legacy diffusion_coefficient + extreme solid tort for old callers.
    """
    if diffusion_coefficient is not None:
        # legacy path (used by simulate_darcy flow writer)
        base_diff = diffusion_coefficient
        pore_tort = 1.0
        solid_tort = 1.0e-20
    else:
        base_diff = max(pore_diffusivity, solid_diffusivity)
        pore_tort = pore_diffusivity / base_diff
        solid_tort = solid_diffusivity / base_diff

    lx = nx * dx
    ly = ny * dx
    zmax = dz
    ymid = 0.5 * ly
    zmid = 0.5 * zmax
    geom_name = geometry_h5.name

    return f"""# Darcy + tracer case on shared pore geometry

SIMULATION
  SIMULATION_TYPE SUBSURFACE
  PROCESS_MODELS
    SUBSURFACE_FLOW flow
      MODE RICHARDS
    /
    SUBSURFACE_TRANSPORT transport
      MODE GIRT
    /
  /
END

SUBSURFACE

NUMERICAL_METHODS FLOW
  LINEAR_SOLVER
    SOLVER DIRECT
  /
END

NUMERICAL_METHODS TRANSPORT
  LINEAR_SOLVER
    SOLVER DIRECT
  /
END

!------------------------------------------------------------------------------
! Grid: {nx} x {ny} x 1, dx = dy = {dx:.6g} m
!------------------------------------------------------------------------------

GRID
  TYPE STRUCTURED
  NXYZ {nx} {ny} 1
  BOUNDS
    0.d0 0.d0 0.d0
    {fmt_pf(lx)} {fmt_pf(ly)} {fmt_pf(zmax)}
  /
END

CHEMISTRY
  PRIMARY_SPECIES
    Tracer
  /
  OUTPUT
    TOTAL
    ALL
  /
END

FLUID_PROPERTY
  DIFFUSION_COEFFICIENT {fmt_pf(base_diff)}
END

DATASET perm_dataset
  HDF5_DATASET_NAME Permeability
  FILENAME {geom_name}
END

DATASET poro_dataset
  HDF5_DATASET_NAME Porosity
  FILENAME {geom_name}
END

MATERIAL_PROPERTY pore
  ID 1
  CHARACTERISTIC_CURVES default
  POROSITY DATASET poro_dataset
  TORTUOSITY {fmt_pf(pore_tort)}
  PERMEABILITY
    DATASET perm_dataset
  /
END

MATERIAL_PROPERTY solid
  ID 2
  CHARACTERISTIC_CURVES default
  POROSITY 0.01d0
  TORTUOSITY {fmt_pf(solid_tort)}
  PERMEABILITY
    DATASET perm_dataset
  /
END

CHARACTERISTIC_CURVES default
  DEFAULT
END

REGION all
  COORDINATES
    0.d0 0.d0 0.d0
    {fmt_pf(lx)} {fmt_pf(ly)} {fmt_pf(zmax)}
  /
END

REGION inlet
  FACE WEST
  COORDINATES
    0.d0 0.d0 0.d0
    0.d0 {fmt_pf(ly)} {fmt_pf(zmax)}
  /
END

REGION outlet
  FACE EAST
  COORDINATES
    {fmt_pf(lx)} 0.d0 0.d0
    {fmt_pf(lx)} {fmt_pf(ly)} {fmt_pf(zmax)}
  /
END

REGION bottom
  FACE SOUTH
  COORDINATES
    0.d0 0.d0 0.d0
    {fmt_pf(lx)} 0.d0 {fmt_pf(zmax)}
  /
END

REGION top
  FACE NORTH
  COORDINATES
    0.d0 {fmt_pf(ly)} 0.d0
    {fmt_pf(lx)} {fmt_pf(ly)} {fmt_pf(zmax)}
  /
END

REGION obs_outlet
  FACE EAST
  COORDINATES
    {fmt_pf(lx)} 0.d0 0.d0
    {fmt_pf(lx)} {fmt_pf(ly)} {fmt_pf(zmax)}
  /
END

OBSERVATION
  BOUNDARY_CONDITION outlet
END

FLOW_CONDITION inflow
  TYPE
    LIQUID_PRESSURE DIRICHLET
  /
  LIQUID_PRESSURE {fmt_pf(p_in)}
END

FLOW_CONDITION outlet
  TYPE
    LIQUID_PRESSURE DIRICHLET
  /
  LIQUID_PRESSURE {fmt_pf(p_out)}
END

TRANSPORT_CONDITION background
  TYPE DIRICHLET_ZERO_GRADIENT
  CONSTRAINT background
    CONCENTRATIONS
      Tracer {fmt_pf(background_concentration)} F
    /
  /
END

CONSTRAINT tracer_on
  CONCENTRATIONS
    Tracer {fmt_pf(tracer_on_concentration)} F
  /
/

CONSTRAINT tracer_off
  CONCENTRATIONS
    Tracer {fmt_pf(background_concentration)} F
  /
/

TRANSPORT_CONDITION tracer_pulse
  TYPE DIRICHLET_ZERO_GRADIENT
  CONSTRAINT_LIST
    0.d0  tracer_on
    {fmt_pf(pulse_duration)}  tracer_off
  /
END

INITIAL_CONDITION initial
  FLOW_CONDITION      inflow
  TRANSPORT_CONDITION background
  REGION all
END

BOUNDARY_CONDITION inlet
  FLOW_CONDITION      inflow
  TRANSPORT_CONDITION tracer_pulse
  REGION inlet
END

BOUNDARY_CONDITION outlet
  FLOW_CONDITION      outlet
  TRANSPORT_CONDITION background
  REGION outlet
END

TIME
  FINAL_TIME {fmt_pf(final_time)} s
  INITIAL_TIMESTEP_SIZE {fmt_pf(initial_dt)} s
  MAXIMUM_TIMESTEP_SIZE {fmt_pf(max_dt)} s
END

OUTPUT
  PERIODIC TIME {fmt_pf(output_period)} s
  PERIODIC_OBSERVATION TIMESTEP 1
  FORMAT HDF5
  VELOCITY_AT_CENTER
END

STRATA
  FILE {geom_name}
END

END_SUBSURFACE
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write darcy_tracer.in from build_pflotran_hdf5_from_npy.py outputs."
    )
    parser.add_argument(
        "--geometry",
        default="pore_geometry.h5",
        help="Geometry HDF5 written by build_pflotran_hdf5_from_npy.py",
    )
    parser.add_argument(
        "--out",
        default="darcy_tracer.in",
        help="PFLOTRAN input file to write",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=20e-6,
        help="Cell size in x and y [m]",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=1e-3,
        help="Domain thickness in z [m]",
    )
    parser.add_argument(
        "--pressure-in",
        type=float,
        default=101326.0,
        help="Inlet pressure [Pa]",
    )
    parser.add_argument(
        "--pressure-out",
        type=float,
        default=101325.0,
        help="Outlet pressure [Pa]",
    )
    parser.add_argument(
        "--pulse-duration",
        type=float,
        default=2.0,
        help="Tracer pulse duration [s]",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=100.0,
        help="Simulation final time [s]",
    )
    parser.add_argument(
        "--output-period",
        type=float,
        default=5.0,
        help="Output interval [s]",
    )
    parser.add_argument(
        "--initial-dt",
        type=float,
        default=1e-1,
        help="Initial timestep [s]",
    )
    parser.add_argument(
        "--max-dt",
        type=float,
        default=5.0,
        help="Maximum timestep [s]",
    )
    parser.add_argument(
        "--tracer-on-concentration",
        type=float,
        default=1.0,
        help="Tracer concentration during pulse [M]",
    )
    parser.add_argument(
        "--pore-diffusivity",
        type=float,
        default=1.0e-9,
        help="Pore-region diffusivity [m^2/s] (used to set base DIFFUSION_COEFFICIENT and pore tortuosity)",
    )
    parser.add_argument(
        "--solid-diffusivity",
        type=float,
        default=1.0e-11,
        help="Solid-region diffusivity [m^2/s] (used to set solid tortuosity)",
    )
    args = parser.parse_args()

    geometry_h5 = Path(args.geometry)
    out_path = Path(args.out)
    nx, ny = load_grid_shape(geometry_h5)
    text = build_input_text(
        geometry_h5=geometry_h5,
        nx=nx,
        ny=ny,
        dx=args.dx,
        dz=args.dz,
        p_in=args.pressure_in,
        p_out=args.pressure_out,
        pulse_duration=args.pulse_duration,
        final_time=args.final_time,
        output_period=args.output_period,
        initial_dt=args.initial_dt,
        max_dt=args.max_dt,
        pore_diffusivity=args.pore_diffusivity,
        solid_diffusivity=args.solid_diffusivity,
        tracer_on_concentration=args.tracer_on_concentration,
    )
    out_path.write_text(text)
    print(f"Wrote {out_path} from {geometry_h5} with grid {nx} x {ny}")


if __name__ == "__main__":
    main()
