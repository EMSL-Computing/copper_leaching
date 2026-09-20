"""Write Darcy reactive-transport PFLOTRAN input with Cu-leaching chemistry.

Example:
python write_darcy_input_for_transport.py \
  --base-geometry pore_geometry.h5 \
  --out-geometry pore_geometry_darcy_rt.h5 \
  --out-input darcy_transport.in \
  --pore-perm 1e-12 \
  --pore-poro 0.4 \
  --dx 20e-6 \
  --p-in 201326 \
  --p-out 201325.9
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import numpy as np

from darcy_vs_stokes.darcy_utils import copy_geometry_with_contrast
from darcy_vs_stokes.write_darcy_tracer_input import fmt_pf, load_grid_shape

def read_diffusivities_from_h5(geom_h5: Path) -> tuple[float, float]:
    """Return (pore_diffusivity, solid_diffusivity) stored in a geometry H5.

    Reads the 'Diffusivity' and 'Materials/Material Ids' datasets written by
    build_pflotran_hdf5_from_npy.py.  Pore cells have Material Id == 1.
    Falls back to defaults (1e-9, 1e-11) if the dataset is absent.
    """
    with h5py.File(geom_h5, "r") as f:
        if "Diffusivity" not in f or "Materials/Material Ids" not in f:
            return 1.0e-9, 1.0e-11
        diff = np.asarray(f["Diffusivity"]).ravel()
        mids = np.asarray(f["Materials/Material Ids"]).ravel()
    pore_diff = float(diff[mids == 1].mean()) if (mids == 1).any() else 1.0e-9
    solid_diff = float(diff[mids != 1].mean()) if (mids != 1).any() else 1.0e-11
    return pore_diff, solid_diff


def write_darcy_input_for_transport(
    pflotran_input: Path,
    base_geom_h5: Path,
    out_geom_h5: Path,
    pore_perm: float,
    dx: float,
    p_in: float,
    p_out: float,
    final_time_s: float = 12.0 * 3600.0,
    output_period_s: float = 600.0,
    pulse_duration_s: float | None = None,
    initial_dt_s: float = 1.0e-2,
    max_dt_s: float = 600.0,
    solid_perm: float = 1.0e-20,
    pore_diffusivity: float = 1.0e-9,
    solid_diffusivity: float = 1.0e-11,
    dz: float | None = None,
    geometry_npy: Path | None = None,
    restart_file: Path | None = None,
    restart_time_s: float | None = None,
    checkpoint_time_s: float | None = None,
) -> None:
    """Write Darcy reactive-transport PFLOTRAN input with Cu-leaching chemistry."""
    if dz is None:
        dz = dx
    if pore_diffusivity <= 0.0 or solid_diffusivity <= 0.0:
        raise ValueError("pore_diffusivity and solid_diffusivity must be positive.")
    base_diffusivity = max(pore_diffusivity, solid_diffusivity)
    pore_tortuosity = pore_diffusivity / base_diffusivity
    solid_tortuosity = solid_diffusivity / base_diffusivity

    copy_geometry_with_contrast(
        base_geom_h5,
        out_geom_h5,
        pore_perm=pore_perm,
        solid_perm=solid_perm,
    )

    nx, ny = load_grid_shape(base_geom_h5, geometry_npy=geometry_npy)
    lx = nx * dx
    ly = ny * dx
    zmax = dz
    geom_name = out_geom_h5.name
    repo_root = Path(__file__).resolve().parent
    database_file = repo_root / "database" / "hanford.dat"
    database_ref = Path(os.path.relpath(database_file, pflotran_input.parent)).as_posix()

    initial_transport_condition = """TRANSPORT_CONDITION initial
  TYPE DIRICHLET_ZERO_GRADIENT
  CONSTRAINT_LIST
    0.d0 initial
  /
END"""

    if pulse_duration_s is None:
        inlet_transport_condition = """TRANSPORT_CONDITION inlet
  TYPE DIRICHLET_ZERO_GRADIENT
  CONSTRAINT_LIST
    0.d0 inlet
  /
END"""
    else:
        inlet_transport_condition = f"""TRANSPORT_CONDITION inlet
  TYPE DIRICHLET_ZERO_GRADIENT
  CONSTRAINT_LIST
    0.d0 inlet
    {fmt_pf(pulse_duration_s)} initial
  /
END"""

    restart_block = ""
    if restart_file is not None:
        if restart_time_s is None:
            raise ValueError("restart_time_s is required when restart_file is set.")
        restart_block = f"  RESTART {restart_file.as_posix()} {fmt_pf(restart_time_s)}\n"

    checkpoint_block = ""
    if checkpoint_time_s is not None:
        checkpoint_block = f"""  CHECKPOINT
    FORMAT BINARY
    TIMES sec {fmt_pf(checkpoint_time_s)}
  /
"""

    text = f"""!=============================================================================
! Darcy reactive-transport with Cu-leaching chemistry (GIRT mode)
! Pore-scale simulation on unstructured grid derived from grain image.
! Grid: {nx} x {ny} x 1, domain {fmt_pf(lx)} x {fmt_pf(ly)} x {fmt_pf(zmax)} m
! Inlet (west): p = {fmt_pf(p_in)} Pa; Outlet (east): p = {fmt_pf(p_out)} Pa
!=============================================================================

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
{restart_block}{checkpoint_block}END

SUBSURFACE

!------------------------------------------------------------------------------
! Numerical Methods
!------------------------------------------------------------------------------

NUMERICAL_METHODS FLOW
  LINEAR_SOLVER
    SOLVER DIRECT
  /
  TIMESTEPPER
    TS_ACCELERATION 2
  /
END

NUMERICAL_METHODS TRANSPORT
  TIMESTEPPER
    TS_ACCELERATION 2
  /
  NEWTON_SOLVER
    RTOL 1.d-4
    STOL 1.d-7
    MAXIMUM_NUMBER_OF_ITERATIONS 25
  /
    LINEAR_SOLVER
    SOLVER DIRECT
    /
END

!------------------------------------------------------------------------------
! Chemistry & Reactive Transport
! UPDATE_POROSITY: recalculates cell porosity as minerals dissolve/precipitate.
! UPDATE_TORTUOSITY: rescales effective diffusivity via Millington-Quirk
!   (tau_eff = phi^(4/3) — acid access to deeper rind accelerates as it dissolves)
! UPDATE_PERMEABILITY: adjusts permeability with porosity changes
!------------------------------------------------------------------------------

CHEMISTRY
  UPDATE_POROSITY
  UPDATE_TORTUOSITY
  UPDATE_PERMEABILITY

  PRIMARY_SPECIES
    H+
    Cu++
    # Ca++
    SiO2(aq)
    # HCO3-
    SO4--
    Tracer
  /

  SECONDARY_SPECIES
    OH-
    HSO4-
    # CO3--
    # CO2(aq)
    # CaOH+
    # CaHCO3+
    # CaCO3(aq)
    CuSO4(aq)
    CuOH+
  /

  PASSIVE_GAS_SPECIES
    # CO2(g)
  /

  MINERALS
    Chrysocolla2
    # Calcite
    Quartz
  /

  MINERAL_KINETICS
    Chrysocolla2
      SURFACE_AREA_FUNCTION VOLUME_FRACTION_RATIO
      SURFACE_AREA_VOL_FRAC_POWER 0.666666667d0
      PREFACTOR
        RATE_CONSTANT 1.d-10 mol/cm^2-sec
        PREFACTOR_SPECIES H+
          ALPHA 0.39
        /
      /
    /
    # Calcite
    #   RATE_CONSTANT 1.d-9 mol/cm^2-sec
    # /
    Quartz
      RATE_CONSTANT 1.d-14 mol/cm^2-sec
    /
  /

  DATABASE {database_ref}
  LOG_FORMULATION
  ACTIVITY_COEFFICIENTS TIMESTEP
  OUTPUT
    PH
    TOTAL
    FREE_ION
    MINERAL_SURFACE_AREA
    ALL
  /
END

!------------------------------------------------------------------------------
! Grid: {nx} x {ny} x 1, dx = dy = {fmt_pf(dx)} m
!------------------------------------------------------------------------------

GRID
  TYPE STRUCTURED
  NXYZ {nx} {ny} 1
  BOUNDS
    0.d0 0.d0 0.d0
    {fmt_pf(lx)} {fmt_pf(ly)} {fmt_pf(zmax)}
  /
END

!------------------------------------------------------------------------------
! Fluid Properties
!------------------------------------------------------------------------------

FLUID_PROPERTY
  DIFFUSION_COEFFICIENT {fmt_pf(base_diffusivity)}
END

!------------------------------------------------------------------------------
! Data Imports from HDF5 Geometry File
!------------------------------------------------------------------------------

DATASET perm_dataset
  HDF5_DATASET_NAME Permeability
  FILENAME {geom_name}
END

DATASET poro_dataset
  HDF5_DATASET_NAME Porosity
  FILENAME {geom_name}
END

DATASET chrysocolla_vf
  HDF5_DATASET_NAME Chrysocolla2_VF
  FILENAME {geom_name}
END

# DATASET calcite_vf
#   HDF5_DATASET_NAME Calcite_VF
#   FILENAME {geom_name}
# END

DATASET quartz_vf
  HDF5_DATASET_NAME Quartz_VF
  FILENAME {geom_name}
END

!------------------------------------------------------------------------------
! Material Properties
!------------------------------------------------------------------------------

MATERIAL_PROPERTY pore
  ID 1
  POROSITY DATASET poro_dataset
  TORTUOSITY {fmt_pf(pore_tortuosity)}
  PERMEABILITY
    DATASET perm_dataset
  /
  CHARACTERISTIC_CURVES default
END

MATERIAL_PROPERTY solid
  ID 2
  POROSITY 0.01d0
  TORTUOSITY {fmt_pf(solid_tortuosity)}
  PERMEABILITY
    DATASET perm_dataset
  /
  CHARACTERISTIC_CURVES default
END

CHARACTERISTIC_CURVES default
  SATURATION_FUNCTION VAN_GENUCHTEN
    LIQUID_RESIDUAL_SATURATION 0.d0
    ALPHA 1.d-4
    M 0.5d0
  /
  PERMEABILITY_FUNCTION MUALEM
    LIQUID_RESIDUAL_SATURATION 0.d0
    M 0.5d0
  /
END

!------------------------------------------------------------------------------
! Regions
!------------------------------------------------------------------------------

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

REGION obs_outlet
  COORDINATE {fmt_pf(lx)} {fmt_pf(0.5 * ly)} {fmt_pf(0.5 * zmax)}
END

OBSERVATION
  REGION obs_outlet
END

!------------------------------------------------------------------------------
! Flow Conditions
!------------------------------------------------------------------------------

FLOW_CONDITION inlet
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

!------------------------------------------------------------------------------
! Transport Conditions
!------------------------------------------------------------------------------

{initial_transport_condition}

{inlet_transport_condition}

!------------------------------------------------------------------------------
! Constraints (initial & inlet chemistry)
!------------------------------------------------------------------------------

CONSTRAINT initial
  CONCENTRATIONS
    #Ca++       6.5d-4   T
    H+         8.d0     P
    Cu++       6.4d-9   M Chrysocolla2
    SiO2(aq)   1.8d-4   T
    SO4--      5.0d-4   T
    #HCO3-      -3.d0    G CO2(g)
    Tracer     1.d-10   F
  /
  MINERALS
    Chrysocolla2  DATASET chrysocolla_vf  1.d0 cm^2/cm^3
    #Calcite       DATASET calcite_vf      1.d0 cm^2/cm^3
    Quartz        DATASET quartz_vf       1.d0 cm^2/cm^3
  /
END

CONSTRAINT inlet
  CONCENTRATIONS
    #Ca++       1.1d-2   T
    H+         1.d0     P
    Cu++       1.0d-8   T
    SiO2(aq)   1.9d-3   T
    SO4--      6.1d-2   T
    #HCO3-      1.0d-8   T
    Tracer     1.d0     F
  /
END

!------------------------------------------------------------------------------
! Initial & Boundary Conditions
!------------------------------------------------------------------------------

INITIAL_CONDITION
  FLOW_CONDITION inlet
  TRANSPORT_CONDITION initial
  REGION all
END

BOUNDARY_CONDITION inlet
  FLOW_CONDITION inlet
  TRANSPORT_CONDITION inlet
  REGION inlet
END

BOUNDARY_CONDITION outlet
  FLOW_CONDITION outlet
  TRANSPORT_CONDITION initial
  REGION outlet
END

!------------------------------------------------------------------------------
! Output
!------------------------------------------------------------------------------

OUTPUT
  PERIODIC TIME {fmt_pf(output_period_s)} s
  PERIODIC_OBSERVATION TIMESTEP 1
  FORMAT HDF5
  VELOCITY_AT_CENTER
  VELOCITY_AT_FACE
  VARIABLES
    POROSITY
    TORTUOSITY
    PERMEABILITY
  /
END

!------------------------------------------------------------------------------
! Time Stepping
!------------------------------------------------------------------------------

TIME
  FINAL_TIME {fmt_pf(final_time_s)} s
  INITIAL_TIMESTEP_SIZE {fmt_pf(initial_dt_s)} s
  MAXIMUM_TIMESTEP_SIZE {fmt_pf(max_dt_s)} s
END

!------------------------------------------------------------------------------
! Strata (material ID mapping from HDF5)
!------------------------------------------------------------------------------

STRATA
  FILE {geom_name}
END

END_SUBSURFACE
"""
    pflotran_input.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-geometry", type=Path, required=True)
    parser.add_argument("--out-geometry", type=Path, required=True)
    parser.add_argument("--out-input", type=Path, required=True)
    parser.add_argument("--pore-perm", type=float, required=True)
    parser.add_argument("--pore-poro", type=float, required=True)
    parser.add_argument("--solid-perm", type=float, default=1.0e-20)
    parser.add_argument("--solid-poro", type=float, default=1.0e-2)
    parser.add_argument("--pore-diffusivity", type=float, default=1.0e-9)
    parser.add_argument("--solid-diffusivity", type=float, default=1.0e-11)
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dz", type=float, default=None)
    parser.add_argument("--p-in", type=float, required=True)
    parser.add_argument("--p-out", type=float, required=True)
    parser.add_argument("--final-time-s", type=float, default=12.0 * 3600.0)
    parser.add_argument("--output-period-s", type=float, default=600.0)
    parser.add_argument(
        "--pulse-duration-s",
        type=float,
        default=None,
        help="Optional inlet pulse duration [s]; omit for continuous injection.",
    )
    parser.add_argument("--initial-dt-s", type=float, default=1.0e-2)
    parser.add_argument("--max-dt-s", type=float, default=600.0)
    args = parser.parse_args()

    write_darcy_input_for_transport(
        pflotran_input=args.out_input,
        base_geom_h5=args.base_geometry,
        out_geom_h5=args.out_geometry,
        pore_perm=args.pore_perm,
        dx=args.dx,
        p_in=args.p_in,
        p_out=args.p_out,
        final_time_s=args.final_time_s,
        output_period_s=args.output_period_s,
        pulse_duration_s=args.pulse_duration_s,
        initial_dt_s=args.initial_dt_s,
        max_dt_s=args.max_dt_s,
        solid_perm=args.solid_perm,
        pore_diffusivity=args.pore_diffusivity,
        solid_diffusivity=args.solid_diffusivity,
        dz=args.dz,
    )


if __name__ == "__main__":
    main()
