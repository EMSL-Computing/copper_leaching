"""Prepare solver geometry and chrysocolla PFLOTRAN HDF5 realizations.

Mirrors the geometry steps in run_accessibility_mineral_pipeline.sh:
  1) optional bean-bucket design generation
  2) normalize to solver-facing network_correct.npy
  3) build fixed-inventory chrysocolla_real*.h5 files

Example:
    python prepare_accessibility_geometry.py --network-name bean_bucket_network
    python prepare_accessibility_geometry.py --network-name multi_realization_slurm --geometry-source bean_bucket --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import h5py
import numpy as np

from build_pflotran_hdf5_from_npy import load_geometry_npy
from generate_chrysocolla_accessibility_realizations import build_realization
from normalize_design_to_solver import normalize_design_to_solver
from write_darcy_input_for_transport import read_diffusivities_from_h5

from _repo import REPO_ROOT, SRC_DIR

HERE = REPO_ROOT
MANIFEST_NAME = "geometry_manifest.json"


@dataclass(frozen=True)
class AccessibilityGeometryConfig:
    network_name: str
    geometry_source: str = "existing"
    bean_bucket_height: int = 400
    bean_bucket_width: int = 100
    bean_bucket_seed: int = 422
    n_realizations: int = 1
    seed: int = 42
    chrysocolla_solid_fraction: float = 0.50
    pore_diffusivity: float = 1.0e-9
    solid_diffusivity: float = 1.0e-11
    force: bool = False


@dataclass(frozen=True)
class AccessibilityGeometryPaths:
    network_dir: Path
    geom_root: Path
    raw_npy: Path
    geom_npy: Path
    realizations_dir: Path
    manifest_path: Path

    def realization_h5(self, index: int = 0) -> Path:
        return self.realizations_dir / f"chrysocolla_real{index:03d}.h5"


def log(msg: str) -> None:
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_cmd(args: list[str], *, cwd: Path | None = None) -> None:
    log(f"RUN: {' '.join(args)}")
    subprocess.run(args, cwd=cwd or HERE, check=True)


def resolve_paths(config: AccessibilityGeometryConfig) -> AccessibilityGeometryPaths:
    # Historical chip tree only (archived). Prefer accessibility_ensemble for paper data.
    network_dir = REPO_ROOT / "archive" / "Chip_Designs" / config.network_name
    geom_root = network_dir / "geometry"
    realizations_dir = network_dir / "chrysocolla_realizations"
    return AccessibilityGeometryPaths(
        network_dir=network_dir,
        geom_root=geom_root,
        raw_npy=geom_root / "network.npy",
        geom_npy=geom_root / "network_correct.npy",
        realizations_dir=realizations_dir,
        manifest_path=realizations_dir / MANIFEST_NAME,
    )


def config_payload(config: AccessibilityGeometryConfig) -> dict:
    payload = asdict(config)
    payload.pop("force", None)
    return payload


def load_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def manifest_matches(path: Path, config: AccessibilityGeometryConfig) -> bool:
    saved = load_manifest(path)
    return saved is not None and saved.get("config") == config_payload(config)


def write_manifest(paths: AccessibilityGeometryPaths, config: AccessibilityGeometryConfig) -> None:
    paths.realizations_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_payload(config),
        "paths": {
            "raw_npy": str(paths.raw_npy),
            "geom_npy": str(paths.geom_npy),
            "realizations_dir": str(paths.realizations_dir),
        },
    }
    paths.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def validate_realization_h5(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as h5:
        if "Chrysocolla2_VF" not in h5:
            raise KeyError(f"{h5_path} missing Chrysocolla2_VF")
    pore_d, solid_d = read_diffusivities_from_h5(h5_path)
    if pore_d <= 0.0 or solid_d <= 0.0:
        raise ValueError(f"{h5_path} has non-positive diffusivities: {pore_d}, {solid_d}")


def maybe_generate_bean_bucket(paths: AccessibilityGeometryPaths, config: AccessibilityGeometryConfig) -> None:
    if config.geometry_source != "bean_bucket":
        return
    paths.geom_root.mkdir(parents=True, exist_ok=True)
    if paths.raw_npy.is_file() and not config.force:
        log(f"Reusing bean-bucket design: {paths.raw_npy}")
        return
    run_cmd(
        [
            sys.executable,
            str(SRC_DIR / "bean_bucket_generator.py"),
            "--height",
            str(config.bean_bucket_height),
            "--width",
            str(config.bean_bucket_width),
            "--seed",
            str(config.bean_bucket_seed),
            "--out",
            str(paths.raw_npy),
        ]
    )


def ensure_normalized_geometry(paths: AccessibilityGeometryPaths, config: AccessibilityGeometryConfig) -> Path:
    paths.geom_root.mkdir(parents=True, exist_ok=True)
    if paths.geom_npy.is_file() and not config.force:
        log(f"Reusing solver geometry: {paths.geom_npy}")
        return paths.geom_npy

    if config.geometry_source == "bean_bucket":
        if not paths.raw_npy.is_file():
            raise FileNotFoundError(f"Missing raw design for normalization: {paths.raw_npy}")
        source = paths.raw_npy
    elif paths.raw_npy.is_file():
        source = paths.raw_npy
    elif paths.geom_npy.is_file():
        log(f"Only solver geometry present; keeping {paths.geom_npy}")
        return paths.geom_npy
    else:
        raise FileNotFoundError(
            f"No geometry under {paths.geom_root}. Expected network.npy or network_correct.npy"
        )

    geom_npy = normalize_design_to_solver(source, paths.geom_root)
    log(f"Normalized solver geometry: {geom_npy}")
    return geom_npy


def build_one_mineral_realization(
    geom_npy: Path,
    out_dir: Path,
    *,
    mineral_seed: int,
    realization: int = 0,
    chrysocolla_solid_fraction: float = 0.50,
    pore_diffusivity: float = 1.0e-9,
    solid_diffusivity: float = 1.0e-11,
    inlet: str = "left",
    smooth_passes: int = 4,
    accessibility_bias: float = 0.0,
) -> Path:
    """Build one chrysocolla HDF5 on a solver geometry without touching network dirs."""
    img = load_geometry_npy(geom_npy)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_realization(
        img=img,
        out_dir=out_dir,
        realization=realization,
        seed=int(mineral_seed),
        chrysocolla_solid_fraction=chrysocolla_solid_fraction,
        calcite_solid_fraction=0.0,
        solid_vf=0.99,
        pore_diffusivity=pore_diffusivity,
        solid_diffusivity=solid_diffusivity,
        pore_porosity=0.99,
        solid_porosity=0.01,
        inlet=inlet,
        smooth_passes=smooth_passes,
        accessibility_bias=accessibility_bias,
    )
    h5_path = out_dir / f"chrysocolla_real{realization:03d}.h5"
    validate_realization_h5(h5_path)
    return h5_path


def ensure_realizations(paths: AccessibilityGeometryPaths, config: AccessibilityGeometryConfig, geom_npy: Path) -> None:
    paths.realizations_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        i
        for i in range(config.n_realizations)
        if config.force or not paths.realization_h5(i).is_file()
    ]
    if not missing and manifest_matches(paths.manifest_path, config):
        log(f"Reusing {config.n_realizations} chrysocolla realization(s) under {paths.realizations_dir}")
        for i in range(config.n_realizations):
            validate_realization_h5(paths.realization_h5(i))
        return

    if missing:
        log(f"Building chrysocolla realizations: {missing}")
    for i in missing:
        build_one_mineral_realization(
            geom_npy,
            paths.realizations_dir,
            mineral_seed=int(config.seed + i),
            realization=i,
            chrysocolla_solid_fraction=config.chrysocolla_solid_fraction,
            pore_diffusivity=config.pore_diffusivity,
            solid_diffusivity=config.solid_diffusivity,
        )

    write_manifest(paths, config)


def prepare_accessibility_geometry(config: AccessibilityGeometryConfig) -> AccessibilityGeometryPaths:
    """Ensure network_correct.npy and chrysocolla_real*.h5 exist for one network."""
    paths = resolve_paths(config)
    if (
        not config.force
        and paths.geom_npy.is_file()
        and all(paths.realization_h5(i).is_file() for i in range(config.n_realizations))
        and manifest_matches(paths.manifest_path, config)
    ):
        log(f"Geometry ready for {config.network_name} (manifest match)")
        for i in range(config.n_realizations):
            validate_realization_h5(paths.realization_h5(i))
        return paths

    if (
        paths.manifest_path.is_file()
        and not manifest_matches(paths.manifest_path, config)
        and not config.force
        and any(paths.realization_h5(i).is_file() for i in range(config.n_realizations))
    ):
        raise RuntimeError(
            f"Existing geometry manifest does not match requested config: {paths.manifest_path}. "
            "Use --force to rebuild."
        )

    maybe_generate_bean_bucket(paths, config)
    geom_npy = ensure_normalized_geometry(paths, config)
    ensure_realizations(paths, config, geom_npy)
    log(f"Geometry ready: {paths.realization_h5(0)}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare accessibility geometry HDF5 realizations.")
    parser.add_argument("--network-name", required=True)
    parser.add_argument("--geometry-source", choices=["existing", "bean_bucket"], default="existing")
    parser.add_argument("--bean-bucket-height", type=int, default=400)
    parser.add_argument("--bean-bucket-width", type=int, default=100)
    parser.add_argument("--bean-bucket-seed", type=int, default=422)
    parser.add_argument("--n-realizations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chrysocolla-solid-fraction", type=float, default=0.50)
    parser.add_argument("--pore-diffusivity", type=float, default=1.0e-9)
    parser.add_argument("--solid-diffusivity", type=float, default=1.0e-11)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = AccessibilityGeometryConfig(
        network_name=args.network_name,
        geometry_source=args.geometry_source,
        bean_bucket_height=args.bean_bucket_height,
        bean_bucket_width=args.bean_bucket_width,
        bean_bucket_seed=args.bean_bucket_seed,
        n_realizations=args.n_realizations,
        seed=args.seed,
        chrysocolla_solid_fraction=args.chrysocolla_solid_fraction,
        pore_diffusivity=args.pore_diffusivity,
        solid_diffusivity=args.solid_diffusivity,
        force=args.force,
    )
    paths = prepare_accessibility_geometry(config)
    print(paths.realization_h5(0))


if __name__ == "__main__":
    main()