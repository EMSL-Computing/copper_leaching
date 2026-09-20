"""Normalize a chip-design mask into the solver-facing convention.

The bean-bucket source generator (`bean_bucket_generator.py`) writes the raw
chip design with `0 = pore` and `1 = solid` in image-order (`img[y, x]`).

The Stokes and PFLOTRAN downstream solvers expect the opposite material
polarity and a solver-oriented layout, so this adapter:

* inverts the binary field so `1 = pore` and `0 = solid`
* transposes once so a rectangular image becomes solver-aligned on the x axis

Example:
    python normalize_design_to_solver.py Chip_Designs/network0.npy --out-dir Chip_Designs/network0/geometry
"""

import argparse
from pathlib import Path

import _repo  # noqa: F401 — path bootstrap

import numpy as np


def normalize_design_to_solver(design_path: Path, out_dir: Path | None = None) -> Path:
    """Write the solver-facing binary mask as `<name>_correct.npy`.

    The output uses `1 = pore`, `0 = solid`, and a transposed layout so that
    the solver-facing x direction matches the long chip axis.
    """
    img = np.load(design_path, allow_pickle=False)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D design array, got shape {img.shape}")

    vals = np.unique(img)
    if not np.all(np.isin(vals, [0, 1, 0.0, 1.0])):
        raise ValueError(f"Expected binary values 0/1, found {vals[:10]}")

    # Keep the canonical image-order input intact elsewhere; this file is the
    # solver-facing adapter and emits the orientation expected by downstream
    # PFLOTRAN/Stokes consumers.
    corrected = (1.0 - img.astype(np.float64)).T
    if out_dir is None:
        out_path = design_path.with_name(f"{design_path.stem}_correct.npy")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{design_path.stem}_correct.npy"
    np.save(out_path, corrected)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invert binary chip design .npy and save as *_correct.npy."
    )
    parser.add_argument(
        "design_path",
        nargs="?",
        default="Chip_Designs/network0.npy",
        help="Path to chip design .npy file",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory for *_correct.npy",
    )

    args = parser.parse_args()
    out_path = normalize_design_to_solver(Path(args.design_path), args.out_dir)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
