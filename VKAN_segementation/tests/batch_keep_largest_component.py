r"""Batch keep the largest connected component in every pretrain NIfTI.

Preview all changes first:
    python tests/batch_keep_largest_component.py --data_root "F:\PCG data\dataset\test4all_sample"

Apply the result in place and regenerate each STL:
    python tests/batch_keep_largest_component.py --data_root "F:\PCG data\dataset\test4all_sample" --in-place
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib

from pretrain.preprocess import _keep_largest_connected_component
from utils.common import nifti_mask_to_stl


def _discover_pretrains(data_root: Path) -> list[Path]:
    if (data_root / "pretrain.nii.gz").exists():
        return [data_root / "pretrain.nii.gz"]
    return sorted(data_root.glob("*/pretrain.nii.gz"))


def _save_mask(mask_xyz: np.ndarray, source: nib.Nifti1Image, output_path: Path) -> None:
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(mask_xyz.astype(np.uint8), source.affine, header)
    qform, qcode = source.get_qform(coded=True)
    sform, scode = source.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qcode))
    if sform is not None:
        output.set_sform(sform, int(scode))
    nib.save(output, str(output_path))


def process_case(pretrain_path: Path, in_place: bool) -> dict[str, int | str]:
    source = nib.load(str(pretrain_path))
    mask_xyz = np.asarray(source.dataobj) > 0
    if mask_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape={mask_xyz.shape}")

    # The preprocessing pipeline uses z-y-x internally; convert back for NIfTI output.
    filtered_zyx, info = _keep_largest_connected_component(np.transpose(mask_xyz, (2, 1, 0)))
    filtered_xyz = np.transpose(filtered_zyx, (2, 1, 0))
    result: dict[str, int | str] = {
        "case": pretrain_path.parent.name,
        "input_voxels": int(mask_xyz.sum()),
        "output_voxels": int(filtered_xyz.sum()),
        "removed_voxels": int(mask_xyz.sum() - filtered_xyz.sum()),
        "components": int(info["components"]),
    }

    if in_place and int(filtered_xyz.sum()) != 0:
        _save_mask(filtered_xyz, source, pretrain_path)
        nifti_mask_to_stl(filtered_xyz, source.affine, pretrain_path.with_name("pretrain.stl"), name="pretrain")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default=r"F:\PCG data\dataset\test4all_sample",
        type=Path,
        help="Dataset root containing patient folders.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite pretrain.nii.gz and regenerate pretrain.stl.",
    )
    args = parser.parse_args()

    paths = _discover_pretrains(args.data_root)
    if not paths:
        raise SystemExit(f"No pretrain.nii.gz found under {args.data_root}")

    mode = "in-place" if args.in_place else "preview"
    print(f"[largest-component] mode={mode} cases={len(paths)}")
    failed = 0
    for path in paths:
        try:
            result = process_case(path, args.in_place)
            print(
                f"[largest-component] {result['case']}: "
                f"components={result['components']} "
                f"voxels={result['input_voxels']}->{result['output_voxels']} "
                f"removed={result['removed_voxels']}"
            )
        except Exception as exc:
            failed += 1
            print(f"[largest-component] {path.parent.name}: FAILED: {exc}")

    if failed:
        raise SystemExit(f"{failed} case(s) failed")


if __name__ == "__main__":
    main()
