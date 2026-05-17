from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch.nn.functional as F

try:
    from ..utils.common import INVALID_MARKERS, discover_patients, nifti_mask_to_stl, stl_to_voxels, voxels_to_stl
except (ImportError, ValueError):
    try:
        from VKAN_segementation.utils.common import INVALID_MARKERS, discover_patients, nifti_mask_to_stl, stl_to_voxels, voxels_to_stl
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import INVALID_MARKERS, discover_patients, nifti_mask_to_stl, stl_to_voxels, voxels_to_stl

try:
    from .dataset import _foreground_bbox, _load_nii, _resize_volume
except (ImportError, ValueError):
    try:
        from VKAN_segementation.refinement.dataset import _foreground_bbox, _load_nii, _resize_volume
    except ImportError:
        from refinement.dataset import _foreground_bbox, _load_nii, _resize_volume


def predict_case(case, checkpoint: dict, threshold: float = 0.5, out_path: Path | None = None) -> Path:
    import torch
    try:
        from .model import VesselVKAN
    except ImportError:
        try:
            from VKAN_segementation.refinement.model import VesselVKAN
        except ImportError:
            from refinement.model import VesselVKAN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = int(checkpoint.get("grid_size", checkpoint.get("args", {}).get("grid_size", 96)))
    base_channels = int(checkpoint.get("base_channels", checkpoint.get("args", {}).get("base_channels", 16)))
    model = VesselVKAN(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args = checkpoint.get("args", {})
    if args.get("dataset", "stl") == "nii":
        return predict_case_nii(case.path, checkpoint, threshold=threshold, out_path=out_path)

    grid, bounds = stl_to_voxels(case.pretrain_stl, grid_size=grid_size)
    x = torch.from_numpy(grid[None, None]).float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return _save_prediction_outputs(case, prob, bounds, threshold=threshold, out_path=out_path)


def _save_prediction_outputs(case, prob: np.ndarray, bounds: np.ndarray, threshold: float = 0.5, out_path: Path | None = None) -> Path:
    mask = (np.asarray(prob) >= threshold).astype(np.uint8)
    return voxels_to_stl(mask, bounds, out_path or case.predict_stl, threshold=0.5)


def predict_case_nii(patient_dir: Path, checkpoint: dict, threshold: float = 0.5, out_path: Path | None = None) -> Path:
    import nibabel as nib
    import torch
    try:
        from .model import VesselVKAN
    except ImportError:
        try:
            from VKAN_segementation.refinement.model import VesselVKAN
        except ImportError:
            from refinement.model import VesselVKAN

    args = checkpoint.get("args", {})
    grid_size = int(checkpoint.get("grid_size", args.get("grid_size", 96)))
    base_channels = int(checkpoint.get("base_channels", args.get("base_channels", 16)))
    pretrain_name = args.get("pretrain_name", "pretrain.nii.gz")
    roi_margin = int(args.get("roi_margin", 16))
    patient_dir = Path(patient_dir)
    pretrain_path = patient_dir / pretrain_name
    if not pretrain_path.exists():
        raise FileNotFoundError(pretrain_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselVKAN(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pre, affine = _load_nii(pretrain_path)
    pre_mask = pre > 0.5
    crop = _foreground_bbox(pre_mask, roi_margin)
    pre_crop = pre_mask[crop].astype(np.float32)
    x = _resize_volume(pre_crop, grid_size, "trilinear")[None, None].to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu()[None, None]
    crop_shape = tuple(int(s.stop - s.start) for s in crop)
    prob_crop = F.interpolate(prob, size=crop_shape, mode="trilinear", align_corners=False)[0, 0].numpy()
    full = np.zeros(pre.shape, dtype=np.float32)
    full[crop] = prob_crop
    mask = (full >= threshold).astype(np.uint8)

    mask_path = out_path or (patient_dir / "predict_mask.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))
    nifti_mask_to_stl(mask, affine, patient_dir / "predict.stl")
    return Path(mask_path)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description="Generate prediction from pretrain NIfTI or STL.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--checkpoint", default=r'E:\pycharm_code\liver_cirrhosis\VKAN_segementation\VKAN_segementation\runs\vkan\best.pt')
    parser.add_argument("--patient", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    if ckpt_args.get("dataset", "stl") == "nii":
        root = Path(args.data_root)
        patient_dirs = [root] if (root / ckpt_args.get("pretrain_name", "pretrain.nii.gz")).exists() else sorted(p for p in root.iterdir() if p.is_dir())
        cases = [p for p in patient_dirs if not any(marker in p.name for marker in INVALID_MARKERS) and (p / ckpt_args.get("pretrain_name", "pretrain.nii.gz")).exists()]
        if args.patient:
            cases = [p for p in cases if p.name == args.patient]
        if not cases:
            raise RuntimeError("No matching cases with pretrain NIfTI found.")
        for case_dir in cases:
            out = predict_case_nii(case_dir, checkpoint, threshold=args.threshold)
            print(f"[predict] {case_dir.name}: wrote {out} and {case_dir / 'predict.stl'}")
    else:
        cases = [case for case in discover_patients(args.data_root) if case.pretrain_stl.exists()]
        if args.patient:
            cases = [case for case in cases if case.name == args.patient]
        if not cases:
            raise RuntimeError("No matching cases with pretrain.stl found.")
        for case in cases:
            out = predict_case(case, checkpoint, threshold=args.threshold)
            print(f"[predict] {case.name}: wrote {out}")


if __name__ == "__main__":
    main()
