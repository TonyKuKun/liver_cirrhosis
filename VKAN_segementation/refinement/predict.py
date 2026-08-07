from __future__ import annotations

import argparse
import json
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


def _case_name(case) -> str:
    """Return a stable patient name for either a Path or a dataset case."""
    return str(case.name) if hasattr(case, "name") else Path(case).name


def _load_training_case_names(checkpoint_path: Path) -> list[str]:
    """Load the exact training case order, excluding $-marked patients."""
    manifest_path = checkpoint_path.with_name("cases.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training case manifest is required to reproduce the seeded split: {manifest_path}"
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read training case manifest: {manifest_path}") from exc

    names = payload.get("cases")
    if not isinstance(names, list):
        raise RuntimeError(f"Training case manifest has no 'cases' list: {manifest_path}")
    if any(not isinstance(name, str) for name in names):
        raise RuntimeError(f"Training case manifest contains a non-string case name: {manifest_path}")

    return [name for name in names if "$" not in name]


def _select_test_cases(cases: list, seed: int, val_ratio: float) -> list:
    """Select the held-out split using the exact split rule used by train.py."""
    import torch
    from torch.utils.data import random_split

    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if len(cases) <= 1:
        return []

    test_len = max(1, int(round(len(cases) * val_ratio)))
    train_len = len(cases) - test_len

    _, test_subset = random_split(
        cases,
        [train_len, test_len],
        generator=torch.Generator().manual_seed(int(seed)),
    )
    return [cases[int(index)] for index in test_subset.indices]


def _select_checkpoint_test_cases(
    cases: list,
    checkpoint_path: Path,
    seed: int,
    val_ratio: float,
) -> tuple[list, int]:
    """Split the training manifest first, then resolve selected names to inputs."""
    available_by_name = {
        _case_name(case): case
        for case in cases
        if "$" not in _case_name(case)
    }
    training_names = _load_training_case_names(checkpoint_path)
    selected_names = _select_test_cases(training_names, seed, val_ratio)

    missing = [name for name in selected_names if name not in available_by_name]
    if missing:
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += f", ... (+{len(missing) - 8})"
        raise RuntimeError(
            "Held-out cases from the training split are missing the required prediction input: "
            f"{preview}"
        )

    return [available_by_name[name] for name in selected_names], len(training_names)


def predict_case(case, checkpoint: dict, threshold: float = 0.5, out_path: Path | None = None) -> Path:
    import torch
    try:
        from .model import create_refinement_model
    except ImportError:
        try:
            from VKAN_segementation.refinement.model import create_refinement_model
        except ImportError:
            from refinement.model import create_refinement_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = checkpoint.get("args", {})
    model_name = checkpoint.get("model_name", args.get("model", "vkan"))
    grid_size = int(checkpoint.get("grid_size", checkpoint.get("args", {}).get("grid_size", 96)))
    base_channels = int(checkpoint.get("base_channels", checkpoint.get("args", {}).get("base_channels", 16)))
    model = create_refinement_model(model_name, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
        from .model import create_refinement_model
    except ImportError:
        try:
            from VKAN_segementation.refinement.model import create_refinement_model
        except ImportError:
            from refinement.model import create_refinement_model

    args = checkpoint.get("args", {})
    model_name = checkpoint.get("model_name", args.get("model", "vkan"))
    grid_size = int(checkpoint.get("grid_size", args.get("grid_size", 96)))
    base_channels = int(checkpoint.get("base_channels", args.get("base_channels", 16)))
    pretrain_name = args.get("pretrain_name", "pretrain.nii.gz")
    roi_margin = int(args.get("roi_margin", 16))
    patient_dir = Path(patient_dir)
    pretrain_path = patient_dir / pretrain_name
    if not pretrain_path.exists():
        raise FileNotFoundError(pretrain_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_refinement_model(model_name, base_channels=base_channels).to(device)
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
    parser.add_argument("--checkpoint", default=r'E:\pycharm_code\liver_cirrhosis\VKAN_segementation\VKAN_segementation\runs\nnVnet4\best.pt')
    parser.add_argument("--patient", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--seed",
        type=int,
        default=30,
        help="Predict only the held-out validation/test split generated with this training seed.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args", {})
    if ckpt_args.get("dataset", "stl") == "nii":
        root = Path(args.data_root)
        patient_dirs = [root] if (root / ckpt_args.get("pretrain_name", "pretrain.nii.gz")).exists() else sorted(p for p in root.iterdir() if p.is_dir())
        cases = [p for p in patient_dirs if (p / ckpt_args.get("pretrain_name", "pretrain.nii.gz")).exists()]
        if args.seed is None:
            cases = [p for p in cases if not any(marker in p.name for marker in INVALID_MARKERS)]
        case_kind = "pretrain NIfTI"
    else:
        cases = [case for case in discover_patients(args.data_root) if case.pretrain_stl.exists()]
        case_kind = "pretrain STL"

    if args.seed is not None:
        val_ratio = float(ckpt_args.get("val_ratio", 0.2))
        cases, split_source_count = _select_checkpoint_test_cases(
            cases,
            checkpoint_path,
            args.seed,
            val_ratio,
        )
        print(
            f"[predict] selected held-out split: cases={len(cases)} "
            f"from_training_cases={split_source_count} seed={args.seed} val_ratio={val_ratio}"
        )

    if args.patient:
        cases = [case for case in cases if _case_name(case) == args.patient]
    if not cases:
        raise RuntimeError(f"No matching cases with {case_kind} found.")

    if ckpt_args.get("dataset", "stl") == "nii":
        for case_dir in cases:
            out = predict_case_nii(case_dir, checkpoint, threshold=args.threshold)
            print(f"[predict] {case_dir.name}: wrote {out} and {case_dir / 'predict.stl'}")
    else:
        for case in cases:
            out = predict_case(case, checkpoint, threshold=args.threshold)
            print(f"[predict] {case.name}: wrote {out}")


if __name__ == "__main__":
    main()
