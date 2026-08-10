"""Run CT-aware refinement2 inference and evaluate in original NIfTI space."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .dataset import CASE_EXCLUSION_MARKERS, inference_case, prepare_case
    from .model import CTPretrainNNVNet
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from refinement2.dataset import CASE_EXCLUSION_MARKERS, inference_case, prepare_case
    from refinement2.model import CTPretrainNNVNet

try:
    from utils.common import nifti_mask_to_stl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.common import nifti_mask_to_stl


def _remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1 or not mask.any():
        return mask
    try:
        from scipy import ndimage as ndi
    except ImportError as exc:
        raise ImportError("scipy is required for component filtering") from exc
    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.flatnonzero((np.arange(count + 1) > 0) & (sizes >= min_voxels))
    return np.isin(labels, keep)


def _metrics(prediction: np.ndarray, label: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(prediction, dtype=bool)
    target = np.asarray(label, dtype=bool)
    intersection = int((pred & target).sum())
    pred_voxels = int(pred.sum())
    label_voxels = int(target.sum())
    return {
        "dice": float((2.0 * intersection + 1.0) / (pred_voxels + label_voxels + 1.0)),
        "precision": float((intersection + 1.0) / (pred_voxels + 1.0)),
        "recall": float((intersection + 1.0) / (label_voxels + 1.0)),
        "prediction_voxels": pred_voxels,
        "label_voxels": label_voxels,
        "intersection_voxels": intersection,
    }


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[CTPretrainNNVNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") not in {"ct_pretrain_nnvnet", "ct_pretrain_residual_nnvnet"} or checkpoint.get("input_channels") != 2:
        raise ValueError(f"{checkpoint_path} is not a refinement2 two-channel checkpoint")
    model = CTPretrainNNVNet(
        base_channels=int(checkpoint["base_channels"]),
        prior_strength=float(checkpoint.get("prior_strength", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_case(
    patient_dir: str | Path,
    model: CTPretrainNNVNet,
    checkpoint: dict[str, Any],
    device: torch.device,
    threshold: float = 0.5,
    min_component_voxels: int = 0,
    output_name: str = "predict_ct_mask.nii.gz",
    stl_name: str = "predict_ct.stl",
    write_stl: bool = True,
) -> tuple[Path, dict[str, float | int] | None]:
    args = checkpoint["args"]
    case = inference_case(
        patient_dir,
        orig_name=args["orig_name"],
        pretrain_name=args["pretrain_name"],
        label_name=args["label_name"],
    )
    prepared = prepare_case(
        case,
        grid_size=int(checkpoint["grid_size"]),
        roi_margin=int(args["roi_margin"]),
        hu_min=float(args["hu_min"]),
        hu_max=float(args["hu_max"]),
    )
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        probability_grid = torch.sigmoid(model(prepared.input[None].to(device)))[0, 0][None, None]
    crop_shape = tuple(int(part.stop - part.start) for part in prepared.crop)
    probability_crop = F.interpolate(
        probability_grid, size=crop_shape, mode="trilinear", align_corners=False
    )[0, 0].cpu().numpy()
    crop_mask = _remove_small_components(probability_crop >= threshold, int(min_component_voxels))
    full_mask = np.zeros(prepared.original_shape, dtype=np.uint8)
    full_mask[prepared.crop] = crop_mask.astype(np.uint8)

    import nibabel as nib

    output_path = case.path / output_name
    nib.save(nib.Nifti1Image(full_mask, prepared.affine), str(output_path))
    if write_stl:
        nifti_mask_to_stl(full_mask, prepared.affine, case.path / stl_name, name="refinement2")

    metrics = None
    if prepared.label_full is not None:
        metrics = _metrics(full_mask, prepared.label_full)
        metrics_path = case.path / "vkan_work" / "refinement2_prediction_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path, metrics


def _contains_excluded_marker(name: str) -> bool:
    return any(marker in name for marker in CASE_EXCLUSION_MARKERS)


def _discover_patient_dirs(data_root: Path, orig_name: str, pretrain_name: str) -> list[Path]:
    if (data_root / orig_name).exists():
        return [data_root]
    return [
        path for path in sorted(data_root.iterdir())
        if path.is_dir()
        and not _contains_excluded_marker(path.name)
        and (path / orig_name).exists()
        and (path / pretrain_name).exists()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer a vessel mask from CT HU and pretrain mask.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--min_component_voxels",
        type=int,
        default=0,
        help="Remove only components smaller than this source-grid volume; 0 disables filtering.",
    )
    parser.add_argument("--output_name", default="predict_ct_mask.nii.gz")
    parser.add_argument("--stl_name", default="predict_ct.stl")
    parser.add_argument("--no_stl", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if args.min_component_voxels < 0:
        raise ValueError("min_component_voxels must be non-negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    checkpoint_args = checkpoint["args"]
    root = Path(args.data_root)
    if args.patient:
        patient_dirs = [root / name for name in args.patient]
    else:
        patient_dirs = _discover_patient_dirs(root, checkpoint_args["orig_name"], checkpoint_args["pretrain_name"])
    if not patient_dirs:
        raise RuntimeError("No compatible patient folders found")

    for patient_dir in patient_dirs:
        output_path, metrics = predict_case(
            patient_dir,
            model,
            checkpoint,
            device,
            threshold=args.threshold,
            min_component_voxels=args.min_component_voxels,
            output_name=args.output_name,
            stl_name=args.stl_name,
            write_stl=not args.no_stl,
        )
        summary = "no_label"
        if metrics is not None:
            summary = f"dice={metrics['dice']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
        print(f"[inference] {patient_dir.name}: {summary}; output={output_path}", flush=True)


if __name__ == "__main__":
    main()
