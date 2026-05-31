from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_OUT_DIR = r"VKAN_segementation/runs/nnVnet3"
DEFAULT_REFINEMENT_MODEL = "nnVnet"


def run(cmd: list[str]) -> None:
    print("[pipeline]", " ".join(cmd))
    subprocess.check_call(cmd)


def _add_patient_arg(cmd: list[str], patient: str | None) -> list[str]:
    if patient:
        cmd += ["--patient", patient]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end TotalSegmentator -> pretrain -> nnVnet -> STL smoothing workflow.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--patient", default=None, help="Process one patient folder name or path.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--totalseg_device", default="gpu", choices=("gpu", "cpu"))
    parser.add_argument("--totalseg_fast", action="store_true", help="Run TotalSegmentator in fast mode.")
    parser.add_argument("--totalseg_structures", nargs="+", default=None, help="Optional TotalSegmentator structures to extract.")
    parser.add_argument("--force_totalseg", action="store_true", help="Overwrite existing TotalSegmentator organ outputs.")
    parser.add_argument("--grid_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--label_name", default="mask.nii.gz")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smooth_iterations", type=int, default=8)
    parser.add_argument("--force_preprocess", action="store_true", help="Regenerate pretrain.stl/pretrain.nii.gz even when cached outputs are current.")
    parser.add_argument(
        "--skip_existing_pretrain",
        action="store_true",
        help="Skip preprocessing for patients that already have pretrain.stl and pretrain.nii.gz.",
    )
    parser.add_argument("--only_dollar_patients", action="store_true", help='Only preprocess patient folders whose names contain "$".')
    parser.add_argument("--include_review", action="store_true", help="Include cases marked pretrain_quality=review during training.")
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume nnVnet training from out_dir/last.pt, or pass a checkpoint path.")
    parser.add_argument("--force_postprocess", action="store_true", help="Regenerate predict_smooth.stl even if it already exists.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable

    totalseg = [
        py,
        str(root / "pretrain" / "totalseg.py"),
        "--data_root",
        args.data_root,
        "--device",
        args.totalseg_device,
    ]
    _add_patient_arg(totalseg, args.patient)
    if args.totalseg_fast:
        totalseg += ["--fast"]
    if args.force_totalseg:
        totalseg += ["--overwrite"]
    if args.totalseg_structures:
        totalseg += ["--structures", *args.totalseg_structures]
    run(totalseg)

    preprocess = [py, str(root / "pretrain" / "preprocess.py"), "--data_root", args.data_root]
    _add_patient_arg(preprocess, args.patient)
    if args.force_preprocess:
        preprocess += ["--force"]
    if args.skip_existing_pretrain:
        preprocess += ["--skip_existing_pretrain"]
    if args.only_dollar_patients:
        preprocess += ["--only_dollar_patients"]
    run(preprocess)

    train = [
        py,
        str(root / "refinement" / "train.py"),
        "--data_root",
        args.data_root,
        "--out_dir",
        args.out_dir,
        "--dataset",
        "nii",
        "--model",
        DEFAULT_REFINEMENT_MODEL,
        "--grid_size",
        str(args.grid_size),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--base_channels",
        str(args.base_channels),
        "--lr",
        str(args.lr),
        "--val_ratio",
        str(args.val_ratio),
        "--label_name",
        args.label_name,
    ]
    if args.include_review:
        train += ["--include_review"]
    if args.resume:
        train += ["--resume"] if args.resume == "auto" else ["--resume", args.resume]
    run(train)

    ckpt = str(Path(args.out_dir) / "best.pt")
    predict = [
        py,
        str(root / "refinement" / "predict.py"),
        "--data_root",
        args.data_root,
        "--checkpoint",
        ckpt,
        "--threshold",
        str(args.threshold),
    ]
    _add_patient_arg(predict, args.patient)
    run(predict)

    check = [
        py,
        str(root / "postprocess" / "check_and_smooth.py"),
        "--data_root",
        args.data_root,
        "--iterations",
        str(args.smooth_iterations),
    ]
    _add_patient_arg(check, args.patient)
    if args.force_postprocess:
        check += ["--force"]
    run(check)


if __name__ == "__main__":
    main()
