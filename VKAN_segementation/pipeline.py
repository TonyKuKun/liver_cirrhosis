from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[pipeline]", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end patient/dcm to portal vein STL workflow.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--out_dir", default=r"VKAN_segementation/runs/vkan")
    parser.add_argument("--api_key", default='AIzaSyCUnLZXaUWemZe6aXEIpc_6KjZRB9PR288')
    parser.add_argument("--api_base_url", default='https://generativelanguage.googleapis.com/v1beta/openai')
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--grid_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--force_preprocess",default='True', action="store_true", help="Force preprocessing for all patients, even if pretrain.stl already exists. Default is false, so only patients without pretrain.stl are processed.")
    parser.add_argument(
        "--skip_existing_pretrain",
        action="store_true",
        help="Skip preprocessing for patients that already have pretrain.stl. Default is false, so all patients are regenerated.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    preprocess = [py, str(root / "pretrain" / "preprocess.py"), "--data_root", args.data_root, "--model", args.model]
    if args.api_key:
        preprocess += ["--api_key", args.api_key]
    if args.api_base_url:
        preprocess += ["--api_base_url", args.api_base_url]
    if args.force_preprocess:
        preprocess += ["--force"]
    if args.skip_existing_pretrain:
        preprocess += ["--skip_existing_pretrain"]
    run(preprocess)

    run(
        [
            py,
            str(root / "refinement" / "train.py"),
            "--data_root",
            args.data_root,
            "--out_dir",
            args.out_dir,
            "--grid_size",
            str(args.grid_size),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
        ]
    )
    ckpt = str(Path(args.out_dir) / "best.pt")
    run([py, str(root / "refinement" / "predict.py"), "--data_root", args.data_root, "--checkpoint", ckpt])
    check = [py, str(root / "postprocess" / "check_and_smooth.py"), "--data_root", args.data_root, "--model", args.model]
    if args.api_key:
        check += ["--api_key", args.api_key]
    if args.api_base_url:
        check += ["--api_base_url", args.api_base_url]
    run(check)


if __name__ == "__main__":
    main()
