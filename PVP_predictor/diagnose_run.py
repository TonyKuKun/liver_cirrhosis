import argparse
import os

import torch

from diagnostics import collect_oof_predictions, write_group_summary, write_prediction_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str, default=r"F:\PCG data\dataset\test4all_sample")
    ap.add_argument("--n_points", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = collect_oof_predictions(
        args.checkpoint_dir,
        args.data_root,
        n_points=args.n_points,
        batch_size=args.batch_size,
        device=device,
    )
    pred_path = os.path.join(args.checkpoint_dir, "oof_predictions.csv")
    summary_path = os.path.join(args.checkpoint_dir, "oof_group_summary.json")
    write_prediction_csv(rows, pred_path)
    write_group_summary(rows, summary_path)
    print(f"[Diagnose] Wrote {len(rows)} rows -> {pred_path}")
    print(f"[Diagnose] Group summary -> {summary_path}")


if __name__ == "__main__":
    main()
