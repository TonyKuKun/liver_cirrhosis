from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from statsmodels.sandbox.stats.multicomp import TukeyHSDResults
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import VesselNiiDataset, VesselSTLDataset, collate_fn
    from .model import MODEL_NAMES, DiceBCELoss, create_refinement_model, dice_score
except ImportError:
    try:
        from VKAN_segementation.refinement.dataset import VesselNiiDataset, VesselSTLDataset, collate_fn
        from VKAN_segementation.refinement.model import MODEL_NAMES, DiceBCELoss, create_refinement_model, dice_score
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from refinement.dataset import VesselNiiDataset, VesselSTLDataset, collate_fn
        from refinement.model import MODEL_NAMES, DiceBCELoss, create_refinement_model, dice_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["label"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        b = x.shape[0]
        total_loss += float(loss.item()) * b
        total_dice += dice_score(logits.detach(), y) * b
        n += b
    return {"loss": total_loss / max(n, 1), "dice": total_dice / max(n, 1)}


def _checkpoint_payload(model, optimizer, args, epoch: int, best_dice: float, history: list[dict]) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "model_name": args.model,
        "grid_size": args.grid_size,
        "base_channels": args.base_channels,
        "best_dice": best_dice,
        "history": history,
    }


def _dataset_case_names(ds) -> list[str]:
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        return [ds.dataset.cases[int(i)].name for i in ds.indices]
    if hasattr(ds, "cases"):
        return [case.name for case in ds.cases]
    return []


def _preview_names(names: list[str], limit: int = 8) -> str:
    if not names:
        return "-"
    preview = ", ".join(names[:limit])
    if len(names) > limit:
        preview += f", ... (+{len(names) - limit})"
    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VKAN refinement model from cropped NIfTI masks.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--out_dir", default="VKAN_segementation/runs/nnVnet3")
    parser.add_argument("--dataset", choices=("nii", "stl"), default="nii")
    parser.add_argument("--model", choices=MODEL_NAMES, default="nnVnet", help="Refinement model architecture.")
    parser.add_argument("--pretrain_name", default="pretrain.nii.gz")
    parser.add_argument("--pretrain_stl_name", default="pretrain.stl")
    parser.add_argument("--label_name", default="mask.nii.gz", help="Label NIfTI name, or auto for mask_label/mask_smooth.")
    parser.add_argument("--label_threshold", type=float, default=0.5)
    parser.add_argument("--roi_margin", type=int, default=24)
    parser.add_argument("--crop_source", choices=("union", "pretrain", "label"), default="pretrain")
    parser.add_argument("--include_invalid", action="store_true", help="Compatibility option; refinement datasets only skip $-marked folders.")
    parser.add_argument("--grid_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_review", action="store_true", help="Include cases marked pretrain_quality=review.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        # default="auto",
        help="Resume training. Use without a value to load out_dir/last.pt, or pass a checkpoint path.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"[train] loading dataset={args.dataset} data_root={args.data_root} "
        f"grid_size={args.grid_size} device={device}",
        flush=True,
    )
    if args.dataset == "nii":
        ds = VesselNiiDataset(
            args.data_root,
            grid_size=args.grid_size,
            pretrain_name=args.pretrain_name,
            pretrain_stl_name=args.pretrain_stl_name,
            label_name=args.label_name,
            label_threshold=args.label_threshold,
            roi_margin=args.roi_margin,
            crop_source=args.crop_source,
            include_invalid=args.include_invalid,
        )
    else:
        ds = VesselSTLDataset(args.data_root, grid_size=args.grid_size, require_pretrain=True, include_review=args.include_review)
    print(f"[train] usable patients={len(ds)}", flush=True)
    print(f"[train] patients preview: {_preview_names([case.name for case in ds.cases])}", flush=True)

    val_len = max(1, int(round(len(ds) * args.val_ratio))) if len(ds) > 1 else 0
    train_len = len(ds) - val_len
    if val_len > 0:
        train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = ds, None
    train_names = _dataset_case_names(train_ds)
    val_names = [] if val_ds is None else _dataset_case_names(val_ds)
    print(
        f"[train] split train={len(train_ds)} val={0 if val_ds is None else len(val_ds)} "
        f"val_ratio={args.val_ratio} seed={args.seed}",
        flush=True,
    )
    print(f"[train] train preview: {_preview_names(train_names)}", flush=True)
    if val_ds is not None:
        print(f"[train] val preview: {_preview_names(val_names)}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(
        f"[train] dataloaders ready train_batches={len(train_loader)} "
        f"val_batches={0 if val_loader is None else len(val_loader)} batch_size={args.batch_size}",
        flush=True,
    )

    best_dice = -1.0
    history = []
    start_epoch = 1
    resume_path = None
    checkpoint = None
    if args.resume is None:
        existing = [name for name in ("last.pt", "best.pt", "history.json") if (out_dir / name).exists()]
        if existing:
            print(
                f"[train] starting fresh; existing outputs in {out_dir} will be overwritten as training saves: "
                f"{', '.join(existing)}",
                flush=True,
            )
        else:
            print(f"[train] starting fresh; output_dir={out_dir}", flush=True)
    if args.resume is not None:
        resume_path = out_dir / "last.pt" if args.resume == "auto" else Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        ckpt_args = checkpoint.get("args", {})
        args.model = checkpoint.get("model_name", ckpt_args.get("model", args.model))

    model = create_refinement_model(args.model, base_channels=args.base_channels).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_dice = float(checkpoint.get("best_dice", best_dice))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[train] resumed from {resume_path}; start_epoch={start_epoch}", flush=True)
    (out_dir / "cases.json").write_text(
        json.dumps(
            {
                "cases": [case.name for case in ds.cases],
                "dataset": args.dataset,
                "model": args.model,
                "label_name": args.label_name,
                "pretrain_name": args.pretrain_name,
                "pretrain_stl_name": args.pretrain_stl_name,
                "grid_size": args.grid_size,
                "roi_margin": args.roi_margin,
                "crop_source": args.crop_source,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"[train] epoch={epoch:03d} start", flush=True)
            train_log = run_epoch(model, train_loader, criterion, device, optimizer)
            val_log = run_epoch(model, val_loader, criterion, device) if val_loader is not None else train_log
            history.append({"epoch": epoch, "train": train_log, "val": val_log})
            checkpoint = _checkpoint_payload(model, optimizer, args, epoch, best_dice, history)
            if val_log["dice"] > best_dice:
                best_dice = val_log["dice"]
                checkpoint = _checkpoint_payload(model, optimizer, args, epoch, best_dice, history)
                torch.save(checkpoint, out_dir / "best.pt")
            torch.save(checkpoint, out_dir / "last.pt")
            if epoch == 1 or epoch % 5 == 0:
                print(
                    f"[train] epoch={epoch:03d} loss={train_log['loss']:.4f} dice={train_log['dice']:.4f} "
                    f"val_loss={val_log['loss']:.4f} val_dice={val_log['dice']:.4f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        if history:
            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(
                f"[train] interrupted; saved completed-epoch history to {out_dir / 'history.json'} "
                f"and checkpoint to {out_dir / 'last.pt'}",
                flush=True,
            )
        else:
            print("[train] interrupted before the first epoch completed; no new checkpoint was saved", flush=True)
        raise SystemExit(130)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train] best dice={best_dice:.4f}; checkpoint={out_dir / 'best.pt'}; last={out_dir / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()

