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
    from .augmentation import FILL_VARIANTS_PER_CASE, FixedFillAugmentedDataset
    from .dataset import VesselNiiDataset, VesselSTLDataset, collate_fn
    from .model import MODEL_NAMES, DiceBCECLDiceLoss, create_refinement_model, dice_score
except ImportError:
    try:
        from VKAN_segementation.refinement.augmentation import FILL_VARIANTS_PER_CASE, FixedFillAugmentedDataset
        from VKAN_segementation.refinement.dataset import VesselNiiDataset, VesselSTLDataset, collate_fn
        from VKAN_segementation.refinement.model import MODEL_NAMES, DiceBCECLDiceLoss, create_refinement_model, dice_score
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from refinement.augmentation import FILL_VARIANTS_PER_CASE, FixedFillAugmentedDataset
        from refinement.dataset import VesselNiiDataset, VesselSTLDataset, collate_fn
        from refinement.model import MODEL_NAMES, DiceBCECLDiceLoss, create_refinement_model, dice_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    dice_largest_component: bool = False,
) -> dict[str, float]:
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
        total_dice += dice_score(logits.detach(), y, largest_component=dice_largest_component) * b
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


def _epochs_without_improvement(history: list[dict]) -> int:
    """Return the current validation-Dice plateau length for resumed training."""
    best_dice = float("-inf")
    epochs_without_improvement = 0
    for entry in history:
        val_dice = float(entry["val"]["dice"])
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    return epochs_without_improvement


def _cldice_weight_for_epoch(epoch: int, max_weight: float, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs == 0:
        return float(max_weight)
    progress = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return float(max_weight) * progress


def _plot_training_history(history: list[dict], plot_path: Path) -> Path | None:
    if not history:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train"]["loss"] for entry in history]
    val_loss = [entry["val"]["loss"] for entry in history]
    train_dice = [entry["train"]["dice"] for entry in history]
    val_dice = [entry["val"]["dice"] for entry in history]

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (loss_ax, dice_ax) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    loss_ax.plot(epochs, train_loss, label="Train")
    loss_ax.plot(epochs, val_loss, label="Validation")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    dice_ax.plot(epochs, train_dice, label="Train")
    dice_ax.plot(epochs, val_dice, label="Validation")
    dice_ax.set_title("Dice")
    dice_ax.set_xlabel("Epoch")
    dice_ax.set_ylabel("Dice")
    dice_ax.set_ylim(0.0, 1.0)
    dice_ax.grid(alpha=0.3)
    dice_ax.legend()

    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _save_training_history(history: list[dict], out_dir: Path, plot_path: Path) -> Path | None:
    history_path = out_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plotted_path = _plot_training_history(history, plot_path)
    if plotted_path is not None:
        print(f"[train] training curve saved to {plotted_path}", flush=True)
    return plotted_path


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


def _add_input_augmentation_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--input_error_augmentation",
        dest="input_error_augmentation",
        action="store_true",
        help="Expand each training case into the original plus five fixed SMV centerline truncations.",
    )
    group.add_argument(
        "--no_input_error_augmentation",
        dest="input_error_augmentation",
        action="store_false",
        help="Use only the original training cases.",
    )
    parser.set_defaults(input_error_augmentation=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VKAN refinement model from cropped NIfTI masks.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--out_dir", default="VKAN_segementation/runs/nnVnet_loss0.2")
    parser.add_argument("--dataset", choices=("nii", "stl"), default="nii")
    parser.add_argument("--model", choices=MODEL_NAMES, default="nnVnet", help="Refinement model architecture.")
    parser.add_argument("--pretrain_name", default="pretrain.nii.gz")
    parser.add_argument("--pretrain_stl_name", default="pretrain.stl")
    parser.add_argument("--label_name", default="mask.nii.gz", help="Label NIfTI name, or auto for mask_label/mask_smooth.")
    parser.add_argument("--label_threshold", type=float, default=0.5)
    parser.add_argument("--roi_margin", type=int, default=24)
    parser.add_argument("--crop_source", choices=("union", "pretrain", "label"), default="pretrain")
    parser.add_argument("--include_invalid", action="store_true", help="Compatibility option; refinement datasets only skip $-marked folders.")
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument(
        "--cache_dir",
        default=str(Path(__file__).resolve().parent / "cache"),
        help="Directory for preprocessed NIfTI tensors (created by data_preprocess.py).",
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable the NIfTI preprocessing cache.")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=24)
    _add_input_augmentation_arguments(parser)
    parser.add_argument(
        "--cldice_weight",
        type=float,
        default=0.2,
        help="Final soft-clDice weight in the Dice+BCE/clDice blend; 0 disables clDice.",
    )
    parser.add_argument(
        "--cldice_warmup_epochs",
        type=int,
        default=10,
        help="Train with Dice+BCE only for this many initial epochs.",
    )
    parser.add_argument(
        "--cldice_ramp_epochs",
        type=int,
        default=30,
        help="Linearly increase clDice to its final weight over this many epochs.",
    )
    parser.add_argument(
        "--cldice_iterations",
        type=int,
        default=5,
        help="Number of differentiable 3D skeletonization iterations.",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument(
        "--dice_largest_component",
        action="store_true",
        help="Keep only the largest predicted 26-connected component before reporting Dice.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Stop after this many epochs without validation Dice improvement; 0 disables early stopping.",
    )
    parser.add_argument(
        "--plot_path",
        default=None,
        help="Training curve PNG path; defaults to refinement/training_curve_<out_dir>.png.",
    )
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
    if args.patience < 0:
        parser.error("--patience must be greater than or equal to 0")
    if not 0.0 <= args.cldice_weight <= 1.0:
        parser.error("--cldice_weight must be between 0 and 1")
    if args.cldice_warmup_epochs < 0:
        parser.error("--cldice_warmup_epochs must be greater than or equal to 0")
    if args.cldice_ramp_epochs < 0:
        parser.error("--cldice_ramp_epochs must be greater than or equal to 0")
    if args.cldice_iterations < 0:
        parser.error("--cldice_iterations must be greater than or equal to 0")
    if args.input_error_augmentation and args.dataset != "nii":
        parser.error("--input_error_augmentation requires --dataset nii")
    plot_path = (
        Path(args.plot_path)
        if args.plot_path is not None
        else Path(__file__).resolve().parent / f"training_curve_{out_dir.name}.png"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"[train] loading dataset={args.dataset} data_root={args.data_root} "
        f"grid_size={args.grid_size} device={device}",
        flush=True,
    )
    if args.dataset == "nii":
        cache_dir = None if args.no_cache else Path(args.cache_dir)
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
            cache_dir=cache_dir,
        )
        if cache_dir is not None:
            print(f"[train] preparing NIfTI cache in {cache_dir}", flush=True)
            written = ds.build_cache()
            print(f"[train] cache ready: {len(ds)} cases ({written} files written)", flush=True)
    else:
        cache_dir = None
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
    if args.input_error_augmentation:
        base_train_len = len(train_ds)
        feature_count = sum(
            (Path(args.data_root) / name / "features" / "unified_features.json").exists()
            for name in train_names
        )
        train_ds = FixedFillAugmentedDataset(train_ds, data_root=args.data_root)
        print(
            f"[train] fixed SMV fill augmentation: {base_train_len} cases x "
            f"{FILL_VARIANTS_PER_CASE} variants = {len(train_ds)} training samples; "
            f"centerline_features={feature_count}/{base_train_len}",
            flush=True,
        )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(
        f"[train] dataloaders ready train_batches={len(train_loader)} "
        f"val_batches={0 if val_loader is None else len(val_loader)} batch_size={args.batch_size}",
        flush=True,
    )
    print(
        f"[train] input_error_augmentation={args.input_error_augmentation}",
        flush=True,
    )

    best_dice = -1.0
    history = []
    epochs_without_improvement = 0
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
    criterion = DiceBCECLDiceLoss(
        cldice_weight=0.0,
        skeleton_iterations=args.cldice_iterations,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_dice = float(checkpoint.get("best_dice", best_dice))
        history = list(checkpoint.get("history", []))
        epochs_without_improvement = _epochs_without_improvement(history)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(
            f"[train] resumed from {resume_path}; start_epoch={start_epoch}; "
            f"epochs_without_improvement={epochs_without_improvement}",
            flush=True,
        )
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
                "cache_dir": None if cache_dir is None else str(cache_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            cldice_weight = _cldice_weight_for_epoch(
                epoch,
                max_weight=args.cldice_weight,
                warmup_epochs=args.cldice_warmup_epochs,
                ramp_epochs=args.cldice_ramp_epochs,
            )
            criterion.set_cldice_weight(cldice_weight)
            print(f"[train] epoch={epoch:03d} start cldice_weight={cldice_weight:.4f}", flush=True)
            train_log = run_epoch(
                model, train_loader, criterion, device, optimizer,
                dice_largest_component=args.dice_largest_component,
            )
            val_log = (
                run_epoch(model, val_loader, criterion, device, dice_largest_component=args.dice_largest_component)
                if val_loader is not None else train_log
            )
            train_log["cldice_weight"] = cldice_weight
            val_log["cldice_weight"] = cldice_weight
            history.append({"epoch": epoch, "train": train_log, "val": val_log})
            checkpoint = _checkpoint_payload(model, optimizer, args, epoch, best_dice, history)
            if val_log["dice"] > best_dice:
                best_dice = val_log["dice"]
                epochs_without_improvement = 0
                checkpoint = _checkpoint_payload(model, optimizer, args, epoch, best_dice, history)
                torch.save(checkpoint, out_dir / "best.pt")
            else:
                epochs_without_improvement += 1
            torch.save(checkpoint, out_dir / "last.pt")
            print(
                f"[train] epoch={epoch:03d} loss={train_log['loss']:.4f} dice={train_log['dice']:.4f} "
                f"val_loss={val_log['loss']:.4f} val_dice={val_log['dice']:.4f}",
                flush=True,
            )
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"[train] early stopping at epoch={epoch:03d}; validation Dice did not improve "
                    f"for {epochs_without_improvement} epoch(s) (patience={args.patience})",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        if history:
            _save_training_history(history, out_dir, plot_path)
            print(
                f"[train] interrupted; saved completed-epoch history to {out_dir / 'history.json'} "
                f"and checkpoint to {out_dir / 'last.pt'}",
                flush=True,
            )
        else:
            print("[train] interrupted before the first epoch completed; no new checkpoint was saved", flush=True)
        raise SystemExit(130)

    _save_training_history(history, out_dir, plot_path)
    print(f"[train] best dice={best_dice:.4f}; checkpoint={out_dir / 'best.pt'}; last={out_dir / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()

