"""Train a CT-HU plus pretrain-mask vessel refinement model."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REFINEMENT_DIR = Path(__file__).resolve().parent
CANONICAL_RUN_DIR = REFINEMENT_DIR / "runs" / "ct_pretrain_nnvnet"
LEGACY_RUN_DIR = REFINEMENT_DIR / "refinement2" / "runs" / "ct_pretrain_nnvnet"
RESIDUAL_RUN_DIR = REFINEMENT_DIR / "runs" / "ct_pretrain_residual_nnvnet"
DEFAULT_CACHE_DIR = (
    LEGACY_RUN_DIR / "tensor_cache"
    if (LEGACY_RUN_DIR / "tensor_cache").exists()
    else RESIDUAL_RUN_DIR / "tensor_cache"
)
# An earlier relative default could create this nested directory when PyCharm used
# refinement2 as its working directory. Preserve it for automatic continuation.
DEFAULT_RUN_DIR = CANONICAL_RUN_DIR if CANONICAL_RUN_DIR.exists() else LEGACY_RUN_DIR

try:
    from .dataset import CTHUVesselDataset, NiiCase, collate_fn
    from .model import CTPretrainNNVNet, create_loss, dice_per_case
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from refinement2.dataset import CTHUVesselDataset, NiiCase, collate_fn
    from refinement2.model import CTPretrainNNVNet, create_loss, dice_per_case


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patient_group_key(name: str) -> str:
    """Group serial scans from the same patient before train/validation splitting."""
    without_date_or_id = re.sub(r"^\d+", "", name)
    base_name = re.split(r"[#$@!&]", without_date_or_id, maxsplit=1)[0]
    return base_name or name


def grouped_split(cases: list[NiiCase], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    groups: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        groups.setdefault(patient_group_key(case.name), []).append(index)
    if len(groups) < 2:
        raise ValueError("Need at least two patient groups for a grouped validation split")

    group_names = sorted(groups)
    random.Random(seed).shuffle(group_names)
    target_count = max(1, int(round(len(cases) * val_ratio)))
    val_indices: list[int] = []
    for group_name in group_names:
        if val_indices and len(val_indices) >= target_count:
            break
        val_indices.extend(groups[group_name])
    val_set = set(val_indices)
    train_indices = [index for index in range(len(cases)) if index not in val_set]
    if not train_indices:
        raise ValueError("Grouped split placed every case in validation")
    return train_indices, sorted(val_indices)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)  # type: ignore[union-attr]
        target = batch["label"].to(device, non_blocking=True)  # type: ignore[union-attr]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, target)
            if training:
                if scaler is None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
        batch_size = int(inputs.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_dice += float(dice_per_case(logits.detach(), target).sum().item())
        count += batch_size
    return {"loss": total_loss / max(count, 1), "dice": total_dice / max(count, 1)}


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def write_data_manifest(
    path: Path,
    cases: list[NiiCase],
    train_indices: list[int],
    val_indices: list[int],
) -> None:
    train_set = set(train_indices)
    payload: dict[str, Any] = {
        "case_count": len(cases),
        "train_cases": [cases[index].name for index in train_indices],
        "val_cases": [cases[index].name for index in val_indices],
        "cases": [],
    }
    for index, case in enumerate(cases):
        payload["cases"].append(
            {
                "name": case.name,
                "split": "train" if index in train_set else "val",
                "group": patient_group_key(case.name),
                "quality": case.quality,
                "orig": _file_signature(case.orig_nii),
                "pretrain": _file_signature(case.pretrain_nii),
                "label": _file_signature(case.label_nii) if case.label_nii is not None else None,
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_data_manifest(path: Path, cases: list[NiiCase]) -> tuple[list[int], list[int]]:
    """Reject resume runs when a pretrain, CT, or label file has changed."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read resume data manifest: {path}") from exc
    recorded = {entry["name"]: entry for entry in payload.get("cases", [])}
    current = {case.name: case for case in cases}
    if set(recorded) != set(current):
        raise ValueError("Resume dataset cases differ from the saved data manifest")
    for name, case in current.items():
        entry = recorded[name]
        expected = {
            "orig": _file_signature(case.orig_nii),
            "pretrain": _file_signature(case.pretrain_nii),
            "label": _file_signature(case.label_nii) if case.label_nii is not None else None,
        }
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Resume input changed since training started: {name}")
    index_by_name = {case.name: index for index, case in enumerate(cases)}
    train_names = payload.get("train_cases", [])
    val_names = payload.get("val_cases", [])
    if not isinstance(train_names, list) or not isinstance(val_names, list):
        raise ValueError("Resume data manifest has invalid split lists")
    if set(train_names) | set(val_names) != set(index_by_name) or set(train_names) & set(val_names):
        raise ValueError("Resume data manifest split does not match current cases")
    return [index_by_name[name] for name in train_names], [index_by_name[name] for name in val_names]


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    best_dice: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "architecture": "ct_pretrain_residual_nnvnet" if args.residual_prior_strength > 0 else "ct_pretrain_nnvnet",
        "input_channels": CTPretrainNNVNet.input_channels,
        "prior_strength": float(args.residual_prior_strength),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "base_channels": args.base_channels,
        "grid_size": args.grid_size,
        "args": vars(args),
        "history": history,
    }


def _save_history(history: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _save_plot(history: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(epochs, [item["train"]["loss"] for item in history], label="train")
    axes[0].plot(epochs, [item["val"]["loss"] for item in history], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [item["train"]["dice"] for item in history], label="train")
    axes[1].plot(epochs, [item["val"]["dice"] for item in history], label="validation")
    axes[1].set_title("Dice")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[1].set_ylim(0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Define all paths and training defaults in code; CLI values only override them."""
    parser = argparse.ArgumentParser(description="Train CT-HU + pretrain-mask vessel refinement.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample")
    parser.add_argument("--out_dir", default=str(RESIDUAL_RUN_DIR))
    parser.add_argument("--orig_name", default="orig.nii.gz")
    parser.add_argument("--pretrain_name", default="pretrain.nii.gz")
    parser.add_argument("--label_name", default="mask.nii.gz")
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--roi_margin", type=int, default=32)
    parser.add_argument("--hu_min", type=float, default=-80.0)
    parser.add_argument("--hu_max", type=float, default=600.0)
    parser.add_argument("--loss", choices=("dice_bce", "tversky", "focal_tversky", "dice_focal_tversky"), default="dice_bce")
    parser.add_argument("--dice_weight", type=float, default=0.6)
    parser.add_argument("--tversky_alpha", type=float, default=0.7, help="False-positive penalty.")
    parser.add_argument("--tversky_beta", type=float, default=0.3, help="False-negative penalty.")
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--residual_prior_strength",
        type=float,
        default=2.0,
        help="Logit prior from pretrain; 0 restores the old free-form CT+mask model.",
    )
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--resume",
        default="auto",
        help="Resume from last.pt when it exists (the default PyCharm launch behavior).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.grid_size < 16 or args.grid_size % 8 != 0:
        raise ValueError("grid_size must be a multiple of 8 and at least 16")
    if args.roi_margin < 0 or args.epochs < 1 or args.patience < 0 or args.residual_prior_strength < 0:
        raise ValueError("roi_margin, epochs, and patience must be non-negative")
    set_seed(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "last.pt"
    if args.resume == "auto" and not checkpoint_path.exists():
        # A clean default directory starts a new run; an existing checkpoint resumes automatically.
        args.resume = None
    if args.resume is None and any((out_dir / name).exists() for name in ("best.pt", "last.pt", "history.json")):
        raise FileExistsError(
            f"Incomplete training outputs found in {out_dir} but last.pt is missing; "
            "choose a new --out_dir or restore last.pt"
        )

    cache_dir = None if args.no_cache else Path(args.cache_dir)
    dataset = CTHUVesselDataset(
        args.data_root,
        grid_size=args.grid_size,
        roi_margin=args.roi_margin,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        orig_name=args.orig_name,
        pretrain_name=args.pretrain_name,
        label_name=args.label_name,
        cache_dir=cache_dir,
    )
    manifest_path = out_dir / "data_manifest.json"
    if args.resume is None:
        train_indices, val_indices = grouped_split(dataset.cases, args.val_ratio, args.seed)
        write_data_manifest(manifest_path, dataset.cases, train_indices, val_indices)
    else:
        train_indices, val_indices = verify_data_manifest(manifest_path, dataset.cases)
    if cache_dir is not None:
        print(f"[cache] preparing {len(dataset)} cases in {cache_dir}", flush=True)
        dataset.build_cache()
        print("[cache] ready", flush=True)
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CTPretrainNNVNet(
        base_channels=args.base_channels,
        prior_strength=args.residual_prior_strength,
    ).to(device)
    criterion = create_loss(
        args.loss,
        dice_weight=args.dice_weight,
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        gamma=args.focal_gamma,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled) if amp_enabled else None
    history: list[dict[str, Any]] = []
    best_dice = float("-inf")
    start_epoch = 1

    if args.resume is not None:
        resume_path = checkpoint_path if args.resume == "auto" else Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        expected_architecture = "ct_pretrain_residual_nnvnet" if args.residual_prior_strength > 0 else "ct_pretrain_nnvnet"
        if checkpoint.get("architecture") != expected_architecture or checkpoint.get("input_channels") != 2:
            raise ValueError(f"{resume_path} is not a refinement2 two-channel checkpoint")
        if int(checkpoint["base_channels"]) != args.base_channels or int(checkpoint["grid_size"]) != args.grid_size:
            raise ValueError("Resume model configuration must match --base_channels and --grid_size")
        if abs(float(checkpoint.get("prior_strength", 0.0)) - args.residual_prior_strength) > 1e-6:
            raise ValueError("Resume model configuration must match --residual_prior_strength")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        best_dice = float(checkpoint.get("best_dice", best_dice))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    print(
        f"[train] device={device} cases={len(dataset)} train={len(train_indices)} val={len(val_indices)} "
        f"loss={args.loss} grid={args.grid_size} input_channels=2 amp={amp_enabled}",
        flush=True,
    )
    stale_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_log = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp_enabled)
        val_log = run_epoch(model, val_loader, criterion, device, amp_enabled=amp_enabled)
        history.append({"epoch": epoch, "train": train_log, "val": val_log})
        improved = val_log["dice"] > best_dice
        if improved:
            best_dice = val_log["dice"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        payload = _checkpoint_payload(model, optimizer, args, epoch, best_dice, history)
        torch.save(payload, checkpoint_path)
        if improved:
            torch.save(payload, out_dir / "best.pt")
        _save_history(history, out_dir / "history.json")
        print(
            f"[train] epoch={epoch:03d} train_loss={train_log['loss']:.4f} train_dice={train_log['dice']:.4f} "
            f"val_loss={val_log['loss']:.4f} val_dice={val_log['dice']:.4f}",
            flush=True,
        )
        if args.patience and stale_epochs >= args.patience:
            print(f"[train] early stopping after {stale_epochs} epochs without validation Dice improvement", flush=True)
            break
    _save_plot(history, out_dir / "training_curve.png")
    print(f"[train] best_val_dice={best_dice:.4f}; checkpoint={out_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
