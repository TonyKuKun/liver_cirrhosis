"""Run staged ablations for the new physics-constrained model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation.report import summarize


@dataclass(frozen=True)
class Variant:
    name: str
    category: str
    changed_component: str
    hypothesis: str
    args: Sequence[str]


MODEL_VARIANTS = [
    Variant("full_model", "reference", "none", "All proposed modules enabled.", []),
    Variant("with_organ_flow_scale", "module", "OrganFlowScaleNet", "Tests whether hard spleen/liver volume q-scaling should be enabled.", ["--use_organ_flow_scale"]),
    Variant("no_global_flow_corrector", "module", "GlobalFlowCorrector", "Tests whether global features improve flow features.", ["--no_global_flow_corrector"]),
    Variant("no_flow_graph", "module", "FlowGraphRefiner", "Tests whether graph message passing helps flow correction.", ["--no_flow_graph"]),
    Variant("no_physics_residual", "module", "PhysicsResidualNet", "Tests whether non-ideal physics residuals help.", ["--no_physics_residual"]),
    Variant("fixed_physics_params", "module", "LearnablePhysicsLayer parameters", "Tests whether learning physical calibration beats fixed Poiseuille-like constants.", ["--fixed_physics_params"]),
    Variant("all_profile_channels", "geometry", "optional profile channels", "Tests whether optional geometry channels are useful or noisy.", ["--use_all_profile_channels"]),
    Variant("use_unreliable_raw_lengths", "geometry", "unreliable raw length strategy", "Tests whether using raw SMV/LPV/RPV lengths hurts the physics path.", ["--use_unreliable_raw_lengths"]),
    Variant("six_vessel_layout", "layout", "Merged collateral/TIPS branch", "Tests the six-vessel layout with LGV/PGV/TIPS merged into one typed collateral branch.", ["--use_six_vessel_layout"]),
    Variant("three_vessel_layout", "layout", "Collateral/MPV/SV compact layout", "Tests the aggressive three-vessel layout with SMV/LPV/RPV as helper global/loss signals.", ["--use_three_vessel_layout"]),
]


LOSS_VARIANTS = [
    Variant(
        "loss_mse_only",
        "reference",
        "main supervision only",
        "Pure MSE regression without any physics regularization.",
        ["--disable_physics_losses"],
    ),
    Variant(
        "loss_mse_plus_continuity",
        "loss",
        "within-vessel continuity",
        "MSE plus the A*v flow-continuity constraint only.",
        ["--lambda_press", "0", "--lambda_mono", "0", "--lambda_smooth", "0", "--lambda_physio", "0", "--lambda_residual", "0"],
    ),
    Variant(
        "loss_mse_plus_split",
        "loss",
        "junction split conservation",
        "MSE plus the junction flow-split conservation constraint only.",
        ["--lambda_flow_prior", "0", "--lambda_mono", "0", "--lambda_smooth", "0", "--lambda_physio", "0", "--lambda_residual", "0"],
    ),
    Variant(
        "loss_mse_plus_pressure_mono",
        "loss",
        "pressure monotonicity",
        "MSE plus cumulative pressure monotonicity only.",
        ["--lambda_flow_prior", "0", "--lambda_press", "0", "--lambda_smooth", "0", "--lambda_physio", "0", "--lambda_residual", "0"],
    ),
    Variant(
        "loss_mse_plus_flow_conservation",
        "loss",
        "flow conservation family",
        "MSE plus both flow-continuity and junction split constraints.",
        ["--lambda_mono", "0", "--lambda_smooth", "0", "--lambda_physio", "0", "--lambda_residual", "0"],
    ),
    Variant(
        "loss_mse_plus_all_simple_physics",
        "loss",
        "all simplified physics constraints",
        "MSE plus flow conservation and pressure monotonicity.",
        ["--lambda_smooth", "0", "--lambda_physio", "0", "--lambda_residual", "0"],
    ),
]


DEFAULT_VARIANTS = MODEL_VARIANTS + LOSS_VARIANTS


def variants_for_suite(suite: str):
    if suite == "model":
        return MODEL_VARIANTS
    if suite == "loss":
        return LOSS_VARIANTS
    return DEFAULT_VARIANTS


def select_variants(names: Sequence[str] | None, suite: str = "all"):
    candidates = variants_for_suite(suite)
    if not names:
        return candidates
    lookup = {v.name: v for v in DEFAULT_VARIANTS}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(f"Unknown variants: {missing}. Valid: {sorted(lookup)}")
    return [lookup[n] for n in names]


def build_command(args, variant: Variant, out_dir: Path, epochs: int, n_folds: int):
    return [
        args.python,
        str(ROOT / "train.py"),
        "--data_root", args.data_root,
        "--out_dir", str(out_dir),
        "--n_points", str(args.n_points),
        "--n_folds", str(n_folds),
        "--seed", str(args.seed),
        "--epochs", str(epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--print_every", str(args.print_every),
        "--d_hidden", str(args.d_hidden),
        "--dropout", str(args.dropout),
        "--flow_gnn_layers", str(args.flow_gnn_layers),
        "--lambda_flow_prior", str(args.lambda_flow_prior),
        "--lambda_press", str(args.lambda_press),
        "--lambda_smooth", str(args.lambda_smooth),
        "--lambda_physio", str(args.lambda_physio),
        "--lambda_mono", str(args.lambda_mono),
        "--lambda_residual", str(args.lambda_residual),
        "--lambda_spread", str(args.lambda_spread),
        "--sample_power", str(args.sample_power),
        *variant.args,
    ]


def run_stage(args, variants: Sequence[Variant], stage_name: str, epochs: int, n_folds: int):
    out_root = Path(args.out_root) / stage_name
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for variant in variants:
        out_dir = out_root / variant.name
        cmd = build_command(args, variant, out_dir, epochs=epochs, n_folds=n_folds)
        item = asdict(variant)
        item.update({"out_dir": str(out_dir), "command": cmd, "stage": stage_name})
        manifest.append(item)
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        for item in manifest:
            print(" ".join(item["command"]))
        return summarize(out_root, manifest)

    for item in manifest:
        summary = Path(item["out_dir"]) / "summary.json"
        if summary.exists() and not args.force:
            print(f"[Ablation:{stage_name}] {item['name']} exists; skipping.")
            continue
        print(f"[Ablation:{stage_name}] Running {item['name']}")
        subprocess.run(item["command"], cwd=str(ROOT), check=True)
        if args.prune_checkpoints:
            prune_checkpoints(Path(item["out_dir"]))
    return summarize(out_root, manifest)


def prune_checkpoints(out_dir: Path) -> None:
    removed = 0
    for checkpoint in out_dir.glob("fold_*/best.pt"):
        try:
            checkpoint.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    if removed:
        print(f"[Ablation] Pruned {removed} fold checkpoints under {out_dir}.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=str, default=r"F:\PCG data\dataset\test4all_sample")
    ap.add_argument("--out_root", type=str, default=str(ROOT / "ablation" / "runs" / "ablations"))
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--suite", choices=["model", "loss", "all"], default="all")
    ap.add_argument("--stage", choices=["smoke", "full", "all"], default="all")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prune_checkpoints", dest="prune_checkpoints", action="store_true", default=True)
    ap.add_argument("--no_prune_checkpoints", dest="prune_checkpoints", action="store_false")
    ap.add_argument("--n_points", type=int, default=200)
    ap.add_argument("--smoke_n_folds", type=int, default=2)
    ap.add_argument("--smoke_epochs", type=int, default=3)
    ap.add_argument("--full_n_folds", type=int, default=5)
    ap.add_argument("--full_epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--d_hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--flow_gnn_layers", type=int, default=2)
    ap.add_argument("--lambda_flow_prior", type=float, default=0.05)
    ap.add_argument("--lambda_press", type=float, default=0.03)
    ap.add_argument("--lambda_smooth", type=float, default=0.0)
    ap.add_argument("--lambda_physio", type=float, default=0.0)
    ap.add_argument("--lambda_mono", type=float, default=0.02)
    ap.add_argument("--lambda_residual", type=float, default=0.0)
    ap.add_argument("--lambda_spread", type=float, default=0.0)
    ap.add_argument("--sample_power", type=float, default=1.5)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    variants = select_variants(args.variants, suite=args.suite)
    if args.stage in {"smoke", "all"}:
        run_stage(args, variants, "smoke", epochs=args.smoke_epochs, n_folds=args.smoke_n_folds)
    if args.stage in {"full", "all"}:
        run_stage(args, variants, "full", epochs=args.full_epochs, n_folds=args.full_n_folds)


if __name__ == "__main__":
    main()
