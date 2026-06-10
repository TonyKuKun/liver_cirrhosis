"""Run one-factor ablations for the current final PVP model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

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
    Variant("full_model", "reference", "none", "8-vessel model with organ global features, one PVP head, training dropout regularization, and L2 plus light core shunt loss.", []),
    Variant("no_organ_global_features", "module", "organ global features", "Tests liver/spleen volumes as global context.", ["--no_organ_global_features"]),
    Variant("no_global_flow_corrector", "module", "GlobalFlowCorrector", "Tests whether global features improve corrected Q states.", ["--no_global_flow_corrector"]),
    Variant("with_physics_residual", "module", "PhysicsResidualNet", "Tests whether the extra internal physics residual correction branch still helps.", ["--use_physics_residual"]),
    Variant("no_dropout_regularizer", "module", "AuxiliaryDropoutRegularizer", "Tests the training-only stochastic regularizer.", ["--no_dropout_regularizer"]),
    Variant("no_flow_graph", "module", "FlowGraphRefiner", "Tests whether CenterlinePoints-aware graph message passing helps.", ["--no_flow_graph"]),
    Variant("fixed_physics_params", "module", "LearnablePhysicsLayer parameters", "Tests learnable physical calibration versus fixed constants.", ["--fixed_physics_params"]),
    Variant("all_profile_channels", "geometry", "optional profile channels", "Tests whether optional geometry channels add signal or noise.", ["--use_all_profile_channels"]),
    Variant("use_unreliable_raw_lengths", "geometry", "unreliable raw lengths", "Tests whether raw SMV/LPV/RPV lengths should stay excluded.", ["--use_unreliable_raw_lengths"]),
    Variant("six_vessel_layout", "layout", "six-vessel layout", "Tests merged collateral/TIPS layout.", ["--use_six_vessel_layout"]),
    Variant("three_vessel_layout", "layout", "three-vessel layout", "Tests compact collateral/MPV/SV layout.", ["--use_three_vessel_layout"]),
]


LOSS_VARIANTS = [
    Variant("loss_l2_only", "loss", "L2 only", "Pure PVP regression supervision.", ["--lambda_shunt", "0"]),
    Variant("loss_l2_plus_core_split", "loss", "add core confluence shunt loss", "Tests the selected light MPV=SMV+SV core confluence constraint.", ["--lambda_shunt", "0.005", "--split_loss_mode", "core_confluence"]),
    Variant("loss_l2_plus_full_split", "loss", "add full shunt loss", "Tests the broad split-flow residual at the same light weight.", ["--lambda_shunt", "0.005", "--split_loss_mode", "full"]),
]


def variants_for_suite(suite: str):
    if suite == "model":
        return MODEL_VARIANTS
    if suite == "loss":
        return LOSS_VARIANTS
    return MODEL_VARIANTS + LOSS_VARIANTS


def select_variants(names: Sequence[str] | None, suite: str):
    candidates = variants_for_suite(suite)
    if not names:
        return candidates
    lookup = {v.name: v for v in MODEL_VARIANTS + LOSS_VARIANTS}
    missing = [n for n in names if n not in lookup]
    if missing:
        raise ValueError(f"Unknown variants: {missing}. Valid: {sorted(lookup)}")
    return [lookup[n] for n in names]


def build_command(args, variant: Variant, out_dir: Path, epochs: int, n_folds: int):
    loss_args = (
        list(variant.args)
        if variant.category == "loss"
        else ["--lambda_shunt", str(args.lambda_shunt), "--split_loss_mode", args.split_loss_mode, *variant.args]
    )
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
        "--sample_power", str(args.sample_power),
        "--no_organ_flow_scale",
        *loss_args,
    ]


def run_stage(args, variants: Sequence[Variant], stage_name: str, epochs: int, n_folds: int):
    base_out_root = Path(args.out_root)
    if not base_out_root.is_absolute():
        base_out_root = ROOT / base_out_root
    out_root = base_out_root / stage_name
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for variant in variants:
        out_dir = out_root / variant.name
        item = asdict(variant)
        item.update({
            "out_dir": str(out_dir),
            "command": build_command(args, variant, out_dir, epochs, n_folds),
            "stage": stage_name,
        })
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
    ap.add_argument("--out_root", type=str, default=str(ROOT / "ablation" / "runs" / "final_20260610"))
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
    ap.add_argument("--lambda_shunt", type=float, default=0.005)
    ap.add_argument("--split_loss_mode", choices=["full", "core_confluence"], default="core_confluence")
    ap.add_argument("--sample_power", type=float, default=1.5)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    variants = select_variants(args.variants, suite=args.suite)
    if args.stage in {"smoke", "all"}:
        run_stage(args, variants, "smoke", args.smoke_epochs, args.smoke_n_folds)
    if args.stage in {"full", "all"}:
        run_stage(args, variants, "full", args.full_epochs, args.full_n_folds)


if __name__ == "__main__":
    main()
