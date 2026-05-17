"""Ablation definitions for the PVP predictor.

Each variant is intentionally expressed as command-line overrides for
``train.py``.  This keeps the ablation runner thin and makes every experiment
reproducible from the saved command in ``manifest.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class AblationVariant:
    name: str
    category: str
    changed_component: str
    hypothesis: str
    args: Sequence[str]


DEFAULT_VARIANTS: List[AblationVariant] = [
    AblationVariant(
        name="full_model",
        category="reference",
        changed_component="none",
        hypothesis="Reference model with all modules and losses enabled.",
        args=[],
    ),
    AblationVariant(
        name="module_no_residual",
        category="module",
        changed_component="PhysicsResidualNet",
        hypothesis="Tests whether the non-Poiseuille residual correction improves PVP prediction.",
        args=["--no_residual", "--lambda_residual", "0"],
    ),
    AblationVariant(
        name="module_no_gnn",
        category="module",
        changed_component="VesselGraphNet",
        hypothesis="Tests whether message passing between portal branches adds value beyond independent branch embeddings.",
        args=["--gnn_layers", "0"],
    ),
    AblationVariant(
        name="module_no_q_scale",
        category="module",
        changed_component="SplenicFlowEstimator",
        hypothesis="Tests whether patient-specific flow scale from organ volumes helps beyond fixed reference flow.",
        args=["--no_q_scale"],
    ),
    AblationVariant(
        name="module_no_physics_baseline",
        category="module",
        changed_component="Poiseuille baseline anchor",
        hypothesis="Tests whether the explicit pressure-drop baseline is helping or just adding noisy bias.",
        args=["--no_physics_baseline"],
    ),
    AblationVariant(
        name="module_no_aux",
        category="module",
        changed_component="Auxiliary/system scalars",
        hypothesis="Tests how much the model depends on clinical/system-level shortcut signals such as TIPS state.",
        args=["--no_aux"],
    ),
    AblationVariant(
        name="module_no_flow_features",
        category="module",
        changed_component="Flow and junction features in predictor",
        hypothesis="Tests whether learned flow fractions and junction residuals add information to the final predictor.",
        args=["--no_flow_features"],
    ),
    AblationVariant(
        name="module_no_branch_embed",
        category="module",
        changed_component="Learned pointwise geometry embeddings",
        hypothesis="Tests whether the learned branch encoder contributes beyond physics, flow, and auxiliary features.",
        args=["--no_branch_embed"],
    ),
    AblationVariant(
        name="module_data_mlp_only",
        category="module",
        changed_component="Physics path removed from final predictor",
        hypothesis="Tests a mostly data-driven predictor by removing the physics anchor and physics-feature inputs.",
        args=[
            "--no_physics_baseline",
            "--no_flow_features",
            "--lambda_murray", "0",
            "--lambda_press", "0",
            "--lambda_smooth", "0",
            "--lambda_physio", "0",
            "--lambda_mono", "0",
        ],
    ),
    AblationVariant(
        name="loss_no_murray",
        category="loss",
        changed_component="Murray flow-prior loss",
        hypothesis="Tests whether Murray-law regularization stabilizes flow allocation.",
        args=["--lambda_murray", "0"],
    ),
    AblationVariant(
        name="loss_no_press",
        category="loss",
        changed_component="Pressure consistency loss",
        hypothesis="Tests whether bifurcation pressure residual regularization helps prediction.",
        args=["--lambda_press", "0"],
    ),
    AblationVariant(
        name="loss_no_smooth",
        category="loss",
        changed_component="Effective-radius smoothness loss",
        hypothesis="Tests whether smoothing geometry-derived resistance avoids overfitting noisy centerline profiles.",
        args=["--lambda_smooth", "0"],
    ),
    AblationVariant(
        name="loss_no_physio",
        category="loss",
        changed_component="WSS/Re physiological range loss",
        hypothesis="Tests whether physiological WSS/Re constraints improve generalization.",
        args=["--lambda_physio", "0"],
    ),
    AblationVariant(
        name="loss_no_mono",
        category="loss",
        changed_component="Pressure monotonicity loss",
        hypothesis="Tests whether enforcing monotonic cumulative pressure drop matters.",
        args=["--lambda_mono", "0"],
    ),
    AblationVariant(
        name="loss_no_residual_penalty",
        category="loss",
        changed_component="Residual magnitude penalty",
        hypothesis="Tests whether penalizing the residual branch prevents it from overpowering the physics path.",
        args=["--lambda_residual", "0"],
    ),
    AblationVariant(
        name="loss_no_spread",
        category="loss",
        changed_component="Anti-shrinkage spread loss",
        hypothesis="Tests whether the spread loss prevents regression-to-mean predictions.",
        args=["--lambda_spread", "0"],
    ),
    AblationVariant(
        name="loss_no_tail_weight",
        category="loss",
        changed_component="High/low pressure extremity weighting",
        hypothesis="Tests whether tail weighting helps high-PVP and low-PVP extremes or just adds variance.",
        args=["--extremity_alpha", "0"],
    ),
    AblationVariant(
        name="loss_main_only",
        category="loss",
        changed_component="All physics regularizers",
        hypothesis="Tests a pure supervised objective while keeping the architecture intact.",
        args=[
            "--lambda_murray", "0",
            "--lambda_press", "0",
            "--lambda_smooth", "0",
            "--lambda_physio", "0",
            "--lambda_mono", "0",
            "--lambda_residual", "0",
            "--lambda_spread", "0",
        ],
    ),
    AblationVariant(
        name="train_no_extreme_sampler",
        category="training",
        changed_component="Extreme-value oversampling",
        hypothesis="Tests whether oversampling pressure extremes helps OOF accuracy or overfits rare tails.",
        args=["--sample_power", "0"],
    ),
]


def variant_by_name() -> Dict[str, AblationVariant]:
    return {v.name: v for v in DEFAULT_VARIANTS}


def select_variants(names: Iterable[str] | None) -> List[AblationVariant]:
    if not names:
        return list(DEFAULT_VARIANTS)
    lookup = variant_by_name()
    selected = []
    for name in names:
        if name not in lookup:
            valid = ", ".join(sorted(lookup))
            raise ValueError(f"Unknown ablation variant '{name}'. Valid names: {valid}")
        selected.append(lookup[name])
    return selected

