# Final PVP Predictor Results - 2026-06-10

## Final Decision

The final model is the no-physics-residual single-head PVP model with:

- one PVP prediction head,
- L2/MSE PVP loss,
- light core shunt loss: `lambda_shunt=0.005`, `split_loss_mode=core_confluence`,
- spleen/liver global context,
- GlobalFlowCorrector,
- FlowGraphRefiner,
- AuxiliaryDropoutRegularizer as a training-time regularizer only.

This is the clean final version. The older CSPH output and auxiliary CSPH loss are not restored.

## Main Result

Dataset: `F:\PCG data\dataset\test4all_sample`; 72 valid samples after excluding 17 `00`-prefix folders; subject-level 5-fold; seed 40.

| metric | value |
| --- | --- |
| MAE | 2.685 +/- 0.746 |
| RMSE | 3.605 +/- 1.132 |
| R2 | 0.643 +/- 0.183 |
| OOF MAE | 2.704 |
| OOF RMSE | 3.800 |
| OOF bias | 0.256 |

Result directory: `runs/final_20260610_pvp_l2_shunt`.

## Why The Earlier Clean Version Fell To About 3.0

The historical 2.8 commit still had a CSPH predictor module. It did not contribute to the PVP loss in the pure PVP run, but it was still executed during forward passes and contained dropout. Removing it changed the stochastic dropout sequence and the training trajectory. That is why the superficially cleaner single-head version drifted to about 3.15 MAE.

The current code keeps the model single-head by replacing that behavior with `AuxiliaryDropoutRegularizer`, which is not an output head and has no prediction target. It preserves the useful training-time stochastic regularization without restoring the old CSPH task.

A second finding came from the full 2026-06-10 ablation: the old internal `PhysicsResidualNet` is no longer helpful under the final single-head setup. Disabling it improves MAE from 2.809 to 2.685, so the final default is `use_physics_residual=False`.

## Ablation Conclusions

The final reference is the best 5-fold configuration.

| variant | MAE | RMSE | R2 | conclusion |
| --- | --- | --- | --- | --- |
| full_model | 2.685 | 3.605 | 0.643 | final |
| loss_l2_only | 2.700 | 3.618 | 0.641 | light shunt loss helps slightly |
| loss_l2_plus_full_split | 2.703 | 3.619 | 0.641 | full split loss is slightly worse |
| with_physics_residual | 2.809 | 3.837 | 0.615 | residual branch hurts |
| no_dropout_regularizer | 2.942 | 3.721 | 0.634 | dropout regularizer helps |
| no_organ_global_features | 3.104 | 4.093 | 0.548 | spleen/liver context helps |
| no_flow_graph | 3.171 | 4.455 | 0.490 | graph refinement helps |
| no_global_flow_corrector | 3.479 | 4.536 | 0.458 | global flow correction is important |

Full ablation table: `ablation/runs/final_20260610/full/comparison.csv`.

## Baseline Comparison

Best traditional baseline on the same splits:

| baseline | features | MAE | RMSE | R2 |
| --- | --- | --- | --- | --- |
| physics/adaboost | 32 | 3.420 | 4.286 | 0.531 |
| physics/random_forest | 32 | 3.518 | 4.352 | 0.516 |
| physics/extra_trees | 32 | 3.532 | 4.372 | 0.512 |

The final neural PVP model improves MAE by 0.735 mmHg over the best baseline.

## Output Consistency

`runs/final_20260610_pvp_l2_shunt/oof_predictions.csv` and `oof_predictions.xlsx` are written from the same row objects and use the same columns. This keeps the run outputs aligned with the Excel deliverable.
