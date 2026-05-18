# Architecture Benchmark Analysis

Best by MAE: `numeric_cnn_gnn` (numeric_only / numeric_cnn_gnn) MAE 4.976, RMSE 6.120, R2 0.029.

Note: completed experiments use n=52 loaded samples. Compare against older 62-sample runs with caution unless the same dataset filtering is restored.

## Top Experiments
- numeric_cnn_gnn: MAE 4.976, RMSE 6.120, R2 0.029, delta vs current +1.727, delta vs traditional baseline +1.494.
- stl_centerline_gnn: MAE 5.143, RMSE 6.205, R2 0.001, delta vs current +1.895, delta vs traditional baseline +1.662.
- fusion_numeric_stl: MAE 5.176, RMSE 6.188, R2 0.007, delta vs current +1.927, delta vs traditional baseline +1.695.

## Reading Guide
- If `numeric_cnn_gnn` is best, centerline numeric profiles plus topology are sufficient for now.
- If `stl_centerline_gnn` beats numeric models, 3D centerline geometry is carrying signal beyond hand-crafted profile channels.
- If `stl_pointnet` is weak but centerline GNN is strong, vessel surface STL is probably noisy and centerline structure is the better 3D input.
- If `fusion_numeric_stl` wins, STL and numeric features are complementary and should be fused in the next main model.
- Compare liver_valid and spleen_valid groups in `per_group_summary.json` before trusting liver-driven improvements.
