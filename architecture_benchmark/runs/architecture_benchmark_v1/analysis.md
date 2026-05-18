# Architecture Benchmark Analysis

Best by MAE: `numeric_transformer` (numeric_only / numeric_transformer) MAE 3.352, RMSE 4.119, R2 0.586.

## Top Experiments
- numeric_transformer: MAE 3.352, RMSE 4.119, R2 0.586, delta vs current +0.103, delta vs traditional baseline -0.130.
- numeric_gnn: MAE 3.707, RMSE 4.646, R2 0.473, delta vs current +0.458, delta vs traditional baseline +0.225.
- numeric_cnn: MAE 3.796, RMSE 4.677, R2 0.466, delta vs current +0.547, delta vs traditional baseline +0.314.
- numeric_mlp: MAE 3.812, RMSE 4.906, R2 0.413, delta vs current +0.563, delta vs traditional baseline +0.330.
- fusion_numeric_stl: MAE 3.864, RMSE 4.617, R2 0.480, delta vs current +0.615, delta vs traditional baseline +0.383.
- numeric_cnn_gnn: MAE 4.235, RMSE 5.256, R2 0.326, delta vs current +0.986, delta vs traditional baseline +0.753.
- stl_pointnet_centerline_gnn: MAE 4.533, RMSE 5.682, R2 0.212, delta vs current +1.284, delta vs traditional baseline +1.051.
- stl_pointnet: MAE 4.535, RMSE 5.638, R2 0.224, delta vs current +1.287, delta vs traditional baseline +1.054.

## Reading Guide
- If `numeric_cnn_gnn` is best, centerline numeric profiles plus topology are sufficient for now.
- If `stl_centerline_gnn` beats numeric models, 3D centerline geometry is carrying signal beyond hand-crafted profile channels.
- If `stl_pointnet` is weak but centerline GNN is strong, vessel surface STL is probably noisy and centerline structure is the better 3D input.
- If `fusion_numeric_stl` wins, STL and numeric features are complementary and should be fused in the next main model.
- Compare liver_valid and spleen_valid groups in `per_group_summary.json` before trusting liver-driven improvements.
