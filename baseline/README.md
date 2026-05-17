# Traditional Baselines

This folder contains reproducible non-deep-learning baselines for PVP
prediction.  The goal is a fair comparison against the main geometric
physics-informed model, using the same `PortalVeinDataset`, the same
subject-level PVP-balanced folds, and the same OOF metric style.

## Why These Baselines

Direct portal-vein centerline-geometry baselines for PVP are uncommon in the
literature.  Portal-hypertension work more often uses CT radiomics, while
coronary CT-FFR work commonly uses vessel geometry, stenosis descriptors,
pressure-drop proxies, CFD, or reduced-order hemodynamic models.  This suite
therefore includes:

- naive train-fold mean and median predictors;
- pure geometry summaries from branch profiles;
- Poiseuille/Murray-inspired resistance and pressure-drop proxies;
- auxiliary/system features from `AUX_KEYS` and extra unified-feature scalars;
- traditional regressors on each feature set.

Useful reference anchors:

- CT radiomics for portal pressure: https://pubmed.ncbi.nlm.nih.gov/32146345/
- CT-derived FFR/CFD: https://pmc.ncbi.nlm.nih.gov/articles/PMC3790916/
- Reduced-order FFR: https://pubmed.ncbi.nlm.nih.gov/28600860/
- Arterial stenosis pressure-drop reduced-order modeling: https://pmc.ncbi.nlm.nih.gov/articles/PMC8486142/

## Run

```bash
conda run -n pytorch python baseline/run_baselines.py \
  --data_root "F:\PCG data\dataset\test4all_sample" \
  --split_json runs/v5.1/splits.json \
  --out_dir runs/baseline_v1 \
  --n_points 200
```

By default the script evaluates `geometry`, `physics`, `aux`, and `combined`
feature sets with:

`LinearRegression`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, RBF `SVR`, `KNN`,
`RandomForest`, `ExtraTrees`, `GradientBoosting`, `HistGradientBoosting`, and
`AdaBoost`, plus train-fold mean and median baselines.

## Outputs

- `oof_predictions.csv`: long-form OOF predictions with `feature_set` and
  `model` columns.
- `summary.csv`: one row per baseline with overall and fold-mean metrics.
- `summary.json`: full metrics and fold records.
- `per_group_summary.json`: diagnostic group summaries per baseline.
- `feature_schema.json`: generated tabular features and dropped all-missing
  columns.
- `feature_importance.csv`: tree importances or absolute linear coefficients
  averaged across folds when available.

