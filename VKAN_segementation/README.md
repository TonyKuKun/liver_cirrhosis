# VKAN portal vein STL extraction

This folder implements the workflow:

1. Read `patient/dcm` DICOM CT slices.
2. Use Gemma/OpenAI-compatible API when configured to choose a high-recall HU window and crop box.
3. Fall back to portal-venous heuristics when the API is unavailable.
4. Save coarse `patient/pretrain.stl`.
5. Train a VKAN-style 3D refinement network with `pretrain.stl` as input and `vessel.stl` as label.
6. Save `patient/predict.stl`, then `patient/predict_smooth.stl`.

Patient naming:

- `20210909WuJinHeng`: pre-TIPS.
- `20210921WuJinHeng#`: post-TIPS, keeps brighter TIPS voxels.
- Names containing `@` or `!` are skipped.

## Install

```powershell
py -m pip install -r VKAN_segementation\requirements.txt
```

## Configure model API

The client is OpenAI-compatible and calls:

```text
{GEMMA_API_BASE_URL}/chat/completions
```

Set these if you want LLM-assisted threshold/crop and final mesh review:

```powershell
$env:GEMMA_API_KEY="your-api-key"
$env:GEMMA_API_BASE_URL="https://your-provider/v1"
```

The default model name is `gemma-4-31b-it`. If the API is not configured or fails, preprocessing still runs with deterministic heuristics.

## Step-by-step

Generate coarse `pretrain.stl`:

```powershell
py VKAN_segementation\pretrain\preprocess.py --data_root D:\your_patient_root --model gemma-4-31b-it
```

Train:

```powershell
py VKAN_segementation\refinement\train.py --data_root D:\your_patient_root --out_dir VKAN_segementation\runs\vkan --grid_size 96 --epochs 120 --batch_size 1
```

Predict:

```powershell
py VKAN_segementation\refinement\predict.py --data_root D:\your_patient_root --checkpoint VKAN_segementation\runs\vkan\best.pt
```

Smooth and quality-check:

```powershell
py VKAN_segementation\postprocess\check_and_smooth.py --data_root D:\your_patient_root --iterations 8
```

One command:

```powershell
py VKAN_segementation\pipeline.py --data_root D:\your_patient_root --out_dir VKAN_segementation\runs\vkan --epochs 120
```

## Outputs per patient

- `pretrain.stl`: coarse, high-recall vessel candidate.
- `vkan_work/coarse_plan.json`: HU range and crop box used.
- `vkan_work/pretrain_mask.npy`: coarse mask for debugging.
- `predict.stl`: VKAN refined vessel.
- `predict_smooth.stl`: smoothed final mesh.
- `vkan_work/predict_check.json`: mesh summary and optional LLM check.

## Notes

- Coarse preprocessing intentionally prioritizes recall. It keeps portal vein, splenic vein, short SMV, LPV/RPV, compensation veins when visible, and TIPS for post-TIPS folders.
- The refinement model learns a full target occupancy, not a subtraction mask, so it can both delete false positives and fill small false negatives.
- `grid_size=96` is a practical default. Increase to `128` if GPU memory allows.

## Code layout

- `pretrain/`: DICOM loading, LLM/heuristic coarse planning, threshold/crop segmentation, and `pretrain.stl` export.
- `refinement/`: STL dataset, VKAN-style model, training, prediction, and the original `vkan.py` model kept for reuse.
- `postprocess/`: final mesh check and smoothing.
- `utils/`: shared patient discovery, Gemma client, STL conversion, voxelization, and smoothing helpers.
