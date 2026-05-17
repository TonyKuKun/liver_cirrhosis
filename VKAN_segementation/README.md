# VKAN portal vein STL extraction

This folder implements the workflow:

1. Read DICOM slices from `patient/dcm/`.
2. Use Gemma/OpenAI-compatible API when configured to choose a high-recall HU window and crop box.
3. Fall back to portal-venous heuristics when the API is unavailable.
4. Save coarse `patient/pretrain.stl` for training and visual review.
5. Train a VKAN-style 3D refinement network from cropped `pretrain.nii.gz` and label NIfTI masks, then save `patient/predict_mask.nii.gz`, `patient/predict.stl`, and `patient/predict_smooth.stl`.

Patient naming:

- `20210909WuJinHeng`: pre-TIPS.
- `20210921WuJinHeng#`: post-TIPS, keeps brighter TIPS voxels.
- Names containing `@`, `!`, or `&` are skipped.

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

Generate inspection/training `pretrain.stl` from `patient/dcm/`:

```powershell
py VKAN_segementation\pretrain\preprocess.py --data_root D:\your_patient_root --model gemma-4-31b-it
```

By default preprocessing regenerates every patient's `pretrain.stl`. Add `--skip_existing_pretrain`
when you want to skip patients that already have `pretrain.stl`.

Train:

```powershell
py VKAN_segementation\refinement\train.py --data_root D:\your_patient_root --out_dir VKAN_segementation\runs\vkan --dataset nii --grid_size 96 --epochs 120 --batch_size 1
```

Training writes `best.pt` when validation dice improves and overwrites `last.pt`
after every epoch. If training is interrupted, continue from `last.pt` with:

```powershell
py VKAN_segementation\refinement\train.py --data_root D:\your_patient_root --out_dir VKAN_segementation\runs\vkan --dataset nii --resume
```

NIfTI training uses `pretrain.nii.gz` as input and `mask.nii.gz` as the default target.
Use `--label_name auto` to pick `mask_label.nii.gz` or `mask_smooth.nii.gz` instead.

If your current `mask.nii.gz` is an overlay volume containing `orig.nii.gz + mask`,
first derive a clean binary mask:

```powershell
py VKAN_segementation\pretrain\derive_mask_from_overlay.py --data_root D:\your_patient_root
```

The script renames the overlay to `origm.nii.gz`, then writes a new binary
`mask.nii.gz` from `origm.nii.gz - orig.nii.gz`. Use `--force` to rebuild an
existing `mask.nii.gz`.

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

- `pretrain.stl`: coarse vessel candidate for visual inspection; empty masks and cases over 20,000KB are flagged for review.
- `pre.stl`: optional correct-case example for debugging failed `pretrain.stl`; it is not used as a preprocessing prior.
- `mask_label.nii.gz` / `mask_smooth.nii.gz`: binary manual vessel label used by VKAN training.
- `vessel.stl`: optional manual vessel label kept for STL debugging and overlap diagnostics; it is not used by default NIfTI training.
- `vkan_work/coarse_plan.json`: HU range and crop box used.
- `vkan_work/pretrain_meta.json`: preprocessing version, DICOM input timestamp, QA status, overlap diagnostics, and output statistics.
- `vkan_work/pretrain_mask.npy`: coarse mask for debugging.
- `predict.stl`: VKAN refined vessel.
- `predict_smooth.stl`: smoothed final mesh.
- `vkan_work/predict_check.json`: mesh summary and optional LLM check.

## Notes

- Coarse preprocessing now uses `patient/dcm/` directly and ignores any existing `.nii.gz` files in the patient folder. Cases marked `pretrain_quality=review` are skipped by training unless `--include_review` is passed.
- `pre.stl` and `vessel.stl` are debug/evaluation references only. Coarse preprocessing must extract `pretrain.stl` from `patient/dcm/` without using those STL files as crop, seed, envelope, or threshold priors.
- Coarse preprocessing intentionally prioritizes recall. It keeps portal vein, splenic vein, short SMV, LPV/RPV, compensation veins when visible, and TIPS for post-TIPS folders.
- The refinement model learns a full target occupancy, not a subtraction mask, so it can both delete false positives and fill small false negatives.
- NIfTI refinement crops each case around the `pretrain.nii.gz` foreground before resizing to `grid_size`, which keeps more detail than compressing the full 512x512xZ scan into the training grid. The default `grid_size=160` and `base_channels=24` are intended for a 12GB GPU; reduce to `128`/`24` or `96`/`16` if memory is tight.

## Code layout

- `pretrain/`: DICOM loading, LLM/heuristic coarse planning, threshold/crop segmentation, and `pretrain.stl` export.
- `refinement/`: STL dataset, VKAN-style model, training, prediction, and the original `vkan.py` model kept for reuse.
- `postprocess/`: final mesh check and smoothing.
- `utils/`: shared patient discovery, Gemma client, STL conversion, voxelization, and smoothing helpers.
