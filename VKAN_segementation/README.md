# Portal vein STL extraction

This folder implements the workflow:

1. Run TotalSegmentator from `patient/orig.nii.gz` to extract organ masks.
2. Use deterministic preprocessing to build coarse `patient/pretrain.stl` and `patient/pretrain.nii.gz`.
3. Train the nnVnet refinement model from cropped `pretrain.nii.gz` and label NIfTI masks.
4. Predict `patient/predict_mask.nii.gz` and `patient/predict.stl`.
5. Smooth and locally quality-check the final mesh as `patient/predict_smooth.stl`.

Patient naming:

- `20210909WuJinHeng`: pre-TIPS.
- `20210921WuJinHeng#`: post-TIPS, keeps brighter TIPS voxels.
- Names containing `@`, `!`, or `&` are skipped.

## Install

```powershell
py -m pip install -r VKAN_segementation\requirements.txt
```

## Step-by-step

Run TotalSegmentator organ extraction:

```powershell
py VKAN_segementation\pretrain\totalseg.py --data_root D:\your_patient_root
```

Generate inspection/training `pretrain.stl` from TotalSegmentator outputs and `patient/orig.nii.gz`:

```powershell
py VKAN_segementation\pretrain\preprocess.py --data_root D:\your_patient_root
```

By default preprocessing reuses current outputs when the metadata says they are up to date. Add `--force`
to regenerate outputs, or `--skip_existing_pretrain` to skip patients that already have `pretrain.stl`.
Add `--only_dollar_patients` when you only want to rerun patient folders whose names contain `$`.

Train:

```powershell
py VKAN_segementation\refinement\train.py --data_root D:\your_patient_root --out_dir VKAN_segementation\runs\nnVnet3 --dataset nii --model nnVnet --grid_size 96 --epochs 400 --batch_size 1
```

Training writes `best.pt` when validation dice improves and overwrites `last.pt`
after every epoch. If training is interrupted, continue from `last.pt` with:

```powershell
py VKAN_segementation\refinement\train.py --data_root D:\your_patient_root --out_dir VKAN_segementation\runs\nnVnet3 --dataset nii --model nnVnet --resume
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
py VKAN_segementation\refinement\predict.py --data_root D:\your_patient_root --checkpoint VKAN_segementation\runs\nnVnet3\best.pt
```

Smooth and quality-check:

```powershell
py VKAN_segementation\postprocess\check_and_smooth.py --data_root D:\your_patient_root --iterations 8
```

One command:

```powershell
py VKAN_segementation\pipeline.py --data_root D:\your_patient_root --out_dir VKAN_segementation\runs\nnVnet3 --epochs 400
```

## Outputs per patient

- `pretrain.stl`: coarse vessel candidate for visual inspection; empty masks and cases over 20,000KB are flagged for review.
- `pre.stl`: optional correct-case example for debugging failed `pretrain.stl`; it is not used as a preprocessing prior.
- `mask_label.nii.gz` / `mask_smooth.nii.gz`: binary manual vessel label used by nnVnet training.
- `vessel.stl`: optional manual vessel label kept for STL debugging and overlap diagnostics; it is not used by default NIfTI training.
- `vkan_work/coarse_plan.json`: HU range and crop box used.
- `vkan_work/pretrain_meta.json`: preprocessing version, DICOM input timestamp, QA status, overlap diagnostics, and output statistics.
- `vkan_work/pretrain_mask.npy`: coarse mask for debugging.
- `predict.stl`: nnVnet refined vessel.
- `predict_smooth.stl`: smoothed final mesh.
- `vkan_work/predict_check.json`: mesh summary and deterministic quality check.

## Notes

- Coarse preprocessing now uses `patient/dcm/` directly and ignores any existing `.nii.gz` files in the patient folder. Cases marked `pretrain_quality=review` are skipped by training unless `--include_review` is passed.
- `pre.stl` and `vessel.stl` are debug/evaluation references only. Coarse preprocessing must extract `pretrain.stl` from `patient/dcm/` without using those STL files as crop, seed, envelope, or threshold priors.
- Coarse preprocessing intentionally prioritizes recall. It keeps portal vein, splenic vein, short SMV, LPV/RPV, compensation veins when visible, and TIPS for post-TIPS folders.
- The refinement model learns a full target occupancy, not a subtraction mask, so it can both delete false positives and fill small false negatives.
- NIfTI refinement crops each case around the `pretrain.nii.gz` foreground before resizing to `grid_size`, which keeps more detail than compressing the full 512x512xZ scan into the training grid. The default `grid_size=160` and `base_channels=24` are intended for a 12GB GPU; reduce to `128`/`24` or `96`/`16` if memory is tight.

## Code layout

- `pretrain/`: TotalSegmentator-backed organ masks, deterministic threshold/crop segmentation, and `pretrain.stl` export.
- `refinement/`: NIfTI/STL datasets, nnVnet training/prediction, and the original `vkan.py` model kept for reuse.
- `postprocess/`: final mesh check and smoothing.
- `utils/`: shared patient discovery, STL conversion, voxelization, and smoothing helpers.
