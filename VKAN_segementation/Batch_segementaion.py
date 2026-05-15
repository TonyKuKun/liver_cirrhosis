#!/usr/bin/env python3
"""
批量CT分割 + STL导出工具
========================
输入：包含多个病人文件夹的目录，每个病人文件夹下有 orig.nii.gz
输出：每个病人文件夹下生成 segmentation/ 目录，包含:
      - spleen.stl
      - liver.stl
      - bone.stl     (所有骨骼合并)
      - kidney.stl   (左右肾合并)

依赖安装:
    pip install TotalSegmentator nibabel numpy numpy-stl scikit-image --break-system-packages

用法:
    python batch_segment_to_stl.py /path/to/patients
    python batch_segment_to_stl.py /path/to/patients --fast        # 快速模式(低分辨率)
    python batch_segment_to_stl.py /path/to/patients --device gpu   # 指定GPU
"""

import argparse
import sys
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage import measure
from stl import mesh as stl_mesh


# ── TotalSegmentator 输出的类名映射 ──────────────────────────────────
# 需要分割的 ROI 子集（减少运行时间和内存）
ROI_SUBSET = [
    "spleen",
    "liver",
    "kidney_left", "kidney_right",
    # 骨骼 — TotalSegmentator 的主要骨骼类名
    "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
    "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
    "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
    "vertebrae_T2", "vertebrae_T1",
    "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3",
    "vertebrae_C2", "vertebrae_C1",
    "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
    "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
    "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
    "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
    "hip_left", "hip_right",
    "sacrum",
    "femur_left", "femur_right",
    "scapula_left", "scapula_right",
    "clavicula_left", "clavicula_right",
    "humerus_left", "humerus_right",
    "sternum",
]

# 定义输出STL与分割类名的对应关系
STL_GROUPS = {
    "spleen.stl": ["spleen"],
    "liver.stl": ["liver"],
    "kidney.stl": ["kidney_left", "kidney_right"],
    "bone.stl": [name for name in ROI_SUBSET if name not in
                 ("spleen", "liver", "kidney_left", "kidney_right")],
}


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def iter_with_progress(items: list[Path]):
    try:
        from tqdm import tqdm
        return tqdm(items, total=len(items), desc="Patients", unit="patient")
    except ImportError:
        return items


def discover_patient_dirs(root: Path) -> list[Path]:
    """Use the same patient discovery rules as totalseg.py when available."""
    try:
        from utils.common import discover_patients
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from utils.common import discover_patients
        except ImportError:
            return sorted([
                d for d in root.iterdir()
                if d.is_dir() and (d / "orig.nii.gz").exists()
            ])

    return [case.path for case in discover_patients(root)]


def stl_is_done(stl_path: Path) -> bool:
    """Return True when an STL already exists and is not empty."""
    return stl_path.exists() and stl_path.stat().st_size > 0


def missing_stl_names(seg_dir: Path, overwrite: bool) -> list:
    """Return STL files that still need to be generated."""
    if overwrite:
        return list(STL_GROUPS.keys())
    return [
        stl_name for stl_name in STL_GROUPS
        if not stl_is_done(seg_dir / stl_name)
    ]


def masks_exist_for_stls(ts_output: Path, stl_names: list) -> bool:
    """Return True when the TotalSegmentator masks needed by STL files exist."""
    required_classes = {
        class_name
        for stl_name in stl_names
        for class_name in STL_GROUPS[stl_name]
    }
    return all((ts_output / f"{class_name}.nii.gz").exists()
               for class_name in required_classes)


def nifti_mask_to_stl(mask_data: np.ndarray, affine: np.ndarray,
                      output_path: Path, smooth: bool = True,
                      step_size: int = 1) -> bool:
    """
    将二值 mask (numpy array) 通过 Marching Cubes 转换为 STL 文件。
    affine 用于把体素坐标变换到物理坐标(mm)。
    返回 True 表示成功生成，False 表示 mask 为空。
    """
    if mask_data.sum() == 0:
        return False

    # Marching Cubes 提取表面
    verts, faces, normals, _ = measure.marching_cubes(
        mask_data.astype(np.float32),
        level=0.5,
        step_size=step_size,
        allow_degenerate=False,
    )

    # 体素坐标 → 物理坐标 (mm)
    # affine 是 4x4 矩阵: [R|t; 0 1]
    ones = np.ones((verts.shape[0], 1))
    verts_homo = np.hstack([verts, ones])                   # (N, 4)
    verts_phys = (affine @ verts_homo.T).T[:, :3]           # (N, 3)

    # 构造 numpy-stl 的 Mesh 对象
    stl_model = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_model.vectors[i][j] = verts_phys[f[j], :]

    stl_model.save(str(output_path))
    return True


def merge_masks(seg_dir: Path, class_names: list) -> tuple:
    """
    读取并合并多个分割 NIfTI 文件为一个二值 mask。
    返回 (merged_mask, affine) 或 (None, None) 如果全部不存在。
    """
    merged = None
    affine = None

    for name in class_names:
        nii_path = seg_dir / f"{name}.nii.gz"
        if not nii_path.exists():
            continue
        img = nib.load(str(nii_path))
        data = img.get_fdata().astype(np.uint8)
        if merged is None:
            merged = data
            affine = img.affine
        else:
            merged = np.maximum(merged, data)

    return merged, affine


def run_totalsegmentator(input_nii: Path, output_dir: Path,
                         device: str = "gpu", fast: bool = False) -> bool:
    """调用 TotalSegmentator CLI 进行分割。"""
    cmd = [
        "TotalSegmentator",
        "-i", str(input_nii),
        "-o", str(output_dir),
        "--roi_subset", *ROI_SUBSET,
        "--device", device,
        "--nr_thr_resamp", "1",
        "--nr_thr_saving", "1",
    ]
    if fast:
        cmd.append("--fast")

    print(f"  ▸ 运行 TotalSegmentator ...")
    print(f"    命令: {' '.join(cmd[:6])} ... (共 {len(ROI_SUBSET)} 个 ROI)")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"  ✗ TotalSegmentator 失败:\n{result.stderr}")
        return False
    return True


def process_patient(patient_dir: Path, device: str, fast: bool,
                    step_size: int, overwrite: bool) -> str:
    """处理单个病人文件夹。"""
    patient_start = time.perf_counter()
    input_nii = patient_dir / "orig.nii.gz"
    seg_dir = patient_dir / "segmentation"
    ts_output = seg_dir / "ts_raw"          # TotalSegmentator 原始输出

    print(f"\n{'='*60}")
    print(f"病人: {patient_dir.name}")
    print(f"{'='*60}")

    if not input_nii.exists():
        print(f"  ⚠ 跳过: 未找到 {input_nii}")
        print(f"  用时: {format_duration(time.perf_counter() - patient_start)}")
        return "missing_orig"

    # 创建输出目录
    seg_dir.mkdir(exist_ok=True)
    ts_output.mkdir(exist_ok=True)

    stls_to_generate = missing_stl_names(seg_dir, overwrite)
    if not stls_to_generate:
        print("  ✓ 已有全部目标 STL，跳过")
        print(f"  用时: {format_duration(time.perf_counter() - patient_start)}")
        return "skipped"

    # Step 1: 运行 TotalSegmentator
    if overwrite or not masks_exist_for_stls(ts_output, stls_to_generate):
        if not run_totalsegmentator(input_nii, ts_output, device, fast):
            print(f"  用时: {format_duration(time.perf_counter() - patient_start)}")
            return "failed"
    else:
        print("  ✓ 复用已有 TotalSegmentator mask，跳过分割")

    # Step 2: 合并 mask 并转换为 STL
    for stl_name, class_names in STL_GROUPS.items():
        stl_path = seg_dir / stl_name
        if not overwrite and stl_is_done(stl_path):
            print(f"  ✓ 已存在 {stl_name}，跳过")
            continue

        print(f"  ▸ 生成 {stl_name} (合并 {len(class_names)} 个类) ...", end=" ")

        merged_mask, affine = merge_masks(ts_output, class_names)

        if merged_mask is None:
            print("⚠ 未找到对应分割文件，跳过")
            continue

        success = nifti_mask_to_stl(merged_mask, affine, stl_path,
                                     step_size=step_size)
        if success:
            size_mb = stl_path.stat().st_size / 1024 / 1024
            print(f"✓ ({size_mb:.1f} MB)")
        else:
            print("⚠ mask 为空，跳过")

    print(f"  ✓ 完成！输出目录: {seg_dir}")
    print(f"  用时: {format_duration(time.perf_counter() - patient_start)}")
    return "done"


def main():
    parser = argparse.ArgumentParser(
        description="批量CT分割 → STL导出 (基于TotalSegmentator)")
    parser.add_argument("--input_dir", type=str, default=r"F:\PCG data\dataset\test4all_sample",
                        help="包含病人文件夹的根目录")
    parser.add_argument("--device", type=str, default="gpu",
                        choices=["gpu", "cpu"],
                        help="运算设备 (默认: gpu)")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式: 使用3mm低分辨率模型")
    parser.add_argument("--step-size", type=int, default=1,
                        help="Marching Cubes 步长 (越大越快但越粗糙，默认: 1)")
    parser.add_argument("--patient", default=None,
                        help="只处理一个病人文件夹名，和 totalseg.py 一致")
    parser.add_argument("--patients", nargs="*", default=None,
                        help="只处理指定的病人文件夹名 (默认: 全部)")
    parser.add_argument("--skip-patients", nargs="*", default=[],
                        help="跳过指定的病人文件夹名")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--resume", dest="overwrite", action="store_false",
                            help="断点续跑: 已有目标 STL 就跳过 (默认)")
    mode_group.add_argument("--overwrite", dest="overwrite", action="store_true",
                            help="覆盖重跑: 重新分割并覆盖已有 STL")
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.is_dir():
        print(f"错误: 目录不存在 → {root}")
        sys.exit(1)

    patient_dirs = discover_patient_dirs(root)
    if args.patient:
        patient_dirs = [p for p in patient_dirs if p.name == args.patient]
    if args.patients:
        selected = set(args.patients)
        patient_dirs = [p for p in patient_dirs if p.name in selected]
    if args.skip_patients:
        skipped = set(args.skip_patients)
        patient_dirs = [p for p in patient_dirs if p.name not in skipped]

    if not patient_dirs:
        print(f"错误: 在 {root} 下未找到包含 orig.nii.gz 的病人文件夹")
        sys.exit(1)

    print(f"找到 {len(patient_dirs)} 个病人文件夹")
    print(f"设备: {args.device} | 快速模式: {args.fast} | MC步长: {args.step_size}")
    print(f"运行模式: {'覆盖重跑' if args.overwrite else '断点续跑'}")
    if args.skip_patients:
        print(f"跳过病人: {', '.join(args.skip_patients)}")

    total_start = time.perf_counter()
    status_counts: dict[str, int] = {}
    for patient_dir in iter_with_progress(patient_dirs):
        status = process_patient(patient_dir, args.device, args.fast,
                                 args.step_size, args.overwrite)
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"\n{'='*60}")
    print(f"全部完成! 共处理 {len(patient_dirs)} 个病人")
    print(f"总用时: {format_duration(time.perf_counter() - total_start)}")
    print("结果统计: " + ", ".join(
        f"{name}={count}" for name, count in sorted(status_counts.items())
    ))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
