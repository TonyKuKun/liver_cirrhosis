"""TotalSegmentator 集成模块。

输入：patient/orig.nii.gz（已从 DICOM 转好的 NIfTI）
输出：patient/segmentation/{structure}.nii.gz + {structure}.stl

特性：
- 用 orig.nii.gz 直接输入，不重复读 DICOM
- STL 做 Laplacian 平滑，输出干净的网格
- 可选提取哪些结构（--structures 参数）
- 已存在的结构自动跳过（--force 强制重新提取）

用法：
    # 提取全部
    python totalseg_integration.py --data_root /data

    # 只提取骨骼和门静脉
    python totalseg_integration.py --data_root /data --structures bone_all portal_vein

    # 强制重新提取
    python totalseg_integration.py --data_root /data --force
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

try:
    from skimage.measure import marching_cubes
except ImportError:
    try:
        from skimage.measure import marching_cubes_lewiner as marching_cubes
    except ImportError:
        marching_cubes = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# TotalSegmentator label 映射 (v2)
# ---------------------------------------------------------------------------

BONE_LABELS = {
    "vertebrae_L5": 1, "vertebrae_L4": 2, "vertebrae_L3": 3, "vertebrae_L2": 4, "vertebrae_L1": 5,
    "vertebrae_T12": 6, "vertebrae_T11": 7, "vertebrae_T10": 8, "vertebrae_T9": 9, "vertebrae_T8": 10,
    "vertebrae_T7": 11, "vertebrae_T6": 12, "vertebrae_T5": 13, "vertebrae_T4": 14, "vertebrae_T3": 15,
    "vertebrae_T2": 16, "vertebrae_T1": 17,
    "vertebrae_C7": None, "vertebrae_C6": None, "vertebrae_C5": None, "vertebrae_C4": None,
    "vertebrae_C3": None, "vertebrae_C2": None, "vertebrae_C1": None,
    "rib_left_1": 18, "rib_left_2": 19, "rib_left_3": 20, "rib_left_4": 21, "rib_left_5": 22,
    "rib_left_6": 23, "rib_left_7": 24, "rib_left_8": 25, "rib_left_9": 26, "rib_left_10": 27,
    "rib_left_11": 28, "rib_left_12": 29,
    "rib_right_1": 30, "rib_right_2": 31, "rib_right_3": 32, "rib_right_4": 33, "rib_right_5": 34,
    "rib_right_6": 35, "rib_right_7": 36, "rib_right_8": 37, "rib_right_9": 38, "rib_right_10": 39,
    "rib_right_11": 40, "rib_right_12": 41,
    "hip_left": 42, "hip_right": 43, "sacrum": 44,
    "femur_left": None, "femur_right": None,
    "scapula_left": None, "scapula_right": None,
    "clavicula_left": None, "clavicula_right": None,
    "humerus_left": None, "humerus_right": None,
    "sternum": None,
}

ORGAN_LABELS = {
    "spleen": 45,
    "kidney_right": 46,
    "kidney_left": 47,
    "liver": 48,
    "aorta": 52,
    "inferior_vena_cava": 53,
    "portal_vein_and_splenic_vein": 54,
}

# 我们支持的结构名 → 提取方式
TS_ROI_SUBSET = [
    *BONE_LABELS.keys(),
    "spleen",
    "liver",
    "kidney_left",
    "kidney_right",
    "inferior_vena_cava",
    "aorta",
    "portal_vein_and_splenic_vein",
]

ALL_STRUCTURES = [
    "bone_all",
    "spleen",
    "liver",
    "liver_left",
    "liver_right",
    "kidney_left",
    "kidney_right",
    "inferior_vena_cava",
    "aorta",
    "portal_vein",
]

# 用于 pretrain 排除的结构
EXCLUSION_STRUCTURES = [
    "bone_all", "spleen", "liver", "kidney_left", "kidney_right",
    "inferior_vena_cava", "aorta",
]


# =========================================================================
# 主入口
# =========================================================================

def run_segmentation(
    case,
    structures: list[str] | None = None,
    force: bool = False,
    device: str = "gpu",
    fast: bool = True,
    smooth_iterations: int = 15,
    smooth_relaxation: float = 0.3,
) -> dict:
    """为一个患者运行 TotalSegmentator 并提取指定结构。

    参数：
        case: 患者 case 对象，需有 case.path
        structures: 要提取的结构列表，None=全部
        force: True=强制重新提取所有，False=已存在的跳过
        device: "gpu" 或 "cpu"
        fast: TotalSegmentator 快速模式
        smooth_iterations: STL 平滑迭代次数
        smooth_relaxation: STL 平滑松弛因子
    """
    seg_dir = case.path / "segmentation"
    meta_path = seg_dir / "segmentation_meta.json"
    seg_dir.mkdir(parents=True, exist_ok=True)

    if structures is None:
        structures = list(ALL_STRUCTURES)
    else:
        structures = [s for s in structures if s in ALL_STRUCTURES]

    # 检查哪些结构需要提取（跳过已存在的）
    if not force:
        needed = []
        for name in structures:
            if _structure_done(seg_dir, name):
                continue  # 跳过
            needed.append(name)
    else:
        needed = list(structures)

    if not needed:
        print(f"  all {len(structures)} structures exist, skipping")
        return {
            "status": "complete",
            "structures": {name: {"status": "existed"} for name in structures},
            "requested": structures,
            "extracted": [],
            "skipped": structures,
        }

    # === Step 1: 运行 TotalSegmentator（如果还没跑过）===
    ts_output = seg_dir / "totalseg_output"
    batch_ts_output = seg_dir / "ts_raw"
    if not force:
        if _has_required_totalseg_output(ts_output, needed):
            pass
        elif _has_required_totalseg_output(batch_ts_output, needed):
            ts_output = batch_ts_output
    combined_seg = _find_totalseg_output(ts_output)

    if not _has_required_totalseg_output(ts_output, needed) or force:
        orig_nii = case.path / "orig.nii.gz"
        if not orig_nii.exists():
            return {"status": "error", "error": f"orig.nii.gz not found at {orig_nii}"}

        ts_output = seg_dir / "totalseg_output"
        ts_output.mkdir(parents=True, exist_ok=True)
        success, ts_info = _run_totalsegmentator(orig_nii, ts_output, device, fast)
        if not success:
            info = {"status": "failed", "totalsegmentator": ts_info}
            meta_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            return info
        combined_seg = _find_totalseg_output(ts_output)

    # === Step 2: 加载分割结果 ===
    seg_data, seg_affine = None, None
    if combined_seg is not None:
        try:
            seg_data, seg_affine, _ = _load_nifti(combined_seg)
        except Exception as e:
            return {"status": "error", "error": f"Failed to load {combined_seg}: {e}"}

    # === Step 3: 逐结构提取 ===
    structure_info = {}
    # 先加载已有 meta 里的信息
    existing_meta = _load_existing_meta(meta_path)
    if isinstance(existing_meta.get("structures"), dict):
        structure_info.update(existing_meta["structures"])

    for name in structures:
        nii_out = seg_dir / f"{name}.nii.gz"
        stl_out = seg_dir / f"{name}.stl"

        # 跳过已存在的
        if not force and _structure_done(seg_dir, name):
            previous = structure_info.get(name, {})
            structure_info[name] = {"status": "existed", **previous}
            structure_info[name]["status"] = "existed"
            continue

        try:
            mask, affine = _extract_structure(name, ts_output, seg_data, seg_affine)
            if mask is None or mask.sum() == 0:
                structure_info[name] = {"status": "empty"}
                continue

            # 保存 nii.gz
            _save_nifti(mask.astype(np.uint8), affine, nii_out)

            # 保存平滑后的 stl
            _mask_to_smooth_stl(
                mask, affine, stl_out,
                smooth_iterations=smooth_iterations,
                smooth_relaxation=smooth_relaxation,
            )

            stl_size = stl_out.stat().st_size if stl_out.exists() else 0
            structure_info[name] = {
                "status": "ok",
                "voxels": int(mask.sum()),
                "nii": str(nii_out),
                "stl": str(stl_out),
                "stl_bytes": stl_size,
            }
            if name == "bone_all":
                structure_info[name]["source_bones"] = list(BONE_LABELS)
            print(f"    {name}: {int(mask.sum())} voxels, stl {stl_size // 1024}KB")

        except Exception as e:
            structure_info[name] = {"status": "error", "error": str(e)}
            print(f"    {name}: ERROR {e}")

    info = {
        "status": "complete",
        "structures": structure_info,
        "requested": structures,
        "extracted": [n for n in structures if structure_info.get(n, {}).get("status") == "ok"],
        "skipped": [n for n in structures if structure_info.get(n, {}).get("status") == "existed"],
    }
    meta_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


def _load_existing_meta(meta_path: Path) -> dict:
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _structure_stl_done(seg_dir: Path, name: str) -> bool:
    stl = seg_dir / f"{name}.stl"
    return stl.exists() and stl.stat().st_size > 0


def _structure_nii_done(seg_dir: Path, name: str) -> bool:
    nii = seg_dir / f"{name}.nii.gz"
    return nii.exists() and nii.stat().st_size > 0


def _structure_done(seg_dir: Path, name: str) -> bool:
    if not (_structure_nii_done(seg_dir, name) and _structure_stl_done(seg_dir, name)):
        return False
    if name == "bone_all":
        return _bone_source_files_done(seg_dir) and _bone_meta_done(seg_dir)
    return True


def _bone_meta_done(seg_dir: Path) -> bool:
    meta = _load_existing_meta(seg_dir / "segmentation_meta.json")
    info = meta.get("structures", {}).get("bone_all", {})
    source_bones = info.get("source_bones", [])
    return set(source_bones) >= set(BONE_LABELS)


def _bone_source_files_done(seg_dir: Path) -> bool:
    for ts_output in (seg_dir / "totalseg_output", seg_dir / "ts_raw"):
        if all(
            (ts_output / f"{bone_name}.nii.gz").exists()
            and (ts_output / f"{bone_name}.nii.gz").stat().st_size > 0
            for bone_name in BONE_LABELS
        ):
            return True
    return False


def _find_totalseg_output(ts_output: Path) -> Path | None:
    """找到 TotalSegmentator 的输出文件。"""
    for name in ("segmentations.nii.gz", "ct.nii.gz"):
        p = ts_output / name
        if p.exists():
            return p
    # --ml 模式输出每个结构一个文件，检查是否有
    if (ts_output / "spleen.nii.gz").exists():
        return None  # 标记为逐文件模式
    return None


# =========================================================================
# TotalSegmentator 调用
# =========================================================================

def _has_totalseg_output(ts_output: Path) -> bool:
    if not ts_output.exists():
        return False
    if _find_totalseg_output(ts_output) is not None:
        return True
    return any(ts_output.glob("*.nii.gz"))


def _required_ts_files(structures: list[str]) -> list[str]:
    files: list[str] = []
    for name in structures:
        if name == "bone_all":
            files.extend(f"{bone_name}.nii.gz" for bone_name in BONE_LABELS)
        elif name in {"liver_left", "liver_right"}:
            files.append("liver.nii.gz")
        elif name == "portal_vein":
            files.append("portal_vein_and_splenic_vein.nii.gz")
        else:
            files.append(f"{name}.nii.gz")
    return sorted(set(files))


def _has_required_totalseg_output(ts_output: Path, structures: list[str]) -> bool:
    if not ts_output.exists():
        return False
    if _find_totalseg_output(ts_output) is not None:
        return True
    return all((ts_output / filename).exists() for filename in _required_ts_files(structures))


def _run_totalsegmentator(
    input_nii: Path,
    output_dir: Path,
    device: str = "gpu",
    fast: bool = True,
) -> tuple[bool, dict]:
    """调用 TotalSegmentator。"""
    info: dict = {}
    cmd = [
        "TotalSegmentator",
        "-i", str(input_nii),
        "-o", str(output_dir),
        "--roi_subset", *TS_ROI_SUBSET,
        "--device", device,
        "--nr_thr_resamp", "1",
        "--nr_thr_saving", "1",
    ]
    if fast:
        cmd.append("--fast")
    info["command"] = " ".join(cmd)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
        info["returncode"] = result.returncode
        if result.returncode != 0:
            info["stderr"] = (result.stderr or "")[:2000]
            return False, info
        return True, info
    except FileNotFoundError:
        info["error"] = "totalsegmentator not installed. pip install TotalSegmentator"
        return False, info
    except subprocess.TimeoutExpired:
        info["error"] = "timeout (600s)"
        return False, info
    except Exception as e:
        info["error"] = str(e)
        return False, info


# =========================================================================
# 结构提取
# =========================================================================

def _extract_structure(
    name: str,
    ts_output: Path,
    combined_data: np.ndarray | None,
    combined_affine: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """从 TotalSegmentator 输出中提取单个结构。"""

    # 优先尝试独立文件（--ml 有时也生成逐文件）
    # TotalSegmentator 的文件命名可能和我们的不一样，做映射
    ts_name_map = {
        "bone_all": None,  # 合并标签
        "portal_vein": "portal_vein_and_splenic_vein",
        "liver_left": None,   # 需要后处理
        "liver_right": None,
    }
    ts_filename = ts_name_map.get(name, name)

    if ts_filename is not None:
        individual = ts_output / f"{ts_filename}.nii.gz"
        if individual.exists():
            data, affine, _ = _load_nifti(individual)
            return (data > 0), affine

    if name == "bone_all":
        mask, affine = None, None
        for bone_name in BONE_LABELS:
            individual = ts_output / f"{bone_name}.nii.gz"
            if not individual.exists():
                continue
            data, affine, _ = _load_nifti(individual)
            part = data > 0
            mask = part if mask is None else (mask | part)
        if mask is not None:
            return mask, affine

    if name in {"liver_left", "liver_right"}:
        individual = ts_output / "liver.nii.gz"
        if individual.exists():
            data, affine, _ = _load_nifti(individual)
            liver = data > 0
            if not liver.any():
                return None, affine
            side = "left" if name == "liver_left" else "right"
            return _split_liver(liver, affine, side), affine

    # 从合并标签图提取
    if combined_data is None:
        return None, None

    if name == "bone_all":
        mask = np.zeros(combined_data.shape, dtype=bool)
        for label_id in BONE_LABELS.values():
            if label_id is None:
                continue
            mask |= (combined_data == label_id)
        return mask, combined_affine

    elif name == "portal_vein":
        lid = ORGAN_LABELS.get("portal_vein_and_splenic_vein")
        if lid is not None:
            return (combined_data == lid), combined_affine
        return None, combined_affine

    elif name == "liver_left":
        liver = (combined_data == ORGAN_LABELS.get("liver", -1))
        if not liver.any():
            return None, combined_affine
        return _split_liver(liver, combined_affine, "left"), combined_affine

    elif name == "liver_right":
        liver = (combined_data == ORGAN_LABELS.get("liver", -1))
        if not liver.any():
            return None, combined_affine
        return _split_liver(liver, combined_affine, "right"), combined_affine

    else:
        lid = ORGAN_LABELS.get(name)
        if lid is not None:
            return (combined_data == lid), combined_affine
        return None, combined_affine


def _split_liver(liver_mask: np.ndarray, affine: np.ndarray, side: str) -> np.ndarray:
    """沿肝脏 x 方向中位数分左右叶。"""
    coords = np.argwhere(liver_mask)
    if len(coords) == 0:
        return np.zeros_like(liver_mask)
    x_mid = int(np.median(coords[:, 2]))
    result = liver_mask.copy()
    if side == "left":
        result[:, :, :x_mid] = False
    else:
        result[:, :, x_mid:] = False
    return result


# =========================================================================
# NIfTI 读写
# =========================================================================

def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    import nibabel as nib
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img.affine.copy(), {"shape": list(img.shape)}


def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path):
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


# =========================================================================
# STL 生成 + 平滑
# =========================================================================

def _mask_to_smooth_stl(
    mask: np.ndarray,
    affine: np.ndarray,
    out_path: Path,
    step_size: int = 1,
    smooth_iterations: int = 15,
    smooth_relaxation: float = 0.3,
):
    """mask → marching cubes → Laplacian 平滑 → 世界坐标 STL。"""
    if marching_cubes is None:
        raise ImportError("skimage.measure.marching_cubes not available")

    if mask.sum() == 0:
        _write_stl(out_path, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))
        return

    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, step_size=step_size)

    # Laplacian 平滑
    if smooth_iterations > 0:
        verts = _laplacian_smooth(verts, faces, smooth_iterations, smooth_relaxation)

    # 体素坐标 → 世界坐标
    ones = np.ones((len(verts), 1), dtype=np.float64)
    verts_world = (affine @ np.hstack([verts, ones]).T).T[:, :3]

    _write_stl(out_path, verts_world.astype(np.float32), faces)


def _laplacian_smooth(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 15,
    relaxation: float = 0.3,
) -> np.ndarray:
    """Laplacian 网格平滑。

    对每个顶点，将其位置向邻居的平均位置移动 relaxation 比例。
    重复 iterations 次。
    """
    n_verts = len(verts)
    verts = verts.copy().astype(np.float64)

    # 构建邻接表
    neighbors: list[list[int]] = [[] for _ in range(n_verts)]
    for f in faces:
        v0, v1, v2 = int(f[0]), int(f[1]), int(f[2])
        neighbors[v0].append(v1)
        neighbors[v0].append(v2)
        neighbors[v1].append(v0)
        neighbors[v1].append(v2)
        neighbors[v2].append(v0)
        neighbors[v2].append(v1)
    # 去重
    neighbors = [list(set(nb)) for nb in neighbors]

    for _ in range(iterations):
        new_verts = verts.copy()
        for i in range(n_verts):
            nb = neighbors[i]
            if not nb:
                continue
            avg = verts[nb].mean(axis=0)
            new_verts[i] = verts[i] + relaxation * (avg - verts[i])
        verts = new_verts

    return verts.astype(np.float32)


def _write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray):
    """写二进制 STL。"""
    import struct as st
    path.parent.mkdir(parents=True, exist_ok=True)
    n_tri = len(faces)
    with path.open("wb") as f:
        f.write(b"\x00" * 80)
        f.write(st.pack("<I", n_tri))
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len
            f.write(st.pack("<3f", *normal))
            f.write(st.pack("<3f", *v0))
            f.write(st.pack("<3f", *v1))
            f.write(st.pack("<3f", *v2))
            f.write(st.pack("<H", 0))


# =========================================================================
# pretrain 集成接口
# =========================================================================

def load_organ_mask(case, organ_name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """加载指定器官的 mask。返回 (mask_bool, affine)。"""
    seg_dir = case.path / "segmentation"
    aliases = (organ_name,)
    if organ_name == "portal_vein":
        aliases = ("portal_vein", "portal_vein_and_splenic_vein")
    for directory in (seg_dir / "totalseg_output", seg_dir / "ts_raw", seg_dir):
        for alias in aliases:
            nii_path = directory / f"{alias}.nii.gz"
            if not nii_path.exists():
                continue
            try:
                data, affine, _ = _load_nifti(nii_path)
                return data > 0, affine
            except Exception:
                continue
    return None, None


def get_exclusion_mask(
    case,
    vol_shape: tuple[int, int, int],
    dilate_bone: int = 3,
    dilate_organ: int = 2,
) -> tuple[np.ndarray, dict]:
    """骨骼+脾+肾+IVC+主动脉的组合排除 mask。"""
    info: dict = {"loaded": []}
    exclusion = np.zeros(vol_shape, dtype=bool)
    portal_protect = None

    for struct_name in EXCLUSION_STRUCTURES:
        mask, affine = load_organ_mask(case, struct_name)
        if mask is None:
            continue
        if mask.shape != vol_shape:
            mask = _resample_mask(mask, vol_shape)
            if mask is None:
                continue
        dilate = dilate_bone if "bone" in struct_name else dilate_organ
        if ndi is not None and dilate > 0:
            mask = ndi.binary_dilation(mask, iterations=dilate)
        if struct_name == "liver":
            if portal_protect is None:
                portal_protect, _ = load_organ_mask(case, "portal_vein")
                if portal_protect is not None and portal_protect.shape != vol_shape:
                    portal_protect = _resample_mask(portal_protect, vol_shape)
                if portal_protect is not None and ndi is not None and dilate_organ > 0:
                    portal_protect = ndi.binary_dilation(portal_protect, iterations=dilate_organ)
                info["portal_protection"] = {
                    "status": "ok" if portal_protect is not None and portal_protect.any() else "missing",
                    "voxels": int(portal_protect.sum()) if portal_protect is not None else 0,
                }
            if portal_protect is not None:
                mask = mask & ~portal_protect
        exclusion |= mask
        info["loaded"].append(struct_name)

    info["total_excluded_voxels"] = int(exclusion.sum())
    return exclusion, info


def get_portal_seed(
    case,
    spacing_zyx: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> tuple[tuple[float, float, float] | None, dict]:
    """从门静脉分割结果获取 seed 点。"""
    info: dict = {}
    mask, affine = load_organ_mask(case, "portal_vein")
    if mask is None or mask.sum() == 0:
        return None, {"status": "no_portal_vein_mask"}

    coords = np.argwhere(mask)
    centroid_ijk = coords.mean(axis=0)
    centroid_world = (affine @ np.append(centroid_ijk, 1.0))[:3]

    sp = np.asarray(spacing_zyx, dtype=np.float64)
    og = np.asarray(origin_xyz, dtype=np.float64)
    seed = (
        float((centroid_world[2] - og[2]) / sp[0]),
        float((centroid_world[1] - og[1]) / sp[1]),
        float((centroid_world[0] - og[0]) / sp[2]),
    )
    info.update({"status": "ok", "seed_zyx": [round(s, 1) for s in seed],
                 "portal_voxels": int(mask.sum())})
    return seed, info


def get_portal_vein_mask(case, vol_shape):
    """门静脉 mask 用于区域生长。"""
    mask, _ = load_organ_mask(case, "portal_vein")
    if mask is None:
        return None
    if mask.shape != vol_shape:
        mask = _resample_mask(mask, vol_shape)
    return mask


def get_liver_mask(case, vol_shape):
    """肝脏 mask。"""
    mask, _ = load_organ_mask(case, "liver")
    if mask is None:
        return None
    if mask.shape != vol_shape:
        mask = _resample_mask(mask, vol_shape)
    return mask


def get_z_range_from_bone(case, vol_shape, spacing_zyx, margin_mm=20.0):
    """用 bone mask 做 Z 轴标定。"""
    bone, _ = load_organ_mask(case, "bone_all")
    if bone is None:
        return None, None, {"status": "no_bone"}
    if bone.shape != vol_shape:
        bone = _resample_mask(bone, vol_shape)
        if bone is None:
            return None, None, {"status": "resample_failed"}

    nz = vol_shape[0]
    dz = float(spacing_zyx[0])
    margin = max(1, int(round(margin_mm / max(0.1, dz))))

    z_has = np.any(np.any(bone, axis=2), axis=1)
    if not z_has.any():
        return None, None, {"status": "empty_bone"}

    bone_z = np.where(z_has)[0]
    z_start = max(0, int(bone_z.min()) - margin)

    # 下界：找髂骨（骨质横向跨度突然增大的层）
    z_end = min(nz - 1, int(bone_z.max()) + margin)
    for z in range(int(bone_z.max()), int(bone_z.min()), -1):
        if not bone[z].any():
            continue
        x_cols = np.where(np.any(bone[z], axis=0))[0]
        width_mm = float((x_cols[-1] - x_cols[0]) * spacing_zyx[2])
        if width_mm > 90.0:
            z_end = max(0, z - margin)
            break

    return z_start, z_end, {"status": "ok", "z_start": z_start, "z_end": z_end,
                             "z_range_mm": round(float((z_end - z_start) * dz), 1)}


def _resample_mask(mask, target_shape):
    if ndi is None:
        return None
    zoom = np.asarray(target_shape, dtype=np.float64) / np.asarray(mask.shape, dtype=np.float64)
    try:
        return ndi.zoom(mask.astype(np.float32), zoom, order=0) > 0.5
    except Exception:
        return None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def _progress_line(done: int, total: int, start_time: float) -> str:
    width = 28
    ratio = done / total if total else 1.0
    filled = min(width, int(round(width * ratio)))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - start_time
    eta = elapsed * (total - done) / done if done else 0
    return (
        f"[totalseg] progress {done}/{total} "
        f"[{bar}] {ratio * 100:5.1f}% "
        f"elapsed {_format_duration(elapsed)} eta {_format_duration(eta)}"
    )


def discover_orig_patients(root: str | Path) -> list[SimpleNamespace]:
    """Find every patient folder with orig.nii.gz."""
    root = Path(root)
    if not root.exists():
        return []
    if (root / "orig.nii.gz").is_file():
        patient_dirs = [root]
    else:
        patient_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    return [
        SimpleNamespace(name=path.name, path=path)
        for path in patient_dirs
        if (path / "orig.nii.gz").is_file()
    ]


# =========================================================================
# CLI
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator and extract organ masks (nii.gz + smoothed STL).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample", help="Root data directory")
    parser.add_argument("--patient", default=None, help="Process one patient only, default=None")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume",
        dest="overwrite",
        action="store_false",
        default=False,
        help="Resume interrupted runs: skip structures whose STL already exists (default)",
    )
    run_mode.add_argument(
        "--overwrite",
        "--force",
        dest="overwrite",
        default=False,
        action="store_true",
        help="Overwrite and re-extract structures even if STL files already exist",
    )
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--fast", action="store_true", default=False,
                        help="TotalSegmentator fast mode (3mm resolution)")
    parser.add_argument("--no-fast", dest="fast", action="store_false")
    parser.add_argument(
        "--structures", nargs="+", default=None,
        help="Which structures to extract. Default: all.\n"
             f"Available: {', '.join(ALL_STRUCTURES)}",
    )
    parser.add_argument("--smooth-iterations", type=int, default=15,
                        help="STL Laplacian smoothing iterations (default: 15)")
    parser.add_argument("--smooth-relaxation", type=float, default=0.3,
                        help="STL smoothing relaxation factor (default: 0.3)")
    args = parser.parse_args()

    cases = discover_orig_patients(args.data_root)
    if args.patient:
        cases = [c for c in cases if c.name == args.patient]

    structs = args.structures
    if structs:
        invalid = [s for s in structs if s not in ALL_STRUCTURES]
        if invalid:
            print(f"Unknown structures: {invalid}")
            print(f"Available: {ALL_STRUCTURES}")
            sys.exit(1)

    total_start = time.perf_counter()
    total_cases = len(cases)
    mode = "overwrite" if args.overwrite else "resume"
    print(f"[totalseg] {total_cases} patients, structures: {structs or 'all'}, mode: {mode}")
    print(_progress_line(0, total_cases, total_start))
    for idx, case in enumerate(cases, start=1):
        case_start = time.perf_counter()
        print(f"[totalseg] {idx}/{total_cases} {case.name}:")
        try:
            info = run_segmentation(
                case,
                structures=structs,
                force=args.overwrite,
                device=args.device,
                fast=args.fast,
                smooth_iterations=args.smooth_iterations,
                smooth_relaxation=args.smooth_relaxation,
            )
            case_elapsed = time.perf_counter() - case_start
            if info.get("status") in {"error", "failed"}:
                print(f"  FAILED in {_format_duration(case_elapsed)}: {info}")
                print(_progress_line(idx, total_cases, total_start))
                continue
            extracted = info.get("extracted", [])
            skipped = info.get("skipped", [])
            print(
                f"  done: {len(extracted)} extracted, {len(skipped)} skipped "
                f"in {_format_duration(case_elapsed)}"
            )
        except Exception as e:
            case_elapsed = time.perf_counter() - case_start
            print(f"  FAILED in {_format_duration(case_elapsed)}: {e}")
        print(_progress_line(idx, total_cases, total_start))
    print(f"[totalseg] all done in {_format_duration(time.perf_counter() - total_start)}")


if __name__ == "__main__":
    main()
