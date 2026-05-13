"""pretrain v4 — 完整的门静脉预分割流程。

依赖模块：
    vertebra_detector  — 椎体逐节检测 + L3 定位 + Z 轴标定
    threshold_scorer   — 解剖结构感知的阈值评分 + 搜索

流程：
    1. DICOM 加载
    2. 脊柱检测（硬排除）
    3. 椎体分节 → 膈肌 + L3 上缘 → Z 轴标定
    4. CT 预览图（冠状位/矢状位 MIP + 肝门区关键层）
    5. LLM 初始规划（喂入检测到的解剖信息）
    6. 阈值搜索（完整性 + 可分离性 + 连通性评分）
    7. 分割 + 区域生长
    8. STL 渲染 → LLM 迭代精修（最多 2 轮）
    9. 输出 pretrain.stl + meta
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

try:
    from ..utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, stl_to_voxels
except ImportError:
    try:
        from VKAN_segementation.utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, stl_to_voxels
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import DicomVolume, GemmaClient, discover_patients, mask_to_stl, stl_to_voxels

# 本地模块
try:
    from .vertebra_detector import standardize_z_range, detect_vertebrae
    from .threshold_scorer import search_best_threshold, score_threshold
except ImportError:
    try:
        from vertebra_detector import standardize_z_range, detect_vertebrae
        from threshold_scorer import search_best_threshold, score_threshold
    except ImportError:
        standardize_z_range = None  # type: ignore[assignment]
        detect_vertebrae = None  # type: ignore[assignment]
        search_best_threshold = None  # type: ignore[assignment]
        score_threshold = None  # type: ignore[assignment]


PRETRAIN_ALGORITHM_VERSION = "2026-05-14-v4-anatomy-aware"
PRETRAIN_META_NAME = "pretrain_meta.json"
MAX_STL_BYTES = 20_000 * 1024
TARGET_VOXELS = 420_000
TARGET_VOXELS_TIPS = 330_000

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CORTICAL_BONE_HU = 400.0
CANCELLOUS_BONE_HU = 250.0
SPINE_DILATE_ITERATIONS = 4
REGION_GROW_BRIDGE_MM = 8.0
REGION_GROW_MAX_SEED_SNAP_MM = 25.0
MAX_REFINE_ROUNDS = 2


@dataclass(frozen=True)
class PretrainResult:
    path: Path
    status: str


# =========================================================================
# DICOM 加载
# =========================================================================

def load_dicom_series(dcm_dir: str | Path) -> DicomVolume:
    try:
        import pydicom
    except ImportError:
        return _load_dicom_series_minimal(dcm_dir)
    files = [p for p in Path(dcm_dir).rglob("*") if p.is_file()]
    slices = []
    for file in files:
        try:
            ds = pydicom.dcmread(str(file), force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        raise FileNotFoundError(f"No readable DICOM slices found in {dcm_dir}")
    slices.sort(key=_slice_position)
    arrays = []
    for ds in slices:
        try:
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            arr = _raw_pixel_array(ds).astype(np.float32)
        arrays.append(arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0)))
    volume = np.stack(arrays, axis=0)
    ps = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    sy, sx = float(ps[0]), float(ps[1])
    dz = abs(_slice_position(slices[1]) - _slice_position(slices[0])) if len(slices) > 1 else 0.0
    if dz <= 0:
        dz = float(getattr(slices[0], "SliceThickness", 1.0))
    ipp = getattr(slices[0], "ImagePositionPatient", [0.0, 0.0, 0.0])
    return DicomVolume(volume_hu=volume, spacing_zyx=(dz, sy, sx),
                       origin_xyz=(float(ipp[0]), float(ipp[1]), float(ipp[2])))


def _load_dicom_series_minimal(dcm_dir: str | Path) -> DicomVolume:
    slices = []
    for file in Path(dcm_dir).rglob("*"):
        if not file.is_file():
            continue
        try:
            meta = _read_minimal_dicom(file)
        except Exception:
            continue
        if meta and meta.get("pixel_data") is not None:
            slices.append(meta)
    if not slices:
        raise FileNotFoundError(f"No readable uncompressed DICOM slices found in {dcm_dir}")
    slices.sort(key=_minimal_slice_position)
    arrays = []
    for ds in slices:
        arr = _raw_pixel_array_minimal(ds).astype(np.float32)
        arrays.append(arr * float(ds.get("rescale_slope", 1.0)) + float(ds.get("rescale_intercept", 0.0)))
    volume = np.stack(arrays, axis=0)
    sy, sx = [float(v) for v in slices[0].get("pixel_spacing", [1.0, 1.0])[:2]]
    dz = abs(_minimal_slice_position(slices[1]) - _minimal_slice_position(slices[0])) if len(slices) > 1 else 0.0
    if dz <= 0:
        dz = float(slices[0].get("slice_thickness", 1.0))
    ipp = [float(v) for v in slices[0].get("image_position_patient", [0.0, 0.0, 0.0])[:3]]
    return DicomVolume(volume_hu=volume, spacing_zyx=(dz, sy, sx), origin_xyz=(ipp[0], ipp[1], ipp[2]))


def _slice_position(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return float(ipp[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def _minimal_slice_position(ds: dict) -> float:
    ipp = ds.get("image_position_patient")
    if ipp and len(ipp) >= 3:
        return float(ipp[2])
    if ds.get("slice_location") is not None:
        return float(ds["slice_location"])
    return float(ds.get("instance_number", 0))


def _read_minimal_dicom(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    pos = 132 if len(data) > 132 and data[128:132] == b"DICM" else 0
    out: dict = {}
    while pos + 8 <= len(data):
        group, elem = struct.unpack_from("<HH", data, pos)
        pos += 4
        vr = data[pos:pos + 2].decode("ascii", errors="ignore")
        if vr in {"OB", "OD", "OF", "OL", "OV", "OW", "SQ", "UC", "UR", "UT", "UN"}:
            pos += 4
            length = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        elif vr and vr[0].isalpha() and vr[1].isalpha():
            pos += 2
            length = struct.unpack_from("<H", data, pos)[0]
            pos += 2
        else:
            length = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            vr = ""
        if length == 0xFFFFFFFF or pos + length > len(data):
            break
        value = data[pos:pos + length]
        pos += length + (length % 2)
        tag = (group, elem)
        if tag == (0x0028, 0x0010):
            out["rows"] = _decode_int(value)
        elif tag == (0x0028, 0x0011):
            out["columns"] = _decode_int(value)
        elif tag == (0x0028, 0x0103):
            out["pixel_representation"] = _decode_int(value)
        elif tag == (0x0028, 0x0030):
            out["pixel_spacing"] = _decode_numbers(value)
        elif tag == (0x0028, 0x1052):
            out["rescale_intercept"] = _decode_float(value, -1024.0)
        elif tag == (0x0028, 0x1053):
            out["rescale_slope"] = _decode_float(value, 1.0)
        elif tag == (0x0018, 0x0050):
            out["slice_thickness"] = _decode_float(value, 1.0)
        elif tag == (0x0020, 0x0032):
            out["image_position_patient"] = _decode_numbers(value)
        elif tag == (0x0020, 0x1041):
            out["slice_location"] = _decode_float(value, 0.0)
        elif tag == (0x0020, 0x0013):
            out["instance_number"] = _decode_int(value)
        elif tag == (0x7FE0, 0x0010):
            out["pixel_data"] = value
            break
    if "rows" not in out or "columns" not in out:
        return {}
    return out


def _decode_text(v: bytes) -> str:
    return v.rstrip(b"\x00 ").decode("ascii", errors="ignore")

def _decode_numbers(v: bytes) -> list[float]:
    return [float(p) for p in _decode_text(v).split("\\") if p]

def _decode_float(v: bytes, d: float) -> float:
    try:
        return float(_decode_text(v))
    except ValueError:
        return d

def _decode_int(v: bytes) -> int:
    if len(v) == 2:
        return int(struct.unpack("<H", v)[0])
    if len(v) == 4:
        return int(struct.unpack("<I", v)[0])
    t = _decode_text(v)
    return int(t) if t else 0

def _raw_pixel_array(ds) -> np.ndarray:
    dtype = "<i2" if int(getattr(ds, "PixelRepresentation", 0)) else "<u2"
    return np.frombuffer(ds.PixelData, dtype=dtype,
                         count=int(ds.Rows) * int(ds.Columns)).reshape(int(ds.Rows), int(ds.Columns))

def _raw_pixel_array_minimal(m: dict) -> np.ndarray:
    dtype = "<i2" if int(m.get("pixel_representation", 0)) else "<u2"
    return np.frombuffer(m["pixel_data"], dtype=dtype,
                         count=int(m["rows"]) * int(m["columns"])).reshape(int(m["rows"]), int(m["columns"]))


# =========================================================================
# 脊柱检测
# =========================================================================

def _detect_spine_mask(vol: np.ndarray) -> np.ndarray:
    """逐层检测脊柱区域并膨胀覆盖松质骨。"""
    if ndi is None:
        return np.zeros(vol.shape, dtype=bool)
    nz, ny, nx = vol.shape
    spine = np.zeros(vol.shape, dtype=bool)
    centers: list[tuple[float, float] | None] = []
    for z in range(nz):
        bone = vol[z] > CORTICAL_BONE_HU
        if bone.sum() < 20:
            centers.append(None)
            continue
        labels, n = ndi.label(bone)
        if n == 0:
            centers.append(None)
            continue
        best_lb, best_post, best_cy, best_cx = 0, -1.0, 0.0, 0.0
        for lb in range(1, n + 1):
            coords = np.argwhere(labels == lb)
            if len(coords) < 10:
                continue
            post = float(coords[:, 0].max())
            if post > best_post:
                best_post = post
                best_lb = lb
                best_cy = float(coords[:, 0].mean())
                best_cx = float(coords[:, 1].mean())
        if best_lb > 0:
            spine[z] = (labels == best_lb)
            centers.append((best_cy, best_cx))
        else:
            centers.append(None)
    valid = [c for c in centers if c is not None]
    if len(valid) > 3:
        mcx = float(np.median([cx for _, cx in valid]))
        mcy = float(np.median([cy for cy, _ in valid]))
        for z in range(nz):
            if centers[z] is None:
                continue
            cy, cx = centers[z]
            if abs(cx - mcx) > nx * 0.15 or abs(cy - mcy) > ny * 0.20:
                spine[z] = False
    spine = ndi.binary_dilation(spine, iterations=SPINE_DILATE_ITERATIONS)
    near = ndi.binary_dilation(spine, iterations=2)
    spine = spine | ((vol > CANCELLOUS_BONE_HU) & (vol <= CORTICAL_BONE_HU) & near)
    return spine


def _spine_center_yx(spine_mask: np.ndarray) -> tuple[float, float] | None:
    """返回脊柱中心 (y, x) 的归一化坐标。内存安全：逐层统计而非全量 argwhere。"""
    nz, ny, nx = spine_mask.shape
    sum_y, sum_x, total = 0.0, 0.0, 0
    # 每 4 层采样一次，足够精确且节省内存
    for z in range(0, nz, 4):
        ys, xs = np.where(spine_mask[z])
        if len(ys) > 0:
            sum_y += float(ys.sum())
            sum_x += float(xs.sum())
            total += len(ys)
    if total == 0:
        return None
    return (float(sum_y / total) / ny, float(sum_x / total) / nx)


# =========================================================================
# CT 预览图生成
# =========================================================================

def _save_planning_previews(vol: np.ndarray, work_dir: Path) -> tuple[list[Path], dict]:
    """生成 LLM 规划预览图。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    previews: list[Path] = []
    info: dict = {}

    ny, nx = vol.shape[1], vol.shape[2]

    # 1. 冠状位 MIP（y 25-60%，覆盖脾静脉）
    y0, y1 = max(0, int(ny * 0.25)), min(ny, int(ny * 0.60))
    if y1 > y0:
        mip = vol[:, y0:y1, :].max(axis=1)
        p = _save_array_png(mip, work_dir / "coronal_mip.png", (-50, 300))
        if p:
            previews.append(p)
            info["coronal_mip"] = f"y=[{y0},{y1}], window=[-50,300]"

    # 2. 矢状位 MIP（x 30-70%）
    x0, x1 = max(0, int(nx * 0.30)), min(nx, int(nx * 0.70))
    if x1 > x0:
        mip = vol[:, :, x0:x1].max(axis=2)
        p = _save_array_png(mip, work_dir / "sagittal_mip.png", (-50, 350))
        if p:
            previews.append(p)
            info["sagittal_mip"] = f"x=[{x0},{x1}], window=[-50,350]"

    # 3. 肝门区关键层
    key_z = _find_porta_hepatis_slices(vol, n=5)
    p = _save_key_slices(vol, work_dir / "axial_porta.png", key_z, (-50, 300))
    if p:
        previews.append(p)
        info["porta_hepatis_slices"] = key_z

    # 4. 骨窗（TIPS/骨骼定位）
    p = _save_mosaic(vol, work_dir / "bone_window.png", (200, 1000))
    if p:
        previews.append(p)

    # 5. 全范围 mosaic
    p = _save_mosaic(vol, work_dir / "ct_mosaic.png")
    if p:
        previews.append(p)

    return previews, info


def _find_porta_hepatis_slices(vol: np.ndarray, n: int = 5) -> list[int]:
    nz, ny = vol.shape[0], vol.shape[1]
    zs, ze = int(nz * 0.35), int(nz * 0.70)
    yc = int(ny * 0.70)
    scores = [(z, int(((vol[z, :yc, :] > 80) & (vol[z, :yc, :] < 350)).sum()))
              for z in range(zs, ze)]
    scores.sort(key=lambda x: x[1], reverse=True)
    sel: list[int] = []
    for z, _ in scores:
        if all(abs(z - s) > 3 for s in sel):
            sel.append(z)
        if len(sel) >= n:
            break
    return sorted(sel)


def _save_array_png(arr, path, window=(-50, 350), target_w=512):
    try:
        from PIL import Image
    except ImportError:
        return None
    lo, hi = window
    img = np.clip((arr - lo) / max(hi - lo, 1e-3), 0, 1)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    h, w = arr.shape
    if w > 0:
        pil = pil.resize((target_w, max(1, int(h * target_w / w))))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(path)
    return Path(path)


def _save_key_slices(vol, path, indices, window=(-50, 350)):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    lo, hi = window
    sz = 192
    imgs = []
    for idx in indices:
        idx = max(0, min(vol.shape[0] - 1, idx))
        img = np.clip((vol[idx] - lo) / max(hi - lo, 1e-3), 0, 1)
        imgs.append(Image.fromarray((img * 255).astype(np.uint8)).resize((sz, sz)))
    n = len(imgs)
    cols = min(n, 5)
    rows = max(1, (n + cols - 1) // cols)
    canvas = Image.new("L", (cols * sz, rows * sz), 0)
    draw = ImageDraw.Draw(canvas)
    for j, im in enumerate(imgs):
        x, y = (j % cols) * sz, (j // cols) * sz
        canvas.paste(im, (x, y))
        draw.text((x + 4, y + 4), f"z={indices[j]}", fill=255)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return Path(path)


def _save_mosaic(vol, path, window=None, n_slices=12):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    path = Path(path)
    if window is None:
        lo, hi = np.percentile(vol, [1, 99])
        lo, hi = min(lo, -100.0), max(hi, 350.0)
    else:
        lo, hi = window
    idx = np.linspace(max(0, vol.shape[0] * 0.15), max(0, vol.shape[0] * 0.85), n_slices)
    imgs = []
    for i in idx.astype(int):
        img = np.clip((vol[i] - lo) / max(hi - lo, 1e-3), 0, 1)
        imgs.append(Image.fromarray((img * 255).astype(np.uint8)).resize((192, 192)))
    canvas = Image.new("L", (4 * 192, int(np.ceil(len(imgs) / 4)) * 192), 0)
    draw = ImageDraw.Draw(canvas)
    for j, im in enumerate(imgs):
        x, y = (j % 4) * 192, (j // 4) * 192
        canvas.paste(im, (x, y))
        draw.text((x + 4, y + 4), str(int(idx[j])), fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


# =========================================================================
# LLM 初始规划 — 注入检测到的解剖信息
# =========================================================================

def _build_anatomy_context(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    z_info: dict,
    spine_info: dict,
    stats: dict,
) -> dict:
    """将所有已检测的解剖信息汇总为 LLM 可读的上下文。"""
    nz, ny, nx = vol.shape
    ctx: dict = {
        "volume_shape_zyx": [nz, ny, nx],
        "spacing_zyx_mm": [round(s, 3) for s in spacing_zyx],
        "hu_statistics": stats,
    }

    # Z 轴标定结果
    ctx["z_axis"] = {
        "diaphragm_z": z_info.get("diaphragm_z"),
        "diaphragm_z_normalized": round(z_info["diaphragm_z"] / nz, 3) if z_info.get("diaphragm_z") is not None else None,
        "l3_z": z_info.get("l3_detection", {}).get("l3_z"),
        "l3_z_normalized": round(z_info["l3_detection"]["l3_z"] / nz, 3)
            if z_info.get("l3_detection", {}).get("l3_z") is not None else None,
        "valid_z_range": [z_info.get("z_start", 0), z_info.get("z_end", nz - 1)],
        "valid_z_range_normalized": [
            round(z_info.get("z_start", 0) / nz, 3),
            round(z_info.get("z_end", nz - 1) / nz, 3),
        ],
        "valid_z_range_mm": z_info.get("z_range_mm"),
        "z_direction": z_info.get("l3_detection", {}).get("z_direction", {}).get("direction"),
    }

    # 椎体检测结果
    vertebrae = z_info.get("vertebrae", [])
    if vertebrae:
        ctx["vertebrae_detected"] = {
            "count": len(vertebrae),
            "summary": [
                {"index": v["index"],
                 "z_range": [v["z_start"], v["z_end"]],
                 "z_normalized": [round(v["z_start"] / nz, 3), round(v["z_end"] / nz, 3)],
                 "height_mm": round(v.get("height_mm", 0), 1)}
                for v in vertebrae
            ],
        }

    # 脊柱信息
    ctx["spine"] = {
        "total_voxels": spine_info.get("spine_voxels", 0),
        "fraction_of_volume": round(spine_info.get("spine_voxels", 0) / max(1, vol.size), 5),
    }
    spine_ctr = spine_info.get("center_yx_normalized")
    if spine_ctr:
        ctx["spine"]["center_y_normalized"] = round(spine_ctr[0], 3)
        ctx["spine"]["center_x_normalized"] = round(spine_ctr[1], 3)

    return ctx


def _build_planning_system_prompt() -> str:
    return (
        "You are an expert radiologist assisting in portal venous CT vessel extraction.\n"
        "Return STRICT JSON only — no markdown, no explanation.\n\n"

        "== YOUR TASK ==\n"
        "Analyze the CT preview images and the detected anatomy data provided below.\n"
        "Identify the portal vein system and suggest segmentation parameters.\n\n"

        "== TARGET VESSELS (must capture ALL) ==\n"
        "For NON-TIPS patients:\n"
        "  - Main portal vein trunk (MPV)\n"
        "  - Superior mesenteric vein (SMV) — extends inferiorly from confluence\n"
        "  - Splenic vein — extends horizontally to the left along posterior pancreas\n"
        "  - Left portal vein (LPV) — branches into left lobe of liver\n"
        "  - Right portal vein (RPV) — branches into right lobe of liver\n"
        "  - If portal hypertension: left gastric vein OR posterior gastric vein (variceal, tortuous)\n"
        "For TIPS patients:\n"
        "  - All of above PLUS TIPS stent (very bright, HU > 400, linear structure)\n"
        "  - LPV/RPV may be absent or reduced\n\n"

        "== CRITICAL INSTRUCTIONS FOR THRESHOLD SELECTION ==\n"
        "1. The hu_low value is the MOST IMPORTANT parameter.\n"
        "   - Too low (< 80): liver parenchyma merges with portal vein branches, making it\n"
        "     impossible to separate LPV/RPV from surrounding liver by region growth.\n"
        "   - Too high (> 160): weakly enhanced branches (especially splenic vein distal segments\n"
        "     and LPV/RPV small branches) are lost.\n"
        "   - Ideal: the lowest value where LPV/RPV cross-sections are still clearly DISTINCT\n"
        "     from liver parenchyma (a visible gap in HU between vessel lumen and liver).\n"
        "2. Look at the AXIAL porta hepatis slices carefully:\n"
        "   - Find the portal vein trunk (brightest round/oval structure near center).\n"
        "   - Estimate its HU value.\n"
        "   - Then look at LPV/RPV branches — they are typically 20-40 HU dimmer than the trunk.\n"
        "   - Liver parenchyma is typically 40-80 HU below the trunk.\n"
        "   - Set hu_low BETWEEN the liver HU and the dimmest branch HU you want to keep.\n"
        "3. hu_high for the portal channel should be ≤ 380 (spine is excluded separately).\n\n"

        "== PORTAL SEED PLACEMENT ==\n"
        "- Place portal_seed at the portal vein TRUNK near the confluence (where MPV, SMV, and\n"
        "  splenic vein meet). This is the thickest part of the portal vein.\n"
        "- The confluence is typically at the porta hepatis, near the geometric center of the\n"
        "  liver hilum.\n"
        "- NEVER place the seed on spine, kidney, or any posterior structure.\n"
        "- Use the detected anatomy data: spine center is provided, place your seed ANTERIOR to it.\n\n"

        "== USING THE DETECTED ANATOMY DATA ==\n"
        "- z_axis.valid_z_range_normalized: this is the diaphragm-to-L3 range we detected.\n"
        "  Your crop z should be WITHIN this range.\n"
        "- spine.center_y/x: the spine location. Your portal_seed.y must be well anterior\n"
        "  (smaller y value) to this.\n"
        "- vertebrae_detected: individual vertebra positions. The portal vein confluence is\n"
        "  typically at the T12-L1 vertebral level.\n\n"

        "== PREVIEW IMAGES (in order) ==\n"
        "1. coronal_mip.png — Coronal MIP (y 25-60%). LOOK HERE FIRST.\n"
        "   The portal vein tree appears as an inverted-Y. Splenic vein is horizontal, going left.\n"
        "2. sagittal_mip.png — Sagittal MIP. Confirm portal vein is ANTERIOR to spine.\n"
        "3. axial_porta.png — Key axial slices at porta hepatis level.\n"
        "   THIS IS WHERE YOU ESTIMATE HU VALUES. Look at the brightness difference between\n"
        "   portal vein lumen, liver parenchyma, and the gap between them.\n"
        "4. bone_window.png — Bone window. See spine, ribs, TIPS stent.\n"
        "5. ct_mosaic.png — Standard soft tissue mosaic.\n"
    )


def _build_planning_request(is_post_tips: bool) -> str:
    return (
        "Return STRICT JSON:\n"
        "{\n"
        "  \"hu_low\": <number, the lower HU threshold for portal vein channel>,\n"
        "  \"hu_high\": <number, upper HU threshold, typically 300-380>,\n"
        "  \"crop\": {\"z\": [start, end], \"y\": [start, end], \"x\": [start, end]},\n"
        "  \"portal_seed\": {\"z\": <normalized 0-1>, \"y\": <normalized 0-1>, \"x\": <normalized 0-1>},\n"
        + (
            "  \"include_tips\": <boolean>,\n"
            "  \"tips_hu_low\": <number, typically 400-600>,\n"
            "  \"tips_hu_high\": <number, typically 1500-3071>,\n"
        if is_post_tips else "")
        + "  \"estimated_portal_hu\": <number, your estimate of portal vein trunk HU>,\n"
        "  \"estimated_liver_hu\": <number, your estimate of liver parenchyma HU>,\n"
        "  \"estimated_hu_gap\": <number, portal_hu - liver_hu, this gap determines separability>,\n"
        "  \"reasoning\": <string, brief explanation of your threshold choice>,\n"
        "  \"notes\": <string>\n"
        "}\n\n"
        "REMEMBER:\n"
        "- hu_low should capture the dimmest branch you want to keep, NOT the trunk.\n"
        "- If estimated_hu_gap < 30, set hu_low conservatively low (80-100) because\n"
        "  separation will be difficult regardless.\n"
        "- If estimated_hu_gap > 50, you can afford a higher hu_low for cleaner results.\n"
        "- We will search multiple thresholds around your suggestion, so an approximate value is fine.\n"
        "- portal_seed.y MUST be < spine.center_y (anterior to spine).\n"
    )


def ask_for_coarse_plan(
    client: GemmaClient,
    patient_name: str,
    is_post_tips: bool,
    stats: dict,
    previews: list[Path],
    anatomy_context: dict,
) -> dict:
    """LLM 初始规划：喂入所有检测到的解剖信息。"""
    system = _build_planning_system_prompt()
    prompt = {
        "patient": patient_name,
        "is_post_tips": is_post_tips,
        "detected_anatomy": anatomy_context,
        "request": _build_planning_request(is_post_tips),
    }
    return client.chat_json(system, json.dumps(prompt, ensure_ascii=True), previews)


# =========================================================================
# STL 渲染 + LLM 迭代精修
# =========================================================================

def _stl_vertices_xyz(path: str | Path) -> np.ndarray:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) >= 84:
        triangles = struct.unpack_from("<I", raw, 80)[0]
        if 84 + triangles * 50 == len(raw):
            arr = np.frombuffer(
                raw,
                dtype=np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
                offset=84, count=triangles,
            )
            pts = arr["v"].reshape(-1, 3)
            if len(pts):
                return pts.astype(np.float32, copy=False)
    pts = []
    for line in raw.decode("ascii", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            try:
                pts.append([float(p) for p in line.split()[1:4]])
            except Exception:
                pass
    if not pts:
        raise ValueError(f"No STL vertices: {path}")
    return np.asarray(pts, dtype=np.float32)


def _stl_triangles_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) >= 84:
        n_tri = struct.unpack_from("<I", raw, 80)[0]
        if 84 + n_tri * 50 == len(raw):
            arr = np.frombuffer(
                raw,
                dtype=np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")]),
                offset=84, count=n_tri,
            )
            return arr["v"].astype(np.float32, copy=False)
    raise ValueError(f"Cannot parse binary STL: {path}")


def render_stl_views(stl_path: str | Path, out_dir: str | Path,
                     views: list[str] | None = None) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if views is None:
        views = ["anterior", "right", "superior"]
    try:
        return _render_stl_matplotlib(stl_path, out_dir, views)
    except Exception:
        return _render_stl_scatter(stl_path, out_dir, views)


def _render_stl_matplotlib(stl_path, out_dir, views):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    triangles = _stl_triangles_array(stl_path)
    center = triangles.reshape(-1, 3).mean(axis=0)
    t = (triangles - center) / max(1e-6, np.abs(triangles - center).max())
    angles = {"anterior": (0, 0), "posterior": (0, 180), "left": (0, 90),
              "right": (0, -90), "superior": (90, 0), "inferior": (-90, 0)}
    paths = []
    for vn in views:
        elev, azim = angles.get(vn, (0, 0))
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        faces = t[np.random.choice(len(t), min(30000, len(t)), replace=False)] if len(t) > 30000 else t
        mesh = Poly3DCollection(faces, alpha=0.6, facecolor="#8888cc", edgecolor="#666688", linewidth=0.1)
        ax.add_collection3d(mesh)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{vn} view"); ax.set_box_aspect([1, 1, 1])
        out = out_dir / f"stl_{vn}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
        paths.append(out)
    return paths


def _render_stl_scatter(stl_path, out_dir, views):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    verts = _stl_vertices_xyz(stl_path)
    if len(verts) > 50000:
        verts = verts[np.random.choice(len(verts), 50000, replace=False)]
    verts = verts - verts.mean(axis=0)
    proj = {"anterior": (0, 2), "posterior": (0, 2), "left": (1, 2),
            "right": (1, 2), "superior": (0, 1), "inferior": (0, 1)}
    paths = []
    for vn in views:
        a0, a1 = proj.get(vn, (0, 2))
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.scatter(verts[:, a0], verts[:, a1], s=0.3, c="#8888cc", alpha=0.3)
        ax.set_title(f"{vn} view"); ax.set_aspect("equal")
        out = out_dir / f"stl_{vn}.png"
        fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
        paths.append(out)
    return paths


def _build_refine_system_prompt() -> str:
    return (
        "You are reviewing 3D renderings of a portal vein segmentation result.\n"
        "Return STRICT JSON only.\n\n"

        "== TARGET structures (KEEP these) ==\n"
        "- Main portal vein trunk: thick tube at the center, this is the core structure.\n"
        "- SMV: extends inferiorly (downward) from the confluence.\n"
        "- Splenic vein: extends to the LEFT, horizontally, from the confluence.\n"
        "- LPV: branches into the LEFT liver lobe, extends superiorly from trunk.\n"
        "- RPV: branches into the RIGHT liver lobe, extends superiorly from trunk.\n"
        "- TIPS stent (if post-TIPS): bright linear tube connecting portal and hepatic veins.\n"
        "- Variceal gastric veins: tortuous vessels extending superiorly (if present).\n\n"

        "== UNWANTED structures (REMOVE these) ==\n"
        "- Kidney: bean-shaped blob, bilateral, posterior-lateral. Completely separate from portal tree.\n"
        "- Spleen: large oval mass on one side, NOT connected to the portal tree main structure.\n"
        "  NOTE: the splenic VEIN (thin tube) going to the spleen is wanted, but the spleen itself is not.\n"
        "- Liver parenchyma: large amorphous mass engulfing portal vein branches → hu_low is too low.\n"
        "- Aorta/IVC: vertical tubes running posterior to the portal vein.\n"
        "- Ribs/bone: any remaining bone fragments.\n"
        "- Any disconnected blob that is clearly not a vessel.\n\n"

        "== HOW TO DISTINGUISH ==\n"
        "- Vessels are TUBULAR (thin, elongated, branching). Organs are BLOB-LIKE (round, bulky).\n"
        "- The portal tree has a characteristic shape: inverted-Y in the anterior view,\n"
        "  with branches going up (LPV/RPV), one arm going left (splenic), one going down (SMV).\n"
        "- If the result looks like a clean tree with branches → assessment is 'ok'.\n"
        "- If there's a large blob attached → it's likely liver (hu_low too low) or kidney/spleen.\n\n"

        "== VIEWS PROVIDED ==\n"
        "1. anterior: looking from front. Portal tree should be visible as inverted-Y.\n"
        "2. right: looking from right side. See anterior-posterior depth.\n"
        "3. superior: looking from above. See left-right spread.\n"
    )


def _build_refine_request(round_num: int, mask_stats: dict, threshold_context: dict | None) -> str:
    ctx = ""
    if threshold_context:
        ctx = (
            f"Current hu_low = {threshold_context.get('hu_low', '?')}. "
            f"Threshold search tested range 75-195 and picked this as best balance of "
            f"completeness vs liver separation. "
        )
    return (
        f"Round {round_num} review. {ctx}"
        "Examine the 3D renderings and return STRICT JSON:\n"
        "{\n"
        '  "assessment": "ok" or "needs_cleanup",\n'
        '  "portal_vein_visible": <boolean, can you see the portal vein tree?>,\n'
        '  "description": <string, describe what you see in 2-3 sentences>,\n'
        '  "unwanted_structures": [\n'
        '    {"name": <string>, "description": <string, shape and location>}\n'
        "  ],\n"
        '  "remove_disconnected_blobs": <boolean, remove anything not connected to portal tree>,\n'
        '  "remove_regions": [\n'
        '    {"description": <string>,\n'
        '     "z_frac": [start, end], "y_frac": [start, end], "x_frac": [start, end]}\n'
        "  ],\n"
        '  "hu_adjustment": "raise_low" or "lower_low" or "ok",\n'
        '  "notes": <string>\n'
        "}\n\n"
        "RULES:\n"
        '- If the result is a clean portal vein tree → set assessment to "ok" and stop.\n'
        '- If there are blobs but they look disconnected → set remove_disconnected_blobs=true.\n'
        "- Only use remove_regions for large unwanted masses that are connected to the portal tree.\n"
        '- Set hu_adjustment to "raise_low" ONLY if you see liver parenchyma fused with branches.\n'
        '- Set hu_adjustment to "lower_low" ONLY if the tree looks incomplete (missing branches).\n'
        f"- Current mask has {mask_stats.get('voxels', 0)} voxels.\n"
    )


def _ask_llm_refine_stl(client, stl_views, patient_name, is_post_tips,
                         round_num, mask_stats, threshold_context=None):
    system = _build_refine_system_prompt()
    prompt = {
        "patient": patient_name,
        "is_post_tips": is_post_tips,
        "request": _build_refine_request(round_num, mask_stats, threshold_context),
    }
    return client.chat_json(system, json.dumps(prompt), stl_views)


def _apply_llm_removal(mask, vol, refine_result, seed_zyx, spacing_zyx):
    info = {"regions_removed": 0, "voxels_before": int(mask.sum())}
    if refine_result.get("assessment") == "ok":
        info.update({"action": "no_change", "voxels_after": int(mask.sum())})
        return mask, info
    mask = mask.copy()
    nz, ny, nx = mask.shape
    for region in refine_result.get("remove_regions", []):
        try:
            zf, yf, xf = region.get("z_frac", [0, 1]), region.get("y_frac", [0, 1]), region.get("x_frac", [0, 1])
            mask[int(zf[0]*nz):int(zf[1]*nz), int(yf[0]*ny):int(yf[1]*ny), int(xf[0]*nx):int(xf[1]*nx)] = False
            info["regions_removed"] += 1
        except Exception:
            continue
    if refine_result.get("remove_disconnected_blobs") and seed_zyx is not None and ndi is not None:
        mask, gi = _region_grow_from_seed(mask, seed_zyx, spacing_zyx, REGION_GROW_BRIDGE_MM, "refine")
        info["region_grow_after"] = gi
    info.update({"voxels_after": int(mask.sum()), "action": "cleaned"})
    return mask, info


def _adjust_threshold_from_llm(plan, refine_result):
    plan = dict(plan)
    adj = refine_result.get("hu_adjustment", "ok")
    if adj == "raise_low":
        plan["hu_low"] = min(plan["hu_low"] + 20.0, 200.0)
    elif adj == "lower_low":
        plan["hu_low"] = max(plan["hu_low"] - 20.0, 50.0)
    return plan


# =========================================================================
# STL 参考、seed、region grow
# =========================================================================

def _tree_mtime(p):
    p = Path(p)
    if not p.exists():
        return 0.0
    latest = p.stat().st_mtime
    if p.is_file():
        return latest
    for i in p.rglob("*"):
        if i.is_file():
            latest = max(latest, i.stat().st_mtime)
    return latest


def _stl_bounds_xyz(path):
    pts = _stl_vertices_xyz(path)
    return pts.min(axis=0), pts.max(axis=0)


def _reference_crop_from_stl(case, volume, padding=(0.06, 0.05, 0.05)):
    ref = case.path / "pre.stl"
    if not ref.exists():
        return None
    try:
        bmin, bmax = _stl_bounds_xyz(ref)
    except Exception:
        return None
    sh = np.asarray(volume.volume_hu.shape, dtype=np.float32)
    sp = np.asarray(volume.spacing_zyx, dtype=np.float32)
    og = np.asarray(volume.origin_xyz, dtype=np.float32)
    lo = np.asarray([(bmin[2] - og[2]) / sp[0], (bmin[1] - og[1]) / sp[1], (bmin[0] - og[0]) / sp[2]], dtype=np.float32) / sh
    hi = np.asarray([(bmax[2] - og[2]) / sp[0], (bmax[1] - og[1]) / sp[1], (bmax[0] - og[0]) / sp[2]], dtype=np.float32) / sh
    pad = np.asarray(padding, dtype=np.float32)
    lo, hi = np.maximum(0.0, lo - pad), np.minimum(1.0, hi + pad)
    if np.any(hi - lo < 0.03):
        return None
    return {"z": [float(lo[0]), float(hi[0])], "y": [float(lo[1]), float(hi[1])], "x": [float(lo[2]), float(hi[2])]}


def _reference_envelope_mask(case, volume, radius_mm=16.0):
    ref = case.path / "pre.stl"
    info = {"enabled": False, "source": None, "radius_mm": float(radius_mm), "voxels": 0}
    if not ref.exists() or ndi is None:
        return None, info
    try:
        pts = _stl_vertices_xyz(ref)
    except Exception as e:
        info["error"] = str(e)
        return None, info
    sp = np.asarray(volume.spacing_zyx, dtype=np.float32)
    og = np.asarray(volume.origin_xyz, dtype=np.float32)
    sh = np.asarray(volume.volume_hu.shape, dtype=np.int64)
    idx = np.empty((len(pts), 3), dtype=np.int64)
    idx[:, 0] = np.rint((pts[:, 2] - og[2]) / sp[0]).astype(np.int64)
    idx[:, 1] = np.rint((pts[:, 1] - og[1]) / sp[1]).astype(np.int64)
    idx[:, 2] = np.rint((pts[:, 0] - og[0]) / sp[2]).astype(np.int64)
    valid = np.all((idx >= 0) & (idx < sh), axis=1)
    if not np.any(valid):
        info["error"] = "outside_volume"
        return None, info
    env = np.zeros(tuple(int(v) for v in sh), dtype=bool)
    idx = idx[valid]
    env[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    iters = max(1, int(round(radius_mm / float(np.min(sp)))))
    env = ndi.binary_dilation(env, iterations=iters)
    info.update({"enabled": True, "source": "pre.stl", "voxels": int(env.sum()), "dilation_iterations": iters})
    return env, info


def _stl_center_seed(path, volume):
    try:
        bmin, bmax = _stl_bounds_xyz(path)
    except Exception:
        return None
    c = (bmin + bmax) / 2.0
    sp = np.asarray(volume.spacing_zyx, dtype=np.float32)
    og = np.asarray(volume.origin_xyz, dtype=np.float32)
    s = (float((c[2] - og[2]) / sp[0]), float((c[1] - og[1]) / sp[1]), float((c[0] - og[0]) / sp[2]))
    sh = volume.volume_hu.shape
    if s[0] < -1 or s[1] < -1 or s[2] < -1 or s[0] > sh[0] or s[1] > sh[1] or s[2] > sh[2]:
        return None
    return s


def _portal_seed_from_reference(case, volume):
    for name, src in (("vessel.stl", "vessel.stl"), ("pre.stl", "pre.stl")):
        p = case.path / name
        if not p.exists():
            continue
        s = _stl_center_seed(p, volume)
        if s is not None:
            return s, src
    return None, None


def _portal_seed_from_plan(plan, volume):
    seed = plan.get("portal_seed")
    if not isinstance(seed, dict):
        return None, None
    try:
        sh = np.asarray(volume.volume_hu.shape, dtype=np.float32) - 1.0
        return (
            float(max(0, min(1, float(seed["z"]))) * sh[0]),
            float(max(0, min(1, float(seed["y"]))) * sh[1]),
            float(max(0, min(1, float(seed["x"]))) * sh[2]),
        ), "model_portal_seed"
    except Exception:
        return None, None


def _label_int32(mask: np.ndarray):
    """内存优化版 ndi.label：输出 int32 而非 int64，节省 50% 内存。"""
    out = np.empty(mask.shape, dtype=np.int32)
    n = ndi.label(mask, output=out)
    return out, int(n)


def _count_labels(labels: np.ndarray, n: int) -> np.ndarray:
    """内存安全的 label 计数，逐层统计避免创建完整 int64 副本。

    np.bincount(labels.ravel().astype(np.intp)) 会分配 ~860 MiB（112M×8B），
    此函数逐层处理，峰值内存仅 512×512×4B ≈ 1 MiB。
    """
    counts = np.zeros(n + 1, dtype=np.int64)
    for z in range(labels.shape[0]):
        # 每次只取一层 ravel — 仅 512*512 个 int32 = 1 MiB
        sl = labels[z].ravel()
        # np.bincount 对 int32 小数组不需要 astype
        c = np.bincount(sl, minlength=n + 1)
        counts[:len(c)] += c
    return counts


def _find_seed_label(labels: np.ndarray, mask: np.ndarray, seed_zyx,
                     spacing_zyx, max_snap_mm: float = REGION_GROW_MAX_SEED_SNAP_MM):
    """在 seed 附近找到最近的前景体素的 label，不做全量 argwhere。"""
    nz, ny, nx = labels.shape
    sp = np.asarray(spacing_zyx, dtype=np.float32)
    sz, sy, sx = int(round(seed_zyx[0])), int(round(seed_zyx[1])), int(round(seed_zyx[2]))
    sz, sy, sx = max(0, min(nz - 1, sz)), max(0, min(ny - 1, sy)), max(0, min(nx - 1, sx))

    # 先直接检查 seed 位置
    if mask[sz, sy, sx]:
        return int(labels[sz, sy, sx]), 0.0, (sz, sy, sx)

    # 在递增的搜索半径内找最近的前景体素
    max_r_voxels = max(3, int(round(max_snap_mm / float(np.min(sp))))) + 1
    for r in range(1, max_r_voxels + 1):
        z0, z1 = max(0, sz - r), min(nz, sz + r + 1)
        y0, y1 = max(0, sy - r), min(ny, sy + r + 1)
        x0, x1 = max(0, sx - r), min(nx, sx + r + 1)
        patch = mask[z0:z1, y0:y1, x0:x1]
        if patch.any():
            local_coords = np.argwhere(patch)  # 小 patch，内存安全
            # 转回全局坐标
            global_coords = local_coords + np.array([z0, y0, x0])
            dists_mm = np.sqrt(np.sum(((global_coords.astype(np.float32)
                                        - np.array(seed_zyx, dtype=np.float32)) * sp) ** 2, axis=1))
            best = int(np.argmin(dists_mm))
            gz, gy, gx = global_coords[best]
            return int(labels[gz, gy, gx]), float(dists_mm[best]), (int(gz), int(gy), int(gx))

    return None, float("inf"), None


def _region_grow_from_seed(mask, seed_zyx, spacing_zyx=(1, 1, 1),
                           bridge_mm=REGION_GROW_BRIDGE_MM, seed_source="explicit"):
    """内存优化版区域生长：int32 labels + 局部 seed 搜索 + 原地结果构建。"""
    mask = np.asarray(mask, dtype=bool)
    info = {"enabled": bool(seed_zyx is not None and ndi is not None),
            "seed_source": seed_source, "input_voxels": int(mask.sum())}
    if seed_zyx is None or ndi is None or mask.sum() == 0:
        info["output_voxels"] = int(mask.sum())
        return mask, info

    labels, n = _label_int32(mask)
    if n <= 1:
        del labels
        info["output_voxels"] = int(mask.sum())
        return mask, info

    # 局部搜索 seed 最近的前景体素（不做全量 argwhere）
    ml, nearest_mm, nearest_pos = _find_seed_label(
        labels, mask, seed_zyx, spacing_zyx, REGION_GROW_MAX_SEED_SNAP_MM,
    )
    info["nearest_seed_distance_mm"] = round(nearest_mm, 2)

    if ml is None:
        del labels
        info.update({"output_voxels": int(mask.sum()), "skipped_reason": "seed_too_far"})
        return mask, info

    # 主连通域
    mc = (labels == ml)
    sp = np.asarray(spacing_zyx, dtype=np.float32)
    bv = max(1, int(round(bridge_mm / float(np.min(sp)))))
    bz = ndi.binary_dilation(mc, iterations=bv)
    del mc  # 释放

    # 桥接检测
    bridged = {ml}
    counts = _count_labels(labels, n)
    nz_vol = labels.shape[0]
    for lb in range(1, n + 1):
        if lb == ml or counts[lb] < 32:
            continue
        # 逐层检查重叠，避免创建完整 (labels==lb) 数组
        found = False
        for zz in range(nz_vol):
            if np.any((labels[zz] == lb) & bz[zz]):
                found = True
                break
        if found:
            bridged.add(lb)
    del bz  # 释放

    # 原地构建结果：不用 np.isin（它会创建完整副本）
    result = np.zeros(mask.shape, dtype=bool)
    for lb in bridged:
        result |= (labels == lb)
    del labels  # 释放

    info.update({"output_voxels": int(result.sum()), "bridged_components": len(bridged) - 1,
                 "removed_components": n - len(bridged)})
    return result, info


# =========================================================================
# 分割核心
# =========================================================================

def _segment_once(vol, plan, is_post_tips, spine_mask=None):
    target = TARGET_VOXELS_TIPS if is_post_tips else TARGET_VOXELS
    if is_post_tips:
        pp = dict(plan)
        pp["hu_high"] = min(380.0, float(plan["hu_high"]))
        pp["crop"] = _intersect_crop(plan["crop"],
                                     {"z": [0.20, 0.90], "y": [0.18, 0.82], "x": [0.08, 0.92]})
        portal = _threshold_components(vol, pp, keep=8, max_voxels=target, spine_mask=spine_mask)
        if not bool(plan.get("include_tips", True)):
            return portal
        tp = {
            "hu_low": float(plan.get("tips_hu_low", 430)),
            "hu_high": float(plan.get("tips_hu_high", 3071)),
            "crop": _intersect_crop(plan["crop"],
                                    {"z": [0.20, 0.92], "y": [0.18, 0.78], "x": [0.15, 0.85]}),
        }
        tips = _threshold_components(vol, tp, keep=4, close_iterations=1,
                                     max_voxels=target // 2, spine_mask=spine_mask)
        return _largest_components(portal | tips, keep=10, min_voxels=64)
    return _threshold_components(vol, plan, keep=8, max_voxels=target, spine_mask=spine_mask)


def _threshold_components(vol, plan, keep, close_iterations=2,
                          max_voxels=TARGET_VOXELS, spine_mask=None):
    cs = _crop_slices(vol.shape, plan["crop"])
    roi = vol[cs]
    raw = (roi >= plan["hu_low"]) & (roi <= plan["hu_high"])
    if spine_mask is not None:
        raw = raw & ~spine_mask[cs]
    mr = raw
    if ndi is not None:
        f = ndi.binary_opening(raw, iterations=1)
        f = ndi.binary_closing(f, iterations=close_iterations)
        f = _fill_small_holes(f, 500)
        f = _largest_components(f, keep=keep, min_voxels=64)
        mr = f if int(f.sum()) > 0 else raw
    mask = np.zeros(vol.shape, dtype=bool)
    mask[cs] = mr
    return mask


def _fill_small_holes(mask, max_vox=500):
    if ndi is None:
        return mask
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    del filled
    if holes.sum() == 0:
        del holes
        return mask
    hl, nh = _label_int32(holes)
    del holes
    if nh == 0:
        del hl
        return mask
    counts = _count_labels(hl, nh)
    sm = np.zeros_like(mask)
    for i in range(1, nh + 1):
        if counts[i] <= max_vox:
            sm |= (hl == i)
    del hl
    return mask | sm


# =========================================================================
# Plan sanitize
# =========================================================================

def _default_plan(is_post_tips):
    if is_post_tips:
        return {"hu_low": 90.0, "hu_high": 350.0,
                "crop": {"z": [0.25, 0.88], "y": [0.22, 0.78], "x": [0.10, 0.90]},
                "notes": "v4 wide capture"}
    return {"hu_low": 100.0, "hu_high": 350.0,
            "crop": {"z": [0.30, 0.82], "y": [0.25, 0.75], "x": [0.15, 0.85]},
            "notes": "v4 wide capture fallback"}


def _sanitize_plan(plan, is_post_tips):
    default = _default_plan(is_post_tips)
    out = dict(default)
    hu_bounds = (50.0, 380.0) if is_post_tips else (60.0, 380.0)
    for key in ("hu_low", "hu_high"):
        try:
            out[key] = float(plan.get(key, default[key]))
        except Exception:
            out[key] = default[key]
    out["hu_low"] = max(hu_bounds[0], min(out["hu_low"], hu_bounds[1] - 20))
    out["hu_high"] = max(out["hu_low"] + 20, min(out["hu_high"], hu_bounds[1]))
    out["include_tips"] = bool(plan.get("include_tips", is_post_tips))
    if is_post_tips:
        try:
            out["tips_hu_low"] = max(250, min(float(plan.get("tips_hu_low", 430)), 1200))
        except Exception:
            out["tips_hu_low"] = 430.0
        try:
            out["tips_hu_high"] = max(out["tips_hu_low"] + 20, min(float(plan.get("tips_hu_high", 3071)), 3071))
        except Exception:
            out["tips_hu_high"] = 3071.0
    seed = plan.get("portal_seed")
    if isinstance(seed, dict):
        try:
            sv = {a: float(max(0, min(float(seed[a]), 1))) for a in ("z", "y", "x")}
            if sv["y"] < 0.70:
                out["portal_seed"] = sv
        except Exception:
            pass
    for info_key in ("estimated_portal_hu", "estimated_liver_hu", "estimated_hu_gap", "reasoning"):
        if info_key in plan:
            out[info_key] = plan[info_key]
    crop = default["crop"].copy()
    for axis in ("z", "y", "x"):
        vals = (plan.get("crop", {}) or {}).get(axis, crop[axis])
        try:
            a, b = max(0, min(float(vals[0]), 1)), max(0, min(float(vals[1]), 1))
            if b - a >= 0.05:
                crop[axis] = [a, b]
        except Exception:
            pass
    crop["y"][1] = min(crop["y"][1], 0.82)
    out["crop"] = _limit_crop_span(
        crop,
        {"z": 0.68, "y": 0.58, "x": 0.84} if is_post_tips else {"z": 0.55, "y": 0.52, "x": 0.74},
    )
    out["notes"] = str(plan.get("notes", default["notes"]))
    return out


def _apply_z_standardization(plan, z_start, z_end, nz, has_ref_crop=False):
    plan = dict(plan)
    plan["crop"] = {ax: list(v) for ax, v in plan["crop"].items()}
    if has_ref_crop:
        return plan
    z0, z1 = z_start / nz, z_end / nz
    plan["crop"]["z"] = [max(plan["crop"]["z"][0], z0), min(plan["crop"]["z"][1], z1)]
    if plan["crop"]["z"][1] - plan["crop"]["z"][0] < 0.05:
        plan["crop"]["z"] = [z0, z1]
    return plan


def _limit_crop_span(crop, max_span):
    out = {}
    for ax in ("z", "y", "x"):
        a, b = crop[ax]
        if b - a > max_span[ax]:
            c = (a + b) / 2
            a, b = c - max_span[ax] / 2, c + max_span[ax] / 2
            if a < 0:
                b -= a
                a = 0
            if b > 1:
                a -= b - 1
                b = 1
        out[ax] = [round(float(max(0, a)), 4), round(float(min(1, b)), 4)]
    return out


def _intersect_crop(a, b):
    out = {}
    for ax in ("z", "y", "x"):
        out[ax] = [max(float(a[ax][0]), float(b[ax][0])), min(float(a[ax][1]), float(b[ax][1]))]
        if out[ax][1] - out[ax][0] < 0.05:
            out[ax] = list(b[ax])
    return out


def _crop_slices(shape, crop):
    spans = []
    for ax, n in zip(("z", "y", "x"), shape):
        a, b = crop[ax]
        s = max(0, min(n, int(round(a * n))))
        e = max(s + 1, min(n, int(round(b * n))))
        spans.append(slice(s, e))
    return tuple(spans)


def _largest_components(mask, keep=6, min_voxels=64):
    if ndi is None:
        return mask
    labels, n = _label_int32(mask)
    if n == 0:
        del labels
        return mask
    counts = np.bincount(labels.ravel().astype(np.intp))
    counts[0] = 0
    chosen = [i for i in np.argsort(counts)[::-1][:keep] if counts[i] >= min_voxels]
    if not chosen:
        del labels
        return np.zeros(mask.shape, dtype=bool)
    # 原地构建，不用 np.isin
    result = np.zeros(mask.shape, dtype=bool)
    for lb in chosen:
        result |= (labels == lb)
    del labels
    return result


def _binary_stl_triangle_count(path):
    with Path(path).open("rb") as f:
        f.seek(80)
        raw = f.read(4)
    return int(struct.unpack("<I", raw)[0]) if len(raw) == 4 else 0


# =========================================================================
# 质量评估
# =========================================================================

def _pretrain_quality_details(mask, stl_bytes, max_voxels=TARGET_VOXELS):
    issues = []
    mask = np.asarray(mask, dtype=bool)
    voxels = int(mask.sum())
    stats = {"voxels": voxels}
    if voxels == 0:
        issues.append("empty")
    if stl_bytes > MAX_STL_BYTES:
        issues.append("stl_over_20mb")
    if voxels > max_voxels:
        issues.append("too_many_voxels")
    if ndi and voxels > 0:
        labels, n = _label_int32(mask)
        counts = np.bincount(labels.ravel().astype(np.intp))
        cc = int(np.count_nonzero(counts[1:] >= 64))
        stats["components"] = cc
        if cc > 16:
            issues.append("too_many_components")
        del labels
    return ("review" if issues else "ok", issues, stats)


def _evaluate_pretrain_against_label(case, grid_size=96):
    if not case.label_stl.exists() or not case.pretrain_stl.exists():
        return None
    try:
        pre, bounds = stl_to_voxels(case.pretrain_stl, grid_size=grid_size)
        label, _ = stl_to_voxels(case.label_stl, grid_size=grid_size, bounds=bounds)
    except Exception as e:
        return {"error": str(e)}
    pm, lm = pre > 0.5, label > 0.5
    inter = int(np.logical_and(pm, lm).sum())
    pc, lc = int(pm.sum()), int(lm.sum())
    d = pc + lc
    return {
        "dice": float((2 * inter / d) if d else 1.0),
        "precision": float((inter / pc) if pc else 0.0),
        "recall": float((inter / lc) if lc else 0.0),
        "pretrain_voxels": pc,
        "label_voxels": lc,
    }


# =========================================================================
# 缓存 + 加载
# =========================================================================

def load_case_volume(case):
    return load_dicom_series(case.dcm_dir), case.dcm_dir, _tree_mtime(case.dcm_dir), "dcm"


def _should_rebuild(case, meta_path, input_mtime):
    meta_path = Path(meta_path)
    if not case.pretrain_stl.exists():
        return True, "missing_stl"
    if not meta_path.exists():
        return True, "missing_meta"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True, "invalid_meta"
    if meta.get("algorithm_version") != PRETRAIN_ALGORITHM_VERSION:
        return True, "version"
    if abs(float(meta.get("input_mtime", -1)) - float(input_mtime)) > 1e-3:
        return True, "input_changed"
    return False, "up_to_date"


# =========================================================================
# 主流程
# =========================================================================

def pretrain_patient(case, client: GemmaClient | None = None, force: bool = False) -> PretrainResult:
    work_dir = case.path / "vkan_work"
    meta_path = work_dir / PRETRAIN_META_NAME
    dcm, input_path, input_mtime, input_source = load_case_volume(case)

    if not force:
        ok, reason = _should_rebuild(case, meta_path, input_mtime)
        if not ok:
            return PretrainResult(case.pretrain_stl, "reused")
    else:
        reason = "forced"

    vol = dcm.volume_hu
    work_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 1: 脊柱检测
    # ================================================================
    spine_mask = _detect_spine_mask(vol)
    spine_center = _spine_center_yx(spine_mask)
    spine_info = {
        "spine_voxels": int(spine_mask.sum()),
        "center_yx_normalized": list(spine_center) if spine_center else None,
    }

    # ================================================================
    # Step 2: Z 轴标定（椎体逐节检测 + L3 定位）
    # ================================================================
    if standardize_z_range is not None:
        z_start, z_end, z_info = standardize_z_range(
            vol, dcm.spacing_zyx, spine_mask=spine_mask, margin_mm=20.0,
        )
    else:
        # fallback
        nz = vol.shape[0]
        z_start, z_end = int(nz * 0.25), int(nz * 0.80)
        z_info = {"z_start": z_start, "z_end": z_end, "method": "fallback", "vertebrae": []}

    # ================================================================
    # Step 3: CT 预览图 + LLM 初始规划
    # ================================================================
    previews, preview_info = _save_planning_previews(vol, work_dir)

    stats = {
        "p01": float(np.percentile(vol, 1)),
        "p50": float(np.percentile(vol, 50)),
        "p95": float(np.percentile(vol, 95)),
        "p99": float(np.percentile(vol, 99)),
        "shape_zyx": list(vol.shape),
    }

    # 构建给 LLM 的解剖上下文
    anatomy_context = _build_anatomy_context(vol, dcm.spacing_zyx, z_info, spine_info, stats)

    raw_plan: dict = {}
    if client and client.enabled:
        raw_plan = ask_for_coarse_plan(
            client, case.name, case.is_post_tips, stats, previews, anatomy_context,
        )

    # 参考 STL
    ref_crop = _reference_crop_from_stl(case, dcm)
    if ref_crop:
        raw_plan = dict(raw_plan)
        raw_plan["crop"] = ref_crop
        raw_plan["notes"] = "pre.stl reference crop"

    plan = _sanitize_plan(raw_plan, case.is_post_tips)
    plan = _apply_z_standardization(plan, z_start, z_end, vol.shape[0], has_ref_crop=bool(ref_crop))

    # seed
    seed_zyx, seed_src = _portal_seed_from_reference(case, dcm)
    if seed_zyx is None:
        seed_zyx, seed_src = _portal_seed_from_plan(plan, dcm)

    # envelope
    envelope, envelope_info = _reference_envelope_mask(case, dcm)

    # ================================================================
    # Step 4: 阈值搜索
    # ================================================================
    if search_best_threshold is not None:
        best_plan, threshold_info = search_best_threshold(
            vol=vol,
            plan=plan,
            spine_mask=spine_mask,
            seed_zyx=seed_zyx,
            spacing_zyx=dcm.spacing_zyx,
            is_post_tips=case.is_post_tips,
            segment_fn=_segment_once,
            region_grow_fn=lambda m, s, sp: _region_grow_from_seed(m, s, sp, REGION_GROW_BRIDGE_MM),
            reference_envelope=envelope,
        )
        plan = best_plan
    else:
        threshold_info = {"status": "module_unavailable"}

    (work_dir / "threshold_search.json").write_text(
        json.dumps(threshold_info, indent=2, default=str), encoding="utf-8",
    )

    # ================================================================
    # Step 5: 分割 + 区域生长
    # ================================================================
    mask = _segment_once(vol, plan, case.is_post_tips, spine_mask=spine_mask)

    if envelope is not None:
        em = mask & envelope
        envelope_info["applied"] = int(em.sum()) > 0
        if int(em.sum()) > 0:
            mask = em

    if not envelope_info.get("applied"):
        mask, grow_info = _region_grow_from_seed(
            mask, seed_zyx, dcm.spacing_zyx, REGION_GROW_BRIDGE_MM, seed_src,
        )
    else:
        grow_info = {"skipped": "envelope_applied"}

    # ================================================================
    # Step 6: STL 渲染 → LLM 迭代精修
    # ================================================================
    refine_rounds: list[dict] = []
    current_plan = plan

    for round_num in range(MAX_REFINE_ROUNDS):
        temp_stl = work_dir / f"pretrain_round{round_num}.stl"
        mask_to_stl(mask, dcm.spacing_zyx, temp_stl, origin_xyz=dcm.origin_xyz)

        stl_views = render_stl_views(temp_stl, work_dir / f"stl_views_round{round_num}")
        if not stl_views or not client or not client.enabled:
            refine_rounds.append({"round": round_num, "skipped": "no_views_or_no_client"})
            break

        mask_stats = {"voxels": int(mask.sum()), "round": round_num}
        threshold_ctx = {"hu_low": current_plan.get("hu_low")}

        refine_result = _ask_llm_refine_stl(
            client, stl_views, case.name, case.is_post_tips,
            round_num, mask_stats, threshold_ctx,
        )
        refine_rounds.append({"round": round_num, "llm_response": refine_result})

        if refine_result.get("assessment") == "ok":
            break

        mask, removal_info = _apply_llm_removal(mask, vol, refine_result, seed_zyx, dcm.spacing_zyx)
        refine_rounds[-1]["removal"] = removal_info

        if refine_result.get("hu_adjustment", "ok") != "ok":
            current_plan = _adjust_threshold_from_llm(current_plan, refine_result)
            mask = _segment_once(vol, current_plan, case.is_post_tips, spine_mask=spine_mask)
            if not envelope_info.get("applied"):
                mask, _ = _region_grow_from_seed(
                    mask, seed_zyx, dcm.spacing_zyx, REGION_GROW_BRIDGE_MM, seed_src,
                )
            refine_rounds[-1]["re_segmented"] = True
            refine_rounds[-1]["adjusted_plan"] = current_plan

    # ================================================================
    # Step 7: 输出
    # ================================================================
    (work_dir / "coarse_plan.json").write_text(
        json.dumps(current_plan, indent=2), encoding="utf-8",
    )
    np.save(work_dir / "pretrain_mask.npy", mask.astype(np.uint8))
    out_path = mask_to_stl(mask, dcm.spacing_zyx, case.pretrain_stl, origin_xyz=dcm.origin_xyz)
    stl_bytes = int(out_path.stat().st_size)
    quality, issues, qstats = _pretrain_quality_details(
        mask, stl_bytes, TARGET_VOXELS_TIPS if case.is_post_tips else TARGET_VOXELS,
    )
    eval_metrics = _evaluate_pretrain_against_label(case, grid_size=96)

    meta = {
        "algorithm_version": PRETRAIN_ALGORITHM_VERSION,
        "status_reason": reason,
        "input_source": input_source,
        "input_dcm": str(input_path),
        "input_mtime": input_mtime,
        "is_post_tips": bool(case.is_post_tips),
        "pretrain_stl": str(out_path),
        "reference_stl": str(case.path / "pre.stl") if ref_crop else None,
        "plan": current_plan,
        "anatomy_context": anatomy_context,
        "z_standardization": z_info,
        "threshold_search": {
            "final_hu_low": current_plan.get("hu_low"),
            "summary": {
                k: threshold_info.get(k)
                for k in ("status", "best_hu_low", "best_score", "best_voxels", "n_candidates")
                if k in threshold_info
            },
        },
        "spine_exclusion": spine_info,
        "reference_envelope": envelope_info,
        "initial_region_grow": grow_info,
        "refine_rounds": refine_rounds,
        "pretrain_quality": quality,
        "quality_issues": issues,
        "quality_stats": qstats,
        "pretrain_vessel_eval": eval_metrics,
        "volume_shape_zyx": list(vol.shape),
        "spacing_zyx": list(dcm.spacing_zyx),
        "origin_xyz": list(dcm.origin_xyz),
        "mask_voxels": int(mask.sum()),
        "stl_bytes": stl_bytes,
        "stl_triangles": _binary_stl_triangle_count(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return PretrainResult(out_path, "review" if quality == "review" else "wrote")


def coarse_segment_patient(case, client=None, force=False):
    return pretrain_patient(case, client=client, force=force).path


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="v4: anatomy-aware portal vein extraction with vertebra detection + threshold scoring.",
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--patient", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = GemmaClient(api_key=args.api_key, model=args.model, base_url=args.api_base_url)
    cases = discover_patients(args.data_root)
    if args.patient:
        cases = [c for c in cases if c.name == args.patient]
    print(f"[v4] found {len(cases)} patients")
    for case in cases:
        try:
            r = pretrain_patient(case, client=client, force=args.force)
            print(f"[v4] {case.name}: {r.status} {r.path}")
        except Exception as e:
            print(f"[v4] {case.name}: failed: {e}")


if __name__ == "__main__":
    main()