"""椎体逐节检测 + L3 上缘定位。

用法：
    from vertebra_detector import detect_vertebrae, locate_l3_upper_border

    vertebrae, info = detect_vertebrae(vol, spacing_zyx)
    l3_z, l3_info = locate_l3_upper_border(vol, spacing_zyx, vertebrae)
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as ndi
    from scipy.signal import find_peaks
except ImportError:
    ndi = None
    find_peaks = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CORTICAL_BONE_HU = 400.0
# 椎体在轴位上的最小面积（像素数），太小的不算
MIN_VERTEBRAL_BODY_PIXELS = 80
# 椎体在 z 方向的典型高度范围 (mm)
VERTEBRAL_HEIGHT_MM = (15.0, 35.0)
# 椎间盘典型高度 (mm)
DISC_HEIGHT_MM = (4.0, 15.0)
# 从 T12 到 L3 上缘跨越约 3 个椎体 ≈ 60-90mm
T12_TO_L3_MM = 75.0
# 膈肌到 T12 的典型距离 (mm) —— T12 椎体通常紧邻膈肌下方
DIAPHRAGM_TO_T12_MM = 15.0


def detect_vertebrae(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    spine_mask: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """检测每个椎体的 z 范围。

    原理：
    1. 沿 z 轴统计脊柱区域的骨质面积（截面积曲线）
    2. 椎体 = 截面积的峰值区域（骨质密实）
    3. 椎间盘 = 截面积的谷值（HU 较低，面积骤降）
    4. 用 valley detection 分割每个椎体

    参数：
        vol: z-y-x HU 体积
        spacing_zyx: 体素间距 (mm)
        spine_mask: 可选，预计算的脊柱 mask。如果没有则自动检测。

    返回：
        vertebrae: [{"z_start": int, "z_end": int, "z_center": float,
                     "area_mean": float, "index": int}, ...]
                   按 z 从小到大排列，index=0 是最先出现的椎体
        info: 调试信息
    """
    if ndi is None or find_peaks is None:
        return [], {"error": "scipy_not_available"}

    nz, ny, nx = vol.shape
    dz = float(spacing_zyx[0])
    info: dict = {"dz_mm": dz}

    # --- Step 1: 获取脊柱中心线位置 ---
    if spine_mask is None:
        spine_center_y, spine_center_x = _estimate_spine_center(vol)
    else:
        coords = np.argwhere(spine_mask)
        if len(coords) == 0:
            return [], {"error": "empty_spine_mask"}
        spine_center_y = float(np.median(coords[:, 1]))
        spine_center_x = float(np.median(coords[:, 2]))

    info["spine_center_yx"] = [spine_center_y, spine_center_x]

    # --- Step 2: 在脊柱区域提取 z 方向截面积曲线 ---
    # 取脊柱中心附近的一个窗口（约 40x60mm）来统计骨质面积
    wy = max(5, int(round(20.0 / float(spacing_zyx[1]))))  # ±20mm
    wx = max(5, int(round(30.0 / float(spacing_zyx[2]))))  # ±30mm
    cy, cx = int(round(spine_center_y)), int(round(spine_center_x))
    y0, y1 = max(0, cy - wy), min(ny, cy + wy + 1)
    x0, x1 = max(0, cx - wx), min(nx, cx + wx + 1)

    # 逐层统计高 HU 体素数 → 椎体截面积曲线
    bone_area = np.zeros(nz, dtype=np.float32)
    for z in range(nz):
        roi = vol[z, y0:y1, x0:x1]
        bone_area[z] = float((roi > CORTICAL_BONE_HU).sum())

    # 平滑消除噪声
    bone_area_smooth = ndi.gaussian_filter1d(bone_area, sigma=1.5)
    info["bone_area_max"] = float(bone_area_smooth.max())

    if bone_area_smooth.max() < MIN_VERTEBRAL_BODY_PIXELS:
        return [], {"error": "no_significant_bone", **info}

    # --- Step 3: 找椎间盘位置（骨质面积的谷值）---
    # 椎间盘是骨质截面积的局部最小值
    # 反转曲线找峰值 = 找原曲线谷值
    inverted = bone_area_smooth.max() - bone_area_smooth

    # 椎间盘间距约 20-35mm（一个椎体高度）
    min_distance_slices = max(3, int(round(VERTEBRAL_HEIGHT_MM[0] / max(0.1, dz))))
    # prominence: 谷值要比两侧椎体低至少 30% 的峰值
    prominence = float(bone_area_smooth.max()) * 0.15

    valleys, valley_props = find_peaks(
        inverted,
        distance=min_distance_slices,
        prominence=max(1.0, prominence),
    )

    # 只保留在有骨质的区域内的谷值
    # 要求谷值两侧都有一定的骨质
    valid_valleys = []
    for v in sorted(valleys):
        # 左右各看 5 层
        left_area = float(bone_area_smooth[max(0, v - 5):v].max()) if v > 0 else 0
        right_area = float(bone_area_smooth[v + 1:min(nz, v + 6)].max()) if v < nz - 1 else 0
        if left_area > MIN_VERTEBRAL_BODY_PIXELS and right_area > MIN_VERTEBRAL_BODY_PIXELS:
            valid_valleys.append(int(v))

    info["n_valleys_raw"] = len(valleys)
    info["n_valleys_valid"] = len(valid_valleys)
    info["valley_positions"] = valid_valleys

    if len(valid_valleys) < 2:
        # 谷值太少，无法分割椎体
        # 退化：把整个有骨质的区域当作一整段脊柱
        has_bone = bone_area_smooth > MIN_VERTEBRAL_BODY_PIXELS * 0.3
        if not has_bone.any():
            return [], {"error": "no_bone_region", **info}
        bone_region = np.where(has_bone)[0]
        return [{"z_start": int(bone_region.min()), "z_end": int(bone_region.max()),
                 "z_center": float((bone_region.min() + bone_region.max()) / 2),
                 "area_mean": float(bone_area_smooth[bone_region].mean()),
                 "index": 0}], {**info, "method": "single_segment"}

    # --- Step 4: 用谷值分割椎体 ---
    vertebrae = []
    # 第一个椎体：从有骨质的开始到第一个谷值
    has_bone = bone_area_smooth > MIN_VERTEBRAL_BODY_PIXELS * 0.3
    if has_bone.any():
        first_bone = int(np.where(has_bone)[0].min())
    else:
        first_bone = 0

    boundaries = [first_bone] + valid_valleys
    # 最后一个椎体：从最后一个谷值到骨质消失
    if has_bone.any():
        last_bone = int(np.where(has_bone)[0].max())
    else:
        last_bone = nz - 1
    boundaries.append(last_bone)

    for i in range(len(boundaries) - 1):
        z_start = boundaries[i]
        z_end = boundaries[i + 1]
        if z_end - z_start < 2:
            continue
        segment = bone_area_smooth[z_start:z_end + 1]
        if segment.max() < MIN_VERTEBRAL_BODY_PIXELS * 0.3:
            continue
        vertebrae.append({
            "z_start": z_start,
            "z_end": z_end,
            "z_center": float((z_start + z_end) / 2),
            "height_mm": float((z_end - z_start) * dz),
            "area_mean": float(segment.mean()),
            "area_max": float(segment.max()),
            "index": len(vertebrae),
        })

    info["n_vertebrae"] = len(vertebrae)
    info["method"] = "valley_segmentation"
    return vertebrae, info


def _estimate_spine_center(vol: np.ndarray) -> tuple[float, float]:
    """在没有 spine_mask 的情况下，估算脊柱的中心 y, x。"""
    nz, ny, nx = vol.shape
    # 在中间 30% 层取平均，找高密度骨质的质心
    z_start = int(nz * 0.35)
    z_end = int(nz * 0.65)
    bone_sum_yx = np.zeros((ny, nx), dtype=np.float64)
    for z in range(z_start, z_end):
        bone_sum_yx += (vol[z] > CORTICAL_BONE_HU).astype(np.float64)

    if bone_sum_yx.max() == 0:
        # fallback：假设后正中
        return float(ny * 0.75), float(nx * 0.50)

    if ndi is not None:
        # 找最大的骨质团块（应该是脊柱）
        binary = bone_sum_yx > bone_sum_yx.max() * 0.3
        labels, n = ndi.label(binary)
        if n > 0:
            counts = np.bincount(labels.ravel())
            counts[0] = 0
            biggest = int(counts.argmax())
            coords = np.argwhere(labels == biggest)
            return float(coords[:, 0].mean()), float(coords[:, 1].mean())

    coords = np.argwhere(bone_sum_yx > bone_sum_yx.max() * 0.3)
    if len(coords) == 0:
        return float(ny * 0.75), float(nx * 0.50)
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def classify_vertebrae_direction(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    vertebrae: list[dict],
) -> tuple[str, dict]:
    """判断 z 轴方向：z 增大是头端(superior)还是足端(inferior)。

    利用肺组织检测：如果 z 较小端有肺，则 z 从上到下（z 增大 = 向足端）。
    """
    nz, ny, nx = vol.shape
    info: dict = {}

    # 检测肺组织在 z 的哪一端
    lung_count_low = 0  # z 较小端
    lung_count_high = 0  # z 较大端
    check_range = max(5, nz // 10)

    for z in range(min(check_range, nz)):
        lung_count_low += int((vol[z] < -400).sum())
    for z in range(max(0, nz - check_range), nz):
        lung_count_high += int((vol[z] < -400).sum())

    info["lung_count_low_z"] = lung_count_low
    info["lung_count_high_z"] = lung_count_high

    if lung_count_low > lung_count_high * 2:
        direction = "z_increasing_is_inferior"  # z=0 头端，z 增大向下
    elif lung_count_high > lung_count_low * 2:
        direction = "z_increasing_is_superior"  # z=0 足端，z 增大向上
    else:
        # 不确定 → 看椎体大小趋势（腰椎比胸椎大）
        if len(vertebrae) >= 3:
            first_area = vertebrae[0]["area_mean"]
            last_area = vertebrae[-1]["area_mean"]
            if last_area > first_area * 1.3:
                direction = "z_increasing_is_inferior"  # 后面的椎体更大 = 腰椎
            elif first_area > last_area * 1.3:
                direction = "z_increasing_is_superior"
            else:
                direction = "unknown"
        else:
            direction = "unknown"

    info["direction"] = direction
    return direction, info


def locate_l3_upper_border(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    vertebrae: list[dict] | None = None,
    spine_mask: np.ndarray | None = None,
    diaphragm_z: int | None = None,
) -> tuple[int | None, dict]:
    """定位 L3 椎体上缘。

    策略（按优先级）：
    1. 如果检测到了多个椎体，从膈肌端向下数：
       - T12 ≈ 膈肌附近第一个椎体
       - L1 = T12 下一个
       - L2 = L1 下一个
       - L3 上缘 = L2 下一个椎体的起始位置
    2. 如果椎体检测失败但有膈肌位置，用距离估算
    3. 最终 fallback

    参数：
        vol: z-y-x HU 体积
        spacing_zyx: 体素间距
        vertebrae: detect_vertebrae 的输出
        spine_mask: 可选
        diaphragm_z: 膈肌 z 位置

    返回：
        l3_z: L3 上缘的 z index，或 None
        info: 调试信息
    """
    nz = vol.shape[0]
    dz = float(spacing_zyx[0])
    info: dict = {"dz_mm": dz}

    # --- 椎体检测（如果没有传入）---
    if vertebrae is None:
        vertebrae, detect_info = detect_vertebrae(vol, spacing_zyx, spine_mask)
        info["vertebra_detection"] = detect_info

    if len(vertebrae) < 2:
        # 椎体检测失败 → 用膈肌距离估算
        return _fallback_l3_from_diaphragm(vol, spacing_zyx, diaphragm_z, info)

    # --- 判断 z 方向 ---
    direction, dir_info = classify_vertebrae_direction(vol, spacing_zyx, vertebrae)
    info["z_direction"] = dir_info

    # --- 找到膈肌附近的椎体（T12）---
    if diaphragm_z is not None:
        # 找离膈肌最近的椎体 → 大致是 T11 或 T12
        t12_candidates = []
        for v in vertebrae:
            dist = abs(v["z_center"] - diaphragm_z) * dz
            t12_candidates.append((dist, v))
        t12_candidates.sort(key=lambda x: x[0])
        diaphragm_vertebra = t12_candidates[0][1]
        diaphragm_vertebra_idx = diaphragm_vertebra["index"]
        info["diaphragm_vertebra_idx"] = diaphragm_vertebra_idx
        info["diaphragm_vertebra_dist_mm"] = t12_candidates[0][0]
    else:
        # 没有膈肌信息 → 用肺组织判断哪端是头端
        if direction == "z_increasing_is_inferior":
            diaphragm_vertebra_idx = 0  # z 最小端 = 头端，第一个椎体 ≈ 最高的胸椎
        elif direction == "z_increasing_is_superior":
            diaphragm_vertebra_idx = len(vertebrae) - 1
        else:
            # 不确定方向 → 取面积较小的一端（胸椎比腰椎小）
            if vertebrae[0]["area_mean"] < vertebrae[-1]["area_mean"]:
                diaphragm_vertebra_idx = 0
            else:
                diaphragm_vertebra_idx = len(vertebrae) - 1
        info["diaphragm_vertebra_idx"] = diaphragm_vertebra_idx
        info["diaphragm_vertebra_method"] = "area_heuristic"

    # --- 从 T12 向腰椎方向数 3 个椎体 → L3 上缘 ---
    # T12 ≈ 膈肌处椎体
    # 向腰椎方向 = z 增大方向（如果 z_increasing_is_inferior）
    #              或 z 减小方向（如果 z_increasing_is_superior）

    if direction == "z_increasing_is_inferior":
        # z 增大 = 向下 = 向腰椎
        # T12 之后第3个椎体 = L3
        l3_idx = diaphragm_vertebra_idx + 3
        if l3_idx < len(vertebrae):
            l3_z = vertebrae[l3_idx]["z_start"]
            info.update({"l3_vertebra_idx": l3_idx, "l3_z": l3_z, "method": "count_from_diaphragm"})
            return l3_z, info
        else:
            # 椎体不够 → 用最后一个椎体的下端
            l3_z = vertebrae[-1]["z_end"]
            info.update({"l3_z": l3_z, "method": "last_vertebra_end", "vertebrae_short": True})
            return l3_z, info

    elif direction == "z_increasing_is_superior":
        # z 增大 = 向上 = 向胸椎。从膈肌椎体向 z 减小方向数
        l3_idx = diaphragm_vertebra_idx - 3
        if l3_idx >= 0:
            l3_z = vertebrae[l3_idx]["z_end"]  # z_end 是较大的 z = 上缘（因为 z 增大是向上）
            info.update({"l3_vertebra_idx": l3_idx, "l3_z": l3_z, "method": "count_from_diaphragm"})
            return l3_z, info
        else:
            l3_z = vertebrae[0]["z_start"]
            info.update({"l3_z": l3_z, "method": "first_vertebra_start", "vertebrae_short": True})
            return l3_z, info

    else:
        # 方向不确定 → 取所有椎体约 65% 处作为 L3 估算
        total_span = vertebrae[-1]["z_end"] - vertebrae[0]["z_start"]
        l3_z = int(vertebrae[0]["z_start"] + total_span * 0.65)
        info.update({"l3_z": l3_z, "method": "65_percent_fallback"})
        return l3_z, info


def _fallback_l3_from_diaphragm(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    diaphragm_z: int | None,
    info: dict,
) -> tuple[int | None, dict]:
    """椎体检测失败时的 fallback。"""
    nz = vol.shape[0]
    dz = float(spacing_zyx[0])

    if diaphragm_z is not None:
        # T12 ≈ 膈肌下方 15mm，L3 上缘 ≈ T12 + 75mm
        offset_mm = DIAPHRAGM_TO_T12_MM + T12_TO_L3_MM
        offset_slices = int(round(offset_mm / max(0.1, dz)))
        mid_z = nz // 2
        if diaphragm_z < mid_z:
            l3_z = min(nz - 1, diaphragm_z + offset_slices)
        else:
            l3_z = max(0, diaphragm_z - offset_slices)
        info.update({"l3_z": l3_z, "method": "diaphragm_offset_fallback", "offset_mm": offset_mm})
        return l3_z, info

    # 都没有 → 用 volume 的 65%
    l3_z = int(nz * 0.65)
    info.update({"l3_z": l3_z, "method": "volume_fraction_fallback"})
    return l3_z, info


def standardize_z_range(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    spine_mask: np.ndarray | None = None,
    diaphragm_z: int | None = None,
    margin_mm: float = 20.0,
) -> tuple[int, int, dict]:
    """统一接口：返回 (z_start, z_end, info)。

    z_start = 膈肌层（上界）
    z_end = L3 上缘层（下界）
    """
    nz = vol.shape[0]
    dz = float(spacing_zyx[0])
    info: dict = {}

    # 膈肌检测
    if diaphragm_z is None:
        diaphragm_z = _detect_diaphragm_slice(vol)
    info["diaphragm_z"] = diaphragm_z

    # 椎体检测 + L3 定位
    vertebrae, vert_info = detect_vertebrae(vol, spacing_zyx, spine_mask)
    info["vertebra_detection"] = vert_info

    l3_z, l3_info = locate_l3_upper_border(vol, spacing_zyx, vertebrae, spine_mask, diaphragm_z)
    info["l3_detection"] = l3_info

    if diaphragm_z is not None and l3_z is not None:
        z_start = min(diaphragm_z, l3_z)
        z_end = max(diaphragm_z, l3_z)
    elif diaphragm_z is not None:
        z_start = min(diaphragm_z, nz - 1)
        z_end = max(diaphragm_z, nz - 1)
    elif l3_z is not None:
        z_start = 0
        z_end = l3_z
    else:
        z_start = int(nz * 0.25)
        z_end = int(nz * 0.80)

    # 安全边距
    margin_slices = max(1, int(round(margin_mm / max(0.1, dz))))
    z_start = max(0, z_start - margin_slices)
    z_end = min(nz - 1, z_end + margin_slices)

    info.update({
        "z_start": z_start,
        "z_end": z_end,
        "z_range_slices": z_end - z_start,
        "z_range_mm": float((z_end - z_start) * dz),
        "n_vertebrae_detected": len(vertebrae),
        "vertebrae": vertebrae,
    })
    return z_start, z_end, info


def _detect_diaphragm_slice(vol: np.ndarray) -> int | None:
    """检测膈肌层面（与 v3e 一致）。"""
    if ndi is None:
        return None
    nz, ny, nx = vol.shape
    lung_fracs = []
    for z in range(nz):
        body = vol[z] > -500.0
        body = ndi.binary_fill_holes(body) if ndi else body
        body_area = int(body.sum())
        if body_area < int(ny * nx * 0.05):
            lung_fracs.append(0.0)
            continue
        lung_fracs.append(float(((vol[z] < -400.0) & body).sum()) / float(body_area))
    lung_fracs_arr = np.asarray(lung_fracs, dtype=np.float32)
    has_lung = lung_fracs_arr > 0.03
    if not has_lung.any():
        return None
    lung_idx = np.where(has_lung)[0]
    lung_center = (lung_idx.min() + lung_idx.max()) / 2.0
    return int(lung_idx.max()) if lung_center < nz // 2 else int(lung_idx.min())