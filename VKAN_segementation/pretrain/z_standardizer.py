"""Z 轴标定：横膈膜（上界）+ 髂骨（下界）。

删繁就简——不数椎体，不定位第几腰椎。
横膈膜：肺组织消失的层面。
髂骨：骨质横向宽度突然增大的层面（从窄脊柱变成宽髂翼）。

用法：
    from z_standardizer import standardize_z_range
    z_start, z_end, info = standardize_z_range(vol, spacing_zyx)
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


LUNG_HU_THRESHOLD = -400.0
LUNG_FRACTION_THRESHOLD = 0.03
BONE_HU_THRESHOLD = 250.0
# 脊柱宽度大约 30-50mm，髂翼出现后总骨宽跳到 100-200mm+
# 当某层骨质横向跨度超过此值（mm），认为进入了髂骨区域
ILIAC_WIDTH_THRESHOLD_MM = 90.0
# 髂骨确认：连续多少层宽度都大 → 确定是髂骨而非偶发噪声
ILIAC_CONFIRM_SLICES = 3
MIN_VALID_Z_RANGE_MM = 120.0
FALLBACK_MIN_Z_RANGE_MM = 150.0
PORTAL_VEIN_HU_LOW = 80.0
PORTAL_VEIN_HU_HIGH = 350.0


def standardize_z_range(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    margin_mm: float = 20.0,
) -> tuple[int, int, dict]:
    """返回 (z_start, z_end, info)。

    z_start = 横膈膜层（如果有胸腔）或 volume 顶部附近
    z_end = 髂骨首次出现的层（如果检测到）或 volume 底部附近
    """
    nz = vol.shape[0]
    dz = float(spacing_zyx[0])
    info: dict = {"nz": nz, "dz_mm": round(dz, 3)}

    # === 1. 检测横膈膜（上界）===
    diaphragm_z, z_direction = _detect_diaphragm(vol)
    info["diaphragm_z"] = diaphragm_z
    info["z_direction"] = z_direction

    # === 2. 检测髂骨（下界）===
    iliac_z, iliac_info = _detect_iliac_bone(vol, spacing_zyx, z_direction)
    info["iliac"] = iliac_info

    # === 3. 确定范围 ===
    if z_direction == "z_down":
        # z 增大 = 向足端：上界=膈肌(小z), 下界=髂骨(大z)
        z_top = diaphragm_z if diaphragm_z is not None else _safe_top(vol, z_direction)
        z_bottom = iliac_z if iliac_z is not None else _safe_bottom(nz, z_direction)
        z_start, z_end = z_top, z_bottom
    elif z_direction == "z_up":
        # z 增大 = 向头端：上界=膈肌(大z), 下界=髂骨(小z)
        z_top = diaphragm_z if diaphragm_z is not None else _safe_top(vol, z_direction)
        z_bottom = iliac_z if iliac_z is not None else _safe_bottom(nz, z_direction)
        z_start, z_end = z_bottom, z_top
    else:
        # 方向不确定（纯腹部CT，无肺无髂骨）→ 保守取中间 60%
        z_start = int(nz * 0.10)
        z_end = int(nz * 0.85)
        if iliac_z is not None:
            # 有髂骨但无膈肌 → 从顶部到髂骨
            z_start = max(0, int(nz * 0.02))
            z_end = iliac_z

    # 确保 start < end
    if z_start > z_end:
        z_start, z_end = z_end, z_start

    # 安全边距
    margin_slices = max(1, int(round(margin_mm / max(0.1, dz))))
    z_start = max(0, z_start - margin_slices)
    z_end = min(nz - 1, z_end + margin_slices)

    validated, fallback_reason = _validate_z_range(
        z_start, z_end, dz, diaphragm_z, iliac_z, iliac_info,
    )
    portal_peak_z = _estimate_portal_density_peak(vol)
    if not validated:
        z_start, z_end, fb_info = _fallback_portal_z_range(nz, dz, portal_peak_z)
        info["z_fallback_reason"] = fallback_reason
        info["z_fallback"] = fb_info

    info.update({
        "z_start": z_start,
        "z_end": z_end,
        "z_range_slices": z_end - z_start,
        "z_range_mm": round(float((z_end - z_start) * dz), 1),
        "z_range_validated": bool(validated),
        "portal_density_peak_z": portal_peak_z,
    })
    return z_start, z_end, info


def _validate_z_range(
    z_start: int,
    z_end: int,
    dz: float,
    diaphragm_z: int | None,
    iliac_z: int | None,
    iliac_info: dict,
) -> tuple[bool, str | None]:
    z_range_mm = float(max(0, z_end - z_start) * dz)
    if z_range_mm < MIN_VALID_Z_RANGE_MM:
        return False, "z_range_too_short"
    if diaphragm_z is not None and iliac_z is not None:
        if abs(int(diaphragm_z) - int(iliac_z)) * dz < MIN_VALID_Z_RANGE_MM:
            return False, "diaphragm_iliac_too_close"
    if iliac_info.get("status") == "detected":
        width = float(iliac_info.get("width_at_iliac_mm", 0.0))
        if width < ILIAC_WIDTH_THRESHOLD_MM * 0.7:
            return False, "iliac_edge_width_unstable"
    return True, None


def _estimate_portal_density_peak(vol: np.ndarray) -> int | None:
    if vol.ndim != 3:
        return None
    nz, ny, nx = vol.shape
    if nz <= 0:
        return None
    y_end = max(1, int(ny * 0.72))
    x_start = int(nx * 0.12)
    x_end = max(int(nx * 0.88), x_start + 1)
    density = np.zeros(nz, dtype=np.float32)
    for z in range(nz):
        roi = vol[z, :y_end, x_start:x_end]
        density[z] = float(((roi >= PORTAL_VEIN_HU_LOW) & (roi <= PORTAL_VEIN_HU_HIGH)).sum())
    if ndi is not None:
        density = ndi.gaussian_filter1d(density, sigma=max(1.0, nz / 80.0))
    if float(density.max()) <= 0.0:
        return None
    high = density >= float(density.max()) * 0.65
    if high.any():
        idx = np.where(high)[0]
        weights = density[idx]
        return int(round(float(np.average(idx, weights=weights))))
    return int(np.argmax(density))


def _fallback_portal_z_range(
    nz: int,
    dz: float,
    portal_peak_z: int | None,
) -> tuple[int, int, dict]:
    min_slices = max(1, int(round(FALLBACK_MIN_Z_RANGE_MM / max(0.1, dz))))
    min_slices = min(nz - 1, min_slices)
    if portal_peak_z is None:
        z_start = int(nz * 0.20)
        z_end = int(nz * 0.82)
        method = "body_fraction"
    else:
        center = max(0, min(nz - 1, int(portal_peak_z)))
        half = max(min_slices // 2, int(round(75.0 / max(0.1, dz))))
        z_start = center - half
        z_end = center + half
        method = "portal_density_peak"
    if z_start < 0:
        z_end -= z_start
        z_start = 0
    if z_end > nz - 1:
        z_start -= z_end - (nz - 1)
        z_end = nz - 1
    z_start = max(0, z_start)
    z_end = min(nz - 1, max(z_end, z_start + min_slices))
    if z_end > nz - 1:
        z_end = nz - 1
        z_start = max(0, z_end - min_slices)
    return z_start, z_end, {
        "method": method,
        "min_range_mm": FALLBACK_MIN_Z_RANGE_MM,
        "range_mm": round(float((z_end - z_start) * dz), 1),
    }


# =========================================================================
# 横膈膜检测
# =========================================================================

def _detect_diaphragm(vol: np.ndarray) -> tuple[int | None, str | None]:
    """检测横膈膜层面。

    返回 (diaphragm_z, z_direction)。
    z_direction: "z_down" = z增大向足端, "z_up" = z增大向头端。

    针对腹部CT的特殊处理：
    - 如果肺组织只出现在最顶部几层（不到总层数的 10%），
      说明这是腹部CT扫到了一点肺底，膈肌就在那几层附近。
    - 如果完全没有肺组织 → 返回 None（纯腹部扫描）。
    """
    if ndi is None:
        return None, None

    nz, ny, nx = vol.shape
    total_area = ny * nx

    # 逐层计算肺组织占比
    lung_fracs = np.zeros(nz, dtype=np.float32)
    for z in range(nz):
        # 用体表轮廓限制，避免体外空气干扰
        body = vol[z] > -500.0
        body = ndi.binary_fill_holes(body)
        body_area = int(body.sum())
        if body_area < int(total_area * 0.05):
            continue
        lung_fracs[z] = float(((vol[z] < LUNG_HU_THRESHOLD) & body).sum()) / float(body_area)

    has_lung = lung_fracs > LUNG_FRACTION_THRESHOLD
    if not has_lung.any():
        return None, None

    lung_indices = np.where(has_lung)[0]
    lung_start = int(lung_indices.min())
    lung_end = int(lung_indices.max())
    lung_span = lung_end - lung_start + 1
    lung_center = (lung_start + lung_end) / 2.0
    mid_z = nz // 2

    # 判断 z 方向
    if lung_center < mid_z:
        z_direction = "z_down"  # 肺在 z 小端 = z 从上到下
        diaphragm_z = lung_end
    else:
        z_direction = "z_up"  # 肺在 z 大端 = z 从下到上
        diaphragm_z = lung_start

    # 特殊处理：腹部CT只扫到一点肺底
    # 如果肺组织跨度不到总层数的 15%，且紧贴 volume 边缘
    if lung_span < nz * 0.15:
        if lung_start < nz * 0.10:
            # 肺紧贴 z=0 端 → 膈肌 = 肺消失处
            z_direction = "z_down"
            diaphragm_z = lung_end
        elif lung_end > nz * 0.90:
            # 肺紧贴 z=max 端
            z_direction = "z_up"
            diaphragm_z = lung_start

    return int(diaphragm_z), z_direction


def _safe_top(vol: np.ndarray, z_direction: str | None) -> int:
    """没有膈肌时的上界 fallback。"""
    nz = vol.shape[0]
    # 纯腹部CT：上界取 volume 顶部附近（留一点边距）
    if z_direction == "z_down":
        return max(0, int(nz * 0.02))
    elif z_direction == "z_up":
        return min(nz - 1, int(nz * 0.98))
    return int(nz * 0.10)


def _safe_bottom(nz: int, z_direction: str | None) -> int:
    """没有髂骨时的下界 fallback。"""
    if z_direction == "z_down":
        return int(nz * 0.85)
    elif z_direction == "z_up":
        return int(nz * 0.15)
    return int(nz * 0.85)


# =========================================================================
# 髂骨检测
# =========================================================================

def _detect_iliac_bone(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    z_direction: str | None,
) -> tuple[int | None, dict]:
    """检测髂骨首次出现的层面。

    原理非常简单：
    - 脊柱是一条窄的骨质柱（横向跨度 30-50mm）
    - 髂骨翼是一对宽大的翼状骨（横向跨度 150-250mm）
    - 当某层的骨质横向总跨度突然从 ~40mm 跳到 ~120mm+，就是髂骨出现了

    具体做法：
    1. 逐层测量骨质（>250HU）的最左到最右的跨度（mm）
    2. 找到跨度突然增大的层 = 髂骨上缘
    """
    nz, ny, nx = vol.shape
    sx = float(spacing_zyx[2])
    info: dict = {}

    # 逐层计算骨质横向跨度
    bone_width_mm = np.zeros(nz, dtype=np.float32)
    for z in range(nz):
        axial = vol[z]
        bone = axial > BONE_HU_THRESHOLD
        # 在 x 方向投影：哪些列有骨质
        x_has_bone = np.any(bone, axis=0)  # shape (nx,)
        if not x_has_bone.any():
            continue
        x_indices = np.where(x_has_bone)[0]
        width_pixels = int(x_indices[-1] - x_indices[0] + 1)
        bone_width_mm[z] = float(width_pixels) * sx

    # 平滑一下消除噪声
    if ndi is not None:
        bone_width_smooth = ndi.gaussian_filter1d(bone_width_mm, sigma=2.0)
    else:
        bone_width_smooth = bone_width_mm.copy()

    info["max_bone_width_mm"] = round(float(bone_width_smooth.max()), 1)

    # 如果最大宽度都没超过阈值 → 没有髂骨（CT 没扫到骨盆）
    if bone_width_smooth.max() < ILIAC_WIDTH_THRESHOLD_MM:
        info["status"] = "no_iliac_detected"
        return None, info

    # 找髂骨首次出现的位置
    # 从腹部方向（足端→头端 或 头端→足端）扫描，找到宽度首次超过阈值的层
    threshold = ILIAC_WIDTH_THRESHOLD_MM
    iliac_z = None

    if z_direction == "z_down":
        # z 增大 = 向足端，从最大 z 向上扫（从足端向头端找髂骨消失处）
        for z in range(nz - 1, -1, -1):
            if bone_width_smooth[z] >= threshold:
                # 确认：接下来的几层（继续向足端）是否也宽
                confirm_end = min(nz, z + ILIAC_CONFIRM_SLICES + 1)
                if all(bone_width_smooth[zz] >= threshold * 0.7 for zz in range(z, confirm_end)):
                    iliac_z = z
                    break
        # 如果找到了，向上找到髂骨上缘（宽度开始变窄的位置）
        if iliac_z is not None:
            while iliac_z > 0 and bone_width_smooth[iliac_z - 1] >= threshold * 0.6:
                iliac_z -= 1

    elif z_direction == "z_up":
        # z 增大 = 向头端，从最小 z 向上扫
        for z in range(0, nz):
            if bone_width_smooth[z] >= threshold:
                confirm_start = max(0, z - ILIAC_CONFIRM_SLICES)
                if all(bone_width_smooth[zz] >= threshold * 0.7 for zz in range(confirm_start, z + 1)):
                    iliac_z = z
                    break
        if iliac_z is not None:
            while iliac_z < nz - 1 and bone_width_smooth[iliac_z + 1] >= threshold * 0.6:
                iliac_z += 1

    else:
        # 方向未知：从两端各扫，取更合理的
        # 尝试两端，看哪端有髂骨
        # 髂骨在 CT 的足端（不论 z 方向如何）
        # 策略：找宽度最大的那段区域的边缘
        wide_slices = bone_width_smooth >= threshold
        if wide_slices.any():
            wide_indices = np.where(wide_slices)[0]
            # 髂骨是一个连续的宽骨区域，取它向腹部方向的边缘
            # 假设宽骨区域在 volume 的一端
            wide_center = (wide_indices.min() + wide_indices.max()) / 2.0
            if wide_center < nz / 2:
                # 宽骨在 z 小端 → z 小端是足端
                iliac_z = int(wide_indices.max())  # 取上缘（向腹部方向）
            else:
                iliac_z = int(wide_indices.min())

    if iliac_z is not None:
        info["status"] = "detected"
        info["iliac_z"] = iliac_z
        info["width_at_iliac_mm"] = round(float(bone_width_smooth[iliac_z]), 1)

        # 计算髂骨上缘处的脊柱宽度作为对比
        # 在髂骨之前（腹部方向）取几层看脊柱宽度
        if z_direction == "z_down" and iliac_z > 10:
            spine_width = float(np.median(bone_width_smooth[max(0, iliac_z - 20):iliac_z - 5]))
            info["spine_width_above_mm"] = round(spine_width, 1)
        elif z_direction == "z_up" and iliac_z < nz - 10:
            spine_width = float(np.median(bone_width_smooth[iliac_z + 5:min(nz, iliac_z + 20)]))
            info["spine_width_above_mm"] = round(spine_width, 1)
    else:
        info["status"] = "not_detected"

    return iliac_z, info
