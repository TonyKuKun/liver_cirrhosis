"""椎体逐节检测 + L3 上缘定位 + Z 轴标定。

v2 重写：
- 逐层追踪椎体中心（不用全局中位数）
- 用椎体横截面积的平滑曲线 + 梯度分析分割椎体
- 用腰椎面积渐增的特征定位腰段
- 多信号融合：膈肌距离 + 椎体大小 + 增强血管密度

用法：
    from vertebra_detector import standardize_z_range
    z_start, z_end, info = standardize_z_range(vol, spacing_zyx, spine_mask)
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


CORTICAL_BONE_HU = 400.0
VERTEBRAL_BODY_HU_LOW = 150.0
VERTEBRAL_BODY_HU_HIGH = 1200.0
MIN_VERTEBRAL_AREA_MM2 = 200.0
LUNG_HU_THRESHOLD = -400.0
LUNG_FRACTION_THRESHOLD = 0.03
PORTAL_VEIN_HU_RANGE = (70.0, 360.0)


# =========================================================================
# 1. 逐层脊柱追踪
# =========================================================================

def _track_spine_per_slice(
    vol: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    spine_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """逐层追踪脊柱椎体：返回每层的中心坐标和截面积。

    与 v1 的区别：不用"最 posterior 的骨块 = 脊柱"，
    而是从已知位置出发，利用椎体位置的连续性逐层追踪。

    返回：
        center_y, center_x: shape (nz,)，无椎体则 NaN
        area_mm2: shape (nz,)
        info
    """
    nz, ny, nx = vol.shape
    sy, sx = float(spacing_zyx[1]), float(spacing_zyx[2])
    pixel_area = sy * sx

    center_y = np.full(nz, np.nan, dtype=np.float32)
    center_x = np.full(nz, np.nan, dtype=np.float32)
    area_mm2 = np.zeros(nz, dtype=np.float32)

    mid_z = nz // 2
    init_cy, init_cx = _find_initial_spine_center(vol, spacing_zyx, spine_mask, mid_z)

    if init_cy is None:
        return center_y, center_x, area_mm2, {"error": "no_initial_spine_center"}

    search_ry = max(5, int(round(35.0 / sy)))
    search_rx = max(5, int(round(35.0 / sx)))

    for direction in (1, -1):
        prev_cy, prev_cx = init_cy, init_cx
        start = mid_z if direction == 1 else mid_z - 1
        end = nz if direction == 1 else -1

        for z in range(start, end, direction):
            cy_int, cx_int = int(round(prev_cy)), int(round(prev_cx))
            y0 = max(0, cy_int - search_ry)
            y1 = min(ny, cy_int + search_ry + 1)
            x0 = max(0, cx_int - search_rx)
            x1 = min(nx, cx_int + search_rx + 1)

            roi = vol[z, y0:y1, x0:x1]
            bone = (roi >= VERTEBRAL_BODY_HU_LOW) & (roi <= VERTEBRAL_BODY_HU_HIGH)

            if ndi is not None:
                labels, n_labels = ndi.label(bone)
                if n_labels == 0:
                    continue
                counts = np.bincount(labels.ravel())
                counts[0] = 0
                biggest = int(counts.argmax())
                if counts[biggest] < max(10, int(MIN_VERTEBRAL_AREA_MM2 / pixel_area * 0.3)):
                    continue
                bone = (labels == biggest)

            bone_area = float(bone.sum()) * pixel_area
            if bone_area < MIN_VERTEBRAL_AREA_MM2 * 0.3:
                continue

            coords = np.argwhere(bone)
            local_cy = float(coords[:, 0].mean())
            local_cx = float(coords[:, 1].mean())
            global_cy = local_cy + y0
            global_cx = local_cx + x0

            center_y[z] = global_cy
            center_x[z] = global_cx
            area_mm2[z] = bone_area

            prev_cy = 0.7 * prev_cy + 0.3 * global_cy
            prev_cx = 0.7 * prev_cx + 0.3 * global_cx

    valid = ~np.isnan(center_y)
    info = {
        "valid_slices": int(valid.sum()),
        "total_slices": nz,
        "init_center_yz": [float(init_cy), float(init_cx)],
    }
    return center_y, center_x, area_mm2, info


def _find_initial_spine_center(vol, spacing_zyx, spine_mask, target_z):
    nz, ny, nx = vol.shape
    if spine_mask is not None:
        search_range = range(max(0, target_z - 10), min(nz, target_z + 11))
        for z in search_range:
            if spine_mask[z].any():
                coords = np.argwhere(spine_mask[z])
                return float(coords[:, 0].mean()), float(coords[:, 1].mean())
    y_start = int(ny * 0.50)
    x_start, x_end = int(nx * 0.30), int(nx * 0.70)
    search_range = range(max(0, target_z - 15), min(nz, target_z + 16))
    for z in search_range:
        roi = vol[z, y_start:, x_start:x_end]
        bone = roi > CORTICAL_BONE_HU
        if bone.sum() < 20:
            continue
        coords = np.argwhere(bone)
        return float(coords[:, 0].mean()) + y_start, float(coords[:, 1].mean()) + x_start
    return None, None


# =========================================================================
# 2. 椎体分节
# =========================================================================

def _segment_vertebrae(area_mm2, center_y, spacing_zyx):
    if ndi is None:
        return [], {"error": "no_scipy"}
    nz = len(area_mm2)
    dz = float(spacing_zyx[0])
    info = {"dz_mm": dz}

    sigma = max(0.5, 1.5 / max(0.1, dz))
    smoothed = ndi.gaussian_filter1d(area_mm2.astype(np.float64), sigma=sigma).astype(np.float32)

    threshold = max(MIN_VERTEBRAL_AREA_MM2 * 0.4, float(smoothed.max()) * 0.15)
    valid = smoothed > threshold
    if not valid.any():
        return [], {**info, "error": "no_valid_bone_region"}

    segments = _find_continuous_segments(valid)
    info["n_bone_segments"] = len(segments)
    if not segments:
        return [], {**info, "error": "no_segments"}

    longest = max(segments, key=lambda s: s[1] - s[0])
    seg_start, seg_end = longest
    seg_area = smoothed[seg_start:seg_end + 1]

    min_disc_gap_slices = max(2, int(round(12.0 / max(0.1, dz))))
    half_w = min_disc_gap_slices

    disc_positions = []
    for i in range(half_w, len(seg_area) - half_w):
        val = seg_area[i]
        left_max = seg_area[max(0, i - half_w):i].max()
        right_max = seg_area[i + 1:min(len(seg_area), i + half_w + 1)].max()
        if val < left_max * 0.80 and val < right_max * 0.80:
            disc_positions.append(seg_start + i)

    filtered_discs = _filter_close_valleys(disc_positions, smoothed, min_disc_gap_slices)
    info["n_disc_positions"] = len(filtered_discs)

    boundaries = [seg_start] + filtered_discs + [seg_end]
    vertebrae = []
    for i in range(len(boundaries) - 1):
        z_s, z_e = boundaries[i], boundaries[i + 1]
        height_mm = float((z_e - z_s) * dz)
        if height_mm < 10.0:
            continue
        seg = smoothed[z_s:z_e + 1]
        vertebrae.append({
            "z_start": z_s, "z_end": z_e,
            "z_center": float((z_s + z_e) / 2),
            "height_mm": round(height_mm, 1),
            "area_mean_mm2": round(float(seg.mean()), 1),
            "area_max_mm2": round(float(seg.max()), 1),
            "index": len(vertebrae),
        })

    info["n_vertebrae"] = len(vertebrae)
    return vertebrae, info


def _find_continuous_segments(valid):
    segments = []
    in_seg, start = False, 0
    for i in range(len(valid)):
        if valid[i] and not in_seg:
            start = i; in_seg = True
        elif not valid[i] and in_seg:
            if i - start >= 3:
                segments.append((start, i - 1))
            in_seg = False
    if in_seg and len(valid) - start >= 3:
        segments.append((start, len(valid) - 1))
    return segments


def _filter_close_valleys(positions, curve, min_gap):
    if not positions:
        return []
    filtered = [positions[0]]
    for p in positions[1:]:
        if p - filtered[-1] >= min_gap:
            filtered.append(p)
        elif curve[p] < curve[filtered[-1]]:
            filtered[-1] = p
    return filtered


# =========================================================================
# 3. L3 定位：多信号融合
# =========================================================================

def _detect_diaphragm_slice(vol):
    if ndi is None:
        return None, None
    nz, ny, nx = vol.shape
    lung_fracs = np.zeros(nz, dtype=np.float32)
    for z in range(nz):
        body = vol[z] > -500.0
        body = ndi.binary_fill_holes(body)
        body_area = int(body.sum())
        if body_area < int(ny * nx * 0.05):
            continue
        lung_fracs[z] = float(((vol[z] < LUNG_HU_THRESHOLD) & body).sum()) / float(body_area)
    has_lung = lung_fracs > LUNG_FRACTION_THRESHOLD
    if not has_lung.any():
        return None, None
    lung_idx = np.where(has_lung)[0]
    lung_center = (lung_idx.min() + lung_idx.max()) / 2.0
    mid_z = nz // 2
    if lung_center < mid_z:
        return int(lung_idx.max()), "z_down"
    else:
        return int(lung_idx.min()), "z_up"


def _estimate_portal_vessel_z_peak(vol, spacing_zyx):
    if ndi is None:
        return None
    nz, ny, nx = vol.shape
    y_end = int(ny * 0.65)
    x_start, x_end = int(nx * 0.15), int(nx * 0.85)
    vessel_density = np.zeros(nz, dtype=np.float32)
    for z in range(nz):
        roi = vol[z, :y_end, x_start:x_end]
        vessel_density[z] = float(((roi >= PORTAL_VEIN_HU_RANGE[0]) &
                                    (roi <= PORTAL_VEIN_HU_RANGE[1])).sum())
    smoothed = ndi.gaussian_filter1d(vessel_density, sigma=5.0)
    if smoothed.max() <= 0:
        return None
    threshold = smoothed.max() * 0.6
    high_density = smoothed > threshold
    if not high_density.any():
        return int(smoothed.argmax())
    indices = np.where(high_density)[0]
    weights = smoothed[indices]
    return int(round(float(np.average(indices, weights=weights))))


def locate_l3_upper_border(vol, spacing_zyx, vertebrae, diaphragm_z, z_direction, area_mm2=None):
    nz = vol.shape[0]
    dz = float(spacing_zyx[0])
    info = {}

    l3_from_counting = None
    if diaphragm_z is not None and len(vertebrae) >= 4 and z_direction is not None:
        l3_from_counting, count_info = _l3_by_counting(vertebrae, diaphragm_z, z_direction, dz)
        info["counting"] = count_info

    l3_from_size = None
    if len(vertebrae) >= 5 and area_mm2 is not None:
        l3_from_size, size_info = _l3_by_size_transition(vertebrae, area_mm2, z_direction, dz)
        info["size_transition"] = size_info

    portal_peak = _estimate_portal_vessel_z_peak(vol, spacing_zyx)
    l3_from_portal = None
    if portal_peak is not None:
        offset_slices = int(round(55.0 / max(0.1, dz)))
        if z_direction == "z_down":
            l3_from_portal = min(nz - 1, portal_peak + offset_slices)
        elif z_direction == "z_up":
            l3_from_portal = max(0, portal_peak - offset_slices)
        else:
            a = min(nz - 1, portal_peak + offset_slices)
            b = max(0, portal_peak - offset_slices)
            l3_from_portal = a if abs(a - nz // 2) < abs(b - nz // 2) else b
        info["portal_peak_z"] = portal_peak
        info["l3_from_portal"] = l3_from_portal

    estimates = []
    if l3_from_counting is not None:
        estimates.append(("counting", l3_from_counting, 1.0))
    if l3_from_size is not None:
        estimates.append(("size", l3_from_size, 0.7))
    if l3_from_portal is not None:
        estimates.append(("portal", l3_from_portal, 0.5))

    if not estimates:
        if diaphragm_z is not None:
            offset = int(round(90.0 / max(0.1, dz)))
            if z_direction == "z_down":
                l3_z = min(nz - 1, diaphragm_z + offset)
            elif z_direction == "z_up":
                l3_z = max(0, diaphragm_z - offset)
            else:
                l3_z = int(nz * 0.65)
            info.update({"method": "diaphragm_offset_fallback", "l3_z": l3_z})
            return l3_z, info
        info.update({"method": "volume_fraction_fallback", "l3_z": int(nz * 0.65)})
        return int(nz * 0.65), info

    total_weight = sum(w for _, _, w in estimates)
    weighted_z = sum(z * w for _, z, w in estimates) / total_weight
    l3_z = max(0, min(nz - 1, int(round(weighted_z))))
    info.update({"method": "multi_signal_fusion",
                 "estimates": [{"signal": n, "z": z, "weight": w} for n, z, w in estimates],
                 "l3_z": l3_z})
    return l3_z, info


def _l3_by_counting(vertebrae, diaphragm_z, z_direction, dz):
    info = {}
    dists = [(abs(v["z_center"] - diaphragm_z) * dz, v) for v in vertebrae]
    dists.sort(key=lambda x: x[0])
    nearest_dist_mm = dists[0][0]
    t12_idx = dists[0][1]["index"]
    info.update({"t12_candidate_idx": t12_idx, "t12_dist_mm": round(nearest_dist_mm, 1)})
    if nearest_dist_mm > 50.0:
        return None, {**info, "warning": "t12_too_far"}
    if z_direction == "z_down":
        l3_idx = t12_idx + 3
        if l3_idx < len(vertebrae):
            return vertebrae[l3_idx]["z_start"], {**info, "l3_idx": l3_idx}
        return vertebrae[-1]["z_end"], {**info, "l3_idx": "clamped_last"}
    elif z_direction == "z_up":
        l3_idx = t12_idx - 3
        if l3_idx >= 0:
            return vertebrae[l3_idx]["z_end"], {**info, "l3_idx": l3_idx}
        return vertebrae[0]["z_start"], {**info, "l3_idx": "clamped_first"}
    return None, {**info, "error": "unknown_direction"}


def _l3_by_size_transition(vertebrae, area_mm2, z_direction, dz):
    info = {}
    if len(vertebrae) < 5:
        return None, {"error": "too_few"}
    areas = np.array([v["area_mean_mm2"] for v in vertebrae])
    median_area = float(np.median(areas))
    lumbar_start_idx = None
    for i in range(len(areas)):
        if areas[i] > median_area * 1.15:
            if i + 2 < len(areas) and all(areas[j] > median_area for j in range(i, min(i + 3, len(areas)))):
                lumbar_start_idx = i
                break
    if lumbar_start_idx is None:
        return None, {"error": "no_transition"}
    l3_idx = lumbar_start_idx + 2
    l3_z = vertebrae[l3_idx]["z_start"] if l3_idx < len(vertebrae) else vertebrae[-1]["z_end"]
    info.update({"lumbar_start_idx": lumbar_start_idx, "l3_idx": l3_idx, "l3_z": l3_z})
    return l3_z, info


# =========================================================================
# 4. 统一接口
# =========================================================================

def detect_vertebrae(vol, spacing_zyx, spine_mask=None):
    """返回 (vertebrae_list, area_mm2_per_slice, info)。"""
    center_y, center_x, area_mm2, track_info = _track_spine_per_slice(vol, spacing_zyx, spine_mask)
    if track_info.get("error"):
        return [], area_mm2, track_info
    vertebrae, seg_info = _segment_vertebrae(area_mm2, center_y, spacing_zyx)
    info = {"tracking": track_info, "segmentation": seg_info, "n_vertebrae": len(vertebrae)}
    return vertebrae, area_mm2, info


def standardize_z_range(vol, spacing_zyx, spine_mask=None, diaphragm_z=None, margin_mm=20.0):
    """统一接口：返回 (z_start, z_end, info)。"""
    nz = vol.shape[0]
    dz = float(spacing_zyx[0])
    info = {}

    z_direction = None
    if diaphragm_z is None:
        diaphragm_z, z_direction = _detect_diaphragm_slice(vol)
    info["diaphragm_z"] = diaphragm_z
    info["z_direction"] = z_direction

    vertebrae, area_mm2, vert_info = detect_vertebrae(vol, spacing_zyx, spine_mask)
    info["vertebra_detection"] = vert_info

    l3_z, l3_info = locate_l3_upper_border(vol, spacing_zyx, vertebrae, diaphragm_z, z_direction, area_mm2)
    info["l3_detection"] = l3_info

    if diaphragm_z is not None and l3_z is not None:
        z_start, z_end = min(diaphragm_z, l3_z), max(diaphragm_z, l3_z)
    elif diaphragm_z is not None:
        default_range = int(round(180.0 / max(0.1, dz)))
        if z_direction == "z_down":
            z_start, z_end = diaphragm_z, min(nz - 1, diaphragm_z + default_range)
        elif z_direction == "z_up":
            z_start, z_end = max(0, diaphragm_z - default_range), diaphragm_z
        else:
            z_start, z_end = max(0, diaphragm_z - default_range // 2), min(nz - 1, diaphragm_z + default_range // 2)
    elif l3_z is not None:
        z_start, z_end = 0, l3_z
    else:
        z_start, z_end = int(nz * 0.20), int(nz * 0.82)

    margin_slices = max(1, int(round(margin_mm / max(0.1, dz))))
    z_start = max(0, z_start - margin_slices)
    z_end = min(nz - 1, z_end + margin_slices)

    info.update({
        "z_start": z_start, "z_end": z_end,
        "z_range_slices": z_end - z_start,
        "z_range_mm": round(float((z_end - z_start) * dz), 1),
        "vertebrae": vertebrae,
    })
    return z_start, z_end, info