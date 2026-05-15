"""阈值评分模块：基于解剖结构的门静脉分割质量评估。

核心思路：
- 好的阈值 → 血管是细管状分支结构，LPV/RPV 与肝实质清晰分离
- 阈值过低 → 肝实质被大片捞入，LPV/RPV 和肝脏融合成大团块
- 阈值过高 → 丢失弱增强分支，血管不完整

评分维度：
1. 完整性：所需血管分支是否都存在（向各个方向延伸）
2. 可分离性：LPV/RPV 区域是否为细管状而非大团块
3. 连通性：主连通域占比是否高
4. 尺寸合理性：体素数是否在合理范围

用法：
    from threshold_scorer import score_threshold, search_best_threshold
"""
from __future__ import annotations

import gc

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None


def _label_int32(mask):
    """int32 labeling to save memory."""
    out = np.empty(mask.shape, dtype=np.int32)
    n = ndi.label(mask, output=out)
    return out, int(n)


def _count_labels(labels: np.ndarray, n: int) -> np.ndarray:
    """逐层 bincount，避免 ravel().astype(int64) 的巨量内存分配。"""
    counts = np.zeros(n + 1, dtype=np.int64)
    for z in range(labels.shape[0]):
        c = np.bincount(labels[z].ravel(), minlength=n + 1)
        counts[:len(c)] += c
    return counts


# ---------------------------------------------------------------------------
# 门静脉系统的解剖方向先验
# ---------------------------------------------------------------------------
# 从 seed（门静脉主干汇合处）出发，各分支的大致方向：
#   z 方向：LPV/RPV 向上（头端），SMV 向下（足端）
#   y 方向：脾静脉偏 posterior，SMV 偏 anterior
#   x 方向：脾静脉向一侧延伸，RPV 向另一侧

# 各分支存在性检测的最小体素数
MIN_BRANCH_VOXELS = 30
CORONAL_MIN_Z_EXTENT_MM = 35.0
CORONAL_MIN_X_EXTENT_MM = 35.0
CORONAL_MAX_TREE_FILL_RATIO = 0.45
CORONAL_LOW_THRESHOLD_MAX_FILL_RATIO = 0.62
CORONAL_LOW_THRESHOLD_MAX_MEDIAN_AREA_MM2 = 500.0
CORONAL_LOW_THRESHOLD_SCORE_MARGIN = 18.0


def score_threshold(
    mask: np.ndarray,
    seed_zyx: tuple[float, float, float] | None,
    spacing_zyx: tuple[float, float, float],
    is_post_tips: bool = False,
    target_voxels: int = 420_000,
) -> tuple[float, dict]:
    """对一个阈值产生的 mask 打分。

    返回 (score, details)。score 满分 100。
    """
    mask = np.asarray(mask, dtype=bool)
    details: dict = {"voxels": int(mask.sum())}

    if mask.sum() == 0:
        return 0.0, {**details, "reason": "empty"}

    seed_reliable = seed_zyx is not None and ndi is not None
    details["unreliable_score"] = not seed_reliable

    # === 1. 完整性评分 (0-35) ===
    completeness_score, completeness_info = _score_completeness(
        mask, seed_zyx, spacing_zyx, is_post_tips,
    )
    details["completeness"] = completeness_info

    # === 2. 可分离性评分 (0-30) ===
    separability_score, separability_info = _score_separability(
        mask, seed_zyx, spacing_zyx,
    )
    details["separability"] = separability_info

    # Coronal MIP is usually the clearest view of the portal tree. Keep this
    # separate from the axial score so threshold search can prefer tree-like
    # masks instead of tiny, highly connected remnants.
    coronal_score, coronal_info = _score_coronal_tree(mask, spacing_zyx)
    details["coronal_tree"] = coronal_info

    # === 3. 连通性评分 (0-20) ===
    connectivity_score, connectivity_info = _score_connectivity(mask, seed_zyx)
    details["connectivity"] = connectivity_info

    # === 4. 尺寸合理性 (0-15) ===
    size_score, size_info = _score_size(mask, target_voxels)
    details["size"] = size_info

    if not seed_reliable:
        connectivity_score = min(connectivity_score, 5.0)
        size_score = min(size_score, 8.0)

    total = completeness_score + separability_score + connectivity_score + size_score
    details["scores"] = {
        "completeness": round(completeness_score, 2),
        "separability": round(separability_score, 2),
        "coronal_tree": round(coronal_score, 2),
        "connectivity": round(connectivity_score, 2),
        "size": round(size_score, 2),
        "total": round(total, 2),
    }
    return total, details


# =========================================================================
# 1. 完整性：各分支是否存在
# =========================================================================

def _score_completeness(
    mask: np.ndarray,
    seed_zyx: tuple[float, float, float] | None,
    spacing_zyx: tuple[float, float, float],
    is_post_tips: bool,
) -> tuple[float, dict]:
    """检查门静脉各分支是否存在。

    从 seed 点出发，检测 mask 在六个半空间（z±, y±, x±）的分布。
    门静脉系统应该有：
    - 向 z_superior（头端）延伸的分支 → LPV/RPV 进入肝脏
    - 向 z_inferior（足端）延伸的分支 → SMV
    - 向一侧 x 延伸的分支 → 脾静脉
    - seed 附近有足够体素 → 门静脉主干本身
    """
    info: dict = {"branches": {}}

    if seed_zyx is None or ndi is None:
        return 0.0, {**info, "method": "no_seed_unreliable"}

    nz, ny, nx = mask.shape
    sz, sy, sx = int(round(seed_zyx[0])), int(round(seed_zyx[1])), int(round(seed_zyx[2]))
    sz = max(0, min(nz - 1, sz))
    sy = max(0, min(ny - 1, sy))
    sx = max(0, min(nx - 1, sx))

    # 从 seed 出发做连通域提取（只分析 seed 所在连通域）
    labels, n = _label_int32(mask)
    if n == 0:
        del labels
        return 0.0, {**info, "error": "no_components"}

    # 在 seed 附近找最近的前景体素（不做全量 argwhere）
    sp = np.asarray(spacing_zyx, dtype=np.float32)
    main_label = None
    # 先检查 seed 位置本身
    if mask[sz, sy, sx]:
        main_label = int(labels[sz, sy, sx])
    else:
        # 逐步扩大搜索半径
        max_r = max(3, int(round(25.0 / float(np.min(sp))))) + 1
        for r in range(1, max_r + 1):
            z0, z1 = max(0, sz - r), min(nz, sz + r + 1)
            y0, y1 = max(0, sy - r), min(ny, sy + r + 1)
            x0, x1 = max(0, sx - r), min(nx, sx + r + 1)
            patch = mask[z0:z1, y0:y1, x0:x1]
            if patch.any():
                lc = np.argwhere(patch)
                gc = lc + np.array([z0, y0, x0])
                dists = np.sum(((gc.astype(np.float32) - np.array([sz, sy, sx], dtype=np.float32)) * sp) ** 2, axis=1)
                best = gc[int(np.argmin(dists))]
                main_label = int(labels[best[0], best[1], best[2]])
                break
    if main_label is None or main_label == 0:
        del labels
        return 0.0, {**info, "error": "no_foreground_near_seed"}

    main_mask = (labels == main_label)
    del labels  # 释放大数组

    if main_mask.sum() == 0:
        del main_mask
        return 0.0, {**info, "error": "empty_main_component"}

    # --- 检测各方向的分支 ---
    score = 0.0
    total_possible = 35.0

    # a) seed 附近区域（门静脉主干）—— 7 分
    trunk_radius_z = max(3, int(round(15.0 / max(0.1, spacing_zyx[0]))))
    trunk_radius_yx = max(3, int(round(15.0 / max(0.1, spacing_zyx[1]))))
    trunk_region = main_mask[
        max(0, sz - trunk_radius_z):min(nz, sz + trunk_radius_z + 1),
        max(0, sy - trunk_radius_yx):min(ny, sy + trunk_radius_yx + 1),
        max(0, sx - trunk_radius_yx):min(nx, sx + trunk_radius_yx + 1),
    ]
    trunk_voxels = int(trunk_region.sum())
    trunk_present = trunk_voxels > MIN_BRANCH_VOXELS
    info["branches"]["trunk"] = {"voxels": trunk_voxels, "present": trunk_present}
    if trunk_present:
        score += 7.0

    # b) z_superior 方向（LPV/RPV 进入肝脏）—— 8 分
    # 从 seed 向头端看（具体方向取决于 z 轴定向，这里假设 z 增大 = 更多层 = 任一方向都检查）
    z_above = main_mask[max(0, sz + trunk_radius_z):, :, :]
    z_below = main_mask[:max(1, sz - trunk_radius_z), :, :]
    above_voxels = int(z_above.sum())
    below_voxels = int(z_below.sum())

    # LPV/RPV 在 seed 上方或下方（取决于 z 方向）—— 取较大的那个方向
    lpv_rpv_voxels = max(above_voxels, below_voxels)
    lpv_rpv_present = lpv_rpv_voxels > MIN_BRANCH_VOXELS * 3
    info["branches"]["lpv_rpv_region"] = {
        "above_voxels": above_voxels,
        "below_voxels": below_voxels,
        "present": lpv_rpv_present,
    }
    if lpv_rpv_present:
        score += 8.0

    # c) SMV 方向 —— 8 分
    # SMV 在 seed 的另一端
    smv_voxels = min(above_voxels, below_voxels)
    smv_present = smv_voxels > MIN_BRANCH_VOXELS * 2
    info["branches"]["smv_region"] = {"voxels": smv_voxels, "present": smv_present}
    if smv_present:
        score += 8.0

    # d) 脾静脉方向（x 方向一侧延伸）—— 7 分
    x_left = main_mask[:, :, :max(1, sx - trunk_radius_yx)]
    x_right = main_mask[:, :, min(nx, sx + trunk_radius_yx + 1):]
    left_voxels = int(x_left.sum())
    right_voxels = int(x_right.sum())
    splenic_voxels = max(left_voxels, right_voxels)
    splenic_present = splenic_voxels > MIN_BRANCH_VOXELS * 2
    info["branches"]["splenic_region"] = {
        "left_voxels": left_voxels,
        "right_voxels": right_voxels,
        "present": splenic_present,
    }
    if splenic_present:
        score += 7.0

    # e) TIPS 支架（仅 post-TIPS 患者）—— 5 分（从其他项匀出）
    if is_post_tips:
        # TIPS 在评分中作为 bonus，不减少其他项
        # 如果所有其他分支都在，加上 TIPS bonus
        info["branches"]["tips"] = {"checked": True}
        # TIPS 检测由高 HU 通道处理，这里只检查是否有高于正常的 HU 区域
        # 不在此处评分

    # z 方向跨度 bonus（确保整体延伸足够）—— 5 分
    z_has_mask = np.any(np.any(main_mask, axis=2), axis=1)  # shape (nz,), 内存安全
    z_indices = np.where(z_has_mask)[0]
    if len(z_indices) > 1:
        z_extent_mm = float((z_indices[-1] - z_indices[0]) * spacing_zyx[0])
    else:
        z_extent_mm = 0.0
    del main_mask
    extent_present = z_extent_mm > 40.0  # 门静脉系统至少跨 40mm
    info["z_extent_mm"] = round(z_extent_mm, 1)
    if extent_present:
        score += 5.0

    info["score"] = round(score, 2)
    info["max_score"] = total_possible
    return score, info


# =========================================================================
# 2. 可分离性：LPV/RPV 是否与肝脏融合
# =========================================================================

def _score_separability(
    mask: np.ndarray,
    seed_zyx: tuple[float, float, float] | None,
    spacing_zyx: tuple[float, float, float],
) -> tuple[float, dict]:
    """检查 LPV/RPV 区域是否与肝实质清晰分离。

    关键指标：在 seed 上方/下方（LPV/RPV 所在区域），
    每层的 mask 截面积分布。

    - 血管截面：小而分散，每个连通域面积小（管状截面 < 200mm²）
    - 肝实质融合：某些层出现大片连通域（> 500mm² 的团块）

    评分逻辑：
    - 计算 LPV/RPV 区域每层的最大连通域面积
    - 如果多数层的最大连通域 < 阈值 → 高分（血管清晰）
    - 如果大量层出现大面积团块 → 低分（肝脏融合）
    """
    info: dict = {}
    max_score = 30.0

    if seed_zyx is None or ndi is None:
        return 0.0, {"method": "no_seed_unreliable"}

    nz, ny, nx = mask.shape
    sz = int(round(seed_zyx[0]))
    sz = max(0, min(nz - 1, sz))
    pixel_area_mm2 = float(spacing_zyx[1]) * float(spacing_zyx[2])

    # LPV/RPV 所在区域：seed 上方和下方各取一段
    # 取 z 方向两端各 30% 的 mask 覆盖范围
    mask_z_indices = np.any(np.any(mask, axis=2), axis=1)
    if not mask_z_indices.any():
        return 0.0, {"error": "empty_mask"}

    z_with_mask = np.where(mask_z_indices)[0]
    z_min, z_max = int(z_with_mask.min()), int(z_with_mask.max())
    z_range = z_max - z_min + 1

    # 分析 seed 上方和下方的区域
    above_start = min(nz, sz + 5)
    above_end = z_max + 1
    below_start = z_min
    below_end = max(0, sz - 5)

    # 取两个端部区域中较大的那个（更可能是 LPV/RPV 所在区域）
    if above_end - above_start > below_end - below_start:
        check_start, check_end = above_start, above_end
    else:
        check_start, check_end = below_start, below_end

    if check_end <= check_start:
        return 15.0, {"method": "no_branch_region"}

    # 逐层分析最大连通域面积
    max_areas_mm2 = []
    tubularity_scores = []

    for z in range(check_start, check_end):
        slice_mask = mask[z]
        if slice_mask.sum() == 0:
            continue

        labels_2d, n_2d = ndi.label(slice_mask)
        if n_2d == 0:
            continue

        counts = np.bincount(labels_2d.ravel())
        counts[0] = 0
        max_area_pixels = int(counts.max())
        max_area_mm2 = float(max_area_pixels) * pixel_area_mm2
        max_areas_mm2.append(max_area_mm2)

        # 管状性（tubularity）：周长 / 面积
        # 对最大连通域计算
        if max_area_pixels > 10:
            biggest = (labels_2d == int(counts.argmax()))
            # 边界像素数 ≈ 周长
            eroded = ndi.binary_erosion(biggest, iterations=1)
            perimeter = int(biggest.sum() - eroded.sum())
            tubularity = float(perimeter) / max(1, max_area_pixels)
            tubularity_scores.append(tubularity)

    if not max_areas_mm2:
        return 15.0, {"method": "no_data"}

    max_areas = np.array(max_areas_mm2)
    info["n_slices_analyzed"] = len(max_areas)
    info["max_area_median_mm2"] = round(float(np.median(max_areas)), 1)
    info["max_area_p90_mm2"] = round(float(np.percentile(max_areas, 90)), 1)
    info["max_area_max_mm2"] = round(float(max_areas.max()), 1)

    # 评分规则：
    # - 中位截面积 < 150mm² → 管状，30 分
    # - 中位截面积 150-400mm² → 部分融合，线性衰减
    # - 中位截面积 > 400mm² → 严重融合，接近 0 分
    # - P90 截面积 > 800mm² → 有大面积融合，额外扣分

    median_area = float(np.median(max_areas))
    p90_area = float(np.percentile(max_areas, 90))

    if median_area < 150:
        base_score = max_score
    elif median_area < 400:
        base_score = max_score * (1.0 - (median_area - 150) / 250)
    else:
        base_score = max_score * max(0.0, 0.1 - (median_area - 400) / 5000)

    # P90 惩罚
    if p90_area > 800:
        penalty = min(10.0, (p90_area - 800) / 200)
        base_score = max(0.0, base_score - penalty)

    # 管状性 bonus
    if tubularity_scores:
        mean_tubularity = float(np.mean(tubularity_scores))
        info["mean_tubularity"] = round(mean_tubularity, 4)
        # 管状结构的 tubularity 约 0.15-0.40，团块约 0.05-0.10
        if mean_tubularity > 0.15:
            base_score = min(max_score, base_score + 3.0)

    info["score"] = round(base_score, 2)
    return base_score, info


# =========================================================================
# 2b. 冠状面树形：门静脉在冠状 MIP 上应为长而不实心的树状结构
# =========================================================================

def _score_coronal_tree(
    mask: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> tuple[float, dict]:
    max_score = 20.0
    info: dict = {}

    if ndi is None:
        return 0.0, {"method": "no_scipy"}

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0.0, {"error": "empty"}

    coronal = np.any(mask, axis=1)  # z-x projection, matching coronal MIP logic.
    labels, n = ndi.label(coronal)
    if n == 0:
        return 0.0, {"error": "empty_projection"}
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    main_label = int(counts.argmax())
    main = labels == main_label
    coords = np.argwhere(main)
    if len(coords) == 0:
        return 0.0, {"error": "empty_main_projection"}

    z_min, x_min = coords.min(axis=0)
    z_max, x_max = coords.max(axis=0)
    z_extent_mm = float((z_max - z_min + 1) * spacing_zyx[0])
    x_extent_mm = float((x_max - x_min + 1) * spacing_zyx[2])
    bbox_area = max(1, int((z_max - z_min + 1) * (x_max - x_min + 1)))
    proj_area = int(main.sum())
    fill_ratio = float(proj_area / bbox_area)
    main_fraction = float(proj_area / max(1, int(coronal.sum())))

    info.update({
        "projection_area_px": proj_area,
        "z_extent_mm": round(z_extent_mm, 1),
        "x_extent_mm": round(x_extent_mm, 1),
        "fill_ratio": round(fill_ratio, 4),
        "main_projection_fraction": round(main_fraction, 4),
    })

    z_score = min(1.0, z_extent_mm / CORONAL_MIN_Z_EXTENT_MM)
    x_score = min(1.0, x_extent_mm / CORONAL_MIN_X_EXTENT_MM)
    if fill_ratio <= 0.12:
        fill_score = fill_ratio / 0.12
    elif fill_ratio <= 0.32:
        fill_score = 1.0
    elif fill_ratio <= CORONAL_MAX_TREE_FILL_RATIO:
        fill_score = 1.0 - 0.5 * (fill_ratio - 0.32) / (CORONAL_MAX_TREE_FILL_RATIO - 0.32)
    else:
        fill_score = max(0.0, 0.5 - (fill_ratio - CORONAL_MAX_TREE_FILL_RATIO) / 0.25)
    component_score = min(1.0, main_fraction / 0.65)

    score = max_score * (0.35 * z_score + 0.30 * x_score + 0.25 * fill_score + 0.10 * component_score)
    info["score"] = round(float(score), 2)
    return float(score), info


# =========================================================================
# 3. 连通性
# =========================================================================

def _score_connectivity(
    mask: np.ndarray,
    seed_zyx: tuple[float, float, float] | None,
) -> tuple[float, dict]:
    """seed 所在连通域占总体素的比例。

    门静脉系统理想情况下是一棵连通的树。
    主连通域占比越高 → 分割越干净（没有散落的噪声片段）。
    """
    max_score = 20.0
    info: dict = {}

    if ndi is None:
        return 10.0, {"method": "no_scipy"}

    mask = np.asarray(mask, dtype=bool)
    total = int(mask.sum())
    if total == 0:
        return 0.0, {"error": "empty"}

    labels, n = _label_int32(mask)
    counts = _count_labels(labels, n)
    counts[0] = 0
    info["n_components"] = int(n)
    info["n_significant_components"] = int(np.count_nonzero(counts >= 64))

    if seed_zyx is not None:
        nz, ny, nx = mask.shape
        sz = max(0, min(nz - 1, int(round(seed_zyx[0]))))
        sy = max(0, min(ny - 1, int(round(seed_zyx[1]))))
        sx = max(0, min(nx - 1, int(round(seed_zyx[2]))))
        if mask[sz, sy, sx]:
            main_count = int(counts[labels[sz, sy, sx]])
        else:
            # 小范围搜索
            sp = np.asarray((1.0, 1.0, 1.0), dtype=np.float32)
            main_count = int(counts.max())
            for r in range(1, 10):
                z0, z1 = max(0, sz - r), min(nz, sz + r + 1)
                y0, y1 = max(0, sy - r), min(ny, sy + r + 1)
                x0, x1 = max(0, sx - r), min(nx, sx + r + 1)
                patch = mask[z0:z1, y0:y1, x0:x1]
                if patch.any():
                    lc = np.argwhere(patch)
                    gc = lc + np.array([z0, y0, x0])
                    gz, gy, gx = gc[0]
                    main_count = int(counts[labels[gz, gy, gx]])
                    break
    else:
        main_count = int(counts.max())
    del labels

    main_fraction = float(main_count / max(1, total))
    info["main_fraction"] = round(main_fraction, 4)
    info["main_voxels"] = main_count

    # 评分：主连通域占比
    # > 85% → 满分
    # 60-85% → 线性
    # < 60% → 低分（太多散落碎片）
    if main_fraction > 0.85:
        score = max_score
    elif main_fraction > 0.60:
        score = max_score * (main_fraction - 0.60) / 0.25
    elif main_fraction > 0.30:
        score = max_score * 0.3 * (main_fraction - 0.30) / 0.30
    else:
        score = 0.0

    info["score"] = round(score, 2)
    return score, info


# =========================================================================
# 4. 尺寸合理性
# =========================================================================

def _score_size(mask: np.ndarray, target_voxels: int) -> tuple[float, dict]:
    """体素数是否在合理范围。

    不像之前严格要求接近 target，而是：
    - 太少（< target 的 5%）→ 明显丢了分支
    - 适中（5%-80%）→ 满分（宽松，因为后续有区域生长清理）
    - 过多（> target）→ 扣分但不致命
    """
    max_score = 15.0
    voxels = int(mask.sum())
    ratio = float(voxels) / max(1, target_voxels)
    info = {"voxels": voxels, "target": target_voxels, "ratio": round(ratio, 4)}

    if ratio < 0.03:
        score = 0.0  # 几乎空
    elif ratio < 0.05:
        score = max_score * 0.3 * (ratio - 0.03) / 0.02
    elif ratio <= 0.80:
        score = max_score  # 满分区间（宽松）
    elif ratio <= 1.5:
        score = max_score * (1.0 - 0.3 * (ratio - 0.80) / 0.70)
    elif ratio <= 3.0:
        score = max_score * (0.7 - 0.5 * (ratio - 1.5) / 1.5)
    else:
        score = max_score * 0.1  # 过多但不为零

    info["score"] = round(score, 2)
    return score, info


# =========================================================================
# 搜索最佳阈值
# =========================================================================

def search_best_threshold(
    vol: np.ndarray,
    plan: dict,
    spine_mask: np.ndarray | None,
    seed_zyx: tuple[float, float, float] | None,
    spacing_zyx: tuple[float, float, float],
    is_post_tips: bool,
    segment_fn,
    region_grow_fn=None,
    reference_envelope: np.ndarray | None = None,
    hu_low_candidates: list[float] | None = None,
    return_best_mask: bool = False,
):
    """搜索最佳 hu_low。

    参数：
        segment_fn: 分割函数 segment_fn(vol, plan, is_post_tips, spine_mask) -> mask
        region_grow_fn: 可选，区域生长函数 (mask, seed, spacing) -> (mask, info)
        reference_envelope: 可选，参考 STL 的 envelope mask
        hu_low_candidates: 可选，自定义搜索候选值

    返回：
        best_plan: 最优 plan（hu_low 已更新）
        search_info: 详细搜索日志
    """
    target = TARGET_VOXELS_TIPS if is_post_tips else TARGET_VOXELS

    if hu_low_candidates is None:
        hu_low_candidates = [80, 90, 100, 110, 120, 135, 150, 170, 190]
        # 加入 plan 中 LLM 建议的值
        llm_low = plan.get("hu_low")
        if llm_low is not None:
            hu_low_candidates.append(float(llm_low))
        hu_low_candidates = sorted(set(float(v) for v in hu_low_candidates))

    candidates = []
    best_mask = None
    for hu_low in hu_low_candidates:
        if hu_low >= float(plan.get("hu_high", 380)) - 20:
            continue

        trial_plan = dict(plan)
        trial_plan["hu_low"] = float(hu_low)

        # 分割
        mask = segment_fn(vol, trial_plan, is_post_tips, spine_mask)

        # 可选：apply envelope
        if reference_envelope is not None:
            overlap = mask & reference_envelope
            if int(overlap.sum()) > 0:
                mask = overlap

        # 可选：区域生长
        grow_info = None
        if region_grow_fn is not None and seed_zyx is not None:
            mask, grow_info = region_grow_fn(mask, seed_zyx, spacing_zyx)

        # 打分
        score, details = score_threshold(
            mask, seed_zyx, spacing_zyx, is_post_tips, target,
        )

        # 只保留摘要，不存完整 details（节省内存）
        summary = {
            "hu_low": float(hu_low),
            "score": round(score, 2),
            "voxels": int(mask.sum()),
            "completeness": details.get("scores", {}).get("completeness", 0),
            "separability": details.get("scores", {}).get("separability", 0),
            "coronal_tree": details.get("scores", {}).get("coronal_tree", 0),
            "connectivity": details.get("scores", {}).get("connectivity", 0),
            "size": details.get("scores", {}).get("size", 0),
            "unreliable_score": bool(details.get("unreliable_score", False)),
        }
        if grow_info:
            summary["seed_distance_mm"] = grow_info.get("nearest_seed_distance_mm")
            summary["seed_grow_voxels"] = grow_info.get("output_voxels")
        # 可分离性关键指标
        sep = details.get("separability", {})
        if "max_area_median_mm2" in sep:
            summary["median_area_mm2"] = sep["max_area_median_mm2"]
        cor = details.get("coronal_tree", {})
        for k in ("z_extent_mm", "x_extent_mm", "fill_ratio", "projection_area_px", "main_projection_fraction"):
            if k in cor:
                summary[f"coronal_{k}"] = cor[k]
        # 连通性关键指标
        conn = details.get("connectivity", {})
        if "main_fraction" in conn:
            summary["main_fraction"] = conn["main_fraction"]

        summary["selection_score"] = round(_threshold_selection_score(summary), 2)
        summary["coronal_low_threshold_candidate"] = _is_coronal_low_threshold_candidate(summary)

        candidates.append(summary)
        if best_mask is None or summary["selection_score"] > max(c["selection_score"] for c in candidates[:-1]):
            best_mask = mask.copy()
        del mask, details
        gc.collect()

    if not candidates:
        result = (plan, {"status": "no_candidates"})
        return (*result, None) if return_best_mask else result

    # Prefer candidates that keep weak branches in coronal MIP without merging
    # broad liver areas. This guards against high HU thresholds winning only
    # because the remaining mask is small and connected.
    best = _select_best_candidate(candidates)

    # 如果最高分仍然很低（< 30），说明可能所有阈值都不好
    # 此时保守选择得分最高的
    best_plan = dict(plan)
    best_plan["hu_low"] = best["hu_low"]

    search_info = {
        "status": "ok",
        "best_hu_low": best["hu_low"],
        "best_score": best["score"],
        "best_selection_score": best["selection_score"],
        "best_voxels": best["voxels"],
        "n_candidates": len(candidates),
        "selection_reason": _threshold_selection_reason(best),
        "candidates": candidates,
    }
    if return_best_mask:
        mask = segment_fn(vol, best_plan, is_post_tips, spine_mask)
        if reference_envelope is not None:
            overlap = mask & reference_envelope
            if int(overlap.sum()) > 0:
                mask = overlap
        if region_grow_fn is not None and seed_zyx is not None:
            mask, _ = region_grow_fn(mask, seed_zyx, spacing_zyx)
        return best_plan, search_info, mask
    return best_plan, search_info


def _threshold_selection_score(candidate: dict) -> float:
    score = float(candidate.get("score", 0.0))
    hu_low = float(candidate.get("hu_low", 0.0))
    voxels = float(candidate.get("voxels", 0.0))
    median_area = float(candidate.get("median_area_mm2", 0.0) or 0.0)
    main_fraction = float(candidate.get("main_fraction", 0.0) or 0.0)
    coronal_score = float(candidate.get("coronal_tree", 0.0) or 0.0)
    coronal_fill = float(candidate.get("coronal_fill_ratio", 0.0) or 0.0)
    coronal_z = float(candidate.get("coronal_z_extent_mm", 0.0) or 0.0)
    coronal_x = float(candidate.get("coronal_x_extent_mm", 0.0) or 0.0)

    selection = score + coronal_score * 1.35
    if candidate.get("unreliable_score"):
        selection -= 25.0
    if median_area > 800:
        selection -= min(25.0, (median_area - 800.0) / 60.0)
    elif median_area > 450:
        selection -= min(15.0, (median_area - 450.0) / 70.0)
    if voxels < 20_000:
        selection -= min(20.0, (20_000.0 - voxels) / 1000.0)
    if hu_low > 150 and voxels < 120_000:
        selection -= min(18.0, (hu_low - 150.0) / 3.0)
    if coronal_fill > 0.50:
        selection -= min(30.0, (coronal_fill - 0.50) * 80.0)
    if coronal_z < CORONAL_MIN_Z_EXTENT_MM or coronal_x < CORONAL_MIN_X_EXTENT_MM:
        selection -= 12.0
    if 80 <= hu_low <= 120 and median_area <= 800 and main_fraction >= 0.45:
        selection += 8.0
    if 80 <= hu_low <= 135 and coronal_score >= 14.0 and coronal_fill <= CORONAL_MAX_TREE_FILL_RATIO:
        selection += 10.0
    return float(selection)


def _is_coronal_low_threshold_candidate(candidate: dict) -> bool:
    hu_low = float(candidate.get("hu_low", 0.0) or 0.0)
    voxels = float(candidate.get("voxels", 0.0) or 0.0)
    median_area = float(candidate.get("median_area_mm2", 0.0) or 0.0)
    coronal_score = float(candidate.get("coronal_tree", 0.0) or 0.0)
    coronal_fill = float(candidate.get("coronal_fill_ratio", 0.0) or 0.0)
    coronal_z = float(candidate.get("coronal_z_extent_mm", 0.0) or 0.0)
    coronal_x = float(candidate.get("coronal_x_extent_mm", 0.0) or 0.0)
    main_fraction = float(candidate.get("main_fraction", 0.0) or 0.0)

    return (
        hu_low <= 150.0
        and voxels >= 20_000.0
        and median_area <= CORONAL_LOW_THRESHOLD_MAX_MEDIAN_AREA_MM2
        and coronal_score >= 12.0
        and 0.10 <= coronal_fill <= CORONAL_LOW_THRESHOLD_MAX_FILL_RATIO
        and coronal_z >= CORONAL_MIN_Z_EXTENT_MM
        and coronal_x >= CORONAL_MIN_X_EXTENT_MM
        and main_fraction >= 0.45
    )


def _select_best_candidate(candidates: list[dict]) -> dict:
    ranked_best = max(candidates, key=lambda c: (c["selection_score"], -c["hu_low"]))
    coronal_viable = [c for c in candidates if c.get("coronal_low_threshold_candidate")]
    if not coronal_viable:
        return ranked_best

    viable_best = max(coronal_viable, key=lambda c: (c["selection_score"], -c["hu_low"]))
    if viable_best["selection_score"] < ranked_best["selection_score"] - CORONAL_LOW_THRESHOLD_SCORE_MARGIN:
        return ranked_best

    close_viable = [
        c for c in coronal_viable
        if c["selection_score"] >= viable_best["selection_score"] - 6.0
    ]
    return min(close_viable, key=lambda c: (c["hu_low"], -c["selection_score"]))


def _threshold_selection_reason(best: dict) -> str:
    if best.get("unreliable_score"):
        return "selected_with_unreliable_seed_penalty"
    if best.get("coronal_low_threshold_candidate") and float(best.get("hu_low", 0.0) or 0.0) <= 150.0:
        return "coronal_mip_preserved_low_threshold_branches"
    median_area = best.get("median_area_mm2")
    if median_area is not None and float(median_area) > 450:
        return "best_available_despite_possible_liver_merge"
    if float(best.get("coronal_tree", 0.0) or 0.0) >= 14.0:
        return "coronal_tree_mip_supported_threshold"
    if float(best.get("hu_low", 0.0)) <= 120:
        return "kept_low_threshold_for_weak_enhancement"
    return "balanced_connectivity_and_separability"


# 导出的 target 常量
TARGET_VOXELS = 420_000
TARGET_VOXELS_TIPS = 330_000
