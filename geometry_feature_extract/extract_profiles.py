"""
中心线逐点剖面特征提取（v3 - 读取分段 JSON 驱动）
==================================================
不再做解剖识别, 直接读 centerline_profiles.json 拿到每段路径,
对每段提取逐点剖面 (面积/周长/直径/圆度/曲率/内切半径)。

输出文件: centerline_pointwise_profiles.json
        (注意: 与分段文件 centerline_profiles.json 区分)

支持的段:
    MPV / SV / SMV / LPV / RPV / TIPS / LGV / PGV
    (任何 segment_vessels.py 输出的非 None 段都会被处理)
"""

import os
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
import trimesh
import trimesh.intersections

from utils import (load_tree, path_to_coords, voxelize_stl, physical_to_voxel)
from features_layout import (
    POINTWISE_TEMP_NAME,
    SEGMENT_ASSIGNMENTS_NAME,
    features_dir,
    feature_path,
    resolve_feature_path,
)


# ============================================================
# 截面计算核心 (与原版一致)
# ============================================================

def _make_orthonormal_basis(normal):
    """为法向量构造正交基 (u, v)"""
    n = normal / (np.linalg.norm(normal) + 1e-15)
    ref = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(n, ref)
    u /= (np.linalg.norm(u) + 1e-15)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-15)
    return u, v


def _polygon_aspect_ratio(poly_coords):
    """
    用 PCA 估计多边形顶点的长短轴比 (aspect_ratio = √(λ_max / λ_min)).

    aspect_ratio = 1.0  圆形/正方形
    aspect_ratio ≈ 1.4  椭圆 (b/a=2/3, 真实血管常见)
    aspect_ratio > 4    显著拉长 (沿管轴薄片切, 或跨血管切)

    返回值在 [1, +∞), 顶点不足或退化时返回 999.0
    """
    pts = np.asarray(poly_coords, dtype=float)
    if len(pts) < 3:
        return 999.0
    if pts.ndim != 2 or pts.shape[1] < 2:
        return 999.0
    pts = pts[:, :2]
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # 协方差矩阵 (2x2)
    try:
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
    except Exception:
        return 999.0
    eigvals = np.clip(eigvals, 0.0, None)
    if eigvals[1] < 1e-12 or eigvals[0] < 1e-12:
        return 999.0
    return float(np.sqrt(eigvals[1] / eigvals[0]))


def _pick_polygon_from_geometry(geom, center):
    """从 Polygon/MultiPolygon 中选中心线锚点所属的主多边形。"""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return geom if geom.is_valid and geom.area > 0 else None
    if not hasattr(geom, 'geoms'):
        return None

    polys = [g for g in geom.geoms
             if getattr(g, 'geom_type', None) == 'Polygon'
             and g.is_valid and g.area > 0]
    if not polys:
        return None

    containing = [p for p in polys if p.covers(center)]
    if containing:
        return max(containing, key=lambda p: p.area)
    return min(polys, key=lambda p: p.distance(center))


def _center_owned_polygon(poly, center, ownership_factor=1.8,
                          min_owned_area_fraction=0.55,
                          min_raw_circularity_to_skip=0.35,
                          max_raw_aspect_to_skip=4.0):
    """
    用中心线锚定的最大内切半径裁剪截面。

    r_anchor 是中心点到截面边界的最短距离。限制圆半径取
    ownership_factor * r_anchor: 圆形血管不受影响, 椭圆血管保留主体,
    分叉污染向外伸出的区域会被裁掉。
    """
    if poly is None or poly.is_empty or not poly.is_valid:
        return None, 0.0, 0.0

    if ownership_factor is None or ownership_factor <= 0:
        return poly, 0.0, 0.0

    if not poly.covers(center):
        return poly, 0.0, 0.0

    anchor_radius = float(poly.boundary.distance(center))
    if anchor_radius <= 1e-6:
        return poly, anchor_radius, 0.0

    raw_area = float(poly.area)
    raw_peri = float(poly.exterior.length)
    raw_circularity = (
        float(4.0 * np.pi * raw_area / (raw_peri * raw_peri))
        if raw_peri > 1e-6 else 0.0
    )
    raw_aspect = _polygon_aspect_ratio(list(poly.exterior.coords))

    owned_radius = float(anchor_radius * ownership_factor)
    limiter = center.buffer(owned_radius, resolution=64)
    owned_geom = poly.intersection(limiter)
    owned_poly = _pick_polygon_from_geometry(owned_geom, center)
    if owned_poly is None or owned_poly.area <= 1e-9:
        return poly, anchor_radius, owned_radius

    # Guard against artificial bottlenecks when the centerline point is not
    # perfectly centered in the lumen. In that case the nearest-wall distance
    # can be much smaller than the real section radius, and the ownership
    # circle clips away a large part of an otherwise compact, plausible section.
    # We still keep clipping for irregular/elongated branch-contaminated
    # sections, where removing spill-over is the intended behavior.
    if (raw_area > 1e-9
            and owned_poly.area < min_owned_area_fraction * raw_area
            and raw_circularity >= min_raw_circularity_to_skip
            and raw_aspect <= max_raw_aspect_to_skip):
        return poly, anchor_radius, owned_radius

    return owned_poly, anchor_radius, owned_radius


def _section_discrete_polygons(mesh, point, normal, u, v):
    """Rebuild closed section loops using Trimesh's path assembler.

    ``mesh_plane`` can return valid triangle-intersection segments that do not
    share exact endpoints, leaving Shapely unable to polygonize them. Trimesh
    reconstructs graph paths before exposing ``discrete`` loops, which is a
    reliable fallback for those otherwise valid watertight meshes.
    """
    try:
        section = mesh.section(plane_origin=point, plane_normal=normal)
        if section is None:
            return []
        from shapely.geometry import Polygon

        polygons = []
        for loop in section.discrete:
            loop = np.asarray(loop, dtype=float)
            if loop.ndim != 2 or loop.shape[0] < 3:
                continue
            coords_2d = [
                (float(np.dot(vertex - point, u)),
                 float(np.dot(vertex - point, v)))
                for vertex in loop
            ]
            if len(coords_2d) < 3:
                continue
            if coords_2d[0] != coords_2d[-1]:
                coords_2d.append(coords_2d[0])
            polygon = Polygon(coords_2d)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
            if hasattr(polygon, 'geoms'):
                polygons.extend(
                    geom for geom in polygon.geoms
                    if geom.is_valid and geom.area > 0)
            elif polygon.is_valid and polygon.area > 0:
                polygons.append(polygon)
        return polygons
    except Exception:
        return []


def _section_one(mesh, point, normal, max_eq_diameter=None,
                 min_eq_diameter=None,
                 ownership_factor=1.8,
                 return_ring=False, return_metrics=False,
                 return_raw=False, return_extras=False):
    """
    用一个法线做截面, 返回截面几何 + 形状质量指标。

    流程: mesh_plane → 交线段 → 投影 2D → polygonize → 候选多边形过滤
    候选选择策略 (按优先级):
      1. 多边形必须包含中心点 (0,0) — 否则该多边形属于其他血管
      2. 包含中心的多边形中, 面积最小者 (避免选到合并的"图8"形状外环)
      3. 若无包含中心者, 退而求距中心最近的有效多边形

    形状质量指标 (用于上层做"自适应"过滤, 无需固定大小阈值):
      - aspect_ratio: PCA 长短轴比. 1.0 = 圆/正方; >4 通常是沿管轴薄片切或跨血管.
      - circularity:  4πA/P². 1.0 = 完美圆; <0.3 形状极不规则.

    额外形状感知量 (return_extras=True, 用于 PVT / 血栓识别):
      - n_components: 切平面下"有效闭合多边形"个数. 正常血管 = 1;
                      血栓把管腔从中间隔断 → 2+; 圆环形血栓 = 1 (仍连通).
      - solidity:     所选多边形面积 / 其凸包面积 ∈ (0, 1].
                      凸截面 (圆/椭圆) = 1; 月牙/凹缺口形 < 1; 越小代表
                      凹缺口越深 (典型 PVT 边缘血栓).

    中心线锚定清洗:
      先取真实截面多边形 P, 再用当前中心线投影点 (0,0) 到 P 边界的
      最短距离作为锚定内切半径 r_anchor。最终用于面积统计的是
      P ∩ circle((0,0), ownership_factor·r_anchor)。这能保留当前血管
      主体, 同时裁掉分叉/汇合处伸向邻近血管的污染区域。

    防止边界效应 (邻近血管"渗透"):
      若 max_eq_diameter (一般取 1.6 ~ 2 倍局部内切直径) 给定,
      且清洗后候选多边形的等效直径 > max_eq_diameter, 视为污染并丢弃.

    参数:
        max_eq_diameter: float 或 None — 等效直径上界 (mm)
        ownership_factor: 中心锚定裁剪圆半径 / 锚定内切半径, 默认 1.8
        return_ring:     是否同时返回 2D 多边形轮廓 (用于可视化)
        return_metrics:  是否同时返回 (aspect_ratio, circularity)
        return_raw:      是否追加原始未裁剪的 area/perimeter 与锚定半径
        return_extras:   是否同时返回 (n_components, solidity)

    返回 (按 flag 组合, extras 永远放在末尾, 不影响既有调用方):
        默认                                            (area, peri)
        return_metrics=True                             (area, peri, AR, circ)
        return_ring=True                                (area, peri, ring_2d)
        return_ring=True, return_metrics=True           (area, peri, AR, circ, ring_2d)
        return_raw=True 时, 再追加 (raw_area, raw_peri, anchor_r, owned_r)
        return_extras=True 时, 最后追加 (n_components, solidity)
        失败时各位置填 0/0/999/0/None/0/0
    """
    base_fail = (0.0, 0.0)
    extras_fail = (0, 0.0)
    if return_ring and return_metrics:
        fail = (0.0, 0.0, 999.0, 0.0, None)
    elif return_metrics:
        fail = (0.0, 0.0, 999.0, 0.0)
    elif return_ring:
        fail = (0.0, 0.0, None)
    else:
        fail = base_fail
    if return_raw:
        fail = fail + (0.0, 0.0, 0.0, 0.0)
    if return_extras:
        fail = fail + extras_fail

    try:
        lines = trimesh.intersections.mesh_plane(
            mesh, plane_normal=normal, plane_origin=point)
        if lines is None or len(lines) == 0:
            return fail

        u, v = _make_orthonormal_basis(normal)

        segs_2d = []
        for seg in lines:
            r0, r1 = seg[0] - point, seg[1] - point
            p0 = (float(np.dot(r0, u)), float(np.dot(r0, v)))
            p1 = (float(np.dot(r1, u)), float(np.dot(r1, v)))
            if abs(p0[0] - p1[0]) > 1e-8 or abs(p0[1] - p1[1]) > 1e-8:
                segs_2d.append((p0, p1))

        if len(segs_2d) < 3:
            return fail

        from shapely.geometry import LineString, Point as SPoint
        from shapely.ops import polygonize, unary_union

        ls_list = [LineString([s[0], s[1]]) for s in segs_2d]
        merged = unary_union(ls_list)
        polys = list(polygonize(merged))

        if not polys:
            try:
                from shapely.ops import snap
            except ImportError:
                from shapely import snap
            snapped = snap(merged, merged, tolerance=0.05)
            polys = list(polygonize(snapped))

        if not polys:
            polys = _section_discrete_polygons(mesh, point, normal, u, v)

        if not polys:
            buffered = merged.buffer(0.01)
            if hasattr(buffered, 'geoms'):
                polys = list(buffered.geoms)
            elif buffered.area > 0:
                polys = [buffered]

        if not polys:
            return fail

        center = SPoint(0.0, 0.0)
        # 有效"非微小"多边形 — 用于估计 lumen 连通分量数
        # 阈值 0.1mm² 排除离散化产生的针状碎片
        valid_polys = [p for p in polys if p.is_valid and p.area > 0]
        nontrivial = [p for p in valid_polys if p.area > 0.1]
        candidate_pool = nontrivial if nontrivial else valid_polys
        if not candidate_pool:
            return fail

        center_tol = 0.25

        def _center_distance(poly):
            return float(poly.distance(center))

        centered = [
            p for p in candidate_pool
            if p.covers(center) or _center_distance(p) <= center_tol
        ]
        if centered:
            # Polygonize can emit small self-intersection slivers that still
            # contain the projected centerline point. Prefer the largest
            # centered lumen candidate, then fall back if size filters reject it.
            candidate_polys = sorted(centered, key=lambda p: p.area, reverse=True)
        else:
            # Avoid selecting a tiny nearest fragment when no closed loop covers
            # the center. The candidate still has to be close relative to its
            # equivalent radius.
            near = []
            for p in candidate_pool:
                eq_r = float(np.sqrt(max(float(p.area), 0.0) / np.pi))
                if _center_distance(p) <= max(center_tol, 0.35 * eq_r):
                    near.append(p)
            candidate_polys = sorted(
                near, key=lambda p: (_center_distance(p), -float(p.area)))

        if not candidate_polys:
            return fail

        best = None
        for cand in candidate_polys:
            owned, _, _ = _center_owned_polygon(
                cand, center, ownership_factor=ownership_factor)
            if owned is None or owned.is_empty:
                continue
            cand_area = float(owned.area)
            cand_peri = float(owned.exterior.length)
            if cand_area <= 1e-9 or cand_peri <= 1e-9:
                continue
            eq_d = float(np.sqrt(4.0 * cand_area / np.pi))
            if max_eq_diameter is not None and eq_d > max_eq_diameter:
                continue
            if min_eq_diameter is not None and eq_d < min_eq_diameter:
                continue
            best = cand
            break

        if best is None:
            return fail

        raw_area = float(best.area)
        raw_peri = float(best.exterior.length)

        owned, anchor_radius, owned_radius = _center_owned_polygon(
            best, center, ownership_factor=ownership_factor)
        if owned is None:
            return fail

        area = float(owned.area)
        peri = float(owned.exterior.length)

        eq_d = float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0
        if max_eq_diameter is not None and eq_d > max_eq_diameter:
            return fail
        if min_eq_diameter is not None and eq_d < min_eq_diameter:
            return fail

        # 边界效应保护: 若给了内切直径上界, 直接拒绝越界候选
        if max_eq_diameter is not None and area > 0:
            eq_d = float(np.sqrt(4.0 * area / np.pi))
            if eq_d > max_eq_diameter:
                return fail

        # 形状指标
        raw_ring_2d_list = list(best.exterior.coords)
        owned_ring_2d_list = list(owned.exterior.coords)
        if return_metrics:
            aspect_ratio = _polygon_aspect_ratio(owned_ring_2d_list)
            if peri > 1e-6:
                circularity = float(min(1.5, 4.0 * np.pi * area / (peri * peri)))
            else:
                circularity = 0.0

        if return_extras:
            # n_components: 切平面下"有效非微小"多边形数 (lumen 是否被血栓隔断)
            # 不考虑跨血管 — 跨血管候选会在后续 shape filter 中被剔除
            n_components = int(len(nontrivial)) if nontrivial else 1

            # solidity: 所选多边形面积 / 其凸包面积
            # 凸 (圆/椭圆) = 1.0; 月牙/凹缺口 < 1.0
            solidity = 1.0
            try:
                from scipy.spatial import ConvexHull
                ring_arr = np.asarray(owned_ring_2d_list, dtype=float)
                if len(ring_arr) >= 3 and ring_arr.shape[1] >= 2:
                    hull = ConvexHull(ring_arr[:, :2])
                    hull_area = float(hull.volume)  # 2D 下 .volume 即面积
                    if hull_area > 1e-9:
                        solidity = float(min(1.0, area / hull_area))
            except Exception:
                solidity = 1.0
            extras_tuple = (n_components, solidity)

        if return_ring and return_metrics:
            base = (area, peri, aspect_ratio, circularity, raw_ring_2d_list)
        elif return_metrics:
            base = (area, peri, aspect_ratio, circularity)
        elif return_ring:
            base = (area, peri, raw_ring_2d_list)
        else:
            base = (area, peri)
        if return_raw:
            base = base + (raw_area, raw_peri, anchor_radius, owned_radius)
        if return_extras:
            return base + extras_tuple
        return base

    except Exception:
        return fail


_DEFAULT_NORMAL_SEARCH_POLICY = {
    'low_curvature_threshold': 0.08,
    'high_curvature_threshold': 0.20,
    'high_curvature_context_mm': 8.0,
    # One degree of freedom: 20 symmetric candidates spanning -30 to +30
    # degrees.  The same policy is used at every curvature level so the
    # selected section remains comparable along a vessel.
    'low_n_perturb': 20,
    'medium_n_perturb': 20,
    'high_n_perturb': 20,
    'low_max_angle_deg': 60.0,
    'medium_max_angle_deg': 60.0,
    'high_max_angle_deg': 60.0,
    'normal_reference_length_mm': 8.0,
    # Keep the recorded centreline unchanged, but derive section normals from
    # a local physical-scale smoothing so sub-millimetre centreline noise does
    # not rotate neighbouring planes independently.
    'normal_tangent_smoothing_mm': 4.0,
    'normal_continuity_max_angle_deg': 15.0,
    'normal_continuity_area_ratio': 1.25,
}


def _normal_search_policy(policy=None):
    settings = dict(_DEFAULT_NORMAL_SEARCH_POLICY)
    if policy:
        settings.update(policy)
    return settings


def _curvature_search_context(curvature, arc_length, context_mm):
    """Expand high-curvature search effort to an arc-length neighbourhood."""
    curvature = np.asarray(curvature, dtype=float)
    arc = np.asarray(arc_length, dtype=float)
    if len(curvature) < 2 or len(arc) != len(curvature) or context_mm <= 0:
        return np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    step = float(np.median(np.diff(arc)))
    if not np.isfinite(step) or step <= 1e-8:
        return np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    half_window = max(0, int(np.ceil(float(context_mm) / step)))
    if half_window == 0:
        return np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    return ndimage.maximum_filter1d(
        clean, size=2 * half_window + 1, mode='nearest')


def _normal_search_parameters(effective_curvature, policy=None):
    """Choose normal perturbation density from local curvature evidence."""
    settings = _normal_search_policy(policy)
    value = float(effective_curvature) if np.isfinite(effective_curvature) else 0.0
    if value >= float(settings['high_curvature_threshold']):
        return (int(settings['high_n_perturb']),
                float(settings['high_max_angle_deg']), 'high')
    if value >= float(settings['low_curvature_threshold']):
        return (int(settings['medium_n_perturb']),
                float(settings['medium_max_angle_deg']), 'medium')
    return (int(settings['low_n_perturb']),
            float(settings['low_max_angle_deg']), 'low')


def _normal_offset_values(max_angle_deg, n_perturb):
    """Return symmetric non-zero offsets, e.g. 60/20 -> -30..30 by 3 deg."""
    steps = max(2, int(n_perturb))
    if steps % 2:
        raise ValueError('single-degree normal search requires an even step count')
    step_deg = float(max_angle_deg) / steps
    return np.concatenate((
        -np.arange(steps // 2, 0, -1, dtype=float) * step_deg,
        np.arange(1, steps // 2 + 1, dtype=float) * step_deg,
    ))


def _generate_normal_candidates(normal, n_perturb=12, max_angle_deg=15,
                                reference_direction=None, return_offsets=False):
    """Generate a single-degree normal rotation in the reference-vector plane."""
    normal = normal / (np.linalg.norm(normal) + 1e-15)
    reference = np.asarray(reference_direction, dtype=float) if reference_direction is not None else None
    if reference is None or reference.shape != (3,) or not np.all(np.isfinite(reference)):
        reference = _make_orthonormal_basis(normal)[0]
    direction = reference - float(np.dot(reference, normal)) * normal
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-9:
        direction = _make_orthonormal_basis(normal)[0]
    else:
        direction /= direction_norm
    candidates = []
    for offset_deg in _normal_offset_values(max_angle_deg, n_perturb):
        pert = normal + np.tan(np.radians(offset_deg)) * direction
        pert /= np.linalg.norm(pert)
        candidates.append((pert, float(offset_deg)) if return_offsets else pert)
    return candidates


def _shape_score(area, aspect_ratio, circularity):
    """
    Return the smallest valid cross-section area.

    Aspect ratio, circularity, and diameter have already been applied as hard
    filters in ``_compute_cross_section``.  After that screening, selecting
    the minimum area prevents the candidate search from drifting into a larger
    neighbouring vessel at a confluence or side branch.
    """
    del aspect_ratio, circularity
    return float(area)


def _plane_angle_deg(first, second):
    """Angle between two plane normals, treating opposite signs as equal."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first) + 1e-15
    second /= np.linalg.norm(second) + 1e-15
    cosine = abs(float(np.clip(np.dot(first, second), -1.0, 1.0)))
    return float(np.degrees(np.arccos(cosine)))


def _select_continuous_section_candidate(candidates, previous_normal=None,
                                         previous_area=None,
                                         max_normal_change_deg=None,
                                         max_area_ratio=None):
    """Choose the minimum-area valid candidate within local continuity limits.

    Each position is still evaluated geometrically.  Continuity only prevents
    a single clipped/oblique candidate from replacing its neighbouring plane
    with an unrelated minimum-area slice.
    """
    if not candidates:
        return None
    if previous_normal is None:
        return min(candidates, key=lambda item: item['area'])

    for item in candidates:
        item['normal_change_deg'] = _plane_angle_deg(
            item['normal'], previous_normal)
    if max_normal_change_deg is not None and max_normal_change_deg > 0:
        normal_candidates = [
            item for item in candidates
            if item['normal_change_deg'] <= max_normal_change_deg
        ]
    else:
        normal_candidates = list(candidates)
    if not normal_candidates:
        # A genuine sharp bend can exceed the normal window.  Preserve the
        # closest plane instead of falling back to an arbitrary minimum area.
        normal_candidates = [min(
            candidates,
            key=lambda item: (item['normal_change_deg'], item['area']))]

    if previous_area is not None and previous_area > 0 and \
            max_area_ratio is not None and max_area_ratio > 1:
        lower = previous_area / max_area_ratio
        upper = previous_area * max_area_ratio
        area_candidates = [
            item for item in normal_candidates
            if lower <= item['area'] <= upper
        ]
        if area_candidates:
            return min(area_candidates, key=lambda item: item['area'])
        # No smooth-area candidate exists in the normal-consistent set.  The
        # closest area is more reliable than an isolated minimum-area artifact.
        return min(normal_candidates, key=lambda item: (
            abs(np.log(item['area'] / previous_area)), item['area']))
    return min(normal_candidates, key=lambda item: item['area'])


def _compute_cross_section(mesh, point, normal,
                           n_perturb=12, max_angle_deg=15,
                           max_eq_diameter=None,
                           min_eq_diameter=None,
                           ownership_factor=1.8,
                           max_aspect_ratio=4.0,
                            min_circularity=0.30,
                            return_normal=False,
                            return_normal_offset=False,
                            reference_direction=None,
                            return_raw=False,
                           return_extras=False,
                           previous_normal=None,
                           previous_area=None,
                           max_normal_change_deg=None,
                           max_area_ratio=None):
    """
    鲁棒截面: 扰动法线 → 形状硬过滤 → 综合评分选最佳.

    自适应判定 (无固定面积阈值, 仅靠几何形状):
      硬剔除:
        1. aspect_ratio > max_aspect_ratio (默认 4.0): 沿轴向薄片切 / 跨血管切
        2. circularity   < min_circularity  (默认 0.30): 形状极不规则
        3. eq_diameter   > max_eq_diameter (若给定): 越界穿透

      综合打分 (见 _shape_score): area × elongation_pen × irregularity_pen
      选 score 最小的候选, 等价于"面积小且形状接近圆"的真正垂直切.

    若所有候选都被形状过滤掉 → 返回 0 (该点截面记为缺失, 后续插值或 NaN).
    这比"硬选一个明显错误的"对训练集更友好 — 缺失值上层可处理, 错误值会污染统计.

    参数:
        max_eq_diameter:     等效直径上界 (mm), 默认 None 不限.
        ownership_factor:    中心线锚定裁剪圆半径 / 锚定内切半径.
        max_aspect_ratio:    硬剔除阈值, 默认 4.0.
        min_circularity:     硬剔除阈值, 默认 0.30.
        return_normal:       是否返回所选最佳法线 (供可视化复现).
        return_normal_offset: 是否返回所选法线在参考向量平面内的偏移角.
        return_raw:          是否追加原始未裁剪 area/perimeter 与锚定半径.
        return_extras:       是否额外返回 (n_components, solidity) — PVT/血栓
                             形状指标. 见 _section_one 文档.

    返回:
        默认                   : (area, perimeter)
        return_normal=True     : (area, perimeter, best_normal)
        return_raw=True        : 追加 (raw_area, raw_perimeter, anchor_radius, owned_radius)
        return_extras=True     : 末尾追加 (n_components, solidity)
    """
    normal = normal / (np.linalg.norm(normal) + 1e-15)

    candidates = _generate_normal_candidates(
        normal, n_perturb, max_angle_deg, reference_direction,
        return_offsets=True)

    valid_candidates = []
    for n, offset_deg in candidates:
        if return_extras:
            if return_raw:
                a, p, ar, circ, raw_a, raw_p, anchor_r, owned_r, ncomp, sol = _section_one(
                    mesh, point, n,
                    max_eq_diameter=max_eq_diameter,
                    min_eq_diameter=min_eq_diameter,
                    ownership_factor=ownership_factor,
                    return_metrics=True,
                    return_raw=True,
                    return_extras=True)
            else:
                a, p, ar, circ, ncomp, sol = _section_one(
                    mesh, point, n,
                    max_eq_diameter=max_eq_diameter,
                    min_eq_diameter=min_eq_diameter,
                    ownership_factor=ownership_factor,
                    return_metrics=True,
                    return_extras=True)
                raw_a, raw_p, anchor_r, owned_r = a, p, 0.0, 0.0
        else:
            if return_raw:
                a, p, ar, circ, raw_a, raw_p, anchor_r, owned_r = _section_one(
                    mesh, point, n,
                    max_eq_diameter=max_eq_diameter,
                    min_eq_diameter=min_eq_diameter,
                    ownership_factor=ownership_factor,
                    return_metrics=True,
                    return_raw=True)
            else:
                a, p, ar, circ = _section_one(
                    mesh, point, n,
                    max_eq_diameter=max_eq_diameter,
                    min_eq_diameter=min_eq_diameter,
                    ownership_factor=ownership_factor,
                    return_metrics=True)
                raw_a, raw_p, anchor_r, owned_r = a, p, 0.0, 0.0
            ncomp, sol = 0, 0.0
        if a <= 0:
            continue
        # 形状硬过滤
        if ar > max_aspect_ratio or circ < min_circularity:
            continue
        valid_candidates.append({
            'area': float(_shape_score(a, ar, circ)),
            'perimeter': float(p),
            'normal': n,
            'offset_deg': float(offset_deg),
            'raw_area': float(raw_a),
            'raw_perimeter': float(raw_p),
            'anchor_radius': float(anchor_r),
            'owned_radius': float(owned_r),
            'n_components': int(ncomp),
            'solidity': float(sol),
        })

    best = _select_continuous_section_candidate(
        valid_candidates,
        previous_normal=previous_normal,
        previous_area=previous_area,
        max_normal_change_deg=max_normal_change_deg,
        max_area_ratio=max_area_ratio)
    if best is None:
        # 所有候选均不合格 → 该点截面无效
        base = (0.0, 0.0)
        if return_normal:
            base = base + (normal,)
        if return_normal_offset:
            base = base + (0.0,)
        if return_raw:
            base = base + (0.0, 0.0, 0.0, 0.0)
        if return_extras:
            base = base + (0, 0.0)
        return base

    best_area = best['area']
    best_peri = best['perimeter']
    best_normal = best['normal']
    best_offset_deg = best['offset_deg']
    best_raw_area = best['raw_area']
    best_raw_peri = best['raw_perimeter']
    best_anchor_radius = best['anchor_radius']
    best_owned_radius = best['owned_radius']
    best_ncomp = best['n_components']
    best_solidity = best['solidity']

    base = (best_area, best_peri)
    if return_normal:
        base = base + (best_normal,)
    if return_normal_offset:
        base = base + (float(best_offset_deg),)
    if return_raw:
        base = base + (best_raw_area, best_raw_peri,
                       best_anchor_radius, best_owned_radius)
    if return_extras:
        base = base + (int(best_ncomp), float(best_solidity))
    return base


def _compute_tangents(coords, smooth_window=5):
    """
    中心线每点的切线方向.

    使用 ±half 邻居端点连线作为切线 (default window=5 ⇒ ±2 邻居),
    比 3 点中心差分更平滑 — 在中心线轻微抖动 / 分叉点近邻处更稳定,
    避免切平面与血管轴近似平行造成"沿轴薄片切"的错误截面.
    """
    M = len(coords)
    tangents = np.zeros((M, 3))
    half = max(1, smooth_window // 2)
    for i in range(M):
        lo = max(0, i - half)
        hi = min(M - 1, i + half)
        if hi == lo:
            tangents[i] = np.array([0, 0, 1])
            continue
        t = coords[hi] - coords[lo]
        norm = np.linalg.norm(t)
        tangents[i] = t / norm if norm > 1e-10 else np.array([0, 0, 1])
    return tangents


def _smooth_coords_for_normal_tangents(coords, arc_length, smoothing_mm):
    """Smooth only the coordinate copy used to estimate plane normals."""
    coords = np.asarray(coords, dtype=float)
    arc = np.asarray(arc_length, dtype=float)
    if len(coords) < 3 or len(arc) != len(coords) or smoothing_mm <= 0:
        return coords
    steps = np.diff(arc)
    steps = steps[np.isfinite(steps) & (steps > 1e-8)]
    if len(steps) == 0:
        return coords
    sigma_samples = float(smoothing_mm) / float(np.median(steps))
    if sigma_samples <= 0.25:
        return coords
    smoothed = ndimage.gaussian_filter1d(
        coords, sigma=sigma_samples, axis=0, mode='nearest')
    smoothed[0] = coords[0]
    smoothed[-1] = coords[-1]
    return smoothed


def _normal_reference_directions(coords, arc_length, normals,
                                 reference_length_mm=8.0):
    """Build one forward fixed-distance vector per centreline point."""
    coords = np.asarray(coords, dtype=float)
    arc = np.asarray(arc_length, dtype=float)
    normals = np.asarray(normals, dtype=float)
    directions = np.zeros_like(normals)
    length = max(float(reference_length_mm), 1e-3)
    for idx, point in enumerate(coords):
        target = float(arc[idx] + length)
        if target <= float(arc[-1]):
            far = _point_at_arc(coords, arc, target)
            direction = far - point
        else:
            back = _point_at_arc(coords, arc, max(0.0, float(arc[idx] - length)))
            direction = point - back
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            direction = _make_orthonormal_basis(normals[idx])[0]
        else:
            direction /= norm
        directions[idx] = direction
    return directions


def _robust_radius_for_section_filter(inscribed_radius, window=15,
                                      low_ratio=0.55):
    """
    Build a conservative radius reference for the max-diameter filter.

    `inscribed_radius` is the nearest distance from the centerline point to the
    STL surface. When the centerline drifts toward the vessel wall, that value
    can drop abruptly even though the true cross-section is not narrow. Using
    the raw dip as a hard upper bound makes valid sections look "too large" and
    forces the algorithm to keep tiny clipped sections. We raise only isolated
    low dips to their local median; sustained narrow segments keep their smaller
    radius because the local median also drops.
    """
    r = np.asarray(inscribed_radius, dtype=float)
    out = r.copy()
    if len(r) < 3:
        return out

    half = max(1, window // 2)
    for i in range(len(r)):
        if not np.isfinite(r[i]) or r[i] <= 0:
            continue
        lo, hi = max(0, i - half), min(len(r), i + half + 1)
        win = r[lo:hi]
        win = win[np.isfinite(win) & (win > 0)]
        if len(win) < 3:
            continue
        med = float(np.median(win))
        if med > 0 and r[i] < low_ratio * med:
            out[i] = med
    return out


def _remove_rate_outliers(area, perimeter, eq_diameter, arc_length,
                          max_rate_per_mm=0.5):
    """
    沿管轴"变化速率"过滤: 真实血管的直径沿管轴是缓变的, 哪怕在缩窄/狭窄处,
    每 mm 的相对直径变化也很少超过 50% (max_rate_per_mm=0.5).
    单点出现急剧塌陷/急剧膨胀 → 截面渗透到邻近血管 / 沿轴薄片切 / 分叉伪影.

    判定: 对每个有效采样点 i, 找其最近的左右有效邻居 j ∈ {prev, next},
    计算相对变化率
        r_j = |D[i] − D[j]| / (mean_D · Δs_ij)        单位 1/mm
    - 若两侧邻居均给出 r_j > max_rate_per_mm → 孤立尖峰, 剔除
    - 若只有一侧邻居 (段端), 该侧 r_j > 2·max_rate_per_mm 才剔除
      (段端单边判据更严, 避免误伤端点处的真实收口)

    与 `_remove_local_outliers` (MAD) 互补:
      MAD : 适合捕捉与"局部分布"显著偏离的点 (含成簇异常)
      rate: 适合捕捉单点"突变 / 阶梯", 含图像 2.png / 3.png 中 MPV 沿轴
            单点截面塌陷 (大血管中突现 1.8mm² 极小值) 这类伪影.

    参数:
        max_rate_per_mm: 允许的相对直径变化率上限 (1/mm), 默认 0.5

    返回:
        (area, perimeter, eq_diameter, n_removed) — 原地修改
    """
    M = len(area)
    if M < 3 or max_rate_per_mm <= 0:
        return area, perimeter, eq_diameter, 0

    valid = eq_diameter > 0
    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 3:
        return area, perimeter, eq_diameter, 0

    flagged = np.zeros(M, dtype=bool)

    for k, i in enumerate(valid_idx):
        rates = []
        neighbors = []
        if k > 0:
            neighbors.append(valid_idx[k - 1])
        if k < len(valid_idx) - 1:
            neighbors.append(valid_idx[k + 1])
        for j in neighbors:
            ds = abs(arc_length[i] - arc_length[j])
            if ds < 1e-6:
                continue
            mean_d = 0.5 * (eq_diameter[i] + eq_diameter[j])
            if mean_d < 1e-6:
                continue
            rates.append(abs(eq_diameter[i] - eq_diameter[j]) / (mean_d * ds))

        if not rates:
            continue
        if len(rates) >= 2:
            # 两侧都有邻居: 两侧均超阈才剔除 (剔除孤立尖峰)
            if all(r > max_rate_per_mm for r in rates):
                flagged[i] = True
        else:
            # 段端单边: 阈值加倍, 更保守
            if rates[0] > 2.0 * max_rate_per_mm:
                flagged[i] = True

    n_removed = int(np.sum(flagged))
    if n_removed > 0:
        area[flagged] = 0.0
        perimeter[flagged] = 0.0
        eq_diameter[flagged] = 0.0
    return area, perimeter, eq_diameter, n_removed


def _remove_local_outliers(area, perimeter, eq_diameter,
                           window=15, mad_factor=3.5):
    """
    沿中心线的局部一致性检测: 用滑窗中位数 + MAD 自适应剔除异常截面.

    思想: 真实血管的横截面沿管轴是缓变的. 若某点的等效直径相对其
    局部邻居 (±half 个采样点) 的中位数偏差超过 mad_factor × 1.4826 × MAD,
    认为该点是污染 (邻近血管渗透 / 沿轴薄片), 标记为 0 (后续 NaN).

    完全自适应: 阈值由数据自身分布决定, 无任何硬编码尺寸. 在 MPV 这种
    粗血管处容忍大值, 在 LGV 等细血管处容忍小值.

    参数:
        window:     滑窗大小, 默认 15 (≈ 5 mm 在 1mm 间距下).
        mad_factor: MAD 倍数门槛 (近似 σ 倍数), 默认 3.5.

    返回:
        (area, perimeter, eq_diameter, n_removed) —— 原数组就地修改.
    """
    M = len(area)
    if M < window:
        return area, perimeter, eq_diameter, 0

    valid = eq_diameter > 0
    if int(np.sum(valid)) < window // 2:
        return area, perimeter, eq_diameter, 0

    half = window // 2
    n_removed = 0
    flagged = np.zeros(M, dtype=bool)

    for i in range(M):
        if not valid[i]:
            continue
        lo, hi = max(0, i - half), min(M, i + half + 1)
        # 排除自身, 取邻居有效值
        win = eq_diameter[lo:hi]
        win = win[win > 0]
        if len(win) < 5:
            continue
        med = float(np.median(win))
        mad = float(np.median(np.abs(win - med)))
        if mad < 1e-6:
            continue
        # 1.4826·MAD ≈ σ (正态)
        sigma_est = 1.4826 * mad
        deviation = abs(eq_diameter[i] - med) / sigma_est
        if deviation > mad_factor:
            flagged[i] = True

    if np.any(flagged):
        n_removed = int(np.sum(flagged))
        area[flagged] = 0.0
        perimeter[flagged] = 0.0
        eq_diameter[flagged] = 0.0

    return area, perimeter, eq_diameter, n_removed


def _torsion_sliding_window(coords, arc_length, smooth_sigma=2.0,
                             min_curvature_for_torsion=1e-3):
    """
    Frenet-Serret 挠率 τ (1/mm), 描述中心线在 3D 空间的"扭转"程度.

    曲率 κ 描述"弯不弯"; 挠率 τ 描述"扭不扭". 平面曲线 τ=0; 螺旋线 τ>0.
    门静脉海绵样变 / 重度迂曲的代偿血管 → τ 显著升高.

    公式 (对弧长 s 的导数):
        τ = ((P' × P'') · P''') / |P' × P''|²

    数值实现:
      1. 用 Gaussian 平滑坐标 (σ=smooth_sigma 个点), 抑制离散噪声
      2. np.gradient 对弧长求一/二/三阶导数
      3. 直线段 (|P' × P''| 几乎 0, ≡ κ ≈ 0) 数值不稳定 → 置 NaN

    参数:
        coords:                    (N, 3) 中心线坐标
        arc_length:                (N,) 累积弧长 (单调递增)
        smooth_sigma:              坐标平滑核宽 (点数), 默认 2
        min_curvature_for_torsion: 曲率低于此值的点上挠率置 NaN
                                    (避免直线段的 0/0 数值噪声)

    返回:
        (N,) 挠率数组, 不可信处为 NaN.
    """
    N = len(coords)
    if N < 5:
        return np.full(N, np.nan)
    try:
        from scipy.ndimage import gaussian_filter1d
        coords_s = gaussian_filter1d(coords, sigma=smooth_sigma,
                                      axis=0, mode='nearest')
    except Exception:
        coords_s = np.asarray(coords, dtype=float)

    s = np.asarray(arc_length, dtype=float)
    # 弧长退化 (重复点) 时 np.gradient 会发散
    if not np.all(np.diff(s) > 1e-8):
        return np.full(N, np.nan)

    p1 = np.gradient(coords_s, s, axis=0)
    p2 = np.gradient(p1, s, axis=0)
    p3 = np.gradient(p2, s, axis=0)

    cross_12 = np.cross(p1, p2)                   # (N, 3)
    denom = np.sum(cross_12 ** 2, axis=1)         # |P'×P''|²

    numer = np.einsum('ij,ij->i', cross_12, p3)   # (P'×P'') · P'''
    torsion = numer / (denom + 1e-12)

    # 在曲率近 0 处, 数值不稳定 → NaN
    curv = np.sqrt(np.maximum(denom, 0.0)) / (
        np.linalg.norm(p1, axis=1) ** 3 + 1e-12)
    bad = (curv < min_curvature_for_torsion) | ~np.isfinite(torsion)
    torsion[bad] = np.nan
    return torsion


def _curvature_sliding_window(coords, window=7):
    """滑窗法离散曲率 (1/mm)"""
    N = len(coords)
    curvatures = np.zeros(N)
    if N < 3:
        return curvatures
    half = window // 2
    for i in range(N):
        lo, hi = max(0, i - half), min(N - 1, i + half)
        a = coords[i] - coords[lo]
        b = coords[hi] - coords[i]
        la, lb = np.linalg.norm(a), np.linalg.norm(b)
        lc = np.linalg.norm(coords[hi] - coords[lo])
        if la < 1e-10 or lb < 1e-10 or lc < 1e-10:
            continue
        area2 = np.linalg.norm(np.cross(a, b))
        curvatures[i] = 2.0 * area2 / (la * lb * lc)
    return curvatures


# ============================================================
# 沿分支提取逐点特征
# ============================================================

def _compute_inscribed_radius_per_point(coords, mesh):
    """
    对每个中心线点, 计算其到 STL 表面的最近距离 (≈ 局部内切球半径).
    使用 trimesh.proximity.signed_distance: 内部为正, 外部为负.

    返回 (M,) ndarray, 单位 mm. 失败时返回 0 数组.
    """
    try:
        import trimesh.proximity
        sd = trimesh.proximity.signed_distance(mesh, coords)
        sd = np.asarray(sd, dtype=float)
        # 中心线点应该在 mesh 内部 → sd > 0; 取正值, 异常点置 0
        sd = np.clip(sd, 0.0, None)
        return sd
    except Exception as e:
        print(f"    [warn] inscribed_radius 计算失败: {e}, 用 0 填充")
        return np.zeros(len(coords))


def _effective_section_step(n_centerline_points, requested_step,
                            max_section_samples=None):
    """Cap redundant raw planes while retaining a dense resampling source."""
    step = max(1, int(requested_step))
    if max_section_samples is None or n_centerline_points <= 2:
        return step
    target = max(2, int(max_section_samples))
    budget_step = int(np.ceil((n_centerline_points - 1) / (target - 1)))
    return max(step, budget_step)


def _extract_branch_raw_profile(branch_path, nodes, mesh,
                                dt=None, origin=None, pitch=None,
                                curvature_window=7, section_step=1,
                                inscribed_factor=1.8,
                                ownership_factor=1.8,
                                max_diameter_rate_per_mm=0.5,
                                branch_coords=None,
                                max_section_samples=None,
                                normal_search_policy=None):
    """
    沿一段中心线提取逐点剖面.

    inscribed_factor: 等效直径相对于内切直径 (2*r) 的最大允许倍数.
        越界则该位置截面记为 0 (后续会变 NaN).
        默认 1.8: 真实截面通常 1.0~1.4 倍 (圆形=1, 椭圆稍大).
        分叉点处穿透到邻近血管时, 比值会显著 > 2.

    ownership_factor: 中心线锚定清洗半径倍数.
        clean_area = raw_section ∩ circle(center, ownership_factor*r_anchor).
        默认 1.8, 保留椭圆主体并裁掉分叉污染外伸区域.

    max_diameter_rate_per_mm: 沿管轴允许的等效直径相对变化率 (1/mm).
        超过此速率的孤立点视为伪影 (单点塌陷/膨胀), 见 _remove_rate_outliers.
        默认 0.5 = 每 mm 最多 50% 相对变化.

    返回: dict (含原始 area/eq_diameter/inscribed_radius/...) 或 None.
    """
    if len(branch_path) < 2:
        return None

    if branch_coords is not None:
        coords = np.asarray(branch_coords, dtype=float)
    else:
        coords = path_to_coords(branch_path, nodes)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 2:
        return None
    M = len(coords)

    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(diffs)))

    search_policy = _normal_search_policy(normal_search_policy)
    normal_tangent_coords = _smooth_coords_for_normal_tangents(
        coords, arc_length, search_policy['normal_tangent_smoothing_mm'])
    tangents = _compute_tangents(normal_tangent_coords)
    curvature = _curvature_sliding_window(coords, curvature_window)
    search_curvature = _curvature_search_context(
        curvature, arc_length, search_policy['high_curvature_context_mm'])
    normal_references = _normal_reference_directions(
        coords, arc_length, tangents,
        search_policy['normal_reference_length_mm'])

    # ---- 内切半径 (来自 STL 表面距离, 用于边界效应过滤) ----
    inscribed_radius = _compute_inscribed_radius_per_point(coords, mesh)
    radius_for_filter = _robust_radius_for_section_filter(
        inscribed_radius, window=15, low_ratio=0.55)

    area = np.zeros(M)           # clean/owned area, downstream default
    perimeter = np.zeros(M)      # clean/owned perimeter
    raw_area = np.zeros(M)       # original STL section before owned clipping
    raw_perimeter = np.zeros(M)
    anchor_radius = np.zeros(M)
    owned_radius = np.zeros(M)
    solidity = np.zeros(M)            # (新) area / convex_hull_area
    n_components = np.zeros(M, dtype=np.int16)  # (新) lumen 连通分量数
    # Persist the selected plane normal so the Web surface contour can replay
    # the same section rather than falling back to a tangent-derived circle.
    section_normal = tangents.copy()
    section_normal_offset_deg = np.zeros(M)

    effective_section_step = _effective_section_step(
        M, section_step, max_section_samples)
    indices = list(range(0, M, effective_section_step))
    if indices[-1] != M - 1:
        indices.append(M - 1)

    n_success = 0
    n_rejected = 0
    n_relaxed_bounds = 0
    normal_search_counts = {'low': 0, 'medium': 0, 'high': 0}
    normal_search_candidates = 0
    previous_normal = None
    previous_area = None
    for idx in indices:
        # 局部允许的截面等效直径上限: inscribed_factor × 内切直径
        r_loc = radius_for_filter[idx]
        max_eq_d = (2.0 * r_loc * inscribed_factor) if r_loc > 0.5 else None
        min_eq_d = (2.0 * r_loc * 0.55) if r_loc > 0.5 else None
        n_perturb, max_angle_deg, search_level = _normal_search_parameters(
            search_curvature[idx], search_policy)
        normal_search_counts[search_level] += 1
        normal_search_candidates += n_perturb
        a, p, best_normal, best_offset_deg, raw_a, raw_p, anchor_r, owned_r, ncomp, sol = _compute_cross_section(
            mesh, coords[idx], tangents[idx],
            n_perturb=n_perturb,
            max_angle_deg=max_angle_deg,
            max_eq_diameter=max_eq_d,
            min_eq_diameter=min_eq_d,
            ownership_factor=ownership_factor,
            return_normal=True,
            return_normal_offset=True,
            reference_direction=normal_references[idx],
            return_raw=True,
            return_extras=True,
            previous_normal=previous_normal,
            previous_area=previous_area,
            max_normal_change_deg=search_policy[
                'normal_continuity_max_angle_deg'],
            max_area_ratio=search_policy['normal_continuity_area_ratio'])
        if a <= 0 and (max_eq_d is not None or min_eq_d is not None):
            # The signed-distance radius is a useful prior, but it can be too
            # small when a point is off-center or close to a topology change.
            # Retry without diameter bounds and let shape/outlier filters judge
            # the section instead of collapsing the entire profile to zeros.
            relaxed = _compute_cross_section(
                mesh, coords[idx], tangents[idx],
                n_perturb=n_perturb,
                max_angle_deg=max_angle_deg,
                max_eq_diameter=None,
                min_eq_diameter=None,
                ownership_factor=ownership_factor,
                return_normal=True,
                return_normal_offset=True,
                reference_direction=normal_references[idx],
                return_raw=True,
                return_extras=True,
                previous_normal=previous_normal,
                previous_area=previous_area,
                max_normal_change_deg=search_policy[
                    'normal_continuity_max_angle_deg'],
                max_area_ratio=search_policy['normal_continuity_area_ratio'])
            if relaxed[0] > 0:
                a, p, best_normal, best_offset_deg, raw_a, raw_p, anchor_r, owned_r, ncomp, sol = relaxed
                n_relaxed_bounds += 1
        area[idx] = a
        perimeter[idx] = p
        raw_area[idx] = raw_a
        raw_perimeter[idx] = raw_p
        anchor_radius[idx] = anchor_r
        owned_radius[idx] = owned_r
        solidity[idx] = sol
        n_components[idx] = ncomp
        section_normal[idx] = best_normal
        section_normal_offset_deg[idx] = best_offset_deg
        if a > 0:
            n_success += 1
            previous_normal = best_normal
            previous_area = a
        elif r_loc > 0.5:
            # 计算成功了但被尺寸/形状过滤掉
            n_rejected += 1

    # ---- 局部一致性后处理 ----
    # 仅在采样点上做异常剔除 (因为只有这些点是真实计算结果, 其余是 0)
    sampled_idx = np.array(indices, dtype=int)
    sampled_area = area[sampled_idx].copy()
    sampled_peri = perimeter[sampled_idx].copy()
    sampled_eq = np.sqrt(4.0 * sampled_area / np.pi)
    sampled_eq[sampled_area <= 0] = 0.0
    sampled_arc = arc_length[sampled_idx]

    # (1) MAD 局部异常 (与局部分布偏离)
    sampled_area, sampled_peri, sampled_eq, n_outliers = \
        _remove_local_outliers(sampled_area, sampled_peri, sampled_eq,
                               window=15, mad_factor=3.5)
    # (2) 沿管轴变化速率过滤 (单点突变/塌陷, 与 MAD 互补)
    sampled_area, sampled_peri, sampled_eq, n_rate_outliers = \
        _remove_rate_outliers(sampled_area, sampled_peri, sampled_eq,
                              sampled_arc,
                              max_rate_per_mm=max_diameter_rate_per_mm)
    # 写回
    area[sampled_idx] = sampled_area
    perimeter[sampled_idx] = sampled_peri
    # 同步: 被剔除的位置 (area=0) 把 solidity / n_components 也清掉, 防止
    # 后续 resample 把无效残留传到 100 点输出.
    zeroed = sampled_area <= 0
    if np.any(zeroed):
        raw_area[sampled_idx[zeroed]] = 0.0
        raw_perimeter[sampled_idx[zeroed]] = 0.0
        anchor_radius[sampled_idx[zeroed]] = 0.0
        owned_radius[sampled_idx[zeroed]] = 0.0
        solidity[sampled_idx[zeroed]] = 0.0
        n_components[sampled_idx[zeroed]] = 0

    n_final_success = int(np.sum(area[sampled_idx] > 0))

    # 对跳过的点插值 (仅对成功截面插值, 0 值不插)
    if effective_section_step > 1 and n_success >= 2:
        sampled_arc = arc_length[indices]
        for arr in [area, perimeter, raw_area, raw_perimeter,
                    anchor_radius, owned_radius, solidity]:
            sampled = arr[indices]
            valid = sampled > 0
            if np.sum(valid) >= 2:
                f = interp1d(sampled_arc[valid], sampled[valid],
                             kind='linear', bounds_error=False,
                             fill_value=(sampled[valid][0], sampled[valid][-1]))
                arr[:] = np.clip(f(arc_length), 0, None)

    eq_diameter = np.sqrt(4.0 * area / np.pi)
    eq_diameter[area <= 0] = 0.0
    raw_eq_diameter = np.sqrt(4.0 * raw_area / np.pi)
    raw_eq_diameter[raw_area <= 0] = 0.0

    circularity = np.zeros(M)
    valid_mask = (area > 0) & (perimeter > 0)
    circularity[valid_mask] = (4.0 * np.pi * area[valid_mask]) / (perimeter[valid_mask] ** 2)

    # ---- 形状/水力派生通道 ----
    # 水力直径 D_h = 4 A / P (适用于任意非圆截面)
    hydraulic_diameter = np.zeros(M)
    hydraulic_diameter[valid_mask] = (4.0 * area[valid_mask]
                                      / perimeter[valid_mask])
    # 瓶颈比 = 2·r_inscribed / D_eq ∈ (0, 1]
    # 圆形 ≈ 1; 月牙/血栓挤压 → << 1 (真实通道宽 比 乐观估计窄)
    r_insc_to_r_eq_ratio = np.zeros(M)
    eq_valid = eq_diameter > 1e-6
    r_insc_to_r_eq_ratio[eq_valid] = np.clip(
        (2.0 * inscribed_radius[eq_valid]) / eq_diameter[eq_valid],
        0.0, 1.5)
    # solidity 已经在采样点直接得到, 用 0 标记缺失. circularity 同步,
    # 在 _resample_profile 里会按 area>0 做插值.

    # 曲率 + 挠率 (中心线本身的几何, 不受截面有效性影响)
    torsion = _torsion_sliding_window(coords, arc_length)
    chord = float(np.linalg.norm(coords[-1] - coords[0])) if len(coords) >= 2 else 0.0
    total_length = float(arc_length[-1]) if len(arc_length) else 0.0
    arc_chord = total_length / chord if chord > 1e-8 else 1.0
    finite_curv = curvature[np.isfinite(curvature)]

    return {
        'arc_length': arc_length,
        'centerline_coords': coords,
        'section_normal': section_normal,
        'section_normal_reference': normal_references,
        'section_normal_offset_deg': section_normal_offset_deg,
        'area': area,
        'perimeter': perimeter,
        'eq_diameter': eq_diameter,
        'raw_area': raw_area,
        'raw_perimeter': raw_perimeter,
        'raw_eq_diameter': raw_eq_diameter,
        'anchor_radius': anchor_radius,
        'owned_radius': owned_radius,
        'hydraulic_diameter': hydraulic_diameter,
        'circularity': circularity,
        'solidity': solidity,
        'n_components': n_components.astype(float),  # 便于和其它通道共用插值
        'r_insc_to_r_eq_ratio': r_insc_to_r_eq_ratio,
        'curvature': curvature,
        'normal_search_curvature': search_curvature,
        'torsion': torsion,
        'inscribed_radius': inscribed_radius,
        '_n_sampled': len(indices),
        '_section_step_effective': int(effective_section_step),
        '_normal_search_policy': search_policy,
        '_normal_search_counts': normal_search_counts,
        '_normal_search_candidate_count': int(normal_search_candidates),
        '_n_success': n_success,
        '_n_final_success': n_final_success,
        '_n_rejected_oversize': n_rejected,
        '_n_relaxed_bounds': int(n_relaxed_bounds),
        '_n_local_outliers': int(n_outliers),
        '_n_rate_outliers': int(n_rate_outliers),
        '_centerline_chord_mm': chord,
        '_centerline_arc_chord_tortuosity': float(arc_chord),
        '_centerline_mean_curvature': float(np.mean(finite_curv)) if len(finite_curv) else 0.0,
        '_centerline_max_curvature': float(np.max(finite_curv)) if len(finite_curv) else 0.0,
    }


def _copy_section_values(profile, src_idx, dst_mask, keys):
    """把一个可信截面的主通道复制到一组目标点。"""
    n = len(profile['position'])
    dst_idx = np.where(dst_mask)[0]
    if len(dst_idx) == 0:
        return
    for key in keys:
        if key not in profile:
            continue
        values = list(profile[key])
        if src_idx < 0 or src_idx >= len(values):
            continue
        src_val = values[src_idx]
        for i in dst_idx:
            if 0 <= i < n:
                values[i] = src_val
        profile[key] = values


def _refresh_dA_ds_norm(profile):
    """根据当前 area 重新计算归一化面积变化率。"""
    try:
        area = np.asarray(profile.get('area', []), dtype=float)
        arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
        if len(area) != len(arc) or len(area) < 3 or not np.all(np.diff(arc) > 0):
            return
        if np.sum(np.isfinite(area) & (area > 0)) < 3:
            profile['dA_ds_norm'] = [float('nan')] * len(area)
            return
        grad = np.gradient(area, arc)
        with np.errstate(divide='ignore', invalid='ignore'):
            dA_ds = grad / np.where(area > 1e-6, area, np.nan)
        dA_ds[(area <= 0) | ~np.isfinite(area)] = np.nan
        profile['dA_ds_norm'] = dA_ds.tolist()
    except Exception:
        return


def _mask_implausibly_small_sections(profile,
                                     min_eq_to_inscribed_ratio=1.10,
                                     min_inscribed_radius=0.5):
    """
    Remove section fragments that are too small to be compatible with the
    centerline-to-surface distance.

    For a valid lumen section, the equivalent diameter should not be far below
    the local inscribed-radius scale. Very small values here are almost always
    mesh-plane polygonization fragments or branch-junction artifacts. Masking
    them prevents junction handling from copying a bad "minimum valid" section
    across multiple points.
    """
    if profile is None:
        return profile

    try:
        eq = np.asarray(profile.get('eq_diameter', []), dtype=float)
        ins = np.asarray(profile.get('inscribed_radius', []), dtype=float)
    except Exception:
        profile['n_implausibly_small_sections'] = 0
        return profile

    n = len(eq)
    if n == 0 or len(ins) != n:
        profile['n_implausibly_small_sections'] = 0
        return profile

    bad = (
        np.isfinite(eq) & (eq > 0)
        & np.isfinite(ins) & (ins > min_inscribed_radius)
        & (eq < min_eq_to_inscribed_ratio * ins)
    )
    bad_idx = np.where(bad)[0]
    if len(bad_idx) == 0:
        profile['implausibly_small_section'] = [0.0] * n
        profile['n_implausibly_small_sections'] = 0
        return profile

    section_keys = [
        'area', 'perimeter', 'eq_diameter',
        'raw_area', 'raw_perimeter', 'raw_eq_diameter',
        'anchor_radius', 'owned_radius',
        'circularity', 'hydraulic_diameter', 'solidity',
        'r_insc_to_r_eq_ratio', 'n_components', 'dA_ds_norm'
    ]
    for key in section_keys:
        if key not in profile:
            continue
        values = list(profile[key])
        if len(values) != n:
            continue
        for i in bad_idx:
            values[int(i)] = float('nan')
        profile[key] = values

    marker = [0.0] * n
    for i in bad_idx:
        marker[int(i)] = 1.0
    profile['implausibly_small_section'] = marker
    profile['n_implausibly_small_sections'] = int(len(bad_idx))
    return profile


def _apply_endpoint_mask(profile, edge_margin_pct=0.05,
                         edge_margin_mm=8.0,
                         branchpoint_arcs=None,
                         terminal_start=True,
                         terminal_end=True,
                         junction_policy='min_valid'):
    """
    处理段端点/交叉点附近的截面值。

    真实血管末端附近仍标记为 NaN, 避免 STL 开口/收口伪影。
    分叉/交叉点附近不再丢弃: 默认用该段可信区域的最小 clean area
    对应截面替换这些点, 让它们参与平均面积等统计, 但不再可能成为
    错误的最大截面。

    判定:
      - 真实末端: 距起/终点 < edge_margin_mm 或落在端点百分比保护带
      - 交叉点: 距 branchpoint_arcs 任一弧长 < junction_margin

    junction_policy:
      - 'min_valid': 用非末端、非交叉保护区中 clean area 最小的截面替换
      - 'cap_min':  只把交叉区中大于最小可信面积的点封顶到最小可信截面
      - 'keep':     交叉区不处理

    参数:
        profile:           _resample_profile 返回的 dict (100 点剖面)
        edge_margin_pct:   端点保护比例 (默认 0.05 = 前后 5%)
        edge_margin_mm:    端点保护绝对距离 mm (默认 8.0)
        branchpoint_arcs:   当前段路径上所有分叉点的弧长位置(mm)
        terminal_start/end: 当前段首/尾是否真实血管末端; 分叉点端不是末端

    返回:
        修改后的 profile (原地修改)
    """
    if profile is None:
        return profile

    n = len(profile['position'])
    pos = np.array(profile['position'])  # 0..1
    arc = np.array(profile['arc_length_mm'])  # 0..total_length
    total = profile.get('total_length_mm', arc[-1] if len(arc) > 0 else 0)

    # 真实末端保护: 只对非分叉的开口/末端置 NaN.
    start_pct_mask = pos < edge_margin_pct
    end_pct_mask = pos > 1 - edge_margin_pct

    dist_to_start = arc
    dist_to_end = total - arc
    start_mm_mask = dist_to_start < edge_margin_mm
    end_mm_mask = dist_to_end < edge_margin_mm

    terminal_mask = np.zeros(n, dtype=bool)
    if terminal_start:
        terminal_mask |= start_pct_mask | start_mm_mask
    if terminal_end:
        terminal_mask |= end_pct_mask | end_mm_mask

    # 交叉点保护: 不丢弃, 用本段可信最小截面替换/封顶.
    junction_mask = np.zeros(n, dtype=bool)
    branchpoint_arcs = branchpoint_arcs or []
    junction_margin = max(float(edge_margin_mm), float(total) * edge_margin_pct)
    for bp_arc in branchpoint_arcs:
        try:
            bp_arc = float(bp_arc)
        except Exception:
            continue
        junction_mask |= np.abs(arc - bp_arc) < junction_margin
    junction_mask &= ~terminal_mask

    # 标记的 keys (截面相关特征 + 新增形状/水力派生)
    section_keys = ['area', 'perimeter', 'eq_diameter',
                    'raw_area', 'raw_perimeter', 'raw_eq_diameter',
                    'anchor_radius', 'owned_radius',
                    'circularity', 'inscribed_radius',
                    'hydraulic_diameter', 'solidity',
                    'r_insc_to_r_eq_ratio', 'n_components',
                    'dA_ds_norm']

    n_masked = int(np.sum(terminal_mask))
    if n_masked > 0:
        for key in section_keys:
            if key in profile:
                values = list(profile[key])
                for i in range(n):
                    if terminal_mask[i]:
                        values[i] = float('nan')
                profile[key] = values

    n_junction = int(np.sum(junction_mask))
    n_junction_replaced = 0
    area = np.asarray(profile.get('area', []), dtype=float)
    trusted_mask = (
        np.isfinite(area) & (area > 0)
        & ~terminal_mask & ~junction_mask
    )
    if n_junction > 0 and junction_policy in ('min_valid', 'cap_min'):
        reference_mask = trusted_mask
        if not np.any(reference_mask):
            # 短段可能几乎全在交叉保护区内; 这时退回到整段有效最小值,
            # 仍然避免交叉区异常大截面成为最大截面.
            reference_mask = np.isfinite(area) & (area > 0) & ~terminal_mask
        if np.any(reference_mask):
            trusted_idx = np.where(reference_mask)[0]
            min_idx = int(trusted_idx[np.argmin(area[trusted_idx])])
            main_keys = ['area', 'perimeter', 'eq_diameter',
                         'hydraulic_diameter', 'circularity', 'solidity',
                         'r_insc_to_r_eq_ratio', 'n_components']
            if junction_policy == 'min_valid':
                replace_mask = junction_mask
            else:
                replace_mask = junction_mask & np.isfinite(area) & (
                    area > area[min_idx])
            _copy_section_values(profile, min_idx, replace_mask, main_keys)
            n_junction_replaced = int(np.sum(replace_mask))

    # 标记哪些点来自交叉区替换, 便于可视化和诊断.
    marker = [0.0] * n
    for i in np.where(junction_mask)[0]:
        marker[int(i)] = 1.0
    profile['junction_replaced'] = marker
    _refresh_dA_ds_norm(profile)

    # 元信息记录
    profile['edge_margin_pct'] = float(edge_margin_pct)
    profile['edge_margin_mm'] = float(edge_margin_mm)
    profile['n_masked_endpoints'] = n_masked
    profile['n_junction_protected'] = n_junction
    profile['n_junction_replaced'] = n_junction_replaced
    profile['junction_policy'] = junction_policy

    return profile

def _resample_profile(raw_profile, n_points=100):
    """
    重采样到 n_points (沿弧长均匀)。

    修正: 对面积/周长等截面特征, 只用 area>0 的原始点插值,
          避免未采样点的 0 值污染最大值。
    """
    if raw_profile is None:
        return None

    arc = raw_profile['arc_length']
    total_length = arc[-1]
    if total_length < 1e-6:
        return None

    t_raw = arc / total_length
    # The extraction path has already been resampled to n_points.  Reusing its
    # exact arc fractions prevents selected normal-grid offsets from being
    # interpolated away from the discrete audit values.
    t_uniform = t_raw.copy() if len(t_raw) == n_points else np.linspace(0, 1, n_points)

    result = {
        'position': t_uniform.tolist(),
        'arc_length_mm': (t_uniform * total_length).tolist(),
        'total_length_mm': float(total_length),
        'n_raw_points': len(arc),
        'profile_sample_index': list(range(n_points)),
        'profile_sample_count': int(n_points),
        'n_section_success': raw_profile.get('_n_success', 0),
        'centerline_chord_mm': float(raw_profile.get('_centerline_chord_mm', 0.0)),
        'centerline_arc_chord_tortuosity': float(
            raw_profile.get('_centerline_arc_chord_tortuosity', 1.0)),
        'centerline_mean_curvature': float(
            raw_profile.get('_centerline_mean_curvature', 0.0)),
        'centerline_max_curvature': float(
            raw_profile.get('_centerline_max_curvature', 0.0)),
    }

    centerline_coords = np.asarray(raw_profile.get('centerline_coords', []), dtype=float)
    if centerline_coords.ndim == 2 and centerline_coords.shape[1] == 3 and len(centerline_coords) == len(t_raw):
        for axis, key in enumerate(('centerline_x', 'centerline_y', 'centerline_z')):
            mask = np.concatenate(([True], np.diff(t_raw) > 1e-10))
            t_c = t_raw[mask]
            values = centerline_coords[:, axis][mask]
            if len(t_c) >= 2:
                f = interp1d(t_c, values, kind='linear',
                             bounds_error=False,
                             fill_value=(values[0], values[-1]))
                result[key] = f(t_uniform).tolist()
            else:
                result[key] = [float(values[0])] * n_points

    section_normal = np.asarray(raw_profile.get('section_normal', []), dtype=float)
    if section_normal.shape == (len(t_raw), 3):
        interpolated_normal = np.empty((n_points, 3), dtype=float)
        mask = np.concatenate(([True], np.diff(t_raw) > 1e-10))
        t_c = t_raw[mask]
        for axis in range(3):
            values = section_normal[:, axis][mask]
            if len(t_c) >= 2:
                interpolated_normal[:, axis] = interp1d(
                    t_c, values, kind='linear', bounds_error=False,
                    fill_value=(values[0], values[-1]))(t_uniform)
            else:
                interpolated_normal[:, axis] = float(values[0])
        norms = np.linalg.norm(interpolated_normal, axis=1)
        valid_normals = norms > 1e-9
        interpolated_normal[valid_normals] /= norms[valid_normals, None]
        interpolated_normal[~valid_normals] = np.array([0.0, 0.0, 1.0])
        for axis, key in enumerate(('section_normal_x', 'section_normal_y',
                                    'section_normal_z')):
            result[key] = interpolated_normal[:, axis].tolist()

    section_reference = np.asarray(
        raw_profile.get('section_normal_reference', []), dtype=float)
    if section_reference.shape == (len(t_raw), 3):
        interpolated_reference = np.empty((n_points, 3), dtype=float)
        mask = np.concatenate(([True], np.diff(t_raw) > 1e-10))
        t_c = t_raw[mask]
        for axis in range(3):
            values = section_reference[:, axis][mask]
            if len(t_c) >= 2:
                interpolated_reference[:, axis] = interp1d(
                    t_c, values, kind='linear', bounds_error=False,
                    fill_value=(values[0], values[-1]))(t_uniform)
            else:
                interpolated_reference[:, axis] = float(values[0])
        for axis, key in enumerate(('section_normal_reference_x',
                                    'section_normal_reference_y',
                                    'section_normal_reference_z')):
            result[key] = interpolated_reference[:, axis].tolist()

    for key in ('section_normal_offset_deg',):
        values = np.asarray(raw_profile.get(key, []), dtype=float)
        if len(values) != len(t_raw):
            continue
        mask = np.concatenate(([True], np.diff(t_raw) > 1e-10))
        t_c, values = t_raw[mask], values[mask]
        if len(t_c) >= 2:
            result[key] = interp1d(
                t_c, values, kind='linear', bounds_error=False,
                fill_value=(values[0], values[-1]))(t_uniform).tolist()
        elif len(values) == 1:
            result[key] = [float(values[0])] * n_points

    # 哪些 key 需要"只用有效值插值"(截面计算的, 0 值代表缺失)
    section_keys = {'area', 'perimeter', 'eq_diameter', 'circularity',
                    'hydraulic_diameter', 'solidity',
                    'raw_area', 'raw_perimeter', 'raw_eq_diameter',
                    'anchor_radius', 'owned_radius'}
    # 哪些 key 直接用所有点(中心线本身的几何, 没有 0 值问题)
    geometry_keys = {'curvature', 'inscribed_radius', 'r_insc_to_r_eq_ratio'}
    # 整数离散 (lumen 分量数), 用最近邻
    integer_keys = {'n_components'}
    # 含 NaN 的几何 (挠率), 单独处理 — NaN 不参与插值
    nanable_keys = {'torsion'}

    # 用 area > 0 作为"截面成功"的掩码
    area_arr = np.asarray(raw_profile['area'])
    success_mask = area_arr > 0
    n_success = int(np.sum(success_mask))

    available_keys = section_keys | geometry_keys | integer_keys | nanable_keys
    available_keys = {k for k in available_keys if k in raw_profile}

    for key in available_keys:
        values = np.asarray(raw_profile[key])
        try:
            if key in section_keys:
                # 只用截面成功的原始点插值
                if n_success >= 2:
                    t_valid = t_raw[success_mask]
                    v_valid = values[success_mask]
                    # 去重 (单调要求)
                    mask = np.concatenate(([True], np.diff(t_valid) > 1e-10))
                    t_c, v_c = t_valid[mask], v_valid[mask]
                    f = interp1d(t_c, v_c, kind='linear',
                                 bounds_error=False,
                                 fill_value=(v_c[0], v_c[-1]))
                    resampled = np.clip(f(t_uniform), 0, None)
                elif n_success == 1:
                    resampled = np.full(n_points, float(values[success_mask][0]))
                else:
                    resampled = np.full(n_points, np.nan)
            elif key in integer_keys:
                # 离散整数: 用最近邻插值 (取整) + 端点延拓
                if n_success >= 2:
                    t_valid = t_raw[success_mask]
                    v_valid = values[success_mask]
                    mask = np.concatenate(([True], np.diff(t_valid) > 1e-10))
                    t_c, v_c = t_valid[mask], v_valid[mask]
                    f = interp1d(t_c, v_c, kind='nearest',
                                 bounds_error=False,
                                 fill_value=(v_c[0], v_c[-1]))
                    resampled = np.clip(f(t_uniform), 0, None)
                elif n_success == 1:
                    resampled = np.full(n_points, float(values[success_mask][0]))
                else:
                    resampled = np.full(n_points, np.nan)
            elif key in nanable_keys:
                # NaN-aware: 跳过 NaN 做线性插值, 不可信处仍保留 NaN
                finite = np.isfinite(values)
                if np.sum(finite) >= 2:
                    t_c, v_c = t_raw[finite], values[finite]
                    mask = np.concatenate(([True], np.diff(t_c) > 1e-10))
                    t_c, v_c = t_c[mask], v_c[mask]
                    f = interp1d(t_c, v_c, kind='linear',
                                 bounds_error=False,
                                 fill_value=np.nan)
                    resampled = f(t_uniform)
                else:
                    resampled = np.full(n_points, np.nan)
            else:
                # 中心线几何特征: 用所有原始点
                mask = np.concatenate(([True], np.diff(t_raw) > 1e-10))
                t_c, v_c = t_raw[mask], values[mask]
                if len(t_c) < 2:
                    resampled = np.zeros(n_points)
                else:
                    f = interp1d(t_c, v_c, kind='linear',
                                 bounds_error=False,
                                 fill_value='extrapolate')
                    resampled = np.clip(f(t_uniform), 0, None)

            if key == 'circularity':
                resampled = np.clip(resampled, 0, 1.5)
            if key == 'solidity':
                resampled = np.clip(resampled, 0, 1.0)
            if key == 'r_insc_to_r_eq_ratio':
                resampled = np.clip(resampled, 0, 1.5)
            result[key] = resampled.tolist()
        except Exception:
            result[key] = ([float('nan')] * n_points
                           if key in nanable_keys else [0.0] * n_points)

    # ---- dA/ds 归一化变化率 (沿重采样均匀点计算, 数值稳定) ----
    try:
        area_uniform = np.asarray(result.get('area', [0.0] * n_points),
                                   dtype=float)
        arc_uniform = np.asarray(result['arc_length_mm'], dtype=float)
        if np.sum(area_uniform > 0) >= 3 and np.all(np.diff(arc_uniform) > 0):
            grad = np.gradient(area_uniform, arc_uniform)
            with np.errstate(divide='ignore', invalid='ignore'):
                dA_ds = grad / np.where(area_uniform > 1e-6,
                                         area_uniform, np.nan)
            # 缺失区段置 NaN, 不污染下游
            dA_ds[area_uniform <= 0] = np.nan
            result['dA_ds_norm'] = dA_ds.tolist()
        else:
            result['dA_ds_norm'] = [float('nan')] * n_points
    except Exception:
        result['dA_ds_norm'] = [float('nan')] * n_points

    return result


def _write_section_normal_audit(parentdir, profiles):
    """Write per-segment selected normal parameters for Web and offline audit."""
    audit_dir = features_dir(parentdir, create=True) / 'section_normals'
    audit_dir.mkdir(parents=True, exist_ok=True)
    for old_file in audit_dir.glob('*.json'):
        old_file.unlink()

    for seg_name, profile in profiles.items():
        if not isinstance(profile, dict) or seg_name.startswith('_'):
            continue
        position = profile.get('position', [])
        n = len(position)
        required = (
            'section_normal_x', 'section_normal_y', 'section_normal_z',
            'section_normal_reference_x', 'section_normal_reference_y',
            'section_normal_reference_z', 'section_normal_offset_deg',
        )
        if n == 0 or any(len(profile.get(key, [])) != n for key in required):
            continue
        area = np.asarray(profile.get('area', [0.0] * n), dtype=float)
        valid = (np.isfinite(area) & (area > 0)).astype(float).tolist()
        payload = {
            'segment': seg_name,
            'normal_search_policy': profile.get('normal_search_policy', {}),
            'position': position,
            'arc_length_mm': profile.get('arc_length_mm', []),
            'section_valid': valid,
            'normal_offset_deg': profile['section_normal_offset_deg'],
            'reference_x': profile['section_normal_reference_x'],
            'reference_y': profile['section_normal_reference_y'],
            'reference_z': profile['section_normal_reference_z'],
            'normal_x': profile['section_normal_x'],
            'normal_y': profile['section_normal_y'],
            'normal_z': profile['section_normal_z'],
        }
        with (audit_dir / f'{seg_name}.json').open('w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2,
                      allow_nan=True)

# ============================================================
# 主入口 (改为读 JSON 驱动)
# ============================================================
def _coords_from_segment_info(seg_info, nodes, path_ids):
    coords = seg_info.get('smoothed_coords') if isinstance(seg_info, dict) else None
    if coords:
        arr = np.asarray(coords, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 2:
            return arr
    return path_to_coords(path_ids, nodes)


def _trim_coords_for_analysis(coords, start_fraction, end_fraction):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return coords
    start_fraction = float(np.clip(start_fraction, 0.0, 1.0))
    end_fraction = float(np.clip(end_fraction, 0.0, 1.0))
    if end_fraction <= start_fraction:
        return coords
    seg_lens = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(arc[-1])
    if total <= 1e-9:
        return coords

    def point_at(distance):
        idx = int(np.searchsorted(arc, distance, side='right') - 1)
        idx = max(0, min(len(coords) - 2, idx))
        local = ((distance - arc[idx]) / (arc[idx + 1] - arc[idx])
                 if arc[idx + 1] > arc[idx] else 0.0)
        return coords[idx] + local * (coords[idx + 1] - coords[idx])

    start_d = start_fraction * total
    end_d = end_fraction * total
    trimmed = [point_at(start_d)]
    for idx, point in enumerate(coords):
        if start_d < arc[idx] < end_d:
            trimmed.append(point)
    trimmed.append(point_at(end_d))
    return np.asarray(trimmed, dtype=float)


def _arc_length_for_coords(coords):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.asarray([0.0], dtype=float)
    return np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(coords, axis=0), axis=1))))


def _resample_coords_by_arc(coords, n_points):
    """Return exactly ``n_points`` centreline locations uniformly by arc length."""
    if coords is None:
        return None
    coords = np.asarray(coords, dtype=float)
    n_points = max(2, int(n_points))
    if len(coords) < 2:
        return coords
    arc = _arc_length_for_coords(coords)
    total = float(arc[-1])
    if total <= 1e-9:
        return coords
    targets = np.linspace(0.0, total, n_points)
    return np.column_stack([
        np.interp(targets, arc, coords[:, axis])
        for axis in range(3)
    ])


def _point_at_arc(coords, arc, distance):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return coords[0] if len(coords) else np.zeros(3)
    distance = float(np.clip(distance, 0.0, float(arc[-1])))
    idx = int(np.searchsorted(arc, distance, side='right') - 1)
    idx = max(0, min(len(coords) - 2, idx))
    local = ((distance - arc[idx]) / (arc[idx + 1] - arc[idx])
             if arc[idx + 1] > arc[idx] else 0.0)
    return coords[idx] + local * (coords[idx + 1] - coords[idx])


def _trim_coords_by_distance(coords, trim_start_mm=0.0, trim_end_mm=0.0):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return coords
    arc = _arc_length_for_coords(coords)
    total = float(arc[-1])
    start_d = float(max(0.0, trim_start_mm))
    end_d = total - float(max(0.0, trim_end_mm))
    if total <= 1e-9 or end_d - start_d < max(1.0, 0.05 * total):
        return coords

    trimmed = [_point_at_arc(coords, arc, start_d)]
    for idx, point in enumerate(coords):
        if start_d < arc[idx] < end_d:
            trimmed.append(point)
    trimmed.append(_point_at_arc(coords, arc, end_d))
    return np.asarray(trimmed, dtype=float)


def _project_point_to_arc(coords, point):
    coords = np.asarray(coords, dtype=float)
    point = np.asarray(point, dtype=float)
    if len(coords) < 2:
        return 0.0
    arc = _arc_length_for_coords(coords)
    best_arc = 0.0
    best_dist = float('inf')
    for idx in range(len(coords) - 1):
        a = coords[idx]
        b = coords[idx + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
        proj = a + t * ab
        dist = float(np.linalg.norm(point - proj))
        if dist < best_dist:
            best_dist = dist
            best_arc = float(arc[idx] + t * (arc[idx + 1] - arc[idx]))
    return best_arc


def _median_positive_radius(coords, mesh):
    try:
        radii = np.asarray(_compute_inscribed_radius_per_point(coords, mesh), dtype=float)
    except Exception:
        return None
    valid = radii[np.isfinite(radii) & (radii > 0.25)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def _path_contains(path, node_id):
    return int(node_id) in {int(nid) for nid in path}


def _path_internal_contains(path, node_id):
    return int(node_id) in {int(nid) for nid in path[1:-1]}


def _choose_receiving_vessel(vessel, node_id, paths):
    preferences = {
        'sv': ['mpv'],
        'smv': ['mpv'],
        'lpv': ['mpv'],
        'rpv': ['mpv'],
        'tips': ['rpv', 'lpv', 'mpv'],
        'lgv': ['mpv'],
        'pgv': ['sv'],
    }
    if vessel not in preferences:
        return None
    for candidate in preferences.get(vessel, []):
        if candidate in paths and _path_contains(paths[candidate], node_id):
            return candidate
    internal = [
        name for name, path in paths.items()
        if name != vessel and _path_internal_contains(path, node_id)
    ]
    if internal:
        return internal[0]
    touching = [
        name for name, path in paths.items()
        if name != vessel and _path_contains(path, node_id)
    ]
    return touching[0] if touching else None


def _build_clinical_junction_plan(seg_data, nodes, coords_by_seg, radii_by_seg,
                                  endpoint_factor=1.25,
                                  side_branch_factor=1.25):
    segments = seg_data.get('segments') or {}
    paths = {
        name: [int(nid) for nid in info.get('path', [])]
        for name, info in segments.items()
        if info and info.get('path')
    }
    plan = {
        name: {
            'trim_start_mm': 0.0,
            'trim_end_mm': 0.0,
            'trim_sources': [],
            'interpolate_intervals_mm': [],
            'interpolate_sources': [],
            'endpoint_junctions': [],
            'side_branch_anchors': [],
        }
        for name in paths
    }

    for vessel, path in paths.items():
        for side, node_id in (('start', path[0]), ('end', path[-1])):
            receiving = _choose_receiving_vessel(vessel, node_id, paths)
            if not receiving:
                continue
            radius = radii_by_seg.get(receiving)
            if radius is None or radius <= 0:
                continue
            # Do not shorten the assigned vessel by a fixed radius.  The
            # shared endpoint can still contain valid sections.  It is only
            # masked later when its area persistently behaves like the known
            # receiving vessel.
            max_distance = float(2.0 * endpoint_factor * radius)
            plan[vessel]['endpoint_junctions'].append({
                'side': side,
                'junction_node_id': int(node_id),
                'receiving_vessel': receiving,
                'receiving_median_radius_mm': float(radius),
                'max_search_distance_mm': max_distance,
            })

    for side_vessel, side_path in paths.items():
        side_radius = radii_by_seg.get(side_vessel)
        if side_radius is None or side_radius <= 0:
            continue
        for node_id in (side_path[0], side_path[-1]):
            node = nodes.get(int(node_id))
            if node is None:
                continue
            point = np.asarray([node['x'], node['y'], node['z']], dtype=float)
            for main_vessel, main_path in paths.items():
                if main_vessel == side_vessel:
                    continue
                if not _path_internal_contains(main_path, node_id):
                    continue
                main_coords = coords_by_seg.get(main_vessel)
                if main_coords is None or len(main_coords) < 2:
                    continue
                total = float(_arc_length_for_coords(main_coords)[-1])
                center_arc = _project_point_to_arc(main_coords, point)
                if center_arc <= 0 or center_arc >= total:
                    continue
                main_radius = radii_by_seg.get(main_vessel)
                plan[main_vessel]['side_branch_anchors'].append({
                    'side_branch': side_vessel,
                    'junction_node_id': int(node_id),
                    'side_branch_median_radius_mm': float(side_radius),
                    'parent_median_radius_mm': (
                        float(main_radius) if main_radius is not None else None),
                    'local_margin_factor': float(side_branch_factor),
                    'arc_center_mm': float(center_arc),
                })
    return plan


def _shift_intervals_after_trim(intervals, trim_start_mm, effective_total):
    shifted = []
    for start, end in intervals or []:
        s = float(start) - float(trim_start_mm)
        e = float(end) - float(trim_start_mm)
        s = max(0.0, min(float(effective_total), s))
        e = max(0.0, min(float(effective_total), e))
        if e > s:
            shifted.append((s, e))
    return shifted


def _shift_side_branch_anchors(anchors, arc_offset_mm, effective_total_mm):
    """Express anatomical side-branch anchors in the analysed path frame."""
    shifted = []
    for anchor in anchors or []:
        center = float(anchor.get('arc_center_mm', -1.0)) - float(arc_offset_mm)
        if 0.0 < center < float(effective_total_mm):
            copied = dict(anchor)
            copied['arc_center_mm'] = center
            shifted.append(copied)
    return shifted


def _interpolate_profile_intervals(profile, intervals_mm):
    n = len(profile.get('position', []))
    if n == 0 or not intervals_mm:
        profile['junction_replaced'] = profile.get('junction_replaced', [0.0] * n)
        profile['n_junction_protected'] = 0
        profile['n_junction_replaced'] = 0
        profile['junction_policy'] = 'clinical_radius_interpolation'
        return profile

    arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
    if len(arc) != n:
        return profile
    interval_mask = np.zeros(n, dtype=bool)
    for start, end in intervals_mm:
        interval_mask |= (arc >= float(start)) & (arc <= float(end))

    section_keys = [
        'area', 'perimeter', 'eq_diameter',
        'raw_area', 'raw_perimeter', 'raw_eq_diameter',
        'anchor_radius', 'owned_radius',
        'circularity', 'hydraulic_diameter', 'solidity',
        'r_insc_to_r_eq_ratio', 'n_components',
    ]
    for key in section_keys:
        if key not in profile:
            continue
        values = np.asarray(profile[key], dtype=float)
        if len(values) != n:
            continue
        valid = np.isfinite(values) & ~interval_mask
        if np.sum(valid) >= 2:
            kind = 'nearest' if key == 'n_components' else 'linear'
            f = interp1d(arc[valid], values[valid], kind=kind,
                         bounds_error=False,
                         fill_value=(values[valid][0], values[valid][-1]))
            values[interval_mask] = f(arc[interval_mask])
            if key != 'n_components':
                values = np.clip(values, 0, None)
            profile[key] = values.tolist()

    marker = [0.0] * n
    for idx in np.where(interval_mask)[0]:
        marker[int(idx)] = 1.0
    profile['junction_replaced'] = marker
    profile['n_junction_protected'] = int(np.sum(interval_mask))
    profile['n_junction_replaced'] = int(np.sum(interval_mask))
    profile['junction_policy'] = 'clinical_radius_interpolation'
    _refresh_dA_ds_norm(profile)
    return profile


def _contiguous_true_runs(mask):
    """Return inclusive index ranges for each contiguous True run."""
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    if len(indices) == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    return [(int(group[0]), int(group[-1]))
            for group in np.split(indices, splits) if len(group)]


def _mask_topology_anchored_side_branch_sections(
        profile, anchors, ratio_threshold=1.8, min_persistence_mm=1.0):
    """Mask only proven high-area contamination around side-branch anchors.

    A side vessel is allowed to affect the profile only close to its known
    insertion point.  The two parent-vessel sides provide independent local
    baselines, so a normal taper or a large but valid remote section is never
    treated as a branch artefact.
    """
    n = len(profile.get('position', []))
    area = np.asarray(profile.get('area', []), dtype=float)
    arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
    if n < 5 or len(area) != n or len(arc) != n or not anchors:
        profile['side_branch_contamination_mask'] = [0.0] * n
        profile['side_branch_contamination_events'] = []
        profile['n_side_branch_contamination_masked'] = 0
        return profile

    ratio_threshold = max(float(ratio_threshold), 1.01)
    min_persistence_mm = max(float(min_persistence_mm), 0.0)
    spacing = float(np.median(np.diff(arc)))
    valid = np.isfinite(area) & (area > 0)
    combined_mask = np.zeros(n, dtype=bool)
    events = []

    for anchor in anchors:
        center = float(anchor.get('arc_center_mm', -1.0))
        side_radius = float(anchor.get('side_branch_median_radius_mm') or 0.0)
        parent_radius = float(anchor.get('parent_median_radius_mm') or 0.0)
        margin_factor = float(anchor.get('local_margin_factor') or 1.25)
        if not (0.0 < center < float(arc[-1])) or side_radius <= 0:
            continue

        # The search band is deliberately local.  Reference windows start
        # outside it, preventing the contaminated overlap from biasing the
        # parent-vessel baseline.
        search_half = max(2.0 * spacing, margin_factor * side_radius,
                          0.75 * parent_radius, 2.0)
        max_half = max(search_half, 2.0 * margin_factor * side_radius,
                       1.25 * parent_radius, 4.0)
        max_half = min(max_half, 0.20 * float(arc[-1]))
        reference_width = max(2.0 * spacing, side_radius, 2.0)
        left_ref = _window_median(
            area, arc, valid, center - max_half - reference_width,
            center - max_half)
        right_ref = _window_median(
            area, arc, valid, center + max_half,
            center + max_half + reference_width)
        if left_ref is None and right_ref is None:
            continue

        local = valid & (np.abs(arc - center) <= max_half)
        expected = np.full(n, np.nan, dtype=float)
        if left_ref is not None and right_ref is not None:
            span = max(2.0 * max_half, np.finfo(float).eps)
            weight = np.clip((arc - (center - max_half)) / span, 0.0, 1.0)
            expected = left_ref + weight * (right_ref - left_ref)
        else:
            expected[:] = left_ref if left_ref is not None else right_ref
        high = local & (area >= ratio_threshold * expected)

        # Select only a high-area run connected to the anatomical insertion.
        # Remote high runs inside the search band remain untouched.
        anchor_tolerance = max(1.5 * spacing, 0.5 * side_radius)
        selected = []
        for start, end in _contiguous_true_runs(high):
            run_start, run_end = float(arc[start]), float(arc[end])
            if run_end - run_start + spacing < min_persistence_mm:
                continue
            if run_start - anchor_tolerance <= center <= run_end + anchor_tolerance:
                selected.append((start, end))
        if not selected:
            continue

        for start, end in selected:
            combined_mask[start:end + 1] = True
            baseline_values = [x for x in (left_ref, right_ref) if x is not None]
            events.append({
                'type': 'side_branch_local_mask',
                'side_branch': anchor.get('side_branch'),
                'junction_node_id': anchor.get('junction_node_id'),
                'arc_center_mm': center,
                'arc_start_mm': float(arc[start]),
                'arc_end_mm': float(arc[end]),
                'area_ratio': float(np.median(area[start:end + 1]) /
                                    np.median(baseline_values)),
            })

    _mask_profile_sections(profile, combined_mask)
    marker = np.asarray(profile.get('junction_replaced', [0.0] * n), dtype=float)
    if len(marker) != n:
        marker = np.zeros(n, dtype=float)
    marker[combined_mask] = 1.0
    profile['junction_replaced'] = marker.tolist()
    profile['side_branch_contamination_mask'] = combined_mask.astype(float).tolist()
    profile['side_branch_contamination_events'] = events
    profile['n_side_branch_contamination_masked'] = int(np.sum(combined_mask))
    profile['n_junction_replaced'] = int(np.sum(marker > 0))
    profile['junction_policy'] = 'topology_anchored_local_mask'
    _refresh_dA_ds_norm(profile)
    return profile


_AREA_JUMP_INTERPOLATION_KEYS = [
    'area', 'perimeter', 'eq_diameter',
    'anchor_radius', 'owned_radius',
    'circularity', 'hydraulic_diameter', 'solidity',
    'r_insc_to_r_eq_ratio', 'n_components',
]


_AREA_JUMP_MASK_KEYS = _AREA_JUMP_INTERPOLATION_KEYS + [
    'raw_area', 'raw_perimeter', 'raw_eq_diameter',
    'inscribed_radius', 'dA_ds_norm',
]


def _interpolate_profile_mask(profile, mask, keys=None):
    """Replace a masked internal interval using adjacent valid sections."""
    n = len(profile.get('position', []))
    arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if n == 0 or len(arc) != n or len(mask) != n or not np.any(mask):
        return profile

    for key in keys or _AREA_JUMP_INTERPOLATION_KEYS:
        if key not in profile:
            continue
        values = np.asarray(profile[key], dtype=float)
        if len(values) != n:
            continue
        valid = np.isfinite(values) & ~mask
        if np.sum(valid) < 2:
            continue
        kind = 'nearest' if key == 'n_components' else 'linear'
        interpolator = interp1d(
            arc[valid], values[valid], kind=kind, bounds_error=False,
            fill_value=(values[valid][0], values[valid][-1]))
        values[mask] = interpolator(arc[mask])
        if key != 'n_components':
            values = np.clip(values, 0, None)
        profile[key] = values.tolist()
    return profile


def _mask_profile_sections(profile, mask):
    """Mask section-derived channels while retaining centerline geometry."""
    n = len(profile.get('position', []))
    mask = np.asarray(mask, dtype=bool)
    if n == 0 or len(mask) != n or not np.any(mask):
        return profile
    for key in _AREA_JUMP_MASK_KEYS:
        if key not in profile:
            continue
        values = list(profile[key])
        if len(values) != n:
            continue
        for idx in np.where(mask)[0]:
            values[int(idx)] = float('nan')
        profile[key] = values
    return profile


def _mask_clinical_endpoint_anchor_sections(profile, mask_start=False,
                                              mask_end=False,
                                              n_endpoint_sections=6,
                                              min_endpoint_distance_mm=3.0):
    """Exclude a fixed-count and physical-distance confluence neighbourhood."""
    n = len(profile.get('position', []))
    mask = np.zeros(n, dtype=bool)
    arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
    if n:
        count = max(1, min(int(n_endpoint_sections), n))
        min_distance = max(0.0, float(min_endpoint_distance_mm))
        use_arc = len(arc) == n and np.all(np.isfinite(arc))
        if mask_start:
            start_count = count
            if use_arc:
                start_count = max(
                    start_count,
                    int(np.searchsorted(arc, min_distance, side='right')))
            mask[:min(n, start_count)] = True
        if mask_end:
            end_count = count
            if use_arc:
                end_count = max(
                    end_count,
                    int(np.sum(arc >= float(arc[-1]) - min_distance)))
            mask[max(0, n - end_count):] = True
    _mask_profile_sections(profile, mask)
    profile['junction_endpoint_excluded'] = mask.astype(float).tolist()
    profile['n_junction_endpoint_excluded'] = int(np.sum(mask))
    profile['junction_endpoint_exclusion_count'] = int(
        max(1, n_endpoint_sections) if (mask_start or mask_end) else 0)
    profile['junction_endpoint_exclusion_min_distance_mm'] = float(
        min_endpoint_distance_mm if (mask_start or mask_end) else 0.0)
    _refresh_dA_ds_norm(profile)
    return profile


def _window_median(values, arc, valid, start_mm, end_mm):
    in_window = valid & (arc >= start_mm) & (arc <= end_mm)
    if np.sum(in_window) < 2:
        return None
    return float(np.median(values[in_window]))


def _adjacent_valid_median(values, valid, boundary, side, n_points=5):
    """Median of the nearest valid samples on the interior side of a boundary."""
    count = max(1, int(n_points))
    valid_indices = np.where(np.asarray(valid, dtype=bool))[0]
    if side == 'start':
        nearby = valid_indices[valid_indices > int(boundary)][:count]
    else:
        nearby = valid_indices[valid_indices < int(boundary)][-count:]
    if len(nearby) < min(3, count):
        return None
    return float(np.median(np.asarray(values, dtype=float)[nearby]))


def _detect_terminal_area_jump(area, arc, valid, side, ratio_threshold,
                               window_mm, min_persistence_mm,
                               max_terminal_extension_mm,
                               reference_points=5):
    """Find a persistent high-area terminal run relative to the segment interior."""
    total = float(arc[-1]) if len(arc) else 0.0
    max_extent = min(float(max_terminal_extension_mm), 0.35 * total)
    if total <= 0 or max_extent < min_persistence_mm:
        return None

    candidates = []
    if side == 'start':
        boundary_indices = np.where(
            (arc >= min_persistence_mm) & (arc <= max_extent))[0]
        for boundary in boundary_indices:
            edge = valid & (arc <= arc[boundary])
            reference = _adjacent_valid_median(
                area, valid, boundary, 'start', reference_points)
            if np.sum(edge) < 2 or reference is None or reference <= 0:
                continue
            edge_values = area[edge]
            edge_median = float(np.median(edge_values))
            ratio = edge_median / reference
            far_reference = _window_median(
                area, arc, valid, total - window_mm, total)
            if (far_reference is not None and far_reference > 0
                    and edge_median / far_reference <= 1.35):
                # A high-low-high pattern is a possible stenosis/thrombus,
                # not a terminal transition into a larger vessel.
                continue
            high_fraction = float(np.mean(
                edge_values >= ratio_threshold * reference))
            if ratio >= ratio_threshold and high_fraction >= 0.60:
                candidates.append((int(boundary), ratio))
    else:
        boundary_indices = np.where(
            (total - arc >= min_persistence_mm)
            & (total - arc <= max_extent))[0]
        for boundary in boundary_indices:
            edge = valid & (arc >= arc[boundary])
            reference = _adjacent_valid_median(
                area, valid, boundary, 'end', reference_points)
            if np.sum(edge) < 2 or reference is None or reference <= 0:
                continue
            edge_values = area[edge]
            edge_median = float(np.median(edge_values))
            ratio = edge_median / reference
            far_reference = _window_median(area, arc, valid, 0.0, window_mm)
            if (far_reference is not None and far_reference > 0
                    and edge_median / far_reference <= 1.35):
                continue
            high_fraction = float(np.mean(
                edge_values >= ratio_threshold * reference))
            if ratio >= ratio_threshold and high_fraction >= 0.60:
                candidates.append((int(boundary), ratio))

    if not candidates:
        return None
    # The first threshold crossing from the junction is the clinical boundary.
    # Continuing the scan can consume normal vessel taper beyond the transition.
    if side == 'start':
        return min(candidates, key=lambda item: item[0])
    return max(candidates, key=lambda item: item[0])


def _detect_symmetric_area_runs(area, arc, valid, ratio_threshold,
                                window_mm, min_persistence_mm,
                                detect_high):
    """Find short runs that differ substantially from similar left/right baselines."""
    n = len(area)
    max_run_mm = max(3.0 * window_mm, min_persistence_mm)
    valid_float = valid.astype(float)
    area_sum = np.concatenate(([0.0], np.cumsum(np.where(valid, area, 0.0))))
    valid_count = np.concatenate(([0.0], np.cumsum(valid_float)))

    def interval_mean(start, end):
        count = valid_count[end] - valid_count[start]
        if count < 2:
            return None
        return float((area_sum[end] - area_sum[start]) / count)

    left_reference = [None] * n
    right_reference = [None] * n
    for idx in range(n):
        left_start = int(np.searchsorted(arc, arc[idx] - window_mm, side='left'))
        left_reference[idx] = interval_mean(left_start, idx)
        right_end = int(np.searchsorted(arc, arc[idx] + window_mm, side='right'))
        right_reference[idx] = interval_mean(idx + 1, right_end)

    candidates = []
    for start in range(1, n - 2):
        for end in range(start + 1, n - 1):
            span = float(arc[end] - arc[start])
            if span < min_persistence_mm:
                continue
            if span > max_run_mm:
                break
            if valid_count[end + 1] - valid_count[start] != end - start + 1:
                continue
            # Open intervals prevent the candidate run from contributing to
            # either reference statistic.
            left = left_reference[start]
            right = right_reference[end]
            if left is None or right is None or min(left, right) <= 0:
                continue
            # A real taper changes one baseline relative to the other. The
            # inserted-vessel model needs comparable parent-vessel baselines.
            if max(left, right) / min(left, right) > 1.35:
                continue
            run_values = area[start:end + 1]
            run_mean = float((area_sum[end + 1] - area_sum[start])
                             / (end - start + 1))
            if detect_high:
                baseline = max(left, right)
                ratio = run_mean / baseline
                fraction = float(np.mean(
                    run_values >= ratio_threshold * baseline))
                edge_consistent = bool(
                    run_values[0] >= ratio_threshold * baseline
                    and run_values[-1] >= ratio_threshold * baseline)
            else:
                baseline = min(left, right)
                ratio = baseline / run_mean
                fraction = float(np.mean(run_values <= baseline / ratio_threshold))
                edge_consistent = bool(
                    run_values[0] <= baseline / ratio_threshold
                    and run_values[-1] <= baseline / ratio_threshold)
            if (ratio >= ratio_threshold and fraction >= 0.80
                    and edge_consistent):
                candidates.append((ratio, start, end))

    # The same plateau yields overlapping candidates. Keep non-overlapping
    # representatives, prioritising the strongest and longest evidence.
    selected = []
    occupied = np.zeros(n, dtype=bool)
    for ratio, start, end in sorted(
            candidates, key=lambda item: item[0] * (item[2] - item[1] + 1),
            reverse=True):
        if np.any(occupied[start:end + 1]):
            continue
        occupied[start:end + 1] = True
        selected.append((ratio, start, end))
    return selected


def _detect_bracketed_internal_high_runs(
        area, arc, valid, ratio_threshold=1.5, max_run_mm=8.0,
        max_baseline_ratio=1.5):
    """Find short high-area plateaus bounded by compatible local sections."""
    area = np.asarray(area, dtype=float)
    arc = np.asarray(arc, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    n = len(area)
    if n < 5 or len(arc) != n or len(valid) != n:
        return []

    candidates = []
    for start in range(1, n - 2):
        if not (valid[start - 1] and valid[start]):
            continue
        for end in range(start + 1, n - 1):
            if not valid[end] or not valid[end + 1]:
                break
            if float(arc[end] - arc[start]) > max_run_mm:
                break
            left, right = float(area[start - 1]), float(area[end + 1])
            baseline = max(left, right)
            if baseline <= 0 or baseline / max(min(left, right), 1e-12) > max_baseline_ratio:
                continue
            values = area[start:end + 1]
            if float(np.median(values)) < ratio_threshold * baseline:
                continue
            if float(np.mean(values >= ratio_threshold * baseline)) < 0.80:
                continue
            candidates.append((
                float(np.median(values) / baseline), start, end))

    selected = []
    occupied = np.zeros(n, dtype=bool)
    for ratio, start, end in sorted(
            candidates, key=lambda item: (
                item[0] * (item[2] - item[1] + 1)), reverse=True):
        if np.any(occupied[start:end + 1]):
            continue
        occupied[start:end + 1] = True
        selected.append((ratio, start, end))
    return selected


def _apply_persistent_area_jump_filter(
        profile, ratio_threshold=1.6, window_mm=6.0,
        min_persistence_mm=4.0, max_terminal_extension_mm=15.0,
        allow_terminal_start=True, allow_terminal_end=True,
        max_terminal_start_extension_mm=None,
        max_terminal_end_extension_mm=None,
        terminal_padding_sections=0, terminal_reference_points=5):
    """
    Detect a persistent section-area expansion caused by a neighbouring vessel.

    A one-point expansion is handled by existing local/rate outlier filters. This
    method targets several consecutive planes that enter a receiving vessel at
    an assigned segment endpoint.  Interior side-branch handling is performed
    separately by ``_mask_topology_anchored_side_branch_sections`` because an
    area jump away from a known anatomical anchor can be a valid vessel part.

    Only sustained *increases* are removed. Sustained decreases are reported as
    ``area_drop_candidate`` because they can reflect true stenosis or thrombus
    and must not be replaced by an image-processing rule.
    """
    if profile is None:
        return profile
    n = len(profile.get('position', []))
    area = np.asarray(profile.get('area', []), dtype=float)
    arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
    if n < 5 or len(area) != n or len(arc) != n or not np.all(np.diff(arc) > 0):
        return profile

    ratio_threshold = max(float(ratio_threshold), 1.01)
    window_mm = max(float(window_mm), 0.1)
    min_persistence_mm = max(float(min_persistence_mm), 0.0)
    max_terminal_extension_mm = max(
        float(max_terminal_extension_mm), min_persistence_mm)
    start_extension_mm = max(
        min_persistence_mm,
        float(max_terminal_start_extension_mm)
        if max_terminal_start_extension_mm is not None
        else max_terminal_extension_mm)
    end_extension_mm = max(
        min_persistence_mm,
        float(max_terminal_end_extension_mm)
        if max_terminal_end_extension_mm is not None
        else max_terminal_extension_mm)
    terminal_padding_sections = max(0, int(terminal_padding_sections))
    terminal_reference_points = max(3, int(terminal_reference_points))
    existing_junction = np.asarray(
        profile.get('junction_replaced', [0.0] * n), dtype=float)
    if len(existing_junction) != n:
        existing_junction = np.zeros(n, dtype=float)
    valid = np.isfinite(area) & (area > 0) & (existing_junction <= 0)

    terminal_mask = np.zeros(n, dtype=bool)
    events = []
    start_event = None
    if allow_terminal_start:
        start_event = _detect_terminal_area_jump(
            area, arc, valid, 'start', ratio_threshold, window_mm,
            min_persistence_mm, start_extension_mm,
            reference_points=terminal_reference_points)
    if start_event is not None:
        boundary, ratio = start_event
        padded_boundary = min(n - 1, boundary + terminal_padding_sections)
        terminal_mask[:padded_boundary + 1] = True
        events.append({
            'type': 'terminal_start', 'arc_start_mm': float(arc[0]),
            'arc_end_mm': float(arc[padded_boundary]),
            'detected_arc_end_mm': float(arc[boundary]),
            'endpoint_padding_sections': terminal_padding_sections,
            'area_ratio': float(ratio),
        })

    end_event = None
    if allow_terminal_end:
        end_event = _detect_terminal_area_jump(
            area, arc, valid, 'end', ratio_threshold, window_mm,
            min_persistence_mm, end_extension_mm,
            reference_points=terminal_reference_points)
    if end_event is not None:
        boundary, ratio = end_event
        padded_boundary = max(0, boundary - terminal_padding_sections)
        terminal_mask[padded_boundary:] = True
        events.append({
            'type': 'terminal_end', 'arc_start_mm': float(arc[padded_boundary]),
            'detected_arc_start_mm': float(arc[boundary]),
            'arc_end_mm': float(arc[-1]), 'area_ratio': float(ratio),
            'endpoint_padding_sections': terminal_padding_sections,
        })

    _mask_profile_sections(profile, terminal_mask)
    _refresh_dA_ds_norm(profile)

    # A short interior plateau that is high on both edges relative to similar
    # left/right parent-vessel baselines is the characteristic footprint of a
    # neighbouring vessel entering the plane.  Preserve the sample locations,
    # but interpolate their measurements so the profile remains continuous.
    # This is deliberately stricter than a generic outlier filter: both
    # baselines must agree and the anomalous run must persist in arc length.
    internal_mask = np.zeros(n, dtype=bool)
    interior_valid = valid & ~terminal_mask
    # Internal contamination requires less contrast than an endpoint entering
    # a receiving vessel.  It still needs bilateral evidence, unlike a normal
    # taper that changes one local baseline only.
    internal_ratio_threshold = min(float(ratio_threshold), 1.25)
    for ratio, start, end in _detect_symmetric_area_runs(
            area, arc, interior_valid, internal_ratio_threshold, window_mm,
            min_persistence_mm, detect_high=True):
        padded_start = max(1, start - 2)
        padded_end = min(n - 2, end + 2)
        internal_mask[padded_start:padded_end + 1] = True
        events.append({
            'type': 'internal_high_plateau_interpolated',
            'arc_start_mm': float(arc[padded_start]),
            'arc_end_mm': float(arc[padded_end]),
            'detected_arc_start_mm': float(arc[start]),
            'detected_arc_end_mm': float(arc[end]),
            'area_ratio': float(ratio),
        })
    for ratio, start, end in _detect_bracketed_internal_high_runs(
            area, arc, interior_valid & ~internal_mask,
            ratio_threshold=1.5, max_run_mm=max(2.0 * window_mm, 8.0),
            max_baseline_ratio=1.5):
        padded_start = max(1, start - 1)
        padded_end = min(n - 2, end + 1)
        internal_mask[padded_start:padded_end + 1] = True
        events.append({
            'type': 'internal_bracketed_high_plateau_interpolated',
            'arc_start_mm': float(arc[padded_start]),
            'arc_end_mm': float(arc[padded_end]),
            'detected_arc_start_mm': float(arc[start]),
            'detected_arc_end_mm': float(arc[end]),
            'area_ratio': float(ratio),
        })
    if np.any(internal_mask):
        _interpolate_profile_mask(profile, internal_mask)
        _refresh_dA_ds_norm(profile)

    profile['area_jump_terminal_mask'] = terminal_mask.astype(float).tolist()
    profile['area_jump_interpolated'] = internal_mask.astype(float).tolist()
    profile['area_drop_candidate'] = [0.0] * n
    profile['n_area_jump_terminal_masked'] = int(np.sum(terminal_mask))
    profile['n_area_jump_interpolated'] = int(np.sum(internal_mask))
    profile['n_area_drop_candidates'] = 0
    profile['area_jump_parameters'] = {
        'ratio_threshold': ratio_threshold,
        'internal_ratio_threshold': internal_ratio_threshold,
        'window_mm': window_mm,
        'min_persistence_mm': min_persistence_mm,
        'max_terminal_extension_mm': max_terminal_extension_mm,
        'max_terminal_start_extension_mm': start_extension_mm,
        'max_terminal_end_extension_mm': end_extension_mm,
        'allow_terminal_start': bool(allow_terminal_start),
        'allow_terminal_end': bool(allow_terminal_end),
        'terminal_padding_sections': terminal_padding_sections,
        'terminal_reference_points': terminal_reference_points,
    }
    profile['area_jump_events'] = events
    profile['area_drop_events'] = []
    return profile


def _branchpoint_arcs_for_path(seg_path, nodes, branchpoint_ids):
    """返回当前段路径上所有分叉点的弧长位置。"""
    if not seg_path or not branchpoint_ids:
        return []
    coords = path_to_coords(seg_path, nodes)
    if len(coords) != len(seg_path):
        return []
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(diffs)))
    return [float(arc[i]) for i, nid in enumerate(seg_path)
            if int(nid) in branchpoint_ids]


def _load_analysis_ranges(parentdir):
    range_path = resolve_feature_path(parentdir, SEGMENT_ASSIGNMENTS_NAME)
    if range_path is None:
        return {}
    try:
        with open(range_path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        ranges = data.get('analysis_ranges', {}) if isinstance(data, dict) else {}
        return ranges if isinstance(ranges, dict) else {}
    except Exception as exc:
        print(f"  Warning: unable to read analysis ranges: {exc}")
        return {}


def _trim_path_for_analysis(seg_path, nodes, range_info):
    original = [int(nid) for nid in seg_path]
    if not isinstance(range_info, dict) or len(original) < 3:
        return original, 0.0, 1.0
    try:
        start = min(1.0, max(0.0, float(range_info.get('start_fraction', 0.0))))
        end = min(1.0, max(0.0, float(range_info.get('end_fraction', 1.0))))
    except Exception:
        return original, 0.0, 1.0
    if end - start < 0.02:
        return original, 0.0, 1.0
    coords = path_to_coords(original, nodes)
    if len(coords) != len(original):
        return original, 0.0, 1.0
    arc = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    total = float(arc[-1])
    if total <= 1e-9:
        return original, 0.0, 1.0
    start_idx = int(np.argmin(np.abs(arc - start * total)))
    end_idx = int(np.argmin(np.abs(arc - end * total)))
    if end_idx - start_idx < 1:
        return original, 0.0, 1.0
    return (
        original[start_idx:end_idx + 1],
        float(arc[start_idx] / total),
        float(arc[end_idx] / total),
    )


def extract_profiles(stl_path, n_points=100, pitch=0.5,
                     curvature_window=7, section_step=3,
                     edge_margin_pct=0.05,
                     edge_margin_mm=8.0,
                     inscribed_factor=1.8,
                     ownership_factor=1.8,
                     junction_policy='min_valid',
                     max_diameter_rate_per_mm=0.5,
                     area_jump_ratio_threshold=1.6,
                     area_jump_window_mm=6.0,
                     area_jump_min_persistence_mm=4.0,
                     area_jump_max_terminal_extension_mm=15.0,
                     area_jump_reference_points=5,
                     endpoint_min_distance_mm=3.0,
                     max_section_samples_per_segment=None,
                     normal_search_policy=None):
    """
    为每个解剖段提取 100 点剖面 (含截面特征)。

    输出:
        <patient_dir>/centerline_pointwise_profiles.json

    参数:
        stl_path:          vessel.stl 路径
        n_points:          重采样点数 (默认 100)
        pitch:             体素化分辨率 mm
        curvature_window:  曲率计算窗口
        section_step:      原始截面采样步长 (每隔 N 个中心线点算一次截面)
        edge_margin_pct:   端点保护比例 (默认 0.05)
        edge_margin_mm:    端点保护绝对距离 mm (默认 8.0)
        inscribed_factor:  截面等效直径相对于内切直径 (2*r) 的最大允许倍数
                           (默认 1.8). 用于过滤穿透到邻近血管的"超大"截面.
        ownership_factor:  中心线锚定清洗半径倍数 (默认 1.8).
                           用于保留当前血管主体并裁剪分叉污染区域.
        junction_policy:   分叉/交叉点保护策略:
                           'min_valid' 用本段可信最小截面替换交叉区;
                           'cap_min' 只封顶异常大截面;
                           'keep' 保留 clean area, 不替换。
        max_diameter_rate_per_mm: 沿管轴允许的等效直径相对变化率 (1/mm),
                                  默认 0.5 = 每 mm 最多 50% 相对变化.
                                  超阈孤立点视为伪影截面 (单点塌陷/膨胀).
    """
    parentdir = os.path.dirname(stl_path)
    if max_section_samples_per_segment is None:
        # The final profile has n_points samples. A modestly denser raw source
        # preserves interpolation and outlier detection without spending time
        # on hundreds of redundant planes in a long vessel.
        max_section_samples_per_segment = max(125, int(n_points) + 25)
    seg_path = resolve_feature_path(parentdir, SEGMENT_ASSIGNMENTS_NAME)
    if seg_path is None:
        print(f"  跳过 (无分段文件): {seg_path}")
        return

    with open(seg_path, 'r', encoding='utf-8') as f:
        seg_data = json.load(f)
    analysis_ranges = _load_analysis_ranges(parentdir)

    nodes, _, _ = load_tree(stl_path)
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, 'geometry'):
            mesh = list(mesh.geometry.values())[0]
        else:
            print("  STL 加载失败")
            return

    profiles = {}
    n_total_masked = 0
    n_total_junction_protected = 0
    n_total_junction_replaced = 0
    n_total_rejected_oversize = 0
    n_total_relaxed_bounds = 0
    n_total_local_outliers = 0
    n_total_rate_outliers = 0
    n_total_implausibly_small = 0
    n_total_area_jump_terminal_masked = 0
    n_total_area_jump_interpolated = 0
    n_total_area_drop_candidates = 0
    branchpoint_ids = {
        int(bp['id']) for bp in seg_data.get('branch_points', [])
        if isinstance(bp, dict) and 'id' in bp
    }
    original_paths = {
        name: [int(nid) for nid in info.get('path', [])]
        for name, info in (seg_data.get('segments') or {}).items()
        if info and info.get('path')
    }
    coords_by_seg = {
        name: _coords_from_segment_info(seg_data['segments'][name], nodes, path)
        for name, path in original_paths.items()
    }
    radii_by_seg = {
        name: _median_positive_radius(coords, mesh)
        for name, coords in coords_by_seg.items()
        if coords is not None and len(coords) >= 2
    }
    clinical_plan = _build_clinical_junction_plan(
        seg_data, nodes, coords_by_seg, radii_by_seg,
        endpoint_factor=1.25,
        side_branch_factor=1.25)
    n_total_clinical_endpoint_trimmed = 0
    n_total_clinical_endpoint_junctions = 0
    n_total_clinical_interpolated = 0

    for seg_name, seg_info in seg_data['segments'].items():
        if seg_info is None:
            profiles[seg_name] = None
            continue

        try:
            # Compute exactly one candidate search for every final profile point.
            original_seg_path_ids = [int(nid) for nid in seg_info['path']]
            range_info = analysis_ranges.get(seg_name)
            seg_path_ids, actual_start, actual_end = _trim_path_for_analysis(
                original_seg_path_ids, nodes, range_info)
            branch_coords = coords_by_seg.get(seg_name)
            if branch_coords is not None and range_info:
                branch_coords = _trim_coords_for_analysis(
                    branch_coords, actual_start, actual_end)
            branch_coords = _resample_coords_by_arc(branch_coords, n_points)
            plan = clinical_plan.get(seg_name, {})
            # A shared endpoint may contain valid assigned-vessel sections.
            # Keep the full path and let the topology-anchored area test make
            # the final decision after section extraction.
            trim_start_mm = 0.0
            trim_end_mm = 0.0
            effective_total = float(_arc_length_for_coords(branch_coords)[-1]) if branch_coords is not None else 0.0
            original_coords = coords_by_seg.get(seg_name)
            original_total = float(_arc_length_for_coords(original_coords)[-1]) if original_coords is not None else 0.0
            side_branch_anchors = _shift_side_branch_anchors(
                plan.get('side_branch_anchors', []),
                actual_start * original_total, effective_total)
            raw_profile = _extract_branch_raw_profile(
                seg_path_ids, nodes, mesh,
                curvature_window=curvature_window,
                section_step=1,
                inscribed_factor=inscribed_factor,
                ownership_factor=ownership_factor,
                max_diameter_rate_per_mm=max_diameter_rate_per_mm,
                branch_coords=branch_coords,
                max_section_samples=n_points,
                normal_search_policy=normal_search_policy)

            if raw_profile is None:
                profiles[seg_name] = None
                continue

            n_total_rejected_oversize += raw_profile.get(
                '_n_rejected_oversize', 0)
            n_total_relaxed_bounds += raw_profile.get(
                '_n_relaxed_bounds', 0)
            n_total_local_outliers += raw_profile.get(
                '_n_local_outliers', 0)
            n_total_rate_outliers += raw_profile.get(
                '_n_rate_outliers', 0)

            # 重采样到 n_points
            resampled = _resample_profile(raw_profile, n_points=n_points)
            if resampled is None:
                profiles[seg_name] = None
                continue
            resampled = _mask_implausibly_small_sections(resampled)
            n_total_implausibly_small += int(
                resampled.get('n_implausibly_small_sections', 0))

            branchpoint_arcs = _branchpoint_arcs_for_path(
                seg_path_ids, nodes, branchpoint_ids)
            terminal_start = (
                seg_path_ids[0] == original_seg_path_ids[0]
                and original_seg_path_ids[0] not in branchpoint_ids)
            terminal_end = (
                seg_path_ids[-1] == original_seg_path_ids[-1]
                and original_seg_path_ids[-1] not in branchpoint_ids)

            # 应用真实末端掩码 + 交叉区最小截面替换/封顶
            endpoint_junctions = plan.get('endpoint_junctions', [])
            start_junction = next(
                (item for item in endpoint_junctions
                 if item.get('side') == 'start'), None)
            end_junction = next(
                (item for item in endpoint_junctions
                 if item.get('side') == 'end'), None)
            resampled = _mask_topology_anchored_side_branch_sections(
                resampled, side_branch_anchors,
                ratio_threshold=area_jump_ratio_threshold,
                min_persistence_mm=area_jump_min_persistence_mm)
            # First mask the six planes at the known clinical endpoint.  When
            # persistent area growth also shows that the branch has entered a
            # receiving vessel, delete that detected run and add six more
            # assigned-vessel planes beyond its boundary.
            resampled = _apply_persistent_area_jump_filter(
                resampled,
                ratio_threshold=area_jump_ratio_threshold,
                window_mm=area_jump_window_mm,
                min_persistence_mm=area_jump_min_persistence_mm,
                max_terminal_extension_mm=area_jump_max_terminal_extension_mm,
                allow_terminal_start=(start_junction is not None and actual_start <= 0.0),
                allow_terminal_end=(end_junction is not None and actual_end >= 1.0),
                max_terminal_start_extension_mm=(
                    min(area_jump_max_terminal_extension_mm,
                        start_junction.get('max_search_distance_mm'))
                    if start_junction is not None else None),
                max_terminal_end_extension_mm=(
                    min(area_jump_max_terminal_extension_mm,
                        end_junction.get('max_search_distance_mm'))
                    if end_junction is not None else None),
                terminal_padding_sections=6,
                terminal_reference_points=area_jump_reference_points)
            # Apply the baseline physical confluence neighbourhood only after
            # the terminal-area detector has seen its unmasked evidence.
            resampled = _mask_clinical_endpoint_anchor_sections(
                resampled,
                mask_start=(start_junction is not None and actual_start <= 0.0),
                mask_end=(end_junction is not None and actual_end >= 1.0),
                min_endpoint_distance_mm=endpoint_min_distance_mm)
            if endpoint_junctions:
                n_total_clinical_endpoint_junctions += len(endpoint_junctions)
            n_total_clinical_interpolated += int(
                resampled.get('n_junction_replaced', 0))
            resampled['clinical_junction_plan'] = {
                'endpoint_trim_start_mm': trim_start_mm,
                'endpoint_trim_end_mm': trim_end_mm,
                'endpoint_junctions': endpoint_junctions,
                'side_branch_anchors': side_branch_anchors,
                'radius_factor': 1.25,
                'method': 'topology_anchored_endpoint_and_side_branch_masking',
            }

            # 透传过滤元信息
            resampled['n_rejected_oversize'] = int(
                raw_profile.get('_n_rejected_oversize', 0))
            resampled['n_relaxed_bounds'] = int(
                raw_profile.get('_n_relaxed_bounds', 0))
            resampled['n_section_success_final'] = int(
                raw_profile.get('_n_final_success', 0))
            resampled['n_local_outliers'] = int(
                raw_profile.get('_n_local_outliers', 0))
            resampled['n_rate_outliers'] = int(
                raw_profile.get('_n_rate_outliers', 0))
            resampled['n_section_success'] = int(
                raw_profile.get('_n_success', 0))
            resampled['section_step_effective'] = int(
                raw_profile.get('_section_step_effective', section_step))
            resampled['normal_search_policy'] = raw_profile.get(
                '_normal_search_policy', _normal_search_policy())
            resampled['normal_search_counts'] = raw_profile.get(
                '_normal_search_counts', {})
            resampled['normal_search_candidate_count'] = int(
                raw_profile.get('_normal_search_candidate_count', 0))
            if range_info:
                resampled['analysis_range'] = {
                    'start_fraction': actual_start,
                    'end_fraction': actual_end,
                    'requested_start_fraction': float(
                        range_info.get('start_fraction', actual_start)),
                    'requested_end_fraction': float(
                        range_info.get('end_fraction', actual_end)),
                }
                resampled['analysis_path'] = [int(nid) for nid in seg_path_ids]

            n_total_masked += resampled.get('n_masked_endpoints', 0)
            n_total_junction_protected += resampled.get(
                'n_junction_protected', 0)
            n_total_junction_replaced += resampled.get(
                'n_junction_replaced', 0)
            n_total_area_jump_terminal_masked += resampled.get(
                'n_area_jump_terminal_masked', 0)
            n_total_area_jump_interpolated += resampled.get(
                'n_area_jump_interpolated', 0)
            n_total_area_drop_candidates += resampled.get(
                'n_area_drop_candidates', 0)
            profiles[seg_name] = resampled

        except Exception as e:
            print(f"    [{seg_name}] 剖面提取失败: {e}")
            profiles[seg_name] = None

    # 元数据
    profiles['_meta'] = {
        'patient_id': seg_data.get('patient_id'),
        'is_post_tips': seg_data.get('is_post_tips'),
        'n_points': n_points,
        'edge_margin_pct': float(edge_margin_pct),
        'edge_margin_mm': float(edge_margin_mm),
        'inscribed_factor': float(inscribed_factor),
        'ownership_factor': float(ownership_factor),
        'junction_policy': junction_policy,
        'max_diameter_rate_per_mm': float(max_diameter_rate_per_mm),
        'area_jump_ratio_threshold': float(area_jump_ratio_threshold),
        'area_jump_window_mm': float(area_jump_window_mm),
        'area_jump_min_persistence_mm': float(area_jump_min_persistence_mm),
        'area_jump_max_terminal_extension_mm': float(
            area_jump_max_terminal_extension_mm),
        'area_jump_reference_points': int(area_jump_reference_points),
        'endpoint_min_distance_mm': float(endpoint_min_distance_mm),
        'max_section_samples_per_segment': int(
            max_section_samples_per_segment),
        'normal_search_policy': _normal_search_policy(normal_search_policy),
        'n_total_masked': int(n_total_masked),
        'n_total_junction_protected': int(n_total_junction_protected),
        'n_total_junction_replaced': int(n_total_junction_replaced),
        'n_total_rejected_oversize': int(n_total_rejected_oversize),
        'n_total_relaxed_bounds': int(n_total_relaxed_bounds),
        'n_total_local_outliers': int(n_total_local_outliers),
        'n_total_rate_outliers': int(n_total_rate_outliers),
        'n_total_implausibly_small_sections': int(n_total_implausibly_small),
        'n_total_area_jump_terminal_masked': int(
            n_total_area_jump_terminal_masked),
        'n_total_area_jump_interpolated': int(
            n_total_area_jump_interpolated),
        'n_total_area_drop_candidates': int(n_total_area_drop_candidates),
        'clinical_junction_method': 'topology_anchored_endpoint_and_side_branch_masking',
        'clinical_radius_factor': 1.25,
        'n_total_clinical_endpoint_trimmed': int(n_total_clinical_endpoint_trimmed),
        'n_total_clinical_endpoint_junctions': int(
            n_total_clinical_endpoint_junctions),
        'n_total_clinical_interpolated': int(n_total_clinical_interpolated),
        'median_centerline_radius_by_segment_mm': {
            str(k): float(v) for k, v in radii_by_seg.items()
            if v is not None
        },
        'analysis_ranges_file': SEGMENT_ASSIGNMENTS_NAME if analysis_ranges else None,
        'analysis_ranges_applied': sorted(analysis_ranges.keys()),
        # 新增逐点通道清单 (便于训练侧统一索引)
        'pointwise_channels': [
            'position', 'arc_length_mm',
            'area', 'perimeter', 'eq_diameter',
            'raw_area', 'raw_perimeter', 'raw_eq_diameter',
            'anchor_radius', 'owned_radius',
            'hydraulic_diameter',        # 4A/P, 非圆截面有效直径
            'circularity',
            'solidity',                  # A / 凸包面积, ∈ (0,1], 1=凸
            'r_insc_to_r_eq_ratio',      # 2r_insc / D_eq, 瓶颈程度
            'n_components',              # lumen 分量数 (1=正常, 2+=被血栓隔断)
            'junction_replaced',         # 1=topology-anchored junction mask
            'junction_endpoint_excluded',
            'side_branch_contamination_mask',
            'section_valid',             # unified export: 1=draw/statistically valid
            'area_jump_terminal_mask',   # 1=persistent high-area terminal section excluded
            'area_jump_interpolated',    # 1=persistent high-area internal section interpolated
            'area_drop_candidate',       # 1=persistent area decrease, diagnostic only
            'implausibly_small_section',
            'curvature',
            'torsion',                   # Frenet 挠率, 中心线 3D 扭转 (NaN 友好)
            'dA_ds_norm',                # (dA/ds)/A, 局部锥度 (NaN 友好)
            'inscribed_radius',
            'section_normal_offset_deg',
            'section_normal_reference_x', 'section_normal_reference_y',
            'section_normal_reference_z',
            'section_normal_x', 'section_normal_y', 'section_normal_z',
        ],
    }

    out_path = str(feature_path(parentdir, POINTWISE_TEMP_NAME, create=True))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False, allow_nan=True)
    _write_section_normal_audit(parentdir, profiles)

    valid_segs = [k for k, v in profiles.items()
                   if v is not None and not k.startswith('_')]
    print(f"  剖面提取完成: {len(valid_segs)} 个段, "
          f"端点掩码 {n_total_masked} 处, "
          f"交叉区保护 {n_total_junction_protected} 处 "
          f"(替换/封顶 {n_total_junction_replaced} 处), "
          f"形状/内切超限剔除 {n_total_rejected_oversize} 处, "
          f"局部异常剔除 {n_total_local_outliers} 处, "
          f"变化率剔除 {n_total_rate_outliers} 处, "
          f"面积跃迁端点排除 {n_total_area_jump_terminal_masked} 处, "
          f"段内插值 {n_total_area_jump_interpolated} 处, "
          f"面积下降候选 {n_total_area_drop_candidates} 段")
    return profiles

def _diagnose_centerline_mesh(branch_path, nodes, mesh):
    """诊断 MPV 中心线点和 mesh 的对齐情况"""
    if len(branch_path) < 3:
        return

    coords = path_to_coords(branch_path, nodes)
    test_indices = [0, len(branch_path)//4, len(branch_path)//2,
                    3*len(branch_path)//4, len(branch_path)-1]

    mb = mesh.bounds
    print(f"  [诊断] mesh: x=[{mb[0][0]:.1f},{mb[1][0]:.1f}], "
          f"y=[{mb[0][1]:.1f},{mb[1][1]:.1f}], "
          f"z=[{mb[0][2]:.1f},{mb[1][2]:.1f}]")

    n_inside = 0
    for idx in test_indices:
        if mesh.contains([coords[idx]])[0]:
            n_inside += 1
    print(f"  [诊断] MPV 测试点 {n_inside}/{len(test_indices)} 在 mesh 内部")

    mid = len(branch_path) // 2
    pt = coords[mid]
    tangent = (coords[min(mid+1, len(coords)-1)]
               - coords[max(mid-1, 0)])
    tangent /= (np.linalg.norm(tangent) + 1e-15)

    try:
        lines = trimesh.intersections.mesh_plane(
            mesh, plane_normal=tangent, plane_origin=pt)
        n_segs = len(lines) if lines is not None else 0
        a, p = _section_one(mesh, pt, tangent)
        print(f"  [诊断] MPV 中点截面: {n_segs}线段, "
              f"面积={a:.2f}mm², 周长={p:.2f}mm")
    except Exception as e:
        print(f"  [诊断] 截面测试异常: {e}")


# ============================================================
# 批量
# ============================================================

def batch_extract_profiles(root_folder, n_points=100, pitch=0.5,
                           section_step=3, stl_name="vessel.stl"):
    print(f"\n{'='*60}")
    print(f"批量剖面提取: {root_folder}")
    print(f"{'='*60}")

    subfolders = sorted(
        d for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d)))

    success, fail = 0, 0
    for folder in subfolders:
        fp = os.path.join(root_folder, folder)
        stl = os.path.join(fp, stl_name)
        if not os.path.exists(stl):
            continue
        seg_json = resolve_feature_path(fp, SEGMENT_ASSIGNMENTS_NAME)
        if seg_json is None:
            print(f"  {folder}: 缺少分段 JSON, 跳过")
            fail += 1
            continue
        try:
            extract_profiles(stl, n_points, pitch, section_step=section_step)
            success += 1
        except Exception as e:
            print(f"  {folder}: 失败 ({e})")
            fail += 1

    print(f"\n完成: {success} 成功, {fail} 失败")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        extract_profiles(sys.argv[1])
    else:
        batch_extract_profiles(r"F:\PCG data\dataset\zhengzhou_vkan_qian47")
