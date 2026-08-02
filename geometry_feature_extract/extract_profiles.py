"""
中心线逐点剖面特征提取（v3 - 读取分段 JSON 驱动）
==================================================
不再做解剖识别, 直接读 centerline_profiles.json 拿到每段路径,
对每段提取逐点剖面 (面积/周长/直径/圆度/曲率/内切半径)。

输出文件: pointwise_profiles.json
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
from smooth_centerline import smooth_internal_anatomical_junctions


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


def _clip_convex_polygon_2d(vertices, normal, limit, tolerance=1e-9):
    """Clip a convex 2-D polygon by ``normal dot point <= limit``."""
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) == 0:
        return vertices
    output = []
    previous = vertices[-1]
    previous_value = float(np.dot(normal, previous) - limit)
    previous_inside = previous_value <= tolerance
    for current in vertices:
        current_value = float(np.dot(normal, current) - limit)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > 1e-15:
                fraction = previous_value / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return np.asarray(output, dtype=float)


def _centerline_voronoi_cell_2d(point, normal, centerline_coords,
                                centerline_index, extent,
                                local_exclusion_mm=5.0,
                                centerline_arc_length=None,
                                competing_centerlines=None,
                                site_radius_mm=0.0):
    """Return the current centerline site's 3-D Voronoi cell on a plane.

    Nearby sites along the curve are ignored because their bisectors would
    turn a sub-millimetre sampling grid into artificial strips. Sites farther
    than ``local_exclusion_mm`` prevent a section from entering another turn
    of the same highly curved vessel.
    """
    from shapely.geometry import Polygon

    coords = np.asarray(centerline_coords, dtype=float)
    index = int(centerline_index)
    if coords.ndim != 2 or coords.shape[1] != 3 or not 0 <= index < len(coords):
        return None
    if centerline_arc_length is None:
        centerline_arc_length = np.concatenate(([0.0], np.cumsum(
            np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    arc = np.asarray(centerline_arc_length, dtype=float)
    if len(arc) != len(coords):
        return None

    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    u, v = _make_orthonormal_basis(normal)
    extent = max(float(extent), 1.0)
    vertices = np.array([
        [-extent, -extent], [extent, -extent],
        [extent, extent], [-extent, extent],
    ], dtype=float)
    site = coords[index]
    local_exclusion_mm = max(0.0, float(local_exclusion_mm))
    def clip_against_site(vertices, other, other_radius_mm=0.0,
                          weighted=False):
        other = np.asarray(other, dtype=float)
        if other.shape != (3,) or not np.all(np.isfinite(other)):
            return vertices
        delta = other - site
        halfplane_normal = np.array([
            float(np.dot(delta, u)), float(np.dot(delta, v))
        ])
        if np.linalg.norm(halfplane_normal) <= 1e-10:
            return vertices
        limit = 0.5 * (
            float(np.dot(other - point, other - point))
            - float(np.dot(site - point, site - point))
        )
        if weighted:
            limit += 0.5 * (
                max(float(site_radius_mm), 0.0) ** 2
                - max(float(other_radius_mm), 0.0) ** 2)
        return _clip_convex_polygon_2d(vertices, halfplane_normal, limit)

    for other_index, other in enumerate(coords):
        if other_index == index:
            continue
        if abs(float(arc[other_index] - arc[index])) <= local_exclusion_mm:
            continue
        vertices = clip_against_site(vertices, other)
        if len(vertices) < 3:
            return None

    # Internal side branches compete with the parent centreline at every
    # section. Junction-local branch samples were removed when the network was
    # built, so the remaining sites need no parent-arc exclusion here.
    for competitor in competing_centerlines or []:
        competitor_coords = (
            competitor.get('centerline_coords')
            if isinstance(competitor, dict) else competitor)
        competitor_coords = np.asarray(competitor_coords, dtype=float)
        if competitor_coords.ndim != 2 or competitor_coords.shape[1] != 3:
            continue
        for other in competitor_coords:
            competitor_radius = (
                float(competitor.get('radius_mm', 0.0))
                if isinstance(competitor, dict) else 0.0)
            vertices = clip_against_site(
                vertices, other, competitor_radius, weighted=True)
            if len(vertices) < 3:
                return None

    cell = Polygon(vertices)
    if not cell.is_valid:
        cell = cell.buffer(0)
    return cell if not cell.is_empty and cell.area > 0 else None


def _clip_section_to_centerline_voronoi(poly, center, point, normal,
                                        centerline_coords=None,
                                        centerline_index=None,
                                        local_exclusion_mm=5.0,
                                        centerline_arc_length=None,
                                        competing_centerlines=None,
                                        site_radius_mm=0.0):
    """Keep only the section area owned by this centerline arc location."""
    if centerline_coords is None or centerline_index is None:
        return poly
    ring = np.asarray(poly.exterior.coords, dtype=float)
    extent = max(1.0, 1.05 * float(np.max(np.linalg.norm(ring, axis=1))))
    cell = _centerline_voronoi_cell_2d(
        point, normal, centerline_coords, centerline_index, extent,
        local_exclusion_mm=local_exclusion_mm,
        centerline_arc_length=centerline_arc_length,
        competing_centerlines=competing_centerlines,
        site_radius_mm=site_radius_mm)
    if cell is None:
        return None
    return _pick_polygon_from_geometry(poly.intersection(cell), center)


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

    if not poly.covers(center):
        return poly, 0.0, 0.0

    anchor_radius = float(poly.boundary.distance(center))
    if anchor_radius <= 1e-6:
        return poly, anchor_radius, 0.0

    if ownership_factor is None or ownership_factor <= 0:
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
                 return_raw=False, return_extras=False,
                 centerline_coords=None, centerline_index=None,
                 centerline_voronoi_exclusion_mm=5.0,
                 centerline_arc_length=None,
                 competing_centerlines=None,
                 centerline_site_radius_mm=0.0):
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
        return_extras:   是否同时返回 (solidity,)

    返回 (按 flag 组合, extras 永远放在末尾, 不影响既有调用方):
        默认                                            (area, peri)
        return_metrics=True                             (area, peri, AR, circ)
        return_ring=True                                (area, peri, ring_2d)
        return_ring=True, return_metrics=True           (area, peri, AR, circ, ring_2d)
        return_raw=True 时, 再追加 (raw_area, raw_peri, anchor_r, owned_r)
        return_extras=True 时, 最后追加 (solidity,)
        失败时各位置填 0/0/999/0/None/0/0
    """
    base_fail = (0.0, 0.0)
    extras_fail = (0.0,)
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
            section_poly = _clip_section_to_centerline_voronoi(
                cand, center, point, normal,
                centerline_coords=centerline_coords,
                centerline_index=centerline_index,
                local_exclusion_mm=centerline_voronoi_exclusion_mm,
                centerline_arc_length=centerline_arc_length,
                competing_centerlines=competing_centerlines,
                site_radius_mm=centerline_site_radius_mm)
            if section_poly is None or section_poly.is_empty:
                continue
            effective_ownership_factor = (
                None if centerline_coords is not None else ownership_factor)
            owned, _, _ = _center_owned_polygon(
                section_poly, center,
                ownership_factor=effective_ownership_factor)
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
            best = (cand, section_poly)
            break

        if best is None:
            return fail

        raw_poly, section_poly = best
        raw_area = float(raw_poly.area)
        raw_peri = float(raw_poly.exterior.length)

        effective_ownership_factor = (
            None if centerline_coords is not None else ownership_factor)
        owned, anchor_radius, owned_radius = _center_owned_polygon(
            section_poly, center,
            ownership_factor=effective_ownership_factor)
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
        raw_ring_2d_list = list(raw_poly.exterior.coords)
        owned_ring_2d_list = list(owned.exterior.coords)
        if return_metrics:
            aspect_ratio = _polygon_aspect_ratio(owned_ring_2d_list)
            if peri > 1e-6:
                circularity = float(min(1.5, 4.0 * np.pi * area / (peri * peri)))
            else:
                circularity = 0.0

        if return_extras:
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
            extras_tuple = (solidity,)

        if return_ring and return_metrics:
            base = (area, peri, aspect_ratio, circularity, owned_ring_2d_list)
        elif return_metrics:
            base = (area, peri, aspect_ratio, circularity)
        elif return_ring:
            base = (area, peri, owned_ring_2d_list)
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
    'normal_tangent_smoothing_mm': 4.0,
}

def _normal_search_policy(policy=None):
    settings = dict(_DEFAULT_NORMAL_SEARCH_POLICY)
    if policy:
        settings.update(policy)
    return settings


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
                                branch_coords=None,
                                max_section_samples=None,
                                normal_search_policy=None,
                                centerline_voronoi_exclusion_mm=5.0,
                                network_voronoi_centerlines=None):
    """
    沿一段中心线提取逐点剖面.

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
    normal_references = tangents.copy()

    # 内切半径仅作为逐点几何特征，不再参与截面删除。
    inscribed_radius = _compute_inscribed_radius_per_point(coords, mesh)

    area = np.zeros(M)           # clean/owned area, downstream default
    perimeter = np.zeros(M)      # clean/owned perimeter
    raw_area = np.zeros(M)       # original STL section before owned clipping
    raw_perimeter = np.zeros(M)
    anchor_radius = np.zeros(M)
    owned_radius = np.zeros(M)
    solidity = np.zeros(M)            # (新) area / convex_hull_area
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
    n_failed = 0
    normal_search_counts = {'deterministic': len(indices)}
    normal_search_candidates = len(indices)
    for idx in indices:
        a, p, ar, circ, raw_a, raw_p, anchor_r, owned_r, sol = _section_one(
            mesh, coords[idx], tangents[idx],
            max_eq_diameter=None,
            min_eq_diameter=None,
            ownership_factor=None,
            return_metrics=True,
            return_raw=True,
            return_extras=True,
            centerline_coords=coords,
            centerline_index=idx,
            centerline_voronoi_exclusion_mm=centerline_voronoi_exclusion_mm,
            centerline_arc_length=arc_length,
            competing_centerlines=network_voronoi_centerlines,
            centerline_site_radius_mm=float(inscribed_radius[idx]))
        area[idx] = a
        perimeter[idx] = p
        raw_area[idx] = raw_a
        raw_perimeter[idx] = raw_p
        anchor_radius[idx] = anchor_r
        owned_radius[idx] = owned_r
        solidity[idx] = sol
        section_normal[idx] = tangents[idx]
        section_normal_offset_deg[idx] = 0.0
        if a > 0:
            n_success += 1
        else:
            n_failed += 1

    sampled_idx = np.array(indices, dtype=int)
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
        'r_insc_to_r_eq_ratio': r_insc_to_r_eq_ratio,
        'curvature': curvature,
        'torsion': torsion,
        'inscribed_radius': inscribed_radius,
        '_n_sampled': len(indices),
        '_section_step_effective': int(effective_section_step),
        '_normal_search_policy': search_policy,
        '_normal_search_counts': normal_search_counts,
        '_normal_search_candidate_count': int(normal_search_candidates),
        '_section_assignment_method': (
            'centerline_network_voronoi'
            if network_voronoi_centerlines else 'centerline_voronoi'),
        '_centerline_voronoi_exclusion_mm': float(
            centerline_voronoi_exclusion_mm),
        '_n_success': n_success,
        '_n_final_success': n_final_success,
        '_n_section_failures': int(n_failed),
        '_centerline_chord_mm': chord,
        '_centerline_arc_chord_tortuosity': float(arc_chord),
        '_centerline_mean_curvature': float(np.mean(finite_curv)) if len(finite_curv) else 0.0,
        '_centerline_max_curvature': float(np.max(finite_curv)) if len(finite_curv) else 0.0,
    }


def _contiguous_index_runs(mask):
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    if len(indices) == 0:
        return []
    split_at = np.where(np.diff(indices) > 1)[0] + 1
    return [group for group in np.split(indices, split_at) if len(group)]


def _resample_valid_section_runs(t_raw, values, valid, t_target,
                                 kind='linear'):
    """Resample only inside contiguous successful runs, never across gaps."""
    values = np.asarray(values, dtype=float)
    output = np.zeros(len(t_target), dtype=float)
    if len(values) != len(t_raw):
        return output
    if len(t_target) == len(t_raw) and np.allclose(
            t_target, t_raw, rtol=0.0, atol=1e-12):
        output[:] = values
        output[~np.asarray(valid, dtype=bool)] = 0.0
        return output
    for run in _contiguous_index_runs(valid):
        if len(run) == 1:
            nearest = int(np.argmin(np.abs(t_target - t_raw[run[0]])))
            output[nearest] = float(values[run[0]])
            continue
        target_mask = (
            (t_target >= t_raw[run[0]]) & (t_target <= t_raw[run[-1]])
        )
        if not np.any(target_mask):
            continue
        output[target_mask] = interp1d(
            t_raw[run], values[run], kind=kind,
            bounds_error=False, fill_value=0.0)(t_target[target_mask])
    return np.clip(output, 0.0, None)


def _normalized_area_gradient(area, arc):
    """Compute (dA/ds)/A independently inside each valid section run."""
    area = np.asarray(area, dtype=float)
    arc = np.asarray(arc, dtype=float)
    output = np.zeros(len(area), dtype=float)
    if (len(area) != len(arc) or len(area) < 2
            or not np.all(np.diff(arc) > 0)):
        return output
    valid = np.isfinite(area) & (area > 0) & np.isfinite(arc)
    for run in _contiguous_index_runs(valid):
        if len(run) < 2:
            continue
        grad = np.gradient(area[run], arc[run])
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = grad / area[run]
        normalized[~np.isfinite(normalized)] = 0.0
        output[run] = normalized
    return output


def _refresh_dA_ds_norm(profile):
    """根据当前 area 重新计算归一化面积变化率。"""
    try:
        area = np.asarray(profile.get('area', []), dtype=float)
        arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
        if len(area) != len(arc) or len(area) < 2:
            return
        profile['dA_ds_norm'] = _normalized_area_gradient(area, arc).tolist()
    except Exception:
        return


def _resample_profile(raw_profile, n_points=200):
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

    # Section-derived channels retain explicit zero gaps. They are interpolated
    # only within contiguous successful runs when the output grid changes.
    section_keys = {'area', 'perimeter', 'eq_diameter', 'circularity',
                    'hydraulic_diameter', 'solidity',
                    'raw_area', 'raw_perimeter', 'raw_eq_diameter',
                    'anchor_radius', 'owned_radius',
                    'r_insc_to_r_eq_ratio'}
    # 哪些 key 直接用所有点(中心线本身的几何, 没有 0 值问题)
    geometry_keys = {'curvature', 'inscribed_radius'}
    # 整数离散 (lumen 分量数), 用最近邻
    # 含 NaN 的几何 (挠率), 单独处理 — NaN 不参与插值
    nanable_keys = {'torsion'}

    # 用 area > 0 作为"截面成功"的掩码
    area_arr = np.asarray(raw_profile['area'])
    success_mask = area_arr > 0

    available_keys = section_keys | geometry_keys | nanable_keys
    available_keys = {k for k in available_keys if k in raw_profile}

    for key in available_keys:
        values = np.asarray(raw_profile[key])
        try:
            if key in section_keys:
                resampled = _resample_valid_section_runs(
                    t_raw, values, success_mask, t_uniform, kind='linear')
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

    # ---- dA/ds is computed independently inside each successful run ----
    try:
        area_uniform = np.asarray(result.get('area', [0.0] * n_points),
                                   dtype=float)
        arc_uniform = np.asarray(result['arc_length_mm'], dtype=float)
        result['dA_ds_norm'] = _normalized_area_gradient(
            area_uniform, arc_uniform).tolist()
    except Exception:
        result['dA_ds_norm'] = [0.0] * n_points

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
            'section_assignment_method': profile.get(
                'section_assignment_method', 'centerline_voronoi'),
            'centerline_voronoi_exclusion_mm': profile.get(
                'centerline_voronoi_exclusion_mm', 5.0),
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
            'endpoint_junctions': [],
            'side_branch_anchors': [],
        }
        for name in paths
    }

    for vessel, path in paths.items():
        for side, node_id in (('start', path[0]), ('end', path[-1])):
            connected = [
                name for name, candidate_path in paths.items()
                if name != vessel and _path_contains(candidate_path, node_id)
            ]
            if not connected:
                continue
            receiving = _choose_receiving_vessel(vessel, node_id, paths)
            internal_receivers = [
                name for name in connected
                if _path_internal_contains(paths[name], node_id)
            ]
            radius_sources = [vessel, *connected]
            candidate_radii = [
                radii_by_seg.get(name) for name in radius_sources
                if radii_by_seg.get(name) is not None
                and radii_by_seg.get(name) > 0
            ]
            radius = max(candidate_radii) if candidate_radii else None
            if radius is None or radius <= 0:
                continue
            plan[vessel]['endpoint_junctions'].append({
                'side': side,
                'junction_node_id': int(node_id),
                'junction_type': (
                    'side_branch_endpoint'
                    if internal_receivers else 'shared_endpoint'),
                'receiving_vessel': receiving,
                'connected_vessels': connected,
                'receiving_median_radius_mm': float(radius),
            })

    for side_vessel, side_path in paths.items():
        side_radius = radii_by_seg.get(side_vessel)
        if side_radius is None or side_radius <= 0:
            continue
        for side, node_id in (('start', side_path[0]), ('end', side_path[-1])):
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
                    'side_branch_junction_side': side,
                    'junction_node_id': int(node_id),
                    'side_branch_median_radius_mm': float(side_radius),
                    'parent_median_radius_mm': (
                        float(main_radius) if main_radius is not None else None),
                    'local_margin_factor': float(side_branch_factor),
                    'arc_center_mm': float(center_arc),
                })
    return plan


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


def _build_network_voronoi_centerlines(
        anchors, coords_by_seg, radii_by_seg, n_points,
        junction_exclusion_mm=5.0):
    """Return unique side-branch centrelines competing with a parent vessel."""
    competitors = []
    seen = set()
    for anchor in anchors or []:
        segment = str(anchor.get('side_branch') or '')
        if not segment or segment in seen:
            continue
        coords = coords_by_seg.get(segment)
        if coords is None or len(coords) < 2:
            continue
        sampled = _resample_coords_by_arc(coords, n_points)
        if sampled is None or len(sampled) < 2:
            continue
        arc = _arc_length_for_coords(sampled)
        exclusion = max(0.0, float(junction_exclusion_mm))
        if anchor.get('side_branch_junction_side') == 'end':
            keep = (float(arc[-1]) - arc) > exclusion
        else:
            keep = arc > exclusion
        sampled = sampled[keep]
        if len(sampled) < 1:
            continue
        competitors.append({
            'segment': segment,
            'junction_node_id': int(anchor.get('junction_node_id')),
            'junction_exclusion_mm': exclusion,
            'radius_mm': float(radii_by_seg.get(segment) or 0.0),
            'centerline_coords': np.asarray(sampled, dtype=float),
        })
        seen.add(segment)
    return competitors


_SECTION_MASK_KEYS = [
    'area', 'perimeter', 'eq_diameter',
    'anchor_radius', 'owned_radius',
    'circularity', 'hydraulic_diameter', 'solidity',
    'r_insc_to_r_eq_ratio',
    'raw_area', 'raw_perimeter', 'raw_eq_diameter',
    'inscribed_radius', 'dA_ds_norm',
]


def _mask_profile_sections(profile, mask):
    """Set section-derived channels to zero without interpolation."""
    n = len(profile.get('position', []))
    mask = np.asarray(mask, dtype=bool)
    if n == 0 or len(mask) != n or not np.any(mask):
        return profile
    for key in _SECTION_MASK_KEYS:
        if key not in profile:
            continue
        values = list(profile[key])
        if len(values) != n:
            continue
        for idx in np.where(mask)[0]:
            values[int(idx)] = 0.0
        profile[key] = values
    return profile


def _window_median(values, arc, valid, start_mm, end_mm):
    in_window = valid & (arc >= start_mm) & (arc <= end_mm)
    if np.sum(in_window) < 2:
        return None
    return float(np.median(values[in_window]))


def _adjacent_valid_median(values, valid, boundary, side, n_points=5):
    """Median of up to ``n_points`` interior samples nearest a boundary."""
    count = max(1, int(n_points))
    valid_indices = np.where(np.asarray(valid, dtype=bool))[0]
    if side == 'start':
        nearby = valid_indices[valid_indices > int(boundary)][:count]
    else:
        nearby = valid_indices[valid_indices < int(boundary)][-count:]
    if len(nearby) == 0:
        return None
    return float(np.median(np.asarray(values, dtype=float)[nearby]))


def _terminal_valid_median(values, valid, boundary, side, n_points=5):
    """Median of available samples immediately on the junction side."""
    count = max(1, int(n_points))
    valid_indices = np.where(np.asarray(valid, dtype=bool))[0]
    if side == 'start':
        nearby = valid_indices[valid_indices <= int(boundary)][-count:]
    else:
        nearby = valid_indices[valid_indices >= int(boundary)][:count]
    if len(nearby) == 0:
        return None
    return float(np.median(np.asarray(values, dtype=float)[nearby]))


def _scan_local_area_transitions(area, valid, ratio_threshold,
                                 reference_points=5):
    """Scan both directions for local area steps at two adjacent scales."""
    values = np.asarray(area, dtype=float)
    valid_indices = np.where(np.asarray(valid, dtype=bool))[0]
    window = max(1, int(reference_points))
    window_sizes = sorted({window, 2 * window})
    candidates = []

    for split in range(len(valid_indices) - 1):
        measurements = []
        for scale in window_sizes:
            left_indices = valid_indices[
                max(0, split - scale + 1):split + 1]
            right_indices = valid_indices[split + 1:split + 1 + scale]
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            left_level = float(np.median(values[left_indices]))
            right_level = float(np.median(values[right_indices]))
            if left_level <= 0 or right_level <= 0:
                continue

            if left_level >= right_level:
                direction = 'down'
                ratio = left_level / right_level
            else:
                direction = 'up'
                ratio = right_level / left_level
            measurements.append({
                'split_order': int(split),
                'left_index': int(left_indices[-1]),
                'right_index': int(right_indices[0]),
                'left_level': left_level,
                'right_level': right_level,
                'direction': direction,
                'ratio': float(ratio),
                'window_points': int(scale),
            })
        if not measurements:
            continue
        evidence = max(measurements, key=lambda item: item['ratio'])
        if evidence['ratio'] < ratio_threshold:
            continue
        base = next(
            item for item in measurements
            if item['window_points'] == window)
        evidence = dict(evidence)
        evidence['localization_ratio'] = (
            float(base['ratio'])
            if base['direction'] == evidence['direction'] else 1.0)
        candidates.append(evidence)

    if not candidates:
        return []

    # A single physical step is usually detected at several neighbouring
    # splits because both medians use multiple samples. Collapse each run to
    # its strongest representative before classifying rises and falls.
    groups = [[candidates[0]]]
    for candidate in candidates[1:]:
        previous = groups[-1][-1]
        if (candidate['direction'] == previous['direction']
                and candidate['split_order'] == previous['split_order'] + 1):
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    transitions = []
    for group in groups:
        direction = group[0]['direction']
        evidence_ratio = max(item['ratio'] for item in group)
        if direction == 'down':
            selected = max(
                group,
                key=lambda item: (
                    item['localization_ratio'], item['left_index']))
        else:
            selected = max(
                group,
                key=lambda item: (
                    item['localization_ratio'], -item['right_index']))
        selected = dict(selected)
        selected['evidence_ratio'] = float(evidence_ratio)
        transitions.append(selected)
    return transitions


def _paired_area_transition_indices(
        transitions, ratio_threshold, max_span_points=None):
    """Pair local bumps and valleys whose area returns to its prior level."""
    max_span = (
        None if max_span_points is None
        else max(1, int(max_span_points)))
    pair_candidates = []
    for index in range(len(transitions) - 1):
        first = transitions[index]
        second = transitions[index + 1]
        if first['direction'] == second['direction']:
            continue
        span_points = int(
            second['right_index'] - first['left_index'])
        if max_span is not None and span_points > max_span:
            continue
        before = float(first['left_level'])
        after = float(second['right_level'])
        if before <= 0 or after <= 0:
            continue
        recovery_ratio = max(before, after) / min(before, after)
        if recovery_ratio >= ratio_threshold:
            continue
        pair_candidates.append((
            span_points,
            float(recovery_ratio), index, index + 1))

    # Prefer the shortest closed excursion. This leaves a true terminal step
    # unpaired when a side-branch bump or thrombus valley occurs farther in.
    paired = set()
    pairs = []
    for _, recovery_ratio, first_index, second_index in sorted(pair_candidates):
        if first_index in paired or second_index in paired:
            continue
        paired.update((first_index, second_index))
        pairs.append((first_index, second_index, recovery_ratio))
    return paired, pairs


def _select_terminal_transition(
        transitions, eligible_indices, ratio_threshold, side, direction,
        local_span_points):
    """Choose the endpoint-side group in a same-direction transition chain."""
    ordered = sorted(eligible_indices)
    if side == 'end':
        ordered.reverse()
    strong_position = next((
        position for position, index in enumerate(ordered)
        if transitions[index]['direction'] == direction
        and transitions[index]['evidence_ratio'] >= ratio_threshold
    ), None)
    if strong_position is None:
        return None

    chain_start = strong_position
    while chain_start > 0:
        previous = transitions[ordered[chain_start - 1]]
        if previous['direction'] != direction:
            break
        chain_start -= 1

    chain = ordered[chain_start:strong_position + 1]
    selected_position = 0
    while selected_position + 1 < len(chain):
        current = transitions[chain[selected_position]]
        following = transitions[chain[selected_position + 1]]
        current_index = (
            current['left_index'] if side == 'start'
            else current['right_index'])
        following_index = (
            following['left_index'] if side == 'start'
            else following['right_index'])
        if abs(following_index - current_index) > local_span_points:
            break
        selected_position += 1

    selected = dict(transitions[chain[selected_position]])
    selected['terminal_evidence_ratio'] = float(max(
        transitions[index]['evidence_ratio'] for index in chain))
    return selected


def _detect_terminal_area_jumps(
        area, arc, valid, ratio_threshold, reference_points=5,
        allow_terminal_start=True, allow_terminal_end=True):
    """Find endpoint-connected expansion while ignoring closed bumps/valleys."""
    total = float(arc[-1]) if len(arc) else 0.0
    if total <= 0:
        return {'start': None, 'end': None}, [], []

    context_threshold = float(np.sqrt(ratio_threshold))
    transitions = _scan_local_area_transitions(
        area, valid, context_threshold, reference_points=reference_points)
    if not transitions:
        return {'start': None, 'end': None}, [], []

    pairing_span_points = 8 * max(1, int(reference_points))
    paired, pairs = _paired_area_transition_indices(
        transitions, ratio_threshold,
        max_span_points=pairing_span_points)
    strong_indices = {
        index for index, transition in enumerate(transitions)
        if transition['evidence_ratio'] >= ratio_threshold
    }
    unpaired_indices = set(range(len(transitions))) - paired
    terminal_indices = strong_indices - paired

    # A vessel such as MPV may genuinely have enlarged junction sections at
    # both ends. Area alone cannot distinguish those from one broad central
    # low region, so anatomical eligibility of both endpoints takes priority.
    if (allow_terminal_start and allow_terminal_end and len(transitions) >= 2
            and transitions[0]['direction'] == 'down'
            and transitions[0]['evidence_ratio'] >= ratio_threshold
            and transitions[-1]['direction'] == 'up'
            and transitions[-1]['evidence_ratio'] >= ratio_threshold):
        terminal_indices.update((0, len(transitions) - 1))

    detected = {'start': None, 'end': None}

    if allow_terminal_start:
        selected = _select_terminal_transition(
            transitions, unpaired_indices, ratio_threshold,
            side='start', direction='down',
            local_span_points=pairing_span_points)
        if selected is not None:
            detected['start'] = (
                int(selected['left_index']),
                float(selected['terminal_evidence_ratio']))

    if allow_terminal_end:
        selected = _select_terminal_transition(
            transitions, unpaired_indices, ratio_threshold,
            side='end', direction='up',
            local_span_points=pairing_span_points)
        if selected is not None:
            detected['end'] = (
                int(selected['right_index']),
                float(selected['terminal_evidence_ratio']))

    return detected, transitions, pairs


def _mask_endpoint_junction_sections(
        profile, ratio_threshold=1.6,
        allow_terminal_start=True, allow_terminal_end=True,
        terminal_padding_sections=6, terminal_reference_points=5,
        protected_side_branch_arcs=None,
        side_branch_protection_mm=5.0):
    """Zero each junction endpoint using the unowned STL section area."""
    if profile is None:
        return profile
    n = len(profile.get('position', []))
    raw_area = np.asarray(profile.get('raw_area', []), dtype=float)
    arc = np.asarray(profile.get('arc_length_mm', []), dtype=float)
    if (n < 5 or len(raw_area) != n or len(arc) != n
            or not np.all(np.diff(arc) > 0)):
        return profile

    ratio_threshold = max(float(ratio_threshold), 1.01)
    terminal_padding_sections = max(0, int(terminal_padding_sections))
    terminal_reference_points = max(1, int(terminal_reference_points))
    protected_arcs = sorted(
        float(value) for value in (protected_side_branch_arcs or [])
        if np.isfinite(value) and float(arc[0]) < float(value) < float(arc[-1]))
    side_branch_protection_mm = max(0.0, float(side_branch_protection_mm))
    existing_junction = np.asarray(
        profile.get('junction_replaced', [0.0] * n), dtype=float)
    if len(existing_junction) != n:
        existing_junction = np.zeros(n, dtype=float)
    valid = (
        np.isfinite(raw_area) & (raw_area > 0) & (existing_junction <= 0))

    terminal_mask = np.zeros(n, dtype=bool)
    events = []
    detected, transitions, transition_pairs = _detect_terminal_area_jumps(
        raw_area, arc, valid, ratio_threshold,
        reference_points=terminal_reference_points,
        allow_terminal_start=allow_terminal_start,
        allow_terminal_end=allow_terminal_end)
    start_event = detected['start']
    if start_event is not None:
        boundary, ratio = start_event
        padded_boundary = min(n - 1, boundary + terminal_padding_sections)
        original_padded_boundary = padded_boundary
        protected_arc = None
        if protected_arcs:
            protected_arc = protected_arcs[0]
            maximum_mask_arc = protected_arc - side_branch_protection_mm
            topology_limit = int(np.searchsorted(
                arc, maximum_mask_arc, side='right') - 1)
            padded_boundary = min(padded_boundary, max(0, topology_limit))
        terminal_mask[:padded_boundary + 1] = True
        events.append({
            'type': 'endpoint_start_interval_zeroed',
            'critical_index': int(boundary),
            'masked_start_index': 0,
            'masked_end_index': int(padded_boundary),
            'arc_start_mm': float(arc[0]),
            'arc_end_mm': float(arc[padded_boundary]),
            'detected_arc_end_mm': float(arc[boundary]),
            'endpoint_padding_sections': terminal_padding_sections,
            'area_ratio': float(ratio),
            'side_branch_topology_clamped': bool(
                padded_boundary != original_padded_boundary),
            'protected_side_branch_arc_mm': protected_arc,
        })

    end_event = detected['end']
    if end_event is not None:
        boundary, ratio = end_event
        padded_boundary = max(0, boundary - terminal_padding_sections)
        original_padded_boundary = padded_boundary
        protected_arc = None
        if protected_arcs:
            protected_arc = protected_arcs[-1]
            minimum_mask_arc = protected_arc + side_branch_protection_mm
            topology_limit = int(np.searchsorted(
                arc, minimum_mask_arc, side='left'))
            padded_boundary = max(
                padded_boundary, min(n - 1, topology_limit))
        terminal_mask[padded_boundary:] = True
        events.append({
            'type': 'endpoint_end_interval_zeroed',
            'critical_index': int(boundary),
            'masked_start_index': int(padded_boundary),
            'masked_end_index': int(n - 1),
            'arc_start_mm': float(arc[padded_boundary]),
            'detected_arc_start_mm': float(arc[boundary]),
            'arc_end_mm': float(arc[-1]), 'area_ratio': float(ratio),
            'endpoint_padding_sections': terminal_padding_sections,
            'side_branch_topology_clamped': bool(
                padded_boundary != original_padded_boundary),
            'protected_side_branch_arc_mm': protected_arc,
        })

    _mask_profile_sections(profile, terminal_mask)
    _refresh_dA_ds_norm(profile)

    profile['area_jump_terminal_mask'] = terminal_mask.astype(float).tolist()
    profile['endpoint_junction_mask'] = terminal_mask.astype(float).tolist()
    profile['area_jump_interpolated'] = [0.0] * n
    profile['area_drop_candidate'] = [0.0] * n
    n_terminal_flagged = int(np.sum(terminal_mask))
    profile['n_area_jump_terminal_flagged'] = n_terminal_flagged
    profile['n_area_jump_terminal_masked'] = n_terminal_flagged
    profile['n_endpoint_junction_zeroed'] = n_terminal_flagged
    profile['area_jump_terminal_values_masked'] = True
    profile['n_area_jump_interpolated'] = 0
    profile['n_area_drop_candidates'] = 0
    profile['area_jump_parameters'] = {
        'area_channel': 'raw_area',
        'ratio_threshold': ratio_threshold,
        'allow_terminal_start': bool(allow_terminal_start),
        'allow_terminal_end': bool(allow_terminal_end),
        'terminal_padding_sections': terminal_padding_sections,
        'terminal_reference_points': terminal_reference_points,
        'transition_scan_method': 'bidirectional_local_area_ratio',
        'transition_pairing_method': 'closed_bump_or_valley',
        'transition_context_ratio_threshold': float(np.sqrt(ratio_threshold)),
        'transition_pairing_max_span_points': int(
            8 * terminal_reference_points),
        'transition_window_scales_points': sorted({
            terminal_reference_points, 2 * terminal_reference_points}),
        'n_strong_local_transitions': int(sum(
            transition['evidence_ratio'] >= ratio_threshold
            for transition in transitions)),
        'n_local_transitions': int(len(transitions)),
        'n_paired_local_transitions': int(2 * len(transition_pairs)),
        'protected_side_branch_arcs_mm': protected_arcs,
        'side_branch_protection_mm': side_branch_protection_mm,
    }
    profile['area_jump_events'] = events
    profile['area_drop_events'] = []
    return profile


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


def extract_profiles(stl_path, n_points=200, pitch=0.5,
                     curvature_window=7, section_step=3,
                     area_jump_ratio_threshold=1.6,
                     area_jump_reference_points=5,
                     max_section_samples_per_segment=None,
                     normal_search_policy=None,
                     centerline_voronoi_exclusion_mm=5.0,
                     smooth_segment_junctions=True,
                     junction_kink_angle_threshold_deg=15.0,
                     junction_kink_excess_threshold_deg=10.0,
                     junction_smoothing_half_window_mm=6.0,
                     junction_tangent_span_mm=2.0):
    """
    为每个解剖段提取 200 点剖面 (含截面特征)。

    输出:
        <patient_dir>/pointwise_profiles.json

    参数:
        stl_path:          vessel.stl 路径
        n_points:          重采样点数 (默认 200)
        pitch:             体素化分辨率 mm
        curvature_window:  曲率计算窗口
        section_step:      原始截面采样步长 (每隔 N 个中心线点算一次截面)

        只有共享端点或侧枝自身端点可按面积比例置 0。主干内部侧枝
        使用血管网络 Voronoi 裁剪，不再删除交点附近的主干截面。
    """
    parentdir = os.path.dirname(stl_path)
    if max_section_samples_per_segment is None:
        # Bound optional raw sampling work; the standard pipeline still computes
        # exactly one section for every final point.
        max_section_samples_per_segment = max(125, int(n_points) + 25)
    seg_path = resolve_feature_path(parentdir, SEGMENT_ASSIGNMENTS_NAME)
    if seg_path is None:
        print(f"  跳过 (无分段文件): {seg_path}")
        return

    junction_smoothing = {
        'applied': False,
        'method': 'local_c1_cubic_hermite',
        'events': [],
    }
    if smooth_segment_junctions:
        try:
            junction_smoothing = smooth_internal_anatomical_junctions(
                stl_path,
                angle_threshold_deg=junction_kink_angle_threshold_deg,
                local_angle_excess_threshold_deg=(
                    junction_kink_excess_threshold_deg),
                half_window_mm=junction_smoothing_half_window_mm,
                tangent_span_mm=junction_tangent_span_mm)
            if (junction_smoothing.get('applied')
                    and not junction_smoothing.get('reused_existing')):
                print(
                    "  中心线内部拼接平滑: "
                    f"{len(junction_smoothing.get('events', []))} 处")
            elif junction_smoothing.get('reused_existing'):
                print("  中心线内部拼接平滑: 复用已保存结果")
        except Exception as exc:
            print(f"  [warn] 中心线内部拼接平滑失败: {exc}")

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
    n_total_section_failures = 0
    n_total_endpoint_zeroed = 0
    n_total_side_branch_voronoi = 0
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
    n_total_clinical_endpoint_junctions = 0

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
            network_voronoi_centerlines = _build_network_voronoi_centerlines(
                side_branch_anchors, coords_by_seg, radii_by_seg, n_points,
                junction_exclusion_mm=centerline_voronoi_exclusion_mm)
            raw_profile = _extract_branch_raw_profile(
                seg_path_ids, nodes, mesh,
                curvature_window=curvature_window,
                section_step=1,
                branch_coords=branch_coords,
                max_section_samples=n_points,
                normal_search_policy=normal_search_policy,
                centerline_voronoi_exclusion_mm=(
                    centerline_voronoi_exclusion_mm),
                network_voronoi_centerlines=network_voronoi_centerlines)

            if raw_profile is None:
                profiles[seg_name] = None
                continue

            n_total_section_failures += int(raw_profile.get(
                '_n_section_failures', 0))

            # 重采样到 n_points
            resampled = _resample_profile(raw_profile, n_points=n_points)
            if resampled is None:
                profiles[seg_name] = None
                continue
            # Internal side branches use network Voronoi ownership.  Only
            # actual vessel endpoints are eligible for interval masking.
            endpoint_junctions = plan.get('endpoint_junctions', [])
            start_junction = next(
                (item for item in endpoint_junctions
                 if item.get('side') == 'start'), None)
            end_junction = next(
                (item for item in endpoint_junctions
                 if item.get('side') == 'end'), None)
            resampled['side_branch_contamination_mask'] = [0.0] * n_points
            resampled['side_branch_contamination_events'] = []
            resampled['n_side_branch_contamination_masked'] = 0
            resampled['n_side_branch_junction_zeroed'] = 0
            resampled['side_branch_network_voronoi'] = bool(
                network_voronoi_centerlines)
            resampled['network_voronoi_competitors'] = [
                {
                    'segment': item['segment'],
                    'junction_node_id': item['junction_node_id'],
                    'junction_exclusion_mm': item['junction_exclusion_mm'],
                    'radius_mm': item['radius_mm'],
                    'centerline_coords': item['centerline_coords'].tolist(),
                }
                for item in network_voronoi_centerlines
            ]
            resampled = _mask_endpoint_junction_sections(
                resampled,
                ratio_threshold=area_jump_ratio_threshold,
                allow_terminal_start=(start_junction is not None and actual_start <= 0.0),
                allow_terminal_end=(end_junction is not None and actual_end >= 1.0),
                terminal_padding_sections=6,
                terminal_reference_points=area_jump_reference_points,
                protected_side_branch_arcs=[
                    item['arc_center_mm'] for item in side_branch_anchors
                    if item.get('arc_center_mm') is not None
                ],
                side_branch_protection_mm=(
                    centerline_voronoi_exclusion_mm))
            endpoint_mask = list(resampled.get(
                'area_jump_terminal_mask', [0.0] * n_points))
            endpoint_count = int(resampled.get(
                'n_area_jump_terminal_masked', 0))
            resampled['junction_endpoint_excluded'] = endpoint_mask
            resampled['n_junction_endpoint_flagged'] = endpoint_count
            resampled['n_junction_endpoint_excluded'] = endpoint_count
            resampled['junction_endpoint_values_masked'] = True
            n_total_endpoint_zeroed += endpoint_count
            n_total_side_branch_voronoi += len(network_voronoi_centerlines)
            if endpoint_junctions:
                n_total_clinical_endpoint_junctions += len(endpoint_junctions)
            resampled['clinical_junction_plan'] = {
                'endpoint_trim_start_mm': trim_start_mm,
                'endpoint_trim_end_mm': trim_end_mm,
                'endpoint_junctions': endpoint_junctions,
                'side_branch_anchors': side_branch_anchors,
                'radius_factor': 1.25,
                'method': (
                    'endpoint_raw_area_ratio_and_side_branch_network_voronoi'),
            }

            resampled['n_section_success_final'] = int(
                raw_profile.get('_n_final_success', 0))
            resampled['n_section_failures'] = int(
                raw_profile.get('_n_section_failures', 0))
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
            resampled['section_assignment_method'] = raw_profile.get(
                '_section_assignment_method', 'centerline_voronoi')
            resampled['centerline_voronoi_exclusion_mm'] = float(
                raw_profile.get('_centerline_voronoi_exclusion_mm',
                                centerline_voronoi_exclusion_mm))
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

            profiles[seg_name] = resampled

        except Exception as e:
            print(f"    [{seg_name}] 剖面提取失败: {e}")
            profiles[seg_name] = None

    # 元数据
    profiles['_meta'] = {
        'patient_id': seg_data.get('patient_id'),
        'is_post_tips': seg_data.get('is_post_tips'),
        'n_points': n_points,
        'area_jump_ratio_threshold': float(area_jump_ratio_threshold),
        'area_jump_reference_points': int(area_jump_reference_points),
        'max_section_samples_per_segment': int(
            max_section_samples_per_segment),
        'normal_search_policy': _normal_search_policy(normal_search_policy),
        'section_assignment_method': 'centerline_voronoi',
        'side_branch_assignment_method': 'centerline_network_voronoi',
        'centerline_voronoi_exclusion_mm': float(
            centerline_voronoi_exclusion_mm),
        'section_filter_policy': 'endpoint_zero_only',
        'n_total_section_failures': int(n_total_section_failures),
        'n_total_endpoint_junction_zeroed': int(n_total_endpoint_zeroed),
        'n_total_side_branch_junction_zeroed': 0,
        'n_total_side_branch_network_voronoi': int(
            n_total_side_branch_voronoi),
        'clinical_junction_method': (
            'endpoint_raw_area_ratio_and_side_branch_network_voronoi'),
        'clinical_radius_factor': 1.25,
        'n_total_clinical_endpoint_junctions': int(
            n_total_clinical_endpoint_junctions),
        'median_centerline_radius_by_segment_mm': {
            str(k): float(v) for k, v in radii_by_seg.items()
            if v is not None
        },
        'analysis_ranges_file': SEGMENT_ASSIGNMENTS_NAME if analysis_ranges else None,
        'analysis_ranges_applied': sorted(analysis_ranges.keys()),
        'junction_smoothing': junction_smoothing,
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
            'endpoint_junction_mask',
            'side_branch_contamination_mask',
            'section_valid',             # unified export: 1=draw/statistically valid
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
          f"求交失败 {n_total_section_failures} 处, "
          f"汇合端置零 {n_total_endpoint_zeroed} 处, "
          f"侧支网络 Voronoi {n_total_side_branch_voronoi} 处")
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

def batch_extract_profiles(root_folder, n_points=200, pitch=0.5,
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
        batch_extract_profiles(r"F:\PCG data\dataset\test4all_sample")
