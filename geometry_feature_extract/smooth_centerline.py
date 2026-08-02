"""
中心线平滑模块（STL版）
========================
坐标已是物理坐标(mm)，无需spacing转换。

流程：
  1. 读取中心线树 → 邻接表 → 分类节点
  2. 提取所有"段"(关键点之间的路径)
  3. 逐段样条平滑
  4. 邻接表重组 → BFS建树 → 输出 newCenterlist.txt
"""

import json
import os
import numpy as np
from scipy.interpolate import CubicHermiteSpline, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict, deque

from utils import (load_raw_tree, load_tree, classify_nodes, path_to_coords,
                   save_tree)
from features_layout import (SEGMENT_ASSIGNMENTS_NAME, SMOOTH_CENTERLINE_NAME,
                             feature_path, resolve_feature_path)


def smooth_centerline(stl_path, output_txt_path=None,
                      smooth_factor=500, n_mult=3,
                      w_key=1e3, w_mid=10.0):
    """
    对中心线进行样条平滑。

    参数:
        stl_path: str, STL文件路径（用于定位中心线文件目录）
        output_txt_path: str or None, 输出路径，默认 newCenterlist.txt
        smooth_factor: float, 越大越平滑
        n_mult: int, 采样密度倍数
        w_key: float, 关键点（端点/分支点）权重
        w_mid: float, 普通点权重

    返回:
        new_tree: list of list [ID, x, y, z, parentID, leftChildID, rightChildID]
    """
    # ========== 第1步：读取 ==========
    print("[1/4] 读取原始中心线(CenterlinePoints.txt)...")
    nodes, adj, parentdir = load_raw_tree(stl_path)
    print(f"  节点数: {len(nodes)}")

    endpoints, branch_points = classify_nodes(nodes, adj)
    key_points = endpoints | branch_points
    print(f"  端点: {len(endpoints)}, 分支点: {len(branch_points)}")

    # ========== 第2步：提取所有段 ==========
    print("[2/4] 提取分支段...")
    segments = _extract_segments(nodes, adj, key_points)

    if len(segments) == 0 and len(nodes) > 0:
        # 退化情况：无关键点，整条线作为一段
        start_id = min(nodes.keys())
        segment = [start_id]
        visited = {start_id}
        current = start_id
        while True:
            next_nodes = [n for n in adj[current] if n not in visited]
            if not next_nodes:
                break
            current = next_nodes[0]
            segment.append(current)
            visited.add(current)
        segments.append(segment)

    print(f"  段数: {len(segments)}")

    # ========== 第3步：逐段平滑 ==========
    print("[3/4] 样条平滑...")
    smoothed_segments = []
    for seg_idx, segment in enumerate(segments):
        coords = [[nodes[nid]['x'], nodes[nid]['y'], nodes[nid]['z']]
                   for nid in segment]

        if len(coords) < 2:
            smoothed_segments.append(coords)
        elif len(coords) == 2:
            p0, p1 = np.array(coords[0]), np.array(coords[1])
            n_interp = max(3, n_mult * 2)
            interp = [tuple(p0 + t * (p1 - p0)) for t in np.linspace(0, 1, n_interp)]
            smoothed_segments.append(interp)
        else:
            try:
                smoothed = _fit_spline_segment(
                    coords, smooth_factor, n_mult, w_key, w_mid)
                smoothed_segments.append(smoothed)
            except Exception as e:
                print(f"    段{seg_idx}平滑失败({e})，保留原始")
                smoothed_segments.append([tuple(c) for c in coords])

    # ========== 第4步：重建树 ==========
    print("[4/4] 重建树...")
    new_tree = _rebuild_tree(
        segments, smoothed_segments, nodes, key_points, endpoints)

    if output_txt_path is None:
        output_txt_path = str(feature_path(parentdir, SMOOTH_CENTERLINE_NAME, create=True))
    else:
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    save_tree(new_tree, output_txt_path)

    n_ep = sum(1 for r in new_tree if r[5] == -1 and r[6] == -1)
    n_br = sum(1 for r in new_tree if r[5] != -1 and r[6] != -1)
    print(f"  新节点: {len(new_tree)}, 端点: {n_ep}, 分支点: {n_br}")
    print(f"  已保存: {output_txt_path}")

    return new_tree


def smooth_existing_anatomical_segment(stl_path, segment_name, sigma_mm=3.0):
    """Re-smooth one already segmented vessel without changing its topology.

    The original node IDs, anatomical endpoints, and branch connectivity are
    retained.  Coordinates are first made uniform in arc length, then low-pass
    filtered in physical units and resampled back to the original node count.
    This removes high-frequency centreline oscillation without changing the
    segment assignment that downstream profile extraction relies on.
    """
    if sigma_mm <= 0:
        raise ValueError('sigma_mm must be positive')

    parentdir = os.path.dirname(stl_path)
    assignments_path = resolve_feature_path(
        parentdir, SEGMENT_ASSIGNMENTS_NAME)
    if assignments_path is None:
        raise FileNotFoundError('segment_assignments.json is required')
    with open(assignments_path, 'r', encoding='utf-8') as handle:
        assignments = json.load(handle)
    info = (assignments.get('segments') or {}).get(segment_name)
    path = [int(node_id) for node_id in (info or {}).get('path', [])]
    if len(path) < 3:
        raise ValueError(f'{segment_name} has fewer than three centreline nodes')

    nodes, _, _ = load_tree(stl_path)
    if any(node_id not in nodes for node_id in path):
        raise ValueError(f'{segment_name} path does not match the active centreline')
    coords = path_to_coords(path, nodes)
    arc = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(coords, axis=0), axis=1))))
    total = float(arc[-1])
    if total <= 1e-9:
        raise ValueError(f'{segment_name} has zero centreline length')

    n_points = len(coords)
    targets = np.linspace(0.0, total, n_points)
    uniform = np.column_stack([
        np.interp(targets, arc, coords[:, axis]) for axis in range(3)
    ])
    step_mm = total / (n_points - 1)
    filtered = gaussian_filter1d(
        uniform, sigma=float(sigma_mm) / step_mm, axis=0, mode='nearest')
    # Junction locations are anatomical anchors and must not drift.
    filtered[0] = coords[0]
    filtered[-1] = coords[-1]

    filtered_arc = np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(filtered, axis=0), axis=1))))
    if float(filtered_arc[-1]) <= 1e-9:
        raise ValueError(f'{segment_name} smoothing collapsed the centreline')
    resample_targets = np.linspace(0.0, float(filtered_arc[-1]), n_points)
    smoothed = np.column_stack([
        np.interp(resample_targets, filtered_arc, filtered[:, axis])
        for axis in range(3)
    ])
    smoothed[0] = coords[0]
    smoothed[-1] = coords[-1]

    for node_id, coord in zip(path, smoothed):
        nodes[node_id]['x'] = float(coord[0])
        nodes[node_id]['y'] = float(coord[1])
        nodes[node_id]['z'] = float(coord[2])

    output_path = feature_path(parentdir, SMOOTH_CENTERLINE_NAME, create=True)
    tree = [[
        node_id,
        node['x'], node['y'], node['z'],
        node['parent'], node['left'], node['right'],
    ] for node_id, node in sorted(nodes.items())]
    save_tree(tree, str(output_path))
    return {
        'segment': segment_name,
        'n_points': n_points,
        'sigma_mm': float(sigma_mm),
        'input_length_mm': total,
        'output_length_mm': float(filtered_arc[-1]),
        'output_path': str(output_path),
    }


def _path_arc_length(coords):
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return np.asarray([0.0], dtype=float)
    return np.concatenate(([0.0], np.cumsum(
        np.linalg.norm(np.diff(coords, axis=0), axis=1))))


def _unit_vector(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def _point_at_path_arc(coords, arc, distance):
    distance = float(np.clip(distance, float(arc[0]), float(arc[-1])))
    return np.asarray([
        np.interp(distance, arc, np.asarray(coords, dtype=float)[:, axis])
        for axis in range(3)
    ], dtype=float)


def _junction_turn_angle_degrees(coords, index, tangent_span_mm=2.0):
    """Angle between fitted incoming and outgoing tangents at one path node."""
    coords = np.asarray(coords, dtype=float)
    index = int(index)
    if not 0 < index < len(coords) - 1:
        return 0.0
    arc = _path_arc_length(coords)
    span = max(float(tangent_span_mm), 1e-6)
    center = coords[index]
    left = _point_at_path_arc(coords, arc, arc[index] - span)
    right = _point_at_path_arc(coords, arc, arc[index] + span)
    incoming = _unit_vector(center - left)
    outgoing = _unit_vector(right - center)
    if incoming is None or outgoing is None:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(
        np.dot(incoming, outgoing), -1.0, 1.0))))


def _vertex_turn_angle_degrees(coords, index):
    """Immediate discrete tangent jump at one centerline node."""
    coords = np.asarray(coords, dtype=float)
    index = int(index)
    if not 0 < index < len(coords) - 1:
        return 0.0
    incoming = _unit_vector(coords[index] - coords[index - 1])
    outgoing = _unit_vector(coords[index + 1] - coords[index])
    if incoming is None or outgoing is None:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(
        np.dot(incoming, outgoing), -1.0, 1.0))))


def _junction_kink_metrics(coords, index, neighbor_radius=3):
    """Measure whether a turn is a localized tangent discontinuity."""
    coords = np.asarray(coords, dtype=float)
    index = int(index)
    radius = max(1, int(neighbor_radius))
    vertex_angle = _vertex_turn_angle_degrees(coords, index)
    neighbor_angles = [
        _vertex_turn_angle_degrees(coords, candidate)
        for candidate in range(
            max(1, index - radius),
            min(len(coords) - 1, index + radius + 1))
        if candidate != index
    ]
    background = (
        float(np.median(neighbor_angles)) if neighbor_angles else 0.0)
    return {
        'vertex_angle_deg': vertex_angle,
        'neighbor_angle_median_deg': background,
        'angle_excess_deg': vertex_angle - background,
    }


def _path_edges(path):
    """Return undirected edges making up a centreline path."""
    return {
        tuple(sorted((int(a), int(b))))
        for a, b in zip(path, path[1:])
    }


def _atomic_centerline_segments(adjacency):
    """Return Web-compatible maximal centreline atom segments.

    A segment starts at a node whose topological degree is not two and is
    followed until the next such node.  Therefore every internal atom node is
    guaranteed to have no attached branch, matching the editable segments
    shown by the Web workbench.
    """
    if not adjacency:
        return []
    adjacency = {
        int(node_id): {int(nb) for nb in neighbors}
        for node_id, neighbors in adjacency.items()
    }
    anchors = sorted(
        node_id for node_id, neighbors in adjacency.items()
        if len(neighbors) != 2
    )
    seen_edges = set()
    atoms = []

    for anchor in anchors:
        for first in sorted(adjacency.get(anchor, ())):
            edge = tuple(sorted((anchor, first)))
            if edge in seen_edges:
                continue
            path = [anchor, first]
            seen_edges.add(edge)
            previous, current = anchor, first
            while len(adjacency.get(current, ())) == 2:
                next_nodes = [
                    nb for nb in adjacency[current] if nb != previous
                ]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                edge = tuple(sorted((current, nxt)))
                if edge in seen_edges:
                    break
                path.append(nxt)
                seen_edges.add(edge)
                previous, current = current, nxt
            start_id, end_id = int(path[0]), int(path[-1])
            atoms.append({
                'id': f'{min(start_id, end_id)}:{max(start_id, end_id)}',
                'start_id': start_id,
                'end_id': end_id,
                'path': [int(node_id) for node_id in path],
            })

    atoms.sort(key=lambda item: (item['start_id'], item['end_id']))
    return atoms


def _anatomical_atomic_join_candidates(assignments, adjacency,
                                        segment_names=None):
    """Find shared endpoints of atoms belonging to one anatomical vessel.

    Manual atom assignments from Web are authoritative.  For patients without
    manual assignments, an atom is assigned automatically when all of its
    edges are contained in an anatomical segment path.  A node is a join
    candidate only when at least two atoms of the same vessel terminate there;
    this prevents a different-vessel side branch from being treated as a
    splice of the main vessel.
    """
    assignments = assignments if isinstance(assignments, dict) else {}
    adjacency = adjacency if isinstance(adjacency, dict) else {}
    segments = assignments.get('segments') or {}
    selected = None if segment_names is None else {str(name) for name in segment_names}
    segment_edges = {
        str(vessel): _path_edges(info.get('path') or [])
        for vessel, info in segments.items()
        if isinstance(info, dict) and info.get('path')
        and (selected is None or str(vessel) in selected)
    }
    manual = assignments.get('assignments') or {}
    if not isinstance(manual, dict):
        manual = {}

    atoms = _atomic_centerline_segments(adjacency)
    by_vessel = defaultdict(lambda: defaultdict(list))
    for atom in atoms:
        atom_id = atom['id']
        saved = manual.get(atom_id)
        vessel = ''
        if isinstance(saved, dict):
            vessel = str(saved.get('vessel') or '').lower()
        if not vessel:
            atom_edges = _path_edges(atom['path'])
            vessel = next(
                (name for name, edges in segment_edges.items()
                 if atom_edges and atom_edges <= edges),
                '',
            )
        if not vessel or vessel not in segment_edges:
            continue
        by_vessel[vessel][atom['start_id']].append(atom_id)
        by_vessel[vessel][atom['end_id']].append(atom_id)

    candidates = defaultdict(list)
    for vessel, endpoint_atoms in by_vessel.items():
        path = [int(node_id) for node_id in segments[vessel].get('path', [])]
        if len(path) < 3:
            continue
        path_endpoints = {path[0], path[-1]}
        for node_id, atom_ids in endpoint_atoms.items():
            if node_id in path_endpoints or len(atom_ids) < 2:
                continue
            # Preserve Web's deterministic atom ordering and remove duplicates.
            unique_ids = sorted(set(atom_ids))
            candidates[vessel].append({
                'junction_node_id': int(node_id),
                'atomic_segment_ids': unique_ids,
            })
    for vessel in candidates:
        candidates[vessel].sort(key=lambda item: item['junction_node_id'])
    return dict(candidates)


def _endpoint_path_tangent(coords, index):
    coords = np.asarray(coords, dtype=float)
    index = int(index)
    if index <= 0:
        return _unit_vector(coords[1] - coords[0])
    if index >= len(coords) - 1:
        return _unit_vector(coords[-1] - coords[-2])
    return _unit_vector(coords[index + 1] - coords[index - 1])


def _smooth_path_at_junction(coords, index, half_window_mm=6.0,
                             tangent_span_mm=2.0):
    """Apply a local C1 Hermite curve while keeping the junction fixed."""
    coords = np.asarray(coords, dtype=float)
    index = int(index)
    if not 1 < index < len(coords) - 2:
        return coords.copy(), None

    arc = _path_arc_length(coords)
    center_arc = float(arc[index])
    half_window = max(float(half_window_mm), float(tangent_span_mm), 1e-6)
    left_index = int(np.searchsorted(
        arc, center_arc - half_window, side='left'))
    right_index = int(np.searchsorted(
        arc, center_arc + half_window, side='right') - 1)
    left_index = max(0, min(index - 2, left_index))
    right_index = min(len(coords) - 1, max(index + 2, right_index))
    if not left_index < index < right_index:
        return coords.copy(), None

    before_angle = _junction_turn_angle_degrees(
        coords, index, tangent_span_mm=tangent_span_mm)
    vertex_angle_before = _vertex_turn_angle_degrees(coords, index)
    incoming = _unit_vector(
        coords[index] - _point_at_path_arc(
            coords, arc, center_arc - tangent_span_mm))
    outgoing = _unit_vector(
        _point_at_path_arc(
            coords, arc, center_arc + tangent_span_mm) - coords[index])
    center_tangent = (
        _unit_vector(incoming + outgoing)
        if incoming is not None and outgoing is not None else None)
    if center_tangent is None:
        center_tangent = _unit_vector(coords[right_index] - coords[left_index])
    left_tangent = _endpoint_path_tangent(coords, left_index)
    right_tangent = _endpoint_path_tangent(coords, right_index)
    if any(item is None for item in (
            center_tangent, left_tangent, right_tangent)):
        return coords.copy(), None

    knots = np.asarray(
        [arc[left_index], arc[index], arc[right_index]], dtype=float)
    values = coords[[left_index, index, right_index]]
    derivatives = np.vstack((left_tangent, center_tangent, right_tangent))
    curve = CubicHermiteSpline(
        knots, values, derivatives, axis=0, extrapolate=False)

    smoothed = coords.copy()
    smoothed[left_index:right_index + 1] = curve(
        arc[left_index:right_index + 1])
    # These anchors define the local edit and the branch attachment.  Restoring
    # them exactly also makes repeated profile extraction idempotent.
    smoothed[left_index] = coords[left_index]
    smoothed[index] = coords[index]
    smoothed[right_index] = coords[right_index]
    after_angle = _junction_turn_angle_degrees(
        smoothed, index, tangent_span_mm=tangent_span_mm)
    vertex_angle_after = _vertex_turn_angle_degrees(smoothed, index)
    displacement = np.linalg.norm(smoothed - coords, axis=1)
    return smoothed, {
        'path_index': index,
        'left_path_index': left_index,
        'right_path_index': right_index,
        'angle_before_deg': before_angle,
        'angle_after_deg': after_angle,
        'vertex_angle_before_deg': vertex_angle_before,
        'vertex_angle_after_deg': vertex_angle_after,
        'max_displacement_mm': float(np.max(displacement)),
        'half_window_mm': half_window,
        'tangent_span_mm': float(tangent_span_mm),
    }


def _segment_geometry_metrics(coords):
    coords = np.asarray(coords, dtype=float)
    arc = _path_arc_length(coords)
    length = float(arc[-1]) if len(arc) else 0.0
    chord = float(np.linalg.norm(coords[-1] - coords[0])) if len(coords) else 0.0
    tortuosity = float(1.0 - chord / length) if length > 1e-9 else 0.0
    curvature = []
    for index in range(1, len(coords) - 1):
        incoming = coords[index] - coords[index - 1]
        outgoing = coords[index + 1] - coords[index]
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length <= 1e-9 or outgoing_length <= 1e-9:
            continue
        angle = float(np.arccos(np.clip(
            np.dot(incoming, outgoing)
            / (incoming_length * outgoing_length), -1.0, 1.0)))
        curvature.append(angle / (0.5 * (
            incoming_length + outgoing_length)))
    return {
        'length_mm': length,
        'tortuosity': tortuosity,
        'mean_curvature': float(np.mean(curvature)) if curvature else 0.0,
    }


def _refresh_assignment_geometry(assignments, nodes):
    for info in (assignments.get('segments') or {}).values():
        if not info or not info.get('path'):
            continue
        path = [int(node_id) for node_id in info['path']]
        if any(node_id not in nodes for node_id in path):
            continue
        coords = path_to_coords(path, nodes)
        info.update(_segment_geometry_metrics(coords))
        info['n_points'] = len(path)
        info['endpoints_id'] = [int(path[0]), int(path[-1])]
        info['endpoints_coord'] = [
            [float(value) for value in coords[0]],
            [float(value) for value in coords[-1]],
        ]
        if 'smoothed_coords' in info:
            info['smoothed_coords'] = coords.tolist()

    for key in ('branch_points', 'endpoints'):
        for item in assignments.get(key, []) or []:
            if not isinstance(item, dict) or item.get('id') is None:
                continue
            node = nodes.get(int(item['id']))
            if node is not None:
                item['coord'] = [
                    float(node['x']), float(node['y']), float(node['z'])]


def smooth_internal_anatomical_junctions(
        stl_path, segment_names=None, angle_threshold_deg=15.0,
        local_angle_excess_threshold_deg=10.0,
        half_window_mm=6.0, tangent_span_mm=2.0):
    """Repair non-C1 internal joins and synchronize both Web geometry files.

    Only topology junctions inside an anatomical segment are eligible.  True
    segment endpoints are excluded, node IDs and connectivity are preserved,
    and the shared branch attachment coordinate remains fixed.
    """
    parentdir = os.path.dirname(stl_path)
    assignments_path = resolve_feature_path(
        parentdir, SEGMENT_ASSIGNMENTS_NAME)
    if assignments_path is None:
        raise FileNotFoundError('segment_assignments.json is required')
    with open(assignments_path, 'r', encoding='utf-8') as handle:
        assignments = json.load(handle)

    nodes, adjacency, _ = load_tree(stl_path)
    segments = assignments.get('segments') or {}
    selected_names = (
        list(segments) if segment_names is None
        else [str(name) for name in segment_names])
    join_candidates = _anatomical_atomic_join_candidates(
        assignments, adjacency, segment_names=selected_names)
    events = []

    for segment_name in selected_names:
        info = segments.get(segment_name)
        path = [int(node_id) for node_id in (info or {}).get('path', [])]
        if len(path) < 5 or any(node_id not in nodes for node_id in path):
            continue
        coords = path_to_coords(path, nodes)
        path_indices = {
            int(node_id): index for index, node_id in enumerate(path)
        }
        for candidate in join_candidates.get(segment_name, ()):
            node_id = int(candidate['junction_node_id'])
            path_index = path_indices.get(node_id)
            if path_index is None or not 1 < path_index < len(path) - 2:
                continue
            angle = _junction_turn_angle_degrees(
                coords, path_index, tangent_span_mm=tangent_span_mm)
            kink = _junction_kink_metrics(coords, path_index)
            if (kink['vertex_angle_deg'] < float(angle_threshold_deg)
                    or kink['angle_excess_deg']
                    < float(local_angle_excess_threshold_deg)):
                continue
            updated, detail = _smooth_path_at_junction(
                coords, path_index,
                half_window_mm=half_window_mm,
                tangent_span_mm=tangent_span_mm)
            if (detail is None
                    or detail['angle_after_deg'] >= angle
                    or detail['vertex_angle_after_deg']
                    >= kink['vertex_angle_deg']):
                continue
            coords = updated
            for current_id, coord in zip(path, coords):
                nodes[current_id]['x'] = float(coord[0])
                nodes[current_id]['y'] = float(coord[1])
                nodes[current_id]['z'] = float(coord[2])
            events.append({
                'segment': segment_name,
                'junction_node_id': node_id,
                'atomic_segment_ids': list(candidate['atomic_segment_ids']),
                'candidate_method': (
                    'web_atomic_segments_within_anatomical_segment'),
                'neighbor_angle_median_deg': (
                    kink['neighbor_angle_median_deg']),
                'angle_excess_deg': kink['angle_excess_deg'],
                **detail,
            })

    result = {
        'applied': bool(events),
        'method': 'local_c1_cubic_hermite',
        'angle_threshold_deg': float(angle_threshold_deg),
        'local_angle_excess_threshold_deg': float(
            local_angle_excess_threshold_deg),
        'half_window_mm': float(half_window_mm),
        'tangent_span_mm': float(tangent_span_mm),
        'candidate_generation_method': (
            'web_atomic_segments_within_anatomical_segment'),
        'candidate_count': int(sum(
            len(items) for items in join_candidates.values())),
        'events': events,
    }
    if not events:
        previous = assignments.get('junction_smoothing')
        if isinstance(previous, dict) and previous.get('applied'):
            result = dict(previous)
            candidate_lookup = {
                (str(vessel), int(item['junction_node_id'])): item
                for vessel, items in join_candidates.items()
                for item in items
            }
            result['events'] = []
            for previous_event in previous.get('events', ()):
                event = dict(previous_event)
                candidate = candidate_lookup.get((
                    str(event.get('segment')),
                    int(event.get('junction_node_id', -1)),
                ))
                if candidate is not None:
                    event['atomic_segment_ids'] = list(
                        candidate['atomic_segment_ids'])
                    event['candidate_method'] = (
                        'web_atomic_segments_within_anatomical_segment')
                result['events'].append(event)
            result['reused_existing'] = True
            result['candidate_generation_method'] = (
                'web_atomic_segments_within_anatomical_segment')
            result['candidate_count'] = int(sum(
                len(items) for items in join_candidates.values()))
        return result

    _refresh_assignment_geometry(assignments, nodes)
    assignments['junction_smoothing'] = result
    tree = [[
        int(node_id),
        float(node['x']), float(node['y']), float(node['z']),
        int(node['parent']), int(node['left']), int(node['right']),
    ] for node_id, node in sorted(nodes.items())]

    centerline_path = feature_path(
        parentdir, SMOOTH_CENTERLINE_NAME, create=True)
    centerline_tmp = str(centerline_path) + '.tmp'
    assignments_tmp = str(assignments_path) + '.tmp'
    save_tree(tree, centerline_tmp)
    with open(assignments_tmp, 'w', encoding='utf-8') as handle:
        json.dump(assignments, handle, ensure_ascii=False, indent=2)
    os.replace(centerline_tmp, centerline_path)
    os.replace(assignments_tmp, assignments_path)
    result['centerline_path'] = str(centerline_path)
    result['segment_assignments_path'] = str(assignments_path)
    return result


# ============================================================
# 内部函数
# ============================================================

def _extract_segments(nodes, adj, key_points):
    """提取关键点之间的所有段"""
    segments = []
    visited_edges = set()

    for start in key_points:
        for neighbor in adj[start]:
            edge = (min(start, neighbor), max(start, neighbor))
            if edge in visited_edges:
                continue

            segment = [start]
            prev = start
            current = neighbor

            while current not in key_points:
                segment.append(current)
                next_nodes = [n for n in adj[current] if n != prev]
                if not next_nodes:
                    break
                prev = current
                current = next_nodes[0]

            segment.append(current)

            for i in range(len(segment) - 1):
                e = (min(segment[i], segment[i + 1]), max(segment[i], segment[i + 1]))
                visited_edges.add(e)

            segments.append(segment)

    return segments


def _rebuild_tree(segments, smoothed_segments, nodes, key_points, endpoints):
    """从平滑后的段重建树结构"""
    new_adj = defaultdict(set)
    new_coords = {}
    new_id_counter = [0]
    key_new_id = {}

    def alloc_id():
        nid = new_id_counter[0]
        new_id_counter[0] += 1
        return nid

    # 为关键点分配ID
    for kp in key_points:
        n = nodes[kp]
        nid = alloc_id()
        key_new_id[kp] = nid
        new_coords[nid] = (n['x'], n['y'], n['z'])

    # 处理每段：插入中间点
    for seg_idx, segment in enumerate(segments):
        smoothed = smoothed_segments[seg_idx]
        start_new = key_new_id[segment[0]]
        end_new = key_new_id[segment[-1]]
        mid_coords = smoothed[1:-1]

        if len(mid_coords) == 0:
            new_adj[start_new].add(end_new)
            new_adj[end_new].add(start_new)
        else:
            prev_nid = start_new
            for coord in mid_coords:
                cur_nid = alloc_id()
                new_coords[cur_nid] = tuple(coord)
                new_adj[prev_nid].add(cur_nid)
                new_adj[cur_nid].add(prev_nid)
                prev_nid = cur_nid
            new_adj[prev_nid].add(end_new)
            new_adj[end_new].add(prev_nid)

    # BFS建树
    root_new = None
    for kp in endpoints:
        if kp in key_new_id:
            root_new = key_new_id[kp]
            break
    if root_new is None:
        root_new = 0

    visited = set()
    queue = deque([(root_new, -1)])
    visited.add(root_new)
    bfs_order = []

    while queue:
        nid, pid = queue.popleft()
        bfs_order.append((nid, pid))
        for nb in sorted(new_adj[nid]):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, nid))

    children_map = defaultdict(list)
    for nid, pid in bfs_order:
        if pid >= 0:
            children_map[pid].append(nid)

    old_to_final = {}
    for final_id, (nid, pid) in enumerate(bfs_order):
        old_to_final[nid] = final_id

    new_tree = []
    for final_id, (nid, pid) in enumerate(bfs_order):
        phys = new_coords[nid]
        parent_final = old_to_final[pid] if pid >= 0 else -1
        children = children_map.get(nid, [])
        lc = old_to_final[children[0]] if len(children) >= 1 else -1
        rc = old_to_final[children[1]] if len(children) >= 2 else -1
        new_tree.append([final_id, phys[0], phys[1], phys[2], parent_final, lc, rc])

    return new_tree


def _fit_spline_segment(coords, smooth_factor=500, n_mult=3,
                        w_key=1e3, w_mid=10.0):
    """对一段坐标做样条平滑"""
    pts = np.asarray(coords, float)
    diffs = np.diff(pts, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    t_pts = np.concatenate(([0], np.cumsum(seglen)))
    L = t_pts[-1]

    if L <= 0:
        return [tuple(p) for p in pts]

    weights = np.ones(len(pts)) * w_mid
    weights[0] = w_key
    weights[-1] = w_key

    M = len(t_pts)
    k = min(3, M - 1)

    sx = UnivariateSpline(t_pts, pts[:, 0], w=weights, k=k, s=smooth_factor)
    sy = UnivariateSpline(t_pts, pts[:, 1], w=weights, k=k, s=smooth_factor)
    sz = UnivariateSpline(t_pts, pts[:, 2], w=weights, k=k, s=smooth_factor)

    # 密采样后等弧长重采样
    dense = max(2000, M * 10)
    us = np.linspace(0, L, dense)
    curve = np.vstack((sx(us), sy(us), sz(us))).T

    dseg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.insert(np.cumsum(dseg), 0, 0.0)
    tot = cum[-1]
    if tot <= 0:
        return [tuple(p) for p in pts]

    N = max(len(pts) * n_mult, 3)
    target = np.linspace(0, tot, N)
    sampled = []
    idx = 0

    for td in target:
        while idx < len(cum) - 1 and cum[idx + 1] < td:
            idx += 1
        if idx >= len(cum) - 1:
            sampled.append(curve[-1])
        else:
            t0, t1 = cum[idx], cum[idx + 1]
            a = (td - t0) / (t1 - t0) if t1 > t0 else 0
            sampled.append(curve[idx] + a * (curve[idx + 1] - curve[idx]))

    sampled = [tuple(p) for p in sampled]
    sampled[0] = tuple(pts[0])
    sampled[-1] = tuple(pts[-1])
    return sampled


if __name__ == '__main__':
    import sys, time
    path = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    t0 = time.time()
    smooth_centerline(path)
    print(f"\n耗时: {time.time() - t0:.2f}s")
