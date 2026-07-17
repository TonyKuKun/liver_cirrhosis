"""
血管解剖分段模块（v4 - LPV/RPV 主导的 MPV 扩展）
================================================
基于中心线树拓扑（分支点/端点）+ 物理先验（长度/曲率/方向）
对门静脉系统进行解剖分段。

支持的解剖结构:
  MPV  - 门静脉主脉
  SV   - 脾静脉
  SMV  - 肠系膜上静脉
  LPV  - 肝门左静脉
  RPV  - 肝门右静脉
  TIPS - TIPS术后支架（仅术后）
  LGV  - 胃左静脉（术前代偿）
  PGV  - 胃后静脉（术前代偿）

判别准则:
  (1) MPV 初始: 两端均为分支点的段, 多条候选时按 L·exp(-2·τ) 选最长最直
  (2) SV 端 vs 肝侧端: 子树中 SV-score = L·(τ+0.01) 高者为 SV 端
  (3) SV / SMV: SV-score 最高 = SV (长且弯), 剩余 = SMV
  (4) TIPS: 肝侧子树全部端点段按 TIPS-score = L·exp(-2.5·τ) 评分,
            最高者 = TIPS (长且直, 长度主导)
  (5) LPV / RPV: 端点 X 坐标 (LPS 坐标系: X 大者 = LPV)
  (6) MPV 终点扩展: bp_mpv_end = LPV/RPV 起点中沿弧长距 SV 端最远的 bp
                  (TIPS 不参与, 因其为人工分流)
  (7) 段裁剪: 起点 == bp_mpv_end 时沿段找下一个 bp; 否则段不动
  (8) 术前 LGV vs PGV: 3 个 bp 排成链 bp1-bp2-bp3, 计算 bp1<->bp3 路径 τ
                       小 -> LGV 代偿 (MPV 贯穿), 大 -> PGV 代偿 (MPV = bp1->bp2)
  (9) PGV 代偿下 SV-distal vs PGV: 方向一致性 cos(SV入射, 候选出射)
                                   值大者 = SV-distal, 值小者 = PGV
      并做 PGV 合理性质控: 若 PGV 分叉点过近 SV-SMV 汇合且候选过短,
      降级为无代偿分段, 避免把汇合处短毛刺误标为 PGV.
"""

import os
import json
import numpy as np
from collections import deque

from utils import (
    load_tree, classify_nodes, find_path,
    path_to_coords, path_physical_length
)
from anatomical_segmentation import segment_anatomically

SEGMENT_VESSELS_VERSION = "global-topology-patient-coordinates-v4-20260713-post-tips-hilar-transition"

# Active path: segment_vessels() delegates to segment_anatomically(). The
# legacy private helpers below are retained only for result comparison and are
# not used by the production entry point.


# ============================================================
# 文件夹命名规则
# ============================================================

def is_post_tips(folder_name):
    """是否 TIPS 术后: 名称包含 #"""
    return '#' in folder_name


# ============================================================
# 几何工具：曲率 / 方向
# ============================================================

def _path_tortuosity(coords):
    """1 - 弦/弧长。0 = 笔直, 越大越弯曲。"""
    coords = np.asarray(coords)
    if len(coords) < 2:
        return 0.0
    chord = np.linalg.norm(coords[-1] - coords[0])
    arclen = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    if arclen <= 1e-6:
        return 0.0
    return float(1.0 - chord / arclen)


def _path_mean_curvature(coords):
    """路径平均离散曲率 (1/mm)"""
    coords = np.asarray(coords)
    if len(coords) < 3:
        return 0.0
    ks = []
    for i in range(1, len(coords) - 1):
        v1, v2 = coords[i] - coords[i - 1], coords[i + 1] - coords[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        ks.append(np.arccos(cos_a) / (0.5 * (n1 + n2)))
    return float(np.mean(ks)) if ks else 0.0


def _direction_at_start(coords, sample_dist=8.0):
    """路径从首端出发的单位方向 (沿弧长走 sample_dist mm 取参考点)。"""
    coords = np.asarray(coords)
    if len(coords) < 2:
        return None
    start = coords[0]
    cumlen = 0.0
    sample_pt = coords[-1]
    for i in range(1, len(coords)):
        cumlen += np.linalg.norm(coords[i] - coords[i - 1])
        if cumlen >= sample_dist:
            sample_pt = coords[i]
            break
    direction = sample_pt - start
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-6 else None


def _direction_at_end(coords, sample_dist=8.0):
    """路径在末端的单位入射方向 (指向最后一点)。"""
    coords = np.asarray(coords)
    if len(coords) < 2:
        return None
    end = coords[-1]
    cumlen = 0.0
    sample_pt = coords[0]
    for i in range(len(coords) - 1, 0, -1):
        cumlen += np.linalg.norm(coords[i] - coords[i - 1])
        if cumlen >= sample_dist:
            sample_pt = coords[i - 1]
            break
    direction = end - sample_pt
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-6 else None


# ============================================================
# 段评分
# ============================================================

def _seg_score_sv(seg, nodes):
    """SV 评分: 长度 × (曲率 + ε)。SV 偏长且偏弯。"""
    coords = path_to_coords(seg, nodes)
    L = path_physical_length(seg, nodes)
    t = _path_tortuosity(coords)
    return L * (t + 0.01)


def _seg_score_tips(seg, nodes):
    """
    TIPS 评分: 长度主导, 曲率乘性衰减。
    score = L * exp(-2.5 * tortuosity)
    """
    coords = path_to_coords(seg, nodes)
    L = path_physical_length(seg, nodes)
    t = _path_tortuosity(coords)
    return L * np.exp(-2.5 * t)


def _mpv_init_score(seg, nodes):
    """MPV 初始候选评分: 长度主导, 曲率轻微衰减。"""
    coords = path_to_coords(seg, nodes)
    L = path_physical_length(seg, nodes)
    t = _path_tortuosity(coords)
    return L * np.exp(-2.0 * t)


# ============================================================
# 段提取
# ============================================================

def _extract_all_segments(nodes, adj, endpoints, branch_points):
    """提取所有"段"(关键点之间的路径)。"""
    key_points = endpoints | branch_points
    segments = []
    visited_edges = set()

    for start in key_points:
        for neighbor in adj[start]:
            edge = (min(start, neighbor), max(start, neighbor))
            if edge in visited_edges:
                continue

            seg = [start]
            prev = start
            current = neighbor
            while current not in key_points:
                seg.append(current)
                next_nodes = [n for n in adj[current] if n != prev]
                if not next_nodes:
                    break
                prev = current
                current = next_nodes[0]
            seg.append(current)

            for i in range(len(seg) - 1):
                e = (min(seg[i], seg[i + 1]), max(seg[i], seg[i + 1]))
                visited_edges.add(e)
            segments.append(seg)

    return segments


def _find_endpoint_branches_at(segments_raw, bp, endpoints):
    """从分支点 bp 出发, 返回所有另一端是端点的段(统一: bp 在头, 端点在尾)。"""
    result = []
    for seg in segments_raw:
        if seg[0] == bp and seg[-1] in endpoints:
            result.append(list(seg))
        elif seg[-1] == bp and seg[0] in endpoints:
            result.append(list(seg[::-1]))
    return result


def _find_bp_to_bp_segments(segments_raw, branch_points):
    """返回所有两端均为分支点的段。"""
    return [seg for seg in segments_raw
            if seg[0] in branch_points and seg[-1] in branch_points]


# ============================================================
# 子树收集
# ============================================================

def _collect_subtree(adj, root_bp, exclude_neighbor, endpoints, branch_points):
    """
    从 root_bp 出发收集子树, 不回溯到 exclude_neighbor 方向。

    返回 dict:
        'root_branches':   直接从 root_bp 出去的端点分支 (root_bp 在头)
        'deeper_branches': 嵌套在子树深处的端点分支 (deeper_bp 在头)
        'all_branches':    上面两者合并
        'visited_bps':     子树中遇到的所有 bp
    """
    key_points = endpoints | branch_points
    root_branches = []
    deeper_branches = []
    visited_bps = {root_bp}
    visited_edges = set()

    queue = deque()
    for nb in adj[root_bp]:
        if nb == exclude_neighbor:
            continue
        queue.append((root_bp, nb))

    while queue:
        bp_start, first_nb = queue.popleft()
        edge = (min(bp_start, first_nb), max(bp_start, first_nb))
        if edge in visited_edges:
            continue
        visited_edges.add(edge)

        seg = [bp_start]
        prev = bp_start
        current = first_nb
        while current not in key_points:
            seg.append(current)
            next_nodes = [n for n in adj[current] if n != prev]
            if not next_nodes:
                break
            prev = current
            current = next_nodes[0]
        seg.append(current)

        for i in range(len(seg) - 1):
            e = (min(seg[i], seg[i + 1]), max(seg[i], seg[i + 1]))
            visited_edges.add(e)

        if seg[-1] in endpoints:
            if bp_start == root_bp:
                root_branches.append(seg)
            else:
                deeper_branches.append(seg)
        elif seg[-1] in branch_points:
            next_bp = seg[-1]
            if next_bp not in visited_bps:
                visited_bps.add(next_bp)
                for nb in adj[next_bp]:
                    if nb == seg[-2]:
                        continue
                    e2 = (min(next_bp, nb), max(next_bp, nb))
                    if e2 not in visited_edges:
                        queue.append((next_bp, nb))

    return {
        'root_branches': root_branches,
        'deeper_branches': deeper_branches,
        'all_branches': root_branches + deeper_branches,
        'visited_bps': visited_bps,
    }


# ============================================================
# SV / SMV、LPV / RPV 选择
# ============================================================

def _unit_vector(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _orient_path_from(seg, start_id):
    if seg is None:
        return None
    seg = list(seg)
    if not seg:
        return seg
    if seg[0] == start_id:
        return seg
    if seg[-1] == start_id:
        return list(reversed(seg))
    return seg


def _branch_endpoint_direction(seg, nodes):
    """Direction from branch point to terminal endpoint."""
    if seg is None or len(seg) < 2:
        return None
    coords = path_to_coords(seg, nodes)
    return _unit_vector(coords[-1] - coords[0])


def _branch_start_direction(seg, nodes, sample_dist=8.0):
    if seg is None or len(seg) < 2:
        return None
    return _direction_at_start(path_to_coords(seg, nodes), sample_dist)


def _angle_deg(u, v):
    if u is None or v is None:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def _smv_z_dot(seg, nodes):
    """SMV should run roughly toward negative Z, so lower dot(+Z) is better."""
    d = _branch_endpoint_direction(seg, nodes)
    if d is None:
        return 1.0
    return float(d[2])


def _select_smv_by_z(branches, nodes):
    if not branches:
        return None
    return min(branches,
               key=lambda s: (_smv_z_dot(s, nodes),
                              -path_physical_length(s, nodes)))


def _seg_score_sv_trifurcation(seg, nodes, max_len, max_curv, max_tort):
    coords = path_to_coords(seg, nodes)
    L = path_physical_length(seg, nodes)
    curv = _path_mean_curvature(coords)
    tort = _path_tortuosity(coords)
    len_part = L / max(max_len, 1e-6)
    curv_part = curv / max(max_curv, 1e-6)
    tort_part = tort / max(max_tort, 1e-6)
    return 0.50 * len_part + 0.35 * curv_part + 0.15 * tort_part


def _select_sv_for_trifurcation(branches, nodes):
    stats = []
    for br in branches:
        coords = path_to_coords(br, nodes)
        stats.append((
            br,
            path_physical_length(br, nodes),
            _path_mean_curvature(coords),
            _path_tortuosity(coords),
        ))
    max_len = max((x[1] for x in stats), default=1.0)
    max_curv = max((x[2] for x in stats), default=1.0)
    max_tort = max((x[3] for x in stats), default=1.0)
    return max(
        branches,
        key=lambda s: _seg_score_sv_trifurcation(s, nodes, max_len, max_curv, max_tort)
    )


def _select_pgv_by_mpv_sv_angle(candidates, nodes, mpv_seg=None, sv_seg=None):
    """
    PGV tends to leave through the smaller complementary angle near MPV;
    SMV leaves through the larger complementary angle.
    This is used only after SV is known and two lower-end candidates remain.
    """
    if len(candidates) != 2 or mpv_seg is None or sv_seg is None:
        return None

    mpv_dir = _branch_start_direction(mpv_seg, nodes)
    if mpv_dir is None:
        return None

    scored = []
    for br in candidates:
        d = _branch_start_direction(br, nodes)
        a_mpv = _angle_deg(d, mpv_dir)
        scored.append((a_mpv if a_mpv is not None else 1e6, br))

    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def _select_sv_smv_pgv(branches, nodes, mpv_seg_from_lower=None):
    """
    Lower MPV endpoint selection.

    - Three terminal branches: SV is long and curved; the remaining lower
      extra branch is only a raw candidate here. Callers decide whether it is
      LGV/PGV from its anatomical attachment point.
    - Two terminal branches: use the same negative-Z SMV constraint, then
      fall back to the previous SV score when Z is ambiguous.
    """
    if not branches:
        return None, None, None
    if len(branches) == 1:
        return branches[0], None, None

    if len(branches) >= 3:
        sv_seg = _select_sv_for_trifurcation(branches, nodes)
        rest = [s for s in branches if s is not sv_seg]
        if len(rest) > 2:
            rest = sorted(rest,
                          key=lambda s: path_physical_length(s, nodes),
                          reverse=True)[:2]

        smv_seg = _select_smv_by_z(rest, nodes)
        pgv_seg = next((s for s in rest if s is not smv_seg), None)

        angle_pgv = _select_pgv_by_mpv_sv_angle(
            rest, nodes, mpv_seg=mpv_seg_from_lower, sv_seg=sv_seg)
        if angle_pgv is not None:
            angle_smv = next((s for s in rest if s is not angle_pgv), None)
            z_smv = _smv_z_dot(angle_smv, nodes) if angle_smv else 1.0
            z_pgv = _smv_z_dot(angle_pgv, nodes)
            smv_seg, pgv_seg = angle_smv, angle_pgv
            if z_smv > 0.25 and z_pgv <= 0.10:
                smv_seg, pgv_seg = angle_pgv, angle_smv

        print(
            "    lower trifurcation: "
            f"SV endpoint={sv_seg[-1] if sv_seg else None}, "
            f"SMV endpoint={smv_seg[-1] if smv_seg else None}, "
            f"extra endpoint={pgv_seg[-1] if pgv_seg else None}, "
            f"SMV dot(+Z)={_smv_z_dot(smv_seg, nodes):.3f}"
        )
        return sv_seg, smv_seg, pgv_seg

    s0, s1 = branches[0], branches[1]
    z0, z1 = _smv_z_dot(s0, nodes), _smv_z_dot(s1, nodes)
    z_gap = abs(z0 - z1)
    if (z0 <= 0.10 < z1) or (z_gap >= 0.25 and z0 < z1):
        return s1, s0, None
    if (z1 <= 0.10 < z0) or (z_gap >= 0.25 and z1 < z0):
        return s0, s1, None

    if _seg_score_sv(s0, nodes) >= _seg_score_sv(s1, nodes):
        return s0, s1, None
    return s1, s0, None


def _select_sv_smv(branches, nodes):
    sv_seg, smv_seg, _ = _select_sv_smv_pgv(branches, nodes)
    return sv_seg, smv_seg


def _enforce_smv_z_constraint(sv_seg, smv_seg, nodes, context=""):
    """
    Final label guard: SMV should point to negative Z from its branch point to
    endpoint, i.e. its angle with +Z should be >= about 90 degrees.
    """
    if sv_seg is None or smv_seg is None:
        return sv_seg, smv_seg
    sv_z = _smv_z_dot(sv_seg, nodes)
    smv_z = _smv_z_dot(smv_seg, nodes)
    if sv_z <= 0.10 and sv_z + 0.20 < smv_z:
        tag = f" ({context})" if context else ""
        print(
            f"    SMV Z约束校正{tag}: swap SV/SMV "
            f"(SV dot+Z={sv_z:.3f}, SMV dot+Z={smv_z:.3f})"
        )
        return smv_seg, sv_seg
    return sv_seg, smv_seg


def _assign_lpv_rpv(branches, nodes):
    """
    LPS 坐标系约定 (DICOM 默认): X 越大 -> patient's left -> LPV。
    若数据是 RAS 坐标系, 把 if x0 > x1 反一下即可。
    """
    if not branches:
        return None, None
    if len(branches) == 1:
        return branches[0], None

    if len(branches) > 2:
        branches = sorted(branches,
                          key=lambda s: path_physical_length(s, nodes),
                          reverse=True)[:2]

    s0, s1 = branches[0], branches[1]
    x0, x1 = nodes[s0[-1]]['x'], nodes[s1[-1]]['x']
    if x0 > x1:
        return s0, s1
    return s1, s0


def _assign_tips_lpv_rpv(branches, nodes):
    """Post-TIPS liver side: always choose TIPS first, even with only two branches."""
    tips_seg = lpv_seg = rpv_seg = None
    if not branches:
        return tips_seg, lpv_seg, rpv_seg

    if len(branches) == 1:
        return branches[0], None, None

    tips_seg = max(branches, key=lambda s: _seg_score_tips(s, nodes))
    print(f"    TIPS candidate scores (L*exp(-2.5*tortuosity)):")
    for br in sorted(branches,
                     key=lambda s: _seg_score_tips(s, nodes),
                     reverse=True):
        L = path_physical_length(br, nodes)
        t = _path_tortuosity(path_to_coords(br, nodes))
        sc = _seg_score_tips(br, nodes)
        tag = "  <- TIPS" if br is tips_seg else ""
        print(f"      root_bp={br[0]}, endpoint={br[-1]}, "
              f"L={L:6.1f}mm, tort={t:.3f}, score={sc:6.1f}{tag}")

    leftover = [s for s in branches if s is not tips_seg]
    lpv_seg, rpv_seg = _assign_lpv_rpv(leftover, nodes)
    return tips_seg, lpv_seg, rpv_seg


def _select_sv_distal_pgv(branches, sv_main_path, nodes, sample_dist=8.0):
    """
    在 bp_svsub 处区分 SV-distal 和 PGV (方向一致性)。
    SV 是连续血管, SV-distal 出射方向延续 SV-proximal 入射方向;
    PGV 从 SV 上分支出去, 方向有偏转。
    """
    if not branches:
        return None, None
    if len(branches) == 1:
        return branches[0], None

    sv_main_coords = path_to_coords(sv_main_path, nodes)
    incoming_dir = _direction_at_end(sv_main_coords, sample_dist)
    if incoming_dir is None:
        return _select_sv_smv(branches, nodes)

    scored = []
    for br in branches:
        out_dir = _direction_at_start(path_to_coords(br, nodes), sample_dist)
        score = float(np.dot(incoming_dir, out_dir)) if out_dir is not None else -2.0
        scored.append((score, br))

    scored.sort(key=lambda x: x[0], reverse=True)
    sv_distal = scored[0][1]
    pgv = scored[1][1]
    print(f"    SV/PGV 方向一致性: SV={scored[0][0]:.3f}, PGV={scored[1][0]:.3f}")
    return sv_distal, pgv


def _pgv_candidate_quality(pgv_seg, sv_distal_seg, sv_proximal,
                           smv_seg, nodes):
    """
    判断 PGV 候选是否像真实代偿支。

    这个门控只用于防止把 SV-SMV 汇合点附近的短小骨架支误标为 PGV。
    真实 PGV 通常应从 SV 远端分出, 与汇合点有一定距离, 且长度不能只像
    一个局部表面/端点伪分支。
    """
    if pgv_seg is None or len(pgv_seg) < 2:
        return False, "无 PGV 候选"

    pgv_len = path_physical_length(pgv_seg, nodes)
    sv_distal_len = path_physical_length(sv_distal_seg, nodes) if sv_distal_seg else 0.0
    sv_prox_len = path_physical_length(sv_proximal, nodes) if sv_proximal else 0.0
    smv_len = path_physical_length(smv_seg, nodes) if smv_seg else 0.0

    # PGV 分叉点离 SV-SMV 汇合太近时, 很容易是中心线拓扑毛刺或短端点。
    near_confluence = sv_prox_len < 8.0

    # 长度门限用相对值兜底, 避免固定阈值误杀整体较小的样本。
    reference_len = max(sv_distal_len, smv_len, 1.0)
    short_vs_system = pgv_len < max(10.0, 0.20 * reference_len)

    # 若 PGV 比它竞争的 SV 远端短太多, 且起点就在汇合附近, 大概率不是
    # 真实代偿血管, 而是本例图中这种被误分出的短支。
    tiny_vs_sv = sv_distal_len > 1e-6 and pgv_len < 0.35 * sv_distal_len

    if near_confluence and (short_vs_system or tiny_vs_sv):
        return False, (
            f"PGV 起点离 SV-SMV 汇合仅 {sv_prox_len:.1f}mm, "
            f"且候选较短 {pgv_len:.1f}mm "
            f"(SV远端 {sv_distal_len:.1f}mm, SMV {smv_len:.1f}mm)"
        )

    # 极短支即使不在汇合点附近, 也更像骨架毛刺。
    if pgv_len < 6.0:
        return False, f"PGV 候选过短 {pgv_len:.1f}mm"

    return True, (
        f"PGV 候选通过: L={pgv_len:.1f}mm, "
        f"距汇合={sv_prox_len:.1f}mm"
    )


# ============================================================
# MPV 终点扩展 + 段裁剪
# ============================================================

def _find_mpv_end_by_liver_branches(adj, nodes, bp_sv_init, lpv_seg, rpv_seg,
                                     bp_liver_init):
    """
    确定 MPV 真正终点。
    定义: LPV/RPV 起点中, 沿中心线距 SV 端 (bp_sv_init) 弧长更远的那个 bp。

    解剖学语义:
        肝门是 MPV 主干自然分叉为左右肝静脉的位置。若 LPV 早早从主干分出
        (常见解剖变异), RPV 在更深处分出, 则 MPV 应延伸到 RPV 起点,
        中间过渡段并入 MPV。TIPS 不参与判定 (人工分流不属于自然血管树)。
    """
    candidates = []
    if lpv_seg is not None and len(lpv_seg) >= 1:
        candidates.append(lpv_seg[0])
    if rpv_seg is not None and len(rpv_seg) >= 1:
        candidates.append(rpv_seg[0])

    if not candidates:
        return bp_liver_init
    if len(candidates) == 1:
        return candidates[0]

    def _arc_dist(bp):
        path = find_path(adj, bp_sv_init, bp)
        if path is None:
            return -1.0
        return path_physical_length(path, nodes)

    # Clinical boundary: MPV ends at the first natural left/right portal
    # division. If the right portal trunk continues farther before a
    # TIPS/right-branch split, that downstream trunk belongs to RPV.
    return min(candidates, key=_arc_dist)


def _trim_branch_to_subbp(branch_seg, mpv_end_bp, branch_points):
    """段起点 == MPV 终点时, 沿段找下一个 bp 作为新起点。"""
    if branch_seg is None or len(branch_seg) < 2:
        return branch_seg
    if branch_seg[0] != mpv_end_bp:
        return branch_seg
    for i in range(1, len(branch_seg)):
        nid = branch_seg[i]
        if nid in branch_points and nid != mpv_end_bp:
            return branch_seg[i:]
    return branch_seg


def _trim_branches_to_mpv_end(branch_segs_dict, bp_mpv_end,
                               adj, branch_points, nodes):
    """
    把字典中所有段裁剪到"MPV 终点之后"开始。
      - 起点 == bp_mpv_end: 沿段找下一个 bp, 从该 bp 开始
      - 起点 != bp_mpv_end: 段不动 (从 MPV 干道中部分出, 几何路径正确)
    """
    trimmed = {}
    for name, seg in branch_segs_dict.items():
        if seg is None or len(seg) < 2:
            trimmed[name] = seg
            continue
        if name not in {"lpv", "rpv"} or seg[0] == bp_mpv_end:
            trimmed[name] = seg
            continue
        connector = find_path(adj, bp_mpv_end, seg[0])
        if connector is not None and len(connector) >= 2:
            trimmed[name] = connector + seg[1:]
        else:
            trimmed[name] = seg
    return trimmed


# ============================================================
# 三 bp 工具 (术前用)
# ============================================================

def _order_bp_chain(bps, adj):
    """将 3 个分支点排序: bp_end - bp_mid - bp_end。"""
    if len(bps) != 3:
        raise ValueError(f"期望 3 个分支点, 得到 {len(bps)}")
    a, b, c = bps
    p_ab = find_path(adj, a, b)
    p_ac = find_path(adj, a, c)
    if c in p_ab:
        return a, c, b
    elif b in p_ac:
        return a, b, c
    else:
        return b, a, c


# ============================================================
# 主入口
# ============================================================

def segment_vessels(stl_path, post_tips=None, output_json_path=None,
                    lgv_pgv_tortuosity_threshold=0.05,
                    coordinate_system="LPS"):
    """对中心线进行解剖分段并输出 JSON。"""
    print(f"  segment_vessels version: {SEGMENT_VESSELS_VERSION}")
    nodes, adj, parentdir = load_tree(stl_path)
    folder_name = os.path.basename(parentdir)
    if post_tips is None:
        post_tips = is_post_tips(folder_name)

    endpoints, branch_points = classify_nodes(nodes, adj)
    print(f"  节点统计: 端点 {len(endpoints)}, 分支点 {len(branch_points)}")
    print(f"  类型: {'TIPS术后' if post_tips else 'TIPS术前'}")

    segments_raw = _extract_all_segments(nodes, adj, endpoints, branch_points)

    result = segment_anatomically(
        nodes, adj, endpoints, branch_points,
        post_tips=post_tips,
        coordinate_system=coordinate_system,
        stl_path=stl_path,
    )

    output = _build_output_json(folder_name, post_tips, result,
                                nodes, branch_points, endpoints)

    if output_json_path is None:
        output_json_path = os.path.join(parentdir, "centerline_profiles.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    seg_names = [n for n, v in output['segments'].items() if v is not None]
    print(f"  识别血管: {seg_names}")
    if output['has_compensation']:
        print(f"  代偿类型: {output['compensation_type']}")
    confidence = output.get('confidence', {})
    if confidence:
        print(
            f"  解剖一致性: {confidence.get('level')} "
            f"(score={confidence.get('score', 0):.3f}, "
            f"margin={confidence.get('margin_to_second', 0):.3f})"
        )
        if confidence.get('needs_manual_review'):
            print(f"  [REVIEW] {confidence.get('review_reasons', [])}")
    print(f"  分段结果已保存: {output_json_path}")
    return output


# ============================================================
# 术后 (post-TIPS)
# ============================================================

def _dedupe_v2_candidates(candidates):
    out = []
    seen = set()
    for item in candidates:
        key = (
            item.get("source"),
            item.get("branch_bp"),
            tuple(item.get("path") or []),
            tuple(item.get("branch") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _select_post_lower_sv_smv(nodes, adj, endpoints, segments_raw,
                              sv_subtree, bp_lower, mpv_from_lower):
    """
    Post-TIPS lower side can contain an extra branch point along the splenic
    side. In that topology, root_branches alone misses the distal SV path.
    Build complete lower-to-visceral-to-endpoint candidates before assigning.
    """
    deeper_bps = sorted(bp for bp in sv_subtree.get('visited_bps', set())
                        if bp != bp_lower)
    candidates = []
    for bp_visceral in deeper_bps:
        candidates.extend(_v2_make_lower_candidates(
            adj, nodes, segments_raw, endpoints, bp_lower, bp_visceral))
        candidates.extend(_v2_make_terminal_candidates(
            segments_raw, endpoints, bp_visceral, "terminal_visceral"))
    candidates = _dedupe_v2_candidates(candidates)

    if len(candidates) >= 2:
        print(
            f"    Post lower V2 candidates: deeper_bps={deeper_bps}, "
            f"n={len(candidates)}"
        )
        smv_candidate = _v2_select_smv_candidate(candidates, nodes)
        sv_candidate = (
            _v2_select_sv_candidate(candidates, smv_candidate, nodes)
            if smv_candidate is not None else None)
        if smv_candidate is not None and sv_candidate is not None:
            confluence = smv_candidate["branch_bp"]
            smv_seg = _v2_terminal_branch_from_anchor(smv_candidate, confluence)
            sv_seg = _v2_path_from_anchor(sv_candidate["path"], confluence)
            if sv_seg and smv_seg:
                print(
                    f"    Post lower V2 labels: SV endpoint={sv_seg[-1]}, "
                    f"SMV endpoint={smv_seg[-1]}, confluence={confluence}"
                )
                return sv_seg, smv_seg, confluence

    sv_brs = sv_subtree['root_branches'] or sv_subtree['all_branches']
    sv_seg, smv_seg, _ = _select_sv_smv_pgv(
        sv_brs, nodes, mpv_seg_from_lower=mpv_from_lower)
    return sv_seg, smv_seg, bp_lower


def _segment_post_tips(nodes, adj, endpoints, branch_points, segments_raw):
    """TIPS 术后分段。"""
    if len(branch_points) < 2:
        raise ValueError(f"TIPS术后期望 ≥2 分支点, 实际 {len(branch_points)}")

    # ----- 1. MPV 初始候选 -----
    bp_bp_segs = _find_bp_to_bp_segments(segments_raw, branch_points)
    if not bp_bp_segs:
        raise ValueError("找不到 MPV (两端均为分支点的段)")

    mpv_init_seg = max(bp_bp_segs, key=lambda s: _mpv_init_score(s, nodes))
    bp_init_a, bp_init_b = mpv_init_seg[0], mpv_init_seg[-1]

    # ----- 2. 双侧子树 -----
    sub_a = _collect_subtree(adj, bp_init_a, mpv_init_seg[1],
                             endpoints, branch_points)
    sub_b = _collect_subtree(adj, bp_init_b, mpv_init_seg[-2],
                             endpoints, branch_points)

    # ----- 3. SV 端 vs 肝侧端 -----
    score_a = max((_seg_score_sv(s, nodes) for s in sub_a['all_branches']),
                  default=0)
    score_b = max((_seg_score_sv(s, nodes) for s in sub_b['all_branches']),
                  default=0)

    if score_a >= score_b:
        sv_subtree, liver_subtree = sub_a, sub_b
        bp_sv_init, bp_liver_init = bp_init_a, bp_init_b
    else:
        sv_subtree, liver_subtree = sub_b, sub_a
        bp_sv_init, bp_liver_init = bp_init_b, bp_init_a

    print(f"    SV-score: a={score_a:.1f}, b={score_b:.1f}")
    print(f"    肝侧子树: root_brs={len(liver_subtree['root_branches'])}, "
          f"deeper_brs={len(liver_subtree['deeper_branches'])}, "
          f"all={len(liver_subtree['all_branches'])}")

    # ----- 4. SV / SMV -----
    mpv_from_sv = _orient_path_from(mpv_init_seg, bp_sv_init)
    sv_seg, smv_seg, bp_svjct = _select_post_lower_sv_smv(
        nodes, adj, endpoints, segments_raw, sv_subtree, bp_sv_init, mpv_from_sv)
    # Post-TIPS cases must not expose compensation vessels.
    pgv_seg = None
    sv_seg, smv_seg = _enforce_smv_z_constraint(
        sv_seg, smv_seg, nodes, context="post lower")

    # ----- 5. TIPS / LPV / RPV (术后必须先识别 TIPS) -----
    all_liver_brs = liver_subtree['all_branches']
    tips_seg, lpv_seg, rpv_seg = _assign_tips_lpv_rpv(all_liver_brs, nodes)

    # ----- 6. MPV 终点 = LPV/RPV 中更靠肝侧者 (TIPS 不参与) -----
    bp_mpv_end = _find_mpv_end_by_liver_branches(
        adj, nodes, bp_sv_init, lpv_seg, rpv_seg, bp_liver_init)

    # ----- 7. 子分支起点裁剪 -----
    trimmed = _trim_branches_to_mpv_end(
        {'tips': tips_seg, 'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end, adj, branch_points, nodes)
    tips_seg = trimmed['tips']
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    # ----- 8. 最终 MPV -----
    mpv_seg = find_path(adj, bp_sv_init, bp_mpv_end)

    L_init = path_physical_length(mpv_init_seg, nodes)
    L_final = path_physical_length(mpv_seg, nodes)
    print(f"    MPV: 起点={bp_svjct}, 终点={bp_mpv_end}")
    print(f"         长度 {L_init:.1f}mm -> {L_final:.1f}mm "
          f"(扩展 +{L_final - L_init:.1f}mm)")
    print(f"    分支起点: TIPS={tips_seg[0] if tips_seg else None}, "
          f"LPV={lpv_seg[0] if lpv_seg else None}, "
          f"RPV={rpv_seg[0] if rpv_seg else None}")

    return {
        'segments': {
            'mpv': mpv_seg,
            'sv': sv_seg, 'smv': smv_seg,
            'pgv': pgv_seg,
            'tips': tips_seg,
            'lpv': lpv_seg, 'rpv': rpv_seg,
        },
        'has_compensation': False,
        'compensation_type': None,
    }


# ============================================================
# 术前 (pre-TIPS)
# ============================================================

def _segment_pre_tips(nodes, adj, endpoints, branch_points,
                      segments_raw, lgv_pgv_threshold):
    """术前分段路由。"""
    n_bp = len(branch_points)

    if n_bp == 2:
        return _segment_pre_tips_no_comp(
            nodes, adj, endpoints, branch_points, segments_raw)
    elif n_bp == 3:
        return _segment_pre_tips_with_comp(
            nodes, adj, endpoints, branch_points, segments_raw,
            lgv_pgv_threshold)
    elif n_bp > 3:
        print(f"  警告: 分支点数={n_bp} > 3, 退化处理")
        return _segment_pre_tips_fallback(
            nodes, adj, endpoints, branch_points, segments_raw,
            lgv_pgv_threshold)
    else:
        raise ValueError(f"TIPS术前期望 ≥2 分支点, 实际 {n_bp}")


def _segment_pre_tips_no_comp(nodes, adj, endpoints, branch_points,
                              segments_raw):
    """术前无代偿: 仅 MPV/SV/SMV/LPV/RPV. 含 MPV 终点扩展。"""
    bp_bp_segs = _find_bp_to_bp_segments(segments_raw, branch_points)
    if not bp_bp_segs:
        raise ValueError("找不到 MPV")

    mpv_init_seg = max(bp_bp_segs, key=lambda s: _mpv_init_score(s, nodes))
    bp_init_a, bp_init_b = mpv_init_seg[0], mpv_init_seg[-1]

    sub_a = _collect_subtree(adj, bp_init_a, mpv_init_seg[1],
                             endpoints, branch_points)
    sub_b = _collect_subtree(adj, bp_init_b, mpv_init_seg[-2],
                             endpoints, branch_points)

    score_a = max((_seg_score_sv(s, nodes) for s in sub_a['all_branches']),
                  default=0)
    score_b = max((_seg_score_sv(s, nodes) for s in sub_b['all_branches']),
                  default=0)

    if score_a >= score_b:
        sv_subtree, liver_subtree = sub_a, sub_b
        bp_sv_init, bp_liver_init = bp_init_a, bp_init_b
    else:
        sv_subtree, liver_subtree = sub_b, sub_a
        bp_sv_init, bp_liver_init = bp_init_b, bp_init_a

    sv_brs = sv_subtree['root_branches'] or sv_subtree['all_branches']
    mpv_from_sv = _orient_path_from(mpv_init_seg, bp_sv_init)
    sv_seg, smv_seg, pgv_seg = _select_sv_smv_pgv(
        sv_brs, nodes, mpv_seg_from_lower=mpv_from_sv)
    # A third branch directly at the MPV/SV/SMV confluence is LGV, not PGV.
    # PGV is reserved for branches that attach to the SV mid/distal segment.
    lgv_seg = pgv_seg
    pgv_seg = None
    sv_seg, smv_seg = _enforce_smv_z_constraint(
        sv_seg, smv_seg, nodes, context="pre lower")

    liver_brs = liver_subtree['all_branches']
    tips_seg = None
    if len(liver_brs) >= 3:
        tips_seg, lpv_seg, rpv_seg = _assign_tips_lpv_rpv(liver_brs, nodes)
    else:
        lpv_seg, rpv_seg = _assign_lpv_rpv(liver_brs, nodes)

    bp_mpv_end = _find_mpv_end_by_liver_branches(
        adj, nodes, bp_svjct, lpv_seg, rpv_seg, bp_liver_init)
    trimmed = _trim_branches_to_mpv_end(
        {'tips': tips_seg, 'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end, adj, branch_points, nodes)
    tips_seg = trimmed['tips']
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    mpv_seg = find_path(adj, bp_svjct, bp_mpv_end)
    _check_sv_smv_mpv_confluence(
        mpv_seg, sv_seg, smv_seg, "post lower confluence")

    L_init = path_physical_length(mpv_init_seg, nodes)
    L_final = path_physical_length(mpv_seg, nodes)
    print(f"    MPV: 起点={bp_sv_init}, 终点={bp_mpv_end}, "
          f"长度 {L_init:.1f}->{L_final:.1f}mm")
    print(f"    分支起点: LPV={lpv_seg[0] if lpv_seg else None}, "
          f"RPV={rpv_seg[0] if rpv_seg else None}")
    if tips_seg is not None:
        print(f"    上端三分叉: TIPS={tips_seg[0]} -> {tips_seg[-1]}")

    return {
        'segments': {
            'mpv': mpv_seg, 'sv': sv_seg, 'smv': smv_seg,
            'lgv': lgv_seg, 'pgv': pgv_seg,
            'tips': tips_seg,
            'lpv': lpv_seg, 'rpv': rpv_seg,
        },
        'has_compensation': lgv_seg is not None,
        'compensation_type': 'LGV' if lgv_seg is not None else None,
    }


def _segment_pre_tips_with_comp(nodes, adj, endpoints, branch_points,
                                segments_raw, lgv_pgv_threshold):
    """术前 3 个分支点: 区分 LGV / PGV。"""
    bps = list(branch_points)
    bp1, bp2, bp3 = _order_bp_chain(bps, adj)

    path_13 = find_path(adj, bp1, bp3)
    coords_13 = path_to_coords(path_13, nodes)
    tort_13 = _path_tortuosity(coords_13)

    print(f"  3 分支点链: {bp1}-{bp2}-{bp3}")
    print(f"  bp1<->bp3 tortuosity = {tort_13:.4f} (阈值 {lgv_pgv_threshold})")

    if tort_13 < lgv_pgv_threshold:
        print(f"  -> 判定: LGV 代偿 (路径较直, MPV 贯穿 bp1->bp3)")
        return _build_lgv_segments(
            nodes, adj, endpoints, segments_raw, bp1, bp2, bp3, branch_points)
    else:
        print(f"  -> 判定: PGV 代偿 (路径有转折, MPV = bp1->bp2)")
        return _build_pgv_segments(
            nodes, adj, endpoints, segments_raw, bp1, bp2, bp3, branch_points)


def _build_lgv_segments(nodes, adj, endpoints, segments_raw,
                        bp1, bp2, bp3, branch_points):
    """LGV 代偿: MPV 贯穿 bp1-bp2-bp3, LGV 从 bp2 分出。"""
    branches_1 = _find_endpoint_branches_at(segments_raw, bp1, endpoints)
    branches_3 = _find_endpoint_branches_at(segments_raw, bp3, endpoints)

    score_1 = max((_seg_score_sv(s, nodes) for s in branches_1), default=0)
    score_3 = max((_seg_score_sv(s, nodes) for s in branches_3), default=0)

    if score_3 >= score_1:
        sv_branches, liver_branches_direct = branches_3, branches_1
        bp_svjct, bp_liver = bp3, bp1
    else:
        sv_branches, liver_branches_direct = branches_1, branches_3
        bp_svjct, bp_liver = bp1, bp3

    L_to_svjct = path_physical_length(find_path(adj, bp2, bp_svjct), nodes)
    L_to_liver = path_physical_length(find_path(adj, bp2, bp_liver), nodes)
    print(f"    LGV分叉点位置: 到SV端={L_to_svjct:.1f}mm, "
          f"到肝侧={L_to_liver:.1f}mm")

    sv_seg, smv_seg = _select_sv_smv(sv_branches, nodes)
    sv_seg, smv_seg = _enforce_smv_z_constraint(
        sv_seg, smv_seg, nodes, context="LGV")

    # 收集肝侧子树以处理嵌套 LPV/RPV
    path_liver_to_svjct = find_path(adj, bp_liver, bp_svjct)
    excl_nb = path_liver_to_svjct[1] if len(path_liver_to_svjct) >= 2 else None
    liver_subtree = _collect_subtree(
        adj, bp_liver, excl_nb, endpoints, branch_points)

    liver_brs_all = liver_subtree['all_branches'] or liver_branches_direct
    tips_seg = None
    if len(liver_brs_all) >= 3:
        tips_seg, lpv_seg, rpv_seg = _assign_tips_lpv_rpv(liver_brs_all, nodes)
    else:
        lpv_seg, rpv_seg = _assign_lpv_rpv(liver_brs_all, nodes)

    # MPV 终点扩展 (肝侧)
    bp_mpv_end_liver = _find_mpv_end_by_liver_branches(
        adj, nodes, bp_svjct, lpv_seg, rpv_seg, bp_liver)
    trimmed = _trim_branches_to_mpv_end(
        {'tips': tips_seg, 'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end_liver, adj, branch_points, nodes)
    tips_seg = trimmed['tips']
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    # MPV = bp_mpv_end_liver -> bp_svjct, 注意 bp2 仍在路径中, LGV 仍从 bp2 分出
    mpv_seg = find_path(adj, bp_mpv_end_liver, bp_svjct)

    lgv_branches = _find_endpoint_branches_at(segments_raw, bp2, endpoints)
    lgv_seg = lgv_branches[0] if lgv_branches else None

    print(f"    MPV: 肝侧={bp_mpv_end_liver} -> SV交汇={bp_svjct}")
    print(f"    分支起点: LPV={lpv_seg[0] if lpv_seg else None}, "
          f"RPV={rpv_seg[0] if rpv_seg else None}")
    if tips_seg is not None:
        print(f"    上端三分叉: TIPS={tips_seg[0]} -> {tips_seg[-1]}")

    return {
        'segments': {
            'mpv': mpv_seg, 'sv': sv_seg, 'smv': smv_seg,
            'tips': tips_seg,
            'lpv': lpv_seg, 'rpv': rpv_seg, 'lgv': lgv_seg,
        },
        'has_compensation': True,
        'compensation_type': 'LGV',
    }


def _build_pgv_segments(nodes, adj, endpoints, segments_raw,
                        bp1, bp2, bp3, branch_points):
    """PGV 代偿: bp_liver - MPV - bp_svjct - SV-prox - bp_svsub - SV-distal/PGV."""
    branches_1 = _find_endpoint_branches_at(segments_raw, bp1, endpoints)
    branches_3 = _find_endpoint_branches_at(segments_raw, bp3, endpoints)

    score_1 = max((_seg_score_sv(s, nodes) for s in branches_1), default=0)
    score_3 = max((_seg_score_sv(s, nodes) for s in branches_3), default=0)

    if score_3 >= score_1:
        bp_liver, bp_svsub = bp1, bp3
        liver_branches_direct, svsub_branches = branches_1, branches_3
    else:
        bp_liver, bp_svsub = bp3, bp1
        liver_branches_direct, svsub_branches = branches_3, branches_1

    bp_svjct = bp2
    print(f"    PGV拓扑: 肝侧={bp_liver}, MPV/SV交汇={bp_svjct}, "
          f"SV远端bp={bp_svsub}")

    # 1) LPV / RPV: 子树扫描以处理嵌套
    path_liver_to_svjct = find_path(adj, bp_liver, bp_svjct)
    excl_nb = path_liver_to_svjct[1] if len(path_liver_to_svjct) >= 2 else None
    liver_subtree = _collect_subtree(
        adj, bp_liver, excl_nb, endpoints, branch_points)

    liver_brs_all = liver_subtree['all_branches'] or liver_branches_direct
    tips_seg = None
    if len(liver_brs_all) >= 3:
        tips_seg, lpv_seg, rpv_seg = _assign_tips_lpv_rpv(liver_brs_all, nodes)
    else:
        lpv_seg, rpv_seg = _assign_lpv_rpv(liver_brs_all, nodes)

    # 2) SMV 在 bp_svjct
    smv_branches = _find_endpoint_branches_at(segments_raw, bp_svjct, endpoints)
    smv_seg = _select_smv_by_z(smv_branches, nodes) if smv_branches else None

    # 3) SV-distal vs PGV: 方向一致性
    sv_main_path = find_path(adj, bp_svjct, bp_svsub)
    sv_distal_seg, pgv_seg = _select_sv_distal_pgv(
        svsub_branches, sv_main_path, nodes)

    # 4) MPV 终点扩展 (肝侧) + LPV/RPV 裁剪
    bp_mpv_end_liver = _find_mpv_end_by_liver_branches(
        adj, nodes, bp_svjct, lpv_seg, rpv_seg, bp_liver)
    trimmed = _trim_branches_to_mpv_end(
        {'tips': tips_seg, 'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end_liver, adj, branch_points, nodes)
    tips_seg = trimmed['tips']
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    # 5) MPV = bp_mpv_end_liver -> bp_svjct
    mpv_seg = find_path(adj, bp_mpv_end_liver, bp_svjct)

    # 6) SV = bp_svjct -> bp_svsub + SV-distal
    sv_proximal = find_path(adj, bp_svjct, bp_svsub)
    if sv_distal_seg is not None:
        sv_seg = sv_proximal + sv_distal_seg[1:]
    else:
        sv_seg = sv_proximal
    sv_seg, smv_seg = _enforce_smv_z_constraint(
        sv_seg, smv_seg, nodes, context="PGV")

    pgv_ok, pgv_reason = _pgv_candidate_quality(
        pgv_seg, sv_distal_seg, sv_proximal, smv_seg, nodes)
    if pgv_ok:
        print(f"    PGV质控: {pgv_reason}")
    else:
        print(f"    PGV质控: {pgv_reason} -> 降级为无代偿分段")
        pgv_seg = None

    print(f"    MPV: 肝侧={bp_mpv_end_liver} -> SV交汇={bp_svjct}")
    print(f"    分支起点: LPV={lpv_seg[0] if lpv_seg else None}, "
          f"RPV={rpv_seg[0] if rpv_seg else None}")
    if tips_seg is not None:
        print(f"    上端三分叉: TIPS={tips_seg[0]} -> {tips_seg[-1]}")

    return {
        'segments': {
            'mpv': mpv_seg, 'sv': sv_seg, 'smv': smv_seg,
            'tips': tips_seg,
            'lpv': lpv_seg, 'rpv': rpv_seg, 'pgv': pgv_seg,
        },
        'has_compensation': bool(pgv_ok),
        'compensation_type': 'PGV' if pgv_ok else None,
    }


def _segment_pre_tips_fallback(nodes, adj, endpoints, branch_points,
                               segments_raw, lgv_pgv_threshold):
    """>3 分支点的退化处理。"""
    sorted_bps = sorted(branch_points,
                        key=lambda b: len(adj[b]), reverse=True)
    chosen = sorted_bps[:3] if len(sorted_bps) >= 3 else list(branch_points)
    if len(chosen) < 3:
        return _segment_pre_tips_no_comp(
            nodes, adj, endpoints, set(chosen), segments_raw)
    return _segment_pre_tips_with_comp(
        nodes, adj, endpoints, set(chosen), segments_raw, lgv_pgv_threshold)


def _v2_candidate_z_dot(path, nodes):
    d = _branch_endpoint_direction(path, nodes)
    if d is None:
        return 1.0
    return float(d[2])


def _v2_candidate_sv_score(path, nodes, source):
    score = _seg_score_sv(path, nodes)
    z = _v2_candidate_z_dot(path, nodes)
    if z <= 0.10:
        score *= 0.35
    if source == "direct_lower":
        score *= 0.60
    elif source == "via_visceral":
        score *= 1.25
    return score


def _v2_point_at_path_distance(path, nodes, start_idx, direction, target_mm=6.0):
    if not path:
        return None
    idx = int(start_idx)
    idx = max(0, min(len(path) - 1, idx))
    prev = path[idx]
    dist = 0.0
    while 0 <= idx + direction < len(path) and dist < target_mm:
        nxt_idx = idx + direction
        nxt = path[nxt_idx]
        p0 = np.array([nodes[prev]['x'], nodes[prev]['y'], nodes[prev]['z']], dtype=float)
        p1 = np.array([nodes[nxt]['x'], nodes[nxt]['y'], nodes[nxt]['z']], dtype=float)
        dist += float(np.linalg.norm(p1 - p0))
        idx = nxt_idx
        prev = nxt
    return np.array([nodes[path[idx]]['x'], nodes[path[idx]]['y'], nodes[path[idx]]['z']], dtype=float)


def _v2_branch_turn_continuity(path, branch_bp, nodes, target_mm=6.0):
    """
    SV should continue smoothly through an SV/LGV side-branch junction.
    Returns tangent dot product at the branch point: 1 is straight, -1 is a U-turn.
    """
    path = list(path or [])
    if branch_bp not in path:
        return 0.0
    idx = path.index(branch_bp)
    if idx <= 0 or idx >= len(path) - 1:
        return 0.0
    center = np.array([nodes[branch_bp]['x'], nodes[branch_bp]['y'], nodes[branch_bp]['z']], dtype=float)
    before = _v2_point_at_path_distance(path, nodes, idx, -1, target_mm=target_mm)
    after = _v2_point_at_path_distance(path, nodes, idx, 1, target_mm=target_mm)
    if before is None or after is None:
        return 0.0
    incoming = center - before
    outgoing = after - center
    ni = float(np.linalg.norm(incoming))
    no = float(np.linalg.norm(outgoing))
    if ni <= 1e-9 or no <= 1e-9:
        return 0.0
    return float(np.dot(incoming / ni, outgoing / no))


def _v2_make_lower_candidates(adj, nodes, segments_raw, endpoints,
                              bp_lower, bp_visceral):
    candidates = []
    for br in _find_endpoint_branches_at(segments_raw, bp_lower, endpoints):
        candidates.append({
            "path": br,
            "source": "direct_lower",
            "branch": br,
            "branch_bp": bp_lower,
        })

    lower_to_visceral = find_path(adj, bp_lower, bp_visceral)
    if lower_to_visceral:
        for br in _find_endpoint_branches_at(segments_raw, bp_visceral, endpoints):
            candidates.append({
                "path": lower_to_visceral + br[1:],
                "source": "via_visceral",
                "branch": br,
                "branch_bp": bp_visceral,
            })
    return candidates


def _v2_make_terminal_candidates(segments_raw, endpoints, bp, source):
    return [
        {
            "path": br,
            "source": source,
            "branch": br,
            "branch_bp": bp,
        }
        for br in _find_endpoint_branches_at(segments_raw, bp, endpoints)
    ]


def _v2_select_smv_candidate(candidates, nodes):
    terminal_candidates = [
        c for c in candidates
        if c["source"] in ("direct_lower", "terminal_visceral")
    ]
    if not terminal_candidates:
        return None
    print("    V2 direct SMV candidates:")
    for c in sorted(terminal_candidates, key=lambda item: _v2_candidate_z_dot(item["path"], nodes)):
        p = c["path"]
        print(
            f"      source={c['source']}, endpoint={p[-1]}, "
            f"L={path_physical_length(p, nodes):.1f}mm, "
            f"dot+Z={_v2_candidate_z_dot(p, nodes):.3f}"
        )
    return min(
        terminal_candidates,
        key=lambda c: (_v2_candidate_z_dot(c["path"], nodes),
                       -path_physical_length(c["path"], nodes))
    )


def _v2_select_sv_candidate(candidates, smv_candidate, nodes):
    rest = [
        c for c in candidates
        if c is not smv_candidate
        and not _v2_same_path(c.get("branch"), smv_candidate.get("branch"))
    ]
    if not rest:
        return None
    sv_side = [c for c in rest if c["source"] == "via_visceral"]
    if sv_side:
        rest = sv_side
        print("    V2 SV continuity candidates:")
        for c in sorted(rest, key=lambda item: _v2_branch_turn_continuity(
                item["path"], item["branch_bp"], nodes), reverse=True):
            continuity = _v2_branch_turn_continuity(c["path"], c["branch_bp"], nodes)
            p = c["path"]
            print(
                f"      endpoint={p[-1]}, "
                f"turn_dot={continuity:.3f}, "
                f"L={path_physical_length(p, nodes):.1f}mm, "
                f"sv_score={_v2_candidate_sv_score(p, nodes, c['source']):.3f}"
            )
        return max(
            rest,
            key=lambda c: (
                _v2_branch_turn_continuity(c["path"], c["branch_bp"], nodes),
                _v2_candidate_sv_score(c["path"], nodes, c["source"]),
            )
        )
    return max(
        rest,
        key=lambda c: _v2_candidate_sv_score(c["path"], nodes, c["source"])
    )


def _v2_same_path(a, b):
    return a is not None and b is not None and list(a) == list(b)


def _v2_path_from_anchor(path, anchor):
    """Orient/slice a candidate path so it starts at the SV/SMV confluence."""
    path = list(path or [])
    if not path:
        return path
    if anchor not in path:
        return path
    idx = path.index(anchor)
    if idx < len(path) - 1:
        return path[idx:]
    return list(reversed(path[:idx + 1]))


def _v2_terminal_branch_from_anchor(candidate, anchor):
    """SMV must be a terminal branch from confluence to endpoint."""
    branch = list(candidate.get("branch") or [])
    if branch and anchor in branch:
        return _v2_path_from_anchor(branch, anchor)
    return _v2_path_from_anchor(candidate.get("path") or [], anchor)


def _v2_pick_extra_branch(branches, used_branch):
    extras = [br for br in branches if not _v2_same_path(br, used_branch)]
    if not extras:
        return None
    return max(extras, key=len)


def _v2_pick_unused_terminal_path(candidates, used_candidates, nodes):
    candidate = _v2_pick_unused_terminal_candidate(candidates, used_candidates, nodes)
    return candidate["path"] if candidate is not None else None


def _v2_pick_unused_terminal_candidate(candidates, used_candidates, nodes):
    used_branches = [c.get("branch") for c in used_candidates if c is not None]
    options = []
    for c in candidates:
        if c["source"] not in ("direct_lower", "terminal_visceral"):
            continue
        if any(_v2_same_path(c.get("branch"), used) for used in used_branches):
            continue
        options.append(c)
    if not options:
        return None
    return max(options, key=lambda c: path_physical_length(c["path"], nodes))


def _v2_assign_compensation_branch(extra_candidate, sv_seg, sv_smv_bp):
    """
    PGV attaches to the SV mid/distal path.
    LGV attaches to MPV/lower port rather than to the SV midline.
    """
    if extra_candidate is None:
        return None, None, None
    extra_seg = extra_candidate["path"]
    attach_bp = extra_candidate.get("branch_bp")
    sv_mid_nodes = set(sv_seg[1:-1]) if sv_seg and len(sv_seg) > 2 else set()
    if attach_bp in sv_mid_nodes and attach_bp != sv_smv_bp:
        return None, extra_seg, "PGV"
    return extra_seg, None, "LGV"


def _segment_pre_tips_with_comp(nodes, adj, endpoints, branch_points,
                                segments_raw, lgv_pgv_threshold):
    """
    V2 anatomy-anchored compensation logic.

    Decision order:
      1. Keep the MPV bp chain order and identify the liver end by low SV score.
      2. At the lower MPV port, select SMV first by negative-Z direction.
      3. Select SV only from the remaining non-SMV candidates.
      4. A side branch is PGV only if it is attached to the selected SV path.
         If it attaches to MPV/lower port/SMV-side topology, label it LGV.
    """
    bps = list(branch_points)
    bp1, bp2, bp3 = _order_bp_chain(bps, adj)
    branches_1 = _find_endpoint_branches_at(segments_raw, bp1, endpoints)
    branches_3 = _find_endpoint_branches_at(segments_raw, bp3, endpoints)
    score_1 = max((_seg_score_sv(s, nodes) for s in branches_1), default=0)
    score_3 = max((_seg_score_sv(s, nodes) for s in branches_3), default=0)

    if score_1 <= score_3:
        bp_liver, bp_visceral = bp1, bp3
        liver_branches_direct, visceral_branches = branches_1, branches_3
    else:
        bp_liver, bp_visceral = bp3, bp1
        liver_branches_direct, visceral_branches = branches_3, branches_1
    bp_lower = bp2

    print(f"  V2 3-bp anatomy chain: liver={bp_liver}, lower={bp_lower}, visceral={bp_visceral}")

    candidates = _v2_make_lower_candidates(
        adj, nodes, segments_raw, endpoints, bp_lower, bp_visceral)
    candidates.extend(_v2_make_terminal_candidates(
        segments_raw, endpoints, bp_visceral, "terminal_visceral"))
    smv_candidate = _v2_select_smv_candidate(candidates, nodes)
    sv_candidate = _v2_select_sv_candidate(candidates, smv_candidate, nodes)
    if smv_candidate is None or sv_candidate is None:
        print("  V2 fallback: insufficient lower candidates, using conservative compensation logic")
        return _build_lgv_segments(
            nodes, adj, endpoints, segments_raw, bp1, bp2, bp3, branch_points)

    # The SV/SMV confluence is the branch point of the terminal SMV segment.
    # SV may pass through an extra side-branch point, but SMV must not.
    sv_smv_bp = smv_candidate["branch_bp"]

    smv_seg = _v2_terminal_branch_from_anchor(smv_candidate, sv_smv_bp)
    sv_seg = _v2_path_from_anchor(sv_candidate["path"], sv_smv_bp)
    sv_z = _v2_candidate_z_dot(sv_seg, nodes)
    smv_z = _v2_candidate_z_dot(smv_seg, nodes)
    print(
        f"    V2 lower labels: SV endpoint={sv_seg[-1]}, "
        f"SMV endpoint={smv_seg[-1]}, "
        f"confluence={sv_smv_bp}, "
        f"SV dot+Z={sv_z:.3f}, SMV dot+Z={smv_z:.3f}"
    )

    extra_candidate = _v2_pick_unused_terminal_candidate(
        candidates, [sv_candidate, smv_candidate], nodes)
    lgv_seg, pgv_seg, comp_type = _v2_assign_compensation_branch(
        extra_candidate, sv_seg, sv_smv_bp)
    print(
        "    V2 compensation: "
        f"attach_bp={extra_candidate.get('branch_bp') if extra_candidate else None}, "
        f"label={comp_type or 'None'}"
    )

    path_liver_to_lower = find_path(adj, bp_liver, bp_lower)
    excl_nb = path_liver_to_lower[1] if path_liver_to_lower and len(path_liver_to_lower) >= 2 else None
    liver_subtree = _collect_subtree(
        adj, bp_liver, excl_nb, endpoints, branch_points)
    liver_brs_all = liver_subtree['all_branches'] or liver_branches_direct
    tips_seg = None
    if len(liver_brs_all) >= 3:
        tips_seg, lpv_seg, rpv_seg = _assign_tips_lpv_rpv(liver_brs_all, nodes)
    else:
        lpv_seg, rpv_seg = _assign_lpv_rpv(liver_brs_all, nodes)

    bp_mpv_end_liver = _find_mpv_end_by_liver_branches(
        adj, nodes, sv_smv_bp, lpv_seg, rpv_seg, bp_liver)
    trimmed = _trim_branches_to_mpv_end(
        {'tips': tips_seg, 'lpv': lpv_seg, 'rpv': rpv_seg},
        bp_mpv_end_liver, adj, branch_points, nodes)
    tips_seg = trimmed['tips']
    lpv_seg = trimmed['lpv']
    rpv_seg = trimmed['rpv']

    mpv_seg = find_path(adj, bp_mpv_end_liver, sv_smv_bp)
    _check_sv_smv_mpv_confluence(mpv_seg, sv_seg, smv_seg, "V2 lower confluence")
    return {
        'segments': {
            'mpv': mpv_seg,
            'sv': sv_seg,
            'smv': smv_seg,
            'tips': tips_seg,
            'lpv': lpv_seg,
            'rpv': rpv_seg,
            'lgv': lgv_seg,
            'pgv': pgv_seg,
        },
        'has_compensation': comp_type is not None,
        'compensation_type': comp_type,
    }


# ============================================================
# JSON 构造
# ============================================================

def _check_sv_smv_mpv_confluence(mpv_seg, sv_seg, smv_seg, label="confluence"):
    """Log the hard anatomical rule: SV and SMV share one endpoint, then form MPV."""
    if not mpv_seg or not sv_seg or not smv_seg:
        print(f"  [WARN] {label}: MPV/SV/SMV segment missing, cannot check confluence rule")
        return False
    sv_ends = {sv_seg[0], sv_seg[-1]}
    smv_ends = {smv_seg[0], smv_seg[-1]}
    common = sv_ends & smv_ends
    mpv_ends = {mpv_seg[0], mpv_seg[-1]}
    valid_common = common & mpv_ends
    if not valid_common:
        print(
            f"  [WARN] {label}: SV/SMV/MPV do not share one confluence "
            f"(MPV={mpv_seg[0]}->{mpv_seg[-1]}, "
            f"SV={sv_seg[0]}->{sv_seg[-1]}, SMV={smv_seg[0]}->{smv_seg[-1]})"
        )
        return False
    confluence = next(iter(valid_common))
    print(f"  [OK] {label}: SV and SMV merge at {confluence}, then continue as MPV")
    return True


def _build_output_json(folder_name, post_tips, result, nodes,
                       branch_points, endpoints):
    out = {
        "patient_id": folder_name,
        "segment_vessels_version": SEGMENT_VESSELS_VERSION,
        "is_post_tips": post_tips,
        "has_compensation": result.get('has_compensation', False),
        "compensation_type": result.get('compensation_type', None),
        "n_branch_points": len(branch_points),
        "n_endpoints": len(endpoints),
        "branch_points": [
            {"id": int(bp),
             "coord": [float(nodes[bp]['x']),
                       float(nodes[bp]['y']),
                       float(nodes[bp]['z'])]}
            for bp in sorted(branch_points)
        ],
        "segments": {}
    }
    for key in (
            "segmentation_method", "coordinate_system",
            "anatomical_landmarks", "confidence", "diagnostics"):
        if key in result:
            out[key] = result[key]
    for name, path in result['segments'].items():
        if path is None or len(path) < 2:
            out['segments'][name] = None
            continue
        coords = path_to_coords(path, nodes)
        out['segments'][name] = {
            "path": [int(n) for n in path],
            "endpoints_id": [int(path[0]), int(path[-1])],
            "endpoints_coord": [
                [float(nodes[path[0]]['x']),
                 float(nodes[path[0]]['y']),
                 float(nodes[path[0]]['z'])],
                [float(nodes[path[-1]]['x']),
                 float(nodes[path[-1]]['y']),
                 float(nodes[path[-1]]['z'])],
            ],
            "n_points": len(path),
            "length_mm": float(path_physical_length(path, nodes)),
            "tortuosity": float(_path_tortuosity(coords)),
            "mean_curvature": float(_path_mean_curvature(coords)),
        }
    return out


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stl_path", nargs="?", default=r"F:\example\vessel.stl")
    parser.add_argument(
        "--post-tips",
        choices=["auto", "0", "1", "false", "true"],
        default="auto",
        help="Override TIPS status: auto, 1/true, or 0/false.",
    )
    parser.add_argument(
        "--coordinate-system",
        choices=["LPS", "RAS", "lps", "ras"],
        default="LPS",
        help="Patient coordinate convention used by STL/centerline coordinates.",
    )
    args = parser.parse_args()

    if args.post_tips in ("1", "true"):
        post_tips_arg = True
    elif args.post_tips in ("0", "false"):
        post_tips_arg = False
    else:
        post_tips_arg = None
    segment_vessels(
        args.stl_path,
        post_tips=post_tips_arg,
        coordinate_system=args.coordinate_system.upper(),
    )
