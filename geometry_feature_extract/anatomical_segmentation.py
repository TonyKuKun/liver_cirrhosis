"""Topology- and patient-coordinate-based portal venous segmentation.

The core vessel labels are selected jointly. Length is deliberately excluded
from anatomical classification; it is suitable for skeleton pruning, but not
for deciding vessel identity.
"""

from collections import deque
from itertools import permutations

import numpy as np

from utils import find_path, path_to_coords


METHOD_VERSION = "global-topology-patient-coordinates-v4-post-tips-hilar-transition"
POST_TIPS_HILAR_TRANSITION_MM = 8.0


def _coord(nodes, node_id):
    node = nodes[node_id]
    return np.array([node["x"], node["y"], node["z"]], dtype=float)


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else None


def _clip01(value):
    return float(np.clip(value, 0.0, 1.0))


def _positive_projection(dot_value):
    """Map an axis projection from [-1, 1] to an interpretable [0, 1]."""
    return _clip01(0.5 * (float(dot_value) + 1.0))


def _coordinate_axes(coordinate_system):
    system = str(coordinate_system or "LPS").upper()
    if system not in {"LPS", "RAS"}:
        raise ValueError(f"Unsupported coordinate system: {coordinate_system}")

    # Both conventions use +Z as superior. LPS uses +X as patient-left,
    # whereas RAS uses -X as patient-left. Posterior is only an auxiliary cue.
    left = np.array([1.0, 0.0, 0.0]) if system == "LPS" else np.array([-1.0, 0.0, 0.0])
    superior = np.array([0.0, 0.0, 1.0])
    posterior = np.array([0.0, 1.0, 0.0]) if system == "LPS" else np.array([0.0, -1.0, 0.0])
    return {
        "name": system,
        "left": left,
        "right": -left,
        "superior": superior,
        "inferior": -superior,
        "posterior": posterior,
    }


def _path_direction(path, nodes):
    """Robust direction from the anatomical anchor toward the distal end."""
    coords = path_to_coords(path, nodes)
    if len(coords) < 2:
        return None
    window = max(1, min(5, len(coords) // 5))
    start = np.mean(coords[:window], axis=0)
    end = np.mean(coords[-window:], axis=0)
    return _unit(end - start)


def _path_tortuosity(path, nodes):
    coords = path_to_coords(path, nodes)
    if len(coords) < 2:
        return 0.0
    chord = float(np.linalg.norm(coords[-1] - coords[0]))
    arc = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
    if arc <= 1e-9:
        return 0.0
    return float(np.clip(1.0 - chord / arc, 0.0, 1.0))


def _path_length(path, nodes):
    coords = path_to_coords(path, nodes)
    if len(coords) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def _common_prefix(path_a, path_b):
    prefix = []
    for node_a, node_b in zip(path_a, path_b):
        if node_a != node_b:
            break
        prefix.append(node_a)
    return prefix


def _direction_from_start(path, nodes, sample_dist=8.0):
    coords = path_to_coords(path, nodes)
    if len(coords) < 2:
        return None
    anchor = coords[0]
    sample = coords[-1]
    arc = 0.0
    for index in range(1, len(coords)):
        arc += float(np.linalg.norm(coords[index] - coords[index - 1]))
        if arc >= sample_dist:
            sample = coords[index]
            break
    return _unit(sample - anchor)


def _direction_into_end(path, nodes, sample_dist=8.0):
    coords = path_to_coords(path, nodes)
    if len(coords) < 2:
        return None
    anchor = coords[-1]
    sample = coords[0]
    arc = 0.0
    for index in range(len(coords) - 1, 0, -1):
        arc += float(np.linalg.norm(coords[index] - coords[index - 1]))
        if arc >= sample_dist:
            sample = coords[index - 1]
            break
    return _unit(anchor - sample)


def _split_path_from_end(path, nodes, trim_mm):
    """Return (kept_prefix, reassigned_suffix) split near the distal end."""
    if not path or len(path) < 4 or trim_mm <= 0:
        return path, None
    arc = 0.0
    split_index = None
    for index in range(len(path) - 1, 0, -1):
        a = _coord(nodes, path[index])
        b = _coord(nodes, path[index - 1])
        arc += float(np.linalg.norm(a - b))
        if arc >= trim_mm:
            split_index = index - 1
            break
    if split_index is None or split_index <= 0 or split_index >= len(path) - 1:
        return path, None
    return path[:split_index + 1], path[split_index:]


def _apply_post_tips_hilar_transition(core_paths, tips, nodes):
    """
    In post-TIPS anatomy, MPV should not claim the short hilar transition that
    has already entered the left/right portal system. Reassign the distal MPV
    transition to whichever hepatic branch is directionally continuous.
    """
    if tips is None:
        return core_paths, None
    required = {"mpv", "lpv", "rpv"}
    if not required <= set(core_paths):
        return core_paths, None

    mpv_path = list(core_paths["mpv"] or [])
    lpv_path = list(core_paths["lpv"] or [])
    rpv_path = list(core_paths["rpv"] or [])
    if min(map(len, (mpv_path, lpv_path, rpv_path))) < 2:
        return core_paths, None

    max_trim = min(
        POST_TIPS_HILAR_TRANSITION_MM,
        max(0.0, 0.35 * _path_length(mpv_path, nodes)),
    )
    trimmed_mpv, transition = _split_path_from_end(mpv_path, nodes, max_trim)
    if transition is None or len(transition) < 2:
        return core_paths, None

    incoming = _direction_into_end(mpv_path, nodes)
    branch_scores = {}
    for name, path in (("lpv", lpv_path), ("rpv", rpv_path)):
        direction = _direction_from_start(path, nodes)
        branch_scores[name] = (
            float(np.dot(incoming, direction))
            if incoming is not None and direction is not None else -2.0
        )
    assigned = max(branch_scores, key=branch_scores.get)

    updated = {name: list(path) if path is not None else None
               for name, path in core_paths.items()}
    updated["mpv"] = trimmed_mpv
    branch_path = updated[assigned]
    updated[assigned] = transition + branch_path[1:]

    return updated, {
        "assigned_to": assigned,
        "from_node": int(transition[0]),
        "to_node": int(transition[-1]),
        "length_mm": _path_length(transition, nodes),
        "branch_direction_scores": branch_scores,
    }


def _truncate_after_anchor(path, branch_points):
    """Keep the named vessel trunk only up to its next anatomical bifurcation."""
    if not path:
        return path
    for index, node_id in enumerate(path[1:], start=1):
        if node_id in branch_points:
            return path[:index + 1]
    return path


def _core_candidate(confluence, sv_endpoint, smv_endpoint,
                    lpv_endpoint, rpv_endpoint, endpoint_paths,
                    branch_points, nodes, axes):
    sv_path = endpoint_paths[sv_endpoint]
    smv_path = endpoint_paths[smv_endpoint]
    lpv_full = endpoint_paths[lpv_endpoint]
    rpv_full = endpoint_paths[rpv_endpoint]

    # SV, SMV and MPV must leave the confluence through three different ports.
    if min(map(len, (sv_path, smv_path, lpv_full, rpv_full))) < 2:
        return None
    if sv_path[1] == smv_path[1]:
        return None

    hepatic_prefix = _common_prefix(lpv_full, rpv_full)
    if len(hepatic_prefix) < 2:
        return None
    hepatic_bifurcation = hepatic_prefix[-1]
    if hepatic_bifurcation == confluence or hepatic_bifurcation not in branch_points:
        return None

    mpv_path = hepatic_prefix
    mpv_port = mpv_path[1]
    if mpv_port in {sv_path[1], smv_path[1]}:
        return None

    lpv_classification_path = lpv_full[len(hepatic_prefix) - 1:]
    rpv_classification_path = rpv_full[len(hepatic_prefix) - 1:]
    if (len(lpv_classification_path) < 2
            or len(rpv_classification_path) < 2
            or lpv_classification_path[1] == rpv_classification_path[1]):
        return None

    lpv_path = _truncate_after_anchor(lpv_classification_path, branch_points)
    rpv_path = _truncate_after_anchor(rpv_classification_path, branch_points)

    directions = {
        "sv": _path_direction(sv_path, nodes),
        "smv": _path_direction(smv_path, nodes),
        "mpv": _path_direction(mpv_path, nodes),
        "lpv": _path_direction(lpv_classification_path, nodes),
        "rpv": _path_direction(rpv_classification_path, nodes),
    }
    if any(direction is None for direction in directions.values()):
        return None

    left = axes["left"]
    superior = axes["superior"]
    sv_left = float(np.dot(directions["sv"], left))
    smv_left = float(np.dot(directions["smv"], left))
    sv_inferior = float(np.dot(directions["sv"], axes["inferior"]))
    smv_inferior = float(np.dot(directions["smv"], axes["inferior"]))
    mpv_superior = float(np.dot(directions["mpv"], superior))
    lpv_left = float(np.dot(directions["lpv"], left))
    rpv_left = float(np.dot(directions["rpv"], left))
    lpv_superior = float(np.dot(directions["lpv"], superior))
    rpv_superior = float(np.dot(directions["rpv"], superior))

    # These relative relationships are more stable than absolute thresholds.
    # Permutations already test the opposite assignment, so impossible ordering
    # can be rejected without using vessel length as a tie-breaker.
    if lpv_left <= rpv_left:
        return None

    sv_horizontal = 1.0 - abs(float(np.dot(directions["sv"], superior)))
    lr_separation = _clip01((lpv_left - rpv_left) / 1.5)
    lower_separation = 0.5 * (
        _clip01((sv_left - smv_left + 1.0) / 2.0)
        + _clip01((smv_inferior - sv_inferior + 1.0) / 2.0)
    )

    components = {
        "sv_leftward": _positive_projection(sv_left),
        "sv_horizontal": _clip01(sv_horizontal),
        "smv_inferior": _positive_projection(smv_inferior),
        "mpv_superior": _positive_projection(mpv_superior),
        "lpv_leftward": _positive_projection(lpv_left),
        "rpv_rightward": _positive_projection(-rpv_left),
        "hepatic_superior": 0.5 * (
            _positive_projection(lpv_superior)
            + _positive_projection(rpv_superior)
        ),
        "left_right_separation": lr_separation,
        "lower_branch_separation": lower_separation,
    }
    weights = {
        "sv_leftward": 0.16,
        "sv_horizontal": 0.05,
        "smv_inferior": 0.19,
        "mpv_superior": 0.16,
        "lpv_leftward": 0.12,
        "rpv_rightward": 0.12,
        "hepatic_superior": 0.09,
        "left_right_separation": 0.07,
        "lower_branch_separation": 0.04,
    }
    score = sum(weights[name] * components[name] for name in weights)

    # Soft penalties preserve unusual anatomy while making contradictions visible.
    penalties = []
    if sv_left <= smv_left:
        score -= 0.10
        penalties.append("SV is not more leftward than SMV")
    if smv_inferior <= sv_inferior:
        score -= 0.10
        penalties.append("SMV is not more inferior than SV")
    if mpv_superior < 0.0:
        score -= 0.12
        penalties.append("MPV does not run toward the liver/superior side")
    if min(lpv_superior, rpv_superior) < -0.25:
        score -= 0.08
        penalties.append("one hepatic branch points markedly inferiorly")

    return {
        "score": _clip01(score),
        "confluence": confluence,
        "hepatic_bifurcation": hepatic_bifurcation,
        "endpoints": {
            "sv": sv_endpoint,
            "smv": smv_endpoint,
            "lpv": lpv_endpoint,
            "rpv": rpv_endpoint,
        },
        "paths": {
            "sv": sv_path,
            "smv": smv_path,
            "mpv": mpv_path,
            "lpv": lpv_path,
            "rpv": rpv_path,
        },
        "components": components,
        "penalties": penalties,
    }


def _enumerate_core_candidates(nodes, adj, endpoints, branch_points, axes):
    candidates = []
    endpoints = sorted(endpoints)

    for confluence in sorted(branch_points):
        endpoint_paths = {
            endpoint: find_path(adj, confluence, endpoint)
            for endpoint in endpoints
        }
        endpoint_paths = {
            endpoint: path for endpoint, path in endpoint_paths.items()
            if path is not None and len(path) >= 2
        }
        available = sorted(endpoint_paths)
        if len(available) < 4:
            continue

        for sv_endpoint, smv_endpoint in permutations(available, 2):
            sv_path = endpoint_paths[sv_endpoint]
            smv_path = endpoint_paths[smv_endpoint]
            if sv_path[1] == smv_path[1]:
                continue

            remaining = [
                endpoint for endpoint in available
                if endpoint not in {sv_endpoint, smv_endpoint}
            ]
            for lpv_endpoint, rpv_endpoint in permutations(remaining, 2):
                candidate = _core_candidate(
                    confluence, sv_endpoint, smv_endpoint,
                    lpv_endpoint, rpv_endpoint, endpoint_paths,
                    branch_points, nodes, axes)
                if candidate is not None:
                    candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

    # Different terminal endpoints inside the same LPV/RPV subtree describe
    # the same named proximal vessel segment. Keep one representative so the
    # confidence margin measures genuinely different anatomical assignments.
    deduplicated = []
    seen = set()
    for candidate in candidates:
        signature = tuple(
            tuple(candidate["paths"][name])
            for name in ("mpv", "sv", "smv", "lpv", "rpv")
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduplicated.append(candidate)
    return deduplicated


def _nearest_core_path(endpoint, core_nodes, adj):
    queue = deque([(endpoint, [endpoint])])
    visited = {endpoint}
    while queue:
        current, path = queue.popleft()
        if current in core_nodes:
            return list(reversed(path))
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def _attachment_region(attachment, core_paths, confluence, hepatic_bifurcation):
    if attachment == confluence:
        return "confluence"
    if attachment in set(core_paths["sv"][1:-1]):
        return "sv"
    if attachment in set(core_paths["smv"][1:-1]):
        return "smv"
    if attachment in set(core_paths["mpv"][1:-1]):
        return "mpv"
    if attachment == hepatic_bifurcation:
        return "hepatic_bifurcation"
    if attachment in set(core_paths["lpv"][1:]):
        return "lpv"
    if attachment in set(core_paths["rpv"][1:]):
        return "rpv"
    return "core_endpoint"


def _extra_branches(best, endpoints, adj):
    core_paths = best["paths"]
    core_endpoints = set(best["endpoints"].values())
    core_nodes = set().union(*(set(path) for path in core_paths.values()))
    extras = []
    for endpoint in sorted(set(endpoints) - core_endpoints):
        path = _nearest_core_path(endpoint, core_nodes, adj)
        if path is None or len(path) < 2:
            continue
        attachment = path[0]
        extras.append({
            "endpoint": endpoint,
            "attachment": attachment,
            "attachment_region": _attachment_region(
                attachment, core_paths, best["confluence"],
                best["hepatic_bifurcation"]),
            "path": path,
        })
    return extras


def _load_mesh_for_radius(stl_path):
    if not stl_path:
        return None
    try:
        import trimesh
        mesh = trimesh.load(stl_path, force="mesh", process=False)
        return mesh if hasattr(mesh, "vertices") else None
    except Exception:
        return None


def _radius_uniformity(path, nodes, mesh):
    if mesh is None or len(path) < 3:
        return None
    try:
        import trimesh.proximity
        coords = path_to_coords(path, nodes)
        if len(coords) > 40:
            indices = np.linspace(0, len(coords) - 1, 40).astype(int)
            coords = coords[indices]
        radii = np.abs(np.asarray(
            trimesh.proximity.signed_distance(mesh, coords), dtype=float))
        radii = radii[np.isfinite(radii) & (radii > 1e-3)]
        if len(radii) < 3 or float(np.mean(radii)) <= 1e-6:
            return None
        coefficient_of_variation = float(np.std(radii) / np.mean(radii))
        return _clip01(np.exp(-2.5 * coefficient_of_variation))
    except Exception:
        return None


def _select_tips(extras, nodes, axes, stl_path):
    if not extras:
        return None, [], "TIPS was expected from the sample name, but no extra branch remained"

    mesh = _load_mesh_for_radius(stl_path)
    scored = []
    hepatic_regions = {"mpv", "hepatic_bifurcation", "lpv", "rpv"}
    for extra in extras:
        direction = _path_direction(extra["path"], nodes)
        superior = float(np.dot(direction, axes["superior"])) if direction is not None else -1.0
        straightness = 1.0 - _path_tortuosity(extra["path"], nodes)
        radius_uniformity = _radius_uniformity(extra["path"], nodes, mesh)

        values = {
            "hepatic_attachment": 1.0 if extra["attachment_region"] in hepatic_regions else 0.0,
            "superior_direction": _positive_projection(superior),
            "straightness": _clip01(straightness),
        }
        weights = {
            "hepatic_attachment": 0.55,
            "superior_direction": 0.25,
            "straightness": 0.20,
        }
        if radius_uniformity is not None:
            values["radius_uniformity"] = radius_uniformity
            weights = {
                "hepatic_attachment": 0.45,
                "superior_direction": 0.20,
                "straightness": 0.15,
                "radius_uniformity": 0.20,
            }
        score = sum(weights[key] * values[key] for key in weights)
        scored.append({**extra, "score": score, "components": values})

    scored.sort(key=lambda item: item["score"], reverse=True)
    selected = scored[0]
    reason = None
    if selected["attachment_region"] not in hepatic_regions:
        reason = "best TIPS candidate is not attached to the hepatic/MPV core"
    elif selected["score"] < 0.60:
        reason = "TIPS morphology score is low"
    return selected, scored, reason


def _select_collaterals(extras, nodes, axes):
    lgv_candidates = []
    pgv_candidates = []
    for extra in extras:
        direction = _path_direction(extra["path"], nodes)
        if direction is None:
            continue
        superior = _positive_projection(float(np.dot(direction, axes["superior"])))
        leftward = _positive_projection(float(np.dot(direction, axes["left"])))
        posterior = _positive_projection(float(np.dot(direction, axes["posterior"])))
        collateral_tortuosity = _clip01(_path_tortuosity(extra["path"], nodes) / 0.20)

        if extra["attachment_region"] in {"confluence", "mpv"}:
            score = 0.65 + 0.20 * superior + 0.15 * leftward
            lgv_candidates.append({**extra, "score": score})
        elif extra["attachment_region"] == "sv":
            score = (
                0.60
                + 0.15 * superior
                + 0.15 * posterior
                + 0.10 * collateral_tortuosity
            )
            pgv_candidates.append({**extra, "score": score})

    lgv_candidates.sort(key=lambda item: item["score"], reverse=True)
    pgv_candidates.sort(key=lambda item: item["score"], reverse=True)
    lgv = lgv_candidates[0] if lgv_candidates else None
    pgv = pgv_candidates[0] if pgv_candidates else None
    return lgv, pgv, lgv_candidates, pgv_candidates


def _confidence(best, candidates, post_tips, tips_reason):
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    margin = float(best["score"] - second_score)
    reasons = list(best.get("penalties", []))

    if best["score"] < 0.66:
        reasons.append("overall anatomical consistency score is low")
    if margin < 0.04:
        reasons.append("the best and second-best anatomical assignments are close")
    if post_tips and tips_reason:
        reasons.append(tips_reason)

    if best["score"] >= 0.78 and margin >= 0.08 and not reasons:
        level = "high"
    elif best["score"] >= 0.66 and margin >= 0.04:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "score": float(best["score"]),
        "margin_to_second": margin,
        "needs_manual_review": bool(reasons),
        "review_reasons": reasons,
    }


def segment_anatomically(nodes, adj, endpoints, branch_points,
                         post_tips=False, coordinate_system="LPS",
                         stl_path=None):
    """Return a globally consistent portal venous anatomy assignment."""
    if len(endpoints) < 4 or len(branch_points) < 2:
        raise ValueError(
            "Anatomical segmentation requires at least four endpoints and two branch points")

    axes = _coordinate_axes(coordinate_system)
    candidates = _enumerate_core_candidates(
        nodes, adj, endpoints, branch_points, axes)
    if not candidates:
        raise ValueError(
            "No topology-consistent SV/SMV-MPV-LPV/RPV assignment was found")

    best = candidates[0]
    extras = _extra_branches(best, endpoints, adj)
    tips = lgv = pgv = None
    tips_scores = []
    collateral_scores = {"lgv": [], "pgv": []}
    tips_reason = None

    if post_tips:
        tips, tips_scores, tips_reason = _select_tips(
            extras, nodes, axes, stl_path)
    else:
        lgv, pgv, lgv_scores, pgv_scores = _select_collaterals(
            extras, nodes, axes)
        collateral_scores = {"lgv": lgv_scores, "pgv": pgv_scores}

    confidence = _confidence(best, candidates, post_tips, tips_reason)
    unexplained_extras = [
        extra for extra in extras
        if extra["attachment_region"] not in {
            "hepatic_bifurcation", "lpv", "rpv", "core_endpoint"
        }
    ]
    if not post_tips and unexplained_extras and lgv is None and pgv is None:
        confidence["needs_manual_review"] = True
        confidence["review_reasons"].append(
            "one or more extra branches could not be assigned by attachment anatomy")

    compensation_type = None
    if lgv is not None and pgv is not None:
        compensation_type = "LGV+PGV"
    elif lgv is not None:
        compensation_type = "LGV"
    elif pgv is not None:
        compensation_type = "PGV"

    core_paths = best["paths"]
    post_tips_hilar_transition = None
    if post_tips:
        core_paths, post_tips_hilar_transition = _apply_post_tips_hilar_transition(
            core_paths, tips, nodes)

    anatomical_landmarks = {
        "sv_smv_confluence_id": int(best["confluence"]),
        "hepatic_bifurcation_id": int(best["hepatic_bifurcation"]),
    }
    if post_tips_hilar_transition is not None:
        anatomical_landmarks["post_tips_mpv_trim_node_id"] = int(
            post_tips_hilar_transition["from_node"])

    return {
        "segments": {
            "mpv": core_paths["mpv"],
            "sv": core_paths["sv"],
            "smv": core_paths["smv"],
            "lpv": core_paths["lpv"],
            "rpv": core_paths["rpv"],
            "tips": tips["path"] if tips is not None else None,
            "lgv": lgv["path"] if lgv is not None else None,
            "pgv": pgv["path"] if pgv is not None else None,
        },
        "has_compensation": compensation_type is not None,
        "compensation_type": compensation_type,
        "segmentation_method": METHOD_VERSION,
        "coordinate_system": axes["name"],
        "anatomical_landmarks": anatomical_landmarks,
        "confidence": confidence,
        "diagnostics": {
            "n_core_candidates": len(candidates),
            "best_components": best["components"],
            "post_tips_hilar_transition": post_tips_hilar_transition,
            "top_core_candidates": [
                {
                    "score": float(candidate["score"]),
                    "confluence": int(candidate["confluence"]),
                    "hepatic_bifurcation": int(candidate["hepatic_bifurcation"]),
                    "endpoints": {
                        key: int(value)
                        for key, value in candidate["endpoints"].items()
                    },
                }
                for candidate in candidates[:3]
            ],
            "extra_branches": [
                {
                    "endpoint": int(extra["endpoint"]),
                    "attachment": int(extra["attachment"]),
                    "attachment_region": extra["attachment_region"],
                }
                for extra in extras
            ],
            "tips_candidates": [
                {
                    "endpoint": int(item["endpoint"]),
                    "attachment": int(item["attachment"]),
                    "attachment_region": item["attachment_region"],
                    "score": float(item["score"]),
                    "components": item["components"],
                }
                for item in tips_scores
            ],
            "collateral_candidates": {
                name: [
                    {
                        "endpoint": int(item["endpoint"]),
                        "attachment": int(item["attachment"]),
                        "score": float(item["score"]),
                    }
                    for item in items
                ]
                for name, items in collateral_scores.items()
            },
        },
    }
