from collections import defaultdict

from anatomical_segmentation import segment_anatomically
from utils import classify_nodes


def _tree(coords, edges):
    nodes = {
        node_id: {
            "id": node_id,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
            "parent": -1,
            "left": -1,
            "right": -1,
        }
        for node_id, coord in coords.items()
    }
    adj = defaultdict(set)
    for node_a, node_b in edges:
        adj[node_a].add(node_b)
        adj[node_b].add(node_a)
    return nodes, adj


def _basic_tree(ras=False):
    x_sign = -1.0 if ras else 1.0
    coords = {
        0: (0, 0, 0),                 # SV/SMV confluence
        1: (-10 * x_sign, 5, 22),     # hepatic bifurcation
        2: (38 * x_sign, 3, 2),       # SV
        3: (-2 * x_sign, -4, -45),    # SMV
        4: (24 * x_sign, -12, 42),    # LPV
        5: (-38 * x_sign, 16, 39),    # RPV
    }
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    return _tree(coords, edges)


def test_core_labels_follow_topology_and_patient_axes():
    nodes, adj = _basic_tree()
    endpoints, branch_points = classify_nodes(nodes, adj)

    result = segment_anatomically(
        nodes, adj, endpoints, branch_points,
        post_tips=False, coordinate_system="LPS")

    assert result["segments"]["sv"][-1] == 2
    assert result["segments"]["smv"][-1] == 3
    assert result["segments"]["mpv"] == [0, 1]
    assert result["segments"]["lpv"][-1] == 4
    assert result["segments"]["rpv"][-1] == 5
    assert result["anatomical_landmarks"] == {
        "sv_smv_confluence_id": 0,
        "hepatic_bifurcation_id": 1,
    }


def test_ras_coordinate_convention_reverses_x_axis_semantics():
    nodes, adj = _basic_tree(ras=True)
    endpoints, branch_points = classify_nodes(nodes, adj)

    result = segment_anatomically(
        nodes, adj, endpoints, branch_points,
        post_tips=False, coordinate_system="RAS")

    assert result["segments"]["sv"][-1] == 2
    assert result["segments"]["smv"][-1] == 3
    assert result["segments"]["lpv"][-1] == 4
    assert result["segments"]["rpv"][-1] == 5


def test_branch_attached_to_sv_is_classified_as_pgv():
    coords = {
        0: (0, 0, 0),
        1: (-10, 5, 22),
        2: (42, 0, 3),                 # SV endpoint
        3: (0, -3, -44),               # SMV endpoint
        4: (25, -12, 42),              # LPV endpoint
        5: (-38, 16, 39),              # RPV endpoint
        6: (14, 1, 1),                 # SV side-branch point
        7: (18, 22, 16),               # PGV endpoint
    }
    edges = [
        (0, 1), (0, 3), (0, 6),
        (6, 2), (6, 7),
        (1, 4), (1, 5),
    ]
    nodes, adj = _tree(coords, edges)
    endpoints, branch_points = classify_nodes(nodes, adj)

    result = segment_anatomically(
        nodes, adj, endpoints, branch_points,
        post_tips=False, coordinate_system="LPS")

    assert result["segments"]["sv"][-1] == 2
    assert result["segments"]["smv"][-1] == 3
    assert result["segments"]["pgv"] == [6, 7]
    assert result["segments"]["lgv"] is None
    assert result["compensation_type"] == "PGV"


def test_post_tips_name_prior_enables_only_hepatic_extra_branch():
    nodes, adj = _basic_tree()
    nodes[6] = {
        "id": 6,
        "x": -8.0,
        "y": 7.0,
        "z": 55.0,
        "parent": -1,
        "left": -1,
        "right": -1,
    }
    adj[1].add(6)
    adj[6].add(1)
    endpoints, branch_points = classify_nodes(nodes, adj)

    result = segment_anatomically(
        nodes, adj, endpoints, branch_points,
        post_tips=True, coordinate_system="LPS")

    assert result["segments"]["tips"] == [1, 6]
    assert result["segments"]["lgv"] is None
    assert result["segments"]["pgv"] is None
    assert result["has_compensation"] is False


def test_right_portal_trunk_starts_at_first_lpv_rpv_division():
    coords = {
        0: (0, 0, 0),                  # SV/SMV confluence
        1: (-8, 4, 18),                # first portal division: LPV branches
        6: (-14, 7, 28),               # right portal/TIPS split
        2: (38, 2, 1),                 # SV endpoint
        3: (-1, -4, -42),              # SMV endpoint
        4: (25, -8, 32),               # LPV endpoint
        5: (-40, 15, 38),              # RPV endpoint
        7: (-10, 8, 58),               # TIPS endpoint
    }
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 6),
        (6, 5), (6, 7),
    ]
    nodes, adj = _tree(coords, edges)
    endpoints, branch_points = classify_nodes(nodes, adj)

    result = segment_anatomically(
        nodes, adj, endpoints, branch_points,
        post_tips=True, coordinate_system="LPS")

    assert result["segments"]["mpv"] == [0, 1]
    assert result["segments"]["rpv"] == [1, 6]
    assert result["segments"]["tips"] == [6, 7]


def test_post_tips_hilar_transition_is_not_mpv():
    coords = {
        0: (0, 0, 0),                  # SV/SMV confluence
        8: (-3, 1, 8),                 # MPV trunk
        9: (-6, 2, 16),                # MPV clinical distal boundary
        1: (-8, 4, 24),                # hilar LPV/right-portal division
        6: (-14, 7, 34),               # right portal/TIPS split
        2: (38, 2, 1),                 # SV endpoint
        3: (-1, -4, -42),              # SMV endpoint
        4: (25, -8, 34),               # LPV endpoint
        5: (-40, 15, 42),              # RPV endpoint
        7: (-10, 8, 60),               # TIPS endpoint
    }
    edges = [
        (0, 8), (8, 9), (9, 1),
        (0, 2), (0, 3),
        (1, 4), (1, 6),
        (6, 5), (6, 7),
    ]
    nodes, adj = _tree(coords, edges)
    endpoints, branch_points = classify_nodes(nodes, adj)

    result = segment_anatomically(
        nodes, adj, endpoints, branch_points,
        post_tips=True, coordinate_system="LPS")

    assert result["segments"]["mpv"] == [0, 8, 9]
    assert result["segments"]["rpv"] == [9, 1, 6]
    assert result["segments"]["tips"] == [6, 7]
    assert result["diagnostics"]["post_tips_hilar_transition"]["assigned_to"] == "rpv"
