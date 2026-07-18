"""
SV-SMV 夹角独立调用接口（v3 - 读 JSON 驱动）
"""

import os
import json
from utils import load_tree
from extract_features import _compute_sv_smv_angle_from_segments
from features_layout import SEGMENT_ASSIGNMENTS_NAME, feature_path


def compute_sv_smv_angle(stl_path, n_fit_points=10, output_dir=None):
    """独立计算 SV-SMV 夹角。"""
    print(f"\n===== SV-SMV 夹角计算 =====")
    nodes, adj, parentdir = load_tree(stl_path)
    seg_json = feature_path(parentdir, SEGMENT_ASSIGNMENTS_NAME)
    if not seg_json.exists():
        raise FileNotFoundError(f"分段 JSON 不存在: {seg_json}")
    with open(seg_json, 'r', encoding='utf-8') as f:
        seg_dict = json.load(f).get('segments', {})

    result, err = _compute_sv_smv_angle_from_segments(
        seg_dict, nodes, n_fit_points=n_fit_points)

    if result is None:
        raise ValueError(f"无法计算夹角: {err}")

    print(f"  夹角: {result['angle_degrees']:.1f}°")
    return result


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else r"F:\example\vessel.stl"
    compute_sv_smv_angle(p)
