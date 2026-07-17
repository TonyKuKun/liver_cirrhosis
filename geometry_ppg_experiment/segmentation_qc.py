"""QC portal-vein anatomical segmentation labels for all samples.

The script reads the existing unified feature outputs and centerline profiles.
It does not regenerate segmentations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_DATA_ROOT = Path(r"F:\PCG data\dataset\test4all_sample")
CORE_SEGMENTS = ("mpv", "smv", "sv")
INTRAHEPATIC_SEGMENTS = ("lpv", "rpv")
OPTIONAL_SEGMENTS = ("tips", "lgv", "pgv")
ALL_SEGMENTS = CORE_SEGMENTS + INTRAHEPATIC_SEGMENTS + OPTIONAL_SEGMENTS


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def as_point(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    coords = tuple(safe_float(v) for v in value[:3])
    if not all(math.isfinite(v) for v in coords):
        return None
    return coords  # type: ignore[return-value]


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def segment_profiles(patient_dir: Path, unified: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = read_json(patient_dir / "centerline_profiles.json")
    segments = profiles.get("segments")
    if isinstance(segments, dict):
        return {str(k).lower(): v for k, v in segments.items() if isinstance(v, dict)}
    segments = (unified.get("centerline_profiles") or {}).get("segments")
    if isinstance(segments, dict):
        return {str(k).lower(): v for k, v in segments.items() if isinstance(v, dict)}
    return {}


def pointwise_profiles(patient_dir: Path, unified: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = read_json(patient_dir / "centerline_pointwise_profiles.json")
    if isinstance(profiles, dict) and profiles:
        return {str(k).lower(): v for k, v in profiles.items() if isinstance(v, dict)}
    pointwise = unified.get("pointwise")
    if isinstance(pointwise, dict):
        return {str(k).lower(): v for k, v in pointwise.items() if isinstance(v, dict)}
    return {}


def path_len(record: dict[str, Any]) -> int:
    path = record.get("path")
    return len(path) if isinstance(path, list) else 0


def has_area_profile(record: dict[str, Any]) -> bool:
    area = record.get("area")
    return isinstance(area, list) and sum(1 for x in area if safe_float(x) > 0) >= 3


def length_mm(record: dict[str, Any]) -> float:
    for key in ("total_length_mm", "length_mm", "arc_length_total_mm"):
        value = safe_float(record.get(key))
        if math.isfinite(value):
            return value
    arc = record.get("arc_length_mm")
    if isinstance(arc, list) and arc:
        values = [safe_float(v) for v in arc]
        values = [v for v in values if math.isfinite(v)]
        if values:
            return max(values) - min(values)
    return math.nan


def diameter_mm(record: dict[str, Any], unified: dict[str, Any], segment: str) -> float:
    for key in ("mean_diameter_mm", "diameter_mm", "median_diameter_mm"):
        value = safe_float(record.get(key))
        if math.isfinite(value):
            return value
    area = record.get("area")
    if isinstance(area, list):
        diameters = []
        for value in area:
            a = safe_float(value)
            if a > 0:
                diameters.append(2.0 * math.sqrt(a / math.pi))
        if diameters:
            return float(median(diameters))
    portal = unified.get("portal_vein_features") or unified
    for key in (
        f"{segment}_mean_diameter_mm",
        f"{segment}_diameter_mm",
        f"{segment}_median_diameter_mm",
        f"{segment}_max_diameter_mm",
    ):
        value = safe_float(portal.get(key))
        if math.isfinite(value):
            return value
    return math.nan


def endpoints(record: dict[str, Any]) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    for start_key, end_key in (("start_point", "end_point"), ("start_coord", "end_coord"), ("p0", "p1")):
        start = as_point(record.get(start_key))
        end = as_point(record.get(end_key))
        if start and end:
            return start, end
    endpoint_coords = record.get("endpoints_coord")
    if isinstance(endpoint_coords, list) and len(endpoint_coords) >= 2:
        start = as_point(endpoint_coords[0])
        end = as_point(endpoint_coords[-1])
        if start and end:
            return start, end
    points = record.get("points") or record.get("centerline_points") or record.get("coords")
    if isinstance(points, list) and len(points) >= 2:
        start = as_point(points[0])
        end = as_point(points[-1])
        if start and end:
            return start, end
    return None, None


def presence(unified: dict[str, Any], centerline: dict[str, dict[str, Any]], pointwise: dict[str, dict[str, Any]], segment: str) -> bool:
    record = centerline.get(segment, {})
    if path_len(record) >= 2 or has_area_profile(pointwise.get(segment, {})):
        return True
    vp = unified.get("vessel_presence") or {}
    info = vp.get(segment)
    if isinstance(info, dict):
        return bool(info.get("present"))
    return bool(info)


def closest_endpoint_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    a0, a1 = endpoints(a)
    b0, b1 = endpoints(b)
    pts_a = [p for p in (a0, a1) if p]
    pts_b = [p for p in (b0, b1) if p]
    if not pts_a or not pts_b:
        return math.nan
    return min(dist(x, y) for x in pts_a for y in pts_b)


def issue(severity: str, code: str, detail: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "detail": detail}


def qc_patient(patient_dir: Path) -> dict[str, Any]:
    name = patient_dir.name
    is_post_tips = "#" in name
    unified_path = patient_dir / "unified_features.json"
    unified = read_json(unified_path)
    centerline = segment_profiles(patient_dir, unified)
    pointwise = pointwise_profiles(patient_dir, unified)
    issues: list[dict[str, str]] = []

    if not unified_path.exists():
        issues.append(issue("critical", "missing_unified_features", "缺少 unified_features.json，无法读取分段结果"))
    if not centerline:
        issues.append(issue("critical", "missing_centerline_profiles", "缺少 centerline_profiles.json 或其中无 segments"))

    present = {seg: presence(unified, centerline, pointwise, seg) for seg in ALL_SEGMENTS}
    lengths = {seg: length_mm(centerline.get(seg, {})) for seg in ALL_SEGMENTS}
    diameters = {
        seg: diameter_mm(pointwise.get(seg, {}) or centerline.get(seg, {}), unified, seg)
        for seg in ALL_SEGMENTS
    }
    paths = {seg: path_len(centerline.get(seg, {})) for seg in ALL_SEGMENTS}

    for seg in CORE_SEGMENTS:
        if not present[seg]:
            issues.append(issue("critical", f"missing_required_{seg}", f"必需血管 {seg.upper()} 不存在"))
        elif paths[seg] < 2 and not has_area_profile(pointwise.get(seg, {})):
            issues.append(issue("critical", f"weak_required_{seg}", f"{seg.upper()} 仅有 presence 标记，缺少有效中心线路径/面积序列"))

    if is_post_tips and not present["tips"]:
        issues.append(issue("critical", "post_tips_missing_tips", "样本名含 #，应有 TIPS 手术管分段，但未检测到 tips"))
    if not is_post_tips and present["tips"]:
        issues.append(issue("warning", "pre_tips_has_tips", "样本名不含 #，但检测到 tips 分段，请确认是否命名或分段有误"))

    for seg in ALL_SEGMENTS:
        if not present[seg]:
            continue
        if paths[seg] == 1:
            issues.append(issue("major", f"{seg}_single_node_path", f"{seg.upper()} 中心线路径只有 1 个节点"))
        if math.isfinite(lengths[seg]) and lengths[seg] <= 1.0:
            issues.append(issue("major", f"{seg}_very_short", f"{seg.upper()} 长度 {lengths[seg]:.2f} mm，疑似分段过短"))
        if math.isfinite(diameters[seg]) and (diameters[seg] < 1.0 or diameters[seg] > 35.0):
            issues.append(issue("major", f"{seg}_diameter_outlier", f"{seg.upper()} 典型直径 {diameters[seg]:.2f} mm 超出生理/分割合理范围"))

    if present["lpv"] and present["rpv"]:
        l_d, r_d = diameters["lpv"], diameters["rpv"]
        if math.isfinite(l_d) and math.isfinite(r_d) and min(l_d, r_d) > 0:
            ratio = max(l_d, r_d) / min(l_d, r_d)
            if ratio > 3.5:
                issues.append(issue("warning", "lpv_rpv_diameter_imbalance", f"LPV/RPV 直径差异过大，较大/较小={ratio:.2f}"))
    if present["smv"] and present["sv"]:
        s_d, v_d = diameters["smv"], diameters["sv"]
        if math.isfinite(s_d) and math.isfinite(v_d) and min(s_d, v_d) > 0:
            ratio = max(s_d, v_d) / min(s_d, v_d)
            if ratio > 3.5:
                issues.append(issue("warning", "smv_sv_diameter_imbalance", f"SMV/SV 直径差异过大，较大/较小={ratio:.2f}"))

    connectivity_pairs = (("smv", "mpv"), ("sv", "mpv"), ("mpv", "lpv"), ("mpv", "rpv"))
    for a, b in connectivity_pairs:
        if present[a] and present[b]:
            gap = closest_endpoint_gap(centerline.get(a, {}), centerline.get(b, {}))
            if math.isfinite(gap) and gap > 25.0:
                issues.append(issue("warning", f"{a}_{b}_endpoint_gap", f"{a.upper()} 与 {b.upper()} 最近端点距离 {gap:.1f} mm，疑似拓扑不连续或标签串错"))

    if is_post_tips and present["tips"] and present["mpv"]:
        gap = closest_endpoint_gap(centerline.get("tips", {}), centerline.get("mpv", {}))
        if math.isfinite(gap) and gap > 35.0:
            issues.append(issue("warning", "tips_mpv_endpoint_gap", f"TIPS 与 MPV 最近端点距离 {gap:.1f} mm，疑似 TIPS 管未接入门静脉侧"))

    severities = {item["severity"] for item in issues}
    if "critical" in severities:
        status = "critical"
    elif "major" in severities:
        status = "major"
    elif "warning" in severities:
        status = "warning"
    else:
        status = "ok"

    return {
        "sample": name,
        "is_post_tips_by_name": is_post_tips,
        "status": status,
        "present": present,
        "path_nodes": paths,
        "length_mm": lengths,
        "diameter_mm": diameters,
        "issues": issues,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "sample",
        "status",
        "is_post_tips_by_name",
        "present_mpv",
        "present_smv",
        "present_sv",
        "present_lpv",
        "present_rpv",
        "present_tips",
        "issue_count",
        "critical_count",
        "major_count",
        "warning_count",
        "issue_codes",
        "issue_details",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            counts = Counter(item["severity"] for item in row["issues"])
            out = {
                "sample": row["sample"],
                "status": row["status"],
                "is_post_tips_by_name": int(row["is_post_tips_by_name"]),
                "issue_count": len(row["issues"]),
                "critical_count": counts["critical"],
                "major_count": counts["major"],
                "warning_count": counts["warning"],
                "issue_codes": ";".join(item["code"] for item in row["issues"]),
                "issue_details": " | ".join(item["detail"] for item in row["issues"]),
            }
            for seg in ("mpv", "smv", "sv", "lpv", "rpv", "tips"):
                out[f"present_{seg}"] = int(row["present"].get(seg, False))
            writer.writerow(out)


def write_markdown(rows: list[dict[str, Any]], path: Path, data_root: Path) -> None:
    status_counts = Counter(row["status"] for row in rows)
    issue_counts = Counter(item["code"] for row in rows for item in row["issues"])
    post_tips = [row for row in rows if row["is_post_tips_by_name"]]
    missing_required = {
        seg: [row["sample"] for row in rows if any(item["code"] == f"missing_required_{seg}" for item in row["issues"])]
        for seg in CORE_SEGMENTS
    }
    missing_tips = [row["sample"] for row in rows if any(item["code"] == "post_tips_missing_tips" for item in row["issues"])]

    lines = [
        "# 解剖分段 QC 统计",
        "",
        f"- 数据根目录：`{data_root}`",
        f"- 扫描样本目录：{len(rows)}",
        f"- 名字含 `#` 的 TIPS 术后样本：{len(post_tips)}",
        f"- 状态统计：critical={status_counts['critical']}, major={status_counts['major']}, warning={status_counts['warning']}, ok={status_counts['ok']}",
        "",
        "## 规则",
        "",
        "- `critical`：缺少统一特征文件、缺少中心线分段、或 MPV/SMV/SV/TIPS 这类硬性必需分段缺失。",
        "- `major`：已标记存在但中心线只有 1 个节点、长度极短、或直径明显超出生理/分割合理范围。",
        "- `warning`：分支直径严重失衡、端点距离过大、TIPS 命名与分段不一致等。",
        "",
        "## 核心硬性问题",
        "",
        f"- 缺 MPV：{len(missing_required['mpv'])} 个",
        f"- 缺 SMV：{len(missing_required['smv'])} 个",
        f"- 缺 SV：{len(missing_required['sv'])} 个",
        f"- 名字含 `#` 但缺 tips：{len(missing_tips)} 个",
        "",
        "## 问题类型计数",
        "",
        "| 问题代码 | 样本数 |",
        "|---|---:|",
    ]
    for code, count in issue_counts.most_common():
        lines.append(f"| `{code}` | {count} |")

    lines.extend(["", "## 有问题样本清单", "", "| 样本 | 状态 | 问题 |", "|---|---|---|"])
    for row in rows:
        if row["status"] == "ok":
            continue
        details = "<br>".join(f"[{item['severity']}] {item['detail']}" for item in row["issues"])
        lines.append(f"| `{row['sample']}` | {row['status']} | {details} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    patients = sorted(
        [p for p in args.data_root.iterdir() if p.is_dir() and p.name[:1].isdigit()],
        key=lambda p: p.name.lower(),
    )
    rows = [qc_patient(p) for p in patients]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "segmentation_qc_report.json"
    csv_path = args.out_dir / "segmentation_qc_report.csv"
    md_path = args.out_dir / "segmentation_qc_report.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args.data_root)

    status_counts = Counter(row["status"] for row in rows)
    print(f"rows={len(rows)}")
    print("status_counts=" + json.dumps(status_counts, ensure_ascii=False, sort_keys=True))
    print(f"json={json_path.resolve()}")
    print(f"csv={csv_path.resolve()}")
    print(f"markdown={md_path.resolve()}")


if __name__ == "__main__":
    main()
