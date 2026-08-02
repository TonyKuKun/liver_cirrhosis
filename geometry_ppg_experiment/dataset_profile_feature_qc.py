"""QC extracted dataset profile features for each anatomical segment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import extract_features as ef


DEFAULT_DATA_ROOT = Path(r"F:\PCG data\dataset\test4all_sample")
SEGMENTS = ("mpv", "sv", "smv", "lpv", "rpv", "tips", "lgv", "pgv")
CORE_SEGMENTS = ("mpv", "smv", "sv")
PROFILE_ARRAY_KEYS = (
    "position",
    "arc_length_mm",
    "area",
    "eq_diameter",
    "perimeter",
    "raw_area",
    "raw_eq_diameter",
    "raw_perimeter",
    "anchor_radius",
    "owned_radius",
    "hydraulic_diameter",
    "circularity",
    "solidity",
    "r_insc_to_r_eq_ratio",
    "junction_replaced",
    "curvature",
    "torsion",
    "dA_ds_norm",
    "inscribed_radius",
)
PROFILE_SCALAR_KEYS = (
    "total_length_mm",
    "n_raw_points",
    "n_section_success",
    "edge_margin_pct",
    "edge_margin_mm",
    "n_masked_endpoints",
    "n_junction_protected",
    "n_junction_replaced",
)
RADIUS_AREA_KEYS = (
    "area",
    "raw_area",
    "eq_diameter",
    "raw_eq_diameter",
    "hydraulic_diameter",
    "inscribed_radius",
    "owned_radius",
    "anchor_radius",
)
STAT_KEYS = (
    "length",
    "tortuosity",
    "mean_curvature",
    "max_curvature",
    "mean_diameter",
    "max_diameter",
    "mean_area",
    "area_cv",
    "mean_circularity",
)


def finite_positive_count(values: Any) -> int:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return 0
    return int(np.isfinite(arr).sum()) if arr.dtype.kind in "fcbiu" else 0


def valid_array_info(record: dict[str, Any], key: str, expected_len: int | None) -> dict[str, Any]:
    value = record.get(key)
    info: dict[str, Any] = {
        "exists": key in record,
        "is_list": isinstance(value, list),
        "length": len(value) if isinstance(value, list) else None,
        "valid_n": 0,
        "length_ok": True,
    }
    if isinstance(value, list):
        info["valid_n"] = finite_positive_count(value)
    if expected_len is not None and isinstance(value, list):
        info["length_ok"] = len(value) == expected_len
    return info


def scalar_valid(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, bool):
        return True
    number = ef.safe_float(value)
    return bool(np.isfinite(number))


def segment_present(sources: dict[str, Any], segment: str) -> bool:
    return ef.segment_present(sources, segment)


def qc_segment(
    sources: dict[str, Any],
    portal_features: dict[str, Any],
    segment: str,
    required: bool,
) -> dict[str, Any]:
    present = segment_present(sources, segment)
    record = ef.pointwise_segment(sources, segment)
    result: dict[str, Any] = {
        "present": present,
        "required": required,
        "profile_exists": bool(record),
        "profile_n": 0,
        "missing_arrays": [],
        "invalid_arrays": [],
        "length_mismatch_arrays": [],
        "missing_scalars": [],
        "invalid_scalars": [],
        "missing_stats": [],
        "invalid_stats": [],
        "radius_area_valid": {},
        "issues": [],
    }

    if not present:
        if required:
            result["issues"].append(f"{segment}_missing")
        return result

    if not record:
        result["issues"].append(f"{segment}_profile_missing")
        return result

    position = ef.points_array(record.get("position"))
    arc = ef.finite_array(record.get("arc_length_mm"))
    expected_len = int(position.shape[0] or arc.size or 0)
    result["profile_n"] = expected_len

    for key in PROFILE_ARRAY_KEYS:
        info = valid_array_info(record, key, expected_len if expected_len > 0 else None)
        if not info["exists"]:
            result["missing_arrays"].append(key)
            continue
        if not info["is_list"]:
            result["invalid_arrays"].append(key)
            continue
        if not info["length_ok"]:
            result["length_mismatch_arrays"].append(key)
        if int(info["valid_n"]) == 0:
            result["invalid_arrays"].append(key)

    for key in PROFILE_SCALAR_KEYS:
        if key not in record:
            result["missing_scalars"].append(key)
        elif not scalar_valid(record.get(key)):
            result["invalid_scalars"].append(key)

    for key in RADIUS_AREA_KEYS:
        arr = ef.finite_array(record.get(key))
        result["radius_area_valid"][key] = int(np.isfinite(arr).sum()) if arr.size else 0

    for key in STAT_KEYS:
        flat_key = f"{segment}_{key}"
        if flat_key not in portal_features:
            result["missing_stats"].append(flat_key)
        elif not scalar_valid(portal_features.get(flat_key)):
            result["invalid_stats"].append(flat_key)

    if result["missing_arrays"]:
        result["issues"].append(f"{segment}_missing_profile_arrays")
    if result["invalid_arrays"]:
        result["issues"].append(f"{segment}_invalid_profile_arrays")
    if result["length_mismatch_arrays"]:
        result["issues"].append(f"{segment}_profile_array_length_mismatch")
    if result["missing_scalars"] or result["invalid_scalars"]:
        result["issues"].append(f"{segment}_bad_profile_scalars")
    if result["missing_stats"] or result["invalid_stats"]:
        result["issues"].append(f"{segment}_bad_summary_stats")

    return result


def load_sources(patient_dir: Path) -> dict[str, Any]:
    return {
        "unified": ef.read_json(patient_dir / "unified_features.json"),
        "centerline_profiles": ef.read_json(patient_dir / "centerline_profiles.json"),
        "pointwise_profiles": ef.read_json(patient_dir / "centerline_pointwise_profiles.json"),
        "portal_vein_features": ef.read_json(patient_dir / "portal_vein_features.json"),
        "nodes": ef.load_centerline_nodes(patient_dir),
        "patient_dir": patient_dir,
    }


def qc_patient(patient_dir: Path) -> dict[str, Any]:
    sources = load_sources(patient_dir)
    portal_features = sources.get("portal_vein_features") or {}
    is_post_tips = "#" in patient_dir.name
    segments = {}
    issues: list[str] = []

    for segment in SEGMENTS:
        required = segment in CORE_SEGMENTS or (segment == "tips" and is_post_tips)
        seg_result = qc_segment(sources, portal_features, segment, required)
        segments[segment] = seg_result
        issues.extend(seg_result["issues"])

    status = "ok" if not issues else "problem"
    return {
        "sample": patient_dir.name,
        "is_post_tips_by_name": is_post_tips,
        "status": status,
        "issues": issues,
        "segments": segments,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "sample",
        "status",
        "segment",
        "required",
        "present",
        "profile_exists",
        "profile_n",
        "issues",
        "missing_arrays",
        "invalid_arrays",
        "length_mismatch_arrays",
        "missing_scalars",
        "invalid_scalars",
        "missing_stats",
        "invalid_stats",
        *[f"{key}_valid_n" for key in RADIUS_AREA_KEYS],
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for segment, seg in row["segments"].items():
                if not seg["present"] and not seg["required"]:
                    continue
                out = {
                    "sample": row["sample"],
                    "status": row["status"],
                    "segment": segment,
                    "required": int(seg["required"]),
                    "present": int(seg["present"]),
                    "profile_exists": int(seg["profile_exists"]),
                    "profile_n": seg["profile_n"],
                    "issues": ";".join(seg["issues"]),
                    "missing_arrays": ";".join(seg["missing_arrays"]),
                    "invalid_arrays": ";".join(seg["invalid_arrays"]),
                    "length_mismatch_arrays": ";".join(seg["length_mismatch_arrays"]),
                    "missing_scalars": ";".join(seg["missing_scalars"]),
                    "invalid_scalars": ";".join(seg["invalid_scalars"]),
                    "missing_stats": ";".join(seg["missing_stats"]),
                    "invalid_stats": ";".join(seg["invalid_stats"]),
                }
                for key in RADIUS_AREA_KEYS:
                    out[f"{key}_valid_n"] = seg["radius_area_valid"].get(key, 0)
                writer.writerow(out)


def write_markdown(rows: list[dict[str, Any]], path: Path, data_root: Path) -> None:
    issue_counts = Counter(issue for row in rows for issue in row["issues"])
    segment_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    radius_failures: list[tuple[str, str, list[str]]] = []
    for row in rows:
        for segment, seg in row["segments"].items():
            for issue in seg["issues"]:
                segment_issue_counts[segment][issue] += 1
            if seg["present"] and any(seg["radius_area_valid"].get(key, 0) == 0 for key in ("area", "eq_diameter", "inscribed_radius")):
                empty = [key for key in RADIUS_AREA_KEYS if seg["radius_area_valid"].get(key, 0) == 0]
                radius_failures.append((row["sample"], segment, empty))

    lines = [
        "# 数据集剖面特征 QC",
        "",
        f"- 数据根目录：`{data_root}`",
        f"- 扫描样本：{len(rows)}",
        f"- 完全通过样本：{sum(row['status'] == 'ok' for row in rows)}",
        f"- 有问题样本：{sum(row['status'] != 'ok' for row in rows)}",
        "",
        "## 检查口径",
        "",
        "- MPV、SMV、SV 为必查血管；名字含 `#` 的样本额外必查 TIPS。",
        "- LPV/RPV/LGV/PGV/TIPS 这类非必需血管：不存在不算错，存在则检查剖面字段。",
        "- 剖面字段来自 `centerline_pointwise_profiles.json`，汇总统计来自 `portal_vein_features.json`。",
        "",
        "## 问题类型计数",
        "",
        "| 问题代码 | 数量 |",
        "|---|---:|",
    ]
    for issue, count in issue_counts.most_common():
        lines.append(f"| `{issue}` | {count} |")

    lines.extend(["", "## 按血管统计", "", "| 血管 | 问题 | 数量 |", "|---|---|---:|"])
    for segment in SEGMENTS:
        for issue, count in segment_issue_counts[segment].most_common():
            lines.append(f"| `{segment}` | `{issue}` | {count} |")

    lines.extend(["", "## 半径/截面积字段为空的血管", "", "| 样本 | 血管 | 空字段 |", "|---|---|---|"])
    for sample, segment, empty in radius_failures:
        lines.append(f"| `{sample}` | `{segment}` | `{';'.join(empty)}` |")

    lines.extend(["", "## 核心血管问题（MPV/SMV/SV）", "", "| 样本 | 血管 | 无效剖面数组 | 无效汇总特征 |", "|---|---|---|---|"])
    for row in rows:
        for segment in CORE_SEGMENTS:
            seg = row["segments"][segment]
            if not seg["issues"]:
                continue
            lines.append(
                f"| `{row['sample']}` | `{segment}` | "
                f"`{';'.join(seg['invalid_arrays'])}` | `{';'.join(seg['invalid_stats'])}` |"
            )

    lines.extend(["", "## 有问题样本", "", "| 样本 | 问题 |", "|---|---|"])
    for row in rows:
        if row["status"] == "ok":
            continue
        lines.append(f"| `{row['sample']}` | `{';'.join(row['issues'])}` |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    patients = sorted(
        [
            p for p in args.data_root.iterdir()
            if p.is_dir() and p.name[:1].isdigit() and (p / "unified_features.json").exists()
        ],
        key=lambda p: p.name.lower(),
    )
    rows = [qc_patient(patient) for patient in patients]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.out_dir / "dataset_profile_feature_qc_report.json"
    csv_path = args.out_dir / "dataset_profile_feature_qc_report.csv"
    md_path = args.out_dir / "dataset_profile_feature_qc_report.md"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, args.data_root)

    print(f"rows={len(rows)}")
    print("status_counts=" + json.dumps(Counter(row["status"] for row in rows), ensure_ascii=False, sort_keys=True))
    print(f"json={json_path.resolve()}")
    print(f"csv={csv_path.resolve()}")
    print(f"markdown={md_path.resolve()}")


if __name__ == "__main__":
    main()
