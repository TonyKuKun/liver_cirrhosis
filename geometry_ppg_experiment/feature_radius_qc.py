"""QC radius and area availability behind geometry feature extraction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import extract_features as ef


DEFAULT_DATA_ROOT = Path(r"F:\PCG data\dataset\test4all_sample")
SEGMENTS = ("mpv", "smv", "sv", "lpv", "rpv", "tips", "lgv", "pgv")
REQUIRED_RADIUS_SEGMENTS = ("mpv", "smv", "sv")


def valid_area_stats(seg_data: dict[str, Any]) -> dict[str, Any]:
    raw_area = ef.finite_array(seg_data.get("area"))
    raw_valid = raw_area[np.isfinite(raw_area) & (raw_area > 0)]
    area = ef.area_profile(seg_data)
    valid = area[np.isfinite(area) & (area > 0)]
    if valid.size:
        radii = np.sqrt(valid / math.pi)
        return {
            "raw_area_n": int(raw_area.size),
            "raw_area_valid_n": int(raw_valid.size),
            "area_n": int(area.size),
            "area_valid_n": int(valid.size),
            "radius_median": float(np.nanmedian(radii)),
            "radius_min": float(np.nanmin(radii)),
            "radius_max": float(np.nanmax(radii)),
        }
    return {
        "raw_area_n": int(raw_area.size),
        "raw_area_valid_n": int(raw_valid.size),
        "area_n": int(area.size),
        "area_valid_n": 0,
        "radius_median": math.nan,
        "radius_min": math.nan,
        "radius_max": math.nan,
    }


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
    row: dict[str, Any] = {
        "sample": patient_dir.name,
        "has_unified": (patient_dir / "unified_features.json").exists(),
        "is_post_tips_by_name": "#" in patient_dir.name,
        "issues": [],
        "segments": {},
        "features": {},
    }

    for seg in SEGMENTS:
        present = ef.segment_present(sources, seg)
        stats = valid_area_stats(ef.pointwise_segment(sources, seg))
        stats["present"] = bool(present)
        row["segments"][seg] = stats
        if seg in REQUIRED_RADIUS_SEGMENTS:
            if not present:
                row["issues"].append(f"{seg}_missing")
            elif stats["area_valid_n"] == 0:
                row["issues"].append(f"{seg}_missing_valid_area")

    if row["is_post_tips_by_name"]:
        stats = row["segments"]["tips"]
        if not stats["present"]:
            row["issues"].append("tips_missing_for_hash_sample")
        elif stats["area_valid_n"] == 0:
            row["issues"].append("tips_missing_valid_area")

    feature_fns = {
        "R_total": ef.compute_r_total,
        "D_Murray": ef.compute_d_murray,
        "R_collateral": ef.compute_r_collateral,
        "Ratio_SMV_SV": ef.compute_ratio_smv_sv,
        "theta_SMV_SV": ef.compute_theta_smv_sv,
        "Ratio_LPV_RPV": ef.compute_ratio_lpv_rpv,
    }
    for name, fn in feature_fns.items():
        value, report = fn(sources)
        row["features"][name] = {
            "finite": bool(np.isfinite(value)),
            "value": float(value) if np.isfinite(value) else math.nan,
            "status": report.get("status") if isinstance(report, dict) else None,
        }
        if name in ("R_total", "D_Murray", "Ratio_SMV_SV") and not np.isfinite(value):
            row["issues"].append(f"{name}_not_finite:{row['features'][name]['status']}")

    row["status"] = "ok" if not row["issues"] else "problem"
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "sample",
        "status",
        "issues",
        "mpv_area_valid_n",
        "smv_area_valid_n",
        "sv_area_valid_n",
        "tips_area_valid_n",
        "mpv_raw_area_valid_n",
        "smv_raw_area_valid_n",
        "sv_raw_area_valid_n",
        "tips_raw_area_valid_n",
        "mpv_radius_median",
        "smv_radius_median",
        "sv_radius_median",
        "tips_radius_median",
        "R_total_status",
        "D_Murray_status",
        "Ratio_SMV_SV_status",
        "R_collateral_status",
        "Ratio_LPV_RPV_status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {
                "sample": row["sample"],
                "status": row["status"],
                "issues": ";".join(row["issues"]),
            }
            for seg in ("mpv", "smv", "sv", "tips"):
                stats = row["segments"][seg]
                out[f"{seg}_area_valid_n"] = stats["area_valid_n"]
                out[f"{seg}_raw_area_valid_n"] = stats["raw_area_valid_n"]
                out[f"{seg}_radius_median"] = stats["radius_median"]
            for feature in ("R_total", "D_Murray", "Ratio_SMV_SV", "R_collateral", "Ratio_LPV_RPV"):
                out[f"{feature}_status"] = row["features"][feature]["status"]
            writer.writerow(out)


def write_markdown(rows: list[dict[str, Any]], path: Path, data_root: Path) -> None:
    issue_counts = Counter(issue for row in rows for issue in row["issues"])
    feature_counts = {
        feature: Counter(row["features"][feature]["status"] for row in rows)
        for feature in ("R_total", "D_Murray", "Ratio_SMV_SV", "R_collateral", "Ratio_LPV_RPV")
    }
    lines = [
        "# 特征半径/面积 QC",
        "",
        f"- 数据根目录：`{data_root}`",
        f"- 扫描已分段样本：{len(rows)}",
        f"- MPV/SMV/SV 缺有效面积序列：{sum(any(issue.endswith('_missing_valid_area') for issue in row['issues']) for row in rows)}",
        "",
        "## 问题类型计数",
        "",
        "| 问题 | 样本数 |",
        "|---|---:|",
    ]
    for issue, count in issue_counts.most_common():
        lines.append(f"| `{issue}` | {count} |")

    lines.extend(["", "## 特征状态计数", ""])
    for feature, counts in feature_counts.items():
        lines.append(f"### {feature}")
        lines.append("")
        lines.append("| status | 样本数 |")
        lines.append("|---|---:|")
        for status, count in counts.most_common():
            lines.append(f"| `{status}` | {count} |")
        lines.append("")

    lines.extend(["## 需检查样本", "", "| 样本 | 问题 |", "|---|---|"])
    for row in rows:
        if row["status"] == "ok":
            continue
        lines.append(f"| `{row['sample']}` | {'; '.join(row['issues'])} |")
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
    json_path = args.out_dir / "feature_radius_qc_report.json"
    csv_path = args.out_dir / "feature_radius_qc_report.csv"
    md_path = args.out_dir / "feature_radius_qc_report.md"
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
