"""One-time migration from root-level geometry outputs to ``features/``.

The migration is intentionally strict: canonical files are never silently
overwritten with different content, and manual assignments/ranges are merged
into ``segment_assignments.json`` before old generated outputs are removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from features_layout import (
    RAW_CENTERLINE_NAME,
    SEGMENT_ASSIGNMENTS_NAME,
    SMOOTH_CENTERLINE_NAME,
    UNIFIED_FEATURES_NAME,
    features_dir,
)


OLD_CORE_FILES = {
    "CenterlinePoints.txt": RAW_CENTERLINE_NAME,
    "newCenterlist.txt": SMOOTH_CENTERLINE_NAME,
    "centerline_profiles.json": SEGMENT_ASSIGNMENTS_NAME,
    "unified_features.json": UNIFIED_FEATURES_NAME,
}

OLD_DERIVED_FILES = (
    "manual_segment_assignments.json",
    "analysis_ranges.json",
    "centerline_pointwise_profiles.json",
    "portal_vein_features.json",
    "feature_description.json",
    "sv_smv_angle.json",
    "vis_interactive.html",
    "vis_overview.png",
    "centerline_screenshot.png",
    "segment_screenshot.png",
)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _merge_segment_metadata(patient: Path, target: Path) -> bool:
    source = patient / "centerline_profiles.json"
    if target.exists():
        data = _read_json(target)
    elif source.exists():
        data = _read_json(source)
    else:
        return False

    manual = patient / "manual_segment_assignments.json"
    if manual.exists():
        manual_data = _read_json(manual)
        assignments = manual_data.get("assignments")
        if isinstance(assignments, dict):
            data["assignments"] = assignments
            data["manual_assignment"] = True
            data["manual_assignment_version"] = int(manual_data.get("version") or 1)

    ranges = patient / "analysis_ranges.json"
    if ranges.exists():
        ranges_data = _read_json(ranges)
        values = ranges_data.get("ranges")
        if isinstance(values, dict):
            data["analysis_ranges"] = values

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return True


def migrate_patient(patient: Path, execute: bool) -> dict:
    target_dir = features_dir(patient, create=execute)
    result = {"patient": patient.name, "moved": [], "removed": [], "missing": []}

    if not execute:
        for old_name, new_name in OLD_CORE_FILES.items():
            if (patient / old_name).exists():
                result["moved"].append(f"{old_name} -> features/{new_name}")
            elif not (target_dir / new_name).exists():
                result["missing"].append(new_name)
        return result

    segment_target = target_dir / SEGMENT_ASSIGNMENTS_NAME
    _merge_segment_metadata(patient, segment_target)

    for old_name, new_name in OLD_CORE_FILES.items():
        source = patient / old_name
        target = target_dir / new_name
        if old_name == "centerline_profiles.json":
            if source.exists():
                source.unlink()
                result["removed"].append(old_name)
            if target.exists():
                result["moved"].append(f"{old_name} -> features/{new_name}")
            else:
                result["missing"].append(new_name)
            continue
        if source.exists():
            if target.exists() and _digest(source) != _digest(target):
                raise RuntimeError(f"Conflicting old/new files: {source} and {target}")
            if not target.exists():
                shutil.copy2(source, target)
            source.unlink()
            result["moved"].append(f"{old_name} -> features/{new_name}")
        elif not target.exists():
            result["missing"].append(new_name)

    for name in OLD_DERIVED_FILES:
        path = patient / name
        if path.exists():
            path.unlink()
            result["removed"].append(name)

    for path in target_dir.iterdir():
        if path.is_file() and path.name not in {
            RAW_CENTERLINE_NAME,
            SMOOTH_CENTERLINE_NAME,
            SEGMENT_ASSIGNMENTS_NAME,
            UNIFIED_FEATURES_NAME,
        }:
            path.unlink()
            result["removed"].append(f"features/{path.name}")
    return result


def migrate_root(root: Path, execute: bool) -> list[dict]:
    patients = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and re.match(r"^\d", path.name)
    ]
    return [migrate_patient(patient, execute=execute) for patient in patients]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = migrate_root(root, execute=args.execute)
    summary = {
        "mode": "execute" if args.execute else "dry-run",
        "root": str(root),
        "patients": len(rows),
        "moved": sum(len(row["moved"]) for row in rows),
        "removed": sum(len(row["removed"]) for row in rows),
        "missing_by_file": {
            name: sum(name in row["missing"] for row in rows)
            for name in (
                RAW_CENTERLINE_NAME,
                SMOOTH_CENTERLINE_NAME,
                SEGMENT_ASSIGNMENTS_NAME,
                UNIFIED_FEATURES_NAME,
            )
        },
        "rows": rows,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
