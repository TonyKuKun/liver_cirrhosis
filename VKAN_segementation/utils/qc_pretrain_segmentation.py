from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # pragma: no cover - optional deep mode dependency
    ndi = None


KEY_SEGMENTATION_FILES = (
    "segmentation/portal_vein.nii.gz",
    "segmentation/totalseg_output/portal_vein_and_splenic_vein.nii.gz",
    "segmentation/liver.nii.gz",
    "segmentation/totalseg_output/liver.nii.gz",
    "segmentation/spleen.nii.gz",
    "segmentation/totalseg_output/spleen.nii.gz",
    "segmentation/bone_all.nii.gz",
    "segmentation/inferior_vena_cava.nii.gz",
    "segmentation/aorta.nii.gz",
)

PORTAL_FILES = (
    "segmentation/portal_vein.nii.gz",
    "segmentation/totalseg_output/portal_vein_and_splenic_vein.nii.gz",
)

FALLBACK_FILES = (
    "segmentation/liver.nii.gz",
    "segmentation/totalseg_output/liver.nii.gz",
    "segmentation/spleen.nii.gz",
    "segmentation/totalseg_output/spleen.nii.gz",
)


@dataclass(frozen=True)
class NiftiHeader:
    path: Path
    shape_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    affine: np.ndarray | None
    qform_code: int
    sform_code: int
    datatype: int
    vox_offset: int
    endian: str

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        x, y, z = self.shape_xyz
        return (z, y, x)


def _open_maybe_gzip(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _read_nifti_header(path: Path) -> NiftiHeader | None:
    if not path.exists():
        return None
    with _open_maybe_gzip(path) as fp:
        raw = fp.read(352)
    if len(raw) < 348:
        return None

    endian = "<"
    if struct.unpack("<i", raw[:4])[0] != 348:
        endian = ">"
        if struct.unpack(">i", raw[:4])[0] != 348:
            return None

    dim = struct.unpack(endian + "8h", raw[40:56])
    ndim = int(dim[0])
    if ndim < 3:
        return None
    shape_xyz = tuple(int(v) for v in dim[1:4])
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    spacing_xyz = tuple(abs(float(v)) for v in pixdim[1:4])
    datatype = int(struct.unpack(endian + "h", raw[70:72])[0])
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    qform_code, sform_code = struct.unpack(endian + "2h", raw[252:256])
    srowx = struct.unpack(endian + "4f", raw[280:296])
    srowy = struct.unpack(endian + "4f", raw[296:312])
    srowz = struct.unpack(endian + "4f", raw[312:328])
    affine = None
    if sform_code > 0:
        affine = np.array([srowx, srowy, srowz, [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return NiftiHeader(
        path=path,
        shape_xyz=shape_xyz,
        spacing_xyz=spacing_xyz,
        affine=affine,
        qform_code=int(qform_code),
        sform_code=int(sform_code),
        datatype=datatype,
        vox_offset=vox_offset,
        endian=endian,
    )


def _affine_max_abs_diff(a: NiftiHeader | None, b: NiftiHeader | None) -> float | None:
    if a is None or b is None or a.affine is None or b.affine is None:
        return None
    return float(np.max(np.abs(a.affine - b.affine)))


def _shape_matches(a: NiftiHeader | None, b: NiftiHeader | None) -> bool | None:
    if a is None or b is None:
        return None
    return a.shape_xyz == b.shape_xyz


def _safe_get(obj: dict[str, Any], *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}g}"


def _read_bool_mask(path: Path, header: NiftiHeader) -> np.ndarray:
    dtype_map = {
        2: "u1",
        4: "i2",
        8: "i4",
        16: "f4",
        64: "f8",
        256: "i1",
        512: "u2",
        768: "u4",
    }
    dtype_code = dtype_map.get(header.datatype)
    if dtype_code is None:
        raise ValueError(f"Unsupported NIfTI datatype {header.datatype}: {path}")
    with _open_maybe_gzip(path) as fp:
        raw = fp.read()
    dtype = np.dtype(header.endian + dtype_code)
    arr = np.frombuffer(raw, dtype=dtype, offset=header.vox_offset)
    return arr.reshape(header.shape_xyz, order="F") != 0


def _deep_pretrain_stats(patient_dir: Path, max_components_voxels: int) -> dict[str, Any]:
    path = patient_dir / "pretrain.nii.gz"
    header = _read_nifti_header(path)
    if header is None:
        return {"deep_error": "missing_pretrain_nii"}
    try:
        mask = _read_bool_mask(path, header)
    except Exception as exc:  # pragma: no cover - defensive report path
        return {"deep_error": str(exc)}

    voxels = int(mask.sum())
    stats: dict[str, Any] = {"deep_voxels": voxels}
    if voxels:
        coords = np.argwhere(mask)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        stats["deep_bbox_xyz"] = f"{tuple(int(v) for v in (maxs - mins))}"
    if ndi is not None and 0 < voxels <= max_components_voxels:
        labels, ncomp = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        counts = np.bincount(labels.ravel())[1:]
        largest = int(counts.max()) if counts.size else 0
        stats["deep_components"] = int(ncomp)
        stats["deep_largest_ratio"] = round(largest / voxels, 4) if voxels else 0.0
    elif voxels > max_components_voxels:
        stats["deep_components"] = "skipped_large_mask"
    elif ndi is None:
        stats["deep_components"] = "scipy_missing"
    return stats


def _collect_header_issues(patient_dir: Path, affine_tolerance: float) -> tuple[list[str], list[str], dict[str, Any]]:
    orig = _read_nifti_header(patient_dir / "orig.nii.gz")
    pretrain = _read_nifti_header(patient_dir / "pretrain.nii.gz")
    label = _read_nifti_header(patient_dir / "mask.nii.gz")
    categories: list[str] = []
    suspect_files: list[str] = []
    details: dict[str, Any] = {}

    details["orig_shape_xyz"] = str(orig.shape_xyz) if orig else ""
    details["pretrain_shape_xyz"] = str(pretrain.shape_xyz) if pretrain else ""
    details["mask_shape_xyz"] = str(label.shape_xyz) if label else ""
    details["pretrain_orig_affine_diff"] = _fmt_float(_affine_max_abs_diff(pretrain, orig))
    details["mask_orig_affine_diff"] = _fmt_float(_affine_max_abs_diff(label, orig))

    if orig is None:
        categories.append("missing_orig")
    if pretrain is None:
        categories.append("missing_pretrain")
    if label is None:
        categories.append("missing_mask")
    if _shape_matches(pretrain, orig) is False:
        categories.append("pretrain_shape_mismatch")
        suspect_files.append("pretrain.nii.gz")
    if _shape_matches(label, orig) is False:
        categories.append("label_shape_mismatch")
        suspect_files.append("mask.nii.gz")

    pretrain_affine_diff = _affine_max_abs_diff(pretrain, orig)
    if pretrain_affine_diff is not None and pretrain_affine_diff > affine_tolerance:
        categories.append("pretrain_affine_mismatch")
        suspect_files.append("pretrain.nii.gz")

    bad_seg: list[str] = []
    bad_affine_count = 0
    bad_shape_count = 0
    for rel in KEY_SEGMENTATION_FILES:
        path = patient_dir / rel
        header = _read_nifti_header(path)
        if header is None:
            continue
        shape_ok = _shape_matches(header, orig)
        aff_diff = _affine_max_abs_diff(header, orig)
        issues: list[str] = []
        if shape_ok is False:
            bad_shape_count += 1
            issues.append(f"shape={header.shape_xyz}")
            suspect_files.append(rel)
        if aff_diff is not None and aff_diff > affine_tolerance:
            bad_affine_count += 1
            issues.append(f"affine_diff={aff_diff:.3g}")
        if issues:
            bad_seg.append(f"{rel} ({', '.join(issues)})")
    if bad_shape_count:
        categories.append("seg_shape_mismatch")
    if bad_affine_count:
        categories.append("seg_affine_mismatch")
    details["seg_header_issues"] = "; ".join(bad_seg)
    return categories, suspect_files, details


def _classify_from_meta(meta: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    categories: list[str] = []
    suspect_files: list[str] = []
    details: dict[str, Any] = {}

    voxels = _safe_get(meta, "mask_voxels")
    dice = _safe_get(meta, "pretrain_vessel_eval", "dice")
    precision = _safe_get(meta, "pretrain_vessel_eval", "precision")
    recall = _safe_get(meta, "pretrain_vessel_eval", "recall")
    label_voxels = _safe_get(meta, "pretrain_vessel_eval", "label_voxels")
    portal_voxels = _safe_get(meta, "hu_sampling", "voxels")
    portal_p50 = _safe_get(meta, "hu_sampling", "p50")
    portal_p90 = _safe_get(meta, "hu_sampling", "p90")
    hu_low = _safe_get(meta, "hu_sampling", "hu_low")
    hu_high = _safe_get(meta, "hu_sampling", "hu_high")
    region_in = _safe_get(meta, "region_grow", "input_voxels")
    region_out = _safe_get(meta, "region_grow", "output_voxels")
    region_components = _safe_get(meta, "region_grow", "components_total")
    seed_distance = _safe_get(meta, "region_grow", "nearest_seed_distance_mm")
    fallback_status = _safe_get(meta, "liver_spleen_fallback", "status")

    details.update(
        {
            "pretrain_voxels": voxels,
            "dice": _fmt_float(dice),
            "precision": _fmt_float(precision),
            "recall": _fmt_float(recall),
            "label_voxels": label_voxels,
            "portal_voxels": portal_voxels,
            "portal_p50": _fmt_float(portal_p50),
            "portal_p90": _fmt_float(portal_p90),
            "hu_low": _fmt_float(hu_low),
            "hu_high": _fmt_float(hu_high),
            "region_input_voxels": region_in,
            "region_output_voxels": region_out,
            "region_components_total": region_components,
            "nearest_seed_distance_mm": _fmt_float(seed_distance),
            "fallback_status": fallback_status or "",
        }
    )

    if portal_voxels is not None and portal_voxels < 1000:
        categories.append("portal_empty_or_tiny")
        suspect_files.extend(PORTAL_FILES)
    elif portal_p50 is not None and portal_p50 < 100:
        categories.append("portal_low_hu_unreliable")
        suspect_files.extend(PORTAL_FILES)

    if voxels is not None and voxels <= 1500:
        categories.append("final_tiny")
    if voxels is not None and voxels >= 700000:
        categories.append("final_huge_leak")
    if dice is not None and dice == 0:
        categories.append("zero_dice")
    elif dice is not None and dice < 0.25:
        categories.append("very_low_dice")
    elif dice is not None and dice < 0.5:
        categories.append("low_dice")

    if fallback_status in {"missing_liver_or_spleen", "empty_candidate", "no_anatomic_component"}:
        categories.append("fallback_failed")
        suspect_files.extend(FALLBACK_FILES)

    return categories, suspect_files, details


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _load_meta(patient_dir: Path) -> dict[str, Any]:
    path = patient_dir / "vkan_work" / "pretrain_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"meta_error": "invalid_json"}


def qc_patient(patient_dir: Path, affine_tolerance: float, deep_mask_stats: bool, max_components_voxels: int) -> dict[str, Any]:
    meta = _load_meta(patient_dir)
    meta_categories, meta_suspects, meta_details = _classify_from_meta(meta)
    header_categories, header_suspects, header_details = _collect_header_issues(patient_dir, affine_tolerance)

    categories = _dedupe(meta_categories + header_categories)
    suspects = _dedupe(meta_suspects + header_suspects)
    if not categories:
        categories = ["no_obvious_issue_from_qc"]

    row: dict[str, Any] = {
        "patient": patient_dir.name,
        "category": "; ".join(categories),
        "suspect_files": "; ".join(suspects),
        **meta_details,
        **header_details,
    }
    if meta.get("meta_error"):
        row["meta_error"] = meta["meta_error"]
    if deep_mask_stats:
        row.update(_deep_pretrain_stats(patient_dir, max_components_voxels))
    return row


def _iter_patients(data_root: Path, marker: str, include_all: bool) -> list[Path]:
    if (data_root / "orig.nii.gz").exists():
        return [data_root]
    patients = [p for p in data_root.iterdir() if p.is_dir()]
    if include_all:
        return sorted(patients, key=lambda p: p.name)
    return sorted((p for p in patients if marker in p.name), key=lambda p: p.name)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(rows: list[dict[str, Any]], path: Path, data_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Pretrain segmentation QC",
        "",
        f"Data root: `{data_root}`",
        f"Patients checked: {len(rows)}",
        "",
        "| Patient | Category | Key Evidence | Suspect files |",
        "|---|---|---|---|",
    ]
    for row in rows:
        evidence = (
            f"vox={row.get('pretrain_voxels', '')}, "
            f"dice={row.get('dice', '')}, "
            f"portal_vox={row.get('portal_voxels', '')}, "
            f"portal_p50={row.get('portal_p50', '')}, "
            f"rg={row.get('region_input_voxels', '')}->{row.get('region_output_voxels', '')}, "
            f"fallback={row.get('fallback_status', '')}"
        )
        suspect = str(row.get("suspect_files", "")).replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("patient", "")).replace("|", "\\|"),
                    str(row.get("category", "")).replace("|", "\\|"),
                    evidence.replace("|", "\\|"),
                    suspect,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Header issues")
    lines.append("")
    for row in rows:
        issue = row.get("seg_header_issues")
        if issue:
            lines.append(f"- `{row['patient']}`: {issue}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only QC for $-marked patients with suspicious pretrain/segmentation outputs."
    )
    parser.add_argument("--data-root", default=r'F:\PCG data\dataset\test4all_sample', type=Path, help="Dataset root or one patient folder.")
    parser.add_argument("--marker", default="$", help="Only scan patient folders containing this marker.")
    parser.add_argument("--include-all", action="store_true", help="Scan all patient folders instead of only marker folders.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for qc reports. Defaults to data root.")
    parser.add_argument("--prefix", default="pretrain_seg_qc", help="Output filename prefix.")
    parser.add_argument("--affine-tolerance", type=float, default=1e-3, help="Max affine absolute difference before flagging.")
    parser.add_argument("--deep-mask-stats", action="store_true", help="Read pretrain.nii.gz and compute bbox/components.")
    parser.add_argument(
        "--max-components-voxels",
        type=int,
        default=900_000,
        help="Skip connected components above this voxel count in deep mode.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    output_dir = args.output_dir or data_root
    patients = _iter_patients(data_root, args.marker, args.include_all)
    rows = [
        qc_patient(p, args.affine_tolerance, args.deep_mask_stats, args.max_components_voxels)
        for p in patients
    ]

    csv_path = output_dir / f"{args.prefix}.csv"
    json_path = output_dir / f"{args.prefix}.json"
    md_path = output_dir / f"{args.prefix}.md"
    _write_csv(rows, csv_path)
    _write_json(rows, json_path)
    _write_markdown(rows, md_path, data_root)

    print(f"checked_patients={len(rows)}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    if rows:
        counts: dict[str, int] = {}
        for row in rows:
            for category in str(row.get("category", "")).split("; "):
                counts[category] = counts.get(category, 0) + 1
        print("category_counts=" + json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
