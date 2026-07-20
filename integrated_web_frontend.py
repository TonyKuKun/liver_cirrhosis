"""Integrated PortaFlow web server.

Run:
    python integrated_web_frontend.py --host 127.0.0.1 --port 8788
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

_GEOMETRY_IMPORT_ROOT = Path(__file__).resolve().parent / "geometry_feature_extract"
if str(_GEOMETRY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEOMETRY_IMPORT_ROOT))

from geometry_feature_extract import web_frontend as geometry_web
from geometry_feature_extract.features_layout import (
    FEATURES_DIRNAME,
    PUBLIC_FEATURE_NAMES,
    RAW_CENTERLINE_NAME,
    SMOOTH_CENTERLINE_NAME,
    SEGMENT_ASSIGNMENTS_NAME,
    UNIFIED_FEATURES_NAME,
    remove_generated_outputs,
    resolve_feature_path,
)


APP_ROOT = Path(__file__).resolve().parent
WEB_ROOT = APP_ROOT / "web"
WEB_CONFIG_PATH = WEB_ROOT / "web_modules.json"


def _load_web_config() -> dict:
    if not WEB_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(WEB_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cfg_path(value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback.resolve()
    path = Path(str(value))
    if not path.is_absolute():
        path = APP_ROOT / path
    return path.resolve()


WEB_CONFIG = _load_web_config()
STAGE_CONFIG = WEB_CONFIG.get("stages") if isinstance(WEB_CONFIG.get("stages"), dict) else {}
STATIC_ROOT = _cfg_path((WEB_CONFIG.get("integrated") or {}).get("static_root"), WEB_ROOT)
VKAN_ROOT = _cfg_path((STAGE_CONFIG.get("segmentation") or {}).get("root"), APP_ROOT / "VKAN_segementation")
CENTERLINE_ROOT = _cfg_path((STAGE_CONFIG.get("features") or {}).get("root"), APP_ROOT / "geometry_feature_extract")
GEOMETRY_STATIC_ROOT = WEB_ROOT / "geometry"
PVP_ROOT = _cfg_path((STAGE_CONFIG.get("pvp") or {}).get("root"), APP_ROOT / "PVP_predictor")

STAGES = ["segmentation", "features", "pvp"]
STAGE_LABELS = {
    "segmentation": "CT segmentation",
    "features": "Centerline geometry",
    "pvp": "PVP inference",
}

OUTPUTS = [
    "dcm",
    "orig.nii.gz",
    "pretrain.stl",
    "predict.stl",
    "predict_smooth.stl",
    "vessel.stl",
    *PUBLIC_FEATURE_NAMES,
    "PVP_predict.txt",
    "pvp_prediction.json",
]

FILE_ALIASES = {
    "pretrain.stl": [
        "pretrain.stl",
        "pre.stl",
        "vkan_work/pretrain_round1.stl",
        "vkan_work/pretrain_round0.stl",
    ],
    "predict.stl": ["predict.stl"],
    "predict_smooth.stl": ["predict_smooth.stl", "predict_smoothed.stl", "smooth_predict.stl"],
    "vessel.stl": ["vessel.stl", "segmentation/portal_vein.stl"],
}

SESSIONS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
GEOMETRY_STL_CACHE: dict[str, tuple[tuple, Path]] = {}
GEOMETRY_STL_CACHE_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def _runtime() -> dict:
    return {
        "python": sys.executable,
        "python_prefix": sys.prefix,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV") or "",
        "web_root": str(STATIC_ROOT),
        "web_config": str(WEB_CONFIG_PATH),
        "vkan_root": str(VKAN_ROOT),
        "centerline_root": str(CENTERLINE_ROOT),
        "geometry_static_root": str(GEOMETRY_STATIC_ROOT),
        "pvp_root": str(PVP_ROOT),
        "pvp_python": str(_pvp_python()),
    }


def _pvp_python() -> Path:
    pvp_config = STAGE_CONFIG.get("pvp") or {}
    configured = _cfg_path(pvp_config.get("python"), Path(sys.executable))
    return configured if configured.exists() else Path(sys.executable)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_label_value(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
    if not text:
        return ""
    return " ".join(text.split())[:120]


PATIENT_META_ALIASES = {
    "age": {"age", "年龄", "岁数"},
    "sex": {"sex", "gender", "性别", "patientsex", "患者性别"},
    "birth_date": {"birthdate", "dateofbirth", "出生日期", "生日", "patientbirthdate", "患者出生日期"},
    "primary_disease": {"primarydisease", "disease", "diagnosis", "基础疾病", "原发病", "诊断"},
    "symptoms": {"symptoms", "symptom", "complications", "complication", "并发症", "症状"},
    "shunt_type": {"shunttype", "tipstype", "分流类型", "手术类型"},
    "exam_date": {"examdate", "checkdate", "studydate", "检查日期", "检查时间"},
    "surgery_date": {"surgerydate", "operationdate", "tipsdate", "手术日期", "手术时间"},
    "measured_pvp": {"pvp", "pressure", "measuredpvp", "门静脉压力", "门静脉压"},
}


def _meta_key(value: str) -> str | None:
    normalized = re.sub(r"[\s_\-./()（）]+", "", str(value or "").strip().lower())
    for key, aliases in PATIENT_META_ALIASES.items():
        if normalized in aliases:
            return key
    return None


def _metadata_from_mapping(data: dict) -> dict:
    meta = {}
    for raw_key, raw_value in data.items():
        key = _meta_key(str(raw_key))
        if key and raw_value not in (None, ""):
            meta[key] = " ".join(str(raw_value).split())[:120]
    return meta


def _read_patient_info(path: Path) -> dict:
    if not path.exists() or not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore").strip()
    except Exception:
        return {}
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _metadata_from_mapping(data)
    except json.JSONDecodeError:
        pass

    meta = {}
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if not line:
            continue
        parts = re.split(r"\s*[:：=\t]\s*", line, maxsplit=1)
        if len(parts) == 1:
            parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) != 2:
            continue
        key = _meta_key(parts[0])
        value = parts[1].strip()
        if key and value:
            meta[key] = " ".join(value.split())[:120]
    return meta


def _parse_date(value: str | None) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw[:10] if fmt != "%Y%m%d" else raw[:8], fmt).date()
        except ValueError:
            pass
    match = re.search(r"(20\d{2}|19\d{2})[-_/年.]?(\d{1,2})[-_/月.]?(\d{1,2})", raw)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _folder_exam_date(path: Path) -> date | None:
    match = re.match(r"(\d{8})", path.name)
    return _parse_date(match.group(1)) if match else None


def _patient_match_key(name: str) -> str:
    text = re.sub(r"^\d{8}", "", name).replace("#", "")
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text.lower())


def _numeric_text(value: str | None) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _read_pvp_value(patient_dir: Path) -> float | None:
    prediction = _read_json(patient_dir / "pvp_prediction.json")
    if isinstance(prediction, dict):
        value = prediction.get("pvp_mean")
        if isinstance(value, (int, float)):
            return float(value)
    predict_txt = patient_dir / "PVP_predict.txt"
    if predict_txt.exists():
        try:
            text = predict_txt.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"pvp_mean_mmHg\s*:\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        except Exception:
            pass
    return _numeric_text(_read_label_value(patient_dir / "label" / "PVP.txt"))


def _patient_label_metadata(path: Path) -> dict:
    label_dir = path / "label"
    meta = _read_patient_info(path / "patient_info.txt")
    if not label_dir.exists():
        return meta
    fields = {
        "age": ["age.txt", "Age.txt", "AGE.txt", "年龄.txt"],
        "sex": ["sex.txt", "Sex.txt", "gender.txt", "Gender.txt", "性别.txt"],
        "primary_disease": ["primary_disease.txt", "disease.txt", "diagnosis.txt", "基础疾病.txt", "原发病.txt"],
        "symptoms": ["symptoms.txt", "complications.txt", "complication.txt", "并发症.txt", "症状.txt"],
        "shunt_type": ["shunt_type.txt", "tips_type.txt", "分流类型.txt"],
        "exam_date": ["exam_date.txt", "check_date.txt", "study_date.txt", "检查日期.txt"],
        "surgery_date": ["surgery_date.txt", "operation_date.txt", "tips_date.txt", "手术日期.txt"],
        "measured_pvp": ["PVP.txt", "pvp.txt", "pressure.txt", "门静脉压力.txt"],
    }
    for key, names in fields.items():
        for name in names:
            value = _read_label_value(label_dir / name)
            if value:
                meta[key] = value
                break
    for file_path in label_dir.glob("*.txt"):
        key = _meta_key(file_path.stem)
        if key and key not in meta:
            value = _read_label_value(file_path)
            if value:
                meta[key] = value
    for structured in ["clinical.json", "patient.json", "metadata.json", "label.json"]:
        data = _read_json(label_dir / structured)
        if isinstance(data, dict):
            for key, value in _metadata_from_mapping(data).items():
                if key not in meta:
                    meta[key] = value
    if not meta.get("age") and meta.get("birth_date"):
        birth_date = _parse_date(meta.get("birth_date"))
        exam_date = _parse_date(meta.get("exam_date")) or _folder_exam_date(path)
        if birth_date and exam_date:
            age = exam_date.year - birth_date.year - (
                (exam_date.month, exam_date.day) < (birth_date.month, birth_date.day)
            )
            if 0 <= age <= 120:
                meta["age"] = str(age)
    return meta


def _find_preop_patient(patient: Path, root: Path | None) -> Path | None:
    if not root or not root.exists() or "#" not in patient.name:
        return None
    target_key = _patient_match_key(patient.name)
    candidates = []
    for item in root.iterdir():
        if item == patient or not _valid_patient(item) or "#" in item.name:
            continue
        if _patient_match_key(item.name) == target_key:
            candidates.append(item)
    if not candidates:
        return None
    exam_date = _folder_exam_date(patient)
    if exam_date:
        dated = [(p, _folder_exam_date(p)) for p in candidates]
        before = [(p, d) for p, d in dated if d and d <= exam_date]
        if before:
            return sorted(before, key=lambda item: item[1], reverse=True)[0][0]
    return sorted(candidates, key=lambda p: p.name)[0]


def _clinical_context(patient: Path, root: Path | None = None) -> dict:
    meta = _patient_label_metadata(patient)
    exam_date = _parse_date(meta.get("exam_date")) or _folder_exam_date(patient)
    surgery_date = _parse_date(meta.get("surgery_date"))
    days_from_surgery = (exam_date - surgery_date).days if exam_date and surgery_date else None
    timing = "术后" if patient.name.endswith("#") else "术前"
    if days_from_surgery is not None:
        if days_from_surgery > 0:
            timing = f"术后第 {days_from_surgery} 天"
        elif days_from_surgery == 0:
            timing = "手术当天"
        else:
            timing = f"术前 {abs(days_from_surgery)} 天"

    current_pvp = _read_pvp_value(patient)
    preop = _find_preop_patient(patient, root)
    preop_pvp = _read_pvp_value(preop) if preop else None
    pressure_drop = preop_pvp - current_pvp if preop_pvp is not None and current_pvp is not None else None
    pressure_drop_pct = pressure_drop / preop_pvp * 100 if pressure_drop is not None and preop_pvp else None
    return {
        "exam_date": exam_date.isoformat() if exam_date else "",
        "surgery_date": surgery_date.isoformat() if surgery_date else meta.get("surgery_date", ""),
        "days_from_surgery": days_from_surgery,
        "timing": timing,
        "is_post_tips": "#" in patient.name,
        "measured_pvp": _numeric_text(meta.get("measured_pvp")),
        "current_pvp": current_pvp,
        "preop_match": {
            "id": preop.name,
            "folder": str(preop),
            "pvp": preop_pvp,
            "exam_date": (_folder_exam_date(preop).isoformat() if _folder_exam_date(preop) else ""),
        } if preop else None,
        "pressure_drop": pressure_drop,
        "pressure_drop_pct": pressure_drop_pct,
    }


def _file_info(path: Path) -> dict:
    return {
        "exists": path.exists(),
        "path": str(path),
        "size": path.stat().st_size if path.exists() else 0,
        "modified": path.stat().st_mtime if path.exists() else None,
        "is_dir": path.is_dir() if path.exists() else False,
    }


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_patient_file(patient: Path, rel: str, recursive: bool = True) -> Path | None:
    patient = patient.resolve()
    normalized = str(rel or "").replace("\\", "/").strip("/")
    if not normalized:
        return None
    direct = (patient / normalized).resolve()
    if _inside(patient, direct) and direct.exists():
        return direct

    aliases = FILE_ALIASES.get(normalized.lower(), [normalized])
    for alias in aliases:
        candidate = (patient / alias).resolve()
        if _inside(patient, candidate) and candidate.exists():
            return candidate

    if "/" in normalized or not recursive:
        return None
    target = Path(normalized).name.lower()
    matches = [
        path for path in patient.rglob("*")
        if path.name.lower() == target and path.exists()
    ]
    if not matches:
        return None

    def priority(path: Path) -> tuple[int, int, str]:
        rel_path = path.relative_to(patient)
        parts = [part.lower() for part in rel_path.parts]
        if len(parts) == 1:
            group = 0
        elif parts[0] == "vkan_work":
            group = 1
        elif parts[0] == "segmentation":
            group = 2
        else:
            group = 3
        return (group, len(parts), str(rel_path).lower())

    return sorted(matches, key=priority)[0]


def _patient_file_info(patient: Path, rel: str, recursive: bool = False) -> dict:
    path = _resolve_patient_file(patient, rel, recursive=recursive) or (patient / rel)
    info = _file_info(path)
    info["requested"] = rel
    return info


def _resolve_feature_file(patient: Path, name: str) -> Path | None:
    if name not in PUBLIC_FEATURE_NAMES:
        raise ValueError(f"Unsupported geometry feature file: {name}")
    return resolve_feature_path(patient.resolve(), name)


def _output_file_info(patient: Path, name: str, recursive: bool = True) -> dict:
    if name in PUBLIC_FEATURE_NAMES:
        path = _resolve_feature_file(patient, name) or (patient / FEATURES_DIRNAME / name)
        info = _file_info(path)
        info["requested"] = name
        return info
    return _patient_file_info(patient, name, recursive=recursive)


def _looks_like_patient(path: Path) -> bool:
    return any([
        (path / "dcm").is_dir(),
        (path / "orig.nii.gz").exists(),
        _resolve_patient_file(path, "pretrain.stl", recursive=False) is not None,
        _resolve_patient_file(path, "predict.stl", recursive=False) is not None,
        _resolve_patient_file(path, "predict_smooth.stl", recursive=False) is not None,
        _resolve_patient_file(path, "vessel.stl", recursive=False) is not None,
        _resolve_feature_file(path, UNIFIED_FEATURES_NAME) is not None,
    ])


def _valid_patient(path: Path) -> bool:
    return path.is_dir() and not any(x in path.name for x in ("@", "!", "&")) and _looks_like_patient(path)


def _discover_patients(root: Path) -> list[dict]:
    root = root.resolve()
    if _valid_patient(root):
        return [_patient_record(root, include_label=False)]
    if not root.exists():
        return []
    return [_patient_record(p, include_label=False) for p in sorted(root.iterdir(), key=lambda x: x.name.lower()) if _valid_patient(p)]


def _patient_record(path: Path, include_label: bool = True) -> dict:
    path = path.resolve()
    return {
        "id": path.name,
        "name": path.name,
        "folder": str(path),
        "is_post_tips": "#" in path.name,
        "label_meta": _patient_label_metadata(path) if include_label else {},
    }


def _model_dir(explicit: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    pvp_config = STAGE_CONFIG.get("pvp") or {}
    configured_default = _cfg_path(pvp_config.get("default_model_dir"), PVP_ROOT / "runs" / "final_20260609_pvp_l2_shunt")
    runs = _cfg_path(pvp_config.get("runs"), PVP_ROOT / "runs")
    if runs.exists():
        candidates.extend(sorted([p for p in runs.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True))
    candidates.extend([
        configured_default,
        runs / "final_20260610_pvp_l2_shunt",
        runs / "final_20260609_pvp_l2_shunt",
    ])
    seen = set()
    for item in candidates:
        item = item.resolve()
        if item in seen:
            continue
        seen.add(item)
        if (item / "normalization.pt").exists() and list(item.glob("fold_*/best.pt")):
            return item
    return None


def _vkan_checkpoint() -> Path | None:
    seg_config = STAGE_CONFIG.get("segmentation") or {}
    for path in [
        _cfg_path(seg_config.get("default_checkpoint"), VKAN_ROOT / "refinement" / "VKAN_segementation" / "runs" / "nnVnet2" / "last.pt"),
        VKAN_ROOT / "refinement" / "VKAN_segementation" / "runs" / "nnVnet2" / "last.pt",
        VKAN_ROOT / "runs" / "nnVnet3" / "best.pt",
        VKAN_ROOT / "runs" / "nnVnet" / "best.pt",
        VKAN_ROOT / "runs" / "vkan" / "best.pt",
    ]:
        if path.exists():
            return path
    return None


def _stage_status(patient: Path, stage: str, model_dir: Path | None, include_outputs: bool = True) -> dict:
    if stage == "segmentation":
        ready = (patient / "dcm").is_dir() or (patient / "orig.nii.gz").exists()
        done = _resolve_patient_file(patient, "predict_smooth.stl", recursive=False) is not None
        outputs = ["pretrain.stl", "predict.stl", "predict_smooth.stl", "segmentation/liver.stl", "segmentation/spleen.stl"]
    elif stage == "features":
        ready = (
            _resolve_patient_file(patient, "predict_smooth.stl", recursive=False) is not None
            or _resolve_patient_file(patient, "vessel.stl", recursive=False) is not None
        )
        done = _resolve_feature_file(patient, UNIFIED_FEATURES_NAME) is not None
        outputs = list(PUBLIC_FEATURE_NAMES)
    elif stage == "pvp":
        ready = _resolve_feature_file(patient, UNIFIED_FEATURES_NAME) is not None and bool(model_dir)
        done = (patient / "PVP_predict.txt").exists() or (patient / "pvp_prediction.json").exists()
        outputs = ["PVP_predict.txt", "pvp_prediction.json"]
    else:
        ready, done, outputs = False, False, []
    return {
        "status": "done" if done else "ready" if ready else "missing",
        "ready": ready,
        "done": done,
        "outputs": {name: _output_file_info(patient, name) for name in outputs} if include_outputs else {},
    }


def _stage_state(patient: Path, stage: str, model_dir: Path | None) -> dict:
    return _stage_status(patient, stage, model_dir)


def _patient_organs(patient: Path) -> dict:
    seg_dir = patient / "segmentation"
    organs = {}
    if seg_dir.exists():
        for stl in sorted(seg_dir.glob("*.stl")):
            organs[stl.stem] = _file_info(stl)
    return organs


def _patient_output_files(patient: Path, recursive: bool = False) -> dict:
    # The patient-list request is intentionally shallow. Searching through a
    # DICOM tree once per output file makes loading a large cohort expensive.
    return {name: _output_file_info(patient, name, recursive=recursive) for name in OUTPUTS}


def _feature_source(system: dict | None) -> dict:
    if not isinstance(system, dict):
        return {}
    for key in ("all_values", "available"):
        value = system.get(key)
        if isinstance(value, dict) and value:
            return value
    return system


def _feature_summary(features: dict | None) -> dict:
    if not isinstance(features, dict):
        return {}
    stat = features.get("statistical") if isinstance(features.get("statistical"), dict) else {}
    system = features.get("system") if isinstance(features.get("system"), dict) else {}
    global_data = features.get("global") if isinstance(features.get("global"), dict) else {}
    vessel_presence = features.get("vessel_presence") if isinstance(features.get("vessel_presence"), dict) else {}
    sys_values = _feature_source(system)
    segments = []
    for key in ["mpv", "sv", "smv", "lpv", "rpv", "tips", "lgv", "pgv"]:
        value = stat.get(key)
        if isinstance(value, dict):
            segments.append({
                "id": key,
                "label": key.upper(),
                "length": value.get("length"),
                "mean_diameter": value.get("mean_diameter"),
                "mean_area": value.get("mean_area"),
                "max_curvature": value.get("max_curvature"),
                "mean_circularity": value.get("mean_circularity"),
            })
    preferred_metrics = [
        ("total_centerline_length", global_data.get("total_centerline_length")),
        ("sv_smv_diameter_ratio", global_data.get("sv_smv_diameter_ratio")),
        ("sv_smv_angle", global_data.get("sv_smv_angle")),
        ("angle_sv_smv", sys_values.get("angle_sv_smv")),
        ("confluence_murray3_deviation", sys_values.get("confluence_murray3_deviation")),
        ("inflow_resistance_asymmetry", sys_values.get("inflow_resistance_asymmetry")),
        ("collateral_burden_score", sys_values.get("collateral_burden_score")),
        ("splenic_dominance_index", sys_values.get("splenic_dominance_index")),
    ]
    key_metrics = {}
    for key, value in preferred_metrics:
        if value not in (None, ""):
            key_metrics[key] = value
    if "angle_sv_smv" not in key_metrics and sys_values.get("angle_sv_smv") not in (None, ""):
        key_metrics["angle_sv_smv"] = sys_values.get("angle_sv_smv")
    if "sv_smv_angle" not in key_metrics and global_data.get("sv_smv_angle") not in (None, ""):
        key_metrics["sv_smv_angle"] = global_data.get("sv_smv_angle")
    return {
        "segments": segments,
        "key_metrics": key_metrics,
        "global": global_data,
        "vessel_presence": {
            key: {
                "present": bool(value.get("present")),
                "pointwise_status": value.get("pointwise_status"),
                "valid_diameter_points": (value.get("pointwise_diag") or {}).get("valid_diameter_points"),
            }
            for key, value in vessel_presence.items() if isinstance(value, dict)
        },
    }


def _patient_status(patient: Path, model_dir: Path | None, root: Path | None = None, detailed: bool = True) -> dict:
    stages = {stage: _stage_status(patient, stage, model_dir, include_outputs=detailed) for stage in STAGES}
    if not detailed:
        return {
            "folder": str(patient),
            "label_meta": {},
            "files": _patient_output_files(patient),
            "organs": _patient_organs(patient),
            "stages": stages,
            "features_summary": {},
            "prediction": _read_json(patient / "pvp_prediction.json"),
            "clinical": {},
            "preview": {},
        }

    organs = _patient_organs(patient)
    features_path = _resolve_feature_file(patient, UNIFIED_FEATURES_NAME)
    features = _read_json(features_path) if features_path else None
    prediction = _read_json(patient / "pvp_prediction.json")
    preview_stl = (
        _resolve_patient_file(patient, "predict_smooth.stl", recursive=False)
        or _resolve_patient_file(patient, "predict.stl", recursive=False)
        or _resolve_patient_file(patient, "pretrain.stl", recursive=False)
        or _resolve_patient_file(patient, "vessel.stl", recursive=False)
    )
    return {
        "folder": str(patient),
        "label_meta": _patient_label_metadata(patient),
        "files": _patient_output_files(patient),
        "organs": organs,
        "stages": stages,
        "features_summary": _feature_summary(features),
        "prediction": prediction,
        "clinical": _clinical_context(patient, root),
        "preview": {
            "vis_html": False,
            "vis_png": _file_info(patient / "vis_overview.png"),
            "stl": _file_info(preview_stl) if preview_stl else _file_info(patient / "predict_smooth.stl"),
        },
    }


def _create_session(payload: dict) -> dict:
    root = Path(str(payload.get("root_folder") or "").strip()).resolve()
    if not root.exists():
        raise ValueError(f"Folder does not exist: {root}")
    model = _model_dir(str(payload.get("model_dir") or "").strip() or None)
    patients = _discover_patients(root)
    if not patients:
        raise ValueError(f"No patient folders found under {root}")
    session = {
        "id": _new_id(),
        "root": str(root),
        "patients": patients,
        "model_dir": str(model or ""),
        "model_valid": bool(model),
        "vkan_checkpoint": str(_vkan_checkpoint() or ""),
        "created": _now(),
        "runtime": _runtime(),
    }
    with LOCK:
        SESSIONS[session["id"]] = session
    return session


def _session_from_payload(payload: dict) -> dict:
    session_id = str(payload.get("session_id") or "")
    if session_id:
        with LOCK:
            session = SESSIONS.get(session_id)
        if session:
            return session
    if payload.get("root_folder"):
        return _create_session(payload)
    raise ValueError("Session not found")


def _resolve_patient(session: dict, patient_id: str | None) -> dict | None:
    patients = session.get("patients") or []
    if not patients:
        return None
    if not patient_id or patient_id == "first":
        return patients[0]
    for patient in patients:
        if patient.get("id") == patient_id:
            return patient
    return patients[0]


def _job_view(job: dict) -> dict:
    return {k: v for k, v in job.items() if not k.startswith("_")}


def _append(job: dict, text: str):
    if not text:
        return
    with LOCK:
        job.setdefault("logs", []).append(str(text)[-16000:])
        job["updated"] = _now()


def _run_command(cmd: list[str], cwd: Path, job: dict):
    _append(job, "> " + " ".join(str(x) for x in cmd))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if proc.stdout.strip():
        _append(job, proc.stdout.strip())
    if proc.stderr.strip():
        _append(job, proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}")


def _run_segmentation(patient_dir: Path, session: dict, payload: dict, job: dict):
    seg_params = payload.get("segmentation") if isinstance(payload.get("segmentation"), dict) else {}
    mode = str(seg_params.get("mode") or "auto").lower()
    if mode not in {"auto", "pretrain", "predict", "smooth"}:
        mode = "auto"
    try:
        smooth_iterations = int(seg_params.get("smooth_iterations") or 8)
    except (TypeError, ValueError):
        smooth_iterations = 8
    smooth_iterations = max(0, min(30, smooth_iterations))
    checkpoint = Path(str(payload.get("checkpoint") or session.get("vkan_checkpoint") or ""))
    if mode in {"auto", "predict"} and not checkpoint.exists():
        raise FileNotFoundError(f"VKAN checkpoint not found: {checkpoint}")
    py = sys.executable
    if mode in {"auto", "pretrain"}:
        _run_command([
            py, str(VKAN_ROOT / "pretrain" / "totalseg.py"), "--data_root", str(patient_dir),
            "--patient", patient_dir.name, "--device", str(payload.get("device") or "gpu"),
            "--structures", "bone_all", "spleen", "liver", "kidney_left", "kidney_right",
            "inferior_vena_cava", "aorta", "portal_vein", "--resume", "--fast",
        ], VKAN_ROOT, job)
        script = (
            "from pathlib import Path; from types import SimpleNamespace; "
            "from pretrain.preprocess import pretrain_patient; "
            f"p=Path(r'''{patient_dir}'''); "
            "case=SimpleNamespace(name=p.name,path=p,dcm_dir=p/'dcm',label_stl=p/'vessel.stl',"
            "pretrain_stl=p/'pretrain.stl',predict_stl=p/'predict.stl',is_post_tips='#' in p.name); "
            f"print(pretrain_patient(case, force={bool(payload.get('force'))}))"
        )
        _run_command([py, "-c", script], VKAN_ROOT, job)
    if mode in {"auto", "predict"}:
        _run_command([
            py, str(VKAN_ROOT / "refinement" / "predict.py"), "--data_root", str(patient_dir),
            "--patient", patient_dir.name, "--checkpoint", str(checkpoint), "--threshold", "0.5",
        ], VKAN_ROOT, job)
    if mode in {"auto", "predict", "smooth"}:
        smooth = (
            "from pathlib import Path; from types import SimpleNamespace; "
            "from postprocess.check_and_smooth import check_and_smooth_case; "
            f"p=Path(r'''{patient_dir}'''); "
            "case=SimpleNamespace(name=p.name,path=p,dcm_dir=p/'dcm',label_stl=p/'vessel.stl',"
            "pretrain_stl=p/'pretrain.stl',predict_stl=p/'predict.stl',is_post_tips='#' in p.name); "
            f"print(check_and_smooth_case(case, iterations={smooth_iterations}, force=True))"
        )
        _run_command([py, "-c", smooth], VKAN_ROOT, job)


def _geometry_file_signature(path: Path | None) -> tuple:
    if not path:
        return ("", 0, 0)
    try:
        stat = path.stat()
        return (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (str(path), 0, 0)


def _point_center(points) -> tuple[float, float, float] | None:
    coords = []
    for point in points if points is not None else []:
        try:
            coords.append((float(point[0]), float(point[1]), float(point[2])))
        except (TypeError, ValueError, IndexError):
            continue
    if not coords:
        return None
    count = float(len(coords))
    return tuple(sum(point[axis] for point in coords) / count for axis in range(3))


def _centerline_center(path: Path | None) -> tuple[float, float, float] | None:
    if not path:
        return None
    nodes = geometry_web._read_centerline_file(path)
    if not nodes:
        return None
    return _point_center((node["x"], node["y"], node["z"]) for node in nodes.values())


def _mesh_center(path: Path) -> tuple[float, float, float] | None:
    mesh = geometry_web._load_mesh(path, max_faces=5000)
    return _point_center((mesh or {}).get("vertices"))


def _feature_stl(patient_dir: Path) -> Path:
    candidates = []
    for name in ["predict_smooth.stl", "predict.stl", "vessel.stl", "pretrain.stl"]:
        path = _resolve_patient_file(patient_dir, name)
        if path and path.is_file() and path not in candidates:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("Need predict_smooth.stl, predict.stl, vessel.stl, or pretrain.stl for geometry extraction.")

    centerline = (
        resolve_feature_path(patient_dir, SMOOTH_CENTERLINE_NAME)
        or resolve_feature_path(patient_dir, RAW_CENTERLINE_NAME)
    )
    signature = (
        _geometry_file_signature(centerline),
        tuple(_geometry_file_signature(path) for path in candidates),
    )
    cache_key = str(patient_dir.resolve())
    with GEOMETRY_STL_CACHE_LOCK:
        cached = GEOMETRY_STL_CACHE.get(cache_key)
        if cached and cached[0] == signature and cached[1] in candidates:
            return cached[1]

    selected = candidates[0]
    center = _centerline_center(centerline)
    if center is not None and len(candidates) > 1:
        scored = []
        for order, path in enumerate(candidates):
            mesh_center = _mesh_center(path)
            if mesh_center is None:
                continue
            distance_sq = sum((center[axis] - mesh_center[axis]) ** 2 for axis in range(3))
            scored.append((distance_sq, order, path))
        if scored:
            selected = min(scored)[2]

    with GEOMETRY_STL_CACHE_LOCK:
        GEOMETRY_STL_CACHE[cache_key] = (signature, selected)
    return selected


def _run_features(patient_dir: Path, job: dict):
    stl = _feature_stl(patient_dir)
    script = f"""
import shutil
from pathlib import Path
from extract_centerline import extract_centerline
from smooth_centerline import smooth_centerline
from segment_vessels import segment_vessels
from extract_profiles import extract_profiles
from extract_features import extract_all_features

stl = Path(r'''{stl}''')
extract_centerline(str(stl), pitch=0.5, min_branch_length_mm=10.0, min_relative_length=0.05,
                   min_radius_ratio=0.4, keep_radius_ratio=0.55,
                   absolute_min_branch_length_mm=3.0, absolute_min_radius_mm=0.5,
                   merge_bp_distance_mm=5.0)
smooth_centerline(str(stl))
segment_vessels(str(stl), post_tips='#' in stl.parent.name)
extract_profiles(str(stl), n_points=200, pitch=0.5, curvature_window=7, section_step=3,
                 ownership_factor=1.8, junction_policy='min_valid', max_diameter_rate_per_mm=0.5)
extract_all_features(str(stl), n_fit_points=10, curvature_window=7, sample_step=3, pitch=0.5)
from features_layout import PUBLIC_FEATURE_NAMES, remove_generated_outputs
remove_generated_outputs(stl.parent, keep_public=True)
patient_root = Path(r'''{patient_dir}''')
source_features = stl.parent / 'features'
target_features = patient_root / 'features'
target_features.mkdir(parents=True, exist_ok=True)
if source_features.resolve() != target_features.resolve():
    for name in PUBLIC_FEATURE_NAMES:
        source = source_features / name
        target = target_features / name
        if source.exists():
            if target.exists():
                target.unlink()
            shutil.move(str(source), str(target))
    if source_features.exists() and not any(source_features.iterdir()):
        source_features.rmdir()
remove_generated_outputs(patient_root, keep_public=True)
"""
    _run_command([sys.executable, "-c", script], CENTERLINE_ROOT, job)


def _run_pvp(root: Path, patient_id: str, session: dict, payload: dict, job: dict):
    model = Path(str(payload.get("model_dir") or session.get("model_dir") or ""))
    if not ((model / "normalization.pt").exists() and list(model.glob("fold_*/best.pt"))):
        raise FileNotFoundError(f"PVP model directory is not usable: {model}")
    device = str(payload.get("device") or "auto")
    if device == "gpu":
        device = "cuda"
    _run_command([
        str(_pvp_python()), str(_cfg_path((STAGE_CONFIG.get("pvp") or {}).get("infer"), PVP_ROOT / "infer.py")),
        "--data_root", str(root), "--model_dir", str(model), "--patient", patient_id,
        "--device", device,
    ], PVP_ROOT, job)


def _new_job(session: dict, stage: str, patients: list[dict], payload: dict) -> dict:
    stages = STAGES if stage == "all" else [stage]
    job = {
        "id": _new_id(),
        "session_id": session["id"],
        "stage": stage,
        "stages": stages,
        "status": "running",
        "created": _now(),
        "updated": _now(),
        "current": "",
        "completed": 0,
        "total": len(stages) * len(patients),
        "logs": [],
        "errors": [],
        "results": {},
        "_session": session,
        "_patients": patients,
        "_payload": payload,
    }
    with LOCK:
        JOBS[job["id"]] = job
    return _job_view(job)


def _run_job(job_id: str):
    with LOCK:
        job = JOBS[job_id]
        session = dict(job["_session"])
        patients = list(job["_patients"])
        payload = dict(job["_payload"])
        stages = list(job["stages"])
    root = Path(session["root"])
    try:
        loop_stages = [stage for stage in stages if not (stage == "pvp" and len(patients) > 1)]
        for patient in patients:
            patient_dir = Path(patient["folder"])
            for stage in loop_stages:
                with LOCK:
                    job["current"] = f"{patient['id']} / {STAGE_LABELS[stage]}"
                    job["updated"] = _now()
                ok = True
                started = time.time()
                try:
                    if stage == "segmentation":
                        if _resolve_patient_file(patient_dir, "predict_smooth.stl") and not payload.get("force"):
                            _append(job, f"[skip] {patient['id']} segmentation outputs already exist")
                        else:
                            _run_segmentation(patient_dir, session, payload, job)
                    elif stage == "features":
                        if _resolve_feature_file(patient_dir, UNIFIED_FEATURES_NAME) is not None and not payload.get("force"):
                            _append(job, f"[skip] {patient['id']} feature outputs already exist")
                        else:
                            _run_features(patient_dir, job)
                    elif stage == "pvp":
                        _run_pvp(root, patient["id"], session, payload, job)
                except Exception as exc:
                    ok = False
                    with LOCK:
                        job["errors"].append(f"{patient['id']} / {stage}: {type(exc).__name__}: {exc}")
                    _append(job, traceback.format_exc())
                _append(job, f"[{'OK' if ok else 'FAIL'}] {patient['id']} / {stage} ({time.time() - started:.1f}s)")
                with LOCK:
                    job["results"].setdefault(patient["id"], {})[stage] = ok
                    job["completed"] += 1
                    job["updated"] = _now()
        if "pvp" in stages and len(patients) > 1:
            with LOCK:
                job["current"] = f"all patients / {STAGE_LABELS['pvp']}"
            ok = True
            started = time.time()
            try:
                _run_pvp(root, "all", session, payload, job)
            except Exception as exc:
                ok = False
                with LOCK:
                    job["errors"].append(f"all patients / pvp: {type(exc).__name__}: {exc}")
                _append(job, traceback.format_exc())
            _append(job, f"[{'OK' if ok else 'FAIL'}] all patients / pvp ({time.time() - started:.1f}s)")
            with LOCK:
                for patient in patients:
                    job["results"].setdefault(patient["id"], {})["pvp"] = ok
                job["completed"] += len(patients)
                job["updated"] = _now()
        with LOCK:
            job["status"] = "failed" if job["errors"] else "done"
            job["current"] = ""
            job["updated"] = _now()
    except Exception as exc:
        with LOCK:
            job["status"] = "failed"
            job["errors"].append(f"{type(exc).__name__}: {exc}")
            job["logs"].append(traceback.format_exc())
            job["updated"] = _now()


def _zip_outputs(patients: list[dict]) -> bytes:
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for patient in patients:
            root = Path(patient["folder"])
            prefix = patient["id"]
            for rel in OUTPUTS:
                path = _resolve_patient_file(root, rel)
                if path and path.exists() and path.is_file():
                    zf.write(path, f"{prefix}/{rel}")
            seg_dir = root / "segmentation"
            if seg_dir.exists():
                for path in sorted(seg_dir.glob("*.stl")):
                    zf.write(path, f"{prefix}/segmentation/{path.name}")
    return bio.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "PortaFlowIntegrated/1.0"

    def do_HEAD(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                self._json({"ok": True, "time": _now(), "runtime": _runtime()}, head_only=True)
            elif path == "/favicon.ico":
                self._bytes(b"", "image/x-icon", status=204, head_only=True)
            elif path.startswith("/api/"):
                self._json({"error": "Not found"}, status=404, head_only=True)
            else:
                self._static(path, head_only=True)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500, head_only=True)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                self._json({"ok": True, "time": _now(), "runtime": _runtime()})
            elif path == "/api/geometry/health":
                self._json({"ok": True, "time": _now(), "runtime": geometry_web._runtime_info()})
            elif path.startswith("/api/geometry/session/") and path.endswith("/data"):
                self._geometry_session_data(path, parsed.query)
            elif path.startswith("/api/geometry/session/") and path.endswith("/download"):
                self._geometry_download(path, parsed.query)
            elif path.startswith("/api/geometry/job/"):
                self._geometry_job(path)
            elif path == "/assets/plotly.min.js":
                self._serve_plotly()
            elif path == "/api/geometry/workbench":
                self._geometry_workbench(parsed.query)
            elif path.startswith("/api/session/") and path.endswith("/data"):
                self._session_data(path, parsed.query)
            elif path.startswith("/api/session/") and path.endswith("/download"):
                self._download(path, parsed.query)
            elif path.startswith("/api/session/") and path.endswith("/patient-file"):
                self._patient_file(path, parsed.query)
            elif path.startswith("/api/job/"):
                self._job(path)
            elif path == "/favicon.ico":
                self._bytes(b"", "image/x-icon", status=204)
            elif path == "/geometry" or path.startswith("/geometry/"):
                self._geometry_static(path)
            else:
                self._static(path)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/session":
                self._json({"session": _create_session(self._body_json())})
            elif parsed.path == "/api/geometry/session/from-parent":
                self._geometry_session_from_parent()
            elif parsed.path == "/api/geometry/session":
                self._geometry_create_session()
            elif parsed.path == "/api/geometry/run":
                self._geometry_run()
            elif parsed.path == "/api/geometry/centerline/delete-branches":
                self._geometry_edit("delete")
            elif parsed.path == "/api/geometry/centerline/manual-segments":
                self._geometry_edit("manual")
            elif parsed.path == "/api/geometry/analysis/suggest-ranges":
                self._geometry_edit("suggest")
            elif parsed.path == "/api/geometry/analysis/save-ranges":
                self._geometry_edit("save-ranges")
            elif parsed.path == "/api/run-stage":
                self._run_stage()
            else:
                self._json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def log_message(self, fmt, *args):
        try:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))
        except Exception:
            pass

    def _body_json(self):
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8"))

    def _json(self, data, status=200, head_only=False):
        payload = json.dumps(data, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if not head_only:
            self.wfile.write(payload)

    def _bytes(self, data: bytes, content_type: str, status=200, headers: dict | None = None, head_only=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def _static(self, path: str, head_only=False):
        if path in ("", "/"):
            file_path = STATIC_ROOT / "index.html"
        else:
            file_path = (STATIC_ROOT / Path(path.lstrip("/"))).resolve()
        if not str(file_path).startswith(str(STATIC_ROOT.resolve())):
            self._json({"error": "Forbidden"}, status=403, head_only=head_only)
            return
        if not file_path.exists() or not file_path.is_file():
            self._json({"error": "Not found"}, status=404, head_only=head_only)
            return
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self._bytes(file_path.read_bytes(), ctype, headers={"Cache-Control": "no-store"}, head_only=head_only)

    def _serve_plotly(self, head_only=False):
        try:
            import plotly
            file_path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
            if file_path.exists():
                self._bytes(file_path.read_bytes(), "application/javascript; charset=utf-8",
                            headers={"Cache-Control": "no-store"}, head_only=head_only)
                return
        except Exception:
            pass
        self._json({"error": "Local Plotly asset not found"}, status=404, head_only=head_only)

    def _geometry_static(self, path: str, head_only=False):
        rel = "index.html" if path in ("/geometry", "/geometry/") else path[len("/geometry/"):]
        file_path = (GEOMETRY_STATIC_ROOT / rel).resolve()
        if not _inside(GEOMETRY_STATIC_ROOT, file_path) or not file_path.is_file():
            self._json({"error": "Not found"}, status=404, head_only=head_only)
            return
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self._bytes(file_path.read_bytes(), ctype, headers={"Cache-Control": "no-store"}, head_only=head_only)

    def _geometry_create_session(self):
        payload = self._body_json()
        session = geometry_web._create_session_batch(payload)
        self._json({"session": session})

    def _geometry_session_from_parent(self):
        payload = self._body_json()
        parent = self._session(str(payload.get("session_id") or ""))
        patient = _resolve_patient(parent, payload.get("patient_id"))
        if not patient:
            raise ValueError("Patient not found")
        patient_dir = Path(patient["folder"])
        stl = _feature_stl(patient_dir)
        patient_key = uuid.uuid5(uuid.NAMESPACE_URL, str(patient_dir.resolve())).hex[:12]
        session_id = f"integrated-{parent['id']}-{patient_key}"
        session = {
            "id": session_id,
            "mode": "single",
            "created": _now(),
            "root": str(patient_dir),
            "stl_name": stl.name,
            "patients": [geometry_web._patient_record(stl)],
            "params": dict(geometry_web.DEFAULT_PARAMS),
            "runtime": geometry_web._runtime_info(),
            "parent_session_id": parent["id"],
        }
        with geometry_web.STATE_LOCK:
            geometry_web.SESSIONS[session_id] = session
        self._json({"session": session})

    def _geometry_session(self, session_id: str) -> dict:
        with geometry_web.STATE_LOCK:
            session = geometry_web.SESSIONS.get(session_id)
        if not session:
            raise ValueError("Geometry session not found")
        return session

    def _geometry_session_data(self, path: str, query: str):
        session_id = unquote(path.split("/")[4])
        session = self._geometry_session(session_id)
        qs = parse_qs(query)
        patient = geometry_web._resolve_patient(session, (qs.get("patient") or [None])[0])
        if not patient:
            raise ValueError("Patient not found")
        data = geometry_web.build_visualization_data(
            Path(patient["stl_path"]),
            section_stride=geometry_web._safe_int((qs.get("section_stride") or [10])[0], 10),
            max_faces=geometry_web._safe_int((qs.get("max_faces") or [80000])[0], 80000),
            include_surface_sections=(qs.get("surface_sections") or ["0"])[0] == "1",
        )
        data["session"] = session
        self._json(data)

    def _geometry_run(self):
        payload = self._body_json()
        session = self._geometry_session(str(payload.get("session_id") or ""))
        steps = [step for step in (payload.get("steps") or []) if step in geometry_web.PIPELINE_STEPS]
        if not steps:
            raise ValueError("No valid geometry steps selected")
        raw_modes = payload.get("step_modes") or {}
        step_modes = {step: "reuse" if raw_modes.get(step) == "reuse" else "recompute" for step in steps}
        patients = session.get("patients") or []
        patient_id = payload.get("patient_id")
        if patient_id and patient_id != "all":
            patient = geometry_web._resolve_patient(session, patient_id)
            patients = [patient] if patient else []
        params = geometry_web._merge_params(payload.get("params"))
        job = geometry_web._new_job(session["id"], steps, patients, step_modes=step_modes)
        with geometry_web.STATE_LOCK:
            job["_patients_runtime"] = patients
            session["params"] = params
        threading.Thread(
            target=geometry_web._run_job,
            args=(job["id"], params, payload.get("post_tips_mode") or "auto", bool(payload.get("export_png"))),
            daemon=True,
        ).start()
        self._json({"job": job})

    def _geometry_edit(self, operation: str):
        payload = self._body_json()
        session = self._geometry_session(str(payload.get("session_id") or ""))
        patient = geometry_web._resolve_patient(session, payload.get("patient_id"))
        if not patient:
            raise ValueError("Patient not found")
        stl = Path(patient["stl_path"])
        if operation == "delete":
            result = geometry_web.delete_centerline_terminal_branches(stl, payload.get("branch_ids") or [])
        elif operation == "manual":
            result = geometry_web.save_manual_segment_assignments(stl, payload.get("assignments") or [])
        elif operation == "suggest":
            result = geometry_web.suggest_analysis_ranges(stl)
        else:
            result = geometry_web.save_analysis_ranges(stl, payload.get("ranges") or [])
        self._json({"ok": True, "result": result})

    def _geometry_job(self, path: str):
        job_id = unquote(path.rstrip("/").split("/")[-1])
        with geometry_web.STATE_LOCK:
            job = geometry_web.JOBS.get(job_id)
        if not job:
            self._json({"error": "Job not found"}, status=404)
            return
        self._json({"job": job})

    def _geometry_download(self, path: str, query: str):
        session_id = unquote(path.split("/")[4])
        session = self._geometry_session(session_id)
        patient_id = (parse_qs(query).get("patient") or ["all"])[0]
        if patient_id == "all":
            patients = session.get("patients") or []
        else:
            patient = geometry_web._resolve_patient(session, patient_id)
            patients = [patient] if patient else []
        payload = geometry_web._zip_patient_outputs(patients)
        self._bytes(payload, "application/zip", headers={"Content-Disposition": f'attachment; filename="geometry_{session_id}.zip"'})

    def _session(self, session_id: str) -> dict:
        with LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Session not found")
        return session

    def _geometry_workbench(self, query: str):
        qs = parse_qs(query)
        session_id = (qs.get("session_id") or [""])[0]
        patient = (qs.get("patient") or [""])[0]
        iframe_url = "/geometry/?embed=1&autoload=1&ui=7"
        if session_id:
            iframe_url += "&session_id=" + quote(session_id)
        if patient:
            iframe_url += "&patient=" + quote(patient)
        self._json({"workbench": {
            "stage": "features",
            "running": True,
            "url": "/geometry/",
            "iframe_url": iframe_url,
            "pid": os.getpid(),
        }})

    def _session_data(self, path: str, query: str):
        session = self._session(path.split("/")[3])
        qs = parse_qs(query)
        patient_id = (qs.get("patient") or [None])[0]
        detailed = (qs.get("detail") or [""])[0] == "1" or bool(patient_id and patient_id != "all")
        model = Path(session["model_dir"]) if session.get("model_dir") else None
        root = Path(session["root"]) if session.get("root") else None
        patients = session.get("patients") or []
        if patient_id and patient_id != "all":
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
        data = []
        for patient in patients:
            item = dict(patient)
            status = _patient_status(Path(patient["folder"]), model, root, detailed=detailed)
            item["status"] = status
            # The frontend renders patient cards and clinical fields from the
            # top-level record. Promote detailed metadata from the status
            # payload so the selected patient shows the same source of truth.
            if detailed:
                item["label_meta"] = status.get("label_meta") or {}
            data.append(item)
        self._json({"session": session, "patients": data, "stage_labels": STAGE_LABELS})

    def _run_stage(self):
        payload = self._body_json()
        session = _session_from_payload(payload)
        stage = str(payload.get("stage") or "")
        if stage not in set(STAGES + ["all"]):
            raise ValueError("stage must be segmentation, features, pvp, or all")
        patient_id = str(payload.get("patient_id") or "all")
        if patient_id == "all":
            patients = session.get("patients") or []
        else:
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
        if not patients:
            raise ValueError("No patients selected")
        job = _new_job(session, stage, patients, payload)
        threading.Thread(target=_run_job, args=(job["id"],), daemon=True).start()
        self._json({"job": job, "session": session})

    def _job(self, path: str):
        job_id = path.rstrip("/").split("/")[-1]
        with LOCK:
            job = JOBS.get(job_id)
        if not job:
            self._json({"error": "Job not found"}, status=404)
            return
        self._json({"job": _job_view(job)})

    def _download(self, path: str, query: str):
        session = self._session(path.split("/")[3])
        qs = parse_qs(query)
        patient_id = (qs.get("patient") or ["all"])[0]
        if patient_id == "all":
            patients = session.get("patients") or []
            name = f"portaflow_outputs_{session['id']}.zip"
        else:
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
            name = f"portaflow_outputs_{patient_id}.zip"
        self._bytes(_zip_outputs(patients), "application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    def _patient_file(self, path: str, query: str):
        session = self._session(path.split("/")[3])
        qs = parse_qs(query)
        patient = _resolve_patient(session, (qs.get("patient") or [None])[0])
        rel = (qs.get("file") or [""])[0].replace("\\", "/").strip("/")
        if not patient or not rel:
            raise ValueError("patient and file are required")
        root = Path(patient["folder"]).resolve()
        file_path = _resolve_patient_file(root, rel)
        if not file_path:
            file_path = (root / rel).resolve()
        if not str(file_path).startswith(str(root)) or not file_path.exists() or not file_path.is_file():
            self._json({"error": "Not found"}, status=404)
            return
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self._bytes(file_path.read_bytes(), ctype)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PortaFlow workbench running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
