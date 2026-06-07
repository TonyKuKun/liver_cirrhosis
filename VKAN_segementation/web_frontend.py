"""
Local browser workbench for the portal-vein segmentation pipeline.

Run:
    python web_frontend.py --host 127.0.0.1 --port 8775

The server follows the same standard-library HTTP pattern as the STL
workbench in liver_pre_process/zxx_stl. Scientific steps still call the
existing project scripts, and each step reuses patient outputs when the
expected files are already present unless "force" is requested from the UI.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import mimetypes
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "web"
DEFAULT_CONFIG_PATH = APP_ROOT / "web_frontend_config.json"

DEFAULT_CHECKPOINTS = [
    APP_ROOT / "refinement" / "VKAN_segementation" / "runs" / "nnVnet" / "best.pt",
    APP_ROOT / "VKAN_segementation" / "runs" / "vkan" / "best.pt",
    APP_ROOT / "refinement" / "VKAN_segementation" / "runs" / "vkan" / "best.pt",
    APP_ROOT / "refinement" / "VKAN_segementation" / "runs" / "vkan2" / "best.pt",

]

ORGAN_LABELS = {
    "bone_all": "Bone",
    "spleen": "Spleen",
    "liver": "Liver",
    "liver_left": "Liver L",
    "liver_right": "Liver R",
    "kidney_left": "Kidney L",
    "kidney_right": "Kidney R",
    "inferior_vena_cava": "IVC",
    "aorta": "Aorta",
    "portal_vein": "Portal vein",
}

ORGAN_COLORS = {
    "bone_all": "#d8dde5",
    "spleen": "#8b5cf6",
    "liver": "#8fb339",
    "liver_left": "#6aa84f",
    "liver_right": "#9bbb59",
    "kidney_left": "#4f8cc9",
    "kidney_right": "#2f75b5",
    "inferior_vena_cava": "#35a7ff",
    "aorta": "#ef4444",
    "portal_vein": "#f59e0b",
}

MODEL_MESHES = {
    "pretrain": {"label": "Pretrain", "file": "pretrain.stl", "color": "#7c8da0"},
    "predict": {"label": "Predict", "file": "predict.stl", "color": "#f97316"},
    "smooth": {"label": "Smooth", "file": "predict_smooth.stl", "color": "#10b981"},
    "manual": {"label": "Manual edit", "file": "manual_edit.stl", "color": "#0f766e"},
    "vessel": {"label": "Manual label", "file": "vessel.stl", "color": "#2563eb"},
}

NIFTI_BACKED_MESHES = {
    "pretrain": "pretrain.nii.gz",
    "predict": "predict_mask.nii.gz",
}

MASK_COLORS = [
    "#f97316",
    "#10b981",
    "#2563eb",
    "#e11d48",
    "#8b5cf6",
    "#14b8a6",
    "#f59e0b",
    "#64748b",
]

OUTPUT_FILES = [
    "orig.nii.gz",
    "pretrain.nii.gz",
    "predict_mask.nii.gz",
    "pretrain.stl",
    "predict.stl",
    "predict_smooth.stl",
    "manual_edit.stl",
    "vessel.stl",
    "vkan_work/pretrain_meta.json",
    "vkan_work/predict_check.json",
    "vkan_work/predict_quality_report.png",
]

PIPELINE_STEPS = ["totalseg", "pretrain", "refine"]
STEP_LABELS = {
    "totalseg": "TotalSegmentator organs",
    "pretrain": "Coarse pretrain portal vein",
    "refine": "nnVnet refinement + smoothing",
}

SESSIONS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
STATE_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return str(value)


def _sanitize_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _read_json_file(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_config(path: str | Path | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = APP_ROOT / config_path
    if not config_path.exists():
        return {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _runtime_info() -> dict:
    return {
        "python": sys.executable,
        "python_prefix": sys.prefix,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV") or "",
        "conda_prefix": os.environ.get("CONDA_PREFIX") or "",
    }


def _find_conda_exe(explicit: str | None = None) -> str:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_exe = os.environ.get("CONDA_EXE")
    if env_exe:
        candidates.append(Path(env_exe))
    candidates.append(Path(sys.prefix) / "Scripts" / "conda.exe")
    candidates.append(Path(sys.prefix) / "condabin" / "conda.bat")
    which_conda = shutil.which("conda")
    if which_conda:
        candidates.append(Path(which_conda))
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Could not locate conda. Pass --conda-exe or start from an Anaconda prompt.")


def _maybe_reexec_in_conda(args, argv: list[str]) -> None:
    requested = (args.conda_env or "").strip()
    if not requested or args.no_conda_reexec:
        return
    current_env = os.environ.get("CONDA_DEFAULT_ENV") or ""
    prefix_name = Path(sys.prefix).name
    executable_parts = {part.lower() for part in Path(sys.executable).parts}
    if (
        current_env.lower() == requested.lower()
        or prefix_name.lower() == requested.lower()
        or requested.lower() in executable_parts
    ):
        return
    conda_exe = _find_conda_exe(args.conda_exe)
    script = str(Path(__file__).resolve())
    conda_command = [
        conda_exe,
        "run",
        "-n",
        requested,
        "python",
        script,
        "--no-conda-reexec",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if conda_exe.lower().endswith((".bat", ".cmd")):
        command = ["cmd.exe", "/d", "/c", "call", *conda_command]
    else:
        command = conda_command
    if args.config:
        command.extend(["--config", str(args.config)])
    if args.conda_exe:
        command.extend(["--conda-exe", str(args.conda_exe)])
    passthrough = []
    skip_next = False
    consumed = {"--conda-env", "--conda-exe", "--config", "--host", "--port"}
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item in consumed:
            skip_next = True
            continue
        if any(item.startswith(flag + "=") for flag in consumed):
            continue
        if item == "--no-conda-reexec":
            continue
        passthrough.append(item)
    command.extend(passthrough)
    print(f"Restarting web frontend in conda env: {requested}")
    raise SystemExit(subprocess.call(command, cwd=str(APP_ROOT)))


def _new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def _is_invalid_patient(path: Path) -> bool:
    return any(marker in path.name for marker in ("@", "!", "&", "$"))


def _looks_like_patient(path: Path) -> bool:
    return (path / "orig.nii.gz").exists() or (path / "dcm").is_dir() or (path / "pretrain.stl").exists()


def _patient_record(path: Path) -> dict:
    path = Path(path).resolve()
    return {
        "id": path.name,
        "folder": str(path),
        "is_post_tips": "#" in path.name,
        "has_orig": (path / "orig.nii.gz").exists(),
        "has_dcm": (path / "dcm").is_dir(),
    }


def _discover_patients(root_folder: Path) -> list[dict]:
    root_folder = Path(root_folder)
    if not root_folder.exists():
        return []
    if _looks_like_patient(root_folder) and not _is_invalid_patient(root_folder):
        return [_patient_record(root_folder)]
    patients = []
    for child in sorted(root_folder.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and _looks_like_patient(child) and not _is_invalid_patient(child):
            patients.append(_patient_record(child))
    return patients


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


def _available_checkpoint() -> Path | None:
    for path in DEFAULT_CHECKPOINTS:
        if path.exists():
            return path
    return None


def _create_session(payload: dict) -> dict:
    root = Path(str(payload.get("root_folder") or "").strip())
    if not root.exists():
        raise ValueError(f"Folder does not exist: {root}")
    patients = _discover_patients(root)
    if not patients:
        raise ValueError(f"No patient folders found under {root}")
    session_id = _new_session_id()
    session = {
        "id": session_id,
        "created": _now(),
        "root": str(root),
        "patients": patients,
        "runtime": _runtime_info(),
        "default_checkpoint": str(_available_checkpoint() or ""),
    }
    with STATE_LOCK:
        SESSIONS[session_id] = session
    return session


def _file_info(path: Path) -> dict:
    return {
        "exists": path.exists(),
        "path": str(path),
        "size": path.stat().st_size if path.exists() else 0,
        "modified": path.stat().st_mtime if path.exists() else None,
    }


def _patient_status(patient_dir: Path) -> dict:
    files = {name.replace("/", "_").replace(".nii.gz", "").replace(".stl", ""): _file_info(patient_dir / name) for name in OUTPUT_FILES}
    organs = []
    seg_dir = patient_dir / "segmentation"
    for stl in sorted(seg_dir.glob("*.stl")) if seg_dir.exists() else []:
        name = stl.stem
        organs.append({
            "name": name,
            "label": ORGAN_LABELS.get(name, name),
            "color": ORGAN_COLORS.get(name, "#94a3b8"),
            "stl": str(stl),
            "nii": str(seg_dir / f"{name}.nii.gz"),
            "size": stl.stat().st_size,
        })
    return {
        "folder": str(patient_dir),
        "files": files,
        "organs": organs,
        "masks": _mask_catalog(patient_dir),
        "pretrain_meta": _read_json_file(patient_dir / "vkan_work" / "pretrain_meta.json"),
        "predict_check": _read_json_file(patient_dir / "vkan_work" / "predict_check.json") or _read_json_file(patient_dir / "predict_check.json"),
    }


def _mask_catalog(patient_dir: Path) -> list[dict]:
    patient_dir = patient_dir.resolve()
    masks = []
    seen: set[str] = set()
    candidates = [
        patient_dir / "pretrain.nii.gz",
        patient_dir / "predict_mask.nii.gz",
        patient_dir / "vkan_work" / "manual_mask.nii.gz",
    ]
    masks_dir = patient_dir / "masks"
    if masks_dir.exists():
        candidates.extend(sorted(masks_dir.glob("*.nii.gz")))
    seg_dir = patient_dir / "segmentation"
    if seg_dir.exists():
        candidates.extend(sorted(seg_dir.glob("*.nii.gz")))
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        rel = _patient_relative(patient_dir, path)
        if rel == "orig.nii.gz" or rel in seen:
            continue
        seen.add(rel)
        mask_name = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
        label = mask_name
        if path.parent.name == "masks":
            label = f"手工 {label}"
        if path.parent.name == "segmentation":
            label = f"器官 {ORGAN_LABELS.get(mask_name, mask_name)}"
        masks.append({
            "id": rel,
            "label": label,
            "path": str(path),
            "color": _mask_color(rel, len(masks)),
            "size": path.stat().st_size,
        })
    return masks


def _mask_color(mask_id: str, idx: int = 0) -> str:
    rel = str(mask_id).replace("\\", "/").strip("/")
    if rel == "pretrain.nii.gz":
        return MODEL_MESHES["pretrain"]["color"]
    if rel == "predict_mask.nii.gz":
        return MODEL_MESHES["predict"]["color"]
    if rel in {"vkan_work/manual_mask.nii.gz", "masks/manual_mask.nii.gz"}:
        return MODEL_MESHES["manual"]["color"]
    if rel == "vessel.nii.gz":
        return MODEL_MESHES["vessel"]["color"]
    if rel.startswith("segmentation/"):
        return ORGAN_COLORS.get(Path(rel).name[:-7], MASK_COLORS[idx % len(MASK_COLORS)])
    return MASK_COLORS[idx % len(MASK_COLORS)]


def _safe_mask_name(value: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value).strip())
    name = name.strip("_") or "manual_mask"
    if not name.endswith(".nii.gz"):
        name += ".nii.gz"
    return name


def _mask_path(patient_dir: Path, mask_id: str, *, create_parent: bool = False) -> Path:
    rel = str(mask_id or "").replace("\\", "/").strip("/")
    if not rel:
        raise ValueError("mask id is required")
    if "/" not in rel and not rel.endswith(".nii.gz"):
        rel = f"masks/{_safe_mask_name(rel)}"
    path = (patient_dir / rel).resolve()
    if not str(path).startswith(str(patient_dir.resolve())):
        raise ValueError("mask path escapes patient folder")
    if path.name == "orig.nii.gz" or not path.name.endswith(".nii.gz"):
        raise ValueError("mask must be a .nii.gz file and cannot be orig.nii.gz")
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _patient_relative(patient_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(patient_dir.resolve()).as_posix()


def _mesh_path(patient_dir: Path, mesh_name: str) -> Path:
    if mesh_name.startswith("organ:"):
        organ = mesh_name.split(":", 1)[1]
        path = patient_dir / "segmentation" / f"{organ}.stl"
    elif mesh_name in MODEL_MESHES:
        path = patient_dir / MODEL_MESHES[mesh_name]["file"]
        _repair_nifti_backed_stl_if_needed(patient_dir, mesh_name, path)
        if mesh_name == "smooth":
            _repair_smooth_stl_if_needed(patient_dir, path)
    else:
        raise ValueError(f"Unknown mesh: {mesh_name}")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _nifti_world_bounds(path: Path) -> np.ndarray | None:
    try:
        import nibabel as nib
        from nibabel.affines import apply_affine
    except ImportError:
        return None
    if not path.exists():
        return None
    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape[:3])
    corners = np.asarray(
        [
            [0, 0, 0],
            [shape[0], 0, 0],
            [0, shape[1], 0],
            [0, 0, shape[2]],
            [shape[0], shape[1], 0],
            [shape[0], 0, shape[2]],
            [0, shape[1], shape[2]],
            [shape[0], shape[1], shape[2]],
        ],
        dtype=np.float64,
    )
    world = apply_affine(img.affine, corners)
    return np.stack([world.min(axis=0), world.max(axis=0)], axis=0)


def _stl_bounds(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        import trimesh

        mesh = trimesh.load(str(path), force="mesh")
        if hasattr(mesh, "geometry"):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if len(mesh.vertices) == 0:
            return None
        return np.asarray(mesh.bounds, dtype=np.float64)
    except Exception:
        mesh = _load_mesh_fallback(path, max_faces=200000)
        vertices = np.asarray((mesh or {}).get("vertices") or [], dtype=np.float64)
        if vertices.size == 0:
            return None
        return np.stack([vertices.min(axis=0), vertices.max(axis=0)], axis=0)


def _bounds_look_aligned(mesh_bounds: np.ndarray | None, reference_bounds: np.ndarray | None) -> bool:
    if mesh_bounds is None:
        return False
    if reference_bounds is None:
        return True
    ref_extent = np.maximum(reference_bounds[1] - reference_bounds[0], 1.0)
    ref_center = reference_bounds.mean(axis=0)
    mesh_center = mesh_bounds.mean(axis=0)
    distance = np.abs(mesh_center - ref_center)
    tolerance = np.maximum(ref_extent * 0.75, 80.0)
    return bool(np.all(distance <= tolerance))


def _bounds_match_for_smoothing(smooth_bounds: np.ndarray | None, predict_bounds: np.ndarray | None) -> bool:
    if smooth_bounds is None or predict_bounds is None:
        return False
    predict_extent = np.maximum(predict_bounds[1] - predict_bounds[0], 1.0)
    smooth_extent = np.maximum(smooth_bounds[1] - smooth_bounds[0], 1.0)
    center_distance = np.abs(smooth_bounds.mean(axis=0) - predict_bounds.mean(axis=0))
    extent_ratio = smooth_extent / predict_extent
    return bool(np.all(center_distance <= np.maximum(predict_extent * 0.08, 3.0)) and np.all((extent_ratio > 0.65) & (extent_ratio < 1.35)))


def _repair_nifti_backed_stl_if_needed(patient_dir: Path, mesh_name: str, stl_path: Path, force: bool = False) -> None:
    nii_name = NIFTI_BACKED_MESHES.get(mesh_name)
    if not nii_name:
        return
    nii_path = patient_dir / nii_name
    if not nii_path.exists():
        return
    rebuild = bool(force) or not stl_path.exists()
    if not rebuild and nii_path.stat().st_mtime > stl_path.stat().st_mtime:
        rebuild = True
    if not rebuild:
        reference = _nifti_world_bounds(patient_dir / "orig.nii.gz")
        if reference is None:
            reference = _nifti_world_bounds(nii_path)
        rebuild = not _bounds_look_aligned(_stl_bounds(stl_path), reference)
    if not rebuild:
        return
    try:
        import nibabel as nib
        from utils.common import nifti_mask_to_stl

        img = nib.load(str(nii_path))
        mask = np.asarray(img.dataobj) > 0
        if int(mask.sum()) == 0:
            raise ValueError(f"{nii_path.name} is empty; run refinement again before smoothing.")
        nifti_mask_to_stl(mask.astype(np.uint8), img.affine, stl_path, name=mesh_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to rebuild {stl_path.name} from {nii_path.name}: {exc}") from exc


def _repair_smooth_stl_if_needed(patient_dir: Path, smooth_path: Path) -> None:
    predict_path = patient_dir / "predict.stl"
    if not predict_path.exists():
        return
    rebuild = not smooth_path.exists()
    if not rebuild and predict_path.stat().st_mtime > smooth_path.stat().st_mtime:
        rebuild = True
    if not rebuild:
        rebuild = not _bounds_match_for_smoothing(_stl_bounds(smooth_path), _stl_bounds(predict_path))
    if not rebuild:
        return
    try:
        from utils.common import smooth_stl

        smooth_stl(predict_path, smooth_path, iterations=8)
    except Exception as exc:
        raise RuntimeError(f"Failed to rebuild {smooth_path.name} from predict.stl: {exc}") from exc
    if not _bounds_match_for_smoothing(_stl_bounds(smooth_path), _stl_bounds(predict_path)):
        smooth_path.write_bytes(predict_path.read_bytes())


def _mesh_style(mesh_name: str) -> dict:
    if mesh_name.startswith("organ:"):
        organ = mesh_name.split(":", 1)[1]
        return {
            "label": ORGAN_LABELS.get(organ, organ),
            "color": ORGAN_COLORS.get(organ, "#94a3b8"),
            "kind": "organ",
        }
    info = MODEL_MESHES.get(mesh_name, {})
    return {"label": info.get("label", mesh_name), "color": info.get("color", "#94a3b8"), "kind": "model"}


def _load_mesh(stl_path: Path, max_faces: int = 80000) -> dict | None:
    try:
        return _load_mesh_with_trimesh(stl_path, max_faces=max_faces)
    except Exception:
        return _load_mesh_fallback(stl_path, max_faces=max_faces)


def _load_mesh_with_trimesh(stl_path: Path, max_faces: int = 80000) -> dict | None:
    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh")
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return None
    return _compact_sampled_mesh(
        np.asarray(mesh.vertices, dtype=float),
        np.asarray(mesh.faces, dtype=np.int64),
        max_faces=max_faces,
        source="trimesh",
    )


def _load_mesh_fallback(stl_path: Path, max_faces: int = 80000) -> dict | None:
    try:
        with stl_path.open("rb") as f:
            _header = f.read(80)
            raw = f.read(4)
            if len(raw) == 4:
                n_faces = struct.unpack("<I", raw)[0]
                if n_faces > 0 and stl_path.stat().st_size == 84 + n_faces * 50:
                    return _read_binary_stl_sampled(f, n_faces, max_faces)
        return _read_ascii_stl_sampled(stl_path, max_faces)
    except Exception:
        return None


def _read_binary_stl_sampled(handle, n_faces: int, max_faces: int) -> dict:
    stride = max(1, int(math.ceil(n_faces / max_faces)))
    vertices = []
    faces = []
    for idx in range(n_faces):
        chunk = handle.read(50)
        if len(chunk) < 50:
            break
        if idx % stride:
            continue
        vals = struct.unpack("<12fH", chunk)
        tri = vals[3:12]
        base = len(vertices)
        vertices.extend([[tri[0], tri[1], tri[2]], [tri[3], tri[4], tri[5]], [tri[6], tri[7], tri[8]]])
        faces.append([base, base + 1, base + 2])
    return {
        "vertices": vertices,
        "faces": faces,
        "n_faces": int(n_faces),
        "n_faces_rendered": int(len(faces)),
        "source": "stdlib-binary-stl",
    }


def _read_ascii_stl_sampled(stl_path: Path, max_faces: int) -> dict | None:
    tris = []
    current = []
    with stl_path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("vertex"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            current.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(current) == 3:
                tris.append(current)
                current = []
    if not tris:
        return None
    stride = max(1, int(math.ceil(len(tris) / max_faces)))
    vertices = []
    faces = []
    for tri in tris[::stride]:
        base = len(vertices)
        vertices.extend(tri)
        faces.append([base, base + 1, base + 2])
    return {
        "vertices": vertices,
        "faces": faces,
        "n_faces": int(len(tris)),
        "n_faces_rendered": int(len(faces)),
        "source": "stdlib-ascii-stl",
    }


def _compact_sampled_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int, source: str) -> dict:
    n_faces = int(len(faces))
    if n_faces == 0 or len(vertices) == 0:
        return {"vertices": [], "faces": [], "n_faces": n_faces, "n_faces_rendered": 0, "source": source}
    if n_faces > max_faces:
        faces = faces[:: int(math.ceil(n_faces / max_faces))]
    used = np.unique(faces.reshape(-1))
    remap = {int(old): i for i, old in enumerate(used)}
    compact_faces = np.asarray([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces], dtype=np.int64)
    return {
        "vertices": np.round(vertices[used], 5).tolist(),
        "faces": compact_faces.tolist(),
        "n_faces": n_faces,
        "n_faces_rendered": int(len(compact_faces)),
        "source": source,
    }


def _normalize_slice(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (out * 255).astype(np.uint8)


def _downsample_step(shape: tuple[int, int], max_side: int = 384) -> int:
    h, w = shape[:2]
    scale = max(h / max_side, w / max_side, 1.0)
    return max(1, int(math.ceil(scale)))


def _downsample_2d(arr: np.ndarray, max_side: int = 384, step: int | None = None) -> np.ndarray:
    if step is None:
        step = _downsample_step(arr.shape[:2], max_side=max_side)
    return arr[::step, ::step]


def _slice_volume(data: np.ndarray, ax: int, index: int) -> np.ndarray:
    if ax == 0:
        return data[index, :, :].T
    if ax == 1:
        return data[:, index, :].T
    return data[:, :, index].T


def _voxel_from_slice_point(axis: str, index: int, x: int, y: int, step: int, shape: tuple[int, int, int]) -> np.ndarray:
    i = int(round(x * step))
    j = int(round(y * step))
    k = int(index)
    if axis == "sagittal":
        voxel = np.asarray([index, i, j], dtype=np.int64)
    elif axis == "coronal":
        voxel = np.asarray([i, index, j], dtype=np.int64)
    else:
        voxel = np.asarray([i, j, index], dtype=np.int64)
    hi = np.asarray(shape, dtype=np.int64) - 1
    return np.minimum(np.maximum(voxel, 0), hi)


def _load_ct_slice(patient_dir: Path, axis: str, index: int | None = None, mask_ids: list[str] | None = None) -> dict:
    import nibabel as nib

    path = patient_dir / "orig.nii.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    shape = tuple(int(v) for v in data.shape)
    axis_map = {"sagittal": 0, "coronal": 1, "axial": 2}
    ax = axis_map.get(axis, 2)
    max_idx = shape[ax] - 1
    if index is None:
        index = max_idx // 2
    index = max(0, min(int(index), max_idx))
    sl = _slice_volume(data, ax, index)
    step = _downsample_step(sl.shape[:2])
    sl_ds = _downsample_2d(sl, step=step)
    img8 = _normalize_slice(sl_ds)
    finite = sl_ds[np.isfinite(sl_ds)]
    if finite.size:
        win_lo, win_hi = np.percentile(finite, [1, 99])
        default_level = float((win_lo + win_hi) / 2.0)
        default_window = float(max(win_hi - win_lo, 1.0))
    else:
        default_level = 40.0
        default_window = 400.0
    overlays = []
    for idx, mask_id in enumerate(mask_ids or []):
        try:
            mask_path = _mask_path(patient_dir, mask_id)
            if not mask_path.exists():
                continue
            mask_img = nib.load(str(mask_path))
            mask = np.asarray(mask_img.dataobj) > 0
            if tuple(mask.shape[:3]) != shape:
                continue
            mask8 = (_downsample_2d(_slice_volume(mask.astype(np.uint8), ax, index), step=step) * 255).astype(np.uint8)
            overlays.append({
                "id": mask_id,
                "color": _mask_color(mask_id, idx),
                "pixels": mask8.tolist(),
            })
        except Exception:
            continue
    return {
        "axis": axis,
        "index": index,
        "max_index": max_idx,
        "shape": shape,
        "width": int(img8.shape[1]),
        "height": int(img8.shape[0]),
        "step": int(step),
        "affine": np.asarray(img.affine, dtype=float).round(8).tolist(),
        "inv_affine": np.asarray(np.linalg.inv(img.affine), dtype=float).round(8).tolist(),
        "window": default_window,
        "level": default_level,
        "values": np.round(sl_ds, 3).tolist(),
        "pixels": img8.tolist(),
        "masks": overlays,
    }


def _new_job(session_id: str, steps: list[str], patients: list[dict], payload: dict) -> dict:
    job_id = uuid.uuid4().hex[:12]
    total = max(1, len(steps) * len(patients))
    job = {
        "id": job_id,
        "session_id": session_id,
        "status": "running",
        "created": _now(),
        "updated": _now(),
        "steps": steps,
        "total": total,
        "completed": 0,
        "current": "",
        "logs": [],
        "errors": [],
        "results": {},
        "_patients_runtime": patients,
        "_payload_runtime": payload,
    }
    with STATE_LOCK:
        JOBS[job_id] = job
    return job


def _append_job_log(job: dict, message: str) -> None:
    with STATE_LOCK:
        job["logs"].append(message)
        job["logs"] = job["logs"][-600:]
        job["updated"] = _now()


def _set_job_progress(job: dict, current: str | None = None, completed_delta: int = 0) -> None:
    with STATE_LOCK:
        if current is not None:
            job["current"] = current
        job["completed"] += completed_delta
        job["updated"] = _now()


def _run_command(command: list[str], cwd: Path, job: dict) -> None:
    _append_job_log(job, "[cmd] " + " ".join(command))
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail.append(line)
            if len(tail) >= 10:
                _append_job_log(job, "\n".join(tail))
                tail = []
    if tail:
        _append_job_log(job, "\n".join(tail))
    rc = proc.wait()
    if rc:
        raise RuntimeError(f"command failed with exit code {rc}")


def _step_already_done(patient_dir: Path, step: str, payload: dict | None = None) -> bool:
    if step == "totalseg":
        seg_dir = patient_dir / "segmentation"
        structures = (payload or {}).get("structures") or [
            "bone_all",
            "spleen",
            "liver",
            "kidney_left",
            "kidney_right",
            "inferior_vena_cava",
            "aorta",
            "portal_vein",
        ]
        return all((seg_dir / f"{name}.stl").exists() and (seg_dir / f"{name}.nii.gz").exists() for name in structures)
    if step == "pretrain":
        return (patient_dir / "pretrain.stl").exists() and (patient_dir / "pretrain.nii.gz").exists()
    if step == "refine":
        return (patient_dir / "predict.stl").exists() and (patient_dir / "predict_mask.nii.gz").exists()
    return False


def _smooth_prediction_best_effort(patient_dir: Path, payload: dict, job: dict) -> None:
    """Try to write predict_smooth.stl, but keep predict.stl as the primary output."""
    if not (patient_dir / "predict.stl").exists():
        _append_job_log(job, "[smooth warning] predict.stl is missing; skipped smoothing")
        return
    case = SimpleNamespace(
        name=patient_dir.name,
        path=patient_dir,
        dcm_dir=patient_dir / "dcm",
        label_stl=patient_dir / "vessel.stl",
        pretrain_stl=patient_dir / "pretrain.stl",
        predict_stl=patient_dir / "predict.stl",
        is_post_tips="#" in patient_dir.name,
    )
    buffer = io.StringIO()
    try:
        from postprocess.check_and_smooth import check_and_smooth_case

        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            out = check_and_smooth_case(
                case,
                iterations=int(payload.get("smooth_iterations", 8)),
                force=True,
            )
        text = buffer.getvalue().strip()
        if text:
            _append_job_log(job, text)
        _append_job_log(job, f"[smooth] wrote {out}")
    except Exception as exc:
        text = buffer.getvalue().strip()
        if text:
            _append_job_log(job, text)
        _append_job_log(job, f"[smooth warning] {type(exc).__name__}: {exc}; predict.stl remains the main result")


def _run_pipeline_step(step: str, patient_dir: Path, payload: dict, job: dict) -> None:
    force = bool(payload.get("force", False))
    if not force and _step_already_done(patient_dir, step, payload):
        _append_job_log(job, f"[skip] {patient_dir.name} / {STEP_LABELS[step]} outputs already exist")
        return

    py = sys.executable
    if step == "totalseg":
        structures = payload.get("structures") or ["bone_all", "spleen", "liver", "kidney_left", "kidney_right", "inferior_vena_cava", "aorta", "portal_vein"]
        cmd = [
            py,
            str(APP_ROOT / "totalseg.py"),
            "--data_root",
            str(patient_dir),
            "--patient",
            patient_dir.name,
            "--device",
            str(payload.get("device") or "gpu"),
            "--structures",
            *[str(s) for s in structures],
        ]
        if force:
            cmd.append("--force")
        else:
            cmd.append("--resume")
        if bool(payload.get("fast", True)):
            cmd.append("--fast")
        else:
            cmd.append("--no-fast")
        _run_command(cmd, APP_ROOT, job)
    elif step == "pretrain":
        from pretrain.preprocess import pretrain_patient

        case = SimpleNamespace(
            name=patient_dir.name,
            path=patient_dir,
            dcm_dir=patient_dir / "dcm",
            label_stl=patient_dir / "vessel.stl",
            pretrain_stl=patient_dir / "pretrain.stl",
            predict_stl=patient_dir / "predict.stl",
            is_post_tips="#" in patient_dir.name,
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result = pretrain_patient(case, force=force)
        text = buffer.getvalue().strip()
        if text:
            _append_job_log(job, text)
        _append_job_log(job, f"[pretrain] wrote {result.path} ({result.status})")
    elif step == "refine":
        checkpoint = Path(str(payload.get("checkpoint") or _available_checkpoint() or ""))
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        cmd = [
            py,
            str(APP_ROOT / "refinement" / "predict.py"),
            "--data_root",
            str(patient_dir),
            "--patient",
            patient_dir.name,
            "--checkpoint",
            str(checkpoint),
            "--threshold",
            str(float(payload.get("threshold", 0.5))),
        ]
        _run_command(cmd, APP_ROOT, job)
        _smooth_prediction_best_effort(patient_dir, payload, job)
    else:
        raise ValueError(f"Unknown step: {step}")


def _run_job(job_id: str) -> None:
    with STATE_LOCK:
        job = JOBS[job_id]
        patients = list(job.pop("_patients_runtime", []))
        payload = dict(job.pop("_payload_runtime", {}))
        steps = list(job["steps"])
    try:
        for patient in patients:
            patient_dir = Path(patient["folder"])
            for step in steps:
                label = STEP_LABELS.get(step, step)
                _set_job_progress(job, current=f"{patient['id']} - {label}")
                started = time.time()
                ok = True
                try:
                    _run_pipeline_step(step, patient_dir, payload, job)
                except Exception as exc:
                    ok = False
                    _append_job_log(job, traceback.format_exc())
                    with STATE_LOCK:
                        job["errors"].append(f"{patient['id']} / {label}: {type(exc).__name__}: {exc}")
                elapsed = time.time() - started
                _append_job_log(job, f"[{'OK' if ok else 'FAIL'}] {patient['id']} / {label} ({elapsed:.1f}s)")
                with STATE_LOCK:
                    job["results"].setdefault(patient["id"], {})[step] = ok
                _set_job_progress(job, completed_delta=1)
        with STATE_LOCK:
            job["status"] = "failed" if job["errors"] else "done"
            job["current"] = ""
            job["updated"] = _now()
    except Exception as exc:
        with STATE_LOCK:
            job["status"] = "failed"
            job["errors"].append(f"{type(exc).__name__}: {exc}")
            job["logs"].append(traceback.format_exc())
            job["updated"] = _now()


def _edit_mesh(patient_dir: Path, payload: dict) -> dict:
    import trimesh

    source_name = str(payload.get("source") or "manual")
    mode = str(payload.get("mode") or "delete")
    radius = max(0.1, float(payload.get("radius") or 5.0))
    point = np.asarray(payload.get("point") or [], dtype=np.float64)
    if point.shape != (3,):
        raise ValueError("point must be [x, y, z]")

    manual = patient_dir / "manual_edit.stl"
    if source_name == "manual" and manual.exists():
        source = manual
    else:
        source = _mesh_path(patient_dir, source_name)
    mesh = trimesh.load(str(source), force="mesh")
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(faces) == 0:
        raise ValueError("source mesh has no faces")
    centroids = vertices[faces].mean(axis=1)
    inside = np.linalg.norm(centroids - point[None, :], axis=1) <= radius
    if mode == "keep":
        keep = inside
    elif mode == "delete":
        keep = ~inside
    else:
        raise ValueError("mode must be delete or keep")
    if not np.any(keep):
        raise ValueError("edit would remove all faces")
    edited = mesh.submesh([np.flatnonzero(keep)], append=True, repair=False)
    edited.export(str(manual))
    return {
        "path": str(manual),
        "removed_faces": int(np.count_nonzero(~keep)),
        "kept_faces": int(np.count_nonzero(keep)),
        "source": str(source),
    }


def _load_reference_nifti(patient_dir: Path):
    import nibabel as nib

    orig = patient_dir / "orig.nii.gz"
    if not orig.exists():
        raise FileNotFoundError(orig)
    return nib.load(str(orig))


def _save_mask_like(reference_img, mask: np.ndarray, out_path: Path) -> None:
    import nibabel as nib

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask, reference_img.affine, header), str(out_path))


def _create_mask(patient_dir: Path, payload: dict) -> dict:
    reference = _load_reference_nifti(patient_dir)
    name = _safe_mask_name(str(payload.get("name") or "manual_mask"))
    out_path = _mask_path(patient_dir, f"masks/{name}", create_parent=True)
    if out_path.exists() and not bool(payload.get("overwrite", False)):
        raise FileExistsError(f"mask already exists: {_patient_relative(patient_dir, out_path)}")
    mask = np.zeros(tuple(int(v) for v in reference.shape[:3]), dtype=np.uint8)
    source_id = str(payload.get("source") or "").strip()
    if source_id:
        source_path = _mask_path(patient_dir, source_id)
        if source_path.exists():
            import nibabel as nib

            mask = (np.asarray(nib.load(str(source_path)).dataobj) > 0).astype(np.uint8)
    _save_mask_like(reference, mask, out_path)
    return {"id": _patient_relative(patient_dir, out_path), "path": str(out_path), "voxels": int(mask.sum())}


def _threshold_mask(patient_dir: Path, payload: dict) -> dict:
    reference = _load_reference_nifti(patient_dir)
    data = np.asarray(reference.dataobj, dtype=np.float32)
    lo = float(payload.get("lower", -150.0))
    hi = float(payload.get("upper", 250.0))
    if hi < lo:
        lo, hi = hi, lo
    mask = (data >= lo) & (data <= hi)
    target = _mask_path(patient_dir, str(payload.get("target") or "masks/threshold.nii.gz"), create_parent=True)
    mode = str(payload.get("mode") or "replace")
    mask = _combine_with_existing_mask(target, mask, mode)
    _save_mask_like(reference, mask, target)
    _rebuild_known_mask_stl(patient_dir, target)
    return {"id": _patient_relative(patient_dir, target), "path": str(target), "voxels": int(mask.sum()), "mode": mode}


def _region_grow_mask(patient_dir: Path, payload: dict) -> dict:
    import nibabel as nib
    from nibabel.affines import apply_affine
    from scipy import ndimage

    reference = _load_reference_nifti(patient_dir)
    data = np.asarray(reference.dataobj, dtype=np.float32)
    if payload.get("voxel") is not None:
        seed = np.asarray(payload.get("voxel") or [], dtype=int)
        if seed.shape != (3,):
            raise ValueError("voxel must be [i, j, k]")
    else:
        point = np.asarray(payload.get("point") or [], dtype=np.float64)
        if point.shape != (3,):
            raise ValueError("point must be [x, y, z]")
        seed = np.rint(apply_affine(np.linalg.inv(reference.affine), point)).astype(int)
    shape = np.asarray(data.shape[:3], dtype=int)
    if np.any(seed < 0) or np.any(seed >= shape):
        raise ValueError("seed point is outside orig.nii.gz")
    seed_value = float(data[tuple(seed)])
    tolerance = max(0.0, float(payload.get("tolerance", 40.0)))
    lower = max(float(payload.get("lower", -np.inf)), seed_value - tolerance)
    upper = min(float(payload.get("upper", np.inf)), seed_value + tolerance)
    if upper < lower:
        lower, upper = upper, lower
    candidates = (data >= lower) & (data <= upper)
    if not bool(candidates[tuple(seed)]):
        raise ValueError("seed is outside the selected threshold range")
    labels, _num = ndimage.label(candidates, structure=np.ones((3, 3, 3), dtype=np.uint8))
    label = int(labels[tuple(seed)])
    grown = labels == label
    target = _mask_path(patient_dir, str(payload.get("target") or "masks/region_grow.nii.gz"), create_parent=True)
    mode = str(payload.get("mode") or "replace")
    mask = _combine_with_existing_mask(target, grown, mode)
    _save_mask_like(reference, mask, target)
    _rebuild_known_mask_stl(patient_dir, target)
    return {
        "id": _patient_relative(patient_dir, target),
        "path": str(target),
        "voxels": int(mask.sum()),
        "seed_value": seed_value,
        "mode": mode,
    }


def _combine_with_existing_mask(target: Path, incoming: np.ndarray, mode: str) -> np.ndarray:
    incoming = np.asarray(incoming) > 0
    if mode == "replace" or not target.exists():
        return incoming
    import nibabel as nib

    existing = np.asarray(nib.load(str(target)).dataobj) > 0
    if mode == "add":
        return existing | incoming
    if mode == "subtract":
        return existing & ~incoming
    if mode == "intersect":
        return existing & incoming
    raise ValueError("mode must be replace, add, subtract, or intersect")


def _boolean_masks(patient_dir: Path, payload: dict) -> dict:
    import nibabel as nib

    reference = _load_reference_nifti(patient_dir)
    left_path = _mask_path(patient_dir, str(payload.get("left") or ""))
    right_path = _mask_path(patient_dir, str(payload.get("right") or ""))
    if not left_path.exists() or not right_path.exists():
        raise FileNotFoundError("both mask operands must exist")
    left = np.asarray(nib.load(str(left_path)).dataobj) > 0
    right = np.asarray(nib.load(str(right_path)).dataobj) > 0
    op = str(payload.get("op") or "union")
    if op == "union":
        out = left | right
    elif op == "intersect":
        out = left & right
    elif op == "subtract":
        out = left & ~right
    else:
        raise ValueError("op must be union, intersect, or subtract")
    target = _mask_path(patient_dir, str(payload.get("target") or "masks/boolean.nii.gz"), create_parent=True)
    _save_mask_like(reference, out, target)
    _rebuild_known_mask_stl(patient_dir, target)
    return {"id": _patient_relative(patient_dir, target), "path": str(target), "voxels": int(out.sum()), "op": op}


def _smooth_mask(patient_dir: Path, payload: dict) -> dict:
    import nibabel as nib
    from scipy import ndimage

    reference = _load_reference_nifti(patient_dir)
    target = _mask_path(patient_dir, str(payload.get("target") or ""))
    if not target.exists():
        raise FileNotFoundError(target)
    mask = np.asarray(nib.load(str(target)).dataobj) > 0
    mode = str(payload.get("mode") or "closing")
    iterations = max(1, min(5, int(payload.get("iterations") or 1)))
    structure = ndimage.generate_binary_structure(3, 1)
    if mode == "closing":
        out = ndimage.binary_closing(mask, structure=structure, iterations=iterations)
    elif mode == "opening":
        out = ndimage.binary_opening(mask, structure=structure, iterations=iterations)
    elif mode == "median":
        size = iterations * 2 + 1
        out = ndimage.median_filter(mask.astype(np.uint8), size=size) > 0
    else:
        raise ValueError("mode must be closing, opening, or median")
    _save_mask_like(reference, out, target)
    _rebuild_known_mask_stl(patient_dir, target)
    return {
        "id": _patient_relative(patient_dir, target),
        "path": str(target),
        "voxels": int(out.sum()),
        "mode": mode,
        "iterations": iterations,
    }


def _fill_mask(patient_dir: Path, payload: dict) -> dict:
    import nibabel as nib
    from scipy import ndimage

    reference = _load_reference_nifti(patient_dir)
    target = _mask_path(patient_dir, str(payload.get("target") or ""))
    if not target.exists():
        raise FileNotFoundError(target)
    mask = np.asarray(nib.load(str(target)).dataobj) > 0
    out = ndimage.binary_fill_holes(mask)
    _save_mask_like(reference, out, target)
    _rebuild_known_mask_stl(patient_dir, target)
    return {
        "id": _patient_relative(patient_dir, target),
        "path": str(target),
        "voxels": int(out.sum()),
        "added_voxels": int(np.count_nonzero(out & ~mask)),
    }


def _voxel_spacing_from_affine(affine: np.ndarray) -> np.ndarray:
    spacing = np.linalg.norm(np.asarray(affine, dtype=np.float64)[:3, :3], axis=0)
    spacing[spacing <= 0] = 1.0
    return spacing


def _paint_mask(patient_dir: Path, payload: dict) -> dict:
    import nibabel as nib

    reference = _load_reference_nifti(patient_dir)
    shape = tuple(int(v) for v in reference.shape[:3])
    target = _mask_path(patient_dir, str(payload.get("target") or "masks/manual_mask.nii.gz"), create_parent=True)
    if target.exists():
        mask_img = nib.load(str(target))
        mask = np.asarray(mask_img.dataobj) > 0
        if tuple(mask.shape[:3]) != shape:
            raise ValueError("target mask shape does not match orig.nii.gz")
    else:
        mask = np.zeros(shape, dtype=bool)

    voxel = np.asarray(payload.get("voxel") or [], dtype=int)
    if voxel.shape != (3,):
        axis = str(payload.get("axis") or "axial")
        index = int(payload.get("index") or 0)
        x = int(payload.get("x") or 0)
        y = int(payload.get("y") or 0)
        step = max(1, int(payload.get("step") or 1))
        voxel = _voxel_from_slice_point(axis, index, x, y, step, shape)
    voxel = np.minimum(np.maximum(voxel, 0), np.asarray(shape, dtype=int) - 1)

    radius_mm = max(0.1, float(payload.get("radius_mm") or payload.get("radius") or 3.0))
    spacing = _voxel_spacing_from_affine(reference.affine)
    radius_vox = np.maximum(1, np.ceil(radius_mm / spacing).astype(int))
    mins = np.maximum(voxel - radius_vox, 0)
    maxs = np.minimum(voxel + radius_vox + 1, np.asarray(shape, dtype=int))
    grids = np.ogrid[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    dist = (
        ((grids[0] - voxel[0]) * spacing[0]) ** 2
        + ((grids[1] - voxel[1]) * spacing[1]) ** 2
        + ((grids[2] - voxel[2]) * spacing[2]) ** 2
    )
    brush = dist <= radius_mm ** 2
    region = mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    mode = str(payload.get("mode") or "add")
    if mode == "subtract":
        region[brush] = False
    elif mode == "add":
        region[brush] = True
    else:
        raise ValueError("paint mode must be add or subtract")
    mask[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] = region
    _save_mask_like(reference, mask, target)
    _rebuild_known_mask_stl(patient_dir, target)
    return {
        "id": _patient_relative(patient_dir, target),
        "path": str(target),
        "voxels": int(mask.sum()),
        "voxel": voxel.tolist(),
        "mode": mode,
    }


def _rebuild_known_mask_stl(patient_dir: Path, mask_path: Path) -> None:
    patient_dir = patient_dir.resolve()
    mask_path = mask_path.resolve()
    rel = _patient_relative(patient_dir, mask_path)
    if rel == "predict_mask.nii.gz":
        _repair_nifti_backed_stl_if_needed(patient_dir, "predict", patient_dir / "predict.stl", force=True)
    elif rel == "pretrain.nii.gz":
        _repair_nifti_backed_stl_if_needed(patient_dir, "pretrain", patient_dir / "pretrain.stl", force=True)


def _zip_patient_outputs(patients: list[dict]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for patient in patients:
            patient_dir = Path(patient["folder"])
            prefix = patient["id"]
            for name in OUTPUT_FILES:
                path = patient_dir / name
                if path.exists() and path.is_file():
                    zf.write(path, f"{prefix}/{name}")
            seg_dir = patient_dir / "segmentation"
            if seg_dir.exists():
                for path in sorted(seg_dir.glob("*.stl")):
                    zf.write(path, f"{prefix}/segmentation/{path.name}")
    return bio.getvalue()


class VKANWorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "VKANWorkbench/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                self._send_json({"ok": True, "time": _now(), "runtime": _runtime_info()})
            elif path.startswith("/api/session/") and path.endswith("/data"):
                self._handle_session_data(path)
            elif path.startswith("/api/session/") and path.endswith("/mesh"):
                self._handle_mesh(path, parsed.query)
            elif path.startswith("/api/session/") and path.endswith("/ct"):
                self._handle_ct(path, parsed.query)
            elif path.startswith("/api/session/") and path.endswith("/masks"):
                self._handle_masks(path, parsed.query)
            elif path.startswith("/api/session/") and path.endswith("/download"):
                self._handle_download(path, parsed.query)
            elif path.startswith("/api/job/"):
                self._handle_job(path)
            elif path == "/assets/plotly.min.js":
                self._serve_plotly()
            elif path == "/favicon.ico":
                self._send_bytes(b"", "image/x-icon", status=204)
            else:
                self._serve_static(path)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/session":
                self._handle_create_session()
            elif parsed.path == "/api/run":
                self._handle_run()
            elif parsed.path == "/api/edit":
                self._handle_edit()
            elif parsed.path == "/api/mask/create":
                self._handle_mask_create()
            elif parsed.path == "/api/mask/threshold":
                self._handle_mask_threshold()
            elif parsed.path == "/api/mask/region-grow":
                self._handle_mask_region_grow()
            elif parsed.path == "/api/mask/boolean":
                self._handle_mask_boolean()
            elif parsed.path == "/api/mask/paint":
                self._handle_mask_paint()
            elif parsed.path == "/api/mask/smooth":
                self._handle_mask_smooth()
            elif parsed.path == "/api/mask/fill":
                self._handle_mask_fill()
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)

    def log_message(self, fmt, *args):
        stream = getattr(sys, "stderr", None)
        if stream is not None:
            stream.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or "0")
        return self.rfile.read(length) if length else b""

    def _read_json_body(self) -> dict:
        body = self._read_body()
        if not body:
            return {}
        return json.loads(body.decode("utf-8"))

    def _send_json(self, data, status: int = 200):
        payload = json.dumps(_sanitize_json(data), ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, data: bytes, content_type: str, status=200, extra_headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, path: str):
        if path in ("", "/"):
            file_path = STATIC_ROOT / "index.html"
        else:
            rel = Path(path.lstrip("/"))
            file_path = (STATIC_ROOT / rel).resolve()
            if not str(file_path).startswith(str(STATIC_ROOT.resolve())):
                self._send_json({"error": "Forbidden"}, status=403)
                return
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "Not found"}, status=404)
            return
        ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self._send_bytes(file_path.read_bytes(), ctype)

    def _serve_plotly(self):
        try:
            import plotly

            path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
            if path.exists():
                self._send_bytes(path.read_bytes(), "application/javascript; charset=utf-8")
                return
        except Exception:
            pass
        self._send_json({"error": "Local Plotly asset not found"}, status=404)

    def _session_and_patient(self, path: str, query: str = "") -> tuple[dict, dict]:
        session_id = path.split("/")[3]
        qs = parse_qs(query)
        patient_id = (qs.get("patient") or [None])[0]
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Session not found")
        patient = _resolve_patient(session, patient_id)
        if not patient:
            raise ValueError("Patient not found")
        return session, patient

    def _handle_create_session(self):
        session = _create_session(self._read_json_body())
        self._send_json({"session": session})

    def _handle_session_data(self, path: str):
        session, _patient = self._session_and_patient(path)
        data = []
        for patient in session.get("patients") or []:
            record = dict(patient)
            record["status"] = _patient_status(Path(patient["folder"]))
            data.append(record)
        self._send_json({"session": session, "patients": data, "model_meshes": MODEL_MESHES})

    def _handle_mesh(self, path: str, query: str):
        _session, patient = self._session_and_patient(path, query)
        qs = parse_qs(query)
        mesh_name = (qs.get("mesh") or ["predict"])[0]
        max_faces = int((qs.get("max_faces") or ["80000"])[0])
        stl = _mesh_path(Path(patient["folder"]), mesh_name)
        self._send_json({
            "mesh_name": mesh_name,
            "style": _mesh_style(mesh_name),
            "path": str(stl),
            "mesh": _load_mesh(stl, max_faces=max_faces),
        })

    def _handle_ct(self, path: str, query: str):
        _session, patient = self._session_and_patient(path, query)
        qs = parse_qs(query)
        axis = (qs.get("axis") or ["axial"])[0]
        index_value = (qs.get("index") or [None])[0]
        index = int(index_value) if index_value not in (None, "") else None
        mask_ids = []
        for raw in qs.get("mask") or []:
            mask_ids.extend([item for item in raw.split(",") if item])
        self._send_json(_load_ct_slice(Path(patient["folder"]), axis, index, mask_ids=mask_ids))

    def _handle_masks(self, path: str, query: str):
        _session, patient = self._session_and_patient(path, query)
        self._send_json({"masks": _mask_catalog(Path(patient["folder"]))})

    def _handle_run(self):
        payload = self._read_json_body()
        session_id = str(payload.get("session_id") or "")
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session")
        steps = payload.get("steps") or []
        steps = [step for step in steps if step in PIPELINE_STEPS]
        if not steps:
            raise ValueError("No valid steps selected")
        patient_id = payload.get("patient_id")
        patients = session.get("patients") or []
        if patient_id and patient_id != "all":
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
        if not patients:
            raise ValueError("No patients selected")
        job = _new_job(session_id, steps, patients, payload)
        thread = threading.Thread(target=_run_job, args=(job["id"],), daemon=True)
        thread.start()
        self._send_json({"job": job})

    def _handle_job(self, path: str):
        job_id = path.rstrip("/").split("/")[-1]
        with STATE_LOCK:
            job = JOBS.get(job_id)
        if not job:
            self._send_json({"error": "Job not found"}, status=404)
            return
        self._send_json({"job": job})

    def _handle_edit(self):
        payload = self._read_json_body()
        session_id = str(payload.get("session_id") or "")
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session")
        patient = _resolve_patient(session, str(payload.get("patient_id") or ""))
        if not patient:
            raise ValueError("Patient not found")
        result = _edit_mesh(Path(patient["folder"]), payload)
        self._send_json({"edit": result})

    def _patient_from_payload(self, payload: dict) -> dict:
        session_id = str(payload.get("session_id") or "")
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session")
        patient = _resolve_patient(session, str(payload.get("patient_id") or ""))
        if not patient:
            raise ValueError("Patient not found")
        return patient

    def _handle_mask_create(self):
        payload = self._read_json_body()
        patient = self._patient_from_payload(payload)
        result = _create_mask(Path(patient["folder"]), payload)
        self._send_json({"mask": result, "masks": _mask_catalog(Path(patient["folder"]))})

    def _handle_mask_threshold(self):
        payload = self._read_json_body()
        patient = self._patient_from_payload(payload)
        result = _threshold_mask(Path(patient["folder"]), payload)
        self._send_json({"mask": result, "masks": _mask_catalog(Path(patient["folder"]))})

    def _handle_mask_region_grow(self):
        payload = self._read_json_body()
        patient = self._patient_from_payload(payload)
        result = _region_grow_mask(Path(patient["folder"]), payload)
        self._send_json({"mask": result, "masks": _mask_catalog(Path(patient["folder"]))})

    def _handle_mask_boolean(self):
        payload = self._read_json_body()
        patient = self._patient_from_payload(payload)
        result = _boolean_masks(Path(patient["folder"]), payload)
        self._send_json({"mask": result, "masks": _mask_catalog(Path(patient["folder"]))})

    def _handle_mask_paint(self):
        payload = self._read_json_body()
        patient = self._patient_from_payload(payload)
        result = _paint_mask(Path(patient["folder"]), payload)
        self._send_json({"mask": result, "masks": _mask_catalog(Path(patient["folder"]))})

    def _handle_mask_smooth(self):
        payload = self._read_json_body()
        patient = self._patient_from_payload(payload)
        result = _smooth_mask(Path(patient["folder"]), payload)
        self._send_json({"mask": result, "masks": _mask_catalog(Path(patient["folder"]))})

    def _handle_mask_fill(self):
        payload = self._read_json_body()
        patient = self._patient_from_payload(payload)
        result = _fill_mask(Path(patient["folder"]), payload)
        self._send_json({"mask": result, "masks": _mask_catalog(Path(patient["folder"]))})

    def _handle_download(self, path: str, query: str):
        session, _patient = self._session_and_patient(path, query)
        qs = parse_qs(query)
        patient_id = (qs.get("patient") or ["all"])[0]
        if patient_id == "all":
            patients = session.get("patients") or []
            name = f"vkan_outputs_{session['id']}.zip"
        else:
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
            name = f"vkan_outputs_{patient_id}.zip"
        payload = _zip_patient_outputs(patients)
        self._send_bytes(payload, "application/zip", extra_headers={"Content-Disposition": f'attachment; filename="{name}"'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--conda-exe", default=None)
    parser.add_argument("--no-conda-reexec", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = _load_config(args.config)
    args.host = args.host or str(config.get("host") or "127.0.0.1")
    args.port = args.port or int(config.get("port") or 8775)
    args.conda_env = args.conda_env if args.conda_env is not None else str(config.get("conda_env") or "").strip() or None
    args.conda_exe = args.conda_exe if args.conda_exe is not None else str(config.get("conda_exe") or "").strip() or None
    _maybe_reexec_in_conda(args, sys.argv[1:])
    server = ThreadingHTTPServer((args.host, args.port), VKANWorkbenchHandler)
    print(f"VKAN workbench running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
