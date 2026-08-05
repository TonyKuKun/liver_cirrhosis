"""
Local browser workbench for the portal-vein STL pipeline.

Run:
    python web/geometry_backend.py --host 127.0.0.1 --port 8765

The server intentionally uses the standard library so the UI can start before
the scientific pipeline dependencies are installed. Pipeline steps still use
the existing repository modules and will report missing dependencies in the job
log when they are unavailable.
"""

from __future__ import annotations

import argparse
import contextlib
import cgi
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
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

WEB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WEB_ROOT.parent
APP_ROOT = REPO_ROOT / "geometry_feature_extract"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry_feature_extract.features_layout import (
    FEATURES_DIRNAME,
    POINTWISE_TEMP_NAME,
    PUBLIC_FEATURE_NAMES,
    RAW_CENTERLINE_NAME,
    SMOOTH_CENTERLINE_NAME,
    SEGMENT_ASSIGNMENTS_NAME,
    UNIFIED_FEATURES_NAME,
    feature_path,
    features_dir,
    remove_generated_outputs,
    resolve_feature_path,
)


STATIC_ROOT = WEB_ROOT / "geometry"
RUNS_ROOT = WEB_ROOT / "geometry_runs"
DEFAULT_CONFIG_PATH = WEB_ROOT / "geometry_backend_config.json"
WEB_FRONTEND_VERSION = "analysis-ranges-pointwise-only-20260804-v3"
MANUAL_SEGMENT_FILE = SEGMENT_ASSIGNMENTS_NAME
POINTWISE_ANALYSIS_ZERO_KEYS = {
    "area",
    "perimeter",
    "eq_diameter",
    "raw_area",
    "raw_perimeter",
    "raw_eq_diameter",
    "anchor_radius",
    "owned_radius",
    "hydraulic_diameter",
    "circularity",
    "solidity",
    "r_insc_to_r_eq_ratio",
    "curvature",
    "torsion",
    "dA_ds_norm",
    "inscribed_radius",
    "section_normal_offset_deg",
    "implausibly_small_section",
}

RUNS_ROOT.mkdir(exist_ok=True)


def cleanup_runs(keep: int = 5) -> dict:
    """Remove old generated run folders while retaining the newest ones.

    ``geometry_runs`` contains uploaded STL files and generated outputs for
    single-file sessions.  They are temporary and can be regenerated, but the
    newest few runs are useful when restarting the local server during an
    active investigation.  Only direct child directories are considered.
    """
    try:
        keep = max(0, int(keep))
    except (TypeError, ValueError):
        keep = 5
    runs = [path for path in RUNS_ROOT.iterdir() if path.is_dir()]
    runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    removed = 0
    errors = []
    for path in runs[keep:]:
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            errors.append(f"{path.name}: {exc}")
    return {"found": len(runs), "kept": min(keep, len(runs)), "removed": removed, "errors": errors}

SEGMENT_COLORS = {
    "mpv": "#ff3333",
    "sv": "#3380ff",
    "smv": "#ff9933",
    "lpv": "#b34dff",
    "rpv": "#33e666",
    "tips": "#00c7c7",
    "lgv": "#d6a800",
    "pgv": "#ff4dee",
}

SEGMENT_LABELS = {
    "mpv": "MPV",
    "sv": "SV",
    "smv": "SMV",
    "lpv": "LPV",
    "rpv": "RPV",
    "tips": "TIPS",
    "lgv": "LGV",
    "pgv": "PGV",
}

PIPELINE_STEPS = [
    "centerline",
    "smooth",
    "segment",
    "features",
    "export",
]

STEP_LABELS = {
    "centerline": "Centerline extraction",
    "smooth": "Centerline smoothing",
    "segment": "Anatomical segmentation",
    "features": "Pointwise and unified feature extraction",
    "export": "Visualization export",
}

DEFAULT_PARAMS = {
    "pitch": 0.5,
    "min_branch_length_mm": 10.0,
    "min_relative_length": 0.05,
    "min_radius_ratio": 0.4,
    "keep_radius_ratio": 0.55,
    "absolute_min_branch_length_mm": 3.0,
    "absolute_min_radius_mm": 0.5,
    "merge_bp_distance_mm": 5.0,
    "n_fit_points": 10,
    "angle_fit_length_mm": 10.0,
    "n_profile_points": 200,
    "curvature_window": 7,
    "sample_step": 3,
}

OUTPUT_FILES = [
    *PUBLIC_FEATURE_NAMES,
]

STEP_OUTPUTS = {
    "centerline": [RAW_CENTERLINE_NAME],
    "smooth": [SMOOTH_CENTERLINE_NAME],
    "segment": [SEGMENT_ASSIGNMENTS_NAME],
    "features": [POINTWISE_TEMP_NAME, UNIFIED_FEATURES_NAME],
    "export": [],
}

SESSIONS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
STATE_LOCK = threading.Lock()


def _now() -> float:
    return time.time()


def _load_config(path: str | Path | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = WEB_ROOT / config_path
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        raise ValueError(f"Failed to read config {config_path}: {exc}") from exc


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
    raise FileNotFoundError(
        "Could not locate conda. Pass --conda-exe or start from an Anaconda prompt."
    )


def _maybe_reexec_in_conda(args, argv: list[str]) -> None:
    requested = (args.conda_env or "").strip()
    if not requested or args.no_conda_reexec:
        return
    current = os.environ.get("CONDA_DEFAULT_ENV") or ""
    if current == requested:
        return

    conda_exe = _find_conda_exe(args.conda_exe)
    script = str(Path(__file__).resolve())
    forwarded = []
    skip_next = False
    consumed_with_value = {"--conda-env", "--conda-exe", "--config", "--host", "--port"}
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item in consumed_with_value:
            skip_next = True
            continue
        if any(item.startswith(flag + "=") for flag in consumed_with_value):
            continue
        if item == "--no-conda-reexec":
            continue
        forwarded.append(item)

    command = [
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
    if args.config:
        command.extend(["--config", str(args.config)])
    if args.conda_exe:
        command.extend(["--conda-exe", str(args.conda_exe)])
    command.extend(forwarded)

    if getattr(sys, "stdout", None) is not None:
        print(f"Restarting web frontend in conda env: {requested}")
    raise SystemExit(subprocess.call(command, cwd=str(APP_ROOT)))


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
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
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_post_tips(folder_name: str) -> bool:
    return "#" in folder_name


def _safe_float(value, default=None):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _merge_params(user_params: dict | None) -> dict:
    params = dict(DEFAULT_PARAMS)
    if not isinstance(user_params, dict):
        return params
    for key, default in DEFAULT_PARAMS.items():
        if key not in user_params:
            continue
        if isinstance(default, int) and not isinstance(default, bool):
            params[key] = _safe_int(user_params[key], default)
        elif isinstance(default, float):
            params[key] = _safe_float(user_params[key], default)
        else:
            params[key] = user_params[key]
    return params


def _new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]


def _patient_record(stl_path: Path) -> dict:
    folder = stl_path.parent
    return {
        "id": folder.name or stl_path.stem,
        "folder": str(folder),
        "stl_path": str(stl_path),
        "stl_name": stl_path.name,
        "is_post_tips": _is_post_tips(folder.name),
    }


def _discover_batch(root_folder: Path, stl_name: str) -> list[dict]:
    patients: list[dict] = []
    if not root_folder.exists():
        return patients
    direct = root_folder / stl_name
    if direct.exists():
        patients.append(_patient_record(direct))
    for child in sorted(root_folder.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        stl = child / stl_name
        if stl.exists():
            patients.append(_patient_record(stl))
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


def _read_centerline_file(path: Path):
    if not path.exists():
        return None
    nodes = {}
    try:
        with path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                while len(parts) < 7:
                    parts.append("-1")
                nid = int(float(parts[0]))
                nodes[nid] = {
                    "id": nid,
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                    "parent": int(float(parts[4])),
                    "left": int(float(parts[5])),
                    "right": int(float(parts[6])),
                }
    except Exception:
        return None
    return nodes


def _feature_file(parent: Path, name: str) -> Path:
    """Resolve a public feature file, preferring the canonical features dir."""
    return resolve_feature_path(parent, name) or feature_path(parent, name)


def _line_arrays_from_nodes(nodes: dict | None) -> dict | None:
    if not nodes:
        return None
    x, y, z = [], [], []
    seen = set()
    for nid, node in nodes.items():
        for nb in (node.get("parent"), node.get("left"), node.get("right")):
            if nb is None or nb < 0 or nb not in nodes:
                continue
            edge = tuple(sorted((nid, nb)))
            if edge in seen:
                continue
            seen.add(edge)
            other = nodes[nb]
            x.extend([node["x"], other["x"], None])
            y.extend([node["y"], other["y"], None])
            z.extend([node["z"], other["z"], None])
    return {"x": x, "y": y, "z": z, "n_nodes": len(nodes), "n_edges": len(seen)}


def _centerline_adjacency(nodes: dict) -> dict[int, set[int]]:
    adj = {int(nid): set() for nid in nodes}
    for nid, node in nodes.items():
        nid = int(nid)
        for nb in (node.get("parent"), node.get("left"), node.get("right")):
            if nb is None or nb < 0 or nb not in nodes:
                continue
            nb = int(nb)
            adj[nid].add(nb)
            adj[nb].add(nid)
    return adj


def _path_length_from_nodes(path: list[int], nodes: dict) -> float:
    if len(path) < 2:
        return 0.0
    coords = np.asarray([
        [nodes[nid]["x"], nodes[nid]["y"], nodes[nid]["z"]]
        for nid in path if nid in nodes
    ], dtype=float)
    if len(coords) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def _editable_centerline_branches(nodes: dict | None) -> list[dict]:
    """Return terminal endpoint-to-branchpoint paths that may be removed."""
    if not nodes:
        return []
    adj = _centerline_adjacency(nodes)
    endpoints = [nid for nid, nbs in adj.items() if len(nbs) == 1]
    out = []
    seen = set()

    for endpoint in endpoints:
        path = [endpoint]
        prev = None
        cur = endpoint
        while True:
            neighbors = [n for n in adj[cur] if n != prev]
            if len(neighbors) != 1:
                break
            nxt = neighbors[0]
            path.append(nxt)
            degree = len(adj[nxt])
            if degree != 2:
                if degree >= 3:
                    endpoint_to_junction = path
                    junction_to_endpoint = list(reversed(endpoint_to_junction))
                    key = (endpoint, nxt)
                    if key in seen:
                        break
                    seen.add(key)
                    coords = _coords_for_path(junction_to_endpoint, nodes)
                    if coords is None:
                        break
                    branch_id = f"{endpoint}:{nxt}"
                    out.append({
                        "id": branch_id,
                        "endpoint_id": int(endpoint),
                        "junction_id": int(nxt),
                        "path": [int(n) for n in junction_to_endpoint],
                        "x": coords[:, 0].tolist(),
                        "y": coords[:, 1].tolist(),
                        "z": coords[:, 2].tolist(),
                        "length_mm": _path_length_from_nodes(endpoint_to_junction, nodes),
                        "n_points": len(endpoint_to_junction),
                    })
                break
            prev = cur
            cur = nxt

    out.sort(key=lambda item: item["length_mm"], reverse=True)
    return out


def _path_edges(path: list[int]) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(a), int(b))))
        for a, b in zip(path, path[1:])
    }


def _atomic_centerline_segments(nodes: dict | None) -> list[dict]:
    """Return maximal paths whose internal nodes have no attached branch."""
    if not nodes:
        return []
    adj = _centerline_adjacency(nodes)
    anchors = sorted(nid for nid, nbs in adj.items() if len(nbs) != 2)
    seen_edges: set[tuple[int, int]] = set()
    out = []

    for anchor in anchors:
        for first in sorted(adj.get(anchor, set())):
            first_edge = tuple(sorted((anchor, first)))
            if first_edge in seen_edges:
                continue
            path = [anchor, first]
            seen_edges.add(first_edge)
            prev, cur = anchor, first
            while len(adj.get(cur, set())) == 2:
                nxt = next(nb for nb in adj[cur] if nb != prev)
                edge = tuple(sorted((cur, nxt)))
                if edge in seen_edges:
                    break
                path.append(nxt)
                seen_edges.add(edge)
                prev, cur = cur, nxt

            coords = _coords_for_path(path, nodes)
            if coords is None:
                continue
            start, end = int(path[0]), int(path[-1])
            segment_id = f"{min(start, end)}:{max(start, end)}"
            out.append({
                "id": segment_id,
                "start_id": start,
                "end_id": end,
                "start_degree": len(adj.get(start, set())),
                "end_degree": len(adj.get(end, set())),
                "path": [int(nid) for nid in path],
                "x": coords[:, 0].tolist(),
                "y": coords[:, 1].tolist(),
                "z": coords[:, 2].tolist(),
                "length_mm": _path_length_from_nodes(path, nodes),
                "n_points": len(path),
            })

    out.sort(key=lambda item: (item["start_id"], item["end_id"]))
    return out


def _saved_manual_assignments(parent: Path) -> dict:
    saved = _read_json_file(_feature_file(parent, SEGMENT_ASSIGNMENTS_NAME))
    if not isinstance(saved, dict):
        return {}
    assignments = saved.get("assignments") or {}
    return assignments if isinstance(assignments, dict) else {}


def _annotated_atomic_segments(nodes: dict | None, seg_data: dict | None, parent: Path) -> list[dict]:
    atoms = _atomic_centerline_segments(nodes)
    saved = _saved_manual_assignments(parent)
    segment_edges = {}
    if seg_data:
        for vessel, info in (seg_data.get("segments") or {}).items():
            if info and info.get("path"):
                segment_edges[vessel] = _path_edges(info["path"])

    for atom in atoms:
        existing = saved.get(atom["id"])
        if isinstance(existing, dict):
            atom["kept"] = bool(existing.get("kept", False))
            vessel = str(existing.get("vessel") or "").lower()
            atom["vessel"] = vessel if vessel in SEGMENT_LABELS else ""
            atom["assignment_source"] = "manual"
            continue

        atom_edges = _path_edges(atom["path"])
        vessel = next(
            (name for name, edges in segment_edges.items() if atom_edges and atom_edges <= edges),
            "",
        )
        atom["kept"] = True
        atom["vessel"] = vessel
        atom["assignment_source"] = "automatic" if vessel else "needs_assignment"
    return atoms


def _join_atomic_paths(paths: list[list[int]], vessel: str) -> list[int]:
    if not paths:
        return []
    edge_adj: dict[int, set[int]] = {}
    edges: set[tuple[int, int]] = set()
    for path in paths:
        for a, b in zip(path, path[1:]):
            a, b = int(a), int(b)
            edge = tuple(sorted((a, b)))
            edges.add(edge)
            edge_adj.setdefault(a, set()).add(b)
            edge_adj.setdefault(b, set()).add(a)
    if any(len(nbs) > 2 for nbs in edge_adj.values()):
        raise ValueError(f"{SEGMENT_LABELS[vessel]} assignments form a branch; one vessel must be a single path.")
    ends = sorted(nid for nid, nbs in edge_adj.items() if len(nbs) == 1)
    if len(ends) != 2:
        raise ValueError(f"{SEGMENT_LABELS[vessel]} assignments do not form one open continuous path.")

    result = [ends[0]]
    prev = None
    cur = ends[0]
    walked = set()
    while True:
        next_nodes = [nb for nb in edge_adj.get(cur, set()) if nb != prev]
        if not next_nodes:
            break
        nxt = next_nodes[0]
        walked.add(tuple(sorted((cur, nxt))))
        result.append(nxt)
        prev, cur = cur, nxt
    if walked != edges:
        raise ValueError(f"{SEGMENT_LABELS[vessel]} assignments are disconnected; assign one continuous vessel path.")
    return result


def _paths_touch(path_a: list[int] | None, path_b: list[int] | None) -> set[int]:
    if not path_a or not path_b:
        return set()
    return set(path_a) & set(path_b)


def save_manual_segment_assignments(
    stl_path: Path,
    assignments_payload: list[dict],
    recompute_features: bool = True,
) -> dict:
    parent = stl_path.parent
    smooth_path = _feature_file(parent, SMOOTH_CENTERLINE_NAME)
    raw_path = _feature_file(parent, RAW_CENTERLINE_NAME)
    smooth_nodes = _read_centerline_file(smooth_path)
    raw_nodes = _read_centerline_file(raw_path)
    nodes = smooth_nodes or raw_nodes
    if not nodes:
        raise ValueError("No centerline is available. Run centerline extraction or import a saved centerline first.")
    canonical_raw = feature_path(parent, RAW_CENTERLINE_NAME, create=True)
    if raw_path.exists() and raw_path.resolve() != canonical_raw.resolve():
        shutil.copy2(raw_path, canonical_raw)

    atoms = _atomic_centerline_segments(nodes)
    by_id = {item["id"]: item for item in atoms}
    received = {}
    for item in assignments_payload:
        segment_id = str(item.get("id") or "")
        if segment_id not in by_id:
            raise ValueError(f"Unknown atomic segment: {segment_id}")
        kept = bool(item.get("kept", False))
        vessel = str(item.get("vessel") or "").lower()
        if kept and vessel not in SEGMENT_LABELS:
            raise ValueError(f"Kept atomic segment {segment_id} must be assigned to a vessel.")
        received[segment_id] = {"kept": kept, "vessel": vessel if kept else ""}
    if set(received) != set(by_id):
        raise ValueError("Every atomic centerline segment must be kept/removed and assigned before saving.")

    is_post_tips = _is_post_tips(parent.name)
    used_vessels = {item["vessel"] for item in received.values() if item["kept"]}
    if is_post_tips and ({"lgv", "pgv"} & used_vessels):
        raise ValueError("Post-TIPS patients (#) cannot be assigned LGV or PGV.")

    paths_by_vessel = {name: [] for name in SEGMENT_LABELS}
    for segment_id, assignment in received.items():
        if assignment["kept"]:
            paths_by_vessel[assignment["vessel"]].append(by_id[segment_id]["path"])
    paths = {
        vessel: (_join_atomic_paths(segment_paths, vessel) if segment_paths else None)
        for vessel, segment_paths in paths_by_vessel.items()
    }

    if paths["mpv"] and paths["sv"] and paths["smv"]:
        common = _paths_touch(paths["sv"], paths["smv"]) & set(paths["mpv"])
        if not common:
            raise ValueError("SV and SMV must meet at the same endpoint and continue as MPV.")
    if paths["pgv"]:
        sv_middle = set(paths["sv"][1:-1]) if paths["sv"] and len(paths["sv"]) > 2 else set()
        if not (_paths_touch(paths["pgv"], paths["sv"]) & sv_middle):
            raise ValueError("PGV must attach to the middle portion of SV.")
    if paths["lgv"] and not _paths_touch(paths["lgv"], paths["mpv"]):
        raise ValueError("LGV must attach to MPV or its lower endpoint.")

    adj = _centerline_adjacency(nodes)
    endpoints = [nid for nid, nbs in adj.items() if len(nbs) == 1]
    branch_points = [nid for nid, nbs in adj.items() if len(nbs) >= 3]
    try:
        from geometry_feature_extract.segment_vessels import _build_output_json
    except ImportError:
        from segment_vessels import _build_output_json

    result = {
        "segments": paths,
        "has_compensation": bool(paths["lgv"] or paths["pgv"]),
        "compensation_type": "PGV" if paths["pgv"] else ("LGV" if paths["lgv"] else None),
    }
    output = _build_output_json(parent.name, is_post_tips, result, nodes, branch_points, endpoints)
    output["segment_vessels_version"] = "manual-atomic-assignment-v1"
    output["manual_assignment"] = True
    smoothed_vessels = _apply_manual_segment_smoothing(output, nodes)
    tree = _rebuild_smoothed_assignment_tree(output, nodes)
    centerline_path = feature_path(parent, SMOOTH_CENTERLINE_NAME, create=True)
    removed_outputs = remove_generated_outputs(
        parent, keep_public=False, preserve=(RAW_CENTERLINE_NAME,))
    _write_centerline_tree(centerline_path, tree)
    output["assignments"] = received
    output["manual_assignment_version"] = 1
    output_path = feature_path(parent, SEGMENT_ASSIGNMENTS_NAME, create=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    removed_outputs.extend(remove_generated_outputs(parent, keep_public=True))
    features_recomputed = False
    if recompute_features:
        try:
            from geometry_feature_extract.extract_profiles import extract_profiles
            from geometry_feature_extract.extract_features import extract_all_features
        except ImportError:
            from extract_profiles import extract_profiles
            from extract_features import extract_all_features

        extract_profiles(
            str(stl_path), n_points=DEFAULT_PARAMS["n_profile_points"])
        extract_all_features(str(stl_path), write_legacy=False)
        remove_generated_outputs(parent, keep_public=True)
        features_recomputed = feature_path(
            parent, UNIFIED_FEATURES_NAME).exists()
    return {
        "n_atomic_segments": len(atoms),
        "n_kept": sum(1 for item in received.values() if item["kept"]),
        "vessels": sorted(used_vessels),
        "smoothed_vessels": sorted(smoothed_vessels),
        "centerline_file": str(centerline_path),
        "segment_assignments_file": str(output_path),
        "features_recomputed": features_recomputed,
        "removed_outputs": removed_outputs,
    }


def _pointwise_profile_count(profile: dict | None) -> int:
    if not isinstance(profile, dict):
        return 0
    for key in ("area", "eq_diameter", "perimeter", "section_valid", "position"):
        values = profile.get(key)
        if isinstance(values, list) and values:
            return len(values)
    lengths = [len(value) for value in profile.values() if isinstance(value, list)]
    return max(set(lengths), key=lengths.count) if lengths else 0


def _positive_pointwise_value(value) -> bool:
    number = _safe_float(value)
    return number is not None and math.isfinite(number) and number > 0


def _pointwise_valid_mask(profile: dict | None) -> list[bool]:
    """Return the validity of every serialized pointwise sample."""
    count = _pointwise_profile_count(profile)
    if count <= 0 or not isinstance(profile, dict):
        return []
    core_keys = [
        key for key in ("area", "eq_diameter", "perimeter")
        if isinstance(profile.get(key), list) and len(profile[key]) == count
    ]
    section_valid = profile.get("section_valid")
    has_section_valid = (
        isinstance(section_valid, list) and len(section_valid) == count
    )
    valid = []
    for index in range(count):
        sample_valid = all(
            _positive_pointwise_value(profile[key][index])
            for key in core_keys
        ) if core_keys else True
        if has_section_valid:
            sample_valid = (
                sample_valid
                and _positive_pointwise_value(section_valid[index])
            )
        valid.append(sample_valid)
    return valid


def _pointwise_range_from_profile(profile: dict | None) -> dict:
    """Map leading/trailing invalid samples to exact sample-count fractions."""
    count = _pointwise_profile_count(profile)
    if count <= 0:
        return {
            "start_fraction": 0.0,
            "end_fraction": 1.0,
            "source": "full",
            "n_points": 0,
            "leading_invalid_points": 0,
            "trailing_invalid_points": 0,
        }

    valid = _pointwise_valid_mask(profile)

    valid_indices = [index for index, is_valid in enumerate(valid) if is_valid]
    if not valid_indices:
        start_index = 0
        end_index = count
        source = "full_no_valid_profile_samples"
    else:
        start_index = valid_indices[0]
        end_index = valid_indices[-1] + 1
        source = "pointwise_endpoint_mask"
    return {
        "start_fraction": float(start_index / count),
        "end_fraction": float(end_index / count),
        "source": source,
        "n_points": int(count),
        "leading_invalid_points": int(start_index),
        "trailing_invalid_points": int(count - end_index),
    }


def _pointwise_analysis_ranges(pointwise: dict | None) -> dict:
    ranges = {}
    for vessel, profile in (pointwise or {}).items():
        if str(vessel).startswith("_") or not isinstance(profile, dict):
            continue
        if _pointwise_profile_count(profile) <= 0:
            continue
        ranges[str(vessel).lower()] = _pointwise_range_from_profile(profile)
    return ranges


def _normalize_range(value, default: float) -> float:
    number = _safe_float(value, default)
    return float(min(1.0, max(0.0, number)))


def _mask_pointwise_profile(profile: dict, start: float, end: float) -> dict:
    count = _pointwise_profile_count(profile)
    if count <= 0:
        return dict(profile)
    start_index = min(count, max(0, int(math.floor(start * count + 0.5))))
    end_index = min(count, max(0, int(math.floor(end * count + 0.5))))
    keep = [start_index <= index < end_index for index in range(count)]
    originally_valid = _pointwise_valid_mask(profile)
    masked = dict(profile)
    for key in POINTWISE_ANALYSIS_ZERO_KEYS:
        values = profile.get(key)
        if isinstance(values, list) and len(values) == count:
            masked[key] = [value if keep[index] else 0.0 for index, value in enumerate(values)]

    masked["section_valid"] = [
        1.0 if keep[index] and originally_valid[index] else 0.0
        for index in range(count)
    ]
    masked["analysis_range_mask"] = [
        1.0 if retained else 0.0 for retained in keep
    ]
    masked["n_analysis_range_excluded"] = int(sum(not retained for retained in keep))
    return masked


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=True)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def save_analysis_ranges(stl_path: Path, ranges_payload: list[dict]) -> dict:
    parent = stl_path.parent
    pointwise_path = _feature_file(parent, POINTWISE_TEMP_NAME)
    pointwise = _read_json_file(pointwise_path)
    if not isinstance(pointwise, dict):
        raise ValueError("pointwise_profiles.json is required before saving an analysis range.")
    available = {
        str(name).lower() for name, profile in pointwise.items()
        if not str(name).startswith("_")
        and isinstance(profile, dict)
        and _pointwise_profile_count(profile) > 0
    }
    detected_ranges = _pointwise_analysis_ranges(pointwise)
    saved = {}
    for item in ranges_payload:
        vessel = str(item.get("vessel") or "").lower()
        if vessel not in available:
            continue
        detected = detected_ranges.get(vessel) or {}
        start = max(
            _normalize_range(item.get("start_fraction"), 0.0),
            _normalize_range(detected.get("start_fraction"), 0.0),
        )
        end = min(
            _normalize_range(item.get("end_fraction"), 1.0),
            _normalize_range(detected.get("end_fraction"), 1.0),
        )
        if end - start < 0.02:
            raise ValueError(f"{SEGMENT_LABELS.get(vessel, vessel)} valid analysis range is too short.")
        saved[vessel] = {
            "start_fraction": start,
            "end_fraction": end,
            "source": "pointwise_mask",
        }
    if not saved:
        raise ValueError("No valid vessel analysis ranges were supplied.")

    updated_pointwise = dict(pointwise)
    masked_counts = {}
    for vessel, range_info in saved.items():
        profile = pointwise.get(vessel)
        if not isinstance(profile, dict):
            continue
        masked = _mask_pointwise_profile(
            profile,
            range_info["start_fraction"],
            range_info["end_fraction"],
        )
        updated_pointwise[vessel] = masked
        masked_counts[vessel] = masked.get("n_analysis_range_excluded", 0)
    original_unified = _read_json_file(_feature_file(parent, UNIFIED_FEATURES_NAME))
    pointwise_path = feature_path(parent, POINTWISE_TEMP_NAME, create=True)
    unified_path = feature_path(parent, UNIFIED_FEATURES_NAME, create=True)
    try:
        _write_json_atomic(pointwise_path, updated_pointwise)
        app_root = str(APP_ROOT)
        if app_root not in sys.path:
            sys.path.insert(0, app_root)
        from extract_features import extract_all_features
        rebuilt_features = extract_all_features(str(stl_path), write_legacy=False)
        if not rebuilt_features or not unified_path.exists():
            raise RuntimeError("unified_features.json was not rebuilt.")
    except Exception:
        _write_json_atomic(pointwise_path, pointwise)
        if isinstance(original_unified, dict):
            _write_json_atomic(unified_path, original_unified)
        else:
            with contextlib.suppress(OSError):
                unified_path.unlink()
        raise

    return {
        "ranges": _pointwise_analysis_ranges(updated_pointwise),
        "masked_points": masked_counts,
        "pointwise_profiles_file": str(pointwise_path),
        "unified_features_file": str(unified_path),
        "unified_recomputed": True,
        "removed_outputs": [],
    }


def _rebuild_centerline_tree(nodes: dict, adj: dict[int, set[int]]) -> list[list[float | int]]:
    remaining = sorted(nid for nid in nodes if nid in adj)
    if not remaining:
        raise ValueError("Centerline would be empty after deletion.")

    root_candidates = [nid for nid in remaining if nodes[nid].get("parent") == -1]
    if root_candidates:
        root = root_candidates[0]
    else:
        endpoints = [nid for nid in remaining if len(adj.get(nid, set())) <= 1]
        root = endpoints[0] if endpoints else remaining[0]

    visited = {root}
    queue = deque([(root, -1)])
    bfs_order = []
    while queue:
        node, parent = queue.popleft()
        bfs_order.append((node, parent))
        for nb in sorted(adj.get(node, set())):
            if nb in visited:
                continue
            visited.add(nb)
            queue.append((nb, node))

    if len(visited) != len(remaining):
        largest = _largest_component_nodes(adj)
        dropped = set(remaining) - largest
        if not largest:
            raise ValueError("Centerline has no connected component after deletion.")
        root = root if root in largest else sorted(largest)[0]
        visited = {root}
        queue = deque([(root, -1)])
        bfs_order = []
        while queue:
            node, parent = queue.popleft()
            bfs_order.append((node, parent))
            for nb in sorted(adj.get(node, set())):
                if nb in visited or nb not in largest:
                    continue
                visited.add(nb)
                queue.append((nb, node))
        if dropped:
            print(f"       [warn] centerline edit dropped disconnected nodes: {len(dropped)}")

    old_to_new = {old: idx for idx, (old, _) in enumerate(bfs_order)}
    children = {old: [] for old, _ in bfs_order}
    for old, parent in bfs_order:
        if parent != -1 and parent in children:
            children[parent].append(old)

    tree = []
    for old, parent in bfs_order:
        node = nodes[old]
        child_ids = children[old]
        tree.append([
            old_to_new[old],
            float(node["x"]),
            float(node["y"]),
            float(node["z"]),
            old_to_new[parent] if parent != -1 else -1,
            old_to_new[child_ids[0]] if len(child_ids) >= 1 else -1,
            old_to_new[child_ids[1]] if len(child_ids) >= 2 else -1,
        ])
    return tree


def _largest_component_nodes(adj: dict[int, set[int]]) -> set[int]:
    remaining = set(adj)
    best = set()
    while remaining:
        start = next(iter(remaining))
        comp = {start}
        queue = deque([start])
        remaining.discard(start)
        while queue:
            cur = queue.popleft()
            for nb in adj.get(cur, set()):
                if nb not in remaining:
                    continue
                remaining.discard(nb)
                comp.add(nb)
                queue.append(nb)
        if len(comp) > len(best):
            best = comp
    return best


def _write_centerline_tree(path: Path, tree: list[list[float | int]]):
    with path.open("w", encoding="utf-8") as f:
        for row in tree:
            f.write(" ".join(str(v) for v in row) + "\n")


def delete_centerline_terminal_branches(stl_path: Path, branch_ids: list[str]) -> dict:
    parent = stl_path.parent
    centerline_path = _feature_file(parent, RAW_CENTERLINE_NAME)
    nodes = _read_centerline_file(centerline_path)
    if not nodes:
        raise ValueError(f"{RAW_CENTERLINE_NAME} not found or empty.")

    requested = {str(item) for item in branch_ids if str(item)}
    editable = {item["id"]: item for item in _editable_centerline_branches(nodes)}
    invalid = sorted(requested - set(editable))
    if invalid:
        raise ValueError(f"Only endpoint-to-branchpoint branches can be deleted. Invalid: {', '.join(invalid)}")
    if not requested:
        return {"deleted": [], "remaining_branches": list(editable.values()), "removed_nodes": 0}

    remove_nodes = set()
    deleted = []
    for branch_id in sorted(requested):
        item = editable[branch_id]
        path = list(reversed(item["path"]))  # endpoint -> junction
        junction = item["junction_id"]
        remove_nodes.update(n for n in path if n != junction)
        deleted.append(item)

    kept_nodes = {nid: node for nid, node in nodes.items() if nid not in remove_nodes}
    if len(kept_nodes) < 2:
        raise ValueError("Cannot delete branches because the centerline would become too small.")

    adj = _centerline_adjacency(kept_nodes)
    tree = _rebuild_centerline_tree(kept_nodes, adj)
    removed_outputs = remove_generated_outputs(parent, keep_public=False)
    centerline_path = feature_path(parent, RAW_CENTERLINE_NAME, create=True)
    _write_centerline_tree(centerline_path, tree)

    new_nodes = _read_centerline_file(centerline_path)
    return {
        "deleted": deleted,
        "removed_nodes": len(remove_nodes),
        "removed_outputs": removed_outputs,
        "remaining_branches": _editable_centerline_branches(new_nodes),
    }


def _coords_for_path(path: list[int], nodes: dict) -> np.ndarray | None:
    coords = []
    for nid in path:
        if nid not in nodes:
            return None
        n = nodes[nid]
        coords.append([n["x"], n["y"], n["z"]])
    if len(coords) < 2:
        return None
    return np.asarray(coords, dtype=float)


def _coords_from_segment_info(info: dict, nodes: dict) -> np.ndarray | None:
    coords = info.get("smoothed_coords") if isinstance(info, dict) else None
    if coords:
        arr = np.asarray(coords, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) >= 2:
            return arr
    return _coords_for_path(info.get("path", []), nodes)


def _sv_smv_angle_detail_from_segments(
    seg_data: dict | None,
    nodes: dict | None,
    fit_length_mm: float = 10.0,
) -> dict:
    """Measure both confluence branches over one shared physical arc length."""
    segments = (seg_data or {}).get("segments") or {}
    sv = segments.get("sv")
    smv = segments.get("smv")
    if not isinstance(sv, dict) or not isinstance(smv, dict) or not nodes:
        return {}
    sv_path = sv.get("path") or []
    smv_path = smv.get("path") or []
    if len(sv_path) < 2 or len(smv_path) < 2:
        return {}
    common_endpoints = ({sv_path[0], sv_path[-1]} & {smv_path[0], smv_path[-1]})
    if len(common_endpoints) != 1:
        return {}
    confluence_id = common_endpoints.pop()
    sv_coords = _coords_for_path(sv_path, nodes)
    smv_coords = _coords_for_path(smv_path, nodes)
    if sv_coords is None or smv_coords is None:
        return {}
    if sv_path[-1] == confluence_id:
        sv_coords = sv_coords[::-1]
    if smv_path[-1] == confluence_id:
        smv_coords = smv_coords[::-1]
    def arc_length(coords: np.ndarray) -> float:
        return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))

    shared_length = min(float(fit_length_mm), arc_length(sv_coords), arc_length(smv_coords))
    if not np.isfinite(shared_length) or shared_length < 2.0:
        return {}

    def resample_window(coords: np.ndarray, length_mm: float, count: int = 9) -> np.ndarray:
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        targets = np.linspace(0.0, length_mm, count)
        return np.column_stack([
            np.interp(targets, cumulative, coords[:, axis]) for axis in range(3)
        ])

    sv_coords = resample_window(sv_coords, shared_length)
    smv_coords = resample_window(smv_coords, shared_length)

    def direction(coords: np.ndarray):
        distances = np.linspace(0.0, shared_length, len(coords))
        centered_distances = distances - np.mean(distances)
        vector = (centered_distances @ (coords - np.mean(coords, axis=0))) / np.sum(centered_distances ** 2)
        length = float(np.linalg.norm(vector))
        return vector / length if length > 1e-8 else None

    sv_direction = direction(sv_coords)
    smv_direction = direction(smv_coords)
    if sv_direction is None or smv_direction is None:
        return {}
    angle = float(np.degrees(np.arccos(np.clip(np.dot(sv_direction, smv_direction), -1.0, 1.0))))
    return {
        "angle_degrees": round(angle, 2),
        "confluence_point_physical": [round(float(value), 2) for value in sv_coords[0]],
        "confluence_node_id": int(confluence_id),
        "branch1_direction": [round(float(value), 4) for value in sv_direction],
        "branch2_direction": [round(float(value), 4) for value in smv_direction],
        "n_fit_points": len(sv_coords) - 1,
        "fit_length_mm": round(shared_length, 2),
    }


def _path_length_from_coords(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def _path_tortuosity_from_coords(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return 0.0
    arc = _path_length_from_coords(coords)
    chord = float(np.linalg.norm(coords[-1] - coords[0]))
    return float(1.0 - chord / arc) if arc > 1e-9 else 0.0


def _path_mean_curvature_from_coords(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 3:
        return 0.0
    values = []
    for index in range(1, len(coords) - 1):
        v1 = coords[index] - coords[index - 1]
        v2 = coords[index + 1] - coords[index]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-9 or n2 <= 1e-9:
            continue
        angle = float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
        values.append(angle / (0.5 * (n1 + n2)))
    return float(np.mean(values)) if values else 0.0


def _resample_coords_by_arc(coords: np.ndarray, n_points: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return coords.copy()
    seg_lens = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(arc[-1])
    if total <= 1e-9:
        return coords[[0, -1]].copy()
    targets = np.linspace(0.0, total, max(2, int(n_points)))
    sampled = []
    for target in targets:
        idx = int(np.searchsorted(arc, target, side="right") - 1)
        idx = max(0, min(len(coords) - 2, idx))
        local = ((target - arc[idx]) / (arc[idx + 1] - arc[idx])
                 if arc[idx + 1] > arc[idx] else 0.0)
        sampled.append(coords[idx] + local * (coords[idx + 1] - coords[idx]))
    sampled = np.asarray(sampled, dtype=float)
    sampled[0] = coords[0]
    sampled[-1] = coords[-1]
    return sampled


def _dedupe_consecutive_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        return coords
    keep = [0]
    for index in range(1, len(coords)):
        if np.linalg.norm(coords[index] - coords[keep[-1]]) > 1e-9:
            keep.append(index)
    return coords[keep]


def _smooth_manual_segment_coords(coords: np.ndarray) -> np.ndarray | None:
    coords = _dedupe_consecutive_coords(np.asarray(coords, dtype=float))
    if len(coords) < 2:
        return None
    n_points = max(3, len(coords) * 3)
    if len(coords) == 2:
        return _resample_coords_by_arc(coords, n_points)
    try:
        try:
            from geometry_feature_extract.smooth_centerline import _fit_spline_segment
        except ImportError:
            from smooth_centerline import _fit_spline_segment

        smoothed = np.asarray(_fit_spline_segment(
            coords.tolist(),
            smooth_factor=500,
            n_mult=3,
            w_key=1e3,
            w_mid=10.0,
        ), dtype=float)
    except Exception:
        smoothed = _resample_coords_by_arc(coords, n_points)
    if len(smoothed) < 2:
        return None
    smoothed[0] = coords[0]
    smoothed[-1] = coords[-1]
    return smoothed


def _pin_shared_path_nodes(
    smoothed: np.ndarray,
    path: list[int],
    nodes: dict,
    shared_node_ids: set[int],
) -> np.ndarray:
    """Keep branch attachment coordinates on an otherwise whole-path spline."""
    anchors = [index for index, nid in enumerate(path)
               if int(nid) in shared_node_ids and 0 < index < len(path) - 1]
    if not anchors or len(smoothed) < 3:
        return smoothed

    original = _coords_for_path(path, nodes)
    if original is None:
        return smoothed
    original_arc = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))))
    total = float(original_arc[-1])
    if total <= 1e-9:
        return smoothed

    pinned = np.asarray(smoothed, dtype=float).copy()
    used_indices: set[int] = set()
    for anchor_index in anchors:
        target = int(round((original_arc[anchor_index] / total) * (len(pinned) - 1)))
        target = max(1, min(len(pinned) - 2, target))
        while target in used_indices and target < len(pinned) - 2:
            target += 1
        used_indices.add(target)
        pinned[target] = original[anchor_index]
    return pinned


def _apply_manual_segment_smoothing(output: dict, nodes: dict) -> list[str]:
    membership: dict[int, int] = {}
    for info in (output.get("segments") or {}).values():
        if not info:
            continue
        for nid in set(int(value) for value in (info.get("path") or [])):
            membership[nid] = membership.get(nid, 0) + 1
    shared_node_ids = {nid for nid, count in membership.items() if count > 1}

    smoothed_vessels = []
    for vessel, info in (output.get("segments") or {}).items():
        if not info or not info.get("path"):
            continue
        coords = _coords_for_path(info["path"], nodes)
        if coords is None:
            continue
        smoothed = _smooth_manual_segment_coords(coords)
        if smoothed is None:
            continue
        smoothed = _pin_shared_path_nodes(
            smoothed, info["path"], nodes, shared_node_ids)
        info["topology_n_points"] = int(len(info.get("path", [])))
        info["smoothed_coords"] = [
            [float(point[0]), float(point[1]), float(point[2])]
            for point in smoothed
        ]
        info["n_points"] = int(len(smoothed))
        info["length_mm"] = _path_length_from_coords(smoothed)
        info["tortuosity"] = _path_tortuosity_from_coords(smoothed)
        info["mean_curvature"] = _path_mean_curvature_from_coords(smoothed)
        info["smoothing"] = {
            "applied": True,
            "source": "manual_segment_assignment",
            "method": "whole_anatomical_segment_spline",
            "input_n_points": int(len(coords)),
            "output_n_points": int(len(smoothed)),
        }
        smoothed_vessels.append(vessel)
    output["manual_segment_smoothing"] = {
        "applied": bool(smoothed_vessels),
        "vessels": smoothed_vessels,
    }
    return smoothed_vessels


def _rebuild_smoothed_assignment_tree(output: dict, nodes: dict) -> list[list[float | int]]:
    """Build one connected centerline tree from the edited vessel paths.

    Each vessel is smoothed as a complete path first.  Shared endpoints are
    merged by coordinate, so a vessel split by a junction (for example MPV
    around an LPV attachment) is smoothed continuously and remains connected
    to the branch in the rewritten tree.
    """
    coord_ids: dict[tuple[float, float, float], int] = {}
    generated_nodes: dict[int, dict] = {}
    chains: dict[str, list[int]] = {}

    def node_for(point) -> int:
        key = tuple(round(float(value), 8) for value in point)
        if key not in coord_ids:
            nid = len(coord_ids)
            coord_ids[key] = nid
            generated_nodes[nid] = {
                "id": nid,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "parent": -1,
                "left": -1,
                "right": -1,
            }
        return coord_ids[key]

    for vessel, info in (output.get("segments") or {}).items():
        if not info or not info.get("path"):
            continue
        coords = np.asarray(info.get("smoothed_coords") or [], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 2:
            coords = _coords_for_path(info["path"], nodes)
        if coords is None or len(coords) < 2:
            continue
        chain = [node_for(point) for point in coords]
        deduped = [chain[0]]
        for nid in chain[1:]:
            if nid != deduped[-1]:
                deduped.append(nid)
        chains[vessel] = deduped

    adjacency = _centerline_adjacency(generated_nodes)
    for chain in chains.values():
        for a, b in zip(chain, chain[1:]):
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    tree = _rebuild_centerline_tree(generated_nodes, adjacency)
    id_by_coord = {
        tuple(round(float(value), 8) for value in row[1:4]): int(row[0])
        for row in tree
    }
    for vessel, chain in chains.items():
        info = output["segments"][vessel]
        coords = np.asarray(info.get("smoothed_coords"), dtype=float)
        info["path"] = [
            id_by_coord[tuple(round(float(value), 8) for value in point)]
            for point in coords
        ]
        info["n_points"] = len(info["path"])

    final_nodes = _read_centerline_file_from_rows(tree)
    final_adj = _centerline_adjacency(final_nodes)
    output["branch_points"] = [
        {
            "id": int(nid),
            "coord": [node["x"], node["y"], node["z"]],
        }
        for nid, nbs in final_adj.items()
        if len(nbs) >= 3
        for node in [final_nodes[nid]]
    ]
    output["endpoints"] = [
        {
            "id": int(nid),
            "coord": [node["x"], node["y"], node["z"]],
        }
        for nid, nbs in final_adj.items()
        if len(nbs) == 1
        for node in [final_nodes[nid]]
    ]
    output["centerline_source"] = SMOOTH_CENTERLINE_NAME
    output["manual_segment_smoothing"]["rewrote_centerline"] = True
    output["manual_segment_smoothing"]["centerline_file"] = SMOOTH_CENTERLINE_NAME
    return tree


def _read_centerline_file_from_rows(tree: list[list[float | int]]) -> dict:
    return {
        int(row[0]): {
            "id": int(row[0]),
            "x": float(row[1]),
            "y": float(row[2]),
            "z": float(row[3]),
            "parent": int(row[4]),
            "left": int(row[5]),
            "right": int(row[6]),
        }
        for row in tree
    }


def _point_and_tangent_at_fraction(coords: np.ndarray, frac: float):
    frac = min(1.0, max(0.0, float(frac)))
    if len(coords) < 2:
        return None, None
    seg_lens = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total = float(arc[-1])
    if total <= 1e-9:
        return coords[0], np.array([0.0, 0.0, 1.0])
    target = total * frac
    idx = int(np.searchsorted(arc, target) - 1)
    idx = max(0, min(len(coords) - 2, idx))
    a0, a1 = arc[idx], arc[idx + 1]
    local = (target - a0) / (a1 - a0) if a1 > a0 else 0.0
    point = coords[idx] + local * (coords[idx + 1] - coords[idx])
    lo = max(0, idx - 1)
    hi = min(len(coords) - 1, idx + 2)
    tangent = coords[hi] - coords[lo]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        tangent = np.array([0.0, 0.0, 1.0])
    else:
        tangent = tangent / norm
    return point, tangent


def _basis_from_normal(normal: np.ndarray):
    n = np.asarray(normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-15)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u = u / (np.linalg.norm(u) + 1e-15)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-15)
    return u, v


def _circle_arrays(point: np.ndarray, normal: np.ndarray, radius: float, n_pts: int = 40):
    if radius is None or not math.isfinite(radius) or radius <= 0:
        return None
    u, v = _basis_from_normal(normal)
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts)
    pts = np.asarray([point + radius * (math.cos(t) * u + math.sin(t) * v) for t in theta])
    return pts


def _load_surface_section_mesh(stl_path: Path):
    try:
        import trimesh

        mesh = trimesh.load(str(stl_path), force="mesh")
        if hasattr(mesh, "section") and hasattr(mesh, "vertices"):
            return mesh
    except Exception:
        return None
    return None


def _surface_section_contours(mesh, point: np.ndarray, normal: np.ndarray) -> list[np.ndarray]:
    if mesh is None:
        return []
    try:
        section = mesh.section(
            plane_origin=np.asarray(point, dtype=float),
            plane_normal=np.asarray(normal, dtype=float),
        )
        if section is None:
            return []
        return [
            np.asarray(contour, dtype=float)
            for contour in section.discrete
            if contour is not None and len(contour) >= 2
        ]
    except Exception:
        return []


def _surface_section_arrays(mesh, point: np.ndarray, normal: np.ndarray):
    """Return the true mesh-plane contour nearest this centerline point."""
    contours = _surface_section_contours(mesh, point, normal)
    if not contours:
        return None
    return min(
        contours,
        key=lambda contour: float(np.min(np.linalg.norm(contour - point, axis=1))),
    )


_VORONOI_SECTION_CLIPPER = None
_VORONOI_SECTION_CLIPPER_UNAVAILABLE = False
_VORONOI_ASSIGNMENT_METHODS = {
    "centerline_voronoi",
    "centerline_network_voronoi",
}


def _load_voronoi_section_clipper():
    """Load the extractor's ownership clip lazily with scientific dependencies."""
    global _VORONOI_SECTION_CLIPPER, _VORONOI_SECTION_CLIPPER_UNAVAILABLE
    if _VORONOI_SECTION_CLIPPER is not None:
        return _VORONOI_SECTION_CLIPPER
    if _VORONOI_SECTION_CLIPPER_UNAVAILABLE:
        return None
    try:
        app_root = str(APP_ROOT)
        if app_root not in sys.path:
            sys.path.insert(0, app_root)
        from extract_profiles import _clip_section_to_centerline_voronoi
        from shapely.geometry import Point, Polygon

        _VORONOI_SECTION_CLIPPER = (
            _clip_section_to_centerline_voronoi, Point, Polygon)
    except Exception:
        _VORONOI_SECTION_CLIPPER_UNAVAILABLE = True
        return None
    return _VORONOI_SECTION_CLIPPER


def _voronoi_surface_section_arrays(
    contour: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    profile: dict | None,
    index: int | None,
    centerline_coords: np.ndarray | None,
    centerline_arc_length: np.ndarray | None,
):
    """Apply the extractor's centerline ownership clip to a Web STL contour."""
    if (
        not isinstance(profile, dict)
        or profile.get(
            "section_assignment_method") not in _VORONOI_ASSIGNMENT_METHODS
        or centerline_coords is None
        or index is None
    ):
        return None
    helpers = _load_voronoi_section_clipper()
    if helpers is None:
        return None
    clip_section, point_type, polygon_type = helpers
    try:
        u, v = _basis_from_normal(normal)
        relative = np.asarray(contour, dtype=float) - np.asarray(point, dtype=float)
        planar = np.column_stack((relative @ u, relative @ v))
        polygon = polygon_type(planar)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        competing_centerlines = []
        for competitor in profile.get("network_voronoi_competitors", []):
            values = (
                competitor.get("centerline_coords")
                if isinstance(competitor, dict) else competitor)
            coords = np.asarray(values, dtype=float)
            if (coords.ndim == 2 and coords.shape[1] == 3
                    and np.all(np.isfinite(coords))):
                competing_centerlines.append({
                    "centerline_coords": coords,
                    "radius_mm": _safe_float(competitor.get("radius_mm")) or 0.0,
                })
        radii = profile.get("inscribed_radius") or []
        site_radius = (
            _safe_float(radii[index])
            if index < len(radii) else 0.0)
        clipped = clip_section(
            polygon,
            point_type(0.0, 0.0),
            point,
            normal,
            centerline_coords=centerline_coords,
            centerline_index=index,
            local_exclusion_mm=float(
                profile.get("centerline_voronoi_exclusion_mm", 5.0)),
            centerline_arc_length=centerline_arc_length,
            competing_centerlines=competing_centerlines,
            site_radius_mm=site_radius or 0.0,
        )
        if clipped is None or clipped.is_empty:
            return None
        ring = np.asarray(clipped.exterior.coords, dtype=float)
        if len(ring) < 3:
            return None
        return (
            np.asarray(point, dtype=float)
            + ring[:, :1] * u
            + ring[:, 1:2] * v
        )
    except Exception:
        return None


def _pointwise_surface_section_arrays(
    mesh,
    point: np.ndarray,
    normal: np.ndarray,
    expected_diameter: float | None,
    *,
    profile: dict | None = None,
    index: int | None = None,
    centerline_coords: np.ndarray | None = None,
    centerline_arc_length: np.ndarray | None = None,
):
    """Draw every valid pointwise section with no independent Web rejection."""
    if expected_diameter is None or not math.isfinite(expected_diameter) or expected_diameter <= 0:
        return None
    metrics = _surface_section_metrics(
        mesh, point, normal, nearby_radius=expected_diameter / 2.0)
    if metrics:
        clipped = _voronoi_surface_section_arrays(
            metrics["contour"], point, normal, profile, index,
            centerline_coords, centerline_arc_length)
        if clipped is not None:
            return clipped
        if not isinstance(profile, dict) or profile.get(
                "section_assignment_method") not in _VORONOI_ASSIGNMENT_METHODS:
            return metrics["contour"]
    return _circle_arrays(point, normal, expected_diameter / 2.0, n_pts=36)


def _polygon_area_2d(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    ordered = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(ordered) <= 2:
        return np.asarray(ordered, dtype=float)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _surface_section_metrics(mesh, point: np.ndarray, normal: np.ndarray, nearby_radius: float | None = None):
    contours = _surface_section_contours(mesh, point, normal)
    if not contours:
        return None
    contour = min(contours, key=lambda c: float(np.min(np.linalg.norm(c - point, axis=1))))
    u, v = _basis_from_normal(normal)
    relative = contour - point
    planar = np.column_stack((relative @ u, relative @ v))
    area = _polygon_area_2d(planar)
    perimeter = float(np.sum(np.linalg.norm(np.diff(contour, axis=0), axis=1)))
    if area <= 1e-9 or perimeter <= 1e-9:
        return None
    hull_area = _polygon_area_2d(_convex_hull_2d(planar))
    eq_diameter = float(2.0 * math.sqrt(area / math.pi))
    radius = max(float(nearby_radius or 0.0), 1.5 * eq_diameter)
    near_components = sum(
        1 for candidate in contours
        if float(np.min(np.linalg.norm(candidate - point, axis=1))) <= radius
    )
    return {
        "contour": contour,
        "area": area,
        "perimeter": perimeter,
        "eq_diameter": eq_diameter,
        "circularity": float(4.0 * math.pi * area / (perimeter * perimeter)),
        "solidity": float(area / hull_area) if hull_area > 1e-9 else None,
        "n_near_components": int(near_components),
    }


def _finite_array(values):
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float)
    return arr


def _profile_positions(profile: dict) -> np.ndarray:
    n = len(profile.get("position") or profile.get("area") or [])
    if n <= 1:
        return np.asarray([], dtype=float)
    pos = profile.get("position")
    if pos and len(pos) == n:
        return np.asarray(pos, dtype=float)
    return np.linspace(0.0, 1.0, n)


def _profile_resample_count(positions: np.ndarray, profile: dict | None = None) -> int:
    """Recover the original uniform profile count after endpoint filtering."""
    if isinstance(profile, dict):
        explicit = _safe_float(profile.get("profile_sample_count"))
        if explicit is not None and explicit >= 2:
            return int(explicit)
    positions = np.asarray(positions, dtype=float)
    diffs = np.diff(positions)
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
    if diffs.size:
        inferred = int(round(1.0 / float(np.min(diffs)))) + 1
        if 2 <= inferred <= 5000:
            return inferred
    return max(2, len(positions))


def _is_original_profile_sample(positions: np.ndarray, index: int,
                                section_stride: int,
                                profile: dict | None = None) -> bool:
    """Keep display sampling aligned to pre-filter resampled profile indices."""
    count = _profile_resample_count(positions, profile)
    sample_index = profile.get("profile_sample_index") if isinstance(profile, dict) else None
    if isinstance(sample_index, list) and index < len(sample_index):
        original_index = int(sample_index[index])
    else:
        original_index = int(round(float(positions[index]) * (count - 1)))
    return original_index % section_stride == 0


def _section_normal_at(profile: dict, index: int, fallback: np.ndarray) -> np.ndarray:
    """Use the persisted extraction normal, with the tangent as an old-data fallback."""
    values = [_valid_numeric_at(profile, key, index) for key in (
        "section_normal_x", "section_normal_y", "section_normal_z")]
    if all(value is not None for value in values):
        normal = np.asarray(values, dtype=float)
        magnitude = float(np.linalg.norm(normal))
        if magnitude > 1e-9:
            return normal / magnitude
    return np.asarray(fallback, dtype=float)


def _section_is_valid(profile: dict, index: int) -> bool:
    """Use only the validity encoded by the pointwise profile."""
    values = profile.get("section_valid")
    if values is not None and index < len(values):
        value = _safe_float(values[index])
        if value is None or value <= 0:
            return False
    if values is not None:
        return True
    diameter = _valid_numeric_at(profile, "eq_diameter", index)
    return diameter is not None and diameter > 0


def _profile_centerline_coords(profile: dict) -> np.ndarray | None:
    """Return the exact centreline locations used during profile extraction."""
    values = [profile.get(key) for key in (
        "centerline_x", "centerline_y", "centerline_z")]
    if not all(isinstance(value, list) and len(value) >= 2 for value in values):
        return None
    coords = np.column_stack([np.asarray(value, dtype=float) for value in values])
    return coords if np.all(np.isfinite(coords)) else None


def _profile_point_and_tangent(coords: np.ndarray, positions: np.ndarray, index: int):
    """Use stored profile coordinates directly; fall back to fraction lookup."""
    if len(coords) == len(positions) and 0 <= index < len(coords):
        lo = max(0, index - 2)
        hi = min(len(coords) - 1, index + 2)
        tangent = coords[hi] - coords[lo]
        norm = float(np.linalg.norm(tangent))
        tangent = tangent / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        return coords[index], tangent
    return _point_and_tangent_at_fraction(coords, float(positions[index]))


def _build_segments(seg_data: dict | None, nodes: dict | None) -> dict:
    out = {}
    if not seg_data or not nodes:
        return out
    for name, info in (seg_data.get("segments") or {}).items():
        if not info:
            continue
        coords = _coords_from_segment_info(info, nodes)
        if coords is None:
            continue
        label = SEGMENT_LABELS.get(name, name.upper())
        out[name] = {
            "label": label,
            "color": SEGMENT_COLORS.get(name, "#888888"),
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "z": coords[:, 2].tolist(),
            "midpoint": coords[len(coords) // 2].tolist(),
            "length_mm": info.get("length_mm"),
            "tortuosity": info.get("tortuosity"),
            "mean_curvature": info.get("mean_curvature"),
            "n_points": info.get("n_points", len(coords)),
            "path": info.get("path", []),
        }
    return out


def _valid_numeric_at(profile: dict, key: str, idx: int):
    values = profile.get(key)
    if not values or idx >= len(values):
        return None
    value = _safe_float(values[idx])
    return value


def _build_pointwise_layers(
    seg_data: dict | None,
    nodes: dict | None,
    pointwise: dict | None,
    section_stride: int,
    surface_mesh=None,
):
    features = {}
    sampled_sections = {}
    max_sections = {}
    mean_sections = {}
    surface_sections = {}
    surface_max_sections = {}
    surface_mean_sections = {}
    if not seg_data or not nodes or not pointwise:
        return {
            "feature_points": features,
            "sampled_sections": sampled_sections,
            "max_sections": max_sections,
            "mean_sections": mean_sections,
            "surface_sections": surface_sections,
            "surface_max_sections": surface_max_sections,
            "surface_mean_sections": surface_mean_sections,
        }
    section_stride = max(1, int(section_stride or 10))

    for seg_name, seg_info in (seg_data.get("segments") or {}).items():
        profile = pointwise.get(seg_name)
        if not seg_info or not profile:
            continue
        coords = _profile_centerline_coords(profile)
        if coords is None:
            coords = _coords_for_path(
                profile.get("analysis_path") or seg_info.get("path", []), nodes)
        if coords is None:
            continue
        positions = _profile_positions(profile)
        if len(positions) == 0:
            continue
        centerline_arc_length = np.asarray(
            profile.get("arc_length_mm", []), dtype=float)
        if (
            len(centerline_arc_length) != len(coords)
            or not np.all(np.isfinite(centerline_arc_length))
        ):
            centerline_arc_length = None

        fx, fy, fz, feature_positions = [], [], [], []
        curvature_values, sizes, hover = [], [], []
        ring_x, ring_y, ring_z, ring_positions = [], [], [], []
        surface_x, surface_y, surface_z, surface_positions = [], [], [], []
        area = _finite_array(profile.get("area"))
        diameter = _finite_array(profile.get("eq_diameter"))
        curvature = _finite_array(profile.get("curvature"))
        circularity = _finite_array(profile.get("circularity"))
        inscribed = _finite_array(profile.get("inscribed_radius"))

        for i, frac in enumerate(positions):
            global_fraction = float(min(1.0, max(0.0, frac)))
            point, tangent = _profile_point_and_tangent(coords, positions, i)
            if point is None or not _section_is_valid(profile, i):
                continue
            curv = _valid_numeric_at(profile, "curvature", i)
            dia = _valid_numeric_at(profile, "eq_diameter", i)
            ar = _valid_numeric_at(profile, "area", i)
            circ = _valid_numeric_at(profile, "circularity", i)
            ins = _valid_numeric_at(profile, "inscribed_radius", i)
            if curv is not None or dia is not None or ar is not None:
                fx.append(float(point[0]))
                fy.append(float(point[1]))
                fz.append(float(point[2]))
                feature_positions.append(global_fraction)
                curvature_values.append(curv if curv is not None else 0.0)
                sizes.append(max(4.0, min(15.0, 3.0 + (dia or 0.0) * 0.45)))
                hover.append(
                    f"{SEGMENT_LABELS.get(seg_name, seg_name.upper())}<br>"
                    f"point: {i}<br>"
                    f"curvature: {_format_metric(curv, 5)} 1/mm<br>"
                    f"diameter: {_format_metric(dia, 3)} mm<br>"
                    f"area: {_format_metric(ar, 3)} mm^2<br>"
                    f"circularity: {_format_metric(circ, 3)}<br>"
                    f"inscribed radius: {_format_metric(ins, 3)} mm"
                )
            # Endpoint masking removes entries from the stored profile.  Sampling
            # by the compacted list index would immediately redraw its first
            # retained plane and visually erase the exclusion gap.
            if _is_original_profile_sample(positions, i, section_stride, profile):
                dia = _valid_numeric_at(profile, "eq_diameter", i)
                normal = _section_normal_at(profile, i, tangent)
                if dia is not None and dia > 0:
                    circle = _circle_arrays(point, normal, dia / 2.0, n_pts=36)
                    if circle is not None:
                        ring_x.extend(circle[:, 0].tolist() + [None])
                        ring_y.extend(circle[:, 1].tolist() + [None])
                        ring_z.extend(circle[:, 2].tolist() + [None])
                        ring_positions.extend(
                            [global_fraction] * len(circle) + [None])
                contour = _pointwise_surface_section_arrays(
                    surface_mesh, point, normal, dia,
                    profile=profile,
                    index=i,
                    centerline_coords=coords,
                    centerline_arc_length=centerline_arc_length)
                if contour is not None:
                    surface_x.extend(contour[:, 0].tolist() + [None])
                    surface_y.extend(contour[:, 1].tolist() + [None])
                    surface_z.extend(contour[:, 2].tolist() + [None])
                    surface_positions.extend(
                        [global_fraction] * len(contour) + [None])

        features[seg_name] = {
            "label": SEGMENT_LABELS.get(seg_name, seg_name.upper()),
            "color": SEGMENT_COLORS.get(seg_name, "#888888"),
            "x": fx,
            "y": fy,
            "z": fz,
            "position": feature_positions,
            "curvature": curvature_values,
            "size": sizes,
            "hover": hover,
        }
        if ring_x:
            sampled_sections[seg_name] = {
                "label": SEGMENT_LABELS.get(seg_name, seg_name.upper()),
                "color": SEGMENT_COLORS.get(seg_name, "#888888"),
                "x": ring_x,
                "y": ring_y,
                "z": ring_z,
                "position": ring_positions,
            }
        if surface_x:
            surface_sections[seg_name] = {
                "label": SEGMENT_LABELS.get(seg_name, seg_name.upper()),
                "color": SEGMENT_COLORS.get(seg_name, "#888888"),
                "x": surface_x,
                "y": surface_y,
                "z": surface_z,
                "position": surface_positions,
            }

        max_idx = _best_index(area, mode="max")
        mean_idx = _best_index(area, mode="mean")
        max_ring = _section_at_index(coords, profile, max_idx)
        mean_ring = _section_at_index(coords, profile, mean_idx)
        if max_ring:
            max_sections[seg_name] = max_ring
        if mean_ring:
            mean_sections[seg_name] = mean_ring
        max_surface = _surface_section_at_index(coords, profile, max_idx, surface_mesh)
        mean_surface = _surface_section_at_index(coords, profile, mean_idx, surface_mesh)
        if max_surface:
            surface_max_sections[seg_name] = max_surface
        if mean_surface:
            surface_mean_sections[seg_name] = mean_surface

    return {
        "feature_points": features,
        "sampled_sections": sampled_sections,
        "max_sections": max_sections,
        "mean_sections": mean_sections,
        "surface_sections": surface_sections,
        "surface_max_sections": surface_max_sections,
        "surface_mean_sections": surface_mean_sections,
    }


def _best_index(arr: np.ndarray, mode: str):
    if arr.size == 0:
        return None
    valid = np.isfinite(arr) & (arr > 0)
    if not np.any(valid):
        return None
    if mode == "max":
        masked = np.where(valid, arr, -np.inf)
        return int(np.argmax(masked))
    mean_val = float(np.nanmean(arr[valid]))
    dist = np.where(valid, np.abs(arr - mean_val), np.inf)
    return int(np.argmin(dist))


def _section_at_index(coords: np.ndarray, profile: dict, idx: int | None):
    if idx is None:
        return None
    positions = _profile_positions(profile)
    if idx < 0 or idx >= len(positions):
        return None
    if not _section_is_valid(profile, idx):
        return None
    dia = _valid_numeric_at(profile, "eq_diameter", idx)
    area = _valid_numeric_at(profile, "area", idx)
    if dia is None or dia <= 0:
        return None
    point, tangent = _profile_point_and_tangent(coords, positions, idx)
    if point is None:
        return None
    normal = _section_normal_at(profile, idx, tangent)
    circle = _circle_arrays(point, normal, dia / 2.0, n_pts=48)
    if circle is None:
        return None
    return {
        "x": circle[:, 0].tolist(),
        "y": circle[:, 1].tolist(),
        "z": circle[:, 2].tolist(),
        "index": int(idx),
        "position": float(min(1.0, max(0.0, positions[idx]))),
        "diameter": dia,
        "area": area,
    }


def _surface_section_at_index(coords: np.ndarray, profile: dict, idx: int | None, surface_mesh):
    if idx is None or surface_mesh is None:
        return None
    positions = _profile_positions(profile)
    if idx < 0 or idx >= len(positions):
        return None
    if not _section_is_valid(profile, idx):
        return None
    point, tangent = _profile_point_and_tangent(coords, positions, idx)
    if point is None:
        return None
    diameter = _valid_numeric_at(profile, "eq_diameter", idx)
    normal = _section_normal_at(profile, idx, tangent)
    centerline_arc_length = np.asarray(
        profile.get("arc_length_mm", []), dtype=float)
    if (
        len(centerline_arc_length) != len(coords)
        or not np.all(np.isfinite(centerline_arc_length))
    ):
        centerline_arc_length = None
    contour = _pointwise_surface_section_arrays(
        surface_mesh, point, normal, diameter,
        profile=profile,
        index=idx,
        centerline_coords=coords,
        centerline_arc_length=centerline_arc_length)
    if contour is None:
        return None
    return {
        "x": contour[:, 0].tolist(),
        "y": contour[:, 1].tolist(),
        "z": contour[:, 2].tolist(),
        "index": int(idx),
        "position": float(min(1.0, max(0.0, positions[idx]))),
        "diameter": diameter,
        "area": _valid_numeric_at(profile, "area", idx),
    }


def _format_metric(value, digits=3) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def _load_mesh(stl_path: Path, max_faces: int = 80000) -> dict | None:
    try:
        return _load_mesh_with_trimesh(stl_path, max_faces=max_faces)
    except Exception:
        return _load_mesh_fallback(stl_path, max_faces=max_faces)


def _load_mesh_with_trimesh(stl_path: Path, max_faces: int = 80000) -> dict | None:
    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh")
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return None
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return _compact_sampled_mesh(vertices, faces, max_faces=max_faces, source="trimesh")


def _load_mesh_fallback(stl_path: Path, max_faces: int = 80000) -> dict | None:
    try:
        with stl_path.open("rb") as f:
            header = f.read(80)
            n_raw = f.read(4)
            if len(n_raw) == 4:
                n_faces = struct.unpack("<I", n_raw)[0]
                expected = 84 + n_faces * 50
                actual = stl_path.stat().st_size
                if n_faces > 0 and expected == actual:
                    return _read_binary_stl_sampled(f, n_faces, max_faces=max_faces)
        return _read_ascii_stl_sampled(stl_path, max_faces=max_faces)
    except Exception:
        return None


def _read_binary_stl_sampled(handle, n_faces: int, max_faces: int):
    stride = max(1, int(math.ceil(n_faces / max_faces)))
    vertices = []
    faces = []
    face_idx = 0
    for i in range(n_faces):
        chunk = handle.read(50)
        if len(chunk) < 50:
            break
        if i % stride != 0:
            continue
        vals = struct.unpack("<12fH", chunk)
        tri = vals[3:12]
        base = len(vertices)
        vertices.extend([
            [tri[0], tri[1], tri[2]],
            [tri[3], tri[4], tri[5]],
            [tri[6], tri[7], tri[8]],
        ])
        faces.append([base, base + 1, base + 2])
        face_idx += 1
    return {
        "vertices": vertices,
        "faces": faces,
        "n_faces": int(n_faces),
        "n_faces_rendered": int(face_idx),
        "source": "stdlib-binary-stl",
    }


def _read_ascii_stl_sampled(stl_path: Path, max_faces: int):
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


def _compact_sampled_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int, source: str):
    n_faces = int(len(faces))
    if n_faces > max_faces:
        stride = int(math.ceil(n_faces / max_faces))
        faces = faces[::stride]
    used = np.unique(faces.reshape(-1))
    remap = {int(old): i for i, old in enumerate(used)}
    compact_vertices = vertices[used]
    compact_faces = np.asarray([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces], dtype=np.int64)
    return {
        "vertices": np.round(compact_vertices, 5).tolist(),
        "faces": compact_faces.tolist(),
        "n_faces": n_faces,
        "n_faces_rendered": int(len(compact_faces)),
        "source": source,
    }


def _load_feature_blocks(parent: Path):
    unified = _read_json_file(_feature_file(parent, UNIFIED_FEATURES_NAME))
    if unified:
        return {
            "source": UNIFIED_FEATURES_NAME,
            "meta": unified.get("_meta", {}),
            "vessel_presence": unified.get("vessel_presence", {}),
            "statistical": unified.get("statistical", {}),
            "system": unified.get("system", {}),
            "global": unified.get("global", {}),
            "sv_smv_angle": unified.get("sv_smv_angle", {}),
            "segments_meta": unified.get("segments_meta", {}),
            "pointwise_meta": unified.get("pointwise_meta", {}),
        }
    return {
        "source": None,
        "meta": {},
        "vessel_presence": {},
        "statistical": {},
        "system": {},
        "global": {},
        "sv_smv_angle": {},
        "segments_meta": {},
        "pointwise_meta": {},
    }


def build_visualization_data(
    stl_path: Path,
    section_stride: int = 10,
    max_faces: int = 80000,
    include_surface_sections: bool = False,
) -> dict:
    parent = stl_path.parent
    raw_nodes = _read_centerline_file(_feature_file(parent, RAW_CENTERLINE_NAME))
    smooth_nodes = _read_centerline_file(_feature_file(parent, SMOOTH_CENTERLINE_NAME))
    nodes = smooth_nodes or raw_nodes
    seg_data = _read_json_file(_feature_file(parent, SEGMENT_ASSIGNMENTS_NAME))
    pointwise = _read_json_file(_feature_file(parent, POINTWISE_TEMP_NAME))
    if not isinstance(pointwise, dict):
        pointwise = {}
    pointwise_analysis_ranges = _pointwise_analysis_ranges(pointwise)

    surface_mesh = _load_surface_section_mesh(stl_path) if include_surface_sections else None
    pointwise_layers = _build_pointwise_layers(
        seg_data, nodes, pointwise, section_stride, surface_mesh=surface_mesh)
    feature_blocks = _load_feature_blocks(parent)
    # Recompute for the viewer so historical feature files cannot retain a
    # point-count based direction estimate after the physical-length fix.
    angle_detail = _sv_smv_angle_detail_from_segments(
        seg_data, nodes, fit_length_mm=DEFAULT_PARAMS["angle_fit_length_mm"])
    feature_blocks["sv_smv_angle"] = angle_detail
    if angle_detail.get("angle_degrees") is not None:
        feature_blocks.setdefault("global", {})["sv_smv_angle"] = angle_detail["angle_degrees"]
    branch_points = []
    if seg_data:
        for bp in seg_data.get("branch_points", []):
            if isinstance(bp, dict) and "coord" in bp:
                branch_points.append(bp)

    return _sanitize_json({
        "patient": _patient_record(stl_path),
        "segment_profile": SEGMENT_ASSIGNMENTS_NAME,
        "mesh": _load_mesh(stl_path, max_faces=max_faces),
        "centerlines": {
            "raw": _line_arrays_from_nodes(raw_nodes),
            "smooth": _line_arrays_from_nodes(smooth_nodes),
        },
        "centerline_edit": {
            "branches": _editable_centerline_branches(raw_nodes),
        },
        "manual_segmentation": {
            "atomic_segments": _annotated_atomic_segments(nodes, seg_data, parent),
            "vessels": [
                {
                    "id": name,
                    "label": SEGMENT_LABELS[name],
                    "color": SEGMENT_COLORS[name],
                }
                for name in SEGMENT_LABELS
            ],
            "saved": _feature_file(parent, SEGMENT_ASSIGNMENTS_NAME).exists(),
        },
        "analysis_regions": {
            "ranges": pointwise_analysis_ranges,
            "available_vessels": sorted(pointwise_analysis_ranges),
            "source": POINTWISE_TEMP_NAME if pointwise_analysis_ranges else None,
        },
        "segments": _build_segments(seg_data, nodes),
        "branch_points": branch_points,
        "pointwise": pointwise_layers,
        "features": feature_blocks,
        "files": _available_outputs(parent),
        "step_files": _step_file_status(parent),
    })


def _available_outputs(parent: Path) -> list[dict]:
    files = []
    for name in OUTPUT_FILES:
        p = feature_path(parent, name)
        if p.exists():
            files.append({
                "name": name,
                "size": p.stat().st_size,
                "modified": p.stat().st_mtime,
            })
    return files


def _step_file_status(parent: Path) -> dict:
    status = {}
    for step, names in STEP_OUTPUTS.items():
        files = []
        for name in names:
            p = feature_path(parent, name)
            files.append({
                "name": name,
                "exists": p.exists(),
                "size": p.stat().st_size if p.exists() else 0,
                "modified": p.stat().st_mtime if p.exists() else None,
            })
        status[step] = {
            "ready": all(item["exists"] for item in files),
            "files": files,
        }
    return status


def _reuse_pipeline_step(step: str, stl_path: Path):
    required = STEP_OUTPUTS.get(step) or []
    search_dir = features_dir(stl_path.parent, create=False)
    missing = [name for name in required if not (search_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot import saved result for {STEP_LABELS.get(step, step)}; "
            f"searched folder: {search_dir}; "
            f"missing: {', '.join(str(search_dir / name) for name in missing)}"
        )
    print(f"Reused saved result for {STEP_LABELS.get(step, step)}: {', '.join(required)}")


def _create_session_single(fields) -> dict:
    session_id = _new_session_id()
    file_item = fields["stl_file"] if "stl_file" in fields else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise ValueError("Missing STL file.")
    original_name = Path(file_item.filename).name or "vessel.stl"
    if not original_name.lower().endswith(".stl"):
        original_name += ".stl"
    output_dir = None
    if "output_dir" in fields:
        raw_output = str(fields.getvalue("output_dir") or "").strip()
        if raw_output:
            output_dir = Path(raw_output)
    if output_dir:
        patient_dir = output_dir
        patient_dir.mkdir(parents=True, exist_ok=True)
    else:
        patient_dir = RUNS_ROOT / session_id / Path(original_name).stem
        patient_dir.mkdir(parents=True, exist_ok=True)
    stl_path = patient_dir / original_name
    with stl_path.open("wb") as out:
        shutil.copyfileobj(file_item.file, out)
    session = {
            "id": session_id,
            "mode": "single",
            "created": _now(),
            "root": str(patient_dir),
            "patients": [_patient_record(stl_path)],
            "params": dict(DEFAULT_PARAMS),
            "runtime": _runtime_info(),
        }
    with STATE_LOCK:
        SESSIONS[session_id] = session
    return session


def _create_session_batch(payload: dict) -> dict:
    root = Path(str(payload.get("root_folder") or "").strip())
    stl_name = str(payload.get("stl_name") or "vessel.stl").strip() or "vessel.stl"
    if not root.exists():
        raise ValueError(f"Folder does not exist: {root}")
    patients = _discover_batch(root, stl_name)
    if not patients:
        raise ValueError(f"No {stl_name} files found under {root}")
    session_id = _new_session_id()
    session = {
        "id": session_id,
        "mode": "batch",
        "created": _now(),
        "root": str(root),
        "stl_name": stl_name,
        "patients": patients,
        "params": dict(DEFAULT_PARAMS),
        "runtime": _runtime_info(),
    }
    with STATE_LOCK:
        SESSIONS[session_id] = session
    return session


def _new_job(session_id: str, steps: list[str], patients: list[dict], step_modes: dict | None = None) -> dict:
    job_id = uuid.uuid4().hex[:12]
    total = max(1, len(steps) * len(patients))
    job = {
        "id": job_id,
        "session_id": session_id,
        "status": "running",
        "created": _now(),
        "updated": _now(),
        "steps": steps,
        "step_modes": step_modes or {},
        "total": total,
        "completed": 0,
        "current": "",
        "logs": [],
        "errors": [],
        "results": {},
    }
    with STATE_LOCK:
        JOBS[job_id] = job
    return job


def _append_job_log(job: dict, message: str):
    with STATE_LOCK:
        job["logs"].append(message)
        job["logs"] = job["logs"][-500:]
        job["updated"] = _now()


def _set_job_progress(job: dict, current: str | None = None, completed_delta: int = 0):
    with STATE_LOCK:
        if current is not None:
            job["current"] = current
        job["completed"] += completed_delta
        job["updated"] = _now()


def _run_job(job_id: str, params: dict, post_tips_mode: str, export_png: bool):
    with STATE_LOCK:
        job = JOBS[job_id]
        session = SESSIONS[job["session_id"]]
        patients = list(job.get("_patients_runtime", []))
        steps = list(job["steps"])
        step_modes = dict(job.get("step_modes") or {})
        job.pop("_patients_runtime", None)
    try:
        for patient in patients:
            stl_path = Path(patient["stl_path"])
            for step in steps:
                label = STEP_LABELS.get(step, step)
                _set_job_progress(job, current=f"{patient['id']} - {label}")
                buffer = io.StringIO()
                started = time.time()
                ok = True
                err_msg = None
                try:
                    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                        if step_modes.get(step) == "reuse":
                            _reuse_pipeline_step(step, stl_path)
                        else:
                            _run_pipeline_step(step, stl_path, params, post_tips_mode, export_png)
                except Exception as exc:
                    ok = False
                    err_msg = f"{type(exc).__name__}: {exc}"
                    buffer.write("\n")
                    buffer.write(traceback.format_exc())
                elapsed = time.time() - started
                text = buffer.getvalue().strip()
                status_line = f"[{'OK' if ok else 'FAIL'}] {patient['id']} / {label} ({elapsed:.1f}s)"
                if err_msg:
                    status_line += f" - {err_msg}"
                _append_job_log(job, status_line)
                if text:
                    _append_job_log(job, text)
                with STATE_LOCK:
                    job["results"].setdefault(patient["id"], {})[step] = ok
                    if not ok:
                        job["errors"].append(status_line)
                _set_job_progress(job, completed_delta=1)
            remove_generated_outputs(stl_path.parent, keep_public=True)
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


def _run_pipeline_step(step: str, stl_path: Path, params: dict, post_tips_mode: str, export_png: bool):
    if step == "centerline":
        try:
            from geometry_feature_extract.extract_centerline import extract_centerline
        except ImportError:
            from extract_centerline import extract_centerline

        extract_centerline(
            str(stl_path),
            pitch=params["pitch"],
            min_branch_length_mm=params["min_branch_length_mm"],
            min_relative_length=params["min_relative_length"],
            min_radius_ratio=params["min_radius_ratio"],
            keep_radius_ratio=params["keep_radius_ratio"],
            absolute_min_branch_length_mm=params["absolute_min_branch_length_mm"],
            absolute_min_radius_mm=params["absolute_min_radius_mm"],
            merge_bp_distance_mm=params["merge_bp_distance_mm"],
        )
    elif step == "smooth":
        try:
            from geometry_feature_extract.smooth_centerline import smooth_centerline
        except ImportError:
            from smooth_centerline import smooth_centerline

        smooth_centerline(str(stl_path))
    elif step == "segment":
        script = APP_ROOT / "segment_vessels.py"
        post_tips = "1" if _post_tips_value(stl_path, post_tips_mode) else "0"
        cmd = [sys.executable, str(script), str(stl_path), "--post-tips", post_tips]
        print(f"  running segment subprocess: {' '.join(cmd)}")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            env=env,
        )
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())
        if result.returncode != 0:
            raise RuntimeError(f"segment_vessels failed with exit code {result.returncode}")
    elif step == "features":
        try:
            from geometry_feature_extract.extract_profiles import extract_profiles
            from geometry_feature_extract.extract_features import extract_all_features
        except ImportError:
            from extract_profiles import extract_profiles
            from extract_features import extract_all_features

        extract_profiles(
            str(stl_path),
            n_points=params["n_profile_points"],
            pitch=params["pitch"],
            curvature_window=params["curvature_window"],
            section_step=params["sample_step"],
        )
        extract_all_features(
            str(stl_path),
            n_fit_points=params["n_fit_points"],
            angle_fit_length_mm=params["angle_fit_length_mm"],
            curvature_window=params["curvature_window"],
            sample_step=params["sample_step"],
            pitch=params["pitch"],
        )
        remove_generated_outputs(stl_path.parent, keep_public=True)
        missing = [
            name for name in STEP_OUTPUTS["features"]
            if not feature_path(stl_path.parent, name).exists()
        ]
        if missing:
            raise RuntimeError(f"Feature extraction did not produce: {', '.join(missing)}")
    elif step == "export":
        try:
            from geometry_feature_extract.export_visualization import export_patient_visualization
        except ImportError:
            from export_visualization import export_patient_visualization

        export_patient_visualization(str(stl_path), export_html=True, export_png=export_png, verbose=True)
        remove_generated_outputs(stl_path.parent, keep_public=True)
    else:
        raise ValueError(f"Unknown step: {step}")


def _post_tips_value(stl_path: Path, mode: str):
    if mode == "pre":
        return False
    if mode == "post":
        return True
    return _is_post_tips(stl_path.parent.name)


def _zip_patient_outputs(patients: list[dict]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for patient in patients:
            stl_path = Path(patient["stl_path"])
            parent = stl_path.parent
            prefix = patient["id"]
            if stl_path.exists():
                zf.write(stl_path, f"{prefix}/{stl_path.name}")
            for name in OUTPUT_FILES:
                p = feature_path(parent, name)
                if p.exists():
                    zf.write(p, f"{prefix}/{FEATURES_DIRNAME}/{name}")
    return bio.getvalue()


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "PPGWorkbench/1.0"

    @staticmethod
    def _api_path(path: str) -> str:
        prefix = "/api/geometry"
        return "/api" + path[len(prefix):] if path.startswith(prefix) else path

    def do_GET(self):
        parsed = urlparse(self.path)
        raw_path = unquote(parsed.path)
        path = self._api_path(raw_path)
        try:
            if path == "/api/health":
                self._send_json({
                    "ok": True,
                    "time": _now(),
                    "version": WEB_FRONTEND_VERSION,
                    "file": str(Path(__file__).resolve()),
                    "pid": os.getpid(),
                    "runtime": _runtime_info(),
                })
            elif path.startswith("/api/session/") and path.endswith("/data"):
                self._handle_session_data(path, parsed.query)
            elif path.startswith("/api/session/") and path.endswith("/download"):
                self._handle_download(path, parsed.query)
            elif path.startswith("/api/job/"):
                self._handle_job(path)
            elif path == "/assets/plotly.min.js":
                self._serve_plotly()
            else:
                self._serve_static(raw_path)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = self._api_path(unquote(parsed.path))
        try:
            if path == "/api/session":
                self._handle_create_session()
            elif path == "/api/run":
                self._handle_run()
            elif path == "/api/centerline/delete-branches":
                self._handle_delete_centerline_branches()
            elif path == "/api/centerline/manual-segments":
                self._handle_manual_segments()
            elif path == "/api/analysis/save-ranges":
                self._handle_save_analysis_ranges()
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

    def _send_json(self, data, status=200):
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
        self._send_bytes(file_path.read_bytes(), ctype, extra_headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        })

    def _serve_plotly(self):
        try:
            import plotly

            p = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
            if p.exists():
                self._send_bytes(p.read_bytes(), "application/javascript; charset=utf-8")
                return
        except Exception:
            pass
        self._send_json({"error": "Local Plotly asset not found"}, status=404)

    def _handle_create_session(self):
        ctype = self.headers.get("Content-Type", "")
        if ctype.startswith("multipart/form-data"):
            fields = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": ctype,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            mode = fields.getvalue("mode") or "single"
            if mode != "single":
                raise ValueError("Multipart session creation only supports single-file mode.")
            session = _create_session_single(fields)
        else:
            payload = self._read_json_body()
            mode = payload.get("mode") or "batch"
            if mode == "batch":
                session = _create_session_batch(payload)
            else:
                raise ValueError("Single-file mode requires multipart upload.")
        self._send_json({"session": session})

    def _handle_run(self):
        payload = self._read_json_body()
        session_id = str(payload.get("session_id") or "")
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session.")
        steps = payload.get("steps") or []
        steps = [s for s in steps if s in PIPELINE_STEPS]
        if not steps:
            raise ValueError("No valid pipeline steps selected.")
        raw_step_modes = payload.get("step_modes") or {}
        step_modes = {
            s: "reuse" if raw_step_modes.get(s) == "reuse" else "recompute"
            for s in steps
        }
        params = _merge_params(payload.get("params"))
        post_tips_mode = payload.get("post_tips_mode") or "auto"
        export_png = bool(payload.get("export_png", False))
        patient_id = payload.get("patient_id")
        patients = session.get("patients") or []
        if patient_id and patient_id != "all":
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
        if not patients:
            raise ValueError("No patients selected.")
        job = _new_job(session_id, steps, patients, step_modes=step_modes)
        with STATE_LOCK:
            job["_patients_runtime"] = patients
            session["params"] = params
        thread = threading.Thread(
            target=_run_job,
            args=(job["id"], params, post_tips_mode, export_png),
            daemon=True,
        )
        thread.start()
        self._send_json({"job": job})

    def _handle_delete_centerline_branches(self):
        payload = self._read_json_body()
        session_id = str(payload.get("session_id") or "")
        patient_id = payload.get("patient_id")
        branch_ids = payload.get("branch_ids") or []
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session.")
        patient = _resolve_patient(session, patient_id)
        if not patient:
            raise ValueError("Patient not found.")
        result = delete_centerline_terminal_branches(
            Path(patient["stl_path"]), [str(item) for item in branch_ids])
        self._send_json({"ok": True, "result": result})

    def _handle_manual_segments(self):
        payload = self._read_json_body()
        session_id = str(payload.get("session_id") or "")
        patient_id = payload.get("patient_id")
        assignments = payload.get("assignments") or []
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session.")
        patient = _resolve_patient(session, patient_id)
        if not patient:
            raise ValueError("Patient not found.")
        if not isinstance(assignments, list):
            raise ValueError("Manual segment assignments must be a list.")
        result = save_manual_segment_assignments(Path(patient["stl_path"]), assignments)
        self._send_json({"ok": True, "result": result})

    def _handle_save_analysis_ranges(self):
        payload = self._read_json_body()
        session_id = str(payload.get("session_id") or "")
        patient_id = payload.get("patient_id")
        ranges = payload.get("ranges") or []
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            raise ValueError("Unknown session.")
        patient = _resolve_patient(session, patient_id)
        if not patient:
            raise ValueError("Patient not found.")
        if not isinstance(ranges, list):
            raise ValueError("Analysis ranges must be a list.")
        result = save_analysis_ranges(Path(patient["stl_path"]), ranges)
        self._send_json({"ok": True, "result": result})

    def _handle_job(self, path: str):
        job_id = path.rstrip("/").split("/")[-1]
        with STATE_LOCK:
            job = JOBS.get(job_id)
        if not job:
            self._send_json({"error": "Job not found"}, status=404)
            return
        self._send_json({"job": job})

    def _handle_session_data(self, path: str, query: str):
        session_id = path.split("/")[3]
        qs = parse_qs(query)
        patient_id = (qs.get("patient") or [None])[0]
        section_stride = _safe_int((qs.get("section_stride") or [10])[0], 10)
        max_faces = _safe_int((qs.get("max_faces") or [80000])[0], 80000)
        include_surface_sections = (qs.get("surface_sections") or ["0"])[0] == "1"
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            self._send_json({"error": "Session not found"}, status=404)
            return
        patient = _resolve_patient(session, patient_id)
        if not patient:
            self._send_json({"error": "Patient not found"}, status=404)
            return
        data = build_visualization_data(
            Path(patient["stl_path"]),
            section_stride=section_stride,
            max_faces=max_faces,
            include_surface_sections=include_surface_sections,
        )
        data["session"] = session
        self._send_json(data)

    def _handle_download(self, path: str, query: str):
        session_id = path.split("/")[3]
        qs = parse_qs(query)
        patient_id = (qs.get("patient") or ["all"])[0]
        with STATE_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            self._send_json({"error": "Session not found"}, status=404)
            return
        if patient_id == "all":
            patients = session.get("patients") or []
            name = f"ppg_outputs_{session_id}.zip"
        else:
            patient = _resolve_patient(session, patient_id)
            patients = [patient] if patient else []
            name = f"ppg_outputs_{patient_id or session_id}.zip"
        payload = _zip_patient_outputs(patients)
        self._send_bytes(
            payload,
            "application/zip",
            extra_headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--conda-env", default=None,
                        help="Restart the server under this conda environment before serving.")
    parser.add_argument("--conda-exe", default=None,
                        help="Path to conda.exe or conda.bat if it is not discoverable.")
    parser.add_argument("--no-conda-reexec", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = _load_config(args.config)
    args.host = args.host or str(config.get("host") or "127.0.0.1")
    args.port = args.port or int(config.get("port") or 8765)
    args.conda_env = (
        args.conda_env
        if args.conda_env is not None
        else str(config.get("conda_env") or "").strip() or None
    )
    args.conda_exe = (
        args.conda_exe
        if args.conda_exe is not None
        else str(config.get("conda_exe") or "").strip() or None
    )
    _maybe_reexec_in_conda(args, sys.argv[1:])
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    if getattr(sys, "stdout", None) is not None:
        print(f"PPG workbench version: {WEB_FRONTEND_VERSION}")
        print(f"Serving file: {Path(__file__).resolve()}")
        print(f"Process id: {os.getpid()}")
        print(f"PPG workbench running at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
