"""
Local browser workbench for STL + hemodynamics NPZ inspection.

Run:
    python stl_npz_viewer.py --host 127.0.0.1 --port 8775

Workflow:
    1. Enter a patient folder that contains one or more STL files.
    2. Scan the folder and choose the matching *.hemodynamics.npz file.
    3. Pick a scalar field, such as wss_pa or velocity_m_per_s.
    4. Render the STL colored by the nearest exported centerline value.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


APP_ROOT = Path(__file__).resolve().parent
SEGMENTS = ["mpv", "sv", "smv", "lpv", "rpv", "tips", "lgv", "pgv"]
PREVIEW_MAX_FACES = 30000
DEFAULT_FIELDS = [
    "wss_pa",
    "velocity_m_per_s",
    "pressure_drop_pa",
    "reynolds",
    "radius_m",
    "area_m2",
    "local_R_pa_s_per_m4",
    "cum_R_pa_s_per_m3",
    "dean",
    "area_gradient",
    "shape_alpha",
]

STL_CACHE: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
MAP_CACHE: dict[tuple, tuple[np.ndarray, list[str], np.ndarray]] = {}
CACHE_LOCK = threading.Lock()


def read_stl_ascii_or_binary(stl_path):
    """Return STL vertices and triangular faces."""
    stl_path = os.fspath(stl_path)
    with open(stl_path, "rb") as f:
        head = f.read(80)
        count_bytes = f.read(4)

    is_ascii = head[:5].lower() == b"solid"
    if is_ascii:
        try:
            return _read_stl_ascii(stl_path)
        except ValueError:
            pass

    if len(count_bytes) != 4:
        raise ValueError(f"Invalid STL file: {stl_path}")
    return _read_stl_binary(stl_path)


def _read_stl_ascii(stl_path):
    verts = []
    faces = []
    facet = []
    with open(stl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("vertex"):
                parts = line.split()
                if len(parts) != 4:
                    continue
                facet.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("endfacet"):
                if len(facet) == 3:
                    base = len(verts)
                    verts.extend(facet)
                    faces.append([base, base + 1, base + 2])
                facet = []
    if not verts:
        raise ValueError("No ASCII STL vertices found")
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _read_stl_binary(stl_path):
    with open(stl_path, "rb") as f:
        f.read(80)
        n_tri = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        data = f.read(n_tri * 50)

    if len(data) < n_tri * 50:
        raise ValueError(f"Binary STL is truncated: {stl_path}")

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_tri, 50)
    triangles = raw[:, 12:48].copy().view(np.float32).reshape(n_tri, 3, 3)
    verts = triangles.reshape(-1, 3).astype(np.float32, copy=False)
    faces = np.arange(n_tri * 3, dtype=np.int32).reshape(n_tri, 3)
    return verts, faces


def write_ply_with_scalar(ply_path, vertices, faces, scalar, scalar_name="value"):
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"property float {scalar_name}\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, s in zip(vertices, scalar):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {float(s):.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def available_scalar_fields(npz):
    fields = set()
    for seg in SEGMENTS:
        arc_key = f"{seg}_arc_length_mm"
        if arc_key not in npz.files:
            continue
        n_points = np.asarray(npz[arc_key]).reshape(-1).size
        prefix = f"{seg}_"
        for key in npz.files:
            if not key.startswith(prefix):
                continue
            field = key[len(prefix):]
            value = np.asarray(npz[key])
            if value.reshape(-1).size == n_points and field not in {
                "arc_length_mm",
                "point_valid",
            }:
                fields.add(field)

    ordered = [f for f in DEFAULT_FIELDS if f in fields]
    ordered.extend(sorted(fields - set(ordered)))
    return ordered


def build_centerline_from_npz(npz, scalar_name, segment_name="All segments"):
    selected = SEGMENTS if segment_name == "All segments" else [segment_name]
    points = []
    values = []
    used_segments = []

    for seg in selected:
        keys = {
            "arc": f"{seg}_arc_length_mm",
            "valid": f"{seg}_point_valid",
            "endpoints": f"{seg}_endpoints_3d",
            "scalar": f"{seg}_{scalar_name}",
            "present": f"{seg}_present",
        }
        if any(k not in npz.files for k in keys.values() if k != keys["present"]):
            continue
        if keys["present"] in npz.files and not bool(np.asarray(npz[keys["present"]]).reshape(-1)[0]):
            continue

        arc = np.asarray(npz[keys["arc"]], dtype=float).reshape(-1)
        valid = np.asarray(npz[keys["valid"]]).reshape(-1) > 0
        endpoints = np.asarray(npz[keys["endpoints"]], dtype=float).reshape(2, 3)
        scalar = np.asarray(npz[keys["scalar"]], dtype=float).reshape(-1)

        if arc.size != scalar.size or valid.size != arc.size:
            continue
        if valid.sum() < 2 or np.allclose(endpoints, 0):
            continue

        valid_arc = arc[valid]
        denom = max(float(valid_arc.max() - valid_arc.min()), 1e-6)
        t = (arc - float(valid_arc.min())) / denom
        xyz = endpoints[0][None, :] + t[:, None] * (endpoints[1][None, :] - endpoints[0][None, :])

        points.append(xyz[valid])
        values.append(scalar[valid])
        used_segments.append(seg)

    if not points:
        raise ValueError(
            f"No mappable centerline values for field '{scalar_name}' and segment '{segment_name}'."
        )

    return np.concatenate(points, axis=0), np.concatenate(values, axis=0), used_segments


def map_scalar_to_vertices(vertices, centerline_points, centerline_values):
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(centerline_points).query(vertices, k=1)
        return centerline_values[idx]
    except Exception:
        return _nearest_neighbor_chunked(vertices, centerline_points, centerline_values)


def _nearest_neighbor_chunked(vertices, centerline_points, centerline_values, chunk_size=5000):
    out = np.empty(len(vertices), dtype=np.float32)
    for start in range(0, len(vertices), chunk_size):
        stop = min(start + chunk_size, len(vertices))
        d2 = ((vertices[start:stop, None, :] - centerline_points[None, :, :]) ** 2).sum(axis=-1)
        out[start:stop] = centerline_values[d2.argmin(axis=1)]
    return out


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


def _load_stl_cached(stl_path: str):
    key = (stl_path, os.path.getmtime(stl_path))
    with CACHE_LOCK:
        if key in STL_CACHE:
            return STL_CACHE[key]
    vertices, faces = read_stl_ascii_or_binary(stl_path)
    with CACHE_LOCK:
        STL_CACHE.clear()
        STL_CACHE[key] = (vertices, faces)
    return vertices, faces


def _mapped_scalar_cached(stl_path: str, npz_path: str, scalar_name: str, segment_name: str):
    vertices, _ = _load_stl_cached(stl_path)
    key = (
        stl_path,
        os.path.getmtime(stl_path),
        npz_path,
        os.path.getmtime(npz_path),
        scalar_name,
        segment_name,
    )
    with CACHE_LOCK:
        if key in MAP_CACHE:
            return MAP_CACHE[key]

    with np.load(npz_path, allow_pickle=True) as npz:
        centerline_points, centerline_values, used_segments = build_centerline_from_npz(
            npz, scalar_name, segment_name
        )
    vertex_scalar = map_scalar_to_vertices(vertices, centerline_points, centerline_values)
    value = (vertex_scalar, used_segments, centerline_points)
    with CACHE_LOCK:
        MAP_CACHE[key] = value
        while len(MAP_CACHE) > 8:
            MAP_CACHE.pop(next(iter(MAP_CACHE)))
    return value


def _discover_folder(folder: str):
    folder_path = Path(folder).expanduser()
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder_path}")

    stls = sorted(str(p) for p in folder_path.rglob("*.stl"))
    npzs = sorted(str(p) for p in folder_path.rglob("*.npz"))
    project_npz = APP_ROOT / "inference_out" / f"{folder_path.name}.hemodynamics.npz"
    if project_npz.exists() and str(project_npz) not in npzs:
        npzs.insert(0, str(project_npz))

    selected_stl = ""
    if stls:
        vessel = [p for p in stls if Path(p).name.lower() == "vessel.stl"]
        selected_stl = vessel[0] if vessel else stls[0]

    return {
        "folder": str(folder_path),
        "stls": stls,
        "npzs": npzs,
        "selected_stl": selected_stl,
        "selected_npz": npzs[0] if npzs else "",
    }


def _fields_for_npz(npz_path: str):
    if not npz_path:
        return []
    with np.load(npz_path, allow_pickle=True) as npz:
        return available_scalar_fields(npz)


def _scalar_limits(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _build_render_payload(payload: dict):
    stl_path = str(payload.get("stl_path") or "").strip()
    npz_path = str(payload.get("npz_path") or "").strip()
    scalar_name = str(payload.get("scalar_name") or "wss_pa").strip()
    segment_name = str(payload.get("segment_name") or "All segments").strip()
    max_faces = int(payload.get("max_faces") or PREVIEW_MAX_FACES)
    max_faces = max(1000, min(200000, max_faces))

    if not stl_path or not Path(stl_path).exists():
        raise FileNotFoundError(f"STL file does not exist: {stl_path}")
    if not npz_path or not Path(npz_path).exists():
        raise FileNotFoundError(f"NPZ file does not exist: {npz_path}")

    started = time.time()
    vertices, faces = _load_stl_cached(stl_path)
    vertex_scalar, used_segments, centerline_points = _mapped_scalar_cached(
        stl_path, npz_path, scalar_name, segment_name
    )

    if len(faces) > max_faces:
        step = int(np.ceil(len(faces) / max_faces))
        draw_faces = faces[::step]
    else:
        step = 1
        draw_faces = faces

    used_vertex_idx, inverse = np.unique(draw_faces.reshape(-1), return_inverse=True)
    render_vertices = vertices[used_vertex_idx]
    render_faces = inverse.reshape(-1, 3).astype(np.int32, copy=False)
    render_scalar = vertex_scalar[used_vertex_idx]
    cmin, cmax = _scalar_limits(render_scalar)

    center_step = max(1, int(np.ceil(len(centerline_points) / 1200)))
    centerline = centerline_points[::center_step]

    return {
        "mesh": {
            "vertices": render_vertices,
            "faces": render_faces,
            "scalar": render_scalar,
            "cmin": cmin,
            "cmax": cmax,
            "n_vertices": int(len(vertices)),
            "n_faces": int(len(faces)),
            "n_faces_rendered": int(len(draw_faces)),
            "sample_step": int(step),
        },
        "centerline": {
            "x": centerline[:, 0] if len(centerline) else [],
            "y": centerline[:, 1] if len(centerline) else [],
            "z": centerline[:, 2] if len(centerline) else [],
            "n_points": int(len(centerline_points)),
        },
        "summary": {
            "stl_name": Path(stl_path).name,
            "npz_name": Path(npz_path).name,
            "scalar_name": scalar_name,
            "segment_name": segment_name,
            "used_segments": used_segments,
            "scalar_min": float(np.nanmin(vertex_scalar)),
            "scalar_max": float(np.nanmax(vertex_scalar)),
            "elapsed_sec": time.time() - started,
        },
    }


def _build_ply_bytes(payload: dict):
    stl_path = str(payload.get("stl_path") or "").strip()
    npz_path = str(payload.get("npz_path") or "").strip()
    scalar_name = str(payload.get("scalar_name") or "wss_pa").strip()
    segment_name = str(payload.get("segment_name") or "All segments").strip()
    vertices, faces = _load_stl_cached(stl_path)
    vertex_scalar, _, _ = _mapped_scalar_cached(stl_path, npz_path, scalar_name, segment_name)
    bio = io.StringIO()
    bio.write("ply\nformat ascii 1.0\n")
    bio.write(f"element vertex {len(vertices)}\n")
    bio.write("property float x\nproperty float y\nproperty float z\n")
    bio.write(f"property float {scalar_name}\n")
    bio.write(f"element face {len(faces)}\n")
    bio.write("property list uchar int vertex_indices\n")
    bio.write("end_header\n")
    for v, s in zip(vertices, vertex_scalar):
        bio.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {float(s):.6f}\n")
    for face in faces:
        bio.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    return bio.getvalue().encode("utf-8")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PVP Flow Surface Workbench</title>
  <script src="/assets/plotly.min.js"></script>
  <script>window.Plotly || document.write('<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"><\/script>')</script>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef2f5;
      --panel: #ffffff;
      --panel-2: #f7f9fb;
      --text: #17212b;
      --muted: #627386;
      --line: #d6dee7;
      --accent: #177e89;
      --accent-2: #d9822b;
      --danger: #b42318;
      --ok: #217a52;
      --shadow: 0 8px 24px rgba(23, 33, 43, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background: var(--bg);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }
    button, input, select { font: inherit; }
    button {
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      cursor: pointer;
    }
    button:disabled { cursor: not-allowed; opacity: 0.55; }
    button:hover:not(:disabled) { border-color: #91a6b8; }
    .primary-btn { border-color: var(--accent); background: var(--accent); color: #fff; }
    .app-shell {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 12px;
      overflow-y: auto;
      border-right: 1px solid var(--line);
      background: #f6f8fa;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 8px;
    }
    .brand h1 { margin: 0; font-size: 18px; line-height: 1.2; }
    .brand p { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
    .brand-mark {
      display: grid;
      width: 46px;
      height: 46px;
      place-items: center;
      border-radius: 8px;
      background: #183642;
      color: #fff;
      font-weight: 700;
      letter-spacing: 0;
    }
    .panel {
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }
    .panel-title {
      margin-bottom: 10px;
      color: #243447;
      font-size: 13px;
      font-weight: 700;
    }
    .field {
      display: grid;
      gap: 5px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .field input, .field select {
      width: 100%;
      min-height: 34px;
      padding: 6px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      background: #fff;
    }
    .run-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
    }
    .toggle-line {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .status-pill {
      margin-top: 10px;
      padding: 7px 8px;
      border-radius: 6px;
      background: #eef7f5;
      color: var(--accent);
      font-size: 12px;
      line-height: 1.45;
    }
    .log-panel { flex: 1; min-height: 210px; }
    pre {
      overflow: auto;
      max-height: 310px;
      margin: 0;
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #111820;
      color: #dbe7ef;
      white-space: pre-wrap;
      font-size: 11px;
      line-height: 1.45;
    }
    .workspace {
      display: grid;
      grid-template-rows: auto minmax(360px, 1fr) 220px;
      min-width: 0;
      min-height: 100vh;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .layer-group, .slider-group {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px 12px;
    }
    .layer-group label, .slider-group label {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: #33465a;
      font-size: 13px;
    }
    .slider-group input { width: 130px; }
    .viewer-wrap {
      position: relative;
      min-height: 0;
      background: #f8fafc;
    }
    #viewer {
      width: 100%;
      height: 100%;
      min-height: 360px;
    }
    .empty-state {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: var(--muted);
      pointer-events: none;
      text-align: center;
      padding: 24px;
    }
    .empty-state.hidden { display: none; }
    .inspector {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr) minmax(260px, 1fr);
      gap: 10px;
      padding: 10px;
      overflow: hidden;
      border-top: 1px solid var(--line);
      background: #f6f8fa;
    }
    .metric-card, .picked-info {
      height: 165px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 8px;
    }
    .metric {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 5px 0;
      border-bottom: 1px solid #edf1f5;
      color: var(--muted);
      font-size: 12px;
    }
    .metric:last-child { border-bottom: 0; }
    .metric strong { color: var(--text); font-weight: 600; text-align: right; }
    .hidden { display: none !important; }
    @media (max-width: 980px) {
      .app-shell { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .workspace { grid-template-rows: auto 520px auto; }
      .inspector { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <header class="brand">
        <div class="brand-mark">PVP</div>
        <div>
          <h1>Flow Surface Workbench</h1>
          <p id="serverState">Connecting to local service</p>
        </div>
      </header>

      <section class="panel">
        <div class="panel-title">Input</div>
        <label class="field">
          <span>Patient folder</span>
          <input id="patientFolder" type="text" placeholder="E:\dataset\patient001" />
        </label>
        <div class="run-row">
          <button id="scanBtn" class="primary-btn" type="button">Scan Folder</button>
          <button id="clearBtn" type="button">Clear</button>
        </div>
        <label class="field">
          <span>STL file</span>
          <select id="stlSelect"></select>
        </label>
        <label class="field">
          <span>NPZ file</span>
          <select id="npzSelect"></select>
        </label>
        <div id="folderStatus" class="status-pill">Enter a local folder path and scan.</div>
      </section>

      <section class="panel">
        <div class="panel-title">Render</div>
        <label class="field">
          <span>Scalar field</span>
          <select id="fieldSelect"></select>
        </label>
        <label class="field">
          <span>Segment</span>
          <select id="segmentSelect">
            <option>All segments</option>
            <option>mpv</option><option>sv</option><option>smv</option><option>lpv</option>
            <option>rpv</option><option>tips</option><option>lgv</option><option>pgv</option>
          </select>
        </label>
        <label class="field">
          <span>Preview face limit</span>
          <input id="maxFaces" type="number" min="1000" max="200000" step="1000" value="30000" />
        </label>
        <div class="run-row">
          <button id="renderBtn" class="primary-btn" type="button">Render</button>
          <button id="exportBtn" type="button">Export PLY</button>
        </div>
      </section>

      <section class="panel log-panel">
        <div class="panel-title">Log</div>
        <pre id="logs"></pre>
      </section>
    </aside>

    <main class="workspace">
      <div class="toolbar">
        <div class="layer-group">
          <label><input id="meshToggle" type="checkbox" checked />Surface</label>
          <label><input id="centerlineToggle" type="checkbox" checked />Centerline points</label>
        </div>
        <div class="slider-group">
          <label>Opacity <input id="meshOpacity" type="range" min="20" max="100" value="96" /></label>
          <span id="opacityValue">96%</span>
        </div>
      </div>

      <section class="viewer-wrap">
        <div id="viewer"></div>
        <div id="emptyState" class="empty-state">Scan a folder, choose STL + NPZ, then render the mapped surface.</div>
      </section>

      <section class="inspector">
        <div>
          <div class="panel-title">Surface</div>
          <div id="surfaceMetrics" class="metric-card"></div>
        </div>
        <div>
          <div class="panel-title">Mapping</div>
          <div id="mappingMetrics" class="metric-card"></div>
        </div>
        <div>
          <div class="panel-title">Selection</div>
          <div id="pickedInfo" class="picked-info">No point selected.</div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const DEFAULT_FIELDS = %DEFAULT_FIELDS%;
    const state = { folder: null, data: null, busy: false };
    const $ = (id) => document.getElementById(id);

    function init() {
      $("scanBtn").addEventListener("click", scanFolder);
      $("clearBtn").addEventListener("click", clearAll);
      $("npzSelect").addEventListener("change", refreshFields);
      $("renderBtn").addEventListener("click", renderSurface);
      $("exportBtn").addEventListener("click", exportPly);
      $("meshOpacity").addEventListener("input", () => {
        $("opacityValue").textContent = `${$("meshOpacity").value}%`;
        renderScene();
      });
      $("meshToggle").addEventListener("change", renderScene);
      $("centerlineToggle").addEventListener("change", renderScene);
      fillSelect($("fieldSelect"), DEFAULT_FIELDS);
      checkHealth();
      renderMetrics();
    }

    async function checkHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        $("serverState").textContent = data.ok ? `Local service ready | Python ${data.python}` : "Service error";
      } catch (err) {
        $("serverState").textContent = "Service unavailable";
      }
    }

    async function scanFolder() {
      setBusy(true);
      try {
        const folder = $("patientFolder").value.trim();
        if (!folder) throw new Error("Please enter a patient folder.");
        const payload = await postJson("/api/folder", { folder });
        state.folder = payload;
        fillSelect($("stlSelect"), payload.stls);
        fillSelect($("npzSelect"), payload.npzs);
        $("stlSelect").value = payload.selected_stl || "";
        $("npzSelect").value = payload.selected_npz || "";
        $("folderStatus").textContent = `Found ${payload.stls.length} STL file(s), ${payload.npzs.length} NPZ file(s).`;
        await refreshFields();
        logLine(`Scanned ${payload.folder}`);
      } catch (err) {
        showError(err);
      } finally {
        setBusy(false);
      }
    }

    async function refreshFields() {
      const npz = $("npzSelect").value;
      if (!npz) {
        fillSelect($("fieldSelect"), DEFAULT_FIELDS);
        return;
      }
      try {
        const payload = await postJson("/api/fields", { npz_path: npz });
        fillSelect($("fieldSelect"), payload.fields.length ? payload.fields : DEFAULT_FIELDS);
      } catch (err) {
        showError(err);
      }
    }

    async function renderSurface() {
      setBusy(true);
      $("emptyState").classList.remove("hidden");
      try {
        const payload = await postJson("/api/render", currentPayload());
        state.data = payload;
        renderScene();
        renderMetrics();
        logLine(`Rendered ${payload.summary.stl_name} in ${fmt(payload.summary.elapsed_sec, 2)}s.`);
      } catch (err) {
        showError(err);
      } finally {
        setBusy(false);
      }
    }

    async function exportPly() {
      setBusy(true);
      try {
        const res = await fetch("/api/export", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(currentPayload()),
        });
        if (!res.ok) {
          const payload = await res.json().catch(() => ({}));
          throw new Error(payload.error || `HTTP ${res.status}`);
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const stlStem = ($("stlSelect").value.split(/[\\/]/).pop() || "surface").replace(/\.stl$/i, "");
        a.href = url;
        a.download = `${stlStem}.${$("fieldSelect").value}.ply`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
        logLine("Exported full-resolution PLY.");
      } catch (err) {
        showError(err);
      } finally {
        setBusy(false);
      }
    }

    function currentPayload() {
      const stl = $("stlSelect").value;
      const npz = $("npzSelect").value;
      if (!stl || !npz) throw new Error("Please choose both STL and NPZ files.");
      return {
        stl_path: stl,
        npz_path: npz,
        scalar_name: $("fieldSelect").value || "wss_pa",
        segment_name: $("segmentSelect").value || "All segments",
        max_faces: Number($("maxFaces").value || 30000),
      };
    }

    function renderScene() {
      if (!state.data || !window.Plotly) return;
      const traces = [];
      const mesh = state.data.mesh;
      const vertices = mesh.vertices || [];
      const faces = mesh.faces || [];
      if ($("meshToggle").checked && vertices.length && faces.length) {
        traces.push({
          type: "mesh3d",
          name: state.data.summary.scalar_name,
          x: vertices.map((v) => v[0]),
          y: vertices.map((v) => v[1]),
          z: vertices.map((v) => v[2]),
          i: faces.map((f) => f[0]),
          j: faces.map((f) => f[1]),
          k: faces.map((f) => f[2]),
          intensity: mesh.scalar,
          intensitymode: "vertex",
          colorscale: "Turbo",
          cmin: mesh.cmin,
          cmax: mesh.cmax,
          opacity: Number($("meshOpacity").value) / 100,
          colorbar: { title: state.data.summary.scalar_name, thickness: 13 },
          hovertemplate: `${state.data.summary.scalar_name}: %{intensity:.4g}<extra></extra>`,
          flatshading: false,
          lighting: { ambient: 0.55, diffuse: 0.8, specular: 0.12 },
        });
      }
      const cl = state.data.centerline || {};
      if ($("centerlineToggle").checked && cl.x?.length) {
        traces.push({
          type: "scatter3d",
          mode: "markers",
          name: "Centerline samples",
          x: cl.x,
          y: cl.y,
          z: cl.z,
          marker: { size: 3, color: "#111820", opacity: 0.72 },
          hoverinfo: "skip",
        });
      }
      const layout = {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: "#f8fafc",
        scene: {
          aspectmode: "data",
          xaxis: axisLayout("X"),
          yaxis: axisLayout("Y"),
          zaxis: axisLayout("Z"),
          camera: { eye: { x: 1.55, y: 1.45, z: 1.05 }, up: { x: 0, y: 0, z: 1 } },
        },
        legend: {
          x: 0.01,
          y: 0.99,
          bgcolor: "rgba(255,255,255,0.82)",
          bordercolor: "#d6dee7",
          borderwidth: 1,
          font: { size: 11 },
        },
        uirevision: "stl-npz-workbench",
      };
      Plotly.react("viewer", traces, layout, { displaylogo: false, responsive: true, scrollZoom: true });
      $("emptyState").classList.toggle("hidden", traces.length > 0);
      const plot = $("viewer");
      if (typeof plot.removeAllListeners === "function") plot.removeAllListeners("plotly_click");
      plot.on("plotly_click", (event) => {
        const point = event.points?.[0];
        if (!point) return;
        $("pickedInfo").innerHTML = `
          <div class="metric"><span>Trace</span><strong>${escapeHtml(point.data?.name || "")}</strong></div>
          <div class="metric"><span>X</span><strong>${fmt(point.x, 4)}</strong></div>
          <div class="metric"><span>Y</span><strong>${fmt(point.y, 4)}</strong></div>
          <div class="metric"><span>Z</span><strong>${fmt(point.z, 4)}</strong></div>
          <div class="metric"><span>Value</span><strong>${fmt(point.intensity, 5)}</strong></div>
        `;
      });
    }

    function axisLayout(title) {
      return {
        title,
        backgroundcolor: "#f8fafc",
        gridcolor: "#d9e1e8",
        zerolinecolor: "#c9d4df",
        showspikes: false,
      };
    }

    function renderMetrics() {
      const data = state.data;
      const surface = $("surfaceMetrics");
      const mapping = $("mappingMetrics");
      if (!data) {
        surface.innerHTML = metricRow("Status", "No surface loaded");
        mapping.innerHTML = metricRow("Status", "No mapping loaded");
        return;
      }
      surface.innerHTML = [
        metricRow("STL", data.summary.stl_name),
        metricRow("Vertices", data.mesh.n_vertices.toLocaleString()),
        metricRow("Faces", data.mesh.n_faces.toLocaleString()),
        metricRow("Rendered faces", data.mesh.n_faces_rendered.toLocaleString()),
        metricRow("Sample step", data.mesh.sample_step),
      ].join("");
      mapping.innerHTML = [
        metricRow("NPZ", data.summary.npz_name),
        metricRow("Field", data.summary.scalar_name),
        metricRow("Segment", data.summary.segment_name),
        metricRow("Used segments", data.summary.used_segments.join(", ")),
        metricRow("Range", `${fmt(data.summary.scalar_min, 5)} - ${fmt(data.summary.scalar_max, 5)}`),
      ].join("");
    }

    function fillSelect(select, values) {
      select.innerHTML = "";
      values.forEach((value) => {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      });
    }

    function metricRow(label, value) {
      return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? "NA")}</strong></div>`;
    }

    function clearAll() {
      state.folder = null;
      state.data = null;
      ["stlSelect", "npzSelect"].forEach((id) => $(id).innerHTML = "");
      fillSelect($("fieldSelect"), DEFAULT_FIELDS);
      $("folderStatus").textContent = "Enter a local folder path and scan.";
      $("logs").textContent = "";
      $("pickedInfo").textContent = "No point selected.";
      $("emptyState").classList.remove("hidden");
      if (window.Plotly) Plotly.purge("viewer");
      renderMetrics();
    }

    function setBusy(isBusy) {
      state.busy = isBusy;
      document.querySelectorAll("button").forEach((button) => button.disabled = isBusy);
    }

    function logLine(text) {
      $("logs").textContent += `${new Date().toLocaleTimeString()}  ${text}\n`;
      $("logs").scrollTop = $("logs").scrollHeight;
    }

    function showError(err) {
      const message = err?.message || String(err);
      $("folderStatus").textContent = message;
      $("folderStatus").style.color = "var(--danger)";
      logLine(`ERROR: ${message}`);
    }

    async function postJson(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      return data;
    }

    function fmt(value, digits = 3) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "NA";
      return n.toFixed(digits);
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    window.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
""".replace("%DEFAULT_FIELDS%", json.dumps(DEFAULT_FIELDS))


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "StlNpzWorkbench/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/health":
                self._send_json({
                    "ok": True,
                    "time": time.time(),
                    "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                })
            elif path == "/assets/plotly.min.js":
                self._serve_plotly()
            elif path in ("", "/"):
                self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/folder":
                self._send_json(_discover_folder(str(payload.get("folder") or "")))
            elif parsed.path == "/api/fields":
                self._send_json({"fields": _fields_for_npz(str(payload.get("npz_path") or ""))})
            elif parsed.path == "/api/render":
                self._send_json(_build_render_payload(payload))
            elif parsed.path == "/api/export":
                data = _build_ply_bytes(payload)
                stl_name = Path(str(payload.get("stl_path") or "surface.stl")).with_suffix("").name
                scalar = str(payload.get("scalar_name") or "value")
                self._send_bytes(
                    data,
                    "application/octet-stream",
                    extra_headers={"Content-Disposition": f'attachment; filename="{stl_name}.{scalar}.ply"'},
                )
            else:
                self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
                status=400,
            )

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

    def _serve_plotly(self):
        try:
            import plotly

            plotly_path = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
            if plotly_path.exists():
                self._send_bytes(plotly_path.read_bytes(), "application/javascript; charset=utf-8")
                return
        except Exception:
            pass
        self._send_json({"error": "Local Plotly asset not found"}, status=404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"STL/NPZ workbench running at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
