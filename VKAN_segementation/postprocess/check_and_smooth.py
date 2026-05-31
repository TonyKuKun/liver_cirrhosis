from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    from ..utils.common import discover_patients, smooth_stl
except (ImportError, ValueError):
    try:
        from VKAN_segementation.utils.common import discover_patients, smooth_stl
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from utils.common import discover_patients, smooth_stl


def _mesh_summary(path: Path) -> dict:
    try:
        import trimesh
    except ImportError:
        return {"path": str(path), "trimesh": False}
    mesh = trimesh.load_mesh(str(path), process=True)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    components = mesh.split(only_watertight=False)
    face_counts = [int(len(component.faces)) for component in components]
    largest_component_face_ratio = 0.0
    if face_counts and len(mesh.faces):
        largest_component_face_ratio = max(face_counts) / len(mesh.faces)
    return {
        "path": str(path),
        "trimesh": True,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
        "components": int(len(components)),
        "largest_component_face_ratio": float(largest_component_face_ratio),
        "watertight": bool(mesh.is_watertight),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }


def _quality_check(summary: dict) -> dict:
    issues: list[str] = []
    if not summary.get("trimesh", False):
        issues.append("trimesh is not installed; mesh summary is limited")
        return {"quality": "review", "issues": issues}

    vertices = int(summary.get("vertices") or 0)
    faces = int(summary.get("faces") or 0)
    if vertices == 0 or faces == 0:
        issues.append("mesh is empty")

    bounds = summary.get("bounds") or []
    flat_bounds = [value for row in bounds for value in row] if bounds else []
    if len(flat_bounds) != 6 or any(not math.isfinite(float(value)) for value in flat_bounds):
        issues.append("mesh bounds are invalid")

    components = int(summary.get("components") or 0)
    largest_ratio = float(summary.get("largest_component_face_ratio") or 0.0)
    if components > 10:
        issues.append(f"mesh has many disconnected components ({components})")
    elif components > 1 and largest_ratio < 0.7:
        issues.append(f"mesh is fragmented ({components} components, largest face ratio {largest_ratio:.2f})")

    if not summary.get("watertight", False):
        issues.append("mesh is not watertight")

    return {"quality": "review" if issues else "ok", "issues": issues}


def check_and_smooth_case(case, iterations: int = 8, force: bool = False) -> Path:
    if not case.predict_stl.exists():
        raise FileNotFoundError(case.predict_stl)
    out = case.path / "predict_smooth.stl"
    if out.exists() and not force:
        return out
    summary = _mesh_summary(case.predict_stl)
    iterations = max(0, min(int(iterations), 20))
    report = {"mesh": summary, "quality_check": _quality_check(summary), "smooth_iterations": iterations}
    (case.path / "vkan_work").mkdir(parents=True, exist_ok=True)
    (case.path / "vkan_work" / "predict_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return smooth_stl(case.predict_stl, out, iterations=iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and smooth predict.stl.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cases = [case for case in discover_patients(args.data_root) if case.predict_stl.exists()]
    if args.patient:
        cases = [case for case in cases if case.name == args.patient]
    for case in cases:
        out = check_and_smooth_case(case, iterations=args.iterations, force=args.force)
        print(f"[check] {case.name}: wrote {out}")


if __name__ == "__main__":
    main()

