from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    np = None  # type: ignore[assignment]
    print("[WARN] matplotlib or numpy not installed, quality plot will be skipped", file=sys.stderr)


def _load_mesh(path: Path, process: bool = True):
    import trimesh

    mesh = trimesh.load_mesh(str(path), process=process)
    if hasattr(mesh, "geometry"):
        meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"No Trimesh geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Loaded mesh is not a Trimesh object: {type(mesh)}")
    return mesh


def _mesh_summary(path: Path) -> dict:
    try:
        mesh = _load_mesh(path, process=True)
    except ImportError:
        return {"path": str(path), "trimesh": False}
    except Exception as exc:
        return {"path": str(path), "trimesh": True, "error": str(exc)}

    components = mesh.split(only_watertight=False)
    face_counts = [int(len(component.faces)) for component in components]
    largest_component_face_ratio = max(face_counts) / len(mesh.faces) if face_counts and len(mesh.faces) else 0.0
    bounds = mesh.bounds.tolist() if len(mesh.vertices) else []
    extents = []
    if len(bounds) == 2:
        extents = [float(bounds[1][idx]) - float(bounds[0][idx]) for idx in range(3)]
    return {
        "path": str(path),
        "trimesh": True,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": bounds,
        "extents": extents,
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
    if summary.get("error"):
        issues.append(f"mesh could not be loaded: {summary['error']}")
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


def _effective_iterations(iterations: int) -> int:
    iterations = max(0, min(int(iterations), 300))
    if iterations == 0:
        return 0
    if iterations <= 20:
        return max(10, iterations * 5)
    return iterations


def _call_mesh_cleanup(mesh) -> None:
    for name in ("remove_duplicate_faces", "remove_degenerate_faces", "remove_infinite_values", "merge_vertices"):
        method = getattr(mesh, name, None)
        if callable(method):
            method()


def smooth_stl(input_path: Path, output_path: Path, iterations: int = 80, strength: float = 0.55) -> tuple[Path, dict]:
    import trimesh
    from trimesh.smoothing import filter_humphrey, filter_laplacian, filter_taubin

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_iterations = max(0, int(iterations))
    effective_iterations = _effective_iterations(requested_iterations)
    mesh = _load_mesh(input_path, process=False)

    if effective_iterations == 0:
        shutil.copy2(input_path, output_path)
        return output_path, {
            "method": "copy",
            "requested_iterations": requested_iterations,
            "effective_iterations": 0,
            "strength": 0.0,
            "mean_vertex_displacement": 0.0,
            "max_vertex_displacement": 0.0,
        }

    _call_mesh_cleanup(mesh)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0 or mesh.is_empty:
        raise RuntimeError("input mesh is empty after cleanup")

    original_vertices = mesh.vertices.copy()
    original_center = mesh.bounds.mean(axis=0)
    strength = max(0.05, min(float(strength), 0.85))

    # Taubin preserves volume better than plain Laplacian. Humphrey then makes the
    # stair-step boundary visibly softer without collapsing thin branches as quickly.
    filter_taubin(mesh, lamb=strength, nu=strength * 0.53, iterations=effective_iterations)
    filter_humphrey(mesh, alpha=0.08, beta=0.55, iterations=max(1, effective_iterations // 3))
    filter_laplacian(mesh, lamb=min(0.35, strength * 0.6), iterations=max(1, effective_iterations // 8), volume_constraint=True)

    mesh.fix_normals()
    new_center = mesh.bounds.mean(axis=0)
    if np is not None and np.all(np.isfinite(original_center)) and np.all(np.isfinite(new_center)):
        mesh.apply_translation(original_center - new_center)

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0 or mesh.is_empty:
        raise RuntimeError("smoothing produced empty or invalid mesh")

    if len(mesh.vertices) == len(original_vertices) and np is not None:
        displacement = np.linalg.norm(mesh.vertices - original_vertices, axis=1)
        mean_displacement = float(np.mean(displacement))
        max_displacement = float(np.max(displacement))
    else:
        mean_displacement = None
        max_displacement = None

    mesh.export(str(output_path), file_type="stl")
    return output_path, {
        "method": "taubin+humphrey+laplacian",
        "requested_iterations": requested_iterations,
        "effective_iterations": effective_iterations,
        "strength": strength,
        "mean_vertex_displacement": mean_displacement,
        "max_vertex_displacement": max_displacement,
    }


def _fmt_num(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _plot_quality_report(case, report_data: dict, output_png: Path) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return

    before = report_data.get("input_mesh", {})
    after = report_data.get("smooth_mesh", report_data.get("mesh", {}))
    check = report_data.get("quality_check", {})
    smoothing = report_data.get("smoothing", {})
    issues = check.get("issues", [])

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    fig.suptitle(f"Predict Smooth Quality - {case.name}", fontsize=15, weight="bold")

    ax_status = plt.axes([0.05, 0.72, 0.9, 0.18])
    ax_status.axis("off")
    quality = check.get("quality", "unknown")
    status_color = "#1f9d55" if quality == "ok" else "#c77700"
    ax_status.add_patch(plt.Rectangle((0, 0.15), 1, 0.7, color=status_color, alpha=0.12, transform=ax_status.transAxes))
    ax_status.text(0.03, 0.55, quality.upper(), color=status_color, fontsize=24, weight="bold", va="center")
    ax_status.text(
        0.22,
        0.55,
        f"method: {smoothing.get('method', 'n/a')}    "
        f"iterations: {smoothing.get('effective_iterations', 'n/a')}    "
        f"mean move: {_fmt_num(smoothing.get('mean_vertex_displacement'))} mm    "
        f"max move: {_fmt_num(smoothing.get('max_vertex_displacement'))} mm",
        fontsize=11,
        va="center",
    )

    ax_metrics = plt.axes([0.05, 0.34, 0.45, 0.32])
    labels = ["vertices", "faces", "components"]
    before_values = [before.get("vertices", 0), before.get("faces", 0), before.get("components", 0)]
    after_values = [after.get("vertices", 0), after.get("faces", 0), after.get("components", 0)]
    x = np.arange(len(labels))
    ax_metrics.bar(x - 0.18, before_values, 0.36, label="before", color="#94a3b8")
    ax_metrics.bar(x + 0.18, after_values, 0.36, label="smooth", color="#177e89")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(labels)
    ax_metrics.set_title("Before / After Counts")
    ax_metrics.legend()
    ax_metrics.grid(axis="y", alpha=0.25)

    ax_size = plt.axes([0.58, 0.34, 0.37, 0.32])
    extents = after.get("extents") or [0, 0, 0]
    ax_size.bar(["X", "Y", "Z"], extents, color=["#177e89", "#3b82f6", "#f59e0b"])
    ax_size.set_title("Smooth Mesh Size (mm)")
    ax_size.grid(axis="y", alpha=0.25)

    ax_issues = plt.axes([0.05, 0.06, 0.9, 0.22])
    ax_issues.axis("off")
    if issues:
        text = "\n".join(f"- {issue}" for issue in issues[:8])
        ax_issues.text(0.02, 0.88, "Issues", fontsize=12, weight="bold", color="#b42318", va="top")
        ax_issues.text(0.02, 0.66, text, fontsize=10, color="#4b5563", va="top", wrap=True)
    else:
        ax_issues.text(0.02, 0.55, "No quality issues detected.", fontsize=12, color="#1f9d55", weight="bold", va="center")

    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {case.name}: saved quality plot to {output_png}")


def _report_paths(case_path: Path) -> tuple[Path, Path, Path]:
    work_dir = case_path / "vkan_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir / "predict_check.json", case_path / "predict_check.json", work_dir / "predict_quality_report.png"


def check_and_smooth_case(case, iterations: int = 80, force: bool = False, strength: float = 0.55) -> Path:
    if not case.predict_stl.exists():
        raise FileNotFoundError(f"predict.stl not found: {case.predict_stl}")

    out = case.path / "predict_smooth.stl"
    if out.exists() and not force:
        return out

    input_summary = _mesh_summary(case.predict_stl)
    input_quality = _quality_check(input_summary)
    smoothing_stats = None
    error = None
    try:
        _out, smoothing_stats = smooth_stl(case.predict_stl, out, iterations=iterations, strength=strength)
    except Exception as exc:
        error = str(exc)
        print(f"[error] Smoothing failed for {case.name}: {exc}; falling back to copy original", file=sys.stderr)
        shutil.copy2(case.predict_stl, out)
        (case.path / "smooth_error.log").write_text(error, encoding="utf-8")

    smooth_summary = _mesh_summary(out)
    smooth_quality = _quality_check(smooth_summary)
    if error:
        smooth_quality = {"quality": "review", "issues": [f"smoothing failed: {error}", *smooth_quality.get("issues", [])]}

    report = {
        "mesh": smooth_summary,
        "input_mesh": input_summary,
        "smooth_mesh": smooth_summary,
        "input_quality_check": input_quality,
        "quality_check": smooth_quality,
        "smooth_iterations": (smoothing_stats or {}).get("effective_iterations", _effective_iterations(iterations)),
        "smoothing": smoothing_stats or {"method": "copy-fallback", "requested_iterations": int(iterations), "error": error},
        "outputs": {"smooth_stl": str(out)},
    }

    json_path, legacy_json_path, png_path = _report_paths(case.path)
    payload = json.dumps(report, indent=2)
    json_path.write_text(payload, encoding="utf-8")
    legacy_json_path.write_text(payload, encoding="utf-8")
    print(f"[json] {case.name}: saved report to {json_path}")

    if MATPLOTLIB_AVAILABLE:
        _plot_quality_report(case, report, png_path)

    return out


class PatientCase:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.predict_stl = path / "predict.stl"


def discover_patients(data_root: str | Path):
    root = Path(data_root)
    patient_dirs = [root] if (root / "predict.stl").exists() else sorted(p for p in root.iterdir() if p.is_dir())
    return [PatientCase(path) for path in patient_dirs if (path / "predict.stl").exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check, smooth and visualize STL meshes.")
    parser.add_argument("--data_root", default=r"F:\PCG data\dataset\test4all_sample", help="Root directory containing patient subdirectories")
    parser.add_argument("--patient", default=None, help="Only process a specific patient")
    parser.add_argument("--iterations", type=int, default=80, help="Smooth iterations; <=20 is amplified for visible smoothing")
    parser.add_argument("--strength", type=float, default=0.55, help="Smoothing strength, clamped to 0.05..0.85")
    parser.add_argument("--force", default=True, help="Overwrite existing smooth STL and reports")
    args = parser.parse_args()

    cases = discover_patients(args.data_root)
    if args.patient:
        cases = [case for case in cases if case.name == args.patient]

    if not cases:
        print(f"[check] No cases with predict.stl found in {args.data_root}", file=sys.stderr)
        sys.exit(1)

    for case in cases:
        try:
            out = check_and_smooth_case(case, iterations=args.iterations, force=args.force, strength=args.strength)
            print(f"[check] {case.name}: wrote smooth STL to {out}")
        except Exception as exc:
            print(f"[check] {case.name}: FAILED - {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
