from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ..utils.common import GemmaClient, discover_patients, smooth_stl
except ImportError:
    from VKAN_segementation.utils.common import GemmaClient, discover_patients, smooth_stl


def _mesh_summary(path: Path) -> dict:
    try:
        import trimesh
    except ImportError:
        return {"path": str(path), "trimesh": False}
    mesh = trimesh.load_mesh(str(path), process=True)
    if hasattr(mesh, "geometry"):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return {
        "path": str(path),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": mesh.bounds.tolist(),
        "watertight": bool(mesh.is_watertight),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }


def llm_mesh_check(client: GemmaClient, case_name: str, summary: dict, is_post_tips: bool) -> dict:
    if not client.enabled:
        return {}
    system = "You are checking portal vein STL extraction quality. Return strict JSON only."
    prompt = {
        "patient": case_name,
        "is_post_tips": is_post_tips,
        "mesh_summary": summary,
        "checklist": [
            "include SV, short SMV, LPV, RPV",
            "keep LGV/PGV if compensation exists",
            "keep TIPS stent/tube for post-TIPS cases",
            "flag severe fragmentation, missing branch, or obvious non-vessel shell",
        ],
        "return_schema": {"quality": "ok|review", "issues": ["string"], "smooth_iterations": "integer 0..20"},
    }
    return client.chat_json(system, json.dumps(prompt, ensure_ascii=True), [])


def check_and_smooth_case(case, client: GemmaClient | None = None, iterations: int = 8, force: bool = False) -> Path:
    if not case.predict_stl.exists():
        raise FileNotFoundError(case.predict_stl)
    out = case.path / "predict_smooth.stl"
    if out.exists() and not force:
        return out
    summary = _mesh_summary(case.predict_stl)
    llm = llm_mesh_check(client, case.name, summary, case.is_post_tips) if client else {}
    try:
        iterations = max(0, min(int(llm.get("smooth_iterations", iterations)), 20))
    except Exception:
        pass
    report = {"mesh": summary, "llm_check": llm, "smooth_iterations": iterations}
    (case.path / "vkan_work").mkdir(parents=True, exist_ok=True)
    (case.path / "vkan_work" / "predict_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return smooth_stl(case.predict_stl, out, iterations=iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and smooth predict.stl.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    client = GemmaClient(api_key=args.api_key, model=args.model, base_url=args.api_base_url)
    cases = [case for case in discover_patients(args.data_root) if case.predict_stl.exists()]
    if args.patient:
        cases = [case for case in cases if case.name == args.patient]
    for case in cases:
        out = check_and_smooth_case(case, client, iterations=args.iterations, force=args.force)
        print(f"[check] {case.name}: wrote {out}")


if __name__ == "__main__":
    main()

