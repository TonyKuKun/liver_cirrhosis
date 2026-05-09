from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .config import discover_patients
    from .mesh_ops import stl_to_voxels, voxels_to_stl
    from .model import VesselVKAN
except ImportError:
    from config import discover_patients
    from mesh_ops import stl_to_voxels, voxels_to_stl
    from model import VesselVKAN


def predict_case(case, checkpoint: dict, checkpoint_path: Path, threshold: float = 0.5, out_path: Path | None = None) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = int(checkpoint.get("grid_size", checkpoint.get("args", {}).get("grid_size", 96)))
    base_channels = int(checkpoint.get("base_channels", checkpoint.get("args", {}).get("base_channels", 16)))
    model = VesselVKAN(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    grid, bounds = stl_to_voxels(case.pretrain_stl, grid_size=grid_size)
    x = torch.from_numpy(grid[None, None]).float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    out_path = out_path or case.predict_stl
    return voxels_to_stl(prob, bounds, out_path, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predict.stl from pretrain.stl.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cases = [case for case in discover_patients(args.data_root) if case.pretrain_stl.exists()]
    if args.patient:
        cases = [case for case in cases if case.name == args.patient]
    if not cases:
        raise RuntimeError("No matching cases with pretrain.stl found.")
    for case in cases:
        out = predict_case(case, checkpoint, ckpt_path, threshold=args.threshold)
        print(f"[predict] {case.name}: wrote {out}")


if __name__ == "__main__":
    main()

