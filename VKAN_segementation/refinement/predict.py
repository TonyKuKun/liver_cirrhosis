from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from ..utils.common import discover_patients, stl_to_voxels, voxels_to_stl
    from .model import VesselVKAN
except ImportError:
    from VKAN_segementation.utils.common import discover_patients, stl_to_voxels, voxels_to_stl
    from VKAN_segementation.refinement.model import VesselVKAN


def predict_case(case, checkpoint: dict, threshold: float = 0.5, out_path: Path | None = None) -> Path:
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
    return voxels_to_stl(prob, bounds, out_path or case.predict_stl, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predict.stl from pretrain.stl.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    cases = [case for case in discover_patients(args.data_root) if case.pretrain_stl.exists()]
    if args.patient:
        cases = [case for case in cases if case.name == args.patient]
    if not cases:
        raise RuntimeError("No matching cases with pretrain.stl found.")
    for case in cases:
        out = predict_case(case, checkpoint, threshold=args.threshold)
        print(f"[predict] {case.name}: wrote {out}")


if __name__ == "__main__":
    main()

