from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DicomVolume:
    volume_hu: np.ndarray
    spacing_zyx: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]


def _slice_position(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return float(ipp[2])
    if hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def load_dicom_series(dcm_dir: str | Path) -> DicomVolume:
    """Load a CT DICOM folder as a z-y-x HU volume."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required to read DICOM files.") from exc

    dcm_dir = Path(dcm_dir)
    files = [p for p in dcm_dir.rglob("*") if p.is_file()]
    slices = []
    for file in files:
        try:
            ds = pydicom.dcmread(str(file), force=True)
            if hasattr(ds, "PixelData"):
                slices.append(ds)
        except Exception:
            continue
    if not slices:
        raise FileNotFoundError(f"No readable DICOM slices found in {dcm_dir}")

    slices.sort(key=_slice_position)
    arrays = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + intercept)

    volume = np.stack(arrays, axis=0)
    first = slices[0]
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    sy, sx = float(pixel_spacing[0]), float(pixel_spacing[1])
    if len(slices) > 1:
        dz = abs(_slice_position(slices[1]) - _slice_position(slices[0]))
        if dz <= 0:
            dz = float(getattr(first, "SliceThickness", 1.0))
    else:
        dz = float(getattr(first, "SliceThickness", 1.0))
    ipp = getattr(first, "ImagePositionPatient", [0.0, 0.0, 0.0])
    origin = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    return DicomVolume(volume_hu=volume, spacing_zyx=(dz, sy, sx), origin_xyz=origin)


def save_mosaic_png(volume_hu: np.ndarray, out_path: str | Path, n_slices: int = 12) -> Path | None:
    """Save a compact CT preview for optional multimodal API prompts."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    out_path = Path(out_path)
    vol = np.asarray(volume_hu)
    lo, hi = np.percentile(vol, [1, 99])
    lo = min(lo, -100.0)
    hi = max(hi, 350.0)
    idx = np.linspace(max(0, vol.shape[0] * 0.15), max(0, vol.shape[0] * 0.85), n_slices)
    slices = []
    for i in idx.astype(int):
        img = np.clip((vol[i] - lo) / max(hi - lo, 1e-3), 0, 1)
        img = (img * 255).astype(np.uint8)
        slices.append(Image.fromarray(img).resize((192, 192)))
    cols = 4
    rows = int(np.ceil(len(slices) / cols))
    canvas = Image.new("L", (cols * 192, rows * 192), 0)
    draw = ImageDraw.Draw(canvas)
    for j, img in enumerate(slices):
        x = (j % cols) * 192
        y = (j // cols) * 192
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + 4), str(int(idx[j])), fill=255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path

