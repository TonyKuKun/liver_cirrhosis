"""Canonical on-disk layout for per-patient geometry results.

Only the four public files are kept in ``<patient>/features``.
"""

from __future__ import annotations

import shutil
from pathlib import Path


FEATURES_DIRNAME = "features"
RAW_CENTERLINE_NAME = "centerline.txt"
SMOOTH_CENTERLINE_NAME = "newcenterline.txt"
SEGMENT_ASSIGNMENTS_NAME = "segment_assignments.json"
UNIFIED_FEATURES_NAME = "unified_features.json"
UNIFIED_FEATURES_BACKUP_NAME = "unified_features0.json"
POINTWISE_TEMP_NAME = ".pointwise_profiles.json"

PUBLIC_FEATURE_NAMES = (
    RAW_CENTERLINE_NAME,
    SMOOTH_CENTERLINE_NAME,
    SEGMENT_ASSIGNMENTS_NAME,
    UNIFIED_FEATURES_NAME,
    UNIFIED_FEATURES_BACKUP_NAME,
)

def patient_dir_from_stl(stl_path: str | Path) -> Path:
    return Path(stl_path).resolve().parent


def features_dir(patient_dir: str | Path, create: bool = False) -> Path:
    path = Path(patient_dir) / FEATURES_DIRNAME
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def feature_path(patient_dir: str | Path, name: str, create: bool = False) -> Path:
    return features_dir(patient_dir, create=create) / name


def resolve_feature_path(patient_dir: str | Path, canonical_name: str) -> Path | None:
    """Return a canonical feature path only when it exists."""
    canonical = feature_path(patient_dir, canonical_name)
    return canonical if canonical.exists() else None


def remove_generated_outputs(
    patient_dir: str | Path,
    keep_public: bool = True,
    preserve: tuple[str, ...] = (),
) -> list[str]:
    """Remove non-public files from a patient's canonical features folder."""
    parent = Path(patient_dir)
    removed: list[str] = []
    fdir = features_dir(parent, create=False)
    if fdir.exists():
        keep = (set(PUBLIC_FEATURE_NAMES) if keep_public else set()) | set(preserve)
        for path in fdir.iterdir():
            if path.is_file() and path.name not in keep:
                path.unlink()
                removed.append(str(path))
            elif path.is_dir() and not keep_public:
                shutil.rmtree(path)
                removed.append(str(path))
    return removed


def ensure_public_layout(patient_dir: str | Path) -> Path:
    """Create ``features`` and return it; no files are copied implicitly."""
    return features_dir(patient_dir, create=True)
