from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


INVALID_MARKERS = ("@", "!")


@dataclass(frozen=True)
class PatientCase:
    name: str
    path: Path
    dcm_dir: Path
    label_stl: Path
    pretrain_stl: Path
    predict_stl: Path
    is_post_tips: bool


def is_valid_patient_name(name: str) -> bool:
    return not any(marker in name for marker in INVALID_MARKERS)


def discover_patients(root: str | Path) -> list[PatientCase]:
    """Find patient folders following the project convention."""
    root = Path(root)
    cases: list[PatientCase] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if not is_valid_patient_name(path.name):
            continue
        dcm_dir = path / "dcm"
        label_stl = path / "vessel.stl"
        if not dcm_dir.exists():
            continue
        cases.append(
            PatientCase(
                name=path.name,
                path=path,
                dcm_dir=dcm_dir,
                label_stl=label_stl,
                pretrain_stl=path / "pretrain.stl",
                predict_stl=path / "predict.stl",
                is_post_tips="#" in path.name,
            )
        )
    return cases


def require_existing_labels(cases: Iterable[PatientCase]) -> list[PatientCase]:
    return [case for case in cases if case.label_stl.exists()]

