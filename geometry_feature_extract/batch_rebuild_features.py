"""Rebuild pointwise profiles and unified features for a patient dataset."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from extract_features import extract_all_features
from extract_profiles import extract_profiles


def _ready_patients(root: Path) -> list[Path]:
    return sorted(
        patient
        for patient in root.iterdir()
        if patient.is_dir()
        and (patient / "vessel.stl").exists()
        and (patient / "features" / "newcenterline.txt").exists()
        and (patient / "features" / "segment_assignments.json").exists()
    )


def rebuild_dataset(root: Path, n_points: int = 200) -> dict:
    patients = _ready_patients(root)
    succeeded = []
    failed = {}
    started = time.time()
    print(f"BATCH_START patients={len(patients)}", flush=True)

    for index, patient in enumerate(patients, 1):
        patient_started = time.time()
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                profiles = extract_profiles(
                    str(patient / "vessel.stl"), n_points=n_points)
                features = extract_all_features(str(patient / "vessel.stl"))
            if not profiles or not features:
                raise RuntimeError("extractor returned an empty result")
            succeeded.append(patient.name)
            elapsed = time.time() - patient_started
            print(
                f"[{index}/{len(patients)}] OK {patient.name} {elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failed[patient.name] = {
                "error": repr(exc),
                "tail": captured.getvalue()[-2000:],
            }
            elapsed = time.time() - patient_started
            print(
                f"[{index}/{len(patients)}] FAIL {patient.name} "
                f"{elapsed:.1f}s {exc!r}",
                flush=True,
            )

    result = {
        "ready": len(patients),
        "success": len(succeeded),
        "failed": failed,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    print("BATCH_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--n-points", type=int, default=200)
    args = parser.parse_args()
    result = rebuild_dataset(args.root, n_points=args.n_points)
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
