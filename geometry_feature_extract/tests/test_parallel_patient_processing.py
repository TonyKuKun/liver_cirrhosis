from pathlib import Path

import main

from main import DEFAULT_PARAMS, PipelineSteps, process_stl_files


def _disabled_steps():
    steps = PipelineSteps()
    steps.extract_centerline = False
    steps.smooth_centerline = False
    steps.segment_vessels = False
    steps.extract_features = False
    steps.extract_profiles = False
    steps.export_visualization = False
    steps.visualize = False
    return steps


def test_process_stl_files_runs_patients_in_process_pool(tmp_path):
    for name in ("patient_a", "patient_b"):
        patient_dir = tmp_path / name
        patient_dir.mkdir()
        (patient_dir / "vessel.stl").touch()

    summary = process_stl_files(
        str(tmp_path),
        params=dict(DEFAULT_PARAMS),
        steps=_disabled_steps(),
        max_workers=2,
    )

    assert summary["total"] == 2
    assert summary["no_stl"] == 0
    assert (tmp_path / "patient_a" / "features").is_dir()
    assert (tmp_path / "patient_b" / "features").is_dir()


def test_process_stl_files_handles_empty_job_list(tmp_path):
    (tmp_path / "patient_without_stl").mkdir()

    summary = process_stl_files(
        str(tmp_path),
        params=dict(DEFAULT_PARAMS),
        steps=_disabled_steps(),
    )

    assert summary["total"] == 1
    assert summary["no_stl"] == 1


def test_process_stl_files_skips_failed_patient_in_serial_mode(tmp_path, monkeypatch):
    for name in ("patient_a", "patient_b"):
        patient_dir = tmp_path / name
        patient_dir.mkdir()
        (patient_dir / "vessel.stl").touch()

    processed = []

    def fake_process(stl_path, post_tips, params, steps):
        patient = Path(stl_path).parent.name
        if patient == "patient_a":
            raise RuntimeError("synthetic patient failure")
        processed.append(patient)
        return {
            "centerline": False,
            "smooth": False,
            "segment": False,
            "features": False,
            "profiles": False,
            "export_vis": False,
            "visualize": False,
        }

    monkeypatch.setattr(main, "_process_one_patient", fake_process)
    process_stl_files(
        str(tmp_path),
        params=dict(DEFAULT_PARAMS),
        steps=_disabled_steps(),
        max_workers=1,
    )

    assert processed == ["patient_b"]
