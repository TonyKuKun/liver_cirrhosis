import json
import importlib.util
from pathlib import Path

import pytest

import integrated_web_frontend as web


def test_discover_patients_accepts_expected_inputs(tmp_path):
    (tmp_path / "case_dcm" / "dcm").mkdir(parents=True)
    (tmp_path / "case_orig").mkdir()
    (tmp_path / "case_orig" / "orig.nii.gz").write_bytes(b"nii")
    (tmp_path / "case_stl").mkdir()
    (tmp_path / "case_stl" / "vessel.stl").write_bytes(b"solid")
    (tmp_path / "case_bad!").mkdir()
    (tmp_path / "case_bad!" / "vessel.stl").write_bytes(b"solid")

    patients = web._discover_patients(tmp_path)
    assert [item["id"] for item in patients] == ["case_dcm", "case_orig", "case_stl"]


def test_patient_record_reads_label_metadata(tmp_path):
    patient = tmp_path / "case"
    (patient / "dcm").mkdir(parents=True)
    label = patient / "label"
    label.mkdir()
    (label / "age.txt").write_text("51\n", encoding="utf-8")
    (label / "sex.txt").write_text("female\n", encoding="utf-8")
    (label / "symptoms.txt").write_text("abdominal_distension\n", encoding="utf-8")

    record = web._patient_record(patient)

    assert record["label_meta"] == {
        "age": "51",
        "sex": "female",
        "symptoms": "abdominal_distension",
    }


def test_clinical_context_reads_surgery_timing_and_pvp(tmp_path):
    patient = tmp_path / "20201224Case#"
    label = patient / "label"
    label.mkdir(parents=True)
    (label / "surgery_date.txt").write_text("2020-12-22\n", encoding="utf-8")
    (label / "PVP.txt").write_text("19.12\n", encoding="utf-8")

    clinical = web._clinical_context(patient, tmp_path)

    assert clinical["exam_date"] == "2020-12-24"
    assert clinical["surgery_date"] == "2020-12-22"
    assert clinical["days_from_surgery"] == 2
    assert clinical["timing"] == "术后第 2 天"
    assert clinical["measured_pvp"] == 19.12


def test_session_from_payload_recreates_missing_session(tmp_path):
    patient = tmp_path / "case"
    patient.mkdir()
    (patient / "unified_features.json").write_text("{}", encoding="utf-8")

    session = web._session_from_payload({
        "session_id": "missing-session",
        "root_folder": str(tmp_path),
    })

    assert session["root"] == str(tmp_path.resolve())
    assert [item["id"] for item in session["patients"]] == ["case"]


def test_stage_state_reports_ready_and_done(tmp_path):
    patient = tmp_path / "case"
    patient.mkdir()
    (patient / "vessel.stl").write_bytes(b"solid")

    feature_state = web._stage_state(patient, "features", None)
    assert feature_state["status"] == "ready"

    (patient / "unified_features.json").write_text("{}", encoding="utf-8")
    feature_state = web._stage_state(patient, "features", None)
    assert feature_state["status"] == "done"


def test_pvp_stage_accepts_text_prediction_output(tmp_path):
    patient = tmp_path / "case"
    patient.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (patient / "unified_features.json").write_text("{}", encoding="utf-8")
    (patient / "PVP_predict.txt").write_text("pvp_mean_mmHg: 18.50\n", encoding="utf-8")

    pvp_state = web._stage_state(patient, "pvp", model)

    assert pvp_state["status"] == "done"
    assert pvp_state["outputs"]["PVP_predict.txt"]["exists"]


def test_pvp_dataset_loads_without_label(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in this Python environment")
    from PVP_predictor.dataset import PortalVeinDataset

    patient = tmp_path / "case"
    patient.mkdir()
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    unified = {
        "_meta": {"patient_id": "case", "is_post_tips": False},
        "vessel_presence": {"mpv": {"present": True}},
        "pointwise": {
            "mpv": {
                "area": values,
                "hydraulic_diameter": values,
                "perimeter": values,
                "curvature": [0.01] * len(values),
                "torsion": [0.0] * len(values),
                "inscribed_radius": [5.0] * len(values),
                "solidity": [1.0] * len(values),
                "r_insc_to_r_eq_ratio": [1.0] * len(values),
                "dA_ds_norm": [0.0] * len(values),
                "circularity": [0.9] * len(values),
                "n_components": [1.0] * len(values),
            }
        },
        "statistical": {},
        "system": {},
        "global": {},
    }
    (patient / "unified_features.json").write_text(json.dumps(unified), encoding="utf-8")

    ds = PortalVeinDataset(str(tmp_path), n_points=8, require_labels=False, verbose=False)
    assert len(ds) == 1
    item = ds[0]
    assert item["name"] == "case"
    assert float(item["label_present"]) == 0.0
