import json
import importlib.util
from pathlib import Path

import pytest

from web import server as web


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


def test_patient_metadata_merges_patient_info_and_label_directory(tmp_path):
    patient = tmp_path / "case"
    label = patient / "label"
    label.mkdir(parents=True)
    (patient / "patient_info.txt").write_text(
        "年龄：62\n性别: 男\n检查日期=2021-09-09\n手术日期\t2021-09-17\n",
        encoding="utf-8",
    )
    (label / "原发病.txt").write_text("肝硬化\n", encoding="utf-8")
    (label / "并发症.txt").write_text("腹水\n", encoding="utf-8")

    meta = web._patient_label_metadata(patient)

    assert meta == {
        "age": "62",
        "sex": "男",
        "exam_date": "2021-09-09",
        "surgery_date": "2021-09-17",
        "primary_disease": "肝硬化",
        "symptoms": "腹水",
    }


def test_clinical_context_prefers_patient_info_exam_date(tmp_path):
    patient = tmp_path / "case"
    patient.mkdir()
    (patient / "patient_info.txt").write_text(
        "检查日期: 2021/09/09\n手术日期: 2021/09/17\n",
        encoding="utf-8",
    )

    clinical = web._clinical_context(patient, tmp_path)

    assert clinical["exam_date"] == "2021-09-09"
    assert clinical["surgery_date"] == "2021-09-17"
    assert clinical["days_from_surgery"] == -8
    assert clinical["timing"] == "术前 8 天"


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
    features = patient / "features"
    features.mkdir(parents=True)
    (features / "unified_features.json").write_text("{}", encoding="utf-8")

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

    features = patient / "features"
    features.mkdir()
    (features / "unified_features.json").write_text("{}", encoding="utf-8")
    feature_state = web._stage_state(patient, "features", None)
    assert feature_state["status"] == "done"


def test_feature_status_only_reads_patient_features_folder(tmp_path):
    patient = tmp_path / "case"
    output_dir = patient / "segmentation"
    output_dir.mkdir(parents=True)
    (output_dir / "portal_vein.stl").write_bytes(b"solid")
    (output_dir / "unified_features.json").write_text(json.dumps({
        "global": {"total_centerline_length": 999.0},
    }), encoding="utf-8")

    legacy_status = web._patient_status(patient, None)
    assert legacy_status["stages"]["features"]["status"] == "ready"
    assert not legacy_status["stages"]["features"]["done"]
    assert legacy_status["features_summary"] == {}

    features = patient / "features"
    features.mkdir()
    (features / "unified_features.json").write_text(json.dumps({
        "statistical": {
            "mpv": {"length": 12.5, "mean_diameter": 4.2},
        },
        "system": {"available": {"angle_sv_smv": 73.0}},
        "global": {"total_centerline_length": 12.5},
    }), encoding="utf-8")

    status = web._patient_status(patient, None)

    assert status["stages"]["features"]["status"] == "done"
    assert status["features_summary"]["segments"][0]["length"] == 12.5
    assert status["features_summary"]["key_metrics"] == {
        "total_centerline_length": 12.5,
        "angle_sv_smv": 73.0,
    }
    assert status["stages"]["features"]["outputs"]["unified_features.json"]["path"] == str(
        features / "unified_features.json"
    )


def test_feature_summary_reads_available_system_metrics():
    summary = web._feature_summary({
        "system": {
            "available": {
                "angle_sv_smv": 91.2,
                "confluence_murray3_deviation": 0.31,
                "inflow_resistance_asymmetry": -0.18,
            }
        },
        "global": {
            "total_centerline_length": 321.0,
            "sv_smv_diameter_ratio": 0.82,
            "sv_smv_angle": 90.5,
        },
    })

    assert summary["key_metrics"]["total_centerline_length"] == 321.0
    assert summary["key_metrics"]["sv_smv_diameter_ratio"] == 0.82
    assert summary["key_metrics"]["sv_smv_angle"] == 90.5
    assert summary["key_metrics"]["angle_sv_smv"] == 91.2
    assert summary["key_metrics"]["confluence_murray3_deviation"] == 0.31
    assert summary["key_metrics"]["inflow_resistance_asymmetry"] == -0.18


def test_pvp_stage_accepts_text_prediction_output(tmp_path):
    patient = tmp_path / "case"
    features = patient / "features"
    features.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (features / "unified_features.json").write_text("{}", encoding="utf-8")
    (patient / "PVP_predict.txt").write_text("pvp_mean_mmHg: 18.50\n", encoding="utf-8")

    pvp_state = web._stage_state(patient, "pvp", model)

    assert pvp_state["status"] == "done"
    assert pvp_state["outputs"]["PVP_predict.txt"]["exists"]


def test_pvp_dataset_loads_without_label(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in this Python environment")
    from PVP_predictor.dataset import PortalVeinDataset

    patient = tmp_path / "case"
    features = patient / "features"
    features.mkdir(parents=True)
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
            }
        },
        "statistical": {},
        "system": {},
        "global": {},
    }
    (features / "unified_features.json").write_text(json.dumps(unified), encoding="utf-8")

    ds = PortalVeinDataset(str(tmp_path), n_points=8, require_labels=False, verbose=False)
    assert len(ds) == 1
    item = ds[0]
    assert item["name"] == "case"
    assert float(item["label_present"]) == 0.0
