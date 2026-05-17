"""Tabular feature extraction for traditional PVP baselines.

The deep model consumes branch-wise point profiles.  Baseline models need a
fixed-width table, so this module converts each patient into geometry,
physics-inspired, and auxiliary/system scalar features.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from dataset import (
    AUX_KEYS,
    P_AREA,
    P_CIRC,
    P_CURV,
    P_HDIAM,
    P_INSC,
    P_NCOMP,
    P_PERIM,
    P_RRAT,
    P_SOLID,
    P_TORS,
    PROFILE_KEYS,
    SEGMENTS,
    SEG_INDEX,
)


STATS = ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90")


@dataclass
class FeatureTable:
    X: np.ndarray
    y: np.ndarray
    sample_names: List[str]
    feature_names: List[str]
    groups: Dict[str, List[int]]
    metadata: List[Dict[str, object]]
    dropped_all_nan: List[str]


def subject_id_from_name(name: str) -> str:
    """Remove leading date and TIPS suffix so paired samples share an id."""
    core = re.sub(r"^\d+", "", str(name))
    return core.split("#", 1)[0]


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> Dict[str, float]:
    values = _finite(values)
    if values.size == 0:
        return {k: np.nan for k in STATS}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def _segment_values(item: Mapping[str, object], seg_idx: int, feat_idx: int) -> np.ndarray:
    profiles = np.asarray(item["profiles"], dtype=np.float64)
    valid = np.asarray(item["point_valid"], dtype=np.float64)
    segment_mask = np.asarray(item["segment_mask"], dtype=np.float64)
    if segment_mask[seg_idx] <= 0.5:
        return np.asarray([], dtype=np.float64)
    mask = valid[seg_idx] > 0.5
    return profiles[seg_idx, :, feat_idx][mask]


def _valid_profile(item: Mapping[str, object], seg_idx: int) -> tuple[np.ndarray, np.ndarray]:
    profiles = np.asarray(item["profiles"], dtype=np.float64)
    valid = np.asarray(item["point_valid"], dtype=np.float64)
    segment_mask = np.asarray(item["segment_mask"], dtype=np.float64)
    if segment_mask[seg_idx] <= 0.5:
        return profiles[seg_idx, :0, :], np.asarray([], dtype=bool)
    mask = valid[seg_idx] > 0.5
    return profiles[seg_idx, mask, :], mask


def _safe_lengths_mm(arc: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arc = np.asarray(arc, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if arc.size == 0 or not np.any(valid_mask):
        return np.asarray([], dtype=np.float64)
    arc_valid = arc[valid_mask]
    if arc_valid.size == 1:
        return np.asarray([1.0], dtype=np.float64)
    ds = np.diff(arc_valid, prepend=arc_valid[0])
    if ds.size > 1:
        ds[0] = ds[1]
    ds = np.where(np.isfinite(ds), ds, 0.0)
    return np.clip(ds, 1e-6, None)


def _effective_radius_mm(profile: np.ndarray) -> np.ndarray:
    hdiam = np.clip(profile[:, P_HDIAM], 1e-6, None)
    inscribed = np.clip(profile[:, P_INSC], 1e-6, None)
    solidity = np.clip(profile[:, P_SOLID], 0.01, 1.0)
    alpha = np.clip(1.0 - solidity, 0.0, 1.0)
    return (1.0 - alpha) * (0.5 * hdiam) + alpha * inscribed


def _branch_resistance_proxy(item: Mapping[str, object], seg_idx: int) -> float:
    profile, mask = _valid_profile(item, seg_idx)
    if profile.size == 0:
        return np.nan
    arc = np.asarray(item["arc_lengths"], dtype=np.float64)[seg_idx]
    ds = _safe_lengths_mm(arc, mask)
    r_eff = _effective_radius_mm(profile)
    n = min(ds.size, r_eff.size)
    if n == 0:
        return np.nan
    return float(np.sum(ds[:n] / np.clip(r_eff[:n], 1e-6, None) ** 4))


def _append(
    row: List[float],
    names: List[str],
    groups: Dict[str, List[int]],
    group: str,
    name: str,
    value: float,
) -> None:
    groups.setdefault(group, []).append(len(row))
    names.append(name)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = np.nan
    row.append(value if np.isfinite(value) else np.nan)


def _diameter_stat(item: Mapping[str, object], seg_name: str, stat: str = "p50") -> float:
    values = _segment_values(item, SEG_INDEX[seg_name], P_HDIAM)
    return _stats(values)[stat]


def _murray_deviation(parent_diam: float, child_diams: Iterable[float]) -> float:
    if not np.isfinite(parent_diam) or parent_diam <= 0:
        return np.nan
    child_power = 0.0
    n = 0
    for d in child_diams:
        if np.isfinite(d) and d > 0:
            child_power += d ** 3
            n += 1
    if n == 0:
        return np.nan
    return float((child_power - parent_diam ** 3) / (parent_diam ** 3 + 1e-12))


def _add_geometry_features(
    item: Mapping[str, object],
    row: List[float],
    names: List[str],
    groups: Dict[str, List[int]],
) -> None:
    segment_mask = np.asarray(item["segment_mask"], dtype=np.float64)
    point_valid = np.asarray(item["point_valid"], dtype=np.float64)
    arc_lengths = np.asarray(item["arc_lengths"], dtype=np.float64)
    endpoints = np.asarray(item.get("endpoints_3d", np.zeros((len(SEGMENTS), 2, 3))), dtype=np.float64)

    _append(row, names, groups, "geometry", "geom__n_present_segments", np.sum(segment_mask > 0.5))

    for si, seg in enumerate(SEGMENTS):
        present = float(segment_mask[si] > 0.5)
        valid = point_valid[si] > 0.5
        _append(row, names, groups, "geometry", f"geom__{seg}__present", present)
        _append(row, names, groups, "geometry", f"geom__{seg}__valid_fraction", np.mean(valid))

        if present and np.any(valid):
            arc_valid = arc_lengths[si][valid]
            length = np.nanmax(arc_valid) - np.nanmin(arc_valid)
            chord = np.linalg.norm(endpoints[si, 1] - endpoints[si, 0])
            chord = chord if chord > 1e-6 else np.nan
        else:
            length = np.nan
            chord = np.nan
        _append(row, names, groups, "geometry", f"geom__{seg}__length_mm", length)
        _append(row, names, groups, "geometry", f"geom__{seg}__path_chord_ratio", _safe_ratio(length, chord))

        for fi, key in enumerate(PROFILE_KEYS):
            for stat, value in _stats(_segment_values(item, si, fi)).items():
                _append(row, names, groups, "geometry", f"geom__{seg}__{key}__{stat}", value)

        profile, _ = _valid_profile(item, si)
        if profile.size:
            r_eff = _effective_radius_mm(profile)
            area = profile[:, P_AREA]
            hdiam = profile[:, P_HDIAM]
            curvature = np.abs(profile[:, P_CURV])
            torsion = np.abs(profile[:, P_TORS])
            dads = np.abs(profile[:, 8])
            _append(row, names, groups, "geometry", f"geom__{seg}__r_eff_mm__mean", np.mean(r_eff))
            _append(row, names, groups, "geometry", f"geom__{seg}__r_eff_mm__min", np.min(r_eff))
            _append(row, names, groups, "geometry", f"geom__{seg}__area_min_to_median", _safe_ratio(np.min(area), np.median(area)))
            _append(row, names, groups, "geometry", f"geom__{seg}__hdiam_min_to_max", _safe_ratio(np.min(hdiam), np.max(hdiam)))
            _append(row, names, groups, "geometry", f"geom__{seg}__area_p10_to_p90", _safe_ratio(np.percentile(area, 10), np.percentile(area, 90)))
            _append(row, names, groups, "geometry", f"geom__{seg}__curv_torsion_energy", np.mean(curvature ** 2 + torsion ** 2))
            _append(row, names, groups, "geometry", f"geom__{seg}__mean_abs_dA_ds", np.mean(dads))
            _append(row, names, groups, "geometry", f"geom__{seg}__noncircular_burden", np.mean(1.0 - np.clip(profile[:, P_CIRC], 0.0, 1.0)))
            _append(row, names, groups, "geometry", f"geom__{seg}__fragmentation_burden", np.mean(np.clip(profile[:, P_NCOMP] - 1.0, 0.0, None)))
        else:
            for suffix in [
                "r_eff_mm__mean",
                "r_eff_mm__min",
                "area_min_to_median",
                "hdiam_min_to_max",
                "area_p10_to_p90",
                "curv_torsion_energy",
                "mean_abs_dA_ds",
                "noncircular_burden",
                "fragmentation_burden",
            ]:
                _append(row, names, groups, "geometry", f"geom__{seg}__{suffix}", np.nan)


def _add_physics_features(
    item: Mapping[str, object],
    row: List[float],
    names: List[str],
    groups: Dict[str, List[int]],
) -> None:
    resistance = {seg: _branch_resistance_proxy(item, SEG_INDEX[seg]) for seg in SEGMENTS}
    for seg in SEGMENTS:
        value = resistance[seg]
        _append(row, names, groups, "physics", f"phys__{seg}__resistance_proxy", value)
        _append(row, names, groups, "physics", f"phys__{seg}__log_resistance_proxy", np.log1p(value) if np.isfinite(value) else np.nan)

    mpv = resistance["mpv"]
    lpv = resistance["lpv"]
    rpv = resistance["rpv"]
    tips = resistance["tips"]
    lgv = resistance["lgv"]
    pgv = resistance["pgv"]
    sv = resistance["sv"]
    smv = resistance["smv"]

    liver_terms = [r for r in (lpv, rpv) if np.isfinite(r)]
    liver_path_mean = mpv + float(np.mean(liver_terms)) if np.isfinite(mpv) and liver_terms else np.nan
    liver_parallel = 1.0 / (1.0 / lpv + 1.0 / rpv) if np.isfinite(lpv) and np.isfinite(rpv) and lpv > 0 and rpv > 0 else np.nan
    inflow_parallel = 1.0 / (1.0 / sv + 1.0 / smv) if np.isfinite(sv) and np.isfinite(smv) and sv > 0 and smv > 0 else np.nan
    collateral_parallel = np.nan
    collat_terms = [r for r in (lgv, pgv) if np.isfinite(r) and r > 0]
    if collat_terms:
        collateral_parallel = 1.0 / sum(1.0 / r for r in collat_terms)

    _append(row, names, groups, "physics", "phys__path_mpv_liver_mean_resistance", liver_path_mean)
    _append(row, names, groups, "physics", "phys__path_mpv_lpv_resistance", mpv + lpv if np.isfinite(mpv) and np.isfinite(lpv) else np.nan)
    _append(row, names, groups, "physics", "phys__path_mpv_rpv_resistance", mpv + rpv if np.isfinite(mpv) and np.isfinite(rpv) else np.nan)
    _append(row, names, groups, "physics", "phys__liver_parallel_resistance", liver_parallel)
    _append(row, names, groups, "physics", "phys__inflow_parallel_resistance", inflow_parallel)
    _append(row, names, groups, "physics", "phys__collateral_parallel_resistance", collateral_parallel)
    _append(row, names, groups, "physics", "phys__tips_to_liver_resistance_ratio", _safe_ratio(tips, liver_parallel))
    _append(row, names, groups, "physics", "phys__collateral_to_liver_resistance_ratio", _safe_ratio(collateral_parallel, liver_parallel))
    _append(row, names, groups, "physics", "phys__sv_smv_resistance_ratio", _safe_ratio(sv, smv))
    _append(row, names, groups, "physics", "phys__lpv_rpv_resistance_ratio", _safe_ratio(lpv, rpv))

    diam = {seg: _diameter_stat(item, seg) for seg in SEGMENTS}
    _append(row, names, groups, "physics", "phys__murray_inflow_dev", _murray_deviation(diam["mpv"], [diam["sv"], diam["smv"]]))
    _append(row, names, groups, "physics", "phys__murray_bifurc_dev", _murray_deviation(diam["mpv"], [diam["lpv"], diam["rpv"]]))
    _append(row, names, groups, "physics", "phys__murray_bifurc_with_tips_dev", _murray_deviation(diam["mpv"], [diam["lpv"], diam["rpv"], diam["tips"]]))

    conductances = []
    for seg in ("lpv", "rpv", "tips"):
        r = resistance[seg]
        conductances.append(1.0 / r if np.isfinite(r) and r > 0 else 0.0)
    total_c = sum(conductances)
    for seg, c in zip(("lpv", "rpv", "tips"), conductances):
        _append(row, names, groups, "physics", f"phys__conductance_fraction__{seg}", c / total_c if total_c > 0 else np.nan)


def _add_aux_features(
    item: Mapping[str, object],
    row: List[float],
    names: List[str],
    groups: Dict[str, List[int]],
    extra_keys: Sequence[str],
) -> None:
    aux = np.asarray(item["aux_scalars"], dtype=np.float64)
    aux_mask = np.asarray(item["aux_mask"], dtype=np.float64)
    for i, key in enumerate(AUX_KEYS):
        value = aux[i] if i < aux.size and aux_mask[i] > 0.5 else np.nan
        _append(row, names, groups, "aux", f"aux__{key}", value)
        _append(row, names, groups, "aux", f"aux__{key}__present", float(i < aux_mask.size and aux_mask[i] > 0.5))

    extras = item.get("extras_for_eval", {}) or {}
    for key in extra_keys:
        _append(row, names, groups, "aux", f"extra__{key}", extras.get(key, np.nan))


def _metadata(item: Mapping[str, object]) -> Dict[str, object]:
    name = str(item["name"])
    aux = np.asarray(item["aux_scalars"], dtype=np.float64)
    segment_mask = np.asarray(item["segment_mask"], dtype=np.float64)

    def aux_value(key: str, default: float = 0.0) -> float:
        try:
            return float(aux[AUX_KEYS.index(key)])
        except (ValueError, IndexError):
            return default

    return {
        "name": name,
        "subject_id": subject_id_from_name(name),
        "post_tips": int(bool(item.get("is_post_tips", "#" in name))),
        "has_lgv": int(aux_value("has_lgv") > 0.5),
        "has_pgv": int(aux_value("has_pgv") > 0.5),
        "has_rpv": int(segment_mask[SEG_INDEX["rpv"]] > 0.5),
        "pvt_severity": int(round(aux_value("pvt_severity_grade", 0.0))),
    }


def _drop_all_nan_columns(table: FeatureTable) -> FeatureTable:
    if table.X.size == 0:
        return table
    keep = ~np.all(~np.isfinite(table.X), axis=0)
    dropped = [name for name, keep_col in zip(table.feature_names, keep) if not keep_col]
    index_map = {}
    new_names = []
    for old_idx, keep_col in enumerate(keep):
        if keep_col:
            index_map[old_idx] = len(new_names)
            new_names.append(table.feature_names[old_idx])
    new_groups = {
        group: [index_map[i] for i in idxs if i in index_map]
        for group, idxs in table.groups.items()
    }
    return FeatureTable(
        X=table.X[:, keep],
        y=table.y,
        sample_names=table.sample_names,
        feature_names=new_names,
        groups=new_groups,
        metadata=table.metadata,
        dropped_all_nan=dropped,
    )


def build_feature_table_from_records(
    records: Sequence[Mapping[str, object]],
    drop_all_nan: bool = True,
) -> FeatureTable:
    rows: List[List[float]] = []
    sample_names: List[str] = []
    labels: List[float] = []
    metadata: List[Dict[str, object]] = []
    feature_names: List[str] | None = None
    groups_template: Dict[str, List[int]] | None = None
    extra_keys = sorted({
        str(key)
        for item in records
        for key in (item.get("extras_for_eval", {}) or {}).keys()
    })

    for item in records:
        row: List[float] = []
        names: List[str] = []
        groups: Dict[str, List[int]] = {}
        _add_geometry_features(item, row, names, groups)
        _add_physics_features(item, row, names, groups)
        _add_aux_features(item, row, names, groups, extra_keys)

        if feature_names is None:
            feature_names = names
            groups_template = groups
        elif feature_names != names:
            raise ValueError("Feature schema changed between records")

        rows.append(row)
        sample_names.append(str(item["name"]))
        labels.append(float(item["label"]))
        metadata.append(_metadata(item))

    table = FeatureTable(
        X=np.asarray(rows, dtype=np.float64),
        y=np.asarray(labels, dtype=np.float64),
        sample_names=sample_names,
        feature_names=feature_names or [],
        groups=groups_template or {},
        metadata=metadata,
        dropped_all_nan=[],
    )
    return _drop_all_nan_columns(table) if drop_all_nan else table


def build_feature_table(dataset, drop_all_nan: bool = True) -> FeatureTable:
    return build_feature_table_from_records(dataset.data, drop_all_nan=drop_all_nan)


def indices_for_feature_set(table: FeatureTable, feature_set: str) -> List[int]:
    feature_set = feature_set.lower()
    if feature_set == "combined":
        groups = ("geometry", "physics", "aux")
    elif feature_set in table.groups:
        groups = (feature_set,)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    indices: List[int] = []
    for group in groups:
        indices.extend(table.groups.get(group, []))
    return sorted(set(indices))


def feature_schema(table: FeatureTable) -> Dict[str, object]:
    return {
        "n_samples": int(table.X.shape[0]),
        "n_features": int(table.X.shape[1]),
        "groups": {k: len(v) for k, v in table.groups.items()},
        "feature_names": table.feature_names,
        "dropped_all_nan": table.dropped_all_nan,
    }
