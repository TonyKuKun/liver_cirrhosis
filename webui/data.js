// Demo data for the PortaFlow workbench prototype.
// Numbers are drawn from the real project outputs:
//   PVP_predictor/runs/final_20260610_pvp_l2_shunt/{summary.json,oof_predictions.csv}
// so the prototype reads like the actual study rather than placeholder noise.

export const MODEL_METRICS = {
  run: "final_20260610_pvp_l2_shunt",
  nSamples: 72,
  folds: 5,
  seed: 40,
  mae: 2.685,
  maeStd: 0.746,
  rmse: 3.605,
  r2: 0.643,
  oofMae: 2.704,
  oofRmse: 3.8,
  bias: 0.256,
  foldMae: [3.687, 1.884, 1.752, 3.026, 3.078],
  bestBaseline: { name: "physics / adaboost", mae: 3.42 },
};

// 8-vessel branch order used by the model (PVP_predictor/dataset.py SEGMENTS).
export const SEGMENTS = ["mpv", "sv", "smv", "lpv", "rpv", "tips", "lgv", "pgv"];

export const VESSEL_META = {
  mpv:  { label: "MPV",  full: "门静脉主干 · Main portal vein",        baseR: 6.4 },
  sv:   { label: "SV",   full: "脾静脉 · Splenic vein",               baseR: 4.6 },
  smv:  { label: "SMV",  full: "肠系膜上静脉 · Sup. mesenteric vein", baseR: 5.0 },
  lpv:  { label: "LPV",  full: "门静脉左支 · Left portal vein",        baseR: 3.8 },
  rpv:  { label: "RPV",  full: "门静脉右支 · Right portal vein",       baseR: 4.0 },
  tips: { label: "TIPS", full: "经颈静脉肝内门体分流 · Stent",        baseR: 5.0 },
  lgv:  { label: "LGV",  full: "胃左/冠状静脉 · Left gastric vein",    baseR: 2.6 },
  pgv:  { label: "PGV",  full: "门静脉旁/附脐侧支 · Para-portal vein", baseR: 2.4 },
};

// Junction topology (PVP_predictor/dataset.py JUNCTIONS), drives the schematic graph.
export const JUNCTIONS = {
  inflow: { children: ["sv", "smv"] },
  confluence_outflow: { children: ["mpv", "lgv", "pgv"] },
  bifurcation: { parent: "mpv", children: ["lpv", "rpv", "tips"] },
};

// 3D-ish anatomical layout for the vessel tree (mm-scale, +y is cranial).
// Each branch is a poly-line from a proximal to distal point.
export const VESSEL_PATHS = {
  smv: [[6, -58, -6], [3, -28, -3], [0, 0, 0]],
  sv:  [[-52, 6, 9], [-26, 3, 5], [0, 0, 0]],
  mpv: [[0, 0, 0], [4, 18, 0], [9, 36, 0]],
  rpv: [[9, 36, 0], [28, 48, -6], [47, 56, -9]],
  lpv: [[9, 36, 0], [-10, 52, 8], [-27, 62, 13]],
  tips: [[9, 36, 0], [26, 60, 4], [40, 86, 6]],
  lgv: [[0, 0, 0], [-16, 18, 16], [-31, 31, 21]],
  pgv: [[0, 0, 0], [-7, 16, 20], [-12, 27, 26]],
};

// Geometry feature channels (PVP_predictor/dataset.py per-point channels + summary scalars).
export const GEOMETRY_FIELDS = [
  { key: "area",        label: "截面积 Area",            unit: "mm²" },
  { key: "hdiam",       label: "水力直径 4A/P",          unit: "mm" },
  { key: "rinsc",       label: "内切半径",               unit: "mm" },
  { key: "curvature",   label: "曲率",                   unit: "1/mm" },
  { key: "solidity",    label: "实心度",                 unit: "" },
  { key: "circularity", label: "圆形度",                 unit: "" },
  { key: "dads",        label: "归一化 dA/ds",           unit: "" },
  { key: "arclen",      label: "弧长",                   unit: "mm" },
];

// Patient roster — real names/labels/preds + flow fractions from oof_predictions.csv.
// q_* are the model's learned flow-split proxies; we reuse them for the branch table.
export const PATIENTS = [
  mk("P-019", "LiJinFeng",   31.63, 31.69, 0, { lgv: 0, pgv: 1, rpv: 1 }, 1, { mpv: 0.30, lpv: 0.70, rpv: 0.78, tips: 0.0,  coll: 0.00 }),
  mk("P-024", "XieFengE",    38.25, 35.11, 0, { lgv: 1, pgv: 1, rpv: 1 }, 2, { mpv: 0.23, lpv: 0.04, rpv: 0.55, tips: 0.0,  coll: 0.74 }),
  mk("P-031", "JinJunTing",  36.04, 36.05, 0, { lgv: 1, pgv: 1, rpv: 1 }, 2, { mpv: 0.82, lpv: 0.13, rpv: 0.61, tips: 0.0,  coll: 0.04 }),
  mk("P-033", "JiZhangKui",  33.10, 35.31, 0, { lgv: 1, pgv: 1, rpv: 1 }, 1, { mpv: 0.59, lpv: 0.55, rpv: 0.66, tips: 0.0,  coll: 0.41 }),
  mk("P-040", "LiHuaMin",    30.89, 35.46, 0, { lgv: 1, pgv: 1, rpv: 1 }, 1, { mpv: 0.91, lpv: 0.07, rpv: 0.72, tips: 0.0,  coll: 0.01 }),
  mk("P-052", "JiaXiuLian",  27.95, 29.47, 0, { lgv: 0, pgv: 0, rpv: 1 }, 0, { mpv: 0.54, lpv: 0.46, rpv: 0.61, tips: 0.0,  coll: 0.00 }),
  mk("P-058", "ZhaoSuCai",   25.01, 30.71, 0, { lgv: 0, pgv: 1, rpv: 1 }, 0, { mpv: 0.96, lpv: 0.04, rpv: 0.50, tips: 0.0,  coll: 0.00 }),
  mk("P-007", "WuJinHeng",   20.60, 26.00, 0, { lgv: 0, pgv: 1, rpv: 1 }, 1, { mpv: 0.81, lpv: 0.19, rpv: 0.55, tips: 0.0,  coll: 0.00 }),
  mk("P-008", "WuJinHeng#",  16.92, 25.65, 1, { lgv: 0, pgv: 1, rpv: 1 }, 1, { mpv: 0.71, lpv: 0.14, rpv: 0.42, tips: 0.15, coll: 0.00 }),
  mk("P-026", "XieFengE#",   24.27, 17.32, 1, { lgv: 0, pgv: 1, rpv: 1 }, 1, { mpv: 0.03, lpv: 0.02, rpv: 0.03, tips: 0.96, coll: 0.00 }),
];

function mk(id, name, label, pred, postTips, has, pvt, flow) {
  const present = new Set(SEGMENTS);
  if (!has.lgv) present.delete("lgv");
  if (!has.pgv) present.delete("pgv");
  if (!has.rpv) present.delete("rpv");
  if (!postTips) present.delete("tips");
  return {
    id, name, label, pred, postTips,
    pvtSeverity: pvt,
    err: +(pred - label).toFixed(2),
    present,
    flow,
    geometry: synthGeometry(present, label),
  };
}

// Build a plausible per-branch geometry table seeded by the patient's pressure,
// so values move sensibly across patients without pretending to be measured data.
function synthGeometry(present, label) {
  const out = {};
  const s = (label - 16) / 24; // 0..~1 severity
  for (const seg of SEGMENTS) {
    if (!present.has(seg)) continue;
    const r = VESSEL_META[seg].baseR;
    const dilate = 1 + 0.22 * s + jitter(seg) * 0.12;
    const hdiam = +(2 * r * dilate).toFixed(2);
    const area = +(Math.PI * Math.pow(hdiam / 2, 2)).toFixed(1);
    out[seg] = {
      area,
      hdiam,
      rinsc: +((hdiam / 2) * (0.86 - jitter(seg, 3) * 0.12)).toFixed(2),
      curvature: +(0.012 + jitter(seg, 7) * 0.03).toFixed(3),
      solidity: +(0.97 - jitter(seg, 11) * 0.06).toFixed(3),
      circularity: +(0.93 - jitter(seg, 13) * 0.18).toFixed(3),
      dads: +((jitter(seg, 17) - 0.5) * 0.4).toFixed(3),
      arclen: +(40 + jitter(seg, 19) * 60).toFixed(1),
    };
  }
  return out;
}

function jitter(seg, salt = 1) {
  let h = salt;
  for (const c of seg) h = (h * 131 + c.charCodeAt(0)) % 9973;
  return (h % 1000) / 1000;
}
