import {
  MODEL_METRICS, SEGMENTS, VESSEL_META, VESSEL_PATHS,
  GEOMETRY_FIELDS, PATIENTS,
} from "./data.js";

/* ============================ state ============================ */
const state = {
  stage: 1,
  patient: null,
  layers: { pretrain: false, predict: true, smooth: false },
  tasks: [],
  completed: {},        // patientId -> Set(stages)
  running: false,
};
const $ = (id) => document.getElementById(id);

/* pressure colormap (blue -> red), domain fixed to a clinical mmHg window */
const PRESS_MIN = 10, PRESS_MAX = 40, PRESS_THRESH = 22;
const CMAP = ["#2f6df6", "#22b8d8", "#3fbf73", "#f0c419", "#f08a26", "#e23b34"];
function pressColor(p) {
  const t = clamp((p - PRESS_MIN) / (PRESS_MAX - PRESS_MIN), 0, 1) * (CMAP.length - 1);
  const i = Math.floor(t), f = t - i;
  return i >= CMAP.length - 1 ? CMAP[CMAP.length - 1] : lerpHex(CMAP[i], CMAP[i + 1], f);
}
const STAGE_TITLES = { 1: "3D 重建视图 · 分割", 2: "几何分析 · 中心线与截面", 3: "压力映射 · 门静脉树" };

/* per-branch endpoint pressures (proximal→distal in VESSEL_PATHS order) */
function branchPressures(base) {
  return {
    smv: [base + 2.4, base + 0.6], sv: [base + 3.0, base + 0.7],
    mpv: [base + 0.5, base + 0.0], rpv: [base, base - 2.2], lpv: [base, base - 2.6],
    tips: [base, base * 0.36], lgv: [base + 0.6, base + 4.2], pgv: [base + 0.5, base + 3.1],
  };
}

/* ============================ boot ============================ */
function init() {
  buildStepper();
  renderPatients(PATIENTS);
  $("patientCount").textContent = PATIENTS.length;
  $("patientSearch").addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    renderPatients(PATIENTS.filter((p) => (p.name + p.id).toLowerCase().includes(q)));
  });
  selectPatient(PATIENTS[0]);
  initViz();
}

/* ============================ stepper ============================ */
function buildStepper() {
  $("stepper").querySelectorAll(".step").forEach((b) =>
    b.addEventListener("click", () => setStage(+b.dataset.stage)));
}
function setStage(stage) {
  state.stage = stage;
  const done = state.patient ? state.completed[state.patient.id] || new Set() : new Set();
  $("stepper").querySelectorAll(".step").forEach((b) => {
    const s = +b.dataset.stage;
    b.classList.toggle("active", s === stage);
    b.classList.toggle("done", done.has(s) && s !== stage);
  });
  $("vpTitle").textContent = STAGE_TITLES[stage];
  $("mprStrip").hidden = stage !== 1;
  $("legend").hidden = stage !== 3;
  buildTools();
  buildRightPanel();
  buildFoot();
  rebuildScene();
}

/* ============================ patients ============================ */
function renderPatients(list) {
  const ul = $("patientList");
  ul.innerHTML = "";
  for (const p of list) {
    const li = document.createElement("li");
    li.className = "patient-card" + (state.patient?.id === p.id ? " active" : "");
    li.innerHTML = `
      <div class="pc-name">${p.name.replace("#", "")}</div>
      <div class="pc-pvp" style="background:${pressColor(p.pred)}">${p.pred.toFixed(1)}<small> mmHg</small></div>
      <div class="pc-id">${p.id}</div>
      <div class="pc-tags">
        <span class="tag ${p.postTips ? "tips" : ""}">${p.postTips ? "post-TIPS" : "pre-TIPS"}</span>
        ${p.pvtSeverity ? `<span class="tag pvt">PVT ${p.pvtSeverity}</span>` : ""}
      </div>`;
    li.addEventListener("click", () => selectPatient(p));
    ul.appendChild(li);
  }
}
function selectPatient(p) {
  state.patient = p;
  if (!state.completed[p.id]) state.completed[p.id] = new Set();
  $("patientChip").classList.add("set");
  $("patientChipName").textContent = `${p.name.replace("#", "")} · ${p.id}`;
  renderPatients(PATIENTS);
  setStage(state.stage);
  drawMpr();
}

/* ============================ viewport tools ============================ */
function buildTools() {
  const host = $("vpTools");
  if (state.stage === 1) {
    const L = [["predict", "Predict", "#f97316"], ["smooth", "Smooth", "#2dd4bf"], ["pretrain", "Pretrain", "#7c8da0"]];
    host.innerHTML = L.map(([k, lbl, c]) =>
      `<button class="chip-btn ${state.layers[k] ? "on" : ""}" data-layer="${k}">
        <span class="swatch" style="background:${c}"></span>${lbl}</button>`).join("");
    host.querySelectorAll("[data-layer]").forEach((b) => b.addEventListener("click", () => {
      const k = b.dataset.layer; state.layers[k] = !state.layers[k];
      if (!state.layers.predict && !state.layers.smooth && !state.layers.pretrain) state.layers[k] = true;
      buildTools(); rebuildScene();
    }));
  } else if (state.stage === 2) {
    host.innerHTML = `
      <button class="chip-btn on" data-t="centerline"><span class="swatch" style="background:#facc15"></span>中心线</button>
      <button class="chip-btn on" data-t="planes"><span class="swatch" style="background:#2dd4bf"></span>截面</button>
      <button class="chip-btn on" data-t="seeds"><span class="swatch" style="background:#ef5a5a"></span>种子点</button>`;
    host.querySelectorAll("[data-t]").forEach((b) => b.addEventListener("click", () => { b.classList.toggle("on"); rebuildScene(); }));
  } else {
    host.innerHTML = `<button class="chip-btn on">伪彩压力</button>
      <button class="chip-btn" id="rotBtn">⟳ 自动旋转</button>`;
    $("rotBtn") && $("rotBtn").addEventListener("click", (e) => { viz.autorotate = !viz.autorotate; e.target.classList.toggle("on", viz.autorotate); });
  }
}
function toolOn(t) { const b = $("vpTools").querySelector(`[data-t="${t}"]`); return b ? b.classList.contains("on") : true; }

/* ============================ viewport footer ============================ */
function buildFoot() {
  const p = state.patient; if (!p) return;
  const f = $("vpFoot");
  if (state.stage === 1) {
    const present = [...p.present].length;
    f.innerHTML = `网格 <b>predict_smooth.stl</b> · 顶点 <b>~48k</b> · 三角面 <b>~96k</b> · 检出血管段 <b>${present}/8</b> · QC <b style="color:var(--good)">通过</b>`;
  } else if (state.stage === 2) {
    f.innerHTML = `中心线采样点 <b>${[...p.present].length * 64}</b> · 截面 profile 通道 <b>11</b> · 系统级标量 <b>52</b> · 解剖连接 <b>3 处汇合</b>`;
  } else {
    f.innerHTML = `预测 PVP <b style="color:${pressColor(p.pred)}">${p.pred.toFixed(2)} mmHg</b> · 金标准 <b>${p.label.toFixed(2)}</b> · 误差 <b>${(p.pred - p.label).toFixed(2)}</b> · 模型 <b>${MODEL_METRICS.run}</b>`;
  }
}

/* ============================ right panel ============================ */
function buildRightPanel() {
  const host = $("rightPanel");
  if (!state.patient) { host.innerHTML = ""; return; }
  host.innerHTML = state.stage === 1 ? panelSeg() : state.stage === 2 ? panelGeom() : panelPred();
  wirePanel();
}

function panelSeg() {
  const p = state.patient;
  return `
  <div class="section">
    <h3>分割流水线 <span class="h-line"></span></h3>
    <div class="field"><label>数据来源</label>
      <input value="patient/${p.name} · orig.nii.gz" readonly></div>
    <div class="field field-row">
      <div style="flex:1"><label>分割模型</label>
        <select><option>nnVnet (refinement)</option><option>VKAN</option></select></div>
      <div style="width:108px"><label>网格上限</label>
        <select><option>80k faces</option><option>120k</option><option>40k</option></select></div>
    </div>
    <button class="btn-run" id="runBtn">▶ 运行 CT → STL 流水线</button>
    <button class="btn-ghost" id="dlBtn">下载 predict_smooth.stl</button>
    <div class="runlog" id="runlog"></div>
  </div>
  <div class="section">
    <h3>网格质检 <span class="h-line"></span></h3>
    <div class="kv"><span class="k">连通分量</span><span class="v">1</span></div>
    <div class="kv"><span class="k">非流形边</span><span class="v" style="color:var(--good)">0</span></div>
    <div class="kv"><span class="k">自交三角面</span><span class="v" style="color:var(--good)">0</span></div>
    <div class="kv"><span class="k">平滑迭代</span><span class="v">8</span></div>
    <div class="kv"><span class="k">文件大小</span><span class="v">${(6 + [...p.present].length).toFixed(0)},204 KB</span></div>
    <p class="note">由 <code>VKAN_segementation/pipeline.py</code> 驱动：TotalSegmentator → 确定性 <code>pretrain.stl</code> → nnVnet 精修 → 平滑 + 确定性质检。</p>
  </div>`;
}

function panelGeom() {
  const p = state.patient;
  const rows = SEGMENTS.filter((s) => p.present.has(s)).map((s) => {
    const g = p.geometry[s], c = vesselColor(s);
    return `<tr>
      <td class="vname"><span class="vdot" style="background:${c}"></span>${VESSEL_META[s].label}</td>
      <td>${g.area.toFixed(0)}</td><td>${g.hdiam.toFixed(1)}</td>
      <td>${g.rinsc.toFixed(1)}</td><td>${g.circularity.toFixed(2)}</td>
      <td>${g.solidity.toFixed(2)}</td></tr>`;
  }).join("");
  return `
  <div class="section">
    <h3>几何特征提取 <span class="h-line"></span></h3>
    <p class="note" style="margin-top:0">沿中心线对每段血管采样横截面，提取 11 个 pointwise 通道与 52 个系统级标量。下表为各段的代表性聚合值。</p>
    <button class="btn-run" id="runBtn">▶ 提取中心线与几何特征</button>
    <div class="runlog" id="runlog"></div>
  </div>
  <div class="section">
    <h3>8-血管几何摘要 <span class="h-line"></span></h3>
    <table class="gtable">
      <thead><tr><th>血管</th><th>Area<br>mm²</th><th>4A/P<br>mm</th><th>r_insc<br>mm</th><th>圆形</th><th>实心</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>
  <div class="section">
    <h3>解剖连接 <span class="h-line"></span></h3>
    <div class="kv"><span class="k">流入汇合</span><span class="v">SV + SMV → 汇合点</span></div>
    <div class="kv"><span class="k">汇合流出</span><span class="v">MPV · LGV · PGV</span></div>
    <div class="kv"><span class="k">门静脉分叉</span><span class="v">MPV → LPV · RPV${p.postTips ? " · TIPS" : ""}</span></div>
    <p class="note">连接关系驱动 <code>FlowGraphRefiner</code> 的图消息传播。</p>
  </div>`;
}

function panelPred() {
  const p = state.patient;
  const band = pressureBand(p.pred);
  const conf = clamp(1 - Math.abs(p.pred - p.label) / 12, 0.45, 0.97);
  const bp = branchPressures(p.pred);
  const branchRows = SEGMENTS.filter((s) => p.present.has(s)).map((s) => {
    const pr = (bp[s][0] + bp[s][1]) / 2;
    const flow = flowFor(s, p);
    return `<tr>
      <td class="vname"><span class="vdot" style="background:${pressColor(pr)}"></span>${VESSEL_META[s].label}</td>
      <td class="flow">${pr.toFixed(1)}</td>
      <td><div class="flowbar"><i style="width:${Math.round(flow * 100)}%;background:${vesselColor(s)}"></i></div></td>
      <td>${(flow * 100).toFixed(0)}%</td></tr>`;
  }).join("");
  return `
  <div class="section">
    <h3>门静脉压力预测 <span class="h-line"></span></h3>
    <div class="pvp-hero">
      <div class="pvp-value" style="color:${pressColor(p.pred)}"><span class="num">${p.pred.toFixed(1)}</span><span class="unit">mmHg</span></div>
      <div class="pvp-band" style="color:${band.c};background:${band.bg};border:1px solid ${band.c}55">● ${band.label}</div>
      <div class="pvp-sub">金标准 <b>${p.label.toFixed(1)}</b> · 误差 <b>${(p.pred - p.label).toFixed(2)}</b> mmHg</div>
    </div>
    <div style="margin-top:16px">
      <div style="font-size:11px;color:var(--ink-1);margin-bottom:4px">预测置信度</div>
      <div class="conf-row"><div class="conf-track"><div class="conf-fill" style="width:${(conf * 100).toFixed(0)}%"></div></div>
        <span class="conf-val">${(conf * 100).toFixed(0)}%</span></div>
    </div>
    <button class="btn-run" id="runBtn" style="margin-top:16px">▶ 运行 PVP 推理</button>
    <div class="runlog" id="runlog"></div>
  </div>
  <div class="section">
    <h3>分支压力与流量分配 <span class="h-line"></span></h3>
    <table class="gtable">
      <thead><tr><th>血管</th><th>压力<br>mmHg</th><th>流量占比</th><th></th></tr></thead>
      <tbody>${branchRows}</tbody>
    </table>
    ${p.postTips ? `<p class="note">post-TIPS：支架内压力显著衰减，分流分数 <code>${(p.flow.tips * 100).toFixed(0)}%</code>。</p>` : ""}
  </div>
  <div class="section">
    <h3>模型性能 (5-fold, n=${MODEL_METRICS.nSamples}) <span class="h-line"></span></h3>
    <div class="metric-grid">
      <div class="metric"><div class="m-k">MAE</div><div class="m-v">${MODEL_METRICS.mae.toFixed(2)}<small> ±${MODEL_METRICS.maeStd.toFixed(2)}</small></div></div>
      <div class="metric"><div class="m-k">RMSE</div><div class="m-v">${MODEL_METRICS.rmse.toFixed(2)}</div></div>
      <div class="metric"><div class="m-k">R²</div><div class="m-v">${MODEL_METRICS.r2.toFixed(3)}</div></div>
      <div class="metric"><div class="m-k">最佳 baseline</div><div class="m-v">${MODEL_METRICS.bestBaseline.mae.toFixed(2)}<small> MAE</small></div></div>
    </div>
    <p class="note">深度模型相比最佳手工特征 baseline（${MODEL_METRICS.bestBaseline.name}）降低 MAE <code>${(MODEL_METRICS.bestBaseline.mae - MODEL_METRICS.mae).toFixed(2)}</code>。结构主干：物理代理 + GlobalFlowCorrector + FlowGraphRefiner + 单 PVP 头。</p>
  </div>`;
}

function pressureBand(v) {
  if (v < 18) return { label: "轻度门脉高压", c: "#138a55", bg: "#e4f4eb" };
  if (v < 26) return { label: "中度门脉高压", c: "#b97606", bg: "#fbf2dd" };
  if (v < 32) return { label: "显著门脉高压", c: "#cf6f1c", bg: "#fbeede" };
  return { label: "重度门脉高压", c: "#cf3a32", bg: "#fbe6e4" };
}
function flowFor(s, p) {
  const m = { mpv: p.flow.mpv, lpv: p.flow.lpv, rpv: p.flow.rpv, tips: p.flow.tips,
    lgv: p.flow.coll, pgv: p.flow.coll * 0.6, sv: 0.42, smv: 0.55 };
  return clamp(m[s] ?? 0.3, 0.02, 1);
}

function wirePanel() {
  const run = $("runBtn"); if (run) run.addEventListener("click", runPipeline);
  const dl = $("dlBtn"); if (dl) dl.addEventListener("click", () => addTask("下载结果", state.patient.name, "done"));
}

/* ============================ pipeline simulation ============================ */
const PIPELINES = {
  1: ["TotalSegmentator 器官提取", "生成 pretrain.stl", "nnVnet 精修", "平滑 + 质检"],
  2: ["中心线提取 (VMTK)", "横截面重采样", "11 通道几何特征", "52 系统级标量"],
  3: ["特征装配 (8 血管)", "LearnablePhysicsProxy", "GlobalFlowCorrector", "FlowGraphRefiner → PVP 头"],
};
function runPipeline() {
  if (state.running) return;
  state.running = true;
  const steps = PIPELINES[state.stage];
  const log = $("runlog");
  log.innerHTML = steps.map((s, i) => `<div class="run-step" data-i="${i}"><span class="rs-dot"></span>${s}</div>`).join("");
  const btn = $("runBtn"); if (btn) { btn.disabled = true; btn.textContent = "运行中…"; }
  const taskName = ["分割 CT→STL", "几何特征提取", "PVP 推理"][state.stage - 1];
  const task = addTask(taskName, state.patient.name, "run");
  let i = 0;
  const tick = () => {
    const els = log.querySelectorAll(".run-step");
    if (i > 0) els[i - 1].className = "run-step done";
    if (i < steps.length) {
      els[i].className = "run-step run";
      i++; setTimeout(tick, 520 + Math.random() * 280);
    } else {
      state.running = false;
      if (btn) { btn.disabled = false; btn.textContent = state.stage === 3 ? "▶ 运行 PVP 推理" : "▶ 重新运行"; }
      state.completed[state.patient.id].add(state.stage);
      task.status = "done"; renderTasks();
      setStage(state.stage);
    }
  };
  tick();
}
function addTask(name, patient, status) {
  const t = { name, patient, status, time: new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) };
  state.tasks.unshift(t); renderTasks(); return t;
}
function renderTasks() {
  const ul = $("taskList");
  if (!state.tasks.length) { ul.innerHTML = `<li class="task-empty">尚无运行任务</li>`; return; }
  ul.innerHTML = state.tasks.slice(0, 12).map((t) => `
    <li class="task-item">
      <div class="task-ic ${t.status}">${t.status === "done" ? "✓" : "…"}</div>
      <div class="task-main"><div class="task-name">${t.name}</div>
        <div class="task-meta">${t.patient} · ${t.time}</div></div>
    </li>`).join("");
}

/* ============================ MPR thumbnails ============================ */
function drawMpr() {
  $("mprStrip").querySelectorAll("canvas").forEach((cv, idx) => {
    const ctx = cv.getContext("2d"); const W = cv.width = 264, H = cv.height = 264;
    const img = ctx.createImageData(W, H), d = img.data;
    const cx = W / 2 + (idx - 1) * 16, cy = H / 2;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const k = (y * W + x) * 4;
      const r = Math.hypot(x - W / 2, y - H / 2) / (W / 2);
      let v = 22 + 40 * (1 - r) + (Math.random() * 26 - 13);
      // abdomen soft-tissue ring
      if (r > 0.86) v *= 0.25;
      // vessel cross-sections (bright)
      const dv = Math.hypot(x - cx, y - cy);
      if (dv < 16) v += 150 * Math.exp(-(dv * dv) / 120);
      const dv2 = Math.hypot(x - cx - 34, y - cy + 20);
      if (dv2 < 10) v += 110 * Math.exp(-(dv2 * dv2) / 60);
      v = clamp(v, 0, 255);
      d[k] = v * 0.92; d[k + 1] = v * 0.98; d[k + 2] = v; d[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  });
}

/* ============================ Three.js 3D ============================ */
const viz = { ready: false, autorotate: true, three: null, renderer: null, scene: null, camera: null, group: null, sph: { r: 175, theta: 0.7, phi: 1.15 }, dragging: false };

async function initViz() {
  try {
    const THREE = await import("three");
    viz.three = THREE;
    const host = $("canvasHost");
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    host.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 2000);
    scene.add(new THREE.AmbientLight(0x8da0b8, 0.85));
    const d1 = new THREE.DirectionalLight(0xffffff, 1.1); d1.position.set(60, 90, 80); scene.add(d1);
    const d2 = new THREE.DirectionalLight(0x4fd1c5, 0.5); d2.position.set(-80, -20, -40); scene.add(d2);
    const grp = new THREE.Group(); scene.add(grp);
    Object.assign(viz, { renderer, scene, camera, group: grp, ready: true });

    new ResizeObserver(resizeViz).observe(host);
    resizeViz();
    bindOrbit(renderer.domElement);
    rebuildScene();
    (function loop() {
      requestAnimationFrame(loop);
      if (viz.autorotate && !viz.dragging) viz.sph.theta += 0.0017;
      updateCamera();
      renderer.render(scene, camera);
    })();
  } catch (e) {
    console.warn("WebGL/three unavailable, SVG fallback", e);
    buildSvgFallback();
  }
}
function resizeViz() {
  if (!viz.ready) return;
  const host = $("canvasHost"); const w = host.clientWidth, h = host.clientHeight;
  viz.renderer.setSize(w, h, false); viz.camera.aspect = w / h; viz.camera.updateProjectionMatrix();
}
function updateCamera() {
  const { r, theta, phi } = viz.sph;
  viz.camera.position.set(r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.cos(theta));
  viz.camera.lookAt(0, 0, 0);
}
function bindOrbit(el) {
  let last = null;
  el.addEventListener("pointerdown", (e) => { viz.dragging = true; last = [e.clientX, e.clientY]; el.setPointerCapture(e.pointerId); });
  el.addEventListener("pointermove", (e) => {
    if (!viz.dragging) return;
    viz.sph.theta -= (e.clientX - last[0]) * 0.006;
    viz.sph.phi = clamp(viz.sph.phi - (e.clientY - last[1]) * 0.006, 0.2, 2.95);
    last = [e.clientX, e.clientY];
  });
  const stop = () => { viz.dragging = false; };
  el.addEventListener("pointerup", stop); el.addEventListener("pointerleave", stop);
  el.addEventListener("wheel", (e) => { e.preventDefault(); viz.sph.r = clamp(viz.sph.r * (1 + e.deltaY * 0.0011), 90, 360); }, { passive: false });
}

function rebuildScene() {
  if (!viz.ready) { buildSvgFallback(); return; }
  const THREE = viz.three, grp = viz.group, p = state.patient;
  while (grp.children.length) grp.remove(grp.children[0]);
  if (!p) return;
  $("vpHint").textContent = state.stage === 1 ? "拖拽旋转 · 滚轮缩放" : state.stage === 2 ? "黄=中心线 · 青环=采样截面" : "颜色编码局部门静脉压";

  const present = SEGMENTS.filter((s) => p.present.has(s));
  const bp = branchPressures(p.pred);
  const flat = stage1Color();

  for (const seg of present) {
    const pts = VESSEL_PATHS[seg].map((q) => new THREE.Vector3(q[0], q[1] - 14, q[2] - 8));
    const curve = new THREE.CatmullRomCurve3(pts);
    const radius = VESSEL_META[seg].baseR * (state.stage === 1 && state.layers.pretrain && !state.layers.predict && !state.layers.smooth ? 1.16 : 1);
    const geo = new THREE.TubeGeometry(curve, 48, radius, 16, false);
    let mat;
    if (state.stage === 3) {
      const uv = geo.attributes.uv, pos = geo.attributes.position;
      const colors = new Float32Array(pos.count * 3);
      for (let i = 0; i < pos.count; i++) {
        const t = uv.getX(i);
        const pr = bp[seg][0] * (1 - t) + bp[seg][1] * t;
        const [r, g, b] = hexRgb(pressColor(pr));
        colors[i * 3] = r; colors[i * 3 + 1] = g; colors[i * 3 + 2] = b;
      }
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.45, metalness: 0.05 });
    } else {
      const col = state.stage === 2 ? 0x2dd4bf : flat.color;
      mat = new THREE.MeshStandardMaterial({ color: col, roughness: flat.rough, metalness: 0.04,
        transparent: flat.opacity < 1, opacity: flat.opacity });
    }
    grp.add(new THREE.Mesh(geo, mat));

    if (state.stage === 2) {
      if (toolOn("centerline")) {
        const cl = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(60)),
          new THREE.LineBasicMaterial({ color: 0xfacc15 }));
        grp.add(cl);
      }
      if (toolOn("planes")) {
        for (const t of [0.25, 0.55, 0.85]) {
          const c = curve.getPointAt(t), tan = curve.getTangentAt(t);
          const ring = new THREE.Mesh(new THREE.TorusGeometry(radius * 1.5, 0.5, 8, 24),
            new THREE.MeshStandardMaterial({ color: 0x2dd4bf, emissive: 0x0d9488, emissiveIntensity: 0.4 }));
          ring.position.copy(c); ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), tan.normalize());
          grp.add(ring);
        }
      }
      if (toolOn("seeds")) {
        const s0 = new THREE.Mesh(new THREE.SphereGeometry(2.4, 16, 16),
          new THREE.MeshStandardMaterial({ color: 0xef5a5a, emissive: 0x661111 }));
        s0.position.copy(curve.getPointAt(0)); grp.add(s0);
      }
    }
  }
  // confluence node marker
  const node = new THREE.Mesh(new viz.three.SphereGeometry(2, 16, 16),
    new viz.three.MeshStandardMaterial({ color: state.stage === 3 ? 0xffffff : 0x9fb2c6, emissive: 0x223344 }));
  node.position.set(0, -14, -8); grp.add(node);
}
function stage1Color() {
  if (state.layers.smooth) return { color: 0x2dd4bf, rough: 0.4, opacity: 1 };
  if (state.layers.predict) return { color: 0xf97316, rough: 0.55, opacity: 1 };
  return { color: 0x7c8da0, rough: 0.85, opacity: 0.92 };
}

/* SVG fallback when WebGL/CDN is unavailable — keeps pseudocolor + topology */
function buildSvgFallback() {
  const fb = $("canvasFallback"); fb.hidden = false; $("canvasHost").querySelector("canvas")?.remove();
  const p = state.patient; if (!p) { fb.innerHTML = ""; return; }
  const bp = branchPressures(p.pred);
  const proj = ([x, y, z]) => [380 + x * 4.4 + z * 1.2, 470 - y * 4.4 - z * 0.6];
  let paths = "";
  for (const seg of SEGMENTS) {
    if (!p.present.has(seg)) continue;
    const pts = VESSEL_PATHS[seg].map(proj);
    const d = pts.map((q, i) => (i ? "L" : "M") + q[0].toFixed(0) + " " + q[1].toFixed(0)).join(" ");
    const col = state.stage === 3 ? pressColor((bp[seg][0] + bp[seg][1]) / 2)
      : state.stage === 2 ? "#2dd4bf" : "#f97316";
    paths += `<path d="${d}" stroke="${col}" stroke-width="${VESSEL_META[seg].baseR * 1.7}" fill="none" stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>`;
  }
  fb.innerHTML = `<svg viewBox="0 0 760 760" style="width:100%;height:100%">
    <g style="filter:drop-shadow(0 0 6px rgba(0,0,0,.6))">${paths}</g>
    <circle cx="380" cy="470" r="5" fill="#fff"/></svg>`;
}

/* ============================ helpers ============================ */
function vesselColor(s) {
  return { mpv: "#f97316", sv: "#22d3ee", smv: "#38bdf8", lpv: "#a78bfa", rpv: "#34d399",
    tips: "#facc15", lgv: "#f0a3c0", pgv: "#fb923c" }[s] || "#2dd4bf";
}
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function hexRgb(h) { const n = parseInt(h.slice(1), 16); return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }
function lerpHex(a, b, t) {
  const A = parseInt(a.slice(1), 16), B = parseInt(b.slice(1), 16);
  const r = Math.round(((A >> 16) & 255) + (((B >> 16) & 255) - ((A >> 16) & 255)) * t);
  const g = Math.round(((A >> 8) & 255) + (((B >> 8) & 255) - ((A >> 8) & 255)) * t);
  const bl = Math.round((A & 255) + ((B & 255) - (A & 255)) * t);
  return `rgb(${r},${g},${bl})`;
}

/* set legend threshold marker text */
$("legThresh").textContent = `⌃ ${PRESS_THRESH}`;
$("legLow").textContent = PRESS_MIN; $("legHigh").textContent = PRESS_MAX;

/* boot — invoked last so all module-scope consts (viz, etc.) are initialized */
init();
