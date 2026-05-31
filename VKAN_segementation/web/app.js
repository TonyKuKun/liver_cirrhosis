const STEPS = [
  ["totalseg", "TotalSegmentator 器官"],
  ["pretrain", "生成 pretrain"],
  ["refine", "VKAN refinement"],
];

const DEFAULT_MODELS = {
  pretrain: { label: "Pretrain", file: "pretrain.stl", color: "#7c8da0" },
  predict: { label: "Predict", file: "predict.stl", color: "#f97316" },
  smooth: { label: "Smooth", file: "predict_smooth.stl", color: "#10b981" },
  manual: { label: "Manual edit", file: "manual_edit.stl", color: "#0f766e" },
  vessel: { label: "Manual label", file: "vessel.stl", color: "#2563eb" },
};

const MASK_COLORS = ["#f97316", "#10b981", "#2563eb", "#e11d48", "#8b5cf6", "#14b8a6", "#f59e0b", "#64748b"];
const AXES = ["coronal", "axial", "sagittal"];
const AXIS_CANVAS = { axial: "ctAxial", coronal: "ctCoronal", sagittal: "ctSagittal" };

const state = {
  session: null,
  patients: [],
  currentPatient: null,
  modelMeshes: DEFAULT_MODELS,
  activeModels: new Set(["predict"]),
  activeOrgans: new Set(),
  activeMasks: new Set(),
  meshCache: new Map(),
  maskCatalog: [],
  ctSlices: {},
  ctViews: {
    axial: { zoom: 1, panX: 0, panY: 0, window: null, level: null },
    coronal: { zoom: 1, panX: 0, panY: 0, window: null, level: null },
    sagittal: { zoom: 1, panX: 0, panY: 0, window: null, level: null },
  },
  crosshair: null,
  job: null,
  pollTimer: null,
  pickedPoint: null,
  painting: false,
  windowing: false,
  panning: false,
  lastPointer: null,
  paintBusy: false,
  lastPaintAt: 0,
};

const $ = (id) => document.getElementById(id);

function init() {
  buildStepButtons();
  bindEvents();
  updateCtMode();
  checkHealth();
}

function bindEvents() {
  $("loadBtn").addEventListener("click", createSession);
  $("nextPatientBtn").addEventListener("click", goNextPatient);
  $("refreshBtn").addEventListener("click", refreshData);
  $("runAllBtn").addEventListener("click", () => runSteps(STEPS.map(([key]) => key), true));
  $("downloadBtn").addEventListener("click", downloadResults);
  $("toggleSettingsBtn").addEventListener("click", () => $("settingsPanel").classList.toggle("hidden"));
  $("patientSelect").addEventListener("change", () => {
    resetPatientViewState();
    refreshData();
  });
  $("showCtPanel").addEventListener("change", updateCtMode);
  $("meshOpacity").addEventListener("input", renderScene);
  $("maxFaces").addEventListener("change", () => {
    state.meshCache.clear();
    loadActiveMeshes();
  });
  document.querySelectorAll("[data-axis]").forEach((input) => {
    input.addEventListener("input", () => {
      updateCrosshairFromAxis(input.dataset.axis, Number(input.value));
      loadAllCtSlices();
    });
  });
  document.querySelectorAll("[data-axis-canvas]").forEach((canvas) => {
    canvas.addEventListener("pointerdown", (event) => handleCtPointer(event, canvas.dataset.axisCanvas, true));
    canvas.addEventListener("pointermove", (event) => handleCtPointer(event, canvas.dataset.axisCanvas, false));
    canvas.addEventListener("pointerup", () => endCtGesture());
    canvas.addEventListener("pointerleave", () => endCtGesture());
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("wheel", (event) => handleCtWheel(event, canvas.dataset.axisCanvas), { passive: false });
  });
  $("deleteBrushBtn").addEventListener("click", () => applyBrush("delete"));
  $("keepBrushBtn").addEventListener("click", () => applyBrush("keep"));
  $("newMaskBtn").addEventListener("click", createMask);
  $("thresholdMaskBtn").addEventListener("click", applyThresholdMask);
  $("regionGrowBtn").addEventListener("click", applyRegionGrow);
  $("applyBooleanBtn").addEventListener("click", applyBooleanMask);
  $("smoothMaskBtn").addEventListener("click", applySmoothMask);
  $("fillMaskBtn").addEventListener("click", applyFillMask);
  $("closeToolTray").addEventListener("click", () => showToolPanel(null));
  document.querySelectorAll("[data-tool]").forEach((button) => {
    button.addEventListener("click", () => showToolPanel(button.dataset.tool));
  });
}

function resetPatientViewState() {
  state.meshCache.clear();
  state.activeMasks.clear();
  state.crosshair = null;
  state.pickedPoint = null;
  state.ctSlices = {};
  Object.values(state.ctViews).forEach((view) => {
    view.zoom = 1;
    view.panX = 0;
    view.panY = 0;
    view.window = null;
    view.level = null;
    view.last = null;
  });
}

async function goNextPatient() {
  const select = $("patientSelect");
  const count = select.options.length;
  if (!state.session || count < 2) return;
  select.selectedIndex = (select.selectedIndex + 1) % count;
  resetPatientViewState();
  await refreshData();
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const envName = data.runtime?.conda_env || "base";
    $("serverState").textContent = data.ok ? `本地服务已连接 · ${envName}` : "服务异常";
  } catch (err) {
    $("serverState").textContent = "服务未连接";
  }
}

function buildStepButtons() {
  const wrap = $("stepButtons");
  wrap.innerHTML = "";
  STEPS.forEach(([key, label], idx) => {
    const btn = document.createElement("button");
    btn.className = "step-button";
    btn.type = "button";
    btn.innerHTML = `
      <span class="step-index">${idx + 1}</span>
      <span>${label}</span>
      <span class="step-state" data-step-state="${key}">待运行</span>
    `;
    btn.addEventListener("click", () => runSteps([key], false));
    wrap.appendChild(btn);
  });
}

function showToolPanel(name) {
  const selected = name ? document.querySelector(`[data-tool="${name}"]`) : null;
  const open = Boolean(selected && !selected.classList.contains("active"));
  document.querySelectorAll("[data-tool]").forEach((button) => {
    button.classList.toggle("active", Boolean(open && button.dataset.tool === name));
  });
  document.querySelectorAll("[data-tool-panel]").forEach((panel) => {
    panel.classList.toggle("hidden", !(open && panel.dataset.toolPanel === name));
  });
  $("toolTray").classList.toggle("hidden", !open);
}

function updateCtMode() {
  const enabled = $("showCtPanel").checked;
  $("viewGrid").classList.toggle("ct-enabled", enabled);
  $("viewGrid").classList.toggle("ct-disabled", !enabled);
  if (enabled) loadAllCtSlices();
  renderScene();
}

async function createSession() {
  clearJob();
  setBusy(true);
  try {
    const root = $("rootFolder").value.trim();
    if (!root) throw new Error("请输入 patient 文件夹或批量根目录");
    const res = await fetchJson("/api/session", { root_folder: root });
    const payload = await readResponse(res);
    state.session = payload.session;
    state.crosshair = null;
    $("checkpointPath").value = state.session.default_checkpoint || "";
    logLine(`Session ${state.session.id} loaded.`);
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function refreshData() {
  if (!state.session) return;
  try {
    const res = await fetch(`/api/session/${state.session.id}/data`);
    const data = await readResponse(res);
    state.session = data.session;
    state.patients = data.patients || [];
    state.modelMeshes = data.model_meshes || DEFAULT_MODELS;
    populatePatients();
    syncCurrentPatient();
    syncMasksFromPatient();
    buildModelLayers();
    buildOrganLayers();
    buildMaskLayers();
    renderInspector();
    await loadActiveMeshes();
    if ($("showCtPanel").checked) loadAllCtSlices();
  } catch (err) {
    showError(err);
  }
}

function populatePatients() {
  const select = $("patientSelect");
  const old = select.value;
  select.innerHTML = "";
  state.patients.forEach((patient) => {
    const opt = document.createElement("option");
    opt.value = patient.id;
    opt.textContent = patient.id;
    select.appendChild(opt);
  });
  if (state.patients.some((p) => p.id === old)) {
    select.value = old;
  } else if (state.patients[0]) {
    select.value = state.patients[0].id;
  }
  $("nextPatientBtn").disabled = state.patients.length < 2;
}

function syncCurrentPatient() {
  const id = $("patientSelect").value;
  state.currentPatient = state.patients.find((p) => p.id === id) || state.patients[0] || null;
  if (state.currentPatient) {
    const status = state.currentPatient.status || {};
    const organCount = status.organs?.length || 0;
    $("patientSummary").textContent = `${state.currentPatient.id} · organs ${organCount} · ${status.folder || ""}`;
    if (!state.crosshair && state.ctSlices.axial?.shape) {
      const shape = state.ctSlices.axial.shape;
      state.crosshair = [Math.floor(shape[0] / 2), Math.floor(shape[1] / 2), Math.floor(shape[2] / 2)];
    }
  } else {
    $("patientSummary").textContent = "尚未载入";
  }
}

function syncMasksFromPatient() {
  const masks = state.currentPatient?.status?.masks || [];
  state.maskCatalog = masks;
  const available = new Set(masks.map((m) => m.id));
  Array.from(state.activeMasks).forEach((id) => {
    if (!available.has(id)) state.activeMasks.delete(id);
  });
  if (!state.activeMasks.size) {
    const primary = masks.find((m) => m.id === "predict_mask.nii.gz") || masks.find((m) => m.id === "pretrain.nii.gz") || masks[0];
    if (primary) state.activeMasks.add(primary.id);
  }
  syncMaskSelects();
}

function syncMaskSelects() {
  const ids = ["newMaskSource", "thresholdTarget", "regionTarget", "editTarget", "maskA", "maskB", "maskBooleanTarget", "smoothTarget", "fillTarget"];
  ids.forEach((id) => {
    const select = $(id);
    const old = select.value;
    const allowsBlank = id === "newMaskSource";
    select.innerHTML = allowsBlank ? `<option value="">空白蒙版</option>` : "";
    state.maskCatalog.forEach((mask) => {
      const opt = document.createElement("option");
      opt.value = mask.id;
      opt.textContent = mask.label || mask.id;
      select.appendChild(opt);
    });
    const preferred = state.maskCatalog.find((m) => m.id === "predict_mask.nii.gz") || state.maskCatalog[0];
    if (allowsBlank && old === "") select.value = "";
    else if (state.maskCatalog.some((m) => m.id === old)) select.value = old;
    else if (preferred && !allowsBlank) select.value = preferred.id;
  });
}

function buildModelLayers() {
  const wrap = $("modelLayers");
  wrap.innerHTML = "";
  const files = state.currentPatient?.status?.files || {};
  Object.entries(state.modelMeshes).forEach(([key, info]) => {
    const fileKey = (info.file || "").replace(".stl", "").replace(".nii.gz", "").replace("/", "_");
    const exists = files[fileKey]?.exists || false;
    if (!exists && !["pretrain", "predict"].includes(key)) return;
    if (key === "vessel") return;
    const label = document.createElement("label");
    label.className = "layer-row";
    label.innerHTML = `
      <input type="checkbox" data-model="${key}" ${state.activeModels.has(key) ? "checked" : ""} ${exists ? "" : "disabled"} />
      <span class="swatch" style="background:${info.color || "#94a3b8"}"></span>
      <span>${info.label || key}</span>
      <span class="${exists ? "ok" : "warn"}">${exists ? "可用" : "缺失"}</span>
    `;
    label.querySelector("input").addEventListener("change", (event) => {
      if (event.target.checked) state.activeModels.add(key);
      else state.activeModels.delete(key);
      loadActiveMeshes();
    });
    wrap.appendChild(label);
  });
}

function buildOrganLayers() {
  const wrap = $("organLayers");
  wrap.innerHTML = "";
  const organs = state.currentPatient?.status?.organs || [];
  if (!organs.length) {
    wrap.textContent = "还没有 segmentation/*.stl，先运行 TotalSegmentator";
    return;
  }
  organs.forEach((organ) => {
    const key = `organ:${organ.name}`;
    const label = document.createElement("label");
    label.className = "organ-row";
    label.innerHTML = `
      <input type="checkbox" data-organ="${organ.name}" ${state.activeOrgans.has(organ.name) ? "checked" : ""} />
      <span class="swatch" style="background:${organ.color || "#94a3b8"}"></span>
      <span>${organ.label || organ.name}</span>
    `;
    label.querySelector("input").addEventListener("change", (event) => {
      if (event.target.checked) state.activeOrgans.add(organ.name);
      else state.activeOrgans.delete(organ.name);
      if (!event.target.checked) state.meshCache.delete(key);
      loadActiveMeshes();
    });
    wrap.appendChild(label);
  });
}

function buildMaskLayers() {
  const wrap = $("maskLayers");
  wrap.innerHTML = "";
  if (!state.maskCatalog.length) {
    wrap.textContent = "还没有可叠加的 NIfTI mask";
    syncMaskSelects();
    return;
  }
  state.maskCatalog.forEach((mask, idx) => {
    const color = mask.color || MASK_COLORS[idx % MASK_COLORS.length];
    const label = document.createElement("label");
    label.className = "mask-row";
    label.innerHTML = `
      <input type="checkbox" data-mask="${escapeHtml(mask.id)}" ${state.activeMasks.has(mask.id) ? "checked" : ""} />
      <span class="swatch" style="background:${color}"></span>
      <span>${escapeHtml(mask.label || mask.id)}</span>
      <span>${formatBytes(mask.size)}</span>
    `;
    label.querySelector("input").addEventListener("change", (event) => {
      if (event.target.checked) state.activeMasks.add(mask.id);
      else state.activeMasks.delete(mask.id);
      loadAllCtSlices();
    });
    wrap.appendChild(label);
  });
  syncMaskSelects();
}

async function loadActiveMeshes() {
  if (!state.session || !state.currentPatient) return;
  const meshNames = [
    ...Array.from(state.activeModels),
    ...Array.from(state.activeOrgans).map((name) => `organ:${name}`),
  ];
  const maxFaces = Number($("maxFaces").value || 70000);
  await Promise.allSettled(meshNames.map((name) => loadMesh(name, maxFaces)));
  renderScene();
}

async function loadMesh(meshName, maxFaces) {
  const cacheKey = `${state.currentPatient.id}:${meshName}:${maxFaces}`;
  if (state.meshCache.has(cacheKey)) return state.meshCache.get(cacheKey);
  try {
    const url = `/api/session/${state.session.id}/mesh?patient=${encodeURIComponent(state.currentPatient.id)}&mesh=${encodeURIComponent(meshName)}&max_faces=${maxFaces}`;
    const payload = await readResponse(await fetch(url));
    state.meshCache.set(cacheKey, payload);
    return payload;
  } catch (err) {
    logLine(`mesh ${meshName}: ${err.message}`);
    return null;
  }
}

function renderScene() {
  if (!window.Plotly) return;
  const traces = [];
  const maxFaces = Number($("maxFaces").value || 70000);
  const opacity = Number($("meshOpacity").value || 38) / 100;
  const active = [
    ...Array.from(state.activeModels),
    ...Array.from(state.activeOrgans).map((name) => `organ:${name}`),
  ];

  active.forEach((name) => {
    const item = state.meshCache.get(`${state.currentPatient?.id}:${name}:${maxFaces}`);
    if (!item?.mesh?.vertices?.length) return;
    const mesh = item.mesh;
    const style = item.style || {};
    const vertices = mesh.vertices;
    const faces = mesh.faces || [];
    traces.push({
      type: "mesh3d",
      name: `${style.label || name} (${mesh.n_faces_rendered || faces.length})`,
      x: vertices.map((v) => v[0]),
      y: vertices.map((v) => v[1]),
      z: vertices.map((v) => v[2]),
      i: faces.map((f) => f[0]),
      j: faces.map((f) => f[1]),
      k: faces.map((f) => f[2]),
      color: style.color || "#94a3b8",
      opacity: style.kind === "organ" ? Math.min(opacity, 0.32) : opacity,
      flatshading: false,
      lighting: { ambient: 0.55, diffuse: 0.8, specular: 0.12 },
      hovertemplate: `<b>${style.label || name}</b><br>x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<extra></extra>`,
    });
  });

  const markerPoint = state.pickedPoint || worldFromVoxel(state.crosshair);
  if (markerPoint) {
    traces.push({
      type: "scatter3d",
      mode: "markers",
      name: "CT cursor",
      x: [markerPoint[0]],
      y: [markerPoint[1]],
      z: [markerPoint[2]],
      marker: { size: 6, color: "#e31b23", symbol: "circle" },
      hovertemplate: "cursor<br>x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<extra></extra>",
      showlegend: false,
    });
  }

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#ffffff",
    scene: {
      aspectmode: "data",
      xaxis: axisLayout("X"),
      yaxis: axisLayout("Y"),
      zaxis: axisLayout("Z"),
      camera: { eye: { x: 1.55, y: 1.45, z: 1.05 }, up: { x: 0, y: 0, z: 1 } },
    },
    legend: {
      x: 0.01,
      y: 0.99,
      bgcolor: "rgba(255,255,255,0.82)",
      bordercolor: "#d6dee7",
      borderwidth: 1,
      font: { size: 11 },
    },
    uirevision: "vkan-workbench",
  };
  Plotly.react("viewer", traces, layout, { displaylogo: false, responsive: true, scrollZoom: true });
  $("emptyState").classList.toggle("hidden", traces.length > 0);
  const plot = $("viewer");
  if (typeof plot.removeAllListeners === "function") plot.removeAllListeners("plotly_click");
  plot.on("plotly_click", async (event) => {
    const pt = event.points?.[0];
    if (!pt) return;
    const x = Number(pt.x);
    const y = Number(pt.y);
    const z = Number(pt.z);
    if (![x, y, z].every(Number.isFinite)) return;
    state.pickedPoint = [x, y, z];
    const voxel = voxelFromWorld(state.pickedPoint);
    if (voxel) {
      state.crosshair = voxel;
      updateSlidersFromCrosshair();
      updateSelectionText();
      if ($("showCtPanel").checked) await loadAllCtSlices();
      renderScene();
      return;
    }
    const text = `3D x=${x.toFixed(2)}, y=${y.toFixed(2)}, z=${z.toFixed(2)}`;
    $("pickedPoint").textContent = text;
    $("selectionBox").textContent = text;
  });
}

function axisLayout(title) {
  return {
    title,
    backgroundcolor: "#ffffff",
    gridcolor: "#d9e1e8",
    zerolinecolor: "#c9d4df",
    showspikes: false,
  };
}

async function runSteps(steps, allPatients) {
  if (!state.session) {
    showError(new Error("请先载入病例"));
    return;
  }
  clearJob();
  setBusy(true);
  try {
    const selected = $("patientSelect").value;
    const payload = {
      session_id: state.session.id,
      patient_id: allPatients && state.patients.length > 1 ? "all" : selected,
      steps,
      force: $("forceRun").checked,
      checkpoint: $("checkpointPath").value.trim(),
      threshold: Number($("threshold").value || 0.5),
      smooth_iterations: Number($("smoothIterations").value || 8),
      device: $("device").value,
      fast: $("fastTotalSeg").checked,
    };
    const response = await readResponse(await fetchJson("/api/run", payload));
    state.job = response.job;
    startPolling();
  } catch (err) {
    setBusy(false);
    showError(err);
  }
}

function startPolling() {
  if (!state.job) return;
  pollJob();
  state.pollTimer = setInterval(pollJob, 1500);
}

async function pollJob() {
  if (!state.job) return;
  try {
    const payload = await readResponse(await fetch(`/api/job/${state.job.id}`));
    state.job = payload.job;
    renderJob();
    if (["done", "failed"].includes(state.job.status)) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      setBusy(false);
      state.meshCache.clear();
      await refreshData();
    }
  } catch (err) {
    logLine(`job polling failed: ${err.message}`);
  }
}

function renderJob() {
  const job = state.job;
  if (!job) return;
  const pct = job.total ? Math.round((job.completed / job.total) * 100) : 0;
  $("jobStatus").textContent = `${job.status} ${pct}% ${job.current || ""}`;
  $("jobStatus").style.color = job.status === "failed" ? "var(--danger)" : "var(--accent)";
  $("logs").textContent = (job.logs || []).join("\n\n");
  $("logs").scrollTop = $("logs").scrollHeight;

  const patientResults = state.currentPatient && job.results ? job.results[state.currentPatient.id] : null;
  STEPS.forEach(([key]) => {
    const el = document.querySelector(`[data-step-state="${key}"]`);
    if (!el) return;
    if (patientResults && key in patientResults) {
      el.textContent = patientResults[key] ? "完成" : "失败";
      el.style.color = patientResults[key] ? "var(--ok)" : "var(--danger)";
    }
  });
}

async function applyBrush(mode) {
  if (!state.session || !state.currentPatient) {
    showError(new Error("请先载入病例"));
    return;
  }
  if (!state.pickedPoint) {
    showError(new Error("请先在 3D 模型上点击一个点"));
    return;
  }
  setBusy(true);
  try {
    const payload = {
      session_id: state.session.id,
      patient_id: state.currentPatient.id,
      source: $("editSource").value,
      mode,
      radius: Number($("brushRadius").value || 6),
      point: state.pickedPoint,
    };
    const result = await readResponse(await fetchJson("/api/edit", payload));
    logLine(`[edit] ${mode}: removed ${result.edit.removed_faces}, kept ${result.edit.kept_faces}`);
    state.activeModels.add("manual");
    state.meshCache.clear();
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function createMask() {
  if (!state.session || !state.currentPatient) {
    showError(new Error("请先载入病例"));
    return;
  }
  setBusy(true);
  try {
    const payload = baseMaskPayload({
      name: $("newMaskName").value.trim() || "manual_mask",
      source: $("newMaskSource").value,
    });
    const result = await readResponse(await fetchJson("/api/mask/create", payload));
    logLine(`[mask] created ${result.mask.id}`);
    state.activeMasks.add(result.mask.id);
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function applyThresholdMask() {
  const target = $("thresholdTarget").value;
  if (!target) {
    showError(new Error("请先新建或选择目标 mask"));
    return;
  }
  setBusy(true);
  try {
    const payload = baseMaskPayload({
      target,
      lower: Number($("maskLower").value || -150),
      upper: Number($("maskUpper").value || 250),
      mode: $("thresholdWriteMode").value,
    });
    const result = await readResponse(await fetchJson("/api/mask/threshold", payload));
    logLine(`[mask] threshold ${result.mask.id}: ${result.mask.voxels} voxels`);
    state.activeMasks.add(result.mask.id);
    state.meshCache.clear();
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function applyRegionGrow() {
  const target = $("regionTarget").value;
  if (!target) {
    showError(new Error("请先新建或选择目标 mask"));
    return;
  }
  if (!state.crosshair && !state.pickedPoint) {
    showError(new Error("请先在 CT 或 3D 模型上点击区域生长种子点"));
    return;
  }
  setBusy(true);
  try {
    const payload = baseMaskPayload({
      target,
      voxel: state.crosshair,
      point: state.pickedPoint,
      lower: Number($("growLower").value || -150),
      upper: Number($("growUpper").value || 250),
      tolerance: Number($("growTolerance").value || 40),
      mode: $("regionWriteMode").value,
    });
    const result = await readResponse(await fetchJson("/api/mask/region-grow", payload));
    logLine(`[mask] grow ${result.mask.id}: ${result.mask.voxels} voxels, seed ${result.mask.seed_value.toFixed(1)} HU`);
    state.activeMasks.add(result.mask.id);
    state.meshCache.clear();
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function applyBooleanMask() {
  const left = $("maskA").value;
  const right = $("maskB").value;
  const target = $("maskBooleanTarget").value || left;
  if (!left || !right || !target) {
    showError(new Error("请选择参与交并操作的 mask"));
    return;
  }
  setBusy(true);
  try {
    const payload = baseMaskPayload({
      left,
      right,
      target,
      op: $("maskBooleanOp").value,
    });
    const result = await readResponse(await fetchJson("/api/mask/boolean", payload));
    logLine(`[mask] ${result.mask.op} -> ${result.mask.id}: ${result.mask.voxels} voxels`);
    state.activeMasks.add(result.mask.id);
    state.meshCache.clear();
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function applySmoothMask() {
  const target = $("smoothTarget").value;
  if (!target) {
    showError(new Error("请选择需要平滑的 mask"));
    return;
  }
  setBusy(true);
  try {
    const result = await readResponse(await fetchJson("/api/mask/smooth", baseMaskPayload({
      target,
      mode: $("smoothMode").value,
      iterations: Number($("maskSmoothIterations").value || 1),
    })));
    logLine(`[mask] smooth ${result.mask.id}: ${result.mask.voxels} voxels`);
    state.activeMasks.add(result.mask.id);
    state.meshCache.clear();
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

async function applyFillMask() {
  const target = $("fillTarget").value;
  if (!target) {
    showError(new Error("请选择需要填充的 mask"));
    return;
  }
  setBusy(true);
  try {
    const result = await readResponse(await fetchJson("/api/mask/fill", baseMaskPayload({ target })));
    logLine(`[mask] fill ${result.mask.id}: +${result.mask.added_voxels} voxels`);
    state.activeMasks.add(result.mask.id);
    state.meshCache.clear();
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

function baseMaskPayload(extra) {
  if (!state.session || !state.currentPatient) throw new Error("请先载入病例");
  return {
    session_id: state.session.id,
    patient_id: state.currentPatient.id,
    ...extra,
  };
}

async function loadAllCtSlices() {
  if (!$("showCtPanel").checked || !state.session || !state.currentPatient) return;
  await Promise.allSettled(AXES.map((axis) => loadCtSlice(axis, axisIndexFromCrosshair(axis, state.crosshair))));
}

async function loadCtSlice(axis, index = null) {
  if (!state.session || !state.currentPatient) return;
  try {
    const qs = new URLSearchParams({ patient: state.currentPatient.id, axis });
    if (index !== null && Number.isFinite(index)) qs.set("index", String(index));
    Array.from(state.activeMasks).forEach((mask) => qs.append("mask", mask));
    const payload = await readResponse(await fetch(`/api/session/${state.session.id}/ct?${qs.toString()}`));
    state.ctSlices[axis] = payload;
    const view = state.ctViews[axis];
    if (view.window === null || view.level === null) {
      view.window = Number(payload.window || 400);
      view.level = Number(payload.level || 40);
    }
    if (!state.crosshair) {
      state.crosshair = centerVoxelFromPayload(axis, payload);
      state.pickedPoint = worldFromVoxel(state.crosshair);
    }
    const slider = document.querySelector(`[data-axis="${axis}"]`);
    if (slider) {
      slider.max = payload.max_index;
      slider.value = axisIndexFromCrosshair(axis, state.crosshair) ?? payload.index;
    }
    drawSlice($(AXIS_CANVAS[axis]), payload);
  } catch (err) {
    logLine(`CT ${axis}: ${err.message}`);
  }
}

function centerVoxelFromPayload(axis, payload) {
  const shape = payload.shape || [0, 0, 0];
  const voxel = [Math.floor(shape[0] / 2), Math.floor(shape[1] / 2), Math.floor(shape[2] / 2)];
  if (axis === "sagittal") voxel[0] = payload.index;
  else if (axis === "coronal") voxel[1] = payload.index;
  else voxel[2] = payload.index;
  return voxel;
}

function axisIndexFromCrosshair(axis, voxel) {
  if (!voxel) return null;
  if (axis === "sagittal") return voxel[0];
  if (axis === "coronal") return voxel[1];
  return voxel[2];
}

function updateCrosshairFromAxis(axis, index) {
  const payload = state.ctSlices[axis];
  const shape = payload?.shape || [0, 0, 0];
  const voxel = state.crosshair || [Math.floor(shape[0] / 2), Math.floor(shape[1] / 2), Math.floor(shape[2] / 2)];
  if (axis === "sagittal") voxel[0] = index;
  else if (axis === "coronal") voxel[1] = index;
  else voxel[2] = index;
  state.crosshair = clampVoxel(voxel, shape);
  updateSelectionText();
}

function drawSlice(canvas, payload) {
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width || payload.width));
  const h = Math.max(1, Math.round(rect.height || payload.height));
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const src = renderSourceSlice(payload);
  const view = state.ctViews[payload.axis] || { zoom: 1, panX: 0, panY: 0 };
  const fit = Math.min(w / src.width, h / src.height);
  const scale = fit * Math.max(0.25, Math.min(view.zoom || 1, 12));
  const drawW = src.width * scale;
  const drawH = src.height * scale;
  const originX = (w - drawW) / 2 + (view.panX || 0);
  const originY = (h - drawH) / 2 + (view.panY || 0);
  view.last = { originX, originY, scale, sourceW: src.width, sourceH: src.height };
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(src, originX, originY, drawW, drawH);
  drawCrosshair(ctx, payload, w, h);
}

function renderSourceSlice(payload) {
  const w = payload.width;
  const h = payload.height;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  const view = state.ctViews[payload.axis] || {};
  const windowValue = Math.max(1, Number(view.window ?? payload.window ?? 400));
  const levelValue = Number(view.level ?? payload.level ?? 40);
  const lo = levelValue - windowValue / 2;
  let p = 0;
  for (let y = 0; y < h; y += 1) {
    const row = (payload.values && payload.values[y]) || payload.pixels[y] || [];
    for (let x = 0; x < w; x += 1) {
      const raw = row[x] || 0;
      const v = payload.values ? Math.max(0, Math.min(255, Math.round(((raw - lo) / windowValue) * 255))) : raw;
      image.data[p++] = v;
      image.data[p++] = v;
      image.data[p++] = v;
      image.data[p++] = 255;
    }
  }
  blendMaskOverlays(image, payload);
  ctx.putImageData(image, 0, 0);
  return canvas;
}

function blendMaskOverlays(image, payload) {
  const w = payload.width;
  const h = payload.height;
  (payload.masks || []).forEach((mask, idx) => {
    const rgb = hexToRgb(mask.color || MASK_COLORS[idx % MASK_COLORS.length]);
    const pixels = mask.pixels || [];
    for (let y = 0; y < h; y += 1) {
      const row = pixels[y] || [];
      for (let x = 0; x < w; x += 1) {
        if ((row[x] || 0) <= 0) continue;
        const p = (y * w + x) * 4;
        image.data[p] = Math.round(image.data[p] * 0.25 + rgb.r * 0.75);
        image.data[p + 1] = Math.round(image.data[p + 1] * 0.25 + rgb.g * 0.75);
        image.data[p + 2] = Math.round(image.data[p + 2] * 0.25 + rgb.b * 0.75);
        image.data[p + 3] = 255;
      }
    }
  });
}

function drawMaskOverlay(ctx, mask, w, h, idx) {
  const rgb = hexToRgb(mask.color || MASK_COLORS[idx % MASK_COLORS.length]);
  ctx.save();
  ctx.globalAlpha = 0.7;
  ctx.fillStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
  const pixels = mask.pixels || [];
  for (let y = 0; y < h; y += 1) {
    const row = pixels[y] || [];
    for (let x = 0; x < w; x += 1) {
      if ((row[x] || 0) > 0) ctx.fillRect(x, y, 1, 1);
    }
  }
  ctx.restore();
}

function drawCrosshair(ctx, payload, w, h) {
  if (!state.crosshair || !payload.shape) return;
  const pos = displayPointFromVoxel(payload.axis, state.crosshair, payload.step || 1);
  if (!pos) return;
  ctx.save();
  ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) / 260));
  ctx.shadowBlur = 7;
  ctx.shadowColor = "rgba(0,0,0,0.55)";
  ctx.strokeStyle = "#e31b23";
  ctx.beginPath();
  ctx.moveTo(0, pos.y + 0.5);
  ctx.lineTo(w, pos.y + 0.5);
  ctx.stroke();
  ctx.strokeStyle = "#39d353";
  ctx.beginPath();
  ctx.moveTo(pos.x + 0.5, 0);
  ctx.lineTo(pos.x + 0.5, h);
  ctx.stroke();
  ctx.restore();
}

function canvasPointFromVoxel(axis, voxel, step) {
  if (!voxel) return null;
  if (axis === "sagittal") return { x: Math.round(voxel[1] / step), y: Math.round(voxel[2] / step) };
  if (axis === "coronal") return { x: Math.round(voxel[0] / step), y: Math.round(voxel[2] / step) };
  return { x: Math.round(voxel[0] / step), y: Math.round(voxel[1] / step) };
}

function displayPointFromVoxel(axis, voxel, step) {
  const point = canvasPointFromVoxel(axis, voxel, step);
  const view = state.ctViews[axis]?.last;
  if (!point || !view) return point;
  return {
    x: view.originX + point.x * view.scale,
    y: view.originY + point.y * view.scale,
  };
}

async function handleCtPointer(event, axis, start) {
  const payload = state.ctSlices[axis];
  if (!payload) return;
  event.preventDefault();
  if (start) {
    event.currentTarget.setPointerCapture?.(event.pointerId);
    state.lastPointer = { x: event.clientX, y: event.clientY };
    state.windowing = event.button === 2;
    state.panning = event.button === 1 || (event.button === 0 && (event.shiftKey || event.altKey));
    state.painting = event.button === 0 && !state.panning && !state.windowing;
  }
  if (!start && state.windowing) {
    adjustWindowLevel(event, axis);
    state.lastPointer = { x: event.clientX, y: event.clientY };
    return;
  }
  if (!start && state.panning) {
    panCtView(event, axis);
    state.lastPointer = { x: event.clientX, y: event.clientY };
    return;
  }
  if (!start && (!$("paintMaskToggle").checked || !state.painting)) return;
  const voxel = voxelFromPointer(event, axis, payload);
  state.crosshair = voxel;
  state.pickedPoint = worldFromVoxel(voxel);
  updateSlidersFromCrosshair();
  updateSelectionText();
  AXES.forEach((name) => {
    const item = state.ctSlices[name];
    if (item) drawSlice($(AXIS_CANVAS[name]), item);
  });
  renderScene();
  if ($("paintMaskToggle").checked) {
    await paintAtVoxel(axis, payload, voxel);
  } else if (start) {
    await loadAllCtSlices();
  }
}

function endCtGesture() {
  state.painting = false;
  state.windowing = false;
  state.panning = false;
  state.lastPointer = null;
}

function adjustWindowLevel(event, axis) {
  const view = state.ctViews[axis];
  const last = state.lastPointer || { x: event.clientX, y: event.clientY };
  const dx = event.clientX - last.x;
  const dy = event.clientY - last.y;
  view.window = Math.max(1, Number(view.window || state.ctSlices[axis]?.window || 400) + dx * 3);
  view.level = Number(view.level || state.ctSlices[axis]?.level || 40) - dy * 2;
  drawSlice($(AXIS_CANVAS[axis]), state.ctSlices[axis]);
}

function panCtView(event, axis) {
  const view = state.ctViews[axis];
  const last = state.lastPointer || { x: event.clientX, y: event.clientY };
  view.panX = (view.panX || 0) + event.clientX - last.x;
  view.panY = (view.panY || 0) + event.clientY - last.y;
  drawSlice($(AXIS_CANVAS[axis]), state.ctSlices[axis]);
}

function handleCtWheel(event, axis) {
  const payload = state.ctSlices[axis];
  if (!payload) return;
  event.preventDefault();
  const view = state.ctViews[axis];
  const before = imagePointFromPointer(event, axis, payload);
  const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
  view.zoom = Math.max(0.25, Math.min(12, (view.zoom || 1) * factor));
  drawSlice($(AXIS_CANVAS[axis]), payload);
  const afterView = state.ctViews[axis].last;
  if (before && afterView) {
    const rect = event.currentTarget.getBoundingClientRect();
    const cx = event.clientX - rect.left;
    const cy = event.clientY - rect.top;
    view.panX += cx - (afterView.originX + before.x * afterView.scale);
    view.panY += cy - (afterView.originY + before.y * afterView.scale);
    drawSlice($(AXIS_CANVAS[axis]), payload);
  }
}

function voxelFromPointer(event, axis, payload) {
  const point = imagePointFromPointer(event, axis, payload);
  const x = Math.round(point?.x ?? 0);
  const y = Math.round(point?.y ?? 0);
  const step = payload.step || 1;
  const shape = payload.shape || [1, 1, 1];
  let voxel;
  if (axis === "sagittal") voxel = [payload.index, x * step, y * step];
  else if (axis === "coronal") voxel = [x * step, payload.index, y * step];
  else voxel = [x * step, y * step, payload.index];
  return clampVoxel(voxel, shape);
}

function imagePointFromPointer(event, axis, payload) {
  const canvas = event.currentTarget;
  const rect = canvas.getBoundingClientRect();
  const view = state.ctViews[axis]?.last;
  if (!view) {
    return {
      x: ((event.clientX - rect.left) / rect.width) * payload.width,
      y: ((event.clientY - rect.top) / rect.height) * payload.height,
    };
  }
  return {
    x: (event.clientX - rect.left - view.originX) / view.scale,
    y: (event.clientY - rect.top - view.originY) / view.scale,
  };
}

function clampVoxel(voxel, shape) {
  return voxel.map((v, idx) => Math.max(0, Math.min(Math.round(v), Math.max(0, (shape[idx] || 1) - 1))));
}

function updateSlidersFromCrosshair() {
  if (!state.crosshair) return;
  AXES.forEach((axis) => {
    const slider = document.querySelector(`[data-axis="${axis}"]`);
    if (slider) slider.value = axisIndexFromCrosshair(axis, state.crosshair);
  });
}

function updateSelectionText() {
  if (!state.crosshair) return;
  const world = worldFromVoxel(state.crosshair);
  const suffix = world ? ` · x=${world[0].toFixed(2)}, y=${world[1].toFixed(2)}, z=${world[2].toFixed(2)}` : "";
  const text = `CT voxel i=${state.crosshair[0]}, j=${state.crosshair[1]}, k=${state.crosshair[2]}${suffix}`;
  $("pickedPoint").textContent = text;
  $("selectionBox").textContent = text;
}

function activeAffinePayload() {
  return state.ctSlices.axial || state.ctSlices.coronal || state.ctSlices.sagittal || null;
}

function worldFromVoxel(voxel) {
  const payload = activeAffinePayload();
  if (!payload?.affine || !voxel) return null;
  return applyMatrix4(payload.affine, [voxel[0], voxel[1], voxel[2], 1]).slice(0, 3);
}

function voxelFromWorld(world) {
  const payload = activeAffinePayload();
  if (!payload?.inv_affine || !world) return null;
  const raw = applyMatrix4(payload.inv_affine, [world[0], world[1], world[2], 1]).slice(0, 3);
  return clampVoxel(raw, payload.shape || [1, 1, 1]);
}

function applyMatrix4(matrix, vector) {
  return matrix.map((row) => row.reduce((sum, value, idx) => sum + value * vector[idx], 0));
}

async function paintAtVoxel(axis, payload, voxel) {
  if (!state.session || !state.currentPatient || state.paintBusy) return;
  const now = Date.now();
  if (now - state.lastPaintAt < 110) return;
  const target = $("editTarget").value;
  if (!target) {
    showError(new Error("请先选择 CT 画笔写入的目标 mask"));
    return;
  }
  state.paintBusy = true;
  state.lastPaintAt = now;
  try {
    const result = await readResponse(await fetchJson("/api/mask/paint", baseMaskPayload({
      target,
      axis,
      index: payload.index,
      voxel,
      radius_mm: Number($("ctBrushRadius").value || 3),
      mode: $("paintMode").value,
    })));
    state.activeMasks.add(result.mask.id);
    state.meshCache.clear();
    state.maskCatalog = result.masks || state.maskCatalog;
    buildMaskLayers();
    await loadActiveMeshes();
    await loadAllCtSlices();
  } catch (err) {
    showError(err);
  } finally {
    state.paintBusy = false;
  }
}

function renderInspector() {
  const status = state.currentPatient?.status || {};
  const files = status.files || {};
  const fileList = $("fileList");
  fileList.innerHTML = "";
  Object.entries(files).forEach(([key, info]) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `<span>${escapeHtml(key)}</span><strong class="${info.exists ? "ok" : "warn"}">${info.exists ? formatBytes(info.size) : "missing"}</strong>`;
    fileList.appendChild(row);
  });
  if (!Object.keys(files).length) fileList.textContent = "暂无文件信息";
}

function downloadResults() {
  if (!state.session) {
    showError(new Error("请先载入病例"));
    return;
  }
  const patient = $("patientSelect").value || "all";
  window.location.href = `/api/session/${state.session.id}/download?patient=${encodeURIComponent(patient)}`;
}

function clearJob() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.job = null;
  $("jobStatus").textContent = "空闲";
  $("logs").textContent = "";
}

function logLine(text) {
  $("logs").textContent += `${text}\n`;
  $("logs").scrollTop = $("logs").scrollHeight;
}

function setBusy(isBusy) {
  document.querySelectorAll("button").forEach((btn) => {
    if (["toggleSettingsBtn"].includes(btn.id)) return;
    btn.disabled = isBusy;
  });
}

function showError(err) {
  const message = err?.message || String(err);
  $("jobStatus").textContent = message;
  $("jobStatus").style.color = "var(--danger)";
  logLine(`ERROR: ${message}`);
}

function fetchJson(url, payload) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function readResponse(res) {
  const payload = await res.json();
  if (!res.ok || payload.error) throw new Error(payload.error || `HTTP ${res.status}`);
  return payload;
}

function formatBytes(size) {
  const n = Number(size || 0);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function hexToRgb(hex) {
  const clean = String(hex || "#f97316").replace("#", "");
  const value = Number.parseInt(clean.length === 3 ? clean.split("").map((c) => c + c).join("") : clean, 16);
  return { r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255 };
}

window.addEventListener("DOMContentLoaded", init);
