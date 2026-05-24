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

const state = {
  session: null,
  patients: [],
  currentPatient: null,
  modelMeshes: DEFAULT_MODELS,
  activeModels: new Set(["predict"]),
  activeOrgans: new Set(),
  meshCache: new Map(),
  traces: [],
  job: null,
  pollTimer: null,
  pickedPoint: null,
};

const $ = (id) => document.getElementById(id);

function init() {
  buildStepButtons();
  bindEvents();
  checkHealth();
}

function bindEvents() {
  $("loadBtn").addEventListener("click", createSession);
  $("refreshBtn").addEventListener("click", refreshData);
  $("runAllBtn").addEventListener("click", () => runSteps(STEPS.map(([key]) => key), true));
  $("downloadBtn").addEventListener("click", downloadResults);
  $("toggleSettingsBtn").addEventListener("click", () => $("settingsPanel").classList.toggle("hidden"));
  $("patientSelect").addEventListener("change", () => {
    state.meshCache.clear();
    refreshData();
  });
  $("meshOpacity").addEventListener("input", renderScene);
  $("maxFaces").addEventListener("change", () => {
    state.meshCache.clear();
    loadActiveMeshes();
  });
  $("showCtPanel").addEventListener("change", () => {
    $("ctPanel").classList.toggle("hidden", !$("showCtPanel").checked);
    if ($("showCtPanel").checked) loadAllCtSlices();
  });
  document.querySelectorAll("[data-axis]").forEach((input) => {
    input.addEventListener("input", () => loadCtSlice(input.dataset.axis, Number(input.value)));
  });
  $("deleteBrushBtn").addEventListener("click", () => applyBrush("delete"));
  $("keepBrushBtn").addEventListener("click", () => applyBrush("keep"));
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

async function createSession() {
  clearJob();
  setBusy(true);
  try {
    const root = $("rootFolder").value.trim();
    if (!root) throw new Error("请输入 patient 文件夹或批量根目录");
    const res = await fetchJson("/api/session", { root_folder: root });
    const payload = await readResponse(res);
    state.session = payload.session;
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
    buildModelLayers();
    buildOrganLayers();
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
}

function syncCurrentPatient() {
  const id = $("patientSelect").value;
  state.currentPatient = state.patients.find((p) => p.id === id) || state.patients[0] || null;
  if (state.currentPatient) {
    const status = state.currentPatient.status || {};
    const organCount = status.organs?.length || 0;
    $("patientSummary").textContent = `${state.currentPatient.id} · organs ${organCount} · ${status.folder || ""}`;
  } else {
    $("patientSummary").textContent = "尚未载入";
  }
}

function buildModelLayers() {
  const wrap = $("modelLayers");
  wrap.innerHTML = "";
  const files = state.currentPatient?.status?.files || {};
  Object.entries(state.modelMeshes).forEach(([key, info]) => {
    const fileKey = (info.file || "").replace(".stl", "").replace(".nii.gz", "").replace("/", "_");
    const exists = files[fileKey]?.exists || false;
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

async function loadActiveMeshes() {
  if (!state.session || !state.currentPatient) return;
  const meshNames = [
    ...Array.from(state.activeModels),
    ...Array.from(state.activeOrgans).map((name) => `organ:${name}`),
  ];
  const maxFaces = Number($("maxFaces").value || 70000);
  const requests = meshNames.map((name) => loadMesh(name, maxFaces));
  await Promise.allSettled(requests);
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
      customdata: vertices.map((v) => `${style.label || name}\n${v.map((x) => Number(x).toFixed(2)).join(", ")}`),
      hovertemplate: `<b>${style.label || name}</b><br>x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<extra></extra>`,
    });
  });

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#f8fafc",
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
  plot.on("plotly_click", (event) => {
    const pt = event.points?.[0];
    if (!pt) return;
    const x = Number(pt.x);
    const y = Number(pt.y);
    const z = Number(pt.z);
    if (![x, y, z].every(Number.isFinite)) return;
    state.pickedPoint = [x, y, z];
    const text = `x=${x.toFixed(2)}, y=${y.toFixed(2)}, z=${z.toFixed(2)}`;
    $("pickedPoint").textContent = text;
    $("selectionBox").textContent = text;
  });
}

function axisLayout(title) {
  return {
    title,
    backgroundcolor: "#f8fafc",
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

async function loadAllCtSlices() {
  await Promise.allSettled(["axial", "coronal", "sagittal"].map((axis) => loadCtSlice(axis)));
}

async function loadCtSlice(axis, index = null) {
  if (!state.session || !state.currentPatient) return;
  try {
    const qs = new URLSearchParams({ patient: state.currentPatient.id, axis });
    if (index !== null && Number.isFinite(index)) qs.set("index", String(index));
    const payload = await readResponse(await fetch(`/api/session/${state.session.id}/ct?${qs.toString()}`));
    const slider = document.querySelector(`[data-axis="${axis}"]`);
    if (slider) {
      slider.max = payload.max_index;
      slider.value = payload.index;
    }
    const canvasId = axis === "axial" ? "ctAxial" : axis === "coronal" ? "ctCoronal" : "ctSagittal";
    drawSlice($(canvasId), payload);
  } catch (err) {
    logLine(`CT ${axis}: ${err.message}`);
  }
}

function drawSlice(canvas, payload) {
  const w = payload.width;
  const h = payload.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  let p = 0;
  for (let y = 0; y < h; y += 1) {
    const row = payload.pixels[y] || [];
    for (let x = 0; x < w; x += 1) {
      const v = row[x] || 0;
      image.data[p++] = v;
      image.data[p++] = v;
      image.data[p++] = v;
      image.data[p++] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
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

  const meta = status.pretrain_meta || {};
  const check = status.predict_check || {};
  const lines = [];
  if (meta.pretrain_quality) lines.push(`pretrain_quality: ${meta.pretrain_quality}`);
  if (meta.mask_voxels !== undefined) lines.push(`mask_voxels: ${meta.mask_voxels}`);
  if (meta.stl_bytes !== undefined) lines.push(`pretrain_stl: ${formatBytes(meta.stl_bytes)}`);
  if (meta.quality_issues?.length) lines.push(`issues: ${meta.quality_issues.join(", ")}`);
  if (check.mesh?.faces !== undefined) lines.push(`predict_faces: ${check.mesh.faces}`);
  if (check.smooth_iterations !== undefined) lines.push(`smooth_iterations: ${check.smooth_iterations}`);
  $("metaBox").textContent = lines.length ? lines.join("\n") : "暂无";
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

window.addEventListener("DOMContentLoaded", init);
