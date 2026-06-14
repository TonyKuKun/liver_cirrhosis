const STAGES = [
  ["segmentation", "CT分割"],
  ["features", "几何特征"],
  ["pvp", "PVP预测"],
];

const state = {
  session: null,
  patients: [],
  currentPatient: null,
  job: null,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

function init() {
  bindEvents();
  renderTimeline();
  checkHealth();
}

function bindEvents() {
  $("loadBtn").addEventListener("click", createSession);
  $("refreshBtn").addEventListener("click", refreshData);
  $("downloadBtn").addEventListener("click", downloadResults);
  $("patientSelect").addEventListener("change", refreshData);
  $("prevPatientBtn").addEventListener("click", () => stepPatient(-1));
  $("nextPatientBtn").addEventListener("click", () => stepPatient(1));
  $("openPreviewBtn").addEventListener("click", openPreview);
  document.querySelectorAll("[data-run-stage]").forEach((button) => {
    button.addEventListener("click", () => runStage(button.dataset.runStage));
  });
}

async function checkHealth() {
  try {
    const payload = await readResponse(await fetch("/api/health"));
    const envName = payload.runtime?.conda_env || "base";
    $("serverState").textContent = `本地服务已连接 · ${envName}`;
  } catch (err) {
    $("serverState").textContent = "本地服务未连接";
  }
}

async function createSession() {
  clearJob();
  setBusy(true);
  try {
    const root = $("rootFolder").value.trim();
    if (!root) throw new Error("请输入病人根目录");
    const payload = await readResponse(await fetchJson("/api/session", {
      root_folder: root,
      model_dir: $("modelDir").value.trim(),
    }));
    state.session = payload.session;
    if (!$("modelDir").value.trim() && state.session.model_dir) {
      $("modelDir").value = state.session.model_dir;
    }
    populatePatients();
    await refreshData();
    logLine(`Loaded ${state.session.patients.length} patient(s).`);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

function populatePatients() {
  const select = $("patientSelect");
  select.innerHTML = "";
  (state.session?.patients || []).forEach((patient) => {
    const option = document.createElement("option");
    option.value = patient.id;
    option.textContent = patient.id;
    select.appendChild(option);
  });
  const count = select.options.length;
  $("prevPatientBtn").disabled = count < 2;
  $("nextPatientBtn").disabled = count < 2;
}

function stepPatient(delta) {
  const select = $("patientSelect");
  if (!select.options.length) return;
  select.selectedIndex = (select.selectedIndex + delta + select.options.length) % select.options.length;
  refreshData();
}

async function refreshData() {
  if (!state.session) return;
  try {
    const patient = $("patientSelect").value || state.session.patients?.[0]?.id || "";
    const url = `/api/session/${state.session.id}/data?patient=${encodeURIComponent(patient)}`;
    const payload = await readResponse(await fetch(url));
    state.session = payload.session;
    state.patients = payload.patients || [];
    state.currentPatient = state.patients[0] || null;
    renderAll();
  } catch (err) {
    showError(err);
  }
}

function renderAll() {
  renderHeader();
  renderTimeline();
  renderStageFiles();
  renderPvp();
  renderFeatureSummary();
  renderOrganSummary();
  renderPreview();
}

function renderHeader() {
  const patient = state.currentPatient;
  if (!patient) {
    $("workspaceTitle").textContent = "等待载入病人";
    $("workspaceSubtitle").textContent = "输入目录后选择病人，按阶段运行和查看结果。";
    $("patientMeta").textContent = "尚未加载";
    return;
  }
  const status = patient.status || {};
  $("workspaceTitle").textContent = patient.id;
  $("workspaceSubtitle").textContent = status.folder || "";
  $("patientMeta").innerHTML = `
    <div>路径：${escapeHtml(status.folder || "")}</div>
    <div>模型：${escapeHtml(state.session?.model_dir || "未找到可推理模型")}</div>
  `;
}

function renderTimeline() {
  const wrap = $("timeline");
  const stages = state.currentPatient?.status?.stages || {};
  wrap.innerHTML = "";
  STAGES.forEach(([key, label], idx) => {
    const status = stages[key]?.status || "missing";
    const item = document.createElement("div");
    item.className = "timeline-item";
    item.innerHTML = `
      <span class="timeline-index">${idx + 1}</span>
      <span class="timeline-label">${label}</span>
      <span class="badge ${status}">${statusText(status)}</span>
    `;
    wrap.appendChild(item);
  });
}

function renderStageFiles() {
  renderFiles("segmentationFiles", state.currentPatient?.status?.stages?.segmentation?.outputs || {});
  renderFiles("featureFiles", state.currentPatient?.status?.stages?.features?.outputs || {});
}

function renderFiles(targetId, files) {
  const wrap = $(targetId);
  wrap.innerHTML = "";
  const entries = Object.entries(files);
  if (!entries.length) {
    wrap.innerHTML = `<div class="file-row"><span>尚无状态</span><strong class="warn">waiting</strong></div>`;
    return;
  }
  entries.forEach(([name, info]) => {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `
      <span title="${escapeHtml(info.path || name)}">${escapeHtml(name)}</span>
      <strong class="${info.exists ? "ok" : "warn"}">${info.exists ? formatBytes(info.size) : "missing"}</strong>
    `;
    wrap.appendChild(row);
  });
}

function renderPvp() {
  const box = $("pvpResult");
  const pred = state.currentPatient?.status?.prediction;
  if (!pred) {
    box.innerHTML = `
      <div class="file-row"><span>pvp_prediction.json</span><strong class="warn">missing</strong></div>
      <div class="file-row"><span>模型目录</span><strong>${escapeHtml(state.session?.model_valid ? "ready" : "missing")}</strong></div>
    `;
    return;
  }
  const warnings = (pred.warnings || []).join("; ") || "无";
  box.innerHTML = `
    <div class="pvp-number">
      <span>PVP mean</span>
      <strong>${fmt(pred.pvp_mean, 2)}</strong>
      <span>mmHg</span>
    </div>
    ${metricRow("fold std", fmt(pred.pvp_std, 3))}
    ${metricRow("fold range", `${fmt(pred.pvp_min, 2)} - ${fmt(pred.pvp_max, 2)}`)}
    ${metricRow("checkpoint", `${pred.n_checkpoints || 0}`)}
    ${metricRow("warnings", warnings)}
  `;
}

function renderFeatureSummary() {
  const wrap = $("featureSummary");
  const summary = state.currentPatient?.status?.features_summary || {};
  const segments = summary.segments || [];
  const metrics = summary.key_metrics || {};
  wrap.innerHTML = "";
  if (!segments.length && !Object.keys(metrics).length) {
    wrap.innerHTML = `<div class="metric-row"><span>unified_features.json</span><strong class="warn">missing</strong></div>`;
    return;
  }
  Object.entries(metrics).forEach(([key, value]) => {
    wrap.appendChild(metricElement(key, fmt(value, 4)));
  });
  segments.slice(0, 8).forEach((seg) => {
    wrap.appendChild(metricElement(
      `${seg.name} length / diameter`,
      `${fmt(seg.length, 1)} mm / ${fmt(seg.mean_diameter, 2)} mm`,
    ));
  });
}

function renderOrganSummary() {
  const wrap = $("organSummary");
  const organs = state.currentPatient?.status?.organs || {};
  wrap.innerHTML = "";
  const names = Object.keys(organs);
  if (!names.length) {
    wrap.innerHTML = `<div class="organ-row"><span>segmentation/*.stl</span><strong class="warn">missing</strong></div>`;
    return;
  }
  names.forEach((name) => {
    const info = organs[name];
    const row = document.createElement("div");
    row.className = "organ-row";
    row.innerHTML = `<span>${escapeHtml(name)}</span><strong>${formatBytes(info.size)}</strong>`;
    wrap.appendChild(row);
  });
}

function renderPreview() {
  const patient = state.currentPatient;
  const hasHtml = patient?.status?.preview?.feature_html;
  const empty = $("previewEmpty");
  const frame = $("featurePreview");
  if (!patient || !hasHtml || !state.session) {
    frame.removeAttribute("src");
    empty.classList.remove("hidden");
    $("openPreviewBtn").disabled = true;
    return;
  }
  const url = `/api/session/${state.session.id}/patient-file?patient=${encodeURIComponent(patient.id)}&file=${encodeURIComponent("vis_interactive.html")}`;
  frame.src = url;
  empty.classList.add("hidden");
  $("openPreviewBtn").disabled = false;
}

function openPreview() {
  const patient = state.currentPatient;
  if (!patient || !state.session) return;
  const url = `/api/session/${state.session.id}/patient-file?patient=${encodeURIComponent(patient.id)}&file=${encodeURIComponent("vis_interactive.html")}`;
  window.open(url, "_blank", "noopener");
}

async function runStage(stage) {
  if (!state.session) {
    showError(new Error("请先加载病人"));
    return;
  }
  clearJob();
  setBusy(true);
  try {
    const patientId = $("runAllPatients").checked ? "all" : ($("patientSelect").value || "all");
    const payload = await readResponse(await fetchJson("/api/run-stage", {
      session_id: state.session.id,
      stage,
      patient_id: patientId,
      force: $("forceRun").checked,
      model_dir: $("modelDir").value.trim(),
      device: $("deviceSelect").value,
    }));
    state.job = payload.job;
    pollJob();
    state.pollTimer = setInterval(pollJob, 1500);
  } catch (err) {
    setBusy(false);
    showError(err);
  }
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
      await refreshData();
    }
  } catch (err) {
    logLine(`Job polling failed: ${err.message}`);
  }
}

function renderJob() {
  if (!state.job) return;
  const pct = state.job.total ? Math.round((state.job.completed / state.job.total) * 100) : 0;
  $("jobStatus").textContent = `${statusText(state.job.status)} ${pct}% ${state.job.current || ""}`;
  $("jobStatus").className = `job-status ${state.job.status === "failed" ? "danger" : ""}`;
  $("logs").textContent = (state.job.logs || []).join("\n\n");
  $("logs").scrollTop = $("logs").scrollHeight;
}

function downloadResults() {
  if (!state.session) {
    showError(new Error("请先加载病人"));
    return;
  }
  const patient = $("patientSelect").value || "all";
  window.location.href = `/api/session/${state.session.id}/download?patient=${encodeURIComponent(patient)}`;
}

function setBusy(isBusy) {
  document.querySelectorAll("button").forEach((button) => {
    if (["refreshBtn"].includes(button.id)) return;
    button.disabled = isBusy;
  });
  if (!isBusy) {
    const count = $("patientSelect").options.length;
    $("prevPatientBtn").disabled = count < 2;
    $("nextPatientBtn").disabled = count < 2;
    $("openPreviewBtn").disabled = !state.currentPatient?.status?.preview?.feature_html;
  }
}

function clearJob() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.job = null;
  $("jobStatus").textContent = "空闲";
  $("jobStatus").className = "job-status";
  $("logs").textContent = "";
}

function logLine(text) {
  $("logs").textContent += `${text}\n`;
  $("logs").scrollTop = $("logs").scrollHeight;
}

function showError(err) {
  const message = err?.message || String(err);
  $("jobStatus").textContent = message;
  $("jobStatus").className = "job-status danger";
  logLine(`ERROR: ${message}`);
}

function statusText(status) {
  return {
    done: "完成",
    ready: "可运行",
    missing: "缺输入",
    running: "运行中",
    failed: "失败",
  }[status] || status;
}

function metricRow(label, value) {
  return `<div class="metric-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function metricElement(label, value) {
  const row = document.createElement("div");
  row.className = "metric-row";
  row.innerHTML = `<span title="${escapeHtml(label)}">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>`;
  return row;
}

function fetchJson(url, payload) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function readResponse(response) {
  const payload = await response.json();
  if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function fmt(value, digits = 2) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "NA";
  return n.toFixed(digits);
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
