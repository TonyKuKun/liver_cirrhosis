const DEFAULT_PARAMS = {
  pitch: 0.5,
  min_branch_length_mm: 10.0,
  min_relative_length: 0.05,
  min_radius_ratio: 0.4,
  keep_radius_ratio: 0.55,
  absolute_min_branch_length_mm: 3.0,
  absolute_min_radius_mm: 0.5,
  merge_bp_distance_mm: 5.0,
  n_fit_points: 10,
  n_profile_points: 100,
  curvature_window: 7,
  sample_step: 3,
  ownership_factor: 1.8,
  junction_policy: "min_valid",
  max_diameter_rate_per_mm: 0.5,
};

const PARAM_LABELS = {
  pitch: "体素间距",
  min_branch_length_mm: "最小分支长度",
  min_relative_length: "相对长度阈值",
  min_radius_ratio: "最小半径比例",
  keep_radius_ratio: "保留半径比例",
  absolute_min_branch_length_mm: "硬剪枝长度",
  absolute_min_radius_mm: "硬剪枝半径",
  merge_bp_distance_mm: "分叉点合并距离",
  n_fit_points: "拟合点数",
  n_profile_points: "剖面点数",
  curvature_window: "曲率窗口",
  sample_step: "截面采样步长",
  ownership_factor: "截面归属半径倍数",
  junction_policy: "交叉区策略",
  max_diameter_rate_per_mm: "直径变化率上限",
};

const STEPS = [
  ["centerline", "中心线提取"],
  ["smooth", "中心线平滑"],
  ["segment", "解剖分段"],
  ["profiles", "截面特征"],
  ["features", "统计特征"],
  ["export", "导出可视化"],
];

const LAYERS = {
  mesh: true,
  rawCenterline: false,
  smoothCenterline: true,
  segments: true,
  branchPoints: false,
  featurePoints: true,
  sampledSections: true,
  surfaceSections: false,
  maxSections: true,
  meanSections: true,
  labels: true,
};

const CENTERLINE_EDIT_COLORS = [
  "#d9822b",
  "#7c3aed",
  "#0f9f6e",
  "#e11d48",
  "#2563eb",
  "#ca8a04",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#9333ea",
];

const state = {
  mode: "batch",
  session: null,
  data: null,
  params: { ...DEFAULT_PARAMS },
  stepModes: Object.fromEntries(STEPS.map(([key]) => [key, "recompute"])),
  layers: { ...LAYERS },
  job: null,
  pollTimer: null,
  centerlineEdit: {
    active: false,
    selected: new Set(),
  },
  manualSegment: {
    active: false,
    edits: new Map(),
    dirty: false,
  },
  analysisRange: {
    active: false,
    ranges: new Map(),
    dirty: false,
    suggestions: null,
    boundarySections: {},
  },
};

const $ = (id) => document.getElementById(id);

function init() {
  buildStepButtons();
  buildParamInputs();
  bindEvents();
  renderCenterlineEditControls();
  renderManualSegmentationControls();
  renderAnalysisRangeControls();
  checkHealth();
}

function bindEvents() {
  $("createSessionBtn").addEventListener("click", createSession);
  $("nextPatientBtn").addEventListener("click", nextPatient);
  $("runAllBtn").addEventListener("click", () => runSteps(STEPS.map(([key]) => key), true));
  $("refreshBtn").addEventListener("click", refreshData);
  $("downloadBtn").addEventListener("click", downloadResults);
  $("paramsBtn").addEventListener("click", () => $("paramsPanel").classList.toggle("hidden"));
  $("patientSelect").addEventListener("change", () => {
    resetStepModesToRecompute();
    resetManualSegmentationEditor();
    resetAnalysisRangeEditor();
    refreshData();
  });
  $("sectionStride").addEventListener("input", () => {
    $("sectionStrideValue").textContent = $("sectionStride").value;
  });
  $("sectionStride").addEventListener("change", refreshData);
  $("meshOpacity").addEventListener("input", () => renderScene());
  $("centerlineEditBtn").addEventListener("click", toggleCenterlineEdit);
  $("centerlinePickBtn").addEventListener("click", toggleSelectedCenterlineBranch);
  $("centerlineBranchSelect").addEventListener("change", renderCenterlineEditControls);
  $("centerlineUndoBtn").addEventListener("click", clearCenterlineSelection);
  $("centerlineSaveBtn").addEventListener("click", saveCenterlineDeletion);
  $("manualSegmentBtn").addEventListener("click", toggleManualSegmentation);
  $("manualAtomicSelect").addEventListener("change", renderManualSegmentationControls);
  $("manualKeep").addEventListener("change", updateManualAssignment);
  $("manualVesselSelect").addEventListener("change", updateManualAssignment);
  $("manualResetBtn").addEventListener("click", resetManualChanges);
  $("manualSaveBtn").addEventListener("click", saveManualSegmentation);
  $("analysisRangeBtn").addEventListener("click", toggleAnalysisRange);
  $("analysisVesselSelect").addEventListener("change", renderAnalysisRangeControls);
  $("analysisSuggestBtn").addEventListener("click", suggestAnalysisRanges);
  $("analysisStart").addEventListener("input", updateAnalysisRange);
  $("analysisEnd").addEventListener("input", updateAnalysisRange);
  $("analysisResetBtn").addEventListener("click", resetAnalysisChanges);
  $("analysisSaveBtn").addEventListener("click", saveAnalysisRanges);

  document.querySelectorAll(".layer-toggle").forEach((input) => {
    input.addEventListener("change", () => {
      state.layers[input.dataset.layer] = input.checked;
      if (input.dataset.layer === "surfaceSections" && input.checked) {
        refreshData();
      } else {
        renderScene();
      }
    });
  });
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const envName = data.runtime?.conda_env || "unknown";
    $("serverState").textContent = data.ok ? `本地服务已连接 · ${envName}` : "服务异常";
  } catch (err) {
    $("serverState").textContent = "服务未连接";
  }
}

function buildStepButtons() {
  const wrap = $("stepButtons");
  wrap.innerHTML = "";
  STEPS.forEach(([key, label], idx) => {
    const row = document.createElement("div");
    row.className = "step-item";
    row.dataset.stepItem = key;

    const btn = document.createElement("button");
    btn.className = "step-button";
    btn.type = "button";
    btn.dataset.step = key;
    btn.innerHTML = `
      <span class="step-index">${idx + 1}</span>
      <span>${label}</span>
      <span class="step-state" data-step-state="${key}">待运行</span>
    `;
    btn.addEventListener("click", () => runSteps([key], false));

    const mode = document.createElement("select");
    mode.className = "step-mode";
    mode.title = "选择该步骤导入已有中间结果或重新计算";
    mode.dataset.stepMode = key;
    mode.innerHTML = `
      <option value="recompute">重新计算</option>
      <option value="reuse">导入已有</option>
    `;
    mode.value = state.stepModes[key] || "recompute";
    mode.addEventListener("change", () => {
      state.stepModes[key] = mode.value;
      renderStepAvailability();
    });

    row.appendChild(btn);
    row.appendChild(mode);
    wrap.appendChild(row);
  });
}

function resetStepModesToRecompute() {
  state.stepModes = Object.fromEntries(STEPS.map(([key]) => [key, "recompute"]));
  document.querySelectorAll("[data-step-mode]").forEach((select) => {
    select.value = "recompute";
  });
}

function buildParamInputs() {
  const wrap = $("paramGrid");
  wrap.innerHTML = "";
  Object.entries(DEFAULT_PARAMS).forEach(([key, value]) => {
    const label = document.createElement("label");
    label.textContent = PARAM_LABELS[key] || key;
    label.title = key;
    let input;
    if (key === "junction_policy") {
      input = document.createElement("select");
      ["min_valid", "cap_min", "keep"].forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        input.appendChild(opt);
      });
      input.value = value;
    } else {
      input = document.createElement("input");
      input.type = "number";
      input.step = Number.isInteger(value) ? "1" : "0.01";
      input.value = value;
    }
    input.dataset.param = key;
    input.addEventListener("change", readParams);
    wrap.appendChild(label);
    wrap.appendChild(input);
  });
}

function readParams() {
  const next = { ...state.params };
  document.querySelectorAll("[data-param]").forEach((input) => {
    const key = input.dataset.param;
    if (key === "junction_policy") {
      next[key] = input.value;
    } else {
      const value = Number(input.value);
      next[key] = Number.isFinite(value) ? value : DEFAULT_PARAMS[key];
    }
  });
  state.params = next;
  return next;
}

async function createSession() {
  clearJob();
  resetStepModesToRecompute();
  resetManualSegmentationEditor();
  resetAnalysisRangeEditor();
  setBusy(true);
  try {
    state.mode = "batch";
    const rootFolder = $("batchRoot").value.trim();
    if (!rootFolder) throw new Error("请输入 patient 根目录或批量目录");
    const res = await fetchJson("/api/session", {
      mode: "batch",
      root_folder: rootFolder,
      stl_name: $("stlName").value.trim() || "vessel.stl",
    });
    const payload = await readResponse(res);
    state.session = payload.session;
    populatePatients();
    await refreshData();
    logLine(`Session ${state.session.id} loaded: ${state.session.patients?.length || 0} patient(s).`);
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
  }
}

function populatePatients() {
  const select = $("patientSelect");
  select.innerHTML = "";
  const patients = state.session?.patients || [];
  patients.forEach((patient) => {
    const opt = document.createElement("option");
    opt.value = patient.id;
    opt.textContent = patient.id;
    select.appendChild(opt);
  });
  if (patients[0]) select.value = patients[0].id;
  $("nextPatientBtn").disabled = patients.length <= 1;
}

async function nextPatient() {
  const select = $("patientSelect");
  if (!state.session || select.options.length <= 1) return;
  const nextIndex = (select.selectedIndex + 1) % select.options.length;
  select.selectedIndex = nextIndex;
  resetStepModesToRecompute();
  resetManualSegmentationEditor();
  resetAnalysisRangeEditor();
  await refreshData();
}

async function runSteps(steps, allPatients) {
  if (!state.session) {
    showError(new Error("请先载入输入"));
    return;
  }
  readParams();
  if (steps.some((step) => ["centerline", "smooth", "segment"].includes(step))) {
    resetManualSegmentationEditor();
    resetAnalysisRangeEditor();
  }
  clearJob();
  setBusy(true);
  try {
    const selected = $("patientSelect").value;
    const patientId = allPatients && state.session.mode === "batch" ? "all" : (selected === "all" ? null : selected);
    const res = await fetchJson("/api/run", {
      session_id: state.session.id,
      steps,
      params: state.params,
      step_modes: state.stepModes,
      patient_id: patientId,
      post_tips_mode: $("postTipsMode").value,
      export_png: $("exportPng").checked,
    });
    const payload = await readResponse(res);
    state.job = payload.job;
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
    const res = await fetch(`/api/job/${state.job.id}`);
    const payload = await res.json();
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
  const job = state.job;
  if (!job) return;
  const pct = job.total ? Math.round((job.completed / job.total) * 100) : 0;
  $("jobStatus").textContent = `${job.status} ${pct}% ${job.current || ""}`;
  $("logs").textContent = (job.logs || []).join("\n\n");
  $("logs").scrollTop = $("logs").scrollHeight;

  const selected = $("patientSelect").value;
  const patientResults = selected && job.results ? job.results[selected] : null;
  STEPS.forEach(([key]) => {
    const el = document.querySelector(`[data-step-state="${key}"]`);
    if (!el) return;
    if (patientResults && key in patientResults) {
      el.textContent = patientResults[key] ? "完成" : "失败";
      el.style.color = patientResults[key] ? "var(--ok)" : "var(--danger)";
    }
  });
}

async function refreshData() {
  if (!state.session) return;
  const selected = $("patientSelect").value;
  const patient = selected === "all" ? state.session.patients[0]?.id : selected;
  if (!patient) return;
  $("emptyState").classList.remove("hidden");
  try {
    const stride = $("sectionStride").value;
    const cacheBust = Date.now();
    const surfaceSections = state.layers.surfaceSections ? "1" : "0";
    const res = await fetch(
      `/api/session/${state.session.id}/data?patient=${encodeURIComponent(patient)}&section_stride=${stride}&surface_sections=${surfaceSections}&_=${cacheBust}`,
      { cache: "no-store" },
    );
    const data = await readResponse(res);
    state.data = data;
    reconcileCenterlineSelection();
    syncManualAssignmentsFromData();
    syncAnalysisRangesFromData();
    renderStepAvailability();
    renderScene();
    renderInspector();
    renderCenterlineEditControls();
    renderManualSegmentationControls();
    renderAnalysisRangeControls();
  } catch (err) {
    showError(err);
  }
}

function renderStepAvailability() {
  const status = state.data?.step_files || {};
  STEPS.forEach(([key]) => {
    const row = document.querySelector(`[data-step-item="${key}"]`);
    const select = document.querySelector(`[data-step-mode="${key}"]`);
    const step = status[key];
    if (!row || !select) return;
    const reuse = state.stepModes[key] === "reuse";
    const ready = Boolean(step?.ready);
    row.classList.toggle("reuse-mode", reuse);
    row.classList.toggle("missing-reuse", reuse && !ready);
    select.title = ready
      ? "已找到该步骤保存的中间结果"
      : "未找到该步骤需要的中间结果文件";
  });
}

function currentPatientId() {
  const selected = $("patientSelect").value;
  return selected === "all" ? state.session?.patients?.[0]?.id : selected;
}

function toggleCenterlineEdit() {
  if (!state.centerlineEdit.active) {
    state.manualSegment.active = false;
    state.analysisRange.active = false;
  }
  state.centerlineEdit.active = !state.centerlineEdit.active;
  if (state.centerlineEdit.active) {
    state.layers.rawCenterline = true;
    const rawToggle = document.querySelector('.layer-toggle[data-layer="rawCenterline"]');
    if (rawToggle) rawToggle.checked = true;
  } else {
    state.centerlineEdit.selected.clear();
  }
  renderScene();
  renderCenterlineEditControls();
  renderAnalysisRangeControls();
}

function clearCenterlineSelection() {
  state.centerlineEdit.selected.clear();
  renderScene();
  renderCenterlineEditControls();
}

function centerlineBranchColor(index) {
  return CENTERLINE_EDIT_COLORS[index % CENTERLINE_EDIT_COLORS.length];
}

function centerlineBranchLabel(branch, index) {
  return `#${index + 1} 端点 ${branch.endpoint_id} → 分叉 ${branch.junction_id} · ${fmt(branch.length_mm, 1)} mm`;
}

function reconcileCenterlineSelection() {
  const valid = new Set((state.data?.centerline_edit?.branches || []).map((item) => item.id));
  for (const id of Array.from(state.centerlineEdit.selected)) {
    if (!valid.has(id)) state.centerlineEdit.selected.delete(id);
  }
}

function renderCenterlineEditControls() {
  const branches = state.data?.centerline_edit?.branches || [];
  const selected = state.centerlineEdit.selected.size;
  const branchSelect = $("centerlineBranchSelect");
  const previousValue = branchSelect.value;
  branchSelect.innerHTML = "";
  if (branches.length) {
    branches.forEach((branch, index) => {
      const option = document.createElement("option");
      option.value = branch.id;
      option.textContent = centerlineBranchLabel(branch, index);
      option.style.color = centerlineBranchColor(index);
      branchSelect.appendChild(option);
    });
    const validPrevious = branches.some((branch) => branch.id === previousValue);
    branchSelect.value = validPrevious ? previousValue : branches[0].id;
  } else {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无可删分支";
    branchSelect.appendChild(option);
  }
  const chosenId = branchSelect.value;
  const chosenIndex = branches.findIndex((branch) => branch.id === chosenId);
  if (chosenIndex >= 0) {
    branchSelect.style.borderColor = centerlineBranchColor(chosenIndex);
  } else {
    branchSelect.style.borderColor = "";
  }
  $("centerlineEditBtn").classList.toggle("active", state.centerlineEdit.active);
  $("centerlineEditBtn").disabled = !state.session || !state.data?.centerlines?.raw;
  branchSelect.disabled = !state.centerlineEdit.active || branches.length === 0;
  $("centerlinePickBtn").disabled = !state.centerlineEdit.active || !chosenId;
  $("centerlinePickBtn").textContent = chosenId && state.centerlineEdit.selected.has(chosenId)
    ? "取消该段"
    : "选择删除";
  $("centerlineUndoBtn").disabled = !state.centerlineEdit.active || selected === 0;
  $("centerlineSaveBtn").disabled = !state.centerlineEdit.active || selected === 0;
  $("centerlineEditStatus").textContent = state.centerlineEdit.active
    ? `可删 ${branches.length} 段 · 已选 ${selected}`
    : `可删 ${branches.length} 段`;
}

function toggleSelectedCenterlineBranch() {
  if (!state.centerlineEdit.active) return;
  const branchId = $("centerlineBranchSelect").value;
  if (!branchId) return;
  toggleCenterlineBranchSelection(branchId);
}

function toggleCenterlineBranchSelection(branchId) {
  if (state.centerlineEdit.selected.has(branchId)) {
    state.centerlineEdit.selected.delete(branchId);
  } else {
    state.centerlineEdit.selected.add(branchId);
  }
  const branches = state.data?.centerline_edit?.branches || [];
  const branch = branches.find((item) => item.id === branchId);
  const index = branches.findIndex((item) => item.id === branchId);
  $("pickedInfo").innerHTML = branch
    ? `原始中心线分支<br>编号: #${index + 1}<br>端点: ${branch.endpoint_id}<br>分叉点: ${branch.junction_id}<br>长度: ${fmt(branch.length_mm, 2)} mm<br>状态: ${state.centerlineEdit.selected.has(branchId) ? "待删除" : "未选择"}`
    : "原始中心线分支";
  renderScene();
}

async function saveCenterlineDeletion() {
  if (!state.session || state.centerlineEdit.selected.size === 0) return;
  const patientId = currentPatientId();
  if (!patientId) return;
  setBusy(true);
  try {
    const branchIds = Array.from(state.centerlineEdit.selected);
    const res = await fetchJson("/api/centerline/delete-branches", {
      session_id: state.session.id,
      patient_id: patientId,
      branch_ids: branchIds,
    });
    const payload = await readResponse(res);
    const removed = payload.result?.removed_nodes ?? 0;
    const stale = payload.result?.removed_outputs || [];
    logLine(`Deleted ${branchIds.length} centerline branch(es), removed ${removed} node(s).`);
    if (stale.length) logLine(`Cleared derived outputs: ${stale.join(", ")}`);
    state.centerlineEdit.selected.clear();
    state.centerlineEdit.active = false;
    resetManualSegmentationEditor();
    state.stepModes.centerline = "reuse";
    const centerlineMode = document.querySelector('[data-step-mode="centerline"]');
    if (centerlineMode) centerlineMode.value = "reuse";
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
    renderCenterlineEditControls();
  }
}

function resetManualSegmentationEditor() {
  state.manualSegment.active = false;
  state.manualSegment.edits = new Map();
  state.manualSegment.dirty = false;
}

function syncManualAssignmentsFromData() {
  if (state.manualSegment.dirty) return;
  state.manualSegment.edits = new Map(
    (state.data?.manual_segmentation?.atomic_segments || []).map((segment) => [
      segment.id,
      {
        kept: Boolean(segment.kept),
        vessel: segment.vessel || "",
      },
    ]),
  );
}

function atomicSegments() {
  return state.data?.manual_segmentation?.atomic_segments || [];
}

function manualAssignment(segmentId) {
  return state.manualSegment.edits.get(segmentId) || { kept: true, vessel: "" };
}

function manualVesselColor(vessel) {
  const info = (state.data?.manual_segmentation?.vessels || []).find((item) => item.id === vessel);
  return info?.color || "#d9822b";
}

function manualSegmentLabel(segment, index) {
  return `#${index + 1} ${segment.start_id} -> ${segment.end_id} · ${fmt(segment.length_mm, 1)} mm`;
}

function toggleManualSegmentation() {
  state.manualSegment.active = !state.manualSegment.active;
  if (state.manualSegment.active) {
    state.centerlineEdit.active = false;
    state.centerlineEdit.selected.clear();
    state.analysisRange.active = false;
    state.layers.smoothCenterline = true;
    const smoothToggle = document.querySelector('.layer-toggle[data-layer="smoothCenterline"]');
    if (smoothToggle) smoothToggle.checked = true;
  }
  renderCenterlineEditControls();
  renderManualSegmentationControls();
  renderAnalysisRangeControls();
  renderScene();
}

function resetManualChanges() {
  state.manualSegment.dirty = false;
  syncManualAssignmentsFromData();
  renderManualSegmentationControls();
  renderScene();
}

function renderManualSegmentationControls() {
  const segments = atomicSegments();
  const select = $("manualAtomicSelect");
  const previous = select.value;
  select.innerHTML = "";
  if (!segments.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无原子段";
    select.appendChild(option);
  } else {
    segments.forEach((segment, index) => {
      const option = document.createElement("option");
      option.value = segment.id;
      option.textContent = manualSegmentLabel(segment, index);
      select.appendChild(option);
    });
    select.value = segments.some((segment) => segment.id === previous) ? previous : segments[0].id;
  }

  const vesselSelect = $("manualVesselSelect");
  vesselSelect.innerHTML = '<option value="">选择血管</option>';
  (state.data?.manual_segmentation?.vessels || []).forEach((vessel) => {
    if (state.data?.patient?.is_post_tips && ["lgv", "pgv"].includes(vessel.id)) return;
    const option = document.createElement("option");
    option.value = vessel.id;
    option.textContent = vessel.label;
    option.style.color = vessel.color;
    vesselSelect.appendChild(option);
  });

  const selectedId = select.value;
  const assignment = manualAssignment(selectedId);
  $("manualKeep").checked = Boolean(assignment.kept);
  vesselSelect.value = assignment.vessel || "";
  if (!assignment.kept) vesselSelect.value = "";

  const active = state.manualSegment.active;
  const kept = segments.filter((segment) => manualAssignment(segment.id).kept).length;
  const unassigned = segments.filter((segment) => {
    const edit = manualAssignment(segment.id);
    return edit.kept && !edit.vessel;
  }).length;
  $("manualSegmentBtn").classList.toggle("active", active);
  $("manualSegmentBtn").disabled = !state.session || !segments.length;
  select.disabled = !active || !segments.length;
  $("manualKeep").disabled = !active || !selectedId;
  vesselSelect.disabled = !active || !selectedId || !$("manualKeep").checked;
  $("manualResetBtn").disabled = !active || !state.manualSegment.dirty;
  $("manualSaveBtn").disabled = !active || !segments.length || unassigned > 0;
  $("manualStatus").textContent = active
    ? `原子段 ${segments.length} · 保留 ${kept} · 未归属 ${unassigned}`
    : `原子段 ${segments.length}`;
}

function updateManualAssignment() {
  if (!state.manualSegment.active) return;
  const segmentId = $("manualAtomicSelect").value;
  if (!segmentId) return;
  const kept = $("manualKeep").checked;
  const vessel = kept ? $("manualVesselSelect").value : "";
  state.manualSegment.edits.set(segmentId, { kept, vessel });
  state.manualSegment.dirty = true;
  renderManualSegmentationControls();
  renderScene();
}

async function saveManualSegmentation() {
  if (!state.session || !state.manualSegment.active) return;
  const patientId = currentPatientId();
  if (!patientId) return;
  const assignments = atomicSegments().map((segment) => ({
    id: segment.id,
    ...manualAssignment(segment.id),
  }));
  setBusy(true);
  try {
    const res = await fetchJson("/api/centerline/manual-segments", {
      session_id: state.session.id,
      patient_id: patientId,
      assignments,
    });
    const payload = await readResponse(res);
    const result = payload.result || {};
    logLine(`Manual segmentation saved: ${result.n_kept || 0}/${result.n_atomic_segments || 0} atomic segments retained.`);
    if (result.vessels?.length) logLine(`Assigned vessels: ${result.vessels.join(", ")}`);
    if (result.smoothed_vessels?.length) logLine(`Smoothed final vessels: ${result.smoothed_vessels.join(", ")}`);
    if (result.removed_outputs?.length) logLine(`Cleared derived outputs: ${result.removed_outputs.join(", ")}`);
    state.manualSegment.active = false;
    state.manualSegment.dirty = false;
    state.stepModes.segment = "reuse";
    const segmentMode = document.querySelector('[data-step-mode="segment"]');
    if (segmentMode) segmentMode.value = "reuse";
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
    renderManualSegmentationControls();
  }
}

function resetAnalysisRangeEditor() {
  state.analysisRange.active = false;
  state.analysisRange.ranges = new Map();
  state.analysisRange.dirty = false;
  state.analysisRange.suggestions = null;
  state.analysisRange.boundarySections = {};
}

function analysisVessels() {
  return Object.entries(state.data?.segments || {}).map(([id, segment]) => ({
    id,
    label: segment.label || id.toUpperCase(),
    color: segment.color,
    length_mm: Number(segment.length_mm || 0),
  }));
}

function syncAnalysisRangesFromData() {
  if (state.analysisRange.dirty) return;
  const saved = state.data?.analysis_regions?.ranges || {};
  state.analysisRange.ranges = new Map(
    analysisVessels().map((vessel) => {
      const range = saved[vessel.id] || {};
      return [vessel.id, {
        start_fraction: Number(range.start_fraction ?? 0),
        end_fraction: Number(range.end_fraction ?? 1),
        source: range.source || "full",
      }];
    }),
  );
}

function currentAnalysisRange(vesselId) {
  return state.analysisRange.ranges.get(vesselId) || {
    start_fraction: 0,
    end_fraction: 1,
    source: "full",
  };
}

function toggleAnalysisRange() {
  state.analysisRange.active = !state.analysisRange.active;
  if (state.analysisRange.active) {
    state.centerlineEdit.active = false;
    state.centerlineEdit.selected.clear();
    state.manualSegment.active = false;
    state.layers.smoothCenterline = true;
    const smoothToggle = document.querySelector('.layer-toggle[data-layer="smoothCenterline"]');
    if (smoothToggle) smoothToggle.checked = true;
  }
  renderCenterlineEditControls();
  renderManualSegmentationControls();
  renderAnalysisRangeControls();
  renderScene();
}

function resetAnalysisChanges() {
  state.analysisRange.dirty = false;
  state.analysisRange.suggestions = null;
  state.analysisRange.boundarySections = {};
  syncAnalysisRangesFromData();
  renderAnalysisRangeControls();
  renderScene();
}

function renderAnalysisRangeControls() {
  const vessels = analysisVessels();
  const select = $("analysisVesselSelect");
  const previous = select.value;
  select.innerHTML = "";
  if (!vessels.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "无血管";
    select.appendChild(option);
  } else {
    vessels.forEach((vessel) => {
      const option = document.createElement("option");
      option.value = vessel.id;
      option.textContent = vessel.label;
      option.style.color = vessel.color;
      select.appendChild(option);
    });
    select.value = vessels.some((vessel) => vessel.id === previous) ? previous : vessels[0].id;
  }
  const selected = vessels.find((vessel) => vessel.id === select.value);
  const range = currentAnalysisRange(select.value);
  const startPct = Math.round(range.start_fraction * 100);
  const endPct = Math.round(range.end_fraction * 100);
  $("analysisStart").value = String(startPct);
  $("analysisEnd").value = String(endPct);
  $("analysisStartValue").textContent = `${startPct}%`;
  $("analysisEndValue").textContent = `${endPct}%`;

  const active = state.analysisRange.active;
  $("analysisRangeBtn").classList.toggle("active", active);
  $("analysisRangeBtn").disabled = !state.session || !vessels.length;
  select.disabled = !active || !vessels.length;
  $("analysisSuggestBtn").disabled = !active || !vessels.length;
  $("analysisStart").disabled = !active || !selected;
  $("analysisEnd").disabled = !active || !selected;
  $("analysisResetBtn").disabled = !active || !state.analysisRange.dirty;
  $("analysisSaveBtn").disabled = !active || !vessels.length || !state.analysisRange.dirty;
  if (!selected) {
    $("analysisStatus").textContent = "未设置";
    return;
  }
  const length = selected.length_mm;
  const validLength = Math.max(0, (range.end_fraction - range.start_fraction) * length);
  const suffix = state.data?.analysis_regions?.saved && !state.analysisRange.dirty ? " | 已保存" : "";
  $("analysisStatus").textContent =
    `${selected.label}: ${fmt(range.start_fraction * length, 1)}-${fmt(range.end_fraction * length, 1)} mm | 有效 ${fmt(validLength, 1)} mm${suffix}`;
}

function updateAnalysisRange() {
  if (!state.analysisRange.active) return;
  const vessel = $("analysisVesselSelect").value;
  if (!vessel) return;
  let start = Number($("analysisStart").value) / 100;
  let end = Number($("analysisEnd").value) / 100;
  if (end - start < 0.02) {
    if (document.activeElement === $("analysisStart")) {
      start = Math.max(0, end - 0.02);
    } else {
      end = Math.min(1, start + 0.02);
    }
  }
  state.analysisRange.ranges.set(vessel, {
    start_fraction: start,
    end_fraction: end,
    source: "manual",
  });
  state.analysisRange.dirty = true;
  renderAnalysisRangeControls();
  renderScene();
}

async function suggestAnalysisRanges() {
  if (!state.session || !state.analysisRange.active) return;
  const patientId = currentPatientId();
  if (!patientId) return;
  setBusy(true);
  try {
    const res = await fetchJson("/api/analysis/suggest-ranges", {
      session_id: state.session.id,
      patient_id: patientId,
    });
    const payload = await readResponse(res);
    const result = payload.result || {};
    state.analysisRange.ranges = new Map(
      Object.entries(result.ranges || {}).map(([vessel, range]) => [vessel, range]),
    );
    state.analysisRange.suggestions = result;
    state.analysisRange.boundarySections = result.boundary_sections || {};
    state.analysisRange.dirty = true;
    state.layers.surfaceSections = true;
    const surfaceToggle = document.querySelector('.layer-toggle[data-layer="surfaceSections"]');
    if (surfaceToggle) surfaceToggle.checked = true;
    await refreshData();
    logLine("Generated effective-range suggestions from true surface-section stability metrics.");
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
    renderAnalysisRangeControls();
  }
}

async function saveAnalysisRanges() {
  if (!state.session || !state.analysisRange.active) return;
  const patientId = currentPatientId();
  if (!patientId) return;
  const ranges = analysisVessels().map((vessel) => ({
    vessel: vessel.id,
    ...currentAnalysisRange(vessel.id),
  }));
  setBusy(true);
  try {
    const res = await fetchJson("/api/analysis/save-ranges", {
      session_id: state.session.id,
      patient_id: patientId,
      ranges,
    });
    const payload = await readResponse(res);
    const cleared = payload.result?.removed_outputs || [];
    state.analysisRange.dirty = false;
    state.analysisRange.suggestions = null;
    state.analysisRange.boundarySections = {};
    state.stepModes.profiles = "recompute";
    state.stepModes.features = "recompute";
    document.querySelector('[data-step-mode="profiles"]').value = "recompute";
    document.querySelector('[data-step-mode="features"]').value = "recompute";
    logLine("Effective analysis ranges saved. Run Pointwise cross-sections and Feature extraction to update measurements.");
    if (cleared.length) logLine(`Cleared derived outputs: ${cleared.join(", ")}`);
    await refreshData();
  } catch (err) {
    showError(err);
  } finally {
    setBusy(false);
    renderAnalysisRangeControls();
  }
}

function renderScene() {
  if (!state.data || !window.Plotly) return;
  const traces = [];
  const data = state.data;
  const opacity = Number($("meshOpacity").value || 22) / 100;

  if (data.mesh && data.mesh.vertices && state.layers.mesh) {
    const vertices = data.mesh.vertices;
    const faces = data.mesh.faces || [];
    traces.push({
      type: "mesh3d",
      name: `STL mesh (${data.mesh.n_faces_rendered || faces.length} faces)`,
      x: vertices.map((v) => v[0]),
      y: vertices.map((v) => v[1]),
      z: vertices.map((v) => v[2]),
      i: faces.map((f) => f[0]),
      j: faces.map((f) => f[1]),
      k: faces.map((f) => f[2]),
      color: "#b8c3cc",
      opacity,
      hoverinfo: "skip",
      flatshading: false,
      lighting: { ambient: 0.55, diffuse: 0.8, specular: 0.1 },
    });
  }

  addCenterlineTrace(traces, data.centerlines?.raw, "原始中心线", "#6b7280", state.layers.rawCenterline || state.centerlineEdit.active);
  addCenterlineTrace(traces, data.centerlines?.smooth, "平滑中心线", "#111827", state.layers.smoothCenterline && !state.manualSegment.active && !state.analysisRange.active);
  addEditableCenterlineTraces(traces, data.centerline_edit?.branches || []);
  addManualSegmentTraces(traces, data.manual_segmentation?.atomic_segments || []);
  if (state.analysisRange.active) {
    addAnalysisRangeTraces(traces, data.segments || {});
    addAnalysisBoundarySectionTraces(traces, state.analysisRange.boundarySections);
  } else if (!state.manualSegment.active) {
    addSegmentTraces(traces, data.segments || {});
  }
  addBranchPointTrace(traces, data.branch_points || []);
  addFeaturePointTraces(traces, data.pointwise?.feature_points || {});
  addSectionTraces(traces, data.pointwise?.sampled_sections || {}, "sampledSections", "间隔截面", 2, 0.38);
  addSectionTraces(traces, data.pointwise?.surface_sections || {}, "surfaceSections", "表面相交截面", 4, 0.96);
  addNamedSectionTraces(traces, data.pointwise?.max_sections || {}, "maxSections", "最大截面", 6);
  addNamedSectionTraces(traces, data.pointwise?.mean_sections || {}, "meanSections", "平均截面", 4);
  addSurfaceNamedSectionTraces(traces, data.pointwise?.surface_max_sections || {}, "maxSections", "最大表面截面", 7);
  addSurfaceNamedSectionTraces(traces, data.pointwise?.surface_mean_sections || {}, "meanSections", "平均表面截面", 6);
  if (!state.manualSegment.active) addLabelTrace(traces, data.segments || {});

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
    uirevision: "ppg-vessel-workbench",
  };
  Plotly.react("viewer", traces, layout, { displaylogo: false, responsive: true, scrollZoom: true });
  $("emptyState").classList.add("hidden");
  const plot = $("viewer");
  if (typeof plot.removeAllListeners === "function") {
    plot.removeAllListeners("plotly_click");
  }
  plot.on("plotly_click", (event) => {
    const pt = event.points?.[0];
    if (handleManualSegmentClick(pt)) return;
    if (handleCenterlineEditClick(pt)) return;
    if (pt?.customdata) {
      $("pickedInfo").innerHTML = escapeHtml(String(pt.customdata)).replaceAll("\n", "<br>").replaceAll("&lt;br&gt;", "<br>");
    }
  });
  renderCenterlineEditControls();
}

function addEditableCenterlineTraces(traces, branches) {
  if (!state.centerlineEdit.active || !branches.length) return;
  branches.forEach((branch, index) => {
    const selected = state.centerlineEdit.selected.has(branch.id);
    const color = centerlineBranchColor(index);
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: selected ? `待删除 #${index + 1}` : `可删分支 #${index + 1}`,
      x: branch.x,
      y: branch.y,
      z: branch.z,
      customdata: branch.x.map(() => `centerline-edit:${branch.id}`),
      line: {
        color: selected ? "#b42318" : color,
        width: selected ? 12 : 9,
      },
      opacity: selected ? 0.98 : 0.88,
      hovertemplate: `#${index + 1}<br>端点 ${branch.endpoint_id} → 分叉点 ${branch.junction_id}<br>length: ${fmt(branch.length_mm, 2)} mm<extra></extra>`,
    });
  });
}

function addManualSegmentTraces(traces, segments) {
  if (!state.manualSegment.active || !segments.length) return;
  const selectedId = $("manualAtomicSelect").value;
  segments.forEach((segment, index) => {
    const assignment = manualAssignment(segment.id);
    const selected = selectedId === segment.id;
    const color = assignment.kept
      ? (assignment.vessel ? manualVesselColor(assignment.vessel) : "#d9822b")
      : "#9aa7b2";
    const vessel = assignment.vessel
      ? (state.data.manual_segmentation.vessels.find((item) => item.id === assignment.vessel)?.label || assignment.vessel)
      : "未归属";
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: `原子段 #${index + 1} ${assignment.kept ? vessel : "不保留"}`,
      x: segment.x,
      y: segment.y,
      z: segment.z,
      customdata: segment.x.map(() => `manual-segment:${segment.id}`),
      line: {
        color,
        width: selected ? 13 : 9,
        dash: assignment.kept ? "solid" : "dot",
      },
      opacity: assignment.kept ? 0.96 : 0.5,
      hovertemplate: `#${index + 1}<br>${segment.start_id} -> ${segment.end_id}<br>length: ${fmt(segment.length_mm, 2)} mm<br>${assignment.kept ? vessel : "不保留"}<extra></extra>`,
    });
  });
}

function handleManualSegmentClick(point) {
  const marker = point?.customdata;
  if (!state.manualSegment.active || typeof marker !== "string" || !marker.startsWith("manual-segment:")) {
    return false;
  }
  $("manualAtomicSelect").value = marker.slice("manual-segment:".length);
  renderManualSegmentationControls();
  renderScene();
  return true;
}

function handleCenterlineEditClick(point) {
  const marker = point?.customdata;
  if (!state.centerlineEdit.active || typeof marker !== "string" || !marker.startsWith("centerline-edit:")) {
    return false;
  }
  const branchId = marker.slice("centerline-edit:".length);
  $("centerlineBranchSelect").value = branchId;
  toggleCenterlineBranchSelection(branchId);
  return true;
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

function addCenterlineTrace(traces, line, name, color, visible) {
  if (!line || !visible) return;
  traces.push({
    type: "scatter3d",
    mode: "lines",
    name,
    x: line.x,
    y: line.y,
    z: line.z,
    line: { color, width: 3 },
    hoverinfo: "skip",
  });
}

function addSegmentTraces(traces, segments) {
  if (!state.layers.segments) return;
  Object.entries(segments).forEach(([key, seg]) => {
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: seg.label || key.toUpperCase(),
      x: seg.x,
      y: seg.y,
      z: seg.z,
      line: { color: seg.color, width: 8 },
      hovertemplate: `<b>${seg.label}</b><br>length: ${fmt(seg.length_mm, 2)} mm<br>tortuosity: ${fmt(seg.tortuosity, 4)}<br>mean curvature: ${fmt(seg.mean_curvature, 5)}<extra></extra>`,
    });
  });
}

function sliceSegmentByFraction(segment, start, end) {
  const points = (segment.x || []).map((x, index) => [x, segment.y[index], segment.z[index]]);
  if (points.length < 2 || end <= start) return null;
  const arc = [0];
  for (let index = 1; index < points.length; index += 1) {
    const dx = points[index][0] - points[index - 1][0];
    const dy = points[index][1] - points[index - 1][1];
    const dz = points[index][2] - points[index - 1][2];
    arc.push(arc[index - 1] + Math.sqrt(dx * dx + dy * dy + dz * dz));
  }
  const total = arc[arc.length - 1];
  if (!(total > 0)) return null;
  const at = (fraction) => {
    const distance = Math.max(0, Math.min(1, fraction)) * total;
    let index = 1;
    while (index < arc.length && arc[index] < distance) index += 1;
    index = Math.min(index, arc.length - 1);
    const prior = index - 1;
    const local = arc[index] > arc[prior] ? (distance - arc[prior]) / (arc[index] - arc[prior]) : 0;
    return points[prior].map((value, axis) => value + local * (points[index][axis] - value));
  };
  const startDistance = start * total;
  const endDistance = end * total;
  const sliced = [at(start)];
  points.forEach((point, index) => {
    if (arc[index] > startDistance && arc[index] < endDistance) sliced.push(point);
  });
  sliced.push(at(end));
  return {
    x: sliced.map((point) => point[0]),
    y: sliced.map((point) => point[1]),
    z: sliced.map((point) => point[2]),
  };
}

function addAnalysisRangeTraces(traces, segments) {
  if (!state.layers.segments) return;
  Object.entries(segments).forEach(([key, segment]) => {
    const range = currentAnalysisRange(key);
    const valid = sliceSegmentByFraction(segment, range.start_fraction, range.end_fraction);
    if (range.start_fraction > 0) {
      const excludedStart = sliceSegmentByFraction(segment, 0, range.start_fraction);
      if (excludedStart) traces.push({
        type: "scatter3d",
        mode: "lines",
        name: `${segment.label} 排除起端`,
        ...excludedStart,
        line: { color: "#7f8a96", width: 7, dash: "dot" },
        opacity: 0.7,
        hovertemplate: `<b>${segment.label}</b><br>Excluded junction transition<extra></extra>`,
      });
    }
    if (range.end_fraction < 1) {
      const excludedEnd = sliceSegmentByFraction(segment, range.end_fraction, 1);
      if (excludedEnd) traces.push({
        type: "scatter3d",
        mode: "lines",
        name: `${segment.label} 排除末端`,
        ...excludedEnd,
        line: { color: "#7f8a96", width: 7, dash: "dot" },
        opacity: 0.7,
        hovertemplate: `<b>${segment.label}</b><br>Excluded junction transition<extra></extra>`,
      });
    }
    if (valid) traces.push({
      type: "scatter3d",
      mode: "lines",
      name: `${segment.label} 有效分析区`,
      ...valid,
      line: { color: segment.color, width: 10 },
      hovertemplate: `<b>${segment.label}</b><br>Effective analysis range<br>${Math.round(range.start_fraction * 100)}%-${Math.round(range.end_fraction * 100)}%<extra></extra>`,
    });
  });
}

function addAnalysisBoundarySectionTraces(traces, sections) {
  Object.entries(sections || {}).forEach(([key, boundaries]) => {
    const segment = state.data?.segments?.[key];
    Object.entries(boundaries || {}).forEach(([side, contour]) => {
      traces.push({
        type: "scatter3d",
        mode: "lines",
        name: `${segment?.label || key.toUpperCase()} 建议${side === "start" ? "起点" : "终点"}`,
        x: contour.x,
        y: contour.y,
        z: contour.z,
        line: { color: segment?.color || "#177e89", width: 8, dash: "dash" },
        opacity: 1,
        hovertemplate: `Automatic suggested boundary: ${Math.round(contour.fraction * 100)}%<extra></extra>`,
      });
    });
  });
}

function addBranchPointTrace(traces, branchPoints) {
  if (!state.layers.branchPoints || !branchPoints.length) return;
  traces.push({
    type: "scatter3d",
    mode: "markers",
    name: "分叉点",
    x: branchPoints.map((p) => p.coord?.[0]),
    y: branchPoints.map((p) => p.coord?.[1]),
    z: branchPoints.map((p) => p.coord?.[2]),
    marker: { size: 5, color: "#111820", line: { color: "#fff", width: 1 } },
    text: branchPoints.map((p) => `BP ${p.id}`),
    hovertemplate: "%{text}<extra></extra>",
  });
}

function addFeaturePointTraces(traces, featurePoints) {
  if (!state.layers.featurePoints) return;
  let colorbarShown = false;
  Object.entries(featurePoints).forEach(([key, fp]) => {
    if (!fp.x?.length) return;
    traces.push({
      type: "scatter3d",
      mode: "markers",
      name: `${fp.label} 曲率点`,
      x: fp.x,
      y: fp.y,
      z: fp.z,
      customdata: fp.hover,
      marker: {
        size: fp.size,
        color: fp.curvature,
        colorscale: "Viridis",
        opacity: 0.86,
        colorbar: colorbarShown ? undefined : { title: "curvature", thickness: 12 },
        showscale: !colorbarShown,
        line: { width: 0 },
      },
      hovertemplate: "%{customdata}<extra></extra>",
    });
    colorbarShown = true;
  });
}

function addSectionTraces(traces, sections, layerKey, label, width, opacity) {
  if (!state.layers[layerKey]) return;
  Object.entries(sections).forEach(([key, sec]) => {
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: `${sec.label} ${label}`,
      x: sec.x,
      y: sec.y,
      z: sec.z,
      line: { color: sec.color, width },
      opacity,
      hoverinfo: "skip",
    });
  });
}

function addNamedSectionTraces(traces, sections, layerKey, label, width) {
  if (!state.layers[layerKey]) return;
  Object.entries(sections).forEach(([key, sec]) => {
    const color = state.data.segments?.[key]?.color || "#177e89";
    const segLabel = state.data.segments?.[key]?.label || key.toUpperCase();
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: `${segLabel} ${label}`,
      x: sec.x,
      y: sec.y,
      z: sec.z,
      line: { color, width, dash: layerKey === "meanSections" ? "dash" : "solid" },
      hovertemplate: `<b>${segLabel} ${label}</b><br>point: ${sec.index}<br>diameter: ${fmt(sec.diameter, 3)} mm<br>area: ${fmt(sec.area, 3)} mm²<extra></extra>`,
    });
  });
}

function addSurfaceNamedSectionTraces(traces, sections, layerKey, label, width) {
  if (!state.layers.surfaceSections || !state.layers[layerKey]) return;
  Object.entries(sections).forEach(([key, sec]) => {
    const color = state.data.segments?.[key]?.color || "#177e89";
    const segLabel = state.data.segments?.[key]?.label || key.toUpperCase();
    traces.push({
      type: "scatter3d",
      mode: "lines",
      name: `${segLabel} ${label}`,
      x: sec.x,
      y: sec.y,
      z: sec.z,
      line: { color, width },
      opacity: 0.98,
      hovertemplate: `<b>${segLabel} ${label}</b><br>point: ${sec.index}<br>diameter: ${fmt(sec.diameter, 3)} mm<br>area: ${fmt(sec.area, 3)} mm²<extra></extra>`,
    });
  });
}

function addLabelTrace(traces, segments) {
  if (!state.layers.labels) return;
  const x = [];
  const y = [];
  const z = [];
  const text = [];
  Object.values(segments).forEach((seg) => {
    if (!seg.midpoint) return;
    x.push(seg.midpoint[0]);
    y.push(seg.midpoint[1]);
    z.push(seg.midpoint[2]);
    text.push(`<b>${seg.label}</b>`);
  });
  if (!x.length) return;
  traces.push({
    type: "scatter3d",
    mode: "text",
    name: "标签",
    x,
    y,
    z,
    text,
    textfont: { size: 16, color: "#17212b" },
    showlegend: false,
    hoverinfo: "skip",
  });
}

function renderInspector() {
  const data = state.data;
  renderSegmentCards(data.features?.statistical || {}, data.segments || {});
  renderSystemFeatures(data.features || {});
}

function renderSegmentCards(stats, segments) {
  const wrap = $("segmentCards");
  wrap.innerHTML = "";
  const names = Object.keys(segments);
  if (!names.length) {
    wrap.textContent = "暂无分段结果";
    return;
  }
  names.forEach((name) => {
    const seg = segments[name];
    const block = stats[name] || {};
    const card = document.createElement("article");
    card.className = "segment-card";
    card.style.borderLeftColor = seg.color || "#177e89";
    card.innerHTML = `
      <h3>${seg.label}</h3>
      ${metricRow("长度", block.length ?? seg.length_mm, "mm")}
      ${metricRow("平均直径", block.mean_diameter, "mm")}
      ${metricRow("最大直径", block.max_diameter, "mm")}
      ${metricRow("平均面积", block.mean_area, "mm²")}
      ${metricRow("最大曲率", block.max_curvature, "1/mm")}
    `;
    wrap.appendChild(card);
  });
}

function metricRow(label, value, unit) {
  return `<div class="metric"><span>${label}</span><strong>${fmt(value, 3)} ${unit}</strong></div>`;
}

function renderSystemFeatures(features) {
  const wrap = $("systemFeatures");
  wrap.innerHTML = "";
  const flat = flattenSystem(features.system || {});
  const global = features.global || {};
  const rows = [
    ...Object.entries(global).map(([key, value]) => [key, value]),
    ...Object.entries(flat).map(([key, value]) => [key, value]),
  ].filter(([, value]) => value !== null && value !== undefined && value !== "");
  if (!rows.length) {
    wrap.textContent = "暂无系统特征";
    return;
  }
  rows.slice(0, 80).forEach(([key, value]) => {
    const row = document.createElement("div");
    row.className = "feature-row";
    row.innerHTML = `<span title="${escapeHtml(key)}">${escapeHtml(key)}</span><strong>${escapeHtml(formatValue(value))}</strong>`;
    wrap.appendChild(row);
  });
}

function flattenSystem(system) {
  if (system.all_values && typeof system.all_values === "object") return system.all_values;
  if (system.available && typeof system.available === "object") {
    return { ...system.available, ...(system.unavailable || {}) };
  }
  return system;
}

function formatValue(value) {
  if (typeof value === "number") return fmt(value, 4);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return `[${value.slice(0, 3).map((v) => fmt(v, 3)).join(", ")}${value.length > 3 ? ", ..." : ""}]`;
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "NA";
  const n = Number(value);
  if (!Number.isFinite(n)) return "NA";
  return n.toFixed(digits);
}

async function downloadResults() {
  if (!state.session) {
    showError(new Error("请先载入输入"));
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
    if (btn.id === "paramsBtn") return;
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
  if (!res.ok || payload.error) {
    throw new Error(payload.error || `HTTP ${res.status}`);
  }
  return payload;
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
