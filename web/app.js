const STAGES = [
  { key: "segmentation", index: 1, label: "CT 分割", title: "CT 分割 - CT 到门静脉 STL", sub: "CT -> STL" },
  { key: "features", index: 2, label: "几何特征", title: "几何特征 - 中心线、分段与截面", sub: "STL -> 特征" },
  { key: "pvp", index: 3, label: "PVP 预测", title: "PVP 预测 - 物理先验约束网络", sub: "特征 -> PVP" },
];
const FEATURE_LABELS = {
  total_centerline_length: "总中心线长度",
  sv_smv_diameter_ratio: "SV/SMV 直径比",
  sv_smv_angle: "SV-SMV 夹角",
  angle_sv_smv: "SV-SMV 汇合角",
  confluence_murray3_deviation: "汇合 Murray³ 偏离",
  inflow_resistance_asymmetry: "入流阻力不对称",
  collateral_burden_score: "侧支负担评分",
  splenic_dominance_index: "脾主导指数",
};
const STAGE_BY_KEY = Object.fromEntries(STAGES.map((stage) => [stage.key, stage]));
const CENTERLINE_LAYER_CONTROLS = [
  ["mesh", "模型", true],
  ["rawCenterline", "原始线", false],
  ["smoothCenterline", "平滑线", true],
  ["segments", "分段", true],
  ["branchPoints", "分叉点", false],
  ["featurePoints", "特征点", true],
  ["sampledSections", "间隔截面", true],
  ["surfaceSections", "表面相交截面", false],
  ["maxSections", "最大截面", true],
  ["meanSections", "平均截面", true],
  ["labels", "标签", true],
];
const state = {
  stage: "segmentation",
  session: null,
  patients: [],
  currentPatient: null,
  jobs: [],
  legacy: {},
  pollTimer: null,
  segmentationViewer: null,
  segmentationParamsOpen: false,
  segmentationOpacityTarget: "",
  segmentationLayerOpacity: {},
  viewerToken: 0,
  centerlineLayers: Object.fromEntries(CENTERLINE_LAYER_CONTROLS.map(([key, , checked]) => [key, checked])),
};

const $ = (id) => document.getElementById(id);

function init() {
  $("loadBtn").addEventListener("click", loadSession);
  $("patientSearch").addEventListener("input", renderPatientList);
  $("stepper").querySelectorAll(".step").forEach((button) => {
    button.addEventListener("click", () => setStage(button.dataset.stage));
  });
  hydrateDefaults();
  applyQueryBootstrap();
  checkHealth();
  setStage("segmentation");
}

function hydrateDefaults() {
  const rememberedRoot = localStorage.getItem("portaflow.rootFolder") || "";
  $("rootFolder").value = rememberedRoot;
  $("modelDir").value = "";
}

function applyQueryBootstrap() {
  const params = new URLSearchParams(window.location.search);
  const root = params.get("root");
  const model = params.get("model_dir");
  if (root) $("rootFolder").value = root;
  if (model) $("modelDir").value = model;
  if (root && params.get("autoload") === "1") {
    window.setTimeout(() => loadSession(), 150);
  }
}

async function checkHealth() {
  try {
    const data = await api("/api/health");
    $("healthChip").textContent = data.ok ? "online" : "offline";
    $("healthChip").title = `Python: ${data.runtime?.python || ""}`;
  } catch (error) {
    $("healthChip").textContent = "offline";
    $("healthChip").title = error.message;
  }
}

async function loadSession() {
  const root = $("rootFolder").value.trim();
  const model = $("modelDir").value.trim();
  if (!root) {
    setLoadNote("请先输入数据根目录。", true);
    return;
  }
  $("loadBtn").disabled = true;
  setLoadNote("正在扫描病人目录...");
  try {
    const payload = { root_folder: root };
    if (model) payload.model_dir = model;
    const data = await api("/api/session", { method: "POST", body: payload });
    state.session = data.session;
    state.legacy = {};
    localStorage.setItem("portaflow.rootFolder", root);
    $("sessionBadge").textContent = state.session.id.split("-").slice(0, 2).join("-");
    setLoadNote("");
    await refreshPatients();
    const firstPatientId = state.patients[0]?.id;
    selectPatient(firstPatientId);
    if (firstPatientId) await refreshPatients(firstPatientId);
  } catch (error) {
    setLoadNote(error.message, true);
  } finally {
    $("loadBtn").disabled = false;
  }
}

function setLoadNote(message, isError = false) {
  $("loadNote").textContent = message;
  $("loadNote").classList.toggle("error", isError);
}

async function refreshPatients(patientId = null) {
  if (!state.session) return;
  const target = patientId || "all";
  const data = await api(`/api/session/${encodeURIComponent(state.session.id)}/data?patient=${encodeURIComponent(target)}`);
  if (patientId) {
    const updated = data.patients[0];
    state.patients = state.patients.map((item) => item.id === patientId ? updated : item);
    if (state.currentPatient?.id === patientId) state.currentPatient = updated;
  } else {
    state.patients = data.patients;
    if (state.currentPatient) {
      state.currentPatient = state.patients.find((item) => item.id === state.currentPatient.id) || state.patients[0] || null;
    }
  }
  renderPatientList();
  renderAll();
}

function selectPatient(patientId) {
  if (!patientId) return;
  const previousId = state.currentPatient?.id || "";
  state.currentPatient = state.patients.find((patient) => patient.id === patientId) || null;
  if (state.currentPatient?.id && state.currentPatient.id !== previousId) {
    delete state.legacy.segmentation;
    delete state.legacy.features;
  }
  renderPatientList();
  renderAll();
}

function setStage(stageKey) {
  state.stage = stageKey;
  renderAll();
}

function renderAll() {
  document.body.classList.toggle("legacy-active", state.stage === "features");
  renderTopbar();
  renderTools();
  renderStage();
  renderRightPanel();
  renderFoot();
}

function renderTopbar() {
  const patient = state.currentPatient;
  $("patientChip").classList.toggle("set", Boolean(patient));
  $("patientChipName").textContent = patient ? patient.id : "未选择病人";
  $("vpTitle").textContent = STAGE_BY_KEY[state.stage].title;

  $("stepper").querySelectorAll(".step").forEach((button) => {
    const key = button.dataset.stage;
    const status = patient?.status?.stages?.[key]?.status || "missing";
    button.classList.toggle("active", key === state.stage);
    button.classList.toggle("done", status === "done" && key !== state.stage);
    button.title = `${STAGE_BY_KEY[key].label}: ${statusLabel(status)}`;
  });
}

function renderPatientList() {
  const query = $("patientSearch").value.trim().toLowerCase();
  const list = state.patients.filter((patient) => {
    const haystack = `${patient.id} ${patientMetaText(patient)}`.toLowerCase();
    return haystack.includes(query);
  });
  $("patientCount").textContent = `个数 ${state.patients.length}`;
  const host = $("patientList");
  if (!state.session) {
    host.innerHTML = `<li class="list-empty">请先加载数据根目录</li>`;
    return;
  }
  if (!list.length) {
    host.innerHTML = `<li class="list-empty">没有匹配的病人</li>`;
    return;
  }
  host.innerHTML = list.map((patient) => patientCard(patient)).join("");
  host.querySelectorAll("[data-patient]").forEach((item) => {
    item.addEventListener("click", () => selectPatient(item.dataset.patient));
  });
}

function patientCard(patient) {
  const stages = patient.status?.stages || {};
  const pred = patient.status?.prediction;
  const mean = numeric(pred?.pvp_mean);
  const risk = pressureRisk(mean);
  const score = mean == null ? "PVP" : `${mean.toFixed(1)}<small> mmHg</small>`;
  const active = state.currentPatient?.id === patient.id ? " active" : "";
  const meta = patientMetaText(patient);
  return `<li class="patient-card${active}" data-patient="${escapeAttr(patient.id)}">
    <div class="pc-name">${escapeHtml(patient.id)}</div>
    <div class="pc-pvp ${mean == null ? "neutral" : `risk-${risk.key}`}">${score}</div>
    ${meta ? `<div class="pc-id">${escapeHtml(meta)}</div>` : ""}
    <div class="pc-tags">
      ${STAGES.map((stage) => statusPill(stage.index, stages[stage.key]?.status)).join("")}
    </div>
  </li>`;
}

function patientMetaText(patient) {
  const meta = patient.label_meta || {};
  const items = [];
  if (meta.age) items.push(`${meta.age}岁`);
  if (meta.sex) items.push(formatSex(meta.sex));
  if (meta.primary_disease) items.push(formatLabelValue(meta.primary_disease));
  if (meta.symptoms) items.push(formatLabelValue(meta.symptoms));
  if (meta.shunt_type) items.push(formatLabelValue(meta.shunt_type));
  return items.filter(Boolean).join(" · ");
}

function statusPill(index, status = "missing") {
  return `<span class="tag ${status}">${index} ${statusLabel(status)}</span>`;
}

function renderTools() {
  const patient = state.currentPatient;
  const stage = state.stage;
  const status = patient?.status?.stages?.[stage]?.status || "missing";
  const segmentationTools = "";
  const centerlineLayerTools = stage === "features" ? `
    <span class="centerline-layer-tools" aria-label="中心线图层显示">
      ${CENTERLINE_LAYER_CONTROLS.map(([key, label]) => `
        <label class="centerline-layer-chip">
          <input class="centerline-layer-toggle" data-centerline-layer="${key}" type="checkbox" ${state.centerlineLayers[key] ? "checked" : ""} />
          <span>${label}</span>
        </label>
      `).join("")}
    </span>
  ` : "";
  $("vpTools").innerHTML = `
    ${segmentationTools}
    ${centerlineLayerTools}
    <button class="chip-btn on" id="runStageBtn">${STAGE_BY_KEY[stage].label}运行</button>
    <button class="chip-btn" id="refreshBtn">刷新</button>
    <button class="chip-btn" id="downloadBtn">导出</button>
    <span class="status-badge ${status}">${statusLabel(status)}</span>
  `;
  $("runStageBtn").addEventListener("click", () => runStage(stage));
  $("refreshBtn").addEventListener("click", () => refreshPatients(patient?.id));
  $("downloadBtn").addEventListener("click", downloadCurrent);
  $("vpTools").querySelectorAll("[data-seg-tool]").forEach((button) => {
    button.addEventListener("click", () => sendSegmentationTool(button.dataset.segTool, button));
  });
  $("vpTools").querySelectorAll("[data-centerline-layer]").forEach((input) => {
    input.addEventListener("change", () => {
      state.centerlineLayers[input.dataset.centerlineLayer] = input.checked;
      sendCenterlineLayers();
    });
  });
  if (stage === "features") window.setTimeout(sendCenterlineLayers, 0);
}

function sendSegmentationTool(toolName, button) {
  const frame = document.querySelector(".legacy-frame");
  if (!frame?.contentWindow || state.stage !== "segmentation") return;
  const isActive = button.classList.contains("active");
  document.querySelectorAll("[data-seg-tool]").forEach((item) => item.classList.remove("active"));
  button.classList.toggle("active", !isActive);
  frame.contentWindow.postMessage({
    source: "portaflow-workbench",
    type: "segmentation-tool",
    tool: isActive ? null : toolName,
  }, "*");
}

function sendCenterlineLayers() {
  const frame = document.querySelector(".legacy-frame");
  if (!frame?.contentWindow || state.stage !== "features") return;
  frame.contentWindow.postMessage({
    source: "portaflow-workbench",
    type: "centerline-layers",
    layers: state.centerlineLayers,
  }, "*");
}

function renderStage() {
  const patient = state.currentPatient;
  const host = $("stageBody");
  if (!state.session) {
    host.innerHTML = emptyStage("等待数据", "输入数据根目录并加载病人后，这里会展示三阶段的真实输出。");
    return;
  }
  if (!patient) {
    host.innerHTML = emptyStage("请选择病人", "左侧病人列表为空或尚未选择病人。");
    return;
  }
  if (state.stage === "segmentation") {
    host.innerHTML = segmentationStage(patient);
    initSegmentationViewer(patient);
  }
  if (state.stage === "features") {
    host.innerHTML = legacyStage(patient, "features");
    bindLegacyControls();
    ensureLegacyStage("features");
  }
  if (state.stage === "pvp") host.innerHTML = pvpCompactClinicalStage(patient);
}

function legacyStage(patient, stageKey) {
  const legacy = state.legacy[stageKey];
  const label = stageKey === "segmentation" ? "Segmentation Workbench" : "Centerline Workbench";
  if (!legacy || legacy.status === "loading") {
    return `<div class="legacy-loading">
      <div class="spinner"></div>
      <h3>Loading ${label}</h3>
      <p>Loading the original interactive module.</p>
    </div>`;
  }
  if (legacy.status === "error") {
    return `<div class="legacy-loading error">
      <h3>${label} failed to start</h3>
      <p>${escapeHtml(legacy.error || "Unknown error")}</p>
      <button class="btn-run legacy-retry" data-legacy-retry="${stageKey}">Retry</button>
    </div>`;
  }
  return `<div class="legacy-shell">
    <iframe class="legacy-frame" title="${label}" src="${escapeAttr(legacy.iframe_url)}"></iframe>
  </div>`;
}

function bindLegacyControls() {
  const legacyFrame = document.querySelector(".legacy-frame");
  if (legacyFrame && state.stage === "features") {
    legacyFrame.addEventListener("load", () => sendCenterlineLayers(), { once: true });
  }
  document.querySelectorAll("[data-legacy-retry]").forEach((button) => {
    button.addEventListener("click", () => {
      const stage = button.dataset.legacyRetry;
      delete state.legacy[stage];
      renderStage();
    });
  });
  document.querySelectorAll("[data-open-legacy]").forEach((button) => {
    button.addEventListener("click", () => window.open(button.dataset.openLegacy, "_blank"));
  });
}

async function ensureLegacyStage(stageKey) {
  if (!state.session || !["segmentation", "features"].includes(stageKey)) return;
  const cached = state.legacy[stageKey];
  const patient = state.currentPatient?.id || "";
  if (cached && cached.status !== "error" && cached.patientId === patient) return;
  state.legacy[stageKey] = { status: "loading" };
  try {
    const root = state.session.root || $("rootFolder").value.trim();
    const data = await api(`/api/legacy-workbench?stage=${encodeURIComponent(stageKey)}&root_folder=${encodeURIComponent(root)}&patient=${encodeURIComponent(patient)}`);
    state.legacy[stageKey] = { status: "ready", patientId: patient, ...data.workbench };
  } catch (error) {
    state.legacy[stageKey] = { status: "error", patientId: patient, error: error.message };
  }
  if (state.stage === stageKey) renderStage();
}

function segmentationStage(patient) {
  const files = patient.status.files || {};
  const primary = bestSegmentationFile(files);
  const layerCount = segmentationLayers(patient).filter((layer) => layer.exists).length;
  return `<div class="seg-workspace">
    <div class="seg-viewer-card compact">
      <div class="seg-viewer" id="segViewer">
        <div id="segPlotly" class="seg-plotly"></div>
        <div class="seg-viewer-hud">
          <span>${escapeHtml(patient.id)}</span>
          <b>${primary ? escapeHtml(primary.label) : "未找到 STL"}</b>
          <em>${layerCount} 个 STL 图层 · ${statusLabel(patient.status.stages.segmentation.status)}</em>
        </div>
        <div class="seg-viewer-empty" id="segViewerEmpty">正在加载 STL 模型...</div>
      </div>
    </div>
  </div>`;
}

function bestSegmentationFile(files) {
  const candidates = [
    ["predict_smooth.stl", "Predict smooth"],
    ["predict.stl", "Predict"],
    ["pretrain.stl", "Pretrain"],
    ["vessel.stl", "Vessel"],
  ];
  const found = candidates.find(([name]) => files[name]?.exists);
  return found ? { file: found[0], label: found[1] } : null;
}

function featuresStage(patient) {
  const summary = patient.status.features_summary || {};
  const segments = summary.segments || [];
  const visUrl = patientFileUrl("vis_interactive.html");
  if (patient.status.preview?.vis_html) {
    return `<iframe class="stage-iframe" title="中心线与几何特征交互预览" src="${visUrl}"></iframe>`;
  }
  return `<div class="stage-grid">
    <div class="stage-card stage-image-card">
      <img class="stage-image" src="assets/portal-vein-hero.png" alt="门静脉中心线与几何分析视觉参考" />
      <div class="stage-overlay">
        <b>中心线/几何特征</b>
        <span>运行后会优先嵌入 vis_interactive.html 作为真实三维预览。</span>
      </div>
    </div>
    <div class="stage-card">
      <h3>分段摘要</h3>
      ${segments.length ? featureTable(segments) : `<p class="note">尚未生成 unified_features.json。</p>`}
    </div>
    <div class="stage-card">
      <h3>关键系统特征</h3>
      ${keyMetricList(summary.key_metrics || {})}
    </div>
  </div>`;
}

function pvpCompactClinicalStage(patient) {
  const pred = patient.status.prediction || {};
  const clinical = patient.status.clinical || {};
  const summary = patient.status.features_summary || {};
  const mean = numeric(pred.pvp_mean);
  const std = numeric(pred.pvp_std);
  const min = numeric(pred.pvp_min);
  const max = numeric(pred.pvp_max);
  const risk = pressureRisk(mean);
  return `<div class="pvp-compact-report">
    <section class="pvp-clinical-strip compact risk-${risk.key}">
      <div>
        <div class="strip-k">当前病例</div>
        <div class="strip-title">${escapeHtml(patient.id)}</div>
        <div class="strip-sub">${patientMetaText(patient) || "暂无临床标签"} · ${escapeHtml(clinical.timing || (patient.is_post_tips ? "术后" : "术前"))}</div>
      </div>
      <div class="strip-risk">
        <div class="strip-pvp"><span>${mean == null ? "--" : mean.toFixed(2)}</span><small>mmHg</small><b>${risk.label}</b></div>
        <div class="strip-model">std ${std == null ? "--" : std.toFixed(2)} · ${min == null ? "--" : min.toFixed(1)}-${max == null ? "--" : max.toFixed(1)}</div>
      </div>
    </section>

    <section class="stage-card pvp-compact-card info">
      <h3>病人信息</h3>
      ${clinicalInfoRows(patient)}
    </section>

    <section class="stage-card pvp-compact-card feature">
      <h3>关键几何特征</h3>
      ${pvpFeatureSummaryCompact(summary)}
    </section>

    <section class="stage-card pvp-compact-card compare">
      <h3>TIPS 前后压降</h3>
      ${pressureComparison(clinical, mean)}
    </section>

    <section class="stage-card pvp-compact-card explanation">
      <h3>预测解释</h3>
      <p>${predictionExplanationWithGeometry(mean, std, min, max, clinical, summary)}</p>
    </section>
  </div>`;
}

function pvpClinicalStage(patient) {
  const pred = patient.status.prediction || {};
  const clinical = patient.status.clinical || {};
  const summary = patient.status.features_summary || {};
  const mean = numeric(pred.pvp_mean);
  const std = numeric(pred.pvp_std);
  const min = numeric(pred.pvp_min);
  const max = numeric(pred.pvp_max);
  const risk = pressureRisk(mean);
  return `<div class="pvp-report">
    <section class="pvp-clinical-strip risk-${risk.key}">
      <div>
        <div class="strip-k">当前病例</div>
        <div class="strip-title">${escapeHtml(patient.id)}</div>
        <div class="strip-sub">${patientMetaText(patient) || "暂无临床标签"} · ${escapeHtml(clinical.timing || (patient.is_post_tips ? "术后" : "术前"))}</div>
      </div>
      <div class="strip-pvp"><span>${mean == null ? "--" : mean.toFixed(2)}</span><small>mmHg</small><b>${risk.label}</b></div>
    </section>

    <section class="stage-card pvp-result-card risk-${risk.key}">
      <h3>PVP 预测与风险</h3>
      <div class="pvp-value ${mean == null ? "muted" : ""}">
        <span class="num">${mean == null ? "--" : mean.toFixed(1)}</span><span class="unit">mmHg</span>
      </div>
      <div class="risk-band">${risk.label}</div>
      <div class="risk-scale"><span>低风险 &lt;16</span><span>升高 16-22</span><span>显著 ≥22</span></div>
      <div class="pvp-sub">fold std: <b>${std == null ? "--" : std.toFixed(2)}</b> · range: <b>${min == null ? "--" : min.toFixed(1)} - ${max == null ? "--" : max.toFixed(1)}</b></div>
    </section>

    <section class="stage-card pvp-model-card">
      <h3>门静脉模型</h3>
      ${pvpModelPreview(patient)}
    </section>

    <section class="stage-card pvp-info-card">
      <h3>病人信息</h3>
      ${clinicalInfoRows(patient)}
    </section>

    <section class="stage-card pvp-compare-card">
      <h3>TIPS 前后压降</h3>
      ${pressureComparison(clinical, mean)}
    </section>

    <section class="stage-card pvp-explain-card">
      <h3>预测解释</h3>
      <p>${predictionExplanation(mean, std, min, max, clinical)}</p>
    </section>

    <section class="stage-card pvp-feature-card">
      <h3>关键几何特征</h3>
      ${pvpFeatureSummary(summary)}
    </section>

    <section class="stage-card pvp-fold-card">
      <h3>模型一致性</h3>
      ${foldTable(pred.fold_predictions || [])}
      ${warningList(pred.warnings || [])}
    </section>
  </div>`;
}

function pvpStage(patient) {
  const pred = patient.status.prediction || {};
  const mean = numeric(pred.pvp_mean);
  const std = numeric(pred.pvp_std);
  const min = numeric(pred.pvp_min);
  const max = numeric(pred.pvp_max);
  return `<div class="pvp-layout">
    <div class="stage-card pvp-main">
      <div class="pvp-value ${mean == null ? "muted" : ""}">
        <span class="num">${mean == null ? "--" : mean.toFixed(1)}</span><span class="unit">mmHg</span>
      </div>
      <div class="pvp-band">${mean == null ? "等待推理" : pressureBand(mean)}</div>
      <div class="pvp-sub">fold std: <b>${std == null ? "--" : std.toFixed(2)}</b> · range: <b>${min == null ? "--" : min.toFixed(1)} - ${max == null ? "--" : max.toFixed(1)}</b></div>
    </div>
    <div class="stage-card">
      <h3>模型与 checkpoint</h3>
      <div class="kv"><span class="k">模型目录</span><span class="v wrap-value">${escapeHtml(pred.model_dir || state.session?.model_dir || "未找到")}</span></div>
      <div class="kv"><span class="k">模型权重</span><span class="v">${pred.n_checkpoints ?? "--"}</span></div>
      <div class="kv"><span class="k">设备</span><span class="v">${escapeHtml(pred.device || "auto")}</span></div>
      <p class="note">PVP 阶段调用 PVP_predictor/infer.py 做真实推理；结果会直接保存在当前 patient 目录下的 PVP_predict.txt，同时保留 pvp_prediction.json 给网页读取。</p>
    </div>
    <div class="stage-card">
      <h3>模型一致性</h3>
      ${foldTable(pred.fold_predictions || [])}
      ${warningList(pred.warnings || [])}
    </div>
  </div>`;
}

function renderRightPanel() {
  const patient = state.currentPatient;
  if (state.stage === "features") {
    $("rightPanel").innerHTML = "";
    return;
  }
  const host = $("rightPanel");
  if (!patient) {
    host.innerHTML = `<div class="section"><h3>工作台说明 <span class="h-line"></span></h3><p class="note">加载目录并选择病人后，右侧会出现当前阶段的运行与结果面板。</p></div>`;
    return;
  }
  if (state.stage === "segmentation") host.innerHTML = segmentationPanel(patient);
  if (state.stage === "features") host.innerHTML = featuresPanel(patient);
  if (state.stage === "pvp") host.innerHTML = pvpClinicalPanel(patient);
  host.querySelectorAll("[data-run-stage]").forEach((button) => {
    button.addEventListener("click", () => runStage(button.dataset.runStage));
  });
  host.querySelectorAll("[data-seg-pipeline]").forEach((button) => {
    button.addEventListener("click", () => {
      const mode = button.dataset.segMode;
      const select = document.querySelector("[data-seg-param='mode']");
      if (select && mode) select.value = mode;
      runStage("segmentation");
    });
  });
  host.querySelectorAll("[data-file]").forEach((button) => {
    button.addEventListener("click", () => openPatientFile(button.dataset.file));
  });
  host.querySelectorAll("[data-refresh-patient]").forEach((button) => {
    button.addEventListener("click", () => refreshPatients(patient?.id));
  });
  host.querySelectorAll("[data-seg-params-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      state.segmentationParamsOpen = !state.segmentationParamsOpen;
      renderRightPanel();
    });
  });
  host.querySelectorAll("[data-seg-layer]").forEach((input) => {
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("change", () => {
      state.segmentationViewer?.setVisible(input.dataset.segLayer, input.checked);
    });
  });
  host.querySelectorAll("[data-opacity-layer]").forEach((row) => {
    row.addEventListener("click", () => {
      if (row.classList.contains("missing")) return;
      state.segmentationOpacityTarget = row.dataset.opacityLayer;
      renderRightPanel();
    });
  });
  host.querySelectorAll("[data-seg-opacity]").forEach((input) => {
    input.addEventListener("input", () => {
      const layerId = state.segmentationOpacityTarget;
      const opacity = Number(input.value) / 100;
      if (!layerId || !Number.isFinite(opacity)) return;
      state.segmentationLayerOpacity[layerId] = opacity;
      state.segmentationViewer?.setOpacity(layerId, opacity);
    });
  });
}

function segmentationPanel(patient) {
  const stage = patient.status.stages.segmentation;
  const files = patient.status.files || {};
  const layers = segmentationLayers(patient);
  const opacityLayers = layers.filter((layer) => layer.kind === "organ" && layer.exists);
  const opacityTarget = state.segmentationOpacityTarget && opacityLayers.some((layer) => layer.id === state.segmentationOpacityTarget)
    ? state.segmentationOpacityTarget
    : opacityLayers[0]?.id || "";
  state.segmentationOpacityTarget = opacityTarget;
  const opacityLayer = opacityLayers.find((layer) => layer.id === opacityTarget);
  const opacityValue = Math.round((state.segmentationLayerOpacity[opacityTarget] ?? opacityLayer?.opacity ?? .24) * 100);
  const forceId = `force-${patient.id.replace(/[^a-z0-9_-]/gi, "")}`;
  const modelPath = state.session?.vkan_checkpoint || "未配置";
  const paramsHidden = state.segmentationParamsOpen ? "" : "hidden";
  const paramsToggleLabel = state.segmentationParamsOpen ? "收起参数" : "参数";
  return `<div class="section">
    <h3>CT 分割流水线 <span class="h-line"></span></h3>
    <div class="seg-action-row">
      <button class="btn-run" data-run-stage="segmentation">运行全流程</button>
      <button class="btn-ghost compact" data-refresh-patient>刷新</button>
    </div>
    <label class="check-row">
      <input id="${escapeAttr(forceId)}" type="checkbox" data-run-force />
      <span>强制重算已有结果</span>
    </label>
    <div class="pipeline-steps">
      ${pipelineStep(1, "TotalSegmentator 粗分割", "pretrain", files["orig.nii.gz"]?.exists || files["dcm"]?.exists, files["pretrain.stl"]?.exists)}
      ${pipelineStep(2, "nnVnet 门静脉精修", "predict", files["pretrain.stl"]?.exists, files["predict.stl"]?.exists)}
      ${pipelineStep(3, "平滑 / 填洞 / STL 质控", "smooth", files["predict.stl"]?.exists, files["predict_smooth.stl"]?.exists)}
    </div>
    <button class="btn-ghost seg-param-toggle ${state.segmentationParamsOpen ? "active" : ""}" type="button" data-seg-params-toggle>${paramsToggleLabel}</button>
    <div class="seg-params-section" ${paramsHidden}>
      <label class="param-row">
        <span>运行模式</span>
        <select data-seg-param="mode">
          <option value="auto">自动选择已有输入</option>
          <option value="pretrain">仅生成 pretrain</option>
          <option value="predict">从 pretrain 精修</option>
          <option value="smooth">仅平滑 / 填洞</option>
        </select>
      </label>
      <label class="param-row">
        <span>平滑迭代</span>
        <input data-seg-param="smooth_iterations" type="number" min="0" max="30" value="8" />
      </label>
      <label class="param-row">
        <span>模型参数位置</span>
        <input type="text" readonly value="${escapeAttr(modelPath)}" />
      </label>
    </div>
  </div>
  <div class="section">
    <h3>模型 / 器官图层 <span class="h-line"></span></h3>
    <div class="opacity-control">
      <label>
        <span>透明度 · ${opacityLayer ? escapeHtml(opacityLayer.label) : "未选择"}</span>
        <input data-seg-opacity type="range" min="5" max="80" value="${opacityValue}" ${opacityTarget ? "" : "disabled"} />
      </label>
    </div>
    <h4 class="layer-group-title">模型层</h4>
    <div class="layer-list">
      ${layers.filter((layer) => layer.kind === "vessel").map((layer) => layerControl(layer)).join("")}
    </div>
    <h4 class="layer-group-title">器官层</h4>
    <div class="layer-list">
      ${layers.filter((layer) => layer.kind === "organ").map((layer) => layerControl(layer)).join("")}
    </div>
  </div>`;
}

function pipelineStep(index, label, mode, ready, done) {
  const stateClass = done ? "done" : ready ? "ready" : "missing";
  const text = done ? "完成" : ready ? "可运行" : "等待输入";
  return `<button type="button" class="pipeline-step ${stateClass}" data-seg-pipeline data-seg-mode="${mode}">
    <span>${index}</span>
    <b>${escapeHtml(label)}</b>
    <em>${text}</em>
  </button>`;
}

function layerControl(layer) {
  const disabled = layer.exists ? "" : "disabled";
  const checked = layer.visible && layer.exists ? "checked" : "";
  const status = layer.exists ? "已找到" : "缺失";
  const canTuneOpacity = layer.kind === "organ" && layer.exists;
  const selected = canTuneOpacity && state.segmentationOpacityTarget === layer.id ? " selected" : "";
  const selectableClass = canTuneOpacity ? " selectable" : "";
  const opacityAttr = canTuneOpacity ? ` data-opacity-layer="${escapeAttr(layer.id)}"` : "";
  return `<div class="layer-row ${layer.exists ? "" : "missing"}${selected}${selectableClass}"${opacityAttr}>
    <input data-seg-layer="${escapeAttr(layer.id)}" type="checkbox" ${checked} ${disabled} />
    <span class="layer-swatch" style="background:${escapeAttr(layer.color)}"></span>
    <span class="layer-name">${escapeHtml(layer.label)}</span>
    <b>${status}</b>
  </div>`;
}

function segmentationLayers(patient) {
  const files = patient.status.files || {};
  const organs = patient.status.organs || {};
  const layers = [
    { id: "pretrain", label: "Pretrain", file: "pretrain.stl", color: "#38bdf8", opacity: .46, visible: Boolean(files["pretrain.stl"]?.exists), exists: Boolean(files["pretrain.stl"]?.exists), kind: "vessel" },
    { id: "predict", label: files["predict_smooth.stl"]?.exists ? "Predict smooth" : "Predict", file: files["predict_smooth.stl"]?.exists ? "predict_smooth.stl" : "predict.stl", color: "#10a66a", opacity: .78, visible: Boolean(files["predict_smooth.stl"]?.exists || files["predict.stl"]?.exists), exists: Boolean(files["predict_smooth.stl"]?.exists || files["predict.stl"]?.exists), kind: "vessel" },
    { id: "vessel", label: "Vessel label", file: "vessel.stl", color: "#047857", opacity: .60, visible: !files["predict_smooth.stl"]?.exists && !files["predict.stl"]?.exists && Boolean(files["vessel.stl"]?.exists), exists: Boolean(files["vessel.stl"]?.exists), kind: "vessel" },
  ];
  const organColors = {
    liver: "#a7c957",
    spleen: "#a78bfa",
    kidney: "#93c5fd",
    kidney_left: "#93c5fd",
    kidney_right: "#60a5fa",
    aorta: "#ef4444",
    inferior_vena_cava: "#60a5fa",
    bone: "#cbd5e1",
    bone_all: "#cbd5e1",
    portal_vein: "#16a34a",
  };
  Object.entries(organs).forEach(([name, info]) => {
    const key = String(name).toLowerCase();
    layers.push({
      id: `organ-${key}`,
      label: organLabel(key),
      file: `segmentation/${name}.stl`,
      color: organColors[key] || "#9ca3af",
      opacity: key === "portal_vein" ? .55 : key === "liver" ? .24 : .30,
      visible: ["liver", "spleen", "kidney", "kidney_left", "kidney_right", "aorta", "inferior_vena_cava", "portal_vein"].includes(key),
      exists: Boolean(info?.exists),
      kind: "organ",
    });
  });
  return layers;
}

function organLabel(key) {
  return {
    liver: "Liver",
    liver_left: "Liver L",
    liver_right: "Liver R",
    spleen: "Spleen",
    kidney: "Kidney",
    kidney_left: "Kidney L",
    kidney_right: "Kidney R",
    aorta: "Aorta",
    inferior_vena_cava: "IVC",
    bone: "Bone",
    bone_all: "Bone all",
    portal_vein: "Portal vein",
  }[key] || key;
}

function featuresPanel(patient) {
  const stage = patient.status.stages.features;
  const summary = patient.status.features_summary || {};
  return `<div class="section">
    <h3>中心线与几何特征 <span class="h-line"></span></h3>
    <div class="kv"><span class="k">阶段状态</span><span class="v">${statusLabel(stage.status)}</span></div>
    <div class="kv"><span class="k">血管段</span><span class="v">${(summary.segments || []).length || "--"}</span></div>
    <button class="btn-run" data-run-stage="features">提取中心线/几何特征</button>
    <button class="btn-ghost" data-file="vis_interactive.html">打开交互预览</button>
    <p class="note">默认使用 predict_smooth.stl；若不存在，回退到 vessel.stl。</p>
  </div>
  <div class="section">
    <h3>关键几何解释项 <span class="h-line"></span></h3>
    ${keyMetricList(summary.key_metrics || {})}
  </div>`;
}

function pvpClinicalPanel(patient) {
  const stage = patient.status.stages.pvp;
  const pred = patient.status.prediction || {};
  const clinical = patient.status.clinical || {};
  const risk = pressureRisk(pred.pvp_mean);
  return `<div class="section pvp-side-risk risk-${risk.key}">
    <h3>PVP 报告操作 <span class="h-line"></span></h3>
    <div class="kv"><span class="k">阶段状态</span><span class="v">${statusLabel(stage.status)}</span></div>
    <div class="kv"><span class="k">风险分层</span><span class="v">${risk.label}</span></div>
    <div class="kv"><span class="k">输入特征</span><span class="v">${patient.status.files?.["unified_features.json"]?.exists ? "已存在" : "缺失"}</span></div>
    <div class="kv"><span class="k">文本结果</span><span class="v">${patient.status.files?.["PVP_predict.txt"]?.exists ? "已生成" : "未生成"}</span></div>
    <button class="btn-run" data-run-stage="pvp">运行 PVP 推理</button>
    <button class="btn-ghost" data-file="PVP_predict.txt">打开 PVP_predict.txt</button>
    <button class="btn-ghost" data-file="pvp_prediction.json">打开预测 JSON</button>
  </div>
  <div class="section">
    <h3>技术详情 <span class="h-line"></span></h3>
    <div class="kv"><span class="k">模型目录</span><span class="v wrap-value">${escapeHtml(pred.model_dir || state.session?.model_dir || "未找到")}</span></div>
    <div class="kv"><span class="k">checkpoint</span><span class="v">${pred.n_checkpoints ?? "--"}</span></div>
    <div class="kv"><span class="k">设备</span><span class="v">${escapeHtml(pred.device || "auto")}</span></div>
    <div class="kv"><span class="k">检查日期</span><span class="v">${escapeHtml(clinical.exam_date || "--")}</span></div>
    <div class="kv"><span class="k">手术日期</span><span class="v">${escapeHtml(clinical.surgery_date || "--")}</span></div>
    <p class="note">主视图区展示临床摘要、风险解释、门静脉模型、术前术后压降和关键几何特征。</p>
  </div>`;
}

function pvpPanel(patient) {
  const stage = patient.status.stages.pvp;
  const pred = patient.status.prediction || {};
  return `<div class="section">
    <h3>PVP 真实推理 <span class="h-line"></span></h3>
    <div class="kv"><span class="k">阶段状态</span><span class="v">${statusLabel(stage.status)}</span></div>
    <div class="kv"><span class="k">输入特征</span><span class="v">${patient.status.files?.["unified_features.json"]?.exists ? "已存在" : "缺失"}</span></div>
    <div class="kv"><span class="k">模型有效</span><span class="v">${state.session?.model_valid ? "是" : "否"}</span></div>
    <div class="kv"><span class="k">文本结果</span><span class="v">${patient.status.files?.["PVP_predict.txt"]?.exists ? "已生成" : "未生成"}</span></div>
    <button class="btn-run" data-run-stage="pvp">运行 PVP 推理</button>
    <button class="btn-ghost" data-file="PVP_predict.txt">打开 PVP_predict.txt</button>
    <button class="btn-ghost" data-file="pvp_prediction.json">打开预测 JSON</button>
    <p class="note">当前默认模型：${escapeHtml(state.session?.model_dir || "未找到可推理模型")}</p>
  </div>
  <div class="section">
    <h3>预测摘要 <span class="h-line"></span></h3>
    <div class="metric-grid">
      <div class="metric"><div class="m-k">PVP mean</div><div class="m-v">${formatNum(pred.pvp_mean)}</div></div>
      <div class="metric"><div class="m-k">一致性 std</div><div class="m-v">${formatNum(pred.pvp_std)}</div></div>
      <div class="metric"><div class="m-k">min</div><div class="m-v">${formatNum(pred.pvp_min)}</div></div>
      <div class="metric"><div class="m-k">max</div><div class="m-v">${formatNum(pred.pvp_max)}</div></div>
    </div>
    ${warningList(pred.warnings || [])}
  </div>`;
}

function pressureRisk(value) {
  const n = numeric(value);
  if (n == null) return { key: "unknown", label: "等待推理", text: "暂无预测结果" };
  if (n >= 22) return { key: "high", label: "显著门脉高压", text: "≥22 mmHg" };
  if (n >= 16) return { key: "elevated", label: "PVP 升高", text: "16-22 mmHg" };
  return { key: "low", label: "低风险", text: "<16 mmHg" };
}

function clinicalInfoRows(patient) {
  const meta = patient.label_meta || {};
  const clinical = patient.status.clinical || {};
  return `<div class="clinical-grid">
    ${clinicalCell("年龄", meta.age ? `${escapeHtml(meta.age)} 岁` : "--")}
    ${clinicalCell("性别", meta.sex ? formatSex(meta.sex) : "--")}
    ${clinicalCell("原发病", meta.primary_disease ? formatLabelValue(meta.primary_disease) : "--")}
    ${clinicalCell("症状", meta.symptoms ? formatLabelValue(meta.symptoms) : "--")}
    ${clinicalCell("分流类型", meta.shunt_type ? formatLabelValue(meta.shunt_type) : "--", "wide")}
    ${clinicalCell("检查日期", clinical.exam_date || "--")}
    ${clinicalCell("手术日期", clinical.surgery_date || "--")}
    ${clinicalCell("时间关系", clinical.timing || (patient.is_post_tips ? "术后" : "术前"))}
  </div>`;
}

function clinicalCell(label, value, extra = "") {
  return `<div class="clinical-cell ${extra}"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`;
}

function pressureComparison(clinical, currentMean) {
  const current = numeric(currentMean ?? clinical.current_pvp);
  if (!clinical.is_post_tips) {
    return `<div class="compare-empty">当前病例判断为术前病例，后续术后病例生成后可在这里展示压降。</div>`;
  }
  if (!clinical.preop_match) {
    return `<div class="compare-empty">未在数据根目录中找到匹配的术前病例。匹配规则：去掉日期和 # 后病人名一致。</div>
      <div class="compare-current">${compareMetric("当前术后 PVP", current, "mmHg")}</div>`;
  }
  const pre = numeric(clinical.preop_match.pvp);
  const drop = numeric(clinical.pressure_drop);
  const pct = numeric(clinical.pressure_drop_pct);
  return `<div class="compare-grid">
    ${compareMetric("术前 PVP", pre, "mmHg")}
    ${compareMetric("术后 PVP", current, "mmHg")}
    ${compareMetric("压降", drop, "mmHg", drop == null ? "" : drop >= 0 ? "good" : "warn")}
    ${compareMetric("压降比例", pct, "%", pct == null ? "" : pct >= 0 ? "good" : "warn")}
  </div>
  <p class="note">匹配术前病例：${escapeHtml(clinical.preop_match.id || "--")}${clinical.preop_match.exam_date ? ` · ${escapeHtml(clinical.preop_match.exam_date)}` : ""}</p>`;
}

function compareMetric(label, value, unit, tone = "") {
  const n = numeric(value);
  const digits = unit === "%" ? 1 : 2;
  return `<div class="compare-metric ${tone}"><span>${escapeHtml(label)}</span><b>${n == null ? "--" : n.toFixed(digits)}<small>${escapeHtml(unit)}</small></b></div>`;
}

function predictionExplanation(mean, std, min, max, clinical) {
  const risk = pressureRisk(mean);
  if (numeric(mean) == null) return "当前病例尚未生成 PVP 预测结果。请先运行 PVP 推理。";
  const timing = clinical?.timing ? `当前病例为${clinical.timing}` : "";
  return `该病例预测 PVP 为 ${numeric(mean).toFixed(2)} mmHg，处于「${risk.label}」区间。${timing ? timing + "，" : ""}建议结合术前 PVP、术后随访和临床症状综合判断 TIPS 压降效果。`;
}

function predictionExplanationWithGeometry(mean, std, min, max, clinical, summary) {
  const n = numeric(mean);
  if (n == null) return "当前病例尚未生成 PVP 预测结果。请先运行 PVP 推理。";
  const risk = pressureRisk(n);
  const global = summary?.global || {};
  const segments = summary?.segments || [];
  const segById = Object.fromEntries(segments.map((seg) => [String(seg.id || "").toLowerCase(), seg]));
  const geom = [];
  const totalLength = numeric(global.total_centerline_length);
  const ratio = numeric(global.sv_smv_diameter_ratio);
  const angle = numeric(global.sv_smv_angle);
  const mpvDiameter = numeric(segById.mpv?.mean_diameter);
  const tipsDiameter = numeric(segById.tips?.mean_diameter);
  if (totalLength != null) geom.push(`门静脉中心线总长 ${totalLength.toFixed(1)} mm`);
  if (mpvDiameter != null) geom.push(`MPV 平均直径 ${mpvDiameter.toFixed(1)} mm`);
  if (tipsDiameter != null) geom.push(`TIPS 平均直径 ${tipsDiameter.toFixed(1)} mm`);
  if (ratio != null) geom.push(`SV/SMV 直径比 ${ratio.toFixed(2)}`);
  if (angle != null) geom.push(`SV-SMV 夹角 ${angle.toFixed(1)}°`);
  const timing = clinical?.timing ? `当前为${clinical.timing}` : "";
  const geomText = geom.length ? `几何上可见 ${geom.join("、")}，这些血管通径、汇合角度和支架相关特征共同参与了 PVP 估计。` : "当前几何摘要不足，建议先完成中心线和特征提取。";
  return `预测 PVP 为 ${n.toFixed(2)} mmHg，属于「${risk.label}」。${geomText}${timing ? timing + "，" : ""}建议结合术前压力、术后随访和出血/腹水等症状判断 TIPS 减压效果。`;
}

function pvpModelPreview(patient) {
  const preview = patient.status.preview || {};
  if (preview.vis_html) {
    return `<div class="model-frame-wrap"><iframe class="pvp-model-frame" title="门静脉模型交互预览" src="${patientFileUrl("vis_interactive.html")}"></iframe></div>
      <p class="note">可在模型窗口内拖拽旋转、缩放查看门静脉几何结构。</p>`;
  }
  if (preview.vis_png?.exists) {
    return `<img class="pvp-model-img" src="${patientFileUrl("vis_overview.png")}" alt="门静脉模型预览" />
      <p class="note">当前为静态预览图；生成 vis_interactive.html 后可交互旋转。</p>`;
  }
  const stl = preview.stl;
  return `<div class="model-empty">未找到交互模型预览</div>
    <p class="note">${stl?.exists ? `已找到 STL：${escapeHtml(shortPath(stl.path))}` : "请先完成分割和几何特征阶段。"}</p>`;
}

function pvpFeatureSummary(summary) {
  const segments = summary.segments || [];
  const global = summary.global || {};
  const presence = summary.vessel_presence || {};
  return `<div class="global-metrics">
    ${compareMetric("总中心线长度", global.total_centerline_length, "mm")}
    ${compareMetric("SV/SMV 直径比", global.sv_smv_diameter_ratio, "")}
    ${compareMetric("SV-SMV 夹角", global.sv_smv_angle, "°")}
    ${textMetric("TIPS", global.has_tips ? "存在" : "未见")}
  </div>
  ${vesselPresenceChips(presence)}
  ${segments.length ? pvpSegmentTable(segments) : `<p class="note">尚未生成血管几何特征。</p>`}`;
}

function pvpFeatureSummaryCompact(summary) {
  const segments = summary.segments || [];
  const global = summary.global || {};
  const presence = summary.vessel_presence || {};
  const core = segments.filter((seg) => ["mpv", "sv", "smv", "tips", "lpv", "rpv"].includes(String(seg.id || "").toLowerCase())).slice(0, 6);
  return `<div class="global-metrics compact">
    ${compareMetric("中心线总长", global.total_centerline_length, "mm")}
    ${compareMetric("SV/SMV 比", global.sv_smv_diameter_ratio, "")}
    ${compareMetric("SV-SMV 角", global.sv_smv_angle, "°")}
    ${textMetric("TIPS", global.has_tips ? "存在" : "未见")}
  </div>
  ${vesselPresenceChips(presence)}
  ${core.length ? pvpSegmentTableCompact(core) : `<p class="note">尚未生成血管几何特征。</p>`}`;
}

function pvpSegmentTableCompact(segments) {
  return `<table class="gtable pvp-seg-table compact">
    <thead><tr><th>血管</th><th>长度</th><th>直径</th><th>面积</th><th>曲率</th></tr></thead>
    <tbody>${segments.map((seg) => `<tr>
      <td class="vname">${escapeHtml(seg.label || seg.id)}</td>
      <td>${formatNum(seg.length, 1)}</td>
      <td>${formatNum(seg.mean_diameter, 1)}</td>
      <td>${formatNum(seg.mean_area, 1)}</td>
      <td>${formatNum(seg.max_curvature, 2)}</td>
    </tr>`).join("")}</tbody>
  </table>`;
}

function textMetric(label, value) {
  return `<div class="compare-metric"><span>${escapeHtml(label)}</span><b>${escapeHtml(value)}</b></div>`;
}

function vesselPresenceChips(presence) {
  const keys = ["mpv", "sv", "smv", "lpv", "rpv", "tips", "lgv", "pgv"];
  return `<div class="vessel-chips">${keys.map((key) => {
    const item = presence[key] || {};
    return `<span class="${item.present ? "on" : "off"}">${key.toUpperCase()}</span>`;
  }).join("")}</div>`;
}

function pvpSegmentTable(segments) {
  return `<table class="gtable pvp-seg-table">
    <thead><tr><th>血管</th><th>长度</th><th>平均直径</th><th>平均面积</th><th>最大曲率</th></tr></thead>
    <tbody>${segments.map((seg) => `<tr>
      <td class="vname">${escapeHtml(seg.label || seg.id)}</td>
      <td>${formatNum(seg.length)}</td>
      <td>${formatNum(seg.mean_diameter)}</td>
      <td>${formatNum(seg.mean_area)}</td>
      <td>${formatNum(seg.max_curvature, 3)}</td>
    </tr>`).join("")}</tbody>
  </table>`;
}

async function initSegmentationViewer(patient) {
  const plot = $("segPlotly");
  const empty = $("segViewerEmpty");
  if (!plot || !patient) return;
  const token = ++state.viewerToken;
  const rawLayers = segmentationLayers(patient).filter((layer) => layer.exists);
  const initialLayers = rawLayers.filter((layer) => layer.visible);
  const loadingLayers = new Set();
  const model = {
    layers: [],
    center: [0, 0, 0],
    radius: 1,
    rx: -0.28,
    ry: 0.62,
    zoom: 1,
    dragging: false,
    last: [0, 0],
    raf: 0,
    drawRaf: 0,
  };

  state.segmentationViewer = {
    setVisible(id, visible) {
      const layer = model.layers.find((item) => item.id === id);
      if (layer) {
        layer.visible = visible;
        renderSegmentationPlot(plot, model);
        return;
      }
      const layerDef = rawLayers.find((item) => item.id === id);
      if (visible && layerDef && !loadingLayers.has(id)) {
        loadingLayers.add(id);
        empty.hidden = false;
        empty.textContent = "正在加载所选 STL 图层...";
        loadSegmentationLayer(layerDef, token).then((loadedLayer) => {
          if (!loadedLayer || token !== state.viewerToken) return;
          model.layers.push(layerWithStoredOpacity({ ...loadedLayer, visible: true }));
          fitSegmentationModel(model);
          empty.hidden = model.layers.length > 0;
          renderSegmentationPlot(plot, model);
        }).finally(() => loadingLayers.delete(id));
      }
    },
    setOpacity(id, value) {
      const layer = model.layers.find((item) => item.id === id);
      if (layer && Number.isFinite(value)) {
        layer.opacity = value;
        renderSegmentationPlot(plot, model);
      }
    },
  };

  if (!rawLayers.length) {
    empty.textContent = "未找到 pretrain / predict / 器官 STL，请确认病人目录。";
    empty.hidden = false;
    renderSegmentationPlot(plot, model);
    return;
  }

  empty.hidden = false;
  empty.textContent = "正在加载 STL 模型...";
  try {
    const loaded = [];
    for (const layer of initialLayers) {
      if (token !== state.viewerToken) return;
      const loadedLayer = await loadSegmentationLayer(layer, token);
      if (loadedLayer) loaded.push(layerWithStoredOpacity(loadedLayer));
    }
    if (token !== state.viewerToken) return;
    model.layers = loaded;
    fitSegmentationModel(model);
    empty.hidden = loaded.length > 0;
    if (!loaded.length) empty.textContent = "STL 文件存在，但未能解析出三角面。";
    renderSegmentationPlot(plot, model);
  } catch (error) {
    if (token !== state.viewerToken) return;
    empty.textContent = `STL 加载失败：${error.message}`;
    empty.hidden = false;
  }
}

function renderSegmentationPlot(plot, model) {
  if (!window.Plotly || !plot) return;
  const traces = model.layers.filter((layer) => layer.visible).map((layer) => {
    const vertices = [];
    const faces = [];
    layer.triangles.forEach((tri) => {
      const offset = vertices.length;
      vertices.push(...tri);
      faces.push([offset, offset + 1, offset + 2]);
    });
    return {
      type: "mesh3d",
      name: layer.label,
      x: vertices.map((p) => p[0]),
      y: vertices.map((p) => p[1]),
      z: vertices.map((p) => p[2]),
      i: faces.map((f) => f[0]),
      j: faces.map((f) => f[1]),
      k: faces.map((f) => f[2]),
      color: layer.color,
      opacity: layer.opacity,
      flatshading: false,
      hoverinfo: "skip",
      lighting: { ambient: 0.62, diffuse: 0.82, specular: 0.08, roughness: 0.72 },
    };
  });
  const axis = { showgrid: true, gridcolor: "#d9e1ea", zeroline: false, showbackground: true, backgroundcolor: "#f8fafc" };
  Plotly.react(plot, traces, {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#f8fafc",
    plot_bgcolor: "#f8fafc",
    scene: {
      aspectmode: "data",
      xaxis: { ...axis, title: "X" },
      yaxis: { ...axis, title: "Y" },
      zaxis: { ...axis, title: "Z" },
      camera: { eye: { x: 1.55, y: 1.45, z: 1.05 }, up: { x: 0, y: 0, z: 1 } },
    },
    showlegend: false,
    uirevision: "segmentation-plot",
  }, { displaylogo: false, responsive: true, scrollZoom: true });
}

async function loadSegmentationLayer(layer, token) {
  if (token !== state.viewerToken) return null;
  const response = await fetch(patientFileUrl(layer.file));
  if (!response.ok || token !== state.viewerToken) return null;
  // Plotly renders the mesh on the GPU; keep the complete surface so it does
  // not turn into a sparse cloud of unrelated triangles.
  const triangles = parseStl(await response.arrayBuffer(), 1000000);
  return triangles.length ? { ...layer, triangles } : null;
}

function layerWithStoredOpacity(layer) {
  const stored = state.segmentationLayerOpacity[layer.id];
  return {
    ...layer,
    opacity: Number.isFinite(stored) ? stored : layer.opacity,
  };
}

function bindSegmentationCanvas(canvas, ctx, model) {
  if (canvas.dataset.bound === "1") return;
  canvas.dataset.bound = "1";
  window.addEventListener("resize", () => {
    if (canvas.isConnected) resizeSegmentationCanvas(canvas, ctx, model);
  });
  canvas.addEventListener("pointerdown", (event) => {
    model.dragging = true;
    model.last = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!model.dragging) return;
    const dx = event.clientX - model.last[0];
    const dy = event.clientY - model.last[1];
    model.last = [event.clientX, event.clientY];
    model.ry += dx * 0.01;
    model.rx += dy * 0.01;
    scheduleSegmentationDraw(canvas, ctx, model);
  });
  canvas.addEventListener("pointerup", () => {
    model.dragging = false;
    scheduleSegmentationDraw(canvas, ctx, model);
  });
  canvas.addEventListener("pointercancel", () => {
    model.dragging = false;
    scheduleSegmentationDraw(canvas, ctx, model);
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    model.zoom = clamp(model.zoom * (event.deltaY > 0 ? .92 : 1.08), .35, 4);
    scheduleSegmentationDraw(canvas, ctx, model);
  }, { passive: false });
}

function resizeSegmentationCanvas(canvas, ctx, model) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  scheduleSegmentationDraw(canvas, ctx, model);
}

function scheduleSegmentationDraw(canvas, ctx, model) {
  if (model.drawRaf) return;
  model.drawRaf = requestAnimationFrame(() => {
    model.drawRaf = 0;
    drawSegmentationViewer(canvas, ctx, model);
  });
}

function fitSegmentationModel(model) {
  const bounds = [[Infinity, Infinity, Infinity], [-Infinity, -Infinity, -Infinity]];
  model.layers.forEach((layer) => {
    layer.triangles.forEach((tri) => tri.forEach((p) => {
      for (let i = 0; i < 3; i += 1) {
        bounds[0][i] = Math.min(bounds[0][i], p[i]);
        bounds[1][i] = Math.max(bounds[1][i], p[i]);
      }
    }));
  });
  if (!Number.isFinite(bounds[0][0])) return;
  model.center = [0, 1, 2].map((i) => (bounds[0][i] + bounds[1][i]) / 2);
  model.radius = Math.max(...[0, 1, 2].map((i) => bounds[1][i] - bounds[0][i])) || 1;
}

function drawSegmentationViewer(canvas, ctx, model) {
  const rect = canvas.getBoundingClientRect();
  const w = rect.width || 1;
  const h = rect.height || 1;
  ctx.clearRect(0, 0, w, h);
  drawViewerScene(ctx, w, h);
  if (!model.layers.length) return;
  const faces = [];
  model.layers.forEach((layer) => {
    if (!layer.visible) return;
    const opacity = layer.opacity;
    layer.triangles.forEach((tri) => {
      const projected = tri.map((p) => projectPoint(p, model, w, h));
      const z = (projected[0].z + projected[1].z + projected[2].z) / 3;
      faces.push({ layer, projected, z, opacity });
    });
  });
  faces.sort((a, b) => a.z - b.z);
  faces.forEach((face) => {
    const [a, b, c] = face.projected;
    const shade = clamp(.62 + face.z * .0009, .38, 1);
    ctx.globalAlpha = face.opacity;
    ctx.fillStyle = tintColor(face.layer.color, shade);
    ctx.strokeStyle = face.layer.kind === "vessel" ? tintColor(face.layer.color, 1.08) : "rgba(80,92,110,.18)";
    ctx.lineWidth = face.layer.kind === "vessel" ? 0.8 : 0.35;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.lineTo(c.x, c.y);
    ctx.closePath();
    ctx.fill();
    if (face.layer.kind === "vessel") ctx.stroke();
  });
  ctx.globalAlpha = 1;
}

function drawViewerScene(ctx, w, h) {
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, w, h);

  drawViewerGrid(ctx, w, h);

  drawViewerAxis(ctx, w, h);
}

function drawViewerGrid(ctx, w, h) {
  const horizon = h * 0.43;
  const floorBottom = h - 1;
  const centerX = w * 0.5;
  const spacing = Math.max(36, Math.min(64, w / 12));

  ctx.save();
  ctx.strokeStyle = "rgba(203,213,225,.78)";
  ctx.lineWidth = 1;

  for (let x = -spacing * 10; x <= spacing * 10; x += spacing) {
    ctx.beginPath();
    ctx.moveTo(centerX + x * 0.12, horizon);
    ctx.lineTo(centerX + x, floorBottom);
    ctx.stroke();
  }

  const rows = 8;
  for (let i = 0; i <= rows; i += 1) {
    const t = i / rows;
    const eased = t * t;
    const y = horizon + (floorBottom - horizon) * eased;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(148,163,184,.72)";
  ctx.beginPath();
  ctx.moveTo(0, horizon);
  ctx.lineTo(w, horizon);
  ctx.stroke();
  ctx.restore();
}

function drawViewerAxis(ctx, w, h) {
  const x0 = 26;
  const y0 = h - 34;
  const axes = [
    ["#ef4444", 34, 0, "X"],
    ["#22c55e", 0, -34, "Y"],
    ["#3b82f6", 24, -21, "Z"],
  ];
  ctx.save();
  ctx.lineWidth = 2;
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "middle";
  axes.forEach(([color, dx, dy, label]) => {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x0 + dx, y0 + dy);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x0 + dx, y0 + dy, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(label, x0 + dx + 6, y0 + dy);
  });
  ctx.restore();
}

function projectPoint(point, model, w, h) {
  const x0 = point[0] - model.center[0];
  const y0 = point[1] - model.center[1];
  const z0 = point[2] - model.center[2];
  const cy = Math.cos(model.ry);
  const sy = Math.sin(model.ry);
  const cx = Math.cos(model.rx);
  const sx = Math.sin(model.rx);
  const x1 = x0 * cy + z0 * sy;
  const z1 = -x0 * sy + z0 * cy;
  const y1 = y0 * cx - z1 * sx;
  const z2 = y0 * sx + z1 * cx;
  const scale = Math.min(w, h) * .72 * model.zoom / model.radius;
  return { x: w / 2 + x1 * scale, y: h / 2 - y1 * scale, z: z2 };
}

function parseStl(buffer, maxTriangles) {
  const view = new DataView(buffer);
  const binaryCount = buffer.byteLength >= 84 ? view.getUint32(80, true) : 0;
  const expected = 84 + binaryCount * 50;
  if (binaryCount > 0 && expected <= buffer.byteLength) {
    const stride = Math.max(1, Math.ceil(binaryCount / maxTriangles));
    const triangles = [];
    for (let i = 0; i < binaryCount; i += stride) {
      const offset = 84 + i * 50 + 12;
      triangles.push([
        [view.getFloat32(offset, true), view.getFloat32(offset + 4, true), view.getFloat32(offset + 8, true)],
        [view.getFloat32(offset + 12, true), view.getFloat32(offset + 16, true), view.getFloat32(offset + 20, true)],
        [view.getFloat32(offset + 24, true), view.getFloat32(offset + 28, true), view.getFloat32(offset + 32, true)],
      ]);
    }
    return triangles;
  }
  const text = new TextDecoder("utf-8", { fatal: false }).decode(buffer);
  const vertices = [...text.matchAll(/vertex\s+([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/gi)]
    .map((match) => [Number(match[1]), Number(match[2]), Number(match[3])]);
  const total = Math.floor(vertices.length / 3);
  const stride = Math.max(1, Math.ceil(total / maxTriangles));
  const triangles = [];
  for (let i = 0; i < total; i += stride) {
    triangles.push([vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]]);
  }
  return triangles.filter((tri) => tri.every((p) => p && p.every(Number.isFinite)));
}

function tintColor(hex, factor) {
  const value = hex.replace("#", "");
  const r = clamp(parseInt(value.slice(0, 2), 16) * factor, 0, 255);
  const g = clamp(parseInt(value.slice(2, 4), 16) * factor, 0, 255);
  const b = clamp(parseInt(value.slice(4, 6), 16) * factor, 0, 255);
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function renderFoot() {
  const patient = state.currentPatient;
  if (!patient) {
    $("vpFoot").textContent = state.session ? "请选择病人" : "后端 API: /api/session · /api/run-stage · /api/job";
    return;
  }
  const stage = patient.status.stages[state.stage];
  const done = Object.values(stage.outputs || {}).filter((file) => file.exists).length;
  $("vpFoot").innerHTML = `病人 <b>${escapeHtml(patient.id)}</b> · 当前阶段 <b>${STAGE_BY_KEY[state.stage].label}</b> · 状态 <b>${statusLabel(stage.status)}</b> · 输出文件 <b>${done}</b>`;
}

async function runStage(stageKey) {
  if (!state.session || !state.currentPatient) return;
  const root = state.session.root || $("rootFolder").value.trim();
  const segmentationOptions = stageKey === "segmentation" ? collectSegmentationOptions() : {};
  const runButtons = document.querySelectorAll("#runStageBtn, [data-run-stage]");
  runButtons.forEach((button) => button.disabled = true);
  try {
    const data = await api("/api/run-stage", {
      method: "POST",
      body: {
        session_id: state.session.id,
        root_folder: root,
        stage: stageKey,
        patient_id: state.currentPatient.id,
        model_dir: $("modelDir").value.trim() || undefined,
        device: "auto",
        ...segmentationOptions,
      },
    });
    if (data.session) {
      state.session = data.session;
      $("sessionBadge").textContent = state.session.id.split("-").slice(0, 2).join("-");
    }
    state.jobs.unshift(data.job);
    renderJobs();
    pollJob(data.job.id);
  } catch (error) {
    state.jobs.unshift({ id: "local-error", status: "failed", current: error.message, logs: [error.message], errors: [error.message] });
    renderJobs();
  } finally {
    runButtons.forEach((button) => button.disabled = false);
  }
}

function collectSegmentationOptions() {
  const mode = document.querySelector("[data-seg-param='mode']")?.value || "auto";
  const smoothIterations = Number(document.querySelector("[data-seg-param='smooth_iterations']")?.value || 8);
  const force = Boolean(document.querySelector("[data-run-force]")?.checked);
  return {
    force,
    segmentation: {
      mode,
      smooth_iterations: Number.isFinite(smoothIterations) ? smoothIterations : 8,
    },
  };
}

async function pollJob(jobId) {
  clearInterval(state.pollTimer);
  const tick = async () => {
    const data = await api(`/api/job/${encodeURIComponent(jobId)}`);
    state.jobs = state.jobs.map((job) => job.id === jobId ? data.job : job);
    renderJobs();
    if (["done", "failed"].includes(data.job.status)) {
      clearInterval(state.pollTimer);
      $("jobBadge").textContent = data.job.status;
      await refreshPatients(state.currentPatient?.id);
    }
  };
  $("jobBadge").textContent = "running";
  await tick();
  state.pollTimer = setInterval(tick, 1500);
}

function renderJobs() {
  const host = $("taskList");
  if (!state.jobs.length) {
    host.innerHTML = `<li class="task-empty">暂无运行任务</li>`;
    return;
  }
  host.innerHTML = state.jobs.slice(0, 8).map((job) => {
    const pct = job.total ? Math.round((job.completed || 0) / job.total * 100) : 0;
    const logs = (job.logs || []).slice(-3).map(escapeHtml).join("<br>");
    return `<li class="task-item">
      <div class="task-ic ${job.status === "done" ? "done" : job.status === "running" ? "run" : "fail"}">${job.status === "done" ? "✓" : job.status === "failed" ? "!" : "…"}</div>
      <div class="task-main">
        <div class="task-name">${escapeHtml(job.current || job.stage || job.id)}</div>
        <div class="task-meta">${job.status} · ${pct}% · ${escapeHtml(job.id)}</div>
        ${logs ? `<div class="task-log">${logs}</div>` : ""}
      </div>
    </li>`;
  }).join("");
}

function downloadCurrent() {
  if (!state.session) return;
  const patient = state.currentPatient?.id || "all";
  location.href = `/api/session/${encodeURIComponent(state.session.id)}/download?patient=${encodeURIComponent(patient)}`;
}

function openPatientFile(file) {
  if (!state.session || !state.currentPatient) return;
  window.open(patientFileUrl(file), "_blank");
}

function patientFileUrl(file) {
  if (!state.session || !state.currentPatient) return "#";
  return `/api/session/${encodeURIComponent(state.session.id)}/patient-file?patient=${encodeURIComponent(state.currentPatient.id)}&file=${encodeURIComponent(file)}`;
}

async function api(url, options = {}) {
  const init = { method: options.method || "GET", headers: options.headers || {} };
  if (options.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  const response = await fetch(url, init);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok || data.error) throw new Error(data.error || response.statusText);
  return data;
}

function emptyStage(title, message) {
  return `<div class="empty-state"><div class="es-mark">${escapeHtml(title)}</div><p>${escapeHtml(message)}</p></div>`;
}

function fileRow(label, info = {}) {
  const exists = Boolean(info?.exists);
  const size = exists ? formatBytes(info.size || 0) : "缺失";
  return `<div class="file-row ${exists ? "exists" : "missing"}">
    <span><i></i>${escapeHtml(label)}</span>
    <b>${size}</b>
  </div>`;
}

function dirInfo(patient, name) {
  return { exists: false, size: 0, path: `${patient.folder}\\${name}` };
}

function featureTable(segments) {
  return `<table class="gtable">
    <thead><tr><th>血管</th><th>长度</th><th>平均直径</th><th>面积</th><th>曲率</th></tr></thead>
    <tbody>${segments.map((seg) => `<tr>
      <td class="vname">${escapeHtml(seg.label || seg.id)}</td>
      <td>${formatNum(seg.length)}</td>
      <td>${formatNum(seg.mean_diameter)}</td>
      <td>${formatNum(seg.mean_area)}</td>
      <td>${formatNum(seg.max_curvature, 3)}</td>
    </tr>`).join("")}</tbody>
  </table>`;
}

function keyMetricList(metrics) {
  const entries = Object.entries(metrics);
  if (!entries.length) return `<p class="note">暂无系统级关键特征；运行几何阶段后生成。</p>`;
  return `<div class="file-list compact">${entries.map(([key, value]) => `
    <div class="file-row exists"><span title="${escapeHtml(key)}">${escapeHtml(featureLabel(key))}</span><b>${formatNum(value, 3)}</b></div>
  `).join("")}</div>`;
}

function featureLabel(key) {
  return FEATURE_LABELS[key] || String(key).replaceAll("_", " ");
}

function foldTable(rows) {
  if (!rows.length) return `<p class="note">暂无模型一致性结果。</p>`;
  return `<table class="gtable">
    <thead><tr><th>模型</th><th>PVP</th></tr></thead>
    <tbody>${rows.map((row) => `<tr><td class="vname">${escapeHtml(row.fold)}</td><td>${formatNum(row.pred)}</td></tr>`).join("")}</tbody>
  </table>`;
}

function warningList(warnings) {
  if (!warnings.length) return "";
  return `<div class="notice warn">${warnings.map(escapeHtml).join("<br>")}</div>`;
}

function statusLabel(status = "missing") {
  return { done: "完成", ready: "可运行", missing: "缺输入", running: "运行中", failed: "失败" }[status] || status;
}

function pressureBand(value) {
  if (value >= 22) return "显著门脉高压";
  if (value >= 16) return "升高";
  return "低风险";
}

function formatNum(value, digits = 2) {
  const n = numeric(value);
  return n == null ? "--" : n.toFixed(digits);
}

function numeric(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function formatBytes(bytes) {
  if (!bytes) return "存在";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function shortPath(path) {
  if (!path) return "";
  const parts = String(path).split(/[\\/]/);
  return parts.slice(-2).join("\\");
}

function formatSex(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (["m", "male", "man", "男"].includes(raw)) return "男";
  if (["f", "female", "woman", "女"].includes(raw)) return "女";
  return formatLabelValue(value);
}

function formatLabelValue(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  }[char]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#096;");
}

init();
