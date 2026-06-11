# PortaFlow — 门静脉压力分析工作台 (UI 原型)

一个对标 **HeartFlow / OHIF** 临床工作流的前端原型，把本仓库的三段式流程整合进一个
工作站式界面：

```
① 分割  CT → STL        (VKAN_segementation)
② 几何  STL → 几何特征   (PCG_Prediction: 中心线 / 截面 / 8-血管特征)
③ 预测  几何特征 → PVP   (PVP_predictor)
```

## 布局（对标 HeartFlow + OHIF 三栏式）

- **顶部**：工作流进度条 `分割 → 几何 → 预测`，标明各阶段状态。
- **左栏**：病例列表（带 PVP 伪彩读数、pre/post-TIPS 与 PVT 标签）+ 任务历史。
- **中栏**：主 3D / MPR 视区
  - 阶段 1：STL 网格（Pretrain / Predict / Smooth 图层切换）+ 三向 MPR 缩略图。
  - 阶段 2：血管网格 + 黄色中心线 + 青色采样截面环 + 种子点。
  - 阶段 3：门静脉树伪彩压力映射（蓝→红，对标 FFRCT 的数值映射回血管树）。
- **右栏**：当前阶段的参数与结果
  - 阶段 1：分割流水线参数 + 网格质检。
  - 阶段 2：8-血管几何摘要表 + 解剖连接。
  - 阶段 3：预测 PVP 大数字读数 + 置信度 + 分支压力/流量表 + 模型性能。

## 页面结构

- `index.html` — **首页 / 落地页**（对标 HeartFlow 官网风格）：导航 + Hero（产品截图卡 + 伪彩血管树）
  + 指标条 + 三步工作流 + 核心技术 + 临床验证 + CTA。点击任意「进入工作台」进入应用。
- `workbench.html` — **工作台**：三栏式分析界面（分割 / 几何 / 预测），左上角 logo 可返回首页。

## 运行

纯静态页面，但用到 ES module，需要通过 HTTP 打开（不能直接双击 `file://`）：

```powershell
python -m http.server 8821 --directory webui
# 浏览器打开 http://127.0.0.1:8821/  → 首页 → 点击「进入工作台」
```

3D 渲染使用 Three.js（通过 CDN 的 importmap 加载）。若无网络 / WebGL 不可用，
会自动降级为 SVG 示意图，仍保留伪彩与拓扑。

## 数据来源

界面里的病例、预测值、流量分配和模型指标均取自真实产物，便于演示时“所见即真实研究”：

- `PVP_predictor/runs/final_20260610_pvp_l2_shunt/summary.json`（MAE 2.685 / RMSE 3.605 / R² 0.643, n=72, 5-fold）
- `PVP_predictor/runs/final_20260610_pvp_l2_shunt/oof_predictions.csv`（病例名、PVP 真值/预测、`q_*` 流量代理）
- 8-血管布局与解剖连接对应 `PVP_predictor/dataset.py` 的 `SEGMENTS` 与 `JUNCTIONS`。

每段血管的几何摘要由病例压力确定性地合成（见 `data.js` 中 `synthGeometry`），
是占位展示而非测量值；接入真实 pipeline 时应替换为后端返回的几何特征。

## 文件

| 文件 | 作用 |
|---|---|
| `index.html` | 首页 / 落地页（HeartFlow 风格营销页 + Hero 产品截图卡） |
| `landing.css` | 落地页样式（独立 tokens，亮色 + 医疗蓝） |
| `workbench.html` | 工作台：三栏布局骨架、顶部进度条、importmap |
| `styles.css` | 工作台设计系统（HeartFlow 风格亮色临床主题 + 聚焦深色 3D 视区 + 压力色标） |
| `app.js` | 阶段切换、病例逻辑、Three.js 3D 渲染、伪彩映射、各阶段右栏 |
| `data.js` | 取自真实产物的演示数据 |

## 接入真实后端的路径

原型目前用演示数据驱动。要变成可用工具，可按阶段对接：

1. **阶段 1**：已有 `VKAN_segementation/web_frontend.py` 标准库 HTTP 后端
   （`/api/run`、`/api/mesh`、`/api/ct` 等）。把本原型中阶段 1 的 `运行` 按钮与
   网格/MPR 取数改为调用这些接口即可（该后端原配套的 `app.js` 缺少 `index.html`，
   本原型可作为统一前端替代）。
2. **阶段 2**：中心线提取（VMTK 一类）与截面采样放服务端，前端用 `/api/geometry`
   回显中心线点集、截面 profile 与 8-血管几何摘要。
3. **阶段 3**：`PVP_predictor` 推理放服务端，前端 `/api/predict` 取回标量 PVP、
   分支压力/流量与置信度，映射到血管树伪彩。
