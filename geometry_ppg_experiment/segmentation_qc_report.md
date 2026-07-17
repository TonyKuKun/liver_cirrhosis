# 解剖分段 QC 统计

- 数据根目录：`F:\PCG data\dataset\test4all_sample`
- 扫描样本目录：104
- 名字含 `#` 的 TIPS 术后样本：40
- 状态统计：critical=0, major=2, warning=6, ok=96

## 规则

- `critical`：缺少统一特征文件、缺少中心线分段、或 MPV/SMV/SV/TIPS 这类硬性必需分段缺失。
- `major`：已标记存在但中心线只有 1 个节点、长度极短、或直径明显超出生理/分割合理范围。
- `warning`：分支直径严重失衡、端点距离过大、TIPS 命名与分段不一致等。

## 核心硬性问题

- 缺 MPV：0 个
- 缺 SMV：0 个
- 缺 SV：0 个
- 名字含 `#` 但缺 tips：0 个

## 问题类型计数

| 问题代码 | 样本数 |
|---|---:|
| `pre_tips_has_tips` | 4 |
| `mpv_rpv_endpoint_gap` | 2 |
| `lpv_very_short` | 2 |
| `rpv_very_short` | 1 |

## 有问题样本清单

| 样本 | 状态 | 问题 |
|---|---|---|
| `0020022521HouZhengXu` | warning | [warning] 样本名不含 #，但检测到 tips 分段，请确认是否命名或分段有误 |
| `20210305XuErMin@@@@` | warning | [warning] 样本名不含 #，但检测到 tips 分段，请确认是否命名或分段有误 |
| `20210412FanYuYing@xueshuan` | warning | [warning] 样本名不含 #，但检测到 tips 分段，请确认是否命名或分段有误 |
| `20211208DuanXiuXia` | warning | [warning] 样本名不含 #，但检测到 tips 分段，请确认是否命名或分段有误 |
| `20221227JinJunTing#` | warning | [warning] MPV 与 RPV 最近端点距离 34.1 mm，疑似拓扑不连续或标签串错 |
| `20230719LiuYanChang#` | major | [major] LPV 长度 0.71 mm，疑似分段过短 |
| `20230831XieSuiQing#` | warning | [warning] MPV 与 RPV 最近端点距离 73.1 mm，疑似拓扑不连续或标签串错 |
| `20230902ZhaoSuCai#` | major | [major] LPV 长度 0.71 mm，疑似分段过短<br>[major] RPV 长度 0.71 mm，疑似分段过短 |
