# PortaFlow 几何特征工作区

几何工作台保留原有页面布局、六步流水线、三维图层、中心线编辑、人工分段、
有效分析区和特征展示，由整体 PortaFlow 后端统一提供，不再启动独立端口。

从仓库根目录启动：

```powershell
python ..\web\server.py --host 127.0.0.1 --port 8788
```

打开 `http://127.0.0.1:8788/workbench.html`，在顶部选择“几何特征”。

蓝色几何界面属于整体站点，源码位于 `web/geometry/`。统一后端只复用
`web/geometry_backend.py` 及本目录算法模块中的处理和可视化数据函数；
`geometry_feature_extract` 不再保存或启动 Web 页面与 HTTP 服务。

整体 session、患者路径、六步任务和下载路由均由根目录
`web/server.py` 统一管理。
