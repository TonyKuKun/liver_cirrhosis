# PortaFlow Integrated Web

This folder contains only the integrated PortaFlow website:

- `index.html`: cover page.
- `workbench.html`: integrated clinical workbench shell.
- `app.js`: workbench interaction, patient list, stage switching, PVP panel.
- `styles.css`: workbench styles.
- `landing.css`: cover page styles.
- `assets/`: integrated website images.
- `geometry/`: the blue geometry-stage UI designed for the integrated workbench.
- `web_modules.json`: locations of processing code, checkpoints, and model assets used by each stage.

The geometry UI is owned by this integrated site under `web/geometry`. It calls geometry
processing and visualization-data implementations from `geometry_feature_extract` through the
integrated backend. `geometry_feature_extract/web` is not mounted or served, and no additional
port or process is used.

The integrated backend is `../integrated_web_frontend.py`. It serves both the main workbench and
the embedded geometry workspace, and reads `web_modules.json` to locate processing and model code.

Run:

```powershell
python ..\integrated_web_frontend.py --host 127.0.0.1 --port 8788
```

Then open:

```text
http://127.0.0.1:8788/
```
