# PortaFlow Integrated Web

This folder contains only the integrated PortaFlow website:

- `index.html`: cover page.
- `workbench.html`: integrated clinical workbench shell.
- `app.js`: workbench interaction, patient list, stage switching, PVP panel.
- `styles.css`: workbench styles.
- `landing.css`: cover page styles.
- `assets/`: integrated website images.
- `web_modules.json`: locations of the code and standalone web apps used by each stage.

Standalone module web apps stay in their own projects:

- Segmentation: `VKAN_segementation/web`
- Centerline geometry: `E:/pycharm_code/liver_pre_process/zxx_stl/web`
- PVP code: `PVP_predictor`

The integrated backend is still `../integrated_web_frontend.py`. It serves this folder as static files and reads `web_modules.json` to locate the stage backends and model code.

Run:

```powershell
python ..\integrated_web_frontend.py --host 127.0.0.1 --port 8788
```

Then open:

```text
http://127.0.0.1:8788/
```
