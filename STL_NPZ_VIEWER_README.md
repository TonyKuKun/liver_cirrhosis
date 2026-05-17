# STL/NPZ Viewer

Launch the local viewer:

```bash
python stl_npz_viewer.py
```

Basic workflow:

1. Choose the patient folder that contains the STL file.
   - If the folder contains `vessel.stl`, it is selected automatically.
2. Choose the matching `*.hemodynamics.npz` exported by `visualize.py`.
3. Select a field such as `wss_pa`, `velocity_m_per_s`, `pressure_drop_pa`, or `reynolds`.
4. Select `All segments` for a full portal-vein STL, or a single segment such as `mpv` for a segment-only STL.
5. Click `Render`. Use `Export PLY` to save a colored surface for ParaView or MeshLab.

The mapping uses nearest-centerline coloring. It assumes the STL and NPZ were generated from the same patient and coordinate system.

Performance notes:

- Rendering runs in a background thread so the window stays responsive.
- The preview uses a downsampled face set for smoother interaction.
- `Export PLY` still writes the full-resolution STL surface with the mapped scalar.
- Re-rendering the same STL/NPZ/field combination uses cached data.
