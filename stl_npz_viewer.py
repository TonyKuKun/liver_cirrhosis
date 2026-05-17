"""
Small local STL + hemodynamics NPZ viewer.

Run:
    python stl_npz_viewer.py

Workflow:
    1. Choose a patient folder that contains one or more STL files.
    2. Choose the matching *.hemodynamics.npz file exported by visualize.py.
    3. Pick a scalar field, such as wss_pa or velocity_m_per_s.
    4. Render the STL colored by the nearest exported centerline value.
"""

import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


SEGMENTS = ["mpv", "sv", "smv", "lpv", "rpv", "tips", "lgv", "pgv"]
PREVIEW_MAX_FACES = 30000
DEFAULT_FIELDS = [
    "wss_pa",
    "velocity_m_per_s",
    "pressure_drop_pa",
    "reynolds",
    "radius_m",
    "area_m2",
    "local_R_pa_s_per_m4",
    "cum_R_pa_s_per_m3",
    "dean",
    "area_gradient",
    "shape_alpha",
]


def read_stl_ascii_or_binary(stl_path):
    """Return STL vertices and triangular faces."""
    stl_path = os.fspath(stl_path)
    with open(stl_path, "rb") as f:
        head = f.read(80)
        count_bytes = f.read(4)

    is_ascii = head[:5].lower() == b"solid"
    if is_ascii:
        try:
            return _read_stl_ascii(stl_path)
        except ValueError:
            pass

    if len(count_bytes) != 4:
        raise ValueError(f"Invalid STL file: {stl_path}")
    return _read_stl_binary(stl_path)


def _read_stl_ascii(stl_path):
    verts = []
    faces = []
    facet = []
    with open(stl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("vertex"):
                parts = line.split()
                if len(parts) != 4:
                    continue
                facet.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("endfacet"):
                if len(facet) == 3:
                    base = len(verts)
                    verts.extend(facet)
                    faces.append([base, base + 1, base + 2])
                facet = []
    if not verts:
        raise ValueError("No ASCII STL vertices found")
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _read_stl_binary(stl_path):
    with open(stl_path, "rb") as f:
        f.read(80)
        n_tri = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        data = f.read(n_tri * 50)

    if len(data) < n_tri * 50:
        raise ValueError(f"Binary STL is truncated: {stl_path}")

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_tri, 50)
    triangles = raw[:, 12:48].copy().view(np.float32).reshape(n_tri, 3, 3)
    verts = triangles.reshape(-1, 3).astype(np.float32, copy=False)
    faces = np.arange(n_tri * 3, dtype=np.int32).reshape(n_tri, 3)
    return verts, faces


def write_ply_with_scalar(ply_path, vertices, faces, scalar, scalar_name="value"):
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"property float {scalar_name}\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v, s in zip(vertices, scalar):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {float(s):.6f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def available_scalar_fields(npz):
    fields = set()
    for seg in SEGMENTS:
        arc_key = f"{seg}_arc_length_mm"
        if arc_key not in npz.files:
            continue
        n_points = np.asarray(npz[arc_key]).reshape(-1).size
        prefix = f"{seg}_"
        for key in npz.files:
            if not key.startswith(prefix):
                continue
            field = key[len(prefix):]
            value = np.asarray(npz[key])
            if value.reshape(-1).size == n_points and field not in {
                "arc_length_mm",
                "point_valid",
            }:
                fields.add(field)

    ordered = [f for f in DEFAULT_FIELDS if f in fields]
    ordered.extend(sorted(fields - set(ordered)))
    return ordered


def build_centerline_from_npz(npz, scalar_name, segment_name="All segments"):
    selected = SEGMENTS if segment_name == "All segments" else [segment_name]
    points = []
    values = []
    used_segments = []

    for seg in selected:
        keys = {
            "arc": f"{seg}_arc_length_mm",
            "valid": f"{seg}_point_valid",
            "endpoints": f"{seg}_endpoints_3d",
            "scalar": f"{seg}_{scalar_name}",
            "present": f"{seg}_present",
        }
        if any(k not in npz.files for k in keys.values() if k != keys["present"]):
            continue
        if keys["present"] in npz.files and not bool(np.asarray(npz[keys["present"]]).reshape(-1)[0]):
            continue

        arc = np.asarray(npz[keys["arc"]], dtype=float).reshape(-1)
        valid = np.asarray(npz[keys["valid"]]).reshape(-1) > 0
        endpoints = np.asarray(npz[keys["endpoints"]], dtype=float).reshape(2, 3)
        scalar = np.asarray(npz[keys["scalar"]], dtype=float).reshape(-1)

        if arc.size != scalar.size or valid.size != arc.size:
            continue
        if valid.sum() < 2 or np.allclose(endpoints, 0):
            continue

        valid_arc = arc[valid]
        denom = max(float(valid_arc.max() - valid_arc.min()), 1e-6)
        t = (arc - float(valid_arc.min())) / denom
        xyz = endpoints[0][None, :] + t[:, None] * (endpoints[1][None, :] - endpoints[0][None, :])

        points.append(xyz[valid])
        values.append(scalar[valid])
        used_segments.append(seg)

    if not points:
        raise ValueError(
            f"No mappable centerline values for field '{scalar_name}' and segment '{segment_name}'."
        )

    return np.concatenate(points, axis=0), np.concatenate(values, axis=0), used_segments


def map_scalar_to_vertices(vertices, centerline_points, centerline_values):
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(centerline_points).query(vertices, k=1)
        return centerline_values[idx]
    except Exception:
        return _nearest_neighbor_chunked(vertices, centerline_points, centerline_values)


def _nearest_neighbor_chunked(vertices, centerline_points, centerline_values, chunk_size=5000):
    out = np.empty(len(vertices), dtype=np.float32)
    for start in range(0, len(vertices), chunk_size):
        stop = min(start + chunk_size, len(vertices))
        d2 = ((vertices[start:stop, None, :] - centerline_points[None, :, :]) ** 2).sum(axis=-1)
        out[start:stop] = centerline_values[d2.argmin(axis=1)]
    return out


def set_axes_equal(ax, vertices):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


class StlNpzViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PVP Flow Surface Viewer")
        self.geometry("1280x820")
        self.configure(bg="#0f172a")

        self.patient_dir = tk.StringVar()
        self.stl_path = tk.StringVar()
        self.npz_path = tk.StringVar()
        self.scalar_name = tk.StringVar(value="wss_pa")
        self.segment_name = tk.StringVar(value="All segments")
        self.status = tk.StringVar(value="Choose a patient folder. vessel.stl is selected automatically when present.")

        self.vertices = None
        self.faces = None
        self.vertex_scalar = None
        self.colorbar = None
        self.render_button = None
        self.export_button = None
        self.stats_text = tk.StringVar(value="No surface loaded")
        self._stl_cache = {}
        self._map_cache = {}
        self._render_job = 0

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0f172a")
        style.configure("Panel.TFrame", background="#111827")
        style.configure("TLabel", background="#0f172a", foreground="#dbeafe")
        style.configure("Panel.TLabel", background="#111827", foreground="#dbeafe")
        style.configure("Status.TLabel", background="#020617", foreground="#93c5fd")
        style.configure("TButton", background="#2563eb", foreground="#eff6ff", padding=(10, 6))
        style.map("TButton", background=[("active", "#38bdf8"), ("disabled", "#334155")])
        style.configure("TCombobox", fieldbackground="#e0f2fe", background="#0f172a")

        header = ttk.Frame(self, style="Panel.TFrame", padding=(14, 10))
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(
            header,
            text="PVP Flow Surface Viewer",
            style="Panel.TLabel",
            font=("Segoe UI", 17, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Label(
            header,
            textvariable=self.stats_text,
            style="Panel.TLabel",
            font=("Segoe UI", 10),
        ).pack(side=tk.RIGHT)

        panel = ttk.Frame(self, style="Panel.TFrame", padding=10)
        panel.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(panel, text="Patient folder", style="Panel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.patient_dir, width=80).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(panel, text="Browse", command=self.choose_patient_dir).grid(row=0, column=2, padx=4)

        ttk.Label(panel, text="STL").grid(row=1, column=0, sticky="w")
        self.stl_combo = ttk.Combobox(panel, textvariable=self.stl_path, width=80)
        self.stl_combo.grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(panel, text="Browse", command=self.choose_stl).grid(row=1, column=2, padx=4)

        ttk.Label(panel, text="NPZ").grid(row=2, column=0, sticky="w")
        self.npz_combo = ttk.Combobox(panel, textvariable=self.npz_path, width=80)
        self.npz_combo.grid(row=2, column=1, sticky="we", padx=4)
        ttk.Button(panel, text="Browse", command=self.choose_npz).grid(row=2, column=2, padx=4)

        ttk.Label(panel, text="Field").grid(row=3, column=0, sticky="w")
        self.scalar_combo = ttk.Combobox(panel, textvariable=self.scalar_name, values=DEFAULT_FIELDS, width=26)
        self.scalar_combo.grid(row=3, column=1, sticky="w", padx=4)

        ttk.Label(panel, text="Segment").grid(row=3, column=1, sticky="e", padx=(0, 210))
        self.segment_combo = ttk.Combobox(
            panel,
            textvariable=self.segment_name,
            values=["All segments"] + SEGMENTS,
            width=18,
            state="readonly",
        )
        self.segment_combo.grid(row=3, column=1, sticky="e", padx=(0, 4))

        self.render_button = ttk.Button(panel, text="Render", command=self.render)
        self.render_button.grid(row=3, column=2, sticky="we", padx=4)
        self.export_button = ttk.Button(panel, text="Export PLY", command=self.export_ply)
        self.export_button.grid(row=3, column=3, sticky="we", padx=4)

        panel.columnconfigure(1, weight=1)

        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor="#020617")
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self)

        status = ttk.Label(self, textvariable=self.status, anchor="w", padding=7, style="Status.TLabel")
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def choose_patient_dir(self):
        path = filedialog.askdirectory(title="Choose patient folder")
        if not path:
            return
        self.patient_dir.set(path)
        self.refresh_folder_files(path)

    def choose_stl(self):
        path = filedialog.askopenfilename(
            title="Choose STL",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
            initialdir=self.patient_dir.get() or os.getcwd(),
        )
        if path:
            self.stl_path.set(path)

    def choose_npz(self):
        path = filedialog.askopenfilename(
            title="Choose hemodynamics NPZ",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")],
            initialdir=self.patient_dir.get() or os.getcwd(),
        )
        if path:
            self.npz_path.set(path)
            self.refresh_fields(path)

    def refresh_folder_files(self, folder):
        folder_path = Path(folder)
        stls = sorted(str(p) for p in folder_path.rglob("*.stl"))
        npzs = sorted(str(p) for p in folder_path.rglob("*.npz"))

        project_npz = Path(__file__).resolve().parent / "inference_out" / f"{folder_path.name}.hemodynamics.npz"
        if project_npz.exists():
            npzs.insert(0, str(project_npz))

        self.stl_combo["values"] = stls
        self.npz_combo["values"] = npzs
        if stls:
            vessel_stls = [p for p in stls if Path(p).name.lower() == "vessel.stl"]
            self.stl_path.set(vessel_stls[0] if vessel_stls else stls[0])
        if npzs:
            self.npz_path.set(npzs[0])
            self.refresh_fields(npzs[0])
        self.status.set(
            f"Found {len(stls)} STL file(s), {len(npzs)} NPZ file(s). "
            f"{'Selected vessel.stl.' if stls and Path(self.stl_path.get()).name.lower() == 'vessel.stl' else ''}"
        )

    def refresh_fields(self, npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                fields = available_scalar_fields(npz)
        except Exception as exc:
            messagebox.showerror("NPZ error", str(exc))
            return
        self.scalar_combo["values"] = fields
        if fields and self.scalar_name.get() not in fields:
            self.scalar_name.set(fields[0])

    def render(self):
        stl_path = self.stl_path.get()
        npz_path = self.npz_path.get()
        if not stl_path or not npz_path:
            messagebox.showwarning("Missing input", "Please choose both STL and NPZ files.")
            return

        self._render_job += 1
        job_id = self._render_job
        self._set_busy(True, "Preparing render in background...")
        args = (job_id, stl_path, npz_path, self.scalar_name.get(), self.segment_name.get())
        threading.Thread(target=self._render_worker, args=args, daemon=True).start()

    def _set_busy(self, busy, text):
        self.status.set(text)
        state = "disabled" if busy else "normal"
        if self.render_button is not None:
            self.render_button.configure(state=state)
        if self.export_button is not None:
            self.export_button.configure(state=state)

    def _render_worker(self, job_id, stl_path, npz_path, scalar_name, segment_name):
        try:
            vertices, faces = self._load_stl_cached(stl_path)
            map_key = (
                stl_path,
                os.path.getmtime(stl_path),
                npz_path,
                os.path.getmtime(npz_path),
                scalar_name,
                segment_name,
            )
            if map_key in self._map_cache:
                vertex_scalar, used_segments = self._map_cache[map_key]
            else:
                with np.load(npz_path, allow_pickle=True) as npz:
                    cl_xyz, cl_values, used_segments = build_centerline_from_npz(
                        npz, scalar_name, segment_name
                    )
                vertex_scalar = map_scalar_to_vertices(vertices, cl_xyz, cl_values)
                self._map_cache[map_key] = (vertex_scalar, used_segments)
                if len(self._map_cache) > 6:
                    self._map_cache.pop(next(iter(self._map_cache)))
            self.after(0, self._finish_render, job_id, vertices, faces, vertex_scalar, used_segments)
        except Exception as exc:
            self.after(0, self._fail_render, job_id, str(exc))

    def _load_stl_cached(self, stl_path):
        cache_key = (stl_path, os.path.getmtime(stl_path))
        if cache_key not in self._stl_cache:
            self._stl_cache.clear()
            self._stl_cache[cache_key] = read_stl_ascii_or_binary(stl_path)
        return self._stl_cache[cache_key]

    def _finish_render(self, job_id, vertices, faces, vertex_scalar, used_segments):
        if job_id != self._render_job:
            return
        self.vertices = vertices
        self.faces = faces
        self.vertex_scalar = vertex_scalar
        self._draw_mesh(vertices, faces, vertex_scalar, used_segments)
        self._set_busy(False, self.status.get())

    def _fail_render(self, job_id, error):
        if job_id != self._render_job:
            return
        self._set_busy(False, "Render failed.")
        messagebox.showerror("Render failed", error)

    def _draw_mesh(self, vertices, faces, vertex_scalar, used_segments):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#020617")
        self.ax.grid(False)

        draw_faces = faces
        if len(faces) > PREVIEW_MAX_FACES:
            step = int(np.ceil(len(faces) / PREVIEW_MAX_FACES))
            draw_faces = faces[::step]
        else:
            step = 1

        face_values = vertex_scalar[draw_faces].mean(axis=1)
        lo, hi = np.nanpercentile(face_values, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(face_values)), float(np.nanmax(face_values))
        if lo == hi:
            hi = lo + 1e-6

        norm = Normalize(vmin=lo, vmax=hi)
        cmap = matplotlib.colormaps.get_cmap("turbo")
        colors = cmap(norm(face_values))
        triangles = vertices[draw_faces]

        mesh = Poly3DCollection(
            triangles,
            facecolors=colors,
            linewidths=0.0,
            alpha=0.96,
            antialiased=False,
        )
        self.ax.add_collection3d(mesh)
        set_axes_equal(self.ax, vertices)
        self.ax.view_init(elev=22, azim=-58)
        self.ax.set_xlabel("X", color="#bfdbfe")
        self.ax.set_ylabel("Y", color="#bfdbfe")
        self.ax.set_zlabel("Z", color="#bfdbfe")
        self.ax.tick_params(colors="#64748b", labelsize=8)
        self.ax.set_title(
            f"{self.scalar_name.get()} on {Path(self.stl_path.get()).name}",
            color="#e0f2fe",
            pad=16,
        )

        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(face_values)
        cbar = self.figure.colorbar(sm, ax=self.ax, shrink=0.72, pad=0.08, label=self.scalar_name.get())
        cbar.ax.yaxis.label.set_color("#dbeafe")
        cbar.ax.tick_params(colors="#bfdbfe")

        self.canvas.draw_idle()
        self.stats_text.set(
            f"{len(vertices):,} vertices | {len(faces):,} faces | "
            f"{float(np.nanmin(vertex_scalar)):.3g}-{float(np.nanmax(vertex_scalar)):.3g}"
        )
        self.status.set(
            f"Rendered {len(draw_faces):,}/{len(faces):,} faces, mapped from "
            f"{', '.join(used_segments)}. Export PLY keeps all faces."
        )

    def export_ply(self):
        if self.vertices is None or self.faces is None or self.vertex_scalar is None:
            messagebox.showwarning("Nothing to export", "Render first, then export PLY.")
            return
        default_name = Path(self.stl_path.get()).with_suffix("").name + f".{self.scalar_name.get()}.ply"
        path = filedialog.asksaveasfilename(
            title="Export colored PLY",
            defaultextension=".ply",
            initialfile=default_name,
            filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            write_ply_with_scalar(path, self.vertices, self.faces, self.vertex_scalar, self.scalar_name.get())
            self.status.set(f"Exported {path}")
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))


def main():
    app = StlNpzViewer()
    app.mainloop()


if __name__ == "__main__":
    main()
