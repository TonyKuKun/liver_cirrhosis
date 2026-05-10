"""
Visualization & Interpretability Tools for PVP Predictor v3
=============================================================

Three layers of "what the model is doing":

1.  **Hemodynamic export** — per-patient NPZ with all named per-point fields
    (velocity_m_per_s, wss_pa, reynolds, pressure_drop_pa, …) plus the
    3D coordinates of segment endpoints, ready for ParaView/VTK/CFD overlay.

2.  **Centerline → STL surface mapping** — KDTree-based transfer of any
    per-centerline-point scalar field (e.g. predicted WSS) onto STL
    vertices, so you can color the surface by the model's hemodynamic
    prediction in 3D. Output: PLY with vertex scalar attribute.

3.  **Interpretability diagnostics**:
        - per-branch attention curves    (which centerline regions drive PVP)
        - flow-rate split bar chart      (model's Q distribution vs Murray prior)
        - junction residual table        (mass / Murray / pressure residuals)
        - model-vs-CFD scatter           (Pearson correlation per field)

Usage
─────
    from visualize import (
        run_inference, export_patient_hemodynamics,
        map_centerline_to_stl, compare_with_cfd,
        plot_attention, plot_flow_splits, plot_junction_table,
    )

    out = run_inference(checkpoint_dir, data_root, patient_name)
    export_patient_hemodynamics(out, '/tmp/Pat001.npz')
    map_centerline_to_stl(out, 'mpv.stl', 'wss_pa', '/tmp/mpv_pred.ply')
"""

import os
import json
import numpy as np
import torch

from dataset import PortalVeinDataset, collate_fn, SEGMENTS, SEG_INDEX, N_SEGMENTS
from model import PortalPressureNet, BLOOD_VISCOSITY_PA_S, Q_REF_M3_PER_S


# =====================================================================
# Inference for one patient (or batch)
# =====================================================================
def load_checkpoint(checkpoint_dir, fold=0, device='cpu'):
    """Loads a fold's best.pt + dataset normalization."""
    norm_path = os.path.join(checkpoint_dir, 'normalization.pt')
    norm = torch.load(norm_path, map_location='cpu', weights_only=False)
    fold_path = os.path.join(checkpoint_dir, f'fold_{fold}', 'best.pt')
    ckpt = torch.load(fold_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = PortalPressureNet(
        d_hidden=args.get('d_hidden', 32),
        dropout=args.get('dropout', 0.3),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, norm, ckpt


def run_inference(checkpoint_dir, data_root, patient_name=None, fold=0,
                  n_points=100, device='cpu'):
    """
    Run inference. If `patient_name` is None, runs all patients in data_root.
    Returns a list of per-patient dicts containing hemodynamics + metadata.
    """
    model, norm, _ = load_checkpoint(checkpoint_dir, fold=fold, device=device)

    ds = PortalVeinDataset(data_root, n_points=n_points, verbose=False)
    # Override normalization with the trained-set's stats
    ds.profile_mean = norm['profile_mean']
    ds.profile_std  = norm['profile_std']
    ds.aux_mean     = norm['aux_mean']
    ds.aux_std      = norm['aux_std']
    ds.label_mean   = norm['label_mean']
    ds.label_std    = norm['label_std']

    indices = (range(len(ds)) if patient_name is None
               else [i for i, d in enumerate(ds.data) if d['name'] == patient_name])
    if not indices:
        raise ValueError(f"No patient '{patient_name}' found in {data_root}")

    results = []
    for i in indices:
        item = ds[i]
        batch = collate_fn([item])
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        with torch.no_grad():
            out = model(batch)
        # Build per-patient export dict
        results.append(_format_inference(out, batch, ds, i))
    return results


def _format_inference(out, batch, ds, i):
    """Assemble a clean per-patient inference dict in numpy."""
    # Predicted PVP (real scale)
    pvp_norm = out['pvp_pred'].squeeze(-1).cpu().numpy()[0]
    pvp_pred = pvp_norm * ds.label_std + ds.label_mean
    pvp_true = float(batch['label'].cpu().numpy()[0])

    # Per-segment hemodynamics → numpy
    hemo = []
    for si in range(N_SEGMENTS):
        h = out['hemo_per_seg'][si]
        seg = {'name': SEGMENTS[si],
               'present': bool(batch['segment_mask'][0, si].item() > 0),
               'arc_length_mm': batch['arc_lengths'][0, si].cpu().numpy(),
               'point_valid':   batch['point_valid'][0, si].cpu().numpy()}
        for k, v in h.items():
            seg[k] = v[0].cpu().numpy() if v.dim() > 0 else float(v.item())
        # Add raw geometry for cross-reference
        prof = batch['profiles'][0, si].cpu().numpy()
        seg['area_mm2']      = prof[..., 0]
        seg['eq_diameter_mm'] = prof[..., 1]
        seg['curvature_1_mm']  = prof[..., 2]
        seg['inscribed_radius_mm'] = prof[..., 3]
        seg['endpoints_3d'] = batch['endpoints_3d'][0, si].cpu().numpy()
        hemo.append(seg)

    return {
        'name':          ds.data[i]['name'],
        'pvp_pred_mmHg': float(pvp_pred),
        'pvp_true_mmHg': pvp_true,
        'is_post_tips':  bool(batch['is_post_tips'][0].item() > 0.5),
        'Q_per_segment': out['Q'][0].cpu().numpy(),
        'inflow_split':       out['flow_out']['inflow_frac'][0].cpu().numpy(),
        'conf_outflow_split': out['flow_out']['conf_outflow_frac'][0].cpu().numpy(),
        'bif_outflow_split':  out['flow_out']['bif_outflow_frac'][0].cpu().numpy(),
        'inflow_delta':       out['flow_out']['inflow_delta'][0].cpu().numpy(),
        'conf_outflow_delta': out['flow_out']['conf_outflow_delta'][0].cpu().numpy(),
        'bif_outflow_delta':  out['flow_out']['bif_outflow_delta'][0].cpu().numpy(),
        'collateral_fraction': float(out['flow_out']['collateral_fraction'][0].item()),
        'attn_weights':  out['attn_weights'][0].cpu().numpy(),
        'hemo':          hemo,
        'junction':      {k: (v[0].cpu().numpy() if torch.is_tensor(v) and v.dim() > 0
                              else (float(v.item()) if torch.is_tensor(v) else v))
                          for k, v in out['junction'].items()},
        'confluence_3d': batch['confluence_3d'][0].cpu().numpy(),
        'extras_for_eval': batch['extras_for_eval'][0],
    }


# =====================================================================
# Hemodynamic export (NPZ)
# =====================================================================
def export_patient_hemodynamics(patient_result, npz_path):
    """Write a flat dict suitable for np.savez, ready for ParaView/CFD."""
    out_dict = {
        'patient_name':  patient_result['name'],
        'pvp_pred_mmHg': patient_result['pvp_pred_mmHg'],
        'pvp_true_mmHg': patient_result['pvp_true_mmHg'],
        'Q_per_segment': patient_result['Q_per_segment'],
        'segment_names': np.array(SEGMENTS),
        'confluence_3d': patient_result['confluence_3d'],
    }
    for seg in patient_result['hemo']:
        prefix = seg['name'] + '_'
        for k, v in seg.items():
            if isinstance(v, np.ndarray):
                out_dict[prefix + k] = v
            elif isinstance(v, (int, float, bool)):
                out_dict[prefix + k] = np.array([v])
            elif isinstance(v, str):
                out_dict[prefix + k] = np.array(v)
    np.savez(npz_path, **out_dict)
    print(f"[Export] {patient_result['name']} → {npz_path}")


# =====================================================================
# Centerline → STL surface mapping
# =====================================================================
def _read_stl_ascii_or_binary(stl_path):
    """Returns (vertices, faces) as numpy arrays. Supports ASCII and binary STL."""
    with open(stl_path, 'rb') as f:
        head = f.read(80)
    is_ascii = head[:5] == b'solid'
    if is_ascii:
        # ASCII parse
        verts, faces = [], []
        with open(stl_path, 'r') as f:
            facet = []
            for line in f:
                line = line.strip()
                if line.startswith('vertex'):
                    parts = line.split()
                    facet.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if line.startswith('endfacet'):
                    if len(facet) == 3:
                        idxs = []
                        for v in facet:
                            verts.append(v)
                            idxs.append(len(verts) - 1)
                        faces.append(idxs)
                    facet = []
        verts = np.array(verts, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
    else:
        # Binary: 80-byte header, 4-byte uint count, then 50-byte facets
        with open(stl_path, 'rb') as f:
            f.read(80)
            n_tri = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
            data = np.frombuffer(f.read(n_tri * 50), dtype=np.uint8)
        # Each facet is 50 bytes: 12 (normal) + 36 (3 vertices × 12) + 2 (attr)
        verts_flat = []
        faces = []
        for i in range(n_tri):
            off = i * 50 + 12
            tri = np.frombuffer(data[off:off + 36].tobytes(), dtype=np.float32).reshape(3, 3)
            base = len(verts_flat)
            for v in tri:
                verts_flat.append(v.tolist())
            faces.append([base, base + 1, base + 2])
        verts = np.array(verts_flat, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
    return verts, faces


def _write_ply_with_scalar(ply_path, vertices, faces, scalar, scalar_name='value'):
    """Write a PLY with one scalar attribute per vertex (ParaView-friendly)."""
    with open(ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"property float {scalar_name}\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for i, v in enumerate(vertices):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {scalar[i]:.6f}\n")
        for fc in faces:
            f.write(f"3 {fc[0]} {fc[1]} {fc[2]}\n")


def _gather_centerline(patient_result, segments=None):
    """
    Collect (centerline_xyz, scalar_per_point) by interpolating the per-segment
    arc-length-indexed fields back to a 3D position via the segment endpoints.

    Returns:
        centerline_pts (M, 3) — physical positions
        seg_id         (M,)   — which segment each point came from
        arc            (M,)   — arc length along its segment
    Note: physical 3D positions for centerline points are reconstructed by
    linear interpolation between the two stored endpoint coords. This is an
    approximation — for accurate placement, the actual centerline polyline
    in physical space should be loaded from your pre-processing pipeline.
    """
    if segments is None:
        segments = list(range(N_SEGMENTS))
    pts, sids, arcs = [], [], []
    for si in segments:
        seg = patient_result['hemo'][si]
        if not seg['present']:
            continue
        ep = seg['endpoints_3d']  # (2, 3)
        if np.allclose(ep, 0):    # endpoints not provided
            continue
        n = len(seg['arc_length_mm'])
        v = seg['point_valid'] > 0
        if v.sum() < 2:
            continue
        # Linear interp 3D pos as fraction along arc-length
        s = seg['arc_length_mm']
        s_min = s[v].min(); s_max = s[v].max()
        t = np.where(v, (s - s_min) / max(s_max - s_min, 1e-6), 0.0)
        p = ep[0:1, :] + t[:, None] * (ep[1:2, :] - ep[0:1, :])
        pts.append(p[v])
        sids.append(np.full(v.sum(), si, dtype=np.int32))
        arcs.append(s[v])
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros(0)
    return np.concatenate(pts, 0), np.concatenate(sids), np.concatenate(arcs)


def map_centerline_to_stl(patient_result, stl_path, scalar_name, ply_out_path,
                          segment_filter=None):
    """
    Color an STL surface by a model-predicted scalar (e.g. 'wss_pa').

    Each STL vertex receives the scalar value of the nearest valid centerline
    point. Output is a PLY file with a `scalar_name` per-vertex attribute that
    ParaView, MeshLab, or pyvista can render as a colormap.

    segment_filter: list of segment names (e.g. ['mpv']) to restrict the
                    centerline points used for nearest-neighbor mapping.
    """
    verts, faces = _read_stl_ascii_or_binary(stl_path)
    seg_idxs = ([SEG_INDEX[s] for s in segment_filter]
                if segment_filter is not None else None)
    cl_xyz, cl_seg, cl_arc = _gather_centerline(patient_result, seg_idxs)
    if len(cl_xyz) == 0:
        raise ValueError("No valid centerline points to map. "
                         "Check `endpoints_3d` are populated.")

    # Build scalar array aligned with centerline points
    scalar_per_cl = np.zeros(len(cl_xyz), dtype=np.float32)
    cursor = 0
    for si in (seg_idxs or range(N_SEGMENTS)):
        seg = patient_result['hemo'][si]
        if not seg['present'] or scalar_name not in seg:
            continue
        valid = seg['point_valid'] > 0
        if valid.sum() < 2:
            continue
        ep = seg['endpoints_3d']
        if np.allclose(ep, 0):
            continue
        n_seg_pts = int(valid.sum())
        scalar_per_cl[cursor:cursor + n_seg_pts] = seg[scalar_name][valid]
        cursor += n_seg_pts
    scalar_per_cl = scalar_per_cl[:cursor]
    cl_xyz       = cl_xyz[:cursor]

    # Nearest-neighbor mapping
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(cl_xyz)
        _, nn_idx = tree.query(verts, k=1)
    except ImportError:
        # Brute force fallback
        d2 = ((verts[:, None, :] - cl_xyz[None, :, :]) ** 2).sum(axis=-1)
        nn_idx = d2.argmin(axis=1)

    vert_scalar = scalar_per_cl[nn_idx]
    _write_ply_with_scalar(ply_out_path, verts, faces, vert_scalar,
                           scalar_name=scalar_name)
    print(f"[STL→PLY] {stl_path} colored by '{scalar_name}' → {ply_out_path} "
          f"(min={vert_scalar.min():.3f}, max={vert_scalar.max():.3f})")


# =====================================================================
# Compare with CFD ground truth (per-point scalars)
# =====================================================================
def compare_with_cfd(model_npz_path, cfd_npz_path, fields=None):
    """
    Both files are NPZ with keys `<segment_name>_<field>`, e.g. 'mpv_wss_pa'.
    Returns per-field Pearson correlation.

    fields: list of field names to compare. Defaults to a sensible set.
    """
    if fields is None:
        fields = ['velocity_m_per_s', 'wss_pa', 'pressure_drop_pa', 'reynolds']
    md = np.load(model_npz_path, allow_pickle=True)
    cd = np.load(cfd_npz_path,   allow_pickle=True)

    results = {}
    for seg in SEGMENTS:
        for f in fields:
            key = f'{seg}_{f}'
            if key not in md.files or key not in cd.files:
                continue
            m, c = md[key], cd[key]
            # Use only valid points
            valid_key = f'{seg}_point_valid'
            if valid_key in md.files:
                v = (md[valid_key] > 0)
                m = m[v]; c = c[v]
            if len(m) < 3:
                continue
            r = float(np.corrcoef(m, c)[0, 1]) if np.std(m) > 0 and np.std(c) > 0 else float('nan')
            mae = float(np.mean(np.abs(m - c)))
            rel_mae = float(np.mean(np.abs(m - c)) / (np.mean(np.abs(c)) + 1e-9))
            results[key] = {'pearson': r, 'mae': mae, 'rel_mae': rel_mae,
                            'n_points': int(len(m))}
    return results


# =====================================================================
# Diagnostics — minimal matplotlib-free text/JSON output
# =====================================================================
def plot_attention(patient_result, ax_dict=None, save_path=None):
    """Plot attention curve per branch. Returns figure (if matplotlib available)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not installed; "
              "use export_patient_hemodynamics() and plot externally.")
        return None
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    aw = patient_result['attn_weights']
    for si, (ax, sname) in enumerate(zip(axes, SEGMENTS)):
        seg = patient_result['hemo'][si]
        if not seg['present']:
            ax.set_title(f"{sname} (absent)", fontsize=10)
            ax.axis('off')
            continue
        s = seg['arc_length_mm']
        v = seg['point_valid'] > 0
        ax.fill_between(s[v], 0, aw[si][v], alpha=0.6)
        ax.set_title(f"{sname}: peak attn @ {s[aw[si].argmax()]:.1f} mm", fontsize=10)
        ax.set_xlabel('arc length (mm)')
        ax.set_ylabel('attention')
    fig.suptitle(f"{patient_result['name']}: per-branch attention "
                 f"(PVP_pred={patient_result['pvp_pred_mmHg']:.1f} vs "
                 f"true={patient_result['pvp_true_mmHg']:.1f} mmHg)",
                 fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[Plot] Attention → {save_path}")
    return fig


def plot_flow_splits(patient_result, save_path=None):
    """3-panel bar chart: model splits vs Murray-3 priors at all 3 junctions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not installed.")
        return None

    # Murray prior from junction-end diameters (idx 0)
    d = lambda sn: patient_result['hemo'][SEG_INDEX[sn]]['eq_diameter_mm'][0]
    pres = lambda sn: patient_result['hemo'][SEG_INDEX[sn]]['present']
    def murray3(diams_with_mask):
        cubes = np.array([di**3 if mi else 0.0 for di, mi in diams_with_mask])
        s = cubes.sum() + 1e-9
        return cubes / s

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ── Panel 1: Inflow split (sv, smv) ─────────────────────
    diams_in = [(d('sv'),  pres('sv')),  (d('smv'), pres('smv'))]
    prior_in = murray3(diams_in)
    pred_in  = patient_result['inflow_split']
    x = np.arange(2)
    axes[0].bar(x - 0.18, prior_in, 0.35, label='Murray-3 prior', color='steelblue', alpha=0.7)
    axes[0].bar(x + 0.18, pred_in,  0.35, label='Model',         color='salmon',    alpha=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(['SV', 'SMV'])
    axes[0].set_title('Inflow split  (Q_sv + Q_smv = 1)')
    axes[0].set_ylabel('flow fraction'); axes[0].set_ylim(0, 1)
    axes[0].legend(loc='upper right', fontsize=8)

    # ── Panel 2: Confluence outflow split (mpv, lgv, pgv) — NEW ──
    diams_co = [(d('mpv'), pres('mpv')),
                (d('lgv'), pres('lgv')),
                (d('pgv'), pres('pgv'))]
    prior_co = murray3(diams_co)
    pred_co  = patient_result['conf_outflow_split']
    names_co = ['MPV',
                'LGV' if pres('lgv') else 'LGV\n(absent)',
                'PGV' if pres('pgv') else 'PGV\n(absent)']
    x = np.arange(3)
    axes[1].bar(x - 0.18, prior_co, 0.35, label='Murray-3 prior', color='steelblue', alpha=0.7)
    axes[1].bar(x + 0.18, pred_co,  0.35, label='Model',         color='salmon',    alpha=0.9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names_co)
    axes[1].set_title(f'Confluence outflow  '
                      f'(collateral burden = {1-pred_co[0]:.2f})')
    axes[1].set_ylim(0, 1); axes[1].legend(loc='upper right', fontsize=8)

    # ── Panel 3: Bifurcation split (lpv, rpv, tips) ─────────
    diams_bo = [(d('lpv'),  pres('lpv')),
                (d('rpv'),  pres('rpv')),
                (d('tips'), pres('tips'))]
    prior_bo = murray3(diams_bo)
    pred_bo  = patient_result['bif_outflow_split']
    names_bo = ['LPV', 'RPV',
                'TIPS' if pres('tips') else 'TIPS\n(absent)']
    x = np.arange(3)
    axes[2].bar(x - 0.18, prior_bo, 0.35, label='Murray-3 prior', color='steelblue', alpha=0.7)
    axes[2].bar(x + 0.18, pred_bo,  0.35, label='Model',         color='salmon',    alpha=0.9)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names_bo)
    axes[2].set_title(f'Bifurcation outflow  (Q_mpv = {pred_co[0]:.2f})')
    axes[2].set_ylim(0, 1); axes[2].legend(loc='upper right', fontsize=8)

    fig.suptitle(f"{patient_result['name']}: flow rate parameterization "
                 f"(PVP_pred={patient_result['pvp_pred_mmHg']:.1f} vs "
                 f"true={patient_result['pvp_true_mmHg']:.1f} mmHg)", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[Plot] Flow splits → {save_path}")
    return fig


def junction_diagnostic_table(patient_results):
    """Per-patient junction physics summary."""
    rows = []
    for r in patient_results:
        j = r['junction']
        rows.append({
            'name': r['name'],
            'pvp_pred': r['pvp_pred_mmHg'],
            'pvp_true': r['pvp_true_mmHg'],
            'is_post_tips':           r['is_post_tips'],
            'collateral_fraction':    r['collateral_fraction'],
            'murray_dev_inflow':      float(j.get('murray_dev_inflow', 0)),
            'murray_dev_conf_out':    float(j.get('murray_dev_conf_out', 0)),
            'murray_dev_bif_out':     float(j.get('murray_dev_bif_out', 0)),
            'press_resid_bifurc':     float(j.get('press_resid_bifurc', 0)),
        })
    return rows


# =====================================================================
# CLI entry: dump all diagnostics for one patient or whole cohort
# =====================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint_dir', type=str, required=True,
                    help='Directory containing fold_*/best.pt and normalization.pt')
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--out_dir',   type=str, default='./inference_out')
    ap.add_argument('--patient',   type=str, default=None,
                    help='Single patient name; default = all.')
    ap.add_argument('--fold',      type=int, default=0)
    ap.add_argument('--n_points',  type=int, default=100)
    ap.add_argument('--make_plots', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = run_inference(args.checkpoint_dir, args.data_root,
                            patient_name=args.patient, fold=args.fold,
                            n_points=args.n_points)

    # Per-patient export
    for r in results:
        npz = os.path.join(args.out_dir, f"{r['name']}.hemodynamics.npz")
        export_patient_hemodynamics(r, npz)
        if args.make_plots:
            plot_attention(r,    save_path=os.path.join(args.out_dir, f"{r['name']}.attention.png"))
            plot_flow_splits(r,  save_path=os.path.join(args.out_dir, f"{r['name']}.flow_splits.png"))

    # Cohort-wide diagnostic table
    diag = junction_diagnostic_table(results)
    with open(os.path.join(args.out_dir, 'diagnostics.json'), 'w') as f:
        json.dump(diag, f, indent=2, default=float)
    print(f"\n[Done] Diagnostics: {os.path.join(args.out_dir, 'diagnostics.json')}")
    print(f"[Done] {len(results)} patients exported to {args.out_dir}")


if __name__ == '__main__':
    main()