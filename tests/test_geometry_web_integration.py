import json
import inspect
import time
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from web import server as web


def _write_tetra_stl(path, origin):
    x, y, z = origin
    vertices = [
        (x, y, z),
        (x + 1, y, z),
        (x, y + 1, z),
        (x, y, z + 1),
    ]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    rows = ["solid vessel"]
    for a, b, c in faces:
        rows.extend([
            "facet normal 0 0 0",
            " outer loop",
            f"  vertex {' '.join(map(str, vertices[a]))}",
            f"  vertex {' '.join(map(str, vertices[b]))}",
            f"  vertex {' '.join(map(str, vertices[c]))}",
            " endloop",
            "endfacet",
        ])
    rows.append("endsolid vessel")
    path.write_text("\n".join(rows), encoding="ascii")


def _write_centerline(path, origin):
    x, y, z = origin
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        f"0 {x} {y} {z} -1 1 -1\n"
        f"1 {x + 0.5} {y + 0.5} {z + 0.5} 0 -1 -1\n",
        encoding="ascii",
    )


def _request_json(url, payload=None):
    request = Request(url)
    if payload is not None:
        request.data = json.dumps(payload).encode("utf-8")
        request.add_header("Content-Type", "application/json")
        request.method = "POST"
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_geometry_workbench_uses_integrated_server(tmp_path):
    assert "removeprefix" not in inspect.getsource(web.Handler._geometry_static)
    patient_id = "case#01"
    patient = tmp_path / patient_id
    patient.mkdir()
    (patient / "vessel.stl").write_text(
        """solid vessel
facet normal 0 0 -1
 outer loop
  vertex 0 0 0
  vertex 0 1 0
  vertex 1 0 0
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 0 1
 endloop
endfacet
facet normal -1 0 0
 outer loop
  vertex 0 0 0
  vertex 0 0 1
  vertex 0 1 0
 endloop
endfacet
facet normal 1 1 1
 outer loop
  vertex 1 0 0
  vertex 0 1 0
  vertex 0 0 1
 endloop
endfacet
endsolid vessel
""",
        encoding="ascii",
    )
    parent_session = web._create_session({"root_folder": str(tmp_path)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        query = urlencode({
            "session_id": parent_session["id"],
            "patient": patient_id,
        })
        status, workbench = _request_json(f"{base}/api/geometry/workbench?{query}")
        assert status == 200
        assert workbench["workbench"]["iframe_url"].startswith("/geometry/?embed=1&autoload=1&ui=9")

        status, geometry_session = _request_json(
            f"{base}/api/geometry/session/from-parent",
            {"session_id": parent_session["id"], "patient_id": patient_id},
        )
        assert status == 200
        assert geometry_session["session"]["patients"][0]["id"] == patient_id
        geometry_id = geometry_session["session"]["id"]
        assert "#" not in geometry_id

        data_query = urlencode({"patient": patient_id, "section_stride": 10})
        status, geometry_data = _request_json(
            f"{base}/api/geometry/session/{geometry_id}/data?{data_query}"
        )
        assert status == 200
        assert geometry_data["patient"]["id"] == patient_id
        assert geometry_data["mesh"]["faces"]
        assert "step_files" in geometry_data

        with urlopen(f"{base}/geometry/index.html", timeout=5) as response:
            assert response.status == 200
            geometry_html = response.read().decode("utf-8")
            assert 'class="embed-flow-panel panel"' in geometry_html
            assert 'href="styles.css?v=8"' in geometry_html
            assert 'src="app.js?v=23"' in geometry_html
            assert 'id="effectiveAnalysisBtn"' in geometry_html
            assert 'id="analysisSuggestBtn"' not in geometry_html

        with urlopen(f"{base}/geometry/app.js", timeout=5) as response:
            assert response.status == 200
            geometry_app = response.read().decode("utf-8")
            assert "/api/geometry" in geometry_app
            assert "encodeURIComponent(state.session.id)" in geometry_app
            assert "[...Object.keys(stats), ...Object.keys(segments)]" in geometry_app
            assert "state.analysisRange.active = false;" in geometry_app
            assert "available_vessels" in geometry_app
            assert "baselineRanges" in geometry_app
            assert "toggleEffectiveAnalysis" in geometry_app
            assert '$("effectiveAnalysisBtn").disabled = !state.session || !state.data;' in geometry_app
            assert "缺少 pointwise_profiles.json" in geometry_app
            assert "排除起端" not in geometry_app
            assert "排除末端" not in geometry_app
            assert "suggestAnalysisRanges" not in geometry_app
            assert "待删除截面" in geometry_app
            assert "addAnalysisRangeTraces" not in geometry_app
            assert "const ranges = analysisVessels().map" in geometry_app
            save_range_source = geometry_app.split(
                "async function saveAnalysisRanges()", 1)[1].split(
                    "function renderScene()", 1)[0]
            assert "await refreshData();" not in save_range_source
            assert "if (state.analysisRange.active || data.analysis_regions?.saved)" not in geometry_app
            assert "平均曲率" in geometry_app
            assert "迂曲度" in geometry_app
            assert "分段剖面特征 (${segmentCount})" in geometry_app

        assert web.GEOMETRY_STATIC_ROOT == web.WEB_ROOT / "geometry"
        assert web.GEOMETRY_STATIC_ROOT != web.CENTERLINE_ROOT / "web"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_geometry_view_cache_can_be_invalidated(tmp_path):
    stl = tmp_path / "vessel.stl"
    stl.write_bytes(b"solid vessel\nendsolid vessel\n")
    cache = tmp_path / web.GEOMETRY_VIEW_CACHE_NAME
    cache.write_bytes(b"stale")

    web._invalidate_geometry_view_cache(stl)
    web._invalidate_geometry_view_cache(stl)

    assert not cache.exists()


def test_feature_stl_matches_existing_centerline_coordinates(tmp_path):
    patient = tmp_path / "case01"
    patient.mkdir()
    predict = patient / "predict_smooth.stl"
    vessel = patient / "vessel.stl"
    _write_tetra_stl(predict, (1000, 1000, 1000))
    _write_tetra_stl(vessel, (10, 20, 30))
    centerline = patient / "features" / web.SMOOTH_CENTERLINE_NAME
    _write_centerline(centerline, (10, 20, 30))

    assert web._feature_stl(patient) == vessel

    time.sleep(0.002)
    _write_centerline(centerline, (1000, 1000, 1000))
    assert web._feature_stl(patient) == predict


def test_geometry_data_keeps_pointwise_feature_visualization_layers(tmp_path):
    patient = tmp_path / "case01"
    patient.mkdir()
    stl = patient / "vessel.stl"
    _write_tetra_stl(stl, (0, 0, 0))
    features = patient / "features"
    features.mkdir()
    centerline = features / web.SMOOTH_CENTERLINE_NAME
    centerline.write_text(
        "0 0 0 0 -1 1 -1\n"
        "1 0.5 0.5 0.5 0 2 -1\n"
        "2 1 1 1 1 -1 -1\n",
        encoding="ascii",
    )
    (features / web.SEGMENT_ASSIGNMENTS_NAME).write_text(
        json.dumps({"segments": {"mpv": {"path": [0, 1, 2]}}}),
        encoding="utf-8",
    )
    (features / web.geometry_web.POINTWISE_TEMP_NAME).write_text(
        json.dumps({
            "mpv": {
                "analysis_path": [0, 1, 2],
                "position": [0.0, 0.5, 1.0],
                "area": [1.0, 2.0, 1.5],
                "eq_diameter": [1.1, 1.6, 1.3],
                "perimeter": [3.0, 4.0, 3.5],
                "curvature": [0.01, 0.04, 0.02],
                "circularity": [0.9, 0.95, 0.92],
                "inscribed_radius": [0.4, 0.7, 0.5],
            },
        }),
        encoding="utf-8",
    )

    data = web.geometry_web.build_visualization_data(stl, section_stride=1, max_faces=1000)
    pointwise = data["pointwise"]
    assert pointwise["feature_points"]["mpv"]["x"]
    assert pointwise["feature_points"]["mpv"]["position"] == [0.0, 0.5, 1.0]
    assert pointwise["sampled_sections"]["mpv"]["x"]
    assert len(pointwise["sampled_sections"]["mpv"]["position"]) == len(
        pointwise["sampled_sections"]["mpv"]["x"]
    )
    assert pointwise["max_sections"]["mpv"]["x"]
    assert pointwise["max_sections"]["mpv"]["position"] == 0.5
    assert pointwise["mean_sections"]["mpv"]["x"]


def test_geometry_data_does_not_guess_analysis_ranges_from_unified(tmp_path):
    patient = tmp_path / "case01"
    patient.mkdir()
    stl = patient / "vessel.stl"
    _write_tetra_stl(stl, (0, 0, 0))
    features = patient / "features"
    features.mkdir()
    (features / web.UNIFIED_FEATURES_NAME).write_text(
        json.dumps({
            "pointwise": {
                "mpv": {
                    "position": [0.0, 0.5, 1.0],
                    "area": [1.0, 2.0, 1.5],
                    "eq_diameter": [1.1, 1.6, 1.3],
                    "perimeter": [3.0, 4.0, 3.5],
                },
            },
        }),
        encoding="utf-8",
    )

    data = web.geometry_web.build_visualization_data(
        stl, section_stride=1, max_faces=1000)

    assert data["analysis_regions"] == {
        "ranges": {},
        "available_vessels": [],
        "source": None,
    }
    assert data["pointwise"]["feature_points"] == {}
