"""MeshCat overlay used to preview authored poses and highlight targets."""

from __future__ import annotations

import numpy as np
import pytest

from decanting_workspace.models import BoxPrimitive, MeshObstacle
from decanting_workspace.viewer import CellViewer


class _RecordingNode:
    def __init__(self, path: str) -> None:
        self.path = path
        self.objects: list[object] = []
        self.transforms: list[np.ndarray] = []
        self.deleted = 0

    def delete(self) -> None:
        self.deleted += 1

    def set_object(self, geometry, *args) -> None:
        self.objects.append(geometry)

    def set_transform(self, matrix) -> None:
        self.transforms.append(np.asarray(matrix).copy())


class _RecordingViewer:
    def __init__(self) -> None:
        self.nodes: dict[str, _RecordingNode] = {}

    def __getitem__(self, path: str) -> _RecordingNode:
        return self.nodes.setdefault(path, _RecordingNode(path))


def _viewer() -> tuple[CellViewer, _RecordingViewer]:
    recorder = _RecordingViewer()
    viewer = object.__new__(CellViewer)
    viewer.viewer = recorder
    return viewer, recorder


def _box(name: str = "pallet") -> BoxPrimitive:
    return BoxPrimitive(
        name=name,
        role="pallet",
        center_m=(1.0, 2.0, 0.5),
        size_m=(1.2, 0.8, 0.15),
        yaw_deg=30.0,
    )


def _mesh(name: str = "conveyor_level_1") -> MeshObstacle:
    return MeshObstacle(
        name=name,
        role="conveyor",
        vertices_m=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        triangles=((0, 1, 2),),
        source_prim_path="/World/ConveyorGroup/Conv3",
    )


def test_markers_draw_three_axes_per_pose_under_the_study_root():
    viewer, recorder = _viewer()
    transform = np.eye(4)
    transform[:3, 3] = (0.5, -1.0, 1.2)

    drawn = viewer.render_study_markers([("c1_start", transform, 0x3478C5)])

    assert drawn == 1
    for axis in ("x", "y", "z"):
        assert f"decanting/study/markers/c1_start/axis_{axis}" in recorder.nodes
    assert recorder.nodes["decanting/study/markers"].deleted == 1


def test_segments_draw_one_line_between_the_two_endpoints():
    viewer, recorder = _viewer()
    drawn = viewer.render_study_markers(
        (), [("c1", (0.0, 0.0, 1.0), (0.0, 0.0, 1.3), 0x3478C5)]
    )
    assert drawn == 1
    node = recorder.nodes["decanting/study/markers/c1/segment"]
    assert node.objects


def test_a_zero_length_segment_is_skipped_rather_than_drawn():
    viewer, recorder = _viewer()
    drawn = viewer.render_study_markers(
        (), [("c1", (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0x3478C5)]
    )
    assert drawn == 0
    assert "decanting/study/markers/c1/segment" not in recorder.nodes


def test_each_marker_call_replaces_the_previous_overlay():
    viewer, recorder = _viewer()
    viewer.render_study_markers([("first", np.eye(4), 0x3478C5)])
    viewer.render_study_markers([("second", np.eye(4), 0x3478C5)])
    assert recorder.nodes["decanting/study/markers"].deleted == 2


def test_marker_names_cannot_create_extra_meshcat_nesting():
    viewer, recorder = _viewer()
    viewer.render_study_markers([("a/b ../c", np.eye(4), 0x3478C5)])
    paths = [path for path in recorder.nodes if path.endswith("axis_x")]
    assert paths == ["decanting/study/markers/a_b_.._c/axis_x"]


def test_markers_reject_a_non_finite_or_misshaped_transform():
    viewer, _ = _viewer()
    broken = np.eye(4)
    broken[0, 3] = float("nan")
    with pytest.raises(ValueError, match="finite 4 x 4"):
        viewer.render_study_markers([("c1", broken, 0x3478C5)])
    with pytest.raises(ValueError, match="finite 4 x 4"):
        viewer.render_study_markers([("c1", np.eye(3), 0x3478C5)])


def test_markers_reject_an_invalid_colour_or_axis_length():
    viewer, _ = _viewer()
    with pytest.raises(ValueError, match="color must be an RGB integer"):
        viewer.render_study_markers([("c1", np.eye(4), 0x1000000)])
    with pytest.raises(ValueError, match="color must be an RGB integer"):
        viewer.render_study_markers([("c1", np.eye(4), True)])
    with pytest.raises(ValueError, match="axis_length_m"):
        viewer.render_study_markers(
            [("c1", np.eye(4), 0x3478C5)], axis_length_m=0.0
        )


def test_segments_reject_non_finite_or_wrong_sized_points():
    viewer, _ = _viewer()
    with pytest.raises(ValueError, match="two XYZ points"):
        viewer.render_study_markers((), [("c1", (0.0, 0.0), (1.0, 1.0, 1.0), 0)])
    with pytest.raises(ValueError, match="must be finite"):
        viewer.render_study_markers(
            (), [("c1", (0.0, 0.0, float("inf")), (1.0, 1.0, 1.0), 0)]
        )


def test_highlight_outlines_boxes_and_overlays_meshes():
    viewer, recorder = _viewer()
    drawn = viewer.render_study_target_highlight([_box()], [_mesh()])

    assert drawn == 2
    assert "decanting/study/targets/box/pallet" in recorder.nodes
    mesh_node = recorder.nodes["decanting/study/targets/mesh/conveyor_level_1"]
    geometry = mesh_node.objects[0]
    np.testing.assert_array_equal(
        geometry.vertices, np.asarray(_mesh().vertices_m, dtype=float)
    )
    np.testing.assert_array_equal(mesh_node.transforms[0], np.eye(4))
    assert recorder.nodes["decanting/study/targets"].deleted == 1


def test_highlight_box_outline_follows_the_oriented_box_corners():
    viewer, recorder = _viewer()
    box = _box()
    viewer.render_study_target_highlight([box])
    points = recorder.nodes[
        "decanting/study/targets/box/pallet"
    ].objects[0].geometry.position
    coordinates = np.asarray(points, dtype=float)
    # Twelve edges, two endpoints each, centred on the box.
    assert coordinates.shape[1] == 24
    assert np.allclose(coordinates.mean(axis=1), box.center_m, atol=1e-9)


def test_highlight_rejects_an_invalid_colour():
    viewer, _ = _viewer()
    with pytest.raises(ValueError, match="color must be an RGB integer"):
        viewer.render_study_target_highlight([_box()], color=-1)


def test_empty_highlight_clears_without_drawing():
    viewer, recorder = _viewer()
    assert viewer.render_study_target_highlight() == 0
    assert recorder.nodes["decanting/study/targets"].deleted == 1


def test_clear_study_overlays_removes_the_whole_subtree():
    viewer, recorder = _viewer()
    viewer.clear_study_overlays()
    assert recorder.nodes["decanting/study"].deleted == 1


def test_study_overlays_never_touch_the_environment_or_playback_subtrees():
    viewer, recorder = _viewer()
    viewer.render_study_markers(
        [("c1", np.eye(4), 0x3478C5)], [("c1", (0, 0, 0), (0, 0, 1), 0x3478C5)]
    )
    viewer.render_study_target_highlight([_box()], [_mesh()])
    viewer.clear_study_overlays()
    assert all(path.startswith("decanting/study") for path in recorder.nodes)
