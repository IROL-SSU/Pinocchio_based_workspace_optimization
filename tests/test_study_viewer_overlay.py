"""MeshCat overlay used to preview authored poses and highlight targets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from decanting_workspace.models import BoxPrimitive, MeshObstacle
from decanting_workspace.viewer import CellViewer

# MeshCat's create_camera sets /Cameras/default/rotated to a +pi/2 turn
# about X, so a local camera offset (x, y, z) is World (x, -z, y).
_ROTATED_X90 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)


class _RecordingNode:
    def __init__(self, path: str) -> None:
        self.path = path
        self.objects: list[object] = []
        self.transforms: list[np.ndarray] = []
        self.properties: list[tuple[str, object]] = []
        self.deleted = 0

    def delete(self) -> None:
        self.deleted += 1

    def set_object(self, geometry, *args) -> None:
        self.objects.append(geometry)

    def set_transform(self, matrix) -> None:
        self.transforms.append(np.asarray(matrix).copy())

    def set_property(self, name, value) -> None:
        self.properties.append((name, value))


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
    viewer.render_study_payload(
        [("box_a", (0.3, 0.2, 0.25), np.eye(4))], evaluated=False
    )
    viewer.clear_study_overlays()
    assert all(path.startswith("decanting/study") for path in recorder.nodes)


# ------------------------------------------- payload boxes and the camera


def _payload(name: str = "box_a", z: float = 1.2):
    transform = np.eye(4)
    transform[2, 3] = z
    return (name, (0.3, 0.2, 0.25), transform)


def test_payload_boxes_are_drawn_under_the_study_root_by_provenance():
    viewer, recorder = _viewer()
    assert viewer.render_study_payload([_payload()], evaluated=False) == 1
    schematic = "decanting/study/payload/schematic/box_a"
    assert schematic in recorder.nodes
    assert len(recorder.nodes[schematic].objects) == 1
    assert recorder.nodes[schematic].transforms[0][2, 3] == pytest.approx(1.2)

    assert viewer.render_study_payload([_payload()], evaluated=True) == 1
    assert "decanting/study/payload/evaluated/box_a" in recorder.nodes


def test_a_schematic_and_an_evaluated_payload_are_never_shown_together():
    # A grey authored box next to a solved one would read as two results.
    viewer, recorder = _viewer()
    viewer.render_study_payload([_payload()], evaluated=False)
    viewer.render_study_payload([_payload()], evaluated=True)
    root = recorder.nodes["decanting/study/payload"]
    # Each call deletes the whole payload root before drawing its own kind.
    assert root.deleted == 2


def test_an_empty_payload_clears_without_drawing():
    viewer, recorder = _viewer()
    assert viewer.render_study_payload((), evaluated=True) == 0
    assert recorder.nodes["decanting/study/payload"].deleted == 1
    assert not any(
        node.objects
        for path, node in recorder.nodes.items()
        if path.startswith("decanting/study/payload/")
    )


def test_payload_boxes_reject_a_bad_transform_or_size():
    viewer, _ = _viewer()
    with pytest.raises(ValueError, match="finite 4 x 4"):
        viewer.render_study_payload(
            [("box_a", (0.3, 0.2, 0.25), np.eye(3))], evaluated=False
        )
    with pytest.raises(ValueError, match="finite 4 x 4"):
        viewer.render_study_payload(
            [("box_a", (0.3, 0.2, 0.25), np.full((4, 4), np.nan))],
            evaluated=False,
        )
    with pytest.raises(ValueError, match="three positive sizes"):
        viewer.render_study_payload(
            [("box_a", (0.3, 0.0, 0.25), np.eye(4))], evaluated=False
        )
    with pytest.raises(ValueError, match="three positive sizes"):
        viewer.render_study_payload(
            [("box_a", (0.3, 0.2), np.eye(4))], evaluated=False
        )


def test_payload_names_cannot_create_extra_meshcat_nesting():
    viewer, recorder = _viewer()
    viewer.render_study_payload([_payload("../../escape")], evaluated=False)
    assert all(
        path.startswith("decanting/study/payload/schematic/")
        and path.count("/") == 4
        for path in recorder.nodes
        if path.startswith("decanting/study/payload/schematic/")
    )


def test_focus_camera_looks_down_at_the_orbit_centre():
    viewer, recorder = _viewer()
    distance = 6.0
    viewer.focus_camera((1.0, -7.0, 0.8), distance)
    centre = recorder.nodes["/Cameras/default"]
    assert centre.transforms[-1][:3, 3] == pytest.approx((1.0, -7.0, 0.8))
    name, position = recorder.nodes[
        "/Cameras/default/rotated/<object>"
    ].properties[-1]
    assert name == "position"
    # The camera hangs under "/rotated", which MeshCat's create_camera turns
    # by +pi/2 about X, so the recorded local offset has to be mapped into
    # World before it says anything about where the viewer actually stands.
    world = _ROTATED_X90 @ np.asarray(position, dtype=float)
    # Above the cell, not under the floor: the screenshot regression this
    # guards against had the camera below z = 0 looking up.
    assert world[2] > 0.0
    # Three quarter view: back off along -Y and +X so the front faces of the
    # cell stay visible.
    assert world[0] > 0.0
    assert world[1] < 0.0
    # Elevation stays a shallow look-down rather than a top view.
    elevation_deg = math.degrees(
        math.atan2(world[2], float(np.linalg.norm(world[:2])))
    )
    assert 15.0 < elevation_deg < 45.0
    # The offset has to scale with the requested distance, or a large cell
    # still opens inside its own geometry.
    assert float(np.linalg.norm(world)) == pytest.approx(
        distance * float(np.linalg.norm((0.72, 0.72, 0.52)))
    )


def test_focus_camera_rejects_a_bad_target_or_distance():
    viewer, _ = _viewer()
    with pytest.raises(ValueError, match="finite XYZ point"):
        viewer.focus_camera((1.0, 2.0), 3.0)
    with pytest.raises(ValueError, match="finite XYZ point"):
        viewer.focus_camera((1.0, 2.0, float("nan")), 3.0)
    with pytest.raises(ValueError, match="finite and positive"):
        viewer.focus_camera((1.0, 2.0, 3.0), 0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        viewer.focus_camera((1.0, 2.0, 3.0), float("inf"))
