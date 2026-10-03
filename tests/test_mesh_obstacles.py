from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import gc
import weakref

import numpy as np
import pytest

from decanting_workspace import load_scene_spec, materialize_scene
import decanting_workspace.collision_world as collision_module
from decanting_workspace.collision_world import (
    AttachedBox,
    CollisionPhase,
    PinocchioCollisionChecker,
    mesh_collision_geometry,
)
from decanting_workspace.evaluation import collision_phase_for_check
from decanting_workspace.models import BoxPrimitive, MeshObstacle
from decanting_workspace.viewer import CellViewer


def _wall(name="wall", x=0.0, *, collision_enabled=True):
    return MeshObstacle(
        name=name,
        role="obstacle",
        vertices_m=((x, -1.0, -1.0), (x, 1.0, -1.0), (x, 0.0, 1.0)),
        triangles=((0, 1, 2),),
        collision_enabled=collision_enabled,
    )


def _cube(name="solid_cube", center=(0.0, 0.0, 0.0), size=2.0):
    offsets = (
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    )
    return MeshObstacle(
        name=name,
        role="obstacle",
        vertices_m=tuple(
            tuple(center[axis] + offset[axis] * size / 2.0 for axis in range(3))
            for offset in offsets
        ),
        triangles=(
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        ),
    )


def _join(first, second):
    return replace(
        first,
        vertices_m=first.vertices_m + second.vertices_m,
        triangles=first.triangles + tuple(
            tuple(index + len(first.vertices_m) for index in face)
            for face in second.triangles
        ),
    )


@pytest.fixture(scope="module")
def probe_bundle():
    import coal
    import pinocchio as pin

    model = pin.Model()
    model.addFrame(
        pin.Frame(
            "payload_frame", 0, 0,
            pin.SE3(np.eye(3), np.array((4.0, 0.0, 0.0))),
            pin.FrameType.OP_FRAME,
        )
    )
    geometries = pin.GeometryModel()
    geometries.addGeometryObject(
        pin.GeometryObject("probe", 0, 0, pin.SE3.Identity(), coal.Sphere(0.25))
    )
    return SimpleNamespace(name="probe", model=model, collision_model=geometries)


def _checker(bundle, meshes, **kwargs):
    return PinocchioCollisionChecker(
        bundle,
        CollisionPhase(obstacles=(), meshes=meshes, floor_z_m=None, **kwargs),
        include_self_collision=False,
    )


def test_triangle_surface_collision_preserves_logical_name(probe_bundle):
    checker = _checker(probe_bundle, (_wall(),))

    report = checker.check(())

    assert checker.collision_pair_count == 1
    assert report.in_collision
    assert {(c.first, c.second) for c in report.contacts} == {("probe", "wall")}
    world_mesh = checker.geometry_model.geometryObjects[1]
    np.testing.assert_array_equal(world_mesh.placement.homogeneous, np.eye(4))


def test_triangle_mesh_gap_does_not_collide_with_its_bounding_box(probe_bundle):
    mesh = MeshObstacle(
        name="two_sheets",
        role="obstacle",
        vertices_m=(
            (-2.0, -1.0, -1.0), (-2.0, 1.0, -1.0), (-2.0, 0.0, 1.0),
            (2.0, -1.0, -1.0), (2.0, 1.0, -1.0), (2.0, 0.0, 1.0),
        ),
        triangles=((0, 1, 2), (3, 4, 5)),
    )

    report = _checker(probe_bundle, (mesh,)).check(())

    assert not report.in_collision
    assert report.minimum_distance_m == pytest.approx(1.75)


def test_mesh_disabled_and_allowed_contact(probe_bundle):
    disabled = _checker(probe_bundle, (_wall(collision_enabled=False),))
    allowed = _checker(
        probe_bundle, (_wall(),), allowed_contacts=frozenset({("probe", "wall")})
    )

    assert disabled.geometry_model.ngeoms == 1
    assert disabled.collision_pair_count == 0
    assert not disabled.check(()).in_collision
    assert not allowed.check(()).in_collision


def test_mesh_bvh_reuse_keeps_checker_requests_independent(probe_bundle):
    mesh = _wall("shared_wall")
    geometry = mesh_collision_geometry(mesh)
    first = _checker(probe_bundle, (mesh,))
    second = _checker(
        probe_bundle, (mesh,), allowed_contacts=frozenset({("probe", mesh.name)})
    )

    assert mesh_collision_geometry(mesh) is geometry
    assert first.geometry_data is not second.geometry_data
    assert first.check(()).in_collision
    assert not second.check(()).in_collision
    assert first.check(()).in_collision
    # Equal mesh values still have independent identity entries.
    assert mesh_collision_geometry(replace(mesh)) is not geometry


def test_mesh_bvh_cache_retains_large_living_scene_without_dataclass_hashing(
    monkeypatch, probe_bundle,
):
    monkeypatch.setattr(collision_module, "_MESH_COLLISION_CACHE", {})
    def forbidden_hash(self):
        raise AssertionError("mesh vertices must not be hashed for cache lookup")
    monkeypatch.setattr(MeshObstacle, "__hash__", forbidden_hash)
    topology_calls = []
    topology = collision_module.mesh_component_topology
    def counted_topology(vertices, triangles):
        topology_calls.append(len(vertices))
        return topology(vertices, triangles)
    monkeypatch.setattr(collision_module, "mesh_component_topology", counted_topology)
    meshes = tuple(_wall(f"cache_{index}", x=float(index)) for index in range(130))
    first = PinocchioCollisionChecker(
        probe_bundle, CollisionPhase(obstacles=(), meshes=meshes),
        include_self_collision=False,
    )
    entries = tuple(collision_module._MESH_COLLISION_CACHE.values())
    second = PinocchioCollisionChecker(
        probe_bundle, CollisionPhase(obstacles=(), meshes=meshes),
        include_self_collision=False,
    )

    assert len(topology_calls) == len(meshes)
    assert len(collision_module._MESH_COLLISION_CACHE) == len(meshes)
    assert all(
        mesh_collision_geometry(mesh) is entry.geometry
        for mesh, entry in zip(meshes, entries)
    )
    assert first.geometry_data is not second.geometry_data


def test_mesh_bvh_cache_releases_dead_scene_owners(monkeypatch, probe_bundle):
    monkeypatch.setattr(collision_module, "_MESH_COLLISION_CACHE", {})
    mesh = _wall()
    owner = weakref.ref(mesh)
    checker = _checker(probe_bundle, (mesh,))
    assert id(mesh) in collision_module._MESH_COLLISION_CACHE

    del mesh
    gc.collect()

    assert owner() is None
    assert not collision_module._MESH_COLLISION_CACHE
    # A checker owns the BVH and can finish an in-flight request independently
    # after the original scene has been replaced and its cache entries released.
    assert checker.check(()).in_collision


def test_imported_floor_extends_to_mesh_and_phase_box_bounds():
    import coal
    import pinocchio as pin

    model = pin.Model()
    geometries = pin.GeometryModel()
    geometries.addGeometryObject(pin.GeometryObject(
        "probe", 0, 0, pin.SE3(np.eye(3), np.array((50.0, 5.0, -0.1))),
        coal.Sphere(0.25),
    ))
    bundle = SimpleNamespace(name="probe", model=model, collision_model=geometries)
    box = BoxPrimitive("remote_support", "obstacle", (50.0, 5.0, 4.0), (2.0, 2.0, 1.0))
    checker = PinocchioCollisionChecker(
        bundle, CollisionPhase(obstacles=(box,), meshes=(_wall(x=40.0),)),
        include_self_collision=False,
    )

    report = checker.check(())

    assert ("probe", "floor") in {(c.first, c.second) for c in report.contacts}
    floor = next(g for g in checker.geometry_model.geometryObjects if g.name == "world::floor")
    assert floor.geometry.halfSide[0] > 51.0
    baseline = PinocchioCollisionChecker(
        bundle, CollisionPhase(obstacles=(box,)), include_self_collision=False,
    )
    baseline_floor = next(g for g in baseline.geometry_model.geometryObjects if g.name == "world::floor")
    np.testing.assert_array_equal(baseline_floor.geometry.halfSide[:2], (15.0, 15.0))


def test_attached_payload_is_paired_with_world_mesh(probe_bundle):
    payload = AttachedBox(
        name="carried_box",
        size_m=(0.2, 0.2, 0.2),
        frame_name="payload_frame",
        frame_T_box=np.eye(4),
    )

    report = _checker(
        probe_bundle, (_wall(x=4.0),), attached_boxes=(payload,)
    ).check(())

    assert {(c.first, c.second) for c in report.contacts} == {("carried_box", "wall")}


def test_closed_environment_contains_robot_primitive(probe_bundle):
    checker = _checker(probe_bundle, (_cube(),))

    report = checker.check(())

    assert report.in_collision
    assert {(c.first, c.second) for c in report.contacts} == {("probe", "solid_cube")}
    assert report.minimum_distance_m == 0.0
    assert report.nearest_pair == ("probe", "solid_cube")
    assert not checker.is_collision_free(())


def test_closed_containment_respects_allowed_pairs_and_open_surfaces(probe_bundle):
    cube = _cube()
    allowed = _checker(
        probe_bundle, (cube,), allowed_contacts=frozenset({("probe", cube.name)})
    )
    disabled = _checker(probe_bundle, (replace(cube, collision_enabled=False),))
    open_shell = _checker(probe_bundle, (replace(cube, triangles=cube.triangles[:-2]),))

    assert not allowed.check(()).in_collision
    assert not disabled.check(()).in_collision
    assert not open_shell.check(()).in_collision


def test_closed_environment_contains_attached_payload(probe_bundle):
    payload = AttachedBox(
        name="carried_box", size_m=(0.2, 0.2, 0.2),
        frame_name="payload_frame", frame_T_box=np.eye(4),
    )
    cube = _cube(center=(4.0, 0.0, 0.0))
    report = _checker(probe_bundle, (cube,), attached_boxes=(payload,)).check(())

    assert {(c.first, c.second) for c in report.contacts} == {("carried_box", cube.name)}
    assert report.minimum_distance_m == 0.0
    allowed = _checker(
        probe_bundle, (cube,), attached_boxes=(payload,),
        allowed_contacts=frozenset({("carried_box", cube.name)}),
    )
    assert not allowed.check(()).in_collision


def test_closed_environment_preserves_component_gaps_and_nested_cavity(probe_bundle):
    gap = _join(_cube(center=(-3.0, 0.0, 0.0)), _cube(center=(3.0, 0.0, 0.0)))
    hollow = _join(_cube(size=4.0), _cube(size=2.0))

    assert not _checker(probe_bundle, (gap,)).check(()).in_collision
    assert not _checker(probe_bundle, (hollow,)).check(()).in_collision
    mixed = _join(_cube(), _wall(x=10.0))
    assert _checker(probe_bundle, (mixed,)).check(()).in_collision


def test_robot_bvh_containment_uses_actual_vertices_not_geometry_origin():
    import pinocchio as pin

    model = pin.Model()
    shapes = pin.GeometryModel()
    displaced = _join(_wall(x=10.0), _wall(x=4.0))
    shapes.addGeometryObject(pin.GeometryObject(
        "displaced_robot_mesh", 0, 0, pin.SE3.Identity(),
        mesh_collision_geometry(displaced),
    ))
    bundle = SimpleNamespace(name="mesh_robot", model=model, collision_model=shapes)
    # The local origin lies inside this cube, but every real vertex lies outside.
    assert not _checker(bundle, (_cube(),)).check(()).in_collision
    # A later disconnected component lies wholly within this cube. Its local
    # origin and first component do not; testing all real vertices finds it.
    report = _checker(bundle, (_cube(center=(4.0, 0.0, 0.0), size=4.0),)).check(())
    assert report.in_collision
    assert report.minimum_distance_m == 0.0


def test_small_environment_inside_robot_bvh_and_robot_cavity():
    import pinocchio as pin

    model = pin.Model()
    solid = _cube(center=(4.0, 0.0, 0.0), size=4.0)
    hollow = _join(solid, _cube(center=(4.0, 0.0, 0.0), size=2.0))
    small = _cube("small_environment", center=(4.0, 0.0, 0.0), size=0.2)
    def bundle_for(mesh):
        shapes = pin.GeometryModel()
        shapes.addGeometryObject(pin.GeometryObject(
            "robot_bvh", 0, 0, pin.SE3.Identity(), mesh_collision_geometry(mesh),
        ))
        return SimpleNamespace(name="mesh_robot", model=model, collision_model=shapes)

    report = _checker(bundle_for(solid), (small,)).check(())

    assert report.in_collision
    assert report.minimum_distance_m == 0.0
    assert not _checker(bundle_for(hollow), (small,)).check(()).in_collision
    inside_shell_material = _cube(center=(5.5, 0.0, 0.0), size=0.2)
    assert _checker(bundle_for(hollow), (inside_shell_material,)).check(()).in_collision


def test_snapshot_and_phase_keep_meshes_without_semantic_proxy_collision():
    spec = load_scene_spec()
    proxy = replace(spec.static_boxes[0], collision_enabled=False)
    mapped = _wall(proxy.name)
    disabled = _wall("display_only", collision_enabled=False)
    spec = replace(spec, static_boxes=(proxy,), static_meshes=(mapped, disabled))
    # The tote support proxy is required by scene construction.
    original = load_scene_spec()
    support = next(b for b in original.static_boxes if b.name == spec.tote.motion_support_box)
    spec = replace(spec, static_boxes=(proxy, support))
    snapshot = materialize_scene(spec)
    step = SimpleNamespace(suppress_snapshot_objects=(), active_extra_obstacles=())
    check = SimpleNamespace(object_states=(), allowed_contacts=())

    phase = collision_phase_for_check(spec, snapshot, step, check)

    assert snapshot.meshes == (mapped, disabled)
    assert phase.meshes == (mapped,)
    assert proxy.name not in {b.name for b in phase.obstacles}
    suppressed_step = SimpleNamespace(
        suppress_snapshot_objects=(mapped.name,), active_extra_obstacles=()
    )
    assert collision_phase_for_check(spec, snapshot, suppressed_step, check).meshes == ()


def test_mesh_and_active_box_names_must_be_distinct():
    spec = load_scene_spec()
    mesh = _wall(spec.static_boxes[0].name)
    snapshot = materialize_scene(replace(spec, static_meshes=(mesh,)))
    step = SimpleNamespace(suppress_snapshot_objects=(), active_extra_obstacles=())
    check = SimpleNamespace(object_states=(), allowed_contacts=())

    with pytest.raises(ValueError, match="duplicate collision obstacle"):
        collision_phase_for_check(spec, snapshot, step, check)


class _RecordingNode:
    def __init__(self):
        self.objects = []
        self.transforms = []

    def delete(self):
        pass

    def set_object(self, geometry, *args):
        self.objects.append(geometry)

    def set_transform(self, matrix):
        self.transforms.append(np.asarray(matrix).copy())


class _RecordingViewer:
    def __init__(self):
        self.nodes = {}

    def __getitem__(self, name):
        return self.nodes.setdefault(name, _RecordingNode())


def test_viewer_displays_triangles_and_keeps_alternative_samples():
    spec = load_scene_spec()
    spec = replace(
        spec,
        static_boxes=tuple(replace(b, collision_enabled=False) for b in spec.static_boxes),
        static_meshes=(_wall("usd:new_mesh", x=20.0),),
    )
    snapshot = materialize_scene(spec)
    recorder = _RecordingViewer()
    viewer = object.__new__(CellViewer)
    viewer.viewer = recorder

    viewer.update_scene(spec, snapshot, show_frames=False, show_sr_cutting_workspace=False)

    node = recorder.nodes["decanting/environment/obstacle/usd:new_mesh"]
    geometry = node.objects[0]
    np.testing.assert_array_equal(geometry.vertices, spec.static_meshes[0].vertices_m)
    np.testing.assert_array_equal(geometry.faces, spec.static_meshes[0].triangles)
    np.testing.assert_array_equal(node.transforms[0], np.eye(4))
    assert not any(
        f"decanting/environment/{b.role}/{b.name}" in recorder.nodes
        for b in spec.static_boxes
    )
    alternatives = [b for b in snapshot.boxes if b.role == "box_sample_alternative"]
    assert alternatives
    assert all(f"decanting/environment/{b.role}/{b.name}" in recorder.nodes for b in alternatives)
    floor = recorder.nodes["decanting/environment/floor"]
    floor_right = floor.transforms[0][0, 3] + floor.objects[0].lengths[0] / 2.0
    assert floor_right > 20.0


def test_update_scene_hides_exactly_the_named_snapshot_objects():
    # A caller that redraws one phase's moving objects itself has to be able
    # to remove the nominal copy, or the same carton is shown twice.
    spec = load_scene_spec()
    snapshot = materialize_scene(spec)
    movable = [
        box
        for box in snapshot.boxes
        if box.role in {"box", "box_sample_alternative", "tote"}
    ]
    assert len(movable) >= 2
    carried, other = movable[0], movable[1]
    recorder = _RecordingViewer()
    viewer = object.__new__(CellViewer)
    viewer.viewer = recorder

    viewer.update_scene(
        spec,
        snapshot,
        show_frames=False,
        show_sr_cutting_workspace=False,
        hide_object_names=(carried.name,),
    )
    assert (
        f"decanting/environment/{carried.role}/{carried.name}"
        not in recorder.nodes
    )
    assert f"decanting/environment/{other.role}/{other.name}" in recorder.nodes
    # A name that is not in the scene must simply hide nothing.
    plain = _RecordingViewer()
    viewer.viewer = plain
    viewer.update_scene(
        spec,
        snapshot,
        show_frames=False,
        show_sr_cutting_workspace=False,
        hide_object_names=("no_such_object",),
    )
    assert f"decanting/environment/{carried.role}/{carried.name}" in plain.nodes
