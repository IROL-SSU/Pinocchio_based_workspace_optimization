"""Meshcat adapter for SceneSnapshot and Pinocchio URDF models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .models import (
    BasePose,
    BoxPrimitive,
    CutterProxySpec,
    MeshObstacle,
    SceneSpec,
    SuctionProxySpec,
)
from .robots import (
    RobotBundle,
    configure_sr_j3_stroke,
    floating_configuration,
    load_robot_bundle,
)
from .scene import SceneSnapshot, sr_cutting_workspace_footprint_box
from .transforms import pose_matrix, quaternion_matrix


ROLE_STYLE: dict[str, tuple[int, float]] = {
    "conveyor": (0x326A91, 0.82),
    "worktable": (0x497C9E, 0.82),
    "camera_support": (0x34383D, 0.90),
    "pallet": (0x9B7048, 0.95),
    "box": (0xB77736, 0.88),
    "box_sample_alternative": (0xB77736, 0.28),
    "tote": (0x70418A, 0.55),
    "pedestal": (0x4D525A, 0.92),
}

_UR20_SUCTION_PROXY_ROOT = "decanting/robots/ur20/provisional_suction"


@dataclass
class LoadedRobot:
    bundle: RobotBundle
    visualizer: object
    visual_only: bool = False
    native_j3_stroke_m: float | None = None


class CellViewer:
    """Render the same candidate snapshot used by later collision evaluation."""

    def __init__(self, viewer: object | None = None) -> None:
        import meshcat

        self.viewer = viewer or meshcat.Visualizer()
        self._robots: list[LoadedRobot] = []
        self._ur20_suction_attachment: (
            tuple[RobotBundle, SuctionProxySpec, str] | None
        ) = None

    def render(
        self,
        spec: SceneSpec,
        snapshot: SceneSnapshot,
        *,
        ur20_urdf: str | Path | None = None,
        ur20_package_dirs: Sequence[str | Path] = (),
        sr12ia_urdf: str | Path | None = None,
        sr12ia_package_dirs: Sequence[str | Path] = (),
        sr12ia_visual_only: bool = False,
        show_collisions: bool = False,
        show_frames: bool = True,
        show_installation_region: bool = True,
        show_sr_cutting_workspace: bool = True,
        show_sr_footprint_fill: bool = True,
    ) -> None:
        """Clear and render a complete cell state."""

        self.viewer.delete()
        self._robots.clear()
        self._ur20_suction_attachment = None
        self._set_background()
        boxes = _visible_environment_boxes(spec, snapshot)
        self._render_floor(spec, boxes, snapshot.meshes)
        for box in boxes:
            self._render_box(box)
        for mesh in snapshot.meshes:
            self._render_mesh(mesh)
        if show_installation_region:
            self._render_installation_region(spec)
        if show_frames:
            for frame in snapshot.frames.values():
                self._render_frame(frame.name, frame.translation_m, frame.rotation_xyzw)
        if show_sr_cutting_workspace:
            self._render_sr_cutting_workspace(
                spec,
                snapshot.sr_base,
                snapshot.state.clearance_m,
                show_fill=show_sr_footprint_fill,
            )

        ur_spec = spec.robots["ur20"]
        sr_spec = spec.robots["sr12ia"]
        ur_bundle = load_robot_bundle(
            "ur20",
            ur20_urdf or ur_spec.urdf_path,
            package_dirs=ur20_package_dirs,
            load_visual=True,
            suction_proxy=ur_spec.suction_proxy,
        )
        sr_bundle = load_robot_bundle(
            "sr12ia",
            sr12ia_urdf or sr_spec.urdf_path,
            package_dirs=sr12ia_package_dirs,
            load_visual=True,
            cutter_proxy=sr_spec.cutter_proxy,
        )
        if not sr12ia_visual_only:
            configure_sr_j3_stroke(sr_bundle, snapshot.state.sr_j3_stroke_m)
        ur_q = self._render_robot(
            ur_bundle,
            snapshot.ur_base,
            ur_spec.nominal_q,
            root_name="decanting/robots/ur20",
            show_collisions=show_collisions,
        )
        if (
            ur_spec.suction_proxy is not None
            and not ur_bundle.model.existFrame("suction_gripper")
        ):
            self._render_suction_proxy(
                ur_bundle,
                ur_q,
                ur_spec.suction_proxy,
                root_name=_UR20_SUCTION_PROXY_ROOT,
            )
        sr_q = list(sr_spec.nominal_q)
        native_sr_stroke = _joint_upper_limit(sr_bundle, "joint3")
        sr_q[2] = min(
            sr_q[2],
            snapshot.state.sr_j3_stroke_m,
            native_sr_stroke,
        )
        rendered_sr_q = self._render_robot(
            sr_bundle,
            snapshot.sr_base,
            sr_q,
            root_name="decanting/robots/sr12ia",
            show_collisions=show_collisions,
            visual_only=sr12ia_visual_only,
            native_j3_stroke_m=native_sr_stroke,
        )
        if (
            sr_spec.cutter_proxy is not None
            and not sr_bundle.model.existFrame("cutter_proxy")
        ):
            self._render_cutter_proxy(
                sr_bundle,
                rendered_sr_q,
                sr_spec.cutter_proxy,
                root_name="decanting/robots/sr12ia/provisional_cutter",
            )

    def open(self) -> None:
        self.viewer.open()

    def loaded_robot(self, name: str) -> LoadedRobot:
        """Return one initialized robot visualizer by bundle name."""

        matches = tuple(robot for robot in self._robots if robot.bundle.name == name)
        if len(matches) != 1:
            raise KeyError(f"expected one loaded robot named {name!r}, got {len(matches)}")
        return matches[0]

    def update_ur20_suction_proxy(
        self,
        q: Sequence[float] | np.ndarray,
    ) -> bool:
        """Move the provisional suction display to the current UR ``tool0``.

        The proxy is intentionally separate from the licensed UR mesh, so a
        normal ``MeshcatVisualizer.display`` call cannot update it.  Playback
        invokes this method immediately after every displayed configuration.
        """

        attachment = self._ur20_suction_attachment
        if attachment is None:
            return False
        bundle, proxy, root_name = attachment
        configuration = np.asarray(q, dtype=float).reshape(-1)
        if (
            configuration.size != bundle.model.nq
            or not np.all(np.isfinite(configuration))
        ):
            raise ValueError(
                f"UR20 suction update requires {bundle.model.nq} finite values"
            )

        import pinocchio as pin

        frame_id = bundle.model.getFrameId("tool0")
        if frame_id >= bundle.model.nframes:
            raise RuntimeError(f"{bundle.name} has no tool0 frame for the suction proxy")
        data = bundle.model.createData()
        pin.forwardKinematics(bundle.model, data, configuration)
        pin.updateFramePlacements(bundle.model, data)
        tool_pose = np.asarray(data.oMf[frame_id].homogeneous)

        cylinder_local = pose_matrix(
            (0.0, 0.0, proxy.cylinder_length_m / 2.0)
        )
        cylinder_local[:3, :3] = _rotation_x(math.pi / 2.0)
        self.viewer[f"{root_name}/cylinder"].set_transform(
            tool_pose @ cylinder_local
        )

        pad_local = pose_matrix(
            (
                0.0,
                0.0,
                proxy.cylinder_length_m + proxy.pad_thickness_m / 2.0,
            )
        )
        pad_local[:3, :3] = _rotation_x(math.pi / 2.0)
        self.viewer[f"{root_name}/pad"].set_transform(tool_pose @ pad_local)
        return True

    def update_scene(
        self,
        spec: SceneSpec,
        snapshot: SceneSnapshot,
        *,
        show_frames: bool = True,
        show_sr_cutting_workspace: bool = True,
        show_sr_footprint_fill: bool = True,
        suppress_process_objects: bool = False,
    ) -> None:
        """Update case-dependent primitives without reloading either robot mesh."""

        self.viewer["decanting/environment"].delete()
        boxes = _visible_environment_boxes(
            spec, snapshot, suppress_process_objects=suppress_process_objects
        )
        self._render_floor(spec, boxes, snapshot.meshes)
        for box in boxes:
            self._render_box(box)
        for mesh in snapshot.meshes:
            self._render_mesh(mesh)

        self.viewer["decanting/frames"].delete()
        if show_frames:
            for frame in snapshot.frames.values():
                self._render_frame(
                    frame.name,
                    frame.translation_m,
                    frame.rotation_xyzw,
                )

        self.viewer["decanting/guides/sr12ia/cutting_workspace"].delete()
        if show_sr_cutting_workspace:
            self._render_sr_cutting_workspace(
                spec,
                snapshot.sr_base,
                snapshot.state.clearance_m,
                show_fill=show_sr_footprint_fill,
            )

    def render_target_pose(
        self,
        world_T_target: Sequence[Sequence[float]] | np.ndarray | None,
        *,
        successful: bool,
        axis_length_m: float = 0.18,
    ) -> None:
        """Show the selected cached TCP target as three World-oriented axes."""

        root = "decanting/playback/target"
        self.viewer[root].delete()
        if world_T_target is None:
            return
        matrix = np.asarray(world_T_target, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("world_T_target must be a finite 4 x 4 matrix")
        origin = matrix[:3, 3]
        colors = (
            (0xD14A3D, 0x3A9B55, 0x3478C5)
            if successful
            else (0xC94D44, 0xC94D44, 0xC94D44)
        )
        for index, (axis_name, color) in enumerate(zip(("x", "y", "z"), colors)):
            endpoint = origin + axis_length_m * matrix[:3, index]
            self._render_line_segments(
                f"{root}/{axis_name}",
                np.column_stack((origin, endpoint)),
                color,
                opacity=0.98,
                linewidth=3.0,
            )

    def render_process_objects(
        self,
        objects: Iterable[tuple[str, str, Sequence[float], np.ndarray]],
    ) -> None:
        """Replace playback box/tote objects using full World transforms."""

        import meshcat.geometry as geometry

        root = "decanting/playback/objects"
        self.viewer[root].delete()
        for name, role, size_m, world_T_object in objects:
            matrix = np.asarray(world_T_object, dtype=float)
            size = np.asarray(tuple(size_m), dtype=float).reshape(-1)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} playback transform must be finite 4 x 4")
            if size.size != 3 or np.any(size <= 0.0) or not np.all(np.isfinite(size)):
                raise ValueError(f"{name} playback size must contain three positive values")
            color, opacity = ROLE_STYLE.get(role, (0x777777, 0.80))
            node = self.viewer[f"{root}/{name}"]
            node.set_object(
                geometry.Box(size),
                geometry.MeshLambertMaterial(
                    color=color,
                    transparent=opacity < 1.0,
                    opacity=opacity,
                ),
            )
            node.set_transform(matrix)

    def render_study_markers(
        self,
        markers: Iterable[tuple[str, Sequence[Sequence[float]] | np.ndarray, int]] = (),
        segments: Iterable[
            tuple[str, Sequence[float], Sequence[float], int]
        ] = (),
        *,
        axis_length_m: float = 0.12,
    ) -> int:
        """Replace the authored-check pose markers and straight-segment previews.

        Markers are drawn as three World-oriented axes; a segment is the
        straight TCP translation between one check's start and end pose.  This
        is a display overlay only and never participates in collision input.
        """

        root = "decanting/study/markers"
        self.viewer[root].delete()
        if not math.isfinite(axis_length_m) or axis_length_m <= 0.0:
            raise ValueError("axis_length_m must be finite and positive")
        drawn = 0
        for name, transform, color in markers:
            matrix = np.asarray(transform, dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"study marker {name} needs a finite 4 x 4 transform")
            _validate_overlay_color(color, f"study marker {name}")
            node_name = _overlay_node_name(name)
            origin = matrix[:3, 3]
            for index, axis_name in enumerate(("x", "y", "z")):
                endpoint = origin + axis_length_m * matrix[:3, index]
                self._render_line_segments(
                    f"{root}/{node_name}/axis_{axis_name}",
                    np.column_stack((origin, endpoint)),
                    color if index == 2 else (0xD14A3D, 0x3A9B55, 0x3478C5)[index],
                    opacity=0.95,
                    linewidth=3.0,
                )
            drawn += 1
        for name, start, end, color in segments:
            first = np.asarray(tuple(start), dtype=float).reshape(-1)
            second = np.asarray(tuple(end), dtype=float).reshape(-1)
            if first.size != 3 or second.size != 3:
                raise ValueError(f"study segment {name} needs two XYZ points")
            if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
                raise ValueError(f"study segment {name} must be finite")
            _validate_overlay_color(color, f"study segment {name}")
            if float(np.linalg.norm(second - first)) <= 1e-9:
                continue
            self._render_line_segments(
                f"{root}/{_overlay_node_name(name)}/segment",
                np.column_stack((first, second)),
                color,
                opacity=0.9,
                linewidth=2.0,
            )
            drawn += 1
        return drawn

    def render_study_target_highlight(
        self,
        boxes: Iterable[BoxPrimitive] = (),
        meshes: Iterable[MeshObstacle] = (),
        *,
        color: int = 0xE8C33C,
    ) -> int:
        """Outline the selected collision targets without changing the scene."""

        import meshcat.geometry as geometry

        root = "decanting/study/targets"
        self.viewer[root].delete()
        _validate_overlay_color(color, "study highlight")
        drawn = 0
        for box in boxes:
            self._render_line_segments(
                f"{root}/box/{_overlay_node_name(box.name)}",
                self._oriented_box_edge_segments(box),
                color,
                opacity=0.95,
                linewidth=3.0,
            )
            drawn += 1
        for mesh in meshes:
            node = self.viewer[f"{root}/mesh/{_overlay_node_name(mesh.name)}"]
            node.set_object(
                geometry.TriangularMeshGeometry(
                    np.asarray(mesh.vertices_m, dtype=float),
                    np.asarray(mesh.triangles, dtype=np.int32),
                ),
                geometry.MeshLambertMaterial(
                    color=color, transparent=True, opacity=0.45
                ),
            )
            node.set_transform(np.eye(4))
            drawn += 1
        return drawn

    def clear_study_overlays(self) -> None:
        """Remove every study marker, segment and target highlight."""

        self.viewer["decanting/study"].delete()

    def render_sr_keepout(self, box: BoxPrimitive | None) -> None:
        """Show or clear the cached simultaneous-work SR exclusion box."""

        import meshcat.geometry as geometry

        path = "decanting/playback/sr_keepout"
        self.viewer[path].delete()
        if box is None:
            return
        node = self.viewer[path]
        node.set_object(
            geometry.Box(np.asarray(box.size_m, dtype=float)),
            geometry.MeshLambertMaterial(
                color=0xD04A44,
                opacity=0.13,
                transparent=True,
            ),
        )
        node.set_transform(
            pose_matrix(box.center_m, yaw_rad=math.radians(box.yaw_deg))
        )

    def url(self) -> str:
        return str(self.viewer.url())

    def save_static_html(self, path: str | Path) -> Path:
        """Serialize the view; callers must ensure embedded mesh redistribution is allowed."""

        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.viewer.static_html(), encoding="utf-8")
        return output

    def _render_robot(
        self,
        bundle: RobotBundle,
        base: BasePose,
        arm_q: Sequence[float],
        *,
        root_name: str,
        show_collisions: bool,
        visual_only: bool = False,
        native_j3_stroke_m: float | None = None,
    ) -> np.ndarray:
        from pinocchio.visualize import MeshcatVisualizer

        if bundle.visual_model is None:
            raise RuntimeError(
                f"{bundle.name} was loaded without visuals; reload it with load_visual=True"
            )

        visualizer = MeshcatVisualizer(
            bundle.model,
            bundle.collision_model,
            bundle.visual_model,
        )
        visualizer.initViewer(viewer=self.viewer, open=False)
        visualizer.loadViewerModel(rootNodeName=root_name)
        q = floating_configuration(bundle, base, arm_q)
        visualizer.display(q)
        visualizer.displayVisuals(True)
        visualizer.displayCollisions(show_collisions)
        self._robots.append(
            LoadedRobot(
                bundle=bundle,
                visualizer=visualizer,
                visual_only=visual_only,
                native_j3_stroke_m=native_j3_stroke_m,
            )
        )
        return q

    def _render_suction_proxy(
        self,
        bundle: RobotBundle,
        q: np.ndarray,
        proxy: SuctionProxySpec,
        *,
        root_name: str,
    ) -> None:
        """Render the provisional Isaac suction geometry at the UR ``tool0`` frame."""

        import meshcat.geometry as geometry
        cylinder_material = geometry.MeshLambertMaterial(color=0x25282B, opacity=0.96)

        cylinder = self.viewer[f"{root_name}/cylinder"]
        cylinder.set_object(
            geometry.Cylinder(proxy.cylinder_length_m, proxy.cylinder_radius_m),
            cylinder_material,
        )

        pad = self.viewer[f"{root_name}/pad"]
        pad.set_object(
            geometry.Cylinder(proxy.pad_thickness_m, proxy.pad_radius_m),
            cylinder_material,
        )
        self._ur20_suction_attachment = (bundle, proxy, root_name)
        self.update_ur20_suction_proxy(q)

    def _render_cutter_proxy(
        self,
        bundle: RobotBundle,
        q: np.ndarray,
        proxy: CutterProxySpec,
        *,
        root_name: str,
    ) -> None:
        """Render the configured provisional cutter box at ``tool0``."""

        import meshcat.geometry as geometry
        import pinocchio as pin

        frame_id = bundle.model.getFrameId("tool0")
        if frame_id >= bundle.model.nframes:
            raise RuntimeError(f"{bundle.name} has no tool0 frame for the cutter proxy")
        data = bundle.model.createData()
        pin.forwardKinematics(bundle.model, data, q)
        pin.updateFramePlacements(bundle.model, data)
        tool_pose = np.asarray(data.oMf[frame_id].homogeneous)
        local = pose_matrix(proxy.center_local_m)
        node = self.viewer[root_name]
        node.set_object(
            geometry.Box(np.asarray(proxy.size_m)),
            geometry.MeshLambertMaterial(color=0xD75B24, opacity=0.92),
        )
        node.set_transform(tool_pose @ local)

    def _render_box(self, box: BoxPrimitive) -> None:
        import meshcat.geometry as geometry

        color, opacity = ROLE_STYLE.get(box.role, (0x777777, 0.80))
        material = geometry.MeshLambertMaterial(
            color=color,
            transparent=opacity < 1.0,
            opacity=opacity,
        )
        node = self.viewer[f"decanting/environment/{box.role}/{box.name}"]
        node.set_object(geometry.Box(np.asarray(box.size_m)), material)
        node.set_transform(
            pose_matrix(box.center_m, yaw_rad=math.radians(box.yaw_deg))
        )

    def _render_floor(
        self,
        spec: SceneSpec,
        boxes: Iterable[BoxPrimitive],
        meshes: Iterable[MeshObstacle] = (),
    ) -> None:
        import meshcat.geometry as geometry

        bounds = [
            (
                box.center_m[0] - box.size_m[0] / 2.0,
                box.center_m[0] + box.size_m[0] / 2.0,
                box.center_m[1] - box.size_m[1] / 2.0,
                box.center_m[1] + box.size_m[1] / 2.0,
            )
            for box in boxes
        ]
        for mesh in meshes:
            vertices = np.asarray(mesh.vertices_m, dtype=float)
            bounds.append(
                (vertices[:, 0].min(), vertices[:, 0].max(),
                 vertices[:, 1].min(), vertices[:, 1].max())
            )
        if not bounds:
            region = spec.installation_region
            bounds.append((region.raw_xy_min_m[0], region.raw_xy_max_m[0],
                           region.raw_xy_min_m[1], region.raw_xy_max_m[1]))
        x_min = min(bound[0] for bound in bounds) - 0.7
        x_max = max(bound[1] for bound in bounds) + 0.7
        y_min = min(bound[2] for bound in bounds) - 0.7
        y_max = max(bound[3] for bound in bounds) + 0.7
        thickness = 0.02
        node = self.viewer["decanting/environment/floor"]
        node.set_object(
            geometry.Box(np.array((x_max - x_min, y_max - y_min, thickness))),
            geometry.MeshLambertMaterial(color=0xB8BBB8, opacity=0.58, transparent=True),
        )
        node.set_transform(
            pose_matrix(
                ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, spec.floor_z_m - thickness / 2.0)
            )
        )

    def _render_mesh(self, mesh: MeshObstacle) -> None:
        import meshcat.geometry as geometry

        color, opacity = ROLE_STYLE.get(mesh.role, (0x777777, 0.80))
        node = self.viewer[f"decanting/environment/{mesh.role}/{mesh.name}"]
        node.set_object(
            geometry.TriangularMeshGeometry(
                np.asarray(mesh.vertices_m, dtype=float),
                np.asarray(mesh.triangles, dtype=np.int32),
            ),
            geometry.MeshLambertMaterial(
                color=color, transparent=opacity < 1.0, opacity=opacity,
            ),
        )
        node.set_transform(np.eye(4))

    def _render_installation_region(self, spec: SceneSpec) -> None:
        import meshcat.geometry as geometry

        region = spec.installation_region
        width = region.raw_xy_max_m[0] - region.raw_xy_min_m[0]
        depth = region.raw_xy_max_m[1] - region.raw_xy_min_m[1]
        center_x = (region.raw_xy_min_m[0] + region.raw_xy_max_m[0]) / 2.0
        center_y = (region.raw_xy_min_m[1] + region.raw_xy_max_m[1]) / 2.0
        floor_node = self.viewer["decanting/guides/base_candidate_region"]
        floor_node.set_object(
            geometry.Box(np.array((width, depth, 0.012))),
            geometry.MeshLambertMaterial(color=0x22A06B, opacity=0.16, transparent=True),
        )
        floor_node.set_transform(pose_matrix((center_x, center_y, spec.floor_z_m + 0.006)))

        side_plane = self.viewer["decanting/guides/side_access_plane_1296mm"]
        side_plane.set_object(
            geometry.Box(np.array((width, 0.008, 2.5))),
            geometry.MeshLambertMaterial(color=0x22A06B, opacity=0.10, transparent=True),
        )
        side_plane.set_transform(
            pose_matrix((center_x, region.raw_xy_max_m[1], spec.floor_z_m + 1.25))
        )

    def _render_frame(
        self,
        name: str,
        translation_m: Sequence[float],
        quaternion_xyzw: Sequence[float],
        axis_length_m: float = 0.16,
    ) -> None:
        transform = quaternion_matrix(tuple(translation_m), tuple(quaternion_xyzw))
        origin = transform[:3, 3]
        colors = (0xD14A3D, 0x3A9B55, 0x3478C5)
        axis_names = ("x", "y", "z")
        for axis_index, (axis_name, color) in enumerate(zip(axis_names, colors)):
            endpoint = origin + axis_length_m * transform[:3, axis_index]
            self._render_line_segments(
                f"decanting/frames/{name}/{axis_name}",
                np.column_stack((origin, endpoint)),
                color,
                opacity=0.95,
                linewidth=2.0,
            )

    def _render_sr_cutting_workspace(
        self,
        spec: SceneSpec,
        base: BasePose,
        state_clearance_m: float | None,
        *,
        show_fill: bool,
    ) -> None:
        import meshcat.geometry as geometry

        task_box = sr_cutting_workspace_footprint_box(
            spec,
            base,
            clearance_m=state_clearance_m,
        )
        root = "decanting/guides/sr12ia/cutting_workspace"
        self._render_line_segments(
            f"{root}/boundary",
            self._oriented_box_edge_segments(task_box),
            0xD79B2E,
            opacity=0.88,
            linewidth=2.0,
        )

        if show_fill:
            node = self.viewer[f"{root}/volume"]
            node.set_object(
                geometry.Box(np.asarray(task_box.size_m)),
                geometry.MeshLambertMaterial(
                    color=0xD79B2E, opacity=0.10, transparent=True
                ),
            )
            node.set_transform(
                pose_matrix(
                    task_box.center_m,
                    yaw_rad=math.radians(task_box.yaw_deg),
                )
            )

    @staticmethod
    def _oriented_box_edge_segments(
        box: BoxPrimitive,
    ) -> np.ndarray:
        """Return twelve box edges as paired Meshcat line vertices."""

        half = np.asarray(box.size_m, dtype=float) / 2.0
        local_corners = np.asarray(
            [
                (sign_x * half[0], sign_y * half[1], sign_z * half[2])
                for sign_z in (-1.0, 1.0)
                for sign_y in (-1.0, 1.0)
                for sign_x in (-1.0, 1.0)
            ]
        )
        yaw = math.radians(box.yaw_deg)
        rotation = np.asarray(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        corners = local_corners @ rotation.T + np.asarray(box.center_m)
        edges = (
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        points = np.empty((3, len(edges) * 2), dtype=float)
        for index, (start, end) in enumerate(edges):
            points[:, 2 * index] = corners[start]
            points[:, 2 * index + 1] = corners[end]
        return points

    def _render_line_segments(
        self,
        path: str,
        vertices: np.ndarray,
        color: int,
        *,
        opacity: float,
        linewidth: float,
    ) -> None:
        import meshcat.geometry as geometry

        material = geometry.LineBasicMaterial(
            color=color,
            opacity=opacity,
            transparent=opacity < 1.0,
            linewidth=linewidth,
        )
        self.viewer[path].set_object(
            geometry.LineSegments(geometry.PointsGeometry(vertices), material)
        )

    def _set_background(self) -> None:
        self.viewer["/Background"].set_property("top_color", [0.88, 0.90, 0.92])
        self.viewer["/Background"].set_property("bottom_color", [0.72, 0.75, 0.78])


def _visible_environment_boxes(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    *,
    suppress_process_objects: bool = False,
) -> tuple[BoxPrimitive, ...]:
    """Keep workflow proxies available without displaying them over USD meshes."""

    mesh_names = {mesh.name for mesh in snapshot.meshes}
    static_names = {box.name for box in spec.static_boxes}
    return tuple(
        box
        for box in snapshot.boxes
        if box.name not in mesh_names
        and not (
            snapshot.meshes and box.name in static_names and not box.collision_enabled
        )
        and not (
            suppress_process_objects
            and box.role in {"box", "box_sample_alternative", "tote"}
        )
    )


def _overlay_node_name(name: str) -> str:
    """Return a MeshCat path component that cannot create extra nesting."""

    text = str(name).strip()
    if not text:
        raise ValueError("overlay node name must not be empty")
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in text
    )
    return safe[:120]


def _validate_overlay_color(color: object, name: str) -> None:
    if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 0xFFFFFF:
        raise ValueError(f"{name} color must be an RGB integer")


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _joint_upper_limit(bundle: RobotBundle, joint_name: str) -> float:
    """Return one scalar joint's native upper position limit."""

    joint_id = bundle.model.getJointId(joint_name)
    if joint_id >= bundle.model.njoints:
        raise ValueError(f"{bundle.name} has no {joint_name}")
    joint = bundle.model.joints[joint_id]
    if joint.nq != 1:
        raise ValueError(f"{bundle.name} {joint_name} is not one-DOF")
    value = float(bundle.native_upper_position_limits[joint.idx_q])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{bundle.name} {joint_name} has no positive upper limit")
    return value
