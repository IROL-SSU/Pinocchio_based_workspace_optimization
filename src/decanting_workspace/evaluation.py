"""Collision-aware Pinocchio evaluation of UR20 process pose segments.

This is the layer immediately before base-layout optimization.  It evaluates
one already-expanded process scenario and returns raw feasibility and
Jacobian-ellipsoid metrics at every sampled pose.  It deliberately does not
combine them into an objective or rank base candidates.

The local Cartesian and joint interpolation used here is a diagnostic path
check, not a global motion planner.  Consequently ``not_found`` means that
the deterministic multi-start search did not find a solution; it is not a
mathematical proof that no solution or collision-free path exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from .collision_world import (
    AttachedBox,
    CollisionPhase,
    CollisionReport,
    PinocchioCollisionChecker,
)
from .kinematics import (
    IKOptions,
    IKResult,
    ManipulabilityMetrics,
    compute_manipulability,
    solve_frame_ik,
)
from .models import BoxPrimitive, SceneSpec
from .robots import RobotBundle
from .scene import SceneSnapshot
from .workflow import (
    Criticality,
    LinearPoseCheck,
    ProcessScenario,
    RigidPose,
    TaskStepSpec,
)


CheckStatus = Literal[
    "success",
    "ik_not_found",
    "solution_in_collision",
    "joint_segment_collision",
]
StepStatus = Literal["success", "failed", "skipped"]


@dataclass(frozen=True)
class EvaluationOptions:
    """Sampling and metric settings for one candidate evaluation."""

    cartesian_translation_step_m: float = 0.05
    cartesian_rotation_step_rad: float = math.radians(5.0)
    joint_interpolation_samples: int = 3
    characteristic_length_m: float = 0.5
    ik: IKOptions = IKOptions()
    extra_arm_seeds: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        positive = {
            "cartesian_translation_step_m": self.cartesian_translation_step_m,
            "cartesian_rotation_step_rad": self.cartesian_rotation_step_rad,
            "characteristic_length_m": self.characteristic_length_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.joint_interpolation_samples < 0:
            raise ValueError("joint_interpolation_samples must be non-negative")


@dataclass(frozen=True)
class PoseSampleResult:
    fraction: float
    target: RigidPose
    ik: IKResult
    collision: CollisionReport | None
    manipulability: ManipulabilityMetrics | None


@dataclass(frozen=True)
class LinearCheckResult:
    name: str
    criticality: Criticality
    status: CheckStatus
    samples: tuple[PoseSampleResult, ...]
    failed_sample_index: int | None = None
    joint_collision: CollisionReport | None = None
    include_in_metric_summary: bool = True

    @property
    def success(self) -> bool:
        return self.status == "success"


@dataclass(frozen=True)
class StepEvaluation:
    index: int
    name: str
    criticality: Criticality
    status: StepStatus
    checks: tuple[LinearCheckResult, ...]
    waste_dimension_fits: bool | None = None

    @property
    def success(self) -> bool:
        return self.status in {"success", "skipped"}

    @property
    def soft_bonus_success(self) -> bool:
        """Whether both optional pose and dimension checks succeeded."""

        return self.success and self.waste_dimension_fits is not False


@dataclass(frozen=True)
class MetricSummary:
    sample_count: int
    translational_volume_min: float
    translational_volume_mean: float
    translational_sigma_min: float
    normalized_volume_min: float
    normalized_volume_mean: float
    normalized_sigma_min: float
    normalized_condition_max: float


@dataclass(frozen=True)
class ScenarioEvaluation:
    scenario: ProcessScenario
    steps: tuple[StepEvaluation, ...]
    hard_feasible: bool
    soft_step_successes: int
    soft_step_count: int
    metrics: MetricSummary | None


def evaluate_task_sequence(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    scenario: ProcessScenario,
    ur_bundle: RobotBundle,
    steps: Sequence[TaskStepSpec],
    *,
    options: EvaluationOptions | None = None,
) -> ScenarioEvaluation:
    """Evaluate one task sequence with fixed-base Pinocchio IK and collision.

    The previous successful waypoint is the first seed for the next target.
    The configured UR nominal posture and caller-supplied branch seeds are
    always retained as deterministic fallbacks.
    """

    settings = options or EvaluationOptions()
    nominal_arm = tuple(float(value) for value in spec.robots["ur20"].nominal_q)
    previous_arm: np.ndarray | None = None
    step_results: list[StepEvaluation] = []

    for step in steps:
        if step.criticality is Criticality.SKIP:
            step_results.append(
                StepEvaluation(
                    index=step.index,
                    name=step.name,
                    criticality=step.criticality,
                    status="skipped",
                    checks=(),
                    waste_dimension_fits=(
                        None if step.waste_fit is None else step.waste_fit.fits
                    ),
                )
            )
            continue

        check_results: list[LinearCheckResult] = []
        for check in step.pose_checks:
            phase = collision_phase_for_check(spec, snapshot, step, check)
            checker = PinocchioCollisionChecker(ur_bundle, phase)
            seeds: list[Sequence[float]] = []
            if previous_arm is not None:
                seeds.append(previous_arm)
            seeds.append(nominal_arm)
            seeds.extend(settings.extra_arm_seeds)
            check_result, last_arm = _evaluate_linear_check(
                snapshot,
                ur_bundle,
                check,
                checker,
                seeds,
                settings,
            )
            check_results.append(check_result)
            if last_arm is not None:
                previous_arm = last_arm

        succeeded = all(result.success for result in check_results)
        # The optional waste step awards success only when both independent
        # questions are positive: the empty box fits the level-3 conveyor and
        # its release pose is feasible.  It remains a SOFT step, so this does
        # not affect ``hard_feasible``.
        if step.waste_fit is not None:
            succeeded = succeeded and step.waste_fit.fits
        step_results.append(
            StepEvaluation(
                index=step.index,
                name=step.name,
                criticality=step.criticality,
                status="success" if succeeded else "failed",
                checks=tuple(check_results),
                waste_dimension_fits=(
                    None if step.waste_fit is None else step.waste_fit.fits
                ),
            )
        )

    hard_feasible = all(
        result.success
        for result in step_results
        if result.criticality is Criticality.HARD
    )
    soft = [
        result
        for result in step_results
        if result.criticality is Criticality.SOFT
    ]
    metrics = _summarize_metrics(step_results)
    return ScenarioEvaluation(
        scenario=scenario,
        steps=tuple(step_results),
        hard_feasible=hard_feasible,
        soft_step_successes=sum(result.soft_bonus_success for result in soft),
        soft_step_count=len(soft),
        metrics=metrics,
    )


def collision_phase_for_check(
    spec: SceneSpec,
    snapshot: SceneSnapshot,
    step: TaskStepSpec,
    check: LinearPoseCheck,
) -> CollisionPhase:
    """Translate workflow object state into a concrete Pinocchio phase."""

    suppressed = set(step.suppress_snapshot_objects)
    obstacles: list[BoxPrimitive] = [
        box
        for box in snapshot.collision_boxes
        if box.name not in suppressed
    ]
    obstacles.extend(step.active_extra_obstacles)
    attached: list[AttachedBox] = []

    for state in check.object_states:
        if state.attached_to_tcp or state.follows_tcp:
            if state.tcp_T_object is None:  # guarded by ObjectState validation
                raise ValueError(f"TCP-coupled object {state.name} has no transform")
            attached.append(
                AttachedBox(
                    name=state.name,
                    size_m=state.size_m,
                    frame_name="suction_tcp",
                    frame_T_box=state.tcp_T_object.matrix,
                )
            )
        else:
            obstacles.append(_object_state_box(state))

    # Logical names are intentionally preserved by the collision checker.
    # Accept the older workflow spelling as an alias for the actual pad
    # collision shape rather than exempting every geometry on the TCP joint.
    allowed = frozenset(
        (
            "suction_pad" if first == "suction_tcp" else first,
            "suction_pad" if second == "suction_tcp" else second,
        )
        for first, second in check.allowed_contacts
    )
    collision_boxes = _deduplicate_boxes(obstacles)
    collision_meshes = tuple(
        mesh for mesh in snapshot.collision_meshes if mesh.name not in suppressed
    )
    names = [box.name for box in collision_boxes] + [mesh.name for mesh in collision_meshes]
    if len(names) != len(set(names)):
        raise ValueError("duplicate collision obstacle name between boxes and meshes")
    return CollisionPhase(
        obstacles=collision_boxes,
        attached_boxes=tuple(attached),
        allowed_contacts=allowed,
        floor_z_m=spec.floor_z_m,
        meshes=collision_meshes,
    )


def interpolate_rigid_poses(
    start: RigidPose,
    end: RigidPose,
    *,
    translation_step_m: float,
    rotation_step_rad: float,
) -> tuple[tuple[float, RigidPose], ...]:
    """Sample a straight translation and shortest SO(3) interpolation."""

    import pinocchio as pin

    translation_distance = float(
        np.linalg.norm(
            np.asarray(end.translation_m) - np.asarray(start.translation_m)
        )
    )
    relative_log = np.asarray(
        pin.log3(start.rotation.T @ end.rotation),
        dtype=float,
    ).reshape(3)
    rotation_distance = float(np.linalg.norm(relative_log))
    intervals = max(
        1,
        int(math.ceil(translation_distance / translation_step_m)),
        int(math.ceil(rotation_distance / rotation_step_rad)),
    )
    if translation_distance <= 1e-12 and rotation_distance <= 1e-12:
        intervals = 0

    result: list[tuple[float, RigidPose]] = []
    fractions = (0.0,) if intervals == 0 else np.linspace(0.0, 1.0, intervals + 1)
    start_translation = np.asarray(start.translation_m)
    end_translation = np.asarray(end.translation_m)
    for raw_fraction in fractions:
        fraction = float(raw_fraction)
        matrix = np.eye(4)
        matrix[:3, :3] = start.rotation @ pin.exp3(fraction * relative_log)
        matrix[:3, 3] = (
            (1.0 - fraction) * start_translation
            + fraction * end_translation
        )
        result.append((fraction, RigidPose(matrix)))
    return tuple(result)


def evaluate_linear_check(
    snapshot: SceneSnapshot,
    bundle: RobotBundle,
    check: LinearPoseCheck,
    checker: PinocchioCollisionChecker,
    initial_seeds: Sequence[Sequence[float]],
    options: EvaluationOptions,
) -> tuple[LinearCheckResult, np.ndarray | None]:
    """Evaluate exactly one pose segment with a caller-supplied phase checker.

    This is the same routine :func:`evaluate_task_sequence` uses for every
    generated check.  It is exported so a user-authored study can evaluate
    individual checks without reproducing the sampling, seeding, and
    joint-segment rules, and therefore without drifting from the default
    process evaluation.
    """

    return _evaluate_linear_check(
        snapshot, bundle, check, checker, initial_seeds, options
    )


def _evaluate_linear_check(
    snapshot: SceneSnapshot,
    bundle: RobotBundle,
    check: LinearPoseCheck,
    checker: PinocchioCollisionChecker,
    initial_seeds: Sequence[Sequence[float]],
    options: EvaluationOptions,
) -> tuple[LinearCheckResult, np.ndarray | None]:
    import pinocchio as pin

    targets = interpolate_rigid_poses(
        check.start_tcp,
        check.end_tcp,
        translation_step_m=options.cartesian_translation_step_m,
        rotation_step_rad=options.cartesian_rotation_step_rad,
    )
    samples: list[PoseSampleResult] = []
    seeds = list(initial_seeds)
    previous_q: np.ndarray | None = None
    last_arm: np.ndarray | None = None

    for sample_index, (fraction, target) in enumerate(targets):
        rejected_endpoint_reports: list[CollisionReport] = []
        rejected_joint_reports: list[CollisionReport] = []

        def accept(configuration: np.ndarray) -> bool:
            report = checker.check(configuration)
            if report.in_collision:
                rejected_endpoint_reports.append(report)
                return False
            # A collision-free endpoint is not enough when this Cartesian
            # sample follows another one. Reject a branch whose local joint
            # segment collides so solve_frame_ik can continue with the next
            # deterministic seed instead of returning a false-negative for
            # the whole check after the first endpoint solution.
            if previous_q is not None and options.joint_interpolation_samples:
                for joint_index in range(1, options.joint_interpolation_samples + 1):
                    joint_fraction = joint_index / (
                        options.joint_interpolation_samples + 1
                    )
                    interpolated_q = np.asarray(
                        pin.interpolate(
                            bundle.model,
                            previous_q,
                            configuration,
                            joint_fraction,
                        ),
                        dtype=float,
                    )
                    joint_report = checker.check(interpolated_q)
                    if joint_report.in_collision:
                        rejected_joint_reports.append(joint_report)
                        return False
            return True

        ik_result = solve_frame_ik(
            bundle,
            snapshot.ur_base,
            target.matrix,
            frame_name="suction_tcp",
            seeds_arm=seeds,
            options=options.ik,
            accept_configuration=accept,
        )
        if not ik_result.success:
            if rejected_joint_reports:
                status: CheckStatus = "joint_segment_collision"
            elif rejected_endpoint_reports:
                status = "solution_in_collision"
            else:
                status = "ik_not_found"
            collision = (
                rejected_endpoint_reports[0]
                if status == "solution_in_collision"
                else None
            )
            samples.append(
                PoseSampleResult(
                    fraction=fraction,
                    target=target,
                    ik=ik_result,
                    collision=collision,
                    manipulability=None,
                )
            )
            return (
                LinearCheckResult(
                    name=check.name,
                    criticality=check.criticality,
                    status=status,
                    samples=tuple(samples),
                    failed_sample_index=sample_index,
                    joint_collision=(
                        rejected_joint_reports[0]
                        if rejected_joint_reports
                        else None
                    ),
                    include_in_metric_summary=check.include_in_metric_summary,
                ),
                last_arm,
            )

        collision = checker.check(ik_result.q)
        if collision.in_collision:  # defensive: the acceptor should prevent this
            samples.append(
                PoseSampleResult(fraction, target, ik_result, collision, None)
            )
            return (
                LinearCheckResult(
                    name=check.name,
                    criticality=check.criticality,
                    status="solution_in_collision",
                    samples=tuple(samples),
                    failed_sample_index=sample_index,
                    include_in_metric_summary=check.include_in_metric_summary,
                ),
                last_arm,
            )

        # Defensive recheck: the same segment was part of the IK acceptance
        # callback above, but retaining this guard catches a broken/custom IK
        # implementation that does not call its acceptor.
        if previous_q is not None and options.joint_interpolation_samples:
            for joint_index in range(1, options.joint_interpolation_samples + 1):
                joint_fraction = joint_index / (
                    options.joint_interpolation_samples + 1
                )
                interpolated_q = np.asarray(
                    pin.interpolate(
                        bundle.model,
                        previous_q,
                        ik_result.q,
                        joint_fraction,
                    ),
                    dtype=float,
                )
                joint_report = checker.check(interpolated_q)
                if joint_report.in_collision:
                    samples.append(
                        PoseSampleResult(
                            fraction,
                            target,
                            ik_result,
                            collision,
                            None,
                        )
                    )
                    return (
                        LinearCheckResult(
                            name=check.name,
                            criticality=check.criticality,
                            status="joint_segment_collision",
                            samples=tuple(samples),
                            failed_sample_index=sample_index,
                            joint_collision=joint_report,
                            include_in_metric_summary=(
                                check.include_in_metric_summary
                            ),
                        ),
                        last_arm,
                    )

        metrics = compute_manipulability(
            bundle,
            ik_result.q,
            frame_name="suction_tcp",
            characteristic_length_m=options.characteristic_length_m,
        )
        samples.append(
            PoseSampleResult(
                fraction=fraction,
                target=target,
                ik=ik_result,
                collision=collision,
                manipulability=metrics,
            )
        )
        previous_q = np.array(ik_result.q, copy=True)
        last_arm = np.array(ik_result.arm_q, copy=True)
        seeds = [last_arm, *initial_seeds]

    return (
        LinearCheckResult(
            name=check.name,
            criticality=check.criticality,
            status="success",
            samples=tuple(samples),
            include_in_metric_summary=check.include_in_metric_summary,
        ),
        last_arm,
    )


def _object_state_box(state: object) -> BoxPrimitive:
    rotation = state.world_T_object.rotation
    if not (
        np.allclose(rotation[2, :], (0.0, 0.0, 1.0), atol=1e-8)
        and np.allclose(rotation[:, 2], (0.0, 0.0, 1.0), atol=1e-8)
    ):
        raise ValueError(
            f"world object {state.name} is not a yaw-only collision box"
        )
    yaw_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return BoxPrimitive(
        name=state.name,
        role="process_object",
        center_m=state.world_T_object.translation_m,
        size_m=state.size_m,
        yaw_deg=yaw_deg,
        collision_enabled=True,
    )


def _deduplicate_boxes(boxes: Sequence[BoxPrimitive]) -> tuple[BoxPrimitive, ...]:
    result: list[BoxPrimitive] = []
    names: set[str] = set()
    for box in boxes:
        if box.name in names:
            raise ValueError(f"duplicate collision obstacle name: {box.name}")
        names.add(box.name)
        result.append(box)
    return tuple(result)


def _summarize_metrics(
    steps: Sequence[StepEvaluation],
) -> MetricSummary | None:
    metrics = [
        sample.manipulability
        for step in steps
        if step.criticality is Criticality.HARD
        for check in step.checks
        if check.include_in_metric_summary
        for sample in check.samples
        if sample.manipulability is not None
    ]
    if not metrics:
        return None
    translation_volumes = np.asarray(
        [metric.translational_volume for metric in metrics]
    )
    normalized_volumes = np.asarray(
        [metric.normalized_volume for metric in metrics]
    )
    return MetricSummary(
        sample_count=len(metrics),
        translational_volume_min=float(np.min(translation_volumes)),
        translational_volume_mean=float(np.mean(translation_volumes)),
        translational_sigma_min=float(
            min(metric.translational_min_singular_value for metric in metrics)
        ),
        normalized_volume_min=float(np.min(normalized_volumes)),
        normalized_volume_mean=float(np.mean(normalized_volumes)),
        normalized_sigma_min=float(
            min(metric.normalized_min_singular_value for metric in metrics)
        ),
        normalized_condition_max=float(
            max(metric.normalized_condition_number for metric in metrics)
        ),
    )
