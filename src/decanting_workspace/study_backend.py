"""Backend for the user-authored UR20 check study.

The study shares the already-loaded scene, robot bundles and MeshCat viewer
with the live evaluation UI.  It never changes the meaning of the original
``/api/evaluate`` result: a study result is a separate payload with its own
status vocabulary and its own saved recipes.

Three invariants shape this module.

* Actual calculation is serialized.  Every Pinocchio/viewer touch happens
  while holding a lock shared with the live backend.  The lock is taken per
  check so a long run cannot block the live UI, and so cancellation lands on
  a check boundary.
* Nothing immutable is mutated in place.  Scene specs, mesh geometry, object
  states and collision phases are rebuilt with :func:`dataclasses.replace`,
  which also preserves the identity-keyed BVH cache for USD meshes.
* A weakened collision contract can never be certified.  Excluding targets,
  adding allowed contacts, or dropping floor/self collision forces the result
  into a diagnostic state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import itertools
import math
import random
import threading
from time import perf_counter, time
from typing import Callable, Mapping, Sequence
import uuid

import numpy as np

from .collision import sr_concurrent_exclusion_box
from .collision_world import CollisionPhase, PinocchioCollisionChecker
from .evaluation import (
    EvaluationOptions,
    LinearCheckResult,
    collision_phase_for_check,
    evaluate_linear_check,
)
from .models import BasePose, BoxPrimitive, MeshObstacle, SceneSpec, SceneState
from .precompute import CaseKey, scene_spec_fingerprint
from .robots import RobotBundle, configure_sr_j3_stroke
from .scene import CORNER_SIGNS, SceneSnapshot, materialize_scene
from .study_recipe import (
    STUDY_SCHEMA_VERSION,
    CollisionScenario,
    Pose6D,
    RecipeError,
    StudyCheck,
    StudyRecipe,
    TemplateRef,
    WORLD_FRAME,
    default_collision_scenario,
)
from .study_store import StudyRecipeStore
from .transforms import inverse_transform, quaternion_matrix
from .workflow import (
    CoordinationMode,
    Criticality,
    LinearPoseCheck,
    ObjectState,
    ProcessScenario,
    RigidPose,
    TaskStepSpec,
    build_ur20_task_sequence,
)


STUDY_API_VERSION = 1
PROCESS_OBJECT_ROLES = frozenset({"box", "box_sample_alternative", "tote"})
MAX_SCOUT_CANDIDATES = 64
MAX_JOB_HISTORY = 8

_MARKER_COLOR_HARD = 0x3478C5
_MARKER_COLOR_SOFT = 0xD2A33B


# --------------------------------------------------------------------------
# Catalog types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CollisionTarget:
    """One nameable collision input the user can exclude or highlight."""

    name: str
    role: str
    source: str
    geometry_kind: str
    source_prim_path: str = ""
    detail: str = ""

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "source": self.source,
            "geometry_kind": self.geometry_kind,
            "source_prim_path": self.source_prim_path,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CheckTemplate:
    """One generated workflow element that can seed an authored check."""

    reference: TemplateRef
    label: str
    step_name: str
    criticality: str
    payload_name: str
    contact_object_name: str
    object_names: tuple[str, ...]
    allowed_contacts: tuple[tuple[str, str], ...]
    start: Pose6D | None
    end: Pose6D | None

    def payload(self) -> dict[str, object]:
        return {
            "token": self.reference.token,
            "reference": self.reference.payload(),
            "label": self.label,
            "step_name": self.step_name,
            "criticality": self.criticality,
            "payload_name": self.payload_name,
            "contact_object_name": self.contact_object_name,
            "object_names": list(self.object_names),
            "allowed_contacts": [list(pair) for pair in self.allowed_contacts],
            "supplies_pose": self.reference.supplies_pose,
            "start": None if self.start is None else self.start.payload(),
            "end": None if self.end is None else self.end.payload(),
        }


@dataclass(frozen=True)
class ResolvedCheck:
    """An authored check translated into actual evaluator input."""

    recipe_check: StudyCheck
    pose_check: LinearPoseCheck
    phase: CollisionPhase
    scenario: CollisionScenario
    include_self_collision: bool
    world_start: np.ndarray
    world_end: np.ndarray
    excluded_applied: tuple[str, ...]
    missing_excluded: tuple[str, ...]


# --------------------------------------------------------------------------
# Scene/workflow context
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkflowContext:
    """One materialized scene plus its generated task sequence."""

    key: CaseKey
    state: SceneState
    snapshot: SceneSnapshot
    steps: tuple[TaskStepSpec, ...]

    def step(self, index: int) -> TaskStepSpec:
        for step in self.steps:
            if step.index == index:
                return step
        raise RecipeError(f"generated workflow has no step {index}")

    def template_pose_check(self, reference: TemplateRef) -> LinearPoseCheck:
        step = self.step(reference.step_index)
        for check in step.pose_checks:
            if check.name == reference.check_name:
                return check
        raise RecipeError(
            f"generated step {reference.step_index} has no pose check "
            f"{reference.check_name!r}"
        )


# --------------------------------------------------------------------------
# Job bookkeeping
# --------------------------------------------------------------------------


@dataclass
class StudyJob:
    """Mutable progress record for one asynchronous study run."""

    job_id: str
    kind: str
    recipe_hash: str
    recipe_id: str
    total: int
    started_unix: float
    state: str = "running"
    completed: int = 0
    current_label: str = ""
    message: str = ""
    result: dict[str, object] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "recipe_hash": self.recipe_hash,
            "recipe_id": self.recipe_id,
            "total": self.total,
            "completed": self.completed,
            "current_label": self.current_label,
            "message": self.message,
            "started_unix": self.started_unix,
            "cancel_requested": self.cancel_event.is_set(),
            "result": self.result,
        }


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


class StudyEvaluationBackend:
    """Author, validate, store and evaluate user-defined UR20 check sets."""

    def __init__(
        self,
        spec: SceneSpec,
        initial_key: CaseKey,
        ur_bundle: RobotBundle,
        sr_bundle: RobotBundle,
        viewer: object | None,
        *,
        profiles: Mapping[str, EvaluationOptions],
        default_profile: str,
        compute_lock: threading.Lock | None = None,
        store: StudyRecipeStore | None = None,
        approach_distance_m: float = 0.15,
        keepout_margin_m: float = 0.0,
        tote_present: bool = True,
        clock: Callable[[], float] = perf_counter,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
    ) -> None:
        if not profiles:
            raise ValueError("at least one study evaluation profile is required")
        if default_profile not in profiles:
            raise ValueError(f"unknown default study profile: {default_profile}")
        self.spec = spec
        self.initial_key = initial_key
        self.ur_bundle = ur_bundle
        self.sr_bundle = sr_bundle
        self.viewer = viewer
        self.profiles = dict(profiles)
        self.default_profile = default_profile
        self.compute_lock = compute_lock or threading.Lock()
        self.store = store or StudyRecipeStore()
        self.approach_distance_m = _positive(approach_distance_m, "approach_distance_m")
        self.keepout_margin_m = _nonnegative(keepout_margin_m, "keepout_margin_m")
        self.tote_present = bool(tote_present)
        self.clock = clock
        self.thread_factory = thread_factory
        self.scene_fingerprint = scene_spec_fingerprint(spec)
        self._jobs: dict[str, StudyJob] = {}
        self._job_order: list[str] = []
        self._active_job_id: str | None = None
        self._jobs_lock = threading.Lock()
        self._playback = None

    # -- catalog ---------------------------------------------------------

    def catalog(self, request: Mapping[str, object] | None = None) -> Mapping[str, object]:
        """Return templates, collision targets, frames and a baseline recipe."""

        parameters = None
        if isinstance(request, Mapping) and request.get("parameters") is not None:
            parameters = _required_mapping(request.get("parameters"), "parameters")
        context = self._workflow_context(parameters)
        templates = self._templates(context)
        targets = self._collision_targets(context)
        baseline = self.baseline_recipe(context, templates)
        return {
            "schema_version": STUDY_API_VERSION,
            "recipe_schema_version": STUDY_SCHEMA_VERSION,
            "mode": "study",
            "scene_fingerprint": self.scene_fingerprint,
            "parameters": _parameter_payload(context.key),
            "profiles": [
                {
                    "id": profile_id,
                    "label": profile_id,
                    "cartesian_translation_step_m": options.cartesian_translation_step_m,
                    "cartesian_rotation_step_deg": math.degrees(
                        options.cartesian_rotation_step_rad
                    ),
                    "joint_interpolation_samples": options.joint_interpolation_samples,
                }
                for profile_id, options in self.profiles.items()
            ],
            "default_profile": self.default_profile,
            "approach_distance_m": self.approach_distance_m,
            "templates": [template.payload() for template in templates],
            "collision_targets": [target.payload() for target in targets],
            "reference_frames": self._reference_frames(context),
            "contact_names": sorted(self._valid_contact_names(context)),
            "baseline_recipe": baseline.payload(),
            "units": {
                "ui_length": "mm",
                "ui_angle": "deg",
                "stored_length": "m",
                "stored_angle": "deg",
                "rotation_convention": "Rz(yaw) Ry(pitch) Rx(roll)",
                "template_relative_order": (
                    "tcp_local: world_T_new = world_T_template * offset; "
                    "world: world_T_new = offset * world_T_template"
                ),
                "absolute_order": "world_T_new = world_T_reference_frame * pose",
            },
            "limits": {
                "max_scout_candidates": MAX_SCOUT_CANDIDATES,
            },
            "saved_recipes": [info.payload() for info in self.store.listing()],
            "evaluation_serialized": True,
        }

    def baseline_recipe(
        self,
        context: WorkflowContext | None = None,
        templates: Sequence[CheckTemplate] | None = None,
    ) -> StudyRecipe:
        """Return the generated process checks as an editable zero-offset recipe."""

        resolved_context = context or self._workflow_context(None)
        resolved_templates = (
            templates
            if templates is not None
            else self._templates(resolved_context)
        )
        checks: list[StudyCheck] = []
        used: set[str] = set()
        for template in resolved_templates:
            if not template.reference.supplies_pose:
                continue
            if template.criticality not in {"hard", "soft"}:
                continue
            check_id = _unique_identifier(
                f"s{template.reference.step_index}-{template.reference.check_name}",
                used,
            )
            checks.append(
                StudyCheck(
                    check_id=check_id,
                    label=template.label,
                    criticality=template.criticality,
                    enabled=True,
                    scenario_id="default",
                    pose_mode="template_relative",
                    template=template.reference,
                    payload_source="template",
                    offset_frame="tcp_local",
                    source_check_name=template.reference.check_name,
                )
            )
        return StudyRecipe(
            recipe_id="baseline",
            label="Generated process checks (zero offset)",
            parameters=_parameter_payload(resolved_context.key),
            scene_fingerprint=self.scene_fingerprint,
            profile=self.default_profile,
            approach_distance_m=self.approach_distance_m,
            checks=tuple(checks),
            scenarios=(default_collision_scenario(),),
        )

    # -- validation / preview -------------------------------------------

    def validate(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Strictly validate a recipe and report its resolved poses."""

        recipe, context, allow_mismatch = self._recipe_from_request(request)
        resolved = self._resolve_checks(recipe, context)
        coverage = self._coverage(recipe, context)
        return {
            "schema_version": STUDY_API_VERSION,
            "valid": True,
            "recipe_hash": recipe.recipe_hash,
            "recipe": recipe.payload(),
            "fingerprint_match": self._fingerprint_match(recipe),
            "fingerprint_override": bool(allow_mismatch),
            "expected_scene_fingerprint": self.scene_fingerprint,
            "diagnostic": self._diagnostic_report(recipe, resolved, coverage),
            "coverage": coverage,
            "resolved_checks": [
                _resolved_check_payload(item) for item in resolved
            ],
        }

    def preview(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Draw authored pose markers, segments and target highlights."""

        recipe, context, _ = self._recipe_from_request(request)
        resolved = self._resolve_checks(recipe, context)
        selected_ids = _optional_string_list(
            request.get("check_ids"), "check_ids"
        )
        highlight = _optional_string_list(
            request.get("highlight_targets"), "highlight_targets"
        )
        show_markers = _boolean(request.get("show_markers", True), "show_markers")
        if selected_ids is not None:
            wanted = set(selected_ids)
            unknown = sorted(wanted - {item.recipe_check.check_id for item in resolved})
            if unknown:
                raise RecipeError(f"unknown check ids: {unknown}")
            shown = [item for item in resolved if item.recipe_check.check_id in wanted]
        else:
            shown = [item for item in resolved if item.recipe_check.enabled]

        markers: list[tuple[str, np.ndarray, int]] = []
        segments: list[tuple[str, Sequence[float], Sequence[float], int]] = []
        if show_markers:
            for item in shown:
                name = item.recipe_check.check_id
                color = (
                    _MARKER_COLOR_HARD
                    if item.recipe_check.criticality_is_hard
                    else _MARKER_COLOR_SOFT
                )
                markers.append((f"{name}_start", item.world_start, color))
                markers.append((f"{name}_end", item.world_end, color))
                segments.append(
                    (
                        name,
                        item.world_start[:3, 3],
                        item.world_end[:3, 3],
                        color,
                    )
                )

        boxes, meshes, missing = self._highlight_geometry(context, highlight or ())
        marker_count = 0
        highlight_count = 0
        if self.viewer is not None:
            with self.compute_lock:
                marker_count = self.viewer.render_study_markers(markers, segments)
                highlight_count = self.viewer.render_study_target_highlight(
                    boxes, meshes
                )
        return {
            "schema_version": STUDY_API_VERSION,
            "recipe_hash": recipe.recipe_hash,
            "marker_nodes": marker_count,
            "highlighted": highlight_count,
            "missing_highlight_targets": list(missing),
            "previewed_checks": [item.recipe_check.check_id for item in shown],
            "viewer_available": self.viewer is not None,
        }

    def clear_preview(self, request: Mapping[str, object] | None = None) -> Mapping[str, object]:
        """Remove every study overlay from the shared viewer."""

        if self.viewer is not None:
            with self.compute_lock:
                self.viewer.clear_study_overlays()
        return {"schema_version": STUDY_API_VERSION, "cleared": True}

    # -- recipe storage --------------------------------------------------

    def save_recipe(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Validate then atomically store a recipe in the study directory."""

        recipe, context, _ = self._recipe_from_request(request)
        # Resolving before saving refuses recipes that cannot be evaluated at
        # all (unknown frame, missing template, unknown contact name).
        self._resolve_checks(recipe, context)
        stamped = recipe.with_fingerprint(self.scene_fingerprint)
        info = self.store.save(stamped)
        return {
            "schema_version": STUDY_API_VERSION,
            "saved": info.payload(),
            "recipe_hash": stamped.recipe_hash,
            "recipe": stamped.payload(),
            "saved_recipes": [item.payload() for item in self.store.listing()],
            "directory_name": self.store.directory.name,
        }

    def load_recipe(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Read a stored recipe and report whether it still fits this scene."""

        recipe_id = request.get("recipe_id")
        if not isinstance(recipe_id, str):
            raise RecipeError("recipe_id must be a string")
        recipe = self.store.load(recipe_id)
        return self._applicability(recipe, request)

    def import_recipe(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Validate an uploaded recipe document without storing it."""

        document = request.get("document")
        if isinstance(document, str):
            recipe = StudyRecipe.from_json(document)
        elif isinstance(document, Mapping):
            recipe = StudyRecipe.from_mapping(document, "document")
        else:
            raise RecipeError("document must be a JSON object or JSON text")
        return self._applicability(recipe, request)

    def list_recipes(self, request: Mapping[str, object] | None = None) -> Mapping[str, object]:
        return {
            "schema_version": STUDY_API_VERSION,
            "saved_recipes": [item.payload() for item in self.store.listing()],
            "directory_name": self.store.directory.name,
            "scene_fingerprint": self.scene_fingerprint,
        }

    def _applicability(
        self,
        recipe: StudyRecipe,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        match = self._fingerprint_match(recipe)
        payload: dict[str, object] = {
            "schema_version": STUDY_API_VERSION,
            "recipe": recipe.payload(),
            "recipe_hash": recipe.recipe_hash,
            "fingerprint_match": match,
            "expected_scene_fingerprint": self.scene_fingerprint,
            "applied": False,
            "resolve_error": None,
        }
        if not match:
            payload["message"] = (
                "scene/model fingerprint differs from this session; the recipe "
                "was loaded for review but not applied. Re-check it and pass "
                "allow_fingerprint_mismatch to evaluate it as a diagnostic."
            )
            return payload
        try:
            context = self._workflow_context(recipe.parameters or None)
            self._resolve_checks(recipe, context)
        except (RecipeError, ValueError, KeyError) as exc:
            payload["resolve_error"] = str(exc)
            payload["message"] = (
                "fingerprint matches but the recipe does not resolve against "
                f"the current scene: {exc}"
            )
            return payload
        payload["applied"] = True
        payload["message"] = "recipe matches this scene and resolves completely"
        return payload

    # -- evaluation ------------------------------------------------------

    def start_evaluation(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Validate a recipe and start one asynchronous evaluation job."""

        recipe, context, allow_mismatch = self._recipe_from_request(request)
        selected = _optional_string_list(request.get("check_ids"), "check_ids")
        resolved = self._resolve_checks(recipe, context)
        if selected is not None:
            wanted = set(selected)
            unknown = sorted(wanted - {item.recipe_check.check_id for item in resolved})
            if unknown:
                raise RecipeError(f"unknown check ids: {unknown}")
            queue = [
                item for item in resolved if item.recipe_check.check_id in wanted
            ]
        else:
            queue = [item for item in resolved if item.recipe_check.enabled]
        if not queue:
            raise RecipeError(
                "no check selected; enable at least one check or pass check_ids"
            )
        options = self._options(recipe.profile)
        job = self._create_job(
            kind="evaluate",
            recipe=recipe,
            total=len(queue),
        )
        thread = self.thread_factory(
            target=self._run_evaluation_job,
            args=(job, recipe, context, queue, options, bool(allow_mismatch), selected),
            name=f"study-eval-{job.job_id}",
            daemon=True,
        )
        thread.start()
        return {
            "schema_version": STUDY_API_VERSION,
            "job": job.snapshot(),
            "accepted": True,
        }

    def job_status(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Return one job's progress without touching Pinocchio or the viewer."""

        job_id = request.get("job_id")
        with self._jobs_lock:
            if job_id is None:
                job = self._jobs.get(self._active_job_id) if self._active_job_id else None
                if job is None and self._job_order:
                    job = self._jobs[self._job_order[-1]]
            else:
                if not isinstance(job_id, str):
                    raise RecipeError("job_id must be a string")
                job = self._jobs.get(job_id)
                if job is None:
                    raise RecipeError(f"unknown job: {job_id}")
            active = self._active_job_id
            snapshot = None if job is None else job.snapshot()
        return {
            "schema_version": STUDY_API_VERSION,
            "busy": active is not None,
            "active_job_id": active,
            "job": snapshot,
        }

    def cancel_job(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Request cancellation; it takes effect on the next check boundary."""

        job_id = request.get("job_id")
        with self._jobs_lock:
            if job_id is None:
                job_id = self._active_job_id
            if not isinstance(job_id, str) or job_id not in self._jobs:
                raise RecipeError("no such study job to cancel")
            job = self._jobs[job_id]
            job.cancel_event.set()
            if job.state == "running":
                job.message = "cancellation requested; stopping after the current check"
            snapshot = job.snapshot()
        return {"schema_version": STUDY_API_VERSION, "job": snapshot}

    def show_configuration(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Display one stored successful configuration in the shared viewer."""

        values = request.get("q")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise RecipeError("q must be an array of joint values")
        configuration = np.asarray([_finite(item, "q") for item in values], dtype=float)
        if configuration.size != int(self.ur_bundle.model.nq):
            raise RecipeError(
                f"q must contain {int(self.ur_bundle.model.nq)} values"
            )
        if self.viewer is None:
            return {
                "schema_version": STUDY_API_VERSION,
                "displayed": False,
                "message": "no viewer is attached to this backend",
            }
        with self.compute_lock:
            self._display(configuration)
        return {"schema_version": STUDY_API_VERSION, "displayed": True}

    # -- bounded scout ---------------------------------------------------

    def start_scout(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Start a bounded UR20 base comparison over the authored HARD checks.

        This is a small enumerated or sampled comparison of base candidates,
        not an optimizer.  It reports a study metric and never claims a cell
        layout decision.
        """

        recipe, context, allow_mismatch = self._recipe_from_request(request)
        settings = _scout_settings(request.get("scout"), context.key)
        if not recipe.enabled_hard_checks:
            raise RecipeError(
                "a bounded scout needs at least one enabled HARD check"
            )
        candidates = _scout_candidates(settings)
        if not candidates:
            raise RecipeError("scout bounds produced no candidate")
        options = self._options(recipe.profile)
        job = self._create_job(kind="scout", recipe=recipe, total=len(candidates))
        thread = self.thread_factory(
            target=self._run_scout_job,
            args=(job, recipe, candidates, options, settings, bool(allow_mismatch)),
            name=f"study-scout-{job.job_id}",
            daemon=True,
        )
        thread.start()
        return {
            "schema_version": STUDY_API_VERSION,
            "job": job.snapshot(),
            "accepted": True,
            "candidate_count": len(candidates),
            "settings": settings.payload(),
        }

    # ------------------------------------------------------------------
    # Internals: workflow / templates / targets
    # ------------------------------------------------------------------

    def _workflow_context(
        self,
        parameters: Mapping[str, object] | None,
        *,
        ur_base: BasePose | None = None,
    ) -> WorkflowContext:
        key = (
            self.initial_key
            if parameters is None
            else _case_key_from_parameters(self.spec, parameters)
        )
        if ur_base is not None:
            key = replace(key, ur_base=ur_base)
        state = SceneState(
            lift_height_m=key.lift_height_m,
            sku=key.sku,
            corner_samples=(key.corner,),
            tote_present=self.tote_present,
            tote_long_axis_offset_m=key.tote_offset_m,
            sr_j3_stroke_m=key.sr_j3_stroke_m,
            clearance_m=key.clearance_m,
        )
        snapshot = materialize_scene(
            self.spec, state, ur_base=key.ur_base, sr_base=key.sr_base
        )
        scenario = ProcessScenario(state, key.corner, key.coordination_mode)
        keepout = None
        if key.coordination_mode is CoordinationMode.SIMULTANEOUS:
            configure_sr_j3_stroke(self.sr_bundle, key.sr_j3_stroke_m)
            keepout = sr_concurrent_exclusion_box(
                self.spec,
                snapshot,
                self.sr_bundle,
                margin_m=self.keepout_margin_m,
            )
        steps = build_ur20_task_sequence(
            self.spec,
            snapshot,
            scenario,
            self.ur_bundle,
            sr_keepout=keepout,
            approach_distance_m=self.approach_distance_m,
        )
        return WorkflowContext(
            key=key, state=state, snapshot=snapshot, steps=tuple(steps)
        )

    def _templates(self, context: WorkflowContext) -> tuple[CheckTemplate, ...]:
        templates: list[CheckTemplate] = []
        for step in context.steps:
            for check in step.pose_checks:
                templates.append(
                    CheckTemplate(
                        reference=TemplateRef(
                            kind="pose_check",
                            step_index=step.index,
                            check_name=check.name,
                        ),
                        label=f"{step.index}. {step.name} / {check.name}",
                        step_name=step.name,
                        criticality=check.criticality.value,
                        payload_name=check.payload_name or "",
                        contact_object_name=check.contact_object_name or "",
                        object_names=tuple(
                            state.name for state in check.object_states
                        ),
                        allowed_contacts=tuple(
                            tuple(sorted(pair)) for pair in check.allowed_contacts
                        ),
                        start=_pose_to_6d(check.start_tcp.matrix),
                        end=_pose_to_6d(check.end_tcp.matrix),
                    )
                )
            if step.object_states:
                union: list[tuple[str, str]] = []
                for check in step.pose_checks:
                    for pair in check.allowed_contacts:
                        canonical = tuple(sorted(pair))
                        if canonical not in union:
                            union.append(canonical)
                attached = [
                    state.name
                    for state in step.object_states
                    if state.attached_to_tcp or state.follows_tcp
                ]
                templates.append(
                    CheckTemplate(
                        reference=TemplateRef(
                            kind="step_objects", step_index=step.index
                        ),
                        label=(
                            f"{step.index}. {step.name} / payload only "
                            "(no generated pose)"
                        ),
                        step_name=step.name,
                        criticality=step.criticality.value,
                        payload_name=attached[0] if len(attached) == 1 else "",
                        contact_object_name="",
                        object_names=tuple(
                            state.name for state in step.object_states
                        ),
                        allowed_contacts=tuple(union),
                        start=None,
                        end=None,
                    )
                )
        return tuple(templates)

    def _collision_targets(
        self, context: WorkflowContext
    ) -> tuple[CollisionTarget, ...]:
        snapshot = context.snapshot
        mesh_names = {mesh.name for mesh in snapshot.meshes}
        targets: list[CollisionTarget] = []
        for box in snapshot.boxes:
            if box.name in mesh_names:
                # The USD mesh with the same logical name is the real input.
                continue
            targets.append(
                CollisionTarget(
                    name=box.name,
                    role=box.role,
                    source=(
                        "process_object"
                        if box.role in PROCESS_OBJECT_ROLES
                        else "generated_box"
                    ),
                    geometry_kind="box",
                    detail=(
                        f"size {box.size_m[0]:.3f} x {box.size_m[1]:.3f} x "
                        f"{box.size_m[2]:.3f} m, yaw {box.yaw_deg:g} deg"
                        + ("" if box.collision_enabled else ", collision disabled")
                    ),
                )
            )
        for mesh in snapshot.meshes:
            targets.append(
                CollisionTarget(
                    name=mesh.name,
                    role=mesh.role,
                    source="usd_mesh",
                    geometry_kind="mesh",
                    source_prim_path=mesh.source_prim_path,
                    detail=(
                        f"{len(mesh.vertices_m)} vertices, "
                        f"{len(mesh.triangles)} triangles"
                        + ("" if mesh.collision_enabled else ", collision disabled")
                    ),
                )
            )
        known = {target.name for target in targets}
        for step in context.steps:
            for box in step.active_extra_obstacles:
                if box.name in known:
                    continue
                known.add(box.name)
                targets.append(
                    CollisionTarget(
                        name=box.name,
                        role=box.role,
                        source="generated_keepout",
                        geometry_kind="box",
                        detail=f"active in step {step.index} ({step.name})",
                    )
                )
            for state in itertools.chain(
                step.object_states,
                *(check.object_states for check in step.pose_checks),
            ):
                if state.name in known:
                    continue
                known.add(state.name)
                targets.append(
                    CollisionTarget(
                        name=state.name,
                        role="process_object",
                        geometry_kind="box",
                        source="phase_object",
                        detail="phase object supplied by the selected template",
                    )
                )
        targets.sort(key=lambda item: (item.source, item.role, item.name))
        return tuple(targets)

    def _reference_frames(self, context: WorkflowContext) -> list[dict[str, object]]:
        frames: list[dict[str, object]] = [
            {
                "name": WORLD_FRAME,
                "translation_m": [0.0, 0.0, 0.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        ]
        for name, frame in sorted(context.snapshot.frames.items()):
            frames.append(
                {
                    "name": name,
                    "translation_m": list(frame.translation_m),
                    "rotation_xyzw": list(frame.rotation_xyzw),
                }
            )
        return frames

    def _valid_contact_names(self, context: WorkflowContext) -> set[str]:
        names = {target.name for target in self._collision_targets(context)}
        names.add("floor")
        for geometry in self.ur_bundle.collision_model.geometryObjects:
            names.add(str(geometry.name))
            text = str(geometry.name)
            if text.endswith("_0"):
                names.add(text[:-2])
        # Logical aliases resolved by the collision adapter.
        names.update({"suction_pad", "suction_stem", "suction_tcp"})
        return names

    def _highlight_geometry(
        self,
        context: WorkflowContext,
        wanted: Sequence[str],
    ) -> tuple[tuple[BoxPrimitive, ...], tuple[MeshObstacle, ...], tuple[str, ...]]:
        requested = list(dict.fromkeys(str(item) for item in wanted))
        if len(requested) > 500:
            raise RecipeError("too many highlight targets")
        boxes = tuple(
            box for box in context.snapshot.boxes if box.name in set(requested)
        )
        meshes = tuple(
            mesh for mesh in context.snapshot.meshes if mesh.name in set(requested)
        )
        found = {box.name for box in boxes} | {mesh.name for mesh in meshes}
        missing = tuple(name for name in requested if name not in found)
        return boxes, meshes, missing

    # ------------------------------------------------------------------
    # Internals: resolution
    # ------------------------------------------------------------------

    def _recipe_from_request(
        self, request: Mapping[str, object]
    ) -> tuple[StudyRecipe, WorkflowContext, bool]:
        raw = request.get("recipe")
        if raw is None:
            raise RecipeError("recipe is required")
        recipe = (
            StudyRecipe.from_json(raw)
            if isinstance(raw, str)
            else StudyRecipe.from_mapping(raw, "recipe")
        )
        allow_mismatch = _boolean(
            request.get("allow_fingerprint_mismatch", False),
            "allow_fingerprint_mismatch",
        )
        if not self._fingerprint_match(recipe) and not allow_mismatch:
            raise RecipeError(
                "recipe scene_fingerprint does not match this session "
                f"({recipe.scene_fingerprint or 'absent'} != "
                f"{self.scene_fingerprint}); review it and resend with "
                "allow_fingerprint_mismatch=true to continue as a diagnostic"
            )
        parameters = request.get("parameters")
        if parameters is None:
            parameters = recipe.parameters or None
        if parameters is not None:
            parameters = _required_mapping(parameters, "parameters")
        context = self._workflow_context(parameters)
        return recipe, context, allow_mismatch

    def _fingerprint_match(self, recipe: StudyRecipe) -> bool:
        return recipe.scene_fingerprint == self.scene_fingerprint

    def _options(self, profile: str) -> EvaluationOptions:
        if profile not in self.profiles:
            raise RecipeError(
                f"unknown evaluation profile {profile!r}; available: "
                f"{sorted(self.profiles)}"
            )
        return self.profiles[profile]

    def _resolve_checks(
        self, recipe: StudyRecipe, context: WorkflowContext
    ) -> tuple[ResolvedCheck, ...]:
        valid_names = self._valid_contact_names(context)
        return tuple(
            self._resolve_check(recipe, context, check, valid_names)
            for check in recipe.checks
        )

    def _resolve_check(
        self,
        recipe: StudyRecipe,
        context: WorkflowContext,
        check: StudyCheck,
        valid_names: set[str],
    ) -> ResolvedCheck:
        scenario = recipe.scenario(check.scenario_id)
        for first, second in scenario.extra_allowed_contacts:
            for name in (first, second):
                if name not in valid_names:
                    raise RecipeError(
                        f"scenario {scenario.scenario_id} allows a contact with "
                        f"unknown logical/geometry name {name!r}"
                    )
        template_check: LinearPoseCheck | None = None
        template_step: TaskStepSpec | None = None
        if check.template is not None:
            template_step = context.step(check.template.step_index)
            if check.template.supplies_pose:
                template_check = context.template_pose_check(check.template)

        world_start, world_end = self._resolve_poses(
            context, check, template_check
        )
        object_states, allowed, payload_name, contact_name = self._resolve_payload(
            check, template_check, template_step, world_start
        )
        pose_check = LinearPoseCheck(
            name=check.check_id,
            start_tcp=RigidPose(world_start),
            end_tcp=RigidPose(world_end),
            criticality=(
                Criticality.HARD if check.criticality_is_hard else Criticality.SOFT
            ),
            object_states=object_states,
            payload_name=payload_name,
            contact_object_name=contact_name,
            allowed_contacts=allowed,
            include_in_metric_summary=check.include_in_metric_summary,
        )
        phase_step = template_step or _free_space_step(context.snapshot)
        phase = collision_phase_for_check(
            self.spec, context.snapshot, phase_step, pose_check
        )
        phase, applied, missing = _apply_scenario(phase, scenario)
        return ResolvedCheck(
            recipe_check=check,
            pose_check=pose_check,
            phase=phase,
            scenario=scenario,
            include_self_collision=scenario.include_self_collision,
            world_start=world_start,
            world_end=world_end,
            excluded_applied=applied,
            missing_excluded=missing,
        )

    def _resolve_poses(
        self,
        context: WorkflowContext,
        check: StudyCheck,
        template_check: LinearPoseCheck | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if check.pose_mode == "template_relative":
            if template_check is None:  # guarded by StudyCheck validation
                raise RecipeError(
                    f"check {check.check_id} has no pose template to offset"
                )
            start = _apply_offset(
                np.asarray(template_check.start_tcp.matrix, dtype=float),
                check.start_offset,
                check.offset_frame,
            )
            end = _apply_offset(
                np.asarray(template_check.end_tcp.matrix, dtype=float),
                check.end_offset,
                check.offset_frame,
            )
            return start, end
        base = self._frame_matrix(context, check.reference_frame)
        return base @ check.start_pose.matrix(), base @ check.end_pose.matrix()

    def _frame_matrix(self, context: WorkflowContext, name: str) -> np.ndarray:
        if name == WORLD_FRAME:
            return np.eye(4)
        frame = context.snapshot.frames.get(name)
        if frame is None:
            raise RecipeError(
                f"unknown reference frame {name!r}; available: "
                f"{[WORLD_FRAME, *sorted(context.snapshot.frames)]}"
            )
        return quaternion_matrix(frame.translation_m, frame.rotation_xyzw)

    @staticmethod
    def _resolve_payload(
        check: StudyCheck,
        template_check: LinearPoseCheck | None,
        template_step: TaskStepSpec | None,
        world_start: np.ndarray,
    ) -> tuple[
        tuple[ObjectState, ...], tuple[tuple[str, str], ...], str | None, str | None
    ]:
        if check.payload_source == "none":
            return (), (), None, None
        if template_check is not None:
            source_states = template_check.object_states
            allowed = tuple(template_check.allowed_contacts)
            payload_name = template_check.payload_name
            contact_name = template_check.contact_object_name
        elif template_step is not None:
            source_states = template_step.object_states
            seen: list[tuple[str, str]] = []
            for nested in template_step.pose_checks:
                for pair in nested.allowed_contacts:
                    canonical = tuple(sorted(pair))
                    if canonical not in seen:
                        seen.append(canonical)
            allowed = tuple(seen)
            attached = [
                state.name
                for state in source_states
                if state.attached_to_tcp or state.follows_tcp
            ]
            payload_name = attached[0] if len(attached) == 1 else None
            contact_name = None
        else:  # guarded by StudyCheck validation
            raise RecipeError(
                f"check {check.check_id} requests a template payload without a template"
            )

        states: list[ObjectState] = []
        for state in source_states:
            if state.attached_to_tcp or state.follows_tcp:
                if state.tcp_T_object is None:
                    raise RecipeError(
                        f"template payload {state.name} has no TCP transform"
                    )
                # The TCP-relative transform is reused by identity; only the
                # derived world pose follows the authored start pose.
                states.append(
                    ObjectState(
                        name=state.name,
                        size_m=state.size_m,
                        world_T_object=RigidPose(
                            world_start @ state.tcp_T_object.matrix
                        ),
                        attached_to_tcp=state.attached_to_tcp,
                        tcp_T_object=state.tcp_T_object,
                        follows_tcp=state.follows_tcp,
                    )
                )
            else:
                states.append(state)
        return tuple(states), allowed, payload_name, contact_name

    def _coverage(
        self, recipe: StudyRecipe, context: WorkflowContext
    ) -> dict[str, object]:
        """Compare the authored check set against the generated process set."""

        generated_hard: list[str] = []
        for step in context.steps:
            if step.criticality is not Criticality.HARD:
                continue
            for check in step.pose_checks:
                if check.criticality is Criticality.HARD:
                    generated_hard.append(f"{step.index}:{check.name}")
        authored = {
            f"{check.template.step_index}:{check.template.check_name}"
            for check in recipe.enabled_checks
            if check.template is not None and check.template.supplies_pose
        }
        unmodified = {
            f"{check.template.step_index}:{check.template.check_name}"
            for check in recipe.enabled_checks
            if check.template is not None
            and check.template.supplies_pose
            and check.is_unmodified_template
        }
        omitted = [name for name in generated_hard if name not in authored]
        modified = [
            name
            for name in generated_hard
            if name in authored and name not in unmodified
        ]
        authored_only = [
            check.check_id
            for check in recipe.enabled_checks
            if check.template is None
            or not check.template.supplies_pose
            or f"{check.template.step_index}:{check.template.check_name}"
            not in set(generated_hard)
        ]
        return {
            "generated_hard_check_count": len(generated_hard),
            "generated_hard_checks": generated_hard,
            "omitted_generated_hard_checks": omitted,
            "modified_generated_hard_checks": modified,
            "extra_authored_checks": authored_only,
            "covers_generated_hard_set": not omitted,
            # Steps 6 and the global transfer path are never generated, so a
            # complete authored set is still not a full process verification.
            "skipped_process_steps": [
                step.index
                for step in context.steps
                if step.criticality is Criticality.SKIP
            ],
        }

    def _diagnostic_report(
        self,
        recipe: StudyRecipe,
        resolved: Sequence[ResolvedCheck],
        coverage: Mapping[str, object],
    ) -> dict[str, object]:
        used = {
            item.scenario.scenario_id
            for item in resolved
            if item.recipe_check.enabled
        }
        reasons: list[str] = []
        excluded: set[str] = set()
        extra_contacts: set[tuple[str, str]] = set()
        for scenario in recipe.scenarios:
            if scenario.scenario_id not in used:
                continue
            excluded.update(scenario.excluded_targets)
            extra_contacts.update(scenario.extra_allowed_contacts)
            for reason in scenario.diagnostic_reasons():
                reasons.append(f"scenario {scenario.scenario_id}: {reason}")
        missing = sorted(
            {
                name
                for item in resolved
                if item.recipe_check.enabled
                for name in item.missing_excluded
            }
        )
        if missing:
            reasons.append(
                "excluded target(s) absent from the resolved phase input: "
                + ", ".join(missing)
            )
        omitted = list(coverage.get("omitted_generated_hard_checks") or ())
        if omitted:
            reasons.append(
                f"{len(omitted)} generated HARD check(s) omitted from this study"
            )
        modified = list(coverage.get("modified_generated_hard_checks") or ())
        if modified:
            reasons.append(
                f"{len(modified)} generated HARD check(s) replaced by edited poses"
            )
        return {
            "weakened_collision_contract": bool(
                excluded
                or extra_contacts
                or any(
                    not scenario.include_floor
                    or not scenario.include_self_collision
                    for scenario in recipe.scenarios
                    if scenario.scenario_id in used
                )
            ),
            "excluded_targets": sorted(excluded),
            "extra_allowed_contacts": [list(pair) for pair in sorted(extra_contacts)],
            "unresolved_excluded_targets": missing,
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    # Internals: jobs
    # ------------------------------------------------------------------

    def _create_job(
        self, *, kind: str, recipe: StudyRecipe, total: int
    ) -> StudyJob:
        with self._jobs_lock:
            if self._active_job_id is not None:
                active = self._jobs[self._active_job_id]
                raise RecipeError(
                    f"a study {active.kind} job is already running "
                    f"({active.completed}/{active.total}, current: "
                    f"{active.current_label or 'starting'}); cancel it first"
                )
            job = StudyJob(
                job_id=uuid.uuid4().hex,
                kind=kind,
                recipe_hash=recipe.recipe_hash,
                recipe_id=recipe.recipe_id,
                total=int(total),
                started_unix=time(),
            )
            self._jobs[job.job_id] = job
            self._job_order.append(job.job_id)
            while len(self._job_order) > MAX_JOB_HISTORY:
                self._jobs.pop(self._job_order.pop(0), None)
            self._active_job_id = job.job_id
            return job

    def _finish_job(self, job: StudyJob) -> None:
        with self._jobs_lock:
            if self._active_job_id == job.job_id:
                self._active_job_id = None

    def _run_evaluation_job(
        self,
        job: StudyJob,
        recipe: StudyRecipe,
        context: WorkflowContext,
        queue: Sequence[ResolvedCheck],
        options: EvaluationOptions,
        fingerprint_override: bool,
        selected: Sequence[str] | None,
    ) -> None:
        started = self.clock()
        entries: list[dict[str, object]] = []
        cancelled = False
        try:
            previous_arm: np.ndarray | None = None
            nominal = tuple(
                float(value) for value in self.spec.robots["ur20"].nominal_q
            )
            for item in queue:
                if job.cancel_event.is_set():
                    cancelled = True
                    break
                with self._jobs_lock:
                    job.current_label = item.recipe_check.label
                seeds: list[Sequence[float]] = []
                if previous_arm is not None:
                    seeds.append(previous_arm)
                seeds.append(nominal)
                seeds.extend(options.extra_arm_seeds)
                with self.compute_lock:
                    checker = PinocchioCollisionChecker(
                        self.ur_bundle,
                        item.phase,
                        include_self_collision=item.include_self_collision,
                    )
                    result, last_arm = evaluate_linear_check(
                        context.snapshot,
                        self.ur_bundle,
                        item.pose_check,
                        checker,
                        seeds,
                        options,
                    )
                    pair_count = checker.collision_pair_count
                if last_arm is not None:
                    previous_arm = last_arm
                entries.append(
                    _check_result_payload(item, result, pair_count)
                )
                with self._jobs_lock:
                    job.completed += 1
            coverage = self._coverage(recipe, context)
            diagnostic = self._diagnostic_report(recipe, queue, coverage)
            result_payload = _study_result_payload(
                recipe=recipe,
                context=context,
                entries=entries,
                coverage=coverage,
                diagnostic=diagnostic,
                evaluated_ids=[item.recipe_check.check_id for item in queue],
                selected_ids=None if selected is None else list(selected),
                cancelled=cancelled,
                fingerprint_override=fingerprint_override,
                elapsed_seconds=max(0.0, self.clock() - started),
                options=options,
            )
            with self._jobs_lock:
                job.result = result_payload
                job.state = "cancelled" if cancelled else "completed"
                job.current_label = ""
                job.message = (
                    "stopped at a check boundary" if cancelled else "finished"
                )
        except BaseException as exc:  # noqa: BLE001 - reported to the browser
            with self._jobs_lock:
                job.state = "failed"
                job.message = f"{type(exc).__name__}: {exc}"
                job.result = None
        finally:
            self._finish_job(job)

    def _run_scout_job(
        self,
        job: StudyJob,
        recipe: StudyRecipe,
        candidates: Sequence[BasePose],
        options: EvaluationOptions,
        settings: "ScoutSettings",
        fingerprint_override: bool,
    ) -> None:
        started = self.clock()
        rows: list[dict[str, object]] = []
        cancelled = False
        try:
            for base in candidates:
                if job.cancel_event.is_set():
                    cancelled = True
                    break
                label = (
                    f"UR20 base x={base.x_m:.3f} y={base.y_m:.3f} "
                    f"z={base.z_m:.3f} yaw={base.yaw_deg:.1f}"
                )
                with self._jobs_lock:
                    job.current_label = label
                rows.append(
                    self._scout_candidate(
                        recipe, base, options, job.cancel_event
                    )
                )
                with self._jobs_lock:
                    job.completed += 1
            feasible = [row for row in rows if row["status"] == "feasible"]
            feasible.sort(
                key=lambda row: -float(row["worst_normalized_min_singular_value"])
            )
            with self._jobs_lock:
                job.result = {
                    "schema_version": STUDY_API_VERSION,
                    "kind": "scout",
                    "recipe_hash": recipe.recipe_hash,
                    "recipe_id": recipe.recipe_id,
                    "settings": settings.payload(),
                    "candidates": rows,
                    "ranking": feasible,
                    "cancelled": cancelled,
                    "complete": not cancelled,
                    "fingerprint_override": fingerprint_override,
                    "elapsed_seconds": max(0.0, self.clock() - started),
                    "metric": "worst sampled normalized minimum singular value",
                    "interpretation": (
                        "study comparison of enabled HARD checks only; this is "
                        "not a cell layout optimization result"
                    ),
                }
                job.state = "cancelled" if cancelled else "completed"
                job.current_label = ""
                job.message = (
                    "stopped at a candidate boundary" if cancelled else "finished"
                )
        except BaseException as exc:  # noqa: BLE001 - reported to the browser
            with self._jobs_lock:
                job.state = "failed"
                job.message = f"{type(exc).__name__}: {exc}"
                job.result = None
        finally:
            self._finish_job(job)

    def _scout_candidate(
        self,
        recipe: StudyRecipe,
        base: BasePose,
        options: EvaluationOptions,
        cancel_event: threading.Event,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "ur_base": {
                "x_m": base.x_m,
                "y_m": base.y_m,
                "z_m": base.z_m,
                "yaw_deg": base.yaw_deg,
            },
            "status": "unknown",
            "evaluated_checks": 0,
            "hard_check_count": 0,
            "failed_check": None,
            "worst_normalized_min_singular_value": 0.0,
            "message": "",
        }
        try:
            # Regenerating the workflow runs Pinocchio forward kinematics, so
            # it belongs inside the shared compute lock just like the checks.
            with self.compute_lock:
                context = self._workflow_context(
                    recipe.parameters or None, ur_base=base
                )
                resolved = [
                    item
                    for item in self._resolve_checks(recipe, context)
                    if item.recipe_check.enabled
                    and item.recipe_check.criticality_is_hard
                ]
        except (RecipeError, ValueError, KeyError, RuntimeError) as exc:
            row["status"] = "rejected"
            row["message"] = str(exc)
            return row
        row["hard_check_count"] = len(resolved)
        worst = math.inf
        previous_arm: np.ndarray | None = None
        nominal = tuple(float(value) for value in self.spec.robots["ur20"].nominal_q)
        for item in resolved:
            if cancel_event.is_set():
                row["status"] = "cancelled"
                return row
            seeds: list[Sequence[float]] = []
            if previous_arm is not None:
                seeds.append(previous_arm)
            seeds.append(nominal)
            seeds.extend(options.extra_arm_seeds)
            with self.compute_lock:
                checker = PinocchioCollisionChecker(
                    self.ur_bundle,
                    item.phase,
                    include_self_collision=item.include_self_collision,
                )
                result, last_arm = evaluate_linear_check(
                    context.snapshot,
                    self.ur_bundle,
                    item.pose_check,
                    checker,
                    seeds,
                    options,
                )
            if last_arm is not None:
                previous_arm = last_arm
            row["evaluated_checks"] = int(row["evaluated_checks"]) + 1
            if result.status != "success":
                row["status"] = "infeasible"
                row["failed_check"] = {
                    "check_id": item.recipe_check.check_id,
                    "status": result.status,
                }
                # A partially evaluated candidate must never contribute a
                # metric-based ranking position.
                row["worst_normalized_min_singular_value"] = 0.0
                return row
            if item.recipe_check.include_in_metric_summary:
                for sample in result.samples:
                    if sample.manipulability is None:
                        continue
                    worst = min(
                        worst,
                        float(
                            sample.manipulability.normalized_min_singular_value
                        ),
                    )
        if not resolved:
            row["status"] = "rejected"
            row["message"] = "no enabled HARD check resolved for this candidate"
            return row
        row["status"] = "feasible"
        row["worst_normalized_min_singular_value"] = (
            0.0 if not math.isfinite(worst) else worst
        )
        row["metric_available"] = math.isfinite(worst)
        return row

    def _display(self, configuration: np.ndarray) -> None:
        if self._playback is None:
            from .meshcat_playback import UR20MeshcatPlayback

            loaded = self.viewer.loaded_robot("ur20")
            self._playback = UR20MeshcatPlayback(
                self.viewer.viewer,
                loaded.visualizer,
                loaded.bundle,
                on_configuration_displayed=self.viewer.update_ur20_suction_proxy,
            )
        self._playback.display_configuration(configuration)
        self._playback.hide_ellipsoid()


# --------------------------------------------------------------------------
# Scenario application
# --------------------------------------------------------------------------


def _apply_scenario(
    phase: CollisionPhase,
    scenario: CollisionScenario,
) -> tuple[CollisionPhase, tuple[str, ...], tuple[str, ...]]:
    """Return a new phase with the scenario's exclusions and exceptions.

    Geometry objects are reused by identity so the immutable USD BVH cache
    keeps working; only the containing tuples are rebuilt.
    """

    excluded = set(scenario.excluded_targets)
    kept_obstacles = tuple(box for box in phase.obstacles if box.name not in excluded)
    kept_meshes = tuple(mesh for mesh in phase.meshes if mesh.name not in excluded)
    kept_attached = tuple(
        item for item in phase.attached_boxes if item.name not in excluded
    )
    present = (
        {box.name for box in phase.obstacles}
        | {mesh.name for mesh in phase.meshes}
        | {item.name for item in phase.attached_boxes}
    )
    applied = tuple(sorted(excluded & present))
    missing = tuple(sorted(excluded - present))
    allowed = frozenset(phase.allowed_contacts) | frozenset(
        tuple(sorted(pair)) for pair in scenario.extra_allowed_contacts
    )
    updated = replace(
        phase,
        obstacles=kept_obstacles,
        meshes=kept_meshes,
        attached_boxes=kept_attached,
        allowed_contacts=allowed,
        floor_z_m=phase.floor_z_m if scenario.include_floor else None,
    )
    return updated, applied, missing


def _free_space_step(snapshot: SceneSnapshot) -> TaskStepSpec:
    """Return the phase context for a check with no generated template.

    The suppressed set matches every generated step: process boxes and totes
    are phase objects, not static environment, so a free-space check must not
    see them at their nominal scene positions.
    """

    suppressed = tuple(
        dict.fromkeys(
            (
                *(
                    box.name
                    for box in snapshot.boxes
                    if box.role in PROCESS_OBJECT_ROLES
                ),
                "representative_tote",
            )
        )
    )
    return TaskStepSpec(
        index=0,
        name="study_free_space",
        criticality=Criticality.SOFT,
        pose_checks=(),
        object_states=(),
        suppress_snapshot_objects=suppressed,
    )


# --------------------------------------------------------------------------
# Pose helpers
# --------------------------------------------------------------------------


def _apply_offset(
    template: np.ndarray, offset: Pose6D, offset_frame: str
) -> np.ndarray:
    matrix = np.asarray(template, dtype=float).reshape(4, 4)
    delta = offset.matrix()
    result = matrix @ delta if offset_frame == "tcp_local" else delta @ matrix
    return _orthonormalized(result)


def _orthonormalized(matrix: np.ndarray) -> np.ndarray:
    """Remove accumulated float drift so ``RigidPose`` accepts the product."""

    result = np.array(matrix, dtype=float).reshape(4, 4)
    rotation = result[:3, :3]
    u, _, vh = np.linalg.svd(rotation)
    corrected = u @ vh
    if float(np.linalg.det(corrected)) < 0.0:
        raise RecipeError("authored pose rotation is not right-handed")
    result[:3, :3] = corrected
    result[3, :] = (0.0, 0.0, 0.0, 1.0)
    if not np.all(np.isfinite(result)):
        raise RecipeError("authored pose is not finite")
    return result


def _pose_to_6d(matrix: np.ndarray) -> Pose6D:
    """Return the ``Rz Ry Rx`` decomposition of one homogeneous transform."""

    value = np.asarray(matrix, dtype=float).reshape(4, 4)
    rotation = value[:3, :3]
    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    if abs(math.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    else:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return Pose6D(
        x_m=float(value[0, 3]),
        y_m=float(value[1, 3]),
        z_m=float(value[2, 3]),
        roll_deg=math.degrees(roll),
        pitch_deg=math.degrees(pitch),
        yaw_deg=math.degrees(yaw),
    )


def relative_pose_6d(reference: np.ndarray, pose: np.ndarray) -> Pose6D:
    """Return ``pose`` expressed in ``reference`` as a 6-DOF value."""

    return _pose_to_6d(inverse_transform(reference) @ np.asarray(pose, dtype=float))


# --------------------------------------------------------------------------
# Result payloads
# --------------------------------------------------------------------------


def _check_result_payload(
    item: ResolvedCheck,
    result: LinearCheckResult,
    collision_pair_count: int,
) -> dict[str, object]:
    samples = []
    for index, sample in enumerate(result.samples):
        metrics = sample.manipulability
        collision = sample.collision
        samples.append(
            {
                "index": index,
                "fraction": float(sample.fraction),
                "target": _pose_to_6d(sample.target.matrix).payload(),
                "ik_status": sample.ik.status,
                "ik_position_error_m": float(sample.ik.position_error_m),
                "ik_orientation_error_rad": float(sample.ik.orientation_error_rad),
                "ik_iterations": int(sample.ik.iterations),
                "ik_seed_index": int(sample.ik.seed_index),
                "ik_rejected_by_acceptor": int(sample.ik.rejected_by_acceptor),
                "q": (
                    [float(value) for value in sample.ik.q]
                    if sample.ik.success
                    else None
                ),
                "minimum_distance_m": (
                    None
                    if collision is None
                    else _finite_or_none(collision.minimum_distance_m)
                ),
                "nearest_pair": (
                    None
                    if collision is None or collision.nearest_pair is None
                    else list(collision.nearest_pair)
                ),
                "contacts": (
                    []
                    if collision is None
                    else [
                        [contact.first, contact.second]
                        for contact in collision.contacts
                    ]
                ),
                "normalized_min_singular_value": (
                    None
                    if metrics is None
                    else float(metrics.normalized_min_singular_value)
                ),
                "translational_volume": (
                    None if metrics is None else float(metrics.translational_volume)
                ),
                "normalized_condition_number": (
                    None
                    if metrics is None
                    else _finite_or_none(metrics.normalized_condition_number)
                ),
            }
        )
    joint_collision = result.joint_collision
    return {
        "check_id": item.recipe_check.check_id,
        "label": item.recipe_check.label,
        "criticality": item.recipe_check.criticality,
        "scenario_id": item.scenario.scenario_id,
        "status": result.status,
        "success": bool(result.success),
        "failed_sample_index": result.failed_sample_index,
        "collision_pair_count": int(collision_pair_count),
        "pose_mode": item.recipe_check.pose_mode,
        "template": (
            None
            if item.recipe_check.template is None
            else item.recipe_check.template.token
        ),
        "payload_source": item.recipe_check.payload_source,
        "payload_objects": [
            {
                "name": state.name,
                "attached_to_tcp": state.attached_to_tcp,
                "follows_tcp": state.follows_tcp,
            }
            for state in item.pose_check.object_states
        ],
        "allowed_contacts": [
            list(pair) for pair in sorted(item.pose_check.allowed_contacts)
        ],
        "phase_allowed_contacts": [
            list(pair) for pair in sorted(item.phase.allowed_contacts)
        ],
        "excluded_targets_applied": list(item.excluded_applied),
        "excluded_targets_missing": list(item.missing_excluded),
        "include_floor": item.scenario.include_floor,
        "include_self_collision": item.scenario.include_self_collision,
        "world_start": _pose_to_6d(item.world_start).payload(),
        "world_end": _pose_to_6d(item.world_end).payload(),
        "joint_segment_contacts": (
            []
            if joint_collision is None
            else [
                [contact.first, contact.second]
                for contact in joint_collision.contacts
            ]
        ),
        "samples": samples,
    }


def _study_result_payload(
    *,
    recipe: StudyRecipe,
    context: WorkflowContext,
    entries: Sequence[Mapping[str, object]],
    coverage: Mapping[str, object],
    diagnostic: Mapping[str, object],
    evaluated_ids: Sequence[str],
    selected_ids: Sequence[str] | None,
    cancelled: bool,
    fingerprint_override: bool,
    elapsed_seconds: float,
    options: EvaluationOptions,
) -> dict[str, object]:
    by_id = {str(entry["check_id"]): entry for entry in entries}
    hard_ids = [
        check.check_id
        for check in recipe.enabled_checks
        if check.criticality_is_hard
    ]
    evaluated_hard = [name for name in hard_ids if name in by_id]
    unevaluated_hard = [name for name in hard_ids if name not in by_id]
    hard_failures = [
        name for name in evaluated_hard if not by_id[name]["success"]
    ]
    soft_ids = [
        check.check_id
        for check in recipe.enabled_checks
        if not check.criticality_is_hard
    ]
    soft_failures = [
        name for name in soft_ids if name in by_id and not by_id[name]["success"]
    ]

    weakened = bool(diagnostic.get("weakened_collision_contract"))
    if not hard_ids:
        status = "no_hard_checks"
        certification = "none"
        headline = (
            "No enabled HARD check. An empty HARD set is never a success."
        )
    elif unevaluated_hard or cancelled:
        status = "unknown"
        certification = "none"
        headline = (
            "Incomplete: "
            f"{len(unevaluated_hard)} enabled HARD check(s) were not calculated"
            + (" (run cancelled)" if cancelled else "")
            + ". An uncalculated check is not a success."
        )
    elif hard_failures:
        status = "study_infeasible"
        certification = "none"
        headline = (
            f"{len(hard_failures)} of {len(evaluated_hard)} HARD check(s) failed."
        )
    elif weakened or coverage.get("omitted_generated_hard_checks"):
        status = "study_feasible_diagnostic"
        certification = "study_feasible_diagnostic"
        headline = (
            "All evaluated HARD checks passed, but the collision contract or "
            "the check set was weakened. This is a diagnostic result, not a "
            "feasibility claim."
        )
    else:
        status = "study_feasible"
        certification = "study_feasible"
        headline = (
            "All enabled HARD checks passed with the default collision "
            "contract. study_feasible only: it does not cover the skipped "
            "pouring step, transfer paths between checks, every SKU/corner, "
            "or mass/suction/dynamics."
        )
    return {
        "schema_version": STUDY_API_VERSION,
        "kind": "evaluate",
        "recipe_hash": recipe.recipe_hash,
        "recipe_id": recipe.recipe_id,
        "parameters": _parameter_payload(context.key),
        "profile": recipe.profile,
        "profile_settings": {
            "cartesian_translation_step_m": options.cartesian_translation_step_m,
            "cartesian_rotation_step_deg": math.degrees(
                options.cartesian_rotation_step_rad
            ),
            "joint_interpolation_samples": options.joint_interpolation_samples,
            "characteristic_length_m": options.characteristic_length_m,
            "ik_max_iterations": options.ik.max_iterations,
        },
        "status": status,
        "certification": certification,
        "headline": headline,
        "cancelled": bool(cancelled),
        "complete": not cancelled and not unevaluated_hard,
        "fingerprint_override": bool(fingerprint_override),
        "elapsed_seconds": float(elapsed_seconds),
        "enabled_check_count": len(recipe.enabled_checks),
        "evaluated_check_ids": list(evaluated_ids),
        "selected_check_ids": None if selected_ids is None else list(selected_ids),
        "hard_check_ids": hard_ids,
        "unevaluated_hard_check_ids": unevaluated_hard,
        "failed_hard_check_ids": hard_failures,
        "failed_soft_check_ids": soft_failures,
        "coverage": dict(coverage),
        "diagnostic": dict(diagnostic),
        "checks": [dict(entry) for entry in entries],
    }


def _resolved_check_payload(item: ResolvedCheck) -> dict[str, object]:
    return {
        "check_id": item.recipe_check.check_id,
        "label": item.recipe_check.label,
        "enabled": item.recipe_check.enabled,
        "criticality": item.recipe_check.criticality,
        "scenario_id": item.scenario.scenario_id,
        "pose_mode": item.recipe_check.pose_mode,
        "offset_frame": item.recipe_check.offset_frame,
        "reference_frame": item.recipe_check.reference_frame,
        "world_start": _pose_to_6d(item.world_start).payload(),
        "world_end": _pose_to_6d(item.world_end).payload(),
        "segment_length_m": float(
            np.linalg.norm(item.world_end[:3, 3] - item.world_start[:3, 3])
        ),
        "payload_source": item.recipe_check.payload_source,
        "payload_objects": [
            {
                "name": state.name,
                "attached_to_tcp": state.attached_to_tcp,
                "follows_tcp": state.follows_tcp,
                "tcp_relative": (
                    None
                    if state.tcp_T_object is None
                    else _pose_to_6d(state.tcp_T_object.matrix).payload()
                ),
            }
            for state in item.pose_check.object_states
        ],
        "allowed_contacts": [
            list(pair) for pair in sorted(item.pose_check.allowed_contacts)
        ],
        # The template's intended contacts plus the scenario's extra pairs:
        # this is what the collision checker actually exempts.
        "phase_allowed_contacts": [
            list(pair) for pair in sorted(item.phase.allowed_contacts)
        ],
        "obstacle_count": len(item.phase.obstacles),
        "mesh_count": len(item.phase.meshes),
        "attached_count": len(item.phase.attached_boxes),
        "excluded_targets_applied": list(item.excluded_applied),
        "excluded_targets_missing": list(item.missing_excluded),
        "include_floor": item.scenario.include_floor,
        "include_self_collision": item.scenario.include_self_collision,
        "is_unmodified_template": item.recipe_check.is_unmodified_template,
    }


# --------------------------------------------------------------------------
# Bounded scout settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoutSettings:
    mode: str
    base: BasePose
    x_range_m: tuple[float, float]
    y_range_m: tuple[float, float]
    z_range_m: tuple[float, float]
    yaw_range_deg: tuple[float, float]
    counts: tuple[int, int, int, int]
    max_candidates: int
    seed: int

    def payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "center_base": {
                "x_m": self.base.x_m,
                "y_m": self.base.y_m,
                "z_m": self.base.z_m,
                "yaw_deg": self.base.yaw_deg,
            },
            "x_range_m": list(self.x_range_m),
            "y_range_m": list(self.y_range_m),
            "z_range_m": list(self.z_range_m),
            "yaw_range_deg": list(self.yaw_range_deg),
            "counts": list(self.counts),
            "max_candidates": self.max_candidates,
            "seed": self.seed,
        }


def _scout_settings(raw: object, key: CaseKey) -> ScoutSettings:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise RecipeError("scout must be an object")
    allowed = {
        "mode",
        "x_range_m",
        "y_range_m",
        "z_range_m",
        "yaw_range_deg",
        "counts",
        "max_candidates",
        "seed",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise RecipeError(f"scout has unknown keys: {sorted(unknown)}")
    mode = str(raw.get("mode", "grid"))
    if mode not in {"grid", "random"}:
        raise RecipeError("scout mode must be 'grid' or 'random'")
    base = key.ur_base
    ranges = {}
    for name, default in (
        ("x_range_m", (base.x_m, base.x_m)),
        ("y_range_m", (base.y_m, base.y_m)),
        ("z_range_m", (base.z_m, base.z_m)),
        ("yaw_range_deg", (base.yaw_deg, base.yaw_deg)),
    ):
        value = raw.get(name, default)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise RecipeError(f"scout {name} must be a [min, max] pair")
        if len(value) != 2:
            raise RecipeError(f"scout {name} must contain exactly two values")
        low = _finite(value[0], f"scout {name}[0]")
        high = _finite(value[1], f"scout {name}[1]")
        if low > high:
            raise RecipeError(f"scout {name} min must not exceed max")
        ranges[name] = (low, high)
    if ranges["z_range_m"][0] <= 0.0:
        raise RecipeError("scout z range must stay above the floor")
    if abs(ranges["yaw_range_deg"][1] - ranges["yaw_range_deg"][0]) > 720.0:
        raise RecipeError("scout yaw range must span at most 720 degrees")

    counts_raw = raw.get("counts", (1, 1, 1, 1))
    if not isinstance(counts_raw, Sequence) or isinstance(counts_raw, (str, bytes)):
        raise RecipeError("scout counts must be an array of four integers")
    if len(counts_raw) != 4:
        raise RecipeError("scout counts must contain exactly four values")
    counts: list[int] = []
    for index, value in enumerate(counts_raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise RecipeError(f"scout counts[{index}] must be an integer")
        if not 1 <= value <= MAX_SCOUT_CANDIDATES:
            raise RecipeError(
                f"scout counts[{index}] must be in [1, {MAX_SCOUT_CANDIDATES}]"
            )
        counts.append(int(value))
    maximum = raw.get("max_candidates", MAX_SCOUT_CANDIDATES)
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise RecipeError("scout max_candidates must be an integer")
    if not 1 <= maximum <= MAX_SCOUT_CANDIDATES:
        raise RecipeError(
            f"scout max_candidates must be in [1, {MAX_SCOUT_CANDIDATES}]"
        )
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise RecipeError("scout seed must be an integer in [0, 2^31)")
    product = counts[0] * counts[1] * counts[2] * counts[3]
    if mode == "grid" and product > maximum:
        raise RecipeError(
            f"scout grid would produce {product} candidates, above the "
            f"max_candidates bound of {maximum}"
        )
    return ScoutSettings(
        mode=mode,
        base=base,
        x_range_m=ranges["x_range_m"],
        y_range_m=ranges["y_range_m"],
        z_range_m=ranges["z_range_m"],
        yaw_range_deg=ranges["yaw_range_deg"],
        counts=(counts[0], counts[1], counts[2], counts[3]),
        max_candidates=maximum,
        seed=seed,
    )


def _scout_candidates(settings: ScoutSettings) -> tuple[BasePose, ...]:
    if settings.mode == "grid":
        axes = [
            _axis_values(settings.x_range_m, settings.counts[0]),
            _axis_values(settings.y_range_m, settings.counts[1]),
            _axis_values(settings.z_range_m, settings.counts[2]),
            _axis_values(settings.yaw_range_deg, settings.counts[3]),
        ]
        poses = [
            BasePose.from_sequence((x, y, z, yaw), "scout candidate")
            for x, y, z, yaw in itertools.product(*axes)
        ]
    else:
        generator = random.Random(settings.seed)
        count = min(settings.max_candidates, max(settings.counts))
        poses = [
            BasePose.from_sequence(
                (
                    generator.uniform(*settings.x_range_m),
                    generator.uniform(*settings.y_range_m),
                    generator.uniform(*settings.z_range_m),
                    generator.uniform(*settings.yaw_range_deg),
                ),
                "scout candidate",
            )
            for _ in range(count)
        ]
    return tuple(poses[: settings.max_candidates])


def _axis_values(bounds: tuple[float, float], count: int) -> tuple[float, ...]:
    low, high = bounds
    if count <= 1 or high <= low:
        return (low,)
    step = (high - low) / (count - 1)
    return tuple(low + step * index for index in range(count))


# --------------------------------------------------------------------------
# Parameter helpers (shared contract with the live backend)
# --------------------------------------------------------------------------


def _parameter_payload(key: CaseKey) -> dict[str, object]:
    return {
        "ur_x": key.ur_base.x_m,
        "ur_y": key.ur_base.y_m,
        "ur_z": key.ur_base.z_m,
        "ur_yaw": key.ur_base.yaw_deg,
        "sr_x": key.sr_base.x_m,
        "sr_y": key.sr_base.y_m,
        "sr_z": key.sr_base.z_m,
        "sr_yaw": key.sr_base.yaw_deg,
        "sku": key.sku,
        "lift": key.lift_height_m,
        "tote": key.tote_offset_m,
        "stroke": key.sr_j3_stroke_m,
        "clearance": key.clearance_m,
        "corner": key.corner,
        "mode": key.coordination_mode.value,
    }


def _case_key_from_parameters(
    spec: SceneSpec, parameters: Mapping[str, object]
) -> CaseKey:
    required = {
        "ur_x",
        "ur_y",
        "ur_z",
        "ur_yaw",
        "sr_x",
        "sr_y",
        "sr_z",
        "sr_yaw",
        "sku",
        "lift",
        "tote",
        "stroke",
        "clearance",
        "corner",
        "mode",
    }
    missing = sorted(required - set(parameters))
    unknown = sorted(set(parameters) - required)
    if missing:
        raise RecipeError(f"missing study parameters: {missing}")
    if unknown:
        raise RecipeError(f"unknown study parameters: {unknown}")
    ur_base = BasePose.from_sequence(
        tuple(
            _finite(parameters[name], name)
            for name in ("ur_x", "ur_y", "ur_z", "ur_yaw")
        ),
        "ur_base",
    )
    sr_base = BasePose.from_sequence(
        tuple(
            _finite(parameters[name], name)
            for name in ("sr_x", "sr_y", "sr_z", "sr_yaw")
        ),
        "sr_base",
    )
    if ur_base.z_m <= 0.0 or sr_base.z_m <= 0.0:
        raise RecipeError("robot base z must be positive")
    sku = parameters["sku"]
    corner = parameters["corner"]
    mode = parameters["mode"]
    if not isinstance(sku, str) or sku not in spec.box_skus:
        raise RecipeError(f"unknown box SKU: {sku!r}")
    if not isinstance(corner, str) or corner not in CORNER_SIGNS:
        raise RecipeError(f"unknown pallet corner: {corner!r}")
    try:
        coordination_mode = CoordinationMode(str(mode))
    except ValueError as exc:
        raise RecipeError(f"unknown coordination mode: {mode!r}") from exc
    clearance_raw = parameters["clearance"]
    clearance = (
        None if clearance_raw is None else _nonnegative(clearance_raw, "clearance")
    )
    return CaseKey(
        ur_base=ur_base,
        sr_base=sr_base,
        lift_height_m=_finite(parameters["lift"], "lift"),
        tote_offset_m=_finite(parameters["tote"], "tote"),
        corner=corner,
        coordination_mode=coordination_mode,
        sku=sku,
        sr_j3_stroke_m=_positive(parameters["stroke"], "stroke"),
        clearance_m=clearance,
    )


# --------------------------------------------------------------------------
# Small validators
# --------------------------------------------------------------------------


def _unique_identifier(candidate: str, used: set[str]) -> str:
    text = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(candidate).lower()
    ).strip("-_")
    if not text or not text[0].isalnum():
        text = f"c{text}"
    text = text[:60] or "check"
    result = text
    suffix = 2
    while result in used:
        result = f"{text[:56]}-{suffix}"
        suffix += 1
    used.add(result)
    return result


def _required_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RecipeError(f"{name} must be an object")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RecipeError(f"{name} must be finite")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise RecipeError(f"{name} must be positive")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise RecipeError(f"{name} must be non-negative")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise RecipeError(f"{name} must be a boolean")
    return value


def _optional_string_list(value: object, name: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RecipeError(f"{name} must be an array of strings")
    if len(value) > 500:
        raise RecipeError(f"{name} is too long")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise RecipeError(f"{name} entries must be non-empty strings")
        if item not in result:
            result.append(item)
    return result


def _finite_or_none(value: float) -> float | None:
    """Keep strict JSON output: infinity never reaches the browser."""

    result = float(value)
    return result if math.isfinite(result) else None
