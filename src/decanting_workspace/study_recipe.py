"""Validated, serializable recipe for a user-authored UR20 check study.

A recipe is the complete reproducible input of one study run: the live
parameter set, user-authored TCP pose checks, and reusable collision
scenarios.  It is deliberately independent from Pinocchio: this module only
validates and transports values.  :mod:`decanting_workspace.study_backend`
turns a recipe into actual ``LinearPoseCheck`` objects and evaluates them.

Units follow the stored/computed convention of the rest of the project:
lengths are metres and angles are radians-free degrees only where a field
name says ``_deg``.  The browser converts millimetres for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


STUDY_SCHEMA_VERSION = 1
"""Bump whenever the stored recipe contract changes."""

MAX_RECIPE_BYTES = 1024 * 1024
MAX_CHECKS = 200
MAX_SCENARIOS = 40
MAX_ALLOWED_CONTACTS = 200
MAX_EXCLUDED_TARGETS = 500

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_OFFSET_FRAMES = ("tcp_local", "world")
_POSE_MODES = ("template_relative", "absolute")
_PAYLOAD_SOURCES = ("template", "none")
_CRITICALITIES = ("hard", "soft")
_TEMPLATE_KINDS = ("pose_check", "step_objects")
WORLD_FRAME = "world"


class RecipeError(ValueError):
    """Raised for any invalid recipe input; never escapes as a 500."""


@dataclass(frozen=True)
class Pose6D:
    """XYZ metres plus intrinsic ``Rz(yaw) Ry(pitch) Rx(roll)`` degrees."""

    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecipeError(f"pose {name} must be numeric")
            if not math.isfinite(float(value)):
                raise RecipeError(f"pose {name} must be finite")
            object.__setattr__(self, name, float(value))
        if max(abs(self.x_m), abs(self.y_m), abs(self.z_m)) > 1000.0:
            raise RecipeError("pose translation must stay within +/-1000 m")
        for name in ("roll_deg", "pitch_deg", "yaw_deg"):
            if abs(getattr(self, name)) > 36000.0:
                raise RecipeError(f"pose {name} must stay within +/-36000 deg")

    @property
    def is_identity(self) -> bool:
        return all(
            value == 0.0
            for value in (
                self.x_m,
                self.y_m,
                self.z_m,
                self.roll_deg,
                self.pitch_deg,
                self.yaw_deg,
            )
        )

    def matrix(self) -> np.ndarray:
        """Return the homogeneous transform for this pose."""

        result = np.eye(4)
        result[:3, :3] = rotation_matrix_zyx(
            self.roll_deg, self.pitch_deg, self.yaw_deg
        )
        result[:3, 3] = (self.x_m, self.y_m, self.z_m)
        return result

    def payload(self) -> dict[str, float]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
        }

    @classmethod
    def from_mapping(cls, raw: Any, name: str) -> "Pose6D":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise RecipeError(f"{name} must be an object")
        unknown = set(raw) - {
            "x_m",
            "y_m",
            "z_m",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        }
        if unknown:
            raise RecipeError(f"{name} has unknown keys: {sorted(unknown)}")
        try:
            return cls(
                x_m=_number(raw.get("x_m", 0.0), f"{name}.x_m"),
                y_m=_number(raw.get("y_m", 0.0), f"{name}.y_m"),
                z_m=_number(raw.get("z_m", 0.0), f"{name}.z_m"),
                roll_deg=_number(raw.get("roll_deg", 0.0), f"{name}.roll_deg"),
                pitch_deg=_number(raw.get("pitch_deg", 0.0), f"{name}.pitch_deg"),
                yaw_deg=_number(raw.get("yaw_deg", 0.0), f"{name}.yaw_deg"),
            )
        except RecipeError as exc:
            raise RecipeError(f"{name}: {exc}") from exc


def rotation_matrix_zyx(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Return ``Rz(yaw) Ry(pitch) Rx(roll)`` for degree inputs.

    Both authored offsets and absolute poses go through this single function,
    so the UI's documented convention cannot diverge between the two modes.
    """

    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


@dataclass(frozen=True)
class TemplateRef:
    """Reference to one generated workflow element.

    ``pose_check`` supplies both endpoint poses and the phase payload.
    ``step_objects`` supplies only the payload/object state of a step, which
    is how a step without generated pose checks (step 6) can still be used as
    an explicit payload template for a user-authored absolute pose.
    """

    kind: str
    step_index: int
    check_name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in _TEMPLATE_KINDS:
            raise RecipeError(f"template kind must be one of {_TEMPLATE_KINDS}")
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise RecipeError("template step_index must be an integer")
        if not 1 <= self.step_index <= 64:
            raise RecipeError("template step_index must be in [1, 64]")
        if self.kind == "pose_check":
            if not isinstance(self.check_name, str) or not self.check_name:
                raise RecipeError("pose_check template needs a check_name")
            if len(self.check_name) > 128:
                raise RecipeError("template check_name is too long")
        else:
            object.__setattr__(self, "check_name", "")

    @property
    def token(self) -> str:
        if self.kind == "pose_check":
            return f"pose_check:{self.step_index}:{self.check_name}"
        return f"step_objects:{self.step_index}"

    @property
    def supplies_pose(self) -> bool:
        return self.kind == "pose_check"

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "step_index": self.step_index,
            "check_name": self.check_name,
        }

    @classmethod
    def from_mapping(cls, raw: Any, name: str) -> "TemplateRef":
        if not isinstance(raw, Mapping):
            raise RecipeError(f"{name} must be an object")
        unknown = set(raw) - {"kind", "step_index", "check_name"}
        if unknown:
            raise RecipeError(f"{name} has unknown keys: {sorted(unknown)}")
        step = raw.get("step_index")
        if isinstance(step, float) and step.is_integer():
            step = int(step)
        return cls(
            kind=_text(raw.get("kind"), f"{name}.kind", 32),
            step_index=step,
            check_name=_optional_text(
                raw.get("check_name"), f"{name}.check_name", 128
            ),
        )


@dataclass(frozen=True)
class CollisionScenario:
    """Reusable collision configuration selected per check.

    ``excluded_targets`` names environment/process collision targets that are
    removed from the checker input.  ``extra_allowed_contacts`` adds pair
    exceptions on top of the template's intended contacts; it is never a
    global collision switch.
    """

    scenario_id: str
    label: str
    excluded_targets: tuple[str, ...] = ()
    extra_allowed_contacts: tuple[tuple[str, str], ...] = ()
    include_self_collision: bool = True
    include_floor: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_id", _identifier(self.scenario_id, "scenario_id")
        )
        object.__setattr__(self, "label", _text(self.label, "scenario label", 128))
        targets = tuple(
            _text(item, "excluded target", 200) for item in self.excluded_targets
        )
        if len(targets) > MAX_EXCLUDED_TARGETS:
            raise RecipeError("too many excluded collision targets")
        if len(set(targets)) != len(targets):
            raise RecipeError(
                f"scenario {self.scenario_id} repeats an excluded collision target"
            )
        object.__setattr__(self, "excluded_targets", targets)
        contacts: list[tuple[str, str]] = []
        for pair in self.extra_allowed_contacts:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
                raise RecipeError("allowed contact must be a two-name pair")
            if len(pair) != 2:
                raise RecipeError("allowed contact must contain exactly two names")
            first = _text(pair[0], "allowed contact name", 200)
            second = _text(pair[1], "allowed contact name", 200)
            if first == second:
                raise RecipeError(
                    f"allowed contact pair repeats the same name: {first}"
                )
            contacts.append(tuple(sorted((first, second))))
        if len(contacts) > MAX_ALLOWED_CONTACTS:
            raise RecipeError("too many extra allowed contacts")
        if len(set(contacts)) != len(contacts):
            raise RecipeError(
                f"scenario {self.scenario_id} repeats an allowed contact pair"
            )
        object.__setattr__(self, "extra_allowed_contacts", tuple(contacts))
        for name in ("include_self_collision", "include_floor"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise RecipeError(f"scenario {name} must be a boolean")

    @property
    def is_diagnostic(self) -> bool:
        """Whether this scenario weakens the default collision contract."""

        return bool(
            self.excluded_targets
            or self.extra_allowed_contacts
            or not self.include_self_collision
            or not self.include_floor
        )

    def diagnostic_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.excluded_targets:
            reasons.append(
                f"{len(self.excluded_targets)} collision target(s) excluded"
            )
        if self.extra_allowed_contacts:
            reasons.append(
                f"{len(self.extra_allowed_contacts)} extra allowed contact pair(s)"
            )
        if not self.include_self_collision:
            reasons.append("robot self-collision disabled")
        if not self.include_floor:
            reasons.append("floor collision disabled")
        return tuple(reasons)

    def payload(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "label": self.label,
            "excluded_targets": list(self.excluded_targets),
            "extra_allowed_contacts": [list(pair) for pair in self.extra_allowed_contacts],
            "include_self_collision": self.include_self_collision,
            "include_floor": self.include_floor,
            "is_diagnostic": self.is_diagnostic,
            "diagnostic_reasons": list(self.diagnostic_reasons()),
        }

    @classmethod
    def from_mapping(cls, raw: Any, name: str) -> "CollisionScenario":
        if not isinstance(raw, Mapping):
            raise RecipeError(f"{name} must be an object")
        allowed = {
            "scenario_id",
            "label",
            "excluded_targets",
            "extra_allowed_contacts",
            "include_self_collision",
            "include_floor",
            # Derived fields are accepted and ignored so a downloaded recipe
            # can be uploaded again unchanged.
            "is_diagnostic",
            "diagnostic_reasons",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise RecipeError(f"{name} has unknown keys: {sorted(unknown)}")
        return cls(
            scenario_id=raw.get("scenario_id"),
            label=raw.get("label") or str(raw.get("scenario_id") or ""),
            excluded_targets=tuple(_sequence(raw.get("excluded_targets", ()), f"{name}.excluded_targets")),
            extra_allowed_contacts=tuple(
                _sequence(raw.get("extra_allowed_contacts", ()), f"{name}.extra_allowed_contacts")
            ),
            include_self_collision=_boolean(
                raw.get("include_self_collision", True),
                f"{name}.include_self_collision",
            ),
            include_floor=_boolean(
                raw.get("include_floor", True), f"{name}.include_floor"
            ),
        )


def default_collision_scenario() -> CollisionScenario:
    """Return the scenario that reproduces the original evaluator contract."""

    return CollisionScenario(
        scenario_id="default",
        label="Default (process contract)",
    )


@dataclass(frozen=True)
class StudyCheck:
    """One user-authored straight TCP pose segment."""

    check_id: str
    label: str
    criticality: str = "hard"
    enabled: bool = True
    scenario_id: str = "default"
    pose_mode: str = "template_relative"
    template: TemplateRef | None = None
    payload_source: str = "template"
    offset_frame: str = "tcp_local"
    start_offset: Pose6D = field(default_factory=Pose6D)
    end_offset: Pose6D = field(default_factory=Pose6D)
    reference_frame: str = WORLD_FRAME
    start_pose: Pose6D = field(default_factory=Pose6D)
    end_pose: Pose6D = field(default_factory=Pose6D)
    include_in_metric_summary: bool = True
    source_check_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", _identifier(self.check_id, "check_id"))
        object.__setattr__(self, "label", _text(self.label, "check label", 128))
        if self.criticality not in _CRITICALITIES:
            raise RecipeError(f"check criticality must be one of {_CRITICALITIES}")
        if self.pose_mode not in _POSE_MODES:
            raise RecipeError(f"check pose_mode must be one of {_POSE_MODES}")
        if self.payload_source not in _PAYLOAD_SOURCES:
            raise RecipeError(f"check payload_source must be one of {_PAYLOAD_SOURCES}")
        if self.offset_frame not in _OFFSET_FRAMES:
            raise RecipeError(f"check offset_frame must be one of {_OFFSET_FRAMES}")
        for name in ("enabled", "include_in_metric_summary"):
            if not isinstance(getattr(self, name), bool):
                raise RecipeError(f"check {name} must be a boolean")
        object.__setattr__(
            self, "scenario_id", _identifier(self.scenario_id, "check scenario_id")
        )
        object.__setattr__(
            self, "reference_frame", _text(self.reference_frame, "reference_frame", 128)
        )
        if self.pose_mode == "template_relative":
            if self.template is None:
                raise RecipeError(
                    f"check {self.check_id} is template-relative but has no template"
                )
            if not self.template.supplies_pose:
                raise RecipeError(
                    f"check {self.check_id} template {self.template.token} supplies "
                    "no pose; use pose_mode='absolute'"
                )
        if self.payload_source == "template" and self.template is None:
            raise RecipeError(
                f"check {self.check_id} requests a template payload but selects "
                "no template; choose a template or payload_source='none'"
            )

    @property
    def criticality_is_hard(self) -> bool:
        return self.criticality == "hard"

    def payload(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "label": self.label,
            "criticality": self.criticality,
            "enabled": self.enabled,
            "scenario_id": self.scenario_id,
            "pose_mode": self.pose_mode,
            "template": None if self.template is None else self.template.payload(),
            "payload_source": self.payload_source,
            "offset_frame": self.offset_frame,
            "start_offset": self.start_offset.payload(),
            "end_offset": self.end_offset.payload(),
            "reference_frame": self.reference_frame,
            "start_pose": self.start_pose.payload(),
            "end_pose": self.end_pose.payload(),
            "include_in_metric_summary": self.include_in_metric_summary,
            "source_check_name": self.source_check_name,
        }

    @property
    def is_unmodified_template(self) -> bool:
        """Whether this check reproduces its generated workflow original."""

        return (
            self.pose_mode == "template_relative"
            and self.payload_source == "template"
            and self.start_offset.is_identity
            and self.end_offset.is_identity
        )

    @classmethod
    def from_mapping(cls, raw: Any, name: str) -> "StudyCheck":
        if not isinstance(raw, Mapping):
            raise RecipeError(f"{name} must be an object")
        allowed = {
            "check_id",
            "label",
            "criticality",
            "enabled",
            "scenario_id",
            "pose_mode",
            "template",
            "payload_source",
            "offset_frame",
            "start_offset",
            "end_offset",
            "reference_frame",
            "start_pose",
            "end_pose",
            "include_in_metric_summary",
            "source_check_name",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise RecipeError(f"{name} has unknown keys: {sorted(unknown)}")
        template_raw = raw.get("template")
        return cls(
            check_id=raw.get("check_id"),
            label=raw.get("label") or str(raw.get("check_id") or ""),
            criticality=_text(raw.get("criticality", "hard"), f"{name}.criticality", 16),
            enabled=_boolean(raw.get("enabled", True), f"{name}.enabled"),
            scenario_id=raw.get("scenario_id", "default"),
            pose_mode=_text(
                raw.get("pose_mode", "template_relative"), f"{name}.pose_mode", 32
            ),
            template=(
                None
                if template_raw is None
                else TemplateRef.from_mapping(template_raw, f"{name}.template")
            ),
            payload_source=_text(
                raw.get("payload_source", "template"), f"{name}.payload_source", 16
            ),
            offset_frame=_text(
                raw.get("offset_frame", "tcp_local"), f"{name}.offset_frame", 16
            ),
            start_offset=Pose6D.from_mapping(raw.get("start_offset"), f"{name}.start_offset"),
            end_offset=Pose6D.from_mapping(raw.get("end_offset"), f"{name}.end_offset"),
            reference_frame=_text(
                raw.get("reference_frame", WORLD_FRAME), f"{name}.reference_frame", 128
            ),
            start_pose=Pose6D.from_mapping(raw.get("start_pose"), f"{name}.start_pose"),
            end_pose=Pose6D.from_mapping(raw.get("end_pose"), f"{name}.end_pose"),
            include_in_metric_summary=_boolean(
                raw.get("include_in_metric_summary", True),
                f"{name}.include_in_metric_summary",
            ),
            source_check_name=_optional_text(
                raw.get("source_check_name"), f"{name}.source_check_name", 128
            ),
        )


@dataclass(frozen=True)
class StudyRecipe:
    """Complete reproducible study input."""

    recipe_id: str
    label: str
    parameters: Mapping[str, object]
    scene_fingerprint: str = ""
    profile: str = "quick"
    approach_distance_m: float = 0.15
    checks: tuple[StudyCheck, ...] = ()
    scenarios: tuple[CollisionScenario, ...] = field(
        default_factory=lambda: (default_collision_scenario(),)
    )
    schema_version: int = STUDY_SCHEMA_VERSION
    notes: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != STUDY_SCHEMA_VERSION:
            raise RecipeError(
                f"unsupported study recipe schema: {self.schema_version!r}; "
                f"this build writes {STUDY_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "recipe_id", _identifier(self.recipe_id, "recipe_id"))
        object.__setattr__(self, "label", _text(self.label, "recipe label", 200))
        object.__setattr__(self, "notes", _optional_text(self.notes, "notes", 2000))
        object.__setattr__(
            self,
            "scene_fingerprint",
            _optional_text(self.scene_fingerprint, "scene_fingerprint", 128),
        )
        if self.scene_fingerprint and not re.fullmatch(
            r"[0-9a-f]{64}", self.scene_fingerprint
        ):
            raise RecipeError("scene_fingerprint must be a sha256 hex digest")
        if not isinstance(self.profile, str) or not self.profile:
            raise RecipeError("profile must be a non-empty string")
        approach = _number(self.approach_distance_m, "approach_distance_m")
        if not 0.0 < approach <= 2.0:
            raise RecipeError("approach_distance_m must be in (0, 2]")
        object.__setattr__(self, "approach_distance_m", approach)
        if not isinstance(self.parameters, Mapping):
            raise RecipeError("parameters must be an object")
        object.__setattr__(self, "parameters", dict(self.parameters))

        if not self.scenarios:
            raise RecipeError("at least one collision scenario is required")
        if len(self.scenarios) > MAX_SCENARIOS:
            raise RecipeError(f"at most {MAX_SCENARIOS} collision scenarios are allowed")
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise RecipeError("collision scenario ids must be unique")
        if len(self.checks) > MAX_CHECKS:
            raise RecipeError(f"at most {MAX_CHECKS} checks are allowed")
        check_ids = [check.check_id for check in self.checks]
        if len(set(check_ids)) != len(check_ids):
            raise RecipeError("check ids must be unique")
        known = set(scenario_ids)
        for check in self.checks:
            if check.scenario_id not in known:
                raise RecipeError(
                    f"check {check.check_id} references unknown collision "
                    f"scenario {check.scenario_id!r}"
                )

    def scenario(self, scenario_id: str) -> CollisionScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise RecipeError(f"unknown collision scenario: {scenario_id!r}")

    def check(self, check_id: str) -> StudyCheck:
        for check in self.checks:
            if check.check_id == check_id:
                return check
        raise RecipeError(f"unknown study check: {check_id!r}")

    @property
    def enabled_checks(self) -> tuple[StudyCheck, ...]:
        return tuple(check for check in self.checks if check.enabled)

    @property
    def enabled_hard_checks(self) -> tuple[StudyCheck, ...]:
        return tuple(
            check for check in self.enabled_checks if check.criticality_is_hard
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "label": self.label,
            "notes": self.notes,
            "scene_fingerprint": self.scene_fingerprint,
            "profile": self.profile,
            "approach_distance_m": self.approach_distance_m,
            "parameters": dict(self.parameters),
            "scenarios": [scenario.payload() for scenario in self.scenarios],
            "checks": [check.payload() for check in self.checks],
        }

    def canonical_json(self) -> str:
        """Return the deterministic identity text used for hashing."""

        payload = self.payload()
        for scenario in payload["scenarios"]:
            scenario.pop("is_diagnostic", None)
            scenario.pop("diagnostic_reasons", None)
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def recipe_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def with_fingerprint(self, fingerprint: str) -> "StudyRecipe":
        return replace(self, scene_fingerprint=fingerprint)

    @classmethod
    def from_mapping(cls, raw: Any, name: str = "recipe") -> "StudyRecipe":
        if not isinstance(raw, Mapping):
            raise RecipeError(f"{name} must be an object")
        allowed = {
            "schema_version",
            "recipe_id",
            "label",
            "notes",
            "scene_fingerprint",
            "profile",
            "approach_distance_m",
            "parameters",
            "scenarios",
            "checks",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise RecipeError(f"{name} has unknown keys: {sorted(unknown)}")
        version = raw.get("schema_version", STUDY_SCHEMA_VERSION)
        if isinstance(version, float) and version.is_integer():
            version = int(version)
        scenarios_raw = _sequence(
            raw.get("scenarios", ()), f"{name}.scenarios", empty_ok=True
        )
        scenarios = tuple(
            CollisionScenario.from_mapping(item, f"{name}.scenarios[{index}]")
            for index, item in enumerate(scenarios_raw)
        ) or (default_collision_scenario(),)
        checks_raw = _sequence(raw.get("checks", ()), f"{name}.checks", empty_ok=True)
        return cls(
            schema_version=version,
            recipe_id=raw.get("recipe_id"),
            label=raw.get("label") or str(raw.get("recipe_id") or ""),
            notes=raw.get("notes", ""),
            scene_fingerprint=raw.get("scene_fingerprint", ""),
            profile=_text(raw.get("profile", "quick"), f"{name}.profile", 32),
            approach_distance_m=_number(
                raw.get("approach_distance_m", 0.15), f"{name}.approach_distance_m"
            ),
            parameters=raw.get("parameters", {}),
            scenarios=scenarios,
            checks=tuple(
                StudyCheck.from_mapping(item, f"{name}.checks[{index}]")
                for index, item in enumerate(checks_raw)
            ),
        )

    @classmethod
    def from_json(cls, text: str, name: str = "recipe") -> "StudyRecipe":
        encoded = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if len(encoded) > MAX_RECIPE_BYTES:
            raise RecipeError(
                f"recipe JSON exceeds {MAX_RECIPE_BYTES} bytes"
            )
        try:
            decoded = json.loads(encoded.decode("utf-8"), parse_constant=_reject_constant)
        except UnicodeDecodeError as exc:
            raise RecipeError("recipe JSON must be UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise RecipeError(f"invalid recipe JSON: {exc.msg}") from exc
        return cls.from_mapping(decoded, name)


def _reject_constant(value: str) -> object:
    raise RecipeError(f"recipe JSON must not contain {value}")


def recipe_file_name(recipe_id: str) -> str:
    """Return the sanitized, traversal-free file name of a stored recipe."""

    return f"{_identifier(recipe_id, 'recipe_id')}.json"


def _identifier(value: Any, name: str) -> str:
    text = _text(value, name, 64)
    if not _IDENTIFIER.fullmatch(text):
        raise RecipeError(
            f"{name} must match [a-z0-9][a-z0-9_-]{{0,63}}; got {text!r}"
        )
    return text


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise RecipeError(f"{name} must be a string")
    text = value.strip()
    if not text:
        raise RecipeError(f"{name} must not be empty")
    if len(text) > limit:
        raise RecipeError(f"{name} must be at most {limit} characters")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in text):
        raise RecipeError(f"{name} must not contain control characters")
    return text


def _optional_text(value: Any, name: str, limit: int) -> str:
    if value is None or value == "":
        return ""
    return _text(value, name, limit)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RecipeError(f"{name} must be finite")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RecipeError(f"{name} must be a boolean")
    return value


def _sequence(value: Any, name: str, *, empty_ok: bool = True) -> list[Any]:
    if value is None:
        if not empty_ok:
            raise RecipeError(f"{name} is required")
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RecipeError(f"{name} must be an array")
    if len(value) > max(MAX_CHECKS, MAX_EXCLUDED_TARGETS):
        raise RecipeError(f"{name} is too long")
    return list(value)
