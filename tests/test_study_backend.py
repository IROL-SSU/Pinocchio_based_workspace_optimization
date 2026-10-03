from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading

import numpy as np
import pytest

from decanting_workspace.evaluation import EvaluationOptions
from decanting_workspace.kinematics import IKOptions
from decanting_workspace.models import BasePose, load_scene_spec
from decanting_workspace.precompute import iter_case_keys, scene_spec_fingerprint
from decanting_workspace.precompute_cli import load_case_grid_yaml
from decanting_workspace.robots import load_robot_bundle
from decanting_workspace.study_backend import (
    MAX_SCOUT_CANDIDATES,
    StudyEvaluationBackend,
    _apply_offset,
    _apply_scenario,
    _pose_to_6d,
    _scout_candidates,
    _scout_settings,
    _study_result_payload,
    check_outcome,
    relative_pose_6d,
)
from decanting_workspace.study_recipe import (
    CollisionScenario,
    Pose6D,
    RecipeError,
    StudyCheck,
    StudyRecipe,
    TemplateRef,
    rotation_matrix_zyx,
)
from decanting_workspace.study_store import StudyRecipeStore
from decanting_workspace.paths import repository_path


QUICK = EvaluationOptions(
    cartesian_translation_step_m=10.0,
    cartesian_rotation_step_rad=math.pi,
    joint_interpolation_samples=0,
    characteristic_length_m=0.5,
    ik=IKOptions(
        position_tolerance_m=1e-6,
        orientation_tolerance_rad=math.radians(1e-4),
    ),
)


@dataclass
class _Assets:
    spec: object
    key: object
    ur: object
    sr: object


@pytest.fixture(scope="module")
def assets() -> _Assets:
    spec = load_scene_spec()
    grid = load_case_grid_yaml(repository_path("config", "case_grid_working.yaml"))
    key = next(iter_case_keys(spec, grid))
    ur = load_robot_bundle(
        "ur20",
        spec.robots["ur20"].urdf_path,
        floating_base=True,
        load_visual=False,
        suction_proxy=spec.robots["ur20"].suction_proxy,
    )
    sr = load_robot_bundle(
        "sr12ia",
        spec.robots["sr12ia"].urdf_path,
        floating_base=True,
        load_visual=False,
    )
    return _Assets(spec=spec, key=key, ur=ur, sr=sr)


def _backend(assets: _Assets, tmp_path, viewer=None) -> StudyEvaluationBackend:
    return StudyEvaluationBackend(
        assets.spec,
        assets.key,
        assets.ur,
        assets.sr,
        viewer,
        profiles={"quick": QUICK},
        default_profile="quick",
        store=StudyRecipeStore(tmp_path),
    )


@pytest.fixture
def backend(assets, tmp_path) -> StudyEvaluationBackend:
    return _backend(assets, tmp_path)


@pytest.fixture(scope="module")
def catalog(assets, tmp_path_factory):
    return _backend(assets, tmp_path_factory.mktemp("catalog")).catalog({})


@pytest.fixture(scope="module")
def baseline(catalog) -> StudyRecipe:
    return StudyRecipe.from_mapping(catalog["baseline_recipe"])


# ======================================================== pure transforms


def test_tcp_local_offset_post_multiplies_the_template_pose():
    template = np.eye(4)
    template[:3, :3] = rotation_matrix_zyx(0.0, 0.0, 90.0)
    template[:3, 3] = (1.0, 2.0, 3.0)

    moved = _apply_offset(template, Pose6D(x_m=0.1), "tcp_local")

    # +X in the template's own frame is +Y in World after a 90 degree yaw.
    assert np.allclose(moved[:3, 3], (1.0, 2.1, 3.0))
    assert np.allclose(moved[:3, :3], template[:3, :3])


def test_world_offset_pre_multiplies_and_therefore_moves_the_tcp_origin():
    template = np.eye(4)
    template[:3, :3] = rotation_matrix_zyx(0.0, 0.0, 90.0)
    template[:3, 3] = (1.0, 0.0, 0.0)

    moved = _apply_offset(template, Pose6D(yaw_deg=90.0), "world")

    # A World rotation is about the World origin, so the position rotates too.
    assert np.allclose(moved[:3, 3], (0.0, 1.0, 0.0), atol=1e-12)
    assert np.allclose(
        moved[:3, :3], rotation_matrix_zyx(0.0, 0.0, 180.0), atol=1e-12
    )


def test_tcp_local_and_world_offsets_differ_for_a_rotated_template():
    template = np.eye(4)
    template[:3, :3] = rotation_matrix_zyx(0.0, 0.0, 90.0)
    template[:3, 3] = (1.0, 0.0, 0.0)
    local = _apply_offset(template, Pose6D(x_m=0.25), "tcp_local")
    world = _apply_offset(template, Pose6D(x_m=0.25), "world")
    assert not np.allclose(local, world)
    assert np.allclose(world[:3, 3], (1.25, 0.0, 0.0))


def test_zero_offset_is_the_identity_in_both_frames():
    template = np.eye(4)
    template[:3, :3] = rotation_matrix_zyx(10.0, 20.0, 30.0)
    template[:3, 3] = (0.4, -1.2, 0.9)
    for frame in ("tcp_local", "world"):
        assert np.allclose(_apply_offset(template, Pose6D(), frame), template)


def test_offset_products_stay_orthonormal_under_repeated_application():
    matrix = np.eye(4)
    for _ in range(400):
        matrix = _apply_offset(matrix, Pose6D(yaw_deg=1.0, pitch_deg=0.7), "tcp_local")
    rotation = matrix[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-10)


def test_pose_to_6d_round_trips_through_the_declared_convention():
    original = Pose6D(0.1, -0.2, 0.3, 12.0, -34.0, 56.0)
    decomposed = _pose_to_6d(original.matrix())
    assert np.allclose(decomposed.matrix(), original.matrix(), atol=1e-12)


def test_pose_to_6d_handles_the_gimbal_lock_pitch():
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix_zyx(0.0, 90.0, 40.0)
    decomposed = _pose_to_6d(matrix)
    assert np.allclose(decomposed.matrix()[:3, :3], matrix[:3, :3], atol=1e-9)


def test_relative_pose_recovers_a_known_frame_offset():
    reference = np.eye(4)
    reference[:3, :3] = rotation_matrix_zyx(0.0, 0.0, 90.0)
    reference[:3, 3] = (1.0, 1.0, 0.0)
    offset = Pose6D(x_m=0.3, z_m=-0.2, yaw_deg=15.0)
    recovered = relative_pose_6d(reference, reference @ offset.matrix())
    assert np.allclose(recovered.matrix(), offset.matrix(), atol=1e-12)


# ============================================================ catalog data


def test_catalog_exposes_pose_and_payload_templates_for_every_step(catalog):
    tokens = {template["token"] for template in catalog["templates"]}
    assert "pose_check:1:pallet_box_approach" in tokens
    # Step 6 generates no pose check, but its attached payload is still usable.
    assert "step_objects:6" in tokens
    step6 = next(
        item for item in catalog["templates"] if item["token"] == "step_objects:6"
    )
    assert step6["supplies_pose"] is False
    assert step6["start"] is None
    assert step6["payload_name"]


def test_catalog_pose_templates_carry_world_poses_and_contacts(catalog):
    template = next(
        item
        for item in catalog["templates"]
        if item["token"] == "pose_check:1:pallet_box_lift"
    )
    assert template["supplies_pose"] is True
    assert template["start"] and template["end"]
    assert template["payload_name"]
    assert any("suction_pad" in pair for pair in template["allowed_contacts"])


def test_catalog_lists_collision_targets_with_roles_and_sources(catalog):
    targets = {item["name"]: item for item in catalog["collision_targets"]}
    assert "pallet" in targets
    assert targets["pallet"]["role"] == "pallet"
    assert targets["pallet"]["geometry_kind"] == "box"
    assert "ur20_pedestal" in targets
    # The dimension-based scene has no USD meshes, so prim paths are blank
    # rather than invented.
    assert all(item["source_prim_path"] == "" for item in targets.values())
    assert {item["source"] for item in targets.values()} <= {
        "generated_box",
        "generated_keepout",
        "process_object",
        "phase_object",
        "usd_mesh",
    }


def test_catalog_reference_frames_start_with_world(catalog):
    names = [frame["name"] for frame in catalog["reference_frames"]]
    assert names[0] == "world"
    assert "UncasingLoadFrame" in names
    world = catalog["reference_frames"][0]
    assert world["translation_m"] == [0.0, 0.0, 0.0]
    assert world["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]


def test_catalog_contact_names_include_robot_geometry_and_logical_aliases(catalog):
    names = set(catalog["contact_names"])
    assert "suction_pad" in names
    assert "floor" in names
    assert "pallet" in names
    assert any("wrist" in name for name in names)


def test_catalog_documents_units_and_transform_order(catalog):
    units = catalog["units"]
    assert units["ui_length"] == "mm"
    assert units["stored_length"] == "m"
    assert units["rotation_convention"] == "Rz(yaw) Ry(pitch) Rx(roll)"
    assert "world_T_new = world_T_template * offset" in units["template_relative_order"]


def test_catalog_is_strictly_json_serializable(catalog):
    json.dumps(catalog, allow_nan=False)


def test_baseline_recipe_references_the_generated_checks_at_zero_offset(baseline):
    assert baseline.checks
    for check in baseline.checks:
        assert check.pose_mode == "template_relative"
        assert check.payload_source == "template"
        assert check.template is not None
        assert check.template.supplies_pose
        assert check.start_offset.is_identity
        assert check.end_offset.is_identity
        assert check.is_unmodified_template
    assert baseline.scene_fingerprint
    assert len(baseline.scenarios) == 1
    assert not baseline.scenarios[0].is_diagnostic


def test_baseline_covers_every_generated_hard_check(backend, baseline):
    coverage = backend.validate({"recipe": baseline.payload()})["coverage"]
    assert coverage["covers_generated_hard_set"] is True
    assert coverage["omitted_generated_hard_checks"] == []
    assert coverage["modified_generated_hard_checks"] == []
    # Step 6 is never generated, so a complete authored set is still partial.
    assert 6 in coverage["skipped_process_steps"]


# ========================================================= pose resolution


def test_zero_offset_resolution_reproduces_the_generated_world_pose(
    backend, baseline, catalog
):
    resolved = {
        item["check_id"]: item
        for item in backend.validate({"recipe": baseline.payload()})[
            "resolved_checks"
        ]
    }
    for template in catalog["templates"]:
        if not template["supplies_pose"]:
            continue
        match = next(
            (
                item
                for check, item in zip(baseline.checks, resolved.values())
                if check.template is not None
                and check.template.token == template["token"]
            ),
            None,
        )
        if match is None:
            continue
        for key in ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"):
            assert match["world_start"][key] == pytest.approx(
                template["start"][key], abs=1e-9
            )
            assert match["world_end"][key] == pytest.approx(
                template["end"][key], abs=1e-9
            )


def test_a_local_offset_moves_the_resolved_pose_by_exactly_that_amount(
    backend, baseline
):
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:1]
    before = backend.validate({"recipe": payload})["resolved_checks"][0]
    payload["checks"][0]["start_offset"]["z_m"] = 0.05
    after = backend.validate({"recipe": payload})["resolved_checks"][0]

    before_matrix = Pose6D(**before["world_start"]).matrix()
    after_matrix = Pose6D(**after["world_start"]).matrix()
    delta = np.linalg.inv(before_matrix) @ after_matrix
    assert np.allclose(delta, Pose6D(z_m=0.05).matrix(), atol=1e-9)
    assert after["is_unmodified_template"] is False


def test_absolute_pose_is_composed_with_the_named_reference_frame(
    backend, baseline, catalog
):
    frame = next(
        item
        for item in catalog["reference_frames"]
        if item["name"] == "UncasingLoadFrame"
    )
    payload = baseline.payload()
    payload["checks"] = [
        {
            "check_id": "abs1",
            "label": "absolute probe",
            "criticality": "soft",
            "enabled": True,
            "scenario_id": "default",
            "pose_mode": "absolute",
            "template": None,
            "payload_source": "none",
            "offset_frame": "tcp_local",
            "start_offset": Pose6D().payload(),
            "end_offset": Pose6D().payload(),
            "reference_frame": "UncasingLoadFrame",
            "start_pose": Pose6D(x_m=0.2, z_m=0.4, yaw_deg=30.0).payload(),
            "end_pose": Pose6D(x_m=0.2, z_m=0.5, yaw_deg=30.0).payload(),
            "include_in_metric_summary": True,
            "source_check_name": "",
        }
    ]
    resolved = backend.validate({"recipe": payload})["resolved_checks"][0]

    from decanting_workspace.transforms import quaternion_matrix

    expected = quaternion_matrix(
        frame["translation_m"], frame["rotation_xyzw"]
    ) @ Pose6D(x_m=0.2, z_m=0.4, yaw_deg=30.0).matrix()
    assert np.allclose(
        Pose6D(**resolved["world_start"]).matrix(), expected, atol=1e-9
    )
    assert resolved["segment_length_m"] == pytest.approx(0.1, abs=1e-9)


def test_world_reference_frame_is_the_identity(backend, baseline):
    payload = baseline.payload()
    payload["checks"] = [
        {
            **baseline.payload()["checks"][0],
            "check_id": "abs-world",
            "pose_mode": "absolute",
            "payload_source": "none",
            "template": None,
            "reference_frame": "world",
            "start_pose": Pose6D(x_m=-1.0, y_m=-7.0, z_m=1.2).payload(),
            "end_pose": Pose6D(x_m=-1.0, y_m=-7.0, z_m=1.3).payload(),
        }
    ]
    resolved = backend.validate({"recipe": payload})["resolved_checks"][0]
    assert resolved["world_start"]["x_m"] == pytest.approx(-1.0)
    assert resolved["world_start"]["z_m"] == pytest.approx(1.2)


def test_unknown_reference_frame_is_rejected_before_any_calculation(
    backend, baseline
):
    payload = baseline.payload()
    payload["checks"] = [
        {
            **payload["checks"][0],
            "pose_mode": "absolute",
            "payload_source": "none",
            "template": None,
            "reference_frame": "NoSuchFrame",
        }
    ]
    with pytest.raises(RecipeError, match="unknown reference frame"):
        backend.validate({"recipe": payload})


def test_missing_source_check_is_rejected(backend, baseline):
    payload = baseline.payload()
    payload["checks"] = [
        {
            **payload["checks"][0],
            "template": {
                "kind": "pose_check",
                "step_index": 1,
                "check_name": "no_such_check",
            },
        }
    ]
    with pytest.raises(RecipeError, match="no pose check"):
        backend.validate({"recipe": payload})


def test_missing_source_step_is_rejected(backend, baseline):
    payload = baseline.payload()
    payload["checks"] = [
        {
            **payload["checks"][0],
            "template": {
                "kind": "pose_check",
                "step_index": 42,
                "check_name": "whatever",
            },
        }
    ]
    with pytest.raises(RecipeError, match="no step 42"):
        backend.validate({"recipe": payload})


# ======================================================== payload cloning


def test_cloning_a_check_preserves_the_tcp_relative_payload_transform(
    backend, baseline
):
    payload = baseline.payload()
    source = next(
        item
        for item in payload["checks"]
        if item["template"]["check_name"] == "pallet_box_lift"
    )
    clone = {
        **json.loads(json.dumps(source)),
        "check_id": "clone1",
        "label": "clone of lift",
    }
    clone["start_offset"]["z_m"] = 0.12
    clone["end_offset"]["z_m"] = 0.12
    payload["checks"] = [source, clone]

    resolved = {
        item["check_id"]: item
        for item in backend.validate({"recipe": payload})["resolved_checks"]
    }
    original = resolved[source["check_id"]]
    copied = resolved["clone1"]

    assert original["payload_objects"] and copied["payload_objects"]
    assert [item["name"] for item in copied["payload_objects"]] == [
        item["name"] for item in original["payload_objects"]
    ]
    for first, second in zip(
        original["payload_objects"], copied["payload_objects"]
    ):
        assert first["attached_to_tcp"] == second["attached_to_tcp"]
        assert first["follows_tcp"] == second["follows_tcp"]
        # The payload must keep its exact TCP-relative transform even though
        # the authored start pose moved.
        assert first["tcp_relative"] == second["tcp_relative"]
    assert original["allowed_contacts"] == copied["allowed_contacts"]
    assert original["attached_count"] == copied["attached_count"]
    # The moved clone must not resolve to the original world pose.
    assert copied["world_start"]["z_m"] != pytest.approx(
        original["world_start"]["z_m"]
    )


def test_free_space_payload_drops_objects_and_template_contacts(backend, baseline):
    payload = baseline.payload()
    source = next(
        item
        for item in payload["checks"]
        if item["template"]["check_name"] == "pallet_box_lift"
    )
    free = {**json.loads(json.dumps(source)), "check_id": "free1", "payload_source": "none"}
    payload["checks"] = [source, free]
    resolved = {
        item["check_id"]: item
        for item in backend.validate({"recipe": payload})["resolved_checks"]
    }
    assert resolved[source["check_id"]]["payload_objects"]
    assert resolved["free1"]["payload_objects"] == []
    assert resolved["free1"]["allowed_contacts"] == []
    assert resolved["free1"]["attached_count"] == 0
    # The free-space phase must not see the process boxes at nominal positions.
    assert resolved["free1"]["obstacle_count"] >= 1


def test_step_six_payload_template_supplies_the_attached_box_for_a_new_pose(
    backend, baseline, catalog
):
    template = next(
        item for item in catalog["templates"] if item["token"] == "step_objects:6"
    )
    source = baseline.payload()["checks"][0]
    payload = baseline.payload()
    payload["checks"] = [
        {
            **source,
            "check_id": "pour-probe",
            "label": "authored pouring probe",
            "criticality": "soft",
            "pose_mode": "absolute",
            "payload_source": "template",
            "template": template["reference"],
            "reference_frame": "world",
            "start_pose": Pose6D(x_m=-1.0, y_m=-7.0, z_m=1.4).payload(),
            "end_pose": Pose6D(x_m=-1.0, y_m=-7.0, z_m=1.4).payload(),
        }
    ]
    resolved = backend.validate({"recipe": payload})["resolved_checks"][0]
    attached = [item for item in resolved["payload_objects"] if item["attached_to_tcp"]]
    assert attached, "step 6 must contribute its attached carton"
    assert resolved["attached_count"] == len(attached)
    # It is a user-authored probe, not the generated pouring verification.
    assert resolved["is_unmodified_template"] is False


def test_template_relative_mode_is_refused_for_a_payload_only_template(
    backend, baseline, catalog
):
    template = next(
        item for item in catalog["templates"] if item["token"] == "step_objects:6"
    )
    payload = baseline.payload()
    payload["checks"] = [
        {
            **payload["checks"][0],
            "pose_mode": "template_relative",
            "template": template["reference"],
        }
    ]
    with pytest.raises(RecipeError, match="supplies"):
        backend.validate({"recipe": payload})


# ====================================================== collision scenario


def _two_scenario_payload(baseline: StudyRecipe, **scenario) -> dict:
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:1]
    payload["scenarios"] = [
        payload["scenarios"][0],
        {
            "scenario_id": "probe",
            "label": "probe scenario",
            "excluded_targets": [],
            "extra_allowed_contacts": [],
            "include_self_collision": True,
            "include_floor": True,
            **scenario,
        },
    ]
    return payload


def test_excluding_a_target_removes_it_from_the_actual_checker_input(
    backend, baseline
):
    payload = _two_scenario_payload(baseline, excluded_targets=["pallet"])
    before = backend.validate({"recipe": payload})["resolved_checks"][0]
    payload["checks"][0]["scenario_id"] = "probe"
    after = backend.validate({"recipe": payload})["resolved_checks"][0]

    assert after["obstacle_count"] == before["obstacle_count"] - 1
    assert after["excluded_targets_applied"] == ["pallet"]
    assert after["excluded_targets_missing"] == []


def test_excluding_an_attached_payload_removes_the_attached_geometry(
    backend, baseline
):
    payload = baseline.payload()
    lift = next(
        item
        for item in payload["checks"]
        if item["template"]["check_name"] == "pallet_box_lift"
    )
    payload["checks"] = [lift]
    before = backend.validate({"recipe": payload})["resolved_checks"][0]
    assert before["attached_count"] >= 1
    carried = before["payload_objects"][0]["name"]

    payload["scenarios"] = [
        payload["scenarios"][0],
        {
            "scenario_id": "probe",
            "label": "probe",
            "excluded_targets": [carried],
            "extra_allowed_contacts": [],
            "include_self_collision": True,
            "include_floor": True,
        },
    ]
    payload["checks"][0]["scenario_id"] = "probe"
    after = backend.validate({"recipe": payload})["resolved_checks"][0]
    assert after["attached_count"] == before["attached_count"] - 1


def test_excluding_an_absent_name_is_reported_rather_than_silently_accepted(
    backend, baseline
):
    payload = _two_scenario_payload(baseline, excluded_targets=["conveyor_level_3"])
    payload["checks"][0]["scenario_id"] = "probe"
    report = backend.validate({"recipe": payload})
    resolved = report["resolved_checks"][0]
    # The name is a real scene target, but this phase may not contain it.
    assert set(resolved["excluded_targets_applied"]) | set(
        resolved["excluded_targets_missing"]
    ) == {"conveyor_level_3"}
    if resolved["excluded_targets_missing"]:
        assert any(
            "absent" in reason for reason in report["diagnostic"]["reasons"]
        )


def test_dropping_floor_or_self_collision_is_visible_in_the_resolved_check(
    backend, baseline
):
    payload = _two_scenario_payload(
        baseline, include_floor=False, include_self_collision=False
    )
    payload["checks"][0]["scenario_id"] = "probe"
    resolved = backend.validate({"recipe": payload})["resolved_checks"][0]
    assert resolved["include_floor"] is False
    assert resolved["include_self_collision"] is False


def test_extra_allowed_contacts_are_added_to_the_phase(backend, baseline):
    payload = _two_scenario_payload(
        baseline, extra_allowed_contacts=[["suction_pad", "pallet"]]
    )
    before = backend.validate({"recipe": payload})["resolved_checks"][0]
    payload["checks"][0]["scenario_id"] = "probe"
    after = backend.validate({"recipe": payload})["resolved_checks"][0]

    # The template's intended contacts are unchanged; the extra pair is added
    # to the set the collision checker actually exempts.
    assert after["allowed_contacts"] == before["allowed_contacts"]
    assert ["pallet", "suction_pad"] in after["phase_allowed_contacts"]
    assert ["pallet", "suction_pad"] not in before["phase_allowed_contacts"]
    assert len(after["phase_allowed_contacts"]) == len(
        before["phase_allowed_contacts"]
    ) + 1


def test_an_unknown_allowed_contact_name_is_rejected(backend, baseline):
    payload = _two_scenario_payload(
        baseline, extra_allowed_contacts=[["suction_pad", "not_a_geometry"]]
    )
    payload["checks"][0]["scenario_id"] = "probe"
    with pytest.raises(RecipeError, match="unknown logical/geometry name"):
        backend.validate({"recipe": payload})


def test_multiple_scenarios_apply_per_check(backend, baseline):
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:2]
    payload["scenarios"] = [
        payload["scenarios"][0],
        {
            "scenario_id": "no-pallet",
            "label": "pallet excluded",
            "excluded_targets": ["pallet"],
            "extra_allowed_contacts": [],
            "include_self_collision": True,
            "include_floor": True,
        },
    ]
    payload["checks"][1]["scenario_id"] = "no-pallet"
    resolved = backend.validate({"recipe": payload})["resolved_checks"]
    assert resolved[0]["excluded_targets_applied"] == []
    assert resolved[1]["excluded_targets_applied"] == ["pallet"]


def test_apply_scenario_reuses_geometry_objects_by_identity(backend, baseline):
    from decanting_workspace.collision_world import CollisionPhase
    from decanting_workspace.models import BoxPrimitive

    keep = BoxPrimitive(
        name="keep", role="obstacle", center_m=(0.0, 0.0, 0.5), size_m=(1.0, 1.0, 1.0)
    )
    drop = BoxPrimitive(
        name="drop", role="obstacle", center_m=(2.0, 0.0, 0.5), size_m=(1.0, 1.0, 1.0)
    )
    phase = CollisionPhase(obstacles=(keep, drop), floor_z_m=0.0)
    scenario = CollisionScenario(
        scenario_id="s", label="s", excluded_targets=("drop",)
    )
    updated, applied, missing = _apply_scenario(phase, scenario)

    assert [box.name for box in updated.obstacles] == ["keep"]
    assert updated.obstacles[0] is keep  # identity preserved for BVH reuse
    assert applied == ("drop",)
    assert missing == ()
    # The input phase is untouched.
    assert [box.name for box in phase.obstacles] == ["keep", "drop"]


def test_apply_scenario_disables_the_floor_only_when_requested(backend):
    from decanting_workspace.collision_world import CollisionPhase

    phase = CollisionPhase(obstacles=(), floor_z_m=0.25)
    kept, _, _ = _apply_scenario(
        phase, CollisionScenario(scenario_id="s", label="s")
    )
    dropped, _, _ = _apply_scenario(
        phase, CollisionScenario(scenario_id="s", label="s", include_floor=False)
    )
    assert kept.floor_z_m == 0.25
    assert dropped.floor_z_m is None


# ================================================= fingerprint and storage


def test_fingerprint_mismatch_is_refused_without_an_explicit_override(
    backend, baseline
):
    payload = baseline.payload()
    payload["scene_fingerprint"] = "0" * 64
    with pytest.raises(RecipeError, match="does not match this session"):
        backend.validate({"recipe": payload})

    report = backend.validate(
        {"recipe": payload, "allow_fingerprint_mismatch": True}
    )
    assert report["fingerprint_match"] is False
    assert report["fingerprint_override"] is True


def test_loading_a_mismatched_recipe_reports_it_without_applying(
    backend, baseline, tmp_path
):
    payload = baseline.payload()
    payload["scene_fingerprint"] = "1" * 64
    payload["recipe_id"] = "stale"
    report = backend.import_recipe({"document": json.dumps(payload)})
    assert report["applied"] is False
    assert report["fingerprint_match"] is False
    assert "not applied" in report["message"]


def test_saving_stamps_the_current_fingerprint_and_round_trips(backend, baseline):
    payload = baseline.payload()
    payload["recipe_id"] = "saved-one"
    saved = backend.save_recipe({"recipe": payload})
    assert saved["saved"]["recipe_id"] == "saved-one"
    assert saved["recipe"]["scene_fingerprint"] == backend.scene_fingerprint

    reloaded = backend.load_recipe({"recipe_id": "saved-one"})
    assert reloaded["applied"] is True
    assert reloaded["fingerprint_match"] is True
    assert reloaded["recipe_hash"] == saved["recipe_hash"]
    assert reloaded["recipe"]["checks"] == saved["recipe"]["checks"]


def test_scene_fingerprint_matches_the_shared_project_helper(backend, assets):
    assert backend.scene_fingerprint == scene_spec_fingerprint(assets.spec)


def test_importing_a_recipe_that_cannot_resolve_reports_the_reason(
    backend, baseline
):
    payload = baseline.payload()
    payload["scene_fingerprint"] = backend.scene_fingerprint
    payload["checks"] = [
        {
            **payload["checks"][0],
            "pose_mode": "absolute",
            "payload_source": "none",
            "template": None,
            "reference_frame": "GhostFrame",
        }
    ]
    report = backend.import_recipe({"document": payload})
    assert report["applied"] is False
    assert "unknown reference frame" in report["resolve_error"]


def test_import_rejects_a_non_document_request(backend):
    with pytest.raises(RecipeError, match="document must be"):
        backend.import_recipe({"document": 17})


def test_recipe_listing_reports_the_study_directory_name(backend, baseline):
    payload = baseline.payload()
    payload["recipe_id"] = "listed"
    backend.save_recipe({"recipe": payload})
    listing = backend.list_recipes({})
    assert [item["recipe_id"] for item in listing["saved_recipes"]] == ["listed"]
    assert listing["scene_fingerprint"] == backend.scene_fingerprint


# ================================================== actual evaluation jobs


def _run_to_completion(backend: StudyEvaluationBackend, job_id: str) -> dict:
    # The injected thread factory runs the worker synchronously, so the job is
    # already finished when start_* returns.
    status = backend.job_status({"job_id": job_id})
    assert status["job"]["state"] != "running"
    return status["job"]


class _InlineThread:
    """Run a study worker synchronously so tests stay deterministic."""

    def __init__(self, target, args=(), name="", daemon=True) -> None:
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


@pytest.fixture
def inline_backend(assets, tmp_path) -> StudyEvaluationBackend:
    return StudyEvaluationBackend(
        assets.spec,
        assets.key,
        assets.ur,
        assets.sr,
        None,
        profiles={"quick": QUICK},
        default_profile="quick",
        store=StudyRecipeStore(tmp_path),
        thread_factory=_InlineThread,
    )


def test_evaluating_one_generated_check_matches_the_default_evaluator(
    inline_backend, assets, baseline
):
    """The authored zero-offset check must reproduce the original result.

    This is the contract's core parity requirement: the study must not be a
    second, differently behaving evaluator.
    """

    from decanting_workspace.collision import sr_concurrent_exclusion_box
    from decanting_workspace.evaluation import evaluate_task_sequence
    from decanting_workspace.models import SceneState
    from decanting_workspace.robots import configure_sr_j3_stroke
    from decanting_workspace.scene import materialize_scene
    from decanting_workspace.workflow import (
        CoordinationMode,
        ProcessScenario,
        build_ur20_task_sequence,
    )

    key = assets.key
    state = SceneState(
        lift_height_m=key.lift_height_m,
        sku=key.sku,
        corner_samples=(key.corner,),
        tote_present=True,
        tote_long_axis_offset_m=key.tote_offset_m,
        sr_j3_stroke_m=key.sr_j3_stroke_m,
        clearance_m=key.clearance_m,
    )
    snapshot = materialize_scene(
        assets.spec, state, ur_base=key.ur_base, sr_base=key.sr_base
    )
    scenario = ProcessScenario(state, key.corner, key.coordination_mode)
    keepout = None
    if key.coordination_mode is CoordinationMode.SIMULTANEOUS:
        configure_sr_j3_stroke(assets.sr, key.sr_j3_stroke_m)
        keepout = sr_concurrent_exclusion_box(
            assets.spec, snapshot, assets.sr, margin_m=0.0
        )
    steps = build_ur20_task_sequence(
        assets.spec,
        snapshot,
        scenario,
        assets.ur,
        sr_keepout=keepout,
        approach_distance_m=0.15,
    )
    subset = tuple(step for step in steps if step.index == 1)
    default = evaluate_task_sequence(
        assets.spec, snapshot, scenario, assets.ur, subset, options=QUICK
    )
    expected = {
        check.name: check.status
        for step in default.steps
        for check in step.checks
    }

    wanted = [
        check.check_id
        for check in baseline.checks
        if check.template is not None and check.template.step_index == 1
    ]
    job = inline_backend.start_evaluation(
        {"recipe": baseline.payload(), "check_ids": wanted}
    )["job"]
    result = _run_to_completion(inline_backend, job["job_id"])["result"]

    actual = {
        entry["template"].split(":", 2)[2]: entry["status"]
        for entry in result["checks"]
    }
    assert actual == expected
    assert set(actual) == {"pallet_box_approach", "pallet_box_lift"}


def test_evaluation_result_is_strict_json_and_carries_the_recipe_hash(
    inline_backend, baseline
):
    wanted = [baseline.checks[0].check_id]
    job = inline_backend.start_evaluation(
        {"recipe": baseline.payload(), "check_ids": wanted}
    )["job"]
    result = _run_to_completion(inline_backend, job["job_id"])["result"]
    json.dumps(result, allow_nan=False)
    assert result["recipe_hash"] == baseline.recipe_hash
    assert result["kind"] == "evaluate"
    assert result["profile"] == "quick"
    entry = result["checks"][0]
    assert entry["samples"]
    assert entry["collision_pair_count"] > 0
    for sample in entry["samples"]:
        assert sample["minimum_distance_m"] is None or math.isfinite(
            sample["minimum_distance_m"]
        )
        assert sample["normalized_condition_number"] is None or math.isfinite(
            sample["normalized_condition_number"]
        )


def test_a_partial_run_is_never_reported_as_feasible(inline_backend, baseline):
    wanted = [baseline.checks[0].check_id]
    job = inline_backend.start_evaluation(
        {"recipe": baseline.payload(), "check_ids": wanted}
    )["job"]
    result = _run_to_completion(inline_backend, job["job_id"])["result"]
    assert result["certification"] == "none"
    assert result["status"] != "study_feasible"
    assert result["complete"] is False
    assert len(result["unevaluated_hard_check_ids"]) > 0
    # However the run ends, a partial study must say so in its own headline.
    assert "not calculated" in result["headline"]
    # The aggregate follows the one evaluated check exactly: an actual
    # collision witness settles it as infeasible, while an exhausted IK search
    # or an uncalculated check can only ever be UNKNOWN.
    outcome = result["checks"][0]["outcome"]
    assert result["status"] == (
        "study_infeasible" if outcome == "collision" else "unknown"
    )


def test_an_empty_hard_set_is_never_a_success(inline_backend, baseline):
    payload = baseline.payload()
    payload["checks"] = [
        {**payload["checks"][0], "criticality": "soft"}
    ]
    job = inline_backend.start_evaluation({"recipe": payload})["job"]
    result = _run_to_completion(inline_backend, job["job_id"])["result"]
    assert result["status"] == "no_hard_checks"
    assert result["certification"] == "none"
    assert "never a success" in result["headline"]


def test_all_checks_disabled_cannot_be_started(inline_backend, baseline):
    payload = baseline.payload()
    for check in payload["checks"]:
        check["enabled"] = False
    with pytest.raises(RecipeError, match="no check selected"):
        inline_backend.start_evaluation({"recipe": payload})


def test_excluding_a_target_changes_the_actual_collision_pair_count(
    inline_backend, baseline
):
    payload = _two_scenario_payload(baseline, excluded_targets=["pallet"])
    first = inline_backend.start_evaluation({"recipe": payload})["job"]
    baseline_pairs = _run_to_completion(inline_backend, first["job_id"])["result"][
        "checks"
    ][0]["collision_pair_count"]

    payload["checks"][0]["scenario_id"] = "probe"
    second = inline_backend.start_evaluation({"recipe": payload})["job"]
    probe_result = _run_to_completion(inline_backend, second["job_id"])["result"]
    probe_pairs = probe_result["checks"][0]["collision_pair_count"]

    assert probe_pairs < baseline_pairs
    assert probe_result["checks"][0]["excluded_targets_applied"] == ["pallet"]


def test_a_weakened_contract_can_never_be_certified(inline_backend, baseline):
    payload = baseline.payload()
    payload["scenarios"] = [
        {
            "scenario_id": "default",
            "label": "weakened default",
            "excluded_targets": ["pallet"],
            "extra_allowed_contacts": [],
            "include_self_collision": True,
            "include_floor": True,
        }
    ]
    payload["checks"] = [
        {**payload["checks"][0], "criticality": "hard"}
    ]
    job = inline_backend.start_evaluation({"recipe": payload})["job"]
    result = _run_to_completion(inline_backend, job["job_id"])["result"]
    assert result["certification"] != "study_feasible"
    assert result["diagnostic"]["weakened_collision_contract"] is True
    assert result["diagnostic"]["excluded_targets"] == ["pallet"]


def test_omitting_generated_hard_checks_is_reported_as_partial_coverage(
    inline_backend, baseline
):
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:1]
    job = inline_backend.start_evaluation({"recipe": payload})["job"]
    result = _run_to_completion(inline_backend, job["job_id"])["result"]
    assert result["coverage"]["covers_generated_hard_set"] is False
    assert result["coverage"]["omitted_generated_hard_checks"]
    assert result["certification"] != "study_feasible"


def test_a_cancelled_job_reports_cancelled_and_never_feasible(
    inline_backend, baseline
):
    class _CancellingThread(_InlineThread):
        def start(self) -> None:
            job = self._args[0]
            job.cancel_event.set()
            self._target(*self._args)

    inline_backend.thread_factory = _CancellingThread
    job = inline_backend.start_evaluation({"recipe": baseline.payload()})["job"]
    finished = _run_to_completion(inline_backend, job["job_id"])
    assert finished["state"] == "cancelled"
    result = finished["result"]
    assert result["cancelled"] is True
    assert result["status"] == "unknown"
    assert result["certification"] == "none"
    assert result["checks"] == []


def test_a_second_job_is_refused_while_one_is_active(assets, tmp_path, baseline):
    gate = threading.Event()
    release = threading.Event()

    class _BlockingThread:
        def __init__(self, target, args=(), name="", daemon=True) -> None:
            self._target = target
            self._args = args

        def start(self) -> None:
            def run() -> None:
                gate.set()
                release.wait(10.0)
                self._target(*self._args)

            threading.Thread(target=run, daemon=True).start()

    backend = StudyEvaluationBackend(
        assets.spec,
        assets.key,
        assets.ur,
        assets.sr,
        None,
        profiles={"quick": QUICK},
        default_profile="quick",
        store=StudyRecipeStore(tmp_path),
        thread_factory=_BlockingThread,
    )
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:1]
    backend.start_evaluation({"recipe": payload})
    gate.wait(10.0)
    try:
        with pytest.raises(RecipeError, match="already running"):
            backend.start_evaluation({"recipe": payload})
        assert backend.job_status({})["busy"] is True
    finally:
        release.set()


def test_cancel_requires_a_known_job(backend):
    with pytest.raises(RecipeError, match="no such study job"):
        backend.cancel_job({"job_id": "nope"})
    with pytest.raises(RecipeError, match="no such study job"):
        backend.cancel_job({})


def test_job_status_rejects_an_unknown_identifier(backend):
    with pytest.raises(RecipeError, match="unknown job"):
        backend.job_status({"job_id": "missing"})


def test_unknown_profile_is_rejected(backend, baseline):
    payload = baseline.payload()
    payload["profile"] = "turbo"
    with pytest.raises(RecipeError, match="unknown evaluation profile"):
        backend.start_evaluation({"recipe": payload})


def test_show_configuration_validates_the_joint_vector_length(backend):
    with pytest.raises(RecipeError, match="must contain"):
        backend.show_configuration({"q": [0.0, 0.0]})
    with pytest.raises(RecipeError, match="must be an array"):
        backend.show_configuration({"q": "nope"})
    with pytest.raises(RecipeError, match="finite"):
        backend.show_configuration({"q": [float("nan")] * 13})


def test_show_configuration_without_a_viewer_says_so(backend, assets):
    nq = int(assets.ur.model.nq)
    result = backend.show_configuration({"q": [0.0] * nq})
    assert result["displayed"] is False


# ================================================== parameters and request


def test_missing_and_unknown_parameters_are_rejected(backend, baseline):
    with pytest.raises(RecipeError, match="missing the effective case keys"):
        backend.catalog({"parameters": {"ur_x": 0.0}})
    payload = dict(baseline.parameters)
    payload["surprise"] = 1
    with pytest.raises(RecipeError, match="unknown case keys"):
        backend.catalog({"parameters": payload})


def test_nonfinite_and_unknown_categorical_parameters_are_rejected(
    backend, baseline
):
    payload = dict(baseline.parameters)
    payload["ur_x"] = float("inf")
    with pytest.raises(RecipeError, match="finite"):
        backend.catalog({"parameters": payload})

    payload = dict(baseline.parameters)
    payload["sku"] = "nonexistent"
    with pytest.raises(RecipeError, match="unknown box SKU"):
        backend.catalog({"parameters": payload})

    payload = dict(baseline.parameters)
    payload["corner"] = "middle"
    with pytest.raises(RecipeError, match="unknown pallet corner"):
        backend.catalog({"parameters": payload})

    payload = dict(baseline.parameters)
    payload["mode"] = "telepathic"
    with pytest.raises(RecipeError, match="unknown coordination mode"):
        backend.catalog({"parameters": payload})


def test_changed_parameters_regenerate_the_workflow_poses(backend, baseline):
    payload = baseline.payload()
    # Step 5 grasps the nearest vertical box face, so its generated pose is a
    # function of the candidate UR20 base.
    payload["checks"] = [
        item
        for item in payload["checks"]
        if item["template"]["check_name"] == "opened_box_front_approach"
    ]
    assert len(payload["checks"]) == 1
    original = backend.validate({"recipe": payload})["resolved_checks"][0]

    moved = dict(payload["parameters"])
    moved["ur_x"] = float(moved["ur_x"]) + 3.0
    shifted = backend.validate({"recipe": {**payload, "parameters": moved}})[
        "resolved_checks"
    ][0]
    assert shifted["world_start"] != original["world_start"]


def test_a_base_independent_generated_pose_stays_fixed(backend, baseline):
    payload = baseline.payload()
    # The pallet pick target is defined by the carton's top face, not by the
    # candidate base, so moving the base must not silently move it.
    payload["checks"] = [
        item
        for item in payload["checks"]
        if item["template"]["check_name"] == "pallet_box_approach"
    ]
    original = backend.validate({"recipe": payload})["resolved_checks"][0]
    moved = dict(payload["parameters"])
    moved["ur_y"] = float(moved["ur_y"]) + 0.10
    shifted = backend.validate({"recipe": {**payload, "parameters": moved}})[
        "resolved_checks"
    ][0]
    assert shifted["world_start"] == original["world_start"]


def test_a_request_may_not_override_the_recipes_effective_case(backend, baseline):
    # The case is hashed into recipe_hash, so computing one case while
    # reporting another recipe's hash would make the result unattributable.
    payload = baseline.payload()
    conflicting = dict(payload["parameters"])
    conflicting["ur_x"] = float(conflicting["ur_x"]) + 1.0
    with pytest.raises(RecipeError, match="conflict with recipe"):
        backend.validate({"recipe": payload, "parameters": conflicting})
    # An identical echo is accepted, because it names the same case.
    echoed = backend.validate(
        {"recipe": payload, "parameters": dict(payload["parameters"])}
    )
    assert echoed["recipe_hash"] == baseline.recipe_hash
    assert echoed["effective"]["parameters"]["ur_x"] == pytest.approx(
        float(payload["parameters"]["ur_x"])
    )


def test_a_request_without_a_recipe_is_rejected(backend):
    with pytest.raises(RecipeError, match="recipe is required"):
        backend.validate({})


def test_selecting_an_unknown_check_id_is_rejected(backend, baseline):
    with pytest.raises(RecipeError, match="unknown check ids"):
        backend.start_evaluation(
            {"recipe": baseline.payload(), "check_ids": ["ghost"]}
        )


# ===================================================== preview and viewer


class _RecordingViewer:
    def __init__(self) -> None:
        self.markers = None
        self.segments = None
        self.highlight = None
        self.cleared = 0
        self.scene_updates: list[tuple[str, ...]] = []
        self.payloads: list[tuple[bool, list[str]]] = []
        self.framed = 0

    def update_scene(self, spec, snapshot, *, hide_object_names=(), **kwargs) -> None:
        self.scene_updates.append(tuple(hide_object_names))

    def render_study_payload(self, objects=(), *, evaluated: bool) -> int:
        drawn = list(objects)
        self.payloads.append((evaluated, drawn))
        return len(drawn)

    def frame_scene(self, spec, snapshot, **kwargs) -> dict:
        self.framed += 1
        return {
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "center_z_m": 0.0,
            "radius_m": 1.0,
            "distance_m": 2.0,
        }

    def render_study_markers(self, markers=(), segments=(), **kwargs) -> int:
        self.markers = list(markers)
        self.segments = list(segments)
        return len(self.markers) + len(self.segments)

    def render_study_target_highlight(self, boxes=(), meshes=(), **kwargs) -> int:
        self.highlight = ([box.name for box in boxes], [m.name for m in meshes])
        return len(self.highlight[0]) + len(self.highlight[1])

    def clear_study_overlays(self) -> None:
        self.cleared += 1


def test_preview_draws_two_markers_and_one_segment_per_check(
    assets, tmp_path, baseline
):
    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:2]
    result = backend.preview({"recipe": payload})

    assert len(viewer.markers) == 4
    assert len(viewer.segments) == 2
    assert result["previewed_checks"] == [
        item["check_id"] for item in payload["checks"]
    ]
    for _, transform, _ in viewer.markers:
        assert np.all(np.isfinite(transform))


def test_preview_highlights_named_targets_and_reports_missing_ones(
    assets, tmp_path, baseline
):
    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:1]
    result = backend.preview(
        {
            "recipe": payload,
            "highlight_targets": ["pallet", "no_such_target"],
            "show_markers": False,
        }
    )
    assert viewer.highlight[0] == ["pallet"]
    assert result["missing_highlight_targets"] == ["no_such_target"]
    assert viewer.markers == []


def test_clear_preview_removes_the_overlay(assets, tmp_path):
    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    backend.clear_preview({})
    assert viewer.cleared == 1


def test_preview_rejects_an_unknown_check_id(assets, tmp_path, baseline):
    backend = _backend(assets, tmp_path, viewer=_RecordingViewer())
    with pytest.raises(RecipeError, match="unknown check ids"):
        backend.preview({"recipe": baseline.payload(), "check_ids": ["ghost"]})


# ============================================================ scout bounds


def test_scout_grid_enumerates_the_declared_axes(assets):
    settings = _scout_settings(
        {
            "mode": "grid",
            "x_range_m": [-1.4, -1.2],
            "counts": [3, 1, 1, 1],
            "max_candidates": 8,
        },
        assets.key,
    )
    candidates = _scout_candidates(settings)
    assert [round(pose.x_m, 3) for pose in candidates] == [-1.4, -1.3, -1.2]
    assert all(pose.y_m == assets.key.ur_base.y_m for pose in candidates)


def test_scout_random_mode_is_seed_deterministic(assets):
    request = {
        "mode": "random",
        "x_range_m": [-1.5, -1.1],
        "yaw_range_deg": [80.0, 100.0],
        "counts": [5, 1, 1, 1],
        "max_candidates": 5,
        "seed": 7,
    }
    first = _scout_candidates(_scout_settings(dict(request), assets.key))
    second = _scout_candidates(_scout_settings(dict(request), assets.key))
    assert [pose.x_m for pose in first] == [pose.x_m for pose in second]
    assert len(first) == 5
    request["seed"] = 8
    third = _scout_candidates(_scout_settings(request, assets.key))
    assert [pose.x_m for pose in third] != [pose.x_m for pose in first]


def test_scout_rejects_bad_bounds_counts_and_limits(assets):
    key = assets.key
    with pytest.raises(RecipeError, match="min must not exceed max"):
        _scout_settings({"x_range_m": [1.0, -1.0]}, key)
    with pytest.raises(RecipeError, match="above the floor"):
        _scout_settings({"z_range_m": [0.0, 1.0]}, key)
    with pytest.raises(RecipeError, match="finite"):
        _scout_settings({"x_range_m": [float("nan"), 1.0]}, key)
    with pytest.raises(RecipeError, match="exactly two values"):
        _scout_settings({"x_range_m": [1.0]}, key)
    with pytest.raises(RecipeError, match="counts"):
        _scout_settings({"counts": [0, 1, 1, 1]}, key)
    with pytest.raises(RecipeError, match="counts"):
        _scout_settings({"counts": [1, 1, 1]}, key)
    with pytest.raises(RecipeError, match="max_candidates"):
        _scout_settings({"max_candidates": MAX_SCOUT_CANDIDATES + 1}, key)
    with pytest.raises(RecipeError, match="seed"):
        _scout_settings({"seed": -1}, key)
    with pytest.raises(RecipeError, match="unknown keys"):
        _scout_settings({"nope": 1}, key)
    with pytest.raises(RecipeError, match="above the"):
        _scout_settings(
            {"counts": [8, 8, 1, 1], "max_candidates": 4, "mode": "grid"}, key
        )


def test_scout_needs_at_least_one_enabled_hard_check(backend, baseline):
    payload = baseline.payload()
    for check in payload["checks"]:
        check["criticality"] = "soft"
    with pytest.raises(RecipeError, match="at least one enabled HARD check"):
        backend.start_scout({"recipe": payload, "scout": {}})


def test_scout_ranks_only_fully_feasible_candidates(inline_backend, baseline):
    payload = baseline.payload()
    payload["checks"] = [
        item
        for item in payload["checks"]
        if item["template"]["step_index"] == 1
    ]
    job = inline_backend.start_scout(
        {
            "recipe": payload,
            "scout": {
                "mode": "grid",
                "x_range_m": [
                    float(baseline.parameters["ur_x"]) - 0.6,
                    float(baseline.parameters["ur_x"]),
                ],
                "counts": [2, 1, 1, 1],
                "max_candidates": 2,
            },
        }
    )["job"]
    result = _run_to_completion(inline_backend, job["job_id"])["result"]
    json.dumps(result, allow_nan=False)
    assert result["kind"] == "scout"
    assert len(result["candidates"]) == 2
    assert "not a cell layout optimization" in result["interpretation"]
    for row in result["candidates"]:
        if row["status"] != "feasible":
            # A candidate that failed before every metric was collected must
            # not carry a comparable score.
            assert row["worst_normalized_min_singular_value"] == 0.0
    assert all(row["status"] == "feasible" for row in result["ranking"])
    scores = [
        row["worst_normalized_min_singular_value"] for row in result["ranking"]
    ]
    assert scores == sorted(scores, reverse=True)


def test_scout_candidate_outside_the_floor_is_rejected_not_crashed(
    inline_backend, baseline
):
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:1]
    job = inline_backend.start_scout(
        {
            "recipe": payload,
            "scout": {
                "mode": "grid",
                # A base far outside the installation region must be refused
                # by the original installation gate, before any IK cost, and
                # it must never reach any ranking.
                "x_range_m": [40.0, 40.0],
                "counts": [1, 1, 1, 1],
                "max_candidates": 1,
            },
        }
    )["job"]
    result = _run_to_completion(inline_backend, job["job_id"])["result"]
    row = result["candidates"][0]
    assert row["status"] == "installation_invalid"
    assert row["installation"]["valid"] is False
    assert row["installation"]["issues"]
    assert row["evaluated_checks"] == 0
    assert row["worst_normalized_min_singular_value"] == 0.0
    assert result["ranking"] == []
    assert result["ranking_strict"] == []
    assert result["ranking_diagnostic"] == []
    json.dumps(result, allow_nan=False)


# ====================================== effective case, gates and overlays
#
# Everything below covers behaviour the study gained after its first
# implementation: the recipe owns its own effective case, the original
# installation check stays independent of any collision scenario, an
# exhausted IK search is distinguished from an actual collision witness, and
# the 3-D overlay has to agree with the phase it claims to show.


def _pose_matrix(payload: dict) -> np.ndarray:
    return Pose6D(**payload).matrix()


def _single_check_payload(baseline: StudyRecipe, check_name: str) -> dict:
    payload = baseline.payload()
    payload["checks"] = [
        item
        for item in payload["checks"]
        if item["template"] is not None
        and item["template"]["check_name"] == check_name
    ]
    assert len(payload["checks"]) == 1, check_name
    return payload


def _moved_case(payload: dict, **changes) -> dict:
    return {**payload, "parameters": {**payload["parameters"], **changes}}


class _RecordingPlayback:
    """Stand in for the MeshCat playback so no robot mesh has to load."""

    def __init__(self) -> None:
        self.displayed: list[np.ndarray] = []

    def display_configuration(self, q) -> None:
        self.displayed.append(np.asarray(q, dtype=float))

    def hide_ellipsoid(self) -> None:
        return None


# ------------------------------------------------- result status ladder


def test_check_outcome_separates_a_collision_witness_from_an_unresolved_search():
    # Only an actual collision at an attempted pose or along an attempted
    # segment is evidence of infeasibility.  An exhausted deterministic IK
    # search proves nothing, so it must never map onto "collision".
    assert check_outcome("success") == "success"
    assert check_outcome("solution_in_collision") == "collision"
    assert check_outcome("joint_segment_collision") == "collision"
    assert check_outcome("ik_not_found") == "unresolved"
    # An unrecognised status is unresolved, never a success and never a
    # collision claim the evaluator did not actually make.
    assert check_outcome("some_future_status") == "unresolved"


def _ladder_result(recipe, context, entries):
    return _study_result_payload(
        recipe=recipe,
        context=context,
        entries=entries,
        coverage={
            "omitted_generated_hard_checks": [],
            "downgraded_generated_hard_checks": [],
            "skipped_process_steps": [],
        },
        diagnostic={"weakened_collision_contract": False, "reasons": []},
        evaluated_ids=[str(entry["check_id"]) for entry in entries],
        selected_ids=None,
        cancelled=False,
        fingerprint_override=False,
        elapsed_seconds=0.1,
        options=QUICK,
    )


@pytest.fixture
def ladder(backend, baseline):
    """One HARD check plus the context of its own effective case."""

    payload = _single_check_payload(baseline, "pallet_box_approach")
    recipe = StudyRecipe.from_mapping(payload)
    context = backend._workflow_context(
        recipe.parameters, approach_distance_m=recipe.approach_distance_m
    )
    return recipe, context, recipe.checks[0].check_id


def test_every_hard_check_succeeding_at_a_valid_base_is_study_feasible(ladder):
    recipe, context, check_id = ladder
    assert context.placement.valid is True
    result = _ladder_result(
        recipe,
        context,
        [{"check_id": check_id, "success": True, "outcome": "success"}],
    )
    assert result["status"] == "study_feasible"
    assert result["certification"] == "study_feasible"


def test_a_collision_witness_is_infeasible_but_an_exhausted_search_is_unknown(
    ladder,
):
    recipe, context, check_id = ladder
    collided = _ladder_result(
        recipe,
        context,
        [
            {
                "check_id": check_id,
                "success": False,
                "outcome": "collision",
            }
        ],
    )
    assert collided["status"] == "study_infeasible"
    assert collided["collision_hard_check_ids"] == [check_id]
    assert collided["unresolved_hard_check_ids"] == []

    unresolved = _ladder_result(
        recipe,
        context,
        [
            {
                "check_id": check_id,
                "success": False,
                "outcome": "unresolved",
            }
        ],
    )
    assert unresolved["status"] == "unknown"
    assert unresolved["certification"] == "none"
    assert unresolved["unresolved_hard_check_ids"] == [check_id]
    assert unresolved["collision_hard_check_ids"] == []
    assert "proof that no solution exists" in unresolved["headline"]


def test_an_installation_invalid_base_can_never_be_certified(backend, baseline):
    # Every HARD check passing is not enough: the original installation gate
    # is a separate, geometric precondition and a collision scenario cannot
    # waive it.
    payload = _moved_case(
        _single_check_payload(baseline, "pallet_box_approach"), ur_x=40.0
    )
    recipe = StudyRecipe.from_mapping(payload)
    context = backend._workflow_context(
        recipe.parameters, approach_distance_m=recipe.approach_distance_m
    )
    assert context.placement.valid is False
    result = _ladder_result(
        recipe,
        context,
        [
            {
                "check_id": recipe.checks[0].check_id,
                "success": True,
                "outcome": "success",
            }
        ],
    )
    assert result["status"] == "installation_invalid"
    assert result["certification"] == "none"
    assert result["installation"]["valid"] is False
    assert result["installation"]["issues"]
    assert "installation" in result["headline"]


def test_a_fingerprint_override_can_never_be_certified(ladder):
    recipe, context, check_id = ladder
    result = _study_result_payload(
        recipe=recipe,
        context=context,
        entries=[{"check_id": check_id, "success": True, "outcome": "success"}],
        coverage={
            "omitted_generated_hard_checks": [],
            "downgraded_generated_hard_checks": [],
            "skipped_process_steps": [],
        },
        diagnostic={"weakened_collision_contract": False, "reasons": []},
        evaluated_ids=[check_id],
        selected_ids=None,
        cancelled=False,
        fingerprint_override=True,
        elapsed_seconds=0.1,
        options=QUICK,
    )
    assert result["status"] == "study_feasible_diagnostic"
    assert result["certification"] == "study_feasible_diagnostic"


# ------------------------------------------- installation is independent


def test_installation_validity_ignores_every_collision_scenario(
    backend, baseline
):
    # Excluding targets, exempting contacts and switching the floor and self
    # collision off must not change the installation verdict, because that
    # verdict comes from scene geometry alone.
    payload = _single_check_payload(baseline, "pallet_box_approach")
    strict = backend.validate({"recipe": payload})
    assert strict["installation"]["valid"] is True

    weakened = _two_scenario_payload(
        baseline,
        excluded_targets=["pallet", "ur20_pedestal"],
        extra_allowed_contacts=[["suction_pad", "pallet"]],
        include_floor=False,
        include_self_collision=False,
    )
    weakened["checks"][0]["scenario_id"] = "probe"
    report = backend.validate({"recipe": weakened})
    assert report["installation"] == strict["installation"]
    # The weakening is still reported, just not as an installation verdict.
    assert report["diagnostic"]["weakened_collision_contract"] is True

    # An out-of-region base is refused even with everything excluded.
    moved = _moved_case(weakened, ur_x=40.0)
    assert backend.validate({"recipe": moved})["installation"]["valid"] is False


def test_an_unknown_excluded_collision_target_is_rejected(backend, baseline):
    # A misspelled exclusion must not read as an applied diagnostic that
    # happened to change nothing.
    payload = _two_scenario_payload(
        baseline, excluded_targets=["palet", "pallet"]
    )
    payload["checks"][0]["scenario_id"] = "probe"
    with pytest.raises(RecipeError, match="do not exist in this scene"):
        backend.validate({"recipe": payload})
    # The correctly spelled name on its own is accepted.
    payload["scenarios"][1]["excluded_targets"] = ["pallet"]
    assert backend.validate({"recipe": payload})["resolved_checks"][0][
        "excluded_targets_applied"
    ] == ["pallet"]


def test_downgrading_a_generated_hard_check_to_soft_leaves_it_uncovered(
    backend, baseline
):
    # A HARD check that is kept but downgraded to SOFT no longer gates the
    # result, so it must count as uncovered exactly like a deleted one.
    payload = baseline.payload()
    target = next(
        item for item in payload["checks"] if item["criticality"] == "hard"
    )
    token = f"{target['template']['step_index']}:{target['template']['check_name']}"
    payload["checks"] = [
        {**item, "criticality": "soft"} if item is target else item
        for item in payload["checks"]
    ]
    report = backend.validate({"recipe": payload})
    coverage = report["coverage"]
    assert token in coverage["downgraded_generated_hard_checks"]
    assert token in coverage["omitted_generated_hard_checks"]
    assert coverage["covers_generated_hard_set"] is False
    assert any(
        "downgraded to SOFT" in reason
        for reason in report["diagnostic"]["reasons"]
    )


# --------------------------------------- the effective case round-trips


def test_catalog_honours_an_approach_distance_override(backend, baseline):
    default = backend.catalog({})
    assert default["approach_distance_m"] == default["startup_approach_distance_m"]

    further = backend.catalog(
        {
            "parameters": dict(baseline.parameters),
            "approach_distance_m": default["approach_distance_m"] + 0.05,
        }
    )
    assert further["approach_distance_m"] == pytest.approx(
        default["approach_distance_m"] + 0.05
    )
    # The startup value is reported separately, so the UI can always show what
    # the session was launched with.
    assert further["startup_approach_distance_m"] == pytest.approx(
        default["startup_approach_distance_m"]
    )
    assert further["baseline_recipe"]["approach_distance_m"] == pytest.approx(
        default["approach_distance_m"] + 0.05
    )
    # A longer approach moves the generated approach poses, so the baseline
    # recipe is genuinely a different study.
    assert (
        further["baseline_recipe"]["parameters"]
        == default["baseline_recipe"]["parameters"]
    )


def test_saving_round_trips_the_effective_case_and_approach_distance(
    backend, baseline
):
    payload = _moved_case(baseline.payload(), ur_x=-1.25, sku="430060")
    payload["recipe_id"] = "case-roundtrip"
    payload["approach_distance_m"] = 0.123
    saved = backend.save_recipe({"recipe": payload})
    reloaded = backend.load_recipe({"recipe_id": "case-roundtrip"})

    assert reloaded["recipe"]["parameters"] == payload["parameters"]
    assert reloaded["recipe"]["approach_distance_m"] == pytest.approx(0.123)
    assert reloaded["recipe_hash"] == saved["recipe_hash"]
    # The hash follows the case, so the stored result can never be read
    # against a different one.
    assert saved["recipe_hash"] != baseline.recipe_hash

    report = backend.validate({"recipe": reloaded["recipe"]})
    assert report["effective"]["approach_distance_m"] == pytest.approx(0.123)
    assert report["effective"]["parameters"]["sku"] == "430060"
    assert report["effective"]["parameters"]["ur_x"] == pytest.approx(-1.25)


# ------------------------------------------------- camera and overlays


def test_frame_cell_recentres_the_shared_camera(assets, tmp_path, baseline):
    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    result = backend.frame_cell({"parameters": dict(baseline.parameters)})
    assert result["framed"] is True
    assert viewer.framed == 1
    assert set(result["camera"]) >= {"center_x_m", "center_y_m", "distance_m"}
    assert result["parameters"]["ur_x"] == pytest.approx(
        float(baseline.parameters["ur_x"])
    )
    # Framing is camera-only: it must not redraw the scene.
    assert viewer.scene_updates == []


def test_frame_cell_without_a_viewer_says_so(backend):
    result = backend.frame_cell({})
    assert result["framed"] is False


def test_frame_scene_centres_on_the_cell_not_the_world_origin(assets):
    from decanting_workspace.models import SceneState
    from decanting_workspace.scene import materialize_scene
    from decanting_workspace.viewer import _scene_bounds

    key = assets.key
    snapshot = materialize_scene(
        assets.spec,
        SceneState(
            lift_height_m=key.lift_height_m,
            sku=key.sku,
            corner_samples=(key.corner,),
            tote_long_axis_offset_m=key.tote_offset_m,
            sr_j3_stroke_m=key.sr_j3_stroke_m,
        ),
        ur_base=key.ur_base,
        sr_base=key.sr_base,
    )
    center, radius = _scene_bounds(assets.spec, snapshot)
    assert np.all(np.isfinite(center)) and math.isfinite(radius)
    assert radius > 0.0
    # The cell sits metres away from the World origin, which is exactly why a
    # default camera shows empty floor.
    assert float(np.linalg.norm(center)) > 1.0
    # Every rendered box has to fall inside the reported bounding sphere.
    for box in snapshot.boxes:
        offset = np.asarray(box.center_m, dtype=float) - center
        assert float(np.linalg.norm(offset)) <= radius + 1e-6


def test_preview_puts_the_payload_on_the_authored_tcp_and_hides_the_nominal_copy(
    assets, tmp_path, baseline
):
    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    payload = _single_check_payload(baseline, "pallet_box_lift")
    resolved = backend.validate({"recipe": payload})["resolved_checks"][0]
    carried = next(
        item for item in resolved["payload_objects"] if item["attached_to_tcp"]
    )

    result = backend.preview({"recipe": payload})
    assert result["payload_display"] == "schematic_authored_start"
    assert result["payload_boxes"] >= 1
    assert "not an evaluated joint solution" in result["payload_note"]
    # The carried carton is removed from the nominal scene, so it cannot be
    # shown twice: once in its pallet slot and once on the TCP.
    assert carried["name"] in result["hidden_snapshot_objects"]
    assert viewer.scene_updates == [tuple(result["hidden_snapshot_objects"])]

    evaluated, drawn = viewer.payloads[-1]
    assert evaluated is False
    expected = _pose_matrix(resolved["world_start"]) @ _pose_matrix(
        carried["tcp_relative"]
    )
    actual = next(
        transform for name, _size, transform in drawn if name == carried["name"]
    )
    assert np.allclose(np.asarray(actual, dtype=float), expected, atol=1e-7)


def test_preview_of_several_checks_keeps_the_nominal_scene_and_draws_no_payload(
    assets, tmp_path, baseline
):
    # Several checks have no single collision phase, so showing one check's
    # moving objects for all of them would be a false picture.
    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:3]
    result = backend.preview({"recipe": payload})
    assert result["payload_display"] == "none"
    assert result["payload_boxes"] == 0
    assert result["hidden_snapshot_objects"] == []
    assert viewer.scene_updates == [()]
    assert "nominal scene positions" in result["payload_note"]


def test_preview_frames_the_cell_only_when_asked(assets, tmp_path, baseline):
    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    payload = baseline.payload()
    payload["checks"] = payload["checks"][:1]
    assert backend.preview({"recipe": payload})["camera"] is None
    assert viewer.framed == 0
    framed = backend.preview({"recipe": payload, "frame_cell": True})
    assert framed["camera"] is not None
    assert viewer.framed == 1


def test_show_configuration_places_the_payload_on_the_solved_tcp(
    assets, tmp_path, baseline
):
    from decanting_workspace.kinematics import frame_pose
    from decanting_workspace.robots import floating_configuration

    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    backend._playback = _RecordingPlayback()
    payload = _single_check_payload(baseline, "pallet_box_lift")
    resolved = backend.validate({"recipe": payload})["resolved_checks"][0]
    carried = next(
        item for item in resolved["payload_objects"] if item["attached_to_tcp"]
    )
    check_id = resolved["check_id"]

    q = floating_configuration(
        assets.ur,
        BasePose(
            x_m=float(baseline.parameters["ur_x"]),
            y_m=float(baseline.parameters["ur_y"]),
            z_m=float(baseline.parameters["ur_z"]),
            yaw_deg=float(baseline.parameters["ur_yaw"]),
        ),
        assets.spec.robots["ur20"].nominal_q,
    )
    result = backend.show_configuration(
        {"recipe": payload, "check_id": check_id, "q": [float(v) for v in q]}
    )
    assert result["displayed"] is True
    assert result["payload_display"] == "evaluated_solved_pose"
    assert result["payload_boxes"] >= 1

    evaluated, drawn = viewer.payloads[-1]
    assert evaluated is True
    world_T_tcp = frame_pose(assets.ur, q, "suction_tcp")
    expected = world_T_tcp @ _pose_matrix(carried["tcp_relative"])
    actual = next(
        transform for name, _size, transform in drawn if name == carried["name"]
    )
    assert np.allclose(np.asarray(actual, dtype=float), expected, atol=1e-7)
    # The authored schematic box must not survive next to an evaluated one.
    assert [entry[0] for entry in viewer.payloads] == [True]
    # The nominal copy of the carried carton is hidden here too.
    assert carried["name"] in viewer.scene_updates[-1]


def test_show_configuration_without_a_check_clears_any_schematic_payload(
    assets, tmp_path
):
    viewer = _RecordingViewer()
    backend = _backend(assets, tmp_path, viewer=viewer)
    backend._playback = _RecordingPlayback()
    nq = int(assets.ur.model.nq)
    q = [0.0] * nq
    q[6] = 1.0  # unit quaternion for the free-flyer root
    result = backend.show_configuration({"q": q})
    assert result["displayed"] is True
    assert result["payload_display"] == "none"
    assert viewer.payloads == [(True, [])]
    # No recipe was supplied, so the scene must not be re-materialized.
    assert viewer.scene_updates == []


def test_a_non_reentrant_compute_lock_is_refused(assets, tmp_path):
    # The study re-enters the shared lock, so a plain Lock would deadlock the
    # whole server rather than fail visibly.
    with pytest.raises(ValueError, match="reentrant"):
        StudyEvaluationBackend(
            assets.spec,
            assets.key,
            assets.ur,
            assets.sr,
            None,
            profiles={"quick": QUICK},
            default_profile="quick",
            store=StudyRecipeStore(tmp_path),
            compute_lock=threading.Lock(),
        )
    backend = StudyEvaluationBackend(
        assets.spec,
        assets.key,
        assets.ur,
        assets.sr,
        None,
        profiles={"quick": QUICK},
        default_profile="quick",
        store=StudyRecipeStore(tmp_path),
        compute_lock=threading.RLock(),
    )
    assert backend.catalog({})["schema_version"] >= 1


# ------------------------------- payload-only template contact inheritance


def test_a_payload_only_template_inherits_its_carried_contacts_with_provenance(
    catalog,
):
    # A step such as 6 carries object state but generates no pose check of its
    # own, so its intended contacts have to be recovered from the step that
    # actually grasped the same payload instance.  Without this an authored
    # pouring probe would collision-check its own carried carton against the
    # support it is being poured over.
    template = next(
        item for item in catalog["templates"] if item["token"] == "step_objects:6"
    )
    carried = next(
        name for name in template["object_names"] if name.startswith("box_")
    )
    contacts = {tuple(pair) for pair in template["allowed_contacts"]}
    assert tuple(sorted(("suction_pad", carried))) in contacts
    provenance = template["contact_provenance"]
    assert any("grasp contact" in entry for entry in provenance)
    assert any("same carried payload" in entry for entry in provenance)
    # Every reported pair must be traceable, never anonymous.
    assert len(provenance) >= 1


def test_a_payload_only_template_without_an_attached_object_inherits_nothing(
    catalog,
):
    # Identity matching is what keeps the inheritance narrow: a step whose
    # object state is not TCP-coupled must not pick up another phase's
    # exemptions just because an object shares its name.
    for token in ("step_objects:5", "step_objects:8"):
        template = next(
            item for item in catalog["templates"] if item["token"] == token
        )
        assert template["allowed_contacts"] == [], token
        assert template["contact_provenance"] == [], token


def test_inherited_contacts_reach_the_resolved_phase_of_an_authored_probe(
    backend, baseline, catalog
):
    template = next(
        item for item in catalog["templates"] if item["token"] == "step_objects:6"
    )
    payload = baseline.payload()
    payload["checks"] = [
        {
            **payload["checks"][0],
            "check_id": "pour-probe",
            "criticality": "soft",
            "pose_mode": "absolute",
            "payload_source": "template",
            "template": template["reference"],
            "reference_frame": "world",
            "start_pose": Pose6D(x_m=-1.0, y_m=-7.0, z_m=1.4).payload(),
            "end_pose": Pose6D(x_m=-1.0, y_m=-7.0, z_m=1.4).payload(),
        }
    ]
    resolved = backend.validate({"recipe": payload})["resolved_checks"][0]
    intended = {tuple(pair) for pair in resolved["allowed_contacts"]}
    exempt = {tuple(pair) for pair in resolved["phase_allowed_contacts"]}
    assert intended == {tuple(pair) for pair in template["allowed_contacts"]}
    # What the checker actually exempts must cover the intended set, so the
    # two reported lists can never disagree silently.
    assert intended <= exempt
