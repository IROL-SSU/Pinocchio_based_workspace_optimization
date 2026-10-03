from __future__ import annotations

import json
import math

import numpy as np
import pytest

from decanting_workspace.study_recipe import (
    CASE_PARAMETER_KEYS,
    MAX_RECIPE_BYTES,
    STUDY_SCHEMA_VERSION,
    CollisionScenario,
    Pose6D,
    RecipeError,
    StudyCheck,
    StudyRecipe,
    TemplateRef,
    case_parameters_equal,
    default_collision_scenario,
    recipe_file_name,
    rotation_matrix_zyx,
    validate_case_parameters,
)


# A recipe now names its own effective case, so every fixture supplies the
# full 15-key parameter set the evaluator actually consumes.
CASE = {
    "ur_x": -1.3,
    "ur_y": -7.35,
    "ur_z": 0.85,
    "ur_yaw": 90.0,
    "sr_x": -0.3,
    "sr_y": -7.073643684,
    "sr_z": 0.665,
    "sr_yaw": -90.0,
    "sku": "123591",
    "lift": 0.2,
    "tote": 0.7,
    "stroke": 0.45,
    "clearance": None,
    "corner": "southwest",
    "mode": "sequential",
}


def _recipe(**overrides) -> StudyRecipe:
    base = {
        "recipe_id": "r1",
        "label": "recipe one",
        "parameters": dict(CASE),
        "checks": (
            StudyCheck(
                check_id="c1",
                label="check one",
                template=TemplateRef("pose_check", 1, "pallet_box_approach"),
            ),
        ),
    }
    base.update(overrides)
    return StudyRecipe(**base)


# ------------------------------------------------------------------ poses


def test_rotation_convention_is_rz_ry_rx_and_matches_known_values():
    rotation = rotation_matrix_zyx(0.0, 0.0, 90.0)
    assert np.allclose(rotation @ np.array((1.0, 0.0, 0.0)), (0.0, 1.0, 0.0))

    # Rz(yaw) Ry(pitch) Rx(roll): a +90 deg roll about X then +90 deg yaw.
    composed = rotation_matrix_zyx(90.0, 0.0, 90.0)
    expected = rotation_matrix_zyx(0.0, 0.0, 90.0) @ rotation_matrix_zyx(90.0, 0.0, 0.0)
    assert np.allclose(composed, expected)
    # The opposite order is genuinely different, so the convention matters.
    assert not np.allclose(
        composed,
        rotation_matrix_zyx(90.0, 0.0, 0.0) @ rotation_matrix_zyx(0.0, 0.0, 90.0),
    )


def test_pose_matrix_is_translation_then_rotation_in_the_declared_order():
    pose = Pose6D(x_m=0.1, y_m=0.2, z_m=0.3, yaw_deg=90.0)
    matrix = pose.matrix()
    assert np.allclose(matrix[:3, 3], (0.1, 0.2, 0.3))
    assert np.allclose(matrix[:3, :3], rotation_matrix_zyx(0.0, 0.0, 90.0))
    assert np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_pose_rejects_nonfinite_values(value):
    with pytest.raises(RecipeError, match="finite"):
        Pose6D(x_m=value)


def test_pose_rejects_booleans_and_strings():
    with pytest.raises(RecipeError, match="numeric"):
        Pose6D(x_m=True)
    with pytest.raises(RecipeError, match="numeric"):
        Pose6D.from_mapping({"x_m": "0.1"}, "start")


def test_pose_rejects_absurd_magnitudes():
    with pytest.raises(RecipeError, match="1000 m"):
        Pose6D(x_m=5000.0)
    with pytest.raises(RecipeError, match="36000"):
        Pose6D(yaw_deg=1.0e6)


def test_pose_from_mapping_rejects_unknown_keys():
    with pytest.raises(RecipeError, match="unknown keys"):
        Pose6D.from_mapping({"x_mm": 1.0}, "start")


def test_identity_pose_is_detected():
    assert Pose6D().is_identity
    assert not Pose6D(z_m=1e-9).is_identity


# -------------------------------------------------------------- templates


def test_template_tokens_distinguish_pose_and_payload_sources():
    pose = TemplateRef("pose_check", 5, "opened_box_lift")
    payload = TemplateRef("step_objects", 6)
    assert pose.token == "pose_check:5:opened_box_lift"
    assert payload.token == "step_objects:6"
    assert pose.supplies_pose
    assert not payload.supplies_pose


def test_template_rejects_bad_kind_step_and_missing_name():
    with pytest.raises(RecipeError, match="template kind"):
        TemplateRef("whatever", 1, "x")
    with pytest.raises(RecipeError, match="step_index"):
        TemplateRef("pose_check", 0, "x")
    with pytest.raises(RecipeError, match="check_name"):
        TemplateRef("pose_check", 1, "")
    with pytest.raises(RecipeError, match="step_index must be an integer"):
        TemplateRef("pose_check", 1.5, "x")


# -------------------------------------------------------------- scenarios


def test_default_scenario_is_not_diagnostic():
    scenario = default_collision_scenario()
    assert not scenario.is_diagnostic
    assert scenario.diagnostic_reasons() == ()
    assert scenario.include_floor and scenario.include_self_collision


@pytest.mark.parametrize(
    "overrides",
    [
        {"excluded_targets": ("pallet",)},
        {"extra_allowed_contacts": (("suction_pad", "pallet"),)},
        {"include_floor": False},
        {"include_self_collision": False},
    ],
)
def test_any_weakening_marks_the_scenario_diagnostic(overrides):
    scenario = CollisionScenario(scenario_id="s", label="s", **overrides)
    assert scenario.is_diagnostic
    assert scenario.diagnostic_reasons()


def test_scenario_canonicalizes_and_deduplicates_contact_pairs():
    scenario = CollisionScenario(
        scenario_id="s",
        label="s",
        extra_allowed_contacts=(("b", "a"),),
    )
    assert scenario.extra_allowed_contacts == (("a", "b"),)
    with pytest.raises(RecipeError, match="repeats an allowed contact"):
        CollisionScenario(
            scenario_id="s",
            label="s",
            extra_allowed_contacts=(("a", "b"), ("b", "a")),
        )


def test_scenario_rejects_self_pairs_and_duplicate_exclusions():
    with pytest.raises(RecipeError, match="repeats the same name"):
        CollisionScenario(
            scenario_id="s", label="s", extra_allowed_contacts=(("a", "a"),)
        )
    with pytest.raises(RecipeError, match="repeats an excluded"):
        CollisionScenario(
            scenario_id="s", label="s", excluded_targets=("pallet", "pallet")
        )


# ------------------------------------------------------------------ check


def test_template_relative_check_requires_a_pose_template():
    with pytest.raises(RecipeError, match="no template"):
        StudyCheck(check_id="c", label="c", pose_mode="template_relative")
    with pytest.raises(RecipeError, match="supplies\nno pose|supplies"):
        StudyCheck(
            check_id="c",
            label="c",
            pose_mode="template_relative",
            template=TemplateRef("step_objects", 6),
        )


def test_template_payload_requires_a_template_selection():
    with pytest.raises(RecipeError, match="payload_source='none'"):
        StudyCheck(
            check_id="c",
            label="c",
            pose_mode="absolute",
            payload_source="template",
            template=None,
        )


def test_free_space_absolute_check_is_valid_without_a_template():
    check = StudyCheck(
        check_id="c",
        label="c",
        pose_mode="absolute",
        payload_source="none",
        template=None,
    )
    assert check.template is None
    assert not check.is_unmodified_template


def test_unmodified_template_detection_follows_offsets():
    template = TemplateRef("pose_check", 1, "pallet_box_approach")
    unchanged = StudyCheck(check_id="c", label="c", template=template)
    assert unchanged.is_unmodified_template
    moved = StudyCheck(
        check_id="c",
        label="c",
        template=template,
        start_offset=Pose6D(z_m=0.01),
    )
    assert not moved.is_unmodified_template


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("criticality", "skip", "criticality"),
        ("pose_mode", "magic", "pose_mode"),
        ("payload_source", "maybe", "payload_source"),
        ("offset_frame", "base", "offset_frame"),
    ],
)
def test_check_rejects_unknown_enumerations(field, value, match):
    with pytest.raises(RecipeError, match=match):
        StudyCheck(
            check_id="c",
            label="c",
            template=TemplateRef("pose_check", 1, "a"),
            **{field: value},
        )


# ----------------------------------------------------------------- recipe


def test_recipe_rejects_duplicate_check_ids_and_unknown_scenarios():
    template = TemplateRef("pose_check", 1, "a")
    with pytest.raises(RecipeError, match="check ids must be unique"):
        _recipe(
            checks=(
                StudyCheck(check_id="c1", label="a", template=template),
                StudyCheck(check_id="c1", label="b", template=template),
            )
        )
    with pytest.raises(RecipeError, match="unknown collision\nscenario|unknown collision"):
        _recipe(
            checks=(
                StudyCheck(
                    check_id="c1", label="a", template=template, scenario_id="ghost"
                ),
            )
        )


def test_recipe_rejects_duplicate_scenario_ids_and_empty_scenario_list():
    with pytest.raises(RecipeError, match="scenario ids must be unique"):
        _recipe(
            scenarios=(
                CollisionScenario(scenario_id="s", label="one"),
                CollisionScenario(scenario_id="s", label="two"),
            ),
            checks=(),
        )
    with pytest.raises(RecipeError, match="at least one collision scenario"):
        _recipe(scenarios=(), checks=())


@pytest.mark.parametrize("bad", ["Upper", "1 space", "-leading", "a" * 65, ""])
def test_recipe_identifiers_are_strictly_constrained(bad):
    with pytest.raises(RecipeError):
        _recipe(recipe_id=bad, checks=())


def test_recipe_rejects_a_foreign_schema_version():
    with pytest.raises(RecipeError, match="unsupported study recipe schema"):
        _recipe(schema_version=STUDY_SCHEMA_VERSION + 1, checks=())


def test_recipe_rejects_nonpositive_or_huge_approach_distance():
    for value in (0.0, -0.1, 5.0):
        with pytest.raises(RecipeError, match="approach_distance_m"):
            _recipe(approach_distance_m=value, checks=())


def test_recipe_rejects_a_malformed_scene_fingerprint():
    with pytest.raises(RecipeError, match="sha256"):
        _recipe(scene_fingerprint="not-a-digest", checks=())
    assert _recipe(scene_fingerprint="a" * 64, checks=()).scene_fingerprint == "a" * 64


def test_enabled_and_hard_views_exclude_disabled_checks():
    template = TemplateRef("pose_check", 1, "a")
    recipe = _recipe(
        checks=(
            StudyCheck(check_id="h1", label="h1", template=template),
            StudyCheck(
                check_id="h2", label="h2", template=template, enabled=False
            ),
            StudyCheck(
                check_id="s1", label="s1", template=template, criticality="soft"
            ),
        )
    )
    assert [check.check_id for check in recipe.enabled_checks] == ["h1", "s1"]
    assert [check.check_id for check in recipe.enabled_hard_checks] == ["h1"]


# ------------------------------------------------------------ round trips


def test_payload_round_trip_is_lossless_and_hash_stable():
    recipe = _recipe()
    reloaded = StudyRecipe.from_mapping(recipe.payload())
    assert reloaded == recipe
    assert reloaded.recipe_hash == recipe.recipe_hash

    text = json.dumps(recipe.payload())
    assert StudyRecipe.from_json(text).recipe_hash == recipe.recipe_hash


def test_hash_ignores_derived_scenario_fields_but_tracks_real_edits():
    recipe = _recipe()
    payload = recipe.payload()
    payload["scenarios"][0]["is_diagnostic"] = True
    payload["scenarios"][0]["diagnostic_reasons"] = ["injected"]
    assert StudyRecipe.from_mapping(payload).recipe_hash == recipe.recipe_hash

    payload = recipe.payload()
    payload["checks"][0]["start_offset"]["z_m"] = 0.01
    assert StudyRecipe.from_mapping(payload).recipe_hash != recipe.recipe_hash


def test_canonical_json_never_contains_nan_or_infinity():
    text = _recipe().canonical_json()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_from_json_rejects_nan_infinity_literals():
    for literal in ("NaN", "Infinity", "-Infinity"):
        document = json.dumps(
            {"recipe_id": "r", "label": "r", "parameters": CASE}
        )
        document = document[:-1] + f',"approach_distance_m":{literal}}}'
        with pytest.raises(RecipeError, match="must not contain|must be finite"):
            StudyRecipe.from_json(document)


def test_recipe_requires_the_full_effective_case():
    with pytest.raises(RecipeError, match="missing the effective case keys"):
        _recipe(parameters={"ur_x": -1.3}, checks=())
    with pytest.raises(RecipeError, match="unknown case keys"):
        _recipe(parameters={**CASE, "surprise": 1}, checks=())
    with pytest.raises(RecipeError, match="must be finite"):
        _recipe(parameters={**CASE, "ur_x": float("inf")}, checks=())
    with pytest.raises(RecipeError, match="must be numeric"):
        _recipe(parameters={**CASE, "lift": "0.2"}, checks=())
    # A nullable clearance stays nullable, and nothing else may be null.
    assert _recipe(parameters={**CASE, "clearance": None}, checks=()).parameters[
        "clearance"
    ] is None


def test_the_effective_case_is_part_of_the_recipe_hash():
    first = _recipe(parameters=dict(CASE), checks=())
    moved = _recipe(parameters={**CASE, "ur_x": -1.0}, checks=())
    assert first.recipe_hash != moved.recipe_hash
    other_sku = _recipe(parameters={**CASE, "sku": "430060"}, checks=())
    assert first.recipe_hash != other_sku.recipe_hash


def test_approach_distance_is_part_of_the_recipe_hash():
    first = _recipe(checks=())
    further = _recipe(approach_distance_m=0.8, checks=())
    assert first.recipe_hash != further.recipe_hash


def test_case_parameters_equal_treats_none_and_numbers_exactly():
    assert case_parameters_equal(dict(CASE), dict(CASE))
    assert case_parameters_equal({**CASE}, {**CASE, "ur_x": -1.3})
    assert not case_parameters_equal({**CASE}, {**CASE, "ur_x": -1.30001})
    assert not case_parameters_equal({**CASE}, {**CASE, "clearance": 0.0})
    assert not case_parameters_equal({**CASE}, {**CASE, "sku": "430060"})
    assert not case_parameters_equal({"ur_x": 1.0}, dict(CASE))


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None])
def test_schema_version_rejects_booleans_and_non_integers(value):
    # ``True == 1`` in Python, so a boolean must not pass as version 1.
    with pytest.raises(RecipeError, match="schema_version|unsupported"):
        _recipe(schema_version=value, checks=())


def test_from_json_rejects_oversized_documents():
    oversized = " " * (MAX_RECIPE_BYTES + 1)
    with pytest.raises(RecipeError, match="exceeds"):
        StudyRecipe.from_json(oversized)


def test_from_mapping_rejects_unknown_top_level_keys():
    payload = _recipe().payload()
    payload["surprise"] = 1
    with pytest.raises(RecipeError, match="unknown keys"):
        StudyRecipe.from_mapping(payload)


def test_from_json_rejects_non_object_roots_and_broken_text():
    with pytest.raises(RecipeError, match="must be an object"):
        StudyRecipe.from_json("[1,2,3]")
    with pytest.raises(RecipeError, match="invalid recipe JSON"):
        StudyRecipe.from_json("{")


# ------------------------------------------------------------- file names


@pytest.mark.parametrize(
    "bad",
    ["../escape", "a/b", "a\\b", "..", "/abs", "C:recipe", "a.b"],
)
def test_recipe_file_name_rejects_path_traversal(bad):
    with pytest.raises(RecipeError):
        recipe_file_name(bad)


def test_recipe_file_name_appends_json_once():
    assert recipe_file_name("baseline") == "baseline.json"


def test_control_characters_are_rejected_in_free_text():
    with pytest.raises(RecipeError, match="control characters"):
        _recipe(label="bad\nlabel", checks=())


def test_degenerate_rotation_inputs_stay_finite():
    for pitch in (-90.0, 90.0, 89.999999):
        rotation = rotation_matrix_zyx(0.0, pitch, 0.0)
        assert np.all(np.isfinite(rotation))
        assert math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9)
