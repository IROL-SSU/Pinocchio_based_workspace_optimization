from __future__ import annotations

import pytest

from decanting_workspace.live_ui import build_live_control_page
from decanting_workspace.study_ui import build_study_page


@pytest.fixture(scope="module")
def page() -> str:
    return build_study_page("http://127.0.0.1:7000/static/")


def test_page_embeds_the_shared_meshcat_view_and_escapes_the_url():
    page = build_study_page("http://127.0.0.1:7000/static/?a=1&b=2")
    assert (
        '<iframe id="meshcat" src="http://127.0.0.1:7000/static/?a=1&amp;b=2"'
        in page
    )
    # A quote in the URL must not be able to break out of the attribute.
    hostile = build_study_page('http://127.0.0.1:7000/s/?a="onload="x')
    assert 'onload="x' not in hostile
    assert "&quot;onload=&quot;x" in hostile


def test_page_rejects_a_non_http_viewer_url():
    with pytest.raises(ValueError, match="absolute http"):
        build_study_page("javascript:alert(1)")
    with pytest.raises(ValueError, match="absolute http"):
        build_study_page("/relative/path")


def test_live_page_links_to_the_study_page_and_scopes_the_claim():
    live = build_live_control_page("http://127.0.0.1:7000/static/")
    assert 'href="/study"' in live
    assert "UR20 check study" in live
    assert "결과 의미는 바뀌지 않습니다" in live


def test_study_page_links_back_to_the_live_page(page):
    assert 'href="/"' in page


def test_page_exposes_every_authoring_control(page):
    for identifier in (
        "recipeId",
        "recipeLabel",
        "recipeProfile",
        "checkList",
        "checkDetail",
        "checkLabel",
        "checkIdField",
        "checkCriticality",
        "checkEnabled",
        "checkMetric",
        "checkScenario",
        "checkTemplate",
        "checkPayload",
        "checkPoseMode",
        "checkOffsetFrame",
        "checkReferenceFrame",
        "copyCheck",
        "removeCheck",
        "newCheck",
        "toggleAll",
        "resetBaseline",
    ):
        assert f'id="{identifier}"' in page, identifier


def test_page_exposes_collision_scenario_and_target_controls(page):
    for identifier in (
        "scenarioSelect",
        "newScenario",
        "removeScenario",
        "scenarioLabel",
        "scenarioSelf",
        "scenarioFloor",
        "targetSearch",
        "targetRole",
        "targetList",
        "excludeShown",
        "keepShown",
        "highlightShown",
        "contactFirst",
        "contactSecond",
        "contactNames",
        "addContact",
        "contactList",
    ):
        assert f'id="{identifier}"' in page, identifier


def test_page_exposes_run_save_and_result_controls(page):
    for identifier in (
        "validateRecipe",
        "previewRecipe",
        "runSelected",
        "runAll",
        "cancelJob",
        "clearOverlay",
        "jobProgress",
        "jobLine",
        "saveRecipe",
        "downloadRecipe",
        "uploadRecipe",
        "uploadInput",
        "savedRecipes",
        "loadSaved",
        "resultBox",
        "resultCheck",
        "resultSample",
        "showPose",
        "sampleBox",
        "staleNote",
        "fingerprintNote",
    ):
        assert f'id="{identifier}"' in page, identifier


def test_page_exposes_the_bounded_scout_inputs(page):
    for identifier in (
        "scoutToggle",
        "scoutPanel",
        "scoutMode",
        "scoutXMin",
        "scoutXMax",
        "scoutXN",
        "scoutYawMin",
        "scoutYawMax",
        "scoutMax",
        "scoutSeed",
        "runScout",
    ):
        assert f'id="{identifier}"' in page, identifier


def test_page_posts_only_the_documented_study_endpoints(page):
    expected = {
        "/api/study/catalog",
        "/api/study/validate",
        "/api/study/preview",
        "/api/study/preview/clear",
        "/api/study/frame",
        "/api/study/recipe/save",
        "/api/study/recipe/load",
        "/api/study/recipe/import",
        "/api/study/evaluate",
        "/api/study/scout",
        "/api/study/job",
        "/api/study/cancel",
        "/api/study/show",
    }
    import re

    used = set(re.findall(r'postJson\("(/api/[^"]+)"', page))
    assert used == expected


def test_page_never_computes_a_feasibility_label_itself(page):
    # Status strings must be read from the backend payload, never derived in
    # the browser. The page may map a backend status onto a colour class.
    assert 'result.status === "study_feasible"' in page
    assert "`Status: ${result.status}`" in page
    assert "`Certification: ${result.certification}`" in page
    for forbidden in (
        'result.status = ',
        'result.certification = ',
        'status: "study_feasible"',
        'certification: "study_feasible"',
    ):
        assert forbidden not in page, forbidden


def test_page_declares_the_transform_order_and_units(page):
    assert "world_T_new = world_T_template" in page
    assert "world_T_tcp = world_T_frame" in page
    assert "길이 입력은 mm, 회전 입력은 degree" in page
    assert "Rz(yaw) Ry(pitch) Rx(roll)" in page


def test_page_converts_millimetres_for_translation_only(page):
    assert '{key: "x_m", label: "X (mm)", scale: 1000}' in page
    assert '{key: "roll_deg", label: "roll (deg)", scale: 1}' in page
    assert "raw / field.scale" in page


def test_page_marks_results_stale_after_a_recipe_edit(page):
    assert "markRecipeChanged" in page
    assert "stale" in page
    assert "resultRecipeHash" in page


def test_page_scopes_the_study_claim_and_does_not_promise_unbuilt_features(page):
    assert "study_feasible" in page
    assert "전체 공정 검증이 아닙니다" in page
    assert "전체 공정 최적화 결과가 아닙니다" in page
    # Picking and gizmos are explicitly out of scope; the page must not offer
    # controls that do not exist.
    for absent in ("drag", "gizmo", "click pick", "surface pick"):
        assert absent not in page.lower()


def test_page_shows_both_intended_and_checker_exempt_contacts(page):
    assert "intended contacts (template)" in page
    assert "checker-exempt pairs" in page
    assert "phase_allowed_contacts" in page


def test_page_warns_when_a_fingerprint_override_was_used(page):
    assert "fingerprint_override" in page
    assert "scene/model fingerprint가 다릅니다" in page


def test_page_polls_rather_than_blocking_on_a_long_run(page):
    assert "POLL_MS" in page
    assert "setTimeout(pollJob, POLL_MS)" in page
    assert '"/api/study/job"' in page
    assert "cancelJob" in page


# ------------------------------------------- the recipe-owned effective case


def test_page_exposes_the_effective_case_panel(page):
    for identifier in (
        "caseControls",
        "caseApproach",
        "applyCase",
        "revertCase",
        "caseStatus",
        "frameCell",
        "applyScoutBest",
    ):
        assert f'id="{identifier}"' in page, identifier
    # The panel has to be built from the server's own bounds, never from
    # numbers hard-coded in the browser.
    assert "catalog.parameter_schema" in page
    assert "catalog has no parameter_schema" in page


def test_page_never_sends_a_parameter_override_beside_a_recipe(page):
    # The effective case is hashed into recipe_hash, so a per-request override
    # would detach the displayed result from the recipe that names it.
    assert "return Object.assign({recipe: recipePayload()}, extra || {});" in page
    assert "parameters: catalog.parameters" not in page


def test_page_reports_the_effective_case_and_installation_with_every_result(page):
    assert "Effective case: UR20" in page
    assert "Installation (independent of collision scenarios)" in page
    # The SR cutting workspace is context, not part of the UR20 verdict.
    assert "not part of this certification" in page


def test_page_distinguishes_unresolved_ik_from_a_collision(page):
    assert "unresolved (ik_not_found)" in page
    assert "It is not a proof that no solution exists" in page
    assert "collision_hard_check_ids" in page
    assert "unresolved_hard_check_ids" in page


def test_page_separates_the_strict_and_diagnostic_scout_rankings(page):
    assert "ranking_strict" in page
    assert "ranking_diagnostic" in page
    assert "NOT comparable to the strict list" in page


def test_page_labels_the_exclusion_list_as_exclusion_and_the_toggles_as_include(
    page,
):
    # The two controls run in opposite directions, which is exactly the kind of
    # thing a user gets wrong once and never notices.
    assert "INCLUDE robot self collision" in page
    assert "INCLUDE floor collision" in page
    assert "<strong>EXCLUDED</strong>" in page


def test_page_marks_a_preview_payload_as_schematic_rather_than_solved(page):
    assert "payload_display" in page
    assert "계산된 자세 아님" in page


def test_page_restores_the_previous_case_when_the_server_refuses_one(page):
    assert "recipe.parameters = previousParameters;" in page
    assert "recipe.approach_distance_m = previousApproach;" in page


def test_page_will_not_place_a_payload_for_a_since_edited_recipe(page):
    # An edited recipe names a different effective case, so its scene and
    # payload do not belong to the stored joint configuration.
    assert "const stale = staleNote.innerHTML !== \"\";" in page
    assert "? {q: sample.q}" in page
    assert "payload/scene은 표시하지 않았습니다" in page
