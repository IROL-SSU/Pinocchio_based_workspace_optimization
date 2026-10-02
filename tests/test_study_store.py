from __future__ import annotations

import json

import pytest

from decanting_workspace.study_recipe import (
    RecipeError,
    StudyCheck,
    StudyRecipe,
    TemplateRef,
)
from decanting_workspace.study_store import (
    MAX_STORED_RECIPES,
    StudyRecipeStore,
    default_study_directory,
)


def _recipe(recipe_id: str = "r1") -> StudyRecipe:
    return StudyRecipe(
        recipe_id=recipe_id,
        label=f"recipe {recipe_id}",
        parameters={"ur_x": -1.3, "corner": "southwest"},
        checks=(
            StudyCheck(
                check_id="c1",
                label="check one",
                template=TemplateRef("pose_check", 1, "pallet_box_approach"),
            ),
        ),
    )


def test_default_directory_is_repository_owned_and_named():
    directory = default_study_directory()
    assert directory.name == "study_recipes"
    assert directory.parent.name == "outputs"


def test_save_then_load_round_trips_the_recipe(tmp_path):
    store = StudyRecipeStore(tmp_path)
    recipe = _recipe()
    info = store.save(recipe)

    assert info.recipe_id == "r1"
    assert info.recipe_hash == recipe.recipe_hash
    assert info.check_count == 1
    assert info.enabled_check_count == 1
    assert (tmp_path / "r1.json").is_file()

    reloaded = store.load("r1")
    assert reloaded == recipe
    assert reloaded.recipe_hash == recipe.recipe_hash


def test_saved_file_is_valid_indented_json_without_nonfinite_values(tmp_path):
    store = StudyRecipeStore(tmp_path)
    store.save(_recipe())
    text = (tmp_path / "r1.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert "\n" in text  # human-reviewable indentation
    assert json.loads(text)["recipe_id"] == "r1"


def test_save_replaces_atomically_and_leaves_no_temporary_file(tmp_path):
    store = StudyRecipeStore(tmp_path)
    store.save(_recipe())
    store.save(_recipe())
    assert sorted(path.name for path in tmp_path.iterdir()) == ["r1.json"]


def test_load_of_a_missing_recipe_is_a_recipe_error(tmp_path):
    with pytest.raises(RecipeError, match="not found"):
        StudyRecipeStore(tmp_path).load("absent")


@pytest.mark.parametrize(
    "bad", ["../escape", "sub/dir", "..", "/etc/passwd", "a\\b", "Upper"]
)
def test_traversal_and_invalid_ids_never_touch_the_file_system(tmp_path, bad):
    store = StudyRecipeStore(tmp_path)
    with pytest.raises(RecipeError):
        store.load(bad)
    assert list(tmp_path.iterdir()) == []


def test_stored_file_whose_id_was_renamed_is_rejected(tmp_path):
    store = StudyRecipeStore(tmp_path)
    store.save(_recipe("r1"))
    payload = json.loads((tmp_path / "r1.json").read_text(encoding="utf-8"))
    payload["recipe_id"] = "r2"
    (tmp_path / "r1.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RecipeError, match="does not match its file name"):
        store.load("r1")


def test_listing_sorts_newest_first_and_skips_unreadable_files(tmp_path):
    store = StudyRecipeStore(tmp_path)
    store.save(_recipe("older"))
    store.save(_recipe("newer"))
    # A hand-edited or foreign document must not break the listing.
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "foreign.json").write_text('{"other": true}', encoding="utf-8")
    import os
    import time

    now = time.time()
    os.utime(tmp_path / "older.json", (now - 100, now - 100))
    os.utime(tmp_path / "newer.json", (now, now))

    listing = store.listing()
    assert [item.recipe_id for item in listing] == ["newer", "older"]


def test_listing_of_a_missing_directory_is_empty(tmp_path):
    assert StudyRecipeStore(tmp_path / "nothing-here").listing() == ()


def test_new_ids_are_refused_once_the_directory_is_full(tmp_path, monkeypatch):
    store = StudyRecipeStore(tmp_path)
    monkeypatch.setattr(
        StudyRecipeStore, "_file_count", lambda self: MAX_STORED_RECIPES
    )
    with pytest.raises(RecipeError, match="already holds"):
        store.save(_recipe("fresh"))
    # Overwriting an existing id stays allowed because it adds no file.
    (tmp_path / "existing.json").write_text("{}", encoding="utf-8")
    store.save(_recipe("existing"))
    assert store.load("existing").recipe_id == "existing"


def test_save_rejects_an_oversized_recipe(tmp_path, monkeypatch):
    import decanting_workspace.study_store as store_module

    monkeypatch.setattr(store_module, "MAX_RECIPE_BYTES", 10)
    with pytest.raises(RecipeError, match="exceeds"):
        StudyRecipeStore(tmp_path).save(_recipe())
    assert list(tmp_path.iterdir()) == []
