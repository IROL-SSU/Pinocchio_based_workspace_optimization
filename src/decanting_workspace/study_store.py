"""Atomic, path-traversal-free storage for study recipes.

Recipes live in one server-owned directory.  Clients never supply a file
system path; they supply a validated ``recipe_id`` and the server derives the
file name.  Writes land on a sibling temporary file and are published with
``os.replace`` so a partially written recipe is never visible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from .paths import repository_path
from .study_recipe import (
    MAX_RECIPE_BYTES,
    RecipeError,
    StudyRecipe,
    recipe_file_name,
)


MAX_STORED_RECIPES = 500


@dataclass(frozen=True)
class StoredRecipeInfo:
    recipe_id: str
    label: str
    recipe_hash: str
    scene_fingerprint: str
    check_count: int
    enabled_check_count: int
    modified_unix: float

    def payload(self) -> dict[str, object]:
        return {
            "recipe_id": self.recipe_id,
            "label": self.label,
            "recipe_hash": self.recipe_hash,
            "scene_fingerprint": self.scene_fingerprint,
            "check_count": self.check_count,
            "enabled_check_count": self.enabled_check_count,
            "modified_unix": self.modified_unix,
        }


def default_study_directory() -> Path:
    """Return the repository-owned directory that holds saved recipes."""

    return repository_path("outputs", "study_recipes")


class StudyRecipeStore:
    """File-backed recipe store confined to one directory."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(
            default_study_directory() if directory is None else directory
        ).resolve()

    def _path_for(self, recipe_id: str) -> Path:
        # ``recipe_file_name`` rejects separators, dots and absolute inputs, so
        # the join cannot escape. The containment assertion keeps that
        # guarantee local and checkable.
        target = (self.directory / recipe_file_name(recipe_id)).resolve()
        if target.parent != self.directory:
            raise RecipeError("recipe_id resolved outside the study directory")
        return target

    def save(self, recipe: StudyRecipe) -> StoredRecipeInfo:
        """Write one recipe atomically and return its stored description."""

        encoded = json.dumps(
            recipe.payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > MAX_RECIPE_BYTES:
            raise RecipeError(f"recipe exceeds {MAX_RECIPE_BYTES} bytes")
        target = self._path_for(recipe.recipe_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not target.exists() and self._file_count() >= MAX_STORED_RECIPES:
            raise RecipeError(
                f"study directory already holds {MAX_STORED_RECIPES} recipes; "
                "remove one before saving a new id"
            )
        handle, temporary = tempfile.mkstemp(
            prefix=f".{recipe.recipe_id}.", suffix=".tmp", dir=self.directory
        )
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return self._info(recipe, target.stat().st_mtime)

    def load(self, recipe_id: str) -> StudyRecipe:
        """Read and strictly re-validate one stored recipe."""

        target = self._path_for(recipe_id)
        try:
            raw = target.read_bytes()
        except FileNotFoundError as exc:
            raise RecipeError(f"saved recipe not found: {recipe_id}") from exc
        except OSError as exc:
            raise RecipeError(f"could not read saved recipe {recipe_id}: {exc}") from exc
        recipe = StudyRecipe.from_json(raw.decode("utf-8", errors="strict"))
        if recipe.recipe_id != recipe_id:
            raise RecipeError(
                f"stored recipe id {recipe.recipe_id!r} does not match its file name"
            )
        return recipe

    def listing(self) -> tuple[StoredRecipeInfo, ...]:
        """Return every readable stored recipe, newest first."""

        if not self.directory.is_dir():
            return ()
        results: list[StoredRecipeInfo] = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                recipe = StudyRecipe.from_json(path.read_bytes().decode("utf-8"))
            except (RecipeError, OSError, UnicodeDecodeError):
                # A hand-edited or foreign file must not break the listing.
                continue
            results.append(self._info(recipe, path.stat().st_mtime))
        results.sort(key=lambda item: (-item.modified_unix, item.recipe_id))
        return tuple(results)

    def _file_count(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json"))

    @staticmethod
    def _info(recipe: StudyRecipe, modified_unix: float) -> StoredRecipeInfo:
        return StoredRecipeInfo(
            recipe_id=recipe.recipe_id,
            label=recipe.label,
            recipe_hash=recipe.recipe_hash,
            scene_fingerprint=recipe.scene_fingerprint,
            check_count=len(recipe.checks),
            enabled_check_count=len(recipe.enabled_checks),
            modified_unix=float(modified_unix),
        )
