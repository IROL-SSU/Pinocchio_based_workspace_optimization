"""Run the on-demand Pinocchio evaluation UI with a live MeshCat viewer."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from io import StringIO
import math
from pathlib import Path
import sys
from threading import RLock
import webbrowser

import yaml

from .asset_prep import (
    default_sr12ia_output_urdf,
    validate_prepared_sr12ia_asset,
)
from .evaluation import EvaluationOptions
from .kinematics import IKOptions
from .live_backend import EvaluationProfile, LiveEvaluationBackend
from .live_ui import build_live_control_page
from .meshcat_ui import JsonRoute, MeshcatControlServer
from .models import SceneSpec, SceneState, default_config_path, load_scene_spec
from .paths import repository_path
from .precompute import CaseGrid, iter_case_keys
from .precompute_cli import load_case_grid_yaml
from .robots import RobotBundle, load_robot_bundle, resolve_official_ur20
from .scene import materialize_scene
from .study_backend import StudyEvaluationBackend
from .study_ui import build_study_page
from .viewer import CellViewer


@dataclass(frozen=True)
class LiveDependencies:
    """Injectable I/O and heavyweight construction boundary for :func:`main`."""

    spec_loader: Callable[..., SceneSpec]
    grid_loader: Callable[[str | Path], CaseGrid]
    official_ur20_resolver: Callable[..., tuple[Path, tuple[Path, ...]]]
    robot_loader: Callable[..., RobotBundle]
    viewer_factory: Callable[[], CellViewer]
    backend_factory: Callable[..., object]
    server_factory: Callable[..., MeshcatControlServer]
    browser_open: Callable[[str], object]
    prepared_sr12ia_resolver: Callable[[], Path] | None = None
    study_backend_factory: Callable[..., object] = StudyEvaluationBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Open a live MeshCat UI that evaluates arbitrary decanting-cell "
            "parameters on demand with Pinocchio. Results remain in memory."
        )
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--usd",
        type=Path,
        help=(
            "Read environment geometry directly from a local USD/USDA/USDC "
            "scene; task and robot settings still come from --config."
        ),
    )
    parser.add_argument(
        "--initial-grid",
        type=Path,
        default=default_initial_grid_path(),
        help=(
            "Case-grid YAML whose first exact key supplies the initial UI "
            "parameters. It is not a result cache."
        ),
    )
    parser.add_argument(
        "--default-profile",
        choices=("quick", "full"),
        default="quick",
        help="Evaluation profile selected when the live page opens.",
    )
    parser.add_argument(
        "--ur20-source",
        choices=("official", "primitive"),
        default="official",
        help="UR20 model used for both evaluation and display.",
    )
    parser.add_argument(
        "--ur20-urdf",
        type=Path,
        help="Explicit UR20 evaluation/display URDF; overrides --ur20-source.",
    )
    parser.add_argument(
        "--sr12ia-urdf",
        type=Path,
        help=(
            "Explicit SR evaluation proxy used for workspace and keep-out "
            "calculations; defaults to the configured proxy."
        ),
    )
    parser.add_argument(
        "--sr12ia-visual-urdf",
        type=Path,
        help=(
            "Explicit display-only SR URDF. By default the locally prepared "
            "FANUC/Isaac mesh is displayed."
        ),
    )
    parser.add_argument(
        "--allow-sr-visual-fallback",
        action="store_true",
        help=(
            "Explicitly display the evaluation proxy when the prepared FANUC "
            "mesh is unavailable."
        ),
    )

    parser.add_argument("--approach-mm", type=_positive_float, default=150.0)
    parser.add_argument(
        "--cartesian-step-mm",
        type=_positive_float,
        default=50.0,
        help="Full-profile Cartesian translation sampling step.",
    )
    parser.add_argument(
        "--cartesian-rotation-step-deg",
        type=_positive_float,
        default=5.0,
        help="Full-profile Cartesian rotation sampling step.",
    )
    parser.add_argument(
        "--joint-interpolation-samples",
        type=_nonnegative_int,
        default=3,
        help="Full-profile interior joint-segment samples.",
    )
    parser.add_argument(
        "--characteristic-length-mm",
        type=_positive_float,
        default=500.0,
    )
    parser.add_argument(
        "--keepout-margin-mm",
        type=_nonnegative_float,
        default=0.0,
    )
    parser.add_argument(
        "--position-tolerance-mm",
        type=_positive_float,
        default=0.001,
    )
    parser.add_argument(
        "--orientation-tolerance-deg",
        type=_positive_float,
        default=0.0001,
    )
    parser.add_argument("--ik-max-iterations", type=_positive_int, default=500)
    parser.add_argument("--ik-damping", type=_positive_float, default=1e-6)
    parser.add_argument(
        "--ik-max-backtracking-steps",
        type=_positive_int,
        default=14,
    )
    parser.add_argument(
        "--extra-arm-seed-deg",
        action="append",
        nargs=6,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=[],
        help="Additional deterministic UR20 IK seed; may be repeated.",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8766)
    parser.add_argument("--show-collisions", action="store_true")
    parser.add_argument("--hide-frames", action="store_true")
    parser.add_argument("--open", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    dependencies: LiveDependencies | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    deps = dependencies or _default_dependencies()
    server = None
    try:
        source_spec = (
            deps.spec_loader(args.config)
            if args.usd is None
            else deps.spec_loader(args.config, usd_path=args.usd)
        )
        grid = deps.grid_loader(args.initial_grid)

        evaluation_urdf = args.ur20_urdf
        ur_package_dirs: tuple[Path, ...] = ()
        if evaluation_urdf is None and args.ur20_source == "official":
            evaluation_urdf, ur_package_dirs = deps.official_ur20_resolver()
        if evaluation_urdf is None:
            evaluation_urdf = source_spec.robots["ur20"].urdf_path
        evaluation_sr_urdf = (
            args.sr12ia_urdf or source_spec.robots["sr12ia"].urdf_path
        )
        spec = _with_robot_paths(
            source_spec,
            Path(evaluation_urdf),
            Path(evaluation_sr_urdf),
        )

        initial_key = next(iter_case_keys(spec, grid), None)
        if initial_key is None:  # CaseGrid normally rejects every empty axis.
            raise ValueError("initial case grid contains no cases")
        if not grid.tote_present:
            raise ValueError("live evaluation requires tote_present=true")

        ur_bundle = deps.robot_loader(
            "ur20",
            evaluation_urdf,
            package_dirs=ur_package_dirs,
            floating_base=True,
            load_visual=False,
            suction_proxy=spec.robots["ur20"].suction_proxy,
        )
        sr_bundle = deps.robot_loader(
            "sr12ia",
            evaluation_sr_urdf,
            floating_base=True,
            load_visual=False,
        )

        state = SceneState(
            lift_height_m=initial_key.lift_height_m,
            sku=initial_key.sku,
            corner_samples=(initial_key.corner,),
            tote_present=grid.tote_present,
            tote_long_axis_offset_m=initial_key.tote_offset_m,
            sr_j3_stroke_m=initial_key.sr_j3_stroke_m,
            clearance_m=initial_key.clearance_m,
        )
        snapshot = materialize_scene(
            spec,
            state,
            ur_base=initial_key.ur_base,
            sr_base=initial_key.sr_base,
        )

        visual_sr_urdf = args.sr12ia_visual_urdf
        using_prepared_sr_mesh = False
        if visual_sr_urdf is None:
            try:
                resolver = (
                    deps.prepared_sr12ia_resolver
                    or _resolve_prepared_sr12ia_visual
                )
                visual_sr_urdf = resolver()
                using_prepared_sr_mesh = True
            except (FileNotFoundError, OSError, ValueError) as exc:
                if not args.allow_sr_visual_fallback:
                    raise RuntimeError(
                        "prepared FANUC SR-12iA visual asset is required; run "
                        "'decanting-setup --accept-fanuc-license' (or provide "
                        "--sr12ia-visual-urdf). Use "
                        "--allow-sr-visual-fallback only when the approximate "
                        "display is intentional"
                    ) from exc
                visual_sr_urdf = Path(evaluation_sr_urdf)
                print(
                    "WARNING: prepared FANUC SR-12iA mesh is unavailable; "
                    "--allow-sr-visual-fallback requested, so the evaluation "
                    f"model is displayed ({exc}).",
                    file=sys.stderr,
                )
        display_only_sr_model = (
            Path(visual_sr_urdf).resolve()
            != Path(evaluation_sr_urdf).resolve()
        )

        viewer = deps.viewer_factory()
        viewer.render(
            spec,
            snapshot,
            ur20_urdf=evaluation_urdf,
            ur20_package_dirs=ur_package_dirs,
            sr12ia_urdf=visual_sr_urdf,
            sr12ia_visual_only=display_only_sr_model,
            show_collisions=args.show_collisions,
            show_frames=not args.hide_frames,
        )
        profiles = _evaluation_profiles(args)
        backend = deps.backend_factory(
            spec,
            initial_key,
            ur_bundle,
            sr_bundle,
            viewer,
            profiles=profiles,
            default_profile=args.default_profile,
            approach_distance_m=args.approach_mm / 1000.0,
            keepout_margin_m=args.keepout_margin_mm / 1000.0,
        )
        # One reentrant lock serializes every actual Pinocchio/viewer touch
        # across the legacy live endpoints, the study request handlers and the
        # study worker threads.  It is reentrant so a serialized route and the
        # backend method it calls can both take it on the same thread.
        compute_lock = RLock()
        study_backend = deps.study_backend_factory(
            spec,
            initial_key,
            ur_bundle,
            sr_bundle,
            viewer,
            profiles={
                profile.profile_id: profile.options for profile in profiles
            },
            default_profile=args.default_profile,
            compute_lock=compute_lock,
            approach_distance_m=args.approach_mm / 1000.0,
            keepout_margin_m=args.keepout_margin_mm / 1000.0,
            tote_present=grid.tote_present,
        )
        server = deps.server_factory(
            (args.host, args.port),
            backend,
            viewer.url(),
            page_builder=build_live_control_page,
            extra_pages={"/study": build_study_page(viewer.url())},
            extra_json_post=_study_routes(study_backend),
            selection_lock=compute_lock,
        )

        control_url = server.control_url
        print("LIVE CONTROL UI (parameters + calculated results):")
        print(control_url, flush=True)
        print(f"Embedded MeshCat viewer only (no controls): {viewer.url()}")
        print(f"Initial case: {initial_key.stable_id}")
        print(f"Default evaluation profile: {args.default_profile}")
        print(f"SR-12iA visual model: {Path(visual_sr_urdf).resolve()}")
        if using_prepared_sr_mesh:
            print(
                "NOTE: the prepared FANUC SR mesh is display-only; all live "
                "feasibility calculations use the configured evaluation proxy."
            )
        if display_only_sr_model and args.show_collisions:
            print(
                "WARNING: displayed SR collision geometry belongs to the visual "
                "surrogate, not the live evaluation collision model.",
                file=sys.stderr,
            )
        if args.open:
            opened = deps.browser_open(control_url)
            if opened is False:
                print(
                    "WARNING: the browser did not open automatically; open "
                    f"the live control URL manually: {control_url}",
                    file=sys.stderr,
                )
        print(f"OPEN OR REFRESH THE LIVE CONTROL UI: {control_url}", flush=True)
        print("Press Ctrl+C to stop the live evaluation UI.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.server_close()


def _study_routes(backend: object) -> dict[str, JsonRoute]:
    """Map the study JSON API onto the shared control server.

    Only endpoints that actually touch Pinocchio or MeshCat are serialized.
    Job control stays unserialized so progress and cancellation remain
    responsive while a worker holds the compute lock for one check.
    """

    def route(
        name: str,
        *,
        serialized: bool,
        invalid: str,
        failed: str,
    ) -> JsonRoute:
        handler = getattr(backend, name, None)
        if not callable(handler):
            raise RuntimeError(f"study backend has no {name} endpoint")
        return JsonRoute(
            handler=handler,
            serialized=serialized,
            invalid_error=invalid,
            failed_error=failed,
        )

    return {
        "/api/study/catalog": route(
            "catalog",
            serialized=True,
            invalid="invalid_study_request",
            failed="study_catalog_failed",
        ),
        "/api/study/validate": route(
            "validate",
            serialized=True,
            invalid="invalid_study_recipe",
            failed="study_validation_failed",
        ),
        "/api/study/preview": route(
            "preview",
            serialized=True,
            invalid="invalid_study_recipe",
            failed="study_preview_failed",
        ),
        "/api/study/preview/clear": route(
            "clear_preview",
            serialized=True,
            invalid="invalid_study_request",
            failed="study_preview_failed",
        ),
        "/api/study/recipe/save": route(
            "save_recipe",
            serialized=True,
            invalid="invalid_study_recipe",
            failed="study_save_failed",
        ),
        "/api/study/recipe/load": route(
            "load_recipe",
            serialized=True,
            invalid="invalid_study_recipe",
            failed="study_load_failed",
        ),
        "/api/study/recipe/import": route(
            "import_recipe",
            serialized=True,
            invalid="invalid_study_recipe",
            failed="study_import_failed",
        ),
        "/api/study/recipe/list": route(
            "list_recipes",
            serialized=False,
            invalid="invalid_study_request",
            failed="study_list_failed",
        ),
        "/api/study/evaluate": route(
            "start_evaluation",
            serialized=True,
            invalid="invalid_study_recipe",
            failed="study_evaluation_failed",
        ),
        "/api/study/scout": route(
            "start_scout",
            serialized=True,
            invalid="invalid_study_recipe",
            failed="study_scout_failed",
        ),
        "/api/study/job": route(
            "job_status",
            serialized=False,
            invalid="invalid_study_request",
            failed="study_job_failed",
        ),
        "/api/study/cancel": route(
            "cancel_job",
            serialized=False,
            invalid="invalid_study_request",
            failed="study_cancel_failed",
        ),
        "/api/study/show": route(
            "show_configuration",
            serialized=True,
            invalid="invalid_study_request",
            failed="study_show_failed",
        ),
    }


def default_initial_grid_path() -> Path:
    """Return the verified working grid used only for initial UI values."""

    return repository_path("config", "case_grid_working.yaml")


def _evaluation_profiles(args: argparse.Namespace) -> tuple[EvaluationProfile, ...]:
    seeds = tuple(
        tuple(math.radians(float(value)) for value in seed)
        for seed in args.extra_arm_seed_deg
    )
    ik = IKOptions(
        position_tolerance_m=args.position_tolerance_mm / 1000.0,
        orientation_tolerance_rad=math.radians(
            args.orientation_tolerance_deg
        ),
        max_iterations=args.ik_max_iterations,
        damping=args.ik_damping,
        max_backtracking_steps=args.ik_max_backtracking_steps,
    )
    characteristic_length = args.characteristic_length_mm / 1000.0
    common = {
        "characteristic_length_m": characteristic_length,
        "ik": ik,
        "extra_arm_seeds": seeds,
    }
    quick = EvaluationProfile(
        profile_id="quick",
        label="Quick preview",
        description=(
            "Endpoint-oriented 10 m / 180 degree sampling without interior "
            "joint-segment samples."
        ),
        options=EvaluationOptions(
            cartesian_translation_step_m=10.0,
            cartesian_rotation_step_rad=math.pi,
            joint_interpolation_samples=0,
            **common,
        ),
    )
    full = EvaluationProfile(
        profile_id="full",
        label="Full evaluation",
        description=(
            "Configured Cartesian and joint-segment sampling for collision-aware "
            "evaluation."
        ),
        options=EvaluationOptions(
            cartesian_translation_step_m=args.cartesian_step_mm / 1000.0,
            cartesian_rotation_step_rad=math.radians(
                args.cartesian_rotation_step_deg
            ),
            joint_interpolation_samples=args.joint_interpolation_samples,
            **common,
        ),
    )
    return quick, full


def _with_robot_paths(spec: SceneSpec, urdf: Path, sr_urdf: Path) -> SceneSpec:
    robots = dict(spec.robots)
    robots["ur20"] = replace(robots["ur20"], urdf_path=urdf.resolve())
    robots["sr12ia"] = replace(
        robots["sr12ia"],
        urdf_path=sr_urdf.resolve(),
    )
    return replace(spec, robots=robots)


def _default_dependencies() -> LiveDependencies:
    return LiveDependencies(
        spec_loader=load_scene_spec,
        grid_loader=load_case_grid_yaml,
        official_ur20_resolver=resolve_official_ur20,
        robot_loader=load_robot_bundle,
        prepared_sr12ia_resolver=_resolve_prepared_sr12ia_visual,
        viewer_factory=_quiet_cell_viewer,
        backend_factory=LiveEvaluationBackend,
        server_factory=MeshcatControlServer,
        browser_open=webbrowser.open,
    )


def _quiet_cell_viewer() -> CellViewer:
    """Create MeshCat without advertising its raw, controls-free URL."""

    with redirect_stdout(StringIO()):
        return CellViewer()


def _resolve_prepared_sr12ia_visual() -> Path:
    path = default_sr12ia_output_urdf()
    validate_prepared_sr12ia_asset(path)
    return path.resolve()


def _port(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 0 <= result <= 65535:
        raise argparse.ArgumentTypeError("port must be in [0, 65535]")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
