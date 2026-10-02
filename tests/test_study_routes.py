"""Server routing for the study API, including legacy-endpoint regression."""

from __future__ import annotations

import json
from threading import RLock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from decanting_workspace.live_cli import _study_routes
from decanting_workspace.meshcat_ui import JsonRoute, MeshcatControlServer

MESHCAT_URL = "http://127.0.0.1:7999/static/"


class _LegacyBackend:
    """Minimal stand-in for the live backend's catalog/select/evaluate API."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def catalog(self) -> dict:
        return {"schema_version": 1, "mode": "live", "cases": []}

    def select(self, request: dict) -> dict:
        self.calls.append(("select", dict(request)))
        return {"selected": True, "case_id": request.get("case_id")}

    def evaluate(self, request: dict) -> dict:
        self.calls.append(("evaluate", dict(request)))
        return {"evaluation_id": 1, "persisted": False}


class _StudyStub:
    """Records which study endpoint the server routed each request to."""

    def __init__(self, lock) -> None:
        self.lock = lock
        self.seen: list[str] = []

    def _record(self, name: str, request: dict) -> dict:
        self.seen.append(name)
        return {"endpoint": name, "request": dict(request)}

    def catalog(self, request):
        return self._record("catalog", request)

    def validate(self, request):
        return self._record("validate", request)

    def preview(self, request):
        return self._record("preview", request)

    def clear_preview(self, request):
        return self._record("clear_preview", request)

    def save_recipe(self, request):
        return self._record("save_recipe", request)

    def load_recipe(self, request):
        return self._record("load_recipe", request)

    def import_recipe(self, request):
        return self._record("import_recipe", request)

    def list_recipes(self, request):
        return self._record("list_recipes", request)

    def start_evaluation(self, request):
        return self._record("start_evaluation", request)

    def start_scout(self, request):
        return self._record("start_scout", request)

    def job_status(self, request):
        return self._record("job_status", request)

    def cancel_job(self, request):
        return self._record("cancel_job", request)

    def show_configuration(self, request):
        return self._record("show_configuration", request)


def _serve(study: _StudyStub, lock) -> tuple[MeshcatControlServer, _LegacyBackend]:
    backend = _LegacyBackend()
    server = MeshcatControlServer(
        ("127.0.0.1", 0),
        backend,
        MESHCAT_URL,
        page_builder=lambda url: f"<html>live {url}</html>",
        extra_pages={"/study": "<html>study page</html>"},
        extra_json_post=_study_routes(study),
        selection_lock=lock,
    )
    server.start_background()
    return server, backend


@pytest.fixture
def serve_study():
    """Serve a study stub whose handlers are already bound to the routes.

    Routes capture bound methods when the server is constructed, so a test
    that needs a failing endpoint must install it before serving.
    """

    created: list[MeshcatControlServer] = []

    def factory(**overrides):
        lock = RLock()
        study = _StudyStub(lock)
        for name, handler in overrides.items():
            setattr(study, name, handler)
        server, backend = _serve(study, lock)
        created.append(server)
        return server, backend, study

    try:
        yield factory
    finally:
        for server in created:
            server.shutdown()
            server.server_close()


@pytest.fixture
def served(serve_study):
    return serve_study()


def _get(server, path: str):
    with urlopen(f"{server.control_url.rstrip('/')}{path}", timeout=10) as response:
        return response.status, response.read().decode("utf-8")


def _post(server, path: str, payload: object):
    request = Request(
        f"{server.control_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


# ------------------------------------------------------- legacy regression


def test_legacy_root_page_and_catalog_are_unchanged(served):
    server, _, _ = served
    status, body = _get(server, "/")
    assert status == 200
    assert "live" in body

    status, body = _get(server, "/api/catalog")
    assert status == 200
    payload = json.loads(body)
    assert payload["mode"] == "live"
    assert payload["meshcat_url"] == MESHCAT_URL


def test_legacy_healthz_still_responds(served):
    server, _, _ = served
    status, body = _get(server, "/healthz")
    assert status == 200
    assert json.loads(body) == {"status": "ok"}


def test_legacy_select_and_evaluate_still_reach_the_live_backend(served):
    server, backend, _ = served
    status, payload = _post(server, "/api/select", {"case_id": "abc"})
    assert status == 200
    assert payload["case_id"] == "abc"

    status, payload = _post(server, "/api/evaluate", {"parameters": {}})
    assert status == 200
    assert payload["evaluation_id"] == 1
    assert [name for name, _ in backend.calls] == ["select", "evaluate"]


def test_legacy_endpoints_keep_their_64_kib_body_limit(served):
    server, _, _ = served
    oversized = {"case_id": "x" * (64 * 1024)}
    with pytest.raises(HTTPError) as error:
        _post(server, "/api/evaluate", oversized)
    assert error.value.code == 400
    assert json.loads(error.value.read())["error"] == "invalid_length"


def test_an_unknown_path_is_still_a_json_404(served):
    server, _, _ = served
    with pytest.raises(HTTPError) as error:
        _get(server, "/nope")
    assert error.value.code == 404
    with pytest.raises(HTTPError) as error:
        _post(server, "/api/nope", {})
    assert error.value.code == 404


# ------------------------------------------------------------ study routes


def test_study_page_is_served_at_its_own_path(served):
    server, _, _ = served
    status, body = _get(server, "/study")
    assert status == 200
    assert "study page" in body


@pytest.mark.parametrize(
    "path,endpoint",
    [
        ("/api/study/catalog", "catalog"),
        ("/api/study/validate", "validate"),
        ("/api/study/preview", "preview"),
        ("/api/study/preview/clear", "clear_preview"),
        ("/api/study/recipe/save", "save_recipe"),
        ("/api/study/recipe/load", "load_recipe"),
        ("/api/study/recipe/import", "import_recipe"),
        ("/api/study/recipe/list", "list_recipes"),
        ("/api/study/evaluate", "start_evaluation"),
        ("/api/study/scout", "start_scout"),
        ("/api/study/job", "job_status"),
        ("/api/study/cancel", "cancel_job"),
        ("/api/study/show", "show_configuration"),
    ],
)
def test_every_study_route_reaches_its_backend_endpoint(served, path, endpoint):
    server, _, study = served
    status, payload = _post(server, path, {"probe": 1})
    assert status == 200
    assert payload["endpoint"] == endpoint
    assert payload["request"] == {"probe": 1}


def test_job_control_routes_are_not_serialized(served):
    _, _, study = served
    routes = _study_routes(study)
    assert routes["/api/study/job"].serialized is False
    assert routes["/api/study/cancel"].serialized is False
    assert routes["/api/study/recipe/list"].serialized is False
    # Anything that touches Pinocchio or MeshCat must hold the shared lock.
    for path in (
        "/api/study/catalog",
        "/api/study/validate",
        "/api/study/preview",
        "/api/study/evaluate",
        "/api/study/scout",
        "/api/study/show",
    ):
        assert routes[path].serialized is True, path


def test_study_routes_report_a_missing_backend_endpoint(served):
    class _Incomplete:
        def catalog(self, request):
            return {}

    with pytest.raises(RuntimeError, match="has no validate endpoint"):
        _study_routes(_Incomplete())


def test_a_study_recipe_error_becomes_a_400_with_its_message(serve_study):
    def failing(request):
        raise ValueError("recipe scene_fingerprint does not match")

    server, _, _ = serve_study(validate=failing)
    with pytest.raises(HTTPError) as error:
        _post(server, "/api/study/validate", {})
    assert error.value.code == 400
    body = json.loads(error.value.read())
    assert body["error"] == "invalid_study_recipe"
    assert "fingerprint" in body["message"]


def test_an_unexpected_study_failure_becomes_a_500_not_a_crash(serve_study):
    def exploding(request):
        raise RuntimeError("coal exploded")

    server, _, _ = serve_study(start_evaluation=exploding)
    with pytest.raises(HTTPError) as error:
        _post(server, "/api/study/evaluate", {})
    assert error.value.code == 500
    assert json.loads(error.value.read())["error"] == "study_evaluation_failed"


def test_study_routes_accept_a_larger_upload_than_legacy_endpoints(served):
    server, _, study = served
    document = {"document": "x" * (200 * 1024)}
    status, payload = _post(server, "/api/study/recipe/import", document)
    assert status == 200
    assert payload["endpoint"] == "import_recipe"


def test_malformed_json_is_a_400_before_the_handler_runs(served):
    server, _, study = served
    request = Request(
        f"{server.control_url.rstrip('/')}/api/study/validate",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=10)
    assert error.value.code == 400
    assert study.seen == []


def test_a_non_object_json_root_is_rejected(served):
    server, _, study = served
    with pytest.raises(HTTPError) as error:
        _post(server, "/api/study/validate", [1, 2, 3])
    assert error.value.code == 400
    assert "must be an object" in json.loads(error.value.read())["message"]


def test_responses_never_serialize_nonfinite_numbers(serve_study):
    server, _, _ = serve_study(validate=lambda request: {"value": float("inf")})
    with pytest.raises(HTTPError) as error:
        _post(server, "/api/study/validate", {})
    # Strict output refuses infinity instead of emitting invalid JSON such as
    # a bare ``Infinity`` token that no conforming parser accepts.
    assert error.value.code == 400
    body = json.loads(error.value.read())
    assert body["error"] == "invalid_study_recipe"
    assert "not JSON compliant" in body["message"] or "Infinity" in body["message"]


# -------------------------------------------------------- route validation


def test_reserved_paths_cannot_be_overridden():
    for path in ("/", "/healthz", "/api/catalog", "/api/select", "/api/evaluate"):
        with pytest.raises(ValueError, match="reserved"):
            MeshcatControlServer(
                ("127.0.0.1", 0),
                _LegacyBackend(),
                MESHCAT_URL,
                extra_pages={path: "<html></html>"},
            )


def test_route_paths_must_be_absolute_and_clean():
    for path in ("study", "/with space", "/q?x=1", "/frag#a", "/" + "a" * 200):
        with pytest.raises(ValueError):
            MeshcatControlServer(
                ("127.0.0.1", 0),
                _LegacyBackend(),
                MESHCAT_URL,
                extra_pages={path: "<html></html>"},
            )


def test_extra_routes_must_be_json_route_instances():
    with pytest.raises(TypeError, match="JsonRoute"):
        MeshcatControlServer(
            ("127.0.0.1", 0),
            _LegacyBackend(),
            MESHCAT_URL,
            extra_json_post={"/api/x": (lambda request: {})},
        )


def test_json_route_requires_a_callable_handler_and_boolean_flag():
    with pytest.raises(TypeError, match="callable"):
        JsonRoute(handler=None)
    with pytest.raises(TypeError, match="boolean"):
        JsonRoute(handler=lambda request: {}, serialized="yes")
