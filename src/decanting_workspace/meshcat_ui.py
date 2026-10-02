"""Local web controls wrapped around a live MeshCat view.

The browser never solves IK itself.  A supplied control page can either submit
exact cache selections or request an on-demand evaluation from the local
Python backend, which then updates the already-running MeshCat viewer.  This
also keeps licensed robot meshes in the local session instead of embedding
them in a redistributable HTML report.
"""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import os
import socket
from threading import Lock, Thread
from typing import Callable, Mapping, Protocol
from urllib.parse import urlsplit


class PlaybackBackend(Protocol):
    """Common catalog/selection interface consumed by the control server."""

    def catalog(self) -> Mapping[str, object]:
        """Return JSON-compatible exact cases and step/check/sample metadata."""

    def select(self, request: Mapping[str, object]) -> Mapping[str, object]:
        """Apply one cached selection and return its display/status details."""


@dataclass(frozen=True)
class JsonRoute:
    """One extra JSON endpoint registered beside the legacy playback API.

    ``serialized`` requests the shared compute lock for the duration of the
    call.  Job control endpoints set it to ``False`` so progress polling and
    cancellation stay responsive while a long worker run holds the lock
    between checks.
    """

    handler: Callable[[Mapping[str, object]], Mapping[str, object]]
    serialized: bool = True
    invalid_error: str = "invalid_request"
    failed_error: str = "request_failed"

    def __post_init__(self) -> None:
        if not callable(self.handler):
            raise TypeError("JsonRoute handler must be callable")
        if not isinstance(self.serialized, bool):
            raise TypeError("JsonRoute serialized must be a boolean")


class MeshcatControlServer(ThreadingHTTPServer):
    """Threaded localhost server for the parameter panel and JSON API."""

    daemon_threads = True
    # ``HTTPServer`` enables SO_REUSEADDR.  On Windows that permits two live
    # processes to bind the same port, so a restart can keep serving a stale
    # control page nondeterministically.  Keep quick rebinding on POSIX, but
    # request an exclusive listener on Windows.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_EXCLUSIVEADDRUSE,
                1,
            )
        super().server_bind()

    def __init__(
        self,
        address: tuple[str, int],
        backend: PlaybackBackend,
        meshcat_url: str,
        *,
        page_builder: Callable[[str], str] | None = None,
        extra_pages: Mapping[str, str] | None = None,
        extra_json_get: Mapping[str, JsonRoute] | None = None,
        extra_json_post: Mapping[str, JsonRoute] | None = None,
        selection_lock: object | None = None,
    ) -> None:
        _validate_meshcat_url(meshcat_url)
        self.backend = backend
        self.meshcat_url = meshcat_url
        self.control_page = (page_builder or build_control_page)(meshcat_url)
        self.selection_lock = selection_lock or Lock()
        self.extra_pages = _validated_pages(extra_pages)
        self.extra_json_get = _validated_routes(extra_json_get, "extra_json_get")
        self.extra_json_post = _validated_routes(extra_json_post, "extra_json_post")
        super().__init__(address, _PlaybackRequestHandler)

    @property
    def control_url(self) -> str:
        host, port = self.server_address[:2]
        visible_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
        return f"http://{visible_host}:{port}/"

    def start_background(self) -> Thread:
        """Start ``serve_forever`` in one daemon thread."""

        thread = Thread(target=self.serve_forever, name="meshcat-control", daemon=True)
        thread.start()
        return thread


class _PlaybackRequestHandler(BaseHTTPRequestHandler):
    server: MeshcatControlServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/":
            self._send_bytes(
                HTTPStatus.OK,
                self.server.control_page.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/catalog":
            payload = dict(self.server.backend.catalog())
            payload["meshcat_url"] = self.server.meshcat_url
            self._send_json(HTTPStatus.OK, payload)
            return
        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        page = self.server.extra_pages.get(path)
        if page is not None:
            self._send_bytes(
                HTTPStatus.OK,
                page.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        route = self.server.extra_json_get.get(path)
        if route is not None:
            self._run_route(route, {})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        operations = {
            "/api/select": ("select", "invalid_selection", "selection_failed"),
            "/api/evaluate": ("evaluate", "invalid_parameters", "evaluation_failed"),
        }
        operation = operations.get(path)
        extra = self.server.extra_json_post.get(path)
        if operation is None and extra is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if operation is not None:
            method = getattr(self.server.backend, operation[0], None)
            if not callable(method):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            route = JsonRoute(
                handler=method,
                serialized=True,
                invalid_error=operation[1],
                failed_error=operation[2],
            )
        else:
            route = extra
        limit = 64 * 1024 if operation is not None else 4 * 1024 * 1024
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_length"})
            return
        if length <= 0 or length > limit:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_length"})
            return
        try:
            decoded = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(decoded, Mapping):
                raise ValueError("request root must be an object")
        except (UnicodeDecodeError, ValueError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": route.invalid_error, "message": str(exc)},
            )
            return
        self._run_route(route, decoded)

    def _run_route(self, route: JsonRoute, payload: Mapping[str, object]) -> None:
        try:
            if route.serialized:
                with self.server.selection_lock:
                    result = route.handler(payload)
            else:
                result = route.handler(payload)
            self._send_json(HTTPStatus.OK, dict(result))
        except (KeyError, TypeError, ValueError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": route.invalid_error, "message": str(exc)},
            )
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": route.failed_error, "message": str(exc)},
            )

    def log_message(self, format: str, *args: object) -> None:
        """Keep the interactive CLI quiet; status is visible in the panel."""

    def _send_json(self, status: HTTPStatus, value: object) -> None:
        body = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


def build_control_page(meshcat_url: str) -> str:
    """Return the dependency-free parameter/step UI around a MeshCat iframe."""

    _validate_meshcat_url(meshcat_url)
    escaped_url = html.escape(meshcat_url, quote=True)
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decanting workspace playback</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
* {{ box-sizing: border-box; }}
html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; background: #171a1f; color: #eef1f5; }}
#layout {{ display: grid; grid-template-columns: minmax(300px, 360px) 1fr; width: 100%; height: 100%; }}
#panel {{ overflow-y: auto; padding: 14px; background: #20242b; border-right: 1px solid #3c424c; }}
#meshcat {{ width: 100%; height: 100%; border: 0; background: #111; }}
h1 {{ margin: 0 0 12px; font-size: 18px; font-weight: 600; }}
fieldset {{ margin: 0 0 12px; padding: 10px; border: 1px solid #49515d; border-radius: 7px; }}
legend {{ padding: 0 5px; color: #b8c0cc; }}
label {{ display: grid; grid-template-columns: 1fr minmax(118px, 1.2fr); align-items: center; gap: 8px; margin: 7px 0; font-size: 13px; }}
select, input, button {{ width: 100%; min-width: 0; padding: 6px 7px; border: 1px solid #596270; border-radius: 5px; background: #171a1f; color: #eef1f5; }}
input[type="range"] {{ padding: 0; }}
.numeric-control {{ display: grid; grid-template-columns: minmax(72px, 1fr) minmax(76px, 0.72fr) auto; align-items: center; gap: 6px; }}
.numeric-unit {{ color: #aeb7c3; font-size: 12px; min-width: 14px; }}
.numeric-control input:invalid {{ border-color: #d76055; }}
button {{ cursor: pointer; background: #2d5f8b; border-color: #3976aa; font-weight: 600; }}
.row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
#status {{ padding: 10px; border-radius: 7px; background: #171a1f; white-space: pre-wrap; font: 12px/1.45 ui-monospace, monospace; }}
#status[data-level="success"] {{ border-left: 4px solid #39a96b; }}
#status[data-level="failed"] {{ border-left: 4px solid #d76055; }}
#status[data-level="skipped"] {{ border-left: 4px solid #d2a33b; }}
.muted {{ color: #aeb7c3; font-size: 12px; }}
.verified {{ margin: 0 0 12px; padding: 8px 10px; border-radius: 7px; background: #193b2a; color: #9de0b9; font-size: 12px; }}
@media (max-width: 760px) {{
  #layout {{ grid-template-columns: 1fr; grid-template-rows: minmax(310px, 44%) 1fr; }}
  #panel {{ border-right: 0; border-bottom: 1px solid #3c424c; }}
}}
</style>
</head>
<body>
<main id="layout">
  <section id="panel" aria-label="사전 계산 결과 선택">
    <h1>Decanting case playback</h1>
    <p id="cacheSummary" class="verified">검증 결과를 불러오는 중…</p>
    <fieldset><legend>Robot base</legend>
      <label>UR20 X <span class="numeric-control"><input type="range" data-slider-key="ur_x" aria-label="UR20 X slider"><input type="number" step="any" data-input-key="ur_x" aria-label="UR20 X input"><span class="numeric-unit">m</span></span></label>
      <label>UR20 Y <span class="numeric-control"><input type="range" data-slider-key="ur_y" aria-label="UR20 Y slider"><input type="number" step="any" data-input-key="ur_y" aria-label="UR20 Y input"><span class="numeric-unit">m</span></span></label>
      <label>UR20 Z <span class="numeric-control"><input type="range" data-slider-key="ur_z" aria-label="UR20 Z slider"><input type="number" step="any" data-input-key="ur_z" aria-label="UR20 Z input"><span class="numeric-unit">m</span></span></label>
      <label>UR20 yaw <span class="numeric-control"><input type="range" data-slider-key="ur_yaw" aria-label="UR20 yaw slider"><input type="number" step="any" data-input-key="ur_yaw" aria-label="UR20 yaw input"><span class="numeric-unit">°</span></span></label>
      <label>SR-12iA X <span class="numeric-control"><input type="range" data-slider-key="sr_x" aria-label="SR-12iA X slider"><input type="number" step="any" data-input-key="sr_x" aria-label="SR-12iA X input"><span class="numeric-unit">m</span></span></label>
      <label>SR-12iA Y <span class="numeric-control"><input type="range" data-slider-key="sr_y" aria-label="SR-12iA Y slider"><input type="number" step="any" data-input-key="sr_y" aria-label="SR-12iA Y input"><span class="numeric-unit">m</span></span></label>
      <label>SR-12iA Z <span class="numeric-control"><input type="range" data-slider-key="sr_z" aria-label="SR-12iA Z slider"><input type="number" step="any" data-input-key="sr_z" aria-label="SR-12iA Z input"><span class="numeric-unit">m</span></span></label>
      <label>SR-12iA yaw <span class="numeric-control"><input type="range" data-slider-key="sr_yaw" aria-label="SR-12iA yaw slider"><input type="number" step="any" data-input-key="sr_yaw" aria-label="SR-12iA yaw input"><span class="numeric-unit">°</span></span></label>
    </fieldset>
    <fieldset><legend>Process case</legend>
      <label>SKU <select data-key="sku"></select></label>
      <label>Lift height <span class="numeric-control"><input type="range" data-slider-key="lift" aria-label="Lift height slider"><input type="number" step="any" data-input-key="lift" aria-label="Lift height input"><span class="numeric-unit">m</span></span></label>
      <label>Tote offset <span class="numeric-control"><input type="range" data-slider-key="tote" aria-label="Tote offset slider"><input type="number" step="any" data-input-key="tote" aria-label="Tote offset input"><span class="numeric-unit">m</span></span></label>
      <label>SR J3 stroke <span class="numeric-control"><input type="range" data-slider-key="stroke" aria-label="SR J3 stroke slider"><input type="number" step="any" data-input-key="stroke" aria-label="SR J3 stroke input"><span class="numeric-unit">m</span></span></label>
      <label>Clearance <span class="numeric-control"><input type="range" data-slider-key="clearance" aria-label="Clearance slider"><input type="number" step="any" data-input-key="clearance" aria-label="Clearance input"><span class="numeric-unit">m</span></span></label>
      <label>Corner <select data-key="corner"></select></label>
      <label>Coordination <select data-key="mode"></select></label>
    </fieldset>
    <fieldset><legend>Task pose</legend>
      <label>Step <select id="step"></select></label>
      <label>Check <select id="check"></select></label>
      <label>Sample <select id="sample"></select></label>
      <label>Ellipsoid scale <input id="ellipsoidScale" type="range" min="0.02" max="0.5" step="0.01" value="0.15"></label>
      <label><span>Ellipsoid</span><input id="showEllipsoid" type="checkbox" checked></label>
      <label><span>Wireframe ellipsoid</span><input id="ellipsoidWireframe" type="checkbox"></label>
      <div class="row"><button id="previous" type="button">Previous step</button><button id="next" type="button">Next step</button></div>
    </fieldset>
    <p class="muted">슬라이더와 직접 입력은 cache에 미리 계산된 정확한 값만 선택합니다. 없는 값은 case grid에 추가한 뒤 precompute해야 합니다.</p>
    <output id="status" aria-live="polite">Loading cache…</output>
  </section>
  <iframe id="meshcat" src="{escaped_url}" title="MeshCat robot view"></iframe>
</main>
<script>
"use strict";
const numericKeys = ["ur_x","ur_y","ur_z","ur_yaw","sr_x","sr_y","sr_z","sr_yaw","lift","tote","stroke","clearance"];
const categoricalKeys = ["sku","corner","mode"];
const parameterKeys = [...numericKeys.slice(0,8),"sku",...numericKeys.slice(8),"corner","mode"];
const selects = Object.fromEntries(categoricalKeys.map(k => [k, document.querySelector(`[data-key="${{k}}"]`)]));
const sliders = Object.fromEntries(numericKeys.map(k => [k, document.querySelector(`[data-slider-key="${{k}}"]`)]));
const numberInputs = Object.fromEntries(numericKeys.map(k => [k, document.querySelector(`[data-input-key="${{k}}"]`)]));
const stepSelect = document.getElementById("step");
const checkSelect = document.getElementById("check");
const sampleSelect = document.getElementById("sample");
const statusNode = document.getElementById("status");
const cacheSummaryNode = document.getElementById("cacheSummary");
let catalog = null;
let activeCase = null;
let requestSerial = 0;

function textValue(value, unit="") {{ return value === null ? "default" : `${{value}}${{unit}}`; }}
function flattenKey(item) {{
  const k = item.key, u = k.ur_base, s = k.sr_base;
  return {{ur_x:u.x_m,ur_y:u.y_m,ur_z:u.z_m,ur_yaw:u.yaw_deg,sr_x:s.x_m,sr_y:s.y_m,sr_z:s.z_m,sr_yaw:s.yaw_deg,sku:k.sku,lift:k.lift_height_m,tote:k.tote_offset_m,stroke:k.sr_j3_stroke_m,clearance:k.clearance_m,corner:k.corner,mode:k.coordination_mode}};
}}
function optionLabel(key, value) {{
  if (["ur_x","ur_y","ur_z","sr_x","sr_y","sr_z","lift","tote","stroke","clearance"].includes(key)) return textValue(value, value === null ? "" : " m");
  if (["ur_yaw","sr_yaw"].includes(key)) return textValue(value, "°");
  return String(value);
}}
function availableValues(key) {{
  const values = [...new Map(catalog.cases.map(c => {{ const v=flattenKey(c)[key]; return [JSON.stringify(v),v]; }})).values()];
  return numericKeys.includes(key) ? values.sort((a,b) => (a ?? -Infinity) - (b ?? -Infinity)) : values;
}}
function syncParameterControls(item) {{
  const values = flattenKey(item);
  for (const key of categoricalKeys) {{
    const all = availableValues(key);
    const select = selects[key]; select.replaceChildren();
    for (const value of all) {{ const option=document.createElement("option"); option.value=JSON.stringify(value); option.textContent=optionLabel(key,value); select.append(option); }}
    select.value=JSON.stringify(values[key]);
  }}
  for (const key of numericKeys) {{
    const all = availableValues(key);
    const slider = sliders[key], input = numberInputs[key];
    const index = all.findIndex(value => Object.is(value, values[key]));
    slider.min="0"; slider.max=String(Math.max(0,all.length-1)); slider.step="1"; slider.value=String(Math.max(0,index)); slider.disabled=all.length < 2;
    input.value=values[key] === null ? "" : String(values[key]);
    input.placeholder=values[key] === null ? "default" : "";
    input.title=`Precomputed values: ${{all.map(value => textValue(value)).join(", ")}}`;
    input.setCustomValidity("");
  }}
}}
function chooseByParameter(changedKey, wanted) {{
  const current = flattenKey(activeCase);
  let candidates = catalog.cases.filter(c => Object.is(flattenKey(c)[changedKey], wanted));
  candidates.sort((a,b) => parameterKeys.reduce((score,key) => score + (Object.is(flattenKey(a)[key], current[key]) ? -1 : 0), 0) - parameterKeys.reduce((score,key) => score + (Object.is(flattenKey(b)[key], current[key]) ? -1 : 0), 0));
  if (candidates.length) {{ activeCase=candidates[0]; syncParameterControls(activeCase); populateSteps(); submitSelection(); }}
}}
function chooseNumericValue(key, wanted) {{
  const input=numberInputs[key], all=availableValues(key);
  const exact=wanted === null ? all.find(value => value === null) : all.find(value => value !== null && Math.abs(value-wanted) <= 1e-9*Math.max(1,Math.abs(value),Math.abs(wanted)));
  if ((wanted !== null && !Number.isFinite(wanted)) || exact === undefined) {{
    const choices=all.map(value => textValue(value)).join(", ");
    const message=`${{key}}=${{input.value || "(empty)"}} is not precomputed. Available: ${{choices}}`;
    input.setCustomValidity(message); input.reportValidity();
    statusNode.dataset.level="failed"; statusNode.textContent=message;
    return;
  }}
  input.setCustomValidity(""); chooseByParameter(key,exact);
}}
function populateSteps() {{
  stepSelect.replaceChildren();
  for (const step of activeCase.steps) {{ const o=document.createElement("option"); o.value=String(step.index); o.textContent=`${{step.index}}. ${{step.name}} [${{step.status}}]`; stepSelect.append(o); }}
  populateChecks();
}}
function selectedStep() {{ return activeCase.steps.find(s => s.index === Number(stepSelect.value)); }}
function populateChecks() {{
  const step=selectedStep(); checkSelect.replaceChildren();
  (step?.checks || []).forEach((check,index) => {{ const o=document.createElement("option"); o.value=String(index); o.textContent=`${{check.name}} [${{check.status}}]`; checkSelect.append(o); }});
  checkSelect.disabled=!step || step.checks.length===0;
  if (step?.checks?.length) checkSelect.value=String(step.checks.length-1);
  populateSamples();
}}
function populateSamples() {{
  const step=selectedStep(); const check=step?.checks?.[Number(checkSelect.value)]; sampleSelect.replaceChildren();
  (check?.samples || []).forEach((sample,index) => {{ const o=document.createElement("option"); o.value=String(index); o.textContent=`${{index+1}} / ${{check.samples.length}} (f=${{sample.fraction.toFixed(3)}})`; sampleSelect.append(o); }});
  sampleSelect.disabled=!check || check.samples.length===0;
  if (check?.samples?.length) sampleSelect.value=String(check.samples.length-1);
}}
async function submitSelection() {{
  if (!activeCase) return;
  const serial = ++requestSerial;
  const request={{case_id:activeCase.id,step_index:Number(stepSelect.value),check_index:checkSelect.disabled?null:Number(checkSelect.value),sample_index:sampleSelect.disabled?null:Number(sampleSelect.value),ellipsoid_scale:Number(document.getElementById("ellipsoidScale").value),show_ellipsoid:document.getElementById("showEllipsoid").checked,ellipsoid_wireframe:document.getElementById("ellipsoidWireframe").checked}};
  try {{
    const response=await fetch("/api/select",{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify(request)}});
    const result=await response.json(); if(!response.ok) throw new Error(result.message || result.error); if(serial!==requestSerial)return;
    statusNode.dataset.level=result.step_status || "skipped"; statusNode.textContent=result.status_text || JSON.stringify(result,null,2);
  }} catch(error) {{ if(serial!==requestSerial)return; statusNode.dataset.level="failed"; statusNode.textContent=`UI error: ${{error.message}}`; }}
}}
for (const key of categoricalKeys) selects[key].addEventListener("change",()=>chooseByParameter(key,JSON.parse(selects[key].value)));
for (const key of numericKeys) {{
  sliders[key].addEventListener("input",()=>{{ const all=availableValues(key); chooseByParameter(key,all[Number(sliders[key].value)]); }});
  numberInputs[key].addEventListener("change",()=>{{ const raw=numberInputs[key].value.trim(); chooseNumericValue(key,raw === "" ? null : Number(raw)); }});
}}
stepSelect.addEventListener("change",()=>{{populateChecks();submitSelection();}});
checkSelect.addEventListener("change",()=>{{populateSamples();submitSelection();}});
sampleSelect.addEventListener("change",submitSelection);
document.getElementById("ellipsoidScale").addEventListener("change",submitSelection);
document.getElementById("showEllipsoid").addEventListener("change",submitSelection);
document.getElementById("ellipsoidWireframe").addEventListener("change",submitSelection);
document.getElementById("previous").addEventListener("click",()=>{{stepSelect.selectedIndex=Math.max(0,stepSelect.selectedIndex-1);populateChecks();submitSelection();}});
document.getElementById("next").addEventListener("click",()=>{{stepSelect.selectedIndex=Math.min(stepSelect.options.length-1,stepSelect.selectedIndex+1);populateChecks();submitSelection();}});
fetch("/api/catalog").then(r=>r.json()).then(data=>{{
  catalog=data;
  if(!data.cases.length)throw new Error("cache에 case가 없습니다");
  activeCase=data.cases.find(c=>c.id===data.default_case_id) || data.cases[0];
  const verified=Number(data.cell_feasible_case_count || 0);
  cacheSummaryNode.textContent=verified>0 ? `전체 공정 통과 case ${{verified}} / ${{data.case_count}} · 통과 case로 시작` : `전체 공정 통과 case 0 / ${{data.case_count}} · 진단 결과로 시작`;
  cacheSummaryNode.style.background=verified>0 ? "#193b2a" : "#46391a";
  cacheSummaryNode.style.color=verified>0 ? "#9de0b9" : "#f2d07a";
  syncParameterControls(activeCase);populateSteps();submitSelection();
}}).catch(error=>{{statusNode.dataset.level="failed";statusNode.textContent=error.message;}});
</script>
</body>
</html>
"""


_RESERVED_PATHS = frozenset(
    {"/", "/healthz", "/api/catalog", "/api/select", "/api/evaluate"}
)


def _validated_pages(pages: Mapping[str, str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for path, body in (pages or {}).items():
        _validate_route_path(path, "extra page")
        if not isinstance(body, str) or not body:
            raise ValueError(f"extra page {path} must be a non-empty string")
        result[path] = body
    return result


def _validated_routes(
    routes: Mapping[str, "JsonRoute"] | None,
    name: str,
) -> dict[str, "JsonRoute"]:
    result: dict[str, JsonRoute] = {}
    for path, route in (routes or {}).items():
        _validate_route_path(path, name)
        if not isinstance(route, JsonRoute):
            raise TypeError(f"{name} entries must be JsonRoute instances")
        result[path] = route
    return result


def _validate_route_path(path: object, name: str) -> None:
    if not isinstance(path, str) or not path.startswith("/") or len(path) > 128:
        raise ValueError(f"{name} path must be an absolute path under 128 characters")
    if path in _RESERVED_PATHS:
        raise ValueError(f"{name} path {path} is reserved by the legacy API")
    if any(character in path for character in " ?#"):
        raise ValueError(f"{name} path must not contain spaces, '?' or '#'")


def _validate_meshcat_url(value: str) -> None:
    parsed = urlsplit(str(value))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("meshcat_url must be an absolute http(s) URL")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("meshcat_url contains an invalid port") from exc
