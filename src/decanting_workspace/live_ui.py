"""Dependency-free browser controls for live candidate evaluation.

The page built here owns only browser state.  Pinocchio evaluation and
MeshCat updates remain in the localhost backend exposed through the catalog,
evaluate, and select JSON endpoints.
"""

from __future__ import annotations

import html

from .meshcat_ui import _validate_meshcat_url


def build_live_control_page(meshcat_url: str) -> str:
    """Return the live-parameter control page around a MeshCat iframe."""

    _validate_meshcat_url(meshcat_url)
    return _LIVE_CONTROL_PAGE.replace(
        "__MESHCAT_URL__",
        html.escape(meshcat_url, quote=True),
    )


_LIVE_CONTROL_PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decanting workspace live evaluation</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #171a1f; color: #eef1f5; }
#layout { display: grid; grid-template-columns: minmax(320px, 390px) 1fr; width: 100%; height: 100%; }
#panel { overflow-y: auto; padding: 14px; background: #20242b; border-right: 1px solid #3c424c; }
#meshcat { width: 100%; height: 100%; border: 0; background: #111; }
h1 { margin: 0 0 8px; font-size: 19px; font-weight: 650; }
.live-badge { display: inline-block; margin: 0 0 8px; padding: 4px 7px; border-radius: 999px; background: #193b2a; color: #9de0b9; font-size: 11px; font-weight: 700; letter-spacing: .04em; }
fieldset { margin: 0 0 12px; padding: 10px; border: 1px solid #49515d; border-radius: 7px; }
legend { padding: 0 5px; color: #b8c0cc; }
label { display: grid; grid-template-columns: minmax(105px, 1fr) minmax(150px, 1.35fr); align-items: center; gap: 8px; margin: 7px 0; font-size: 13px; }
select, input, button { width: 100%; min-width: 0; padding: 6px 7px; border: 1px solid #596270; border-radius: 5px; background: #171a1f; color: #eef1f5; }
input[type="range"] { padding: 0; }
input[type="checkbox"] { width: auto; justify-self: start; }
.numeric-control { display: grid; grid-template-columns: minmax(75px, 1fr) minmax(76px, .72fr) auto; align-items: center; gap: 6px; }
.unit { color: #aeb7c3; min-width: 16px; font-size: 12px; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.muted { margin: 7px 0; color: #aeb7c3; font-size: 12px; line-height: 1.4; }
a { color: #7fb6e8; }
#evaluate { background: #2d5f8b; border-color: #3976aa; cursor: pointer; font-weight: 650; }
#evaluate:hover { background: #356f9f; }
#busy { margin: 8px 0 0; color: #f0c66c; font-size: 12px; }
#elapsed { font-variant-numeric: tabular-nums; }
.result-box { display: block; min-height: 70px; padding: 10px; border-radius: 7px; background: #171a1f; white-space: pre-wrap; font: 12px/1.45 ui-monospace, monospace; }
.result-box[data-level="success"] { border-left: 4px solid #39a96b; }
.result-box[data-level="failed"] { border-left: 4px solid #d76055; }
.result-box[data-level="skipped"] { border-left: 4px solid #d2a33b; }
@media (max-width: 760px) {
  #layout { grid-template-columns: 1fr; grid-template-rows: minmax(340px, 48%) 1fr; }
  #panel { border-right: 0; border-bottom: 1px solid #3c424c; }
}
</style>
</head>
<body>
<main id="layout">
  <section id="panel" aria-label="실시간 파라미터 계산">
    <p class="live-badge">LIVE CONTROL UI · PARAMETERS + RESULTS</p>
    <h1>Decanting live evaluation</h1>
    <p class="muted">파라미터를 바꾸면 로컬 Python backend가 Pinocchio 계산을 수행하고 현재 결과를 MeshCat에 표시합니다.</p>
    <p class="muted"><a id="studyLink" href="/study">UR20 check study &rarr;</a> — check pose와 collision target을 직접 지정해 계산합니다. 같은 scene과 로봇 모델을 공유하며, 이 화면의 공정 평가 결과 의미는 바뀌지 않습니다.</p>
    <fieldset>
      <legend>Evaluation</legend>
      <label>Profile <select id="profile" aria-label="Evaluation profile"></select></label>
      <p id="profileDescription" class="muted"></p>
      <div id="parameterControls"></div>
      <label><span>Auto calculate</span><input id="autoEvaluate" type="checkbox" checked></label>
      <button id="evaluate" type="button">Calculate now</button>
      <p id="busy" role="status" hidden>계산 중… 변경 사항은 최신 한 건만 이어서 계산합니다.</p>
    </fieldset>
    <fieldset id="taskControls">
      <legend>Task pose</legend>
      <label>Step <select id="step"></select></label>
      <label>Check <select id="check"></select></label>
      <label>Sample <select id="sample"></select></label>
      <label>Ellipsoid scale <span class="numeric-control"><input id="ellipsoidScale" type="range" min="0.02" max="0.5" step="0.01" value="0.15"><output id="ellipsoidScaleValue">0.15</output><span class="unit"></span></span></label>
      <label><span>Show ellipsoid</span><input id="showEllipsoid" type="checkbox" checked></label>
      <label><span>Wireframe ellipsoid</span><input id="ellipsoidWireframe" type="checkbox"></label>
    </fieldset>
    <fieldset>
      <legend>Calculated result</legend>
      <output id="evaluationSummary" class="result-box" aria-live="polite">첫 계산을 기다리는 중…</output>
    </fieldset>
    <p class="muted">Elapsed: <output id="elapsed">—</output></p>
    <output id="status" class="result-box" aria-live="polite">설정을 불러오는 중…</output>
  </section>
  <iframe id="meshcat" src="__MESHCAT_URL__" title="MeshCat robot view"></iframe>
</main>
<script>
"use strict";

const EVALUATION_DEBOUNCE_MS = 500;
const parameterControls = document.getElementById("parameterControls");
const profileSelect = document.getElementById("profile");
const profileDescription = document.getElementById("profileDescription");
const autoEvaluate = document.getElementById("autoEvaluate");
const evaluateButton = document.getElementById("evaluate");
const taskControls = document.getElementById("taskControls");
const stepSelect = document.getElementById("step");
const checkSelect = document.getElementById("check");
const sampleSelect = document.getElementById("sample");
const ellipsoidScale = document.getElementById("ellipsoidScale");
const ellipsoidScaleValue = document.getElementById("ellipsoidScaleValue");
const showEllipsoid = document.getElementById("showEllipsoid");
const ellipsoidWireframe = document.getElementById("ellipsoidWireframe");
const busyNode = document.getElementById("busy");
const elapsedNode = document.getElementById("elapsed");
const evaluationSummary = document.getElementById("evaluationSummary");
const statusNode = document.getElementById("status");

const numericControls = new Map();
const choiceControls = new Map();
let catalog = null;
let activeCase = null;
let evaluationTimer = null;
let selectionTimer = null;
let evaluationInFlight = false;
let evaluationRerunRequested = false;
let selectionInFlight = false;
let selectionRerunRequested = false;
let selectionVersion = 0;

function setStatus(level, message) {
  statusNode.dataset.level = level;
  statusNode.textContent = message;
}

function resultValue(value) {
  if (value === true) return "PASS";
  if (value === false) return "FAIL";
  return "N/A";
}

function showEvaluationSummary(result) {
  const item = result.case;
  const level = item.cell_feasible === true
    ? "success"
    : (item.cell_feasible === false ? "failed" : "skipped");
  evaluationSummary.dataset.level = level;
  evaluationSummary.textContent = [
    `Evaluation #${result.evaluation_id ?? "?"} · profile=${result.profile ?? "?"}`,
    `Cell feasible: ${resultValue(item.cell_feasible)}`,
    `Installation valid: ${resultValue(item.installation_valid)}`,
    `Hard task checks: ${resultValue(item.scenario_hard_feasible)}`,
    `SR workspace: ${resultValue(item.sr_workspace_feasible)}`,
    `Case status: ${item.status ?? "unknown"}`,
  ].join("\n");
}

function setBusy(value) {
  busyNode.hidden = !value;
  document.body.setAttribute("aria-busy", value ? "true" : "false");
  taskControls.disabled = value;
}

function profileEntries(raw) {
  if (Array.isArray(raw)) return raw;
  if (raw && typeof raw === "object") {
    return Object.entries(raw).map(([id, value]) => (
      value && typeof value === "object" ? {id, ...value} : {id, label: String(value)}
    ));
  }
  throw new Error("profiles must be an array or object");
}

function profileId(entry) {
  if (typeof entry === "string") return entry;
  return entry.id ?? entry.value ?? entry.name;
}

function populateProfiles() {
  profileSelect.replaceChildren();
  for (const entry of profileEntries(catalog.profiles)) {
    const id = profileId(entry);
    if (id === undefined || id === null) throw new Error("profile has no id");
    const option = document.createElement("option");
    option.value = JSON.stringify(id);
    option.textContent = typeof entry === "string" ? entry : String(entry.label ?? id);
    profileSelect.append(option);
  }
  const wanted = JSON.stringify(catalog.default_profile);
  if ([...profileSelect.options].some(option => option.value === wanted)) {
    profileSelect.value = wanted;
  }
  showProfileDescription();
}

function showProfileDescription() {
  const selected = selectedProfile();
  const entry = profileEntries(catalog.profiles).find(item => profileId(item) === selected);
  profileDescription.textContent = (
    entry && typeof entry === "object" ? String(entry.description ?? "") : ""
  );
}

function scheduleParameterEvaluation() {
  if (!autoEvaluate.checked) {
    setStatus("skipped", "파라미터가 변경되었습니다. Calculate now를 누르세요.");
    return;
  }
  clearTimeout(selectionTimer);
  selectionTimer = null;
  clearTimeout(evaluationTimer);
  evaluationTimer = setTimeout(() => {
    evaluationTimer = null;
    enqueueEvaluation();
  }, EVALUATION_DEBOUNCE_MS);
}

function addNumberControl(key, schema, initialValue) {
  const row = document.createElement("label");
  const label = document.createElement("span");
  const controls = document.createElement("span");
  const slider = document.createElement("input");
  const number = document.createElement("input");
  const unit = document.createElement("span");
  row.dataset.parameterRow = key;
  label.textContent = schema.label ?? key;
  controls.className = "numeric-control";
  slider.type = "range";
  slider.dataset.parameterSlider = key;
  number.type = "number";
  number.dataset.parameterInput = key;
  for (const input of [slider, number]) {
    input.min = String(schema.min);
    input.max = String(schema.max);
    input.step = String(schema.step);
  }
  slider.value = String(initialValue ?? schema.min);
  number.value = initialValue === null || initialValue === undefined ? "" : String(initialValue);
  if (schema.nullable) number.placeholder = "default";
  unit.className = "unit";
  unit.textContent = schema.unit ?? "";
  slider.addEventListener("input", () => { number.value = slider.value; });
  slider.addEventListener("change", scheduleParameterEvaluation);
  number.addEventListener("input", () => {
    const value = Number(number.value);
    if (number.value.trim() !== "" && Number.isFinite(value)) slider.value = String(value);
  });
  number.addEventListener("change", scheduleParameterEvaluation);
  controls.append(slider, number, unit);
  row.append(label, controls);
  parameterControls.append(row);
  numericControls.set(key, {schema, slider, number});
}

function addChoiceControl(key, schema, initialValue) {
  const row = document.createElement("label");
  const label = document.createElement("span");
  const select = document.createElement("select");
  row.dataset.parameterRow = key;
  label.textContent = schema.label ?? key;
  select.dataset.parameterChoice = key;
  for (const item of schema.options ?? []) {
    const option = document.createElement("option");
    const value = item && typeof item === "object" ? item.value : item;
    option.value = JSON.stringify(value);
    option.textContent = item && typeof item === "object" ? String(item.label ?? value) : String(value);
    select.append(option);
  }
  select.value = JSON.stringify(initialValue);
  select.addEventListener("change", scheduleParameterEvaluation);
  row.append(label, select);
  parameterControls.append(row);
  choiceControls.set(key, {schema, select});
}

function populateParameterControls() {
  parameterControls.replaceChildren();
  numericControls.clear();
  choiceControls.clear();
  const schema = catalog.parameter_schema;
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    throw new Error("parameter_schema must be a key mapping");
  }
  const initial = catalog.initial_parameters;
  if (!initial || typeof initial !== "object" || Array.isArray(initial)) {
    throw new Error("initial_parameters must be a key mapping");
  }
  for (const [key, definition] of Object.entries(schema)) {
    if (definition.kind === "number") addNumberControl(key, definition, initial[key]);
    else if (definition.kind === "choice") addChoiceControl(key, definition, initial[key]);
    else throw new Error(`unsupported parameter kind for ${key}: ${definition.kind}`);
  }
}

function selectedProfile() {
  return JSON.parse(profileSelect.value);
}

function readParameters() {
  const parameters = {};
  for (const [key, control] of numericControls) {
    const raw = control.number.value.trim();
    if (raw === "" && control.schema.nullable) {
      parameters[key] = null;
      continue;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error(`${key} must be a finite number`);
    if (value < Number(control.schema.min) || value > Number(control.schema.max)) {
      throw new Error(`${key} must be in [${control.schema.min}, ${control.schema.max}]`);
    }
    parameters[key] = value;
  }
  for (const [key, control] of choiceControls) {
    parameters[key] = JSON.parse(control.select.value);
  }
  return parameters;
}

function evaluationPayload() {
  const parameters = readParameters();
  const profile = selectedProfile();
  const ellipsoid_scale = Number(ellipsoidScale.value);
  const show_ellipsoid = showEllipsoid.checked;
  const ellipsoid_wireframe = ellipsoidWireframe.checked;
  return {parameters, profile, ellipsoid_scale, show_ellipsoid, ellipsoid_wireframe};
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.message || result.error || `HTTP ${response.status}`);
  return result;
}

function enqueueEvaluation() {
  if (evaluationInFlight) {
    evaluationRerunRequested = true;
    return;
  }
  void evaluateLatest();
}

async function evaluateLatest() {
  let payload;
  try {
    payload = evaluationPayload();
  } catch (error) {
    setStatus("failed", `입력 오류: ${error.message}`);
    evaluationSummary.dataset.level = "failed";
    evaluationSummary.textContent = `계산하지 못함: ${error.message}`;
    setBusy(false);
    return;
  }
  // A selection requested for the preceding case may still be waiting for the
  // serialized backend.  Its eventual response must not overwrite this run.
  selectionVersion += 1;
  clearTimeout(selectionTimer);
  selectionTimer = null;
  selectionRerunRequested = false;
  evaluationInFlight = true;
  setBusy(true);
  setStatus("skipped", "Pinocchio 계산 중…");
  evaluationSummary.dataset.level = "skipped";
  evaluationSummary.textContent = "Pinocchio 계산 중…";
  try {
    const result = await postJson("/api/evaluate", payload);
    if (!result.case || !Array.isArray(result.case.steps)) {
      throw new Error("evaluation response has no case hierarchy");
    }
    activeCase = result.case;
    showEvaluationSummary(result);
    if (result.profile !== undefined) {
      const encoded = JSON.stringify(result.profile);
      if ([...profileSelect.options].some(option => option.value === encoded)) profileSelect.value = encoded;
    }
    populateSteps(result.selection ?? {});
    showSelection(result.selection ?? {});
    elapsedNode.textContent = Number.isFinite(Number(result.elapsed_seconds))
      ? `${Number(result.elapsed_seconds).toFixed(3)} s`
      : "—";
  } catch (error) {
    setStatus("failed", `계산 실패: ${error.message}`);
    evaluationSummary.dataset.level = "failed";
    evaluationSummary.textContent = `계산 실패: ${error.message}`;
  } finally {
    evaluationInFlight = false;
    if (evaluationRerunRequested) {
      evaluationRerunRequested = false;
      // The next run snapshots every control again, so it also subsumes a
      // debounce timer created by a later edit during the preceding run.
      clearTimeout(evaluationTimer);
      evaluationTimer = null;
      void evaluateLatest();
    } else {
      setBusy(false);
      if (selectionRerunRequested) {
        selectionRerunRequested = false;
        enqueueSelection();
      }
    }
  }
}

function selectedStep() {
  return activeCase?.steps?.find(step => String(step.index) === stepSelect.value) ?? null;
}

function populateSteps(selection = {}) {
  stepSelect.replaceChildren();
  for (const step of activeCase.steps) {
    const option = document.createElement("option");
    option.value = String(step.index);
    option.textContent = `${step.index}. ${step.name} [${step.status}]`;
    stepSelect.append(option);
  }
  stepSelect.disabled = activeCase.steps.length === 0;
  if (selection.step_index !== undefined && selection.step_index !== null) {
    stepSelect.value = String(selection.step_index);
  }
  populateChecks(selection);
}

function populateChecks(selection = {}) {
  const step = selectedStep();
  checkSelect.replaceChildren();
  (step?.checks ?? []).forEach((check, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${check.name} [${check.status}]`;
    checkSelect.append(option);
  });
  checkSelect.disabled = !step || step.checks.length === 0;
  if (!checkSelect.disabled) {
    const preferred = selection.check_index ?? step.checks.length - 1;
    checkSelect.value = String(preferred);
  }
  populateSamples(selection);
}

function populateSamples(selection = {}) {
  const step = selectedStep();
  const check = step?.checks?.[Number(checkSelect.value)];
  sampleSelect.replaceChildren();
  (check?.samples ?? []).forEach((sample, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    const fraction = Number(sample.fraction);
    const fractionText = Number.isFinite(fraction) ? fraction.toFixed(3) : "?";
    option.textContent = `${index + 1} / ${check.samples.length} (f=${fractionText})`;
    sampleSelect.append(option);
  });
  sampleSelect.disabled = !check || check.samples.length === 0;
  if (!sampleSelect.disabled) {
    const preferred = selection.sample_index ?? check.samples.length - 1;
    sampleSelect.value = String(preferred);
  }
}

function selectionPayload() {
  return {
    case_id: activeCase.id,
    step_index: stepSelect.disabled ? null : Number(stepSelect.value),
    check_index: checkSelect.disabled ? null : Number(checkSelect.value),
    sample_index: sampleSelect.disabled ? null : Number(sampleSelect.value),
    ellipsoid_scale: Number(ellipsoidScale.value),
    show_ellipsoid: showEllipsoid.checked,
    ellipsoid_wireframe: ellipsoidWireframe.checked,
  };
}

function showSelection(selection) {
  const level = selection.step_status ?? selection.status ?? "skipped";
  const message = selection.status_text ?? JSON.stringify(selection, null, 2);
  setStatus(level, message);
}

function scheduleSelection() {
  if (evaluationInFlight || evaluationTimer !== null) {
    selectionRerunRequested = true;
    return;
  }
  if (!activeCase) return;
  clearTimeout(selectionTimer);
  selectionTimer = setTimeout(() => {
    selectionTimer = null;
    enqueueSelection();
  }, EVALUATION_DEBOUNCE_MS);
}

function enqueueSelection() {
  if (selectionInFlight) {
    selectionRerunRequested = true;
    return;
  }
  void submitSelection();
}

async function submitSelection() {
  if (!activeCase) return;
  const version = selectionVersion;
  const payload = selectionPayload();
  selectionInFlight = true;
  try {
    const result = await postJson("/api/select", payload);
    if (version === selectionVersion && payload.case_id === activeCase?.id) {
      showSelection(result.selection ?? result);
    }
  } catch (error) {
    if (version === selectionVersion) {
      setStatus("failed", `표시 실패: ${error.message}`);
    }
  } finally {
    selectionInFlight = false;
    if (selectionRerunRequested) {
      selectionRerunRequested = false;
      if (!evaluationInFlight && evaluationTimer === null) {
        void submitSelection();
      }
    }
  }
}

stepSelect.addEventListener("change", () => { populateChecks(); scheduleSelection(); });
checkSelect.addEventListener("change", () => { populateSamples(); scheduleSelection(); });
sampleSelect.addEventListener("change", scheduleSelection);
profileSelect.addEventListener("change", () => {
  showProfileDescription();
  scheduleParameterEvaluation();
});
ellipsoidScale.addEventListener("input", () => { ellipsoidScaleValue.value = ellipsoidScale.value; });
ellipsoidScale.addEventListener("change", scheduleSelection);
showEllipsoid.addEventListener("change", scheduleSelection);
ellipsoidWireframe.addEventListener("change", scheduleSelection);
evaluateButton.addEventListener("click", () => {
  clearTimeout(evaluationTimer);
  evaluationTimer = null;
  clearTimeout(selectionTimer);
  selectionTimer = null;
  enqueueEvaluation();
});
autoEvaluate.addEventListener("change", () => {
  if (autoEvaluate.checked) scheduleParameterEvaluation();
  else {
    clearTimeout(evaluationTimer);
    evaluationTimer = null;
    evaluationRerunRequested = false;
    const replayPending = selectionRerunRequested;
    selectionRerunRequested = false;
    if (replayPending) scheduleSelection();
  }
});

async function initialize() {
  try {
    const response = await fetch("/api/catalog", {headers: {"Accept": "application/json"}});
    const result = await response.json();
    if (!response.ok) throw new Error(result.message || result.error || `HTTP ${response.status}`);
    catalog = result;
    populateProfiles();
    populateParameterControls();
    enqueueEvaluation();
  } catch (error) {
    setStatus("failed", `초기화 실패: ${error.message}`);
    evaluationSummary.dataset.level = "failed";
    evaluationSummary.textContent = `초기화 실패: ${error.message}`;
  }
}

void initialize();
</script>
</body>
</html>
"""
