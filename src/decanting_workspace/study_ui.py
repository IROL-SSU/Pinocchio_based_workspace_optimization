"""Dependency-free browser page for the user-authored UR20 check study.

The page owns only browser state.  Every pose transform, collision decision
and feasibility label comes from the local Python backend over the
``/api/study/*`` endpoints, so the browser cannot invent a result.
"""

from __future__ import annotations

import html

from .meshcat_ui import _validate_meshcat_url


def build_study_page(meshcat_url: str) -> str:
    """Return the study authoring page around a shared MeshCat iframe."""

    _validate_meshcat_url(meshcat_url)
    return _STUDY_PAGE.replace(
        "__MESHCAT_URL__", html.escape(meshcat_url, quote=True)
    )


_STUDY_PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UR20 check study</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #171a1f; color: #eef1f5; }
#layout { display: grid; grid-template-columns: minmax(430px, 520px) 1fr; width: 100%; height: 100%; }
#panel { overflow-y: auto; padding: 14px; background: #20242b; border-right: 1px solid #3c424c; }
#meshcat { width: 100%; height: 100%; border: 0; background: #111; }
h1 { margin: 0 0 6px; font-size: 19px; font-weight: 650; }
h2 { margin: 0 0 6px; font-size: 14px; font-weight: 650; color: #cfd6e0; }
a { color: #7fb6e8; }
.badge { display: inline-block; margin: 0 6px 8px 0; padding: 4px 7px; border-radius: 999px; background: #2a3550; color: #a9c7ea; font-size: 11px; font-weight: 700; letter-spacing: .04em; }
fieldset { margin: 0 0 12px; padding: 10px; border: 1px solid #49515d; border-radius: 7px; }
legend { padding: 0 5px; color: #b8c0cc; font-size: 13px; }
label { display: grid; grid-template-columns: minmax(108px, 1fr) minmax(130px, 1.2fr); align-items: center; gap: 8px; margin: 6px 0; font-size: 13px; }
label.wide { grid-template-columns: 1fr; }
select, input, button, textarea { width: 100%; min-width: 0; padding: 6px 7px; border: 1px solid #596270; border-radius: 5px; background: #171a1f; color: #eef1f5; font-family: inherit; font-size: 13px; }
input[type="checkbox"] { width: auto; justify-self: start; }
button { cursor: pointer; background: #2d5f8b; border-color: #3976aa; font-weight: 650; }
button:hover:enabled { background: #356f9f; }
button:disabled { opacity: .45; cursor: not-allowed; }
button.secondary { background: #343b45; border-color: #4d5663; }
button.danger { background: #7d3832; border-color: #a04a42; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin: 6px 0; }
.row3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin: 6px 0; }
.row4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 6px; margin: 6px 0; }
.muted { margin: 6px 0; color: #aeb7c3; font-size: 12px; line-height: 1.45; }
.mono { font: 12px/1.45 ui-monospace, monospace; }
.pose-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 4px 0 8px; }
.pose-grid span { display: grid; gap: 2px; font-size: 11px; color: #aeb7c3; }
.list { max-height: 210px; overflow-y: auto; border: 1px solid #434b57; border-radius: 6px; background: #171a1f; }
.list-item { display: grid; grid-template-columns: auto 1fr auto; gap: 7px; align-items: center; padding: 5px 7px; border-bottom: 1px solid #2b313a; font-size: 12px; }
.list-item:last-child { border-bottom: 0; }
.list-item.active { background: #2a3550; }
.list-item .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.list-item .tag { color: #9aa5b3; font-size: 11px; }
.tag.hard { color: #8fc5f0; }
.tag.soft { color: #e0c276; }
.tag.off { color: #8a929d; text-decoration: line-through; }
.target-row { display: grid; grid-template-columns: auto 1fr; gap: 7px; align-items: start; padding: 4px 7px; border-bottom: 1px solid #2b313a; font-size: 12px; }
.target-row:last-child { border-bottom: 0; }
.target-row .prim { display: block; color: #8a929d; font-size: 11px; word-break: break-all; }
.result-box { display: block; min-height: 54px; padding: 10px; border-radius: 7px; background: #171a1f; white-space: pre-wrap; font: 12px/1.45 ui-monospace, monospace; }
.result-box[data-level="success"] { border-left: 4px solid #39a96b; }
.result-box[data-level="diagnostic"] { border-left: 4px solid #d2a33b; }
.result-box[data-level="failed"] { border-left: 4px solid #d76055; }
.result-box[data-level="unknown"] { border-left: 4px solid #6f7885; }
.warn { margin: 6px 0; padding: 7px 9px; border-radius: 6px; background: #46391a; color: #f2d07a; font-size: 12px; }
.stale { margin: 6px 0; padding: 7px 9px; border-radius: 6px; background: #4a2b28; color: #f0aaa2; font-size: 12px; }
progress { width: 100%; height: 10px; }
#checkDetail[hidden], #scoutPanel[hidden] { display: none; }
@media (max-width: 900px) {
  #layout { grid-template-columns: 1fr; grid-template-rows: minmax(360px, 52%) 1fr; }
  #panel { border-right: 0; border-bottom: 1px solid #3c424c; }
}
</style>
</head>
<body>
<main id="layout">
  <section id="panel" aria-label="UR20 check study">
    <p><span class="badge">STUDY · USER-AUTHORED CHECKS</span><a href="/">&larr; live evaluation</a></p>
    <h1>UR20 check study</h1>
    <p class="muted">공정 원본 check를 불러와 pose와 collision 조건을 편집하고, 실제 Pinocchio IK·충돌 검사를 실행합니다. 결과는 <code>study_feasible</code>까지만 의미하며 전체 공정 검증이 아닙니다.</p>
    <output id="loadStatus" class="result-box" aria-live="polite">설정을 불러오는 중…</output>

    <fieldset>
      <legend>Recipe</legend>
      <label>Recipe ID <input id="recipeId" type="text" maxlength="64" spellcheck="false"></label>
      <label>Label <input id="recipeLabel" type="text" maxlength="200"></label>
      <label>Profile <select id="recipeProfile"></select></label>
      <div class="row">
        <button id="resetBaseline" class="secondary" type="button">Reload baseline checks</button>
        <button id="saveRecipe" type="button">Save to server</button>
      </div>
      <div class="row">
        <button id="downloadRecipe" class="secondary" type="button">Download JSON</button>
        <button id="uploadRecipe" class="secondary" type="button">Upload JSON…</button>
      </div>
      <input id="uploadInput" type="file" accept="application/json,.json" hidden>
      <label>Saved <select id="savedRecipes"></select></label>
      <button id="loadSaved" class="secondary" type="button">Load selected saved recipe</button>
      <p id="fingerprintNote" class="muted"></p>
    </fieldset>

    <fieldset>
      <legend>Case parameters (the effective case, hashed into the recipe)</legend>
      <p class="muted">이 값이 실제로 계산되는 case입니다. recipe에 저장되고 <code>recipe_hash</code>에 포함되므로, 불러온 recipe의 case가 조용히 시작값으로 바뀌지 않습니다. 변경하면 template/frame/target 목록을 다시 받아옵니다.</p>
      <div id="caseControls"></div>
      <label>Approach distance <span class="numeric-control"><input id="caseApproach" type="number" step="any"><span class="unit">mm</span></span></label>
      <div class="row">
        <button id="applyCase" type="button">Apply case &amp; refresh catalog</button>
        <button id="revertCase" class="secondary" type="button">Revert to recipe</button>
      </div>
      <div class="row">
        <button id="frameCell" class="secondary" type="button">Frame cell (reset camera)</button>
        <button id="applyScoutBest" class="secondary" type="button" disabled>Apply best scout base</button>
      </div>
      <output id="caseStatus" class="result-box mono" aria-live="polite">—</output>
    </fieldset>

    <fieldset>
      <legend>Checks <span id="checkCount" class="muted"></span></legend>
      <div id="checkList" class="list" role="listbox" aria-label="Authored checks"></div>
      <div class="row">
        <button id="copyCheck" class="secondary" type="button">Copy selected</button>
        <button id="removeCheck" class="danger" type="button">Remove selected</button>
      </div>
      <div class="row">
        <button id="newCheck" class="secondary" type="button">New check…</button>
        <button id="toggleAll" class="secondary" type="button">Enable / disable all</button>
      </div>
    </fieldset>

    <fieldset id="checkDetail" hidden>
      <legend>Selected check</legend>
      <label>Label <input id="checkLabel" type="text" maxlength="128"></label>
      <label>Check ID <input id="checkIdField" type="text" maxlength="64" spellcheck="false"></label>
      <label>Criticality <select id="checkCriticality"><option value="hard">HARD</option><option value="soft">SOFT</option></select></label>
      <label><span>Enabled</span><input id="checkEnabled" type="checkbox"></label>
      <label><span>In metric summary</span><input id="checkMetric" type="checkbox"></label>
      <label>Collision scenario <select id="checkScenario"></select></label>
      <label>Template <select id="checkTemplate"></select></label>
      <label>Payload <select id="checkPayload"><option value="template">Template payload + intended contacts</option><option value="none">None (free space)</option></select></label>
      <label>Pose mode <select id="checkPoseMode"><option value="template_relative">Template-relative offset</option><option value="absolute">Absolute pose in a frame</option></select></label>
      <div id="relativeControls">
        <label>Offset frame <select id="checkOffsetFrame"><option value="tcp_local">TCP local</option><option value="world">World</option></select></label>
        <p id="offsetFormula" class="muted mono"></p>
        <h2>Start offset</h2>
        <div class="pose-grid" data-pose="start_offset"></div>
        <h2>End offset</h2>
        <div class="pose-grid" data-pose="end_offset"></div>
      </div>
      <div id="absoluteControls">
        <label>Reference frame <select id="checkReferenceFrame"></select></label>
        <p class="muted mono">world_T_tcp = world_T_frame &times; Translate(XYZ) &times; Rz(yaw) Ry(pitch) Rx(roll)</p>
        <h2>Start pose</h2>
        <div class="pose-grid" data-pose="start_pose"></div>
        <h2>End pose</h2>
        <div class="pose-grid" data-pose="end_pose"></div>
        <button id="seedFromTemplate" class="secondary" type="button">Fill from template world pose</button>
      </div>
      <p class="muted">길이 입력은 mm, 회전 입력은 degree입니다. 저장과 계산은 m를 씁니다.</p>
      <output id="resolvedPose" class="result-box mono" aria-live="polite">—</output>
    </fieldset>

    <fieldset>
      <legend>Collision scenarios</legend>
      <label>Scenario <select id="scenarioSelect"></select></label>
      <div class="row">
        <button id="newScenario" class="secondary" type="button">New scenario</button>
        <button id="removeScenario" class="danger" type="button">Remove scenario</button>
      </div>
      <label>Label <input id="scenarioLabel" type="text" maxlength="128"></label>
      <label><span>INCLUDE robot self collision</span><input id="scenarioSelf" type="checkbox"></label>
      <label><span>INCLUDE floor collision</span><input id="scenarioFloor" type="checkbox"></label>
      <h2>Collision targets <span id="targetCount" class="muted"></span></h2>
      <p class="warn" id="targetLegend">아래 목록의 체크 = <strong>EXCLUDED</strong> (checker 입력에서 제외). 위의 self/floor 체크는 반대로 <strong>INCLUDE</strong>입니다. 제외는 진단 실험이며 결과가 <code>study_feasible</code>로 확정되지 않습니다.</p>
      <label>Search <input id="targetSearch" type="search" placeholder="name / role / prim path"></label>
      <label>Role <select id="targetRole"></select></label>
      <div class="row3">
        <button id="excludeShown" class="secondary" type="button">Exclude shown</button>
        <button id="keepShown" class="secondary" type="button">Keep shown</button>
        <button id="highlightShown" class="secondary" type="button">Highlight shown</button>
      </div>
      <div id="targetList" class="list"></div>
      <h2>Extra allowed contacts</h2>
      <div class="row">
        <input id="contactFirst" type="text" list="contactNames" placeholder="first name" spellcheck="false">
        <input id="contactSecond" type="text" list="contactNames" placeholder="second name" spellcheck="false">
      </div>
      <datalist id="contactNames"></datalist>
      <button id="addContact" class="secondary" type="button">Add allowed contact pair</button>
      <div id="contactList" class="list"></div>
    </fieldset>

    <fieldset>
      <legend>Run</legend>
      <div class="row">
        <button id="validateRecipe" class="secondary" type="button">Validate</button>
        <button id="previewRecipe" class="secondary" type="button">Preview in 3D</button>
      </div>
      <div class="row">
        <button id="runSelected" type="button">Evaluate selected check</button>
        <button id="runAll" type="button">Evaluate all enabled</button>
      </div>
      <div class="row">
        <button id="cancelJob" class="danger" type="button" disabled>Cancel run</button>
        <button id="clearOverlay" class="secondary" type="button">Clear 3D overlay</button>
      </div>
      <progress id="jobProgress" value="0" max="1" hidden></progress>
      <p id="jobLine" class="muted" role="status"></p>
      <label><span>Bounded scout</span><input id="scoutToggle" type="checkbox"></label>
      <div id="scoutPanel" hidden>
        <p class="muted">UR20 base 범위에서 소수 후보를 비교하는 study 지표입니다. 전체 공정 최적화 결과가 아닙니다.</p>
        <label>Mode <select id="scoutMode"><option value="grid">grid</option><option value="random">random</option></select></label>
        <div class="row4">
          <span class="muted">X min/max (mm)</span><input id="scoutXMin" type="number" step="any"><input id="scoutXMax" type="number" step="any"><input id="scoutXN" type="number" min="1" max="64" step="1" value="1">
        </div>
        <div class="row4">
          <span class="muted">Y min/max (mm)</span><input id="scoutYMin" type="number" step="any"><input id="scoutYMax" type="number" step="any"><input id="scoutYN" type="number" min="1" max="64" step="1" value="1">
        </div>
        <div class="row4">
          <span class="muted">Z min/max (mm)</span><input id="scoutZMin" type="number" step="any"><input id="scoutZMax" type="number" step="any"><input id="scoutZN" type="number" min="1" max="64" step="1" value="1">
        </div>
        <div class="row4">
          <span class="muted">Yaw min/max (deg)</span><input id="scoutYawMin" type="number" step="any"><input id="scoutYawMax" type="number" step="any"><input id="scoutYawN" type="number" min="1" max="64" step="1" value="1">
        </div>
        <div class="row">
          <label>Max candidates <input id="scoutMax" type="number" min="1" max="64" step="1" value="8"></label>
          <label>Seed <input id="scoutSeed" type="number" min="0" step="1" value="0"></label>
        </div>
        <button id="runScout" type="button">Run bounded scout</button>
      </div>
    </fieldset>

    <fieldset>
      <legend>Result</legend>
      <div id="staleNote"></div>
      <output id="resultBox" class="result-box" aria-live="polite">아직 계산하지 않았습니다.</output>
      <label>Check result <select id="resultCheck"></select></label>
      <label>Sample <select id="resultSample"></select></label>
      <button id="showPose" class="secondary" type="button" disabled>Show this joint pose in 3D</button>
      <output id="sampleBox" class="result-box mono" aria-live="polite">—</output>
    </fieldset>
  </section>
  <iframe id="meshcat" src="__MESHCAT_URL__" title="MeshCat robot view"></iframe>
</main>
<script>
"use strict";

const POLL_MS = 700;
const POSE_FIELDS = [
  {key: "x_m", label: "X (mm)", scale: 1000},
  {key: "y_m", label: "Y (mm)", scale: 1000},
  {key: "z_m", label: "Z (mm)", scale: 1000},
  {key: "roll_deg", label: "roll (deg)", scale: 1},
  {key: "pitch_deg", label: "pitch (deg)", scale: 1},
  {key: "yaw_deg", label: "yaw (deg)", scale: 1},
];

const el = id => document.getElementById(id);
const loadStatus = el("loadStatus");
const resultBox = el("resultBox");
const sampleBox = el("sampleBox");
const resolvedPose = el("resolvedPose");
const jobLine = el("jobLine");
const jobProgress = el("jobProgress");
const staleNote = el("staleNote");

let catalog = null;
let recipe = null;
let selectedCheckId = null;
let selectedScenarioId = null;
let resolvedByCheck = new Map();
let latestResult = null;
let resultRecipeHash = null;
let pollTimer = null;
let activeJobId = null;
let suppressSync = false;

function setBox(node, level, text) {
  node.dataset.level = level;
  node.textContent = text;
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload ?? {}),
  });
  let result;
  try { result = await response.json(); }
  catch (error) { throw new Error(`HTTP ${response.status}`); }
  if (!response.ok) throw new Error(result.message || result.error || `HTTP ${response.status}`);
  return result;
}

function clone(value) { return JSON.parse(JSON.stringify(value)); }

function recipePayload() {
  const copy = clone(recipe);
  copy.recipe_id = el("recipeId").value.trim();
  copy.label = el("recipeLabel").value.trim() || copy.recipe_id;
  copy.profile = el("recipeProfile").value;
  return copy;
}

const caseControls = new Map();

function buildCaseControls() {
  const container = el("caseControls");
  container.replaceChildren();
  caseControls.clear();
  const schema = catalog.parameter_schema;
  if (!schema || typeof schema !== "object") throw new Error("catalog has no parameter_schema");
  for (const [key, definition] of Object.entries(schema)) {
    const row = document.createElement("label");
    const label = document.createElement("span");
    label.textContent = definition.label ?? key;
    row.dataset.caseKey = key;
    let input;
    if (definition.kind === "choice") {
      input = document.createElement("select");
      for (const option of definition.options ?? []) {
        const node = document.createElement("option");
        node.value = JSON.stringify(option.value);
        node.textContent = String(option.label ?? option.value);
        input.append(node);
      }
    } else {
      const wrap = document.createElement("span");
      wrap.className = "numeric-control";
      input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      if (definition.nullable) input.placeholder = "default";
      const unit = document.createElement("span");
      unit.className = "unit";
      unit.textContent = definition.unit ?? "";
      wrap.append(input, unit);
      row.append(label, wrap);
      container.append(row);
      caseControls.set(key, {definition, input});
      input.addEventListener("change", () => markCaseEdited());
      continue;
    }
    input.dataset.caseInput = key;
    input.addEventListener("change", () => markCaseEdited());
    row.append(label, input);
    container.append(row);
    caseControls.set(key, {definition, input});
  }
  el("caseApproach").addEventListener("change", () => markCaseEdited());
  syncCaseControls();
}

function syncCaseControls() {
  for (const [key, control] of caseControls) {
    const stored = recipe.parameters[key];
    if (control.definition.kind === "choice") {
      control.input.value = JSON.stringify(stored);
      continue;
    }
    const scale = Number(control.definition.ui_scale ?? 1);
    control.input.value = stored === null || stored === undefined
      ? "" : String(Number(stored) * scale);
  }
  el("caseApproach").value = String(recipe.approach_distance_m * 1000);
  showCaseStatus("unknown", "현재 recipe의 case와 동일합니다.");
}

function readCaseControls() {
  const parameters = {};
  for (const [key, control] of caseControls) {
    if (control.definition.kind === "choice") {
      parameters[key] = JSON.parse(control.input.value);
      continue;
    }
    const raw = control.input.value.trim();
    if (raw === "") {
      if (!control.definition.nullable) throw new Error(`${key} is required`);
      parameters[key] = null;
      continue;
    }
    const scale = Number(control.definition.ui_scale ?? 1);
    const value = Number(raw) / scale;
    if (!Number.isFinite(value)) throw new Error(`${key} must be a finite number`);
    const min = Number(control.definition.min);
    const max = Number(control.definition.max);
    if (Number.isFinite(min) && Number.isFinite(max) && (value < min - 1e-9 || value > max + 1e-9)) {
      throw new Error(`${control.definition.label ?? key} must be within [${min * scale}, ${max * scale}] ${control.definition.unit ?? ""}`);
    }
    parameters[key] = value;
  }
  const approach = Number(el("caseApproach").value) / 1000;
  if (!Number.isFinite(approach) || approach <= 0) throw new Error("approach distance must be positive");
  return {parameters, approach_distance_m: approach};
}

function markCaseEdited() {
  showCaseStatus("diagnostic", "Case가 편집되었습니다. 'Apply case & refresh catalog'를 눌러 적용하세요.");
}

function showCaseStatus(level, message) {
  setBox(el("caseStatus"), level, message);
}

async function applyCase() {
  let wanted;
  try {
    wanted = readCaseControls();
  } catch (error) {
    showCaseStatus("failed", `Case 입력 오류: ${error.message}`);
    return false;
  }
  const previousParameters = recipe.parameters;
  const previousApproach = recipe.approach_distance_m;
  recipe.parameters = wanted.parameters;
  recipe.approach_distance_m = wanted.approach_distance_m;
  if (!await refreshCatalogForRecipe()) {
    // A case the server refuses must not be left inside the recipe: the
    // recipe would then carry a case that cannot be evaluated or saved.
    recipe.parameters = previousParameters;
    recipe.approach_distance_m = previousApproach;
    return false;
  }
  markRecipeChanged();
  return true;
}

async function refreshCatalogForRecipe() {
  try {
    const refreshed = await postJson("/api/study/catalog", {
      parameters: recipe.parameters,
      approach_distance_m: recipe.approach_distance_m,
    });
    applyCatalog(refreshed);
    renderAll();
    const resolved = await refreshResolved();
    const install = refreshed.installation;
    showCaseStatus(install.valid ? "success" : "failed", [
      `templates ${refreshed.templates.length} · targets ${refreshed.collision_targets.length} · frames ${refreshed.reference_frames.length}`,
      `installation: ${install.valid ? "valid" : "INVALID"}${install.issues.length ? " · " + install.issues.map(i => i.code).join(", ") : ""}`,
      `SR workspace: ${refreshed.sr_workspace.checked ? (refreshed.sr_workspace.feasible ? "feasible" : "infeasible") : "not checked"} (reported only)`,
      `recipe_hash: ${resolved.recipe_hash.slice(0, 16)}…`,
    ].join("\n"));
    return true;
  } catch (error) {
    showCaseStatus("failed", `Catalog 갱신 실패: ${error.message}`);
    return false;
  }
}

function request(extra) {
  // The recipe carries its own effective case, so no separate parameter
  // override is ever sent: the server would reject a conflicting one, and a
  // silently applied one would detach the result from recipe_hash.
  return Object.assign({recipe: recipePayload()}, extra || {});
}

function currentCheck() {
  return recipe.checks.find(item => item.check_id === selectedCheckId) ?? null;
}

function currentScenario() {
  return recipe.scenarios.find(item => item.scenario_id === selectedScenarioId) ?? null;
}

function markRecipeChanged() {
  if (!latestResult) return;
  staleNote.innerHTML =
    '<div class="stale">Recipe가 변경되었습니다. 아래 결과는 recipe_hash ' +
    String(resultRecipeHash).slice(0, 12) +
    '… 의 것이며 현재 편집 내용을 반영하지 않습니다 (stale).</div>';
}

// ---------------------------------------------------------------- checks

function renderCheckList() {
  const list = el("checkList");
  list.replaceChildren();
  for (const item of recipe.checks) {
    const row = document.createElement("div");
    row.className = "list-item" + (item.check_id === selectedCheckId ? " active" : "");
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = Boolean(item.enabled);
    toggle.title = "enable this check";
    toggle.addEventListener("change", () => {
      item.enabled = toggle.checked;
      renderCheckList();
      markRecipeChanged();
    });
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = item.label || item.check_id;
    name.title = `${item.check_id} · ${item.pose_mode}`;
    name.addEventListener("click", () => { selectedCheckId = item.check_id; renderCheckList(); syncCheckDetail(); });
    const tag = document.createElement("span");
    const classes = ["tag", item.criticality];
    if (!item.enabled) classes.push("off");
    tag.className = classes.join(" ");
    tag.textContent = item.criticality.toUpperCase();
    row.append(toggle, name, tag);
    list.append(row);
  }
  el("checkCount").textContent = `${recipe.checks.filter(c => c.enabled).length} enabled / ${recipe.checks.length} total`;
}

function buildPoseGrid(container, poseKey) {
  container.replaceChildren();
  const check = currentCheck();
  if (!check) return;
  for (const field of POSE_FIELDS) {
    const wrap = document.createElement("span");
    const label = document.createElement("small");
    label.textContent = field.label;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.value = String((check[poseKey]?.[field.key] ?? 0) * field.scale);
    input.addEventListener("change", () => {
      const raw = Number(input.value);
      if (!Number.isFinite(raw)) {
        setBox(resolvedPose, "failed", `${field.label} must be a finite number`);
        return;
      }
      check[poseKey] = check[poseKey] || {};
      check[poseKey][field.key] = raw / field.scale;
      markRecipeChanged();
      void refreshResolved();
    });
    wrap.append(label, input);
    container.append(wrap);
  }
}

function syncCheckDetail() {
  const check = currentCheck();
  const detail = el("checkDetail");
  detail.hidden = !check;
  if (!check) return;
  suppressSync = true;
  el("checkLabel").value = check.label ?? "";
  el("checkIdField").value = check.check_id;
  el("checkCriticality").value = check.criticality;
  el("checkEnabled").checked = Boolean(check.enabled);
  el("checkMetric").checked = check.include_in_metric_summary !== false;
  el("checkPayload").value = check.payload_source ?? "template";
  el("checkPoseMode").value = check.pose_mode;
  el("checkOffsetFrame").value = check.offset_frame ?? "tcp_local";
  el("checkScenario").replaceChildren();
  for (const scenario of recipe.scenarios) {
    const option = document.createElement("option");
    option.value = scenario.scenario_id;
    option.textContent = `${scenario.label}${scenario.is_diagnostic ? " (diagnostic)" : ""}`;
    el("checkScenario").append(option);
  }
  el("checkScenario").value = check.scenario_id;
  el("checkTemplate").replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "(no template · free space)";
  el("checkTemplate").append(none);
  for (const template of catalog.templates) {
    const option = document.createElement("option");
    option.value = template.token;
    option.textContent = template.label + (template.supplies_pose ? "" : " · payload only");
    el("checkTemplate").append(option);
  }
  el("checkTemplate").value = templateToken(check.template);
  el("checkReferenceFrame").replaceChildren();
  for (const frame of catalog.reference_frames) {
    const option = document.createElement("option");
    option.value = frame.name;
    option.textContent = frame.name;
    el("checkReferenceFrame").append(option);
  }
  el("checkReferenceFrame").value = check.reference_frame ?? "world";
  const relative = check.pose_mode === "template_relative";
  el("relativeControls").hidden = !relative;
  el("absoluteControls").hidden = relative;
  el("offsetFormula").textContent = (check.offset_frame ?? "tcp_local") === "tcp_local"
    ? "world_T_new = world_T_template × Translate(XYZ) × Rz(yaw) Ry(pitch) Rx(roll)"
    : "world_T_new = Translate(XYZ) × Rz(yaw) Ry(pitch) Rx(roll) × world_T_template";
  buildPoseGrid(document.querySelector('[data-pose="start_offset"]'), "start_offset");
  buildPoseGrid(document.querySelector('[data-pose="end_offset"]'), "end_offset");
  buildPoseGrid(document.querySelector('[data-pose="start_pose"]'), "start_pose");
  buildPoseGrid(document.querySelector('[data-pose="end_pose"]'), "end_pose");
  suppressSync = false;
  showResolved();
}

function templateToken(template) {
  if (!template) return "";
  return template.kind === "pose_check"
    ? `pose_check:${template.step_index}:${template.check_name}`
    : `step_objects:${template.step_index}`;
}

function templateFromToken(token) {
  if (!token) return null;
  const entry = catalog.templates.find(item => item.token === token);
  return entry ? clone(entry.reference) : null;
}

function emptyPose() {
  return {x_m: 0, y_m: 0, z_m: 0, roll_deg: 0, pitch_deg: 0, yaw_deg: 0};
}

function uniqueId(base) {
  let text = String(base).toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^[-_]+/, "").slice(0, 56);
  if (!text) text = "check";
  let candidate = text;
  let suffix = 2;
  const used = new Set(recipe.checks.map(item => item.check_id));
  while (used.has(candidate)) { candidate = `${text}-${suffix}`; suffix += 1; }
  return candidate;
}

function templateProvenance(resolved) {
  const entry = catalog?.templates?.find(item => item.token === resolved.template);
  return entry?.contact_provenance;
}

function showResolved() {
  const resolved = resolvedByCheck.get(selectedCheckId);
  if (!resolved) { setBox(resolvedPose, "unknown", "Validate를 눌러 해석된 world pose를 확인하세요."); return; }
  const fmt = pose => `x=${(pose.x_m * 1000).toFixed(1)} y=${(pose.y_m * 1000).toFixed(1)} z=${(pose.z_m * 1000).toFixed(1)} mm · rpy=${pose.roll_deg.toFixed(2)}/${pose.pitch_deg.toFixed(2)}/${pose.yaw_deg.toFixed(2)} deg`;
  const lines = [
    `world start: ${fmt(resolved.world_start)}`,
    `world end:   ${fmt(resolved.world_end)}`,
    `segment: ${(resolved.segment_length_m * 1000).toFixed(1)} mm`,
    `phase: ${resolved.obstacle_count} box + ${resolved.mesh_count} mesh obstacles, ${resolved.attached_count} attached`,
    `intended contacts (template): ${resolved.allowed_contacts.map(p => p.join(" ~ ")).join(", ") || "(none)"}`,
    `checker-exempt pairs (template + scenario): ${resolved.phase_allowed_contacts.map(p => p.join(" ~ ")).join(", ") || "(none beyond defaults)"}`,
    `contact provenance: ${(templateProvenance(resolved) ?? []).join(" | ") || "(authored check)"}`,
    `payload: ${resolved.payload_objects.map(o => `${o.name}${o.attached_to_tcp ? " [attached]" : (o.follows_tcp ? " [follows]" : "")}`).join(", ") || "(free space)"}`,
    `floor: ${resolved.include_floor ? "on" : "OFF"} · self collision: ${resolved.include_self_collision ? "on" : "OFF"}`,
  ];
  if (resolved.excluded_targets_applied.length) lines.push(`excluded (applied): ${resolved.excluded_targets_applied.join(", ")}`);
  if (resolved.excluded_targets_missing.length) lines.push(`excluded but absent from this phase: ${resolved.excluded_targets_missing.join(", ")}`);
  setBox(resolvedPose, resolved.is_unmodified_template ? "success" : "diagnostic", lines.join("\n"));
}

async function refreshResolved() {
  try {
    const result = await postJson("/api/study/validate", request());
    resolvedByCheck = new Map(result.resolved_checks.map(item => [item.check_id, item]));
    el("fingerprintNote").textContent = result.fingerprint_match
      ? `scene fingerprint ok (${result.expected_scene_fingerprint.slice(0, 12)}…)`
      : "scene fingerprint mismatch";
    showResolved();
    return result;
  } catch (error) {
    setBox(resolvedPose, "failed", `검증 실패: ${error.message}`);
    throw error;
  }
}

// ------------------------------------------------------------- scenarios

function renderScenarioControls() {
  const select = el("scenarioSelect");
  select.replaceChildren();
  for (const scenario of recipe.scenarios) {
    const option = document.createElement("option");
    option.value = scenario.scenario_id;
    option.textContent = `${scenario.label}${scenario.is_diagnostic ? " (diagnostic)" : ""}`;
    select.append(option);
  }
  if (!recipe.scenarios.some(item => item.scenario_id === selectedScenarioId)) {
    selectedScenarioId = recipe.scenarios[0]?.scenario_id ?? null;
  }
  select.value = selectedScenarioId ?? "";
  const scenario = currentScenario();
  if (!scenario) return;
  el("scenarioLabel").value = scenario.label;
  el("scenarioSelf").checked = scenario.include_self_collision !== false;
  el("scenarioFloor").checked = scenario.include_floor !== false;
  renderTargetList();
  renderContactList();
}

function shownTargets() {
  const query = el("targetSearch").value.trim().toLowerCase();
  const role = el("targetRole").value;
  return catalog.collision_targets.filter(target => {
    if (role && target.role !== role) return false;
    if (!query) return true;
    return [target.name, target.role, target.source, target.source_prim_path]
      .join(" ").toLowerCase().includes(query);
  });
}

function renderTargetList() {
  const scenario = currentScenario();
  const list = el("targetList");
  list.replaceChildren();
  if (!scenario) return;
  const excluded = new Set(scenario.excluded_targets);
  const shown = shownTargets();
  for (const target of shown) {
    const row = document.createElement("div");
    row.className = "target-row";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = excluded.has(target.name);
    toggle.title = "exclude this target from the checker input";
    toggle.addEventListener("change", () => {
      const current = new Set(scenario.excluded_targets);
      if (toggle.checked) current.add(target.name); else current.delete(target.name);
      scenario.excluded_targets = [...current].sort();
      scenario.is_diagnostic = undefined;
      renderScenarioControls();
      markRecipeChanged();
    });
    const body = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = target.name;
    const meta = document.createElement("span");
    meta.className = "prim";
    meta.textContent = `${target.role} · ${target.source}${target.detail ? " · " + target.detail : ""}`;
    body.append(title, meta);
    if (target.source_prim_path) {
      const prim = document.createElement("span");
      prim.className = "prim";
      prim.textContent = target.source_prim_path;
      body.append(prim);
    }
    row.append(toggle, body);
    list.append(row);
  }
  el("targetCount").textContent = `${shown.length} shown / ${catalog.collision_targets.length} total · ${excluded.size} excluded`;
}

function renderContactList() {
  const scenario = currentScenario();
  const list = el("contactList");
  list.replaceChildren();
  if (!scenario) return;
  for (const pair of scenario.extra_allowed_contacts) {
    const row = document.createElement("div");
    row.className = "list-item";
    const spacer = document.createElement("span");
    const name = document.createElement("span");
    name.className = "name mono";
    name.textContent = `${pair[0]} ~ ${pair[1]}`;
    const remove = document.createElement("button");
    remove.className = "danger";
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      scenario.extra_allowed_contacts = scenario.extra_allowed_contacts.filter(
        item => !(item[0] === pair[0] && item[1] === pair[1]));
      renderScenarioControls();
      markRecipeChanged();
    });
    row.append(spacer, name, remove);
    list.append(row);
  }
}

// ------------------------------------------------------------------ jobs

function setJobBusy(busy) {
  el("cancelJob").disabled = !busy;
  for (const id of ["runSelected", "runAll", "runScout", "validateRecipe", "previewRecipe", "saveRecipe"]) {
    el(id).disabled = busy;
  }
  jobProgress.hidden = !busy;
}

function startPolling(jobId) {
  activeJobId = jobId;
  setJobBusy(true);
  clearTimeout(pollTimer);
  pollTimer = setTimeout(pollJob, POLL_MS);
}

async function pollJob() {
  if (!activeJobId) return;
  try {
    const status = await postJson("/api/study/job", {job_id: activeJobId});
    const job = status.job;
    if (!job) { activeJobId = null; setJobBusy(false); return; }
    jobProgress.max = Math.max(1, job.total);
    jobProgress.value = job.completed;
    jobLine.textContent = `${job.kind} ${job.state} · ${job.completed}/${job.total}` +
      (job.current_label ? ` · 현재: ${job.current_label}` : "") +
      (job.message ? ` · ${job.message}` : "");
    if (job.state === "running") {
      pollTimer = setTimeout(pollJob, POLL_MS);
      return;
    }
    activeJobId = null;
    setJobBusy(false);
    if (job.state === "failed") { setBox(resultBox, "failed", `계산 실패: ${job.message}`); return; }
    if (job.result) showResult(job.result);
  } catch (error) {
    activeJobId = null;
    setJobBusy(false);
    jobLine.textContent = `진행 상태 조회 실패: ${error.message}`;
  }
}

function resultLevel(result) {
  if (result.status === "study_feasible") return "success";
  if (result.status === "study_feasible_diagnostic") return "diagnostic";
  if (result.status === "study_infeasible") return "failed";
  return "unknown";
}

function showResult(result) {
  latestResult = result;
  resultRecipeHash = result.recipe_hash;
  staleNote.innerHTML = "";
  if (result.kind === "scout") { showScoutResult(result); return; }
  const effective = result.effective ?? {};
  const parameters = effective.parameters ?? {};
  const install = result.installation ?? {valid: null, issues: []};
  const lines = [
    `Status: ${result.status}`,
    `Certification: ${result.certification}`,
    result.headline,
    "",
    `Effective case: UR20 x=${(parameters.ur_x * 1000).toFixed(0)} y=${(parameters.ur_y * 1000).toFixed(0)} z=${(parameters.ur_z * 1000).toFixed(0)} mm yaw=${parameters.ur_yaw} deg`,
    `  SKU ${parameters.sku} · corner ${parameters.corner} · lift ${(parameters.lift * 1000).toFixed(0)} mm · tote ${(parameters.tote * 1000).toFixed(0)} mm · mode ${parameters.mode}`,
    `  approach ${(Number(effective.approach_distance_m) * 1000).toFixed(0)} mm · profile ${result.profile}`,
    `Installation (independent of collision scenarios): ${install.valid === null ? "n/a" : (install.valid ? "valid" : "INVALID")}` +
      (install.issues && install.issues.length ? ` · ${install.issues.map(i => i.code).join(", ")}` : ""),
    `SR cutting workspace (reported only, not part of this certification): ${result.sr_workspace?.checked ? (result.sr_workspace.feasible ? "feasible" : "infeasible") : "not checked"}`,
    "",
    `Evaluated: ${result.evaluated_check_ids.length} check(s) in ${Number(result.elapsed_seconds).toFixed(2)} s`,
    `HARD: ${result.hard_check_ids.length} enabled · ${(result.collision_hard_check_ids ?? []).length} collision · ${(result.unresolved_hard_check_ids ?? []).length} unresolved (ik_not_found) · ${result.unevaluated_hard_check_ids.length} not calculated`,
    `SOFT failures: ${result.failed_soft_check_ids.join(", ") || "none"}`,
    `Generated HARD coverage: ${result.coverage.covers_generated_hard_set ? "complete" : "partial"} (omitted: ${result.coverage.omitted_generated_hard_checks.join(", ") || "none"})`,
    `  downgraded to SOFT: ${(result.coverage.downgraded_generated_hard_checks ?? []).join(", ") || "none"}`,
    `Skipped process steps (never generated): ${result.coverage.skipped_process_steps.join(", ") || "none"}`,
  ];
  if ((result.unresolved_hard_check_ids ?? []).length) {
    lines.push(
      "",
      "unresolved = the deterministic IK search was exhausted for that pose.",
      "It is not a proof that no solution exists, so the aggregate is UNKNOWN.",
    );
  }
  if (result.diagnostic.reasons.length) {
    lines.push("", "Diagnostic reasons:");
    for (const reason of result.diagnostic.reasons) lines.push(`  - ${reason}`);
  }
  if (result.fingerprint_override) lines.push("", "WARNING: evaluated with a scene fingerprint mismatch override.");
  lines.push("", `recipe_hash: ${result.recipe_hash}`);
  setBox(resultBox, resultLevel(result), lines.join("\n"));
  populateResultChecks(result);
}

function showScoutResult(result) {
  const describe = base =>
    `x=${(base.x_m * 1000).toFixed(0)} y=${(base.y_m * 1000).toFixed(0)} ` +
    `z=${(base.z_m * 1000).toFixed(0)} yaw=${base.yaw_deg.toFixed(1)}`;
  const strict = result.ranking_strict ?? result.ranking ?? [];
  const diagnostic = result.ranking_diagnostic ?? [];
  const lines = [
    `Bounded scout ${result.cancelled ? "cancelled" : "complete"} · ${result.candidates.length} candidate(s) evaluated in ${Number(result.elapsed_seconds).toFixed(2)} s`,
    result.interpretation,
    "",
    "Gates applied to every ranked candidate:",
  ];
  for (const gate of result.gates ?? []) lines.push(`  - ${gate}`);
  lines.push("", `Strict ranking (full collision contract) — ${strict.length} candidate(s)`);
  lines.push("rank  worst normalized sigma_min   UR20 base (mm, deg)");
  strict.forEach((row, index) => {
    lines.push(`${String(index + 1).padStart(4)}  ${Number(row.worst_normalized_min_singular_value).toExponential(4)}        ${describe(row.ur_base)}`);
  });
  if (!strict.length) lines.push("  (none)");
  if (diagnostic.length) {
    lines.push("", `Diagnostic ranking (weakened contract — NOT comparable to the strict list) — ${diagnostic.length}`);
    diagnostic.forEach((row, index) => {
      lines.push(`${String(index + 1).padStart(4)}  ${Number(row.worst_normalized_min_singular_value).toExponential(4)}        ${describe(row.ur_base)}` +
        ` · ${row.diagnostic_reasons.join("; ")}`);
    });
  }
  const skipped = result.candidates.filter(row => row.status !== "feasible");
  if (skipped.length) {
    lines.push("", "Not ranked (no metric is reported for these):");
    for (const row of skipped) {
      lines.push(`  ${row.status}: ${describe(row.ur_base)}` +
        (row.installation && !row.installation.valid
          ? ` · installation INVALID (${row.installation.issues.map(i => i.code).join(", ")})` : "") +
        (row.failed_check ? ` · ${row.failed_check.check_id} ${row.failed_check.status}` : "") +
        ((row.unresolved_checks ?? []).length ? ` · ${row.unresolved_checks.length} unresolved` : "") +
        (row.message ? ` · ${row.message}` : ""));
    }
  }
  setBox(resultBox, strict.length ? "success" : (diagnostic.length ? "diagnostic" : "unknown"), lines.join("\n"));
  el("resultCheck").replaceChildren();
  el("resultSample").replaceChildren();
  el("showPose").disabled = true;
  el("applyScoutBest").disabled = !(strict.length || diagnostic.length);
}

function populateResultChecks(result) {
  const select = el("resultCheck");
  select.replaceChildren();
  result.checks.forEach((entry, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${entry.label} [${entry.status}]`;
    select.append(option);
  });
  select.disabled = result.checks.length === 0;
  if (result.checks.length) select.value = "0";
  populateResultSamples();
}

function populateResultSamples() {
  const select = el("resultSample");
  select.replaceChildren();
  const entry = latestResult?.checks?.[Number(el("resultCheck").value)];
  (entry?.samples ?? []).forEach((sample, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index + 1}/${entry.samples.length} f=${sample.fraction.toFixed(3)} [${sample.ik_status}]`;
    select.append(option);
  });
  select.disabled = !entry || entry.samples.length === 0;
  if (!select.disabled) select.value = "0";
  showSample();
}

function showSample() {
  const entry = latestResult?.checks?.[Number(el("resultCheck").value)];
  const sample = entry?.samples?.[Number(el("resultSample").value)];
  if (!entry || !sample) { setBox(sampleBox, "unknown", "—"); el("showPose").disabled = true; return; }
  const lines = [
    `check ${entry.check_id} [${entry.criticality.toUpperCase()}] status=${entry.status} outcome=${entry.outcome ?? "?"}`,
    `scenario=${entry.scenario_id} · collision pairs=${entry.collision_pair_count}` +
      ` · floor=${entry.include_floor ? "on" : "OFF"} · self=${entry.include_self_collision ? "on" : "OFF"}`,
    entry.excluded_targets_applied.length ? `excluded targets applied: ${entry.excluded_targets_applied.join(", ")}` : "excluded targets applied: none",
    "",
    `sample fraction ${sample.fraction.toFixed(4)} · ik ${sample.ik_status} (${sample.ik_iterations} iters, seed ${sample.ik_seed_index}, ${sample.ik_rejected_by_acceptor} rejected by collision)`,
    `position error ${(sample.ik_position_error_m * 1000).toExponential(3)} mm · orientation error ${sample.ik_orientation_error_rad.toExponential(3)} rad`,
    `target x=${(sample.target.x_m * 1000).toFixed(1)} y=${(sample.target.y_m * 1000).toFixed(1)} z=${(sample.target.z_m * 1000).toFixed(1)} mm` +
      ` rpy=${sample.target.roll_deg.toFixed(2)}/${sample.target.pitch_deg.toFixed(2)}/${sample.target.yaw_deg.toFixed(2)} deg`,
    `minimum distance ${sample.minimum_distance_m === null ? "n/a" : (sample.minimum_distance_m * 1000).toFixed(2) + " mm"}` +
      ` · nearest pair ${sample.nearest_pair ? sample.nearest_pair.join(" ~ ") : "n/a"}`,
    `colliding pairs: ${sample.contacts.length ? sample.contacts.map(p => p.join(" ~ ")).join(", ") : "none"}`,
    `checker-exempt pairs: ${entry.phase_allowed_contacts.map(p => p.join(" ~ ")).join(", ") || "none beyond defaults"}`,
    `manipulability normalized sigma_min ${sample.normalized_min_singular_value === null ? "n/a" : sample.normalized_min_singular_value.toExponential(4)}` +
      ` · translational volume ${sample.translational_volume === null ? "n/a" : sample.translational_volume.toExponential(4)}` +
      ` · condition ${sample.normalized_condition_number === null ? "inf/n-a" : sample.normalized_condition_number.toExponential(3)}`,
  ];
  if (entry.joint_segment_contacts.length) {
    lines.push(`joint-segment collision pairs: ${entry.joint_segment_contacts.map(p => p.join(" ~ ")).join(", ")}`);
  }
  setBox(sampleBox, sample.ik_status === "success" && entry.success ? "success" : "failed", lines.join("\n"));
  el("showPose").disabled = !Array.isArray(sample.q);
}

// -------------------------------------------------------------- handlers

el("resetBaseline").addEventListener("click", async () => {
  recipe = clone(catalog.baseline_recipe);
  selectedCheckId = recipe.checks[0]?.check_id ?? null;
  selectedScenarioId = recipe.scenarios[0]?.scenario_id ?? null;
  el("recipeId").value = recipe.recipe_id;
  el("recipeLabel").value = recipe.label;
  renderCheckList();
  renderScenarioControls();
  syncCheckDetail();
  await refreshResolved().catch(() => {});
});

el("copyCheck").addEventListener("click", () => {
  const check = currentCheck();
  if (!check) return;
  const copy = clone(check);
  copy.check_id = uniqueId(`${check.check_id}-copy`);
  copy.label = `${check.label} (copy)`;
  recipe.checks.splice(recipe.checks.indexOf(check) + 1, 0, copy);
  selectedCheckId = copy.check_id;
  renderCheckList();
  syncCheckDetail();
  markRecipeChanged();
  void refreshResolved().catch(() => {});
});

el("removeCheck").addEventListener("click", () => {
  const check = currentCheck();
  if (!check) return;
  recipe.checks = recipe.checks.filter(item => item !== check);
  selectedCheckId = recipe.checks[0]?.check_id ?? null;
  renderCheckList();
  syncCheckDetail();
  markRecipeChanged();
});

el("newCheck").addEventListener("click", () => {
  const check = {
    check_id: uniqueId("authored-check"),
    label: "New authored check",
    criticality: "soft",
    enabled: true,
    scenario_id: selectedScenarioId ?? recipe.scenarios[0].scenario_id,
    pose_mode: "absolute",
    template: null,
    payload_source: "none",
    offset_frame: "tcp_local",
    start_offset: emptyPose(),
    end_offset: emptyPose(),
    reference_frame: "world",
    start_pose: emptyPose(),
    end_pose: emptyPose(),
    include_in_metric_summary: true,
    source_check_name: "",
  };
  recipe.checks.push(check);
  selectedCheckId = check.check_id;
  renderCheckList();
  syncCheckDetail();
  markRecipeChanged();
});

el("toggleAll").addEventListener("click", () => {
  const enable = !recipe.checks.every(item => item.enabled);
  for (const item of recipe.checks) item.enabled = enable;
  renderCheckList();
  markRecipeChanged();
});

function bindCheckField(id, apply) {
  el(id).addEventListener("change", () => {
    if (suppressSync) return;
    const check = currentCheck();
    if (!check) return;
    apply(check, el(id));
    renderCheckList();
    syncCheckDetail();
    markRecipeChanged();
    void refreshResolved().catch(() => {});
  });
}

bindCheckField("checkLabel", (check, node) => { check.label = node.value.trim() || check.check_id; });
bindCheckField("checkIdField", (check, node) => {
  const wanted = node.value.trim();
  if (!/^[a-z0-9][a-z0-9_-]{0,63}$/.test(wanted)) {
    setBox(resolvedPose, "failed", "Check ID는 소문자/숫자로 시작하고 소문자·숫자·_·- 만 쓸 수 있습니다.");
    node.value = check.check_id;
    return;
  }
  if (recipe.checks.some(item => item !== check && item.check_id === wanted)) {
    setBox(resolvedPose, "failed", `Check ID ${wanted}는 이미 있습니다.`);
    node.value = check.check_id;
    return;
  }
  check.check_id = wanted;
  selectedCheckId = wanted;
});
bindCheckField("checkCriticality", (check, node) => { check.criticality = node.value; });
bindCheckField("checkEnabled", (check, node) => { check.enabled = node.checked; });
bindCheckField("checkMetric", (check, node) => { check.include_in_metric_summary = node.checked; });
bindCheckField("checkScenario", (check, node) => { check.scenario_id = node.value; });
bindCheckField("checkPayload", (check, node) => { check.payload_source = node.value; });
bindCheckField("checkOffsetFrame", (check, node) => { check.offset_frame = node.value; });
bindCheckField("checkReferenceFrame", (check, node) => { check.reference_frame = node.value; });
bindCheckField("checkTemplate", (check, node) => {
  check.template = templateFromToken(node.value);
  if (!check.template) {
    check.pose_mode = "absolute";
    check.payload_source = "none";
  } else if (!catalog.templates.find(item => item.token === node.value).supplies_pose) {
    check.pose_mode = "absolute";
  }
});
bindCheckField("checkPoseMode", (check, node) => {
  const entry = catalog.templates.find(item => item.token === templateToken(check.template));
  if (node.value === "template_relative" && !(entry && entry.supplies_pose)) {
    setBox(resolvedPose, "failed", "Template-relative에는 pose를 제공하는 template이 필요합니다.");
    node.value = check.pose_mode;
    return;
  }
  check.pose_mode = node.value;
});

el("seedFromTemplate").addEventListener("click", () => {
  const check = currentCheck();
  const entry = catalog.templates.find(item => item.token === templateToken(check?.template));
  if (!check || !entry || !entry.start) {
    setBox(resolvedPose, "failed", "선택한 template에 world pose가 없습니다.");
    return;
  }
  check.reference_frame = "world";
  check.start_pose = clone(entry.start);
  check.end_pose = clone(entry.end ?? entry.start);
  syncCheckDetail();
  markRecipeChanged();
  void refreshResolved().catch(() => {});
});

el("scenarioSelect").addEventListener("change", () => {
  selectedScenarioId = el("scenarioSelect").value;
  renderScenarioControls();
});

el("newScenario").addEventListener("click", () => {
  const used = new Set(recipe.scenarios.map(item => item.scenario_id));
  let index = recipe.scenarios.length + 1;
  while (used.has(`scenario-${index}`)) index += 1;
  const scenario = {
    scenario_id: `scenario-${index}`,
    label: `Scenario ${index}`,
    excluded_targets: [],
    extra_allowed_contacts: [],
    include_self_collision: true,
    include_floor: true,
  };
  recipe.scenarios.push(scenario);
  selectedScenarioId = scenario.scenario_id;
  renderScenarioControls();
  syncCheckDetail();
  markRecipeChanged();
});

el("removeScenario").addEventListener("click", () => {
  const scenario = currentScenario();
  if (!scenario) return;
  if (recipe.scenarios.length <= 1) {
    setBox(resultBox, "failed", "최소 한 개의 collision scenario가 필요합니다.");
    return;
  }
  const used = recipe.checks.filter(item => item.scenario_id === scenario.scenario_id);
  if (used.length) {
    setBox(resultBox, "failed", `scenario ${scenario.scenario_id}를 쓰는 check가 ${used.length}개 있습니다. 먼저 다른 scenario로 바꾸세요.`);
    return;
  }
  recipe.scenarios = recipe.scenarios.filter(item => item !== scenario);
  selectedScenarioId = recipe.scenarios[0].scenario_id;
  renderScenarioControls();
  markRecipeChanged();
});

el("scenarioLabel").addEventListener("change", () => {
  const scenario = currentScenario();
  if (!scenario) return;
  scenario.label = el("scenarioLabel").value.trim() || scenario.scenario_id;
  renderScenarioControls();
  syncCheckDetail();
  markRecipeChanged();
});

for (const [id, key] of [["scenarioSelf", "include_self_collision"], ["scenarioFloor", "include_floor"]]) {
  el(id).addEventListener("change", () => {
    const scenario = currentScenario();
    if (!scenario) return;
    scenario[key] = el(id).checked;
    renderScenarioControls();
    markRecipeChanged();
  });
}

el("targetSearch").addEventListener("input", renderTargetList);
el("targetRole").addEventListener("change", renderTargetList);

el("excludeShown").addEventListener("click", () => {
  const scenario = currentScenario();
  if (!scenario) return;
  const current = new Set(scenario.excluded_targets);
  for (const target of shownTargets()) current.add(target.name);
  scenario.excluded_targets = [...current].sort();
  renderScenarioControls();
  markRecipeChanged();
});

el("keepShown").addEventListener("click", () => {
  const scenario = currentScenario();
  if (!scenario) return;
  const shown = new Set(shownTargets().map(item => item.name));
  scenario.excluded_targets = scenario.excluded_targets.filter(name => !shown.has(name));
  renderScenarioControls();
  markRecipeChanged();
});

el("highlightShown").addEventListener("click", async () => {
  try {
    const result = await postJson("/api/study/preview", request({
      highlight_targets: shownTargets().map(item => item.name),
      show_markers: false,
    }));
    jobLine.textContent = `3D에 ${result.highlighted}개 target을 강조했습니다.` +
      (result.missing_highlight_targets.length ? ` (형상이 없는 항목: ${result.missing_highlight_targets.join(", ")})` : "");
  } catch (error) {
    jobLine.textContent = `강조 실패: ${error.message}`;
  }
});

el("addContact").addEventListener("click", () => {
  const scenario = currentScenario();
  if (!scenario) return;
  const first = el("contactFirst").value.trim();
  const second = el("contactSecond").value.trim();
  if (!first || !second || first === second) {
    setBox(resultBox, "failed", "서로 다른 두 이름을 입력하세요.");
    return;
  }
  const known = new Set(catalog.contact_names);
  const unknown = [first, second].filter(name => !known.has(name));
  if (unknown.length) {
    setBox(resultBox, "failed", `유효하지 않은 logical/geometry 이름: ${unknown.join(", ")}`);
    return;
  }
  const pair = [first, second].sort();
  if (!scenario.extra_allowed_contacts.some(item => item[0] === pair[0] && item[1] === pair[1])) {
    scenario.extra_allowed_contacts.push(pair);
  }
  el("contactFirst").value = "";
  el("contactSecond").value = "";
  renderScenarioControls();
  markRecipeChanged();
});

el("validateRecipe").addEventListener("click", async () => {
  try {
    const result = await refreshResolved();
    const lines = [
      "Recipe는 유효합니다.",
      `recipe_hash: ${result.recipe_hash}`,
      `scene fingerprint: ${result.fingerprint_match ? "일치" : "불일치"}`,
      `generated HARD coverage: ${result.coverage.covers_generated_hard_set ? "complete" : "partial"}`,
      `omitted generated HARD: ${result.coverage.omitted_generated_hard_checks.join(", ") || "none"}`,
      `weakened collision contract: ${result.diagnostic.weakened_collision_contract ? "yes" : "no"}`,
    ];
    for (const reason of result.diagnostic.reasons) lines.push(`  - ${reason}`);
    setBox(resultBox, result.diagnostic.weakened_collision_contract ? "diagnostic" : "success", lines.join("\n"));
  } catch (error) {
    setBox(resultBox, "failed", `검증 실패: ${error.message}`);
  }
});

el("previewRecipe").addEventListener("click", async () => {
  try {
    const result = await postJson("/api/study/preview", request({frame_cell: true}));
    jobLine.textContent =
      `${result.previewed_checks.length}개 check의 marker/segment를 표시했습니다` +
      ` · scene ${result.scene_synced ? "recipe case로 갱신" : "미갱신"}` +
      ` · payload ${result.payload_display}` +
      (result.payload_display === "schematic_authored_start"
        ? " (도식 표시, 계산된 자세 아님)" : "") +
      (result.viewer_available ? "" : " (viewer 없음)");
  } catch (error) {
    jobLine.textContent = `미리보기 실패: ${error.message}`;
  }
});

el("clearOverlay").addEventListener("click", async () => {
  try { await postJson("/api/study/preview/clear", {}); jobLine.textContent = "3D overlay를 지웠습니다."; }
  catch (error) { jobLine.textContent = `overlay 삭제 실패: ${error.message}`; }
});

async function startEvaluation(checkIds) {
  try {
    staleNote.innerHTML = "";
    const result = await postJson("/api/study/evaluate", request(checkIds ? {check_ids: checkIds} : {}));
    setBox(resultBox, "unknown", "계산 중… check 경계에서 진행률이 갱신되고 취소할 수 있습니다.");
    startPolling(result.job.job_id);
  } catch (error) {
    setBox(resultBox, "failed", `계산을 시작하지 못함: ${error.message}`);
  }
}

el("runSelected").addEventListener("click", () => {
  if (!selectedCheckId) { setBox(resultBox, "failed", "check를 먼저 선택하세요."); return; }
  void startEvaluation([selectedCheckId]);
});
el("runAll").addEventListener("click", () => { void startEvaluation(null); });

el("cancelJob").addEventListener("click", async () => {
  if (!activeJobId) return;
  try { await postJson("/api/study/cancel", {job_id: activeJobId}); jobLine.textContent = "취소를 요청했습니다. 현재 check가 끝나면 중단됩니다."; }
  catch (error) { jobLine.textContent = `취소 실패: ${error.message}`; }
});

el("showPose").addEventListener("click", async () => {
  const entry = latestResult?.checks?.[Number(el("resultCheck").value)];
  const sample = entry?.samples?.[Number(el("resultSample").value)];
  if (!sample || !Array.isArray(sample.q)) return;
  // Sending the recipe and the check id places the payload with the actually
  // solved TCP pose instead of leaving a schematic box behind.  That is only
  // honest while the recipe still matches the run: a since-edited recipe has
  // a different effective case, so its scene and payload would not belong to
  // this q.  In that case only the joint configuration is shown.
  const stale = staleNote.innerHTML !== "";
  const body = stale
    ? {q: sample.q}
    : request({q: sample.q, check_id: entry.check_id});
  try {
    const result = await postJson("/api/study/show", body);
    jobLine.textContent = `성공한 관절 자세를 3D에 표시했습니다 · payload ${result.payload_display} (${result.payload_boxes} box)` +
      (stale ? " · recipe가 편집되어 payload/scene은 표시하지 않았습니다" : "");
  } catch (error) {
    jobLine.textContent = `자세 표시 실패: ${error.message}`;
  }
});

el("applyCase").addEventListener("click", () => { void applyCase(); });
el("revertCase").addEventListener("click", () => { syncCaseControls(); });

el("frameCell").addEventListener("click", async () => {
  try {
    const result = await postJson("/api/study/frame", {parameters: recipe.parameters});
    jobLine.textContent = result.framed
      ? `카메라를 cell 중심 (${result.camera.center_x_m.toFixed(2)}, ${result.camera.center_y_m.toFixed(2)}, ${result.camera.center_z_m.toFixed(2)}) m 로 맞췄습니다.`
      : `카메라를 맞추지 못함: ${result.message}`;
  } catch (error) {
    jobLine.textContent = `Frame cell 실패: ${error.message}`;
  }
});

el("applyScoutBest").addEventListener("click", () => {
  const best = latestResult?.ranking_strict?.[0] ?? latestResult?.ranking_diagnostic?.[0];
  if (!best) { jobLine.textContent = "적용할 scout 후보가 없습니다."; return; }
  for (const [key, source] of [["ur_x", "x_m"], ["ur_y", "y_m"], ["ur_z", "z_m"], ["ur_yaw", "yaw_deg"]]) {
    const control = caseControls.get(key);
    if (!control) continue;
    const scale = Number(control.definition.ui_scale ?? 1);
    control.input.value = String(best.ur_base[source] * scale);
  }
  markCaseEdited();
  jobLine.textContent = "scout 후보 base를 Case 입력에 넣었습니다. Apply case를 눌러 적용하세요.";
});

el("resultCheck").addEventListener("change", populateResultSamples);
el("resultSample").addEventListener("change", showSample);

el("scoutToggle").addEventListener("change", () => { el("scoutPanel").hidden = !el("scoutToggle").checked; });

el("runScout").addEventListener("click", async () => {
  const scout = {
    mode: el("scoutMode").value,
    x_range_m: [Number(el("scoutXMin").value) / 1000, Number(el("scoutXMax").value) / 1000],
    y_range_m: [Number(el("scoutYMin").value) / 1000, Number(el("scoutYMax").value) / 1000],
    z_range_m: [Number(el("scoutZMin").value) / 1000, Number(el("scoutZMax").value) / 1000],
    yaw_range_deg: [Number(el("scoutYawMin").value), Number(el("scoutYawMax").value)],
    counts: [Number(el("scoutXN").value), Number(el("scoutYN").value), Number(el("scoutZN").value), Number(el("scoutYawN").value)],
    max_candidates: Number(el("scoutMax").value),
    seed: Number(el("scoutSeed").value),
  };
  try {
    const result = await postJson("/api/study/scout", request({scout}));
    setBox(resultBox, "unknown", `bounded scout 시작: 후보 ${result.candidate_count}개`);
    startPolling(result.job.job_id);
  } catch (error) {
    setBox(resultBox, "failed", `scout를 시작하지 못함: ${error.message}`);
  }
});

el("saveRecipe").addEventListener("click", async () => {
  try {
    const result = await postJson("/api/study/recipe/save", request({}));
    recipe = clone(result.recipe);
    renderSavedRecipes(result.saved_recipes);
    setBox(resultBox, "success", `${result.saved.recipe_id}를 서버의 ${result.directory_name} 폴더에 저장했습니다.\nrecipe_hash: ${result.recipe_hash}`);
  } catch (error) {
    setBox(resultBox, "failed", `저장 실패: ${error.message}`);
  }
});

el("downloadRecipe").addEventListener("click", () => {
  const text = JSON.stringify(recipePayload(), null, 2);
  const blob = new Blob([text], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${el("recipeId").value.trim() || "study-recipe"}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
});

el("uploadRecipe").addEventListener("click", () => el("uploadInput").click());
el("uploadInput").addEventListener("change", async () => {
  const file = el("uploadInput").files?.[0];
  el("uploadInput").value = "";
  if (!file) return;
  if (file.size > 1024 * 1024) { setBox(resultBox, "failed", "recipe JSON은 1 MiB 이하여야 합니다."); return; }
  try {
    const document_text = await file.text();
    const result = await postJson("/api/study/recipe/import", {document: document_text});
    applyLoadedRecipe(result);
  } catch (error) {
    setBox(resultBox, "failed", `업로드 실패: ${error.message}`);
  }
});

el("loadSaved").addEventListener("click", async () => {
  const recipeId = el("savedRecipes").value;
  if (!recipeId) return;
  try {
    const result = await postJson("/api/study/recipe/load", {recipe_id: recipeId});
    applyLoadedRecipe(result);
  } catch (error) {
    setBox(resultBox, "failed", `불러오기 실패: ${error.message}`);
  }
});

function applyLoadedRecipe(result) {
  if (!result.fingerprint_match) {
    recipe = clone(result.recipe);
    renderAll();
    showCaseStatus("failed", "fingerprint 불일치: case를 적용하지 않았습니다.");
    setBox(resultBox, "failed",
      `scene/model fingerprint가 다릅니다. recipe를 편집용으로만 불러왔고 자동 적용하지 않았습니다.\n` +
      `${result.message}\n현재 scene: ${result.expected_scene_fingerprint}\nrecipe: ${result.recipe.scene_fingerprint || "(없음)"}`);
    return;
  }
  recipe = clone(result.recipe);
  renderAll();
  if (!result.applied) {
    setBox(resultBox, "failed", `recipe를 적용하지 못했습니다: ${result.resolve_error ?? result.message}`);
    return;
  }
  // The loaded recipe's own case is now the effective case; templates,
  // frames and targets must be regenerated for it.
  void refreshCatalogForRecipe().then(() => {
    const p = recipe.parameters;
    setBox(resultBox, "success", [
      `${result.recipe.recipe_id}를 불러왔습니다.`,
      `effective case: UR20 x=${(p.ur_x * 1000).toFixed(0)} y=${(p.ur_y * 1000).toFixed(0)} z=${(p.ur_z * 1000).toFixed(0)} mm yaw=${p.ur_yaw} deg`,
      `SKU ${p.sku} · corner ${p.corner} · lift ${(p.lift * 1000).toFixed(0)} mm · tote ${(p.tote * 1000).toFixed(0)} mm · mode ${p.mode}`,
      `approach ${(recipe.approach_distance_m * 1000).toFixed(0)} mm`,
      `recipe_hash: ${result.recipe_hash}`,
    ].join("\n"));
  });
}

function renderSavedRecipes(entries) {
  const select = el("savedRecipes");
  const previous = select.value;
  select.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = entries.length ? "(선택)" : "(저장된 recipe 없음)";
  select.append(none);
  for (const entry of entries) {
    const option = document.createElement("option");
    option.value = entry.recipe_id;
    option.textContent = `${entry.recipe_id} · ${entry.enabled_check_count}/${entry.check_count} checks`;
    select.append(option);
  }
  if ([...select.options].some(option => option.value === previous)) select.value = previous;
}

function applyCatalog(result) {
  catalog = result;
  const roles = [...new Set(result.collision_targets.map(item => item.role))].sort();
  const roleSelect = el("targetRole");
  const previousRole = roleSelect.value;
  roleSelect.replaceChildren();
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "(all roles)";
  roleSelect.append(all);
  for (const role of roles) {
    const option = document.createElement("option");
    option.value = role;
    option.textContent = role;
    roleSelect.append(option);
  }
  if ([...roleSelect.options].some(option => option.value === previousRole)) {
    roleSelect.value = previousRole;
  }
  const names = el("contactNames");
  names.replaceChildren();
  for (const name of result.contact_names) {
    const option = document.createElement("option");
    option.value = name;
    names.append(option);
  }
  el("fingerprintNote").textContent =
    `scene fingerprint ${result.scene_fingerprint.slice(0, 12)}… · rotation ${result.units.rotation_convention}`;
}

function renderAll() {
  if (!recipe.scenarios.length) recipe.scenarios = [clone(catalog.baseline_recipe.scenarios[0])];
  selectedCheckId = recipe.checks.some(item => item.check_id === selectedCheckId)
    ? selectedCheckId : (recipe.checks[0]?.check_id ?? null);
  selectedScenarioId = recipe.scenarios.some(item => item.scenario_id === selectedScenarioId)
    ? selectedScenarioId : recipe.scenarios[0].scenario_id;
  el("recipeId").value = recipe.recipe_id;
  el("recipeLabel").value = recipe.label;
  if ([...el("recipeProfile").options].some(option => option.value === recipe.profile)) {
    el("recipeProfile").value = recipe.profile;
  }
  renderCheckList();
  renderScenarioControls();
  syncCheckDetail();
  if (caseControls.size) syncCaseControls();
}

async function initialize() {
  try {
    const result = await postJson("/api/study/catalog", {});
    catalog = result;
    recipe = clone(result.baseline_recipe);
    el("recipeProfile").replaceChildren();
    for (const profile of result.profiles) {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = `${profile.label} · ${(profile.cartesian_translation_step_m * 1000).toFixed(0)} mm / ${profile.cartesian_rotation_step_deg.toFixed(0)}° / ${profile.joint_interpolation_samples} joint samples`;
      el("recipeProfile").append(option);
    }
    el("recipeProfile").value = result.default_profile;
    applyCatalog(result);
    buildCaseControls();
    const base = result.parameters;
    el("scoutXMin").value = String(base.ur_x * 1000 - 200);
    el("scoutXMax").value = String(base.ur_x * 1000 + 200);
    el("scoutYMin").value = String(base.ur_y * 1000);
    el("scoutYMax").value = String(base.ur_y * 1000);
    el("scoutZMin").value = String(base.ur_z * 1000);
    el("scoutZMax").value = String(base.ur_z * 1000);
    el("scoutYawMin").value = String(base.ur_yaw);
    el("scoutYawMax").value = String(base.ur_yaw);
    renderSavedRecipes(result.saved_recipes);
    renderAll();
    setBox(loadStatus, "success", [
      `templates ${result.templates.length} · collision targets ${result.collision_targets.length} · frames ${result.reference_frames.length}`,
      `baseline checks ${result.baseline_recipe.checks.length} (zero offset, references the generated workflow)`,
      `installation: ${result.installation.valid ? "valid" : "INVALID"} · SR workspace: ${result.sr_workspace.checked ? (result.sr_workspace.feasible ? "feasible" : "infeasible") : "not checked"} (reported only)`,
      `UI length mm / angle deg · stored length m · rotation ${result.units.rotation_convention}`,
      result.units.template_relative_order,
    ].join("\n"));
    await refreshResolved().catch(() => {});
  } catch (error) {
    setBox(loadStatus, "failed", `초기화 실패: ${error.message}`);
  }
}

void initialize();
</script>
</body>
</html>
"""
