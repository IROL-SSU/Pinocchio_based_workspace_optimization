# Pinocchio-based decanting workspace

Python-only scene, task, and robot-model foundation for the decanting-cell base-placement study. The runtime is independent of Isaac Sim and ROS/MoveIt: one validated `SceneSnapshot` feeds Pinocchio FK/IK/Jacobians, Coal collision checks, conservative SR task-workspace screening, and the reusable base-candidate evaluator.

For continuation work, read [the code architecture](docs/ARCHITECTURE.md),
[the project handoff](docs/HANDOFF.md), and the Korean
[run-command guide](docs/RUN_COMMANDS.md). The handoff records confirmed
process requirements, current results, unresolved decisions, and the
fresh-machine checklist.

Implemented in this stage:

- direct USD/USDA/USDC environment loading with triangle-mesh visualization and collision, including parent transforms, units and instances (see [USD scenes](docs/USD_SCENES.md));
- fixed conveyor, worktable, and camera-support proxies extracted from `cu_usd_simplified.usd`;
- a synthesized `World Z = 0` floor;
- the `1.350 x 1.000 x 0.150 m` pallet and `0...0.760 m` lift range;
- all five real box sizes and the four alternative lowest-layer corner samples;
- one representative `0.660 x 0.440 x 0.300 m` tote;
- independently movable UR20 and SR-12iA bases, including mounting height and pedestal;
- the official mesh-based UR20 description as the visualization default;
- a local FANUC/Isaac SR-12iA mesh preparation path and a corrected conservative primitive fallback;
- a provisional UR20 suction proxy; the SR-12iA currently has no process tool;
- an SR-12iA `tool0` cutting-task footprint tied to the support-cube top;
- nominal display poses aimed toward the pallet (UR20), with the SR-12iA
  `tool0` on the opposite-elbow branch, 0.10 m beyond `UncasingLoadFrame`
  toward the worktable, 0.02 m inside the configured task workspace's
  conveyor-side edge, and aligned to `UncaseGroup +Y`;
- a representative tote rotated 90 degrees so its long axis follows the
  worktable's short axis;
- the eight-step UR20 process as phase-specific poses and moving-object states;
- fixed-base, arm-only Pinocchio IK and Jacobian-ellipsoid metrics;
- deterministic IK branch retry when a numerically valid endpoint branch has
  a colliding local joint interpolation segment;
- explicit robot/self/environment/payload collision pairs (the URDFs load
  with geometry but no collision-pair list);
- separate simultaneous and sequential scenarios, with a conservative
  Pinocchio-derived SR concurrent exclusion OBB for steps 3 and 4;
- conservative SR box-top checks against the configured footprint, planar 2R
  joint ranges, and the selected 300/450 mm J3 range;
- an objective-free candidate API and compact JSON report CLI ready for a
  later optimization loop;
- an explicit finite case-grid cache containing replayable Pinocchio `q`, TCP
  poses, collision diagnostics, and manipulability-ellipsoid axes;
- an immutable cache-playback panel that performs no IK during replay;
- a separate on-demand control panel that evaluates one parameter set in the
  local Python process and immediately presents its in-memory result in
  Meshcat.

## UR20 check study

`decanting-live`를 실행한 뒤 `http://127.0.0.1:8766/study`를 열면 check pose와 collision scenario를 직접 편집할 수 있다. 기존 live 화면의 **UR20 check study** 링크로도 들어갈 수 있다. `--port`를 지정했다면 같은 port의 `/study`를 사용한다.

1. `Case parameters`에서 base·SKU·corner 등의 값을 정하고 **Apply case & refresh catalog**로 적용한다.
2. 기존 check를 복제하고 TCP pose, payload template, collision scenario를 편집한다. 화면 길이는 mm, 저장 길이는 m다.
3. **Preview in 3D**로 marker를 확인하고 **Evaluate selected check** 또는 **Evaluate all enabled**로 계산한다. **Frame cell**은 camera를 cell에 맞춘다.
4. **Save to server**로 `outputs/study_recipes/`에 recipe를 저장한다. JSON download/upload와 저장본 재로딩도 지원한다.

Collision target checkbox는 **켜면 제외**이며 제외·추가 contact 허용은 diagnostic 결과로 구분한다. 설치 조건은 별도로 유지한다. Bounded scout는 설치 조건과 활성 HARD check를 통과한 후보를 비교하되 strict·diagnostic 순위를 분리한다. IK 탐색에서 해를 못 찾은 상태는 `UNKNOWN`이며, 전체 cycle 불가능의 증명이 아니다.

이 화면은 사용자 check의 기구학·충돌 study다. 전체 공정 간 transfer planning, pouring 자동 생성, 모든 SKU·오차를 포함한 production optimizer는 후속 범위다. [Proposal·GUI 사용법](https://github.com/dhlee04/research-wiki/blob/master/docs/%5BCodex%5D%20UR20_Study_GUI_%EC%82%AC%EC%9A%A9%EB%B2%95_2026-10-03.md)은 연구 문서 저장소 접근 권한이 필요하다.

## Fresh-machine setup

Pixi is not used. Clone the repository at any filesystem location. BGF is
reference material rather than a runtime dependency; initialize only its root
submodule because the current BGF revision contains an incomplete nested
submodule declaration:

```text
git clone REPOSITORY_URL
cd Pinocchio_based_workspace_optimization
git submodule update --init BGF
```

Create the named Miniconda environment and install this checkout in
editable mode:

```text
conda env create -n pinocchio-workspace --file environment.yml
conda activate pinocchio-workspace
python -m pip install -e . --no-deps --no-build-isolation
```

Prepare the licensed SR mesh and validate both portable robot bundles, the
scene, the tracked reports, and both tracked result caches:

```text
decanting-setup --accept-fanuc-license
python -m pytest
```

The acceptance flag is intentional: it downloads the pinned source directly
from NVIDIA, verifies SHA-256, converts it locally, and never commits FANUC
geometry. If an authorized `geometries.usd` is already available, use
`decanting-setup --sr12ia-usd path/to/geometries.usd` instead. Re-running
`decanting-setup` without either option only validates an existing bundle.

Without shell activation, use `conda run -n pinocchio-workspace COMMAND`. On
Windows, use the activated environment or `conda run` so Conda's DLL paths are
configured. Set `PYTHONUTF8=1` when working from a non-ASCII checkout path.
All repository-owned paths are resolved through the installed module rather
than a username, drive, or checkout folder name. If Python is installed
non-editably, set `DECANTING_WORKSPACE_ROOT` to the complete cloned repository
before running the commands.

## Use a USD environment

Download the USD archive and asset files from
http://irol.iptime.org/public/share/fmJieNofpXugLi-i2Juqvw, then place the
downloaded ZIP file and the extracted USD files under the repository's `USD/`
directory. The full-cell environment should be available as
`USD/cu_usd_simplified.usd` before running the commands below.

Read an unsimplified polygon-mesh environment directly while retaining the
configured task frames and robot URDFs:

```powershell
decanting-live --usd path/to/environment.usd --open
```

See [USD scenes](docs/USD_SCENES.md) for environment selection, conveyor/table
bindings, supported geometry, and cache regeneration. The scene-only viewer,
evaluator, precompute, playback, and setup commands also accept `--usd`.
For this repository's full cell at `USD/cu_usd_simplified.usd`, use
`decanting-live --config config/cell_usd.yaml --open` to retain its fixed supports
and omit missing camera/floor payloads and generated moving objects.

## Robot assets

### UR20

`--ur20-source official` is the default. A portable bundle of the Universal
Robots ROS 2 description release 4.3.1 at immutable commit
`ae333289875f9ba5a9ea6649a54036efb5ccabee` is stored under
`assets/robots/ur20/official`. It contains the relative-path URDF, DAE visual
meshes, STL collision meshes, both applicable license texts, and a SHA-256
provenance manifest. The default resolver validates every file and works
offline; it no longer generates a machine-specific URDF under `.cache`.

For a deterministic mesh-free fallback, use `--ur20-source primitive`. An explicit model always takes precedence:

```text
python -m decanting_workspace --ur20-urdf path/to/ur20.urdf --open
```

The suction tool dimensions are provisional and live in `config/cell_nominal.yaml`. With the official UR20 they are attached to `tool0` both as a viewer primitive and as two collision geometries. Loading with `suction_proxy=...` also creates a `suction_tcp` operational frame at the configured contact plane; use that frame, not the bare flange `tool0`, for IK and Jacobians.

### FANUC SR-12iA

FANUC provides product outline CAD to registered MyFANUC users. Isaac Sim 6.0
also contains a link-separated FANUC SR-12iA asset at
`Isaac/Robots/Fanuc/sr12ia/sr12ia.usd`. Either convert an authorized local
source:

```text
decanting-assets path/to/Fanuc/sr12ia/payloads/geometries.usd
```

or explicitly accept the linked terms and download the pinned NVIDIA source:

```text
decanting-assets --download --accept-fanuc-license
```

Both paths write eight OBJ files, a relative-path four-axis URDF, and a
provenance manifest to the ignored directory `.cache/robot_assets/sr12ia`.
The pinned input digest lives in
`assets/robots/sr12ia/download_manifest.json`. The bundle can be moved with
the checkout and is revalidated before use. The supplied Isaac asset is the
standard 300 mm J3 model, so its generated URDF is limited to `0...0.300 m`.
The scene viewer uses it only for the 300 mm scenario; 450 mm still requires
the corrected provisional model or an explicit matching manufacturer model.

The fallback follows the real 2R-P-R skeleton: `L1 = L2 = 0.450 m`, the arm plane is `0.336 m` above the base, J3 moves downward through `0...0.450 m`, and J4/tool0 has no artificial translational offset. At zero joint position, `tool0 = [0.900, 0, 0.336] m`. Its shaft uses the longer 450 mm-option height for both choices, conservatively over-approximating the 300 mm fallback until the corresponding manufacturer asset is available. No process tool or provisional cutter is attached to the current SR model.

The default SR guide is deliberately **not** the arm's full `0.9 m` mechanical reach or a swept-link cylinder. It is the permitted XY target region for `tool0`: the oriented top footprint of the SR support cube, inset only by the selected `--clearance-mm`. Candidate evaluation places the real SKU on `UncasingLoadFrame`, checks its four top corners against this footprint, verifies a joint-limit-valid planar 2R branch, and checks cutting Z against the selected J3 range. No cutter offset is assumed. During simultaneous UR work, a separate OBB encloses both this configured task region and the SR collision geometry at its designated cutting posture; sequential mode omits that dynamic OBB while retaining the fixed pedestal and camera supports. An exact SR cutting path and final tool collision model remain later inputs.

The Universal Robots graphical documentation and FANUC/Isaac assets have
separate licenses. The UR terms are distributed with the vendored bundle as
required. The FANUC 3D Content Sharing Agreement restricts standalone
redistribution, so derived SR mesh files stay under `.cache` and are not
committed. A clone is made complete through the explicit setup step rather
than by copying licensed geometry into Git.

Sources: [Universal Robots description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description), [FANUC SR-12iA specifications](https://www.fanucamerica.com/products/robot/sr-12ia), [FANUC CAD download information](https://www.fanuc.co.jp/ja/product/outlinedata.html), [Isaac Sim robot asset catalog](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html), and [FANUC 3D Content Sharing Agreement](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/_downloads/8ea27bd6ad258f4242d675e67e8f7b58/3D%20Content%20Sharing%20Agreement_FANUC.pdf).

## Visualize a scenario

Open a scene-only interactive Meshcat view. This materializes the selected
environment and nominal robot poses; it is not `decanting-live` candidate
evaluation and it does not replay a result cache:

```powershell
python -m decanting_workspace --open
```

Or save a self-contained review snapshot:

```powershell
python -m decanting_workspace `
  --ur20-source primitive `
  --sr12ia-source approximate `
  --lift-height-mm 380 `
  --sku 049995 `
  --sr-j3-stroke-mm 450 `
  --save-html outputs\cell-lift-380.html
```

Self-contained Meshcat HTML embeds mesh bytes. The CLI therefore rejects
static export when the pinned UR20 or prepared FANUC asset is selected; use the
live `--open` view for those models. Explicit third-party URDF licensing remains
the caller's responsibility.

Candidate/scenario controls are:

```text
--ur-base X_M Y_M Z_M YAW_DEG
--sr-base X_M Y_M Z_M YAW_DEG
--lift-height-mm 0..760
--sku 123591|430060|180043|049995|000509
--corner all|southwest|southeast|northwest|northeast
--sr-j3-stroke-mm 300|450
--clearance-mm VALUE
--tote-table-offset-mm VALUE
```

`--corner all` shows four translucent alternative samples; it does not create four simultaneous collision boxes. Selecting one corner creates one physical box scenario.

`--tote-table-offset-mm` translates both the representative tote and its
`ToteLoadFrame` along the supporting worktable's long axis. For the nominal
geometry, positive offset is World `+X` and the valid range is approximately
`-273.355...1086.645 mm`; values outside it are rejected so the tote does not
cross a long table edge.

## Evaluate a base candidate

The evaluation command has no viewer in its loop. It uses the pinned official
UR20 collision mesh by default and writes a compact report without storing the
large per-sample joint vectors or Jacobian matrices:

```powershell
python -m decanting_workspace.candidate_cli `
  --sku 123591 `
  --corner all `
  --coordination both `
  --lift-height-mm 0 `
  --sr-j3-stroke-mm 300 `
  --output-json outputs\candidate.json
```

The task contract is:

1. top-down pick from each selected pallet corner independently;
2. top-down placement at `UncasingLoadFrame`, with World-Z yaw `+180 deg`;
3. front pick of the level-2 empty tote, followed by a vertical lift;
4. placement at the movable `ToteLoadFrame`;
5. front pick of the opened box from its UR-nearest vertical face;
6. pouring sequence marker only (no pose test yet);
7. optional level-3 waste fit and release IK (Soft; never rejects a base);
8. push from the conveyor-opposite tote face to the exact
   `ToteSupplyFrame` endpoint, then independently verify the final pushing
   posture at that endpoint even if the sampled push path stops early.

The independent final-pose sample retains its collision result and
manipulability ellipsoid for UI inspection, but is excluded from scenario
metric-summary means because the same endpoint is already sampled by the push
segment. This prevents accidental double weighting in a later objective.

Steps 1--5 and 8 are Hard. A coordination result is UR-feasible only when all
selected corner alternatives satisfy every Hard check. `cell_feasible` also
requires the SR cutting-workspace report to pass. Simultaneous and sequential
results are kept separate.

Every target is `World_T_suction_tcp`. The pallet top-down rotation is obtained
by Pinocchio FK at the candidate base and nominal arm pose. The floating root
is then frozen exactly: IK and manipulability use only the six UR arm columns.
The reported ellipsoid inputs include translational singular values/volume and
a dimensionally normalized 6-D metric using a configurable characteristic
length; no weights, aggregation objective, or candidate ranking are imposed.

Approach segments are sampled in Cartesian space and adjacent IK solutions are
sampled in joint space for collision diagnostics. The solver is deterministic
multi-start damped least squares. Therefore `not_found` means that this search
did not find a valid solution; it is not a proof of global IK or motion-planning
infeasibility. Long station-to-station global paths, the pour pose, exact SR
cutting trajectory, payload dynamics, suction limits, and push forces are not
claimed by this stage.

The checked nominal report is
[`outputs/nominal_candidate_all_corners.json`](outputs/nominal_candidate_all_corners.json).
With the settings recorded inside that file, pallet pick and Uncasing placement
pass for all four corner alternatives. The nominal SR base passes footprint and
planar reach but its arm plane is `0.107659328 m` below the SKU 123591 box top,
so its vertical task check fails. Sequential UR handling then fails at the
current tote placement, opened-box side-pick, and tote-push poses because the
reported IK branches collide with the worktable/camera support or robot itself.
Simultaneous mode additionally fails tote pickup against the conservative SR
reserved volume. These are diagnostics for the starting layout, not an
optimized result or a global infeasibility proof.

## Evaluate parameters live in Meshcat

The live mode is separate from immutable cache playback. It evaluates exactly
one parameter set on demand, converts that result to the existing
step/check/sample hierarchy in memory, and displays it immediately:

```powershell
decanting-live --open
```

Use the URL labeled `LIVE CONTROL UI (parameters + calculated results)`,
normally `http://127.0.0.1:8766/`. The separate Meshcat `/static/` URL is only
the embedded 3-D viewer and intentionally has no parameter controls. When
running without an activated shell, preserve the live URL output with
`conda run --no-capture-output -n pinocchio-workspace decanting-live --open`.

`config/case_grid_working.yaml` is used only to supply the first parameter
values shown when the page opens. `--initial-grid PATH` can select another
grid for those initial values, but the command does not load or write a result
cache. Continuous controls cover both robot base poses, lift height, tote
offset, and clearance; SKU, pallet corner, coordination mode, and SR J3 stroke
are explicit choices. After one evaluation, moving among its steps, checks,
and samples only replays that latest in-memory result and does not recompute
IK. The Task pose panel also has an unchecked `Wireframe ellipsoid` control;
toggling it, ellipsoid visibility, or scale only selects a new display state
and does not run another evaluation. These display controls are separate from
the 15 calculation parameters.

Both live and cached selection APIs accept `ellipsoid_wireframe` as a strict
JSON boolean. It defaults to `false`, preserving the translucent solid style.
An unchanged style reuses the existing sphere geometry and material; switching
between solid and wireframe rebuilds the display material but does not repeat
the calculation.

The page starts with automatic calculation enabled. Parameter edits are
debounced for 500 ms, only one calculation runs at a time, and edits made
while it is running collapse to the latest pending parameter set. Disable
auto calculation to stage several edits and use **Calculate now** explicitly.
The backend also serializes evaluate/select requests because the collision
checker and mutable SR J3 limit are not shared-worker-safe.

Two profiles are available:

- `quick` is the default responsive diagnostic. It uses endpoint-oriented
  `10 m` translation and `180 deg` rotation steps with no interior joint
  samples. It does **not** establish a continuous collision-free path.
- `full` uses the configured local sampling, by default `50 mm`, `5 deg`, and
  three interior joint-segment samples. It is more detailed than `quick`, but
  remains a local diagnostic and is **not** a global motion planner or a proof
  that a station-to-station path exists.

Open with `full` selected, optionally overriding its sampling settings:

```powershell
decanting-live `
  --default-profile full `
  --cartesian-step-mm 25 `
  --cartesian-rotation-step-deg 2.5 `
  --joint-interpolation-samples 5 `
  --open
```

The official UR20 model is used for evaluation and display by default. The
locally prepared FANUC/Isaac SR-12iA asset is display-only. Live feasibility,
including a selected `450 mm` J3 study, uses the configured conservative
300/450 mm proxy (or an explicit `--sr12ia-urdf` evaluation proxy), never the
prepared mesh. The prepared visual has a native 300 mm J3 range; a displayed
posture outside that range is rejected rather than clamped. Use
`--sr12ia-visual-urdf` only to change the visual model, or
`--allow-sr-visual-fallback` to intentionally display the evaluation proxy
when the prepared asset is unavailable.

## Precompute cases and replay them in Meshcat

### Verified commissioning setup

`config/case_grid_working.yaml` is the reproducible opening setup for the UI:

- UR20 base: `[-1.300, -7.350, 0.850, 90 deg]`
- SR-12iA base: `[-0.300, -7.073643684, 0.665, -90 deg]`
- all five SKUs, pallet lift `0.200 m`, tote offset `+0.700 m`
- SR J3 stroke `0.450 m`

The generated `outputs/precomputed_working_cases.json.gz` uses the official
UR20 collision meshes and the normal `50 mm` Cartesian, `5 deg` rotation, and
three-sample joint-segment checks. All 20 sequential SKU/corner cases pass UR20
HARD steps 1--5 and 8. Twelve also pass the SR cutting-workspace check: all four
corners for SKUs `123591`, `430060`, and `000509`. Step 7 remains SOFT and may
fail IK, as permitted by the process definition. The 20 simultaneous cases are
retained intentionally: tote handling fails against the conservative SR
exclusion box, so they are diagnostic rather than presented as feasible.

Regenerate the verified cache with:

```powershell
decanting-precompute `
  --grid config\case_grid_working.yaml `
  --output-json outputs\precomputed_working_cases.json.gz
```

Open it directly; `--cache` now defaults to that verified cache:

```powershell
decanting-playback --open
```

The panel reports `12 / 40` full-cell-feasible cases and opens the first passing
sequential case. The smallest observed collision clearance is approximately
`6.464 mm` at the box/SR support boundary, so this is a simulation
commissioning point rather than a robust production clearance. With the
current no-tool-offset SR model and solid floor-to-base pedestal proxy, SKUs
`180043` and `049995` cannot satisfy SR vertical reach without the pedestal
entering the box support volume. They must therefore be revisited when the
cutter/tool offset or the mounting/support geometry is defined; the preset does
not hide that model limitation.

### Explicit exploration grid

Continuous base placement and lift/tote motion do not have a finite set of
"all" values. `config/case_grid_nominal.yaml` therefore declares every value
that is precomputed. Its current 720 exact cases cover all five known SKUs,
three lift heights, three tote-frame positions, four independent pallet
corners, both coordination modes, and both SR J3 options. UR20 and SR-12iA are
nominal-only because their search bounds have not yet been supplied. Add base
axis values to that YAML when the bounds are fixed; the exact Cartesian-product
count is printed before either robot is loaded.

Create a full path-sampled cache with the official UR20 collision model:

```powershell
decanting-precompute `
  --grid config\case_grid_nominal.yaml `
  --output-json outputs\precomputed-cases.json.gz
```

The checked-in workspace also contains the generated
`outputs/precomputed_nominal_cases.json.gz`. It is deliberately a task-pose
playback cache: each Cartesian check stores its start/end samples, while joint
interpolation is disabled. Its settings are preserved inside the cache and it
must not be presented as a continuous collision-free path result. It was
generated using:

```powershell
decanting-precompute `
  --grid config\case_grid_nominal.yaml `
  --output-json outputs\precomputed_nominal_cases.json.gz `
  --cartesian-step-mm 10000 `
  --cartesian-rotation-step-deg 180 `
  --joint-interpolation-samples 0
```

Open the broader diagnostic cache explicitly:

```powershell
decanting-playback `
  --cache outputs\precomputed_nominal_cases.json.gz `
  --open
```

The playback page combines a finite-case parameter panel and a Meshcat iframe.
Selecting a case, process step, check, or sample performs an exact immutable
cache lookup; it never invokes the live evaluator. A
successful sample displays the stored robot configuration and the
translational ellipsoid reconstructed from its cached World-aligned SVD
directions and singular values. The unchecked `Wireframe ellipsoid` control
switches that display between translucent solid and wireframe through a cache
selection only; it never invokes IK or recomputes manipulability. A failed IK
sample shows its best diagnostic configuration with a red target frame but
hides the ellipsoid. Step 6 retains
the previous successful posture and is labelled as having no pose check.

Robot visual models are loaded only once. The default playback UI uses the pinned,
repository-owned official UR20 mesh without embedding it in an exported HTML
file. Playback also loads the locally prepared FANUC/Isaac SR-12iA mesh from
`.cache/robot_assets/sr12ia/sr12ia_mesh.urdf`; this display-only choice does
not replace the SR collision/kinematic model recorded in the cache. Use
`--sr12ia-visual-urdf PATH` to override only the displayed SR model, and use
`--sr12ia-urdf PATH` when the evaluation/fingerprint model itself must change.
If the prepared mesh is missing or stale, playback now fails with the setup
command instead of silently showing another robot. The approximate visual can
only be selected intentionally with `--allow-sr-visual-fallback`.
Cache loading verifies the scene, workflow contract, and actual evaluation
URDF fingerprints. Robot fingerprints replace local geometry paths with the
referenced file content hashes, so identical checkouts at different absolute
paths validate the same cache while changed geometry is rejected. The
prepared Isaac mesh is the 300 mm J3 option;
it may visualize a 450 mm study only while the cached displayed J3 posture is
within 300 mm. Feasibility remains explicitly tied to the 450 mm evaluation
model in that case.

## Computation/visualization split

Visual meshes must not be loaded inside a base-optimization iteration. Resolve
and load each robot once before the candidate loop. The live UI follows the
same rule: it loads evaluation bundles and display models once at startup, then
serializes parameter evaluations. For UR20, `load_visual=False` keeps the
kinematic model and collision STL geometry while omitting the DAE viewer model:

```python
from decanting_workspace import SceneState, evaluate_base_candidate, load_scene_spec
from decanting_workspace.robots import load_robot_bundle, resolve_official_ur20

spec = load_scene_spec()
urdf, package_dirs = resolve_official_ur20()
ur20 = load_robot_bundle(
    "ur20",
    urdf,
    package_dirs=package_dirs,
    load_visual=False,
    suction_proxy=spec.robots["ur20"].suction_proxy,
)
sr12ia = load_robot_bundle(
    "sr12ia",
    spec.robots["sr12ia"].urdf_path,
    load_visual=False,
)

state = SceneState()  # vary SKU/lift/tote offset/J3 outside the base loop
for ur_base, sr_base in base_candidates:
    raw_result = evaluate_base_candidate(
        spec,
        state,
        ur_base,
        sr_base,
        ur20,
        sr12ia,
    )
    # A later optimizer consumes raw_result; this layer does not rank it.
```

`evaluate_base_candidate` applies the selected SR J3 limit on each call and
rejects a 450 mm study if the supplied SR model supports only 300 mm. The
bundled fallback conservatively supports both choices. `config/cell_nominal.yaml`
defines task dimensions and the default environment proxies. In USD mode,
selected triangles provide environment geometry and bound support dimensions;
task frames and SKU dimensions remain configured. The remaining optimization work is to choose sampling bounds and
an objective/constraint aggregation over these raw results.
