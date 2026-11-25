# Verification Runbook — Nemotron 3 Nano & TwoTower Visualizations

This runbook documents how to verify mathematical parity, unit invariants, Topology A MPA bundle isolation, real-browser WebGL2 rendering, accessibility, production subpath builds, and CI workflows for **Nemotron 3 Nano 30B-A3B** (`/` & `/llm-viz/`) and **Nemotron-Labs-TwoTower 30B-A3B** (`/twotower/` & `/llm-viz/twotower/`).

---

## Environment Requirements

- **Node.js (required for build, typecheck, Vitest, and Playwright):** `22.12+ (LTS)`, `24.x (LTS)`, or `26+` (`^22.12.0 || ^24.0.0 || >=26.0.0`, matching `vitest@5.1.1`; CI uses Node `22`).
- **Python + PyTorch (optional, offline parity fixture regeneration only):** Python `3.12.x` (`>=3.12,<3.13`) and pinned `torch==2.15.0` (managed automatically via `uv run` PEP 723 metadata or `scripts/reference/requirements-ref.txt`). **Not required** for `npm test`, `npm run build`, or browser execution.

---

## 1. Static Type-Check & Vitest Unit / Parity Suite

Run the complete Node verification gate (zero browser or Python runtime needed):

```bash
npm ci
npm run typecheck
npm test
```

### Suite Coverage (`283` baseline tests across `37` files in `tests/*.test.ts` + Stage TT2 session suite)

#### Nano v1.0.0 Core Suite (`236` tests across `33` files)

| Category | Test Files | Verified Properties |
| --- | --- | --- |
| **Spec & Logical Memory** | `spec.test.ts`, `memory.test.ts` | Exact 52-layer `hybrid_override_pattern` (`23 M · 23 E · 6 *`), 4 Figure 2 brackets, 12 sensitive BF16 bays, `47.1 MiB` (`49,364,992 B`) fixed Mamba state, `+6 / +3 KiB/tok` KV growth, `1 → 1M` crossovers (`1,048 / 2,096` and `8,035 / 16,070` tokens). |
| **Single-Layer Simulators** | `gqaSim.test.ts`, `mambaSim.test.ts`, `moeSim.test.ts` | Exact `NemotronH` equations: NoPE `32Q/2KV` GQA, 4-tap Conv1D + FP32 SSM + gate-first 8-group `MambaRMSNormGated`, biased MoE selection + unbiased Top-6 `×2.5` normalization + non-gated `ReLU²` + double-width shared expert. |
| **Unified 52-Layer & Incremental AR Decode** | `hybrid-flow.test.ts`, `incremental-decode.test.ts`, `hardening-math-state.test.ts` | Bit-exact (`0` ULP) full-sequence vs. prefill+incremental and arbitrary chunk equivalence, `75.00 KiB` default snapshot accounting, length-prefixed `stateDigest` tamper rejection, isolated lazy reconstruction, 52-bay weight LRU sharing, active-branch-safe branch deletion & bounded pruning (`maxBranches=16`, `maxUniqueSteps=128`). |
| **Independent PyTorch `2.15.0` Parity** | `pytorch-parity.test.ts`, `incremental-decode.test.ts` | Single-layer, full 52-layer end-to-end, and 52-layer cached incremental AR decode parity against checked-in PyTorch `2.15.0` JSON fixtures, with strict `NaN`/`±Infinity` rejection (`tests/helpers/parityAssert.ts`). |
| **3D Cutaways, Line Packing & Focused Accumulation** | `cutaways.test.ts`, `bookend-cutaways.test.ts`, `gqa-foundation.test.ts`, `gqa-spatial.test.ts`, `mamba-spatial.test.ts`, `moe-spatial.test.ts`, `unified-cutaways.test.ts`, `incremental-cutaways.test.ts`, `line-packing.test.ts`, `spatial-primitives.test.ts`, `visual-renderer.test.ts`, `focused-accumulation.test.ts`, `v2-gqa-reveal.test.ts`, `v2-mamba-reveal.test.ts`, `v2-moe-reveal.test.ts` | Shared 18-float `LinesPass` guide-line packing across all 8 cutaway builders, RG32F atlas packing, progressive reveal frontiers, focused-cell partial sums, running MoE `Σσ` denominator without premature weight exposure, and Output Top-5 label visibility. |
| **Walkthrough, Controls, Picking & A11y/Perf** | `walkthrough.test.ts`, `walkthrough-beats.test.ts`, `v5-controls.test.ts`, `phase1-camera-ui.test.ts`, `phase1-focus.test.ts`, `pick.test.ts`, `overlay-callout.test.ts`, `rc-continuity-integration.test.ts`, `hardening-a11y-perf.test.ts` | 22-beat 3-Act walkthrough continuity, isolated Tour vs. Free Explore AR sessions, native `Space`/`Enter` deferral, WAI-ARIA `role="tab"` roving `tabindex` and modal focus traps, `RenderDevice` FBO leak/latch guards, idle-RAF gating, RG32F atlas content-key dedup, and spine-wave geometry throttling. |

#### TwoTower Unit & PyTorch `2.15.0` Parity Regression Suite (`47` baseline tests across `4` files, extended by Stage TT2 session tests)

| Category | Test Files | Verified Properties |
| --- | --- | --- |
| **Spec, adaLN Modulation & Sampler** | `twotower-spec-mod-sampler.test.ts` | Pinned HF revision `35b64985`, `0`-byte analytical vs. 24-shard safetensors payload reconciliation (`63,185,902,848` params / `126,371,817,472 B`), 9-entry discrepancy ledger, `twoTowerMemoryFor` across `ar`/`mock_ar`/`mask_diffusion`, COS→SIN timestep embedding, split `Mod→Norm` (`M`/`*`) vs. `Norm→Mod` (`E`), and MDLM/SUBS unshifted confidence unmasking (mask suppression, carry-over lock, strict `>` threshold, progress/budget floors, deterministic Gumbel-max). |
| **Shared Dual-Tower Kernels** | `twotower-kernels.test.ts` | `RouterPrecisionPolicy` (`float64_reference` vs. `float32_nano`), `AdaRMSNorm` modulation hook, bidirectional rectangular `[B × (P+B)]` GQA attention & future-token influence, externally seeded Mamba-2 stitching, `V = 34` vocabulary extension, and deterministic in-generation Denoiser weight perturbation without buffer aliasing. |
| **Independent Parameters, Conditioning & Engine** | `twotower-engine.test.ts` | Non-aliased Context/Denoiser/TimeConditioning weights (`860` tensors / `600,120` Float32s), `TwoTowerContextSnapshot` + read-only `TwoTowerContextConditioningState` (`ownerRole: 'context_tower'`, `consumerRole: 'denoiser_tower'`) digest & tamper guards, `0`-ULP Context immutability across refinements, `0`-ULP AR vs. Mock-AR identity equivalence, lazy `traceOrigin` reconstruction, and strict headless / zero-`Math.random()` source guardrails on all 7 files in `src/twotower/`. |
| **Independent PyTorch `2.15.0` TwoTower Parity** | `twotower-pytorch-parity.test.ts` | All 8 mandatory TwoTower verification categories against 3 checked-in PyTorch `2.15.0` JSON fixtures (timestep/split-adaLN + order-inversion sensitivity, bidirectional GQA, seeded/high-magnitude FP32 Mamba-2, FP64 MoE near-tie/exact-tie routing, 52-layer handoff, 2-block diffusion trajectory, AR vs. Mock-AR alignment, and MDLM/Gumbel sampler edge cases). |
| **TT2 Interactive Session Store (Stage TT2)** | `twotower-session.test.ts` *(added in TT2 Wave 1A)* | Deterministic trajectory stepping, refinement scrubbing, block commits, mode switching (`AR` / `Mock-AR` / `Mask-Diffusion`), branch isolation, and UI-independent controller invariants. |
| **TT3 Spatial Scenes & Cutaways (Stage TT3)** | `twotower-spatial-macro.test.ts`, `twotower-spatial-gqa-mamba.test.ts`, `twotower-spatial-timestep-moe-bookends.test.ts` | Dual-tower `52×2` macro geometry/linkers/timelines, asymmetric `35/65` Paired GQA (`128×128` RG32F atlas) and Paired Mamba-2 (`256×128` atlas + analytical `4×4` decomposition), dedicated Timestep Conditioning cutaway, Granular MoE `128+1` with FP64 6th-vs-7th margin & partial-denominator non-exposure, and `V = 34` Embedding/Output bookend cutaways with zero-`Infinity` MDLM sampler presentation. |
| **TT4 Wave 1 Presentation Correctness (Stage TT4)** | `twotower-wave1-correctness.test.ts` | Canonical timestep `t = nMasked / B` (never `r / K`) stage separation (`executedPass` with separate diagnostic `overrideValue` vs `empiricalMaskFraction`, `currentBlockMask`, `nextEligibleStep`, `blockCompletionState`, and 5-way `conditioningMode` distinguishing `inactive_no_pass_yet` from `bypassed_ar`/`bypassed_mock_ar`), historical Context cache boundary provenance (`P_cond=8` vs `P=12` after commit, mode switch, and Block 1 init across all 6 spatial scene builders + `pruned_unavailable` and `historicalConfig` invariants), and exact miniature dimensional inventory (`32Q/2KV dHead=4` GQA; `16` Mamba heads / `8` groups / `4×4` SSM state) clearly separated from selected atlas slices (`2` GQA score heads, `1` Mamba decomposition head). |

### Independent PyTorch `2.15.0` Parity Tolerances

TypeScript accumulates inner dot products in JS `float64` prior to `Math.fround` writes while PyTorch `torch.float32` accumulates in `float32` BLAS/SIMD. Checked-in fixtures (`tests/fixtures/*.json`) enforce:

| Fixture File | Scope | Max Absolute Tolerance |
| --- | --- | --- |
| `tests/fixtures/pytorch-reference-parity.json` | Single-layer components (`RMSNorm` `ε=1e-5`/`1e-6`, Embedding NoPE, LM Head, GQA `32Q/2KV`, Mamba-2, Granular MoE `128+1`) | Embedding: **`0`** (bit-exact) · RMSNorm / LM Head / GQA: **`<= 1e-5`** · Mamba-2 / MoE: **`<= 2e-5`** · MoE Top-6 indices: **exact match** |
| `tests/fixtures/pytorch-52layer-parity.json` | Full 52-layer end-to-end chain (`createDefaultHybridModel()`, `299,192` Float32 weights, SHA-256 `489d45a05d24317a4bb6ee4bb21bed115c3d750c887d9faf8ac44ee5d77a398b`, `parameterFingerprint: dc0568e222e22a4e`) | Embedding `X_0`: **`0`** · Residual checkpoints `X_1..X_52`: **`<= 2e-5`** · Final RMSNorm: **`<= 2e-5`** · LM Head logits: **`<= 5e-5`** · LM Head probs: **`<= 2e-5`** · All 23 MoE Top-6 & Top-5 token rankings: **exact match** |
| `tests/fixtures/pytorch-incremental-52layer-parity.json` | 52-layer cached incremental AR decode (`[2, 5, 2]` prefill + `[11]`, `[0]`, `[19]` single-token steps + externally seeded non-zero boundary continuation) | Residual checkpoints / cache states / GQA scores & softmax: **`<= 2e-5`** · LM Head logits: **`<= 5e-5`** · LM Head probs: **`<= 2e-5`** (plus TypeScript internal full-vs-incremental equivalence at **`0` ULP**) |
| `tests/fixtures/pytorch-twotower-blocks-parity.json` | TwoTower Categories 1–4 (`createDefaultTwoTowerModel()`, `600,120` Float32s, SHA-256 `1bd365e2d5ae0ac11893446c60f59b317492f203bbeb6745a21c44cb5f8130cf`): timestep/split-adaLN + order-inversion sensitivity, bidirectional `[B × (P+B)]` GQA, seeded & `O(1e3)` Mamba-2, FP64 MoE routing | Modulation & GQA: **`<= 1e-5`** · Mamba-2 & MoE: **`<= 2e-5`** · High-magnitude Mamba-2 rel: **`<= 1e-5`** · MoE Top-6 indices: **exact match** |
| `tests/fixtures/pytorch-twotower-handoff-parity.json` | TwoTower Category 5: complete 52-layer Context-to-Denoiser handoff (`P = 8`, `B = 4`, `kvCapacity = 24`, `t = 1.0`) | Embedding: **`0`** · Modulation: **`<= 1e-5`** · Cache states / residuals / traces / final norm / probs: **`<= 2e-5`** · LM Head logits: **`<= 5e-5`** · Context immutability: **`0` ULP** |
| `tests/fixtures/pytorch-twotower-trajectory-parity.json` | TwoTower Categories 6–8: 2-block diffusion trajectory (`γ = 0.22`), AR vs. Mock-AR alignment (default + identity `scale=0`, SHA-256 `a19e49afad6f455b2ede40b0cae0f090a8ebf7a1d996828efb59ad245bf24dc9`), MDLM/SUBS & deterministic Gumbel-max sampler | Cache states / residuals / probs: **`<= 2e-5`** · LM Head logits: **`<= 5e-5`** · Sampler log-probs & probs: **`<= 1e-5`** · Identity AR vs. Mock-AR: **`0` ULP** |

### Regenerating PyTorch Reference Parity Fixtures

Whenever any equation or default weight initialization in `src/math/` or `src/twotower/` changes, regenerate and re-verify all fixtures against pinned PyTorch `2.15.0`:

```bash
# --- Nano v1.0.0 fixtures ---
# 1. Export all 52 layers of createDefaultHybridModel() as exact IEEE-754 little-endian Float32 + SHA-256
node scripts/reference/export_52layer_params.mjs /tmp/nemotron-viz-ref-params

# 2. Run pinned Python 3.12 + PyTorch 2.15.0 over single-layer, full 52-layer, and incremental decode paths
uv run scripts/reference/generate_pytorch_parity_fixtures.py \
  --model-manifest /tmp/nemotron-viz-ref-params/model_52layer_manifest.json

# --- TwoTower fixtures ---
# 3. Export all 860 canonical TwoTower Float32 tensors (default + identity models)
node scripts/reference/export_twotower_params.mjs

# 4. Run pinned Python 3.12 + PyTorch 2.15.0 clean-room TwoTower generator (Categories 1-8)
uv run scripts/reference/generate_twotower_pytorch_fixtures.py

# 5. Re-run Vitest parity verification in Node
npm test
```

---

## 2. Topology A Multi-Entry MPA Build & Nano Bundle-Isolation Gate

`vite.config.ts` builds two independent Topology A entries (`index.html` for Nano and `twotower/index.html` for TwoTower) under `base: process.env.VITE_BASE_PATH ?? '/'`. Run the automated isolation gate after building both root (`/`) and GitHub Pages (`/llm-viz/`) targets:

```bash
# Root deployment build (`/` + `/twotower/`)
VITE_BASE_PATH=/ npm run build
npm run verify:isolation

# GitHub Pages subpath build (`/llm-viz/` + `/llm-viz/twotower/`)
VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages-check
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages-check
rm -rf test-results/dist-pages-check
```

### Hard Gates Enforced by `scripts/verify_bundle_isolation.mjs`

Exits non-zero on any violation across all 7 checks:

1. **Both Entry HTML Files Present:** Verifies `index.html` and `twotower/index.html` exist in the output directory.
2. **Base-Path-Aware HTML Asset Parsing:** Parses all `<script src>`, `<link rel="modulepreload" href>`, and `<link rel="stylesheet" href>` tags across both `/` and `/llm-viz/` base paths and confirms every referenced file exists.
3. **Transitive ES Module Graph Walk:** Parses every reachable JS chunk via TypeScript's ECMAScript AST (`import`, `export ... from`, and string-literal `import(...)`) to assemble complete transitive graphs for Nano and TwoTower.
4. **Strict Source-Map Isolation Gate:** Reads `.map` `sources` for every JS chunk in Nano's transitive graph and fails if any source resolves to `src/twotower/` or `twotower/`.
5. **Strict Payload Marker Gate:** Scans every JS and CSS file in Nano's transitive graph and fails if any TwoTower marker (`NemotronHTwoTowerForCausalLM`, `twotower-denoiser-v1`, `twotower-context-v1`, `TwoTowerSession`, `diffusion_block_commit`, `TWOTOWER_SPEC`) appears.
6. **TwoTower Positive Presence Gate:** Confirms TwoTower's transitive JS sourcemaps include `src/twotower/` sources (proving real sourcemap inspection occurred).
7. **Nano Payload Budget Gate:** Measures exact raw and level-9 gzip (`node:zlib`) totals for Nano, TwoTower, and shared chunks, failing if Nano's transitive assets exceed **`215 KiB` (`220,160 B`)** gzip (prints both a human summary table and machine-readable JSON).

---

## 3. Playwright / Chromium WebGL2 E2E Smoke Suite

The browser end-to-end smoke suite (`tests/e2e/smoke.spec.ts`, configured by `playwright.config.ts`) runs real Chromium against production builds to verify WebGL2 rendering, shader compilation, interactive UI, accessibility, responsive layout, and subpath deployment.

```bash
# First-time local browser setup
npx playwright install chromium

# Run the E2E smoke suite
npm run test:e2e
```

### Headless CI WebGL2 Flags

Headless Linux runners have no physical GPU, so Playwright launches Chromium with software WebGL2 via ANGLE/SwiftShader:

- `--use-gl=angle`
- `--use-angle=swiftshader`
- `--enable-webgl`
- `--ignore-gpu-blocklist`

Every test fails on uncaught `pageerror`, `console.error`, or unexpected WebGL fallback activation.

### Covered E2E Scenarios

1. **52-Layer Tower Boot** — Clean initial WebGL2 frame, zero console/page errors, `#webgl-fallback` hidden.
2. **All 5 Exploded Cutaways** — Switching across Embedding (NoPE), Mamba-2, MoE `128+1`, GQA, and Output/LM Head with active Math Dock / cell inspection and zero WebGL errors.
3. **Precision, Token Scrubbing & 1 → 1M Chart** — Toggling `BF16` ↔ `FP8 Selective`, scrubbing cached tokens, and opening/closing the `1 → 1M` Memory Scaling modal.
4. **Guided Tour & AR Session Isolation** — Advancing the 22-beat walkthrough into Act III cached AR Decode while confirming the user's Free Explore AR decode branches remain strictly isolated upon exiting Tour.
5. **Native Keyboard & WAI-ARIA Compliance** — Verifying `Space`/`Enter` trigger native activation on focused buttons instead of hijacking into tour/zoom shortcuts, `role="tab"` roving `tabindex` + `ArrowRight` navigation on `.mode-pills` and `#gqa-step-pills`, and modal focus capture/trap + `Escape` focus restoration on `?` and `M`.
6. **Reduced-Motion Media Emulation** — Emulating `prefers-reduced-motion: reduce` (including live toggles) to confirm animations settle cleanly without breaking interactive inspection.
7. **Output Top-5, MoE Running `Σσ` & Popover Containment** — Verifying all 5 Output candidate pills (`#1..#5`) render visibly without overlap, MoE Step 3 displays the running `Σσ` denominator while masking uncomputed sidebar weights, and floating cell popovers stay inside stage viewport bounds.
8. **Responsive Viewports** — Validating desktop (`1440×900`), narrow (`1024×800`, verifying unclipped `.spine-wave-hud` and toolbar), and mobile (`390×844`, verifying clean Tower framing and unclipped `#shortcuts-popover`).
9. **GitHub Pages `/llm-viz/` Subpath Build** — Building with `VITE_BASE_PATH=/llm-viz/ npm run build` and verifying `/llm-viz/` serves all JS/CSS assets with HTTP `200`, zero `404`s, and functional WebGL2 cutaways.

---

## 4. CI & Deployment Architecture

Verification and production deployment are strictly separated into two workflows under `.github/workflows/`:

| Workflow | Triggers | Permissions | Steps |
| --- | --- | --- | --- |
| **PR & Push CI** (`.github/workflows/ci.yml`) | `pull_request` (to `main` and `feat/twotower`), `push` (to `main` and `feat/twotower`), `workflow_dispatch` | Read-only (`contents: read`) — **never deploys** | `actions/setup-node@v4` (Node `22`) → `npm ci` → `npm run typecheck` → `npm test` → `npm run build` → `npm run verify:isolation` → `npx playwright install --with-deps chromium` → `npm run test:e2e` (incl. `/llm-viz/` subpath check) → upload Playwright report artifact on failure only. |
| **GitHub Pages Deploy** (`.github/workflows/deploy.yml`) | `push` (`branches: [main]` only) and manual `workflow_dispatch` — **never runs on PRs or `feat/twotower`** | `contents: read`, `pages: write`, `id-token: write` | `actions/setup-node@v4` (Node `22`) → `npm ci` → `npm run typecheck` → `npm test` → `VITE_BASE_PATH=/llm-viz/ npm run build` → `actions/upload-pages-artifact` → `actions/deploy-pages`. |

---

## 5. Known Limitations & Scope Boundaries

- **Miniature Synthetic Simulators:** Browser execution uses synthetic Float32/Float64 weights (`d = 8`, `V_mini = 32` for Nano, `V_mini = 34` for TwoTower). Neither entry loads NVIDIA checkpoints, runs the `131,072`-token Nemotron tokenizer, or simulates hardware FP8 E4M3 quantization rounding.
- **Single-Sequence Logical Accounting:** Memory readouts report logical per-sequence tensor payload sizes (`47.1 MiB` fixed Mamba state + `+6 / +3 KiB/tok` KV cache; plus TwoTower ephemeral block KV, `mamba_s2`, and exact `117.69 GiB` dual-tower BF16 weight metadata) excluding allocator padding, batching, and temporary activation buffers.
- **Software vs. Hardware GPU Differences:** Headless CI exercises Chromium WebGL2 via ANGLE/SwiftShader. Visual fidelity checks on hardware GPUs (high-DPI macOS Metal / Windows D3D11) remain a targeted manual pre-release check when shaders or CSS change.
- **TwoTower Rollout Phasing:** Headless TwoTower math, independent PyTorch `2.15.0` parity fixtures, Topology A multi-entry MPA build, and automated Nano bundle-isolation gating are active on `feat/twotower` while keeping `main` / `v1.0.0` and `.github/workflows/deploy.yml` untouched (see [`docs/TWOTOWER_ARCHITECTURE.md`](TWOTOWER_ARCHITECTURE.md)). Interactive TwoTower session controls and diagnostic UI wire into the Topology A `/twotower/` shell in subsequent TT2 waves.
