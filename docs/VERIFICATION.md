# Verification & Reproducibility Runbook — Nemotron 3 Nano & TwoTower

This runbook documents how to reproduce and verify mathematical parity, unit/contract invariants, Topology A MPA bundle isolation, real-browser WebGL2 rendering (software SwiftShader and hardware Metal/GPU), accessibility, static artifact checksums, and CI/Pages workflows for both applications:

- **Nemotron 3 Nano 30B-A3B** (`/` & `/llm-viz/` · `v1.0.0` preserved)
- **Nemotron-Labs-TwoTower 30B-A3B** (`/twotower/` & `/llm-viz/twotower/` · `twotower-v1.0.0` release candidate)

---

## Environment Requirements

- **Node.js (required for build, typecheck, Vitest, isolation gate, and Playwright):** `22.12+ (LTS)`, `24.x (LTS)`, or `26+` (`^22.12.0 || ^24.0.0 || >=26.0.0`, matching `vitest@5.1.1`; CI uses Node `22`, pinned in `.nvmrc`).
- **Python + PyTorch (optional, offline parity fixture regeneration only):** Python `3.12.x` (`>=3.12,<3.13`) and pinned `torch==2.15.0` (managed via `uv run` PEP 723 metadata or `scripts/reference/requirements-ref.txt`). **Not required** for `npm test`, `npm run build`, `npm run verify:isolation`, `npm run test:e2e`, or browser execution.

---

## 1. End-to-End Reproducibility Runbook (Quick Sequence)

Execute every automated verification gate in order from a clean checkout:

```bash
# 1. Clean reproducible install (exact lockfile, 0 runtime dependencies)
npm ci

# 2. Strict TypeScript verification
npm run typecheck

# 3. Full Vitest unit, contract & PyTorch-parity suite (365+ tests across 45 files; zero Python needed)
npm test

# 4. (Optional) Regenerate independent PyTorch 2.15.0 parity fixtures (Nano + TwoTower Categories 1-8)
node scripts/reference/export_52layer_params.mjs /tmp/nemotron-viz-ref-params
uv run scripts/reference/generate_pytorch_parity_fixtures.py \
  --model-manifest /tmp/nemotron-viz-ref-params/model_52layer_manifest.json
node scripts/reference/export_twotower_params.mjs
uv run scripts/reference/generate_twotower_pytorch_fixtures.py
npm test

# 5. Root production build (`/` + `/twotower/`) & 7-gate Nano bundle-isolation verification
VITE_BASE_PATH=/ npm run build
npm run verify:isolation

# 6. GitHub Pages subpath build (`/llm-viz/` + `/llm-viz/twotower/`) & isolation verification
VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages

# 7. Real Chromium WebGL2 E2E smoke suite across both apps & both base paths
#    (first time: npx playwright install chromium)
npm run test:e2e
```

---

## 2. Vitest Unit, Contract & PyTorch `2.15.0` Parity Suite (`365+` Tests Across `45` Files)

The Node regression suite runs `45` test files (`33` Nano files + `12` TwoTower files) totaling **`365+` tests** (`236` Nano + `129+` TwoTower):

### A. Nano `v1.0.0` Core Suite (`236` Tests Across `33` Files)

| Category | Test Files | Verified Properties |
| --- | --- | --- |
| **Spec & Logical Memory** | `spec.test.ts`, `memory.test.ts` | Exact 52-layer `hybrid_override_pattern` (`23 M · 23 E · 6 *`), 4 Figure 2 brackets, 12 sensitive BF16 bays, `47.08 MiB` (`49,364,992 B`) fixed Mamba state, `+6 / +3 KiB/tok` KV growth, `1 → 1M` crossovers (`1,048 / 2,096` and `8,035 / 16,070` tokens). |
| **Single-Layer Simulators** | `gqaSim.test.ts`, `mambaSim.test.ts`, `moeSim.test.ts` | Exact `NemotronH` equations: NoPE `32Q/2KV` GQA, 4-tap Conv1D + FP32 SSM + gate-first 8-group `MambaRMSNormGated`, biased MoE selection + unbiased Top-6 `×2.5` normalization + non-gated `ReLU²` + double-width shared expert. |
| **Unified 52-Layer & Incremental AR Decode** | `hybrid-flow.test.ts`, `incremental-decode.test.ts`, `hardening-math-state.test.ts` | Bit-exact (`0` ULP) full-sequence vs. prefill+incremental and arbitrary chunk equivalence, `75.00 KiB` default snapshot accounting, length-prefixed `stateDigest` tamper rejection, isolated lazy reconstruction, 52-bay weight LRU sharing, active-branch-safe branch deletion & bounded pruning (`maxBranches=16`, `maxUniqueSteps=128`). |
| **Independent PyTorch `2.15.0` Parity** | `pytorch-parity.test.ts`, `incremental-decode.test.ts` | Single-layer, full 52-layer end-to-end, and 52-layer cached incremental AR decode parity against checked-in PyTorch `2.15.0` JSON fixtures, with strict `NaN`/`±Infinity` rejection (`tests/helpers/parityAssert.ts`). |
| **3D Cutaways, Line Packing & Focused Accumulation** | `cutaways.test.ts`, `bookend-cutaways.test.ts`, `gqa-foundation.test.ts`, `gqa-spatial.test.ts`, `mamba-spatial.test.ts`, `moe-spatial.test.ts`, `unified-cutaways.test.ts`, `incremental-cutaways.test.ts`, `line-packing.test.ts`, `spatial-primitives.test.ts`, `visual-renderer.test.ts`, `focused-accumulation.test.ts`, `v2-gqa-reveal.test.ts`, `v2-mamba-reveal.test.ts`, `v2-moe-reveal.test.ts` | Shared 18-float `LinesPass` guide-line packing across all 8 cutaway builders, `RG32F` atlas packing, progressive reveal frontiers, focused-cell partial sums, running MoE `Σσ` denominator without premature weight exposure, and Output Top-5 label visibility. |
| **Walkthrough, Controls, Picking & A11y/Perf** | `walkthrough.test.ts`, `walkthrough-beats.test.ts`, `v5-controls.test.ts`, `phase1-camera-ui.test.ts`, `phase1-focus.test.ts`, `pick.test.ts`, `overlay-callout.test.ts`, `rc-continuity-integration.test.ts`, `hardening-a11y-perf.test.ts` | 22-beat 3-Act walkthrough continuity, isolated Tour vs. Free Explore AR sessions, native `Space`/`Enter` deferral, WAI-ARIA `role="tab"` roving `tabindex` and modal focus traps, `RenderDevice` FBO leak/latch guards, idle-RAF gating, `RG32F` atlas content-key dedup, and spine-wave geometry throttling. |

### B. TwoTower `v1.0.0` Suite (`129+` Tests Across `12` Files)

| Category | Test File | Verified Properties |
| --- | --- | --- |
| **Spec, adaLN Modulation & Sampler** | `twotower-spec-mod-sampler.test.ts` | Pinned HF revision `35b6498508be8856d380cdfe4a900a79b918d1ea`, `0`-byte analytical vs. 24-shard safetensors reconciliation (`63,185,902,848` params / `126,371,817,472 B`), 9-entry discrepancy ledger, `twoTowerMemoryFor`, COS→SIN sinusoidal embedding, split `Mod→Norm` (`M`/`*`) vs. `Norm→Mod` (`E`), MDLM/SUBS unmasking & deterministic Gumbel-max. |
| **Shared Dual-Tower Kernels** | `twotower-kernels.test.ts` | `RouterPrecisionPolicy` (`float64_reference` vs. `float32_nano`), `AdaRMSNorm` hook, bidirectional rectangular `[B × (P+B)]` GQA attention & future-token influence, externally seeded Mamba-2 stitching, `V = 34` extension, and deterministic non-aliased Denoiser perturbation. |
| **Parameters, Ownership & Engine** | `twotower-engine.test.ts` | Non-aliased Context/Denoiser/TimeConditioning weights (`860` tensors / `600,120` Float32s), `TwoTowerContextSnapshot` & read-only `TwoTowerContextConditioningState` digest/tamper guards, `0`-ULP Context immutability, `0`-ULP AR vs. Mock-AR identity equivalence, lazy `traceOrigin` reconstruction, and headless / zero-`Math.random()` guardrails. |
| **Independent PyTorch `2.15.0` Parity (8 Categories)** | `twotower-pytorch-parity.test.ts` | All 8 mandatory TwoTower verification categories against 3 checked-in PyTorch `2.15.0` fixtures (split-adaLN + order-inversion sensitivity, bidirectional GQA, seeded & `O(1e3)` Mamba-2, FP64 MoE near-tie/exact-tie routing, 52-layer handoff, 2-block trajectory, AR vs. Mock-AR alignment, MDLM/Gumbel edge cases). |
| **Interactive Session & Branching** | `twotower-session.test.ts` | 4-step block diffusion lifecycle, `AR`/`Mock-AR` switching, non-destructive scrubbing, auto-forking, active-branch-safe deletion, bounded retention, and digest provenance. |
| **Context & Sampler Contracts** | `twotower-context-sampler-contracts.test.ts` | Strict Context ownership invariants, non-retroactive config scope separation (`record.configAtStep` vs. `session.pendingConfig`), independent `B`/`K` budget floors, calibration metadata, and zero-`Infinity` sampler presentations. |
| **Spatial Macro Scene** | `twotower-spatial-macro.test.ts` | Dual-tower `52×2` macro geometry, `29` aligned transfer linkers vs. `23` stateless MoE bays, linker channel filters, instructional Macro Flow Replay, and 10-axis bay picking. |
| **Spatial Paired GQA & Mamba-2 Cutaways** | `twotower-spatial-gqa-mamba.test.ts` | Asymmetric 35/65 Paired GQA (`128×128` `RG32F` atlas, bidirectional `[B × (P+B)]`) and Paired Mamba-2 (`256×128` atlas, 4-tap Conv1D `3→2→1→0` boundary replacement, `4×4` analytical SSM decomposition, partial-accumulation non-exposure). |
| **Spatial Timestep, MoE & Bookend Cutaways** | `twotower-spatial-timestep-moe-bookends.test.ts` | Dedicated Timestep Conditioning cutaway (`Mod→Norm` vs. `Norm→Mod`, unmodulated `norm_f`), Granular MoE `128+1` FP64 6th-vs-7th margin & non-spoiling `Σσ` denominator, and `V = 34` Embedding/Output cutaways with zero-`Infinity` MDLM output. |
| **Presentation Correctness & Provenance** | `twotower-wave1-correctness.test.ts` | Canonical `t = nMasked / B` (never `r / K`), 5-way `conditioningMode`, historical Context boundary provenance (`P_cond` vs. current `P` across all 6 spatial scene builders + `pruned_unavailable` & `historicalConfig` invariants), and logical miniature inventory vs. selected atlas slice separation. |
| **Production Architectural Readouts & Scaling** | `twotower-arch-readouts.test.ts` | Dual SI (`GB`/`MB`) & IEC (`GiB`/`MiB`) checkpoint inventory, FP32 `e_score_correction_bias` (`+11,776 B`), 3-regime context-length scaling (`≤ 128K` operational · `128K–256K` configured limit · `> 256K` extrapolated), 4-way memory separation with explicit peak runtime workspace exclusion, persistent nested timeline, and advanced sampler view-models. |
| **Guided 4-Act / 28-Beat Walkthrough** | `twotower-tour.test.ts` | All 4 Acts / 28 Beats navigation, dynamic trajectory-derived live metrics (zero fabricated values, zero `Infinity`/`NaN`), strict Tour vs. Free Explore session isolation, and exact state restoration on exit. |

### Independent PyTorch `2.15.0` Parity Tolerances

TypeScript accumulates inner dot products in JS `float64` prior to `Math.fround` writes while PyTorch `torch.float32` accumulates in `float32` BLAS/SIMD. Checked-in fixtures (`tests/fixtures/*.json`) enforce:

| Fixture File | Scope | Max Absolute Tolerance |
| --- | --- | --- |
| `tests/fixtures/pytorch-reference-parity.json` | Nano single-layer components (`RMSNorm` `ε=1e-5`/`1e-6`, Embedding NoPE, LM Head, GQA `32Q/2KV`, Mamba-2, Granular MoE `128+1`) | Embedding: **`0`** · RMSNorm / LM Head / GQA: **`<= 1e-5`** · Mamba-2 / MoE: **`<= 2e-5`** · MoE Top-6 indices: **exact match** |
| `tests/fixtures/pytorch-52layer-parity.json` | Nano full 52-layer chain (`299,192` Float32s, SHA-256 `489d45a05d24317a4bb6ee4bb21bed115c3d750c887d9faf8ac44ee5d77a398b`, `parameterFingerprint: dc0568e222e22a4e`) | Embedding: **`0`** · Residuals / Final Norm / Probs: **`<= 2e-5`** · Logits: **`<= 5e-5`** · All 23 MoE Top-6 & Top-5 rankings: **exact match** |
| `tests/fixtures/pytorch-incremental-52layer-parity.json` | Nano 52-layer cached incremental AR decode + externally seeded non-zero continuation | Residuals / caches / scores / softmax / probs: **`<= 2e-5`** · Logits: **`<= 5e-5`** · TS full-vs-incremental: **`0` ULP** |
| `tests/fixtures/pytorch-twotower-blocks-parity.json` | TwoTower Categories 1–4 (`600,120` Float32s, SHA-256 `1bd365e2d5ae0ac11893446c60f59b317492f203bbeb6745a21c44cb5f8130cf`): timestep/split-adaLN + order-inversion, bidirectional GQA, seeded & `O(1e3)` Mamba-2, FP64 MoE | Modulation & GQA: **`<= 1e-5`** · Mamba-2 & MoE: **`<= 2e-5`** · High-magnitude Mamba-2 rel: **`<= 1e-5`** · MoE Top-6 indices: **exact match** |
| `tests/fixtures/pytorch-twotower-handoff-parity.json` | TwoTower Category 5: complete 52-layer Context-to-Denoiser handoff (`P = 8, B = 4, kvCapacity = 24, t = 1.0`) | Embedding: **`0`** · Modulation: **`<= 1e-5`** · Caches / residuals / traces / norm / probs: **`<= 2e-5`** · Logits: **`<= 5e-5`** · Context immutability: **`0` ULP** |
| `tests/fixtures/pytorch-twotower-trajectory-parity.json` | TwoTower Categories 6–8: 2-block diffusion trajectory (`γ = 0.22`), AR vs. Mock-AR (default + identity `scale=0`, SHA-256 `a19e49afad6f455b2ede40b0cae0f090a8ebf7a1d996828efb59ad245bf24dc9`), MDLM/SUBS & Gumbel-max | Caches / residuals / probs: **`<= 2e-5`** · Logits: **`<= 5e-5`** · Sampler log-probs & probs: **`<= 1e-5`** · Identity AR vs. Mock-AR: **`0` ULP** |

---

## 3. Topology A Multi-Entry MPA Build & Nano Bundle-Isolation Gate

Run the automated 7-gate verifier (`scripts/verify_bundle_isolation.mjs`) against **both** root (`/`) and GitHub Pages (`/llm-viz/`) builds:

```bash
# Root build (`/` + `/twotower/`)
VITE_BASE_PATH=/ npm run build
npm run verify:isolation

# GitHub Pages subpath build (`/llm-viz/` + `/llm-viz/twotower/`)
VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages
```

Enforced hard gates (non-zero exit on any failure):

1. Both `index.html` and `twotower/index.html` exist.
2. Every `<script src>`, `<link rel="modulepreload" href>`, and `<link rel="stylesheet" href>` resolves cleanly under `/` or `/llm-viz/`.
3. TypeScript ECMAScript AST walks every transitive static `import`/`export ... from` and literal dynamic `import(...)` edge.
4. Zero `.map` sources in Nano's transitive graph match `src/twotower/` or `twotower/`.
5. Zero forbidden TwoTower markers (`NemotronHTwoTowerForCausalLM`, `twotower-denoiser-v1`, `twotower-context-v1`, `TwoTowerSession`, `diffusion_block_commit`, `TWOTOWER_SPEC`) appear in Nano's transitive JS/CSS assets.
6. TwoTower's transitive JS sourcemaps positively contain `src/twotower/` sources.
7. Nano's transitive JS+CSS level-9 gzip total stays `<= 215.00 KiB` (`220,160 B`; current baseline `207.53 KiB` / `212,508 B` with `127.27 KiB` shared across `31` Nano primitives and `~311 KiB` TwoTower gzip).

---

## 4. Playwright / Chromium WebGL2 E2E Smoke Suite (`20+` Tests)

`playwright.config.ts` starts two isolated preview servers (`127.0.0.1:4173` for root `/` and `127.0.0.1:4174/llm-viz/` for GitHub Pages subpath) and runs both E2E suites:

```bash
# First-time local browser setup
npx playwright install chromium

# Run all Nano + TwoTower WebGL2 E2E tests (SwiftShader headless Chromium)
npm run test:e2e
```

### Headless CI Software WebGL2 Flags (`playwright.config.ts`)

GPU-less Linux CI runners execute Chromium via ANGLE/SwiftShader using all 5 required flags:

- `--use-gl=angle`
- `--use-angle=swiftshader`
- `--enable-webgl`
- `--ignore-gpu-blocklist`
- `--enable-unsafe-swiftshader` *(required by Chromium 137+ headless on GPU-less runners)*

Every test fails on any uncaught `pageerror`, `console.error`, or unexpected `#webgl-fallback` activation.

### Hardware GPU / macOS Metal Verification

Software SwiftShader exercises shader compilation and WebGL2 correctness but does not substitute for hardware GPU rendering/compositing checks. Before tagging a release, run a targeted hardware GPU pass on a local workstation (e.g. Apple Silicon / macOS Metal or Windows D3D11):

```bash
# 1. Start local production preview for root (or /llm-viz/)
VITE_BASE_PATH=/ npm run build
npx vite preview --outDir dist --base / --port 4173 --strictPort --host 127.0.0.1

# 2. Open http://127.0.0.1:4173/ and http://127.0.0.1:4173/twotower/ in hardware-accelerated Chrome/Safari
#    Verify in DevTools / footer telemetry:
#    - WebGL2 renderer reports hardware GPU (e.g. "Apple M-series / ANGLE Metal" — tag shows "HW", not "SW")
#    - Continuous orbit/fly maintains smooth RAF Δ cadence and enters idle RAF sleep cleanly when stationary
#    - High-DPI crispness (devicePixelRatio >= 2) across all 7 TwoTower views, 5 Nano cutaways, and reciprocal links
#    - Zero console errors, zero WebGL errors, zero layout clipping at 1440×900, 1024×800, 768×1024, and 390×844 / 430×932
```

### Performance Telemetry Interpretation Note

> [!NOTE]
> **Application-Side JS Callback Time vs. Full GPU Execution Latency**
> Measurements captured with `performance.now()` around `drawFrame()` (`lastFrameMs` / `JS cb` in the TwoTower footer) quantify **application-side main-thread CPU processing and WebGL command submission time** — they do **NOT** measure full asynchronous GPU rasterization, fragment shading, or display compositor latency. Observed smoothness during active camera flies or replay is tracked separately via continuous **observed RAF interval cadence (`RAF Δ`)**, measured strictly between consecutive animating frames (excluding idle-sleep pauses).

### Covered E2E Scenarios (`20+` Tests Across 2 Files)

#### Nano E2E Suite (`tests/e2e/smoke.spec.ts` — `9` Tests)
1. **52-Layer Tower Boot** — Clean initial WebGL2 frame, zero console/page errors, `#webgl-fallback` hidden.
2. **All 5 Exploded Cutaways** — Switching across Embedding, Mamba-2, MoE `128+1`, GQA, and Output/LM Head with camera-fly settlement and live landmarks.
3. **`LinesPass` Guide-Line Geometry** — Rendered guide lines verified across Embedding, Mamba, MoE, and GQA cutaways.
4. **Guided Tour & AR Session Isolation** — Beats `7.1–7.3` continuity, AR branching, and Tour vs. Free Explore isolation.
5. **Native Keyboard & WAI-ARIA Compliance** — Native `Space`/`Enter` control activation, roving `tabindex`, and modal focus capture/restore (`?` and `M`).
6. **Reduced-Motion Emulation** — Emulating and toggling `prefers-reduced-motion: reduce`.
7. **Output Top-5, MoE Running `Σσ` & Popover Containment** — Camera-fly-settled `#1..#5` non-overlapping Output pills, MoE Step 3 running `Σσ` masking, and popover viewport containment.
8. **Responsive Viewports** — `1440×900`, `1024×800`, and mobile `390×844` layout containment.
9. **GitHub Pages `/llm-viz/` Subpath Build** — HTTP `200` assets, zero `404`s, functional cutaways.

#### TwoTower E2E Suite (`tests/e2e/twotower-smoke.spec.ts` — `11+` Tests)
1. **Root `/twotower/` Dual-Tower Macro Boot** — Non-blank WebGL2 render, epistemic pill, linker channels, 10-axis bay picking, and quiet reciprocal navigation.
2. **Full Diffusion Lifecycle & All 7 Spatial Views** — `Init` → `Refine` → `Commit Conf` → `Commit → Context`, `128×128` GQA, Mamba Conv1D replacement, MoE partial-denom guard, Output zero-`Infinity`.
3. **AR, Mock-AR (`BYPASSED`), Rewind, Branch Fork & Synced Workbench Drawer (`W`)**.
4. **Reduced-Motion, Idle-RAF Gating, Native `Space`/`Enter` & Narrow `1024×800` Layout**.
5. **GitHub Pages `/llm-viz/twotower/` Subpath Build & HTTP `200` Assets**.
6. **Guided 4-Act / 28-Beat Walkthrough (`T`) & Strict User-Session Restoration**.
7. **Persistent Nested Timeline, Completed-Uncommitted vs. Committed Boundary Warning & Branching**.
8. **Advanced Sampler Inspector (`0.22`/`0.80`/`0.90` Presets, Decoupled `B`/`K`, Zero-`Infinity`, Non-Retroactive Scope)**.
9. **Production Architecture & 3-Regime Context-Length Scaling Tab (`63.19B` / `126.37 GB` / `117.69 GiB`, 4-Way Caches, Ledger, SVG Chart)**.
10. **Keyboard Navigation (`T`/Arrows/`Esc`/`W`/`Tab`), Reduced-Motion Instant Transitions, Zero Geometry Rebuilds on Orbit & Idle RAF Sleep**.
11. **GitHub Pages `/llm-viz/twotower/` Parity (Tour, Timeline, Sampler, Architecture & Reciprocal `/llm-viz/` Navigation)**.

---

## 5. Static Portable & GitHub Pages Artifact Packaging (`shasum -a 256`)

To produce verifiable, self-contained static release tarballs for both root (`/`) and GitHub Pages (`/llm-viz/`) deployments:

```bash
# Clean output staging directory (gitignored)
mkdir -p test-results/release-artifacts

# 1. Portable static archive (serves at root `/` and `/twotower/`)
VITE_BASE_PATH=/ npx vite build --outDir test-results/dist-portable
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-portable
tar -czf test-results/release-artifacts/nemotron-viz-twotower-v1.0.0-portable.tar.gz \
  -C test-results/dist-portable .

# 2. GitHub Pages subpath archive (serves at `/llm-viz/` and `/llm-viz/twotower/`)
VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages
tar -czf test-results/release-artifacts/nemotron-viz-twotower-v1.0.0-pages.tar.gz \
  -C test-results/dist-pages .

# 3. Generate and verify SHA-256 checksums
(
  cd test-results/release-artifacts
  shasum -a 256 nemotron-viz-twotower-v1.0.0-*.tar.gz > SHA256SUMS.txt
  shasum -a 256 -c SHA256SUMS.txt
)
```

---

## 6. CI & Deployment Workflows

Verification and production deployment remain strictly separated under `.github/workflows/`:

| Workflow | Triggers | Permissions | Steps |
| --- | --- | --- | --- |
| **PR & Push CI** (`.github/workflows/ci.yml`) | `pull_request` (`main`, `feat/twotower`), `push` (`main`, `feat/twotower`), `workflow_dispatch` | Read-only (`contents: read`) — **never deploys** | Node `22` → `npm ci` → `npm run typecheck` → `npm test` → `npm run build` → `npm run verify:isolation` (root `/`) → `VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages` + `node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages` (`/llm-viz/`) → `npx playwright install --with-deps chromium` → `npm run test:e2e` → upload Playwright report on failure. |
| **GitHub Pages Deploy** (`.github/workflows/deploy.yml`) | `push` (`branches: [main]` **only**) and manual `workflow_dispatch` — **never runs on PRs or `feat/twotower`** | `contents: read`, `pages: write`, `id-token: write` | Node `22` → `npm ci` → `npm run typecheck` → `npm test` → `VITE_BASE_PATH=/llm-viz/ npm run build` → `npm run verify:isolation` → `actions/configure-pages` → `actions/upload-pages-artifact` (`./dist`) → `actions/deploy-pages`. |

---

## 7. Known Limitations & Scope Boundaries

- **Miniature Synthetic Simulators:** Browser execution uses deterministic synthetic Float32/Float64 weights (`d = 8`, `V_mini = 32` for Nano; `V_mini = 34` for TwoTower). Neither entry loads NVIDIA checkpoints, runs the `131,072`-token Nemotron tokenizer, or simulates hardware FP8 E4M3 quantization rounding.
- **4-Way Memory Boundary (Peak Runtime Workspace Excluded):** Production memory readouts account for resident BF16/FP32 weights, Persistent Context Cache (`47.08 MiB + P × 6 KiB`), optional snapshots/mirrors (`mamba_s2` `+47.08 MiB` in `Mock-AR`, optional cross-GPU mirror), and Transient Denoiser Block KV (`B × 6 KiB`). Complete peak inference memory (intermediate activations, `[B × 131,072]` logits, CUDA/framework scratchpad, allocator fragmentation) is explicitly excluded.
- **3-Regime Context Scaling Caveat:** `262,144` (`256K`) reflects `config.json max_position_embeddings` and is explicitly distinguished from the model card's `128K` operational note and from `> 256K` analytical extrapolation.
- **Software vs. Hardware GPU Verification:** Headless CI runs Chromium WebGL2 on ANGLE/SwiftShader; hardware GPU visual/cadence inspection remains a targeted pre-release step.
