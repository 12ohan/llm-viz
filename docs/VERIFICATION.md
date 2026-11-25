# Verification Runbook — Nemotron 3 Nano Visualization

This runbook documents how to verify mathematical parity, unit invariants, real-browser WebGL2 rendering, accessibility, production subpath builds, and CI workflows for **Nemotron 3 Nano 30B-A3B — Architecture Visualization**.

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

### Suite Coverage (`236` tests across `33` files in `tests/*.test.ts`)

| Category | Test Files | Verified Properties |
| --- | --- | --- |
| **Spec & Logical Memory** | `spec.test.ts`, `memory.test.ts` | Exact 52-layer `hybrid_override_pattern` (`23 M · 23 E · 6 *`), 4 Figure 2 brackets, 12 sensitive BF16 bays, `47.1 MiB` (`49,364,992 B`) fixed Mamba state, `+6 / +3 KiB/tok` KV growth, `1 → 1M` crossovers (`1,048 / 2,096` and `8,035 / 16,070` tokens). |
| **Single-Layer Simulators** | `gqaSim.test.ts`, `mambaSim.test.ts`, `moeSim.test.ts` | Exact `NemotronH` equations: NoPE `32Q/2KV` GQA, 4-tap Conv1D + FP32 SSM + gate-first 8-group `MambaRMSNormGated`, biased MoE selection + unbiased Top-6 `×2.5` normalization + non-gated `ReLU²` + double-width shared expert. |
| **Unified 52-Layer & Incremental AR Decode** | `hybrid-flow.test.ts`, `incremental-decode.test.ts`, `hardening-math-state.test.ts` | Bit-exact (`0` ULP) full-sequence vs. prefill+incremental and arbitrary chunk equivalence, `75.00 KiB` default snapshot accounting, length-prefixed `stateDigest` tamper rejection, isolated lazy reconstruction, 52-bay weight LRU sharing, active-branch-safe branch deletion & bounded pruning (`maxBranches=16`, `maxUniqueSteps=128`). |
| **Independent PyTorch `2.15.0` Parity** | `pytorch-parity.test.ts`, `incremental-decode.test.ts` | Single-layer, full 52-layer end-to-end, and 52-layer cached incremental AR decode parity against checked-in PyTorch `2.15.0` JSON fixtures, with strict `NaN`/`±Infinity` rejection (`tests/helpers/parityAssert.ts`). |
| **3D Cutaways, Line Packing & Focused Accumulation** | `cutaways.test.ts`, `bookend-cutaways.test.ts`, `gqa-foundation.test.ts`, `gqa-spatial.test.ts`, `mamba-spatial.test.ts`, `moe-spatial.test.ts`, `unified-cutaways.test.ts`, `incremental-cutaways.test.ts`, `line-packing.test.ts`, `spatial-primitives.test.ts`, `visual-renderer.test.ts`, `focused-accumulation.test.ts`, `v2-gqa-reveal.test.ts`, `v2-mamba-reveal.test.ts`, `v2-moe-reveal.test.ts` | Shared 18-float `LinesPass` guide-line packing across all 8 cutaway builders, RG32F atlas packing, progressive reveal frontiers, focused-cell partial sums, running MoE `Σσ` denominator without premature weight exposure, and Output Top-5 label visibility. |
| **Walkthrough, Controls, Picking & A11y/Perf** | `walkthrough.test.ts`, `walkthrough-beats.test.ts`, `v5-controls.test.ts`, `phase1-camera-ui.test.ts`, `phase1-focus.test.ts`, `pick.test.ts`, `overlay-callout.test.ts`, `rc-continuity-integration.test.ts`, `hardening-a11y-perf.test.ts` | 22-beat 3-Act walkthrough continuity, isolated Tour vs. Free Explore AR sessions, native `Space`/`Enter` deferral, WAI-ARIA `role="tab"` roving `tabindex` and modal focus traps, `RenderDevice` FBO leak/latch guards, idle-RAF gating, RG32F atlas content-key dedup, and spine-wave geometry throttling. |

### Independent PyTorch `2.15.0` Parity Tolerances

TypeScript accumulates inner dot products in JS `float64` prior to `Math.fround` writes while PyTorch `torch.float32` accumulates in `float32` BLAS/SIMD. Checked-in fixtures (`tests/fixtures/*.json`) enforce:

| Fixture File | Scope | Max Absolute Tolerance |
| --- | --- | --- |
| `tests/fixtures/pytorch-reference-parity.json` | Single-layer components (`RMSNorm` `ε=1e-5`/`1e-6`, Embedding NoPE, LM Head, GQA `32Q/2KV`, Mamba-2, Granular MoE `128+1`) | Embedding: **`0`** (bit-exact) · RMSNorm / LM Head / GQA: **`<= 1e-5`** · Mamba-2 / MoE: **`<= 2e-5`** · MoE Top-6 indices: **exact match** |
| `tests/fixtures/pytorch-52layer-parity.json` | Full 52-layer end-to-end chain (`createDefaultHybridModel()`, `299,192` Float32 weights, SHA-256 `489d45a05d24317a4bb6ee4bb21bed115c3d750c887d9faf8ac44ee5d77a398b`, `parameterFingerprint: dc0568e222e22a4e`) | Embedding `X_0`: **`0`** · Residual checkpoints `X_1..X_52`: **`<= 2e-5`** · Final RMSNorm: **`<= 2e-5`** · LM Head logits: **`<= 5e-5`** · LM Head probs: **`<= 2e-5`** · All 23 MoE Top-6 & Top-5 token rankings: **exact match** |
| `tests/fixtures/pytorch-incremental-52layer-parity.json` | 52-layer cached incremental AR decode (`[2, 5, 2]` prefill + `[11]`, `[0]`, `[19]` single-token steps + externally seeded non-zero boundary continuation) | Residual checkpoints / cache states / GQA scores & softmax: **`<= 2e-5`** · LM Head logits: **`<= 5e-5`** · LM Head probs: **`<= 2e-5`** (plus TypeScript internal full-vs-incremental equivalence at **`0` ULP**) |

### Regenerating PyTorch Reference Parity Fixtures

Whenever any equation or default weight initialization in `src/math/` changes, regenerate and re-verify all fixtures against pinned PyTorch `2.15.0`:

```bash
# 1. Export all 52 layers of createDefaultHybridModel() as exact IEEE-754 little-endian Float32 + SHA-256
node scripts/reference/export_52layer_params.mjs /tmp/nemotron-viz-ref-params

# 2. Run pinned Python 3.12 + PyTorch 2.15.0 over single-layer, full 52-layer, and incremental decode paths
uv run scripts/reference/generate_pytorch_parity_fixtures.py \
  --model-manifest /tmp/nemotron-viz-ref-params/model_52layer_manifest.json

# 3. Re-run Vitest parity verification in Node
npm test
```

---

## 2. Playwright / Chromium WebGL2 E2E Smoke Suite

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

## 3. CI & Deployment Architecture

Verification and production deployment are strictly separated into two workflows under `.github/workflows/`:

| Workflow | Triggers | Permissions | Steps |
| --- | --- | --- | --- |
| **PR & Push CI** (`.github/workflows/ci.yml`) | `pull_request` (to `main`), `push` (to `main`), `workflow_dispatch` | Read-only (`contents: read`) — **never deploys** | `actions/setup-node@v4` (Node `22`) → `npm ci` → `npm run typecheck` → `npm test` → `npm run build` → `npx playwright install --with-deps chromium` → `npm run test:e2e` (incl. `/llm-viz/` subpath check) → upload Playwright report artifact on failure only. |
| **GitHub Pages Deploy** (`.github/workflows/deploy.yml`) | `push` (`branches: [main]` only) and manual `workflow_dispatch` — **never runs on PRs** | `contents: read`, `pages: write`, `id-token: write` | `actions/setup-node@v4` (Node `22`) → `npm ci` → `npm run typecheck` → `npm test` → `VITE_BASE_PATH=/llm-viz/ npm run build` → `actions/upload-pages-artifact` → `actions/deploy-pages`. |

---

## 4. Known Limitations & Explicitly Deferred Items

- **Miniature Synthetic Simulator:** Browser execution uses synthetic Float32 weights (`d = 8`, `V_mini = 32`, max `16` interactive tokens). It does not load NVIDIA checkpoints, run the `131,072`-token Nemotron tokenizer, or simulate actual FP8 E4M3 quantization rounding.
- **Single-Sequence Logical Accounting:** Memory readouts report logical per-sequence tensor payload sizes (`47.1 MiB` fixed Mamba state + `+6 / +3 KiB/tok` KV cache) excluding model weights, quantization scale factors, allocator padding, batching, and temporary activation buffers.
- **Software vs. Hardware GPU Differences:** Headless CI exercises Chromium WebGL2 via ANGLE/SwiftShader. Visual fidelity checks on hardware GPUs (high-DPI macOS Metal / Windows D3D11) remain a targeted manual pre-release check when shaders or CSS change.
- **Deferred Architecture Work:** Splitting `src/main.ts` into feature controllers and implementing future TwoTower features (layer-aligned KV transfer UI, externally seeded Mamba state UI, bidirectional block attention, timestep conditioning, and diffusion refinement trajectories) are explicitly deferred past v1.0 (see [`docs/ARCHITECTURE.md`](ARCHITECTURE.md#4-deferred-technical-debt--future-twotower-extension-points)).
