# Verification & Reproducibility Runbook — Nemotron 3 Nano, TwoTower, Diffusion-14B, NLD-Uni & Dory1.0-4B

This runbook documents how to reproduce and verify mathematical parity, unit/contract invariants, 5-way Topology A MPA bundle isolation (`20` directional pairs), real-browser WebGL2 rendering (software SwiftShader and hardware Metal/GPU), accessibility, static artifact checksums, and CI/Pages workflows across all five applications:

- **Nemotron 3 Nano 30B-A3B** (`/` & `/llm-viz/` · `v1.0.0` preserved)
- **Nemotron-Labs-TwoTower 30B-A3B** (`/twotower/` & `/llm-viz/twotower/` · `twotower-v1.0.0`)
- **Nemotron-Labs-Diffusion-14B** (`/diffusion/` & `/llm-viz/diffusion/` · `diffusion-v1.0.0`)
- **Nemotron-Labs-Diffusion-Uni (NLD-Uni)** (`/uni/` & `/llm-viz/uni/` · `uni-v1.0.0`)
- **Nemotron-Labs-Dory1.0-4B (`Dory1.0-4B`)** (`/dory/` & `/llm-viz/dory/` · `dory-v1.0.0`)

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

# 3. Full Vitest unit, contract & PyTorch-parity suite (731 tests across 85 files; zero Python needed)
npm test

# 4. (Optional) Regenerate independent PyTorch 2.15.0 parity fixtures
#    (Nano, TwoTower Cat 1-8, Diffusion Cat 1-11, NLD-Uni Backbone/Multimodal/Diffusion, Dory1.0-4B Backbone & Spec)
node scripts/reference/export_52layer_params.mjs /tmp/nemotron-viz-ref-params
uv run scripts/reference/generate_pytorch_parity_fixtures.py \
  --model-manifest /tmp/nemotron-viz-ref-params/model_52layer_manifest.json
node scripts/reference/export_twotower_params.mjs
uv run scripts/reference/generate_twotower_pytorch_fixtures.py
node scripts/reference/export_diffusion_params.mjs
uv run scripts/reference/generate_diffusion_pytorch_fixtures.py
uv run scripts/reference/generate_diffusion_trajectory_fixtures.py
node scripts/reference/export_uni_params.mjs
uv run scripts/reference/generate_uni_backbone_pytorch_fixtures.py
node scripts/reference/export_uni_multimodal_params.mjs
uv run scripts/reference/generate_uni_multimodal_pytorch_fixtures.py
uv run scripts/reference/generate_uni_diffusion_pytorch_fixtures.py
node scripts/reference/export_dory_params.mjs
uv run scripts/reference/generate_dory_pytorch_fixtures.py
uv run scripts/reference/generate_dory_spec_fixtures.py
npm test

# 5. Root production build (`/` + `/twotower/` + `/diffusion/` + `/uni/` + `/dory/`) & 5-way 20-pair isolation check
VITE_BASE_PATH=/ npm run build
npm run verify:isolation

# 6. GitHub Pages subpath build (`/llm-viz/` + `/llm-viz/twotower/` + `/llm-viz/diffusion/` + `/llm-viz/uni/` + `/llm-viz/dory/`) & 5-way isolation check
VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages

# 7. Real Chromium WebGL2 E2E smoke suite across all 5 apps & both base paths
#    (first time: npx playwright install chromium)
npm run test:e2e
```

---

## 2. Vitest Unit, Contract & PyTorch `2.15.0` Parity Suite (`731` Tests Across `85` Files)

The Node regression suite runs `85` test files (`33` Nano files + `13` TwoTower files + `11` Diffusion files + `18` NLD-Uni files + `9` Dory1.0-4B files + `1` shared navigation switcher contract file `tests/navigation-switcher-contract.test.ts` [`4` tests]) totaling **`731` tests** (`236` Nano + `133` TwoTower + `70` Diffusion + `188` NLD-Uni + `100` Dory1.0-4B + `4` shared switcher):

### A. Nano `v1.0.0` Core Suite (`236` Tests Across `33` Files)

| Category | Test Files | Verified Properties |
| --- | --- | --- |
| **Spec & Logical Memory** | `spec.test.ts`, `memory.test.ts` | Exact 52-layer `hybrid_override_pattern` (`23 M · 23 E · 6 *`), 4 Figure 2 brackets, 12 sensitive BF16 bays, `47.08 MiB` (`49,364,992 B`) fixed Mamba state, `+6 / +3 KiB/tok` KV growth, `1 → 1M` crossovers (`1,048 / 2,096` and `8,035 / 16,070` tokens). |
| **Single-Layer Simulators** | `gqaSim.test.ts`, `mambaSim.test.ts`, `moeSim.test.ts` | Exact `NemotronH` equations: NoPE `32Q/2KV` GQA, 4-tap Conv1D + FP32 SSM + gate-first 8-group `MambaRMSNormGated`, biased MoE selection + unbiased Top-6 `×2.5` normalization + non-gated `ReLU²` + double-width shared expert. |
| **Unified 52-Layer & Incremental AR Decode** | `hybrid-flow.test.ts`, `incremental-decode.test.ts`, `hardening-math-state.test.ts` | Bit-exact (`0` ULP) full-sequence vs. prefill+incremental and arbitrary chunk equivalence, `75.00 KiB` default snapshot accounting, length-prefixed `stateDigest` tamper rejection, isolated lazy reconstruction, 52-bay weight LRU sharing, active-branch-safe branch deletion & bounded pruning (`maxBranches=16`, `maxUniqueSteps=128`). |
| **Independent PyTorch `2.15.0` Parity** | `pytorch-parity.test.ts`, `incremental-decode.test.ts` | Single-layer, full 52-layer end-to-end, and 52-layer cached incremental AR decode parity against checked-in PyTorch `2.15.0` JSON fixtures, with strict `NaN`/`±Infinity` rejection (`tests/helpers/parityAssert.ts`). |
| **3D Cutaways, Line Packing & Focused Accumulation** | `cutaways.test.ts`, `bookend-cutaways.test.ts`, `gqa-foundation.test.ts`, `gqa-spatial.test.ts`, `mamba-spatial.test.ts`, `moe-spatial.test.ts`, `unified-cutaways.test.ts`, `incremental-cutaways.test.ts`, `line-packing.test.ts`, `spatial-primitives.test.ts`, `visual-renderer.test.ts`, `focused-accumulation.test.ts`, `v2-gqa-reveal.test.ts`, `v2-mamba-reveal.test.ts`, `v2-moe-reveal.test.ts` | Shared 18-float `LinesPass` guide-line packing across all 8 cutaway builders, `RG32F` atlas packing, progressive reveal frontiers, focused-cell partial sums, running MoE `Σσ` denominator without premature weight exposure, and Output Top-5 label visibility. |
| **Walkthrough, Controls, Picking & A11y/Perf** | `walkthrough.test.ts`, `walkthrough-beats.test.ts`, `v5-controls.test.ts`, `phase1-camera-ui.test.ts`, `phase1-focus.test.ts`, `pick.test.ts`, `overlay-callout.test.ts`, `rc-continuity-integration.test.ts`, `hardening-a11y-perf.test.ts` | 22-beat 3-Act walkthrough continuity, isolated Tour vs. Free Explore AR sessions, native `Space`/`Enter` deferral, WAI-ARIA `role="tab"` roving `tabindex` and modal focus traps, `RenderDevice` FBO leak/latch guards, idle-RAF gating, `RG32F` atlas content-key dedup, and spine-wave geometry throttling. |

### B. TwoTower `v1.0.0` Suite (`133` Tests Across `13` Files)

| Category | Test File | Verified Properties |
| --- | --- | --- |
| **Spec, adaLN Modulation & Sampler** | `twotower-spec-mod-sampler.test.ts` | Pinned HF revision `35b6498508be8856d380cdfe4a900a79b918d1ea`, `0`-byte analytical vs. 24-shard safetensors reconciliation (`63,185,902,848` params / `126,371,817,472 B`), 9-entry discrepancy ledger, `twoTowerMemoryFor`, COS→SIN sinusoidal embedding, split `Mod→Norm` (`M`/`*`) vs. `Norm→Mod` (`E`), MDLM/SUBS unmasking & deterministic Gumbel-max. |
| **Shared Dual-Tower Kernels** | `twotower-kernels.test.ts` | `RouterPrecisionPolicy` (`float64_reference` vs. `float32_nano`), `AdaRMSNorm` hook, bidirectional rectangular `[B × (P+B)]` GQA attention & future-token influence, externally seeded Mamba-2 stitching, `V = 34` extension, and deterministic non-aliased Denoiser perturbation. |
| **Parameters, Ownership & Engine** | `twotower-engine.test.ts` | Non-aliased Context/Denoiser/TimeConditioning weights (`860` tensors / `600,120` Float32s), `TwoTowerContextSnapshot` & read-only `TwoTowerContextConditioningState` digest/tamper guards, `0`-ULP Context immutability, `0`-ULP AR vs. Mock-AR identity equivalence, lazy `traceOrigin` reconstruction, and headless / zero-`Math.random()` guardrails. |
| **Independent PyTorch `2.15.0` Parity (8 Categories)** | `twotower-pytorch-parity.test.ts` | All 8 mandatory TwoTower verification categories against 3 checked-in PyTorch `2.15.0` fixtures (split-adaLN + order-inversion sensitivity, bidirectional GQA, seeded & `O(1e3)` Mamba-2, FP64 MoE near-tie/exact-tie routing, 52-layer handoff, 2-block trajectory, AR vs. Mock-AR alignment, MDLM/Gumbel edge cases). |
| **Interactive Session & Branching** | `twotower-session.test.ts` | 4-step block diffusion lifecycle, `AR`/`Mock-AR` switching, non-destructive scrubbing, auto-forking, active-branch-safe deletion, bounded retention, and digest provenance. |
| **Context & Sampler Contracts** | `twotower-context-sampler-contracts.test.ts` | Strict Context ownership invariants, non-retroactive config scope separation (`record.configAtStep` vs. `session.pendingConfig`), independent `B`/`K` budget floors, calibration metadata, and zero-`Infinity` sampler presentations. |
| **Spatial Macro Scene** | `twotower-spatial-macro.test.ts` | Dual-tower `52×2` macro geometry, `29` aligned transfer linkers vs. `23` stateless MoE bays, linker channel filters, instructional Macro Flow Replay, and 10-axis bay picking. |
| **Spatial Paired GQA & Mamba-2 Cutaways** | `twotower-spatial-gqa-mamba.test.ts` (`12` tests) | Asymmetric 35/65 Paired GQA (`128×128` `RG32F` atlas, bidirectional `[B × (P+B)]`) and Paired Mamba-2 (`256×128` atlas, 4-tap Conv1D `3→2→1→0` boundary replacement, `4×4` analytical SSM decomposition, plus Mamba partial-accumulation non-exposure test). |
| **Spatial Timestep, MoE & Bookend Cutaways** | `twotower-spatial-timestep-moe-bookends.test.ts` | Dedicated Timestep Conditioning cutaway (`Mod→Norm` vs. `Norm→Mod`, unmodulated `norm_f`), Granular MoE `128+1` FP64 6th-vs-7th margin & non-spoiling `Σσ` denominator, and `V = 34` Embedding/Output cutaways with zero-`Infinity` MDLM output. |
| **Presentation Correctness & Provenance** | `twotower-wave1-correctness.test.ts` | Canonical `t = nMasked / B` (never `r / K`), 5-way `conditioningMode`, historical Context boundary provenance (`P_cond` vs. current `P` across all 6 spatial scene builders + `pruned_unavailable` & `historicalConfig` invariants), and logical miniature inventory vs. selected atlas slice separation. |
| **Production Architectural Readouts & Scaling** | `twotower-arch-readouts.test.ts` | Dual SI (`GB`/`MB`) & IEC (`GiB`/`MiB`) checkpoint inventory, FP32 `e_score_correction_bias` (`+11,776 B`), 3-regime context-length scaling (`≤ 128K` operational · `128K–256K` configured limit · `> 256K` extrapolated), 4-way memory separation with explicit peak runtime workspace exclusion, persistent nested timeline, and advanced sampler view-models. |
| **Guided 4-Act / 28-Beat Walkthrough** | `twotower-tour.test.ts` | All 4 Acts / 28 Beats navigation, dynamic trajectory-derived live metrics (zero fabricated values, zero `Infinity`/`NaN`), strict Tour vs. Free Explore session isolation, and exact state restoration on exit. |
| **Reciprocal Navigation & Release Metadata Contract** | `twotower-navigation-contract.test.ts` (`3` tests) | Base-path-safe Nano → TwoTower link (`index.html` `%BASE_URL%twotower/`, `.top-nav-link` focus styling, `0` forbidden Nano markers), clean release metadata (`twotower/index.html`, `diagnosticUi.ts` `role="region"`, zero internal stage tags), and TwoTower → Nano link ordering in `spatialApp.ts`. |

### C. Nemotron-Labs-Diffusion-14B Suite (`70` Tests Across `11` Files)

| Category | Test File | Verified Properties |
| --- | --- | --- |
| **Spec, 0 B LFS Inventory, Masks & `D1–D24`** | `diffusion-spec-masks.test.ts` (`13` tests) | Pinned `nvidia/Nemotron-Labs-Diffusion-14B` (`f8c3e2c0`), `0 B` reconciliation across all `13` LFS safetensors files (`14B` `25.16 GiB` BF16 + `180.00 MiB` FP32 LoRA), `D1–D24` ledger, `M_OBC` `Δp >= 1`, 5 mask builders, `D23` 3-coordinate divergence, `kvCapacity = 32` boundary. |
| **PyTorch `2.15.0` Parity (Categories 1–6)** | `diffusion-kernels-parity.test.ts` (`6` tests) | Exact dyadic YaRN `inv_freq` (`0`-ULP), Llama-4 `1.0 → 1.06931472` scale, 5-mode masks, `32Q/8KV` GQA + draft `o_proj` LoRA (`+4.0·B(A·a)`), Dense SwiGLU, and full 40L forward pass (`maxAbs <= 1.907e-6`, exact Top-5). |
| **PyTorch `2.15.0` Parity (Categories 7–11)** | `diffusion-engine.test.ts` (`5` tests) | Cached incremental AR vs. full 40L (`0`-ULP TS equivalence), strict Sampler A vs. Sampler B separation, speculative KV rollback & `D19` dual-crop provenance, `0`-ULP verify LoRA-OFF invariance, and deterministic stochastic verification (`D11`). |
| **Session, Branching & `0`-ULP Reconstruction** | `diffusion-session.test.ts` (`6` tests) | `KvProvenanceMismatchError` re-prefill guard, copy-on-step branching, active-only eager trace retention (`[0, 19, 39]`), `0`-ULP lazy trace & cropped suffix reconstruction, active-branch-safe deletion. |
| **Guards, `D12` NFE & Category 12 Memory Cap** | `diffusion-guards-memory.test.ts` (`6` tests) | `D18` infinite `[MASK]` draft loop guard (`non_progress_mask_loop` & `maxDraftPassesPerCycle = B`), `D12` 3-counter NFE + `D21` prefill-EOS, and adversarial 12-branch / 128-record `B = 8` stress test (`5,097,012 B` = **`4.86 MiB <= 12.00 MiB`**, `145` unique shared slabs). |
| **Spatial Macro 40L & RoPE+YaRN+GQA+LoRA** | `diffusion-spatial-macro-rope.test.ts` (`7` tests) | 40L spine + `0..31` KV rail + detachable LoRA bridge, `256×128` `RG32F` GQA+RoPE+LoRA atlas packing, 2D RoPE phasor telemetry, `p = 16` Llama-4 jump, projection-width reversal disclosure. |
| **Spatial SwiGLU, 5-Tab Masks, Spec & Bookends** | `diffusion-spatial-swiglu-masks-spec.test.ts` (`6` tests) | `64×64` SwiGLU progressive accumulation, `64×64` 5-tab Mask Explorer live G-channel binding (`D24` guard), Spec Verify shifted alignment + `D19` crop, `V = 36` Embedding & Output `+1` vs. `+0` logits. |
| **Architecture Readouts & `D1–D24` Filter** | `diffusion-arch-readouts.test.ts` (`8` tests) | Dual `GiB/GB` & `MiB/MB` formatting, `14B` component table, 7-variant matrix, 4-tier context KV scaling (`16,384` Llama-4 boundary), 4-way browser memory boundaries, `D1–D24` search/filter, zero `Infinity`/`NaN`. |
| **Guided 4-Act / 26-Beat Walkthrough** | `diffusion-tour.test.ts` (`3` tests) | All 4 Acts / 26 Beats covering all 7 views, 5 mask tabs, and 4 modes (`causal_context` T/F), live tensor metrics, isolated `DiffusionSession`, and exact Free-Explore restoration on exit. |
| **Unified Timeline View-Model & WAI-ARIA A11y** | `diffusion-timeline-a11y.test.ts` (`6` tests) | Tri-mode timeline (`prefill`, `ar_step`, `dlm_block`, `spec_cycle`), `12.00 MiB` budget meter, roving `tabindex` (`Home`/`End`/`ArrowRight`), `shouldDeferGlobalShortcut` deferral, `Escape` precedence, reduced-motion helper. |
| **3-Way Navigation & Release Metadata Contract** | `diffusion-navigation-contract.test.ts` (`4` tests) | Base-path-safe `nav-to-diffusion` (`%BASE_URL%diffusion/`) in `index.html` and `src/twotower/app/spatialApp.ts`, `nav-to-nano` / `nav-to-twotower` in `src/diffusion/app/spatialApp.ts`, clean `diffusion/index.html`, zero cross-app imports or markers. |

### D. Nemotron-Labs-Diffusion-Uni (`uni-v1.0.0`) Suite (`188` Tests Across `18` Files)

| Category | Test Files | Verified Properties |
| --- | --- | --- |
| **Spec, 0 B 8-Shard Inventory & `U1–U30`** | `uni-spec.test.ts` (`16` tests) | Pinned `nvidia/NLD-Uni` (`5db755a2`), `0 B` reconciliation across `8` safetensors shards (`1,557` tensors = `18,267,552,262` params = `36,535,104,542 B`), `U1–U30` ledger, `D1–D5` locked decisions, and 4-tier context honesty (`U3`). |
| **34L Elastic-MoT Backbone & PyTorch `2.15.0` Parity** | `uni-backbone-math.test.ts` (`14`), `uni-pytorch-backbone-parity.test.ts` (`9`) | `18` Joint `L0..17` + `16` Split `L18..33` twin `U`/`G` towers (`0` shared learned weights), 1D dyadic YaRN + Llama-4 `p = 16` + 2D `SinCos-G`, `D5` `HF-Faithful` vs. `Elastic-Pruned` `0`-ULP Stage-2 equivalence, and PyTorch `2.15.0` parity (`pytorch-uni-backbone.json`). |
| **Multimodal Vision, VQ-VAE, Codec & Understanding Parity** | `uni-vision.test.ts` (`16`), `uni-vq-codec.test.ts` (`9`), `uni-understanding-parity.test.ts` (`13`) | `RADIO-H` + `2×2` SquaredReLU projector, `Emu3` IBQ + `2×2` `PackedLinearImageCodec`, FLARE shifted-logit understanding, `<LOC_0..16>` 2D grounding (`U27`), and PyTorch `2.15.0` parity (`pytorch-uni-multimodal.json`). |
| **Semantic Ball-Tree, Diffusion & OPD Parity** | `uni-semantic-tree.test.ts` (`19`), `uni-diffusion-parity.test.ts` (`11`), `uni-opd-readouts.test.ts` (`10`) | `BalancedSemanticTree` (`D = 8` mini / `U26` `D = 17` floor-rounding mismatch), stratified quadtree refinement, Dual-CFG edit (`legacy`/`independent`/`ip2p`, `U18` border-lock), Read-Only OPD (`U28`), and PyTorch `2.15.0` parity (`pytorch-uni-diffusion.json`). |
| **Session Stress, All 7 Spatial Scenes, Tour & App Contracts** | `uni-session-stress.test.ts` (`7`), `uni-macro-scene.test.ts` (`7`), `uni-visual-input-scene.test.ts` (`7`), `uni-joint-attn-scene.test.ts` (`6`), `uni-ball-tree-scene.test.ts` (`5`), `uni-refinement-edit-scene.test.ts` (`9`), `uni-opd-scene.test.ts` (`6`), `uni-walkthrough.test.ts` (`5`), `uni-app-contracts.test.ts` (`19`) | `<= 12.00 MiB` `Set<ArrayBufferLike>` session budget, all 7 WebGL2 scenes + 2D RGB canvas insets, 5-act / 28-beat isolated tour, WAI-ARIA helpers, zero-idle-RAF, and `D4` isolation contracts. |

### E. Nemotron-Labs-Dory1.0-4B (`dory-v1.0.0`) Suite (`100` Tests Across `9` Files)

| Category | Test File | Verified Properties |
| --- | --- | --- |
| **Backbone Math & PyTorch `2.15.0` Parity** | `dory-math-parity.test.ts` | Cryptographic `SHA-256` match (`cd3a8ad3...`, `507` tensors / `93,248` floats = `372,992 B`), `DR1–DR24` & `D1–D10`, single-op nGPT kernels (`<= 1e-6`), planted `|α_hat| > 1` NLERP (`DR7`), sublayers (`<= 1e-5`), full 72L `k ∈ {1..4, 6}` (`<= 2e-5`), `per_loop` long-chunk `L = 24, 32 > cap_swa = 16` (`DR24`), `last_loop` Theorems 1–3 + constructive `max_abs > 1e-2` divergence (`DR17`) + `W_v = 0` counterexample, and upstream differential variants (`DR5/DR6/DR8/DR9`). |
| **Contract A §6.2 Self-Spec (6-Layer Parity)** | `dory-speculative-parity.test.ts` | All 6 verification layers against `pytorch-dory-spec.json`: (1) binary64 algebraic identity `<= 1e-15`, (2) `V = 4` exact analytical enumeration `<= 1e-15`, (3) explicit-variate boundary checks, (4) adversarial edges (`p === q`, disjoint, `q = 1e-40`, `Δ = 4e-13`, one-hot), (5) multi-cycle Contract A `104`-bank `k = 4` verify & `output_shallow` `k = 1` draft parity (`<= 2e-5`) across `S > 8` rollback & greedy `===` AR, and (6) Pearson `χ² < 103.51` (`N = 20,000, α = 0.001, df = 63`) + mandatory failing negative controls. |
| **Session Immutability, `D10` LRU Pool & Stress** | `dory-session-stress.test.ts` | Non-aliased CoW branch isolation, `DR18` immutable depth/policy fork guard, `DR24` `last_loop` `chunkSizes[]` `0-ULP` lazy trace replay, and 12-branch / 128-record stress strictly `<= 12.00 MiB` at **both `k = 4` and `k = 8`** via `D10` `3.00 MiB` LRU sparse-checkpoint pool. |
| **Spatial Provenance, Views 1–6 & `DR20`/`DR21`** | `dory-scene-provenance.test.ts`, `dory-scene-macro-hypersphere.test.ts`, `dory-scene-mixers-kv-spec.test.ts` | `0` `NaN`/`Inf`, `0` atlas overlaps, `<= 512 KiB` `RG32F`, `DR20` throw guard on unmasked SWA/Full plates, `DR21` `pick.ts` immutability, Folded/Unrolled shared weights, anti-SLERP / Exploratory 2D PCA (`8D` metrics) / read-only Counterfactual Coda, Dual-RoPE GQA, SwiGLU, `kv_compare`, and Contract A `self_spec`. |
| **View 7, Readouts & 4-Act / 14-Beat Tour** | `dory-readouts-tour.test.ts` | `0 B` diff 4-checkpoint constants, `$S$-invariant (4 + 5k) / 9` ratio, `READ-ONLY · NO GRADIENTS` TBPTL guard, full 8-tuple `DoryExecutionCoord` validation on all 14 beats, and byte-exact isolation from external Free-Explore `DorySession`. |
| **App Shell, Charts, Overlay & Static `D4` Guard** | `dory-app-contracts.test.ts` | `idle-RAF`, dual-hash atlas dedup, `<= 64` pooled overlay cells, 6 headless `<svg>` charts, in-place slider/dock DOM identity, and recursive static guard banning Gate 4b/4c Nano modules, sibling entries, and `Math.random(`. |
| **5-Way MPA Navigation & Release Metadata** | `dory-navigation-contract.test.ts` (`6` tests) | 5-entry `MODEL_REGISTRY` (`/` & `/llm-viz/`), clean `dory/index.html` & `vite.config.ts`, `0` forbidden Dory markers across all 4 prior apps & shared switcher, light-theme switcher mount, `verify_bundle_isolation.mjs` registration, and 5-way WAI-ARIA keyboard navigation. |

### Independent PyTorch `2.15.0` Parity Tolerances

TypeScript accumulates inner dot products in JS `float64` prior to binary32 write-backs while PyTorch `torch.float32` accumulates in `float32` BLAS/SIMD. All `13` checked-in SHA-256-verified independent PyTorch `2.15.0` reference fixtures (`tests/fixtures/*.json`) pass under Vitest:

| Fixture File | Scope | Observed / Enforced Max Absolute Tolerance |
| --- | --- | --- |
| `tests/fixtures/pytorch-reference-parity.json` | Nano single-layer components (`RMSNorm` `ε=1e-5`/`1e-6`, Embedding NoPE, LM Head, GQA `32Q/2KV`, Mamba-2, Granular MoE `128+1`) | Embedding: **`0`** · RMSNorm / LM Head / GQA: **`<= 1e-5`** · Mamba-2 / MoE: **`<= 2e-5`** · MoE Top-6 indices: **exact match** |
| `tests/fixtures/pytorch-52layer-parity.json` | Nano full 52-layer chain (`299,192` Float32s, SHA-256 `489d45a05d24317a4bb6ee4bb21bed115c3d750c887d9faf8ac44ee5d77a398b`, `parameterFingerprint: dc0568e222e22a4e`) | Embedding: **`0`** · Residuals / Final Norm / Probs: **`<= 2e-5`** · Logits: **`<= 5e-5`** · All 23 MoE Top-6 & Top-5 rankings: **exact match** |
| `tests/fixtures/pytorch-incremental-52layer-parity.json` | Nano 52-layer cached incremental AR decode + externally seeded non-zero continuation | Residuals / caches / scores / softmax / probs: **`<= 2e-5`** · Logits: **`<= 5e-5`** · TS full-vs-incremental: **`0` ULP** |
| `tests/fixtures/pytorch-twotower-blocks-parity.json` | TwoTower Categories 1–4 (`600,120` Float32s, SHA-256 `1bd365e2d5ae0ac11893446c60f59b317492f203bbeb6745a21c44cb5f8130cf`) | Modulation & GQA: **`<= 1e-5`** · Mamba-2 & MoE: **`<= 2e-5`** · High-magnitude Mamba-2 rel: **`<= 1e-5`** · MoE Top-6 indices: **exact match** |
| `tests/fixtures/pytorch-twotower-handoff-parity.json` | TwoTower Category 5: complete 52-layer Context-to-Denoiser handoff (`P = 8, B = 4, kvCapacity = 24, t = 1.0`) | Embedding: **`0`** · Modulation: **`<= 1e-5`** · Caches / residuals / traces / norm / probs: **`<= 2e-5`** · Logits: **`<= 5e-5`** · Context immutability: **`0` ULP** |
| `tests/fixtures/pytorch-twotower-trajectory-parity.json` | TwoTower Categories 6–8: 2-block diffusion trajectory, AR vs. Mock-AR, MDLM/SUBS & Gumbel-max | Caches / residuals / probs: **`<= 2e-5`** · Logits: **`<= 5e-5`** · Sampler log-probs & probs: **`<= 1e-5`** · Identity AR vs. Mock-AR: **`0` ULP** |
| `tests/fixtures/pytorch-diffusion-blocks-parity.json` | Diffusion Categories 1–6 (`271,304` Float32s, dyadic YaRN `inv_freq`, Llama-4 `1.0 → 1.06931472`, 5-mode masks + `M_OBC` `Δp >= 1`, `32Q/8KV` GQA + LoRA, Dense SwiGLU, 40L forward) | `inv_freq`: **`0`-ULP exact dyadic** · `cat1_params_lfs` / `cat3_masks`: **`0.000e+0`** · `cat2_rope_cos_sin_llama4`: **`5.960e-8`** · `cat2_rope_qk_rot`: **`2.865e-7`** · `cat4_gqa_lora`: **`1.907e-6`** · `cat5_swiglu`: **`3.576e-7`** · `cat6_40layer_resid_norm_probs`: **`1.013e-6`** · `cat6_40layer_logits`: **`1.073e-6`** · Exact 5-mode masks & Top-5 rankings |
| `tests/fixtures/pytorch-diffusion-trajectories-parity.json` | Diffusion Categories 7–11 (cached AR vs. full 40L, Sampler A & B, speculative KV rollback & `D19` EOS crop, `0`-ULP verify LoRA-OFF, stochastic verification) | `cat7_logits_vs_pytorch`: **`2.384e-6`** · `cat7_probs_kv_vs_pytorch`: **`2.623e-6`** · `cat7_ts_incremental_vs_full`: **`0.000e+0`** · `cat8_probs_conf`: **`2.384e-7`** · `cat9_cropped_kv_vs_pytorch`: **`2.503e-6`** · `cat10_verify_logits_vs_pytorch`: **`1.825e-6`** · `cat10_draft_logits_vs_pytorch`: **`2.198e-6`** · `cat11_stochastic_logits_vs_pytorch`: **`1.550e-6`** · Exact greedy `AR == LinearSpec == LinearSpecLoRA` · `0`-ULP verify LoRA-OFF & prefix immutability |
| `tests/fixtures/pytorch-uni-backbone.json` | NLD-Uni 34L Elastic-MoT backbone (18 Joint + 16 Split, 1D dyadic YaRN + Llama-4 `p = 16` + 2D `SinCos-G`, `D5` `HF-Faithful` vs. `Elastic-Pruned`) | Dyadic YaRN `inv_freq`: **`0`-ULP** · Backbone logits/residuals/KV: **`<= 5e-5`** · `D5` Stage-2 equivalence: **`0` ULP** |
| `tests/fixtures/pytorch-uni-multimodal.json` | NLD-Uni Multimodal Understanding (`RADIO-H`, `Emu3` IBQ, `2×2` `PackedLinearImageCodec`, FLARE shifted-logit understanding, `<LOC_0..16>` grounding) | Patch/codebook/codec projections & logits: **`<= 5e-5`** · Argmax VQ codes & `<LOC_*>` tokens: **exact match** |
| `tests/fixtures/pytorch-uni-diffusion.json` | NLD-Uni Discrete Diffusion (`BalancedSemanticTree`, stratified quadtree refinement, Dual-CFG edit `legacy`/`independent`/`ip2p`, Read-Only OPD `tail_kl`) | Tree/CFG logits & probs: **`<= 5e-5`** · Selected codes, `U18` border-lock & OPD buckets: **exact match** |
| `tests/fixtures/pytorch-dory-backbone.json` | Dory1.0-4B 72L Full-nGPT backbone (`507` tensors / `93,248` F32, SHA-256 `cd3a8ad3ff987fd9e175a3d1f593322965a171446ce971e357bbf79353defa1b`), 5 scale families, Dual-RoPE, `k ∈ {1..4, 6}`, `per_loop` long-chunk (`L = 24, 32`), `last_loop` Theorems 1–3, & upstream differential variants | Single-op: **`<= 1e-6`** · Sublayers: **`<= 1e-5`** · Full 72L & `per_loop`/`last_loop` chunks: **`<= 2e-5`** · `last_loop` Theorem 3 divergence: **`> 1e-2`** |
| `tests/fixtures/pytorch-dory-spec.json` | Dory1.0-4B Paper §6.2 Contract A Self-Speculation (`104` Dual-Coda banks, `k = 1 → 4`, `γ = 4`, binary64 division-free rejection & residual, 6-layer hierarchy + 2 failing negative controls) | Binary64 algebraic & `V = 4` exact enum: **`<= 1e-15`** · Multi-cycle `k = 4` verify & `output_shallow` `k = 1` draft: **`<= 2e-5`** · Greedy: **`===` AR** · Pearson `χ² < 103.51` (`N = 20,000, α = 0.001`) |

---

## 3. Topology A 5-Way Multi-Entry MPA Build & Bundle-Isolation Gate

Run the automated 7-gate N-way verifier (`scripts/verify_bundle_isolation.mjs`) against **both** root (`/`) and GitHub Pages (`/llm-viz/`) builds:

```bash
# Root build (`/` + `/twotower/` + `/diffusion/` + `/uni/` + `/dory/`)
VITE_BASE_PATH=/ npm run build
npm run verify:isolation

# GitHub Pages subpath build (`/llm-viz/` + `/llm-viz/twotower/` + `/llm-viz/diffusion/` + `/llm-viz/uni/` + `/llm-viz/dory/`)
VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages
```

Enforced hard gates across all configured entries (non-zero exit on any failure):

1. All 5 HTML entries exist: `index.html` (Nano), `twotower/index.html` (TwoTower), `diffusion/index.html` (Diffusion), `uni/index.html` (Uni), and `dory/index.html` (Dory).
2. Every `<script src>`, `<link rel="modulepreload" href>`, and `<link rel="stylesheet" href>` resolves cleanly under `/` or `/llm-viz/`.
3. TypeScript ECMAScript AST walks every transitive static `import`/`export ... from` and literal dynamic `import(...)` edge across all graphs.
4. Strict 5-way sourcemap mutual isolation across all **`20` directional pairs** plus **Gate 4b** (zero Nano UI/controller/walkthrough/cutaway-exclusive sources in any non-Nano entry, generalizing `D4`) and **Gate 4c** (zero Nano/TwoTower hybrid core sources in any non-hybrid entry: Diffusion, Uni, and Dory).
5. Strict 5-way payload marker mutual isolation across all **`20` directional pairs**.
6. Positive presence: Nano (`2`), TwoTower (`22`), Diffusion (`31`), Uni (`36`), and Dory (`32`) transitive JS sourcemaps positively contain their own exclusive sources (`0` runtime dependencies).
7. Nano payload budget gate (`<= 215.00 KiB` / `220,160 B` gzip, verified on both `/` and `/llm-viz/`):
   - **Nano (5 JS, 2 CSS):** `774,270 B` (`/`) / `774,278 B` (`/llm-viz/`) raw | **`216,915 B` (`211.83 KiB`)** `/` / **`216,926 B` (`211.84 KiB`)** `/llm-viz/` gzip (**`+3.17 KiB` / `+3,245 B` headroom**, `0` leaks).
   - **TwoTower (5 JS, 2 CSS):** `1,129,331 B` (`/`) / `1,129,339 B` (`/llm-viz/`) raw | **`321,265 B` (`313.74 KiB`)** `/` / **`321,274 B` (`313.74 KiB`)** `/llm-viz/` gzip (`22` `src/twotower/*` sources, `0` leaks, `nanoUI=0`).
   - **Diffusion (4 JS, 2 CSS):** `492,139 B` (`/`) / `492,147 B` (`/llm-viz/`) raw | **`147,294 B` (`143.84 KiB`)** `/` / **`147,301 B` (`143.85 KiB`)** `/llm-viz/` gzip (`31` `src/diffusion/*` sources, `0` leaks, `nanoUI=0`, `nonHybridCore=0`).
   - **NLD-Uni (2 JS, 2 CSS):** `578,473 B` (`/`) / `578,481 B` (`/llm-viz/`) raw | **`169,458 B` (`165.49 KiB`)** `/` / **`169,464 B` (`165.49 KiB`)** `/llm-viz/` gzip (`36` `src/uni/*` sources, `0` leaks, `nanoUI=0`, `nonHybridCore=0`).
   - **Dory1.0-4B (3 JS, 2 CSS):** `753,226 B` (`/`) / `753,242 B` (`/llm-viz/`) raw | **`217,516 B` (`212.42 KiB`)** `/` / **`217,531 B` (`212.43 KiB`)** `/llm-viz/` gzip (`32` `src/dory/*` sources, `0` leaks, `nanoUI=0`, `nonHybridCore=0`).
   - **Shared Chunks (4 JS, 1 CSS):** `assets/atlasPacker-*.js` (`shared(all5)`, `25,767 B` gzip) + `assets/atlasPacker-*.css` (`shared(all5)`, `1,178 B` gzip) + `assets/moeCutaway-*.js` (`shared(n+tt)`, `94,962 B` gzip) + `assets/overlayLayout-*.js` (`shared(n+tt+df+dory)`, `2,548 B` gzip) + `assets/sheetInspector-*.js` (`shared(n+tt+df)`, `2,102 B` gzip) = `424,128 B` (`414.19 KiB`) raw / **`126,557 B` (`123.59 KiB`)** gzip.

---

## 4. Playwright / Chromium WebGL2 E2E Smoke Suite (`43` Tests Across `6` Files)

`playwright.config.ts` starts two isolated preview servers (`127.0.0.1:4173` for root `/` and `127.0.0.1:4174/llm-viz/` for GitHub Pages subpath) and runs all six E2E suites:

```bash
# First-time local browser setup
npx playwright install chromium

# Run all Nano + TwoTower + Diffusion + NLD-Uni + Dory1.0-4B WebGL2 E2E tests (SwiftShader headless Chromium)
npm run test:e2e
```

### Headless CI Software WebGL2 Flags (`playwright.config.ts`)

GPU-less Linux CI runners execute Chromium via ANGLE/SwiftShader using all 5 required flags:

- `--use-gl=angle`
- `--use-angle=swiftshader`
- `--enable-webgl`
- `--ignore-gpu-blocklist`
- `--enable-unsafe-swiftshader` *(required by Chromium 137+ headless on GPU-less runners)*

Every test fails on any uncaught `pageerror`, `console.error`, failed network request, HTTP `>= 400` response, or unexpected WebGL fallback activation.

### Hardware GPU / macOS Metal Verification

Software SwiftShader exercises shader compilation and WebGL2 correctness but does not substitute for hardware GPU rendering/compositing checks. Before tagging a release, run a targeted hardware GPU pass on a local workstation (e.g. Apple Silicon / macOS Metal or Windows D3D11):

```bash
# 1. Start local production preview for root (or /llm-viz/)
VITE_BASE_PATH=/ npm run build
npx vite preview --outDir dist --base / --port 4173 --strictPort --host 127.0.0.1

# 2. Open http://127.0.0.1:4173/, http://127.0.0.1:4173/twotower/, and http://127.0.0.1:4173/diffusion/
#    Verify in DevTools / telemetry:
#    - WebGL2 renderer reports hardware GPU (not SwiftShader)
#    - Continuous orbit/fly enters idle RAF sleep cleanly when stationary (rafActive === false)
#    - High-DPI crispness (devicePixelRatio >= 2) across all Nano, TwoTower & Diffusion views + 3-way links
#    - Zero console errors, zero WebGL errors, zero layout clipping at 1440×900, 1024×800, and 390×844
```

### Performance Telemetry Interpretation Note

> [!NOTE]
> **Application-Side JS Callback Time vs. Full GPU Execution Latency**
> Measurements captured with `performance.now()` around `drawFrame()` (`lastFrameMs` / `JS cb`) quantify **application-side main-thread CPU processing and WebGL command submission time** — they do **NOT** measure full asynchronous GPU rasterization, fragment shading, or display compositor latency. Observed smoothness during active camera flies or replay is tracked separately via continuous **observed RAF interval cadence (`RAF Δ`)**, measured strictly between consecutive animating frames (excluding idle-sleep pauses).

### Covered E2E Scenarios (`43` Tests Across `6` Files, Shared Helpers in `tests/e2e/helpers.ts`)

#### Nano E2E Suite (`tests/e2e/smoke.spec.ts` — `9` Tests, hardened with observable `waitForProjectedOverlaySettled`)
1. **52-Layer Tower Boot** — Clean initial WebGL2 frame, zero console/page errors, `#webgl-fallback` hidden.
2. **All 5 Exploded Cutaways** — Switching across Embedding, Mamba-2, MoE `128+1`, GQA, and Output/LM Head with observable `waitForProjectedOverlaySettled` camera-fly settlement and live landmarks.
3. **`LinesPass` Guide-Line Geometry** — Rendered guide lines verified across Embedding, Mamba, MoE, and GQA cutaways.
4. **Guided Tour & AR Session Isolation** — Beats `7.1–7.3` continuity, AR branching, and Tour vs. Free Explore isolation.
5. **Native Keyboard & WAI-ARIA Compliance** — Native `Space`/`Enter` control activation, roving `tabindex`, and modal focus capture/restore (`?` and `M`).
6. **Reduced-Motion Emulation** — Emulating and toggling `prefers-reduced-motion: reduce`.
7. **Output Top-5, MoE Running `Σσ` & Popover Containment** — Observable `waitForProjectedOverlaySettled` `#1..#5` non-overlapping Output pills, MoE Step 3 running `Σσ` masking, and popover viewport containment.
8. **Responsive Viewports** — `1440×900`, `1024×800`, and mobile `390×844` layout containment.
9. **GitHub Pages `/llm-viz/` Subpath Build** — HTTP `200` assets, zero `404`s, functional cutaways.

#### TwoTower E2E Suite (`tests/e2e/twotower-smoke.spec.ts` — `11` Tests)
1. **Root `/twotower/` Dual-Tower Macro Boot** — Non-blank WebGL2 render, epistemic pill, linker channels, 10-axis bay picking, and switcher contract.
2. **Full Diffusion Lifecycle & All 7 Spatial Views** — `Init` → `Refine` → `Commit Conf` → `Commit → Context`, `128×128` GQA, Mamba Conv1D replacement, MoE partial-denom guard, Output zero-`Infinity`.
3. **AR, Mock-AR (`BYPASSED`), Rewind, Branch Fork & Synced Workbench Drawer (`W`)**.
4. **Reduced-Motion, Idle-RAF Gating, Native `Space`/`Enter` & Narrow `1024×800` Layout**.
5. **GitHub Pages `/llm-viz/twotower/` Subpath Build & HTTP `200` Assets**.
6. **Guided 4-Act / 28-Beat Walkthrough (`T`) & Strict User-Session Restoration**.
7. **Persistent Nested Timeline, Completed-Uncommitted vs. Committed Boundary Warning & Branching**.
8. **Advanced Sampler Inspector (`0.22` / `0.80` / `0.90` Presets, Decoupled `B`/`K`, Zero-`Infinity`, Non-Retroactive Scope)**.
9. **Production Architecture & 3-Regime Context-Length Scaling Tab (`63.19B` / `126.37 GB` / `117.69 GiB`, 4-Way Caches, Ledger, SVG Chart)**.
10. **Keyboard Navigation (`T`/Arrows/`Esc`/`W`/`Tab`), Reduced-Motion Instant Transitions, Zero Geometry Rebuilds on Orbit & Idle RAF Sleep**.
11. **GitHub Pages `/llm-viz/twotower/` Subpath Parity (Tour, Timeline, Sampler & Architecture)**.

#### Diffusion-14B E2E Suite (`tests/e2e/diffusion-smoke.spec.ts` — `5` Tests)
1. **Root `/diffusion/` Boot, All 7 Spatial Views, 5 Mask Tabs & 2D RoPE Phasor Inset** — Verifies non-blank WebGL2 and `glErrorCount === 0` across all 7 views (`macro`, `gqa_rope`, `swiglu`, `mask_explorer`, `spec_verify`, `embedding`, `output`), scrubs `p = 17` across the `p = 16` Llama-4 boundary (`1.0693`), switches rotary planes `0..3`, and sweeps all 5 Mask Explorer tabs.
2. **All 4 Live Execution Modes, Graceful Re-Prefill Flow & Unified Timeline Branching** — Steps in `ar` and `dlm_causal`, triggers `KvProvenanceMismatchError` re-prefill banner on `dlm_bidir`, confirms re-prefill, rewinds via timeline click, forks `linear_spec` and `linear_spec_lora` (`loraActiveOnDraft === true`), and verifies non-blank `spec_verify`.
3. **Guided 4-Act / 26-Beat Tour, Exact Free-Explore Restoration, 3-Tab Inspector & WAI-ARIA + Idle-RAF** — Visits Acts 1–4, asserts exact Free-Explore restoration on exit, inspects `Inspect`, `Architecture` (`25.16 GiB`, `180.00 MiB`, `16,384`, `12.00 MiB`), and `Ledger (D1–D24)` + `D19` search with zero `Infinity`/`NaN`/`undefined`, checks roving `tabindex` (`Home`/`ArrowRight`/`End`) + `Escape`, and verifies strict idle-RAF sleep (`frameDelta === 0`, `rafActive === false`).
4. **Responsive Viewports (`1440×900`, `1024×800`, `390×844`)** — Verifies non-blank WebGL2, accessible controls/switcher, zero horizontal overflow, and `glErrorCount === 0` across all 3 viewports.
5. **GitHub Pages Subpath `/llm-viz/diffusion/` Parity** — Verifies `/llm-viz/diffusion/` HTTP `200` assets, Macro & RoPE GQA render, Tour, Architecture tab, and subpath switcher hrefs.

#### NLD-Uni E2E Suite (`tests/e2e/uni-smoke.spec.ts` — `8` Tests)
1. **Root `/uni/` Boot, Zero GL Errors & Idle-RAF Settled** — Verifies clean boot, non-blank Macro render, `<= 64` overlay pool, memory budget, and idle-RAF sleep.
2. **All 7 Spatial Views + `RG32F` Atlas Upload Deduplication** — Sweeps all 7 views (`macro` through `opd_readonly`) and verifies zero spurious atlas re-uploads on hover/orbit/round-trip.
3. **Live `D5` Prefill Toggle (`hf_faithful` vs. `elastic_pruned`) in Views 1 & 3** — Verifies dock & inspector toggles across `Macro` and `Joint / Split`.
4. **View 4 `U26` `D = 17` Mismatch Panel & View 7 `READ-ONLY · NO GRADIENTS` Banner** — Verifies mismatch and quarantined OPD diagnostics.
5. **2D RGB Canvas Pixel Insets on Views 2, 5 & 6 + `U18` Lock & `<LOC_*>` Grounding Pills** — Verifies non-blank 2D RGB insets, `image_edit` border locks, and `object_grounding` pills.
6. **In-Place Slider DOM Identity & Hover Controls-Slot Preservation** — Verifies sliders preserve DOM nodes/focus on input and hover never replaces controls slot.
7. **28-Beat Guided Tour + `1280×800` Zero-Overlap Layout + `Escape` Deferral + Free-Explore Restoration** — Verifies tour lock, ledger ref chips, `Escape` deferral inside search, and Free-Explore restoration.
8. **GitHub Pages Subpath `/llm-viz/uni/` Parity** — Verifies `/llm-viz/uni/` HTTP `200` assets, WebGL2 render, `D5` toggle, and subpath switcher hrefs.

#### Dory1.0-4B E2E Suite (`tests/e2e/dory-smoke.spec.ts` — `7` Tests)
1. **Root `/dory/` Boot, Zero GL Errors & Idle-RAF Settled** — Verifies clean WebGL2 boot, non-blank Macro frame (`k = 2, 112L, 56 per_loop banks`), `<= 64` overlay pool, `<= 512 KiB` atlas, `D10` memory bounds, and strict `rafActive === false` rest window.
2. **All 7 Editorial Spatial Views + 2D Analytical `<svg>` Insets + `RG32F` Atlas Deduplication** — Sweeps all 7 views (`macro` through `bookends`), verifies all 6 `<svg>` charts (`spanning_subspace`, `exploratory_pca`, `dual_rope_phasors`, `kv_scaling_256k`, `self_spec_probs`, `tbptl_schematic`), and confirms `0` spurious atlas re-uploads on hover, orbit-drag, tab re-click, or inspector toggle.
3. **Elastic Depth `k` (`1..4` & `+6,8`) + `per_loop`/`last_loop` `DR18` Immutable Forking** — Verifies `k = 2 → 4` (`192L, 96` banks), Folded/Unrolled Macro toggle, `+6,8` extrapolation (`k = 8, 352L, 176` banks bounded `<= 12 MiB`), `last_loop` (`36` frozen banks), View 5 `DR17` partition modes, and preservation of all 5 immutable branches.
4. **Contract A Paper §6.2 Self-Speculative View (`104` Banks) & View 7 Read-Only Bookends/Readouts/Ledger** — Verifies `Reconstructed from Paper §6.2` banner, live `104`-bank Contract A activation & `γ = 4` cycle, View 7 `READ-ONLY · NO GRADIENTS` banner & zero fake quant toggles, `0 B` diff 4-checkpoint Readouts table, and `DR23` Ledger search.
5. **In-Place Slider DOM Identity & Hover Controls-Slot Preservation** — Verifies exact DOM node & focus identity across sequential inputs on `dory-layer-slider`, `dory-gqa-layer-slider`, `dory-swiglu-reveal-slider`, and `dory-dock-temp-slider`, plus hover controls-slot preservation across the WebGL2 canvas.
6. **4-Act / 14-Beat Guided Tour + Tour Lock + `Escape` Deferral + Free-Explore Isolation** — Verifies representative beats across all 4 Acts & 7 views, live 8-tuple `DoryExecutionCoord`, dock/inspector lock during tour, Ledger ref chips, `Escape` deferral inside search, and identical Free-Explore restoration on exit.
7. **GitHub Pages Subpath `/llm-viz/dory/` Parity** — Verifies `/llm-viz/assets/*` HTTP `200`, non-blank WebGL2 render, all 5 subpath switcher hrefs (`aria-current="page"` on `nav-to-dory`), View 2 `<svg>` insets, and live `104`-bank Contract A activation under `/llm-viz/dory/`.

#### Unified 5-Model Switcher & Containment E2E Suite (`tests/e2e/navigation.spec.ts` — `3` Tests)
1. **Root `/` 5-Way Switcher Navigation (Click & Keyboard `ArrowDown`/`Enter`)** — Full cycle across all 5 apps (`Nano → TwoTower → Diffusion → Uni → Dory → Nano`) verifying non-blank WebGL2 canvases and `glErrorCount === 0`.
2. **GitHub Pages `/llm-viz/` Subpath 5-Way Switcher Navigation (Click & Keyboard)** — Full 5-app cycle verifying `/llm-viz/assets/*` HTTP `200`, subpath hrefs, non-blank WebGL2, and `glErrorCount === 0`.
3. **5 Apps × 4 Viewports Popover Non-Clipping (`1440×900`, `1280×800`, `1024×800`, `390×844`)** — Verifies the top-bar model switcher popover never clips horizontally or vertically, introduces `0` horizontal overflow, marks `aria-current="page"`, and restores focus on `Escape` across all `20` app × viewport combinations.

---

## 5. Static Portable & GitHub Pages Artifact Packaging (`shasum -a 256`)

To produce verifiable, self-contained static release tarballs for both root (`/`) and GitHub Pages (`/llm-viz/`) deployments:

```bash
# Clean output staging directory (gitignored)
mkdir -p test-results/release-artifacts

# 1. Portable static archive (serves at root `/`, `/twotower/`, `/diffusion/`, `/uni/`, and `/dory/`)
VITE_BASE_PATH=/ npx vite build --outDir test-results/dist-portable
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-portable
tar -czf test-results/release-artifacts/nemotron-viz-static-portable.tar.gz \
  -C test-results/dist-portable .

# 2. GitHub Pages subpath archive (serves at `/llm-viz/`, `/llm-viz/twotower/`, `/llm-viz/diffusion/`, `/llm-viz/uni/`, and `/llm-viz/dory/`)
VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages
tar -czf test-results/release-artifacts/nemotron-viz-ghpages-llm-viz.tar.gz \
  -C test-results/dist-pages .

# 3. Generate and verify SHA-256 checksums
(
  cd test-results/release-artifacts
  shasum -a 256 nemotron-viz-static-portable.tar.gz nemotron-viz-ghpages-llm-viz.tar.gz > SHA256SUMS.txt
  shasum -a 256 -c SHA256SUMS.txt
)
```

---

## 6. CI & Deployment Workflows

Verification and production deployment remain strictly separated under `.github/workflows/`:

| Workflow | Triggers | Permissions | Steps |
| --- | --- | --- | --- |
| **PR & Push CI** (`.github/workflows/ci.yml`) | `pull_request` (`main`), `push` (`main`, `feat/**`), `workflow_dispatch` | Read-only (`contents: read`) — **never deploys** | Node `22` → `npm ci` → `npm run typecheck` → `npm test` → `npm run build` + `npm run verify:isolation` (root `/`) → `VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages` + `node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages` (`/llm-viz/`) → `npx playwright install --with-deps chromium` → `PLAYWRIGHT_SKIP_BUILD=1 npm run test:e2e` → upload Playwright report on failure. |
| **GitHub Pages Deploy** (`.github/workflows/deploy.yml`) | `push` (`branches: [main]` **only**) and manual `workflow_dispatch` — **never runs on PRs or feature branches** | `contents: read`, `pages: write`, `id-token: write` | Node `22` → `npm ci` → `npm run typecheck` → `npm test` → `VITE_BASE_PATH=/llm-viz/ npm run build` → `npm run verify:isolation` → `actions/configure-pages` → `actions/upload-pages-artifact` (`./dist`) → `actions/deploy-pages`. |

---

## 7. Known Limitations & Scope Boundaries

- **Miniature Synthetic Simulators:** Browser execution uses deterministic synthetic Float32/Float64 weights (`d = 8`, `V_mini = 32` for Nano; `V_mini = 34` for TwoTower; `V_mini = 36` for Diffusion; `V_txt = 64` + `V_vq = 256` for NLD-Uni; `V_mini = 64` across `72` sublayers / `507` tensors for Dory1.0-4B). None of the five entries loads NVIDIA checkpoints, GGUFs, `131K`/`132K` tokenizers, or `pca_tree.pt`, nor simulates hardware FP8/FP4 quantization rounding.
- **Dory1.0-4B Paper §6.2 Contract A Disclosure (`DR23`):** Live self-speculation (`104` Dual-Coda banks, `k = 1 → 4, γ = 4`) is reconstructed from Paper §6.2 (absent from upstream runtimes) and explicitly badged in View 6, the timeline dock, and the tour; production 4-checkpoint sizes and TBPTL depth-1 diagrams are strictly read-only (`zeroFakeQuantToggles = true`, `hasGradients = false`).
- **Diffusion Projection-Width Reversal (`Mini` vs. `14B`):** Miniature `40L` Diffusion expands `8 → 256` (`qConcat = 256`, `kvConcat = 64`), whereas canonical `14B` contracts `5,120 → 4,096` (`kv = 1,024`); explicitly disclosed in `gqa_rope`, `macro`, and `Architecture` readouts.
- **Excluded Peak Inference Workspace:** Accounted production weight & KV totals explicitly exclude intermediate layer activations, `[B × 131,072]` logits, CUDA/framework scratchpads, and allocator fragmentation.
- **VLM-8B Quadratic SS Reference Only (`D10`, `D24`):** Quadratic Self-Speculation is absent from all 6 text checkpoints and shown strictly as a badged read-only reference tab inside the Diffusion Mask Explorer.
- **PyTorch Fixture Verification vs. Regeneration:** Checked-in SHA-256-verified PyTorch `2.15.0` fixtures are executed directly in Vitest without requiring local Python/PyTorch during CI or browser runs.
- **Software vs. Hardware GPU Verification:** Headless CI runs Chromium WebGL2 on ANGLE/SwiftShader; hardware GPU visual/cadence inspection remains a targeted pre-release step.
