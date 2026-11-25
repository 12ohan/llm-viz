# Changelog

All notable changes to **Nemotron 3 Nano 30B-A3B — Architecture Visualization** are documented in this file.

> **Release Status:** Finalized **Nemotron 3 Nano `v1.0.0`** milestone release ([Live Demo](https://12ohan.github.io/llm-viz/)). Regression baseline: `236 / 236` Vitest tests (`33` files), `9 / 9` Playwright Chromium WebGL2 tests, full/incremental `0`-ULP parity, independent PyTorch `2.15.0` fixtures, and zero browser runtime dependencies.

## [1.0.0] — 2026-11-24

### Added

- **52-Layer Hybrid Vertical Tower**: Full WebGL2 3D spine derived directly from Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/config.json` (commit `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`, `23 Mamba-2 · 23 MoE 128+1 · 6 GQA Attention`), four NVIDIA Figure 2 motif spans (`01–35`, `36–41`, `42–44`, `45–52`), continuous `2,688`-channel emerald residual stream, and untied Embedding / Output bookends.
- **Unified 52-Layer Float32 Miniature Simulator (`src/math/hybridFlow.ts`)**: Deterministic clean-room Float32 forward pass (`d = 8`, `V_mini = 32`, `ε = 1e-5`) spanning Embedding (NoPE) → all 52 layers → Final RMSNorm → Untied LM Head + Softmax, verified against independent **PyTorch `2.15.0`** reference fixtures with SHA-256 weight proofs (`scripts/reference/`, `tests/fixtures/`).
- **Cached Incremental Autoregressive (AR) Decoding & Branching (`src/math/incrementalDecode.ts`, `src/math/cacheContracts.ts`)**: `Full Seq` vs. `AR Decode` mode toggle, prefill + single-token AR steps with bit-exact (`0` ULP) full-sequence equivalence, immutable copy-on-step `HybridDecodeCacheState` snapshots (`75.00 KiB` default miniature snapshot), rewind/fork continuation branches, and isolated Tour vs. Free Explore sessions (`DualDecodeSessionStore`).
- **Five Exploded 3D Layer Cutaways**: Dedicated 3D thin-sheet cutaways with live `RG32F` value atlases and interactive Math Dock inspection for **Embedding (NoPE)**, **Mamba-2 (Bay 01 & Bay 42 Feeder)**, **Granular MoE 128+1 (Bay 02 & Bay 52)**, **GQA Attention 32Q/2KV (Bay 06 & Bay 43)**, and **Output / LM Head (Top-5 strip)**.
- **Focused-Cell Partial-Sum Accumulation & Non-Spoiling Reveal**: Operand-by-operand cutaway stepper (`~400 ms/term`), running partial sums in Math Dock / floating popover / zoomed tensor cells, and running MoE sigmoid denominator (`Σσ`) without premature normalized weight exposure.
- **22-Beat, 3-Act Guided Walkthrough (`src/story/walkthrough.ts`)**: Non-destructive interactive tour (`Act I · Full-Sequence Prefill`, `Act II · Precision Boundary`, `Act III · Live Cached AR Decode & 1M Scaling`) plus fast (`0.9 s`) and slow instructional (`5.5 s`) 52-layer Tower spine propagation replay with pauseable HUD.
- **Selective FP8 Mode & Interactive `1 → 1M` Memory Scaling Chart**: Live `[ BF16 | FP8 Selective ]` toggle highlighting NVIDIA §4.2–4.3 preservation boundaries (12 BF16 bays + 23 BF16 Conv1D strips + FP8 KV caches) and an inline zero-dependency SVG log-log scaling chart (`3.05 GiB` FP8 / `6.05 GiB` BF16 at 1M tokens vs. `26.0 GiB` / `52.0 GiB` 52-layer GQA).
- **Automated Verification & Release Documentation**: Playwright/Chromium WebGL2 E2E smoke suite (`npm run test:e2e`), separated PR verification CI (`.github/workflows/ci.yml`) and main-only Pages deployment (`.github/workflows/deploy.yml`), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/VERIFICATION.md`](docs/VERIFICATION.md), and in-app attribution credits in `index.html`.

### Changed

- **Cache & Memory Hardening**: Raised default bay weight cache to all 52 bays (`~1.14 MiB`) sharing read-only `Float32Array` buffers across executions/traces; added length-prefixed `prefixTokenLabels` to `stateDigest`, isolated lazy-reconstruction closures, active-branch-safe branch deletion, and conservative session bounds (`maxBranches = 16`, `maxUniqueSteps = 128`, inactive trace pruning).
- **Runtime Rendering & Performance**: Consolidated all 8 cutaway builders onto shared 18-float `LinesPass` guide-line packing (`src/render/linePacking.ts`), gated continuous RAF strictly to active motion/playback, coalesced `pointermove` hover picking, deduplicated `RG32F` atlas uploads via `computeValueAtlasContentKey`, throttled 5.5 s spine-wave rebuilds to discrete bay transitions, and moved 60 fps stepper UI ticks onto `syncStepperProgressOnly()`.
- **Milestone 1 Briefs Archived**: Moved superseded Milestone 1 planning documents into [`docs/archive/DESIGN_BRIEF.md`](docs/archive/DESIGN_BRIEF.md) and [`docs/archive/IMPLEMENTATION_PLAN.md`](docs/archive/IMPLEMENTATION_PLAN.md).

### Fixed

- **Accessibility & Keyboard Parity**: Deferred global `Space`/`Enter` shortcuts when native buttons/controls have focus; added WAI-ARIA `role="tab"` + roving `tabindex` + arrow navigation on `.mode-pills` and `#gqa-step-pills`; added modal focus capture/trap and `Escape` focus restoration for `#shortcuts-popover` and `#memory-chart-drawer`; synchronized all shortcuts (`1–4`, `5`, `6`, `Space`, `Shift+Space`, `←/→`, `↑/↓`, `Enter`, `P`, `M`, `R`, `?`, `Esc`) and pre-hydration `Tour 1/22` copy across `index.html`, `src/main.ts`, and `README.md`.
- **WebGL Context-Loss Recovery & FBO Cleanup**: Prevented leaked FBO/renderbuffer handles on partial allocation failure (`fboAllocFailed` latch), paused all animation clocks on `webglcontextlost`, and reconstructed all 5 passes cleanly on `webglcontextrestored`.
- **Visual & Layout Polish**: Restored visible cutaway guide lines across all 8 builders, eliminated MoE premature weight exposure in Step 3 (`Top-6`), improved floating popover active-cell avoidance, ensured all 5 Output Top-5 candidate pills remain visible and non-overlapping, and tightened narrow (`1024×800`) and mobile (`390×844`) layout containment.
