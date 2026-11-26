# NVIDIA Nemotron 3 Nano & Nemotron-Labs-TwoTower — Interactive 3D Architecture Visualizations

A zero-runtime-dependency WebGL2 multi-page application (MPA) hosting **two isolated interactive 3D architectural explainers and miniature forward-pass simulators** in the visual language of Brendan Bycroft's `llm-viz`:

| Interactive App | Entry Path | GitHub Pages URL | Release Status | Architecture Scope |
| --- | --- | --- | --- | --- |
| **Nemotron 3 Nano 30B-A3B** | `/` (`index.html`) | <https://12ohan.github.io/llm-viz/> | **`v1.0.0`** (preserved & live) | Single-tower 52-layer causal hybrid (`23 Mamba-2 · 23 MoE 128+1 · 6 GQA`), live cached AR decode & branching, 5 cutaways, 22-beat tour, Selective FP8 & `1 → 1M` chart. |
| **Nemotron-Labs-TwoTower 30B-A3B** | `/twotower/` (`twotower/index.html`) | <https://12ohan.github.io/llm-viz/twotower/> | **`twotower-v1.0.0`** (released & live) | Dual-tower `2 × 52`-layer Context + Block-Diffusion Denoiser (`63.19B` / `126.37 GB` SI / `117.69 GiB` IEC), 7 spatial views, 3 execution modes, persistent nested timeline, MDLM sampler, 4-act / 28-beat tour. |

Both apps compile as an isolated **Topology A multi-entry Vite MPA** enforced by an automated 7-gate bundle-isolation check (`npm run verify:isolation`) so TwoTower code never leaks into Nano's `/` bundle, connected by **quiet, base-path-aware reciprocal navigation** (`TwoTower ↗` in Nano's top bar and `Nano ↗` in TwoTower's top bar) across both `/` ↔ `/twotower/` and `/llm-viz/` ↔ `/llm-viz/twotower/`.

> [!IMPORTANT]
> **3-Tier Epistemic Sourcing Disclosure (Applies to Both Applications)**
>
> 1. **Directly Inspected Public Metadata:** Production layer topologies, tensor dimensions, precision policies, and byte totals come strictly from pinned public sources — **Nano:** [`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848) & Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (commit `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`); **TwoTower:** [`arXiv:2606.26493v2`](https://arxiv.org/abs/2606.26493) & Hugging Face `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16` (revision `35b6498508be8856d380cdfe4a900a79b918d1ea`), cross-checked via a read-only offline 24-shard safetensors header metadata audit (`0` integrity errors; **zero weight values stored or redistributed**).
> 2. **Analytically Derived Quantities:** Exact parameter inventories (`63,185,902,848` params / `126,371,817,472 B`, `0 B` discrepancy vs. `model.safetensors.index.json`), dual SI (`GB`/`MB`) & IEC (`GiB`/`MiB`) byte conversions, 3-regime context-length scaling curves (`≤ 128K` operational · `128K–256K` configured positional limit · `> 256K` analytical extrapolation), and 4-way memory boundaries (explicitly excluding peak runtime inference workspace) are computed analytically from pinned config shapes.
> 3. **Synthetic Miniature Browser Execution:** All live in-browser forward passes run deterministic clean-room miniature Float32/Float64 simulators (**Nano:** `d = 8`, `V_mini = 32`, `299,192` floats; **TwoTower:** `d = 8`, `V_mini = 34`, `600,120` floats, seeded synthetic weights). Neither app loads NVIDIA production checkpoints, runs the `131,072`-token Nemotron tokenizer, or redistributes NVIDIA model weights.

---

## 1. Nemotron-Labs-TwoTower 30B-A3B (`/twotower/`)

Interactive WebGL2 explorer for NVIDIA's dual-tower **Autoregressive Context + Block-Diffusion Denoiser** hybrid (`NemotronHTwoTowerForCausalLM`), verified against independent **PyTorch `2.15.0`** reference fixtures across all 8 verification categories:

- **Seven Interactive WebGL2 Spatial Views**:
  1. **`Macro`** — Side-by-side `52 × 2` dual-tower architecture showing all `29` aligned read-only Context → Denoiser state transfer linkers (`23` Mamba-2 + `6` GQA) vs. `23` stateless MoE bays (`0 B` transfer), 10-axis bay picking, and pauseable instructional Macro Flow Replay (`Instructional replay ≠ measured compute`).
  2. **`GQA`** — Asymmetric 35/65 Paired GQA cutaway across the `6` attention bays (`06, 13, 20, 27, 34, 43`), packing all `32` miniature Q heads and `2` KV heads into a `128 × 128` `RG32F` atlas with inherited read-only Context `KV[0..P-1]` and bidirectional `[B × (P+B)]` intra-block attention.
  3. **`Mamba`** — Asymmetric 35/65 Paired Mamba-2 cutaway across the `23` Mamba bays, packing all `16` miniature heads and `8` groups into a `256 × 128` `RG32F` atlas alongside 4-tap Conv1D boundary replacement (`3→2→1→0`) and closed-form `4 × 4` per-head analytical SSM state decomposition (`H_p = A_cum·H_init + ΔH_write + ε_f32`).
  4. **`Timestep`** — Canonical masked-fraction timestep conditioning (`t = nMasked / B`, never `r / K`), COS→SIN sinusoidal embedding → `t_embedder` → `t_block` → `52 scale_shift_tables`, and exact split PixArt-α `adaLN-single` ordering (`modulate → RMSNorm` on `M`/`*` vs. `RMSNorm → modulate` on `E`, unmodulated final `norm_f`).
  5. **`MoE`** — Stateless `128+1` Granular MoE cutaway (`0 B` sequence cache) highlighting FP64 router 6th-vs-7th cutoff margins, FP32 `e_score_correction_bias`, and non-spoiling partial `Σσ` denominator accumulation.
  6. **`Embed`** — Independent untied NoPE token embeddings (`W_embed_ctx` vs. `W_embed_den`, `V = 34` with `[MASK]` `#32` and `<eos>` `#33`; production repurposed `[INST]` ID `3`).
  7. **`Output`** — Unmodulated final RMSNorm + untied Denoiser LM head showing unshifted block predictions and raw LM-head vs. constrained MDLM/SUBS distributions (`maskTokenProb = 0`, zero-`Infinity` presentation).
- **Three Execution Modes (`Mask-Diffusion`, `AR`, `Mock-AR`)**:
  - **`Mask-Diffusion`**: Full 4-step block lifecycle (`Init Block` → `Refine` → `Commit Conf` → `Completed (Uncommitted)` → `Commit → Context`) with immutable Context boundaries (`0`-ULP invariance across refinements) and explicit historical boundary mismatch banners whenever inspecting a prior Denoiser pass after Context advances.
  - **`AR`**: Context-only single-token causal decode with Denoiser and timestep conditioning bypassed (`t = null`).
  - **`Mock-AR`**: Denoiser causal `S = 1` re-evaluation seeded from Context `KV[:-1]` + `mamba_s2` with `adaLN-single` bypassed (`0`-ULP parity with Context AR when `ΔW = 0`).
- **Persistent Nested Timeline & Branching**: Bottom timeline dock for scrubbing any historical milestone non-destructively, auto-forking branches from rewound cursors, active-branch-safe branch deletion, and retained memory tracking.
- **Advanced MDLM Sampler Inspector (`Sampler` Tab)**: Confidence threshold & temperature controls (browser miniature educational default `γ = 0.22, T = 0.0`, published paper operating point `γ = 0.80`, model-card usage example `γ = 0.80, T = 0.1`, reference code default `γ = 0.90, T = 0.0`), decoupled Block Width (`B ∈ {4, 8}`) and Refinement Budget (`K ∈ [1..12]`), threshold vs. remaining-budget floor (`ceil(nMasked / remainingSteps)`) unmasking reasons, deterministic Gumbel-max (`T > 0` with explicit seed), explicit non-retroactive config scope badges, and unscaled calibration caveats.
- **4-Act / 28-Beat Guided Walkthrough (`T`)**: Strictly isolated educational tour (`Act I` Architecture & Modes · `Act II` Layers & Mechanics · `Act III` Output & Sampler · `Act IV` Commit & Production) that restores your Free Explore session untouched upon exit.
- **Production Architecture & 3-Regime Context Scaling (`Architecture` Tab)**: Dual `GB`/`GiB` checkpoint reconciliation (`63.19B` · `126.37 GB` / `117.69 GiB`), 4-way memory separation (explicitly excluding peak runtime inference workspace), 3-regime context scaling (`≤ 128K` Model-Card Operational · `128K–256K` Configured Positional Limit · `> 256K` Analytical Extrapolation), 9-entry discrepancy ledger, and collapsible Developer Diagnostic Workbench (`W`).

---

## 2. Nemotron 3 Nano 30B-A3B (`/` · `v1.0.0` Preserved)

The finalized single-tower **`v1.0.0`** visualizer remains completely intact and bundle-isolated at `/`:

- **52-Layer Hybrid Vertical Spine** (`23 Mamba-2 · 23 MoE 128+1 · 6 GQA`, 4 NVIDIA Figure 2 motif spans, continuous `2,688`-channel emerald residual stream, untied `131,072`-vocab bookends).
- **Unified 52-Layer Float32 Miniature Simulator (`d = 8, V_mini = 32`)** verified against independent PyTorch `2.15.0` fixtures (`SHA-256` weight proof over `299,192` floats).
- **Cached Incremental AR Decoding & Copy-on-Step Branching** (`Full Seq` vs. `AR Decode`, bit-exact `0`-ULP equivalence, `75.00 KiB` miniature snapshot).
- **Five Exploded 3D Cutaways** (**Embedding NoPE**, **Mamba-2**, **Granular MoE 128+1**, **GQA 32Q/2KV**, **Output / LM Head**), focused-cell partial-sum accumulation, fast (`0.9 s`) & slow (`5.5 s`) spine propagation replay, **22-Beat / 3-Act Guided Walkthrough**, and **Selective FP8 Mode + `1 → 1M` Memory Scaling Chart**.

---

## Miniature Simulator vs. Production Metadata Summary

| Dimension / Property | Nano Mini (`src/math/`) | Nano Production (`v1.0.0`) | TwoTower Mini (`src/twotower/`) | TwoTower Production (`rev 35b64985`) |
| --- | --- | --- | --- | --- |
| **Towers & Pattern** | `1 × 52L` (`23 M · 23 E · 6 *`) | `1 × 52L` (`23 M · 23 E · 6 *`) | `2 × 52L` independent towers | `2 × 52L` independent (`crossTowerWeightSharing: false`) |
| **Weights & Payload** | `299,192` Float32 (`~1.14 MiB`) | `31.6B` total (`3.2B`/`3.6B` active) | `600,120` Float32 (`2.29 MiB`, `860` tensors) | `63,185,902,848` (`63.19B`) · `126.37 GB` SI / `117.69 GiB` IEC (`12,544` tensors) |
| **Hidden Width (`d`)** | `8` | `2,688` | `8` | `2,688` |
| **Vocabulary (`V`, untied)** | `32` (`V_mini`) | `131,072` | `34` (`0..31` + `[MASK]` `#32` + `<eos>` `#33`) | `131,072` (`MASK = 3` `[INST]`, `EOS = [2, 11]`) |
| **Timestep Conditioning** | None (AR only) | None (AR only) | `freqDim = 16`, `58` tensors | `freqDim = 256`, `58` tensors (`30,022,272` params · `60.04 MB` / `57.26 MiB`) |
| **Mamba-2 / GQA / MoE** | `16`H/`8`G/`4×4` · `32Q/2KV` (`d=4`) · `128+1` | `64`H/`8`G/`64×128` (`47.08 MiB`) · `32Q/2KV` (`+6/+3 KiB/tok`) · `128+1` | Same geometry; `kvCapacity = 24`; bidirectional `[B×(P+B)]` GQA; `FP64` MoE router | `47.08 MiB` Context Mamba + `+6 KiB/tok` Context KV + `B×6 KiB` transient block KV; `FP64` MoE router + `46` `FP32` biases |
| **Context Regimes** | `maxTokens = 16` | Up to `1M` analytical scaling | `P = 8`, `B ∈ {4, 8}`, `K ∈ [1..12]` | `≤ 128K` card operational · `128K–256K` config positional limit · `> 256K` extrapolated |

---

## Controls & Keyboard Shortcuts

Pointer controls on both 3D stages: **Left-drag** to orbit · **Right-drag / Shift-drag** to pan · **Scroll / Pinch** to zoom · **Click / Tap** any bay, slab, or matrix cell to inspect. Global `Space`/`Enter` automatically defer to native activation when interactive controls have focus.

- **TwoTower (`/twotower/`):** `T` (launch/exit 4-Act Guided Tour) · `←` / `→` (step Tour beats, or navigate focused view/sidebar tablists with `Home`/`End`) · `Space` (play/pause Tour autoplay) · `R` (fit/reset camera to current view) · `W` (toggle Developer Diagnostic Workbench) · `Esc` (close Workbench, exit Tour, or close mobile Inspector drawer). See [`docs/TWOTOWER_USER_GUIDE.md`](docs/TWOTOWER_USER_GUIDE.md) for full UI controls.
- **Nano (`/`):** `1`–`6` (Tower, Mamba-2, MoE, GQA, Embedding, Output) · `Space` / `Shift+Space` (Tour beat or cutaway stepper) · `←`/`→`/`↑`/`↓` (step beats/bays) · `Enter` (zoom bay) · `P` (`BF16`/`FP8 Selective`) · `M` (`1 → 1M` Memory Scaling chart) · `R` (reset view) · `?` (Shortcuts & Attribution) · `Esc` (close/back).

---

## Local Development, Build & Verification

**Supported Node.js versions:** `Node.js 22.12+ (LTS)`, `24.x (LTS)`, or `26+` (`^22.12.0 || ^24.0.0 || >=26.0.0`, matching `vitest@5.1.1`; CI uses Node `22`).

```bash
# 1. Install exact lockfile dependencies (0 runtime dependencies)
npm ci

# 2. Start Vite dev server (Nano at http://localhost:3000/, TwoTower at http://localhost:3000/twotower/)
npm run dev

# 3. Strict TypeScript check
npm run typecheck

# 4. Run full Vitest unit, contract & PyTorch-parity suite (zero Python required)
npm test

# 5. Production build for root path (`/` and `/twotower/`) + 7-gate Nano isolation check
npm run build
npm run verify:isolation

# 6. Production build for GitHub Pages subpath (`/llm-viz/` and `/llm-viz/twotower/`)
VITE_BASE_PATH=/llm-viz/ npm run build
npm run verify:isolation

# 7. Run Playwright Chromium WebGL2 E2E smoke suite across both apps & both base paths
#    (first time: npx playwright install chromium)
npm run test:e2e
```

**Measured Production Bundle Sizes (`npm run verify:isolation`, enforced on both `/` and `/llm-viz/`):**

- **Nano transitive JS+CSS:** **`207.60 KiB`** (`212,585 B`) gzip (`764,974 B` raw) vs. `215.00 KiB` (`220,160 B`) hard budget (`7.40 KiB` headroom, `0` sourcemap/marker leaks, `+77 B` gzip CSS / `0 B` Nano JS for the quiet `TwoTower ↗` link).
- **Shared `assets/overlay-*.js`:** **`127.27 KiB`** (`130,321 B`) gzip (`442,996 B` raw, `0` TwoTower sources).
- **TwoTower transitive JS+CSS:** **`317.74 KiB`** (`325,363 B` root / `325,370 B` `/llm-viz/`) gzip (`1,149,477 B` / `1,149,485 B` raw, `22` confirmed `src/twotower/*` modules, `0` runtime dependencies).

---

## Documentation

- [**TwoTower User Guide (`docs/TWOTOWER_USER_GUIDE.md`)**](docs/TWOTOWER_USER_GUIDE.md) — Practical walkthrough of all 7 spatial views, 3 execution modes, 4-step block diffusion lifecycle, `t = nMasked / B` & historical mismatch banners, Persistent Nested Timeline, Advanced MDLM Sampler, 4-Act Guided Tour, 3-regime scaling, Workbench (`W`), and shortcuts.
- [**TwoTower Architecture & Checkpoint Audit (`docs/TWOTOWER_ARCHITECTURE.md`)**](docs/TWOTOWER_ARCHITECTURE.md) — Pinned provenance (`35b64985`), 3-tier epistemic sourcing matrix, dual `GB`/`GiB` reconciliation, 3-regime context scaling, 4-way memory boundary, 9-entry discrepancy ledger, and Topology A isolation contract.
- [**Nano Architecture Guide (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md) — Single-tower Nano module boundaries, 5-pass WebGL2 pipeline, context-loss recovery, runtime performance architecture, immutable AR cache contracts, and shared Topology A seams.
- [**Verification & Reproducibility Runbook (`docs/VERIFICATION.md`)**](docs/VERIFICATION.md) — Complete Vitest (`369` tests / `46` files), independent PyTorch `2.15.0` parity (Nano + 8 TwoTower categories), Topology A isolation gate (`207.60 KiB` Nano / `127.27 KiB` shared / `317.74 KiB` TwoTower gzip), Playwright Chromium WebGL2 E2E (`21` tests, SwiftShader & hardware Metal), CI/Pages workflows, and static tarball checksum steps.
- [**TwoTower v1.0.0 Release Notes (`docs/RELEASE_NOTES_TWOTOWER_v1.0.0.md`)**](docs/RELEASE_NOTES_TWOTOWER_v1.0.0.md) — Release summary, live public URLs, verification snapshot, discrepancy ledger, and known limitations for `twotower-v1.0.0`.
- [**Third-Party Notices (`THIRD_PARTY_NOTICES.md`)**](THIRD_PARTY_NOTICES.md) — Verified upstream licenses, architectural citations, non-redistribution statement, and exact dev-only dependency inventory.
- [**Changelog (`CHANGELOG.md`)**](CHANGELOG.md) — Release history for Nano `v1.0.0` and TwoTower `twotower-v1.0.0`.
- **Historical Milestone 1 Briefs (superseded):** [`docs/archive/DESIGN_BRIEF.md`](docs/archive/DESIGN_BRIEF.md) · [`docs/archive/IMPLEMENTATION_PLAN.md`](docs/archive/IMPLEMENTATION_PLAN.md).

---

## Attribution & License

1. **Repository Heritage & Visual Language (`bbycroft/llm-viz`, MIT License):** Forks Brendan Bycroft's [`bbycroft/llm-viz`](https://github.com/bbycroft/llm-viz) (`Copyright (c) 2023-2026 Brendan Bycroft`, preserved verbatim in [`LICENSE`](LICENSE)) and builds on its interactive architectural idiom. All Nano and TwoTower browser modules (`src/`), tests (`tests/`), isolation checks (`scripts/verify_bundle_isolation.mjs`), and offline parity scripts (`scripts/reference/`) were newly authored (`Copyright (c) 2026 Rohan Master`).
2. **Referenced Architectural Specifications (Not Redistributed Artifacts):**
   - **Nemotron-Labs-TwoTower 30B-A3B:** [`arXiv:2606.26493v2`](https://arxiv.org/abs/2606.26493) and Hugging Face [`nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16/tree/35b6498508be8856d380cdfe4a900a79b918d1ea) pinned at revision `35b6498508be8856d380cdfe4a900a79b918d1ea` (clean-room equations referencing Apache-2.0 `modeling_nemotron_twotower.py`, PixArt-α `adaLN-single`, and MDLM/SUBS).
   - **Nemotron 3 Nano 30B-A3B:** [`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848) and Hugging Face [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/tree/bf77c3174f68ad409e1c2aa60daeb46e32d1c606) pinned at commit `bf77c3174f68ad409e1c2aa60daeb46e32d1c606` (clean-room equations referencing Apache-2.0 `modeling_nemotron_h.py`).
   - **No NVIDIA model weights, checkpoints, safetensors, tokenizers, or NVIDIA Open Model License artifacts are redistributed in this repository or in `dist/`.** NVIDIA and Nemotron are trademarks of NVIDIA Corporation; this project is an independent educational visualization not affiliated with, sponsored by, or endorsed by NVIDIA Corporation.
3. **Zero Browser Runtime Dependencies:** Offline parity generation uses pinned PyTorch `2.15.0` (BSD-3-Clause, `scripts/reference/`, absent from `dist/`), and build/test tooling uses 5 exact-pinned npm `devDependencies` (`@playwright/test@1.65.1`, `@types/node@22.20.5`, `typescript@5.9.3`, `vite@8.3.8`, `vitest@5.1.1`).

Licensed under the **MIT License** — see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
