# NVIDIA Nemotron 3 Nano 30B-A3B — Interactive 3D Architecture Visualization

An interactive WebGL2 3D architectural explainer and live miniature forward-pass simulator for **NVIDIA Nemotron 3 Nano 30B-A3B**, built with zero browser runtime dependencies in the visual language of Brendan Bycroft's `llm-viz`.

**Live demo:** <https://12ohan.github.io/llm-viz/>

> [!IMPORTANT]
> **Epistemic Boundary — Miniature Float32 Simulator vs. Production Metadata**
>
> This browser application runs a **deterministic miniature Float32 simulator** (`d = 8`, `V_mini = 32`, seeded synthetic weights, educational token labels such as `"How does Nemotron use …"`) executing NVIDIA's exact `NemotronH` architectural equations across all 52 layers. It does **not** run 31.6B checkpoint inference or real Nemotron tokenizer logits, and **no NVIDIA model weights, checkpoints, or tokenizers are redistributed in this repository or built bundle**.
>
> All production scale readouts (`hidden = 2,688`, `vocab = 131,072`, `31.6B` total / `3.2B` active excluding embeddings / `3.6B` including embeddings, `47.1 MiB` fixed Mamba-2 state, `+6 KiB/tok` BF16 or `+3 KiB/tok` FP8 KV growth, and Selective FP8 layer boundaries) reflect **pinned production architectural metadata** from [`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848) and Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/config.json` (commit `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`).

---

## What This Visualization Demonstrates

- **52-Layer Hybrid Vertical Spine** — All 52 layers derived directly from `hybrid_override_pattern` (`23 Mamba-2 · 23 MoE 128+1 · 6 GQA Attention`) partitioned into NVIDIA Figure 2's four motif spans (`01–35 [M·E·M·E·M·*·E] ×5`, `36–41 [M·E] ×3`, `42–44 [M·*·E] ×1`, `45–52 [M·E] ×4`), bridged by the continuous `2,688`-channel emerald residual stream between untied `131,072`-vocab embeddings and the Final RMSNorm + untied LM head.
- **Unified 52-Layer Float32 Miniature Forward Pass** — Real end-to-end miniature execution (`Embedding → 52 hybrid bays → Final RMSNorm → Untied LM Head + Softmax`) where every cutaway inspects live activations from the same unified forward trajectory, verified against independent **PyTorch `2.15.0`** reference fixtures (cryptographic SHA-256 weight proof over all `299,192` Float32 parameters).
- **Cached Incremental Autoregressive (AR) Decoding & Branching** — Switch between `Full Seq` and `AR Decode` to prefill a prompt, append tokens one step at a time with bit-exact (`0` ULP) parity against full-sequence execution, rewind to any historical step, and fork immutable copy-on-step continuation branches (`branch-0`, `branch-1`, …) while observing growing `[1 × (P+1)]` GQA KV caches beside fixed-size Mamba-2 states and `0 B` MoE caches.
- **Five Exploded 3D Layer Cutaways**:
  1. **Embedding (NoPE)** — Untied token row lookup (`W_embed`) with an explicit NoPE callout showing zero positional embeddings added.
  2. **Mamba-2 (Bay 01 & Bay 42 Feeder)** — 4-tap depthwise Causal Conv1D (always BF16) + persistent FP32 recurrent SSM state (`64 × 64 × 128` production / `16 × 4 × 4` mini) updating in place (`47.1 MiB` total across all 23 bays) + gate-first 8-group `MambaRMSNormGated` (`norm_before_gate = False`).
  3. **Granular MoE 128+1 (Bay 02 & Bay 52)** — FP32 sigmoid router with `e_score_correction_bias` for Top-6 selection, dropping bias for unbiased Top-6 `×2.5` normalization across 6 of 128 non-gated `ReLU²` routed experts (`2,688 → 1,856 → 2,688`) alongside 1 always-active double-width shared expert (`3,712 = 2 × 1,856`), `0 B` sequence cache.
  4. **GQA Attention (Bay 06 & Bay 43)** — `32` Query heads sharing `2` KV heads (`16:1`, NoPE) with live causal `QKᵀ/√d` scores, row softmax, weighted `V` reads, and growing K/V token stacks (`+6 KiB/tok` BF16 / `+3 KiB/tok` FP8 across all 6 bays).
  5. **Output / LM Head** — Final RMSNorm (`ε = 1e-5`), untied LM head projection, and Top-5 candidate probability strip.
- **22-Beat, 3-Act Guided Walkthrough** — Non-destructive interactive tour (`Tour 1/22` through `Act I · Full-Sequence Prefill`, `Act II · Precision Boundary`, and `Act III · Live Cached AR Decode & 1M Scaling`) with isolated Tour vs. Free Explore AR decode sessions while keeping free 3D orbit/pan/zoom unlocked.
- **Focused-Cell Partial-Sum Accumulation & Non-Spoiling Reveal** — Cutaway Math Stepper (`▶ Play` or scrub) steps operand-by-operand (`~400 ms/term`) across GQA dot products/weighted `V`, Mamba-2 Conv1D/SSM/readout, and MoE Top-6 running sigmoid denominator (`Σσ`) and 6-branch sum, showing live partial sums in the Math Dock, floating popover, and zoomed tensor cells without prematurely spoiling uncomputed values.
- **Fast (`0.9 s`) & Slow Instructional (`5.5 s`) Spine Propagation Replay** — Visualize a token traveling through all 52 layers on the Tower with a live pauseable stage HUD (auto-settling cleanly under `prefers-reduced-motion`).
- **Selective FP8 Mode & Interactive `1 → 1M` Memory Scaling Chart** — Toggle `[ BF16 | FP8 Selective ]` to highlight the 12 BF16-preserved sensitive bays (6 GQA + 6 preceding Mamba feeders `05, 12, 19, 26, 33, 42`) and 23 BF16 Conv1D strips alongside frosted FP8 KV caches, and open the inline SVG log-log scaling chart comparing Nemotron 3 Nano (`3.05 GiB` FP8 / `6.05 GiB` BF16 at 1M tokens) against a 52-layer pure GQA Transformer (`26.0 GiB` / `52.0 GiB` — `~8.5×` larger), with exact crossovers at `1,048 / 2,096` and `8,035 / 16,070` tokens.

---

## Miniature Simulator vs. Production Metadata Summary

| Dimension / Property | Interactive Miniature Simulator (`src/math/`) | Pinned Production Architecture (`src/model/spec.ts`) |
| --- | --- | --- |
| Weights & Precision | Deterministic synthetic Float32 (`299,192` floats, `~1.14 MiB`) | `31.6B` total · `3.2B` active excl. embeddings (`3.6B` incl.) |
| Hidden Width (`d`) | `8` | `2,688` |
| Vocabulary (`V`, untied) | `32` educational tokens (`V_mini`) | `131,072` (`tie_word_embeddings: false`) |
| Layer Topology | Exact 52-layer `hybrid_override_pattern` (`23 M · 23 E · 6 *`) | Exact 52-layer `hybrid_override_pattern` (`23 M · 23 E · 6 *`) |
| Mamba-2 Geometry | `16` heads × `4` ch × `4` state (`8` groups, `k=4`, `69.00 KiB` total) | `64` heads × `64` ch × `128` state (`8` groups, `k=4`, `47.1 MiB` total) |
| Granular MoE (`128+1`) | `128` routed (Top-6, `d_exp=4`, `ReLU²`) + `1` shared (`d_shared=8`) | `128` routed (Top-6, `1,856`, `ReLU²`) + `1` shared (`3,712`) |
| GQA Attention (NoPE) | `32` Q heads / `2` KV heads (`d_head=4`, `6.00 KiB` alloc @ `T=16`) | `32` Q heads / `2` KV heads (`d_head=128`, `+6 / +3 KiB/tok`) |
| BF16 / FP8 Selective Toggle | Visual cue + metadata tag (simulator stays Float32) | NVIDIA §4.2–4.3 precision boundary & KV byte accounting |

---

## Keyboard & Pointer Controls

Pointer controls on the 3D stage: **Left-drag** to orbit · **Right-drag / Shift-drag** to pan · **Scroll / Pinch** to zoom · **Click / Tap** any bay, slab, or matrix cell to inspect.

Global keyboard shortcuts (synchronized with `src/main.ts` and `index.html`; `Space` and `Enter` automatically defer to native activation when a button or control is focused):

| Key | Action |
| --- | --- |
| `1` – `4` | Switch view: `1` 52-Layer Tower · `2` Mamba-2 · `3` MoE 128+1 · `4` GQA |
| `5` | Switch view: Cutaway — Embedding (NoPE) |
| `6` | Switch view: Cutaway — Output / LM Head |
| `Space` | Advance beat (Guided Tour) or Play / pause cutaway math stepper |
| `Shift` + `Space` | Previous beat (Guided Tour) |
| `←` / `→` | Step beat (Guided Tour) or step bay (Tower); move focus inside active tablists |
| `↑` / `↓` | Step bay (Tower, when Guided Tour is inactive) |
| `Enter` | Zoom selected bay (Tower, when Guided Tour is inactive) |
| `P` | Toggle `BF16` / `FP8 Selective` precision metadata mode |
| `M` | Toggle interactive `1 → 1M` Memory Scaling chart dialog |
| `R` | Reset camera view, `T = 8`, and `BF16` |
| `?` | Toggle Keyboard Shortcuts & Attribution dialog |
| `Esc` | Close open popover/dialog, clear pinned cell / stepper / spine wave, clear selection, or exit cutaway back to Tower |

---

## Local Development

**Supported Node.js versions:** `Node.js 22.12+ (LTS)`, `24.x (LTS)`, or `26+` (`^22.12.0 || ^24.0.0 || >=26.0.0`, matching `vitest@5.1.1`; CI uses Node `22`).

```bash
# 1. Install exact lockfile dependencies
npm ci

# 2. Start Vite dev server on http://localhost:3000
npm run dev

# 3. Type-check (strict TypeScript)
npm run typecheck

# 4. Run Vitest unit & PyTorch-parity test suite (zero Python required)
npm test

# 5. Run Playwright Chromium WebGL2 E2E smoke suite (first time: npx playwright install chromium)
npm run test:e2e

# 6. Production build (root path `/`)
npm run build

# 7. Production build for GitHub Pages subpath (`/llm-viz/`)
VITE_BASE_PATH=/llm-viz/ npm run build
```

---

## Documentation

- [**Architecture Guide (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md) — Module boundaries, 5-pass WebGL2 pipeline, context-loss recovery, runtime performance architecture, immutable AR cache contracts, and future TwoTower extension points.
- [**Verification Runbook (`docs/VERIFICATION.md`)**](docs/VERIFICATION.md) — Vitest & PyTorch `2.15.0` parity tolerances, SHA-256 fixture regeneration workflow, Playwright WebGL2 E2E coverage, CI vs. Pages deploy workflows, and known limitations.
- [**Third-Party Notices (`THIRD_PARTY_NOTICES.md`)**](THIRD_PARTY_NOTICES.md) — Verified upstream licenses, architectural attributions, non-redistribution statement, and dev-only dependency inventory.
- [**Changelog (`CHANGELOG.md`)**](CHANGELOG.md) — Nemotron 3 Nano `v1.0.0` release notes.
- **Historical Milestone 1 Briefs (superseded):** [`docs/archive/DESIGN_BRIEF.md`](docs/archive/DESIGN_BRIEF.md) · [`docs/archive/IMPLEMENTATION_PLAN.md`](docs/archive/IMPLEMENTATION_PLAN.md).

---

## Attribution & License

1. **Repository Heritage & Visual Language (`bbycroft/llm-viz`, MIT License):** This repository forks Brendan Bycroft's [`bbycroft/llm-viz`](https://github.com/bbycroft/llm-viz) (`Copyright (c) 2023-2026 Brendan Bycroft`, preserved verbatim in [`LICENSE`](LICENSE)) and builds upon its visual/pedagogical idiom. All Nemotron 3 Nano browser application modules (`src/`), test suites (`tests/`), and reference scripts (`scripts/reference/`) were newly authored (`Copyright (c) 2026 Rohan Master`).
2. **Architectural Specifications (Not Redistributed Artifacts):** Architecture topology, memory equations, and clean-room forward equations follow NVIDIA's Nemotron 3 Nano 30B-A3B Technical Report ([`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848)), Hugging Face `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/config.json` pinned at commit `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`, and reference equations from `modeling_nemotron_h.py` (Apache License 2.0). **No NVIDIA model weights, checkpoints, tokenizers, or NVIDIA Open Model License artifacts are redistributed in this repository or in `dist/`.**
3. **Dev-Only Tooling:** Offline parity fixture generation uses pinned PyTorch `2.15.0` (BSD-3-Clause, `scripts/reference/`, absent from `dist/`), and build/test tooling uses npm `devDependencies` (`vite`, `vitest`, `typescript`, `@types/node`, `@playwright/test`), with **0 third-party runtime dependencies** bundled in `dist/`.

Licensed under the **MIT License** — see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
