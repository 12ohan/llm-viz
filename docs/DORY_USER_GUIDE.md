# NVIDIA Nemotron-Labs-Dory1.0-4B — User Guide

This guide explains how to explore the **NVIDIA Nemotron-Labs-Dory1.0-4B** interactive 3D WebGL2 drafting-table visualizer (`/dory/` locally, live at <https://12ohan.github.io/llm-viz/dory/>), switch elastic loop depths `k ∈ {1..4}` (plus optional base-model extrapolation `{6, 8}`), compare `per_loop` vs. `last_loop` KV policies, inspect reconstructed Paper §6.2 Contract A self-speculative decoding, and step through the isolated 4-Act / 14-Beat Guided Tour.

> [!NOTE]
> **Synthetic Miniature Simulator vs. Canonical 3.975B Production Metadata**
> The live 3D stage executes Dory1.0-4B's exact **72-sublayer looped Full-nGPT** equations (`16` Input Prelude + `40` Shared Recurrent Core + `16` Output Coda, `507` tensors, all `5` scaling families with active `σ_0 = 0.0177`, channel-wise NLERP, 3:1 SWA/Full Gated GQA with Dual RoPE, Normalized SwiGLU, `per_loop`/`last_loop` KV, and Contract A self-speculation) on deterministic synthetic miniature weights (`d = 8`, `d_ff = 24`, `8Q / 2KV`, `d_h = 8`, `q_dim = 64` [`8.00×` vs. prod `1.60×`], `kv_dim = 16` [`2.00×` vs. prod `0.40×`], `V = 64`, `S_max = 32`, `W = 8`, `cap_swa = 16`, `θ_S = 16 / θ_F = 1,024`). Production `0 B`-diff 4-checkpoint sizes (`BF16`, `MXFP8`, `NVFP4`, `Q5_K_M`) and `1 → 256K` KV scaling curves are analytical readouts pinned from `nvidia/nemotron-labs-dory` and `arXiv:2611.03998`.

---

## 1. Workspace Layout

The light `#F7F8FA` editorial drafting-table interface is organized into six coordinated regions:

1. **Top Navigation Bar:** Dory1.0-4B title, the **7 Editorial Spatial View tabs** (`1 Macro 72L` through `7 Bookends`), **`Guided Tour (14)`**, **`Reset View`**, **`Hide/Show Charts`**, **`Hide/Show Inspector`**, **`?`** (Keyboard Shortcuts modal), and the compact **5-Model Top-Bar Switcher** (`Models (5) ▾`) connecting Nano, TwoTower, Diffusion-14B, NLD-Uni, and Dory1.0-4B.
2. **Stage Title & Provenance Banner Strip:** Active view title, subtitle, provenance chips, plus explicit banners on View 6 (**`Reconstructed from Paper §6.2 (Contract A) · Not in Upstream Runtimes · per_loop Only`**) and View 7 (**`READ-ONLY · NO GRADIENTS · ZERO FAKE QUANT TOGGLES`**).
3. **Interactive 3D WebGL2 Stage & Pooled Overlay:** 5-pass WebGL2 rendering with strict idle-RAF sleep (`rafActive === false` at rest), content-addressed `RG32F` atlas deduplication, pooled 3D→2D callouts (`<= 64` on-cell numeric nodes), and floating contextual cell popovers on hover.
4. **2D Analytical `<svg>` Charts Inset (`C`):** Context-sensitive diagrams docked above the timeline on Views 2, 3, 5, 6, and 7 (Exact 2D `span(a, b)` NLERP, Exploratory 2D PCA trajectory, 4-plane Dual-RoPE phasors, `1 → 256K` BF16 KV scaling, Contract A `(q_j, p_j)` rejection bars, and `READ-ONLY` TBPTL depth-1 schematic).
5. **Bottom Elastic-Depth, KV-Policy & Trajectory Dock:**
   - **Row 1:** Elastic loop budget `k = 1..4` (plus `+6,8` extrapolation checkbox), KV cache policy (`per_loop` vs. `last_loop`), execution mode (`Standard` vs. `Self-Spec §6.2`), sampling temperature slider (`T = 0.00` greedy to `1.50` stochastic), and deterministic prompt preset selector (`Reasoning 12`, `Tool Call 10`, `Long 16`, `Short 6`).
   - **Row 2:** Transport controls (`◀`, `▶ Play`, `▶`, trajectory scrubber, `+ Step` / `+ Spec Cycle (γ=4)`), immutable branch switcher (`Fork`, `Reset`), live executed-depth & KV-bank telemetry pill, and `D10` `<= 12.00 MiB` memory + `3.00 MiB` LRU checkpoint pool meter.
6. **Collapsible 3-Tab Editorial Inspector Drawer (`I`):**
   - **`Inspect`:** Context-sensitive View Controls (sliders & chips preserving DOM identity mid-drag and on canvas hover), live Ray-Pick Cell & Provenance Inspector, and View Diagnostics.
   - **`Readouts`:** Epistemic governance scope, `0 B` diff 4-checkpoint production table (`BF16 / MXFP8 / NVFP4 / Q5_K_M`), `(4 + 5k) / 9` elastic depth table, `1 → 256K` KV scaling points, `D10` memory bounds, and explicitly badged Paper Results (`arXiv:2611.03998`).
   - **`Ledger DR1–DR24`:** Instant substring search (`DR7`, `DR17`, `DR23`, etc.) and category filters (`All (24)`, `Guards`, `Spec`, `nGPT Math`, `Precision`, `KV Cache`, `Self-Spec`, `Provenance`).

---

## 2. The Seven Editorial Spatial Views

Switch views via the top tablist or keys `1`–`7`:

| # | Tab | What It Shows | Key Interactive Controls & 2D `<svg>` Insets |
| --- | --- | --- | --- |
| **1** | **`Macro 72L`** | Physical **72-sublayer (`16 + 40 + 16`)** topology toggling between **Folded** (shared recurrent loop arc + persistent anchor rail `x`) and **Unrolled** (`32 + 40k` executed depth with shared SHA-256 weight-ID rails), paired with the **KV-Bank Rail** (`36` `last_loop`, `16 + 20k` `per_loop`, or `104` Contract A banks). | Toggle **`Folded (72L Phys)` / `Unrolled`**, scrub physical sublayer `0..71` (`Pair L{⌊p/2⌋}`), select recurrent loop pass `t = 1..k`. |
| **2** | **`nGPT Sphere`** | **Channel-wise nGPT NLERP** residual cutaway (`N(h)`, `N(o)`, `b - a`, dual raw `α` vs. effective `|α_hat|` highlighting planted `> 1` extrapolation channels at `p = 16` ch `[6, 7]`, pre-norm blend, and unit-sphere output). Badged **`NOT Trigonometric SLERP` (`DR7`)**. | Scrub sublayer `0..71`; toggle **Read-Only Diagnostic Counterfactual Coda** (`0` KV mutation) and probe `s_1..s_k`. Includes **Exact 2D `span(a, b)`** + **Exploratory 2D PCA** (`8D`-only metrics + local `120°` Jacobian caveat) `<svg>` insets. |
| **3** | **`GQA & RoPE`** | **4:1 Gated GQA & Dual-RoPE** cutaway contrasting **SWA (`W = 8, θ_S = 16`)** vs. **Full (`θ_F = 1,024`)**: `Q/K/V/Gate` projections (`8 → 64` expansion disclosed vs. prod `1.60×`), `RoPE → Q/K L2Norm` (`DR6`), **Q-only `sqk²`**, live `RG32F` G-channel masks (`DR20`), sigmoid output gate (`s_gate²`), and concatenated `attn_sub_norm`. | Scrub even attention layers `0..70` (or quick presets `P0 SWA`, `P6 Full`, `P16 Rec SWA`, `P22 Rec Full`), switch `Q0..Q7` (`KV0/KV1`) & rotary planes `0..3`. Includes **4-Plane Dual-RoPE Phasor** `<svg>` inset. |
| **4** | **`SwiGLU`** | Bias-free **Normalized SwiGLU** (`8 → 24`, `3.00×` vs. prod `2.80×`), **Family 4 signed `suv_gate` / `suv_up`** (`raw * √d`, init `1.0`, NO `.abs()`), `SiLU` product, `ffn_sub_norm`, `down_proj`, and NLERP residual. | Scrub odd MLP layers `1..71`, intermediate channel `0..23`, and **`down_proj Accum Reveal` (`0%..100%`)**. |
| **5** | **`KV & last_loop`** | Side-by-side **immutable dual-session** comparison (`per_loop` vs. `last_loop`, identical prompt & seed), highlighting SWA eviction (`S > 8`) and **`DR17` Theorems 1–3** prefill-partition divergence. | Switch partition probe (**`Single-Shot (Thm 1)`**, **`Chunked [4, rem] (Thm 3)`**, **`1-by-1 Decode (Thm 3)`**) and attention bank layer. Includes **`1 → 256K` Production BF16 KV Scaling** `<svg>` chart (`4×` knee at `S = 1,024` & `S`-invariant `(4+5k)/9` ratio). |
| **6** | **`Self-Spec §6.2`** | **Reconstructed Paper §6.2 Contract A (`104` banks)**: 4 shallow (`k = 1`) draft passes writing shared Input + Loop-1 KV & persistent **`output_shallow` (`8` banks)**, batched `k = 4` verify reusing `(x, s_1)` + Input/Loop-1 KV, **`output_deep` (`8` banks)**, division-free binary64 rejection/residual, and `cap_swa = 16` rollback (`DR19`). | Inspect verify offsets `d_1..d_4` & `Bonus`, switch attention layer (`P16/P22/P56`), click **`Fork Live Contract A Session`** or **`Self-Spec §6.2`** in the dock, and adjust `T` to observe stochastic rejection/residual cycles. Includes **`(q_j, p_j)` Rejection & Residual** `<svg>` inset. |
| **7** | **`Bookends`** | Untied unit-norm **`embeddings` & `lm_head`** separating raw logits → FP32 `logit_scale / 0.0177` scaled logits → binary64 softmax, plus **`0 B` diff 4-checkpoint readouts** and **TBPTL depth-1** (`READ-ONLY · NO GRADIENTS`). | Inspect read-only checkpoint rows (`BF16`, `MXFP8`, `NVFP4`, `Q5_K_M` — zero fake quant toggles!) and scrub vocab token ID `0..63`. Includes **`READ-ONLY` TBPTL Depth-1 & Training** `<svg>` schematic. |

---

## 3. Elastic Depth `k`, `DR18` Immutable Forking & Self-Speculative Mode

- **Why Switching `k` or `per_loop ↔ last_loop` Forks a Branch (`D6` / `DR18`):** Populated KV slabs are tied to their depth (`16 + 20k` vs. `36` banks) and loop-writer provenance. Whenever you click a new `k` (`1..4`, or `6, 8` after checking **`+6,8`**), switch `per_loop ↔ last_loop`, or enter **`Self-Spec §6.2`**, Dory spawns a clean non-aliased branch from the active prompt preset and displays an inline **`DR18` status banner** — your prior branch remains completely untouched in the branch selector!
- **Trying Stochastic Contract A Self-Speculation:** Click **`Self-Spec §6.2`** in the bottom dock (forks a `104`-bank `k = 1 → 4, per_loop` branch), optionally drag the **`T` slider** above `0.00` (e.g., `T = 0.60`), and click **`+ Spec Cycle (γ=4)`** to run live division-free binary64 rejection sampling and inspect accepted drafts, residual corrections, bonus tokens, 1-token Shallow Coda catch-up (on full acceptance), or `0` extra Shallow Coda work (on rejection) with `cap_swa = 16` rollback.

---

## 4. 4-Act / 14-Beat Guided Tour & Keyboard Shortcuts

Click **`Guided Tour (14)`** (or press **`T`**) to launch a strictly isolated walkthrough across **4 Acts / 14 Beats** covering all 7 views and displaying the full immutable 8-tuple `DoryExecutionCoord` (`p`, `e`, `t`, `pos`, `step`, `branch`, `policy/bank`, `writerLoop`) on every beat:

1. **Act I · Hypersphere Foundations (Beats 1–3):** Untied unit-sphere bookends, physical `72`-sublayer (`16 + 40 + 16`) topology, and channel-wise nGPT NLERP (`DR7`) + active `σ_0 = 0.0177`.
2. **Act II · Inside the Normalized Mixers (Beats 4–6):** Gated 4:1 SWA GQA (`sqk²`, `DR6`, live G-channel mask, sigmoid gate, `attn_sub_norm`), interleaved 3:1 Full attention (`64×` mini / `1,000×` prod `θ_F`), and Normalized SwiGLU (signed `suv`, `ffn_sub_norm`).
3. **Act III · Shared-Weight Latent Recurrence & Dual-Axis KV (Beats 7–10):** Unrolling `k = 1 → 4` on identical weights (`192L`, `96` banks), anchor re-injection `u_t = N(s_{t-1} + x)` + Exploratory 2D PCA & read-only Counterfactual Coda, `per_loop` vs. `last_loop` `DR17` partition divergence, and SWA eviction + `1 → 256K` `(4 + 5k) / 9` scaling.
4. **Act IV · Elastic Depth, Paper §6.2 Self-Speculation & Production Context (Beats 11–14):** Reconstructed Contract A `104`-bank self-speculation, division-free binary64 rejection sampling & `cap_swa = 16` rollback, `0 B` diff 4-checkpoint production readouts, and `READ-ONLY` TBPTL depth-1 + `D10` `<= 12 MiB` memory summary.

Clicking any `DR*` reference chip inside the Tour HUD opens the **`Ledger DR1–DR24`** tab filtered to that entry. Exiting the tour (`Exit Tour` or `Esc` outside inputs) restores your Free-Explore branch, record, view, focus, camera, and inspector state identically.

### Pointer & Keyboard Shortcuts

| Input | Action |
| --- | --- |
| **Left-drag / Right-drag (or Shift-drag) / Scroll** | Orbit, pan, and zoom the 3D WebGL2 drafting stage |
| **Hover / Click** | Ray-pick 3D sublayer blocks, NLERP sheets, SWA/Full plates, KV banks, or bookend matrices |
| **`1` – `7`** | Switch across all 7 editorial spatial views |
| **`T`** | Start or exit the isolated 4-Act (14-beat) Guided Tour |
| **`←` / `→`** | Step previous / next beat while Guided Tour is active (or navigate focused tablists with `Home`/`End`) |
| **`R`** | Reset 3D camera to the active view preset bounds |
| **`I`** / **`C`** | Toggle the 3-tab Inspector drawer / 2D analytical `<svg>` charts inset |
| **`?`** | Open the focus-trapped Keyboard Shortcuts dialog |
| **`Esc`** | Close modal or exit active Guided Tour (automatically deferred inside text inputs such as Ledger search) |
