# Nemotron-Labs-TwoTower — User Guide

This guide explains how to explore the **Nemotron-Labs-TwoTower 30B-A3B** interactive 3D WebGL2 visualizer (`/twotower/` locally, planned at `https://12ohan.github.io/llm-viz/twotower/`), understand its dual-tower block-diffusion lifecycle, and interpret every inspector readout accurately.

> [!NOTE]
> **Synthetic Miniature Execution vs. Production Metadata**
> The live 3D stage executes NVIDIA's exact `NemotronHTwoTowerForCausalLM` equations on a deterministic miniature model (`d = 8`, `V = 34`, synthetic seeded weights) so every step computes in milliseconds with zero external runtime dependencies. Production weight/cache numbers (`63.19B` params · `126.37 GB` SI / `117.69 GiB` IEC · `128K` operational / `256K` config limit) reflect pinned Hugging Face revision `35b6498508be8856d380cdfe4a900a79b918d1ea` and [`arXiv:2606.26493v2`](https://arxiv.org/abs/2606.26493).

---

## 1. Dual-Tower Workspace Layout

The screen is organized into five coordinated regions:

1. **Top Bar:** Brand title + epistemic pill (`d=8 · V=34 · 2×52L · Synthetic Mini`), the **7 Spatial View tabs** (`Macro`, `GQA`, `Mamba`, `Timestep`, `MoE`, `Embed`, `Output`), and action buttons (**`Guided Tour`** [`T`], **`Reset View`** [`R`], **`Workbench`** [`W`], mobile **`Inspector`**, and quiet reciprocal **`Nano ↗`** navigation to the Nemotron 3 Nano single-tower visualizer).
2. **Execution & Playback Strip:** Mode selector (`Mask-Diffusion`, `AR`, `Mock-AR`), step buttons (`Init Block`, `Refine`, `Commit Conf`, `Commit → Context`, `Step AR`, `Step Mock-AR`, quick-advance `Step`), live status badges (phase, KV capacity, canonical timestep `t`), and the clearly labeled **`Macro Flow Replay`** button (`Instructional replay ≠ measured compute`).
3. **3D WebGL2 Viewport & Bottom View Dock:** Interactive 3D canvas with floating HUD title/historical badge, optional Guided Tour card, and view-specific bottom controls (bay selectors, head/position pickers, linker channel toggles, and partial-accumulation Math Steppers).
4. **Right 3-Tab Editorial Inspector:** Three segmented tabs — **`Inspect`** (historical config, timestep status, 10-axis selection identity, live formula, view diagnostics), **`Sampler`** (Advanced MDLM Sampler inspector + non-retroactive config scope), and **`Architecture`** (production `63.19B` inventory, 4-way memory separation, 3-regime context scaling chart/table, discrepancy ledger) — plus a bottom telemetry footer (`JS cb` main-thread callback time & continuous `RAF Δ` cadence).
5. **Persistent Nested Timeline Dock:** Bottom strip with `← Back`, `Forward →`, branch selector, `Fork`, `Delete Branch`, `Restart`, retained-bytes readout, and clickable nested milestone nodes.

---

## 2. The Seven Spatial Views

Switch views via the top tablist (or `←` / `→` / `Home` / `End` when the tablist is focused):

| Tab | What It Shows | Key Interactive Controls |
| --- | --- | --- |
| **`Macro`** | Side-by-side **`52 × 2`** dual towers: **Context Tower** (causal AR persistent cache owner) on the left and **Denoiser Tower** (block-diffusion ephemeral scratchpad) on the right, bridged by **`29` aligned read-only transfer linkers** (`23` Mamba-2 + `6` GQA) and **`0` MoE linkers**. | Click any Context or Denoiser bay for 10-axis provenance; filter linker channels (`All`, `GQA KV`, `Mamba`, `Off`); run pauseable **`Macro Flow Replay`**. |
| **`GQA`** | Asymmetric **35/65 Paired GQA** across the `6` attention bays (`06, 13, 20, 27, 34, 43`). Context shows causal prefix `KV[0..P-1]`; Denoiser shows rectangular bidirectional **`[B × (P+B)]`** attention reading inherited Context KV plus full intra-block tokens. Packs all `32` miniature Q heads + `2` KV heads into a `128×128` `RG32F` atlas while displaying 2 selected Q-head score/softmax slices. | Choose bay (`06..43`), focus tower (`Both`, `Context`, `Denoiser`), select Q head per KV group, click matrix cells, and play/scrub the partial-accumulation Math Stepper. |
| **`Mamba`** | Asymmetric **35/65 Paired Mamba-2** across the `23` Mamba bays. Visualizes 4-tap **Conv1D boundary replacement (`3→2→1→0`)**, local causality (note how Bay 08+ still receives bidirectionally mixed inputs from Bay 06 GQA!), and a closed-form **`4×4` analytical SSM decomposition** (`H_p = A_cum·H_init + ΔH_write + ε_f32`) for a selected head alongside the `16`-head / `8`-group `256×128` `RG32F` atlas. *Note: decay and L2 norms measure algebraic state magnitude, not semantic retention.* | Choose Mamba bay (`01..51`), head (`H00..H15`), block position (`p = 0..B-1`), and step through Conv1D/SSM/readout partial sums (final values stay masked while accumulating). |
| **`Timestep`** | Dedicated **PixArt-α `adaLN-single`** conditioning cutaway tracing `t = nMasked / B` → COS→SIN sinusoidal `φ(t)` → `t_embedder` → `t_block` → `52 scale_shift_tables` → `(shift, scale, gate)`. Contrasts split ordering: **`modulate → RMSNorm`** on `M`/`*` bays vs. **`RMSNorm → modulate`** on `E` bays, untouched residual skip `X`, and unmodulated final `norm_f`. | Pick bay kind/index (`M`, `*`, `E`), inspect live `(shift, scale, gate)` channels, or observe `BYPASSED` state during `AR` and `Mock-AR`. |
| **`MoE`** | **Granular MoE (`128+1`)** across the `23` stateless MoE bays (`0 B` sequence cache, zero cross-tower transfer). Inspects the **FP64 router 6th-vs-7th cutoff margin**, FP32 `e_score_correction_bias`, Top-6 `×2.5` unbiased normalization, `ReLU²` routed experts, and the always-active double-wide shared expert. | Choose MoE bay (`02..52`), token position, and step through Top-6 partial `Σσ` accumulation (final normalized weights remain hidden until step completion). |
| **`Embed`** | Independent untied **NoPE token embeddings** (`W_embed_ctx` vs. `W_embed_den`, `V = 34` with `[MASK]` `#32` and `<eos>` `#33`). Notes that production TwoTower repurposes `[INST]` (ID `3`) as `[MASK]` with near-orthogonal Denoiser divergence. | Inspect active prompt/block token lookups across both towers and verify zero positional tables are added. |
| **`Output`** | Unmodulated final `norm_f` → untied Denoiser `W_lm_head_den` predicting unshifted clean tokens `x₀[P + j]`. Compares **raw LM-head probabilities** vs. **constrained MDLM/SUBS probabilities** (`maskTokenProb = 0`, locked prior carry-over, zero `Infinity` display). | Select block slot `j ∈ [0..B-1]`, click Top-K candidate pills, and switch to the **`Sampler`** sidebar tab for full per-slot unmasking logic. |

---

## 3. Execution Modes & the 4-Step Block Diffusion Lifecycle

Choose the active mode in the Execution strip (`Mask-Diffusion`, `AR`, or `Mock-AR`):

- **`Mask-Diffusion` (Default):** Generates `B` tokens in parallel on Denoiser conditioned on Context's frozen `P`-token cache boundary, then commits the finished block once to Context.
- **`AR` (`Step AR`):** Runs standard single-token causal decoding on Context Tower only. Denoiser and timestep modulation are bypassed (`t = null`).
- **`Mock-AR` (`Step Mock-AR`):** Transfers Context's `KV[:-1]` and pre-last-token Mamba snapshot **`mamba_s2`** (`+47.08 MiB` in production) so Denoiser re-evaluates `lastTokenId` causally (`S = 1`) with `adaLN-single` bypassed (`t = null`). Serves as an exact conditioning parity harness (`0`-ULP agreement with Context AR when `ΔW = 0`).

### The 4-Step Block Diffusion Lifecycle

1. **`Init Block` (`diffusion_block_init`):** Initializes a fresh `B`-token Denoiser block filled with `[MASK]` tokens (`nMasked = B`) bound to Context's current prefix length `P` and cache digest. No Denoiser pass has run yet (`conditioningMode: inactive_no_pass_yet`).
2. **`Refine` (`diffusion_refinement` — Eval):** Executes a full 52-layer Denoiser pass at canonical **`t = nMasked / B`** using a read-only clone of Context's `P`-token boundary (`23` Mamba states + `6` GQA KV caches). Context's persistent cache stays bit-for-bit (`0`-ULP) untouched.
3. **`Commit Conf` (`diffusion_confidence_commit`):** Runs the MDLM/SUBS confidence selector (`conf > γ` or remaining-budget floor `ceil(nMasked / remainingSteps)`), locking chosen positions monotonically. Repeat **`Refine` → `Commit Conf`** while `nMasked > 0` and `remainingSteps > 0`.
4. **`Completed (Uncommitted)` → `Commit → Context` (`diffusion_block_commit`):** Once `nMasked = 0` (either early or on step `K`), the block enters **`Completed (Uncommitted)`** — finalized on Denoiser, **not yet ingested into Context** (`P` is unchanged, no further Denoiser evaluation is needed). Clicking **`Commit → Context`** runs one causal Context forward pass over the finalized `B` tokens, advancing `P → P + B`, updating Context's persistent cache digest, and checking post-block `<eos>` (`#33` mini / `[2, 11]` prod).

> **Tip:** Click the primary **`Step`** button to automatically trigger the next legal action in the current mode's lifecycle.

---

## 4. Reading `t = nMasked / B` & Historical Boundary Mismatch Banners

- **Why `t = nMasked / B` (Never `r / K`):** In TwoTower, the timestep scalar fed into `t_embedder` is strictly the **empirical fraction of `[MASK]` tokens entering the executed Denoiser pass** (`nMasked / B`). If step `r = 0` unmasks 2 of 4 slots, step `r = 1` immediately jumps to `t = 2/4 = 0.500` (not `1/4` or `3/4`).
- **Reading the Timestep Status Card (`Inspect` tab):** Separates:
  - **Executed Pass `t`:** The actual input mask fraction of the displayed Denoiser pass (and any diagnostic override shown separately from empirical counts).
  - **Current Block Mask:** Remaining `[MASK]` count on the current record (updated immediately after `Commit Conf`).
  - **Next Eligible Step:** Shown only while `nMasked > 0 && remainingSteps > 0`.
  - **Conditioning Mode:** Clearly distinguishes active diffusion (`active_diffusion`), pre-pass initialization (`inactive_no_pass_yet`), historical pass inspection (`historical_prior_pass`), and genuine `AR` / `Mock-AR` bypass (`bypassed_ar` / `bypassed_mock_ar`).
- **Historical Denoiser Boundary Mismatch Banner:** Whenever you inspect a record that has **no Denoiser pass of its own** — for example, right after `Commit → Context` (`P = 8 → 12`), after an `AR` step, or on Block 1 `Init Block` before pass 0 — Denoiser views show the most recent prior Denoiser pass paired with a prominent **Historical Boundary Mismatch** badge and banner stating both `P_cond` (the prefix length that actually conditioned that pass) and the current Context `P`. Prior Denoiser conditioning (`KV[0..7]`) is **never** silently relabeled as belonging to `P = 12`.

---

## 5. Persistent Nested Timeline & Branching

The bottom **Timeline Dock** tracks your full copy-on-step generation history:

- **Scrubbing:** Click any milestone pill (Context prefill, AR/Mock-AR steps, or nested Diffusion Block segments showing `Init`, `r=0..K-1` Eval/Commit pairs, and `Commit → Context`) or use **`← Back`** / **`Forward →`** to inspect any historical state without mutating later records.
- **Branching & Auto-Forking:** Click **`Fork`** to explicitly create a new branch from the current cursor, or simply execute a new step while rewound to automatically fork a new branch (`branch-1`, `branch-2`, …) preserving the original branch intact.
- **Branch Management:** Switch branches via the branch dropdown, delete any inactive branch with **`Delete Branch`** (strictly active-branch-safe — the active or last remaining branch cannot be deleted), or click **`Restart`** to reset to the initial `P = 8` prefill. The retention badge tracks total retained session typed-array bytes across deduplicated shared ancestors.

---

## 6. Advanced MDLM Sampler Inspector (`Sampler` Tab)

Open the **`Sampler`** tab in the right inspector rail to inspect and tune block unmasking:

- **Confidence Threshold `γ` Presets:**
  - **`0.22 Mini Default`** — Calibrated for synthetic `V = 34` logits so default trajectories show informative multi-step unmasking and early completion.
  - **`0.80 Paper`** — `arXiv:2606.26493v2` (§3.1 / §3.6) and model-card operating point.
  - **`0.90 Ref Code`** — Reference `modeling_nemotron_twotower.py` default.
  - Fine-tune any custom `γ ∈ [0.00, 1.00]` via the slider or number box.
- **Decoupled Block Width `B` & Refinement Budget `K`:** Independently select **`B = 4` or `B = 8`** and **`K ∈ [1..12]`** to explore `K < B`, `K = B`, and `K > B` regimes.
- **Temperature `T` & Deterministic Seed:** `T = 0.0` runs exact greedy argmax. Setting `T > 0` runs **deterministic Gumbel-max sampling** seeded by the explicit integer Seed (zero `Math.random()`).
- **Per-Slot Decision Table:** Shows every block slot's candidate token, raw vs. constrained MDLM confidence (`[MASK]` clamped to `0%`), and commit reason:
  - `Locked (prior step)` — Carried over immutably from an earlier refinement.
  - `Threshold (conf > γ)` — Exceeded strict `>` threshold.
  - `Budget floor` — Committed via `ceil(nMasked / remainingSteps)` to guarantee completion within `K` steps even if every remaining slot falls below `γ`.
  - `Remain [MASK]` — Deferred to the next refinement pass.
- **Calibration Caveat:** Unscaled MDLM softmax confidence (`x_θ`) is a relative heuristic for ordering block unmasking on synthetic miniature weights — **not** a calibrated measure of semantic correctness.
- **Non-Retroactive Config Scope:** Editing pending settings **never rewrites history**. Scope badges specify exact activation timing:
  - `γ`, `T`, `Seed` → **`Applies: Next Refinement`**
  - `B`, `K` → **`Applies: Next Block`**
  - Execution Mode → **`Applies: Next Step`**
  - Prompt / KV Capacity → **`Applies: Session Restart`**
  - Whenever your pending config differs from the inspected record's `configAtStep`, a **`Pending ≠ Hist`** diff banner highlights the distinction.

---

## 7. Guided 4-Act / 28-Beat Educational Walkthrough (`T`)

Press **`T`** or click **`Guided Tour`** to launch the self-contained 28-beat narrative tour across 4 Acts:

1. **Act I · Dual-Tower Architecture, Prefill & Three Execution Modes** (Beats `1–6`) — Independent `2×52` towers, `P = 8` NoPE prefill, Context cache regimes, isolated side branches for `AR` and `Mock-AR`, and `Mask-Diffusion` overview.
2. **Act II · Conditioning, Paired Layers, Recurrent Mechanics & Routing** (Beats `7–16`) — `[MASK]^4` init, `t = nMasked / B`, split `adaLN-single` ordering, 29 linker channels, Paired GQA `[B×(P+B)]`, Mamba Conv1D replacement (`3→2→1→0`), local causality after Bay 06, `4×4` SSM decay vs. writes, gated norm vs. residual gate, and stateless FP64 MoE.
3. **Act III · Denoiser Output, MDLM Sampler & Iterative Refinement** (Beats `17–23`) — Unshifted `V = 34` logits, raw vs. constrained MDLM (`maskTokenProb = 0`), threshold `γ` & calibration caveat, budget floor, live `t = nMasked / B` jump, monotonic locked slots, and Completed (Uncommitted) early finish.
4. **Act IV · Context Commitment, Next Block & Production Architecture** (Beats `24–28`) — Single Context block commit (`P = 8 → 12`), Block 1 conditioning on the new boundary, persistent vs. transient cache separation, production `BF16/FP32/FP64` & 3-regime scaling, and architectural trade-offs recap.

- **Strict Session Isolation:** The Tour runs on its own dedicated deterministic session (`P = 8, B = 4, K = 4, γ = 0.22, T = 0`). Your Free Explore branches, cursor, and pending config are saved untouched and restored identically when you exit Tour (`Esc` or `T`).
- Use the floating Tour card's Act jump pills, **`Prev`** / **`Next`** (or `←` / `→`), and **`Play`** / **`Pause`** (`Space`) while keeping free 3D orbit/pan/zoom unlocked.

---

## 8. Production Architecture & 3-Regime Context Scaling (`Architecture` Tab)

Open the **`Architecture`** tab in the right inspector rail to inspect pinned production-scale facts for `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16` (`rev 35b64985`):

- **Exact Dual-Unit Weight Inventory:** Context (`63.16 GB` SI / `58.82 GiB` IEC), Denoiser Backbone (`63.16 GB` / `58.82 GiB`), Time-Conditioning (`58` tensors · `30.02M` params · `60.04 MB` / `57.26 MiB`), Denoiser Total (`63.22 GB` / `58.87 GiB`), Grand Total (`63.19B` params · `126.37 GB` / `117.69 GiB`, `0 B` audit diff), and `46` FP32 `e_score_correction_bias` tensors (`+11,776 B` extra over BF16).
- **4-Way Memory Separation:** Explicitly separates (1) Persistent Context Cache (`47.08 MiB + P × 6 KiB`), (2) Optional Implementation Snapshots/Mirrors (`mamba_s2` `+47.08 MiB` in `Mock-AR`; optional cross-GPU read-only mirror), (3) Transient Denoiser Block KV (`B × 6 KiB`), and (4) **Explicitly Excluded Peak Runtime Workspace** (intermediate activations, `[B × 131,072]` logits, CUDA/framework scratchpad, allocator fragmentation).
- **3-Regime Context-Length Scaling Chart & Table:** Plots hybrid Context persistent cache (`1K → 1M`) vs. a 52-layer pure GQA Transformer (`8,035`-token crossover) across three clearly shaded regimes:
  1. **`Operational (≤128K)`** — Model-card operational range (`≤ 131,072` tokens).
  2. **`Configured Limit (≤256K)`** — `config.json` `max_position_embeddings = 262,144` (explicitly marked as **NOT** empirically validated production runtime support).
  3. **`Extrapolated (>256K)`** — Analytical linear KV scaling projection (`512K`, `1M`).
- **Discrepancy Ledger & Community NVFP4 Note:** Expand the inline ledger to review all 9 verified paper-vs-checkpoint discrepancies (`1.5M` vs. `30.02M` adaLN, `128K` vs. `256K`, `γ = 0.8` vs. `0.9`, etc.) and the explicitly separated non-official community NVFP4 (~`42 GiB`) reference.

---

## 9. Developer Workbench (`W`), Reciprocal Navigation, Shortcuts & Accessibility

- **Developer Diagnostic Workbench (`W` or `Workbench` button):** Opens a collapsible, bidirectionally synced diagnostic drawer displaying full session state digests, transition provenance, and manual execution controls. Press **`Esc`** or **`W`** to close.
- **Quiet Reciprocal Nano Navigation:** Click **`Nano ↗`** in the trailing top bar to jump directly to the single-tower Nemotron 3 Nano `v1.0.0` visualizer (`/` locally or `/llm-viz/` on GitHub Pages); Nano carries a matching **`TwoTower ↗`** link back to `/twotower/` (`/llm-viz/twotower/`).

### Pointer & Keyboard Shortcuts

- **Pointer:** **Left-drag** orbit · **Right-drag / Shift-drag** pan · **Scroll / Pinch** zoom · **Click / Tap** pick bay or tensor cell.

| Key | Action |
| --- | --- |
| `T` | Start or exit the 4-Act / 28-Beat Guided Walkthrough |
| `W` | Open or close the Developer Diagnostic Workbench drawer |
| `R` | Fit / reset the 3D camera to the active spatial view |
| `←` / `→` | Step Tour beat (when Tour is active outside tablists); switch focused View or Sidebar tabs |
| `Home` / `End` | Jump to first / last tab inside a focused tablist |
| `Space` | Toggle Tour autoplay (defers to native button activation when a control has focus) |
| `Esc` | Close Workbench drawer → exit Guided Tour → close mobile Inspector drawer (restores focus) |

### Reduced-Motion & Mobile / Narrow Viewports

- **Reduced Motion (`prefers-reduced-motion: reduce`):** Camera flies and tour transitions settle instantaneously, continuous ribbon pulses and Macro Flow Replay disable automatically, and all inspection/stepping remains fully functional.
- **Narrow & Mobile Controls (`≤ 768px`):** The right inspector collapses cleanly into an on-demand slide-over drawer toggled by the **`Inspector`** button in the top bar (closed with **`Esc`** or its close control), while the bottom view dock and timeline controls wrap cleanly without clipping.
