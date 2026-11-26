# Nemotron-Labs-Diffusion-14B — User Guide

This guide explains how to explore the **Nemotron-Labs-Diffusion-14B** interactive 3D WebGL2 drafting-table visualizer (`/diffusion/` locally, live at <https://12ohan.github.io/llm-viz/diffusion/>), switch between all 4 live generation paths, scrub the 2D unit-circle YaRN RoPE phasor across the `p = 16` Llama-4 boundary, and inspect every production metric and `D1–D24` discrepancy entry.

> [!NOTE]
> **Synthetic Miniature Simulator vs. Canonical 14B Instruct Metadata**
> The live 3D stage executes NVIDIA's exact **Ministral 3 / Nemotron-Labs-Diffusion** equations (`40` dense layers, `32Q/8KV`, dyadic YaRN RoPE + Llama-4 query temperature scaling, SwiGLU, draft-only `o_proj` LoRA) on deterministic synthetic miniature Float32 weights (`d = 8`, `dFf = 24`, `V = 36`, `rLora = 4, α = 16`). Note the **explicit projection-width reversal**: miniature expands `8 → 256/64`, whereas canonical `14B` (`nvidia/Nemotron-Labs-Diffusion-14B` @ `f8c3e2c0`) contracts `5,120 → 4,096/1,024` (`25.16 GiB` / `27.01 GB` BF16 backbone + `180.00 MiB` / `188.74 MB` FP32 LoRA).

---

## 1. Workspace Layout

The screen is organized into six coordinated drafting-table regions:

1. **Top Navigation Bar:** Model title + epistemic pill (`40L` miniature simulator disclosure), the **7 Spatial View tabs** (`Macro 40L`, `RoPE GQA`, `SwiGLU`, `Masks`, `Spec & KV`, `Embedding`, `Output Head`), **`Guided Tour`**, **`Inspector`**, **`Fit View`**, and quiet 3-way reciprocal links (**`Nano ↗`** and **`TwoTower ↗`**).
2. **Mode, Execution & Context Control Strip:** Five execution mode presets (`AR`, `DLM Causal`, `DLM Bidir`, `Linear Spec`, `Spec + LoRA`), transport buttons (`◀ Back`, `Fwd ▶`, `Step`, `Run`, `Reset`), and context-sensitive view controls (Layer `L01..L40`, `Q00..Q31`, `KV0..KV7`, 5-tab Mask Explorer switcher, SwiGLU `Accum Reveal` slider, and multi-step operation steppers).
3. **Provenance Badge Strip & Guards:** Live KV provenance badge (`token_causal` vs. `block_bidirectional`), adapter state (`LoRA OFF` vs. `o_proj LoRA ON`), record index, `KV len/32`, and an inline **Re-Prefill Confirmation Banner** whenever switching between causal and bidirectional KV provenance.
4. **Interactive 3D WebGL2 Stage & Overlays:** Full 5-pass WebGL2 scene with 3D→2D projected callouts, floating cell popovers, the **2D Unit-Circle RoPE Phasor SVG Inset** (in `RoPE GQA`), the **4-Act / 26-Beat Guided Tour HUD** (`Guided Tour`), and the **3-Tab Editorial Inspector Drawer** (`Inspect | Architecture | Ledger D1–D24`).
5. **Unified Tri-Mode Event Timeline Dock:** Horizontal multi-branch timeline showing `Prefill`, `AR Step`, `DLM Block`, and `Spec Cycle` nodes, copy-on-step auto-fork badges, inactive branch delete buttons (`×`), and live `<= 12.00 MiB` retained-memory meter.
6. **Docked Formula Inspector Bar:** Displays the active or clicked tensor equation, coordinate, live numeric value, and committed `V = 36` token stream preview.

---

## 2. The Seven Spatial Views

Select any view from the top tablist (or focus a tab and press `←` / `→` / `Home` / `End`):

| Tab | What It Shows | Key Interactive Controls |
| --- | --- | --- |
| **`Macro 40L`** | Single physical **40-layer homogeneous dense decoder** (`encoder.layers.0..39`), continuous residual spine, **`0..31` shared KV slab rail** (`20.00 KiB/tok` mini / `160.00 KiB/tok` 14B), and detachable **draft-only `o_proj` LoRA bridge** (`80` FP32 tensors). | Select layer `L01..L40`, click **`Inspect L01 GQA →`** (or double-click a layer bay), or click any bay to inspect specs. |
| **`RoPE GQA`** | **`32Q / 8KV` (4:1) Grouped-Query Attention** cutaway (`Q00..Q31`, `KV0..KV7`), dyadic YaRN rotary planes, Llama-4 query temperature scaling, and `o_proj` base vs. `+4.0 · B(A · a)` LoRA delta packed into a `256×128` (up to `256×256` authorized) `RG32F` atlas. | Switch `Layer`, `Q Head`, and `KV Group`; use the **2D Unit-Circle RoPE Phasor Inset** to switch rotary planes `(0,4)..(3,7)` and scrub position `p = 0..31` across **`p = 16`**. |
| **`SwiGLU`** | Post-attention RMSNorm → parallel `W_gate → SiLU` and `W_up` → Hadamard `⊙` → `W_down` → residual add (`64×64` `RG32F` atlas), highlighting that SwiGLU comprises `74.53%` of 14B backbone parameters. | Select `Layer` (`L01..L40`) and drag **`Accum Reveal` (`0%..100%`)** to watch progressive dot-product accumulation. |
| **`Masks`** | **5-Tab Attention Mask Explorer** (`64×64` `RG32F` G-channel bound texture) comparing training and inference attention visibility alongside **3-coordinate labels** (`seq_idx`, `ropePositionIds`, `cachePositions`, `D23`). | Switch across **`Joint [x_t‖x_0]`**, **`AR Causal`**, **`Block Bidir`**, **`Commit / Verify`**, and **`VLM Quadratic SS`**. |
| **`Spec & KV`** | **Linear Speculative Verify & KV Rollback Strip** contrasting `B-1` drafted tokens against shifted causal verify predictions (`verify[i] → draft[i+1]`), plus `0..31` KV cache slot lifecycle and **`D19` dual-crop provenance**. | Run **`Linear Spec`** or **`Spec + LoRA`** steps and click alignment connectors or KV slots to inspect accepted/cropped positions. |
| **`Embedding`** | Untied **`V = 36` `encoder.embed_tokens`** lookup (`0..31` words plus `32 [MASK]`, `33 <|im_end|>`, `34 <s>`, `35 </think>`), explicitly noting zero additive positional table (positions enter via per-layer RoPE). | Click token rows or embedding cells to inspect live vector values. |
| **`Output Head`** | Final `encoder.norm` → untied **`diffusion_head`**, contrasting **causal shifted (`+1`)** next-token logits vs. **diffusion same-position (`+0`)** logits and showing un-suppressed **`[MASK]` softmax probability** (`D18`). | Switch between `AR` / `Linear Spec` and `DLM` records to compare `+1` vs. `+0` prediction alignment. |

---

## 3. Execution Modes & Graceful Re-Prefill Guard

Choose any of the 5 mode buttons in the control strip:

- **`AR` (`ar_generate`):** Standard token-causal autoregressive decoding (1 token/step, `o_proj` LoRA OFF, `token_causal` KV provenance, explicit causal mask per `D8`).
- **`DLM Causal` (`causal_context = true`):** Block diffusion (`B = 4`) keeping **`token_causal`** KV provenance. Slot `0` is seeded from prior causal logits (`D7`), leaving `B - 1` `[MASK]` slots; denoising concatenates transient KVs (`use_cache = False`), runs Sampler A, and commits causally (`use_cache = True`).
- **`DLM Bidir` (`causal_context = false`):** Block diffusion with **`block_bidirectional`** KV provenance where all `B` slots start `[MASK]`.
  - **Graceful Re-Prefill Guard:** Because causal and bidirectional KV caches cannot be silently mixed (`KvProvenanceMismatchError`), switching from a causal record to **`DLM Bidir`** (or vice versa) opens the amber **Re-Prefill Banner**. Click **`Re-Prefill KV & Switch`** to explicitly re-prefill committed tokens under the new mask provenance, or **`Cancel`** to stay on the current mode.
- **`Linear Spec` (`linear_spec_generate`, LoRA OFF):** Drafts `B - 1` tokens in `use_cache = False` bidirectional mode via **Sampler B**, verifies in 1 causal pass, and crops rejected suffix KVs via `_crop_dynamic_cache`.
- **`Spec + LoRA` (`linear_spec_lora`):** Turns the **`o_proj` LoRA adapter (`+4.0 · B(A · a)`) ON exclusively for draft passes** and disables it identically (`0`-ULP OFF) on causal verify passes (`D17`).

> [!TIP]
> Approaching `kvCapacity = 32` never silently shrinks `B` down to `1` (Clarification #2). When fewer than `B` slots remain (`< 1` for `AR`), the session halts cleanly with `lifecyclePhase = 'capacity_exhausted'`.

---

## 4. Using the 2D RoPE Phasor Inset & 5-Tab Mask Explorer

### 2D Unit-Circle RoPE Phasor Inset (`RoPE GQA` View)

- **4 Dyadic YaRN Planes:** Click `(0,4)` (`1.0`, `original`), `(1,5)` (`0.3125`, `blended`), `(2,6)` (`0.0625`, `interpolated`), or `(3,7)` (`0.03125`, `interpolated`) to rotate live `q_pre` and `k_pre` pairs on the unit circle.
- **`p = 16` Llama-4 Scrubber:** Drag the position slider (`p = 0..31`) past the **`p = 16` boundary marker** (equivalent to `16,384` in 14B production):
  - For `p <= 15`, `q_scaled` stays on the inner unit circle (`s_L4 = 1.0000×`).
  - At `p >= 16`, `q_scaled` jumps outward onto the dashed **`r = 1.069 (L4 p≥16)`** ring (`+6.93%`), while `k_rot` stays fixed on `r = 1.0`. Click **`Sync Live`** to return to the active record's query position.

### 5-Tab Attention Mask Explorer (`Masks` View)

1. **`Joint [x_t‖x_0]`:** BD3-LM dual-stream training plate showing `M_BD` (noisy intra-block), `M_OBC` (noisy attending to clean prefix with strict `Δp >= 1`), and `M_BC` (clean causal prefix), along with `seq_idx` vs. `ropePositionIds` vs. `cachePositions` (`D23`).
2. **`AR Causal`:** Strict lower-triangular causal mask (`D8`).
3. **`Block Bidir`:** Block-diagonal bidirectional diffusion mask.
4. **`Commit / Verify`:** Committed prefix + shifted causal verify attention mask.
5. **`VLM Quadratic SS`:** Badged **read-only reference** from `VLM-8B` (`k(k+1)` interleaved queries over `[draft_kv ‖ clean_kv]`; `D10` notes absence from all 6 text checkpoints, `D24` notes upstream greedy-only behavior).

---

## 5. Unified Tri-Mode Event Timeline & 3-Tab Inspector

### Unified Tri-Mode Event Timeline Dock

- **Click Any Node to Rewind:** Click any `Prefill`, `AR Step`, `DLM Block`, or `Spec Cycle` pill to seek non-destructively to that record. Historical non-eager layers and cropped speculative suffixes reconstruct at **`0`-ULP** precision on demand.
- **Copy-on-Step Auto-Forking:** Stepping from a rewound record automatically forks a new branch (`branch-2`, `branch-3`, …) while preserving the original branch intact.
- **Branch Cleanup & Budget:** Click **`×`** next to any inactive branch to delete it (the active or sole remaining branch is protected). The top-right memory meter tracks live usage against the **`12.00 MiB`** hard session retention cap.

### 3-Tab Editorial Inspector (`Inspector` Button)

1. **`Inspect`:** Live record provenance, **3-counter `D12` NFE & TPF** (`Actual Fwd`, `Decode Fwd`, `Ref NFE`), active formula/part details, **`D19` dual-crop provenance** (`Browser Post-EOS Crop` vs. `Upstream Pre-EOS Crop`), guard warnings, and decoded `V = 36` token chips.
2. **`Architecture`:** Canonical `14B` Instruct `0 B` LFS summary (`25.16 GiB / 27.01 GB` backbone + `180.00 MiB / 188.74 MB` LoRA), projection-width reversal callout, 12-row `14B` component breakdown, 7-variant family matrix, **4-tier context KV scaling table** (highlighting the `16,384` Llama-4 crossing), 4-way browser memory boundaries, and all 13 `0 B`-diff LFS files.
3. **`Ledger (D1–D24)`:** Instant substring search (`D19`, `YaRN`, `LoRA`, etc.) and 4 category filters (`All (24)`, `Architecture`, `RoPE & Masks`, `Exec & KV`, `Samplers & Guards`) across all 24 verified discrepancies.

---

## 6. Guided 4-Act / 26-Beat Walkthrough & Keyboard Shortcuts

Click **`Guided Tour`** in the top bar to launch an **isolated 26-beat tour** across 4 Acts:

1. **Act 1 · 40L & `[x_t‖x_0]`:** Shared 40L homogeneous dense backbone, projection-width reversal disclosure, `[x_t ‖ x_0]` mask, and untied `diffusion_head`.
2. **Act 2 · YaRN & SwiGLU:** `32Q/8KV` GQA, exact dyadic YaRN frequency ramp, crossing `p = 16` Llama-4 scale (`1.0000× → 1.0693×`), live `RG32F` attention softmax, and `74.5%` Dense SwiGLU cutaway.
3. **Act 3 · AR vs. DLM:** Causal AR vs. `DLM Causal` (AR-seeded slot 0) vs. `DLM Bidir` (explicit re-prefill & `block_bidirectional` KV), plus read-only VLM-8B Quadratic SS reference.
4. **Act 4 · Spec, LoRA & KV:** `Linear Spec` (LoRA OFF), detachable draft-only `o_proj` LoRA (`0`-ULP OFF on verify), shifted verify alignment, `_crop_dynamic_cache` rollback + `D19` provenance, and `<= 12.00 MiB` shared-slab memory summary.

Your Free-Explore branches, active record, view, layer, RoPE scrubber, and inspector state are saved on entry and **restored identically** when you click **`Exit Tour`** (or press `Esc` with the inspector closed).

### Pointer & Keyboard Shortcuts

- **Pointer:** **Left-drag** orbit · **Right-drag / Shift-drag** pan · **Scroll** zoom · **Click** pick bay or sheet cell · **Double-click** layer bay in `Macro 40L` to jump into `RoPE GQA` for that layer.
- **Tablist A11y:** When focus is inside the Top View tablist, Mask Explorer tablist, or Inspector tablist, use **`←` / `→`**, **`Home`**, and **`End`** (WAI-ARIA roving `tabindex`).
- **Tour & Drawer Keys:** **`←` / `→`** step Tour beats (outside inputs/tablists) · **`Esc`** closes Inspector Drawer (if open) → clears pinned cell → exits Guided Tour.
- **Reduced Motion (`prefers-reduced-motion: reduce`):** Camera flies and tour transitions settle instantaneously with zero continuous animation.
