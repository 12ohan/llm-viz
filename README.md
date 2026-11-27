# NVIDIA Nemotron 3 Nano, Nemotron-Labs-TwoTower, Nemotron-Labs-Diffusion-14B & Nemotron-Labs-Diffusion-Uni — Interactive 3D Architecture Visualizations

A zero-runtime-dependency WebGL2 multi-page application (MPA) hosting **four isolated interactive 3D architectural explainers and miniature forward-pass simulators** in the visual language of Brendan Bycroft's `llm-viz`:

| Interactive App | Entry Path | GitHub Pages URL | Release Status | Architecture Scope |
| --- | --- | --- | --- | --- |
| **Nemotron 3 Nano 30B-A3B** | `/` (`index.html`) | <https://12ohan.github.io/llm-viz/> | **`v1.0.0`** (preserved & live) | Single-tower 52-layer causal hybrid (`23 Mamba-2 · 23 MoE 128+1 · 6 GQA`), live cached AR decode & branching, 5 cutaways, 22-beat tour, Selective FP8 & `1 → 1M` chart. |
| **Nemotron-Labs-TwoTower 30B-A3B** | `/twotower/` (`twotower/index.html`) | <https://12ohan.github.io/llm-viz/twotower/> | **`twotower-v1.0.0`** (released & live) | Dual-tower `2 × 52`-layer Context + Block-Diffusion Denoiser (`63.19B` / `126.37 GB` SI / `117.69 GiB` IEC), 7 spatial views, 3 execution modes, persistent nested timeline, MDLM sampler, 4-act / 28-beat tour. |
| **Nemotron-Labs-Diffusion-14B** | `/diffusion/` (`diffusion/index.html`) | <https://12ohan.github.io/llm-viz/diffusion/> | **`diffusion-v1.0.0`** (released & live) | Single `40`-layer homogeneous Ministral 3 dense decoder (`25.16 GiB` BF16 backbone + `180.00 MiB` FP32 draft-only `o_proj` LoRA), dyadic YaRN RoPE + Llama-4 `p = 16` scaling, 7 spatial views, 5-tab Mask Explorer, 4 live modes (`AR`, `DLM Causal/Bidir`, `Linear Spec`, `Spec + LoRA`), unified timeline, 3-tab Inspector (`D1–D24`), 4-act / 26-beat tour. |
| **Nemotron-Labs-Diffusion-Uni (NLD-Uni)** | `/uni/` (`uni/index.html`) | <https://12ohan.github.io/llm-viz/uni/> | **`uni-v1.0.0`** (released & live) | Unified multimodal `18.27B` Elastic-MoT (`34L`: `18` Joint `L0..17` + `16` Split `L18..33` parallel `U`/`G` towers, RADIO-H + Emu3 IBQ + `2×2` Packed Codec), 7 spatial views + 2D RGB canvas insets, live `D5` (`HF-Faithful` vs. `Elastic-Pruned`) toggle, 256-code PCA ball-tree (`U26` `D = 17` mismatch), Dual-CFG editing (`U18` border-lock), `<LOC_0..16>` grounding (`U27`), Read-Only OPD (`U28`), 3-tab Inspector (`U1–U30`), 5-act / 28-beat tour. |

All four apps compile as an isolated **Topology A 4-entry Vite MPA** enforced by an automated 7-gate 4-way mutual bundle-isolation check (`npm run verify:isolation`, covering all 12 directional pairs plus Uni's `D4` zero-`src/ui/controls.ts` gate) so TwoTower, Diffusion, and Uni code never leak into Nano's `/` bundle (and never cross-import each other), connected by **quiet, base-path-aware 4-way reciprocal navigation** (`Nano ↗`, `TwoTower ↗`, `Diffusion ↗`, `Uni ↗`) across both root `/` and GitHub Pages `/llm-viz/`.

> [!IMPORTANT]
> **3-Tier Epistemic Sourcing Disclosure (Applies to All Four Applications)**
>
> 1. **Directly Inspected Public Metadata:** Production topologies, dimensions, precision policies, and byte totals come strictly from pinned public sources — **Nano:** [`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848) & `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (`bf77c317`); **TwoTower:** [`arXiv:2606.26493v2`](https://arxiv.org/abs/2606.26493) & `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16` (`35b64985`, 24-shard header audit); **Diffusion-14B:** [`arXiv:2607.05722v1`](https://arxiv.org/abs/2607.05722), `nvidia/Nemotron-Labs-Diffusion-14B` (`f8c3e2c0`), & `NVlabs/Nemotron-Labs-Diffusion` (`2aabcf75`), reconciled with **`0 B` diff** across all `13` LFS safetensors files; **NLD-Uni:** [`arXiv:2610.36466v1`](https://arxiv.org/abs/2610.36466), `nvidia/NLD-Uni` (`5db755a2`), & `NVlabs/Nemotron-Labs-Diffusion-Uni` (`16916949`), reconciled with **`0 B` diff** across all `8` safetensors shards (`1,557` tensors = `18,267,552,262` params = `36,535,104,542 B` payload). **Zero production weights, tokenizers, or `pca_tree.pt` binaries are stored or redistributed.**
> 2. **Analytically Derived Quantities:** Exact parameter inventories (`63.19B` / `126.37 GB` TwoTower; `13,506,073,600` backbone + `47,185,920` LoRA Diffusion-14B; `18,267,552,262` params / `36.535 GB` / `34.026 GiB` NLD-Uni across 6 execution views), dual SI (`GB`/`MB`) & IEC (`GiB`/`MiB`) byte conversions, context-length honesty tiers, and memory boundaries (explicitly excluding peak runtime inference workspace) are computed analytically from pinned configs.
> 3. **Synthetic Miniature Browser Execution:** All live in-browser forward passes run deterministic clean-room miniature Float32/Float64 simulators (**Nano:** `d = 8`, `V = 32`, `299,192` floats; **TwoTower:** `d = 8`, `V = 34`, `600,120` floats; **Diffusion:** `d = 8`, `V = 36`, `271,304` floats [`229,064` backbone + `42,240` LoRA]; **NLD-Uni:** `d = 8`, `V_txt = 64` [`+17 <LOC_0..16>` bins], `V_vq = 256` [`D = 8` PCA ball-tree], `766` tensors = `191,235` floats = `764,940 B` + `3,068 B` tree = `768,008 B`). Note Diffusion and NLD-Uni's explicit **projection-width reversal** (`8 → 256/64` in Diffusion and `8 → 64/16` in NLD-Uni so `headDim = 8` retains all 4 dyadic YaRN rotary planes).

---

## 1. Nemotron-Labs-Diffusion-Uni (`/uni/` & `/llm-viz/uni/`)

Interactive editorial WebGL2 explorer for NVIDIA's unified multimodal `18.27B` **Elastic Mixture-of-Transformers (Elastic-MoT)** discrete diffusion model (`arXiv:2610.36466v1`, `nvidia/NLD-Uni` @ `5db755a2`, `NVlabs/Nemotron-Labs-Diffusion-Uni` @ `16916949`), verified clean-room against independent **PyTorch `2.15.0`** reference fixtures:

- **Twin 34-Layer Elastic-MoT Backbone (`18` Joint `L0..17` + `16` Split `L18..33`)**: Parallel Understanding (`U`, mineral indigo) and Generation (`G`, emerald-teal) towers with `0` shared learned weights across all `1,557` checkpoint tensors (sharing only the non-parametric 1D YaRN `inv_freq` buffer), paired with continuous `RADIO-H` ViT + `2×2` SquaredReLU projector and discrete `Emu3` IBQ VQ-VAE + `2×2` `PackedLinearImageCodec`.
- **Seven Interactive WebGL2 Spatial Views + Dedicated 2D RGB Canvas Insets (`D4`)**:
  1. **`Macro 34L`** — Twin `U`/`G` 34-layer spines split at the `L17 | L18` boundary (`dual_tower_layers = 18`, `U5`), task-reactive tower dimming across 6 execution views (`8.50B` Text, `9.19B` VLM/Ground, `13.36B` T2I, `14.24B` Edit, `18.27B` Interleaved `U7`), and live `D5` split-layer `U` prefix KV waste slabs (`0 B` `RG32F` atlas upload).
  2. **`Visual Input`** — Side-by-side continuous `RADIO-H` (`16×16` native / `16×8` panorama + literal `"\n"` row delimiter vs. dead `encoder.image_newline`, `U8`) and discrete `Emu3` IBQ (`argmax(z @ E^T)`, `U14`) + `2×2` `PackedLinearImageCodec` (`U13`), with `U18` `8/16` border-lock mask hatching on dedicated 2D `<canvas>` RGB insets.
  3. **`Joint / Split`** — Layer-selectable (`L0..33`) routed `U`/`G` `4:1` GQA + `3.5×` SwiGLU cutaway exposing live `RG32F` G-channel masking (`U25`) across the `L17` Joint → `L18` Split (`U ⊥ G`) boundary, 1D dyadic YaRN RoPE (`[1, 0.28125, 0.03125, 0.015625]`) + Llama-4 query temperature scale (`1.000 → 1.069` at `pos >= 16`), additive 2D `SinCos-G` coordinates (`marker_column = !isSourceEdit`), and `D5` dead prefix-KV slices.
  4. **`Ball-Tree`** — 3D PCA projection of all `256` Emu3 VQ codebook vectors with nested depth-8 `BalancedSemanticTree` shells (`N = 4` funnel `256 → 64 → 16 → 4 → 1`, `N = 1` exact binary MDM reduction, `U11`) and the live **`U26` Production `D = 17` Floor-Rounding Mismatch** panel contrasting training `[13, 9, 5]` vs. inference `[12, 8, 4]` (`2×` wider subtrees at `3/3` internal stages for `K = 4`; `0` drift in miniature `D = 8`).
  5. **`Refinement`** — Live `4×6` marked VQ child canvas (`4×4` content + `2` row/end marker columns) illustrating reversed inference age polarity (`U9`), additive mask noise `E_gen[u] + η · E_gen[MASK]` with zero AdaLN (`U10`, `U17`), deterministic `stratified_random(4, 42)` quadtree ordering with `shift = 5.0` quotas and `N - 1` flush rounds (`U12`, `U23`), and `2×2` parent codec brackets.
  6. **`Output & Edit`** — Task-reactive stage contrasting `3`-branch Dual-CFG Image Editing (`legacy` vs. `independent` vs. `ip2p`, `U16`, `U18` border-lock verification & `16×8` panorama crop-back, `U30`), FLARE `B = 4` shifted-logit Visual Understanding (`U19–U21`), and `<LOC_0..16>` 2D Object Grounding overlays exposing the `U27` `skip_special_tokens=True` silent box-erasure hazard.
  7. **`OPD (Read-Only)`** — Strictly quarantined forward-only diagnostic explorer badged **`READ-ONLY · NO GRADIENTS`** (zero backward passes, zero optimizer steps, zero teacher weights): inspects transitive `129`-bucket `tail_kl` AR→Diff distillation (`U24`), continuous-flow `matched_sigmas(0.9)` ODE re-quantization, and all 3 verified `minimal_training/` trainer bugs (`U28`: `~6.915×` / `1,000×` SFT double-norm, accidental `<|im_end|>` supervision, and `<LOC_*>` tail leakage).
- **Live `D5` Elastic Prefill Toggle, Timeline Dock, 3-Tab Inspector & 5-Act / 28-Beat Tour**: Toggle `HF-Faithful` (`34L U`, `47.1%` T2I / `31.4%` Edit prefix KV waste) vs. `Elastic-Pruned` (`18L U`, `0` wasted KV, `0`-ULP Stage-2 equivalence, `U6`) live in Views 1 & 3; scrub/fork multi-task branches within the `<= 12.00 MiB` `Set<ArrayBufferLike>` session budget; inspect `Inspect | Architecture | Ledger U1–U30` (with locked decisions `D1–D5` and 4-tier context honesty `U3`); and run the isolated 5-Act / 28-Beat Guided Tour with exact Free-Explore restoration on exit.

---

## 2. Nemotron-Labs-Diffusion-14B (`/diffusion/` & `/llm-viz/diffusion/`)

Interactive editorial WebGL2 explorer for NVIDIA's 40-layer homogeneous Ministral 3 dense decoder (`nvidia/Nemotron-Labs-Diffusion-14B` @ `f8c3e2c0`), verified against independent **PyTorch `2.15.0`** reference fixtures across Categories 1–12:

- **Seven Interactive WebGL2 Spatial Views**:
  1. **`Macro 40L`** — Single physical 40-layer dense spine (`encoder.layers.0..39`), `0..31` shared KV slab rail (`20.00 KiB/tok` mini / `160.00 KiB/tok` 14B), and detachable draft-only `o_proj` LoRA bridge (`80` FP32 tensors).
  2. **`RoPE GQA`** — `32Q / 8KV` (4:1) grouped-query attention with exact dyadic YaRN RoPE (`inv_freq = [1.0, 0.3125, 0.0625, 0.03125]`, `truncate = True`), live **2D Unit-Circle RoPE Phasor SVG Inset** with `p = 16` Llama-4 query scale scrubber (`1.0000× → 1.0693×`), and draft-only `+4.0 · B(A · a)` `o_proj` LoRA delta (`256×256` `RG32F` atlas, `512 KiB`, `0` overlaps).
  3. **`SwiGLU`** — Post-attention RMSNorm → parallel `W_gate → SiLU` & `W_up` → Hadamard `⊙` → `W_down` → residual add (`64×64` `RG32F` atlas, `74.53%` of 14B backbone parameters) with progressive `Accum Reveal` slider.
  4. **`Masks` (5-Tab Attention Mask Explorer)** — Live `RG32F` G-channel bound plate comparing **`Joint [x_t‖x_0]`** (`M_BD`, `M_OBC` `Δp >= 1`, `M_BC` + `seq_idx` vs. `ropePositionIds` vs. `cachePositions`), **`AR Causal`**, **`Block Bidir`**, **`Commit / Verify`**, and read-only **`VLM Quadratic SS`** (`D10`, `D24`).
  5. **`Spec & KV`** — Shifted draft-vs-causal-verify alignment connectors, `0..31` KV cache rollback (`_crop_dynamic_cache`), and **`D19` dual-crop provenance** (`Browser Post-EOS Crop` vs. `Upstream Pre-EOS Crop`).
  6. **`Embedding`** — Untied `V = 36` `encoder.embed_tokens` (`[MASK]`, `<|im_end|>`, `<s>`, `</think>`) with zero additive positional table.
  7. **`Output Head`** — Final `encoder.norm` → untied `diffusion_head` contrasting causal `+1` vs. diffusion `+0` predictions and un-suppressed `[MASK]` softmax (`D18`).
- **Four Live Execution Paths + Re-Prefill Guard**: **`AR`**, **`DLM Causal`** (AR-seeded slot 0, `D7`), **`DLM Bidir`** (guarded by `KvProvenanceMismatchError` re-prefill confirmation banner), **`Linear Spec`**, and **`Spec + LoRA`** (`o_proj` LoRA ON on draft only; `0`-ULP OFF on verify, `D17`).
- **Unified Tri-Mode Timeline, 3-Tab Inspector & 4-Act / 26-Beat Tour**: Copy-on-step auto-forking timeline (`<= 12.00 MiB` hard cap; `4.86 MiB` under 12-branch/128-record stress), 3-tab Inspector (`Inspect` with `D12` 3-counter NFE · `Architecture` with 4-tier context scaling & 13-file `0 B` LFS inventory · searchable `Ledger D1–D24`), and an isolated 26-beat Guided Tour restoring Free-Explore state identically on exit.

---

## 3. Nemotron-Labs-TwoTower 30B-A3B (`/twotower/` & `/llm-viz/twotower/`)

Interactive WebGL2 explorer for NVIDIA's dual-tower **Autoregressive Context + Block-Diffusion Denoiser** hybrid (`NemotronHTwoTowerForCausalLM`), verified against independent **PyTorch `2.15.0`** reference fixtures across all 8 verification categories:

- **Seven Spatial Views**: `Macro` (`52 × 2` dual towers + `29` aligned read-only linkers vs. `23` stateless MoE bays), `GQA` (`128×128` `RG32F` atlas, `[B × (P+B)]` bidirectional attention), `Mamba` (`256×128` `RG32F` atlas, 4-tap Conv1D `3→2→1→0` replacement, `4×4` analytical SSM decomposition), `Timestep` (`t = nMasked / B`, split `adaLN-single` `Mod→Norm` vs. `Norm→Mod`), `MoE` (`128+1` FP64 router), `Embed` (`V = 34`), and `Output` (constrained MDLM/SUBS, zero-`Infinity`).
- **Three Execution Modes (`Mask-Diffusion`, `AR`, `Mock-AR`)**, **Persistent Nested Timeline & Branching**, **Advanced MDLM Sampler Inspector** (`0.22` mini / `0.80` paper / `0.90` ref presets, decoupled `B`/`K`, deterministic Gumbel-max), **4-Act / 28-Beat Guided Walkthrough (`T`)**, **Production `Architecture` Tab** (`63.19B` / `126.37 GB` / `117.69 GiB`, 4-way memory separation, 3-regime scaling), and **Developer Workbench (`W`)**.

---

## 4. Nemotron 3 Nano 30B-A3B (`/` & `/llm-viz/` · `v1.0.0` Preserved)

The finalized single-tower **`v1.0.0`** visualizer remains completely intact and bundle-isolated at `/`:

- **52-Layer Hybrid Vertical Spine** (`23 Mamba-2 · 23 MoE 128+1 · 6 GQA`, 4 NVIDIA Figure 2 motif spans, continuous `2,688`-channel emerald residual stream, untied `131,072`-vocab bookends).
- **Unified 52-Layer Float32 Miniature Simulator (`d = 8, V_mini = 32`)** verified against independent PyTorch `2.15.0` fixtures (`SHA-256` weight proof over `299,192` floats), with cached incremental AR decoding (`0`-ULP equivalence) & copy-on-step branching.
- **Five Exploded 3D Cutaways** (**Embedding NoPE**, **Mamba-2**, **Granular MoE 128+1**, **GQA 32Q/2KV**, **Output / LM Head**), **22-Beat / 3-Act Guided Walkthrough**, and **Selective FP8 Mode + `1 → 1M` Memory Scaling Chart**.

---

## Miniature Simulator vs. Production Metadata Summary

| Dimension / Property | Nano (`v1.0.0` · Mini / Prod) | TwoTower (`rev 35b64985` · Mini / Prod) | Diffusion-14B (`rev f8c3e2c0` · Mini / Prod) | NLD-Uni (`rev 5db755a2` · Mini / Prod) |
| --- | --- | --- | --- | --- |
| **Topology** | `1 × 52L` hybrid (`23M · 23E · 6*`) | `2 × 52L` independent (`crossTowerWeightSharing: false`) | `1 × 40L` homogeneous Ministral 3 dense decoder | `2 × 34L` Elastic-MoT (`18` Joint `L0..17` + `16` Split `L18..33`) + RADIO-H + Emu3 IBQ + `2×2` Codec |
| **Weights & Payload** | `299,192` F32 / `31.6B` (`3.2B`/`3.6B` active) | `600,120` F32 (`2.29 MiB`) / `63.19B` · `126.37 GB` (`117.69 GiB`) | `229,064` + `42,240` F32 / `13.51B` (`25.16 GiB`) + `47.19M` LoRA (`180.00 MiB`) | `766` tensors · `191,235` F32 (`764,940 B` + `3,068 B` tree) / `1,557` tensors · `18.27B` · `36.535 GB` (`34.026 GiB`) |
| **Hidden Width (`d`)** | `8` / `2,688` | `8` / `2,688` | `8` (`dFf=24`) / `5,120` (`dFf=16,384`) | `8` (`dFf=28`, `3.5×`) / `4,096` (`dFf=14,336`, `3.5×`) |
| **GQA & Projection** | `32Q/2KV` (`dHead=4` / `128`) | `32Q/2KV` (`dHead=4` / `128`), `[B×(P+B)]` | `32Q/8KV` (`dHead=8` mini expands `8→256` vs. `128` 14B contracts `5120→4096`) | `8Q/2KV` mini (`dHead=8`, expands `8→64/16`) / `32Q/8KV` prod (`dHead=128`, `4096→4096/1024`) |
| **Position / Scaling** | NoPE / `1M` analytical | NoPE / `≤128K` op · `256K` cfg · `>256K` extrap | Dyadic YaRN (`truncate=True`) + Llama-4 `β=0.1` (`p=16` / `16,384` · `max_pos=256K`) | 1D Dyadic YaRN (`factor=8`, `p=16` / `16,384` · `max_pos=128K`, `U3`) + additive 2D `SinCos-G` (`scale=0.0018`) |
| **Vocabulary (`V`)** | `32` / `131,072` | `34` (`32 [MASK]`, `33 <eos>`) / `131,072` (`MASK=3`) | `36` (`32 [MASK]`, `33 <|im_end|>`, `34 <s>`, `35 </think>`) / `131,072` (`MASK=100`) | `V_txt=64` (`+17 <LOC_0..16>`, `MASK=25`) + `V_vq=256` (`D=8`) / `132,101` (`+1,029` loc/box, `U4`) + `131,072` (`D=17`) |

---

## Controls & Keyboard Shortcuts

Pointer controls on all four 3D stages: **Left-drag** to orbit · **Right-drag / Shift-drag** to pan · **Scroll / Pinch** to zoom · **Click / Tap** any bay, slab, codebook node, or matrix cell to inspect. Global shortcuts automatically defer when editable form inputs or interactive controls have focus.

- **NLD-Uni (`/uni/`):** `1`–`7` (switch all 7 spatial views) · `T` (launch/exit 28-beat Guided Tour) · `I` (toggle 3-tab Inspector Drawer) · `R` (fit/reset camera) · `←` / `→` / `Home` / `End` (navigate focused View, Task, or Inspector tablists with roving `tabindex`; `←` / `→` also step Tour beats outside inputs) · `Esc` (exit Guided Tour outside inputs, deferring inside `Ledger` search).
- **Diffusion-14B (`/diffusion/`):** Double-click layer bay in `Macro 40L` to open `RoPE GQA` · `←` / `→` / `Home` / `End` (navigate focused View, Mask Explorer, or Inspector tablists with roving `tabindex`; `←` / `→` also step Tour beats outside inputs/tablists) · `Esc` (close Inspector Drawer → clear pinned cell → exit Guided Tour). See [`docs/DIFFUSION_USER_GUIDE.md`](docs/DIFFUSION_USER_GUIDE.md) for full UI controls.
- **TwoTower (`/twotower/`):** `T` (Guided Tour) · `←` / `→` (Tour beats or tablists with `Home`/`End`) · `Space` (Tour autoplay) · `R` (fit/reset camera) · `W` (Workbench drawer) · `Esc` (close/exit). See [`docs/TWOTOWER_USER_GUIDE.md`](docs/TWOTOWER_USER_GUIDE.md).
- **Nano (`/`):** `1`–`6` (Tower & 5 cutaways) · `Space` / `Shift+Space` (Tour or stepper) · `←`/`→`/`↑`/`↓` · `Enter` (zoom bay) · `P` (`BF16`/`FP8 Selective`) · `M` (`1 → 1M` chart) · `R` (reset) · `?` (Shortcuts) · `Esc` (close/back).

---

## Local Development, Build & Verification

**Supported Node.js versions:** `Node.js 22.12+ (LTS)`, `24.x (LTS)`, or `26+` (`^22.12.0 || ^24.0.0 || >=26.0.0`, matching `vitest@5.1.1`; CI uses Node `22`).

```bash
# 1. Install exact lockfile dependencies (0 runtime dependencies)
npm ci

# 2. Start Vite dev server
#    Nano:      http://localhost:3000/
#    TwoTower:  http://localhost:3000/twotower/
#    Diffusion: http://localhost:3000/diffusion/
#    NLD-Uni:   http://localhost:3000/uni/
npm run dev

# 3. Strict TypeScript check
npm run typecheck

# 4. Run full Vitest unit, contract & PyTorch-parity suite across all 4 apps (627 tests / 75 files; zero Python needed)
npm test

# 5. Production build for root path (`/`, `/twotower/`, `/diffusion/`, `/uni/`) + 4-way isolation check
npm run build
npm run verify:isolation

# 6. Production build for GitHub Pages subpath (`/llm-viz/`, `/llm-viz/twotower/`, `/llm-viz/diffusion/`, `/llm-viz/uni/`)
VITE_BASE_PATH=/llm-viz/ npm run build
npm run verify:isolation

# 7. Run Playwright Chromium WebGL2 E2E smoke suite across all 4 apps & both base paths (34 tests / 4 files)
#    (first time: npx playwright install chromium)
npm run test:e2e
```

**Measured 4-Way Production Bundle Sizes (`npm run verify:isolation`, enforced on both `/` and `/llm-viz/`):**

- **Nano transitive JS+CSS (`4` JS, `1` CSS):** **`209.35 KiB`** (`214,375 B`) gzip (`765,603 B` raw) vs. `215.00 KiB` (`220,160 B`) hard cap (**`+5.65 KiB` / `+5,785 B` headroom**, `0` leaks across all 12 directional pairs).
- **TwoTower transitive JS+CSS (`3` JS, `1` CSS):** **`318.40 KiB`** (`326,045 B` root) / **`318.41 KiB`** (`326,056 B` `/llm-viz/`) gzip (`1,149,920 B` / `1,149,928 B` raw, `22` `src/twotower/*` sources, `0` leaks).
- **Diffusion transitive JS+CSS (`4` JS, `1` CSS):** **`294.82 KiB`** (`301,892 B` root / `301,900 B` `/llm-viz/`) gzip (`1,025,690 B` / `1,025,698 B` raw, `31` `src/diffusion/*` sources, `0` leaks).
- **NLD-Uni transitive JS+CSS (`2` JS, `1` CSS):** **`188.96 KiB`** (`193,495 B` root) / **`188.97 KiB`** (`193,502 B` `/llm-viz/`) gzip (`661,685 B` / `661,693 B` raw, `37` `src/uni/*` sources, `0` leaks, Locked Decision **`D4`** **`0`** `src/ui/controls.ts` sourcemap references).
- **Shared Chunks (`3` JS):** `assets/atlasPacker-*.js` (`shared(all4)`, `50,302 B` gzip) + `assets/controls-*.js` (`shared(n+df)`, `54,829 B` gzip) + `assets/overlay-*.js` (`shared(n+tt+df)`, `80,754 B` gzip) = **`181.53 KiB`** (`185,885 B` gzip / `634,667 B` raw).

---

## Documentation

- [**Diffusion-14B User Guide (`docs/DIFFUSION_USER_GUIDE.md`)**](docs/DIFFUSION_USER_GUIDE.md) — Walkthrough of all 7 spatial views, 4 execution modes + re-prefill guard, 2D RoPE phasor & `p = 16` scrubber, 5-tab Mask Explorer, Unified Timeline, 3-tab Inspector (`Inspect | Architecture | Ledger D1–D24`), 4-act / 26-beat Guided Tour, and shortcuts.
- [**Diffusion-14B Architecture & Audit Guide (`docs/DIFFUSION_ARCHITECTURE.md`)**](docs/DIFFUSION_ARCHITECTURE.md) — Pinned provenance (`f8c3e2c0`), 3-tier epistemic disclosure, 40L Ministral 3 dense decoder + projection-width reversal disclosure, exact dyadic YaRN RoPE + Llama-4 scaling, 4-path engine & Sampler A vs. B separation, `<= 12.00 MiB` shared-slab KV proof, 4-tier context scaling, 13-file `0 B` LFS inventory, `D1–D24` ledger summary, and Topology A contract.
- [**TwoTower User Guide (`docs/TWOTOWER_USER_GUIDE.md`)**](docs/TWOTOWER_USER_GUIDE.md) — Practical walkthrough of all 7 TwoTower views, 3 execution modes, MDLM Sampler, 4-Act Guided Tour, 3-regime scaling, Workbench (`W`), and shortcuts.
- [**TwoTower Architecture & Checkpoint Audit (`docs/TWOTOWER_ARCHITECTURE.md`)**](docs/TWOTOWER_ARCHITECTURE.md) — Pinned provenance (`35b64985`), dual `GB`/`GiB` reconciliation, 3-regime context scaling, 4-way memory boundary, and 9-entry discrepancy ledger.
- [**Nano Architecture Guide (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md) — Single-tower Nano module boundaries, 5-pass WebGL2 pipeline, context-loss recovery, runtime performance architecture, and shared Topology A seams.
- [**Verification & Reproducibility Runbook (`docs/VERIFICATION.md`)**](docs/VERIFICATION.md) — Complete Vitest unit & independent PyTorch `2.15.0` parity suites, Topology A isolation gate, Playwright Chromium WebGL2 E2E (SwiftShader & hardware Metal), CI/Pages workflows, and static tarball checksum steps.
- [**TwoTower v1.0.0 Release Notes (`docs/RELEASE_NOTES_TWOTOWER_v1.0.0.md`)**](docs/RELEASE_NOTES_TWOTOWER_v1.0.0.md) — Release summary, live public URLs, verification snapshot, discrepancy ledger, and known limitations for `twotower-v1.0.0`.
- [**Third-Party Notices (`THIRD_PARTY_NOTICES.md`)**](THIRD_PARTY_NOTICES.md) — Verified upstream licenses, architectural citations, non-redistribution statement, and exact dev-only dependency inventory.
- [**Changelog (`CHANGELOG.md`)**](CHANGELOG.md) — Release history for Nano `v1.0.0`, TwoTower `twotower-v1.0.0`, Diffusion-14B `diffusion-v1.0.0`, and NLD-Uni `uni-v1.0.0`.
- **Historical Milestone 1 Briefs (superseded):** [`docs/archive/DESIGN_BRIEF.md`](docs/archive/DESIGN_BRIEF.md) · [`docs/archive/IMPLEMENTATION_PLAN.md`](docs/archive/IMPLEMENTATION_PLAN.md).

---

## Attribution & License

1. **Repository Heritage & Visual Language (`bbycroft/llm-viz`, MIT License):** Forks Brendan Bycroft's [`bbycroft/llm-viz`](https://github.com/bbycroft/llm-viz) (`Copyright (c) 2023-2026 Brendan Bycroft`, preserved verbatim in [`LICENSE`](LICENSE)) and builds on its interactive architectural idiom. All Nano, TwoTower, Diffusion, and NLD-Uni browser modules (`src/`), tests (`tests/`), isolation checks (`scripts/verify_bundle_isolation.mjs`), and offline parity scripts (`scripts/reference/`) were newly authored (`Copyright (c) 2026 Rohan Master`).
2. **Referenced Architectural Specifications (Not Redistributed Artifacts):**
   - **Nemotron-Labs-Diffusion-Uni (NLD-Uni):** [`arXiv:2610.36466v1`](https://arxiv.org/abs/2610.36466), Hugging Face [`nvidia/NLD-Uni`](https://huggingface.co/nvidia/NLD-Uni/tree/5db755a2eb0ab84654e3dcf80c7d0245bfe36a50) (`5db755a2`), and [`NVlabs/Nemotron-Labs-Diffusion-Uni`](https://github.com/NVlabs/Nemotron-Labs-Diffusion-Uni/tree/16916949402ca738dcc74a7a37e0c43facafe347) (`16916949`; 100% clean-room equations & TypeScript/PyTorch parity; zero remote source lines, weights, tokenizer payloads, or `pca_tree.pt` binaries copied or redistributed).
   - **Nemotron-Labs-Diffusion-14B:** [`arXiv:2607.05722v1`](https://arxiv.org/abs/2607.05722), Hugging Face [`nvidia/Nemotron-Labs-Diffusion-14B`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B/tree/f8c3e2c078e193599b8882d965b1001c456ba738) (`f8c3e2c0`), and [`NVlabs/Nemotron-Labs-Diffusion`](https://github.com/NVlabs/Nemotron-Labs-Diffusion/tree/2aabcf75ecf70264d4047ff090d842450853f2e4) (`2aabcf75`; clean-room Ministral 3 / YaRN / Llama-4 / PEFT `o_proj` LoRA equations).
   - **Nemotron-Labs-TwoTower 30B-A3B:** [`arXiv:2606.26493v2`](https://arxiv.org/abs/2606.26493) and Hugging Face [`nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16/tree/35b6498508be8856d380cdfe4a900a79b918d1ea) (`35b64985`).
   - **Nemotron 3 Nano 30B-A3B:** [`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848) and Hugging Face [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/tree/bf77c3174f68ad409e1c2aa60daeb46e32d1c606) (`bf77c317`).
   - **No NVIDIA model weights, checkpoints, safetensors, LoRA adapters, `pca_tree.pt` binaries, tokenizers, or NVIDIA license-restricted artifacts are redistributed in this repository or in `dist/`.** NVIDIA and Nemotron are trademarks of NVIDIA Corporation; this project is an independent educational visualization not affiliated with, sponsored by, or endorsed by NVIDIA Corporation.
3. **Zero Browser Runtime Dependencies:** Offline parity generation uses pinned PyTorch `2.15.0` (BSD-3-Clause, `scripts/reference/`, absent from `dist/`), and build/test tooling uses 5 exact-pinned npm `devDependencies` (`@playwright/test@1.65.1`, `@types/node@22.20.5`, `typescript@5.9.3`, `vite@8.3.8`, `vitest@5.1.1`).

Licensed under the **MIT License** — see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
