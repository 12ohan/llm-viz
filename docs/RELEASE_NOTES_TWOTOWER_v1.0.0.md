# Draft Release Notes — `twotower-v1.0.0` (Nemotron-Labs-TwoTower 30B-A3B Visualizer)

> [!NOTE]
> **Pre-Release Candidate Draft** — These are draft release notes prepared for the reserved git tag **`twotower-v1.0.0`** on branch `feat/twotower`. No git tag has been created yet, and production GitHub Pages remains locked to `main` until merge and final sign-off.

---

## Planned Public Demo URLs (Upon Merge to `main`)

- **Nemotron-Labs-TwoTower 30B-A3B (`/twotower/`):** <https://12ohan.github.io/llm-viz/twotower/>
- **Nemotron 3 Nano 30B-A3B (`/` · `v1.0.0` preserved):** <https://12ohan.github.io/llm-viz/>

Both applications ship together as a zero-runtime-dependency **Topology A multi-entry Vite MPA** connected by quiet, base-path-aware reciprocal navigation (`TwoTower ↗` / `Nano ↗`), while preserving Nano `v1.0.0`'s strict bundle isolation (`<= 215.00 KiB` gzip hard budget, zero TwoTower sourcemap or marker leakage).

---

## Principal Features in `twotower-v1.0.0`

1. **Clean-Room Dual-Tower Engine (`NemotronHTwoTowerForCausalLM`)**:
   - Models two independent `52`-layer hybrid towers (`23 Mamba-2 · 23 Granular MoE 128+1 · 6 GQA Attention`, `crossTowerWeightSharing: false`, `tie_word_embeddings: false`) grounded in [`arXiv:2606.26493v2`](https://arxiv.org/abs/2606.26493) and pinned Hugging Face revision `35b6498508be8856d380cdfe4a900a79b918d1ea`.
   - Enforces canonical **`t = nMasked / B`** timestep semantics (never `r / K`), COS→SIN sinusoidal embedding, exact split PixArt-α `adaLN-single` ordering (`modulate → RMSNorm` on `M`/`*` vs. `RMSNorm → modulate` on `E`, unmodulated final `norm_f`), `FP64` MoE router matmul (`--moe-router-dtype fp64`), and unshifted MDLM/SUBS confidence unmasking with deterministic Gumbel-max (`T > 0`).
   - Supports all three execution modes — **`Mask-Diffusion`**, **`AR`**, and **`Mock-AR`** (`KV[:-1]` + `mamba_s2`, unmodulated, `0`-ULP parity with Context AR when `ΔW = 0`) — over immutable, digest-verified Context boundaries (`0`-ULP invariance across Denoiser refinements).
2. **Seven Interactive WebGL2 Spatial Views**:
   - **`Macro`** (`52 × 2` dual towers, `29` aligned transfer linkers vs. `23` stateless MoE bays, instructional Macro Flow Replay, 10-axis picking), **`GQA`** (`35/65` paired view, `128×128` `RG32F` atlas, bidirectional `[B × (P+B)]`), **`Mamba`** (`35/65` paired view, `256×128` `RG32F` atlas, 4-tap Conv1D boundary replacement `3→2→1→0`, native `4×4` analytical SSM decomposition), **`Timestep`** (split `adaLN-single` + bypass state), **`MoE`** (`128+1` stateless FP64 6th-vs-7th margin + non-spoiling partial `Σσ`), **`Embed`** (independent untied NoPE `V = 34`), and **`Output`** (unmodulated `norm_f` + zero-`Infinity` MDLM probabilities).
   - Surfaces explicit **Historical Denoiser Boundary Mismatch** banners (`P_cond` vs. current `P`) and non-spoiling partial-accumulation formulas across GQA, Mamba-2, and MoE steppers.
3. **Persistent Nested Timeline, Advanced MDLM Sampler & Developer Workbench**:
   - Copy-on-step `TwoTowerSession` with clickable nested milestone timeline, non-destructive scrubbing, auto-forking from rewound history, and active-branch-safe branch deletion.
   - Advanced Sampler inspector with `0.22` (mini default), `0.80` (paper), and `0.90` (reference code) threshold presets, decoupled `B ∈ {4, 8}` and `K ∈ [1..12]`, explicit non-retroactive config scope badges (`Next Refinement` / `Next Block` / `Next Step` / `Session Restart`), and unscaled calibration caveats.
   - Collapsible, bidirectionally synced Developer Diagnostic Workbench (`W`).
4. **Isolated 4-Act / 28-Beat Guided Educational Walkthrough (`T`)**:
   - Self-contained tour spanning Dual-Tower Architecture & Modes, Conditioning & Paired Layers, Output & MDLM Sampler, and Context Commitment & Production Scaling, restoring the user's Free Explore session untouched on exit.
5. **Production Architecture & 3-Regime Context-Length Scaling Readouts**:
   - Dual SI (`GB`/`MB`) & IEC (`GiB`/`MiB`) checkpoint inventory reconciled to `0 B` (`63,185,902,848` params · `126,371,817,472 B` = `126.37 GB` / `117.69 GiB`, including `46` FP32 `e_score_correction_bias` tensors = `+11,776 B` extra over BF16 and `58` time-conditioning tensors = `30.02M` params / `60.04 MB` / `57.26 MiB`).
   - Explicit **4-way memory separation** (Persistent Context Cache vs. Optional `mamba_s2`/Cross-GPU Mirrors vs. Transient Denoiser Block KV vs. Explicitly Excluded Peak Inference Runtime Workspace) and **3-regime context scaling** (`≤ 128K` Model-Card Operational · `128K–256K` Configured Positional Limit [not empirically validated runtime support] · `> 256K` Analytical Extrapolation).

---

## Verification Summary

| Verification Gate | Result / Coverage |
| --- | --- |
| **TypeScript (`npm run typecheck`)** | `0` errors across Nano & TwoTower (`typescript@5.9.3`). |
| **Vitest Unit, Contract & Parity (`npm test`)** | **`365+` tests across `45` files** (`236` Nano / `33` files + `129+` TwoTower / `12` files). |
| **Independent PyTorch `2.15.0` Parity** | `6` checked-in fixtures (`3` Nano + `3` TwoTower covering all **8 mandatory TwoTower categories** with SHA-256 weight proofs, `0`-ULP Context immutability, and `0`-ULP identity AR vs. Mock-AR parity). |
| **Topology A Bundle Isolation (`npm run verify:isolation`)** | **7/7 hard gates PASS** on both `/` and `/llm-viz/` builds (`0` sourcemap leaks, `0` marker hits; Nano gzip `207.53 KiB` / `212,508 B` vs. `215.00 KiB` budget; `127.27 KiB` shared across `31` Nano primitives; `~311 KiB` TwoTower gzip). Enforced in both `.github/workflows/ci.yml` and `.github/workflows/deploy.yml`. |
| **Playwright Chromium WebGL2 E2E (`npm run test:e2e`)** | **`20+` E2E tests** (`9` Nano in `smoke.spec.ts` with test-only camera-fly settlement hardening + `11+` TwoTower in `twotower-smoke.spec.ts`) across `/` and `/llm-viz/`. |
| **Dependencies & Security Hygiene** | **`0` browser runtime dependencies** (`73/73` lockfile entries `dev: true`), `npm audit` **`0` vulnerabilities**, `0` local path / credential / weight leaks. |

---

## 3-Tier Epistemic Sourcing & Discrepancy Ledger Highlights

- **Tier 1 (Directly Inspected Metadata):** Pinned HF `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16` revision `35b6498508be8856d380cdfe4a900a79b918d1ea` (`config.json`, `generation_config.json`, `tokenizer_config.json`, `model.safetensors.index.json`) plus a read-only offline 24-shard safetensors header audit (`0` integrity errors; zero weights stored or redistributed).
- **Tier 2 (Analytically Derived Quantities):** Exact parameter totals, `0 B` payload reconciliation, dual SI/IEC conversions, 3-regime context scaling, and 4-way memory boundaries.
- **Tier 3 (Synthetic Miniature Execution):** Deterministic synthetic Float32/Float64 miniature model (`d = 8`, `V = 34`, `860` tensors / `600,120` floats).
- **9 Verified Paper-vs-Checkpoint Discrepancies Documented in UI & Docs:** `adaLN-single` `1.5M` claimed vs. **`30.02M`** (`30,022,272`) actual; `"2 shared experts"` vs. `1` double-wide (`3,712`); `~60B` vs. **`63.19B`** (`126.37 GB` / `117.69 GiB`); `128K` model card vs. `256K` `config.json` `max_position_embeddings`; `γ = 0.8` paper vs. `0.9` reference code (`T = 0.0` vs. `0.1`); split `Mod→Norm` (`M`/`*`) vs. `Norm→Mod` (`E`); `FP64` MoE router matmul + `46` `FP32` biases; repurposed `[INST]` (ID `3`) `[MASK]` & EOS `[2, 11]`; and Mamba `chunk_size = 128` inference vs. `S = 16` training (`~2.1T` vs. `~1.4T` §2.3).

---

## Known Limitations & Intentional Deferrals

- **Miniature Synthetic Scale Only:** Browser simulation runs `d = 8, V = 34` synthetic weights — no real `63.19B` checkpoint inference, WASM tokenizer, or WebGPU compute shaders.
- **Peak Inference Runtime Workspace Excluded:** Logical weight and cache readouts explicitly exclude intermediate activations, `[B × 131,072]` logits, CUDA/framework scratchpads, and allocator fragmentation.
- **Unscaled Sampler Confidence:** Softmax probability `x_θ` is a relative heuristic for block unmasking order, not calibrated semantic correctness.
- **Non-Official Community NVFP4 Note:** Referenced solely as a clearly labeled community quantization benchmark (~`42 GiB`), separate from NVIDIA's official `126.37 GB` (`117.69 GiB`) BF16 release.
- **Hardware GPU Visual Check:** Automated CI executes ANGLE/SwiftShader WebGL2; native hardware GPU verification (macOS Metal / Windows D3D11) is documented in [`docs/VERIFICATION.md`](VERIFICATION.md).

---

## Reproducing Portable & GitHub Pages Static Release Artifacts

```bash
npm ci && npm run typecheck && npm test
mkdir -p test-results/release-artifacts

# Portable static root bundle (`/` + `/twotower/`)
VITE_BASE_PATH=/ npx vite build --outDir test-results/dist-portable
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-portable
tar -czf test-results/release-artifacts/nemotron-viz-twotower-v1.0.0-portable.tar.gz \
  -C test-results/dist-portable .

# GitHub Pages subpath bundle (`/llm-viz/` + `/llm-viz/twotower/`)
VITE_BASE_PATH=/llm-viz/ npx vite build --outDir test-results/dist-pages
node scripts/verify_bundle_isolation.mjs --dir test-results/dist-pages
tar -czf test-results/release-artifacts/nemotron-viz-twotower-v1.0.0-pages.tar.gz \
  -C test-results/dist-pages .

(cd test-results/release-artifacts && shasum -a 256 nemotron-viz-twotower-v1.0.0-*.tar.gz > SHA256SUMS.txt && shasum -a 256 -c SHA256SUMS.txt)
```

---

## Attribution & License

- **Visual Heritage:** Adapted from Brendan Bycroft's [`bbycroft/llm-viz`](https://github.com/bbycroft/llm-viz) (MIT License; copyright preserved verbatim in [`LICENSE`](../LICENSE)).
- **Architectural References:** NVIDIA [`arXiv:2606.26493v2`](https://arxiv.org/abs/2606.26493) & HF `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16` (`35b6498508be8856d380cdfe4a900a79b918d1ea`); NVIDIA [`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848) & HF `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (`bf77c3174f68ad409e1c2aa60daeb46e32d1c606`); PixArt-α (`arXiv:2310.00426`); MDLM/SUBS (`arXiv:2406.07524`). Clean-room implementation with **zero NVIDIA weights, checkpoints, tokenizers, or upstream source files redistributed**.
- **Non-Affiliation:** Independent educational visualization not affiliated with, sponsored by, or endorsed by NVIDIA Corporation. Licensed under the **MIT License** — see [`LICENSE`](../LICENSE) and [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
