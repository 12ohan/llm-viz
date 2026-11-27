# Third-Party Notices & Attribution

This document records the verified provenance, licenses, and applicable obligations for upstream projects, architectural references, and development tooling associated with all five interactive WebGL2 applications in this repository:

- **Nemotron 3 Nano 30B-A3B** (`/` · `v1.0.0`)
- **Nemotron-Labs-TwoTower 30B-A3B** (`/twotower/` · `twotower-v1.0.0`)
- **Nemotron-Labs-Diffusion-14B** (`/diffusion/` · `diffusion-v1.0.0`)
- **Nemotron-Labs-Diffusion-Uni (NLD-Uni)** (`/uni/` · `uni-v1.0.0`)
- **Nemotron-Labs-Dory1.0-4B (`Dory1.0-4B`)** (`/dory/` · `dory-v1.0.0`)

---

## 1. Incorporated / Forked Repository History & Visual Language

| Project | Author | License | Repository |
| --- | --- | --- | --- |
| `llm-viz` | Brendan Bycroft | MIT License | <https://github.com/bbycroft/llm-viz> |

- **Copyright Notice (preserved verbatim in [`LICENSE`](LICENSE)):**
  `Copyright (c) 2023-2026 Brendan Bycroft`
- **Fork Contributions:**
  `Copyright (c) 2026 Rohan Master`
- **Scope & Distinction:** This repository descends from `bbycroft/llm-viz` and adapts its interactive architectural walkthrough idiom and adaptive screen-space (`fwidth()`) cell-grid LOD principle (credited in `src/render/sheets.ts`). All Nemotron 3 Nano, Nemotron-Labs-TwoTower, Nemotron-Labs-Diffusion-14B, Nemotron-Labs-Diffusion-Uni, and Nemotron-Labs-Dory1.0-4B browser application modules (`src/`), unit/parity/E2E test suites (`tests/`), isolation gates (`scripts/verify_bundle_isolation.mjs`), and reference verification scripts (`scripts/reference/`) were newly authored; legacy `bbycroft/llm-viz` source files and `public/` assets were removed prior to `v1.0.0`.
- **Applicable Obligation:** Retain the upstream MIT copyright notice and permission notice in [`LICENSE`](LICENSE) and in copies or substantial portions of the Software.

---

## 2. Referenced Architectural Equations & Pinned Public Metadata (Not Redistributed Artifacts)

All five interactive explainers ground their layer topologies, tensor dimensions, memory formulas, and precision policies in publicly published research papers and Hugging Face model metadata. All mathematical forward passes are **clean-room first-principles implementations** — zero proprietary, AGPL, or upstream model source files are copied into this repository.

### A. Nemotron 3 Nano 30B-A3B (`/` Entry)

1. **Technical Report:** NVIDIA, *NVIDIA Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*, [`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848) (§2.1, Figure 2, Table 1, Table 3, §§4.2–4.3).
2. **Pinned Public Model Config:** Hugging Face [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/config.json`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/config.json) pinned at commit `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`.
3. **Reference Forward-Pass Equations:** Mathematical definitions from NVIDIA's [`modeling_nemotron_h.py`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/modeling_nemotron_h.py) (**Apache License 2.0**) — covering `NemotronHRMSNorm`, `NemotronHMamba2Mixer` + `MambaRMSNormGated` (`norm_before_gate=False`), `NemotronHMOE` + `NemotronHTopkRouter` + non-gated `ReLU²` `NemotronHMLP`, NoPE `32Q/2KV` GQA Attention, untied token embeddings, and untied LM head.

### B. Nemotron-Labs-TwoTower 30B-A3B (`/twotower/` Entry)

1. **Technical Paper:** NVIDIA Nemotron Labs, *Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context*, [`arXiv:2606.26493v2`](https://arxiv.org/abs/2606.26493).
2. **Pinned Public Checkpoint Metadata:** Hugging Face [`nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16/tree/35b6498508be8856d380cdfe4a900a79b918d1ea) pinned at revision `35b6498508be8856d380cdfe4a900a79b918d1ea` (`config.json`, `generation_config.json`, `tokenizer_config.json`, `model.safetensors.index.json`). An offline, read-only, CPU-only audit of the 24 safetensors shard headers verified exact tensor counts (`12,544`), dtypes (`12,498` BF16 + `46` FP32 `e_score_correction_bias`), parameter totals (`63,185,902,848`), and payload byte totals (`126,371,817,472 B`) with zero weight values exported, stored, or redistributed.
3. **Reference Forward-Pass Equations (Clean-Room Apache-2.0 Reference):** Mathematical equations of `NemotronHTwoTowerForCausalLM` (`modeling_nemotron_twotower.py`, **Apache License 2.0**) — including independent Context and Denoiser backbones, aligned 29-bay read-only state handoff, split `adaLN-single` ordering (`modulate → RMSNorm` on `M`/`*` vs. `RMSNorm → modulate` on `E`, unmodulated final `norm_f`), FP64 MoE router matmul (`--moe-router-dtype fp64`), and `AR` / `Mock-AR` / `Mask-Diffusion` execution paths — implemented strictly clean-room in TypeScript (`src/twotower/`) and offline PyTorch (`scripts/reference/generate_twotower_pytorch_fixtures.py`). **Zero upstream source files, proprietary code, or AGPL code are copied into this repository.**
4. **Foundational Methodological Citations:**
   - **PixArt-α `adaLN-single` Timestep Modulation:** Junsong Chen et al., *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis* ([`arXiv:2310.00426`](https://arxiv.org/abs/2310.00426)).
   - **MDLM / SUBS Masked Diffusion Parameterization:** Subham Sekhar Sahoo et al., *Simple and Effective Masked Diffusion Language Models* ([`arXiv:2406.07524`](https://arxiv.org/abs/2406.07524)) — unshifted clean-token prediction, `[MASK]` probability clamping (`maskTokenProb = 0`), and monotonic carry-over locking.
5. **Non-Official Community Quantization Reference:** The Architecture tab includes an explicitly labeled **non-official community NVFP4 / W4A16 reference note** (`TWOTOWER_COMMUNITY_NVFP4_REFERENCE` in `src/twotower/spec.ts`, ~`42 GiB` expert-only compression) strictly distinguished from NVIDIA's official `126.37 GB` (`117.69 GiB`) BF16 checkpoint release.

### C. Nemotron-Labs-Diffusion-14B (`/diffusion/` Entry)

1. **Technical Paper:** NVIDIA Nemotron Labs, *Nemotron-Labs-Diffusion*, [`arXiv:2607.05722v1`](https://arxiv.org/abs/2607.05722) (`2026-07-07`, 21 pages).
2. **Pinned Public Checkpoint & GitHub Metadata:**
   - **Canonical Checkpoint:** Hugging Face [`nvidia/Nemotron-Labs-Diffusion-14B`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B/tree/f8c3e2c078e193599b8882d965b1001c456ba738) pinned at revision `f8c3e2c078e193599b8882d965b1001c456ba738`, reconciled with `0 B` diff across all `13` LFS safetensors files in the 7-repo family (`14B`, `14B-Base`, `8B`, `8B-Base`, `3B`, `3B-Base`, `VLM-8B`) plus 3 `linear_spec_lora` adapters (`src/diffusion/spec.ts`).
   - **Reference GitHub Repository:** [`NVlabs/Nemotron-Labs-Diffusion`](https://github.com/NVlabs/Nemotron-Labs-Diffusion/tree/2aabcf75ecf70264d4047ff090d842450853f2e4) pinned at commit `2aabcf75ecf70264d4047ff090d842450853f2e4` (code license: **Apache License 2.0**; weights license: **NVIDIA Open Model License**, Oct 24, 2025).
3. **Clean-Room Mathematical Reference:** Ministral 3 (`Ministral3Model`) 40-layer dense decoder equations, YaRN RoPE (`truncate=True`), Llama-4 query attention temperature scaling (`beta=0.1, orig=16384`), Dense SwiGLU, BD3-LM / block-diffusion / linear speculative verification masks, Sampler A / Sampler B rules, and draft-only PEFT `o_proj` LoRA (`r=128, α=512, scale=4.0`) are implemented strictly clean-room in TypeScript (`src/diffusion/`) and verified offline against `scripts/reference/generate_diffusion_pytorch_fixtures.py` and `scripts/reference/generate_diffusion_trajectory_fixtures.py` (`torch==2.15.0`). **Zero upstream NVIDIA Python source files, weights, LoRA safetensors, or tokenizers are copied or redistributed.**

### D. Nemotron-Labs-Diffusion-Uni / NLD-Uni (`/uni/` Entry)

1. **Technical Paper:** NVIDIA Nemotron Labs, *Nemotron-Labs-Diffusion-Uni*, [`arXiv:2610.36466v1`](https://arxiv.org/abs/2610.36466).
2. **Pinned Public Checkpoint & GitHub Metadata:**
   - **Canonical Checkpoint:** Hugging Face [`nvidia/NLD-Uni`](https://huggingface.co/nvidia/NLD-Uni/tree/5db755a2eb0ab84654e3dcf80c7d0245bfe36a50) pinned at revision `5db755a2eb0ab84654e3dcf80c7d0245bfe36a50`, reconciled with `0 B` diff across all `8` safetensors shards (`1,557` tensors = `18,267,552,262` params = `36,535,104,542 B` payload, `src/uni/spec.ts`).
   - **Reference GitHub Repository:** [`NVlabs/Nemotron-Labs-Diffusion-Uni`](https://github.com/NVlabs/Nemotron-Labs-Diffusion-Uni/tree/16916949402ca738dcc74a7a37e0c43facafe347) pinned at commit `16916949402ca738dcc74a7a37e0c43facafe347` (code license: **Apache License 2.0**; weights license: **NVIDIA Open Model License**).
3. **Clean-Room Mathematical Reference:** Twin 34-layer Elastic Mixture-of-Transformers (`18` Joint `L0..17` + `16` Split `L18..33` parallel Understanding `U` and Generation `G` towers), continuous `RADIO-H` ViT + `2×2` SquaredReLU projector, discrete `Emu3` IBQ VQ-VAE + `2×2` `PackedLinearImageCodec`, 1D dyadic YaRN RoPE + Llama-4 query temperature scaling + additive 2D `SinCos-G` coordinates, `BalancedSemanticTree` hierarchical ball-tree discrete diffusion, stratified quadtree refinement, Dual-CFG image editing (`legacy` / `independent` / `ip2p`), FLARE shifted-logit understanding, `<LOC_0..16>` 2D grounding, and read-only forward-only OPD diagnostic reference are implemented strictly clean-room in TypeScript (`src/uni/`) and verified offline against `scripts/reference/generate_uni_{backbone,multimodal,diffusion}_pytorch_fixtures.py` (`torch==2.15.0`). **Zero upstream remote source lines, checkpoint weights, tokenizer files, or `pca_tree.pt` binaries are copied or redistributed.**

### E. Nemotron-Labs-Dory1.0-4B (`Dory1.0-4B` · `/dory/` Entry)

1. **Technical Paper:** NVIDIA Nemotron Labs, *Dory1.0-4B: A Looped Normalized-GPT*, [`arXiv:2611.03998`](https://arxiv.org/abs/2611.03998) (**CC BY 4.0**; TeX template `nvidiatechreport.cls` CC BY-SA 4.0; paper metadata & equations referenced cleanly — zero TeX figures or source files redistributed).
2. **Pinned Public Checkpoint Metadata (`nvidia/nemotron-labs-dory`, Weights License: OpenMDW-1.1):**
   - `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` @ `ee517dd388a9b348890ae1e00dacc5eb4d0c750d` (`507` BF16 tensors, `3,975,229,440` params, `7,950,515,472 B`, `0 B` diff via HTTP `Range` JSON header audit; `0 B` weight bytes downloaded or redistributed).
   - `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8` @ `f57fc2d5353e0a3847ae0544c33e5601672f4065` (`795` tensors, `4,750,755,304 B`, `0 B` diff).
   - `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4` @ `17065a6849266dd478221735be7747689a0a4331` (`1,083` tensors, `3,202,498,032 B`, `0 B` diff).
   - `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF` @ `df53755cb8741843af8e019e088819fbef17e29b` (`NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`, `3,673,288,864 B`).
   - *(Note `DR16`: Governing model weight license across all 4 HF README cards and paper §1 is **OpenMDW-1.1** [`https://openmdw.ai/license/1-1/`]; `BF16/explainability.md` subcard lags with a legacy link.)*
3. **Upstream Open-Source Runtime Cross-Checks (Clean-Room Mathematical Verification Only — Zero Code Copied):**
   - `huggingface/transformers` @ `3a96649547593bacad4b65d43442ccdf15d0ae0f` (**Apache License 2.0**, Copyright 2026 NVIDIA & Hugging Face) — paired-36L `LayerPairCollapse` & `DoryPastReader` semantics.
   - `vllm-project/vllm` @ `75fc05a53a366068f534a61ab3ee6b50485419f2` (PR `#60149`, **Apache License 2.0**) — physical-72L FP32-scaled forward semantics.
   - `ggml-org/llama.cpp` PR `#30306` @ `8481bd0fde646c3f0423cd09320fe9b85b5fb0a5` (**MIT License**) — physical-72L graph, `get_can_shift() = false`, and `last_loop` partition behavior.
   - All `src/dory/` TypeScript modules and offline PyTorch `2.15.0` scripts (`scripts/reference/generate_dory_{pytorch,spec}_fixtures.py`) were authored 100% clean-room from first-principles equations.
4. **Foundational Lineage Citations:**
   - **nGPT (Normalized Transformer on the Hypersphere):** Ilya Loshchilov et al., *nGPT: Normalized Transformer with Representation Learning on the Hypersphere* ([`arXiv:2410.01131`](https://arxiv.org/abs/2410.01131)).
   - **ELR (Elastic Latent Recurrence) & NVFP4:** Ladhak et al., `arXiv:2610.12582`; NVIDIA NVFP4 training, `arXiv:2509.25149`.

> **No NVIDIA Model Artifacts Redistributed**
>
> **NO NVIDIA model weights, checkpoints, safetensors shards, GGUF binaries, LoRA adapters, `pca_tree.pt` binaries, tokenizers, vocabulary files, upstream `modeling_*.py` / `dory.py` / `dory.cpp` source files, private research datasets, or OpenMDW-1.1 / NVIDIA Open Model License artifacts are redistributed in this repository or in any built `dist/` bundle.**
>
> All five browser simulators and offline PyTorch parity generators are clean-room implementations driven solely by deterministic synthetic miniature Float32/Float64 weights (**Nano:** `d = 8`, `V_mini = 32`, `299,192` Float32s; **TwoTower:** `d = 8`, `V_mini = 34`, `600,120` Float32s; **Diffusion:** `d = 8`, `V_mini = 36`, `271,304` Float32s [`229,064` backbone + `42,240` LoRA]; **NLD-Uni:** `d = 8`, `V_txt = 64` [`+17 <LOC_0..16>` bins], `V_vq = 256` [`D = 8` synthetic ball-tree], `766` tensors = `191,235` Float32s + `3,068 B` tree; **Dory1.0-4B:** `72` physical sublayers, `507` tensors = `93,248` Float32s [`372,992 B` = `0.356 MiB`], `d = 8`, `d_ff = 24`, `8Q / 2KV`, `d_h = 8`, `V = 64`, `seed = 0xD0411004`, generated via deterministic seeded PRNGs) while presenting pinned production architectural metadata.

**Trademark Notice:** NVIDIA and Nemotron are trademarks or registered trademarks of NVIDIA Corporation. This project is an independent educational visualization and is not affiliated with, sponsored by, or endorsed by NVIDIA Corporation.

---

## 3. Development-Only Offline Parity, Build & Test Tooling

### Browser Runtime Bundles (`dist/`, `dist/twotower/`, `dist/diffusion/`, `dist/uni/` & `dist/dory/`)

All five compiled browser applications have **zero third-party runtime dependencies** (`dependencies` is omitted/empty in `package.json`; all `73/73` non-root entries in `package-lock.json` are `dev: true`). All WebGL2 rendering, math simulation, layout, timeline/sampler/tour logic, and SVG/DOM charting run exclusively from first-party TypeScript and CSS modules.

### Offline PyTorch Reference Parity Tooling (`scripts/reference/`)

Used strictly offline by developers to regenerate and cross-check the checked-in JSON numerical parity fixtures under `tests/fixtures/`. Python and PyTorch are **never** bundled into `dist/` and are **not** required to run `npm test`, `npm run build`, or any browser app.

| Package | Pinned Version | License | Usage |
| --- | --- | --- | --- |
| PyTorch (`torch`) | `2.15.0` | BSD-3-Clause | Offline clean-room reference execution in `scripts/reference/generate_pytorch_parity_fixtures.py` (Nano), `scripts/reference/generate_twotower_pytorch_fixtures.py` (TwoTower Categories 1–8), `scripts/reference/generate_diffusion_pytorch_fixtures.py` & `scripts/reference/generate_diffusion_trajectory_fixtures.py` (Diffusion Categories 1–11), `scripts/reference/generate_uni_{backbone,multimodal,diffusion}_pytorch_fixtures.py` (NLD-Uni), and `scripts/reference/generate_dory_pytorch_fixtures.py` & `scripts/reference/generate_dory_spec_fixtures.py` (Dory1.0-4B backbone, upstream differential variants & 6-layer Contract A self-speculative parity) (`requirements-ref.txt`, PEP 723 metadata) |

### Direct npm `devDependencies` (Exact `package.json` Pins — Build, Type-Check & Test Only)

Installed only during local development and CI (`.npmrc` sets `save-exact=true`); none of these packages are bundled into the static browser output (`dist/`).

| Package | Exact Pinned Version (`package.json`) | License | Usage |
| --- | --- | --- | --- |
| `@playwright/test` | `1.65.1` | Apache-2.0 | Headless Chromium WebGL2 E2E smoke suite (`tests/e2e/smoke.spec.ts`, `tests/e2e/twotower-smoke.spec.ts`, `tests/e2e/diffusion-smoke.spec.ts`, `tests/e2e/uni-smoke.spec.ts`, `tests/e2e/dory-smoke.spec.ts`, `tests/e2e/navigation.spec.ts`); downloads Chromium binary strictly for test execution |
| `@types/node` | `22.20.5` | MIT | Node.js type definitions for build, isolation-gate, and test tooling |
| `typescript` | `5.9.3` | Apache-2.0 | Static type checker (`tsc --noEmit`) and ECMAScript AST parser inside `scripts/verify_bundle_isolation.mjs` |
| `vite` | `8.3.8` | MIT | Dev server, SSR module loader for offline parameter export, Topology A multi-entry static production bundler |
| `vitest` | `5.1.1` | MIT | Node unit, integration, contract, and numerical parity test runner |

Transitive npm `devDependencies` recorded in `package-lock.json` (`73` non-root dev-only entries; `0` runtime dependencies; `0` npm audit vulnerabilities) are build/test-time utilities under permissive or file-level licenses (`MIT` [52], `MPL-2.0` [12, dev-only `lightningcss` + platform binaries], `Apache-2.0` [6], `ISC` [1], `BSD-3-Clause` [1], `0BSD` [1]; `0` GPL/AGPL/LGPL/SSPL) and are never shipped in `dist/`.
