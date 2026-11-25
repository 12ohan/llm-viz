# Third-Party Notices & Attribution

This document records the verified provenance, licenses, and applicable obligations for upstream projects, architectural references, and development tooling associated with **Nemotron 3 Nano 30B-A3B — Architecture Visualization**.

---

## 1. Incorporated / Forked Repository History & Visual Language

| Project | Author | License | Repository |
| --- | --- | --- | --- |
| `llm-viz` | Brendan Bycroft | MIT License | <https://github.com/bbycroft/llm-viz> |

- **Copyright Notice (preserved verbatim in [`LICENSE`](LICENSE)):**
  `Copyright (c) 2023-2026 Brendan Bycroft`
- **Fork Contributions:**
  `Copyright (c) 2026 Rohan Master`
- **Scope & Distinction:** This repository descends from `bbycroft/llm-viz` and adapts its interactive architectural walkthrough idiom and adaptive screen-space (`fwidth()`) cell-grid LOD principle (credited in `src/render/sheets.ts`). All Nemotron 3 Nano browser application modules (`src/`), unit/parity test suites (`tests/`), and reference verification scripts (`scripts/reference/`) were newly authored for NVIDIA Nemotron 3 Nano 30B-A3B; legacy `bbycroft/llm-viz` source files and `public/` assets were removed prior to v1.0.
- **Applicable Obligation:** Retain the upstream MIT copyright notice and permission notice in [`LICENSE`](LICENSE) and in copies or substantial portions of the Software.

---

## 2. Referenced Architectural Equations & Pinned Public Metadata (Not Redistributed Artifacts)

This interactive explainer grounds its layer topology, tensor dimensions, memory formulas, and quantization boundaries in publicly published NVIDIA specifications:

1. **Technical Report:** NVIDIA, *NVIDIA Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*, [`arXiv:2512.20848v1`](https://arxiv.org/abs/2512.20848) (§2.1, Figure 2, Table 1, Table 3, §§4.2–4.3).
2. **Pinned Public Model Config:** Hugging Face [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/config.json`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/config.json) pinned at commit `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`.
3. **Reference Forward-Pass Equations:** Mathematical definitions from NVIDIA's [`modeling_nemotron_h.py`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/bf77c3174f68ad409e1c2aa60daeb46e32d1c606/modeling_nemotron_h.py) (**Apache License 2.0**) — covering `NemotronHRMSNorm`, `NemotronHMamba2Mixer` + `MambaRMSNormGated` (`norm_before_gate=False`), `NemotronHMOE` + `NemotronHTopkRouter` + non-gated `ReLU²` `NemotronHMLP`, NoPE `32Q/2KV` GQA Attention, untied token embeddings, and untied LM head.

> **No NVIDIA Model Artifacts Redistributed**
>
> **NO NVIDIA model weights, checkpoints, safetensors, tokenizers, vocabulary files, third-party `modeling_nemotron_h.py` source files, or NVIDIA Open Model License artifacts are redistributed in this repository or in the built `dist/` bundle.**
>
> Both the interactive TypeScript browser simulator (`src/math/`) and the offline PyTorch parity generator (`scripts/reference/generate_pytorch_parity_fixtures.py`) are clean-room implementations driven solely by deterministic synthetic miniature Float32 weights (`d = 8`, `V_mini = 32`, generated via Mulberry32 + Box-Muller PRNG) while displaying pinned production BF16/FP8 architectural metadata (`hidden = 2,688`, `vocab = 131,072`, `31.6B` total / `3.6B` active parameters).

**Trademark Notice:** NVIDIA and Nemotron are trademarks or registered trademarks of NVIDIA Corporation. This project is an independent educational visualization and is not affiliated with, sponsored by, or endorsed by NVIDIA Corporation.

---

## 3. Development-Only Offline Parity, Build & Test Tooling

### Browser Runtime Bundle (`dist/`)

The compiled browser bundle has **zero third-party runtime dependencies** (`dependencies` is empty/omitted in `package.json`). All WebGL2 rendering, math simulation, layout, and SVG/DOM charting run from first-party TypeScript and CSS modules.

### Offline PyTorch Reference Parity Tooling (`scripts/reference/`)

Used strictly offline by developers to generate and cross-check the checked-in JSON numerical parity fixtures under `tests/fixtures/`. Python and PyTorch are **never** bundled into `dist/` and are **not** required to run `npm test`, `npm run build`, or the browser app.

| Package | Pinned Version | License | Usage |
| --- | --- | --- | --- |
| PyTorch (`torch`) | `2.15.0` | BSD-3-Clause | Offline clean-room reference execution in `scripts/reference/generate_pytorch_parity_fixtures.py` (`requirements-ref.txt`, PEP 723 metadata) |

### Direct npm `devDependencies` (Build, Type-Check & Test Only)

Installed only during local development and CI; none of these packages are bundled into the static browser output (`dist/`).

| Package | Version / Range | License | Usage |
| --- | --- | --- | --- |
| `vite` | `8.3.8` | MIT | Dev server, SSR module loader for fixture export, static production bundler |
| `vitest` | `5.1.1` | MIT | Node unit, integration, and numerical parity test runner |
| `typescript` | `^5.6.3` (`5.9.3` in lockfile) | Apache-2.0 | Static type checker (`tsc --noEmit`) |
| `@types/node` | `^22.12.0` (`22.20.5` in lockfile) | MIT | Node.js type definitions for build/test tooling |
| `@playwright/test` | Dev-only E2E test harness | Apache-2.0 | Headless Chromium WebGL2 E2E smoke test runner (downloads Chromium browser binary only for local/CI test execution; never bundled into `dist/`) |

Transitive npm `devDependencies` recorded in `package-lock.json` are build/test-time utilities under permissive or file-level licenses (`0BSD`, `Apache-2.0`, `BSD-3-Clause`, `ISC`, `MIT`, `MPL-2.0`) and are not shipped in `dist/`.
