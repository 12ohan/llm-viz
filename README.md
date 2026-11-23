# NVIDIA Nemotron 3 Nano 30B-A3B — 3D Interactive Architecture Visualization

An interactive WebGL2 3D architectural explainer of **NVIDIA Nemotron 3 Nano 30B-A3B**, built in the visual language of Brendan Bycroft's `llm-viz`.

**Live demo:** [https://12ohan.github.io/llm-viz/](https://12ohan.github.io/llm-viz/)

## Highlights

- **52-Layer Hybrid Vertical Tower** — All 52 layers derived directly from the checkpoint `hybrid_override_pattern` (`23 Mamba-2 · 23 MoE 128+1 · 6 GQA Attention`) partitioned into NVIDIA Figure 2's four motif spans (`01–35`, `36–41`, `42–44`, `45–52`), flanked by the continuous 2,688-channel emerald residual stream, untied 131,072-token embedding matrix, and final RMSNorm + untied LM head.
- **Three Exploded 3D Layer Cutaways** — Click any bay or switch tabs to inspect internal tensor geometry:
  - **Mamba-2 (Bay 01 & Bay 42 Feeder)**: Causal Conv1D (`k=4`, `6,144` channels, always BF16) + 64-head (`64 × 64 × 128`) FP32 recurrent state updating in place (`47.1 MiB` total across all 23 bays).
  - **Granular MoE 128+1 (Bay 02)**: Parallel sigmoid router activating 6 of 128 non-gated ReLU² routed experts (`2,688 → 1,856 → 2,688`, 2 matrices each) beside 1 double-width shared expert (`3,712 = 2 × 1,856`), `0 B` sequence cache.
  - **GQA Attention (Bay 06 & Bay 43)**: 32 Query heads sharing 2 KV heads (`16:1`, `NoPE`) with live growing K/V token stacks (`+6 KiB/tok` BF16 / `+3 KiB/tok` FP8 across all 6 GQA bays).
- **Guided 7-Chapter Walkthrough** — Step-by-step architectural tour (`Continue →` / `←` / `→` / `Exit Tour`) that flies the camera and toggles cutaways and FP8 Selective mode while keeping free 3D exploration unlocked at all times.
- **Interactive 1 → 1M Token Memory Scaling Chart** — Zero-dependency inline SVG log-log chart comparing Nemotron 3 Nano total sequence state (`3.05 GiB` FP8 / `6.05 GiB` BF16 at 1M tokens) against a 52-layer pure GQA Transformer (`26.0 GiB` FP8 / `52.0 GiB` BF16 — ~8.5× larger), marking exact crossovers at `1,048 / 2,096` and `8,035 / 16,070` tokens.
- **Selective FP8 Quantization Mode (NVIDIA §4.2–4.3)** — Live `[ BF16 | FP8 Selective ]` toggle highlighting the 12 BF16-preserved sensitive bays (6 GQA + 6 preceding Mamba-2 feeders `05, 12, 19, 26, 33, 42`) and 23 BF16 Conv1D strips while quantizing the remaining 40 layers and all 6 KV caches to FP8.
- **Keyboard Shortcuts** — `1` Tower, `2` Mamba-2, `3` MoE, `4` GQA, `P` toggle BF16/FP8, `M` Memory Scaling chart, `←`/`→` step chapter or bay, `R` reset view, `?` shortcuts help.

## Local Development

Requires Node.js 20+ (CI uses Node 22).

```bash
# Install dependencies
npm install

# Start Vite dev server on http://localhost:3000
npm run dev

# Type-check
npx tsc --noEmit

# Run Vitest unit test suite
npm test

# Production build (root path)
npm run build

# Production build for GitHub Pages subpath (/llm-viz/)
VITE_BASE_PATH=/llm-viz/ npm run build
```

## Architecture & Citations

All layer counts, tensor dimensions, memory equations, and quantization sensitivity boundaries are grounded in:

- **NVIDIA Nemotron 3 Nano Technical Report**: [arXiv:2512.20848v1](https://arxiv.org/abs/2512.20848) (§2.1, Figure 2, Table 1, Table 3, §4.2–4.3).
- **Released BF16 Checkpoint Config**: [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/config.json`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json) (pinned revision `bf77c3174f68ad409e1c2aa60daeb46e32d1c606`).

## Credits & License

- Visual design language and interactive walkthrough inspiration are credited to **Brendan Bycroft's [`bbycroft/llm-viz`](https://github.com/bbycroft/llm-viz)**, which this repository forks and extends for Nemotron 3 Nano 30B-A3B.
- **License:** MIT (see [`LICENSE`](LICENSE)).
