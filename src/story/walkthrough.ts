import type { PrecisionMode } from '../model/spec';
import type { TocSectionId, ViewMode } from '../model/state';

export interface WalkthroughChapter {
  readonly index: number; // 0-based (0..6)
  readonly number: number; // 1-based (1..7)
  readonly id: string;
  readonly shortTitle: string;
  readonly title: string;
  readonly badge: string;
  readonly mode: ViewMode;
  readonly section: TocSectionId;
  readonly precision: PrecisionMode;
  readonly tokens: number;
  readonly selectedBay: number | null;
  readonly mambaBayId?: 1 | 42;
  readonly gqaBayId?: 6 | 43;
  readonly highlightMemoryChart?: boolean;
  readonly paragraphs: readonly string[];
  readonly takeaways: readonly string[];
}

export const WALKTHROUGH_CHAPTERS: readonly WalkthroughChapter[] = Object.freeze([
  {
    index: 0,
    number: 1,
    id: 'hybrid-spine',
    shortTitle: 'Hybrid 52-Layer Spine',
    title: 'Hybrid 52-Layer Spine',
    badge: '23 M · 23 E · 6 *',
    mode: 'tower',
    section: 'overview',
    precision: 'bf16',
    tokens: 8,
    selectedBay: null,
    paragraphs: [
      'NVIDIA Nemotron 3 Nano 30B-A3B stacks 52 sequential layers along a vertical hybrid spine: 23 Mamba-2 state-space layers (M), 23 sparse Mixture-of-Experts banks (E), and only 6 Grouped-Query Attention layers (* at bays 06, 13, 20, 27, 34, 43).',
      'NVIDIA Figure 2 partitions the tower into four motif spans — 01–35 [M·E·M·E·M·*·E] ×5, 36–41 [M·E] ×3, 42–44 [M·*·E] ×1, and 45–52 [M·E] ×4. Cutting full attention from 52 layers down to 6 slashes per-token KV cache growth by 8.7× (+6 KiB/tok vs. +52 KiB/tok in BF16).',
    ],
    takeaways: [
      '31.6B total parameters · 3.2B active per token excluding embeddings (3.6B including embeddings)',
      'Only 6 bays grow K/V token tracks along +Z; 23 Mamba bays hold fixed state and 23 MoE bays store 0 B',
      'Drag anywhere to orbit or click any bay at any time — free exploration stays unlocked',
    ],
  },
  {
    index: 1,
    number: 2,
    id: 'embedding-residual',
    shortTitle: 'Embedding & Residual Stream',
    title: 'Embedding & Residual Stream',
    badge: '131,072 × 2,688',
    mode: 'tower',
    section: 'embedding',
    precision: 'bf16',
    tokens: 8,
    selectedBay: null,
    paragraphs: [
      'Every input token ID indexes one 2,688-channel row of the 131,072 × 2,688 embedding table and injects it into the continuous emerald residual highway running along the right side of the tower.',
      'Embeddings are untied from the final output head (tie_word_embeddings: false), so input lookup and logit projection maintain independent weights. Each of the 52 layers taps the stream through Pre-RMSNorm and adds its delta back without rescaling the bypass.',
    ],
    takeaways: [
      'Untied embedding matrix: 131,072 vocab × 2,688 hidden (~352M parameters)',
      'Continuous 2,688-channel emerald residual spine runs unbroken from Embedding to Final RMSNorm',
      'Pre-RMSNorm keeps gradient and activation flow stable across mixed Mamba, MoE, and GQA blocks',
    ],
  },
  {
    index: 2,
    number: 3,
    id: 'mamba-fixed-memory',
    shortTitle: 'Mamba-2: Fixed-Size Memory',
    title: 'Mamba-2: Fixed-Size Memory',
    badge: 'Bay 01 · 47.1 MiB Fixed',
    mode: 'cutaway_mamba',
    section: 'mamba',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 1,
    mambaBayId: 1,
    paragraphs: [
      'Inside Mamba-2 (Bay 01), the input projection splits into three parallel paths: bypass gate z (4,096), a 4-tap depthwise Causal Conv1D over 6,144 channels (x, B, C), and 64 per-head decay steps Δt.',
      'The centerpiece is a 64-head × 64 × 128 FP32 recurrent state array that updates in place via h_t = A·h_{t−1} + B·x_t. Whether generating token 1 or token 1,048,576, all 23 Mamba-2 bays combined stay strictly fixed at 47.1 MiB (49,364,992 bytes).',
    ],
    takeaways: [
      'Causal Conv1D (k=4, 6,144 channels, BF16): 48 KiB/layer · SSM State (64×64×128, FP32): 2.00 MiB/layer',
      '2.05 MiB per Mamba bay × 23 bays = 47.1 MiB constant from 1 to 1M tokens',
      'Try scrubbing Cached tokens (1 → 16) below: state cells refresh in place without growing geometry',
    ],
  },
  {
    index: 3,
    number: 4,
    id: 'moe-parallel-sparsity',
    shortTitle: 'Granular MoE (128+1): Parallel Sparsity',
    title: 'Granular MoE (128+1): Parallel Sparsity',
    badge: 'Bay 02 · 6 Active + Shared',
    mode: 'cutaway_moe',
    section: 'moe',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 2,
    paragraphs: [
      'Roughly 94% of the ~63 GB BF16 checkpoint (29.8B of 31.6B parameters) lives inside the 23 Mixture-of-Experts bays. Yet each token only activates 6 of 128 routed experts via a sigmoid router, running in parallel alongside 1 always-active double-width shared expert (width 3,712 = 2 × 1,856).',
      'Every expert uses non-gated squared ReLU (ReLU²) with just two weight matrices (up_proj + down_proj) instead of SwiGLU’s three. The other 122 routed experts (95.3%) stay completely idle for that token, and MoE layers add 0 B to the sequence cache.',
    ],
    takeaways: [
      'Routed experts: 6 active of 128 (width 1,856, ReLU², 2 matrices: up + down — no gate_proj)',
      'Shared expert: width 3,712 (= 2 × 1,856 fused in checkpoint) runs in parallel on every token',
      '1.30B params/layer × 23 = 29.8B total MoE weights · 122 experts idle · 0 B sequence cache',
    ],
  },
  {
    index: 4,
    number: 5,
    id: 'gqa-nope',
    shortTitle: 'GQA Attention (×6) & NoPE',
    title: 'GQA Attention (×6) & NoPE',
    badge: 'Bay 06 · 32Q / 2KV · T=12',
    mode: 'cutaway_gqa',
    section: 'gqa',
    precision: 'bf16',
    tokens: 12,
    selectedBay: 6,
    gqaBayId: 6,
    paragraphs: [
      'Only six layers perform exact token-to-token retrieval. Each GQA bay pairs 32 Query heads (2 combs × 16 heads) against just 2 Key/Value heads (16:1 sharing, head_dim 128) with NoPE — no positional embeddings, because positional order already arrives from the preceding Mamba-2 layer.',
      'Unlike Mamba’s fixed register, these 6 layers append verbatim K and V slices on every token. Across all 6 attention bays, each added token costs +6 KiB in BF16 — 8.7× less than +52 KiB/tok for a 52-layer pure GQA Transformer.',
    ],
    takeaways: [
      '32 Q heads (4,096) share 2 KV heads (256 each) — 16× smaller KV footprint per attention layer',
      'NoPE (No Positional Embeddings): preceding Mamba-2 Conv1D + SSM encode sequence order',
      'At T=12 in BF16: 12 KiB in Bay 06 (72 KiB across 6 GQA bays vs. 624 KiB for 52-layer GQA)',
    ],
  },
  {
    index: 5,
    number: 6,
    id: 'selective-fp8',
    shortTitle: 'Selective FP8 & Mamba→Attn Feeders',
    title: 'Selective FP8 & Mamba→Attn Feeders',
    badge: 'Bays 42–43 · FP8 Selective',
    mode: 'tower',
    section: 'kv_fp8',
    precision: 'fp8_selective',
    tokens: 9,
    selectedBay: 42,
    paragraphs: [
      'FP8 Selective mode is now enabled: watch NVIDIA §4.2–4.3’s sensitivity boundary light up around Bays 42–43. Quantizing all 52 layers indiscriminately degrades attention, so NVIDIA preserves BF16 compute on the 6 attention bays, the 6 immediately preceding Mamba-2 feeder bays (05, 12, 19, 26, 33, 42), and all 23 Conv1D strips.',
      'The remaining 40 layers (17 Mamba + 23 MoE) and all 6 KV caches quantize cleanly to FP8 — halving KV growth to +3 KiB/tok while recovering ~99% median BF16 accuracy (MMLU-Pro 78.30 → 78.10; AIME25 w/ tools 99.17 → 98.80).',
    ],
    takeaways: [
      '12 BF16-preserved bays (gold rims): 6 GQA (06, 13, 20, 27, 34, 43) + 6 Mamba feeders (05, 12, 19, 26, 33, 42)',
      'All 23 Conv1D strips stay BF16; all 23 SSM states stay FP32; all 6 KV caches switch to FP8 (+3 KiB/tok)',
      '~99% median accuracy recovery: MMLU-Pro 78.30 → 78.10 · AIME25 w/ tools 99.17 → 98.80 (NVIDIA §4.3)',
    ],
  },
  {
    index: 6,
    number: 7,
    id: 'long-context-output',
    shortTitle: 'Long-Context Scaling (1M) & Output Head',
    title: 'Long-Context Scaling (1M) & Output Head',
    badge: '1M Context · 131K Logits',
    mode: 'tower',
    section: 'output',
    precision: 'fp8_selective',
    tokens: 9,
    selectedBay: null,
    highlightMemoryChart: true,
    paragraphs: [
      'After Bay 52 (MoE), the emerald residual stream passes through Final RMSNorm (2,688) and the untied LM Head (2,688 → 131,072) to emit next-token vocabulary logits.',
      'At 1M tokens (1,048,576), Nemotron’s hybrid sequence state totals just 3.05 GiB in FP8 (6.05 GiB BF16) versus 26.0 GiB FP8 (52.0 GiB BF16) for a 52-layer GQA Transformer — ~8.5× smaller. Open the highlighted Memory Scaling (1 → 1M) chart below to explore the exact crossover curves.',
    ],
    takeaways: [
      'Final RMSNorm (2,688) → Untied LM Head (2,688 × 131,072) → 131,072 next-token logits',
      'At 1M tokens: 3.05 GiB (FP8) / 6.05 GiB (BF16) vs. 26.0 GiB / 52.0 GiB pure GQA (~8.5× smaller)',
      'Click "Memory Scaling 1 → 1M ↗" (or press M) to inspect crossovers at 1,048 / 2,096 and 8,035 / 16,070 tokens',
    ],
  },
]);

export const WALKTHROUGH_COUNT = WALKTHROUGH_CHAPTERS.length;

export function getWalkthroughChapter(index: number): WalkthroughChapter {
  const clamped = Math.max(0, Math.min(WALKTHROUGH_COUNT - 1, Math.trunc(index)));
  return WALKTHROUGH_CHAPTERS[clamped];
}

export function nextChapterIndex(current: number): number | null {
  return current + 1 < WALKTHROUGH_COUNT ? current + 1 : null;
}

export function prevChapterIndex(current: number): number | null {
  return current - 1 >= 0 ? current - 1 : null;
}
