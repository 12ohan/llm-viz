import type { PrecisionMode } from '../model/spec';
import type { ExecutionMode, TocSectionId, ViewMode } from '../model/state';

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

// ============================================================================
// V5 REASSESSED 22-BEAT, 3-ACT UNIFIED + AR DECODE GUIDED WALKTHROUGH
// ============================================================================

export type WalkthroughBeatMode = ViewMode | 'cutaway_embedding' | 'cutaway_output';

export type WalkthroughAct = 1 | 2 | 3;

export type WalkthroughArAction =
  | 'ensure_prefill'
  | 'step_once_with_wave'
  | 'inspect_cached_gqa'
  | 'invite_branch_and_chart';

export interface WalkthroughBeat {
  readonly index: number;
  readonly id: string;
  readonly chapterIndex: number;
  readonly beatInChapter: number;
  readonly chapterBeatCount: number;
  readonly act: WalkthroughAct;
  readonly actLabel: string;
  readonly title: string;
  readonly badge: string;
  readonly mode: WalkthroughBeatMode;
  readonly section: TocSectionId;
  readonly executionMode: ExecutionMode;
  readonly precision: PrecisionMode;
  readonly tokens: number;
  readonly selectedBay: number | null;
  readonly mambaBayId?: 1 | 42;
  readonly moeBayId?: 2 | 52;
  readonly gqaBayId?: 6 | 43;
  readonly gqaHead?: number;
  readonly stepperStep?: number;
  readonly stepperAutoPlay?: boolean;
  readonly arAction?: WalkthroughArAction;
  readonly highlightMemoryChart?: boolean;
  readonly paragraphs: readonly string[];
  readonly takeaways: readonly string[];
  readonly interactivePrompt?: string;
}

const ACT_1_LABEL = 'Act I · Full-Sequence Prefill';
const ACT_2_LABEL = 'Act II · Precision Boundary';
const ACT_3_LABEL = 'Act III · Live Cached AR Decode & 1M Scaling';

/**
 * Canonical guided-walkthrough prompt length (`T = 8`: "How does Nemotron use only six attention
 * layers", positions `t0..t7`). Every Act I, Act II, and Beat 7.1 Output beat shares this exact
 * 8-token prefill prefix, and Act III Beats 7.2–7.4 step directly from it (`8P + 1S = 9`).
 */
export const TOUR_PREFILL_TOKENS = 8 as const;

/** Canonical token IDs (`t0..t7`) for the unified 8-token guided walkthrough prompt. */
export const TOUR_PREFILL_TOKEN_IDS: readonly number[] = Object.freeze([0, 1, 2, 3, 4, 5, 6, 7]);

export const WALKTHROUGH_BEATS: readonly WalkthroughBeat[] = Object.freeze([
  // --------------------------------------------------------------------------
  // ACT I · FULL-SEQUENCE PREFILL — Ch 1: Hybrid 52-Layer Spine (2 beats)
  // --------------------------------------------------------------------------
  {
    index: 0,
    id: '1.1',
    chapterIndex: 0,
    beatInChapter: 1,
    chapterBeatCount: 2,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Hybrid 52-Layer Spine',
    badge: '23 M · 23 E · 6 *',
    mode: 'tower',
    section: 'overview',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: null,
    paragraphs: [
      'NVIDIA Nemotron 3 Nano 30B-A3B stacks 52 sequential layers along one unbroken 2,688-channel emerald residual spine: 23 Mamba-2 state-space layers (M), 23 sparse Mixture-of-Experts banks (E), and only 6 Grouped-Query Attention layers (* at bays 06, 13, 20, 27, 34, 43).',
      'Four NVIDIA Figure 2 motif spans organize the architecture: 01–35 [M·E·M·E·M·*·E] ×5, 36–41 [M·E] ×3, 42–44 [M·*·E] ×1, and 45–52 [M·E] ×4 — pairing 31.6B total parameters with just 3.2B active per token (3.6B including embeddings).',
    ],
    takeaways: [
      '52 real sequential bays (23 M · 23 E · 6 *) bridged by one continuous 2,688-channel residual stream',
      '31.6B total params · 3.2B active per token excluding embeddings (3.6B including embeddings)',
    ],
    interactivePrompt: 'Drag to orbit or click any bay · Press Space to continue',
  },
  {
    index: 1,
    id: '1.2',
    chapterIndex: 0,
    beatInChapter: 2,
    chapterBeatCount: 2,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Three Memory Regimes Over One Stream',
    badge: '47.1 MiB · 0 B · +6 KiB/tok',
    mode: 'tower',
    section: 'overview',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: null,
    paragraphs: [
      'Follow the 8-token prompt "How does Nemotron use only six attention layers" through all 52 unified layers: X₀ enters Bay 01 Mamba-2, each layer adds its delta back onto the residual highway, and Bay 52 Y reaches the Output Head.',
      'Only 6 attention bays grow verbatim K/V tracks along +Z (+6 KiB/tok in BF16 vs. +52 KiB/tok for 52-layer GQA — 8.7× less). The 23 Mamba-2 bays hold strictly fixed state (47.1 MiB prod / 69 KiB mini) and the 23 MoE bays store 0 B.',
    ],
    takeaways: [
      '23 Mamba-2 bays: 47.1 MiB fixed state · 23 MoE bays: 0 B sequence cache · 6 GQA bays: +6 KiB/tok (BF16)',
      'Cutting attention from 52 layers to 6 reduces per-token KV cache growth by 8.7×',
    ],
    interactivePrompt: 'Inspect the 6 growing KV tracks along +Z · Press Space for Embedding',
  },

  // --------------------------------------------------------------------------
  // ACT I · FULL-SEQUENCE PREFILL — Ch 2: Untied NoPE Embedding (2 beats)
  // --------------------------------------------------------------------------
  {
    index: 2,
    id: '2.1',
    chapterIndex: 1,
    beatInChapter: 1,
    chapterBeatCount: 2,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Untied Token Embedding Lookup',
    badge: '131,072 × 2,688 · Untied',
    mode: 'cutaway_embedding',
    section: 'embedding',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: null,
    stepperStep: 0,
    stepperAutoPlay: false,
    paragraphs: [
      'Every input token ID indexes one exact row of the untied embedding matrix W_embed (131,072 × 2,688 production · 32 × 8 live miniature) and copies it directly into activation deck X₀ — watch token t₇ = "layers" pull its row across.',
      'Because tie_word_embeddings is false, W_embed and the final LM Head keep independent parameter matrices (~352M parameters each in production).',
    ],
    takeaways: [
      'Pure row lookup: X₀[:, t] = W_embed[tokenId(t), :] — zero matrix multiply needed',
      'Untied weights (tie_word_embeddings: false) decouple input geometry from output logits',
    ],
    interactivePrompt: 'Hover any cell in X₀ or W_embed to trace lookup · Press Space',
  },
  {
    index: 3,
    id: '2.2',
    chapterIndex: 1,
    beatInChapter: 2,
    chapterBeatCount: 2,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'NoPE: No Positional Embeddings',
    badge: 'NoPE · X₀ → Bay 01',
    mode: 'cutaway_embedding',
    section: 'embedding',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: null,
    stepperStep: 1,
    stepperAutoPlay: false,
    paragraphs: [
      'Notice the ghosted No Pos Embed slot beside the copy bridge: unlike GPT-2 or RoPE Transformers, Nemotron 3 Nano uses NoPE (zero positional embeddings added here).',
      'The unshifted X₀ activation deck hands off bit-exact onto the emerald residual highway directly into Bay 01 Mamba-2, where causal convolution and selective recurrence encode sequence order naturally.',
    ],
    takeaways: [
      'NoPE eliminates positional tables and RoPE rotations — supporting clean 1M-token length extrapolation',
      'Bit-exact hand-off: Embedding output X₀ === Bay 01 Mamba-2 Input X',
    ],
    interactivePrompt: 'Compare the absent Pos Embed slot · Press Space to enter Bay 01 Mamba-2',
  },

  // --------------------------------------------------------------------------
  // ACT I · FULL-SEQUENCE PREFILL — Ch 3: Mamba-2 Bay 01 (4 beats)
  // --------------------------------------------------------------------------
  {
    index: 4,
    id: '3.1',
    chapterIndex: 2,
    beatInChapter: 1,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'RMSNorm & Fused In-Proj Split',
    badge: 'Bay 01 · Steps 1–2 · W_in',
    mode: 'cutaway_mamba',
    section: 'mamba',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 1,
    mambaBayId: 1,
    stepperStep: 1,
    stepperAutoPlay: true,
    paragraphs: [
      'Bay 01 taps X₀ through Pre-RMSNorm (ε = 1e−5) without altering the emerald bypass highway, producing normalized vector U.',
      'The upright W_in wall projects U into three parallel lanes: bypass gate z (4,096), pre-convolution stream xBC_pre (6,144), and 64 per-head discretization step sizes Δt.',
    ],
    takeaways: [
      'Single fused W_in projection (10,304 prod · 208 mini) splits into z, xBC_pre, and Δt',
      'Per-head Δt controls how much new input enters versus how fast prior state decays',
    ],
    interactivePrompt: 'Hover cells in W_in, z, or Δt · Press Space for Causal Conv1D',
  },
  {
    index: 5,
    id: '3.2',
    chapterIndex: 2,
    beatInChapter: 2,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: '4-Tap Causal Conv1D (Always BF16)',
    badge: 'Bay 01 · Step 3 · k=4 BF16',
    mode: 'cutaway_mamba',
    section: 'mamba',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 1,
    mambaBayId: 1,
    stepperStep: 2,
    stepperAutoPlay: true,
    paragraphs: [
      'A depthwise 4-tap Causal Conv1D (k = 4, 6,144 channels) progressively mixes the rolling causal window t₄..t₇ for token t₇ = "layers" and applies SiLU before splitting into x, B, and C.',
      'This 4-token local filter gives NoPE crisp positional awareness; NVIDIA preserves all 23 Conv1D strips in BF16 (48 KiB/layer) even when FP8 Selective mode is active.',
    ],
    takeaways: [
      'Rolling 4-tap window (t₄..t₇) stays strictly fixed at 48 KiB/layer regardless of context length',
      'Always kept in BF16 across all 23 Mamba-2 bays (NVIDIA §4.2 sensitivity rule)',
    ],
    interactivePrompt: 'Scrub Step 3 to watch lags t₄..t₇ accumulate · Press Space for SSM State',
  },
  {
    index: 6,
    id: '3.3',
    chapterIndex: 2,
    beatInChapter: 3,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Selective SSM Recurrence in Fixed Memory',
    badge: 'Bay 01 · Step 4 · 47.1 MiB Fixed',
    mode: 'cutaway_mamba',
    section: 'mamba',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 1,
    mambaBayId: 1,
    stepperStep: 3,
    stepperAutoPlay: true,
    paragraphs: [
      'Each of the 64 heads (16 in miniature) updates an FP32 state matrix in place via H_t = exp(Δt·A)·H_{t−1} + Δt·x·B, folding new token evidence directly into the existing register.',
      'Because H_t overwrites H_{t−1} instead of appending slices, all 23 Mamba-2 bays combined stay frozen at 47.1 MiB (49,364,992 B) whether generating token 1 or token 1,048,576.',
    ],
    takeaways: [
      'Per Mamba bay: 48 KiB Conv1D (BF16) + 2.00 MiB SSM state (64×64×128 FP32) = 2.05 MiB constant',
      '23 Mamba bays × 2.05 MiB = 47.1 MiB fixed from 1 to 1M tokens',
    ],
    interactivePrompt: 'Hover any cell in H_t to trace H_{t−1} and x·B · Press Space to continue',
  },
  {
    index: 7,
    id: '3.4',
    chapterIndex: 2,
    beatInChapter: 4,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Readout, Gate-First 8-Group Norm & Residual',
    badge: 'Bay 01 · Steps 5–6 · Gate→Norm',
    mode: 'cutaway_mamba',
    section: 'mamba',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 1,
    mambaBayId: 1,
    stepperStep: 5,
    stepperAutoPlay: true,
    paragraphs: [
      'State readout contracts H_t with group vector C and adds the per-head skip connection D·x. Next comes a signature Nemotron detail: norm_before_gate = False — multiply by SiLU(z) first, then apply 8-Group RMSNorm (ε = 1e−5).',
      'W_out projects the normalized channels back to hidden width 2,688 and adds the delta onto the emerald spine, handing Bay 01 Y bit-exact into Bay 02 MoE as X.',
    ],
    takeaways: [
      'Exact order (MambaRMSNormGated): y_gate = (H_t·C + D·x) · SiLU(z) → 8-Group RMSNorm → W_out',
      'Bit-exact cross-bay continuity: Bay 01 Output Y === Bay 02 MoE Input X',
    ],
    interactivePrompt: 'Compare Steps 5 & 6 in the Math Dock · Press Space for Bay 02 MoE',
  },

  // --------------------------------------------------------------------------
  // ACT I · FULL-SEQUENCE PREFILL — Ch 4: MoE 128+1 Bay 02 (4 beats)
  // --------------------------------------------------------------------------
  {
    index: 8,
    id: '4.1',
    chapterIndex: 3,
    beatInChapter: 1,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Parallel Fork: U Feeds Experts, Router Only Scales',
    badge: 'Bay 02 · Step 1 · U = RMSNorm(X)',
    mode: 'cutaway_moe',
    section: 'moe',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 2,
    moeBayId: 2,
    stepperStep: 0,
    stepperAutoPlay: true,
    paragraphs: [
      'Bay 02 receives Bay 01 Y bit-exact and applies Pre-RMSNorm (ε = 1e−5) to produce U = RMSNorm(X), which immediately forks in parallel into both the sparse 128-expert path and the always-on 2× Shared Expert.',
      'Crucial distinction: expert MLPs (W_up → ReLU² → W_down) consume U directly — router weights never alter expert inputs; they strictly select which 6 experts execute and scale their summed outputs (Σ₆ wᵢ·yᵢ).',
    ],
    takeaways: [
      'Normalized hidden vector U = RMSNorm(X) feeds BOTH routed expert MLPs and the 2× Shared Expert',
      'Router weights strictly select the 6 active experts and scale their summed outputs (Σ₆ wᵢ·yᵢ)',
    ],
    interactivePrompt: 'Trace U forking into Router, Routed Experts & Shared · Press Space',
  },
  {
    index: 9,
    id: '4.2',
    chapterIndex: 3,
    beatInChapter: 2,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'FP32 Router + Correction Bias Pick Top-6',
    badge: 'Bay 02 · Step 2 · 6 Active / 122 Idle',
    mode: 'cutaway_moe',
    section: 'moe',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 2,
    moeBayId: 2,
    stepperStep: 1,
    stepperAutoPlay: true,
    paragraphs: [
      'Inside the sparse branch, an FP32 router scores U against all 128 experts via sigmoid(W_router·U) + e_score_correction_bias (n_group = 1, topk_group = 1).',
      'Only the Top-6 tiles rise active across the 8×16 bank; the other 122 experts (95.3%) stay completely idle with zero FLOPs and zero output contribution.',
    ],
    takeaways: [
      '29.8B of 31.6B total params live in 23 MoE bays, yet 95.3% of experts (122/128) stay idle per token',
      'e_score_correction_bias balances expert utilization during selection without touching MLP weights',
    ],
    interactivePrompt: 'Hover risen active tiles vs. 122 idle slots · Press Space for Top-6 Weights',
  },
  {
    index: 10,
    id: '4.3',
    chapterIndex: 3,
    beatInChapter: 3,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Bias Dropped: Unbiased σ Normalized ×2.5',
    badge: 'Bay 02 · Step 3 · Unbiased σ ×2.5',
    mode: 'cutaway_moe',
    section: 'moe',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 2,
    moeBayId: 2,
    stepperStep: 2,
    stepperAutoPlay: true,
    paragraphs: [
      'The moment the 6 winners are chosen, e_score_correction_bias is dropped so load-balancing offsets never distort output magnitudes.',
      'Only the 6 unbiased sigmoid scores σ_j enter the 6-term denominator and scale by routed_scaling_factor = 2.5: w_j = (σ_j / Σ₆ σ_m) × 2.5.',
    ],
    takeaways: [
      'Selection uses biased choice scores (σ + bias); weighting uses unbiased σ over the 6-term sum',
      'routed_scaling_factor = 2.5 scales only the 6 routed expert outputs — never the Shared Expert',
    ],
    interactivePrompt: 'Hover any Top-6 weight slot to verify the 6-term denominator · Press Space',
  },
  {
    index: 11,
    id: '4.4',
    chapterIndex: 3,
    beatInChapter: 4,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Non-Gated ReLU² Experts, 2× Shared & 0 B Cache',
    badge: 'Bay 02 · Steps 4–6 · 0 B Cache',
    mode: 'cutaway_moe',
    section: 'moe',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: 8,
    selectedBay: 2,
    moeBayId: 2,
    stepperStep: 5,
    stepperAutoPlay: true,
    paragraphs: [
      'Each active expert runs a non-gated squared-ReLU MLP directly on U (W_up → ReLU² → W_down, width 1,856 — two matrices, zero gate_proj) and scales its output by wᵢ into the 6-branch sum Σ₆ wᵢ·yᵢ.',
      'In parallel, the always-on double-width Shared Expert (3,712 = 2 × 1,856, unscaled by 2.5) runs on U and merges with Σ₆ wᵢ·yᵢ onto the emerald residual spine while allocating strictly 0 B of sequence cache.',
    ],
    takeaways: [
      'Non-gated ReLU² (only W_up + W_down, no gate_proj) + unscaled 2× Shared Expert on U',
      'MoE is 100% per-token stateless: 0 B sequence cache across all 23 MoE bays at any context length',
    ],
    interactivePrompt: 'Click any active expert tile or scrub Steps 4–6 · Press Space for Bay 06 GQA',
  },

  // --------------------------------------------------------------------------
  // ACT I · FULL-SEQUENCE PREFILL — Ch 5: GQA Attention Bay 06 (4 beats)
  // --------------------------------------------------------------------------
  {
    index: 12,
    id: '5.1',
    chapterIndex: 4,
    beatInChapter: 1,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'NoPE & 16:1 Query-to-KV Asymmetry',
    badge: 'Bay 06 · 32Q / 2KV · T=8',
    mode: 'cutaway_gqa',
    section: 'gqa',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: TOUR_PREFILL_TOKENS,
    selectedBay: 6,
    gqaBayId: 6,
    gqaHead: 0,
    stepperStep: 1,
    stepperAutoPlay: true,
    paragraphs: [
      'Only six layers perform exact token-to-token retrieval across our 8-token prompt (t₀..t₇, T = 8). Entering Bay 06 from Bay 05 Mamba-2, Pre-RMSNorm feeds Q/K/V projections directly under NoPE — zero positional rotations needed.',
      'Contrast the projection geometry: upright W_Q emits 32 Query heads (4,096 channels), whereas W_K and W_V emit just 2 shared KV heads (256 each) — a 16:1 width asymmetry.',
    ],
    takeaways: [
      '32 Q heads (head_dim 128) share just 2 KV heads — shrinking per-layer KV storage 16×',
      'Preceding Mamba-2 feeder Bay 05 supplies positional context so GQA operates cleanly with NoPE at T = 8',
    ],
    interactivePrompt: 'Compare wide W_Q against narrow W_K / W_V · Press Space to compare Q00 vs Q16',
  },
  {
    index: 13,
    id: '5.2',
    chapterIndex: 4,
    beatInChapter: 2,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: '16:1 Verbatim KV Sharing: Q00→KV₀ vs. Q16→KV₁',
    badge: 'Bay 06 · Q16 → KV₁ · T=8',
    mode: 'cutaway_gqa',
    section: 'gqa',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: TOUR_PREFILL_TOKENS,
    selectedBay: 6,
    gqaBayId: 6,
    gqaHead: 16,
    stepperStep: 1,
    stepperAutoPlay: false,
    paragraphs: [
      'Watch the highlight shift from Group 0 (Q00..Q15 → KV₀) onto Group 1 (Q16..Q31 → KV₁): sixteen independent Query heads read the exact same verbatim K and V token stack across t₀..t₇.',
      'Every token still appends one verbatim K and V slice along +Z, but storing only 2 KV heads keeps each attention bay to 1 KiB/token in BF16 (8 KiB in Bay 06 at T = 8 · 0.5 KiB/token in FP8).',
    ],
    takeaways: [
      'Group 0 (Q00..Q15) shares KV₀; Group 1 (Q16..Q31) shares KV₁ verbatim without duplicating keys',
      '32 specialized query patterns at 1/16th the KV memory cost of Multi-Head Attention (8 KiB in Bay 06 at T = 8)',
    ],
    interactivePrompt: 'Click any head comb (Q00..Q31) to flip KV₀ ↔ KV₁ · Press Space for QKᵀ/√d',
  },
  {
    index: 14,
    id: '5.3',
    chapterIndex: 4,
    beatInChapter: 3,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Progressive Causal QKᵀ/√d & Row Softmax',
    badge: 'Bay 06 · Steps 3–4 · Causal −∞',
    mode: 'cutaway_gqa',
    section: 'gqa',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: TOUR_PREFILL_TOKENS,
    selectedBay: 6,
    gqaBayId: 6,
    gqaHead: 0,
    stepperStep: 2,
    stepperAutoPlay: true,
    paragraphs: [
      'Watch the 8 × 8 Scores sheet populate cell-by-cell across causal keys k ≤ q via dot(Q_h[:, q], K_g[:, k]) / √d_h, while every future position k > q stays strictly masked at −∞.',
      'Row Softmax then normalizes each causal row — including query t₇ = "layers" attending back over t₀..t₇ — into attention probabilities summing to 1.000.',
    ],
    takeaways: [
      'Causal mask forces upper-triangle keys (k > q) to −∞ so future tokens contribute 0.000 probability',
      'Progressive cell accumulation reveals each dot product live from its exact Q and shared K operands',
    ],
    interactivePrompt: 'Hover any cell in Scores or scrub 0→100% · Press Space to continue',
  },
  {
    index: 15,
    id: '5.4',
    chapterIndex: 4,
    beatInChapter: 4,
    chapterBeatCount: 4,
    act: 1,
    actLabel: ACT_1_LABEL,
    title: 'Weighted Shared V, W_O & 8.7× KV Reduction',
    badge: 'Bay 06 · 48 KiB vs 416 KiB · 8.7×',
    mode: 'cutaway_gqa',
    section: 'gqa',
    executionMode: 'full_sequence',
    precision: 'bf16',
    tokens: TOUR_PREFILL_TOKENS,
    selectedBay: 6,
    gqaBayId: 6,
    gqaHead: 0,
    stepperStep: 5,
    stepperAutoPlay: true,
    paragraphs: [
      'Softmax probabilities blend shared V_g slices into V_out, all 32 heads concatenate (4,096 prod · 128 mini), and W_O projects the retrieved update back onto the emerald residual spine.',
      'Across all 6 attention bays at T = 8 in BF16, verbatim KV tracks cost 48 KiB (8 KiB in Bay 06 · +6 KiB/tok) versus 416 KiB (+52 KiB/tok) for a 52-layer pure GQA Transformer — an 8.7× reduction.',
    ],
    takeaways: [
      'At T = 8 in BF16: 8 KiB in Bay 06 · 48 KiB across 6 GQA bays vs. 416 KiB for 52-layer pure GQA (8.7× smaller)',
      'Zoom into V₀ or Scores at close range to read live Float32 cell numbers directly on the sheet',
    ],
    interactivePrompt: 'Zoom onto V₀ or Scores to inspect cell values · Press Space for Act II (FP8)',
  },

  // --------------------------------------------------------------------------
  // ACT II · PRECISION BOUNDARY — Ch 6: Selective FP8 (2 beats)
  // --------------------------------------------------------------------------
  {
    index: 16,
    id: '6.1',
    chapterIndex: 5,
    beatInChapter: 1,
    chapterBeatCount: 2,
    act: 2,
    actLabel: ACT_2_LABEL,
    title: 'Gold Preservation Boundary: 12 BF16 Bays',
    badge: 'Bays 42–43 · 12 BF16 + 23 Conv1D',
    mode: 'tower',
    section: 'kv_fp8',
    executionMode: 'full_sequence',
    precision: 'fp8_selective',
    tokens: TOUR_PREFILL_TOKENS,
    selectedBay: 42,
    paragraphs: [
      'Selective FP8 is now active over our same 8-token prompt (t₀..t₇): watch NVIDIA §4.2–4.3’s sensitivity boundary glow gold around Bay 42 (Mamba feeder) and Bay 43 (GQA). Quantizing all 52 layers uniformly degrades long-context attention.',
      'To protect fidelity, NVIDIA keeps BF16 compute on exactly 12 bays — all 6 GQA layers (06, 13, 20, 27, 34, 43) plus their 6 preceding Mamba-2 feeder bays (05, 12, 19, 26, 33, 42) — and preserves all 23 Conv1D strips in BF16 with FP32 SSM states.',
    ],
    takeaways: [
      '12 gold-rimmed BF16 bays: 6 GQA + 6 immediately preceding Mamba-2 feeder bays (05..42)',
      'All 23 Conv1D strips remain BF16 and all 23 SSM recurrent states remain FP32',
    ],
    interactivePrompt: 'Inspect gold-rimmed Bay 42 → Bay 43 boundary · Press Space for FP8 KV tracks',
  },
  {
    index: 17,
    id: '6.2',
    chapterIndex: 5,
    beatInChapter: 2,
    chapterBeatCount: 2,
    act: 2,
    actLabel: ACT_2_LABEL,
    title: 'Frosted FP8 KV Tracks & ~99% Accuracy Recovery',
    badge: 'FP8 KV · 24 KiB vs 48 KiB · ~99%',
    mode: 'tower',
    section: 'kv_fp8',
    executionMode: 'full_sequence',
    precision: 'fp8_selective',
    tokens: TOUR_PREFILL_TOKENS,
    selectedBay: 43,
    paragraphs: [
      'The remaining 40 bays (17 Mamba + 23 MoE) and all 6 KV caches quantize cleanly to FP8 — halving linear KV cache growth from +6 KiB/tok to +3 KiB/tok (24 KiB FP8 vs. 48 KiB BF16 across T = 8 tokens) even beside BF16 attention bodies.',
      'This targeted boundary recovers ~99% median BF16 accuracy across reasoning benchmarks (MMLU-Pro 78.30 → 78.10 · AIME25 w/ tools 99.17 → 98.80).',
    ],
    takeaways: [
      'At T = 8: FP8 KV drops all 6 attention bays to 24 KiB (+3 KiB/tok) vs. 48 KiB (+6 KiB/tok) in BF16',
      '~99% median BF16 accuracy recovery: MMLU-Pro 78.30 → 78.10 · AIME25 w/ tools 99.17 → 98.80',
    ],
    interactivePrompt: 'Toggle BF16 ↔ FP8 Selective below to compare KV bytes · Press Space for Act III',
  },

  // --------------------------------------------------------------------------
  // ACT III · LIVE CACHED AR DECODE & 1M SCALING — Ch 7 (4 beats)
  // --------------------------------------------------------------------------
  {
    index: 18,
    id: '7.1',
    chapterIndex: 6,
    beatInChapter: 1,
    chapterBeatCount: 4,
    act: 3,
    actLabel: ACT_3_LABEL,
    title: 'Final RMSNorm, Untied LM Head & Top-5 Strip',
    badge: 'Bay 52 Y · t₇ "layers" → t₈ Top-5',
    mode: 'cutaway_output',
    section: 'output',
    executionMode: 'full_sequence',
    precision: 'fp8_selective',
    tokens: TOUR_PREFILL_TOKENS,
    selectedBay: null,
    stepperStep: 0,
    stepperAutoPlay: false,
    arAction: 'ensure_prefill',
    paragraphs: [
      'At the bottom of the 52-layer tower, Bay 52 MoE’s emerald residual stream Y for final prompt token t₇ = "layers" passes through Final RMSNorm (ε = 1e−5) and projects through the untied W_lm_head (2,688 → 131,072 prod · 8 → 32 live mini).',
      'Raw vocabulary logits pass through Softmax to produce the exact Top-5 next-token distribution for position t₈ — the #1 greedy candidate shown here is the exact token Beat 7.2 will emit via cached AR decode.',
    ],
    takeaways: [
      'Bay 52 Y(t₇="layers") → Final RMSNorm (ε = 1e−5) → Untied W_lm_head (≠ W_embed) → Softmax → Top-5 for t₈',
      '8-token prefill (t₀..t₇) finishes synced across all 52 layers — ready to emit t₈ directly from cache',
    ],
    interactivePrompt: 'Hover Logits or elevated Top-5 Probs bars for t₈ · Press Space for Live AR Decode',
  },
  {
    index: 19,
    id: '7.2',
    chapterIndex: 6,
    beatInChapter: 2,
    chapterBeatCount: 4,
    act: 3,
    actLabel: ACT_3_LABEL,
    title: 'Live Cached AR Decode Step & 52-Layer Spine Wave',
    badge: 'AR Decode · branch-0 · t₈ (8P+1S)',
    mode: 'tower',
    section: 'overview',
    executionMode: 'ar_decode',
    precision: 'fp8_selective',
    tokens: TOUR_PREFILL_TOKENS + 1,
    selectedBay: null,
    arAction: 'step_once_with_wave',
    paragraphs: [
      'Live cached AR Decode just stepped Beat 7.1’s #1 greedy prediction as token t₈ directly from the 8-token prefill (t₀..t₇, 8P + 1S = 9 total tokens): watch the 52-layer propagation wave sweep down the emerald spine from Embedding through every bay to Output.',
      'Each bay reveals its true scaling law in real time — 23 Mamba-2 bays refresh in place (69 KiB mini / 47.1 MiB prod fixed), 23 MoE bays hold 0 B, and only 6 GQA bays snap +1 new KV slice for t₈ onto +Z.',
    ],
    takeaways: [
      'Single-token step t₈ (8P + 1S = 9) updates 23 fixed Mamba registers (69 KiB mini / 47.1 MiB prod) without growing memory',
      '23 MoE bays add 0 B; only 6 GQA bays append +1 KV slice (+3 KiB/tok FP8 prod · +0.375 KiB/tok mini · 27 KiB at T=9)',
    ],
    interactivePrompt: 'Watch the 52-layer wave land · Press Space to inspect Bay 43 [1 × 9] Cached GQA',
  },
  {
    index: 20,
    id: '7.3',
    chapterIndex: 6,
    beatInChapter: 3,
    chapterBeatCount: 4,
    act: 3,
    actLabel: ACT_3_LABEL,
    title: 'Cached [1 × (P+1)] GQA Decode (Zero Prefix Recompute)',
    badge: 'Bay 43 · AR Decode · [1 × 9]',
    mode: 'cutaway_gqa',
    section: 'gqa',
    executionMode: 'ar_decode',
    precision: 'fp8_selective',
    tokens: TOUR_PREFILL_TOKENS + 1,
    selectedBay: 43,
    gqaBayId: 43,
    gqaHead: 0,
    stepperStep: 2,
    stepperAutoPlay: true,
    arAction: 'inspect_cached_gqa',
    paragraphs: [
      'Inside Bay 43 GQA during cached AR decode, prefill’s square lower triangle collapses into one truthful rectangular [1 × 9] query row ([1 × (P+1)]): only newly generated token t₈ computes Q, K, and V.',
      'The single new query attends across 8 reused cached prefix KV slices (t₀..t₇) plus 1 newly appended KV slice (t₈) — zero prefix recompute, matching full-sequence attention bit-exact (0 ulp).',
    ],
    takeaways: [
      'Rectangular [1 × 9] scores: 8 reused prefix KV slices (t₀..t₇) + 1 new KV slice (t₈) with zero prefix recompute',
      'Bit-exact (0 ulp) parity with full-sequence execution across all 52 layers — contrast Bay 01 carried H_{t−1}',
    ],
    interactivePrompt: 'Scrub the [1 × 9] decode row or inspect Bay 01 H_{t−1} · Press Space',
  },
  {
    index: 21,
    id: '7.4',
    chapterIndex: 6,
    beatInChapter: 4,
    chapterBeatCount: 4,
    act: 3,
    actLabel: ACT_3_LABEL,
    title: 'Fork Branch-1 & 1M-Token Hybrid Scaling',
    badge: 'AR Branching · 3.05 vs 26.0 GiB',
    mode: 'tower',
    section: 'overview',
    executionMode: 'ar_decode',
    precision: 'fp8_selective',
    tokens: TOUR_PREFILL_TOKENS + 1,
    selectedBay: null,
    arAction: 'invite_branch_and_chart',
    highlightMemoryChart: true,
    paragraphs: [
      'Every decode cache snapshot is strictly immutable (copy-on-step): from branch-0 at t₈ (8P + 1S = 9 tokens), click any alternative Top-5 candidate in the AR panel to fork branch-1 and compare divergent generations without mutating branch-0.',
      'Scale this exact hybrid architecture to 1,048,576 tokens and total sequence state reaches just 3.05 GiB in FP8 (6.05 GiB BF16) versus 26.0 GiB FP8 (52.0 GiB BF16) for a 52-layer GQA Transformer — 8.5× smaller.',
    ],
    takeaways: [
      'Immutable copy-on-step cache snapshots enable instant alternative-token branching (branch-0 ↔ branch-1)',
      'At 1M tokens: 3.05 GiB FP8 (6.05 GiB BF16) vs. 26.0 GiB / 52.0 GiB pure GQA (8.5× smaller)',
      'Open Memory Scaling 1 → 1M ↗ (or press M) to inspect crossovers at 1,048 / 2,096 and 8,035 / 16,070 tokens',
    ],
    interactivePrompt: 'Click #2 candidate below to fork branch-1 · Press M for 1M Scaling Chart',
  },
]);

export const WALKTHROUGH_BEAT_COUNT = WALKTHROUGH_BEATS.length;

/**
 * Semantic coverage of stepper steps (0..5) per cutaway beat ID so `findMatchingBeat`
 * maps multi-step beats (e.g. Mamba Steps 1–2 / 5–6, MoE Steps 4–6, GQA Steps 3–4 / 5–6)
 * cleanly without off-by-one nearest-neighbor ambiguity.
 */
const BEAT_STEP_COVERAGE: Readonly<Record<string, readonly number[]>> = Object.freeze({
  '2.1': [0],
  '2.2': [1],
  '3.1': [0, 1],
  '3.2': [2],
  '3.3': [3],
  '3.4': [4, 5],
  '4.1': [0],
  '4.2': [1],
  '4.3': [2],
  '4.4': [3, 4, 5],
  '5.1': [0, 1],
  '5.2': [1],
  '5.3': [2, 3],
  '5.4': [4, 5],
  '7.1': [0],
  '7.3': [0, 1, 2, 3, 4, 5],
});

export function getWalkthroughBeat(index: number): WalkthroughBeat {
  const safe = Number.isFinite(index) ? Math.trunc(index) : 0;
  const clamped = Math.max(0, Math.min(WALKTHROUGH_BEAT_COUNT - 1, safe));
  return WALKTHROUGH_BEATS[clamped];
}

export function nextBeatIndex(current: number): number | null {
  if (!Number.isFinite(current)) return null;
  const next = Math.trunc(current) + 1;
  return next >= 0 && next < WALKTHROUGH_BEAT_COUNT ? next : null;
}

export function prevBeatIndex(current: number): number | null {
  if (!Number.isFinite(current)) return null;
  const prev = Math.trunc(current) - 1;
  return prev >= 0 && prev < WALKTHROUGH_BEAT_COUNT ? prev : null;
}

export function firstBeatIndexForChapter(chapterIndex: number): number {
  const safe = Number.isFinite(chapterIndex) ? Math.trunc(chapterIndex) : 0;
  const clamped = Math.max(0, Math.min(WALKTHROUGH_COUNT - 1, safe));
  const match = WALKTHROUGH_BEATS.find((beat) => beat.chapterIndex === clamped);
  return match ? match.index : 0;
}

export function beatsForChapter(chapterIndex: number): readonly WalkthroughBeat[] {
  if (!Number.isFinite(chapterIndex)) return [];
  const target = Math.trunc(chapterIndex);
  return WALKTHROUGH_BEATS.filter((beat) => beat.chapterIndex === target);
}

export function findMatchingBeat(
  mode: WalkthroughBeatMode,
  stepperStep?: number,
  executionMode?: ExecutionMode,
): WalkthroughBeat | null {
  const modeMatches = WALKTHROUGH_BEATS.filter((beat) => beat.mode === mode);
  if (modeMatches.length === 0) return null;

  const execMatches =
    executionMode !== undefined
      ? modeMatches.filter((beat) => beat.executionMode === executionMode)
      : modeMatches;
  const candidates = execMatches.length > 0 ? execMatches : modeMatches;

  if (stepperStep === undefined || !Number.isFinite(stepperStep)) {
    return candidates[0];
  }

  const targetStep = Math.trunc(stepperStep);

  // 1. Exact primary stepperStep match
  const exact = candidates.find((beat) => beat.stepperStep === targetStep);
  if (exact) return exact;

  // 2. Declared multi-step semantic coverage match (e.g. MoE steps 3..5 -> Beat 4.4)
  const covered = candidates.find((beat) => BEAT_STEP_COVERAGE[beat.id]?.includes(targetStep));
  if (covered) return covered;

  // 3. Nearest beat with a defined stepperStep, otherwise first candidate
  let best: WalkthroughBeat | null = null;
  let bestDist = Infinity;
  for (const beat of candidates) {
    if (beat.stepperStep === undefined) continue;
    const dist = Math.abs(beat.stepperStep - targetStep);
    if (dist < bestDist) {
      bestDist = dist;
      best = beat;
    }
  }
  return best ?? candidates[0];
}

export interface LiveDecodeBeatSummary {
  readonly stepIndex: number;
  readonly totalSteps: number;
  readonly branchId: string;
  readonly prefixTokens: number;
  readonly newTokens: number;
  readonly totalTokens: number;
  readonly lastTokenPos: number;
  readonly lastTokenLabel: string;
  readonly greedyNextLabel: string;
}

export function resolveBeatDisplayCopy(
  beat: WalkthroughBeat,
  liveDecode?: LiveDecodeBeatSummary | null,
): {
  readonly badge: string;
  readonly paragraphs: readonly string[];
  readonly takeaways: readonly string[];
} {
  if (!liveDecode || beat.executionMode !== 'ar_decode') {
    return {
      badge: beat.badge,
      paragraphs: beat.paragraphs,
      takeaways: beat.takeaways,
    };
  }

  const prefixTokens = Math.max(0, Math.trunc(liveDecode.prefixTokens));
  const newTokens = Math.max(1, Math.trunc(liveDecode.newTokens));
  const totalTokens = Math.max(prefixTokens + newTokens, Math.trunc(liveDecode.totalTokens));
  const lastTokenPos = Math.max(0, Math.trunc(liveDecode.lastTokenPos));
  const stepIndex = Math.max(0, Math.trunc(liveDecode.stepIndex));
  const maxStepIndex = Math.max(stepIndex, Math.trunc(liveDecode.totalSteps) - 1);
  const branchId = liveDecode.branchId.trim() || 'branch-0';
  const lastTokenLabel = liveDecode.lastTokenLabel;
  const greedyNextLabel = liveDecode.greedyNextLabel;
  const tokenTag = `t${lastTokenPos} "${lastTokenLabel}"`;
  const nextPos = lastTokenPos + 1;
  const matrixShape = `[${newTokens} × ${totalTokens}]`;
  const stepSplit = `${prefixTokens}P+${newTokens}S`;
  const kvReusePhrase = `${prefixTokens} reused prefix + ${newTokens} new`;
  const prefixSpan =
    prefixTokens > 1 ? `t0..t${prefixTokens - 1}` : prefixTokens === 1 ? 't0' : 'no cached prefix';
  const kvFp8KiB = totalTokens * 3;

  if (beat.id === '7.2') {
    return {
      badge: `AR Decode · ${branchId} · ${tokenTag} · ${stepSplit}`,
      paragraphs:
        prefixTokens > 0
          ? [
              `Live cached AR Decode on ${branchId} (step ${stepIndex}/${maxStepIndex}) just processed ${tokenTag} using ${kvReusePhrase} (${stepSplit} = ${totalTokens} total tokens, ${matrixShape}): watch the 52-layer propagation wave sweep down the emerald spine from Embedding through every bay to Output.`,
              `Each bay reveals its true scaling law in real time — 23 Mamba-2 bays refresh in place (69 KiB mini / 47.1 MiB prod fixed), 23 MoE bays hold 0 B, and only 6 GQA bays snap +${newTokens} new KV slice onto +Z (${kvFp8KiB} KiB FP8 total). Next greedy candidate for t${nextPos}: "${greedyNextLabel}".`,
            ]
          : [
              `Inspecting prefill step 0/${maxStepIndex} on ${branchId} (${kvReusePhrase} = ${totalTokens} tokens ending at ${tokenTag}, ${matrixShape}): all 52 layers are primed to emit the next token directly from cache.`,
              `Each bay reveals its true scaling law — 23 Mamba-2 bays hold fixed recurrent state (69 KiB mini / 47.1 MiB prod), 23 MoE bays store 0 B, and 6 GQA bays populate ${totalTokens} verbatim KV slices (${kvFp8KiB} KiB FP8). Next greedy candidate for t${nextPos}: "${greedyNextLabel}".`,
            ],
      takeaways: [
        `${branchId} step ${stepIndex}/${maxStepIndex} (${tokenTag}, ${stepSplit} = ${totalTokens} tokens): 23 fixed Mamba registers (69 KiB mini / 47.1 MiB prod) update in place without growing memory`,
        `23 MoE bays add 0 B; only 6 GQA bays append KV slices (+3 KiB/tok FP8 prod · +0.375 KiB/tok mini · ${kvFp8KiB} KiB at T=${totalTokens}) · next greedy t${nextPos}: "${greedyNextLabel}"`,
      ],
    };
  }

  if (beat.id === '7.3') {
    return {
      badge: `Bay 43 · ${branchId} · ${matrixShape}`,
      paragraphs:
        prefixTokens > 0
          ? [
              `Inside Bay 43 GQA on ${branchId} during cached AR decode, prefill’s square lower triangle collapses into one truthful rectangular ${matrixShape} query row: only newly generated ${tokenTag} computes Q, K, and V.`,
              `That single query attends across ${kvReusePhrase} KV slices (${prefixSpan} reused + t${lastTokenPos} appended, ${totalTokens} total) — zero prefix recompute, matching full-sequence attention bit-exact (0 ulp) before predicting t${nextPos} "${greedyNextLabel}".`,
            ]
          : [
              `Inside Bay 43 GQA on ${branchId} at prefill step 0, attention runs as a full causal ${matrixShape} sheet across ${newTokens} tokens ending at ${tokenTag} (${kvReusePhrase}).`,
              `Step forward to emit t${nextPos} "${greedyNextLabel}" and watch this triangle collapse into a single rectangular [1 × ${totalTokens + 1}] cached decode row with zero prefix recompute (0 ulp parity).`,
            ],
      takeaways: [
        `Rectangular ${matrixShape} scores on ${branchId}: ${kvReusePhrase} KV slices (${totalTokens} total, zero prefix recompute)`,
        `Bit-exact (0 ulp) parity with full-sequence execution for ${tokenTag} across all 52 layers · next greedy t${nextPos}: "${greedyNextLabel}"`,
      ],
    };
  }

  if (beat.id === '7.4') {
    return {
      badge: `${branchId} · ${tokenTag} · 3.05 vs 26.0 GiB`,
      paragraphs: [
        `Every decode cache snapshot is strictly immutable (copy-on-step): on ${branchId} at ${tokenTag} (${kvReusePhrase} = ${totalTokens} tokens, ${matrixShape}), greedy continues with t${nextPos} "${greedyNextLabel}" — click any alternative Top-5 candidate in the AR panel to fork a new branch without mutating prior snapshots.`,
        `Scale this exact hybrid architecture to 1,048,576 tokens and total sequence state reaches just 3.05 GiB in FP8 (6.05 GiB BF16) versus 26.0 GiB FP8 (52.0 GiB BF16) for a 52-layer GQA Transformer — 8.5× smaller.`,
      ],
      takeaways: [
        `Active snapshot: ${branchId} step ${stepIndex}/${maxStepIndex} (${tokenTag}, ${stepSplit} = ${totalTokens} tokens, next greedy "${greedyNextLabel}") — copy-on-step keeps branches isolated`,
        `At 1M tokens: 3.05 GiB FP8 (6.05 GiB BF16) vs. 26.0 GiB / 52.0 GiB pure GQA (8.5× smaller)`,
        `Open Memory Scaling 1 → 1M ↗ (or press M) to inspect crossovers at 1,048 / 2,096 and 8,035 / 16,070 tokens`,
      ],
    };
  }

  return {
    badge: beat.badge,
    paragraphs: beat.paragraphs,
    takeaways: beat.takeaways,
  };
}
