/**
 * 4-Act Isolated Pedagogical Guided Walkthrough for NVIDIA Nemotron-Labs-Dory1.0-4B
 * (`src/dory/story/guidedTour.ts`; re-exported via `src/dory/story/walkthrough.ts`)
 *
 * Implements `DoryWalkthroughModuleContract` (`DORY5` Slice 5B; Audit §6.3, §12.2):
 *   - Strictly isolated private `DorySession`: `createDoryGuidedTour()` NEVER accepts or mutates
 *     any external Free-Explore `DorySession` (`D6`, `DR18`). All 5 deterministic tour scenarios
 *     (`standard_k1_per_loop`, `standard_k2_per_loop`, `standard_k4_per_loop`,
 *     `standard_k4_last_loop_chunked`, `self_spec_contract_a`) execute exclusively on the tour's
 *     own private session using shared read-only singleton parameters (`0` weight cloning).
 *   - Authoritative 4-Act, 14-beat (`3 + 3 + 4 + 4`) narrative across all 7 `D7` spatial views:
 *       * Act I   — Hypersphere Foundations (`bookends`, `macro`, `hypersphere`)
 *       * Act II  — Inside the Normalized Mixers (`gqa_rope` SWA & Full, `swiglu`)
 *       * Act III — Shared-Weight Latent Recurrence & Dual-Axis KV (`macro` unrolled, `hypersphere` PCA + counterfactual, `kv_compare`)
 *       * Act IV  — Elastic Depth, Paper §6.2 Self-Speculation & Production Context (`self_spec`, `bookends`)
 *   - Full immutable 8-tuple `DoryExecutionCoord` (`Audit §6.3`) resolved from real reconstructed
 *     sublayer traces and verified via `validateDoryExecutionCoord()` on every beat.
 *   - 100% live scalar readouts computed from actual miniature tensors, traces, and records, paired
 *     with mandatory epistemic disclosures (`DR7` anti-SLERP, `DR17` partition divergence, `DR23`
 *     Contract A §6.2 reconstruction, `DORY_EXPLORATORY_PCA_BADGE`, `DORY_ANCHOR_MAP_CAVEAT_BADGE`,
 *     `DORY_COUNTERFACTUAL_CODA_BADGE`, `DORY_TBPTL_READONLY_BADGE`, `DORY_PAPER_RESULTS_BADGE`).
 *   - Strictly `D4`-compliant: ZERO imports from Gate 4b/4c Nano modules, sibling apps, DOM, WebGL,
 *     Node built-ins, or ambient RNG.
 */

import { fmtSigned, formatBytesAdaptive, formatKiB, formatMiB } from '../../shared/format';
import {
  argmaxIndex,
  effectiveLogitScale,
  effectiveSGateSquared,
  effectiveSqkSquared,
  effectiveSuvScale,
  l2Distance,
  softmaxFloat64,
  vectorL2Norm,
} from '../ngptMath';
import { getDefaultDoryModelParameters } from '../parameters';
import { deriveDorySeed } from '../prng';
import { buildDoryDualRopeTables } from '../rope';
import { buildDoryGqaRopeScene } from '../scene/gqaRopeScene';
import {
  buildDoryHypersphereScene,
  computeKlDivergenceFloat64,
} from '../scene/hypersphereScene';
import { buildDoryKvCompareScene } from '../scene/kvCompareScene';
import { buildDoryMacroScene } from '../scene/macroScene';
import {
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_TBPTL_READONLY_BADGE,
  resolveDorySessionContext,
} from '../scene/provenance';
import { buildDorySelfSpecScene } from '../scene/selfSpecScene';
import { buildDorySwigluScene } from '../scene/swigluScene';
import {
  DEFAULT_DORY_SPATIAL_FOCUS_STATE,
  DORY_SPATIAL_VIEW_MODES,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DorySpatialFocusState,
  type DorySpatialScene,
  type DorySpatialViewMode,
} from '../scene/types';
import { createDorySession } from '../session';
import {
  DORY_CHECKPOINT_FORMATS,
  DORY_PRODUCTION_SPEC,
  computeEffectivePairedDepth,
  computeExecutedDepth,
  computeExecutedLayerIdx,
  computeKvBankCounts,
  computeMiniKvCacheBytes,
  computeProductionKvBytesBf16,
  decodeDoryToken,
  getDoryPhysicalLayerSpec,
  getDoryStage,
} from '../spec';
import {
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_DEFAULT_GREEDY_SAMPLING,
  DORY_MINI_CONFIG,
  DORY_MINI_FLOAT_COUNT,
  DORY_MINI_TENSOR_COUNT,
  DORY_MINI_WEIGHT_BYTES,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  DORY_SPEC_DRAFT_LOOPS,
  DORY_SPEC_VERIFY_LOOPS,
  type DoryAttentionSublayerTrace,
  type DoryAttentionSublayerWeights,
  type DoryCacheRole,
  type DoryDualRopeTables,
  type DoryExecutionCoord,
  type DoryKvBankKind,
  type DoryKvCacheMode,
  type DoryMlpSublayerTrace,
  type DoryMlpSublayerWeights,
  type DoryModelParameters,
  type DorySamplingConfig,
  type DorySession,
  type DorySessionMemoryAudit,
} from '../types';
import {
  DORY_PAPER_RESULTS_BADGE,
  DORY_TOUR_ACT_DESCRIPTORS,
  DORY_TOUR_MIN_BEATS_PER_ACT,
  DORY_TOUR_SEED,
  DORY_TOUR_TOTAL_ACTS,
  type DoryExecutionCoordValidationResult,
  type DoryFreeExploreRestoreState,
  type DoryGuidedTourController,
  type DoryTourActDescriptor,
  type DoryTourActId,
  type DoryTourActSummary,
  type DoryTourBeatSnapshot,
  type DoryTourBeatSpec,
  type DoryTourLiveMetric,
  type DoryTourOptions,
  type DoryTourScenarioKey,
  type DoryTourState,
  type DoryTourStatus,
  type DoryWalkthroughModuleContract,
} from './types';

export {
  DORY_PAPER_RESULTS_BADGE,
  DORY_TOUR_ACT_DESCRIPTORS,
  DORY_TOUR_MIN_BEATS_PER_ACT,
  DORY_TOUR_SEED,
  DORY_TOUR_TOTAL_ACTS,
};

// ============================================================================
// DETERMINISTIC TOUR PROMPT & SCENARIO CONSTANTS (CROSSES W = 8 EVICTION)
// ============================================================================

/**
 * Deterministic 12-token non-EOS tour prompt (`S = 12 > W_mini = 8`, `< S_max = 32`):
 * `['<s>', '<|im_start|>', 'user', '\n', 'solve', 'loop', 'depth', '4', 'sphere', 'anchor', 'cache', 'verify']`
 * Crosses the `W = 8` SWA eviction boundary immediately and matches the Audit §7.4 `[4, 8]`
 * partition-divergence fixture geometry while leaving room for decode steps & `gamma = 4` cycles.
 */
export const DORY_TOUR_PROMPT_TOKENS: readonly number[] = Object.freeze([
  1, 10, 4, 18, 40, 25, 26, 57, 29, 30, 31, 37,
]);

/** Canonical `[4, 8]` chunked prefill partition for `standard_k4_last_loop_chunked` (`DR17`/`DR24`). */
const DORY_TOUR_LAST_LOOP_PARTITION: readonly number[] = Object.freeze([4, 8]);

/**
 * Sharp stochastic sampling configuration for the tour's Contract A self-speculation branch
 * (`T = 0.2, topP = 0.95`) so Cycle 1 showcases full acceptance + 1-token Shallow Coda catch-up
 * while Cycle 2 showcases live division-free binary64 rejection, residual `Z > 0`, and atomic
 * `cap_swa = 16` rollback across `L > W = 8`.
 */
const DORY_TOUR_SPEC_SAMPLING: DorySamplingConfig = Object.freeze({
  temperature: 0.2,
  topK: 0,
  topP: 0.95,
});

/** Valid KV cache policies, bank kinds, and cache roles for strict coordinate validation. */
const VALID_KV_POLICIES = new Set<DoryKvCacheMode>(['per_loop', 'last_loop']);
const VALID_BANK_KINDS = new Set<DoryKvBankKind>([
  'input',
  'recurrent',
  'output_deep',
  'output_shallow',
]);
const VALID_CACHE_ROLES = new Set<DoryCacheRole>([
  'standard',
  'spec_draft',
  'spec_verify_reused',
  'spec_verify_deep',
]);

// ============================================================================
// 4-ACT, 14-BEAT PEDAGOGICAL WALKTHROUGH SPECIFICATION (`DORY_TOUR_BEATS`)
// ============================================================================

function freezeBeatSpec(spec: DoryTourBeatSpec): DoryTourBeatSpec {
  return Object.freeze({
    ...spec,
    discrepancyRefs: Object.freeze([...spec.discrepancyRefs]),
    designDecisionRefs: Object.freeze([...spec.designDecisionRefs]),
    coordTarget: Object.freeze({ ...spec.coordTarget }),
    focusPatch: Object.freeze({
      ...spec.focusPatch,
      ...(spec.focusPatch.kvCompareChunkPartition != null
        ? { kvCompareChunkPartition: Object.freeze([...spec.focusPatch.kvCompareChunkPartition]) }
        : {}),
      ...(spec.focusPatch.selfSpecPreviewSampling != null
        ? { selfSpecPreviewSampling: Object.freeze({ ...spec.focusPatch.selfSpecPreviewSampling }) }
        : {}),
    }),
    cameraOverride: spec.cameraOverride ? Object.freeze({ ...spec.cameraOverride }) : null,
  });
}

/**
 * Authoritative 4-Act, 14-beat (`3 + 3 + 4 + 4`) guided walkthrough covering all 7 `D7` views,
 * all 5 isolated tour scenarios, and the full immutable 8-tuple `DoryExecutionCoord` (`Audit §6.3`).
 */
export const DORY_TOUR_BEATS: readonly DoryTourBeatSpec[] = Object.freeze([
  // --------------------------------------------------------------------------
  // ACT I — HYPERSPHERE FOUNDATIONS (Views 7, 1, 2 · 3 Beats)
  // --------------------------------------------------------------------------
  freezeBeatSpec({
    index: 0,
    id: 'act1-untied-unit-sphere-bookends',
    actId: 'act1',
    actNumber: 1,
    actIndex: 0,
    beatInAct: 1,
    title: 'Untied Tokens on the Unit Hypersphere',
    subtitle: 'Unit-norm embeddings enter Layer 0 directly; untied LM head applies FP32 logit_scale / 0.0177',
    takeaway:
      'Every residual state in Dory1.0-4B lives on the unit hypersphere S^(d-1) with zero learned norm weights, zero linear biases, and untied input/output vocab matrices.',
    narrative:
      'Dory1.0-4B keeps backbone.embeddings.weight and lm_head.weight strictly untied (tie_word_embeddings=false, DR1). Following vLLM, llama.cpp, and Paper TikZ (DR5), raw unit-norm synthetic embeddings enter physical Layer 0 directly without pre-normalization, while the output LM head separates raw logits from FP32 scaling z_scaled = z_raw * (logit_scale / 0.0177) before binary64 softmax.',
    equation: 'h_0 = E[tok] (||E[v]||_2 = 1),   z_scaled = (h_out W_lm^T) ⊙ (s_z / 0.0177)',
    epistemicNote:
      'Miniature V=64, d=8 preserves exact production special token IDs (0..2, 10..17), untied matrices, and unmodified sigma_0=0.0177; production uses V=131,072, d=2,560.',
    discrepancyRefs: ['DR1', 'DR5', 'DR9'],
    designDecisionRefs: ['D2', 'D5', 'D7'],
    viewMode: 'bookends',
    scenarioKey: 'standard_k2_per_loop',
    kLoops: 2,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 0,
      loopIteration: 0,
      tokenOffset: 'last',
      expectedBankKind: 'input',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      selectedPhysicalLayerIdx: 0,
      selectedLoopIteration: 0,
      selectedCheckpointFormat: 'bf16',
      hoveredPartId: 'bookend_embed_weight',
    },
    highlightPartId: 'bookend_embed_weight',
    cameraOverride: { azimuthDeg: -28, elevationDeg: 26 },
  }),

  freezeBeatSpec({
    index: 1,
    id: 'act1-folded-72l-topology',
    actId: 'act1',
    actNumber: 1,
    actIndex: 0,
    beatInAct: 2,
    title: 'Physical 72-Sublayer Topology (16 + 40 + 16)',
    subtitle: 'Repeating 8-sublayer 3:1 SWA:Full motif across Prelude, Shared Recurrent Core, and Coda',
    takeaway:
      'Dory executes 72 physical single-sublayer blocks (16 Prelude + 40 Shared Recurrent Core + 16 Coda) with zero loop adapters and an exact 3:1 SWA:Full ratio in every stage.',
    narrative:
      'The folded macro architecture reveals Dory’s canonical physical-72 indexing (D1, DR4): even indices 0..70 are attention sublayers and odd indices 1..71 are SwiGLU MLPs, matching the 507-tensor HF checkpoint keys (with Pair L{p>>1} badges mapping to transformers’ 36 paired layers). At default k=2, the 40-sublayer core executes twice for 112 executed sublayers (56 paired layers) without cloning a single weight.',
    equation: 'D_sub(k) = 16 + 40k + 16 = 32 + 40k   (k=2 -> 112 sublayers, 507 physical tensors)',
    epistemicNote:
      'Physical 72-sublayer topology and 507 tensor count match production BF16 1:1; miniature d=8 matches the 8-channel sheet grammar of all nemotron-viz apps (D2).',
    discrepancyRefs: ['DR2', 'DR4', 'DR12'],
    designDecisionRefs: ['D1', 'D2', 'D7'],
    viewMode: 'macro',
    scenarioKey: 'standard_k2_per_loop',
    kLoops: 2,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 16,
      loopIteration: 1,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      macroDisplayMode: 'folded',
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 1,
      hoveredPartId: 'phys_l16',
    },
    highlightPartId: 'phys_l16',
    cameraOverride: { azimuthDeg: -34, elevationDeg: 25 },
  }),

  freezeBeatSpec({
    index: 2,
    id: 'act1-ngpt-nlerp-and-five-scales',
    actId: 'act1',
    actNumber: 1,
    actIndex: 0,
    beatInAct: 3,
    title: 'Channel-Wise nGPT NLERP & Active sigma_0 = 0.0177',
    subtitle: 'Anti-SLERP residual interpolation with unclamped |alpha_hat| >= 0 and Exact 2D span(a, b)',
    takeaway:
      'Despite vLLM naming its helper slerp_norm, Dory NEVER computes trigonometric SLERP — it computes unclamped channel-wise NLERP driven by active runtime divisor sigma_0 = 0.0177.',
    narrative:
      'Every sublayer blends normalized residual a = N(x_res) and sublayer output b = N(x_sub) via channel-wise NLERP: N(a + |alpha * (0.05 / 0.0177)| ⊙ (b - a)) (DR7). Because |alpha_hat| is unclamped, channels with |alpha_hat_c| > 1 extrapolate past b before unit-sphere re-projection — directly demonstrated by planted channels [6, 7] at physical layer 16 inside the Exact 2D Spanning-Subspace diagram span(a, b).',
    equation: 'x_next = N_d(a + |alpha · 0.05 / 0.0177| ⊙ (b - a)),   a = N_d(x_res),  b = N_d(x_sub)',
    epistemicNote:
      'Anti-SLERP guard (DR7): channel-wise NLERP — NOT trigonometric SLERP. Channels [6, 7] at p=16 are deliberately planted in (1.05, 1.25) to exercise unclamped >1 extrapolation under D5 FP32-storage/binary64-accum semantics.',
    discrepancyRefs: ['DR7', 'DR9'],
    designDecisionRefs: ['D5', 'D7'],
    viewMode: 'hypersphere',
    scenarioKey: 'standard_k2_per_loop',
    kLoops: 2,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 16,
      loopIteration: 1,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 1,
      includeCounterfactualCoda: false,
      hoveredPartId: 'nlerp_effective_alpha',
    },
    highlightPartId: 'nlerp_effective_alpha',
    cameraOverride: { azimuthDeg: -26, elevationDeg: 28 },
  }),

  // --------------------------------------------------------------------------
  // ACT II — INSIDE THE NORMALIZED MIXERS (Views 3, 4 · 3 Beats)
  // --------------------------------------------------------------------------
  freezeBeatSpec({
    index: 3,
    id: 'act2-gated-gqa-swa-rope',
    actId: 'act2',
    actNumber: 2,
    actIndex: 1,
    beatInAct: 1,
    title: 'Gated 4:1 GQA, Q-Only sqk^2 & Sliding-Window RoPE',
    subtitle: 'Q/Gate expansion, RoPE -> Q/K L2Norm (DR6), live G-channel SWA W=8 mask, sigmoid gate & attn_sub_norm',
    takeaway:
      'Dory expands Q and Gate beyond d, applies NeoX RoPE before per-head Q/K L2Norm, folds squared sqk^2 onto Q only to leave cached K untouched, and normalizes concatenated gated heads before o_proj.',
    narrative:
      'At physical layer 16 (Recurrent SWA, Pair L8, W=8, theta_S=16), bias-free Q and G project 8 -> 64 (8.0x mini vs. 1.60x prod 2560 -> 4096) while K and V form 4:1 GQA (8 Q : 2 KV heads). NeoX rotate_half RoPE runs BEFORE per-head Q/K L2Norm (DR6); Q alone receives squared scale sqk^2 so cached K stays scale-free. Every SWA score/softmax plate binds a live RG32F G-channel mask (DR20), followed by input-dependent sigmoid gating (DR3) and full-concat attn_sub_norm over all 64 dims.',
    equation: 'score = d_h^(-1/2) <sqk^2 ⊙ N(RoPE_S(q)), N(RoPE_S(k))>,   Y = N_{q_dim}(C ⊙ sigma(G ⊙ s_gate^2))',
    epistemicNote:
      'Explicit D2 scaling disclosure: miniature q_dim/d = 8.0x (64/8) and kv_dim/d = 2.0x (16/8) preserve 4:1 GQA & 4 rotary planes; production uses q_dim/d = 1.60x (4096/2560) and kv_dim/d = 0.40x (1024/2560).',
    discrepancyRefs: ['DR3', 'DR6', 'DR20'],
    designDecisionRefs: ['D2', 'D5', 'D7'],
    viewMode: 'gqa_rope',
    scenarioKey: 'standard_k2_per_loop',
    kLoops: 2,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 16,
      loopIteration: 1,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      gqaLayerIdx: 16,
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 1,
      selectedQHead: 0,
      selectedRotaryPlane: 0,
      hoveredPartId: 'gqa_attn_sub_norm',
    },
    highlightPartId: 'gqa_attn_sub_norm',
    cameraOverride: { azimuthDeg: -24, elevationDeg: 32 },
  }),

  freezeBeatSpec({
    index: 4,
    id: 'act2-full-attention-dual-rope',
    actId: 'act2',
    actNumber: 2,
    actIndex: 1,
    beatInAct: 2,
    title: '3:1 Interleaved Full Attention & Long-Range RoPE',
    subtitle: 'Every 4th attention layer switches to Full causal visibility with 64x mini / 1,000x prod theta_F',
    takeaway:
      'Nine physical Full-attention layers (motif slot 6: 6, 14, 22, 30, 38, 46, 54, 62, 70) retain full causal history with slow long-context RoPE frequencies while 27 SWA layers bound local attention to W.',
    narrative:
      'Stepping from SWA layer 16 to Recurrent Full layer 22 (Pair L11, motif slot 6) switches the RoPE base from dyadic theta_S=16 to slow theta_F=1,024 (mirroring production’s 1,000x jump from 10,000 to 10,000,000, DR2) and lifts the sliding window cap so the query attends across every committed token [0..pos] via a live RG32F full-causal mask (DR20).',
    equation: 'omega_j = theta_F^(-2j / d_h),   0 <= pos_k <= pos_q   (Full: theta_F = 1,024 mini / 10^7 prod)',
    epistemicNote:
      'Because cached K is stored post-RoPE and post-L2Norm with layer-specific theta_S / theta_F, llama.cpp disables KV context shifting (DR11).',
    discrepancyRefs: ['DR2', 'DR6', 'DR20'],
    designDecisionRefs: ['D1', 'D2', 'D7'],
    viewMode: 'gqa_rope',
    scenarioKey: 'standard_k2_per_loop',
    kLoops: 2,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 22,
      loopIteration: 1,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      gqaLayerIdx: 22,
      selectedPhysicalLayerIdx: 22,
      selectedLoopIteration: 1,
      selectedQHead: 0,
      selectedRotaryPlane: 3,
      hoveredPartId: 'gqa_probs_h0',
    },
    highlightPartId: 'gqa_probs_h0',
    cameraOverride: { azimuthDeg: -22, elevationDeg: 33 },
  }),

  freezeBeatSpec({
    index: 5,
    id: 'act2-normalized-swiglu-mixer',
    actId: 'act2',
    actNumber: 2,
    actIndex: 1,
    beatInAct: 3,
    title: 'Normalized SwiGLU: Signed suv Scales & ffn_sub_norm',
    subtitle: 'Family 4 scales initialize near 1.0 (not 0.0177) with signed sqrt(d) multiplier and pre-down_proj L2Norm',
    takeaway:
      'SwiGLU is the sole scale family whose raw weights initialize at 1.0 without dividing by sigma_0 = 0.0177 and without taking .abs(), preserving signed gating before ffn_sub_norm.',
    narrative:
      'At odd physical layer 17 (first Recurrent MLP, Pair L8 · MLP), gate_proj and up_proj expand 8 -> 24 (3.00x mini vs. 2.80x prod 2560 -> 7168). Separate signed scales suv_gate * sqrt(d) and suv_up * sqrt(d) modulate the SiLU product, followed by weightless L2Norm (ffn_sub_norm) across all d_ff=24 channels strictly before down_proj and channel-wise NLERP residual blending.',
    equation: 'M = N_{d_ff}(SiLU((h W_g^T) ⊙ (s_{uv,g} sqrt(d))) ⊙ ((h W_u^T) ⊙ (s_{uv,u} sqrt(d)))),  O = M W_d^T',
    epistemicNote:
      'Structured synthetic suv_gate / suv_up intentionally include negative signed entries to verify no spurious .abs() is applied (DR9, D5).',
    discrepancyRefs: ['DR7', 'DR9'],
    designDecisionRefs: ['D2', 'D5', 'D7'],
    viewMode: 'swiglu',
    scenarioKey: 'standard_k2_per_loop',
    kLoops: 2,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 17,
      loopIteration: 1,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      swigluLayerIdx: 17,
      selectedPhysicalLayerIdx: 17,
      selectedLoopIteration: 1,
      swigluChannel: 0,
      swigluOutputDim: 0,
      swigluAccumReveal: 1.0,
      hoveredPartId: 'swiglu_ffn_sub_norm',
    },
    highlightPartId: 'swiglu_ffn_sub_norm',
    cameraOverride: { azimuthDeg: -28, elevationDeg: 28 },
  }),

  // --------------------------------------------------------------------------
  // ACT III — SHARED-WEIGHT LATENT RECURRENCE & DUAL-AXIS KV (Views 1, 2, 5 · 4 Beats)
  // --------------------------------------------------------------------------
  freezeBeatSpec({
    index: 6,
    id: 'act3-unrolled-shared-weights-k4',
    actId: 'act3',
    actNumber: 3,
    actIndex: 2,
    beatInAct: 1,
    title: 'Unrolling Elastic Depth k = 1 -> 4 With Identical Weights',
    subtitle: '192 executed sublayers share one SHA-256 modelWeightId while generating distinct per-loop states & 96 KV banks',
    takeaway:
      'Increasing k from 1 to 4 expands executed depth from 72 to 192 sublayers (2.667x body compute) with zero parameter cloning, while per_loop KV banks scale from 36 to 96.',
    narrative:
      'Switching to the isolated k=4 per_loop branch (DR18: changing k never mutates an existing KV cache — it forks a clean branch) unrolls all four passes of physical core 16..55 across executed indices 16..175. Horizontal provenance rails bind every pass to the identical singleton modelWeightId while the parallel KV rail widens to 96 persistent banks (72 SWA + 24 Full).',
    equation: 'D_sub(4) = 32 + 40·4 = 192L,   Banks_per_loop(4) = 16 + 20·4 = 96,   Ratio = (4 + 5k)/9 = 2.667x',
    epistemicNote:
      'Body compute ratio (4+5k)/9 is NOT an end-to-end latency or throughput multiplier (untied LM head is k-invariant and decode is bandwidth-bound; Audit §6.1).',
    discrepancyRefs: ['DR4', 'DR11', 'DR18'],
    designDecisionRefs: ['D1', 'D6', 'D7'],
    viewMode: 'macro',
    scenarioKey: 'standard_k4_per_loop',
    kLoops: 4,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 16,
      loopIteration: 4,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      macroDisplayMode: 'unrolled',
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 4,
      hoveredPartId: 'exec_e136_p16_t4',
    },
    highlightPartId: 'exec_e136_p16_t4',
    cameraOverride: { azimuthDeg: -30, elevationDeg: 24 },
  }),

  freezeBeatSpec({
    index: 7,
    id: 'act3-anchor-injection-and-pca-trajectory',
    actId: 'act3',
    actNumber: 3,
    actIndex: 2,
    beatInAct: 2,
    title: 'Persistent Anchor u_t = N(s_{t-1} + x) & 8D Latent Trajectory',
    subtitle: 'Literal N(2x) entry, Exploratory 2D PCA with 8D-only metrics, local 120-deg Jacobian caveat & read-only Coda probe',
    takeaway:
      'Before every loop t=1..k, the fixed Prelude anchor x is re-injected via literal u_t = N(s_{t-1} + x) (including u_1 = N(2x)); all trajectory metrics stay strictly in 8D and counterfactual Coda probes mutate zero KV entries.',
    narrative:
      'Tracking {x, u_1, s_1, ..., u_4, s_4} across k=4 shows literal loop-entry anchor injection u_t = N(s_{t-1} + x) and explicit post-loop normalization s_t = N(F_theta(u_t)) (DR8, DR12). The Exploratory 2D PCA inset projects 8D unit vectors (projected norms <= 1) while all numeric cosines, displacements, and anchor angles are computed strictly in 8D. The read-only Diagnostic Counterfactual Coda evaluates s_1 against committed historical Deep Coda KV (writeKv=false) without mutating any cache slot.',
    equation: 'u_t = N_d(s_{t-1} + x),  s_t = N_d(F_theta(u_t)),   ||J_N(s+x)||_2 = 1 / sqrt(4 cos^2(theta/2) + eps)',
    epistemicNote:
      `${DORY_EXPLORATORY_PCA_BADGE}; ${DORY_ANCHOR_MAP_CAVEAT_BADGE}; ${DORY_COUNTERFACTUAL_CODA_BADGE}.`,
    discrepancyRefs: ['DR8', 'DR12'],
    designDecisionRefs: ['D5', 'D7'],
    viewMode: 'hypersphere',
    scenarioKey: 'standard_k4_per_loop',
    kLoops: 4,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 55,
      loopIteration: 4,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 4,
      includeCounterfactualCoda: true,
      counterfactualLoop: 1,
      hoveredPartId: 'latent_trajectory_8d',
    },
    highlightPartId: 'latent_trajectory_8d',
    cameraOverride: { azimuthDeg: -26, elevationDeg: 28 },
  }),

  freezeBeatSpec({
    index: 8,
    id: 'act3-dual-axis-kv-and-last-loop-divergence',
    actId: 'act3',
    actNumber: 3,
    actIndex: 2,
    beatInAct: 3,
    title: 'Dual-Axis KV (per_loop vs. last_loop) & Partition Divergence',
    subtitle: 'last_loop freezes KV at 36 banks by having loops t < k read historical final-loop KV (writerLoop = k)',
    takeaway:
      'For k >= 2, last_loop equals per_loop on empty-cache single-shot prefill (Theorem 1), but is NOT generally invariant to prefill chunk boundaries once historical final-loop KV exists (Theorem 3, DR17).',
    narrative:
      'On our isolated chunked [4, 8] last_loop branch (k=4), loop t=1 of chunk [4..11] reads historical KV authored by loop t=4 (writerLoop=4 != loopIteration=1!) via read-only DoryPastReader semantics (DR10), committing only on t=4 without prematurely advancing SWA ring cursors. Consequently, chunked last_loop diverges materially (> 1e-2) from per_loop and 1-by-1 decode — which is why DorySession records exact chunkSizes[] for 0-ULP lazy trace replay (DR24).',
    equation: 'Logits_last_loop([0..11]) != Logits_last_loop([0..3],[4..11]) != Logits_last_loop([0],[1],...,[11])   (k >= 2)',
    epistemicNote:
      'Theorem 3 (DR17) characterizes generic non-degenerate weights (degenerate W_v=0 counterexamples can still match); changing kvCacheMode always forks a non-aliased branch (DR18).',
    discrepancyRefs: ['DR10', 'DR17', 'DR18', 'DR24'],
    designDecisionRefs: ['D6', 'D7', 'D10'],
    viewMode: 'kv_compare',
    scenarioKey: 'standard_k4_last_loop_chunked',
    kLoops: 4,
    kvCacheMode: 'last_loop',
    executionMode: 'standard',
    targetStepOffset: 1,
    coordTarget: {
      physicalLayerIdx: 16,
      loopIteration: 1,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      kvComparePartitionMode: 'chunked',
      kvCompareLayerIdx: 16,
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 1,
      kvCompareChunkPartition: [4, 8],
      hoveredPartId: 'partition_divergence_matrix',
    },
    highlightPartId: 'partition_divergence_matrix',
    cameraOverride: { azimuthDeg: -22, elevationDeg: 30 },
  }),

  freezeBeatSpec({
    index: 9,
    id: 'act3-swa-eviction-and-256k-scaling',
    actId: 'act3',
    actNumber: 3,
    actIndex: 2,
    beatInAct: 4,
    title: 'SWA Eviction (W = 8) & 1 -> 256K S-Invariant (4+5k)/9 Ratio',
    subtitle: 'Uniform 3:1 SWA:Full across all 3 stages makes per_loop/last_loop strictly independent of context length S',
    takeaway:
      'Crossing S = W_prod = 1,024 drops marginal KV growth by 4x as SWA saturates (75% -> 1.16% at 256K), yet KV_per_loop / KV_last_loop = (4 + 5k) / 9 stays strictly invariant for every S >= 1.',
    narrative:
      'Because Input (6:2), Recurrent (15:5), and Output (6:2) all preserve the exact 3:1 SWA:Full ratio, the (3 min(S, W) + S) context factor cancels identically between per_loop and last_loop at every length S from 1 to 262,144 tokens (2.667x at k=4). In our miniature (committedLen > W=8), persistent SWA banks allocate cap_swa=16 ring slots with per-slot absPosTag (DR19) plus transient workspaces for long chunks L > 16 (DR24).',
    equation: 'KV_per_loop(S, k) / KV_last_loop(S, k) = (4 + 5k) / 9   (invariant to S; 4x marginal slope drop at S > W)',
    epistemicNote:
      'Analytical production BF16 KV scaling reflects logical retained tensor bytes (B=1, 4 KiB/tok/layer), excluding paged-attention block fragmentation.',
    discrepancyRefs: ['DR11', 'DR19', 'DR24'],
    designDecisionRefs: ['D2', 'D7', 'D10'],
    viewMode: 'kv_compare',
    scenarioKey: 'standard_k4_per_loop',
    kLoops: 4,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 16,
      loopIteration: 4,
      tokenOffset: 'last',
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      kvComparePartitionMode: 'single_shot',
      kvCompareLayerIdx: 16,
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 4,
      hoveredPartId: 'swa_ring_tags',
    },
    highlightPartId: 'swa_ring_tags',
    cameraOverride: { azimuthDeg: -20, elevationDeg: 31 },
  }),

  // --------------------------------------------------------------------------
  // ACT IV — ELASTIC DEPTH, PAPER §6.2 SELF-SPECULATION & PRODUCTION CONTEXT (Views 6, 7 · 4 Beats)
  // --------------------------------------------------------------------------
  freezeBeatSpec({
    index: 10,
    id: 'act4-self-spec-contract-a-104-banks',
    actId: 'act4',
    actNumber: 4,
    actIndex: 3,
    beatInAct: 1,
    title: 'Reconstructed Paper §6.2 Self-Speculation: 104 Contract A Banks',
    subtitle: 'Same checkpoint drafts gamma=4 at k=1 and verifies at k=4 reusing Prelude + Loop-1 (x, s_1) & KV',
    takeaway:
      'Absent from upstream runtimes (DR23), our reconstructed Contract A adds 8 dedicated persistent output_shallow Coda KV banks (+20 KiB, 104 total) so shallow drafting matches standalone k=1 Dory.',
    narrative:
      'Paper §6.2 drafts gamma=4 tokens at shallow depth k=1 and verifies batched at deep depth k=4 while sharing Input Prelude (8 banks) and Loop-1 (20 banks) KV caches and reusing cached (x, s_1) activations. Because Paper §6.2 omits Coda KV history across cycles (DR23), Browser Contract A maintains separate persistent output_shallow (banks 96..103) and output_deep (banks 88..95) Coda histories — needing zero extra Shallow Coda work on rejection (a < gamma) and a 1-token catch-up only on full non-EOS acceptance.',
    equation: '104 banks = 8 Input (shared) + 20 Loop-1 (shared) + 60 Loops 2..4 + 8 Deep Coda + 8 Shallow Coda (+20 KiB)',
    epistemicNote:
      `${DORY_SPEC_CONTRACT_A_BADGE}. Paper §6.2 figure speedup curves (1.37–1.71x, DR13) apply to production serving, never synthetic miniature timings.`,
    discrepancyRefs: ['DR13', 'DR23'],
    designDecisionRefs: ['D3', 'D7'],
    viewMode: 'self_spec',
    scenarioKey: 'self_spec_contract_a',
    kLoops: 4,
    kvCacheMode: 'per_loop',
    executionMode: 'self_spec',
    targetStepOffset: 1,
    coordTarget: {
      physicalLayerIdx: 16,
      loopIteration: 1,
      tokenOffset: 0,
      expectedBankKind: 'recurrent',
      expectedCacheRole: 'spec_verify_reused',
    },
    focusPatch: {
      selectedVerifyOffset: 0,
      selfSpecAttentionLayerIdx: 16,
      selectedPhysicalLayerIdx: 16,
      selectedLoopIteration: 1,
      hoveredPartId: 'spec_bank_96',
    },
    highlightPartId: 'spec_bank_96',
    cameraOverride: { azimuthDeg: -26, elevationDeg: 29 },
  }),

  freezeBeatSpec({
    index: 11,
    id: 'act4-self-spec-binary64-rejection-rollback',
    actId: 'act4',
    actNumber: 4,
    actIndex: 3,
    beatInAct: 2,
    title: 'Division-Free Binary64 Rejection Sampling & cap_swa = 16 Rollback',
    subtitle: 'Exact p[d] >= q[d] || r * q[d] < p[d] acceptance, binary64 residual p_res, and atomic ring rewind',
    takeaway:
      'Division-free binary64 rejection sampling preserves the exact k=4 target distribution without 1e-38 clamps or 1e-12 fallbacks, while cap_swa = 16 >= W + gamma + 1 = 13 prevents historical SWA eviction during rollback.',
    narrative:
      'After batched Loops 2..4 and output_deep Coda produce target distributions p_1..p_{gamma+1}, each draft token d_j is tested strictly in binary64: accept unconditionally if p[d] >= q[d], else accept iff r * q[d] < p[d] (zero division by tiny q!). On first rejection, the correction token is drawn from p_res[v] = max(0, p[v] - q[v]) / Z, and all 104 banks rewind atomically to L + a + 1 — with cap_swa = 16 and absPosTag invalidation guaranteeing zero overwritten window history even when L > W = 8 (DR19).',
    equation: 'accept <=> (p[d] >= q[d]) || (r · q[d] < p[d]),   p_res[v] = max(0, p[v] - q[v]) / Z   (binary64)',
    epistemicNote:
      'Verified across 6 independent layers (Gate 2): binary64 algebraic & V=4 exact enumeration <= 1e-15, boundary/tail edge cases, Contract A k=1/k=4 parity <= 2e-5, and Pearson chi^2 + 2 failing negative controls.',
    discrepancyRefs: ['DR19', 'DR20', 'DR23'],
    designDecisionRefs: ['D3', 'D5', 'D7'],
    viewMode: 'self_spec',
    scenarioKey: 'self_spec_contract_a',
    kLoops: 4,
    kvCacheMode: 'per_loop',
    executionMode: 'self_spec',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 56,
      loopIteration: 0,
      tokenOffset: 'last',
      expectedBankKind: 'output_deep',
      expectedCacheRole: 'spec_verify_deep',
    },
    focusPatch: {
      selectedVerifyOffset: 1,
      selfSpecAttentionLayerIdx: 16,
      selectedPhysicalLayerIdx: 56,
      selectedLoopIteration: 0,
      hoveredPartId: 'spec_qp_bars',
    },
    highlightPartId: 'spec_qp_bars',
    cameraOverride: { azimuthDeg: -24, elevationDeg: 30 },
  }),

  freezeBeatSpec({
    index: 12,
    id: 'act4-four-checkpoint-zero-byte-diff',
    actId: 'act4',
    actNumber: 4,
    actIndex: 3,
    beatInAct: 3,
    title: '0 B Diff Production 4-Checkpoint Readouts (BF16 / MXFP8 / NVFP4 / Q5_K_M)',
    subtitle: '219 unquantized BF16 bookend & nGPT scale tensors vs. 288 quantized linear matrices — zero fake quant toggles',
    takeaway:
      'All three HF safetensors checkpoints reconcile to exact 0 B diff against LFS sizes: 288 linear projections quantize while all 219 bookends & nGPT scales remain unquantized BF16 on disk (F32 in GGUF).',
    narrative:
      'Returning to View 7 highlights the strictly read-only production checkpoint ledger reconciled directly from HTTP Range safetensors JSON headers (0 B weight payloads downloaded): BF16 (7.951 GB, 507 tensors), MXFP8-PTQ (4.751 GB, 795 tensors, released config.json observer="mse" exact 2-candidate E8M0 search, DR15), NVFP4-QAD (3.202 GB, 1,083 tensors), and GGUF Q5_K_M (3.673 GB, F32 1D scales, DR14) — all governed by OpenMDW-1.1 (DR16).',
    equation: 'file_bytes = 8 + header_json_bytes + payload_bytes   (0 B diff across BF16, MXFP8, NVFP4)',
    epistemicNote:
      'Strictly read-only metadata (zero fake quantization toggles): browser miniature always executes D5 Float32Array-storage / binary64-accumulation FP32 reference math.',
    discrepancyRefs: ['DR14', 'DR15', 'DR16'],
    designDecisionRefs: ['D2', 'D7'],
    viewMode: 'bookends',
    scenarioKey: 'standard_k4_per_loop',
    kLoops: 4,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 70,
      loopIteration: 0,
      tokenOffset: 'last',
      expectedBankKind: 'output_deep',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      selectedCheckpointFormat: 'nvfp4',
      selectedPhysicalLayerIdx: 70,
      selectedLoopIteration: 0,
      hoveredPartId: 'bookend_ckpt_nvfp4',
    },
    highlightPartId: 'bookend_ckpt_nvfp4',
    cameraOverride: { azimuthDeg: -26, elevationDeg: 27 },
  }),

  freezeBeatSpec({
    index: 13,
    id: 'act4-readonly-tbptl-and-paper-results',
    actId: 'act4',
    actNumber: 4,
    actIndex: 3,
    beatInAct: 4,
    title: 'READ-ONLY TBPTL Depth-1, Bounded D10 Memory & Epistemic Scope',
    subtitle: 'Stop-gradient across loops, 3-stage pretraining + GRPO 4:1:1:1 curriculum, and isolated <= 12 MiB runtime',
    takeaway:
      'View 7 documents Dory’s TBPTL depth-1 training regime strictly as a READ-ONLY · NO GRADIENTS schematic while this guided tour executes entirely inside an isolated <= 12.00 MiB DorySession.',
    narrative:
      'In production training (arXiv:2611.03998), Dory progresses through 4.5T NVFP4 -> 1.5T BF16 Looped (with 0.1 ELR rank override) -> 300B 256K long-context pretraining, followed by SFT and GRPO RLVR with a 4:1:1:1 difficulty curriculum (alpha=0.3). Truncated Backprop Through Loops (TBPTL depth-1) places a stop-gradient between recurrent passes with zero early-exit losses so the Prelude receives gradients exclusively via the final loop’s anchor injection +x in u_k. Exiting this tour restores your untouched Free-Explore session byte-for-byte.',
    equation: 'u_k = N_d(sg(s_{k-1}) + x)   [TBPTL depth-1 · 0 early-exit losses · READ-ONLY NO GRADIENTS]',
    epistemicNote:
      `${DORY_TBPTL_READONLY_BADGE}. ${DORY_PAPER_RESULTS_BADGE}. Synthetic miniature weights never claim reasoning quality gains or output-token shortening with k.`,
    discrepancyRefs: ['DR1', 'DR13', 'DR22'],
    designDecisionRefs: ['D4', 'D7', 'D10'],
    viewMode: 'bookends',
    scenarioKey: 'standard_k1_per_loop',
    kLoops: 1,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    targetStepOffset: -1,
    coordTarget: {
      physicalLayerIdx: 71,
      loopIteration: 0,
      tokenOffset: 'last',
      expectedBankKind: 'output_deep',
      expectedCacheRole: 'standard',
    },
    focusPatch: {
      selectedCheckpointFormat: 'bf16',
      selectedPhysicalLayerIdx: 71,
      selectedLoopIteration: 0,
      hoveredPartId: 'bookend_tbptl_stop_grad',
    },
    highlightPartId: 'bookend_tbptl_stop_grad',
    cameraOverride: { azimuthDeg: -28, elevationDeg: 26 },
  }),
]);

// ============================================================================
// IMMUTABLE 8-TUPLE `DoryExecutionCoord` VALIDATOR (`AUDIT §6.3`)
// ============================================================================

/**
 * Exhaustively validates all 8 components of `DoryExecutionCoord` (`Audit §6.3`):
 *   (1) `modelWeightId`    — non-empty singleton weight identity (exact match if provided)
 *   (2) `physicalLayerIdx` — integer `p in 0..71` (`16` Prelude + `40` Recurrent + `16` Coda)
 *   (3) `executedLayerIdx` — exact `computeExecutedLayerIdx(p, t, k)` (allowing `k=1` draft
 *                            indexing for `output_shallow` / `spec_draft`)
 *   (4) `loopIteration`    — `0` for Prelude/Coda, `1..k` for Recurrent
 *   (5) `absTokenPos`      — integer in `0..(S_max - 1)` (`0..31`)
 *   (6) `forwardStepId`    — non-negative integer
 *   (7) `sessionBranchId`  — non-empty trimmed branch identifier
 *   (8) `cacheOwnership`   — valid `policy`, stage-aligned `bankKind`, `DR23` Contract A
 *                            `per_loop`-only invariant, canonical `bankSlotId` (`-1` for MLP,
 *                            exact bank layout for Attention), stage/policy-consistent `writerLoop`,
 *                            and valid `role`.
 */
export function validateDoryExecutionCoord(
  coord: DoryExecutionCoord,
  expectedModelWeightId?: string,
  kLoops?: number,
): DoryExecutionCoordValidationResult {
  const errors: string[] = [];

  if (coord === null || coord === undefined || typeof coord !== 'object') {
    return Object.freeze({
      valid: false,
      errors: Object.freeze(['DoryExecutionCoord must be a non-null object.']),
    });
  }

  // (1) modelWeightId
  if (typeof coord.modelWeightId !== 'string' || coord.modelWeightId.trim().length === 0) {
    errors.push('Coord #1 (modelWeightId) must be a non-empty string.');
  } else if (
    expectedModelWeightId !== undefined &&
    coord.modelWeightId !== expectedModelWeightId
  ) {
    errors.push(
      `Coord #1 (modelWeightId) mismatch: got '${coord.modelWeightId}', expected '${expectedModelWeightId}'.`,
    );
  }

  // Optional kLoops validation
  let validKLoops: number | undefined;
  if (kLoops !== undefined) {
    if (!Number.isInteger(kLoops) || kLoops < 1) {
      errors.push(`Expected kLoops to be an integer >= 1, got ${String(kLoops)}.`);
    } else {
      validKLoops = kLoops;
    }
  }

  // (2) physicalLayerIdx
  const p = coord.physicalLayerIdx;
  const maxPhys = DORY_MINI_CONFIG.numHiddenLayers - 1;
  const physValid = Number.isInteger(p) && p >= 0 && p <= maxPhys;
  if (!physValid) {
    errors.push(
      `Coord #2 (physicalLayerIdx) must be an integer in 0..${maxPhys}; got ${String(p)}.`,
    );
  }

  const stage = physValid ? getDoryStage(p) : null;
  const isAttn = physValid ? p % 2 === 0 : null;

  // (4) loopIteration
  const t = coord.loopIteration;
  const loopInt = Number.isInteger(t) && t >= 0;
  if (!loopInt) {
    errors.push(`Coord #4 (loopIteration) must be a non-negative integer; got ${String(t)}.`);
  } else if (stage === 'input' || stage === 'output') {
    if (t !== 0) {
      errors.push(
        `Coord #4 (loopIteration) must be 0 for stage='${stage}' (p=${p}); got ${t}.`,
      );
    }
  } else if (stage === 'recurrent') {
    if (t < 1) {
      errors.push(
        `Coord #4 (loopIteration) must be >= 1 for recurrent stage (p=${p}); got ${t}.`,
      );
    }
    if (validKLoops !== undefined && t > validKLoops) {
      errors.push(
        `Coord #4 (loopIteration=${t}) exceeds kLoops=${validKLoops} for recurrent p=${p}.`,
      );
    }
  }

  // (5) absTokenPos
  const pos = coord.absTokenPos;
  const maxPos = DORY_MINI_CONFIG.maxPositionEmbeddings - 1;
  if (!Number.isInteger(pos) || pos < 0 || pos > maxPos) {
    errors.push(
      `Coord #5 (absTokenPos) must be an integer in 0..${maxPos}; got ${String(pos)}.`,
    );
  }

  // (6) forwardStepId
  if (!Number.isInteger(coord.forwardStepId) || coord.forwardStepId < 0) {
    errors.push(
      `Coord #6 (forwardStepId) must be an integer >= 0; got ${String(coord.forwardStepId)}.`,
    );
  }

  // (7) sessionBranchId
  if (
    typeof coord.sessionBranchId !== 'string' ||
    coord.sessionBranchId.trim().length === 0
  ) {
    errors.push('Coord #7 (sessionBranchId) must be a non-empty string.');
  }

  // (8) cacheOwnership
  const co = coord.cacheOwnership;
  const coValidObj = co !== null && co !== undefined && typeof co === 'object';
  if (!coValidObj) {
    errors.push('Coord #8 (cacheOwnership) must be a non-null object.');
  }

  const policyValid = coValidObj && VALID_KV_POLICIES.has(co.policy);
  const bankKindValid = coValidObj && VALID_BANK_KINDS.has(co.bankKind);
  const roleValid = coValidObj && VALID_CACHE_ROLES.has(co.role);

  if (coValidObj) {
    if (!policyValid) {
      errors.push(`Coord #8 (cacheOwnership.policy) invalid: ${String(co.policy)}.`);
    }
    if (!bankKindValid) {
      errors.push(`Coord #8 (cacheOwnership.bankKind) invalid: ${String(co.bankKind)}.`);
    }
    if (!roleValid) {
      errors.push(`Coord #8 (cacheOwnership.role) invalid: ${String(co.role)}.`);
    }

    // Stage <-> bankKind consistency
    if (stage === 'input' && bankKindValid && co.bankKind !== 'input') {
      errors.push(
        `Coord #8 bankKind='${co.bankKind}' mismatches Prelude stage (p=${p}); expected 'input'.`,
      );
    } else if (stage === 'recurrent' && bankKindValid && co.bankKind !== 'recurrent') {
      errors.push(
        `Coord #8 bankKind='${co.bankKind}' mismatches Recurrent stage (p=${p}); expected 'recurrent'.`,
      );
    } else if (
      stage === 'output' &&
      bankKindValid &&
      co.bankKind !== 'output_deep' &&
      co.bankKind !== 'output_shallow'
    ) {
      errors.push(
        `Coord #8 bankKind='${co.bankKind}' mismatches Output Coda stage (p=${p}); expected 'output_deep' or 'output_shallow'.`,
      );
    }

    // DR23 Contract A per_loop-only guards
    if (policyValid && bankKindValid && co.bankKind === 'output_shallow' && co.policy !== 'per_loop') {
      errors.push(
        "Coord #8 DR23 violation: bankKind='output_shallow' is strictly valid under policy='per_loop' only.",
      );
    }
    if (policyValid && roleValid && co.role !== 'standard' && co.policy !== 'per_loop') {
      errors.push(
        `Coord #8 DR23 violation: speculative role='${co.role}' is strictly valid under policy='per_loop' only.`,
      );
    }

    // writerLoop checks
    const wl = co.writerLoop;
    if (!Number.isInteger(wl) || wl < 0) {
      errors.push(`Coord #8 (writerLoop) must be a non-negative integer; got ${String(wl)}.`);
    } else if (stage === 'input' || stage === 'output') {
      if (wl !== 0) {
        errors.push(`Coord #8 (writerLoop) must be 0 for stage='${stage}' (p=${p}); got ${wl}.`);
      }
    } else if (stage === 'recurrent') {
      if (wl < 1) {
        errors.push(`Coord #8 (writerLoop) must be >= 1 for recurrent p=${p}; got ${wl}.`);
      }
      if (validKLoops !== undefined && wl > validKLoops) {
        errors.push(
          `Coord #8 (writerLoop=${wl}) exceeds kLoops=${validKLoops} for recurrent p=${p}.`,
        );
      }
      if (isAttn === false && loopInt && t >= 1 && wl !== t) {
        errors.push(
          `Coord #8 (writerLoop=${wl}) must equal loopIteration=${t} for recurrent MLP sublayer p=${p}.`,
        );
      } else if (isAttn === true && policyValid && loopInt && t >= 1) {
        if (co.policy === 'per_loop' && wl !== t) {
          errors.push(
            `Coord #8 (writerLoop=${wl}) must equal loopIteration=${t} under policy='per_loop' (p=${p}).`,
          );
        } else if (co.policy === 'last_loop') {
          if (validKLoops !== undefined) {
            if (wl !== t && wl !== validKLoops) {
              errors.push(
                `Coord #8 (writerLoop=${wl}) must equal loopIteration=${t} or kLoops=${validKLoops} under policy='last_loop' for recurrent attention p=${p}.`,
              );
            }
          } else if (wl < t) {
            errors.push(
              `Coord #8 (writerLoop=${wl}) cannot be less than loopIteration=${t} under policy='last_loop' for recurrent attention p=${p}.`,
            );
          }
        }
      }
    }

    // bankSlotId checks (MLP = -1; Attention = exact canonical bank layout)
    const slotId = co.bankSlotId;
    if (!Number.isInteger(slotId)) {
      errors.push(`Coord #8 (bankSlotId) must be an integer; got ${String(slotId)}.`);
    } else if (isAttn === false) {
      if (slotId !== -1) {
        errors.push(
          `Coord #8 (bankSlotId) must be -1 for MLP sublayer p=${p}; got ${slotId}.`,
        );
      }
    } else if (isAttn === true && policyValid && bankKindValid) {
      if (co.bankKind === 'input') {
        const expected = p >> 1;
        if (slotId !== expected) {
          errors.push(
            `Coord #8 (bankSlotId=${slotId}) mismatch for input attention p=${p}; expected ${expected}.`,
          );
        }
      } else if (co.bankKind === 'recurrent') {
        const ord = (p - DORY_MINI_CONFIG.nInputLayers) >> 1;
        if (co.policy === 'last_loop') {
          const expected = 8 + ord;
          if (slotId !== expected) {
            errors.push(
              `Coord #8 (bankSlotId=${slotId}) mismatch for last_loop recurrent p=${p}; expected ${expected}.`,
            );
          }
        } else if (loopInt && t >= 1) {
          const expected = 8 + (t - 1) * 20 + ord;
          if (slotId !== expected) {
            errors.push(
              `Coord #8 (bankSlotId=${slotId}) mismatch for per_loop recurrent (p=${p}, t=${t}); expected ${expected}.`,
            );
          }
        }
      } else if (co.bankKind === 'output_deep') {
        const ord =
          (p - (DORY_MINI_CONFIG.nInputLayers + DORY_MINI_CONFIG.nRecurrentLayers)) >> 1;
        if (co.policy === 'last_loop') {
          const expected = 28 + ord;
          if (slotId !== expected) {
            errors.push(
              `Coord #8 (bankSlotId=${slotId}) mismatch for last_loop output_deep p=${p}; expected ${expected}.`,
            );
          }
        } else if (validKLoops !== undefined) {
          const expected = 8 + 20 * validKLoops + ord;
          if (slotId !== expected) {
            errors.push(
              `Coord #8 (bankSlotId=${slotId}) mismatch for per_loop output_deep (p=${p}, k=${validKLoops}); expected ${expected}.`,
            );
          }
        } else if (slotId < 28 + ord || (slotId - 8 - ord) % 20 !== 0) {
          errors.push(
            `Coord #8 (bankSlotId=${slotId}) is not a valid per_loop output_deep slot for p=${p}.`,
          );
        }
      } else if (co.bankKind === 'output_shallow') {
        const ord =
          (p - (DORY_MINI_CONFIG.nInputLayers + DORY_MINI_CONFIG.nRecurrentLayers)) >> 1;
        if (validKLoops !== undefined) {
          const expected = 16 + 20 * validKLoops + ord;
          if (slotId !== expected) {
            errors.push(
              `Coord #8 (bankSlotId=${slotId}) mismatch for Contract A output_shallow (p=${p}, k=${validKLoops}); expected ${expected}.`,
            );
          }
        } else if (slotId < 36 + ord || (slotId - 16 - ord) % 20 !== 0) {
          errors.push(
            `Coord #8 (bankSlotId=${slotId}) is not a valid Contract A output_shallow slot for p=${p}.`,
          );
        }
      }
    }
  }

  // (3) executedLayerIdx (validated after cacheOwnership to respect draft k=1 output_shallow)
  const e = coord.executedLayerIdx;
  const execInt = Number.isInteger(e) && e >= 0;
  if (!execInt) {
    errors.push(`Coord #3 (executedLayerIdx) must be a non-negative integer; got ${String(e)}.`);
  } else if (physValid && loopInt) {
    const isShallowDraft =
      coValidObj && co.bankKind === 'output_shallow' && co.role === 'spec_draft';
    if (stage === 'input') {
      if (e !== p) {
        errors.push(`Coord #3 (executedLayerIdx=${e}) must equal p=${p} in Input Prelude.`);
      }
    } else if (stage === 'recurrent' && t >= 1) {
      const expected =
        DORY_MINI_CONFIG.nInputLayers +
        (t - 1) * DORY_MINI_CONFIG.nRecurrentLayers +
        (p - DORY_MINI_CONFIG.nInputLayers);
      if (e !== expected) {
        errors.push(
          `Coord #3 (executedLayerIdx=${e}) mismatch for recurrent (p=${p}, t=${t}); expected ${expected}.`,
        );
      }
      if (validKLoops !== undefined && e >= computeExecutedDepth(validKLoops)) {
        errors.push(
          `Coord #3 (executedLayerIdx=${e}) exceeds executed depth ${computeExecutedDepth(validKLoops)} for k=${validKLoops}.`,
        );
      }
    } else if (stage === 'output' && t === 0) {
      const codaOffset = p - (DORY_MINI_CONFIG.nInputLayers + DORY_MINI_CONFIG.nRecurrentLayers);
      if (validKLoops !== undefined) {
        const expectedAtK = computeExecutedLayerIdx(p, 0, validKLoops);
        const expectedDraft = computeExecutedLayerIdx(p, 0, DORY_SPEC_DRAFT_LOOPS);
        const ok = e === expectedAtK || (isShallowDraft && e === expectedDraft);
        if (!ok) {
          errors.push(
            `Coord #3 (executedLayerIdx=${e}) mismatch for Output Coda (p=${p}, k=${validKLoops}); expected ${expectedAtK}.`,
          );
        }
      } else {
        const base = DORY_MINI_CONFIG.nInputLayers + codaOffset;
        if (
          e < base + DORY_MINI_CONFIG.nRecurrentLayers ||
          (e - base) % DORY_MINI_CONFIG.nRecurrentLayers !== 0
        ) {
          errors.push(
            `Coord #3 (executedLayerIdx=${e}) is not a valid Output Coda executed index for p=${p}.`,
          );
        }
      }
    }
  }

  return Object.freeze({
    valid: errors.length === 0,
    errors: Object.freeze(errors),
  });
}

// ============================================================================
// ISOLATED PRIVATE TOUR SESSION STAGING & HELPERS
// ============================================================================

interface StagedScenarioBranch {
  readonly scenarioKey: DoryTourScenarioKey;
  readonly branchId: string;
  readonly recordIds: readonly string[];
}

function resolveScenarioRecordId(
  staged: StagedScenarioBranch,
  targetStepOffset: number,
): string {
  const count = staged.recordIds.length;
  if (count < 1) {
    throw new Error(`Tour scenario '${staged.scenarioKey}' has zero staged records.`);
  }
  if (targetStepOffset < 0) {
    const idx = Math.max(0, count + targetStepOffset);
    return staged.recordIds[idx]!;
  }
  const idx = Math.max(0, Math.min(count - 1, Math.trunc(targetStepOffset)));
  return staged.recordIds[idx]!;
}

function stageTourScenarios(
  tourSession: DorySession,
  seed: number,
): ReadonlyMap<DoryTourScenarioKey, StagedScenarioBranch> {
  const map = new Map<DoryTourScenarioKey, StagedScenarioBranch>();

  const register = (key: DoryTourScenarioKey, branchId: string): void => {
    const snap = tourSession.getBranch(branchId);
    if (snap === null) {
      throw new Error(`Missing staged tour branch '${branchId}' for scenario '${key}'.`);
    }
    map.set(
      key,
      Object.freeze({
        scenarioKey: key,
        branchId,
        recordIds: Object.freeze([...snap.recordIds]),
      }),
    );
  };

  // 1. standard_k1_per_loop (72L · 36 banks, greedy prefill + 1 decode step across W=8)
  const k1Prefill = tourSession.startBranch({
    label: 'Tour · k=1 per_loop (72L · 36 banks)',
    promptTokens: DORY_TOUR_PROMPT_TOKENS,
    kLoops: 1,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    sampling: DORY_DEFAULT_GREEDY_SAMPLING,
    seed: deriveDorySeed(seed, 101),
  });
  if (k1Prefill.haltReason === null) {
    tourSession.step({ branchId: k1Prefill.branchId });
  }
  register('standard_k1_per_loop', k1Prefill.branchId);

  // 2. standard_k2_per_loop (112L default · 56 banks, greedy prefill + 1 decode step)
  const k2Prefill = tourSession.startBranch({
    label: 'Tour · k=2 per_loop (112L default · 56 banks)',
    promptTokens: DORY_TOUR_PROMPT_TOKENS,
    kLoops: 2,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    sampling: DORY_DEFAULT_GREEDY_SAMPLING,
    seed: deriveDorySeed(seed, 202),
  });
  if (k2Prefill.haltReason === null) {
    tourSession.step({ branchId: k2Prefill.branchId });
  }
  register('standard_k2_per_loop', k2Prefill.branchId);

  // 3. standard_k4_per_loop (192L · 96 banks, greedy prefill + 1 decode step)
  const k4Prefill = tourSession.startBranch({
    label: 'Tour · k=4 per_loop (192L · 96 banks)',
    promptTokens: DORY_TOUR_PROMPT_TOKENS,
    kLoops: 4,
    kvCacheMode: 'per_loop',
    executionMode: 'standard',
    sampling: DORY_DEFAULT_GREEDY_SAMPLING,
    seed: deriveDorySeed(seed, 404),
  });
  if (k4Prefill.haltReason === null) {
    tourSession.step({ branchId: k4Prefill.branchId });
  }
  register('standard_k4_per_loop', k4Prefill.branchId);

  // 4. standard_k4_last_loop_chunked (192L · 36 banks, chunked [4, 8] DR17/DR24 fixture + 1 decode step)
  const k4LastChunked = tourSession.startBranch({
    label: 'Tour · k=4 last_loop [4, 8] (36 banks · DR17)',
    promptTokens: DORY_TOUR_PROMPT_TOKENS,
    prefillChunkSizes: DORY_TOUR_LAST_LOOP_PARTITION,
    kLoops: 4,
    kvCacheMode: 'last_loop',
    executionMode: 'standard',
    sampling: DORY_DEFAULT_GREEDY_SAMPLING,
    seed: deriveDorySeed(seed, 405),
  });
  if (k4LastChunked.haltReason === null) {
    tourSession.step({ branchId: k4LastChunked.branchId });
  }
  register('standard_k4_last_loop_chunked', k4LastChunked.branchId);

  // 5. self_spec_contract_a (Contract A 104 banks, 2 stochastic gamma=4 cycles across S > W=8)
  // Using initialSampleUniform=0.5 guarantees non-EOS pendingToken across arbitrary caller seeds
  // (EOS tokens occupy indices 2 and 11 out of 64, well below cumulative probability 0.5).
  const specPrefill = tourSession.startBranch({
    label: 'Tour · Contract A Self-Spec (k=1->4 · 104 banks)',
    promptTokens: DORY_TOUR_PROMPT_TOKENS,
    kLoops: DORY_SPEC_VERIFY_LOOPS,
    kvCacheMode: 'per_loop',
    executionMode: 'self_spec',
    speculativeGamma: 4,
    sampling: DORY_TOUR_SPEC_SAMPLING,
    seed: deriveDorySeed(seed, 602),
    initialSampleUniform: 0.5,
  });
  let specTip = specPrefill;
  for (let cycle = 0; cycle < 2; cycle += 1) {
    if (specTip.haltReason !== null) {
      break;
    }
    specTip = tourSession.step({ branchId: specPrefill.branchId });
  }
  register('self_spec_contract_a', specPrefill.branchId);

  return map;
}

function cloneRestoreState(
  restoreState: DoryFreeExploreRestoreState | null | undefined,
): DoryFreeExploreRestoreState | null {
  if (restoreState === null || restoreState === undefined) {
    return null;
  }
  const focusCopy: DorySpatialFocusState = Object.freeze({
    ...restoreState.focus,
    ...(restoreState.focus.kvCompareChunkPartition != null
      ? { kvCompareChunkPartition: Object.freeze([...restoreState.focus.kvCompareChunkPartition]) }
      : {}),
    ...(restoreState.focus.selfSpecPreviewSampling != null
      ? {
          selfSpecPreviewSampling: Object.freeze({
            ...restoreState.focus.selfSpecPreviewSampling,
          }),
        }
      : {}),
  });
  const cameraPoseCopy =
    restoreState.cameraPose != null
      ? Object.freeze({
          ...restoreState.cameraPose,
          target: Object.freeze([
            restoreState.cameraPose.target[0],
            restoreState.cameraPose.target[1],
            restoreState.cameraPose.target[2],
          ] as [number, number, number]),
        })
      : restoreState.cameraPose === null
        ? null
        : undefined;

  return Object.freeze({
    viewMode: restoreState.viewMode,
    focus: focusCopy,
    activeBranchId: restoreState.activeBranchId,
    activeRecordId: restoreState.activeRecordId,
    ...(cameraPoseCopy !== undefined ? { cameraPose: cameraPoseCopy } : {}),
    ...(restoreState.inspectorTab !== undefined ? { inspectorTab: restoreState.inspectorTab } : {}),
    ...(restoreState.inspectorOpen !== undefined
      ? { inspectorOpen: restoreState.inspectorOpen }
      : {}),
  });
}

function computeArrayMinMax(arr: ArrayLike<number>): readonly [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < arr.length; i += 1) {
    const v = arr[i]!;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return [Number.isFinite(lo) ? lo : 0, Number.isFinite(hi) ? hi : 0];
}

function computeMaxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let maxAbs = 0;
  for (let i = 0; i < n; i += 1) {
    const d = Math.abs(a[i]! - b[i]!);
    if (d > maxAbs) maxAbs = d;
  }
  return maxAbs;
}

function buildActSummaries(
  beats: readonly DoryTourBeatSpec[],
  currentActIndex: number,
): readonly DoryTourActSummary[] {
  return Object.freeze(
    DORY_TOUR_ACT_DESCRIPTORS.map((desc: DoryTourActDescriptor) => {
      let startBeatIndex = -1;
      let beatCount = 0;
      for (let i = 0; i < beats.length; i += 1) {
        if (beats[i]!.actId === desc.id) {
          if (startBeatIndex < 0) startBeatIndex = i;
          beatCount += 1;
        }
      }
      return Object.freeze({
        ...desc,
        startBeatIndex: startBeatIndex < 0 ? 0 : startBeatIndex,
        beatCount,
        isCurrent: desc.actIndex === currentActIndex,
      });
    }),
  );
}

// ============================================================================
// LIVE BEAT METRICS & PROVENANCE BADGES (100% REAL MINIATURE TENSORS)
// ============================================================================

function buildLiveBeatMetrics(args: {
  readonly spec: DoryTourBeatSpec;
  readonly ctx: DoryResolvedSessionContext;
  readonly coord: DoryExecutionCoord;
  readonly tokenOffset: number;
  readonly tourSession: DorySession;
  readonly scenarioMap: ReadonlyMap<DoryTourScenarioKey, StagedScenarioBranch>;
}): readonly DoryTourLiveMetric[] {
  const { spec, ctx, coord, tokenOffset, tourSession, scenarioMap } = args;
  const { params, record, branch } = ctx;
  const config = params.config;
  const d = config.hiddenSize;
  const metrics: DoryTourLiveMetric[] = [];

  switch (spec.id) {
    case 'act1-untied-unit-sphere-bookends': {
      const committedSeq = record.committedSequence;
      const tokId = committedSeq[coord.absTokenPos] ?? committedSeq[committedSeq.length - 1] ?? 1;
      const embNorm = vectorL2Norm(params.embeddings, tokId * d, d);
      const lmNorm = vectorL2Norm(params.lmHead, tokId * d, d);
      const untiedDiff = computeMaxAbsDiff(params.embeddings, params.lmHead);
      const effLogitScale = effectiveLogitScale(params.logitScale, config);
      const [sMin, sMax] = computeArrayMinMax(effLogitScale);
      const probsT1 = softmaxFloat64(record.lastTokenScaledLogits, 0, config.vocabSize);
      const topId = argmaxIndex(record.lastTokenScaledLogits, 0, config.vocabSize);
      const topProb = probsT1[topId] ?? 0;

      metrics.push(
        {
          label: `||E['${decodeDoryToken(tokId)}']||_2`,
          value: embNorm.toFixed(6),
          tone: 'prelude_indigo',
          discrepancyId: 'DR5',
        },
        {
          label: `||W_lm['${decodeDoryToken(tokId)}']||_2`,
          value: lmNorm.toFixed(6),
          tone: 'coda_amber',
        },
        {
          label: 'Untied max |E - W_lm|',
          value: untiedDiff.toFixed(4),
          tone: 'nlerp_emerald',
          discrepancyId: 'DR1',
        },
        {
          label: 'FP32 s_z / 0.0177',
          value: `[${sMin.toFixed(3)}, ${sMax.toFixed(3)}]`,
          tone: 'gold',
          discrepancyId: 'DR9',
        },
        {
          label: 'Top-1 Next Token (T=1)',
          value: `'${decodeDoryToken(topId)}' (${(topProb * 100).toFixed(2)}%)`,
          tone: 'muted',
        },
      );
      break;
    }

    case 'act1-folded-72l-topology': {
      const execDepth = computeExecutedDepth(record.kLoops, config);
      const pairedDepth = computeEffectivePairedDepth(record.kLoops, config);
      const banks = computeKvBankCounts(record.kLoops, record.kvCacheMode, false);
      const loop2Metric = record.lastTokenLoopMetrics[record.lastTokenLoopMetrics.length - 1];

      metrics.push(
        {
          label: 'Executed / Physical Depth',
          value: `${execDepth}L exec / 72L phys (${pairedDepth} paired)`,
          tone: 'recurrent_teal',
          discrepancyId: 'DR4',
        },
        {
          label: 'Singleton Weights',
          value: `${DORY_MINI_TENSOR_COUNT} tensors · ${DORY_MINI_FLOAT_COUNT.toLocaleString('en-US')} F32 (${formatKiB(DORY_MINI_WEIGHT_BYTES)})`,
          tone: 'prelude_indigo',
        },
        {
          label: 'KV Banks (k=2 per_loop)',
          value: `${banks.totalBanks} (${banks.totalSwaBanks} SWA + ${banks.totalFullBanks} Full)`,
          tone: 'swa_cyan',
        },
        {
          label: 'Weight ID (SHA-256)',
          value: params.modelWeightId.slice(0, 12),
          tone: 'muted',
        },
      );
      if (loop2Metric !== undefined) {
        metrics.push({
          label: `cos(s_${loop2Metric.loopIteration}, s_${loop2Metric.loopIteration - 1}) (8D)`,
          value: fmtSigned(loop2Metric.consecutiveCosine, 4),
          tone: 'nlerp_emerald',
        });
      }
      break;
    }

    case 'act1-ngpt-nlerp-and-five-scales': {
      const sub = ctx.resolveSublayerRecord(coord.physicalLayerIdx, coord.loopIteration);
      const nlerp = sub.trace.nlerp;
      const aNorm = vectorL2Norm(nlerp.aNorm, tokenOffset * d, d);
      const outNorm = vectorL2Norm(nlerp.output, tokenOffset * d, d);
      const alphaMult = config.ngptAlphaInit / config.initializerRange;

      metrics.push(
        {
          label: `p=${coord.physicalLayerIdx} max |alpha_hat|`,
          value: nlerp.maxEffectiveAlpha.toFixed(4),
          tone: nlerp.extrapolatedChannelCount > 0 ? 'warn' : 'nlerp_emerald',
          discrepancyId: 'DR7',
        },
        {
          label: 'Extrapolating Channels (>1)',
          value: `${nlerp.extrapolatedChannelCount} / ${d} (unclamped)`,
          tone: 'warn',
          discrepancyId: 'DR7',
        },
        {
          label: 'Active alpha_init / sigma_0',
          value: `${alphaMult.toFixed(4)}x (0.05 / 0.0177)`,
          tone: 'gold',
          discrepancyId: 'DR9',
        },
        {
          label: '||N(x_res)||_2 -> ||N(out)||_2',
          value: `${aNorm.toFixed(5)} -> ${outNorm.toFixed(5)}`,
          tone: 'nlerp_emerald',
        },
      );
      break;
    }

    case 'act2-gated-gqa-swa-rope': {
      const sub = ctx.resolveSublayerRecord(coord.physicalLayerIdx, coord.loopIteration);
      const attn = sub.trace as DoryAttentionSublayerTrace;
      const weights = params.layers[coord.physicalLayerIdx] as DoryAttentionSublayerWeights;
      const [sqkMin, sqkMax] = computeArrayMinMax(effectiveSqkSquared(weights.sqk, config));
      const [sgMin, sgMax] = computeArrayMinMax(effectiveSGateSquared(weights.sGate, config));
      const subNormVal = vectorL2Norm(attn.attnSubNorm, tokenOffset * config.qProjOutDim, config.qProjOutDim);
      let gateMean = 0;
      const gateOff = tokenOffset * config.qProjOutDim;
      for (let c = 0; c < config.qProjOutDim; c += 1) {
        gateMean += attn.gateValues[gateOff + c] ?? 0;
      }
      gateMean /= config.qProjOutDim;

      metrics.push(
        {
          label: `SWA Visible Keys (W=${config.slidingWindow})`,
          value: `${attn.kvLen} slots (${attn.pastLen} past + ${attn.seqLen} cur)`,
          tone: 'swa_cyan',
          discrepancyId: 'DR20',
        },
        {
          label: 'Q-Only sqk^2 Range',
          value: `[${sqkMin.toFixed(2)}, ${sqkMax.toFixed(2)}]`,
          tone: 'recurrent_teal',
          discrepancyId: 'DR6',
        },
        {
          label: 'Gate s_gate^2 & Mean sigma',
          value: `[${sgMin.toFixed(2)}, ${sgMax.toFixed(2)}] · mean ${gateMean.toFixed(3)}`,
          tone: 'gold',
          discrepancyId: 'DR3',
        },
        {
          label: '||attn_sub_norm||_2 (64D)',
          value: subNormVal.toFixed(6),
          tone: 'nlerp_emerald',
        },
      );
      break;
    }

    case 'act2-full-attention-dual-rope': {
      const sub = ctx.resolveSublayerRecord(coord.physicalLayerIdx, coord.loopIteration);
      const attn = sub.trace as DoryAttentionSublayerTrace;
      const subNormVal = vectorL2Norm(attn.attnSubNorm, tokenOffset * config.qProjOutDim, config.qProjOutDim);
      const probBase = (0 * attn.seqLen + tokenOffset) * attn.kvLen;
      let maxWeight = 0;
      let entropy = 0;
      for (let k = 0; k < attn.kvLen; k += 1) {
        const w = attn.probs[probBase + k] ?? 0;
        if (w > maxWeight) maxWeight = w;
        if (w > 1e-12) entropy -= w * Math.log(w);
      }

      metrics.push(
        {
          label: 'Full Causal Visible Keys',
          value: `${attn.kvLen} / ${record.committedLen} (zero eviction)`,
          tone: 'full_indigo',
          discrepancyId: 'DR20',
        },
        {
          label: 'Dual-RoPE Base Ratio',
          value: `theta_F=${config.ropeThetaFull} / theta_S=${config.ropeThetaSliding} (64x mini · 1,000x prod)`,
          tone: 'full_indigo',
          discrepancyId: 'DR2',
        },
        {
          label: 'Head 0 Peak Prob / H',
          value: `${(maxWeight * 100).toFixed(1)}% · H=${entropy.toFixed(3)} nats`,
          tone: 'recurrent_teal',
        },
        {
          label: '||attn_sub_norm||_2 (64D)',
          value: subNormVal.toFixed(6),
          tone: 'nlerp_emerald',
        },
      );
      break;
    }

    case 'act2-normalized-swiglu-mixer': {
      const sub = ctx.resolveSublayerRecord(coord.physicalLayerIdx, coord.loopIteration);
      const mlp = sub.trace as DoryMlpSublayerTrace;
      const weights = params.layers[coord.physicalLayerIdx] as DoryMlpSublayerWeights;
      const effGate = effectiveSuvScale(weights.suvGate, config);
      const effUp = effectiveSuvScale(weights.suvUp, config);
      const [gMin, gMax] = computeArrayMinMax(effGate);
      let negCount = 0;
      for (let i = 0; i < effGate.length; i += 1) {
        if (effGate[i]! < 0) negCount += 1;
      }
      for (let i = 0; i < effUp.length; i += 1) {
        if (effUp[i]! < 0) negCount += 1;
      }
      const ffnNormVal = vectorL2Norm(
        mlp.ffnSubNorm,
        tokenOffset * config.intermediateSize,
        config.intermediateSize,
      );
      const outNormVal = vectorL2Norm(mlp.nlerp.output, tokenOffset * d, d);

      metrics.push(
        {
          label: 'Signed suv_gate * sqrt(8)',
          value: `[${fmtSigned(gMin, 2)}, ${fmtSigned(gMax, 2)}] (${negCount} neg signed)`,
          tone: 'gold',
          discrepancyId: 'DR9',
        },
        {
          label: '||ffn_sub_norm||_2 (24D)',
          value: ffnNormVal.toFixed(6),
          tone: 'nlerp_emerald',
        },
        {
          label: 'MLP Expansion Ratio',
          value: `${config.hiddenSize}->${config.intermediateSize} (3.00x mini · 2.80x prod)`,
          tone: 'recurrent_teal',
        },
        {
          label: '||NLERP out||_2 (8D)',
          value: outNormVal.toFixed(6),
          tone: 'nlerp_emerald',
          discrepancyId: 'DR7',
        },
      );
      break;
    }

    case 'act3-unrolled-shared-weights-k4': {
      const execDepth = computeExecutedDepth(record.kLoops, config);
      const banks = computeKvBankCounts(record.kLoops, record.kvCacheMode, false);
      const s1 = record.lastTokenLoopExitStates[0];
      const s4 = record.lastTokenLoopExitStates[record.lastTokenLoopExitStates.length - 1];
      const disp14 = s1 && s4 ? l2Distance(s4, s1, 0, 0, d) : 0;

      metrics.push(
        {
          label: 'Unrolled Depth (k=4)',
          value: `${execDepth}L exec (${execDepth >> 1} paired · 2.667x body)`,
          tone: 'recurrent_teal',
          discrepancyId: 'DR4',
        },
        {
          label: 'Shared Physical Weight ID',
          value: `${params.modelWeightId.slice(0, 12)} (0 loop adapters)`,
          tone: 'prelude_indigo',
        },
        {
          label: 'Live ||s_4 - s_1||_2 (8D)',
          value: disp14.toFixed(4),
          tone: 'nlerp_emerald',
        },
        {
          label: 'per_loop Persistent KV',
          value: `${banks.totalBanks} banks (${formatKiB(branch.kvBytes)})`,
          tone: 'swa_cyan',
          discrepancyId: 'DR18',
        },
      );
      break;
    }

    case 'act3-anchor-injection-and-pca-trajectory': {
      const m1 = record.lastTokenLoopMetrics[0];
      const m4 = record.lastTokenLoopMetrics[record.lastTokenLoopMetrics.length - 1];
      const cf = ctx.resolveCounterfactualCoda(1);
      const finalProbsT1 = softmaxFloat64(record.lastTokenScaledLogits, 0, config.vocabSize);
      const kl = computeKlDivergenceFloat64(finalProbsT1, cf.probs);
      const cfArgMax = argmaxIndex(cf.scaledLogits, 0, config.vocabSize);
      const finalArgMax = argmaxIndex(record.lastTokenScaledLogits, 0, config.vocabSize);

      const exitAnchorCos = m4?.anchorCosine ?? 0;
      const exitAnchorAngleDeg =
        (Math.acos(Math.max(-1, Math.min(1, exitAnchorCos))) * 180) / Math.PI;
      const entryAnchorAngleDeg = m4?.entryAnchorJacobian.angleDegrees ?? 0;

      metrics.push(
        {
          label: '||u_1 = N(2x)||_2 -> ||s_4||_2',
          value: `${(m1?.uNorm ?? 1).toFixed(5)} -> ${(m4?.sNorm ?? 1).toFixed(5)}`,
          tone: 'nlerp_emerald',
          discrepancyId: 'DR8',
        },
        {
          label: 'Anchor cos(s_4, x) & theta_4',
          value: `${fmtSigned(exitAnchorCos, 4)} (${exitAnchorAngleDeg.toFixed(1)} deg)`,
          tone: 'gold',
          discrepancyId: 'DR12',
        },
        {
          label: 'Local ||J_N||_2 (Boundary Only)',
          value: `${(m4?.entryAnchorJacobian.epsStabilizedJacobianNorm ?? 0.5).toFixed(4)} @ theta_entry4=${entryAnchorAngleDeg.toFixed(1)} deg (${m4?.entryAnchorJacobian.isLocalContractiveRegime ? '<1 local' : '>=1'})`,
          tone: 'warn',
        },
        {
          label: 'Counterfactual Coda t=1 KL',
          value: `${kl.toFixed(4)} nats ('${decodeDoryToken(cfArgMax)}' vs '${decodeDoryToken(finalArgMax)}')`,
          tone: 'coda_amber',
        },
      );
      break;
    }

    case 'act3-dual-axis-kv-and-last-loop-divergence': {
      const perLoopStage = scenarioMap.get('standard_k4_per_loop');
      const perLoopPrefill = perLoopStage
        ? tourSession.getRecord(perLoopStage.recordIds[0]!)
        : null;
      const chunkedDivergence =
        perLoopPrefill !== null
          ? computeMaxAbsDiff(record.lastTokenScaledLogits, perLoopPrefill.lastTokenScaledLogits)
          : 0;

      metrics.push(
        {
          label: 'Coord Loop vs. Historical Writer',
          value: `t=${coord.loopIteration} reads writerLoop=${coord.cacheOwnership.writerLoop}`,
          tone: 'gold',
          discrepancyId: 'DR17',
        },
        {
          label: 'Live max |z_last[4,8] - z_per[12]|',
          value: chunkedDivergence.toExponential(3),
          tone: chunkedDivergence > 1e-2 ? 'warn' : 'muted',
          discrepancyId: 'DR17',
        },
        {
          label: 'KV Bank Footprint (k=4)',
          value: `36 last_loop (${formatKiB(branch.kvBytes)}) vs. 96 per_loop`,
          tone: 'swa_cyan',
          discrepancyId: 'DR10',
        },
        {
          label: 'Recorded chunkSizes[] Replay',
          value: `[${record.chunkSizesAfter.join(', ')}] (0-ULP verbatim)`,
          tone: 'recurrent_teal',
          discrepancyId: 'DR24',
        },
      );
      break;
    }

    case 'act3-swa-eviction-and-256k-scaling': {
      const prod1kPl = computeProductionKvBytesBf16(1024, 4, 'per_loop');
      const prod256kPl = computeProductionKvBytesBf16(262_144, 4, 'per_loop');
      const prod256kLl = computeProductionKvBytesBf16(262_144, 4, 'last_loop');
      const evictedCount = Math.max(0, record.committedLen - config.slidingWindow);

      metrics.push(
        {
          label: 'Mini SWA Eviction State',
          value: `S=${record.committedLen} > W=${config.slidingWindow} (${evictedCount} evicted; cap_swa=${config.swaRingCapacity})`,
          tone: 'swa_cyan',
          discrepancyId: 'DR19',
        },
        {
          label: 'S-Invariant Ratio (4+5k)/9',
          value: `${prod1kPl.perLoopToLastLoopRatio.toFixed(4)}x at 1K & 256K`,
          tone: 'nlerp_emerald',
        },
        {
          label: 'Prod S=1,024 Knee (k=4)',
          value: `${formatMiB(prod1kPl.totalBytes, 1)} (${(prod1kPl.swaShare * 100).toFixed(1)}% SWA)`,
          tone: 'gold',
        },
        {
          label: 'Prod S=256K (per / last)',
          value: `${formatBytesAdaptive(prod256kPl.totalBytes)} / ${formatBytesAdaptive(prod256kLl.totalBytes)} (${(prod256kPl.swaShare * 100).toFixed(2)}% SWA)`,
          tone: 'full_indigo',
        },
      );
      break;
    }

    case 'act4-self-spec-contract-a-104-banks': {
      const specSummary = record.specSummary;
      const miniKv = computeMiniKvCacheBytes(DORY_SPEC_VERIFY_LOOPS, 'per_loop', true, config);
      const miniKvNoShallow = computeMiniKvCacheBytes(
        DORY_SPEC_VERIFY_LOOPS,
        'per_loop',
        false,
        config,
      );
      const shallowDeltaBytes = miniKv.kvPayloadBytes - miniKvNoShallow.kvPayloadBytes;

      metrics.push(
        {
          label: 'Contract A Bank Partition',
          value: `${DORY_SPEC_CONTRACT_A_BANK_COUNT} banks (8+20+60+8+8 · +${formatKiB(shallowDeltaBytes)} Shallow)`,
          tone: 'spec_oxide',
          discrepancyId: 'DR23',
        },
        {
          label: 'Prelude + Loop-1 Reuse',
          value: specSummary
            ? `${specSummary.reusedActivationPositions} / ${specSummary.verifiedPositions} verify positions reused`
            : 'Prefill committed (104 banks)',
          tone: 'nlerp_emerald',
          discrepancyId: 'DR23',
        },
        {
          label: 'Draft Horizon (k=1 -> k=4)',
          value: specSummary
            ? `drafted ${specSummary.draftCount} / gamma=${specSummary.gammaEffective}`
            : `gamma=${config.speculativeGamma}`,
          tone: 'recurrent_teal',
        },
        {
          label: 'Shallow Coda Catch-Up',
          value: specSummary
            ? specSummary.shallowCodaCatchupExecuted
              ? '1-token catch-up (full accept)'
              : '0 extra work (rejection path)'
            : 'Batched prefill',
          tone: 'coda_amber',
        },
      );
      break;
    }

    case 'act4-self-spec-binary64-rejection-rollback': {
      const specSummary = record.specSummary;

      metrics.push(
        {
          label: 'Cycle Outcome',
          value: specSummary
            ? `${specSummary.outcome} (${specSummary.acceptedDraftCount}/${specSummary.draftCount} accepted, ${specSummary.emittedTokens.length} emitted)`
            : 'prefill',
          tone: 'spec_oxide',
          discrepancyId: 'DR23',
        },
        {
          label: 'Binary64 Residual / Bonus',
          value:
            specSummary && specSummary.residualMassZ !== null
              ? `Z=${specSummary.residualMassZ.toFixed(5)} -> y*='${decodeDoryToken(specSummary.correctionOrBonusToken ?? 0)}'`
              : specSummary && specSummary.correctionOrBonusToken !== null
                ? `bonus y='${decodeDoryToken(specSummary.correctionOrBonusToken)}'`
                : 'EOS halt',
          tone: 'gold',
        },
        {
          label: 'cap_swa=16 Atomic Rollback',
          value: specSummary
            ? `${specSummary.rolledBackPositions} pos rewound (L: ${specSummary.startCommittedLen}->${specSummary.endCommittedLen}, W=${config.slidingWindow})`
            : `cap_swa=${config.swaRingCapacity} >= 13`,
          tone: 'swa_cyan',
          discrepancyId: 'DR19',
        },
        {
          label: 'Division-Free Rule',
          value: 'p[d] >= q[d] || r * q[d] < p[d]',
          tone: 'nlerp_emerald',
        },
      );
      break;
    }

    case 'act4-four-checkpoint-zero-byte-diff': {
      const bf16 = DORY_CHECKPOINT_FORMATS[0]!;
      const mxfp8 = DORY_CHECKPOINT_FORMATS[1]!;
      const nvfp4 = DORY_CHECKPOINT_FORMATS[2]!;
      const gguf = DORY_CHECKPOINT_FORMATS[3]!;

      metrics.push(
        {
          label: 'BF16 (507 tensors · 0 B diff)',
          value: `${formatBytesAdaptive(bf16.totalFileBytes)} (${(bf16.totalFileBytes / 1e9).toFixed(3)} GB)`,
          tone: 'prelude_indigo',
          discrepancyId: 'DR14',
        },
        {
          label: 'MXFP8-PTQ (795 tensors · MSE)',
          value: `${formatBytesAdaptive(mxfp8.totalFileBytes)} (${(mxfp8.totalFileBytes / 1e9).toFixed(3)} GB)`,
          tone: 'swa_cyan',
          discrepancyId: 'DR15',
        },
        {
          label: 'NVFP4-QAD (1,083 tensors)',
          value: `${formatBytesAdaptive(nvfp4.totalFileBytes)} (${(nvfp4.totalFileBytes / 1e9).toFixed(3)} GB)`,
          tone: 'nlerp_emerald',
          discrepancyId: 'DR14',
        },
        {
          label: 'GGUF Q5_K_M · License',
          value: `${formatBytesAdaptive(gguf.totalFileBytes)} · ${DORY_PRODUCTION_SPEC.weightLicense}`,
          tone: 'coda_amber',
          discrepancyId: 'DR16',
        },
      );
      break;
    }

    case 'act4-readonly-tbptl-and-paper-results': {
      const mem = tourSession.measureMemory();
      const k4Stage = scenarioMap.get('standard_k4_per_loop');
      const k4Prefill = k4Stage ? tourSession.getRecord(k4Stage.recordIds[0]!) : null;
      const k1Stage = scenarioMap.get('standard_k1_per_loop');
      const k1Prefill = k1Stage ? tourSession.getRecord(k1Stage.recordIds[0]!) : null;
      const depthShift =
        k4Prefill !== null && k1Prefill !== null
          ? computeMaxAbsDiff(k4Prefill.lastTokenScaledLogits, k1Prefill.lastTokenScaledLogits)
          : 0;

      metrics.push(
        {
          label: 'TBPTL Schematic Mode',
          value: `${DORY_TBPTL_READONLY_BADGE} (depth=1 · 0 early-exit)`,
          tone: 'readonly',
        },
        {
          label: 'Isolated Tour Session Memory',
          value: `${formatMiB(mem.totalSessionRetainedBytes, 2)} / ${formatMiB(mem.maxSessionRetainedBytes, 1)} (non-aliased=${mem.nonAliasedSlabsVerified})`,
          tone: mem.withinSessionBudget ? 'nlerp_emerald' : 'warn',
        },
        {
          label: 'Live k=4 vs. k=1 Logit Shift',
          value: depthShift.toFixed(4),
          tone: 'recurrent_teal',
        },
        {
          label: 'Paper §6.2 Curves (DR13)',
          value: '1.37–1.71x (1K & 16K · Paper only)',
          tone: 'muted',
          discrepancyId: 'DR13',
        },
      );
      break;
    }

    default: {
      metrics.push({
        label: 'Executed Sublayer',
        value: `e=${coord.executedLayerIdx} (p=${coord.physicalLayerIdx}, t=${coord.loopIteration})`,
        tone: 'recurrent_teal',
      });
    }
  }

  return Object.freeze(metrics);
}

function buildBeatProvenanceBadges(
  spec: DoryTourBeatSpec,
  ctx: DoryResolvedSessionContext,
  coord: DoryExecutionCoord,
): readonly DoryProvenanceBadge[] {
  const layerSpec = getDoryPhysicalLayerSpec(coord.physicalLayerIdx);
  const badges: DoryProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    {
      label: `Coord: p=${coord.physicalLayerIdx} (${layerSpec.transformersPairBadge}) · e=${coord.executedLayerIdx} · t=${coord.loopIteration} · pos=${coord.absTokenPos}`,
      tone:
        layerSpec.stage === 'input'
          ? 'prelude_indigo'
          : layerSpec.stage === 'recurrent'
            ? 'recurrent_teal'
            : 'coda_amber',
      discrepancyId: 'DR4',
    },
  ];

  if (spec.viewMode === 'hypersphere') {
    badges.push({
      label: DORY_ANTI_SLERP_PLAQUE,
      tone: 'nlerp_emerald',
      discrepancyId: 'DR7',
    });
    if (spec.id === 'act3-anchor-injection-and-pca-trajectory') {
      badges.push(
        {
          label: DORY_EXPLORATORY_PCA_BADGE,
          tone: 'muted',
        },
        {
          label: DORY_ANCHOR_MAP_CAVEAT_BADGE,
          tone: 'warn',
        },
        {
          label: DORY_COUNTERFACTUAL_CODA_BADGE,
          tone: 'coda_amber',
        },
      );
    }
  } else if (spec.viewMode === 'bookends') {
    badges.push({
      label: DORY_TBPTL_READONLY_BADGE,
      tone: 'readonly',
    });
    if (spec.id === 'act4-readonly-tbptl-and-paper-results') {
      badges.push({
        label: DORY_PAPER_RESULTS_BADGE,
        tone: 'muted',
        discrepancyId: 'DR13',
      });
    }
  } else if (spec.viewMode === 'self_spec') {
    // ctx.provenanceBadges already includes DORY_SPEC_CONTRACT_A_BADGE when executionMode==='self_spec'
    if (!badges.some((b) => b.label === DORY_SPEC_CONTRACT_A_BADGE)) {
      badges.push({
        label: DORY_SPEC_CONTRACT_A_BADGE,
        tone: 'spec_oxide',
        discrepancyId: 'DR23',
      });
    }
  }

  return Object.freeze(badges);
}

// ============================================================================
// FALLBACK SCENE DISPATCH FOR VIEWS 1–6 (WHEN `sceneDispatcher` IS OMITTED)
// ============================================================================

function dispatchFallbackBeatScene(
  ctx: DoryResolvedSessionContext,
  viewMode: DorySpatialViewMode,
  focus: DorySpatialFocusState,
): DorySpatialScene {
  switch (viewMode) {
    case 'macro':
      return buildDoryMacroScene(ctx, {
        displayMode: focus.macroDisplayMode,
        selectedPhysicalLayerIdx: focus.selectedPhysicalLayerIdx,
        selectedLoopIteration: focus.selectedLoopIteration,
        hoveredPartId: focus.hoveredPartId,
        hoveredCell: focus.hoveredCell,
        recordId: focus.recordId,
      });
    case 'hypersphere':
      return buildDoryHypersphereScene(ctx, {
        physicalLayerIdx: focus.selectedPhysicalLayerIdx,
        loopIteration: focus.selectedLoopIteration,
        tokenOffset: focus.selectedTokenOffset ?? undefined,
        includeCounterfactualCoda: focus.includeCounterfactualCoda,
        counterfactualLoop: focus.counterfactualLoop,
        hoveredCell: focus.hoveredCell,
        recordId: focus.recordId,
      });
    case 'gqa_rope':
      return buildDoryGqaRopeScene(ctx, {
        physicalLayerIdx: focus.gqaLayerIdx,
        loopIteration: focus.selectedLoopIteration,
        selectedQHead: focus.selectedQHead,
        selectedQueryRow: focus.selectedQueryRow ?? undefined,
        selectedKeyCol: focus.selectedKeyCol ?? undefined,
        selectedRotaryPlane: focus.selectedRotaryPlane,
        hoveredCell: focus.hoveredCell,
        recordId: focus.recordId,
      });
    case 'swiglu':
      return buildDorySwigluScene(ctx, {
        physicalLayerIdx: focus.swigluLayerIdx,
        loopIteration: focus.selectedLoopIteration,
        selectedTokenRow: focus.selectedTokenOffset ?? undefined,
        selectedChannel: focus.swigluChannel,
        selectedOutputDim: focus.swigluOutputDim,
        accumReveal: focus.swigluAccumReveal,
        hoveredCell: focus.hoveredCell,
        recordId: focus.recordId,
      });
    case 'kv_compare':
      return buildDoryKvCompareScene(ctx, {
        partitionMode: focus.kvComparePartitionMode,
        chunkPartition: focus.kvCompareChunkPartition ?? undefined,
        physicalLayerIdx: focus.kvCompareLayerIdx,
        loopIteration: focus.selectedLoopIteration,
        kLoopsOverride: focus.kvCompareKLoopsOverride ?? undefined,
        hoveredCell: focus.hoveredCell,
        recordId: focus.recordId,
      });
    case 'self_spec':
      return buildDorySelfSpecScene(ctx, {
        selectedVerifyOffset: focus.selectedVerifyOffset,
        selectedAttentionLayerIdx: focus.selfSpecAttentionLayerIdx,
        previewSampling: focus.selfSpecPreviewSampling ?? undefined,
        hoveredCell: focus.hoveredCell,
        recordId: focus.recordId,
      });
    case 'bookends':
      throw new Error(
        "DoryGuidedTourController.buildBeatScene() for viewMode='bookends' requires options.sceneDispatcher (pass buildDorySpatialScene from 'src/dory/scene/index.ts').",
      );
  }
}

// ============================================================================
// GUIDED WALKTHROUGH CONTROLLER FACTORY (`createDoryGuidedTour`)
// ============================================================================

/**
 * Creates a strictly isolated 4-Act Guided Walkthrough controller backed by its own private
 * `DorySession` (`D6`, `D7`, `D10`). Never references or mutates any caller Free-Explore session.
 *
 * Pass `{ sceneDispatcher: buildDorySpatialScene }` (from `src/dory/scene/index.ts`) to dispatch
 * all 7 views including View 7 (`bookends`) through `buildBeatScene()`.
 */
export function createDoryGuidedTour(options: DoryTourOptions = {}): DoryGuidedTourController {
  const params: DoryModelParameters = options.params ?? getDefaultDoryModelParameters();
  const ropeTables: DoryDualRopeTables =
    options.ropeTables ?? buildDoryDualRopeTables(params.config);
  const rawSeed = options.seed ?? DORY_TOUR_SEED;
  if (!Number.isFinite(rawSeed)) {
    throw new RangeError(`Invalid Dory tour seed: ${String(rawSeed)}.`);
  }
  const seed = rawSeed >>> 0;
  const sceneDispatcher = options.sceneDispatcher;

  // Strictly private isolated DorySession (never aliased to any caller session!)
  const tourSession: DorySession = createDorySession({
    params,
    ropeTables,
    autoStartInitialBranch: false,
  });

  const scenarioMap = stageTourScenarios(tourSession, seed);

  // Per-beat cached non-mutating contexts and resolved beat snapshots
  const contextCache = new Map<number, DoryResolvedSessionContext>();
  const beatSnapshotCache = new Map<number, DoryTourBeatSnapshot>();

  const totalBeats = DORY_TOUR_BEATS.length;
  let status: DoryTourStatus = 'idle';
  let currentBeatIndex = 0;
  let savedRestoreState: DoryFreeExploreRestoreState | null = null;

  function assertValidBeatIndex(beatIndex: number): number {
    if (!Number.isInteger(beatIndex) || beatIndex < 0 || beatIndex >= totalBeats) {
      throw new RangeError(
        `Invalid Dory tour beatIndex=${String(beatIndex)}: expected integer in 0..${totalBeats - 1}.`,
      );
    }
    return beatIndex;
  }

  function clampBeatIndex(beatIndex: number): number {
    if (!Number.isFinite(beatIndex)) {
      throw new RangeError(`Invalid Dory tour beatIndex=${String(beatIndex)}: expected finite number.`);
    }
    return Math.max(0, Math.min(totalBeats - 1, Math.trunc(beatIndex)));
  }

  function getTourContext(beatIndex?: number): DoryResolvedSessionContext {
    const idx = beatIndex === undefined ? currentBeatIndex : assertValidBeatIndex(beatIndex);
    const cached = contextCache.get(idx);
    if (cached !== undefined) {
      return cached;
    }
    const spec = DORY_TOUR_BEATS[idx]!;
    const staged = scenarioMap.get(spec.scenarioKey);
    if (staged === undefined) {
      throw new Error(`Missing staged tour scenario '${spec.scenarioKey}' for beat '${spec.id}'.`);
    }
    const recordId = resolveScenarioRecordId(staged, spec.targetStepOffset);
    const ctx = resolveDorySessionContext(tourSession, recordId);
    contextCache.set(idx, ctx);
    return ctx;
  }

  function resolveBeatSnapshot(beatIndex: number): DoryTourBeatSnapshot {
    const idx = assertValidBeatIndex(beatIndex);
    const cached = beatSnapshotCache.get(idx);
    if (cached !== undefined) {
      return cached;
    }

    const spec = DORY_TOUR_BEATS[idx]!;
    const actDesc = DORY_TOUR_ACT_DESCRIPTORS[spec.actIndex]!;
    const ctx = getTourContext(idx);
    const trace = ctx.resolveForwardTrace();
    const { physicalLayerIdx, loopIteration, tokenOffset, expectedBankKind, expectedCacheRole } =
      spec.coordTarget;

    const subRecord = ctx.resolveSublayerRecord(physicalLayerIdx, loopIteration);
    const rawCoord = subRecord.coord;

    const resolvedTokenOffset =
      tokenOffset === 'last'
        ? ctx.resolveDefaultTokenOffset()
        : Math.max(0, Math.min(trace.seqLen - 1, Math.trunc(tokenOffset)));
    const resolvedAbsTokenPos = trace.startPos + resolvedTokenOffset;

    const coord: DoryExecutionCoord = Object.freeze({
      modelWeightId: rawCoord.modelWeightId,
      physicalLayerIdx: rawCoord.physicalLayerIdx,
      executedLayerIdx: rawCoord.executedLayerIdx,
      loopIteration: rawCoord.loopIteration,
      absTokenPos: resolvedAbsTokenPos,
      forwardStepId: rawCoord.forwardStepId,
      sessionBranchId: rawCoord.sessionBranchId,
      cacheOwnership: Object.freeze({
        policy: rawCoord.cacheOwnership.policy,
        bankKind: rawCoord.cacheOwnership.bankKind,
        bankSlotId: rawCoord.cacheOwnership.bankSlotId,
        writerLoop: rawCoord.cacheOwnership.writerLoop,
        role: rawCoord.cacheOwnership.role,
      }),
    });

    // Hard structural invariants on every beat's live 8-tuple coordinate
    if (expectedBankKind !== undefined && coord.cacheOwnership.bankKind !== expectedBankKind) {
      throw new Error(
        `Tour beat '${spec.id}' bankKind mismatch: got '${coord.cacheOwnership.bankKind}', expected '${expectedBankKind}'.`,
      );
    }
    if (expectedCacheRole !== undefined && coord.cacheOwnership.role !== expectedCacheRole) {
      throw new Error(
        `Tour beat '${spec.id}' cacheRole mismatch: got '${coord.cacheOwnership.role}', expected '${expectedCacheRole}'.`,
      );
    }
    const validation = validateDoryExecutionCoord(coord, params.modelWeightId, ctx.kLoops);
    if (!validation.valid) {
      throw new Error(
        `Tour beat '${spec.id}' produced invalid 8-tuple DoryExecutionCoord: ${validation.errors.join(' | ')}`,
      );
    }

    const focus: DorySpatialFocusState = Object.freeze({
      ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
      ...spec.focusPatch,
      ...(spec.focusPatch.kvCompareChunkPartition != null
        ? { kvCompareChunkPartition: Object.freeze([...spec.focusPatch.kvCompareChunkPartition]) }
        : {}),
      ...(spec.focusPatch.selfSpecPreviewSampling != null
        ? {
            selfSpecPreviewSampling: Object.freeze({
              ...spec.focusPatch.selfSpecPreviewSampling,
            }),
          }
        : {}),
      recordId: ctx.record.recordId,
    });

    const liveMetrics = buildLiveBeatMetrics({
      spec,
      ctx,
      coord,
      tokenOffset: resolvedTokenOffset,
      tourSession,
      scenarioMap,
    });

    const provenanceBadges = buildBeatProvenanceBadges(spec, ctx, coord);

    const viewDesc = DORY_SPATIAL_VIEW_MODES.find((v) => v.id === spec.viewMode);
    const cameraOverride =
      spec.cameraOverride ??
      (viewDesc !== undefined
        ? Object.freeze({
            azimuthDeg: viewDesc.defaultCameraAzimuthDeg,
            elevationDeg: viewDesc.defaultCameraElevationDeg,
          })
        : null);

    const snapshot: DoryTourBeatSnapshot = Object.freeze({
      beatIndex: idx,
      totalBeats,
      beatId: spec.id,
      actId: spec.actId,
      actNumber: spec.actNumber,
      actIndex: spec.actIndex,
      beatInAct: spec.beatInAct,
      actTitle: actDesc.title,
      title: spec.title,
      subtitle: spec.subtitle,
      takeaway: spec.takeaway,
      narrative: spec.narrative,
      equation: spec.equation ?? null,
      epistemicNote: spec.epistemicNote,
      discrepancyRefs: spec.discrepancyRefs,
      designDecisionRefs: spec.designDecisionRefs,
      viewMode: spec.viewMode,
      scenarioKey: spec.scenarioKey,
      branchId: ctx.branch.branchId,
      recordId: ctx.record.recordId,
      kLoops: ctx.kLoops,
      kvCacheMode: ctx.kvCacheMode,
      executionMode: ctx.executionMode,
      coord,
      focus,
      highlightPartId: spec.highlightPartId ?? null,
      cameraOverride,
      liveMetrics,
      provenanceBadges,
    });

    beatSnapshotCache.set(idx, snapshot);
    return snapshot;
  }

  function syncPrivateSessionCursor(beat: DoryTourBeatSnapshot): void {
    const activeRec = tourSession.getActiveRecord();
    if (activeRec === null || activeRec.recordId !== beat.recordId) {
      tourSession.seekRecord(beat.recordId);
    }
  }

  function buildCurrentState(): DoryTourState {
    const currentBeat = resolveBeatSnapshot(currentBeatIndex);
    syncPrivateSessionCursor(currentBeat);
    const acts = buildActSummaries(DORY_TOUR_BEATS, currentBeat.actIndex);
    const currentAct = acts[currentBeat.actIndex]!;

    return Object.freeze({
      status,
      active: status === 'active',
      beatIndex: currentBeatIndex,
      beatCount: totalBeats,
      actIndex: currentBeat.actIndex,
      actCount: DORY_TOUR_TOTAL_ACTS,
      currentAct,
      currentBeat,
      acts,
      activeBranchId: currentBeat.branchId,
      activeRecordId: currentBeat.recordId,
      viewMode: currentBeat.viewMode,
      focus: currentBeat.focus,
      coord: currentBeat.coord,
    });
  }

  // Align initial cursor to Beat 0 without activating the tour yet
  syncPrivateSessionCursor(resolveBeatSnapshot(0));

  const controller: DoryGuidedTourController = {
    get acts(): readonly DoryTourActSummary[] {
      return buildActSummaries(DORY_TOUR_BEATS, DORY_TOUR_BEATS[currentBeatIndex]!.actIndex);
    },
    beatSpecs: DORY_TOUR_BEATS,
    totalActs: DORY_TOUR_TOTAL_ACTS,
    totalBeats,
    tourSession,

    getStatus(): DoryTourStatus {
      return status;
    },

    isActive(): boolean {
      return status === 'active';
    },

    getState(): DoryTourState {
      return buildCurrentState();
    },

    getBeatCount(): number {
      return totalBeats;
    },

    getCurrentBeatIndex(): number {
      return currentBeatIndex;
    },

    getCurrentBeat(): DoryTourBeatSnapshot {
      return resolveBeatSnapshot(currentBeatIndex);
    },

    getBeat(beatIndex: number): DoryTourBeatSnapshot {
      return resolveBeatSnapshot(beatIndex);
    },

    getAllBeats(): readonly DoryTourBeatSnapshot[] {
      const all: DoryTourBeatSnapshot[] = new Array<DoryTourBeatSnapshot>(totalBeats);
      for (let i = 0; i < totalBeats; i += 1) {
        all[i] = resolveBeatSnapshot(i);
      }
      return Object.freeze(all);
    },

    getActs(): readonly DoryTourActSummary[] {
      return buildActSummaries(DORY_TOUR_BEATS, DORY_TOUR_BEATS[currentBeatIndex]!.actIndex);
    },

    getRestoreState(): DoryFreeExploreRestoreState | null {
      return savedRestoreState;
    },

    start(
      restoreState?: DoryFreeExploreRestoreState | null,
      initialBeatIndex = 0,
    ): DoryTourState {
      if (restoreState !== undefined) {
        savedRestoreState = cloneRestoreState(restoreState);
      } else if (status !== 'active') {
        savedRestoreState = null;
      }
      currentBeatIndex = clampBeatIndex(initialBeatIndex);
      status = 'active';
      return buildCurrentState();
    },

    goToBeat(beatIndex: number): DoryTourState {
      currentBeatIndex = clampBeatIndex(beatIndex);
      return buildCurrentState();
    },

    nextBeat(): DoryTourState {
      currentBeatIndex = Math.min(totalBeats - 1, currentBeatIndex + 1);
      return buildCurrentState();
    },

    prevBeat(): DoryTourState {
      currentBeatIndex = Math.max(0, currentBeatIndex - 1);
      return buildCurrentState();
    },

    goToAct(actId: DoryTourActId): DoryTourState {
      const acts = buildActSummaries(DORY_TOUR_BEATS, DORY_TOUR_BEATS[currentBeatIndex]!.actIndex);
      const target = acts.find((a) => a.id === actId);
      if (target === undefined) {
        throw new RangeError(`Unknown Dory tour actId='${String(actId)}'.`);
      }
      currentBeatIndex = target.startBeatIndex;
      return buildCurrentState();
    },

    buildBeatScene(beatIndex?: number): DorySpatialScene {
      const idx = beatIndex === undefined ? currentBeatIndex : assertValidBeatIndex(beatIndex);
      const beat = resolveBeatSnapshot(idx);
      const ctx = getTourContext(idx);
      if (sceneDispatcher !== undefined) {
        return sceneDispatcher(ctx, beat.viewMode, beat.focus);
      }
      return dispatchFallbackBeatScene(ctx, beat.viewMode, beat.focus);
    },

    getTourContext(beatIndex?: number): DoryResolvedSessionContext {
      return getTourContext(beatIndex);
    },

    measureTourMemory(): DorySessionMemoryAudit {
      return tourSession.measureMemory();
    },

    exit(): DoryFreeExploreRestoreState | null {
      status = 'exited';
      return savedRestoreState;
    },
  };

  return controller;
}

// Compile-time verification against frozen `DoryWalkthroughModuleContract`
const _DORY_WALKTHROUGH_MODULE_CONTRACT_CHECK: DoryWalkthroughModuleContract = {
  DORY_TOUR_BEATS,
  validateDoryExecutionCoord,
  createDoryGuidedTour,
};
void _DORY_WALKTHROUGH_MODULE_CONTRACT_CHECK;
