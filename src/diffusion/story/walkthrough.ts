/**
 * Nemotron-Labs-Diffusion-14B Pure Headless 4-Act Guided Walkthrough Engine
 * (`src/diffusion/story/walkthrough.ts`)
 *
 * Pure headless TypeScript — strictly isolated from browser globals, rendering APIs,
 * non-deterministic RNG, and sibling visualizers.
 *
 * Architecture:
 *   - Runs its OWN dedicated, deterministic, strictly isolated `DiffusionSession`
 *     (`samplingSeed = 0x44463401`, `blockLength = 4`, `maxNewTokens = 16`, `temperature = 0`).
 *     Never touches or mutates any external Free-Explore `DiffusionSession`.
 *   - Pre-builds 5 deterministic branches well within `maxBranches = 12`, `maxRecords = 128`,
 *     and `<= 12.00 MiB`:
 *       1. AR (`token_causal`) stepping past `pos = 16` (`kvSeqLen >= 17`) so Act 2 & Act 3
 *          exhibit the live Llama-4 query temperature scale transition (`1.0000×` → `1.0693×`).
 *       2. DLM Causal (`causal_context = true`, `token_causal`) forked from prefill.
 *       3. DLM Bidir (`causal_context = false`, `block_bidirectional`) via explicit re-prefill.
 *       4. Linear Spec (LoRA OFF, `token_causal`) forked from prefill.
 *       5. Linear Spec + LoRA (draft-only `o_proj` LoRA ON; verify OFF) forked from prefill.
 *   - 26 beats across 4 Acts (DF0 Rev 2 §12) covering all 7 spatial views, all 5 Mask Explorer
 *     tabs, and all 4 live execution modes + `causal_context` T/F.
 *   - Every beat's `liveMetrics` are computed 100% from real live tensors, counters, KV snapshots,
 *     and memory reports on the isolated tour session — ZERO fabricated values or speedup claims.
 */

import type { CameraPose } from '../../render/camera';
import { createDefaultDiffusionModel } from '../parameters';
import { computeLlama4AttnScale, computeYarnInvFreq } from '../rope';
import {
  DIFFUSION_EPISTEMIC_NOTE,
  resolveDiffusionSessionContext,
} from '../scene/provenance';
import {
  DEFAULT_DIFFUSION_FOCUS_STATE,
  type DiffusionFocusState,
  type DiffusionPickIdentity,
  type DiffusionResolvedSessionContext,
  type DiffusionSpatialViewMode,
} from '../scene/types';
import { createDiffusionSession, type DiffusionSession } from '../session';
import {
  DIFFUSION_DYADIC_YARN_INV_FREQ,
  DIFFUSION_MINI_MASK_TOKEN_ID,
  DIFFUSION_MINI_VOCAB,
  type DiffusionModelParameters,
  type DiffusionSessionRecord,
} from '../types';
import type {
  DiffusionFreeExploreRestoreState,
  DiffusionGuidedTourController,
  DiffusionTourActId,
  DiffusionTourActSummary,
  DiffusionTourBeatSnapshot,
  DiffusionTourLiveMetric,
  DiffusionTourStatus,
} from './types';

// ============================================================================
// ACT DEFINITIONS & CONSTANTS
// ============================================================================

export const DIFFUSION_TOUR_SAMPLING_SEED = 0x44463401 as const;

export interface DiffusionTourActDef {
  readonly actIndex: number;
  readonly id: DiffusionTourActId;
  readonly title: string;
  readonly shortTitle: string;
}

export const DIFFUSION_TOUR_ACT_DEFS: readonly DiffusionTourActDef[] = Object.freeze([
  Object.freeze({
    actIndex: 0,
    id: 'act-1' as const,
    title: 'Shared 40L Backbone & [x_t ‖ x_0]',
    shortTitle: 'Act 1 · 40L & [x_t‖x_0]',
  }),
  Object.freeze({
    actIndex: 1,
    id: 'act-2' as const,
    title: 'YaRN RoPE, Llama-4 & SwiGLU',
    shortTitle: 'Act 2 · YaRN & SwiGLU',
  }),
  Object.freeze({
    actIndex: 2,
    id: 'act-3' as const,
    title: 'AR vs. Block Diffusion (causal_context T/F)',
    shortTitle: 'Act 3 · AR vs. DLM',
  }),
  Object.freeze({
    actIndex: 3,
    id: 'act-4' as const,
    title: 'Linear Self-Speculation, LoRA & KV Rollback',
    shortTitle: 'Act 4 · Spec, LoRA & KV',
  }),
]);

const EPISTEMIC_MINIATURE_SIM = DIFFUSION_EPISTEMIC_NOTE;
const EPISTEMIC_PROD_CONTRAST =
  'Live 40L miniature tensors (d=8, 32Q/8KV, V=36) contrasted against verified 0 B LFS 14B Instruct specs — NOT 14B checkpoint weights.';
const EPISTEMIC_VLM_REF =
  'Read-only VLM-8B Quadratic Self-Speculation reference mask (D10: absent from all 6 text checkpoints; D24: upstream VLM self_spec_generate is greedy-only).';

// ============================================================================
// PURE TENSOR & CLONING HELPERS
// ============================================================================

function safeNum(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return '0';
  return v.toFixed(digits);
}

function l2Norm(arr: ArrayLike<number>, start = 0, len?: number): number {
  const end = len !== undefined ? Math.min(arr.length, start + len) : arr.length;
  let sum = 0;
  for (let i = start; i < end; i++) {
    const x = arr[i] ?? 0;
    sum += x * x;
  }
  return Math.sqrt(sum);
}

function maxAbs(arr: ArrayLike<number>, start = 0, len?: number): number {
  const end = len !== undefined ? Math.min(arr.length, start + len) : arr.length;
  let m = 0;
  for (let i = start; i < end; i++) {
    const a = Math.abs(arr[i] ?? 0);
    if (a > m) m = a;
  }
  return m;
}

function formatMiB(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0.00 MiB';
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function tokenText(tokenId: number): string {
  return DIFFUSION_MINI_VOCAB[tokenId] ?? `#${tokenId}`;
}

function formatModeLabel(record: DiffusionSessionRecord): string {
  const cfg = record.configAtStep;
  if (cfg.mode === 'ar') return 'AR';
  if (cfg.mode === 'dlm') return cfg.causalContext ? 'DLM Causal' : 'DLM Bidir';
  if (cfg.mode === 'linear_spec') return 'Linear Spec';
  return 'Spec + LoRA';
}

function formatProvenanceSummary(record: DiffusionSessionRecord): string {
  const loraFlag =
    record.configAtStep.mode === 'linear_spec_lora' && record.kind === 'spec_cycle'
      ? 'o_proj LoRA ON (draft only; verify OFF)'
      : 'o_proj LoRA OFF';
  return `${formatModeLabel(record)} · KV ${record.kvMaskProvenance} (${record.kvSnapshot.seqLen}/${record.kvSnapshot.capacity}) · ${loraFlag}`;
}

function makeFocus(overrides: Partial<DiffusionFocusState>): DiffusionFocusState {
  return Object.freeze({
    ...DEFAULT_DIFFUSION_FOCUS_STATE,
    ...overrides,
  });
}

function clonePickIdentity(cell: DiffusionPickIdentity | null): DiffusionPickIdentity | null {
  return cell ? { ...cell } : null;
}

function cloneFocusState(focus: DiffusionFocusState): DiffusionFocusState {
  return {
    ...focus,
    hoveredCell: clonePickIdentity(focus.hoveredCell),
    pinnedCell: clonePickIdentity(focus.pinnedCell),
  };
}

function cloneCameraPose(pose: CameraPose): CameraPose {
  return {
    target: [pose.target[0], pose.target[1], pose.target[2]],
    azimuthDeg: pose.azimuthDeg,
    elevationDeg: pose.elevationDeg,
    distance: pose.distance,
    fovDeg: pose.fovDeg,
  };
}

function cloneRestoreState(
  snapshot: DiffusionFreeExploreRestoreState,
): DiffusionFreeExploreRestoreState {
  return {
    activeRecordId: snapshot.activeRecordId,
    activeBranchId: snapshot.activeBranchId,
    focus: cloneFocusState(snapshot.focus),
    cameraPose: cloneCameraPose(snapshot.cameraPose),
    inspectorTab: snapshot.inspectorTab,
    inspectorOpen: snapshot.inspectorOpen,
    ropeScrubPosition: snapshot.ropeScrubPosition,
  };
}

function findTop1FromLastRow(
  record: DiffusionSessionRecord,
  vocabSize: number,
): { readonly tokenId: number; readonly logit: number; readonly prob: number; readonly maskProb: number } {
  const cp = record.primaryPassCheckpoint;
  const rows = Math.max(1, Math.floor(cp.logits.length / vocabSize));
  const rowOffset = (rows - 1) * vocabSize;
  let bestId = 0;
  let bestLogit = cp.logits[rowOffset] ?? 0;
  let bestProb = cp.probs[rowOffset] ?? 0;
  for (let v = 1; v < vocabSize; v++) {
    const val = cp.logits[rowOffset + v] ?? -1e9;
    if (val > bestLogit) {
      bestLogit = val;
      bestId = v;
      bestProb = cp.probs[rowOffset + v] ?? 0;
    }
  }
  const maskProb = cp.probs[rowOffset + DIFFUSION_MINI_MASK_TOKEN_ID] ?? 0;
  return { tokenId: bestId, logit: bestLogit, prob: bestProb, maskProb };
}

// ============================================================================
// ISOLATED TOUR SESSION PRE-BUILDER
// ============================================================================

interface TourBranchAnchors {
  readonly prefillRec: DiffusionSessionRecord;
  readonly arEarlyRec: DiffusionSessionRecord;
  readonly arBoundaryBeforeRec: DiffusionSessionRecord;
  readonly arBoundaryCrossRec: DiffusionSessionRecord;
  readonly arTipRec: DiffusionSessionRecord;
  readonly dlmCausalRec1: DiffusionSessionRecord;
  readonly dlmCausalRec2: DiffusionSessionRecord;
  readonly dlmBidirPrefillRec: DiffusionSessionRecord;
  readonly dlmBidirRec1: DiffusionSessionRecord;
  readonly dlmBidirRec2: DiffusionSessionRecord;
  readonly specBaseRec1: DiffusionSessionRecord;
  readonly specBaseRec2: DiffusionSessionRecord;
  readonly specLoraRec1: DiffusionSessionRecord;
  readonly specLoraRec2: DiffusionSessionRecord;
}

function buildIsolatedTourSession(model: DiffusionModelParameters): {
  readonly session: DiffusionSession;
  readonly anchors: TourBranchAnchors;
} {
  const session = createDiffusionSession(model, {
    config: {
      mode: 'ar',
      causalContext: true,
      blockLength: 4,
      maxNewTokens: 16,
      temperature: 0,
      samplingSeed: DIFFUSION_TOUR_SAMPLING_SEED,
    },
  });

  // 1. Initial causal prefill (kvSeqLen = 8, prompt tokens 0..7)
  const prefillRec = session.step();

  // 2. AR branch stepping past pos = 16 (`kvSeqLen >= 17`) so Llama-4 scale 1.0000x -> 1.0693x is live
  const arSteps: DiffusionSessionRecord[] = [];
  for (let i = 0; i < 10; i++) {
    const prevId = session.getState().activeRecordId;
    const rec = session.step();
    if (rec.recordId === prevId) break;
    arSteps.push(rec);
    if (rec.kvSnapshot.seqLen >= 18) break;
  }
  const arEarlyRec = arSteps[1] ?? arSteps[0] ?? prefillRec;
  const arBoundaryBeforeRec =
    arSteps.find((r) => r.kvSnapshot.seqLen === 16) ??
    arSteps.find((r) => r.kvSnapshot.seqLen <= 16 && r.kvSnapshot.seqLen >= 12) ??
    arEarlyRec;
  const arBoundaryCrossRec =
    arSteps.find((r) => r.kvSnapshot.seqLen >= 17) ??
    arSteps[arSteps.length - 1] ??
    prefillRec;
  const arTipRec = arSteps[arSteps.length - 1] ?? prefillRec;

  // 3. DLM Causal branch (`causal_context = true`, forked from prefill)
  session.seekRecord(prefillRec.recordId);
  session.updateConfig({
    mode: 'dlm',
    causalContext: true,
    blockLength: 4,
    maxNewTokens: 16,
    temperature: 0,
    samplerAThreshold: null,
  });
  const dlmCausalRec1 = session.step();
  const dlmCausalRec2 = session.step();

  // 4. DLM Bidir branch (`causal_context = false`, explicit re-prefill from prefill)
  session.seekRecord(prefillRec.recordId);
  session.updateConfig({
    mode: 'dlm',
    causalContext: false,
    blockLength: 4,
    maxNewTokens: 16,
    temperature: 0,
    samplerAThreshold: null,
    allowExplicitRePrefill: true,
  });
  const dlmBidirPrefillRec = session.getActiveRecord() ?? prefillRec;
  const dlmBidirRec1 = session.step();
  const dlmBidirRec2 = session.step();

  // 5. Linear Spec (LoRA OFF) branch forked from prefill
  session.seekRecord(prefillRec.recordId);
  session.updateConfig({
    mode: 'linear_spec',
    causalContext: true,
    blockLength: 4,
    maxNewTokens: 16,
    temperature: 0,
    samplerBThreshold: 0.0,
  });
  const specBaseRec1 = session.step();
  const specBaseRec2 = session.step();

  // 6. Linear Spec + LoRA (draft-only o_proj LoRA ON; verify OFF) forked from prefill
  session.seekRecord(prefillRec.recordId);
  session.updateConfig({
    mode: 'linear_spec_lora',
    causalContext: true,
    blockLength: 4,
    maxNewTokens: 16,
    temperature: 0,
    samplerBThreshold: 0.0,
  });
  const specLoraRec1 = session.step();
  const specLoraRec2 = session.step();

  // Leave session resting on Beat 0's record
  session.seekRecord(prefillRec.recordId);

  return {
    session,
    anchors: {
      prefillRec,
      arEarlyRec,
      arBoundaryBeforeRec,
      arBoundaryCrossRec,
      arTipRec,
      dlmCausalRec1,
      dlmCausalRec2,
      dlmBidirPrefillRec,
      dlmBidirRec1,
      dlmBidirRec2,
      specBaseRec1,
      specBaseRec2,
      specLoraRec1,
      specLoraRec2,
    },
  };
}

// ============================================================================
// 26-BEAT AUTHORED CATALOG (100% LIVE TENSOR-DERIVED METRICS)
// ============================================================================

interface BeatSpec {
  readonly beatId: string;
  readonly actId: DiffusionTourActId;
  readonly title: string;
  readonly subtitle: string;
  readonly narrative: string;
  readonly takeaway: string;
  readonly viewMode: DiffusionSpatialViewMode;
  readonly focus: DiffusionFocusState;
  readonly record: DiffusionSessionRecord;
  readonly ropeScrubPosition: number | null;
  readonly epistemicNote: string;
  readonly liveMetrics: readonly DiffusionTourLiveMetric[];
}

function buildTourBeats(
  session: DiffusionSession,
  anchors: TourBranchAnchors,
): readonly DiffusionTourBeatSnapshot[] {
  const cfg = session.model.config;
  const yarn = computeYarnInvFreq(cfg.rope);
  const mem = session.measureMemory();
  const state = session.getState();

  // Live layer traces reconstructed with 0-ULP parity from the isolated tour session
  const prefillEmbNorm = l2Norm(anchors.prefillRec.primaryPassCheckpoint.residualCheckpoints[0]!);
  const arEarlyL0 = session.reconstructLayerTrace(anchors.arEarlyRec.recordId, 0);
  const arEarlyL19Norm = l2Norm(anchors.arEarlyRec.primaryPassCheckpoint.residualCheckpoints[20]!);
  const arEarlyL39Norm = l2Norm(anchors.arEarlyRec.primaryPassCheckpoint.residualCheckpoints[40]!);
  const arEarlyTop = findTop1FromLastRow(anchors.arEarlyRec, cfg.vocabSize);

  const arBeforePos = anchors.arBoundaryBeforeRec.primaryPassCheckpoint.cachePositions[0] ?? 15;
  const arBeforeL0 = session.reconstructLayerTrace(anchors.arBoundaryBeforeRec.recordId, 0);
  const arBeforeScale = computeLlama4AttnScale(
    15,
    cfg.rope.llama4Beta,
    cfg.rope.originalMaxPositionEmbeddings,
  );
  const arBeforeTraceScale = arBeforeL0.attention.llama4Scales[0] ?? 1.0;

  const arCrossPos = anchors.arBoundaryCrossRec.primaryPassCheckpoint.cachePositions[0] ?? 16;
  const arCrossL0 = session.reconstructLayerTrace(anchors.arBoundaryCrossRec.recordId, 0);
  const arCrossL19 = session.reconstructLayerTrace(anchors.arBoundaryCrossRec.recordId, 19);
  const arCrossL39 = session.reconstructLayerTrace(anchors.arBoundaryCrossRec.recordId, 39);
  const arCrossScale = arCrossL0.attention.llama4Scales[0] ?? computeLlama4AttnScale(
    arCrossPos,
    cfg.rope.llama4Beta,
    cfg.rope.originalMaxPositionEmbeddings,
  );
  const arCrossQRotNorm = l2Norm(arCrossL0.attention.qRot);
  const arCrossQScaledNorm = l2Norm(arCrossL0.attention.qScaled);
  const arCrossNormRatio = arCrossQRotNorm > 1e-8 ? arCrossQScaledNorm / arCrossQRotNorm : 1.0;

  // L20 Q12/KV3 attention stats on arCrossRec
  const q12KvLen = arCrossL19.attention.kvLen;
  const q12Offset = 12 * arCrossL19.attention.seqLen * q12KvLen;
  let q12MaxProb = 0;
  let q12ActiveKeys = 0;
  for (let k = 0; k < q12KvLen; k++) {
    const p = arCrossL19.attention.probs[q12Offset + k] ?? 0;
    if (p > q12MaxProb) q12MaxProb = p;
    if (p > 1e-6) q12ActiveKeys++;
  }

  // SwiGLU live norms (L01 early & L40 deep)
  const swigluGateNorm = l2Norm(arEarlyL0.swiglu.gateSilu);
  const swigluUpNorm = l2Norm(arEarlyL0.swiglu.upVal);
  const swigluGatedMax = maxAbs(arEarlyL0.swiglu.gated);
  const swigluOutNorm = l2Norm(arEarlyL0.swiglu.output);
  const swigluDeepOutNorm = l2Norm(arCrossL39.swiglu.output);
  const arCrossFinalNorm = l2Norm(anchors.arBoundaryCrossRec.primaryPassCheckpoint.finalNorm);

  // RoPE plane (1,5) live angle on arEarlyRec
  const arEarlyPos = anchors.arEarlyRec.primaryPassCheckpoint.positionIds[0] ?? 0;
  const plane1InvFreq = DIFFUSION_DYADIC_YARN_INV_FREQ[1];
  const plane1Angle = arEarlyPos * plane1InvFreq;

  // Speculative & LoRA live stats
  const specLoraL0 = session.reconstructLayerTrace(anchors.specLoraRec1.recordId, 0);
  const specLoraSuffixSlabs = session.reconstructSpecCroppedSuffixKvs(anchors.specLoraRec2.recordId);
  const dlmBidirTop = findTop1FromLastRow(anchors.dlmBidirRec2, cfg.vocabSize);

  const specs: readonly BeatSpec[] = [
    // =========================================================================
    // ACT 1: Shared 40L Backbone & [x_t ‖ x_0] (6 beats: 0..5)
    // =========================================================================
    {
      beatId: 'act1-backbone-macro',
      actId: 'act-1',
      title: 'One Physical 40-Layer Dense Backbone',
      subtitle: 'Ministral 3 homogeneous stack — zero separate encoder/decoder towers (D1, D14)',
      narrative:
        'Nemotron-Labs-Diffusion-14B runs every mode — AR, Block Diffusion, and Linear Self-Speculation — through the exact same 40-layer dense Ministral3Model backbone. Only the attention mask, KV cache policy, and optional draft-only o_proj LoRA rail change between modes.',
      takeaway:
        'One shared 40L weight stack powers both causal autoregressive prediction and bidirectional block denoising.',
      viewMode: 'macro',
      focus: makeFocus({ viewMode: 'macro', selectedLayerIndex: 0 }),
      record: anchors.prefillRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_PROD_CONTRAST,
      liveMetrics: [
        { label: 'Backbone', value: `${cfg.numLayers}L Dense`, tone: 'causal' },
        { label: 'KV Prefix', value: `${anchors.prefillRec.kvSnapshot.seqLen}/${cfg.kvCapacity} tok`, tone: 'causal' },
        { label: 'Actual Fwd', value: `${anchors.prefillRec.counters.actualForwardPasses}`, tone: 'muted' },
        { label: 'Session RAM', value: formatMiB(mem.totalRetainedBytes), tone: 'emerald' },
      ],
    },
    {
      beatId: 'act1-embedding-specials',
      actId: 'act-1',
      title: 'Untied Input Embeddings & Control Tokens',
      subtitle: 'encoder.embed_tokens maps V=36 mini (131,072 prod) with [MASK] & <|im_end|> (D5)',
      narrative:
        'Tokens enter through an untied embedding table with no learned absolute positional table — all position geometry is injected later inside each layer via YaRN RoPE. Special IDs include [MASK] (id 32 mini / <SPECIAL_100> id 100 prod), <|im_end|> EOS (33 / 11), <s> BOS (34 / 1), and </think> (35 / 13).',
      takeaway:
        'Untied input embeddings carry token identity only; [MASK] is a first-class vocabulary row fed straight into L01.',
      viewMode: 'embedding',
      focus: makeFocus({ viewMode: 'embedding', selectedTokenRow: 0 }),
      record: anchors.prefillRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Vocab V', value: `${cfg.vocabSize} (mini)`, tone: 'muted' },
        { label: '[MASK] ID', value: `${cfg.maskTokenId} (100 prod)`, tone: 'gold' },
        { label: 'Prompt Len', value: `${anchors.prefillRec.primaryPassCheckpoint.inputTokenIds.length} tok`, tone: 'causal' },
        { label: 'Embed L2', value: safeNum(prefillEmbNorm, 3), tone: 'emerald' },
      ],
    },
    {
      beatId: 'act1-joint-training-mask',
      actId: 'act-1',
      title: 'BD3-LM Joint Training Mask [x_t ‖ x_0]',
      subtitle: 'M_BD (block-diagonal), M_OBC (offset block-causal), M_BC (clean causal) + D23 coordinates',
      narrative:
        'During Block-Diffusion adaptation, noisy sequence x_t and clean sequence x_0 are concatenated into a 2L dual stream. M_BD lets noisy tokens attend within their own block, M_OBC lets them read strictly prior clean blocks, and M_BC keeps clean x_0 block-causal — exposing D23 where cache_position runs 0..2L-1 while position_ids repeat 0..L-1.',
      takeaway:
        'Three distinct coordinate axes — seq_idx, ropePositionIds, and cachePositions — diverge on [x_t ‖ x_0] training passes (D23).',
      viewMode: 'mask_explorer',
      focus: makeFocus({ viewMode: 'mask_explorer', selectedMaskTab: 'joint_training' }),
      record: anchors.prefillRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Mask Tab', value: 'Joint [x_t‖x_0]', tone: 'bidir' },
        { label: 'Block B', value: `${cfg.defaultBlockLength}`, tone: 'gold' },
        { label: 'Regions', value: 'M_BD · M_OBC · M_BC', tone: 'causal' },
        { label: 'Prefill KV', value: `${anchors.prefillRec.kvSnapshot.seqLen} tok`, tone: 'muted' },
      ],
    },
    {
      beatId: 'act1-layer-symmetry',
      actId: 'act-1',
      title: 'Homogeneous L01–L40 Residual Stream',
      subtitle: 'Pre-Attention & Pre-MLP RMSNorm (ε=1e-5) with 0-ULP lazy layer reconstruction',
      narrative:
        'Every layer from L01 through L40 shares the identical layout: input RMSNorm, 32Q/8KV GQA attention with residual add, post-attention RMSNorm, and dense SwiGLU MLP with residual add. The browser session retains compact residual checkpoints for all 40 layers so any layer (like L20 here) reconstructs on demand at 0-ULP parity.',
      takeaway:
        'Homogeneous layers simplify KV slab storage (20.00 KiB/tok mini, 160.00 KiB/tok 14B BF16) across the full depth.',
      viewMode: 'macro',
      focus: makeFocus({ viewMode: 'macro', selectedLayerIndex: 19 }),
      record: anchors.arEarlyRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Selected', value: 'L20 / L40', tone: 'causal' },
        { label: 'L20 Resid L2', value: safeNum(arEarlyL19Norm, 3), tone: 'emerald' },
        { label: 'KV Seq Len', value: `${anchors.arEarlyRec.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'causal' },
        { label: 'Shared Slabs', value: `${mem.uniqueSlabCount}`, tone: 'muted' },
      ],
    },
    {
      beatId: 'act1-output-head-shifts',
      actId: 'act-1',
      title: 'Untied diffusion_head: +1 Causal vs. +0 Diffusion',
      subtitle: 'Final RMSNorm → untied projection logits with un-suppressed [MASK] softmax (D18)',
      narrative:
        'After L40 and encoder.norm, the untied diffusion_head produces vocabulary logits. Causal AR and Verify read row i as the shifted (+1) prediction for position p+i+1, whereas bidirectional DLM reads row i at the same position (+0). Crucially, upstream code never suppresses [MASK] logits before softmax (D18).',
      takeaway:
        'The same untied head serves shifted next-token prediction (+1) and in-place mask unmasking (+0) without clamping [MASK].',
      viewMode: 'output',
      focus: makeFocus({ viewMode: 'output', selectedLayerIndex: 39 }),
      record: anchors.arEarlyRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Top-1 Token', value: `"${tokenText(arEarlyTop.tokenId)}"`, tone: 'emerald' },
        { label: 'Top-1 Prob', value: `${(arEarlyTop.prob * 100).toFixed(1)}%`, tone: 'causal' },
        { label: 'P([MASK])', value: `${(arEarlyTop.maskProb * 100).toFixed(2)}%`, tone: 'gold' },
        { label: 'Shift Rule', value: '+1 AR / +0 DLM', tone: 'muted' },
      ],
    },
    {
      beatId: 'act1-projection-reversal',
      actId: 'act-1',
      title: 'Epistemic Honesty & Projection-Width Reversal',
      subtitle: 'Mini expands 8→256/64 while canonical 14B contracts 5120→4096/1024',
      narrative:
        'To keep all 40 layers, 32 Q heads, and 8 KV heads genuinely executable in browser Float32, the miniature simulator uses d=8 and d_head=8 — so q_proj expands 8→256 and k/v_proj expand 8→64. On canonical 14B Instruct (13,506,073,600 verified BF16 params), d=5120 contracts into 4096 Q and 1024 K/V channels.',
      takeaway:
        'Every geometry ratio is disclosed verbatim; open the Inspector Architecture tab anytime to audit 0 B LFS 14B tables.',
      viewMode: 'macro',
      focus: makeFocus({ viewMode: 'macro', selectedLayerIndex: 39 }),
      record: anchors.arEarlyRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_PROD_CONTRAST,
      liveMetrics: [
        { label: 'L40 Out L2', value: safeNum(arEarlyL39Norm, 3), tone: 'emerald' },
        { label: 'Mini Q/KV', value: '8→256 / 64', tone: 'gold' },
        { label: '14B Q/KV', value: '5120→4096 / 1024', tone: 'causal' },
        { label: 'Mini Floats', value: '229,064 base', tone: 'muted' },
      ],
    },

    // =========================================================================
    // ACT 2: YaRN RoPE, Llama-4 & SwiGLU (7 beats: 6..12)
    // =========================================================================
    {
      beatId: 'act2-gqa-4to1',
      actId: 'act-2',
      title: '32Q / 8KV Grouped-Query Attention (4:1)',
      subtitle: 'Four query heads share each key/value head group across all 40 layers',
      narrative:
        'Inside L01 GQA, pre-attention RMSNorm feeds W_Q (32 heads), W_K (8 heads), and W_V (8 heads). Every contiguous block of 4 query heads (e.g., Q00–Q03) attends over one shared KV head (KV0), cutting KV cache bandwidth 4× compared to 32-head MHA.',
      takeaway:
        '4:1 GQA reduces per-token KV storage by 75% while preserving 32 independent query specializations per layer.',
      viewMode: 'gqa_rope',
      focus: makeFocus({
        viewMode: 'gqa_rope',
        selectedLayerIndex: 0,
        selectedQHead: 0,
        selectedKvHead: 0,
        selectedRotaryPlane: 0,
        activeStepperStep: 0,
      }),
      record: anchors.arEarlyRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'GQA Ratio', value: `${cfg.numAttentionHeads}Q / ${cfg.numKeyValueHeads}KV (4:1)`, tone: 'causal' },
        { label: 'L01 Q_pre L2', value: safeNum(l2Norm(arEarlyL0.attention.qPre), 3), tone: 'emerald' },
        { label: 'L01 K_pre L2', value: safeNum(l2Norm(arEarlyL0.attention.kPre), 3), tone: 'bidir' },
        { label: 'KV Len', value: `${arEarlyL0.attention.kvLen} tok`, tone: 'muted' },
      ],
    },
    {
      beatId: 'act2-yarn-dyadic-planes',
      actId: 'act-2',
      title: 'Exact Dyadic YaRN RoPE Frequency Ramp',
      subtitle: 'truncate=true yields exact binary fractions [1, 5/16, 1/16, 1/32] across 3 zones (D3)',
      narrative:
        'Ministral 3 replaces vanilla RoPE with Hugging Face YaRN (truncate=True, half-split rotate_half). In the miniature config (θ=16, orig=16, factor=4), the 4 rotary planes land cleanly in three zones: plane (0,4) stays original (1.0), plane (1,5) blends (0.3125), and planes (2,6)/(3,7) interpolate (0.0625, 0.03125).',
      takeaway:
        'High-frequency planes preserve sharp local position resolution while low-frequency planes stretch smoothly for long context.',
      viewMode: 'gqa_rope',
      focus: makeFocus({
        viewMode: 'gqa_rope',
        selectedLayerIndex: 0,
        selectedQHead: 0,
        selectedKvHead: 0,
        selectedRotaryPlane: 1,
        activeStepperStep: 1,
      }),
      record: anchors.arEarlyRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Plane (1,5)', value: `inv_freq=${plane1InvFreq}`, tone: 'gold' },
        { label: 'YaRN Zone', value: yarn.zones[1] ?? 'blended', tone: 'gold' },
        { label: 'Live Pos p', value: `p=${arEarlyPos}`, tone: 'causal' },
        { label: 'Phase φ', value: `${safeNum(plane1Angle, 3)} rad`, tone: 'emerald' },
      ],
    },
    {
      beatId: 'act2-llama4-in-range',
      actId: 'act-2',
      title: 'Llama-4 Query Scale Inside Training Span (p < 16)',
      subtitle: 's_L4(p) = 1 + 0.1·ln(1 + ⌊p / 16⌋) remains exactly 1.0000× through p = 15 (D3)',
      narrative:
        'Post-RoPE, Nemotron-Labs-Diffusion multiplies ONLY queries — never keys — by the Llama-4 attention temperature scale. For every position inside original_max_position_embeddings (p = 0..15 mini; 0..16,383 prod), floor(p / orig) = 0, so both q_scaled and k_rot sit cleanly on the unit circle (r = 1.0).',
      takeaway:
        'Within the native pre-extension span, Llama-4 query scaling is an exact 1.0000× no-op.',
      viewMode: 'gqa_rope',
      focus: makeFocus({
        viewMode: 'gqa_rope',
        selectedLayerIndex: 0,
        selectedQHead: 0,
        selectedKvHead: 0,
        selectedRotaryPlane: 0,
        activeStepperStep: 2,
      }),
      record: anchors.arBoundaryBeforeRec,
      ropeScrubPosition: 15,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Scrub Pos', value: 'p = 15 (< 16)', tone: 'causal' },
        { label: 's_L4(15)', value: `${safeNum(arBeforeScale, 4)}×`, tone: 'emerald' },
        { label: 'Live Record p', value: `p = ${arBeforePos} (${safeNum(arBeforeTraceScale, 4)}×)`, tone: 'muted' },
        { label: 'KV Seq Len', value: `${anchors.arBoundaryBeforeRec.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'causal' },
      ],
    },
    {
      beatId: 'act2-llama4-crossing',
      actId: 'act-2',
      title: 'Crossing p = 16: Live Llama-4 Query Boost (1.0693×)',
      subtitle: 'Queries jump to outer dashed ring r = 1 + 0.1·ln(2) while keys stay at r = 1.0',
      narrative:
        'Stepping the AR branch past p = 16 activates the live Llama-4 query temperature multiplier: floor(p / 16) steps from 0 to 1, scaling Q vectors by 1 + 0.1·ln(2) ≈ 1.0693× (+6.93%) to sharpen attention entropy over extended context while keeping cached keys invariant.',
      takeaway:
        'Inspect the 2D RoPE phasor inset: q_scaled pops outward onto the amber dashed r=1.069 ring while k_rot remains on r=1.0.',
      viewMode: 'gqa_rope',
      focus: makeFocus({
        viewMode: 'gqa_rope',
        selectedLayerIndex: 0,
        selectedQHead: 0,
        selectedKvHead: 0,
        selectedRotaryPlane: 0,
        activeStepperStep: 2,
      }),
      record: anchors.arBoundaryCrossRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Live Pos p', value: `p = ${arCrossPos} (≥ 16)`, tone: 'gold' },
        { label: 'Live s_L4', value: `${safeNum(arCrossScale, 4)}×`, tone: 'lora' },
        { label: '‖q_sc‖/‖q_rot‖', value: `${safeNum(arCrossNormRatio, 4)}×`, tone: 'emerald' },
        { label: 'KV Seq Len', value: `${anchors.arBoundaryCrossRec.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'causal' },
      ],
    },
    {
      beatId: 'act2-gqa-softmax-d24',
      actId: 'act-2',
      title: 'Live RG32F Attention Scores & Softmax (D24 Guard)',
      subtitle: 'L20 Q12/KV3 scores & probabilities bound directly to G-channel visibility',
      narrative:
        'Every attention score and softmax plate across all 8 KV groups binds live Float32 values and exact binary mask visibility in the RG32F value atlas. Because the shared shader falls back to geometric cellCol > cellRow if a texture is omitted (D24), our scene builder rejects any unbound mask or score sheet.',
      takeaway:
        'Dynamic causal, block-bidirectional, and custom masks all render faithfully from real Float32 tensor data.',
      viewMode: 'gqa_rope',
      focus: makeFocus({
        viewMode: 'gqa_rope',
        selectedLayerIndex: 19,
        selectedQHead: 12,
        selectedKvHead: 3,
        selectedRotaryPlane: 2,
        activeStepperStep: 3,
      }),
      record: anchors.arBoundaryCrossRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Focus Head', value: 'L20 Q12 / KV3', tone: 'causal' },
        { label: 'Max Attn Prob', value: `${(q12MaxProb * 100).toFixed(1)}%`, tone: 'emerald' },
        { label: 'Visible Keys', value: `${q12ActiveKeys}/${q12KvLen}`, tone: 'bidir' },
        { label: 'Plane (2,6)', value: `inv_freq=${DIFFUSION_DYADIC_YARN_INV_FREQ[2]}`, tone: 'muted' },
      ],
    },
    {
      beatId: 'act2-swiglu-gated-mlp',
      actId: 'act-2',
      title: 'Dense SwiGLU Feed-Forward Cutaway',
      subtitle: 'SiLU(W_gate·u) ⊙ (W_up·u) → W_down accounts for 74.5% of 14B backbone weights',
      narrative:
        'Following the attention residual add and post_attention_layernorm, each layer evaluates a dense SwiGLU MLP (d → d_ff → d: 8→24→8 mini, 5120→16384→5120 prod). Two parallel projections compute gate and up activations, SiLU modulates the gate before elementwise Hadamard multiplication, and W_down projects back into the residual stream.',
      takeaway:
        'Dense SwiGLU carries ~10.07B of the 13.51B canonical 14B weights with zero routing or MoE sparsity.',
      viewMode: 'swiglu',
      focus: makeFocus({
        viewMode: 'swiglu',
        selectedLayerIndex: 0,
        swigluAccumReveal: 0.6,
      }),
      record: anchors.arEarlyRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'SiLU(Gate) L2', value: safeNum(swigluGateNorm, 3), tone: 'causal' },
        { label: 'Up L2', value: safeNum(swigluUpNorm, 3), tone: 'bidir' },
        { label: 'Gated Max|x|', value: safeNum(swigluGatedMax, 3), tone: 'gold' },
        { label: 'MLP Out L2', value: safeNum(swigluOutNorm, 3), tone: 'emerald' },
      ],
    },
    {
      beatId: 'act2-swiglu-deep-layer',
      actId: 'act-2',
      title: 'Deep Layer L40 SwiGLU into Final RMSNorm',
      subtitle: 'Full 100% accumulation reveal feeding encoder.norm and untied diffusion_head',
      narrative:
        'At layer L40 — after 39 preceding attention and SwiGLU residual updates — the final dense MLP output merges into the residual highway and enters encoder.norm. Because the session lazily reconstructs any non-eager layer from its exact checkpoint, L40 matches eager execution to 0 ULP.',
      takeaway:
        'Inspecting any layer from L01 to L40 reveals live Float32 activations without inflating session RAM.',
      viewMode: 'swiglu',
      focus: makeFocus({
        viewMode: 'swiglu',
        selectedLayerIndex: 39,
        swigluAccumReveal: 1.0,
      }),
      record: anchors.arBoundaryCrossRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Layer', value: 'L40 (Final)', tone: 'causal' },
        { label: 'L40 MLP L2', value: safeNum(swigluDeepOutNorm, 3), tone: 'emerald' },
        { label: 'FinalNorm L2', value: safeNum(arCrossFinalNorm, 3), tone: 'gold' },
        { label: 'KV Seq Len', value: `${anchors.arBoundaryCrossRec.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'muted' },
      ],
    },

    // =========================================================================
    // ACT 3: AR vs. Block Diffusion (causal_context T/F) (6 beats: 13..18)
    // =========================================================================
    {
      beatId: 'act3-ar-causal-mask',
      actId: 'act-3',
      title: 'Autoregressive Decoding & Explicit Causal Masking (D8)',
      subtitle: '1 token per decode pass over strict lower-triangular token_causal KV history',
      narrative:
        'In AR mode, each decode step appends one token to the persistent token_causal KV cache and executes one forward pass (actual = 1 prefill + decode steps). Unlike the latent upstream eager bug (D8) where ar_generate omits use_causal_mask=True, our engine always enforces explicit causal masking.',
      takeaway:
        'AR delivers the exact baseline distribution with 1 decode forward pass per emitted token.',
      viewMode: 'mask_explorer',
      focus: makeFocus({ viewMode: 'mask_explorer', selectedMaskTab: 'ar_causal' }),
      record: anchors.arTipRec,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Mode', value: 'AR (Causal)', tone: 'causal' },
        { label: 'Actual Fwd', value: `${anchors.arTipRec.counters.actualForwardPasses}`, tone: 'muted' },
        { label: 'Decode Fwd', value: `${anchors.arTipRec.counters.decodeForwardPasses}`, tone: 'emerald' },
        { label: 'Ref NFE (D12)', value: `${anchors.arTipRec.counters.referenceMethodNfe}`, tone: 'gold' },
      ],
    },
    {
      beatId: 'act3-dlm-causal-block',
      actId: 'act-3',
      title: 'Block Diffusion (causal_context = true): AR-Seeded Slot 0',
      subtitle: 'Slot 0 seeded from prior causal logits (D7), transient denoise concat (D6), causal commit',
      narrative:
        'Default Block Diffusion (causal_context=True) does NOT start with B raw [MASK] tokens: slot 0 is seeded from the prior causal logits (D7), leaving B-1 = 3 masks. Denoising passes run use_cache=False over transient cat([old_k, K]) without cropping (D6), followed by a single causal commit pass that persists the finished block into token_causal KV.',
      takeaway:
        'Transient denoising plus causal commit lets DLM Causal share the exact same token_causal KV provenance as AR.',
      viewMode: 'macro',
      focus: makeFocus({ viewMode: 'macro', selectedLayerIndex: 0 }),
      record: anchors.dlmCausalRec1,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'KV Provenance', value: anchors.dlmCausalRec1.kvMaskProvenance, tone: 'causal' },
        { label: 'Denoise Passes', value: `${anchors.dlmCausalRec1.dlmMeta?.denoisePassCount ?? 0}`, tone: 'gold' },
        { label: 'Actual Fwd', value: `${anchors.dlmCausalRec1.counters.actualForwardPasses}`, tone: 'muted' },
        { label: 'Ref NFE (D12)', value: `${anchors.dlmCausalRec1.counters.referenceMethodNfe}`, tone: 'emerald' },
      ],
    },
    {
      beatId: 'act3-dlm-causal-commit-mask',
      actId: 'act-3',
      title: 'Causal Commit & Shifted Verify Attention Mask',
      subtitle: 'Prefix remains token_causal across consecutive B=4 DLM Causal blocks',
      narrative:
        'After Sampler A finishes unmasking a block, the commit pass writes the B finalized tokens into the KV cache under a strict causal mask so every subsequent block sees a clean token_causal prefix. Notice D12: upstream generate() excludes prefill from referenceMethodNfe, while our 3-counter readout always reports actual, decode, and reference counts side by side.',
      takeaway:
        'Separating denoising (transient bidirectional) from commit (persistent causal) keeps KV history cleanly composable.',
      viewMode: 'mask_explorer',
      focus: makeFocus({ viewMode: 'mask_explorer', selectedMaskTab: 'causal_commit_verify' }),
      record: anchors.dlmCausalRec2,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Mask Tab', value: 'Commit / Verify', tone: 'causal' },
        { label: 'KV Seq Len', value: `${anchors.dlmCausalRec2.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'causal' },
        { label: 'Gen Tokens', value: `${anchors.dlmCausalRec2.generatedTokenIds.length}`, tone: 'emerald' },
        { label: 'Decode Fwd', value: `${anchors.dlmCausalRec2.counters.decodeForwardPasses}`, tone: 'muted' },
      ],
    },
    {
      beatId: 'act3-dlm-bidir-mask',
      actId: 'act-3',
      title: 'Block Diffusion (causal_context = false): Bidir KV Isolation',
      subtitle: 'Explicit re-prefill switches KV provenance to block_bidirectional; all B slots start [MASK]',
      narrative:
        'Switching to causal_context=False changes the semantics of every stored KV vector from token_causal to block_bidirectional. The session refuses silent cache mixing via KvProvenanceMismatchError until an explicit bidirectional re-prefill runs; each block then begins with all B=4 slots masked and commits bidirectionally.',
      takeaway:
        'Strict KV provenance tagging prevents silent corruption between causal and bidirectional attention caches.',
      viewMode: 'mask_explorer',
      focus: makeFocus({ viewMode: 'mask_explorer', selectedMaskTab: 'block_bidir' }),
      record: anchors.dlmBidirRec1,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'KV Provenance', value: anchors.dlmBidirRec1.kvMaskProvenance, tone: 'bidir' },
        { label: 'Denoise Passes', value: `${anchors.dlmBidirRec1.dlmMeta?.denoisePassCount ?? 0}`, tone: 'gold' },
        { label: 'KV Seq Len', value: `${anchors.dlmBidirRec1.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'bidir' },
        { label: 'Actual Fwd', value: `${anchors.dlmBidirRec1.counters.actualForwardPasses}`, tone: 'muted' },
      ],
    },
    {
      beatId: 'act3-dlm-bidir-output',
      actId: 'act-3',
      title: 'Bidirectional Block Output Logits & D12 Counter Contrast',
      subtitle: 'Same-position (+0) diffusion predictions on block_bidirectional context',
      narrative:
        'On the bidirectional DLM branch, every token in the active block attends to all B positions simultaneously during denoising and commit. Inspecting the output head shows live same-position softmax distributions alongside the exact 3-counter NFE breakdown (including the explicit re-prefill pass).',
      takeaway:
        'Comparing DLM Causal vs. DLM Bidir on separate branches makes the KV provenance and NFE tradeoffs immediately inspectable.',
      viewMode: 'output',
      focus: makeFocus({ viewMode: 'output', selectedLayerIndex: 39 }),
      record: anchors.dlmBidirRec2,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Top-1 Token', value: `"${tokenText(dlmBidirTop.tokenId)}"`, tone: 'bidir' },
        { label: 'Gen Tokens', value: `${anchors.dlmBidirRec2.generatedTokenIds.length}`, tone: 'emerald' },
        { label: 'Actual / Dec', value: `${anchors.dlmBidirRec2.counters.actualForwardPasses} / ${anchors.dlmBidirRec2.counters.decodeForwardPasses}`, tone: 'muted' },
        { label: 'Ref NFE (D12)', value: `${anchors.dlmBidirRec2.counters.referenceMethodNfe}`, tone: 'gold' },
      ],
    },
    {
      beatId: 'act3-vlm-quadratic-ref',
      actId: 'act-3',
      title: 'VLM-8B Quadratic Self-Speculation Reference (D10, D24)',
      subtitle: 'Read-only k(k+1) interleaved mask — absent from all 6 text checkpoints',
      narrative:
        'Paper §3.4 describes Quadratic Self-Speculation, but repository auditing (D10) proved it is absent from all 6 text checkpoints and shipped only inside VLM-8B (modeling_nemotron_labs_diffusion_vlm.py). This 5th Mask Explorer tab preserves the exact k(k+1) [draft_kv ‖ clean_kv] reference plate honestly badged as read-only, noting D24 (VLM greedy-only bug).',
      takeaway:
        'The visualizer distinguishes live released text execution paths from VLM-only reference algorithms.',
      viewMode: 'mask_explorer',
      focus: makeFocus({ viewMode: 'mask_explorer', selectedMaskTab: 'vlm_quadratic_ss' }),
      record: anchors.dlmCausalRec2,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_VLM_REF,
      liveMetrics: [
        { label: 'Mask Tab', value: 'VLM Quadratic SS', tone: 'gold' },
        { label: 'Availability', value: 'VLM-8B Only (D10)', tone: 'warn' },
        { label: 'Sampling', value: 'Greedy Bug (D24)', tone: 'warn' },
        { label: 'Active KV', value: `${anchors.dlmCausalRec2.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'muted' },
      ],
    },

    // =========================================================================
    // ACT 4: Linear Self-Speculation, LoRA & KV Rollback (7 beats: 19..25)
    // =========================================================================
    {
      beatId: 'act4-spec-base-draft-verify',
      actId: 'act-4',
      title: 'Linear Self-Speculation (Base Backbone, LoRA OFF)',
      subtitle: 'Sampler B drafts B-1 proposals (use_cache=False), then 1 causal verify checks exact match',
      narrative:
        'Linear Self-Speculation pairs a fast bidirectional draft pass (slot 0 = seed token, slots 1..B-1 = [MASK] filled by Sampler B with use_cache=False) with one causal verify pass over the full B-token block. Every cycle emits matchedDraftCount + 1 tokens (accepted prefix plus 1 verified correction/bonus token), checked by exact equality ar_tokens[i] == block[i+1] (D11).',
      takeaway:
        'Even without LoRA, the single backbone self-drafts and self-verifies in one KV space.',
      viewMode: 'spec_verify',
      focus: makeFocus({ viewMode: 'spec_verify', activeStepperStep: 0 }),
      record: anchors.specBaseRec1,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'o_proj LoRA', value: 'OFF (Base)', tone: 'muted' },
        { label: 'Draft Passes', value: `${anchors.specBaseRec1.specMeta?.draftPassCount ?? 0}`, tone: 'causal' },
        { label: 'Matched Draft', value: `${anchors.specBaseRec1.specMeta?.matchedDraftCount ?? 0}/${cfg.defaultBlockLength - 1}`, tone: 'emerald' },
        { label: 'Accepted', value: `${anchors.specBaseRec1.specMeta?.acceptedCount ?? 0} tok`, tone: 'gold' },
      ],
    },
    {
      beatId: 'act4-spec-lora-macro-rail',
      actId: 'act-4',
      title: 'Detachable Draft-Only o_proj LoRA Rail',
      subtitle: '80 F32 tensors (+0.35% on 14B) ON during draft passes, 0-ULP OFF during verify (D17)',
      narrative:
        'Each Instruct text checkpoint ships a lightweight linear_spec_lora adapter targeting ONLY self_attn.o_proj across all 40 layers (47.19M F32 params = +0.35% of 14B; r=4, α=16, scale=4.0 mini). The rail activates exclusively on bidirectional draft passes to align draft proposals with the causal distribution, then detaches cleanly so causal verification runs on the unmodified base backbone.',
      takeaway:
        'Draft-only LoRA changes proposal quality without ever altering the verified target distribution.',
      viewMode: 'macro',
      focus: makeFocus({ viewMode: 'macro', selectedLayerIndex: 0 }),
      record: anchors.specLoraRec1,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_PROD_CONTRAST,
      liveMetrics: [
        { label: 'LoRA Rail', value: 'ON Draft / OFF Verify', tone: 'lora' },
        { label: 'Matched Draft', value: `${anchors.specLoraRec1.specMeta?.matchedDraftCount ?? 0}/${cfg.defaultBlockLength - 1}`, tone: 'emerald' },
        { label: 'Accepted', value: `${anchors.specLoraRec1.specMeta?.acceptedCount ?? 0} tok`, tone: 'gold' },
        { label: 'KV Seq Len', value: `${anchors.specLoraRec1.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'causal' },
      ],
    },
    {
      beatId: 'act4-spec-lora-gqa-delta',
      actId: 'act-4',
      title: 'Inside L01 o_proj: Base Verify vs. +4.0·B(A·a) Draft Delta',
      subtitle: 'Stateless per-call loraEnabled prevents D17 adapter-leak across modes',
      narrative:
        'Zooming into L01 GQA stage 5 shows the o_proj exit gate: during causal verify (the checkpointed pass here), loraEnabled is strictly false so attn_out equals o_proj(attn_concat) bit-for-bit. During the preceding draft pass, +scale·B(A·attn_concat) adds cleanly in parallel. Stateless flags avoid D17 where upstream ar_generate/generate never disable attached PEFT adapters.',
      takeaway:
        'Passing explicit maskMode and loraEnabled per forward call guarantees zero cross-mode adapter contamination (D17).',
      viewMode: 'gqa_rope',
      focus: makeFocus({
        viewMode: 'gqa_rope',
        selectedLayerIndex: 0,
        selectedQHead: 0,
        selectedKvHead: 0,
        activeStepperStep: 4,
      }),
      record: anchors.specLoraRec1,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Verify LoRA', value: specLoraL0.attention.loraEnabled ? 'ON' : 'OFF (0-ULP)', tone: 'emerald' },
        { label: 'Draft LoRA', value: anchors.specLoraRec1.specMeta?.loraEnabled ? 'ON (scale=4.0)' : 'OFF', tone: 'lora' },
        { label: 'oProjBase L2', value: safeNum(l2Norm(specLoraL0.attention.oProjBase), 3), tone: 'causal' },
        { label: 'Attn Out L2', value: safeNum(l2Norm(specLoraL0.attention.output), 3), tone: 'muted' },
      ],
    },
    {
      beatId: 'act4-spec-shifted-alignment',
      actId: 'act-4',
      title: 'Shifted Draft-vs-Verify Alignment Connectors',
      subtitle: 'Verify row i (at p+i) predicts p+i+1 and tests Draft slot i+1',
      narrative:
        'Because causal verify computes shifted next-token predictions, Verify row i at position p+i predicts token p+i+1 — diagonal connectors therefore link Verify row i directly to Draft slot i+1. Matching slots light emerald; the first mismatch places an oxide-red scissor and emits the gold verify correction/bonus token ar_tokens[matched].',
      takeaway:
        'Shifted alignment guarantees at least 1 verified token per cycle even when 0 drafted tokens match.',
      viewMode: 'spec_verify',
      focus: makeFocus({ viewMode: 'spec_verify', activeStepperStep: 1 }),
      record: anchors.specLoraRec1,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Draft Block', value: `${anchors.specLoraRec1.specMeta?.draftedBlockTokens.length ?? 0} slots`, tone: 'lora' },
        { label: 'Matched', value: `${anchors.specLoraRec1.specMeta?.matchedDraftCount ?? 0}`, tone: 'emerald' },
        { label: 'Accepted', value: `${anchors.specLoraRec1.specMeta?.acceptedCount ?? 0}`, tone: 'gold' },
        { label: 'Emitted', value: `${anchors.specLoraRec1.stepEmittedTokenIds.length} tok`, tone: 'causal' },
      ],
    },
    {
      beatId: 'act4-spec-kv-rollback',
      actId: 'act-4',
      title: 'KV Cache Rollback (_crop_dynamic_cache) & D19 Provenance',
      subtitle: 'Rejected suffix KVs are cropped immediately; browser records both clean & upstream lengths',
      narrative:
        'During causal verify, all B block slots append temporary KVs up to preCropKvLen. Immediately afterward, _crop_dynamic_cache rolls the persistent KV cache back to cache_len + accepted (keeping only verified prefix KVs). Under Clarification #1 (D19), if EOS appears inside the accepted slice, the browser crops cleanly at post-EOS while recording upstreamPreEosCropLen in provenance.',
      takeaway:
        'Unaccepted speculative KV slabs never pollute future attention; lazy reconstruction still lets you inspect their real norms.',
      viewMode: 'spec_verify',
      focus: makeFocus({ viewMode: 'spec_verify', activeStepperStep: 2 }),
      record: anchors.specLoraRec2,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Pre-Crop KV', value: `${anchors.specLoraRec2.cropProvenance?.preCropKvLen ?? anchors.specLoraRec2.kvSnapshot.seqLen}`, tone: 'muted' },
        { label: 'Post-Crop KV', value: `${anchors.specLoraRec2.cropProvenance?.browserPostEosCropLen ?? anchors.specLoraRec2.kvSnapshot.seqLen}`, tone: 'emerald' },
        { label: 'Cropped Suffix', value: `${anchors.specLoraRec2.cropProvenance?.croppedSuffixLen ?? 0} slots`, tone: 'warn' },
        { label: 'Lazy Slabs', value: `${specLoraSuffixSlabs.length}`, tone: 'gold' },
      ],
    },
    {
      beatId: 'act4-d18-d12-honesty',
      actId: 'act-4',
      title: 'Transparent NFE Counters & Browser Safety Guards (D12, D16, D18)',
      subtitle: 'Zero fabricated speedups — live 3-counter NFE, D18 [MASK] loop guard, hard kvCapacity=32',
      narrative:
        'Rather than quoting synthetic speedup claims on miniature weights, the Inspector displays all three exact NFE counters (actual, decode, and upstream referenceMethodNfe per D12) and live tokens/forward ratios (or "—" when zero). Meanwhile, D18 guards against non-progress [MASK] draft loops without clamping logits, and D16 halts cleanly at kvCapacity=32 without silently shrinking B to 1.',
      takeaway:
        'Every counter, guard flag, and capacity boundary is computed directly from the live session record.',
      viewMode: 'spec_verify',
      focus: makeFocus({ viewMode: 'spec_verify', activeStepperStep: 3 }),
      record: anchors.specLoraRec2,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_MINIATURE_SIM,
      liveMetrics: [
        { label: 'Actual Fwd', value: `${anchors.specLoraRec2.counters.actualForwardPasses}`, tone: 'causal' },
        { label: 'Decode Fwd', value: `${anchors.specLoraRec2.counters.decodeForwardPasses}`, tone: 'emerald' },
        { label: 'Ref NFE (D12)', value: `${anchors.specLoraRec2.counters.referenceMethodNfe}`, tone: 'gold' },
        { label: 'KV Capacity', value: `${anchors.specLoraRec2.kvSnapshot.seqLen}/${cfg.kvCapacity}`, tone: 'muted' },
      ],
    },
    {
      beatId: 'act4-session-memory-summary',
      actId: 'act-4',
      title: 'Bounded Multi-Branch Session & Ready to Explore',
      subtitle: 'Structurally shared 20.00 KiB/tok KV slabs stay far under 12.00 MiB — exit to resume Free-Explore',
      narrative:
        'Across all 5 isolated tour branches (AR, DLM Causal, DLM Bidir, Linear Spec, and Spec + LoRA), immutable per-position KV slabs share underlying ArrayBuffers so total retained session memory remains tiny relative to the 12.00 MiB cap. Open the 3-Tab Inspector for Architecture tables and the full D1–D24 Discrepancy Ledger, or Exit Tour to restore your exact Free-Explore state.',
      takeaway:
        'Exiting the tour restores your Free-Explore branch, record, camera pose, focus, and Inspector state verbatim.',
      viewMode: 'macro',
      focus: makeFocus({ viewMode: 'macro', selectedLayerIndex: 0 }),
      record: anchors.specLoraRec2,
      ropeScrubPosition: null,
      epistemicNote: EPISTEMIC_PROD_CONTRAST,
      liveMetrics: [
        { label: 'Tour Branches', value: `${state.branches.length}/${state.bounds.maxBranches}`, tone: 'causal' },
        { label: 'Tour Records', value: `${state.records.size}/${state.bounds.maxRecords}`, tone: 'muted' },
        { label: 'Retained RAM', value: `${formatMiB(mem.totalRetainedBytes)} / 12.00 MiB`, tone: 'emerald' },
        { label: 'Unique Slabs', value: `${mem.uniqueSlabCount}`, tone: 'gold' },
      ],
    },
  ];

  const actById = new Map<DiffusionTourActId, DiffusionTourActDef>();
  for (const def of DIFFUSION_TOUR_ACT_DEFS) {
    actById.set(def.id, def);
  }

  const totalBeats = specs.length;
  return Object.freeze(
    specs.map((spec, beatIndex): DiffusionTourBeatSnapshot => {
      const act = actById.get(spec.actId)!;
      return Object.freeze({
        beatIndex,
        totalBeats,
        beatId: spec.beatId,
        actId: spec.actId,
        actIndex: act.actIndex,
        actTitle: act.title,
        title: spec.title,
        subtitle: spec.subtitle,
        narrative: spec.narrative,
        takeaway: spec.takeaway,
        viewMode: spec.viewMode,
        focus: spec.focus,
        recordId: spec.record.recordId,
        branchId: spec.record.branchId,
        mode: spec.record.configAtStep.mode,
        causalContext: spec.record.configAtStep.causalContext,
        loraActiveOnDraft:
          spec.record.configAtStep.mode === 'linear_spec_lora' &&
          spec.record.kind === 'spec_cycle',
        liveMetrics: Object.freeze(spec.liveMetrics.map((m) => Object.freeze({ ...m }))),
        provenanceSummary: formatProvenanceSummary(spec.record),
        epistemicNote: spec.epistemicNote,
        ropeScrubPosition: spec.ropeScrubPosition,
      });
    }),
  );
}

// ============================================================================
// PUBLIC GUIDED TOUR CONTROLLER FACTORY
// ============================================================================

/**
 * Create a pure headless, strictly isolated 4-Act (26-beat) guided tour controller
 * backed by its own deterministic `DiffusionSession`.
 */
export function createDiffusionGuidedTour(
  model?: DiffusionModelParameters,
): DiffusionGuidedTourController {
  const resolvedModel = model ?? createDefaultDiffusionModel();
  const { session: tourSession, anchors } = buildIsolatedTourSession(resolvedModel);
  const beats = buildTourBeats(tourSession, anchors);

  // Precompute act start indices & counts
  const actMeta: {
    readonly def: DiffusionTourActDef;
    readonly startBeatIndex: number;
    readonly beatCount: number;
  }[] = DIFFUSION_TOUR_ACT_DEFS.map((def) => {
    const startBeatIndex = beats.findIndex((b) => b.actId === def.id);
    const beatCount = beats.filter((b) => b.actId === def.id).length;
    return {
      def,
      startBeatIndex: startBeatIndex >= 0 ? startBeatIndex : 0,
      beatCount,
    };
  });

  let status: DiffusionTourStatus = 'idle';
  let currentBeatIndex = 0;
  let savedRestoreState: DiffusionFreeExploreRestoreState | null = null;
  let cachedTourCtx: DiffusionResolvedSessionContext | null = null;

  function seekToBeat(index: number): DiffusionTourBeatSnapshot {
    const clamped = Math.max(0, Math.min(beats.length - 1, Math.trunc(index)));
    currentBeatIndex = clamped;
    const beat = beats[clamped]!;
    tourSession.seekRecord(beat.recordId);
    cachedTourCtx = resolveDiffusionSessionContext(tourSession, beat.recordId);
    return beat;
  }

  return {
    enter(freeExploreSnapshot: DiffusionFreeExploreRestoreState): DiffusionTourBeatSnapshot {
      savedRestoreState = cloneRestoreState(freeExploreSnapshot);
      status = 'active';
      return seekToBeat(0);
    },

    exit(): DiffusionFreeExploreRestoreState | null {
      if (status !== 'active') return null;
      status = 'exited';
      const restored = savedRestoreState ? cloneRestoreState(savedRestoreState) : null;
      savedRestoreState = null;
      return restored;
    },

    next(): DiffusionTourBeatSnapshot | null {
      if (status !== 'active') return null;
      if (currentBeatIndex >= beats.length - 1) return null;
      return seekToBeat(currentBeatIndex + 1);
    },

    prev(): DiffusionTourBeatSnapshot | null {
      if (status !== 'active') return null;
      if (currentBeatIndex <= 0) return null;
      return seekToBeat(currentBeatIndex - 1);
    },

    goToBeat(index: number): DiffusionTourBeatSnapshot | null {
      if (status !== 'active') return null;
      if (!Number.isInteger(index) || index < 0 || index >= beats.length) return null;
      return seekToBeat(index);
    },

    goToAct(actId: DiffusionTourActId): DiffusionTourBeatSnapshot | null {
      if (status !== 'active') return null;
      const meta = actMeta.find((m) => m.def.id === actId);
      if (!meta || meta.beatCount === 0) return null;
      return seekToBeat(meta.startBeatIndex);
    },

    getStatus(): DiffusionTourStatus {
      return status;
    },

    isActive(): boolean {
      return status === 'active';
    },

    getBeatCount(): number {
      return beats.length;
    },

    getCurrentBeatIndex(): number {
      return status === 'active' ? currentBeatIndex : -1;
    },

    getCurrentBeat(): DiffusionTourBeatSnapshot | null {
      return status === 'active' ? beats[currentBeatIndex] ?? null : null;
    },

    getAllBeats(): readonly DiffusionTourBeatSnapshot[] {
      return beats;
    },

    getActs(): readonly DiffusionTourActSummary[] {
      const currentActId = status === 'active' ? beats[currentBeatIndex]?.actId ?? null : null;
      return Object.freeze(
        actMeta.map((m) =>
          Object.freeze({
            actIndex: m.def.actIndex,
            id: m.def.id,
            title: m.def.title,
            shortTitle: m.def.shortTitle,
            startBeatIndex: m.startBeatIndex,
            beatCount: m.beatCount,
            isCurrent: currentActId === m.def.id,
          }),
        ),
      );
    },

    getTourSession(): DiffusionSession {
      return tourSession;
    },

    getTourContext(): DiffusionResolvedSessionContext {
      if (cachedTourCtx === null) {
        const beat = beats[currentBeatIndex] ?? beats[0]!;
        tourSession.seekRecord(beat.recordId);
        cachedTourCtx = resolveDiffusionSessionContext(tourSession, beat.recordId);
      }
      return cachedTourCtx;
    },

    getRestoreState(): DiffusionFreeExploreRestoreState | null {
      return savedRestoreState ? cloneRestoreState(savedRestoreState) : null;
    },
  };
}
