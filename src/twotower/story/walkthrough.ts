/**
 * Nemotron-Labs-TwoTower Headless 4-Act, 28-Beat Guided Educational Walkthrough Engine
 * (`src/twotower/story/walkthrough.ts`)
 *
 * Pure headless TypeScript — ZERO DOM, ZERO WebGL2, ZERO non-deterministic RNG.
 *
 * Runs its OWN dedicated, deterministic, strictly isolated `TwoTowerSession` over one coherent
 * 8-token miniature prompt (`"How does Nemotron use only six attention layers"`, `P = 8`) with
 * `B = 4, K = 4, γ = 0.22, T = 0` on the main continuous diffusion branch (`branch-0`) plus two
 * isolated Act I side branches forked from `branch-0@0` (`branch-1` for Context AR, `branch-2`
 * for Mock-AR) so the main diffusion trajectory stays 100% continuous from Prefill through
 * Block 0 completion, Context commitment (`P = 8 → 12`), and Block 1 denoising.
 *
 * Every beat navigates the tour's own `TwoTowerSession` to its exact pre-computed
 * `(branchId, recordIndex)` and computes ALL live numerical callouts directly from that record's
 * actual tensors, `SamplerStepPresentation`, `MambaPositionDecomposition`, FP64 MoE 6th-vs-7th
 * cutoff margin, and record provenance — ZERO fabricated model outputs and strictly finite values.
 */

import { formatExpertId } from '../../math/moeSim';
import { formatBytesAdaptive, formatKiB, formatMiB } from '../../model/memory';
import { validateContextSnapshot } from '../contracts';
import { modulationOrderForBayKind } from '../modulation';
import { createDefaultTwoTowerModel, type TwoTowerModelParameters } from '../parameters';
import {
  assertPresentationSafeNoInfinity,
  TWOTOWER_CONFIDENCE_PRESETS,
  type SamplerStepPresentation,
} from '../sampler';
import { extractMambaFullBlockPositionState } from '../scene/mambaDecomposition';
import {
  getTwoTowerBayKind,
  type TwoTowerLinkerChannel,
  type TwoTowerPairedFocus,
  type TwoTowerSpatialViewMode,
} from '../scene/provenance';
import { computeMoeSelectionMargin } from '../scene/twoTowerMoeCutaway';
import {
  commitTwoTowerActiveBlockToContext,
  commitTwoTowerRefinementConfidence,
  createTwoTowerSession,
  DEFAULT_TWOTOWER_PROMPT_TOKENS,
  forkTwoTowerSessionBranch,
  getActiveBranch,
  getActiveRecord,
  getSessionCapabilities,
  initTwoTowerDiffusionBlock,
  measureTwoTowerSessionRetainedBytes,
  navigateTwoTowerSession,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  switchTwoTowerSessionBranch,
  updateTwoTowerSessionConfig,
  type TwoTowerSession,
  type TwoTowerSessionRecord,
} from '../session';
import {
  TWOTOWER_MINI_CONFIG,
  TWOTOWER_SPEC,
  twoTowerMemoryFor,
  type TwoTowerExecutionMode,
} from '../spec';

// ============================================================================
// CONSTANTS & EPISTEMIC TAGS
// ============================================================================

export const TWOTOWER_TOUR_SESSION_ID = 'twotower-guided-tour-session' as const;

export const TWOTOWER_TOUR_PROMPT_TOKENS: readonly number[] = DEFAULT_TWOTOWER_PROMPT_TOKENS;

export const TWOTOWER_TOUR_CONFIG = Object.freeze({
  promptLen: 8 as const,
  blockSize: 4 as const,
  stepsPerBlock: 4 as const,
  confidenceThreshold: 0.22 as const,
  temperature: 0.0 as const,
  kvCapacity: TWOTOWER_MINI_CONFIG.kvCapacity,
});

export const EPISTEMIC_CANONICAL_AND_MINIATURE =
  'Canonical released TwoTower architecture (HF rev 35b64985 · 63.19B BF16) + live deterministic miniature simulator (d=8, V=34 synthetic seeded weights).' as const;

export const EPISTEMIC_MINIATURE_SIMULATION =
  'Live synthetic miniature simulator (d=8, V=34, Float32/Float64 deterministic seeded weights — NOT 63.19B checkpoint weights or real Nemotron tokenizer logits).' as const;

export const EPISTEMIC_MAMBA_ALGEBRAIC =
  'Canonical Mamba-2 recurrence mechanics on miniature d=8 tensors — dA, cumulative decay, and L2 state norms measure algebraic state magnitudes, NOT semantic information retention.' as const;

export const EPISTEMIC_SAMPLER_CALIBRATION =
  'Canonical MDLM / SUBS sampler mechanics on miniature logits (γ=0.22 educational default vs 0.80 paper / 0.90 code) — unscaled softmax confidence is NOT calibrated semantic correctness.' as const;

export const EPISTEMIC_PRODUCTION_VS_MINIATURE =
  'Canonical production precision & scaling spec (BF16 weights, FP32 Mamba SSM & MoE bias, FP64 MoE router; 128K card / 256K config) contrasted against synthetic miniature browser arithmetic.' as const;

// ============================================================================
// PUBLIC TYPES
// ============================================================================

export type TwoTowerTourActId = 'act-1' | 'act-2' | 'act-3' | 'act-4';

export type TwoTowerTourStatus = 'idle' | 'playing' | 'paused' | 'exited';

export type TwoTowerTourMetricTone =
  | 'context'
  | 'denoiser'
  | 'shared'
  | 'warn'
  | 'muted'
  | 'gold';

export interface TwoTowerTourLiveMetric {
  readonly label: string;
  readonly value: string;
  readonly tone: TwoTowerTourMetricTone;
}

export interface TwoTowerTourBeatProvenance {
  readonly branchId: string;
  readonly recordIndex: number;
  readonly recordId: string;
  readonly blockId: number | null;
  readonly refinementIndex: number | null;
  readonly absoluteTokenRange: readonly [number, number];
  readonly sourceCacheDigest: string;
  readonly mode: TwoTowerExecutionMode;
  /** Exact Denoiser executed timestep `t = nMaskedBefore / B` when applicable, else `null`. */
  readonly executedTModel: number | null;
  /** Current unresolved `[MASK]` slot count on the active block at this record (`0` when no active block). */
  readonly currentUnresolvedCount: number;
}

export interface TwoTowerTourActMetadata {
  readonly actIndex: number;
  readonly id: TwoTowerTourActId;
  readonly title: string;
  readonly shortTitle: string;
  readonly startBeatIndex: number;
  readonly beatCount: number;
}

export interface TwoTowerTourActSummary extends TwoTowerTourActMetadata {
  readonly isCurrent: boolean;
}

export interface TwoTowerTourBeatSnapshot {
  readonly beatIndex: number;
  readonly beatId: string;
  readonly actIndex: number;
  readonly actId: TwoTowerTourActId;
  readonly actTitle: string;
  readonly actBeatIndex: number;
  readonly actBeatCount: number;
  readonly title: string;
  readonly subtitle: string;
  readonly badge: string;
  readonly paragraphs: readonly string[];
  readonly takeaways: readonly string[];
  readonly interactivePrompt: string;
  readonly viewMode: TwoTowerSpatialViewMode;
  readonly bayId: number;
  readonly towerFocus: TwoTowerPairedFocus;
  readonly linkerChannel: TwoTowerLinkerChannel;
  readonly blockPos: number;
  readonly opStep: number | null;
  readonly showSamplerPanel: boolean;
  readonly showArchPanel: boolean;
  readonly highlightPartId: string | null;
  readonly liveMetrics: readonly TwoTowerTourLiveMetric[];
  readonly provenance: TwoTowerTourBeatProvenance;
  readonly epistemicTag: string;
}

export interface TwoTowerTourSnapshot extends TwoTowerTourBeatSnapshot {
  readonly status: TwoTowerTourStatus;
  readonly isActive: boolean;
  readonly isPlaying: boolean;
  readonly totalBeats: number;
  readonly totalActs: number;
  readonly canPrev: boolean;
  readonly canNext: boolean;
  readonly progressFraction: number;
  readonly acts: readonly TwoTowerTourActSummary[];
  readonly currentBeat: TwoTowerTourBeatSnapshot;
}

export interface TwoTowerTourController {
  start(): TwoTowerTourSnapshot;
  pause(): TwoTowerTourSnapshot;
  resume(): TwoTowerTourSnapshot;
  togglePlay(): TwoTowerTourSnapshot;
  next(): TwoTowerTourSnapshot;
  prev(): TwoTowerTourSnapshot;
  goToBeat(index: number): TwoTowerTourSnapshot;
  goToAct(actIndex: number): TwoTowerTourSnapshot;
  restart(): TwoTowerTourSnapshot;
  exit(): TwoTowerTourSnapshot;
  getSnapshot(): TwoTowerTourSnapshot;
  /** Read-only handle to the tour's own isolated `TwoTowerSession` positioned at the active beat. */
  getSession(): TwoTowerSession;
  getModel(): TwoTowerModelParameters;
}

export interface TwoTowerTourIsolationReport {
  readonly isolated: boolean;
  readonly sameSessionReference: boolean;
  readonly sameSessionId: boolean;
  readonly sameActiveBranchId: boolean;
  readonly sameCursorIndex: boolean;
  readonly sameMutationSeq: boolean;
  readonly sameBranchCount: boolean;
  readonly sameActiveRecordId: boolean;
  readonly sameContextSnapshotDigest: boolean;
  readonly sameContextCacheDigest: boolean;
  readonly samePendingConfig: boolean;
  readonly distinctFromTourSessionId: boolean;
  readonly distinctFromTourSessionReference: boolean;
  readonly snapshotIntegrityValid: boolean;
  readonly sharedArrayBufferCount: number;
  readonly reasons: readonly string[];
}

// ============================================================================
// ACT DEFINITIONS (4 ACTS, 28 BEATS = 6 + 10 + 7 + 5)
// ============================================================================

export const TWOTOWER_TOUR_ACTS: readonly TwoTowerTourActMetadata[] = Object.freeze([
  Object.freeze({
    actIndex: 0,
    id: 'act-1' as const,
    title: 'Act I · Dual-Tower Architecture, Prefill & Three Execution Modes',
    shortTitle: 'Act I · Architecture & Modes',
    startBeatIndex: 0,
    beatCount: 6,
  }),
  Object.freeze({
    actIndex: 1,
    id: 'act-2' as const,
    title: 'Act II · Conditioning, Paired Layers, Recurrent Mechanics & Routing',
    shortTitle: 'Act II · Layers & Mechanics',
    startBeatIndex: 6,
    beatCount: 10,
  }),
  Object.freeze({
    actIndex: 2,
    id: 'act-3' as const,
    title: 'Act III · Denoiser Output, MDLM Sampler & Iterative Refinement',
    shortTitle: 'Act III · Output & Sampler',
    startBeatIndex: 16,
    beatCount: 7,
  }),
  Object.freeze({
    actIndex: 3,
    id: 'act-4' as const,
    title: 'Act IV · Context Commitment, Next Block & Production Architecture',
    shortTitle: 'Act IV · Commit & Production',
    startBeatIndex: 23,
    beatCount: 5,
  }),
]);

export const TWOTOWER_TOUR_ACT_COUNT = 4 as const;
export const TWOTOWER_TOUR_BEAT_COUNT = 28 as const;

// ============================================================================
// INTERNAL FORMATTING & MATH HELPERS (PURE, FINITE-SAFE)
// ============================================================================

function fmtShortDigest(digest: string): string {
  return digest.length > 8 ? digest.slice(0, 8) : digest;
}

function fmtPct(prob: number, digits = 1): string {
  const safe = Number.isFinite(prob) ? prob : 0;
  return `${(safe * 100).toFixed(digits)}%`;
}

function fmtFixed(val: number, digits = 4): string {
  const safe = Number.isFinite(val) ? val : 0;
  return safe.toFixed(digits);
}

function l2Norm(arr: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i] * arr[i];
  }
  return Math.sqrt(sum);
}

function columnL2Norm(matrix: Float32Array, d: number, T: number, col: number): number {
  const safeCol = Math.max(0, Math.min(T - 1, col));
  let sum = 0;
  for (let row = 0; row < d; row++) {
    const v = matrix[row * T + safeCol];
    sum += v * v;
  }
  return Math.sqrt(sum);
}

function metric(
  label: string,
  value: string,
  tone: TwoTowerTourMetricTone,
): TwoTowerTourLiveMetric {
  return Object.freeze({ label, value, tone });
}

function formatGiB(bytes: number): string {
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

function resolveRecordProvenance(record: TwoTowerSessionRecord): TwoTowerTourBeatProvenance {
  let executedTModel: number | null = null;
  if (record.kind === 'diffusion_refinement') {
    executedTModel = record.tModel;
  } else if (record.kind === 'diffusion_confidence_commit') {
    executedTModel = record.unmaskStep.tModel;
  }

  const currentUnresolvedCount = record.activeBlock !== null ? record.activeBlock.nMasked : 0;

  return Object.freeze({
    branchId: record.identity.branchId,
    recordIndex: record.identity.recordIndex,
    recordId: record.identity.recordId,
    blockId: record.identity.blockId,
    refinementIndex: record.identity.refinementIndex,
    absoluteTokenRange: record.identity.absoluteTokenRange,
    sourceCacheDigest: record.identity.sourceCacheDigest,
    mode: record.configAtStep.mode,
    executedTModel,
    currentUnresolvedCount,
  });
}

// ============================================================================
// PRE-COMPUTED DETERMINISTIC TOUR TRAJECTORY BUILDER
// ============================================================================

interface TourTrajectoryLandmarks {
  readonly mainBranchId: string;
  readonly arBranchId: string;
  readonly mockArBranchId: string;
  readonly prefillIdx: number;
  readonly arStepIdx: number;
  readonly mockArStepIdx: number;
  readonly block0InitIdx: number;
  readonly block0Ref0Idx: number;
  readonly block0Conf0Idx: number;
  readonly block0Ref1Idx: number;
  readonly block0Conf1Idx: number;
  readonly block0CompletedUncommittedIdx: number;
  readonly block0CommitIdx: number;
  readonly block1InitIdx: number;
  readonly block1Ref0Idx: number;
  readonly block1RecapIdx: number;
  readonly block0Nfe: number;
  readonly block0EarlyCompleted: boolean;
}

interface TourTrajectoryBundle {
  readonly baseSession: TwoTowerSession;
  readonly landmarks: TourTrajectoryLandmarks;
}

function buildDeterministicTourTrajectory(model: TwoTowerModelParameters): TourTrajectoryBundle {
  // 1. Create dedicated isolated tour session on coherent 8-token prompt (P = 8)
  let session = createTwoTowerSession(model, {
    sessionId: TWOTOWER_TOUR_SESSION_ID,
    promptTokenIds: TWOTOWER_TOUR_PROMPT_TOKENS,
    kvCapacity: TWOTOWER_TOUR_CONFIG.kvCapacity,
    mode: 'mask_diffusion',
    blockSize: TWOTOWER_TOUR_CONFIG.blockSize,
    stepsPerBlock: TWOTOWER_TOUR_CONFIG.stepsPerBlock,
    confidenceThreshold: TWOTOWER_TOUR_CONFIG.confidenceThreshold,
    temperature: TWOTOWER_TOUR_CONFIG.temperature,
    initialBranchLabel: 'Tour Main Diffusion',
    retainEagerTracesOnHistory: false,
  });

  const mainBranchId = session.activeBranchId; // 'branch-0'
  const prefillIdx = 0;

  // 2. Fork isolated side branch for Act I AR mode beat (P = 8 -> 9)
  session = forkTwoTowerSessionBranch(session, {
    fromCursorIndex: prefillIdx,
    label: 'Tour AR Mode (Isolated)',
    configPatch: { mode: 'ar' },
    switchToNewBranch: true,
  });
  const arBranchId = session.activeBranchId;
  session = stepTwoTowerSessionAr(session);
  const arStepIdx = getActiveBranch(session).cursorIndex;

  // 3. Fork isolated side branch for Act I Mock-AR mode beat (P = 8 -> 9, KV[:-1] + mamba_s2, t=null)
  session = switchTwoTowerSessionBranch(session, mainBranchId);
  session = forkTwoTowerSessionBranch(session, {
    fromCursorIndex: prefillIdx,
    label: 'Tour Mock-AR Mode (Isolated)',
    configPatch: { mode: 'mock_ar' },
    switchToNewBranch: true,
  });
  const mockArBranchId = session.activeBranchId;
  session = stepTwoTowerSessionMockAr(session);
  const mockArStepIdx = getActiveBranch(session).cursorIndex;

  // 4. Switch back to main branch and execute continuous Mask-Diffusion Block 0 (B=4, K=4, γ=0.22, T=0)
  session = switchTwoTowerSessionBranch(session, mainBranchId);
  session = updateTwoTowerSessionConfig(session, {
    mode: 'mask_diffusion',
    blockSize: TWOTOWER_TOUR_CONFIG.blockSize,
    stepsPerBlock: TWOTOWER_TOUR_CONFIG.stepsPerBlock,
    confidenceThreshold: TWOTOWER_TOUR_CONFIG.confidenceThreshold,
    temperature: TWOTOWER_TOUR_CONFIG.temperature,
  });

  session = initTwoTowerDiffusionBlock(session);
  const block0InitIdx = getActiveBranch(session).cursorIndex;

  const block0RefIndices: number[] = [];
  const block0ConfIndices: number[] = [];

  for (let guard = 0; guard < TWOTOWER_TOUR_CONFIG.stepsPerBlock * 2 + 2; guard++) {
    const caps = getSessionCapabilities(session);
    if (caps.canRunDiffusionRefinement) {
      session = runTwoTowerDiffusionRefinement(session);
      block0RefIndices.push(getActiveBranch(session).cursorIndex);
    } else if (caps.canCommitRefinementConfidence) {
      session = commitTwoTowerRefinementConfidence(session);
      block0ConfIndices.push(getActiveBranch(session).cursorIndex);
    } else {
      break;
    }
  }

  const block0Ref0Idx = block0RefIndices[0];
  const block0Conf0Idx = block0ConfIndices[0];
  const block0Ref1Idx = block0RefIndices.length > 1 ? block0RefIndices[1] : block0Ref0Idx;
  const block0Conf1Idx = block0ConfIndices.length > 1 ? block0ConfIndices[1] : block0Conf0Idx;
  const block0CompletedUncommittedIdx = block0ConfIndices[block0ConfIndices.length - 1];

  const completedRec = getActiveRecord(session);
  const block0Nfe = completedRec.activeBlock?.nfe ?? block0RefIndices.length;
  const block0EarlyCompleted = completedRec.activeBlock?.earlyCompleted ?? false;

  // 5. Commit Block 0 explicitly to Context Tower (P = 8 -> 12)
  session = commitTwoTowerActiveBlockToContext(session);
  const block0CommitIdx = getActiveBranch(session).cursorIndex;

  // 6. Initialize Block 1 on new P = 12 boundary and run up to 2 refinements (catching real floor commit!)
  let block1InitIdx = block0CommitIdx;
  let block1Ref0Idx = block0CommitIdx;
  let block1RecapIdx = block0CommitIdx;

  const postCommitCaps = getSessionCapabilities(session);
  if (postCommitCaps.canInitDiffusionBlock) {
    session = initTwoTowerDiffusionBlock(session);
    block1InitIdx = getActiveBranch(session).cursorIndex;

    const block1ConfIndices: number[] = [];
    for (let step = 0; step < 2; step++) {
      const capsRef = getSessionCapabilities(session);
      if (!capsRef.canRunDiffusionRefinement) break;
      session = runTwoTowerDiffusionRefinement(session);
      if (step === 0) {
        block1Ref0Idx = getActiveBranch(session).cursorIndex;
      }
      const capsConf = getSessionCapabilities(session);
      if (!capsConf.canCommitRefinementConfidence) break;
      session = commitTwoTowerRefinementConfidence(session);
      block1ConfIndices.push(getActiveBranch(session).cursorIndex);
    }

    // Prefer a confidence commit where floor commitment triggered, else latest Block 1 confidence commit
    const mainRecords = getActiveBranch(session).records;
    const floorHit = block1ConfIndices.find((idx) => {
      const rec = mainRecords[idx];
      return (
        rec.kind === 'diffusion_confidence_commit' &&
        rec.unmaskStep.floorCommittedIndices.length > 0
      );
    });
    block1RecapIdx =
      floorHit ??
      (block1ConfIndices.length > 0
        ? block1ConfIndices[block1ConfIndices.length - 1]
        : block1Ref0Idx);
  }

  // Park initial cursor cleanly at prefill (beat 0)
  session = switchTwoTowerSessionBranch(session, mainBranchId);
  session = navigateTwoTowerSession(session, prefillIdx);

  return {
    baseSession: session,
    landmarks: Object.freeze({
      mainBranchId,
      arBranchId,
      mockArBranchId,
      prefillIdx,
      arStepIdx,
      mockArStepIdx,
      block0InitIdx,
      block0Ref0Idx,
      block0Conf0Idx,
      block0Ref1Idx,
      block0Conf1Idx,
      block0CompletedUncommittedIdx,
      block0CommitIdx,
      block1InitIdx,
      block1Ref0Idx,
      block1RecapIdx,
      block0Nfe,
      block0EarlyCompleted,
    }),
  };
}

// ============================================================================
// BEAT SPECIFICATION & LIVE DATA RESOLVERS (ALL 28 BEATS)
// ============================================================================

interface TourBeatTarget {
  readonly branchId: string;
  readonly recordIndex: number;
}

interface TourBeatDefinition {
  readonly beatIndex: number;
  readonly beatId: string;
  readonly actIndex: number;
  readonly title: string;
  readonly subtitle: string;
  readonly badge: string;
  readonly viewMode: TwoTowerSpatialViewMode;
  readonly bayId: number;
  readonly towerFocus: TwoTowerPairedFocus;
  readonly linkerChannel: TwoTowerLinkerChannel;
  readonly blockPos: number;
  readonly opStep: number | null;
  readonly showSamplerPanel: boolean;
  readonly showArchPanel: boolean;
  readonly highlightPartId: string | null;
  readonly epistemicTag: string;
  readonly interactivePrompt: string;
  readonly resolveTarget: (landmarks: TourTrajectoryLandmarks) => TourBeatTarget;
  readonly buildNarrativeAndMetrics: (ctx: {
    readonly session: TwoTowerSession;
    readonly record: TwoTowerSessionRecord;
    readonly landmarks: TourTrajectoryLandmarks;
  }) => {
    readonly paragraphs: readonly string[];
    readonly takeaways: readonly string[];
    readonly liveMetrics: readonly TwoTowerTourLiveMetric[];
  };
}

function getBranchRecord(
  session: TwoTowerSession,
  branchId: string,
  recordIndex: number,
): TwoTowerSessionRecord {
  const branch = session.branches.find((b) => b.branchId === branchId);
  if (!branch) {
    throw new RangeError(`Tour branch "${branchId}" not found`);
  }
  const rec = branch.records[recordIndex];
  if (!rec) {
    throw new RangeError(`Tour record #${recordIndex} not found on branch "${branchId}"`);
  }
  return rec;
}

function getTopSoftmaxEntry(
  softmax: Float32Array,
  headIndex: number,
  queryIdx: number,
  queryLen: number,
  keyLen: number,
): { readonly keyIndex: number; readonly weight: number } {
  const base = (headIndex * queryLen + queryIdx) * keyLen;
  let bestKey = 0;
  let bestVal = -Infinity;
  for (let k = 0; k < keyLen; k++) {
    const v = softmax[base + k];
    if (v > bestVal) {
      bestVal = v;
      bestKey = k;
    }
  }
  return { keyIndex: bestKey, weight: Number.isFinite(bestVal) ? bestVal : 0 };
}

function formatSamplerSlotSummary(pres: SamplerStepPresentation, slotIdx: number): string {
  const sl = pres.slots[Math.max(0, Math.min(pres.slots.length - 1, slotIdx))];
  const absPos = sl.absolutePosition ?? sl.slotIndex;
  return `t${absPos} "${sl.selection.selectedTokenLabel}" · ${sl.confidenceDisplay.displayText} (${sl.decision})`;
}

const BEAT_DEFINITIONS: readonly TourBeatDefinition[] = Object.freeze([
  // ==========================================================================
  // ACT I (`act-1`, 6 beats): Dual-Tower Architecture, Prefill & Execution Modes
  // ==========================================================================
  {
    beatIndex: 0,
    beatId: '1.1',
    actIndex: 0,
    title: 'Independent Dual-Tower Architecture',
    subtitle: 'Two 52-layer hybrid towers · Zero weight sharing · Untied bookends',
    badge: '63.19B · 52×2 · Untied',
    viewMode: 'macro',
    bayId: 1,
    towerFocus: 'paired',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'ctx-embed',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Orbit both 52-layer towers · Click Next to enter Context Prefill',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.prefillIdx }),
    buildNarrativeAndMetrics: ({ session }) => {
      const model = session.model;
      const inv = TWOTOWER_SPEC.derivedInventoryCalculations;
      const ctxFp = fmtShortDigest(model.contextModel.parameterFingerprint);
      const denFp = fmtShortDigest(model.denoiserModel.parameterFingerprint);
      const timeFp = fmtShortDigest(model.timeConditioning.fingerprint);
      const miniWeightBytes = model.retainedWeightBytes();
      const promptPreview = session.promptTokenLabels.join(' ');
      return {
        paragraphs: [
          `Nemotron-Labs-TwoTower pairs TWO completely independent 52-layer hybrid towers (23 Mamba-2 · 23 Granular MoE · 6 GQA each): the left Context Tower encodes causal prefix history, while the right Denoiser Tower performs bidirectional masked block denoising. Neither embeddings nor LM heads are tied (tie_word_embeddings = false), and zero backbone weights are shared across towers.`,
          `In the released BF16 checkpoint (HF rev 35b64985), total parameters reach ${(inv.grandTotalParams / 1e9).toFixed(2)}B (${formatGiB(inv.derivedPayloadBytes)} across 24 shards): two 31.58B backbones plus 30.02M denoiser-only adaLN-single conditioning parameters (versus the paper's 1.5M note). This guided walkthrough runs an isolated live miniature simulator on the 8-token prompt "${promptPreview}".`,
        ],
        takeaways: [
          `Distinct tower fingerprints: Context ${ctxFp} ≠ Denoiser ${denFp} · Time-cond ${timeFp} (denoiser-only)`,
          `Untied W_embed and W_lm_head on both towers — four independent vocabulary matrices total`,
        ],
        liveMetrics: [
          metric('Context Fingerprint', ctxFp, 'context'),
          metric('Denoiser Fingerprint', denFp, 'denoiser'),
          metric('Time-Cond Fingerprint', timeFp, 'shared'),
          metric('Prod Dual-Tower BF16', `${(inv.grandTotalParams / 1e9).toFixed(2)}B (${formatGiB(inv.derivedPayloadBytes)})`, 'gold'),
          metric('Mini Retained Weights', formatBytesAdaptive(miniWeightBytes), 'muted'),
        ],
      };
    },
  },
  {
    beatIndex: 1,
    beatId: '1.2',
    actIndex: 0,
    title: 'Context Prefill (P = 8) & Untied NoPE Embeddings',
    subtitle: 'W_embed_ctx lookup · Zero positional tables · Split mamba_s2 prefill',
    badge: 'P = 8 · NoPE · 2 Segs',
    viewMode: 'embedding',
    bayId: 0,
    towerFocus: 'context_solo',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: 1,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'token-strip',
    epistemicTag: EPISTEMIC_MINIATURE_SIMULATION,
    interactivePrompt: 'Inspect W_embed_ctx rows for tokens t0..t7 · Advance for Persistent Hybrid Cache',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.prefillIdx }),
    buildNarrativeAndMetrics: ({ session, record }) => {
      const exec = record.contextExecution!;
      const seg0 = exec.segments[0];
      const seg1 = exec.segments[exec.segments.length - 1];
      const bay1Trace = seg0.getMambaTrace(1);
      const x0L2 = columnL2Norm(bay1Trace.X, bay1Trace.d, bay1Trace.tokens, 0);
      return {
        paragraphs: [
          `Every generation begins with a single causal Context Tower prefill across the prompt "${session.promptTokenLabels.join(' ')}" (positions [0..${record.committedPrefixLen - 1}], P = ${record.committedPrefixLen}). Each token ID looks up one row of Context's untied embedding matrix W_embed_ctx (V = ${session.model.vocab.length} miniature / 131,072 production) with NoPE — zero positional embeddings are added here.`,
          `Internally, multi-token Context prefill splits into ${exec.segments.length} exact segments: prefix [${seg0.tokenRange[0]}..${seg0.tokenRange[1]}] snapshots pre-last-token Mamba state (mamba_s2) before processing final token [${seg1.tokenRange[0]}] ("${seg1.tokenLabels[0]}"), preserving exact conditioning for both Mock-AR and Mask-Diffusion.`,
        ],
        takeaways: [
          `Split prefill segments: [${seg0.tokenRange[0]}..${seg0.tokenRange[1]}] (${seg0.segmentRole}) + [${seg1.tokenRange[0]}..${seg1.tokenRange[1]}] (${seg1.segmentRole})`,
          `NoPE delegates sequence order naturally to Mamba-2's 4-tap Conv1D and selective SSM recurrence`,
        ],
        liveMetrics: [
          metric('Committed Prefix P', `${record.committedPrefixLen} / ${record.kvCapacity} cap`, 'context'),
          metric('Split Prefill Segments', `${exec.segments.length} ([${seg0.tokenRange[0]}..${seg0.tokenRange[1]}] + [${seg1.tokenRange[0]}])`, 'shared'),
          metric('W_embed_ctx', `fp ${fmtShortDigest(session.model.contextModel.parameterFingerprint)}`, 'context'),
          metric('X₀[t0] L2 Norm', fmtFixed(x0L2, 4), 'gold'),
        ],
      };
    },
  },
  {
    beatIndex: 2,
    beatId: '1.3',
    actIndex: 0,
    title: 'Persistent Hybrid Cache: GQA KV + Dual-Compartment Mamba vs. 0 B MoE',
    subtitle: 'Three distinct memory regimes owned solely by Context Tower',
    badge: '47.08 MiB · +6 KiB/tok · 0 B',
    viewMode: 'macro',
    bayId: 6,
    towerFocus: 'context_solo',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'ctx-moe-body-02',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Compare solid Context KV tracks, Mamba compartments, and 0 B MoE bays',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.prefillIdx }),
    buildNarrativeAndMetrics: ({ session, record }) => {
      const prodMem = twoTowerMemoryFor(record.committedPrefixLen, TWOTOWER_TOUR_CONFIG.blockSize, 'mask_diffusion');
      const miniMem = measureTwoTowerSessionRetainedBytes(session);
      return {
        paragraphs: [
          `Context Tower's solid geometry represents the ONLY persistent sequence cache in TwoTower, spanning three distinct memory regimes across 52 bays: 6 GQA attention bays ({06, 13, 20, 27, 34, 43}) append verbatim K/V tracks (+6 KiB/tok in BF16), 23 Mamba-2 bays hold dual-compartment constant registers (4-tap BF16 Conv1D + FP32 SSM H = ${formatMiB(prodMem.persistentContextMambaBytes)} fixed), and 23 MoE bays store strictly 0 B statelessly.`,
          `At P = ${record.committedPrefixLen} tokens, Context's persistent prefix cache totals ${formatMiB(prodMem.persistentContextTotalBytes)} in production BF16 (${formatMiB(prodMem.persistentContextMambaBytes)} Mamba + ${formatKiB(prodMem.persistentContextKvBytes)} GQA KV), with digest ${fmtShortDigest(record.contextSnapshot.ctxState.stateDigest)} anchoring every downstream step.`,
        ],
        takeaways: [
          `6 GQA bays grow linearly (+6 KiB/tok BF16) — 8.7× less KV growth than a 52-layer pure GQA Transformer`,
          `23 Mamba-2 bays keep two distinct compartments (Conv1D 4-tap history + FP32 SSM H); 23 MoE bays hold 0 B`,
        ],
        liveMetrics: [
          metric('Prod Mamba Fixed (×23)', formatMiB(prodMem.persistentContextMambaBytes), 'context'),
          metric(`Prod GQA KV (P=${record.committedPrefixLen})`, formatKiB(prodMem.persistentContextKvBytes), 'context'),
          metric('MoE Sequence Cache (×23)', '0 B (stateless)', 'shared'),
          metric('Context State Digest', fmtShortDigest(record.contextSnapshot.ctxState.stateDigest), 'gold'),
          metric('Mini Snapshot Footprint', formatBytesAdaptive(miniMem.contextSnapshotBytes), 'muted'),
        ],
      };
    },
  },
  {
    beatIndex: 3,
    beatId: '1.4',
    actIndex: 0,
    title: 'Mode 1 · Autoregressive (AR) Context-Only Decode',
    subtitle: 'Isolated side branch · Single active tower · Timestep rail bypassed',
    badge: 'Mode: AR · t = null',
    viewMode: 'macro',
    bayId: 53,
    towerFocus: 'context_solo',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'ctx-output',
    epistemicTag: EPISTEMIC_MINIATURE_SIMULATION,
    interactivePrompt: 'Notice Denoiser & timestep rail dimmed · Advance to compare Mock-AR',
    resolveTarget: (lm) => ({ branchId: lm.arBranchId, recordIndex: lm.arStepIdx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const arRec = record.kind === 'ar_step' ? record : null;
      const tokLabel = arRec ? arRec.selectedTokenLabel : 'n/a';
      const tokId = arRec ? arRec.selectedTokenId : -1;
      const topProb = arRec ? arRec.selectionProbs[tokId] ?? 0 : 0;
      const prodAr = twoTowerMemoryFor(record.committedPrefixLen, 0, 'ar');
      return {
        paragraphs: [
          `On an isolated side branch forked from the exact P = 8 prefill state, Autoregressive ('ar') mode runs ONLY the Context Tower — one active 31.58B tower (${formatGiB(prodAr.activeModelWeightBytes)} BF16 in production). The Denoiser Tower, cross-tower linkers, and adaLN timestep rail remain completely idle (t = null).`,
          `Context's untied W_lm_head_ctx evaluates shifted next-token logits at position t7 and commits token #${tokId} ("${tokLabel}", p = ${fmtPct(topProb)} under synthetic miniature weights) directly onto Context's persistent cache, advancing P = 8 → ${record.committedPrefixLen} (lifecycle: ${record.lifecyclePhase}).`,
        ],
        takeaways: [
          `AR mode activates 1 tower only (Context); zero cross-tower transfer and zero Denoiser scratchpad allocation`,
          `Because this step runs on isolated branch "${record.identity.branchId}", the main diffusion branch stays untouched at P = 8`,
        ],
        liveMetrics: [
          metric('Execution Mode', 'ar (Context-only)', 'context'),
          metric('Prefix Advance', `8 → ${record.committedPrefixLen}`, 'context'),
          metric('Selected Token (Ctx)', `#${tokId} "${tokLabel}" (${fmtPct(topProb)})`, 'gold'),
          metric('Timestep Conditioning', 'BYPASSED (t = null)', 'muted'),
          metric('Prod Active Weights', formatGiB(prodAr.activeModelWeightBytes), 'shared'),
        ],
      };
    },
  },
  {
    beatIndex: 4,
    beatId: '1.5',
    actIndex: 0,
    title: 'Mode 2 · Mock-AR Mode (KV[:-1] + mamba_s2, Unmodulated)',
    subtitle: 'Denoiser causal S=1 re-evaluation · Exact parity harness when ΔW = 0',
    badge: 'Mock-AR · KV[:-1] + mamba_s2',
    viewMode: 'macro',
    bayId: 1,
    towerFocus: 'paired',
    linkerChannel: 'mamba_ssm',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'linker-mamba-ssm-bay-01',
    epistemicTag: EPISTEMIC_MINIATURE_SIMULATION,
    interactivePrompt: 'Compare Mock-AR token against Context AR from Beat 1.4 · Next for Mask-Diffusion',
    resolveTarget: (lm) => ({ branchId: lm.mockArBranchId, recordIndex: lm.mockArStepIdx }),
    buildNarrativeAndMetrics: ({ session, record, landmarks }) => {
      const mockRec = record.kind === 'mock_ar_step' ? record : null;
      const arRec = getBranchRecord(session, landmarks.arBranchId, landmarks.arStepIdx);
      const arLabel = arRec.kind === 'ar_step' ? `#${arRec.selectedTokenId} "${arRec.selectedTokenLabel}"` : 'n/a';
      const tokId = mockRec ? mockRec.selectedTokenId : -1;
      const tokLabel = mockRec ? mockRec.selectedTokenLabel : 'n/a';
      const topProb = mockRec ? mockRec.selectionProbs[tokId] ?? 0 : 0;
      const kvCondLen = mockRec ? mockRec.conditioning.committedPrefixLen : 7;
      const prodMock = twoTowerMemoryFor(record.committedPrefixLen, 0, 'mock_ar');
      return {
        paragraphs: [
          `Mock-AR ('mock_ar') tests cross-tower conditioning in causal 1-token decode: Context transfers KV[:-1] (${kvCondLen} slices, positions [0..${kvCondLen - 1}]) plus its pre-last-token Mamba snapshot mamba_s2 (+${formatMiB(prodMock.mockArMambaS2SnapshotBytes)} in production), and Denoiser re-runs token P − 1 causally with ALL adaLN timestep modulation bypassed (t = null).`,
          `From the exact same P = 8 prefill state, Denoiser's independent weights (perturbation scale ${session.model.config.perturbation.scale}) predict #${tokId} "${tokLabel}" (${fmtPct(topProb)}) versus Context AR's ${arLabel} — demonstrating independent tower parameterization live. When weight perturbation is 0, Mock-AR reproduces Context AR to 0 ULP.`,
        ],
        takeaways: [
          `Mock-AR explicitly requires mamba_s2 (pre-last-token Mamba state) + KV[:-1] and sets tModel = null (unmodulated)`,
          `Live divergence from Context AR (${arLabel} vs #${tokId} "${tokLabel}") confirms non-shared Context/Denoiser weights`,
        ],
        liveMetrics: [
          metric('Execution Mode', 'mock_ar (S=1 causal)', 'denoiser'),
          metric('Conditioned KV[:-1]', `${kvCondLen} slices + mamba_s2`, 'shared'),
          metric('Timestep Conditioning', 'BYPASSED (t = null)', 'muted'),
          metric('Selected Token (Den)', `#${tokId} "${tokLabel}" (${fmtPct(topProb)})`, 'gold'),
          metric('Prod Extra mamba_s2', `+${formatMiB(prodMock.mockArMambaS2SnapshotBytes)}`, 'warn'),
        ],
      };
    },
  },
  {
    beatIndex: 5,
    beatId: '1.6',
    actIndex: 0,
    title: 'Mode 3 · Mask-Diffusion Dual-Tower Overview',
    subtitle: 'Frozen Context boundary (P = 8) + Iterative Denoiser refinement (B = 4, K = 4)',
    badge: 'Mask-Diffusion · B=4 · K=4',
    viewMode: 'macro',
    bayId: 6,
    towerFocus: 'paired',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'pathway-refinement-loop',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Observe the active blue Refinement Loop vs dormant emerald Commit Pathway',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const refRec = record.kind === 'diffusion_refinement' ? record : null;
      const tVal = refRec ? refRec.tModel : 1;
      const B = refRec ? refRec.blockConfig.blockSize : 4;
      const K = refRec ? refRec.blockConfig.stepsPerBlock : 4;
      const align = TWOTOWER_SPEC.layerAlignment;
      return {
        paragraphs: [
          `Now we return to the main continuous Mask-Diffusion branch ('mask_diffusion'). Context Tower holds the P = ${record.committedPrefixLen} prefix boundary completely frozen while Denoiser Tower refines a B = ${B} block ([${record.identity.absoluteTokenRange[0]}..${record.identity.absoluteTokenRange[1]}], step budget K = ${K}) in transient dashed/frosted scratchpad space.`,
          `Across the 52 aligned bays, strictly ${align.alignedTransferBayCount} stateful bays (${align.gqaTransferBayCount} GQA + ${align.mambaTransferBayCount} Mamba-2) condition Denoiser read-only from Context, while ${align.statelessMoeBayCount} MoE bays carry zero cross-tower cache. Each refinement pass r loops on the right until nMasked reaches 0, and Context updates ONLY on explicit block commit.`,
        ],
        takeaways: [
          `Active refinement r = ${refRec?.refinementIndex ?? 0} executes with t = ${fmtFixed(tVal, 3)} over [${record.identity.absoluteTokenRange[0]}..${record.identity.absoluteTokenRange[1]}] while Context stays frozen at P = ${record.committedPrefixLen}`,
          `Two separate loops: ephemeral Denoiser Refinement Loop (right) vs once-per-block Context Commit Pathway (bottom→left)`,
        ],
        liveMetrics: [
          metric('Execution Mode', 'mask_diffusion', 'denoiser'),
          metric('Block & Budget', `Blk ${record.identity.blockId ?? 0} · B=${B} · K=${K}`, 'denoiser'),
          metric('Context Prefix P', `${record.committedPrefixLen} (frozen)`, 'context'),
          metric('Executed t = nM/B', `${fmtFixed(tVal, 3)} (${record.activeBlock?.nMasked ?? B}/${B})`, 'gold'),
          metric('Transfer / Stateless', `${align.alignedTransferBayCount} stateful · ${align.statelessMoeBayCount} MoE (0 B)`, 'shared'),
        ],
      };
    },
  },

  // ==========================================================================
  // ACT II (`act-2`, 10 beats): Conditioning, Paired Layers, Recurrent Mechanics & Routing
  // ==========================================================================
  {
    beatIndex: 6,
    beatId: '2.1',
    actIndex: 1,
    title: 'Masked Block Initialization ([MASK]^4) & Denoiser Embeddings',
    subtitle: 'Shared ~78 KiB conditioning base · Independent W_embed_den lookup',
    badge: 'Blk 0 Init · [MASK]^4',
    viewMode: 'embedding',
    bayId: 0,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: 0,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'token-strip',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Inspect all four [MASK] (#32) input slots entering W_embed_den',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0InitIdx }),
    buildNarrativeAndMetrics: ({ session, record }) => {
      const blk = record.activeBlock!;
      const model = session.model;
      const condDigest = fmtShortDigest(blk.conditioningBase.baseConditioning.conditioningDigest);
      return {
        paragraphs: [
          `Initializing Block ${blk.blockId} allocates a single immutable TwoTowerBlockConditioningBase from Context's P = ${record.committedPrefixLen} snapshot (digest ${condDigest}) shared across every refinement of the block, and fills all B = ${blk.blockConfig.blockSize} block slots ([${blk.absoluteTokenRange[0]}..${blk.absoluteTokenRange[1]}]) with [MASK] (#${model.maskTokenId} in miniature; production repurposes token ID ${TWOTOWER_SPEC.directlyInspectedMetadata.maskTokenId}, surface "${TWOTOWER_SPEC.directlyInspectedMetadata.maskTokenSurfaceForm}").`,
          `Denoiser maps these masked slots through its OWN untied NoPE embedding table W_embed_den (fingerprint ${fmtShortDigest(model.denoiserModel.parameterFingerprint)} ≠ Context ${fmtShortDigest(model.contextModel.parameterFingerprint)}), producing identical starting rows before Context conditioning and bidirectional GQA differentiate positions.`,
        ],
        takeaways: [
          `All ${blk.nMasked}/${blk.blockConfig.blockSize} slots start as [MASK] (#${model.maskTokenId} mini / #3 [INST] prod); zero Denoiser pass has executed yet`,
          `One shared conditioning base serves every refinement r = 0..K−1 of Block ${blk.blockId} without duplicating Context caches`,
        ],
        liveMetrics: [
          metric('Active Block Range', `Blk ${blk.blockId} · [${blk.absoluteTokenRange[0]}..${blk.absoluteTokenRange[1]}]`, 'denoiser'),
          metric('Unresolved Masks', `${blk.nMasked} / ${blk.blockConfig.blockSize} [MASK]`, 'gold'),
          metric('Mask Token ID', `#${model.maskTokenId} mini (#3 [INST] prod)`, 'shared'),
          metric('W_embed_den', `fp ${fmtShortDigest(model.denoiserModel.parameterFingerprint)}`, 'denoiser'),
          metric('Conditioning Digest', condDigest, 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 7,
    beatId: '2.2',
    actIndex: 1,
    title: 'Timestep Conditioning: t = nMasked / B → φ(t) → MLP → 3d tEmb',
    subtitle: 'Noise level tracks remaining masks — NEVER step index r / K',
    badge: 't = nM/B · COS-FIRST φ(t)',
    viewMode: 'timestep',
    bayId: 6,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'timestep',
    blockPos: 0,
    opStep: 1,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'phi-sinusoid',
    epistemicTag: EPISTEMIC_MINIATURE_SIMULATION,
    interactivePrompt: 'Trace scalar t through φ(t), 2-layer SiLU MLP, t_block, and Bay 06 table',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const pass = record.denoiserPass!;
      const proj = pass.timestepProjection!;
      const mod6 = pass.getLayerModulation(6)!;
      const nMasked = record.activeBlock?.nMasked ?? pass.blockSize;
      const tVal = pass.tModel ?? nMasked / pass.blockSize;
      return {
        paragraphs: [
          `Before layer 1 runs, Denoiser computes the continuous noise level strictly from the remaining masked fraction t = nMasked / B = ${nMasked} / ${pass.blockSize} = ${fmtFixed(tVal, 3)} — NEVER from step index r / K! When multi-token confidence unmasking jumps several slots at once, t drops proportionally to actual remaining uncertainty.`,
          `Scalar t scales by 1000 in FP32 into a COS-FIRST sinusoidal embedding φ(t) (maxPeriod = 10000; 16-dim mini / 256-dim prod), maps through a 2-layer SiLU MLP into tRepr (L2 = ${fmtFixed(l2Norm(proj.tRepr), 3)}), projects via t_block into shared 3d global tEmb (L2 = ${fmtFixed(l2Norm(proj.tEmb), 3)}), and adds each bay's learned scale_shift_tables[l] to form per-layer (shift, scale, gate).`,
        ],
        takeaways: [
          `Exact formula: t = nMasked / B (never r / K) → φ(t × 1000) COS-FIRST → MLP → 3d tEmb + scale_shift_tables[l]`,
          `Bay 06 live modulation norms at t = ${fmtFixed(tVal, 3)}: ‖shift‖=${fmtFixed(l2Norm(mod6.shift), 3)}, ‖scale‖=${fmtFixed(l2Norm(mod6.scale), 3)}, ‖gate‖=${fmtFixed(l2Norm(mod6.gate), 3)}`,
        ],
        liveMetrics: [
          metric('Executed t = nM/B', `${fmtFixed(tVal, 3)} (${nMasked}/${pass.blockSize})`, 'gold'),
          metric('φ(t) [16] L2 / cos₀', `${fmtFixed(l2Norm(proj.phi), 3)} · cos₀=${fmtFixed(proj.phi[0], 3)}`, 'denoiser'),
          metric('tRepr [8] L2', fmtFixed(l2Norm(proj.tRepr), 3), 'denoiser'),
          metric('Global tEmb [24] L2', fmtFixed(l2Norm(proj.tEmb), 3), 'denoiser'),
          metric('Bay 06 (s, sc, g) L2', `${fmtFixed(l2Norm(mod6.shift), 2)} / ${fmtFixed(l2Norm(mod6.scale), 2)} / ${fmtFixed(l2Norm(mod6.gate), 2)}`, 'shared'),
        ],
      };
    },
  },
  {
    beatIndex: 8,
    beatId: '2.3',
    actIndex: 1,
    title: 'Asymmetric adaLN-Single Ordering, Untouched Residual X & Unmodulated norm_f',
    subtitle: 'M/* modulate→RMSNorm vs E RMSNorm→modulate · Residual bypass untouched',
    badge: 'Split AdaLN · norm_f Raw',
    viewMode: 'timestep',
    bayId: 2,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'timestep',
    blockPos: 0,
    opStep: 5,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'final-norm-unmodulated',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Compare Bay 02 MoE (RMSNorm→modulate) against Mamba/GQA ordering & untouched X',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const pass = record.denoiserPass!;
      const mod2 = pass.getLayerModulation(2)!;
      const mod1 = pass.getLayerModulation(1)!;
      const orderM = modulationOrderForBayKind('M');
      const orderGqa = modulationOrderForBayKind('*');
      const orderMoe = modulationOrderForBayKind('E');
      const finalNormRms0 = pass.finalNorm.rms[0];
      return {
        paragraphs: [
          `TwoTower's PixArt-α adaLN-single conditioning uses a verified split ordering across layer types: all 23 Mamba-2 ('M') and 6 GQA ('*') bays apply modulate BEFORE RMSNorm (X → X·(1+scale)+shift → RMSNorm), whereas all 23 Granular MoE ('E', shown here at Bay 02) apply RMSNorm BEFORE modulate (X → RMSNorm(X) → U·(1+scale)+shift).`,
          `Equally important: across all 52 layers, modulation touches ONLY the sublayer input branch — the emerald residual highway X passes straight through untouched (Y = X + gate ⊙ sublayer(...)), and the final output normalization norm_f (Bay 52 → LM Head) is strictly UNMODULATED.`,
        ],
        takeaways: [
          `29 bays (M & *): ${orderM} (${orderGqa}) · 23 bays (E): ${orderMoe}`,
          `Untouched residual bypass X + channel-wise outer gate ⊙ sublayer across 52 bays · Final RMSNorm norm_f strictly unmodulated`,
        ],
        liveMetrics: [
          metric('Mamba (×23) & GQA (×6)', orderM, 'shared'),
          metric('MoE (×23 · Bay 02)', orderMoe, 'gold'),
          metric('Bay 02 Gate [8] L2', fmtFixed(l2Norm(mod2.gate), 4), 'denoiser'),
          metric('Bay 01 Gate [8] L2', fmtFixed(l2Norm(mod1.gate), 4), 'denoiser'),
          metric('Residual X / norm_f', `X untouched · norm_f raw (rms₀=${fmtFixed(finalNormRms0, 3)})`, 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 9,
    beatId: '2.4',
    actIndex: 1,
    title: 'Three Cross-Tower Linker Channels (and Zero MoE Linkers)',
    subtitle: '29 aligned transfer bays · Read-only Context prefix conditioning · 0 MoE links',
    badge: '6 KV · 23 Conv · 23 SSM · 0 MoE',
    viewMode: 'macro',
    bayId: 1,
    towerFocus: 'paired',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'linker-gqa-kv-bay-06',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Trace the 3 bridge channels between aligned bays — notice 0 bridges at MoE bays',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const cond = record.conditioning!;
      const align = TWOTOWER_SPEC.layerAlignment;
      return {
        paragraphs: [
          `Across the central cross-tower bridge, Context Tower conditions Denoiser Tower through THREE distinct read-only channels aligned strictly layer-by-layer across ${align.alignedTransferBayCount} stateful bays: (1) GQA KV prefix slices KV[0..${cond.committedPrefixLen - 1}] across the ${align.gqaTransferBayCount} attention bays, (2) Mamba Conv1D 4-tap raw history across the ${align.mambaTransferBayCount} Mamba bays, and (3) Mamba FP32 SSM boundary state H_{P−1} across the same ${align.mambaTransferBayCount} Mamba bays.`,
          `All ${align.statelessMoeBayCount} Granular MoE bays have ZERO cross-tower linkers because MoE routing and expert MLPs are strictly per-token stateless (0 B). Every refinement pass r = 0..K−1 seeds freshly from this exact SAME committed Context boundary (digest ${fmtShortDigest(cond.sourceStateDigest)}), never from prior Denoiser scratch.`,
        ],
        takeaways: [
          `3 separate channels: 6 GQA KV prefix + 23 Mamba Conv1D raw history + 23 Mamba FP32 SSM H_{P−1} (0 at 23 MoE bays)`,
          `Read-only logical prefix conditioning: Context remains the sole state owner; Denoiser never mutates Context buffers`,
        ],
        liveMetrics: [
          metric('Channel 1 · GQA KV', `${cond.gqaPrefixCaches.size} bays × P=${cond.committedPrefixLen} slices`, 'context'),
          metric('Channel 2 · Mamba Conv1D', `${cond.mambaSeedStates.size} bays × 4-col raw history`, 'gold'),
          metric('Channel 3 · Mamba SSM', `${cond.mambaSeedStates.size} bays × FP32 H_{P−1}`, 'shared'),
          metric('MoE Cross-Tower Links', '0 (23 stateless bays)', 'muted'),
          metric('Source Cache Digest', fmtShortDigest(cond.sourceStateDigest), 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 10,
    beatId: '2.5',
    actIndex: 1,
    title: 'Paired GQA (Bay 06): Inherited KV[0..7] + Bidirectional Block [8..11]',
    subtitle: 'Logical read-only prefix · Unmasked intra-block attention · 32Q / 2KV',
    badge: 'Bay 06 · [4 × 12] Bidir',
    viewMode: 'paired_gqa',
    bayId: 6,
    towerFocus: 'paired',
    linkerChannel: 'gqa_kv',
    blockPos: 0,
    opStep: 2,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'conditioning-bridge',
    epistemicTag: EPISTEMIC_MINIATURE_SIMULATION,
    interactivePrompt: 'Compare read-only Context KV[0..7] + Denoiser block KV[8..11] in the 4×12 Scores sheet',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const pass = record.denoiserPass!;
      const gqa = pass.getGqaTrace(6);
      const top0 = getTopSoftmaxEntry(gqa.Softmax, 0, 0, gqa.tokens, gqa.keyTokens);
      const top0Scope = top0.keyIndex < gqa.prefixTokens ? 'inherited Context prefix' : 'active bidirectional block';
      return {
        paragraphs: [
          `Zooming into Paired GQA Bay 06 (layer 5, first of 6 attention layers): the left wing holds Context's persistent prefix KV[0..${gqa.prefixTokens - 1}] logically bridged read-only (no physical copy required) into the right wing's concatenated key/value bank alongside Denoiser's ephemeral block projections KV[${gqa.prefixTokens}..${gqa.keyTokens - 1}].`,
          `Unlike Context's causal upper-triangle mask, Denoiser runs strictly bidirectional block attention (mode: ${gqa.attentionMaskMode}): every block query [0..${gqa.tokens - 1}] attends across ALL ${gqa.keyTokens} keys (${gqa.prefixTokens} inherited + ${gqa.tokens} active block). All ${gqa.nQ} miniature Q heads share ${gqa.nKv} KV heads (16:1 NoPE), with the active head pair rendered directly from the 128×128 value atlas.`,
        ],
        takeaways: [
          `Bidirectional block visibility: [B=${gqa.tokens} × (P+B)=${gqa.keyTokens}] Scores/Softmax — later block positions inform earlier ones`,
          `All 32 miniature Q heads / 2 KV heads (16:1) execute live; selected Group 0 / Group 1 slice packs into the 128×128 atlas`,
        ],
        liveMetrics: [
          metric('Attention Mask Mode', gqa.attentionMaskMode, 'denoiser'),
          metric('KV Bank Composition', `${gqa.prefixTokens} prefix + ${gqa.tokens} block = ${gqa.keyTokens} keys`, 'shared'),
          metric('Head Geometry', `${gqa.nQ}Q / ${gqa.nKv}KV (16:1 · NoPE)`, 'context'),
          metric('Q00 @ p=0 Peak Key', `k=${top0.keyIndex} (${fmtPct(top0.weight)}, ${top0Scope})`, 'gold'),
          metric('Bay 06 Modulation', 'modulate_then_norm + outer gate', 'denoiser'),
        ],
      };
    },
  },
  {
    beatIndex: 11,
    beatId: '2.6',
    actIndex: 1,
    title: 'Paired Mamba-2 Conv1D Boundary Replacement (3→2→1→0)',
    subtitle: '4 allocated cache columns · 3 preceding raw history taps stepping out across p=0..3',
    badge: 'Bay 01 · Conv1D 3→2→1→0',
    viewMode: 'paired_mamba',
    bayId: 1,
    towerFocus: 'paired',
    linkerChannel: 'mamba_conv',
    blockPos: 0,
    opStep: 2,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'den-conv1d-window',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Step local block position p=0..3 to watch inherited Conv1D columns step 3→2→1→0',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const pass = record.denoiserPass!;
      const m1 = pass.getMambaTrace(1);
      const decomps = [0, 1, 2, 3].map((p) =>
        extractMambaFullBlockPositionState(m1, 0, p, { bayId: 1 }),
      );
      const inhSeq = decomps.map((d) => d.convWindow.inheritedHistoryColumnCount).join('→');
      const blkSeq = decomps.map((d) => d.convWindow.blockColumnCount).join('→');
      const d0 = decomps[0];
      return {
        paragraphs: [
          `Inside Paired Mamba-2 Bay 01 (layer 0, pre-GQA), the Conv1D linker carries Context's persistent raw pre-convolution history xBC_pre. NVIDIA allocates ${d0.convWindow.allocatedCacheColumns} cache columns (slots [0..3], slot 3 = newest prior token t${d0.absoluteTokenPosition - 1}), of which exactly ${d0.convWindow.precedingHistoryColumnsAtPos0} preceding raw history columns (lags 3, 2, 1 → slots 1, 2, 3) combine with lag 0 at local position p = 0.`,
          `Moving across active block positions p = 0 → 1 → 2 → 3 (t8..t11), inherited Context Conv1D history taps systematically step down ${inhSeq} while active-block token columns step up ${blkSeq}. By p = 3, the 4-tap depthwise Conv1D window runs entirely over active block tokens.`,
        ],
        takeaways: [
          `Exact cache convention: 4 allocated Conv1D cache columns → 3 active preceding history taps at p = 0`,
          `Real tap replacement across Block 0: inherited ${inhSeq} · active block ${blkSeq} · zero-pad ${d0.convWindow.zeroPadColumnCount}`,
        ],
        liveMetrics: [
          metric('Allocated vs Preceding', `${d0.convWindow.allocatedCacheColumns} cols · ${d0.convWindow.precedingHistoryColumnsAtPos0} active @ p=0`, 'context'),
          metric('Inherited Taps (p=0..3)', inhSeq, 'gold'),
          metric('Block Taps (p=0..3)', blkSeq, 'denoiser'),
          metric('Active Inspect Pos', `p=0 (abs t${d0.absoluteTokenPosition}, "${d0.tokenLabel}")`, 'shared'),
          metric('Bay 01 Pre-GQA Scope', `afterBidirectionalGqa = ${String(d0.isAfterFirstBidirectionalGqa)}`, 'muted'),
        ],
      };
    },
  },
  {
    beatIndex: 12,
    beatId: '2.7',
    actIndex: 1,
    title: 'Full-Block Position Inspection & Local Causality After Bay 06 GQA',
    subtitle: 'Bay 08 Mamba stays locally causal — yet receives bidirectionally mixed inputs X',
    badge: 'Bay 08 · p=2 · Post-GQA',
    viewMode: 'paired_mamba',
    bayId: 8,
    towerFocus: 'paired',
    linkerChannel: 'mamba_ssm',
    blockPos: 2,
    opStep: 3,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'den-h-curr',
    epistemicTag: EPISTEMIC_MAMBA_ALGEBRAIC,
    interactivePrompt: 'Inspect Bay 08 at p=2 (t10) read strictly from full-block H_history[2]',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const pass = record.denoiserPass!;
      const m8 = pass.getMambaTrace(8);
      const d2 = extractMambaFullBlockPositionState(m8, 0, 2, { bayId: 8 });
      return {
        paragraphs: [
          `Bay 08 (layer 7) is the first Mamba-2 layer AFTER Bay 06 (the first bidirectional GQA layer). Position-specific inspection at p = 2 (abs t${d2.absoluteTokenPosition}) reads strictly from the complete executed block's H_history[2] — never by re-running Mamba on a truncated prefix.`,
          `Notice the architectural subtlety: inside Bay 08 itself, Conv1D (${d2.convWindow.inheritedHistoryColumnCount} inherited + ${d2.convWindow.blockColumnCount} block taps) and SSM recurrence remain strictly locally causal across p = 0..3. Yet because upstream Bay 06 GQA mixed all 4 block positions bidirectionally, Bay 08's input activation X[:, p=2] already carries contextual signal from later position p = 3!`,
        ],
        takeaways: [
          `Local recurrence causality ≠ global causal isolation once a Mamba bay sits downstream of bidirectional GQA (Bay 06)`,
          `Full-block invariant: inspecting p = 2 reads exact H_history[2] of the complete 4-token pass (0 ULP match)`,
        ],
        liveMetrics: [
          metric('Bay & Relation to GQA', `Bay 08 (L7) · afterBay06=${String(d2.isAfterFirstBidirectionalGqa)}`, 'gold'),
          metric('Block Position', `p=${d2.localPos} (abs t${d2.absoluteTokenPosition})`, 'denoiser'),
          metric('Conv1D Window @ p=2', `${d2.convWindow.inheritedHistoryColumnCount} inherited + ${d2.convWindow.blockColumnCount} block`, 'shared'),
          metric('H00 Decay dA / Π dA', `${fmtFixed(d2.decayFactor, 4)} / ${fmtFixed(d2.cumulativeDecay, 4)}`, 'denoiser'),
          metric('H00 ‖H_curr‖₂', fmtFixed(d2.hCurrL2Norm, 4), 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 13,
    beatId: '2.8',
    actIndex: 1,
    title: 'Inherited-State Decay vs. New Block Writes (H_p = A_cum·H_init + ΔH_write + ε_f32)',
    subtitle: 'Closed-form 16-head decomposition · Algebraic magnitudes ≠ semantic retention',
    badge: 'Bay 42 · p=3 · 16-Head Decay',
    viewMode: 'paired_mamba',
    bayId: 42,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'mamba_ssm',
    blockPos: 3,
    opStep: 3,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'den-inherited-retention',
    epistemicTag: EPISTEMIC_MAMBA_ALGEBRAIC,
    interactivePrompt: 'Compare Inherited Retention vs Accumulated Block Write plates across H00..H15',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const pass = record.denoiserPass!;
      const m42 = pass.getMambaTrace(42);
      const d3 = extractMambaFullBlockPositionState(m42, 0, 3, { bayId: 42 });
      let minHead = d3.perHeadDecayProfile[0];
      let maxHead = d3.perHeadDecayProfile[0];
      for (const entry of d3.perHeadDecayProfile) {
        if (entry.cumulativeDecay < minHead.cumulativeDecay) minHead = entry;
        if (entry.cumulativeDecay > maxHead.cumulativeDecay) maxHead = entry;
      }
      return {
        paragraphs: [
          `At Bay 42 (layer 41), final block position p = 3 (abs t${d3.absoluteTokenPosition}), linear recurrence allows exact closed-form decomposition of every 4×4 FP32 state matrix: H_p = A_cum·H_inherited + ΔH_block_write + ε_f32. For ${d3.headLabel}, cumulative decay Π dA = ${fmtFixed(d3.cumulativeDecay, 4)}, inherited retention norm = ${fmtFixed(d3.inheritedRetentionL2Norm, 4)}, accumulated block write norm = ${fmtFixed(d3.accumulatedBlockWriteL2Norm, 4)}, and sequential IEEE-754 Float32 rounding residual ε_f32 = ${d3.roundingResidualMaxAbs.toExponential(2)}.`,
          `Across all 16 heterogeneous Mamba heads, cumulative decay spans ${minHead.headLabel} (${fmtFixed(minHead.cumulativeDecay, 4)}) to ${maxHead.headLabel} (${fmtFixed(maxHead.cumulativeDecay, 4)}). Epistemic guardrail: per-head decay dA and Frobenius/L2 state norms describe raw algebraic magnitudes — they do NOT measure retained semantic information or token importance.`,
        ],
        takeaways: [
          `Exact decomposition at Bay 42 H00 p=3: ‖A_cum·H_init‖₂=${fmtFixed(d3.inheritedRetentionL2Norm, 4)} vs ‖ΔH_write‖₂=${fmtFixed(d3.accumulatedBlockWriteL2Norm, 4)} (ε_f32=${d3.roundingResidualMaxAbs.toExponential(2)})`,
          `16 heterogeneous heads span Π dA ∈ [${fmtFixed(minHead.cumulativeDecay, 3)}, ${fmtFixed(maxHead.cumulativeDecay, 3)}]; norms are strictly algebraic, never semantic scores`,
        ],
        liveMetrics: [
          metric('H00 Cumulative Π dA', fmtFixed(d3.cumulativeDecay, 4), 'denoiser'),
          metric('‖A_cum·H_init‖₂', fmtFixed(d3.inheritedRetentionL2Norm, 4), 'context'),
          metric('‖ΔH_block_write‖₂', fmtFixed(d3.accumulatedBlockWriteL2Norm, 4), 'gold'),
          metric('FP32 Residual ε_f32', d3.roundingResidualMaxAbs.toExponential(2), 'shared'),
          metric('16-Head Π dA Range', `${minHead.headLabel} ${fmtFixed(minHead.cumulativeDecay, 3)} .. ${maxHead.headLabel} ${fmtFixed(maxHead.cumulativeDecay, 3)}`, 'muted'),
        ],
      };
    },
  },
  {
    beatIndex: 14,
    beatId: '2.9',
    actIndex: 1,
    title: 'Mamba Readout, 8-Group RMSNorm & Inner SiLU(z) vs. Outer Residual Gate',
    subtitle: '16 heads · 8 groups · 4×4 SSM · Gate-first norm vs outer diffusion residual gate',
    badge: 'Bay 01 · Inner vs Outer Gate',
    viewMode: 'paired_mamba',
    bayId: 1,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'mamba_ssm',
    blockPos: 0,
    opStep: 5,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'den-rms-gate',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Contrast inner SiLU(z) [64] + 8-group rms_gate against outer adaLN gate [8]',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const pass = record.denoiserPass!;
      const m1 = pass.getMambaTrace(1);
      const d0 = extractMambaFullBlockPositionState(m1, 0, 0, { bayId: 1 });
      const mod1 = pass.getLayerModulation(1)!;
      return {
        paragraphs: [
          `After updating all ${m1.nHeads} heads' ${m1.dHead}×${m1.dState} FP32 SSM matrices (${m1.nGroups} B/C groups), Mamba-2 contracts H_p with group vector C, adds per-head skip D·x (D[H00] = ${fmtFixed(d0.skipD, 4)}), and executes Nemotron's gate-first order (norm_before_gate = false): multiply by inner SiLU(z) across d_inner = ${m1.dInner} channels FIRST, then divide by the per-group RMS denominator rms_gate[g=${d0.groupIndex}] = ${fmtFixed(d0.groupRmsDenominator, 4)}.`,
          `Only AFTER W_out projects back to hidden width d = ${m1.d} does the SECOND, completely distinct gate act: TwoTower's outer diffusion adaLN residual gate (g ⊙ Mamba_out, width d = ${m1.d} mini / 2,688 prod, ‖g‖₂ = ${fmtFixed(l2Norm(mod1.gate), 4)}) scales the layer delta before summing onto untouched residual X.`,
        ],
        takeaways: [
          `Inner Mamba gate: y · SiLU(z) (width ${m1.dInner} mini / 4,096 prod) BEFORE 8-group RMSNorm (norm_before_gate = false)`,
          `Outer diffusion gate: channel-wise adaLN gate g ⊙ W_out(...) (width ${m1.d} mini / 2,688 prod) added onto untouched X`,
        ],
        liveMetrics: [
          metric('Mamba Bank Geometry', `${m1.nHeads} heads · ${m1.nGroups} groups · ${m1.dHead}×${m1.dState} SSM`, 'shared'),
          metric('Group 0 rms_gate', fmtFixed(d0.groupRmsDenominator, 4), 'denoiser'),
          metric('Skip D[H00] / ‖gated‖₂', `${fmtFixed(d0.skipD, 3)} / ${fmtFixed(l2Norm(d0.gatedActivation), 4)}`, 'denoiser'),
          metric('Inner vs Outer Width', `SiLU(z) ${m1.dInner} vs adaLN gate ${m1.d}`, 'gold'),
          metric('Outer adaLN ‖gate‖₂', fmtFixed(l2Norm(mod1.gate), 4), 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 15,
    beatId: '2.10',
    actIndex: 1,
    title: 'Stateless MoE (128+1): FP64 Cutoff Margin & Running Partial Denominator',
    subtitle: '0 B cache · Post-norm adaLN · 6th-vs-7th near-tie margin in FP64 router',
    badge: 'Bay 02 · FP64 6th vs 7th',
    viewMode: 'moe',
    bayId: 2,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'all',
    blockPos: 3,
    opStep: 2,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'w-router',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Inspect the 6th-vs-7th cutoff boundary and step running 6-term normalization',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const pass = record.denoiserPass!;
      const moe2 = pass.getMoeTrace(2);
      const margin = computeMoeSelectionMargin(moe2, 3);
      const r6Tag = formatExpertId(margin.rank6ExpertId);
      const r7Tag = formatExpertId(margin.rank7ExpertId);
      return {
        paragraphs: [
          `Granular MoE Bay 02 (128 routed experts + 1 double-wide shared expert, 0 B sequence cache) applies RMSNorm-then-modulate (norm_then_modulate) before routing. Unlike Nano's FP32 router, TwoTower executes router matmul + sigmoid in FP64 (${margin.routerPrecisionPolicy}) because biased choice scores σ + e_score_correction_bias frequently exhibit narrow 6th-vs-7th cutoff margins.`,
          `At Bay 02, block position p = 3 (abs t${margin.absoluteTokenPosition}), rank-6 winner ${r6Tag} (${fmtFixed(margin.rank6ChoiceScore, 5)}) clears rank-7 runner-up ${r7Tag} (${fmtFixed(margin.rank7ChoiceScore, 5)}) by a margin of just Δ = ${fmtFixed(margin.choiceScoreMargin, 6)}${margin.isTightMargin ? ' (TIGHT)' : ''}. Once Top-6 are chosen, bias is dropped and unbiased sigmoid scores normalize term-by-term across a running 6-term partial denominator (× 2.5) alongside the unscaled 2× Shared Expert.`,
        ],
        takeaways: [
          `Live FP64 cutoff margin @ Bay 02 p=3: ${r6Tag} vs ${r7Tag}, Δ = ${fmtFixed(margin.choiceScoreMargin, 6)} (${margin.isTightMargin ? 'tight near-tie' : 'separated'})`,
          `Bias affects Top-6 selection ONLY; unbiased σ normalizes over the running 6-term denominator × 2.5 · Strictly 0 B cache`,
        ],
        liveMetrics: [
          metric('6th Winner (Selected)', `${r6Tag} (${fmtFixed(margin.rank6ChoiceScore, 5)})`, 'denoiser'),
          metric('7th Runner-Up (Idle)', `${r7Tag} (${fmtFixed(margin.rank7ChoiceScore, 5)})`, 'muted'),
          metric('FP64 Cutoff Margin Δ', `${fmtFixed(margin.choiceScoreMargin, 6)}${margin.isTightMargin ? ' · TIGHT' : ''}`, margin.isTightMargin ? 'warn' : 'gold'),
          metric('Router Precision', margin.routerPrecisionPolicy, 'shared'),
          metric('MoE Cache / Scaling', '0 B stateless · Top-6 ×2.5 + 2× Shared', 'context'),
        ],
      };
    },
  },

  // ==========================================================================
  // ACT III (`act-3`, 7 beats): Denoiser Output, MDLM Sampler & Iterative Refinement
  // ==========================================================================
  {
    beatIndex: 16,
    beatId: '3.1',
    actIndex: 2,
    title: 'Unshifted Denoiser Block Output Logits (V = 34, [MASK] #32, <eos> #33)',
    subtitle: 'Unmodulated norm_f → Untied W_lm_head_den · Row j predicts clean x₀[P + j]',
    badge: 'Unshifted [4 × 34] · norm_f Raw',
    viewMode: 'output',
    bayId: 53,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'all',
    blockPos: 2,
    opStep: 3,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'token-strip',
    epistemicTag: EPISTEMIC_MINIATURE_SIMULATION,
    interactivePrompt: 'Inspect unshifted row p=2 (t10) logits before MDLM constraints apply',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref0Idx }),
    buildNarrativeAndMetrics: ({ session, record }) => {
      const pass = record.denoiserPass!;
      const V = session.model.vocab.length;
      const focusPos = Math.min(2, pass.blockSize - 1);
      const top1 = pass.topKByPosition[focusPos][0];
      const rawMaskProb = pass.probs[focusPos * V + session.model.maskTokenId];
      return {
        paragraphs: [
          `Exiting Bay 52, Denoiser passes residual stream Y_52 through strictly UNMODULATED Final RMSNorm (norm_f, ε = 1e−5) and independent untied W_lm_head_den to emit unshifted block logits of shape [${pass.blockSize} × ${V}] (miniature V = ${V} with [MASK] #${session.model.maskTokenId} and <eos> #${session.model.eosTokenId}; production 16 × 131,072).`,
          `Zero shift-by-1 occurs here: unlike Context AR where position t predicts token t + 1, Denoiser row j directly predicts the clean token x₀[P + j] at that exact slot. At slot p = ${focusPos} (abs t${pass.absoluteTokenStart + focusPos}), raw unconstrained top-1 is #${top1.tokenId} "${top1.label}" (logit ${fmtFixed(top1.logit, 3)}, raw p = ${fmtPct(top1.prob)}), while raw [MASK] still holds ${fmtPct(rawMaskProb, 2)} probability.`,
        ],
        takeaways: [
          `Unshifted block semantics: logits row j predicts clean token x₀[P + j] directly (no causal shift-by-1)`,
          `norm_f is strictly unmodulated in both towers; W_lm_head_den is untied from W_embed_den and independent of Context`,
        ],
        liveMetrics: [
          metric('Output Semantics', 'Unshifted x₀[P + j]', 'denoiser'),
          metric('Vocab / Special IDs', `V=${V} · [MASK]=#${session.model.maskTokenId} · <eos>=#${session.model.eosTokenId}`, 'shared'),
          metric(`Slot p=${focusPos} Raw Top-1`, `#${top1.tokenId} "${top1.label}" (${fmtPct(top1.prob)})`, 'gold'),
          metric(`Slot p=${focusPos} Raw [MASK]`, fmtPct(rawMaskProb, 2), 'warn'),
          metric('Final Norm norm_f', 'Strictly UNMODULATED', 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 17,
    beatId: '3.2',
    actIndex: 2,
    title: 'Raw LM-Head vs. Constrained MDLM Distribution (maskTokenProb = 0)',
    subtitle: 'SUBS constraint clamps [MASK] to −1e12 and locks carry-over deltas',
    badge: 'MDLM · P([MASK]) = 0',
    viewMode: 'output',
    bayId: 53,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'all',
    blockPos: 2,
    opStep: 4,
    showSamplerPanel: true,
    showArchPanel: false,
    highlightPartId: 'token-strip',
    epistemicTag: EPISTEMIC_SAMPLER_CALIBRATION,
    interactivePrompt: 'Compare Raw LM Head vs Constrained MDLM columns in the Sampler Inspector',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Conf0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const confRec = record.kind === 'diffusion_confidence_commit' ? record : null;
      const pres = confRec!.samplerPresentation;
      const focusPos = Math.min(2, pres.blockSize - 1);
      const sl = pres.slots[focusPos];
      const topCon = sl.constrainedMdlm.topK[0];
      return {
        paragraphs: [
          `The MDLM / SUBS sampler enforces two strict probability constraints on the raw Denoiser logits before selecting candidates: (1) on every masked position, [MASK] is suppressed to −1e12 pre-softmax so constrained x_θ[[MASK]] === 0.0000 identically, and (2) on any already-unmasked slot, x_θ collapses to an exact carry-over delta (p = 1.0 on the locked token, 0.0 elsewhere).`,
          `Inspecting slot p = ${focusPos} (abs t${sl.absolutePosition ?? focusPos}) at refinement r = ${pres.stepIndex}: raw LM-head probability on [MASK] was ${fmtPct(sl.rawLmHead.maskTokenProb, 2)}, which MDLM zeroes to ${fmtPct(sl.constrainedMdlm.maskTokenProb, 2)}, elevating constrained candidate "${topCon.tokenLabel}" to unscaled p = ${fmtPct(topCon.prob, 1)} under ${sl.selection.samplingMode} (T = ${pres.temperature}).`,
        ],
        takeaways: [
          `Raw LM Head ≠ Constrained MDLM: [MASK] mass drops from ${fmtPct(sl.rawLmHead.maskTokenProb, 2)} → 0.00% before candidate selection`,
          `Greedy argmax (T = 0) or keyed deterministic Gumbel-max (T > 0) samples strictly from constrained non-mask tokens`,
        ],
        liveMetrics: [
          metric(`Slot p=${focusPos} Raw [MASK]`, fmtPct(sl.rawLmHead.maskTokenProb, 2), 'warn'),
          metric('Constrained [MASK] Prob', '0.00% (excluded)', 'shared'),
          metric(`Slot p=${focusPos} MDLM Top-1`, `"${topCon.tokenLabel}" (${fmtPct(topCon.prob, 1)})`, 'gold'),
          metric('Selection Mode', `${sl.selection.samplingMode} (T=${pres.temperature})`, 'denoiser'),
        ],
      };
    },
  },
  {
    beatIndex: 18,
    beatId: '3.3',
    actIndex: 2,
    title: 'Confidence Threshold γ & Unscaled Calibration Boundary',
    subtitle: 'γ = 0.22 mini default vs 0.80 paper / 0.90 code · Confidence ≠ semantic truth',
    badge: 'γ = 0.22 · Unscaled x_θ',
    viewMode: 'output',
    bayId: 53,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'all',
    blockPos: 2,
    opStep: 5,
    showSamplerPanel: true,
    showArchPanel: false,
    highlightPartId: 'token-strip',
    epistemicTag: EPISTEMIC_SAMPLER_CALIBRATION,
    interactivePrompt: 'Check which slots strictly exceed γ = 22.0% in the Sampler Inspector',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Conf0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const confRec = record.kind === 'diffusion_confidence_commit' ? record : null;
      const pres = confRec!.samplerPresentation;
      const u = confRec!.unmaskStep;
      const presetSummary = TWOTOWER_CONFIDENCE_PRESETS.map(
        (p) => `${p.shortLabel}`,
      ).join(' · ');
      const slotConfs = pres.slots
        .map((s) => `p${s.slotIndex}:${s.confidenceDisplay.displayText}`)
        .join(' | ');
      return {
        paragraphs: [
          `Confidence unmasking gathers each masked slot's score strictly from the UNSCALED constrained distribution x_θ[j, y_j] (even when temperature T > 0) and applies a strict > comparison against γ. This miniature simulator uses γ = ${fmtFixed(pres.confidenceThreshold, 2)} so multi-step unmasking is clearly visible, contrasted explicitly against the published paper default (0.80) and official reference code default (0.90).`,
          `At Block 0 refinement r = ${pres.stepIndex} (t = ${fmtFixed(pres.tModel, 3)}), ${u.numAboveThreshold} of ${u.nMaskedBefore} masked slots strictly exceed γ = ${fmtPct(pres.confidenceThreshold, 0)} (${slotConfs}), committing [${u.thresholdCommittedIndices.map((i) => `p${i}`).join(', ') || 'none'}] while remasking [${u.remaskedIndices.map((i) => `p${i}`).join(', ') || 'none'}]. Crucially, unscaled softmax probability is a relative selection heuristic — NOT calibrated semantic correctness.`,
        ],
        takeaways: [
          `Threshold presets: ${presetSummary} · Strict ">" comparison on unscaled x_θ`,
          `Epistemic rule: high unscaled softmax confidence never proves factual or semantic correctness`,
        ],
        liveMetrics: [
          metric('Active Threshold γ', `${fmtFixed(pres.confidenceThreshold, 2)} (Paper 0.80 / Code 0.90)`, 'gold'),
          metric('Strictly Above γ', `${u.numAboveThreshold} / ${u.nMaskedBefore} slots`, 'denoiser'),
          metric('Unmask Transition', `nMasked: ${u.nMaskedBefore} → ${u.nMaskedAfter}`, 'shared'),
          metric('Slot Confidences', slotConfs, 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 19,
    beatId: '3.4',
    actIndex: 2,
    title: 'Remaining-Budget Commitment Floor (ceil(nMasked / remainingSteps))',
    subtitle: 'Guaranteed monotonic completion within K steps even if all slots fall below γ',
    badge: 'Floor = ceil(nM / remK)',
    viewMode: 'output',
    bayId: 53,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: 5,
    showSamplerPanel: true,
    showArchPanel: false,
    highlightPartId: 'token-strip',
    epistemicTag: EPISTEMIC_SAMPLER_CALIBRATION,
    interactivePrompt: 'Compare threshold-qualified count vs minCommitFloor in the Sampler Inspector',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Conf0Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const confRec = record.kind === 'diffusion_confidence_commit' ? record : null;
      const pres = confRec!.samplerPresentation;
      const u = confRec!.unmaskStep;
      return {
        paragraphs: [
          `Because stepsPerBlock (K = ${pres.stepsPerBlock}) is configurable independently of blockSize (B = ${pres.blockSize}), threshold-only unmasking could stall or fail to finish within K evaluations. TwoTower prevents stalls with a dual floor: minCommitFloor = ceil(nMasked / remainingSteps) and tokensToCommit = min(nMasked, max(numAboveThreshold, 1, minCommitFloor)) — committing all remaining masks on the final step.`,
          `On Block 0 step r = ${pres.stepIndex}, remainingSteps = ${pres.remainingSteps}, giving minCommitFloor = ceil(${u.nMaskedBefore} / ${pres.remainingSteps}) = ${u.minCommitFloor}. Because ${u.numAboveThreshold} slots qualified above γ, threshold commitment governs here (commitReason: "${u.commitReason}", thresholdCommitted = ${u.thresholdCommittedIndices.length}, floorCommitted = ${u.floorCommittedIndices.length}) — whereas whenever numAboveThreshold < max(1, minCommitFloor), the floor deterministically commits the highest-confidence masked slots.`,
        ],
        takeaways: [
          `Formula: minCommitFloor = ceil(${u.nMaskedBefore} / ${pres.remainingSteps}) = ${u.minCommitFloor} · tokensToCommit = ${u.tokensCommittedCount}`,
          `Separating threshold-qualified (${u.thresholdCommittedIndices.length}) from floor-committed (${u.floorCommittedIndices.length}) makes budget pressure transparent`,
        ],
        liveMetrics: [
          metric('Remaining Budget', `remK = ${pres.remainingSteps} / K=${pres.stepsPerBlock}`, 'shared'),
          metric('Budget Floor', `ceil(${u.nMaskedBefore}/${pres.remainingSteps}) = ${u.minCommitFloor}`, 'gold'),
          metric('Threshold vs Floor', `${u.thresholdCommittedIndices.length} threshold · ${u.floorCommittedIndices.length} floor`, 'denoiser'),
          metric('Commit Reason', u.commitReason, 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 20,
    beatId: '3.5',
    actIndex: 2,
    title: 'Successive Refinement r = 1: Live t = nMasked / B Jumps Directly',
    subtitle: '2 slots unmasked at r=0 → t = 2/4 = 0.500 (proving t ≠ r/K) · Same P=8 Context seed',
    badge: 'r = 1 · t = 0.500 ≠ r/K',
    viewMode: 'timestep',
    bayId: 6,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'timestep',
    blockPos: 0,
    opStep: 0,
    showSamplerPanel: true,
    showArchPanel: false,
    highlightPartId: 't-scalar',
    epistemicTag: EPISTEMIC_MINIATURE_SIMULATION,
    interactivePrompt: 'Verify executed t = nMasked / B differs from r / K while Context digest stays identical',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Ref1Idx }),
    buildNarrativeAndMetrics: ({ session, record, landmarks }) => {
      const ref1 = record.kind === 'diffusion_refinement' ? record : null;
      const ref0 = getBranchRecord(session, landmarks.mainBranchId, landmarks.block0Ref0Idx);
      const rIdx = ref1?.refinementIndex ?? 1;
      const B = ref1?.blockConfig.blockSize ?? 4;
      const K = ref1?.blockConfig.stepsPerBlock ?? 4;
      const nM = record.activeBlock?.nMasked ?? 2;
      const tActual = ref1?.tModel ?? nM / B;
      const rOverK = rIdx / K;
      const sameDigest = record.identity.sourceCacheDigest === ref0.identity.sourceCacheDigest;
      const tEmbL2 = ref1?.denoiserPass.tEmb ? l2Norm(ref1.denoiserPass.tEmb) : 0;
      return {
        paragraphs: [
          `Entering refinement r = ${rIdx}, the active block input is [${ref1?.xtInputLabels.join(', ') ?? ''}] — ${nM} masked slots remain out of B = ${B}. Consequently, Denoiser executes with t = nMasked / B = ${nM} / ${B} = ${fmtFixed(tActual, 3)}, directly proving that t tracks remaining masks rather than step progress (r / K = ${fmtFixed(rOverK, 3)}, 1 − r / K = ${fmtFixed(1 - rOverK, 3)}).`,
          `Notice the cache provenance: sourceCacheDigest (${fmtShortDigest(record.identity.sourceCacheDigest)}) is 100% IDENTICAL to r = 0 (sameDigest = ${String(sameDigest)}). Denoiser discards r = 0's ephemeral scratchpad and freshly conditions on Context's immutable P = ${record.committedPrefixLen} boundary with updated block tokens and new tEmb (L2 = ${fmtFixed(tEmbL2, 3)}).`,
        ],
        takeaways: [
          `Live proof: executed t = ${fmtFixed(tActual, 3)} (${nM}/${B}) ≠ r/K (${fmtFixed(rOverK, 3)}) and ≠ 1−r/K (${fmtFixed(1 - rOverK, 3)})`,
          `Every refinement re-seeds from the exact SAME P = ${record.committedPrefixLen} Context digest (${fmtShortDigest(record.identity.sourceCacheDigest)})`,
        ],
        liveMetrics: [
          metric('Executed t = nM/B', `${fmtFixed(tActual, 3)} (${nM}/${B})`, 'gold'),
          metric('Step Index r/K (NOT t)', `r=${rIdx}, r/K=${fmtFixed(rOverK, 3)}`, 'warn'),
          metric('Updated tEmb [24] L2', fmtFixed(tEmbL2, 4), 'denoiser'),
          metric('Context Seed Unchanged', `${fmtShortDigest(record.identity.sourceCacheDigest)} (match=${String(sameDigest)})`, 'context'),
        ],
      };
    },
  },
  {
    beatIndex: 21,
    beatId: '3.6',
    actIndex: 2,
    title: 'Locked Prior Positions & Monotonic Unmasking',
    subtitle: 'Prior locked slots display "Locked (prior step)" · Never overwritten',
    badge: 'r = 1 Commit · Finite Locks',
    viewMode: 'output',
    bayId: 53,
    towerFocus: 'denoiser_solo',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: 5,
    showSamplerPanel: true,
    showArchPanel: false,
    highlightPartId: 'token-strip',
    epistemicTag: EPISTEMIC_SAMPLER_CALIBRATION,
    interactivePrompt: 'Verify locked slots display "Locked (prior step)" with strictly finite presentation',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0Conf1Idx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const confRec = record.kind === 'diffusion_confidence_commit' ? record : null;
      const pres = confRec!.samplerPresentation;
      const u = confRec!.unmaskStep;
      const lockedSlots = pres.slots.filter((s) => s.decision === 'locked_prior');
      const newlyCommitted = pres.slots.filter(
        (s) => s.decision !== 'locked_prior' && s.decision !== 'remasked',
      );
      return {
        paragraphs: [
          `On refinement r = ${pres.stepIndex}'s confidence commitment, the ${lockedSlots.length} positions unmasked on earlier steps (${lockedSlots.map((s) => `p${s.slotIndex} "${s.tokenLabelAfter}"`).join(', ') || 'none'}) remain strictly locked carry-overs — their internal sampler sentinel NEVER leaks to presentation, surfacing cleanly as "Locked (prior step)" with strictly finite values across every field.`,
          `Among the ${u.nMaskedBefore} remaining masked slots (executed t = ${fmtFixed(pres.tModel, 3)}, remainingSteps = ${pres.remainingSteps}, floor = ${u.minCommitFloor}), ${u.thresholdCommittedIndices.length} qualified via threshold and ${u.floorCommittedIndices.length} via floor (${newlyCommitted.map((s) => formatSamplerSlotSummary(pres, s.slotIndex)).join('; ')}), reducing unresolved masks ${u.nMaskedBefore} → ${u.nMaskedAfter}.`,
        ],
        takeaways: [
          `Monotonic unmasking: locked prior slots (${lockedSlots.length}) are never re-masked or overwritten, displaying "Locked (prior step)"`,
          `Active block progress: nMasked ${u.nMaskedBefore} → ${u.nMaskedAfter} (${u.thresholdCommittedIndices.length} threshold · ${u.floorCommittedIndices.length} floor)`,
        ],
        liveMetrics: [
          metric('Executed t / Unresolved', `t=${fmtFixed(pres.tModel, 3)} · nM: ${u.nMaskedBefore}→${u.nMaskedAfter}`, 'gold'),
          metric('Prior Locked Slots', `${lockedSlots.length} · "Locked (prior step)"`, 'context'),
          metric('Threshold vs Floor', `${u.thresholdCommittedIndices.length} thresh · ${u.floorCommittedIndices.length} floor (${u.commitReason})`, 'denoiser'),
          metric('Current Block Tokens', record.activeBlock!.currentTokenLabels.join(' · '), 'shared'),
        ],
      };
    },
  },
  {
    beatIndex: 22,
    beatId: '3.7',
    actIndex: 2,
    title: 'Completed-But-Uncommitted Block State (nMasked = 0, Context Still P = 8)',
    subtitle: 'Early completion in 3 NFEs · Block ready on Denoiser · Context NOT yet updated',
    badge: 'completed_uncommitted · P = 8',
    viewMode: 'macro',
    bayId: 53,
    towerFocus: 'paired',
    linkerChannel: 'all',
    blockPos: 1,
    opStep: null,
    showSamplerPanel: true,
    showArchPanel: false,
    highlightPartId: 'pathway-block-commit',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Observe completed_uncommitted state: nMasked=0 while Context stays at P=8',
    resolveTarget: (lm) => ({
      branchId: lm.mainBranchId,
      recordIndex: lm.block0CompletedUncommittedIdx,
    }),
    buildNarrativeAndMetrics: ({ record }) => {
      const blk = record.activeBlock!;
      const confRec = record.kind === 'diffusion_confidence_commit' ? record : null;
      const lastT = confRec?.unmaskStep.tModel ?? 0.25;
      return {
        paragraphs: [
          `After the final refinement (r = ${confRec?.refinementIndex ?? 2}, executed t = ${fmtFixed(lastT, 3)}), all B = ${blk.blockConfig.blockSize} tokens in Block ${blk.blockId} are fully unmasked: [${blk.currentTokenLabels.join(', ')}] (nMasked = ${blk.nMasked}, NFE = ${blk.nfe}/${blk.blockConfig.stepsPerBlock}${blk.earlyCompleted ? ', early-completed!' : ''}).`,
          `Notice the explicit architectural separation: lifecyclePhase is "${record.lifecyclePhase}" and contextCommitted is ${String(blk.contextCommitted)}. Context Tower STILL holds only the original P = ${record.committedPrefixLen} prompt prefix ([0..${record.committedPrefixLen - 1}]), and the emerald Block Commit pathway is staged until Context explicitly ingests the finished block.`,
        ],
        takeaways: [
          `Distinct state: nMasked = ${blk.nMasked} AND contextCommitted = false — Context prefix stays at P = ${record.committedPrefixLen} until explicit commit`,
          `Early completion saved ${blk.blockConfig.stepsPerBlock - blk.nfe} NFE (${blk.nfe} passes used of K = ${blk.blockConfig.stepsPerBlock} budget)`,
        ],
        liveMetrics: [
          metric('Lifecycle Phase', record.lifecyclePhase, 'gold'),
          metric('Unresolved / Committed', `nMasked=${blk.nMasked} · contextCommitted=${String(blk.contextCommitted)}`, 'warn'),
          metric('Context Prefix P', `${record.committedPrefixLen} (NOT yet 12)`, 'context'),
          metric('Block 0 NFE', `${blk.nfe} / ${blk.blockConfig.stepsPerBlock}${blk.earlyCompleted ? ' (early)' : ''}`, 'denoiser'),
          metric('Completed Block Tokens', blk.currentTokenLabels.join(' · '), 'shared'),
        ],
      };
    },
  },

  // ==========================================================================
  // ACT IV (`act-4`, 5 beats): Context Commitment, Next Block & Production Architecture
  // ==========================================================================
  {
    beatIndex: 23,
    beatId: '4.1',
    actIndex: 3,
    title: 'Explicit Context Block Commitment (P = 8 → 12, Split mamba_s2 + Final Token)',
    subtitle: 'Single causal pass ingests finalized block · New Context cache digest',
    badge: 'Commit Blk 0 · P = 8 → 12',
    viewMode: 'macro',
    bayId: 6,
    towerFocus: 'paired',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'pathway-block-commit',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Follow the emerald Commit Pathway advancing Context KV & Mamba caches to P = 12',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block0CommitIdx }),
    buildNarrativeAndMetrics: ({ record }) => {
      const commitRec = record.kind === 'diffusion_block_commit' ? record : null;
      const exec = record.contextExecution!;
      const seg0 = exec.segments[0];
      const seg1 = exec.segments[exec.segments.length - 1];
      return {
        paragraphs: [
          `Committing Block ${commitRec?.blockConfig.blockId ?? 0} routes the ${commitRec?.committedBlockTokens.length ?? 4} finalized tokens [${commitRec?.committedBlockLabels.join(', ') ?? ''}] along the emerald return pathway into Context Tower's Embedding for ONE causal forward pass — updating all 6 GQA KV tracks and all 23 Mamba-2 Conv1D/SSM registers from P = 8 → ${record.committedPrefixLen}.`,
          `Just like prefill, this multi-token Context commit executes in ${exec.segments.length} split segments — prefix [${seg0.tokenRange[0]}..${seg0.tokenRange[1]}] (${seg0.segmentRole}) to maintain an exact mamba_s2 checkpoint, followed by final token [${seg1.tokenRange[0]}] (${seg1.segmentRole}) — transitioning the Context cache digest from ${fmtShortDigest(record.identity.sourceCacheDigest)} → ${fmtShortDigest(record.identity.resultCacheDigest)}.`,
        ],
        takeaways: [
          `Context prefix advances cleanly P = 8 → ${record.committedPrefixLen} via ${exec.segments.length} split segments ([${seg0.tokenRange[0]}..${seg0.tokenRange[1]}] + [${seg1.tokenRange[0]}])`,
          `Post-block EOS check runs ONLY after full block commit (stoppedByEos = ${String(commitRec?.stoppedByEos ?? false)})`,
        ],
        liveMetrics: [
          metric('Context Prefix Advance', `P = 8 → ${record.committedPrefixLen} / ${record.kvCapacity}`, 'context'),
          metric('Split Commit Segments', `${exec.segments.length} ([${seg0.tokenRange[0]}..${seg0.tokenRange[1]}] + [${seg1.tokenRange[0]}])`, 'shared'),
          metric('Cache Digest Change', `${fmtShortDigest(record.identity.sourceCacheDigest)} → ${fmtShortDigest(record.identity.resultCacheDigest)}`, 'gold'),
          metric('Committed Block', commitRec?.committedBlockLabels.join(' · ') ?? '', 'denoiser'),
          metric('Post-Block EOS Check', `stoppedByEos = ${String(commitRec?.stoppedByEos ?? false)}`, 'muted'),
        ],
      };
    },
  },
  {
    beatIndex: 24,
    beatId: '4.2',
    actIndex: 3,
    title: 'Next-Block Conditioning (Blk 1 on New P = 12 Boundary)',
    subtitle: 'Freshly conditioned on P = 12 Context digest · Zero conflation with prior P = 8 pass',
    badge: 'Blk 1 · [12..15] · 16 Keys',
    viewMode: 'paired_gqa',
    bayId: 6,
    towerFocus: 'paired',
    linkerChannel: 'gqa_kv',
    blockPos: 0,
    opStep: 2,
    showSamplerPanel: false,
    showArchPanel: false,
    highlightPartId: 'den-prefix-kv',
    epistemicTag: EPISTEMIC_MINIATURE_SIMULATION,
    interactivePrompt: 'Verify Paired GQA Bay 06 now bridges P = 12 inherited prefix keys + 4 new block keys',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block1Ref0Idx }),
    buildNarrativeAndMetrics: ({ session, record, landmarks }) => {
      const blk0Ref = getBranchRecord(session, landmarks.mainBranchId, landmarks.block0Ref0Idx);
      const pass = record.denoiserPass!;
      const gqa = pass.getGqaTrace(6);
      const distinctBoundary =
        record.identity.sourceCacheDigest !== blk0Ref.identity.sourceCacheDigest;
      return {
        paragraphs: [
          `Block ${record.identity.blockId ?? 1} ([${record.identity.absoluteTokenRange[0]}..${record.identity.absoluteTokenRange[1]}]) initializes on Context's NEW P = ${record.committedPrefixLen} boundary (sourceCacheDigest ${fmtShortDigest(record.identity.sourceCacheDigest)} ≠ Block 0's P = 8 digest ${fmtShortDigest(blk0Ref.identity.sourceCacheDigest)}). Every Denoiser tensor carries explicit blockId, refinementIndex, absoluteTokenRange, and sourceCacheDigest so prior P = 8 passes can NEVER be conflated with P = 12 passes.`,
          `In Paired GQA Bay 06, the read-only inherited prefix now spans ${gqa.prefixTokens} slots ([0..${gqa.prefixTokens - 1}], including all 4 newly committed Block 0 tokens) concatenated with Block 1's ${gqa.tokens} active [MASK] slots ([${gqa.prefixTokens}..${gqa.keyTokens - 1}]) for [${gqa.tokens} × ${gqa.keyTokens}] bidirectional attention.`,
        ],
        takeaways: [
          `Strict boundary provenance: Block 1 sourceCacheDigest (${fmtShortDigest(record.identity.sourceCacheDigest)}) ≠ Block 0 (${fmtShortDigest(blk0Ref.identity.sourceCacheDigest)})`,
          `Attended KV span grows cleanly 12 → ${gqa.keyTokens} (${gqa.prefixTokens} persistent Context + ${gqa.tokens} ephemeral Denoiser)`,
        ],
        liveMetrics: [
          metric('Active Block', `Blk ${record.identity.blockId ?? 1} · [${record.identity.absoluteTokenRange[0]}..${record.identity.absoluteTokenRange[1]}]`, 'denoiser'),
          metric('Attended KV Keys', `${gqa.prefixTokens} prefix + ${gqa.tokens} block = ${gqa.keyTokens}`, 'context'),
          metric('New Boundary Digest', `${fmtShortDigest(record.identity.sourceCacheDigest)} (≠ Blk0: ${String(distinctBoundary)})`, 'gold'),
          metric('Executed t = nM/B', `${fmtFixed(pass.tModel ?? 1, 3)} (r=${record.identity.refinementIndex ?? 0})`, 'shared'),
        ],
      };
    },
  },
  {
    beatIndex: 25,
    beatId: '4.3',
    actIndex: 3,
    title: 'Persistent Context Cache Growth vs. Transient Denoiser Scratchpad',
    subtitle: 'Two resident weight sets (117.69 GiB) ≠ Two persistent generation caches',
    badge: '1 Persistent Cache · 2 Weight Sets',
    viewMode: 'macro',
    bayId: 6,
    towerFocus: 'paired',
    linkerChannel: 'all',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'ctx-moe-body-02',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Compare solid Context cache growth vs dashed ephemeral Denoiser scratchpad',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block1Ref0Idx }),
    buildNarrativeAndMetrics: ({ session, record }) => {
      const prodP12B4 = twoTowerMemoryFor(record.committedPrefixLen, 4, 'mask_diffusion');
      const prodP12B16 = twoTowerMemoryFor(record.committedPrefixLen, 16, 'mask_diffusion');
      const miniMem = measureTwoTowerSessionRetainedBytes(session);
      return {
        paragraphs: [
          `A core architectural principle of Nemotron-Labs-TwoTower: hosting TWO independent resident parameter sets (${formatGiB(prodP12B16.dualTowerBf16WeightBytes)} BF16 = ${formatGiB(prodP12B16.contextTowerBf16WeightBytes)} Context + ${formatGiB(prodP12B16.denoiserTowerBf16WeightBytes)} Denoiser) does NOT create two persistent generation caches.`,
          `Only Context Tower maintains a growing persistent sequence cache (at P = ${record.committedPrefixLen}: ${formatMiB(prodP12B4.persistentContextMambaBytes)} fixed Mamba + ${formatKiB(prodP12B4.persistentContextKvBytes)} GQA KV). Denoiser Tower's dashed/frosted workspace holds strictly ephemeral per-step scratch (${formatKiB(prodP12B4.ephemeralDenoiserBlockKvBytes)} at B = 4 / ${formatKiB(prodP12B16.ephemeralDenoiserBlockKvBytes)} at prod B = 16) that is discarded immediately after each pass.`,
        ],
        takeaways: [
          `Two resident weight sets (${formatGiB(prodP12B16.dualTowerBf16WeightBytes)}) pair with ONE persistent prefix cache + tiny ephemeral block scratchpad`,
          `Tour session deduplicated retained memory (all ${miniMem.uniqueRecordCount} records across ${miniMem.branchCount} branches): ${formatBytesAdaptive(miniMem.totalRetainedBytes)}`,
        ],
        liveMetrics: [
          metric('Prod Resident Weights', `${formatGiB(prodP12B16.dualTowerBf16WeightBytes)} (2 towers)`, 'gold'),
          metric(`Prod Context Cache (P=${record.committedPrefixLen})`, `${formatMiB(prodP12B4.persistentContextMambaBytes)} + ${formatKiB(prodP12B4.persistentContextKvBytes)} KV`, 'context'),
          metric('Prod Denoiser Scratch', `${formatKiB(prodP12B16.ephemeralDenoiserBlockKvBytes)} @ B=16 (discarded)`, 'denoiser'),
          metric('Mini Tour Session RAM', `${formatBytesAdaptive(miniMem.totalRetainedBytes)} (${miniMem.uniqueRecordCount} recs)`, 'muted'),
        ],
      };
    },
  },
  {
    beatIndex: 26,
    beatId: '4.4',
    actIndex: 3,
    title: 'Production Precision (BF16 / FP32 / FP64), Scaling & Miniature Disclosure',
    subtitle: 'Official HF rev 35b64985 precision & 128K/256K scaling vs browser miniature math',
    badge: 'BF16 · FP32 · FP64 · 256K',
    viewMode: 'macro',
    bayId: 42,
    towerFocus: 'paired',
    linkerChannel: 'mamba_ssm',
    blockPos: 0,
    opStep: null,
    showSamplerPanel: false,
    showArchPanel: true,
    highlightPartId: 'timestep-hub',
    epistemicTag: EPISTEMIC_PRODUCTION_VS_MINIATURE,
    interactivePrompt: 'Review production precision boundaries and 128K / 256K Context scaling readouts',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block1Ref0Idx }),
    buildNarrativeAndMetrics: () => {
      const meta = TWOTOWER_SPEC.directlyInspectedMetadata;
      const mem128k = twoTowerMemoryFor(131_072, 16, 'mask_diffusion');
      const mem256k = twoTowerMemoryFor(meta.maxPositionEmbeddings, 16, 'mask_diffusion');
      const nvfp4 = TWOTOWER_SPEC.communityNvfp4Reference;
      return {
        paragraphs: [
          `The released Nemotron-Labs-TwoTower-30B-A3B-Base-BF16 checkpoint (HF rev ${TWOTOWER_SPEC.provenance.hfRevision.slice(0, 8)}, ${meta.safetensorsTensorCount.toLocaleString()} tensors across ${meta.safetensorsShardCount} shards) executes primary weights/activations in BF16, Mamba-2 SSM scans (chunk_size = ${meta.mambaChunkSize}) and all ${meta.fp32ScoreCorrectionBiasTensorCount} MoE e_score_correction_bias tensors (${meta.fp32ScoreCorrectionBiasParams.toLocaleString()} params) in FP32, and MoE router matmul + sigmoid in FP64. Because only 6 of 52 bays store KV caches, persistent Context cache stays ${formatBytesAdaptive(mem128k.persistentContextTotalBytes)} at 128K tokens (model card) and ${formatBytesAdaptive(mem256k.persistentContextTotalBytes)} at 256K tokens (config max_position_embeddings).`,
          `Full disclosure: every live tensor, softmax probability, and unmasked word shown in this browser runs deterministic synthetic miniature Float32/Float64 arithmetic (d = ${TWOTOWER_MINI_CONFIG.d}, V = ${TWOTOWER_MINI_CONFIG.vocabSize}, L = 52) to illuminate exact architectural mechanics — NEVER 63.19B checkpoint weights. Non-official community expert-only NVFP4 (~${nvfp4.approxTotalWeightGiB} GiB) compresses routed experts while keeping Mamba, GQA, routers, shared experts, bookends, and adaLN in BF16/FP32.`,
        ],
        takeaways: [
          `Official precision split: BF16 primary · FP32 Mamba SSM + 46 MoE biases · FP64 MoE router compute`,
          `Context cache scaling: ${formatBytesAdaptive(mem128k.persistentContextTotalBytes)} @ 128K · ${formatBytesAdaptive(mem256k.persistentContextTotalBytes)} @ 256K (+${formatKiB(mem256k.ephemeralDenoiserBlockKvBytes)} ephemeral B=16 KV)`,
        ],
        liveMetrics: [
          metric('Official Precision Split', 'BF16 / FP32 SSM+bias / FP64 router', 'gold'),
          metric('Prod Cache @ 128K (Card)', formatBytesAdaptive(mem128k.persistentContextTotalBytes), 'context'),
          metric('Prod Cache @ 256K (Config)', formatBytesAdaptive(mem256k.persistentContextTotalBytes), 'context'),
          metric('Community NVFP4 (Non-Off.)', `~${nvfp4.approxTotalWeightGiB} GiB (routed MoE only)`, 'muted'),
          metric('Browser Miniature Scale', `d=${TWOTOWER_MINI_CONFIG.d} · V=${TWOTOWER_MINI_CONFIG.vocabSize} · synthetic weights`, 'warn'),
        ],
      };
    },
  },
  {
    beatIndex: 27,
    beatId: '4.5',
    actIndex: 3,
    title: 'Architectural Trade-Offs Recap & Free Exploration',
    subtitle: '2× weights buy lossless single-cache persistence + parallel bidirectional block denoising',
    badge: 'Tour Complete · Free Explore',
    viewMode: 'macro',
    bayId: 6,
    towerFocus: 'paired',
    linkerChannel: 'all',
    blockPos: 1,
    opStep: null,
    showSamplerPanel: true,
    showArchPanel: true,
    highlightPartId: 'pathway-refinement-loop',
    epistemicTag: EPISTEMIC_CANONICAL_AND_MINIATURE,
    interactivePrompt: 'Exit Tour to return seamlessly to your untouched exploration session',
    resolveTarget: (lm) => ({ branchId: lm.mainBranchId, recordIndex: lm.block1RecapIdx }),
    buildNarrativeAndMetrics: ({ record, landmarks }) => {
      const confRec = record.kind === 'diffusion_confidence_commit' ? record : null;
      const u = confRec?.unmaskStep;
      const recapEvent = u
        ? `Blk 1 r=${u.stepIndex}: ${u.thresholdCommittedIndices.length} thresh · ${u.floorCommittedIndices.length} floor (${u.commitReason})`
        : `Blk ${record.identity.blockId ?? 1} active`;
      return {
        paragraphs: [
          `Nemotron-Labs-TwoTower trades doubled resident weights (63.19B BF16 across two 52-layer hybrid towers + 30.02M adaLN-single) to eliminate the core caching dilemma of hybrid diffusion LMs: Context Tower maintains ONE lossless, exact causal hybrid prefix cache (6 GQA KV + 23 fixed Mamba-2 + 23 stateless MoE), while Denoiser Tower performs bidirectional block refinement in ephemeral scratchpad memory over t = nMasked / B.`,
          `Across our continuous tour trajectory, Block 0 early-completed in ${landmarks.block0Nfe}/${TWOTOWER_TOUR_CONFIG.stepsPerBlock} NFEs before committing P = 8 → 12, and Block 1 demonstrates guaranteed floor commitment in action (${recapEvent}). Exit the walkthrough at any time to return to your completely untouched free-exploration session!`,
        ],
        takeaways: [
          `Key trade-off: 2× resident weights vs 1 persistent hybrid cache, 29 clean read-only transfer bays, and multi-token unmasking per NFE`,
          `Zero interference: this tour ran inside its own isolated TwoTowerSession — your personal exploration state was never mutated`,
        ],
        liveMetrics: [
          metric('Block 0 Efficiency', `${landmarks.block0Nfe}/${TWOTOWER_TOUR_CONFIG.stepsPerBlock} NFEs${landmarks.block0EarlyCompleted ? ' (early complete)' : ''}`, 'denoiser'),
          metric('Block 1 Live Decision', recapEvent, 'gold'),
          metric('Cross-Tower Contract', '29 stateful linkers · 23 stateless MoE', 'shared'),
          metric('Cache vs Weights', '1 persistent cache · 2 weight sets (117.69 GiB)', 'context'),
        ],
      };
    },
  },
]);

// ============================================================================
// SNAPSHOT BUILDER & CONTROLLER FACTORY
// ============================================================================

function validateBeatCatalogInvariants(): void {
  if (BEAT_DEFINITIONS.length !== TWOTOWER_TOUR_BEAT_COUNT) {
    throw new RangeError(
      `Expected ${TWOTOWER_TOUR_BEAT_COUNT} tour beats, found ${BEAT_DEFINITIONS.length}`,
    );
  }
  for (let i = 0; i < BEAT_DEFINITIONS.length; i++) {
    const def = BEAT_DEFINITIONS[i];
    if (def.beatIndex !== i) {
      throw new RangeError(`Beat ${def.beatId} has mismatched beatIndex ${def.beatIndex} !== ${i}`);
    }
    // Validate bayId against viewMode
    if (def.viewMode === 'paired_gqa' && getTwoTowerBayKind(def.bayId) !== '*') {
      throw new RangeError(`Beat ${def.beatId} paired_gqa bayId ${def.bayId} is not GQA ('*')`);
    }
    if (def.viewMode === 'paired_mamba' && getTwoTowerBayKind(def.bayId) !== 'M') {
      throw new RangeError(`Beat ${def.beatId} paired_mamba bayId ${def.bayId} is not Mamba ('M')`);
    }
    if (def.viewMode === 'moe' && getTwoTowerBayKind(def.bayId) !== 'E') {
      throw new RangeError(`Beat ${def.beatId} moe bayId ${def.bayId} is not MoE ('E')`);
    }
    if (def.viewMode === 'timestep' && (def.bayId < 1 || def.bayId > 52)) {
      throw new RangeError(`Beat ${def.beatId} timestep bayId ${def.bayId} out of [1..52]`);
    }
    if (def.viewMode === 'embedding' && def.bayId !== 0) {
      throw new RangeError(`Beat ${def.beatId} embedding bayId ${def.bayId} must be 0`);
    }
    if (def.viewMode === 'output' && def.bayId !== 53) {
      throw new RangeError(`Beat ${def.beatId} output bayId ${def.bayId} must be 53`);
    }
  }
}

validateBeatCatalogInvariants();

function buildBeatSnapshot(params: {
  readonly session: TwoTowerSession;
  readonly landmarks: TourTrajectoryLandmarks;
  readonly beatIndex: number;
}): TwoTowerTourBeatSnapshot {
  const { session, landmarks, beatIndex } = params;
  const def = BEAT_DEFINITIONS[beatIndex];
  const act = TWOTOWER_TOUR_ACTS[def.actIndex];
  const target = def.resolveTarget(landmarks);
  const record = getBranchRecord(session, target.branchId, target.recordIndex);

  const { paragraphs, takeaways, liveMetrics } = def.buildNarrativeAndMetrics({
    session,
    record,
    landmarks,
  });

  const provenance = resolveRecordProvenance(record);

  const beatSnapshot: TwoTowerTourBeatSnapshot = Object.freeze({
    beatIndex: def.beatIndex,
    beatId: def.beatId,
    actIndex: act.actIndex,
    actId: act.id,
    actTitle: act.title,
    actBeatIndex: def.beatIndex - act.startBeatIndex,
    actBeatCount: act.beatCount,
    title: def.title,
    subtitle: def.subtitle,
    badge: def.badge,
    paragraphs: Object.freeze([...paragraphs]),
    takeaways: Object.freeze([...takeaways]),
    interactivePrompt: def.interactivePrompt,
    viewMode: def.viewMode,
    bayId: def.bayId,
    towerFocus: def.towerFocus,
    linkerChannel: def.linkerChannel,
    blockPos: def.blockPos,
    opStep: def.opStep,
    showSamplerPanel: def.showSamplerPanel,
    showArchPanel: def.showArchPanel,
    highlightPartId: def.highlightPartId,
    liveMetrics: Object.freeze([...liveMetrics]),
    provenance,
    epistemicTag: def.epistemicTag,
  });

  assertPresentationSafeNoInfinity(beatSnapshot, `tourBeatSnapshot[${def.beatId}]`);
  return beatSnapshot;
}

/**
 * Create a dedicated, deterministic, strictly isolated 4-act / 28-beat guided educational
 * walkthrough controller owning its own private `TwoTowerSession` (never mutating any
 * external user exploration session).
 */
export function createTwoTowerTourController(
  model: TwoTowerModelParameters = createDefaultTwoTowerModel(),
): TwoTowerTourController {
  const { baseSession, landmarks } = buildDeterministicTourTrajectory(model);

  let tourSession: TwoTowerSession = baseSession;
  let status: TwoTowerTourStatus = 'idle';
  let currentBeatIndex = 0;

  // Cache pure immutable beat snapshots per beatIndex (each navigates tourSession cleanly on demand)
  const beatSnapshotCache = new Map<number, TwoTowerTourBeatSnapshot>();

  const syncTourSessionToBeat = (beatIndex: number): TwoTowerTourBeatSnapshot => {
    const clamped = Math.max(0, Math.min(TWOTOWER_TOUR_BEAT_COUNT - 1, Math.trunc(beatIndex)));
    const def = BEAT_DEFINITIONS[clamped];
    const target = def.resolveTarget(landmarks);

    if (tourSession.activeBranchId !== target.branchId) {
      tourSession = switchTwoTowerSessionBranch(tourSession, target.branchId);
    }
    const activeBranch = getActiveBranch(tourSession);
    if (activeBranch.cursorIndex !== target.recordIndex) {
      tourSession = navigateTwoTowerSession(tourSession, target.recordIndex);
    }

    currentBeatIndex = clamped;
    let cached = beatSnapshotCache.get(clamped);
    if (!cached) {
      cached = buildBeatSnapshot({
        session: tourSession,
        landmarks,
        beatIndex: clamped,
      });
      beatSnapshotCache.set(clamped, cached);
    }
    return cached;
  };

  const makeSnapshot = (): TwoTowerTourSnapshot => {
    const beat = syncTourSessionToBeat(currentBeatIndex);
    const isActive = status === 'playing' || status === 'paused';
    const isPlaying = status === 'playing';
    const acts: TwoTowerTourActSummary[] = TWOTOWER_TOUR_ACTS.map((act) =>
      Object.freeze({
        ...act,
        isCurrent: act.actIndex === beat.actIndex,
      }),
    );

    const snapshot: TwoTowerTourSnapshot = Object.freeze({
      ...beat,
      status,
      isActive,
      isPlaying,
      totalBeats: TWOTOWER_TOUR_BEAT_COUNT,
      totalActs: TWOTOWER_TOUR_ACT_COUNT,
      canPrev: beat.beatIndex > 0,
      canNext: beat.beatIndex < TWOTOWER_TOUR_BEAT_COUNT - 1,
      progressFraction: (beat.beatIndex + 1) / TWOTOWER_TOUR_BEAT_COUNT,
      acts: Object.freeze(acts),
      currentBeat: beat,
    });

    assertPresentationSafeNoInfinity(snapshot, 'twoTowerTourSnapshot');
    return snapshot;
  };

  // Pre-align tourSession to beat 0
  syncTourSessionToBeat(0);

  return {
    start(): TwoTowerTourSnapshot {
      currentBeatIndex = 0;
      status = 'playing';
      return makeSnapshot();
    },

    pause(): TwoTowerTourSnapshot {
      if (status === 'playing') {
        status = 'paused';
      }
      return makeSnapshot();
    },

    resume(): TwoTowerTourSnapshot {
      if (status === 'paused' || status === 'idle' || status === 'exited') {
        status = 'playing';
      }
      return makeSnapshot();
    },

    togglePlay(): TwoTowerTourSnapshot {
      if (status === 'playing') {
        status = 'paused';
      } else if (status === 'paused') {
        status = 'playing';
      } else {
        currentBeatIndex = 0;
        status = 'playing';
      }
      return makeSnapshot();
    },

    next(): TwoTowerTourSnapshot {
      if (status === 'idle' || status === 'exited') {
        status = 'paused';
      }
      if (currentBeatIndex < TWOTOWER_TOUR_BEAT_COUNT - 1) {
        currentBeatIndex += 1;
      } else if (status === 'playing') {
        status = 'paused';
      }
      return makeSnapshot();
    },

    prev(): TwoTowerTourSnapshot {
      if (status === 'idle' || status === 'exited') {
        status = 'paused';
      }
      if (currentBeatIndex > 0) {
        currentBeatIndex -= 1;
      }
      return makeSnapshot();
    },

    goToBeat(index: number): TwoTowerTourSnapshot {
      if (!Number.isFinite(index)) {
        throw new RangeError(`Tour beat index must be finite, received ${index}`);
      }
      const clamped = Math.max(0, Math.min(TWOTOWER_TOUR_BEAT_COUNT - 1, Math.trunc(index)));
      currentBeatIndex = clamped;
      if (status === 'idle' || status === 'exited') {
        status = 'paused';
      }
      return makeSnapshot();
    },

    goToAct(actIndex: number): TwoTowerTourSnapshot {
      if (!Number.isFinite(actIndex)) {
        throw new RangeError(`Tour act index must be finite, received ${actIndex}`);
      }
      const clampedAct = Math.max(0, Math.min(TWOTOWER_TOUR_ACT_COUNT - 1, Math.trunc(actIndex)));
      currentBeatIndex = TWOTOWER_TOUR_ACTS[clampedAct].startBeatIndex;
      if (status === 'idle' || status === 'exited') {
        status = 'paused';
      }
      return makeSnapshot();
    },

    restart(): TwoTowerTourSnapshot {
      currentBeatIndex = 0;
      status = 'playing';
      return makeSnapshot();
    },

    exit(): TwoTowerTourSnapshot {
      status = 'exited';
      return makeSnapshot();
    },

    getSnapshot(): TwoTowerTourSnapshot {
      return makeSnapshot();
    },

    getSession(): TwoTowerSession {
      return tourSession;
    },

    getModel(): TwoTowerModelParameters {
      return model;
    },
  };
}

// ============================================================================
// PURE TOUR ISOLATION VERIFIER
// ============================================================================

function collectSessionOwnedArrayBuffers(session: TwoTowerSession): Set<ArrayBufferLike> {
  const buffers = new Set<ArrayBufferLike>();
  const add = (arr: ArrayBufferView | null | undefined) => {
    if (arr) buffers.add(arr.buffer);
  };

  for (let bi = 0; bi < session.branches.length; bi++) {
    const recs = session.branches[bi].records;
    for (let ri = 0; ri < recs.length; ri++) {
      const rec = recs[ri];
      const snap = rec.contextSnapshot;
      add(snap.ctxState.prefixTokenIds);
      for (const m of snap.ctxState.mambaStates.values()) {
        add(m.convState);
        add(m.ssmState);
      }
      for (const g of snap.ctxState.gqaCaches.values()) {
        add(g.K);
        add(g.V);
      }
      if (snap.mambaS2) {
        for (const m of snap.mambaS2.values()) {
          add(m.convState);
          add(m.ssmState);
        }
      }
      add(snap.arLogits);
      add(snap.arProbs);

      if (rec.conditioning) {
        for (const m of rec.conditioning.mambaSeedStates.values()) {
          add(m.convState);
          add(m.ssmState);
        }
        for (const g of rec.conditioning.gqaPrefixCaches.values()) {
          add(g.K);
          add(g.V);
        }
      }

      if (rec.denoiserPass) {
        add(rec.denoiserPass.tokenIds);
        add(rec.denoiserPass.tEmb);
        add(rec.denoiserPass.logits);
        add(rec.denoiserPass.probs);
        for (const rs of rec.denoiserPass.residualStreams) {
          add(rs);
        }
      }

      if (rec.kind === 'ar_step' || rec.kind === 'mock_ar_step') {
        add(rec.selectionLogits);
        add(rec.selectionProbs);
      } else if (rec.kind === 'diffusion_confidence_commit') {
        add(rec.unmaskStep.confidence);
        add(rec.unmaskStep.logXTheta);
        add(rec.unmaskStep.unscaledProbs);
        add(rec.variateProvenance.variates);
      }
    }
  }

  return buffers;
}

function configsEqual(
  a: TwoTowerSession['pendingConfig'],
  b: TwoTowerSession['pendingConfig'],
): boolean {
  if (
    a.mode !== b.mode ||
    a.blockSize !== b.blockSize ||
    a.stepsPerBlock !== b.stepsPerBlock ||
    a.confidenceThreshold !== b.confidenceThreshold ||
    a.temperature !== b.temperature ||
    a.samplingSeed !== b.samplingSeed ||
    a.eosTokenIds.length !== b.eosTokenIds.length
  ) {
    return false;
  }
  for (let i = 0; i < a.eosTokenIds.length; i++) {
    if (a.eosTokenIds[i] !== b.eosTokenIds[i]) return false;
  }
  return true;
}

/**
 * Pure verification helper proving that executing, navigating, restarting, or exiting a
 * `TwoTowerTourController` NEVER mutates an external user exploration `TwoTowerSession` and
 * shares strictly ZERO session-owned typed-array buffers with it.
 */
export function verifyTourIsolation(
  userSessionBefore: TwoTowerSession,
  tourController: TwoTowerTourController,
  userSessionAfter: TwoTowerSession,
): TwoTowerTourIsolationReport {
  const reasons: string[] = [];

  const beforeBranch = getActiveBranch(userSessionBefore);
  const afterBranch = getActiveBranch(userSessionAfter);
  const beforeRec = getActiveRecord(userSessionBefore);
  const afterRec = getActiveRecord(userSessionAfter);
  const tourSession = tourController.getSession();

  const sameSessionReference = userSessionBefore === userSessionAfter;
  const sameSessionId = userSessionBefore.sessionId === userSessionAfter.sessionId;
  const sameActiveBranchId = userSessionBefore.activeBranchId === userSessionAfter.activeBranchId;
  const sameCursorIndex = beforeBranch.cursorIndex === afterBranch.cursorIndex;
  const sameMutationSeq = userSessionBefore.mutationSeq === userSessionAfter.mutationSeq;
  const sameBranchCount = userSessionBefore.branches.length === userSessionAfter.branches.length;
  const sameActiveRecordId = beforeRec.identity.recordId === afterRec.identity.recordId;
  const sameContextSnapshotDigest =
    beforeRec.contextSnapshot.snapshotDigest === afterRec.contextSnapshot.snapshotDigest;
  const sameContextCacheDigest =
    beforeRec.contextSnapshot.ctxState.stateDigest ===
    afterRec.contextSnapshot.ctxState.stateDigest;
  const samePendingConfig = configsEqual(
    userSessionBefore.pendingConfig,
    userSessionAfter.pendingConfig,
  );
  const distinctFromTourSessionId = tourSession.sessionId !== userSessionAfter.sessionId;
  const distinctFromTourSessionReference =
    tourSession !== userSessionBefore && tourSession !== userSessionAfter;

  if (!sameSessionId) reasons.push('User sessionId changed across tour execution.');
  if (!sameActiveBranchId) reasons.push('User activeBranchId changed across tour execution.');
  if (!sameCursorIndex) reasons.push('User active branch cursorIndex changed across tour execution.');
  if (!sameMutationSeq) reasons.push('User mutationSeq changed across tour execution.');
  if (!sameBranchCount) reasons.push('User branch count changed across tour execution.');
  if (!sameActiveRecordId) reasons.push('User active recordId changed across tour execution.');
  if (!sameContextSnapshotDigest) {
    reasons.push('User Context snapshotDigest changed across tour execution.');
  }
  if (!sameContextCacheDigest) {
    reasons.push('User Context stateDigest changed across tour execution.');
  }
  if (!samePendingConfig) reasons.push('User pendingConfig changed across tour execution.');
  if (!distinctFromTourSessionId) {
    reasons.push('Tour sessionId collides with external user sessionId.');
  }
  if (!distinctFromTourSessionReference) {
    reasons.push('Tour session aliases external user session object reference.');
  }

  // Verify every branch structure matches between before & after
  if (sameBranchCount) {
    for (let i = 0; i < userSessionBefore.branches.length; i++) {
      const b0 = userSessionBefore.branches[i];
      const b1 = userSessionAfter.branches[i];
      if (
        b0.branchId !== b1.branchId ||
        b0.cursorIndex !== b1.cursorIndex ||
        b0.records.length !== b1.records.length
      ) {
        reasons.push(`User branch[${i}] (${b0.branchId}) metadata or length diverged.`);
        break;
      }
      for (let r = 0; r < b0.records.length; r++) {
        if (b0.records[r].identity.recordId !== b1.records[r].identity.recordId) {
          reasons.push(`User branch "${b0.branchId}" record[${r}] diverged.`);
          break;
        }
      }
    }
  }

  // Verify snapshot typed-array integrity (detects silent in-place Float32Array tampering!)
  let snapshotIntegrityValid = true;
  try {
    validateContextSnapshot(afterRec.contextSnapshot, userSessionAfter.model.contextModel);
  } catch (err) {
    snapshotIntegrityValid = false;
    reasons.push(
      `User Context snapshot integrity failed: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  // Verify zero session-owned ArrayBuffer aliasing between tour session and user session
  const userBuffers = collectSessionOwnedArrayBuffers(userSessionAfter);
  const tourBuffers = collectSessionOwnedArrayBuffers(tourSession);
  let sharedArrayBufferCount = 0;
  for (const buf of tourBuffers) {
    if (userBuffers.has(buf)) {
      sharedArrayBufferCount++;
    }
  }
  if (sharedArrayBufferCount > 0) {
    reasons.push(
      `Tour session shares ${sharedArrayBufferCount} session-owned ArrayBuffer(s) with user session.`,
    );
  }

  return Object.freeze({
    isolated: reasons.length === 0,
    sameSessionReference,
    sameSessionId,
    sameActiveBranchId,
    sameCursorIndex,
    sameMutationSeq,
    sameBranchCount,
    sameActiveRecordId,
    sameContextSnapshotDigest,
    sameContextCacheDigest,
    samePendingConfig,
    distinctFromTourSessionId,
    distinctFromTourSessionReference,
    snapshotIntegrityValid,
    sharedArrayBufferCount,
    reasons: Object.freeze(reasons),
  });
}
