/**
 * Immutable-by-Contract Interactive TwoTower Session State Architecture (`src/twotower/session.ts`)
 *
 * Pure headless session store for Nemotron-Labs-TwoTower providing:
 *   1. Explicit, individually inspectable lifecycle transitions across ALL three execution modes:
 *      - `'context_prefill'` (`runTwoTowerContextPrefillWithTrace`, split `[0..P-2]` + `[P-1]`)
 *      - `'ar_step'` (`runTwoTowerArStep`)
 *      - `'mock_ar_step'` (`runTwoTowerMockArStep`, `KV[:-1]` + `mambaS2`, unmodulated)
 *      - `'diffusion_block_init'` (single shared ~78 KiB `TwoTowerBlockConditioningBase`)
 *      - `'diffusion_refinement'` (individual Denoiser pass `r`, leaving `xt` unchanged)
 *      - `'diffusion_confidence_commit'` (`runConfidenceUnmaskStep` + presentation-safe view +
 *        deterministic keyed variate provenance)
 *      - `'diffusion_block_commit'` (explicit commit from `'completed_uncommitted'` to Context Tower)
 *   2. Full explicit identity taxonomy (`TwoTowerRecordIdentity`) and frozen `configAtStep` on every
 *      record, with strict parameter-change non-retroactivity.
 *   3. Bounded branching history (`maxBranches`, `maxRecords`), backward/forward navigation,
 *      automatic non-destructive branching when stepping from a rewound cursor, active-branch-safe
 *      deletion, and restart.
 *   4. Automatic trace memory hygiene (`retainEagerTracesOnHistory = false` by default) and explicit
 *      `pruneInactiveSessionTraces` with 0-ULP lazy Context & Denoiser reconstruction.
 *   5. Exact deduplicated `Set<ArrayBufferLike>` session memory accounting
 *      (`measureTwoTowerSessionRetainedBytes`).
 *
 * Pure headless TypeScript — zero DOM, WebGL, scene, or render imports; zero `Math.random()`.
 */

import { DEFAULT_RETAINED_TRACE_BAYS } from '../math/hybridFlow';
import {
  buildBlockConditioningBase,
  deriveRefinementConditioningView,
  measureConditioningTypedArrayBytes,
  type TwoTowerBlockConditioningBase,
  type TwoTowerContextConditioningState,
  type TwoTowerContextSnapshot,
} from './contracts';
import {
  TWOTOWER_DEFAULT_BLOCK_SIZE,
  TWOTOWER_DEFAULT_CONFIDENCE_THRESHOLD,
  TWOTOWER_DEFAULT_KV_CAPACITY,
  TWOTOWER_DEFAULT_STEPS_PER_BLOCK,
  TWOTOWER_DEFAULT_TEMPERATURE,
  TWOTOWER_MAX_QUERY_LEN,
  commitTwoTowerBlockToContextWithTrace,
  runTwoTowerArStep,
  runTwoTowerContextPrefillWithTrace,
  runTwoTowerDenoiserPass,
  runTwoTowerMockArStep,
  type TwoTowerContextExecutionResult,
  type TwoTowerDenoiserPassResult,
} from './execution';
import type { TwoTowerModelParameters } from './parameters';
import {
  blockContainsEos,
  buildSamplerStepPresentation,
  deriveRefinementUniformVariates,
  runConfidenceUnmaskStep,
  type ConfidenceUnmaskStepResult,
  type SamplerStepPresentation,
} from './sampler';
import type { TwoTowerExecutionMode } from './spec';

// ============================================================================
// CONSTANTS & DEFAULTS
// ============================================================================

export const DEFAULT_TWOTOWER_SESSION_MAX_BRANCHES = 12 as const;
export const DEFAULT_TWOTOWER_SESSION_MAX_RECORDS = 128 as const;
export const DEFAULT_TWOTOWER_SESSION_ID = 'twotower-session-1' as const;
export const DEFAULT_TWOTOWER_SESSION_SAMPLING_SEED = 20260701 as const;
export const DEFAULT_TWOTOWER_PROMPT_TOKENS: readonly number[] = Object.freeze([
  0, 1, 2, 3, 4, 5, 6, 7,
]);

// ============================================================================
// PUBLIC TYPES
// ============================================================================

export type TwoTowerSessionTransitionKind =
  | 'context_prefill'
  | 'ar_step'
  | 'mock_ar_step'
  | 'diffusion_block_init'
  | 'diffusion_refinement'
  | 'diffusion_confidence_commit'
  | 'diffusion_block_commit';

export type TwoTowerSessionLifecyclePhase =
  | 'ready'
  | 'block_initialized'
  | 'refinement_denoised'
  | 'block_refining'
  | 'completed_uncommitted'
  | 'eos_terminated'
  | 'capacity_exhausted';

export type TwoTowerActiveBlockStatus =
  | 'initialized'
  | 'awaiting_confidence_commit'
  | 'refining'
  | 'completed_uncommitted';

export type TwoTowerVariateProvenanceMode =
  | 'greedy_none'
  | 'derived_deterministic'
  | 'caller_explicit';

export interface TwoTowerVariateProvenance {
  readonly mode: TwoTowerVariateProvenanceMode;
  readonly samplingSeed: number | null;
  readonly derivationKey: string | null;
  readonly variateDigest: string | null;
  readonly variates: Float32Array | null;
}

export interface TwoTowerSessionConfig {
  readonly mode: TwoTowerExecutionMode;
  readonly blockSize: number;
  readonly stepsPerBlock: number;
  readonly confidenceThreshold: number;
  readonly temperature: number;
  readonly eosTokenIds: readonly number[];
  readonly samplingSeed: number;
}

export interface TwoTowerSessionConfigPatch {
  readonly mode?: TwoTowerExecutionMode;
  readonly blockSize?: number;
  readonly stepsPerBlock?: number;
  readonly confidenceThreshold?: number;
  readonly temperature?: number;
  readonly eosTokenIds?: number | readonly number[];
  readonly samplingSeed?: number;
}

export interface TwoTowerActiveBlockConfig {
  readonly blockId: number;
  readonly blockSize: number;
  readonly stepsPerBlock: number;
}

export interface TwoTowerRecordIdentity {
  readonly modelId: string;
  readonly twoTowerFingerprint: string;
  readonly contextParameterFingerprint: string;
  readonly denoiserParameterFingerprint: string;
  readonly sourceCacheDigest: string;
  readonly sourceSnapshotDigest: string | null;
  readonly resultCacheDigest: string;
  readonly resultSnapshotDigest: string;
  readonly sessionId: string;
  readonly branchId: string;
  readonly recordId: string;
  readonly parentRecordId: string | null;
  readonly recordIndex: number;
  readonly blockId: number | null;
  readonly refinementIndex: number | null;
  readonly absoluteTokenStart: number;
  readonly absoluteTokenRange: readonly [number, number];
}

export type TwoTowerSessionTermination =
  | {
      readonly reason: 'eos';
      readonly mode: TwoTowerExecutionMode;
      readonly committedPrefixLen: number;
      readonly blockId: number | null;
      readonly triggerTokenIds: readonly number[];
    }
  | {
      readonly reason: 'capacity_exhausted';
      readonly committedPrefixLen: number;
      readonly kvCapacity: number;
    };

export interface TwoTowerActiveDiffusionBlock {
  readonly blockId: number;
  readonly status: TwoTowerActiveBlockStatus;
  readonly blockConfig: TwoTowerActiveBlockConfig;
  readonly absoluteTokenStart: number;
  readonly absoluteTokenRange: readonly [number, number];
  /** Single immutable shared conditioning base (~78 KiB) across all refinements of this block. */
  readonly conditioningBase: TwoTowerBlockConditioningBase;
  /** Current noisy/partially-unmasked block token IDs of length `blockConfig.blockSize`. */
  readonly currentTokens: readonly number[];
  readonly currentTokenLabels: readonly string[];
  readonly nMasked: number;
  /** Count of completed `(refinement -> confidence_commit)` pairs for this block. */
  readonly completedRefinements: number;
  /** Pending refinement index when `status === 'awaiting_confidence_commit'`, else `null`. */
  readonly pendingRefinementIndex: number | null;
  readonly pendingConditioning: TwoTowerContextConditioningState | null;
  readonly pendingDenoiserPass: TwoTowerDenoiserPassResult | null;
  readonly latestUnmaskStep: ConfidenceUnmaskStepResult | null;
  readonly latestSamplerPresentation: SamplerStepPresentation | null;
  readonly contextCommitted: false;
  readonly pendingEosInBlock: boolean;
  readonly pendingEosTokenIds: readonly number[];
  readonly nfe: number;
  readonly earlyCompleted: boolean;
}

export interface TwoTowerSessionRecordBase {
  readonly kind: TwoTowerSessionTransitionKind;
  readonly identity: TwoTowerRecordIdentity;
  readonly configAtStep: TwoTowerSessionConfig;
  readonly lifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly contextSnapshot: TwoTowerContextSnapshot;
  readonly committedPrefixLen: number;
  readonly kvCapacity: number;
  readonly nextBlockId: number;
  readonly activeBlock: TwoTowerActiveDiffusionBlock | null;
  readonly termination: TwoTowerSessionTermination | null;
  readonly contextExecution: TwoTowerContextExecutionResult | null;
  readonly conditioning: TwoTowerContextConditioningState | null;
  readonly denoiserPass: TwoTowerDenoiserPassResult | null;
}

export interface TwoTowerSessionContextPrefillRecord extends TwoTowerSessionRecordBase {
  readonly kind: 'context_prefill';
  readonly activeBlock: null;
  readonly contextExecution: TwoTowerContextExecutionResult;
  readonly conditioning: null;
  readonly denoiserPass: null;
  readonly promptTokenIds: readonly number[];
  readonly promptTokenLabels: readonly string[];
}

export interface TwoTowerSessionArStepRecord extends TwoTowerSessionRecordBase {
  readonly kind: 'ar_step';
  readonly activeBlock: null;
  readonly contextExecution: TwoTowerContextExecutionResult;
  readonly conditioning: null;
  readonly denoiserPass: null;
  readonly selectionLogits: Float32Array;
  readonly selectionProbs: Float32Array;
  readonly selectedTokenId: number;
  readonly selectedTokenLabel: string;
  readonly wasExplicitTokenOverride: boolean;
}

export interface TwoTowerSessionMockArStepRecord extends TwoTowerSessionRecordBase {
  readonly kind: 'mock_ar_step';
  readonly activeBlock: null;
  readonly contextExecution: TwoTowerContextExecutionResult;
  readonly conditioning: TwoTowerContextConditioningState;
  readonly denoiserPass: TwoTowerDenoiserPassResult;
  readonly selectionLogits: Float32Array;
  readonly selectionProbs: Float32Array;
  readonly selectedTokenId: number;
  readonly selectedTokenLabel: string;
  readonly wasExplicitTokenOverride: boolean;
}

export interface TwoTowerSessionDiffusionBlockInitRecord extends TwoTowerSessionRecordBase {
  readonly kind: 'diffusion_block_init';
  readonly activeBlock: TwoTowerActiveDiffusionBlock;
  readonly contextExecution: null;
  readonly conditioning: null;
  readonly denoiserPass: null;
  readonly blockConfig: TwoTowerActiveBlockConfig;
  readonly conditioningBase: TwoTowerBlockConditioningBase;
  readonly initialBlockTokens: readonly number[];
  readonly initialBlockLabels: readonly string[];
}

export interface TwoTowerSessionDiffusionRefinementRecord extends TwoTowerSessionRecordBase {
  readonly kind: 'diffusion_refinement';
  readonly activeBlock: TwoTowerActiveDiffusionBlock;
  readonly contextExecution: null;
  readonly conditioning: TwoTowerContextConditioningState;
  readonly denoiserPass: TwoTowerDenoiserPassResult;
  readonly blockConfig: TwoTowerActiveBlockConfig;
  readonly conditioningBase: TwoTowerBlockConditioningBase;
  readonly refinementIndex: number;
  readonly xtInput: readonly number[];
  readonly xtInputLabels: readonly string[];
  readonly tModel: number;
}

export interface TwoTowerSessionDiffusionConfidenceCommitRecord
  extends TwoTowerSessionRecordBase {
  readonly kind: 'diffusion_confidence_commit';
  readonly activeBlock: TwoTowerActiveDiffusionBlock;
  readonly contextExecution: null;
  readonly conditioning: TwoTowerContextConditioningState;
  readonly denoiserPass: TwoTowerDenoiserPassResult;
  readonly blockConfig: TwoTowerActiveBlockConfig;
  readonly conditioningBase: TwoTowerBlockConditioningBase;
  readonly refinementIndex: number;
  readonly unmaskStep: ConfidenceUnmaskStepResult;
  readonly samplerPresentation: SamplerStepPresentation;
  readonly variateProvenance: TwoTowerVariateProvenance;
}

export interface TwoTowerSessionDiffusionBlockCommitRecord extends TwoTowerSessionRecordBase {
  readonly kind: 'diffusion_block_commit';
  readonly activeBlock: null;
  readonly contextExecution: TwoTowerContextExecutionResult;
  readonly conditioning: null;
  readonly denoiserPass: null;
  readonly blockConfig: TwoTowerActiveBlockConfig;
  readonly committedBlockTokens: readonly number[];
  readonly committedBlockLabels: readonly string[];
  readonly nfe: number;
  readonly earlyCompleted: boolean;
  readonly stoppedByEos: boolean;
}

export type TwoTowerSessionRecord =
  | TwoTowerSessionContextPrefillRecord
  | TwoTowerSessionArStepRecord
  | TwoTowerSessionMockArStepRecord
  | TwoTowerSessionDiffusionBlockInitRecord
  | TwoTowerSessionDiffusionRefinementRecord
  | TwoTowerSessionDiffusionConfidenceCommitRecord
  | TwoTowerSessionDiffusionBlockCommitRecord;

export type TwoTowerContextExecutingRecord =
  | TwoTowerSessionContextPrefillRecord
  | TwoTowerSessionArStepRecord
  | TwoTowerSessionMockArStepRecord
  | TwoTowerSessionDiffusionBlockCommitRecord;

export interface TwoTowerLatestContextExecution {
  readonly record: TwoTowerContextExecutingRecord;
  readonly branchCursorIndex: number;
  readonly execution: TwoTowerContextExecutionResult;
}

export interface TwoTowerSessionBranch {
  readonly branchId: string;
  readonly label: string;
  readonly parentBranchId: string | null;
  readonly forkPointRecordId: string | null;
  readonly forkPointIndex: number | null;
  readonly createdAtSeq: number;
  readonly lastActiveSeq: number;
  readonly records: readonly TwoTowerSessionRecord[];
  readonly cursorIndex: number;
}

export interface TwoTowerSessionBounds {
  readonly maxBranches: number;
  readonly maxRecords: number;
}

export interface TwoTowerSessionTraceRetention {
  readonly retainedTraceBays: readonly number[];
  readonly maxRetainedTraces: number;
  readonly retainEagerTracesOnHistory: boolean;
}

export interface TwoTowerSession {
  readonly sessionId: string;
  readonly model: TwoTowerModelParameters;
  readonly kvCapacity: number;
  readonly promptTokenIds: readonly number[];
  readonly promptTokenLabels: readonly string[];
  readonly bounds: TwoTowerSessionBounds;
  readonly traceRetention: TwoTowerSessionTraceRetention;
  readonly pendingConfig: TwoTowerSessionConfig;
  readonly branches: readonly TwoTowerSessionBranch[];
  readonly activeBranchId: string;
  readonly nextBranchOrdinal: number;
  readonly mutationSeq: number;
}

export interface TwoTowerSessionCapabilities {
  readonly lifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly isTerminated: boolean;
  readonly terminationReason: 'eos' | 'capacity_exhausted' | null;
  readonly committedPrefixLen: number;
  readonly kvCapacity: number;
  readonly remainingKvCapacity: number;
  readonly pendingBlockSize: number;
  readonly hasActiveBlock: boolean;
  readonly activeBlockStatus: TwoTowerActiveBlockStatus | null;
  readonly isAtBranchTail: boolean;
  readonly willAutoForkOnMutatingStep: boolean;
  readonly canStepAr: boolean;
  readonly canStepMockAr: boolean;
  readonly canInitDiffusionBlock: boolean;
  readonly canRunDiffusionRefinement: boolean;
  readonly canCommitRefinementConfidence: boolean;
  readonly canCommitActiveBlockToContext: boolean;
  readonly canStepNext: boolean;
  readonly nextTransitionKind: TwoTowerSessionTransitionKind | null;
  readonly canNavigateBack: boolean;
  readonly canNavigateForward: boolean;
  readonly canForkBranch: boolean;
  readonly canDeleteAnyBranch: boolean;
}

export interface TwoTowerSessionMemoryBreakdown {
  /** Total session-owned deduplicated typed-array bytes (excludes read-only `sharedModelWeightBytes`). */
  readonly totalRetainedBytes: number;
  /** Persistent Context snapshots (`ctxState` + `mambaS2` + `arLogits` + `arProbs`). */
  readonly contextSnapshotBytes: number;
  /** Cross-tower read-only conditioning buffers (`TwoTowerBlockConditioningBase` + Mock-AR conditioning). */
  readonly blockConditioningBytes: number;
  /** Context Tower segment trace handles (private checkpoints + retained eager/LRU bay traces). */
  readonly contextTraceBytes: number;
  /** Denoiser Tower pass handles (private checkpoints + deduplicated private conditioning + retained traces). */
  readonly denoiserTraceBytes: number;
  /** Sampler & selection buffers (`unmaskStep`, `variateProvenance`, AR/Mock-AR selection logits/probs). */
  readonly samplerAndSelectionBytes: number;
  /** Shared read-only model weights owned by `session.model` (NEVER duplicated across sessions/branches/records). */
  readonly sharedModelWeightBytes: number;
  /** Combined footprint (`totalRetainedBytes + sharedModelWeightBytes`). */
  readonly totalIncludingModelWeightsBytes: number;
  readonly uniqueRecordCount: number;
  readonly branchCount: number;
  readonly uniqueContextSnapshotCount: number;
  readonly uniqueBlockConditioningCount: number;
}

export interface CreateTwoTowerSessionOptions {
  readonly sessionId?: string;
  readonly promptTokenIds?: readonly number[];
  readonly promptTokenLabels?: readonly string[];
  readonly kvCapacity?: number;
  readonly mode?: TwoTowerExecutionMode;
  readonly blockSize?: number;
  readonly stepsPerBlock?: number;
  readonly confidenceThreshold?: number;
  readonly temperature?: number;
  readonly eosTokenIds?: number | readonly number[];
  readonly samplingSeed?: number;
  readonly maxBranches?: number;
  readonly maxRecords?: number;
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
  readonly retainEagerTracesOnHistory?: boolean;
  readonly initialBranchLabel?: string;
}

export interface RestartTwoTowerSessionOptions {
  readonly sessionId?: string;
  readonly promptTokenIds?: readonly number[];
  readonly promptTokenLabels?: readonly string[];
  readonly kvCapacity?: number;
  readonly configPatch?: TwoTowerSessionConfigPatch;
  readonly maxBranches?: number;
  readonly maxRecords?: number;
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
  readonly retainEagerTracesOnHistory?: boolean;
  readonly initialBranchLabel?: string;
}

export interface ForkTwoTowerSessionBranchOptions {
  readonly fromCursorIndex?: number;
  readonly label?: string;
  readonly configPatch?: TwoTowerSessionConfigPatch;
  readonly switchToNewBranch?: boolean;
}

export interface StepTwoTowerArOptions {
  readonly nextTokenId?: number;
  readonly nextTokenLabel?: string;
}

export interface StepTwoTowerMockArOptions {
  readonly nextTokenId?: number;
  readonly nextTokenLabel?: string;
}

export interface InitTwoTowerDiffusionBlockOptions {
  readonly blockSize?: number;
  readonly stepsPerBlock?: number;
  readonly initialBlockTokens?: readonly number[];
}

export interface RunTwoTowerDiffusionRefinementOptions {
  readonly tOverride?: number;
}

export interface CommitTwoTowerRefinementConfidenceOptions {
  readonly confidenceThreshold?: number;
  readonly temperature?: number;
  readonly samplingSeed?: number;
  readonly uniformVariates?: Float32Array | readonly number[];
}

export interface CommitTwoTowerActiveBlockOptions {
  readonly eosTokenIds?: number | readonly number[];
}

// ============================================================================
// INTERNAL PRIVATE ISOLATION & VALIDATION HELPERS
// ============================================================================

interface PrivatePendingRefinementBuffers {
  readonly logits: Float32Array;
  readonly probs: Float32Array;
}

/**
 * Private detached clones of Denoiser pass `logits`/`probs` held ONLY on `awaiting_confidence_commit`
 * active blocks so mutating exposed `record.denoiserPass.logits` before committing confidence cannot
 * corrupt the subsequent `runConfidenceUnmaskStep` or `buildSamplerStepPresentation`.
 */
const privatePendingRefinementCache = new WeakMap<
  TwoTowerActiveDiffusionBlock,
  PrivatePendingRefinementBuffers
>();

function computeExplicitVariateDigest(variates: Float32Array): string {
  let h1 = 0x811c9dc5 >>> 0;
  let h2 = 0x9e3779b9 >>> 0;
  const f32Buf = new ArrayBuffer(4);
  const f32View = new Float32Array(f32Buf);
  const u32View = new Uint32Array(f32Buf);

  const feedByte = (b: number) => {
    const byte = b & 0xff;
    h1 ^= byte;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= (byte + 0x5a) & 0xff;
    h2 = Math.imul(h2, 0x85ebca6b) >>> 0;
  };
  const feedU32 = (v: number) => {
    const u = v >>> 0;
    feedByte(u);
    feedByte(u >>> 8);
    feedByte(u >>> 16);
    feedByte(u >>> 24);
  };

  // Tag domain "caller_explicit" + length + Float32 bits
  const tag = 'twotower-explicit-variates-v1';
  feedU32(tag.length);
  for (let i = 0; i < tag.length; i++) {
    const c = tag.charCodeAt(i);
    feedByte(c);
    feedByte(c >>> 8);
  }
  feedU32(variates.length);
  for (let i = 0; i < variates.length; i++) {
    f32View[0] = Object.is(variates[i], -0) ? 0 : variates[i];
    feedU32(u32View[0]);
  }
  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

function validateMode(mode: TwoTowerExecutionMode): TwoTowerExecutionMode {
  if (mode !== 'ar' && mode !== 'mock_ar' && mode !== 'mask_diffusion') {
    throw new RangeError(`Unsupported TwoTower execution mode: "${String(mode)}"`);
  }
  return mode;
}

function validateBlockSize(blockSize: number): number {
  if (!Number.isInteger(blockSize) || blockSize < 1 || blockSize > TWOTOWER_MAX_QUERY_LEN) {
    throw new RangeError(
      `blockSize must be an integer in [1..${TWOTOWER_MAX_QUERY_LEN}], received ${blockSize}`,
    );
  }
  return blockSize;
}

function validateStepsPerBlock(stepsPerBlock: number): number {
  if (!Number.isInteger(stepsPerBlock) || stepsPerBlock < 1) {
    throw new RangeError(`stepsPerBlock must be an integer >= 1, received ${stepsPerBlock}`);
  }
  return stepsPerBlock;
}

function validateConfidenceThreshold(confidenceThreshold: number): number {
  if (!Number.isFinite(confidenceThreshold)) {
    throw new RangeError(
      `confidenceThreshold must be finite, received ${confidenceThreshold}`,
    );
  }
  return confidenceThreshold;
}

function validateTemperature(temperature: number): number {
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new RangeError(`temperature must be finite >= 0, received ${temperature}`);
  }
  return temperature;
}

function validateSamplingSeed(samplingSeed: number): number {
  if (!Number.isInteger(samplingSeed)) {
    throw new RangeError(`samplingSeed must be a finite integer, received ${samplingSeed}`);
  }
  return samplingSeed;
}

function normalizeEosTokenIds(
  eosTokenIds: number | readonly number[],
  vocabSize: number,
): readonly number[] {
  const rawList = typeof eosTokenIds === 'number' ? [eosTokenIds] : eosTokenIds;
  const deduped: number[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < rawList.length; i++) {
    const id = rawList[i];
    if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
      throw new RangeError(
        `eosTokenIds[${i}] must be an integer in [0, ${vocabSize - 1}], received ${id}`,
      );
    }
    if (!seen.has(id)) {
      seen.add(id);
      deduped.push(id);
    }
  }
  return Object.freeze(deduped);
}

function validateAndMergeConfig(
  base: TwoTowerSessionConfig,
  patch: TwoTowerSessionConfigPatch | undefined,
  vocabSize: number,
): TwoTowerSessionConfig {
  const mode = validateMode(patch?.mode ?? base.mode);
  const blockSize = validateBlockSize(patch?.blockSize ?? base.blockSize);
  const stepsPerBlock = validateStepsPerBlock(patch?.stepsPerBlock ?? base.stepsPerBlock);
  const confidenceThreshold = validateConfidenceThreshold(
    patch?.confidenceThreshold ?? base.confidenceThreshold,
  );
  const temperature = validateTemperature(patch?.temperature ?? base.temperature);
  const eosTokenIds =
    patch?.eosTokenIds !== undefined
      ? normalizeEosTokenIds(patch.eosTokenIds, vocabSize)
      : base.eosTokenIds;
  const samplingSeed = validateSamplingSeed(patch?.samplingSeed ?? base.samplingSeed);

  return Object.freeze({
    mode,
    blockSize,
    stepsPerBlock,
    confidenceThreshold,
    temperature,
    eosTokenIds,
    samplingSeed,
  });
}

function validateTraceRetention(
  model: TwoTowerModelParameters,
  options?: {
    readonly retainedTraceBays?: readonly number[];
    readonly maxRetainedTraces?: number;
    readonly retainEagerTracesOnHistory?: boolean;
  },
  fallback?: TwoTowerSessionTraceRetention,
): TwoTowerSessionTraceRetention {
  const numLayers = model.contextModel.config.pattern.length;
  const rawBays =
    options?.retainedTraceBays ?? fallback?.retainedTraceBays ?? DEFAULT_RETAINED_TRACE_BAYS;
  const bays: number[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < rawBays.length; i++) {
    const bayId = rawBays[i];
    if (!Number.isInteger(bayId) || bayId < 1 || bayId > numLayers) {
      throw new RangeError(
        `retainedTraceBays[${i}] (${bayId}) must be an integer in [1..${numLayers}]`,
      );
    }
    if (!seen.has(bayId)) {
      seen.add(bayId);
      bays.push(bayId);
    }
  }
  const maxRetainedTraces = options?.maxRetainedTraces ?? fallback?.maxRetainedTraces ?? 12;
  if (!Number.isInteger(maxRetainedTraces) || maxRetainedTraces < 0) {
    throw new RangeError(
      `maxRetainedTraces must be an integer >= 0, received ${maxRetainedTraces}`,
    );
  }
  const retainEagerTracesOnHistory =
    options?.retainEagerTracesOnHistory ?? fallback?.retainEagerTracesOnHistory ?? false;

  return Object.freeze({
    retainedTraceBays: Object.freeze(bays),
    maxRetainedTraces,
    retainEagerTracesOnHistory,
  });
}

function validateBounds(
  maxBranches: number,
  maxRecords: number,
): TwoTowerSessionBounds {
  if (!Number.isInteger(maxBranches) || maxBranches < 1) {
    throw new RangeError(`maxBranches must be an integer >= 1, received ${maxBranches}`);
  }
  if (!Number.isInteger(maxRecords) || maxRecords < 1) {
    throw new RangeError(`maxRecords must be an integer >= 1, received ${maxRecords}`);
  }
  return Object.freeze({ maxBranches, maxRecords });
}

function resolveLabelsForTokens(
  tokens: readonly number[],
  vocab: readonly string[],
): readonly string[] {
  const labels = new Array<string>(tokens.length);
  for (let i = 0; i < tokens.length; i++) {
    labels[i] = vocab[tokens[i]] ?? `id:${tokens[i]}`;
  }
  return Object.freeze(labels);
}

function collectMatchingEosTokens(
  tokens: readonly number[],
  eosTokenIds: readonly number[],
): readonly number[] {
  if (eosTokenIds.length === 0) return Object.freeze([]);
  const eosSet = new Set<number>(eosTokenIds);
  const found: number[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    if (eosSet.has(tok) && !seen.has(tok)) {
      seen.add(tok);
      found.push(tok);
    }
  }
  return Object.freeze(found);
}

function makeRecordId(params: {
  readonly sessionId: string;
  readonly branchId: string;
  readonly recordIndex: number;
  readonly kind: TwoTowerSessionTransitionKind;
  readonly blockId: number | null;
  readonly refinementIndex: number | null;
}): string {
  const blockTag = params.blockId !== null ? `:b${params.blockId}` : '';
  const refTag = params.refinementIndex !== null ? `:r${params.refinementIndex}` : '';
  return `${params.sessionId}:${params.branchId}:rec-${params.recordIndex}:${params.kind}${blockTag}${refTag}`;
}

function buildRecordIdentity(params: {
  readonly model: TwoTowerModelParameters;
  readonly sessionId: string;
  readonly branchId: string;
  readonly parentRecordId: string | null;
  readonly recordIndex: number;
  readonly kind: TwoTowerSessionTransitionKind;
  readonly sourceCacheDigest: string;
  readonly sourceSnapshotDigest: string | null;
  readonly resultSnapshot: TwoTowerContextSnapshot;
  readonly blockId: number | null;
  readonly refinementIndex: number | null;
  readonly absoluteTokenStart: number;
  readonly absoluteTokenRange: readonly [number, number];
}): TwoTowerRecordIdentity {
  const recordId = makeRecordId({
    sessionId: params.sessionId,
    branchId: params.branchId,
    recordIndex: params.recordIndex,
    kind: params.kind,
    blockId: params.blockId,
    refinementIndex: params.refinementIndex,
  });
  return Object.freeze({
    modelId: params.model.modelId,
    twoTowerFingerprint: params.model.twoTowerFingerprint,
    contextParameterFingerprint: params.model.contextModel.parameterFingerprint,
    denoiserParameterFingerprint: params.model.denoiserModel.parameterFingerprint,
    sourceCacheDigest: params.sourceCacheDigest,
    sourceSnapshotDigest: params.sourceSnapshotDigest,
    resultCacheDigest: params.resultSnapshot.ctxState.stateDigest,
    resultSnapshotDigest: params.resultSnapshot.snapshotDigest,
    sessionId: params.sessionId,
    branchId: params.branchId,
    recordId,
    parentRecordId: params.parentRecordId,
    recordIndex: params.recordIndex,
    blockId: params.blockId,
    refinementIndex: params.refinementIndex,
    absoluteTokenStart: params.absoluteTokenStart,
    absoluteTokenRange: Object.freeze([
      params.absoluteTokenRange[0],
      params.absoluteTokenRange[1],
    ] as const),
  });
}

function countUniqueRecords(branches: readonly TwoTowerSessionBranch[]): number {
  const seen = new Set<string>();
  for (let i = 0; i < branches.length; i++) {
    const recs = branches[i].records;
    for (let j = 0; j < recs.length; j++) {
      seen.add(recs[j].identity.recordId);
    }
  }
  return seen.size;
}

function findOldestInactiveBranchIndex(
  branches: readonly TwoTowerSessionBranch[],
  protectedBranchIds: ReadonlySet<string>,
): number {
  let oldestIdx = -1;
  for (let i = 0; i < branches.length; i++) {
    const b = branches[i];
    if (protectedBranchIds.has(b.branchId)) continue;
    if (oldestIdx === -1) {
      oldestIdx = i;
    } else {
      const cur = branches[oldestIdx];
      if (
        b.lastActiveSeq < cur.lastActiveSeq ||
        (b.lastActiveSeq === cur.lastActiveSeq && b.createdAtSeq < cur.createdAtSeq)
      ) {
        oldestIdx = i;
      }
    }
  }
  return oldestIdx;
}

function enforceSessionBounds(params: {
  readonly branches: TwoTowerSessionBranch[];
  readonly activeBranchId: string;
  readonly protectedBranchIds: ReadonlySet<string>;
  readonly bounds: TwoTowerSessionBounds;
}): readonly TwoTowerSessionBranch[] {
  const branches = [...params.branches];
  const { activeBranchId, protectedBranchIds, bounds } = params;

  // 1. Enforce maxBranches by evicting oldest non-protected inactive branches
  while (branches.length > bounds.maxBranches) {
    let evictIdx = findOldestInactiveBranchIndex(branches, protectedBranchIds);
    if (evictIdx === -1) {
      evictIdx = findOldestInactiveBranchIndex(branches, new Set([activeBranchId]));
    }
    if (evictIdx === -1) {
      throw new RangeError(
        `Cannot satisfy maxBranches=${bounds.maxBranches} without evicting active branch "${activeBranchId}"`,
      );
    }
    branches.splice(evictIdx, 1);
  }

  // 2. Enforce maxRecords by evicting oldest inactive branches first
  while (countUniqueRecords(branches) > bounds.maxRecords) {
    const evictIdx = findOldestInactiveBranchIndex(branches, new Set([activeBranchId]));
    if (evictIdx !== -1) {
      branches.splice(evictIdx, 1);
      continue;
    }
    // Only active branch remains: trim oldest leading records while keeping cursor valid
    const activeIdx = branches.findIndex((b) => b.branchId === activeBranchId);
    if (activeIdx === -1) break;
    const active = branches[activeIdx];
    const excess = active.records.length - bounds.maxRecords;
    if (excess <= 0) break;
    const dropCount = Math.min(excess, active.cursorIndex);
    const trimmedRecords = Object.freeze(active.records.slice(dropCount));
    const newCursor = active.cursorIndex - dropCount;
    // If cursor itself was near front and still exceeds maxRecords, trim trailing records beyond cursor
    const finalRecords =
      trimmedRecords.length > bounds.maxRecords
        ? Object.freeze(trimmedRecords.slice(0, bounds.maxRecords))
        : trimmedRecords;
    const finalCursor = Math.min(newCursor, finalRecords.length - 1);
    branches[activeIdx] = Object.freeze({
      ...active,
      records: finalRecords,
      cursorIndex: finalCursor,
    });
    break;
  }

  return Object.freeze(branches);
}

function releaseTracesExceptActive(
  branches: readonly TwoTowerSessionBranch[],
  activeRecord: TwoTowerSessionRecord,
): void {
  const activeContextExec = activeRecord.contextExecution;
  const activeDenoiserPass = activeRecord.denoiserPass;
  const visitedContext = new Set<TwoTowerContextExecutionResult>();
  const visitedDenoiser = new Set<TwoTowerDenoiserPassResult>();

  for (let i = 0; i < branches.length; i++) {
    const recs = branches[i].records;
    for (let j = 0; j < recs.length; j++) {
      const rec = recs[j];
      if (rec === activeRecord) continue;
      if (
        rec.contextExecution !== null &&
        rec.contextExecution !== activeContextExec &&
        !visitedContext.has(rec.contextExecution)
      ) {
        visitedContext.add(rec.contextExecution);
        rec.contextExecution.releaseEagerTraces();
      }
      if (
        rec.denoiserPass !== null &&
        rec.denoiserPass !== activeDenoiserPass &&
        !visitedDenoiser.has(rec.denoiserPass)
      ) {
        visitedDenoiser.add(rec.denoiserPass);
        rec.denoiserPass.releaseEagerTraces();
      }
    }
  }
}

interface PreparedBranchTarget {
  readonly targetBranchId: string;
  readonly sourceBranch: TwoTowerSessionBranch;
  readonly sourceRecord: TwoTowerSessionRecord;
  readonly willAutoFork: boolean;
  readonly nextBranchOrdinal: number;
  readonly nextSeq: number;
}

function prepareMutatingTarget(session: TwoTowerSession): PreparedBranchTarget {
  const sourceBranch = getActiveBranch(session);
  const sourceRecord = sourceBranch.records[sourceBranch.cursorIndex];
  const willAutoFork = sourceBranch.cursorIndex < sourceBranch.records.length - 1;
  if (willAutoFork && session.bounds.maxBranches < 2) {
    throw new RangeError(
      `Cannot perform non-destructive step from rewound cursor (${sourceBranch.cursorIndex}/${sourceBranch.records.length - 1}) when maxBranches is ${session.bounds.maxBranches}`,
    );
  }
  const targetBranchId = willAutoFork
    ? `branch-${session.nextBranchOrdinal}`
    : sourceBranch.branchId;
  return {
    targetBranchId,
    sourceBranch,
    sourceRecord,
    willAutoFork,
    nextBranchOrdinal: willAutoFork ? session.nextBranchOrdinal + 1 : session.nextBranchOrdinal,
    nextSeq: session.mutationSeq + 1,
  };
}

function commitAppendedRecord(
  session: TwoTowerSession,
  target: PreparedBranchTarget,
  newRecord: TwoTowerSessionRecord,
): TwoTowerSession {
  const { sourceBranch, targetBranchId, willAutoFork, nextBranchOrdinal, nextSeq } = target;
  let nextBranches: TwoTowerSessionBranch[];
  let protectedIds: Set<string>;

  if (willAutoFork) {
    const forkPrefix = sourceBranch.records.slice(0, sourceBranch.cursorIndex + 1);
    const forkedRecords = Object.freeze([...forkPrefix, newRecord]);
    const forkedBranch: TwoTowerSessionBranch = Object.freeze({
      branchId: targetBranchId,
      label: `Branch ${nextBranchOrdinal - 1} (from ${sourceBranch.branchId}@${sourceBranch.cursorIndex})`,
      parentBranchId: sourceBranch.branchId,
      forkPointRecordId: target.sourceRecord.identity.recordId,
      forkPointIndex: sourceBranch.cursorIndex,
      createdAtSeq: nextSeq,
      lastActiveSeq: nextSeq,
      records: forkedRecords,
      cursorIndex: forkedRecords.length - 1,
    });
    nextBranches = [...session.branches, forkedBranch];
    // Protect BOTH the newly created active fork and its source branch during maxBranches eviction
    protectedIds = new Set([targetBranchId, sourceBranch.branchId]);
  } else {
    const updatedRecords = Object.freeze([...sourceBranch.records, newRecord]);
    const updatedBranch: TwoTowerSessionBranch = Object.freeze({
      ...sourceBranch,
      lastActiveSeq: nextSeq,
      records: updatedRecords,
      cursorIndex: updatedRecords.length - 1,
    });
    nextBranches = session.branches.map((b) =>
      b.branchId === sourceBranch.branchId ? updatedBranch : b,
    );
    protectedIds = new Set([targetBranchId]);
  }

  const boundedBranches = enforceSessionBounds({
    branches: nextBranches,
    activeBranchId: targetBranchId,
    protectedBranchIds: protectedIds,
    bounds: session.bounds,
  });

  if (!session.traceRetention.retainEagerTracesOnHistory) {
    releaseTracesExceptActive(boundedBranches, newRecord);
  }

  return Object.freeze({
    ...session,
    branches: boundedBranches,
    activeBranchId: targetBranchId,
    nextBranchOrdinal,
    mutationSeq: nextSeq,
  });
}

// ============================================================================
// QUERY & CAPABILITY HELPERS
// ============================================================================

export function getActiveBranch(session: TwoTowerSession): TwoTowerSessionBranch {
  for (let i = 0; i < session.branches.length; i++) {
    if (session.branches[i].branchId === session.activeBranchId) {
      return session.branches[i];
    }
  }
  throw new RangeError(`Active branch "${session.activeBranchId}" not found in session`);
}

export function getActiveRecord(session: TwoTowerSession): TwoTowerSessionRecord {
  const branch = getActiveBranch(session);
  const record = branch.records[branch.cursorIndex];
  if (!record) {
    throw new RangeError(
      `Invalid cursorIndex ${branch.cursorIndex} on branch "${branch.branchId}" (length ${branch.records.length})`,
    );
  }
  return record;
}

/**
 * Return the most recent Context Tower execution (`context_prefill`, `ar_step`, `mock_ar_step`,
 * or `diffusion_block_commit`) at or before the active branch's cursor so Context trace inspection
 * remains available alongside Denoiser scratchpad inspection at every point in the lifecycle.
 */
export function getLatestContextExecution(
  session: TwoTowerSession,
): TwoTowerLatestContextExecution | null {
  const branch = getActiveBranch(session);
  for (let idx = branch.cursorIndex; idx >= 0; idx--) {
    const rec = branch.records[idx];
    if (
      rec.kind === 'context_prefill' ||
      rec.kind === 'ar_step' ||
      rec.kind === 'mock_ar_step' ||
      rec.kind === 'diffusion_block_commit'
    ) {
      return Object.freeze({
        record: rec,
        branchCursorIndex: idx,
        execution: rec.contextExecution,
      });
    }
  }
  return null;
}

export function getSessionCapabilities(session: TwoTowerSession): TwoTowerSessionCapabilities {
  const branch = getActiveBranch(session);
  const record = branch.records[branch.cursorIndex];
  const activeBlock = record.activeBlock;
  const isTerminated = record.termination !== null;
  const terminationReason = record.termination ? record.termination.reason : null;
  const committedPrefixLen = record.committedPrefixLen;
  const kvCapacity = session.kvCapacity;
  const remainingKvCapacity = kvCapacity - committedPrefixLen;
  const pendingBlockSize = session.pendingConfig.blockSize;
  const hasActiveBlock = activeBlock !== null;
  const activeBlockStatus = activeBlock ? activeBlock.status : null;
  const isAtBranchTail = branch.cursorIndex === branch.records.length - 1;
  const willAutoForkOnMutatingStep = !isAtBranchTail;
  const canMutateCursor = isAtBranchTail || session.bounds.maxBranches >= 2;

  const canStepAr = !isTerminated && !hasActiveBlock && remainingKvCapacity >= 1 && canMutateCursor;
  const canStepMockAr =
    !isTerminated && !hasActiveBlock && remainingKvCapacity >= 1 && canMutateCursor;
  const canInitDiffusionBlock =
    !isTerminated &&
    !hasActiveBlock &&
    remainingKvCapacity >= pendingBlockSize &&
    canMutateCursor;
  const canRunDiffusionRefinement =
    !isTerminated &&
    activeBlock !== null &&
    (activeBlock.status === 'initialized' || activeBlock.status === 'refining') &&
    activeBlock.nMasked > 0 &&
    activeBlock.completedRefinements < activeBlock.blockConfig.stepsPerBlock &&
    canMutateCursor;
  const canCommitRefinementConfidence =
    !isTerminated &&
    activeBlock !== null &&
    activeBlock.status === 'awaiting_confidence_commit' &&
    canMutateCursor;
  const canCommitActiveBlockToContext =
    !isTerminated &&
    activeBlock !== null &&
    activeBlock.status === 'completed_uncommitted' &&
    activeBlock.nMasked === 0 &&
    canMutateCursor;

  let nextTransitionKind: TwoTowerSessionTransitionKind | null = null;
  if (!isTerminated && canMutateCursor) {
    if (hasActiveBlock) {
      if (canRunDiffusionRefinement) {
        nextTransitionKind = 'diffusion_refinement';
      } else if (canCommitRefinementConfidence) {
        nextTransitionKind = 'diffusion_confidence_commit';
      } else if (canCommitActiveBlockToContext) {
        nextTransitionKind = 'diffusion_block_commit';
      }
    } else if (session.pendingConfig.mode === 'ar') {
      nextTransitionKind = canStepAr ? 'ar_step' : null;
    } else if (session.pendingConfig.mode === 'mock_ar') {
      nextTransitionKind = canStepMockAr ? 'mock_ar_step' : null;
    } else {
      nextTransitionKind = canInitDiffusionBlock ? 'diffusion_block_init' : null;
    }
  }

  return Object.freeze({
    lifecyclePhase: record.lifecyclePhase,
    isTerminated,
    terminationReason,
    committedPrefixLen,
    kvCapacity,
    remainingKvCapacity,
    pendingBlockSize,
    hasActiveBlock,
    activeBlockStatus,
    isAtBranchTail,
    willAutoForkOnMutatingStep,
    canStepAr,
    canStepMockAr,
    canInitDiffusionBlock,
    canRunDiffusionRefinement,
    canCommitRefinementConfidence,
    canCommitActiveBlockToContext,
    canStepNext: nextTransitionKind !== null,
    nextTransitionKind,
    canNavigateBack: branch.cursorIndex > 0,
    canNavigateForward: branch.cursorIndex < branch.records.length - 1,
    canForkBranch: session.bounds.maxBranches >= 2,
    canDeleteAnyBranch: session.branches.length > 1,
  });
}

// ============================================================================
// SESSION CREATION, CONFIG UPDATE & RESTART
// ============================================================================

export function createTwoTowerSession(
  model: TwoTowerModelParameters,
  options?: CreateTwoTowerSessionOptions,
): TwoTowerSession {
  const sessionId = options?.sessionId ?? DEFAULT_TWOTOWER_SESSION_ID;
  if (typeof sessionId !== 'string' || sessionId.length === 0) {
    throw new RangeError('sessionId must be a non-empty string');
  }

  const vocabSize = model.contextModel.config.vocabSize;
  const kvCapacity = options?.kvCapacity ?? TWOTOWER_DEFAULT_KV_CAPACITY;
  const promptTokenIds = options?.promptTokenIds ?? DEFAULT_TWOTOWER_PROMPT_TOKENS;
  const bounds = validateBounds(
    options?.maxBranches ?? DEFAULT_TWOTOWER_SESSION_MAX_BRANCHES,
    options?.maxRecords ?? DEFAULT_TWOTOWER_SESSION_MAX_RECORDS,
  );
  const traceRetention = validateTraceRetention(model, options);

  const defaultConfig: TwoTowerSessionConfig = Object.freeze({
    mode: 'mask_diffusion',
    blockSize: TWOTOWER_DEFAULT_BLOCK_SIZE,
    stepsPerBlock: TWOTOWER_DEFAULT_STEPS_PER_BLOCK,
    confidenceThreshold: TWOTOWER_DEFAULT_CONFIDENCE_THRESHOLD,
    temperature: TWOTOWER_DEFAULT_TEMPERATURE,
    eosTokenIds: Object.freeze([model.eosTokenId]),
    samplingSeed: DEFAULT_TWOTOWER_SESSION_SAMPLING_SEED,
  });
  const pendingConfig = validateAndMergeConfig(
    defaultConfig,
    {
      mode: options?.mode,
      blockSize: options?.blockSize,
      stepsPerBlock: options?.stepsPerBlock,
      confidenceThreshold: options?.confidenceThreshold,
      temperature: options?.temperature,
      eosTokenIds: options?.eosTokenIds,
      samplingSeed: options?.samplingSeed,
    },
    vocabSize,
  );

  const initialBranchId = 'branch-0';
  const prefillExec = runTwoTowerContextPrefillWithTrace(model, promptTokenIds, {
    kvCapacity,
    promptTokenLabels: options?.promptTokenLabels,
    branchId: initialBranchId,
    retainedTraceBays: traceRetention.retainedTraceBays,
    maxRetainedTraces: traceRetention.maxRetainedTraces,
  });

  const snapshot = prefillExec.snapshot;
  const P = snapshot.ctxState.processedTokens;
  const isCapacityExhausted = P >= kvCapacity;
  const termination: TwoTowerSessionTermination | null = isCapacityExhausted
    ? Object.freeze({
        reason: 'capacity_exhausted' as const,
        committedPrefixLen: P,
        kvCapacity,
      })
    : null;
  const lifecyclePhase: TwoTowerSessionLifecyclePhase = isCapacityExhausted
    ? 'capacity_exhausted'
    : 'ready';

  const identity = buildRecordIdentity({
    model,
    sessionId,
    branchId: initialBranchId,
    parentRecordId: null,
    recordIndex: 0,
    kind: 'context_prefill',
    sourceCacheDigest: prefillExec.prevStateDigest,
    sourceSnapshotDigest: null,
    resultSnapshot: snapshot,
    blockId: null,
    refinementIndex: null,
    absoluteTokenStart: 0,
    absoluteTokenRange: [0, P - 1],
  });

  const prefillRecord: TwoTowerSessionContextPrefillRecord = Object.freeze({
    kind: 'context_prefill',
    identity,
    configAtStep: pendingConfig,
    lifecyclePhase,
    contextSnapshot: snapshot,
    committedPrefixLen: P,
    kvCapacity,
    nextBlockId: 0,
    activeBlock: null,
    termination,
    contextExecution: prefillExec,
    conditioning: null,
    denoiserPass: null,
    promptTokenIds: prefillExec.committedTokenIds,
    promptTokenLabels: prefillExec.committedTokenLabels,
  });

  const branch0: TwoTowerSessionBranch = Object.freeze({
    branchId: initialBranchId,
    label: options?.initialBranchLabel ?? 'Main',
    parentBranchId: null,
    forkPointRecordId: null,
    forkPointIndex: null,
    createdAtSeq: 0,
    lastActiveSeq: 0,
    records: Object.freeze([prefillRecord]),
    cursorIndex: 0,
  });

  return Object.freeze({
    sessionId,
    model,
    kvCapacity,
    promptTokenIds: prefillExec.committedTokenIds,
    promptTokenLabels: prefillExec.committedTokenLabels,
    bounds,
    traceRetention,
    pendingConfig,
    branches: Object.freeze([branch0]),
    activeBranchId: initialBranchId,
    nextBranchOrdinal: 1,
    mutationSeq: 0,
  });
}

/**
 * Update `session.pendingConfig` cleanly without mutating the prior session, any historical record,
 * or any already-initialized in-flight diffusion block's frozen `blockConfig`.
 */
export function updateTwoTowerSessionConfig(
  session: TwoTowerSession,
  patch: TwoTowerSessionConfigPatch,
): TwoTowerSession {
  const vocabSize = session.model.contextModel.config.vocabSize;
  const nextPendingConfig = validateAndMergeConfig(session.pendingConfig, patch, vocabSize);
  return Object.freeze({
    ...session,
    pendingConfig: nextPendingConfig,
  });
}

export function restartTwoTowerSession(
  session: TwoTowerSession,
  options?: RestartTwoTowerSessionOptions,
): TwoTowerSession {
  const vocabSize = session.model.contextModel.config.vocabSize;
  const mergedConfig = validateAndMergeConfig(
    session.pendingConfig,
    options?.configPatch,
    vocabSize,
  );
  const nextPromptIds = options?.promptTokenIds ?? session.promptTokenIds;
  const nextPromptLabels =
    options?.promptTokenLabels !== undefined
      ? options.promptTokenLabels
      : options?.promptTokenIds !== undefined
        ? undefined
        : session.promptTokenLabels;

  return createTwoTowerSession(session.model, {
    sessionId: options?.sessionId ?? session.sessionId,
    promptTokenIds: nextPromptIds,
    promptTokenLabels: nextPromptLabels,
    kvCapacity: options?.kvCapacity ?? session.kvCapacity,
    mode: mergedConfig.mode,
    blockSize: mergedConfig.blockSize,
    stepsPerBlock: mergedConfig.stepsPerBlock,
    confidenceThreshold: mergedConfig.confidenceThreshold,
    temperature: mergedConfig.temperature,
    eosTokenIds: mergedConfig.eosTokenIds,
    samplingSeed: mergedConfig.samplingSeed,
    maxBranches: options?.maxBranches ?? session.bounds.maxBranches,
    maxRecords: options?.maxRecords ?? session.bounds.maxRecords,
    retainedTraceBays: options?.retainedTraceBays ?? session.traceRetention.retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces ?? session.traceRetention.maxRetainedTraces,
    retainEagerTracesOnHistory:
      options?.retainEagerTracesOnHistory ?? session.traceRetention.retainEagerTracesOnHistory,
    initialBranchLabel: options?.initialBranchLabel,
  });
}

// ============================================================================
// LIFECYCLE TRANSITIONS (AR, MOCK-AR, DIFFUSION INIT / REFINE / COMMIT)
// ============================================================================

export function stepTwoTowerSessionAr(
  session: TwoTowerSession,
  options?: StepTwoTowerArOptions,
): TwoTowerSession {
  const target = prepareMutatingTarget(session);
  const { sourceRecord, targetBranchId } = target;

  if (sourceRecord.termination !== null) {
    throw new RangeError(
      `Cannot run AR step on terminated branch (reason: ${sourceRecord.termination.reason})`,
    );
  }
  if (sourceRecord.activeBlock !== null) {
    throw new RangeError(
      `Cannot run AR step while diffusion block ${sourceRecord.activeBlock.blockId} is active (status: ${sourceRecord.activeBlock.status})`,
    );
  }
  const P = sourceRecord.committedPrefixLen;
  if (P + 1 > session.kvCapacity) {
    throw new RangeError(
      `Cannot run AR step: Context prefix (${P} + 1) exceeds kvCapacity (${session.kvCapacity})`,
    );
  }

  const configAtStep: TwoTowerSessionConfig = Object.freeze({
    ...session.pendingConfig,
    mode: 'ar',
  });

  const arRes = runTwoTowerArStep(session.model, sourceRecord.contextSnapshot, {
    nextTokenId: options?.nextTokenId,
    nextTokenLabel: options?.nextTokenLabel,
    branchId: targetBranchId,
    retainedTraceBays: session.traceRetention.retainedTraceBays,
    maxRetainedTraces: session.traceRetention.maxRetainedTraces,
  });

  const nextP = arRes.nextSnapshot.ctxState.processedTokens;
  const eosHits = collectMatchingEosTokens([arRes.tokenId], configAtStep.eosTokenIds);
  let termination: TwoTowerSessionTermination | null = null;
  let lifecyclePhase: TwoTowerSessionLifecyclePhase = 'ready';

  if (eosHits.length > 0) {
    termination = Object.freeze({
      reason: 'eos' as const,
      mode: 'ar' as const,
      committedPrefixLen: nextP,
      blockId: null,
      triggerTokenIds: eosHits,
    });
    lifecyclePhase = 'eos_terminated';
  } else if (nextP >= session.kvCapacity) {
    termination = Object.freeze({
      reason: 'capacity_exhausted' as const,
      committedPrefixLen: nextP,
      kvCapacity: session.kvCapacity,
    });
    lifecyclePhase = 'capacity_exhausted';
  }

  const identity = buildRecordIdentity({
    model: session.model,
    sessionId: session.sessionId,
    branchId: targetBranchId,
    parentRecordId: sourceRecord.identity.recordId,
    recordIndex: sourceRecord.identity.recordIndex + 1,
    kind: 'ar_step',
    sourceCacheDigest: sourceRecord.contextSnapshot.ctxState.stateDigest,
    sourceSnapshotDigest: sourceRecord.contextSnapshot.snapshotDigest,
    resultSnapshot: arRes.nextSnapshot,
    blockId: null,
    refinementIndex: null,
    absoluteTokenStart: P,
    absoluteTokenRange: [P, P],
  });

  const record: TwoTowerSessionArStepRecord = Object.freeze({
    kind: 'ar_step',
    identity,
    configAtStep,
    lifecyclePhase,
    contextSnapshot: arRes.nextSnapshot,
    committedPrefixLen: nextP,
    kvCapacity: session.kvCapacity,
    nextBlockId: sourceRecord.nextBlockId,
    activeBlock: null,
    termination,
    contextExecution: arRes.contextExecution,
    conditioning: null,
    denoiserPass: null,
    selectionLogits: arRes.logits,
    selectionProbs: arRes.probs,
    selectedTokenId: arRes.tokenId,
    selectedTokenLabel: arRes.tokenLabel,
    wasExplicitTokenOverride: options?.nextTokenId !== undefined,
  });

  return commitAppendedRecord(session, target, record);
}

export function stepTwoTowerSessionMockAr(
  session: TwoTowerSession,
  options?: StepTwoTowerMockArOptions,
): TwoTowerSession {
  const target = prepareMutatingTarget(session);
  const { sourceRecord, targetBranchId } = target;

  if (sourceRecord.termination !== null) {
    throw new RangeError(
      `Cannot run Mock-AR step on terminated branch (reason: ${sourceRecord.termination.reason})`,
    );
  }
  if (sourceRecord.activeBlock !== null) {
    throw new RangeError(
      `Cannot run Mock-AR step while diffusion block ${sourceRecord.activeBlock.blockId} is active (status: ${sourceRecord.activeBlock.status})`,
    );
  }
  const P = sourceRecord.committedPrefixLen;
  if (P + 1 > session.kvCapacity) {
    throw new RangeError(
      `Cannot run Mock-AR step: Context prefix (${P} + 1) exceeds kvCapacity (${session.kvCapacity})`,
    );
  }

  const configAtStep: TwoTowerSessionConfig = Object.freeze({
    ...session.pendingConfig,
    mode: 'mock_ar',
  });

  const mockRes = runTwoTowerMockArStep(session.model, sourceRecord.contextSnapshot, {
    nextTokenId: options?.nextTokenId,
    nextTokenLabel: options?.nextTokenLabel,
    branchId: targetBranchId,
    retainedTraceBays: session.traceRetention.retainedTraceBays,
    maxRetainedTraces: session.traceRetention.maxRetainedTraces,
  });

  const nextP = mockRes.nextSnapshot.ctxState.processedTokens;
  const eosHits = collectMatchingEosTokens([mockRes.tokenId], configAtStep.eosTokenIds);
  let termination: TwoTowerSessionTermination | null = null;
  let lifecyclePhase: TwoTowerSessionLifecyclePhase = 'ready';

  if (eosHits.length > 0) {
    termination = Object.freeze({
      reason: 'eos' as const,
      mode: 'mock_ar' as const,
      committedPrefixLen: nextP,
      blockId: null,
      triggerTokenIds: eosHits,
    });
    lifecyclePhase = 'eos_terminated';
  } else if (nextP >= session.kvCapacity) {
    termination = Object.freeze({
      reason: 'capacity_exhausted' as const,
      committedPrefixLen: nextP,
      kvCapacity: session.kvCapacity,
    });
    lifecyclePhase = 'capacity_exhausted';
  }

  const identity = buildRecordIdentity({
    model: session.model,
    sessionId: session.sessionId,
    branchId: targetBranchId,
    parentRecordId: sourceRecord.identity.recordId,
    recordIndex: sourceRecord.identity.recordIndex + 1,
    kind: 'mock_ar_step',
    sourceCacheDigest: sourceRecord.contextSnapshot.ctxState.stateDigest,
    sourceSnapshotDigest: sourceRecord.contextSnapshot.snapshotDigest,
    resultSnapshot: mockRes.nextSnapshot,
    blockId: null,
    refinementIndex: null,
    absoluteTokenStart: P,
    absoluteTokenRange: [P, P],
  });

  const record: TwoTowerSessionMockArStepRecord = Object.freeze({
    kind: 'mock_ar_step',
    identity,
    configAtStep,
    lifecyclePhase,
    contextSnapshot: mockRes.nextSnapshot,
    committedPrefixLen: nextP,
    kvCapacity: session.kvCapacity,
    nextBlockId: sourceRecord.nextBlockId,
    activeBlock: null,
    termination,
    contextExecution: mockRes.contextExecution,
    conditioning: mockRes.conditioning,
    denoiserPass: mockRes.denoiserPass,
    selectionLogits: new Float32Array(mockRes.denoiserPass.logits),
    selectionProbs: new Float32Array(mockRes.denoiserPass.probs),
    selectedTokenId: mockRes.tokenId,
    selectedTokenLabel: mockRes.tokenLabel,
    wasExplicitTokenOverride: options?.nextTokenId !== undefined,
  });

  return commitAppendedRecord(session, target, record);
}

export function initTwoTowerDiffusionBlock(
  session: TwoTowerSession,
  options?: InitTwoTowerDiffusionBlockOptions,
): TwoTowerSession {
  const target = prepareMutatingTarget(session);
  const { sourceRecord, targetBranchId } = target;

  if (sourceRecord.termination !== null) {
    throw new RangeError(
      `Cannot initialize diffusion block on terminated branch (reason: ${sourceRecord.termination.reason})`,
    );
  }
  if (sourceRecord.activeBlock !== null) {
    throw new RangeError(
      `Cannot initialize a new diffusion block while block ${sourceRecord.activeBlock.blockId} is still active (status: ${sourceRecord.activeBlock.status})`,
    );
  }

  const blockSize = validateBlockSize(options?.blockSize ?? session.pendingConfig.blockSize);
  const stepsPerBlock = validateStepsPerBlock(
    options?.stepsPerBlock ?? session.pendingConfig.stepsPerBlock,
  );
  const P = sourceRecord.committedPrefixLen;
  if (P + blockSize > session.kvCapacity) {
    throw new RangeError(
      `Cannot initialize diffusion block: ${P} + ${blockSize} = ${P + blockSize} exceeds kvCapacity (${session.kvCapacity})`,
    );
  }

  const vocabSize = session.model.denoiserModel.config.vocabSize;
  let initialBlockTokens: readonly number[];
  if (options?.initialBlockTokens !== undefined) {
    if (options.initialBlockTokens.length !== blockSize) {
      throw new RangeError(
        `initialBlockTokens length (${options.initialBlockTokens.length}) !== blockSize (${blockSize})`,
      );
    }
    const validated = new Array<number>(blockSize);
    for (let i = 0; i < blockSize; i++) {
      const id = options.initialBlockTokens[i];
      if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
        throw new RangeError(
          `initialBlockTokens[${i}] must be an integer in [0, ${vocabSize - 1}], received ${id}`,
        );
      }
      validated[i] = id;
    }
    initialBlockTokens = Object.freeze(validated);
  } else {
    initialBlockTokens = Object.freeze(
      new Array<number>(blockSize).fill(session.model.maskTokenId),
    );
  }

  const initialBlockLabels = resolveLabelsForTokens(initialBlockTokens, session.model.vocab);
  let nMasked = 0;
  for (let i = 0; i < blockSize; i++) {
    if (initialBlockTokens[i] === session.model.maskTokenId) nMasked++;
  }

  const blockId = sourceRecord.nextBlockId;
  const blockConfig: TwoTowerActiveBlockConfig = Object.freeze({
    blockId,
    blockSize,
    stepsPerBlock,
  });

  // Allocate the read-only ~78 KiB conditioning payload ONCE for this entire block
  const conditioningBase = buildBlockConditioningBase(
    session.model,
    sourceRecord.contextSnapshot,
    blockId,
  );

  const configAtStep: TwoTowerSessionConfig = Object.freeze({
    ...session.pendingConfig,
    mode: 'mask_diffusion',
    blockSize,
    stepsPerBlock,
  });

  const pendingEosTokenIds = collectMatchingEosTokens(
    initialBlockTokens,
    configAtStep.eosTokenIds,
  );
  const alreadyComplete = nMasked === 0;
  const status: TwoTowerActiveBlockStatus = alreadyComplete
    ? 'completed_uncommitted'
    : 'initialized';
  const lifecyclePhase: TwoTowerSessionLifecyclePhase = alreadyComplete
    ? 'completed_uncommitted'
    : 'block_initialized';

  const absoluteTokenRange = Object.freeze([P, P + blockSize - 1] as const);
  const activeBlock: TwoTowerActiveDiffusionBlock = Object.freeze({
    blockId,
    status,
    blockConfig,
    absoluteTokenStart: P,
    absoluteTokenRange,
    conditioningBase,
    currentTokens: initialBlockTokens,
    currentTokenLabels: initialBlockLabels,
    nMasked,
    completedRefinements: 0,
    pendingRefinementIndex: null,
    pendingConditioning: null,
    pendingDenoiserPass: null,
    latestUnmaskStep: null,
    latestSamplerPresentation: null,
    contextCommitted: false as const,
    pendingEosInBlock: pendingEosTokenIds.length > 0,
    pendingEosTokenIds,
    nfe: 0,
    earlyCompleted: alreadyComplete,
  });

  const identity = buildRecordIdentity({
    model: session.model,
    sessionId: session.sessionId,
    branchId: targetBranchId,
    parentRecordId: sourceRecord.identity.recordId,
    recordIndex: sourceRecord.identity.recordIndex + 1,
    kind: 'diffusion_block_init',
    sourceCacheDigest: sourceRecord.contextSnapshot.ctxState.stateDigest,
    sourceSnapshotDigest: sourceRecord.contextSnapshot.snapshotDigest,
    resultSnapshot: sourceRecord.contextSnapshot,
    blockId,
    refinementIndex: null,
    absoluteTokenStart: P,
    absoluteTokenRange,
  });

  const record: TwoTowerSessionDiffusionBlockInitRecord = Object.freeze({
    kind: 'diffusion_block_init',
    identity,
    configAtStep,
    lifecyclePhase,
    contextSnapshot: sourceRecord.contextSnapshot,
    committedPrefixLen: P,
    kvCapacity: session.kvCapacity,
    nextBlockId: sourceRecord.nextBlockId,
    activeBlock,
    termination: null,
    contextExecution: null,
    conditioning: null,
    denoiserPass: null,
    blockConfig,
    conditioningBase,
    initialBlockTokens,
    initialBlockLabels,
  });

  return commitAppendedRecord(session, target, record);
}

export function runTwoTowerDiffusionRefinement(
  session: TwoTowerSession,
  options?: RunTwoTowerDiffusionRefinementOptions,
): TwoTowerSession {
  const target = prepareMutatingTarget(session);
  const { sourceRecord, targetBranchId } = target;

  if (sourceRecord.termination !== null) {
    throw new RangeError(
      `Cannot run diffusion refinement on terminated branch (reason: ${sourceRecord.termination.reason})`,
    );
  }
  const activeBlock = sourceRecord.activeBlock;
  if (activeBlock === null) {
    throw new RangeError(
      'Cannot run diffusion refinement: no active diffusion block (call initTwoTowerDiffusionBlock first)',
    );
  }
  if (activeBlock.status === 'awaiting_confidence_commit') {
    throw new RangeError(
      `Cannot run diffusion refinement: refinement ${activeBlock.pendingRefinementIndex} is awaiting confidence commit`,
    );
  }
  if (activeBlock.status === 'completed_uncommitted' || activeBlock.nMasked === 0) {
    throw new RangeError(
      `Cannot run diffusion refinement: block ${activeBlock.blockId} is already completed and awaiting Context commit`,
    );
  }

  const r = activeBlock.completedRefinements;
  const { blockId, blockSize, stepsPerBlock } = activeBlock.blockConfig;
  if (r >= stepsPerBlock) {
    throw new RangeError(
      `Cannot run diffusion refinement ${r}: exceeds block step budget stepsPerBlock (${stepsPerBlock})`,
    );
  }

  // Share the immutable ~78 KiB conditioning payload via lightweight refinement view
  const conditioning = deriveRefinementConditioningView(activeBlock.conditioningBase, r);
  const denoiserPass = runTwoTowerDenoiserPass(
    session.model,
    conditioning,
    activeBlock.currentTokens,
    {
      tOverride: options?.tOverride,
      tokenLabels: activeBlock.currentTokenLabels,
      retainedTraceBays: session.traceRetention.retainedTraceBays,
      maxRetainedTraces: session.traceRetention.maxRetainedTraces,
    },
  );

  // Leave xt unchanged so Denoiser pass is individually inspectable prior to sampler unmasking
  const nextActiveBlock: TwoTowerActiveDiffusionBlock = Object.freeze({
    ...activeBlock,
    status: 'awaiting_confidence_commit' as const,
    pendingRefinementIndex: r,
    pendingConditioning: conditioning,
    pendingDenoiserPass: denoiserPass,
    nfe: activeBlock.nfe + 1,
  });

  // Isolate pending logits/probs so mutating denoiserPass.logits before confidence commit cannot corrupt sampler
  privatePendingRefinementCache.set(nextActiveBlock, {
    logits: new Float32Array(denoiserPass.logits),
    probs: new Float32Array(denoiserPass.probs),
  });

  const configAtStep: TwoTowerSessionConfig = Object.freeze({
    ...session.pendingConfig,
    mode: 'mask_diffusion',
    blockSize,
    stepsPerBlock,
  });

  const identity = buildRecordIdentity({
    model: session.model,
    sessionId: session.sessionId,
    branchId: targetBranchId,
    parentRecordId: sourceRecord.identity.recordId,
    recordIndex: sourceRecord.identity.recordIndex + 1,
    kind: 'diffusion_refinement',
    sourceCacheDigest: sourceRecord.contextSnapshot.ctxState.stateDigest,
    sourceSnapshotDigest: sourceRecord.contextSnapshot.snapshotDigest,
    resultSnapshot: sourceRecord.contextSnapshot,
    blockId,
    refinementIndex: r,
    absoluteTokenStart: activeBlock.absoluteTokenStart,
    absoluteTokenRange: activeBlock.absoluteTokenRange,
  });

  const record: TwoTowerSessionDiffusionRefinementRecord = Object.freeze({
    kind: 'diffusion_refinement',
    identity,
    configAtStep,
    lifecyclePhase: 'refinement_denoised',
    contextSnapshot: sourceRecord.contextSnapshot,
    committedPrefixLen: sourceRecord.committedPrefixLen,
    kvCapacity: session.kvCapacity,
    nextBlockId: sourceRecord.nextBlockId,
    activeBlock: nextActiveBlock,
    termination: null,
    contextExecution: null,
    conditioning,
    denoiserPass,
    blockConfig: activeBlock.blockConfig,
    conditioningBase: activeBlock.conditioningBase,
    refinementIndex: r,
    xtInput: activeBlock.currentTokens,
    xtInputLabels: activeBlock.currentTokenLabels,
    tModel: denoiserPass.tModel ?? activeBlock.nMasked / blockSize,
  });

  return commitAppendedRecord(session, target, record);
}

export function commitTwoTowerRefinementConfidence(
  session: TwoTowerSession,
  options?: CommitTwoTowerRefinementConfidenceOptions,
): TwoTowerSession {
  const target = prepareMutatingTarget(session);
  const { sourceRecord, targetBranchId } = target;

  if (sourceRecord.termination !== null) {
    throw new RangeError(
      `Cannot commit refinement confidence on terminated branch (reason: ${sourceRecord.termination.reason})`,
    );
  }
  const activeBlock = sourceRecord.activeBlock;
  if (
    activeBlock === null ||
    activeBlock.status !== 'awaiting_confidence_commit' ||
    activeBlock.pendingRefinementIndex === null ||
    activeBlock.pendingConditioning === null ||
    activeBlock.pendingDenoiserPass === null
  ) {
    throw new RangeError(
      'Cannot commit refinement confidence: no denoised refinement awaiting confidence commit',
    );
  }

  const r = activeBlock.pendingRefinementIndex;
  const { blockId, blockSize, stepsPerBlock } = activeBlock.blockConfig;
  const vocabSize = session.model.denoiserModel.config.vocabSize;

  const confidenceThreshold = validateConfidenceThreshold(
    options?.confidenceThreshold ?? session.pendingConfig.confidenceThreshold,
  );
  const temperature = validateTemperature(
    options?.temperature ?? session.pendingConfig.temperature,
  );
  const samplingSeed = validateSamplingSeed(
    options?.samplingSeed ?? session.pendingConfig.samplingSeed,
  );

  // Resolve deterministic or explicit variate provenance (NEVER Math.random())
  let variateProvenance: TwoTowerVariateProvenance;
  if (temperature <= 0) {
    variateProvenance = Object.freeze({
      mode: 'greedy_none' as const,
      samplingSeed: null,
      derivationKey: null,
      variateDigest: null,
      variates: null,
    });
  } else if (options?.uniformVariates !== undefined) {
    const expectedLen = blockSize * vocabSize;
    if (options.uniformVariates.length !== expectedLen) {
      throw new RangeError(
        `uniformVariates expected length ${expectedLen} (B=${blockSize} * V=${vocabSize}), received ${options.uniformVariates.length}`,
      );
    }
    const clonedVariates = new Float32Array(expectedLen);
    for (let i = 0; i < expectedLen; i++) {
      const u = options.uniformVariates[i];
      if (!Number.isFinite(u) || u < 0 || u >= 1) {
        throw new RangeError(
          `uniformVariates[${i}] must be a finite value in [0, 1), received ${u}`,
        );
      }
      clonedVariates[i] = u;
    }
    variateProvenance = Object.freeze({
      mode: 'caller_explicit' as const,
      samplingSeed: null,
      derivationKey: 'caller_explicit',
      variateDigest: computeExplicitVariateDigest(clonedVariates),
      variates: clonedVariates,
    });
  } else {
    const derived = deriveRefinementUniformVariates({
      samplingSeed,
      sessionId: session.sessionId,
      branchId: targetBranchId,
      blockId,
      refinementIndex: r,
      blockSize,
      vocabSize,
    });
    variateProvenance = Object.freeze({
      mode: 'derived_deterministic' as const,
      samplingSeed,
      derivationKey: derived.derivationKey,
      variateDigest: derived.variateDigest,
      variates: derived.variates,
    });
  }

  const pendingPass = activeBlock.pendingDenoiserPass;
  const pendingConditioning = activeBlock.pendingConditioning;
  const privateBufs = privatePendingRefinementCache.get(activeBlock);
  const safeLogits = privateBufs ? privateBufs.logits : pendingPass.logits;
  const safeProbs = privateBufs ? privateBufs.probs : pendingPass.probs;

  const unmaskStep = runConfidenceUnmaskStep({
    xt: activeBlock.currentTokens,
    logits: safeLogits,
    vocabSize,
    maskTokenId: session.model.maskTokenId,
    stepIndex: r,
    stepsPerBlock,
    confidenceThreshold,
    temperature,
    uniformVariates: variateProvenance.variates ?? undefined,
  });

  const samplerPresentation = buildSamplerStepPresentation({
    rawLogits: safeLogits,
    rawProbs: safeProbs,
    unmaskStep,
    vocab: session.model.vocab,
    maskTokenId: session.model.maskTokenId,
    confidenceThreshold,
    temperature,
    absoluteTokenStart: activeBlock.absoluteTokenStart,
  });

  const configAtStep: TwoTowerSessionConfig = Object.freeze({
    ...session.pendingConfig,
    mode: 'mask_diffusion',
    blockSize,
    stepsPerBlock,
    confidenceThreshold,
    temperature,
    samplingSeed,
  });

  const nextTokens = unmaskStep.xtNext;
  const nextLabels = resolveLabelsForTokens(nextTokens, session.model.vocab);
  const pendingEosTokenIds = collectMatchingEosTokens(nextTokens, configAtStep.eosTokenIds);
  const isComplete = unmaskStep.isBlockComplete;

  const nextStatus: TwoTowerActiveBlockStatus = isComplete
    ? 'completed_uncommitted'
    : 'refining';
  const lifecyclePhase: TwoTowerSessionLifecyclePhase = isComplete
    ? 'completed_uncommitted'
    : 'block_refining';

  const nextActiveBlock: TwoTowerActiveDiffusionBlock = Object.freeze({
    blockId,
    status: nextStatus,
    blockConfig: activeBlock.blockConfig,
    absoluteTokenStart: activeBlock.absoluteTokenStart,
    absoluteTokenRange: activeBlock.absoluteTokenRange,
    conditioningBase: activeBlock.conditioningBase,
    currentTokens: nextTokens,
    currentTokenLabels: nextLabels,
    nMasked: unmaskStep.nMaskedAfter,
    completedRefinements: r + 1,
    pendingRefinementIndex: null,
    pendingConditioning: null,
    pendingDenoiserPass: null,
    latestUnmaskStep: unmaskStep,
    latestSamplerPresentation: samplerPresentation,
    contextCommitted: false as const,
    pendingEosInBlock: pendingEosTokenIds.length > 0,
    pendingEosTokenIds,
    nfe: activeBlock.nfe,
    earlyCompleted: isComplete ? unmaskStep.earlyCompleted : false,
  });

  const identity = buildRecordIdentity({
    model: session.model,
    sessionId: session.sessionId,
    branchId: targetBranchId,
    parentRecordId: sourceRecord.identity.recordId,
    recordIndex: sourceRecord.identity.recordIndex + 1,
    kind: 'diffusion_confidence_commit',
    sourceCacheDigest: sourceRecord.contextSnapshot.ctxState.stateDigest,
    sourceSnapshotDigest: sourceRecord.contextSnapshot.snapshotDigest,
    resultSnapshot: sourceRecord.contextSnapshot,
    blockId,
    refinementIndex: r,
    absoluteTokenStart: activeBlock.absoluteTokenStart,
    absoluteTokenRange: activeBlock.absoluteTokenRange,
  });

  const record: TwoTowerSessionDiffusionConfidenceCommitRecord = Object.freeze({
    kind: 'diffusion_confidence_commit',
    identity,
    configAtStep,
    lifecyclePhase,
    contextSnapshot: sourceRecord.contextSnapshot,
    committedPrefixLen: sourceRecord.committedPrefixLen,
    kvCapacity: session.kvCapacity,
    nextBlockId: sourceRecord.nextBlockId,
    activeBlock: nextActiveBlock,
    termination: null,
    contextExecution: null,
    conditioning: pendingConditioning,
    denoiserPass: pendingPass,
    blockConfig: activeBlock.blockConfig,
    conditioningBase: activeBlock.conditioningBase,
    refinementIndex: r,
    unmaskStep,
    samplerPresentation,
    variateProvenance,
  });

  return commitAppendedRecord(session, target, record);
}

export function commitTwoTowerActiveBlockToContext(
  session: TwoTowerSession,
  options?: CommitTwoTowerActiveBlockOptions,
): TwoTowerSession {
  const target = prepareMutatingTarget(session);
  const { sourceRecord, targetBranchId } = target;

  if (sourceRecord.termination !== null) {
    throw new RangeError(
      `Cannot commit active diffusion block on terminated branch (reason: ${sourceRecord.termination.reason})`,
    );
  }
  const activeBlock = sourceRecord.activeBlock;
  if (
    activeBlock === null ||
    activeBlock.status !== 'completed_uncommitted' ||
    activeBlock.nMasked !== 0
  ) {
    throw new RangeError(
      `Cannot commit active diffusion block: block must be in "completed_uncommitted" state with 0 masked tokens (current status: ${activeBlock ? activeBlock.status : 'none'})`,
    );
  }

  const vocabSize = session.model.contextModel.config.vocabSize;
  const eosTokenIds =
    options?.eosTokenIds !== undefined
      ? normalizeEosTokenIds(options.eosTokenIds, vocabSize)
      : session.pendingConfig.eosTokenIds;

  const { blockId, blockSize, stepsPerBlock } = activeBlock.blockConfig;
  const configAtStep: TwoTowerSessionConfig = Object.freeze({
    ...session.pendingConfig,
    mode: 'mask_diffusion',
    blockSize,
    stepsPerBlock,
    eosTokenIds,
  });

  const commitExec = commitTwoTowerBlockToContextWithTrace(
    session.model,
    sourceRecord.contextSnapshot,
    activeBlock.currentTokens,
    {
      blockTokenLabels: activeBlock.currentTokenLabels,
      branchId: targetBranchId,
      retainedTraceBays: session.traceRetention.retainedTraceBays,
      maxRetainedTraces: session.traceRetention.maxRetainedTraces,
    },
  );

  const nextSnapshot = commitExec.snapshot;
  const nextP = nextSnapshot.ctxState.processedTokens;
  const eosHits = collectMatchingEosTokens(activeBlock.currentTokens, eosTokenIds);
  const stoppedByEos = blockContainsEos(activeBlock.currentTokens, eosTokenIds);

  let termination: TwoTowerSessionTermination | null = null;
  let lifecyclePhase: TwoTowerSessionLifecyclePhase = 'ready';

  if (stoppedByEos) {
    termination = Object.freeze({
      reason: 'eos' as const,
      mode: 'mask_diffusion' as const,
      committedPrefixLen: nextP,
      blockId,
      triggerTokenIds: eosHits,
    });
    lifecyclePhase = 'eos_terminated';
  } else if (nextP >= session.kvCapacity) {
    termination = Object.freeze({
      reason: 'capacity_exhausted' as const,
      committedPrefixLen: nextP,
      kvCapacity: session.kvCapacity,
    });
    lifecyclePhase = 'capacity_exhausted';
  }

  const identity = buildRecordIdentity({
    model: session.model,
    sessionId: session.sessionId,
    branchId: targetBranchId,
    parentRecordId: sourceRecord.identity.recordId,
    recordIndex: sourceRecord.identity.recordIndex + 1,
    kind: 'diffusion_block_commit',
    sourceCacheDigest: sourceRecord.contextSnapshot.ctxState.stateDigest,
    sourceSnapshotDigest: sourceRecord.contextSnapshot.snapshotDigest,
    resultSnapshot: nextSnapshot,
    blockId,
    refinementIndex: null,
    absoluteTokenStart: activeBlock.absoluteTokenStart,
    absoluteTokenRange: activeBlock.absoluteTokenRange,
  });

  const record: TwoTowerSessionDiffusionBlockCommitRecord = Object.freeze({
    kind: 'diffusion_block_commit',
    identity,
    configAtStep,
    lifecyclePhase,
    contextSnapshot: nextSnapshot,
    committedPrefixLen: nextP,
    kvCapacity: session.kvCapacity,
    nextBlockId: blockId + 1,
    activeBlock: null,
    termination,
    contextExecution: commitExec,
    conditioning: null,
    denoiserPass: null,
    blockConfig: activeBlock.blockConfig,
    committedBlockTokens: activeBlock.currentTokens,
    committedBlockLabels: activeBlock.currentTokenLabels,
    nfe: activeBlock.nfe,
    earlyCompleted: activeBlock.earlyCompleted,
    stoppedByEos,
  });

  return commitAppendedRecord(session, target, record);
}

/**
 * Execute the single next legal transition determined by `session.pendingConfig.mode` and the
 * active record's `lifecyclePhase`.
 */
export function stepTwoTowerSessionNext(session: TwoTowerSession): TwoTowerSession {
  const caps = getSessionCapabilities(session);
  switch (caps.nextTransitionKind) {
    case 'ar_step':
      return stepTwoTowerSessionAr(session);
    case 'mock_ar_step':
      return stepTwoTowerSessionMockAr(session);
    case 'diffusion_block_init':
      return initTwoTowerDiffusionBlock(session);
    case 'diffusion_refinement':
      return runTwoTowerDiffusionRefinement(session);
    case 'diffusion_confidence_commit':
      return commitTwoTowerRefinementConfidence(session);
    case 'diffusion_block_commit':
      return commitTwoTowerActiveBlockToContext(session);
    default: {
      if (caps.isTerminated) {
        throw new RangeError(
          `Cannot advance terminated session (reason: ${caps.terminationReason})`,
        );
      }
      if (!caps.isAtBranchTail && session.bounds.maxBranches < 2) {
        throw new RangeError(
          `Cannot step from rewound cursor when maxBranches is ${session.bounds.maxBranches}`,
        );
      }
      throw new RangeError(
        `No legal next transition for mode "${session.pendingConfig.mode}" in lifecyclePhase "${caps.lifecyclePhase}" (remainingKvCapacity=${caps.remainingKvCapacity}, pendingBlockSize=${caps.pendingBlockSize})`,
      );
    }
  }
}

// ============================================================================
// NAVIGATION, BRANCHING & TRACE PRUNING
// ============================================================================

export function navigateTwoTowerSession(
  session: TwoTowerSession,
  cursorIndex: number,
): TwoTowerSession {
  const branch = getActiveBranch(session);
  if (!Number.isInteger(cursorIndex) || cursorIndex < 0 || cursorIndex >= branch.records.length) {
    throw new RangeError(
      `cursorIndex (${cursorIndex}) out of bounds [0..${branch.records.length - 1}] for branch "${branch.branchId}"`,
    );
  }
  const nextSeq = session.mutationSeq + 1;
  const updatedBranch: TwoTowerSessionBranch = Object.freeze({
    ...branch,
    cursorIndex,
    lastActiveSeq: nextSeq,
  });
  const nextBranches = Object.freeze(
    session.branches.map((b) => (b.branchId === branch.branchId ? updatedBranch : b)),
  );
  return Object.freeze({
    ...session,
    branches: nextBranches,
    mutationSeq: nextSeq,
  });
}

export function navigateTwoTowerSessionBack(session: TwoTowerSession): TwoTowerSession {
  const branch = getActiveBranch(session);
  if (branch.cursorIndex <= 0) {
    throw new RangeError(
      `Cannot navigate back from cursorIndex 0 on branch "${branch.branchId}"`,
    );
  }
  return navigateTwoTowerSession(session, branch.cursorIndex - 1);
}

export function navigateTwoTowerSessionForward(session: TwoTowerSession): TwoTowerSession {
  const branch = getActiveBranch(session);
  if (branch.cursorIndex >= branch.records.length - 1) {
    throw new RangeError(
      `Cannot navigate forward past tail cursorIndex ${branch.cursorIndex} on branch "${branch.branchId}"`,
    );
  }
  return navigateTwoTowerSession(session, branch.cursorIndex + 1);
}

export function forkTwoTowerSessionBranch(
  session: TwoTowerSession,
  options?: ForkTwoTowerSessionBranchOptions,
): TwoTowerSession {
  if (session.bounds.maxBranches < 2) {
    throw new RangeError(
      `Cannot fork branch: maxBranches (${session.bounds.maxBranches}) must be >= 2`,
    );
  }
  const sourceBranch = getActiveBranch(session);
  const fromCursorIndex = options?.fromCursorIndex ?? sourceBranch.cursorIndex;
  if (
    !Number.isInteger(fromCursorIndex) ||
    fromCursorIndex < 0 ||
    fromCursorIndex >= sourceBranch.records.length
  ) {
    throw new RangeError(
      `fromCursorIndex (${fromCursorIndex}) out of bounds [0..${sourceBranch.records.length - 1}]`,
    );
  }

  const vocabSize = session.model.contextModel.config.vocabSize;
  const nextPendingConfig =
    options?.configPatch !== undefined
      ? validateAndMergeConfig(session.pendingConfig, options.configPatch, vocabSize)
      : session.pendingConfig;

  const ordinal = session.nextBranchOrdinal;
  const newBranchId = `branch-${ordinal}`;
  const forkRecord = sourceBranch.records[fromCursorIndex];
  const nextSeq = session.mutationSeq + 1;
  const switchToNewBranch = options?.switchToNewBranch !== false;

  const newBranch: TwoTowerSessionBranch = Object.freeze({
    branchId: newBranchId,
    label: options?.label ?? `Branch ${ordinal}`,
    parentBranchId: sourceBranch.branchId,
    forkPointRecordId: forkRecord.identity.recordId,
    forkPointIndex: fromCursorIndex,
    createdAtSeq: nextSeq,
    lastActiveSeq: nextSeq,
    records: Object.freeze(sourceBranch.records.slice(0, fromCursorIndex + 1)),
    cursorIndex: fromCursorIndex,
  });

  const nextActiveBranchId = switchToNewBranch ? newBranchId : session.activeBranchId;
  const boundedBranches = enforceSessionBounds({
    branches: [...session.branches, newBranch],
    activeBranchId: nextActiveBranchId,
    protectedBranchIds: new Set([newBranchId, sourceBranch.branchId]),
    bounds: session.bounds,
  });

  return Object.freeze({
    ...session,
    pendingConfig: nextPendingConfig,
    branches: boundedBranches,
    activeBranchId: nextActiveBranchId,
    nextBranchOrdinal: ordinal + 1,
    mutationSeq: nextSeq,
  });
}

export function switchTwoTowerSessionBranch(
  session: TwoTowerSession,
  branchId: string,
): TwoTowerSession {
  const target = session.branches.find((b) => b.branchId === branchId);
  if (!target) {
    throw new RangeError(`Cannot switch to unknown branch "${branchId}"`);
  }
  const nextSeq = session.mutationSeq + 1;
  const nextBranches = Object.freeze(
    session.branches.map((b) =>
      b.branchId === branchId ? Object.freeze({ ...b, lastActiveSeq: nextSeq }) : b,
    ),
  );
  return Object.freeze({
    ...session,
    branches: nextBranches,
    activeBranchId: branchId,
    mutationSeq: nextSeq,
  });
}

export function deleteTwoTowerSessionBranch(
  session: TwoTowerSession,
  branchId: string,
): TwoTowerSession {
  const exists = session.branches.some((b) => b.branchId === branchId);
  if (!exists) {
    throw new RangeError(`Cannot delete unknown branch "${branchId}"`);
  }
  if (branchId === session.activeBranchId) {
    throw new RangeError(`Cannot delete active branch "${branchId}" (switch branches first)`);
  }
  if (session.branches.length <= 1) {
    throw new RangeError('Cannot delete the only remaining branch in a session');
  }
  const nextBranches = Object.freeze(session.branches.filter((b) => b.branchId !== branchId));
  return Object.freeze({
    ...session,
    branches: nextBranches,
    mutationSeq: session.mutationSeq + 1,
  });
}

/**
 * Release eager and LRU-cached per-bay simulator traces on all non-active historical records
 * across all branches while preserving 0-ULP deterministic lazy reconstruction on demand.
 */
export function pruneInactiveSessionTraces(session: TwoTowerSession): TwoTowerSession {
  const activeRecord = getActiveRecord(session);
  releaseTracesExceptActive(session.branches, activeRecord);
  return Object.freeze({
    ...session,
    mutationSeq: session.mutationSeq + 1,
  });
}

// ============================================================================
// EXACT DEDUPLICATED MEMORY ACCOUNTING (`Set<ArrayBufferLike>`)
// ============================================================================

function addUniqueBufferBytes(
  seen: Set<ArrayBufferLike>,
  arr: ArrayBufferView | null | undefined,
): number {
  if (!arr || seen.has(arr.buffer)) return 0;
  seen.add(arr.buffer);
  return arr.buffer.byteLength;
}

export function measureTwoTowerSessionRetainedBytes(
  session: TwoTowerSession,
): TwoTowerSessionMemoryBreakdown {
  const uniqueRecords = new Set<TwoTowerSessionRecord>();
  const uniqueContextSnapshots = new Set<TwoTowerContextSnapshot>();
  const uniqueBlockBases = new Set<TwoTowerBlockConditioningBase>();
  const uniqueConditionings = new Set<TwoTowerContextConditioningState>();
  const uniqueContextExecutions = new Set<TwoTowerContextExecutionResult>();
  const uniqueDenoiserPasses = new Set<TwoTowerDenoiserPassResult>();
  const denoiserPassConditionings = new Map<
    TwoTowerDenoiserPassResult,
    TwoTowerContextConditioningState
  >();

  for (let i = 0; i < session.branches.length; i++) {
    const recs = session.branches[i].records;
    for (let j = 0; j < recs.length; j++) {
      const rec = recs[j];
      if (uniqueRecords.has(rec)) continue;
      uniqueRecords.add(rec);
      uniqueContextSnapshots.add(rec.contextSnapshot);
      if (rec.contextExecution !== null) {
        uniqueContextExecutions.add(rec.contextExecution);
      }
      if (rec.conditioning !== null) {
        uniqueConditionings.add(rec.conditioning);
      }
      if (rec.denoiserPass !== null) {
        uniqueDenoiserPasses.add(rec.denoiserPass);
        if (rec.conditioning !== null) {
          denoiserPassConditionings.set(rec.denoiserPass, rec.conditioning);
        }
      }
      if (rec.activeBlock !== null) {
        uniqueBlockBases.add(rec.activeBlock.conditioningBase);
        uniqueConditionings.add(rec.activeBlock.conditioningBase.baseConditioning);
      }
    }
  }

  const seen = new Set<ArrayBufferLike>();

  // 1. Persistent Context snapshots
  let contextSnapshotBytes = 0;
  for (const snap of uniqueContextSnapshots) {
    contextSnapshotBytes += addUniqueBufferBytes(seen, snap.ctxState.prefixTokenIds);
    for (const m of snap.ctxState.mambaStates.values()) {
      contextSnapshotBytes +=
        addUniqueBufferBytes(seen, m.convState) + addUniqueBufferBytes(seen, m.ssmState);
    }
    for (const g of snap.ctxState.gqaCaches.values()) {
      contextSnapshotBytes +=
        addUniqueBufferBytes(seen, g.K) + addUniqueBufferBytes(seen, g.V);
    }
    for (const m of snap.mambaS2.values()) {
      contextSnapshotBytes +=
        addUniqueBufferBytes(seen, m.convState) + addUniqueBufferBytes(seen, m.ssmState);
    }
    contextSnapshotBytes +=
      addUniqueBufferBytes(seen, snap.arLogits) + addUniqueBufferBytes(seen, snap.arProbs);
  }

  // 2. Cross-tower read-only conditioning buffers (deduplicated across refinement views)
  let blockConditioningBytes = 0;
  for (const cond of uniqueConditionings) {
    for (const m of cond.mambaSeedStates.values()) {
      blockConditioningBytes +=
        addUniqueBufferBytes(seen, m.convState) + addUniqueBufferBytes(seen, m.ssmState);
    }
    for (const g of cond.gqaPrefixCaches.values()) {
      blockConditioningBytes +=
        addUniqueBufferBytes(seen, g.K) + addUniqueBufferBytes(seen, g.V);
    }
  }

  // 3. Context Tower segment trace handles (private clones + retained bay traces)
  let contextTraceBytes = 0;
  for (const exec of uniqueContextExecutions) {
    contextTraceBytes += exec.retainedTypedArrayBytes();
  }

  // 4. Denoiser Tower pass handles (deduplicating execution.ts's shared private conditioning clone
  //    across refinements that share the exact same gqaPrefixCaches Map identity)
  let denoiserTraceBytes = 0;
  const seenPrivateCondMapKeys = new Set<object>();
  for (const pass of uniqueDenoiserPasses) {
    let passBytes = pass.retainedTypedArrayBytes();
    const cond = denoiserPassConditionings.get(pass);
    if (cond !== undefined) {
      if (seenPrivateCondMapKeys.has(cond.gqaPrefixCaches)) {
        passBytes = Math.max(0, passBytes - measureConditioningTypedArrayBytes(cond));
      } else {
        seenPrivateCondMapKeys.add(cond.gqaPrefixCaches);
      }
    }
    denoiserTraceBytes += passBytes;
  }

  // 5. Sampler, variate provenance, and AR/Mock-AR selection buffers
  let samplerAndSelectionBytes = 0;
  for (const rec of uniqueRecords) {
    if (rec.kind === 'ar_step' || rec.kind === 'mock_ar_step') {
      samplerAndSelectionBytes +=
        addUniqueBufferBytes(seen, rec.selectionLogits) +
        addUniqueBufferBytes(seen, rec.selectionProbs);
    } else if (rec.kind === 'diffusion_confidence_commit') {
      samplerAndSelectionBytes +=
        addUniqueBufferBytes(seen, rec.unmaskStep.confidence) +
        addUniqueBufferBytes(seen, rec.unmaskStep.logXTheta) +
        addUniqueBufferBytes(seen, rec.unmaskStep.unscaledProbs) +
        addUniqueBufferBytes(seen, rec.variateProvenance.variates);
    }
    if (rec.activeBlock !== null) {
      const priv = privatePendingRefinementCache.get(rec.activeBlock);
      if (priv) {
        samplerAndSelectionBytes +=
          addUniqueBufferBytes(seen, priv.logits) + addUniqueBufferBytes(seen, priv.probs);
      }
    }
  }

  const totalRetainedBytes =
    contextSnapshotBytes +
    blockConditioningBytes +
    contextTraceBytes +
    denoiserTraceBytes +
    samplerAndSelectionBytes;
  const sharedModelWeightBytes = session.model.retainedWeightBytes();

  return Object.freeze({
    totalRetainedBytes,
    contextSnapshotBytes,
    blockConditioningBytes,
    contextTraceBytes,
    denoiserTraceBytes,
    samplerAndSelectionBytes,
    sharedModelWeightBytes,
    totalIncludingModelWeightsBytes: totalRetainedBytes + sharedModelWeightBytes,
    uniqueRecordCount: uniqueRecords.size,
    branchCount: session.branches.length,
    uniqueContextSnapshotCount: uniqueContextSnapshots.size,
    uniqueBlockConditioningCount: uniqueBlockBases.size,
  });
}
