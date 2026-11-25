/**
 * Pure Headless TwoTower Execution Engine & Block-Diffusion Orchestrator
 * (`src/twotower/execution.ts`)
 *
 * Implements all three Nemotron-Labs-TwoTower execution modes over the shared verified miniature
 * kernels (`runMambaSim`, `runGqaSim`, `runMoeSim`, `runIncrementalDecodeStep`, `applyRmsNorm`,
 * `embedTokens`, `runUntiedLmHead`) without modifying Nano v1.0.0:
 *   1. `'ar'`: Causal Context Tower prefill + single-token/block commits.
 *   2. `'mock_ar'`: Denoiser Tower runs `S = 1` causal pass on `lastTokenId` seeded from Context's
 *      `KV[:-1]` + `mambaS2`, bypassing all adaLN modulation (`tModel = null`, `tEmb = null`).
 *      Under `perturbationScale = 0`, matches Context AR bit-for-bit (`0` ULP).
 *   3. `'mask_diffusion'`: Each refinement `r = 0..K-1` freshly seeds from the SAME immutable
 *      Context boundary (`KV[0..P-1]` + `S-1` Mamba), applies split adaLN-single conditioning
 *      (`M`/`*` modulate→RMSNorm, `E` RMSNorm→modulate, all 52 layers residual-gated, unmodulated
 *      `norm_f` + untied `lm_head`), runs bidirectional `[B × (P+B)]` GQA attention, discards
 *      scratchpad KV/Mamba state between refinements, and commits the completed block once to
 *      Context Tower.
 *
 * Pure headless TypeScript — zero DOM, WebGL, scene, or render imports; zero `Math.random()`.
 */

import type {
  GqaLayerKvCache,
  HybridDecodeCacheState,
  IncrementalStepProvenance,
  MambaLayerBoundaryState,
} from '../math/cacheContracts';
import { runGqaSim, type GqaSimFixture, type GqaSimResult } from '../math/gqaSim';
import {
  DEFAULT_RETAINED_TRACE_BAYS,
  applyRmsNorm,
  embedTokens,
  runUntiedLmHead,
  type HybridFinalNormResult,
  type HybridModelParameters,
  type HybridPrediction,
} from '../math/hybridFlow';
import {
  MAX_INCREMENTAL_STEP_TOKENS,
  createZeroDecodeCacheState,
  runIncrementalDecodeStep,
  type HybridIncrementalStepResult,
} from '../math/incrementalDecode';
import { runMambaSim, type MambaSimFixture, type MambaSimResult } from '../math/mambaSim';
import { runMoeSim, type MoeSimFixture, type MoeSimResult } from '../math/moeSim';
import {
  buildBlockConditioningBase,
  buildContextConditioningState,
  cloneDecodeCacheState,
  createTwoTowerContextSnapshot,
  deriveRefinementConditioningView,
  validateContextConditioningState,
  validateContextSnapshot,
  type TwoTowerBlockConditioningBase,
  type TwoTowerConditioningMode,
  type TwoTowerContextConditioningState,
  type TwoTowerContextSnapshot,
  type TwoTowerTraceOrigin,
} from './contracts';
import {
  computeGlobalTimestepProjection,
  computeLayerModulationParams,
  type GlobalTimestepProjection,
  type LayerModulationParams,
} from './modulation';
import type { TwoTowerModelParameters } from './parameters';
import {
  blockContainsEos,
  runConfidenceUnmaskStep,
  type ConfidenceUnmaskStepResult,
} from './sampler';
import { TWOTOWER_MINI_CONFIG } from './spec';

export const TWOTOWER_DEFAULT_KV_CAPACITY = TWOTOWER_MINI_CONFIG.kvCapacity;
export const TWOTOWER_MAX_QUERY_LEN = MAX_INCREMENTAL_STEP_TOKENS;
export const TWOTOWER_DEFAULT_BLOCK_SIZE = TWOTOWER_MINI_CONFIG.defaultBlockSize;
export const TWOTOWER_DEFAULT_STEPS_PER_BLOCK = TWOTOWER_MINI_CONFIG.defaultStepsPerBlock;
export const TWOTOWER_DEFAULT_CONFIDENCE_THRESHOLD = TWOTOWER_MINI_CONFIG.provisionalDemoThreshold;
export const TWOTOWER_DEFAULT_TEMPERATURE = 0.0 as const;

export interface TwoTowerContextPrefillOptions {
  readonly kvCapacity?: number;
  readonly promptTokenLabels?: readonly string[];
  readonly branchId?: string;
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
}

export interface TwoTowerCommitBlockOptions {
  readonly blockTokenLabels?: readonly string[];
  readonly branchId?: string;
  readonly parentStepExecutionId?: string | null;
  readonly initialStateDigest?: string;
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
}

export type TwoTowerContextSegmentRole =
  | 'single_segment'
  | 'split_prefix_for_mamba_s2'
  | 'split_final_token';

export type TwoTowerContextTraceOrigin = Extract<
  TwoTowerTraceOrigin,
  'context_computed' | 'context_reconstructed'
>;

export type TwoTowerContextLayerTrace =
  | {
      readonly segmentIndex: number;
      readonly segmentCount: number;
      readonly segmentRole: TwoTowerContextSegmentRole;
      readonly stepExecutionId: string;
      readonly provenance: IncrementalStepProvenance;
      readonly traceOrigin: TwoTowerContextTraceOrigin;
      readonly bayId: number;
      readonly kind: 'M';
      readonly sim: MambaSimResult;
    }
  | {
      readonly segmentIndex: number;
      readonly segmentCount: number;
      readonly segmentRole: TwoTowerContextSegmentRole;
      readonly stepExecutionId: string;
      readonly provenance: IncrementalStepProvenance;
      readonly traceOrigin: TwoTowerContextTraceOrigin;
      readonly bayId: number;
      readonly kind: 'E';
      readonly sim: MoeSimResult;
    }
  | {
      readonly segmentIndex: number;
      readonly segmentCount: number;
      readonly segmentRole: TwoTowerContextSegmentRole;
      readonly stepExecutionId: string;
      readonly provenance: IncrementalStepProvenance;
      readonly traceOrigin: TwoTowerContextTraceOrigin;
      readonly bayId: number;
      readonly kind: '*';
      readonly sim: GqaSimResult;
    };

/**
 * Trace handle for one internal Context Tower incremental execution segment (`single_segment`
 * when `N === 1`, or `split_prefix_for_mamba_s2` + `split_final_token` when `N >= 2`).
 *
 * Retains private, non-exposed boundary states & residual checkpoints so eager traces can be
 * released (`releaseEagerTraces()`) while still supporting 0-ULP bit-exact lazy reconstruction
 * (`traceOrigin: 'context_reconstructed'`) strictly over `model.contextModel`.
 */
export interface TwoTowerContextSegmentTraceHandle {
  readonly segmentIndex: number;
  readonly segmentCount: number;
  readonly segmentRole: TwoTowerContextSegmentRole;
  readonly stepExecutionId: string;
  readonly provenance: IncrementalStepProvenance;
  /** Inclusive absolute token range `[startPos, endPos]` processed by this segment. */
  readonly tokenRange: readonly [number, number];
  readonly tokens: number;
  readonly prefixTokens: number;
  readonly totalTokens: number;
  readonly tokenIds: readonly number[];
  readonly tokenLabels: readonly string[];
  getLayerTrace(bayId: number): TwoTowerContextLayerTrace;
  getMambaTrace(bayId: number): MambaSimResult;
  getMoeTrace(bayId: number): MoeSimResult;
  getGqaTrace(bayId: number): GqaSimResult;
  getLayerTraceOrigin(bayId: number): TwoTowerContextTraceOrigin | null;
  retainedTraceBayIds(): readonly number[];
  releaseEagerTraces(): void;
  retainedTypedArrayBytes(): number;
}

/**
 * First-class trace-preserving Context Tower execution result returned by
 * `runTwoTowerContextPrefillWithTrace` and `commitTwoTowerBlockToContextWithTrace`.
 */
export interface TwoTowerContextExecutionResult {
  readonly operation: 'prefill' | 'commit_block';
  readonly snapshot: TwoTowerContextSnapshot;
  readonly prevStateDigest: string;
  readonly nextStateDigest: string;
  /** Inclusive absolute token range `[start, end]` committed by this operation. */
  readonly committedTokenRange: readonly [number, number];
  readonly committedTokenIds: readonly number[];
  readonly committedTokenLabels: readonly string[];
  /** Preserves BOTH split segments (`[0..N-2]` + `[N-1]`) when `N >= 2`, or 1 segment when `N === 1`. */
  readonly segments: readonly TwoTowerContextSegmentTraceHandle[];
  getSegmentForAbsolutePosition(absPos: number): TwoTowerContextSegmentTraceHandle;
  releaseEagerTraces(): void;
  retainedTypedArrayBytes(): number;
}

export interface TwoTowerDenoiserPassOptions {
  /** Optional explicit timestep `t` override (strictly `'mask_diffusion'` mode only). */
  readonly tOverride?: number;
  /** Optional custom display labels aligned 1-to-1 with `blockTokenIds`. */
  readonly tokenLabels?: readonly string[];
  /** Bay IDs retained eagerly in trace cache (defaults to `[1, 2, 6, 42, 43, 52]`). */
  readonly retainedTraceBays?: readonly number[];
  /** Maximum number of layer traces retained in bounded LRU cache (defaults to `12`). */
  readonly maxRetainedTraces?: number;
}

export type TwoTowerDenoiserLayerTrace =
  | {
      readonly traceOrigin: 'denoiser_scratch';
      readonly passExecutionId: string;
      readonly conditioningDigest: string;
      readonly blockId: number;
      readonly refinementIndex: number;
      readonly bayId: number;
      readonly kind: 'M';
      readonly modulation: LayerModulationParams | null;
      readonly sim: MambaSimResult;
    }
  | {
      readonly traceOrigin: 'denoiser_scratch';
      readonly passExecutionId: string;
      readonly conditioningDigest: string;
      readonly blockId: number;
      readonly refinementIndex: number;
      readonly bayId: number;
      readonly kind: 'E';
      readonly modulation: LayerModulationParams | null;
      readonly sim: MoeSimResult;
    }
  | {
      readonly traceOrigin: 'denoiser_scratch';
      readonly passExecutionId: string;
      readonly conditioningDigest: string;
      readonly blockId: number;
      readonly refinementIndex: number;
      readonly bayId: number;
      readonly kind: '*';
      readonly modulation: LayerModulationParams | null;
      readonly sim: GqaSimResult;
    };

export interface TwoTowerDenoiserPassResult {
  readonly passExecutionId: string;
  readonly mode: TwoTowerConditioningMode;
  readonly conditioningDigest: string;
  readonly sourceStateDigest: string;
  readonly blockId: number;
  readonly refinementIndex: number;
  readonly blockSize: number;
  readonly committedPrefixLen: number;
  readonly absoluteTokenStart: number;
  readonly tokenIds: Int32Array;
  readonly tokenLabels: readonly string[];
  /** Masked fraction `t` in `'mask_diffusion'`; strictly `null` in `'mock_ar'`. */
  readonly tModel: number | null;
  /** Sinusoidal timestep projection `[phi, tRepr, tEmb]` in `'mask_diffusion'`; `null` in `'mock_ar'`. */
  readonly timestepProjection: GlobalTimestepProjection | null;
  /** Global modulation vector `[3 * d]` in `'mask_diffusion'`; strictly `null` in `'mock_ar'`. */
  readonly tEmb: Float32Array | null;
  /** All `L + 1` (= 53) detached residual checkpoints (`residualStreams[0] = X_embed`, `residualStreams[k] = Y_k`). */
  readonly residualStreams: readonly Float32Array[];
  /** Unmodulated Final RMSNorm (ε = 1e-5) on `residualStreams[L]`. */
  readonly finalNorm: HybridFinalNormResult;
  /** Unshifted Denoiser logits `[B, vocabSize]` (row-major: `logits[j * vocabSize + v]`). */
  readonly logits: Float32Array;
  /** Unshifted Denoiser softmax probabilities `[B, vocabSize]` (row-major: `probs[j * vocabSize + v]`). */
  readonly probs: Float32Array;
  /** Top-5 Denoiser vocabulary candidates per block position `j = 0..B-1`. */
  readonly topKByPosition: readonly (readonly HybridPrediction[])[];
  /** Deterministic per-bay modulation parameters (`null` in `'mock_ar'`). */
  getLayerModulation(bayId: number): LayerModulationParams | null;
  /** Deterministic access to ANY bay's Denoiser scratchpad trace (`1..L`), lazily rebuilt from private snapshots. */
  getLayerTrace(bayId: number): TwoTowerDenoiserLayerTrace;
  getMambaTrace(bayId: number): MambaSimResult;
  getMoeTrace(bayId: number): MoeSimResult;
  getGqaTrace(bayId: number): GqaSimResult;
  retainedTraceBayIds(): readonly number[];
  /** Drop retained simulator traces to bound memory while preserving 0-ULP lazy reconstruction. */
  releaseEagerTraces(): void;
  /** Exact deduplicated typed-array bytes currently retained by this pass (excluding shared weights). */
  retainedTypedArrayBytes(): number;
}

export interface TwoTowerArStepOptions {
  readonly nextTokenId?: number;
  readonly nextTokenLabel?: string;
  readonly branchId?: string;
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
}

export interface TwoTowerArStepResult {
  readonly mode: 'ar';
  readonly tokenId: number;
  readonly tokenLabel: string;
  readonly logits: Float32Array;
  readonly probs: Float32Array;
  readonly prevSnapshot: TwoTowerContextSnapshot;
  readonly nextSnapshot: TwoTowerContextSnapshot;
  readonly contextExecution: TwoTowerContextExecutionResult;
}

export interface TwoTowerMockArStepOptions {
  readonly blockId?: number;
  readonly refinementIndex?: number;
  readonly nextTokenId?: number;
  readonly nextTokenLabel?: string;
  readonly branchId?: string;
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
}

export interface TwoTowerMockArStepResult {
  readonly mode: 'mock_ar';
  readonly tokenId: number;
  readonly tokenLabel: string;
  readonly conditioning: TwoTowerContextConditioningState;
  readonly denoiserPass: TwoTowerDenoiserPassResult;
  readonly prevSnapshot: TwoTowerContextSnapshot;
  readonly nextSnapshot: TwoTowerContextSnapshot;
  readonly contextExecution: TwoTowerContextExecutionResult;
}

export interface TwoTowerDiffusionRefinementRecord {
  readonly blockId: number;
  readonly refinementIndex: number;
  readonly conditioning: TwoTowerContextConditioningState;
  readonly denoiserPass: TwoTowerDenoiserPassResult;
  readonly unmaskStep: ConfidenceUnmaskStepResult;
}

export interface ExecuteTwoTowerDiffusionRefinementInput {
  readonly model: TwoTowerModelParameters;
  /**
   * Either a deduplicated `TwoTowerBlockConditioningBase` (shares the frozen ~78 KiB conditioning
   * payload across refinements of the same block) or a validated `'mask_diffusion'`
   * `TwoTowerContextConditioningState` matching `refinementIndex`.
   */
  readonly conditioningSource: TwoTowerBlockConditioningBase | TwoTowerContextConditioningState;
  readonly xt: readonly number[];
  readonly refinementIndex: number;
  readonly stepsPerBlock: number;
  readonly confidenceThreshold: number;
  readonly temperature: number;
  readonly uniformVariates?: Float32Array | readonly number[];
  readonly tOverride?: number;
  readonly tokenLabels?: readonly string[];
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
}

export interface TwoTowerDiffusionBlockOptions {
  readonly blockId?: number;
  readonly blockSize?: number;
  /** Independently configurable from `blockSize` (supports `K < B`, `K = B`, or `K > B`). */
  readonly stepsPerBlock?: number;
  readonly confidenceThreshold?: number;
  readonly temperature?: number;
  /** Required per refinement step when `temperature > 0` (`[r]` has length `blockSize * vocabSize`). */
  readonly uniformVariatesPerStep?: readonly (Float32Array | readonly number[])[];
  readonly eosTokenIds?: number | readonly number[];
  /**
   * Whether to commit the completed block to Context Tower (defaults strictly to `true`, matching
   * pinned reference block-commit semantics). Terminal optimization is strictly opt-in via `false`.
   */
  readonly commitToContext?: boolean;
  /** Optional custom initial noisy block tokens (defaults to all `model.maskTokenId`). */
  readonly initialBlockTokens?: readonly number[];
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
}

export interface TwoTowerDiffusionBlockResult {
  readonly blockId: number;
  readonly blockSize: number;
  readonly stepsPerBlock: number;
  readonly initialSnapshot: TwoTowerContextSnapshot;
  readonly nextSnapshot: TwoTowerContextSnapshot;
  readonly contextCommitted: boolean;
  /** Trace-preserving Context block commit execution when `contextCommitted` is `true`, else `null`. */
  readonly contextCommitExecution: TwoTowerContextExecutionResult | null;
  readonly refinements: readonly TwoTowerDiffusionRefinementRecord[];
  readonly nfe: number;
  readonly earlyCompleted: boolean;
  readonly finalBlockTokens: readonly number[];
  readonly finalBlockLabels: readonly string[];
  readonly stoppedByEos: boolean;
}

export interface TwoTowerDiffusionGenerationOptions {
  readonly numBlocks?: number;
  readonly blockSize?: number;
  readonly stepsPerBlock?: number;
  readonly kvCapacity?: number;
  readonly promptTokenLabels?: readonly string[];
  readonly confidenceThreshold?: number;
  readonly temperature?: number;
  readonly eosTokenIds?: number | readonly number[];
  readonly uniformVariatesByBlock?: readonly (readonly (Float32Array | readonly number[])[])[];
  /** Strictly opt-in terminal optimization; defaults to `false` (all blocks commit to Context). */
  readonly skipTerminalBlockCommit?: boolean;
  readonly retainedTraceBays?: readonly number[];
  readonly maxRetainedTraces?: number;
}

export interface TwoTowerDiffusionGenerationResult {
  readonly modelId: string;
  readonly twoTowerFingerprint: string;
  readonly promptTokenIds: readonly number[];
  readonly promptTokenLabels: readonly string[];
  readonly initialSnapshot: TwoTowerContextSnapshot;
  readonly finalSnapshot: TwoTowerContextSnapshot;
  readonly blocks: readonly TwoTowerDiffusionBlockResult[];
  readonly generatedTokenIds: readonly number[];
  readonly generatedTokenLabels: readonly string[];
  readonly fullSequenceTokenIds: readonly number[];
  readonly fullSequenceTokenLabels: readonly string[];
  readonly totalNfe: number;
  readonly stoppedByEos: boolean;
  readonly completedBlocks: number;
}

// ============================================================================
// INTERNAL HELPERS (Validation, Transposition, Deterministic Hashing)
// ============================================================================

function validateTokenSequence(
  tokenIds: readonly number[],
  vocabSize: number,
  label: string,
  maxLen: number = TWOTOWER_MAX_QUERY_LEN,
): Int32Array {
  const len = tokenIds.length;
  if (len < 1 || len > maxLen) {
    throw new RangeError(`${label} length must be in [1..${maxLen}], received ${len}`);
  }
  const out = new Int32Array(len);
  for (let i = 0; i < len; i++) {
    const id = tokenIds[i];
    if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
      throw new RangeError(
        `${label}[${i}] must be an integer in [0, ${vocabSize - 1}], received ${id}`,
      );
    }
    out[i] = id;
  }
  return out;
}

function resolveTokenLabels(
  tokenIds: Int32Array,
  vocab: readonly string[],
  customLabels: readonly string[] | undefined,
  label: string,
): readonly string[] {
  if (customLabels !== undefined && customLabels.length !== tokenIds.length) {
    throw new RangeError(
      `${label} length (${customLabels.length}) !== token count (${tokenIds.length})`,
    );
  }
  const out: string[] = [];
  for (let i = 0; i < tokenIds.length; i++) {
    out.push(customLabels?.[i] ?? vocab[tokenIds[i]] ?? `id:${tokenIds[i]}`);
  }
  return Object.freeze(out);
}

function cloneMambaMap(
  src: ReadonlyMap<number, MambaLayerBoundaryState>,
): Map<number, MambaLayerBoundaryState> {
  const out = new Map<number, MambaLayerBoundaryState>();
  for (const [bayId, m] of src.entries()) {
    out.set(
      bayId,
      Object.freeze({
        ...m,
        convState: new Float32Array(m.convState),
        ssmState: new Float32Array(m.ssmState),
        storageDtypeMetadata: Object.freeze({ conv1d: 'bf16' as const, ssm: 'fp32' as const }),
      }),
    );
  }
  return out;
}

function cloneGqaMap(
  src: ReadonlyMap<number, GqaLayerKvCache>,
): Map<number, GqaLayerKvCache> {
  const out = new Map<number, GqaLayerKvCache>();
  for (const [bayId, g] of src.entries()) {
    out.set(
      bayId,
      Object.freeze({
        ...g,
        K: new Float32Array(g.K),
        V: new Float32Array(g.V),
      }),
    );
  }
  return out;
}

function cloneLayerModulation(
  mod: LayerModulationParams | null,
): LayerModulationParams | null {
  if (mod === null) return null;
  return Object.freeze({
    layerIndex: mod.layerIndex,
    shift: new Float32Array(mod.shift),
    scale: new Float32Array(mod.scale),
    gate: new Float32Array(mod.gate),
  });
}

function addUniqueBytes(
  seen: Set<ArrayBufferLike>,
  arr: ArrayBufferView | undefined,
): number {
  if (!arr || seen.has(arr.buffer)) return 0;
  seen.add(arr.buffer);
  return arr.buffer.byteLength;
}

function collectCacheStateUniqueBytes(
  seen: Set<ArrayBufferLike>,
  state: HybridDecodeCacheState | null,
): number {
  if (!state) return 0;
  let bytes = addUniqueBytes(seen, state.prefixTokenIds);
  for (const m of state.mambaStates.values()) {
    bytes += addUniqueBytes(seen, m.convState) + addUniqueBytes(seen, m.ssmState);
  }
  for (const g of state.gqaCaches.values()) {
    bytes += addUniqueBytes(seen, g.K) + addUniqueBytes(seen, g.V);
  }
  return bytes;
}

function collectConditioningMapsUniqueBytes(
  seen: Set<ArrayBufferLike>,
  mambaSeeds: ReadonlyMap<number, MambaLayerBoundaryState>,
  gqaCaches: ReadonlyMap<number, GqaLayerKvCache>,
): number {
  let bytes = 0;
  for (const m of mambaSeeds.values()) {
    bytes += addUniqueBytes(seen, m.convState) + addUniqueBytes(seen, m.ssmState);
  }
  for (const g of gqaCaches.values()) {
    bytes += addUniqueBytes(seen, g.K) + addUniqueBytes(seen, g.V);
  }
  return bytes;
}

function collectSimResultBytes(
  seen: Set<ArrayBufferLike>,
  kind: 'M' | 'E' | '*',
  sim: MambaSimResult | MoeSimResult | GqaSimResult,
): number {
  if (kind === 'M') {
    const s = sim as MambaSimResult;
    return (
      addUniqueBytes(seen, s.tokenIds) +
      addUniqueBytes(seen, s.initialConvState) +
      addUniqueBytes(seen, s.initialSsmState) +
      addUniqueBytes(seen, s.X) +
      addUniqueBytes(seen, s.X_mod) +
      addUniqueBytes(seen, s.X_norm) +
      addUniqueBytes(seen, s.rms_in) +
      addUniqueBytes(seen, s.U) +
      addUniqueBytes(seen, s.z) +
      addUniqueBytes(seen, s.xBC_pre) +
      addUniqueBytes(seen, s.dt_raw) +
      addUniqueBytes(seen, s.conv_buf) +
      addUniqueBytes(seen, s.conv_out) +
      addUniqueBytes(seen, s.xBC) +
      addUniqueBytes(seen, s.x) +
      addUniqueBytes(seen, s.B) +
      addUniqueBytes(seen, s.C) +
      addUniqueBytes(seen, s.A) +
      addUniqueBytes(seen, s.dt) +
      addUniqueBytes(seen, s.dA) +
      addUniqueBytes(seen, s.H_prev) +
      addUniqueBytes(seen, s.H) +
      addUniqueBytes(seen, s.H_history) +
      addUniqueBytes(seen, s.y_ssm) +
      addUniqueBytes(seen, s.y) +
      addUniqueBytes(seen, s.gated) +
      addUniqueBytes(seen, s.rms_gate) +
      addUniqueBytes(seen, s.y_norm_pre) +
      addUniqueBytes(seen, s.y_norm) +
      addUniqueBytes(seen, s.Mamba_out) +
      addUniqueBytes(seen, s.residualGate) +
      addUniqueBytes(seen, s.Y)
    );
  }
  if (kind === 'E') {
    const s = sim as MoeSimResult;
    return (
      addUniqueBytes(seen, s.tokenIds) +
      addUniqueBytes(seen, s.X) +
      addUniqueBytes(seen, s.rms) +
      addUniqueBytes(seen, s.X_norm) +
      addUniqueBytes(seen, s.U) +
      addUniqueBytes(seen, s.U_mod) +
      addUniqueBytes(seen, s.router_logits) +
      addUniqueBytes(seen, s.scores) +
      addUniqueBytes(seen, s.scores_for_choice) +
      addUniqueBytes(seen, s.router_logits_f64) +
      addUniqueBytes(seen, s.router_scores_f64) +
      addUniqueBytes(seen, s.scores_for_choice_f64) +
      addUniqueBytes(seen, s.topk_indices) +
      addUniqueBytes(seen, s.topk_raw_scores) +
      addUniqueBytes(seen, s.topk_norm_probs) +
      addUniqueBytes(seen, s.topk_weights) +
      addUniqueBytes(seen, s.active_h) +
      addUniqueBytes(seen, s.active_a) +
      addUniqueBytes(seen, s.active_y) +
      addUniqueBytes(seen, s.shared_h) +
      addUniqueBytes(seen, s.shared_a) +
      addUniqueBytes(seen, s.y_shared) +
      addUniqueBytes(seen, s.y_routed) +
      addUniqueBytes(seen, s.y_moe) +
      addUniqueBytes(seen, s.residualGate) +
      addUniqueBytes(seen, s.Y)
    );
  }
  const s = sim as GqaSimResult;
  return (
    addUniqueBytes(seen, s.tokenIds) +
    addUniqueBytes(seen, s.keyTokenIds) +
    addUniqueBytes(seen, s.X) +
    addUniqueBytes(seen, s.X_mod) +
    addUniqueBytes(seen, s.X_norm) +
    addUniqueBytes(seen, s.rms) +
    addUniqueBytes(seen, s.U) +
    addUniqueBytes(seen, s.Q) +
    addUniqueBytes(seen, s.K) +
    addUniqueBytes(seen, s.V) +
    addUniqueBytes(seen, s.Scores) +
    addUniqueBytes(seen, s.Softmax) +
    addUniqueBytes(seen, s.V_out) +
    addUniqueBytes(seen, s.Concat) +
    addUniqueBytes(seen, s.Attn_out) +
    addUniqueBytes(seen, s.residualGate) +
    addUniqueBytes(seen, s.Y)
  );
}

function float32BitEqual(a: Float32Array, b: Float32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) return false;
  }
  return true;
}

interface PrivateConditioningEntry {
  readonly mambaMapRef: ReadonlyMap<number, MambaLayerBoundaryState>;
  readonly mambaSeeds: ReadonlyMap<number, MambaLayerBoundaryState>;
  readonly gqaCaches: ReadonlyMap<number, GqaLayerKvCache>;
}

/**
 * Module-private cache of isolated conditioning boundary clones keyed by the validated
 * `gqaPrefixCaches` Map identity. Refinements derived from the same `TwoTowerBlockConditioningBase`
 * share ONE unexposed private clone (zero redundant 78 KiB allocation per refinement) while
 * verifying exact Float32 bit equality so even adversarial mutate-and-rehash cannot read stale data.
 */
const privateConditioningCache = new WeakMap<
  ReadonlyMap<number, GqaLayerKvCache>,
  PrivateConditioningEntry
>();

function conditioningBuffersBitEqual(
  conditioning: TwoTowerContextConditioningState,
  cached: PrivateConditioningEntry,
): boolean {
  if (
    conditioning.mambaSeedStates.size !== cached.mambaSeeds.size ||
    conditioning.gqaPrefixCaches.size !== cached.gqaCaches.size
  ) {
    return false;
  }
  for (const [bayId, m] of conditioning.mambaSeedStates.entries()) {
    const cm = cached.mambaSeeds.get(bayId);
    if (
      !cm ||
      cm.processedTokens !== m.processedTokens ||
      !float32BitEqual(cm.convState, m.convState) ||
      !float32BitEqual(cm.ssmState, m.ssmState)
    ) {
      return false;
    }
  }
  for (const [bayId, g] of conditioning.gqaPrefixCaches.entries()) {
    const cg = cached.gqaCaches.get(bayId);
    if (
      !cg ||
      cg.seqLen !== g.seqLen ||
      cg.maxTokens !== g.maxTokens ||
      !float32BitEqual(cg.K, g.K) ||
      !float32BitEqual(cg.V, g.V)
    ) {
      return false;
    }
  }
  return true;
}

function getOrClonePrivateConditioningBuffers(
  conditioning: TwoTowerContextConditioningState,
): {
  readonly mambaSeeds: ReadonlyMap<number, MambaLayerBoundaryState>;
  readonly gqaCaches: ReadonlyMap<number, GqaLayerKvCache>;
} {
  const cached = privateConditioningCache.get(conditioning.gqaPrefixCaches);
  if (
    cached &&
    cached.mambaMapRef === conditioning.mambaSeedStates &&
    conditioningBuffersBitEqual(conditioning, cached)
  ) {
    return {
      mambaSeeds: cached.mambaSeeds,
      gqaCaches: cached.gqaCaches,
    };
  }
  const mambaSeeds = cloneMambaMap(conditioning.mambaSeedStates);
  const gqaCaches = cloneGqaMap(conditioning.gqaPrefixCaches);
  if (!cached) {
    privateConditioningCache.set(conditioning.gqaPrefixCaches, {
      mambaMapRef: conditioning.mambaSeedStates,
      mambaSeeds,
      gqaCaches,
    });
  }
  return { mambaSeeds, gqaCaches };
}

/** Extract position `t = S - 1` from `runUntiedLmHead`'s `[vocabSize, S]` layout into `[vocabSize]`. */
function extractLastPositionVector(
  colMajor: Float32Array,
  vocabSize: number,
  S: number,
): Float32Array {
  const out = new Float32Array(vocabSize);
  const lastCol = S - 1;
  for (let v = 0; v < vocabSize; v++) {
    out[v] = colMajor[v * S + lastCol];
  }
  return out;
}

/**
 * Pure lossless layout transpose from `runUntiedLmHead`'s `[vocabSize, B]` (`[v * B + j]`)
 * to row-major `[B, vocabSize]` (`[j * vocabSize + v]`), guaranteeing 0-ULP identity with
 * `runUntiedLmHead` while matching `runConfidenceUnmaskStep` and PyTorch `[B, V]` layout.
 */
function transposeVocabBySeqToRowMajor(
  colMajor: Float32Array,
  vocabSize: number,
  B: number,
): Float32Array {
  const out = new Float32Array(B * vocabSize);
  for (let j = 0; j < B; j++) {
    const rowOff = j * vocabSize;
    for (let v = 0; v < vocabSize; v++) {
      out[rowOff + v] = colMajor[v * B + j];
    }
  }
  return out;
}

/** Deterministic greedy argmax with lower-index tie-break (matches `runUntiedLmHead`). */
function greedyArgmax(logits: Float32Array): number {
  let bestV = 0;
  let bestVal = -Infinity;
  for (let v = 0; v < logits.length; v++) {
    const val = logits[v];
    if (val > bestVal) {
      bestVal = val;
      bestV = v;
    }
  }
  return bestV;
}

/** Compute deterministic unambiguous execution ID for a single Denoiser forward pass. */
export function computeDenoiserPassExecutionId(params: {
  readonly twoTowerFingerprint: string;
  readonly mode: TwoTowerConditioningMode;
  readonly conditioningDigest: string;
  readonly sourceStateDigest: string;
  readonly blockId: number;
  readonly refinementIndex: number;
  readonly tModel: number | null;
  readonly tokenIds: Int32Array;
  readonly tokenLabels: readonly string[];
}): string {
  let h1 = 0x811c9dc5 >>> 0;
  let h2 = 0x9e3779b9 >>> 0;
  const dv = new DataView(new ArrayBuffer(8));

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
  const feedF64 = (v: number) => {
    dv.setFloat64(0, Object.is(v, -0) ? 0 : v, true);
    feedU32(dv.getUint32(0, true));
    feedU32(dv.getUint32(4, true));
  };
  const feedStr = (s: string) => {
    feedU32(s.length);
    for (let i = 0; i < s.length; i++) {
      const c = s.charCodeAt(i);
      feedByte(c);
      feedByte(c >>> 8);
    }
  };

  feedStr('twotower-denoiser-pass-v1');
  feedStr(params.twoTowerFingerprint);
  feedStr(params.mode);
  feedStr(params.conditioningDigest);
  feedStr(params.sourceStateDigest);
  feedU32(params.blockId);
  feedU32(params.refinementIndex);
  if (params.tModel !== null) {
    feedByte(1);
    feedF64(params.tModel);
  } else {
    feedByte(0);
  }
  feedU32(params.tokenIds.length);
  for (let i = 0; i < params.tokenIds.length; i++) {
    feedU32(params.tokenIds[i]);
    feedStr(params.tokenLabels[i] ?? '');
  }

  const hashHex = `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
  const tTag = params.tModel === null ? 'none' : params.tModel.toFixed(4);
  return `denoiser:${params.mode}:b${params.blockId}:r${params.refinementIndex}:t${tTag}:${hashHex}`;
}

// ============================================================================
// 1. CONTEXT TOWER PREFILL & BLOCK COMMIT (TRACE-PRESERVING & BACKWARD-COMPAT)
// ============================================================================

function runSingleContextBay(params: {
  readonly contextModel: HybridModelParameters;
  readonly segmentIndex: number;
  readonly segmentCount: number;
  readonly segmentRole: TwoTowerContextSegmentRole;
  readonly stepExecutionId: string;
  readonly provenance: IncrementalStepProvenance;
  readonly traceOrigin: TwoTowerContextTraceOrigin;
  readonly bayId: number;
  readonly X_in: Float32Array;
  readonly S: number;
  readonly tokenIds: Int32Array;
  readonly tokenLabels: readonly string[];
  readonly prefixTokens: number;
  readonly privatePrevState: HybridDecodeCacheState | null;
}): TwoTowerContextLayerTrace {
  const {
    contextModel,
    segmentIndex,
    segmentCount,
    segmentRole,
    stepExecutionId,
    provenance,
    traceOrigin,
    bayId,
    X_in,
    S,
    tokenIds,
    tokenLabels,
    prefixTokens,
    privatePrevState,
  } = params;

  const weights = contextModel.getBayWeights(bayId);
  const eps = contextModel.config.rmsNormEps;
  const P = prefixTokens;
  const clonedX = new Float32Array(X_in);
  const clonedTokenIds = new Int32Array(tokenIds);

  if (weights.kind === 'M') {
    const prevM = P > 0 && privatePrevState ? privatePrevState.mambaStates.get(bayId) : undefined;
    const fixture: MambaSimFixture = {
      X: clonedX,
      gamma_in: weights.gamma_in,
      W_in: weights.W_in,
      w_conv: weights.w_conv,
      b_conv: weights.b_conv,
      A_log: weights.A_log,
      dt_bias: weights.dt_bias,
      D: weights.D,
      gamma_gate: weights.gamma_gate,
      W_out: weights.W_out,
      eps,
      tokenIds: clonedTokenIds,
      tokenLabels,
      ...(P > 0 && prevM
        ? {
            initialConvState: new Float32Array(prevM.convState),
            initialSsmState: new Float32Array(prevM.ssmState),
          }
        : {}),
      startTokenOffset: P,
    };
    return Object.freeze({
      segmentIndex,
      segmentCount,
      segmentRole,
      stepExecutionId,
      provenance,
      traceOrigin,
      bayId,
      kind: 'M' as const,
      sim: runMambaSim(S, fixture),
    });
  }

  if (weights.kind === 'E') {
    const fixture: MoeSimFixture = {
      X: clonedX,
      gamma_in: weights.gamma_in,
      W_router: weights.W_router,
      e_score_correction_bias: weights.e_score_correction_bias,
      W_up: weights.W_up,
      W_down: weights.W_down,
      W_up_s: weights.W_up_s,
      W_down_s: weights.W_down_s,
      rmsNormEps: eps,
      routerPrecisionPolicy: contextModel.config.routerPrecisionPolicy,
      tokenIds: clonedTokenIds,
      tokenLabels,
      startTokenOffset: P,
    };
    return Object.freeze({
      segmentIndex,
      segmentCount,
      segmentRole,
      stepExecutionId,
      provenance,
      traceOrigin,
      bayId,
      kind: 'E' as const,
      sim: runMoeSim(S, fixture),
    });
  }

  const prevG = P > 0 && privatePrevState ? privatePrevState.gqaCaches.get(bayId) : undefined;
  const fixture: GqaSimFixture = {
    X: clonedX,
    gamma: weights.gamma,
    W_Q: weights.W_Q,
    W_K: weights.W_K,
    W_V: weights.W_V,
    W_O: weights.W_O,
    rmsEps: eps,
    tokenIds: clonedTokenIds,
    tokenLabels,
    ...(P > 0 && prevG && privatePrevState
      ? {
          kvCacheIn: {
            seqLen: prevG.seqLen,
            maxTokens: prevG.maxTokens,
            K: new Float32Array(prevG.K),
            V: new Float32Array(prevG.V),
            prefixTokenIds: new Int32Array(privatePrevState.prefixTokenIds),
            prefixTokenLabels: privatePrevState.prefixTokenLabels,
          },
        }
      : {}),
  };
  return Object.freeze({
    segmentIndex,
    segmentCount,
    segmentRole,
    stepExecutionId,
    provenance,
    traceOrigin,
    bayId,
    kind: '*' as const,
    sim: runGqaSim(S, fixture),
  });
}

const contextSegmentByteCollectors = new WeakMap<
  TwoTowerContextSegmentTraceHandle,
  (seen: Set<ArrayBufferLike>) => number
>();

function buildContextSegmentTraceHandle(params: {
  readonly contextModel: HybridModelParameters;
  readonly step: HybridIncrementalStepResult;
  /** Strictly private, unexposed boundary state (`null` when `step.prefixTokens === 0`). */
  readonly privatePrevState: HybridDecodeCacheState | null;
  readonly segmentIndex: number;
  readonly segmentCount: number;
  readonly segmentRole: TwoTowerContextSegmentRole;
  readonly eagerBays: readonly number[];
  readonly maxRetainedTraces?: number;
}): TwoTowerContextSegmentTraceHandle {
  const {
    contextModel,
    step,
    privatePrevState,
    segmentIndex,
    segmentCount,
    segmentRole,
    eagerBays,
  } = params;

  const numLayers = contextModel.config.pattern.length;
  const stepExecutionId = step.stepExecutionId;
  const provenance = step.provenance;
  const tokenRange = provenance.tokenRange;
  const tokens = step.tokens;
  const prefixTokens = step.prefixTokens;
  const totalTokens = step.totalTokens;
  const privateNewTokenIds = new Int32Array(step.newTokenIds);
  const tokenIds = Object.freeze(Array.from(step.newTokenIds));
  const tokenLabels = Object.freeze([...step.newTokenLabels]);

  // Capture detached private residual checkpoints (53 small Float32Arrays) so we do NOT retain
  // the outer HybridIncrementalStepResult and remain 100% immune to trace or snapshot mutations.
  const privateResidualStreams: readonly Float32Array[] = Object.freeze(
    step.residualStreams.map((s) => s.slice()),
  );
  const isolatedPrevState = prefixTokens > 0 ? privatePrevState : null;

  const eagerSet = new Set<number>(eagerBays);
  const maxRetainedTraces = Math.max(eagerSet.size, params.maxRetainedTraces ?? 12);
  const retainedTraces = new Map<number, TwoTowerContextLayerTrace>();

  for (const bayId of step.retainedTraceBayIds()) {
    const tr = step.getLayerTrace(bayId);
    if (tr.kind === 'M') {
      retainedTraces.set(
        bayId,
        Object.freeze({
          segmentIndex,
          segmentCount,
          segmentRole,
          stepExecutionId,
          provenance,
          traceOrigin: 'context_computed' as const,
          bayId,
          kind: 'M' as const,
          sim: tr.sim,
        }),
      );
    } else if (tr.kind === 'E') {
      retainedTraces.set(
        bayId,
        Object.freeze({
          segmentIndex,
          segmentCount,
          segmentRole,
          stepExecutionId,
          provenance,
          traceOrigin: 'context_computed' as const,
          bayId,
          kind: 'E' as const,
          sim: tr.sim,
        }),
      );
    } else {
      retainedTraces.set(
        bayId,
        Object.freeze({
          segmentIndex,
          segmentCount,
          segmentRole,
          stepExecutionId,
          provenance,
          traceOrigin: 'context_computed' as const,
          bayId,
          kind: '*' as const,
          sim: tr.sim,
        }),
      );
    }
  }

  const assertBayId = (bayId: number): void => {
    if (!Number.isInteger(bayId) || bayId < 1 || bayId > numLayers) {
      throw new RangeError(`Bay ID ${bayId} out of bounds [1..${numLayers}]`);
    }
  };

  const getLayerTrace = (bayId: number): TwoTowerContextLayerTrace => {
    assertBayId(bayId);
    const existing = retainedTraces.get(bayId);
    if (existing) {
      retainedTraces.delete(bayId);
      retainedTraces.set(bayId, existing);
      return existing;
    }

    const rebuilt = runSingleContextBay({
      contextModel,
      segmentIndex,
      segmentCount,
      segmentRole,
      stepExecutionId,
      provenance,
      traceOrigin: 'context_reconstructed',
      bayId,
      X_in: privateResidualStreams[bayId - 1],
      S: tokens,
      tokenIds: privateNewTokenIds,
      tokenLabels,
      prefixTokens,
      privatePrevState: isolatedPrevState,
    });

    if (maxRetainedTraces > 0) {
      retainedTraces.set(bayId, rebuilt);
      while (retainedTraces.size > maxRetainedTraces) {
        let evictId: number | null = null;
        for (const k of retainedTraces.keys()) {
          if (!eagerSet.has(k)) {
            evictId = k;
            break;
          }
        }
        if (evictId === null) {
          evictId = retainedTraces.keys().next().value ?? null;
        }
        if (evictId === null || evictId === bayId) break;
        retainedTraces.delete(evictId);
      }
    }
    return rebuilt;
  };

  const getMambaTrace = (bayId: number): MambaSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== 'M') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected Mamba-2 ("M")`);
    }
    return tr.sim;
  };

  const getMoeTrace = (bayId: number): MoeSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== 'E') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected MoE ("E")`);
    }
    return tr.sim;
  };

  const getGqaTrace = (bayId: number): GqaSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== '*') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected GQA ("*")`);
    }
    return tr.sim;
  };

  const getLayerTraceOrigin = (bayId: number): TwoTowerContextTraceOrigin | null => {
    assertBayId(bayId);
    return retainedTraces.get(bayId)?.traceOrigin ?? null;
  };

  const retainedTraceBayIds = (): readonly number[] =>
    Array.from(retainedTraces.keys()).sort((a, b) => a - b);

  const releaseEagerTraces = (): void => {
    retainedTraces.clear();
    eagerSet.clear();
  };

  const collectSegmentBytes = (seen: Set<ArrayBufferLike>): number => {
    let bytes = addUniqueBytes(seen, privateNewTokenIds);
    bytes += collectCacheStateUniqueBytes(seen, isolatedPrevState);
    for (const s of privateResidualStreams) {
      bytes += addUniqueBytes(seen, s);
    }
    for (const tr of retainedTraces.values()) {
      bytes += collectSimResultBytes(seen, tr.kind, tr.sim);
    }
    return bytes;
  };

  const retainedTypedArrayBytes = (): number => collectSegmentBytes(new Set<ArrayBufferLike>());

  const handle: TwoTowerContextSegmentTraceHandle = Object.freeze({
    segmentIndex,
    segmentCount,
    segmentRole,
    stepExecutionId,
    provenance,
    tokenRange,
    tokens,
    prefixTokens,
    totalTokens,
    tokenIds,
    tokenLabels,
    getLayerTrace,
    getMambaTrace,
    getMoeTrace,
    getGqaTrace,
    getLayerTraceOrigin,
    retainedTraceBayIds,
    releaseEagerTraces,
    retainedTypedArrayBytes,
  });

  contextSegmentByteCollectors.set(handle, collectSegmentBytes);
  return handle;
}

function buildContextExecutionResult(params: {
  readonly operation: 'prefill' | 'commit_block';
  readonly snapshot: TwoTowerContextSnapshot;
  readonly prevStateDigest: string;
  readonly committedTokenRange: readonly [number, number];
  readonly committedTokenIds: readonly number[];
  readonly committedTokenLabels: readonly string[];
  readonly segments: readonly TwoTowerContextSegmentTraceHandle[];
}): TwoTowerContextExecutionResult {
  const {
    operation,
    snapshot,
    prevStateDigest,
    committedTokenRange,
    committedTokenIds,
    committedTokenLabels,
    segments,
  } = params;

  const getSegmentForAbsolutePosition = (absPos: number): TwoTowerContextSegmentTraceHandle => {
    if (!Number.isInteger(absPos)) {
      throw new RangeError(`absPos must be an integer, received ${absPos}`);
    }
    const [start, end] = committedTokenRange;
    if (absPos < start || absPos > end) {
      throw new RangeError(
        `Absolute token position ${absPos} out of committed range [${start}..${end}]`,
      );
    }
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      if (absPos >= seg.tokenRange[0] && absPos <= seg.tokenRange[1]) {
        return seg;
      }
    }
    throw new RangeError(
      `No Context segment covers absolute token position ${absPos} within [${start}..${end}]`,
    );
  };

  const releaseEagerTraces = (): void => {
    for (let i = 0; i < segments.length; i++) {
      segments[i].releaseEagerTraces();
    }
  };

  const retainedTypedArrayBytes = (): number => {
    const seen = new Set<ArrayBufferLike>();
    let bytes = 0;
    for (let i = 0; i < segments.length; i++) {
      const collector = contextSegmentByteCollectors.get(segments[i]);
      bytes += collector ? collector(seen) : segments[i].retainedTypedArrayBytes();
    }
    return bytes;
  };

  return Object.freeze({
    operation,
    snapshot,
    prevStateDigest,
    nextStateDigest: snapshot.ctxState.stateDigest,
    committedTokenRange: Object.freeze([committedTokenRange[0], committedTokenRange[1]] as const),
    committedTokenIds: Object.freeze([...committedTokenIds]),
    committedTokenLabels: Object.freeze([...committedTokenLabels]),
    segments: Object.freeze([...segments]),
    getSegmentForAbsolutePosition,
    releaseEagerTraces,
    retainedTypedArrayBytes,
  });
}

/**
 * Trace-preserving Context Tower prefill over `promptTokenIds` (`1 <= S <= 16`, `S <= kvCapacity`).
 *
 * Preserves BOTH internal split segments (`[0..S-2]` + `[S-1]`) when `S >= 2`, or the single
 * segment when `S === 1`, with private isolated boundary states for deterministic 0-ULP lazy
 * bay trace reconstruction strictly over `model.contextModel`.
 */
export function runTwoTowerContextPrefillWithTrace(
  model: TwoTowerModelParameters,
  promptTokenIds: readonly number[],
  options?: TwoTowerContextPrefillOptions,
): TwoTowerContextExecutionResult {
  const ctxModel = model.contextModel;
  const vocabSize = ctxModel.config.vocabSize;
  const validIds = validateTokenSequence(promptTokenIds, vocabSize, 'promptTokenIds');
  const S = validIds.length;

  const kvCapacity = options?.kvCapacity ?? TWOTOWER_DEFAULT_KV_CAPACITY;
  if (!Number.isInteger(kvCapacity) || kvCapacity < S) {
    throw new RangeError(
      `kvCapacity (${kvCapacity}) must be an integer >= prompt length (${S})`,
    );
  }

  const labels = resolveTokenLabels(
    validIds,
    ctxModel.vocab,
    options?.promptTokenLabels,
    'promptTokenLabels',
  );
  const branchId = options?.branchId ?? 'twotower-context';
  const retainedTraceBays = options?.retainedTraceBays ?? DEFAULT_RETAINED_TRACE_BAYS;
  const committedTokenIds = Array.from(validIds);

  if (S === 1) {
    const zeroState = createZeroDecodeCacheState(ctxModel, {
      maxTokens: kvCapacity,
      precisionMetadata: 'bf16',
    });
    const mambaS2 = cloneMambaMap(zeroState.mambaStates);
    const step = runIncrementalDecodeStep(ctxModel, {
      prevState: zeroState,
      newTokenIds: [validIds[0]],
      newTokenLabels: [labels[0]],
      branchId,
      stepIndex: 0,
      retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    });
    const snapshot = createTwoTowerContextSnapshot(ctxModel, {
      ctxState: step.nextState,
      mambaS2,
      arLogits: extractLastPositionVector(step.logits, vocabSize, 1),
      arProbs: extractLastPositionVector(step.probs, vocabSize, 1),
    });
    const seg0 = buildContextSegmentTraceHandle({
      contextModel: ctxModel,
      step,
      privatePrevState: null,
      segmentIndex: 0,
      segmentCount: 1,
      segmentRole: 'single_segment',
      eagerBays: retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    });
    return buildContextExecutionResult({
      operation: 'prefill',
      snapshot,
      prevStateDigest: zeroState.stateDigest,
      committedTokenRange: [0, 0],
      committedTokenIds,
      committedTokenLabels: labels,
      segments: [seg0],
    });
  }

  // S >= 2: advance prefix [0..S-2] to capture exact mambaS2, then step token [S-1].
  // Because Nano's incremental decode guarantees 0-ULP chunk invariance, this produces
  // bit-identical ctxState/arLogits/arProbs to a single S-token prefill while preserving BOTH
  // segment trace handles cleanly.
  const prefixIds = Array.from(validIds.subarray(0, S - 1));
  const prefixLabels = labels.slice(0, S - 1);
  const headStep = runIncrementalDecodeStep(ctxModel, {
    prevState: null,
    maxTokens: kvCapacity,
    precisionMetadata: 'bf16',
    newTokenIds: prefixIds,
    newTokenLabels: prefixLabels,
    branchId,
    stepIndex: 0,
    retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces,
  });
  const mambaS2 = cloneMambaMap(headStep.nextState.mambaStates);

  const tailStep = runIncrementalDecodeStep(ctxModel, {
    prevState: headStep.nextState,
    newTokenIds: [validIds[S - 1]],
    newTokenLabels: [labels[S - 1]],
    branchId,
    parentStepExecutionId: headStep.stepExecutionId,
    initialStateDigest: headStep.provenance.initialStateDigest,
    stepIndex: 1,
    retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces,
  });

  const snapshot = createTwoTowerContextSnapshot(ctxModel, {
    ctxState: tailStep.nextState,
    mambaS2,
    arLogits: extractLastPositionVector(tailStep.logits, vocabSize, 1),
    arProbs: extractLastPositionVector(tailStep.probs, vocabSize, 1),
  });

  const seg0 = buildContextSegmentTraceHandle({
    contextModel: ctxModel,
    step: headStep,
    privatePrevState: null,
    segmentIndex: 0,
    segmentCount: 2,
    segmentRole: 'split_prefix_for_mamba_s2',
    eagerBays: retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces,
  });
  // `headStep.nextState` is strictly private (never exposed on `snapshot` and non-aliased from `mambaS2`).
  const seg1 = buildContextSegmentTraceHandle({
    contextModel: ctxModel,
    step: tailStep,
    privatePrevState: headStep.nextState,
    segmentIndex: 1,
    segmentCount: 2,
    segmentRole: 'split_final_token',
    eagerBays: retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces,
  });

  return buildContextExecutionResult({
    operation: 'prefill',
    snapshot,
    prevStateDigest: headStep.provenance.prevStateDigest,
    committedTokenRange: [0, S - 1],
    committedTokenIds,
    committedTokenLabels: labels,
    segments: [seg0, seg1],
  });
}

/**
 * Prefill the causal Context Tower on `promptTokenIds` (`1 <= S <= 16`, `S <= kvCapacity`).
 *
 * - Handles `S === 1` cleanly without crashing: `mambaS2` is initialized to zero boundary states
 *   (`processedTokens = 0`) and `ctxState` holds the 1-token causal step.
 * - For `S >= 2`: runs `prompt[0..S-2]` (capturing exact `mambaS2`), then advances by the final
 *   token `prompt[S-1]` to produce exact `ctxState`, `arLogits`, and `arProbs`.
 */
export function runTwoTowerContextPrefill(
  model: TwoTowerModelParameters,
  promptTokenIds: readonly number[],
  options?: TwoTowerContextPrefillOptions,
): TwoTowerContextSnapshot {
  return runTwoTowerContextPrefillWithTrace(model, promptTokenIds, {
    ...options,
    retainedTraceBays: options?.retainedTraceBays ?? [],
  }).snapshot;
}

/**
 * Trace-preserving causal block commit (`1 <= B <= 16`) onto Context Tower without mutating
 * `contextSnapshot`.
 *
 * Preserves BOTH split segments (`[P..P+B-2]` + `[P+B-1]`) when `B >= 2`, or 1 segment when
 * `B === 1`, isolating Segment 0's private boundary state via `cloneDecodeCacheState` so external
 * mutations to public `contextSnapshot.ctxState` cannot corrupt historical trace reconstruction.
 */
export function commitTwoTowerBlockToContextWithTrace(
  model: TwoTowerModelParameters,
  contextSnapshot: TwoTowerContextSnapshot,
  blockTokenIds: readonly number[],
  options?: TwoTowerCommitBlockOptions,
): TwoTowerContextExecutionResult {
  const ctxModel = model.contextModel;
  validateContextSnapshot(contextSnapshot, ctxModel);

  const vocabSize = ctxModel.config.vocabSize;
  const validIds = validateTokenSequence(blockTokenIds, vocabSize, 'blockTokenIds');
  const B = validIds.length;
  const P = contextSnapshot.ctxState.processedTokens;
  const maxTokens = contextSnapshot.ctxState.maxTokens;
  if (P + B > maxTokens) {
    throw new RangeError(
      `Committing block (${P} + ${B} = ${P + B}) exceeds Context KV capacity (${maxTokens})`,
    );
  }

  const labels = resolveTokenLabels(
    validIds,
    ctxModel.vocab,
    options?.blockTokenLabels,
    'blockTokenLabels',
  );
  const branchId = options?.branchId ?? 'twotower-context';
  const retainedTraceBays = options?.retainedTraceBays ?? DEFAULT_RETAINED_TRACE_BAYS;
  const committedTokenIds = Array.from(validIds);
  const prevStateDigest = contextSnapshot.ctxState.stateDigest;

  // Private non-exposed boundary clone isolating Segment 0 reconstruction from public ctxState mutations.
  const privateInitialPrevState = cloneDecodeCacheState(contextSnapshot.ctxState);

  if (B === 1) {
    const mambaS2 = cloneMambaMap(contextSnapshot.ctxState.mambaStates);
    const step = runIncrementalDecodeStep(ctxModel, {
      prevState: contextSnapshot.ctxState,
      newTokenIds: [validIds[0]],
      newTokenLabels: [labels[0]],
      branchId,
      parentStepExecutionId: options?.parentStepExecutionId ?? null,
      initialStateDigest: options?.initialStateDigest,
      stepIndex: P,
      retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    });
    const snapshot = createTwoTowerContextSnapshot(ctxModel, {
      ctxState: step.nextState,
      mambaS2,
      arLogits: extractLastPositionVector(step.logits, vocabSize, 1),
      arProbs: extractLastPositionVector(step.probs, vocabSize, 1),
    });
    const seg0 = buildContextSegmentTraceHandle({
      contextModel: ctxModel,
      step,
      privatePrevState: privateInitialPrevState,
      segmentIndex: 0,
      segmentCount: 1,
      segmentRole: 'single_segment',
      eagerBays: retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    });
    return buildContextExecutionResult({
      operation: 'commit_block',
      snapshot,
      prevStateDigest,
      committedTokenRange: [P, P],
      committedTokenIds,
      committedTokenLabels: labels,
      segments: [seg0],
    });
  }

  // B >= 2: advance B - 1 tokens to capture exact mambaS2, then step the final block token.
  const headIds = Array.from(validIds.subarray(0, B - 1));
  const headLabels = labels.slice(0, B - 1);
  const headStep = runIncrementalDecodeStep(ctxModel, {
    prevState: contextSnapshot.ctxState,
    newTokenIds: headIds,
    newTokenLabels: headLabels,
    branchId,
    parentStepExecutionId: options?.parentStepExecutionId ?? null,
    initialStateDigest: options?.initialStateDigest,
    stepIndex: P,
    retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces,
  });
  const mambaS2 = cloneMambaMap(headStep.nextState.mambaStates);

  const tailStep = runIncrementalDecodeStep(ctxModel, {
    prevState: headStep.nextState,
    newTokenIds: [validIds[B - 1]],
    newTokenLabels: [labels[B - 1]],
    branchId,
    parentStepExecutionId: headStep.stepExecutionId,
    initialStateDigest: headStep.provenance.initialStateDigest,
    stepIndex: P + B - 1,
    retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces,
  });

  const snapshot = createTwoTowerContextSnapshot(ctxModel, {
    ctxState: tailStep.nextState,
    mambaS2,
    arLogits: extractLastPositionVector(tailStep.logits, vocabSize, 1),
    arProbs: extractLastPositionVector(tailStep.probs, vocabSize, 1),
  });

  const seg0 = buildContextSegmentTraceHandle({
    contextModel: ctxModel,
    step: headStep,
    privatePrevState: privateInitialPrevState,
    segmentIndex: 0,
    segmentCount: 2,
    segmentRole: 'split_prefix_for_mamba_s2',
    eagerBays: retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces,
  });
  const seg1 = buildContextSegmentTraceHandle({
    contextModel: ctxModel,
    step: tailStep,
    privatePrevState: headStep.nextState,
    segmentIndex: 1,
    segmentCount: 2,
    segmentRole: 'split_final_token',
    eagerBays: retainedTraceBays,
    maxRetainedTraces: options?.maxRetainedTraces,
  });

  return buildContextExecutionResult({
    operation: 'commit_block',
    snapshot,
    prevStateDigest,
    committedTokenRange: [P, P + B - 1],
    committedTokenIds,
    committedTokenLabels: labels,
    segments: [seg0, seg1],
  });
}

/**
 * Causally commit a completed block (`1 <= B <= 16`) to the Context Tower, returning a fresh
 * immutable `TwoTowerContextSnapshot` without mutating `contextSnapshot`.
 * Always preserves exact `mambaS2` (even for `B >= 2`) so subsequent Mock-AR calls remain 0-ULP exact.
 */
export function commitTwoTowerBlockToContext(
  model: TwoTowerModelParameters,
  contextSnapshot: TwoTowerContextSnapshot,
  blockTokenIds: readonly number[],
  options?: TwoTowerCommitBlockOptions,
): TwoTowerContextSnapshot {
  return commitTwoTowerBlockToContextWithTrace(model, contextSnapshot, blockTokenIds, {
    ...options,
    retainedTraceBays: options?.retainedTraceBays ?? [],
  }).snapshot;
}

// ============================================================================
// 2. DENOISER TOWER FORWARD PASS (`mask_diffusion` & `mock_ar`)
// ============================================================================

function runSingleDenoiserBay(params: {
  readonly model: TwoTowerModelParameters;
  readonly mode: TwoTowerConditioningMode;
  readonly passExecutionId: string;
  readonly conditioningDigest: string;
  readonly blockId: number;
  readonly refinementIndex: number;
  readonly bayId: number;
  readonly X_in: Float32Array;
  readonly B: number;
  readonly tokenIds: Int32Array;
  readonly tokenLabels: readonly string[];
  readonly committedPrefixLen: number;
  readonly absoluteTokenStart: number;
  readonly prefixTokenIds: readonly number[];
  readonly prefixTokenLabels: readonly string[];
  readonly mambaSeeds: ReadonlyMap<number, MambaLayerBoundaryState>;
  readonly gqaCaches: ReadonlyMap<number, GqaLayerKvCache>;
  readonly modulation: LayerModulationParams | null;
}): TwoTowerDenoiserLayerTrace {
  const {
    model,
    mode,
    passExecutionId,
    conditioningDigest,
    blockId,
    refinementIndex,
    bayId,
    X_in,
    B,
    tokenIds,
    tokenLabels,
    committedPrefixLen,
    absoluteTokenStart,
    prefixTokenIds,
    prefixTokenLabels,
    mambaSeeds,
    gqaCaches,
    modulation,
  } = params;

  const denoiser = model.denoiserModel;
  const weights = denoiser.getBayWeights(bayId);
  const eps = denoiser.config.rmsNormEps;
  const clonedX = new Float32Array(X_in);
  const clonedTokenIds = new Int32Array(tokenIds);
  const clonedModulation = cloneLayerModulation(modulation);

  if (weights.kind === 'M') {
    const seed = mambaSeeds.get(bayId);
    if (!seed) {
      throw new RangeError(`Missing Mamba seed state for bay ${bayId}`);
    }
    const fixture: MambaSimFixture = {
      X: clonedX,
      gamma_in: weights.gamma_in,
      W_in: weights.W_in,
      w_conv: weights.w_conv,
      b_conv: weights.b_conv,
      A_log: weights.A_log,
      dt_bias: weights.dt_bias,
      D: weights.D,
      gamma_gate: weights.gamma_gate,
      W_out: weights.W_out,
      eps,
      tokenIds: clonedTokenIds,
      tokenLabels,
      startTokenOffset: absoluteTokenStart,
      ...(committedPrefixLen > 0
        ? {
            initialConvState: new Float32Array(seed.convState),
            initialSsmState: new Float32Array(seed.ssmState),
          }
        : {}),
      ...(modulation !== null
        ? {
            preNormMod: {
              shift: new Float32Array(modulation.shift),
              scale: new Float32Array(modulation.scale),
            },
            residualGate: new Float32Array(modulation.gate),
          }
        : {}),
    };
    return Object.freeze({
      traceOrigin: 'denoiser_scratch' as const,
      passExecutionId,
      conditioningDigest,
      blockId,
      refinementIndex,
      bayId,
      kind: 'M' as const,
      modulation: clonedModulation,
      sim: runMambaSim(B, fixture),
    });
  }

  if (weights.kind === 'E') {
    const fixture: MoeSimFixture = {
      X: clonedX,
      gamma_in: weights.gamma_in,
      W_router: weights.W_router,
      e_score_correction_bias: weights.e_score_correction_bias,
      W_up: weights.W_up,
      W_down: weights.W_down,
      W_up_s: weights.W_up_s,
      W_down_s: weights.W_down_s,
      rmsNormEps: eps,
      routerPrecisionPolicy: denoiser.config.routerPrecisionPolicy,
      tokenIds: clonedTokenIds,
      tokenLabels,
      startTokenOffset: absoluteTokenStart,
      ...(modulation !== null
        ? {
            postNormMod: {
              shift: new Float32Array(modulation.shift),
              scale: new Float32Array(modulation.scale),
            },
            residualGate: new Float32Array(modulation.gate),
          }
        : {}),
    };
    return Object.freeze({
      traceOrigin: 'denoiser_scratch' as const,
      passExecutionId,
      conditioningDigest,
      blockId,
      refinementIndex,
      bayId,
      kind: 'E' as const,
      modulation: clonedModulation,
      sim: runMoeSim(B, fixture),
    });
  }

  const gCache = gqaCaches.get(bayId);
  if (!gCache) {
    throw new RangeError(`Missing GQA KV prefix cache for bay ${bayId}`);
  }
  const fixture: GqaSimFixture = {
    X: clonedX,
    gamma: weights.gamma,
    W_Q: weights.W_Q,
    W_K: weights.W_K,
    W_V: weights.W_V,
    W_O: weights.W_O,
    rmsEps: eps,
    attentionMaskMode: mode === 'mask_diffusion' ? 'bidirectional_block' : 'causal',
    tokenIds: clonedTokenIds,
    tokenLabels,
    ...(committedPrefixLen > 0
      ? {
          kvCacheIn: {
            seqLen: gCache.seqLen,
            maxTokens: gCache.maxTokens,
            K: new Float32Array(gCache.K),
            V: new Float32Array(gCache.V),
            prefixTokenIds,
            prefixTokenLabels,
          },
        }
      : {}),
    ...(modulation !== null
      ? {
          preNormMod: {
            shift: new Float32Array(modulation.shift),
            scale: new Float32Array(modulation.scale),
          },
          residualGate: new Float32Array(modulation.gate),
        }
      : {}),
  };
  return Object.freeze({
    traceOrigin: 'denoiser_scratch' as const,
    passExecutionId,
    conditioningDigest,
    blockId,
    refinementIndex,
    bayId,
    kind: '*' as const,
    modulation: clonedModulation,
    sim: runGqaSim(B, fixture),
  });
}

/**
 * Execute one pure headless Denoiser Tower forward pass (`'mask_diffusion'` or `'mock_ar'`)
 * conditioned strictly read-only on `conditioning`.
 *
 * - Never mutates `conditioning` or any Context snapshot.
 * - Discards all Denoiser scratchpad Mamba states and block KV projections after the pass.
 * - Returns row-major `[B, vocabSize]` unshifted `logits` and `probs` (0-ULP consistent with
 *   `runUntiedLmHead` and directly consumable by `runConfidenceUnmaskStep`).
 */
export function runTwoTowerDenoiserPass(
  model: TwoTowerModelParameters,
  conditioning: TwoTowerContextConditioningState,
  blockTokenIds: readonly number[],
  options?: TwoTowerDenoiserPassOptions,
): TwoTowerDenoiserPassResult {
  const denoiser = model.denoiserModel;
  const { d, vocabSize, pattern, rmsNormEps } = denoiser.config;
  const numLayers = pattern.length;

  const validIds = validateTokenSequence(
    blockTokenIds,
    vocabSize,
    'blockTokenIds',
    TWOTOWER_MAX_QUERY_LEN,
  );
  const B = validIds.length;

  validateContextConditioningState(conditioning, model, B);

  const tokenLabels = resolveTokenLabels(
    validIds,
    denoiser.vocab,
    options?.tokenLabels,
    'tokenLabels',
  );
  const privateTokenIds = new Int32Array(validIds);

  // Capture isolated private boundary clones (deduplicated across refinements sharing a
  // TwoTowerBlockConditioningBase) so post-pass caller mutations to `conditioning` or its
  // Maps/typed arrays cannot corrupt lazy `getLayerTrace(bayId)` reconstruction.
  const { mambaSeeds: privateMambaSeeds, gqaCaches: privateGqaCaches } =
    getOrClonePrivateConditioningBuffers(conditioning);
  const privatePrefixTokenIds = Object.freeze([...conditioning.prefixTokenIds]);
  const privatePrefixTokenLabels = Object.freeze([...conditioning.prefixTokenLabels]);

  let tModel: number | null = null;
  let timestepProjection: GlobalTimestepProjection | null = null;
  let tEmb: Float32Array | null = null;

  if (conditioning.mode === 'mask_diffusion') {
    let nMasked = 0;
    for (let i = 0; i < B; i++) {
      if (validIds[i] === model.maskTokenId) nMasked++;
    }
    const rawT = options?.tOverride !== undefined ? options.tOverride : nMasked / B;
    if (!Number.isFinite(rawT)) {
      throw new RangeError(`Denoiser timestep tModel must be finite, received ${rawT}`);
    }
    tModel = rawT;
    timestepProjection = computeGlobalTimestepProjection(tModel, model.timeConditioning);
    tEmb = timestepProjection.tEmb;
  } else if (options?.tOverride !== undefined) {
    throw new RangeError('tOverride is forbidden in mock_ar mode (all adaLN modulation is bypassed)');
  }

  const passExecutionId = computeDenoiserPassExecutionId({
    twoTowerFingerprint: model.twoTowerFingerprint,
    mode: conditioning.mode,
    conditioningDigest: conditioning.conditioningDigest,
    sourceStateDigest: conditioning.sourceStateDigest,
    blockId: conditioning.blockId,
    refinementIndex: conditioning.refinementIndex,
    tModel,
    tokenIds: privateTokenIds,
    tokenLabels,
  });

  // Precompute strictly unexposed per-layer modulation params (NEVER handed out by reference)
  const privateLayerModulations: (LayerModulationParams | null)[] = new Array(numLayers);
  for (let l = 0; l < numLayers; l++) {
    privateLayerModulations[l] =
      tEmb !== null ? computeLayerModulationParams(tEmb, model.timeConditioning, l) : null;
  }

  // Stage 0: Untied NoPE Token Embedding using Denoiser Tower's own `W_embed`
  const X_embed = embedTokens(privateTokenIds, denoiser.W_embed, d);
  const residualStreams: Float32Array[] = [X_embed.slice()];
  const privateResidualStreams: Float32Array[] = [X_embed.slice()];

  const eagerSet = new Set<number>(options?.retainedTraceBays ?? DEFAULT_RETAINED_TRACE_BAYS);
  const maxRetainedTraces = Math.max(eagerSet.size, options?.maxRetainedTraces ?? 12);
  const retainedTraces = new Map<number, TwoTowerDenoiserLayerTrace>();

  let currentX = privateResidualStreams[0];
  for (let bayId = 1; bayId <= numLayers; bayId++) {
    const mod = privateLayerModulations[bayId - 1];
    const trace = runSingleDenoiserBay({
      model,
      mode: conditioning.mode,
      passExecutionId,
      conditioningDigest: conditioning.conditioningDigest,
      blockId: conditioning.blockId,
      refinementIndex: conditioning.refinementIndex,
      bayId,
      X_in: currentX,
      B,
      tokenIds: privateTokenIds,
      tokenLabels,
      committedPrefixLen: conditioning.committedPrefixLen,
      absoluteTokenStart: conditioning.absoluteTokenStart,
      prefixTokenIds: privatePrefixTokenIds,
      prefixTokenLabels: privatePrefixTokenLabels,
      mambaSeeds: privateMambaSeeds,
      gqaCaches: privateGqaCaches,
      modulation: mod,
    });

    const nextX = trace.sim.Y;
    const isEager = eagerSet.has(bayId);
    const detachedResidual = nextX.slice();
    const privateNextX = isEager ? nextX.slice() : nextX;

    residualStreams.push(detachedResidual);
    privateResidualStreams.push(privateNextX);
    if (isEager) {
      retainedTraces.set(bayId, trace);
    }
    currentX = privateNextX;
  }

  // Stage L+1: Strictly UNMODULATED Final RMSNorm (`eps = 1e-5`)
  const finalX = privateResidualStreams[numLayers];
  const { rms: finalRms, X_norm: finalXNorm, U: finalU } = applyRmsNorm(
    finalX,
    denoiser.finalNormGamma,
    d,
    B,
    rmsNormEps,
  );
  const finalNorm: HybridFinalNormResult = Object.freeze({
    eps: rmsNormEps,
    gamma: denoiser.finalNormGamma,
    rms: finalRms,
    X_norm: finalXNorm,
    U_final: finalU,
  });

  // Stage L+2: Strictly Untied Denoiser LM Head projecting ALL B positions (unshifted [B, V])
  const lmHeadColMajor = runUntiedLmHead(finalU, denoiser.W_lm_head, denoiser.vocab, d, B, 5);
  const logits = transposeVocabBySeqToRowMajor(lmHeadColMajor.logits, vocabSize, B);
  const probs = transposeVocabBySeqToRowMajor(lmHeadColMajor.probs, vocabSize, B);
  const topKByPosition = Object.freeze(
    lmHeadColMajor.topKByToken.map((preds) => Object.freeze([...preds])),
  );

  const exposedTimestepProjection: GlobalTimestepProjection | null =
    timestepProjection !== null
      ? Object.freeze({
          phi: new Float32Array(timestepProjection.phi),
          tRepr: new Float32Array(timestepProjection.tRepr),
          tEmb: new Float32Array(timestepProjection.tEmb),
        })
      : null;
  const exposedTEmb: Float32Array | null = tEmb !== null ? new Float32Array(tEmb) : null;

  const getLayerModulation = (bayId: number): LayerModulationParams | null => {
    if (!Number.isInteger(bayId) || bayId < 1 || bayId > numLayers) {
      throw new RangeError(`Bay ID ${bayId} out of bounds [1..${numLayers}]`);
    }
    return cloneLayerModulation(privateLayerModulations[bayId - 1]);
  };

  const withFreshModulation = (
    trace: TwoTowerDenoiserLayerTrace,
    bayId: number,
  ): TwoTowerDenoiserLayerTrace => {
    const freshMod = cloneLayerModulation(privateLayerModulations[bayId - 1]);
    if (trace.kind === 'M') {
      return Object.freeze({ ...trace, modulation: freshMod });
    }
    if (trace.kind === 'E') {
      return Object.freeze({ ...trace, modulation: freshMod });
    }
    return Object.freeze({ ...trace, modulation: freshMod });
  };

  const getLayerTrace = (bayId: number): TwoTowerDenoiserLayerTrace => {
    if (!Number.isInteger(bayId) || bayId < 1 || bayId > numLayers) {
      throw new RangeError(`Bay ID ${bayId} out of bounds [1..${numLayers}]`);
    }
    const existing = retainedTraces.get(bayId);
    if (existing) {
      retainedTraces.delete(bayId);
      retainedTraces.set(bayId, existing);
      return withFreshModulation(existing, bayId);
    }
    const rebuilt = runSingleDenoiserBay({
      model,
      mode: conditioning.mode,
      passExecutionId,
      conditioningDigest: conditioning.conditioningDigest,
      blockId: conditioning.blockId,
      refinementIndex: conditioning.refinementIndex,
      bayId,
      X_in: privateResidualStreams[bayId - 1],
      B,
      tokenIds: privateTokenIds,
      tokenLabels,
      committedPrefixLen: conditioning.committedPrefixLen,
      absoluteTokenStart: conditioning.absoluteTokenStart,
      prefixTokenIds: privatePrefixTokenIds,
      prefixTokenLabels: privatePrefixTokenLabels,
      mambaSeeds: privateMambaSeeds,
      gqaCaches: privateGqaCaches,
      modulation: privateLayerModulations[bayId - 1],
    });
    if (maxRetainedTraces > 0) {
      retainedTraces.set(bayId, rebuilt);
      while (retainedTraces.size > maxRetainedTraces) {
        let evictId: number | null = null;
        for (const k of retainedTraces.keys()) {
          if (!eagerSet.has(k)) {
            evictId = k;
            break;
          }
        }
        if (evictId === null) {
          evictId = retainedTraces.keys().next().value ?? null;
        }
        if (evictId === null || evictId === bayId) break;
        retainedTraces.delete(evictId);
      }
    }
    return withFreshModulation(rebuilt, bayId);
  };

  const getMambaTrace = (bayId: number): MambaSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== 'M') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected Mamba-2 ("M")`);
    }
    return tr.sim;
  };

  const getMoeTrace = (bayId: number): MoeSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== 'E') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected MoE ("E")`);
    }
    return tr.sim;
  };

  const getGqaTrace = (bayId: number): GqaSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== '*') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected GQA ("*")`);
    }
    return tr.sim;
  };

  const retainedTraceBayIds = (): readonly number[] =>
    Array.from(retainedTraces.keys()).sort((a, b) => a - b);

  const releaseEagerTraces = (): void => {
    retainedTraces.clear();
    eagerSet.clear();
  };

  const retainedTypedArrayBytes = (): number => {
    const seen = new Set<ArrayBufferLike>();
    let bytes =
      addUniqueBytes(seen, validIds) +
      addUniqueBytes(seen, privateTokenIds) +
      addUniqueBytes(seen, exposedTEmb ?? undefined) +
      addUniqueBytes(seen, exposedTimestepProjection?.phi) +
      addUniqueBytes(seen, exposedTimestepProjection?.tRepr) +
      addUniqueBytes(seen, exposedTimestepProjection?.tEmb) +
      addUniqueBytes(seen, finalRms) +
      addUniqueBytes(seen, finalXNorm) +
      addUniqueBytes(seen, finalU) +
      addUniqueBytes(seen, logits) +
      addUniqueBytes(seen, probs);

    for (const mod of privateLayerModulations) {
      if (mod !== null) {
        bytes +=
          addUniqueBytes(seen, mod.shift) +
          addUniqueBytes(seen, mod.scale) +
          addUniqueBytes(seen, mod.gate);
      }
    }
    for (const s of residualStreams) bytes += addUniqueBytes(seen, s);
    for (const s of privateResidualStreams) bytes += addUniqueBytes(seen, s);
    bytes += collectConditioningMapsUniqueBytes(seen, privateMambaSeeds, privateGqaCaches);
    for (const tr of retainedTraces.values()) {
      bytes += collectSimResultBytes(seen, tr.kind, tr.sim);
      if (tr.modulation !== null) {
        bytes +=
          addUniqueBytes(seen, tr.modulation.shift) +
          addUniqueBytes(seen, tr.modulation.scale) +
          addUniqueBytes(seen, tr.modulation.gate);
      }
    }
    return bytes;
  };

  return Object.freeze({
    passExecutionId,
    mode: conditioning.mode,
    conditioningDigest: conditioning.conditioningDigest,
    sourceStateDigest: conditioning.sourceStateDigest,
    blockId: conditioning.blockId,
    refinementIndex: conditioning.refinementIndex,
    blockSize: B,
    committedPrefixLen: conditioning.committedPrefixLen,
    absoluteTokenStart: conditioning.absoluteTokenStart,
    tokenIds: validIds,
    tokenLabels,
    tModel,
    timestepProjection: exposedTimestepProjection,
    tEmb: exposedTEmb,
    residualStreams: Object.freeze(residualStreams),
    finalNorm,
    logits,
    probs,
    topKByPosition,
    getLayerModulation,
    getLayerTrace,
    getMambaTrace,
    getMoeTrace,
    getGqaTrace,
    retainedTraceBayIds,
    releaseEagerTraces,
    retainedTypedArrayBytes,
  });
}

// ============================================================================
// 3. SINGLE-STEP AR & MOCK-AR TRANSITIONS
// ============================================================================

/**
 * Advance Context Tower by one autoregressive token selected greedily from `contextSnapshot.arLogits`
 * (or using an explicit `options.nextTokenId`).
 */
export function runTwoTowerArStep(
  model: TwoTowerModelParameters,
  contextSnapshot: TwoTowerContextSnapshot,
  options?: TwoTowerArStepOptions,
): TwoTowerArStepResult {
  validateContextSnapshot(contextSnapshot, model.contextModel);
  const vocabSize = model.contextModel.config.vocabSize;
  const tokenId =
    options?.nextTokenId !== undefined ? options.nextTokenId : greedyArgmax(contextSnapshot.arLogits);
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
    throw new RangeError(
      `nextTokenId must be an integer in [0, ${vocabSize - 1}], received ${tokenId}`,
    );
  }
  const tokenLabel = options?.nextTokenLabel ?? model.contextModel.vocab[tokenId] ?? `id:${tokenId}`;
  const contextExecution = commitTwoTowerBlockToContextWithTrace(
    model,
    contextSnapshot,
    [tokenId],
    {
      blockTokenLabels: [tokenLabel],
      branchId: options?.branchId,
      retainedTraceBays: options?.retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    },
  );

  return Object.freeze({
    mode: 'ar',
    tokenId,
    tokenLabel,
    logits: new Float32Array(contextSnapshot.arLogits),
    probs: new Float32Array(contextSnapshot.arProbs),
    prevSnapshot: contextSnapshot,
    nextSnapshot: contextExecution.snapshot,
    contextExecution,
  });
}

/**
 * Execute one Mock-AR step:
 *   1. Builds `'mock_ar'` read-only conditioning (`KV[:-1]` + `mambaS2`, `committedPrefixLen = P - 1`).
 *   2. Runs Denoiser causally on `[contextSnapshot.lastTokenId]` with zero adaLN modulation.
 *   3. Selects greedy next token (or `options.nextTokenId`) and commits it to Context Tower.
 */
export function runTwoTowerMockArStep(
  model: TwoTowerModelParameters,
  contextSnapshot: TwoTowerContextSnapshot,
  options?: TwoTowerMockArStepOptions,
): TwoTowerMockArStepResult {
  validateContextSnapshot(contextSnapshot, model.contextModel);
  const vocabSize = model.denoiserModel.config.vocabSize;

  const conditioning = buildContextConditioningState(model, contextSnapshot, {
    mode: 'mock_ar',
    blockId: options?.blockId ?? 0,
    refinementIndex: options?.refinementIndex ?? 0,
  });

  const denoiserPass = runTwoTowerDenoiserPass(
    model,
    conditioning,
    [contextSnapshot.lastTokenId],
    {
      tokenLabels: [contextSnapshot.lastTokenLabel],
      retainedTraceBays: options?.retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    },
  );

  const tokenId =
    options?.nextTokenId !== undefined ? options.nextTokenId : greedyArgmax(denoiserPass.logits);
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
    throw new RangeError(
      `nextTokenId must be an integer in [0, ${vocabSize - 1}], received ${tokenId}`,
    );
  }
  const tokenLabel = options?.nextTokenLabel ?? model.contextModel.vocab[tokenId] ?? `id:${tokenId}`;
  const contextExecution = commitTwoTowerBlockToContextWithTrace(
    model,
    contextSnapshot,
    [tokenId],
    {
      blockTokenLabels: [tokenLabel],
      branchId: options?.branchId,
      retainedTraceBays: options?.retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    },
  );

  return Object.freeze({
    mode: 'mock_ar',
    tokenId,
    tokenLabel,
    conditioning,
    denoiserPass,
    prevSnapshot: contextSnapshot,
    nextSnapshot: contextExecution.snapshot,
    contextExecution,
  });
}

// ============================================================================
// 4. MASKED BLOCK DIFFUSION (SHARED REFINEMENT PRIMITIVE, BLOCK & MULTI-BLOCK)
// ============================================================================

/**
 * Single authoritative masked block-diffusion refinement primitive shared by BOTH
 * `runTwoTowerDiffusionBlock` and interactive `TwoTowerSession` controllers, guaranteeing
 * strictly ONE diffusion algorithm:
 *   1. Resolves read-only `'mask_diffusion'` conditioning for `refinementIndex` (sharing the
 *      frozen ~78 KiB payload when supplied a `TwoTowerBlockConditioningBase`).
 *   2. Runs one bidirectional Denoiser forward pass (`runTwoTowerDenoiserPass`).
 *   3. Runs one confidence-based MDLM unmasking step (`runConfidenceUnmaskStep`).
 */
export function executeTwoTowerDiffusionRefinement(
  input: ExecuteTwoTowerDiffusionRefinementInput,
): TwoTowerDiffusionRefinementRecord {
  const {
    model,
    conditioningSource,
    xt,
    refinementIndex,
    stepsPerBlock,
    confidenceThreshold,
    temperature,
    uniformVariates,
    tOverride,
    tokenLabels,
    retainedTraceBays,
    maxRetainedTraces,
  } = input;

  if (!Number.isInteger(refinementIndex) || refinementIndex < 0) {
    throw new RangeError(
      `refinementIndex must be a nonnegative integer, received ${refinementIndex}`,
    );
  }

  let conditioning: TwoTowerContextConditioningState;
  if ('baseConditioning' in conditioningSource) {
    conditioning = deriveRefinementConditioningView(conditioningSource, refinementIndex);
  } else {
    if (conditioningSource.mode !== 'mask_diffusion') {
      throw new RangeError(
        `executeTwoTowerDiffusionRefinement requires conditioning mode "mask_diffusion", received "${conditioningSource.mode}"`,
      );
    }
    if (conditioningSource.refinementIndex !== refinementIndex) {
      throw new RangeError(
        `conditioning.refinementIndex (${conditioningSource.refinementIndex}) !== refinementIndex (${refinementIndex})`,
      );
    }
    conditioning = conditioningSource;
  }

  const denoiserPass = runTwoTowerDenoiserPass(model, conditioning, xt, {
    tOverride,
    tokenLabels,
    retainedTraceBays,
    maxRetainedTraces,
  });

  const unmaskStep = runConfidenceUnmaskStep({
    xt,
    logits: denoiserPass.logits,
    vocabSize: model.denoiserModel.config.vocabSize,
    maskTokenId: model.maskTokenId,
    stepIndex: refinementIndex,
    stepsPerBlock,
    confidenceThreshold,
    temperature,
    uniformVariates,
  });

  return Object.freeze({
    blockId: conditioning.blockId,
    refinementIndex,
    conditioning,
    denoiserPass,
    unmaskStep,
  });
}

/**
 * Execute a full masked block-diffusion refinement loop over one block of length `B`, seeding
 * EVERY refinement `r = 0..K-1` freshly from the SAME immutable `contextSnapshot` via one shared
 * `TwoTowerBlockConditioningBase` (never from a prior refinement's scratchpad), committing the
 * completed block once to Context Tower by default (`commitToContext !== false`), and evaluating
 * EOS strictly after full block completion.
 */
export function runTwoTowerDiffusionBlock(
  model: TwoTowerModelParameters,
  contextSnapshot: TwoTowerContextSnapshot,
  options?: TwoTowerDiffusionBlockOptions,
): TwoTowerDiffusionBlockResult {
  validateContextSnapshot(contextSnapshot, model.contextModel);

  const blockId = options?.blockId ?? 0;
  if (!Number.isInteger(blockId) || blockId < 0) {
    throw new RangeError(`blockId must be a nonnegative integer, received ${blockId}`);
  }
  const blockSize = options?.blockSize ?? TWOTOWER_DEFAULT_BLOCK_SIZE;
  if (!Number.isInteger(blockSize) || blockSize < 1 || blockSize > TWOTOWER_MAX_QUERY_LEN) {
    throw new RangeError(
      `blockSize must be an integer in [1..${TWOTOWER_MAX_QUERY_LEN}], received ${blockSize}`,
    );
  }
  const stepsPerBlock = options?.stepsPerBlock ?? TWOTOWER_DEFAULT_STEPS_PER_BLOCK;
  if (!Number.isInteger(stepsPerBlock) || stepsPerBlock < 1) {
    throw new RangeError(`stepsPerBlock must be an integer >= 1, received ${stepsPerBlock}`);
  }

  const P = contextSnapshot.ctxState.processedTokens;
  const maxTokens = contextSnapshot.ctxState.maxTokens;
  if (P + blockSize > maxTokens) {
    throw new RangeError(
      `Diffusion block (${P} + ${blockSize} = ${P + blockSize}) exceeds Context KV capacity (${maxTokens})`,
    );
  }

  const confidenceThreshold =
    options?.confidenceThreshold ?? TWOTOWER_DEFAULT_CONFIDENCE_THRESHOLD;
  const temperature = options?.temperature ?? TWOTOWER_DEFAULT_TEMPERATURE;
  const eosTokenIds = options?.eosTokenIds ?? [model.eosTokenId];
  const commitToContext = options?.commitToContext !== false;
  const vocabSize = model.denoiserModel.config.vocabSize;

  let xt: readonly number[];
  if (options?.initialBlockTokens !== undefined) {
    if (options.initialBlockTokens.length !== blockSize) {
      throw new RangeError(
        `initialBlockTokens length (${options.initialBlockTokens.length}) !== blockSize (${blockSize})`,
      );
    }
    const validated = validateTokenSequence(
      options.initialBlockTokens,
      vocabSize,
      'initialBlockTokens',
      TWOTOWER_MAX_QUERY_LEN,
    );
    xt = Object.freeze(Array.from(validated));
  } else {
    xt = Object.freeze(new Array<number>(blockSize).fill(model.maskTokenId));
  }

  // Allocate read-only conditioning base ONCE for the block and reuse across all refinements
  const conditioningBase = buildBlockConditioningBase(model, contextSnapshot, blockId);
  const refinements: TwoTowerDiffusionRefinementRecord[] = [];

  for (let r = 0; r < stepsPerBlock; r++) {
    let nMasked = 0;
    for (let j = 0; j < blockSize; j++) {
      if (xt[j] === model.maskTokenId) nMasked++;
    }
    if (nMasked === 0) {
      break;
    }

    const record = executeTwoTowerDiffusionRefinement({
      model,
      conditioningSource: conditioningBase,
      xt,
      refinementIndex: r,
      stepsPerBlock,
      confidenceThreshold,
      temperature,
      uniformVariates: options?.uniformVariatesPerStep?.[r],
      retainedTraceBays: options?.retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    });

    refinements.push(record);
    xt = record.unmaskStep.xtNext;
    if (record.unmaskStep.isBlockComplete) {
      break;
    }
  }

  const finalBlockTokens = Object.freeze([...xt]);
  const finalBlockLabels = Object.freeze(
    finalBlockTokens.map((id) => model.denoiserModel.vocab[id] ?? `id:${id}`),
  );
  const nfe = refinements.length;
  const earlyCompleted =
    nfe === 0 ? true : refinements[nfe - 1].unmaskStep.earlyCompleted;

  const contextCommitExecution = commitToContext
    ? commitTwoTowerBlockToContextWithTrace(model, contextSnapshot, finalBlockTokens, {
        blockTokenLabels: finalBlockLabels,
        retainedTraceBays: options?.retainedTraceBays,
        maxRetainedTraces: options?.maxRetainedTraces,
      })
    : null;
  const nextSnapshot = contextCommitExecution ? contextCommitExecution.snapshot : contextSnapshot;

  // Post-block EOS halt checked strictly after full block completion
  const stoppedByEos = blockContainsEos(finalBlockTokens, eosTokenIds);

  return Object.freeze({
    blockId,
    blockSize,
    stepsPerBlock,
    initialSnapshot: contextSnapshot,
    nextSnapshot,
    contextCommitted: commitToContext,
    contextCommitExecution,
    refinements: Object.freeze(refinements),
    nfe,
    earlyCompleted,
    finalBlockTokens,
    finalBlockLabels,
    stoppedByEos,
  });
}

/**
 * Pure headless multi-block masked diffusion orchestrator:
 *   Prefill Context -> repeat `numBlocks` times: denoise block -> commit block -> post-block EOS check.
 */
export function generateTwoTowerDiffusion(
  model: TwoTowerModelParameters,
  promptTokenIds: readonly number[],
  options?: TwoTowerDiffusionGenerationOptions,
): TwoTowerDiffusionGenerationResult {
  const numBlocks = options?.numBlocks ?? 2;
  if (!Number.isInteger(numBlocks) || numBlocks < 0) {
    throw new RangeError(`numBlocks must be a nonnegative integer, received ${numBlocks}`);
  }
  const blockSize = options?.blockSize ?? TWOTOWER_DEFAULT_BLOCK_SIZE;
  const stepsPerBlock = options?.stepsPerBlock ?? TWOTOWER_DEFAULT_STEPS_PER_BLOCK;
  const kvCapacity = options?.kvCapacity ?? TWOTOWER_DEFAULT_KV_CAPACITY;
  const skipTerminalBlockCommit = options?.skipTerminalBlockCommit ?? false;

  const initialSnapshot = runTwoTowerContextPrefill(model, promptTokenIds, {
    kvCapacity,
    promptTokenLabels: options?.promptTokenLabels,
  });

  const resolvedPromptIds = Object.freeze(Array.from(initialSnapshot.ctxState.prefixTokenIds));
  const resolvedPromptLabels = Object.freeze([...initialSnapshot.ctxState.prefixTokenLabels]);

  const blocks: TwoTowerDiffusionBlockResult[] = [];
  const generatedTokenIds: number[] = [];
  const generatedTokenLabels: string[] = [];

  let currentSnapshot = initialSnapshot;
  let totalNfe = 0;
  let stoppedByEos = false;

  for (let blockId = 0; blockId < numBlocks; blockId++) {
    const isLastScheduled = blockId === numBlocks - 1;
    const commitToContext = !(skipTerminalBlockCommit && isLastScheduled);

    const blockResult = runTwoTowerDiffusionBlock(model, currentSnapshot, {
      blockId,
      blockSize,
      stepsPerBlock,
      confidenceThreshold: options?.confidenceThreshold,
      temperature: options?.temperature,
      eosTokenIds: options?.eosTokenIds,
      uniformVariatesPerStep: options?.uniformVariatesByBlock?.[blockId],
      commitToContext,
      retainedTraceBays: options?.retainedTraceBays,
      maxRetainedTraces: options?.maxRetainedTraces,
    });

    blocks.push(blockResult);
    totalNfe += blockResult.nfe;
    for (let i = 0; i < blockResult.finalBlockTokens.length; i++) {
      generatedTokenIds.push(blockResult.finalBlockTokens[i]);
      generatedTokenLabels.push(blockResult.finalBlockLabels[i]);
    }
    currentSnapshot = blockResult.nextSnapshot;

    if (blockResult.stoppedByEos) {
      stoppedByEos = true;
      break;
    }
  }

  return Object.freeze({
    modelId: model.modelId,
    twoTowerFingerprint: model.twoTowerFingerprint,
    promptTokenIds: resolvedPromptIds,
    promptTokenLabels: resolvedPromptLabels,
    initialSnapshot,
    finalSnapshot: currentSnapshot,
    blocks: Object.freeze(blocks),
    generatedTokenIds: Object.freeze(generatedTokenIds),
    generatedTokenLabels: Object.freeze(generatedTokenLabels),
    fullSequenceTokenIds: Object.freeze([...resolvedPromptIds, ...generatedTokenIds]),
    fullSequenceTokenLabels: Object.freeze([...resolvedPromptLabels, ...generatedTokenLabels]),
    totalNfe,
    stoppedByEos,
    completedBlocks: blocks.length,
  });
}
