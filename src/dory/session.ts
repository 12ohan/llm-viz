/**
 * Bounded <= 12.00 MiB `DorySession` & `D10` 3.00 MiB LRU Checkpoint Pool (`src/dory/session.ts`).
 *
 * Implements `DorySessionModuleContract` (`D6`, `D10`, `DR18`, `DR19`, `DR23`, `DR24`, Audit §11.2):
 *   - Immutable non-aliased copy-on-write (`cloneDoryKvCache`) branch slabs on every fork and
 *     checkpoint so historical branches can never be mutated or aliased (`DR18`, `DR19`).
 *   - Hard `DR18` / `D6` mid-session depth (`kLoops`) and cache-policy (`per_loop <-> last_loop`)
 *     switch guard: NEVER mutates populated KV slabs — always forks/re-prefills cleanly from
 *     prompt tokens & seed (`DorySessionImmutableConfigError`).
 *   - Hard `DR23` guard: Contract A `self_spec` requires `kvCacheMode === 'per_loop'` and
 *     `kLoops === 4` (`104` persistent banks).
 *   - `D10` `3.00 MiB` (`3,145,728 B`) byte-budgeted LRU sparse-checkpoint pool with active
 *     branch heads pinned and oldest unused historical checkpoints evicted automatically so
 *     12-branch / 128-record sessions stay strictly `<= 12.00 MiB` at BOTH `k = 4` and `k = 8`.
 *   - Exact recorded `chunkSizes[]` per branch replayed verbatim during `0-ULP` lazy step-trace
 *     reconstruction under `last_loop` (never coalescing into single-shot prefill — `DR24`).
 *   - View 2 read-only `Diagnostic Counterfactual` intermediate-loop Coda evaluation
 *     (`writeKv: false`, zero KV/session mutation).
 *   - Deduplicated `Set<ArrayBufferLike>` memory audit helper (`measureDorySessionMemory`).
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL, Node built-ins, ambient RNG, or
 * Gate 4b/4c / sibling-app imports (`D4`).
 */

import { runDoryForwardChunk, runDoryOutputCoda } from './forward';
import { cloneDoryKvCache, createDoryKvCache, measureDoryKvCacheBytes } from './kvCache';
import {
  anchorMapJacobianNorm,
  cosineSimilarity8d,
  l2Distance,
  vectorL2Norm,
} from './ngptMath';
import { getDefaultDoryModelParameters, listDoryCanonicalTensors } from './parameters';
import { createDoryPrng } from './prng';
import { buildDoryDualRopeTables } from './rope';
import { validateDoryRecurrentLoops } from './spec';
import {
  logitsToSamplingProbsFloat64,
  normalizeSamplingConfig,
  prefillDorySelfSpec,
  runDorySelfSpecCycle,
  sampleCategoricalFloat64,
  summarizeSelfSpecCycle,
} from './speculative';
import {
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_DEFAULT_GREEDY_SAMPLING,
  DORY_DEFAULT_SESSION_BOUNDS,
  DORY_DEFAULT_SESSION_PROMPT,
  DORY_DEFAULT_SESSION_SEED,
  DORY_EOS_TOKEN_IDS,
  DORY_SPEC_DEFAULT_GAMMA,
  DORY_SPEC_VERIFY_LOOPS,
  DorySessionImmutableConfigError,
  type DoryCheckpointLruEntryMeta,
  type DoryCheckpointLruStats,
  type DoryCounterfactualCodaResult,
  type DoryDualRopeTables,
  type DoryExecutionMode,
  type DoryForwardChunkResult,
  type DoryHaltReason,
  type DoryKvCache,
  type DoryKvCacheMode,
  type DoryLoopTrajectoryMetrics,
  type DoryMiniConfig,
  type DoryModelParameters,
  type DoryPrng,
  type DoryReconstructedStepResult,
  type DorySamplingConfig,
  type DorySelfSpecCycleResult,
  type DorySession,
  type DorySessionAppendChunkOptions,
  type DorySessionBounds,
  type DorySessionBranchConfig,
  type DorySessionBranchSnapshot,
  type DorySessionCreateOptions,
  type DorySessionForkOptions,
  type DorySessionForkReason,
  type DorySessionMemoryAudit,
  type DorySessionRecord,
  type DorySessionSnapshot,
  type DorySessionStartBranchOptions,
  type DorySessionStepOptions,
  type DorySessionSwitchConfigOptions,
  type DorySpecCyclePrefillResult,
  type DorySpecCycleSummary,
  type DorySpecExplicitVariates,
} from './types';

// ============================================================================
// INTERNAL DATA STRUCTURES (ZERO EXTRA ARRAYBUFFERS ON METADATA)
// ============================================================================

interface InternalBranch {
  readonly branchId: string;
  readonly label: string;
  readonly parentBranchId: string | null;
  readonly forkRecordId: string | null;
  readonly forkReason: DorySessionForkReason;
  readonly config: DorySessionBranchConfig;
  readonly promptTokens: readonly number[];
  readonly committedTokens: number[];
  readonly emittedTokens: number[];
  readonly recordIds: string[];
  headRecordId: string;
  pendingToken: number | null;
  haltReason: DoryHaltReason | null;
  kvCache: DoryKvCache;
  prng: DoryPrng;
  nextStepIndex: number;
  nextForwardStepId: number;
  readonly createdAtSeq: number;
  lastActiveSeq: number;
}

interface InternalRecordReplayMeta {
  readonly preStepPrng: DoryPrng;
  readonly postStepPrng: DoryPrng;
  readonly explicitSampleUniform: number | undefined;
  readonly specExplicitVariates: DorySpecExplicitVariates | undefined;
  readonly specInputPendingToken: number | null;
  readonly specGamma: number;
  readonly branchEmittedSnapshot: readonly number[];
}

interface InternalCheckpointEntry {
  readonly checkpointKey: string;
  readonly branchId: string;
  readonly recordId: string;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly includeShallowCoda: boolean;
  readonly committedLen: number;
  readonly chunkSizes: readonly number[];
  readonly committedTokens: readonly number[];
  readonly kvCache: DoryKvCache;
  readonly bytes: number;
  lastUsedSeq: number;
}

// ============================================================================
// PURE VALIDATION, CLONING & METRIC HELPERS
// ============================================================================

function isEosToken(tokenId: number): boolean {
  for (let i = 0; i < DORY_EOS_TOKEN_IDS.length; i += 1) {
    if (DORY_EOS_TOKEN_IDS[i] === tokenId) {
      return true;
    }
  }
  return false;
}

function resolveSessionBounds(partial?: Partial<DorySessionBounds>): DorySessionBounds {
  const maxBranches = partial?.maxBranches ?? DORY_DEFAULT_SESSION_BOUNDS.maxBranches;
  const maxRecords = partial?.maxRecords ?? DORY_DEFAULT_SESSION_BOUNDS.maxRecords;
  const maxCheckpointPoolBytes =
    partial?.maxCheckpointPoolBytes ?? DORY_DEFAULT_SESSION_BOUNDS.maxCheckpointPoolBytes;
  const maxSessionRetainedBytes =
    partial?.maxSessionRetainedBytes ?? DORY_DEFAULT_SESSION_BOUNDS.maxSessionRetainedBytes;
  const checkpointStride =
    partial?.checkpointStride ?? DORY_DEFAULT_SESSION_BOUNDS.checkpointStride;

  if (!Number.isInteger(maxBranches) || maxBranches < 1) {
    throw new RangeError(`Invalid maxBranches=${String(maxBranches)}: expected integer >= 1.`);
  }
  if (!Number.isInteger(maxRecords) || maxRecords < 1) {
    throw new RangeError(`Invalid maxRecords=${String(maxRecords)}: expected integer >= 1.`);
  }
  if (!Number.isInteger(maxCheckpointPoolBytes) || maxCheckpointPoolBytes < 0) {
    throw new RangeError(
      `Invalid maxCheckpointPoolBytes=${String(maxCheckpointPoolBytes)}: expected integer >= 0.`,
    );
  }
  if (!Number.isInteger(maxSessionRetainedBytes) || maxSessionRetainedBytes < 1) {
    throw new RangeError(
      `Invalid maxSessionRetainedBytes=${String(maxSessionRetainedBytes)}: expected integer >= 1.`,
    );
  }
  if (!Number.isInteger(checkpointStride) || checkpointStride < 1) {
    throw new RangeError(
      `Invalid checkpointStride=${String(checkpointStride)}: expected integer >= 1.`,
    );
  }

  return Object.freeze({
    maxBranches,
    maxRecords,
    maxCheckpointPoolBytes,
    maxSessionRetainedBytes,
    checkpointStride,
  });
}

function validateTokenArray(
  tokens: ArrayLike<number>,
  config: DoryMiniConfig,
  label: string,
): number[] {
  const len = tokens.length;
  if (!Number.isInteger(len) || len < 1) {
    throw new RangeError(`${label} must be non-empty; got length ${String(len)}.`);
  }
  if (len > config.maxPositionEmbeddings) {
    throw new RangeError(
      `${label} length ${len} exceeds maxPositionEmbeddings=${config.maxPositionEmbeddings}.`,
    );
  }
  const out = new Array<number>(len);
  for (let i = 0; i < len; i += 1) {
    const tok = tokens[i];
    if (!Number.isInteger(tok) || tok < 0 || tok >= config.vocabSize) {
      throw new RangeError(
        `Invalid token ID ${String(tok)} at ${label}[${i}] (expected integer in 0..${config.vocabSize - 1}).`,
      );
    }
    out[i] = tok;
  }
  return out;
}

function validateExplicitUniform(u: number | undefined, label: string): void {
  if (u === undefined) {
    return;
  }
  if (!Number.isFinite(u) || u < 0 || u >= 1) {
    throw new RangeError(`${label} must be a finite number in [0, 1); got ${String(u)}.`);
  }
}

function validateSpecExplicitVariates(
  explicit: DorySpecExplicitVariates | undefined,
): void {
  if (explicit === undefined) {
    return;
  }
  if (explicit.draftSampleUniforms !== undefined) {
    for (let i = 0; i < explicit.draftSampleUniforms.length; i += 1) {
      validateExplicitUniform(
        explicit.draftSampleUniforms[i],
        `specExplicitVariates.draftSampleUniforms[${i}]`,
      );
    }
  }
  if (explicit.acceptUniforms !== undefined) {
    for (let i = 0; i < explicit.acceptUniforms.length; i += 1) {
      validateExplicitUniform(
        explicit.acceptUniforms[i],
        `specExplicitVariates.acceptUniforms[${i}]`,
      );
    }
  }
  validateExplicitUniform(
    explicit.correctionOrBonusUniform,
    'specExplicitVariates.correctionOrBonusUniform',
  );
}

function validatePrefillChunkSizes(
  prefillChunkSizes: readonly number[] | undefined,
  promptLen: number,
  executionMode: DoryExecutionMode,
): number[] {
  if (prefillChunkSizes === undefined) {
    return [promptLen];
  }
  if (executionMode !== 'standard') {
    throw new RangeError(
      `prefillChunkSizes is only valid when executionMode === 'standard'; got '${executionMode}'.`,
    );
  }
  if (prefillChunkSizes.length < 1) {
    throw new RangeError('prefillChunkSizes must be non-empty when provided.');
  }
  let sum = 0;
  const out = new Array<number>(prefillChunkSizes.length);
  for (let i = 0; i < prefillChunkSizes.length; i += 1) {
    const sz = prefillChunkSizes[i];
    if (!Number.isInteger(sz) || sz < 1) {
      throw new RangeError(
        `Invalid prefillChunkSizes[${i}]=${String(sz)}: expected positive integer.`,
      );
    }
    sum += sz;
    out[i] = sz;
  }
  if (sum !== promptLen) {
    throw new RangeError(
      `prefillChunkSizes sum (${sum}) must equal promptTokens.length (${promptLen}).`,
    );
  }
  return out;
}

function resolveBranchConfig(
  config: DoryMiniConfig,
  opts: {
    readonly kLoops?: number;
    readonly kvCacheMode?: DoryKvCacheMode;
    readonly executionMode?: DoryExecutionMode;
    readonly speculativeGamma?: number;
    readonly sampling?: Partial<DorySamplingConfig>;
    readonly seed?: number;
  },
  fallback?: DorySessionBranchConfig,
): DorySessionBranchConfig {
  const executionMode: DoryExecutionMode =
    opts.executionMode ?? fallback?.executionMode ?? 'standard';
  if (executionMode !== 'standard' && executionMode !== 'self_spec') {
    throw new RangeError(`Unsupported Dory executionMode: ${String(executionMode)}.`);
  }

  const switchingIntoSpec =
    executionMode === 'self_spec' && fallback !== undefined && fallback.executionMode !== 'self_spec';

  const rawKvMode: DoryKvCacheMode =
    opts.kvCacheMode ??
    (switchingIntoSpec
      ? 'per_loop'
      : fallback?.kvCacheMode ?? (executionMode === 'self_spec' ? 'per_loop' : config.defaultKvCacheMode));

  if (rawKvMode !== 'per_loop' && rawKvMode !== 'last_loop') {
    throw new RangeError(`Unsupported Dory kvCacheMode: ${String(rawKvMode)}.`);
  }

  const rawKLoops: number =
    opts.kLoops ??
    (switchingIntoSpec
      ? DORY_SPEC_VERIFY_LOOPS
      : fallback?.kLoops ??
        (executionMode === 'self_spec' ? DORY_SPEC_VERIFY_LOOPS : config.defaultRecurrentLoops));

  if (executionMode === 'self_spec') {
    if (rawKvMode !== 'per_loop' || rawKLoops !== DORY_SPEC_VERIFY_LOOPS) {
      throw new DorySessionImmutableConfigError(
        'DR23_SPEC_REQUIRES_PER_LOOP_K4',
        `Contract A Paper §6.2 self-speculation (DR23) strictly requires kvCacheMode='per_loop' and kLoops=${DORY_SPEC_VERIFY_LOOPS}; got kvCacheMode='${rawKvMode}', kLoops=${String(rawKLoops)}.`,
      );
    }
  }

  const kLoops = validateDoryRecurrentLoops(rawKLoops);

  const speculativeGamma =
    opts.speculativeGamma ?? fallback?.speculativeGamma ?? DORY_SPEC_DEFAULT_GAMMA;
  const maxSafeGamma = config.swaRingCapacity - config.slidingWindow - 1;
  if (
    !Number.isInteger(speculativeGamma) ||
    speculativeGamma < 1 ||
    speculativeGamma > maxSafeGamma
  ) {
    throw new RangeError(
      `Invalid speculativeGamma=${String(speculativeGamma)}: expected integer in 1..${maxSafeGamma} satisfying DR19 W(${config.slidingWindow}) + gamma + 1 <= cap_swa(${config.swaRingCapacity}).`,
    );
  }

  const mergedSamplingInput: Partial<DorySamplingConfig> =
    fallback !== undefined
      ? {
          temperature: opts.sampling?.temperature ?? fallback.sampling.temperature,
          topK: opts.sampling?.topK ?? fallback.sampling.topK,
          topP: opts.sampling?.topP ?? fallback.sampling.topP,
        }
      : {
          temperature: opts.sampling?.temperature ?? DORY_DEFAULT_GREEDY_SAMPLING.temperature,
          topK: opts.sampling?.topK ?? DORY_DEFAULT_GREEDY_SAMPLING.topK,
          topP: opts.sampling?.topP ?? DORY_DEFAULT_GREEDY_SAMPLING.topP,
        };
  const sampling = Object.freeze(normalizeSamplingConfig(mergedSamplingInput));

  const rawSeed = opts.seed ?? fallback?.seed ?? DORY_DEFAULT_SESSION_SEED;
  if (!Number.isFinite(rawSeed)) {
    throw new RangeError(`Invalid session seed: ${String(rawSeed)}.`);
  }
  const seed = rawSeed >>> 0;

  return Object.freeze({
    kLoops,
    kvCacheMode: rawKvMode,
    executionMode,
    includeShallowCoda: executionMode === 'self_spec',
    speculativeGamma,
    sampling,
    seed,
  });
}

function sliceCopyFloat32(src: Float32Array, offset: number, length: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = src[offset + i];
  }
  return out;
}

function copyFloat32(src: Float32Array): Float32Array {
  return new Float32Array(src);
}

function copyFloat64(src: Float64Array): Float64Array {
  return new Float64Array(src);
}

function copyInt32FromArrayLike(src: ArrayLike<number>): Int32Array {
  const out = new Int32Array(src.length);
  for (let i = 0; i < src.length; i += 1) {
    out[i] = src[i];
  }
  return out;
}

function cloneLoopMetrics(
  metrics: readonly DoryLoopTrajectoryMetrics[],
): readonly DoryLoopTrajectoryMetrics[] {
  const cloned = metrics.map((m) =>
    Object.freeze({
      loopIteration: m.loopIteration,
      tokenOffset: m.tokenOffset,
      absTokenPos: m.absTokenPos,
      uNorm: m.uNorm,
      sNorm: m.sNorm,
      consecutiveCosine: m.consecutiveCosine,
      consecutiveDisplacement: m.consecutiveDisplacement,
      anchorCosine: m.anchorCosine,
      entryAnchorJacobian: Object.freeze({ ...m.entryAnchorJacobian }),
    }),
  );
  return Object.freeze(cloned);
}

function cloneSpecSummary(summary: DorySpecCycleSummary | null): DorySpecCycleSummary | null {
  if (summary === null) {
    return null;
  }
  return Object.freeze({
    contractBadge: summary.contractBadge,
    startCommittedLen: summary.startCommittedLen,
    endCommittedLen: summary.endCommittedLen,
    inputPendingToken: summary.inputPendingToken,
    nextPendingToken: summary.nextPendingToken,
    gammaEffective: summary.gammaEffective,
    draftCount: summary.draftCount,
    draftEndedWithEos: summary.draftEndedWithEos,
    acceptedDraftCount: summary.acceptedDraftCount,
    emittedTokens: new Int32Array(summary.emittedTokens),
    outcome: summary.outcome,
    haltReason: summary.haltReason,
    reusedActivationPositions: summary.reusedActivationPositions,
    verifiedPositions: summary.verifiedPositions,
    shallowCodaCatchupExecuted: summary.shallowCodaCatchupExecuted,
    rolledBackPositions: summary.rolledBackPositions,
    residualMassZ: summary.residualMassZ,
    correctionOrBonusToken: summary.correctionOrBonusToken,
    draftSteps: Object.freeze(summary.draftSteps.map((d) => Object.freeze({ ...d }))),
    verifyPositions: Object.freeze(
      summary.verifyPositions.map((v) =>
        Object.freeze({
          ...v,
          rejection: v.rejection === null ? null : Object.freeze({ ...v.rejection }),
        }),
      ),
    ),
  });
}

function cloneSpecExplicitVariates(
  variates: DorySpecExplicitVariates | undefined,
): DorySpecExplicitVariates | undefined {
  if (variates === undefined) {
    return undefined;
  }
  return Object.freeze({
    draftSampleUniforms:
      variates.draftSampleUniforms !== undefined
        ? Object.freeze([...variates.draftSampleUniforms])
        : undefined,
    acceptUniforms:
      variates.acceptUniforms !== undefined
        ? Object.freeze([...variates.acceptUniforms])
        : undefined,
    correctionOrBonusUniform: variates.correctionOrBonusUniform,
  });
}

function computeLastTokenLoopMetrics(
  xAnchor: Float32Array,
  loopEntryStates: readonly Float32Array[],
  loopExitStates: readonly Float32Array[],
  seqLen: number,
  startPos: number,
  d: number,
  eps: number,
): readonly DoryLoopTrajectoryMetrics[] {
  const kLoops = loopExitStates.length;
  const tokenOffset = seqLen - 1;
  const off = tokenOffset * d;
  const absTokenPos = startPos + tokenOffset;
  const metrics: DoryLoopTrajectoryMetrics[] = new Array<DoryLoopTrajectoryMetrics>(kLoops);

  for (let loopIdx = 0; loopIdx < kLoops; loopIdx += 1) {
    const loopIteration = loopIdx + 1;
    const prevState = loopIdx === 0 ? xAnchor : loopExitStates[loopIdx - 1];
    const uEntry = loopEntryStates[loopIdx];
    const sExit = loopExitStates[loopIdx];

    const prevAnchorCos = cosineSimilarity8d(prevState, xAnchor, off, off, d);
    const clampedCos = Math.max(-1.0, Math.min(1.0, prevAnchorCos));
    const entryAngleRadians = Math.acos(clampedCos);

    metrics[loopIdx] = Object.freeze({
      loopIteration,
      tokenOffset,
      absTokenPos,
      uNorm: vectorL2Norm(uEntry, off, d),
      sNorm: vectorL2Norm(sExit, off, d),
      consecutiveCosine: cosineSimilarity8d(sExit, prevState, off, off, d),
      consecutiveDisplacement: l2Distance(sExit, prevState, off, off, d),
      anchorCosine: cosineSimilarity8d(sExit, xAnchor, off, off, d),
      entryAnchorJacobian: anchorMapJacobianNorm(entryAngleRadians, eps),
    });
  }

  return Object.freeze(metrics);
}

function synchronizeCacheChunkSizes(kvCache: DoryKvCache): void {
  const targetLen = kvCache.committedLen;
  let sum = 0;
  for (let i = 0; i < kvCache.chunkSizes.length; i += 1) {
    sum += kvCache.chunkSizes[i];
  }
  if (sum === targetLen) {
    return;
  }
  if (targetLen === 0) {
    kvCache.chunkSizes.length = 0;
    return;
  }
  if (sum < targetLen) {
    kvCache.chunkSizes.push(targetLen - sum);
    return;
  }
  // Trim excess chunk entries if sum > targetLen
  let acc = 0;
  let keepCount = 0;
  for (let i = 0; i < kvCache.chunkSizes.length; i += 1) {
    const sz = kvCache.chunkSizes[i];
    if (acc + sz <= targetLen) {
      acc += sz;
      keepCount = i + 1;
    } else {
      const remainder = targetLen - acc;
      if (remainder > 0) {
        kvCache.chunkSizes[i] = remainder;
        acc = targetLen;
        keepCount = i + 1;
      }
      break;
    }
  }
  kvCache.chunkSizes.length = keepCount;
}

function isChunkPrefix(
  prefix: readonly number[],
  full: readonly number[],
): boolean {
  if (prefix.length > full.length) {
    return false;
  }
  for (let i = 0; i < prefix.length; i += 1) {
    if (prefix[i] !== full[i]) {
      return false;
    }
  }
  return true;
}

function isTokenPrefix(
  prefix: readonly number[],
  full: ArrayLike<number>,
): boolean {
  if (prefix.length > full.length) {
    return false;
  }
  for (let i = 0; i < prefix.length; i += 1) {
    if (prefix[i] !== full[i]) {
      return false;
    }
  }
  return true;
}

function sampleFromFilteredProbs(
  probs: Float64Array,
  sampling: DorySamplingConfig,
  prng: DoryPrng,
  explicitUniform: number | undefined,
): number {
  if (sampling.temperature === 0) {
    return sampleCategoricalFloat64(probs, 0);
  }
  validateExplicitUniform(explicitUniform, 'sampleUniform');
  const u = explicitUniform !== undefined ? explicitUniform : prng.nextFloat64();
  return sampleCategoricalFloat64(probs, u);
}

function sumUniqueBufferBytes(buffers: ReadonlySet<ArrayBufferLike>): number {
  let total = 0;
  buffers.forEach((buf) => {
    total += buf.byteLength;
  });
  return total;
}

function collectKvCacheBuffers(
  kvCache: DoryKvCache,
  targetSet: Set<ArrayBufferLike>,
  allSessionSet: Set<ArrayBufferLike>,
  slabOwnerSeen: Set<ArrayBufferLike>,
): boolean {
  let nonAliased = true;
  for (let b = 0; b < kvCache.allBanks.length; b += 1) {
    const bank = kvCache.allBanks[b];
    const bufs: readonly ArrayBufferLike[] = [
      bank.k.buffer,
      bank.v.buffer,
      bank.absPosTag.buffer,
      bank.writerLoopTag.buffer,
    ];
    for (let i = 0; i < bufs.length; i += 1) {
      const buf = bufs[i];
      if (slabOwnerSeen.has(buf)) {
        nonAliased = false;
      }
      slabOwnerSeen.add(buf);
      targetSet.add(buf);
      allSessionSet.add(buf);
    }
  }
  return nonAliased;
}

function collectRecordBuffers(
  record: DorySessionRecord,
  recordSet: Set<ArrayBufferLike>,
  allSessionSet: Set<ArrayBufferLike>,
  slabOwnerSeen: ReadonlySet<ArrayBufferLike>,
): boolean {
  let nonAliased = true;
  const add = (buf: ArrayBufferLike): void => {
    if (slabOwnerSeen.has(buf)) {
      nonAliased = false;
    }
    recordSet.add(buf);
    allSessionSet.add(buf);
  };

  add(record.inputTokens.buffer);
  add(record.emittedTokens.buffer);
  add(record.committedSequence.buffer);
  add(record.lastTokenScaledLogits.buffer);
  add(record.lastTokenProbs.buffer);
  add(record.lastTokenXAnchor.buffer);

  for (let i = 0; i < record.lastTokenLoopEntryStates.length; i += 1) {
    add(record.lastTokenLoopEntryStates[i].buffer);
  }
  for (let i = 0; i < record.lastTokenLoopExitStates.length; i += 1) {
    add(record.lastTokenLoopExitStates[i].buffer);
  }
  if (record.specSummary !== null) {
    add(record.specSummary.emittedTokens.buffer);
  }
  return nonAliased;
}

function formatDefaultBranchLabel(
  branchId: string,
  cfg: DorySessionBranchConfig,
  forkReason: DorySessionForkReason,
): string {
  const modeTag =
    cfg.executionMode === 'self_spec' ? 'spec-A k=1->4' : `k=${cfg.kLoops} ${cfg.kvCacheMode}`;
  if (forkReason === 'dr18_re_prefill') {
    return `${branchId} (${modeTag} · DR18 re-prefill)`;
  }
  if (forkReason === 'cursor_fork') {
    return `${branchId} (${modeTag} · fork)`;
  }
  return `${branchId} (${modeTag})`;
}

// ============================================================================
// BOUNDED `DorySession` FACTORY & `Set<ArrayBufferLike>` MEMORY AUDIT HELPER
// ============================================================================

/**
 * Creates a bounded (`<= 12.00 MiB`), immutable-branch `DorySession` enforcing:
 *   - Non-aliased copy-on-write KV slabs (`DR18`, `DR19`)
 *   - `D6` / `DR18` fork/re-prefill depth & KV-policy switch guard
 *   - `DR23` Contract A (`per_loop`, `kLoops = 4`, `104` banks) enforcement
 *   - `D10` `3.00 MiB` (`3,145,728 B`) byte-budgeted LRU sparse-checkpoint pool
 *   - `DR24` verbatim `chunkSizes[]` replay during `0-ULP` lazy step-trace reconstruction
 */
export function createDorySession(options: DorySessionCreateOptions = {}): DorySession {
  const params: DoryModelParameters = options.params ?? getDefaultDoryModelParameters();
  const config: DoryMiniConfig = params.config;
  const ropeTables: DoryDualRopeTables = options.ropeTables ?? buildDoryDualRopeTables(config);
  const bounds: DorySessionBounds = resolveSessionBounds(options.bounds);
  const autoStartInitialBranch = options.autoStartInitialBranch ?? true;
  const initialBranchTemplate = options.initialBranch;

  // Singleton weight buffers are computed once from `params` (never cloned)
  const singletonWeightBuffers = new Set<ArrayBufferLike>();
  const canonicalTensors = listDoryCanonicalTensors(params);
  for (let i = 0; i < canonicalTensors.length; i += 1) {
    singletonWeightBuffers.add(canonicalTensors[i].data.buffer);
  }
  const singletonWeightBytes = sumUniqueBufferBytes(singletonWeightBuffers);

  // Live session state
  const branchesById = new Map<string, InternalBranch>();
  const recordsById = new Map<string, DorySessionRecord>();
  const replayMetaByRecordId = new Map<string, InternalRecordReplayMeta>();
  const checkpointPool = new Map<string, InternalCheckpointEntry>();

  let activeBranchId: string | null = null;
  let activeRecordId: string | null = null;

  let seqCounter = 0;
  let branchIdCounter = 0;
  let recordIdCounter = 0;

  let poolUsedBytes = 0;
  let poolHitCount = 0;
  let poolMissCount = 0;
  let poolEvictionCount = 0;

  function nextSeq(): number {
    seqCounter += 1;
    return seqCounter;
  }

  function allocateBranchId(): string {
    const id = `b${branchIdCounter}`;
    branchIdCounter += 1;
    return id;
  }

  function allocateRecordId(): string {
    const id = `r${recordIdCounter}`;
    recordIdCounter += 1;
    return id;
  }

  // --------------------------------------------------------------------------
  // D10 3.00 MiB BYTE-BUDGETED LRU SPARSE-CHECKPOINT POOL
  // --------------------------------------------------------------------------

  function deleteCheckpointKey(key: string): void {
    const existing = checkpointPool.get(key);
    if (existing !== undefined) {
      checkpointPool.delete(key);
      poolUsedBytes -= existing.bytes;
    }
  }

  function purgeCheckpointsForRecord(recordId: string): void {
    deleteCheckpointKey(recordId);
  }

  function purgeCheckpointsForBranch(branchId: string): void {
    const toRemove: string[] = [];
    checkpointPool.forEach((entry, key) => {
      if (entry.branchId === branchId) {
        toRemove.push(key);
      }
    });
    for (let i = 0; i < toRemove.length; i += 1) {
      deleteCheckpointKey(toRemove[i]);
    }
  }

  function maybeInsertCheckpoint(branch: InternalBranch, record: DorySessionRecord): void {
    if (record.stepIndex % bounds.checkpointStride !== 0) {
      return;
    }
    if (bounds.maxCheckpointPoolBytes <= 0) {
      return;
    }
    // Never alias live branch slabs: clone immediately before measuring/storing (DR18/DR19)
    const clonedKv = cloneDoryKvCache(branch.kvCache);
    synchronizeCacheChunkSizes(clonedKv);
    const bytes = measureDoryKvCacheBytes(clonedKv).totalUniqueBufferBytes;

    // If a single checkpoint exceeds the entire pool budget, do not retain it
    if (bytes > bounds.maxCheckpointPoolBytes) {
      return;
    }

    const key = record.recordId;
    if (checkpointPool.has(key)) {
      deleteCheckpointKey(key);
    }

    // Evict strictly oldest least-recently-used entries until new checkpoint fits within D10 cap
    while (poolUsedBytes + bytes > bounds.maxCheckpointPoolBytes && checkpointPool.size > 0) {
      let oldestKey: string | null = null;
      let oldestSeq = Infinity;
      checkpointPool.forEach((entry, k) => {
        if (entry.lastUsedSeq < oldestSeq) {
          oldestSeq = entry.lastUsedSeq;
          oldestKey = k;
        }
      });
      if (oldestKey === null) {
        break;
      }
      deleteCheckpointKey(oldestKey);
      poolEvictionCount += 1;
    }

    if (poolUsedBytes + bytes > bounds.maxCheckpointPoolBytes) {
      return;
    }

    const entry: InternalCheckpointEntry = {
      checkpointKey: key,
      branchId: branch.branchId,
      recordId: record.recordId,
      kLoops: branch.config.kLoops,
      kvCacheMode: branch.config.kvCacheMode,
      includeShallowCoda: branch.config.includeShallowCoda,
      committedLen: clonedKv.committedLen,
      chunkSizes: Object.freeze([...clonedKv.chunkSizes]),
      committedTokens: Object.freeze([...branch.committedTokens]),
      kvCache: clonedKv,
      bytes,
      lastUsedSeq: nextSeq(),
    };
    checkpointPool.set(key, entry);
    poolUsedBytes += bytes;
  }

  function findBestCheckpointPrefix(
    branchId: string,
    kLoops: number,
    kvCacheMode: DoryKvCacheMode,
    includeShallowCoda: boolean,
    targetChunkSizes: readonly number[],
    targetSequence: ArrayLike<number>,
    targetLen: number,
  ): InternalCheckpointEntry | null {
    if (targetLen <= 0 || targetChunkSizes.length === 0) {
      return null;
    }
    let best: InternalCheckpointEntry | null = null;
    checkpointPool.forEach((entry) => {
      if (
        entry.kLoops !== kLoops ||
        entry.kvCacheMode !== kvCacheMode ||
        entry.includeShallowCoda !== includeShallowCoda ||
        entry.committedLen <= 0 ||
        entry.committedLen > targetLen
      ) {
        return;
      }
      if (!isChunkPrefix(entry.chunkSizes, targetChunkSizes)) {
        return;
      }
      if (!isTokenPrefix(entry.committedTokens, targetSequence)) {
        return;
      }
      if (best === null) {
        best = entry;
        return;
      }
      if (entry.committedLen > best.committedLen) {
        best = entry;
        return;
      }
      if (entry.committedLen === best.committedLen) {
        const entrySameBranch = entry.branchId === branchId;
        const bestSameBranch = best.branchId === branchId;
        if (entrySameBranch && !bestSameBranch) {
          best = entry;
        } else if (entrySameBranch === bestSameBranch && entry.lastUsedSeq > best.lastUsedSeq) {
          best = entry;
        }
      }
    });
    return best;
  }

  function getCheckpointLruStats(): DoryCheckpointLruStats {
    const entries: DoryCheckpointLruEntryMeta[] = [];
    checkpointPool.forEach((entry) => {
      entries.push(
        Object.freeze({
          checkpointKey: entry.checkpointKey,
          branchId: entry.branchId,
          recordId: entry.recordId,
          committedLen: entry.committedLen,
          chunkSizes: entry.chunkSizes,
          bytes: entry.bytes,
          lastUsedSeq: entry.lastUsedSeq,
        }),
      );
    });
    entries.sort((a, b) => a.lastUsedSeq - b.lastUsedSeq);

    return Object.freeze({
      maxPoolBytes: bounds.maxCheckpointPoolBytes,
      usedBytes: poolUsedBytes,
      entryCount: checkpointPool.size,
      hitCount: poolHitCount,
      missCount: poolMissCount,
      evictionCount: poolEvictionCount,
      entries: Object.freeze(entries),
    });
  }

  // --------------------------------------------------------------------------
  // BRANCH & RECORD RETENTION CAPS (`maxBranches = 12`, `maxRecords = 128`)
  // --------------------------------------------------------------------------

  function evictBranchById(branchId: string): void {
    const branch = branchesById.get(branchId);
    if (!branch) {
      return;
    }
    branchesById.delete(branchId);
    purgeCheckpointsForBranch(branchId);
    for (let i = 0; i < branch.recordIds.length; i += 1) {
      const recId = branch.recordIds[i];
      recordsById.delete(recId);
      replayMetaByRecordId.delete(recId);
      purgeCheckpointsForRecord(recId);
    }
    if (activeBranchId === branchId) {
      activeBranchId = null;
      activeRecordId = null;
    }
  }

  function enforceBranchCap(pinnedBranchIds: ReadonlySet<string>): void {
    while (branchesById.size > bounds.maxBranches) {
      let victim: InternalBranch | null = null;
      branchesById.forEach((b) => {
        if (pinnedBranchIds.has(b.branchId)) {
          return;
        }
        if (victim === null || b.lastActiveSeq < victim.lastActiveSeq) {
          victim = b;
        }
      });
      if (victim === null) {
        // Fallback if all branches are pinned (e.g. maxBranches === 1): evict oldest non-active
        branchesById.forEach((b) => {
          if (b.branchId === activeBranchId) {
            return;
          }
          if (victim === null || b.lastActiveSeq < victim.lastActiveSeq) {
            victim = b;
          }
        });
      }
      if (victim === null) {
        break;
      }
      evictBranchById((victim as InternalBranch).branchId);
    }
  }

  function enforceRecordCap(): void {
    while (recordsById.size > bounds.maxRecords) {
      const pinnedRecordIds = new Set<string>();
      branchesById.forEach((b) => {
        pinnedRecordIds.add(b.headRecordId);
      });
      if (activeRecordId !== null) {
        pinnedRecordIds.add(activeRecordId);
      }

      let victim: DorySessionRecord | null = null;
      recordsById.forEach((rec) => {
        if (pinnedRecordIds.has(rec.recordId)) {
          return;
        }
        if (victim === null || rec.createdAtSeq < victim.createdAtSeq) {
          victim = rec;
        }
      });

      if (victim === null) {
        // Pathological case: bounds.maxRecords < live branch count -> evict oldest non-active branch
        let oldestBranch: InternalBranch | null = null;
        branchesById.forEach((b) => {
          if (b.branchId === activeBranchId) {
            return;
          }
          if (oldestBranch === null || b.lastActiveSeq < oldestBranch.lastActiveSeq) {
            oldestBranch = b;
          }
        });
        if (oldestBranch === null) {
          break;
        }
        evictBranchById((oldestBranch as InternalBranch).branchId);
        continue;
      }

      const vRec = victim as DorySessionRecord;
      recordsById.delete(vRec.recordId);
      replayMetaByRecordId.delete(vRec.recordId);
      purgeCheckpointsForRecord(vRec.recordId);

      const owner = branchesById.get(vRec.branchId);
      if (owner !== undefined) {
        const idx = owner.recordIds.indexOf(vRec.recordId);
        if (idx >= 0) {
          owner.recordIds.splice(idx, 1);
        }
      }
    }
  }

  function registerRecord(
    branch: InternalBranch,
    record: DorySessionRecord,
    replayMeta: InternalRecordReplayMeta,
    captureCheckpoint: boolean,
  ): void {
    recordsById.set(record.recordId, record);
    replayMetaByRecordId.set(record.recordId, replayMeta);
    branch.recordIds.push(record.recordId);
    branch.headRecordId = record.recordId;
    branch.lastActiveSeq = record.createdAtSeq;
    activeBranchId = branch.branchId;
    activeRecordId = record.recordId;

    if (captureCheckpoint) {
      maybeInsertCheckpoint(branch, record);
    }
    enforceRecordCap();
  }

  // --------------------------------------------------------------------------
  // IMMUTABLE SNAPSHOT BUILDERS
  // --------------------------------------------------------------------------

  function buildBranchSnapshot(branch: InternalBranch): DorySessionBranchSnapshot {
    const kvBytes = measureDoryKvCacheBytes(branch.kvCache).totalUniqueBufferBytes;
    return Object.freeze({
      branchId: branch.branchId,
      label: branch.label,
      parentBranchId: branch.parentBranchId,
      forkRecordId: branch.forkRecordId,
      forkReason: branch.forkReason,
      config: branch.config,
      promptTokens: copyInt32FromArrayLike(branch.promptTokens),
      chunkSizes: Object.freeze([...branch.kvCache.chunkSizes]),
      committedTokens: copyInt32FromArrayLike(branch.committedTokens),
      committedLen: branch.kvCache.committedLen,
      pendingToken: branch.pendingToken,
      emittedTokens: copyInt32FromArrayLike(branch.emittedTokens),
      recordIds: Object.freeze([...branch.recordIds]),
      headRecordId: branch.headRecordId,
      haltReason: branch.haltReason,
      kvBankCount: branch.kvCache.allBanks.length,
      kvBytes,
      createdAtSeq: branch.createdAtSeq,
      lastActiveSeq: branch.lastActiveSeq,
    });
  }

  // --------------------------------------------------------------------------
  // CHUNK REPLAY ENGINE (DR24 VERBATIM `chunkSizes[]` REPLAY — ZERO COALESCING)
  // --------------------------------------------------------------------------

  function replayChunkSequenceIntoCache(
    kvCache: DoryKvCache,
    sequence: ArrayLike<number>,
    chunkSizes: readonly number[],
    startChunkIdx: number,
    branchId: string,
    includeShallowCoda: boolean,
  ): number[] {
    const replayed: number[] = [];
    let pos = kvCache.committedLen;

    for (let i = startChunkIdx; i < chunkSizes.length; i += 1) {
      const sz = chunkSizes[i];
      const chunkSlice = new Int32Array(sz);
      for (let j = 0; j < sz; j += 1) {
        chunkSlice[j] = sequence[pos + j];
      }

      const fwd = runDoryForwardChunk({
        tokens: chunkSlice,
        params,
        kvCache,
        startPos: pos,
        kLoops: kvCache.kLoops,
        kvCacheMode: kvCache.mode,
        ropeTables,
        codaKind: 'output_deep',
        writeKv: true,
        captureTrace: false,
        forwardStepId: i,
        sessionBranchId: branchId,
        role: 'standard',
      });

      // Contract A self-spec replay keeps the 8 persistent `output_shallow` Coda banks
      // 0-ULP synchronized by running Shallow Coda over cached Loop-1 state `s_1`.
      if (includeShallowCoda) {
        runDoryOutputCoda({
          state: fwd.loopExitStates[0],
          seqLen: sz,
          startPos: pos,
          kLoops: kvCache.kLoops,
          params,
          kvCache,
          ropeTables,
          codaKind: 'output_shallow',
          writeKv: true,
          captureTrace: false,
          forwardStepId: i,
          sessionBranchId: branchId,
          role: 'spec_draft',
        });
      }

      pos += sz;
      replayed.push(sz);
    }

    synchronizeCacheChunkSizes(kvCache);
    return replayed;
  }

  function buildKvCacheFromPrefix(
    branchId: string,
    kLoops: number,
    kvCacheMode: DoryKvCacheMode,
    includeShallowCoda: boolean,
    targetChunkSizes: readonly number[],
    targetSequence: ArrayLike<number>,
    targetLen: number,
    recordLruMetrics = true,
  ): {
    readonly kvCache: DoryKvCache;
    readonly usedCheckpoint: boolean;
    readonly checkpointHitRecordId: string | null;
    readonly replayedChunkSizes: readonly number[];
  } {
    if (targetLen === 0 || targetChunkSizes.length === 0) {
      const fresh = createDoryKvCache({
        mode: kvCacheMode,
        kLoops,
        includeShallowCoda,
        config,
      });
      return {
        kvCache: fresh,
        usedCheckpoint: false,
        checkpointHitRecordId: null,
        replayedChunkSizes: Object.freeze([]),
      };
    }

    const hit = findBestCheckpointPrefix(
      branchId,
      kLoops,
      kvCacheMode,
      includeShallowCoda,
      targetChunkSizes,
      targetSequence,
      targetLen,
    );

    if (hit !== null) {
      if (recordLruMetrics) {
        poolHitCount += 1;
        hit.lastUsedSeq = nextSeq();
      }
      const kvCache = cloneDoryKvCache(hit.kvCache);
      const replayed = replayChunkSequenceIntoCache(
        kvCache,
        targetSequence,
        targetChunkSizes,
        hit.chunkSizes.length,
        branchId,
        includeShallowCoda,
      );
      return {
        kvCache,
        usedCheckpoint: true,
        checkpointHitRecordId: hit.recordId,
        replayedChunkSizes: Object.freeze(replayed),
      };
    }

    if (recordLruMetrics) {
      poolMissCount += 1;
    }
    const fresh = createDoryKvCache({
      mode: kvCacheMode,
      kLoops,
      includeShallowCoda,
      config,
    });
    const replayed = replayChunkSequenceIntoCache(
      fresh,
      targetSequence,
      targetChunkSizes,
      0,
      branchId,
      includeShallowCoda,
    );
    return {
      kvCache: fresh,
      usedCheckpoint: false,
      checkpointHitRecordId: null,
      replayedChunkSizes: Object.freeze(replayed),
    };
  }

  // --------------------------------------------------------------------------
  // STANDARD & CONTRACT A EXECUTION HELPERS
  // --------------------------------------------------------------------------

  function executeStandardChunkOnBranch(
    branch: InternalBranch,
    chunkTokens: readonly number[],
    kind: 'prefill' | 'chunk' | 'decode_step',
    parentRecordId: string | null,
    samplePending: boolean,
    explicitSampleUniform: number | undefined,
  ): DorySessionRecord {
    const startPos = branch.kvCache.committedLen;
    const seqLen = chunkTokens.length;
    const endPos = startPos + seqLen;

    if (endPos > config.maxPositionEmbeddings) {
      throw new RangeError(
        `Capacity exceeded on branch '${branch.branchId}': committedLen(${startPos}) + chunkLen(${seqLen}) > maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
      );
    }

    const chunkSizesBefore = Object.freeze([...branch.kvCache.chunkSizes]);
    const preStepPrng = branch.prng.clone();
    const stepIndex = branch.nextStepIndex;
    const forwardStepId = branch.nextForwardStepId;

    const fwd = runDoryForwardChunk({
      tokens: chunkTokens,
      params,
      kvCache: branch.kvCache,
      startPos,
      kLoops: branch.config.kLoops,
      kvCacheMode: branch.config.kvCacheMode,
      ropeTables,
      codaKind: 'output_deep',
      writeKv: true,
      captureTrace: false,
      forwardStepId,
      sessionBranchId: branch.branchId,
      role: 'standard',
    });

    synchronizeCacheChunkSizes(branch.kvCache);
    const chunkSizesAfter = Object.freeze([...branch.kvCache.chunkSizes]);

    const vocabSize = config.vocabSize;
    const d = config.hiddenSize;
    const lastLogitOff = (seqLen - 1) * vocabSize;
    const lastTokenScaledLogits = sliceCopyFloat32(fwd.scaledLogits, lastLogitOff, vocabSize);
    const lastTokenProbs = logitsToSamplingProbsFloat64(
      lastTokenScaledLogits,
      0,
      vocabSize,
      branch.config.sampling,
    );

    const lastVecOff = (seqLen - 1) * d;
    const lastTokenXAnchor = sliceCopyFloat32(fwd.xAnchor, lastVecOff, d);
    const lastTokenLoopEntryStates = Object.freeze(
      fwd.loopEntryStates.map((s) => sliceCopyFloat32(s, lastVecOff, d)),
    );
    const lastTokenLoopExitStates = Object.freeze(
      fwd.loopExitStates.map((s) => sliceCopyFloat32(s, lastVecOff, d)),
    );
    const lastTokenLoopMetrics = computeLastTokenLoopMetrics(
      fwd.xAnchor,
      fwd.loopEntryStates,
      fwd.loopExitStates,
      seqLen,
      startPos,
      d,
      config.l2NormEps,
    );

    for (let i = 0; i < seqLen; i += 1) {
      branch.committedTokens.push(chunkTokens[i]);
    }

    let pendingToken: number | null = null;
    let emittedTokensArr: Int32Array;
    let haltReason: DoryHaltReason | null = null;

    if (samplePending) {
      const nextTok = sampleFromFilteredProbs(
        lastTokenProbs,
        branch.config.sampling,
        branch.prng,
        explicitSampleUniform,
      );
      pendingToken = nextTok;
      emittedTokensArr = Int32Array.of(nextTok);
      if (isEosToken(nextTok)) {
        haltReason = 'eos';
      } else if (endPos >= config.maxPositionEmbeddings) {
        haltReason = 'capacity_exhausted';
      }
    } else {
      emittedTokensArr = new Int32Array(0);
    }

    branch.pendingToken = pendingToken;
    branch.haltReason = haltReason;
    branch.emittedTokens.length = 0;
    for (let i = 0; i < branch.committedTokens.length; i += 1) {
      branch.emittedTokens.push(branch.committedTokens[i]);
    }
    if (pendingToken !== null) {
      branch.emittedTokens.push(pendingToken);
    }

    branch.nextStepIndex = stepIndex + 1;
    branch.nextForwardStepId = forwardStepId + 1;

    const recordId = allocateRecordId();
    const createdAtSeq = nextSeq();

    const record: DorySessionRecord = Object.freeze({
      recordId,
      branchId: branch.branchId,
      parentRecordId,
      stepIndex,
      forwardStepId,
      kind,
      kLoops: branch.config.kLoops,
      kvCacheMode: branch.config.kvCacheMode,
      executionMode: branch.config.executionMode,
      sampling: branch.config.sampling,
      startPos,
      endPos,
      committedLen: endPos,
      pendingToken,
      inputTokens: copyInt32FromArrayLike(chunkTokens),
      emittedTokens: emittedTokensArr,
      committedSequence: copyInt32FromArrayLike(branch.committedTokens),
      chunkSizesBefore,
      chunkSizesAfter,
      lastTokenScaledLogits,
      lastTokenProbs,
      lastTokenXAnchor,
      lastTokenLoopEntryStates,
      lastTokenLoopExitStates,
      lastTokenLoopMetrics,
      specSummary: null,
      haltReason,
      createdAtSeq,
    });

    const replayMeta: InternalRecordReplayMeta = {
      preStepPrng,
      postStepPrng: branch.prng.clone(),
      explicitSampleUniform,
      specExplicitVariates: undefined,
      specInputPendingToken: null,
      specGamma: branch.config.speculativeGamma,
      branchEmittedSnapshot: Object.freeze([...branch.emittedTokens]),
    };

    registerRecord(branch, record, replayMeta, true);
    return record;
  }

  function executeSpecCycleOnBranch(
    branch: InternalBranch,
    parentRecordId: string | null,
    explicitVariates: DorySpecExplicitVariates | undefined,
  ): DorySessionRecord {
    if (branch.haltReason !== null) {
      throw new RangeError(
        `Cannot step halted self_spec branch '${branch.branchId}' (haltReason='${branch.haltReason}').`,
      );
    }
    if (branch.pendingToken === null) {
      throw new RangeError(
        `Cannot step self_spec branch '${branch.branchId}': pendingToken is null.`,
      );
    }
    if (branch.kvCache.committedLen >= config.maxPositionEmbeddings) {
      throw new RangeError(
        `Cannot step self_spec branch '${branch.branchId}': committedLen(${branch.kvCache.committedLen}) reached maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
      );
    }

    const chunkSizesBefore = Object.freeze([...branch.kvCache.chunkSizes]);
    const preStepPrng = branch.prng.clone();
    const inputPendingToken = branch.pendingToken;
    const stepIndex = branch.nextStepIndex;
    const forwardStepId = branch.nextForwardStepId;

    const cycle: DorySelfSpecCycleResult = runDorySelfSpecCycle({
      kvCache: branch.kvCache,
      pendingToken: inputPendingToken,
      params,
      gamma: branch.config.speculativeGamma,
      sampling: branch.config.sampling,
      prng: branch.prng,
      explicitVariates,
      ropeTables,
      sessionBranchId: branch.branchId,
      forwardStepId,
      captureTrace: false,
    });

    synchronizeCacheChunkSizes(branch.kvCache);
    const chunkSizesAfter = Object.freeze([...branch.kvCache.chunkSizes]);

    const committedDelta = cycle.endCommittedLen - cycle.startCommittedLen;
    const consumedTokens = new Int32Array(committedDelta);
    for (let i = 0; i < committedDelta; i += 1) {
      const tok =
        i < cycle.verifyPositions.length
          ? cycle.verifyPositions[i].inputTokenId
          : i === 0
            ? inputPendingToken
            : cycle.emittedTokens[i - 1];
      consumedTokens[i] = tok;
      branch.committedTokens.push(tok);
    }

    for (let i = 0; i < cycle.emittedTokens.length; i += 1) {
      branch.emittedTokens.push(cycle.emittedTokens[i]);
    }
    branch.pendingToken = cycle.nextPendingToken;
    branch.haltReason = cycle.haltReason;
    branch.nextStepIndex = stepIndex + 1;
    branch.nextForwardStepId = forwardStepId + 1;

    const specSummary = cloneSpecSummary(summarizeSelfSpecCycle(cycle));
    const recordId = allocateRecordId();
    const createdAtSeq = nextSeq();

    const record: DorySessionRecord = Object.freeze({
      recordId,
      branchId: branch.branchId,
      parentRecordId,
      stepIndex,
      forwardStepId,
      kind: 'spec_cycle',
      kLoops: branch.config.kLoops,
      kvCacheMode: branch.config.kvCacheMode,
      executionMode: branch.config.executionMode,
      sampling: branch.config.sampling,
      startPos: cycle.startCommittedLen,
      endPos: cycle.endCommittedLen,
      committedLen: cycle.endCommittedLen,
      pendingToken: cycle.nextPendingToken,
      inputTokens: consumedTokens,
      emittedTokens: copyInt32FromArrayLike(cycle.emittedTokens),
      committedSequence: copyInt32FromArrayLike(branch.committedTokens),
      chunkSizesBefore,
      chunkSizesAfter,
      lastTokenScaledLogits: copyFloat32(cycle.lastCommittedDeepScaledLogits),
      lastTokenProbs: copyFloat64(cycle.lastCommittedDeepProbs),
      lastTokenXAnchor: copyFloat32(cycle.lastCommittedXAnchor),
      lastTokenLoopEntryStates: Object.freeze(
        cycle.lastCommittedLoopEntryStates.map((s) => copyFloat32(s)),
      ),
      lastTokenLoopExitStates: Object.freeze(
        cycle.lastCommittedLoopExitStates.map((s) => copyFloat32(s)),
      ),
      lastTokenLoopMetrics: cloneLoopMetrics(cycle.lastCommittedLoopMetrics),
      specSummary,
      haltReason: cycle.haltReason,
      createdAtSeq,
    });

    const replayMeta: InternalRecordReplayMeta = {
      preStepPrng,
      postStepPrng: branch.prng.clone(),
      explicitSampleUniform: undefined,
      specExplicitVariates: cloneSpecExplicitVariates(explicitVariates),
      specInputPendingToken: inputPendingToken,
      specGamma: branch.config.speculativeGamma,
      branchEmittedSnapshot: Object.freeze([...branch.emittedTokens]),
    };

    registerRecord(branch, record, replayMeta, true);
    return record;
  }

  // --------------------------------------------------------------------------
  // INTERNAL BRANCH CREATION (FRESH ROOT OR DR18 RE-PREFILL)
  // --------------------------------------------------------------------------

  function createFreshPrefilledBranch(args: {
    readonly label?: string;
    readonly promptTokens: readonly number[];
    readonly prefillChunkSizes?: readonly number[];
    readonly branchConfig: DorySessionBranchConfig;
    readonly parentBranchId: string | null;
    readonly forkRecordId: string | null;
    readonly forkReason: DorySessionForkReason;
    readonly initialSampleUniform?: number;
  }): DorySessionRecord {
    const {
      promptTokens,
      prefillChunkSizes,
      branchConfig,
      parentBranchId,
      forkRecordId,
      forkReason,
      initialSampleUniform,
    } = args;

    validateExplicitUniform(initialSampleUniform, 'initialSampleUniform');
    const chunks = validatePrefillChunkSizes(
      prefillChunkSizes,
      promptTokens.length,
      branchConfig.executionMode,
    );

    const branchId = allocateBranchId();
    const label = args.label ?? formatDefaultBranchLabel(branchId, branchConfig, forkReason);
    const createdAtSeq = nextSeq();
    const prng = createDoryPrng(branchConfig.seed);

    if (branchConfig.executionMode === 'self_spec') {
      const preStepPrng = prng.clone();
      const prefill: DorySpecCyclePrefillResult = prefillDorySelfSpec({
        promptTokens,
        params,
        sampling: branchConfig.sampling,
        prng,
        initialSampleUniform,
        ropeTables,
        sessionBranchId: branchId,
        forwardStepId: 0,
        captureTrace: false,
      });

      synchronizeCacheChunkSizes(prefill.kvCache);
      const chunkSizesAfter = Object.freeze([...prefill.kvCache.chunkSizes]);
      const d = config.hiddenSize;
      const lastVecOff = (prefill.committedLen - 1) * d;

      const committedArr = [...promptTokens];
      const emittedArr = [...promptTokens, prefill.pendingToken];

      const branch: InternalBranch = {
        branchId,
        label,
        parentBranchId,
        forkRecordId,
        forkReason,
        config: branchConfig,
        promptTokens: Object.freeze([...promptTokens]),
        committedTokens: committedArr,
        emittedTokens: emittedArr,
        recordIds: [],
        headRecordId: '',
        pendingToken: prefill.pendingToken,
        haltReason: prefill.haltReason,
        kvCache: prefill.kvCache,
        prng,
        nextStepIndex: 1,
        nextForwardStepId: 1,
        createdAtSeq,
        lastActiveSeq: createdAtSeq,
      };

      branchesById.set(branchId, branch);

      const recordId = allocateRecordId();
      const recSeq = nextSeq();
      const record: DorySessionRecord = Object.freeze({
        recordId,
        branchId,
        parentRecordId: forkRecordId,
        stepIndex: 0,
        forwardStepId: 0,
        kind: 'spec_prefill',
        kLoops: branchConfig.kLoops,
        kvCacheMode: branchConfig.kvCacheMode,
        executionMode: branchConfig.executionMode,
        sampling: branchConfig.sampling,
        startPos: 0,
        endPos: prefill.committedLen,
        committedLen: prefill.committedLen,
        pendingToken: prefill.pendingToken,
        inputTokens: copyInt32FromArrayLike(promptTokens),
        emittedTokens: Int32Array.of(prefill.pendingToken),
        committedSequence: copyInt32FromArrayLike(promptTokens),
        chunkSizesBefore: Object.freeze([]),
        chunkSizesAfter,
        lastTokenScaledLogits: copyFloat32(prefill.lastDeepScaledLogits),
        lastTokenProbs: copyFloat64(prefill.lastDeepProbs),
        lastTokenXAnchor: sliceCopyFloat32(prefill.xAnchor, lastVecOff, d),
        lastTokenLoopEntryStates: Object.freeze(
          prefill.loopEntryStates.map((s) => sliceCopyFloat32(s, lastVecOff, d)),
        ),
        lastTokenLoopExitStates: Object.freeze(
          prefill.loopExitStates.map((s) => sliceCopyFloat32(s, lastVecOff, d)),
        ),
        lastTokenLoopMetrics: cloneLoopMetrics(prefill.lastTokenLoopMetrics),
        specSummary: null,
        haltReason: prefill.haltReason,
        createdAtSeq: recSeq,
      });

      const replayMeta: InternalRecordReplayMeta = {
        preStepPrng,
        postStepPrng: prng.clone(),
        explicitSampleUniform: initialSampleUniform,
        specExplicitVariates: undefined,
        specInputPendingToken: null,
        specGamma: branchConfig.speculativeGamma,
        branchEmittedSnapshot: Object.freeze([...emittedArr]),
      };

      registerRecord(branch, record, replayMeta, true);
      const pinned = new Set<string>([branchId]);
      if (parentBranchId !== null) {
        pinned.add(parentBranchId);
      }
      enforceBranchCap(pinned);
      enforceRecordCap();
      return record;
    }

    // Standard mode: allocate fresh empty KV cache and execute exact prefill chunk schedule
    const kvCache = createDoryKvCache({
      mode: branchConfig.kvCacheMode,
      kLoops: branchConfig.kLoops,
      includeShallowCoda: false,
      config,
    });

    const branch: InternalBranch = {
      branchId,
      label,
      parentBranchId,
      forkRecordId,
      forkReason,
      config: branchConfig,
      promptTokens: Object.freeze([...promptTokens]),
      committedTokens: [],
      emittedTokens: [],
      recordIds: [],
      headRecordId: '',
      pendingToken: null,
      haltReason: null,
      kvCache,
      prng,
      nextStepIndex: 0,
      nextForwardStepId: 0,
      createdAtSeq,
      lastActiveSeq: createdAtSeq,
    };

    branchesById.set(branchId, branch);

    let offset = 0;
    let lastRecord: DorySessionRecord | null = null;
    for (let i = 0; i < chunks.length; i += 1) {
      const sz = chunks[i];
      const slice = promptTokens.slice(offset, offset + sz);
      offset += sz;
      const isLastChunk = i === chunks.length - 1;
      const kind: 'prefill' | 'chunk' = i === 0 ? 'prefill' : 'chunk';
      const parentRecId = i === 0 ? forkRecordId : lastRecord!.recordId;
      lastRecord = executeStandardChunkOnBranch(
        branch,
        slice,
        kind,
        parentRecId,
        isLastChunk,
        isLastChunk ? initialSampleUniform : undefined,
      );
    }

    const pinned = new Set<string>([branchId]);
    if (parentBranchId !== null) {
      pinned.add(parentBranchId);
    }
    enforceBranchCap(pinned);
    enforceRecordCap();
    return lastRecord!;
  }

  // --------------------------------------------------------------------------
  // PUBLIC `DorySession` METHODS
  // --------------------------------------------------------------------------

  function getBranch(branchId: string): DorySessionBranchSnapshot | null {
    const branch = branchesById.get(branchId);
    return branch !== undefined ? buildBranchSnapshot(branch) : null;
  }

  function getRecord(recordId: string): DorySessionRecord | null {
    return recordsById.get(recordId) ?? null;
  }

  function getActiveBranch(): DorySessionBranchSnapshot | null {
    if (activeBranchId === null) {
      return null;
    }
    return getBranch(activeBranchId);
  }

  function getActiveRecord(): DorySessionRecord | null {
    if (activeRecordId === null) {
      return null;
    }
    return getRecord(activeRecordId);
  }

  function listBranches(): readonly DorySessionBranchSnapshot[] {
    const ordered: InternalBranch[] = [];
    branchesById.forEach((b) => ordered.push(b));
    ordered.sort((a, b) => a.createdAtSeq - b.createdAtSeq);
    return Object.freeze(ordered.map((b) => buildBranchSnapshot(b)));
  }

  function listBranchRecords(branchId: string): readonly DorySessionRecord[] {
    const branch = branchesById.get(branchId);
    if (branch === undefined) {
      return Object.freeze([]);
    }
    const out: DorySessionRecord[] = [];
    for (let i = 0; i < branch.recordIds.length; i += 1) {
      const rec = recordsById.get(branch.recordIds[i]);
      if (rec !== undefined) {
        out.push(rec);
      }
    }
    return Object.freeze(out);
  }

  function getSnapshot(): DorySessionSnapshot {
    return Object.freeze({
      modelWeightId: params.modelWeightId,
      activeBranchId,
      activeRecordId,
      bounds,
      branches: listBranches(),
      recordCount: recordsById.size,
      checkpointStats: getCheckpointLruStats(),
    });
  }

  function setActiveBranch(branchId: string): DorySessionBranchSnapshot {
    const branch = branchesById.get(branchId);
    if (branch === undefined) {
      throw new RangeError(`Unknown DorySession branchId='${branchId}'.`);
    }
    activeBranchId = branch.branchId;
    activeRecordId = branch.headRecordId;
    branch.lastActiveSeq = nextSeq();
    return buildBranchSnapshot(branch);
  }

  function seekRecord(recordId: string): DorySessionRecord {
    const record = recordsById.get(recordId);
    if (record === undefined) {
      throw new RangeError(`Unknown DorySession recordId='${recordId}'.`);
    }
    const owner = branchesById.get(record.branchId);
    if (owner === undefined) {
      throw new RangeError(
        `Record '${recordId}' references evicted branch '${record.branchId}'.`,
      );
    }
    activeBranchId = owner.branchId;
    activeRecordId = record.recordId;
    owner.lastActiveSeq = nextSeq();
    return record;
  }

  function startBranch(startOptions: DorySessionStartBranchOptions = {}): DorySessionRecord {
    const promptTokens = validateTokenArray(
      startOptions.promptTokens ?? DORY_DEFAULT_SESSION_PROMPT,
      config,
      'promptTokens',
    );
    const branchConfig = resolveBranchConfig(config, startOptions);
    return createFreshPrefilledBranch({
      label: startOptions.label,
      promptTokens,
      prefillChunkSizes: startOptions.prefillChunkSizes,
      branchConfig,
      parentBranchId: null,
      forkRecordId: null,
      forkReason: 'initial',
      initialSampleUniform: startOptions.initialSampleUniform,
    });
  }

  function assertCompatibleBranchConfig(
    branchId: string,
    requested: Partial<Pick<DorySessionBranchConfig, 'kLoops' | 'kvCacheMode' | 'executionMode'>>,
  ): void {
    const branch = branchesById.get(branchId);
    if (branch === undefined) {
      throw new RangeError(`Unknown DorySession branchId='${branchId}'.`);
    }
    if (requested.kLoops !== undefined && requested.kLoops !== branch.config.kLoops) {
      throw new DorySessionImmutableConfigError(
        'DR18_INCOMPATIBLE_BRANCH_CONFIG',
        `DR18 immutable branch guard: branch '${branchId}' is locked to kLoops=${branch.config.kLoops}, cannot mutate to kLoops=${String(requested.kLoops)}. Use switchDepthOrPolicy() or forkBranch(..., { allowRePrefill: true }).`,
      );
    }
    if (
      requested.kvCacheMode !== undefined &&
      requested.kvCacheMode !== branch.config.kvCacheMode
    ) {
      throw new DorySessionImmutableConfigError(
        'DR18_INCOMPATIBLE_BRANCH_CONFIG',
        `DR18 immutable branch guard: branch '${branchId}' is locked to kvCacheMode='${branch.config.kvCacheMode}', cannot mutate to '${String(requested.kvCacheMode)}'. Use switchDepthOrPolicy() or forkBranch(..., { allowRePrefill: true }).`,
      );
    }
    if (
      requested.executionMode !== undefined &&
      requested.executionMode !== branch.config.executionMode
    ) {
      throw new DorySessionImmutableConfigError(
        'DR18_INCOMPATIBLE_BRANCH_CONFIG',
        `DR18 immutable branch guard: branch '${branchId}' is locked to executionMode='${branch.config.executionMode}', cannot mutate to '${String(requested.executionMode)}'. Use switchDepthOrPolicy() or forkBranch(..., { allowRePrefill: true }).`,
      );
    }
  }

  function reconstructKvCacheInternal(
    recordId: string,
    recordLruMetrics: boolean,
  ): DoryKvCache {
    const record = recordsById.get(recordId);
    if (record === undefined) {
      throw new RangeError(`Unknown DorySession recordId='${recordId}'.`);
    }

    // 1. Exact hit in D10 LRU checkpoint pool -> touch (when tracking metrics) & return non-aliased clone
    const exactCkpt = checkpointPool.get(recordId);
    if (exactCkpt !== undefined) {
      if (recordLruMetrics) {
        poolHitCount += 1;
        exactCkpt.lastUsedSeq = nextSeq();
      }
      return cloneDoryKvCache(exactCkpt.kvCache);
    }

    // 2. Live branch head -> clone directly (0 replay cost, strictly non-aliased!)
    const owner = branchesById.get(record.branchId);
    if (owner !== undefined && owner.headRecordId === recordId) {
      return cloneDoryKvCache(owner.kvCache);
    }

    // 3. Reconstruct from deepest matching LRU checkpoint prefix (or cold replay) using exact chunkSizesAfter (DR24)
    const built = buildKvCacheFromPrefix(
      record.branchId,
      record.kLoops,
      record.kvCacheMode,
      record.executionMode === 'self_spec',
      record.chunkSizesAfter,
      record.committedSequence,
      record.committedLen,
      recordLruMetrics,
    );
    return built.kvCache;
  }

  function reconstructKvCacheAtRecord(recordId: string): DoryKvCache {
    return reconstructKvCacheInternal(recordId, true);
  }

  function createCopyOnStepBranchFromRecord(
    sourceBranch: InternalBranch,
    sourceRecord: DorySessionRecord,
  ): InternalBranch {
    const sourceMeta = replayMetaByRecordId.get(sourceRecord.recordId);
    if (sourceMeta === undefined) {
      throw new Error(
        `Missing internal replay metadata for recordId='${sourceRecord.recordId}'.`,
      );
    }
    const clonedKv = reconstructKvCacheAtRecord(sourceRecord.recordId);
    const branchId = allocateBranchId();
    const createdAtSeq = nextSeq();
    const branch: InternalBranch = {
      branchId,
      label: formatDefaultBranchLabel(branchId, sourceBranch.config, 'cursor_fork'),
      parentBranchId: sourceBranch.branchId,
      forkRecordId: sourceRecord.recordId,
      forkReason: 'cursor_fork',
      config: sourceBranch.config,
      promptTokens: Object.freeze([...sourceBranch.promptTokens]),
      committedTokens: Array.from(sourceRecord.committedSequence),
      emittedTokens: [...sourceMeta.branchEmittedSnapshot],
      recordIds: [],
      headRecordId: '',
      pendingToken: sourceRecord.pendingToken,
      haltReason: sourceRecord.haltReason,
      kvCache: clonedKv,
      prng: sourceMeta.postStepPrng.clone(),
      nextStepIndex: sourceRecord.stepIndex + 1,
      nextForwardStepId: sourceRecord.forwardStepId + 1,
      createdAtSeq,
      lastActiveSeq: createdAtSeq,
    };
    branchesById.set(branchId, branch);
    sourceBranch.lastActiveSeq = nextSeq();
    return branch;
  }

  function appendChunk(
    chunkTokensInput: ArrayLike<number>,
    appendOptions: DorySessionAppendChunkOptions = {},
  ): DorySessionRecord {
    validateExplicitUniform(appendOptions.sampleUniform, 'sampleUniform');
    const chunkTokens = validateTokenArray(chunkTokensInput, config, 'chunkTokens');

    let targetBranch: InternalBranch;
    let parentRecordId: string;

    if (appendOptions.branchId !== undefined) {
      const found = branchesById.get(appendOptions.branchId);
      if (found === undefined) {
        throw new RangeError(`Unknown DorySession branchId='${appendOptions.branchId}'.`);
      }
      targetBranch = found;
      parentRecordId = found.headRecordId;
    } else {
      if (activeBranchId === null || activeRecordId === null) {
        throw new RangeError('Cannot appendChunk: no active branch in DorySession.');
      }
      const activeBranch = branchesById.get(activeBranchId);
      if (activeBranch === undefined) {
        throw new RangeError(`Active branch '${activeBranchId}' no longer exists.`);
      }
      if (activeBranch.config.executionMode !== 'standard') {
        throw new DorySessionImmutableConfigError(
          'DR18_INCOMPATIBLE_BRANCH_CONFIG',
          `appendChunk() is only supported on 'standard' branches; branch '${activeBranch.branchId}' uses executionMode='${activeBranch.config.executionMode}'.`,
        );
      }
      if (activeRecordId !== activeBranch.headRecordId) {
        const activeRecord = recordsById.get(activeRecordId);
        if (activeRecord === undefined) {
          throw new RangeError(`Unknown activeRecordId='${activeRecordId}'.`);
        }
        if (activeRecord.committedLen + chunkTokens.length > config.maxPositionEmbeddings) {
          throw new RangeError(
            `Capacity exceeded on copy-on-step appendChunk: committedLen(${activeRecord.committedLen}) + chunkLen(${chunkTokens.length}) > maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
          );
        }
        targetBranch = createCopyOnStepBranchFromRecord(activeBranch, activeRecord);
        parentRecordId = activeRecord.recordId;
        try {
          const record = executeStandardChunkOnBranch(
            targetBranch,
            chunkTokens,
            'chunk',
            parentRecordId,
            true,
            appendOptions.sampleUniform,
          );
          enforceBranchCap(new Set([targetBranch.branchId, activeBranch.branchId]));
          enforceRecordCap();
          return record;
        } catch (err) {
          if (targetBranch.recordIds.length === 0) {
            branchesById.delete(targetBranch.branchId);
          }
          throw err;
        }
      }
      targetBranch = activeBranch;
      parentRecordId = activeBranch.headRecordId;
    }

    if (targetBranch.config.executionMode !== 'standard') {
      throw new DorySessionImmutableConfigError(
        'DR18_INCOMPATIBLE_BRANCH_CONFIG',
        `appendChunk() is only supported on 'standard' branches; branch '${targetBranch.branchId}' uses executionMode='${targetBranch.config.executionMode}'.`,
      );
    }

    return executeStandardChunkOnBranch(
      targetBranch,
      chunkTokens,
      'chunk',
      parentRecordId,
      true,
      appendOptions.sampleUniform,
    );
  }

  function step(stepOptions: DorySessionStepOptions = {}): DorySessionRecord {
    validateExplicitUniform(stepOptions.sampleUniform, 'sampleUniform');
    validateSpecExplicitVariates(stepOptions.specExplicitVariates);

    let targetBranch: InternalBranch;
    let parentRecordId: string;
    let sourceBranchToPin: string | null = null;
    let provisionalBranchId: string | null = null;

    if (stepOptions.branchId !== undefined) {
      const found = branchesById.get(stepOptions.branchId);
      if (found === undefined) {
        throw new RangeError(`Unknown DorySession branchId='${stepOptions.branchId}'.`);
      }
      targetBranch = found;
      parentRecordId = found.headRecordId;
    } else {
      if (activeBranchId === null || activeRecordId === null) {
        throw new RangeError('Cannot step: no active branch in DorySession.');
      }
      const activeBranch = branchesById.get(activeBranchId);
      if (activeBranch === undefined) {
        throw new RangeError(`Active branch '${activeBranchId}' no longer exists.`);
      }
      if (activeRecordId !== activeBranch.headRecordId) {
        const activeRecord = recordsById.get(activeRecordId);
        if (activeRecord === undefined) {
          throw new RangeError(`Unknown activeRecordId='${activeRecordId}'.`);
        }
        if (activeRecord.haltReason !== null) {
          throw new RangeError(
            `Cannot copy-on-step from halted record '${activeRecord.recordId}' (haltReason='${activeRecord.haltReason}').`,
          );
        }
        if (activeRecord.committedLen >= config.maxPositionEmbeddings) {
          throw new RangeError(
            `Cannot copy-on-step from record '${activeRecord.recordId}': committedLen(${activeRecord.committedLen}) reached maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
          );
        }
        if (activeBranch.config.executionMode === 'self_spec' && activeRecord.pendingToken === null) {
          throw new RangeError(
            `Cannot copy-on-step self_spec record '${activeRecord.recordId}': pendingToken is null.`,
          );
        }
        if (activeBranch.config.executionMode === 'standard' && activeRecord.pendingToken === null) {
          const activeMeta = replayMetaByRecordId.get(activeRecord.recordId);
          if (activeMeta === undefined) {
            throw new RangeError(
              `Missing replay metadata for record '${activeRecord.recordId}'.`,
            );
          }
          const previewTok = sampleFromFilteredProbs(
            activeRecord.lastTokenProbs,
            activeBranch.config.sampling,
            activeMeta.postStepPrng.clone(),
            undefined,
          );
          if (isEosToken(previewTok)) {
            throw new RangeError(
              `Cannot copy-on-step from record '${activeRecord.recordId}': sampled pending token is EOS (${previewTok}).`,
            );
          }
        }
        // Non-aliased copy-on-step fork leaving original branch strictly immutable (DR18/DR19)
        targetBranch = createCopyOnStepBranchFromRecord(activeBranch, activeRecord);
        parentRecordId = activeRecord.recordId;
        sourceBranchToPin = activeBranch.branchId;
        provisionalBranchId = targetBranch.branchId;
      } else {
        targetBranch = activeBranch;
        parentRecordId = activeBranch.headRecordId;
      }
    }

    let resultRecord: DorySessionRecord;
    try {
      if (targetBranch.config.executionMode === 'self_spec') {
        resultRecord = executeSpecCycleOnBranch(
          targetBranch,
          parentRecordId,
          stepOptions.specExplicitVariates,
        );
      } else {
        if (targetBranch.haltReason !== null) {
          throw new RangeError(
            `Cannot step halted branch '${targetBranch.branchId}' (haltReason='${targetBranch.haltReason}').`,
          );
        }
        if (targetBranch.kvCache.committedLen >= config.maxPositionEmbeddings) {
          throw new RangeError(
            `Cannot step branch '${targetBranch.branchId}': committedLen(${targetBranch.kvCache.committedLen}) reached maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
          );
        }

        let tokenToConsume = targetBranch.pendingToken;
        if (tokenToConsume === null) {
          const headRec = recordsById.get(parentRecordId);
          if (headRec === undefined) {
            throw new RangeError(
              `Cannot step branch '${targetBranch.branchId}': pendingToken is null and parent record is missing.`,
            );
          }
          tokenToConsume = sampleFromFilteredProbs(
            headRec.lastTokenProbs,
            targetBranch.config.sampling,
            targetBranch.prng,
            undefined,
          );
          if (isEosToken(tokenToConsume)) {
            throw new RangeError(
              `Cannot step branch '${targetBranch.branchId}': sampled pending token is EOS (${tokenToConsume}).`,
            );
          }
        }

        resultRecord = executeStandardChunkOnBranch(
          targetBranch,
          [tokenToConsume],
          'decode_step',
          parentRecordId,
          true,
          stepOptions.sampleUniform,
        );
      }
    } catch (err) {
      if (provisionalBranchId !== null && targetBranch.recordIds.length === 0) {
        branchesById.delete(provisionalBranchId);
      }
      throw err;
    }

    if (sourceBranchToPin !== null) {
      enforceBranchCap(new Set([targetBranch.branchId, sourceBranchToPin]));
      enforceRecordCap();
    }
    return resultRecord;
  }

  function forkBranch(
    fromRecordId: string,
    forkOptions: DorySessionForkOptions = {},
  ): DorySessionRecord {
    const sourceRecord = recordsById.get(fromRecordId);
    if (sourceRecord === undefined) {
      throw new RangeError(`Unknown DorySession fromRecordId='${fromRecordId}'.`);
    }
    const sourceBranch = branchesById.get(sourceRecord.branchId);
    if (sourceBranch === undefined) {
      throw new RangeError(
        `Source record '${fromRecordId}' references evicted branch '${sourceRecord.branchId}'.`,
      );
    }
    const sourceMeta = replayMetaByRecordId.get(fromRecordId);
    if (sourceMeta === undefined) {
      throw new Error(`Missing replay metadata for source record '${fromRecordId}'.`);
    }

    const targetConfig = resolveBranchConfig(config, forkOptions, sourceBranch.config);
    const configChanged =
      targetConfig.kLoops !== sourceBranch.config.kLoops ||
      targetConfig.kvCacheMode !== sourceBranch.config.kvCacheMode ||
      targetConfig.executionMode !== sourceBranch.config.executionMode;

    if (configChanged) {
      if (forkOptions.allowRePrefill !== true) {
        throw new DorySessionImmutableConfigError(
          'DR18_REPREFILL_REQUIRED',
          `DR18 immutable fork guard: cannot fork compatible KV slabs across incompatible config change (source: k=${sourceBranch.config.kLoops}, mode=${sourceBranch.config.kvCacheMode}, exec=${sourceBranch.config.executionMode} -> target: k=${targetConfig.kLoops}, mode=${targetConfig.kvCacheMode}, exec=${targetConfig.executionMode}). Pass allowRePrefill: true or call switchDepthOrPolicy().`,
        );
      }
      sourceBranch.lastActiveSeq = nextSeq();
      return createFreshPrefilledBranch({
        label: forkOptions.label,
        promptTokens: sourceBranch.promptTokens,
        prefillChunkSizes: forkOptions.prefillChunkSizes,
        branchConfig: targetConfig,
        parentBranchId: sourceBranch.branchId,
        forkRecordId: sourceRecord.recordId,
        forkReason: 'dr18_re_prefill',
      });
    }

    // Compatible fork: reconstruct non-aliased KV slabs at `fromRecordId` (DR18/DR19)
    const clonedKv = reconstructKvCacheAtRecord(fromRecordId);
    const branchId = allocateBranchId();
    const label =
      forkOptions.label ?? formatDefaultBranchLabel(branchId, targetConfig, 'cursor_fork');
    const createdAtSeq = nextSeq();

    const useFreshSeedPrng =
      forkOptions.seed !== undefined && targetConfig.seed !== sourceBranch.config.seed;
    const branchPostPrng = useFreshSeedPrng
      ? createDoryPrng(targetConfig.seed)
      : sourceMeta.postStepPrng.clone();

    const committedArr = Array.from(sourceRecord.committedSequence);
    const emittedArr = [...sourceMeta.branchEmittedSnapshot];

    const newBranch: InternalBranch = {
      branchId,
      label,
      parentBranchId: sourceBranch.branchId,
      forkRecordId: sourceRecord.recordId,
      forkReason: 'cursor_fork',
      config: targetConfig,
      promptTokens: Object.freeze([...sourceBranch.promptTokens]),
      committedTokens: committedArr,
      emittedTokens: emittedArr,
      recordIds: [],
      headRecordId: '',
      pendingToken: sourceRecord.pendingToken,
      haltReason: sourceRecord.haltReason,
      kvCache: clonedKv,
      prng: branchPostPrng.clone(),
      nextStepIndex: sourceRecord.stepIndex + 1,
      nextForwardStepId: sourceRecord.forwardStepId + 1,
      createdAtSeq,
      lastActiveSeq: createdAtSeq,
    };

    sourceBranch.lastActiveSeq = nextSeq();
    branchesById.set(branchId, newBranch);

    // Owned non-aliased head record on the forked branch preserving 0-ULP replay fidelity
    const headRecordId = allocateRecordId();
    const recSeq = nextSeq();
    const forkedRecord: DorySessionRecord = Object.freeze({
      recordId: headRecordId,
      branchId,
      parentRecordId: sourceRecord.recordId,
      stepIndex: sourceRecord.stepIndex,
      forwardStepId: sourceRecord.forwardStepId,
      kind: sourceRecord.kind,
      kLoops: sourceRecord.kLoops,
      kvCacheMode: sourceRecord.kvCacheMode,
      executionMode: sourceRecord.executionMode,
      sampling: sourceRecord.sampling,
      startPos: sourceRecord.startPos,
      endPos: sourceRecord.endPos,
      committedLen: sourceRecord.committedLen,
      pendingToken: sourceRecord.pendingToken,
      inputTokens: copyInt32FromArrayLike(sourceRecord.inputTokens),
      emittedTokens: copyInt32FromArrayLike(sourceRecord.emittedTokens),
      committedSequence: copyInt32FromArrayLike(sourceRecord.committedSequence),
      chunkSizesBefore: Object.freeze([...sourceRecord.chunkSizesBefore]),
      chunkSizesAfter: Object.freeze([...sourceRecord.chunkSizesAfter]),
      lastTokenScaledLogits: copyFloat32(sourceRecord.lastTokenScaledLogits),
      lastTokenProbs: copyFloat64(sourceRecord.lastTokenProbs),
      lastTokenXAnchor: copyFloat32(sourceRecord.lastTokenXAnchor),
      lastTokenLoopEntryStates: Object.freeze(
        sourceRecord.lastTokenLoopEntryStates.map((s) => copyFloat32(s)),
      ),
      lastTokenLoopExitStates: Object.freeze(
        sourceRecord.lastTokenLoopExitStates.map((s) => copyFloat32(s)),
      ),
      lastTokenLoopMetrics: cloneLoopMetrics(sourceRecord.lastTokenLoopMetrics),
      specSummary: cloneSpecSummary(sourceRecord.specSummary),
      haltReason: sourceRecord.haltReason,
      createdAtSeq: recSeq,
    });

    const forkedReplayMeta: InternalRecordReplayMeta = {
      preStepPrng: sourceMeta.preStepPrng.clone(),
      postStepPrng: branchPostPrng.clone(),
      explicitSampleUniform: sourceMeta.explicitSampleUniform,
      specExplicitVariates: cloneSpecExplicitVariates(sourceMeta.specExplicitVariates),
      specInputPendingToken: sourceMeta.specInputPendingToken,
      specGamma: sourceMeta.specGamma,
      branchEmittedSnapshot: Object.freeze([...emittedArr]),
    };

    registerRecord(newBranch, forkedRecord, forkedReplayMeta, true);
    enforceBranchCap(new Set([branchId, sourceBranch.branchId]));
    enforceRecordCap();
    return forkedRecord;
  }

  function switchDepthOrPolicy(switchOptions: DorySessionSwitchConfigOptions): DorySessionRecord {
    const sourceBranchId = switchOptions.branchId ?? activeBranchId;
    if (sourceBranchId === null) {
      throw new RangeError('Cannot switchDepthOrPolicy: no active branch in DorySession.');
    }
    const sourceBranch = branchesById.get(sourceBranchId);
    if (sourceBranch === undefined) {
      throw new RangeError(`Unknown DorySession branchId='${sourceBranchId}'.`);
    }

    // DR18 / D6: ALWAYS fork/re-prefill a clean branch from prompt tokens & seed; NEVER mutate source KV slabs!
    const targetConfig = resolveBranchConfig(config, switchOptions, sourceBranch.config);
    sourceBranch.lastActiveSeq = nextSeq();

    return createFreshPrefilledBranch({
      label: switchOptions.label,
      promptTokens: sourceBranch.promptTokens,
      prefillChunkSizes: switchOptions.prefillChunkSizes,
      branchConfig: targetConfig,
      parentBranchId: sourceBranch.branchId,
      forkRecordId: sourceBranch.headRecordId,
      forkReason: 'dr18_re_prefill',
    });
  }

  function reconstructStepTrace(recordId: string): DoryReconstructedStepResult {
    const record = recordsById.get(recordId);
    if (record === undefined) {
      throw new RangeError(`Unknown DorySession recordId='${recordId}'.`);
    }
    const replayMeta = replayMetaByRecordId.get(recordId);
    if (replayMeta === undefined) {
      throw new Error(`Missing internal replay metadata for recordId='${recordId}'.`);
    }

    if (record.kind === 'spec_prefill') {
      const prefill: DorySpecCyclePrefillResult = prefillDorySelfSpec({
        promptTokens: record.inputTokens,
        params,
        sampling: record.sampling,
        prng: replayMeta.preStepPrng.clone(),
        initialSampleUniform: replayMeta.explicitSampleUniform,
        ropeTables,
        sessionBranchId: record.branchId,
        forwardStepId: record.forwardStepId,
        captureTrace: true,
      });
      synchronizeCacheChunkSizes(prefill.kvCache);
      return Object.freeze({
        recordId: record.recordId,
        branchId: record.branchId,
        kind: record.kind,
        usedCheckpoint: false,
        checkpointHitRecordId: null,
        replayedChunkSizes: Object.freeze([]),
        forwardResult: null,
        specPrefillResult: prefill,
        specCycleResult: null,
        postStepKvCache: prefill.kvCache,
      });
    }

    // Reconstruct pre-step KV cache strictly replaying `chunkSizesBefore` verbatim (DR24)
    const preBuilt = buildKvCacheFromPrefix(
      record.branchId,
      record.kLoops,
      record.kvCacheMode,
      record.executionMode === 'self_spec',
      record.chunkSizesBefore,
      record.committedSequence,
      record.startPos,
    );

    if (record.kind === 'spec_cycle') {
      if (replayMeta.specInputPendingToken === null) {
        throw new Error(
          `Missing specInputPendingToken in replay metadata for spec_cycle record '${recordId}'.`,
        );
      }
      const cycle: DorySelfSpecCycleResult = runDorySelfSpecCycle({
        kvCache: preBuilt.kvCache,
        pendingToken: replayMeta.specInputPendingToken,
        params,
        gamma: replayMeta.specGamma,
        sampling: record.sampling,
        prng: replayMeta.preStepPrng.clone(),
        explicitVariates: replayMeta.specExplicitVariates,
        ropeTables,
        sessionBranchId: record.branchId,
        forwardStepId: record.forwardStepId,
        captureTrace: true,
      });
      synchronizeCacheChunkSizes(preBuilt.kvCache);
      return Object.freeze({
        recordId: record.recordId,
        branchId: record.branchId,
        kind: record.kind,
        usedCheckpoint: preBuilt.usedCheckpoint,
        checkpointHitRecordId: preBuilt.checkpointHitRecordId,
        replayedChunkSizes: preBuilt.replayedChunkSizes,
        forwardResult: null,
        specPrefillResult: null,
        specCycleResult: cycle,
        postStepKvCache: preBuilt.kvCache,
      });
    }

    // Standard mode ('prefill' | 'chunk' | 'decode_step')
    const fwd: DoryForwardChunkResult = runDoryForwardChunk({
      tokens: record.inputTokens,
      params,
      kvCache: preBuilt.kvCache,
      startPos: record.startPos,
      kLoops: record.kLoops,
      kvCacheMode: record.kvCacheMode,
      ropeTables,
      codaKind: 'output_deep',
      writeKv: true,
      captureTrace: true,
      forwardStepId: record.forwardStepId,
      sessionBranchId: record.branchId,
      role: 'standard',
    });
    synchronizeCacheChunkSizes(preBuilt.kvCache);

    return Object.freeze({
      recordId: record.recordId,
      branchId: record.branchId,
      kind: record.kind,
      usedCheckpoint: preBuilt.usedCheckpoint,
      checkpointHitRecordId: preBuilt.checkpointHitRecordId,
      replayedChunkSizes: preBuilt.replayedChunkSizes,
      forwardResult: fwd,
      specPrefillResult: null,
      specCycleResult: null,
      postStepKvCache: preBuilt.kvCache,
    });
  }

  function evaluateCounterfactualIntermediateCoda(
    recordId: string,
    loopIteration: number,
  ): DoryCounterfactualCodaResult {
    const record = recordsById.get(recordId);
    if (record === undefined) {
      throw new RangeError(`Unknown DorySession recordId='${recordId}'.`);
    }
    if (
      !Number.isInteger(loopIteration) ||
      loopIteration < 1 ||
      loopIteration > record.kLoops
    ) {
      throw new RangeError(
        `Invalid counterfactual loopIteration=${String(loopIteration)} for record '${recordId}' with kLoops=${record.kLoops}.`,
      );
    }

    const absTokenPos = record.committedLen - 1;
    const state = record.lastTokenLoopExitStates[loopIteration - 1];
    // Reconstruct a strictly non-aliased post-step KV cache (without mutating LRU stats)
    // and run Deep Coda with writeKv=false so committed historical Deep Coda KV [0..absTokenPos-1]
    // is read with ZERO session or KV mutation.
    const readOnlyKv = reconstructKvCacheInternal(recordId, false);
    const coda = runDoryOutputCoda({
      state,
      seqLen: 1,
      startPos: absTokenPos,
      kLoops: record.kLoops,
      params,
      kvCache: readOnlyKv,
      ropeTables,
      codaKind: 'output_deep',
      writeKv: false,
      captureTrace: false,
      forwardStepId: record.forwardStepId,
      sessionBranchId: record.branchId,
      role: 'standard',
    });

    return Object.freeze({
      badge: DORY_COUNTERFACTUAL_CODA_BADGE,
      recordId: record.recordId,
      loopIteration,
      kLoops: record.kLoops,
      absTokenPos,
      codaHidden: coda.codaHidden,
      rawLogits: coda.lmHead.rawLogits,
      scaledLogits: coda.lmHead.scaledLogits,
      probs: coda.lmHead.lastTokenProbs,
    });
  }

  function measureMemory(): DorySessionMemoryAudit {
    const branchKvBuffers = new Set<ArrayBufferLike>();
    const checkpointBuffers = new Set<ArrayBufferLike>();
    const recordBuffers = new Set<ArrayBufferLike>();
    const sessionAllBuffers = new Set<ArrayBufferLike>();
    const slabOwnerSeen = new Set<ArrayBufferLike>();

    let nonAliasedSlabsVerified = true;

    branchesById.forEach((branch) => {
      const ok = collectKvCacheBuffers(
        branch.kvCache,
        branchKvBuffers,
        sessionAllBuffers,
        slabOwnerSeen,
      );
      if (!ok) {
        nonAliasedSlabsVerified = false;
      }
    });

    checkpointPool.forEach((entry) => {
      const ok = collectKvCacheBuffers(
        entry.kvCache,
        checkpointBuffers,
        sessionAllBuffers,
        slabOwnerSeen,
      );
      if (!ok) {
        nonAliasedSlabsVerified = false;
      }
    });

    recordsById.forEach((record) => {
      const ok = collectRecordBuffers(record, recordBuffers, sessionAllBuffers, slabOwnerSeen);
      if (!ok) {
        nonAliasedSlabsVerified = false;
      }
    });

    const branchKvBytes = sumUniqueBufferBytes(branchKvBuffers);
    const checkpointPoolBytes = sumUniqueBufferBytes(checkpointBuffers);
    const recordBytes = sumUniqueBufferBytes(recordBuffers);
    const totalSessionRetainedBytes = sumUniqueBufferBytes(sessionAllBuffers);

    const totalWithWeightsSet = new Set<ArrayBufferLike>(sessionAllBuffers);
    singletonWeightBuffers.forEach((buf) => {
      if (slabOwnerSeen.has(buf)) {
        nonAliasedSlabsVerified = false;
      }
      totalWithWeightsSet.add(buf);
    });
    const totalWithWeightsBytes = sumUniqueBufferBytes(totalWithWeightsSet);

    return Object.freeze({
      branchCount: branchesById.size,
      recordCount: recordsById.size,
      checkpointCount: checkpointPool.size,
      branchKvBytes,
      checkpointPoolBytes,
      recordBytes,
      totalSessionRetainedBytes,
      singletonWeightBytes,
      totalWithWeightsBytes,
      maxCheckpointPoolBytes: bounds.maxCheckpointPoolBytes,
      maxSessionRetainedBytes: bounds.maxSessionRetainedBytes,
      withinCheckpointPoolBudget: checkpointPoolBytes <= bounds.maxCheckpointPoolBytes,
      withinSessionBudget: totalSessionRetainedBytes <= bounds.maxSessionRetainedBytes,
      withinTotalBudgetWithWeights: totalWithWeightsBytes <= bounds.maxSessionRetainedBytes,
      nonAliasedSlabsVerified,
    });
  }

  function reset(resetOptions?: DorySessionStartBranchOptions): DorySessionRecord | null {
    branchesById.clear();
    recordsById.clear();
    replayMetaByRecordId.clear();
    checkpointPool.clear();

    activeBranchId = null;
    activeRecordId = null;
    seqCounter = 0;
    branchIdCounter = 0;
    recordIdCounter = 0;
    poolUsedBytes = 0;
    poolHitCount = 0;
    poolMissCount = 0;
    poolEvictionCount = 0;

    if (resetOptions !== undefined) {
      return startBranch(resetOptions);
    }
    if (autoStartInitialBranch) {
      return startBranch(initialBranchTemplate);
    }
    return null;
  }

  const session: DorySession = {
    params,
    ropeTables,
    bounds,
    getSnapshot,
    getActiveBranch,
    getActiveRecord,
    getBranch,
    getRecord,
    listBranches,
    listBranchRecords,
    setActiveBranch,
    seekRecord,
    startBranch,
    appendChunk,
    step,
    forkBranch,
    switchDepthOrPolicy,
    assertCompatibleBranchConfig,
    reconstructKvCacheAtRecord,
    reconstructStepTrace,
    evaluateCounterfactualIntermediateCoda,
    getCheckpointLruStats,
    measureMemory,
    reset,
  };

  if (autoStartInitialBranch) {
    startBranch(initialBranchTemplate);
  }

  return session;
}

/**
 * Deduplicated `Set<ArrayBufferLike>` memory audit helper for any `DorySession` (`D10`, `DR18`, `DR19`).
 */
export function measureDorySessionMemory(session: DorySession): DorySessionMemoryAudit {
  return session.measureMemory();
}
