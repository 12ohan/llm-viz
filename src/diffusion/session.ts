/**
 * Branching Interactive Diffusion Session with Bounded <=12.00 MiB Retention,
 * Copy-on-Step Branching & 0-ULP Lazy Reconstruction (`src/diffusion/session.ts`)
 *
 * Key invariants:
 *   1. Immutable per-position shared KV slabs (`20.00 KiB / tok`) across parent/child records.
 *   2. Strict `kvMaskProvenance` (`'token_causal' | 'block_bidirectional'`) isolation on config
 *      switches, throwing `KvProvenanceMismatchError` unless `allowExplicitRePrefill: true`.
 *   3. Copy-on-step branching whenever stepping from a rewound cursor before a branch tip.
 *   4. Hard memory hygiene: `eagerLayerTraces` (only `bounds.eagerTraceLayers` `[0, 19, 39]`)
 *      and `activeCroppedSuffixSlabs` are retained ONLY on the active cursor record and evicted
 *      immediately from all historical records on every step, seek, or reset.
 *   5. 0-ULP lazy reconstruction for any non-eager layer (`reconstructLayerTrace`) and any
 *      historical speculative verify cropped suffix (`reconstructSpecCroppedSuffixKvs`).
 *   6. Deduplicated `Set<ArrayBufferLike>` retention accounting (`measureMemory`) enforcing
 *      `<= 12.00 MiB` (`12,582,912 B`) under 12-branch / 128-record stress.
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import {
  DIFFUSION_DEFAULT_SAMPLING_SEED,
  addCounters,
  createEmptyGuardFlags,
  mergeGuardFlags,
  runDlmBlockCycle,
  sliceLastLogitsRow,
} from './execution';
import { runDiffusionForwardPass, runDiffusionLayer } from './forward';
import {
  appendNewKvsToSnapshot,
  assertKvProvenanceCompatible,
  assertKvSnapshotIntegrity,
  createEmptyKvSnapshot,
  cropKvSnapshot,
  extractCroppedSuffixSlabs,
  materializeLayerPastKv,
  materializePastKvs,
} from './kvCache';
import { createXoshiro128ss, mixSeed32 } from './prng';
import { computeYarnInvFreq } from './rope';
import { sampleCausalToken } from './sampler';
import { checkKvCapacityForStep } from './spec';
import { runLinearSpecCycle } from './speculative';
import {
  DEFAULT_DIFFUSION_SESSION_BOUNDS,
  type BrowserGuardFlags,
  type CroppedKvProvenance,
  type DiffusionExecutionMode,
  type DiffusionForwardResult,
  type DiffusionLayerKvSlice,
  type DiffusionLayerTrace,
  type DiffusionLifecyclePhase,
  type DiffusionModelParameters,
  type DiffusionPassCheckpointData,
  type DiffusionSessionBounds,
  type DiffusionSessionConfig,
  type DiffusionSessionConfigPatch,
  type DiffusionSessionMemoryReport,
  type DiffusionSessionRecord,
  type DiffusionSessionRecordKind,
  type ForwardCounters,
  type KvMaskProvenance,
  type KvPositionSlab,
  type PersistentKvSnapshot,
  type Xoshiro128ssGenerator,
} from './types';

// ============================================================================
// PUBLIC SESSION INTERFACES & DEFAULTS
// ============================================================================

export const DEFAULT_DIFFUSION_PROMPT_TOKENS: readonly number[] = Object.freeze([
  0, 1, 2, 3, 4, 5, 6, 7,
]);

export interface DiffusionSessionBranch {
  readonly branchId: string;
  readonly parentBranchId: string | null;
  readonly forkRecordId: string | null;
  readonly recordIds: readonly string[];
  readonly createdAtSeq: number;
  readonly lastActiveSeq: number;
}

export interface DiffusionSessionState {
  readonly activeRecordId: string | null;
  readonly activeBranchId: string;
  readonly branches: readonly DiffusionSessionBranch[];
  readonly records: ReadonlyMap<string, DiffusionSessionRecord>;
  readonly config: DiffusionSessionConfig;
  readonly bounds: DiffusionSessionBounds;
}

export interface DiffusionStepExplicitVariates {
  readonly prefillUniform?: number;
  readonly arUniform?: number;
  readonly dlmGumbelUniforms?: readonly Float64Array[];
  readonly dlmCommitUniform?: number;
  readonly specDraftUniforms?: readonly Float64Array[];
  readonly specVerifyUniforms?: Float64Array;
}

export interface CreateDiffusionSessionOptions {
  readonly bounds?: Partial<DiffusionSessionBounds>;
  readonly config?: DiffusionSessionConfigPatch;
  readonly promptTokenIds?: ArrayLike<number>;
}

const EMPTY_LAYER_SET: ReadonlySet<number> = Object.freeze(new Set<number>());
const EMPTY_LAYER_TRACE_MAP: ReadonlyMap<number, DiffusionLayerTrace> = new Map<
  number,
  DiffusionLayerTrace
>();

// ============================================================================
// VALIDATION & HELPER UTILITIES
// ============================================================================

interface MutableBranch {
  readonly branchId: string;
  readonly parentBranchId: string | null;
  readonly forkRecordId: string | null;
  recordIds: string[];
  readonly createdAtSeq: number;
  lastActiveSeq: number;
}

function requiredKvProvenanceForConfig(config: DiffusionSessionConfig): KvMaskProvenance {
  return config.mode === 'dlm' && !config.causalContext
    ? 'block_bidirectional'
    : 'token_causal';
}

function validateMode(mode: DiffusionExecutionMode): DiffusionExecutionMode {
  if (
    mode !== 'ar' &&
    mode !== 'dlm' &&
    mode !== 'linear_spec' &&
    mode !== 'linear_spec_lora'
  ) {
    throw new RangeError(`Unsupported diffusion mode: ${String(mode)}`);
  }
  return mode;
}

function validateBlockLength(blockLength: 4 | 8): 4 | 8 {
  if (blockLength !== 4 && blockLength !== 8) {
    throw new RangeError(
      `Session blockLength must be 4 or 8, received ${String(blockLength)}`,
    );
  }
  return blockLength;
}

function validateBounds(
  model: DiffusionModelParameters,
  partial?: Partial<DiffusionSessionBounds>,
): DiffusionSessionBounds {
  const maxBranches = partial?.maxBranches ?? DEFAULT_DIFFUSION_SESSION_BOUNDS.maxBranches;
  const maxRecords = partial?.maxRecords ?? DEFAULT_DIFFUSION_SESSION_BOUNDS.maxRecords;
  const kvCapacity = partial?.kvCapacity ?? model.config.kvCapacity;
  const maxRetainedBytes =
    partial?.maxRetainedBytes ?? DEFAULT_DIFFUSION_SESSION_BOUNDS.maxRetainedBytes;
  const rawLayers =
    partial?.eagerTraceLayers ?? DEFAULT_DIFFUSION_SESSION_BOUNDS.eagerTraceLayers;

  if (!Number.isInteger(maxBranches) || maxBranches < 1) {
    throw new RangeError(`maxBranches must be an integer >= 1, received ${maxBranches}`);
  }
  if (!Number.isInteger(maxRecords) || maxRecords < 1) {
    throw new RangeError(`maxRecords must be an integer >= 1, received ${maxRecords}`);
  }
  if (!Number.isInteger(kvCapacity) || kvCapacity < 1) {
    throw new RangeError(`kvCapacity must be an integer >= 1, received ${kvCapacity}`);
  }
  if (!Number.isInteger(maxRetainedBytes) || maxRetainedBytes < 1) {
    throw new RangeError(
      `maxRetainedBytes must be an integer >= 1, received ${maxRetainedBytes}`,
    );
  }

  const dedupedLayers: number[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < rawLayers.length; i++) {
    const l = rawLayers[i]!;
    if (!Number.isInteger(l) || l < 0 || l >= model.config.numLayers) {
      throw new RangeError(
        `eagerTraceLayers[${i}] (${l}) must be in [0, ${model.config.numLayers - 1}]`,
      );
    }
    if (!seen.has(l)) {
      seen.add(l);
      dedupedLayers.push(l);
    }
  }

  return Object.freeze({
    maxBranches,
    maxRecords,
    kvCapacity,
    maxRetainedBytes,
    eagerTraceLayers: Object.freeze(dedupedLayers),
  });
}

function createDefaultConfig(model: DiffusionModelParameters): DiffusionSessionConfig {
  const defaultB = model.config.defaultBlockLength;
  return Object.freeze({
    mode: 'ar' as DiffusionExecutionMode,
    causalContext: true,
    blockLength: defaultB,
    maxNewTokens: 16,
    temperature: 0,
    samplerAThreshold: null,
    samplerBThreshold: 0.0,
    maxDraftPassesPerCycle: defaultB,
    samplingSeed: DIFFUSION_DEFAULT_SAMPLING_SEED,
  });
}

function mergeSessionConfig(
  base: DiffusionSessionConfig,
  patch: DiffusionSessionConfigPatch | undefined,
): DiffusionSessionConfig {
  if (!patch) return base;
  const mode = validateMode(patch.mode ?? base.mode);
  const causalContext = patch.causalContext ?? base.causalContext;
  const blockLength = validateBlockLength(patch.blockLength ?? base.blockLength);
  const maxNewTokens = patch.maxNewTokens ?? base.maxNewTokens;
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1) {
    throw new RangeError(
      `maxNewTokens must be an integer >= 1 (D22 guard), received ${maxNewTokens}`,
    );
  }
  const temperature = patch.temperature ?? base.temperature;
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new RangeError(`temperature must be finite >= 0, received ${temperature}`);
  }
  const samplerAThreshold =
    patch.samplerAThreshold === undefined
      ? base.samplerAThreshold
      : patch.samplerAThreshold;
  if (samplerAThreshold !== null && !Number.isFinite(samplerAThreshold)) {
    throw new RangeError(
      `samplerAThreshold must be null or finite, received ${samplerAThreshold}`,
    );
  }
  const samplerBThreshold = patch.samplerBThreshold ?? base.samplerBThreshold;
  if (!Number.isFinite(samplerBThreshold) || samplerBThreshold < 0) {
    throw new RangeError(
      `samplerBThreshold must be finite >= 0, received ${samplerBThreshold}`,
    );
  }
  const defaultDraftCap =
    patch.blockLength !== undefined && base.maxDraftPassesPerCycle === base.blockLength
      ? blockLength
      : base.maxDraftPassesPerCycle;
  const maxDraftPassesPerCycle = patch.maxDraftPassesPerCycle ?? defaultDraftCap;
  if (!Number.isInteger(maxDraftPassesPerCycle) || maxDraftPassesPerCycle < 1) {
    throw new RangeError(
      `maxDraftPassesPerCycle must be an integer >= 1, received ${maxDraftPassesPerCycle}`,
    );
  }
  const samplingSeed = patch.samplingSeed ?? base.samplingSeed;
  if (!Number.isInteger(samplingSeed)) {
    throw new RangeError(`samplingSeed must be an integer, received ${samplingSeed}`);
  }

  return Object.freeze({
    mode,
    causalContext,
    blockLength,
    maxNewTokens,
    temperature,
    samplerAThreshold,
    samplerBThreshold,
    maxDraftPassesPerCycle,
    samplingSeed,
  });
}

function validatePromptTokens(
  model: DiffusionModelParameters,
  promptTokenIds: ArrayLike<number>,
  kvCapacity: number,
): Int32Array {
  const len = promptTokenIds.length;
  if (len < 1) {
    throw new RangeError('promptTokenIds must contain at least 1 token');
  }
  if (len > kvCapacity) {
    throw new RangeError(
      `promptTokenIds.length (${len}) exceeds session kvCapacity (${kvCapacity})`,
    );
  }
  const vocabSize = model.config.vocabSize;
  const out = new Int32Array(len);
  for (let i = 0; i < len; i++) {
    const id = promptTokenIds[i];
    if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
      throw new RangeError(
        `promptTokenIds[${i}] (${id}) must be an integer in [0, ${vocabSize - 1}]`,
      );
    }
    out[i] = id;
  }
  return out;
}

function extractPassCheckpoint(
  forward: DiffusionForwardResult,
): DiffusionPassCheckpointData {
  return Object.freeze({
    inputTokenIds: forward.tokenIds,
    positionIds: forward.positionIds,
    cachePositions: forward.cachePositions,
    maskMode: forward.maskMode,
    loraEnabled: forward.loraEnabled,
    useCache: forward.useCache,
    residualCheckpoints: Object.freeze(forward.residualCheckpoints.slice()),
    finalNorm: forward.finalNorm,
    logits: forward.logits,
    probs: forward.probs,
  });
}

function filterEagerTraces(
  traces: ReadonlyMap<number, DiffusionLayerTrace>,
  allowedLayers: ReadonlySet<number>,
): ReadonlyMap<number, DiffusionLayerTrace> {
  if (allowedLayers.size === 0 || traces.size === 0) {
    return EMPTY_LAYER_TRACE_MAP;
  }
  const out = new Map<number, DiffusionLayerTrace>();
  for (const layerIdx of allowedLayers) {
    const trace = traces.get(layerIdx);
    if (trace !== undefined) {
      out.set(layerIdx, trace);
    }
  }
  return out;
}

function appendInt32(base: Int32Array, extra: ArrayLike<number>): Int32Array {
  if (extra.length === 0) return base;
  const out = new Int32Array(base.length + extra.length);
  out.set(base, 0);
  for (let i = 0; i < extra.length; i++) {
    out[base.length + i] = extra[i]!;
  }
  return out;
}

function accountBufferView(
  view: ArrayBufferView | null | undefined,
  seen: Set<ArrayBufferLike>,
): number {
  if (!view) return 0;
  const buf = view.buffer;
  if (seen.has(buf)) return 0;
  seen.add(buf);
  return buf.byteLength;
}

function accountLayerTrace(
  trace: DiffusionLayerTrace,
  seen: Set<ArrayBufferLike>,
): number {
  let bytes = 0;
  bytes += accountBufferView(trace.inputNorm, seen);
  bytes += accountBufferView(trace.postAttentionResidual, seen);
  bytes += accountBufferView(trace.postAttentionNorm, seen);
  bytes += accountBufferView(trace.output, seen);

  const a = trace.attention;
  bytes += accountBufferView(a.qPre, seen);
  bytes += accountBufferView(a.kPre, seen);
  bytes += accountBufferView(a.vNew, seen);
  bytes += accountBufferView(a.qRot, seen);
  bytes += accountBufferView(a.kRot, seen);
  bytes += accountBufferView(a.llama4Scales, seen);
  bytes += accountBufferView(a.qScaled, seen);
  bytes += accountBufferView(a.kFull, seen);
  bytes += accountBufferView(a.vFull, seen);
  bytes += accountBufferView(a.scores, seen);
  bytes += accountBufferView(a.probs, seen);
  bytes += accountBufferView(a.attnConcat, seen);
  bytes += accountBufferView(a.oProjBase, seen);
  bytes += accountBufferView(a.loraHidden, seen);
  bytes += accountBufferView(a.loraDelta, seen);
  bytes += accountBufferView(a.output, seen);

  const s = trace.swiglu;
  bytes += accountBufferView(s.gatePre, seen);
  bytes += accountBufferView(s.gateSilu, seen);
  bytes += accountBufferView(s.upVal, seen);
  bytes += accountBufferView(s.gated, seen);
  bytes += accountBufferView(s.output, seen);

  return bytes;
}

// ============================================================================
// DIFFUSION SESSION CLASS
// ============================================================================

export class DiffusionSession {
  readonly model: DiffusionModelParameters;
  private readonly bounds: DiffusionSessionBounds;
  private readonly eagerTraceLayerSet: ReadonlySet<number>;
  private config: DiffusionSessionConfig;
  private promptTokenIds: Int32Array;

  private readonly branches = new Map<string, MutableBranch>();
  private readonly records = new Map<string, DiffusionSessionRecord>();
  private readonly prngSnapshots = new Map<string, Xoshiro128ssGenerator>();

  private activeBranchId: string;
  private activeRecordId: string | null = null;
  private nextBranchOrdinal = 1;
  private nextRecordOrdinal = 1;
  private mutationSeq = 0;

  constructor(
    model: DiffusionModelParameters,
    options?: CreateDiffusionSessionOptions,
  ) {
    this.model = model;
    this.bounds = validateBounds(model, options?.bounds);
    this.eagerTraceLayerSet = Object.freeze(new Set(this.bounds.eagerTraceLayers));
    this.config = mergeSessionConfig(createDefaultConfig(model), options?.config);
    this.promptTokenIds = validatePromptTokens(
      model,
      options?.promptTokenIds ?? DEFAULT_DIFFUSION_PROMPT_TOKENS,
      this.bounds.kvCapacity,
    );

    const rootBranchId = `branch-${this.nextBranchOrdinal++}`;
    this.branches.set(rootBranchId, {
      branchId: rootBranchId,
      parentBranchId: null,
      forkRecordId: null,
      recordIds: [],
      createdAtSeq: ++this.mutationSeq,
      lastActiveSeq: this.mutationSeq,
    });
    this.activeBranchId = rootBranchId;
  }

  // --------------------------------------------------------------------------
  // READ-ONLY QUERIES
  // --------------------------------------------------------------------------

  getState(): DiffusionSessionState {
    return Object.freeze({
      activeRecordId: this.activeRecordId,
      activeBranchId: this.activeBranchId,
      branches: this.getBranches(),
      records: new Map(this.records),
      config: this.config,
      bounds: this.bounds,
    });
  }

  getActiveRecord(): DiffusionSessionRecord | null {
    if (this.activeRecordId === null) return null;
    return this.records.get(this.activeRecordId) ?? null;
  }

  getRecord(recordId: string): DiffusionSessionRecord | null {
    return this.records.get(recordId) ?? null;
  }

  getBranches(): readonly DiffusionSessionBranch[] {
    const out: DiffusionSessionBranch[] = [];
    for (const b of this.branches.values()) {
      out.push(
        Object.freeze({
          branchId: b.branchId,
          parentBranchId: b.parentBranchId,
          forkRecordId: b.forkRecordId,
          recordIds: Object.freeze(b.recordIds.slice()),
          createdAtSeq: b.createdAtSeq,
          lastActiveSeq: b.lastActiveSeq,
        }),
      );
    }
    return Object.freeze(out);
  }

  getBranchRecords(branchId: string): readonly DiffusionSessionRecord[] {
    const branch = this.branches.get(branchId);
    if (!branch) {
      throw new RangeError(`Unknown branchId: "${branchId}"`);
    }
    const out: DiffusionSessionRecord[] = [];
    for (let i = 0; i < branch.recordIds.length; i++) {
      const rec = this.records.get(branch.recordIds[i]!);
      if (rec) out.push(rec);
    }
    return Object.freeze(out);
  }

  // --------------------------------------------------------------------------
  // CONFIGURATION UPDATE (NON-RETROACTIVE + PROVENANCE GUARD)
  // --------------------------------------------------------------------------

  updateConfig(patch: DiffusionSessionConfigPatch): DiffusionSessionConfig {
    const nextConfig = mergeSessionConfig(this.config, patch);
    const active = this.getActiveRecord();

    if (active !== null && active.kvSnapshot.seqLen > 0) {
      const requiredProvenance = requiredKvProvenanceForConfig(nextConfig);
      if (active.kvSnapshot.kvMaskProvenance !== requiredProvenance) {
        if (!patch.allowExplicitRePrefill) {
          assertKvProvenanceCompatible(
            active.kvSnapshot,
            requiredProvenance,
            'DiffusionSession.updateConfig',
          );
        }
        // Explicit re-prefill of committed tokens under the new mask provenance
        this.config = nextConfig;
        this.executeExplicitRePrefill(active, nextConfig, requiredProvenance);
        return this.config;
      }
    }

    this.config = nextConfig;
    return this.config;
  }

  // --------------------------------------------------------------------------
  // STEPPING & EXECUTION
  // --------------------------------------------------------------------------

  step(explicitVariates?: DiffusionStepExplicitVariates): DiffusionSessionRecord {
    const active = this.getActiveRecord();

    // Initial prefill transition
    if (active === null) {
      return this.executeInitialPrefill(explicitVariates);
    }

    // Check if already terminal (allow continuation if maxNewTokens was raised)
    const isMaxTokensStillReached =
      active.lifecyclePhase === 'completed_max_tokens' &&
      active.generatedTokenIds.length >= this.config.maxNewTokens;
    if (
      active.lifecyclePhase === 'completed_eos' ||
      active.lifecyclePhase === 'capacity_exhausted' ||
      isMaxTokensStillReached
    ) {
      return active;
    }

    // Enforce KV provenance compatibility & integrity before mutating
    const requiredProvenance = requiredKvProvenanceForConfig(this.config);
    assertKvProvenanceCompatible(
      active.kvSnapshot,
      requiredProvenance,
      'DiffusionSession.step',
    );
    assertKvSnapshotIntegrity(active.kvSnapshot);

    const prng = this.derivePrngFromRecord(active, this.config);

    switch (this.config.mode) {
      case 'ar':
        return this.executeArStep(active, prng, explicitVariates);
      case 'dlm':
        return this.executeDlmBlockStep(active, prng, explicitVariates);
      case 'linear_spec':
      case 'linear_spec_lora':
        return this.executeSpecCycleStep(active, prng, explicitVariates);
      default: {
        const exhaustive: never = this.config.mode;
        throw new RangeError(`Unsupported mode: ${String(exhaustive)}`);
      }
    }
  }

  runToCompletion(maxSteps?: number): readonly DiffusionSessionRecord[] {
    const stepLimit = maxSteps ?? Math.max(this.bounds.maxRecords, 128);
    if (!Number.isInteger(stepLimit) || stepLimit < 0) {
      throw new RangeError(`maxSteps must be an integer >= 0, received ${stepLimit}`);
    }

    const produced: DiffusionSessionRecord[] = [];
    for (let i = 0; i < stepLimit; i++) {
      const current = this.getActiveRecord();
      if (current !== null) {
        const isTerminal =
          current.lifecyclePhase === 'completed_eos' ||
          current.lifecyclePhase === 'capacity_exhausted' ||
          (current.lifecyclePhase === 'completed_max_tokens' &&
            current.generatedTokenIds.length >= this.config.maxNewTokens);
        if (isTerminal) break;
      }

      const prevId = this.activeRecordId;
      const next = this.step();
      if (next.recordId === prevId) break;
      produced.push(next);

      if (
        next.lifecyclePhase === 'completed_eos' ||
        next.lifecyclePhase === 'completed_max_tokens' ||
        next.lifecyclePhase === 'capacity_exhausted'
      ) {
        break;
      }
    }

    return Object.freeze(produced);
  }

  // --------------------------------------------------------------------------
  // NAVIGATION, BRANCH MANAGEMENT & RESET
  // --------------------------------------------------------------------------

  seekRecord(recordId: string): DiffusionSessionRecord {
    const existing = this.records.get(recordId);
    if (!existing) {
      throw new RangeError(`Unknown recordId: "${recordId}"`);
    }

    const targetBranch = this.findBranchContainingRecord(recordId, existing.branchId);
    if (targetBranch) {
      this.activeBranchId = targetBranch.branchId;
      targetBranch.lastActiveSeq = ++this.mutationSeq;
    }

    this.activeRecordId = recordId;
    this.evictNonActiveCaches();
    const hydrated = this.rehydrateActiveRecordCaches(recordId);
    return hydrated;
  }

  stepBack(): DiffusionSessionRecord | null {
    if (this.activeRecordId === null) return null;
    const branch = this.branches.get(this.activeBranchId);
    if (!branch) return null;
    const idx = branch.recordIds.indexOf(this.activeRecordId);
    if (idx <= 0) return null;
    return this.seekRecord(branch.recordIds[idx - 1]!);
  }

  stepForward(): DiffusionSessionRecord | null {
    if (this.activeRecordId === null) return null;
    const branch = this.branches.get(this.activeBranchId);
    if (!branch) return null;
    const idx = branch.recordIds.indexOf(this.activeRecordId);
    if (idx < 0 || idx >= branch.recordIds.length - 1) return null;
    return this.seekRecord(branch.recordIds[idx + 1]!);
  }

  deleteBranch(branchId: string): boolean {
    if (!this.branches.has(branchId)) return false;
    if (this.branches.size <= 1) return false;

    const wasActive = this.activeBranchId === branchId;
    this.branches.delete(branchId);

    if (wasActive) {
      let bestBranch: MutableBranch | null = null;
      for (const b of this.branches.values()) {
        if (bestBranch === null || b.lastActiveSeq > bestBranch.lastActiveSeq) {
          bestBranch = b;
        }
      }
      if (bestBranch) {
        this.activeBranchId = bestBranch.branchId;
        bestBranch.lastActiveSeq = ++this.mutationSeq;
        this.activeRecordId =
          bestBranch.recordIds.length > 0
            ? bestBranch.recordIds[bestBranch.recordIds.length - 1]!
            : null;
      } else {
        this.activeRecordId = null;
      }
    }

    this.garbageCollectUnreferencedRecords();
    this.evictNonActiveCaches();
    if (this.activeRecordId !== null) {
      this.rehydrateActiveRecordCaches(this.activeRecordId);
    }
    return true;
  }

  reset(
    promptTokenIds?: ArrayLike<number>,
    configPatch?: DiffusionSessionConfigPatch,
  ): void {
    if (configPatch !== undefined) {
      this.config = mergeSessionConfig(this.config, configPatch);
    }
    if (promptTokenIds !== undefined) {
      this.promptTokenIds = validatePromptTokens(
        this.model,
        promptTokenIds,
        this.bounds.kvCapacity,
      );
    }

    this.records.clear();
    this.branches.clear();
    this.prngSnapshots.clear();
    this.nextBranchOrdinal = 1;
    this.nextRecordOrdinal = 1;
    this.mutationSeq = 0;

    const rootBranchId = `branch-${this.nextBranchOrdinal++}`;
    this.branches.set(rootBranchId, {
      branchId: rootBranchId,
      parentBranchId: null,
      forkRecordId: null,
      recordIds: [],
      createdAtSeq: ++this.mutationSeq,
      lastActiveSeq: this.mutationSeq,
    });
    this.activeBranchId = rootBranchId;
    this.activeRecordId = null;
  }

  // --------------------------------------------------------------------------
  // 0-ULP LAZY RECONSTRUCTION
  // --------------------------------------------------------------------------

  reconstructLayerTrace(recordId: string, layerIndex: number): DiffusionLayerTrace {
    const record = this.records.get(recordId);
    if (!record) {
      throw new RangeError(`Unknown recordId: "${recordId}"`);
    }
    if (
      !Number.isInteger(layerIndex) ||
      layerIndex < 0 ||
      layerIndex >= this.model.config.numLayers
    ) {
      throw new RangeError(
        `layerIndex (${layerIndex}) must be in [0, ${this.model.config.numLayers - 1}]`,
      );
    }

    const cached = record.eagerLayerTraces.get(layerIndex);
    if (cached !== undefined) {
      return cached;
    }

    const cp = record.primaryPassCheckpoint;
    const seqLen = cp.inputTokenIds.length;
    const hiddenStates = cp.residualCheckpoints[layerIndex];
    if (!hiddenStates) {
      throw new RangeError(
        `Missing residualCheckpoint[${layerIndex}] on record "${recordId}"`,
      );
    }

    const pastLen = cp.positionIds[0] ?? 0;
    let pastKv: DiffusionLayerKvSlice | null = null;
    if (pastLen > 0) {
      const parent =
        record.parentRecordId !== null
          ? this.records.get(record.parentRecordId)
          : undefined;
      if (parent && parent.kvSnapshot.seqLen === pastLen) {
        pastKv = materializeLayerPastKv(parent.kvSnapshot, layerIndex);
      } else {
        const prefixSnap = cropKvSnapshot(record.kvSnapshot, pastLen);
        pastKv = materializeLayerPastKv(prefixSnap, layerIndex);
      }
    }

    const { invFreq, attentionScaling } = computeYarnInvFreq(this.model.config.rope);
    const { trace } = runDiffusionLayer(
      {
        hiddenStates,
        seqLen,
        layerWeights: this.model.layers[layerIndex]!,
        config: this.model.config,
        invFreq,
        attentionScaling,
        positionIds: cp.positionIds,
        cachePositions: cp.cachePositions,
        maskMode: cp.maskMode,
        loraEnabled: cp.loraEnabled,
        useCache: cp.useCache,
        pastKv,
      },
      layerIndex,
    );

    return trace;
  }

  reconstructSpecCroppedSuffixKvs(recordId: string): readonly KvPositionSlab[] {
    const record = this.records.get(recordId);
    if (!record) {
      throw new RangeError(`Unknown recordId: "${recordId}"`);
    }
    if (record.activeCroppedSuffixSlabs !== null) {
      return record.activeCroppedSuffixSlabs;
    }
    if (
      record.kind !== 'spec_cycle' ||
      !record.specMeta ||
      !record.cropProvenance ||
      record.cropProvenance.croppedSuffixLen === 0
    ) {
      return Object.freeze([] as KvPositionSlab[]);
    }

    const pastLen = record.cropProvenance.cacheLenBeforeVerify;
    const parent =
      record.parentRecordId !== null
        ? this.records.get(record.parentRecordId)
        : undefined;
    const prefixSnap =
      parent && parent.kvSnapshot.seqLen === pastLen
        ? parent.kvSnapshot
        : cropKvSnapshot(record.kvSnapshot, pastLen);

    const pastKvs = materializePastKvs(prefixSnap);
    const verifyForward = runDiffusionForwardPass(this.model, {
      tokenIds: record.specMeta.draftedBlockTokens,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
      pastKvs,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    const preCropSnap = appendNewKvsToSnapshot(
      prefixSnap,
      verifyForward.newKvs,
      record.specMeta.draftedBlockTokens,
      'token_causal',
    );
    return extractCroppedSuffixSlabs(
      preCropSnap,
      record.cropProvenance.browserPostEosCropLen,
    );
  }

  // --------------------------------------------------------------------------
  // DEDUPLICATED MEMORY ACCOUNTING (`Set<ArrayBufferLike>`, <= 12.00 MiB)
  // --------------------------------------------------------------------------

  measureMemory(): DiffusionSessionMemoryReport {
    const seenBuffers = new Set<ArrayBufferLike>();
    const seenSlabs = new Set<KvPositionSlab>();

    let committedKvSlabBytes = 0;
    let activeCroppedSuffixBytes = 0;
    let checkpointsAndLogitsBytes = 0;
    let activeEagerTracesBytes = 0;

    // 1. Committed KV slabs (structurally shared across records & branches)
    for (const record of this.records.values()) {
      const slabs = record.kvSnapshot.slabs;
      for (let i = 0; i < slabs.length; i++) {
        const slab = slabs[i]!;
        seenSlabs.add(slab);
        committedKvSlabBytes +=
          accountBufferView(slab.k, seenBuffers) +
          accountBufferView(slab.v, seenBuffers);
      }
    }

    // 2. Active cropped suffix slabs (retained ONLY on active cursor record)
    for (const record of this.records.values()) {
      const suffix = record.activeCroppedSuffixSlabs;
      if (suffix !== null) {
        for (let i = 0; i < suffix.length; i++) {
          const slab = suffix[i]!;
          activeCroppedSuffixBytes +=
            accountBufferView(slab.k, seenBuffers) +
            accountBufferView(slab.v, seenBuffers);
        }
      }
    }

    // 3. Checkpoints, logits, probabilities, and token arrays
    checkpointsAndLogitsBytes += accountBufferView(this.promptTokenIds, seenBuffers);
    for (const record of this.records.values()) {
      checkpointsAndLogitsBytes += accountBufferView(
        record.generatedTokenIds,
        seenBuffers,
      );
      checkpointsAndLogitsBytes += accountBufferView(
        record.stepEmittedTokenIds,
        seenBuffers,
      );

      const cp = record.primaryPassCheckpoint;
      checkpointsAndLogitsBytes += accountBufferView(cp.inputTokenIds, seenBuffers);
      checkpointsAndLogitsBytes += accountBufferView(cp.positionIds, seenBuffers);
      checkpointsAndLogitsBytes += accountBufferView(cp.cachePositions, seenBuffers);
      for (let i = 0; i < cp.residualCheckpoints.length; i++) {
        checkpointsAndLogitsBytes += accountBufferView(
          cp.residualCheckpoints[i],
          seenBuffers,
        );
      }
      checkpointsAndLogitsBytes += accountBufferView(cp.finalNorm, seenBuffers);
      checkpointsAndLogitsBytes += accountBufferView(cp.logits, seenBuffers);
      checkpointsAndLogitsBytes += accountBufferView(cp.probs, seenBuffers);

      if (record.specMeta) {
        checkpointsAndLogitsBytes += accountBufferView(
          record.specMeta.draftedBlockTokens,
          seenBuffers,
        );
        checkpointsAndLogitsBytes += accountBufferView(
          record.specMeta.verifyArTokens,
          seenBuffers,
        );
      }
      if (record.dlmMeta) {
        checkpointsAndLogitsBytes += accountBufferView(
          record.dlmMeta.finalBlockTokens,
          seenBuffers,
        );
      }
    }

    // 4. Active eager layer traces (retained ONLY on active cursor record)
    for (const record of this.records.values()) {
      for (const trace of record.eagerLayerTraces.values()) {
        activeEagerTracesBytes += accountLayerTrace(trace, seenBuffers);
      }
    }

    const totalRetainedBytes =
      committedKvSlabBytes +
      activeCroppedSuffixBytes +
      checkpointsAndLogitsBytes +
      activeEagerTracesBytes;

    return Object.freeze({
      totalRetainedBytes,
      maxRetainedBytes: this.bounds.maxRetainedBytes,
      withinBudget: totalRetainedBytes <= this.bounds.maxRetainedBytes,
      committedKvSlabBytes,
      activeCroppedSuffixBytes,
      checkpointsAndLogitsBytes,
      activeEagerTracesBytes,
      uniqueBufferCount: seenBuffers.size,
      uniqueSlabCount: seenSlabs.size,
      branchCount: this.branches.size,
      recordCount: this.records.size,
      activeRecordId: this.activeRecordId,
    });
  }

  // --------------------------------------------------------------------------
  // INTERNAL TRANSITION IMPLEMENTATIONS
  // --------------------------------------------------------------------------

  private derivePrngFromRecord(
    parent: DiffusionSessionRecord,
    config: DiffusionSessionConfig,
  ): Xoshiro128ssGenerator {
    if (config.samplingSeed !== parent.configAtStep.samplingSeed) {
      return createXoshiro128ss(mixSeed32(config.samplingSeed, parent.recordIndex + 1));
    }
    const snap = this.prngSnapshots.get(parent.recordId);
    if (snap) {
      return snap.clone();
    }
    return createXoshiro128ss(mixSeed32(config.samplingSeed, parent.recordIndex + 1));
  }

  private executeInitialPrefill(
    explicitVariates?: DiffusionStepExplicitVariates,
  ): DiffusionSessionRecord {
    const cfg = this.config;
    const kvMaskProvenance = requiredKvProvenanceForConfig(cfg);
    const isCausal = kvMaskProvenance === 'token_causal';
    const vocabSize = this.model.config.vocabSize;
    const eosTokenId = this.model.config.eosTokenId;

    const prng = createXoshiro128ss(cfg.samplingSeed);
    const emptyKv = createEmptyKvSnapshot(kvMaskProvenance, this.bounds.kvCapacity);
    const prefillForward = runDiffusionForwardPass(this.model, {
      tokenIds: this.promptTokenIds,
      maskMode: isCausal ? 'causal' : 'bidirectional',
      loraEnabled: false,
      useCache: true,
      retainedTraceLayers: this.eagerTraceLayerSet,
    });
    const kvSnapshot = appendNewKvsToSnapshot(
      emptyKv,
      prefillForward.newKvs,
      this.promptTokenIds,
      kvMaskProvenance,
    );

    const isDlm = cfg.mode === 'dlm';
    const counters: ForwardCounters = Object.freeze({
      actualForwardPasses: 1,
      decodeForwardPasses: 0,
      referenceMethodNfe: isDlm ? 0 : 1,
    });

    let generatedTokenIds = new Int32Array(0);
    let stepEmittedTokenIds = new Int32Array(0);
    let pendingSeedToken: number | null = null;
    let lifecyclePhase: DiffusionLifecyclePhase = 'running';

    if (isCausal) {
      const sample = sampleCausalToken(
        sliceLastLogitsRow(prefillForward, vocabSize),
        cfg.temperature,
        prng,
        explicitVariates?.prefillUniform,
      );
      if (isDlm) {
        // DLM causalContext=true: pendingSeed seeds slot 0 of block 0 (emitted at block 0 commit)
        pendingSeedToken = sample.tokenId;
      } else {
        // AR / Linear Spec: prefill emits token 0 immediately
        stepEmittedTokenIds = Int32Array.of(sample.tokenId);
        generatedTokenIds = Int32Array.of(sample.tokenId);
        if (sample.tokenId === eosTokenId) {
          lifecyclePhase = 'completed_eos';
          pendingSeedToken = null;
        } else {
          pendingSeedToken = sample.tokenId;
          if (generatedTokenIds.length >= cfg.maxNewTokens) {
            lifecyclePhase = 'completed_max_tokens';
          }
        }
      }
    }

    return this.commitNewRecord(
      null,
      {
        kind: 'prefill',
        configAtStep: cfg,
        lifecyclePhase,
        kvMaskProvenance,
        kvSnapshot,
        generatedTokenIds,
        stepEmittedTokenIds,
        pendingSeedToken,
        counters,
        guardFlags: createEmptyGuardFlags(),
        cropProvenance: null,
        primaryPassCheckpoint: extractPassCheckpoint(prefillForward),
        eagerLayerTraces: filterEagerTraces(
          prefillForward.layerTraces,
          this.eagerTraceLayerSet,
        ),
        activeCroppedSuffixSlabs: null,
      },
      prng,
    );
  }

  private executeExplicitRePrefill(
    parent: DiffusionSessionRecord,
    nextConfig: DiffusionSessionConfig,
    newProvenance: KvMaskProvenance,
  ): DiffusionSessionRecord {
    const isCausal = newProvenance === 'token_causal';
    const baseCommittedTokens = Int32Array.from(parent.kvSnapshot.committedTokenIds);
    const vocabSize = this.model.config.vocabSize;
    const eosTokenId = this.model.config.eosTokenId;
    const prng = this.derivePrngFromRecord(parent, nextConfig);

    let committedTokens: Int32Array = baseCommittedTokens;
    let generatedTokenIds: Int32Array = parent.generatedTokenIds;
    const extraWarnings: string[] = [];

    // Handle token_causal -> block_bidirectional when an already-emitted pendingSeedToken is not yet in KV
    if (!isCausal && this.isSeedAlreadyEmitted(parent) && parent.pendingSeedToken !== null) {
      if (baseCommittedTokens.length < this.bounds.kvCapacity) {
        committedTokens = appendInt32(
          baseCommittedTokens,
          Int32Array.of(parent.pendingSeedToken),
        );
      } else {
        const keepGenCount = Math.max(
          0,
          baseCommittedTokens.length - this.promptTokenIds.length,
        );
        generatedTokenIds = parent.generatedTokenIds.slice(0, keepGenCount);
        extraWarnings.push(
          `Pending emitted seed token (${parent.pendingSeedToken}) omitted at kvCapacity (${this.bounds.kvCapacity}) during bidirectional re-prefill to keep generated tokens aligned with KV.`,
        );
      }
    }

    const emptyKv = createEmptyKvSnapshot(newProvenance, this.bounds.kvCapacity);
    const prefillForward = runDiffusionForwardPass(this.model, {
      tokenIds: committedTokens,
      maskMode: isCausal ? 'causal' : 'bidirectional',
      loraEnabled: false,
      useCache: true,
      retainedTraceLayers: this.eagerTraceLayerSet,
    });
    const kvSnapshot = appendNewKvsToSnapshot(
      emptyKv,
      prefillForward.newKvs,
      committedTokens,
      newProvenance,
    );

    const isDlm = nextConfig.mode === 'dlm';
    const counters = addCounters(parent.counters, {
      actualForwardPasses: 1,
      decodeForwardPasses: 0,
      referenceMethodNfe: isDlm ? 0 : 1,
    });

    let stepEmittedTokenIds = new Int32Array(0);
    let pendingSeedToken: number | null = null;
    let lifecyclePhase: DiffusionLifecyclePhase =
      parent.lifecyclePhase === 'completed_eos' ? 'completed_eos' : 'running';

    if (isCausal && lifecyclePhase !== 'completed_eos') {
      const sample = sampleCausalToken(
        sliceLastLogitsRow(prefillForward, vocabSize),
        nextConfig.temperature,
        prng,
      );
      if (isDlm) {
        pendingSeedToken = sample.tokenId;
      } else {
        stepEmittedTokenIds = Int32Array.of(sample.tokenId);
        generatedTokenIds = appendInt32(generatedTokenIds, stepEmittedTokenIds);
        if (sample.tokenId === eosTokenId) {
          lifecyclePhase = 'completed_eos';
          pendingSeedToken = null;
        } else {
          pendingSeedToken = sample.tokenId;
          if (generatedTokenIds.length >= nextConfig.maxNewTokens) {
            lifecyclePhase = 'completed_max_tokens';
          }
        }
      }
    }

    return this.commitNewRecord(
      parent,
      {
        kind: 'prefill',
        configAtStep: nextConfig,
        lifecyclePhase,
        kvMaskProvenance: newProvenance,
        kvSnapshot,
        generatedTokenIds,
        stepEmittedTokenIds,
        pendingSeedToken,
        counters,
        guardFlags: mergeGuardFlags(parent.guardFlags, {
          draftGuardTriggered: 'none',
          residualMasksCommittedCount: 0,
          trailingMasksAfterEosCount: 0,
          capacityExhausted: false,
          warnings: Object.freeze([
            `Explicit re-prefill (${parent.kvMaskProvenance} -> ${newProvenance}) executed over ${committedTokens.length} committed tokens`,
            ...extraWarnings,
          ]),
        }),
        cropProvenance: null,
        primaryPassCheckpoint: extractPassCheckpoint(prefillForward),
        eagerLayerTraces: filterEagerTraces(
          prefillForward.layerTraces,
          this.eagerTraceLayerSet,
        ),
        activeCroppedSuffixSlabs: null,
      },
      prng,
    );
  }

  private isSeedAlreadyEmitted(record: DiffusionSessionRecord): boolean {
    return (
      this.promptTokenIds.length + record.generatedTokenIds.length >
      record.kvSnapshot.seqLen
    );
  }

  private executeArStep(
    parent: DiffusionSessionRecord,
    prng: Xoshiro128ssGenerator,
    explicitVariates?: DiffusionStepExplicitVariates,
  ): DiffusionSessionRecord {
    const cfg = this.config;
    const eosTokenId = this.model.config.eosTokenId;
    const vocabSize = this.model.config.vocabSize;

    const capCheck = checkKvCapacityForStep(
      'ar',
      parent.kvSnapshot.seqLen,
      this.bounds.kvCapacity,
      1,
    );
    if (!capCheck.canProceed) {
      return this.commitCapacityExhaustedRecord(
        parent,
        'ar_step',
        `KV capacity (${this.bounds.kvCapacity}) exhausted at seqLen=${parent.kvSnapshot.seqLen} before AR step`,
        prng,
      );
    }

    const seedAlreadyEmitted = this.isSeedAlreadyEmitted(parent);
    const inputToken =
      parent.pendingSeedToken ??
      (parent.generatedTokenIds.length > 0
        ? parent.generatedTokenIds[parent.generatedTokenIds.length - 1]!
        : this.promptTokenIds[this.promptTokenIds.length - 1]!);

    // If transitioning from DLM causalContext=true where pendingSeedToken was EOS
    if (!seedAlreadyEmitted && inputToken === eosTokenId) {
      const emitted = Int32Array.of(eosTokenId);
      return this.commitNewRecord(
        parent,
        {
          kind: 'ar_step',
          configAtStep: cfg,
          lifecyclePhase: 'completed_eos',
          kvMaskProvenance: 'token_causal',
          kvSnapshot: parent.kvSnapshot,
          generatedTokenIds: appendInt32(parent.generatedTokenIds, emitted),
          stepEmittedTokenIds: emitted,
          pendingSeedToken: null,
          counters: parent.counters,
          guardFlags: parent.guardFlags,
          cropProvenance: null,
          primaryPassCheckpoint: parent.primaryPassCheckpoint,
          eagerLayerTraces: EMPTY_LAYER_TRACE_MAP,
          activeCroppedSuffixSlabs: null,
        },
        prng,
      );
    }

    const pastKvs = materializePastKvs(parent.kvSnapshot);
    const stepForward = runDiffusionForwardPass(this.model, {
      tokenIds: [inputToken],
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
      pastKvs,
      retainedTraceLayers: this.eagerTraceLayerSet,
    });
    const kvSnapshot = appendNewKvsToSnapshot(
      parent.kvSnapshot,
      stepForward.newKvs,
      [inputToken],
      'token_causal',
    );
    const counters = addCounters(parent.counters, {
      actualForwardPasses: 1,
      decodeForwardPasses: 1,
      referenceMethodNfe: 1,
    });

    const stepSample = sampleCausalToken(
      sliceLastLogitsRow(stepForward, vocabSize),
      cfg.temperature,
      prng,
      explicitVariates?.arUniform,
    );

    const stepEmittedTokenIds = seedAlreadyEmitted
      ? Int32Array.of(stepSample.tokenId)
      : Int32Array.of(inputToken, stepSample.tokenId);
    const generatedTokenIds = appendInt32(parent.generatedTokenIds, stepEmittedTokenIds);

    let lifecyclePhase: DiffusionLifecyclePhase = 'running';
    let pendingSeedToken: number | null = stepSample.tokenId;
    if (stepSample.tokenId === eosTokenId) {
      lifecyclePhase = 'completed_eos';
      pendingSeedToken = null;
    } else if (generatedTokenIds.length >= cfg.maxNewTokens) {
      lifecyclePhase = 'completed_max_tokens';
    }

    return this.commitNewRecord(
      parent,
      {
        kind: 'ar_step',
        configAtStep: cfg,
        lifecyclePhase,
        kvMaskProvenance: 'token_causal',
        kvSnapshot,
        generatedTokenIds,
        stepEmittedTokenIds,
        pendingSeedToken,
        counters,
        guardFlags: parent.guardFlags,
        cropProvenance: null,
        primaryPassCheckpoint: extractPassCheckpoint(stepForward),
        eagerLayerTraces: filterEagerTraces(
          stepForward.layerTraces,
          this.eagerTraceLayerSet,
        ),
        activeCroppedSuffixSlabs: null,
      },
      prng,
    );
  }

  private executeDlmBlockStep(
    parent: DiffusionSessionRecord,
    prng: Xoshiro128ssGenerator,
    explicitVariates?: DiffusionStepExplicitVariates,
  ): DiffusionSessionRecord {
    const cfg = this.config;
    const capCheck = checkKvCapacityForStep(
      'dlm',
      parent.kvSnapshot.seqLen,
      this.bounds.kvCapacity,
      cfg.blockLength,
    );
    if (!capCheck.canProceed) {
      return this.commitCapacityExhaustedRecord(
        parent,
        'dlm_block',
        `KV capacity (${this.bounds.kvCapacity}) exhausted at seqLen=${parent.kvSnapshot.seqLen} for DLM blockLength=${cfg.blockLength} (B is never clamped)`,
        prng,
      );
    }

    const seedAlreadyEmitted = this.isSeedAlreadyEmitted(parent);
    const blockResult = runDlmBlockCycle(this.model, {
      kvSnapshot: parent.kvSnapshot,
      blockIndex: parent.recordIndex,
      blockLength: cfg.blockLength,
      causalContext: cfg.causalContext,
      initialSeedToken: cfg.causalContext ? parent.pendingSeedToken : null,
      temperature: cfg.temperature,
      samplerAThreshold: cfg.samplerAThreshold,
      prng,
      explicitBlockGumbelUniforms: explicitVariates?.dlmGumbelUniforms,
      explicitCommitUniform: explicitVariates?.dlmCommitUniform,
      retainedTraceLayers: this.eagerTraceLayerSet,
    });

    const stepEmittedTokenIds =
      cfg.causalContext && seedAlreadyEmitted
        ? blockResult.emittedTokens.slice(1)
        : blockResult.emittedTokens;
    const generatedTokenIds = appendInt32(parent.generatedTokenIds, stepEmittedTokenIds);

    let lifecyclePhase: DiffusionLifecyclePhase = 'running';
    if (blockResult.eosHit) {
      lifecyclePhase = 'completed_eos';
    } else if (generatedTokenIds.length >= cfg.maxNewTokens) {
      lifecyclePhase = 'completed_max_tokens';
    }

    const kvMaskProvenance: KvMaskProvenance = cfg.causalContext
      ? 'token_causal'
      : 'block_bidirectional';

    return this.commitNewRecord(
      parent,
      {
        kind: 'dlm_block',
        configAtStep: cfg,
        lifecyclePhase,
        kvMaskProvenance,
        kvSnapshot: blockResult.commitKvSnapshot,
        generatedTokenIds,
        stepEmittedTokenIds,
        pendingSeedToken: blockResult.nextSeedToken,
        counters: addCounters(parent.counters, blockResult.blockCounters),
        guardFlags: mergeGuardFlags(parent.guardFlags, blockResult.guardFlags),
        cropProvenance: null,
        primaryPassCheckpoint: extractPassCheckpoint(blockResult.commitForward),
        dlmMeta: Object.freeze({
          denoisePassCount: blockResult.denoisePassCount,
          finalBlockTokens: blockResult.finalBlockTokens,
          causalContext: cfg.causalContext,
        }),
        eagerLayerTraces: filterEagerTraces(
          blockResult.commitForward.layerTraces,
          this.eagerTraceLayerSet,
        ),
        activeCroppedSuffixSlabs: null,
      },
      prng,
    );
  }

  private executeSpecCycleStep(
    parent: DiffusionSessionRecord,
    prng: Xoshiro128ssGenerator,
    explicitVariates?: DiffusionStepExplicitVariates,
  ): DiffusionSessionRecord {
    const cfg = this.config;
    const loraEnabled = cfg.mode === 'linear_spec_lora';
    const eosTokenId = this.model.config.eosTokenId;
    const seedAlreadyEmitted = this.isSeedAlreadyEmitted(parent);
    const seedToken =
      parent.pendingSeedToken ??
      (parent.generatedTokenIds.length > 0
        ? parent.generatedTokenIds[parent.generatedTokenIds.length - 1]!
        : this.promptTokenIds[this.promptTokenIds.length - 1]!);

    // If transitioning from DLM causalContext=true where pendingSeedToken was EOS
    if (!seedAlreadyEmitted && seedToken === eosTokenId) {
      const emitted = Int32Array.of(eosTokenId);
      return this.commitNewRecord(
        parent,
        {
          kind: 'spec_cycle',
          configAtStep: cfg,
          lifecyclePhase: 'completed_eos',
          kvMaskProvenance: 'token_causal',
          kvSnapshot: parent.kvSnapshot,
          generatedTokenIds: appendInt32(parent.generatedTokenIds, emitted),
          stepEmittedTokenIds: emitted,
          pendingSeedToken: null,
          counters: parent.counters,
          guardFlags: parent.guardFlags,
          cropProvenance: null,
          primaryPassCheckpoint: parent.primaryPassCheckpoint,
          eagerLayerTraces: EMPTY_LAYER_TRACE_MAP,
          activeCroppedSuffixSlabs: null,
        },
        prng,
      );
    }

    const cycle = runLinearSpecCycle(this.model, {
      kvSnapshot: parent.kvSnapshot,
      seedToken,
      blockLength: cfg.blockLength,
      loraEnabled,
      samplerBConfig: {
        temperature: cfg.temperature,
        threshold: cfg.samplerBThreshold,
        maxDraftPassesPerCycle: cfg.maxDraftPassesPerCycle,
        maskTokenId: this.model.config.maskTokenId,
      },
      verifyTemperature: cfg.temperature,
      eosTokenId,
      prng,
      explicitDraftUniforms: explicitVariates?.specDraftUniforms,
      explicitVerifyUniforms: explicitVariates?.specVerifyUniforms,
      retainedTraceLayers: this.eagerTraceLayerSet,
    });

    if (cycle.capacityExhausted || cycle.verifyForward === null) {
      return this.commitCapacityExhaustedRecord(
        parent,
        'spec_cycle',
        cycle.guardFlags.warnings[0] ??
          `KV capacity (${this.bounds.kvCapacity}) exhausted at seqLen=${parent.kvSnapshot.seqLen} for spec blockLength=${cfg.blockLength}`,
        prng,
      );
    }

    const stepEmittedTokenIds = seedAlreadyEmitted
      ? cycle.emittedTokens
      : appendInt32(Int32Array.of(seedToken), cycle.emittedTokens);
    const generatedTokenIds = appendInt32(parent.generatedTokenIds, stepEmittedTokenIds);

    let lifecyclePhase: DiffusionLifecyclePhase = 'running';
    if (cycle.eosHit || cycle.lifecyclePhase === 'completed_eos') {
      lifecyclePhase = 'completed_eos';
    } else if (generatedTokenIds.length >= cfg.maxNewTokens) {
      lifecyclePhase = 'completed_max_tokens';
    }

    return this.commitNewRecord(
      parent,
      {
        kind: 'spec_cycle',
        configAtStep: cfg,
        lifecyclePhase,
        kvMaskProvenance: 'token_causal',
        kvSnapshot: cycle.postCropKvSnapshot,
        generatedTokenIds,
        stepEmittedTokenIds,
        pendingSeedToken: cycle.nextSeedToken,
        counters: addCounters(parent.counters, cycle.cycleCounters),
        guardFlags: mergeGuardFlags(parent.guardFlags, cycle.guardFlags),
        cropProvenance: cycle.cropProvenance,
        primaryPassCheckpoint: extractPassCheckpoint(cycle.verifyForward),
        specMeta: Object.freeze({
          draftedBlockTokens: cycle.draftedBlockTokens,
          draftPassCount: cycle.draftPassCount,
          verifyArTokens: cycle.verifyArTokens,
          matchedDraftCount: cycle.matchedDraftCount,
          acceptedCount: cycle.acceptedCount,
          loraEnabled,
        }),
        eagerLayerTraces: filterEagerTraces(
          cycle.verifyForward.layerTraces,
          this.eagerTraceLayerSet,
        ),
        activeCroppedSuffixSlabs: cycle.croppedSuffixSlabs,
      },
      prng,
    );
  }

  private commitCapacityExhaustedRecord(
    parent: DiffusionSessionRecord,
    kind: DiffusionSessionRecordKind,
    warning: string,
    prng: Xoshiro128ssGenerator,
  ): DiffusionSessionRecord {
    return this.commitNewRecord(
      parent,
      {
        kind,
        configAtStep: this.config,
        lifecyclePhase: 'capacity_exhausted',
        kvMaskProvenance: parent.kvMaskProvenance,
        kvSnapshot: parent.kvSnapshot,
        generatedTokenIds: parent.generatedTokenIds,
        stepEmittedTokenIds: new Int32Array(0),
        pendingSeedToken: parent.pendingSeedToken,
        counters: parent.counters,
        guardFlags: mergeGuardFlags(parent.guardFlags, {
          draftGuardTriggered: 'none',
          residualMasksCommittedCount: 0,
          trailingMasksAfterEosCount: 0,
          capacityExhausted: true,
          warnings: Object.freeze([warning]),
        }),
        cropProvenance: null,
        primaryPassCheckpoint: parent.primaryPassCheckpoint,
        eagerLayerTraces: EMPTY_LAYER_TRACE_MAP,
        activeCroppedSuffixSlabs: null,
      },
      prng,
    );
  }

  // --------------------------------------------------------------------------
  // BRANCH COMMIT, AUTO-FORK, EVICTION & PRUNING
  // --------------------------------------------------------------------------

  private commitNewRecord(
    parent: DiffusionSessionRecord | null,
    payload: {
      readonly kind: DiffusionSessionRecordKind;
      readonly configAtStep: DiffusionSessionConfig;
      readonly lifecyclePhase: DiffusionLifecyclePhase;
      readonly kvMaskProvenance: KvMaskProvenance;
      readonly kvSnapshot: PersistentKvSnapshot;
      readonly generatedTokenIds: Int32Array;
      readonly stepEmittedTokenIds: Int32Array;
      readonly pendingSeedToken: number | null;
      readonly counters: ForwardCounters;
      readonly guardFlags: BrowserGuardFlags;
      readonly cropProvenance: CroppedKvProvenance | null;
      readonly primaryPassCheckpoint: DiffusionPassCheckpointData;
      readonly specMeta?: DiffusionSessionRecord['specMeta'];
      readonly dlmMeta?: DiffusionSessionRecord['dlmMeta'];
      readonly eagerLayerTraces: ReadonlyMap<number, DiffusionLayerTrace>;
      readonly activeCroppedSuffixSlabs: readonly KvPositionSlab[] | null;
    },
    prngAfter: Xoshiro128ssGenerator,
  ): DiffusionSessionRecord {
    const seq = ++this.mutationSeq;
    let targetBranch = this.branches.get(this.activeBranchId)!;

    // Copy-on-step branching when stepping from a rewound cursor before branch tip
    if (parent !== null) {
      const parentPos = targetBranch.recordIds.indexOf(parent.recordId);
      const isAtTip =
        parentPos >= 0 && parentPos === targetBranch.recordIds.length - 1;
      if (!isAtTip) {
        const newBranchId = `branch-${this.nextBranchOrdinal++}`;
        const inheritedIds =
          parentPos >= 0
            ? targetBranch.recordIds.slice(0, parentPos + 1)
            : this.collectAncestorChain(parent.recordId);
        const forkedBranch: MutableBranch = {
          branchId: newBranchId,
          parentBranchId: targetBranch.branchId,
          forkRecordId: parent.recordId,
          recordIds: inheritedIds,
          createdAtSeq: seq,
          lastActiveSeq: seq,
        };
        this.branches.set(newBranchId, forkedBranch);
        this.activeBranchId = newBranchId;
        targetBranch = forkedBranch;
      }
    }

    const recordIndex = parent === null ? 0 : parent.recordIndex + 1;
    const recordId = `${targetBranch.branchId}:rec-${this.nextRecordOrdinal++}:${payload.kind}`;
    const record: DiffusionSessionRecord = Object.freeze({
      recordId,
      parentRecordId: parent === null ? null : parent.recordId,
      branchId: targetBranch.branchId,
      recordIndex,
      kind: payload.kind,
      configAtStep: payload.configAtStep,
      lifecyclePhase: payload.lifecyclePhase,
      kvMaskProvenance: payload.kvMaskProvenance,
      kvSnapshot: payload.kvSnapshot,
      generatedTokenIds: payload.generatedTokenIds,
      stepEmittedTokenIds: payload.stepEmittedTokenIds,
      pendingSeedToken: payload.pendingSeedToken,
      counters: payload.counters,
      guardFlags: payload.guardFlags,
      cropProvenance: payload.cropProvenance,
      prngStateAfter: prngAfter.getState(),
      primaryPassCheckpoint: payload.primaryPassCheckpoint,
      ...(payload.specMeta !== undefined ? { specMeta: payload.specMeta } : {}),
      ...(payload.dlmMeta !== undefined ? { dlmMeta: payload.dlmMeta } : {}),
      eagerLayerTraces: payload.eagerLayerTraces,
      activeCroppedSuffixSlabs: payload.activeCroppedSuffixSlabs,
    });

    this.records.set(recordId, record);
    this.prngSnapshots.set(recordId, prngAfter.clone());
    targetBranch.recordIds.push(recordId);
    targetBranch.lastActiveSeq = seq;
    this.activeRecordId = recordId;

    // Immediately evict eager traces & cropped suffix slabs from all historical records
    this.evictNonActiveCaches();
    // Enforce maxBranches & maxRecords while protecting the active ancestor chain
    this.enforceRetentionBounds();

    return this.records.get(recordId) ?? record;
  }

  private collectAncestorChain(recordId: string): string[] {
    const chain: string[] = [];
    let curId: string | null = recordId;
    const visited = new Set<string>();
    while (curId !== null && !visited.has(curId)) {
      visited.add(curId);
      const rec = this.records.get(curId);
      if (!rec) break;
      chain.push(curId);
      curId = rec.parentRecordId;
    }
    chain.reverse();
    return chain;
  }

  private findBranchContainingRecord(
    recordId: string,
    preferredBranchId: string,
  ): MutableBranch | null {
    const activeBranch = this.branches.get(this.activeBranchId);
    if (activeBranch && activeBranch.recordIds.includes(recordId)) {
      return activeBranch;
    }
    const preferred = this.branches.get(preferredBranchId);
    if (preferred && preferred.recordIds.includes(recordId)) {
      return preferred;
    }
    let best: MutableBranch | null = null;
    for (const b of this.branches.values()) {
      if (b.recordIds.includes(recordId)) {
        if (best === null || b.lastActiveSeq > best.lastActiveSeq) {
          best = b;
        }
      }
    }
    return best;
  }

  private evictNonActiveCaches(): void {
    for (const [id, rec] of this.records.entries()) {
      if (id === this.activeRecordId) continue;
      if (rec.eagerLayerTraces.size > 0 || rec.activeCroppedSuffixSlabs !== null) {
        this.records.set(
          id,
          Object.freeze({
            ...rec,
            eagerLayerTraces: EMPTY_LAYER_TRACE_MAP,
            activeCroppedSuffixSlabs: null,
          }),
        );
      }
    }
  }

  private rehydrateActiveRecordCaches(recordId: string): DiffusionSessionRecord {
    const rec = this.records.get(recordId)!;
    let eagerLayerTraces = rec.eagerLayerTraces;
    let needsUpdate = false;

    if (this.bounds.eagerTraceLayers.length > 0) {
      let missingAny = false;
      for (let i = 0; i < this.bounds.eagerTraceLayers.length; i++) {
        if (!eagerLayerTraces.has(this.bounds.eagerTraceLayers[i]!)) {
          missingAny = true;
          break;
        }
      }
      if (missingAny) {
        const rebuilt = new Map<number, DiffusionLayerTrace>();
        for (let i = 0; i < this.bounds.eagerTraceLayers.length; i++) {
          const l = this.bounds.eagerTraceLayers[i]!;
          rebuilt.set(l, this.reconstructLayerTrace(recordId, l));
        }
        eagerLayerTraces = rebuilt;
        needsUpdate = true;
      }
    }

    let activeCroppedSuffixSlabs = rec.activeCroppedSuffixSlabs;
    if (
      rec.kind === 'spec_cycle' &&
      rec.specMeta !== undefined &&
      rec.cropProvenance !== null &&
      activeCroppedSuffixSlabs === null
    ) {
      activeCroppedSuffixSlabs = this.reconstructSpecCroppedSuffixKvs(recordId);
      needsUpdate = true;
    }

    if (!needsUpdate) {
      return rec;
    }

    const updated: DiffusionSessionRecord = Object.freeze({
      ...rec,
      eagerLayerTraces,
      activeCroppedSuffixSlabs,
    });
    this.records.set(recordId, updated);
    return updated;
  }

  private garbageCollectUnreferencedRecords(): void {
    const referenced = new Set<string>();
    for (const b of this.branches.values()) {
      for (let i = 0; i < b.recordIds.length; i++) {
        referenced.add(b.recordIds[i]!);
      }
    }
    for (const id of Array.from(this.records.keys())) {
      if (!referenced.has(id)) {
        this.records.delete(id);
        this.prngSnapshots.delete(id);
      }
    }
  }

  private computeProtectedRecordIds(): Set<string> {
    const protectedIds = new Set<string>();
    if (this.activeRecordId === null) return protectedIds;
    const activeBranch = this.branches.get(this.activeBranchId);
    if (activeBranch) {
      const idx = activeBranch.recordIds.indexOf(this.activeRecordId);
      if (idx >= 0) {
        for (let i = 0; i <= idx; i++) {
          protectedIds.add(activeBranch.recordIds[i]!);
        }
        return protectedIds;
      }
    }
    const chain = this.collectAncestorChain(this.activeRecordId);
    for (let i = 0; i < chain.length; i++) {
      protectedIds.add(chain[i]!);
    }
    return protectedIds;
  }

  private enforceRetentionBounds(): void {
    // 1. Prune oldest inactive branches until branches.size <= maxBranches
    while (this.branches.size > this.bounds.maxBranches) {
      let victim: MutableBranch | null = null;
      for (const b of this.branches.values()) {
        if (b.branchId === this.activeBranchId) continue;
        if (
          victim === null ||
          b.lastActiveSeq < victim.lastActiveSeq ||
          (b.lastActiveSeq === victim.lastActiveSeq &&
            b.createdAtSeq < victim.createdAtSeq)
        ) {
          victim = b;
        }
      }
      if (victim === null) break;
      this.branches.delete(victim.branchId);
      this.garbageCollectUnreferencedRecords();
    }

    // 2. Prune records until records.size <= maxRecords while preserving active ancestor chain
    while (this.records.size > this.bounds.maxRecords) {
      const protectedIds = this.computeProtectedRecordIds();

      // Prefer deleting the oldest inactive branch that owns non-protected records
      let victimBranch: MutableBranch | null = null;
      for (const b of this.branches.values()) {
        if (b.branchId === this.activeBranchId) continue;
        const ownsUnprotected = b.recordIds.some((id) => !protectedIds.has(id));
        if (!ownsUnprotected) continue;
        if (
          victimBranch === null ||
          b.lastActiveSeq < victimBranch.lastActiveSeq ||
          (b.lastActiveSeq === victimBranch.lastActiveSeq &&
            b.createdAtSeq < victimBranch.createdAtSeq)
        ) {
          victimBranch = b;
        }
      }

      if (victimBranch !== null) {
        this.branches.delete(victimBranch.branchId);
        this.garbageCollectUnreferencedRecords();
        continue;
      }

      // Otherwise evict a single unprotected record (e.g. redo tail beyond cursor)
      let victimRecordId: string | null = null;
      for (const id of this.records.keys()) {
        if (!protectedIds.has(id)) {
          victimRecordId = id;
          break;
        }
      }

      // Last resort (single branch with > maxRecords ancestors): drop oldest ancestor != activeRecordId
      if (victimRecordId === null) {
        for (const id of this.records.keys()) {
          if (id !== this.activeRecordId) {
            victimRecordId = id;
            break;
          }
        }
      }

      if (victimRecordId === null) break;

      this.records.delete(victimRecordId);
      this.prngSnapshots.delete(victimRecordId);
      for (const b of this.branches.values()) {
        const idx = b.recordIds.indexOf(victimRecordId);
        if (idx >= 0) {
          b.recordIds.splice(idx, 1);
        }
      }
    }
  }
}

// ============================================================================
// FACTORY & MEMORY HELPER EXPORTS
// ============================================================================

export function createDiffusionSession(
  model: DiffusionModelParameters,
  options?: CreateDiffusionSessionOptions,
): DiffusionSession {
  return new DiffusionSession(model, options);
}

export function measureDiffusionSessionMemory(
  session: DiffusionSession,
): DiffusionSessionMemoryReport {
  return session.measureMemory();
}
