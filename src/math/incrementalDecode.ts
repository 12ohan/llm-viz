/**
 * Cached Incremental Autoregressive (AR) Decoding Orchestration (`src/math/incrementalDecode.ts`)
 *
 * Reuses the exact verified miniature kernels (`runMambaSim`, `runMoeSim`, `runGqaSim`,
 * `embedTokens`, `applyRmsNorm`, `runUntiedLmHead`) without duplicating mathematical logic.
 *
 * Architectural guarantees:
 *  1. Immutable-by-contract copy-on-step `HybridDecodeCacheState` snapshots: stepping or
 *     branching from a prior snapshot NEVER mutates any typed array in `prevState`.
 *  2. Exhaustive cache ownership & integrity validation: complete layer coverage (`23 M`,
 *     `6 *`, `23 E`), exact tensor shapes, non-aliased backing buffers, finite populated
 *     values, matching `parameterFingerprint`, and exact `75.00 KiB` (`76,800 B`) default
 *     allocated snapshot accounting.
 *  3. Bit-exact (`0` ulp) full-sequence vs. prefill+incremental AND arbitrary chunk
 *     equivalence across all 52 layers, Final RMSNorm, and Untied LM Head.
 *  4. Unambiguous trace provenance distinguishing `initialStateDigest`, `branchId`,
 *     `parentStepExecutionId`, `stepExecutionId`, `tokenRange`, and `traceOrigin`
 *     (`'computed_this_step'` vs. `'reconstructed_from_checkpoint'`).
 *
 * Epistemic disclaimer:
 *   Educational miniature Float32 simulator (`d = 8`, `V_mini = 32`, synthetic seeded weights).
 *   NOT 31.6B checkpoint inference and NOT real Nemotron tokenizer output.
 */

import { HYBRID_OVERRIDE_PATTERN, NEMOTRON_SPEC, type Kind, type PrecisionMode } from '../model/spec';
import {
  GQA_SIM_CONFIG,
  runGqaSim,
  type GqaSimFixture,
  type GqaSimResult,
} from './gqaSim';
import {
  MAMBA_SIM_CONFIG,
  runMambaSim,
  type MambaSimFixture,
  type MambaSimResult,
} from './mambaSim';
import {
  runMoeSim,
  type MoeSimFixture,
  type MoeSimResult,
} from './moeSim';
import {
  DEFAULT_RETAINED_TRACE_BAYS,
  applyRmsNorm,
  embedTokens,
  runUntiedLmHead,
  type HybridBaySummary,
  type HybridFinalNormResult,
  type HybridLmHeadResult,
  type HybridModelParameters,
  type HybridPrediction,
} from './hybridFlow';
import type {
  DecodeCacheMemoryAccounting,
  ExternalDecodeCacheSeed,
  GqaLayerKvCache,
  HybridDecodeCacheState,
  IncrementalDecodeStepInput,
  IncrementalDecodeStepKind,
  IncrementalDecodeStepResult,
  IncrementalStepProvenance,
  IncrementalTraceOrigin,
  MambaLayerBoundaryState,
} from './cacheContracts';

const DEFAULT_DECODE_MAX_TOKENS = 16 as const;
/** Maximum number of newly arrived tokens `S` accepted by a single incremental decode step. */
export const MAX_INCREMENTAL_STEP_TOKENS = 16 as const;
const DEFAULT_DECODE_PRECISION_METADATA: PrecisionMode = 'fp8_selective';
/** Default conservative cap on retained continuation branches per `HybridDecodeSession`. */
export const DEFAULT_SESSION_MAX_BRANCHES = 16 as const;
/** Default conservative cap on unique retained steps across all branches in a `HybridDecodeSession`. */
export const DEFAULT_SESSION_MAX_UNIQUE_STEPS = 128 as const;

/** Per-bay trace returned by an incremental step, tagged with unambiguous provenance and origin. */
export type HybridIncrementalLayerTrace =
  | {
      readonly stepExecutionId: string;
      readonly provenance: IncrementalStepProvenance;
      readonly traceOrigin: IncrementalTraceOrigin;
      readonly bayId: number;
      readonly kind: 'M';
      readonly sim: MambaSimResult;
    }
  | {
      readonly stepExecutionId: string;
      readonly provenance: IncrementalStepProvenance;
      readonly traceOrigin: IncrementalTraceOrigin;
      readonly bayId: number;
      readonly kind: 'E';
      readonly sim: MoeSimResult;
    }
  | {
      readonly stepExecutionId: string;
      readonly provenance: IncrementalStepProvenance;
      readonly traceOrigin: IncrementalTraceOrigin;
      readonly bayId: number;
      readonly kind: '*';
      readonly sim: GqaSimResult;
    };

/** Full incremental step execution result exposing both cached transition and truthful live traces. */
export interface HybridIncrementalStepResult extends IncrementalDecodeStepResult {
  /** Number of newly processed tokens `S` in this step (`1` for single-token AR decode). */
  readonly tokens: number;
  /** Number of reused cached prefix tokens `P` before this step. */
  readonly prefixTokens: number;
  /** Total sequence length `P + S` after this step. */
  readonly totalTokens: number;
  readonly d: 8;
  readonly vocabSize: number;
  readonly vocab: readonly string[];
  /** Token IDs newly processed in this step (length `S`). */
  readonly newTokenIds: Int32Array;
  /** Display labels for newly processed tokens in this step (length `S`). */
  readonly newTokenLabels: readonly string[];
  /** Cumulative prefix + new token IDs after this step (length `P + S`). */
  readonly sequenceTokenIds: Int32Array;
  /** Cumulative prefix + new display labels after this step (length `P + S`). */
  readonly sequenceTokenLabels: readonly string[];
  /** Untied Embedding output for newly processed tokens `[d, S]`. */
  readonly X_embed: Float32Array;
  /** All `L + 1` (= 53) step residual checkpoints for the `S` newly processed tokens. */
  readonly residualStreams: readonly Float32Array[];
  /** Per-bay residual telemetry for the `S` newly processed tokens across all `L` (= 52) layers. */
  readonly baySummaries: readonly HybridBaySummary[];
  /** Final RMSNorm (ε = 1e-5) for the `S` newly processed tokens. */
  readonly finalNorm: HybridFinalNormResult;
  /** Untied LM Head output for the `S` newly processed tokens. */
  readonly lmHead: HybridLmHeadResult;
  /** Top-5 synthetic next-token candidates for the final position `P + S - 1` (not yet inserted!). */
  readonly predictedNextTokens: readonly HybridPrediction[];
  /** Greedy argmax synthetic next-token prediction (`predictedNextTokens[0]`). */
  readonly greedyNextToken: HybridPrediction;
  /** Exact typed-array byte accounting of `nextState`. */
  readonly cacheAccounting: DecodeCacheMemoryAccounting;
  /**
   * Deterministic access to ANY bay's true incremental simulator trace (`1..L`).
   * Eagerly retained bays return `traceOrigin: 'computed_this_step'`; lazily reconstructed bays
   * rebuild from immutable `(prevState, residualStreams[bayId - 1])` and report
   * `traceOrigin: 'reconstructed_from_checkpoint'`.
   */
  getLayerTrace(bayId: number): HybridIncrementalLayerTrace;
  getMambaTrace(bayId: number): MambaSimResult;
  getMoeTrace(bayId: number): MoeSimResult;
  getGqaTrace(bayId: number): GqaSimResult;
  /** Report the currently retained origin of `bayId` (or `null` if not currently retained). */
  getLayerTraceOrigin(bayId: number): IncrementalTraceOrigin | null;
  retainedTraceBayIds(): readonly number[];
  /** Exact deduplicated typed-array bytes held by this step result (cache + checkpoints + traces). */
  retainedTypedArrayBytes(): number;
}

// ============================================================================
// DETERMINISTIC 64-BIT HASHING (State Digests & Unambiguous Step Identity)
// ============================================================================

function makeDualHasher() {
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
  const feedI32 = (v: number) => feedU32(v | 0);
  const feedF32 = (v: number) => {
    f32View[0] = Object.is(v, -0) ? 0 : v;
    feedU32(u32View[0]);
  };
  const feedStr = (s: string) => {
    feedU32(s.length);
    for (let i = 0; i < s.length; i++) {
      const c = s.charCodeAt(i);
      feedByte(c);
      feedByte(c >>> 8);
    }
  };
  const digest = () => `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;

  return { feedU32, feedI32, feedF32, feedStr, digest };
}

/**
 * Compute a deterministic 64-bit hex digest over `parameterFingerprint`, `processedTokens`,
 * `maxTokens`, `prefixTokenIds`, length-prefixed `prefixTokenLabels`, and every populated
 * Float32 element of every Mamba and GQA cache bay.
 * Unpopulated trailing KV slots (`k >= seqLen`) are intentionally excluded.
 */
export function computeDecodeStateDigest(params: {
  readonly parameterFingerprint: string;
  readonly processedTokens: number;
  readonly maxTokens: number;
  readonly prefixTokenIds: Int32Array;
  readonly prefixTokenLabels: readonly string[];
  readonly gqaCaches: ReadonlyMap<number, GqaLayerKvCache>;
  readonly mambaStates: ReadonlyMap<number, MambaLayerBoundaryState>;
}): string {
  const h = makeDualHasher();
  h.feedStr(params.parameterFingerprint);
  h.feedU32(params.processedTokens);
  h.feedU32(params.maxTokens);
  h.feedU32(params.prefixTokenIds.length);
  for (let i = 0; i < params.prefixTokenIds.length; i++) {
    h.feedI32(params.prefixTokenIds[i]);
  }
  h.feedU32(params.prefixTokenLabels.length);
  for (let i = 0; i < params.prefixTokenLabels.length; i++) {
    h.feedStr(params.prefixTokenLabels[i]);
  }

  const mambaIds = Array.from(params.mambaStates.keys()).sort((a, b) => a - b);
  h.feedU32(mambaIds.length);
  for (const bayId of mambaIds) {
    const m = params.mambaStates.get(bayId)!;
    h.feedU32(bayId);
    h.feedU32(m.processedTokens);
    for (let i = 0; i < m.convState.length; i++) h.feedF32(m.convState[i]);
    for (let i = 0; i < m.ssmState.length; i++) h.feedF32(m.ssmState[i]);
  }

  const gqaIds = Array.from(params.gqaCaches.keys()).sort((a, b) => a - b);
  h.feedU32(gqaIds.length);
  for (const bayId of gqaIds) {
    const g = params.gqaCaches.get(bayId)!;
    h.feedU32(bayId);
    h.feedU32(g.seqLen);
    const kvRows = g.nKv * g.dHead;
    for (let row = 0; row < kvRows; row++) {
      const off = row * g.maxTokens;
      for (let k = 0; k < g.seqLen; k++) {
        h.feedF32(g.K[off + k]);
        h.feedF32(g.V[off + k]);
      }
    }
  }

  return h.digest();
}

function computeIncrementalStepExecutionId(params: {
  readonly modelId: string;
  readonly parameterFingerprint: string;
  readonly branchId: string;
  readonly parentStepExecutionId: string | null;
  readonly initialStateDigest: string;
  readonly prevStateDigest: string;
  readonly stepIndex: number;
  readonly tokenRange: readonly [number, number];
  readonly newTokenIds: Int32Array;
  readonly newTokenLabels: readonly string[];
}): string {
  const h = makeDualHasher();
  h.feedStr(params.modelId);
  h.feedStr(params.parameterFingerprint);
  h.feedStr(params.branchId);
  h.feedStr(params.parentStepExecutionId ?? '<root>');
  h.feedStr(params.initialStateDigest);
  h.feedStr(params.prevStateDigest);
  h.feedU32(params.stepIndex);
  h.feedU32(params.tokenRange[0]);
  h.feedU32(params.tokenRange[1]);
  h.feedU32(params.newTokenIds.length);
  for (let i = 0; i < params.newTokenIds.length; i++) {
    h.feedI32(params.newTokenIds[i]);
    h.feedStr(params.newTokenLabels[i] ?? '');
  }
  const idPart = Array.from(params.newTokenIds).join(',');
  return `step:${params.branchId}:#${params.stepIndex}:pos[${params.tokenRange[0]}..${params.tokenRange[1]}]:t[${idPart}]:${h.digest()}`;
}

// ============================================================================
// EXACT CACHE BYTE ACCOUNTING & STRICT VALIDATION
// ============================================================================

/**
 * Exact typed-array byte accounting for a `HybridDecodeCacheState` snapshot
 * (excludes JS object/Map overhead and browser educational traces).
 *
 * For default Nemotron 3 Nano miniature (`23 M`, `6 *`, `23 E`, `maxTokens = 16`):
 *   - `mambaFixedBytes`         = `70,656 B` (`69.00 KiB`)
 *   - `gqaAllocatedBytes`       =  `6,144 B` ( `6.00 KiB`)
 *   - `moeBytes`                =      `0 B`
 *   - `totalAllocatedBytes`     = `76,800 B` (`75.00 KiB`)
 *   - `bytesPerTokenGqaPopulated` =  `384 B` ( `0.375 KiB / tok`)
 */
export function measureDecodeCacheBytes(state: HybridDecodeCacheState): DecodeCacheMemoryAccounting {
  let mambaFixedBytes = 0;
  for (const m of state.mambaStates.values()) {
    mambaFixedBytes += m.convState.byteLength + m.ssmState.byteLength;
  }

  let gqaAllocatedBytes = 0;
  let gqaPopulatedBytes = 0;
  let bytesPerTokenGqaPopulated = 0;
  for (const g of state.gqaCaches.values()) {
    gqaAllocatedBytes += g.K.byteLength + g.V.byteLength;
    const perTok = 2 * g.nKv * g.dHead * Float32Array.BYTES_PER_ELEMENT;
    bytesPerTokenGqaPopulated += perTok;
    gqaPopulatedBytes += perTok * g.seqLen;
  }

  return {
    mambaLayerCount: state.mambaStates.size,
    gqaLayerCount: state.gqaCaches.size,
    moeLayerCount: state.moeLayerIds.length,
    processedTokens: state.processedTokens,
    maxTokens: state.maxTokens,
    mambaFixedBytes,
    gqaAllocatedBytes,
    gqaPopulatedBytes,
    bytesPerTokenGqaPopulated,
    moeBytes: 0,
    totalAllocatedBytes: mambaFixedBytes + gqaAllocatedBytes,
    totalPopulatedBytes: mambaFixedBytes + gqaPopulatedBytes,
  };
}

function assertAllFinite(arr: Float32Array, label: string): void {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) {
      throw new RangeError(`Non-finite value (${arr[i]}) in ${label}[${i}]`);
    }
  }
}

/**
 * Validate a `HybridDecodeCacheState` against `model`:
 *  - Matching `parameterFingerprint` (guards against cross-model contamination even with duplicated `modelId`)
 *  - Complete coverage of every stateful bay (`M`, `*`) and stateless bay (`E`) in `model.config.pattern`
 *  - Exact tensor dimensions, prefix length consistency, and capacity bounds
 *  - Strictly finite values across all Mamba state buffers and all populated GQA KV slots (`0..seqLen-1`)
 *  - Strict private typed-array ownership (no shared backing `ArrayBuffer` aliasing between bays/tensors)
 *  - Verified `stateDigest` integrity matching `computeDecodeStateDigest(state)`
 */
export function validateDecodeCacheState(
  model: HybridModelParameters,
  state: HybridDecodeCacheState,
): void {
  if (state.parameterFingerprint !== model.parameterFingerprint) {
    throw new TypeError(
      `Decode cache parameterFingerprint mismatch: state=${state.parameterFingerprint} !== model=${model.parameterFingerprint}`,
    );
  }
  if (!Number.isInteger(state.processedTokens) || state.processedTokens < 0) {
    throw new RangeError(`Invalid decode cache processedTokens: ${state.processedTokens}`);
  }
  if (!Number.isInteger(state.maxTokens) || state.maxTokens < 1) {
    throw new RangeError(`Invalid decode cache maxTokens: ${state.maxTokens}`);
  }
  if (state.processedTokens > state.maxTokens) {
    throw new RangeError(
      `Decode cache processedTokens (${state.processedTokens}) exceeds maxTokens (${state.maxTokens})`,
    );
  }
  if (state.prefixTokenIds.length !== state.processedTokens) {
    throw new RangeError(
      `Decode cache prefixTokenIds length (${state.prefixTokenIds.length}) !== processedTokens (${state.processedTokens})`,
    );
  }
  if (state.prefixTokenLabels.length !== state.processedTokens) {
    throw new RangeError(
      `Decode cache prefixTokenLabels length (${state.prefixTokenLabels.length}) !== processedTokens (${state.processedTokens})`,
    );
  }

  const pattern = model.config.pattern;
  const expectedMambaIds: number[] = [];
  const expectedGqaIds: number[] = [];
  const expectedMoeIds: number[] = [];
  for (let i = 0; i < pattern.length; i++) {
    const bayId = i + 1;
    const ch = pattern[i] as Kind;
    if (ch === 'M') expectedMambaIds.push(bayId);
    else if (ch === '*') expectedGqaIds.push(bayId);
    else expectedMoeIds.push(bayId);
  }

  if (state.mambaStates.size !== expectedMambaIds.length) {
    throw new RangeError(
      `Decode cache mambaStates count (${state.mambaStates.size}) !== expected (${expectedMambaIds.length})`,
    );
  }
  if (state.gqaCaches.size !== expectedGqaIds.length) {
    throw new RangeError(
      `Decode cache gqaCaches count (${state.gqaCaches.size}) !== expected (${expectedGqaIds.length})`,
    );
  }
  if (state.moeLayerIds.length !== expectedMoeIds.length) {
    throw new RangeError(
      `Decode cache moeLayerIds count (${state.moeLayerIds.length}) !== expected (${expectedMoeIds.length})`,
    );
  }

  const moeSet = new Set(state.moeLayerIds);
  if (moeSet.size !== expectedMoeIds.length) {
    throw new RangeError('Decode cache moeLayerIds contains duplicates');
  }
  for (const bayId of expectedMoeIds) {
    if (!moeSet.has(bayId)) {
      throw new RangeError(`Decode cache missing stateless MoE bay ${bayId}`);
    }
  }

  const seenBuffers = new Set<ArrayBufferLike>();
  const registerUniqueBuffer = (arr: ArrayBufferView, label: string) => {
    if (seenBuffers.has(arr.buffer)) {
      throw new TypeError(`Decode cache buffer aliasing detected at ${label}`);
    }
    seenBuffers.add(arr.buffer);
  };
  registerUniqueBuffer(state.prefixTokenIds, 'prefixTokenIds');

  const expectedConvLen = MAMBA_SIM_CONFIG.convDim * MAMBA_SIM_CONFIG.convKernel;
  const expectedSsmLen = MAMBA_SIM_CONFIG.nHeads * MAMBA_SIM_CONFIG.dHead * MAMBA_SIM_CONFIG.dState;

  for (const bayId of expectedMambaIds) {
    const m = state.mambaStates.get(bayId);
    if (!m) {
      throw new RangeError(`Decode cache missing Mamba-2 state for bay ${bayId}`);
    }
    if (m.bayId !== bayId || m.kind !== 'M') {
      throw new TypeError(`Mamba state metadata mismatch at bay ${bayId}`);
    }
    if (
      m.convDim !== MAMBA_SIM_CONFIG.convDim ||
      m.convKernel !== MAMBA_SIM_CONFIG.convKernel ||
      m.nHeads !== MAMBA_SIM_CONFIG.nHeads ||
      m.dHead !== MAMBA_SIM_CONFIG.dHead ||
      m.dState !== MAMBA_SIM_CONFIG.dState
    ) {
      throw new RangeError(`Mamba state dimension mismatch at bay ${bayId}`);
    }
    if (m.convState.length !== expectedConvLen) {
      throw new RangeError(
        `Mamba bay ${bayId} convState length (${m.convState.length}) !== expected (${expectedConvLen})`,
      );
    }
    if (m.ssmState.length !== expectedSsmLen) {
      throw new RangeError(
        `Mamba bay ${bayId} ssmState length (${m.ssmState.length}) !== expected (${expectedSsmLen})`,
      );
    }
    if (m.processedTokens !== state.processedTokens) {
      throw new RangeError(
        `Mamba bay ${bayId} processedTokens (${m.processedTokens}) !== snapshot (${state.processedTokens})`,
      );
    }
    registerUniqueBuffer(m.convState, `mamba[${bayId}].convState`);
    registerUniqueBuffer(m.ssmState, `mamba[${bayId}].ssmState`);
    assertAllFinite(m.convState, `mamba[${bayId}].convState`);
    assertAllFinite(m.ssmState, `mamba[${bayId}].ssmState`);
  }

  const kvRows = GQA_SIM_CONFIG.nKv * GQA_SIM_CONFIG.dHead;
  const expectedKvLen = kvRows * state.maxTokens;
  for (const bayId of expectedGqaIds) {
    const g = state.gqaCaches.get(bayId);
    if (!g) {
      throw new RangeError(`Decode cache missing GQA KV cache for bay ${bayId}`);
    }
    if (g.bayId !== bayId || g.kind !== '*') {
      throw new TypeError(`GQA KV cache metadata mismatch at bay ${bayId}`);
    }
    if (g.nKv !== GQA_SIM_CONFIG.nKv || g.dHead !== GQA_SIM_CONFIG.dHead) {
      throw new RangeError(`GQA KV cache head dimension mismatch at bay ${bayId}`);
    }
    if (g.seqLen !== state.processedTokens) {
      throw new RangeError(
        `GQA bay ${bayId} seqLen (${g.seqLen}) !== snapshot processedTokens (${state.processedTokens})`,
      );
    }
    if (g.maxTokens !== state.maxTokens) {
      throw new RangeError(
        `GQA bay ${bayId} maxTokens (${g.maxTokens}) !== snapshot maxTokens (${state.maxTokens})`,
      );
    }
    if (g.K.length !== expectedKvLen || g.V.length !== expectedKvLen) {
      throw new RangeError(
        `GQA bay ${bayId} K/V length (${g.K.length}/${g.V.length}) !== expected (${expectedKvLen})`,
      );
    }
    registerUniqueBuffer(g.K, `gqa[${bayId}].K`);
    registerUniqueBuffer(g.V, `gqa[${bayId}].V`);
    for (let row = 0; row < kvRows; row++) {
      const off = row * g.maxTokens;
      for (let k = 0; k < g.seqLen; k++) {
        if (!Number.isFinite(g.K[off + k])) {
          throw new RangeError(`Non-finite value (${g.K[off + k]}) in gqa[${bayId}].K[${row},${k}]`);
        }
        if (!Number.isFinite(g.V[off + k])) {
          throw new RangeError(`Non-finite value (${g.V[off + k]}) in gqa[${bayId}].V[${row},${k}]`);
        }
      }
    }
  }

  const expectedDigest = computeDecodeStateDigest(state);
  if (state.stateDigest !== expectedDigest) {
    throw new Error(
      `Decode cache stateDigest mismatch: recorded=${state.stateDigest} !== computed=${expectedDigest}`,
    );
  }
}

// ============================================================================
// IMMUTABLE CACHE SNAPSHOT CONSTRUCTORS (Zero, External Seed, Deep Clone)
// ============================================================================

/** Create a freshly allocated, zero-initialized `HybridDecodeCacheState` for `model`. */
export function createZeroDecodeCacheState(
  model: HybridModelParameters,
  options?: {
    readonly maxTokens?: number;
    readonly precisionMetadata?: PrecisionMode;
  },
): HybridDecodeCacheState {
  const maxTokens = Math.max(1, Math.trunc(options?.maxTokens ?? DEFAULT_DECODE_MAX_TOKENS));
  const precisionMetadata = options?.precisionMetadata ?? DEFAULT_DECODE_PRECISION_METADATA;
  const kvDtype: 'bf16' | 'fp8' = precisionMetadata === 'fp8_selective' ? 'fp8' : 'bf16';

  const gqaCaches = new Map<number, GqaLayerKvCache>();
  const mambaStates = new Map<number, MambaLayerBoundaryState>();
  const moeLayerIds: number[] = [];

  const convLen = MAMBA_SIM_CONFIG.convDim * MAMBA_SIM_CONFIG.convKernel;
  const ssmLen = MAMBA_SIM_CONFIG.nHeads * MAMBA_SIM_CONFIG.dHead * MAMBA_SIM_CONFIG.dState;
  const kvLen = GQA_SIM_CONFIG.nKv * GQA_SIM_CONFIG.dHead * maxTokens;

  const pattern = model.config.pattern;
  for (let i = 0; i < pattern.length; i++) {
    const bayId = i + 1;
    const ch = pattern[i] as Kind;
    if (ch === 'M') {
      mambaStates.set(
        bayId,
        Object.freeze({
          bayId,
          kind: 'M' as const,
          convDim: MAMBA_SIM_CONFIG.convDim,
          convKernel: MAMBA_SIM_CONFIG.convKernel,
          convState: new Float32Array(convLen),
          nHeads: MAMBA_SIM_CONFIG.nHeads,
          dHead: MAMBA_SIM_CONFIG.dHead,
          dState: MAMBA_SIM_CONFIG.dState,
          ssmState: new Float32Array(ssmLen),
          processedTokens: 0,
          storageDtypeMetadata: Object.freeze({ conv1d: 'bf16' as const, ssm: 'fp32' as const }),
        }),
      );
    } else if (ch === '*') {
      gqaCaches.set(
        bayId,
        Object.freeze({
          bayId,
          kind: '*' as const,
          nKv: GQA_SIM_CONFIG.nKv,
          dHead: GQA_SIM_CONFIG.dHead,
          seqLen: 0,
          maxTokens,
          K: new Float32Array(kvLen),
          V: new Float32Array(kvLen),
          storageDtypeMetadata: kvDtype,
        }),
      );
    } else {
      moeLayerIds.push(bayId);
    }
  }

  const prefixTokenIds = new Int32Array(0);
  const prefixTokenLabels = Object.freeze([] as string[]);
  const stateDigest = computeDecodeStateDigest({
    parameterFingerprint: model.parameterFingerprint,
    processedTokens: 0,
    maxTokens,
    prefixTokenIds,
    prefixTokenLabels,
    gqaCaches,
    mambaStates,
  });

  return Object.freeze({
    modelId: model.modelId,
    parameterFingerprint: model.parameterFingerprint,
    stateDigest,
    precisionMetadata,
    processedTokens: 0,
    maxTokens,
    prefixTokenIds,
    prefixTokenLabels,
    gqaCaches,
    mambaStates,
    moeLayerIds: Object.freeze(moeLayerIds),
  });
}

/**
 * Initialize a validated, defensively cloned `HybridDecodeCacheState` from arbitrary externally
 * supplied boundary states (e.g. a saved prefix checkpoint or synthetic non-zero state).
 * Subsequent mutations to the caller's input typed arrays CANNOT affect the returned snapshot.
 */
export function createExternalDecodeCacheState(
  model: HybridModelParameters,
  seed: ExternalDecodeCacheSeed,
): HybridDecodeCacheState {
  const P = seed.processedTokens;
  if (!Number.isInteger(P) || P < 0) {
    throw new RangeError(`External decode seed processedTokens must be integer >= 0, got ${P}`);
  }
  let maxTokens: number;
  if (seed.maxTokens !== undefined) {
    if (!Number.isInteger(seed.maxTokens) || seed.maxTokens < 1) {
      throw new RangeError(
        `External decode seed maxTokens must be integer >= 1, got ${seed.maxTokens}`,
      );
    }
    if (seed.maxTokens < P) {
      throw new RangeError(
        `External decode seed maxTokens (${seed.maxTokens}) < processedTokens (${P})`,
      );
    }
    maxTokens = seed.maxTokens;
  } else {
    maxTokens = Math.max(DEFAULT_DECODE_MAX_TOKENS, P);
  }
  const precisionMetadata = seed.precisionMetadata ?? DEFAULT_DECODE_PRECISION_METADATA;
  const kvDtype: 'bf16' | 'fp8' = precisionMetadata === 'fp8_selective' ? 'fp8' : 'bf16';

  const prefixTokenIds = new Int32Array(P);
  const prefixTokenLabels: string[] = [];
  for (let k = 0; k < P; k++) {
    const pid = seed.prefixTokenIds?.[k] ?? -1;
    prefixTokenIds[k] = pid;
    prefixTokenLabels.push(
      seed.prefixTokenLabels?.[k] ??
        (pid >= 0 && pid < model.vocab.length ? model.vocab[pid] : `seed#${k}`),
    );
  }

  const convLen = MAMBA_SIM_CONFIG.convDim * MAMBA_SIM_CONFIG.convKernel;
  const ssmLen = MAMBA_SIM_CONFIG.nHeads * MAMBA_SIM_CONFIG.dHead * MAMBA_SIM_CONFIG.dState;
  const kvRows = GQA_SIM_CONFIG.nKv * GQA_SIM_CONFIG.dHead;

  const gqaCaches = new Map<number, GqaLayerKvCache>();
  const mambaStates = new Map<number, MambaLayerBoundaryState>();
  const moeLayerIds: number[] = [];

  const pattern = model.config.pattern;
  for (let i = 0; i < pattern.length; i++) {
    const bayId = i + 1;
    const ch = pattern[i] as Kind;
    if (ch === 'M') {
      const src = seed.mambaStates.get(bayId);
      if (!src) {
        throw new RangeError(`External decode seed missing Mamba-2 state for bay ${bayId}`);
      }
      if (src.processedTokens !== undefined && src.processedTokens !== P) {
        throw new RangeError(
          `External Mamba bay ${bayId} processedTokens (${src.processedTokens}) !== seed (${P})`,
        );
      }
      if (src.convState.length !== convLen) {
        throw new RangeError(
          `External Mamba bay ${bayId} convState length (${src.convState.length}) !== ${convLen}`,
        );
      }
      if (src.ssmState.length !== ssmLen) {
        throw new RangeError(
          `External Mamba bay ${bayId} ssmState length (${src.ssmState.length}) !== ${ssmLen}`,
        );
      }
      mambaStates.set(
        bayId,
        Object.freeze({
          bayId,
          kind: 'M' as const,
          convDim: MAMBA_SIM_CONFIG.convDim,
          convKernel: MAMBA_SIM_CONFIG.convKernel,
          convState: new Float32Array(src.convState), // defensive clone
          nHeads: MAMBA_SIM_CONFIG.nHeads,
          dHead: MAMBA_SIM_CONFIG.dHead,
          dState: MAMBA_SIM_CONFIG.dState,
          ssmState: new Float32Array(src.ssmState), // defensive clone
          processedTokens: P,
          storageDtypeMetadata: Object.freeze({ conv1d: 'bf16' as const, ssm: 'fp32' as const }),
        }),
      );
    } else if (ch === '*') {
      const src = seed.gqaCaches.get(bayId);
      if (!src) {
        throw new RangeError(`External decode seed missing GQA KV cache for bay ${bayId}`);
      }
      const srcSeqLen = src.seqLen ?? P;
      if (srcSeqLen !== P) {
        throw new RangeError(`External GQA bay ${bayId} seqLen (${srcSeqLen}) !== seed (${P})`);
      }
      let srcStride: number;
      if (src.maxTokens !== undefined) {
        if (!Number.isInteger(src.maxTokens) || src.maxTokens < P) {
          throw new RangeError(
            `External GQA bay ${bayId} maxTokens (${src.maxTokens}) invalid for seqLen (${P})`,
          );
        }
        srcStride = src.maxTokens;
      } else {
        if (src.V.length !== src.K.length) {
          throw new RangeError(
            `External GQA bay ${bayId} K/V length mismatch (${src.K.length} !== ${src.V.length}) when inferring stride without explicit maxTokens`,
          );
        }
        srcStride = src.K.length % kvRows === 0 ? src.K.length / kvRows : -1;
        if (!Number.isInteger(srcStride) || srcStride < P) {
          throw new RangeError(
            `External GQA bay ${bayId} K/V stride (${srcStride}) invalid for seqLen (${P})`,
          );
        }
      }
      if (src.K.length < kvRows * srcStride || src.V.length < kvRows * srcStride) {
        throw new RangeError(
          `External GQA bay ${bayId} K/V lengths (${src.K.length}/${src.V.length}) < ${kvRows * srcStride}`,
        );
      }
      const K = new Float32Array(kvRows * maxTokens);
      const V = new Float32Array(kvRows * maxTokens);
      for (let row = 0; row < kvRows; row++) {
        const srcOff = row * srcStride;
        const dstOff = row * maxTokens;
        for (let k = 0; k < P; k++) {
          K[dstOff + k] = src.K[srcOff + k];
          V[dstOff + k] = src.V[srcOff + k];
        }
      }
      gqaCaches.set(
        bayId,
        Object.freeze({
          bayId,
          kind: '*' as const,
          nKv: GQA_SIM_CONFIG.nKv,
          dHead: GQA_SIM_CONFIG.dHead,
          seqLen: P,
          maxTokens,
          K,
          V,
          storageDtypeMetadata: kvDtype,
        }),
      );
    } else {
      moeLayerIds.push(bayId);
    }
  }

  // Reject extraneous bays not in the model pattern
  if (seed.mambaStates.size !== mambaStates.size) {
    throw new RangeError(
      `External decode seed contains unexpected extra Mamba bays (${seed.mambaStates.size} vs ${mambaStates.size})`,
    );
  }
  if (seed.gqaCaches.size !== gqaCaches.size) {
    throw new RangeError(
      `External decode seed contains unexpected extra GQA bays (${seed.gqaCaches.size} vs ${gqaCaches.size})`,
    );
  }

  const frozenLabels = Object.freeze(prefixTokenLabels);
  const stateDigest = computeDecodeStateDigest({
    parameterFingerprint: model.parameterFingerprint,
    processedTokens: P,
    maxTokens,
    prefixTokenIds,
    prefixTokenLabels: frozenLabels,
    gqaCaches,
    mambaStates,
  });

  const snapshot: HybridDecodeCacheState = Object.freeze({
    modelId: model.modelId,
    parameterFingerprint: model.parameterFingerprint,
    stateDigest,
    precisionMetadata,
    processedTokens: P,
    maxTokens,
    prefixTokenIds,
    prefixTokenLabels: frozenLabels,
    gqaCaches,
    mambaStates,
    moeLayerIds: Object.freeze(moeLayerIds),
  });

  validateDecodeCacheState(model, snapshot);
  return snapshot;
}

// ============================================================================
// STEP EXECUTION (SHARED KERNELS, COPY-ON-STEP SNAPSHOT, LAZY TRACE REBUILD)
// ============================================================================

function computeStreamRms(stream: Float32Array, d: number, T: number): Float32Array {
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let sumSq = 0;
    for (let i = 0; i < d; i++) {
      const v = stream[i * T + t];
      sumSq += v * v;
    }
    out[t] = Math.fround(Math.sqrt(sumSq / d));
  }
  return out;
}

function computeDeltaRms(
  prev: Float32Array,
  next: Float32Array,
  d: number,
  T: number,
): Float32Array {
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let sumSq = 0;
    for (let i = 0; i < d; i++) {
      const diff = next[i * T + t] - prev[i * T + t];
      sumSq += diff * diff;
    }
    out[t] = Math.fround(Math.sqrt(sumSq / d));
  }
  return out;
}

function addUniqueBytes(seen: Set<ArrayBufferLike>, arr: ArrayBufferView | undefined): number {
  if (!arr || seen.has(arr.buffer)) return 0;
  seen.add(arr.buffer);
  return arr.buffer.byteLength;
}

function collectCacheStateUniqueBytes(
  seen: Set<ArrayBufferLike>,
  state: HybridDecodeCacheState,
): number {
  let bytes = addUniqueBytes(seen, state.prefixTokenIds);
  for (const m of state.mambaStates.values()) {
    bytes += addUniqueBytes(seen, m.convState) + addUniqueBytes(seen, m.ssmState);
  }
  for (const g of state.gqaCaches.values()) {
    bytes += addUniqueBytes(seen, g.K) + addUniqueBytes(seen, g.V);
  }
  return bytes;
}

/**
 * Module-private immutable boundary snapshots keyed by the validated `HybridDecodeCacheState`.
 * Shared across all sibling steps/branches forking from the same `prevState` and automatically
 * garbage-collected once the parent snapshot is no longer referenced.
 */
const privateBoundarySnapshots = new WeakMap<HybridDecodeCacheState, HybridDecodeCacheState>();

function getOrCreatePrivateBoundarySnapshot(
  state: HybridDecodeCacheState,
): HybridDecodeCacheState {
  const existing = privateBoundarySnapshots.get(state);
  if (existing) return existing;

  if (state.processedTokens === 0) {
    const emptySnap: HybridDecodeCacheState = Object.freeze({
      modelId: state.modelId,
      parameterFingerprint: state.parameterFingerprint,
      stateDigest: state.stateDigest,
      precisionMetadata: state.precisionMetadata,
      processedTokens: 0,
      maxTokens: state.maxTokens,
      prefixTokenIds: new Int32Array(0),
      prefixTokenLabels: Object.freeze([] as string[]),
      gqaCaches: new Map<number, GqaLayerKvCache>(),
      mambaStates: new Map<number, MambaLayerBoundaryState>(),
      moeLayerIds: Object.freeze([...state.moeLayerIds]),
    });
    privateBoundarySnapshots.set(state, emptySnap);
    return emptySnap;
  }

  const mambaStates = new Map<number, MambaLayerBoundaryState>();
  for (const [bayId, m] of state.mambaStates.entries()) {
    mambaStates.set(
      bayId,
      Object.freeze({
        ...m,
        convState: new Float32Array(m.convState),
        ssmState: new Float32Array(m.ssmState),
        storageDtypeMetadata: Object.freeze({ ...m.storageDtypeMetadata }),
      }),
    );
  }

  const gqaCaches = new Map<number, GqaLayerKvCache>();
  for (const [bayId, g] of state.gqaCaches.entries()) {
    gqaCaches.set(
      bayId,
      Object.freeze({
        ...g,
        K: new Float32Array(g.K),
        V: new Float32Array(g.V),
      }),
    );
  }

  const cloned: HybridDecodeCacheState = Object.freeze({
    modelId: state.modelId,
    parameterFingerprint: state.parameterFingerprint,
    stateDigest: state.stateDigest,
    precisionMetadata: state.precisionMetadata,
    processedTokens: state.processedTokens,
    maxTokens: state.maxTokens,
    prefixTokenIds: new Int32Array(state.prefixTokenIds),
    prefixTokenLabels: Object.freeze([...state.prefixTokenLabels]),
    gqaCaches,
    mambaStates,
    moeLayerIds: Object.freeze([...state.moeLayerIds]),
  });
  privateBoundarySnapshots.set(state, cloned);
  return cloned;
}

const stepByteCollectors = new WeakMap<
  HybridIncrementalStepResult,
  (seen: Set<ArrayBufferLike>) => number
>();

const stepTracePruners = new WeakMap<
  HybridIncrementalStepResult,
  () => HybridIncrementalStepResult
>();

function collectIncrementalTraceBytes(
  seen: Set<ArrayBufferLike>,
  trace: HybridIncrementalLayerTrace,
): number {
  if (trace.kind === 'M') {
    const s = trace.sim;
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
  if (trace.kind === 'E') {
    const s = trace.sim;
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
  const s = trace.sim;
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

function runSingleIncrementalBay(
  model: HybridModelParameters,
  stepExecutionId: string,
  provenance: IncrementalStepProvenance,
  traceOrigin: IncrementalTraceOrigin,
  bayId: number,
  X_in: Float32Array,
  S: number,
  newTokenIds: Int32Array,
  newTokenLabels: readonly string[],
  prevState: HybridDecodeCacheState,
): HybridIncrementalLayerTrace {
  const weights = model.getBayWeights(bayId);
  const eps = model.config.rmsNormEps;
  const P = prevState.processedTokens;

  if (weights.kind === 'M') {
    const prevM = P > 0 ? prevState.mambaStates.get(bayId) : undefined;
    const fixture: MambaSimFixture = {
      X: X_in,
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
      tokenIds: newTokenIds,
      tokenLabels: newTokenLabels,
      ...(P > 0 && prevM
        ? {
            initialConvState: prevM.convState,
            initialSsmState: prevM.ssmState,
          }
        : {}),
      startTokenOffset: P,
    };
    return {
      stepExecutionId,
      provenance,
      traceOrigin,
      bayId,
      kind: 'M',
      sim: runMambaSim(S, fixture),
    };
  }

  if (weights.kind === 'E') {
    const fixture: MoeSimFixture = {
      X: X_in,
      gamma_in: weights.gamma_in,
      W_router: weights.W_router,
      e_score_correction_bias: weights.e_score_correction_bias,
      W_up: weights.W_up,
      W_down: weights.W_down,
      W_up_s: weights.W_up_s,
      W_down_s: weights.W_down_s,
      rmsNormEps: eps,
      routerPrecisionPolicy: model.config.routerPrecisionPolicy,
      tokenIds: newTokenIds,
      tokenLabels: newTokenLabels,
      startTokenOffset: P,
    };
    return {
      stepExecutionId,
      provenance,
      traceOrigin,
      bayId,
      kind: 'E',
      sim: runMoeSim(S, fixture),
    };
  }

  const prevG = P > 0 ? prevState.gqaCaches.get(bayId) : undefined;
  const fixture: GqaSimFixture = {
    X: X_in,
    gamma: weights.gamma,
    W_Q: weights.W_Q,
    W_K: weights.W_K,
    W_V: weights.W_V,
    W_O: weights.W_O,
    rmsEps: eps,
    tokenIds: newTokenIds,
    tokenLabels: newTokenLabels,
    ...(P > 0 && prevG
      ? {
          kvCacheIn: {
            seqLen: prevG.seqLen,
            maxTokens: prevG.maxTokens,
            K: prevG.K,
            V: prevG.V,
            prefixTokenIds: prevState.prefixTokenIds,
            prefixTokenLabels: prevState.prefixTokenLabels,
          },
        }
      : {}),
  };
  return {
    stepExecutionId,
    provenance,
    traceOrigin,
    bayId,
    kind: '*',
    sim: runGqaSim(S, fixture),
  };
}

/**
 * Execute one cached incremental prefill, chunk, or single-token AR decode step over `model`.
 *
 * - Never mutates `input.prevState`.
 * - Uses shared verified `runMambaSim`, `runMoeSim`, and `runGqaSim` kernels.
 * - Matches full-sequence execution bit-for-bit (`0` ulp) across all 52 layers and LM head.
 */
export function runIncrementalDecodeStep(
  model: HybridModelParameters,
  input: IncrementalDecodeStepInput,
): HybridIncrementalStepResult {
  const { d, vocabSize, pattern, rmsNormEps } = model.config;
  const numLayers = pattern.length;

  const rawIds = input.newTokenIds;
  if (rawIds.length < 1) {
    throw new RangeError('Incremental decode step requires at least 1 new token ID');
  }
  if (rawIds.length > MAX_INCREMENTAL_STEP_TOKENS) {
    throw new RangeError(
      `Incremental decode step accepts at most ${MAX_INCREMENTAL_STEP_TOKENS} new tokens per step, received ${rawIds.length}`,
    );
  }
  const S = rawIds.length;

  const prevState =
    input.prevState !== null
      ? input.prevState
      : createZeroDecodeCacheState(model, {
          maxTokens: input.maxTokens ?? DEFAULT_DECODE_MAX_TOKENS,
          precisionMetadata: input.precisionMetadata ?? DEFAULT_DECODE_PRECISION_METADATA,
        });
  validateDecodeCacheState(model, prevState);
  // Capture an isolated private boundary snapshot so post-step caller mutations to
  // `prevState` (or its Maps/typed arrays) cannot corrupt lazy `getLayerTrace(bayId)` calls.
  const privatePrevState = getOrCreatePrivateBoundarySnapshot(prevState);

  const P = prevState.processedTokens;
  const totalTokens = P + S;
  if (totalTokens > prevState.maxTokens) {
    throw new RangeError(
      `Incremental decode step (${P} + ${S} = ${totalTokens}) exceeds cache capacity maxTokens (${prevState.maxTokens})`,
    );
  }

  const newTokenIds = new Int32Array(S);
  const newTokenLabelsArr: string[] = [];
  for (let t = 0; t < S; t++) {
    const id = Math.max(0, Math.min(vocabSize - 1, Math.trunc(rawIds[t])));
    newTokenIds[t] = id;
    newTokenLabelsArr.push(input.newTokenLabels?.[t] ?? model.vocab[id] ?? `id:${id}`);
  }
  const privateNewTokenIds = new Int32Array(newTokenIds);
  const newTokenLabels = Object.freeze(newTokenLabelsArr);

  const sequenceTokenIds = new Int32Array(totalTokens);
  sequenceTokenIds.set(privatePrevState.prefixTokenIds, 0);
  sequenceTokenIds.set(privateNewTokenIds, P);
  const sequenceTokenLabels = Object.freeze([
    ...privatePrevState.prefixTokenLabels,
    ...newTokenLabels,
  ]);

  const branchId = input.branchId ?? 'branch-0';
  const parentStepExecutionId = input.parentStepExecutionId ?? null;
  const initialStateDigest = input.initialStateDigest ?? prevState.stateDigest;
  const stepIndex = input.stepIndex ?? (input.prevState === null ? 0 : 1);
  const stepKind: IncrementalDecodeStepKind =
    P === 0 ? 'prefill' : S === 1 ? 'decode' : 'chunk';
  const tokenRange: readonly [number, number] = Object.freeze([P, totalTokens - 1]);

  // Preliminary provenance before nextStateDigest is computed; finalized right after layer loop
  const stepExecutionId = computeIncrementalStepExecutionId({
    modelId: model.modelId,
    parameterFingerprint: model.parameterFingerprint,
    branchId,
    parentStepExecutionId,
    initialStateDigest,
    prevStateDigest: prevState.stateDigest,
    stepIndex,
    tokenRange,
    newTokenIds: privateNewTokenIds,
    newTokenLabels,
  });

  // Temporary provenance shell replaced with frozen final provenance once nextStateDigest is known
  const tempProvenance: IncrementalStepProvenance = {
    initialStateDigest,
    prevStateDigest: prevState.stateDigest,
    nextStateDigest: '',
    branchId,
    parentStepExecutionId,
    stepIndex,
    stepKind,
    tokenRange,
    stepExecutionId,
  };

  // Stage 0: Untied NoPE Token Embedding ONLY for the S newly arriving tokens.
  // Detach exposed `X_embed`, exposed `residualStreams`, and closure-private `privateResidualStreams`
  // so mutating any returned buffer cannot corrupt lazy trace reconstruction.
  const X_embed = embedTokens(privateNewTokenIds, model.W_embed, d);
  const residualStreams: Float32Array[] = [X_embed.slice()];
  const privateResidualStreams: Float32Array[] = [X_embed.slice()];
  const baySummaries: HybridBaySummary[] = [];

  const eagerSet = new Set<number>(input.retainedTraceBays ?? DEFAULT_RETAINED_TRACE_BAYS);
  const maxRetainedTraces = Math.max(eagerSet.size, input.maxRetainedTraces ?? 12);
  const eagerTracesRaw: HybridIncrementalLayerTrace[] = [];

  // Copy-on-step next cache maps (fresh owned Float32Arrays — NEVER mutates prevState)
  const nextGqaCaches = new Map<number, GqaLayerKvCache>();
  const nextMambaStates = new Map<number, MambaLayerBoundaryState>();
  const kvRows = GQA_SIM_CONFIG.nKv * GQA_SIM_CONFIG.dHead;
  const kvDtype: 'bf16' | 'fp8' = prevState.precisionMetadata === 'fp8_selective' ? 'fp8' : 'bf16';

  let currentX = privateResidualStreams[0];
  let currentRms = computeStreamRms(currentX, d, S);

  for (let bayId = 1; bayId <= numLayers; bayId++) {
    const trace = runSingleIncrementalBay(
      model,
      stepExecutionId,
      tempProvenance,
      'computed_this_step',
      bayId,
      currentX,
      S,
      privateNewTokenIds,
      newTokenLabels,
      privatePrevState,
    );

    if (trace.kind === 'M') {
      nextMambaStates.set(
        bayId,
        Object.freeze({
          bayId,
          kind: 'M' as const,
          convDim: MAMBA_SIM_CONFIG.convDim,
          convKernel: MAMBA_SIM_CONFIG.convKernel,
          convState: new Float32Array(trace.sim.conv_buf), // fresh owned snapshot
          nHeads: MAMBA_SIM_CONFIG.nHeads,
          dHead: MAMBA_SIM_CONFIG.dHead,
          dState: MAMBA_SIM_CONFIG.dState,
          ssmState: new Float32Array(trace.sim.H), // fresh owned snapshot
          processedTokens: totalTokens,
          storageDtypeMetadata: Object.freeze({ conv1d: 'bf16' as const, ssm: 'fp32' as const }),
        }),
      );
    } else if (trace.kind === '*') {
      const nextK = new Float32Array(kvRows * prevState.maxTokens);
      const nextV = new Float32Array(kvRows * prevState.maxTokens);
      for (let row = 0; row < kvRows; row++) {
        const srcOff = row * totalTokens;
        const dstOff = row * prevState.maxTokens;
        for (let k = 0; k < totalTokens; k++) {
          nextK[dstOff + k] = trace.sim.K[srcOff + k];
          nextV[dstOff + k] = trace.sim.V[srcOff + k];
        }
      }
      nextGqaCaches.set(
        bayId,
        Object.freeze({
          bayId,
          kind: '*' as const,
          nKv: GQA_SIM_CONFIG.nKv,
          dHead: GQA_SIM_CONFIG.dHead,
          seqLen: totalTokens,
          maxTokens: prevState.maxTokens,
          K: nextK,
          V: nextV,
          storageDtypeMetadata: kvDtype,
        }),
      );
    }

    const nextX = trace.sim.Y;
    // Detach exposed `residualStreams[bayId]` from `eagerTrace.sim.Y` via `.slice()`,
    // and keep `privateResidualStreams[bayId]` isolated from both exposed surfaces.
    const isEager = eagerSet.has(bayId);
    const detachedResidual = nextX.slice();
    const privateNextX = isEager ? nextX.slice() : nextX;
    const nextRms = computeStreamRms(privateNextX, d, S);
    const deltaRms = computeDeltaRms(currentX, privateNextX, d, S);
    const isFeeder =
      bayId <= NEMOTRON_SPEC.bays.length && pattern === HYBRID_OVERRIDE_PATTERN
        ? NEMOTRON_SPEC.bays[bayId - 1].feeder
        : trace.kind === 'M' && bayId < numLayers && pattern[bayId] === '*';

    const summary: HybridBaySummary =
      trace.kind === 'E'
        ? {
            bayId,
            kind: 'E',
            feeder: false,
            rmsIn: currentRms,
            rmsDelta: deltaRms,
            rmsOut: nextRms,
            moeTopkIndices: new Int32Array(trace.sim.topk_indices),
          }
        : {
            bayId,
            kind: trace.kind,
            feeder: isFeeder,
            rmsIn: currentRms,
            rmsDelta: deltaRms,
            rmsOut: nextRms,
          };

    baySummaries.push(summary);
    residualStreams.push(detachedResidual);
    privateResidualStreams.push(privateNextX);

    if (isEager) {
      eagerTracesRaw.push(trace);
    }

    currentX = privateNextX;
    currentRms = nextRms;
  }

  const frozenResidualStreams = Object.freeze(residualStreams);
  const frozenBaySummaries = Object.freeze(baySummaries);

  // Finalize immutable nextState snapshot + digest + final provenance
  const nextStateDigest = computeDecodeStateDigest({
    parameterFingerprint: model.parameterFingerprint,
    processedTokens: totalTokens,
    maxTokens: prevState.maxTokens,
    prefixTokenIds: sequenceTokenIds,
    prefixTokenLabels: sequenceTokenLabels,
    gqaCaches: nextGqaCaches,
    mambaStates: nextMambaStates,
  });

  const nextState: HybridDecodeCacheState = Object.freeze({
    modelId: model.modelId,
    parameterFingerprint: model.parameterFingerprint,
    stateDigest: nextStateDigest,
    precisionMetadata: prevState.precisionMetadata,
    processedTokens: totalTokens,
    maxTokens: prevState.maxTokens,
    prefixTokenIds: sequenceTokenIds,
    prefixTokenLabels: sequenceTokenLabels,
    gqaCaches: nextGqaCaches,
    mambaStates: nextMambaStates,
    moeLayerIds: prevState.moeLayerIds,
  });

  const provenance: IncrementalStepProvenance = Object.freeze({
    initialStateDigest,
    prevStateDigest: prevState.stateDigest,
    nextStateDigest,
    branchId,
    parentStepExecutionId,
    stepIndex,
    stepKind,
    tokenRange,
    stepExecutionId,
  });

  const initialRetainedTraces = new Map<number, HybridIncrementalLayerTrace>();
  for (const tr of eagerTracesRaw) {
    initialRetainedTraces.set(tr.bayId, { ...tr, provenance });
  }

  // Stage L+1: Final RMSNorm (eps = 1e-5)
  const finalX = privateResidualStreams[numLayers];
  const { rms: finalRms, X_norm: finalXNorm, U: finalU } = applyRmsNorm(
    finalX,
    model.finalNormGamma,
    d,
    S,
    rmsNormEps,
  );

  // Stage L+2: Strictly Untied LM Head + Softmax
  const lmHead = runUntiedLmHead(finalU, model.W_lm_head, model.vocab, d, S, 5);
  const predictedNextTokens = Object.freeze([...lmHead.topKByToken[S - 1]]);
  const greedyNextToken = predictedNextTokens[0];
  const cacheAccounting = measureDecodeCacheBytes(nextState);
  const finalNormResult: HybridFinalNormResult = Object.freeze({
    eps: rmsNormEps,
    gamma: model.finalNormGamma,
    rms: finalRms,
    X_norm: finalXNorm,
    U_final: finalU,
  });

  const createStepView = (
    retainedTraces: Map<number, HybridIncrementalLayerTrace>,
  ): HybridIncrementalStepResult => {
    const getLayerTrace = (bayId: number): HybridIncrementalLayerTrace => {
      if (!Number.isInteger(bayId) || bayId < 1 || bayId > numLayers) {
        throw new RangeError(`Bay ID ${bayId} out of bounds [1..${numLayers}]`);
      }
      const existing = retainedTraces.get(bayId);
      if (existing) {
        retainedTraces.delete(bayId);
        retainedTraces.set(bayId, existing);
        return existing;
      }
      // Lazy deterministic reconstruction strictly from private isolated
      // `(privatePrevState, privateResidualStreams[bayId - 1], privateNewTokenIds)`.
      const rebuilt = runSingleIncrementalBay(
        model,
        stepExecutionId,
        provenance,
        'reconstructed_from_checkpoint',
        bayId,
        privateResidualStreams[bayId - 1],
        S,
        privateNewTokenIds,
        newTokenLabels,
        privatePrevState,
      );
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

    const getLayerTraceOrigin = (bayId: number): IncrementalTraceOrigin | null =>
      retainedTraces.get(bayId)?.traceOrigin ?? null;

    const retainedTraceBayIds = (): readonly number[] =>
      Array.from(retainedTraces.keys()).sort((a, b) => a - b);

    const collectOwnBytes = (seen: Set<ArrayBufferLike>): number => {
      let bytes =
        addUniqueBytes(seen, newTokenIds) +
        addUniqueBytes(seen, privateNewTokenIds) +
        addUniqueBytes(seen, sequenceTokenIds) +
        addUniqueBytes(seen, X_embed) +
        addUniqueBytes(seen, finalRms) +
        addUniqueBytes(seen, finalXNorm) +
        addUniqueBytes(seen, finalU) +
        addUniqueBytes(seen, lmHead.logits) +
        addUniqueBytes(seen, lmHead.probs);
      bytes += collectCacheStateUniqueBytes(seen, nextState);
      bytes += collectCacheStateUniqueBytes(seen, privatePrevState);
      for (const s of frozenResidualStreams) bytes += addUniqueBytes(seen, s);
      for (const s of privateResidualStreams) bytes += addUniqueBytes(seen, s);
      for (const b of frozenBaySummaries) {
        bytes +=
          addUniqueBytes(seen, b.rmsIn) +
          addUniqueBytes(seen, b.rmsDelta) +
          addUniqueBytes(seen, b.rmsOut) +
          addUniqueBytes(seen, b.moeTopkIndices);
      }
      for (const tr of retainedTraces.values()) {
        bytes += collectIncrementalTraceBytes(seen, tr);
      }
      return bytes;
    };

    const retainedTypedArrayBytes = (): number =>
      collectOwnBytes(new Set<ArrayBufferLike>());

    const view: HybridIncrementalStepResult = Object.freeze({
      modelId: model.modelId,
      parameterFingerprint: model.parameterFingerprint,
      executionId: stepExecutionId,
      stepExecutionId,
      provenance,
      prevState,
      nextState,
      logits: lmHead.logits,
      probs: lmHead.probs,
      tokens: S,
      prefixTokens: P,
      totalTokens,
      d,
      vocabSize,
      vocab: model.vocab,
      newTokenIds,
      newTokenLabels,
      sequenceTokenIds,
      sequenceTokenLabels,
      X_embed,
      residualStreams: frozenResidualStreams,
      baySummaries: frozenBaySummaries,
      finalNorm: finalNormResult,
      lmHead,
      predictedNextTokens,
      greedyNextToken,
      cacheAccounting,
      getLayerTrace,
      getMambaTrace,
      getMoeTrace,
      getGqaTrace,
      getLayerTraceOrigin,
      retainedTraceBayIds,
      retainedTypedArrayBytes,
    });

    stepByteCollectors.set(view, collectOwnBytes);
    stepTracePruners.set(view, () =>
      retainedTraces.size === 0 ? view : createStepView(new Map()),
    );
    return view;
  };

  return createStepView(initialRetainedTraces);
}

// ============================================================================
// IMMUTABLE INTERACTIVE PREFILL / DECODE SESSION WITH SAFE BOUNDED BRANCHING
// ============================================================================

export interface HybridDecodeBranch {
  readonly branchId: string;
  readonly forkedFromBranchId: string | null;
  readonly forkedFromStepExecutionId: string | null;
  readonly steps: readonly HybridIncrementalStepResult[];
}

export interface HybridDecodeSession {
  readonly model: HybridModelParameters;
  readonly maxTokens: number;
  readonly precisionMetadata: PrecisionMode;
  readonly initialPromptTokenIds: readonly number[];
  readonly activeBranchId: string;
  readonly activeStepIndex: number;
  readonly nextBranchNumber: number;
  /** Maximum number of branches retained in this session (default 16; >= 1). */
  readonly maxBranches: number;
  /** Maximum number of unique step snapshots retained across all branches (default 128; >= 1). */
  readonly maxUniqueSteps: number;
  /** When true (default), non-active historical steps drop reconstructible eager traces. */
  readonly pruneInactiveStepTraces: boolean;
  readonly branches: ReadonlyMap<string, HybridDecodeBranch>;
}

export interface CreateDecodeSessionOptions {
  readonly promptTokenIds?: readonly number[];
  readonly promptTokenLabels?: readonly string[];
  readonly initialState?: HybridDecodeCacheState | null;
  readonly maxTokens?: number;
  readonly precisionMetadata?: PrecisionMode;
  /** Conservative cap on retained continuation branches (defaults to `16`, must be integer `>= 1`). */
  readonly maxBranches?: number;
  /** Conservative cap on unique retained steps across all branches (defaults to `128`, must be integer `>= 1`). */
  readonly maxUniqueSteps?: number;
  /** Drop reconstructible eager layer traces on non-active historical steps (defaults to `true`). */
  readonly pruneInactiveStepTraces?: boolean;
}

export interface PruneDecodeSessionOptions {
  /** Override maximum retained branches (defaults to `session.maxBranches`). */
  readonly maxBranches?: number;
  /** Override maximum unique retained steps across surviving branches (defaults to `session.maxUniqueSteps`). */
  readonly maxUniqueSteps?: number;
  /** Drop reconstructible eager layer traces on non-active historical steps (defaults to `session.pruneInactiveStepTraces`). */
  readonly pruneInactiveStepTraces?: boolean;
}

function validatePositiveIntBound(value: number | undefined, fallback: number, name: string): number {
  if (value === undefined) return fallback;
  if (!Number.isInteger(value) || value < 1) {
    throw new RangeError(`${name} must be an integer >= 1, got ${value}`);
  }
  return value;
}

function countUniqueStepsAcrossBranches(
  branches: ReadonlyMap<string, HybridDecodeBranch>,
): number {
  const seen = new Set<string>();
  for (const branch of branches.values()) {
    for (const step of branch.steps) {
      seen.add(step.stepExecutionId);
    }
  }
  return seen.size;
}

/**
 * Create a new interactive prefill/AR-decode session on `branch-0` (step 0 = prefill).
 * Defaults to a 4-token educational prompt `[0, 1, 2, 3]` ("How does Nemotron use").
 */
export function createDecodeSession(
  model: HybridModelParameters,
  options?: CreateDecodeSessionOptions,
): HybridDecodeSession {
  const maxTokens = Math.max(1, Math.trunc(options?.maxTokens ?? DEFAULT_DECODE_MAX_TOKENS));
  const precisionMetadata = options?.precisionMetadata ?? DEFAULT_DECODE_PRECISION_METADATA;
  const maxBranches = validatePositiveIntBound(
    options?.maxBranches,
    DEFAULT_SESSION_MAX_BRANCHES,
    'maxBranches',
  );
  const maxUniqueSteps = validatePositiveIntBound(
    options?.maxUniqueSteps,
    DEFAULT_SESSION_MAX_UNIQUE_STEPS,
    'maxUniqueSteps',
  );
  const pruneInactiveTraces = options?.pruneInactiveStepTraces ?? true;
  const promptTokenIds = options?.promptTokenIds
    ? [...options.promptTokenIds]
    : [0, 1, 2, 3];

  const rootStep = runIncrementalDecodeStep(model, {
    newTokenIds: promptTokenIds,
    newTokenLabels: options?.promptTokenLabels,
    prevState: options?.initialState ?? null,
    maxTokens,
    precisionMetadata,
    branchId: 'branch-0',
    parentStepExecutionId: null,
    stepIndex: 0,
  });

  const branches = new Map<string, HybridDecodeBranch>();
  branches.set('branch-0', {
    branchId: 'branch-0',
    forkedFromBranchId: null,
    forkedFromStepExecutionId: null,
    steps: Object.freeze([rootStep]),
  });

  return {
    model,
    maxTokens,
    precisionMetadata,
    initialPromptTokenIds: Object.freeze(promptTokenIds),
    activeBranchId: 'branch-0',
    activeStepIndex: 0,
    nextBranchNumber: 1,
    maxBranches,
    maxUniqueSteps,
    pruneInactiveStepTraces: pruneInactiveTraces,
    branches,
  };
}

/** Return the currently inspected step of `session`. */
export function getActiveDecodeStep(session: HybridDecodeSession): HybridIncrementalStepResult {
  const branch = session.branches.get(session.activeBranchId);
  if (!branch || branch.steps.length === 0) {
    throw new RangeError(`Active decode branch "${session.activeBranchId}" is missing`);
  }
  const idx = Math.max(0, Math.min(branch.steps.length - 1, session.activeStepIndex));
  return branch.steps[idx];
}

/** Return the currently active branch of `session`. */
export function getActiveDecodeBranch(session: HybridDecodeSession): HybridDecodeBranch {
  const branch = session.branches.get(session.activeBranchId);
  if (!branch) {
    throw new RangeError(`Active decode branch "${session.activeBranchId}" is missing`);
  }
  return branch;
}

/**
 * Return a session copy whose non-active historical steps drop reconstructible eager layer
 * traces (while keeping the currently active step's eager traces and leaving any prior
 * `HybridDecodeSession` or previously held step references strictly untouched).
 * Shared prefix steps across surviving branches are deduplicated to the same pruned view.
 */
export function pruneInactiveStepTraces(session: HybridDecodeSession): HybridDecodeSession {
  const activeStepId = getActiveDecodeStep(session).stepExecutionId;
  const prunedCache = new Map<string, HybridIncrementalStepResult>();
  let anyChanged = false;
  const nextBranches = new Map<string, HybridDecodeBranch>();

  for (const [branchId, branch] of session.branches.entries()) {
    let branchChanged = false;
    const nextSteps = branch.steps.map((step) => {
      if (step.stepExecutionId === activeStepId) {
        return step;
      }
      const cached = prunedCache.get(step.stepExecutionId);
      if (cached) {
        if (cached !== step) branchChanged = true;
        return cached;
      }
      const pruner = stepTracePruners.get(step);
      const pruned = pruner ? pruner() : step;
      prunedCache.set(step.stepExecutionId, pruned);
      if (pruned !== step) branchChanged = true;
      return pruned;
    });

    if (branchChanged) {
      anyChanged = true;
      nextBranches.set(branchId, {
        ...branch,
        steps: Object.freeze(nextSteps),
      });
    } else {
      nextBranches.set(branchId, branch);
    }
  }

  return anyChanged ? { ...session, branches: nextBranches } : session;
}

/**
 * Explicitly delete an inactive continuation branch from `session`.
 *
 * Invariants:
 *  - Refuses (throws `RangeError`) if `branchId` does not exist, if `branchId` is
 *    `session.activeBranchId`, or if it is the only remaining branch.
 *  - Never mutates or destroys any shared ancestor step still referenced by surviving branches.
 */
export function deleteDecodeSessionBranch(
  session: HybridDecodeSession,
  branchId: string,
): HybridDecodeSession {
  if (!session.branches.has(branchId)) {
    throw new RangeError(`Unknown decode session branch "${branchId}"`);
  }
  if (branchId === session.activeBranchId) {
    throw new RangeError(
      `Cannot delete active decode session branch "${branchId}"; switch to another branch first`,
    );
  }
  if (session.branches.size <= 1) {
    throw new RangeError(`Cannot delete the only remaining decode session branch "${branchId}"`);
  }

  const nextBranches = new Map(session.branches);
  nextBranches.delete(branchId);
  return {
    ...session,
    branches: nextBranches,
  };
}

/**
 * Enforce conservative retention bounds on `session` by pruning only the oldest inactive
 * branches until `branches.size <= maxBranches` and `uniqueSteps <= maxUniqueSteps`, and
 * optionally dropping reconstructible eager traces from inactive historical steps.
 *
 * Invariants:
 *  - NEVER deletes or truncates `session.activeBranchId` (even if the active branch alone
 *    exceeds `maxUniqueSteps`).
 *  - NEVER corrupts or drops any shared ancestor step referenced by surviving branches.
 */
export function pruneDecodeSessionBranches(
  session: HybridDecodeSession,
  options?: PruneDecodeSessionOptions,
): HybridDecodeSession {
  const maxBranches = validatePositiveIntBound(
    options?.maxBranches,
    session.maxBranches,
    'maxBranches',
  );
  const maxUniqueSteps = validatePositiveIntBound(
    options?.maxUniqueSteps,
    session.maxUniqueSteps,
    'maxUniqueSteps',
  );
  const dropInactiveTraces =
    options?.pruneInactiveStepTraces ?? session.pruneInactiveStepTraces;

  let nextBranches: Map<string, HybridDecodeBranch> | null = null;
  const currentBranches = () => nextBranches ?? session.branches;

  while (
    currentBranches().size > 1 &&
    (currentBranches().size > maxBranches ||
      countUniqueStepsAcrossBranches(currentBranches()) > maxUniqueSteps)
  ) {
    let oldestInactiveId: string | null = null;
    for (const branchId of currentBranches().keys()) {
      if (branchId !== session.activeBranchId) {
        oldestInactiveId = branchId;
        break;
      }
    }
    if (oldestInactiveId === null) break;
    if (nextBranches === null) {
      nextBranches = new Map(session.branches);
    }
    nextBranches.delete(oldestInactiveId);
  }

  let resultSession: HybridDecodeSession =
    nextBranches !== null ? { ...session, branches: nextBranches } : session;
  if (dropInactiveTraces) {
    resultSession = pruneInactiveStepTraces(resultSession);
  }
  return resultSession;
}

/**
 * Append one newly chosen token (`nextTokenId`) as a cached single-token AR decode step
 * after the currently inspected step.
 *
 * - If currently at the tip of the active branch, appends onto the active branch.
 * - If currently inspecting an earlier historical step (`activeStepIndex < branch.steps.length - 1`),
 *   cleanly spawns a new immutable branch (`branch-N`) sharing the immutable prefix steps up to
 *   `activeStepIndex`, so the prior branch and its snapshots remain untouched!
 * - Automatically enforces the session's `maxBranches`, `maxUniqueSteps`, and
 *   `pruneInactiveStepTraces` retention bounds without ever deleting or corrupting the active branch.
 */
export function stepDecodeSession(
  session: HybridDecodeSession,
  nextTokenId: number,
  nextTokenLabel?: string,
): HybridDecodeSession {
  const activeBranch = getActiveDecodeBranch(session);
  const parentStep = getActiveDecodeStep(session);
  if (parentStep.totalTokens >= session.maxTokens) {
    return session;
  }

  const isFork = session.activeStepIndex < activeBranch.steps.length - 1;
  const targetBranchId = isFork ? `branch-${session.nextBranchNumber}` : activeBranch.branchId;
  const nextStepIndex = session.activeStepIndex + 1;

  const nextStep = runIncrementalDecodeStep(session.model, {
    newTokenIds: [nextTokenId],
    newTokenLabels: nextTokenLabel ? [nextTokenLabel] : undefined,
    prevState: parentStep.nextState,
    maxTokens: session.maxTokens,
    precisionMetadata: session.precisionMetadata,
    branchId: targetBranchId,
    parentStepExecutionId: parentStep.stepExecutionId,
    initialStateDigest: parentStep.provenance.initialStateDigest,
    stepIndex: nextStepIndex,
  });

  const nextBranches = new Map(session.branches);
  if (isFork) {
    const prefixSteps = activeBranch.steps.slice(0, session.activeStepIndex + 1);
    nextBranches.set(targetBranchId, {
      branchId: targetBranchId,
      forkedFromBranchId: activeBranch.branchId,
      forkedFromStepExecutionId: parentStep.stepExecutionId,
      steps: Object.freeze([...prefixSteps, nextStep]),
    });
  } else {
    nextBranches.set(activeBranch.branchId, {
      ...activeBranch,
      steps: Object.freeze([...activeBranch.steps, nextStep]),
    });
  }

  const candidateSession: HybridDecodeSession = {
    ...session,
    activeBranchId: targetBranchId,
    activeStepIndex: nextStepIndex,
    nextBranchNumber: isFork ? session.nextBranchNumber + 1 : session.nextBranchNumber,
    branches: nextBranches,
  };
  return pruneDecodeSessionBranches(candidateSession);
}

/** Convenience helper inserting and processing the active step's Greedy argmax candidate. */
export function stepDecodeSessionGreedy(session: HybridDecodeSession): HybridDecodeSession {
  const active = getActiveDecodeStep(session);
  return stepDecodeSession(session, active.greedyNextToken.tokenId);
}

/** Navigate to an existing step index on the active branch without altering snapshots. */
export function navigateDecodeSession(
  session: HybridDecodeSession,
  targetStepIndex: number,
): HybridDecodeSession {
  const branch = getActiveDecodeBranch(session);
  const clamped = Math.max(0, Math.min(branch.steps.length - 1, Math.trunc(targetStepIndex)));
  if (clamped === session.activeStepIndex) return session;
  return { ...session, activeStepIndex: clamped };
}

/** Switch inspection to another preserved continuation branch. */
export function switchDecodeSessionBranch(
  session: HybridDecodeSession,
  branchId: string,
): HybridDecodeSession {
  const target = session.branches.get(branchId);
  if (!target) {
    throw new RangeError(`Unknown decode session branch "${branchId}"`);
  }
  return {
    ...session,
    activeBranchId: branchId,
    activeStepIndex: target.steps.length - 1,
  };
}

/**
 * Measure exact deduplicated retained typed-array bytes across all unique steps stored in
 * `session`, deduplicating shared step objects (by both `stepExecutionId` and object identity)
 * as well as shared underlying `ArrayBuffer`s across steps, and accurately including
 * `rootStep.prevState` bytes.
 */
export function measureDecodeSessionRetainedBytes(session: HybridDecodeSession): {
  readonly uniqueSteps: number;
  readonly branchCount: number;
  readonly rootPrevStateBytes: number;
  readonly totalRetainedBytes: number;
} {
  const seenStepIds = new Set<string>();
  const seenStepObjects = new Set<HybridIncrementalStepResult>();
  const uniqueStepsList: HybridIncrementalStepResult[] = [];
  for (const branch of session.branches.values()) {
    for (const step of branch.steps) {
      if (seenStepIds.has(step.stepExecutionId) || seenStepObjects.has(step)) {
        continue;
      }
      seenStepIds.add(step.stepExecutionId);
      seenStepObjects.add(step);
      uniqueStepsList.push(step);
    }
  }

  const seenBuffers = new Set<ArrayBufferLike>();
  let stepOwnedBytes = 0;
  for (const step of uniqueStepsList) {
    const collector = stepByteCollectors.get(step);
    if (collector) {
      stepOwnedBytes += collector(seenBuffers);
    } else {
      stepOwnedBytes += step.retainedTypedArrayBytes();
      stepOwnedBytes += collectCacheStateUniqueBytes(seenBuffers, step.nextState);
    }
  }

  // Pass 2: account for `prevState` buffers not already owned as any surviving step's `nextState`
  // (e.g., `rootStep.prevState` zero or external seed snapshot).
  let rootPrevStateBytes = 0;
  for (const step of uniqueStepsList) {
    rootPrevStateBytes += collectCacheStateUniqueBytes(seenBuffers, step.prevState);
  }

  return {
    uniqueSteps: uniqueStepsList.length,
    branchCount: session.branches.size,
    rootPrevStateBytes,
    totalRetainedBytes: stepOwnedBytes + rootPrevStateBytes,
  };
}
