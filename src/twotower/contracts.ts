/**
 * Read-Only Cross-Tower Conditioning & Context Snapshot Contracts (`src/twotower/contracts.ts`)
 *
 * Enforces the core architectural ownership boundary of Nemotron-Labs-TwoTower:
 *   - The **Context Tower** exclusively owns persistent sequence caches (`HybridDecodeCacheState`
 *     plus `mambaS2` boundary snapshot for Mock-AR).
 *   - The **Denoiser Tower** consumes a strictly read-only, defensively cloned, non-aliased
 *     `TwoTowerContextConditioningState` (`ownerRole: 'context_tower'`, `consumerRole: 'denoiser_tower'`).
 *   - Context-owned caches are NEVER relabeled as Denoiser-owned caches, and Nano's strict
 *     `validateDecodeCacheState` is always enforced without weakening.
 *   - Only the 29 stateful bays (`6` GQA `*` + `23` Mamba-2 `M`) transfer state across towers;
 *     all `23` MoE (`E`) bays remain strictly stateless.
 *
 * Pure headless TypeScript — zero DOM, WebGL, scene, or render imports.
 */

import type {
  GqaLayerKvCache,
  HybridDecodeCacheState,
  MambaLayerBoundaryState,
} from '../math/cacheContracts';
import { GQA_SIM_CONFIG } from '../math/gqaSim';
import type { HybridModelParameters } from '../math/hybridFlow';
import {
  MAX_INCREMENTAL_STEP_TOKENS,
  validateDecodeCacheState,
} from '../math/incrementalDecode';
import { MAMBA_SIM_CONFIG } from '../math/mambaSim';
import type { Kind } from '../model/spec';
import type { TwoTowerModelParameters } from './parameters';
import type { TwoTowerExecutionMode } from './spec';

export type { TwoTowerExecutionMode };
export type TwoTowerConditioningMode = 'mock_ar' | 'mask_diffusion';
export type TwoTowerAlignedStateRole = 'gqa_kv_prefix' | 'mamba_boundary_seed';

export interface TwoTowerAlignedLayerEntry {
  readonly bayId: number;
  readonly layerIndex: number;
  readonly kind: 'M' | '*';
  readonly stateRole: TwoTowerAlignedStateRole;
}

/**
 * Immutable-by-contract snapshot of the Context Tower's committed state after `P >= 1` tokens.
 * Holds BOTH:
 *   - `ctxState`: Context's committed `S-1` Mamba state + full `KV[0..P-1]` cache
 *   - `mambaS2`: Context's Mamba state immediately prior to the last token (`processedTokens = P - 1`;
 *     zeroed when `P === 1`), preserving exact `S-2 / KV[:-1]` Mock-AR semantics.
 */
export interface TwoTowerContextSnapshot {
  readonly modelId: string;
  readonly parameterFingerprint: string;
  readonly ctxState: HybridDecodeCacheState;
  readonly mambaS2: ReadonlyMap<number, MambaLayerBoundaryState>;
  readonly lastTokenId: number;
  readonly lastTokenLabel: string;
  readonly arLogits: Float32Array;
  readonly arProbs: Float32Array;
  readonly snapshotDigest: string;
}

/**
 * Read-only cross-tower conditioning payload transferred from Context Tower to Denoiser Tower.
 * Explicitly preserves `ownerRole: 'context_tower'` and `consumerRole: 'denoiser_tower'` so
 * Context-owned KV/Mamba caches are NEVER mislabeled as Denoiser-owned caches.
 */
export interface TwoTowerContextConditioningState {
  readonly ownerRole: 'context_tower';
  readonly consumerRole: 'denoiser_tower';
  readonly mode: TwoTowerConditioningMode;
  readonly sourceModelId: string;
  readonly sourceParameterFingerprint: string;
  readonly sourceStateDigest: string;
  readonly targetModelId: string;
  readonly targetParameterFingerprint: string;
  readonly layerAlignment: readonly TwoTowerAlignedLayerEntry[];
  /** Number of Context prefix tokens exposed to Denoiser (`P` in `mask_diffusion`, `P - 1` in `mock_ar`). */
  readonly committedPrefixLen: number;
  /** Total committed prefix tokens `P` held by the source Context snapshot. */
  readonly sourceFullPrefixLen: number;
  /** Allocated KV capacity stride along the sequence axis (`maxTokens`). */
  readonly allocatedKvCapacity: number;
  /** Absolute starting position of Denoiser's query tokens (`P` in `mask_diffusion`, `P - 1` in `mock_ar`). */
  readonly absoluteTokenStart: number;
  readonly blockId: number;
  readonly refinementIndex: number;
  readonly prefixTokenIds: readonly number[];
  readonly prefixTokenLabels: readonly string[];
  /** Defensively cloned, non-aliased GQA prefix KV caches (`seqLen === committedPrefixLen`, zeroed beyond). */
  readonly gqaPrefixCaches: ReadonlyMap<number, GqaLayerKvCache>;
  /** Defensively cloned, non-aliased Mamba boundary seeds (`S-1` for diffusion, `mambaS2` for `mock_ar`). */
  readonly mambaSeedStates: ReadonlyMap<number, MambaLayerBoundaryState>;
  readonly conditioningDigest: string;
}

export interface BuildContextConditioningOptions {
  readonly mode: TwoTowerConditioningMode;
  readonly blockId: number;
  readonly refinementIndex: number;
}

// ============================================================================
// DETERMINISTIC 64-BIT HASHER & INTERNAL HELPERS
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

function assertAllFinite(arr: Float32Array, label: string): void {
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) {
      throw new RangeError(`Non-finite value (${arr[i]}) in ${label}[${i}]`);
    }
  }
}

function registerUniqueBuffer(
  seen: Set<ArrayBufferLike>,
  arr: ArrayBufferView,
  label: string,
): void {
  if (seen.has(arr.buffer)) {
    throw new TypeError(`Buffer aliasing detected at ${label}`);
  }
  seen.add(arr.buffer);
}

/**
 * Build the authoritative ordered list of stateful aligned transfer layers for `pattern`.
 * For the 52-layer Nemotron hybrid motif: `29` entries (`23` Mamba-2 + `6` GQA); the `23` MoE
 * layers are explicitly stateless and omitted from cross-tower state transfer.
 */
export function buildTwoTowerLayerAlignment(pattern: string): readonly TwoTowerAlignedLayerEntry[] {
  const entries: TwoTowerAlignedLayerEntry[] = [];
  for (let layerIndex = 0; layerIndex < pattern.length; layerIndex++) {
    const kind = pattern[layerIndex] as Kind;
    const bayId = layerIndex + 1;
    if (kind === 'M') {
      entries.push(
        Object.freeze({
          bayId,
          layerIndex,
          kind: 'M' as const,
          stateRole: 'mamba_boundary_seed' as const,
        }),
      );
    } else if (kind === '*') {
      entries.push(
        Object.freeze({
          bayId,
          layerIndex,
          kind: '*' as const,
          stateRole: 'gqa_kv_prefix' as const,
        }),
      );
    }
  }
  return Object.freeze(entries);
}

// ============================================================================
// CONTEXT SNAPSHOT DIGEST, CONSTRUCTOR & STRICT VALIDATOR
// ============================================================================

/**
 * Compute deterministic 64-bit hex digest covering `ctxState.stateDigest`, last token identity,
 * every `mambaS2` boundary tensor, and Context AR logits/probs.
 */
export function computeContextSnapshotDigest(params: {
  readonly modelId: string;
  readonly parameterFingerprint: string;
  readonly ctxStateDigest: string;
  readonly lastTokenId: number;
  readonly lastTokenLabel: string;
  readonly mambaS2: ReadonlyMap<number, MambaLayerBoundaryState>;
  readonly arLogits: Float32Array;
  readonly arProbs: Float32Array;
}): string {
  const h = makeDualHasher();
  h.feedStr('twotower-context-snapshot-v1');
  h.feedStr(params.modelId);
  h.feedStr(params.parameterFingerprint);
  h.feedStr(params.ctxStateDigest);
  h.feedI32(params.lastTokenId);
  h.feedStr(params.lastTokenLabel);

  const mambaIds = Array.from(params.mambaS2.keys()).sort((a, b) => a - b);
  h.feedU32(mambaIds.length);
  for (const bayId of mambaIds) {
    const m = params.mambaS2.get(bayId)!;
    h.feedU32(bayId);
    h.feedU32(m.processedTokens);
    for (let i = 0; i < m.convState.length; i++) h.feedF32(m.convState[i]);
    for (let i = 0; i < m.ssmState.length; i++) h.feedF32(m.ssmState[i]);
  }

  h.feedU32(params.arLogits.length);
  for (let i = 0; i < params.arLogits.length; i++) h.feedF32(params.arLogits[i]);
  h.feedU32(params.arProbs.length);
  for (let i = 0; i < params.arProbs.length; i++) h.feedF32(params.arProbs[i]);

  return h.digest();
}

/**
 * Construct and validate a frozen, non-aliased `TwoTowerContextSnapshot`.
 * Defensively clones `mambaS2`, `arLogits`, and `arProbs` so external mutations cannot corrupt it.
 */
export function createTwoTowerContextSnapshot(
  contextModel: HybridModelParameters,
  params: {
    readonly ctxState: HybridDecodeCacheState;
    readonly mambaS2: ReadonlyMap<number, MambaLayerBoundaryState>;
    readonly arLogits: Float32Array;
    readonly arProbs: Float32Array;
  },
): TwoTowerContextSnapshot {
  validateDecodeCacheState(contextModel, params.ctxState);
  const P = params.ctxState.processedTokens;
  if (P < 1) {
    throw new RangeError(
      `TwoTowerContextSnapshot requires at least 1 committed token, received processedTokens=${P}`,
    );
  }

  const lastTokenId = params.ctxState.prefixTokenIds[P - 1];
  const lastTokenLabel = params.ctxState.prefixTokenLabels[P - 1];

  const clonedMambaS2 = new Map<number, MambaLayerBoundaryState>();
  for (const [bayId, m] of params.mambaS2.entries()) {
    clonedMambaS2.set(
      bayId,
      Object.freeze({
        bayId: m.bayId,
        kind: 'M' as const,
        convDim: m.convDim,
        convKernel: m.convKernel,
        convState: new Float32Array(m.convState),
        nHeads: m.nHeads,
        dHead: m.dHead,
        dState: m.dState,
        ssmState: new Float32Array(m.ssmState),
        processedTokens: m.processedTokens,
        storageDtypeMetadata: Object.freeze({ conv1d: 'bf16' as const, ssm: 'fp32' as const }),
      }),
    );
  }

  const arLogits = new Float32Array(params.arLogits);
  const arProbs = new Float32Array(params.arProbs);

  const snapshotDigest = computeContextSnapshotDigest({
    modelId: contextModel.modelId,
    parameterFingerprint: contextModel.parameterFingerprint,
    ctxStateDigest: params.ctxState.stateDigest,
    lastTokenId,
    lastTokenLabel,
    mambaS2: clonedMambaS2,
    arLogits,
    arProbs,
  });

  const snapshot: TwoTowerContextSnapshot = Object.freeze({
    modelId: contextModel.modelId,
    parameterFingerprint: contextModel.parameterFingerprint,
    ctxState: params.ctxState,
    mambaS2: clonedMambaS2,
    lastTokenId,
    lastTokenLabel,
    arLogits,
    arProbs,
    snapshotDigest,
  });

  validateContextSnapshot(snapshot, contextModel);
  return snapshot;
}

/**
 * Strictly validate a `TwoTowerContextSnapshot` against `contextModel`:
 *   - Runs Nano's full `validateDecodeCacheState(contextModel, snapshot.ctxState)` unchanged
 *   - Validates `P >= 1`, `lastTokenId`/`lastTokenLabel`, `mambaS2` complete bay coverage (`processedTokens === P - 1`)
 *   - Rejects any buffer aliasing between `ctxState`, `mambaS2`, `arLogits`, and `arProbs`
 *   - Verifies finite values and `snapshotDigest` integrity.
 */
export function validateContextSnapshot(
  snapshot: TwoTowerContextSnapshot,
  contextModel: HybridModelParameters,
): void {
  if (snapshot.modelId !== contextModel.modelId) {
    throw new TypeError(
      `Context snapshot modelId mismatch: snapshot=${snapshot.modelId} !== contextModel=${contextModel.modelId}`,
    );
  }
  if (snapshot.parameterFingerprint !== contextModel.parameterFingerprint) {
    throw new TypeError(
      `Context snapshot parameterFingerprint mismatch: snapshot=${snapshot.parameterFingerprint} !== contextModel=${contextModel.parameterFingerprint}`,
    );
  }

  // Strictly enforce Nano's full cache validator without weakening
  validateDecodeCacheState(contextModel, snapshot.ctxState);

  const P = snapshot.ctxState.processedTokens;
  if (P < 1) {
    throw new RangeError(`Context snapshot requires processedTokens >= 1, received ${P}`);
  }
  if (snapshot.lastTokenId !== snapshot.ctxState.prefixTokenIds[P - 1]) {
    throw new RangeError(
      `Context snapshot lastTokenId (${snapshot.lastTokenId}) !== ctxState.prefixTokenIds[${P - 1}] (${snapshot.ctxState.prefixTokenIds[P - 1]})`,
    );
  }
  if (snapshot.lastTokenLabel !== snapshot.ctxState.prefixTokenLabels[P - 1]) {
    throw new RangeError(
      `Context snapshot lastTokenLabel ("${snapshot.lastTokenLabel}") !== ctxState.prefixTokenLabels[${P - 1}] ("${snapshot.ctxState.prefixTokenLabels[P - 1]}")`,
    );
  }

  const vocabSize = contextModel.config.vocabSize;
  if (snapshot.arLogits.length !== vocabSize) {
    throw new RangeError(
      `Context snapshot arLogits length (${snapshot.arLogits.length}) !== vocabSize (${vocabSize})`,
    );
  }
  if (snapshot.arProbs.length !== vocabSize) {
    throw new RangeError(
      `Context snapshot arProbs length (${snapshot.arProbs.length}) !== vocabSize (${vocabSize})`,
    );
  }
  assertAllFinite(snapshot.arLogits, 'contextSnapshot.arLogits');
  assertAllFinite(snapshot.arProbs, 'contextSnapshot.arProbs');

  const seen = new Set<ArrayBufferLike>();
  registerUniqueBuffer(seen, snapshot.ctxState.prefixTokenIds, 'ctxState.prefixTokenIds');
  for (const [bayId, m] of snapshot.ctxState.mambaStates.entries()) {
    registerUniqueBuffer(seen, m.convState, `ctxState.mamba[${bayId}].convState`);
    registerUniqueBuffer(seen, m.ssmState, `ctxState.mamba[${bayId}].ssmState`);
  }
  for (const [bayId, g] of snapshot.ctxState.gqaCaches.entries()) {
    registerUniqueBuffer(seen, g.K, `ctxState.gqa[${bayId}].K`);
    registerUniqueBuffer(seen, g.V, `ctxState.gqa[${bayId}].V`);
  }
  registerUniqueBuffer(seen, snapshot.arLogits, 'contextSnapshot.arLogits');
  registerUniqueBuffer(seen, snapshot.arProbs, 'contextSnapshot.arProbs');

  const pattern = contextModel.config.pattern;
  const expectedMambaIds: number[] = [];
  for (let i = 0; i < pattern.length; i++) {
    if (pattern[i] === 'M') expectedMambaIds.push(i + 1);
  }
  if (snapshot.mambaS2.size !== expectedMambaIds.length) {
    throw new RangeError(
      `Context snapshot mambaS2 count (${snapshot.mambaS2.size}) !== expected (${expectedMambaIds.length})`,
    );
  }

  const expectedConvLen = MAMBA_SIM_CONFIG.convDim * MAMBA_SIM_CONFIG.convKernel;
  const expectedSsmLen = MAMBA_SIM_CONFIG.nHeads * MAMBA_SIM_CONFIG.dHead * MAMBA_SIM_CONFIG.dState;
  const expectedS2Processed = P - 1;

  for (const bayId of expectedMambaIds) {
    const m = snapshot.mambaS2.get(bayId);
    if (!m) {
      throw new RangeError(`Context snapshot missing mambaS2 state for bay ${bayId}`);
    }
    if (m.bayId !== bayId || m.kind !== 'M') {
      throw new TypeError(`mambaS2 metadata mismatch at bay ${bayId}`);
    }
    if (
      m.convDim !== MAMBA_SIM_CONFIG.convDim ||
      m.convKernel !== MAMBA_SIM_CONFIG.convKernel ||
      m.nHeads !== MAMBA_SIM_CONFIG.nHeads ||
      m.dHead !== MAMBA_SIM_CONFIG.dHead ||
      m.dState !== MAMBA_SIM_CONFIG.dState
    ) {
      throw new RangeError(`mambaS2 dimension mismatch at bay ${bayId}`);
    }
    if (m.convState.length !== expectedConvLen) {
      throw new RangeError(
        `mambaS2 bay ${bayId} convState length (${m.convState.length}) !== expected (${expectedConvLen})`,
      );
    }
    if (m.ssmState.length !== expectedSsmLen) {
      throw new RangeError(
        `mambaS2 bay ${bayId} ssmState length (${m.ssmState.length}) !== expected (${expectedSsmLen})`,
      );
    }
    if (m.processedTokens !== expectedS2Processed) {
      throw new RangeError(
        `mambaS2 bay ${bayId} processedTokens (${m.processedTokens}) !== expected (${expectedS2Processed})`,
      );
    }
    registerUniqueBuffer(seen, m.convState, `mambaS2[${bayId}].convState`);
    registerUniqueBuffer(seen, m.ssmState, `mambaS2[${bayId}].ssmState`);
    assertAllFinite(m.convState, `mambaS2[${bayId}].convState`);
    assertAllFinite(m.ssmState, `mambaS2[${bayId}].ssmState`);
  }

  const expectedDigest = computeContextSnapshotDigest({
    modelId: snapshot.modelId,
    parameterFingerprint: snapshot.parameterFingerprint,
    ctxStateDigest: snapshot.ctxState.stateDigest,
    lastTokenId: snapshot.lastTokenId,
    lastTokenLabel: snapshot.lastTokenLabel,
    mambaS2: snapshot.mambaS2,
    arLogits: snapshot.arLogits,
    arProbs: snapshot.arProbs,
  });
  if (snapshot.snapshotDigest !== expectedDigest) {
    throw new Error(
      `Context snapshot digest mismatch: recorded=${snapshot.snapshotDigest} !== computed=${expectedDigest}`,
    );
  }
}

// ============================================================================
// READ-ONLY CROSS-TOWER CONDITIONING DIGEST, BUILDER & VALIDATORS
// ============================================================================

/**
 * Compute unambiguous 64-bit hex digest over all source/target identities, mode, alignment,
 * sequence coordinates, block/refinement IDs, prefix tokens/labels, and every tensor element
 * in `mambaSeedStates` and `gqaPrefixCaches`.
 */
export function computeConditioningDigest(
  state: Omit<TwoTowerContextConditioningState, 'conditioningDigest'>,
): string {
  const h = makeDualHasher();
  h.feedStr('twotower-conditioning-v1');
  h.feedStr(state.ownerRole);
  h.feedStr(state.consumerRole);
  h.feedStr(state.mode);
  h.feedStr(state.sourceModelId);
  h.feedStr(state.sourceParameterFingerprint);
  h.feedStr(state.sourceStateDigest);
  h.feedStr(state.targetModelId);
  h.feedStr(state.targetParameterFingerprint);

  h.feedU32(state.layerAlignment.length);
  for (let i = 0; i < state.layerAlignment.length; i++) {
    const e = state.layerAlignment[i];
    h.feedU32(e.bayId);
    h.feedU32(e.layerIndex);
    h.feedStr(e.kind);
    h.feedStr(e.stateRole);
  }

  h.feedU32(state.committedPrefixLen);
  h.feedU32(state.sourceFullPrefixLen);
  h.feedU32(state.allocatedKvCapacity);
  h.feedU32(state.absoluteTokenStart);
  h.feedU32(state.blockId);
  h.feedU32(state.refinementIndex);

  h.feedU32(state.prefixTokenIds.length);
  for (let i = 0; i < state.prefixTokenIds.length; i++) {
    h.feedI32(state.prefixTokenIds[i]);
  }
  h.feedU32(state.prefixTokenLabels.length);
  for (let i = 0; i < state.prefixTokenLabels.length; i++) {
    h.feedStr(state.prefixTokenLabels[i]);
  }

  const mambaIds = Array.from(state.mambaSeedStates.keys()).sort((a, b) => a - b);
  h.feedU32(mambaIds.length);
  for (const bayId of mambaIds) {
    const m = state.mambaSeedStates.get(bayId)!;
    h.feedU32(bayId);
    h.feedU32(m.processedTokens);
    for (let i = 0; i < m.convState.length; i++) h.feedF32(m.convState[i]);
    for (let i = 0; i < m.ssmState.length; i++) h.feedF32(m.ssmState[i]);
  }

  const gqaIds = Array.from(state.gqaPrefixCaches.keys()).sort((a, b) => a - b);
  h.feedU32(gqaIds.length);
  for (const bayId of gqaIds) {
    const g = state.gqaPrefixCaches.get(bayId)!;
    h.feedU32(bayId);
    h.feedU32(g.seqLen);
    h.feedU32(g.maxTokens);
    for (let i = 0; i < g.K.length; i++) h.feedF32(g.K[i]);
    for (let i = 0; i < g.V.length; i++) h.feedF32(g.V[i]);
  }

  return h.digest();
}

/**
 * Build a read-only, defensively cloned, strictly non-aliased `TwoTowerContextConditioningState`
 * from an immutable `TwoTowerContextSnapshot`.
 *
 * - `'mask_diffusion'`: exposes full committed `KV[0..P-1]` and `S-1` Mamba states (`committedPrefixLen = P`).
 * - `'mock_ar'`: exposes truncated `KV[0..P-2]` and `mambaS2` (`committedPrefixLen = P - 1`, `0` when `P === 1`).
 */
export function buildContextConditioningState(
  model: TwoTowerModelParameters,
  contextSnapshot: TwoTowerContextSnapshot,
  options: BuildContextConditioningOptions,
): TwoTowerContextConditioningState {
  const { mode, blockId, refinementIndex } = options;
  if (mode !== 'mock_ar' && mode !== 'mask_diffusion') {
    throw new RangeError(`Unsupported TwoTower conditioning mode: ${String(mode)}`);
  }
  if (!Number.isInteger(blockId) || blockId < 0) {
    throw new RangeError(`blockId must be a nonnegative integer, received ${blockId}`);
  }
  if (!Number.isInteger(refinementIndex) || refinementIndex < 0) {
    throw new RangeError(
      `refinementIndex must be a nonnegative integer, received ${refinementIndex}`,
    );
  }
  if (model.contextModel.config.pattern !== model.denoiserModel.config.pattern) {
    throw new TypeError(
      'TwoTower Context and Denoiser models must share an identical hybrid layer pattern',
    );
  }

  // Strictly validate source Context snapshot (including Nano's full validateDecodeCacheState)
  validateContextSnapshot(contextSnapshot, model.contextModel);

  const ctxState = contextSnapshot.ctxState;
  const P = ctxState.processedTokens;
  const allocatedKvCapacity = ctxState.maxTokens;
  const committedPrefixLen = mode === 'mock_ar' ? P - 1 : P;
  const absoluteTokenStart = committedPrefixLen;

  const prefixTokenIds: number[] = [];
  const prefixTokenLabels: string[] = [];
  for (let i = 0; i < committedPrefixLen; i++) {
    prefixTokenIds.push(ctxState.prefixTokenIds[i]);
    prefixTokenLabels.push(ctxState.prefixTokenLabels[i]);
  }

  const layerAlignment = buildTwoTowerLayerAlignment(model.contextModel.config.pattern);

  // Defensively clone all 6 GQA prefix slices (0..committedPrefixLen-1), strictly zeroed beyond
  const kvRows = GQA_SIM_CONFIG.nKv * GQA_SIM_CONFIG.dHead;
  const gqaPrefixCaches = new Map<number, GqaLayerKvCache>();
  const mambaSeedStates = new Map<number, MambaLayerBoundaryState>();

  const sourceMambaMap = mode === 'mock_ar' ? contextSnapshot.mambaS2 : ctxState.mambaStates;

  for (const entry of layerAlignment) {
    if (entry.kind === '*') {
      const src = ctxState.gqaCaches.get(entry.bayId);
      if (!src) {
        throw new RangeError(`Context snapshot missing GQA KV cache for bay ${entry.bayId}`);
      }
      const K = new Float32Array(kvRows * allocatedKvCapacity);
      const V = new Float32Array(kvRows * allocatedKvCapacity);
      for (let row = 0; row < kvRows; row++) {
        const srcOff = row * src.maxTokens;
        const dstOff = row * allocatedKvCapacity;
        for (let k = 0; k < committedPrefixLen; k++) {
          K[dstOff + k] = src.K[srcOff + k];
          V[dstOff + k] = src.V[srcOff + k];
        }
      }
      gqaPrefixCaches.set(
        entry.bayId,
        Object.freeze({
          bayId: entry.bayId,
          kind: '*' as const,
          nKv: src.nKv,
          dHead: src.dHead,
          seqLen: committedPrefixLen,
          maxTokens: allocatedKvCapacity,
          K,
          V,
          storageDtypeMetadata: src.storageDtypeMetadata,
        }),
      );
    } else {
      const src = sourceMambaMap.get(entry.bayId);
      if (!src) {
        throw new RangeError(`Context snapshot missing Mamba seed state for bay ${entry.bayId}`);
      }
      mambaSeedStates.set(
        entry.bayId,
        Object.freeze({
          bayId: entry.bayId,
          kind: 'M' as const,
          convDim: src.convDim,
          convKernel: src.convKernel,
          convState: new Float32Array(src.convState),
          nHeads: src.nHeads,
          dHead: src.dHead,
          dState: src.dState,
          ssmState: new Float32Array(src.ssmState),
          processedTokens: committedPrefixLen,
          storageDtypeMetadata: Object.freeze({ conv1d: 'bf16' as const, ssm: 'fp32' as const }),
        }),
      );
    }
  }

  const body: Omit<TwoTowerContextConditioningState, 'conditioningDigest'> = {
    ownerRole: 'context_tower',
    consumerRole: 'denoiser_tower',
    mode,
    sourceModelId: model.contextModel.modelId,
    sourceParameterFingerprint: model.contextModel.parameterFingerprint,
    sourceStateDigest: ctxState.stateDigest,
    targetModelId: model.denoiserModel.modelId,
    targetParameterFingerprint: model.denoiserModel.parameterFingerprint,
    layerAlignment,
    committedPrefixLen,
    sourceFullPrefixLen: P,
    allocatedKvCapacity,
    absoluteTokenStart,
    blockId,
    refinementIndex,
    prefixTokenIds: Object.freeze(prefixTokenIds),
    prefixTokenLabels: Object.freeze(prefixTokenLabels),
    gqaPrefixCaches,
    mambaSeedStates,
  };

  const conditioningDigest = computeConditioningDigest(body);
  return Object.freeze({
    ...body,
    conditioningDigest,
  });
}

/**
 * Exhaustively validate a `TwoTowerContextConditioningState` before Denoiser consumption.
 * Rejects role relabeling, fingerprint mismatch, invalid 29-layer alignment, missing bays,
 * non-finite values, unzeroed trailing KV slots, buffer aliasing, capacity overflow, queryLen > 16,
 * and `conditioningDigest` tampering.
 */
export function validateContextConditioningState(
  state: TwoTowerContextConditioningState,
  model: TwoTowerModelParameters,
  expectedQueryLen?: number,
): void {
  if (state.ownerRole !== 'context_tower') {
    throw new TypeError(
      `Invalid conditioning ownerRole: expected "context_tower", received "${String(state.ownerRole)}"`,
    );
  }
  if (state.consumerRole !== 'denoiser_tower') {
    throw new TypeError(
      `Invalid conditioning consumerRole: expected "denoiser_tower", received "${String(state.consumerRole)}"`,
    );
  }
  if (state.mode !== 'mock_ar' && state.mode !== 'mask_diffusion') {
    throw new RangeError(`Invalid conditioning mode: "${String(state.mode)}"`);
  }
  if (model.contextModel.config.pattern !== model.denoiserModel.config.pattern) {
    throw new TypeError(
      'TwoTower Context and Denoiser models must share an identical hybrid layer pattern',
    );
  }
  if (state.sourceModelId !== model.contextModel.modelId) {
    throw new TypeError(
      `Conditioning sourceModelId mismatch: state=${state.sourceModelId} !== contextModel=${model.contextModel.modelId}`,
    );
  }
  if (state.sourceParameterFingerprint !== model.contextModel.parameterFingerprint) {
    throw new TypeError(
      `Conditioning sourceParameterFingerprint mismatch: state=${state.sourceParameterFingerprint} !== contextModel=${model.contextModel.parameterFingerprint}`,
    );
  }
  if (typeof state.sourceStateDigest !== 'string' || state.sourceStateDigest.length === 0) {
    throw new TypeError('Conditioning sourceStateDigest must be a non-empty string');
  }
  if (state.targetModelId !== model.denoiserModel.modelId) {
    throw new TypeError(
      `Conditioning targetModelId mismatch: state=${state.targetModelId} !== denoiserModel=${model.denoiserModel.modelId}`,
    );
  }
  if (state.targetParameterFingerprint !== model.denoiserModel.parameterFingerprint) {
    throw new TypeError(
      `Conditioning targetParameterFingerprint mismatch: state=${state.targetParameterFingerprint} !== denoiserModel=${model.denoiserModel.parameterFingerprint}`,
    );
  }
  if (!Number.isInteger(state.blockId) || state.blockId < 0) {
    throw new RangeError(`Invalid conditioning blockId: ${state.blockId}`);
  }
  if (!Number.isInteger(state.refinementIndex) || state.refinementIndex < 0) {
    throw new RangeError(`Invalid conditioning refinementIndex: ${state.refinementIndex}`);
  }

  if (!Number.isInteger(state.sourceFullPrefixLen) || state.sourceFullPrefixLen < 1) {
    throw new RangeError(`Invalid sourceFullPrefixLen: ${state.sourceFullPrefixLen}`);
  }
  if (!Number.isInteger(state.allocatedKvCapacity) || state.allocatedKvCapacity < 1) {
    throw new RangeError(`Invalid allocatedKvCapacity: ${state.allocatedKvCapacity}`);
  }
  const expectedCommitted =
    state.mode === 'mock_ar' ? state.sourceFullPrefixLen - 1 : state.sourceFullPrefixLen;
  if (state.committedPrefixLen !== expectedCommitted) {
    throw new RangeError(
      `Conditioning committedPrefixLen (${state.committedPrefixLen}) !== expected (${expectedCommitted}) for mode "${state.mode}"`,
    );
  }
  if (state.absoluteTokenStart !== state.committedPrefixLen) {
    throw new RangeError(
      `Conditioning absoluteTokenStart (${state.absoluteTokenStart}) !== committedPrefixLen (${state.committedPrefixLen})`,
    );
  }
  if (state.committedPrefixLen > state.allocatedKvCapacity) {
    throw new RangeError(
      `Conditioning committedPrefixLen (${state.committedPrefixLen}) exceeds allocatedKvCapacity (${state.allocatedKvCapacity})`,
    );
  }
  if (state.sourceFullPrefixLen > state.allocatedKvCapacity) {
    throw new RangeError(
      `Conditioning sourceFullPrefixLen (${state.sourceFullPrefixLen}) exceeds allocatedKvCapacity (${state.allocatedKvCapacity})`,
    );
  }

  if (expectedQueryLen !== undefined) {
    if (
      !Number.isInteger(expectedQueryLen) ||
      expectedQueryLen < 1 ||
      expectedQueryLen > MAX_INCREMENTAL_STEP_TOKENS
    ) {
      throw new RangeError(
        `Denoiser query length (${expectedQueryLen}) must be an integer in [1..${MAX_INCREMENTAL_STEP_TOKENS}]`,
      );
    }
    if (state.mode === 'mock_ar' && expectedQueryLen !== 1) {
      throw new RangeError(
        `Mock-AR mode requires query length === 1, received ${expectedQueryLen}`,
      );
    }
    if (state.committedPrefixLen + expectedQueryLen > state.allocatedKvCapacity) {
      throw new RangeError(
        `Denoiser pass (${state.committedPrefixLen} + ${expectedQueryLen} = ${state.committedPrefixLen + expectedQueryLen}) exceeds allocatedKvCapacity (${state.allocatedKvCapacity})`,
      );
    }
  }

  if (state.prefixTokenIds.length !== state.committedPrefixLen) {
    throw new RangeError(
      `Conditioning prefixTokenIds length (${state.prefixTokenIds.length}) !== committedPrefixLen (${state.committedPrefixLen})`,
    );
  }
  if (state.prefixTokenLabels.length !== state.committedPrefixLen) {
    throw new RangeError(
      `Conditioning prefixTokenLabels length (${state.prefixTokenLabels.length}) !== committedPrefixLen (${state.committedPrefixLen})`,
    );
  }

  // Verify exact aligned transfer layer structure (29 stateful bays for 52-layer motif)
  const expectedAlignment = buildTwoTowerLayerAlignment(model.contextModel.config.pattern);
  if (state.layerAlignment.length !== expectedAlignment.length) {
    throw new RangeError(
      `Conditioning layerAlignment length (${state.layerAlignment.length}) !== expected (${expectedAlignment.length})`,
    );
  }

  let expectedMambaCount = 0;
  let expectedGqaCount = 0;
  for (let i = 0; i < expectedAlignment.length; i++) {
    const actual = state.layerAlignment[i];
    const exp = expectedAlignment[i];
    if (
      actual.bayId !== exp.bayId ||
      actual.layerIndex !== exp.layerIndex ||
      actual.kind !== exp.kind ||
      actual.stateRole !== exp.stateRole
    ) {
      throw new TypeError(
        `Conditioning layerAlignment mismatch at index ${i}: expected bay ${exp.bayId} (${exp.kind}/${exp.stateRole})`,
      );
    }
    if (exp.kind === 'M') expectedMambaCount++;
    else expectedGqaCount++;
  }

  if (state.mambaSeedStates.size !== expectedMambaCount) {
    throw new RangeError(
      `Conditioning mambaSeedStates count (${state.mambaSeedStates.size}) !== expected (${expectedMambaCount})`,
    );
  }
  if (state.gqaPrefixCaches.size !== expectedGqaCount) {
    throw new RangeError(
      `Conditioning gqaPrefixCaches count (${state.gqaPrefixCaches.size}) !== expected (${expectedGqaCount})`,
    );
  }

  const seenBuffers = new Set<ArrayBufferLike>();
  const expectedConvLen = MAMBA_SIM_CONFIG.convDim * MAMBA_SIM_CONFIG.convKernel;
  const expectedSsmLen = MAMBA_SIM_CONFIG.nHeads * MAMBA_SIM_CONFIG.dHead * MAMBA_SIM_CONFIG.dState;
  const kvRows = GQA_SIM_CONFIG.nKv * GQA_SIM_CONFIG.dHead;
  const expectedKvLen = kvRows * state.allocatedKvCapacity;

  for (const exp of expectedAlignment) {
    if (exp.kind === 'M') {
      const m = state.mambaSeedStates.get(exp.bayId);
      if (!m) {
        throw new RangeError(`Conditioning missing Mamba seed state for bay ${exp.bayId}`);
      }
      if (m.bayId !== exp.bayId || m.kind !== 'M') {
        throw new TypeError(`Conditioning Mamba metadata mismatch at bay ${exp.bayId}`);
      }
      if (
        m.convDim !== MAMBA_SIM_CONFIG.convDim ||
        m.convKernel !== MAMBA_SIM_CONFIG.convKernel ||
        m.nHeads !== MAMBA_SIM_CONFIG.nHeads ||
        m.dHead !== MAMBA_SIM_CONFIG.dHead ||
        m.dState !== MAMBA_SIM_CONFIG.dState
      ) {
        throw new RangeError(`Conditioning Mamba dimension mismatch at bay ${exp.bayId}`);
      }
      if (m.convState.length !== expectedConvLen) {
        throw new RangeError(
          `Conditioning Mamba bay ${exp.bayId} convState length (${m.convState.length}) !== expected (${expectedConvLen})`,
        );
      }
      if (m.ssmState.length !== expectedSsmLen) {
        throw new RangeError(
          `Conditioning Mamba bay ${exp.bayId} ssmState length (${m.ssmState.length}) !== expected (${expectedSsmLen})`,
        );
      }
      if (m.processedTokens !== state.committedPrefixLen) {
        throw new RangeError(
          `Conditioning Mamba bay ${exp.bayId} processedTokens (${m.processedTokens}) !== committedPrefixLen (${state.committedPrefixLen})`,
        );
      }
      registerUniqueBuffer(seenBuffers, m.convState, `conditioning.mamba[${exp.bayId}].convState`);
      registerUniqueBuffer(seenBuffers, m.ssmState, `conditioning.mamba[${exp.bayId}].ssmState`);
      assertAllFinite(m.convState, `conditioning.mamba[${exp.bayId}].convState`);
      assertAllFinite(m.ssmState, `conditioning.mamba[${exp.bayId}].ssmState`);
    } else {
      const g = state.gqaPrefixCaches.get(exp.bayId);
      if (!g) {
        throw new RangeError(`Conditioning missing GQA KV prefix cache for bay ${exp.bayId}`);
      }
      if (g.bayId !== exp.bayId || g.kind !== '*') {
        throw new TypeError(`Conditioning GQA metadata mismatch at bay ${exp.bayId}`);
      }
      if (g.nKv !== GQA_SIM_CONFIG.nKv || g.dHead !== GQA_SIM_CONFIG.dHead) {
        throw new RangeError(`Conditioning GQA head dimension mismatch at bay ${exp.bayId}`);
      }
      if (g.seqLen !== state.committedPrefixLen) {
        throw new RangeError(
          `Conditioning GQA bay ${exp.bayId} seqLen (${g.seqLen}) !== committedPrefixLen (${state.committedPrefixLen})`,
        );
      }
      if (g.maxTokens !== state.allocatedKvCapacity) {
        throw new RangeError(
          `Conditioning GQA bay ${exp.bayId} maxTokens (${g.maxTokens}) !== allocatedKvCapacity (${state.allocatedKvCapacity})`,
        );
      }
      if (g.K.length !== expectedKvLen || g.V.length !== expectedKvLen) {
        throw new RangeError(
          `Conditioning GQA bay ${exp.bayId} K/V length (${g.K.length}/${g.V.length}) !== expected (${expectedKvLen})`,
        );
      }
      registerUniqueBuffer(seenBuffers, g.K, `conditioning.gqa[${exp.bayId}].K`);
      registerUniqueBuffer(seenBuffers, g.V, `conditioning.gqa[${exp.bayId}].V`);
      for (let row = 0; row < kvRows; row++) {
        const off = row * g.maxTokens;
        for (let k = 0; k < g.seqLen; k++) {
          if (!Number.isFinite(g.K[off + k])) {
            throw new RangeError(
              `Non-finite value (${g.K[off + k]}) in conditioning.gqa[${exp.bayId}].K[${row},${k}]`,
            );
          }
          if (!Number.isFinite(g.V[off + k])) {
            throw new RangeError(
              `Non-finite value (${g.V[off + k]}) in conditioning.gqa[${exp.bayId}].V[${row},${k}]`,
            );
          }
        }
        for (let k = g.seqLen; k < g.maxTokens; k++) {
          if (g.K[off + k] !== 0 || g.V[off + k] !== 0) {
            throw new RangeError(
              `Unzeroed trailing KV slot at conditioning.gqa[${exp.bayId}][${row},${k}] (beyond committedPrefixLen=${g.seqLen})`,
            );
          }
        }
      }
    }
  }

  const expectedDigest = computeConditioningDigest(state);
  if (state.conditioningDigest !== expectedDigest) {
    throw new Error(
      `Conditioning digest mismatch: recorded=${state.conditioningDigest} !== computed=${expectedDigest}`,
    );
  }
}

/**
 * Assert strictly zero backing `ArrayBuffer` sharing between a Context Tower snapshot and a
 * read-only Denoiser conditioning state (as well as internal non-aliasing inside `conditioning`).
 */
export function assertNonAliasedWithContextSnapshot(
  conditioning: TwoTowerContextConditioningState,
  contextSnapshot: TwoTowerContextSnapshot,
): void {
  const ctxBuffers = new Set<ArrayBufferLike>();
  ctxBuffers.add(contextSnapshot.ctxState.prefixTokenIds.buffer);
  ctxBuffers.add(contextSnapshot.arLogits.buffer);
  ctxBuffers.add(contextSnapshot.arProbs.buffer);

  for (const m of contextSnapshot.ctxState.mambaStates.values()) {
    ctxBuffers.add(m.convState.buffer);
    ctxBuffers.add(m.ssmState.buffer);
  }
  for (const g of contextSnapshot.ctxState.gqaCaches.values()) {
    ctxBuffers.add(g.K.buffer);
    ctxBuffers.add(g.V.buffer);
  }
  for (const m of contextSnapshot.mambaS2.values()) {
    ctxBuffers.add(m.convState.buffer);
    ctxBuffers.add(m.ssmState.buffer);
  }

  const condBuffers = new Set<ArrayBufferLike>();
  const check = (arr: ArrayBufferView, label: string) => {
    if (ctxBuffers.has(arr.buffer)) {
      throw new TypeError(`Cross-tower buffer aliasing detected between Context snapshot and ${label}`);
    }
    if (condBuffers.has(arr.buffer)) {
      throw new TypeError(`Internal conditioning buffer aliasing detected at ${label}`);
    }
    condBuffers.add(arr.buffer);
  };

  for (const [bayId, m] of conditioning.mambaSeedStates.entries()) {
    check(m.convState, `conditioning.mambaSeedStates[${bayId}].convState`);
    check(m.ssmState, `conditioning.mambaSeedStates[${bayId}].ssmState`);
  }
  for (const [bayId, g] of conditioning.gqaPrefixCaches.entries()) {
    check(g.K, `conditioning.gqaPrefixCaches[${bayId}].K`);
    check(g.V, `conditioning.gqaPrefixCaches[${bayId}].V`);
  }
}
