/**
 * Incremental Autoregressive (AR) Decode Cache, Ownership & Provenance Contracts
 *
 * Architectural memory boundary across Nemotron 3 Nano's 3 layer kinds (and extensible
 * to future Nemotron Labs hybrid architectures):
 *  - GQA Attention (`*`, 6 bays): linear per-token K/V cache growth (`seqLen` entries per KV head)
 *  - Mamba-2 (`M`, 23 bays): strictly fixed-size rolling Conv1D buffer + FP32 persistent SSM state
 *  - Granular MoE (`E`, 23 bays): strictly stateless across tokens (`0 B` sequence cache)
 *
 * Ownership & Mutation Rules:
 *  1. **Typed Arrays (`Float32Array`, `Int32Array`)**:
 *     Every `HybridDecodeCacheState` snapshot and every nested `GqaLayerKvCache` /
 *     `MambaLayerBoundaryState` owns private, non-aliased backing `ArrayBuffer`s.
 *     Snapshots are strictly IMMUTABLE BY CONTRACT (copy-on-step): stepping or branching
 *     from a prior snapshot NEVER mutates any typed array inside `prevState`. Mutating an
 *     exposed typed array in place violates this contract, invalidates `stateDigest`
 *     (rejected by `validateDecodeCacheState()`), and is isolated from lazy trace
 *     reconstruction via private cloned boundary snapshots.
 *  2. **`Map` Containers (`gqaCaches`, `mambaStates`, session `branches`)**:
 *     Exposed as `ReadonlyMap` on frozen snapshots and sessions. Because JavaScript's
 *     `Object.freeze()` does not prevent `.set()` / `.delete()` / `.clear()` calls on a
 *     `Map` instance at runtime, callers MUST treat exposed `Map` containers as strictly
 *     read-only. Internal lazy-reconstruction closures capture private non-exposed `Map`
 *     snapshots so runtime container mutation cannot corrupt trace reconstruction.
 *  3. **Residual Checkpoints (`residualStreams`, `X_embed`)**:
 *     Each step exposes a frozen `residualStreams` array whose `Float32Array` buffers are
 *     detached (`.slice()`) from both `eagerTrace.sim.Y` and internal lazy-reconstruction
 *     checkpoints. Mutating returned `residualStreams`, `X_embed`, or eager trace outputs
 *     cannot corrupt subsequent lazy `getLayerTrace(bayId)` reconstruction.
 *  4. **Shared Parameter Buffers (`HybridModelParameters`)**:
 *     `W_embed`, `finalNormGamma`, `W_lm_head`, and per-bay weights returned by
 *     `getBayWeights(bayId)` (retained in a bounded LRU cache defaulting to all 52 bays,
 *     ~1.14 MiB) are owned by `HybridModelParameters` and shared by read-only reference
 *     across executions, decode steps, and layer traces. Callers MUST NEVER mutate weight
 *     buffers in place.
 *  5. **External Boundary Seeds (`createExternalDecodeCacheState`)**:
 *     External typed arrays are defensively cloned and validated (coverage, dimensions,
 *     explicit capacity bounds `maxTokens >= processedTokens`, unambiguous K/V stride
 *     matching, finite populated values, `parameterFingerprint`, and `stateDigest`) so
 *     subsequent caller mutations cannot corrupt any snapshot.
 *
 * Exact Default Miniature Snapshot Accounting (`L = 52`, `23 M`, `6 *`, `23 E`, `maxTokens = 16`):
 *  - Mamba-2 fixed state: `23 × (128×4 + 16×4×4) × 4 B = 23 × 3,072 B = 70,656 B = 69.00 KiB`
 *  - GQA KV allocated:    `6 × 2 × (2×4×16) × 4 B     = 6 × 1,024 B  =  6,144 B =  6.00 KiB`
 *  - GQA KV populated:    `6 × 2 × (2×4×seqLen) × 4 B = 384 × seqLen B` (`0.375 KiB / token`)
 *  - MoE sequence cache:  `0 B`
 *  - Total allocated default snapshot (excluding JS objects & educational traces):
 *    `76,800 B = 75.00 KiB` exactly.
 */

import type { PrecisionMode } from '../model/spec';

/** Distinguishes eagerly computed step traces from lazily reconstructed historical traces. */
export type IncrementalTraceOrigin = 'computed_this_step' | 'reconstructed_from_checkpoint';

/** Distinguishes prefill, multi-token chunk continuation, and single-token AR decode steps. */
export type IncrementalDecodeStepKind = 'prefill' | 'chunk' | 'decode';

/** Per-layer GQA Attention KV cache contract (grows linearly with `seqLen`). */
export interface GqaLayerKvCache {
  readonly bayId: number;
  readonly kind: '*';
  readonly nKv: number;
  readonly dHead: number;
  /** Number of cached prefix tokens currently populated (`0..maxTokens`). */
  readonly seqLen: number;
  /** Allocated capacity in tokens along the sequence axis. */
  readonly maxTokens: number;
  /**
   * Owned, immutable cached key projections, indexed `[(g * dHead + r) * maxTokens + t]`
   * (Float32 miniature storage; length strictly `nKv * dHead * maxTokens`).
   */
  readonly K: Float32Array;
  /**
   * Owned, immutable cached value projections, indexed `[(g * dHead + r) * maxTokens + t]`
   * (Float32 miniature storage; length strictly `nKv * dHead * maxTokens`).
   */
  readonly V: Float32Array;
  /** Production precision metadata (`fp8` under FP8 Selective even beside BF16 attention bodies). */
  readonly storageDtypeMetadata: 'bf16' | 'fp8';
}

/** Per-layer Mamba-2 boundary state contract (strictly fixed size independent of sequence length). */
export interface MambaLayerBoundaryState {
  readonly bayId: number;
  readonly kind: 'M';
  readonly convDim: number;
  readonly convKernel: number;
  /**
   * Owned, immutable rolling causal Conv1D input tap buffer, shape `[convDim, convKernel]`
   * (slot `k-1` = newest prior token; length strictly `convDim * convKernel`).
   */
  readonly convState: Float32Array;
  readonly nHeads: number;
  readonly dHead: number;
  readonly dState: number;
  /**
   * Owned, immutable persistent recurrent SSM hidden state `H_t`, shape `[nHeads, dHead, dState]`
   * (always FP32; length strictly `nHeads * dHead * dState`).
   */
  readonly ssmState: Float32Array;
  /** Number of tokens folded into this fixed-size state so far. */
  readonly processedTokens: number;
  /** Production storage metadata (NVIDIA §4.2: Conv1D BF16, SSM FP32). */
  readonly storageDtypeMetadata: {
    readonly conv1d: 'bf16';
    readonly ssm: 'fp32';
  };
}

/** Explicit stateless marker for Granular MoE layers (`0 B` cross-token sequence cache). */
export interface MoeLayerStatelessContract {
  readonly bayId: number;
  readonly kind: 'E';
  readonly cacheBytes: 0;
}

/** Exact typed-array byte breakdown of a `HybridDecodeCacheState` snapshot (excluding JS overhead & traces). */
export interface DecodeCacheMemoryAccounting {
  readonly mambaLayerCount: number;
  readonly gqaLayerCount: number;
  readonly moeLayerCount: number;
  readonly processedTokens: number;
  readonly maxTokens: number;
  /** Fixed Mamba-2 typed-array bytes (`convState` + `ssmState`, independent of `processedTokens`). */
  readonly mambaFixedBytes: number;
  /** Allocated GQA KV typed-array bytes (`K` + `V` at full `maxTokens` capacity). */
  readonly gqaAllocatedBytes: number;
  /** Populated GQA KV typed-array bytes (`K` + `V` across `0..processedTokens-1`). */
  readonly gqaPopulatedBytes: number;
  /** Linear byte growth per newly appended token across all GQA attention bays. */
  readonly bytesPerTokenGqaPopulated: number;
  /** Granular MoE sequence cache bytes (strictly 0). */
  readonly moeBytes: 0;
  /** `mambaFixedBytes + gqaAllocatedBytes` (= 76,800 B = 75.00 KiB for default 52-layer miniature snapshot). */
  readonly totalAllocatedBytes: number;
  /** `mambaFixedBytes + gqaPopulatedBytes` (active populated bytes at `processedTokens`). */
  readonly totalPopulatedBytes: number;
}

/**
 * Full hybrid model incremental decode cache snapshot spanning all layers.
 * Keyed by BOTH `modelId` and canonical `parameterFingerprint` so snapshots from
 * differently parameterized models (even with duplicated display IDs) can never collide.
 */
export interface HybridDecodeCacheState {
  readonly modelId: string;
  /** Canonical 64-bit model parameter fingerprint binding this snapshot to exact weights & pattern. */
  readonly parameterFingerprint: string;
  /** Deterministic 64-bit hex digest over fingerprint, processedTokens, prefixTokenIds, prefixTokenLabels, and all populated state bytes. */
  readonly stateDigest: string;
  readonly precisionMetadata: PrecisionMode;
  /** Total prefix tokens already absorbed into `gqaCaches` and `mambaStates`. */
  readonly processedTokens: number;
  /** Allocated KV capacity in tokens across every GQA bay in this snapshot. */
  readonly maxTokens: number;
  /** Prefix token IDs already absorbed (`length === processedTokens`; `-1` for synthetic external seeds). */
  readonly prefixTokenIds: Int32Array;
  /** Display labels aligned with `prefixTokenIds` (`length === processedTokens`). */
  readonly prefixTokenLabels: readonly string[];
  /** Keyed by 1-based attention `bayId` (`06, 13, 20, 27, 34, 43` in Nemotron 3 Nano). */
  readonly gqaCaches: ReadonlyMap<number, GqaLayerKvCache>;
  /** Keyed by 1-based Mamba-2 `bayId` (23 layers in Nemotron 3 Nano). */
  readonly mambaStates: ReadonlyMap<number, MambaLayerBoundaryState>;
  /** 1-based MoE `bayId`s (explicitly stateless; 0 B sequence cache). */
  readonly moeLayerIds: readonly number[];
}

/** External boundary-state seed for initializing a validated, defensively cloned `HybridDecodeCacheState`. */
export interface ExternalDecodeCacheSeed {
  readonly processedTokens: number;
  readonly maxTokens?: number;
  readonly precisionMetadata?: PrecisionMode;
  readonly prefixTokenIds?: readonly number[] | Int32Array;
  readonly prefixTokenLabels?: readonly string[];
  readonly gqaCaches: ReadonlyMap<
    number,
    {
      readonly K: Float32Array;
      readonly V: Float32Array;
      readonly seqLen?: number;
      readonly maxTokens?: number;
    }
  >;
  readonly mambaStates: ReadonlyMap<
    number,
    {
      readonly convState: Float32Array;
      readonly ssmState: Float32Array;
      readonly processedTokens?: number;
    }
  >;
}

/** Provenance metadata distinguishing root seed, branch lineage, parent step, step execution, and absolute token coordinates. */
export interface IncrementalStepProvenance {
  /** Digest of the root initial boundary state at step 0 (zero state or external seed). */
  readonly initialStateDigest: string;
  /** Digest of `prevState` consumed by this step. */
  readonly prevStateDigest: string;
  /** Digest of `nextState` produced by this step. */
  readonly nextStateDigest: string;
  /** Session / continuation branch identity (distinguishes diverging continuations from a shared prefix). */
  readonly branchId: string;
  /** `stepExecutionId` of the immediate predecessor step (`null` for root step). */
  readonly parentStepExecutionId: string | null;
  /** 0-based step index along this branch. */
  readonly stepIndex: number;
  readonly stepKind: IncrementalDecodeStepKind;
  /** Inclusive absolute token position range `[startPos, endPos]` = `[P, P + S - 1]`. */
  readonly tokenRange: readonly [number, number];
  /** Unambiguous deterministic identity for this exact step execution. */
  readonly stepExecutionId: string;
}

/** Input contract for a single-step or chunked cached incremental AR decode step. */
export interface IncrementalDecodeStepInput {
  /** New token IDs appended after `prevState.processedTokens` (`length === 1` for AR decode, `>= 1` for prefill/chunk). */
  readonly newTokenIds: readonly number[];
  readonly newTokenLabels?: readonly string[];
  /** Prior boundary state (`null` starts from zeroed Mamba states and empty KV caches). */
  readonly prevState: HybridDecodeCacheState | null;
  /** Optional KV capacity when `prevState === null` (defaults to `16`). */
  readonly maxTokens?: number;
  /** Production precision metadata tag when `prevState === null` (defaults to `'fp8_selective'`). */
  readonly precisionMetadata?: PrecisionMode;
  /** Optional explicit branch identity (defaults to `'branch-0'`). */
  readonly branchId?: string;
  /** Optional explicit parent step execution ID (`null` for root prefill/seed step). */
  readonly parentStepExecutionId?: string | null;
  /** Optional root initial state digest propagated across a multi-step branch. */
  readonly initialStateDigest?: string;
  /** Optional 0-based step index along the branch (defaults to `0` if `prevState === null`, else `1`). */
  readonly stepIndex?: number;
  /** Override which bay IDs retain full simulator traces eagerly (`traceOrigin: 'computed_this_step'`). */
  readonly retainedTraceBays?: readonly number[];
  /** Maximum number of full layer traces retained inside this step result (bounded LRU). */
  readonly maxRetainedTraces?: number;
}

/** Output contract for a cached incremental decode step (matches full-sequence logits bit-for-bit). */
export interface IncrementalDecodeStepResult {
  readonly modelId: string;
  readonly parameterFingerprint: string;
  /** Alias matching `stepExecutionId` for backwards compatibility with execution identity consumers. */
  readonly executionId: string;
  readonly stepExecutionId: string;
  readonly provenance: IncrementalStepProvenance;
  readonly prevState: HybridDecodeCacheState;
  readonly nextState: HybridDecodeCacheState;
  /** Next-token logits for the newly processed positions, shape `[vocabSize, newTokenIds.length]`. */
  readonly logits: Float32Array;
  /** Softmax probabilities for the newly processed positions, shape `[vocabSize, newTokenIds.length]`. */
  readonly probs: Float32Array;
}
