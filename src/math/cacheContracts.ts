/**
 * Forward-Compatible Incremental Decode Cache & Boundary-State Contracts
 * (Design Contracts ONLY — NOT executed in the current full-sequence unified forward pass).
 *
 * A subsequent Nemotron 3 Nano increment will implement cached incremental autoregressive (AR)
 * execution, externally supplied initial states, and full-sequence vs. incremental numerical
 * equivalence verification against these contracts.
 *
 * Architectural boundary across Nemotron 3 Nano's 3 layer kinds (and extensible to future
 * Nemotron Labs hybrid models):
 *  - GQA Attention (`*`, 6 bays): linear per-token K/V cache growth (`seqLen` entries per KV head)
 *  - Mamba-2 (`M`, 23 bays): strictly fixed-size rolling Conv1D buffer + FP32 persistent SSM state
 *  - Granular MoE (`E`, 23 bays): strictly stateless across tokens (`0 B` sequence cache)
 */

import type { PrecisionMode } from '../model/spec';

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
  /** Cached key projections, indexed `[(g * dHead + r) * maxTokens + t]` (Float32 miniature storage). */
  readonly K: Float32Array;
  /** Cached value projections, indexed `[(g * dHead + r) * maxTokens + t]` (Float32 miniature storage). */
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
  /** Rolling causal Conv1D input tap buffer, shape `[convDim, convKernel]` (slot `k-1` = newest). */
  readonly convState: Float32Array;
  readonly nHeads: number;
  readonly dHead: number;
  readonly dState: number;
  /** Persistent recurrent SSM hidden state `H_t`, shape `[nHeads, dHead, dState]` (always FP32). */
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

/**
 * Full hybrid model incremental decode cache snapshot spanning all 52 layers.
 * Keyed by `modelId` so caches from distinct parameter instances can never cross-contaminate.
 */
export interface HybridDecodeCacheState {
  readonly modelId: string;
  readonly precisionMetadata: PrecisionMode;
  /** Total prefix tokens already absorbed into `gqaCaches` and `mambaStates`. */
  readonly processedTokens: number;
  /** Keyed by 1-based attention `bayId` (`06, 13, 20, 27, 34, 43` in Nemotron 3 Nano). */
  readonly gqaCaches: ReadonlyMap<number, GqaLayerKvCache>;
  /** Keyed by 1-based Mamba-2 `bayId` (23 layers in Nemotron 3 Nano). */
  readonly mambaStates: ReadonlyMap<number, MambaLayerBoundaryState>;
  /** 1-based MoE `bayId`s (explicitly stateless; 0 B sequence cache). */
  readonly moeLayerIds: readonly number[];
}

/** Input contract for a future single-step or chunked cached incremental AR decode step. */
export interface IncrementalDecodeStepInput {
  /** New token IDs appended after `prevState.processedTokens` (length `1` for AR step, `>= 1` for chunked prefill). */
  readonly newTokenIds: readonly number[];
  readonly newTokenLabels?: readonly string[];
  /** Prior boundary state (`null` starts from zeroed Mamba states and empty KV caches). */
  readonly prevState: HybridDecodeCacheState | null;
}

/** Output contract for a future cached incremental decode step (must match full-sequence logits). */
export interface IncrementalDecodeStepResult {
  readonly modelId: string;
  readonly executionId: string;
  readonly nextState: HybridDecodeCacheState;
  /** Next-token logits for the newly processed positions, shape `[vocabSize, newTokenIds.length]`. */
  readonly logits: Float32Array;
  /** Softmax probabilities for the newly processed positions, shape `[vocabSize, newTokenIds.length]`. */
  readonly probs: Float32Array;
}
