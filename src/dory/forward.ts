/**
 * Physical 72-Sublayer Depth-Looped Forward Pass (`src/dory/forward.ts`)
 *
 * Implements canonical Physical-72 (`16` Input Prelude + `40` Shared Recurrent Core looped `k`
 * times + `16` Output Coda; effective depth `32 + 40k`) execution for Nemotron-Labs-Dory1.0-4B:
 *   - Stage-modular entry points (`runDoryInputPrelude`, `runDoryRecurrentPass`,
 *     `runDoryOutputCoda`, `runDoryLmHead`) supporting both standard full chunks
 *     (`runDoryForwardChunk`) and Contract A §6.2 self-speculative activation/KV reuse (`DORY2`).
 *   - Literal `DR8` loop boundaries: `u_t = N_d(s_{t-1} + x)` (including literal `u_1 = N_d(2x)`)
 *     and explicit post-loop `s_t = N_d(F_theta(u_t))` with zero outer skip into Output Coda.
 *   - `DR5`: Raw unit-norm synthetic embeddings enter physical Layer 0 directly (no pre-norm).
 *   - `DR17`: In `last_loop`, non-final loops (`t < k`) read committed historical KV without
 *     advancing ring pointers (`writeKv = false`); only final loop `t = k` commits (`writeKv = true`).
 *   - `DR18`: Rejects mismatched `kLoops` or `kvCacheMode` on an existing `DoryKvCache`.
 *   - `DR24`: Records exact forward `chunkSizes[]` on committed chunks for `last_loop` replay.
 *   - Emits the full immutable 8-tuple `DoryExecutionCoord` on every traced sublayer record.
 *
 * 100% clean-room original TypeScript — zero ambient RNG, zero DOM/WebGL/Node/sibling imports.
 */

import { runDoryAttentionSublayer } from './attention';
import {
  createDoryKvCache,
  getDoryInputBank,
  getDoryOutputBank,
  getDoryRecurrentBank,
} from './kvCache';
import {
  anchorMapJacobianNorm,
  cosineSimilarity8d,
  effectiveLogitScale,
  l2Distance,
  linearProject,
  loopEntryAnchorInject,
  loopExitNormalize,
  softmaxFloat64,
  vectorL2Norm,
} from './ngptMath';
import { buildDoryDualRopeTables } from './rope';
import {
  computeExecutedDepth,
  computeExecutedLayerIdx,
  validateDoryRecurrentLoops,
} from './spec';
import { runDoryMlpSublayer } from './swiglu';
import type {
  DoryAttentionSublayerTrace,
  DoryCacheRole,
  DoryCodaBankKind,
  DoryExecutedSublayerRecord,
  DoryExecutionCoord,
  DoryForwardChunkInput,
  DoryForwardChunkResult,
  DoryForwardTrace,
  DoryKvBankKind,
  DoryKvCache,
  DoryLmHeadResult,
  DoryLoopTrajectoryMetrics,
  DoryModelParameters,
  DoryStageCodaInput,
  DoryStageCodaResult,
  DoryStagePreludeInput,
  DoryStagePreludeResult,
  DoryStageRecurrentPassInput,
  DoryStageRecurrentPassResult,
  DorySublayerTrace,
} from './types';

function buildSublayerCoord(
  params: DoryModelParameters,
  kvCache: DoryKvCache,
  physicalLayerIdx: number,
  executedLayerIdx: number,
  loopIteration: number,
  absTokenPos: number,
  forwardStepId: number,
  sessionBranchId: string,
  bankKind: DoryKvBankKind,
  bankSlotId: number,
  writerLoop: number,
  role: DoryCacheRole,
): DoryExecutionCoord {
  return {
    modelWeightId: params.modelWeightId,
    physicalLayerIdx,
    executedLayerIdx,
    loopIteration,
    absTokenPos,
    forwardStepId,
    sessionBranchId,
    cacheOwnership: {
      policy: kvCache.mode,
      bankKind,
      bankSlotId,
      writerLoop,
      role,
    },
  };
}

function resolveAttnWriterLoop(
  trace: DoryAttentionSublayerTrace,
  loopIteration: number,
): number {
  if (trace.stage !== 'recurrent') {
    return 0;
  }
  if (trace.pastLen > 0) {
    return trace.kvWriterLoops[0];
  }
  return loopIteration;
}

function pushTracedRecord(
  records: DoryExecutedSublayerRecord[],
  params: DoryModelParameters,
  kvCache: DoryKvCache,
  physicalLayerIdx: number,
  loopIteration: number,
  kLoops: number,
  absTokenPos: number,
  forwardStepId: number,
  sessionBranchId: string,
  bankKind: DoryKvBankKind,
  role: DoryCacheRole,
  trace: DorySublayerTrace,
): void {
  const executedLayerIdx = computeExecutedLayerIdx(physicalLayerIdx, loopIteration, kLoops);
  const bankSlotId = trace.kind === 'attention' ? trace.bankSlotId : -1;
  const writerLoop =
    trace.kind === 'attention'
      ? resolveAttnWriterLoop(trace, loopIteration)
      : trace.stage === 'recurrent'
        ? loopIteration
        : 0;
  records.push({
    executedLayerIdx,
    coord: buildSublayerCoord(
      params,
      kvCache,
      physicalLayerIdx,
      executedLayerIdx,
      loopIteration,
      absTokenPos,
      forwardStepId,
      sessionBranchId,
      bankKind,
      bankSlotId,
      writerLoop,
      role,
    ),
    trace,
  });
}

/**
 * Executes Stage 1 — Input Prelude (`backbone.embeddings` + physical sublayers `0..15`).
 * Raw embeddings enter Layer 0 directly without pre-normalization (`DR5`).
 */
export function runDoryInputPrelude(input: DoryStagePreludeInput): DoryStagePreludeResult {
  const { tokens, startPos, params, kvCache } = input;
  const config = params.config;
  const seqLen = tokens.length;

  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new RangeError(`tokens must be non-empty; got length ${seqLen}.`);
  }
  if (!Number.isInteger(startPos) || startPos < 0) {
    throw new RangeError(`startPos must be an integer >= 0; got ${startPos}.`);
  }
  if (startPos + seqLen > config.maxPositionEmbeddings) {
    throw new RangeError(
      `Prelude position overflow: startPos(${startPos}) + seqLen(${seqLen}) > maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
    );
  }

  const d = config.hiddenSize;
  const vocabSize = config.vocabSize;
  const embeddings = new Float32Array(seqLen * d);

  for (let t = 0; t < seqLen; t += 1) {
    const tok = tokens[t];
    if (!Number.isInteger(tok) || tok < 0 || tok >= vocabSize) {
      throw new RangeError(
        `Invalid token ID ${tok} at chunk offset ${t} (expected integer in 0..${vocabSize - 1}).`,
      );
    }
    const srcOff = tok * d;
    const dstOff = t * d;
    for (let c = 0; c < d; c += 1) {
      embeddings[dstOff + c] = params.embeddings[srcOff + c];
    }
  }

  const ropeTables = input.ropeTables ?? buildDoryDualRopeTables(config);
  const writeKv = input.writeKv ?? true;
  const captureTrace = input.captureTrace ?? false;
  const forwardStepId = input.forwardStepId ?? 0;
  const sessionBranchId = input.sessionBranchId ?? 'b0';
  const role: DoryCacheRole = input.role ?? 'standard';
  const kLoopsForCoord = input.kLoopsForCoord ?? kvCache.kLoops;
  const headAbsTokenPos = startPos + seqLen - 1;

  const executedSublayers: DoryExecutedSublayerRecord[] = [];
  let h: Float32Array = embeddings;

  for (let p = 0; p < config.nInputLayers; p += 1) {
    const layer = params.layers[p];
    if (layer.kind === 'attention') {
      const kvBank = getDoryInputBank(kvCache, p);
      const res = runDoryAttentionSublayer({
        hiddenStates: h,
        seqLen,
        startPos,
        physicalLayerIdx: p,
        loopIteration: 0,
        weights: layer,
        ropeTables,
        kvBank,
        writeKv,
        config,
        captureTrace,
      });
      h = res.output;
      if (captureTrace && res.trace !== null) {
        pushTracedRecord(
          executedSublayers,
          params,
          kvCache,
          p,
          0,
          kLoopsForCoord,
          headAbsTokenPos,
          forwardStepId,
          sessionBranchId,
          'input',
          role,
          res.trace,
        );
      }
    } else {
      const res = runDoryMlpSublayer({
        hiddenStates: h,
        seqLen,
        physicalLayerIdx: p,
        loopIteration: 0,
        weights: layer,
        config,
        captureTrace,
      });
      h = res.output;
      if (captureTrace && res.trace !== null) {
        pushTracedRecord(
          executedSublayers,
          params,
          kvCache,
          p,
          0,
          kLoopsForCoord,
          headAbsTokenPos,
          forwardStepId,
          sessionBranchId,
          'input',
          role,
          res.trace,
        );
      }
    }
  }

  return {
    embeddings,
    xAnchor: h,
    executedSublayers,
  };
}

/**
 * Executes one pass `t in 1..k` of Stage 2 — Shared Recurrent Core (`16..55`):
 *   - Literal entry anchor injection `u_t = N_d(prevState + xAnchor)` (incl. `u_1 = N_d(2x)`, `DR8`).
 *   - Executes identical shared physical weights `params.layers[16..55]`.
 *   - Literal post-loop normalization `s_t = N_d(F_theta(u_t))` (`DR8`).
 *   - Defaults `writeKv` to `kvCache.mode === 'per_loop' || loopIteration === kLoops` (`DR17`).
 */
export function runDoryRecurrentPass(
  input: DoryStageRecurrentPassInput,
): DoryStageRecurrentPassResult {
  const {
    prevState,
    xAnchor,
    seqLen,
    startPos,
    loopIteration,
    kLoops,
    params,
    kvCache,
  } = input;
  const config = params.config;
  const d = config.hiddenSize;

  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new RangeError(`seqLen must be an integer >= 1; got ${seqLen}.`);
  }
  if (!Number.isInteger(startPos) || startPos < 0) {
    throw new RangeError(`startPos must be an integer >= 0; got ${startPos}.`);
  }
  if (startPos + seqLen > config.maxPositionEmbeddings) {
    throw new RangeError(
      `Recurrent pass position overflow: startPos(${startPos}) + seqLen(${seqLen}) > maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
    );
  }
  if (!Number.isInteger(loopIteration) || loopIteration < 1 || loopIteration > kLoops) {
    throw new RangeError(
      `Invalid loopIteration=${loopIteration} for kLoops=${kLoops}.`,
    );
  }
  if (prevState.length !== seqLen * d || xAnchor.length !== seqLen * d) {
    throw new RangeError(
      `prevState/xAnchor length mismatch: got prev=${prevState.length}, anchor=${xAnchor.length}, expected ${seqLen * d}.`,
    );
  }

  const ropeTables = input.ropeTables ?? buildDoryDualRopeTables(config);
  const writeKv = input.writeKv ?? (kvCache.mode === 'per_loop' || loopIteration === kLoops);
  const captureTrace = input.captureTrace ?? false;
  const forwardStepId = input.forwardStepId ?? 0;
  const sessionBranchId = input.sessionBranchId ?? 'b0';
  const role: DoryCacheRole = input.role ?? 'standard';
  const headAbsTokenPos = startPos + seqLen - 1;

  // Literal u_t = N_d(s_{t-1} + x), including u_1 = N_d(2x) (DR8)
  const uEntry = loopEntryAnchorInject(prevState, xAnchor, seqLen, d, config.l2NormEps);

  const executedSublayers: DoryExecutedSublayerRecord[] = [];
  let h: Float32Array = uEntry;

  const recStart = config.nInputLayers;
  const recEnd = recStart + config.nRecurrentLayers;

  for (let p = recStart; p < recEnd; p += 1) {
    const layer = params.layers[p];
    if (layer.kind === 'attention') {
      const kvBank = getDoryRecurrentBank(kvCache, p, loopIteration);
      const res = runDoryAttentionSublayer({
        hiddenStates: h,
        seqLen,
        startPos,
        physicalLayerIdx: p,
        loopIteration,
        weights: layer,
        ropeTables,
        kvBank,
        writeKv,
        config,
        captureTrace,
      });
      h = res.output;
      if (captureTrace && res.trace !== null) {
        pushTracedRecord(
          executedSublayers,
          params,
          kvCache,
          p,
          loopIteration,
          kLoops,
          headAbsTokenPos,
          forwardStepId,
          sessionBranchId,
          'recurrent',
          role,
          res.trace,
        );
      }
    } else {
      const res = runDoryMlpSublayer({
        hiddenStates: h,
        seqLen,
        physicalLayerIdx: p,
        loopIteration,
        weights: layer,
        config,
        captureTrace,
      });
      h = res.output;
      if (captureTrace && res.trace !== null) {
        pushTracedRecord(
          executedSublayers,
          params,
          kvCache,
          p,
          loopIteration,
          kLoops,
          headAbsTokenPos,
          forwardStepId,
          sessionBranchId,
          'recurrent',
          role,
          res.trace,
        );
      }
    }
  }

  // Literal post-loop normalization s_t = N_d(F_theta(u_t)) (DR8)
  const sExit = loopExitNormalize(h, seqLen, d, config.l2NormEps);

  // Exact 8D loop trajectory metrics & local anchor-map Jacobian (Finding 9)
  const loopMetrics: DoryLoopTrajectoryMetrics[] = new Array<DoryLoopTrajectoryMetrics>(seqLen);
  for (let tokenOffset = 0; tokenOffset < seqLen; tokenOffset += 1) {
    const off = tokenOffset * d;
    const prevAnchorCos = cosineSimilarity8d(prevState, xAnchor, off, off, d);
    const clampedCos = Math.max(-1.0, Math.min(1.0, prevAnchorCos));
    const entryAngleRadians = Math.acos(clampedCos);

    loopMetrics[tokenOffset] = {
      loopIteration,
      tokenOffset,
      absTokenPos: startPos + tokenOffset,
      uNorm: vectorL2Norm(uEntry, off, d),
      sNorm: vectorL2Norm(sExit, off, d),
      consecutiveCosine: cosineSimilarity8d(sExit, prevState, off, off, d),
      consecutiveDisplacement: l2Distance(sExit, prevState, off, off, d),
      anchorCosine: cosineSimilarity8d(sExit, xAnchor, off, off, d),
      entryAnchorJacobian: anchorMapJacobianNorm(entryAngleRadians, config.l2NormEps),
    };
  }

  return {
    uEntry,
    sExit,
    executedSublayers,
    loopMetrics,
  };
}

/**
 * Executes the untied LM Head (`lm_head.weight` + FP32 `lm_head.logit_scale / sigma_0`)
 * and computes binary64 `lastTokenProbs` at `temperature = 1.0`.
 */
export function runDoryLmHead(
  hiddenStates: Float32Array,
  seqLen: number,
  params: DoryModelParameters,
): DoryLmHeadResult {
  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new RangeError(`seqLen must be an integer >= 1; got ${seqLen}.`);
  }
  const config = params.config;
  const d = config.hiddenSize;
  const vocabSize = config.vocabSize;

  if (hiddenStates.length !== seqLen * d) {
    throw new RangeError(
      `LM head hiddenStates length mismatch: got ${hiddenStates.length}, expected ${seqLen * d}.`,
    );
  }

  // Stage 1: Raw logits z_raw = h_out W_lm^T [seqLen, V]
  const rawLogits = linearProject(hiddenStates, params.lmHead, seqLen, d, vocabSize);

  // Stage 2: Effective logit scale s_z / sigma_0 [V]
  const effLogitScale = effectiveLogitScale(params.logitScale, config);

  // Stage 3: Scaled logits z_scaled = z_raw * (s_z / sigma_0) [seqLen, V]
  const scaledLogits = new Float32Array(seqLen * vocabSize);
  for (let t = 0; t < seqLen; t += 1) {
    const rowOff = t * vocabSize;
    for (let v = 0; v < vocabSize; v += 1) {
      scaledLogits[rowOff + v] = rawLogits[rowOff + v] * effLogitScale[v];
    }
  }

  // Stage 4: Binary64 softmax probabilities of final token in chunk
  const lastTokenProbs = softmaxFloat64(scaledLogits, (seqLen - 1) * vocabSize, vocabSize, 1.0);

  return {
    rawLogits,
    scaledLogits,
    effectiveLogitScale: effLogitScale,
    lastTokenProbs,
  };
}

/**
 * Executes Stage 3 — Output Coda (`56..71`, NO outer skip) + untied LM Head.
 * Supports `codaKind = 'output_deep' | 'output_shallow'` (Contract A `D3`) and
 * `writeKv = false` for read-only View 2 Diagnostic Counterfactual logit-lens evaluation.
 */
export function runDoryOutputCoda(input: DoryStageCodaInput): DoryStageCodaResult {
  const { state, seqLen, startPos, kLoops, params, kvCache } = input;
  validateDoryRecurrentLoops(kLoops);
  const config = params.config;
  const d = config.hiddenSize;

  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new RangeError(`seqLen must be an integer >= 1; got ${seqLen}.`);
  }
  if (!Number.isInteger(startPos) || startPos < 0) {
    throw new RangeError(`startPos must be an integer >= 0; got ${startPos}.`);
  }
  if (startPos + seqLen > config.maxPositionEmbeddings) {
    throw new RangeError(
      `Output Coda position overflow: startPos(${startPos}) + seqLen(${seqLen}) > maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
    );
  }
  if (state.length !== seqLen * d) {
    throw new RangeError(
      `Output Coda state length mismatch: got ${state.length}, expected ${seqLen * d}.`,
    );
  }

  const ropeTables = input.ropeTables ?? buildDoryDualRopeTables(config);
  const codaKind: DoryCodaBankKind = input.codaKind ?? 'output_deep';
  const writeKv = input.writeKv ?? true;
  const captureTrace = input.captureTrace ?? false;
  const forwardStepId = input.forwardStepId ?? 0;
  const sessionBranchId = input.sessionBranchId ?? 'b0';
  const role: DoryCacheRole = input.role ?? 'standard';
  const headAbsTokenPos = startPos + seqLen - 1;

  const executedSublayers: DoryExecutedSublayerRecord[] = [];
  let h: Float32Array = state;

  const codaStart = config.nInputLayers + config.nRecurrentLayers;
  for (let p = codaStart; p < config.numHiddenLayers; p += 1) {
    const layer = params.layers[p];
    if (layer.kind === 'attention') {
      const kvBank = getDoryOutputBank(kvCache, p, codaKind);
      const res = runDoryAttentionSublayer({
        hiddenStates: h,
        seqLen,
        startPos,
        physicalLayerIdx: p,
        loopIteration: 0,
        weights: layer,
        ropeTables,
        kvBank,
        writeKv,
        config,
        captureTrace,
      });
      h = res.output;
      if (captureTrace && res.trace !== null) {
        pushTracedRecord(
          executedSublayers,
          params,
          kvCache,
          p,
          0,
          kLoops,
          headAbsTokenPos,
          forwardStepId,
          sessionBranchId,
          codaKind,
          role,
          res.trace,
        );
      }
    } else {
      const res = runDoryMlpSublayer({
        hiddenStates: h,
        seqLen,
        physicalLayerIdx: p,
        loopIteration: 0,
        weights: layer,
        config,
        captureTrace,
      });
      h = res.output;
      if (captureTrace && res.trace !== null) {
        pushTracedRecord(
          executedSublayers,
          params,
          kvCache,
          p,
          0,
          kLoops,
          headAbsTokenPos,
          forwardStepId,
          sessionBranchId,
          codaKind,
          role,
          res.trace,
        );
      }
    }
  }

  const lmHead = runDoryLmHead(h, seqLen, params);

  return {
    codaHidden: h,
    lmHead,
    executedSublayers,
  };
}

/**
 * Executes one complete physical-72 (`32 + 40k` executed sublayers) forward chunk through:
 *   Input Prelude (`0..15`) -> `kLoops` passes of Shared Recurrent Core (`16..55`) ->
 *   Output Coda (`56..71`) -> Untied LM Head.
 *
 * Enforces `DR18` (strict `kLoops`/`kvCacheMode` match on existing caches), `DR17` (`last_loop`
 * read-only non-final passes), `DR24` (long-chunk SWA workspace + `chunkSizes[]` recording),
 * and `DR8` (literal `u_1 = N(2x)` and post-loop `s_t = N(F_theta(u_t))`).
 */
export function runDoryForwardChunk(input: DoryForwardChunkInput): DoryForwardChunkResult {
  const { params } = input;
  const config = params.config;
  const seqLen = input.tokens.length;

  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new RangeError(`tokens must be non-empty; got length ${seqLen}.`);
  }

  // Validate and copy tokens atomically before any cache mutation
  const tokens = new Int32Array(seqLen);
  for (let t = 0; t < seqLen; t += 1) {
    const tok = input.tokens[t];
    if (!Number.isInteger(tok) || tok < 0 || tok >= config.vocabSize) {
      throw new RangeError(
        `Invalid token ID ${tok} at index ${t} (expected integer in 0..${config.vocabSize - 1}).`,
      );
    }
    tokens[t] = tok;
  }

  const codaKind: DoryCodaBankKind = input.codaKind ?? 'output_deep';
  const writeKv = input.writeKv ?? true;
  const captureTrace = input.captureTrace ?? false;
  const role: DoryCacheRole = input.role ?? 'standard';
  const sessionBranchId = input.sessionBranchId ?? 'b0';

  let kvCache: DoryKvCache;
  let kLoops: number;
  let kvCacheMode = input.kvCacheMode ?? config.defaultKvCacheMode;

  if (input.kvCache !== undefined) {
    kvCache = input.kvCache;
    kLoops = validateDoryRecurrentLoops(input.kLoops ?? kvCache.kLoops);
    kvCacheMode = input.kvCacheMode ?? kvCache.mode;
    if (kLoops !== kvCache.kLoops) {
      throw new Error(
        `DR18 mid-session depth mismatch: requested kLoops=${kLoops}, but kvCache was allocated for kLoops=${kvCache.kLoops}. Fork/re-prefill a clean cache instead of mutating a populated session.`,
      );
    }
    if (kvCacheMode !== kvCache.mode) {
      throw new Error(
        `DR18 mid-session KV policy mismatch: requested kvCacheMode='${kvCacheMode}', but kvCache uses '${kvCache.mode}'. Fork/re-prefill a clean cache instead of mutating a populated session.`,
      );
    }
  } else {
    kLoops = validateDoryRecurrentLoops(input.kLoops ?? config.defaultRecurrentLoops);
    kvCache = createDoryKvCache({
      mode: kvCacheMode,
      kLoops,
      includeShallowCoda: codaKind === 'output_shallow',
      config,
    });
  }

  if (codaKind === 'output_shallow' && kvCache.outputShallowBanks === null) {
    throw new Error(
      `Requested codaKind='output_shallow', but kvCache was created with includeShallowCoda=false.`,
    );
  }

  const startPos = input.startPos ?? kvCache.committedLen;
  if (!Number.isInteger(startPos) || startPos < 0) {
    throw new RangeError(`startPos must be an integer >= 0; got ${startPos}.`);
  }
  const endPos = startPos + seqLen;
  if (endPos > config.maxPositionEmbeddings) {
    throw new RangeError(
      `Forward chunk overflow: startPos(${startPos}) + seqLen(${seqLen}) > maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
    );
  }
  if (writeKv && startPos !== kvCache.committedLen) {
    throw new RangeError(
      `Non-contiguous forward chunk commit: startPos=${startPos} !== kvCache.committedLen=${kvCache.committedLen}.`,
    );
  }

  const ropeTables = input.ropeTables ?? buildDoryDualRopeTables(config);
  const forwardStepId = input.forwardStepId ?? kvCache.chunkSizes.length;

  // Stage 1: Input Prelude (physical 0..15)
  const prelude = runDoryInputPrelude({
    tokens,
    startPos,
    params,
    kvCache,
    ropeTables,
    writeKv,
    captureTrace,
    forwardStepId,
    sessionBranchId,
    role,
    kLoopsForCoord: kLoops,
  });

  // Stage 2: Shared Recurrent Core (physical 16..55, looped t = 1..kLoops)
  const loopEntryStates: Float32Array[] = new Array<Float32Array>(kLoops);
  const loopExitStates: Float32Array[] = new Array<Float32Array>(kLoops);
  const executedSublayers: DoryExecutedSublayerRecord[] = captureTrace
    ? [...prelude.executedSublayers]
    : [];
  const loopMetrics: DoryLoopTrajectoryMetrics[] = [];

  let state: Float32Array = prelude.xAnchor; // s_0 = x_anchor
  for (let t = 1; t <= kLoops; t += 1) {
    const passWriteKv = writeKv && (kvCacheMode === 'per_loop' || t === kLoops);
    const pass = runDoryRecurrentPass({
      prevState: state,
      xAnchor: prelude.xAnchor,
      seqLen,
      startPos,
      loopIteration: t,
      kLoops,
      params,
      kvCache,
      ropeTables,
      writeKv: passWriteKv,
      captureTrace,
      forwardStepId,
      sessionBranchId,
      role,
    });
    loopEntryStates[t - 1] = pass.uEntry;
    loopExitStates[t - 1] = pass.sExit;
    if (captureTrace) {
      for (let i = 0; i < pass.executedSublayers.length; i += 1) {
        executedSublayers.push(pass.executedSublayers[i]);
      }
      for (let i = 0; i < pass.loopMetrics.length; i += 1) {
        loopMetrics.push(pass.loopMetrics[i]);
      }
    }
    state = pass.sExit;
  }

  // Stage 3: Output Coda (physical 56..71, no outer skip) + LM Head
  const coda = runDoryOutputCoda({
    state,
    seqLen,
    startPos,
    kLoops,
    params,
    kvCache,
    ropeTables,
    codaKind,
    writeKv,
    captureTrace,
    forwardStepId,
    sessionBranchId,
    role,
  });
  if (captureTrace) {
    for (let i = 0; i < coda.executedSublayers.length; i += 1) {
      executedSublayers.push(coda.executedSublayers[i]);
    }
  }

  // Stage 4: Commit global sequence length & chunk-size schedule (DR24)
  if (writeKv) {
    kvCache.committedLen = endPos;
    kvCache.chunkSizes.push(seqLen);
  }

  const executedSublayerCount = computeExecutedDepth(kLoops, config);
  const trace: DoryForwardTrace | null = captureTrace
    ? {
        modelWeightId: params.modelWeightId,
        sessionBranchId,
        forwardStepId,
        kLoops,
        kvCacheMode,
        role,
        codaKind,
        seqLen,
        startPos,
        executedSublayerCount,
        executedSublayers,
        loopMetrics,
      }
    : null;

  return {
    tokens,
    seqLen,
    startPos,
    endPos,
    kLoops,
    kvCacheMode,
    executedSublayerCount,
    kvCache,
    embeddings: prelude.embeddings,
    xAnchor: prelude.xAnchor,
    loopEntryStates,
    loopExitStates,
    codaHidden: coda.codaHidden,
    rawLogits: coda.lmHead.rawLogits,
    scaledLogits: coda.lmHead.scaledLogits,
    lastTokenProbs: coda.lmHead.lastTokenProbs,
    trace,
  };
}
