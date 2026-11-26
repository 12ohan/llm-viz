/**
 * 40-Layer Homogeneous Dense Decoder Forward Pass (`src/diffusion/forward.ts`)
 *
 * Clean-room first-principles TypeScript execution of the single-backbone
 * 40-layer Ministral 3 / Nemotron-Labs-Diffusion miniature decoder:
 *   - Untied `encoder.embed_tokens` lookup (`[S, 8]`)
 *   - 40 layers of Pre-Norm GQA (YaRN RoPE + Llama-4 query scale + optional LoRA)
 *     + Residual + Pre-Norm Dense SwiGLU + Residual
 *   - 41 residual checkpoints (`[h_0, h_1, ..., h_40]`) for lazy reconstruction
 *   - Optional selective `retainedTraceLayers` for bounded session memory
 *   - Final RMSNorm -> untied `diffusion_head` -> raw un-suppressed `softmax`
 *   - Strictly non-mutating on `options.pastKvs`
 *
 * Pure headless TypeScript — deterministic, self-contained math kernel.
 */

import {
  applyDiffusionRmsNorm,
  linearProject,
  runDiffusionAttention,
  softmaxRows,
} from './attention';
import { computeYarnInvFreq } from './rope';
import { runDiffusionSwiglu } from './swiglu';
import type {
  DiffusionForwardOptions,
  DiffusionForwardResult,
  DiffusionLayerInput,
  DiffusionLayerKvSlice,
  DiffusionLayerTrace,
  DiffusionModelParameters,
} from './types';

/**
 * Gather untied token embeddings from `model.embedTokens` (`[vocabSize, d]`)
 * into a contiguous `[seqLen, d]` row-major Float32Array.
 */
export function embedDiffusionTokens(
  model: DiffusionModelParameters,
  tokenIds: ArrayLike<number>,
): Float32Array {
  const seqLen = tokenIds.length;
  const d = model.config.hiddenSize;
  const vocabSize = model.config.vocabSize;
  const out = new Float32Array(seqLen * d);

  for (let t = 0; t < seqLen; t++) {
    const id = tokenIds[t];
    if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
      throw new RangeError(
        `tokenId at index ${t} (${id}) is out of bounds [0, ${vocabSize - 1}]`,
      );
    }
    const srcBase = id * d;
    const dstBase = t * d;
    for (let i = 0; i < d; i++) {
      out[dstBase + i] = model.embedTokens[srcBase + i];
    }
  }

  return out;
}

/**
 * Run a single homogeneous dense Transformer decoder layer (`Ministral3DecoderLayer`)
 * and return its full layer trace plus the newly computed `[seqLen, 64]` KV slice.
 */
export function runDiffusionLayer(
  input: DiffusionLayerInput,
  layerIndex: number,
): { trace: DiffusionLayerTrace; newKv: DiffusionLayerKvSlice } {
  const {
    hiddenStates,
    seqLen,
    layerWeights,
    config,
    invFreq,
    attentionScaling,
    positionIds,
    cachePositions,
    maskMode,
    customMask,
    loraEnabled,
    useCache,
    pastKv,
  } = input;

  const d = config.hiddenSize;
  const eps = config.rmsNormEps;
  const totalFloats = seqLen * d;

  // 1. Pre-Attention RMSNorm
  const inputNorm = applyDiffusionRmsNorm(
    hiddenStates,
    seqLen,
    d,
    layerWeights.inputLayerNorm,
    eps,
  );

  // 2. 32Q/8KV GQA + YaRN RoPE + Llama-4 Query Scale + Optional Draft-Only LoRA
  const attention = runDiffusionAttention({
    hiddenStates: inputNorm,
    seqLen,
    layerWeights,
    config,
    invFreq,
    attentionScaling,
    positionIds,
    cachePositions,
    maskMode,
    customMask,
    loraEnabled,
    useCache,
    pastKv,
  });

  // 3. First Residual Add
  const postAttentionResidual = new Float32Array(totalFloats);
  for (let i = 0; i < totalFloats; i++) {
    postAttentionResidual[i] = Math.fround(
      hiddenStates[i] + attention.output[i],
    );
  }

  // 4. Pre-MLP RMSNorm
  const postAttentionNorm = applyDiffusionRmsNorm(
    postAttentionResidual,
    seqLen,
    d,
    layerWeights.postAttentionLayerNorm,
    eps,
  );

  // 5. Dense SwiGLU MLP
  const swiglu = runDiffusionSwiglu(
    postAttentionNorm,
    seqLen,
    layerWeights.mlp,
    config,
  );

  // 6. Second Residual Add
  const output = new Float32Array(totalFloats);
  for (let i = 0; i < totalFloats; i++) {
    output[i] = Math.fround(postAttentionResidual[i] + swiglu.output[i]);
  }

  const trace: DiffusionLayerTrace = {
    layerIndex,
    inputNorm,
    attention,
    postAttentionResidual,
    postAttentionNorm,
    swiglu,
    output,
  };

  const newKv: DiffusionLayerKvSlice = {
    k: attention.kRot,
    v: attention.vNew,
    seqLen,
  };

  return { trace, newKv };
}

/**
 * Run a complete 40-layer forward pass through `model`.
 *
 * - Defaults `positionIds` to `[pastLen .. pastLen + seqLen - 1]` and
 *   `cachePositions` to `positionIds`.
 * - Collects all 41 residual checkpoints `[h_0, h_1, ..., h_40]` and all 40
 *   per-layer newly computed `[seqLen, 64]` KV slices (`newKvs`).
 * - Retains detailed `layerTraces` for every layer in `retainedTraceLayers`
 *   (or all 40 layers when `retainedTraceLayers` is omitted).
 * - Computes raw un-suppressed `logits` and `probs` (`[MASK]` is never clamped).
 * - Never mutates `options.pastKvs`.
 */
export function runDiffusionForwardPass(
  model: DiffusionModelParameters,
  options: DiffusionForwardOptions,
): DiffusionForwardResult {
  const { config } = model;
  const numLayers = config.numLayers;
  const d = config.hiddenSize;
  const vocabSize = config.vocabSize;

  const seqLen = options.tokenIds.length;
  if (seqLen <= 0) {
    throw new RangeError('tokenIds must contain at least one token');
  }

  const tokenIds = new Int32Array(seqLen);
  for (let t = 0; t < seqLen; t++) {
    tokenIds[t] = options.tokenIds[t];
  }

  // Validate pastKvs without mutating anything
  let pastLen = 0;
  const pastKvs = options.pastKvs ?? null;
  if (pastKvs !== null) {
    if (pastKvs.length !== numLayers) {
      throw new RangeError(
        `pastKvs length (${pastKvs.length}) must equal numLayers (${numLayers})`,
      );
    }
    let seenNonNull = false;
    for (let l = 0; l < numLayers; l++) {
      const slice = pastKvs[l];
      if (slice !== null && slice !== undefined) {
        if (!seenNonNull) {
          pastLen = slice.seqLen;
          seenNonNull = true;
        } else if (slice.seqLen !== pastLen) {
          throw new RangeError(
            `Inconsistent pastKv seqLen at layer ${l}: expected ${pastLen}, got ${slice.seqLen}`,
          );
        }
      } else if (seenNonNull && pastLen > 0) {
        throw new RangeError(
          `Layer ${l} pastKv is null while earlier layers have pastLen=${pastLen}`,
        );
      }
    }
  }

  const kvLen = pastLen + seqLen;

  let positionIds: Int32Array;
  if (options.positionIds !== undefined) {
    if (options.positionIds.length !== seqLen) {
      throw new RangeError(
        `positionIds length (${options.positionIds.length}) must equal seqLen (${seqLen})`,
      );
    }
    positionIds = new Int32Array(options.positionIds);
  } else {
    positionIds = new Int32Array(seqLen);
    for (let t = 0; t < seqLen; t++) {
      positionIds[t] = pastLen + t;
    }
  }

  let cachePositions: Int32Array;
  if (options.cachePositions !== undefined) {
    if (options.cachePositions.length !== seqLen) {
      throw new RangeError(
        `cachePositions length (${options.cachePositions.length}) must equal seqLen (${seqLen})`,
      );
    }
    cachePositions = new Int32Array(options.cachePositions);
  } else {
    cachePositions = new Int32Array(positionIds);
  }

  const { invFreq, attentionScaling } = computeYarnInvFreq(config.rope);
  const embeddings = embedDiffusionTokens(model, tokenIds);

  const residualCheckpoints: Float32Array[] = new Array<Float32Array>(
    numLayers + 1,
  );
  residualCheckpoints[0] = embeddings;

  const layerTraces = new Map<number, DiffusionLayerTrace>();
  const newKvs: DiffusionLayerKvSlice[] = new Array<DiffusionLayerKvSlice>(
    numLayers,
  );
  const retainAll = options.retainedTraceLayers === undefined;

  let currentHidden = embeddings;
  for (let l = 0; l < numLayers; l++) {
    const pastKv = pastKvs ? (pastKvs[l] ?? null) : null;
    const { trace, newKv } = runDiffusionLayer(
      {
        hiddenStates: currentHidden,
        seqLen,
        layerWeights: model.layers[l],
        config,
        invFreq,
        attentionScaling,
        positionIds,
        cachePositions,
        maskMode: options.maskMode,
        customMask: options.customMask,
        loraEnabled: options.loraEnabled,
        useCache: options.useCache,
        pastKv,
      },
      l,
    );

    currentHidden = trace.output;
    residualCheckpoints[l + 1] = currentHidden;
    newKvs[l] = newKv;

    if (retainAll || options.retainedTraceLayers!.has(l)) {
      layerTraces.set(l, trace);
    }
  }

  const finalNorm = applyDiffusionRmsNorm(
    currentHidden,
    seqLen,
    d,
    model.finalNorm,
    config.rmsNormEps,
  );
  const logits = linearProject(
    finalNorm,
    seqLen,
    d,
    model.diffusionHead,
    vocabSize,
  );
  const probs = softmaxRows(logits, seqLen, vocabSize);

  return {
    tokenIds,
    seqLen,
    pastLen,
    kvLen,
    maskMode: options.maskMode,
    loraEnabled: options.loraEnabled,
    useCache: options.useCache,
    positionIds,
    cachePositions,
    embeddings,
    residualCheckpoints,
    layerTraces,
    newKvs,
    finalNorm,
    logits,
    probs,
  };
}
