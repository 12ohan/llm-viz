/**
 * Ministral 3 RMSNorm, Linear Projections, Softmax & 32Q/8KV GQA + Draft-Only LoRA
 * (`src/diffusion/attention.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching the exact
 * per-layer attention math of `nvidia/Nemotron-Labs-Diffusion-14B`:
 *   - Pre-attention RMSNorm (`(x * rsqrt(mean(x^2) + eps)).to(f32) * weight`)
 *   - Bias-free GQA projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
 *   - Half-split YaRN RoPE + Llama-4 query-only scaling (keys untouched)
 *   - Non-mutating KV concatenation (supports both `useCache=true` and transient `useCache=false`)
 *   - Explicit `causal`, `bidirectional`, and `custom` binary attention masks
 *   - Optional draft-only `o_proj` LoRA (`W_O·a + 4.0·B(A·a)`, 0 ULP off-path invariance)
 *
 * Pure headless TypeScript — deterministic, self-contained math kernel.
 */

import { fmtSigned } from '../shared/format';
import {
  applyRotaryPosEmb,
  computeLlama4AttnScales,
  computeRopeCosSin,
} from './rope';
import type {
  DiffusionAttentionInput,
  DiffusionAttentionMaskMode,
  DiffusionAttentionTrace,
} from './types';

/**
 * Ministral 3 RMSNorm: computes mean square per row, multiplies by `rsqrt(var + eps)`,
 * casts the normalized vector to Float32, and scales by `weight[d]`.
 */
export function applyDiffusionRmsNorm(
  x: Float32Array,
  seqLen: number,
  hiddenSize: number,
  weight: Float32Array,
  eps: number,
): Float32Array {
  if (x.length < seqLen * hiddenSize) {
    throw new RangeError(
      `x length (${x.length}) is smaller than seqLen * hiddenSize (${seqLen * hiddenSize})`,
    );
  }
  if (weight.length !== hiddenSize) {
    throw new RangeError(
      `weight length (${weight.length}) must equal hiddenSize (${hiddenSize})`,
    );
  }

  const out = new Float32Array(seqLen * hiddenSize);

  for (let t = 0; t < seqLen; t++) {
    const base = t * hiddenSize;
    let sumSq = 0;
    for (let d = 0; d < hiddenSize; d++) {
      const v = x[base + d];
      sumSq += v * v;
    }
    const meanSq = sumSq / hiddenSize;
    const invRms = 1.0 / Math.sqrt(meanSq + eps);
    for (let d = 0; d < hiddenSize; d++) {
      const normed = Math.fround(x[base + d] * invRms);
      out[base + d] = Math.fround(normed * weight[d]);
    }
  }

  return out;
}

/**
 * Bias-free linear projection matching PyTorch `F.linear(input, weight)`:
 *   - `input`: `[seqLen, inDim]` row-major
 *   - `weight`: `[outDim, inDim]` row-major
 *   - returns `[seqLen, outDim]` row-major Float32Array
 */
export function linearProject(
  input: Float32Array,
  seqLen: number,
  inDim: number,
  weight: Float32Array,
  outDim: number,
): Float32Array {
  if (input.length < seqLen * inDim) {
    throw new RangeError(
      `input length (${input.length}) is smaller than seqLen * inDim (${seqLen * inDim})`,
    );
  }
  if (weight.length < outDim * inDim) {
    throw new RangeError(
      `weight length (${weight.length}) is smaller than outDim * inDim (${outDim * inDim})`,
    );
  }

  const out = new Float32Array(seqLen * outDim);

  for (let t = 0; t < seqLen; t++) {
    const xBase = t * inDim;
    const yBase = t * outDim;
    for (let o = 0; o < outDim; o++) {
      const wBase = o * inDim;
      let acc = 0;
      for (let i = 0; i < inDim; i++) {
        acc += input[xBase + i] * weight[wBase + i];
      }
      out[yBase + o] = Math.fround(acc);
    }
  }

  return out;
}

/**
 * Numerically stable row-wise softmax returning Float32 probabilities.
 *
 * Rows where every entry is `-Infinity` (fully masked) emit exact `0.0`s.
 * Masked entries (`-Infinity`) always emit exact `0.0`.
 */
export function softmaxRows(
  scores: Float32Array,
  rows: number,
  cols: number,
): Float32Array {
  if (scores.length < rows * cols) {
    throw new RangeError(
      `scores length (${scores.length}) is smaller than rows * cols (${rows * cols})`,
    );
  }

  const out = new Float32Array(rows * cols);
  if (rows === 0 || cols === 0) return out;

  const expScratch = new Float64Array(cols);

  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let maxVal = -Infinity;
    for (let c = 0; c < cols; c++) {
      const s = scores[base + c];
      if (s > maxVal) maxVal = s;
    }

    if (!Number.isFinite(maxVal)) {
      // All entries are -Infinity (or NaN); leave row as 0.0
      continue;
    }

    let sumExp = 0;
    for (let c = 0; c < cols; c++) {
      const s = scores[base + c];
      if (s === -Infinity || (!Number.isFinite(s) && s < 0)) {
        expScratch[c] = 0;
      } else {
        const e = Math.exp(s - maxVal);
        expScratch[c] = e;
        sumExp += e;
      }
    }

    if (sumExp <= 0 || !Number.isFinite(sumExp)) continue;

    for (let c = 0; c < cols; c++) {
      const e = expScratch[c];
      out[base + c] = e === 0 ? 0.0 : Math.fround(e / sumExp);
    }
  }

  return out;
}

/**
 * Run single-layer 32Q/8KV Grouped-Query Attention with YaRN RoPE,
 * Llama-4 query temperature scaling, explicit mask mode, non-mutating
 * KV concatenation, and optional draft-only `o_proj` LoRA.
 */
export function runDiffusionAttention(
  input: DiffusionAttentionInput,
): DiffusionAttentionTrace {
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
  const nQ = config.numAttentionHeads;
  const nKv = config.numKeyValueHeads;
  const kvGroupSize = config.kvGroupSize;
  const headDim = config.headDim;
  const qDim = config.qProjOutDim;
  const kvDim = config.kvProjOutDim;

  if (positionIds.length !== seqLen) {
    throw new RangeError(
      `positionIds length (${positionIds.length}) must equal seqLen (${seqLen})`,
    );
  }
  const effectiveCachePositions = cachePositions ?? positionIds;
  if (effectiveCachePositions.length !== seqLen) {
    throw new RangeError(
      `cachePositions length (${effectiveCachePositions.length}) must equal seqLen (${seqLen})`,
    );
  }

  // 1. Q, K, V linear projections (bias-free)
  const { selfAttn, lora } = layerWeights;
  const qPre = linearProject(hiddenStates, seqLen, d, selfAttn.qProj, qDim);
  const kPre = linearProject(hiddenStates, seqLen, d, selfAttn.kProj, kvDim);
  const vNew = linearProject(hiddenStates, seqLen, d, selfAttn.vProj, kvDim);

  // 2. Half-split YaRN RoPE
  const { cos, sin } = computeRopeCosSin(
    invFreq,
    positionIds,
    attentionScaling,
    headDim,
  );
  const qRot = applyRotaryPosEmb(qPre, seqLen, nQ, headDim, cos, sin);
  const kRot = applyRotaryPosEmb(kPre, seqLen, nKv, headDim, cos, sin);

  // 3. Llama-4 post-RoPE query-only temperature scaling (keys untouched)
  const llama4Scales = computeLlama4AttnScales(
    effectiveCachePositions,
    config.rope.llama4Beta,
    config.rope.originalMaxPositionEmbeddings,
  );
  const qScaled = new Float32Array(seqLen * qDim);
  for (let t = 0; t < seqLen; t++) {
    const s = llama4Scales[t];
    const base = t * qDim;
    for (let j = 0; j < qDim; j++) {
      qScaled[base + j] = Math.fround(qRot[base + j] * s);
    }
  }

  // 4. Non-mutating KV concatenation (`pastKv` is strictly read-only)
  const pastLen = pastKv ? pastKv.seqLen : 0;
  const kvLen = pastLen + seqLen;
  const kFull = new Float32Array(kvLen * kvDim);
  const vFull = new Float32Array(kvLen * kvDim);

  if (pastKv && pastLen > 0) {
    const pastFloats = pastLen * kvDim;
    if (pastKv.k.length < pastFloats || pastKv.v.length < pastFloats) {
      throw new RangeError(
        `pastKv buffer too small for pastLen=${pastLen} * kvDim=${kvDim}`,
      );
    }
    kFull.set(pastKv.k.subarray(0, pastFloats), 0);
    vFull.set(pastKv.v.subarray(0, pastFloats), 0);
  }
  kFull.set(kRot, pastLen * kvDim);
  vFull.set(vNew, pastLen * kvDim);

  if (maskMode === 'custom') {
    if (!customMask || customMask.length !== seqLen * kvLen) {
      throw new RangeError(
        `customMask of length seqLen * kvLen (${seqLen * kvLen}) is required when maskMode === 'custom'`,
      );
    }
  }

  // 5. GQA scaled dot-product attention (`32Q / 8KV`, 4 Q heads per KV head)
  const scale = 1.0 / Math.sqrt(headDim);
  const scores = new Float32Array(nQ * seqLen * kvLen);
  const probs = new Float32Array(nQ * seqLen * kvLen);
  const attnConcat = new Float32Array(seqLen * qDim);

  for (let h = 0; h < nQ; h++) {
    const kvHead = Math.floor(h / kvGroupSize);
    const headPlaneOffset = h * seqLen * kvLen;

    for (let qRow = 0; qRow < seqLen; qRow++) {
      const qVecOffset = qRow * qDim + h * headDim;
      const rowScoreOffset = headPlaneOffset + qRow * kvLen;

      for (let kCol = 0; kCol < kvLen; kCol++) {
        let visible: boolean;
        if (maskMode === 'causal') {
          visible = kCol <= pastLen + qRow;
        } else if (maskMode === 'bidirectional') {
          visible = true;
        } else {
          visible = customMask![qRow * kvLen + kCol] !== 0;
        }

        if (!visible) {
          scores[rowScoreOffset + kCol] = -Infinity;
        } else {
          const kVecOffset = kCol * kvDim + kvHead * headDim;
          let dot = 0;
          for (let i = 0; i < headDim; i++) {
            dot += qScaled[qVecOffset + i] * kFull[kVecOffset + i];
          }
          scores[rowScoreOffset + kCol] = Math.fround(dot * scale);
        }
      }
    }

    const headProbs = softmaxRows(
      scores.subarray(headPlaneOffset, headPlaneOffset + seqLen * kvLen),
      seqLen,
      kvLen,
    );
    probs.set(headProbs, headPlaneOffset);

    for (let qRow = 0; qRow < seqLen; qRow++) {
      const rowProbOffset = headPlaneOffset + qRow * kvLen;
      const outVecOffset = qRow * qDim + h * headDim;
      for (let i = 0; i < headDim; i++) {
        let acc = 0;
        for (let kCol = 0; kCol < kvLen; kCol++) {
          const p = probs[rowProbOffset + kCol];
          if (p !== 0) {
            acc += p * vFull[kCol * kvDim + kvHead * headDim + i];
          }
        }
        attnConcat[outVecOffset + i] = Math.fround(acc);
      }
    }
  }

  // 6. Output projection + optional draft-only LoRA (`W_O·a + 4.0·B(A·a)`)
  const oProjBase = linearProject(attnConcat, seqLen, qDim, selfAttn.oProj, d);

  let loraHidden: Float32Array | null = null;
  let loraDelta: Float32Array | null = null;
  let output: Float32Array;

  if (!loraEnabled) {
    output = new Float32Array(oProjBase);
  } else {
    loraHidden = linearProject(
      attnConcat,
      seqLen,
      qDim,
      lora.loraA,
      config.loraRank,
    );
    const loraRaw = linearProject(
      loraHidden,
      seqLen,
      config.loraRank,
      lora.loraB,
      d,
    );
    loraDelta = new Float32Array(seqLen * d);
    output = new Float32Array(seqLen * d);
    const loraScale = config.loraScale;
    for (let idx = 0; idx < seqLen * d; idx++) {
      const delta = Math.fround(loraScale * loraRaw[idx]);
      loraDelta[idx] = delta;
      output[idx] = Math.fround(oProjBase[idx] + delta);
    }
  }

  return {
    qPre,
    kPre,
    vNew,
    qRot,
    kRot,
    llama4Scales,
    qScaled,
    kFull,
    vFull,
    scores,
    probs,
    attnConcat,
    oProjBase,
    loraHidden,
    loraDelta,
    output,
    pastLen,
    kvLen,
    seqLen,
    maskMode,
    loraEnabled,
    useCache,
  };
}

export interface DiffusionAttentionCellFormulaParams {
  readonly stage: 'score' | 'prob' | 'o_proj';
  readonly qHead: number;
  readonly kvHead: number;
  readonly qRow: number;
  readonly kCol?: number;
  readonly outDimIndex?: number;
  readonly headDim: number;
  readonly maskMode: DiffusionAttentionMaskMode;
  readonly isMasked?: boolean;
  readonly value: number;
  readonly oProjBaseVal?: number;
  readonly loraDeltaVal?: number | null;
  readonly loraScale?: number;
}

/**
 * Format a concise human-readable cell formula for the GQA + LoRA Formula Inspector.
 */
export function formatDiffusionAttentionCellFormula(
  params: DiffusionAttentionCellFormulaParams,
): string {
  const {
    stage,
    qHead,
    kvHead,
    qRow,
    kCol = 0,
    outDimIndex = 0,
    headDim,
    maskMode,
    isMasked = false,
    value,
    oProjBaseVal,
    loraDeltaVal,
    loraScale = 4.0,
  } = params;

  if (stage === 'score') {
    if (isMasked) {
      return `score[h${qHead}→kv${kvHead}, q=${qRow}, k=${kCol}] = −∞ (masked · ${maskMode})`;
    }
    return `score[h${qHead}→kv${kvHead}, q=${qRow}, k=${kCol}] = ⟨Q_scaled, K_rot⟩ / √${headDim} = ${fmtSigned(value)}`;
  }

  if (stage === 'prob') {
    if (isMasked) {
      return `prob[h${qHead}→kv${kvHead}, q=${qRow}, k=${kCol}] = 0.000 (masked · ${maskMode})`;
    }
    return `prob[h${qHead}→kv${kvHead}, q=${qRow}, k=${kCol}] = softmax(score) = ${fmtSigned(value, 4)}`;
  }

  const base = oProjBaseVal ?? value;
  if (loraDeltaVal !== undefined && loraDeltaVal !== null) {
    return `o_proj[t=${qRow}, d=${outDimIndex}] = W_O·a (${fmtSigned(base)}) + ${loraScale.toFixed(1)}·B(A·a) (${fmtSigned(loraDeltaVal)}) = ${fmtSigned(value)}`;
  }
  return `o_proj[t=${qRow}, d=${outDimIndex}] = W_O·a = ${fmtSigned(value)} (LoRA OFF)`;
}
