/**
 * Full-nGPT Gated GQA & Dual-RoPE Attention Sublayer (`src/dory/attention.ts`)
 *
 * Implements even physical sublayers (`p in {0, 2, ..., 70}`) under the locked `D5`
 * browser numerical contract (`Float32Array` storage + native JS binary64 accumulation,
 * fixed ascending reduction order, named binary32 write-back boundaries):
 *   1. Bias-free linear projections (`Q, G: d -> h_q*d_h = 64`; `K, V: d -> h_kv*d_h = 16`).
 *      NO Layer-0 pre-norm (`DR5`); `V` and `G` are never L2-normalized.
 *   2. 100% NeoX `rotate_half` Dual-RoPE (`theta_S = 16` for SWA, `theta_F = 1024` for Full)
 *      applied strictly BEFORE Q/K L2Norm (`DR6`, matching `vLLM`/`llama.cpp`/TeX).
 *   3. Per-head unit-sphere L2Norm (`eps = 1e-6`) on `Q` and `K`, followed by Q-only squared
 *      scale `sqk^2 = (sqk * sqrt(d_h) / sigma_0)^2` (cached `K` remains unit-norm unscaled!).
 *   4. Read-only transient KV workspace `[past_visible (<= W - 1) || current_chunk (L)]` (`DR24`)
 *      enabling long chunks `L > cap_swa = 16` up to `S_max = 32` without ring overwrite.
 *   5. Causal 4:1 GQA (`kvHead = Math.floor(qh / kvGroupSize)`) with exact SWA visibility
 *      `0 <= qPos - kPos < W` (or `0 <= qPos - kPos` for Full) and binary64 softmax.
 *   6. Post-attention KV commit ONLY when `writeKv === true` (never advancing SWA ring pointers
 *      on non-final `last_loop` iterations `t < k`, `DR17`).
 *   7. Input-dependent sigmoid output gate `C * sigmoid(G * s_gate^2)` (honoring custom `g_0`,
 *      `DR3`), followed by `attn_sub_norm = N_{q_dim}(...)` over ALL concatenated heads `h_q * d_h`
 *      (NOT per head!), `o_proj`, and channel-wise nGPT NLERP residual (Anti-SLERP `DR7`).
 *
 * Zero runtime weight normalization, zero scale folding, zero ambient RNG.
 */

import { buildTransientKvWorkspace, commitChunkToKvBank } from './kvCache';
import {
  effectiveSGateSquared,
  effectiveSqkSquared,
  l2NormHeadsInto,
  linearProject,
  ngptNlerpResidualRows,
  ngptNlerpResidualRowsDetailed,
  sigmoidScalar,
} from './ngptMath';
import { applyNeoXRotateHalf } from './rope';
import type {
  DoryAttentionSublayerInput,
  DoryAttentionSublayerResult,
  DoryAttentionSublayerTrace,
  DoryNlerpTrace,
} from './types';

/**
 * Executes one even-indexed Full-nGPT Gated GQA attention sublayer (`p in {0, 2, ..., 70}`).
 * Output tensors and committed KV entries are bit-for-bit identical whether `captureTrace`
 * is `true` or `false`.
 */
export function runDoryAttentionSublayer(
  input: DoryAttentionSublayerInput,
): DoryAttentionSublayerResult {
  const {
    hiddenStates,
    seqLen,
    startPos,
    physicalLayerIdx,
    loopIteration,
    weights,
    ropeTables,
    kvBank,
    writeKv,
    config,
    captureTrace = false,
  } = input;

  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new RangeError(`seqLen must be an integer >= 1; got ${seqLen}.`);
  }
  if (!Number.isInteger(startPos) || startPos < 0) {
    throw new RangeError(`startPos must be an integer >= 0; got ${startPos}.`);
  }
  if (startPos + seqLen > config.maxPositionEmbeddings) {
    throw new RangeError(
      `Position overflow at Attn p=${physicalLayerIdx}: startPos(${startPos}) + seqLen(${seqLen}) > maxPositionEmbeddings(${config.maxPositionEmbeddings}).`,
    );
  }

  const d = config.hiddenSize;
  const hQ = config.numAttentionHeads;
  const hKv = config.numKeyValueHeads;
  const kvGroupSize = config.kvGroupSize;
  const dH = config.headDim;
  const qDim = config.qProjOutDim;
  const kvDim = config.kvProjOutDim;
  const eps = config.l2NormEps;

  if (hiddenStates.length !== seqLen * d) {
    throw new RangeError(
      `hiddenStates length mismatch at Attn p=${physicalLayerIdx}: got ${hiddenStates.length}, expected ${seqLen * d}.`,
    );
  }
  if (weights.physicalLayerIdx !== physicalLayerIdx) {
    throw new RangeError(
      `Attention weight mismatch: input physicalLayerIdx=${physicalLayerIdx}, weights.physicalLayerIdx=${weights.physicalLayerIdx}.`,
    );
  }
  if (kvBank.physicalLayerIdx !== physicalLayerIdx) {
    throw new RangeError(
      `KV bank layer mismatch: input physicalLayerIdx=${physicalLayerIdx}, kvBank.physicalLayerIdx=${kvBank.physicalLayerIdx}.`,
    );
  }
  if (kvBank.attentionKind !== weights.attentionKind) {
    throw new RangeError(
      `KV bank attentionKind mismatch at p=${physicalLayerIdx}: weights=${weights.attentionKind}, bank=${kvBank.attentionKind}.`,
    );
  }

  // 1. Bias-free linear projections (D5 write-back boundaries; NO Layer-0 pre-norm DR5)
  const qPre = linearProject(hiddenStates, weights.qProj, seqLen, d, qDim);
  const kPre = linearProject(hiddenStates, weights.kProj, seqLen, d, kvDim);
  const vNew = linearProject(hiddenStates, weights.vProj, seqLen, d, kvDim);
  const gPre = linearProject(hiddenStates, weights.gProj, seqLen, d, qDim);

  // 2. Dual-RoPE (100% NeoX rotate_half) strictly BEFORE Q/K L2Norm (DR6)
  const ropeTable = weights.ropeRegime === 'sliding' ? ropeTables.sliding : ropeTables.full;
  const qRot = applyNeoXRotateHalf(qPre, seqLen, hQ, dH, startPos, ropeTable);
  const kRot = applyNeoXRotateHalf(kPre, seqLen, hKv, dH, startPos, ropeTable);

  // 3. Per-head unit-sphere L2Norm (eps = 1e-6) + Q-only squared sqk^2 scaling
  const qNorm = new Float32Array(seqLen * qDim);
  l2NormHeadsInto(qRot, qNorm, seqLen, hQ, dH, eps);

  const kCache = new Float32Array(seqLen * kvDim);
  l2NormHeadsInto(kRot, kCache, seqLen, hKv, dH, eps);

  const effectiveSqkSq = effectiveSqkSquared(weights.sqk, config);
  const qScaled = new Float32Array(seqLen * qDim);
  for (let t = 0; t < seqLen; t += 1) {
    const rowOff = t * qDim;
    for (let c = 0; c < qDim; c += 1) {
      qScaled[rowOff + c] = qNorm[rowOff + c] * effectiveSqkSq[c];
    }
  }

  // 4. Build read-only transient KV workspace [past_visible || current_chunk] (DR17, DR19, DR24)
  const workspace = buildTransientKvWorkspace(
    kvBank,
    startPos,
    kCache,
    vNew,
    seqLen,
    loopIteration,
  );
  const kvLen = workspace.totalLen;

  // 5. Exact causal / SWA visibility mask: 0 <= qPos - kPos (< W for SWA)
  const isSliding = weights.attentionKind === 'sliding_attention';
  const slidingWindow = config.slidingWindow;
  const maskVisible = new Uint8Array(seqLen * kvLen);
  for (let qi = 0; qi < seqLen; qi += 1) {
    const qPos = startPos + qi;
    const maskRowOff = qi * kvLen;
    for (let ki = 0; ki < kvLen; ki += 1) {
      const diff = qPos - workspace.absPositions[ki];
      maskVisible[maskRowOff + ki] = diff >= 0 && (!isSliding || diff < slidingWindow) ? 1 : 0;
    }
  }

  // 6. Causal 4:1 GQA with binary64 dot products, softmax, and value aggregation (D5)
  const invSqrtHeadDim = 1.0 / Math.sqrt(dH);
  const attnContext = new Float32Array(seqLen * qDim);
  const traceScores = captureTrace ? new Float32Array(hQ * seqLen * kvLen) : null;
  const traceProbs = captureTrace ? new Float32Array(hQ * seqLen * kvLen) : null;

  const rowScores64 = new Float64Array(kvLen);
  const rowProbs64 = new Float64Array(kvLen);

  for (let qh = 0; qh < hQ; qh += 1) {
    const kvHead = Math.floor(qh / kvGroupSize);
    const qHeadOff = qh * dH;
    const kvHeadOff = kvHead * dH;

    for (let qi = 0; qi < seqLen; qi += 1) {
      const qBase = qi * qDim + qHeadOff;
      const maskRowOff = qi * kvLen;
      const traceRowOff = (qh * seqLen + qi) * kvLen;

      // Pass A: Scaled dot-product scores in fixed ascending order 0..dH-1
      let maxScore = -Infinity;
      for (let ki = 0; ki < kvLen; ki += 1) {
        if (maskVisible[maskRowOff + ki] === 0) {
          rowScores64[ki] = -Infinity;
          if (traceScores !== null) {
            traceScores[traceRowOff + ki] = -Infinity;
          }
          continue;
        }
        const kBase = ki * kvDim + kvHeadOff;
        let dot = 0.0;
        for (let m = 0; m < dH; m += 1) {
          dot += qScaled[qBase + m] * workspace.k[kBase + m];
        }
        const score = dot * invSqrtHeadDim;
        rowScores64[ki] = score;
        if (score > maxScore) {
          maxScore = score;
        }
        if (traceScores !== null) {
          traceScores[traceRowOff + ki] = score;
        }
      }

      // Pass B: Numerically stable binary64 softmax in fixed ascending order 0..kvLen-1
      let expSum = 0.0;
      for (let ki = 0; ki < kvLen; ki += 1) {
        if (maskVisible[maskRowOff + ki] === 0) {
          rowProbs64[ki] = 0.0;
          continue;
        }
        const e = Math.exp(rowScores64[ki] - maxScore);
        rowProbs64[ki] = e;
        expSum += e;
      }
      const invExpSum = expSum > 0.0 ? 1.0 / expSum : 0.0;
      for (let ki = 0; ki < kvLen; ki += 1) {
        const prob = maskVisible[maskRowOff + ki] === 1 ? rowProbs64[ki] * invExpSum : 0.0;
        rowProbs64[ki] = prob;
        if (traceProbs !== null) {
          traceProbs[traceRowOff + ki] = prob;
        }
      }

      // Pass C: Value aggregation P * V -> named binary32 write-back C = attnContext
      for (let m = 0; m < dH; m += 1) {
        let acc = 0.0;
        for (let ki = 0; ki < kvLen; ki += 1) {
          const prob = rowProbs64[ki];
          if (prob !== 0.0) {
            acc += prob * workspace.v[ki * kvDim + kvHeadOff + m];
          }
        }
        attnContext[qBase + m] = acc;
      }
    }
  }

  // 7. Commit chunk to persistent KV bank strictly AFTER attention completes (DR17, DR24)
  if (writeKv) {
    commitChunkToKvBank(kvBank, startPos, kCache, vNew, seqLen, loopIteration);
  }

  // 8. Input-dependent sigmoid output gate + attn_sub_norm over ALL qDim = h_q * d_h channels
  const effectiveSGateSq = effectiveSGateSquared(weights.sGate, config);
  const attnSubNorm = new Float32Array(seqLen * qDim);
  const gateValues = captureTrace ? new Float32Array(seqLen * qDim) : null;
  const gatedContext = captureTrace ? new Float32Array(seqLen * qDim) : null;

  const gatedRow64 = new Float64Array(qDim);

  for (let t = 0; t < seqLen; t += 1) {
    const rowOff = t * qDim;
    let sumSq = 0.0;
    for (let c = 0; c < qDim; c += 1) {
      const idx = rowOff + c;
      const gVal = sigmoidScalar(gPre[idx] * effectiveSGateSq[c]);
      const gated = attnContext[idx] * gVal;
      gatedRow64[c] = gated;
      sumSq += gated * gated;

      if (captureTrace) {
        gateValues![idx] = gVal;
        gatedContext![idx] = gated;
      }
    }
    const invNorm = 1.0 / Math.sqrt(sumSq + eps);
    for (let c = 0; c < qDim; c += 1) {
      attnSubNorm[rowOff + c] = gatedRow64[c] * invNorm;
    }
  }

  // 9. Bias-free Output projection O = Y_norm W_o^T [seqLen, d] (D5 write-back boundary)
  const oProjOut = linearProject(attnSubNorm, weights.oProj, seqLen, qDim, d);

  // 10. Channel-wise nGPT NLERP residual blend (Anti-SLERP DR7; unclamped |alpha_hat| >= 0)
  let output: Float32Array;
  let nlerpTrace: DoryNlerpTrace | null = null;
  if (captureTrace) {
    const detailed = ngptNlerpResidualRowsDetailed(
      hiddenStates,
      oProjOut,
      weights.attnAlpha,
      seqLen,
      d,
      config,
    );
    nlerpTrace = detailed;
    output = detailed.output;
  } else {
    output = ngptNlerpResidualRows(
      hiddenStates,
      oProjOut,
      weights.attnAlpha,
      seqLen,
      d,
      config,
    );
  }

  const trace: DoryAttentionSublayerTrace | null = captureTrace
    ? {
        kind: 'attention',
        physicalLayerIdx,
        pairedLayerIdx: weights.pairedLayerIdx,
        stage: weights.stage,
        loopIteration,
        attentionKind: weights.attentionKind,
        ropeRegime: weights.ropeRegime,
        ropeTheta: ropeTable.theta,
        seqLen,
        startPos,
        pastLen: workspace.pastLen,
        kvLen,
        writeKv,
        bankSlotId: kvBank.bankSlotId,
        bankKind: kvBank.bankKind,
        kvAbsPositions: workspace.absPositions,
        kvWriterLoops: workspace.writerLoops,
        qPre,
        kPre,
        vNew,
        gPre,
        qRot,
        kRot,
        qNorm,
        kCache,
        effectiveSqkSq,
        qScaled,
        maskVisible,
        scores: traceScores!,
        probs: traceProbs!,
        attnContext,
        effectiveSGateSq,
        gateValues: gateValues!,
        gatedContext: gatedContext!,
        attnSubNorm,
        oProjOut,
        nlerp: nlerpTrace!,
      }
    : null;

  return {
    output,
    sublayerOut: oProjOut,
    trace,
  };
}
