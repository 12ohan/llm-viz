/**
 * Full-nGPT Normalized SwiGLU MLP Sublayer (`src/dory/swiglu.ts`)
 *
 * Implements odd physical sublayers (`p in {1, 3, ..., 71}`) under the canonical `D5`
 * browser numerical contract (`Float32Array` storage + native JS binary64 accumulation,
 * fixed ascending `0..dim-1` reduction order, named binary32 write-back boundaries):
 *   1. Bias-free projections: `U_g_raw = h W_gate^T`, `U_u_raw = h W_up^T` (`[L, d_ff]`).
 *   2. Signed effective SwiGLU scales (`NO .abs()`): `suv_gate * sqrt(d)`, `suv_up * sqrt(d)`
 *      (raw init `1.0`, NOT `initializer_range`; Audit §5.3 family #4).
 *   3. `U_g = U_g_raw * effectiveSuvGate`, `U_u = U_u_raw * effectiveSuvUp`.
 *   4. `ffn_sub_norm`: `M = N_{d_ff}(SiLU(U_g) * U_u, eps = 1e-6)` over all `d_ff = 24` dims.
 *   5. Down projection: `O = M W_down^T` (`[L, d]`).
 *   6. Channel-wise nGPT NLERP residual (Anti-SLERP `DR7`, unclamped `|alpha_hat| >= 0`):
 *      `h_next = N_d(N_d(h) + |mlp_alpha * alpha_init / sigma_0| * (N_d(O) - N_d(h)))`.
 *
 * Zero runtime weight normalization, zero scale folding, zero ambient RNG.
 */

import {
  effectiveSuvScale,
  linearProject,
  ngptNlerpResidualRows,
  ngptNlerpResidualRowsDetailed,
  siluScalar,
} from './ngptMath';
import type {
  DoryMlpSublayerInput,
  DoryMlpSublayerResult,
  DoryMlpSublayerTrace,
  DoryNlerpTrace,
} from './types';

/**
 * Executes one odd-indexed Full-nGPT SwiGLU MLP sublayer (`p in {1, 3, ..., 71}`).
 * Output tensors are bit-for-bit identical whether `captureTrace` is `true` or `false`.
 */
export function runDoryMlpSublayer(input: DoryMlpSublayerInput): DoryMlpSublayerResult {
  const {
    hiddenStates,
    seqLen,
    physicalLayerIdx,
    loopIteration,
    weights,
    config,
    captureTrace = false,
  } = input;

  if (!Number.isInteger(seqLen) || seqLen < 1) {
    throw new RangeError(`seqLen must be an integer >= 1; got ${seqLen}.`);
  }
  const d = config.hiddenSize;
  const dFf = config.intermediateSize;
  const eps = config.l2NormEps;

  if (hiddenStates.length !== seqLen * d) {
    throw new RangeError(
      `hiddenStates length mismatch at MLP p=${physicalLayerIdx}: got ${hiddenStates.length}, expected ${seqLen * d}.`,
    );
  }
  if (weights.physicalLayerIdx !== physicalLayerIdx) {
    throw new RangeError(
      `MLP weight mismatch: input physicalLayerIdx=${physicalLayerIdx}, weights.physicalLayerIdx=${weights.physicalLayerIdx}.`,
    );
  }

  // 1. Bias-free Gate & Up linear projections [seqLen, d_ff] (D5 write-back boundary)
  const gatePre = linearProject(hiddenStates, weights.gateProj, seqLen, d, dFf);
  const upPre = linearProject(hiddenStates, weights.upProj, seqLen, d, dFf);

  // 2. Signed effective SwiGLU scales: suv * sqrt(d) (NO .abs(), raw init 1.0)
  const effectiveSuvGate = effectiveSuvScale(weights.suvGate, config);
  const effectiveSuvUp = effectiveSuvScale(weights.suvUp, config);

  // 3. Scaled SiLU * Up product + ffn_sub_norm over all d_ff channels (binary64 acc -> F32 M)
  const ffnSubNorm = new Float32Array(seqLen * dFf);
  const gateScaled = captureTrace ? new Float32Array(seqLen * dFf) : null;
  const upScaled = captureTrace ? new Float32Array(seqLen * dFf) : null;
  const gateSilu = captureTrace ? new Float32Array(seqLen * dFf) : null;
  const swigluProduct = captureTrace ? new Float32Array(seqLen * dFf) : null;

  const prodRow64 = new Float64Array(dFf);

  for (let t = 0; t < seqLen; t += 1) {
    const rowOff = t * dFf;
    let sumSq = 0.0;
    for (let c = 0; c < dFf; c += 1) {
      const idx = rowOff + c;
      const gScaled = gatePre[idx] * effectiveSuvGate[c];
      const uScaled = upPre[idx] * effectiveSuvUp[c];
      const gSilu = siluScalar(gScaled);
      const prod = gSilu * uScaled;
      prodRow64[c] = prod;
      sumSq += prod * prod;

      if (captureTrace) {
        gateScaled![idx] = gScaled;
        upScaled![idx] = uScaled;
        gateSilu![idx] = gSilu;
        swigluProduct![idx] = prod;
      }
    }
    const invNorm = 1.0 / Math.sqrt(sumSq + eps);
    for (let c = 0; c < dFf; c += 1) {
      ffnSubNorm[rowOff + c] = prodRow64[c] * invNorm;
    }
  }

  // 4. Bias-free Down projection O = M W_down^T [seqLen, d] (D5 write-back boundary)
  const downProjOut = linearProject(ffnSubNorm, weights.downProj, seqLen, dFf, d);

  // 5. Channel-wise nGPT NLERP residual blend (Anti-SLERP DR7; unclamped |alpha_hat| >= 0)
  let output: Float32Array;
  let nlerpTrace: DoryNlerpTrace | null = null;
  if (captureTrace) {
    const detailed = ngptNlerpResidualRowsDetailed(
      hiddenStates,
      downProjOut,
      weights.mlpAlpha,
      seqLen,
      d,
      config,
    );
    nlerpTrace = detailed;
    output = detailed.output;
  } else {
    output = ngptNlerpResidualRows(
      hiddenStates,
      downProjOut,
      weights.mlpAlpha,
      seqLen,
      d,
      config,
    );
  }

  const trace: DoryMlpSublayerTrace | null = captureTrace
    ? {
        kind: 'mlp',
        physicalLayerIdx,
        pairedLayerIdx: weights.pairedLayerIdx,
        stage: weights.stage,
        loopIteration,
        seqLen,
        gatePre,
        upPre,
        effectiveSuvGate,
        effectiveSuvUp,
        gateScaled: gateScaled!,
        upScaled: upScaled!,
        gateSilu: gateSilu!,
        swigluProduct: swigluProduct!,
        ffnSubNorm,
        downProjOut,
        nlerp: nlerpTrace!,
      }
    : null;

  return {
    output,
    sublayerOut: downProjOut,
    trace,
  };
}
