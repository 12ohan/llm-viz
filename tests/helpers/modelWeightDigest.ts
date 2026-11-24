import { createHash } from 'node:crypto';
import type { HybridModelParameters } from '../../src/math/hybridFlow';

export interface CanonicalModelWeightTensor {
  readonly name: string;
  readonly data: Float32Array;
}

/**
 * Canonical ordered list of every Float32 parameter tensor owned by a `HybridModelParameters`
 * instance (`W_embed`, `finalNormGamma`, `W_lm_head`, followed by bays `1..L` in fixed field order).
 * Used identically by `scripts/reference/export_52layer_params.mjs` and `tests/pytorch-parity.test.ts`
 * to prove cryptographic SHA-256 bit-identity of all 52 layers' weights across TypeScript and PyTorch.
 */
export function collectCanonicalModelWeightTensors(
  model: HybridModelParameters,
): CanonicalModelWeightTensor[] {
  const list: CanonicalModelWeightTensor[] = [
    { name: 'W_embed', data: model.W_embed },
    { name: 'finalNormGamma', data: model.finalNormGamma },
    { name: 'W_lm_head', data: model.W_lm_head },
  ];
  const L = model.config.pattern.length;
  for (let bayId = 1; bayId <= L; bayId++) {
    const w = model.getBayWeights(bayId);
    const p = `bay_${String(bayId).padStart(2, '0')}`;
    if (w.kind === 'M') {
      list.push(
        { name: `${p}.gamma_in`, data: w.gamma_in },
        { name: `${p}.W_in`, data: w.W_in },
        { name: `${p}.w_conv`, data: w.w_conv },
        { name: `${p}.b_conv`, data: w.b_conv },
        { name: `${p}.A_log`, data: w.A_log },
        { name: `${p}.dt_bias`, data: w.dt_bias },
        { name: `${p}.D`, data: w.D },
        { name: `${p}.gamma_gate`, data: w.gamma_gate },
        { name: `${p}.W_out`, data: w.W_out },
      );
    } else if (w.kind === 'E') {
      list.push(
        { name: `${p}.gamma_in`, data: w.gamma_in },
        { name: `${p}.W_router`, data: w.W_router },
        { name: `${p}.e_score_correction_bias`, data: w.e_score_correction_bias },
        { name: `${p}.W_up`, data: w.W_up },
        { name: `${p}.W_down`, data: w.W_down },
        { name: `${p}.W_up_s`, data: w.W_up_s },
        { name: `${p}.W_down_s`, data: w.W_down_s },
      );
    } else {
      list.push(
        { name: `${p}.gamma`, data: w.gamma },
        { name: `${p}.W_Q`, data: w.W_Q },
        { name: `${p}.W_K`, data: w.W_K },
        { name: `${p}.W_V`, data: w.W_V },
        { name: `${p}.W_O`, data: w.W_O },
      );
    }
  }
  return list;
}

/** Compute canonical SHA-256 hex digest over little-endian IEEE-754 bytes of all model weights. */
export function computeModelWeightsSha256(model: HybridModelParameters): {
  readonly sha256: string;
  readonly totalFloats: number;
  readonly totalBytes: number;
} {
  const hash = createHash('sha256');
  const tensors = collectCanonicalModelWeightTensors(model);
  let totalFloats = 0;
  for (const t of tensors) {
    // Emit explicit little-endian float32 bytes for cross-platform determinism
    const buf = Buffer.allocUnsafe(t.data.length * 4);
    for (let i = 0; i < t.data.length; i++) {
      buf.writeFloatLE(t.data[i], i * 4);
    }
    hash.update(buf);
    totalFloats += t.data.length;
  }
  return {
    sha256: hash.digest('hex'),
    totalFloats,
    totalBytes: totalFloats * 4,
  };
}
