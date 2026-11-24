import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { runGqaSim } from '../src/math/gqaSim';
import {
  applyRmsNorm,
  embedTokens,
  MINI_VOCAB,
  runUntiedLmHead,
} from '../src/math/hybridFlow';
import { runMambaSim } from '../src/math/mambaSim';
import { runMoeSim } from '../src/math/moeSim';

interface ParityFixtureBundle {
  readonly metadata: {
    readonly generator: string;
    readonly torchVersion: string;
    readonly referenceArchitecture: string;
    readonly precisionNote: string;
    readonly tolerances: {
      readonly rmsnormMaxAbs: number;
      readonly embeddingMaxAbs: number;
      readonly lmHeadLogitsMaxAbs: number;
      readonly lmHeadProbsMaxAbs: number;
      readonly gqaMaxAbs: number;
      readonly mamba2MaxAbs: number;
      readonly moeMaxAbs: number;
    };
  };
  readonly rmsnorm: {
    readonly d: number;
    readonly tokens: number;
    readonly X: number[];
    readonly gamma: number[];
    readonly eps_1e5: { readonly eps: number; readonly rms: number[]; readonly X_norm: number[]; readonly U: number[] };
    readonly eps_1e6: { readonly eps: number; readonly rms: number[]; readonly X_norm: number[]; readonly U: number[] };
  };
  readonly embeddingAndLmHead: {
    readonly vocabSize: number;
    readonly d: number;
    readonly tokens: number;
    readonly eps: number;
    readonly tokenIds: number[];
    readonly W_embed: number[];
    readonly X_embed: number[];
    readonly X_final: number[];
    readonly finalNormGamma: number[];
    readonly finalNormRms: number[];
    readonly finalNormU: number[];
    readonly W_lm_head: number[];
    readonly logits: number[];
    readonly probs: number[];
  };
  readonly gqa: {
    readonly d: number;
    readonly tokens: number;
    readonly eps: number;
    readonly X: number[];
    readonly gamma: number[];
    readonly W_Q: number[];
    readonly W_K: number[];
    readonly W_V: number[];
    readonly W_O: number[];
    readonly U: number[];
    readonly Q: number[];
    readonly K: number[];
    readonly V: number[];
    readonly Scores: Array<number | null>;
    readonly Softmax: number[];
    readonly V_out: number[];
    readonly Attn_out: number[];
    readonly Y: number[];
  };
  readonly mamba2: {
    readonly d: number;
    readonly tokens: number;
    readonly eps: number;
    readonly X: number[];
    readonly gamma_in: number[];
    readonly W_in: number[];
    readonly w_conv: number[];
    readonly b_conv: number[];
    readonly A_log: number[];
    readonly dt_bias: number[];
    readonly D: number[];
    readonly gamma_gate: number[];
    readonly W_out: number[];
    readonly U: number[];
    readonly conv_out: number[];
    readonly dt: number[];
    readonly dA: number[];
    readonly H_prev: number[];
    readonly H: number[];
    readonly y: number[];
    readonly y_norm: number[];
    readonly Mamba_out: number[];
    readonly Y: number[];
  };
  readonly moe: {
    readonly d: number;
    readonly tokens: number;
    readonly eps: number;
    readonly X: number[];
    readonly gamma_in: number[];
    readonly W_router: number[];
    readonly e_score_correction_bias: number[];
    readonly W_up: number[];
    readonly W_down: number[];
    readonly W_up_s: number[];
    readonly W_down_s: number[];
    readonly U: number[];
    readonly scores: number[];
    readonly scores_for_choice: number[];
    readonly topk_indices: number[];
    readonly topk_raw_scores: number[];
    readonly topk_weights: number[];
    readonly shared_a: number[];
    readonly y_shared: number[];
    readonly y_routed: number[];
    readonly y_moe: number[];
    readonly Y: number[];
  };
}

const FIXTURE_PATH = resolve(__dirname, 'fixtures/pytorch-reference-parity.json');
const FIXTURE: ParityFixtureBundle = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8'));

function maxAbsDiff(actual: ArrayLike<number>, expected: readonly number[]): number {
  expect(actual.length).toBe(expected.length);
  let maxErr = 0;
  for (let i = 0; i < actual.length; i++) {
    const err = Math.abs(actual[i] - expected[i]);
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}

describe('Independent PyTorch Reference Numerical Parity (Gate A)', () => {
  it('1. matches PyTorch NemotronHRMSNorm at eps=1e-5 and eps=1e-6 with exact epsilon-accounted variance identity', () => {
    const fx = FIXTURE.rmsnorm;
    const tol = FIXTURE.metadata.tolerances.rmsnormMaxAbs;
    const X = new Float32Array(fx.X);
    const gamma = new Float32Array(fx.gamma);

    for (const ref of [fx.eps_1e5, fx.eps_1e6]) {
      const out = applyRmsNorm(X, gamma, fx.d, fx.tokens, ref.eps);
      expect(maxAbsDiff(out.rms, ref.rms)).toBeLessThanOrEqual(tol);
      expect(maxAbsDiff(out.X_norm, ref.X_norm)).toBeLessThanOrEqual(tol);
      expect(maxAbsDiff(out.U, ref.U)).toBeLessThanOrEqual(tol);

      // Exact epsilon-accounted RMSNorm identity: mean(X_norm[:,t]^2) = mean(X[:,t]^2) / (mean(X[:,t]^2) + eps)
      for (let t = 0; t < fx.tokens; t++) {
        let meanSqX = 0;
        let meanSqNorm = 0;
        for (let i = 0; i < fx.d; i++) {
          const xv = X[i * fx.tokens + t];
          const nv = out.X_norm[i * fx.tokens + t];
          meanSqX += xv * xv;
          meanSqNorm += nv * nv;
        }
        meanSqX /= fx.d;
        meanSqNorm /= fx.d;
        const expectedMeanSqNorm = meanSqX / (meanSqX + ref.eps);
        expect(Math.abs(meanSqNorm - expectedMeanSqNorm)).toBeLessThan(1e-5);
      }
    }
  });

  it('2. matches PyTorch F.embedding (NoPE), Final RMSNorm, and Untied LM Head + Softmax', () => {
    const fx = FIXTURE.embeddingAndLmHead;
    const tolLogits = FIXTURE.metadata.tolerances.lmHeadLogitsMaxAbs;
    const tolProbs = FIXTURE.metadata.tolerances.lmHeadProbsMaxAbs;

    const W_embed = new Float32Array(fx.W_embed);
    const X_embed = embedTokens(fx.tokenIds, W_embed, fx.d);
    expect(maxAbsDiff(X_embed, fx.X_embed)).toBe(FIXTURE.metadata.tolerances.embeddingMaxAbs);
    // Repeated token ID 2 at t=0 and t=2 produces bit-identical NoPE embeddings
    for (let i = 0; i < fx.d; i++) {
      expect(X_embed[i * fx.tokens + 0]).toBe(X_embed[i * fx.tokens + 2]);
    }

    const fn = applyRmsNorm(
      new Float32Array(fx.X_final),
      new Float32Array(fx.finalNormGamma),
      fx.d,
      fx.tokens,
      fx.eps,
    );
    expect(maxAbsDiff(fn.rms, fx.finalNormRms)).toBeLessThanOrEqual(tolLogits);
    expect(maxAbsDiff(fn.U, fx.finalNormU)).toBeLessThanOrEqual(tolLogits);

    const lm = runUntiedLmHead(fn.U, new Float32Array(fx.W_lm_head), MINI_VOCAB, fx.d, fx.tokens, 5);
    expect(maxAbsDiff(lm.logits, fx.logits)).toBeLessThanOrEqual(tolLogits);
    expect(maxAbsDiff(lm.probs, fx.probs)).toBeLessThanOrEqual(tolProbs);
  });

  it('3. matches PyTorch 32Q/2KV NoPE GQA Attention with eps=1e-5 across all intermediate tensors', () => {
    const fx = FIXTURE.gqa;
    const tol = FIXTURE.metadata.tolerances.gqaMaxAbs;

    const sim = runGqaSim(fx.tokens, {
      X: new Float32Array(fx.X),
      gamma: new Float32Array(fx.gamma),
      W_Q: new Float32Array(fx.W_Q),
      W_K: new Float32Array(fx.W_K),
      W_V: new Float32Array(fx.W_V),
      W_O: new Float32Array(fx.W_O),
      rmsEps: fx.eps,
    });

    expect(sim.rmsEps).toBe(fx.eps);
    expect(maxAbsDiff(sim.U, fx.U)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.Q, fx.Q)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.K, fx.K)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.V, fx.V)).toBeLessThanOrEqual(tol);

    // Causal masked positions match -Infinity exactly; unmasked scores match PyTorch within tolerance
    expect(sim.Scores.length).toBe(fx.Scores.length);
    for (let i = 0; i < fx.Scores.length; i++) {
      const refVal = fx.Scores[i];
      if (refVal === null) {
        expect(sim.Scores[i]).toBe(-Infinity);
      } else {
        expect(Math.abs(sim.Scores[i] - refVal)).toBeLessThanOrEqual(tol);
      }
    }

    expect(maxAbsDiff(sim.Softmax, fx.Softmax)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.V_out, fx.V_out)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.Attn_out, fx.Attn_out)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.Y, fx.Y)).toBeLessThanOrEqual(tol);
  });

  it('4. matches PyTorch NemotronHMamba2Mixer + MambaRMSNormGated (norm_before_gate=False) across all states', () => {
    const fx = FIXTURE.mamba2;
    const tol = FIXTURE.metadata.tolerances.mamba2MaxAbs;

    const sim = runMambaSim(fx.tokens, {
      X: new Float32Array(fx.X),
      gamma_in: new Float32Array(fx.gamma_in),
      W_in: new Float32Array(fx.W_in),
      w_conv: new Float32Array(fx.w_conv),
      b_conv: new Float32Array(fx.b_conv),
      A_log: new Float32Array(fx.A_log),
      dt_bias: new Float32Array(fx.dt_bias),
      D: new Float32Array(fx.D),
      gamma_gate: new Float32Array(fx.gamma_gate),
      W_out: new Float32Array(fx.W_out),
      eps: fx.eps,
    });

    expect(maxAbsDiff(sim.U, fx.U)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.conv_out, fx.conv_out)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.dt, fx.dt)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.dA, fx.dA)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.H_prev, fx.H_prev)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.H, fx.H)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.y, fx.y)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.y_norm, fx.y_norm)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.Mamba_out, fx.Mamba_out)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.Y, fx.Y)).toBeLessThanOrEqual(tol);
  });

  it('5. matches PyTorch NemotronHMOE (128+1, biased selection, unbiased Top-6 x2.5, non-gated ReLU^2, 2x shared)', () => {
    const fx = FIXTURE.moe;
    const tol = FIXTURE.metadata.tolerances.moeMaxAbs;

    const sim = runMoeSim(fx.tokens, {
      X: new Float32Array(fx.X),
      gamma_in: new Float32Array(fx.gamma_in),
      W_router: new Float32Array(fx.W_router),
      e_score_correction_bias: new Float32Array(fx.e_score_correction_bias),
      W_up: new Float32Array(fx.W_up),
      W_down: new Float32Array(fx.W_down),
      W_up_s: new Float32Array(fx.W_up_s),
      W_down_s: new Float32Array(fx.W_down_s),
      rmsNormEps: fx.eps,
    });

    expect(maxAbsDiff(sim.U, fx.U)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.scores, fx.scores)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.scores_for_choice, fx.scores_for_choice)).toBeLessThanOrEqual(tol);
    // Exact Top-6 routed expert selection match across all tokens!
    expect(Array.from(sim.topk_indices)).toEqual(fx.topk_indices);
    expect(maxAbsDiff(sim.topk_raw_scores, fx.topk_raw_scores)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.topk_weights, fx.topk_weights)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.shared_a, fx.shared_a)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.y_shared, fx.y_shared)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.y_routed, fx.y_routed)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.y_moe, fx.y_moe)).toBeLessThanOrEqual(tol);
    expect(maxAbsDiff(sim.Y, fx.Y)).toBeLessThanOrEqual(tol);
  });
});
