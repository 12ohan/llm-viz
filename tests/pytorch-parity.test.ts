import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { runGqaSim } from '../src/math/gqaSim';
import {
  applyRmsNorm,
  createDefaultHybridModel,
  embedTokens,
  MINI_VOCAB,
  runHybridExecution,
  runUntiedLmHead,
} from '../src/math/hybridFlow';
import { runMambaSim } from '../src/math/mambaSim';
import { runMoeSim } from '../src/math/moeSim';
import { computeModelWeightsSha256 } from './helpers/modelWeightDigest';
import {
  assertBitExactFiniteArray,
  assertCausalScoresParity,
  assertFiniteArrayParity,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';

interface ParityFixtureBundle {
  readonly metadata: {
    readonly generator: string;
    readonly pythonVersion: string;
    readonly torchVersion: string;
    readonly pinnedTorchVersion: string;
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

interface EndToEnd52LayerFixture {
  readonly metadata: {
    readonly generator: string;
    readonly pythonVersion: string;
    readonly torchVersion: string;
    readonly pinnedTorchVersion: string;
    readonly referenceArchitecture: string;
    readonly precisionNote: string;
    readonly tolerances: {
      readonly embeddingMaxAbs: number;
      readonly residualCheckpointMaxAbs: number;
      readonly finalNormMaxAbs: number;
      readonly lmHeadLogitsMaxAbs: number;
      readonly lmHeadProbsMaxAbs: number;
    };
  };
  readonly modelId: string;
  readonly parameterFingerprint: string;
  readonly weightsSha256: string;
  readonly totalWeightFloats: number;
  readonly pattern: string;
  readonly d: number;
  readonly vocabSize: number;
  readonly rmsNormEps: number;
  readonly tokenIds: number[];
  readonly residualStreams: number[][];
  readonly moeTopkByBay: Record<string, number[]>;
  readonly finalNorm: {
    readonly rms: number[];
    readonly X_norm: number[];
    readonly U: number[];
  };
  readonly lmHead: {
    readonly logits: number[];
    readonly probs: number[];
    readonly top5TokenIdsByToken: number[][];
  };
}

const SINGLE_FIXTURE_PATH = resolve(__dirname, 'fixtures/pytorch-reference-parity.json');
const E2E_52_FIXTURE_PATH = resolve(__dirname, 'fixtures/pytorch-52layer-parity.json');

const FIXTURE: ParityFixtureBundle = JSON.parse(readFileSync(SINGLE_FIXTURE_PATH, 'utf-8'));
const FIXTURE_52: EndToEnd52LayerFixture = JSON.parse(readFileSync(E2E_52_FIXTURE_PATH, 'utf-8'));

describe('Independent PyTorch Reference Numerical Parity (Gate A + Full 52-Layer)', () => {
  it('0. parity helpers explicitly reject unexpected NaN and ±Infinity while preserving intentional causal -Infinity masks', () => {
    expect(() =>
      computeStrictMaxAbsDiff(new Float32Array([1.0, Number.NaN, 2.0]), [1.0, 0.5, 2.0], 'nan-actual'),
    ).toThrow(/Non-finite actual/);
    expect(() =>
      computeStrictMaxAbsDiff(new Float32Array([1.0, 0.5, 2.0]), [1.0, Number.NaN, 2.0], 'nan-expected'),
    ).toThrow(/Non-finite expected/);
    expect(() =>
      assertFiniteArrayParity(new Float32Array([1.0, Infinity]), [1.0, 1.0], 1e-3, 'pos-inf'),
    ).toThrow(/Non-finite actual/);
    expect(() =>
      assertFiniteArrayParity(new Float32Array([1.0, -Infinity]), [1.0, 1.0], 1e-3, 'neg-inf'),
    ).toThrow(/Non-finite actual/);
    expect(() =>
      assertBitExactFiniteArray(new Float32Array([Number.NaN]), new Float32Array([Number.NaN]), 'bit-nan'),
    ).toThrow(/Non-finite/);

    // Intentional causal mask (-Infinity for null slots) passes, while NaN / +Inf / finite-in-mask / -Inf-in-unmasked throw!
    expect(
      assertCausalScoresParity(new Float32Array([0.5, -Infinity, 0.25, 0.75]), [0.5, null, 0.25, 0.75], 1e-6, 'valid-causal'),
    ).toBe(0);
    expect(() =>
      assertCausalScoresParity(new Float32Array([0.5, 0.0]), [0.5, null], 1e-6, 'finite-in-mask'),
    ).toThrow(/Expected intentional causal mask -Infinity/);
    expect(() =>
      assertCausalScoresParity(new Float32Array([Number.NaN, -Infinity]), [0.5, null], 1e-6, 'nan-in-unmasked'),
    ).toThrow(/Unexpected non-finite unmasked/);
    expect(() =>
      assertCausalScoresParity(new Float32Array([-Infinity, -Infinity]), [0.5, null], 1e-6, 'neginf-in-unmasked'),
    ).toThrow(/Unexpected non-finite unmasked/);
  });

  it('1. matches PyTorch NemotronHRMSNorm at eps=1e-5 and eps=1e-6 with exact epsilon-accounted variance identity', () => {
    const fx = FIXTURE.rmsnorm;
    const tol = FIXTURE.metadata.tolerances.rmsnormMaxAbs;
    const X = new Float32Array(fx.X);
    const gamma = new Float32Array(fx.gamma);

    for (const ref of [fx.eps_1e5, fx.eps_1e6]) {
      const out = applyRmsNorm(X, gamma, fx.d, fx.tokens, ref.eps);
      assertFiniteArrayParity(out.rms, ref.rms, tol, `rmsnorm(eps=${ref.eps}).rms`);
      assertFiniteArrayParity(out.X_norm, ref.X_norm, tol, `rmsnorm(eps=${ref.eps}).X_norm`);
      assertFiniteArrayParity(out.U, ref.U, tol, `rmsnorm(eps=${ref.eps}).U`);

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
    assertFiniteArrayParity(X_embed, fx.X_embed, FIXTURE.metadata.tolerances.embeddingMaxAbs, 'embedding.X_embed');
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
    assertFiniteArrayParity(fn.rms, fx.finalNormRms, tolLogits, 'finalNorm.rms');
    assertFiniteArrayParity(fn.U, fx.finalNormU, tolLogits, 'finalNorm.U');

    const lm = runUntiedLmHead(fn.U, new Float32Array(fx.W_lm_head), MINI_VOCAB, fx.d, fx.tokens, 5);
    assertFiniteArrayParity(lm.logits, fx.logits, tolLogits, 'lmHead.logits');
    assertFiniteArrayParity(lm.probs, fx.probs, tolProbs, 'lmHead.probs');
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
    assertFiniteArrayParity(sim.U, fx.U, tol, 'gqa.U');
    assertFiniteArrayParity(sim.Q, fx.Q, tol, 'gqa.Q');
    assertFiniteArrayParity(sim.K, fx.K, tol, 'gqa.K');
    assertFiniteArrayParity(sim.V, fx.V, tol, 'gqa.V');
    assertCausalScoresParity(sim.Scores, fx.Scores, tol, 'gqa.Scores');
    assertFiniteArrayParity(sim.Softmax, fx.Softmax, tol, 'gqa.Softmax');
    assertFiniteArrayParity(sim.V_out, fx.V_out, tol, 'gqa.V_out');
    assertFiniteArrayParity(sim.Attn_out, fx.Attn_out, tol, 'gqa.Attn_out');
    assertFiniteArrayParity(sim.Y, fx.Y, tol, 'gqa.Y');
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

    assertFiniteArrayParity(sim.U, fx.U, tol, 'mamba2.U');
    assertFiniteArrayParity(sim.conv_out, fx.conv_out, tol, 'mamba2.conv_out');
    assertFiniteArrayParity(sim.dt, fx.dt, tol, 'mamba2.dt');
    assertFiniteArrayParity(sim.dA, fx.dA, tol, 'mamba2.dA');
    assertFiniteArrayParity(sim.H_prev, fx.H_prev, tol, 'mamba2.H_prev');
    assertFiniteArrayParity(sim.H, fx.H, tol, 'mamba2.H');
    assertFiniteArrayParity(sim.y, fx.y, tol, 'mamba2.y');
    assertFiniteArrayParity(sim.y_norm, fx.y_norm, tol, 'mamba2.y_norm');
    assertFiniteArrayParity(sim.Mamba_out, fx.Mamba_out, tol, 'mamba2.Mamba_out');
    assertFiniteArrayParity(sim.Y, fx.Y, tol, 'mamba2.Y');
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

    assertFiniteArrayParity(sim.U, fx.U, tol, 'moe.U');
    assertFiniteArrayParity(sim.scores, fx.scores, tol, 'moe.scores');
    assertFiniteArrayParity(sim.scores_for_choice, fx.scores_for_choice, tol, 'moe.scores_for_choice');
    expect(Array.from(sim.topk_indices)).toEqual(fx.topk_indices);
    assertFiniteArrayParity(sim.topk_raw_scores, fx.topk_raw_scores, tol, 'moe.topk_raw_scores');
    assertFiniteArrayParity(sim.topk_weights, fx.topk_weights, tol, 'moe.topk_weights');
    assertFiniteArrayParity(sim.shared_a, fx.shared_a, tol, 'moe.shared_a');
    assertFiniteArrayParity(sim.y_shared, fx.y_shared, tol, 'moe.y_shared');
    assertFiniteArrayParity(sim.y_routed, fx.y_routed, tol, 'moe.y_routed');
    assertFiniteArrayParity(sim.y_moe, fx.y_moe, tol, 'moe.y_moe');
    assertFiniteArrayParity(sim.Y, fx.Y, tol, 'moe.Y');
  });

  it('6. matches PyTorch 2.15.0 end-to-end across all 52 layers (53 residual checkpoints, 23 MoE routers, Final RMSNorm, and Untied LM Head)', () => {
    const model = createDefaultHybridModel();
    const digest = computeModelWeightsSha256(model);

    // Cryptographic proof that TypeScript and PyTorch ran on bit-identical 52-layer IEEE-754 weights
    expect(FIXTURE_52.metadata.pinnedTorchVersion).toBe('2.15.0');
    expect(model.modelId).toBe(FIXTURE_52.modelId);
    expect(model.parameterFingerprint).toBe(FIXTURE_52.parameterFingerprint);
    expect(digest.totalFloats).toBe(FIXTURE_52.totalWeightFloats);
    expect(digest.sha256).toBe(FIXTURE_52.weightsSha256);

    const exec = runHybridExecution(model, { tokenIds: FIXTURE_52.tokenIds });
    const tol = FIXTURE_52.metadata.tolerances;

    // 1. All 53 residual checkpoints (X_0 .. X_52) match PyTorch within documented cumulative tolerances
    expect(exec.residualStreams).toHaveLength(53);
    expect(FIXTURE_52.residualStreams).toHaveLength(53);
    assertBitExactFiniteArray(exec.residualStreams[0], FIXTURE_52.residualStreams[0], 'e2e.X_embed');
    for (let k = 1; k <= 52; k++) {
      assertFiniteArrayParity(
        exec.residualStreams[k],
        FIXTURE_52.residualStreams[k],
        tol.residualCheckpointMaxAbs,
        `e2e.residualStreams[${k}]`,
      );
    }

    // 2. All 23 MoE layers match PyTorch Top-6 routed expert indices across all 6 tokens
    for (const [bayKey, refTopk] of Object.entries(FIXTURE_52.moeTopkByBay)) {
      const bayId = Number(bayKey);
      const summary = exec.baySummaries[bayId - 1];
      expect(summary.kind).toBe('E');
      expect(Array.from(summary.moeTopkIndices!)).toEqual(refTopk);
    }

    // 3. Final RMSNorm & Untied LM Head logits, probabilities, and Top-5 token rankings match PyTorch
    assertFiniteArrayParity(exec.finalNorm.rms, FIXTURE_52.finalNorm.rms, tol.finalNormMaxAbs, 'e2e.finalNorm.rms');
    assertFiniteArrayParity(exec.finalNorm.X_norm, FIXTURE_52.finalNorm.X_norm, tol.finalNormMaxAbs, 'e2e.finalNorm.X_norm');
    assertFiniteArrayParity(exec.finalNorm.U_final, FIXTURE_52.finalNorm.U, tol.finalNormMaxAbs, 'e2e.finalNorm.U_final');
    assertFiniteArrayParity(exec.lmHead.logits, FIXTURE_52.lmHead.logits, tol.lmHeadLogitsMaxAbs, 'e2e.lmHead.logits');
    assertFiniteArrayParity(exec.lmHead.probs, FIXTURE_52.lmHead.probs, tol.lmHeadProbsMaxAbs, 'e2e.lmHead.probs');

    for (let t = 0; t < FIXTURE_52.tokenIds.length; t++) {
      const actualTop5 = exec.lmHead.topKByToken[t].map((p) => p.tokenId);
      expect(actualTop5, `e2e.top5[t=${t}]`).toEqual(FIXTURE_52.lmHead.top5TokenIdsByToken[t]);
    }
  });
});
