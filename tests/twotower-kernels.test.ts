import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  describeCellFormula,
  GQA_SIM_CONFIG,
  isAttentionKeyVisible,
  maxVisibleKeyIndex,
  runGqaSim,
  type LayerModulationTerms,
} from '../src/math/gqaSim';
import {
  applyRmsNorm,
  computeModelParameterFingerprint,
  createDefaultHybridModel,
  createHybridModelParameters,
  getOrRunHybridExecution,
  HybridExecutionCache,
  MINI_VOCAB,
  runHybridExecution,
  type GqaBayWeights,
  type MambaBayWeights,
  type MoeBayWeights,
} from '../src/math/hybridFlow';
import {
  createZeroDecodeCacheState,
  measureDecodeCacheBytes,
  runIncrementalDecodeStep,
  validateDecodeCacheState,
} from '../src/math/incrementalDecode';
import {
  describeMambaCellFormula,
  MAMBA_SIM_CONFIG,
  runMambaSim,
} from '../src/math/mambaSim';
import {
  describeMoeCellFormula,
  evaluateSingleRoutedExpert,
  MOE_SIM_CONFIG,
  runMoeSim,
  stableSigmoid,
} from '../src/math/moeSim';
import {
  computeGqaFocusedAccumulation,
  computeMoeFocusedAccumulation,
} from '../src/scene/accumulation';
import { computeModelWeightsSha256 } from './helpers/modelWeightDigest';
import { assertBitExactFiniteArray } from './helpers/parityAssert';

const E2E_52_FIXTURE_PATH = resolve(__dirname, 'fixtures/pytorch-52layer-parity.json');
const FIXTURE_52 = JSON.parse(readFileSync(E2E_52_FIXTURE_PATH, 'utf-8')) as {
  readonly modelId: string;
  readonly parameterFingerprint: string;
  readonly weightsSha256: string;
  readonly totalWeightFloats: number;
};

function makeModulation(shift: number[], scale: number[]): LayerModulationTerms {
  return {
    shift: new Float32Array(shift),
    scale: new Float32Array(scale),
  };
}

function modulateChannelMajor(
  src: Float32Array,
  mod: LayerModulationTerms,
  d: number,
  T: number,
): Float32Array {
  const out = new Float32Array(d * T);
  for (let i = 0; i < d; i++) {
    const scaleFactor = Math.fround(1 + mod.scale[i]);
    const shift = mod.shift[i];
    for (let t = 0; t < T; t++) {
      out[i * T + t] = Math.fround(Math.fround(src[i * T + t] * scaleFactor) + shift);
    }
  }
  return out;
}

describe('Stage TT1a: Additive Shared-Kernel Extensions for Nemotron-Labs-TwoTower', () => {
  describe('1. Nano v1.0.0 0-ULP Invariance & Fingerprint Backward Compatibility', () => {
    it('preserves exact Nano v1.0.0 modelId, parameterFingerprint, SHA-256 weight digest, and 75.00 KiB cache accounting', () => {
      const model = createDefaultHybridModel();
      const digest = computeModelWeightsSha256(model);

      expect(model.modelId).toBe(FIXTURE_52.modelId);
      expect(model.parameterFingerprint).toBe(FIXTURE_52.parameterFingerprint);
      expect(digest.totalFloats).toBe(FIXTURE_52.totalWeightFloats);
      expect(digest.sha256).toBe(FIXTURE_52.weightsSha256);

      expect(model.config.vocabSize).toBe(32);
      expect(model.config.routerPrecisionPolicy).toBe('float32_nano');
      expect(model.config.perturbation).toBeUndefined();
      expect(model.config.generationPolicy).toBeUndefined();

      // Explicit default routerPrecisionPolicy produces the exact same Nano v1.0.0 fingerprint
      const fpDefaultExplicit = computeModelParameterFingerprint({
        seed: model.config.seed,
        pattern: model.config.pattern,
        d: model.config.d,
        vocabSize: model.config.vocabSize,
        rmsNormEps: model.config.rmsNormEps,
        residualOutputScale: model.config.residualOutputScale,
        vocab: model.vocab,
        routerPrecisionPolicy: 'float32_nano',
      });
      expect(fpDefaultExplicit).toBe(FIXTURE_52.parameterFingerprint);

      // Exact 75.00 KiB default cache snapshot accounting remains unchanged
      const zeroState = createZeroDecodeCacheState(model);
      const acct = measureDecodeCacheBytes(zeroState);
      expect(acct.totalAllocatedBytes).toBe(76_800);
      expect(acct.mambaFixedBytes).toBe(70_656);
      expect(acct.gqaAllocatedBytes).toBe(6_144);
      expect(acct.moeBytes).toBe(0);
    });

    it('matches default Nano sims bit-for-bit (0 ULP) under explicit default policies and zero-mod / unit-gate identity', () => {
      const gqaDef = runGqaSim(6);
      expect(gqaDef.attentionMaskMode).toBe('causal');
      expect(gqaDef.X_mod).toBeUndefined();
      expect(gqaDef.residualGateApplied).toBe(false);
      const gqaExplicit = runGqaSim(6, {
        attentionMaskMode: 'causal',
        preNormMod: makeModulation(new Array(8).fill(0), new Array(8).fill(0)),
        residualGate: new Float32Array(8).fill(1),
      });
      assertBitExactFiniteArray(gqaExplicit.X, gqaDef.X, 'gqa.X');
      assertBitExactFiniteArray(gqaExplicit.X_mod!, gqaDef.X, 'gqa.X_mod zero-mod');
      assertBitExactFiniteArray(gqaExplicit.U, gqaDef.U, 'gqa.U');
      assertBitExactFiniteArray(gqaExplicit.Y, gqaDef.Y, 'gqa.Y');

      const mambaDef = runMambaSim(6);
      expect(mambaDef.X_mod).toBeUndefined();
      expect(mambaDef.residualGateApplied).toBe(false);
      const mambaExplicit = runMambaSim(6, {
        preNormMod: makeModulation(new Array(8).fill(0), new Array(8).fill(0)),
        residualGate: new Float32Array(8).fill(1),
      });
      assertBitExactFiniteArray(mambaExplicit.X, mambaDef.X, 'mamba.X');
      assertBitExactFiniteArray(mambaExplicit.X_mod!, mambaDef.X, 'mamba.X_mod zero-mod');
      assertBitExactFiniteArray(mambaExplicit.U, mambaDef.U, 'mamba.U');
      assertBitExactFiniteArray(mambaExplicit.H, mambaDef.H, 'mamba.H');
      assertBitExactFiniteArray(mambaExplicit.Y, mambaDef.Y, 'mamba.Y');

      const moeDef = runMoeSim(6);
      expect(moeDef.routerPrecisionPolicy).toBe('float32_nano');
      expect(moeDef.U_mod).toBeUndefined();
      expect(moeDef.router_logits_f64).toBeUndefined();
      expect(moeDef.router_scores_f64).toBeUndefined();
      expect(moeDef.scores_for_choice_f64).toBeUndefined();
      expect(moeDef.residualGateApplied).toBe(false);
      const moeExplicit = runMoeSim(6, {
        routerPrecisionPolicy: 'float32_nano',
        postNormMod: makeModulation(new Array(8).fill(0), new Array(8).fill(0)),
        residualGate: new Float32Array(8).fill(1),
      });
      assertBitExactFiniteArray(moeExplicit.X, moeDef.X, 'moe.X');
      assertBitExactFiniteArray(moeExplicit.U_mod!, moeDef.U, 'moe.U_mod zero-mod');
      expect(Array.from(moeExplicit.topk_indices)).toEqual(Array.from(moeDef.topk_indices));
      assertBitExactFiniteArray(moeExplicit.topk_weights, moeDef.topk_weights, 'moe.topk_weights');
      assertBitExactFiniteArray(moeExplicit.Y, moeDef.Y, 'moe.Y');
    });
  });

  describe('2. MoE Router Precision Policy (float32_nano vs float64_reference) & Deterministic Tie-Breaking', () => {
    it('distinguishes ~1e-8 rank-6 vs rank-7 near-tie under float64_reference where premature Float32 rounding collapses to lower index', () => {
      const { d, nRoutedExperts, topK, routedScalingFactor, probDenomEps } = MOE_SIM_CONFIG;
      const X = new Float32Array(d).fill(1.0);
      const gamma_in = new Float32Array(d).fill(1.0);
      const W_router = new Float32Array(nRoutedExperts * d);
      const bias = new Float32Array(nRoutedExperts);

      // Experts 0..4 clearly win slots #0..#4
      for (let e = 0; e < 5; e++) {
        W_router[e * d + 0] = Math.fround(3.0 - e * 0.1);
        bias[e] = 0.25;
      }

      // Contender A (expert 5, lower index): z64 = 0, sigmoid64 = 0.5, bias = 0.5 -> choice64 = 1.0
      W_router[5 * d + 0] = 0.0;
      bias[5] = 0.5;

      // Contender B (expert 6, higher index): z64 ≈ 4e-8 -> sigmoid64 ≈ 0.5 + 1e-8 -> choice64 ≈ 1.0 + 1e-8
      // At 0.5, Float32 ULP is 2^-24 ≈ 5.96e-8 (half-ULP ≈ 2.98e-8 > 1e-8), so Math.fround(sigmoid(l32))
      // collapses to 0.5 under float32_nano and tie-breaks to lower index 5, while float64_reference selects 6!
      W_router[6 * d + 1] = Math.fround(4.0e-8);
      bias[6] = 0.5;

      // Experts 7..127 clearly lose (including an extreme negative logit at expert 127 to verify stableSigmoid64)
      for (let e = 7; e < nRoutedExperts; e++) {
        W_router[e * d + 0] = e === 127 ? -1000.0 : -4.0;
        bias[e] = -0.25;
      }

      const sim32 = runMoeSim(1, {
        X,
        gamma_in,
        W_router,
        e_score_correction_bias: bias,
        routerPrecisionPolicy: 'float32_nano',
      });
      const sim64 = runMoeSim(1, {
        X,
        gamma_in,
        W_router,
        e_score_correction_bias: bias,
        routerPrecisionPolicy: 'float64_reference',
      });

      // Float32 router collapsed expert 6's sigmoid to 0.5 -> tie with expert 5 -> lower index 5 selected
      expect(sim32.scores_for_choice[5]).toBe(sim32.scores_for_choice[6]);
      expect(Array.from(sim32.topk_indices)).toEqual([0, 1, 2, 3, 4, 5]);

      // Float64 router preserves the ~1e-8 difference and cleanly selects expert 6!
      expect(sim64.routerPrecisionPolicy).toBe('float64_reference');
      expect(sim64.router_logits_f64).toBeInstanceOf(Float64Array);
      expect(sim64.router_scores_f64).toBeInstanceOf(Float64Array);
      expect(sim64.scores_for_choice_f64).toBeInstanceOf(Float64Array);

      const deltaChoice64 = sim64.scores_for_choice_f64![6] - sim64.scores_for_choice_f64![5];
      expect(deltaChoice64).toBeGreaterThan(5e-9);
      expect(deltaChoice64).toBeLessThan(2e-8);
      expect(Array.from(sim64.topk_indices)).toEqual([0, 1, 2, 3, 4, 6]);

      // Extreme negative logit (-1000) stays finite (0) with zero NaN/Infinity overflow
      expect(Number.isFinite(sim64.router_scores_f64![127])).toBe(true);
      expect(sim64.router_scores_f64![127]).toBe(0);

      // evaluateSingleRoutedExpert agrees with Float64 choiceRank
      const ev6 = evaluateSingleRoutedExpert(sim64, 6, 0);
      const ev5 = evaluateSingleRoutedExpert(sim64, 5, 0);
      expect(ev6.isActive).toBe(true);
      expect(ev6.activeRank).toBe(5);
      expect(ev6.choiceRank).toBe(5);
      expect(ev5.isActive).toBe(false);
      expect(ev5.choiceRank).toBe(6);

      // Verify Top-6 weights match pure Float64 normalization + *2.5 rounded ONLY at final Float32 write
      let sum64 = 0;
      for (let j = 0; j < topK; j++) {
        const eid = sim64.topk_indices[j];
        sum64 += stableSigmoid(sim64.router_logits_f64![eid]);
      }
      for (let j = 0; j < topK; j++) {
        const eid = sim64.topk_indices[j];
        const expectedW = Math.fround(
          (stableSigmoid(sim64.router_logits_f64![eid]) / (sum64 + probDenomEps)) *
            routedScalingFactor,
        );
        expect(sim64.topk_weights[j]).toBe(expectedW);
      }

      // Focused MoE accumulation on final term also matches Float64 weight bit-exact
      const acc = computeMoeFocusedAccumulation(
        sim64,
        { tensor: 'topk_weights', row: 0, col: 5, tokenIndex: 0, rankSlot: 5 },
        2,
        5.5 / 6,
        0,
        6,
      );
      expect(acc).not.toBeNull();
      expect(acc!.isFinal).toBe(true);
      expect(acc!.partialSum).toBe(sim64.topk_weights[5]);
    });

    it('breaks exact Float64 choice-score ties deterministically in favor of the lower expert index', () => {
      const { d, nRoutedExperts } = MOE_SIM_CONFIG;
      const X = new Float32Array(d).fill(1.0);
      const gamma_in = new Float32Array(d).fill(1.0);
      const W_router = new Float32Array(nRoutedExperts * d);
      const bias = new Float32Array(nRoutedExperts);

      for (let e = 0; e < nRoutedExperts; e++) {
        W_router[e * d + 0] = -4.0;
        bias[e] = -0.25;
      }
      // 5 clear winners at high indices
      for (const e of [100, 101, 102, 103, 104]) {
        W_router[e * d + 0] = 3.0;
        bias[e] = 0.25;
      }
      // Exact tie among experts 42, 17, 88 for the 6th slot -> expert 17 MUST win
      for (const e of [42, 17, 88]) {
        W_router[e * d + 0] = 0.75;
        W_router[e * d + 3] = -0.25;
        bias[e] = 0.125;
      }

      const sim64 = runMoeSim(1, {
        X,
        gamma_in,
        W_router,
        e_score_correction_bias: bias,
        routerPrecisionPolicy: 'float64_reference',
      });

      expect(sim64.scores_for_choice_f64![17]).toBe(sim64.scores_for_choice_f64![42]);
      expect(sim64.scores_for_choice_f64![17]).toBe(sim64.scores_for_choice_f64![88]);
      expect(sim64.topk_indices[5]).toBe(17);
      expect(evaluateSingleRoutedExpert(sim64, 17, 0).choiceRank).toBe(5);
      expect(evaluateSingleRoutedExpert(sim64, 42, 0).choiceRank).toBe(6);
      expect(evaluateSingleRoutedExpert(sim64, 88, 0).choiceRank).toBe(7);
    });
  });

  describe('3. Asymmetric Modulation Order (M/* Pre-Norm vs E Post-Norm), Residual Provenance & Validation', () => {
    const mod = makeModulation(
      [0.35, -0.25, 0.42, -0.18, 0.55, -0.40, 0.22, -0.31],
      [0.30, -0.22, 0.45, -0.15, 0.25, -0.35, 0.18, 0.40],
    );
    const gate = new Float32Array([0.5, -0.25, 0.75, 0.1, 1.2, 0.0, -0.6, 0.85]);

    it('applies pre-RMSNorm modulation (modulate -> RMSNorm) and gated residual on original X in Mamba-2 (M)', () => {
      const T = 4;
      const d = MAMBA_SIM_CONFIG.d;
      const base = runMambaSim(T);
      const sim = runMambaSim(T, {
        X: base.X,
        preNormMod: mod,
        residualGate: gate,
      });

      // Original residual X stays untouched
      assertBitExactFiniteArray(sim.X, base.X, 'mamba.X untouched');
      expect(sim.X_mod).toBeDefined();
      expect(sim.residualGateApplied).toBe(true);

      const expectedXMod = modulateChannelMajor(base.X, mod, d, T);
      assertBitExactFiniteArray(sim.X_mod!, expectedXMod, 'mamba.X_mod exact');

      // RMSNorm ran on X_mod (NOT X!)
      const normOfMod = applyRmsNorm(expectedXMod, sim.gamma_in, d, T, sim.eps);
      assertBitExactFiniteArray(sim.rms_in, normOfMod.rms, 'mamba.rms_in = rms(X_mod)');
      assertBitExactFiniteArray(sim.X_norm, normOfMod.X_norm, 'mamba.X_norm = norm(X_mod)');
      assertBitExactFiniteArray(sim.U, normOfMod.U, 'mamba.U = RMSNorm(X_mod)');

      // Strictly differs from post-norm modulation (RMSNorm(X) * (1 + scale) + shift)
      const postNormWrong = modulateChannelMajor(base.U, mod, d, T);
      let maxDiff = 0;
      for (let i = 0; i < sim.U.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(sim.U[i] - postNormWrong[i]));
      }
      expect(maxDiff).toBeGreaterThan(0.1);

      // Gated residual preserves original unmodulated X: Y = X + gate * Mamba_out
      for (let i = 0; i < d; i++) {
        for (let t = 0; t < T; t++) {
          const expectedY = Math.fround(
            sim.X[i * T + t] + Math.fround(gate[i] * sim.Mamba_out[i * T + t]),
          );
          expect(sim.Y[i * T + t]).toBe(expectedY);
        }
      }
      // Channel 5 has gate = 0 -> Y[5, t] === X[5, t] exactly
      for (let t = 0; t < T; t++) {
        expect(sim.Y[5 * T + t]).toBe(sim.X[5 * T + t]);
      }

      // Formula readout reflects X_mod and residual gate
      expect(describeMambaCellFormula(sim, { tensor: 'rmsnorm', row: 0, col: 0 }).subEquation).toContain('X_mod');
      expect(describeMambaCellFormula(sim, { tensor: 'residual', row: 0, col: 0 }).equation).toContain('gate[0]');
    });

    it('applies pre-RMSNorm modulation (modulate -> RMSNorm) and gated residual on original X in GQA (*)', () => {
      const T = 4;
      const d = GQA_SIM_CONFIG.d;
      const base = runGqaSim(T);
      const sim = runGqaSim(T, {
        X: base.X,
        preNormMod: mod,
        residualGate: gate,
      });

      assertBitExactFiniteArray(sim.X, base.X, 'gqa.X untouched');
      expect(sim.X_mod).toBeDefined();
      expect(sim.residualGateApplied).toBe(true);

      const expectedXMod = modulateChannelMajor(base.X, mod, d, T);
      assertBitExactFiniteArray(sim.X_mod!, expectedXMod, 'gqa.X_mod exact');

      const normOfMod = applyRmsNorm(expectedXMod, sim.gamma, d, T, sim.rmsEps);
      assertBitExactFiniteArray(sim.rms, normOfMod.rms, 'gqa.rms = rms(X_mod)');
      assertBitExactFiniteArray(sim.U, normOfMod.U, 'gqa.U = RMSNorm(X_mod)');

      for (let i = 0; i < d; i++) {
        for (let t = 0; t < T; t++) {
          const expectedY = Math.fround(
            sim.X[i * T + t] + Math.fround(gate[i] * sim.Attn_out[i * T + t]),
          );
          expect(sim.Y[i * T + t]).toBe(expectedY);
        }
      }
      expect(describeCellFormula(sim, { tensor: 'rmsnorm', row: 0, col: 0 }).equation).toContain('X_mod');
      expect(describeCellFormula(sim, { tensor: 'residual', row: 0, col: 0 }).equation).toContain('gate[0]');
    });

    it('applies post-RMSNorm modulation (RMSNorm -> modulate) feeding router, routed, shared, and idle evals + gated residual on original X in MoE (E)', () => {
      const T = 4;
      const d = MOE_SIM_CONFIG.d;
      const base = runMoeSim(T);
      const sim = runMoeSim(T, {
        X: base.X,
        postNormMod: mod,
        residualGate: gate,
      });

      // Original X and RMSNorm(X) = U stay bit-exact identical to unmodulated base!
      assertBitExactFiniteArray(sim.X, base.X, 'moe.X untouched');
      assertBitExactFiniteArray(sim.rms, base.rms, 'moe.rms untouched');
      assertBitExactFiniteArray(sim.X_norm, base.X_norm, 'moe.X_norm untouched');
      assertBitExactFiniteArray(sim.U, base.U, 'moe.U = RMSNorm(X) untouched');

      // U_mod = U * (1 + scale) + shift
      expect(sim.U_mod).toBeDefined();
      expect(sim.residualGateApplied).toBe(true);
      const expectedUMod = modulateChannelMajor(sim.U, mod, d, T);
      assertBitExactFiniteArray(sim.U_mod!, expectedUMod, 'moe.U_mod = mod(U)');

      // Router, active routed experts, shared expert, and idle expert preview ALL consume U_mod
      for (let e = 0; e < 8; e++) {
        let dot = 0;
        for (let i = 0; i < d; i++) {
          dot += sim.W_router[e * d + i] * sim.U_mod![i * T + 0];
        }
        expect(sim.router_logits[e * T + 0]).toBe(Math.fround(dot));
      }
      let sharedPre0 = 0;
      for (let i = 0; i < d; i++) {
        sharedPre0 += sim.W_up_s[0 * d + i] * sim.U_mod![i * T + 0];
      }
      expect(sim.shared_h[0 * T + 0]).toBe(Math.fround(sharedPre0));

      // Find an idle expert at t=0 and confirm evaluateSingleRoutedExpert uses U_mod
      const activeSet = new Set(Array.from(sim.topk_indices.subarray(0, 6)));
      let idleId = 0;
      while (activeSet.has(idleId)) idleId++;
      const idleEval = evaluateSingleRoutedExpert(sim, idleId, 0);
      expect(idleEval.isActive).toBe(false);
      let idleH0 = 0;
      for (let i = 0; i < d; i++) {
        idleH0 += sim.W_up[(idleId * sim.dExp + 0) * d + i] * sim.U_mod![i * T + 0];
      }
      expect(idleEval.h[0]).toBe(Math.fround(idleH0));

      // Residual gating uses original unmodulated X
      for (let o = 0; o < d; o++) {
        for (let t = 0; t < T; t++) {
          const expectedY = Math.fround(
            sim.X[o * T + t] + Math.fround(gate[o] * sim.y_moe[o * T + t]),
          );
          expect(sim.Y[o * T + t]).toBe(expectedY);
        }
      }
      expect(describeMoeCellFormula(sim, { tensor: 'rmsnorm', row: 0, col: 0 }, 0, 0).subEquation).toContain('U_mod');
      expect(describeMoeCellFormula(sim, { tensor: 'residual', row: 0, col: 0 }, 0, 0).equation).toContain('gate[0]');
    });

    it('rejects non-finite values, wrong-length modulation/gate vectors, and query length S > 16 across M, *, and E', () => {
      const badNaN = new Float32Array([0, 0, Number.NaN, 0, 0, 0, 0, 0]);
      const badInf = new Float32Array([0, Infinity, 0, 0, 0, 0, 0, 0]);
      const badLen = new Float32Array(7);
      const ok = new Float32Array(8);

      // Non-finite & wrong-length modulation/gate throws RangeError across all 3 kernels
      expect(() => runGqaSim(2, { preNormMod: { shift: badNaN, scale: ok } })).toThrow(RangeError);
      expect(() => runGqaSim(2, { preNormMod: { shift: ok, scale: badInf } })).toThrow(RangeError);
      expect(() => runGqaSim(2, { preNormMod: { shift: badLen, scale: ok } })).toThrow(RangeError);
      expect(() => runGqaSim(2, { residualGate: badNaN })).toThrow(RangeError);
      expect(() => runGqaSim(2, { residualGate: badLen })).toThrow(RangeError);

      expect(() => runMambaSim(2, { preNormMod: { shift: badNaN, scale: ok } })).toThrow(RangeError);
      expect(() => runMambaSim(2, { preNormMod: { shift: ok, scale: badInf } })).toThrow(RangeError);
      expect(() => runMambaSim(2, { preNormMod: { shift: ok, scale: badLen } })).toThrow(RangeError);
      expect(() => runMambaSim(2, { residualGate: badInf })).toThrow(RangeError);
      expect(() => runMambaSim(2, { residualGate: badLen })).toThrow(RangeError);

      expect(() => runMoeSim(2, { postNormMod: { shift: badNaN, scale: ok } })).toThrow(RangeError);
      expect(() => runMoeSim(2, { postNormMod: { shift: ok, scale: badInf } })).toThrow(RangeError);
      expect(() => runMoeSim(2, { postNormMod: { shift: badLen, scale: ok } })).toThrow(RangeError);
      expect(() => runMoeSim(2, { residualGate: badNaN })).toThrow(RangeError);
      expect(() => runMoeSim(2, { residualGate: badLen })).toThrow(RangeError);

      // Query length S > 16 strictly rejected even when kvCacheIn capacity > 16
      const cap = 32;
      const kvRows = GQA_SIM_CONFIG.nKv * GQA_SIM_CONFIG.dHead;
      expect(() =>
        runGqaSim(17, {
          kvCacheIn: {
            seqLen: 4,
            maxTokens: cap,
            K: new Float32Array(kvRows * cap),
            V: new Float32Array(kvRows * cap),
          },
        }),
      ).toThrow(RangeError);
      expect(() => runMambaSim(17, { eps: 1e-5 })).toThrow(RangeError);
      expect(() => runMoeSim(17, { rmsNormEps: 1e-5 })).toThrow(RangeError);
    });
  });

  describe('4. Bidirectional Rectangular GQA & Single Authoritative Visibility Contract', () => {
    it('enforces isAttentionKeyVisible and maxVisibleKeyIndex consistently across causal and bidirectional_block modes', () => {
      // P = 6, B = 4, K = 10
      expect(maxVisibleKeyIndex('causal', 6, 0, 10)).toBe(6);
      expect(maxVisibleKeyIndex('causal', 6, 3, 10)).toBe(9);
      expect(maxVisibleKeyIndex('bidirectional_block', 6, 0, 10)).toBe(9);
      expect(maxVisibleKeyIndex('bidirectional_block', 6, 3, 10)).toBe(9);
      expect(maxVisibleKeyIndex('bidirectional_block', 6, 0, 0)).toBe(-1);

      expect(isAttentionKeyVisible('causal', 6, 0, 6, 10)).toBe(true);
      expect(isAttentionKeyVisible('causal', 6, 0, 7, 10)).toBe(false);
      expect(isAttentionKeyVisible('bidirectional_block', 6, 0, 7, 10)).toBe(true);
      expect(isAttentionKeyVisible('bidirectional_block', 6, 0, 9, 10)).toBe(true);
      expect(isAttentionKeyVisible('bidirectional_block', 6, 0, -1, 10)).toBe(false);
      expect(isAttentionKeyVisible('bidirectional_block', 6, 0, 10, 10)).toBe(false);
    });

    it('runs rectangular [B x (P+B)] bidirectional attention with zero -Infinity scores, nondegenerate future-token influence, and accumulation parity', () => {
      const P = 6;
      const B = 4;
      const cap = 24; // Allocated KV capacity > 16 while S = B = 4 <= 16
      const kvRows = GQA_SIM_CONFIG.nKv * GQA_SIM_CONFIG.dHead;

      // Populate prefix KV from a 6-token causal context pass
      const prefixSim = runGqaSim(P);
      const K_buf = new Float32Array(kvRows * cap);
      const V_buf = new Float32Array(kvRows * cap);
      for (let r = 0; r < kvRows; r++) {
        for (let k = 0; k < P; k++) {
          K_buf[r * cap + k] = prefixSim.K[r * P + k];
          V_buf[r * cap + k] = prefixSim.V[r * P + k];
        }
      }

      const d = GQA_SIM_CONFIG.d;
      const blockX = new Float32Array(d * B);
      for (let i = 0; i < d; i++) {
        for (let t = 0; t < B; t++) {
          blockX[i * B + t] = Math.fround(0.7 * Math.sin((t + 1) * 1.1 + i * 0.6));
        }
      }

      const kvCacheIn = {
        seqLen: P,
        maxTokens: cap,
        K: K_buf,
        V: V_buf,
      };

      const bidirSim = runGqaSim(B, {
        X: blockX,
        kvCacheIn,
        attentionMaskMode: 'bidirectional_block',
      });
      const causalSim = runGqaSim(B, {
        X: blockX,
        kvCacheIn,
        attentionMaskMode: 'causal',
      });

      expect(bidirSim.attentionMaskMode).toBe('bidirectional_block');
      expect(bidirSim.tokens).toBe(B);
      expect(bidirSim.prefixTokens).toBe(P);
      expect(bidirSim.keyTokens).toBe(P + B);

      // Zero -Infinity scores, strictly positive softmax, every row sums to 1.0
      for (let h = 0; h < bidirSim.nQ; h++) {
        for (let q = 0; q < B; q++) {
          let rowSum = 0;
          for (let k = 0; k < P + B; k++) {
            const idx = (h * B + q) * (P + B) + k;
            expect(Number.isFinite(bidirSim.Scores[idx])).toBe(true);
            expect(bidirSim.Softmax[idx]).toBeGreaterThan(0);
            rowSum += bidirSim.Softmax[idx];
          }
          expect(Math.abs(rowSum - 1.0)).toBeLessThan(1e-5);
        }
      }

      // Nondegenerate future-token influence: perturbing block token j=3 changes attention output at q=0
      // under bidirectional_block, whereas causal mode keeps q=0 bit-exact identical!
      const perturbedBlockX = new Float32Array(blockX);
      for (let i = 0; i < d; i++) {
        perturbedBlockX[i * B + 3] = Math.fround(perturbedBlockX[i * B + 3] + (i % 2 === 0 ? 1.8 : -1.5));
      }
      const bidirPert = runGqaSim(B, {
        X: perturbedBlockX,
        kvCacheIn,
        attentionMaskMode: 'bidirectional_block',
      });
      const causalPert = runGqaSim(B, {
        X: perturbedBlockX,
        kvCacheIn,
        attentionMaskMode: 'causal',
      });

      let bidirDiffAtQ0 = 0;
      let causalDiffAtQ0 = 0;
      for (let i = 0; i < d; i++) {
        bidirDiffAtQ0 = Math.max(bidirDiffAtQ0, Math.abs(bidirPert.Attn_out[i * B + 0] - bidirSim.Attn_out[i * B + 0]));
        causalDiffAtQ0 = Math.max(causalDiffAtQ0, Math.abs(causalPert.Attn_out[i * B + 0] - causalSim.Attn_out[i * B + 0]));
      }
      expect(bidirDiffAtQ0).toBeGreaterThan(1e-3);
      expect(causalDiffAtQ0).toBe(0);

      // Formula inspector & focused-cell accumulation agree with authoritative visibility contract
      const futureScoreCell = { tensor: 'scores' as const, headIndex: 2, kvHeadIndex: 0 as const, row: 0, col: P + 3 };
      const causalFormula = describeCellFormula(causalSim, futureScoreCell);
      const bidirFormula = describeCellFormula(bidirSim, futureScoreCell);
      expect(causalFormula.isMasked).toBe(true);
      expect(causalFormula.value).toBe(-Infinity);
      expect(bidirFormula.isMasked).toBe(false);
      expect(bidirFormula.value).toBe(bidirSim.Scores[(2 * B + 0) * (P + B) + (P + 3)]);

      const causalScoreAcc = computeGqaFocusedAccumulation(causalSim, futureScoreCell, 2, 0.9);
      const bidirScoreAcc = computeGqaFocusedAccumulation(bidirSim, futureScoreCell, 2, 0.9);
      expect(causalScoreAcc).toBeNull();
      expect(bidirScoreAcc).not.toBeNull();
      expect(bidirScoreAcc!.isFinal).toBe(true);
      expect(bidirScoreAcc!.partialSum).toBe(bidirSim.Scores[(2 * B + 0) * (P + B) + (P + 3)]);

      // Weighted V accumulation at q=0 spans P+1 keys in causal vs P+B keys in bidirectional_block
      const vOutCell = { tensor: 'v_out' as const, headIndex: 2, kvHeadIndex: 0 as const, row: 0, col: 2 * 4 + 1 };
      const causalVAcc = computeGqaFocusedAccumulation(causalSim, vOutCell, 4, (P + 0.5) / (P + 1));
      const bidirVAcc = computeGqaFocusedAccumulation(bidirSim, vOutCell, 4, (P + B - 0.5) / (P + B));
      expect(causalVAcc).not.toBeNull();
      expect(causalVAcc!.totalTerms).toBe(P + 1);
      expect(causalVAcc!.partialSum).toBe(causalSim.V_out[(2 * 4 + 1) * B + 0]);

      expect(bidirVAcc).not.toBeNull();
      expect(bidirVAcc!.totalTerms).toBe(P + B);
      expect(bidirVAcc!.isFinal).toBe(true);
      expect(bidirVAcc!.partialSum).toBe(bidirSim.V_out[(2 * 4 + 1) * B + 0]);

      const bidirVFormula = describeCellFormula(bidirSim, vOutCell);
      expect(bidirVFormula.upstreamSpans[0].colEnd).toBe(P + B - 1);

      // Prefix length P > 16 works cleanly while query length S <= 16 stays enforced
      const longP = 20;
      const longCap = 28;
      const longSim = runGqaSim(4, {
        attentionMaskMode: 'bidirectional_block',
        kvCacheIn: {
          seqLen: longP,
          maxTokens: longCap,
          K: new Float32Array(kvRows * longCap),
          V: new Float32Array(kvRows * longCap),
        },
      });
      expect(longSim.prefixTokens).toBe(20);
      expect(longSim.tokens).toBe(4);
      expect(longSim.keyTokens).toBe(24);
    });
  });

  describe('5. Generalized Vocab (V=34), Router Policy Threading & Deterministic Perturbation', () => {
    const vocab34 = Object.freeze([...MINI_VOCAB, '<mask>', '<eos>']);

    it('creates independent perturbed Denoiser weights with distinct parameterFingerprint, zero context mutation, and valid IDs 32 & 33', () => {
      const baseSeed = 0x4e454d33;
      const contextModel = createHybridModelParameters({
        seed: baseSeed,
        vocab: vocab34,
        routerPrecisionPolicy: 'float64_reference',
        generationPolicy: 'ar',
      });

      // Snapshot context weights BEFORE constructing denoiserModel
      const ctxEmbedSnap = new Float32Array(contextModel.W_embed);
      const ctxFinalSnap = new Float32Array(contextModel.finalNormGamma);
      const ctxLmSnap = new Float32Array(contextModel.W_lm_head);
      const ctxBay1Snap = new Float32Array((contextModel.getBayWeights(1) as MambaBayWeights).W_in);
      const ctxBay2Snap = new Float32Array((contextModel.getBayWeights(2) as MoeBayWeights).W_router);
      const ctxBay6Snap = new Float32Array((contextModel.getBayWeights(6) as GqaBayWeights).W_Q);

      const perturbSpec = {
        seed: 0x51a7beef,
        scale: 0.025,
        sourceFingerprint: contextModel.parameterFingerprint,
      };
      const denoiserModel = createHybridModelParameters({
        seed: baseSeed,
        vocab: vocab34,
        routerPrecisionPolicy: 'float64_reference',
        perturbation: perturbSpec,
        generationPolicy: 'mask_diffusion',
      });

      // Distinct fingerprints across every config axis
      expect(contextModel.config.vocabSize).toBe(34);
      expect(denoiserModel.config.vocabSize).toBe(34);
      expect(contextModel.vocab).toHaveLength(34);
      expect(denoiserModel.vocab).toHaveLength(34);
      expect(contextModel.W_embed.length).toBe(34 * 8);
      expect(contextModel.W_lm_head.length).toBe(34 * 8);
      expect(denoiserModel.parameterFingerprint).not.toBe(contextModel.parameterFingerprint);

      const diffScaleModel = createHybridModelParameters({
        seed: baseSeed,
        vocab: vocab34,
        routerPrecisionPolicy: 'float64_reference',
        perturbation: { ...perturbSpec, scale: 0.026 },
        generationPolicy: 'mask_diffusion',
      });
      const diffSeedModel = createHybridModelParameters({
        seed: baseSeed,
        vocab: vocab34,
        routerPrecisionPolicy: 'float64_reference',
        perturbation: { ...perturbSpec, seed: 0x51a7beee },
        generationPolicy: 'mask_diffusion',
      });
      const diffRouterPolicyModel = createHybridModelParameters({
        seed: baseSeed,
        vocab: vocab34,
        routerPrecisionPolicy: 'float32_nano',
        perturbation: perturbSpec,
        generationPolicy: 'mask_diffusion',
      });
      expect(diffScaleModel.parameterFingerprint).not.toBe(denoiserModel.parameterFingerprint);
      expect(diffSeedModel.parameterFingerprint).not.toBe(denoiserModel.parameterFingerprint);
      expect(diffRouterPolicyModel.parameterFingerprint).not.toBe(denoiserModel.parameterFingerprint);

      // Buffer independence + bounded deterministic perturbation + zero context mutation
      expect(denoiserModel.W_embed.buffer).not.toBe(contextModel.W_embed.buffer);
      expect(denoiserModel.finalNormGamma.buffer).not.toBe(contextModel.finalNormGamma.buffer);
      expect(denoiserModel.W_lm_head.buffer).not.toBe(contextModel.W_lm_head.buffer);

      const checkPerturbed = (a: Float32Array, b: Float32Array, label: string) => {
        expect(a.buffer, `${label} distinct buffer`).not.toBe(b.buffer);
        expect(a.length).toBe(b.length);
        let maxDiff = 0;
        let diffCount = 0;
        for (let i = 0; i < a.length; i++) {
          const dVal = Math.abs(a[i] - b[i]);
          if (dVal > 0) diffCount++;
          if (dVal > maxDiff) maxDiff = dVal;
        }
        expect(diffCount, `${label} perturbed entries`).toBeGreaterThan(0);
        expect(maxDiff, `${label} bounded by scale`).toBeLessThanOrEqual(perturbSpec.scale + 1e-6);
      };

      checkPerturbed(denoiserModel.W_embed, contextModel.W_embed, 'W_embed');
      checkPerturbed(denoiserModel.finalNormGamma, contextModel.finalNormGamma, 'finalNormGamma');
      checkPerturbed(denoiserModel.W_lm_head, contextModel.W_lm_head, 'W_lm_head');
      checkPerturbed(
        (denoiserModel.getBayWeights(1) as MambaBayWeights).W_in,
        (contextModel.getBayWeights(1) as MambaBayWeights).W_in,
        'Bay 1 Mamba W_in',
      );
      checkPerturbed(
        (denoiserModel.getBayWeights(2) as MoeBayWeights).W_router,
        (contextModel.getBayWeights(2) as MoeBayWeights).W_router,
        'Bay 2 MoE W_router',
      );
      checkPerturbed(
        (denoiserModel.getBayWeights(6) as GqaBayWeights).W_Q,
        (contextModel.getBayWeights(6) as GqaBayWeights).W_Q,
        'Bay 6 GQA W_Q',
      );

      // Context buffers were NEVER mutated
      assertBitExactFiniteArray(contextModel.W_embed, ctxEmbedSnap, 'context.W_embed unchanged');
      assertBitExactFiniteArray(contextModel.finalNormGamma, ctxFinalSnap, 'context.finalNormGamma unchanged');
      assertBitExactFiniteArray(contextModel.W_lm_head, ctxLmSnap, 'context.W_lm_head unchanged');
      assertBitExactFiniteArray((contextModel.getBayWeights(1) as MambaBayWeights).W_in, ctxBay1Snap, 'context.Bay1 unchanged');
      assertBitExactFiniteArray((contextModel.getBayWeights(2) as MoeBayWeights).W_router, ctxBay2Snap, 'context.Bay2 unchanged');
      assertBitExactFiniteArray((contextModel.getBayWeights(6) as GqaBayWeights).W_Q, ctxBay6Snap, 'context.Bay6 unchanged');

      // Determinism independent of bay access order
      const denoiserModelRev = createHybridModelParameters({
        seed: baseSeed,
        vocab: vocab34,
        routerPrecisionPolicy: 'float64_reference',
        perturbation: perturbSpec,
        generationPolicy: 'mask_diffusion',
      });
      expect(denoiserModelRev.parameterFingerprint).toBe(denoiserModel.parameterFingerprint);
      assertBitExactFiniteArray(
        (denoiserModelRev.getBayWeights(52) as MoeBayWeights).W_router,
        (denoiserModel.getBayWeights(52) as MoeBayWeights).W_router,
        'Bay 52 reverse-order deterministic',
      );
      assertBitExactFiniteArray(
        (denoiserModelRev.getBayWeights(1) as MambaBayWeights).W_in,
        (denoiserModel.getBayWeights(1) as MambaBayWeights).W_in,
        'Bay 1 reverse-order deterministic',
      );

      // Token IDs 32 (<mask>) and 33 (<eos>) execute cleanly without clamping to 31 in BOTH hybridFlow and incrementalDecode!
      const tokenIds = [0, 32, 33, 15];
      const directExec = runHybridExecution(contextModel, { tokenIds });
      const cache = new HybridExecutionCache(4);
      const fullExec = getOrRunHybridExecution(contextModel, cache, { tokenIds });
      assertBitExactFiniteArray(fullExec.lmHead.logits, directExec.lmHead.logits, 'cached vs direct V=34 logits');
      expect(Array.from(fullExec.tokenIds)).toEqual([0, 32, 33, 15]);
      expect(fullExec.tokenLabels).toEqual(['How', '<mask>', '<eos>', 'heads']);
      expect(fullExec.vocabSize).toBe(34);
      expect(fullExec.lmHead.logits.length).toBe(34 * 4);
      expect(fullExec.lmHead.probs.length).toBe(34 * 4);

      // Router precision policy threaded into MoE bays in full execution
      const fullMoe2 = fullExec.getMoeTrace(2);
      expect(fullMoe2.routerPrecisionPolicy).toBe('float64_reference');
      expect(fullMoe2.router_logits_f64).toBeInstanceOf(Float64Array);

      // Incremental decode with IDs 32 & 33 matches full execution bit-for-bit and threads float64_reference
      const step1 = runIncrementalDecodeStep(contextModel, {
        newTokenIds: [0, 32],
        prevState: null,
      });
      const step2 = runIncrementalDecodeStep(contextModel, {
        newTokenIds: [33, 15],
        prevState: step1.nextState,
      });
      expect(Array.from(step2.sequenceTokenIds)).toEqual([0, 32, 33, 15]);
      expect(step2.sequenceTokenLabels).toEqual(['How', '<mask>', '<eos>', 'heads']);
      expect(step2.vocabSize).toBe(34);
      expect(step2.lmHead.logits.length).toBe(34 * 2);
      expect(step2.getMoeTrace(2).routerPrecisionPolicy).toBe('float64_reference');
      expect(step2.getMoeTrace(2).router_logits_f64).toBeInstanceOf(Float64Array);

      // Bit-exact logits parity between full execution and 2-step incremental decode across all 34 vocab entries
      for (let v = 0; v < 34; v++) {
        expect(step2.lmHead.logits[v * 2 + 0]).toBe(fullExec.lmHead.logits[v * 4 + 2]);
        expect(step2.lmHead.logits[v * 2 + 1]).toBe(fullExec.lmHead.logits[v * 4 + 3]);
      }

      // validateDecodeCacheState remains strictly enforced against cross-model parameterFingerprint mismatches
      expect(() => validateDecodeCacheState(denoiserModel, step2.nextState)).toThrow(TypeError);

      // Invalid vocab / perturbation inputs throw RangeError
      expect(() => createHybridModelParameters({ vocab: [] })).toThrow(RangeError);
      expect(() =>
        createHybridModelParameters({ perturbation: { seed: 1, scale: -0.01 } }),
      ).toThrow(RangeError);
      expect(() =>
        createHybridModelParameters({ perturbation: { seed: Number.NaN, scale: 0.01 } }),
      ).toThrow(RangeError);
    });
  });
});
