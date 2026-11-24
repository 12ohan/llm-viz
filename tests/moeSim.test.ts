import { describe, expect, it } from 'vitest';
import {
  activeRoutedExpertIndices,
  describeMoeCellFormula,
  evaluateSingleRoutedExpert,
  formatExpertId,
  MOE_SIM_CONFIG,
  MOE_STEPPER_STEPS,
  moeStepperCellForState,
  relu2,
  runMoeSim,
  stableSigmoid,
  type MoeCellFormula,
  type MoeCellRef,
} from '../src/math/moeSim';
import { pickHitToMoeCell, type PickHit } from '../src/render/pick';
import { buildMoECutaway, MOE_FIT_BOUNDS } from '../src/scene/cutaways';
import { SOLID_INSTANCE_FLOATS } from '../src/scene/spine';

describe('Stage A+1 Increment 4: Miniature NemotronH MoE (128+1) Simulator', () => {
  it('1. preserves exact dimensions, 2× shared ratio, 128 routed / top-6 sparsity, and varied routing across T=1, 8, 16', () => {
    expect(MOE_SIM_CONFIG.dShared).toBe(2 * MOE_SIM_CONFIG.dExp);
    expect(MOE_SIM_CONFIG.prodSharedWidth).toBe(2 * MOE_SIM_CONFIG.prodRoutedWidth);
    expect(MOE_SIM_CONFIG.nRoutedExperts).toBe(128);
    expect(MOE_SIM_CONFIG.topK).toBe(6);
    expect(MOE_SIM_CONFIG.routedScalingFactor).toBe(2.5);

    for (const T of [1, 8, 16]) {
      const sim = runMoeSim(T);
      expect(sim.tokens).toBe(T);
      expect(sim.tokenLabels).toHaveLength(T);
      expect(sim.d).toBe(8);
      expect(sim.dExp).toBe(4);
      expect(sim.dShared).toBe(8);
      expect(sim.X.length).toBe(sim.d * T);
      expect(sim.U.length).toBe(sim.d * T);
      expect(sim.router_logits.length).toBe(128 * T);
      expect(sim.scores.length).toBe(128 * T);
      expect(sim.scores_for_choice.length).toBe(128 * T);
      expect(sim.topk_indices.length).toBe(T * 6);
      expect(sim.topk_weights.length).toBe(T * 6);
      expect(sim.y_routed.length).toBe(sim.d * T);
      expect(sim.y_shared.length).toBe(sim.d * T);
      expect(sim.Y.length).toBe(sim.d * T);

      for (let t = 0; t < T; t++) {
        const winners = new Set<number>();
        for (let j = 0; j < 6; j++) {
          const eid = sim.topk_indices[t * 6 + j];
          expect(eid).toBeGreaterThanOrEqual(0);
          expect(eid).toBeLessThan(128);
          winners.add(eid);
        }
        expect(winners.size).toBe(6); // 6 unique active, 122 idle
      }
    }

    // Distinct Top-6 signatures across the 16 tokens
    const sim16 = runMoeSim(16);
    const sigs = new Set<string>();
    for (let t = 0; t < 16; t++) {
      const ids = Array.from(sim16.topk_indices.subarray(t * 6, (t + 1) * 6));
      sigs.add(ids.join(','));
      expect(activeRoutedExpertIndices(t + 1)).toEqual(ids);
    }
    expect(sigs.size).toBeGreaterThanOrEqual(12);
  });

  it('2. computes epsilon-aware RMSNorm (eps = 1e-5) accurately for normal, near-zero, and all-zero fixtures with non-unit gamma_in', () => {
    const d = MOE_SIM_CONFIG.d;
    const eps = MOE_SIM_CONFIG.rmsNormEps;
    const T = 3;
    const X = new Float32Array(d * T);
    const gamma = new Float32Array([1.25, 0.75, 1.5, 0.9, 1.1, 0.8, 1.3, 0.95]);

    // Token 0: normal magnitude
    const normalVals = [0.8, -1.2, 0.5, -0.4, 1.1, -0.7, 0.3, -0.9];
    // Token 1: near-zero (~1e-3) where eps = 1e-5 measurably damps mean(x_norm²)
    const nearZeroVals = [1e-3, -2e-3, 1.5e-3, -1e-3, 0.8e-3, -1.2e-3, 0.5e-3, -0.9e-3];
    // Token 2: exact all-zero vector -> must yield exact 0 with zero NaNs
    for (let i = 0; i < d; i++) {
      X[i * T + 0] = normalVals[i];
      X[i * T + 1] = nearZeroVals[i];
      X[i * T + 2] = 0;
    }

    const sim = runMoeSim(T, { X, gamma_in: gamma });

    for (let t = 0; t < T; t++) {
      let m = 0;
      for (let i = 0; i < d; i++) {
        m += X[i * T + t] * X[i * T + t];
      }
      m /= d;
      let meanXnSq = 0;
      for (let i = 0; i < d; i++) {
        const xn = sim.X_norm[i * T + t];
        const u = sim.U[i * T + t];
        expect(Number.isFinite(xn)).toBe(true);
        expect(Number.isFinite(u)).toBe(true);
        meanXnSq += xn * xn;
        expect(u).toBeCloseTo(xn * gamma[i], 5);
      }
      meanXnSq /= d;
      const expectedMeanSq = m === 0 ? 0 : m / (m + eps);
      expect(meanXnSq).toBeCloseTo(expectedMeanSq, 5);
    }

    // Token 2 all-zero -> exact zero X_norm and U
    for (let i = 0; i < d; i++) {
      expect(sim.X_norm[i * T + 2]).toBe(0);
      expect(sim.U[i * T + 2]).toBe(0);
    }
  });

  it('3. uses e_score_correction_bias strictly for Top-6 selection while gathering unbiased sigmoid scores scaled by 2.5×', () => {
    // 3a. Verify default seeded model has at least one token where bias changes Top-6 ranking/membership
    const simDef = runMoeSim(16);
    let biasAlteredTokens = 0;
    for (let t = 0; t < 16; t++) {
      const unbiasedOrder = Array.from({ length: 128 }, (_, e) => e)
        .sort((a, b) => {
          const diff = simDef.scores[b * 16 + t] - simDef.scores[a * 16 + t];
          return diff !== 0 ? diff : a - b;
        })
        .slice(0, 6);
      const chosenOrder = Array.from(simDef.topk_indices.subarray(t * 6, (t + 1) * 6));
      if (unbiasedOrder.join(',') !== chosenOrder.join(',')) {
        biasAlteredTokens++;
      }

      // Normalized probs sum to 1.0 and final weights sum to 2.5
      let probSum = 0;
      let weightSum = 0;
      for (let j = 0; j < 6; j++) {
        probSum += simDef.topk_norm_probs[t * 6 + j];
        weightSum += simDef.topk_weights[t * 6 + j];
      }
      expect(probSum).toBeCloseTo(1.0, 5);
      expect(weightSum).toBeCloseTo(2.5, 5);
    }
    expect(biasAlteredTokens).toBeGreaterThanOrEqual(1);

    // 3b. Controlled fixture where e_score_correction_bias explicitly flips Top-6 membership!
    const d = MOE_SIM_CONFIG.d;
    const X = new Float32Array(d);
    X[0] = 1.0; // U[0] = 1/sqrt(1/8 + 1e-5) * 1.0
    const gamma = new Float32Array(d).fill(1.0);
    const W_router = new Float32Array(128 * d);
    const bias = new Float32Array(128);

    // Give experts 0..5 higher unbiased logits than expert 99, then boost expert 99 via bias so it enters Top-6
    const logitsWanted = new Float32Array(128).fill(-2.0);
    logitsWanted[0] = 1.2;
    logitsWanted[1] = 1.1;
    logitsWanted[2] = 1.0;
    logitsWanted[3] = 0.9;
    logitsWanted[4] = 0.8;
    logitsWanted[5] = 0.7; // unbiased rank #6
    logitsWanted[99] = 0.4; // unbiased rank #7, lower unbiased score than expert 5!
    const u0 = 1.0 / Math.sqrt(1 / d + MOE_SIM_CONFIG.rmsNormEps);
    for (let e = 0; e < 128; e++) {
      W_router[e * d + 0] = logitsWanted[e] / u0;
    }
    bias[99] = 0.45; // boosts scores_for_choice[99] well above expert 5 (and even above expert 0!)

    const simFix = runMoeSim(1, {
      X,
      gamma_in: gamma,
      W_router,
      e_score_correction_bias: bias,
    });

    const chosen = Array.from(simFix.topk_indices);
    expect(chosen).toContain(99);
    expect(chosen).not.toContain(5);
    expect(chosen[0]).toBe(99); // highest choice score

    // Crucial verification: topk_weights MUST be computed from UNBIASED scores[e], NOT scores_for_choice[e]!
    const unbiasedScores = chosen.map((e) => stableSigmoid(simFix.router_logits[e]));
    const sumUnbiased = unbiasedScores.reduce((a, b) => a + b, 0);
    for (let j = 0; j < 6; j++) {
      const expectedNorm = unbiasedScores[j] / sumUnbiased;
      const expectedWeight = expectedNorm * 2.5;
      expect(simFix.topk_raw_scores[j]).toBeCloseTo(unbiasedScores[j], 5);
      expect(simFix.topk_norm_probs[j]).toBeCloseTo(expectedNorm, 5);
      expect(simFix.topk_weights[j]).toBeCloseTo(expectedWeight, 5);
    }
    // Expert 99 was selected #1 due to bias, yet its unbiased weight MUST be strictly SMALLER than expert 0's!
    const idx99 = chosen.indexOf(99);
    const idx0 = chosen.indexOf(0);
    expect(simFix.topk_weights[idx99]).toBeLessThan(simFix.topk_weights[idx0]);
  });

  it('4. executes exact non-gated ReLU² (0 for h<=0, h² for h>0), unscaled 2× shared expert, and Y = X + y_routed + y_shared', () => {
    expect(relu2(-2.5)).toBe(0);
    expect(relu2(0)).toBe(0);
    expect(relu2(1.5)).toBeCloseTo(2.25, 6);

    const sim = runMoeSim(8);
    let zeroedNegatives = 0;
    let squaredPositives = 0;

    for (let t = 0; t < sim.tokens; t++) {
      for (let j = 0; j < sim.topK; j++) {
        const hBase = (t * sim.topK + j) * sim.dExp;
        for (let k = 0; k < sim.dExp; k++) {
          const hv = sim.active_h[hBase + k];
          const av = sim.active_a[hBase + k];
          if (hv <= 0) {
            expect(av).toBe(0);
            zeroedNegatives++;
          } else {
            expect(av).toBeCloseTo(hv * hv, 5);
            squaredPositives++;
          }
        }
      }

      // Check shared expert ReLU² & exact unscaled y_shared + y_routed + X = Y
      for (let s = 0; s < sim.dShared; s++) {
        const sh = sim.shared_h[s * sim.tokens + t];
        const sa = sim.shared_a[s * sim.tokens + t];
        if (sh <= 0) expect(sa).toBe(0);
        else expect(sa).toBeCloseTo(sh * sh, 5);
      }

      for (let o = 0; o < sim.d; o++) {
        let manualRouted = 0;
        for (let j = 0; j < sim.topK; j++) {
          manualRouted += sim.topk_weights[t * sim.topK + j] * sim.active_y[(t * sim.topK + j) * sim.d + o];
        }
        let manualShared = 0;
        for (let s = 0; s < sim.dShared; s++) {
          manualShared += sim.W_down_s[o * sim.dShared + s] * sim.shared_a[s * sim.tokens + t];
        }
        expect(sim.y_routed[o * sim.tokens + t]).toBeCloseTo(manualRouted, 5);
        // Verify y_shared is NOT multiplied by 2.5
        expect(sim.y_shared[o * sim.tokens + t]).toBeCloseTo(manualShared, 5);
        expect(sim.y_moe[o * sim.tokens + t]).toBeCloseTo(manualRouted + manualShared, 5);
        expect(sim.Y[o * sim.tokens + t]).toBeCloseTo(
          sim.X[o * sim.tokens + t] + manualRouted + manualShared,
          5,
        );
      }
    }

    expect(zeroedNegatives).toBeGreaterThan(0);
    expect(squaredPositives).toBeGreaterThan(0);
  });

  it('5. enforces strict per-token independence (0 B sequence cache) and prefix consistency across T=8 vs T=16', () => {
    const sim8 = runMoeSim(8);
    const sim16 = runMoeSim(16);

    // Prefix consistency: tokens 0..7 in T=16 must match T=8 bit-for-bit
    for (let t = 0; t < 8; t++) {
      for (let j = 0; j < 6; j++) {
        expect(sim16.topk_indices[t * 6 + j]).toBe(sim8.topk_indices[t * 6 + j]);
        expect(sim16.topk_weights[t * 6 + j]).toBe(sim8.topk_weights[t * 6 + j]);
      }
      for (let i = 0; i < sim8.d; i++) {
        expect(sim16.Y[i * 16 + t]).toBe(sim8.Y[i * 8 + t]);
        expect(sim16.y_routed[i * 16 + t]).toBe(sim8.y_routed[i * 8 + t]);
        expect(sim16.y_shared[i * 16 + t]).toBe(sim8.y_shared[i * 8 + t]);
      }
    }

    // Mutate BOTH earlier (t=0, t=2) and later (t=5, t=7) tokens; verify target token t=4 changes by exact 0
    const mutatedX = new Float32Array(sim8.X);
    for (let i = 0; i < sim8.d; i++) {
      mutatedX[i * 8 + 0] += 3.5;
      mutatedX[i * 8 + 2] = -2.25;
      mutatedX[i * 8 + 5] *= -4.0;
      mutatedX[i * 8 + 7] += 1.75;
    }
    const simMut = runMoeSim(8, { X: mutatedX });
    const targetT = 4;
    for (let j = 0; j < 6; j++) {
      expect(simMut.topk_indices[targetT * 6 + j]).toBe(sim8.topk_indices[targetT * 6 + j]);
      expect(simMut.topk_weights[targetT * 6 + j]).toBe(sim8.topk_weights[targetT * 6 + j]);
    }
    for (let i = 0; i < sim8.d; i++) {
      expect(simMut.Y[i * 8 + targetT] - sim8.Y[i * 8 + targetT]).toBe(0);
      expect(simMut.U[i * 8 + targetT] - sim8.U[i * 8 + targetT]).toBe(0);
    }
  });

  it('6. verifies MoE cutaway rendering, 0 voxel arrows, 128 pickable router tiles, FP8 experts vs FP32 router metadata, expert sync & idle explanation', () => {
    const moeBf16 = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
    });
    const moeFp8 = buildMoECutaway({
      bayId: 52,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
    });

    // Zero voxel arrows, smooth flow ribbons present, invariant fit bounds
    expect(moeBf16.ribbonCount).toBe(0);
    expect(moeBf16.moeStats?.voxelArrowCount).toBe(0);
    expect(moeBf16.moeStats?.usesSmoothRibbons).toBe(true);
    expect((moeBf16.flowVertices?.length ?? 0)).toBeGreaterThan(0);
    expect(moeBf16.fitBounds).toEqual(MOE_FIT_BOUNDS);
    expect(moeFp8.fitBounds).toEqual(MOE_FIT_BOUNDS);
    expect(moeFp8.bayId).toBe(52);

    // All 128 router experts addressable via pickBoxes (12 active micro-plates = 6 experts + 122 idle tiles)
    const bankBoxes = moeBf16.solidPickBoxes.filter((b) => b.tensorKey === 'router_bank');
    const uniqueExpertIds = new Set(bankBoxes.map((b) => b.headIndex));
    expect(uniqueExpertIds.size).toBe(128);

    // FP8 Selective metadata: routed & shared W_up/W_down get fp8Flag=1 while W_router stays fp8Flag=0 (FP32)
    expect(moeFp8.moeStats?.weightsFp8).toBe(true);
    expect(moeFp8.moeStats?.routerFp8).toBe(false);
    const fp8FlagForPart = (partId: string) => {
      const flags: number[] = [];
      for (let i = 0; i < moeFp8.solidPickBoxes.length; i++) {
        if (moeFp8.solidPickBoxes[i].partId === partId) {
          flags.push(moeFp8.solidData[i * SOLID_INSTANCE_FLOATS + 7]);
        }
      }
      return flags;
    };
    expect(fp8FlagForPart('w-router').every((f) => f === 0)).toBe(true);
    expect(fp8FlagForPart('router').every((f) => f === 0)).toBe(true);
    expect(fp8FlagForPart('routed-w-up').every((f) => f === 1)).toBe(true);
    expect(fp8FlagForPart('routed-w-down').every((f) => f === 1)).toBe(true);
    expect(fp8FlagForPart('shared-expert').every((f) => f === 1)).toBe(true);
    expect(fp8FlagForPart('shared-w-down').every((f) => f === 1)).toBe(true);

    // Picking round-trip + expert synchronization on clicking Top-6 slot #2 (different from default rank #0)
    const sim = moeBf16.moeSim!;
    const focusTok = 7;
    const topkSlice = sim.topk_indices.subarray(focusTok * 6, (focusTok + 1) * 6);
    const slot2Expert = topkSlice[2];
    const topkHit: PickHit = {
      bayId: 2,
      partId: 'router',
      tensorKey: 'topk_weights',
      hitRow: 0,
      hitCol: 2,
      tokenIndex: focusTok,
      t: 1,
      point: [0, 0, 0],
      isFascia: false,
    };
    const topkCell = pickHitToMoeCell(topkHit, focusTok, topkSlice[0], topkSlice);
    expect(topkCell).not.toBeNull();
    expect(topkCell?.expertId).toBe(slot2Expert);

    const syncedCutaway = buildMoECutaway({
      bayId: 2,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'router',
      hoveredPart: null,
      focusToken: focusTok,
      activeCell: topkCell,
    });
    expect(syncedCutaway.moeStats?.focusExpert).toBe(slot2Expert);
    expect(syncedCutaway.activeFormula?.tensorLabel).toContain('Top-6 Weight #3');
    expect(syncedCutaway.activeFormula?.coordLabel).toContain(formatExpertId(slot2Expert));

    // Clicking an idle expert clearly explains why it was skipped (rank > 6, 0 contribution)
    const activeSet = new Set(Array.from(topkSlice));
    let idleExpert = 0;
    while (activeSet.has(idleExpert)) idleExpert++;
    const idleEval = evaluateSingleRoutedExpert(sim, idleExpert, focusTok);
    expect(idleEval.isActive).toBe(false);
    expect(idleEval.appliedWeight).toBe(0);
    expect(idleEval.choiceRank).toBeGreaterThanOrEqual(6);

    const idleCell: MoeCellRef = {
      tensor: 'router_bank',
      row: Math.floor(idleExpert / 16),
      col: idleExpert % 16,
      tokenIndex: focusTok,
      expertId: idleExpert,
    };
    const idleFormula = describeMoeCellFormula(sim, idleCell, focusTok, idleExpert);
    expect(idleFormula.isIdleExpert).toBe(true);
    expect(idleFormula.isMasked).toBe(true);
    expect(idleFormula.headBadge).toContain('IDLE');
    expect(idleFormula.equation).toContain('skipped');

    // Stepper covers all 6 stages cleanly with valid formula readouts
    expect(MOE_STEPPER_STEPS).toHaveLength(6);
    for (let s = 0; s < MOE_STEPPER_STEPS.length; s++) {
      const cell = moeStepperCellForState(sim, s, 0.25, focusTok, topkSlice[0]);
      const readout = describeMoeCellFormula(sim, cell, focusTok, topkSlice[0]);
      expect(readout.equation.length).toBeGreaterThan(8);
      expect(Number.isFinite(readout.value)).toBe(true);
    }
  });

  it('7. synchronizes MoE routed_sum expert dependencies to an active Top-6 expert of the inspected token', () => {
    const T = 8;
    const sim = runMoeSim(T);
    const t7 = 7;
    const topk7 = Array.from(sim.topk_indices.subarray(t7 * 6, (t7 + 1) * 6));
    const rank0Exp = topk7[0];
    const rank3Exp = topk7[2]; // Active non-rank-0 expert
    expect(rank3Exp).not.toBe(rank0Exp);

    // (a) When an active non-rank-0 expert (Rank #3) is focused, inspecting routed_sum stays synced to Rank #3
    const sumCellT7: MoeCellRef = { tensor: 'routed_sum', row: t7, col: 3, tokenIndex: t7 };
    const activeSynced = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: 'routed-sum',
      hoveredPart: null,
      focusToken: t7,
      activeExpert: rank3Exp,
      activeCell: sumCellT7,
    });
    expect(activeSynced.moeStats?.focusToken).toBe(t7);
    expect(activeSynced.moeStats?.focusExpert).toBe(rank3Exp);
    const activeFormulaA = activeSynced.activeFormula as MoeCellFormula;
    expect(activeFormulaA.activeExpertId).toBe(rank3Exp);
    const activeWDownSpan = activeFormulaA.upstreamSpans.find(
      (s) => s.tensor === 'routed_w_down',
    );
    expect(activeWDownSpan).toMatchObject({
      tensor: 'routed_w_down',
      rowStart: 3,
      rowEnd: 3,
      expertId: rank3Exp,
      tokenIndex: t7,
    });
    expect(activeFormulaA.subEquation).toContain(formatExpertId(rank3Exp));

    // (b) When an IDLE expert is focused, inspecting routed_sum reconciles focusExpert & upstream span to an active Top-6 expert
    const activeSet7 = new Set(topk7);
    let idleExp = 0;
    while (activeSet7.has(idleExp)) idleExp++;
    const idleReconciled = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: `moe-expert-${idleExp}`,
      hoveredPart: 'routed-sum',
      focusToken: t7,
      activeExpert: idleExp,
      activeCell: sumCellT7,
    });
    expect(idleReconciled.moeStats?.focusExpert).toBe(rank0Exp);
    expect(activeSet7.has(idleReconciled.moeStats!.focusExpert!)).toBe(true);
    const idleFormulaB = idleReconciled.activeFormula as MoeCellFormula;
    expect(idleFormulaB.activeExpertId).toBe(rank0Exp);
    const reconciledSpan = idleFormulaB.upstreamSpans.find(
      (s) => s.tensor === 'routed_w_down',
    );
    expect(reconciledSpan).toMatchObject({
      tensor: 'routed_w_down',
      expertId: rank0Exp,
      tokenIndex: t7,
    });
    expect(idleReconciled.landmarks.find((l) => l.id === 'moe-routed-up')?.label).toContain(
      formatExpertId(rank0Exp),
    );

    // (c) Hovering routed_sum at an earlier token t=2 where t=7's active expert is idle reconciles to t=2's Top-6
    const t2 = 2;
    const topk2 = Array.from(sim.topk_indices.subarray(t2 * 6, (t2 + 1) * 6));
    const activeSet2 = new Set(topk2);
    const activeOnlyAt7 = topk7.find((e) => !activeSet2.has(e))!;
    expect(activeOnlyAt7).toBeDefined();

    const sumCellT2: MoeCellRef = { tensor: 'routed_sum', row: t2, col: 1, tokenIndex: t2 };
    const crossTokCutaway = buildMoECutaway({
      bayId: 2,
      tokens: T,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: 'routed-sum',
      focusToken: t7,
      activeExpert: activeOnlyAt7,
      activeCell: sumCellT2,
    });
    expect(crossTokCutaway.moeStats?.focusToken).toBe(t2);
    expect(crossTokCutaway.moeStats?.focusExpert).toBe(topk2[0]);
    expect(activeSet2.has(crossTokCutaway.moeStats!.focusExpert!)).toBe(true);
    const crossFormulaC = crossTokCutaway.activeFormula as MoeCellFormula;
    expect(crossFormulaC.activeExpertId).toBe(topk2[0]);
    const crossSpan = crossFormulaC.upstreamSpans.find(
      (s) => s.tensor === 'routed_w_down',
    );
    expect(crossSpan).toMatchObject({
      tensor: 'routed_w_down',
      expertId: topk2[0],
      tokenIndex: t2,
    });
  });
});

