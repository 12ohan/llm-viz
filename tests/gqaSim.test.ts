import { describe, expect, it } from 'vitest';
import {
  describeCellFormula,
  EDUCATIONAL_TOKENS,
  GQA_SIM_CONFIG,
  runGqaSim,
  stepperCellForState,
} from '../src/math/gqaSim';
import { pickHitToGqaCell, pickMacroBay } from '../src/render/pick';
import { buildGqaValueAtlas } from '../src/scene/gqaAtlas';
import { buildGqaCutaway } from '../src/scene/gqaCutaway';

describe('Stage A+1 Increment 2: Deterministic Float32 GQA Live Mathematics', () => {
  it('produces exact tensor dimensions across T = 1, 8, and 16 with educational token labels', () => {
    expect(EDUCATIONAL_TOKENS).toHaveLength(16);
    const { d, nQ, nKv, dHead, qConcatDim } = GQA_SIM_CONFIG;
    expect(d).toBe(8);
    expect(nQ).toBe(32);
    expect(nKv).toBe(2);
    expect(dHead).toBe(4);
    expect(qConcatDim).toBe(128);
    expect(qConcatDim).not.toBe(d); // Mirrors production 4,096 != 2,688

    for (const T of [1, 8, 16]) {
      const sim = runGqaSim(T);
      expect(sim.tokens).toBe(T);
      expect(sim.tokenLabels).toHaveLength(T);
      expect(sim.X.length).toBe(d * T);
      expect(sim.X_norm.length).toBe(d * T);
      expect(sim.rms.length).toBe(T);
      expect(sim.gamma.length).toBe(d);
      expect(sim.U.length).toBe(d * T);
      expect(sim.W_Q.length).toBe(qConcatDim * d);
      expect(sim.W_K.length).toBe(nKv * dHead * d);
      expect(sim.W_V.length).toBe(nKv * dHead * d);
      expect(sim.Q.length).toBe(nQ * dHead * T);
      expect(sim.K.length).toBe(nKv * dHead * T);
      expect(sim.V.length).toBe(nKv * dHead * T);
      expect(sim.Scores.length).toBe(nQ * T * T);
      expect(sim.Softmax.length).toBe(nQ * T * T);
      expect(sim.V_out.length).toBe(nQ * dHead * T);
      expect(sim.Concat.length).toBe(qConcatDim * T);
      expect(sim.W_O.length).toBe(d * qConcatDim);
      expect(sim.Attn_out.length).toBe(d * T);
      expect(sim.Y.length).toBe(d * T);
    }
  });

  it('enforces unit RMS on pre-gamma X_norm and exact U = X_norm * gamma scaling', () => {
    const sim = runGqaSim(16);
    const { d, tokens: T } = sim;

    // Verify gamma is genuinely non-unit
    let nonUnitCount = 0;
    for (let i = 0; i < d; i++) {
      if (Math.abs(sim.gamma[i] - 1.0) > 1e-3) nonUnitCount++;
    }
    expect(nonUnitCount).toBeGreaterThanOrEqual(4);

    for (let t = 0; t < T; t++) {
      let sumSq = 0;
      for (let i = 0; i < d; i++) {
        const xn = sim.X_norm[i * T + t];
        sumSq += xn * xn;
        // Exact post-gamma scaling check
        const expectedU = Math.fround(xn * sim.gamma[i]);
        expect(Math.abs(sim.U[i * T + t] - expectedU)).toBeLessThan(1e-6);
      }
      const rmsNorm = Math.sqrt(sumSq / d);
      expect(Math.abs(rmsNorm - 1.0)).toBeLessThan(1e-5);
    }
  });

  it('enforces strict causal masking (-Infinity scores, 0 softmax for k > q) and row sums ≈ 1.0', () => {
    const sim = runGqaSim(12);
    const { nQ, tokens: T } = sim;

    for (let h = 0; h < nQ; h++) {
      for (let q = 0; q < T; q++) {
        let rowSum = 0;
        for (let k = 0; k < T; k++) {
          const idx = (h * T + q) * T + k;
          const sc = sim.Scores[idx];
          const sm = sim.Softmax[idx];
          if (k > q) {
            expect(sc).toBe(-Infinity);
            expect(sm).toBe(0);
          } else {
            expect(Number.isFinite(sc)).toBe(true);
            expect(Number.isFinite(sm)).toBe(true);
            expect(sm).toBeGreaterThan(0);
            rowSum += sm;
          }
        }
        expect(Math.abs(rowSum - 1.0)).toBeLessThan(1e-5);
      }
    }
  });

  it('enforces exact 16:1 KV sharing (Q0..15 -> KV0, Q16..31 -> KV1) with distinct Q00/Q01/Q18 matrices', () => {
    const sim = runGqaSim(8);
    const { nQ, dHead, tokens: T } = sim;
    const invSqrtD = 1 / Math.sqrt(dHead);

    // Every query head h must match direct dot product against K[floor(h/16)]
    for (let h = 0; h < nQ; h++) {
      const g = Math.floor(h / 16);
      for (let q = 0; q < T; q++) {
        for (let k = 0; k <= q; k++) {
          let dot = 0;
          for (let r = 0; r < dHead; r++) {
            const qVal = sim.Q[(h * dHead + r) * T + q];
            const kVal = sim.K[(g * dHead + r) * T + k];
            dot += qVal * kVal;
          }
          const expectedScore = dot * invSqrtD;
          const actualScore = sim.Scores[(h * T + q) * T + k];
          expect(Math.abs(actualScore - expectedScore)).toBeLessThan(1e-5);
        }
      }
    }

    // Q00, Q01 (same KV0), and Q18 (KV1) must produce genuinely distinct score/softmax matrices
    const headDiff = (hA: number, hB: number, arr: Float32Array): number => {
      let maxDiff = 0;
      for (let q = 0; q < T; q++) {
        for (let k = 0; k <= q; k++) {
          const a = arr[(hA * T + q) * T + k];
          const b = arr[(hB * T + q) * T + k];
          maxDiff = Math.max(maxDiff, Math.abs(a - b));
        }
      }
      return maxDiff;
    };

    expect(headDiff(0, 1, sim.Scores)).toBeGreaterThan(0.05);
    expect(headDiff(0, 18, sim.Scores)).toBeGreaterThan(0.05);
    expect(headDiff(0, 1, sim.Softmax)).toBeGreaterThan(0.01);
    expect(headDiff(0, 18, sim.Softmax)).toBeGreaterThan(0.01);
  });

  it('maintains exact V_out -> Concat -> W_O -> + Residual numerical consistency within 1e-5', () => {
    const sim = runGqaSim(10);
    const { d, nQ, dHead, qConcatDim, tokens: T } = sim;

    // 1. Direct V_out check against Softmax * V[floor(h/16)]
    for (let h = 0; h < nQ; h++) {
      const g = Math.floor(h / 16);
      for (let r = 0; r < dHead; r++) {
        for (let q = 0; q < T; q++) {
          let acc = 0;
          for (let k = 0; k <= q; k++) {
            const p = sim.Softmax[(h * T + q) * T + k];
            const v = sim.V[(g * dHead + r) * T + k];
            acc += p * v;
          }
          const actual = sim.V_out[(h * dHead + r) * T + q];
          expect(Math.abs(actual - acc)).toBeLessThan(1e-5);
          // 2. Exact Concat mapping
          const c = h * dHead + r;
          expect(sim.Concat[c * T + q]).toBe(actual);
        }
      }
    }

    // 3. Direct W_O * Concat -> Attn_out and Y = X + Attn_out
    for (let i = 0; i < d; i++) {
      for (let t = 0; t < T; t++) {
        let acc = 0;
        for (let c = 0; c < qConcatDim; c++) {
          acc += sim.W_O[i * qConcatDim + c] * sim.Concat[c * T + t];
        }
        expect(Math.abs(sim.Attn_out[i * T + t] - acc)).toBeLessThan(1e-5);
        const expectedY = sim.X[i * T + t] + sim.Attn_out[i * T + t];
        expect(Math.abs(sim.Y[i * T + t] - expectedY)).toBeLessThan(1e-5);
      }
    }
  });

  it('supports cell-level ray picking, masked -Infinity formula formatting, and live head switching in RG32F atlas', () => {
    const gqa = buildGqaCutaway({
      bayId: 6,
      tokens: 8,
      precision: 'bf16',
      selectedPart: 'q-head-18',
      hoveredPart: null,
      activeGroup0Head: 0,
      activeGroup1Head: 18,
    });

    expect(gqa.valueAtlas).toBeDefined();
    expect(gqa.gqaSim).toBeDefined();
    expect(gqa.gqaStats?.activeGroup0Head).toBe(0);
    expect(gqa.gqaStats?.activeGroup1Head).toBe(18);

    // Exterior labels must show active head Q18 for Group 1 Scores/Softmax
    const scoresG1Label = gqa.landmarks.find((l) => l.id === 'g-scores-active');
    const softmaxG1Label = gqa.landmarks.find((l) => l.id === 'g-softmax');
    expect(scoresG1Label?.label).toContain('Q18');
    expect(softmaxG1Label?.label).toContain('Q18');

    // Cast a ray into upper-right masked cell (row=1, col=6) of Scores Group 1 sheet
    const scoresG1Box = gqa.solidPickBoxes.find(
      (b) => b.tensorKey === 'scores' && b.kvHeadIndex === 1,
    );
    expect(scoresG1Box).toBeDefined();
    const b = scoresG1Box!;
    const wx = b.min[0] + ((6 + 0.5) / 8) * (b.max[0] - b.min[0]);
    const wz = b.min[2] + ((1 + 0.5) / 8) * (b.max[2] - b.min[2]);
    const hit = pickMacroBay(
      { origin: [wx, 4.0, wz], direction: [0, -1, 0] },
      gqa.solidPickBoxes,
      [],
    );
    expect(hit).not.toBeNull();
    expect(hit!.hitRow).toBe(1);
    expect(hit!.hitCol).toBe(6);

    const cell = pickHitToGqaCell(hit!, 0, 18);
    expect(cell).toEqual({
      tensor: 'scores',
      row: 1,
      col: 6,
      headIndex: 18,
      kvHeadIndex: 1,
      tokenIndex: undefined,
    });

    const maskedFormula = describeCellFormula(gqa.gqaSim!, cell!);
    expect(maskedFormula.isMasked).toBe(true);
    expect(maskedFormula.valueText).toContain('−∞');
    expect(maskedFormula.equation.toLowerCase()).toContain('mask');
    expect(maskedFormula.headBadge).toBe('Q18 → KV1');

    // Causal lower-triangle cell (row=6, col=2) produces finite score + exact upstream Q and K spans
    const causalFormula = describeCellFormula(gqa.gqaSim!, {
      tensor: 'scores',
      row: 6,
      col: 2,
      headIndex: 18,
      kvHeadIndex: 1,
    });
    expect(causalFormula.isMasked).toBe(false);
    expect(Number.isFinite(causalFormula.value)).toBe(true);
    expect(causalFormula.upstreamSpans.some((s) => s.tensor === 'q' && s.headIndex === 18)).toBe(true);
    expect(causalFormula.upstreamSpans.some((s) => s.tensor === 'kv_k' && s.kvHeadIndex === 1)).toBe(true);

    // Switching Group 1 active head from Q16 -> Q18 genuinely changes RG32F atlas texels
    const atlasQ16 = buildGqaValueAtlas(gqa.gqaSim!, 0, 16);
    const atlasQ18 = buildGqaValueAtlas(gqa.gqaSim!, 0, 18);
    let diffCount = 0;
    for (let i = 0; i < atlasQ16.data.length; i += 2) {
      if (Math.abs(atlasQ16.data[i] - atlasQ18.data[i]) > 1e-4) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(20);

    // Stepper returns valid live formulas across all 6 stages
    for (let step = 0; step < 6; step++) {
      const stepCell = stepperCellForState(gqa.gqaSim!, 18, step, 0.5);
      const stepFormula = describeCellFormula(gqa.gqaSim!, stepCell);
      expect(stepFormula.equation.length).toBeGreaterThan(10);
    }
  });
});
