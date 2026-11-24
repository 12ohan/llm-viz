import { describe, expect, it } from 'vitest';
import {
  applyGroupedRmsNormGated,
  describeMambaCellFormula,
  EDUCATIONAL_TOKENS,
  formatMambaHeadId,
  inferMambaCellHead,
  MAMBA_SIM_CONFIG,
  MAMBA_STEPPER_STEPS,
  mambaGroupForHead,
  mambaStepperCellForState,
  runMambaSim,
  silu,
  stableSoftplus,
} from '../src/math/mambaSim';
import { pickHitToMambaCell, pickMacroBay } from '../src/render/pick';
import { buildMambaValueAtlas } from '../src/scene/mambaAtlas';
import { buildMambaCutaway, MAMBA_FIT_BOUNDS } from '../src/scene/mambaCutaway';

describe('Stage A+1 Increment 3: Deterministic Miniature Mamba-2 Simulator & Cutaway', () => {
  it('1. produces exact tensor dimensions, fixed-size state invariance across T=1,8,16, and numerically stable softplus', () => {
    expect(EDUCATIONAL_TOKENS).toHaveLength(16);
    const { d, nHeads, nGroups, dHead, dInner, dState, convKernel, convDim, inProjDim, rmsEps, normBeforeGate } =
      MAMBA_SIM_CONFIG;
    expect(d).toBe(8);
    expect(nHeads).toBe(16);
    expect(nGroups).toBe(8);
    expect(dHead).toBe(4);
    expect(dInner).toBe(64);
    expect(dState).toBe(4);
    expect(convKernel).toBe(4);
    expect(convDim).toBe(128);
    expect(inProjDim).toBe(208);
    expect(rmsEps).toBe(1e-5);
    expect(normBeforeGate).toBe(false);

    // Numerically stable softplus at extreme ±100 inputs (no overflow/NaN)
    const spPos = stableSoftplus(100);
    const spNeg = stableSoftplus(-100);
    expect(Number.isFinite(spPos)).toBe(true);
    expect(Math.abs(spPos - 100)).toBeLessThan(1e-9);
    expect(Number.isFinite(spNeg)).toBe(true);
    expect(spNeg).toBeGreaterThanOrEqual(0);
    expect(spNeg).toBeLessThan(1e-12);

    for (const T of [1, 8, 16]) {
      const sim = runMambaSim(T);
      expect(sim.tokens).toBe(T);
      expect(sim.tokenLabels).toHaveLength(T);
      expect(sim.X.length).toBe(d * T);
      expect(sim.X_norm.length).toBe(d * T);
      expect(sim.U.length).toBe(d * T);
      expect(sim.W_in.length).toBe(inProjDim * d);
      expect(sim.z.length).toBe(dInner * T);
      expect(sim.xBC_pre.length).toBe(convDim * T);
      expect(sim.dt_raw.length).toBe(nHeads * T);
      expect(sim.conv_out.length).toBe(convDim * T);
      expect(sim.xBC.length).toBe(convDim * T);
      expect(sim.x.length).toBe(nHeads * dHead * T);
      expect(sim.B.length).toBe(nGroups * dState * T);
      expect(sim.C.length).toBe(nGroups * dState * T);
      expect(sim.dt.length).toBe(nHeads * T);
      expect(sim.dA.length).toBe(nHeads * T);
      expect(sim.y.length).toBe(nHeads * dHead * T);
      expect(sim.y_norm.length).toBe(dInner * T);
      expect(sim.W_out.length).toBe(d * dInner);
      expect(sim.Mamba_out.length).toBe(d * T);
      expect(sim.Y.length).toBe(d * T);

      // Fixed-size persistent inference states NEVER grow with T!
      expect(sim.H.length).toBe(nHeads * dHead * dState); // 256
      expect(sim.H_prev.length).toBe(nHeads * dHead * dState); // 256
      expect(sim.conv_buf.length).toBe(convDim * convKernel); // 512
      expect(sim.H_history.length).toBe(T * nHeads * dHead * dState);
    }
  });

  it('2. verifies epsilon-aware RMSNorm (eps=1e-5) for normal, near-zero (~1e-3), and all-zero fixtures', () => {
    const eps = MAMBA_SIM_CONFIG.rmsEps;
    const { d, dInner, nGroups, groupSize } = MAMBA_SIM_CONFIG;

    // (a) Normal non-zero vectors (default sim)
    const simNormal = runMambaSim(8);
    for (let t = 0; t < simNormal.tokens; t++) {
      let sumSq = 0;
      for (let i = 0; i < d; i++) {
        const v = simNormal.X[i * simNormal.tokens + t];
        sumSq += v * v;
      }
      const expectedRms = Math.sqrt(sumSq / d + eps);
      expect(Math.abs(simNormal.rms_in[t] - expectedRms)).toBeLessThan(1e-6);
      for (let i = 0; i < d; i++) {
        const expectedXn = simNormal.X[i * simNormal.tokens + t] / expectedRms;
        const expectedU = expectedXn * simNormal.gamma_in[i];
        expect(Math.abs(simNormal.X_norm[i * simNormal.tokens + t] - expectedXn)).toBeLessThan(1e-5);
        expect(Math.abs(simNormal.U[i * simNormal.tokens + t] - expectedU)).toBeLessThan(1e-5);
      }
    }

    // (b) Tiny near-zero fixture (~1e-3 where mean(x^2) = 1e-6 < eps = 1e-5!)
    // Without eps, rms_norm would have unit RMS = 1.0; with eps = 1e-5, RMS of X_norm is strictly < 1.
    const tinyT = 4;
    const tinyX = new Float32Array(d * tinyT).fill(1e-3);
    const simTiny = runMambaSim(tinyT, { X: tinyX });
    const expectedTinyRms = Math.sqrt(1e-6 + eps);
    for (let t = 0; t < tinyT; t++) {
      expect(Math.abs(simTiny.rms_in[t] - expectedTinyRms)).toBeLessThan(1e-6);
      let normSumSq = 0;
      for (let i = 0; i < d; i++) {
        const xn = simTiny.X_norm[i * tinyT + t];
        expect(Math.abs(xn - 1e-3 / expectedTinyRms)).toBeLessThan(1e-5);
        normSumSq += xn * xn;
      }
      const normRms = Math.sqrt(normSumSq / d);
      expect(normRms).toBeLessThan(0.35); // sqrt(1e-6 / 1.1e-5) ≈ 0.3015, proving eps = 1e-5 dominates
    }

    // Also test 8-group MambaRMSNormGated with tiny gated values (~1e-3)
    const tinyY = new Float32Array(dInner * 2).fill(2e-3);
    const tinyZ = new Float32Array(dInner * 2).fill(0.5); // silu(0.5) ≈ 0.3112 -> gated ≈ 6.22e-4
    const unitGamma = new Float32Array(dInner).fill(1.0);
    const gatedTiny = applyGroupedRmsNormGated(tinyY, tinyZ, unitGamma, 2, nGroups, eps);
    const expectedGated = 2e-3 * silu(0.5);
    const expectedGroupRms = Math.sqrt(expectedGated * expectedGated + eps);
    for (let g = 0; g < nGroups; g++) {
      expect(Math.abs(gatedTiny.rms_gate[g * 2] - expectedGroupRms)).toBeLessThan(1e-6);
      for (let k = 0; k < groupSize; k++) {
        const c = g * groupSize + k;
        expect(Math.abs(gatedTiny.y_norm[c * 2] - expectedGated / expectedGroupRms)).toBeLessThan(1e-5);
      }
    }

    // (c) All-zero fixture: finite 0 everywhere, zero NaN/Infinity
    const zeroX = new Float32Array(d * 3); // all 0
    const zeroBConv = new Float32Array(MAMBA_SIM_CONFIG.convDim);
    const simZero = runMambaSim(3, { X: zeroX, b_conv: zeroBConv });
    for (let i = 0; i < simZero.X_norm.length; i++) {
      expect(Number.isFinite(simZero.X_norm[i])).toBe(true);
      expect(simZero.X_norm[i]).toBe(0);
      expect(Number.isFinite(simZero.U[i])).toBe(true);
      expect(simZero.U[i]).toBe(0);
      expect(Number.isFinite(simZero.Y[i])).toBe(true);
    }
    for (let i = 0; i < simZero.y_norm.length; i++) {
      expect(Number.isFinite(simZero.y_norm[i])).toBe(true);
      expect(simZero.y_norm[i]).toBe(0);
    }
  });

  it('3. enforces strict Conv1D + SSM causality and bit-for-bit prefix consistency across T=8 vs T=16', () => {
    const sim8 = runMambaSim(8);
    const sim16 = runMambaSim(16);
    const { d, dInner, convDim, nHeads, nGroups, dHead, dState } = MAMBA_SIM_CONFIG;
    const stateStride = nHeads * dHead * dState;

    // Prefix t in [0..7] must be bit-for-bit identical between T=8 and T=16
    for (let t = 0; t < 8; t++) {
      for (let i = 0; i < d; i++) {
        expect(sim16.U[i * 16 + t]).toBe(sim8.U[i * 8 + t]);
        expect(sim16.Y[i * 16 + t]).toBe(sim8.Y[i * 8 + t]);
      }
      for (let c = 0; c < dInner; c++) {
        expect(sim16.z[c * 16 + t]).toBe(sim8.z[c * 8 + t]);
        expect(sim16.y[c * 16 + t]).toBe(sim8.y[c * 8 + t]);
        expect(sim16.y_norm[c * 16 + t]).toBe(sim8.y_norm[c * 8 + t]);
      }
      for (let c = 0; c < convDim; c++) {
        expect(sim16.xBC_pre[c * 16 + t]).toBe(sim8.xBC_pre[c * 8 + t]);
        expect(sim16.conv_out[c * 16 + t]).toBe(sim8.conv_out[c * 8 + t]);
        expect(sim16.xBC[c * 16 + t]).toBe(sim8.xBC[c * 8 + t]);
      }
      for (let h = 0; h < nHeads; h++) {
        expect(sim16.dt[h * 16 + t]).toBe(sim8.dt[h * 8 + t]);
        expect(sim16.dA[h * 16 + t]).toBe(sim8.dA[h * 8 + t]);
      }
      for (let g = 0; g < nGroups * dState; g++) {
        expect(sim16.B[g * 16 + t]).toBe(sim8.B[g * 8 + t]);
        expect(sim16.C[g * 16 + t]).toBe(sim8.C[g * 8 + t]);
      }
      for (let s = 0; s < stateStride; s++) {
        expect(sim16.H_history[t * stateStride + s]).toBe(sim8.H_history[t * stateStride + s]);
      }
    }

    // Perturbing future token t=7 must leave all outputs at t < 7 bit-for-bit unchanged
    const perturbedX = new Float32Array(sim8.X);
    for (let i = 0; i < d; i++) {
      perturbedX[i * 8 + 7] += 2.5;
    }
    const simPert = runMambaSim(8, { X: perturbedX });
    for (let t = 0; t < 7; t++) {
      for (let i = 0; i < d; i++) {
        expect(simPert.Y[i * 8 + t]).toBe(sim8.Y[i * 8 + t]);
      }
      for (let s = 0; s < stateStride; s++) {
        expect(simPert.H_history[t * stateStride + s]).toBe(sim8.H_history[t * stateStride + s]);
      }
    }
    // And token t=7 MUST change
    let diffAt7 = 0;
    for (let i = 0; i < d; i++) {
      diffAt7 = Math.max(diffAt7, Math.abs(simPert.Y[i * 8 + 7] - sim8.Y[i * 8 + 7]));
    }
    expect(diffAt7).toBeGreaterThan(0.1);
  });

  it('4. verifies discretized SSM recurrence (A<0, dt>0, dA in (0,1)), readout + D skip, and W_out + residual within 1e-5', () => {
    const sim = runMambaSim(10);
    const { d, nHeads, headsPerGroup, dHead, dInner, dState, tokens: T } = sim;
    const stateStride = nHeads * dHead * dState;

    // A[h] < 0, dt > 0, dA in (0, 1)
    for (let h = 0; h < nHeads; h++) {
      expect(sim.A[h]).toBeLessThan(0);
      for (let t = 0; t < T; t++) {
        const dtVal = sim.dt[h * T + t];
        const dAVal = sim.dA[h * T + t];
        expect(dtVal).toBeGreaterThan(0);
        expect(dAVal).toBeGreaterThan(0);
        expect(dAVal).toBeLessThan(1);
        expect(Math.abs(dAVal - Math.exp(dtVal * sim.A[h]))).toBeLessThan(1e-5);
      }
    }

    // Exact recurrence H_t = dA * H_{t-1} + dt * x * B (with H_{-1} = 0) & readout y = sum_n H_t*C + D*x
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < nHeads; h++) {
        const g = Math.floor(h / headsPerGroup);
        const decay = sim.dA[h * T + t];
        const step = sim.dt[h * T + t];
        for (let p = 0; p < dHead; p++) {
          const xVal = sim.x[(h * dHead + p) * T + t];
          let readout = 0;
          for (let n = 0; n < dState; n++) {
            const localIdx = (h * dHead + p) * dState + n;
            const prevH = t === 0 ? 0 : sim.H_history[(t - 1) * stateStride + localIdx];
            const bVal = sim.B[(g * dState + n) * T + t];
            const cVal = sim.C[(g * dState + n) * T + t];
            const expectedH = decay * prevH + step * xVal * bVal;
            const actualH = sim.H_history[t * stateStride + localIdx];
            expect(Math.abs(actualH - expectedH)).toBeLessThan(1e-5);
            readout += actualH * cVal;
          }
          const expectedY = readout + sim.D[h] * xVal;
          expect(Math.abs(sim.y[(h * dHead + p) * T + t] - expectedY)).toBeLessThan(1e-5);
        }
      }
    }

    // T=1 H_prev is strictly 0
    const sim1 = runMambaSim(1);
    for (let i = 0; i < sim1.H_prev.length; i++) {
      expect(sim1.H_prev[i]).toBe(0);
    }

    // W_out @ y_norm + Residual X = Y
    for (let i = 0; i < d; i++) {
      for (let t = 0; t < T; t++) {
        let acc = 0;
        for (let c = 0; c < dInner; c++) {
          acc += sim.W_out[i * dInner + c] * sim.y_norm[c * T + t];
        }
        expect(Math.abs(sim.Mamba_out[i * T + t] - acc)).toBeLessThan(1e-5);
        expect(Math.abs(sim.Y[i * T + t] - (sim.X[i * T + t] + acc))).toBeLessThan(1e-5);
      }
    }
  });

  it('5. proves nondegenerate norm_before_gate = False order (gate FIRST, then 8-group RMSNorm)', () => {
    const { dInner, nGroups, groupSize, rmsEps } = MAMBA_SIM_CONFIG;
    const T = 4;

    // Construct nondegenerate fixture where SiLU(z) varies sharply across channels within each group
    const yFlat = new Float32Array(dInner * T);
    const zFlat = new Float32Array(dInner * T);
    const gammaGate = new Float32Array(dInner);
    for (let c = 0; c < dInner; c++) {
      gammaGate[c] = 0.85 + 0.3 * ((c % 7) / 6);
      const local = c % groupSize;
      for (let t = 0; t < T; t++) {
        yFlat[c * T + t] = (local % 2 === 0 ? 1.2 : -0.7) + 0.15 * t;
        // Strongly varying z within each 8-channel group: [-2.5 .. +3.2]
        zFlat[c * T + t] = -2.5 + local * 0.82 + 0.1 * t;
      }
    }

    const res = applyGroupedRmsNormGated(yFlat, zFlat, gammaGate, T, nGroups, rmsEps);

    let maxErrCorrect = 0;
    let maxDiffWrongOrder = 0;
    for (let g = 0; g < nGroups; g++) {
      const cStart = g * groupSize;
      for (let t = 0; t < T; t++) {
        // Correct order: RMSNorm(y * SiLU(z)) * gamma
        let sumSqCorrect = 0;
        // Wrong order (norm_before_gate = True): RMSNorm(y) * SiLU(z) * gamma
        let sumSqWrong = 0;
        for (let k = 0; k < groupSize; k++) {
          const idx = (cStart + k) * T + t;
          const gated = yFlat[idx] * silu(zFlat[idx]);
          sumSqCorrect += gated * gated;
          sumSqWrong += yFlat[idx] * yFlat[idx];
        }
        const rmsCorrect = Math.sqrt(sumSqCorrect / groupSize + rmsEps);
        const rmsWrong = Math.sqrt(sumSqWrong / groupSize + rmsEps);
        for (let k = 0; k < groupSize; k++) {
          const c = cStart + k;
          const idx = c * T + t;
          const expectedCorrect = ((yFlat[idx] * silu(zFlat[idx])) / rmsCorrect) * gammaGate[c];
          const wrongOrder = (yFlat[idx] / rmsWrong) * silu(zFlat[idx]) * gammaGate[c];
          maxErrCorrect = Math.max(maxErrCorrect, Math.abs(res.y_norm[idx] - expectedCorrect));
          maxDiffWrongOrder = Math.max(maxDiffWrongOrder, Math.abs(res.y_norm[idx] - wrongOrder));
        }
      }
    }

    expect(maxErrCorrect).toBeLessThan(1e-6);
    expect(maxDiffWrongOrder).toBeGreaterThan(0.05);
  });

  it('6. verifies Mamba-2 cutaway thin sheets, invariant bounds, picking, head-sync, Bay 01 vs Bay 42 metadata, and 6-step stepper', () => {
    const m1 = buildMambaCutaway({
      bayId: 1,
      tokens: 1,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
    });
    const m16 = buildMambaCutaway({
      bayId: 1,
      tokens: 16,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
    });

    // Thin-sheet foundation invariants: 98 invariant sheets, 0 voxel-bead ribbons, smooth vector flow ribbons
    expect(m1.sheetCount).toBe(98);
    expect(m16.sheetCount).toBe(98);
    expect(m1.ribbonCount).toBe(0);
    expect(m16.ribbonCount).toBe(0);
    expect(m1.flowIndices?.length ?? 0).toBeGreaterThan(0);
    expect(m1.fitBounds).toEqual(MAMBA_FIT_BOUNDS);
    expect(m16.fitBounds).toEqual(MAMBA_FIT_BOUNDS);
    expect(m1.valueAtlas).toBeDefined();
    expect(m1.mambaSim).toBeDefined();

    // Fixed-size H_prev / H_curr sheets stay strictly identical in geometry across T=1 vs T=16,
    // while sequence sheets (xbc_pre, z, x, b, c, dt) visibly grow in Z depth!
    const boxDepth = (g: typeof m1, key: string) => {
      const box = g.solidPickBoxes.find((b) => b.tensorKey === key);
      expect(box).toBeDefined();
      return box!.max[2] - box!.min[2];
    };
    expect(boxDepth(m16, 'h_prev')).toBeCloseTo(boxDepth(m1, 'h_prev'), 6);
    expect(boxDepth(m16, 'h_curr')).toBeCloseTo(boxDepth(m1, 'h_curr'), 6);
    expect(boxDepth(m16, 'xbc_pre')).toBeGreaterThan(boxDepth(m1, 'xbc_pre') + 0.2);
    expect(boxDepth(m16, 'z')).toBeGreaterThan(boxDepth(m1, 'z') + 0.5);
    expect(boxDepth(m16, 'x')).toBeGreaterThan(boxDepth(m1, 'x') + 0.3);
    expect(boxDepth(m16, 'b')).toBeGreaterThan(boxDepth(m1, 'b') + 0.3);
    expect(boxDepth(m16, 'c')).toBeGreaterThan(boxDepth(m1, 'c') + 0.3);
    expect(boxDepth(m16, 'dt')).toBeGreaterThan(boxDepth(m1, 'dt') + 0.3);

    // Ray-pick H_curr, conv_buf, and y_norm cells and verify exact formulas
    const hCurrBox = m16.solidPickBoxes.find((b) => b.tensorKey === 'h_curr')!;
    const hwx = hCurrBox.min[0] + ((2 + 0.5) / 4) * (hCurrBox.max[0] - hCurrBox.min[0]);
    const hwz = hCurrBox.min[2] + ((1 + 0.5) / 4) * (hCurrBox.max[2] - hCurrBox.min[2]);
    const hHit = pickMacroBay(
      { origin: [hwx, 4.0, hwz], direction: [0, -1, 0] },
      m16.solidPickBoxes,
      [],
    );
    expect(hHit).not.toBeNull();
    const hCell = pickHitToMambaCell(hHit!, 0, 16);
    expect(hCell).toMatchObject({ tensor: 'h_curr', row: 1, col: 2, headIndex: 0, groupIndex: 0 });
    const hFormula = describeMambaCellFormula(m16.mambaSim!, hCell!);
    expect(hFormula.equation).toContain('H_t');
    expect(hFormula.upstreamSpans.some((s) => s.tensor === 'h_prev')).toBe(true);
    expect(hFormula.upstreamSpans.some((s) => s.tensor === 'x')).toBe(true);
    expect(hFormula.upstreamSpans.some((s) => s.tensor === 'b')).toBe(true);
    expect(hFormula.upstreamSpans.some((s) => s.tensor === 'dt')).toBe(true);

    // Pick Conv1D rolling buffer tap
    const convBox = m16.solidPickBoxes.find((b) => b.tensorKey === 'conv_buf' && b.tokenIndex === 3)!;
    const cwx = convBox.min[0] + ((10 + 0.5) / 128) * (convBox.max[0] - convBox.min[0]);
    const cwz = (convBox.min[2] + convBox.max[2]) * 0.5;
    const cHit = pickMacroBay(
      { origin: [cwx, 4.0, cwz], direction: [0, -1, 0] },
      m16.solidPickBoxes,
      [],
    );
    const cCell = pickHitToMambaCell(cHit!, 0, 16);
    expect(cCell).toMatchObject({ tensor: 'conv_buf', tapIndex: 3, col: 10 });
    const cFormula = describeMambaCellFormula(m16.mambaSim!, cCell!);
    expect(cFormula.headBadge).toContain('BF16');

    // Pick y_norm cell belonging to head H11 (c = 11*4 + 2 = 46) while activeHead = 0 -> auto-syncs to H11!
    const ynBox = m16.solidPickBoxes.find((b) => b.tensorKey === 'y_norm')!;
    const ywx = ynBox.min[0] + ((46 + 0.5) / 64) * (ynBox.max[0] - ynBox.min[0]);
    const ywz = ynBox.min[2] + ((7 + 0.5) / 16) * (ynBox.max[2] - ynBox.min[2]);
    const yHit = pickMacroBay(
      { origin: [ywx, 4.0, ywz], direction: [0, -1, 0] },
      m16.solidPickBoxes,
      [],
    );
    const ynCell = pickHitToMambaCell(yHit!, 0, 16);
    expect(ynCell?.tensor).toBe('y_norm');
    expect(inferMambaCellHead(ynCell)).toBe(11);
    expect(mambaGroupForHead(11)).toBe(5);
    expect(formatMambaHeadId(11)).toBe('H11');

    const syncedH11 = buildMambaCutaway({
      bayId: 1,
      tokens: 16,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
      activeCell: ynCell,
    });
    expect(syncedH11.mambaStats?.activeHead).toBe(11);
    expect(syncedH11.mambaStats?.activeGroup).toBe(5);
    expect(syncedH11.landmarks.find((l) => l.id === 'm-hcurr')?.label).toContain('H11');
    expect(syncedH11.activeFormula?.headBadge).toBe('H11 · G5');
    expect(syncedH11.activeFormula?.subEquation).toContain('norm_before_gate=False');

    // Switching head H00 -> H11 genuinely updates RG32F atlas texels
    const atlasH00 = buildMambaValueAtlas(m16.mambaSim!, 0);
    const atlasH11 = buildMambaValueAtlas(m16.mambaSim!, 11);
    let diffTexels = 0;
    for (let i = 0; i < atlasH00.data.length; i += 2) {
      if (Math.abs(atlasH00.data[i] - atlasH11.data[i]) > 1e-4) diffTexels++;
    }
    expect(diffTexels).toBeGreaterThan(10);

    // Bay 01 vs Bay 42 Feeder FP8 Selective metadata honesty
    const bay01Fp8 = buildMambaCutaway({
      bayId: 1,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
    });
    const bay42Fp8 = buildMambaCutaway({
      bayId: 42,
      tokens: 8,
      precision: 'fp8_selective',
      selectedPart: null,
      hoveredPart: null,
      activeHead: 0,
    });
    expect(bay01Fp8.landmarks.find((l) => l.id === 'm-in-proj')?.label).toContain('FP8');
    expect(bay01Fp8.landmarks.find((l) => l.id === 'm-conv1d')?.label).toContain('BF16');
    expect(bay42Fp8.landmarks.find((l) => l.id === 'm-in-proj')?.label).not.toContain('FP8');
    expect(bay42Fp8.landmarks.find((l) => l.id === 'm-feeder-note')?.label).toContain('Bay 42 Feeder');
    expect(bay42Fp8.landmarks.find((l) => l.id === 'm-feeder-note')?.label).toContain('BF16');

    // All 6 Mamba-2 stepper stages produce valid live formulas
    expect(MAMBA_STEPPER_STEPS).toHaveLength(6);
    for (let step = 0; step < 6; step++) {
      const sc = mambaStepperCellForState(m16.mambaSim!, 11, step, 0.5);
      const sf = describeMambaCellFormula(m16.mambaSim!, sc);
      expect(sf.equation.length).toBeGreaterThan(10);
      expect(Number.isFinite(sf.value)).toBe(true);
    }
  });
});
