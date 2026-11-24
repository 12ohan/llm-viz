import { describe, expect, it } from 'vitest';

import { describeCellFormula, runGqaSim } from '../src/math/gqaSim';
import {
  createDefaultHybridModel,
  createHybridModelParameters,
  defaultTokenIdsForLength,
  getOrRunHybridExecution,
  HybridExecutionCache,
  runHybridExecution,
} from '../src/math/hybridFlow';
import { HYBRID_OVERRIDE_PATTERN } from '../src/model/spec';

function expectBitExactArrays(a: Float32Array, b: Float32Array, label: string): void {
  expect(a.length, `${label} length`).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(a[i], `${label}[${i}]`).toBe(b[i]);
  }
}

describe('Gate A: Unified 52-Layer Miniature Forward Pass (hybridFlow.ts)', () => {
  it('1. separates stable immutable model parameters from executions with zero cross-instance collisions', () => {
    const modelA = createHybridModelParameters({ seed: 0x11111111, modelId: 'model-A' });
    const modelB = createHybridModelParameters({ seed: 0x22222222, modelId: 'model-B' });
    const modelACopy = createHybridModelParameters({ seed: 0x11111111, modelId: 'model-A' });

    expect(modelA.modelId).toBe('model-A');
    expect(modelB.modelId).toBe('model-B');

    // Deterministic reproducibility across independent model instances with identical seed
    expectBitExactArrays(modelA.W_embed, modelACopy.W_embed, 'W_embed');
    expectBitExactArrays(modelA.W_lm_head, modelACopy.W_lm_head, 'W_lm_head');
    for (const bayId of [1, 2, 6, 17, 42, 43, 52]) {
      const wA = modelA.getBayWeights(bayId);
      const wCopy = modelACopy.getBayWeights(bayId);
      const wB = modelB.getBayWeights(bayId);
      expect(wA.kind).toBe(wCopy.kind);
      if (wA.kind === 'M' && wCopy.kind === 'M' && wB.kind === 'M') {
        expectBitExactArrays(wA.W_in, wCopy.W_in, `Bay ${bayId} W_in`);
        expectBitExactArrays(wA.W_out, wCopy.W_out, `Bay ${bayId} W_out`);
        expect(wA.W_in[0]).not.toBe(wB.W_in[0]);
      } else if (wA.kind === 'E' && wCopy.kind === 'E' && wB.kind === 'E') {
        expectBitExactArrays(wA.W_router, wCopy.W_router, `Bay ${bayId} W_router`);
        expect(wA.W_router[0]).not.toBe(wB.W_router[0]);
      } else if (wA.kind === '*' && wCopy.kind === '*' && wB.kind === '*') {
        expectBitExactArrays(wA.W_Q, wCopy.W_Q, `Bay ${bayId} W_Q`);
        expect(wA.W_Q[0]).not.toBe(wB.W_Q[0]);
      }
    }

    // Shared cache keyed by executionId never collides across distinct models on identical tokenIds
    const sharedCache = new HybridExecutionCache(8);
    const execA = getOrRunHybridExecution(modelA, sharedCache, { tokenIds: [0, 1, 2, 3] });
    const execB = getOrRunHybridExecution(modelB, sharedCache, { tokenIds: [0, 1, 2, 3] });
    expect(execA.executionId).not.toBe(execB.executionId);
    expect(sharedCache.size).toBe(2);
    expect(execA.lmHead.logits[0]).not.toBe(execB.lmHead.logits[0]);

    // Bay weights remain strictly independent of sequence length and tokenIds
    const wBay2Before = new Float32Array(
      (modelA.getBayWeights(2) as Extract<ReturnType<typeof modelA.getBayWeights>, { kind: 'E' }>).W_router,
    );
    runHybridExecution(modelA, { tokenIds: [9, 3, 14, 1, 7, 2] });
    const wBay2After = (modelA.getBayWeights(2) as Extract<ReturnType<typeof modelA.getBayWeights>, { kind: 'E' }>).W_router;
    expectBitExactArrays(wBay2After, wBay2Before, 'Bay 2 W_router unchanged across executions');
  });

  it('2. executes exact 52-layer HYBRID_OVERRIDE_PATTERN with bit-exact residual continuity and lazy bay reconstruction', () => {
    const model = createDefaultHybridModel();
    const exec = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(8) });

    expect(exec.residualStreams).toHaveLength(53);
    expect(exec.baySummaries).toHaveLength(52);
    for (let k = 1; k <= 52; k++) {
      expect(exec.baySummaries[k - 1].bayId).toBe(k);
      expect(exec.baySummaries[k - 1].kind).toBe(HYBRID_OVERRIDE_PATTERN[k - 1]);
    }

    // Elementwise bit-exact cross-bay residual continuity (NOT relying on JS object identity)
    expectBitExactArrays(exec.X_embed, exec.residualStreams[0], 'X_embed === stream[0]');
    const bay01 = exec.getMambaTrace(1);
    const bay02 = exec.getMoeTrace(2);
    const bay06 = exec.getGqaTrace(6);
    const bay42 = exec.getMambaTrace(42);
    const bay43 = exec.getGqaTrace(43);
    const bay52 = exec.getMoeTrace(52);

    expectBitExactArrays(bay01.X, exec.residualStreams[0], 'Bay 01 X === X_embed');
    expectBitExactArrays(bay01.Y, bay02.X, 'Bay 01 Y === Bay 02 X');
    expectBitExactArrays(bay02.Y, exec.residualStreams[2], 'Bay 02 Y === stream[2]');
    expectBitExactArrays(bay06.X, exec.residualStreams[5], 'Bay 06 X === stream[5]');
    expectBitExactArrays(bay06.Y, exec.residualStreams[6], 'Bay 06 Y === stream[6]');
    expectBitExactArrays(bay42.Y, bay43.X, 'Bay 42 Feeder Y === Bay 43 GQA X');
    expectBitExactArrays(bay52.Y, exec.residualStreams[52], 'Bay 52 Final MoE Y === stream[52]');

    // Lazy deterministic reconstruction of non-retained intermediate bays matches preserved checkpoints bit-exact
    expect(exec.retainedTraceBayIds()).toEqual([1, 2, 6, 42, 43, 52]);
    for (const lazyBay of [13, 25, 38]) {
      const tr = exec.getLayerTrace(lazyBay);
      expect(tr.executionId).toBe(exec.executionId);
      expect(tr.bayId).toBe(lazyBay);
      expectBitExactArrays(tr.sim.X, exec.residualStreams[lazyBay - 1], `Lazy Bay ${lazyBay} X`);
      expectBitExactArrays(tr.sim.Y, exec.residualStreams[lazyBay], `Lazy Bay ${lazyBay} Y`);
    }
  });

  it('3. upholds strict end-to-end causality across all 52 layers and exact prefix consistency (T=8 vs T=16)', () => {
    const model = createDefaultHybridModel();
    const baseIds = defaultTokenIdsForLength(8);
    const mutatedIds = [...baseIds];
    mutatedIds[5] = 24;
    mutatedIds[7] = 19;

    const baseExec = runHybridExecution(model, { tokenIds: baseIds });
    const mutExec = runHybridExecution(model, { tokenIds: mutatedIds });

    // Every token t < 5 must be bit-exact across ALL 53 residual checkpoints and LM head outputs
    for (let k = 0; k <= 52; k++) {
      const sA = baseExec.residualStreams[k];
      const sB = mutExec.residualStreams[k];
      for (let t = 0; t < 5; t++) {
        for (let i = 0; i < 8; i++) {
          expect(sA[i * 8 + t], `stream[${k}] ch=${i} t=${t}`).toBe(sB[i * 8 + t]);
        }
      }
    }
    for (let t = 0; t < 5; t++) {
      for (let v = 0; v < 32; v++) {
        expect(baseExec.lmHead.logits[v * 8 + t]).toBe(mutExec.lmHead.logits[v * 8 + t]);
        expect(baseExec.lmHead.probs[v * 8 + t]).toBe(mutExec.lmHead.probs[v * 8 + t]);
      }
    }
    // Mutated token t=5 must genuinely diverge at Bay 01 and Output
    let diffAt5 = 0;
    for (let i = 0; i < 8; i++) {
      diffAt5 += Math.abs(baseExec.residualStreams[52][i * 8 + 5] - mutExec.residualStreams[52][i * 8 + 5]);
    }
    expect(diffAt5).toBeGreaterThan(1e-3);

    // Prefix consistency: columns 0..7 of T=16 match T=8 bit-exact across all 53 streams and logits
    const exec16 = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(16) });
    for (let k = 0; k <= 52; k++) {
      const s8 = baseExec.residualStreams[k];
      const s16 = exec16.residualStreams[k];
      for (let t = 0; t < 8; t++) {
        for (let i = 0; i < 8; i++) {
          expect(s16[i * 16 + t], `prefix stream[${k}] ch=${i} t=${t}`).toBe(s8[i * 8 + t]);
        }
      }
    }
    for (let t = 0; t < 8; t++) {
      for (let v = 0; v < 32; v++) {
        expect(exec16.lmHead.logits[v * 16 + t]).toBe(baseExec.lmHead.logits[v * 8 + t]);
      }
    }
  });

  it('4. propagates custom tokenIds/labels without hardcoded leakage, enforces NoPE divergence via Mamba-2, and exposes GQA rmsEps=1e-5', () => {
    const model = createDefaultHybridModel();
    const customIds = [2, 2, 18, 2];
    const customLabels = ['alpha', 'alpha_dup1', 'gamma', 'alpha_dup2'];
    const exec = runHybridExecution(model, { tokenIds: customIds, tokenLabels: customLabels });

    // Custom tokenIds and labels propagate through EVERY simulator result without hardcoded EDUCATIONAL_TOKENS leakage
    for (const sim of [exec.getMambaTrace(1), exec.getMoeTrace(2), exec.getGqaTrace(6)]) {
      expect(Array.from(sim.tokenIds)).toEqual(customIds);
      expect(sim.tokenLabels).toEqual(customLabels);
      expect(sim.tokenLabels).not.toContain('How');
    }

    // Strict NoPE at Embedding: duplicate token ID 2 has bit-identical vectors at t=0, t=1, t=3
    for (let i = 0; i < 8; i++) {
      expect(exec.X_embed[i * 4 + 0]).toBe(exec.X_embed[i * 4 + 1]);
      expect(exec.X_embed[i * 4 + 0]).toBe(exec.X_embed[i * 4 + 3]);
    }
    // Bay 01 Mamba-2 (Conv1D + SSM) breaks symmetry immediately so downstream NoPE GQA Bay 06 sees position-differentiated tokens!
    let diff01 = 0;
    let diff13 = 0;
    for (let i = 0; i < 8; i++) {
      diff01 += Math.abs(exec.residualStreams[1][i * 4 + 0] - exec.residualStreams[1][i * 4 + 1]);
      diff13 += Math.abs(exec.residualStreams[1][i * 4 + 1] - exec.residualStreams[1][i * 4 + 3]);
    }
    expect(diff01).toBeGreaterThan(1e-3);
    expect(diff13).toBeGreaterThan(1e-3);

    // GQA effective epsilon is 1e-5 in unified execution and reflected in formula text, while standalone stays 1e-6
    const unifiedGqa = exec.getGqaTrace(6);
    expect(unifiedGqa.rmsEps).toBe(1e-5);
    const unifiedFormula = describeCellFormula(unifiedGqa, { tensor: 'rmsnorm', row: 2, col: 0 });
    expect(unifiedFormula.subEquation).toContain('1e−5');
    expect(unifiedFormula.coordLabel).toContain('"gamma"(2)');

    const standaloneGqa = runGqaSim(4);
    expect(standaloneGqa.rmsEps).toBe(1e-6);
    const standaloneFormula = describeCellFormula(standaloneGqa, { tensor: 'rmsnorm', row: 2, col: 0 });
    expect(standaloneFormula.subEquation).toContain('1e−6');
  });

  it('5. verifies untied LM head, epsilon-accounted Final RMSNorm, bounded cache eviction, conditioning, and routing diversity', () => {
    const model = createDefaultHybridModel();

    // Untied W_lm_head !== W_embed
    let embedDiff = 0;
    for (let i = 0; i < model.W_embed.length; i++) {
      embedDiff += Math.abs(model.W_embed[i] - model.W_lm_head[i]);
    }
    expect(embedDiff).toBeGreaterThan(10);

    // Numerical conditioning & epsilon-accounted Final RMSNorm across T = 1, 8, 16
    for (const T of [1, 8, 16]) {
      const exec = runHybridExecution(model, { tokenIds: defaultTokenIdsForLength(T) });

      for (let k = 0; k <= 52; k++) {
        const stream = exec.residualStreams[k];
        for (let t = 0; t < T; t++) {
          let sumSq = 0;
          for (let i = 0; i < 8; i++) {
            const v = stream[i * T + t];
            expect(Number.isFinite(v), `finite stream[${k}][${i},${t}]`).toBe(true);
            sumSq += v * v;
          }
          const rms = Math.sqrt(sumSq / 8);
          expect(rms, `RMS stream[${k}] t=${t}`).toBeGreaterThanOrEqual(0.4);
          expect(rms, `RMS stream[${k}] t=${t}`).toBeLessThanOrEqual(3.5);
        }
      }

      // Epsilon-accounted Final RMSNorm identity + softmax sum = 1
      for (let t = 0; t < T; t++) {
        let meanSqX = 0;
        let meanSqNorm = 0;
        for (let i = 0; i < 8; i++) {
          const xv = exec.residualStreams[52][i * T + t];
          const nv = exec.finalNorm.X_norm[i * T + t];
          meanSqX += xv * xv;
          meanSqNorm += nv * nv;
        }
        meanSqX /= 8;
        meanSqNorm /= 8;
        expect(Math.abs(meanSqNorm - meanSqX / (meanSqX + exec.finalNorm.eps))).toBeLessThan(1e-5);

        let probSum = 0;
        for (let v = 0; v < 32; v++) {
          const p = exec.lmHead.probs[v * T + t];
          expect(Number.isFinite(p) && p > 0).toBe(true);
          probSum += p;
        }
        expect(Math.abs(probSum - 1.0)).toBeLessThan(1e-5);
        expect(exec.lmHead.topKByToken[t]).toHaveLength(5);
      }

      // Per-token MoE routing diversity on T = 16: distinct Top-6 signatures across tokens
      if (T === 16) {
        for (const bayId of [2, 52]) {
          const moe = exec.getMoeTrace(bayId);
          const uniqueSets = new Set<string>();
          const uniqueExperts = new Set<number>();
          for (let t = 0; t < T; t++) {
            const top6 = Array.from(moe.topk_indices.subarray(t * 6, (t + 1) * 6)).sort((a, b) => a - b);
            uniqueSets.add(top6.join(','));
            top6.forEach((e) => uniqueExperts.add(e));
          }
          expect(uniqueSets.size, `Bay ${bayId} distinct Top-6 sets`).toBeGreaterThanOrEqual(10);
          expect(uniqueExperts.size, `Bay ${bayId} distinct active experts across 16 tokens`).toBeGreaterThanOrEqual(24);
        }
      }
    }

    // Bounded execution cache never accumulates unbounded custom token sequences
    const boundedCache = new HybridExecutionCache(6);
    for (let i = 0; i < 20; i++) {
      getOrRunHybridExecution(model, boundedCache, {
        tokenIds: [i % 32, (i * 3 + 1) % 32, (i * 7 + 5) % 32],
      });
    }
    expect(boundedCache.size).toBe(6);
    expect(boundedCache.totalRetainedTypedArrayBytes()).toBeGreaterThan(0);
    expect(model.retainedWeightBytes()).toBeGreaterThan(0);
  });
});
