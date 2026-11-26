import { describe, expect, it } from 'vitest';
import {
  buildUniElasticD5Comparison,
  buildUniViewArchReadouts,
  formatUniArchSummaryLines,
} from '../src/uni/archReadouts';
import {
  computeMatchedSigmas,
  computeTailKlBuckets,
  createDefaultOpdDemoInputs,
  evaluateFlowEditingOpdRef,
  evaluateUnderstandingOpdRef,
  getTrainerBugDiagnosticsU28,
  UNI_OPD_AR_WEIGHT,
  UNI_OPD_GROUPED_CE_WEIGHT,
  UNI_OPD_NOISE_EPSILON,
  UNI_OPD_TAIL_FLOOR,
} from '../src/uni/opdRef';
import { UNI_EXECUTION_VIEWS } from '../src/uni/spec';
import type { UniExecutionViewId } from '../src/uni/types';

function makeNormalizedProbs(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  const exps = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.fround(exps[i] / sum);
  }
  return out;
}

describe('UNI3 Slice 3B-opd — Read-Only OPD Reference (U24 / U28)', () => {
  it('computes exact 129-bucket tail_kl (128 union + 1 tail) on V=256 with non-negative, zero-on-identical, asymmetric KL', () => {
    const demo = createDefaultOpdDemoInputs();
    expect(demo.vocabSize).toBe(256);

    const klResult = computeTailKlBuckets(demo.teacherArProbs, demo.studentArProbs);
    expect(klResult.unionIndices.length).toBe(128);
    expect(klResult.bucketCount).toBe(129);
    expect(klResult.pTargetBuckets.length).toBe(129);
    expect(klResult.pStudentBuckets.length).toBe(129);

    // Canonical strictly ascending union indices
    for (let i = 1; i < klResult.unionIndices.length; i++) {
      expect(klResult.unionIndices[i]).toBeGreaterThan(klResult.unionIndices[i - 1]!);
    }

    // Normalized bucket distributions sum to ~1 and tail masses match final bucket
    let sumT = 0;
    let sumS = 0;
    for (let b = 0; b < klResult.bucketCount; b++) {
      expect(klResult.pTargetBuckets[b]).toBeGreaterThan(0);
      expect(klResult.pStudentBuckets[b]).toBeGreaterThan(0);
      sumT += klResult.pTargetBuckets[b]!;
      sumS += klResult.pStudentBuckets[b]!;
    }
    expect(sumT).toBeCloseTo(1.0, 5);
    expect(sumS).toBeCloseTo(1.0, 5);
    expect(klResult.pTargetTailMass).toBe(klResult.pTargetBuckets[128]);
    expect(klResult.pStudentTailMass).toBe(klResult.pStudentBuckets[128]);

    // Non-negativity and genuine asymmetry
    expect(klResult.reverseKl).toBeGreaterThan(0);
    expect(klResult.forwardKl).toBeGreaterThan(0);
    expect(klResult.reverseKl).not.toBeCloseTo(klResult.forwardKl, 6);

    // Exact 0 KL on identical distributions
    const identical = computeTailKlBuckets(demo.teacherArProbs, demo.teacherArProbs);
    expect(identical.bucketCount).toBe(129);
    expect(identical.reverseKl).toBe(0);
    expect(identical.forwardKl).toBe(0);
  });

  it('preserves target top-kT first and tie-breaks smaller indices deterministically', () => {
    const t = new Float32Array([0.1, 0.25, 0.25, 0.25, 0.15]);
    const s = new Float32Array([0.4, 0.1, 0.1, 0.2, 0.2]);
    // kT=2 picks indices [1, 2] (tie-break smaller index among 0.25s)
    // kS=3 picks indices [0, 3, 4]; union limited to min(5, 3)=3 adds [0] -> sorted [0, 1, 2]
    const res = computeTailKlBuckets(t, s, 2, 3, UNI_OPD_TAIL_FLOOR);
    expect(Array.from(res.unionIndices)).toEqual([0, 1, 2]);
    expect(res.bucketCount).toBe(4);
  });

  it('evaluates read-only Understanding OPD loss: 4.0 * reverse + forward / (0.5005 * R)', () => {
    const demo = createDefaultOpdDemoInputs();
    const res = evaluateUnderstandingOpdRef(
      demo.teacherArProbs,
      demo.studentArProbs,
      demo.studentDiffProbs,
      { responseLen: 40 },
    );

    expect(res.readOnly).toBe(true);
    expect(res.hasGradients).toBe(false);
    expect(res.arOpdWeight).toBe(UNI_OPD_AR_WEIGHT);
    expect(res.noiseEpsilon).toBe(UNI_OPD_NOISE_EPSILON);
    expect(res.responseLen).toBe(40);
    expect(res.expectedMaskDenominator).toBeCloseTo(0.5005 * 40, 12);
    expect(res.arOpdLoss).toBeCloseTo(4.0 * res.arTailKl.reverseKl, 12);
    expect(res.selfConsistencyLoss).toBeCloseTo(
      res.consistencyTailKl.forwardKl / (0.5005 * 40),
      12,
    );
    expect(res.totalUnderstandingOpdLoss).toBeCloseTo(
      res.arOpdLoss + res.selfConsistencyLoss,
      12,
    );
  });

  it('computes matched_sigmas(0.9, 1024, 40) starting at 0.9, ending at 0.02, strictly decreasing', () => {
    const sigmasRes = computeMatchedSigmas(0.9, 1024, 40, 0.02);
    expect(sigmasRes.r).toBe(0.9);
    expect(sigmasRes.seqLen).toBe(1024);
    expect(sigmasRes.steps).toBe(40);
    expect(sigmasRes.sigmaStart).toBe(0.9);
    expect(sigmasRes.sigmaEnd).toBe(0.02);
    expect(sigmasRes.sigmas.length).toBe(40);

    const expectedMu = 0.5 + ((1024 - 256) / (4096 - 256)) * (1.15 - 0.5);
    expect(sigmasRes.mu).toBeCloseTo(expectedMu, 12);
    expect(sigmasRes.sigmas[0]).toBeCloseTo(0.9, 6);
    expect(sigmasRes.sigmas[39]).toBeCloseTo(0.02, 6);

    for (let i = 1; i < sigmasRes.sigmas.length; i++) {
      expect(sigmasRes.sigmas[i]).toBeLessThan(sigmasRes.sigmas[i - 1]!);
    }

    // Clamp behavior outside [256, 4096]
    expect(computeMatchedSigmas(0.9, 128, 10).mu).toBe(0.5);
    expect(computeMatchedSigmas(0.9, 8192, 10).mu).toBe(1.15);
  });

  it('evaluates read-only Flow Editing OPD: x^r = 0.1*x0 + 0.9*delta and exact_ce + 0.2 * grouped_ce', () => {
    const x0 = new Float32Array([1.0, -2.0, 0.5, 4.0]);
    const noiseDelta = new Float32Array([0.0, 1.0, -1.5, 2.0]);
    const exactLogits = new Float32Array([2.0, 0.5, -1.0, 0.0]);
    const groupLogits = new Float32Array([0.0, 1.5, -0.5]);

    const res = evaluateFlowEditingOpdRef(x0, noiseDelta, exactLogits, 0, groupLogits, 1);

    expect(res.readOnly).toBe(true);
    expect(res.hasGradients).toBe(false);
    expect(res.r).toBe(0.9);
    expect(res.groupedCeWeight).toBe(UNI_OPD_GROUPED_CE_WEIGHT);

    for (let i = 0; i < x0.length; i++) {
      expect(res.xNoisyR[i]).toBeCloseTo(Math.fround(0.1 * x0[i]! + 0.9 * noiseDelta[i]!), 6);
    }

    const pExact = makeNormalizedProbs(exactLogits);
    const pGroup = makeNormalizedProbs(groupLogits);
    const expectedExactCe = -Math.log(pExact[0]!);
    const expectedGroupCe = -Math.log(pGroup[1]!);

    expect(res.exactCe).toBeCloseTo(expectedExactCe, 5);
    expect(res.groupedCe).toBeCloseTo(expectedGroupCe, 5);
    expect(res.totalEditingOpdLoss).toBeCloseTo(res.exactCe + 0.2 * res.groupedCe, 12);
  });

  it('reports exact U28 trainer bug diagnostics and deterministic demo inputs', () => {
    const diag = getTrainerBugDiagnosticsU28();
    expect(diag.readOnly).toBe(true);

    expect(diag.sftDoubleNorm.tMin).toBe(0.001);
    expect(diag.sftDoubleNorm.tMax).toBe(1.0);
    expect(diag.sftDoubleNorm.expectedInflationFactor).toBeCloseTo(6.91467, 4);
    expect(diag.sftDoubleNorm.peakSpikeAtTMin).toBe(1000);

    expect(diag.accidentalImEndSupervision.eosTokenId).toBe(11);
    expect(diag.accidentalImEndSupervision.ignoredLabel).toBe(-100);

    expect(diag.opdLocTailLeak.fullVocabSize).toBe(132_101);
    expect(diag.opdLocTailLeak.teacherVocabSize).toBe(131_072);
    expect(diag.opdLocTailLeak.locExtensionTokens).toBe(1_029);
    expect(diag.opdLocTailLeak.topK).toBe(64);
    expect(diag.opdLocTailLeak.leakedTailDenom).toBe(132_037);
    expect(diag.opdLocTailLeak.trueTeacherTailDenom).toBe(131_008);
    expect(diag.opdLocTailLeak.leakedMassFractionOfTail).toBeCloseTo(1_029 / 132_037, 12);

    // Deterministic demo builder produces identical distributions for same seed
    const d1 = createDefaultOpdDemoInputs();
    const d2 = createDefaultOpdDemoInputs();
    expect(d1.teacherArProbs).toEqual(d2.teacherArProbs);
    expect(d1.studentArProbs).toEqual(d2.studentArProbs);
    expect(d1.studentDiffProbs).toEqual(d2.studentDiffProbs);
  });

  it('validates invalid inputs on tail_kl, matched_sigmas, and editing OPD', () => {
    expect(() =>
      computeTailKlBuckets(new Float32Array([0.5, 0.5]), new Float32Array([1.0])),
    ).toThrow(RangeError);
    expect(() =>
      computeTailKlBuckets(new Float32Array([0.5, 0.1]), new Float32Array([0.5, 0.5])),
    ).toThrow(RangeError);
    expect(() => computeMatchedSigmas(0.01, 1024, 40, 0.02)).toThrow(RangeError);
    expect(() => computeMatchedSigmas(0.9, 1024, 1, 0.02)).toThrow(RangeError);
  });
});

describe('UNI3 Slice 3B-opd — Architectural Readouts & D5 Elastic Comparison', () => {
  it('covers all 6 execution views with exact prod & miniature metrics', () => {
    const readouts = buildUniViewArchReadouts();
    expect(readouts).toHaveLength(6);

    const viewIds = readouts.map((r) => r.viewId);
    expect(viewIds).toEqual(UNI_EXECUTION_VIEWS.map((v) => v.id));

    const byId = new Map(readouts.map((r) => [r.viewId, r]));

    const textUnd = byId.get('text_understanding')!;
    expect(textUnd.prodActiveParams).toBe(8_497_983_488);
    expect(textUnd.prodKvBytesPerTokenBf16).toBe(139_264);
    expect(textUnd.prodKvAt16kBytes).toBe(139_264 * 16_384);
    expect(textUnd.miniActiveFloats).toBe(67_944);
    expect(textUnd.miniKvBytesPerTokenF32).toBe(4_352);
    expect(textUnd.prodResidentSglangBytes).toBe(18_374_765_568);

    const visUnd = byId.get('visual_understanding')!;
    expect(visUnd.prodActiveParams).toBe(9_187_382_784);
    expect(visUnd.miniActiveFloats).toBe(76_232);
    expect(visUnd.miniKvBytesPerTokenF32).toBe(4_352);

    const grounding = byId.get('object_grounding')!;
    expect(grounding.prodActiveParams).toBe(9_187_382_784);
    expect(grounding.miniActiveFloats).toBe(76_232);
    expect(grounding.miniKvBytesPerTokenF32).toBe(4_352);

    const t2i = byId.get('text_to_image')!;
    expect(t2i.prodActiveParams).toBe(13_362_667_523);
    expect(t2i.prodEagerHfParams).toBe(16_852_463_619);
    expect(t2i.prodWastedSplitUParams).toBe(3_489_796_096);
    expect(t2i.prodKvBytesPerTokenBf16).toBe(278_528);
    expect(t2i.miniActiveFloats).toBe(131_131);
    expect(t2i.miniKvBytesPerTokenF32).toBe(8_704);
    expect(t2i.prodResidentSglangBytes).toBeNull();

    const edit = byId.get('image_edit')!;
    expect(edit.prodActiveParams).toBe(14_236_666_371);
    expect(edit.prodEagerHfParams).toBe(17_726_462_467);
    expect(edit.prodKvBytesPerTokenBf16).toBe(417_792);
    expect(edit.miniActiveFloats).toBe(159_219);
    expect(edit.miniKvBytesPerTokenF32).toBe(13_056);

    const interleaved = byId.get('interleaved_mixed')!;
    expect(interleaved.prodActiveParams).toBe(17_122_811_904);
    expect(interleaved.miniActiveFloats).toBe(139_552);
    expect(interleaved.miniKvBytesPerTokenF32).toBe(4_352);
    expect(interleaved.releasedEndToEndLoop).toBe(false);
    expect(interleaved.badge).toBe('Kernel-Supported · No Released Multi-Turn Outer Loop');
  });

  it('computes exact D5 comparison for text_to_image (16/34 wasted), image_edit, and understanding (0 wasted)', () => {
    const t2i = buildUniElasticD5Comparison('text_to_image', 16);
    expect(t2i.cfgBranches).toBe(2);
    expect(t2i.hfFaithfulLayersExecuted).toBe(34);
    expect(t2i.elasticPrunedLayersExecuted).toBe(18);
    expect(t2i.prodHfFaithfulPrefixKvBytes).toBe(16 * 2 * 34 * 4_096);
    expect(t2i.prodElasticPrunedPrefixKvBytes).toBe(16 * 2 * 18 * 4_096);
    expect(t2i.prodSavedPrefixKvBytes).toBe(16 * 2 * 16 * 4_096);
    expect(t2i.miniHfFaithfulPrefixKvBytes).toBe(16 * 2 * 34 * 128);
    expect(t2i.miniElasticPrunedPrefixKvBytes).toBe(16 * 2 * 18 * 128);
    expect(t2i.miniSavedPrefixKvBytes).toBe(16 * 2 * 16 * 128);
    expect(t2i.wastedPrefixKvFraction).toBeCloseTo(16 / 34, 12);
    expect(t2i.stage2UlpDiff).toBe(0);

    const edit = buildUniElasticD5Comparison('image_edit', 16);
    expect(edit.cfgBranches).toBe(3);
    expect(edit.hfFaithfulLayersExecuted).toBe(34);
    expect(edit.elasticPrunedLayersExecuted).toBe(34);
    expect(edit.prodSavedPrefixKvBytes).toBe(16 * 2 * 16 * 4_096);
    expect(edit.miniSavedPrefixKvBytes).toBe(16 * 2 * 16 * 128);
    expect(edit.wastedPrefixKvFraction).toBeCloseTo((2 / 3) * (16 / 34), 12);
    expect(edit.stage2UlpDiff).toBe(0);

    for (const viewId of [
      'text_understanding',
      'visual_understanding',
      'object_grounding',
      'interleaved_mixed',
    ] as const satisfies readonly UniExecutionViewId[]) {
      const cmp = buildUniElasticD5Comparison(viewId, 16);
      expect(cmp.hfFaithfulLayersExecuted).toBe(34);
      expect(cmp.elasticPrunedLayersExecuted).toBe(34);
      expect(cmp.prodSavedPrefixKvBytes).toBe(0);
      expect(cmp.miniSavedPrefixKvBytes).toBe(0);
      expect(cmp.wastedPrefixKvFraction).toBe(0);
      expect(cmp.stage2UlpDiff).toBe(0);
    }

    expect(() => buildUniElasticD5Comparison('text_to_image', -1)).toThrow(RangeError);
  });

  it('formats dual SI/IEC architectural summary lines cleanly', () => {
    const t2iLines = formatUniArchSummaryLines('text_to_image', 'elastic_pruned', 16);
    expect(t2iLines.length).toBeGreaterThanOrEqual(6);
    expect(t2iLines.some((line) => line.includes('GB') && line.includes('GiB'))).toBe(true);
    expect(t2iLines.some((line) => line.includes('16K (16,384)'))).toBe(true);
    expect(t2iLines.some((line) => line.includes('elastic_pruned') && line.includes('0 ULP'))).toBe(
      true,
    );

    const interleavedLines = formatUniArchSummaryLines('interleaved_mixed', 'hf_faithful', 16);
    expect(
      interleavedLines.some((line) =>
        line.includes('Kernel-Supported · No Released Multi-Turn Outer Loop'),
      ),
    ).toBe(true);
  });
});
