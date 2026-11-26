import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  applyRoutedUniRmsNorm,
  applyUniRmsNorm,
  formatUniAttentionCellFormula,
  linearProject,
  routedLinearProject,
  runUniElasticAttention,
  softmaxRows,
} from '../src/uni/attention';
import {
  applyElasticSplitTowerMask,
  buildBidirectionalMask,
  buildCausalMask,
  buildDualStreamBlockDiffusionMask,
  buildSbdImageMask,
  buildUniAttentionMaskMatrix,
  canonicalizeTokenRouting,
} from '../src/uni/masks';
import {
  applyUniRotaryPosEmb,
  compute2dSinCosPosEmbed,
  computeLlama4AttnScale,
  computeLlama4AttnScales,
  computeUniRopeCosSin,
  computeUniYarnInvFreq,
  formatUniRopeCellFormula,
  getUniYarnMscale,
} from '../src/uni/rope';
import {
  formatUniSwigluCellFormula,
  runUniBranchSwiglu,
  runUniRoutedSwiglu,
  silu,
} from '../src/uni/swiglu';
import {
  UNI_DYADIC_YARN_INV_FREQ,
  UNI_MINI_CONFIG,
  type UniBranchAttentionWeights,
  type UniBranchLayerWeights,
  type UniBranchSwigluWeights,
  type UniElasticLayerWeights,
  type UniLayerKvSlice,
} from '../src/uni/types';
import {
  assertBitExactFiniteArray,
  assertFiniteArrayParity,
} from './helpers/parityAssert';

function deterministicFloat32(
  length: number,
  seed: number,
  scale = 0.25,
): Float32Array {
  const out = new Float32Array(length);
  let s = seed >>> 0;
  for (let i = 0; i < length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const u = (s / 4294967296) * 2 - 1;
    out[i] = Math.fround(u * scale);
  }
  return out;
}

function makeBranchAttnWeights(seed: number): UniBranchAttentionWeights {
  return {
    qProj: deterministicFloat32(64 * 8, seed + 1, 0.2),
    kProj: deterministicFloat32(16 * 8, seed + 2, 0.2),
    vProj: deterministicFloat32(16 * 8, seed + 3, 0.2),
    oProj: deterministicFloat32(8 * 64, seed + 4, 0.1),
  };
}

function makeBranchSwigluWeights(seed: number): UniBranchSwigluWeights {
  return {
    gateProj: deterministicFloat32(28 * 8, seed + 5, 0.2),
    upProj: deterministicFloat32(28 * 8, seed + 6, 0.2),
    downProj: deterministicFloat32(8 * 28, seed + 7, 0.1),
  };
}

function makeBranchLayerWeights(seed: number): UniBranchLayerWeights {
  const normU = new Float32Array(8);
  const normPost = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    normU[i] = Math.fround(1.0 + 0.05 * ((i % 3) - 1));
    normPost[i] = Math.fround(1.0 - 0.04 * ((i % 2) - 0.5));
  }
  return {
    inputLayerNorm: normU,
    selfAttn: makeBranchAttnWeights(seed),
    postAttentionLayerNorm: normPost,
    mlp: makeBranchSwigluWeights(seed),
  };
}

function makeElasticLayerWeights(
  layerIndex: number,
  seed = 1000 + layerIndex * 31,
): UniElasticLayerWeights {
  return {
    layerIndex,
    splitTower: layerIndex >= UNI_MINI_CONFIG.dualTowerLayers,
    u: makeBranchLayerWeights(seed),
    g: makeBranchLayerWeights(seed + 500),
  };
}

function extractRows(
  buf: Float32Array,
  rowWidth: number,
  rowIndices: number[],
): Float32Array {
  const out = new Float32Array(rowIndices.length * rowWidth);
  for (let i = 0; i < rowIndices.length; i++) {
    const r = rowIndices[i];
    out.set(buf.subarray(r * rowWidth, (r + 1) * rowWidth), i * rowWidth);
  }
  return out;
}

describe('UNI1 Slice 1B: 1D YaRN RoPE, Llama-4 Scaling & 2D SinCos', () => {
  it('computes exact 0-ULP dyadic YaRN invFreq, zones, and attentionScaling for UNI_MINI_CONFIG.rope', () => {
    const res = computeUniYarnInvFreq(UNI_MINI_CONFIG.rope);
    expect(res.low).toBe(0);
    expect(res.high).toBe(2);
    expect(res.attentionScaling).toBe(1.0);
    expect(res.zones).toEqual([
      'original',
      'blended',
      'interpolated',
      'interpolated',
    ]);
    assertBitExactFiniteArray(
      res.extrapolationFactor,
      [1.0, 0.5, 0.0, 0.0],
      'yarn.extrapolationFactor',
    );
    assertBitExactFiniteArray(
      res.invFreq,
      UNI_DYADIC_YARN_INV_FREQ,
      'yarn.invFreq',
    );
    expect(Array.from(res.invFreq)).toEqual([1.0, 0.28125, 0.03125, 0.015625]);
    expect(getUniYarnMscale(8, 1)).toBeCloseTo(1.0 + 0.1 * Math.log(8), 12);
    expect(getUniYarnMscale(1, 1)).toBe(1.0);
  });

  it('computes half-split rotate_half [-x2, x1] and preserves per-head L2 norm', () => {
    const yarn = computeUniYarnInvFreq(UNI_MINI_CONFIG.rope);
    const positionIds = new Int32Array([0, 3, 17]);
    const { cos, sin, seqLen, headDim } = computeUniRopeCosSin(
      yarn.invFreq,
      positionIds,
      yarn.attentionScaling,
      UNI_MINI_CONFIG.headDim,
    );
    expect(seqLen).toBe(3);
    expect(headDim).toBe(8);

    // Position 0: cos = 1, sin = 0 -> identity rotation
    for (let d = 0; d < 8; d++) {
      expect(cos[d]).toBe(1.0);
      expect(sin[d]).toBe(0.0);
    }

    const numHeads = 2;
    const x = deterministicFloat32(seqLen * numHeads * headDim, 42, 1.0);
    const rotated = applyUniRotaryPosEmb(x, seqLen, numHeads, headDim, cos, sin);

    // Token 0 must match input bit-exact
    assertBitExactFiniteArray(
      rotated.subarray(0, numHeads * headDim),
      x.subarray(0, numHeads * headDim),
      'rope.pos0.identity',
    );

    // Verify exact half-split [-x2, x1] equation & norm preservation on tokens 1 & 2
    for (let t = 0; t < seqLen; t++) {
      for (let h = 0; h < numHeads; h++) {
        let normIn = 0;
        let normOut = 0;
        const base = (t * numHeads + h) * headDim;
        for (let i = 0; i < 4; i++) {
          const x1 = x[base + i];
          const x2 = x[base + 4 + i];
          const c = cos[t * headDim + i];
          const s = sin[t * headDim + i];
          const expectedFirst = Math.fround(
            Math.fround(x1 * c) + Math.fround(-x2 * s),
          );
          const expectedSecond = Math.fround(
            Math.fround(x2 * c) + Math.fround(x1 * s),
          );
          expect(rotated[base + i]).toBe(expectedFirst);
          expect(rotated[base + 4 + i]).toBe(expectedSecond);
          normIn += x1 * x1 + x2 * x2;
          normOut +=
            rotated[base + i] * rotated[base + i] +
            rotated[base + 4 + i] * rotated[base + 4 + i];
        }
        expect(normOut).toBeCloseTo(normIn, 5);
      }
    }

    const formula = formatUniRopeCellFormula({
      role: 'Q',
      tokenIndex: 1,
      positionId: 17,
      headIndex: 0,
      dimIndex: 1,
      headDim: 8,
      invFreq: yarn.invFreq[1],
      zone: yarn.zones[1],
      x1: 0.5,
      x2: -0.25,
      cosVal: cos[headDim + 1],
      sinVal: sin[headDim + 1],
      rotatedVal: 0.4,
      llama4Scale: 1.0693,
      scaledVal: 0.4277,
      branch: 'G',
    });
    expect(formula).toContain('G:Q_rot[p=17, h=0, d=1]');
    expect(formula).toContain('blended');
    expect(formula).toContain('×s_L4');
  });

  it('computes Llama-4 query temperature scale 1.0 on pos 0..15 and 1.0693147 on pos 16..31', () => {
    const beta = UNI_MINI_CONFIG.rope.llama4Beta;
    const origMaxPos = UNI_MINI_CONFIG.rope.originalMaxPositionEmbeddings;
    const expectedBucket1 = 1.0 + beta * Math.log(2.0);
    expect(expectedBucket1).toBeCloseTo(1.0693147180559945, 12);

    const pos = new Int32Array(36);
    for (let i = 0; i < 36; i++) pos[i] = i;
    const scales = computeLlama4AttnScales(pos, beta, origMaxPos);

    for (let p = 0; p < 16; p++) {
      expect(computeLlama4AttnScale(p, beta, origMaxPos)).toBe(1.0);
      expect(scales[p]).toBe(1.0);
    }
    for (let p = 16; p < 32; p++) {
      expect(computeLlama4AttnScale(p, beta, origMaxPos)).toBeCloseTo(
        expectedBucket1,
        12,
      );
      expect(scales[p]).toBe(Math.fround(expectedBucket1));
    }
    for (let p = 32; p < 36; p++) {
      expect(scales[p]).toBe(Math.fround(1.0 + beta * Math.log(3.0)));
    }
  });

  it('computes exact 2D SinCos frequencies and 0-ULP content-column match for markerColumn=true vs false', () => {
    const marked = compute2dSinCosPosEmbed(2, 3, 8, true);
    const unmarked = compute2dSinCosPosEmbed(2, 2, 8, false);

    expect(marked.height).toBe(2);
    expect(marked.width).toBe(3);
    expect(marked.contentWidth).toBe(2);
    expect(marked.hiddenSize).toBe(8);
    expect(marked.columnStride).toBe(16.0);
    expect(marked.rowStride).toBe(16.0);
    expect(marked.embeddings.length).toBe(2 * 3 * 8);

    expect(unmarked.contentWidth).toBe(2);
    expect(unmarked.columnStride).toBe(16.0);
    expect(unmarked.rowStride).toBe(16.0);
    expect(unmarked.embeddings.length).toBe(2 * 2 * 8);

    // Frequencies: baseTheta^(-0/2) = 1.0, baseTheta^(-1/2) = fround(0.01)
    assertBitExactFiniteArray(
      marked.frequencies,
      [Math.fround(1.0), Math.fround(0.01)],
      'sincos2d.frequencies',
    );
    assertBitExactFiniteArray(
      unmarked.frequencies,
      marked.frequencies,
      'sincos2d.freqMatch',
    );

    // 0-ULP bit-exact match on content columns c in {0, 1} across every row r in {0, 1}
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const mCell = marked.embeddings.subarray(
          (r * 3 + c) * 8,
          (r * 3 + c + 1) * 8,
        );
        const uCell = unmarked.embeddings.subarray(
          (r * 2 + c) * 8,
          (r * 2 + c + 1) * 8,
        );
        assertBitExactFiniteArray(
          mCell,
          uCell,
          `sincos2d.content_r${r}_c${c}`,
        );
      }
    }

    // Cell (0, 0): colCoord = 0, rowCoord = 0 -> [0, 0, scale, scale, 0, 0, scale, scale]
    const s = Math.fround(0.0018);
    assertBitExactFiniteArray(
      unmarked.embeddings.subarray(0, 8),
      [0, 0, s, s, 0, 0, s, s],
      'sincos2d.cell_0_0',
    );

    // Validation errors
    expect(() => compute2dSinCosPosEmbed(2, 1, 8, true)).toThrow(RangeError);
    expect(() => compute2dSinCosPosEmbed(2, 2, 6, false)).toThrow(RangeError);
  });
});

describe('UNI1 Slice 1B: Routing & Elastic-MoT / SBD / OPD Attention Masks', () => {
  it('canonicalizes U, G, and mixed routing plans and rejects invalid entries', () => {
    const uPlan = canonicalizeTokenRouting([0, 0, 0]);
    expect(uPlan.canonical).toBe('u_only');
    expect(uPlan.hasUnderstanding).toBe(true);
    expect(uPlan.hasGeneration).toBe(false);
    expect(Array.from(uPlan.understandingIndices)).toEqual([0, 1, 2]);
    expect(Array.from(uPlan.generationIndices)).toEqual([]);

    const gPlan = canonicalizeTokenRouting([1, 1]);
    expect(gPlan.canonical).toBe('g_only');
    expect(gPlan.hasUnderstanding).toBe(false);
    expect(gPlan.hasGeneration).toBe(true);
    expect(Array.from(gPlan.generationIndices)).toEqual([0, 1]);

    const mPlan = canonicalizeTokenRouting([0, 1, 0, 1, 1]);
    expect(mPlan.canonical).toBe('mixed');
    expect(Array.from(mPlan.understandingIndices)).toEqual([0, 2]);
    expect(Array.from(mPlan.generationIndices)).toEqual([1, 3, 4]);

    expect(() => canonicalizeTokenRouting([0, 2, 1])).toThrow(RangeError);
    expect(() => canonicalizeTokenRouting([-1])).toThrow(RangeError);
  });

  it('builds causal, bidirectional, SBD image (excluding trailing tokens), and OPD dual-stream masks', () => {
    const causal = buildCausalMask(3, 2);
    expect(Array.from(causal)).toEqual([
      1, 1, 1, 0, 0,
      1, 1, 1, 1, 0,
      1, 1, 1, 1, 1,
    ]);

    const bidir = buildBidirectionalMask(2, 2);
    expect(Array.from(bidir)).toEqual([1, 1, 1, 1, 1, 1, 1, 1]);

    // SBD image: totalLen=6, prefix=[0,1], image=[2,3,4], trailing=[5]
    const sbd = buildSbdImageMask(6, 2, 5);
    expect(Array.from(sbd)).toEqual([
      // q=0,1: causal prefix
      1, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0,
      // q=2,3,4: bidirectional across prefix+image, excludes trailing col 5
      1, 1, 1, 1, 1, 0,
      1, 1, 1, 1, 1, 0,
      1, 1, 1, 1, 1, 0,
      // q=5: causal trailing attends to all 0..5
      1, 1, 1, 1, 1, 1,
    ]);

    // OPD dual-stream: cleanLen=4, blockSize=2 -> totalLen=8 ([noisy 0..3, clean 4..7])
    const opd = buildDualStreamBlockDiffusionMask(4, 2);
    expect(Array.from(opd)).toEqual([
      // noisy block 0 (q=0,1): same noisy block (0,1), no clean blocks (<0)
      1, 1, 0, 0, 0, 0, 0, 0,
      1, 1, 0, 0, 0, 0, 0, 0,
      // noisy block 1 (q=2,3): same noisy block (2,3) + clean block 0 (4,5)
      0, 0, 1, 1, 1, 1, 0, 0,
      0, 0, 1, 1, 1, 1, 0, 0,
      // clean tokens (q=4..7): causal over clean (4..7) only
      0, 0, 0, 0, 1, 0, 0, 0,
      0, 0, 0, 0, 1, 1, 0, 0,
      0, 0, 0, 0, 1, 1, 1, 0,
      0, 0, 0, 0, 1, 1, 1, 1,
    ]);
  });

  it('separates U and G into disjoint islands at L18..33 while keeping L0..17 joint', () => {
    const routing = new Uint8Array([0, 1, 0, 1]);
    const base = buildCausalMask(4, 0);

    const jointL17 = applyElasticSplitTowerMask(base, routing, routing, 17);
    expect(Array.from(jointL17)).toEqual(Array.from(base));

    const splitL18 = applyElasticSplitTowerMask(base, routing, routing, 18);
    expect(Array.from(splitL18)).toEqual([
      1, 0, 0, 0, // q=0 (U): k=0 (U)
      0, 1, 0, 0, // q=1 (G): k=1 (G) — k=0 (U) masked!
      1, 0, 1, 0, // q=2 (U): k=0,2 (U) — k=1 (G) masked!
      0, 1, 0, 1, // q=3 (G): k=1,3 (G) — k=0,2 (U) masked!
    ]);

    // Explorer matrix wrapper across modes
    const mJoint = buildUniAttentionMaskMatrix({
      mode: 'split_tower_mixed',
      seqLen: 4,
      queryRouting: routing,
      kvRouting: routing,
      layerIndex: 17,
    });
    expect(mJoint.splitTowerApplied).toBe(false);
    expect(Array.from(mJoint.visible)).toEqual(Array.from(base));

    const mSplit = buildUniAttentionMaskMatrix({
      mode: 'split_tower_mixed',
      seqLen: 4,
      queryRouting: routing,
      kvRouting: routing,
      layerIndex: 18,
    });
    expect(mSplit.splitTowerApplied).toBe(true);
    expect(Array.from(mSplit.visible)).toEqual(Array.from(splitL18));

    const mOpd = buildUniAttentionMaskMatrix({
      mode: 'opd_dual_stream',
      seqLen: 4,
      blockSize: 2,
    });
    expect(mOpd.isReadOnlyReference).toBe(true);
    expect(mOpd.qLen).toBe(8);
    expect(mOpd.kvLen).toBe(8);
  });
});

describe('UNI1 Slice 1B: Routed RMSNorm, 8Q/2KV GQA, D5 0-ULP Invariance & SwiGLU', () => {
  it('computes routed RMSNorm and skips U weight reads when skipU=true', () => {
    const seqLen = 4;
    const d = 8;
    const x = deterministicFloat32(seqLen * d, 77, 1.0);
    const routing = new Uint8Array([0, 1, 0, 1]);
    const weightU = new Float32Array([1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 1.1, 0.95]);
    const weightG = new Float32Array([0.9, 1.15, 1.05, 0.85, 1.2, 0.9, 1.0, 1.1]);
    const eps = UNI_MINI_CONFIG.rmsNormEps;

    const full = applyRoutedUniRmsNorm(x, seqLen, d, routing, weightU, weightG, eps);
    const refU = applyUniRmsNorm(x, seqLen, d, weightU, eps);
    const refG = applyUniRmsNorm(x, seqLen, d, weightG, eps);

    assertBitExactFiniteArray(
      extractRows(full, d, [0, 2]),
      extractRows(refU, d, [0, 2]),
      'rmsNorm.U',
    );
    assertBitExactFiniteArray(
      extractRows(full, d, [1, 3]),
      extractRows(refG, d, [1, 3]),
      'rmsNorm.G',
    );

    // Poison weightU with NaNs to prove skipU=true never reads weightU
    const poisonedWeightU = new Float32Array(d).fill(Number.NaN);
    const pruned = applyRoutedUniRmsNorm(
      x,
      seqLen,
      d,
      routing,
      poisonedWeightU,
      weightG,
      eps,
      true,
    );
    assertBitExactFiniteArray(
      extractRows(pruned, d, [0, 2]),
      new Float32Array(2 * d),
      'rmsNorm.skipU.zeroed',
    );
    assertBitExactFiniteArray(
      extractRows(pruned, d, [1, 3]),
      extractRows(full, d, [1, 3]),
      'rmsNorm.skipU.G_exact',
    );
  });

  it('handles softmaxRows -Infinity and all-masked rows with exact 0.0', () => {
    const scores = new Float32Array([
      1.0, -Infinity, 3.0,
      -Infinity, -Infinity, -Infinity,
    ]);
    const probs = softmaxRows(scores, 2, 3);
    expect(probs[1]).toBe(0.0);
    expect(probs[0] + probs[2]).toBeCloseTo(1.0, 6);
    expect(Array.from(probs.subarray(3, 6))).toEqual([0.0, 0.0, 0.0]);
  });

  it('guarantees D5 0-ULP invariance for dropCachedUnderstanding (null vs non-null U pastKv at L18)', () => {
    const layer18 = makeElasticLayerWeights(18);
    const yarn = computeUniYarnInvFreq(UNI_MINI_CONFIG.rope);
    const pastLen = 5;
    const seqLen = 4;
    const hiddenStates = deterministicFloat32(seqLen * 8, 303, 0.8);
    const routing = new Uint8Array([1, 1, 1, 1]);
    const pastRouting = new Uint8Array(pastLen).fill(0);
    const positionIds = new Int32Array([5, 6, 7, 8]);

    // Non-null U pastKv (simulating hf_faithful 34L prefill) + NaN-poisoned check
    const pastK = deterministicFloat32(pastLen * 16, 901, 0.5);
    const pastV = deterministicFloat32(pastLen * 16, 902, 0.5);
    const pastKSaved = new Float32Array(pastK);
    const pastVSaved = new Float32Array(pastV);
    const pastRoutingSaved = new Uint8Array(pastRouting);
    const pastKv: UniLayerKvSlice = { k: pastK, v: pastV, seqLen: pastLen };

    const poisonedPastKv: UniLayerKvSlice = {
      k: new Float32Array(pastLen * 16).fill(Number.NaN),
      v: new Float32Array(pastLen * 16).fill(Number.NaN),
      seqLen: pastLen,
    };

    const traceHfFaithful = runUniElasticAttention({
      hiddenStates,
      seqLen,
      routing,
      layerWeights: layer18,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKv,
      pastRouting,
    });

    const tracePrunedNull = runUniElasticAttention({
      hiddenStates,
      seqLen,
      routing,
      layerWeights: layer18,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKv: null,
      pastRouting,
    });

    const tracePoisoned = runUniElasticAttention({
      hiddenStates,
      seqLen,
      routing,
      layerWeights: layer18,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKv: poisonedPastKv,
      pastRouting,
    });

    expect(traceHfFaithful.dropCachedUnderstanding).toBe(true);
    expect(tracePrunedNull.dropCachedUnderstanding).toBe(true);
    expect(traceHfFaithful.cachedHasGeneration).toBe(false);
    expect(traceHfFaithful.pastLen).toBe(pastLen);
    expect(traceHfFaithful.kvLen).toBe(pastLen + seqLen);

    // Every single column 0..pastLen-1 in effectiveMask & probs must be 0
    for (let q = 0; q < seqLen; q++) {
      for (let k = 0; k < pastLen; k++) {
        expect(tracePrunedNull.effectiveMask[q * (pastLen + seqLen) + k]).toBe(0);
      }
      for (let k = pastLen; k < pastLen + seqLen; k++) {
        expect(tracePrunedNull.effectiveMask[q * (pastLen + seqLen) + k]).toBe(1);
      }
    }

    assertBitExactFiniteArray(
      tracePrunedNull.probs,
      traceHfFaithful.probs,
      'd5.dropU.probs',
    );
    assertBitExactFiniteArray(
      tracePrunedNull.attnConcat,
      traceHfFaithful.attnConcat,
      'd5.dropU.attnConcat',
    );
    assertBitExactFiniteArray(
      tracePrunedNull.output,
      traceHfFaithful.output,
      'd5.dropU.output',
    );
    assertBitExactFiniteArray(
      tracePoisoned.output,
      traceHfFaithful.output,
      'd5.dropU.poisonedPastKv',
    );

    // Non-mutation of pastKv & pastRouting
    assertBitExactFiniteArray(pastK, pastKSaved, 'pastK.unmutated');
    assertBitExactFiniteArray(pastV, pastVSaved, 'pastV.unmutated');
    expect(Array.from(pastRouting)).toEqual(Array.from(pastRoutingSaved));
  });

  it('guarantees D5 0-ULP invariance for pruneUnderstandingTokens on mixed L18 and isolates cached U in editing', () => {
    const layer18 = makeElasticLayerWeights(18);
    const layer17 = makeElasticLayerWeights(17);
    const yarn = computeUniYarnInvFreq(UNI_MINI_CONFIG.rope);

    // Mixed sequence [U, G, U, G]
    const seqLen = 4;
    const hiddenStates = deterministicFloat32(seqLen * 8, 512, 0.7);
    const routing = new Uint8Array([0, 1, 0, 1]);
    const positionIds = new Int32Array([14, 15, 16, 17]); // crosses pos=16 Llama-4 threshold!

    // Joint L17 allows G to see U, whereas split L18 isolates U and G
    const traceL17 = runUniElasticAttention({
      hiddenStates,
      seqLen,
      routing,
      layerWeights: layer17,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds,
      maskMode: 'causal',
      useCache: true,
    });
    expect(traceL17.splitTower).toBe(false);
    expect(traceL17.effectiveMask[1 * 4 + 0]).toBe(1); // G(1) attends to U(0) in L17

    const unprunedL18 = runUniElasticAttention({
      hiddenStates,
      seqLen,
      routing,
      layerWeights: layer18,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds,
      maskMode: 'causal',
      useCache: true,
      pruneUnderstandingTokens: false,
    });

    const prunedL18 = runUniElasticAttention({
      hiddenStates,
      seqLen,
      routing,
      layerWeights: layer18,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds,
      maskMode: 'causal',
      useCache: true,
      pruneUnderstandingTokens: true,
    });

    expect(unprunedL18.understandingPruned).toBe(false);
    expect(prunedL18.understandingPruned).toBe(true);
    expect(prunedL18.effectiveMask[1 * 4 + 0]).toBe(0); // G(1) blocked from U(0) in L18

    // U rows [0, 2] are zeroed when pruned
    assertBitExactFiniteArray(
      extractRows(prunedL18.output, 8, [0, 2]),
      new Float32Array(16),
      'd5.pruneU.U_zeroed',
    );
    // G rows [1, 3] match unpruned L18 to 0 ULP across every intermediate & output
    assertBitExactFiniteArray(
      extractRows(prunedL18.qScaled, 64, [1, 3]),
      extractRows(unprunedL18.qScaled, 64, [1, 3]),
      'd5.pruneU.G_qScaled',
    );
    assertBitExactFiniteArray(
      extractRows(prunedL18.attnConcat, 64, [1, 3]),
      extractRows(unprunedL18.attnConcat, 64, [1, 3]),
      'd5.pruneU.G_attnConcat',
    );
    assertBitExactFiniteArray(
      extractRows(prunedL18.output, 8, [1, 3]),
      extractRows(unprunedL18.output, 8, [1, 3]),
      'd5.pruneU.G_output',
    );

    // Editing Stage 2 check: cached [U, G, U, G] -> G-only decode at L18
    // Target G attends ONLY to cached G (cols 1, 3) + current G, NEVER cached U (cols 0, 2)
    const pastK = deterministicFloat32(4 * 16, 611, 0.4);
    const pastV = deterministicFloat32(4 * 16, 612, 0.4);
    const poisonedCachedUK = new Float32Array(pastK);
    const poisonedCachedUV = new Float32Array(pastV);
    poisonedCachedUK.fill(Number.NaN, 0, 16);
    poisonedCachedUK.fill(Number.NaN, 32, 48);
    poisonedCachedUV.fill(Number.NaN, 0, 16);
    poisonedCachedUV.fill(Number.NaN, 32, 48);

    const editClean = runUniElasticAttention({
      hiddenStates: deterministicFloat32(2 * 8, 701, 0.5),
      seqLen: 2,
      routing: new Uint8Array([1, 1]),
      layerWeights: layer18,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds: new Int32Array([4, 5]),
      maskMode: 'bidirectional',
      useCache: false,
      pastKv: { k: pastK, v: pastV, seqLen: 4 },
      pastRouting: new Uint8Array([0, 1, 0, 1]),
    });
    const editPoisonedU = runUniElasticAttention({
      hiddenStates: deterministicFloat32(2 * 8, 701, 0.5),
      seqLen: 2,
      routing: new Uint8Array([1, 1]),
      layerWeights: layer18,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds: new Int32Array([4, 5]),
      maskMode: 'bidirectional',
      useCache: false,
      pastKv: { k: poisonedCachedUK, v: poisonedCachedUV, seqLen: 4 },
      pastRouting: new Uint8Array([0, 1, 0, 1]),
    });

    expect(editClean.cachedHasGeneration).toBe(true);
    expect(editClean.dropCachedUnderstanding).toBe(false);
    expect(Array.from(editClean.effectiveMask)).toEqual([
      0, 1, 0, 1, 1, 1,
      0, 1, 0, 1, 1, 1,
    ]);
    assertBitExactFiniteArray(
      editPoisonedU.output,
      editClean.output,
      'edit.L18.cachedU_isolated',
    );

    const attnFormula = formatUniAttentionCellFormula({
      stage: 'score',
      qHead: 3,
      kvHead: 0,
      qRow: 1,
      kCol: 0,
      headDim: 8,
      branch: 'G',
      splitTower: true,
      isMasked: true,
      value: -Infinity,
    });
    expect(attnFormula).toContain('G:split h3→kv0');
    expect(attnFormula).toContain('−∞');
  });

  it('supports sbd_image maskMode inside runUniElasticAttention', () => {
    const layer0 = makeElasticLayerWeights(0);
    const yarn = computeUniYarnInvFreq(UNI_MINI_CONFIG.rope);
    const trace = runUniElasticAttention({
      hiddenStates: deterministicFloat32(5 * 8, 888, 0.4),
      seqLen: 5,
      routing: new Uint8Array([0, 0, 0, 0, 0]),
      layerWeights: layer0,
      config: UNI_MINI_CONFIG,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds: new Int32Array([0, 1, 2, 3, 4]),
      maskMode: 'sbd_image',
      imageBlockStart: 2,
      imageBlockEnd: 4,
      useCache: false,
    });
    expect(Array.from(trace.effectiveMask)).toEqual([
      1, 0, 0, 0, 0,
      1, 1, 0, 0, 0,
      1, 1, 1, 1, 0,
      1, 1, 1, 1, 0,
      1, 1, 1, 1, 1,
    ]);
  });

  it('computes 3.5x SwiGLU (d=8 -> 28 -> 8), SiLU overflow guard, and routed skipU 0-ULP G parity', () => {
    expect(silu(0)).toBe(0);
    expect(silu(-1000)).toBe(0.0);
    expect(silu(2.0)).toBeCloseTo(2.0 / (1 + Math.exp(-2)), 6);

    const seqLen = 4;
    const x = deterministicFloat32(seqLen * 8, 999, 0.9);
    const wU = makeBranchSwigluWeights(100);
    const wG = makeBranchSwigluWeights(200);
    const routing = new Uint8Array([0, 1, 0, 1]);

    const refU = runUniBranchSwiglu(x, seqLen, wU, UNI_MINI_CONFIG);
    const refG = runUniBranchSwiglu(x, seqLen, wG, UNI_MINI_CONFIG);
    expect(refU.d).toBe(8);
    expect(refU.dFf).toBe(28);
    expect(refU.gatePre.length).toBe(seqLen * 28);
    expect(refU.output.length).toBe(seqLen * 8);

    const routed = runUniRoutedSwiglu(
      x,
      seqLen,
      routing,
      wU,
      wG,
      UNI_MINI_CONFIG,
      false,
    );
    assertBitExactFiniteArray(
      extractRows(routed.output, 8, [0, 2]),
      extractRows(refU.output, 8, [0, 2]),
      'swiglu.routed.U',
    );
    assertBitExactFiniteArray(
      extractRows(routed.output, 8, [1, 3]),
      extractRows(refG.output, 8, [1, 3]),
      'swiglu.routed.G',
    );

    // Check against poisoned wU when skipU = true
    const poisonedU: UniBranchSwigluWeights = {
      gateProj: new Float32Array(28 * 8).fill(Number.NaN),
      upProj: new Float32Array(28 * 8).fill(Number.NaN),
      downProj: new Float32Array(8 * 28).fill(Number.NaN),
    };
    const pruned = runUniRoutedSwiglu(
      x,
      seqLen,
      routing,
      poisonedU,
      wG,
      UNI_MINI_CONFIG,
      true,
    );
    assertBitExactFiniteArray(
      extractRows(pruned.output, 8, [0, 2]),
      new Float32Array(16),
      'swiglu.skipU.U_zeroed',
    );
    assertBitExactFiniteArray(
      extractRows(pruned.output, 8, [1, 3]),
      extractRows(routed.output, 8, [1, 3]),
      'swiglu.skipU.G_exact',
    );

    const proj = linearProject(x, 1, 8, wU.gateProj, 28);
    const rProj = routedLinearProject(
      x,
      seqLen,
      8,
      routing,
      wU.gateProj,
      wG.gateProj,
      28,
    );
    assertFiniteArrayParity(
      rProj.subarray(0, 28),
      proj,
      0,
      'linearProject.vs.routed',
    );

    const swigluFormula = formatUniSwigluCellFormula({
      stage: 'output',
      tokenIndex: 1,
      channelIndex: 3,
      value: routed.output[1 * 8 + 3],
      branch: 'G',
    });
    expect(swigluFormula).toContain('mlp_out[G:t=1, d=3]');
    expect(swigluFormula).toContain('0..27');
  });
});

describe('UNI1 Slice 1B: Anti-Contamination & Clean-Room Static Guards', () => {
  it('contains zero Math.random() and zero forbidden imports across Slice 1B modules', () => {
    const files = [
      'src/uni/rope.ts',
      'src/uni/masks.ts',
      'src/uni/attention.ts',
      'src/uni/swiglu.ts',
    ];
    for (const rel of files) {
      const src = readFileSync(resolve(process.cwd(), rel), 'utf8');
      expect(src).not.toContain('Math.random');
      expect(src).not.toContain('twotower');
      expect(src).not.toContain('src/diffusion');
      expect(src).not.toContain('../diffusion');
      expect(src).not.toContain('ui/controls');
    }
  });
});
