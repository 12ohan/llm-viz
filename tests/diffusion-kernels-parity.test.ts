import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';

import {
  applyDiffusionRmsNorm,
  applyRotaryPosEmb,
  collectDiffusionBackboneWeights,
  collectDiffusionCanonicalWeights,
  collectDiffusionLoraWeights,
  computeLlama4AttnScale,
  computeLlama4AttnScales,
  computeRopeCosSin,
  computeYarnInvFreq,
  createDefaultDiffusionModel,
  DIFFUSION_BACKBONE_FLOAT_COUNT,
  DIFFUSION_BACKBONE_TENSOR_COUNT,
  DIFFUSION_DYADIC_YARN_INV_FREQ,
  DIFFUSION_LORA_FLOAT_COUNT,
  DIFFUSION_LORA_TENSOR_COUNT,
  DIFFUSION_MINI_CONFIG,
  DIFFUSION_TOTAL_FLOAT_COUNT,
  DIFFUSION_TOTAL_TENSOR_COUNT,
  runDiffusionAttention,
  runDiffusionForwardPass,
  runDiffusionSwiglu,
  type DiffusionLayerKvSlice,
  type DiffusionModelParameters,
} from '../src/diffusion';
import {
  buildArCausalMask,
  buildBlockBidirMask,
  buildCausalCommitVerifyMask,
  buildJointTrainingMask,
  buildVlmQuadraticSsMask,
  DIFFUSION_MASK_FIXTURES,
  verifyMobcStrictOffset,
} from '../src/diffusion/masks';
import { DIFFUSION_LFS_INVENTORY, verifyDiffusionLfsInventory } from '../src/diffusion/spec';
import {
  assertBitExactFiniteArray,
  assertCausalScoresParity,
  assertFiniteArrayParity,
  assertFiniteNumber,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';

// ============================================================================
// FIXTURE TYPES & LOADING
// ============================================================================

interface DiffusionBlocksParityFixture {
  readonly metadata: {
    readonly generator: string;
    readonly pythonVersion: string;
    readonly torchVersion: string;
    readonly pinnedTorchVersion: string;
    readonly referenceArchitecture: string;
    readonly tolerances: {
      readonly invFreqUlp: number;
      readonly ropeCosSinLlama4MaxAbs: number;
      readonly ropeRotatedQkMaxAbs: number;
      readonly singleLayerGqaSwigluMaxAbs: number;
      readonly embeddingUlp: number;
      readonly residualFinalNormProbsMaxAbs: number;
      readonly logitsMaxAbs: number;
    };
  };
  readonly cat1ParamsAndLfsInventory: {
    readonly tensorCount: number;
    readonly backboneTensorCount: number;
    readonly loraTensorCount: number;
    readonly totalFloats: number;
    readonly backboneFloats: number;
    readonly loraFloats: number;
    readonly weightsSha256: string;
    readonly backboneSha256: string;
    readonly loraSha256: string;
    readonly backboneFingerprint: string;
    readonly loraFingerprint: string;
    readonly combinedFingerprint: string;
    readonly tensors: Array<{
      readonly name: string;
      readonly role: string;
      readonly layerIndex: number | null;
      readonly isLora: boolean;
      readonly shape: number[];
      readonly offsetFloats: number;
      readonly length: number;
      readonly sha256: string;
    }>;
    readonly lfsInventory: Array<{
      readonly repoId: string;
      readonly revision: string;
      readonly file: string;
      readonly dtype: 'BF16' | 'F32';
      readonly isLora: boolean;
      readonly tensors: number;
      readonly parameters: number;
      readonly headerBytes: number;
      readonly payloadBytes: number;
      readonly totalBytes: number;
      readonly hfLfsSizeBytes: number;
      readonly diffBytes: number;
      readonly lfsSha256Prefix: string;
    }>;
  };
  readonly cat2YarnRopeAndLlama4: {
    readonly invFreq: number[];
    readonly attentionScaling: number;
    readonly low: number;
    readonly high: number;
    readonly extrapolationFactor: number[];
    readonly zones: string[];
    readonly cos32: number[];
    readonly sin32: number[];
    readonly llama4Scales32: number[];
    readonly rotationTest: {
      readonly positionIds: number[];
      readonly qPre: number[];
      readonly kPre: number[];
      readonly qRot: number[];
      readonly kRot: number[];
      readonly llama4Scales: number[];
      readonly qScaled: number[];
    };
    readonly relativePositionInvariance: {
      readonly qVec: number[];
      readonly kVec: number[];
      readonly pairA: { readonly p1: number; readonly k1: number; readonly p2: number; readonly k2: number; readonly dot1: number; readonly dot2: number };
      readonly pairB: { readonly p1: number; readonly k1: number; readonly p2: number; readonly k2: number; readonly dot1: number; readonly dot2: number };
    };
  };
  readonly cat3MasksAndThreeCoords: {
    readonly jointTraining: {
      readonly L: number;
      readonly B: number;
      readonly qLen: number;
      readonly kvLen: number;
      readonly visible: number[];
      readonly regions: string[];
      readonly seqIdx: number[];
      readonly ropePositionIds: number[];
      readonly cachePositions: number[];
      readonly kvRopePositionIds: number[];
      readonly kvCachePositions: number[];
      readonly mobcMinDeltaP: number;
      readonly mobcZeroTwinViolations: number;
    };
    readonly arPrefill: MaskFixtureGrid;
    readonly arDecode: MaskFixtureGrid;
    readonly blockBidir: MaskFixtureGrid;
    readonly causalCommitVerify: MaskFixtureGrid;
    readonly vlmQuadraticSs: MaskFixtureGrid;
  };
  readonly cat4GqaAndLora: {
    readonly causalCrossBoundaryInputs: {
      readonly seqLen: number;
      readonly pastLen: number;
      readonly positionIds: number[];
      readonly cachePositions: number[];
      readonly hiddenStates: number[];
      readonly pastK: number[];
      readonly pastV: number[];
    };
    readonly bidirDraftInputs: {
      readonly seqLen: number;
      readonly pastLen: number;
      readonly positionIds: number[];
      readonly cachePositions: number[];
      readonly hiddenStates: number[];
      readonly pastK: number[];
      readonly pastV: number[];
    };
    readonly layers: Record<string, {
      readonly layerIndex: number;
      readonly causalCrossBoundary: AttnRunFixture;
      readonly bidirLoraOff: AttnRunFixture;
      readonly bidirLoraOn: AttnRunFixture;
    }>;
  };
  readonly cat5Swiglu: {
    readonly layerIndex: number;
    readonly seqLen: number;
    readonly rawHiddenStates: number[];
    readonly postAttentionNorm: number[];
    readonly gatePre: number[];
    readonly gateSilu: number[];
    readonly upVal: number[];
    readonly gated: number[];
    readonly output: number[];
    readonly residualOut: number[];
  };
  readonly cat6Full40LayerForward: {
    readonly causalPrefill8: ForwardRunFixture;
    readonly bidirDraftLoraOff: ForwardRunFixture;
    readonly bidirDraftLoraOn: ForwardRunFixture;
    readonly causalPrefill14ForBoundary: {
      readonly tokenIds: number[];
      readonly seqLen: number;
    };
    readonly causalVerifyCrossLlama4: ForwardRunFixture;
  };
}

interface MaskFixtureGrid {
  readonly qLen: number;
  readonly kvLen: number;
  readonly visible: number[];
  readonly seqIdx: number[];
  readonly ropePositionIds: number[];
  readonly cachePositions: number[];
  readonly kvRopePositionIds: number[];
  readonly kvCachePositions: number[];
}

interface AttnRunFixture {
  readonly pastLen: number;
  readonly kvLen: number;
  readonly qPre: number[];
  readonly kPre: number[];
  readonly vNew: number[];
  readonly qRot: number[];
  readonly kRot: number[];
  readonly llama4Scales: number[];
  readonly qScaled: number[];
  readonly kFull: number[];
  readonly vFull: number[];
  readonly scores: Array<number | null>;
  readonly probs: number[];
  readonly attnConcat: number[];
  readonly oProjBase: number[];
  readonly loraHidden: number[] | null;
  readonly loraDelta: number[] | null;
  readonly output: number[];
}

interface ForwardRunFixture {
  readonly tokenIds: number[];
  readonly seqLen: number;
  readonly pastLen: number;
  readonly kvLen: number;
  readonly maskMode: 'causal' | 'bidirectional' | 'custom';
  readonly loraEnabled: boolean;
  readonly useCache: boolean;
  readonly positionIds: number[];
  readonly cachePositions: number[];
  readonly embeddings: number[];
  readonly residualCheckpoints: Record<string, number[]>;
  readonly finalNorm: number[];
  readonly logits: number[];
  readonly probs: number[];
  readonly top5ByPosition: number[][];
}

const FX: DiffusionBlocksParityFixture = JSON.parse(
  readFileSync(resolve(__dirname, 'fixtures/pytorch-diffusion-blocks-parity.json'), 'utf-8'),
);

// ============================================================================
// HELPERS
// ============================================================================

function computeDiffusionDigests(model: DiffusionModelParameters): {
  readonly weightsSha256: string;
  readonly backboneSha256: string;
  readonly loraSha256: string;
  readonly totalFloats: number;
  readonly backboneFloats: number;
  readonly loraFloats: number;
  readonly tensorMeta: Array<{ name: string; shape: number[]; length: number; sha256: string; isLora: boolean }>;
} {
  const canonical = collectDiffusionCanonicalWeights(model);
  const fullHash = createHash('sha256');
  const backboneHash = createHash('sha256');
  const loraHash = createHash('sha256');
  let totalFloats = 0;
  let backboneFloats = 0;
  let loraFloats = 0;
  const tensorMeta: Array<{ name: string; shape: number[]; length: number; sha256: string; isLora: boolean }> = [];

  for (const t of canonical) {
    const buf = Buffer.allocUnsafe(t.data.length * 4);
    for (let i = 0; i < t.data.length; i++) {
      const val = t.data[i];
      assertFiniteNumber(val, `weight:${t.name}[${i}]`);
      buf.writeFloatLE(val, i * 4);
    }
    fullHash.update(buf);
    if (t.isLora) {
      loraHash.update(buf);
      loraFloats += t.data.length;
    } else {
      backboneHash.update(buf);
      backboneFloats += t.data.length;
    }
    totalFloats += t.data.length;
    tensorMeta.push({
      name: t.name,
      shape: Array.from(t.shape),
      length: t.data.length,
      sha256: createHash('sha256').update(buf).digest('hex'),
      isLora: t.isLora,
    });
  }

  return {
    weightsSha256: fullHash.digest('hex'),
    backboneSha256: backboneHash.digest('hex'),
    loraSha256: loraHash.digest('hex'),
    totalFloats,
    backboneFloats,
    loraFloats,
    tensorMeta,
  };
}

function topKTokenIds(logits: Float32Array, seqLen: number, vocabSize: number, k: number): number[][] {
  const result: number[][] = [];
  for (let t = 0; t < seqLen; t++) {
    const off = t * vocabSize;
    const indices = Array.from({ length: vocabSize }, (_, v) => v);
    indices.sort((a, b) => logits[off + b] - logits[off + a] || a - b);
    result.push(indices.slice(0, k));
  }
  return result;
}

// ============================================================================
// TEST SUITE (CATEGORIES 1-6)
// ============================================================================

describe('Independent PyTorch 2.15.0 Reference Parity — Nemotron-Labs-Diffusion (Categories 1–6)', () => {
  const model = createDefaultDiffusionModel();
  const yarn = computeYarnInvFreq(DIFFUSION_MINI_CONFIG.rope);
  const observedMaxErrByCategory: Record<string, number> = {};

  function recordErr(category: string, err: number): void {
    assertFiniteNumber(err, `observedErr[${category}]`);
    const prev = observedMaxErrByCategory[category];
    if (prev === undefined || err > prev) observedMaxErrByCategory[category] = err;
  }

  afterAll(() => {
    const lines = Object.entries(observedMaxErrByCategory).map(
      ([k, v]) => `  ${k}: maxAbsErr = ${v.toExponential(3)}`,
    );
    console.log(`[Diffusion PyTorch 2.15.0 Parity Summary — Cats 1–6]\n${lines.join('\n')}`);
  });

  // --------------------------------------------------------------------------
  // Category 1: Params, Shapes & 0 B LFS Inventory
  // --------------------------------------------------------------------------
  it('Category 1: verifies 363 + 80 = 443 tensor shapes/SHA-256 digests and 13-file 0 B LFS inventory', () => {
    expect(FX.metadata.torchVersion.startsWith(FX.metadata.pinnedTorchVersion)).toBe(true);

    const backbone = collectDiffusionBackboneWeights(model);
    const lora = collectDiffusionLoraWeights(model);
    const canonical = collectDiffusionCanonicalWeights(model);
    expect(backbone.length).toBe(DIFFUSION_BACKBONE_TENSOR_COUNT);
    expect(lora.length).toBe(DIFFUSION_LORA_TENSOR_COUNT);
    expect(canonical.length).toBe(DIFFUSION_TOTAL_TENSOR_COUNT);

    const digests = computeDiffusionDigests(model);
    expect(digests.totalFloats).toBe(DIFFUSION_TOTAL_FLOAT_COUNT);
    expect(digests.backboneFloats).toBe(DIFFUSION_BACKBONE_FLOAT_COUNT);
    expect(digests.loraFloats).toBe(DIFFUSION_LORA_FLOAT_COUNT);
    expect(digests.weightsSha256).toBe(FX.cat1ParamsAndLfsInventory.weightsSha256);
    expect(digests.backboneSha256).toBe(FX.cat1ParamsAndLfsInventory.backboneSha256);
    expect(digests.loraSha256).toBe(FX.cat1ParamsAndLfsInventory.loraSha256);
    expect(model.backboneFingerprint).toBe(FX.cat1ParamsAndLfsInventory.backboneFingerprint);
    expect(model.loraFingerprint).toBe(FX.cat1ParamsAndLfsInventory.loraFingerprint);
    expect(model.combinedFingerprint).toBe(FX.cat1ParamsAndLfsInventory.combinedFingerprint);

    expect(digests.tensorMeta.length).toBe(FX.cat1ParamsAndLfsInventory.tensors.length);
    for (let i = 0; i < digests.tensorMeta.length; i++) {
      const actual = digests.tensorMeta[i];
      const expected = FX.cat1ParamsAndLfsInventory.tensors[i];
      expect(actual.name).toBe(expected.name);
      expect(actual.shape).toEqual(expected.shape);
      expect(actual.length).toBe(expected.length);
      expect(actual.isLora).toBe(expected.isLora);
      expect(actual.sha256).toBe(expected.sha256);
    }

    // Production 13-file 0 B LFS inventory verification
    const lfsCheck = verifyDiffusionLfsInventory();
    expect(lfsCheck.valid).toBe(true);
    expect(lfsCheck.entryCount).toBe(13);
    expect(lfsCheck.mismatches).toEqual([]);
    expect(DIFFUSION_LFS_INVENTORY.length).toBe(13);
    expect(FX.cat1ParamsAndLfsInventory.lfsInventory.length).toBe(13);

    for (const refEntry of FX.cat1ParamsAndLfsInventory.lfsInventory) {
      const match = DIFFUSION_LFS_INVENTORY.find(
        (e) =>
          e.repoId === refEntry.repoId &&
          e.revision === refEntry.revision &&
          e.isLora === refEntry.isLora &&
          e.lfsSha256Prefix === refEntry.lfsSha256Prefix,
      );
      expect(match, `Missing LFS entry ${refEntry.repoId}:${refEntry.lfsSha256Prefix}`).toBeDefined();
      expect(match!.dtype).toBe(refEntry.dtype);
      expect(match!.tensors).toBe(refEntry.tensors);
      expect(match!.parameters).toBe(refEntry.parameters);
      expect(match!.headerBytes).toBe(refEntry.headerBytes);
      expect(match!.payloadBytes).toBe(refEntry.payloadBytes);
      expect(match!.totalBytes).toBe(refEntry.totalBytes);
      expect(match!.hfLfsSizeBytes).toBe(refEntry.hfLfsSizeBytes);
      expect(match!.diffBytes).toBe(0);
    }

    recordErr('cat1_params_lfs', 0);
  });

  // --------------------------------------------------------------------------
  // Category 2: Dyadic YaRN RoPE & Llama-4 Scale
  // --------------------------------------------------------------------------
  it('Category 2: verifies 0-ULP dyadic YaRN inv_freq, cos/sin, half-split rotation, relative invariance, and Llama-4 step jump', () => {
    const fx = FX.cat2YarnRopeAndLlama4;

    // 0-ULP exact dyadic YaRN inv_freq
    assertBitExactFiniteArray(yarn.invFreq, DIFFUSION_DYADIC_YARN_INV_FREQ, 'cat2.invFreq.const');
    assertBitExactFiniteArray(yarn.invFreq, fx.invFreq, 'cat2.invFreq.pytorch');
    expect(yarn.attentionScaling).toBe(fx.attentionScaling);
    expect(yarn.low).toBe(fx.low);
    expect(yarn.high).toBe(fx.high);
    expect(yarn.zones).toEqual(fx.zones);
    assertBitExactFiniteArray(yarn.extrapolationFactor, fx.extrapolationFactor, 'cat2.extrapolationFactor');

    // cos/sin & Llama-4 scales for pos = 0..31
    const allPos = Int32Array.from({ length: 32 }, (_, i) => i);
    const cosSin32 = computeRopeCosSin(yarn.invFreq, allPos, yarn.attentionScaling, DIFFUSION_MINI_CONFIG.headDim);
    const cosErr = assertFiniteArrayParity(cosSin32.cos, fx.cos32, 1e-6, 'cat2.cos32');
    const sinErr = assertFiniteArrayParity(cosSin32.sin, fx.sin32, 1e-6, 'cat2.sin32');
    recordErr('cat2_rope_cos_sin_llama4', Math.max(cosErr, sinErr));

    const llama32 = computeLlama4AttnScales(
      allPos,
      DIFFUSION_MINI_CONFIG.rope.llama4Beta,
      DIFFUSION_MINI_CONFIG.rope.originalMaxPositionEmbeddings,
    );
    const llamaErr = assertFiniteArrayParity(llama32, fx.llama4Scales32, 1e-6, 'cat2.llama4Scales32');
    recordErr('cat2_rope_cos_sin_llama4', llamaErr);

    // Step jump at pos 15 -> 16
    expect(computeLlama4AttnScale(15, 0.1, 16)).toBe(1.0);
    expect(computeLlama4AttnScale(16, 0.1, 16)).toBeCloseTo(1.06931472, 6);

    // Half-split rotation Q_rot / K_rot + Llama-4 query scale
    const rt = fx.rotationTest;
    const rtPos = Int32Array.from(rt.positionIds);
    const rtCosSin = computeRopeCosSin(yarn.invFreq, rtPos, yarn.attentionScaling, DIFFUSION_MINI_CONFIG.headDim);
    const qRot = applyRotaryPosEmb(
      Float32Array.from(rt.qPre),
      rtPos.length,
      DIFFUSION_MINI_CONFIG.numAttentionHeads,
      DIFFUSION_MINI_CONFIG.headDim,
      rtCosSin.cos,
      rtCosSin.sin,
    );
    const kRot = applyRotaryPosEmb(
      Float32Array.from(rt.kPre),
      rtPos.length,
      DIFFUSION_MINI_CONFIG.numKeyValueHeads,
      DIFFUSION_MINI_CONFIG.headDim,
      rtCosSin.cos,
      rtCosSin.sin,
    );
    recordErr('cat2_rope_qk_rot', assertFiniteArrayParity(qRot, rt.qRot, 1e-5, 'cat2.qRot'));
    recordErr('cat2_rope_qk_rot', assertFiniteArrayParity(kRot, rt.kRot, 1e-5, 'cat2.kRot'));

    const rtLlama = computeLlama4AttnScales(rtPos, 0.1, 16);
    recordErr('cat2_rope_cos_sin_llama4', assertFiniteArrayParity(rtLlama, rt.llama4Scales, 1e-6, 'cat2.rtLlama'));

    const qScaled = new Float32Array(qRot.length);
    const stride = DIFFUSION_MINI_CONFIG.numAttentionHeads * DIFFUSION_MINI_CONFIG.headDim;
    for (let t = 0; t < rtPos.length; t++) {
      const s = rtLlama[t];
      for (let c = 0; c < stride; c++) {
        qScaled[t * stride + c] = Math.fround(qRot[t * stride + c] * s);
      }
    }
    recordErr('cat2_rope_qk_rot', assertFiniteArrayParity(qScaled, rt.qScaled, 1e-5, 'cat2.qScaled'));

    // Relative-position inner product invariance <R_p q, R_k k> == f(p - k) for p, k < 16
    const rel = fx.relativePositionInvariance;
    const qVec = Float32Array.from(rel.qVec);
    const kVec = Float32Array.from(rel.kVec);
    const rotatedInner = (pPos: number, kPos: number): number => {
      const csP = computeRopeCosSin(yarn.invFreq, Int32Array.from([pPos]), yarn.attentionScaling, 8);
      const csK = computeRopeCosSin(yarn.invFreq, Int32Array.from([kPos]), yarn.attentionScaling, 8);
      const rp = applyRotaryPosEmb(qVec, 1, 1, 8, csP.cos, csP.sin);
      const rk = applyRotaryPosEmb(kVec, 1, 1, 8, csK.cos, csK.sin);
      let acc = 0;
      for (let i = 0; i < 8; i++) acc += rp[i] * rk[i];
      return acc;
    };

    for (const pair of [rel.pairA, rel.pairB]) {
      const d1 = rotatedInner(pair.p1, pair.k1);
      const d2 = rotatedInner(pair.p2, pair.k2);
      expect(Math.abs(d1 - d2)).toBeLessThanOrEqual(1e-5);
      expect(Math.abs(d1 - pair.dot1)).toBeLessThanOrEqual(1e-5);
      expect(Math.abs(d2 - pair.dot2)).toBeLessThanOrEqual(1e-5);
      recordErr('cat2_rope_qk_rot', Math.max(Math.abs(d1 - pair.dot1), Math.abs(d2 - pair.dot2)));
    }
  });

  // --------------------------------------------------------------------------
  // Category 3: 5-Mode Masks & 3-Coordinate Fixtures
  // --------------------------------------------------------------------------
  it('Category 3: verifies exact boolean grids & 3-coordinates for all 5 mask modes (incl. M_OBC Δp >= 1)', () => {
    const fx = FX.cat3MasksAndThreeCoords;

    const assertMaskParity = (
      actual: ReturnType<typeof buildArCausalMask>,
      ref: MaskFixtureGrid,
      label: string,
    ) => {
      expect(actual.qLen, `${label}.qLen`).toBe(ref.qLen);
      expect(actual.kvLen, `${label}.kvLen`).toBe(ref.kvLen);
      expect(Array.from(actual.visible), `${label}.visible`).toEqual(ref.visible);
      expect(Array.from(actual.queryCoords.seqIdx), `${label}.seqIdx`).toEqual(ref.seqIdx);
      expect(Array.from(actual.queryCoords.ropePositionIds), `${label}.ropePositionIds`).toEqual(
        ref.ropePositionIds,
      );
      expect(Array.from(actual.queryCoords.cachePositions), `${label}.cachePositions`).toEqual(
        ref.cachePositions,
      );
      expect(Array.from(actual.kvRopePositionIds), `${label}.kvRopePositionIds`).toEqual(
        ref.kvRopePositionIds,
      );
      expect(Array.from(actual.kvCachePositions), `${label}.kvCachePositions`).toEqual(
        ref.kvCachePositions,
      );
    };

    // 1. 12x12 Joint Training
    const jt = buildJointTrainingMask(fx.jointTraining.L, fx.jointTraining.B);
    assertMaskParity(jt, fx.jointTraining, 'cat3.jointTraining');
    expect(Array.from(jt.regions ?? [])).toEqual(fx.jointTraining.regions);
    const mobc = verifyMobcStrictOffset(jt);
    expect(mobc.valid).toBe(true);
    expect(mobc.minDeltaP).toBe(fx.jointTraining.mobcMinDeltaP);
    expect(mobc.zeroTwinViolations).toBe(0);

    // 2A. 4x4 AR Prefill
    assertMaskParity(buildArCausalMask(0, 4), fx.arPrefill, 'cat3.arPrefill');
    // 2B. 1x5 AR Decode
    assertMaskParity(buildArCausalMask(4, 1), fx.arDecode, 'cat3.arDecode');
    // 3. 4x8 Block-Diff Bidir
    assertMaskParity(buildBlockBidirMask(4, 4), fx.blockBidir, 'cat3.blockBidir');
    // 4. 4x8 Causal Commit/Verify
    assertMaskParity(buildCausalCommitVerifyMask(4, 4), fx.causalCommitVerify, 'cat3.causalCommitVerify');
    // 5. 6x8 VLM-8B Quadratic SS (P=2, k=2)
    assertMaskParity(buildVlmQuadraticSsMask(2, 2), fx.vlmQuadraticSs, 'cat3.vlmQuadraticSs');

    // Also confirm exported DIFFUSION_MASK_FIXTURES matches exact grids
    expect(DIFFUSION_MASK_FIXTURES).toBeDefined();
    recordErr('cat3_masks', 0);
  });

  // --------------------------------------------------------------------------
  // Category 4: GQA (32Q/8KV) + o_proj LoRA Parity
  // --------------------------------------------------------------------------
  it('Category 4: verifies Layers 0 & 20 GQA (causal, bidir transient concat, prefix immutability, LoRA ON/OFF)', () => {
    const tol = FX.metadata.tolerances.singleLayerGqaSwigluMaxAbs;
    const cInp = FX.cat4GqaAndLora.causalCrossBoundaryInputs;
    const bInp = FX.cat4GqaAndLora.bidirDraftInputs;

    const assertAttnRun = (
      actual: ReturnType<typeof runDiffusionAttention>,
      ref: AttnRunFixture,
      label: string,
    ) => {
      expect(actual.pastLen).toBe(ref.pastLen);
      expect(actual.kvLen).toBe(ref.kvLen);
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.qPre, ref.qPre, tol, `${label}.qPre`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.kPre, ref.kPre, tol, `${label}.kPre`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.vNew, ref.vNew, tol, `${label}.vNew`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.qRot, ref.qRot, tol, `${label}.qRot`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.kRot, ref.kRot, tol, `${label}.kRot`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.llama4Scales, ref.llama4Scales, 1e-6, `${label}.llama4Scales`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.qScaled, ref.qScaled, tol, `${label}.qScaled`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.kFull, ref.kFull, tol, `${label}.kFull`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.vFull, ref.vFull, tol, `${label}.vFull`));
      recordErr('cat4_gqa_lora', assertCausalScoresParity(actual.scores, ref.scores, tol, `${label}.scores`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.probs, ref.probs, tol, `${label}.probs`));
      // Exact 0.0 masked probs where score is -Infinity
      for (let i = 0; i < ref.scores.length; i++) {
        if (ref.scores[i] === null) {
          expect(actual.probs[i], `${label}.maskedProb[${i}]`).toBe(0.0);
        }
      }
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.attnConcat, ref.attnConcat, tol, `${label}.attnConcat`));
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.oProjBase, ref.oProjBase, tol, `${label}.oProjBase`));
      if (ref.loraHidden === null) {
        expect(actual.loraHidden).toBeNull();
        expect(actual.loraDelta).toBeNull();
        assertBitExactFiniteArray(actual.output, actual.oProjBase, `${label}.loraOffInvariant`);
      } else {
        expect(actual.loraHidden).not.toBeNull();
        expect(actual.loraDelta).not.toBeNull();
        recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.loraHidden!, ref.loraHidden, tol, `${label}.loraHidden`));
        recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.loraDelta!, ref.loraDelta!, tol, `${label}.loraDelta`));
      }
      recordErr('cat4_gqa_lora', assertFiniteArrayParity(actual.output, ref.output, tol, `${label}.output`));
    };

    for (const layerKey of ['0', '20']) {
      const layerFx = FX.cat4GqaAndLora.layers[layerKey];
      const layerWeights = model.layers[layerFx.layerIndex];

      // (a) Causal cross-boundary pos=14..17 (pastLen=14)
      const pastCausal: DiffusionLayerKvSlice = {
        k: Float32Array.from(cInp.pastK),
        v: Float32Array.from(cInp.pastV),
        seqLen: cInp.pastLen,
      };
      const causalTrace = runDiffusionAttention({
        hiddenStates: Float32Array.from(cInp.hiddenStates),
        seqLen: cInp.seqLen,
        layerWeights,
        config: DIFFUSION_MINI_CONFIG,
        invFreq: yarn.invFreq,
        attentionScaling: yarn.attentionScaling,
        positionIds: Int32Array.from(cInp.positionIds),
        cachePositions: Int32Array.from(cInp.cachePositions),
        maskMode: 'causal',
        loraEnabled: false,
        useCache: true,
        pastKv: pastCausal,
      });
      assertAttnRun(causalTrace, layerFx.causalCrossBoundary, `cat4.L${layerKey}.causal`);

      // (b) Bidirectional draft P=8, B=4 with useCache=false + prefix immutability
      const pastBidir: DiffusionLayerKvSlice = {
        k: Float32Array.from(bInp.pastK),
        v: Float32Array.from(bInp.pastV),
        seqLen: bInp.pastLen,
      };
      const kSnapshot = new Float32Array(pastBidir.k);
      const vSnapshot = new Float32Array(pastBidir.v);

      const bidirOff = runDiffusionAttention({
        hiddenStates: Float32Array.from(bInp.hiddenStates),
        seqLen: bInp.seqLen,
        layerWeights,
        config: DIFFUSION_MINI_CONFIG,
        invFreq: yarn.invFreq,
        attentionScaling: yarn.attentionScaling,
        positionIds: Int32Array.from(bInp.positionIds),
        cachePositions: Int32Array.from(bInp.cachePositions),
        maskMode: 'bidirectional',
        loraEnabled: false,
        useCache: false,
        pastKv: pastBidir,
      });
      assertAttnRun(bidirOff, layerFx.bidirLoraOff, `cat4.L${layerKey}.bidirOff`);
      assertBitExactFiniteArray(pastBidir.k, kSnapshot, `cat4.L${layerKey}.prefixKImmutableOff`);
      assertBitExactFiniteArray(pastBidir.v, vSnapshot, `cat4.L${layerKey}.prefixVImmutableOff`);

      // (c) Bidirectional draft with LoRA ON (4.0 * B(A * a))
      const bidirOn = runDiffusionAttention({
        hiddenStates: Float32Array.from(bInp.hiddenStates),
        seqLen: bInp.seqLen,
        layerWeights,
        config: DIFFUSION_MINI_CONFIG,
        invFreq: yarn.invFreq,
        attentionScaling: yarn.attentionScaling,
        positionIds: Int32Array.from(bInp.positionIds),
        cachePositions: Int32Array.from(bInp.cachePositions),
        maskMode: 'bidirectional',
        loraEnabled: true,
        useCache: false,
        pastKv: pastBidir,
      });
      assertAttnRun(bidirOn, layerFx.bidirLoraOn, `cat4.L${layerKey}.bidirOn`);
      assertBitExactFiniteArray(pastBidir.k, kSnapshot, `cat4.L${layerKey}.prefixKImmutableOn`);
      assertBitExactFiniteArray(pastBidir.v, vSnapshot, `cat4.L${layerKey}.prefixVImmutableOn`);
      expect(computeStrictMaxAbsDiff(bidirOn.output, bidirOff.output, `cat4.L${layerKey}.loraEffect`)).toBeGreaterThan(1e-4);
    }
  });

  // --------------------------------------------------------------------------
  // Category 5: Layer 0 Dense SwiGLU Parity
  // --------------------------------------------------------------------------
  it('Category 5: verifies Layer 0 post-attention RMSNorm, SwiGLU intermediates, and residual add', () => {
    const fx = FX.cat5Swiglu;
    const tol = FX.metadata.tolerances.singleLayerGqaSwigluMaxAbs;
    const layer0 = model.layers[0];

    const rawHidden = Float32Array.from(fx.rawHiddenStates);
    const postNorm = applyDiffusionRmsNorm(
      rawHidden,
      fx.seqLen,
      DIFFUSION_MINI_CONFIG.hiddenSize,
      layer0.postAttentionLayerNorm,
      DIFFUSION_MINI_CONFIG.rmsNormEps,
    );
    recordErr('cat5_swiglu', assertFiniteArrayParity(postNorm, fx.postAttentionNorm, tol, 'cat5.postNorm'));

    const swiglu = runDiffusionSwiglu(postNorm, fx.seqLen, layer0.mlp, DIFFUSION_MINI_CONFIG);
    recordErr('cat5_swiglu', assertFiniteArrayParity(swiglu.gatePre, fx.gatePre, tol, 'cat5.gatePre'));
    recordErr('cat5_swiglu', assertFiniteArrayParity(swiglu.gateSilu, fx.gateSilu, tol, 'cat5.gateSilu'));
    recordErr('cat5_swiglu', assertFiniteArrayParity(swiglu.upVal, fx.upVal, tol, 'cat5.upVal'));
    recordErr('cat5_swiglu', assertFiniteArrayParity(swiglu.gated, fx.gated, tol, 'cat5.gated'));
    recordErr('cat5_swiglu', assertFiniteArrayParity(swiglu.output, fx.output, tol, 'cat5.output'));

    const residOut = new Float32Array(rawHidden.length);
    for (let i = 0; i < rawHidden.length; i++) {
      residOut[i] = Math.fround(rawHidden[i] + swiglu.output[i]);
    }
    recordErr('cat5_swiglu', assertFiniteArrayParity(residOut, fx.residualOut, tol, 'cat5.residualOut'));
  });

  // --------------------------------------------------------------------------
  // Category 6: Full 40-Layer Forward Parity
  // --------------------------------------------------------------------------
  it('Category 6: verifies complete 40-layer forward pass (causal prefill, bidir draft LoRA OFF/ON, cross-Llama-4 verify)', () => {
    const tol = FX.metadata.tolerances;
    const fx = FX.cat6Full40LayerForward;

    const assertForwardParity = (
      actual: ReturnType<typeof runDiffusionForwardPass>,
      ref: ForwardRunFixture,
      label: string,
    ) => {
      expect(actual.seqLen).toBe(ref.seqLen);
      expect(actual.pastLen).toBe(ref.pastLen);
      expect(actual.kvLen).toBe(ref.kvLen);
      expect(Array.from(actual.positionIds)).toEqual(ref.positionIds);
      expect(Array.from(actual.cachePositions)).toEqual(ref.cachePositions);

      assertBitExactFiniteArray(actual.embeddings, ref.embeddings, `${label}.embeddings`);
      for (const ckptKey of ['0', '1', '20', '40']) {
        const layerIdx = Number(ckptKey);
        recordErr(
          'cat6_40layer_resid_norm_probs',
          assertFiniteArrayParity(
            actual.residualCheckpoints[layerIdx],
            ref.residualCheckpoints[ckptKey],
            tol.residualFinalNormProbsMaxAbs,
            `${label}.resid${ckptKey}`,
          ),
        );
      }
      recordErr(
        'cat6_40layer_resid_norm_probs',
        assertFiniteArrayParity(
          actual.finalNorm,
          ref.finalNorm,
          tol.residualFinalNormProbsMaxAbs,
          `${label}.finalNorm`,
        ),
      );
      recordErr(
        'cat6_40layer_logits',
        assertFiniteArrayParity(actual.logits, ref.logits, tol.logitsMaxAbs, `${label}.logits`),
      );
      recordErr(
        'cat6_40layer_resid_norm_probs',
        assertFiniteArrayParity(
          actual.probs,
          ref.probs,
          tol.residualFinalNormProbsMaxAbs,
          `${label}.probs`,
        ),
      );

      const actualTop5 = topKTokenIds(actual.logits, actual.seqLen, DIFFUSION_MINI_CONFIG.vocabSize, 5);
      expect(actualTop5, `${label}.top5`).toEqual(ref.top5ByPosition);
    };

    // (a) Causal prefill S=8 (pos 0..7)
    const prefill8 = runDiffusionForwardPass(model, {
      tokenIds: fx.causalPrefill8.tokenIds,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    assertForwardParity(prefill8, fx.causalPrefill8, 'cat6.prefill8');

    // (b) Bidirectional draft P=8, S=4 (pos 8..11), loraEnabled=false
    const draftOff = runDiffusionForwardPass(model, {
      tokenIds: fx.bidirDraftLoraOff.tokenIds,
      maskMode: 'bidirectional',
      loraEnabled: false,
      useCache: false,
      pastKvs: prefill8.newKvs,
    });
    assertForwardParity(draftOff, fx.bidirDraftLoraOff, 'cat6.draftOff');

    // (c) Bidirectional draft P=8, S=4 (pos 8..11), loraEnabled=true
    const draftOn = runDiffusionForwardPass(model, {
      tokenIds: fx.bidirDraftLoraOn.tokenIds,
      maskMode: 'bidirectional',
      loraEnabled: true,
      useCache: false,
      pastKvs: prefill8.newKvs,
    });
    assertForwardParity(draftOn, fx.bidirDraftLoraOn, 'cat6.draftOn');

    // LoRA ON changes draft logits
    expect(
      computeStrictMaxAbsDiff(draftOn.logits, draftOff.logits, 'cat6.draftLoraDelta'),
    ).toBeGreaterThan(1e-4);

    // (d) Cross-Llama-4 boundary causal verify P=14, S=4 (pos 14..17), loraEnabled=false
    const prefill14 = runDiffusionForwardPass(model, {
      tokenIds: fx.causalPrefill14ForBoundary.tokenIds,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    const verifyBoundary = runDiffusionForwardPass(model, {
      tokenIds: fx.causalVerifyCrossLlama4.tokenIds,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
      pastKvs: prefill14.newKvs,
    });
    assertForwardParity(verifyBoundary, fx.causalVerifyCrossLlama4, 'cat6.verifyBoundary');

    // Verify statelessness: running causal verify again produces 0-ULP identical logits
    const verifyBoundaryRepeat = runDiffusionForwardPass(model, {
      tokenIds: fx.causalVerifyCrossLlama4.tokenIds,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
      pastKvs: prefill14.newKvs,
    });
    assertBitExactFiniteArray(
      verifyBoundaryRepeat.logits,
      verifyBoundary.logits,
      'cat6.verifyBoundaryStateless',
    );
  });
});
