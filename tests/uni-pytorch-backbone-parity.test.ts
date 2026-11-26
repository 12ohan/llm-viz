/**
 * Independent PyTorch 2.15.0 Core Backbone Parity Suite for
 * Nemotron-Labs-Diffusion-Uni (`tests/uni-pytorch-backbone-parity.test.ts`)
 *
 * Verifies all 7 fixture categories against `tests/fixtures/pytorch-uni-backbone.json`
 * with strict finite-value assertions (`tests/helpers/parityAssert.ts`):
 *   - Cryptographic SHA-256 verification of all 618 core tensors (139,024 floats = 556,096 B)
 *   - 0-ULP: dyadic YaRN inv_freq, 2D SinCos content columns, split-tower masks,
 *            and D5 elastic_pruned vs hf_faithful Stage 2 outputs (T2I + Editing)
 *   - <= 1e-5: single-layer joint (L0, L17) & split-tower (L18, L33) full traces
 *   - <= 2e-5: 34-layer residuals & finalNorm across U-only, G-only, T2I, and Editing
 *   - <= 5e-5: head logits & softmax probabilities (`diffusion_head`, `gen_predictor`)
 *   - Non-mutation of `pastKvs` & `pastRouting`, plus static isolation checks
 */

import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  assertBitExactFiniteArray,
  assertCausalScoresParity,
  assertFiniteArrayParity,
} from './helpers/parityAssert';
import {
  embedUniGenTokens,
  embedUniTextTokens,
  runUniElasticLayer,
  runUniForwardPass,
} from '../src/uni/forward';
import {
  collectUniBackboneWeights,
  createDefaultUniCoreModel,
  createUniCoreModel,
} from '../src/uni/parameters';
import {
  applyUniRotaryPosEmb,
  compute2dSinCosPosEmbed,
  computeLlama4AttnScales,
  computeUniRopeCosSin,
  computeUniYarnInvFreq,
} from '../src/uni/rope';
import {
  UNI_DYADIC_YARN_INV_FREQ,
  UNI_MINI_CORE_FLOAT_COUNT,
  UNI_MINI_CORE_TENSOR_COUNT,
  UNI_MINI_CORE_WEIGHT_BYTES,
  type UniLayerKvSlice,
} from '../src/uni/types';

// ============================================================================
// FIXTURE TYPES & LOADING
// ============================================================================

interface SingleLayerFixture {
  layerIndex: number;
  splitTower: boolean;
  inputNorm: number[];
  qPre: number[];
  kPre: number[];
  vNew: number[];
  qRot: number[];
  kRot: number[];
  llama4Scales: number[];
  qScaled: number[];
  effectiveMask: number[];
  scores: (number | null)[];
  probs: number[];
  attnConcat: number[];
  attnOutput: number[];
  postAttentionResidual: number[];
  postAttentionNorm: number[];
  swigluGatePre: number[];
  swigluGateSilu: number[];
  swigluUpVal: number[];
  swigluGated: number[];
  swigluOutput: number[];
  output: number[];
}

interface UniBackboneFixture {
  schemaVersion: string;
  modelId: string;
  backboneSeed: number;
  backboneFingerprint: string;
  tensorCount: number;
  totalFloats: number;
  totalBytes: number;
  weightsSha256: string;
  torchVersion: string;
  pythonVersion: string;
  keyCheckpointLayers: number[];
  rope: {
    invFreq: number[];
    attentionScaling: number;
    low: number;
    high: number;
    extrapolationFactor: number[];
    positions: number[];
    cos: number[];
    sin: number[];
    llama4Scales: number[];
    sampleRotate: {
      positionIds: number[];
      tokenIds: number[];
      qIn: number[];
      kIn: number[];
      qRot: number[];
      kRot: number[];
    };
  };
  sinCos2d: {
    marked2x3: {
      height: number;
      width: number;
      contentWidth: number;
      hiddenSize: number;
      markerColumn: boolean;
      columnStride: number;
      rowStride: number;
      frequencies: number[];
      embeddings: number[];
    };
    unmarked2x2: {
      height: number;
      width: number;
      contentWidth: number;
      hiddenSize: number;
      markerColumn: boolean;
      columnStride: number;
      rowStride: number;
      frequencies: number[];
      embeddings: number[];
    };
  };
  singleLayers: {
    seqLen: number;
    routing: number[];
    tokenIds: number[];
    positionIds: number[];
    maskMode: 'causal';
    inputsEmbeds: number[];
    layers: Record<string, SingleLayerFixture>;
  };
  understanding34L: {
    prefill: {
      tokenIds: number[];
      positionIds: number[];
      checkpoints: Record<string, number[]>;
      finalNorm: number[];
      textLogits: number[];
      textProbs: number[];
    };
    decode: {
      tokenIds: number[];
      positionIds: number[];
      checkpoints: Record<string, number[]>;
      finalNorm: number[];
      textLogits: number[];
      textProbs: number[];
    };
  };
  genOnly34L: {
    genTokenIds: number[];
    positionIds: number[];
    checkpoints: Record<string, number[]>;
    finalNorm: number[];
    genLogits: number[];
    genProbs: number[];
  };
  t2iTwoStageD5: {
    promptTokenIds: number[];
    genTokenIds: number[];
    stage1Hf: {
      checkpoints: Record<string, number[]>;
      finalNorm: number[];
      textLogits: number[];
    };
    stage1Pruned: {
      skippedLayers: number[];
      checkpoints: Record<string, number[] | null>;
    };
    stage2: {
      positionIds: number[];
      droppedCachedUnderstandingLayers: number[];
      l17EffectiveMask: number[];
      l18EffectiveMask: number[];
      checkpoints: Record<string, number[]>;
      finalNorm: number[];
      genLogits: number[];
      genProbs: number[];
      pytorchBitExactPrunedVsHf: boolean;
    };
  };
  editingTwoStageD5: {
    stage1UnderstandingTokenIds: number[];
    stage1SourceGenTokenIds: number[];
    stage1Routing: number[];
    stage1InputsEmbeds: number[];
    stage2GenTokenIds: number[];
    stage1Hf: {
      checkpoints: Record<string, number[]>;
      finalNorm: number[];
      textLogits: number[];
      genLogits: number[];
    };
    stage1Pruned: {
      checkpoints: Record<string, number[]>;
      finalNorm: number[];
    };
    stage2: {
      positionIds: number[];
      droppedCachedUnderstandingLayers: number[];
      l17EffectiveMask: number[];
      l18EffectiveMask: number[];
      checkpoints: Record<string, number[]>;
      finalNorm: number[];
      genLogits: number[];
      genProbs: number[];
      pytorchBitExactPrunedVsHf: boolean;
      pytorchCachedUPerturbZeroUlp: boolean;
      pytorchCachedGPerturbMaxAbsDiff: number;
    };
  };
}

const FIXTURE_PATH = resolve(process.cwd(), 'tests/fixtures/pytorch-uni-backbone.json');
const fixture: UniBackboneFixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8'));
const model = createDefaultUniCoreModel();

function computeCoreWeightsSha256(): {
  sha256: string;
  tensorCount: number;
  totalFloats: number;
  totalBytes: number;
} {
  const tensors = collectUniBackboneWeights(model);
  let totalFloats = 0;
  for (const t of tensors) {
    totalFloats += t.data.length;
  }
  const buf = Buffer.allocUnsafe(totalFloats * 4);
  let offset = 0;
  for (const t of tensors) {
    for (let i = 0; i < t.data.length; i++) {
      buf.writeFloatLE(t.data[i], (offset + i) * 4);
    }
    offset += t.data.length;
  }
  return {
    sha256: createHash('sha256').update(buf).digest('hex'),
    tensorCount: tensors.length,
    totalFloats,
    totalBytes: buf.byteLength,
  };
}

function collectTsFilesRecursively(dir: string): string[] {
  const results: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...collectTsFilesRecursively(full));
    } else if (entry.isFile() && entry.name.endsWith('.ts')) {
      results.push(full);
    }
  }
  return results;
}

function snapshotKvs(
  kvs: readonly (UniLayerKvSlice | null)[],
): ( { k: Float32Array; v: Float32Array; seqLen: number } | null )[] {
  return kvs.map((slice) =>
    slice === null
      ? null
      : { k: new Float32Array(slice.k), v: new Float32Array(slice.v), seqLen: slice.seqLen },
  );
}

// ============================================================================
// SUITE
// ============================================================================

describe('NLD-Uni 34L Elastic-MoT Core Backbone — Independent PyTorch 2.15.0 Parity', () => {
  it('cryptographically verifies all 618 core tensors (139,024 floats = 556,096 B) and pinned torch==2.15.0', () => {
    expect(fixture.torchVersion.startsWith('2.15.0')).toBe(true);
    expect(fixture.tensorCount).toBe(UNI_MINI_CORE_TENSOR_COUNT);
    expect(fixture.totalFloats).toBe(UNI_MINI_CORE_FLOAT_COUNT);
    expect(fixture.totalBytes).toBe(UNI_MINI_CORE_WEIGHT_BYTES);

    const stats = computeCoreWeightsSha256();
    expect(stats.tensorCount).toBe(618);
    expect(stats.totalFloats).toBe(139_024);
    expect(stats.totalBytes).toBe(556_096);
    expect(stats.sha256).toBe(fixture.weightsSha256);
    expect(model.backboneFingerprint).toBe(fixture.backboneFingerprint);

    // Canonical order spot-checks
    const tensors = collectUniBackboneWeights(model);
    expect(tensors[0].name).toBe('encoder.embed_tokens.weight');
    expect(tensors[1].name).toBe('encoder.layers.0.input_layernorm.weight');
    expect(tensors[10].name).toBe('encoder.layers.0.input_layernorm_gen.weight');
    expect(tensors[1 + 18 * 18].name).toBe('encoder.layers.18.input_layernorm.weight');
    expect(tensors[1 + 18 * 18].splitTower).toBe(true);
    expect(tensors[1 + 18 * 17].splitTower).toBe(false);
    expect(tensors[613].name).toBe('encoder.norm.weight');
    expect(tensors[614].name).toBe('diffusion_head.weight');
    expect(tensors[615].name).toBe('encoder.norm_gen.weight');
    expect(tensors[616].name).toBe('encoder.gen_embedding.weight');
    expect(tensors[617].name).toBe('encoder.gen_predictor.weight');

    // Deterministic reproducibility & seed sensitivity
    const model2 = createDefaultUniCoreModel();
    expect(model2.backboneFingerprint).toBe(model.backboneFingerprint);
    const modelOther = createUniCoreModel({ backboneSeed: 0x12345678 });
    expect(modelOther.backboneFingerprint).not.toBe(model.backboneFingerprint);
  });

  it('Category 1: matches PyTorch 2.15.0 dyadic YaRN inv_freq (0 ULP), cos/sin, rotate_half, and Llama-4 scales across pos 0..35', () => {
    const yarn = computeUniYarnInvFreq(model.config.rope);
    assertBitExactFiniteArray(yarn.invFreq, UNI_DYADIC_YARN_INV_FREQ, 'yarn.invFreq.dyadic');
    assertBitExactFiniteArray(yarn.invFreq, fixture.rope.invFreq, 'yarn.invFreq.pytorch');
    assertBitExactFiniteArray(
      yarn.extrapolationFactor,
      fixture.rope.extrapolationFactor,
      'yarn.extrapolationFactor',
    );
    expect(yarn.attentionScaling).toBe(fixture.rope.attentionScaling);
    expect(yarn.low).toBe(fixture.rope.low);
    expect(yarn.high).toBe(fixture.rope.high);

    const pos = Int32Array.from(fixture.rope.positions);
    const { cos, sin } = computeUniRopeCosSin(
      yarn.invFreq,
      pos,
      yarn.attentionScaling,
      model.config.headDim,
    );
    assertFiniteArrayParity(cos, fixture.rope.cos, 1e-5, 'rope.cos[0..35]');
    assertFiniteArrayParity(sin, fixture.rope.sin, 1e-5, 'rope.sin[0..35]');

    const llama4 = computeLlama4AttnScales(
      pos,
      model.config.rope.llama4Beta,
      model.config.rope.originalMaxPositionEmbeddings,
    );
    assertFiniteArrayParity(llama4, fixture.rope.llama4Scales, 1e-6, 'rope.llama4Scales[0..35]');
    // Verify pos < 16 is exact 1.0, pos 16..31 > 1.069, pos 32..35 > 1.109
    for (let p = 0; p < 16; p++) expect(llama4[p]).toBe(1.0);
    expect(llama4[16]).toBeGreaterThan(1.069);
    expect(llama4[32]).toBeGreaterThan(1.109);

    const sPos = Int32Array.from(fixture.rope.sampleRotate.positionIds);
    const sCosSin = computeUniRopeCosSin(
      yarn.invFreq,
      sPos,
      yarn.attentionScaling,
      model.config.headDim,
    );
    const qRot = applyUniRotaryPosEmb(
      Float32Array.from(fixture.rope.sampleRotate.qIn),
      sPos.length,
      model.config.numAttentionHeads,
      model.config.headDim,
      sCosSin.cos,
      sCosSin.sin,
    );
    const kRot = applyUniRotaryPosEmb(
      Float32Array.from(fixture.rope.sampleRotate.kIn),
      sPos.length,
      model.config.numKeyValueHeads,
      model.config.headDim,
      sCosSin.cos,
      sCosSin.sin,
    );
    assertFiniteArrayParity(qRot, fixture.rope.sampleRotate.qRot, 1e-5, 'rope.sampleRotate.qRot');
    assertFiniteArrayParity(kRot, fixture.rope.sampleRotate.kRot, 1e-5, 'rope.sampleRotate.kRot');
  });

  it('Category 2: matches PyTorch 2.15.0 2D SinCos spatial embeddings and 0-ULP content-column alignment', () => {
    const marked = compute2dSinCosPosEmbed(2, 3, 8, true, model.config.sinCos2d);
    const unmarked = compute2dSinCosPosEmbed(2, 2, 8, false, model.config.sinCos2d);

    expect(marked.contentWidth).toBe(2);
    expect(unmarked.contentWidth).toBe(2);
    expect(marked.columnStride).toBe(fixture.sinCos2d.marked2x3.columnStride);
    expect(unmarked.columnStride).toBe(fixture.sinCos2d.unmarked2x2.columnStride);

    assertFiniteArrayParity(
      marked.frequencies,
      fixture.sinCos2d.marked2x3.frequencies,
      1e-6,
      'sinCos2d.marked.freq',
    );
    assertFiniteArrayParity(
      marked.embeddings,
      fixture.sinCos2d.marked2x3.embeddings,
      1e-6,
      'sinCos2d.marked.emb',
    );
    assertFiniteArrayParity(
      unmarked.embeddings,
      fixture.sinCos2d.unmarked2x2.embeddings,
      1e-6,
      'sinCos2d.unmarked.emb',
    );

    // 0-ULP bit-exact match on content columns c in {0, 1} for every row r in {0, 1}
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 2; c++) {
        const mOff = (r * 3 + c) * 8;
        const uOff = (r * 2 + c) * 8;
        assertBitExactFiniteArray(
          marked.embeddings.subarray(mOff, mOff + 8),
          unmarked.embeddings.subarray(uOff, uOff + 8),
          `sinCos2d.contentCol[${r},${c}]`,
        );
      }
    }
  });

  it('Category 3: matches PyTorch 2.15.0 single-layer Joint (L0, L17) vs Split-Tower (L18, L33) mixed U+G traces (<= 1e-5)', () => {
    const sl = fixture.singleLayers;
    const yarn = computeUniYarnInvFreq(model.config.rope);
    const routing = Uint8Array.from(sl.routing);
    const pos = Int32Array.from(sl.positionIds);
    const inputsEmbeds = Float32Array.from(sl.inputsEmbeds);

    for (const layerIdx of [0, 17, 18, 33] as const) {
      const ref = sl.layers[String(layerIdx)];
      const trace = runUniElasticLayer({
        hiddenStates: inputsEmbeds,
        seqLen: sl.seqLen,
        routing,
        layerWeights: model.layers[layerIdx],
        config: model.config,
        invFreq: yarn.invFreq,
        attentionScaling: yarn.attentionScaling,
        positionIds: pos,
        cachePositions: pos,
        maskMode: sl.maskMode,
        useCache: false,
      });

      expect(trace.splitTower).toBe(layerIdx >= 18);
      assertBitExactFiniteArray(
        trace.attention.effectiveMask,
        ref.effectiveMask,
        `L${layerIdx}.effectiveMask`,
      );
      assertFiniteArrayParity(trace.inputNorm, ref.inputNorm, 1e-5, `L${layerIdx}.inputNorm`);
      assertFiniteArrayParity(trace.attention.qPre, ref.qPre, 1e-5, `L${layerIdx}.qPre`);
      assertFiniteArrayParity(trace.attention.kPre, ref.kPre, 1e-5, `L${layerIdx}.kPre`);
      assertFiniteArrayParity(trace.attention.vNew, ref.vNew, 1e-5, `L${layerIdx}.vNew`);
      assertFiniteArrayParity(trace.attention.qRot, ref.qRot, 1e-5, `L${layerIdx}.qRot`);
      assertFiniteArrayParity(trace.attention.kRot, ref.kRot, 1e-5, `L${layerIdx}.kRot`);
      assertFiniteArrayParity(
        trace.attention.llama4Scales,
        ref.llama4Scales,
        1e-6,
        `L${layerIdx}.llama4Scales`,
      );
      assertFiniteArrayParity(trace.attention.qScaled, ref.qScaled, 1e-5, `L${layerIdx}.qScaled`);
      assertCausalScoresParity(trace.attention.scores, ref.scores, 1e-5, `L${layerIdx}.scores`);
      assertFiniteArrayParity(trace.attention.probs, ref.probs, 1e-5, `L${layerIdx}.probs`);
      assertFiniteArrayParity(
        trace.attention.attnConcat,
        ref.attnConcat,
        1e-5,
        `L${layerIdx}.attnConcat`,
      );
      assertFiniteArrayParity(trace.attention.output, ref.attnOutput, 1e-5, `L${layerIdx}.attnOut`);
      assertFiniteArrayParity(
        trace.postAttentionResidual,
        ref.postAttentionResidual,
        1e-5,
        `L${layerIdx}.postAttnRes`,
      );
      assertFiniteArrayParity(
        trace.postAttentionNorm,
        ref.postAttentionNorm,
        1e-5,
        `L${layerIdx}.postAttnNorm`,
      );
      assertFiniteArrayParity(
        trace.swiglu.gatePre,
        ref.swigluGatePre,
        1e-5,
        `L${layerIdx}.swiglu.gatePre`,
      );
      assertFiniteArrayParity(
        trace.swiglu.gateSilu,
        ref.swigluGateSilu,
        1e-5,
        `L${layerIdx}.swiglu.gateSilu`,
      );
      assertFiniteArrayParity(
        trace.swiglu.upVal,
        ref.swigluUpVal,
        1e-5,
        `L${layerIdx}.swiglu.upVal`,
      );
      assertFiniteArrayParity(
        trace.swiglu.gated,
        ref.swigluGated,
        1e-5,
        `L${layerIdx}.swiglu.gated`,
      );
      assertFiniteArrayParity(
        trace.swiglu.output,
        ref.swigluOutput,
        1e-5,
        `L${layerIdx}.swiglu.output`,
      );
      assertFiniteArrayParity(trace.output, ref.output, 1e-5, `L${layerIdx}.output`);
    }
  });

  it('Category 4: matches PyTorch 2.15.0 full 34L U-only Understanding (prefill 14 + cached decode 4 crossing pos=16)', () => {
    const uRef = fixture.understanding34L;
    const prefill = runUniForwardPass(model, {
      tokenIds: uRef.prefill.tokenIds,
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'u',
    });

    expect(prefill.canonicalRouting).toBe('u_only');
    expect(prefill.telemetry.executedLayerCount).toBe(34);
    expect(prefill.telemetry.wastedPrefixKvLayers).toBe(0);

    for (const k of fixture.keyCheckpointLayers) {
      assertFiniteArrayParity(
        prefill.residualCheckpoints[k]!,
        uRef.prefill.checkpoints[String(k)],
        2e-5,
        `u_prefill.ckpt[${k}]`,
      );
    }
    assertFiniteArrayParity(prefill.finalNorm!, uRef.prefill.finalNorm, 2e-5, 'u_prefill.finalNorm');
    assertFiniteArrayParity(
      prefill.textLogits!,
      uRef.prefill.textLogits,
      5e-5,
      'u_prefill.textLogits',
    );
    assertFiniteArrayParity(prefill.textProbs!, uRef.prefill.textProbs, 5e-5, 'u_prefill.textProbs');

    // Snapshot pastKvs & pastRouting to verify strict non-mutation during cached decode
    const kvSnap = snapshotKvs(prefill.newKvs);
    const routingSnap = new Uint8Array(prefill.routing);

    const decode = runUniForwardPass(model, {
      tokenIds: uRef.decode.tokenIds,
      maskMode: 'causal',
      useCache: true,
      pastKvs: prefill.newKvs,
      pastRouting: prefill.routing,
    });

    expect(Array.from(decode.positionIds)).toEqual([14, 15, 16, 17]);
    for (const k of fixture.keyCheckpointLayers) {
      assertFiniteArrayParity(
        decode.residualCheckpoints[k]!,
        uRef.decode.checkpoints[String(k)],
        2e-5,
        `u_decode.ckpt[${k}]`,
      );
    }
    assertFiniteArrayParity(decode.finalNorm!, uRef.decode.finalNorm, 2e-5, 'u_decode.finalNorm');
    assertFiniteArrayParity(
      decode.textLogits!,
      uRef.decode.textLogits,
      5e-5,
      'u_decode.textLogits',
    );
    assertFiniteArrayParity(decode.textProbs!, uRef.decode.textProbs, 5e-5, 'u_decode.textProbs');

    // Non-mutation verification
    assertBitExactFiniteArray(prefill.routing, routingSnap, 'u_prefill.routing.unmutated');
    for (let l = 0; l < 34; l++) {
      assertBitExactFiniteArray(prefill.newKvs[l]!.k, kvSnap[l]!.k, `u_prefill.kv[${l}].k`);
      assertBitExactFiniteArray(prefill.newKvs[l]!.v, kvSnap[l]!.v, `u_prefill.kv[${l}].v`);
    }
  });

  it('Category 5: matches PyTorch 2.15.0 full 34L G-only Generation forward (len 10, bidirectional)', () => {
    const gRef = fixture.genOnly34L;
    const res = runUniForwardPass(model, {
      genTokenIds: gRef.genTokenIds,
      maskMode: 'bidirectional',
      useCache: false,
    });

    expect(res.canonicalRouting).toBe('g_only');
    expect(res.textLogits).toBeNull();
    expect(res.genLogits).not.toBeNull();

    for (const k of fixture.keyCheckpointLayers) {
      assertFiniteArrayParity(
        res.residualCheckpoints[k]!,
        gRef.checkpoints[String(k)],
        2e-5,
        `g_only.ckpt[${k}]`,
      );
    }
    assertFiniteArrayParity(res.finalNorm!, gRef.finalNorm, 2e-5, 'g_only.finalNorm');
    assertFiniteArrayParity(res.genLogits!, gRef.genLogits, 5e-5, 'g_only.genLogits');
    assertFiniteArrayParity(res.genProbs!, gRef.genProbs, 5e-5, 'g_only.genProbs');
  });

  it('Category 6: T2I 2-Stage D5 — elastic_pruned (18L) vs hf_faithful (34L) produces 0-ULP bit-exact Stage 2 and matches PyTorch 2.15.0', () => {
    const t2i = fixture.t2iTwoStageD5;
    expect(t2i.stage2.pytorchBitExactPrunedVsHf).toBe(true);

    const s1Hf = runUniForwardPass(model, {
      tokenIds: t2i.promptTokenIds,
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'g',
    });
    const s1Pruned = runUniForwardPass(model, {
      tokenIds: t2i.promptTokenIds,
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'elastic_pruned',
      downstreamTarget: 'g',
    });

    // Verify D5 telemetry & skipped layers
    expect(s1Hf.telemetry.executedLayerCount).toBe(34);
    expect(s1Hf.telemetry.wastedPrefixKvLayers).toBe(16);
    expect(s1Pruned.telemetry.executedLayerCount).toBe(18);
    expect(s1Pruned.telemetry.wastedPrefixKvLayers).toBe(0);
    expect(s1Pruned.telemetry.skippedLayerIndices).toEqual(t2i.stage1Pruned.skippedLayers);
    expect(s1Pruned.finalNorm).toBeNull();
    expect(s1Pruned.textLogits).toBeNull();
    for (let l = 18; l < 34; l++) {
      expect(s1Pruned.newKvs[l]).toBeNull();
      expect(s1Pruned.residualCheckpoints[l + 1]).toBeNull();
    }

    // L0..17 Stage 1 checkpoints & KVs match 0 ULP between hf_faithful and elastic_pruned
    for (let l = 0; l <= 18; l++) {
      assertBitExactFiniteArray(
        s1Pruned.residualCheckpoints[l]!,
        s1Hf.residualCheckpoints[l]!,
        `t2i.s1.ckpt[${l}].prunedVsHf`,
      );
    }

    // Stage 1 PyTorch parity
    for (const k of fixture.keyCheckpointLayers) {
      assertFiniteArrayParity(
        s1Hf.residualCheckpoints[k]!,
        t2i.stage1Hf.checkpoints[String(k)],
        2e-5,
        `t2i.s1Hf.ckpt[${k}]`,
      );
    }
    assertFiniteArrayParity(s1Hf.finalNorm!, t2i.stage1Hf.finalNorm, 2e-5, 't2i.s1Hf.finalNorm');
    assertFiniteArrayParity(s1Hf.textLogits!, t2i.stage1Hf.textLogits, 5e-5, 't2i.s1Hf.textLogits');

    // Stage 2 G-only bidirectional denoising from BOTH prefills
    const s1HfKvSnap = snapshotKvs(s1Hf.newKvs);
    const s2FromHf = runUniForwardPass(model, {
      genTokenIds: t2i.genTokenIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKvs: s1Hf.newKvs,
      pastRouting: s1Hf.routing,
    });
    const s2FromPruned = runUniForwardPass(model, {
      genTokenIds: t2i.genTokenIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKvs: s1Pruned.newKvs,
      pastRouting: s1Pruned.routing,
    });

    // 0-ULP BIT-EXACT EQUIVALENCE across all 35 residual checkpoints + finalNorm + genLogits + genProbs!
    for (let l = 0; l <= 34; l++) {
      assertBitExactFiniteArray(
        s2FromPruned.residualCheckpoints[l]!,
        s2FromHf.residualCheckpoints[l]!,
        `t2i.s2.D5.ckpt[${l}]`,
      );
    }
    assertBitExactFiniteArray(s2FromPruned.finalNorm!, s2FromHf.finalNorm!, 't2i.s2.D5.finalNorm');
    assertBitExactFiniteArray(s2FromPruned.genLogits!, s2FromHf.genLogits!, 't2i.s2.D5.genLogits');
    assertBitExactFiniteArray(s2FromPruned.genProbs!, s2FromHf.genProbs!, 't2i.s2.D5.genProbs');

    // Verify droppedCachedUnderstandingLayers & 0-ULP mask parity
    expect(Array.from(s2FromPruned.positionIds)).toEqual(t2i.stage2.positionIds);
    expect(s2FromHf.telemetry.droppedCachedUnderstandingLayers).toEqual(
      t2i.stage2.droppedCachedUnderstandingLayers,
    );
    expect(s2FromPruned.telemetry.droppedCachedUnderstandingLayers).toEqual(
      t2i.stage2.droppedCachedUnderstandingLayers,
    );
    assertBitExactFiniteArray(
      s2FromPruned.layerTraces.get(17)!.attention.effectiveMask,
      t2i.stage2.l17EffectiveMask,
      't2i.s2.L17.mask',
    );
    assertBitExactFiniteArray(
      s2FromPruned.layerTraces.get(18)!.attention.effectiveMask,
      t2i.stage2.l18EffectiveMask,
      't2i.s2.L18.mask',
    );

    // Stage 2 PyTorch parity
    for (const k of fixture.keyCheckpointLayers) {
      assertFiniteArrayParity(
        s2FromPruned.residualCheckpoints[k]!,
        t2i.stage2.checkpoints[String(k)],
        2e-5,
        `t2i.s2.ckpt[${k}]`,
      );
    }
    assertFiniteArrayParity(
      s2FromPruned.finalNorm!,
      t2i.stage2.finalNorm,
      2e-5,
      't2i.s2.finalNorm',
    );
    assertFiniteArrayParity(
      s2FromPruned.genLogits!,
      t2i.stage2.genLogits,
      5e-5,
      't2i.s2.genLogits',
    );
    assertFiniteArrayParity(s2FromPruned.genProbs!, t2i.stage2.genProbs, 5e-5, 't2i.s2.genProbs');

    // Verify pastKvs unmutated
    for (let l = 0; l < 34; l++) {
      assertBitExactFiniteArray(s1Hf.newKvs[l]!.k, s1HfKvSnap[l]!.k, `t2i.s1Hf.kv[${l}].k`);
      assertBitExactFiniteArray(s1Hf.newKvs[l]!.v, s1HfKvSnap[l]!.v, `t2i.s1Hf.kv[${l}].v`);
    }
  });

  it('Category 7: Mixed U+G Editing 2-Stage D5 — elastic_pruned vs hf_faithful is 0-ULP bit-exact and isolates L18..33 U prefix', () => {
    const ed = fixture.editingTwoStageD5;
    expect(ed.stage2.pytorchBitExactPrunedVsHf).toBe(true);
    expect(ed.stage2.pytorchCachedUPerturbZeroUlp).toBe(true);

    // Reconstruct Stage 1 mixed U+G inputsEmbeds cleanly from model embeddings
    const uEmb = embedUniTextTokens(ed.stage1UnderstandingTokenIds, model);
    const gEmb = embedUniGenTokens(ed.stage1SourceGenTokenIds, model);
    const s1Embeds = new Float32Array(uEmb.length + gEmb.length);
    s1Embeds.set(uEmb, 0);
    s1Embeds.set(gEmb, uEmb.length);
    assertBitExactFiniteArray(s1Embeds, ed.stage1InputsEmbeds, 'edit.s1.inputsEmbeds');

    const s1Hf = runUniForwardPass(model, {
      inputsEmbeds: s1Embeds,
      routing: ed.stage1Routing,
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'g',
    });
    const s1Pruned = runUniForwardPass(model, {
      inputsEmbeds: s1Embeds,
      routing: ed.stage1Routing,
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'elastic_pruned',
      downstreamTarget: 'g',
    });

    expect(s1Hf.telemetry.executedLayerCount).toBe(34);
    expect(s1Hf.telemetry.wastedPrefixKvLayers).toBe(16);
    expect(s1Pruned.telemetry.executedLayerCount).toBe(34);
    expect(s1Pruned.telemetry.understandingTokensPrunedAtSplit).toBe(8);
    expect(s1Pruned.telemetry.wastedPrefixKvLayers).toBe(0);
    expect(s1Pruned.textLogits).toBeNull();
    expect(s1Pruned.textProbs).toBeNull();
    expect(s1Pruned.genLogits).not.toBeNull();

    // U tokens (0..7) remain frozen at L17 residual across L18..33 and skipped in finalNorm in elastic_pruned,
    // while G tokens (8..11) match hf_faithful to 0 ULP!
    const d = model.config.hiddenSize;
    const l17U = s1Pruned.residualCheckpoints[18]!.subarray(0, 8 * d);
    for (let i = 0; i < 8 * d; i++) {
      expect(s1Pruned.finalNorm![i]).toBe(0);
    }
    assertBitExactFiniteArray(
      s1Pruned.finalNorm!.subarray(8 * d),
      s1Hf.finalNorm!.subarray(8 * d),
      'edit.s1Pruned.finalNorm.G_exact',
    );
    assertFiniteArrayParity(
      s1Pruned.finalNorm!,
      ed.stage1Pruned.finalNorm,
      2e-5,
      'edit.s1Pruned.finalNorm',
    );
    for (let l = 18; l < 34; l++) {
      const ckptPr = s1Pruned.residualCheckpoints[l + 1]!;
      const ckptHf = s1Hf.residualCheckpoints[l + 1]!;
      assertBitExactFiniteArray(ckptPr.subarray(0, 8 * d), l17U, `edit.s1Pruned.U_frozen[L${l}]`);
      assertBitExactFiniteArray(
        ckptPr.subarray(8 * d),
        ckptHf.subarray(8 * d),
        `edit.s1Pruned.G_exact[L${l}]`,
      );
    }

    // Stage 1 PyTorch parity
    for (const k of fixture.keyCheckpointLayers) {
      assertFiniteArrayParity(
        s1Hf.residualCheckpoints[k]!,
        ed.stage1Hf.checkpoints[String(k)],
        2e-5,
        `edit.s1Hf.ckpt[${k}]`,
      );
      assertFiniteArrayParity(
        s1Pruned.residualCheckpoints[k]!,
        ed.stage1Pruned.checkpoints[String(k)],
        2e-5,
        `edit.s1Pruned.ckpt[${k}]`,
      );
    }

    // Stage 2 target G denoising from hf_faithful vs elastic_pruned
    const s2FromHf = runUniForwardPass(model, {
      genTokenIds: ed.stage2GenTokenIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKvs: s1Hf.newKvs,
      pastRouting: s1Hf.routing,
    });
    const s2FromPruned = runUniForwardPass(model, {
      genTokenIds: ed.stage2GenTokenIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKvs: s1Pruned.newKvs,
      pastRouting: s1Pruned.routing,
    });

    // 0-ULP BIT-EXACT Stage 2 equivalence!
    for (let l = 0; l <= 34; l++) {
      assertBitExactFiniteArray(
        s2FromPruned.residualCheckpoints[l]!,
        s2FromHf.residualCheckpoints[l]!,
        `edit.s2.D5.ckpt[${l}]`,
      );
    }
    assertBitExactFiniteArray(s2FromPruned.finalNorm!, s2FromHf.finalNorm!, 'edit.s2.D5.finalNorm');
    assertBitExactFiniteArray(s2FromPruned.genLogits!, s2FromHf.genLogits!, 'edit.s2.D5.genLogits');
    assertBitExactFiniteArray(s2FromPruned.genProbs!, s2FromHf.genProbs!, 'edit.s2.D5.genProbs');

    // Editing Stage 2 keeps cached G prefix so droppedCachedUnderstandingLayers is empty
    expect(s2FromPruned.telemetry.droppedCachedUnderstandingLayers).toEqual([]);
    assertBitExactFiniteArray(
      s2FromPruned.layerTraces.get(17)!.attention.effectiveMask,
      ed.stage2.l17EffectiveMask,
      'edit.s2.L17.mask',
    );
    assertBitExactFiniteArray(
      s2FromPruned.layerTraces.get(18)!.attention.effectiveMask,
      ed.stage2.l18EffectiveMask,
      'edit.s2.L18.mask',
    );

    // Prove L18..33 target G attends ONLY to cached G (8..11) + current G (12..21), NOT cached U (0..7)
    const kvDim = model.config.kvProjOutDim;
    const perturbedUKvs: UniLayerKvSlice[] = [];
    const perturbedGKvs: UniLayerKvSlice[] = [];
    for (let l = 0; l < 34; l++) {
      const base = s1Hf.newKvs[l]!;
      const kU = new Float32Array(base.k);
      const vU = new Float32Array(base.v);
      const kG = new Float32Array(base.k);
      const vG = new Float32Array(base.v);
      if (l >= 18) {
        for (let i = 0; i < 8 * kvDim; i++) {
          kU[i] = Math.fround(kU[i] * 100.0 + 7.5);
          vU[i] = Math.fround(vU[i] * 100.0 - 3.25);
        }
        for (let i = 8 * kvDim; i < 12 * kvDim; i++) {
          kG[i] = Math.fround(kG[i] * 2.0 + 1.0);
          vG[i] = Math.fround(vG[i] * 2.0 - 1.0);
        }
      }
      perturbedUKvs.push({ k: kU, v: vU, seqLen: base.seqLen });
      perturbedGKvs.push({ k: kG, v: vG, seqLen: base.seqLen });
    }

    const s2PerturbU = runUniForwardPass(model, {
      genTokenIds: ed.stage2GenTokenIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKvs: perturbedUKvs,
      pastRouting: s1Hf.routing,
    });
    const s2PerturbG = runUniForwardPass(model, {
      genTokenIds: ed.stage2GenTokenIds,
      maskMode: 'bidirectional',
      useCache: false,
      pastKvs: perturbedGKvs,
      pastRouting: s1Hf.routing,
    });

    assertBitExactFiniteArray(
      s2PerturbU.genLogits!,
      s2FromHf.genLogits!,
      'edit.s2.cachedUPerturbZeroUlp',
    );
    let maxGPerturbDiff = 0;
    for (let i = 0; i < s2FromHf.genLogits!.length; i++) {
      const diff = Math.abs(s2FromHf.genLogits![i] - s2PerturbG.genLogits![i]);
      if (diff > maxGPerturbDiff) maxGPerturbDiff = diff;
    }
    expect(maxGPerturbDiff).toBeGreaterThan(1e-4);

    // Stage 2 PyTorch parity
    for (const k of fixture.keyCheckpointLayers) {
      assertFiniteArrayParity(
        s2FromPruned.residualCheckpoints[k]!,
        ed.stage2.checkpoints[String(k)],
        2e-5,
        `edit.s2.ckpt[${k}]`,
      );
    }
    assertFiniteArrayParity(
      s2FromPruned.finalNorm!,
      ed.stage2.finalNorm,
      2e-5,
      'edit.s2.finalNorm',
    );
    assertFiniteArrayParity(
      s2FromPruned.genLogits!,
      ed.stage2.genLogits,
      5e-5,
      'edit.s2.genLogits',
    );
    assertFiniteArrayParity(s2FromPruned.genProbs!, ed.stage2.genProbs, 5e-5, 'edit.s2.genProbs');
  });

  it('enforces zero forbidden sibling imports and zero Math.random() across all src/uni/ modules', () => {
    const uniDir = resolve(process.cwd(), 'src/uni');
    const files = collectTsFilesRecursively(uniDir);
    expect(files.length).toBeGreaterThanOrEqual(3);

    for (const file of files) {
      const raw = readFileSync(file, 'utf-8');
      const codeOnly = raw.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
      expect(codeOnly, `Forbidden twotower import in ${file}`).not.toMatch(
        /from\s+['"][^'"]*twotower/,
      );
      expect(codeOnly, `Forbidden diffusion import in ${file}`).not.toMatch(
        /from\s+['"][^'"]*diffusion/,
      );
      expect(codeOnly, `Forbidden ui/controls import in ${file}`).not.toMatch(
        /from\s+['"][^'"]*ui\/controls/,
      );
      expect(codeOnly, `Forbidden Math.random() in ${file}`).not.toMatch(/\bMath\.random\s*\(/);
    }
  });
});
