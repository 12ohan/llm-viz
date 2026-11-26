/**
 * Independent PyTorch 2.15.0 Full Multimodal Parity, KV Cache, Block-Diffusion &
 * Object Grounding Verification Suite (`tests/uni-understanding-parity.test.ts`)
 *
 * Gate G2 Verification Suite covering:
 *   - Cryptographic SHA-256 over ALL 766 tensors (618 core + 148 multimodal = 191,235 floats = 764,940 B)
 *     and pinned `torch==2.15.0`
 *   - Category 1: Mini RADIO-H ViT-2L + RadioHProjector (`sunlit_cat_16x16` 4x4->2x2 &
 *     `harbor_panorama_16x8` 2x4 bilinear CPE->1x2, BOTH D3 gain presets `0.25` & `0.01`, `<= 2e-5`)
 *   - Category 2: Mini Emu3VQ Encoder + IBQ `argmax(z @ E^T)` (0-ULP indices) + Decoder
 *     on `sunlit_cat_16x16` & letterboxed `harbor_panorama_16x8 -> 16x16` (`<= 2e-5`)
 *   - Category 3: PackedLinearImageCodec compress + expand (`gen_predictor` logits) on
 *     target gen (`4x6->2x3`) & source edit (`4x4->2x2`) under BOTH D3 presets (`<= 2e-5`),
 *     plus 0-ULP `+2` row markers, fixed prefix IDs & `_source_padding_vq_mask`
 *   - Category 4: Multimodal Visual Understanding Block-Diffusion (`B=4`, `shift_logits=true`,
 *     spliced RADIO-H embeddings, BOTH `causalContext=true/false`, greedy `T=0` & stochastic `T=0.7`
 *     with shared explicit uniform variates `U23`; 0-ULP tokens & `<= 5e-5` logits/probs)
 *   - Category 5: `<LOC_0..16>` Object Grounding (`steps=2` single-tag & multi-tag, exact
 *     `pixel_box` un-letterboxing, direct ID decode without `U27` stripping; 0-ULP IDs & `<= 5e-5` confs)
 *   - Unit & Guard Coverage: `UniKvCacheState` (`B_cfg ∈ {1,2,3}`, slab immutability/sharing,
 *     D5 waste telemetry `16/34` vs `0`, capacity exhaustion), `U21` `[MASK]` / non-`<LOC>` guard
 *     provenance, `U22` log-domain Gumbel safety with extreme logits `> 800`, AR recovery at `B=1`,
 *     EOS KV cropping (`U30`), and `U27` non-stripped grounding tokens.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  assertBitExactFiniteArray,
  assertFiniteArrayParity,
} from './helpers/parityAssert';
import {
  addGenerationRowMarkers,
  applySourcePaddingVqLock,
  buildFixedPrefixTokenIds,
  collectUniCodecWeights,
  compressPackedImageEmbeddings,
  computeSourcePaddingVqMask,
  createUniPackedCodecWeights,
  expandPackedImageHiddenStates,
} from '../src/uni/codec';
import { embedUniGenTokens, embedUniTextTokens, runUniForwardPass } from '../src/uni/forward';
import {
  appendBatchKvsToCache,
  appendSingleRowKvsToCache,
  computeUniKvWasteTelemetry,
  createEmptyUniKvCache,
  cropUniKvCache,
  extractRowPastKvs,
} from '../src/uni/kvCache';
import {
  collectUniBackboneWeights,
  createDefaultUniCoreModel,
} from '../src/uni/parameters';
import { createXoshiro128ss } from '../src/uni/prng';
import {
  generateWithPrefixCacheBlockDiff,
  groundObjects,
  locBinsToPixelBox,
  sampleLogDomainGumbelArgmax,
} from '../src/uni/textSampler';
import {
  UNI_MINI_CODEC_TENSOR_COUNT,
  UNI_MINI_CORE_FLOAT_COUNT,
  UNI_MINI_CORE_TENSOR_COUNT,
  UNI_MINI_EOS_TOKEN_ID,
  UNI_MINI_FULL_FLOAT_COUNT,
  UNI_MINI_FULL_TENSOR_COUNT,
  UNI_MINI_FULL_WEIGHT_BYTES,
  UNI_MINI_LOC_END_ID,
  UNI_MINI_LOC_START_ID,
  UNI_MINI_MASK_TOKEN_ID,
  UNI_MINI_MULTIMODAL_FLOAT_COUNT,
  UNI_MINI_MULTIMODAL_TENSOR_COUNT,
  UNI_MINI_VISION_SUBSYSTEM_TENSOR_COUNT,
  UNI_MINI_VQVAE_TOTAL_TENSOR_COUNT,
  UNI_ROUTE_G,
  UNI_ROUTE_U,
  type UniCoreModelParameters,
  type UniGainPresetId,
} from '../src/uni/types';
import {
  collectUniVisionWeights,
  createUniScenePreset,
  createUniVisionSubsystemWeights,
  fixOnePrompt,
  padToSquareAndResize,
  runMiniVisionPipeline,
  spliceVisualEmbeddingsIntoPrompt,
} from '../src/uni/vision';
import {
  collectUniVqVaeWeights,
  createUniEmu3VqWeights,
  decodeMiniEmu3Vq,
  encodeMiniEmu3Vq,
} from '../src/uni/vqvae';

// ============================================================================
// FIXTURE TYPES & RUNTIME INITIALIZATION
// ============================================================================

interface VisionProjectorCaseFixture {
  patchGridRows: number;
  patchGridCols: number;
  tokenGridRows: number;
  tokenGridCols: number;
  radioMultiply: number;
  clipNormalized: number[];
  rawPatches: number[];
  interpolatedPosEmbed: number[];
  vitInputTokens: number[];
  patchFeatures: number[];
  gathered2x2: number[];
  normed: number[];
  fc1Out: number[];
  squaredReluOut: number[];
  fc2Out: number[];
  visualEmbeddings: number[];
}

interface VqVaeCaseFixture {
  inputWidth: number;
  inputHeight: number;
  vqGridRows: number;
  vqGridCols: number;
  normalizedInput: number[];
  zContinuous: number[];
  dotScores: number[];
  codes: number[];
  zQuantized: number[];
  rawOutput: number[];
  rgbOutput: number[];
}

interface CodecCaseFixture {
  tokenIds: number[];
  childRows: number;
  childCols: number;
  parentRows: number;
  parentCols: number;
  isSourceEdit: boolean;
  markerColumn: boolean;
  gainPresetId: UniGainPresetId;
  combinedScale: number;
  compress: {
    normedChildren: number[];
    packedConcat: number[];
    projected: number[];
    scaled: number[];
    sinCos2d: number[];
    output: number[];
  };
  expand: {
    normedParents: number[];
    upProjected: number[];
    unpackedChildren: number[];
    logits: number[];
  };
}

interface BlockDiffStepFixture {
  blockIndex: number;
  stepIndex: number;
  blockStartPos: number;
  blockTokensBefore: number[];
  blockTokensAfter: number[];
  unmaskedOffsets: number[];
  slotConfidences: number[];
  rawTop1TokenIds: number[];
  guardIntervenedOffsets: number[];
  shiftedLogits: number[];
  shiftedProbs: number[];
  usedCausalPreSeed: boolean;
}

interface BlockDiffCaseFixture {
  promptTokenIds: number[];
  blockLength: number;
  maxNewTokens: number;
  stepsPerBlock: number;
  confidenceThreshold: number;
  temperature: number;
  causalContext: boolean;
  suppressMaskLogit: boolean;
  explicitUniformVariates: number[] | null;
  generatedTokenIds: number[];
  stoppedByEos: boolean;
  eosOffset: number | null;
  stopReason: 'eos' | 'max_new_tokens' | 'capacity_exhausted';
  nfeCount: number;
  totalGuardInterventions: number;
  steps: BlockDiffStepFixture[];
}

interface GroundingCaseFixture {
  basePromptTokenIds: number[];
  expandedPromptTokenIds: number[];
  visualSlotIndices: number[];
  result: {
    promptTokenIds: number[];
    imageWidth: number;
    imageHeight: number;
    tags: string[];
    stepsRequested: 1 | 2 | 4;
    assembledTemplateIds: number[];
    slotOffsetsInFullSequence: number[];
    boxes: Array<{
      tag: string;
      locTokenIds: [number, number, number, number];
      locTokenStrings: [string, string, string, string];
      normalizedBins: [number, number, number, number];
      pixelBox: [number, number, number, number];
      rawTop1Ids: [number, number, number, number];
      guardIntervenedCount: number;
    }>;
    stepTraces: Array<{
      stepIndex: number;
      slotTokenIdsBefore: number[];
      slotTokenIdsAfter: number[];
      unmaskedSlotIndices: number[];
      slotConfidences: number[];
      rawTop1TokenIds: number[];
      guardIntervenedSlotIndices: number[];
    }>;
    skipSpecialTokensWouldStripCount: number;
    totalGuardInterventions: number;
  };
}

interface UniMultimodalFixture {
  schemaVersion: string;
  modelId: string;
  backboneSeed: number;
  multimodalSeed: number;
  backboneFingerprint: string;
  counts: {
    coreTensors: number;
    coreFloats: number;
    visionTensors: number;
    visionFloats: number;
    codecTensors: number;
    codecFloats: number;
    vqvaeTensors: number;
    vqvaeFloats: number;
    multimodalTensors: number;
    multimodalFloats: number;
    totalTensors: number;
    totalFloats: number;
    totalBytes: number;
  };
  weightsSha256: string;
  coreSha256: string;
  multimodalSha256: string;
  torchVersion: string;
  pythonVersion: string;
  visionProjector: {
    sunlitCatMiniDefault: VisionProjectorCaseFixture;
    sunlitCatProdExact: VisionProjectorCaseFixture;
    harborPanoramaMiniDefault: VisionProjectorCaseFixture;
    harborPanoramaProdExact: VisionProjectorCaseFixture;
  };
  vqvae: {
    sunlitCat16x16: VqVaeCaseFixture;
    harborPanoramaLetterboxed16x16: VqVaeCaseFixture;
  };
  packedCodec: {
    markedSunlitCatCodes: number[];
    fixedPrefixTokenIds: number[];
    harborSourcePaddingVqMask: number[];
    targetGenMiniDefault: CodecCaseFixture;
    targetGenProdExact: CodecCaseFixture;
    sourceEditMiniDefault: CodecCaseFixture;
    sourceEditProdExact: CodecCaseFixture;
  };
  multimodalBlockDiffusion: {
    basePromptTokenIds: number[];
    expandedPromptTokenIds: number[];
    visualSlotIndices: number[];
    causalGreedy: BlockDiffCaseFixture;
    nonCausalGreedy: BlockDiffCaseFixture;
    causalStochastic: BlockDiffCaseFixture;
  };
  objectGrounding: {
    sunlitCatSingleTag: GroundingCaseFixture;
    harborPanoramaMultiTag: GroundingCaseFixture;
  };
}

const FIXTURE_PATH = resolve(process.cwd(), 'tests/fixtures/pytorch-uni-multimodal.json');
const fixture: UniMultimodalFixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8'));

const model = createDefaultUniCoreModel();
const visionWeights = createUniVisionSubsystemWeights();
const codecWeights = createUniPackedCodecWeights();
const vqWeights = createUniEmu3VqWeights();

function computeFull766WeightsSha256(): {
  tensorCount: number;
  totalFloats: number;
  totalBytes: number;
  weightsSha256: string;
  coreSha256: string;
  multimodalSha256: string;
} {
  const coreTensors = collectUniBackboneWeights(model);
  const visionTensors = collectUniVisionWeights(visionWeights);
  const codecTensors = collectUniCodecWeights(codecWeights);
  const vqTensors = collectUniVqVaeWeights(vqWeights);

  expect(coreTensors.length).toBe(UNI_MINI_CORE_TENSOR_COUNT);
  expect(visionTensors.length).toBe(UNI_MINI_VISION_SUBSYSTEM_TENSOR_COUNT);
  expect(codecTensors.length).toBe(UNI_MINI_CODEC_TENSOR_COUNT);
  expect(vqTensors.length).toBe(UNI_MINI_VQVAE_TOTAL_TENSOR_COUNT);

  const allTensors = [...coreTensors, ...visionTensors, ...codecTensors, ...vqTensors];
  let totalFloats = 0;
  for (const t of allTensors) {
    totalFloats += t.data.length;
  }

  const buf = Buffer.allocUnsafe(totalFloats * 4);
  let offset = 0;
  for (const t of allTensors) {
    for (let i = 0; i < t.data.length; i++) {
      buf.writeFloatLE(t.data[i], (offset + i) * 4);
    }
    offset += t.data.length;
  }

  const coreBytes = UNI_MINI_CORE_FLOAT_COUNT * 4;
  return {
    tensorCount: allTensors.length,
    totalFloats,
    totalBytes: buf.byteLength,
    weightsSha256: createHash('sha256').update(buf).digest('hex'),
    coreSha256: createHash('sha256').update(buf.subarray(0, coreBytes)).digest('hex'),
    multimodalSha256: createHash('sha256').update(buf.subarray(coreBytes)).digest('hex'),
  };
}

// ============================================================================
// PART 1: CRYPTOGRAPHIC 766-TENSOR SHA-256 & MULTIMODAL PYTORCH 2.15.0 PARITY
// ============================================================================

describe('NLD-Uni Full 766-Tensor Multimodal PyTorch 2.15.0 Parity (Gate G2)', () => {
  it('cryptographically verifies all 766 tensors (618 core + 148 multimodal = 191,235 floats = 764,940 B) and pinned torch==2.15.0', () => {
    expect(fixture.torchVersion.startsWith('2.15.0')).toBe(true);
    expect(fixture.counts.coreTensors).toBe(UNI_MINI_CORE_TENSOR_COUNT);
    expect(fixture.counts.multimodalTensors).toBe(UNI_MINI_MULTIMODAL_TENSOR_COUNT);
    expect(fixture.counts.multimodalFloats).toBe(UNI_MINI_MULTIMODAL_FLOAT_COUNT);
    expect(fixture.counts.totalTensors).toBe(UNI_MINI_FULL_TENSOR_COUNT);
    expect(fixture.counts.totalFloats).toBe(UNI_MINI_FULL_FLOAT_COUNT);
    expect(fixture.counts.totalBytes).toBe(UNI_MINI_FULL_WEIGHT_BYTES);

    const stats = computeFull766WeightsSha256();
    expect(stats.tensorCount).toBe(766);
    expect(stats.totalFloats).toBe(191_235);
    expect(stats.totalBytes).toBe(764_940);
    expect(stats.coreSha256).toBe(fixture.coreSha256);
    expect(stats.multimodalSha256).toBe(fixture.multimodalSha256);
    expect(stats.weightsSha256).toBe(fixture.weightsSha256);
    expect(model.backboneFingerprint).toBe(fixture.backboneFingerprint);
  });

  it('Category 1: matches MiniRadioVisionTower + MiniRadioHProjector <= 2e-5 across both scenes & both D3 gain presets', () => {
    const sunlitCat = createUniScenePreset('sunlit_cat_16x16');
    const harbor = createUniScenePreset('harbor_panorama_16x8');

    const cases: Array<{
      label: string;
      image: typeof sunlitCat;
      preset: UniGainPresetId;
      ref: VisionProjectorCaseFixture;
    }> = [
      {
        label: 'sunlitCat.mini_default',
        image: sunlitCat,
        preset: 'mini_default',
        ref: fixture.visionProjector.sunlitCatMiniDefault,
      },
      {
        label: 'sunlitCat.prod_exact',
        image: sunlitCat,
        preset: 'prod_exact',
        ref: fixture.visionProjector.sunlitCatProdExact,
      },
      {
        label: 'harbor.mini_default',
        image: harbor,
        preset: 'mini_default',
        ref: fixture.visionProjector.harborPanoramaMiniDefault,
      },
      {
        label: 'harbor.prod_exact',
        image: harbor,
        preset: 'prod_exact',
        ref: fixture.visionProjector.harborPanoramaProdExact,
      },
    ];

    for (const c of cases) {
      const trace = runMiniVisionPipeline(c.image, visionWeights, c.preset);
      expect(trace.imageNewlineUnused).toBe(true);
      expect(trace.vision.patchGridRows).toBe(c.ref.patchGridRows);
      expect(trace.vision.patchGridCols).toBe(c.ref.patchGridCols);
      expect(trace.projector.tokenGridRows).toBe(c.ref.tokenGridRows);
      expect(trace.projector.tokenGridCols).toBe(c.ref.tokenGridCols);
      expect(trace.projector.radioMultiply).toBeCloseTo(c.ref.radioMultiply, 12);

      assertFiniteArrayParity(trace.vision.clipNormalized, c.ref.clipNormalized, 2e-5, `${c.label}.clipNormalized`);
      assertFiniteArrayParity(trace.vision.rawPatches, c.ref.rawPatches, 2e-5, `${c.label}.rawPatches`);
      assertFiniteArrayParity(trace.vision.interpolatedPosEmbed, c.ref.interpolatedPosEmbed, 2e-5, `${c.label}.interpolatedPosEmbed`);
      assertFiniteArrayParity(trace.vision.vitInputTokens, c.ref.vitInputTokens, 2e-5, `${c.label}.vitInputTokens`);
      assertFiniteArrayParity(trace.vision.patchFeatures, c.ref.patchFeatures, 2e-5, `${c.label}.patchFeatures`);
      assertFiniteArrayParity(trace.projector.gathered2x2, c.ref.gathered2x2, 2e-5, `${c.label}.gathered2x2`);
      assertFiniteArrayParity(trace.projector.normed, c.ref.normed, 2e-5, `${c.label}.normed`);
      assertFiniteArrayParity(trace.projector.fc1Out, c.ref.fc1Out, 2e-5, `${c.label}.fc1Out`);
      assertFiniteArrayParity(trace.projector.squaredReluOut, c.ref.squaredReluOut, 2e-5, `${c.label}.squaredReluOut`);
      assertFiniteArrayParity(trace.projector.fc2Out, c.ref.fc2Out, 2e-5, `${c.label}.fc2Out`);
      assertFiniteArrayParity(trace.projector.visualEmbeddings, c.ref.visualEmbeddings, 2e-5, `${c.label}.visualEmbeddings`);
    }
  });

  it('Category 2: matches MiniEmu3VQ Encoder, 0-ULP IBQ argmax(z @ E^T) codes & Decoder <= 2e-5 on both scenes', () => {
    const sunlitCat = createUniScenePreset('sunlit_cat_16x16');
    const harborLb = padToSquareAndResize(createUniScenePreset('harbor_panorama_16x8'), 16);

    const cases = [
      { label: 'vq.sunlitCat', image: sunlitCat, ref: fixture.vqvae.sunlitCat16x16 },
      { label: 'vq.harborLetterboxed', image: harborLb.image, ref: fixture.vqvae.harborPanoramaLetterboxed16x16 },
    ];

    for (const c of cases) {
      const enc = encodeMiniEmu3Vq(c.image, vqWeights);
      expect(enc.vqGridRows).toBe(c.ref.vqGridRows);
      expect(enc.vqGridCols).toBe(c.ref.vqGridCols);

      assertFiniteArrayParity(enc.normalizedInput, c.ref.normalizedInput, 2e-5, `${c.label}.normalizedInput`);
      assertFiniteArrayParity(enc.zContinuous, c.ref.zContinuous, 2e-5, `${c.label}.zContinuous`);
      assertFiniteArrayParity(enc.dotScores, c.ref.dotScores, 2e-5, `${c.label}.dotScores`);
      // Exact 0-ULP IBQ indices
      assertBitExactFiniteArray(enc.codes, c.ref.codes, `${c.label}.codes`);
      assertFiniteArrayParity(enc.zQuantized, c.ref.zQuantized, 2e-5, `${c.label}.zQuantized`);

      const dec = decodeMiniEmu3Vq(enc.codes, enc.vqGridRows, enc.vqGridCols, vqWeights);
      assertFiniteArrayParity(dec.rawOutput, c.ref.rawOutput, 2e-5, `${c.label}.rawOutput`);
      assertFiniteArrayParity(dec.rgbImage.data, c.ref.rgbOutput, 2e-5, `${c.label}.rgbOutput`);
    }
  });

  it('Category 3: matches PackedLinearImageCodec compress/expand <= 2e-5 and 0-ULP markers, fixed prefix & padding lock', () => {
    const sunlitCat = createUniScenePreset('sunlit_cat_16x16');
    const harborLb = padToSquareAndResize(createUniScenePreset('harbor_panorama_16x8'), 16);

    const catCodes = encodeMiniEmu3Vq(sunlitCat, vqWeights).codes;
    const markedCodes = addGenerationRowMarkers(catCodes, 4, 4);
    assertBitExactFiniteArray(markedCodes, fixture.packedCodec.markedSunlitCatCodes, 'codec.markedCodes');
    assertBitExactFiniteArray(buildFixedPrefixTokenIds(), fixture.packedCodec.fixedPrefixTokenIds, 'codec.fixedPrefix');

    const padMask = computeSourcePaddingVqMask(harborLb.info, 4);
    assertBitExactFiniteArray(padMask.lockMask, fixture.packedCodec.harborSourcePaddingVqMask, 'codec.paddingMask');
    const harborCodes = encodeMiniEmu3Vq(harborLb.image, vqWeights).codes;
    const lockedCodes = applySourcePaddingVqLock(new Int32Array(16).fill(77), harborCodes, padMask);
    for (let i = 0; i < 16; i++) {
      expect(lockedCodes[i]).toBe(padMask.lockMask[i] === 1 ? harborCodes[i] : 77);
    }

    const codecCases: Array<{ label: string; ref: CodecCaseFixture }> = [
      { label: 'codec.targetGen.mini_default', ref: fixture.packedCodec.targetGenMiniDefault },
      { label: 'codec.targetGen.prod_exact', ref: fixture.packedCodec.targetGenProdExact },
      { label: 'codec.sourceEdit.mini_default', ref: fixture.packedCodec.sourceEditMiniDefault },
      { label: 'codec.sourceEdit.prod_exact', ref: fixture.packedCodec.sourceEditProdExact },
    ];

    for (const c of codecCases) {
      const childEmbs = embedUniGenTokens(c.ref.tokenIds, model);
      const comp = compressPackedImageEmbeddings(
        childEmbs,
        c.ref.childRows,
        c.ref.childCols,
        codecWeights,
        { isSourceEdit: c.ref.isSourceEdit, gainPresetId: c.ref.gainPresetId },
      );

      expect(comp.parentRows).toBe(c.ref.parentRows);
      expect(comp.parentCols).toBe(c.ref.parentCols);
      expect(comp.markerColumn).toBe(c.ref.markerColumn);
      expect(comp.combinedScale).toBeCloseTo(c.ref.combinedScale, 12);

      assertFiniteArrayParity(comp.normedChildren, c.ref.compress.normedChildren, 2e-5, `${c.label}.normedChildren`);
      assertFiniteArrayParity(comp.packedConcat, c.ref.compress.packedConcat, 2e-5, `${c.label}.packedConcat`);
      assertFiniteArrayParity(comp.projected, c.ref.compress.projected, 2e-5, `${c.label}.projected`);
      assertFiniteArrayParity(comp.scaled, c.ref.compress.scaled, 2e-5, `${c.label}.scaled`);
      assertFiniteArrayParity(comp.sinCos2d.embeddings, c.ref.compress.sinCos2d, 2e-5, `${c.label}.sinCos2d`);
      assertFiniteArrayParity(comp.output, c.ref.compress.output, 2e-5, `${c.label}.output`);

      const exp = expandPackedImageHiddenStates(
        comp.output,
        comp.parentRows,
        comp.parentCols,
        codecWeights,
        model.genPredictor,
      );
      assertFiniteArrayParity(exp.normedParents, c.ref.expand.normedParents, 2e-5, `${c.label}.normedParents`);
      assertFiniteArrayParity(exp.upProjected, c.ref.expand.upProjected, 2e-5, `${c.label}.upProjected`);
      assertFiniteArrayParity(exp.unpackedChildren, c.ref.expand.unpackedChildren, 2e-5, `${c.label}.unpackedChildren`);
      expect(exp.logits).not.toBeNull();
      assertFiniteArrayParity(exp.logits!, c.ref.expand.logits, 2e-5, `${c.label}.logits`);
    }
  });

  it('Category 4: matches Multimodal Visual Understanding Block-Diffusion (causal/non-causal, greedy & stochastic T=0.7) at 0-ULP tokens & <= 5e-5 logits', () => {
    const sunlitCat = createUniScenePreset('sunlit_cat_16x16');
    const vis = runMiniVisionPipeline(sunlitCat, visionWeights, 'mini_default');
    const fixed = fixOnePrompt(
      fixture.multimodalBlockDiffusion.basePromptTokenIds,
      vis.projector.tokenGridRows,
      vis.projector.tokenGridCols,
    );
    assertBitExactFiniteArray(
      fixed.expandedTokenIds,
      fixture.multimodalBlockDiffusion.expandedPromptTokenIds,
      'bd.expandedPromptTokenIds',
    );
    assertBitExactFiniteArray(
      fixed.visualSlotIndices,
      fixture.multimodalBlockDiffusion.visualSlotIndices,
      'bd.visualSlotIndices',
    );

    const textEmbs = embedUniTextTokens(fixed.expandedTokenIds, model);
    const splicedEmbs = spliceVisualEmbeddingsIntoPrompt(
      fixed.expandedTokenIds,
      textEmbs,
      fixed.visualSlotIndices,
      vis.projector.visualEmbeddings,
    );

    const bdCases: Array<{ label: string; ref: BlockDiffCaseFixture }> = [
      { label: 'bd.causalGreedy', ref: fixture.multimodalBlockDiffusion.causalGreedy },
      { label: 'bd.nonCausalGreedy', ref: fixture.multimodalBlockDiffusion.nonCausalGreedy },
      { label: 'bd.causalStochastic', ref: fixture.multimodalBlockDiffusion.causalStochastic },
    ];

    for (const c of bdCases) {
      const res = generateWithPrefixCacheBlockDiff(model, {
        promptTokenIds: fixed.expandedTokenIds,
        inputsEmbeds: splicedEmbs,
        blockLength: c.ref.blockLength,
        maxNewTokens: c.ref.maxNewTokens,
        stepsPerBlock: c.ref.stepsPerBlock,
        confidenceThreshold: c.ref.confidenceThreshold,
        temperature: c.ref.temperature,
        causalContext: c.ref.causalContext,
        suppressMaskLogit: c.ref.suppressMaskLogit,
        ...(c.ref.explicitUniformVariates !== null
          ? { explicitUniformVariates: c.ref.explicitUniformVariates }
          : {}),
      });

      assertBitExactFiniteArray(res.generatedTokenIds, c.ref.generatedTokenIds, `${c.label}.generatedTokenIds`);
      expect(res.stoppedByEos).toBe(c.ref.stoppedByEos);
      expect(res.eosOffset).toBe(c.ref.eosOffset);
      expect(res.stopReason).toBe(c.ref.stopReason);
      expect(res.nfeCount).toBe(c.ref.nfeCount);
      expect(res.totalGuardInterventions).toBe(c.ref.totalGuardInterventions);
      expect(res.steps.length).toBe(c.ref.steps.length);

      for (let i = 0; i < c.ref.steps.length; i++) {
        const sAct = res.steps[i];
        const sRef = c.ref.steps[i];
        expect(sAct.blockIndex).toBe(sRef.blockIndex);
        expect(sAct.stepIndex).toBe(sRef.stepIndex);
        expect(sAct.blockStartPos).toBe(sRef.blockStartPos);
        expect(sAct.usedCausalPreSeed).toBe(sRef.usedCausalPreSeed);
        assertBitExactFiniteArray(sAct.blockTokensBefore, sRef.blockTokensBefore, `${c.label}.step${i}.before`);
        assertBitExactFiniteArray(sAct.blockTokensAfter, sRef.blockTokensAfter, `${c.label}.step${i}.after`);
        assertBitExactFiniteArray(sAct.unmaskedOffsets, sRef.unmaskedOffsets, `${c.label}.step${i}.unmasked`);
        assertBitExactFiniteArray(sAct.rawTop1TokenIds, sRef.rawTop1TokenIds, `${c.label}.step${i}.rawTop1`);
        assertBitExactFiniteArray(sAct.guardIntervenedOffsets, sRef.guardIntervenedOffsets, `${c.label}.step${i}.guard`);
        assertFiniteArrayParity(sAct.slotConfidences, sRef.slotConfidences, 5e-5, `${c.label}.step${i}.confs`);
        assertFiniteArrayParity(sAct.shiftedLogits, sRef.shiftedLogits, 5e-5, `${c.label}.step${i}.logits`);
        assertFiniteArrayParity(sAct.shiftedProbs, sRef.shiftedProbs, 5e-5, `${c.label}.step${i}.probs`);
      }
    }
  });

  it('Category 5: matches <LOC_0..16> Object Grounding (single-tag 16x16 & multi-tag 16x8) at 0-ULP IDs/boxes & <= 5e-5 confidences', () => {
    const sunlitCat = createUniScenePreset('sunlit_cat_16x16');
    const harbor = createUniScenePreset('harbor_panorama_16x8');

    const groundingCases = [
      { label: 'grounding.sunlitCatSingle', image: sunlitCat, cfg: fixture.objectGrounding.sunlitCatSingleTag },
      { label: 'grounding.harborMulti', image: harbor, cfg: fixture.objectGrounding.harborPanoramaMultiTag },
    ];

    for (const c of groundingCases) {
      const vis = runMiniVisionPipeline(c.image, visionWeights, 'mini_default');
      const fixed = fixOnePrompt(
        c.cfg.basePromptTokenIds,
        vis.projector.tokenGridRows,
        vis.projector.tokenGridCols,
      );
      assertBitExactFiniteArray(fixed.expandedTokenIds, c.cfg.expandedPromptTokenIds, `${c.label}.expanded`);
      const textEmbs = embedUniTextTokens(fixed.expandedTokenIds, model);
      const spliced = spliceVisualEmbeddingsIntoPrompt(
        fixed.expandedTokenIds,
        textEmbs,
        fixed.visualSlotIndices,
        vis.projector.visualEmbeddings,
      );

      const res = groundObjects(model, {
        promptTokenIds: fixed.expandedTokenIds,
        promptInputsEmbeds: spliced,
        imageWidth: c.cfg.result.imageWidth,
        imageHeight: c.cfg.result.imageHeight,
        tags: c.cfg.result.tags,
        steps: c.cfg.result.stepsRequested,
        enforceLocDomainGuard: true,
      });

      expect(res.directTokenDecodePreservesLoc).toBe(true);
      expect(res.skipSpecialTokensWouldStripCount).toBe(c.cfg.result.skipSpecialTokensWouldStripCount);
      expect(res.totalGuardInterventions).toBe(c.cfg.result.totalGuardInterventions);
      assertBitExactFiniteArray(res.assembledTemplateIds, c.cfg.result.assembledTemplateIds, `${c.label}.tpl`);
      assertBitExactFiniteArray(res.slotOffsetsInFullSequence, c.cfg.result.slotOffsetsInFullSequence, `${c.label}.slots`);
      expect(res.boxes.length).toBe(c.cfg.result.boxes.length);

      for (let b = 0; b < c.cfg.result.boxes.length; b++) {
        const bAct = res.boxes[b];
        const bRef = c.cfg.result.boxes[b];
        expect(bAct.tag).toBe(bRef.tag);
        assertBitExactFiniteArray(bAct.locTokenIds, bRef.locTokenIds, `${c.label}.box${b}.locIds`);
        expect(bAct.locTokenStrings).toEqual(bRef.locTokenStrings);
        for (const s of bAct.locTokenStrings) {
          expect(s.startsWith('<LOC_')).toBe(true);
        }
        assertBitExactFiniteArray(bAct.normalizedBins, bRef.normalizedBins, `${c.label}.box${b}.bins`);
        assertFiniteArrayParity(bAct.pixelBox, bRef.pixelBox, 1e-6, `${c.label}.box${b}.pixelBox`);
        assertBitExactFiniteArray(bAct.rawTop1Ids, bRef.rawTop1Ids, `${c.label}.box${b}.rawTop1`);
        expect(bAct.guardIntervenedCount).toBe(bRef.guardIntervenedCount);
      }

      expect(res.stepTraces.length).toBe(c.cfg.result.stepTraces.length);
      for (let s = 0; s < c.cfg.result.stepTraces.length; s++) {
        const sAct = res.stepTraces[s];
        const sRef = c.cfg.result.stepTraces[s];
        assertBitExactFiniteArray(sAct.slotTokenIdsBefore, sRef.slotTokenIdsBefore, `${c.label}.step${s}.before`);
        assertBitExactFiniteArray(sAct.slotTokenIdsAfter, sRef.slotTokenIdsAfter, `${c.label}.step${s}.after`);
        assertBitExactFiniteArray(sAct.unmaskedSlotIndices, sRef.unmaskedSlotIndices, `${c.label}.step${s}.unmasked`);
        assertBitExactFiniteArray(sAct.rawTop1TokenIds, sRef.rawTop1TokenIds, `${c.label}.step${s}.rawTop1`);
        assertBitExactFiniteArray(sAct.guardIntervenedSlotIndices, sRef.guardIntervenedSlotIndices, `${c.label}.step${s}.guard`);
        assertFiniteArrayParity(sAct.slotConfidences, sRef.slotConfidences, 5e-5, `${c.label}.step${s}.confs`);
      }
    }
  });
});

// ============================================================================
// PART 2: IMMUTABLE KV SLABS, D5 TELEMETRY & BROWSER GUARDS (U21, U22, U27, U30)
// ============================================================================

describe('UniKvCacheState — Immutability, CFG Batching (1/2/3), D5 Waste Telemetry & Capacity', () => {
  it('supports B_cfg in {1, 2, 3}, preserves slab sharing on branch/crop, and extracts exact row KVs', () => {
    const fwdA = runUniForwardPass(model, {
      tokenIds: [1, 10, 26],
      maskMode: 'causal',
      useCache: true,
      downstreamTarget: 'g',
      prefillMode: 'hf_faithful',
    });
    const fwdB = runUniForwardPass(model, {
      tokenIds: [1, 10, 4],
      maskMode: 'causal',
      useCache: true,
      downstreamTarget: 'g',
      prefillMode: 'hf_faithful',
    });
    const fwdC = runUniForwardPass(model, {
      genTokenIds: [12, 34, 56],
      maskMode: 'bidirectional',
      useCache: true,
      downstreamTarget: 'g',
      prefillMode: 'hf_faithful',
    });

    // B_cfg = 2 T2I hf_faithful: exact 16/34 wasted fraction!
    const empty2 = createEmptyUniKvCache({
      batchSize: 2,
      capacity: 16,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'g',
    });
    const cached2 = appendBatchKvsToCache(
      empty2,
      [fwdA.newKvs, fwdB.newKvs],
      [fwdA.routing, fwdB.routing],
    );
    expect(empty2.seqLen).toBe(0); // non-mutating
    expect(cached2.seqLen).toBe(3);
    expect(cached2.dualTowerHasGeneration).toBe(false);
    expect(cached2.wasteTelemetry.allocatedBytes).toBe(3 * 34 * 2 * 2 * 16 * 4);
    expect(cached2.wasteTelemetry.wastedSplitUBytes).toBe(3 * 16 * 2 * 2 * 16 * 4);
    expect(cached2.wasteTelemetry.wastedSplitULayerCount).toBe(16);
    expect(cached2.wasteTelemetry.wastedFraction).toBeCloseTo(16 / 34, 12);

    // Branching shares exact slab object references (zero duplication!)
    const branch2 = appendBatchKvsToCache(
      cached2,
      [fwdA.newKvs, fwdB.newKvs],
      [fwdA.routing, fwdB.routing],
    );
    expect(branch2.seqLen).toBe(6);
    for (let i = 0; i < 3; i++) {
      expect(branch2.slabs[i]).toBe(cached2.slabs[i]);
    }

    // Cropping shares exact slab references & does not mutate branch2
    const cropped = cropUniKvCache(branch2, 2);
    expect(cropped.seqLen).toBe(2);
    expect(branch2.seqLen).toBe(6);
    expect(cropped.slabs[0]).toBe(cached2.slabs[0]);
    expect(cropped.slabs[1]).toBe(cached2.slabs[1]);

    // Extracted row KVs bit-exact match original forward outputs
    const row1 = extractRowPastKvs(cached2, 1);
    expect(Array.from(row1.pastRouting)).toEqual([UNI_ROUTE_U, UNI_ROUTE_U, UNI_ROUTE_U]);
    assertBitExactFiniteArray(row1.pastKvs[0]!.k, fwdB.newKvs[0]!.k, 'kv.row1.L0.k');
    assertBitExactFiniteArray(row1.pastKvs[33]!.v, fwdB.newKvs[33]!.v, 'kv.row1.L33.v');

    // B_cfg = 3 Image Editing with mixed U/G rows
    const empty3 = createEmptyUniKvCache({
      batchSize: 3,
      capacity: 16,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'g',
    });
    const cached3 = appendBatchKvsToCache(
      empty3,
      [fwdA.newKvs, fwdB.newKvs, fwdC.newKvs],
      [fwdA.routing, fwdB.routing, fwdC.routing],
    );
    expect(cached3.dualTowerHasGeneration).toBe(true);
    // 2 of 3 rows are U (wasted on L18..33), 1 row is G (active on L18..33)
    expect(cached3.wasteTelemetry.wastedSplitUBytes).toBe(3 * 16 * 2 * 2 * 16 * 4);
    expect(cached3.wasteTelemetry.wastedFraction).toBeCloseTo((2 / 3) * (16 / 34), 12);
    expect(Array.from(extractRowPastKvs(cached3, 2).pastRouting)).toEqual([
      UNI_ROUTE_G,
      UNI_ROUTE_G,
      UNI_ROUTE_G,
    ]);
  });

  it('reports 0 wasted KV bytes for elastic_pruned T2I (with null L18..33) and for Understanding (downstreamTarget=u)', () => {
    const prunedFwd = runUniForwardPass(model, {
      tokenIds: [1, 10, 26, 27],
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'elastic_pruned',
      downstreamTarget: 'g',
    });
    const prunedCache = appendSingleRowKvsToCache(
      createEmptyUniKvCache({
        batchSize: 1,
        prefillMode: 'elastic_pruned',
        downstreamTarget: 'g',
      }),
      prunedFwd.newKvs,
      prunedFwd.routing,
    );
    expect(prunedCache.wasteTelemetry.allocatedBytes).toBe(4 * 18 * 2 * 1 * 16 * 4);
    expect(prunedCache.wasteTelemetry.wastedSplitUBytes).toBe(0);
    expect(prunedCache.wasteTelemetry.wastedSplitULayerCount).toBe(0);
    expect(prunedCache.wasteTelemetry.wastedFraction).toBe(0);

    const extractedPruned = extractRowPastKvs(prunedCache, 0);
    expect(extractedPruned.pastKvs[17]).not.toBeNull();
    expect(extractedPruned.pastKvs[18]).toBeNull();
    expect(extractedPruned.pastKvs[33]).toBeNull();

    // Understanding target ('u'): L18..33 U KVs are actively consumed -> 0 waste!
    const undFwd = runUniForwardPass(model, {
      tokenIds: [1, 10, 26, 27],
      maskMode: 'causal',
      useCache: true,
      prefillMode: 'hf_faithful',
      downstreamTarget: 'u',
    });
    const undCache = appendSingleRowKvsToCache(
      createEmptyUniKvCache({ batchSize: 1, prefillMode: 'hf_faithful', downstreamTarget: 'u' }),
      undFwd.newKvs,
      undFwd.routing,
    );
    expect(undCache.wasteTelemetry.wastedSplitUBytes).toBe(0);
    expect(undCache.wasteTelemetry.wastedSplitULayerCount).toBe(0);
    expect(undCache.wasteTelemetry.wastedFraction).toBe(0);

    // Direct telemetry check on empty cache
    const emptyTel = computeUniKvWasteTelemetry([], 2, 'hf_faithful', 'g');
    expect(emptyTel.allocatedBytes).toBe(0);
    expect(emptyTel.wastedFraction).toBe(0);
    expect(emptyTel.wastedSplitULayerCount).toBe(0);
  });

  it('throws RangeError on append beyond capacity and halts block-diffusion cleanly with capacity_exhausted', () => {
    const fwd = runUniForwardPass(model, {
      tokenIds: [1, 10, 26, 27],
      maskMode: 'causal',
      useCache: true,
      downstreamTarget: 'u',
    });
    const tiny = createEmptyUniKvCache({ batchSize: 1, capacity: 3 });
    expect(() => appendSingleRowKvsToCache(tiny, fwd.newKvs, fwd.routing)).toThrow(RangeError);

    // Prompt length 62 with capacity 64 and blockLength 4 -> halts immediately without shrinking B!
    const longPrompt = new Int32Array(62).fill(26);
    longPrompt[0] = 1;
    const res = generateWithPrefixCacheBlockDiff(model, {
      promptTokenIds: longPrompt,
      blockLength: 4,
      maxNewTokens: 4,
      temperature: 0,
    });
    expect(res.stopReason).toBe('capacity_exhausted');
    expect(res.generatedTokenIds.length).toBe(0);
    expect(res.finalCache.seqLen).toBe(62);
    expect(res.nfeCount).toBe(1); // prefill only
  });
});

describe('Browser Guards U21, U22, U27, U30 & AR Recovery (B = 1)', () => {
  it('U22: log-domain Gumbel sampling handles extreme logits (> 800) with zero NaN/Infinity', () => {
    const extremeLogits = new Float32Array(64).fill(-500);
    extremeLogits[25] = -Infinity; // U21 [MASK] suppressed
    extremeLogits[6] = 950.0; // exp(950) would overflow FP64!
    extremeLogits[7] = 949.5;

    const rng = createXoshiro128ss(12345);
    const stream = { nextOpenUnit: () => rng.nextUniformOpenUnit() };
    const picked = sampleLogDomainGumbelArgmax(extremeLogits, 0.5, stream);
    expect(picked === 6 || picked === 7).toBe(true);
    expect(picked).not.toBe(UNI_MINI_MASK_TOKEN_ID);
  });

  it('U21 & U30: records [MASK] guard intervention provenance, recovers AR at B=1, and crops KV cache on EOS', () => {
    // Construct a synthetic model clone whose diffusionHead strongly prefers [MASK] (id 25)
    // and secondarily prefers EOS (id 11) so we verify U21 provenance + U30 EOS KV crop deterministically!
    const tweakedHead = new Float32Array(model.diffusionHead);
    const d = model.config.hiddenSize;
    // Zero out all rows, then give row 25 ([MASK]) huge norm along finalNorm direction and row 11 (EOS) second highest
    tweakedHead.fill(0);
    const probeFwd = runUniForwardPass(model, {
      tokenIds: [1, 10, 26],
      maskMode: 'causal',
      useCache: false,
      downstreamTarget: 'u',
    });
    const lastNorm = probeFwd.finalNorm!.subarray(2 * d, 3 * d);
    for (let c = 0; c < d; c++) {
      tweakedHead[UNI_MINI_MASK_TOKEN_ID * d + c] = lastNorm[c] * 20.0;
      tweakedHead[UNI_MINI_EOS_TOKEN_ID * d + c] = lastNorm[c] * 10.0;
      tweakedHead[6 * d + c] = lastNorm[c] * 2.0; // 'cat'
    }
    const tweakedModel: UniCoreModelParameters = {
      ...model,
      diffusionHead: tweakedHead,
    };

    const res = generateWithPrefixCacheBlockDiff(tweakedModel, {
      promptTokenIds: [1, 10, 26],
      blockLength: 4,
      maxNewTokens: 8,
      temperature: 0,
      causalContext: true,
      suppressMaskLogit: true,
    });

    // U21 intervened on slot 0 where rawTop1 was [MASK] (25) -> selected EOS (11) instead!
    expect(res.totalGuardInterventions).toBeGreaterThanOrEqual(1);
    expect(res.steps[0].rawTop1TokenIds[0]).toBe(UNI_MINI_MASK_TOKEN_ID);
    expect(res.steps[0].guardIntervenedOffsets).toContain(0);
    expect(res.generatedTokenIds[0]).toBe(UNI_MINI_EOS_TOKEN_ID);
    // U30: cropped cleanly through EOS (len 1), KV cache cropped to promptLen + 1 = 4!
    expect(res.stoppedByEos).toBe(true);
    expect(res.stopReason).toBe('eos');
    expect(res.eosOffset).toBe(0);
    expect(res.generatedTokenIds.length).toBe(1);
    expect(res.finalCache.seqLen).toBe(4);

    // B = 1 with causalContext = true recovers pure AR (1 prefill + 1 commit per token, 0 bidirectional denoise!)
    const arRes = generateWithPrefixCacheBlockDiff(model, {
      promptTokenIds: [1, 10, 26],
      blockLength: 1,
      maxNewTokens: 3,
      temperature: 0,
      causalContext: true,
    });
    expect(arRes.nfeCount).toBe(1 + arRes.generatedTokenIds.length);
    for (const st of arRes.steps) {
      expect(st.usedCausalPreSeed).toBe(true);
    }
  });

  it('U27 & §4.2 pixel_box: verifies exact non-square un-letterboxing and non-stripped <LOC_*> tokens', () => {
    // 16x8 panorama letterboxed to 16x16 (padY = 4, padX = 0):
    // bins [2, 4, 14, 12] -> exact source box [2, 0, 14, 8]
    expect(locBinsToPixelBox([2, 4, 14, 12], 16, 8, 16)).toEqual([2, 0, 14, 8]);
    // Out-of-content border bins clamp cleanly to [0, H]
    expect(locBinsToPixelBox([0, 0, 16, 16], 16, 8, 16)).toEqual([0, 0, 16, 8]);
    // Square 16x16 identity mapping
    expect(locBinsToPixelBox([3, 5, 11, 13], 16, 16, 16)).toEqual([3, 5, 11, 13]);

    const g = groundObjects(model, {
      promptTokenIds: [1, 10, 26, 27],
      imageWidth: 16,
      imageHeight: 8,
      tags: ['house', 'water'],
      steps: 2,
      enforceLocDomainGuard: true,
    });
    expect(g.directTokenDecodePreservesLoc).toBe(true);
    // 2 tags * (1 <box_p> + 1 </box_p> + 1 <LOC_BEGIN> + 4 <LOC_*> + 1 <LOC_END>) = 16 strip-vulnerable tokens
    expect(g.skipSpecialTokensWouldStripCount).toBe(16);
    expect(g.skipSpecialTokensHazardNote).toContain('U27');
    for (const box of g.boxes) {
      for (const id of box.locTokenIds) {
        expect(id).toBeGreaterThanOrEqual(UNI_MINI_LOC_START_ID);
        expect(id).toBeLessThanOrEqual(UNI_MINI_LOC_END_ID);
      }
      for (const str of box.locTokenStrings) {
        expect(str).toMatch(/^<LOC_\d+>$/);
      }
    }
  });
});
