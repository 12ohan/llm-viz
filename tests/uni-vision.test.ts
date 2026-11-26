import { describe, expect, it } from 'vitest';
import {
  UNI_CLIP_MEAN,
  UNI_CLIP_STD,
  UNI_GAIN_PRESETS,
  UNI_IMAGE_NEWLINE_TENSOR_NAME,
  UNI_MINI_CONFIG,
  UNI_MINI_IMAGE_NEWLINE_FLOAT_COUNT,
  UNI_MINI_IMAGE_NEWLINE_TENSOR_COUNT,
  UNI_MINI_NEWLINE_TOKEN_ID,
  UNI_MINI_PROJECTOR_FLOAT_COUNT,
  UNI_MINI_PROJECTOR_TENSOR_COUNT,
  UNI_MINI_UND_PAD_TOKEN_ID,
  UNI_MINI_VISION_SUBSYSTEM_FLOAT_COUNT,
  UNI_MINI_VISION_SUBSYSTEM_TENSOR_COUNT,
  UNI_MINI_VISION_TOWER_FLOAT_COUNT,
  UNI_MINI_VISION_TOWER_TENSOR_COUNT,
  UNI_MINI_VIT_EMBED_DIM,
  UNI_MINI_VIT_NUM_HEADS,
  UNI_MINI_VIT_PATCH_DIM,
  UNI_MINI_VIT_PREFIX_TOKENS,
  UNI_MULTIMODAL_SEED,
  type UniScenePresetId,
} from '../src/uni/types';
import {
  UNI_LETTERBOX_PAD_VALUE,
  UNI_SCENE_PRESET_IDS,
  clipNormalizeImage,
  collectUniVisionWeights,
  createUniScenePreset,
  createUniVisionSubsystemWeights,
  fixOnePrompt,
  interpolatePosEmbed2d,
  padToSquareAndResize,
  runMiniRadioHProjector,
  runMiniRadioVisionTower,
  runMiniVisionPipeline,
  spliceVisualEmbeddingsIntoPrompt,
} from '../src/uni/vision';

describe('UNI2 Slice 2A: Procedural RGB scenes & integer-centered letterbox (U18, U30)', () => {
  it('generates all 3 deterministic presets finite in [0, 1] with groundTruthBoxes', () => {
    const expectedDims: Record<UniScenePresetId, [number, number]> = {
      sunlit_cat_16x16: [16, 16],
      harbor_panorama_16x8: [16, 8],
      geometric_blocks_16x16: [16, 16],
    };

    for (const presetId of UNI_SCENE_PRESET_IDS) {
      const img = createUniScenePreset(presetId);
      const [expW, expH] = expectedDims[presetId];
      expect(img.presetId).toBe(presetId);
      expect(img.width).toBe(expW);
      expect(img.height).toBe(expH);
      expect(img.data.length).toBe(3 * expW * expH);
      expect(img.groundTruthBoxes && img.groundTruthBoxes.length).toBeGreaterThanOrEqual(2);

      let minVal = Infinity;
      let maxVal = -Infinity;
      for (let i = 0; i < img.data.length; i++) {
        const v = img.data[i];
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
      // Ensure non-trivial visual contrast
      expect(maxVal - minVal).toBeGreaterThan(0.5);

      // Reproducibility
      const img2 = createUniScenePreset(presetId);
      expect(Array.from(img2.data)).toEqual(Array.from(img.data));
    }

    const catTags = createUniScenePreset('sunlit_cat_16x16').groundTruthBoxes!.map((b) => b.tag);
    expect(catTags).toEqual(['cat', 'sun']);
    const harborTags = createUniScenePreset('harbor_panorama_16x8').groundTruthBoxes!.map(
      (b) => b.tag,
    );
    expect(harborTags).toEqual(['house', 'water']);
  });

  it('padToSquareAndResize is identity for 16x16 -> 16', () => {
    const cat = createUniScenePreset('sunlit_cat_16x16');
    const res = padToSquareAndResize(cat, 16);

    expect(res.info).toEqual({
      sourceWidth: 16,
      sourceHeight: 16,
      targetSize: 16,
      scale: 1,
      resizedWidth: 16,
      resizedHeight: 16,
      padLeft: 0,
      padTop: 0,
      padRight: 0,
      padBottom: 0,
      exactIntegerCentered: true,
    });
    expect(res.image.width).toBe(16);
    expect(res.image.height).toBe(16);
    expect(Array.from(res.image.data)).toEqual(Array.from(cat.data));
  });

  it('padToSquareAndResize is exact integer-centered for 16x8 -> 16 (padTop=4, padBottom=4, 0 drift)', () => {
    const harbor = createUniScenePreset('harbor_panorama_16x8');
    const res = padToSquareAndResize(harbor, 16);

    expect(res.info).toEqual({
      sourceWidth: 16,
      sourceHeight: 8,
      targetSize: 16,
      scale: 1,
      resizedWidth: 16,
      resizedHeight: 8,
      padLeft: 0,
      padTop: 4,
      padRight: 0,
      padBottom: 4,
      exactIntegerCentered: true,
    });

    const dstHW = 16 * 16;
    const srcHW = 16 * 8;
    for (let c = 0; c < 3; c++) {
      // Rows 0..3 and 12..15 must be exact black letterbox padding (0.0)
      for (let y = 0; y < 4; y++) {
        for (let x = 0; x < 16; x++) {
          expect(res.image.data[c * dstHW + y * 16 + x]).toBe(UNI_LETTERBOX_PAD_VALUE);
        }
      }
      for (let y = 12; y < 16; y++) {
        for (let x = 0; x < 16; x++) {
          expect(res.image.data[c * dstHW + y * 16 + x]).toBe(UNI_LETTERBOX_PAD_VALUE);
        }
      }
      // Rows 4..11 must be exact bit-for-bit copies of source rows 0..7
      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 16; x++) {
          expect(res.image.data[c * dstHW + (y + 4) * 16 + x]).toBe(
            harbor.data[c * srcHW + y * 16 + x],
          );
        }
      }
    }

    // Ground truth boxes shifted vertically by +4
    expect(res.image.groundTruthBoxes![0].pixelBox).toEqual([11, 5, 15, 10]);
    expect(res.image.groundTruthBoxes![1].pixelBox).toEqual([0, 8, 16, 12]);
  });

  it('padToSquareAndResize handles arbitrary non-native dimensions via bilinear resize + centered pad', () => {
    const srcW = 8;
    const srcH = 4;
    const data = new Float32Array(3 * srcW * srcH).fill(0.5);
    const res = padToSquareAndResize({ width: srcW, height: srcH, data }, 16);

    expect(res.info.scale).toBe(2);
    expect(res.info.resizedWidth).toBe(16);
    expect(res.info.resizedHeight).toBe(8);
    expect(res.info.padLeft).toBe(0);
    expect(res.info.padTop).toBe(4);
    expect(res.info.padRight).toBe(0);
    expect(res.info.padBottom).toBe(4);
    expect(res.info.exactIntegerCentered).toBe(true);

    // Center region should remain constant 0.5, border 0.0
    for (let y = 4; y < 12; y++) {
      for (let x = 0; x < 16; x++) {
        expect(res.image.data[y * 16 + x]).toBeCloseTo(0.5, 6);
      }
    }
    expect(res.image.data[0]).toBe(0.0);
    expect(res.image.data[15 * 16 + 15]).toBe(0.0);
  });
});

describe('UNI2 Slice 2A: Deterministic 31-tensor vision subsystem weights (8,296 floats)', () => {
  it('generates exact 31 canonical tensors totaling 8,296 floats with pinned order and shapes', () => {
    const weights = createUniVisionSubsystemWeights();
    expect(weights.seed).toBe(UNI_MULTIMODAL_SEED);

    const entries = collectUniVisionWeights(weights);
    expect(entries.length).toBe(UNI_MINI_VISION_SUBSYSTEM_TENSOR_COUNT);
    expect(entries.length).toBe(31);

    const towerEntries = entries.filter((e) => e.subsystem === 'vision_tower');
    const projEntries = entries.filter((e) => e.subsystem === 'mm_projector');
    const newlineEntries = entries.filter((e) => e.subsystem === 'image_newline');

    expect(towerEntries.length).toBe(UNI_MINI_VISION_TOWER_TENSOR_COUNT);
    expect(projEntries.length).toBe(UNI_MINI_PROJECTOR_TENSOR_COUNT);
    expect(newlineEntries.length).toBe(UNI_MINI_IMAGE_NEWLINE_TENSOR_COUNT);

    const sumFloats = (list: typeof entries) =>
      list.reduce((acc, e) => acc + e.data.length, 0);

    expect(sumFloats(towerEntries)).toBe(UNI_MINI_VISION_TOWER_FLOAT_COUNT);
    expect(sumFloats(projEntries)).toBe(UNI_MINI_PROJECTOR_FLOAT_COUNT);
    expect(sumFloats(newlineEntries)).toBe(UNI_MINI_IMAGE_NEWLINE_FLOAT_COUNT);
    expect(sumFloats(entries)).toBe(UNI_MINI_VISION_SUBSYSTEM_FLOAT_COUNT);

    // Verify exact canonical names and order
    expect(entries[0].name).toBe('encoder.vision_tower.patch_generator.embedder.weight');
    expect(entries[0].shape).toEqual([16, 48]);
    expect(entries[1].name).toBe('encoder.vision_tower.pos_embed');
    expect(entries[1].shape).toEqual([16, 16]);
    expect(entries[2].name).toBe('encoder.vision_tower.cls_token');
    expect(entries[2].shape).toEqual([4, 16]);

    for (let b = 0; b < 2; b++) {
      const base = 3 + b * 12;
      expect(entries[base + 0].name).toBe(`encoder.vision_tower.blocks.${b}.norm1.weight`);
      expect(entries[base + 1].name).toBe(`encoder.vision_tower.blocks.${b}.norm1.bias`);
      expect(entries[base + 2].name).toBe(`encoder.vision_tower.blocks.${b}.attn.qkv.weight`);
      expect(entries[base + 3].name).toBe(`encoder.vision_tower.blocks.${b}.attn.qkv.bias`);
      expect(entries[base + 4].name).toBe(`encoder.vision_tower.blocks.${b}.attn.proj.weight`);
      expect(entries[base + 5].name).toBe(`encoder.vision_tower.blocks.${b}.attn.proj.bias`);
      expect(entries[base + 6].name).toBe(`encoder.vision_tower.blocks.${b}.norm2.weight`);
      expect(entries[base + 7].name).toBe(`encoder.vision_tower.blocks.${b}.norm2.bias`);
      expect(entries[base + 8].name).toBe(`encoder.vision_tower.blocks.${b}.mlp.fc1.weight`);
      expect(entries[base + 9].name).toBe(`encoder.vision_tower.blocks.${b}.mlp.fc1.bias`);
      expect(entries[base + 10].name).toBe(`encoder.vision_tower.blocks.${b}.mlp.fc2.weight`);
      expect(entries[base + 11].name).toBe(`encoder.vision_tower.blocks.${b}.mlp.fc2.bias`);
    }

    expect(entries[27].name).toBe('encoder.mm_projector.norm.weight');
    expect(entries[27].shape).toEqual([64]);
    expect(entries[28].name).toBe('encoder.mm_projector.fc1.weight');
    expect(entries[28].shape).toEqual([8, 64]);
    expect(entries[29].name).toBe('encoder.mm_projector.fc2.weight');
    expect(entries[29].shape).toEqual([8, 8]);

    expect(entries[30].name).toBe(UNI_IMAGE_NEWLINE_TENSOR_NAME);
    expect(entries[30].shape).toEqual([8]);
    expect(entries[30].unusedInForward).toBe(true);

    // Every tensor shape matches data.length, finite values, norm weights >= 0.5
    for (const entry of entries) {
      const prod = entry.shape.reduce((a, b) => a * b, 1);
      expect(entry.data.length).toBe(prod);
      for (let i = 0; i < entry.data.length; i++) {
        expect(Number.isFinite(entry.data[i])).toBe(true);
      }
      if (entry.name.endsWith('norm1.weight') || entry.name.endsWith('norm2.weight') || entry.name.endsWith('norm.weight')) {
        for (let i = 0; i < entry.data.length; i++) {
          expect(entry.data[i]).toBeGreaterThanOrEqual(0.5);
        }
      }
    }
  });

  it('is strictly deterministic for identical seeds and distinct across seeds', () => {
    const a = collectUniVisionWeights(createUniVisionSubsystemWeights(UNI_MULTIMODAL_SEED));
    const b = collectUniVisionWeights(createUniVisionSubsystemWeights(UNI_MULTIMODAL_SEED));
    const c = collectUniVisionWeights(createUniVisionSubsystemWeights(0x12345678));

    for (let i = 0; i < a.length; i++) {
      expect(Array.from(a[i].data)).toEqual(Array.from(b[i].data));
      expect(Array.from(a[i].data)).not.toEqual(Array.from(c[i].data));
    }
  });
});

describe('UNI2 Slice 2A: CLIP norm, bilinear CPE pos_embed & MiniRadioVisionTower', () => {
  it('clipNormalizeImage matches (x - mean) / std in Float32', () => {
    const img = createUniScenePreset('sunlit_cat_16x16');
    const norm = clipNormalizeImage(img);
    expect(norm.length).toBe(img.data.length);
    const hw = img.width * img.height;
    for (let c = 0; c < 3; c++) {
      for (let p = 0; p < hw; p++) {
        const expected = Math.fround((img.data[c * hw + p] - UNI_CLIP_MEAN[c]) / UNI_CLIP_STD[c]);
        expect(norm[c * hw + p]).toBe(expected);
      }
    }
  });

  it('interpolatePosEmbed2d is exact identity for 4x4 -> 4x4 and exact average-pair bilinear for 4x4 -> 2x4', () => {
    const base = new Float32Array(16 * 2);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const idx = (r * 4 + c) * 2;
        base[idx] = r * 10 + c;
        base[idx + 1] = -(r * 10 + c);
      }
    }

    const same = interpolatePosEmbed2d(base, 4, 4, 4, 4, 2);
    expect(same).not.toBe(base);
    expect(Array.from(same)).toEqual(Array.from(base));

    // PyTorch align_corners=False for 4->2: dst 0 -> src 0.5 (avg of row 0 & 1), dst 1 -> src 2.5 (avg of row 2 & 3)
    const down = interpolatePosEmbed2d(base, 4, 4, 2, 4, 2);
    expect(down.length).toBe(8 * 2);
    for (let c = 0; c < 4; c++) {
      expect(down[(0 * 4 + c) * 2]).toBeCloseTo(0.5 * (0 + 10) + c, 6);
      expect(down[(1 * 4 + c) * 2]).toBeCloseTo(0.5 * (20 + 30) + c, 6);
    }
  });

  it('runs MiniRadioVisionTower on 16x16 (4x4 patches -> 20 ViT tokens -> 16 after [:, 4:])', () => {
    const weights = createUniVisionSubsystemWeights();
    const cat = createUniScenePreset('sunlit_cat_16x16');
    const trace = runMiniRadioVisionTower(cat, weights.visionTower);

    expect(trace.patchGridRows).toBe(4);
    expect(trace.patchGridCols).toBe(4);
    expect(trace.numPatches).toBe(16);
    expect(trace.numPrefixTokens).toBe(UNI_MINI_VIT_PREFIX_TOKENS);
    expect(trace.rawPatches.length).toBe(16 * UNI_MINI_VIT_PATCH_DIM);
    expect(trace.interpolatedPosEmbed.length).toBe(16 * UNI_MINI_VIT_EMBED_DIM);
    expect(trace.vitInputTokens.length).toBe(20 * UNI_MINI_VIT_EMBED_DIM);
    expect(trace.blockTraces.length).toBe(2);
    expect(trace.patchFeatures.length).toBe(16 * UNI_MINI_VIT_EMBED_DIM);

    // Verify prefix tokens at start of vitInputTokens match cls_token exactly
    expect(Array.from(trace.vitInputTokens.subarray(0, 4 * 16))).toEqual(
      Array.from(weights.visionTower.clsToken),
    );

    // Verify bidirectional attention probabilities per head sum to 1 and are finite
    for (const blk of trace.blockTraces) {
      expect(blk.seqLen).toBe(20);
      expect(blk.attnProbs.length).toBe(UNI_MINI_VIT_NUM_HEADS * 20 * 20);
      for (let h = 0; h < UNI_MINI_VIT_NUM_HEADS; h++) {
        for (let q = 0; q < 20; q++) {
          let rowSum = 0;
          for (let k = 0; k < 20; k++) {
            const p = blk.attnProbs[(h * 20 + q) * 20 + k];
            expect(p).toBeGreaterThan(0);
            expect(p).toBeLessThanOrEqual(1);
            rowSum += p;
          }
          expect(rowSum).toBeCloseTo(1.0, 5);
        }
      }
    }

    // Verify [:, 4:] prefix strip matches last 16 tokens of block 1 output
    const lastBlockOut = trace.blockTraces[1].output;
    expect(Array.from(trace.patchFeatures)).toEqual(Array.from(lastBlockOut.subarray(4 * 16)));
  });

  it('runs MiniRadioVisionTower on 16x8 panorama (2x4 patches -> 12 ViT tokens -> 8 after [:, 4:])', () => {
    const weights = createUniVisionSubsystemWeights();
    const harbor = createUniScenePreset('harbor_panorama_16x8');
    const trace = runMiniRadioVisionTower(harbor, weights.visionTower);

    expect(trace.patchGridRows).toBe(2);
    expect(trace.patchGridCols).toBe(4);
    expect(trace.numPatches).toBe(8);
    expect(trace.numPrefixTokens).toBe(4);
    expect(trace.rawPatches.length).toBe(8 * 48);
    expect(trace.interpolatedPosEmbed.length).toBe(8 * 16);
    expect(trace.vitInputTokens.length).toBe(12 * 16);
    expect(trace.blockTraces[0].seqLen).toBe(12);
    expect(trace.blockTraces[1].seqLen).toBe(12);
    expect(trace.patchFeatures.length).toBe(8 * 16);

    for (let i = 0; i < trace.patchFeatures.length; i++) {
      expect(Number.isFinite(trace.patchFeatures[i])).toBe(true);
    }
  });

  it('rejects images whose height or width is not a multiple of 8', () => {
    const weights = createUniVisionSubsystemWeights();
    expect(() =>
      runMiniRadioVisionTower(
        { width: 12, height: 16, data: new Float32Array(3 * 12 * 16) },
        weights.visionTower,
      ),
    ).toThrow(RangeError);
  });
});

describe('UNI2 Slice 2A: MiniRadioHProjector (D3 gain presets) & dead image_newline (U8)', () => {
  it('performs exact 2x2 ps_version=v2 gather, SquaredReLU, and exact 0.01/0.25 = 0.04 gain ratio (D3)', () => {
    const weights = createUniVisionSubsystemWeights();
    const cat = createUniScenePreset('sunlit_cat_16x16');
    const vision = runMiniRadioVisionTower(cat, weights.visionTower);

    const miniTrace = runMiniRadioHProjector(
      vision.patchFeatures,
      vision.patchGridRows,
      vision.patchGridCols,
      weights.projector,
      'mini_default',
    );
    const prodTrace = runMiniRadioHProjector(
      vision.patchFeatures,
      vision.patchGridRows,
      vision.patchGridCols,
      weights.projector,
      'prod_exact',
    );

    expect(miniTrace.tokenGridRows).toBe(2);
    expect(miniTrace.tokenGridCols).toBe(2);
    expect(miniTrace.numVisualTokens).toBe(4);
    expect(miniTrace.radioMultiply).toBe(UNI_GAIN_PRESETS.mini_default.radioProjectorMultiply);
    expect(prodTrace.radioMultiply).toBe(UNI_GAIN_PRESETS.prod_exact.radioProjectorMultiply);

    // Check exact 2x2 row-major gather for token (br=1, bc=1): patches (2,2), (2,3), (3,2), (3,3)
    const tok3 = miniTrace.gathered2x2.subarray(3 * 64, 4 * 64);
    const p22 = vision.patchFeatures.subarray((2 * 4 + 2) * 16, (2 * 4 + 3) * 16);
    const p23 = vision.patchFeatures.subarray((2 * 4 + 3) * 16, (2 * 4 + 4) * 16);
    const p32 = vision.patchFeatures.subarray((3 * 4 + 2) * 16, (3 * 4 + 3) * 16);
    const p33 = vision.patchFeatures.subarray((3 * 4 + 3) * 16, (3 * 4 + 4) * 16);
    expect(Array.from(tok3.subarray(0, 16))).toEqual(Array.from(p22));
    expect(Array.from(tok3.subarray(16, 32))).toEqual(Array.from(p23));
    expect(Array.from(tok3.subarray(32, 48))).toEqual(Array.from(p32));
    expect(Array.from(tok3.subarray(48, 64))).toEqual(Array.from(p33));

    // Verify SquaredReLU exact formula: max(0, x)^2
    for (let i = 0; i < miniTrace.fc1Out.length; i++) {
      const x = miniTrace.fc1Out[i];
      const expected = x > 0 ? Math.fround(x * x) : 0;
      expect(miniTrace.squaredReluOut[i]).toBe(expected);
      expect(miniTrace.squaredReluOut[i]).toBeGreaterThanOrEqual(0);
    }

    // Verify pre-gain fc2Out is bit-identical across presets and gain ratio is exact 0.01 / 0.25 = 0.04
    expect(Array.from(miniTrace.fc2Out)).toEqual(Array.from(prodTrace.fc2Out));
    let nonZeroChecked = 0;
    for (let i = 0; i < miniTrace.visualEmbeddings.length; i++) {
      const m = miniTrace.visualEmbeddings[i];
      const p = prodTrace.visualEmbeddings[i];
      if (Math.abs(m) > 1e-6) {
        expect(p / m).toBeCloseTo(0.04, 5);
        nonZeroChecked++;
      }
    }
    expect(nonZeroChecked).toBeGreaterThan(0);
  });

  it('proves perturbing weights.imageNewline by +1000 causes 0-ULP change in runMiniVisionPipeline (U8)', () => {
    const baseWeights = createUniVisionSubsystemWeights();
    const perturbedNewline = new Float32Array(baseWeights.imageNewline.length);
    for (let i = 0; i < perturbedNewline.length; i++) {
      perturbedNewline[i] = baseWeights.imageNewline[i] + 1000.0;
    }
    const perturbedWeights = {
      ...baseWeights,
      imageNewline: perturbedNewline,
    };

    for (const presetId of ['sunlit_cat_16x16', 'harbor_panorama_16x8'] as const) {
      const img = createUniScenePreset(presetId);
      const outA = runMiniVisionPipeline(img, baseWeights, 'mini_default');
      const outB = runMiniVisionPipeline(img, perturbedWeights, 'mini_default');

      expect(outA.imageNewlineUnused).toBe(true);
      expect(outB.imageNewlineUnused).toBe(true);

      // Exact 0-ULP bitwise Float32 equality via Int32Array view
      const bitsA = new Int32Array(
        outA.projector.visualEmbeddings.buffer,
        outA.projector.visualEmbeddings.byteOffset,
        outA.projector.visualEmbeddings.length,
      );
      const bitsB = new Int32Array(
        outB.projector.visualEmbeddings.buffer,
        outB.projector.visualEmbeddings.byteOffset,
        outB.projector.visualEmbeddings.length,
      );
      expect(Array.from(bitsA)).toEqual(Array.from(bitsB));
    }
  });
});

describe('UNI2 Slice 2A: fixOnePrompt & spliceVisualEmbeddingsIntoPrompt', () => {
  it('expands 2x2 to [23, 23, 24, 23, 23] and 1x2 to [23, 23] without dead image_newline', () => {
    const prompt = [1, 10, UNI_MINI_UND_PAD_TOKEN_ID, 26, 27, 11];

    const fixed2x2 = fixOnePrompt(prompt, 2, 2);
    expect(Array.from(fixed2x2.originalTokenIds)).toEqual(prompt);
    expect(Array.from(fixed2x2.expandedTokenIds)).toEqual([
      1,
      10,
      UNI_MINI_UND_PAD_TOKEN_ID,
      UNI_MINI_UND_PAD_TOKEN_ID,
      UNI_MINI_NEWLINE_TOKEN_ID,
      UNI_MINI_UND_PAD_TOKEN_ID,
      UNI_MINI_UND_PAD_TOKEN_ID,
      26,
      27,
      11,
    ]);
    expect(Array.from(fixed2x2.visualSlotIndices)).toEqual([2, 3, 5, 6]);
    expect(Array.from(fixed2x2.newlineIndices)).toEqual([4]);
    expect(fixed2x2.tokenGridRows).toBe(2);
    expect(fixed2x2.tokenGridCols).toBe(2);
    expect(fixed2x2.usedDeadImageNewline).toBe(false);

    const fixed1x2 = fixOnePrompt(prompt, 1, 2);
    expect(Array.from(fixed1x2.expandedTokenIds)).toEqual([
      1,
      10,
      UNI_MINI_UND_PAD_TOKEN_ID,
      UNI_MINI_UND_PAD_TOKEN_ID,
      26,
      27,
      11,
    ]);
    expect(Array.from(fixed1x2.visualSlotIndices)).toEqual([2, 3]);
    expect(Array.from(fixed1x2.newlineIndices)).toEqual([]);
    expect(fixed1x2.usedDeadImageNewline).toBe(false);
  });

  it('splices projected visual embeddings into expanded prompt non-mutatingly', () => {
    const weights = createUniVisionSubsystemWeights();
    const cat = createUniScenePreset('sunlit_cat_16x16');
    const pipe = runMiniVisionPipeline(cat, weights, 'mini_default');

    const prompt = [1, 10, UNI_MINI_UND_PAD_TOKEN_ID, 26, 27, 11];
    const fixed = fixOnePrompt(
      prompt,
      pipe.projector.tokenGridRows,
      pipe.projector.tokenGridCols,
    );

    const d = UNI_MINI_CONFIG.hiddenSize; // 8
    const seqLen = fixed.expandedTokenIds.length;
    const textEmb = new Float32Array(seqLen * d);
    for (let i = 0; i < textEmb.length; i++) {
      textEmb[i] = Math.fround(100 + i * 0.25);
    }
    const textEmbSnapshot = new Float32Array(textEmb);

    const spliced = spliceVisualEmbeddingsIntoPrompt(
      fixed.expandedTokenIds,
      textEmb,
      fixed.visualSlotIndices,
      pipe.projector.visualEmbeddings,
      d,
    );

    // Input textEmbeddings must not be mutated
    expect(Array.from(textEmb)).toEqual(Array.from(textEmbSnapshot));
    expect(spliced).not.toBe(textEmb);

    const visSlotSet = new Set(Array.from(fixed.visualSlotIndices));
    for (let t = 0; t < seqLen; t++) {
      const row = Array.from(spliced.subarray(t * d, (t + 1) * d));
      if (visSlotSet.has(t)) {
        const vIdx = Array.from(fixed.visualSlotIndices).indexOf(t);
        const expVis = Array.from(
          pipe.projector.visualEmbeddings.subarray(vIdx * d, (vIdx + 1) * d),
        );
        expect(row).toEqual(expVis);
      } else {
        const expText = Array.from(textEmbSnapshot.subarray(t * d, (t + 1) * d));
        expect(row).toEqual(expText);
      }
    }
  });

  it('throws on invalid visualSlotIndices pointing to non-<SPECIAL_23> tokens', () => {
    const tokens = [1, 10, 23, 24, 23];
    const textEmb = new Float32Array(tokens.length * 8);
    const visEmb = new Float32Array(2 * 8);
    expect(() =>
      spliceVisualEmbeddingsIntoPrompt(tokens, textEmb, [2, 3], visEmb, 8),
    ).toThrow(RangeError);
  });
});
