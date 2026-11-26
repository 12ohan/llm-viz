import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  addGenerationRowMarkers,
  applySourcePaddingEmbeddingLock,
  applySourcePaddingVqLock,
  buildFixedPrefixTokenIds,
  collectUniCodecWeights,
  compressPackedImageEmbeddings,
  computeSourcePaddingVqMask,
  createUniPackedCodecWeights,
  cropPaddedImageToSource,
  embedFixedPrefixTokens,
  expandPackedImageHiddenStates,
  stripGenerationRowMarkers,
} from '../src/uni/codec';
import { compute2dSinCosPosEmbed } from '../src/uni/rope';
import {
  UNI_GAIN_PRESETS,
  UNI_MINI_CODEC_FLOAT_COUNT,
  UNI_MINI_CODEC_TENSOR_COUNT,
  UNI_MINI_CONFIG,
  UNI_MINI_GEN_FIXED_PREFIX_COUNT,
  UNI_MINI_GEN_IMAGE_END_ID,
  UNI_MINI_GEN_ROW_MARKER_ID,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_MINI_VQVAE_CODEBOOK_FLOAT_COUNT,
  UNI_MINI_VQVAE_CODEBOOK_TENSOR_COUNT,
  UNI_MINI_VQVAE_DECODER_FLOAT_COUNT,
  UNI_MINI_VQVAE_DECODER_TENSOR_COUNT,
  UNI_MINI_VQVAE_ENCODER_FLOAT_COUNT,
  UNI_MINI_VQVAE_ENCODER_TENSOR_COUNT,
  UNI_MINI_VQVAE_TOTAL_FLOAT_COUNT,
  UNI_MINI_VQVAE_TOTAL_TENSOR_COUNT,
  UNI_MULTIMODAL_SEED,
  type UniLetterboxInfo,
  type UniRgbImage,
} from '../src/uni/types';
import {
  collectUniVqVaeWeights,
  conv2dChw,
  createUniEmu3VqWeights,
  decodeMiniEmu3Vq,
  encodeMiniEmu3Vq,
  groupNormChw,
  quantizeIbqDotProduct,
  runEmu3AttnBlock,
  runEmu3ResBlock,
} from '../src/uni/vqvae';
import {
  assertBitExactFiniteArray,
  assertFiniteArrayParity,
} from './helpers/parityAssert';

function makeSyntheticRgbImage(
  width: number,
  height: number,
  seed = 1337,
): UniRgbImage {
  const data = new Float32Array(3 * height * width);
  let s = seed >>> 0;
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        const jitter = (s >>> 24) / 255;
        const grad = (x / Math.max(1, width - 1)) * 0.5 + (y / Math.max(1, height - 1)) * 0.3;
        const val = Math.min(1, Math.max(0, 0.1 + 0.6 * grad + 0.2 * jitter + 0.05 * c));
        data[(c * height + y) * width + x] = Math.fround(val);
      }
    }
  }
  return { width, height, data };
}

function makeDeterministicF32(length: number, seed: number, scale = 0.25): Float32Array {
  const out = new Float32Array(length);
  let s = seed >>> 0;
  for (let i = 0; i < length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const u = (s / 4294967296) * 2 - 1;
    out[i] = Math.fround(u * scale);
  }
  return out;
}

describe('UNI2 Slice 2B — Mini Emu3 VQ-VAE & PackedLinearImageCodec', () => {
  describe('Canonical tensor inventory, determinism & isolation', () => {
    it('emits exact 113 VQ-VAE tensors (43,387 floats) and 4 codec tensors (528 floats)', () => {
      const vq = createUniEmu3VqWeights(UNI_MULTIMODAL_SEED);
      const vqTensors = collectUniVqVaeWeights(vq);
      expect(vq.seed).toBe(UNI_MULTIMODAL_SEED);
      expect(vqTensors.length).toBe(UNI_MINI_VQVAE_TOTAL_TENSOR_COUNT);

      let cbCount = 0;
      let cbFloats = 0;
      let encCount = 0;
      let encFloats = 0;
      let decCount = 0;
      let decFloats = 0;
      const seenNames = new Set<string>();

      for (const t of vqTensors) {
        expect(seenNames.has(t.name)).toBe(false);
        seenNames.add(t.name);
        const shapeProd = t.shape.reduce((a, b) => a * b, 1);
        expect(t.data.length).toBe(shapeProd);
        for (let i = 0; i < t.data.length; i++) {
          expect(Number.isFinite(t.data[i])).toBe(true);
        }
        if (t.subsystem === 'vqvae_codebook') {
          cbCount++;
          cbFloats += t.data.length;
        } else if (t.subsystem === 'vqvae_encoder') {
          encCount++;
          encFloats += t.data.length;
        } else if (t.subsystem === 'vqvae_decoder') {
          decCount++;
          decFloats += t.data.length;
        } else {
          throw new Error(`Unexpected subsystem: ${t.subsystem}`);
        }
      }

      expect(cbCount).toBe(UNI_MINI_VQVAE_CODEBOOK_TENSOR_COUNT);
      expect(cbFloats).toBe(UNI_MINI_VQVAE_CODEBOOK_FLOAT_COUNT);
      expect(encCount).toBe(UNI_MINI_VQVAE_ENCODER_TENSOR_COUNT);
      expect(encFloats).toBe(UNI_MINI_VQVAE_ENCODER_FLOAT_COUNT);
      expect(decCount).toBe(UNI_MINI_VQVAE_DECODER_TENSOR_COUNT);
      expect(decFloats).toBe(UNI_MINI_VQVAE_DECODER_FLOAT_COUNT);
      expect(cbFloats + encFloats + decFloats).toBe(UNI_MINI_VQVAE_TOTAL_FLOAT_COUNT);

      // Verify exact boundary tensor names in canonical order
      expect(vqTensors[0].name).toBe('encoder.vqvae.quantize.embedding.weight');
      expect(vqTensors[0].shape).toEqual([256, 8]);
      expect(vqTensors[1].name).toBe('encoder.vqvae.encoder.conv_in.weight');
      expect(vqTensors[56].name).toBe('encoder.vqvae.quant_conv.bias');
      expect(vqTensors[57].name).toBe('encoder.vqvae.post_quant_conv.weight');
      expect(vqTensors[112].name).toBe('encoder.vqvae.decoder.conv_out.bias');

      // PackedLinearImageCodec: 4 tensors, 528 floats
      const codec = createUniPackedCodecWeights(UNI_MULTIMODAL_SEED);
      const codecTensors = collectUniCodecWeights(codec);
      expect(codecTensors.length).toBe(UNI_MINI_CODEC_TENSOR_COUNT);
      expect(codecTensors.map((t) => t.name)).toEqual([
        'encoder.packed_image_codec.down_norm.weight',
        'encoder.packed_image_codec.down_projection.weight',
        'encoder.packed_image_codec.up_norm.weight',
        'encoder.packed_image_codec.up_projection.weight',
      ]);
      expect(codecTensors.map((t) => t.shape)).toEqual([
        [8],
        [8, 32],
        [8],
        [32, 8],
      ]);
      const codecFloats = codecTensors.reduce((acc, t) => acc + t.data.length, 0);
      expect(codecFloats).toBe(UNI_MINI_CODEC_FLOAT_COUNT);
      for (const t of codecTensors) {
        expect(t.subsystem).toBe('packed_image_codec');
        for (let i = 0; i < t.data.length; i++) {
          expect(Number.isFinite(t.data[i])).toBe(true);
        }
      }
    });

    it('is 100% deterministic by seed and uses zero Math.random or forbidden imports', () => {
      const vqA = createUniEmu3VqWeights(UNI_MULTIMODAL_SEED);
      const vqB = createUniEmu3VqWeights(UNI_MULTIMODAL_SEED);
      const vqC = createUniEmu3VqWeights(UNI_MULTIMODAL_SEED + 1);
      assertBitExactFiniteArray(vqA.codebook, vqB.codebook, 'codebook_determinism');
      expect(vqA.codebook[0]).not.toBe(vqC.codebook[0]);

      const codecA = createUniPackedCodecWeights(UNI_MULTIMODAL_SEED);
      const codecB = createUniPackedCodecWeights(UNI_MULTIMODAL_SEED);
      assertBitExactFiniteArray(
        codecA.downProjection,
        codecB.downProjection,
        'codec_determinism',
      );

      for (const relPath of ['src/uni/vqvae.ts', 'src/uni/codec.ts']) {
        const src = readFileSync(resolve(process.cwd(), relPath), 'utf-8');
        expect(src).not.toMatch(/Math\.random/);
        expect(src).not.toMatch(/src\/twotower|twotower\//);
        expect(src).not.toMatch(/src\/diffusion|diffusion\//);
        expect(src).not.toMatch(/ui\/controls/);
      }
    });
  });

  describe('Mini Emu3 VQ-VAE Conv/Norm/Attn/IBQ mechanics', () => {
    it('computes exact GroupNorm (G=4, eps=1e-6) and asymmetric bottom-right Conv2d downsampling', () => {
      const x = makeDeterministicF32(8 * 4 * 4, 7, 1.0);
      const gnWeights = {
        numChannels: 8,
        numGroups: 4,
        weight: Float32Array.from([1, 1, 1, 1, 1, 1, 1, 1]),
        bias: Float32Array.from([0, 0, 0, 0, 0, 0, 0, 0]),
      };
      const normed = groupNormChw(x, 8, 4, 4, gnWeights, 1e-6);
      // Each of the 4 groups has 2 * 4 * 4 = 32 elements with mean ~ 0 and var ~ 1
      for (let g = 0; g < 4; g++) {
        let sum = 0;
        let sq = 0;
        const start = g * 32;
        for (let i = 0; i < 32; i++) {
          const v = normed[start + i];
          sum += v;
          sq += v * v;
        }
        expect(Math.abs(sum / 32)).toBeLessThan(1e-6);
        expect(Math.abs(sq / 32 - 1)).toBeLessThan(1e-3);
      }

      // Asymmetric bottom-right pad (0, 1, 0, 1) + 3x3 stride 2 maps 16x16 -> 8x8 -> 4x4
      const vq = createUniEmu3VqWeights(UNI_MULTIMODAL_SEED);
      const feat16 = makeDeterministicF32(8 * 16 * 16, 11, 0.5);
      const feat8 = conv2dChw(feat16, 8, 16, 16, vq.encoder.down0Conv, {
        stride: 2,
        padding: 0,
        asymBottomRightPad: true,
      });
      expect(feat8.length).toBe(8 * 8 * 8);

      const res16 = runEmu3ResBlock(feat16, 16, 16, vq.encoder.down0Block);
      expect(res16.length).toBe(8 * 16 * 16);

      const feat4Mid = makeDeterministicF32(16 * 4 * 4, 19, 0.5);
      const attnOut = runEmu3AttnBlock(feat4Mid, 4, 4, vq.encoder.midAttn1);
      expect(attnOut.length).toBe(16 * 4 * 4);
      for (let i = 0; i < attnOut.length; i++) {
        expect(Number.isFinite(attnOut[i])).toBe(true);
      }
    });

    it('proves IBQ uses inner product argmax(z @ E^T) rather than Euclidean argmin ||z - e||^2 (U14)', () => {
      // Construct 2 codebook vectors in R^8 where inner product and L2 disagree:
      // z = [2, 0, ...], e_0 = [1, 0, ...], e_1 = [2, 3, ...]
      //   Dot scores: z · e_0 = 2.0, z · e_1 = 4.0 -> IBQ argmax picks 1!
      //   Squared L2: ||z - e_0||^2 = 1.0, ||z - e_1||^2 = 9.0 -> Euclidean argmin would pick 0!
      const codebook = new Float32Array(256 * 8);
      codebook[0 * 8 + 0] = 1.0;
      codebook[1 * 8 + 0] = 2.0;
      codebook[1 * 8 + 1] = 3.0;
      // Leave remaining entries 2..255 at zero (dot = 0)

      // Test also strict '>' tie-breaking on cell 1: e_2 = [4, 0, ...] ties with e_1 dot=4.0,
      // and PyTorch .argmax(-1) picks the earliest index (1 < 2).
      codebook[2 * 8 + 0] = 2.0;
      codebook[2 * 8 + 1] = -3.0;

      // zContinuous in CHW [8, 1, 2]: cell 0 = [2, 0, ...], cell 1 = [2, 0, ...]
      const zContinuous = new Float32Array(8 * 1 * 2);
      zContinuous[0 * 2 + 0] = 2.0;
      zContinuous[0 * 2 + 1] = 2.0;

      const res = quantizeIbqDotProduct(zContinuous, 1, 2, codebook, 8, 256);
      expect(res.dotScores[0]).toBe(2.0);
      expect(res.dotScores[1]).toBe(4.0);
      expect(res.dotScores[2]).toBe(4.0);
      // Must pick index 1 (NOT 0 from L2, and NOT 2 on strict '>' tie-break)
      expect(Array.from(res.codes)).toEqual([1, 1]);
      expect(res.zQuantized[0 * 2 + 0]).toBe(2.0);
      expect(res.zQuantized[1 * 2 + 0]).toBe(3.0);
    });

    it('encodes 16x16 & 16x8 RGB images and decodes codes into finite [0, 1] RGB images', () => {
      const vq = createUniEmu3VqWeights(UNI_MULTIMODAL_SEED);
      const img16 = makeSyntheticRgbImage(16, 16, 42);
      const enc16 = encodeMiniEmu3Vq(img16, vq);

      expect(enc16.vqGridRows).toBe(4);
      expect(enc16.vqGridCols).toBe(4);
      expect(enc16.normalizedInput.length).toBe(3 * 16 * 16);
      expect(enc16.zContinuous.length).toBe(8 * 4 * 4);
      expect(enc16.dotScores.length).toBe(16 * UNI_MINI_VQ_CODEBOOK_SIZE);
      expect(enc16.codes.length).toBe(16);
      expect(enc16.zQuantized.length).toBe(8 * 4 * 4);
      for (let i = 0; i < enc16.codes.length; i++) {
        expect(enc16.codes[i]).toBeGreaterThanOrEqual(0);
        expect(enc16.codes[i]).toBeLessThan(UNI_MINI_VQ_CODEBOOK_SIZE);
      }

      const dec16 = decodeMiniEmu3Vq(enc16.codes, 4, 4, vq);
      expect(dec16.outputWidth).toBe(16);
      expect(dec16.outputHeight).toBe(16);
      expect(dec16.rawOutput.length).toBe(3 * 16 * 16);
      expect(dec16.rgbImage.width).toBe(16);
      expect(dec16.rgbImage.height).toBe(16);
      for (let i = 0; i < dec16.rgbImage.data.length; i++) {
        const px = dec16.rgbImage.data[i];
        expect(Number.isFinite(px)).toBe(true);
        expect(px).toBeGreaterThanOrEqual(0);
        expect(px).toBeLessThanOrEqual(1);
      }

      // Non-square 16x8 image -> 2x4 VQ grid -> 16x8 decoded RGB
      const imgRect = makeSyntheticRgbImage(16, 8, 99);
      const encRect = encodeMiniEmu3Vq(imgRect, vq);
      expect(encRect.vqGridRows).toBe(2);
      expect(encRect.vqGridCols).toBe(4);
      expect(encRect.codes.length).toBe(8);
      const decRect = decodeMiniEmu3Vq(encRect.codes, 2, 4, vq);
      expect(decRect.outputWidth).toBe(16);
      expect(decRect.outputHeight).toBe(8);
      expect(decRect.rgbImage.data.length).toBe(3 * 8 * 16);

      // Verify explicit clamping when conv_out bias pushes rawOutput outside [-1, 1]
      const boostedWeights = {
        ...vq,
        decoder: {
          ...vq.decoder,
          convOut: {
            ...vq.decoder.convOut,
            bias: Float32Array.from([5.0, -5.0, 0.0]),
          },
        },
      };
      const clampedDec = decodeMiniEmu3Vq(enc16.codes, 4, 4, boostedWeights);
      expect(clampedDec.rgbImage.data[0]).toBe(1);
      expect(clampedDec.rgbImage.data[16 * 16]).toBe(0);
    });
  });

  describe('Row/end markers, fixed prefix, PackedLinearImageCodec (D3) & 2D_SinCos alignment', () => {
    it('round-trips +2 row/end markers (4x4 -> 4x6 -> 4x4) and builds 4 fixed-prefix tokens', () => {
      const codes = Int32Array.from([
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
      ]);
      const marked = addGenerationRowMarkers(codes, 4, 4);
      expect(marked.length).toBe(24);
      // Rows 0..2 end with [256, 256]; Row 3 ends with [258, 258]
      expect(Array.from(marked.subarray(4, 6))).toEqual([
        UNI_MINI_GEN_ROW_MARKER_ID,
        UNI_MINI_GEN_ROW_MARKER_ID,
      ]);
      expect(Array.from(marked.subarray(10, 12))).toEqual([256, 256]);
      expect(Array.from(marked.subarray(16, 18))).toEqual([256, 256]);
      expect(Array.from(marked.subarray(22, 24))).toEqual([
        UNI_MINI_GEN_IMAGE_END_ID,
        UNI_MINI_GEN_IMAGE_END_ID,
      ]);

      const stripped = stripGenerationRowMarkers(marked, 4, 6);
      expect(Array.from(stripped)).toEqual(Array.from(codes));

      const prefixIds = buildFixedPrefixTokenIds();
      expect(prefixIds.length).toBe(UNI_MINI_GEN_FIXED_PREFIX_COUNT);
      expect(Array.from(prefixIds)).toEqual([259, 260, 261, 262]);

      const genEmbedding = makeDeterministicF32(264 * 8, 55, 0.25);
      const prefixEmbeds = embedFixedPrefixTokens(genEmbedding, 8);
      expect(prefixEmbeds.length).toBe(4 * 8);
      // Exact uncompressed lookup from rows 259..262
      assertBitExactFiniteArray(
        prefixEmbeds,
        genEmbedding.subarray(259 * 8, 263 * 8),
        'fixed_prefix_uncompressed',
      );
    });

    it('verifies compress/expand shapes, D3 dual gain presets, and 0-ULP 2D_SinCos column alignment', () => {
      const codec = createUniPackedCodecWeights(UNI_MULTIMODAL_SEED);

      // Target generation (4x6 -> 2x3, isSourceEdit = false -> markerColumn = true)
      const targetChildren = makeDeterministicF32(4 * 6 * 8, 101, 0.3);
      const miniTrace = compressPackedImageEmbeddings(targetChildren, 4, 6, codec, {
        isSourceEdit: false,
        gainPresetId: 'mini_default',
      });
      const prodTrace = compressPackedImageEmbeddings(targetChildren, 4, 6, codec, {
        isSourceEdit: false,
        gainPresetId: 'prod_exact',
      });

      expect(miniTrace.parentRows).toBe(2);
      expect(miniTrace.parentCols).toBe(3);
      expect(miniTrace.markerColumn).toBe(true);
      expect(miniTrace.sinCos2d.contentWidth).toBe(2);
      expect(miniTrace.normedChildren.length).toBe(24 * 8);
      expect(miniTrace.packedConcat.length).toBe(6 * 32);
      expect(miniTrace.projected.length).toBe(6 * 8);
      expect(miniTrace.output.length).toBe(6 * 8);

      // D3: projected is identical, while scaled ratio matches prod/mini combinedGenerationScale
      assertBitExactFiniteArray(
        miniTrace.projected,
        prodTrace.projected,
        'codec_projected_invariant_to_gain',
      );
      const expectedRatio =
        UNI_GAIN_PRESETS.prod_exact.combinedGenerationScale /
        UNI_GAIN_PRESETS.mini_default.combinedGenerationScale;
      for (let i = 0; i < miniTrace.projected.length; i++) {
        const expectedProdScaled = Math.fround(
          miniTrace.projected[i] *
            Math.fround(UNI_GAIN_PRESETS.prod_exact.combinedGenerationScale),
        );
        expect(prodTrace.scaled[i]).toBe(expectedProdScaled);
        if (Math.abs(miniTrace.scaled[i]) > 1e-4) {
          expect(
            Math.abs(prodTrace.scaled[i] / miniTrace.scaled[i] - expectedRatio),
          ).toBeLessThan(1e-5);
        }
      }

      // Source edit (4x4 -> 2x2, isSourceEdit = true -> markerColumn = false)
      const editChildren = makeDeterministicF32(4 * 4 * 8, 202, 0.3);
      const editTrace = compressPackedImageEmbeddings(editChildren, 4, 4, codec, {
        isSourceEdit: true,
      });
      expect(editTrace.parentRows).toBe(2);
      expect(editTrace.parentCols).toBe(2);
      expect(editTrace.markerColumn).toBe(false);
      expect(editTrace.sinCos2d.contentWidth).toBe(2);

      // 0-ULP content-column alignment between target gen (2x3, markerColumn=true)
      // and source edit (2x2, markerColumn=false) on all content columns c in {0, 1}
      const refSinCos = compute2dSinCosPosEmbed(2, 2, 8, false);
      assertBitExactFiniteArray(
        editTrace.sinCos2d.embeddings,
        refSinCos.embeddings,
        'edit_2d_sincos',
      );
      for (let r = 0; r < 2; r++) {
        for (let c = 0; c < 2; c++) {
          const targetOffset = (r * 3 + c) * 8;
          const editOffset = (r * 2 + c) * 8;
          assertBitExactFiniteArray(
            miniTrace.sinCos2d.embeddings.subarray(targetOffset, targetOffset + 8),
            editTrace.sinCos2d.embeddings.subarray(editOffset, editOffset + 8),
            `sincos_0ulp_align_r${r}_c${c}`,
          );
        }
      }

      // Expand 2x3 -> 4x6 with and without genPredictor
      const genPredictor = makeDeterministicF32(
        UNI_MINI_VQ_CODEBOOK_SIZE * UNI_MINI_CONFIG.hiddenSize,
        303,
        0.1,
      );
      const expandNoLogits = expandPackedImageHiddenStates(
        miniTrace.output,
        2,
        3,
        codec,
      );
      expect(expandNoLogits.childRows).toBe(4);
      expect(expandNoLogits.childCols).toBe(6);
      expect(expandNoLogits.unpackedChildren.length).toBe(24 * 8);
      expect(expandNoLogits.logits).toBeNull();

      const expandWithLogits = expandPackedImageHiddenStates(
        miniTrace.output,
        2,
        3,
        codec,
        genPredictor,
      );
      expect(expandWithLogits.logits).not.toBeNull();
      expect(expandWithLogits.logits!.length).toBe(24 * 256);
      assertBitExactFiniteArray(
        expandWithLogits.unpackedChildren,
        expandNoLogits.unpackedChildren,
        'expand_unpacked_children',
      );
      for (let i = 0; i < expandWithLogits.logits!.length; i++) {
        expect(Number.isFinite(expandWithLogits.logits![i])).toBe(true);
      }
    });
  });

  describe('Non-square 16x8 -> 16x16 padding lock & exact crop recovery (U18, U30)', () => {
    const letterbox16x8: UniLetterboxInfo = {
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
    };

    it('locks exact 8 border cells in rows 0 & 3 and keeps 8 center cells in rows 1 & 2 editable', () => {
      const mask = computeSourcePaddingVqMask(letterbox16x8, 4);
      expect(mask.vqGridRows).toBe(4);
      expect(mask.vqGridCols).toBe(4);
      expect(mask.lockedCellCount).toBe(8);
      expect(mask.editableCellCount).toBe(8);
      expect(Array.from(mask.lockMask)).toEqual([
        1, 1, 1, 1,
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 1, 1, 1,
      ]);

      // Code lock immutability check
      const sourceCodes = Int32Array.from({ length: 16 }, (_, i) => 100 + i);
      const targetCodes = Int32Array.from({ length: 16 }, (_, i) => 200 + i);
      const targetBefore = Int32Array.from(targetCodes);
      const sourceBefore = Int32Array.from(sourceCodes);

      const lockedCodes = applySourcePaddingVqLock(targetCodes, sourceCodes, mask);
      expect(Array.from(targetCodes)).toEqual(Array.from(targetBefore));
      expect(Array.from(sourceCodes)).toEqual(Array.from(sourceBefore));
      expect(Array.from(lockedCodes)).toEqual([
        100, 101, 102, 103,
        204, 205, 206, 207,
        208, 209, 210, 211,
        112, 113, 114, 115,
      ]);

      // Embedding lock immutability check
      const sourceEmbeds = makeDeterministicF32(16 * 8, 404, 0.5);
      const targetEmbeds = makeDeterministicF32(16 * 8, 505, 0.5);
      const targetEmbedCopy = new Float32Array(targetEmbeds);
      const lockedEmbeds = applySourcePaddingEmbeddingLock(
        targetEmbeds,
        sourceEmbeds,
        mask,
        8,
      );
      assertBitExactFiniteArray(targetEmbeds, targetEmbedCopy, 'target_embeds_immutable');
      for (let cell = 0; cell < 16; cell++) {
        const expectedSrc = mask.lockMask[cell] === 1 ? sourceEmbeds : targetEmbeds;
        assertBitExactFiniteArray(
          lockedEmbeds.subarray(cell * 8, (cell + 1) * 8),
          expectedSrc.subarray(cell * 8, (cell + 1) * 8),
          `locked_embed_cell_${cell}`,
        );
      }
    });

    it('recovers bit-exact 16x8 pixels via cropPaddedImageToSource', () => {
      const source = makeSyntheticRgbImage(16, 8, 777);
      const paddedData = new Float32Array(3 * 16 * 16);
      // Fill border with distinct sentinel 0.25 and place source into rows [4..12)
      paddedData.fill(0.25);
      for (let c = 0; c < 3; c++) {
        for (let y = 0; y < 8; y++) {
          for (let x = 0; x < 16; x++) {
            paddedData[(c * 16 + (y + 4)) * 16 + x] = source.data[(c * 8 + y) * 16 + x];
          }
        }
      }
      const padded: UniRgbImage = { width: 16, height: 16, data: paddedData };
      const cropped = cropPaddedImageToSource(padded, letterbox16x8);

      expect(cropped.width).toBe(16);
      expect(cropped.height).toBe(8);
      assertBitExactFiniteArray(cropped.data, source.data, 'crop_16x16_to_16x8_exact');
      // Also check assertFiniteArrayParity helper usage compiles and passes at 0 tolerance
      expect(assertFiniteArrayParity(cropped.data, source.data, 0, 'crop_parity')).toBe(0);
    });
  });
});
