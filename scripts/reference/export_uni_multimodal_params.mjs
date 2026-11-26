#!/usr/bin/env node
/**
 * Exports ALL 766 canonical Float32 tensors of Nemotron-Labs-Diffusion-Uni
 * miniature full multimodal model (`618` core + `31` vision + `4` codec + `113` VQ-VAE
 * = `766` tensors = `191,235` floats = `764,940 B`) as exact little-endian IEEE-754
 * Float32 binary weights + manifest JSON + cryptographic SHA-256 digests for
 * consumption by the independent Python 3.12 / PyTorch 2.15.0 multimodal parity
 * script (`generate_uni_multimodal_pytorch_fixtures.py`).
 *
 * Also serializes the exact procedural RGB test scenes (`sunlit_cat_16x16`,
 * `harbor_panorama_16x8`, `geometric_blocks_16x16`, plus letterboxed
 * `harbor_panorama_16x8 -> 16x16`) into the manifest JSON so Python evaluates the
 * exact bit-identical input pixels without duplicating procedural scene code.
 *
 * Exports into `[outputDir]` (default `/tmp/nemotron-viz-uni-mm-ref`):
 *   - `uni_mini_full_weights.f32`
 *   - `uni_mini_full_manifest.json`
 *
 * Usage:
 *   node scripts/reference/export_uni_multimodal_params.mjs [outputDir]
 */

import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const EXPECTED_CORE_TENSORS = 618;
const EXPECTED_CORE_FLOATS = 139_024;
const EXPECTED_VISION_TENSORS = 31;
const EXPECTED_VISION_FLOATS = 8_296;
const EXPECTED_CODEC_TENSORS = 4;
const EXPECTED_CODEC_FLOATS = 528;
const EXPECTED_VQVAE_TENSORS = 113;
const EXPECTED_VQVAE_FLOATS = 43_387;

const EXPECTED_MULTIMODAL_TENSORS =
  EXPECTED_VISION_TENSORS + EXPECTED_CODEC_TENSORS + EXPECTED_VQVAE_TENSORS; // 148
const EXPECTED_MULTIMODAL_FLOATS =
  EXPECTED_VISION_FLOATS + EXPECTED_CODEC_FLOATS + EXPECTED_VQVAE_FLOATS; // 52,211

const EXPECTED_FULL_TENSORS = EXPECTED_CORE_TENSORS + EXPECTED_MULTIMODAL_TENSORS; // 766
const EXPECTED_FULL_FLOATS = EXPECTED_CORE_FLOATS + EXPECTED_MULTIMODAL_FLOATS; // 191,235
const EXPECTED_FULL_BYTES = EXPECTED_FULL_FLOATS * 4; // 764,940

const outDir = resolve(process.argv[2] ?? '/tmp/nemotron-viz-uni-mm-ref');
mkdirSync(outDir, { recursive: true });

function encodeFloat32Buffer(data, label) {
  const buf = Buffer.allocUnsafe(data.length * 4);
  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    if (!Number.isFinite(val)) {
      throw new RangeError(`Non-finite float value (${String(val)}) in ${label}[${i}]`);
    }
    buf.writeFloatLE(val, i * 4);
  }
  return buf;
}

function sumFloats(tensors) {
  let total = 0;
  for (const t of tensors) {
    total += t.data.length;
  }
  return total;
}

function serializeScene(image) {
  return {
    presetId: image.presetId ?? null,
    label: image.label ?? null,
    width: image.width,
    height: image.height,
    data: Array.from(image.data),
    groundTruthBoxes: (image.groundTruthBoxes ?? []).map((b) => ({
      tag: b.tag,
      pixelBox: Array.from(b.pixelBox),
    })),
  };
}

const server = await createServer({
  logLevel: 'silent',
  server: { middlewareMode: true },
});

try {
  const [paramsMod, visionMod, codecMod, vqMod, typesMod] = await Promise.all([
    server.ssrLoadModule('/src/uni/parameters.ts'),
    server.ssrLoadModule('/src/uni/vision.ts'),
    server.ssrLoadModule('/src/uni/codec.ts'),
    server.ssrLoadModule('/src/uni/vqvae.ts'),
    server.ssrLoadModule('/src/uni/types.ts'),
  ]);

  const model = paramsMod.createDefaultUniCoreModel();
  const coreTensors = paramsMod.collectUniBackboneWeights(model);

  const visionWeights = visionMod.createUniVisionSubsystemWeights();
  const visionTensors = visionMod.collectUniVisionWeights(visionWeights);

  const codecWeights = codecMod.createUniPackedCodecWeights();
  const codecTensors = codecMod.collectUniCodecWeights(codecWeights);

  const vqWeights = vqMod.createUniEmu3VqWeights();
  const vqTensors = vqMod.collectUniVqVaeWeights(vqWeights);

  // Verify exact per-subsystem counts
  const coreFloats = sumFloats(coreTensors);
  const visionFloats = sumFloats(visionTensors);
  const codecFloats = sumFloats(codecTensors);
  const vqFloats = sumFloats(vqTensors);

  if (coreTensors.length !== EXPECTED_CORE_TENSORS || coreFloats !== EXPECTED_CORE_FLOATS) {
    throw new Error(
      `Core count mismatch: tensors=${coreTensors.length} (expected ${EXPECTED_CORE_TENSORS}), floats=${coreFloats} (expected ${EXPECTED_CORE_FLOATS})`,
    );
  }
  if (visionTensors.length !== EXPECTED_VISION_TENSORS || visionFloats !== EXPECTED_VISION_FLOATS) {
    throw new Error(
      `Vision count mismatch: tensors=${visionTensors.length} (expected ${EXPECTED_VISION_TENSORS}), floats=${visionFloats} (expected ${EXPECTED_VISION_FLOATS})`,
    );
  }
  if (codecTensors.length !== EXPECTED_CODEC_TENSORS || codecFloats !== EXPECTED_CODEC_FLOATS) {
    throw new Error(
      `Codec count mismatch: tensors=${codecTensors.length} (expected ${EXPECTED_CODEC_TENSORS}), floats=${codecFloats} (expected ${EXPECTED_CODEC_FLOATS})`,
    );
  }
  if (vqTensors.length !== EXPECTED_VQVAE_TENSORS || vqFloats !== EXPECTED_VQVAE_FLOATS) {
    throw new Error(
      `VQ-VAE count mismatch: tensors=${vqTensors.length} (expected ${EXPECTED_VQVAE_TENSORS}), floats=${vqFloats} (expected ${EXPECTED_VQVAE_FLOATS})`,
    );
  }

  const allTensors = [
    ...coreTensors.map((t) => ({
      name: t.name,
      group: 'core',
      subsystem: `core_${t.branch}`,
      shape: Array.from(t.shape),
      data: t.data,
      unusedInForward: false,
    })),
    ...visionTensors.map((t) => ({
      name: t.name,
      group: 'multimodal',
      subsystem: t.subsystem,
      shape: Array.from(t.shape),
      data: t.data,
      unusedInForward: Boolean(t.unusedInForward),
    })),
    ...codecTensors.map((t) => ({
      name: t.name,
      group: 'multimodal',
      subsystem: t.subsystem,
      shape: Array.from(t.shape),
      data: t.data,
      unusedInForward: Boolean(t.unusedInForward),
    })),
    ...vqTensors.map((t) => ({
      name: t.name,
      group: 'multimodal',
      subsystem: t.subsystem,
      shape: Array.from(t.shape),
      data: t.data,
      unusedInForward: Boolean(t.unusedInForward),
    })),
  ];

  const totalFloats = sumFloats(allTensors);
  const totalBytes = totalFloats * 4;

  if (
    allTensors.length !== EXPECTED_FULL_TENSORS ||
    totalFloats !== EXPECTED_FULL_FLOATS ||
    totalBytes !== EXPECTED_FULL_BYTES
  ) {
    throw new Error(
      `Full model count mismatch: tensors=${allTensors.length} (expected ${EXPECTED_FULL_TENSORS}), ` +
        `floats=${totalFloats} (expected ${EXPECTED_FULL_FLOATS}), bytes=${totalBytes} (expected ${EXPECTED_FULL_BYTES})`,
    );
  }

  // Ensure all tensor names are strictly unique
  const seenNames = new Set();
  const binBuf = Buffer.allocUnsafe(totalBytes);
  const manifestTensors = [];
  let offsetFloats = 0;

  for (const t of allTensors) {
    if (seenNames.has(t.name)) {
      throw new Error(`Duplicate tensor name in canonical export: ${t.name}`);
    }
    seenNames.add(t.name);

    const tBuf = encodeFloat32Buffer(t.data, t.name);
    tBuf.copy(binBuf, offsetFloats * 4);
    const sha256 = createHash('sha256').update(tBuf).digest('hex');

    manifestTensors.push({
      name: t.name,
      group: t.group,
      subsystem: t.subsystem,
      shape: t.shape,
      offsetFloats,
      length: t.data.length,
      unusedInForward: t.unusedInForward,
      sha256,
    });
    offsetFloats += t.data.length;
  }

  const coreByteLen = EXPECTED_CORE_FLOATS * 4;
  const coreSha256 = createHash('sha256').update(binBuf.subarray(0, coreByteLen)).digest('hex');
  const multimodalSha256 = createHash('sha256').update(binBuf.subarray(coreByteLen)).digest('hex');
  const weightsSha256 = createHash('sha256').update(binBuf).digest('hex');

  // Export deterministic procedural scenes for exact input parity in PyTorch
  const sunlitCat = visionMod.createUniScenePreset('sunlit_cat_16x16');
  const harborPanorama = visionMod.createUniScenePreset('harbor_panorama_16x8');
  const geometricBlocks = visionMod.createUniScenePreset('geometric_blocks_16x16');
  const harborLetterboxed = visionMod.padToSquareAndResize(harborPanorama, 16);

  const binPath = resolve(outDir, 'uni_mini_full_weights.f32');
  const manifestPath = resolve(outDir, 'uni_mini_full_manifest.json');

  writeFileSync(binPath, binBuf);
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        modelId: model.modelId,
        backboneSeed: model.backboneSeed,
        multimodalSeed: typesMod.UNI_MULTIMODAL_SEED,
        backboneFingerprint: model.backboneFingerprint,
        config: model.config,
        vocab: Array.from(typesMod.UNI_MINI_VOCAB),
        gainPresets: typesMod.UNI_GAIN_PRESETS,
        clipMean: Array.from(typesMod.UNI_CLIP_MEAN),
        clipStd: Array.from(typesMod.UNI_CLIP_STD),
        counts: {
          coreTensors: EXPECTED_CORE_TENSORS,
          coreFloats: EXPECTED_CORE_FLOATS,
          visionTensors: EXPECTED_VISION_TENSORS,
          visionFloats: EXPECTED_VISION_FLOATS,
          codecTensors: EXPECTED_CODEC_TENSORS,
          codecFloats: EXPECTED_CODEC_FLOATS,
          vqvaeTensors: EXPECTED_VQVAE_TENSORS,
          vqvaeFloats: EXPECTED_VQVAE_FLOATS,
          multimodalTensors: EXPECTED_MULTIMODAL_TENSORS,
          multimodalFloats: EXPECTED_MULTIMODAL_FLOATS,
          totalTensors: EXPECTED_FULL_TENSORS,
          totalFloats: EXPECTED_FULL_FLOATS,
          totalBytes: EXPECTED_FULL_BYTES,
        },
        weightsSha256,
        coreSha256,
        multimodalSha256,
        weightsFile: binPath,
        scenes: {
          sunlit_cat_16x16: serializeScene(sunlitCat),
          harbor_panorama_16x8: serializeScene(harborPanorama),
          geometric_blocks_16x16: serializeScene(geometricBlocks),
          harbor_panorama_letterboxed_16x16: {
            image: serializeScene(harborLetterboxed.image),
            info: harborLetterboxed.info,
          },
        },
        tensors: manifestTensors,
      },
      null,
      2,
    ) + '\n',
    'utf-8',
  );

  console.log(
    `Exported UNI-FULL[${model.modelId}]: ${allTensors.length} tensors ` +
      `(${EXPECTED_CORE_TENSORS} core [${coreFloats.toLocaleString()} floats] + ` +
      `${EXPECTED_VISION_TENSORS} vision [${visionFloats.toLocaleString()}] + ` +
      `${EXPECTED_CODEC_TENSORS} codec [${codecFloats.toLocaleString()}] + ` +
      `${EXPECTED_VQVAE_TENSORS} vqvae [${vqFloats.toLocaleString()}] = ` +
      `${totalFloats.toLocaleString()} floats = ${totalBytes.toLocaleString()} B, ` +
      `weightsSha256=${weightsSha256.slice(0, 16)}...) -> ${manifestPath}`,
  );
} finally {
  await server.close();
}
