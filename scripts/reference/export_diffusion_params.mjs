#!/usr/bin/env node
/**
 * Exports all 443 canonical Float32 tensors of Nemotron-Labs-Diffusion miniature model
 * (363 backbone tensors + 80 o_proj LoRA tensors = 271,304 floats) as exact little-endian
 * IEEE-754 Float32 binary weights + manifest JSON + cryptographic SHA-256 digests for
 * consumption by the independent Python 3.12 / PyTorch 2.15.0 Diffusion reference script
 * (`generate_diffusion_pytorch_fixtures.py`).
 *
 * Exports into `[outputDir]` (default `/tmp/nemotron-viz-diffusion-ref`):
 *   - `diffusion_mini_weights.f32`
 *   - `diffusion_mini_manifest.json`
 *
 * Usage:
 *   node scripts/reference/export_diffusion_params.mjs [outputDir]
 */

import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const EXPECTED_TOTAL_TENSORS = 443;
const EXPECTED_BACKBONE_TENSORS = 363;
const EXPECTED_LORA_TENSORS = 80;
const EXPECTED_TOTAL_FLOATS = 271_304;
const EXPECTED_BACKBONE_FLOATS = 229_064;
const EXPECTED_LORA_FLOATS = 42_240;

const outDir = resolve(process.argv[2] ?? '/tmp/nemotron-viz-diffusion-ref');
mkdirSync(outDir, { recursive: true });

function encodeFloat32Buffer(data, label) {
  const buf = Buffer.allocUnsafe(data.length * 4);
  for (let i = 0; i < data.length; i++) {
    const val = data[i];
    if (!Number.isFinite(val)) {
      throw new RangeError(`Non-finite weight value (${String(val)}) in ${label}[${i}]`);
    }
    buf.writeFloatLE(val, i * 4);
  }
  return buf;
}

const server = await createServer({
  logLevel: 'silent',
  server: { middlewareMode: true },
});

try {
  const paramsMod = await server.ssrLoadModule('/src/diffusion/parameters.ts');
  const { createDefaultDiffusionModel, collectDiffusionCanonicalWeights } = paramsMod;

  const model = createDefaultDiffusionModel();
  const tensors = collectDiffusionCanonicalWeights(model);

  let backboneTensorCount = 0;
  let loraTensorCount = 0;
  let backboneFloats = 0;
  let loraFloats = 0;

  for (const t of tensors) {
    if (t.isLora) {
      loraTensorCount += 1;
      loraFloats += t.data.length;
    } else {
      backboneTensorCount += 1;
      backboneFloats += t.data.length;
    }
  }

  const totalFloats = backboneFloats + loraFloats;

  if (
    tensors.length !== EXPECTED_TOTAL_TENSORS ||
    backboneTensorCount !== EXPECTED_BACKBONE_TENSORS ||
    loraTensorCount !== EXPECTED_LORA_TENSORS ||
    totalFloats !== EXPECTED_TOTAL_FLOATS ||
    backboneFloats !== EXPECTED_BACKBONE_FLOATS ||
    loraFloats !== EXPECTED_LORA_FLOATS
  ) {
    throw new Error(
      `Unexpected Diffusion tensor/float counts: total=${tensors.length} (${totalFloats} floats), ` +
        `backbone=${backboneTensorCount} (${backboneFloats} floats), lora=${loraTensorCount} (${loraFloats} floats)`,
    );
  }

  const binBuf = Buffer.allocUnsafe(totalFloats * 4);
  const backboneHash = createHash('sha256');
  const loraHash = createHash('sha256');
  const manifestTensors = [];
  let offsetFloats = 0;

  for (const t of tensors) {
    const tBuf = encodeFloat32Buffer(t.data, t.name);
    tBuf.copy(binBuf, offsetFloats * 4);

    if (t.isLora) {
      loraHash.update(tBuf);
    } else {
      backboneHash.update(tBuf);
    }

    const tensorSha256 = createHash('sha256').update(tBuf).digest('hex');
    manifestTensors.push({
      name: t.name,
      role: t.role,
      layerIndex: t.layerIndex,
      isLora: t.isLora,
      shape: Array.from(t.shape),
      offsetFloats,
      length: t.data.length,
      sha256: tensorSha256,
    });
    offsetFloats += t.data.length;
  }

  const weightsSha256 = createHash('sha256').update(binBuf).digest('hex');
  const backboneSha256 = backboneHash.digest('hex');
  const loraSha256 = loraHash.digest('hex');

  const binPath = resolve(outDir, 'diffusion_mini_weights.f32');
  const manifestPath = resolve(outDir, 'diffusion_mini_manifest.json');

  writeFileSync(binPath, binBuf);
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        modelId: model.modelId,
        backboneSeed: model.backboneSeed,
        loraSeed: model.loraSeed,
        backboneFingerprint: model.backboneFingerprint,
        loraFingerprint: model.loraFingerprint,
        combinedFingerprint: model.combinedFingerprint,
        config: model.config,
        tensorCount: tensors.length,
        backboneTensorCount,
        loraTensorCount,
        totalFloats,
        backboneFloats,
        loraFloats,
        totalBytes: binBuf.byteLength,
        weightsSha256,
        backboneSha256,
        loraSha256,
        weightsFile: binPath,
        tensors: manifestTensors,
      },
      null,
      2,
    ) + '\n',
    'utf-8',
  );

  console.log(
    `Exported Diffusion[${model.modelId}]: ${tensors.length} tensors ` +
      `(${backboneTensorCount} backbone + ${loraTensorCount} lora, ${totalFloats.toLocaleString()} floats, ` +
      `${binBuf.byteLength.toLocaleString()} bytes, sha256=${weightsSha256.slice(0, 16)}...) -> ${manifestPath}`,
  );
} finally {
  await server.close();
}
