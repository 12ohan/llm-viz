#!/usr/bin/env node
/**
 * Exports all 618 canonical core Float32 tensors of Nemotron-Labs-Diffusion-Uni
 * miniature Elastic-MoT backbone (`139,024` floats = `556,096 B`) as exact
 * little-endian IEEE-754 Float32 binary weights + manifest JSON + cryptographic
 * SHA-256 digests for consumption by the independent Python 3.12 / PyTorch 2.15.0
 * backbone reference script (`generate_uni_backbone_pytorch_fixtures.py`).
 *
 * Exports into `[outputDir]` (default `/tmp/nemotron-viz-uni-ref`):
 *   - `uni_mini_core_weights.f32`
 *   - `uni_mini_core_manifest.json`
 *
 * Usage:
 *   node scripts/reference/export_uni_params.mjs [outputDir]
 */

import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const EXPECTED_CORE_TENSORS = 618;
const EXPECTED_CORE_FLOATS = 139_024;
const EXPECTED_CORE_BYTES = 556_096;

const outDir = resolve(process.argv[2] ?? '/tmp/nemotron-viz-uni-ref');
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
  const paramsMod = await server.ssrLoadModule('/src/uni/parameters.ts');
  const { createDefaultUniCoreModel, collectUniBackboneWeights } = paramsMod;

  const model = createDefaultUniCoreModel();
  const tensors = collectUniBackboneWeights(model);

  let totalFloats = 0;
  let uTensorCount = 0;
  let gTensorCount = 0;
  let uFloats = 0;
  let gFloats = 0;

  for (const t of tensors) {
    totalFloats += t.data.length;
    if (t.branch === 'u') {
      uTensorCount += 1;
      uFloats += t.data.length;
    } else if (t.branch === 'g') {
      gTensorCount += 1;
      gFloats += t.data.length;
    } else {
      throw new Error(`Unexpected tensor branch "${String(t.branch)}" on ${t.name}`);
    }
  }

  const totalBytes = totalFloats * 4;
  if (
    tensors.length !== EXPECTED_CORE_TENSORS ||
    totalFloats !== EXPECTED_CORE_FLOATS ||
    totalBytes !== EXPECTED_CORE_BYTES
  ) {
    throw new Error(
      `Unexpected UNI core counts: tensors=${tensors.length} (expected ${EXPECTED_CORE_TENSORS}), ` +
        `floats=${totalFloats} (expected ${EXPECTED_CORE_FLOATS}), bytes=${totalBytes} (expected ${EXPECTED_CORE_BYTES})`,
    );
  }

  const binBuf = Buffer.allocUnsafe(totalBytes);
  const manifestTensors = [];
  let offsetFloats = 0;

  for (const t of tensors) {
    const tBuf = encodeFloat32Buffer(t.data, t.name);
    tBuf.copy(binBuf, offsetFloats * 4);

    const tensorSha256 = createHash('sha256').update(tBuf).digest('hex');
    manifestTensors.push({
      name: t.name,
      role: t.role,
      branch: t.branch,
      layerIndex: t.layerIndex,
      splitTower: t.splitTower,
      shape: Array.from(t.shape),
      offsetFloats,
      length: t.data.length,
      sha256: tensorSha256,
    });
    offsetFloats += t.data.length;
  }

  const weightsSha256 = createHash('sha256').update(binBuf).digest('hex');
  const binPath = resolve(outDir, 'uni_mini_core_weights.f32');
  const manifestPath = resolve(outDir, 'uni_mini_core_manifest.json');

  writeFileSync(binPath, binBuf);
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        modelId: model.modelId,
        backboneSeed: model.backboneSeed,
        backboneFingerprint: model.backboneFingerprint,
        config: model.config,
        tensorCount: tensors.length,
        uTensorCount,
        gTensorCount,
        totalFloats,
        uFloats,
        gFloats,
        totalBytes,
        weightsSha256,
        weightsFile: binPath,
        tensors: manifestTensors,
      },
      null,
      2,
    ) + '\n',
    'utf-8',
  );

  console.log(
    `Exported UNI[${model.modelId}]: ${tensors.length} tensors ` +
      `(${uTensorCount} U [${uFloats.toLocaleString()} floats] + ${gTensorCount} G [${gFloats.toLocaleString()} floats], ` +
      `${totalFloats.toLocaleString()} floats, ${totalBytes.toLocaleString()} bytes, ` +
      `sha256=${weightsSha256.slice(0, 16)}...) -> ${manifestPath}`,
  );
} finally {
  await server.close();
}
