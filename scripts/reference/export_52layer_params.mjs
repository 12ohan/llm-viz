#!/usr/bin/env node
/**
 * Exports all 52 layers of `createDefaultHybridModel()` as exact little-endian IEEE-754 Float32
 * binary weights + manifest JSON + SHA-256 digest for consumption by the independent PyTorch
 * 52-layer end-to-end reference script (`generate_pytorch_parity_fixtures.py`).
 *
 * Usage:
 *   node scripts/reference/export_52layer_params.mjs [outputDir]
 */

import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const outDir = resolve(process.argv[2] ?? '/tmp/nemotron-viz-ref-params');
mkdirSync(outDir, { recursive: true });

const server = await createServer({
  logLevel: 'silent',
  server: { middlewareMode: true },
});

try {
  const hybridMod = await server.ssrLoadModule('/src/math/hybridFlow.ts');
  const digestMod = await server.ssrLoadModule('/tests/helpers/modelWeightDigest.ts');

  const model = hybridMod.createDefaultHybridModel();
  const tensors = digestMod.collectCanonicalModelWeightTensors(model);

  let totalFloats = 0;
  for (const t of tensors) totalFloats += t.data.length;

  const binBuf = Buffer.allocUnsafe(totalFloats * 4);
  const manifestTensors = [];
  let offsetFloats = 0;
  for (const t of tensors) {
    for (let i = 0; i < t.data.length; i++) {
      binBuf.writeFloatLE(t.data[i], (offsetFloats + i) * 4);
    }
    manifestTensors.push({
      name: t.name,
      offsetFloats,
      length: t.data.length,
    });
    offsetFloats += t.data.length;
  }

  const weightsSha256 = createHash('sha256').update(binBuf).digest('hex');
  const binPath = resolve(outDir, 'model_52layer_weights.f32');
  const manifestPath = resolve(outDir, 'model_52layer_manifest.json');

  writeFileSync(binPath, binBuf);
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        modelId: model.modelId,
        parameterFingerprint: model.parameterFingerprint,
        seed: model.config.seed,
        pattern: model.config.pattern,
        d: model.config.d,
        vocabSize: model.config.vocabSize,
        rmsNormEps: model.config.rmsNormEps,
        residualOutputScale: model.config.residualOutputScale,
        totalFloats,
        totalBytes: binBuf.byteLength,
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
    `Exported ${tensors.length} tensors (${totalFloats.toLocaleString()} floats, ${binBuf.byteLength.toLocaleString()} bytes, sha256=${weightsSha256.slice(0, 16)}...) -> ${manifestPath}`,
  );
} finally {
  await server.close();
}
