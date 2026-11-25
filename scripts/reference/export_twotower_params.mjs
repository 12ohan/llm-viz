#!/usr/bin/env node
/**
 * Exports all 860 canonical Float32 tensors of Nemotron-Labs-TwoTower miniature models
 * (`context.*`, `denoiser.*`, `time_cond.*`) as exact little-endian IEEE-754 Float32
 * binary weights + manifest JSON + cryptographic SHA-256 digest for consumption by the
 * independent Python 3.12 / PyTorch 2.15.0 TwoTower reference script
 * (`generate_twotower_pytorch_fixtures.py`).
 *
 * Exports TWO variants into `[outputDir]` (default `/tmp/nemotron-viz-twotower-ref`):
 *   1. `twotower_default_weights.f32` + `twotower_default_manifest.json`
 *      (`createDefaultTwoTowerModel()`, V=34, d=8, freqDim=16, perturbationScale=0.15, float64_reference)
 *   2. `twotower_identity_weights.f32` + `twotower_identity_manifest.json`
 *      (`createTwoTowerModelParameters({ perturbationScale: 0 })`, V=34, float64_reference)
 *      for comparable AR vs. Mock-AR alignment parity.
 *
 * Usage:
 *   node scripts/reference/export_twotower_params.mjs [outputDir]
 */

import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const outDir = resolve(process.argv[2] ?? '/tmp/nemotron-viz-twotower-ref');
mkdirSync(outDir, { recursive: true });

function exportVariant(outDirPath, filePrefix, variant, model, collectCanonicalWeights) {
  const tensors = collectCanonicalWeights(model);

  let totalFloats = 0;
  for (const t of tensors) {
    totalFloats += t.data.length;
  }

  const binBuf = Buffer.allocUnsafe(totalFloats * 4);
  const manifestTensors = [];
  let offsetFloats = 0;

  for (const t of tensors) {
    for (let i = 0; i < t.data.length; i++) {
      const val = t.data[i];
      if (!Number.isFinite(val)) {
        throw new RangeError(
          `Non-finite weight value (${String(val)}) in ${variant}:${t.name}[${i}]`,
        );
      }
      binBuf.writeFloatLE(val, (offsetFloats + i) * 4);
    }
    manifestTensors.push({
      name: t.name,
      offsetFloats,
      length: t.data.length,
    });
    offsetFloats += t.data.length;
  }

  const weightsSha256 = createHash('sha256').update(binBuf).digest('hex');
  const binPath = resolve(outDirPath, `${filePrefix}_weights.f32`);
  const manifestPath = resolve(outDirPath, `${filePrefix}_manifest.json`);

  writeFileSync(binPath, binBuf);
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        variant,
        modelId: model.modelId,
        twoTowerFingerprint: model.twoTowerFingerprint,
        contextModelId: model.contextModel.modelId,
        contextFingerprint: model.contextModel.parameterFingerprint,
        denoiserModelId: model.denoiserModel.modelId,
        denoiserFingerprint: model.denoiserModel.parameterFingerprint,
        timeConditioningFingerprint: model.timeConditioning.fingerprint,
        contextSeed: model.config.contextSeed,
        timeConditioningSeed: model.config.timeConditioningSeed,
        pattern: model.config.pattern,
        d: model.config.d,
        vocabSize: model.config.vocabSize,
        maskTokenId: model.maskTokenId,
        eosTokenId: model.eosTokenId,
        freqDim: model.config.freqDim,
        maxPeriodScale: model.timeConditioning.config.maxPeriodScale,
        maxPeriod: model.timeConditioning.config.maxPeriod,
        rmsNormEps: model.contextModel.config.rmsNormEps,
        residualOutputScale: model.contextModel.config.residualOutputScale,
        routerPrecisionPolicy: model.config.routerPrecisionPolicy,
        perturbationSeed: model.config.perturbation.seed,
        perturbationScale: model.config.perturbation.scale,
        tensorCount: tensors.length,
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
    `Exported TwoTower[${variant}]: ${tensors.length} tensors (${totalFloats.toLocaleString()} floats, ${binBuf.byteLength.toLocaleString()} bytes, sha256=${weightsSha256.slice(0, 16)}...) -> ${manifestPath}`,
  );
}

const server = await createServer({
  logLevel: 'silent',
  server: { middlewareMode: true },
});

try {
  const paramsMod = await server.ssrLoadModule('/src/twotower/parameters.ts');
  const {
    createDefaultTwoTowerModel,
    createTwoTowerModelParameters,
    collectTwoTowerCanonicalWeights,
  } = paramsMod;

  const defaultModel = createDefaultTwoTowerModel();
  exportVariant(
    outDir,
    'twotower_default',
    'default',
    defaultModel,
    collectTwoTowerCanonicalWeights,
  );

  const identityModel = createTwoTowerModelParameters({
    perturbationScale: 0,
  });
  exportVariant(
    outDir,
    'twotower_identity',
    'identity',
    identityModel,
    collectTwoTowerCanonicalWeights,
  );
} finally {
  await server.close();
}
