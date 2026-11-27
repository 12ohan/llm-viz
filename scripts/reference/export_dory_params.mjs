#!/usr/bin/env node
/**
 * Exports all 507 canonical Float32 tensors of NVIDIA Nemotron-Labs-Dory1.0-4B
 * miniature Full-nGPT backbone (`93,248` floats = `372,992 B`) as exact
 * little-endian IEEE-754 binary32 binary weights + JSON manifest + cryptographic
 * SHA-256 digests (`weightsSha256` raw payload + canonical `DORY-MINI-F32-V1\n`
 * framed `modelWeightId` + per-tensor SHA-256) for consumption by the independent
 * clean-room Python 3.12 / PyTorch 2.15.0 reference generator
 * (`generate_dory_pytorch_fixtures.py`).
 *
 * Exports into `[outputDir]` (default `/tmp/nemotron-viz-dory-ref`):
 *   - `dory_mini_weights.f32`
 *   - `dory_mini_manifest.json`
 *
 * Usage:
 *   node scripts/reference/export_dory_params.mjs [outputDir]
 */

import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createServer } from 'vite';

const EXPECTED_TENSOR_COUNT = 507;
const EXPECTED_FLOAT_COUNT = 93_248;
const EXPECTED_WEIGHT_BYTES = 372_992;

const outDir = resolve(process.argv[2] ?? '/tmp/nemotron-viz-dory-ref');
mkdirSync(outDir, { recursive: true });

function encodeFloat32LE(data, label) {
  const buf = Buffer.allocUnsafe(data.length * 4);
  for (let i = 0; i < data.length; i += 1) {
    const val = data[i];
    if (!Number.isFinite(val)) {
      throw new RangeError(`Non-finite weight value (${String(val)}) in ${label}[${i}]`);
    }
    buf.writeFloatLE(val, i * 4);
  }
  return buf;
}

function encodeUint32LE(value) {
  const buf = Buffer.allocUnsafe(4);
  buf.writeUInt32LE(value >>> 0, 0);
  return buf;
}

const server = await createServer({
  logLevel: 'silent',
  server: { middlewareMode: true },
});

try {
  const paramsMod = await server.ssrLoadModule('/src/dory/parameters.ts');
  const typesMod = await server.ssrLoadModule('/src/dory/types.ts');

  const {
    getDefaultDoryModelParameters,
    listDoryCanonicalTensors,
    computeDoryWeightDigestHex,
    countDoryParameterStats,
  } = paramsMod;
  const {
    DORY_MINI_TENSOR_COUNT,
    DORY_MINI_FLOAT_COUNT,
    DORY_MINI_WEIGHT_BYTES,
    DORY_WEIGHT_DIGEST_MAGIC,
  } = typesMod;

  if (
    DORY_MINI_TENSOR_COUNT !== EXPECTED_TENSOR_COUNT ||
    DORY_MINI_FLOAT_COUNT !== EXPECTED_FLOAT_COUNT ||
    DORY_MINI_WEIGHT_BYTES !== EXPECTED_WEIGHT_BYTES
  ) {
    throw new Error(
      `DORY_MINI_* constants drifted from locked D2 contract: ` +
        `tensors=${DORY_MINI_TENSOR_COUNT}, floats=${DORY_MINI_FLOAT_COUNT}, bytes=${DORY_MINI_WEIGHT_BYTES}`,
    );
  }

  const params = getDefaultDoryModelParameters();
  const tensors = listDoryCanonicalTensors(params);
  const stats = countDoryParameterStats(params);

  let totalFloats = 0;
  for (const t of tensors) {
    let expectedShapeLen = 1;
    for (const dim of t.shape) {
      expectedShapeLen *= dim;
    }
    if (expectedShapeLen !== t.data.length) {
      throw new Error(
        `Tensor shape/length mismatch on ${t.name}: shape=[${t.shape.join(',')}] (${expectedShapeLen}) vs data.length=${t.data.length}`,
      );
    }
    totalFloats += t.data.length;
  }

  const totalBytes = totalFloats * 4;
  if (
    tensors.length !== EXPECTED_TENSOR_COUNT ||
    totalFloats !== EXPECTED_FLOAT_COUNT ||
    totalBytes !== EXPECTED_WEIGHT_BYTES
  ) {
    throw new Error(
      `Unexpected Dory parameter counts: tensors=${tensors.length} (expected ${EXPECTED_TENSOR_COUNT}), ` +
        `floats=${totalFloats} (expected ${EXPECTED_FLOAT_COUNT}), bytes=${totalBytes} (expected ${EXPECTED_WEIGHT_BYTES})`,
    );
  }

  const binBuf = Buffer.allocUnsafe(totalBytes);
  const framedHash = createHash('sha256');
  framedHash.update(Buffer.from(DORY_WEIGHT_DIGEST_MAGIC, 'utf-8'));

  const nulBuf = Buffer.from([0x00]);
  const manifestTensors = [];
  let offsetFloats = 0;

  for (const t of tensors) {
    const tBuf = encodeFloat32LE(t.data, t.name);
    tBuf.copy(binBuf, offsetFloats * 4);

    // Canonical DORY-MINI-F32-V1 framing cross-checked against pure-TS Sha256Stream
    framedHash.update(Buffer.from(t.name, 'utf-8'));
    framedHash.update(nulBuf);
    framedHash.update(encodeUint32LE(t.shape.length));
    for (const dim of t.shape) {
      framedHash.update(encodeUint32LE(dim));
    }
    framedHash.update(tBuf);

    const tensorSha256 = createHash('sha256').update(tBuf).digest('hex');
    manifestTensors.push({
      name: t.name,
      transformersPairedName: t.transformersPairedName,
      role: t.role,
      physicalLayerIdx: t.physicalLayerIdx,
      pairedLayerIdx: t.pairedLayerIdx,
      stage: t.stage,
      scaleFamily: t.scaleFamily,
      shape: Array.from(t.shape),
      offsetFloats,
      length: t.data.length,
      sha256: tensorSha256,
    });
    offsetFloats += t.data.length;
  }

  const nodeFramedDigest = framedHash.digest('hex');
  const tsDigest = computeDoryWeightDigestHex(params);
  if (nodeFramedDigest !== params.modelWeightId || nodeFramedDigest !== tsDigest) {
    throw new Error(
      `Canonical modelWeightId SHA-256 mismatch: node=${nodeFramedDigest}, ` +
        `params=${params.modelWeightId}, recomputedTs=${tsDigest}`,
    );
  }

  const weightsSha256 = createHash('sha256').update(binBuf).digest('hex');
  const binPath = resolve(outDir, 'dory_mini_weights.f32');
  const manifestPath = resolve(outDir, 'dory_mini_manifest.json');

  writeFileSync(binPath, binBuf);
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        format: DORY_WEIGHT_DIGEST_MAGIC,
        modelWeightId: params.modelWeightId,
        weightsSha256,
        backboneSeed: params.backboneSeed,
        config: params.config,
        tensorCount: tensors.length,
        totalFloats,
        totalBytes,
        stats,
        weightsFile: binPath,
        tensors: manifestTensors,
      },
      null,
      2,
    ) + '\n',
    'utf-8',
  );

  console.log(
    `Exported DORY[${params.modelWeightId.slice(0, 16)}...]: ` +
      `${tensors.length} tensors (${stats.linearWeightTensors} linear + ${stats.ngptScaleTensors} nGPT scales + ` +
      `${stats.bookendMatrixTensors} bookends, 0 norms/biases/adapters), ` +
      `${totalFloats.toLocaleString()} floats, ${totalBytes.toLocaleString()} bytes, ` +
      `weightsSha256=${weightsSha256.slice(0, 16)}... -> ${manifestPath}`,
  );
} finally {
  await server.close();
}
