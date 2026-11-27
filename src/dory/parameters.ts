/**
 * Deterministic 507-Tensor / 93,248-Float Singleton Parameter Builder & Pure-TS SHA-256 Digest
 * for NVIDIA Nemotron-Labs-Dory1.0-4B (`src/dory/parameters.ts`).
 *
 * Implements `DoryParametersModuleContract` (`D1`, `D2`, `D5`, `DR5`, `DR7`, `DR9`):
 *   - Exact 507 tensors = 93,248 F32 floats = 372,992 B (1:1 tensor count with production BF16).
 *   - Zero runtime weight normalization, zero scale folding, zero learned norm weights/biases/adapters.
 *   - Row-wise unit-normalized synthetic embeddings (`DR5`) and linear projections (full-rank, non-degenerate).
 *   - Structured per-channel spread on all 5 nGPT scale families:
 *       * `attn_alpha` / `mlp_alpha`: smooth spread around `0.0177` (effective `|alpha_hat| in [0.02, 0.35]`),
 *         PLUS 2 explicitly planted channels in physical layer 16 (channels 6 and 7) with
 *         effective `|alpha_hat|` = `1.12` and `1.22` in `(1.05, 1.25)` exercising unclamped
 *         NLERP extrapolation and `.abs()` (`DR7`).
 *       * `sqk`, `s_gate`, `lm_head.logit_scale`: smooth spread within `[0.7x, 1.3x] * 0.0177`.
 *       * `suv_gate`, `suv_up`: smooth spread around `1.0` WITH negative signed entries (verifying NO `.abs()`).
 *   - Clean-room synchronous FIPS 180-4 SHA-256 digest matching `DORY_WEIGHT_DIGEST_MAGIC` framing.
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL, Node built-ins, ambient RNG, or sibling imports.
 */

import { createDoryPrng, deriveDorySeed } from './prng';
import { DORY_PHYSICAL_LAYERS } from './spec';
import {
  DORY_BACKBONE_SEED,
  DORY_EMBEDDINGS_TENSOR_NAME,
  DORY_LM_HEAD_LOGIT_SCALE_TENSOR_NAME,
  DORY_LM_HEAD_TENSOR_NAME,
  DORY_MINI_CONFIG,
  DORY_MINI_VOCAB,
  DORY_PLANTED_EXTRAPOLATION_SPEC,
  DORY_WEIGHT_DIGEST_MAGIC,
  doryAttnAlphaTensorName,
  doryAttnProjTensorName,
  doryAttnScaleTensorName,
  doryMlpAlphaTensorName,
  doryMlpProjTensorName,
  doryMlpScaleTensorName,
  doryToTransformersPairedTensorName,
  type DoryAttentionSublayerWeights,
  type DoryCanonicalWeightTensor,
  type DoryMiniConfig,
  type DoryMlpSublayerWeights,
  type DoryModelParameters,
  type DoryParameterCountSummary,
  type DoryPrng,
  type DorySerializedParametersPayload,
  type DorySerializedTensorEntry,
  type DorySublayerWeights,
} from './types';

// ============================================================================
// CLEAN-ROOM SYNCHRONOUS FIPS 180-4 SHA-256 HASHER (BROWSER + NODE COMPATIBLE)
// ============================================================================

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const SHA256_H0 = new Uint32Array([
  0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]);

const IS_LITTLE_ENDIAN =
  new Uint8Array(new Uint32Array([0x01020304]).buffer)[0] === 0x04;

function rotr32(x: number, n: number): number {
  return ((x >>> n) | (x << (32 - n))) >>> 0;
}

class Sha256Stream {
  private readonly h = new Uint32Array(SHA256_H0);
  private readonly block = new Uint8Array(64);
  private readonly w = new Uint32Array(64);
  private blockOffset = 0;
  private totalBytesLo = 0;
  private totalBytesHi = 0;

  private compressBlock(bytes: Uint8Array, offset: number): void {
    const w = this.w;
    for (let i = 0; i < 16; i++) {
      const b = offset + i * 4;
      w[i] =
        ((bytes[b] << 24) |
          (bytes[b + 1] << 16) |
          (bytes[b + 2] << 8) |
          bytes[b + 3]) >>>
        0;
    }
    for (let i = 16; i < 64; i++) {
      const w15 = w[i - 15];
      const w2 = w[i - 2];
      const s0 = (rotr32(w15, 7) ^ rotr32(w15, 18) ^ (w15 >>> 3)) >>> 0;
      const s1 = (rotr32(w2, 17) ^ rotr32(w2, 19) ^ (w2 >>> 10)) >>> 0;
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = this.h[0];
    let b = this.h[1];
    let c = this.h[2];
    let d = this.h[3];
    let e = this.h[4];
    let f = this.h[5];
    let g = this.h[6];
    let hh = this.h[7];

    for (let i = 0; i < 64; i++) {
      const S1 = (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) >>> 0;
      const ch = ((e & f) ^ (~e & g)) >>> 0;
      const temp1 = (hh + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) >>> 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const temp2 = (S0 + maj) >>> 0;

      hh = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    this.h[0] = (this.h[0] + a) >>> 0;
    this.h[1] = (this.h[1] + b) >>> 0;
    this.h[2] = (this.h[2] + c) >>> 0;
    this.h[3] = (this.h[3] + d) >>> 0;
    this.h[4] = (this.h[4] + e) >>> 0;
    this.h[5] = (this.h[5] + f) >>> 0;
    this.h[6] = (this.h[6] + g) >>> 0;
    this.h[7] = (this.h[7] + hh) >>> 0;
  }

  updateBytes(bytes: Uint8Array): void {
    const len = bytes.length;
    const prevLo = this.totalBytesLo;
    this.totalBytesLo = (prevLo + len) >>> 0;
    if (this.totalBytesLo < prevLo) {
      this.totalBytesHi = (this.totalBytesHi + 1) >>> 0;
    }

    let pos = 0;
    if (this.blockOffset > 0) {
      const need = 64 - this.blockOffset;
      if (len < need) {
        this.block.set(bytes, this.blockOffset);
        this.blockOffset += len;
        return;
      }
      this.block.set(bytes.subarray(0, need), this.blockOffset);
      this.compressBlock(this.block, 0);
      this.blockOffset = 0;
      pos = need;
    }

    while (pos + 64 <= len) {
      this.compressBlock(bytes, pos);
      pos += 64;
    }

    if (pos < len) {
      this.block.set(bytes.subarray(pos), 0);
      this.blockOffset = len - pos;
    }
  }

  updateUint32LE(value: number): void {
    const buf = new Uint8Array(4);
    buf[0] = value & 0xff;
    buf[1] = (value >>> 8) & 0xff;
    buf[2] = (value >>> 16) & 0xff;
    buf[3] = (value >>> 24) & 0xff;
    this.updateBytes(buf);
  }

  updateUtf8String(text: string): void {
    const bytes: number[] = [];
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      if (code < 0x80) {
        bytes.push(code);
      } else if (code < 0x800) {
        bytes.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
      } else {
        bytes.push(
          0xe0 | (code >> 12),
          0x80 | ((code >> 6) & 0x3f),
          0x80 | (code & 0x3f),
        );
      }
    }
    this.updateBytes(new Uint8Array(bytes));
  }

  updateFloat32LE(data: Float32Array): void {
    if (IS_LITTLE_ENDIAN) {
      this.updateBytes(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      return;
    }
    const buf = new ArrayBuffer(data.length * 4);
    const view = new DataView(buf);
    for (let i = 0; i < data.length; i++) {
      view.setFloat32(i * 4, data[i], true);
    }
    this.updateBytes(new Uint8Array(buf));
  }

  digestHex(): string {
    // Bit length = totalBytes * 8
    const bitLenHi = ((this.totalBytesHi << 3) | (this.totalBytesLo >>> 29)) >>> 0;
    const bitLenLo = (this.totalBytesLo << 3) >>> 0;

    this.block[this.blockOffset++] = 0x80;
    if (this.blockOffset > 56) {
      this.block.fill(0, this.blockOffset, 64);
      this.compressBlock(this.block, 0);
      this.blockOffset = 0;
    }
    this.block.fill(0, this.blockOffset, 56);

    // Big-endian 64-bit bit length
    this.block[56] = (bitLenHi >>> 24) & 0xff;
    this.block[57] = (bitLenHi >>> 16) & 0xff;
    this.block[58] = (bitLenHi >>> 8) & 0xff;
    this.block[59] = bitLenHi & 0xff;
    this.block[60] = (bitLenLo >>> 24) & 0xff;
    this.block[61] = (bitLenLo >>> 16) & 0xff;
    this.block[62] = (bitLenLo >>> 8) & 0xff;
    this.block[63] = bitLenLo & 0xff;
    this.compressBlock(this.block, 0);

    let hex = '';
    for (let i = 0; i < 8; i++) {
      hex += this.h[i].toString(16).padStart(8, '0');
    }
    return hex;
  }
}

// ============================================================================
// DETERMINISTIC UNIT-NORM & STRUCTURED SCALE INITIALIZATION HELPERS
// ============================================================================

/**
 * Generates a `[rows, cols]` Float32Array matrix whose every row lies strictly on the unit
 * hypersphere `||row||_2 = 1` (mirrors post-training nGPT unit-norm matrix rows & `DR5`).
 */
function sampleRowUnitNormMatrix(rows: number, cols: number, prng: DoryPrng): Float32Array {
  const mat = new Float32Array(rows * cols);
  const tmp = new Float64Array(cols);

  for (let r = 0; r < rows; r++) {
    let sumSq = 0;
    do {
      sumSq = 0;
      for (let c = 0; c < cols; c++) {
        const g = prng.nextGaussian();
        tmp[c] = g;
        sumSq += g * g;
      }
    } while (sumSq < 1e-12);

    const invNorm = 1 / Math.sqrt(sumSq);
    const base = r * cols;
    for (let c = 0; c < cols; c++) {
      mat[base + c] = tmp[c] * invNorm;
    }
  }

  return mat;
}

/**
 * Initializes `attn_alpha` / `mlp_alpha` raw `[d]` vectors centered around `sigma_0 = 0.0177`
 * with structured per-channel spread giving effective `|alpha_hat| in [0.03, 0.34]` (within
 * `[0.02, 0.35]`). At `DORY_PLANTED_EXTRAPOLATION_SPEC.physicalLayerIdx` (`p = 16`), plants
 * channels `6` (`+1.12`) and `7` (`-1.22`, testing BOTH `.abs()` and `> 1` extrapolation `DR7`).
 */
function sampleResidualAlphaVector(
  hiddenSize: number,
  physicalLayerIdx: number,
  config: DoryMiniConfig,
  prng: DoryPrng,
): Float32Array {
  const raw = new Float32Array(hiddenSize);
  const effToRaw = config.initializerRange / config.ngptAlphaInit;
  const denom = Math.max(1, hiddenSize - 1);

  for (let c = 0; c < hiddenSize; c++) {
    const channelRamp = 0.05 + 0.22 * (c / denom);
    const jitter = prng.nextRange(-0.02, 0.05);
    const effTarget = Math.max(0.03, Math.min(0.34, channelRamp + jitter));
    // Include occasional negative raw sign to exercise `.abs()` cleanly.
    const sign = (physicalLayerIdx + c) % 11 === 5 ? -1 : 1;
    raw[c] = sign * effTarget * effToRaw;
  }

  if (physicalLayerIdx === DORY_PLANTED_EXTRAPOLATION_SPEC.physicalLayerIdx) {
    const [ch0, ch1] = DORY_PLANTED_EXTRAPOLATION_SPEC.channelIndices;
    const [eff0, eff1] = DORY_PLANTED_EXTRAPOLATION_SPEC.effectiveTargets;
    // Channel 6: positive raw -> effective 1.12 in (1.05, 1.25)
    raw[ch0] = eff0 * effToRaw;
    // Channel 7: negative raw -> effective |-1.22| = 1.22 in (1.05, 1.25) (tests `.abs()` + `> 1`)
    raw[ch1] = -eff1 * effToRaw;
  }

  return raw;
}

/**
 * Initializes `sqk`, `s_gate`, or `lm_head.logit_scale` raw vectors with smooth spread
 * strictly within `[0.7x, 1.3x] * sigma_0` (`0.0177`, Audit §11.3).
 */
function sampleSigmaCenteredScaleVector(
  length: number,
  sigma0: number,
  prng: DoryPrng,
): Float32Array {
  const raw = new Float32Array(length);
  const denom = Math.max(1, length - 1);
  for (let i = 0; i < length; i++) {
    const ramp = i / denom; // [0, 1]
    const noise = prng.nextFloat64(); // [0, 1)
    const factor = 0.7 + 0.6 * (0.45 * ramp + 0.55 * noise); // strictly in [0.7, 1.3]
    raw[i] = sigma0 * factor;
  }
  return raw;
}

/**
 * Initializes `suv_gate` / `suv_up` raw `[d_ff]` vectors centered around `1.0` (NOT `0.0177`)
 * WITH deterministic negative signed entries so any spurious `.abs()` is immediately caught.
 */
function sampleSignedSuvVector(
  length: number,
  physicalLayerIdx: number,
  saltOffset: number,
  prng: DoryPrng,
): Float32Array {
  const raw = new Float32Array(length);
  const denom = Math.max(1, length - 1);
  for (let i = 0; i < length; i++) {
    const mag = 0.72 + 0.56 * (0.4 * (i / denom) + 0.6 * prng.nextFloat64()); // [0.72, 1.28]
    const sign = (physicalLayerIdx + i + saltOffset) % 4 === 3 ? -1 : 1;
    raw[i] = sign * mag;
  }
  return raw;
}

// ============================================================================
// CANONICAL TENSOR LISTING, SHA-256 DIGEST, SERIALIZATION & STATS
// ============================================================================

export function listDoryCanonicalTensors(
  params: DoryModelParameters,
): readonly DoryCanonicalWeightTensor[] {
  const { config } = params;
  const d = config.hiddenSize;
  const dFf = config.intermediateSize;
  const qDim = config.qProjOutDim;
  const kvDim = config.kvProjOutDim;
  const vocabSize = config.vocabSize;

  const tensors: DoryCanonicalWeightTensor[] = [];

  tensors.push(
    Object.freeze({
      name: DORY_EMBEDDINGS_TENSOR_NAME,
      transformersPairedName: doryToTransformersPairedTensorName(DORY_EMBEDDINGS_TENSOR_NAME),
      role: 'embeddings',
      physicalLayerIdx: null,
      pairedLayerIdx: null,
      stage: 'bookend',
      scaleFamily: null,
      shape: Object.freeze([vocabSize, d]),
      data: params.embeddings,
    }),
  );

  for (let p = 0; p < params.layers.length; p++) {
    const layer = params.layers[p];
    if (layer.kind === 'attention') {
      const alphaName = doryAttnAlphaTensorName(p);
      const qName = doryAttnProjTensorName(p, 'q_proj');
      const kName = doryAttnProjTensorName(p, 'k_proj');
      const vName = doryAttnProjTensorName(p, 'v_proj');
      const gName = doryAttnProjTensorName(p, 'g_proj');
      const oName = doryAttnProjTensorName(p, 'o_proj');
      const sqkName = doryAttnScaleTensorName(p, 'sqk');
      const sGateName = doryAttnScaleTensorName(p, 's_gate');

      tensors.push(
        Object.freeze({
          name: alphaName,
          transformersPairedName: doryToTransformersPairedTensorName(alphaName),
          role: 'attn_alpha',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: 'residual_alpha',
          shape: Object.freeze([d]),
          data: layer.attnAlpha,
        }),
        Object.freeze({
          name: qName,
          transformersPairedName: doryToTransformersPairedTensorName(qName),
          role: 'q_proj',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: null,
          shape: Object.freeze([qDim, d]),
          data: layer.qProj,
        }),
        Object.freeze({
          name: kName,
          transformersPairedName: doryToTransformersPairedTensorName(kName),
          role: 'k_proj',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: null,
          shape: Object.freeze([kvDim, d]),
          data: layer.kProj,
        }),
        Object.freeze({
          name: vName,
          transformersPairedName: doryToTransformersPairedTensorName(vName),
          role: 'v_proj',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: null,
          shape: Object.freeze([kvDim, d]),
          data: layer.vProj,
        }),
        Object.freeze({
          name: gName,
          transformersPairedName: doryToTransformersPairedTensorName(gName),
          role: 'g_proj',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: null,
          shape: Object.freeze([qDim, d]),
          data: layer.gProj,
        }),
        Object.freeze({
          name: oName,
          transformersPairedName: doryToTransformersPairedTensorName(oName),
          role: 'o_proj',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: null,
          shape: Object.freeze([d, qDim]),
          data: layer.oProj,
        }),
        Object.freeze({
          name: sqkName,
          transformersPairedName: doryToTransformersPairedTensorName(sqkName),
          role: 'sqk',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: 'sqk',
          shape: Object.freeze([qDim]),
          data: layer.sqk,
        }),
        Object.freeze({
          name: sGateName,
          transformersPairedName: doryToTransformersPairedTensorName(sGateName),
          role: 's_gate',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: 's_gate',
          shape: Object.freeze([qDim]),
          data: layer.sGate,
        }),
      );
    } else {
      const alphaName = doryMlpAlphaTensorName(p);
      const gateName = doryMlpProjTensorName(p, 'gate_proj');
      const upName = doryMlpProjTensorName(p, 'up_proj');
      const downName = doryMlpProjTensorName(p, 'down_proj');
      const suvGateName = doryMlpScaleTensorName(p, 'suv_gate');
      const suvUpName = doryMlpScaleTensorName(p, 'suv_up');

      tensors.push(
        Object.freeze({
          name: alphaName,
          transformersPairedName: doryToTransformersPairedTensorName(alphaName),
          role: 'mlp_alpha',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: 'residual_alpha',
          shape: Object.freeze([d]),
          data: layer.mlpAlpha,
        }),
        Object.freeze({
          name: gateName,
          transformersPairedName: doryToTransformersPairedTensorName(gateName),
          role: 'gate_proj',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: null,
          shape: Object.freeze([dFf, d]),
          data: layer.gateProj,
        }),
        Object.freeze({
          name: upName,
          transformersPairedName: doryToTransformersPairedTensorName(upName),
          role: 'up_proj',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: null,
          shape: Object.freeze([dFf, d]),
          data: layer.upProj,
        }),
        Object.freeze({
          name: downName,
          transformersPairedName: doryToTransformersPairedTensorName(downName),
          role: 'down_proj',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: null,
          shape: Object.freeze([d, dFf]),
          data: layer.downProj,
        }),
        Object.freeze({
          name: suvGateName,
          transformersPairedName: doryToTransformersPairedTensorName(suvGateName),
          role: 'suv_gate',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: 'suv',
          shape: Object.freeze([dFf]),
          data: layer.suvGate,
        }),
        Object.freeze({
          name: suvUpName,
          transformersPairedName: doryToTransformersPairedTensorName(suvUpName),
          role: 'suv_up',
          physicalLayerIdx: p,
          pairedLayerIdx: layer.pairedLayerIdx,
          stage: layer.stage,
          scaleFamily: 'suv',
          shape: Object.freeze([dFf]),
          data: layer.suvUp,
        }),
      );
    }
  }

  tensors.push(
    Object.freeze({
      name: DORY_LM_HEAD_TENSOR_NAME,
      transformersPairedName: doryToTransformersPairedTensorName(DORY_LM_HEAD_TENSOR_NAME),
      role: 'lm_head',
      physicalLayerIdx: null,
      pairedLayerIdx: null,
      stage: 'bookend',
      scaleFamily: null,
      shape: Object.freeze([vocabSize, d]),
      data: params.lmHead,
    }),
    Object.freeze({
      name: DORY_LM_HEAD_LOGIT_SCALE_TENSOR_NAME,
      transformersPairedName: doryToTransformersPairedTensorName(
        DORY_LM_HEAD_LOGIT_SCALE_TENSOR_NAME,
      ),
      role: 'logit_scale',
      physicalLayerIdx: null,
      pairedLayerIdx: null,
      stage: 'bookend',
      scaleFamily: 'logit_scale',
      shape: Object.freeze([vocabSize]),
      data: params.logitScale,
    }),
  );

  return Object.freeze(tensors);
}

/**
 * Computes the canonical SHA-256 hex digest over all 507 tensors using the exact
 * `DORY_WEIGHT_DIGEST_MAGIC` binary framing shared with PyTorch 2.15.0 parity (`types.ts:L174-184`).
 */
export function computeDoryWeightDigestHex(params: DoryModelParameters): string {
  const tensors = listDoryCanonicalTensors(params);
  const sha = new Sha256Stream();
  const nulByte = new Uint8Array([0]);

  sha.updateUtf8String(DORY_WEIGHT_DIGEST_MAGIC);
  for (let i = 0; i < tensors.length; i++) {
    const tensor = tensors[i];
    sha.updateUtf8String(tensor.name);
    sha.updateBytes(nulByte);
    sha.updateUint32LE(tensor.shape.length);
    for (let d = 0; d < tensor.shape.length; d++) {
      sha.updateUint32LE(tensor.shape[d]);
    }
    sha.updateFloat32LE(tensor.data);
  }

  return sha.digestHex();
}

export function createDoryModelParameters(
  seed: number = DORY_BACKBONE_SEED,
  config: DoryMiniConfig = DORY_MINI_CONFIG,
): DoryModelParameters {
  const normalizedSeed = seed >>> 0;
  const d = config.hiddenSize;
  const dFf = config.intermediateSize;
  const qDim = config.qProjOutDim;
  const kvDim = config.kvProjOutDim;
  const vocabSize = config.vocabSize;
  const sigma0 = config.initializerRange;

  const embedPrng = createDoryPrng(deriveDorySeed(normalizedSeed, 0x0001));
  const embeddings = sampleRowUnitNormMatrix(vocabSize, d, embedPrng);

  const layers: DorySublayerWeights[] = [];
  for (let p = 0; p < config.numHiddenLayers; p++) {
    const spec = DORY_PHYSICAL_LAYERS[p];
    const layerPrng = createDoryPrng(deriveDorySeed(normalizedSeed, 0x1000 + p));

    if (spec.attentionKind !== null && spec.ropeRegime !== null) {
      const attnAlpha = sampleResidualAlphaVector(d, p, config, layerPrng);
      const qProj = sampleRowUnitNormMatrix(qDim, d, layerPrng);
      const kProj = sampleRowUnitNormMatrix(kvDim, d, layerPrng);
      const vProj = sampleRowUnitNormMatrix(kvDim, d, layerPrng);
      const gProj = sampleRowUnitNormMatrix(qDim, d, layerPrng);
      const oProj = sampleRowUnitNormMatrix(d, qDim, layerPrng);
      const sqk = sampleSigmaCenteredScaleVector(qDim, sigma0, layerPrng);
      const sGate = sampleSigmaCenteredScaleVector(qDim, sigma0, layerPrng);

      const attnLayer: DoryAttentionSublayerWeights = Object.freeze({
        kind: 'attention',
        physicalLayerIdx: p,
        pairedLayerIdx: spec.pairedLayerIdx,
        stage: spec.stage,
        attentionKind: spec.attentionKind,
        ropeRegime: spec.ropeRegime,
        attnAlpha,
        qProj,
        kProj,
        vProj,
        gProj,
        oProj,
        sqk,
        sGate,
      });
      layers.push(attnLayer);
    } else {
      const mlpAlpha = sampleResidualAlphaVector(d, p, config, layerPrng);
      const gateProj = sampleRowUnitNormMatrix(dFf, d, layerPrng);
      const upProj = sampleRowUnitNormMatrix(dFf, d, layerPrng);
      const downProj = sampleRowUnitNormMatrix(d, dFf, layerPrng);
      const suvGate = sampleSignedSuvVector(dFf, p, 0, layerPrng);
      const suvUp = sampleSignedSuvVector(dFf, p, 2, layerPrng);

      const mlpLayer: DoryMlpSublayerWeights = Object.freeze({
        kind: 'mlp',
        physicalLayerIdx: p,
        pairedLayerIdx: spec.pairedLayerIdx,
        stage: spec.stage,
        mlpAlpha,
        gateProj,
        upProj,
        downProj,
        suvGate,
        suvUp,
      });
      layers.push(mlpLayer);
    }
  }

  const headPrng = createDoryPrng(deriveDorySeed(normalizedSeed, 0x9001));
  const lmHead = sampleRowUnitNormMatrix(vocabSize, d, headPrng);
  const logitScale = sampleSigmaCenteredScaleVector(vocabSize, sigma0, headPrng);

  const frozenLayers = Object.freeze(layers);
  const draftParams: DoryModelParameters = {
    modelWeightId: '',
    backboneSeed: normalizedSeed,
    config,
    vocab: DORY_MINI_VOCAB,
    embeddings,
    layers: frozenLayers,
    lmHead,
    logitScale,
  };
  const modelWeightId = computeDoryWeightDigestHex(draftParams);

  return Object.freeze({
    modelWeightId,
    backboneSeed: normalizedSeed,
    config,
    vocab: DORY_MINI_VOCAB,
    embeddings,
    layers: frozenLayers,
    lmHead,
    logitScale,
  });
}

let defaultParametersSingleton: DoryModelParameters | null = null;

/**
 * Returns the canonical singleton `DoryModelParameters` (`seed = 0xd0411004`, `507` tensors /
 * `93,248` F32 floats), shared by reference across all loops and session branches.
 */
export function getDefaultDoryModelParameters(): DoryModelParameters {
  if (defaultParametersSingleton === null) {
    defaultParametersSingleton = createDoryModelParameters(DORY_BACKBONE_SEED, DORY_MINI_CONFIG);
  }
  return defaultParametersSingleton;
}

export function serializeDoryParametersJson(
  params: DoryModelParameters,
): DorySerializedParametersPayload {
  const canonicalTensors = listDoryCanonicalTensors(params);
  let floatCount = 0;
  const tensors: DorySerializedTensorEntry[] = canonicalTensors.map((t) => {
    floatCount += t.data.length;
    return {
      name: t.name,
      shape: [...t.shape],
      data: Array.from(t.data),
    };
  });

  return {
    format: DORY_WEIGHT_DIGEST_MAGIC,
    modelWeightId: params.modelWeightId,
    backboneSeed: params.backboneSeed,
    config: params.config,
    tensorCount: tensors.length,
    floatCount,
    tensors,
  };
}

export function countDoryParameterStats(params: DoryModelParameters): DoryParameterCountSummary {
  const tensors = listDoryCanonicalTensors(params);

  let totalFloats = 0;
  let linearWeightTensors = 0;
  let ngptScaleTensors = 0;
  let bookendMatrixTensors = 0;
  let inputStageTensors = 0;
  let inputStageFloats = 0;
  let recurrentStageTensors = 0;
  let recurrentStageFloats = 0;
  let outputStageTensors = 0;
  let outputStageFloats = 0;
  let bookendTensors = 0;
  let bookendFloats = 0;

  for (let i = 0; i < tensors.length; i++) {
    const t = tensors[i];
    const n = t.data.length;
    totalFloats += n;

    if (t.scaleFamily !== null) {
      ngptScaleTensors++;
    } else if (t.role === 'embeddings' || t.role === 'lm_head') {
      bookendMatrixTensors++;
    } else {
      linearWeightTensors++;
    }

    if (t.stage === 'input') {
      inputStageTensors++;
      inputStageFloats += n;
    } else if (t.stage === 'recurrent') {
      recurrentStageTensors++;
      recurrentStageFloats += n;
    } else if (t.stage === 'output') {
      outputStageTensors++;
      outputStageFloats += n;
    } else {
      bookendTensors++;
      bookendFloats += n;
    }
  }

  return Object.freeze({
    totalTensors: tensors.length,
    totalFloats,
    totalBytes: totalFloats * 4,
    linearWeightTensors,
    ngptScaleTensors,
    bookendMatrixTensors,
    normWeightTensors: 0,
    biasTensors: 0,
    loopAdapterTensors: 0,
    inputStageTensors,
    inputStageFloats,
    recurrentStageTensors,
    recurrentStageFloats,
    outputStageTensors,
    outputStageFloats,
    bookendTensors,
    bookendFloats,
  });
}
