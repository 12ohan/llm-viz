/**
 * Miniature Discrete Emu3 IBQ VQ-VAE (`src/uni/vqvae.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching the pinned
 * Nemotron-Labs-Diffusion-Uni Mini Emu3 VQ-VAE contract (`UNI2 Slice 2B`):
 *   - Exact 113 canonical tensors (`43,387` floats):
 *       * `1` IBQ codebook tensor (`2,048` floats, `[256, 8]`, `std = 0.5`)
 *       * `56` encoder tensors (`19,800` floats, `16x16 -> 8x8 -> 4x4`)
 *       * `56` decoder tensors (`21,539` floats, `4x4 -> 8x8 -> 16x16`)
 *   - Deterministic isolated per-tensor FNV-1a(seed, tensorName) + mulberry32 +
 *     Box-Muller Gaussian streams (default `UNI_MULTIMODAL_SEED = 0x4e4c5532`)
 *   - Exact CHW FP32 Conv2d (with asymmetric bottom/right downsample padding and
 *     2x nearest-neighbor upsampling), FP64-accumulated GroupNorm (`G=4, eps=1e-6`),
 *     SiLU ResBlocks, CHW spatial self-attention, and exact IBQ inner-product
 *     quantization `argmax(z @ E^T)` (`U14`, strict `>` tie-break).
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG calls, or sibling-app imports.
 */

import { softmaxRows } from './attention';
import { silu } from './swiglu';
import {
  UNI_MINI_VQ_BASE_CH,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_MINI_VQ_GROUP_NORM_EPS,
  UNI_MINI_VQ_MID_CH,
  UNI_MINI_VQ_NUM_GROUPS,
  UNI_MINI_VQ_Z_CH,
  UNI_MINI_VQVAE_CODEBOOK_FLOAT_COUNT,
  UNI_MINI_VQVAE_DECODER_FLOAT_COUNT,
  UNI_MINI_VQVAE_ENCODER_FLOAT_COUNT,
  UNI_MINI_VQVAE_TOTAL_FLOAT_COUNT,
  UNI_MINI_VQVAE_TOTAL_TENSOR_COUNT,
  UNI_MULTIMODAL_SEED,
  type UniConv2dWeights,
  type UniEmu3AttnBlockWeights,
  type UniEmu3DecoderWeights,
  type UniEmu3EncoderWeights,
  type UniEmu3ResBlockWeights,
  type UniEmu3VqWeights,
  type UniGroupNormWeights,
  type UniMultimodalSubsystem,
  type UniMultimodalTensorEntry,
  type UniRgbImage,
  type UniVqDecodeTrace,
  type UniVqEncodeTrace,
} from './types';

// ============================================================================
// ISOLATED PER-TENSOR PRNG (identical contract to `src/uni/parameters.ts`)
// ============================================================================

const FNV1A_32_OFFSET = 0x811c9dc5;
const FNV1A_32_PRIME = 0x01000193;

function fnv1a32Seed(baseSeed: number, label: string): number {
  let h = (FNV1A_32_OFFSET ^ (baseSeed >>> 0)) >>> 0;
  h = Math.imul(h, FNV1A_32_PRIME) >>> 0;
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i) & 0xff;
    h = Math.imul(h, FNV1A_32_PRIME) >>> 0;
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createGaussianGenerator(seed: number): () => number {
  const nextUniform = mulberry32(seed);
  let spare: number | null = null;

  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u1 = nextUniform();
    while (u1 <= 1e-12) {
      u1 = nextUniform();
    }
    const u2 = nextUniform();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    const angle = 2.0 * Math.PI * u2;
    spare = mag * Math.sin(angle);
    return mag * Math.cos(angle);
  };
}

function sampleGaussianTensor(
  baseSeed: number,
  tensorName: string,
  length: number,
  std: number,
): Float32Array {
  const gauss = createGaussianGenerator(fnv1a32Seed(baseSeed, tensorName));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = Math.fround(gauss() * std);
  }
  return out;
}

function sampleGroupNormWeightTensor(
  baseSeed: number,
  tensorName: string,
  length: number,
): Float32Array {
  const gauss = createGaussianGenerator(fnv1a32Seed(baseSeed, tensorName));
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const v = 1.0 + 0.05 * gauss();
    out[i] = Math.fround(v < 0.5 ? 0.5 : v);
  }
  return out;
}

// ============================================================================
// WEIGHT FACTORY HELPERS & 113-TENSOR CANONICAL COLLECTOR
// ============================================================================

const BIAS_INIT_STD = 0.02;
const CODEBOOK_INIT_STD = 0.5;
const RESIDUAL_OUT_SCALE = 0.5;

function createConv2dWeights(
  seed: number,
  prefix: string,
  inChannels: number,
  outChannels: number,
  kernelSize: 1 | 3,
  residualScale = 1.0,
): UniConv2dWeights {
  const weightStd =
    (0.25 / Math.sqrt(inChannels * kernelSize * kernelSize)) * residualScale;
  const weightLen = outChannels * inChannels * kernelSize * kernelSize;
  return Object.freeze({
    inChannels,
    outChannels,
    kernelSize,
    weight: sampleGaussianTensor(seed, `${prefix}.weight`, weightLen, weightStd),
    bias: sampleGaussianTensor(seed, `${prefix}.bias`, outChannels, BIAS_INIT_STD),
  });
}

function createGroupNormWeights(
  seed: number,
  prefix: string,
  numChannels: number,
  numGroups = UNI_MINI_VQ_NUM_GROUPS,
): UniGroupNormWeights {
  return Object.freeze({
    numChannels,
    numGroups,
    weight: sampleGroupNormWeightTensor(seed, `${prefix}.weight`, numChannels),
    bias: sampleGaussianTensor(seed, `${prefix}.bias`, numChannels, BIAS_INIT_STD),
  });
}

function createResBlockWeights(
  seed: number,
  prefix: string,
  inChannels: number,
  outChannels: number,
): UniEmu3ResBlockWeights {
  return Object.freeze({
    inChannels,
    outChannels,
    norm1: createGroupNormWeights(seed, `${prefix}.norm1`, inChannels),
    conv1: createConv2dWeights(seed, `${prefix}.conv1`, inChannels, outChannels, 3),
    norm2: createGroupNormWeights(seed, `${prefix}.norm2`, outChannels),
    conv2: createConv2dWeights(
      seed,
      `${prefix}.conv2`,
      outChannels,
      outChannels,
      3,
      RESIDUAL_OUT_SCALE,
    ),
    ninShortcut:
      inChannels !== outChannels
        ? createConv2dWeights(seed, `${prefix}.nin_shortcut`, inChannels, outChannels, 1)
        : null,
  });
}

function createAttnBlockWeights(
  seed: number,
  prefix: string,
  channels: number,
): UniEmu3AttnBlockWeights {
  return Object.freeze({
    channels,
    norm: createGroupNormWeights(seed, `${prefix}.norm`, channels),
    q: createConv2dWeights(seed, `${prefix}.q`, channels, channels, 1),
    k: createConv2dWeights(seed, `${prefix}.k`, channels, channels, 1),
    v: createConv2dWeights(seed, `${prefix}.v`, channels, channels, 1),
    projOut: createConv2dWeights(
      seed,
      `${prefix}.proj_out`,
      channels,
      channels,
      1,
      RESIDUAL_OUT_SCALE,
    ),
  });
}

function pushGroupNormTensors(
  out: UniMultimodalTensorEntry[],
  subsystem: UniMultimodalSubsystem,
  prefix: string,
  gn: UniGroupNormWeights,
): void {
  out.push(
    {
      name: `${prefix}.weight`,
      subsystem,
      shape: [gn.numChannels],
      data: gn.weight,
    },
    {
      name: `${prefix}.bias`,
      subsystem,
      shape: [gn.numChannels],
      data: gn.bias,
    },
  );
}

function pushConv2dTensors(
  out: UniMultimodalTensorEntry[],
  subsystem: UniMultimodalSubsystem,
  prefix: string,
  conv: UniConv2dWeights,
): void {
  out.push(
    {
      name: `${prefix}.weight`,
      subsystem,
      shape: [conv.outChannels, conv.inChannels, conv.kernelSize, conv.kernelSize],
      data: conv.weight,
    },
    {
      name: `${prefix}.bias`,
      subsystem,
      shape: [conv.outChannels],
      data: conv.bias,
    },
  );
}

function pushResBlockTensors(
  out: UniMultimodalTensorEntry[],
  subsystem: UniMultimodalSubsystem,
  prefix: string,
  block: UniEmu3ResBlockWeights,
): void {
  pushGroupNormTensors(out, subsystem, `${prefix}.norm1`, block.norm1);
  pushConv2dTensors(out, subsystem, `${prefix}.conv1`, block.conv1);
  pushGroupNormTensors(out, subsystem, `${prefix}.norm2`, block.norm2);
  pushConv2dTensors(out, subsystem, `${prefix}.conv2`, block.conv2);
  if (block.ninShortcut !== null) {
    pushConv2dTensors(out, subsystem, `${prefix}.nin_shortcut`, block.ninShortcut);
  }
}

function pushAttnBlockTensors(
  out: UniMultimodalTensorEntry[],
  subsystem: UniMultimodalSubsystem,
  prefix: string,
  attn: UniEmu3AttnBlockWeights,
): void {
  pushGroupNormTensors(out, subsystem, `${prefix}.norm`, attn.norm);
  pushConv2dTensors(out, subsystem, `${prefix}.q`, attn.q);
  pushConv2dTensors(out, subsystem, `${prefix}.k`, attn.k);
  pushConv2dTensors(out, subsystem, `${prefix}.v`, attn.v);
  pushConv2dTensors(out, subsystem, `${prefix}.proj_out`, attn.projOut);
}

/**
 * Create all 113 deterministic synthetic weights (`43,387` floats) for the
 * Miniature Emu3 IBQ VQ-VAE (`3 -> 8 -> 16 -> 8` encoder, `[256, 8]` codebook,
 * `8 -> 16 -> 8 -> 3` decoder).
 */
export function createUniEmu3VqWeights(
  seed: number = UNI_MULTIMODAL_SEED,
): UniEmu3VqWeights {
  const s = seed >>> 0;
  const baseCh = UNI_MINI_VQ_BASE_CH; // 8
  const midCh = UNI_MINI_VQ_MID_CH;   // 16
  const zCh = UNI_MINI_VQ_Z_CH;       // 8

  const codebook = sampleGaussianTensor(
    s,
    'encoder.vqvae.quantize.embedding.weight',
    UNI_MINI_VQ_CODEBOOK_SIZE * zCh,
    CODEBOOK_INIT_STD,
  );

  const encoder: UniEmu3EncoderWeights = Object.freeze({
    convIn: createConv2dWeights(s, 'encoder.vqvae.encoder.conv_in', 3, baseCh, 3),
    down0Block: createResBlockWeights(
      s,
      'encoder.vqvae.encoder.down.0.block.0',
      baseCh,
      baseCh,
    ),
    down0Conv: createConv2dWeights(
      s,
      'encoder.vqvae.encoder.down.0.downsample.conv',
      baseCh,
      baseCh,
      3,
    ),
    down1Block: createResBlockWeights(
      s,
      'encoder.vqvae.encoder.down.1.block.0',
      baseCh,
      midCh,
    ),
    down1Conv: createConv2dWeights(
      s,
      'encoder.vqvae.encoder.down.1.downsample.conv',
      midCh,
      midCh,
      3,
    ),
    midBlock1: createResBlockWeights(
      s,
      'encoder.vqvae.encoder.mid.block_1',
      midCh,
      midCh,
    ),
    midAttn1: createAttnBlockWeights(s, 'encoder.vqvae.encoder.mid.attn_1', midCh),
    midBlock2: createResBlockWeights(
      s,
      'encoder.vqvae.encoder.mid.block_2',
      midCh,
      midCh,
    ),
    normOut: createGroupNormWeights(s, 'encoder.vqvae.encoder.norm_out', midCh),
    convOut: createConv2dWeights(s, 'encoder.vqvae.encoder.conv_out', midCh, zCh, 3),
    quantConv: createConv2dWeights(s, 'encoder.vqvae.quant_conv', zCh, zCh, 1),
  });

  const decoder: UniEmu3DecoderWeights = Object.freeze({
    postQuantConv: createConv2dWeights(s, 'encoder.vqvae.post_quant_conv', zCh, zCh, 1),
    convIn: createConv2dWeights(s, 'encoder.vqvae.decoder.conv_in', zCh, midCh, 3),
    midBlock1: createResBlockWeights(
      s,
      'encoder.vqvae.decoder.mid.block_1',
      midCh,
      midCh,
    ),
    midAttn1: createAttnBlockWeights(s, 'encoder.vqvae.decoder.mid.attn_1', midCh),
    midBlock2: createResBlockWeights(
      s,
      'encoder.vqvae.decoder.mid.block_2',
      midCh,
      midCh,
    ),
    up1Block: createResBlockWeights(
      s,
      'encoder.vqvae.decoder.up.1.block.0',
      midCh,
      midCh,
    ),
    up1Conv: createConv2dWeights(
      s,
      'encoder.vqvae.decoder.up.1.upsample.conv',
      midCh,
      midCh,
      3,
    ),
    up0Block: createResBlockWeights(
      s,
      'encoder.vqvae.decoder.up.0.block.0',
      midCh,
      baseCh,
    ),
    up0Conv: createConv2dWeights(
      s,
      'encoder.vqvae.decoder.up.0.upsample.conv',
      baseCh,
      baseCh,
      3,
    ),
    normOut: createGroupNormWeights(s, 'encoder.vqvae.decoder.norm_out', baseCh),
    convOut: createConv2dWeights(s, 'encoder.vqvae.decoder.conv_out', baseCh, 3, 3),
  });

  const weights: UniEmu3VqWeights = Object.freeze({
    seed: s,
    codebook,
    encoder,
    decoder,
  });

  const collected = collectUniVqVaeWeights(weights);
  if (collected.length !== UNI_MINI_VQVAE_TOTAL_TENSOR_COUNT) {
    throw new Error(
      `Expected ${UNI_MINI_VQVAE_TOTAL_TENSOR_COUNT} VQ-VAE tensors, got ${collected.length}`,
    );
  }
  let floatCount = 0;
  for (let i = 0; i < collected.length; i++) {
    floatCount += collected[i].data.length;
  }
  if (floatCount !== UNI_MINI_VQVAE_TOTAL_FLOAT_COUNT) {
    throw new Error(
      `Expected ${UNI_MINI_VQVAE_TOTAL_FLOAT_COUNT} VQ-VAE floats, got ${floatCount}`,
    );
  }

  return weights;
}

/**
 * Collect all 113 Mini Emu3 VQ-VAE tensors (`43,387` floats) in canonical order:
 *   - `1` codebook tensor (`2,048` floats, `'vqvae_codebook'`)
 *   - `56` encoder tensors (`19,800` floats, `'vqvae_encoder'`)
 *   - `56` decoder tensors (`21,539` floats, `'vqvae_decoder'`)
 */
export function collectUniVqVaeWeights(
  weights: UniEmu3VqWeights,
): UniMultimodalTensorEntry[] {
  const entries: UniMultimodalTensorEntry[] = [
    {
      name: 'encoder.vqvae.quantize.embedding.weight',
      subsystem: 'vqvae_codebook',
      shape: [UNI_MINI_VQ_CODEBOOK_SIZE, UNI_MINI_VQ_Z_CH],
      data: weights.codebook,
    },
  ];

  // Encoder (56 tensors)
  const enc = weights.encoder;
  pushConv2dTensors(entries, 'vqvae_encoder', 'encoder.vqvae.encoder.conv_in', enc.convIn);
  pushResBlockTensors(
    entries,
    'vqvae_encoder',
    'encoder.vqvae.encoder.down.0.block.0',
    enc.down0Block,
  );
  pushConv2dTensors(
    entries,
    'vqvae_encoder',
    'encoder.vqvae.encoder.down.0.downsample.conv',
    enc.down0Conv,
  );
  pushResBlockTensors(
    entries,
    'vqvae_encoder',
    'encoder.vqvae.encoder.down.1.block.0',
    enc.down1Block,
  );
  pushConv2dTensors(
    entries,
    'vqvae_encoder',
    'encoder.vqvae.encoder.down.1.downsample.conv',
    enc.down1Conv,
  );
  pushResBlockTensors(
    entries,
    'vqvae_encoder',
    'encoder.vqvae.encoder.mid.block_1',
    enc.midBlock1,
  );
  pushAttnBlockTensors(
    entries,
    'vqvae_encoder',
    'encoder.vqvae.encoder.mid.attn_1',
    enc.midAttn1,
  );
  pushResBlockTensors(
    entries,
    'vqvae_encoder',
    'encoder.vqvae.encoder.mid.block_2',
    enc.midBlock2,
  );
  pushGroupNormTensors(entries, 'vqvae_encoder', 'encoder.vqvae.encoder.norm_out', enc.normOut);
  pushConv2dTensors(entries, 'vqvae_encoder', 'encoder.vqvae.encoder.conv_out', enc.convOut);
  pushConv2dTensors(entries, 'vqvae_encoder', 'encoder.vqvae.quant_conv', enc.quantConv);

  // Decoder (56 tensors)
  const dec = weights.decoder;
  pushConv2dTensors(entries, 'vqvae_decoder', 'encoder.vqvae.post_quant_conv', dec.postQuantConv);
  pushConv2dTensors(entries, 'vqvae_decoder', 'encoder.vqvae.decoder.conv_in', dec.convIn);
  pushResBlockTensors(
    entries,
    'vqvae_decoder',
    'encoder.vqvae.decoder.mid.block_1',
    dec.midBlock1,
  );
  pushAttnBlockTensors(
    entries,
    'vqvae_decoder',
    'encoder.vqvae.decoder.mid.attn_1',
    dec.midAttn1,
  );
  pushResBlockTensors(
    entries,
    'vqvae_decoder',
    'encoder.vqvae.decoder.mid.block_2',
    dec.midBlock2,
  );
  pushResBlockTensors(
    entries,
    'vqvae_decoder',
    'encoder.vqvae.decoder.up.1.block.0',
    dec.up1Block,
  );
  pushConv2dTensors(
    entries,
    'vqvae_decoder',
    'encoder.vqvae.decoder.up.1.upsample.conv',
    dec.up1Conv,
  );
  pushResBlockTensors(
    entries,
    'vqvae_decoder',
    'encoder.vqvae.decoder.up.0.block.0',
    dec.up0Block,
  );
  pushConv2dTensors(
    entries,
    'vqvae_decoder',
    'encoder.vqvae.decoder.up.0.upsample.conv',
    dec.up0Conv,
  );
  pushGroupNormTensors(entries, 'vqvae_decoder', 'encoder.vqvae.decoder.norm_out', dec.normOut);
  pushConv2dTensors(entries, 'vqvae_decoder', 'encoder.vqvae.decoder.conv_out', dec.convOut);

  let codebookFloats = 0;
  let encoderFloats = 0;
  let decoderFloats = 0;
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    if (e.subsystem === 'vqvae_codebook') codebookFloats += e.data.length;
    else if (e.subsystem === 'vqvae_encoder') encoderFloats += e.data.length;
    else if (e.subsystem === 'vqvae_decoder') decoderFloats += e.data.length;
  }
  if (
    codebookFloats !== UNI_MINI_VQVAE_CODEBOOK_FLOAT_COUNT ||
    encoderFloats !== UNI_MINI_VQVAE_ENCODER_FLOAT_COUNT ||
    decoderFloats !== UNI_MINI_VQVAE_DECODER_FLOAT_COUNT
  ) {
    throw new Error(
      `VQ-VAE float count mismatch: codebook=${codebookFloats}, encoder=${encoderFloats}, decoder=${decoderFloats}`,
    );
  }

  return entries;
}

// ============================================================================
// CHW CONV2D, GROUPNORM, RESBLOCK, ATTNBLOCK & UPSAMPLE PRIMITIVES
// ============================================================================

function siluArray(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = silu(x[i]);
  }
  return out;
}

function upsample2xNearestChw(
  x: Float32Array,
  C: number,
  H: number,
  W: number,
): Float32Array {
  const outH = H * 2;
  const outW = W * 2;
  const out = new Float32Array(C * outH * outW);
  for (let c = 0; c < C; c++) {
    const inCBase = c * H * W;
    const outCBase = c * outH * outW;
    for (let h = 0; h < H; h++) {
      const inRow = inCBase + h * W;
      const outRow0 = outCBase + (2 * h) * outW;
      const outRow1 = outCBase + (2 * h + 1) * outW;
      for (let w = 0; w < W; w++) {
        const val = x[inRow + w];
        const col = 2 * w;
        out[outRow0 + col] = val;
        out[outRow0 + col + 1] = val;
        out[outRow1 + col] = val;
        out[outRow1 + col + 1] = val;
      }
    }
  }
  return out;
}

/**
 * 2D convolution on CHW `[inC, H, W]` Float32 tensors matching PyTorch `F.conv2d`.
 * When `asymBottomRightPad` is true, pads `right=1, bottom=1` (`F.pad(x, (0, 1, 0, 1))`)
 * before applying `3x3` `stride=2, padding=0`.
 */
export function conv2dChw(
  x: Float32Array,
  inC: number,
  H: number,
  W: number,
  weights: UniConv2dWeights,
  options?: { stride?: 1 | 2; padding?: 0 | 1; asymBottomRightPad?: boolean },
): Float32Array {
  if (!Number.isInteger(H) || H < 1 || !Number.isInteger(W) || W < 1) {
    throw new RangeError(`H (${H}) and W (${W}) must be integers >= 1`);
  }
  if (inC !== weights.inChannels) {
    throw new RangeError(
      `inC (${inC}) must match weights.inChannels (${weights.inChannels})`,
    );
  }
  if (x.length < inC * H * W) {
    throw new RangeError(
      `x length (${x.length}) is smaller than inC * H * W (${inC * H * W})`,
    );
  }

  const outC = weights.outChannels;
  const k = weights.kernelSize;
  const stride = options?.stride ?? 1;
  const asymBottomRightPad = options?.asymBottomRightPad ?? false;
  const padding = options?.padding ?? (asymBottomRightPad ? 0 : k === 3 ? 1 : 0);

  const padTop = asymBottomRightPad ? 0 : padding;
  const padLeft = asymBottomRightPad ? 0 : padding;
  const padBottom = asymBottomRightPad ? 1 : padding;
  const padRight = asymBottomRightPad ? 1 : padding;

  const effH = H + padTop + padBottom;
  const effW = W + padLeft + padRight;
  const outH = Math.floor((effH - k) / stride) + 1;
  const outW = Math.floor((effW - k) / stride) + 1;
  if (outH < 1 || outW < 1) {
    throw new RangeError(`Invalid conv2d output spatial dimensions (${outH}x${outW})`);
  }

  const out = new Float32Array(outC * outH * outW);
  const wData = weights.weight;
  const bData = weights.bias;
  const hw = H * W;
  const outHw = outH * outW;

  for (let oc = 0; oc < outC; oc++) {
    const bias = bData[oc];
    const ocWeightBase = oc * inC * k * k;
    const outPlaneBase = oc * outHw;

    for (let oh = 0; oh < outH; oh++) {
      const ihOrigin = oh * stride - padTop;
      const outRowBase = outPlaneBase + oh * outW;

      for (let ow = 0; ow < outW; ow++) {
        const iwOrigin = ow * stride - padLeft;
        let acc = 0;

        for (let ic = 0; ic < inC; ic++) {
          const inPlaneBase = ic * hw;
          const icWeightBase = ocWeightBase + ic * k * k;

          for (let kh = 0; kh < k; kh++) {
            const ih = ihOrigin + kh;
            if (ih < 0 || ih >= H) continue;
            const inRowBase = inPlaneBase + ih * W;
            const kRowBase = icWeightBase + kh * k;

            for (let kw = 0; kw < k; kw++) {
              const iw = iwOrigin + kw;
              if (iw < 0 || iw >= W) continue;
              acc += x[inRowBase + iw] * wData[kRowBase + kw];
            }
          }
        }

        out[outRowBase + ow] = Math.fround(acc + bias);
      }
    }
  }

  return out;
}

/**
 * CHW GroupNorm matching PyTorch `F.group_norm` with FP64 mean and variance
 * accumulation over `(C / numGroups) * H * W`.
 */
export function groupNormChw(
  x: Float32Array,
  C: number,
  H: number,
  W: number,
  weights: UniGroupNormWeights,
  eps: number = UNI_MINI_VQ_GROUP_NORM_EPS,
): Float32Array {
  if (!Number.isInteger(H) || H < 1 || !Number.isInteger(W) || W < 1) {
    throw new RangeError(`H (${H}) and W (${W}) must be integers >= 1`);
  }
  if (C !== weights.numChannels) {
    throw new RangeError(
      `C (${C}) must match weights.numChannels (${weights.numChannels})`,
    );
  }
  const numGroups = weights.numGroups;
  if (numGroups < 1 || C % numGroups !== 0) {
    throw new RangeError(`C (${C}) must be divisible by numGroups (${numGroups})`);
  }
  if (x.length < C * H * W) {
    throw new RangeError(
      `x length (${x.length}) is smaller than C * H * W (${C * H * W})`,
    );
  }

  const hw = H * W;
  const channelsPerGroup = C / numGroups;
  const groupSize = channelsPerGroup * hw;
  const out = new Float32Array(C * hw);

  for (let g = 0; g < numGroups; g++) {
    const cStart = g * channelsPerGroup;
    const cEnd = cStart + channelsPerGroup;

    let sum = 0;
    for (let c = cStart; c < cEnd; c++) {
      const base = c * hw;
      for (let i = 0; i < hw; i++) {
        sum += x[base + i];
      }
    }
    const mean = sum / groupSize;

    let varSum = 0;
    for (let c = cStart; c < cEnd; c++) {
      const base = c * hw;
      for (let i = 0; i < hw; i++) {
        const diff = x[base + i] - mean;
        varSum += diff * diff;
      }
    }
    const variance = varSum / groupSize;
    const invStd = 1.0 / Math.sqrt(variance + eps);

    for (let c = cStart; c < cEnd; c++) {
      const gamma = weights.weight[c];
      const beta = weights.bias[c];
      const base = c * hw;
      for (let i = 0; i < hw; i++) {
        const normed = Math.fround((x[base + i] - mean) * invStd);
        out[base + i] = Math.fround(normed * gamma + beta);
      }
    }
  }

  return out;
}

/**
 * Emu3 VQ ResNet block:
 *   `GroupNorm -> SiLU -> Conv1 -> GroupNorm -> SiLU -> Conv2 + (ninShortcut ? ninShortcut(x) : x)`
 */
export function runEmu3ResBlock(
  x: Float32Array,
  H: number,
  W: number,
  weights: UniEmu3ResBlockWeights,
): Float32Array {
  let h = groupNormChw(x, weights.inChannels, H, W, weights.norm1, UNI_MINI_VQ_GROUP_NORM_EPS);
  h = siluArray(h);
  h = conv2dChw(h, weights.inChannels, H, W, weights.conv1, { stride: 1, padding: 1 });
  h = groupNormChw(h, weights.outChannels, H, W, weights.norm2, UNI_MINI_VQ_GROUP_NORM_EPS);
  h = siluArray(h);
  h = conv2dChw(h, weights.outChannels, H, W, weights.conv2, { stride: 1, padding: 1 });

  const residual =
    weights.ninShortcut !== null
      ? conv2dChw(x, weights.inChannels, H, W, weights.ninShortcut, { stride: 1, padding: 0 })
      : x;

  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    out[i] = Math.fround(residual[i] + h[i]);
  }
  return out;
}

/**
 * Emu3 VQ single-head spatial self-attention block (`[C, H, W]`):
 *   `h = GroupNorm(x) -> q,k,v = 1x1 conv(h) -> softmax(q^T k / sqrt(C)) -> proj_out -> + x`
 */
export function runEmu3AttnBlock(
  x: Float32Array,
  H: number,
  W: number,
  weights: UniEmu3AttnBlockWeights,
): Float32Array {
  const C = weights.channels;
  const hw = H * W;

  const hNorm = groupNormChw(x, C, H, W, weights.norm, UNI_MINI_VQ_GROUP_NORM_EPS);
  const q = conv2dChw(hNorm, C, H, W, weights.q, { stride: 1, padding: 0 });
  const k = conv2dChw(hNorm, C, H, W, weights.k, { stride: 1, padding: 0 });
  const v = conv2dChw(hNorm, C, H, W, weights.v, { stride: 1, padding: 0 });

  const invSqrtC = 1.0 / Math.sqrt(C);
  const scores = new Float32Array(hw * hw);

  for (let i = 0; i < hw; i++) {
    const rowBase = i * hw;
    for (let j = 0; j < hw; j++) {
      let dot = 0;
      for (let c = 0; c < C; c++) {
        dot += q[c * hw + i] * k[c * hw + j];
      }
      scores[rowBase + j] = Math.fround(dot * invSqrtC);
    }
  }

  const probs = softmaxRows(scores, hw, hw);
  const attnOut = new Float32Array(C * hw);

  for (let c = 0; c < C; c++) {
    const cBase = c * hw;
    for (let i = 0; i < hw; i++) {
      const pBase = i * hw;
      let acc = 0;
      for (let j = 0; j < hw; j++) {
        acc += probs[pBase + j] * v[cBase + j];
      }
      attnOut[cBase + i] = Math.fround(acc);
    }
  }

  const proj = conv2dChw(attnOut, C, H, W, weights.projOut, { stride: 1, padding: 0 });
  const out = new Float32Array(C * hw);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.fround(x[i] + proj[i]);
  }
  return out;
}

// ============================================================================
// IBQ DOT-PRODUCT QUANTIZER (`U14`: argmax(z @ E^T)) & ENCODE / DECODE
// ============================================================================

export interface UniIbqQuantizeResult {
  readonly dotScores: Float32Array;
  readonly codes: Int32Array;
  readonly zQuantized: Float32Array;
}

/**
 * Exact IBQ inner-product quantizer (`U14`):
 * Computes `dotScores[p, k] = z[:, p] · E[k, :]` and picks `argmax_k` with
 * strict `>` tie-breaking matching PyTorch `.argmax(-1)`.
 */
export function quantizeIbqDotProduct(
  zContinuous: Float32Array,
  vqGridRows: number,
  vqGridCols: number,
  codebook: Float32Array,
  latentDim: number = UNI_MINI_VQ_Z_CH,
  codebookSize: number = UNI_MINI_VQ_CODEBOOK_SIZE,
): UniIbqQuantizeResult {
  if (!Number.isInteger(vqGridRows) || vqGridRows < 1 || !Number.isInteger(vqGridCols) || vqGridCols < 1) {
    throw new RangeError(`Invalid VQ grid (${vqGridRows}x${vqGridCols})`);
  }
  const numCells = vqGridRows * vqGridCols;
  if (zContinuous.length < latentDim * numCells) {
    throw new RangeError(
      `zContinuous length (${zContinuous.length}) is smaller than latentDim * numCells (${latentDim * numCells})`,
    );
  }
  if (codebook.length < codebookSize * latentDim) {
    throw new RangeError(
      `codebook length (${codebook.length}) is smaller than codebookSize * latentDim (${codebookSize * latentDim})`,
    );
  }

  const dotScores = new Float32Array(numCells * codebookSize);
  const codes = new Int32Array(numCells);
  const zQuantized = new Float32Array(latentDim * numCells);

  for (let p = 0; p < numCells; p++) {
    const scoreBase = p * codebookSize;
    let bestCode = 0;
    let bestScore = -Infinity;

    for (let k = 0; k < codebookSize; k++) {
      const cbBase = k * latentDim;
      let dot = 0;
      for (let d = 0; d < latentDim; d++) {
        dot += zContinuous[d * numCells + p] * codebook[cbBase + d];
      }
      const s = Math.fround(dot);
      dotScores[scoreBase + k] = s;
      if (s > bestScore) {
        bestScore = s;
        bestCode = k;
      }
    }

    codes[p] = bestCode;
    const bestCbBase = bestCode * latentDim;
    for (let d = 0; d < latentDim; d++) {
      zQuantized[d * numCells + p] = codebook[bestCbBase + d];
    }
  }

  return { dotScores, codes, zQuantized };
}

/**
 * Encode a `[3, H, W]` RGB image (`[0, 1]`, `H % 4 === 0 && W % 4 === 0`)
 * into `[vqGridRows * vqGridCols]` IBQ codebook indices via Mini Emu3 VQ-VAE.
 */
export function encodeMiniEmu3Vq(
  image: UniRgbImage,
  weights: UniEmu3VqWeights,
): UniVqEncodeTrace {
  const H = image.height;
  const W = image.width;
  if (!Number.isInteger(H) || !Number.isInteger(W) || H < 4 || W < 4 || H % 4 !== 0 || W % 4 !== 0) {
    throw new RangeError(
      `Image dimensions (${W}x${H}) must be multiples of 4 and >= 4`,
    );
  }
  if (image.data.length < 3 * H * W) {
    throw new RangeError(
      `image.data length (${image.data.length}) is smaller than 3 * H * W (${3 * H * W})`,
    );
  }

  const normalizedInput = new Float32Array(3 * H * W);
  for (let i = 0; i < normalizedInput.length; i++) {
    normalizedInput[i] = Math.fround(2.0 * image.data[i] - 1.0);
  }

  const enc = weights.encoder;
  const baseCh = UNI_MINI_VQ_BASE_CH;
  const midCh = UNI_MINI_VQ_MID_CH;
  const zCh = UNI_MINI_VQ_Z_CH;

  let h = conv2dChw(normalizedInput, 3, H, W, enc.convIn, { stride: 1, padding: 1 });
  h = runEmu3ResBlock(h, H, W, enc.down0Block);
  h = conv2dChw(h, baseCh, H, W, enc.down0Conv, {
    stride: 2,
    padding: 0,
    asymBottomRightPad: true,
  });

  const h2 = H / 2;
  const w2 = W / 2;
  h = runEmu3ResBlock(h, h2, w2, enc.down1Block);
  h = conv2dChw(h, midCh, h2, w2, enc.down1Conv, {
    stride: 2,
    padding: 0,
    asymBottomRightPad: true,
  });

  const vqGridRows = H / 4;
  const vqGridCols = W / 4;
  h = runEmu3ResBlock(h, vqGridRows, vqGridCols, enc.midBlock1);
  h = runEmu3AttnBlock(h, vqGridRows, vqGridCols, enc.midAttn1);
  h = runEmu3ResBlock(h, vqGridRows, vqGridCols, enc.midBlock2);
  h = groupNormChw(h, midCh, vqGridRows, vqGridCols, enc.normOut, UNI_MINI_VQ_GROUP_NORM_EPS);
  h = siluArray(h);
  h = conv2dChw(h, midCh, vqGridRows, vqGridCols, enc.convOut, { stride: 1, padding: 1 });
  const zContinuous = conv2dChw(h, zCh, vqGridRows, vqGridCols, enc.quantConv, {
    stride: 1,
    padding: 0,
  });

  const { dotScores, codes, zQuantized } = quantizeIbqDotProduct(
    zContinuous,
    vqGridRows,
    vqGridCols,
    weights.codebook,
    zCh,
    UNI_MINI_VQ_CODEBOOK_SIZE,
  );

  return {
    inputWidth: W,
    inputHeight: H,
    vqGridRows,
    vqGridCols,
    normalizedInput,
    zContinuous,
    dotScores,
    codes,
    zQuantized,
  };
}

/**
 * Decode `[vqGridRows * vqGridCols]` VQ codebook indices back into a `[3, 4*R, 4*C]`
 * RGB image clamped to `[0, 1]`.
 */
export function decodeMiniEmu3Vq(
  codes: ArrayLike<number>,
  vqGridRows: number,
  vqGridCols: number,
  weights: UniEmu3VqWeights,
): UniVqDecodeTrace {
  if (!Number.isInteger(vqGridRows) || vqGridRows < 1 || !Number.isInteger(vqGridCols) || vqGridCols < 1) {
    throw new RangeError(`Invalid VQ grid (${vqGridRows}x${vqGridCols})`);
  }
  const numCells = vqGridRows * vqGridCols;
  if (codes.length !== numCells) {
    throw new RangeError(
      `codes length (${codes.length}) must equal vqGridRows * vqGridCols (${numCells})`,
    );
  }

  const zCh = UNI_MINI_VQ_Z_CH;
  const baseCh = UNI_MINI_VQ_BASE_CH;
  const midCh = UNI_MINI_VQ_MID_CH;

  const codesInt32 = new Int32Array(numCells);
  const zQuantized = new Float32Array(zCh * numCells);

  for (let p = 0; p < numCells; p++) {
    const code = codes[p];
    if (!Number.isInteger(code) || code < 0 || code >= UNI_MINI_VQ_CODEBOOK_SIZE) {
      throw new RangeError(
        `VQ code out of range at index ${p}: ${String(code)} (expected 0..${UNI_MINI_VQ_CODEBOOK_SIZE - 1})`,
      );
    }
    codesInt32[p] = code;
    const cbBase = code * zCh;
    for (let d = 0; d < zCh; d++) {
      zQuantized[d * numCells + p] = weights.codebook[cbBase + d];
    }
  }

  const dec = weights.decoder;
  let h = conv2dChw(zQuantized, zCh, vqGridRows, vqGridCols, dec.postQuantConv, {
    stride: 1,
    padding: 0,
  });
  h = conv2dChw(h, zCh, vqGridRows, vqGridCols, dec.convIn, { stride: 1, padding: 1 });
  h = runEmu3ResBlock(h, vqGridRows, vqGridCols, dec.midBlock1);
  h = runEmu3AttnBlock(h, vqGridRows, vqGridCols, dec.midAttn1);
  h = runEmu3ResBlock(h, vqGridRows, vqGridCols, dec.midBlock2);
  h = runEmu3ResBlock(h, vqGridRows, vqGridCols, dec.up1Block);

  const h2 = vqGridRows * 2;
  const w2 = vqGridCols * 2;
  h = upsample2xNearestChw(h, midCh, vqGridRows, vqGridCols);
  h = conv2dChw(h, midCh, h2, w2, dec.up1Conv, { stride: 1, padding: 1 });
  h = runEmu3ResBlock(h, h2, w2, dec.up0Block);

  const outputHeight = vqGridRows * 4;
  const outputWidth = vqGridCols * 4;
  h = upsample2xNearestChw(h, baseCh, h2, w2);
  h = conv2dChw(h, baseCh, outputHeight, outputWidth, dec.up0Conv, {
    stride: 1,
    padding: 1,
  });
  h = groupNormChw(h, baseCh, outputHeight, outputWidth, dec.normOut, UNI_MINI_VQ_GROUP_NORM_EPS);
  h = siluArray(h);
  const rawOutput = conv2dChw(h, baseCh, outputHeight, outputWidth, dec.convOut, {
    stride: 1,
    padding: 1,
  });

  const clamped = new Float32Array(rawOutput.length);
  for (let i = 0; i < rawOutput.length; i++) {
    const v = Math.fround((rawOutput[i] + 1.0) * 0.5);
    clamped[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }

  return {
    vqGridRows,
    vqGridCols,
    outputWidth,
    outputHeight,
    codes: codesInt32,
    zQuantized,
    rawOutput,
    rgbImage: {
      width: outputWidth,
      height: outputHeight,
      data: clamped,
    },
  };
}
