/**
 * 256×256 RG32F Value Atlas Packer for 32Q/8KV RoPE+YaRN+LoRA GQA Cutaway
 * (`src/diffusion/scene/gqaRopeAtlas.ts`)
 *
 * Packs ALL 32 Q heads, ALL 8 KV heads, ALL 8 KV-group selected Score & Softmax
 * plates (with explicit D24 `G ∈ {0.0, 1.0}` mask flags), YaRN `cos`/`sin`/`inv_freq`,
 * Llama-4 query temperature scales, and `o_proj` LoRA `A/hidden/B/delta` into a
 * single disjoint `256 × 256` (`512 KiB`) `RG32F` WebGL2 texture supporting
 * `S <= 16` query rows and `K <= 32` KV columns with ZERO overlaps.
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`,
 * or `src/twotower/` / `src/math/gqaSim` / `src/math/moeSim` imports.
 */

import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fAtlasWriter,
  type Rg32fValueAtlas,
} from '../../scene/atlasPacker';
import { linearProject } from '../attention';
import { dualHashHex } from '../prng';
import {
  computeLlama4AttnScale,
  computeRopeCosSin,
  computeYarnInvFreq,
} from '../rope';
import type {
  DiffusionAttentionMaskMode,
  DiffusionLayerTrace,
  DiffusionLayerWeights,
  DiffusionMiniConfig,
} from '../types';
import { auditAtlasRegions } from './provenance';
import type {
  DiffusionAtlasLayoutAudit,
  DiffusionAtlasRegionBox,
} from './types';

export const DIFFUSION_GQA_ATLAS_WIDTH = 256 as const;
export const DIFFUSION_GQA_ATLAS_HEIGHT = 256 as const;
export const DIFFUSION_GQA_MAX_QUERY = 16 as const;
export const DIFFUSION_GQA_MAX_KEY = 32 as const;

export interface DiffusionGqaRopeAtlasOptions {
  readonly layerIndex: number;
  readonly selectedQHead: number;
  readonly selectedKvHead: number;
  readonly positionIds: Int32Array;
  readonly cachePositions: Int32Array;
  readonly keyPositionIds: Int32Array;
  readonly maskMode: DiffusionAttentionMaskMode;
  readonly loraEnabled: boolean;
  readonly snapshotDigest: string;
  /** Optional exact residual checkpoint `input_x [S, d]`; reconstructed if omitted. */
  readonly inputHiddenStates?: Float32Array;
}

export interface DiffusionGqaRopeAtlasResult {
  readonly atlas: Rg32fValueAtlas;
  readonly audit: DiffusionAtlasLayoutAudit;
  readonly slices: Readonly<Record<string, AtlasSlice>>;
  readonly scoreSlices: readonly AtlasSlice[];
  readonly softmaxSlices: readonly AtlasSlice[];
  readonly selectedQHeadsPerGroup: readonly number[];
  readonly queryTokens: number;
  readonly keyTokens: number;
  readonly maskedScoreCellCount: number;
  readonly maskedSoftmaxCellCount: number;
  readonly atlasKey: string;
}

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(v)));
}

function packRowMajorMatrix(
  writer: Rg32fAtlasWriter,
  regions: DiffusionAtlasRegionBox[],
  id: string,
  x0: number,
  y0: number,
  rows: number,
  cols: number,
  data: Float32Array,
  rowStride: number,
  mode: 0 | 1,
  scaleOverride?: number,
): AtlasSlice {
  let maxAbs = 0.25;
  for (let r = 0; r < rows; r++) {
    const base = r * rowStride;
    for (let c = 0; c < cols; c++) {
      const v = data[base + c] ?? 0;
      if (Number.isFinite(v)) {
        const a = Math.abs(v);
        if (a > maxAbs) maxAbs = a;
      }
      writer.setTexel(x0 + c, y0 + r, v, false);
    }
  }

  regions.push({
    id,
    x0,
    y0,
    cols,
    rows,
    mode,
    isMaskOrScore: false,
  });

  return {
    rect: writer.makeRect(x0, y0, cols, rows),
    scale: scaleOverride ?? 1.0 / maxAbs,
    mode,
  };
}

/** Pack `[outDim, inDim]` weight transposed into `[rows=inDim, cols=outDim]` for drafting layout. */
function packTransposedWeightMatrix(
  writer: Rg32fAtlasWriter,
  regions: DiffusionAtlasRegionBox[],
  id: string,
  x0: number,
  y0: number,
  inDim: number,
  outDim: number,
  weightOutIn: Float32Array,
): AtlasSlice {
  let maxAbs = 0.25;
  for (let r = 0; r < inDim; r++) {
    for (let c = 0; c < outDim; c++) {
      const v = weightOutIn[c * inDim + r] ?? 0;
      if (Number.isFinite(v)) {
        const a = Math.abs(v);
        if (a > maxAbs) maxAbs = a;
      }
      writer.setTexel(x0 + c, y0 + r, v, false);
    }
  }

  regions.push({
    id,
    x0,
    y0,
    cols: outDim,
    rows: inDim,
    mode: 0,
    isMaskOrScore: false,
  });

  return {
    rect: writer.makeRect(x0, y0, outDim, inDim),
    scale: 1.0 / maxAbs,
    mode: 0,
  };
}

export function buildDiffusionGqaRopeAtlas(
  trace: DiffusionLayerTrace,
  layerWeights: DiffusionLayerWeights,
  config: DiffusionMiniConfig,
  options: DiffusionGqaRopeAtlasOptions,
): DiffusionGqaRopeAtlasResult {
  const attn = trace.attention;
  const d = config.hiddenSize; // 8
  const qDim = config.qProjOutDim; // 256
  const kvDim = config.kvProjOutDim; // 64
  const nQ = config.numAttentionHeads; // 32
  const nKv = config.numKeyValueHeads; // 8
  const kvGroupSize = config.kvGroupSize; // 4
  const loraRank = config.loraRank; // 4
  const halfDim = config.headDim / 2; // 4

  const S = clampInt(attn.seqLen, 1, DIFFUSION_GQA_MAX_QUERY);
  const K = clampInt(attn.kvLen, 1, DIFFUSION_GQA_MAX_KEY);
  const fullS = attn.seqLen;
  const fullK = attn.kvLen;

  const clampedLayer = clampInt(options.layerIndex, 0, config.numLayers - 1);
  const clampedQHead = clampInt(options.selectedQHead, 0, nQ - 1);
  const clampedKvHead = clampInt(options.selectedKvHead, 0, nKv - 1);

  const selectedQHeadsPerGroup: number[] = new Array<number>(nKv);
  for (let g = 0; g < nKv; g++) {
    selectedQHeadsPerGroup[g] = g === clampedKvHead ? clampedQHead : g * kvGroupSize;
  }

  // Resolve exact `input_x [S, 8]`
  let inputX: Float32Array;
  if (options.inputHiddenStates && options.inputHiddenStates.length >= S * d) {
    inputX = options.inputHiddenStates;
  } else {
    inputX = new Float32Array(S * d);
    for (let i = 0; i < S * d; i++) {
      inputX[i] = Math.fround(
        (trace.postAttentionResidual[i] ?? 0) - (attn.output[i] ?? 0),
      );
    }
  }

  // Resolve LoRA branch tensors: use trace tensors when present, or compute exact
  // draft adapter delta when `options.loraEnabled` is requested on a verify checkpoint.
  let loraHidden: Float32Array;
  let loraDelta: Float32Array;
  if (attn.loraHidden && attn.loraDelta) {
    loraHidden = attn.loraHidden;
    loraDelta = attn.loraDelta;
  } else if (options.loraEnabled) {
    loraHidden = linearProject(
      attn.attnConcat,
      S,
      qDim,
      layerWeights.lora.loraA,
      loraRank,
    );
    const loraRaw = linearProject(
      loraHidden,
      S,
      loraRank,
      layerWeights.lora.loraB,
      d,
    );
    loraDelta = new Float32Array(S * d);
    for (let i = 0; i < S * d; i++) {
      loraDelta[i] = Math.fround(config.loraScale * loraRaw[i]!);
    }
  } else {
    loraHidden = new Float32Array(S * loraRank);
    loraDelta = new Float32Array(S * d);
  }

  // Precompute 32-position YaRN cos/sin (planes 0..3) and Llama-4 scales (pos 0..31)
  const yarn = computeYarnInvFreq(config.rope);
  const allPos32 = new Int32Array(DIFFUSION_GQA_MAX_KEY);
  for (let p = 0; p < DIFFUSION_GQA_MAX_KEY; p++) {
    allPos32[p] = p;
  }
  const cosSin32 = computeRopeCosSin(
    yarn.invFreq,
    allPos32,
    yarn.attentionScaling,
    config.headDim,
  );
  const ropeCos32x4 = new Float32Array(DIFFUSION_GQA_MAX_KEY * halfDim);
  const ropeSin32x4 = new Float32Array(DIFFUSION_GQA_MAX_KEY * halfDim);
  const llama4Scale32x1 = new Float32Array(DIFFUSION_GQA_MAX_KEY);
  for (let p = 0; p < DIFFUSION_GQA_MAX_KEY; p++) {
    for (let plane = 0; plane < halfDim; plane++) {
      ropeCos32x4[p * halfDim + plane] = cosSin32.cos[p * config.headDim + plane]!;
      ropeSin32x4[p * halfDim + plane] = cosSin32.sin[p * config.headDim + plane]!;
    }
    llama4Scale32x1[p] = Math.fround(
      computeLlama4AttnScale(
        p,
        config.rope.llama4Beta,
        config.rope.originalMaxPositionEmbeddings,
      ),
    );
  }

  const writer = createRg32fAtlas(
    DIFFUSION_GQA_ATLAS_WIDTH,
    DIFFUSION_GQA_ATLAS_HEIGHT,
  );
  const regions: DiffusionAtlasRegionBox[] = [];
  const slices: Record<string, AtlasSlice> = {};

  // ==========================================================================
  // BAND 1: Full-width 256-col matrices (`x = 0..255`, `y = 0..83` max)
  // ==========================================================================
  slices['w_q'] = packTransposedWeightMatrix(
    writer,
    regions,
    'w_q',
    0,
    0,
    d,
    qDim,
    layerWeights.selfAttn.qProj,
  ); // y: 0..7
  slices['w_o'] = packRowMajorMatrix(
    writer,
    regions,
    'w_o',
    0,
    8,
    d,
    qDim,
    layerWeights.selfAttn.oProj,
    qDim,
    0,
  ); // y: 8..15
  slices['lora_a'] = packRowMajorMatrix(
    writer,
    regions,
    'lora_a',
    0,
    16,
    loraRank,
    qDim,
    layerWeights.lora.loraA,
    qDim,
    0,
  ); // y: 16..19

  let yCursor = 20;
  slices['q_pre'] = packRowMajorMatrix(
    writer,
    regions,
    'q_pre',
    0,
    yCursor,
    S,
    qDim,
    attn.qPre,
    qDim,
    0,
  );
  yCursor += S;

  slices['q_rot'] = packRowMajorMatrix(
    writer,
    regions,
    'q_rot',
    0,
    yCursor,
    S,
    qDim,
    attn.qRot,
    qDim,
    0,
  );
  yCursor += S;

  slices['q_scaled'] = packRowMajorMatrix(
    writer,
    regions,
    'q_scaled',
    0,
    yCursor,
    S,
    qDim,
    attn.qScaled,
    qDim,
    0,
  );
  yCursor += S;

  slices['attn_concat'] = packRowMajorMatrix(
    writer,
    regions,
    'attn_concat',
    0,
    yCursor,
    S,
    qDim,
    attn.attnConcat,
    qDim,
    0,
  );
  yCursor += S; // max 84 when S = 16

  // ==========================================================================
  // BAND 2: 64-col KV matrices (`x = 0..255`, `y = band2Y .. band2Y + max(K, 2*S, 16) - 1`)
  // ==========================================================================
  const band2Y = yCursor;
  slices['k_full'] = packRowMajorMatrix(
    writer,
    regions,
    'k_full',
    0,
    band2Y,
    K,
    kvDim,
    attn.kFull,
    kvDim,
    0,
  );
  slices['v_full'] = packRowMajorMatrix(
    writer,
    regions,
    'v_full',
    64,
    band2Y,
    K,
    kvDim,
    attn.vFull,
    kvDim,
    0,
  );
  slices['k_pre'] = packRowMajorMatrix(
    writer,
    regions,
    'k_pre',
    128,
    band2Y,
    S,
    kvDim,
    attn.kPre,
    kvDim,
    0,
  );
  slices['k_rot'] = packRowMajorMatrix(
    writer,
    regions,
    'k_rot',
    128,
    band2Y + S,
    S,
    kvDim,
    attn.kRot,
    kvDim,
    0,
  );
  slices['w_k'] = packTransposedWeightMatrix(
    writer,
    regions,
    'w_k',
    192,
    band2Y,
    d,
    kvDim,
    layerWeights.selfAttn.kProj,
  );
  slices['w_v'] = packTransposedWeightMatrix(
    writer,
    regions,
    'w_v',
    192,
    band2Y + d,
    d,
    kvDim,
    layerWeights.selfAttn.vProj,
  );

  const band2Height = Math.max(K, 2 * S, 2 * d);
  const band3Y = band2Y + band2Height; // max 84 + 32 = 116

  // ==========================================================================
  // BAND 3: Narrow activation, LoRA & RoPE tables (`y = band3Y .. band3Y + 31`)
  // ==========================================================================
  slices['input_x'] = packRowMajorMatrix(
    writer,
    regions,
    'input_x',
    0,
    band3Y,
    S,
    d,
    inputX,
    d,
    0,
  );
  slices['rmsnorm_u'] = packRowMajorMatrix(
    writer,
    regions,
    'rmsnorm_u',
    8,
    band3Y,
    S,
    d,
    trace.inputNorm,
    d,
    0,
  );
  slices['o_proj_base'] = packRowMajorMatrix(
    writer,
    regions,
    'o_proj_base',
    16,
    band3Y,
    S,
    d,
    attn.oProjBase,
    d,
    0,
  );
  slices['lora_delta'] = packRowMajorMatrix(
    writer,
    regions,
    'lora_delta',
    24,
    band3Y,
    S,
    d,
    loraDelta,
    d,
    0,
  );
  slices['attn_out'] = packRowMajorMatrix(
    writer,
    regions,
    'attn_out',
    32,
    band3Y,
    S,
    d,
    attn.output,
    d,
    0,
  );
  slices['post_attn_resid'] = packRowMajorMatrix(
    writer,
    regions,
    'post_attn_resid',
    40,
    band3Y,
    S,
    d,
    trace.postAttentionResidual,
    d,
    0,
  );
  slices['lora_hidden'] = packRowMajorMatrix(
    writer,
    regions,
    'lora_hidden',
    48,
    band3Y,
    S,
    loraRank,
    loraHidden,
    loraRank,
    0,
  );
  slices['lora_b'] = packRowMajorMatrix(
    writer,
    regions,
    'lora_b',
    52,
    band3Y,
    d,
    loraRank,
    layerWeights.lora.loraB,
    loraRank,
    0,
  );
  slices['rope_cos'] = packRowMajorMatrix(
    writer,
    regions,
    'rope_cos',
    56,
    band3Y,
    DIFFUSION_GQA_MAX_KEY,
    halfDim,
    ropeCos32x4,
    halfDim,
    0,
    1.0,
  );
  slices['rope_sin'] = packRowMajorMatrix(
    writer,
    regions,
    'rope_sin',
    60,
    band3Y,
    DIFFUSION_GQA_MAX_KEY,
    halfDim,
    ropeSin32x4,
    halfDim,
    0,
    1.0,
  );
  slices['llama4_scale'] = packRowMajorMatrix(
    writer,
    regions,
    'llama4_scale',
    64,
    band3Y,
    DIFFUSION_GQA_MAX_KEY,
    1,
    llama4Scale32x1,
    1,
    1,
    1.0 / computeMaxAbs(llama4Scale32x1, 1.0),
  );
  slices['yarn_inv_freq'] = packRowMajorMatrix(
    writer,
    regions,
    'yarn_inv_freq',
    65,
    band3Y,
    halfDim,
    1,
    yarn.invFreq,
    1,
    1,
    1.0,
  );

  // ==========================================================================
  // BAND 4: 8 KV-group Score & Softmax slices (`g = 0..7`, D24 live RG32F mask)
  // ==========================================================================
  const scoresBandY = band3Y + DIFFUSION_GQA_MAX_KEY; // max 116 + 32 = 148
  const softmaxBandY = scoresBandY + S; // max 148 + 16 = 164 (ends <= 180 < 256)

  const scoreSlices: AtlasSlice[] = [];
  const softmaxSlices: AtlasSlice[] = [];
  let maskedScoreCellCount = 0;
  let maskedSoftmaxCellCount = 0;

  for (let g = 0; g < nKv; g++) {
    const qHead = selectedQHeadsPerGroup[g]!;
    const headBase = qHead * fullS * fullK;
    const x0 = g * DIFFUSION_GQA_MAX_KEY; // 8 * 32 = 256 exact fit

    let maxAbsScore = 0.25;
    for (let r = 0; r < S; r++) {
      const rowBase = headBase + r * fullK;
      for (let c = 0; c < K; c++) {
        const sVal = attn.scores[rowBase + c]!;
        const isMasked = !Number.isFinite(sVal);
        if (isMasked) {
          writer.setTexel(x0 + c, scoresBandY + r, 0.0, true);
          maskedScoreCellCount++;
        } else {
          const a = Math.abs(sVal);
          if (a > maxAbsScore) maxAbsScore = a;
          writer.setTexel(x0 + c, scoresBandY + r, sVal, false);
        }

        const pVal = attn.probs[rowBase + c]!;
        if (isMasked) {
          writer.setTexel(x0 + c, softmaxBandY + r, 0.0, true);
          maskedSoftmaxCellCount++;
        } else {
          writer.setTexel(x0 + c, softmaxBandY + r, pVal, false);
        }
      }
    }

    const scoreId = `scores_g${g}`;
    const softmaxId = `softmax_g${g}`;

    regions.push({
      id: scoreId,
      x0,
      y0: scoresBandY,
      cols: K,
      rows: S,
      mode: 0,
      isMaskOrScore: true,
    });
    regions.push({
      id: softmaxId,
      x0,
      y0: softmaxBandY,
      cols: K,
      rows: S,
      mode: 1,
      isMaskOrScore: true,
    });

    const scoreSlice: AtlasSlice = {
      rect: writer.makeRect(x0, scoresBandY, K, S),
      scale: 1.0 / maxAbsScore,
      mode: 0,
    };
    const softmaxSlice: AtlasSlice = {
      rect: writer.makeRect(x0, softmaxBandY, K, S),
      scale: 1.0,
      mode: 1,
    };

    scoreSlices.push(scoreSlice);
    softmaxSlices.push(softmaxSlice);
    slices[scoreId] = scoreSlice;
    slices[softmaxId] = softmaxSlice;
  }

  // Mandatory disjointness & bounds audit
  const audit = auditAtlasRegions(
    DIFFUSION_GQA_ATLAS_WIDTH,
    DIFFUSION_GQA_ATLAS_HEIGHT,
    regions,
  );
  if (audit.hasOverlap || audit.outOfBoundsIds.length > 0) {
    throw new Error(
      `GQA RoPE atlas packing failure: overlap=${String(audit.overlapPair)}, oob=${audit.outOfBoundsIds.join(',')}`,
    );
  }

  const atlasKey = dualHashHex(
    [
      'gqa_rope_atlas_256x256_v1',
      options.snapshotDigest,
      `L${clampedLayer}`,
      `Q${clampedQHead}`,
      `KV${clampedKvHead}`,
      options.maskMode,
      options.loraEnabled ? 'lora1' : 'lora0',
      `S${S}`,
      `K${K}`,
      Array.from(options.positionIds).join(','),
      Array.from(options.cachePositions).join(','),
      Array.from(options.keyPositionIds).join(','),
    ].join('|'),
  );

  return {
    atlas: {
      width: writer.width,
      height: writer.height,
      data: writer.data,
    },
    audit,
    slices: Object.freeze(slices),
    scoreSlices: Object.freeze(scoreSlices),
    softmaxSlices: Object.freeze(softmaxSlices),
    selectedQHeadsPerGroup: Object.freeze(selectedQHeadsPerGroup),
    queryTokens: S,
    keyTokens: K,
    maskedScoreCellCount,
    maskedSoftmaxCellCount,
    atlasKey,
  };
}
