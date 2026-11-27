/**
 * 256×256 RG32F Value & Mask Atlas for View 3 (`joint_attn`)
 * (`src/uni/scene/jointAttnAtlas.ts`)
 *
 * Packs a fixed `256 × 256` (`524,288 B`) `RG32F` WebGL2 atlas with `35` disjoint
 * scalar/mask rectangles (`0` overlaps, `0` out-of-bounds) strictly from live `0`-ULP
 * `UniLayerTrace` and `bundle.core.layers[layerIndex]` weights:
 *   - Routed `U`/`G` weights: `W_Q^U/G`, `W_K^U/G`, `W_V^U/G`, `W_O^U/G`,
 *     `W_gate^U/G`, `W_up^U/G`, `W_down^U/G` (`14` rects)
 *   - Input & QKV activations: `hidden_in`, `normed_attn_in`, `q_pre`, `q_rot`,
 *     `q_scaled`, `k_rot`, `v_new` (`7` rects)
 *   - Live attention scores & softmax (`[S, P+S]`, capped `<= 32×32`) for `selectedQHead`
 *     plus compact 8-head softmax overview, encoding `R = score/prob` and `G = 1.0`
 *     where masked (`effectiveMask === 0` / `-Infinity`) else `0.0` (`3` rects)
 *   - Attention output & routed `3.5×` SwiGLU: `attn_out` (`attnConcat`), `o_proj_out`,
 *     `post_attn_resid`, `mlp_norm_in`, `gate_pre`, `gate_silu`, `up_val`, `gated`,
 *     `mlp_out`, `layer_out` (`10` rects)
 *   - Shared 1D dyadic YaRN RoPE + Llama-4 query temperature table (`1` rect)
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`,
 * or `src/twotower/` / `src/diffusion/` / `src/ui/controls.ts` imports.
 */

import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fAtlasWriter,
  type Rg32fValueAtlas,
} from '../../scene/atlasPacker';
import { dualHashHex } from '../prng';
import {
  computeLlama4AttnScale,
  computeUniRopeCosSin,
  computeUniYarnInvFreq,
} from '../rope';
import {
  UNI_MINI_CONFIG,
  type UniElasticLayerWeights,
  type UniLayerTrace,
  type UniMiniConfig,
} from '../types';
import { auditAtlasRegions } from './provenance';
import type { UniAtlasLayoutAudit, UniAtlasRegionBox } from './types';

export const UNI_JOINT_ATTN_ATLAS_WIDTH = 256 as const;
export const UNI_JOINT_ATTN_ATLAS_HEIGHT = 256 as const;
export const UNI_JOINT_ATTN_MAX_QUERY = 32 as const;
export const UNI_JOINT_ATTN_MAX_KEY = 32 as const;
export const UNI_JOINT_ATTN_MAX_REGIONS = 35 as const;

export interface PackMatrixToRg32fOptions {
  readonly id: string;
  readonly x0: number;
  readonly y0: number;
  readonly rows: number;
  readonly cols: number;
  readonly data: Float32Array;
  readonly srcCols: number;
  readonly rowOffset?: number;
  readonly colOffset?: number;
  readonly transpose?: boolean;
  readonly mode?: 0 | 1;
  readonly scaleOverride?: number;
  readonly isMaskOrScore?: boolean;
  readonly mask?: Uint8Array;
  readonly maskCols?: number;
}

export interface PackedMatrixRg32fResult {
  readonly slice: AtlasSlice;
  readonly region: UniAtlasRegionBox;
  readonly maskedCount: number;
  readonly visibleCount: number;
}

/**
 * Pack a 2D slice of `data` into a `Rg32fAtlasWriter` (`R = scalar`, `G = 1.0` where masked else `0.0`).
 */
function packMatrixToRg32f(
  writer: Rg32fAtlasWriter,
  options: PackMatrixToRg32fOptions,
): PackedMatrixRg32fResult {
  const {
    id,
    x0,
    y0,
    rows,
    cols,
    data,
    srcCols,
    rowOffset = 0,
    colOffset = 0,
    transpose = false,
    mode = 0,
    scaleOverride,
    isMaskOrScore = false,
    mask,
    maskCols = srcCols,
  } = options;

  let maxAbs = 0.25;
  let maskedCount = 0;
  let visibleCount = 0;

  for (let r = 0; r < rows; r++) {
    const srcR = rowOffset + r;
    for (let c = 0; c < cols; c++) {
      const srcC = colOffset + c;
      const idx = transpose ? srcC * srcCols + srcR : srcR * srcCols + srcC;
      const val = data[idx] ?? 0;
      const maskBlocked = mask !== undefined && mask[srcR * maskCols + srcC] === 0;
      const isMasked = maskBlocked || !Number.isFinite(val);

      if (isMasked) {
        writer.setTexel(x0 + c, y0 + r, 0.0, true);
        maskedCount += 1;
      } else {
        writer.setTexel(x0 + c, y0 + r, val, false);
        visibleCount += 1;
        const absVal = Math.abs(val);
        if (absVal > maxAbs) {
          maxAbs = absVal;
        }
      }
    }
  }

  const region: UniAtlasRegionBox = Object.freeze({
    id,
    x0,
    y0,
    cols,
    rows,
    mode,
    isMaskOrScore,
  });

  const slice: AtlasSlice = Object.freeze({
    rect: writer.makeRect(x0, y0, cols, rows),
    scale: scaleOverride ?? (mode === 1 ? 1.0 : 1.0 / maxAbs),
    mode,
  });

  return { slice, region, maskedCount, visibleCount };
}

/** Read a single `(x, y)` texel from a packed `Rg32fValueAtlas`. */
export function readAtlasTexel(
  atlas: Rg32fValueAtlas,
  x: number,
  y: number,
): { readonly r: number; readonly g: number } {
  if (x < 0 || x >= atlas.width || y < 0 || y >= atlas.height) {
    throw new RangeError(
      `Atlas coordinate (${x}, ${y}) out of bounds for ${atlas.width}x${atlas.height}`,
    );
  }
  const idx = (y * atlas.width + x) * 2;
  return {
    r: atlas.data[idx]!,
    g: atlas.data[idx + 1]!,
  };
}

export interface UniJointAttnAtlasInput {
  readonly trace: UniLayerTrace;
  readonly layerWeights: UniElasticLayerWeights;
  readonly config?: UniMiniConfig;
  readonly selectedQHead?: number;
  readonly hiddenIn?: Float32Array | null;
  readonly cacheTag?: string;
}

export interface UniJointAttnAtlasSlices {
  readonly wQU: AtlasSlice;
  readonly wQG: AtlasSlice;
  readonly wKU: AtlasSlice;
  readonly wKG: AtlasSlice;
  readonly wVU: AtlasSlice;
  readonly wVG: AtlasSlice;
  readonly wOU: AtlasSlice;
  readonly wOG: AtlasSlice;
  readonly wGateU: AtlasSlice;
  readonly wGateG: AtlasSlice;
  readonly wUpU: AtlasSlice;
  readonly wUpG: AtlasSlice;
  readonly wDownU: AtlasSlice;
  readonly wDownG: AtlasSlice;
  readonly hiddenIn: AtlasSlice;
  readonly normedAttnIn: AtlasSlice;
  readonly qPre: AtlasSlice;
  readonly qRot: AtlasSlice;
  readonly qScaled: AtlasSlice;
  readonly kRot: AtlasSlice;
  readonly vNew: AtlasSlice;
  readonly scoresSelected: AtlasSlice;
  readonly softmaxSelected: AtlasSlice;
  readonly softmaxOverview: AtlasSlice;
  readonly attnOut: AtlasSlice;
  readonly oProjOut: AtlasSlice;
  readonly postAttnResid: AtlasSlice;
  readonly mlpNormIn: AtlasSlice;
  readonly gatePre: AtlasSlice;
  readonly gateSilu: AtlasSlice;
  readonly upVal: AtlasSlice;
  readonly gated: AtlasSlice;
  readonly mlpOut: AtlasSlice;
  readonly layerOut: AtlasSlice;
  readonly ropeLlama4Table: AtlasSlice;
}

export interface UniJointAttnAtlasPack {
  readonly atlas: Rg32fValueAtlas;
  readonly audit: UniAtlasLayoutAudit;
  readonly atlasKey: string;
  readonly slices: UniJointAttnAtlasSlices;
  readonly sliceMap: Readonly<Record<string, AtlasSlice>>;
  readonly regionMap: Readonly<Record<string, UniAtlasRegionBox>>;
  readonly queryTokens: number;
  readonly keyTokens: number;
  readonly qOffset: number;
  readonly kOffset: number;
  readonly selectedQHead: number;
  readonly selectedKvHead: number;
  readonly maskedScoreCellCount: number;
  readonly maskedSoftmaxCellCount: number;
  readonly visibleScoreCellCount: number;
}

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(v)));
}

function resolveHiddenIn(
  trace: UniLayerTrace,
  d: number,
  explicitHiddenIn?: Float32Array | null,
): Float32Array {
  const fullS = trace.attention.seqLen;
  if (explicitHiddenIn && explicitHiddenIn.length >= fullS * d) {
    return explicitHiddenIn;
  }
  const reconstructed = new Float32Array(fullS * d);
  for (let i = 0; i < fullS * d; i++) {
    reconstructed[i] = Math.fround(
      (trace.postAttentionResidual[i] ?? 0) - (trace.attention.output[i] ?? 0),
    );
  }
  return reconstructed;
}

/**
 * Pack all 35 live tensors and attention masks for View 3 (`joint_attn`) into a
 * fixed `256 × 256` (`524,288 B`) `RG32F` WebGL2 atlas with zero overlaps.
 */
export function packUniJointAttnAtlas(
  input: UniJointAttnAtlasInput,
): UniJointAttnAtlasPack {
  const cfg = input.config ?? UNI_MINI_CONFIG;
  const { trace, layerWeights } = input;
  const attn = trace.attention;
  const swiglu = trace.swiglu;

  const d = cfg.hiddenSize; // 8
  const qDim = cfg.qProjOutDim; // 64
  const kvDim = cfg.kvProjOutDim; // 16
  const dFf = cfg.intermediateSize; // 28
  const nQ = cfg.numAttentionHeads; // 8
  const halfDim = cfg.headDim / 2; // 4

  const fullS = attn.seqLen;
  const fullK = attn.kvLen;
  const S = clampInt(fullS, 1, UNI_JOINT_ATTN_MAX_QUERY);
  const K = clampInt(fullK, 1, UNI_JOINT_ATTN_MAX_KEY);
  const qOffset = Math.max(0, fullS - S);
  const kOffset = Math.max(0, fullK - K);

  const selectedQHead = clampInt(input.selectedQHead ?? 0, 0, nQ - 1);
  const selectedKvHead = Math.floor(selectedQHead / cfg.kvGroupSize);
  const hiddenInData = resolveHiddenIn(trace, d, input.hiddenIn);

  const writer = createRg32fAtlas(
    UNI_JOINT_ATTN_ATLAS_WIDTH,
    UNI_JOINT_ATTN_ATLAS_HEIGHT,
  );
  const regions: UniAtlasRegionBox[] = [];
  const sliceMap: Record<string, AtlasSlice> = {};
  const regionMap: Record<string, UniAtlasRegionBox> = {};

  const recordPack = (opts: PackMatrixToRg32fOptions): PackedMatrixRg32fResult => {
    const res = packMatrixToRg32f(writer, opts);
    regions.push(res.region);
    sliceMap[opts.id] = res.slice;
    regionMap[opts.id] = res.region;
    return res;
  };

  // ==========================================================================
  // BAND 1 (`y = 0 .. 8 + S - 1 <= 39`): 64-col matrices (`4 × 64 = 256`)
  // ==========================================================================
  const wQU = recordPack({
    id: 'w_q_u',
    x0: 0,
    y0: 0,
    cols: qDim,
    rows: d,
    data: layerWeights.u.selfAttn.qProj,
    srcCols: d,
    transpose: true,
  }).slice;
  const wQG = recordPack({
    id: 'w_q_g',
    x0: 64,
    y0: 0,
    cols: qDim,
    rows: d,
    data: layerWeights.g.selfAttn.qProj,
    srcCols: d,
    transpose: true,
  }).slice;
  const wOU = recordPack({
    id: 'w_o_u',
    x0: 128,
    y0: 0,
    cols: qDim,
    rows: d,
    data: layerWeights.u.selfAttn.oProj,
    srcCols: qDim,
  }).slice;
  const wOG = recordPack({
    id: 'w_o_g',
    x0: 192,
    y0: 0,
    cols: qDim,
    rows: d,
    data: layerWeights.g.selfAttn.oProj,
    srcCols: qDim,
  }).slice;

  const qPre = recordPack({
    id: 'q_pre',
    x0: 0,
    y0: 8,
    cols: qDim,
    rows: S,
    data: attn.qPre,
    srcCols: qDim,
    rowOffset: qOffset,
  }).slice;
  const qRot = recordPack({
    id: 'q_rot',
    x0: 64,
    y0: 8,
    cols: qDim,
    rows: S,
    data: attn.qRot,
    srcCols: qDim,
    rowOffset: qOffset,
  }).slice;
  const qScaled = recordPack({
    id: 'q_scaled',
    x0: 128,
    y0: 8,
    cols: qDim,
    rows: S,
    data: attn.qScaled,
    srcCols: qDim,
    rowOffset: qOffset,
  }).slice;
  const attnOut = recordPack({
    id: 'attn_out',
    x0: 192,
    y0: 8,
    cols: qDim,
    rows: S,
    data: attn.attnConcat,
    srcCols: qDim,
    rowOffset: qOffset,
  }).slice;

  // ==========================================================================
  // BAND 2 (`y = 40 .. 79`): SwiGLU & KV weights + Sequence Activations + RoPE
  // ==========================================================================
  const wGateU = recordPack({
    id: 'w_gate_u',
    x0: 0,
    y0: 40,
    cols: dFf,
    rows: d,
    data: layerWeights.u.mlp.gateProj,
    srcCols: d,
    transpose: true,
  }).slice;
  const wGateG = recordPack({
    id: 'w_gate_g',
    x0: 28,
    y0: 40,
    cols: dFf,
    rows: d,
    data: layerWeights.g.mlp.gateProj,
    srcCols: d,
    transpose: true,
  }).slice;
  const wUpU = recordPack({
    id: 'w_up_u',
    x0: 56,
    y0: 40,
    cols: dFf,
    rows: d,
    data: layerWeights.u.mlp.upProj,
    srcCols: d,
    transpose: true,
  }).slice;
  const wUpG = recordPack({
    id: 'w_up_g',
    x0: 84,
    y0: 40,
    cols: dFf,
    rows: d,
    data: layerWeights.g.mlp.upProj,
    srcCols: d,
    transpose: true,
  }).slice;

  const wKU = recordPack({
    id: 'w_k_u',
    x0: 112,
    y0: 40,
    cols: kvDim,
    rows: d,
    data: layerWeights.u.selfAttn.kProj,
    srcCols: d,
    transpose: true,
  }).slice;
  const wKG = recordPack({
    id: 'w_k_g',
    x0: 128,
    y0: 40,
    cols: kvDim,
    rows: d,
    data: layerWeights.g.selfAttn.kProj,
    srcCols: d,
    transpose: true,
  }).slice;
  const wVU = recordPack({
    id: 'w_v_u',
    x0: 144,
    y0: 40,
    cols: kvDim,
    rows: d,
    data: layerWeights.u.selfAttn.vProj,
    srcCols: d,
    transpose: true,
  }).slice;
  const wVG = recordPack({
    id: 'w_v_g',
    x0: 160,
    y0: 40,
    cols: kvDim,
    rows: d,
    data: layerWeights.g.selfAttn.vProj,
    srcCols: d,
    transpose: true,
  }).slice;

  const wDownU = recordPack({
    id: 'w_down_u',
    x0: 176,
    y0: 40,
    cols: d,
    rows: dFf,
    data: layerWeights.u.mlp.downProj,
    srcCols: dFf,
    transpose: true,
  }).slice;
  const wDownG = recordPack({
    id: 'w_down_g',
    x0: 184,
    y0: 40,
    cols: d,
    rows: dFf,
    data: layerWeights.g.mlp.downProj,
    srcCols: dFf,
    transpose: true,
  }).slice;

  const gatePre = recordPack({
    id: 'gate_pre',
    x0: 0,
    y0: 48,
    cols: dFf,
    rows: S,
    data: swiglu.gatePre,
    srcCols: dFf,
    rowOffset: qOffset,
  }).slice;
  const gateSilu = recordPack({
    id: 'gate_silu',
    x0: 28,
    y0: 48,
    cols: dFf,
    rows: S,
    data: swiglu.gateSilu,
    srcCols: dFf,
    rowOffset: qOffset,
  }).slice;
  const upVal = recordPack({
    id: 'up_val',
    x0: 56,
    y0: 48,
    cols: dFf,
    rows: S,
    data: swiglu.upVal,
    srcCols: dFf,
    rowOffset: qOffset,
  }).slice;
  const gated = recordPack({
    id: 'gated',
    x0: 84,
    y0: 48,
    cols: dFf,
    rows: S,
    data: swiglu.gated,
    srcCols: dFf,
    rowOffset: qOffset,
  }).slice;

  const kRot = recordPack({
    id: 'k_rot',
    x0: 112,
    y0: 48,
    cols: kvDim,
    rows: S,
    data: attn.kRot,
    srcCols: kvDim,
    rowOffset: qOffset,
  }).slice;
  const vNew = recordPack({
    id: 'v_new',
    x0: 128,
    y0: 48,
    cols: kvDim,
    rows: S,
    data: attn.vNew,
    srcCols: kvDim,
    rowOffset: qOffset,
  }).slice;

  const hiddenIn = recordPack({
    id: 'hidden_in',
    x0: 144,
    y0: 48,
    cols: d,
    rows: S,
    data: hiddenInData,
    srcCols: d,
    rowOffset: qOffset,
  }).slice;
  const normedAttnIn = recordPack({
    id: 'normed_attn_in',
    x0: 152,
    y0: 48,
    cols: d,
    rows: S,
    data: trace.inputNorm,
    srcCols: d,
    rowOffset: qOffset,
  }).slice;
  const oProjOut = recordPack({
    id: 'o_proj_out',
    x0: 160,
    y0: 48,
    cols: d,
    rows: S,
    data: attn.output,
    srcCols: d,
    rowOffset: qOffset,
  }).slice;
  const postAttnResid = recordPack({
    id: 'post_attn_resid',
    x0: 168,
    y0: 48,
    cols: d,
    rows: S,
    data: trace.postAttentionResidual,
    srcCols: d,
    rowOffset: qOffset,
  }).slice;

  const mlpNormIn = recordPack({
    id: 'mlp_norm_in',
    x0: 192,
    y0: 40,
    cols: d,
    rows: S,
    data: trace.postAttentionNorm,
    srcCols: d,
    rowOffset: qOffset,
  }).slice;
  const mlpOut = recordPack({
    id: 'mlp_out',
    x0: 200,
    y0: 40,
    cols: d,
    rows: S,
    data: swiglu.output,
    srcCols: d,
    rowOffset: qOffset,
  }).slice;
  const layerOut = recordPack({
    id: 'layer_out',
    x0: 208,
    y0: 40,
    cols: d,
    rows: S,
    data: trace.output,
    srcCols: d,
    rowOffset: qOffset,
  }).slice;

  // Compact [32, 10] YaRN cos(4) | sin(4) | llama4Scale(1) | invFreq(1 on rows 0..3) table
  const yarn = computeUniYarnInvFreq(cfg.rope);
  const pos32 = new Int32Array(UNI_JOINT_ATTN_MAX_KEY);
  for (let p = 0; p < UNI_JOINT_ATTN_MAX_KEY; p++) {
    pos32[p] = p;
  }
  const cosSin32 = computeUniRopeCosSin(
    yarn.invFreq,
    pos32,
    yarn.attentionScaling,
    cfg.headDim,
  );
  const ropeTableCols = halfDim * 2 + 2; // 10
  const ropeTable = new Float32Array(UNI_JOINT_ATTN_MAX_KEY * ropeTableCols);
  for (let p = 0; p < UNI_JOINT_ATTN_MAX_KEY; p++) {
    const dstBase = p * ropeTableCols;
    const csBase = p * cfg.headDim;
    for (let plane = 0; plane < halfDim; plane++) {
      ropeTable[dstBase + plane] = cosSin32.cos[csBase + plane]!;
      ropeTable[dstBase + halfDim + plane] = cosSin32.sin[csBase + plane]!;
    }
    ropeTable[dstBase + halfDim * 2] = Math.fround(
      computeLlama4AttnScale(
        p,
        cfg.rope.llama4Beta,
        cfg.rope.originalMaxPositionEmbeddings,
      ),
    );
    ropeTable[dstBase + halfDim * 2 + 1] =
      p < halfDim ? yarn.invFreq[p]! : 0.0;
  }
  const ropeLlama4Table = recordPack({
    id: 'rope_llama4_table',
    x0: 216,
    y0: 40,
    cols: ropeTableCols,
    rows: UNI_JOINT_ATTN_MAX_KEY,
    data: ropeTable,
    srcCols: ropeTableCols,
    scaleOverride: 1.0 / computeMaxAbs(ropeTable, 1.0),
  }).slice;

  // ==========================================================================
  // BAND 3 (`y = 80 .. 111`): Multi-Head Softmax Overview (`8 * K` × `S`)
  // ==========================================================================
  const overviewX0 = 0;
  const overviewY0 = 80;
  for (let h = 0; h < nQ; h++) {
    const headBase = h * fullS * fullK;
    for (let r = 0; r < S; r++) {
      const qRow = qOffset + r;
      for (let c = 0; c < K; c++) {
        const kCol = kOffset + c;
        const maskBit = attn.effectiveMask[qRow * fullK + kCol]!;
        const sVal = attn.scores[headBase + qRow * fullK + kCol]!;
        const isMasked = maskBit === 0 || !Number.isFinite(sVal);
        const pVal = attn.probs[headBase + qRow * fullK + kCol]!;
        writer.setTexel(
          overviewX0 + h * K + c,
          overviewY0 + r,
          isMasked ? 0.0 : pVal,
          isMasked,
        );
      }
    }
  }
  const overviewRegion: UniAtlasRegionBox = Object.freeze({
    id: 'softmax_overview',
    x0: overviewX0,
    y0: overviewY0,
    cols: nQ * K,
    rows: S,
    mode: 1,
    isMaskOrScore: true,
  });
  const softmaxOverview: AtlasSlice = Object.freeze({
    rect: writer.makeRect(overviewX0, overviewY0, nQ * K, S),
    scale: 1.0,
    mode: 1,
  });
  regions.push(overviewRegion);
  sliceMap['softmax_overview'] = softmaxOverview;
  regionMap['softmax_overview'] = overviewRegion;

  // ==========================================================================
  // BAND 4 (`y = 112 .. 143`): Selected Q-Head Scores & Softmax (`G = 1` masked)
  // ==========================================================================
  const headOffset = selectedQHead * fullS * fullK;
  const scoresHeadSlice = attn.scores.subarray(headOffset, headOffset + fullS * fullK);
  const probsHeadSlice = attn.probs.subarray(headOffset, headOffset + fullS * fullK);

  const scoreRes = recordPack({
    id: 'scores_selected',
    x0: 0,
    y0: 112,
    cols: K,
    rows: S,
    data: scoresHeadSlice,
    srcCols: fullK,
    rowOffset: qOffset,
    colOffset: kOffset,
    mode: 0,
    isMaskOrScore: true,
    mask: attn.effectiveMask,
    maskCols: fullK,
  });

  const softmaxRes = recordPack({
    id: 'softmax_selected',
    x0: 32,
    y0: 112,
    cols: K,
    rows: S,
    data: probsHeadSlice,
    srcCols: fullK,
    rowOffset: qOffset,
    colOffset: kOffset,
    mode: 1,
    scaleOverride: 1.0,
    isMaskOrScore: true,
    mask: attn.effectiveMask,
    maskCols: fullK,
  });

  const audit = auditAtlasRegions(
    UNI_JOINT_ATTN_ATLAS_WIDTH,
    UNI_JOINT_ATTN_ATLAS_HEIGHT,
    regions,
  );
  if (
    audit.hasOverlap ||
    audit.outOfBoundsIds.length > 0 ||
    !audit.withinBudget ||
    regions.length > UNI_JOINT_ATTN_MAX_REGIONS
  ) {
    throw new Error(
      `View 3 joint_attn atlas audit failed: regions=${regions.length}, overlap=${String(audit.overlapPair)}, oob=${audit.outOfBoundsIds.join(',')}`,
    );
  }

  const atlas: Rg32fValueAtlas = {
    width: UNI_JOINT_ATTN_ATLAS_WIDTH,
    height: UNI_JOINT_ATTN_ATLAS_HEIGHT,
    data: writer.data,
  };

  const atlasKey = dualHashHex(
    [
      'uni_joint_attn_256x256_v1',
      `L${trace.layerIndex}`,
      `Q${selectedQHead}`,
      `S${S}`,
      `K${K}`,
      input.cacheTag ?? '',
    ].join('|'),
  );

  const slices: UniJointAttnAtlasSlices = Object.freeze({
    wQU,
    wQG,
    wKU,
    wKG,
    wVU,
    wVG,
    wOU,
    wOG,
    wGateU,
    wGateG,
    wUpU,
    wUpG,
    wDownU,
    wDownG,
    hiddenIn,
    normedAttnIn,
    qPre,
    qRot,
    qScaled,
    kRot,
    vNew,
    scoresSelected: scoreRes.slice,
    softmaxSelected: softmaxRes.slice,
    softmaxOverview,
    attnOut,
    oProjOut,
    postAttnResid,
    mlpNormIn,
    gatePre,
    gateSilu,
    upVal,
    gated,
    mlpOut,
    layerOut,
    ropeLlama4Table,
  });

  return {
    atlas,
    audit,
    atlasKey,
    slices,
    sliceMap: Object.freeze(sliceMap),
    regionMap: Object.freeze(regionMap),
    queryTokens: S,
    keyTokens: K,
    qOffset,
    kOffset,
    selectedQHead,
    selectedKvHead,
    maskedScoreCellCount: scoreRes.maskedCount,
    maskedSoftmaxCellCount: softmaxRes.maskedCount,
    visibleScoreCellCount: scoreRes.visibleCount,
  };
}
