/**
 * Dedicated 128×128 TwoTower GQA RG32F Value Atlas (`src/twotower/scene/twoTowerGqaAtlas.ts`)
 *
 * Packs live miniature Float32 tensors for the primary tower (Denoiser or solo Context) AND the
 * secondary Context provenance tower (or persistent Context KV cache fallback when historical
 * Context traces were pruned) into ONE non-overlapping 128×128 RG32F texture (16,384 texels
 * = 128 KiB), supporting query lengths T <= 16 and attended KV lengths K <= 24.
 *
 * DIMENSIONAL INVENTORY VS. SELECTED ATLAS SLICES:
 *   - Complete miniature GQA model inventory: 32 query heads (Q00..Q31, 16 per KV group),
 *     2 KV heads (KV0, KV1), head width dHead = 4 (qConcatDim = 128, kvConcatDim = 8).
 *   - All 32 query heads (Q[0..31]) and both KV heads (KV0/KV1) are packed in full;
 *     attention Scores & Softmax regions pack 2 SELECTED head slices (1 per KV group:
 *     `group0Head ∈ [0..15]` and `group1Head ∈ [16..31]`) out of the model's 32 logical Q heads.
 *
 * CRITICAL VISIBILITY CONSISTENCY:
 *   Uses `isAttentionKeyVisible(sim.attentionMaskMode, sim.prefixTokens, q, k, sim.keyTokens)`
 *   to encode channel G (`0.0` visible, `1.0` masked) on BOTH `Scores` AND `Softmax`:
 *     - `'bidirectional_block'` (Denoiser diffusion): 0 masked cells across 0..P+B-1.
 *     - `'causal'` (Context / Mock-AR): k > P + q strictly masked (`G = 1.0`).
 *
 * Texel map (width = 128, height = 128; strictly disjoint non-aliasing regions):
 *   y =   0..  7 : Primary W_Q             (128 cols ×  8 rows) at x = 0..127
 *   y =   8.. 15 : Primary W_O             (128 cols ×  8 rows) at x = 0..127
 *   y =  16.. 31 : Primary Q[0..31]        (128 cols × T1 rows) at x = 0..127 (head h at x = h*4)
 *   y =  32.. 47 : Primary V_out / Concat  (128 cols × T1 rows) at x = 0..127 (Grp0=0..63, Grp1=64..127)
 *   y =  48.. 71 : Primary Attention & KV Bank + Weights + Input:
 *     x =   0.. 23, y = 48..48+T1-1 : Primary Scores[g0]   (K1<=24 cols × T1<=16 rows)
 *     x =  24.. 47, y = 48..48+T1-1 : Primary Softmax[g0]  (K1<=24 cols × T1<=16 rows)
 *     x =  48.. 71, y = 48..48+T1-1 : Primary Scores[g1]   (K1<=24 cols × T1<=16 rows)
 *     x =  72.. 95, y = 48..48+T1-1 : Primary Softmax[g1]  (K1<=24 cols × T1<=16 rows)
 *     x =  96.. 99, y = 48..48+K1-1 : Primary KV0_K        (4 cols × K1<=24 rows; slice k at y=48+k)
 *     x = 100..103, y = 48..48+K1-1 : Primary KV0_V        (4 cols × K1<=24 rows)
 *     x = 104..107, y = 48..48+K1-1 : Primary KV1_K        (4 cols × K1<=24 rows)
 *     x = 108..111, y = 48..48+K1-1 : Primary KV1_V        (4 cols × K1<=24 rows)
 *     x = 112..119, y = 48..55      : Primary W_K          (8 cols × 8 rows)
 *     x = 120..127, y = 48..55      : Primary W_V          (8 cols × 8 rows)
 *     x = 112..119, y = 56..56+T1-1 : Primary Input X      (8 cols × T1<=16 rows)
 *     x = 120..127, y = 56..56+T1-1 : Primary X_mod (opt.) (8 cols × T1<=16 rows)
 *   y =  72.. 87 : Primary U/Y + Secondary Context Scalars & Active Heads:
 *     x =   0..  7, y = 72..72+T1-1 : Primary RMSNorm U    (8 cols × T1<=16 rows)
 *     x =   8.. 15, y = 72..72+T1-1 : Primary Residual Y   (8 cols × T1<=16 rows)
 *     x =  16.. 23, y = 72..72+T2-1 : Secondary Input X    (8 cols × T2<=16 rows)
 *     x =  24.. 31, y = 72..72+T2-1 : Secondary RMSNorm U  (8 cols × T2<=16 rows)
 *     x =  32.. 39, y = 72..72+T2-1 : Secondary Residual Y (8 cols × T2<=16 rows)
 *     x =  40.. 43, y = 72..72+T2-1 : Secondary Q[g0]      (4 cols × T2<=16 rows)
 *     x =  44.. 47, y = 72..72+T2-1 : Secondary Q[g1]      (4 cols × T2<=16 rows)
 *     x =  48.. 51, y = 72..72+T2-1 : Secondary V_out[g0]  (4 cols × T2<=16 rows)
 *     x =  52.. 55, y = 72..72+T2-1 : Secondary V_out[g1]  (4 cols × T2<=16 rows)
 *     x =  56.. 63, y = 72..79      : Secondary W_K        (8 cols × 8 rows)
 *     x =  56.. 63, y = 80..87      : Secondary W_V        (8 cols × 8 rows)
 *   y =  88..111 : Secondary Context Attention & KV Bank (up to K2 <= 24):
 *     x =   0.. 23, y = 88..88+T2-1 : Secondary Scores[g0]  (K2<=24 cols × T2<=16 rows)
 *     x =  24.. 47, y = 88..88+T2-1 : Secondary Softmax[g0] (K2<=24 cols × T2<=16 rows)
 *     x =  48.. 71, y = 88..88+T2-1 : Secondary Scores[g1]  (K2<=24 cols × T2<=16 rows)
 *     x =  72.. 95, y = 88..88+T2-1 : Secondary Softmax[g1] (K2<=24 cols × T2<=16 rows)
 *     x =  96.. 99, y = 88..88+K2-1 : Secondary KV0_K       (4 cols × K2<=24 rows)
 *     x = 100..103, y = 88..88+K2-1 : Secondary KV0_V       (4 cols × K2<=24 rows)
 *     x = 104..107, y = 88..88+K2-1 : Secondary KV1_K       (4 cols × K2<=24 rows)
 *     x = 108..111, y = 88..88+K2-1 : Secondary KV1_V       (4 cols × K2<=24 rows)
 *   y = 112..127 : Secondary Q[0..31]                       (128 cols × T2<=16 rows)
 *
 * Pure headless TypeScript — zero DOM, zero WebGL2, fully deterministic.
 */

import type { GqaLayerKvCache } from '../../math/cacheContracts';
import { isAttentionKeyVisible, type AttentionMaskMode, type GqaSimResult } from '../../math/gqaSim';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
  type TexRect,
} from '../../scene/atlasPacker';

export const TWOTOWER_GQA_ATLAS_WIDTH = 128 as const;
export const TWOTOWER_GQA_ATLAS_HEIGHT = 128 as const;
export const TWOTOWER_GQA_MAX_QUERY = 16 as const;
export const TWOTOWER_GQA_MAX_KEY = 24 as const;

export interface TwoTowerGqaPrimaryAtlasSlices {
  readonly tokens: number;
  readonly prefixTokens: number;
  readonly keyTokens: number;
  readonly attentionMaskMode: AttentionMaskMode;
  readonly maskedScoreCellCount: number;
  readonly maskedSoftmaxCellCount: number;
  readonly input: AtlasSlice;
  readonly modulatedInput: AtlasSlice | null;
  readonly rmsnorm: AtlasSlice;
  readonly residual: AtlasSlice;
  readonly wQ: AtlasSlice;
  readonly wK: AtlasSlice;
  readonly wV: AtlasSlice;
  readonly wO: AtlasSlice;
  readonly qHeads: readonly AtlasSlice[]; // 32 heads
  readonly kvK: readonly [readonly AtlasSlice[], readonly AtlasSlice[]]; // [group][k = 0..K-1]
  readonly kvV: readonly [readonly AtlasSlice[], readonly AtlasSlice[]]; // [group][k = 0..K-1]
  readonly scores: readonly [AtlasSlice, AtlasSlice]; // [group0Head, group1Head]
  readonly softmax: readonly [AtlasSlice, AtlasSlice]; // [group0Head, group1Head]
  readonly vOut: readonly [AtlasSlice, AtlasSlice]; // [group0 (64×T), group1 (64×T)]
  readonly concat: AtlasSlice; // [128×T] (intentional alias over vOut[0] + vOut[1])
}

export interface TwoTowerGqaSecondaryAtlasSlices {
  readonly kind: 'full_sim' | 'persistent_cache_only';
  readonly tokens: number;
  readonly prefixTokens: number;
  readonly keyTokens: number;
  readonly attentionMaskMode: AttentionMaskMode | null;
  readonly maskedScoreCellCount: number;
  readonly maskedSoftmaxCellCount: number;
  readonly input: AtlasSlice | null;
  readonly rmsnorm: AtlasSlice | null;
  readonly residual: AtlasSlice | null;
  readonly wK: AtlasSlice | null;
  readonly wV: AtlasSlice | null;
  readonly activeQ: readonly [AtlasSlice, AtlasSlice] | null;
  readonly activeVOut: readonly [AtlasSlice, AtlasSlice] | null;
  readonly qHeads: readonly AtlasSlice[] | null; // 32 heads when full_sim
  readonly kvK: readonly [readonly AtlasSlice[], readonly AtlasSlice[]]; // [group][k = 0..K-1]
  readonly kvV: readonly [readonly AtlasSlice[], readonly AtlasSlice[]]; // [group][k = 0..K-1]
  readonly scores: readonly [AtlasSlice, AtlasSlice] | null;
  readonly softmax: readonly [AtlasSlice, AtlasSlice] | null;
}

export interface TwoTowerGqaValueAtlas extends Rg32fValueAtlas {
  readonly width: typeof TWOTOWER_GQA_ATLAS_WIDTH;
  readonly height: typeof TWOTOWER_GQA_ATLAS_HEIGHT;
  /** Complete miniature GQA model head inventory (32 Q heads, 2 KV heads, dHead = 4). */
  readonly logicalQueryHeadCount: 32;
  readonly logicalKvHeadCount: 2;
  readonly headDim: 4;
  /** Selected 2-head atlas slice indices (1 per KV group out of 32 logical Q heads). */
  readonly group0Head: number;
  readonly group1Head: number;
  readonly primary: TwoTowerGqaPrimaryAtlasSlices;
  readonly secondary: TwoTowerGqaSecondaryAtlasSlices | null;
}

export interface BuildTwoTowerGqaValueAtlasParams {
  readonly primarySim: GqaSimResult;
  readonly secondarySim?: GqaSimResult | null;
  /** Optional persistent Context KV cache used ONLY when `secondarySim` is null/pruned. */
  readonly secondaryKvCacheFallback?: GqaLayerKvCache | null;
  readonly group0Head?: number;
  readonly group1Head?: number;
}

export interface TwoTowerGqaAtlasLayoutVerification {
  readonly valid: boolean;
  readonly width: number;
  readonly height: number;
  readonly byteLength: number;
  readonly checkedNonAliasingSliceCount: number;
  readonly checkedAliasSliceCount: number;
  readonly errors: readonly string[];
}

function validateGqaSimBounds(sim: GqaSimResult, roleLabel: string): void {
  if (!Number.isInteger(sim.tokens) || sim.tokens < 1 || sim.tokens > TWOTOWER_GQA_MAX_QUERY) {
    throw new RangeError(
      `TwoTower GQA atlas ${roleLabel} query tokens (${sim.tokens}) must be an integer in [1..${TWOTOWER_GQA_MAX_QUERY}]`,
    );
  }
  if (!Number.isInteger(sim.keyTokens) || sim.keyTokens < 1 || sim.keyTokens > TWOTOWER_GQA_MAX_KEY) {
    throw new RangeError(
      `TwoTower GQA atlas ${roleLabel} keyTokens (${sim.keyTokens}) must be an integer in [1..${TWOTOWER_GQA_MAX_KEY}]`,
    );
  }
}

/**
 * Pack every live Float32 tensor for `primarySim` and optional `secondarySim` (or pruned-history
 * `secondaryKvCacheFallback`) into ONE non-overlapping 128×128 RG32F value atlas.
 */
export function buildTwoTowerGqaValueAtlas(
  params: BuildTwoTowerGqaValueAtlasParams,
): TwoTowerGqaValueAtlas {
  const { primarySim, secondarySim = null, secondaryKvCacheFallback = null } = params;
  validateGqaSimBounds(primarySim, 'primarySim');
  if (secondarySim !== null) {
    validateGqaSimBounds(secondarySim, 'secondarySim');
  } else if (secondaryKvCacheFallback !== null) {
    const seqLen = secondaryKvCacheFallback.seqLen;
    if (!Number.isInteger(seqLen) || seqLen < 0 || seqLen > TWOTOWER_GQA_MAX_KEY) {
      throw new RangeError(
        `TwoTower GQA atlas secondaryKvCacheFallback.seqLen (${seqLen}) must be in [0..${TWOTOWER_GQA_MAX_KEY}]`,
      );
    }
  }

  const g0 = Math.max(0, Math.min(15, Math.round(params.group0Head ?? 0)));
  const g1 = Math.max(16, Math.min(31, Math.round(params.group1Head ?? 16)));
  const scoreHeads: readonly [number, number] = [g0, g1];

  const writer = createRg32fAtlas(TWOTOWER_GQA_ATLAS_WIDTH, TWOTOWER_GQA_ATLAS_HEIGHT);
  const { data, setTexel, makeRect } = writer;

  // ==========================================================================
  // 1. PRIMARY TOWER PACKING (y = 0..71 + primary U/Y at x = 0..15, y = 72..87)
  // ==========================================================================
  const T1 = primarySim.tokens;
  const K1 = primarySim.keyTokens;
  const P1 = primarySim.prefixTokens;
  const { d, dHead, qConcatDim } = primarySim;

  // 1a. Primary W_Q [128, 8] -> x=0..127, y=0..7
  for (let j = 0; j < d; j++) {
    for (let outRow = 0; outRow < qConcatDim; outRow++) {
      setTexel(outRow, j, primarySim.W_Q[outRow * d + j]);
    }
  }

  // 1b. Primary W_O [8, 128] -> x=0..127, y=8..15
  for (let i = 0; i < d; i++) {
    for (let c = 0; c < qConcatDim; c++) {
      setTexel(c, 8 + i, primarySim.W_O[i * qConcatDim + c]);
    }
  }

  // 1c. Primary Q[0..31] [128, T1] -> x=0..127, y=16..16+T1-1
  for (let t = 0; t < T1; t++) {
    for (let c = 0; c < qConcatDim; c++) {
      setTexel(c, 16 + t, primarySim.Q[c * T1 + t]);
    }
  }

  // 1d. Primary V_out / Concat [128, T1] -> x=0..127, y=32..32+T1-1
  for (let q = 0; q < T1; q++) {
    for (let c = 0; c < qConcatDim; c++) {
      setTexel(c, 32 + q, primarySim.Concat[c * T1 + q]);
    }
  }

  // 1e. Primary Scores & Softmax [K1<=24 cols × T1<=16 rows] at y=48..48+T1-1
  const scoreOffsets: readonly [number, number] = [0, 48];
  const softOffsets: readonly [number, number] = [24, 72];
  let primaryMaxAbsScore = 1.2;
  let primaryMaskedScores = 0;
  let primaryMaskedSoftmax = 0;

  for (let grp = 0; grp < 2; grp++) {
    const h = scoreHeads[grp];
    const headOff = h * T1 * K1;
    for (let q = 0; q < T1; q++) {
      for (let k = 0; k < K1; k++) {
        const visible = isAttentionKeyVisible(primarySim.attentionMaskMode, P1, q, k, K1);
        const s = primarySim.Scores[headOff + q * K1 + k];
        if (visible && Number.isFinite(s) && Math.abs(s) > primaryMaxAbsScore) {
          primaryMaxAbsScore = Math.abs(s);
        }
        if (!visible) {
          primaryMaskedScores++;
          primaryMaskedSoftmax++;
        }
        setTexel(scoreOffsets[grp] + k, 48 + q, s, !visible);
        const sm = primarySim.Softmax[headOff + q * K1 + k];
        setTexel(softOffsets[grp] + k, 48 + q, sm, !visible);
      }
    }
  }

  // 1f. Primary KV0 & KV1 (4 cols × K1<=24 rows) at x=96..111, y=48..48+K1-1
  for (let g = 0; g < 2; g++) {
    const kBaseX = g === 0 ? 96 : 104;
    const vBaseX = g === 0 ? 100 : 108;
    for (let k = 0; k < K1; k++) {
      for (let r = 0; r < dHead; r++) {
        const row = g * dHead + r;
        setTexel(kBaseX + r, 48 + k, primarySim.K[row * K1 + k]);
        setTexel(vBaseX + r, 48 + k, primarySim.V[row * K1 + k]);
      }
    }
  }

  // 1g. Primary W_K (x=112..119, y=48..55) & W_V (x=120..127, y=48..55)
  for (let j = 0; j < d; j++) {
    for (let outRow = 0; outRow < 8; outRow++) {
      setTexel(112 + outRow, 48 + j, primarySim.W_K[outRow * d + j]);
      setTexel(120 + outRow, 48 + j, primarySim.W_V[outRow * d + j]);
    }
  }

  // 1h. Primary Input X (x=112..119, y=56..56+T1-1), optional X_mod (x=120..127, y=56..56+T1-1),
  //     RMSNorm U (x=0..7, y=72..72+T1-1), Residual Y (x=8..15, y=72..72+T1-1)
  for (let t = 0; t < T1; t++) {
    for (let i = 0; i < d; i++) {
      setTexel(112 + i, 56 + t, primarySim.X[i * T1 + t]);
      if (primarySim.X_mod) {
        setTexel(120 + i, 56 + t, primarySim.X_mod[i * T1 + t]);
      }
      setTexel(0 + i, 72 + t, primarySim.U[i * T1 + t]);
      setTexel(8 + i, 72 + t, primarySim.Y[i * T1 + t]);
    }
  }

  const pScaleX = 1 / computeMaxAbs(primarySim.X, 1.0);
  const pScaleXMod = primarySim.X_mod ? 1 / computeMaxAbs(primarySim.X_mod, 1.0) : pScaleX;
  const pScaleU = 1 / computeMaxAbs(primarySim.U, 1.0);
  const pScaleY = 1 / computeMaxAbs(primarySim.Y, 1.0);
  const pScaleWQ = 1 / computeMaxAbs(primarySim.W_Q, 0.7);
  const pScaleWK = 1 / computeMaxAbs(primarySim.W_K, 0.7);
  const pScaleWV = 1 / computeMaxAbs(primarySim.W_V, 0.7);
  const pScaleWO = 1 / computeMaxAbs(primarySim.W_O, 0.15);
  const pScaleQ = 1 / computeMaxAbs(primarySim.Q, 1.2);
  const pScaleK = 1 / computeMaxAbs(primarySim.K, 1.2);
  const pScaleV = 1 / computeMaxAbs(primarySim.V, 1.2);
  const pScaleScores = 1 / Math.max(primaryMaxAbsScore, 1.6);
  const pScaleVOut = 1 / computeMaxAbs(primarySim.Concat, 0.8);

  const primaryQHeads: AtlasSlice[] = [];
  for (let h = 0; h < 32; h++) {
    primaryQHeads.push({
      rect: makeRect(h * dHead, 16, dHead, T1),
      scale: pScaleQ,
      mode: 0,
    });
  }

  const primaryKvK0: AtlasSlice[] = [];
  const primaryKvV0: AtlasSlice[] = [];
  const primaryKvK1: AtlasSlice[] = [];
  const primaryKvV1: AtlasSlice[] = [];
  for (let k = 0; k < K1; k++) {
    primaryKvK0.push({ rect: makeRect(96, 48 + k, dHead, 1), scale: pScaleK, mode: 0 });
    primaryKvV0.push({ rect: makeRect(100, 48 + k, dHead, 1), scale: pScaleV, mode: 0 });
    primaryKvK1.push({ rect: makeRect(104, 48 + k, dHead, 1), scale: pScaleK, mode: 0 });
    primaryKvV1.push({ rect: makeRect(108, 48 + k, dHead, 1), scale: pScaleV, mode: 0 });
  }

  const primary: TwoTowerGqaPrimaryAtlasSlices = {
    tokens: T1,
    prefixTokens: P1,
    keyTokens: K1,
    attentionMaskMode: primarySim.attentionMaskMode,
    maskedScoreCellCount: primaryMaskedScores,
    maskedSoftmaxCellCount: primaryMaskedSoftmax,
    input: { rect: makeRect(112, 56, d, T1), scale: pScaleX, mode: 0 },
    modulatedInput: primarySim.X_mod
      ? { rect: makeRect(120, 56, d, T1), scale: pScaleXMod, mode: 0 }
      : null,
    rmsnorm: { rect: makeRect(0, 72, d, T1), scale: pScaleU, mode: 0 },
    residual: { rect: makeRect(8, 72, d, T1), scale: pScaleY, mode: 0 },
    wQ: { rect: makeRect(0, 0, qConcatDim, d), scale: pScaleWQ, mode: 0 },
    wK: { rect: makeRect(112, 48, 8, d), scale: pScaleWK, mode: 0 },
    wV: { rect: makeRect(120, 48, 8, d), scale: pScaleWV, mode: 0 },
    wO: { rect: makeRect(0, 8, qConcatDim, d), scale: pScaleWO, mode: 0 },
    qHeads: primaryQHeads,
    kvK: [primaryKvK0, primaryKvK1],
    kvV: [primaryKvV0, primaryKvV1],
    scores: [
      { rect: makeRect(0, 48, K1, T1), scale: pScaleScores, mode: 0 },
      { rect: makeRect(48, 48, K1, T1), scale: pScaleScores, mode: 0 },
    ],
    softmax: [
      { rect: makeRect(24, 48, K1, T1), scale: 1.0, mode: 1 },
      { rect: makeRect(72, 48, K1, T1), scale: 1.0, mode: 1 },
    ],
    vOut: [
      { rect: makeRect(0, 32, 64, T1), scale: pScaleVOut, mode: 0 },
      { rect: makeRect(64, 32, 64, T1), scale: pScaleVOut, mode: 0 },
    ],
    concat: { rect: makeRect(0, 32, qConcatDim, T1), scale: pScaleVOut, mode: 0 },
  };

  // ==========================================================================
  // 2. SECONDARY CONTEXT PROVENANCE TOWER PACKING (y = 72..127)
  // ==========================================================================
  let secondary: TwoTowerGqaSecondaryAtlasSlices | null = null;

  if (secondarySim !== null) {
    const T2 = secondarySim.tokens;
    const K2 = secondarySim.keyTokens;
    const P2 = secondarySim.prefixTokens;

    // 2a. Secondary X, U, Y, active Q[g0/g1], active V_out[g0/g1] at y = 72..72+T2-1
    for (let t = 0; t < T2; t++) {
      for (let i = 0; i < d; i++) {
        setTexel(16 + i, 72 + t, secondarySim.X[i * T2 + t]);
        setTexel(24 + i, 72 + t, secondarySim.U[i * T2 + t]);
        setTexel(32 + i, 72 + t, secondarySim.Y[i * T2 + t]);
      }
      for (let r = 0; r < dHead; r++) {
        setTexel(40 + r, 72 + t, secondarySim.Q[(g0 * dHead + r) * T2 + t]);
        setTexel(44 + r, 72 + t, secondarySim.Q[(g1 * dHead + r) * T2 + t]);
        setTexel(48 + r, 72 + t, secondarySim.V_out[(g0 * dHead + r) * T2 + t]);
        setTexel(52 + r, 72 + t, secondarySim.V_out[(g1 * dHead + r) * T2 + t]);
      }
    }

    // 2b. Secondary W_K (x=56..63, y=72..79) & W_V (x=56..63, y=80..87)
    for (let j = 0; j < d; j++) {
      for (let outRow = 0; outRow < 8; outRow++) {
        setTexel(56 + outRow, 72 + j, secondarySim.W_K[outRow * d + j]);
        setTexel(56 + outRow, 80 + j, secondarySim.W_V[outRow * d + j]);
      }
    }

    // 2c. Secondary Scores & Softmax [K2<=24 cols × T2<=16 rows] at y = 88..88+T2-1
    let secMaxAbsScore = 1.2;
    let secMaskedScores = 0;
    let secMaskedSoftmax = 0;

    for (let grp = 0; grp < 2; grp++) {
      const h = scoreHeads[grp];
      const headOff = h * T2 * K2;
      for (let q = 0; q < T2; q++) {
        for (let k = 0; k < K2; k++) {
          const visible = isAttentionKeyVisible(secondarySim.attentionMaskMode, P2, q, k, K2);
          const s = secondarySim.Scores[headOff + q * K2 + k];
          if (visible && Number.isFinite(s) && Math.abs(s) > secMaxAbsScore) {
            secMaxAbsScore = Math.abs(s);
          }
          if (!visible) {
            secMaskedScores++;
            secMaskedSoftmax++;
          }
          setTexel(scoreOffsets[grp] + k, 88 + q, s, !visible);
          const sm = secondarySim.Softmax[headOff + q * K2 + k];
          setTexel(softOffsets[grp] + k, 88 + q, sm, !visible);
        }
      }
    }

    // 2d. Secondary KV0 & KV1 (4 cols × K2<=24 rows) at x=96..111, y=88..88+K2-1
    for (let g = 0; g < 2; g++) {
      const kBaseX = g === 0 ? 96 : 104;
      const vBaseX = g === 0 ? 100 : 108;
      for (let k = 0; k < K2; k++) {
        for (let r = 0; r < dHead; r++) {
          const row = g * dHead + r;
          setTexel(kBaseX + r, 88 + k, secondarySim.K[row * K2 + k]);
          setTexel(vBaseX + r, 88 + k, secondarySim.V[row * K2 + k]);
        }
      }
    }

    // 2e. Secondary Q[0..31] [128, T2] -> x=0..127, y=112..112+T2-1
    for (let t = 0; t < T2; t++) {
      for (let c = 0; c < qConcatDim; c++) {
        setTexel(c, 112 + t, secondarySim.Q[c * T2 + t]);
      }
    }

    const sScaleX = 1 / computeMaxAbs(secondarySim.X, 1.0);
    const sScaleU = 1 / computeMaxAbs(secondarySim.U, 1.0);
    const sScaleY = 1 / computeMaxAbs(secondarySim.Y, 1.0);
    const sScaleWK = 1 / computeMaxAbs(secondarySim.W_K, 0.7);
    const sScaleWV = 1 / computeMaxAbs(secondarySim.W_V, 0.7);
    const sScaleQ = 1 / computeMaxAbs(secondarySim.Q, 1.2);
    const sScaleK = 1 / computeMaxAbs(secondarySim.K, 1.2);
    const sScaleV = 1 / computeMaxAbs(secondarySim.V, 1.2);
    const sScaleScores = 1 / Math.max(secMaxAbsScore, 1.6);
    const sScaleVOut = 1 / computeMaxAbs(secondarySim.V_out, 0.8);

    const secQHeads: AtlasSlice[] = [];
    for (let h = 0; h < 32; h++) {
      secQHeads.push({ rect: makeRect(h * dHead, 112, dHead, T2), scale: sScaleQ, mode: 0 });
    }
    const secKvK0: AtlasSlice[] = [];
    const secKvV0: AtlasSlice[] = [];
    const secKvK1: AtlasSlice[] = [];
    const secKvV1: AtlasSlice[] = [];
    for (let k = 0; k < K2; k++) {
      secKvK0.push({ rect: makeRect(96, 88 + k, dHead, 1), scale: sScaleK, mode: 0 });
      secKvV0.push({ rect: makeRect(100, 88 + k, dHead, 1), scale: sScaleV, mode: 0 });
      secKvK1.push({ rect: makeRect(104, 88 + k, dHead, 1), scale: sScaleK, mode: 0 });
      secKvV1.push({ rect: makeRect(108, 88 + k, dHead, 1), scale: sScaleV, mode: 0 });
    }

    secondary = {
      kind: 'full_sim',
      tokens: T2,
      prefixTokens: P2,
      keyTokens: K2,
      attentionMaskMode: secondarySim.attentionMaskMode,
      maskedScoreCellCount: secMaskedScores,
      maskedSoftmaxCellCount: secMaskedSoftmax,
      input: { rect: makeRect(16, 72, d, T2), scale: sScaleX, mode: 0 },
      rmsnorm: { rect: makeRect(24, 72, d, T2), scale: sScaleU, mode: 0 },
      residual: { rect: makeRect(32, 72, d, T2), scale: sScaleY, mode: 0 },
      wK: { rect: makeRect(56, 72, 8, d), scale: sScaleWK, mode: 0 },
      wV: { rect: makeRect(56, 80, 8, d), scale: sScaleWV, mode: 0 },
      activeQ: [
        { rect: makeRect(40, 72, dHead, T2), scale: sScaleQ, mode: 0 },
        { rect: makeRect(44, 72, dHead, T2), scale: sScaleQ, mode: 0 },
      ],
      activeVOut: [
        { rect: makeRect(48, 72, dHead, T2), scale: sScaleVOut, mode: 0 },
        { rect: makeRect(52, 72, dHead, T2), scale: sScaleVOut, mode: 0 },
      ],
      qHeads: secQHeads,
      kvK: [secKvK0, secKvK1],
      kvV: [secKvV0, secKvV1],
      scores: [
        { rect: makeRect(0, 88, K2, T2), scale: sScaleScores, mode: 0 },
        { rect: makeRect(48, 88, K2, T2), scale: sScaleScores, mode: 0 },
      ],
      softmax: [
        { rect: makeRect(24, 88, K2, T2), scale: 1.0, mode: 1 },
        { rect: makeRect(72, 88, K2, T2), scale: 1.0, mode: 1 },
      ],
    };
  } else if (secondaryKvCacheFallback !== null && secondaryKvCacheFallback.seqLen > 0) {
    const K2 = secondaryKvCacheFallback.seqLen;
    const stride = secondaryKvCacheFallback.maxTokens;
    for (let g = 0; g < 2; g++) {
      const kBaseX = g === 0 ? 96 : 104;
      const vBaseX = g === 0 ? 100 : 108;
      for (let k = 0; k < K2; k++) {
        for (let r = 0; r < dHead; r++) {
          const row = g * dHead + r;
          setTexel(kBaseX + r, 88 + k, secondaryKvCacheFallback.K[row * stride + k]);
          setTexel(vBaseX + r, 88 + k, secondaryKvCacheFallback.V[row * stride + k]);
        }
      }
    }
    const sScaleK = 1 / computeMaxAbs(secondaryKvCacheFallback.K, 1.2);
    const sScaleV = 1 / computeMaxAbs(secondaryKvCacheFallback.V, 1.2);
    const secKvK0: AtlasSlice[] = [];
    const secKvV0: AtlasSlice[] = [];
    const secKvK1: AtlasSlice[] = [];
    const secKvV1: AtlasSlice[] = [];
    for (let k = 0; k < K2; k++) {
      secKvK0.push({ rect: makeRect(96, 88 + k, dHead, 1), scale: sScaleK, mode: 0 });
      secKvV0.push({ rect: makeRect(100, 88 + k, dHead, 1), scale: sScaleV, mode: 0 });
      secKvK1.push({ rect: makeRect(104, 88 + k, dHead, 1), scale: sScaleK, mode: 0 });
      secKvV1.push({ rect: makeRect(108, 88 + k, dHead, 1), scale: sScaleV, mode: 0 });
    }
    secondary = {
      kind: 'persistent_cache_only',
      tokens: 0,
      prefixTokens: K2,
      keyTokens: K2,
      attentionMaskMode: null,
      maskedScoreCellCount: 0,
      maskedSoftmaxCellCount: 0,
      input: null,
      rmsnorm: null,
      residual: null,
      wK: null,
      wV: null,
      activeQ: null,
      activeVOut: null,
      qHeads: null,
      kvK: [secKvK0, secKvK1],
      kvV: [secKvV0, secKvV1],
      scores: null,
      softmax: null,
    };
  }

  return {
    width: TWOTOWER_GQA_ATLAS_WIDTH,
    height: TWOTOWER_GQA_ATLAS_HEIGHT,
    data,
    logicalQueryHeadCount: 32,
    logicalKvHeadCount: 2,
    headDim: 4,
    group0Head: g0,
    group1Head: g1,
    primary,
    secondary,
  };
}

interface NamedRect {
  readonly name: string;
  readonly rect: TexRect;
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

function rectToTexelBox(
  name: string,
  rect: TexRect,
  width: number,
  height: number,
  errors: string[],
): NamedRect {
  const [u0, v0, uW, vH] = rect;
  const eps = 1e-6;
  if (
    !Number.isFinite(u0) ||
    !Number.isFinite(v0) ||
    !Number.isFinite(uW) ||
    !Number.isFinite(vH)
  ) {
    errors.push(`${name}: non-finite TexRect [${rect.join(', ')}]`);
  }
  if (u0 < -eps || v0 < -eps || uW <= 0 || vH <= 0 || u0 + uW > 1 + eps || v0 + vH > 1 + eps) {
    errors.push(`${name}: TexRect out of [0,1] bounds: [${u0}, ${v0}, ${uW}, ${vH}]`);
  }
  const x0 = Math.round(u0 * width);
  const y0 = Math.round(v0 * height);
  const x1 = Math.round((u0 + uW) * width);
  const y1 = Math.round((v0 + vH) * height);
  if (x0 < 0 || y0 < 0 || x1 > width || y1 > height || x1 <= x0 || y1 <= y0) {
    errors.push(`${name}: texel box [${x0},${y0}..${x1},${y1}) out of [0..${width})×[0..${height})`);
  }
  return { name, rect, x0, y0, x1, y1 };
}

/**
 * Exhaustively verify every `TexRect` in a `TwoTowerGqaValueAtlas`:
 *   1. Exact `128×128` RG32F dimensions (`32,768` floats = `131,072` bytes).
 *   2. Every slice (including intentional alias `primary.concat`) sits strictly inside `[0, 1]`.
 *   3. ZERO distinct non-aliasing slices overlap in texel coordinates.
 */
export function verifyTwoTowerGqaAtlasLayout(
  atlas: TwoTowerGqaValueAtlas,
): TwoTowerGqaAtlasLayoutVerification {
  const errors: string[] = [];
  const { width, height, data, primary, secondary } = atlas;

  if (width !== TWOTOWER_GQA_ATLAS_WIDTH || height !== TWOTOWER_GQA_ATLAS_HEIGHT) {
    errors.push(
      `Expected atlas dimensions ${TWOTOWER_GQA_ATLAS_WIDTH}×${TWOTOWER_GQA_ATLAS_HEIGHT}, got ${width}×${height}`,
    );
  }
  const expectedFloats = width * height * 2;
  if (data.length !== expectedFloats) {
    errors.push(`Expected data.length=${expectedFloats}, got ${data.length}`);
  }

  const nonAliasing: NamedRect[] = [];
  const aliases: NamedRect[] = [];

  const pushSlice = (name: string, slice: AtlasSlice | null | undefined, isAlias = false): void => {
    if (!slice) return;
    const box = rectToTexelBox(name, slice.rect, width, height, errors);
    if (isAlias) aliases.push(box);
    else nonAliasing.push(box);
  };

  pushSlice('primary.input', primary.input);
  pushSlice('primary.modulatedInput', primary.modulatedInput);
  pushSlice('primary.rmsnorm', primary.rmsnorm);
  pushSlice('primary.residual', primary.residual);
  pushSlice('primary.wQ', primary.wQ);
  pushSlice('primary.wK', primary.wK);
  pushSlice('primary.wV', primary.wV);
  pushSlice('primary.wO', primary.wO);
  primary.qHeads.forEach((s, h) => pushSlice(`primary.qHeads[${h}]`, s));
  for (let g = 0; g < 2; g++) {
    primary.kvK[g].forEach((s, k) => pushSlice(`primary.kvK[${g}][${k}]`, s));
    primary.kvV[g].forEach((s, k) => pushSlice(`primary.kvV[${g}][${k}]`, s));
  }
  pushSlice('primary.scores[0]', primary.scores[0]);
  pushSlice('primary.scores[1]', primary.scores[1]);
  pushSlice('primary.softmax[0]', primary.softmax[0]);
  pushSlice('primary.softmax[1]', primary.softmax[1]);
  pushSlice('primary.vOut[0]', primary.vOut[0]);
  pushSlice('primary.vOut[1]', primary.vOut[1]);
  // Intentional alias over vOut[0] + vOut[1]
  pushSlice('primary.concat', primary.concat, true);

  if (secondary !== null) {
    pushSlice('secondary.input', secondary.input);
    pushSlice('secondary.rmsnorm', secondary.rmsnorm);
    pushSlice('secondary.residual', secondary.residual);
    pushSlice('secondary.wK', secondary.wK);
    pushSlice('secondary.wV', secondary.wV);
    if (secondary.activeQ) {
      pushSlice('secondary.activeQ[0]', secondary.activeQ[0]);
      pushSlice('secondary.activeQ[1]', secondary.activeQ[1]);
    }
    if (secondary.activeVOut) {
      pushSlice('secondary.activeVOut[0]', secondary.activeVOut[0]);
      pushSlice('secondary.activeVOut[1]', secondary.activeVOut[1]);
    }
    secondary.qHeads?.forEach((s, h) => pushSlice(`secondary.qHeads[${h}]`, s));
    for (let g = 0; g < 2; g++) {
      secondary.kvK[g].forEach((s, k) => pushSlice(`secondary.kvK[${g}][${k}]`, s));
      secondary.kvV[g].forEach((s, k) => pushSlice(`secondary.kvV[${g}][${k}]`, s));
    }
    if (secondary.scores) {
      pushSlice('secondary.scores[0]', secondary.scores[0]);
      pushSlice('secondary.scores[1]', secondary.scores[1]);
    }
    if (secondary.softmax) {
      pushSlice('secondary.softmax[0]', secondary.softmax[0]);
      pushSlice('secondary.softmax[1]', secondary.softmax[1]);
    }
  }

  for (let i = 0; i < nonAliasing.length; i++) {
    const a = nonAliasing[i];
    for (let j = i + 1; j < nonAliasing.length; j++) {
      const b = nonAliasing[j];
      const overlapX = Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0);
      const overlapY = Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0);
      if (overlapX > 0 && overlapY > 0) {
        errors.push(
          `Overlap between ${a.name} [${a.x0},${a.y0}..${a.x1},${a.y1}) and ${b.name} [${b.x0},${b.y0}..${b.x1},${b.y1})`,
        );
      }
    }
  }

  return Object.freeze({
    valid: errors.length === 0,
    width,
    height,
    byteLength: data.byteLength,
    checkedNonAliasingSliceCount: nonAliasing.length,
    checkedAliasSliceCount: aliases.length,
    errors: Object.freeze(errors),
  });
}
