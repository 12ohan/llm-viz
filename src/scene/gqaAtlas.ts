import type { GqaSimResult } from '../math/gqaSim';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
  type TexRect,
} from './atlasPacker';

export type { TexRect };
export type GqaAtlasSlice = AtlasSlice;

export interface GqaValueAtlas extends Rg32fValueAtlas {
  readonly group0Head: number;
  readonly group1Head: number;
  readonly input: GqaAtlasSlice;
  readonly rmsnorm: GqaAtlasSlice;
  readonly residual: GqaAtlasSlice;
  readonly wQ: GqaAtlasSlice;
  readonly wK: GqaAtlasSlice;
  readonly wV: GqaAtlasSlice;
  readonly wO: GqaAtlasSlice;
  readonly qHeads: readonly GqaAtlasSlice[]; // 32 entries
  readonly kvK: readonly [readonly GqaAtlasSlice[], readonly GqaAtlasSlice[]]; // [group][token 0..T-1]
  readonly kvV: readonly [readonly GqaAtlasSlice[], readonly GqaAtlasSlice[]]; // [group][token 0..T-1]
  readonly scores: readonly [GqaAtlasSlice, GqaAtlasSlice]; // [group0Head, group1Head]
  readonly softmax: readonly [GqaAtlasSlice, GqaAtlasSlice]; // [group0Head, group1Head]
  readonly vOut: readonly [GqaAtlasSlice, GqaAtlasSlice]; // [group0 (64xT), group1 (64xT)]
  readonly concat: GqaAtlasSlice; // [128xT]
}

const ATLAS_W = 128;
const ATLAS_H = 64;

/**
 * Pack every live Float32 tensor from `GqaSimResult` for the active `T` and active
 * `(group0Head, group1Head)` into a single 128×64 RG32F texture atlas.
 *
 * Texel layout (all 1-to-1 exact with visual sheet gridCols × gridRows):
 *   y = 0..15:
 *     x =   0..7   Input X      (8 cols × T rows)
 *     x =   8..15  RMSNorm U    (8 cols × T rows)
 *     x =  16..23  Residual Y   (8 cols × T rows)
 *     x =  24..31  W_K (y=0..7) & W_V (y=8..15)  (8 cols × 8 rows each)
 *     x =  32..47  Scores[g0]   (T cols × T rows)
 *     x =  48..63  Softmax[g0]  (T cols × T rows)
 *     x =  64..79  Scores[g1]   (T cols × T rows)
 *     x =  80..95  Softmax[g1]  (T cols × T rows)
 *     x =  96..99  KV0 K        (4 cols × T rows; slice t at y=t)
 *     x = 100..103 KV0 V        (4 cols × T rows; slice t at y=t)
 *     x = 104..107 KV1 K        (4 cols × T rows; slice t at y=t)
 *     x = 108..111 KV1 V        (4 cols × T rows; slice t at y=t)
 *   y = 16..23:
 *     x =   0..127 W_Q          (128 cols × 8 rows)
 *   y = 24..31:
 *     x =   0..127 W_O          (128 cols × 8 rows)
 *   y = 32..47:
 *     x =   0..127 Q[0..31]     (32 heads × 4 cols × T rows)
 *   y = 48..63:
 *     x =   0..127 V_out / Concat (128 cols × T rows; Grp0=0..63, Grp1=64..127)
 */
export function buildGqaValueAtlas(
  sim: GqaSimResult,
  group0Head = 0,
  group1Head = 16,
): GqaValueAtlas {
  const g0 = Math.max(0, Math.min(15, Math.round(group0Head)));
  const g1 = Math.max(16, Math.min(31, Math.round(group1Head)));
  const T = sim.tokens;
  const { d, dHead, qConcatDim } = sim;

  const writer = createRg32fAtlas(ATLAS_W, ATLAS_H);
  const { data, setTexel, makeRect } = writer;

  // 1. Input X [8, T] -> x=0..7, y=0..T-1 (col=i, row=t)
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < d; i++) {
      setTexel(0 + i, t, sim.X[i * T + t]);
      setTexel(8 + i, t, sim.U[i * T + t]);
      setTexel(16 + i, t, sim.Y[i * T + t]);
    }
  }

  // 2. W_K [8, 8] (y=0..7) & W_V [8, 8] (y=8..15) at x=24..31 (col=outRow, row=j)
  for (let j = 0; j < d; j++) {
    for (let outRow = 0; outRow < 8; outRow++) {
      setTexel(24 + outRow, j, sim.W_K[outRow * d + j]);
      setTexel(24 + outRow, 8 + j, sim.W_V[outRow * d + j]);
    }
  }

  // 3. Scores & Softmax for Group 0 head (g0) & Group 1 head (g1)
  const scoreHeads: readonly [number, number] = [g0, g1];
  const scoreOffsets: readonly [number, number] = [32, 64];
  const softOffsets: readonly [number, number] = [48, 80];
  let maxAbsScore = 1.2;

  for (let grp = 0; grp < 2; grp++) {
    const h = scoreHeads[grp];
    const headOff = h * T * T;
    for (let q = 0; q < T; q++) {
      for (let k = 0; k < T; k++) {
        const s = sim.Scores[headOff + q * T + k];
        if (Number.isFinite(s) && Math.abs(s) > maxAbsScore) {
          maxAbsScore = Math.abs(s);
        }
        setTexel(scoreOffsets[grp] + k, q, s);
        const sm = sim.Softmax[headOff + q * T + k];
        const smIdx = (q * ATLAS_W + (softOffsets[grp] + k)) * 2;
        data[smIdx] = sm;
        data[smIdx + 1] = k > q ? 1.0 : 0.0;
      }
    }
  }

  // 4. Shared KV0 & KV1 (K & V tracks, 4 cols × T rows)
  for (let g = 0; g < 2; g++) {
    const kBaseX = g === 0 ? 96 : 104;
    const vBaseX = g === 0 ? 100 : 108;
    for (let t = 0; t < T; t++) {
      for (let r = 0; r < dHead; r++) {
        const row = g * dHead + r;
        setTexel(kBaseX + r, t, sim.K[row * T + t]);
        setTexel(vBaseX + r, t, sim.V[row * T + t]);
      }
    }
  }

  // 5. W_Q [128, 8] -> x=0..127, y=16..23 (col=outRow, row=j)
  for (let j = 0; j < d; j++) {
    for (let outRow = 0; outRow < qConcatDim; outRow++) {
      setTexel(outRow, 16 + j, sim.W_Q[outRow * d + j]);
    }
  }

  // 6. W_O [8, 128] -> x=0..127, y=24..31 (col=c, row=i)
  for (let i = 0; i < d; i++) {
    for (let c = 0; c < qConcatDim; c++) {
      setTexel(c, 24 + i, sim.W_O[i * qConcatDim + c]);
    }
  }

  // 7. All 32 Q heads [128, T] -> x=0..127, y=32..32+T-1 (col=h*4+r, row=t)
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < qConcatDim; c++) {
      setTexel(c, 32 + t, sim.Q[c * T + t]);
    }
  }

  // 8. V_out / Concat [128, T] -> x=0..127, y=48..48+T-1 (col=h*4+r, row=q)
  for (let q = 0; q < T; q++) {
    for (let c = 0; c < qConcatDim; c++) {
      setTexel(c, 48 + q, sim.Concat[c * T + q]);
    }
  }

  // Per-role normalization scales (smooth, never saturated)
  const scaleX = 1 / computeMaxAbs(sim.X, 1.0);
  const scaleU = 1 / computeMaxAbs(sim.U, 1.0);
  const scaleY = 1 / computeMaxAbs(sim.Y, 1.0);
  const scaleWQ = 1 / computeMaxAbs(sim.W_Q, 0.7);
  const scaleWK = 1 / computeMaxAbs(sim.W_K, 0.7);
  const scaleWV = 1 / computeMaxAbs(sim.W_V, 0.7);
  const scaleWO = 1 / computeMaxAbs(sim.W_O, 0.15);
  const scaleQ = 1 / computeMaxAbs(sim.Q, 1.2);
  const scaleK = 1 / computeMaxAbs(sim.K, 1.2);
  const scaleV = 1 / computeMaxAbs(sim.V, 1.2);
  const scaleScores = 1 / Math.max(maxAbsScore, 1.6);
  const scaleVOut = 1 / computeMaxAbs(sim.Concat, 0.8);

  const qHeads: GqaAtlasSlice[] = [];
  for (let h = 0; h < 32; h++) {
    qHeads.push({
      rect: makeRect(h * dHead, 32, dHead, T),
      scale: scaleQ,
      mode: 0,
    });
  }

  const kvK0: GqaAtlasSlice[] = [];
  const kvV0: GqaAtlasSlice[] = [];
  const kvK1: GqaAtlasSlice[] = [];
  const kvV1: GqaAtlasSlice[] = [];
  for (let t = 0; t < T; t++) {
    kvK0.push({ rect: makeRect(96, t, dHead, 1), scale: scaleK, mode: 0 });
    kvV0.push({ rect: makeRect(100, t, dHead, 1), scale: scaleV, mode: 0 });
    kvK1.push({ rect: makeRect(104, t, dHead, 1), scale: scaleK, mode: 0 });
    kvV1.push({ rect: makeRect(108, t, dHead, 1), scale: scaleV, mode: 0 });
  }

  return {
    width: ATLAS_W,
    height: ATLAS_H,
    data,
    group0Head: g0,
    group1Head: g1,
    input: { rect: makeRect(0, 0, d, T), scale: scaleX, mode: 0 },
    rmsnorm: { rect: makeRect(8, 0, d, T), scale: scaleU, mode: 0 },
    residual: { rect: makeRect(16, 0, d, T), scale: scaleY, mode: 0 },
    wQ: { rect: makeRect(0, 16, qConcatDim, d), scale: scaleWQ, mode: 0 },
    wK: { rect: makeRect(24, 0, 8, d), scale: scaleWK, mode: 0 },
    wV: { rect: makeRect(24, 8, 8, d), scale: scaleWV, mode: 0 },
    wO: { rect: makeRect(0, 24, qConcatDim, d), scale: scaleWO, mode: 0 },
    qHeads,
    kvK: [kvK0, kvK1],
    kvV: [kvV0, kvV1],
    scores: [
      { rect: makeRect(32, 0, T, T), scale: scaleScores, mode: 0 },
      { rect: makeRect(64, 0, T, T), scale: scaleScores, mode: 0 },
    ],
    softmax: [
      { rect: makeRect(48, 0, T, T), scale: 1.0, mode: 1 },
      { rect: makeRect(80, 0, T, T), scale: 1.0, mode: 1 },
    ],
    vOut: [
      { rect: makeRect(0, 48, 64, T), scale: scaleVOut, mode: 0 },
      { rect: makeRect(64, 48, 64, T), scale: scaleVOut, mode: 0 },
    ],
    concat: { rect: makeRect(0, 48, qConcatDim, T), scale: scaleVOut, mode: 0 },
  };
}
