/**
 * Packs live Mamba-2 simulator Float32 tensors into a single 256×64 RG32F WebGL2
 * atlas texture for SheetsPass using the shared atlasPacker.
 *
 * Texel layout (width = 256, height = 64):
 *   y =  0.. 7 : W_in [rows=8, cols=208] at x=0..207;  H_prev [4×4] at x=210..213;  H_curr [4×4] at x=216..219
 *                64-tile production SSM bank summary [8×8] at x=224..231
 *   y =  8..15 : W_out [rows=8, cols=64] at x=0..63;  conv_buf [rows=4, cols=128] at x=64..191 (y=8..11)
 *   y = 16..31 : xBC_pre [rows=T, cols=128] at x=0..127;  z [rows=T, cols=64] at x=128..191;
 *                dt [rows=T, cols=16] at x=192..207;  Input X [rows=T, cols=8] at x=208..215;
 *                RMSNorm U [rows=T, cols=8] at x=216..223;  Residual Y [rows=T, cols=8] at x=224..231
 *   y = 32..47 : x [rows=T, cols=64] at x=0..63 (16 heads × 4 cols);
 *                B [rows=T, cols=32] at x=64..95 (8 groups × 4 cols);
 *                C [rows=T, cols=32] at x=96..127 (8 groups × 4 cols);
 *                y [rows=T, cols=64] at x=128..191;
 *                y_norm [rows=T, cols=64] at x=192..255
 */

import type { MambaSimResult } from '../math/mambaSim';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
} from './atlasPacker';

export interface MambaValueAtlas extends Rg32fValueAtlas {
  readonly input: AtlasSlice;
  readonly rmsnorm: AtlasSlice;
  readonly wIn: AtlasSlice;
  readonly z: AtlasSlice;
  readonly xbcPre: AtlasSlice;
  readonly convBufTaps: readonly AtlasSlice[]; // 4 rolling taps [0..3], each [1 × 128]
  readonly xHeads: readonly AtlasSlice[]; // 16 miniature heads, each [T × 4]
  readonly bGroups: AtlasSlice; // [T × 32]
  readonly cGroups: AtlasSlice; // [T × 32]
  readonly dt: AtlasSlice; // [T × 16]
  readonly hPrev: AtlasSlice; // active head [4 × 4]
  readonly hCurr: AtlasSlice; // active head [4 × 4]
  readonly ssmBankTiles: readonly AtlasSlice[]; // 64 production bank tiles mapped to 16 miniature heads
  readonly y: AtlasSlice; // [T × 64]
  readonly yNorm: AtlasSlice; // [T × 64]
  readonly wOut: AtlasSlice; // [8 × 64]
  readonly residual: AtlasSlice; // [T × 8]
}

export function buildMambaValueAtlas(
  sim: MambaSimResult,
  activeHead: number,
  stateTokenIndex?: number,
): MambaValueAtlas {
  const W = 256;
  const H = 64;
  const { d, nHeads, nGroups, dHead, dInner, dState, convKernel, convDim, inProjDim, tokens: T } = sim;
  const head = Math.max(0, Math.min(nHeads - 1, activeHead));
  const tokIdx = Math.max(0, Math.min(T - 1, stateTokenIndex ?? T - 1));
  const stateStride = nHeads * dHead * dState;

  const atlas = createRg32fAtlas(W, H);
  const { setTexel, makeRect } = atlas;

  // 1. W_in [rows=d(8), cols=inProjDim(208)] at (0, 0)
  for (let r = 0; r < d; r++) {
    for (let c = 0; c < inProjDim; c++) {
      setTexel(c, r, sim.W_in[c * d + r]);
    }
  }
  const wIn: AtlasSlice = {
    rect: makeRect(0, 0, inProjDim, d),
    scale: 1 / computeMaxAbs(sim.W_in, 0.35),
    mode: 0,
  };

  // 2. Active head H_prev [4×4] at (210, 0) and H_curr [4×4] at (216, 0)
  const hScale = 1 / Math.max(computeMaxAbs(sim.H_history, 0.25), 0.35);
  for (let p = 0; p < dHead; p++) {
    for (let n = 0; n < dState; n++) {
      const localIdx = (head * dHead + p) * dState + n;
      const prevVal = tokIdx === 0 ? 0 : sim.H_history[(tokIdx - 1) * stateStride + localIdx];
      const currVal = sim.H_history[tokIdx * stateStride + localIdx];
      setTexel(210 + n, p, prevVal);
      setTexel(216 + n, p, currVal);
    }
  }
  const hPrev: AtlasSlice = { rect: makeRect(210, 0, dState, dHead), scale: hScale, mode: 0 };
  const hCurr: AtlasSlice = { rect: makeRect(216, 0, dState, dHead), scale: hScale, mode: 0 };

  // 3. 64-tile production SSM bank [8×8] at (224, 0) — each tile shows live signed state energy of its mapped miniature head
  const ssmBankTiles: AtlasSlice[] = [];
  for (let tile = 0; tile < 64; tile++) {
    const gx = Math.floor(tile / 8); // 0..7 groups
    const local = tile % 8; // 0..7 prod heads per group
    const miniHead = gx * 2 + Math.floor(local / 4);
    const pSample = local % dHead;
    let meanVal = 0;
    for (let n = 0; n < dState; n++) {
      meanVal += sim.H_history[tokIdx * stateStride + (miniHead * dHead + pSample) * dState + n];
    }
    meanVal /= dState;
    setTexel(224 + gx, local, meanVal);
    ssmBankTiles.push({
      rect: makeRect(224 + gx, local, 1, 1),
      scale: hScale,
      mode: 0,
    });
  }

  // 4. W_out [rows=8, cols=64] at (0, 8)
  for (let i = 0; i < d; i++) {
    for (let c = 0; c < dInner; c++) {
      setTexel(c, 8 + i, sim.W_out[i * dInner + c]);
    }
  }
  const wOut: AtlasSlice = {
    rect: makeRect(0, 8, dInner, d),
    scale: 1 / computeMaxAbs(sim.W_out, 0.25),
    mode: 0,
  };

  // 5. Rolling 4-tap Conv1D buffer [rows=4, cols=128] at (64, 8), synchronized to tokIdx
  //    (bit-for-bit identical to sim.conv_buf when tokIdx === T - 1; causal zero-padded when srcT < 0)
  const convBufScale = 1 / computeMaxAbs(sim.xBC_pre, 0.35);
  const convBufTaps: AtlasSlice[] = [];
  for (let s = 0; s < convKernel; s++) {
    const lag = convKernel - 1 - s;
    const srcT = tokIdx - lag;
    for (let c = 0; c < convDim; c++) {
      const val = srcT >= 0 ? sim.xBC_pre[c * T + srcT] : 0;
      setTexel(64 + c, 8 + s, val);
    }
    convBufTaps.push({
      rect: makeRect(64, 8 + s, convDim, 1),
      scale: convBufScale,
      mode: 0,
    });
  }

  // 6. Sequence band y=16..31: xBC_pre, z, dt, Input X, RMSNorm U, Residual Y
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < convDim; c++) {
      setTexel(c, 16 + t, sim.xBC_pre[c * T + t]);
    }
    for (let c = 0; c < dInner; c++) {
      setTexel(128 + c, 16 + t, sim.z[c * T + t]);
    }
    for (let h = 0; h < nHeads; h++) {
      setTexel(192 + h, 16 + t, sim.dt[h * T + t]);
    }
    for (let i = 0; i < d; i++) {
      setTexel(208 + i, 16 + t, sim.X[i * T + t]);
      setTexel(216 + i, 16 + t, sim.U[i * T + t]);
      setTexel(224 + i, 16 + t, sim.Y[i * T + t]);
    }
  }

  const xbcPre: AtlasSlice = {
    rect: makeRect(0, 16, convDim, T),
    scale: convBufScale,
    mode: 0,
  };
  const z: AtlasSlice = {
    rect: makeRect(128, 16, dInner, T),
    scale: 1 / computeMaxAbs(sim.z, 0.35),
    mode: 0,
  };
  const dt: AtlasSlice = {
    rect: makeRect(192, 16, nHeads, T),
    scale: 1 / computeMaxAbs(sim.dt, 0.5),
    mode: 1, // strictly positive softplus step sizes [0..max]
  };
  const input: AtlasSlice = {
    rect: makeRect(208, 16, d, T),
    scale: 1 / computeMaxAbs(sim.X, 0.4),
    mode: 0,
  };
  const rmsnorm: AtlasSlice = {
    rect: makeRect(216, 16, d, T),
    scale: 1 / computeMaxAbs(sim.U, 0.4),
    mode: 0,
  };
  const residual: AtlasSlice = {
    rect: makeRect(224, 16, d, T),
    scale: 1 / computeMaxAbs(sim.Y, 0.4),
    mode: 0,
  };

  // 7. Sequence band y=32..47: x (16 heads × 4), B (8×4), C (8×4), y (64), y_norm (64)
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < dInner; c++) {
      setTexel(c, 32 + t, sim.x[c * T + t]);
      setTexel(128 + c, 32 + t, sim.y[c * T + t]);
      setTexel(192 + c, 32 + t, sim.y_norm[c * T + t]);
    }
    for (let k = 0; k < nGroups * dState; k++) {
      setTexel(64 + k, 32 + t, sim.B[k * T + t]);
      setTexel(96 + k, 32 + t, sim.C[k * T + t]);
    }
  }

  const xScale = 1 / computeMaxAbs(sim.x, 0.35);
  const xHeads: AtlasSlice[] = [];
  for (let h = 0; h < nHeads; h++) {
    xHeads.push({
      rect: makeRect(h * dHead, 32, dHead, T),
      scale: xScale,
      mode: 0,
    });
  }
  const bGroups: AtlasSlice = {
    rect: makeRect(64, 32, nGroups * dState, T),
    scale: 1 / computeMaxAbs(sim.B, 0.35),
    mode: 0,
  };
  const cGroups: AtlasSlice = {
    rect: makeRect(96, 32, nGroups * dState, T),
    scale: 1 / computeMaxAbs(sim.C, 0.35),
    mode: 0,
  };
  const y: AtlasSlice = {
    rect: makeRect(128, 32, dInner, T),
    scale: 1 / computeMaxAbs(sim.y, 0.35),
    mode: 0,
  };
  const yNorm: AtlasSlice = {
    rect: makeRect(192, 32, dInner, T),
    scale: 1 / computeMaxAbs(sim.y_norm, 0.4),
    mode: 0,
  };

  return {
    width: W,
    height: H,
    data: atlas.data,
    input,
    rmsnorm,
    wIn,
    z,
    xbcPre,
    convBufTaps,
    xHeads,
    bGroups,
    cGroups,
    dt,
    hPrev,
    hCurr,
    ssmBankTiles,
    y,
    yNorm,
    wOut,
    residual,
  };
}
