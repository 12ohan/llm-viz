/**
 * Packs live MoE (128+1) simulator Float32 tensors into a single 256×64 RG32F WebGL2
 * atlas texture for SheetsPass using the shared atlasPacker.
 *
 * Texel layout (width = 256, height = 64):
 *   y =  0.. 7 : Input X [T<=16 × 8] at x=0..7 (y=0..15)
 *                RMSNorm U [T<=16 × 8] at x=8..15 (y=0..15)
 *                Routed Sum [T<=16 × 8] at x=16..23 (y=0..15)
 *                Shared Out [T<=16 × 8] at x=24..31 (y=0..15)
 *                MoE Out [T<=16 × 8] at x=32..39 (y=0..15)
 *                Residual Y [T<=16 × 8] at x=40..47 (y=0..15)
 *                Router Bank (8×16 = 128) at x=48..63 (y=0..7)
 *                Top-6 Weights (1×6) at x=48..53 (y=8)
 *                Focused Routed W_up [4×8] at x=64..71 (y=0..3)
 *                Focused Routed Pre-Act h [4×1] at x=72 (y=0..3)
 *                Focused Routed ReLU² a [4×1] at x=73 (y=0..3)
 *                Focused Routed W_down [8×4] at x=74..77 (y=0..7)
 *                Shared W_up [8×8] at x=80..87 (y=0..7)
 *                Shared ReLU² [8×1] at x=88 (y=0..7)
 *                Shared W_down [8×8] at x=90..97 (y=0..7)
 *                W_router [128 × 8] packed as 8 cols × 128? Wait: atlas height is 64,
 *                so pack W_router transposed or in 2 strips? Wait: SheetsPass maps U→col, V→row!
 *                So if W_router sheet has gridCols=8, gridRows=16 (or atlas height=160!),
 *                making atlas height = 160 (256×160 RG32F = 320 KiB) lets W_router sit cleanly
 *                at (x=0..7, y=32..159) with exact 128 rows × 8 cols and zero subsampling!
 */

import {
  evaluateSingleRoutedExpert,
  MOE_SIM_CONFIG,
  type MoeSimResult,
} from '../math/moeSim';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
} from './atlasPacker';

export interface MoeValueAtlas extends Rg32fValueAtlas {
  readonly input: AtlasSlice;
  readonly rmsnorm: AtlasSlice;
  readonly wRouter: AtlasSlice;
  readonly routerBankTiles: readonly AtlasSlice[]; // 128 individual [1×1] slices (E000..E127)
  readonly topkUnbiased: AtlasSlice; // [1 × 6] unbiased sigmoid scores (bias stripped before norm)
  readonly topkWeights: AtlasSlice; // [1 × 6] normalized ×2.5 router weights
  readonly activeWeightedOut: readonly AtlasSlice[]; // 6 × [1 × 8] per-branch w_j·y_j outputs
  readonly routedWUp: AtlasSlice; // [4 × 8]
  readonly routedPre: AtlasSlice; // [4 × 1]
  readonly routedAct: AtlasSlice; // [4 × 1]
  readonly routedWDown: AtlasSlice; // [8 × 4]
  readonly routedSum: AtlasSlice; // [T × 8]
  readonly sharedWUp: AtlasSlice; // [8 × 8]
  readonly sharedAct: AtlasSlice; // [8 × 1]
  readonly sharedWDown: AtlasSlice; // [8 × 8]
  readonly sharedOut: AtlasSlice; // [T × 8]
  readonly moeOut: AtlasSlice; // [T × 8]
  readonly residual: AtlasSlice; // [T × 8]
}

export function buildMoeValueAtlas(
  sim: MoeSimResult,
  focusToken: number,
  focusExpert: number,
): MoeValueAtlas {
  const W = 128;
  const H = 160;
  const { d, dExp, dShared, nRoutedExperts, topK, tokens: T } = sim;
  const tokIdx = Math.max(0, Math.min(T - 1, focusToken));
  const expId = Math.max(0, Math.min(nRoutedExperts - 1, focusExpert));

  const atlas = createRg32fAtlas(W, H);
  const { setTexel, makeRect } = atlas;

  // 1. Sequence tensors [rows=T, cols=8] at y = 0..15
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < d; i++) {
      setTexel(i, t, sim.X[i * T + t]);
      setTexel(8 + i, t, sim.U[i * T + t]);
      setTexel(16 + i, t, sim.y_routed[i * T + t]);
      setTexel(24 + i, t, sim.y_shared[i * T + t]);
      setTexel(32 + i, t, sim.y_moe[i * T + t]);
      setTexel(40 + i, t, sim.Y[i * T + t]);
    }
  }

  const input: AtlasSlice = {
    rect: makeRect(0, 0, d, T),
    scale: 1 / computeMaxAbs(sim.X, 0.4),
    mode: 0,
  };
  const rmsnorm: AtlasSlice = {
    rect: makeRect(8, 0, d, T),
    scale: 1 / computeMaxAbs(sim.U, 0.4),
    mode: 0,
  };
  const routedSum: AtlasSlice = {
    rect: makeRect(16, 0, d, T),
    scale: 1 / computeMaxAbs(sim.y_routed, 0.35),
    mode: 0,
  };
  const sharedOut: AtlasSlice = {
    rect: makeRect(24, 0, d, T),
    scale: 1 / computeMaxAbs(sim.y_shared, 0.35),
    mode: 0,
  };
  const moeOut: AtlasSlice = {
    rect: makeRect(32, 0, d, T),
    scale: 1 / computeMaxAbs(sim.y_moe, 0.35),
    mode: 0,
  };
  const residual: AtlasSlice = {
    rect: makeRect(40, 0, d, T),
    scale: 1 / computeMaxAbs(sim.Y, 0.45),
    mode: 0,
  };

  // 2. Router Bank (8 rows × 16 cols = 128 experts) at x = 48..63, y = 0..7
  // Active experts show their scaled weight/choice energy; idle experts store choice score + mask=0.55 for subdued read
  const activeSet = new Set<number>();
  for (let j = 0; j < topK; j++) {
    activeSet.add(sim.topk_indices[tokIdx * topK + j]);
  }
  const routerBankTiles: AtlasSlice[] = [];
  for (let e = 0; e < nRoutedExperts; e++) {
    const r = Math.floor(e / MOE_SIM_CONFIG.bankCols);
    const c = e % MOE_SIM_CONFIG.bankCols;
    const isAct = activeSet.has(e);
    const choiceScore = sim.scores_for_choice[e * T + tokIdx];
    const clampedPos = Math.max(0.05, Math.min(1.0, choiceScore));
    setTexel(48 + c, r, isAct ? Math.max(0.68, clampedPos) : clampedPos * 0.42, false);
    routerBankTiles.push({
      rect: makeRect(48 + c, r, 1, 1),
      scale: 1.0,
      mode: 1, // positive probability/score ramp
    });
  }

  // 3. Top-6 Weights [1 × 6] at x = 48..53, y = 8 & Unbiased Sigmoid Scores [1 × 6] at x = 54..59, y = 8
  let maxTopW = 0.5;
  let maxRawS = 0.5;
  for (let j = 0; j < topK; j++) {
    const w = sim.topk_weights[tokIdx * topK + j];
    const rs = sim.topk_raw_scores[tokIdx * topK + j];
    if (w > maxTopW) maxTopW = w;
    if (rs > maxRawS) maxRawS = rs;
    setTexel(48 + j, 8, w);
    setTexel(54 + j, 8, rs);
  }
  const topkWeights: AtlasSlice = {
    rect: makeRect(48, 8, topK, 1),
    scale: 1 / Math.max(0.65, maxTopW),
    mode: 1,
  };
  const topkUnbiased: AtlasSlice = {
    rect: makeRect(54, 8, topK, 1),
    scale: 1 / Math.max(0.65, maxRawS),
    mode: 1,
  };

  // 3B. Six Independent Active Routed Branch Weighted Outputs (w_j · y_j, [1 × 8] each) at x = 100..107, y = 0..5
  let maxBranchOut = 0.25;
  for (let j = 0; j < topK; j++) {
    const w = sim.topk_weights[tokIdx * topK + j];
    const yBase = (tokIdx * topK + j) * d;
    for (let o = 0; o < d; o++) {
      const val = w * sim.active_y[yBase + o];
      if (Math.abs(val) > maxBranchOut) maxBranchOut = Math.abs(val);
      setTexel(100 + o, j, val, false);
    }
  }
  const activeWeightedOut: AtlasSlice[] = [];
  for (let j = 0; j < topK; j++) {
    activeWeightedOut.push({
      rect: makeRect(100, j, d, 1),
      scale: 1 / maxBranchOut,
      mode: 0,
    });
  }

  // 4. Focused Routed Expert Detail (active or on-demand idle preview) at x = 64..77, y = 0..7
  const focusedEval = evaluateSingleRoutedExpert(sim, expId, tokIdx);

  let maxWUp = 0.35;
  for (let h = 0; h < dExp; h++) {
    for (let i = 0; i < d; i++) {
      const v = sim.W_up[(expId * dExp + h) * d + i];
      if (Math.abs(v) > maxWUp) maxWUp = Math.abs(v);
      setTexel(64 + i, h, v, false);
    }
  }
  const routedWUp: AtlasSlice = {
    rect: makeRect(64, 0, d, dExp),
    scale: 1 / maxWUp,
    mode: 0,
  };

  let maxPre = 0.35;
  let maxAct = 0.35;
  for (let h = 0; h < dExp; h++) {
    const hv = focusedEval.h[h];
    const av = focusedEval.a[h];
    if (Math.abs(hv) > maxPre) maxPre = Math.abs(hv);
    if (Math.abs(av) > maxAct) maxAct = Math.abs(av);
    setTexel(72, h, hv, false);
    setTexel(73, h, av, false);
  }
  const routedPre: AtlasSlice = {
    rect: makeRect(72, 0, 1, dExp),
    scale: 1 / maxPre,
    mode: 0,
  };
  const routedAct: AtlasSlice = {
    rect: makeRect(73, 0, 1, dExp),
    scale: 1 / maxAct,
    mode: 1,
  };

  let maxWDown = 0.3;
  for (let o = 0; o < d; o++) {
    for (let h = 0; h < dExp; h++) {
      const v = sim.W_down[(expId * d + o) * dExp + h];
      if (Math.abs(v) > maxWDown) maxWDown = Math.abs(v);
      setTexel(74 + h, o, v, false);
    }
  }
  const routedWDown: AtlasSlice = {
    rect: makeRect(74, 0, dExp, d),
    scale: 1 / maxWDown,
    mode: 0,
  };

  // 5. Parallel Always-On Double-Width Shared Expert at x = 80..97, y = 0..7
  for (let h = 0; h < dShared; h++) {
    for (let i = 0; i < d; i++) {
      setTexel(80 + i, h, sim.W_up_s[h * d + i]);
    }
    const sa = sim.shared_a[h * T + tokIdx];
    setTexel(88, h, sa, false);
  }
  for (let o = 0; o < d; o++) {
    for (let h = 0; h < dShared; h++) {
      setTexel(90 + h, o, sim.W_down_s[o * dShared + h]);
    }
  }

  const sharedWUp: AtlasSlice = {
    rect: makeRect(80, 0, d, dShared),
    scale: 1 / computeMaxAbs(sim.W_up_s, 0.35),
    mode: 0,
  };
  const sharedAct: AtlasSlice = {
    rect: makeRect(88, 0, 1, dShared),
    scale: 1 / computeMaxAbs(sim.shared_a, 0.35),
    mode: 1,
  };
  const sharedWDown: AtlasSlice = {
    rect: makeRect(90, 0, dShared, d),
    scale: 1 / computeMaxAbs(sim.W_down_s, 0.25),
    mode: 0,
  };

  // 6. W_router [rows=128, cols=8] at x = 0..7, y = 32..159 (exact full 128×8 matrix)
  for (let e = 0; e < nRoutedExperts; e++) {
    for (let i = 0; i < d; i++) {
      setTexel(i, 32 + e, sim.W_router[e * d + i]);
    }
  }
  const wRouter: AtlasSlice = {
    rect: makeRect(0, 32, d, nRoutedExperts),
    scale: 1 / computeMaxAbs(sim.W_router, 0.45),
    mode: 0,
  };

  return {
    width: W,
    height: H,
    data: atlas.data,
    input,
    rmsnorm,
    wRouter,
    routerBankTiles,
    topkUnbiased,
    topkWeights,
    activeWeightedOut,
    routedWUp,
    routedPre,
    routedAct,
    routedWDown,
    routedSum,
    sharedWUp,
    sharedAct,
    sharedWDown,
    sharedOut,
    moeOut,
    residual,
  };
}
