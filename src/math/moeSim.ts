/**
 * Deterministic miniature simulator for NVIDIA Nemotron 3 Nano Granular MoE (128+1)
 * with Float32 tensor storage and rounded numerical formula readouts.
 *
 * Strictly executes NVIDIA's verified NemotronHMOE, NemotronHTopkRouter, and
 * non-gated NemotronHMLP (ReLU²) equations (`modeling_nemotron_h.py`):
 *   - 128 routed experts (8×16 bank), top-6 active (95.3% sparse, 122 idle = 0 contribution)
 *   - n_group = 1, topk_group = 1
 *   - Sigmoid router + e_score_correction_bias for Top-6 SELECTION ONLY
 *   - Unbiased sigmoid scores gathered for weighting -> normalized -> * 2.5 (routed_scaling_factor)
 *   - Non-gated ReLU² MLP: ONLY W_up + W_down (ZERO gate_proj, ZERO linear bias)
 *   - Always-on double-width shared expert (d_shared = 2 * d_exp, NO 2.5x scaling)
 *   - 0 B dynamic token cache (strictly per-token independent forward pass)
 *   - Pre-layer RMSNorm with eps = 1e-5 and learned non-unit scale gamma_in
 *
 * Educational note: Miniature live simulator (d=8, d_exp=4, d_shared=8) preserving
 * Nemotron 3 Nano's exact 128 routed / top-6 / 2.5x / 2x shared ratio. Production
 * weights are d=2,688, moe_intermediate_size=1,856, moe_shared_expert_intermediate_size=3,712.
 */

import { EDUCATIONAL_TOKENS } from './gqaSim';
import {
  fmtSigned,
  type LayerFormulaReadout,
  type StepperStepDef,
} from '../scene/sheetInspector';

export const MOE_SIM_CONFIG = {
  d: 8,
  nRoutedExperts: 128,
  bankRows: 8,
  bankCols: 16,
  topK: 6,
  nGroup: 1,
  topkGroup: 1,
  dExp: 4,
  dShared: 8, // strictly 2 * dExp (mirrors 3,712 = 2 * 1,856)
  routedScalingFactor: 2.5,
  normTopkProb: true,
  rmsNormEps: 1e-5,
  probDenomEps: 1e-20,
  maxTokens: 16,
  // Production Nemotron 3 Nano reference metadata
  prodHidden: 2688,
  prodRoutedWidth: 1856,
  prodSharedWidth: 3712,
  prodLayers: 23,
  prodParamsPerLayerB: 1.30,
  prodTotalParamsB: 29.8,
} as const;

export type MoeTensorKey =
  | 'input'
  | 'rmsnorm'
  | 'w_router'
  | 'router_bank'
  | 'topk_weights'
  | 'routed_w_up'
  | 'routed_pre'
  | 'routed_act'
  | 'routed_w_down'
  | 'routed_sum'
  | 'shared_w_up'
  | 'shared_act'
  | 'shared_w_down'
  | 'shared_out'
  | 'moe_out'
  | 'residual';

export interface MoeCellRef {
  readonly tensor: MoeTensorKey;
  /** Row index inside the tensor's display grid (often token t or channel/expert row). */
  readonly row: number;
  /** Column index inside the tensor's display grid. */
  readonly col: number;
  /** Focused token index t in [0..T-1] for token-local views (router bank, topk, expert detail). */
  readonly tokenIndex?: number;
  /** Expert ID in [0..127] when inspecting router bank, top-k slot, or routed expert detail. */
  readonly expertId?: number;
  /** Rank slot in [0..5] when inspecting topk_weights. */
  readonly rankSlot?: number;
}

export interface MoeSheetSpan {
  readonly tensor: MoeTensorKey;
  readonly rowStart: number;
  readonly rowEnd: number;
  readonly colStart: number;
  readonly colEnd: number;
  readonly expertId?: number;
  readonly tokenIndex?: number;
}

export interface MoeCellFormula extends LayerFormulaReadout {
  readonly cell: MoeCellRef;
  readonly upstreamSpans: readonly MoeSheetSpan[];
  readonly activeExpertId: number | null;
  readonly isIdleExpert: boolean;
}

export interface MoeSimFixture {
  /** Input X [d, T] indexed [i * T + t]. */
  readonly X?: Float32Array;
  /** RMSNorm scale gamma_in [d]. */
  readonly gamma_in?: Float32Array;
  /** Router weights [128, d] indexed [e * d + i]. */
  readonly W_router?: Float32Array;
  /** Router score correction bias [128]. */
  readonly e_score_correction_bias?: Float32Array;
  /** Routed expert up weights [128, dExp, d] indexed [(e * dExp + h) * d + i]. */
  readonly W_up?: Float32Array;
  /** Routed expert down weights [128, d, dExp] indexed [(e * d + o) * dExp + h]. */
  readonly W_down?: Float32Array;
  /** Shared expert up weights [dShared, d] indexed [h * d + i]. */
  readonly W_up_s?: Float32Array;
  /** Shared expert down weights [d, dShared] indexed [o * dShared + h]. */
  readonly W_down_s?: Float32Array;
}

export interface SingleRoutedExpertEval {
  readonly expertId: number;
  readonly tokenIndex: number;
  readonly isActive: boolean;
  /** 0..5 if active, or null if idle (rank >= 6). */
  readonly activeRank: number | null;
  /** Full 0..127 rank under scores_for_choice (0 = highest). */
  readonly choiceRank: number;
  readonly logit: number;
  readonly unbiasedScore: number;
  readonly bias: number;
  readonly choiceScore: number;
  /** 0 for idle experts; scaled weight (norm_w * 2.5) for active Top-6 experts. */
  readonly appliedWeight: number;
  /** Pre-activation h_e = W_up[e] @ U[:, t], shape [dExp]. */
  readonly h: Float32Array;
  /** Post-activation a_e = ReLU²(h_e), shape [dExp]. */
  readonly a: Float32Array;
  /** Output y_e = W_down[e] @ a_e, shape [d]. */
  readonly y: Float32Array;
}

export interface MoeSimResult {
  readonly tokens: number;
  readonly tokenLabels: readonly string[];
  readonly d: number;
  readonly nRoutedExperts: number;
  readonly topK: number;
  readonly dExp: number;
  readonly dShared: number;
  readonly routedScalingFactor: number;
  readonly eps: number;

  // Learned / synthetic deterministic parameters
  readonly gamma_in: Float32Array; // [d]
  readonly W_router: Float32Array; // [128, d]
  readonly e_score_correction_bias: Float32Array; // [128]
  readonly W_up: Float32Array; // [128, dExp, d]
  readonly W_down: Float32Array; // [128, d, dExp]
  readonly W_up_s: Float32Array; // [dShared, d]
  readonly W_down_s: Float32Array; // [d, dShared]

  // Forward-pass tensors (row-major by channel/expert, then token t)
  readonly X: Float32Array; // [d, T]
  readonly rms: Float32Array; // [T]
  readonly X_norm: Float32Array; // [d, T]
  readonly U: Float32Array; // [d, T]

  readonly router_logits: Float32Array; // [128, T]
  readonly scores: Float32Array; // [128, T] unbiased sigmoid(logits)
  readonly scores_for_choice: Float32Array; // [128, T] scores + e_score_correction_bias

  /** Top-6 winning expert IDs per token, indexed [t * topK + j], sorted descending by choice score. */
  readonly topk_indices: Int32Array; // [T, 6]
  /** Unbiased sigmoid scores gathered at topk_indices, indexed [t * topK + j]. */
  readonly topk_raw_scores: Float32Array; // [T, 6]
  /** Normalized probabilities (sum = 1.0 per token), indexed [t * topK + j]. */
  readonly topk_norm_probs: Float32Array; // [T, 6]
  /** Final routed weights = norm_probs * 2.5 (sum = 2.5 per token), indexed [t * topK + j]. */
  readonly topk_weights: Float32Array; // [T, 6]

  /** Active expert intermediate states per token & slot j in [0..5]. */
  readonly active_h: Float32Array; // [T, 6, dExp]
  readonly active_a: Float32Array; // [T, 6, dExp]
  readonly active_y: Float32Array; // [T, 6, d]

  /** Shared expert intermediate states (always on, 2× width). */
  readonly shared_h: Float32Array; // [dShared, T]
  readonly shared_a: Float32Array; // [dShared, T]
  readonly y_shared: Float32Array; // [d, T]

  /** Parallel aggregation & residual outputs. */
  readonly y_routed: Float32Array; // [d, T]
  readonly y_moe: Float32Array; // [d, T]
  readonly Y: Float32Array; // [d, T]
}

export function formatExpertId(expertId: number): string {
  return `E${String(Math.max(0, Math.min(127, expertId))).padStart(3, '0')}`;
}

/** Numerically stable logistic sigmoid. */
export function stableSigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/** Exact non-gated ReLU²: max(0, h)² with exact 0 for h <= 0. */
export function relu2(h: number): number {
  return h > 0 ? h * h : 0;
}

/** Deterministic PRNG (Mulberry32) so every session produces identical tensors. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeSignedUniform(rand: () => number, scale: number): number {
  return (rand() * 2 - 1) * scale;
}

interface DefaultMoeWeights {
  readonly X_full: Float32Array; // [d, 16]
  readonly gamma_in: Float32Array; // [d]
  readonly W_router: Float32Array; // [128, d]
  readonly e_score_correction_bias: Float32Array; // [128]
  readonly W_up: Float32Array; // [128, dExp, d]
  readonly W_down: Float32Array; // [128, d, dExp]
  readonly W_up_s: Float32Array; // [dShared, d]
  readonly W_down_s: Float32Array; // [d, dShared]
}

let cachedDefaultWeights: DefaultMoeWeights | null = null;
const defaultSimCache = new Map<number, MoeSimResult>();

function getDefaultWeights(): DefaultMoeWeights {
  if (cachedDefaultWeights) return cachedDefaultWeights;

  const { d, nRoutedExperts, dExp, dShared, maxTokens } = MOE_SIM_CONFIG;
  const rand = mulberry32(0x4e454d4f); // "NEMO"

  // Diverse deterministic input vectors X_full [d, 16] so each token routes to a distinct Top-6 set
  const X_full = new Float32Array(d * maxTokens);
  for (let t = 0; t < maxTokens; t++) {
    const angle = (t / maxTokens) * Math.PI * 2;
    for (let i = 0; i < d; i++) {
      const harmonic =
        0.52 * Math.sin(angle * (1 + (i % 3)) + i * 0.87) +
        0.38 * Math.cos(angle * (2 + (i % 2)) - i * 1.13);
      X_full[i * maxTokens + t] = Math.fround(harmonic + makeSignedUniform(rand, 0.36));
    }
  }

  // Deterministic non-unit RMSNorm scale gamma_in [d]
  const gamma_in = new Float32Array(d);
  const gammaSeed = [1.12, 0.88, 1.24, 0.93, 1.07, 0.82, 1.18, 0.96];
  for (let i = 0; i < d; i++) {
    gamma_in[i] = Math.fround(gammaSeed[i]);
  }

  // Router weights [128, d]: unit-ish diverse prototypes across 128 experts so logits spread cleanly
  const W_router = new Float32Array(nRoutedExperts * d);
  for (let e = 0; e < nRoutedExperts; e++) {
    let normSq = 0;
    const tmp = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      const v = makeSignedUniform(rand, 1.0);
      tmp[i] = v;
      normSq += v * v;
    }
    const inv = 1.45 / Math.sqrt(normSq + 1e-6);
    for (let i = 0; i < d; i++) {
      W_router[e * d + i] = Math.fround(tmp[i] * inv);
    }
  }

  // Deterministic non-zero e_score_correction_bias [128]: large enough (~±0.14) to
  // visibly swap close contenders near the Top-6 cutoff while remaining realistic
  const e_score_correction_bias = new Float32Array(nRoutedExperts);
  for (let e = 0; e < nRoutedExperts; e++) {
    const raw = makeSignedUniform(rand, 0.14);
    e_score_correction_bias[e] = Math.fround(Math.abs(raw) < 0.015 ? (raw >= 0 ? 0.025 : -0.025) : raw);
  }

  // Routed expert non-gated MLP weights (ONLY W_up [128, dExp, d] and W_down [128, d, dExp])
  // Balanced sign distribution guarantees a clear mix of h <= 0 (ReLU² = 0) and h > 0 (h²)
  const W_up = new Float32Array(nRoutedExperts * dExp * d);
  const W_down = new Float32Array(nRoutedExperts * d * dExp);
  for (let e = 0; e < nRoutedExperts; e++) {
    for (let h = 0; h < dExp; h++) {
      for (let i = 0; i < d; i++) {
        W_up[(e * dExp + h) * d + i] = Math.fround(makeSignedUniform(rand, 0.56));
      }
    }
    for (let o = 0; o < d; o++) {
      for (let h = 0; h < dExp; h++) {
        W_down[(e * d + o) * dExp + h] = Math.fround(makeSignedUniform(rand, 0.34));
      }
    }
  }

  // Always-on double-width shared expert (W_up_s [8, 8], W_down_s [8, 8])
  const W_up_s = new Float32Array(dShared * d);
  const W_down_s = new Float32Array(d * dShared);
  for (let h = 0; h < dShared; h++) {
    for (let i = 0; i < d; i++) {
      W_up_s[h * d + i] = Math.fround(makeSignedUniform(rand, 0.48));
    }
  }
  for (let o = 0; o < d; o++) {
    for (let h = 0; h < dShared; h++) {
      W_down_s[o * dShared + h] = Math.fround(makeSignedUniform(rand, 0.28));
    }
  }

  cachedDefaultWeights = {
    X_full,
    gamma_in,
    W_router,
    e_score_correction_bias,
    W_up,
    W_down,
    W_up_s,
    W_down_s,
  };
  return cachedDefaultWeights;
}

/**
 * Evaluate a single routed expert e (active OR idle) on token t on demand.
 * For idle experts, returns its hypothetical W_up/ReLU²/W_down activations while
 * strictly reporting isActive = false and appliedWeight = 0.
 */
export function evaluateSingleRoutedExpert(
  sim: MoeSimResult,
  expertId: number,
  tokenIndex: number,
): SingleRoutedExpertEval {
  const { d, dExp, nRoutedExperts, topK, tokens: T } = sim;
  const e = Math.max(0, Math.min(nRoutedExperts - 1, expertId));
  const t = Math.max(0, Math.min(T - 1, tokenIndex));

  // Find active rank (0..5) if present
  let activeRank: number | null = null;
  for (let j = 0; j < topK; j++) {
    if (sim.topk_indices[t * topK + j] === e) {
      activeRank = j;
      break;
    }
  }

  // Compute full 0..127 rank under scores_for_choice (deterministic tie-break: lower index wins)
  const myChoice = sim.scores_for_choice[e * T + t];
  let choiceRank = 0;
  for (let other = 0; other < nRoutedExperts; other++) {
    if (other === e) continue;
    const oc = sim.scores_for_choice[other * T + t];
    if (oc > myChoice || (oc === myChoice && other < e)) {
      choiceRank++;
    }
  }

  const h = new Float32Array(dExp);
  const a = new Float32Array(dExp);
  const y = new Float32Array(d);

  if (activeRank !== null) {
    const hBase = (t * topK + activeRank) * dExp;
    const yBase = (t * topK + activeRank) * d;
    for (let k = 0; k < dExp; k++) {
      h[k] = sim.active_h[hBase + k];
      a[k] = sim.active_a[hBase + k];
    }
    for (let o = 0; o < d; o++) {
      y[o] = sim.active_y[yBase + o];
    }
  } else {
    for (let k = 0; k < dExp; k++) {
      let sum = 0;
      const wRow = (e * dExp + k) * d;
      for (let i = 0; i < d; i++) {
        sum += sim.W_up[wRow + i] * sim.U[i * T + t];
      }
      const hv = Math.fround(sum);
      h[k] = hv;
      a[k] = Math.fround(relu2(hv));
    }
    for (let o = 0; o < d; o++) {
      let sum = 0;
      const wRow = (e * d + o) * dExp;
      for (let k = 0; k < dExp; k++) {
        sum += sim.W_down[wRow + k] * a[k];
      }
      y[o] = Math.fround(sum);
    }
  }

  return {
    expertId: e,
    tokenIndex: t,
    isActive: activeRank !== null,
    activeRank,
    choiceRank,
    logit: sim.router_logits[e * T + t],
    unbiasedScore: sim.scores[e * T + t],
    bias: sim.e_score_correction_bias[e],
    choiceScore: myChoice,
    appliedWeight: activeRank !== null ? sim.topk_weights[t * topK + activeRank] : 0,
    h,
    a,
    y,
  };
}

/**
 * Execute the deterministic miniature NemotronH MoE (128+1) forward pass for T tokens (1..16).
 * Strictly per-token independent (0 B token cache).
 */
export function runMoeSim(tokens: number, fixture?: MoeSimFixture): MoeSimResult {
  const T = Math.max(1, Math.min(MOE_SIM_CONFIG.maxTokens, Math.round(tokens)));
  if (!fixture) {
    const cached = defaultSimCache.get(T);
    if (cached) return cached;
  }

  const def = getDefaultWeights();
  const { d, nRoutedExperts, topK, dExp, dShared, routedScalingFactor, rmsNormEps, probDenomEps } =
    MOE_SIM_CONFIG;

  const gamma_in = fixture?.gamma_in ? new Float32Array(fixture.gamma_in) : def.gamma_in;
  const W_router = fixture?.W_router ? new Float32Array(fixture.W_router) : def.W_router;
  const e_score_correction_bias = fixture?.e_score_correction_bias
    ? new Float32Array(fixture.e_score_correction_bias)
    : def.e_score_correction_bias;
  const W_up = fixture?.W_up ? new Float32Array(fixture.W_up) : def.W_up;
  const W_down = fixture?.W_down ? new Float32Array(fixture.W_down) : def.W_down;
  const W_up_s = fixture?.W_up_s ? new Float32Array(fixture.W_up_s) : def.W_up_s;
  const W_down_s = fixture?.W_down_s ? new Float32Array(fixture.W_down_s) : def.W_down_s;

  // Slice X [d, T]
  const X = new Float32Array(d * T);
  if (fixture?.X) {
    X.set(fixture.X.subarray(0, d * T));
  } else {
    for (let i = 0; i < d; i++) {
      for (let t = 0; t < T; t++) {
        X[i * T + t] = def.X_full[i * MOE_SIM_CONFIG.maxTokens + t];
      }
    }
  }

  const rms = new Float32Array(T);
  const X_norm = new Float32Array(d * T);
  const U = new Float32Array(d * T);

  const router_logits = new Float32Array(nRoutedExperts * T);
  const scores = new Float32Array(nRoutedExperts * T);
  const scores_for_choice = new Float32Array(nRoutedExperts * T);

  const topk_indices = new Int32Array(T * topK);
  const topk_raw_scores = new Float32Array(T * topK);
  const topk_norm_probs = new Float32Array(T * topK);
  const topk_weights = new Float32Array(T * topK);

  const active_h = new Float32Array(T * topK * dExp);
  const active_a = new Float32Array(T * topK * dExp);
  const active_y = new Float32Array(T * topK * d);

  const shared_h = new Float32Array(dShared * T);
  const shared_a = new Float32Array(dShared * T);
  const y_shared = new Float32Array(d * T);

  const y_routed = new Float32Array(d * T);
  const y_moe = new Float32Array(d * T);
  const Y = new Float32Array(d * T);

  const candidateOrder = new Int32Array(nRoutedExperts);

  // Per-token independent loop (0 B cross-token cache)
  for (let t = 0; t < T; t++) {
    // 1. Pre-Layer RMSNorm (eps = 1e-5) + learned non-unit gamma_in
    let meanSq = 0;
    for (let i = 0; i < d; i++) {
      const xv = X[i * T + t];
      meanSq += xv * xv;
    }
    meanSq /= d;
    const denom = Math.sqrt(meanSq + rmsNormEps);
    rms[t] = Math.fround(denom);

    for (let i = 0; i < d; i++) {
      const xn = X[i * T + t] / denom;
      X_norm[i * T + t] = Math.fround(xn);
      U[i * T + t] = Math.fround(xn * gamma_in[i]);
    }

    // 2. NemotronHTopkRouter: logits = W_router @ U[:, t], sigmoid, + e_score_correction_bias
    for (let e = 0; e < nRoutedExperts; e++) {
      let logit = 0;
      const wOff = e * d;
      for (let i = 0; i < d; i++) {
        logit += W_router[wOff + i] * U[i * T + t];
      }
      const l32 = Math.fround(logit);
      const s32 = Math.fround(stableSigmoid(l32));
      const c32 = Math.fround(s32 + e_score_correction_bias[e]);
      router_logits[e * T + t] = l32;
      scores[e * T + t] = s32;
      scores_for_choice[e * T + t] = c32;
      candidateOrder[e] = e;
    }

    // Select Top-6 strictly via scores_for_choice (deterministic tie-break: lower expert index)
    for (let j = 0; j < topK; j++) {
      let bestPos = j;
      let bestExpert = candidateOrder[j];
      let bestScore = scores_for_choice[bestExpert * T + t];
      for (let m = j + 1; m < nRoutedExperts; m++) {
        const cand = candidateOrder[m];
        const candScore = scores_for_choice[cand * T + t];
        if (candScore > bestScore || (candScore === bestScore && cand < bestExpert)) {
          bestPos = m;
          bestExpert = cand;
          bestScore = candScore;
        }
      }
      if (bestPos !== j) {
        const tmp = candidateOrder[j];
        candidateOrder[j] = candidateOrder[bestPos];
        candidateOrder[bestPos] = tmp;
      }
      topk_indices[t * topK + j] = bestExpert;
      // Gather UNBIASED sigmoid scores (NOT scores_for_choice!)
      topk_raw_scores[t * topK + j] = scores[bestExpert * T + t];
    }

    // Normalize gathered unbiased probabilities & apply routed_scaling_factor = 2.5
    let rawSum = 0;
    for (let j = 0; j < topK; j++) {
      rawSum += topk_raw_scores[t * topK + j];
    }
    const probDenom = rawSum + probDenomEps;
    for (let j = 0; j < topK; j++) {
      const pNorm = topk_raw_scores[t * topK + j] / probDenom;
      topk_norm_probs[t * topK + j] = Math.fround(pNorm);
      topk_weights[t * topK + j] = Math.fround(pNorm * routedScalingFactor);
    }

    // 3. Parallel Non-Gated ReLU² Routed Experts (ONLY top-6 active; 122 idle = 0 contribution)
    for (let j = 0; j < topK; j++) {
      const e = topk_indices[t * topK + j];
      const w = topk_weights[t * topK + j];
      const hBase = (t * topK + j) * dExp;
      const yBase = (t * topK + j) * d;

      for (let k = 0; k < dExp; k++) {
        let sum = 0;
        const wRow = (e * dExp + k) * d;
        for (let i = 0; i < d; i++) {
          sum += W_up[wRow + i] * U[i * T + t];
        }
        const hv = Math.fround(sum);
        active_h[hBase + k] = hv;
        active_a[hBase + k] = Math.fround(relu2(hv));
      }

      for (let o = 0; o < d; o++) {
        let sum = 0;
        const wRow = (e * d + o) * dExp;
        for (let k = 0; k < dExp; k++) {
          sum += W_down[wRow + k] * active_a[hBase + k];
        }
        const yv = Math.fround(sum);
        active_y[yBase + o] = yv;
        y_routed[o * T + t] = Math.fround(y_routed[o * T + t] + w * yv);
      }
    }

    // 4. Parallel Always-On Double-Width Shared Expert (NO routed_scaling_factor!)
    for (let k = 0; k < dShared; k++) {
      let sum = 0;
      const wRow = k * d;
      for (let i = 0; i < d; i++) {
        sum += W_up_s[wRow + i] * U[i * T + t];
      }
      const hv = Math.fround(sum);
      shared_h[k * T + t] = hv;
      shared_a[k * T + t] = Math.fround(relu2(hv));
    }

    for (let o = 0; o < d; o++) {
      let sum = 0;
      const wRow = o * dShared;
      for (let k = 0; k < dShared; k++) {
        sum += W_down_s[wRow + k] * shared_a[k * T + t];
      }
      const ysv = Math.fround(sum);
      y_shared[o * T + t] = ysv;

      // 5. Parallel Merge + Residual
      const moeSum = Math.fround(y_routed[o * T + t] + ysv);
      y_moe[o * T + t] = moeSum;
      Y[o * T + t] = Math.fround(X[o * T + t] + moeSum);
    }
  }

  const result: MoeSimResult = {
    tokens: T,
    tokenLabels: EDUCATIONAL_TOKENS.slice(0, T),
    d,
    nRoutedExperts,
    topK,
    dExp,
    dShared,
    routedScalingFactor,
    eps: rmsNormEps,
    gamma_in,
    W_router,
    e_score_correction_bias,
    W_up,
    W_down,
    W_up_s,
    W_down_s,
    X,
    rms,
    X_norm,
    U,
    router_logits,
    scores,
    scores_for_choice,
    topk_indices,
    topk_raw_scores,
    topk_norm_probs,
    topk_weights,
    active_h,
    active_a,
    active_y,
    shared_h,
    shared_a,
    y_shared,
    y_routed,
    y_moe,
    Y,
  };

  if (!fixture) {
    defaultSimCache.set(T, result);
  }
  return result;
}

/**
 * Deterministic 6 active routed expert IDs [0..127] for the active token `tokens - 1`.
 * Unified with `runMoeSim(tokens)` so cutaway geometry, inspector, and tests always agree.
 */
export function activeRoutedExpertIndices(tokens: number): number[] {
  const clamped = Math.max(1, Math.min(MOE_SIM_CONFIG.maxTokens, Math.round(tokens)));
  const sim = runMoeSim(clamped);
  const t = clamped - 1;
  const out: number[] = [];
  for (let j = 0; j < sim.topK; j++) {
    out.push(sim.topk_indices[t * sim.topK + j]);
  }
  return out;
}

function findActiveExpertRank(sim: MoeSimResult, expertId: number, tokenIndex: number): number | null {
  const base = tokenIndex * sim.topK;
  for (let j = 0; j < sim.topK; j++) {
    if (sim.topk_indices[base + j] === expertId) {
      return j;
    }
  }
  return null;
}

function resolveRoutedSumExpert(
  sim: MoeSimResult,
  tokenIndex: number,
  explicitExpertId?: number,
  fallbackFocusExpert?: number,
): { expertId: number; activeRank: number } {
  const t = Math.max(0, Math.min(sim.tokens - 1, tokenIndex));
  if (explicitExpertId !== undefined) {
    const clamped = Math.max(0, Math.min(sim.nRoutedExperts - 1, explicitExpertId));
    const rank = findActiveExpertRank(sim, clamped, t);
    if (rank !== null) return { expertId: clamped, activeRank: rank };
  }
  if (fallbackFocusExpert !== undefined) {
    const clamped = Math.max(0, Math.min(sim.nRoutedExperts - 1, fallbackFocusExpert));
    const rank = findActiveExpertRank(sim, clamped, t);
    if (rank !== null) return { expertId: clamped, activeRank: rank };
  }
  return { expertId: sim.topk_indices[t * sim.topK], activeRank: 0 };
}

/** Infer the expert ID owned by a MoE cell if applicable, else null. */
export function inferMoeCellExpert(
  cell: MoeCellRef | null,
  sim?: MoeSimResult,
  fallbackFocusExpert?: number,
): number | null {
  if (!cell) return null;
  if (cell.tensor === 'routed_sum' && sim) {
    const t = Math.max(0, Math.min(sim.tokens - 1, cell.tokenIndex ?? cell.row));
    return resolveRoutedSumExpert(sim, t, cell.expertId, fallbackFocusExpert).expertId;
  }
  if (cell.expertId !== undefined) {
    return Math.max(0, Math.min(MOE_SIM_CONFIG.nRoutedExperts - 1, cell.expertId));
  }
  if (cell.tensor === 'router_bank') {
    return Math.max(
      0,
      Math.min(
        MOE_SIM_CONFIG.nRoutedExperts - 1,
        cell.row * MOE_SIM_CONFIG.bankCols + cell.col,
      ),
    );
  }
  if (cell.tensor === 'topk_weights' && sim) {
    const t = Math.max(0, Math.min(sim.tokens - 1, cell.tokenIndex ?? sim.tokens - 1));
    const slot = Math.max(0, Math.min(sim.topK - 1, cell.rankSlot ?? cell.col));
    return sim.topk_indices[t * sim.topK + slot];
  }
  return null;
}

/** Build exact rounded numerical formula readout + upstream dependency spans for any MoE cell. */
export function describeMoeCellFormula(
  sim: MoeSimResult,
  rawCell: MoeCellRef,
  defaultFocusToken: number,
  defaultFocusExpert: number,
): MoeCellFormula {
  const { d, dExp, dShared, topK, tokens: T } = sim;
  const tokIdx = Math.max(0, Math.min(T - 1, rawCell.tokenIndex ?? defaultFocusToken));
  const tokLabel = sim.tokenLabels[tokIdx] ?? `t${tokIdx}`;
  const inferredExp = inferMoeCellExpert(rawCell, sim, defaultFocusExpert);
  const expId = Math.max(0, Math.min(sim.nRoutedExperts - 1, inferredExp ?? defaultFocusExpert));
  const expTag = formatExpertId(expId);

  switch (rawCell.tensor) {
    case 'input': {
      const t = Math.max(0, Math.min(T - 1, rawCell.row));
      const ch = Math.max(0, Math.min(d - 1, rawCell.col));
      const val = sim.X[ch * T + t];
      const tl = sim.tokenLabels[t] ?? `t${t}`;
      return {
        cell: { ...rawCell, row: t, col: ch, tokenIndex: t },
        tensorLabel: 'Input X',
        coordLabel: `t${t} "${tl}" · ch ${ch}`,
        headBadge: 'Residual Spine · d=8 mini / 2,688 prod',
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 3),
        equation: `X[${ch}, t${t}] = ${fmtSigned(val, 3)}  (RMS = ${sim.rms[t].toFixed(3)})`,
        subEquation: `Per-token hidden state from residual spine (0 B sequence cache in MoE).`,
        upstreamSpans: [],
        activeExpertId: null,
        isIdleExpert: false,
      };
    }

    case 'rmsnorm': {
      const t = Math.max(0, Math.min(T - 1, rawCell.row));
      const ch = Math.max(0, Math.min(d - 1, rawCell.col));
      const xv = sim.X[ch * T + t];
      const xn = sim.X_norm[ch * T + t];
      const g = sim.gamma_in[ch];
      const uv = sim.U[ch * T + t];
      const tl = sim.tokenLabels[t] ?? `t${t}`;
      return {
        cell: { ...rawCell, row: t, col: ch, tokenIndex: t },
        tensorLabel: 'RMSNorm U',
        coordLabel: `t${t} "${tl}" · ch ${ch}`,
        headBadge: 'ε = 1e−5 · Pre-MoE',
        value: uv,
        isMasked: false,
        valueText: fmtSigned(uv, 3),
        equation: `U[${ch}, t${t}] = (${fmtSigned(xv, 3)} / ${sim.rms[t].toFixed(3)}) × γ[${ch}](${g.toFixed(2)}) = ${fmtSigned(uv, 3)}`,
        subEquation: `X_norm = ${fmtSigned(xn, 3)} · RMS = √(mean(X²) + 1e−5) = ${sim.rms[t].toFixed(3)} · Forks in parallel to Router & Shared Expert.`,
        upstreamSpans: [
          { tensor: 'input', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, tokenIndex: t },
        ],
        activeExpertId: null,
        isIdleExpert: false,
      };
    }

    case 'w_router': {
      const e = Math.max(0, Math.min(sim.nRoutedExperts - 1, rawCell.row));
      const ch = Math.max(0, Math.min(d - 1, rawCell.col));
      const wv = sim.W_router[e * d + ch];
      return {
        cell: { ...rawCell, row: e, col: ch, expertId: e },
        tensorLabel: 'W_router (FP32)',
        coordLabel: `${formatExpertId(e)} · ch ${ch}`,
        headBadge: 'Always FP32 · No Bias',
        value: wv,
        isMasked: false,
        valueText: fmtSigned(wv, 3),
        equation: `W_router[${formatExpertId(e)}, ${ch}] = ${fmtSigned(wv, 3)}  (e_bias[${formatExpertId(e)}] = ${fmtSigned(sim.e_score_correction_bias[e], 3)})`,
        subEquation: `Prod: 128 × 2,688 linear gate kept strictly FP32 (never FP8) for stable Top-6 selection.`,
        upstreamSpans: [
          { tensor: 'rmsnorm', rowStart: tokIdx, rowEnd: tokIdx, colStart: ch, colEnd: ch, tokenIndex: tokIdx },
        ],
        activeExpertId: e,
        isIdleExpert: false,
      };
    }

    case 'router_bank': {
      const ev = evaluateSingleRoutedExpert(sim, expId, tokIdx);
      const bankRow = Math.floor(expId / MOE_SIM_CONFIG.bankCols);
      const bankCol = expId % MOE_SIM_CONFIG.bankCols;
      const statusText = ev.isActive
        ? `ACTIVE · Top-6 Rank #${(ev.activeRank ?? 0) + 1} · w = ${ev.appliedWeight.toFixed(3)}`
        : `IDLE · Rank #${ev.choiceRank + 1}/128 (> 6) · 0 contribution`;
      return {
        cell: {
          tensor: 'router_bank',
          row: bankRow,
          col: bankCol,
          tokenIndex: tokIdx,
          expertId: expId,
        },
        tensorLabel: ev.isActive ? `Router ${expTag} (Active)` : `Router ${expTag} (Idle)`,
        coordLabel: `t${tokIdx} "${tokLabel}" · ${expTag}`,
        headBadge: statusText,
        value: ev.choiceScore,
        isMasked: !ev.isActive,
        valueText: ev.isActive ? `w=${ev.appliedWeight.toFixed(3)}` : `skip (0)`,
        equation: ev.isActive
          ? `choice = σ(${fmtSigned(ev.logit, 2)})(${ev.unbiasedScore.toFixed(3)}) + bias(${fmtSigned(ev.bias, 3)}) = ${ev.choiceScore.toFixed(3)} → Top-6 #${(ev.activeRank ?? 0) + 1}`
          : `choice = σ(${fmtSigned(ev.logit, 2)})(${ev.unbiasedScore.toFixed(3)}) + bias(${fmtSigned(ev.bias, 3)}) = ${ev.choiceScore.toFixed(3)} → Rank #${ev.choiceRank + 1} (skipped, 0 weight)`,
        subEquation: ev.isActive
          ? `Weight gathered from UNBIASED σ = ${ev.unbiasedScore.toFixed(3)} (bias excluded) → norm × 2.5 = ${ev.appliedWeight.toFixed(3)}.`
          : `Not in Top-6 for t${tokIdx}; detail sheets preview its hypothetical ReLU² weights on demand while layer contribution stays 0.`,
        upstreamSpans: [
          { tensor: 'rmsnorm', rowStart: tokIdx, rowEnd: tokIdx, colStart: 0, colEnd: d - 1, tokenIndex: tokIdx },
          { tensor: 'w_router', rowStart: expId, rowEnd: expId, colStart: 0, colEnd: d - 1, expertId: expId },
        ],
        activeExpertId: expId,
        isIdleExpert: !ev.isActive,
      };
    }

    case 'topk_weights': {
      const slot = Math.max(0, Math.min(topK - 1, rawCell.rankSlot ?? rawCell.col));
      const winnerId = sim.topk_indices[tokIdx * topK + slot];
      const wTag = formatExpertId(winnerId);
      const rawS = sim.topk_raw_scores[tokIdx * topK + slot];
      const normP = sim.topk_norm_probs[tokIdx * topK + slot];
      const scaledW = sim.topk_weights[tokIdx * topK + slot];
      const choiceS = sim.scores_for_choice[winnerId * T + tokIdx];
      const bias = sim.e_score_correction_bias[winnerId];
      let rawSum = 0;
      for (let m = 0; m < topK; m++) rawSum += sim.topk_raw_scores[tokIdx * topK + m];
      const bankRow = Math.floor(winnerId / MOE_SIM_CONFIG.bankCols);
      const bankCol = winnerId % MOE_SIM_CONFIG.bankCols;
      return {
        cell: {
          tensor: 'topk_weights',
          row: 0,
          col: slot,
          tokenIndex: tokIdx,
          expertId: winnerId,
          rankSlot: slot,
        },
        tensorLabel: `Top-6 Weight #${slot + 1}`,
        coordLabel: `t${tokIdx} "${tokLabel}" · ${wTag}`,
        headBadge: `Unbiased σ → norm × 2.5`,
        value: scaledW,
        isMasked: false,
        valueText: scaledW.toFixed(3),
        equation: `w[${wTag}] = (${rawS.toFixed(3)} / ${rawSum.toFixed(3)}) × 2.5 = ${normP.toFixed(3)} × 2.5 = ${scaledW.toFixed(3)}`,
        subEquation: `Selected via choice score ${choiceS.toFixed(3)} (σ + bias ${fmtSigned(bias, 3)}), but weighted using unbiased σ = ${rawS.toFixed(3)}. Σw = 2.500.`,
        upstreamSpans: [
          {
            tensor: 'router_bank',
            rowStart: bankRow,
            rowEnd: bankRow,
            colStart: bankCol,
            colEnd: bankCol,
            expertId: winnerId,
            tokenIndex: tokIdx,
          },
        ],
        activeExpertId: winnerId,
        isIdleExpert: false,
      };
    }

    case 'routed_w_up': {
      const hRow = Math.max(0, Math.min(dExp - 1, rawCell.row));
      const iCol = Math.max(0, Math.min(d - 1, rawCell.col));
      const ev = evaluateSingleRoutedExpert(sim, expId, tokIdx);
      const wv = sim.W_up[(expId * dExp + hRow) * d + iCol];
      return {
        cell: { tensor: 'routed_w_up', row: hRow, col: iCol, tokenIndex: tokIdx, expertId: expId },
        tensorLabel: `${expTag} W_up`,
        coordLabel: `h${hRow} × in${iCol} · ${ev.isActive ? `Active #${(ev.activeRank ?? 0) + 1}` : 'Idle Preview'}`,
        headBadge: `Non-Gated · 1,856 prod (No gate_proj)`,
        value: wv,
        isMasked: !ev.isActive,
        valueText: fmtSigned(wv, 3),
        equation: `W_up[${expTag}, ${hRow}, ${iCol}] = ${fmtSigned(wv, 3)} × U[${iCol}, t${tokIdx}](${fmtSigned(sim.U[iCol * T + tokIdx], 3)})`,
        subEquation: ev.isActive
          ? `Only 2 matrices per expert (W_up + W_down, zero bias). Contributes to h[${hRow}] = ${fmtSigned(ev.h[hRow], 3)}.`
          : `Idle expert (${expTag}, rank #${ev.choiceRank + 1}): previewing weights only; zero compute/contribution in layer sum.`,
        upstreamSpans: [
          { tensor: 'rmsnorm', rowStart: tokIdx, rowEnd: tokIdx, colStart: iCol, colEnd: iCol, tokenIndex: tokIdx },
        ],
        activeExpertId: expId,
        isIdleExpert: !ev.isActive,
      };
    }

    case 'routed_pre': {
      const hRow = Math.max(0, Math.min(dExp - 1, rawCell.row));
      const ev = evaluateSingleRoutedExpert(sim, expId, tokIdx);
      const hv = ev.h[hRow];
      const av = ev.a[hRow];
      return {
        cell: { tensor: 'routed_pre', row: hRow, col: 0, tokenIndex: tokIdx, expertId: expId },
        tensorLabel: `${expTag} Pre-Act h`,
        coordLabel: `t${tokIdx} "${tokLabel}" · h${hRow}`,
        headBadge: hv > 0 ? `Positive → passes ReLU²` : `≤ 0 → zeroed by ReLU²`,
        value: hv,
        isMasked: !ev.isActive,
        valueText: fmtSigned(hv, 3),
        equation: `h[${expTag}, ${hRow}] = Σᵢ W_up[${hRow}, i]·U[i, t${tokIdx}] = ${fmtSigned(hv, 3)}  → ReLU² = ${av.toFixed(3)}`,
        subEquation: ev.isActive
          ? `Non-gated pre-activation (no SwiGLU gate branch). Negative entries become exact 0.`
          : `Hypothetical preview for idle expert ${expTag} (rank #${ev.choiceRank + 1}, 0 layer contribution).`,
        upstreamSpans: [
          { tensor: 'routed_w_up', rowStart: hRow, rowEnd: hRow, colStart: 0, colEnd: d - 1, expertId: expId, tokenIndex: tokIdx },
          { tensor: 'rmsnorm', rowStart: tokIdx, rowEnd: tokIdx, colStart: 0, colEnd: d - 1, tokenIndex: tokIdx },
        ],
        activeExpertId: expId,
        isIdleExpert: !ev.isActive,
      };
    }

    case 'routed_act': {
      const hRow = Math.max(0, Math.min(dExp - 1, rawCell.row));
      const ev = evaluateSingleRoutedExpert(sim, expId, tokIdx);
      const hv = ev.h[hRow];
      const av = ev.a[hRow];
      return {
        cell: { tensor: 'routed_act', row: hRow, col: 0, tokenIndex: tokIdx, expertId: expId },
        tensorLabel: `${expTag} ReLU²`,
        coordLabel: `t${tokIdx} "${tokLabel}" · h${hRow}`,
        headBadge: hv > 0 ? `Squared (+)` : `Exact Zero (h ≤ 0)`,
        value: av,
        isMasked: !ev.isActive || hv <= 0,
        valueText: av.toFixed(3),
        equation:
          hv > 0
            ? `a[${expTag}, ${hRow}] = max(0, ${fmtSigned(hv, 3)})² = (${hv.toFixed(3)})² = ${av.toFixed(3)}`
            : `a[${expTag}, ${hRow}] = max(0, ${fmtSigned(hv, 3)})² = 0.000  (exact zero)`,
        subEquation: ev.isActive
          ? `NemotronHMLP ReLU² induces strong activation sparsity with only 2 matrices (W_up, W_down).`
          : `Hypothetical ReLU² preview for idle expert ${expTag} (skipped: rank #${ev.choiceRank + 1} > 6).`,
        upstreamSpans: [
          { tensor: 'routed_pre', rowStart: hRow, rowEnd: hRow, colStart: 0, colEnd: 0, expertId: expId, tokenIndex: tokIdx },
        ],
        activeExpertId: expId,
        isIdleExpert: !ev.isActive,
      };
    }

    case 'routed_w_down': {
      const oRow = Math.max(0, Math.min(d - 1, rawCell.row));
      const hCol = Math.max(0, Math.min(dExp - 1, rawCell.col));
      const ev = evaluateSingleRoutedExpert(sim, expId, tokIdx);
      const wv = sim.W_down[(expId * d + oRow) * dExp + hCol];
      const yOut = ev.y[oRow];
      return {
        cell: { tensor: 'routed_w_down', row: oRow, col: hCol, tokenIndex: tokIdx, expertId: expId },
        tensorLabel: `${expTag} W_down`,
        coordLabel: `out${oRow} × h${hCol}`,
        headBadge: ev.isActive ? `w = ${ev.appliedWeight.toFixed(3)} · 1,856→2,688` : `Idle Preview · 0 weight`,
        value: wv,
        isMasked: !ev.isActive,
        valueText: fmtSigned(wv, 3),
        equation: `W_down[${expTag}, ${oRow}, ${hCol}] = ${fmtSigned(wv, 3)}  (y_${expTag}[${oRow}] = ${fmtSigned(yOut, 3)})`,
        subEquation: ev.isActive
          ? `Contributes w(${ev.appliedWeight.toFixed(3)}) × y_${expTag}[${oRow}](${fmtSigned(yOut, 3)}) = ${fmtSigned(ev.appliedWeight * yOut, 3)} to Routed Sum.`
          : `Idle expert ${expTag}: 0 contribution to Routed Sum (hypothetical y[${oRow}] = ${fmtSigned(yOut, 3)}).`,
        upstreamSpans: [
          { tensor: 'routed_act', rowStart: hCol, rowEnd: hCol, colStart: 0, colEnd: 0, expertId: expId, tokenIndex: tokIdx },
        ],
        activeExpertId: expId,
        isIdleExpert: !ev.isActive,
      };
    }

    case 'routed_sum': {
      const t = Math.max(0, Math.min(T - 1, rawCell.row));
      const ch = Math.max(0, Math.min(d - 1, rawCell.col));
      const val = sim.y_routed[ch * T + t];
      const tl = sim.tokenLabels[t] ?? `t${t}`;
      const { expertId: resolvedExp, activeRank } = resolveRoutedSumExpert(
        sim,
        t,
        rawCell.expertId,
        defaultFocusExpert,
      );
      const resolvedW = sim.topk_weights[t * topK + activeRank];
      const resolvedY = sim.active_y[(t * topK + activeRank) * d + ch];
      const terms: string[] = [];
      for (let j = 0; j < 3; j++) {
        const eid = sim.topk_indices[t * topK + j];
        const w = sim.topk_weights[t * topK + j];
        const yv = sim.active_y[(t * topK + j) * d + ch];
        terms.push(`${w.toFixed(2)}·${formatExpertId(eid)}(${fmtSigned(yv, 2)})`);
      }
      return {
        cell: { tensor: 'routed_sum', row: t, col: ch, tokenIndex: t, expertId: resolvedExp },
        tensorLabel: 'Routed Sum (Top-6)',
        coordLabel: `t${t} "${tl}" · ch ${ch}`,
        headBadge: 'Σ_{j=1..6} wⱼ·yⱼ (×2.5 scaled)',
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 3),
        equation: `y_routed[${ch}, t${t}] = ${terms.join(' + ')} + … (6 active) = ${fmtSigned(val, 3)}`,
        subEquation: `Parallel sum of 6 active routed experts (122 idle = 0). Focused: ${formatExpertId(resolvedExp)} (Top-6 #${activeRank + 1}, ${resolvedW.toFixed(3)}×${fmtSigned(resolvedY, 3)}).`,
        upstreamSpans: [
          { tensor: 'topk_weights', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: topK - 1, tokenIndex: t },
          { tensor: 'routed_w_down', rowStart: ch, rowEnd: ch, colStart: 0, colEnd: dExp - 1, expertId: resolvedExp, tokenIndex: t },
        ],
        activeExpertId: resolvedExp,
        isIdleExpert: false,
      };
    }

    case 'shared_w_up': {
      const hRow = Math.max(0, Math.min(dShared - 1, rawCell.row));
      const iCol = Math.max(0, Math.min(d - 1, rawCell.col));
      const wv = sim.W_up_s[hRow * d + iCol];
      return {
        cell: { tensor: 'shared_w_up', row: hRow, col: iCol, tokenIndex: tokIdx },
        tensorLabel: 'Shared W_up (2×)',
        coordLabel: `s${hRow} × in${iCol}`,
        headBadge: 'Always-On · 3,712 = 2×1,856',
        value: wv,
        isMasked: false,
        valueText: fmtSigned(wv, 3),
        equation: `W_up_s[${hRow}, ${iCol}] = ${fmtSigned(wv, 3)} × U[${iCol}, t${tokIdx}](${fmtSigned(sim.U[iCol * T + tokIdx], 3)})`,
        subEquation: `Always-on double-width shared expert (d_shared=8 = 2×d_exp; prod 3,712 = 2×1,856, fused in checkpoint).`,
        upstreamSpans: [
          { tensor: 'rmsnorm', rowStart: tokIdx, rowEnd: tokIdx, colStart: iCol, colEnd: iCol, tokenIndex: tokIdx },
        ],
        activeExpertId: null,
        isIdleExpert: false,
      };
    }

    case 'shared_act': {
      const hRow = Math.max(0, Math.min(dShared - 1, rawCell.row));
      const hv = sim.shared_h[hRow * T + tokIdx];
      const av = sim.shared_a[hRow * T + tokIdx];
      return {
        cell: { tensor: 'shared_act', row: hRow, col: 0, tokenIndex: tokIdx },
        tensorLabel: 'Shared ReLU² (2×)',
        coordLabel: `t${tokIdx} "${tokLabel}" · s${hRow}`,
        headBadge: hv > 0 ? 'Always-On · Positive²' : 'Always-On · Exact 0',
        value: av,
        isMasked: hv <= 0,
        valueText: av.toFixed(3),
        equation:
          hv > 0
            ? `a_shared[${hRow}, t${tokIdx}] = max(0, ${fmtSigned(hv, 3)})² = ${av.toFixed(3)}`
            : `a_shared[${hRow}, t${tokIdx}] = max(0, ${fmtSigned(hv, 3)})² = 0.000`,
        subEquation: `Double-width non-gated ReLU² (no SwiGLU gate_proj, no router gating, NO 2.5× scaling).`,
        upstreamSpans: [
          { tensor: 'shared_w_up', rowStart: hRow, rowEnd: hRow, colStart: 0, colEnd: d - 1, tokenIndex: tokIdx },
        ],
        activeExpertId: null,
        isIdleExpert: false,
      };
    }

    case 'shared_w_down': {
      const oRow = Math.max(0, Math.min(d - 1, rawCell.row));
      const hCol = Math.max(0, Math.min(dShared - 1, rawCell.col));
      const wv = sim.W_down_s[oRow * dShared + hCol];
      const ys = sim.y_shared[oRow * T + tokIdx];
      return {
        cell: { tensor: 'shared_w_down', row: oRow, col: hCol, tokenIndex: tokIdx },
        tensorLabel: 'Shared W_down (2×)',
        coordLabel: `out${oRow} × s${hCol}`,
        headBadge: 'Always-On · Unscaled (1.0×)',
        value: wv,
        isMasked: false,
        valueText: fmtSigned(wv, 3),
        equation: `W_down_s[${oRow}, ${hCol}] = ${fmtSigned(wv, 3)}  (y_shared[${oRow}, t${tokIdx}] = ${fmtSigned(ys, 3)})`,
        subEquation: `Projects 3,712 shared width → 2,688 residual (mini 8→8). Added unscaled (no 2.5× multiplier).`,
        upstreamSpans: [
          { tensor: 'shared_act', rowStart: hCol, rowEnd: hCol, colStart: 0, colEnd: 0, tokenIndex: tokIdx },
        ],
        activeExpertId: null,
        isIdleExpert: false,
      };
    }

    case 'shared_out': {
      const t = Math.max(0, Math.min(T - 1, rawCell.row));
      const ch = Math.max(0, Math.min(d - 1, rawCell.col));
      const val = sim.y_shared[ch * T + t];
      const tl = sim.tokenLabels[t] ?? `t${t}`;
      return {
        cell: { tensor: 'shared_out', row: t, col: ch, tokenIndex: t },
        tensorLabel: 'Shared Out (Always-On)',
        coordLabel: `t${t} "${tl}" · ch ${ch}`,
        headBadge: 'Unscaled (1.0×) · 3,712 width',
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 3),
        equation: `y_shared[${ch}, t${t}] = Σₛ W_down_s[${ch}, s]·a_shared[s, t${t}] = ${fmtSigned(val, 3)}`,
        subEquation: `Parallel always-on branch merges directly with Routed Sum (NOT multiplied by routed_scaling_factor = 2.5).`,
        upstreamSpans: [
          { tensor: 'shared_w_down', rowStart: ch, rowEnd: ch, colStart: 0, colEnd: dShared - 1, tokenIndex: t },
        ],
        activeExpertId: null,
        isIdleExpert: false,
      };
    }

    case 'moe_out': {
      const t = Math.max(0, Math.min(T - 1, rawCell.row));
      const ch = Math.max(0, Math.min(d - 1, rawCell.col));
      const yr = sim.y_routed[ch * T + t];
      const ys = sim.y_shared[ch * T + t];
      const ym = sim.y_moe[ch * T + t];
      const tl = sim.tokenLabels[t] ?? `t${t}`;
      return {
        cell: { tensor: 'moe_out', row: t, col: ch, tokenIndex: t },
        tensorLabel: 'MoE Merge Out',
        coordLabel: `t${t} "${tl}" · ch ${ch}`,
        headBadge: 'Routed (×2.5) + Shared (1×)',
        value: ym,
        isMasked: false,
        valueText: fmtSigned(ym, 3),
        equation: `y_moe[${ch}, t${t}] = y_routed(${fmtSigned(yr, 3)}) + y_shared(${fmtSigned(ys, 3)}) = ${fmtSigned(ym, 3)}`,
        subEquation: `Parallel merge of Top-6 sparse routed output and always-on 2× shared expert.`,
        upstreamSpans: [
          { tensor: 'routed_sum', rowStart: t, rowEnd: t, colStart: ch, colEnd: ch, tokenIndex: t },
          { tensor: 'shared_out', rowStart: t, rowEnd: t, colStart: ch, colEnd: ch, tokenIndex: t },
        ],
        activeExpertId: null,
        isIdleExpert: false,
      };
    }

    case 'residual':
    default: {
      const t = Math.max(0, Math.min(T - 1, rawCell.row));
      const ch = Math.max(0, Math.min(d - 1, rawCell.col));
      const xv = sim.X[ch * T + t];
      const ym = sim.y_moe[ch * T + t];
      const yv = sim.Y[ch * T + t];
      const tl = sim.tokenLabels[t] ?? `t${t}`;
      return {
        cell: { tensor: 'residual', row: t, col: ch, tokenIndex: t },
        tensorLabel: 'Residual Out Y',
        coordLabel: `t${t} "${tl}" · ch ${ch}`,
        headBadge: 'X + y_routed + y_shared',
        value: yv,
        isMasked: false,
        valueText: fmtSigned(yv, 3),
        equation: `Y[${ch}, t${t}] = X(${fmtSigned(xv, 3)}) + y_moe(${fmtSigned(ym, 3)}) = ${fmtSigned(yv, 3)}`,
        subEquation: `Returns to the 2,688-channel emerald residual spine. 0 B sequence cache allocated.`,
        upstreamSpans: [
          { tensor: 'moe_out', rowStart: t, rowEnd: t, colStart: ch, colEnd: ch, tokenIndex: t },
          { tensor: 'input', rowStart: t, rowEnd: t, colStart: ch, colEnd: ch, tokenIndex: t },
        ],
        activeExpertId: null,
        isIdleExpert: false,
      };
    }
  }
}

/** 6-Step MoE Math Stepper definitions aligned with the docked Formula Inspector. */
export const MOE_STEPPER_STEPS: readonly StepperStepDef[] = [
  {
    index: 0,
    shortLabel: '1. RMSNorm',
    title: 'Pre-Layer RMSNorm (ε = 1e−5)',
    summary: 'Normalize token vector X[:, t] by √(mean(X²) + 1e−5) × γ_in before parallel fork.',
  },
  {
    index: 1,
    shortLabel: '2. Router + Bias',
    title: 'FP32 Router + e_score_correction_bias',
    summary: 'Compute sigmoid(W_router·U) + e_score_correction_bias across all 128 experts.',
  },
  {
    index: 2,
    shortLabel: '3. Top-6 (×2.5)',
    title: 'Top-6 Selection & Unbiased ×2.5 Weights',
    summary: 'Pick Top-6 via biased choice score, gather unbiased sigmoid scores, normalize, × 2.5.',
  },
  {
    index: 3,
    shortLabel: '4. Routed ReLU²',
    title: 'Parallel Non-Gated Routed Experts (ReLU²)',
    summary: 'Run 6 active experts (W_up → max(0,h)² → W_down); 122 idle experts stay at 0.',
  },
  {
    index: 4,
    shortLabel: '5. Shared (2×)',
    title: 'Always-On Double-Width Shared Expert',
    summary: 'Run parallel 2× wide shared expert (3,712 width, ReLU², unscaled by 2.5).',
  },
  {
    index: 5,
    shortLabel: '6. Merge + Res',
    title: 'Parallel Merge & Residual Add',
    summary: 'Sum y_routed + y_shared + X back into the 2,688-channel emerald residual spine.',
  },
];

/** Map stepper stage + subProgress [0..1) into a deterministic inspected MoE cell. */
export function moeStepperCellForState(
  sim: MoeSimResult,
  stepIndex: number,
  subProgress: number,
  focusToken: number,
  focusExpert: number,
): MoeCellRef {
  const t = Math.max(0, Math.min(sim.tokens - 1, focusToken));
  const p = Math.max(0, Math.min(0.9999, subProgress));
  const topSlot = Math.min(sim.topK - 1, Math.floor(p * sim.topK));
  const slotExpert = sim.topk_indices[t * sim.topK + topSlot];

  switch (Math.max(0, Math.min(5, stepIndex))) {
    case 0: {
      const ch = Math.min(sim.d - 1, Math.floor(p * sim.d));
      return { tensor: 'rmsnorm', row: t, col: ch, tokenIndex: t };
    }
    case 1: {
      const bankRow = Math.floor(slotExpert / MOE_SIM_CONFIG.bankCols);
      const bankCol = slotExpert % MOE_SIM_CONFIG.bankCols;
      return {
        tensor: 'router_bank',
        row: bankRow,
        col: bankCol,
        tokenIndex: t,
        expertId: slotExpert,
      };
    }
    case 2: {
      return {
        tensor: 'topk_weights',
        row: 0,
        col: topSlot,
        tokenIndex: t,
        expertId: slotExpert,
        rankSlot: topSlot,
      };
    }
    case 3: {
      const hRow = Math.min(sim.dExp - 1, Math.floor(p * sim.dExp));
      return {
        tensor: 'routed_act',
        row: hRow,
        col: 0,
        tokenIndex: t,
        expertId: focusExpert,
      };
    }
    case 4: {
      const sRow = Math.min(sim.dShared - 1, Math.floor(p * sim.dShared));
      return {
        tensor: 'shared_act',
        row: sRow,
        col: 0,
        tokenIndex: t,
      };
    }
    case 5:
    default: {
      const ch = Math.min(sim.d - 1, Math.floor(p * sim.d));
      return { tensor: 'residual', row: t, col: ch, tokenIndex: t };
    }
  }
}
