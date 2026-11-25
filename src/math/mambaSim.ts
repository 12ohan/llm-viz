/**
 * Stage A+1 · Increment 3: Deterministic Miniature Mamba-2 Simulator
 *
 * Implements verified NVIDIA NemotronHMamba2Mixer + MambaRMSNormGated equations
 * (rms_norm_eps = 1e-5, norm_before_gate = False, depthwise causal Conv1D k=4,
 * discretized per-head SSM recurrence, skip connection D, 8-group gated RMSNorm)
 * in pure TypeScript with Float32 tensor storage and rounded outputs.
 *
 * Miniature dimensions preserve Nemotron's exact 8-group structure and ratios:
 *   d = 8 (prod 2,688) · n_heads = 16 (prod 64) · n_groups = 8 (prod 8)
 *   d_head = 4 (prod 64) · d_inner = 64 (prod 4,096) · d_state = 4 (prod 128)
 *   conv_kernel = 4 (prod 4) · conv_dim = 128 (prod 6,144) · in_proj = 208 (prod 10,304)
 *
 * Educational miniature simulator — NOT full 31.6B checkpoint inference.
 * Precision toggles (BF16 / FP8 Selective) are displayed strictly as production metadata.
 */

import {
  EDUCATIONAL_TOKENS,
  formatEpsLabel,
  resolveSimTokens,
  validateLayerModulation,
  validateResidualGate,
  type LayerModulationTerms,
} from './gqaSim';
import { fmtSigned, type LayerFormulaReadout, type StepperStepDef } from './formulaFormat';

export { EDUCATIONAL_TOKENS, type LayerModulationTerms };

export const MAMBA_SIM_CONFIG = {
  d: 8,
  nHeads: 16,
  nGroups: 8,
  headsPerGroup: 2,
  dHead: 4,
  dInner: 64,
  dState: 4,
  convKernel: 4,
  convDim: 128, // dInner (64) + 2 * nGroups (8) * dState (4)
  inProjDim: 208, // z (64) + xBC_pre (128) + dt_raw (16)
  groupSize: 8, // dInner / nGroups
  rmsEps: 1e-5,
  normBeforeGate: false,
} as const;

export type MambaTensorKey =
  | 'input'
  | 'rmsnorm'
  | 'w_in'
  | 'z'
  | 'xbc_pre'
  | 'conv_buf'
  | 'x'
  | 'b'
  | 'c'
  | 'dt'
  | 'h_prev'
  | 'h_curr'
  | 'y'
  | 'y_norm'
  | 'w_out'
  | 'residual';

export interface MambaCellRef {
  readonly tensor: MambaTensorKey;
  /** Logical row index in the sheet grid (e.g. token t, state p, feature i) */
  readonly row: number;
  /** Logical col index in the sheet grid (e.g. channel c, state n, tap slot s) */
  readonly col: number;
  /** Owning miniature head 0..15 (when applicable) */
  readonly headIndex?: number;
  /** Owning B/C group 0..7 (when applicable) */
  readonly groupIndex?: number;
  /** Conv1D rolling buffer tap slot 0..3 (slot 3 = newest token T-1) */
  readonly tapIndex?: number;
  /** Token index for state inspection (defaults to T - 1) */
  readonly tokenIndex?: number;
}

export interface MambaSheetSpan {
  readonly tensor: MambaTensorKey;
  readonly rowStart: number;
  readonly rowEnd: number;
  readonly colStart: number;
  readonly colEnd: number;
  readonly headIndex?: number;
  readonly groupIndex?: number;
  readonly tapStart?: number;
  readonly tapEnd?: number;
  readonly tokenIndex?: number;
  readonly role: 'target' | 'operand_a' | 'operand_b';
}

export interface MambaCellFormula extends LayerFormulaReadout {
  readonly cell: MambaCellRef;
  readonly targetSpan: MambaSheetSpan;
  readonly upstreamSpans: readonly MambaSheetSpan[];
}

export interface MambaSimFixture {
  readonly X?: Float32Array;
  readonly gamma_in?: Float32Array;
  readonly W_in?: Float32Array;
  readonly w_conv?: Float32Array;
  readonly b_conv?: Float32Array;
  readonly A_log?: Float32Array;
  readonly dt_bias?: Float32Array;
  readonly D?: Float32Array;
  readonly gamma_gate?: Float32Array;
  readonly W_out?: Float32Array;
  readonly eps?: number;
  /** Actual input token IDs for this execution (length T). */
  readonly tokenIds?: readonly number[] | Int32Array;
  /** Matching display labels for each token position (length T). */
  readonly tokenLabels?: readonly string[];
  /**
   * Optional externally supplied rolling Conv1D state [convDim, convKernel]
   * (slots 0..3 = oldest..newest prior xBC_pre taps before local token 0).
   */
  readonly initialConvState?: Float32Array;
  /**
   * Optional externally supplied persistent SSM state H_{P-1} [nHeads, dHead, dState]
   * immediately prior to local token 0.
   */
  readonly initialSsmState?: Float32Array;
  /** Absolute starting token position offset P >= 0 for incremental steps (defaults to 0). */
  readonly startTokenOffset?: number;
  /** Optional pre-RMSNorm AdaLN modulation (`X_mod = X * (1 + scale) + shift`, then RMSNorm). */
  readonly preNormMod?: LayerModulationTerms;
  /** Optional channel-wise residual gate (`Y = X + gate * Mamba_out`, preserving unmodulated `X`). */
  readonly residualGate?: Float32Array;
}

export interface MambaSimResult {
  readonly tokens: number;
  /** Absolute starting token position offset P >= 0 (0 in standalone / full-sequence runs). */
  readonly startTokenOffset: number;
  /** True when seeded from an externally supplied Conv1D or SSM prior state. */
  readonly hasInitialState: boolean;
  /** Defensive copy of the prior rolling Conv1D state [convDim, convKernel], if supplied. */
  readonly initialConvState?: Float32Array;
  /** Defensive copy of the prior FP32 SSM state H_{P-1} [nHeads, dHead, dState], if supplied. */
  readonly initialSsmState?: Float32Array;
  readonly tokenIds: Int32Array;
  readonly d: number;
  readonly nHeads: number;
  readonly nGroups: number;
  readonly headsPerGroup: number;
  readonly dHead: number;
  readonly dInner: number;
  readonly dState: number;
  readonly convKernel: number;
  readonly convDim: number;
  readonly inProjDim: number;
  readonly groupSize: number;
  readonly eps: number;
  readonly tokenLabels: readonly string[];

  /** Input hidden state X: [d, T] -> X[i * T + t] (always untouched original residual) */
  readonly X: Float32Array;
  /** Optional pre-RMSNorm AdaLN modulated input X_mod = X * (1 + scale) + shift: [d, T] */
  readonly X_mod?: Float32Array;
  /** Pre-gamma RMSNorm: [d, T] */
  readonly X_norm: Float32Array;
  /** Per-token RMS denominator: [T] */
  readonly rms_in: Float32Array;
  /** Input RMSNorm scale gamma_in: [d] */
  readonly gamma_in: Float32Array;
  /** Post-gamma normalized hidden state U = X_norm * gamma_in: [d, T] */
  readonly U: Float32Array;

  /** Fused input projection W_in: [inProjDim, d] -> W_in[row * d + j] */
  readonly W_in: Float32Array;
  /** Bypass gate branch z: [dInner, T] -> z[c * T + t] */
  readonly z: Float32Array;
  /** Pre-convolution xBC stream: [convDim, T] -> xBC_pre[c * T + t] */
  readonly xBC_pre: Float32Array;
  /** Raw step logits dt_raw: [nHeads, T] -> dt_raw[h * T + t] */
  readonly dt_raw: Float32Array;

  /** Depthwise causal Conv1D weights: [convDim, convKernel] (slot 3 = lag 0 / current token) */
  readonly w_conv: Float32Array;
  /** Depthwise causal Conv1D bias: [convDim] */
  readonly b_conv: Float32Array;
  /** Fixed 4-tap rolling Conv1D inference buffer at final token T-1: [convDim, convKernel] */
  readonly conv_buf: Float32Array;
  /** Pre-SiLU causal Conv1D output: [convDim, T] */
  readonly conv_out: Float32Array;
  /** Post-SiLU xBC output: [convDim, T] */
  readonly xBC: Float32Array;

  /** Split SSM input x: [nHeads, dHead, T] -> x[(h * dHead + p) * T + t] */
  readonly x: Float32Array;
  /** Split input projection B (8 groups): [nGroups, dState, T] -> B[(g * dState + n) * T + t] */
  readonly B: Float32Array;
  /** Split output projection C (8 groups): [nGroups, dState, T] -> C[(g * dState + n) * T + t] */
  readonly C: Float32Array;

  /** Log-decay parameter A_log: [nHeads] */
  readonly A_log: Float32Array;
  /** Continuous decay A = -exp(A_log) < 0: [nHeads] */
  readonly A: Float32Array;
  /** Learned step bias dt_bias: [nHeads] */
  readonly dt_bias: Float32Array;
  /** Discretization step dt = softplus(dt_raw + dt_bias) > 0: [nHeads, T] */
  readonly dt: Float32Array;
  /** Discrete decay factor dA = exp(dt * A) in (0, 1): [nHeads, T] */
  readonly dA: Float32Array;
  /** Per-head skip connection D: [nHeads] */
  readonly D: Float32Array;

  /** Fixed-size SSM state immediately prior to final token T-1 (0 when T=1): [nHeads, dHead, dState] */
  readonly H_prev: Float32Array;
  /** Fixed-size persistent SSM state at final token T-1: [nHeads, dHead, dState] */
  readonly H: Float32Array;
  /**
   * Browser-only educational playback history [T, nHeads, dHead, dState] for inspecting
   * recurrence transitions — NOT stored in production inference (only H is persistent).
   */
  readonly H_history: Float32Array;

  /** Pre-skip SSM readout Σ_n H[h,p,n]*C[g,n,t]: [nHeads, dHead, T] */
  readonly y_ssm: Float32Array;
  /** SSM output with skip connection y = y_ssm + D * x: [nHeads, dHead, T] */
  readonly y: Float32Array;

  /** Gated pre-norm activation gated = y * SiLU(z) (norm_before_gate = False): [dInner, T] */
  readonly gated: Float32Array;
  /** 8-group RMS denominators for MambaRMSNormGated: [nGroups, T] -> rms_gate[g * T + t] */
  readonly rms_gate: Float32Array;
  /** Pre-gamma 8-group normalized output: [dInner, T] */
  readonly y_norm_pre: Float32Array;
  /** Learned gated RMSNorm scale gamma_gate: [dInner] */
  readonly gamma_gate: Float32Array;
  /** Final gated-then-normalized output y_norm = (y * SiLU(z)) / rms_g * gamma_gate: [dInner, T] */
  readonly y_norm: Float32Array;

  /** Output projection weights W_out: [d, dInner] -> W_out[i * dInner + c] */
  readonly W_out: Float32Array;
  /** Layer delta Mamba_out = W_out @ y_norm: [d, T] */
  readonly Mamba_out: Float32Array;
  /** Defensive copy of channel-wise residual gate [d], if supplied. */
  readonly residualGate?: Float32Array;
  /** True when a channel-wise residual gate modulated `Mamba_out` before adding to `X`. */
  readonly residualGateApplied: boolean;
  /** Residual output Y = X + (gate ? gate * Mamba_out : Mamba_out): [d, T] */
  readonly Y: Float32Array;
}

/** Numerically stable softplus: log(1 + exp(v)) without overflow/underflow at ±100. */
export function stableSoftplus(v: number): number {
  return Math.max(v, 0) + Math.log1p(Math.exp(-Math.abs(v)));
}

/** Numerically stable SiLU: x * sigmoid(x). */
export function silu(v: number): number {
  if (v >= 0) {
    const ez = Math.exp(-v);
    return v / (1 + ez);
  }
  const ez = Math.exp(v);
  return (v * ez) / (1 + ez);
}

/** Map miniature head h in 0..15 to its B/C group g in 0..7 (2 miniature heads/group). */
export function mambaGroupForHead(headIndex: number): number {
  return Math.max(0, Math.min(MAMBA_SIM_CONFIG.nGroups - 1, Math.floor(headIndex / MAMBA_SIM_CONFIG.headsPerGroup)));
}

export function formatMambaHeadId(headIndex: number): string {
  return `H${String(headIndex).padStart(2, '0')}`;
}

/**
 * Pure MambaRMSNormGated implementation matching NemotronH (`norm_before_gate = False`):
 * 1. Gate FIRST: gated = y * SiLU(z)
 * 2. 8-Group RMSNorm over each group of `dInner / nGroups` channels with `eps = 1e-5`
 * 3. Multiply by learned non-unit scale `gamma_gate`
 */
export function applyGroupedRmsNormGated(
  yFlat: Float32Array,
  z: Float32Array,
  gammaGate: Float32Array,
  tokens: number,
  nGroups: number = MAMBA_SIM_CONFIG.nGroups,
  eps: number = MAMBA_SIM_CONFIG.rmsEps,
): {
  gated: Float32Array;
  rms_gate: Float32Array;
  y_norm_pre: Float32Array;
  y_norm: Float32Array;
} {
  const dInner = gammaGate.length;
  const groupSize = dInner / nGroups;
  const gated = new Float32Array(dInner * tokens);
  const rms_gate = new Float32Array(nGroups * tokens);
  const y_norm_pre = new Float32Array(dInner * tokens);
  const y_norm = new Float32Array(dInner * tokens);

  for (let c = 0; c < dInner; c++) {
    for (let t = 0; t < tokens; t++) {
      const idx = c * tokens + t;
      gated[idx] = Math.fround(yFlat[idx] * Math.fround(silu(z[idx])));
    }
  }

  for (let g = 0; g < nGroups; g++) {
    const cStart = g * groupSize;
    for (let t = 0; t < tokens; t++) {
      let sumSq = 0;
      for (let k = 0; k < groupSize; k++) {
        const val = gated[(cStart + k) * tokens + t];
        sumSq += val * val;
      }
      const r = Math.fround(Math.sqrt(sumSq / groupSize + eps));
      rms_gate[g * tokens + t] = r;
      for (let k = 0; k < groupSize; k++) {
        const c = cStart + k;
        const idx = c * tokens + t;
        const pre = Math.fround(gated[idx] / r);
        y_norm_pre[idx] = pre;
        y_norm[idx] = Math.fround(pre * gammaGate[c]);
      }
    }
  }

  return { gated, rms_gate, y_norm_pre, y_norm };
}

// Deterministic Mulberry32 PRNG for reproducible miniature weights & synthetic embeddings
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeNormalish(rng: () => number, scale: number): number {
  const u = rng() + rng() + rng() - 1.5;
  return Math.fround(u * scale);
}

interface MambaWeights {
  readonly fullX: Float32Array; // [d, 16]
  readonly gamma_in: Float32Array;
  readonly W_in: Float32Array;
  readonly w_conv: Float32Array;
  readonly b_conv: Float32Array;
  readonly A_log: Float32Array;
  readonly dt_bias: Float32Array;
  readonly D: Float32Array;
  readonly gamma_gate: Float32Array;
  readonly W_out: Float32Array;
}

function buildDefaultMambaWeights(): MambaWeights {
  const { d, nHeads, dInner, convKernel, convDim, inProjDim } = MAMBA_SIM_CONFIG;
  const maxT = 16;
  const rng = mulberry32(0x3a7b2e19);

  // Synthetic token hidden vectors X[i, t]
  const fullX = new Float32Array(d * maxT);
  for (let t = 0; t < maxT; t++) {
    for (let i = 0; i < d; i++) {
      const wave = 0.45 * Math.sin((t + 1) * 0.68 + i * 0.92);
      const jitter = makeNormalish(rng, 0.42);
      fullX[i * maxT + t] = Math.fround(wave + jitter);
    }
  }

  // Non-unit input RMSNorm gamma_in near 1.0
  const gamma_in = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    gamma_in[i] = Math.fround(0.86 + 0.28 * rng());
  }

  // W_in [208, 8]
  const W_in = new Float32Array(inProjDim * d);
  for (let r = 0; r < inProjDim; r++) {
    // Slightly distinct scale per branch so z, xBC_pre, and dt_raw stay well-conditioned
    const scale = r < dInner ? 0.52 : r < dInner + convDim ? 0.48 : 0.35;
    for (let j = 0; j < d; j++) {
      W_in[r * d + j] = makeNormalish(rng, scale);
    }
  }

  // Depthwise causal Conv1D kernels [128, 4] + biases [128]
  const w_conv = new Float32Array(convDim * convKernel);
  const b_conv = new Float32Array(convDim);
  for (let c = 0; c < convDim; c++) {
    // Recency-tilted causal taps (slot 0 = t-3 ... slot 3 = t)
    const tapScales = [0.22, 0.32, 0.46, 0.68];
    for (let s = 0; s < convKernel; s++) {
      w_conv[c * convKernel + s] = Math.fround(
        (s === 3 ? 0.38 : 0.0) + makeNormalish(rng, tapScales[s]),
      );
    }
    b_conv[c] = makeNormalish(rng, 0.12);
  }

  // Distinct per-head decay timescales A = -exp(A_log) across the 16 heads
  // Ranges from fast-forgetting heads (A ≈ -1.6) to long-memory heads (A ≈ -0.18)
  const A_log = new Float32Array(nHeads);
  const dt_bias = new Float32Array(nHeads);
  const D = new Float32Array(nHeads);
  for (let h = 0; h < nHeads; h++) {
    const frac = h / (nHeads - 1);
    A_log[h] = Math.fround(-1.55 + frac * 1.95 + makeNormalish(rng, 0.14));
    dt_bias[h] = Math.fround(-0.85 + 0.9 * rng());
    D[h] = Math.fround(0.42 + 0.52 * rng());
  }

  // Non-unit gated RMSNorm scale gamma_gate [64]
  const gamma_gate = new Float32Array(dInner);
  for (let c = 0; c < dInner; c++) {
    gamma_gate[c] = Math.fround(0.84 + 0.32 * rng());
  }

  // Output projection W_out [8, 64]
  const W_out = new Float32Array(d * dInner);
  for (let i = 0; i < d; i++) {
    for (let c = 0; c < dInner; c++) {
      W_out[i * dInner + c] = makeNormalish(rng, 0.20);
    }
  }

  return {
    fullX,
    gamma_in,
    W_in,
    w_conv,
    b_conv,
    A_log,
    dt_bias,
    D,
    gamma_gate,
    W_out,
  };
}

const DEFAULT_WEIGHTS = buildDefaultMambaWeights();
const DEFAULT_SIM_CACHE = new Map<number, MambaSimResult>();

/**
 * Run deterministic miniature Mamba-2 forward pass for T in [1..16].
 * Default calls (no fixture) are memoized per T for instant hover/scrubber updates.
 */
export function runMambaSim(tokens: number, fixture?: MambaSimFixture): MambaSimResult {
  const rawT = Math.round(tokens);
  if (fixture && (!Number.isFinite(tokens) || rawT < 1 || rawT > 16)) {
    throw new RangeError(`Mamba query length S (${tokens}) must be in [1..16]`);
  }
  const T = Math.max(1, Math.min(16, rawT));
  if (!fixture) {
    const cached = DEFAULT_SIM_CACHE.get(T);
    if (cached) return cached;
  }

  const {
    d,
    nHeads,
    nGroups,
    headsPerGroup,
    dHead,
    dInner,
    dState,
    convKernel,
    convDim,
    inProjDim,
    groupSize,
  } = MAMBA_SIM_CONFIG;
  const eps = fixture?.eps ?? MAMBA_SIM_CONFIG.rmsEps;

  const preNormMod = fixture?.preNormMod;
  if (preNormMod) {
    validateLayerModulation(preNormMod, d, 'Mamba preNormMod');
  }
  let residualGate: Float32Array | undefined;
  if (fixture?.residualGate) {
    validateResidualGate(fixture.residualGate, d, 'Mamba residualGate');
    residualGate = new Float32Array(fixture.residualGate);
  }
  const residualGateApplied = residualGate !== undefined;

  const gamma_in = fixture?.gamma_in ?? DEFAULT_WEIGHTS.gamma_in;
  const W_in = fixture?.W_in ?? DEFAULT_WEIGHTS.W_in;
  const w_conv = fixture?.w_conv ?? DEFAULT_WEIGHTS.w_conv;
  const b_conv = fixture?.b_conv ?? DEFAULT_WEIGHTS.b_conv;
  const A_log = fixture?.A_log ?? DEFAULT_WEIGHTS.A_log;
  const dt_bias = fixture?.dt_bias ?? DEFAULT_WEIGHTS.dt_bias;
  const D = fixture?.D ?? DEFAULT_WEIGHTS.D;
  const gamma_gate = fixture?.gamma_gate ?? DEFAULT_WEIGHTS.gamma_gate;
  const W_out = fixture?.W_out ?? DEFAULT_WEIGHTS.W_out;

  const { tokenIds, tokenLabels } = resolveSimTokens(T, fixture?.tokenIds, fixture?.tokenLabels);
  const startTokenOffset = Math.max(0, Math.trunc(fixture?.startTokenOffset ?? 0));

  // Optional externally seeded boundary states (validated + defensively cloned so caller mutations cannot alias)
  const expectedConvLen = convDim * convKernel;
  const stateStride = nHeads * dHead * dState;
  let initialConvState: Float32Array | undefined;
  let initialSsmState: Float32Array | undefined;

  if (fixture?.initialConvState) {
    if (fixture.initialConvState.length !== expectedConvLen) {
      throw new RangeError(
        `Mamba initialConvState length ${fixture.initialConvState.length} !== expected ${expectedConvLen}`,
      );
    }
    const clonedConv = new Float32Array(fixture.initialConvState);
    for (let i = 0; i < expectedConvLen; i++) {
      if (!Number.isFinite(clonedConv[i])) {
        throw new RangeError(
          `Non-finite value (${clonedConv[i]}) in Mamba initialConvState[${i}]`,
        );
      }
    }
    initialConvState = clonedConv;
  }
  if (fixture?.initialSsmState) {
    if (fixture.initialSsmState.length !== stateStride) {
      throw new RangeError(
        `Mamba initialSsmState length ${fixture.initialSsmState.length} !== expected ${stateStride}`,
      );
    }
    const clonedSsm = new Float32Array(fixture.initialSsmState);
    for (let i = 0; i < stateStride; i++) {
      if (!Number.isFinite(clonedSsm[i])) {
        throw new RangeError(
          `Non-finite value (${clonedSsm[i]}) in Mamba initialSsmState[${i}]`,
        );
      }
    }
    initialSsmState = clonedSsm;
  }
  const hasInitialState = initialConvState !== undefined || initialSsmState !== undefined;

  // 1. Input hidden state X [d, T]
  const X = new Float32Array(d * T);
  if (fixture?.X) {
    X.set(fixture.X.subarray(0, d * T));
  } else {
    for (let i = 0; i < d; i++) {
      for (let t = 0; t < T; t++) {
        X[i * T + t] = DEFAULT_WEIGHTS.fullX[i * 16 + t];
      }
    }
  }

  // 2. Optional Pre-Norm AdaLN Modulation (modulate FIRST, then RMSNorm eps = 1e-5) -> X_norm [d, T] & U = X_norm * gamma_in [d, T]
  let X_mod: Float32Array | undefined;
  if (preNormMod) {
    X_mod = new Float32Array(d * T);
    for (let i = 0; i < d; i++) {
      const scaleFactor = Math.fround(1 + preNormMod.scale[i]);
      const shift = preNormMod.shift[i];
      for (let t = 0; t < T; t++) {
        X_mod[i * T + t] = Math.fround(Math.fround(X[i * T + t] * scaleFactor) + shift);
      }
    }
  }
  const normInput = X_mod ?? X;
  const rms_in = new Float32Array(T);
  const X_norm = new Float32Array(d * T);
  const U = new Float32Array(d * T);
  for (let t = 0; t < T; t++) {
    let sumSq = 0;
    for (let i = 0; i < d; i++) {
      const val = normInput[i * T + t];
      sumSq += val * val;
    }
    const r = Math.fround(Math.sqrt(sumSq / d + eps));
    rms_in[t] = r;
    for (let i = 0; i < d; i++) {
      const xn = Math.fround(normInput[i * T + t] / r);
      X_norm[i * T + t] = xn;
      U[i * T + t] = Math.fround(xn * gamma_in[i]);
    }
  }

  // 3. Fused Input Projection: W_in [208, 8] @ U [8, T] -> z [64, T], xBC_pre [128, T], dt_raw [16, T]
  const z = new Float32Array(dInner * T);
  const xBC_pre = new Float32Array(convDim * T);
  const dt_raw = new Float32Array(nHeads * T);

  for (let row = 0; row < inProjDim; row++) {
    const wOff = row * d;
    for (let t = 0; t < T; t++) {
      let acc = 0;
      for (let j = 0; j < d; j++) {
        acc += W_in[wOff + j] * U[j * T + t];
      }
      const val = Math.fround(acc);
      if (row < dInner) {
        z[row * T + t] = val;
      } else if (row < dInner + convDim) {
        xBC_pre[(row - dInner) * T + t] = val;
      } else {
        dt_raw[(row - dInner - convDim) * T + t] = val;
      }
    }
  }

  // 4. Depthwise Causal Conv1D (k=4, left zero-padded or seeded from initialConvState) + SiLU + fixed 4-tap rolling conv_buf
  // Preserves EXACT newest-to-oldest lag accumulation order (lag = 0..convKernel-1) for bit-exact full-vs-incremental equivalence.
  const conv_out = new Float32Array(convDim * T);
  const xBC = new Float32Array(convDim * T);
  const conv_buf = new Float32Array(convDim * convKernel);

  for (let c = 0; c < convDim; c++) {
    const wcOff = c * convKernel;
    const bias = b_conv[c];
    for (let t = 0; t < T; t++) {
      let acc = bias;
      for (let lag = 0; lag < convKernel; lag++) {
        const srcT = t - lag;
        let sample = 0;
        if (srcT >= 0) {
          sample = xBC_pre[c * T + srcT];
        } else if (initialConvState) {
          const initSlot = convKernel + srcT; // srcT in {-1, -2, -3} -> slot in {3, 2, 1}
          if (initSlot >= 0 && initSlot < convKernel) {
            sample = initialConvState[wcOff + initSlot];
          }
        }
        const slot = convKernel - 1 - lag; // lag 0 -> slot 3 (newest)
        acc += w_conv[wcOff + slot] * sample;
      }
      const co = Math.fround(acc);
      conv_out[c * T + t] = co;
      xBC[c * T + t] = Math.fround(silu(co));
    }
    // Persistent 4-tap rolling buffer snapshot at final token T-1 (slots 0..3 = t-3..t)
    const lastT = T - 1;
    for (let s = 0; s < convKernel; s++) {
      const lag = convKernel - 1 - s;
      const srcT = lastT - lag;
      if (srcT >= 0) {
        conv_buf[wcOff + s] = xBC_pre[c * T + srcT];
      } else if (initialConvState) {
        const initSlot = convKernel + srcT;
        conv_buf[wcOff + s] =
          initSlot >= 0 && initSlot < convKernel ? initialConvState[wcOff + initSlot] : 0;
      } else {
        conv_buf[wcOff + s] = 0;
      }
    }
  }

  // Split xBC -> x [16, 4, T], B [8, 4, T], C [8, 4, T]
  const x = new Float32Array(nHeads * dHead * T);
  const B = new Float32Array(nGroups * dState * T);
  const C = new Float32Array(nGroups * dState * T);

  for (let c = 0; c < dInner; c++) {
    for (let t = 0; t < T; t++) {
      x[c * T + t] = xBC[c * T + t];
    }
  }
  const bOff = dInner;
  const cOff = dInner + nGroups * dState;
  for (let k = 0; k < nGroups * dState; k++) {
    for (let t = 0; t < T; t++) {
      B[k * T + t] = xBC[(bOff + k) * T + t];
      C[k * T + t] = xBC[(cOff + k) * T + t];
    }
  }

  // 5. Discretized SSM Recurrence & Readout + D Skip
  const A = new Float32Array(nHeads);
  for (let h = 0; h < nHeads; h++) {
    A[h] = Math.fround(-Math.exp(A_log[h]));
  }

  const dt = new Float32Array(nHeads * T);
  const dA = new Float32Array(nHeads * T);
  for (let h = 0; h < nHeads; h++) {
    for (let t = 0; t < T; t++) {
      const step = Math.fround(stableSoftplus(dt_raw[h * T + t] + dt_bias[h]));
      dt[h * T + t] = step;
      dA[h * T + t] = Math.fround(Math.exp(step * A[h]));
    }
  }

  const H_history = new Float32Array(T * stateStride);
  const H_prev = new Float32Array(stateStride);
  const H = new Float32Array(stateStride);
  const y_ssm = new Float32Array(nHeads * dHead * T);
  const y = new Float32Array(nHeads * dHead * T);

  for (let t = 0; t < T; t++) {
    const curStateOff = t * stateStride;
    const prevStateOff = (t - 1) * stateStride;

    for (let h = 0; h < nHeads; h++) {
      const g = Math.floor(h / headsPerGroup);
      const decay = dA[h * T + t];
      const step = dt[h * T + t];
      const dSkip = D[h];

      for (let p = 0; p < dHead; p++) {
        const xVal = x[(h * dHead + p) * T + t];
        let readout = 0;
        for (let n = 0; n < dState; n++) {
          const localIdx = (h * dHead + p) * dState + n;
          const prevVal =
            t === 0
              ? initialSsmState
                ? initialSsmState[localIdx]
                : 0
              : H_history[prevStateOff + localIdx];
          const bVal = B[(g * dState + n) * T + t];
          const cVal = C[(g * dState + n) * T + t];
          const hNext = Math.fround(decay * prevVal + step * xVal * bVal);
          H_history[curStateOff + localIdx] = hNext;
          readout += hNext * cVal;
        }
        const ySsmVal = Math.fround(readout);
        y_ssm[(h * dHead + p) * T + t] = ySsmVal;
        y[(h * dHead + p) * T + t] = Math.fround(ySsmVal + dSkip * xVal);
      }
    }
  }

  // Snapshot persistent H (at T-1) and H_prev (at T-2, or initialSsmState when T=1, else 0)
  const lastOff = (T - 1) * stateStride;
  H.set(H_history.subarray(lastOff, lastOff + stateStride));
  if (T >= 2) {
    const prevOff = (T - 2) * stateStride;
    H_prev.set(H_history.subarray(prevOff, prevOff + stateStride));
  } else if (initialSsmState) {
    H_prev.set(initialSsmState);
  }

  // 6. MambaRMSNormGated (norm_before_gate = False): Gate FIRST -> 8-Group RMSNorm -> * gamma_gate
  const { gated, rms_gate, y_norm_pre, y_norm } = applyGroupedRmsNormGated(
    y,
    z,
    gamma_gate,
    T,
    nGroups,
    eps,
  );

  // 7. Output Projection W_out [8, 64] @ y_norm [64, T] + Residual X [8, T] (optional residual gate)
  const Mamba_out = new Float32Array(d * T);
  const Y = new Float32Array(d * T);
  for (let i = 0; i < d; i++) {
    const woOff = i * dInner;
    for (let t = 0; t < T; t++) {
      let acc = 0;
      for (let c = 0; c < dInner; c++) {
        acc += W_out[woOff + c] * y_norm[c * T + t];
      }
      const mOut = Math.fround(acc);
      Mamba_out[i * T + t] = mOut;
      const delta = residualGate ? Math.fround(residualGate[i] * mOut) : mOut;
      Y[i * T + t] = Math.fround(X[i * T + t] + delta);
    }
  }

  const result: MambaSimResult = {
    tokens: T,
    startTokenOffset,
    hasInitialState,
    ...(initialConvState ? { initialConvState } : {}),
    ...(initialSsmState ? { initialSsmState } : {}),
    tokenIds,
    d,
    nHeads,
    nGroups,
    headsPerGroup,
    dHead,
    dInner,
    dState,
    convKernel,
    convDim,
    inProjDim,
    groupSize,
    eps,
    tokenLabels,
    X,
    ...(X_mod ? { X_mod } : {}),
    X_norm,
    rms_in,
    gamma_in,
    U,
    W_in,
    z,
    xBC_pre,
    dt_raw,
    w_conv,
    b_conv,
    conv_buf,
    conv_out,
    xBC,
    x,
    B,
    C,
    A_log,
    A,
    dt_bias,
    dt,
    dA,
    D,
    H_prev,
    H,
    H_history,
    y_ssm,
    y,
    gated,
    rms_gate,
    y_norm_pre,
    gamma_gate,
    y_norm,
    W_out,
    Mamba_out,
    ...(residualGate ? { residualGate } : {}),
    residualGateApplied,
    Y,
  };

  if (!fixture) {
    DEFAULT_SIM_CACHE.set(T, result);
  }
  return result;
}

/**
 * Resolve the owning miniature head (0..15) for any MambaCellRef tied to a specific head
 * so inspected cells, formula readouts, H_{t-1}/H_t sheets, and dependency threads stay in sync.
 */
export function inferMambaCellHead(cell: MambaCellRef | null | undefined): number | null {
  if (!cell) return null;
  const clampH = (h: number) => Math.max(0, Math.min(MAMBA_SIM_CONFIG.nHeads - 1, h));
  if (cell.headIndex !== undefined) {
    return clampH(cell.headIndex);
  }
  switch (cell.tensor) {
    case 'x':
    case 'y':
    case 'y_norm':
    case 'z':
    case 'w_out':
      return clampH(Math.floor(cell.col / MAMBA_SIM_CONFIG.dHead));
    case 'dt':
      return clampH(cell.col);
    case 'b':
    case 'c': {
      const g = cell.groupIndex ?? Math.floor(cell.col / MAMBA_SIM_CONFIG.dState);
      return clampH(g * MAMBA_SIM_CONFIG.headsPerGroup);
    }
    case 'xbc_pre':
    case 'conv_buf':
      if (cell.col < MAMBA_SIM_CONFIG.dInner) {
        return clampH(Math.floor(cell.col / MAMBA_SIM_CONFIG.dHead));
      }
      return null;
    case 'w_in': {
      const c = cell.col;
      if (c < MAMBA_SIM_CONFIG.dInner) {
        return clampH(Math.floor(c / MAMBA_SIM_CONFIG.dHead));
      }
      if (c >= MAMBA_SIM_CONFIG.dInner + MAMBA_SIM_CONFIG.convDim) {
        return clampH(c - MAMBA_SIM_CONFIG.dInner - MAMBA_SIM_CONFIG.convDim);
      }
      return null;
    }
    default:
      return null;
  }
}

const MAMBA_SEQUENCE_TENSORS = new Set<MambaTensorKey>([
  'input',
  'rmsnorm',
  'z',
  'xbc_pre',
  'x',
  'b',
  'c',
  'dt',
  'y',
  'y_norm',
  'residual',
]);

/**
 * Resolve the inspected token index (0..tokens-1) for any MambaCellRef so H_{t-1}, H_t,
 * the 64-head bank, and the 4-tap Conv1D rolling buffer synchronize to historical rows t < T-1.
 */
export function inferMambaCellToken(
  cell: MambaCellRef | null | undefined,
  tokens: number,
): number | null {
  if (!cell) return null;
  const maxT = Math.max(0, Math.round(tokens) - 1);
  const clampT = (t: number) => Math.max(0, Math.min(maxT, t));
  if (cell.tokenIndex !== undefined) {
    return clampT(cell.tokenIndex);
  }
  if (MAMBA_SEQUENCE_TENSORS.has(cell.tensor)) {
    return clampT(cell.row);
  }
  return null;
}

/**
 * Authoritative lookup for a single rolling Conv1D tap at `(channel, tokIdx, slot)`
 * respecting externally seeded `initialConvState` for `srcT < 0`.
 * Shared between `mambaAtlas.ts` and `describeMambaCellFormula()` so rendered values and
 * inspected formulas stay 100% identical.
 */
export function readMambaConvTapValue(
  sim: MambaSimResult,
  channel: number,
  tokIdx: number,
  slot: number,
): {
  readonly value: number;
  readonly sourceKind: 'current_step' | 'seeded_state' | 'zero_pad';
  readonly localSrcT: number;
  readonly absSrcT: number;
  readonly initSlot: number;
} {
  const { convKernel, tokens: T } = sim;
  const lag = convKernel - 1 - slot; // slot 3 = lag 0 (tokIdx), slot 0 = lag 3 (tokIdx-3)
  const localSrcT = tokIdx - lag;
  const absSrcT = sim.startTokenOffset + localSrcT;
  if (localSrcT >= 0) {
    return {
      value: sim.xBC_pre[channel * T + localSrcT],
      sourceKind: 'current_step',
      localSrcT,
      absSrcT,
      initSlot: -1,
    };
  }
  const initSlot = convKernel + localSrcT;
  if (sim.initialConvState && initSlot >= 0 && initSlot < convKernel) {
    return {
      value: sim.initialConvState[channel * convKernel + initSlot],
      sourceKind: 'seeded_state',
      localSrcT,
      absSrcT,
      initSlot,
    };
  }
  return {
    value: 0,
    sourceKind: 'zero_pad',
    localSrcT,
    absSrcT,
    initSlot: -1,
  };
}

/**
 * Authoritative lookup for prior SSM state `H_{t-1}` at `(tokIdx, localIdx)`,
 * respecting externally seeded `initialSsmState` at `tokIdx === 0`.
 */
export function readMambaPriorSsmValue(
  sim: MambaSimResult,
  tokIdx: number,
  localIdx: number,
): { readonly value: number; readonly isSeeded: boolean } {
  const stateStride = sim.nHeads * sim.dHead * sim.dState;
  if (tokIdx > 0) {
    return { value: sim.H_history[(tokIdx - 1) * stateStride + localIdx], isSeeded: false };
  }
  if (sim.initialSsmState) {
    return { value: sim.initialSsmState[localIdx], isSeeded: true };
  }
  return { value: 0, isSeeded: false };
}

function tokTag(sim: MambaSimResult, t: number): string {
  const clamped = Math.max(0, Math.min(sim.tokens - 1, t));
  return `t${sim.startTokenOffset + clamped}("${sim.tokenLabels[clamped]}")`;
}

/**
 * Compute exact rounded numerical explanation & upstream operand spans for any Mamba-2 cell.
 */
export function describeMambaCellFormula(
  sim: MambaSimResult,
  cell: MambaCellRef,
): MambaCellFormula {
  const { d, dHead, dInner, dState, convKernel, convDim, inProjDim, groupSize, tokens: T } = sim;
  const clampT = (t: number) => Math.max(0, Math.min(T - 1, t));
  const clampC = (c: number, max: number) => Math.max(0, Math.min(max - 1, c));
  const lastT = T - 1;

  switch (cell.tensor) {
    case 'input': {
      const t = clampT(cell.row);
      const absT = sim.startTokenOffset + t;
      const i = clampC(cell.col, d);
      const val = sim.X[i * T + t];
      return {
        cell,
        tensorLabel: 'Input X',
        coordLabel: `X[ch=${i}, ${tokTag(sim, t)}]`,
        headBadge: null,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `X[${i}, t=${absT}] = ${fmtSigned(val)}  (miniature d=8 · prod 2,688)`,
        subEquation: `Synthetic Float32 token embedding · RMS(X[:, t=${absT}]) = ${sim.rms_in[t].toFixed(3)} (ε=${formatEpsLabel(sim.eps)})`,
        targetSpan: { tensor: 'input', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'rmsnorm': {
      const t = clampT(cell.row);
      const absT = sim.startTokenOffset + t;
      const i = clampC(cell.col, d);
      const normIn = sim.X_mod ?? sim.X;
      const inSym = sim.X_mod ? 'X_mod' : 'X';
      const xVal = normIn[i * T + t];
      const rms = sim.rms_in[t];
      const xn = sim.X_norm[i * T + t];
      const g = sim.gamma_in[i];
      const uVal = sim.U[i * T + t];
      return {
        cell,
        tensorLabel: 'RMSNorm U',
        coordLabel: `U[ch=${i}, ${tokTag(sim, t)}]`,
        headBadge: null,
        value: uVal,
        isMasked: false,
        valueText: fmtSigned(uVal),
        equation: `U[${i}, t=${absT}] = (${fmtSigned(xVal)} / ${rms.toFixed(3)}) · γ[${i}](${g.toFixed(3)}) = ${fmtSigned(uVal)}`,
        subEquation: `X_norm = ${fmtSigned(xn)} · √(mean(${inSym}[:, t=${absT}]²) + ${formatEpsLabel(sim.eps)}) = ${rms.toFixed(4)}`,
        targetSpan: { tensor: 'rmsnorm', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'target' },
        upstreamSpans: [
          { tensor: 'input', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_a' },
        ],
      };
    }

    case 'w_in': {
      const j = clampC(cell.row, d);
      const outCol = clampC(cell.col, inProjDim);
      const val = sim.W_in[outCol * d + j];
      const branch =
        outCol < dInner
          ? `Gate z[${outCol}]`
          : outCol < dInner + convDim
            ? `xBC_pre[${outCol - dInner}]`
            : `Δt_raw[H${String(outCol - dInner - convDim).padStart(2, '0')}]`;
      return {
        cell,
        tensorLabel: 'In-Proj W_in',
        coordLabel: `W_in[${outCol}, ${j}] → ${branch}`,
        headBadge: null,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `W_in[${outCol}, ${j}] = ${fmtSigned(val)}  (fused 208×8 · prod 10,304×2,688)`,
        subEquation: `Maps normalized channel U[${j}, t] into ${branch}`,
        targetSpan: { tensor: 'w_in', rowStart: j, rowEnd: j, colStart: outCol, colEnd: outCol, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'z': {
      const t = clampT(cell.row);
      const absT = sim.startTokenOffset + t;
      const c = clampC(cell.col, dInner);
      const h = Math.floor(c / dHead);
      const g = mambaGroupForHead(h);
      const val = sim.z[c * T + t];
      const sVal = silu(val);
      const terms: string[] = [];
      for (let j = 0; j < 3; j++) {
        terms.push(`${fmtSigned(sim.W_in[c * d + j], 2)}·${fmtSigned(sim.U[j * T + t], 2)}`);
      }
      return {
        cell,
        tensorLabel: 'Gate Branch z',
        coordLabel: `z[c=${c}, ${tokTag(sim, t)}]`,
        headBadge: `${formatMambaHeadId(h)} · G${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `z[${c}, t=${absT}] = Σⱼ W_in[${c},j]·U[j,t] = ${terms.join(' + ')} + … = ${fmtSigned(val)}`,
        subEquation: `Bypasses Conv1D & SSM → SiLU(z) = ${fmtSigned(sVal)} multiplies y[${c}] BEFORE 8-group RMSNorm`,
        targetSpan: { tensor: 'z', rowStart: t, rowEnd: t, colStart: c, colEnd: c, role: 'target' },
        upstreamSpans: [
          { tensor: 'w_in', rowStart: 0, rowEnd: d - 1, colStart: c, colEnd: c, role: 'operand_a' },
          { tensor: 'rmsnorm', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_b' },
        ],
      };
    }

    case 'xbc_pre': {
      const t = clampT(cell.row);
      const absT = sim.startTokenOffset + t;
      const c = clampC(cell.col, convDim);
      const val = sim.xBC_pre[c * T + t];
      const wRow = dInner + c;
      const seg =
        c < dInner
          ? `x_pre[${c}]`
          : c < dInner + sim.nGroups * dState
            ? `B_pre[G${Math.floor((c - dInner) / dState)}, n=${(c - dInner) % dState}]`
            : `C_pre[G${Math.floor((c - dInner - sim.nGroups * dState) / dState)}, n=${(c - dInner - sim.nGroups * dState) % dState}]`;
      return {
        cell,
        tensorLabel: 'Pre-Conv xBC',
        coordLabel: `${seg} @ ${tokTag(sim, t)}`,
        headBadge: null,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `xBC_pre[${c}, t=${absT}] = W_in[${wRow}, :] · U[:, t=${absT}] = ${fmtSigned(val)}`,
        subEquation: `Feeds depthwise causal Conv1D (k=4, always BF16 in production) · conv_dim=128 (prod 6,144)`,
        targetSpan: { tensor: 'xbc_pre', rowStart: t, rowEnd: t, colStart: c, colEnd: c, role: 'target' },
        upstreamSpans: [
          { tensor: 'w_in', rowStart: 0, rowEnd: d - 1, colStart: wRow, colEnd: wRow, role: 'operand_a' },
          { tensor: 'rmsnorm', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_b' },
        ],
      };
    }

    case 'conv_buf': {
      const slot = clampC(cell.tapIndex ?? cell.row, convKernel);
      const c = clampC(cell.col, convDim);
      const tok = clampT(cell.tokenIndex ?? lastT);
      const absTok = sim.startTokenOffset + tok;
      const lag = convKernel - 1 - slot; // slot 3 = lag 0 (tok), slot 0 = lag 3 (tok-3)
      const tap = readMambaConvTapValue(sim, c, tok, slot);
      const bufVal = tap.value;
      const wTap = sim.w_conv[c * convKernel + slot];
      const coVal = sim.conv_out[c * T + tok];
      const siluVal = sim.xBC[c * T + tok];
      const srcNote =
        tap.sourceKind === 'current_step'
          ? tokTag(sim, tap.localSrcT)
          : tap.sourceKind === 'seeded_state'
            ? `carried conv_state[slot=${tap.initSlot}] (abs t${tap.absSrcT})`
            : '0 (causal left-pad)';
      return {
        cell: { ...cell, row: slot, col: c, tapIndex: slot, tokenIndex: tok },
        tensorLabel:
          tap.sourceKind === 'seeded_state' ? 'Carried Conv1D Tap' : 'Causal Conv1D Tap',
        coordLabel: `slot ${slot} (t−${lag} @ t${absTok}) · ch=${c}`,
        headBadge:
          tap.sourceKind === 'seeded_state' ? 'Carried State · BF16 (prod)' : 'Always BF16 (prod)',
        value: bufVal,
        isMasked: false,
        valueText: fmtSigned(bufVal),
        equation: `conv_buf[${c}, slot=${slot}] = ${fmtSigned(bufVal)} (${srcNote}) · w[${slot}]=${fmtSigned(wTap)}`,
        subEquation: `Conv1D(t=${absTok}) = ${fmtSigned(coVal)} → SiLU = ${fmtSigned(siluVal)} · Fixed 4-tap buffer (~1.08 MiB BF16 / 23 bays)`,
        targetSpan: {
          tensor: 'conv_buf',
          rowStart: 0,
          rowEnd: 0,
          colStart: c,
          colEnd: c,
          tapStart: slot,
          tapEnd: slot,
          tokenIndex: tok,
          role: 'target',
        },
        upstreamSpans:
          tap.sourceKind === 'current_step'
            ? [
                {
                  tensor: 'xbc_pre',
                  rowStart: tap.localSrcT,
                  rowEnd: tap.localSrcT,
                  colStart: c,
                  colEnd: c,
                  role: 'operand_a',
                },
              ]
            : [],
      };
    }

    case 'x': {
      const t = clampT(cell.row);
      const absT = sim.startTokenOffset + t;
      const h = cell.headIndex !== undefined
        ? clampC(cell.headIndex, sim.nHeads)
        : clampC(Math.floor(cell.col / dHead), sim.nHeads);
      const p =
        cell.headIndex !== undefined && cell.col >= 0 && cell.col < dHead
          ? cell.col
          : clampC(cell.col % dHead, dHead);
      const c = h * dHead + p;
      const g = mambaGroupForHead(h);
      const val = sim.x[c * T + t];
      const co = sim.conv_out[c * T + t];
      const tapStart = Math.max(0, t - (convKernel - 1));
      const absTapStart = Math.max(0, absT - (convKernel - 1));
      return {
        cell: { ...cell, row: t, col: p, headIndex: h, groupIndex: g, tokenIndex: t },
        tensorLabel: `Post-SiLU x (${formatMambaHeadId(h)})`,
        coordLabel: `x[${formatMambaHeadId(h)}, p=${p}, ${tokTag(sim, t)}]`,
        headBadge: `${formatMambaHeadId(h)} · G${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `x[${formatMambaHeadId(h)}, p=${p}, t=${absT}] = SiLU(Conv1D(${fmtSigned(co)})) = ${fmtSigned(val)}`,
        subEquation: `Depthwise causal k=4 taps t∈[${absTapStart}..${absT}] (b=${fmtSigned(sim.b_conv[c], 2)}) → feeds SSM write & D·x skip`,
        targetSpan: {
          tensor: 'x',
          rowStart: t,
          rowEnd: t,
          colStart: p,
          colEnd: p,
          headIndex: h,
          role: 'target',
        },
        upstreamSpans: [
          {
            tensor: 'xbc_pre',
            rowStart: tapStart,
            rowEnd: t,
            colStart: c,
            colEnd: c,
            role: 'operand_a',
          },
          {
            tensor: 'conv_buf',
            rowStart: 0,
            rowEnd: 0,
            colStart: c,
            colEnd: c,
            tapStart: 0,
            tapEnd: convKernel - 1,
            tokenIndex: t,
            role: 'operand_b',
          },
        ],
      };
    }

    case 'b':
    case 'c': {
      const isB = cell.tensor === 'b';
      const t = clampT(cell.row);
      const absT = sim.startTokenOffset + t;
      const flatCol = clampC(cell.col, sim.nGroups * dState);
      // Support both flat sheet columns (0..31, with or without groupIndex) and explicit local coords (groupIndex > 0, col in 0..3)
      const isLocalWithGroup =
        cell.groupIndex !== undefined &&
        cell.col >= 0 &&
        cell.col < dState &&
        clampC(cell.groupIndex, sim.nGroups) > 0;
      const g = isLocalWithGroup
        ? clampC(cell.groupIndex!, sim.nGroups)
        : Math.floor(flatCol / dState);
      const n = isLocalWithGroup ? cell.col : flatCol % dState;
      const resolvedFlatCol = g * dState + n;
      const convCh = dInner + (isB ? 0 : sim.nGroups * dState) + resolvedFlatCol;
      const val = isB ? sim.B[resolvedFlatCol * T + t] : sim.C[resolvedFlatCol * T + t];
      const co = sim.conv_out[convCh * T + t];
      const tapStart = Math.max(0, t - (convKernel - 1));
      const h0 = g * sim.headsPerGroup;
      const h1 = h0 + sim.headsPerGroup - 1;
      return {
        cell: { ...cell, row: t, col: resolvedFlatCol, groupIndex: g, tokenIndex: t },
        tensorLabel: isB ? `Input Proj B (G${g})` : `Readout Proj C (G${g})`,
        coordLabel: `${isB ? 'B' : 'C'}[G${g}, n=${n}, ${tokTag(sim, t)}]`,
        headBadge: `G${g} → ${formatMambaHeadId(h0)}–${formatMambaHeadId(h1)}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `${isB ? 'B' : 'C'}[G${g}, n=${n}, t=${absT}] = SiLU(Conv1D(${fmtSigned(co)})) = ${fmtSigned(val)}`,
        subEquation: isB
          ? `Shared write key across group G${g} heads (2 mini / 8 prod heads per group)`
          : `Shared readout query across group G${g} heads (y = Σₙ H·C + D·x)`,
        targetSpan: {
          tensor: cell.tensor,
          rowStart: t,
          rowEnd: t,
          colStart: resolvedFlatCol,
          colEnd: resolvedFlatCol,
          groupIndex: g,
          role: 'target',
        },
        upstreamSpans: [
          {
            tensor: 'xbc_pre',
            rowStart: tapStart,
            rowEnd: t,
            colStart: convCh,
            colEnd: convCh,
            role: 'operand_a',
          },
          {
            tensor: 'conv_buf',
            rowStart: 0,
            rowEnd: 0,
            colStart: convCh,
            colEnd: convCh,
            tapStart: 0,
            tapEnd: convKernel - 1,
            tokenIndex: t,
            role: 'operand_b',
          },
        ],
      };
    }

    case 'dt': {
      const t = clampT(cell.row);
      const h = clampC(cell.headIndex ?? cell.col, sim.nHeads);
      const g = mambaGroupForHead(h);
      const raw = sim.dt_raw[h * T + t];
      const bias = sim.dt_bias[h];
      const step = sim.dt[h * T + t];
      const aVal = sim.A[h];
      const decay = sim.dA[h * T + t];
      const wRow = dInner + convDim + h;
      return {
        cell: { ...cell, headIndex: h, groupIndex: g },
        tensorLabel: `Step Δt (${formatMambaHeadId(h)})`,
        coordLabel: `Δt[${formatMambaHeadId(h)}, ${tokTag(sim, t)}]`,
        headBadge: `${formatMambaHeadId(h)} · G${g}`,
        value: step,
        isMasked: false,
        valueText: fmtSigned(step),
        equation: `Δt = softplus(${fmtSigned(raw)} + ${fmtSigned(bias)}) = ${step.toFixed(3)} → dA = exp(Δt·A) = ${decay.toFixed(3)}`,
        subEquation: `A[${formatMambaHeadId(h)}] = −exp(${fmtSigned(sim.A_log[h], 2)}) = ${fmtSigned(aVal, 3)} < 0 · Retention dA ∈ (0, 1)`,
        targetSpan: { tensor: 'dt', rowStart: t, rowEnd: t, colStart: h, colEnd: h, headIndex: h, role: 'target' },
        upstreamSpans: [
          { tensor: 'w_in', rowStart: 0, rowEnd: d - 1, colStart: wRow, colEnd: wRow, role: 'operand_a' },
          { tensor: 'rmsnorm', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_b' },
        ],
      };
    }

    case 'h_prev':
    case 'h_curr': {
      const isPrev = cell.tensor === 'h_prev';
      const h = clampC(cell.headIndex ?? 0, sim.nHeads);
      const g = mambaGroupForHead(h);
      const p = clampC(cell.row, dHead);
      const n = clampC(cell.col, dState);
      const t = clampT(cell.tokenIndex ?? lastT);
      const stateStride = sim.nHeads * dHead * dState;
      const localIdx = (h * dHead + p) * dState + n;

      const prior = readMambaPriorSsmValue(sim, t, localIdx);
      const prevVal = prior.value;
      const currVal = sim.H_history[t * stateStride + localIdx];
      const decay = sim.dA[h * T + t];
      const step = sim.dt[h * T + t];
      const xVal = sim.x[(h * dHead + p) * T + t];
      const bVal = sim.B[(g * dState + n) * T + t];

      if (isPrev) {
        return {
          cell: { ...cell, headIndex: h, groupIndex: g, tokenIndex: t },
          tensorLabel: prior.isSeeded
            ? `Carried State H_{t−1} (${formatMambaHeadId(h)})`
            : `Prior State H_{t−1} (${formatMambaHeadId(h)})`,
          coordLabel: `H_{t−1}[${formatMambaHeadId(h)}, p=${p}, n=${n}]`,
          headBadge: prior.isSeeded
            ? `${formatMambaHeadId(h)} · G${g} · Seeded Cache`
            : `${formatMambaHeadId(h)} · G${g}`,
          value: prevVal,
          isMasked: false,
          valueText: fmtSigned(prevVal),
          equation:
            t === 0 && !prior.isSeeded
              ? `H_{−1}[${formatMambaHeadId(h)}, p=${p}, n=${n}] = +0.000  (initial zero state at T=1)`
              : prior.isSeeded
                ? `H_{t−1}[${formatMambaHeadId(h)}, p=${p}, n=${n}] = ${fmtSigned(prevVal)}  (carried FP32 state before ${tokTag(sim, t)})`
                : `H_{t−1}[${formatMambaHeadId(h)}, p=${p}, n=${n}] = ${fmtSigned(prevVal)}  (before ${tokTag(sim, t)})`,
          subEquation: `Decays by dA=${decay.toFixed(3)} in-place into H_t = ${fmtSigned(currVal)} (46.0 MiB FP32 / 23 bays fixed)`,
          targetSpan: { tensor: 'h_prev', rowStart: p, rowEnd: p, colStart: n, colEnd: n, headIndex: h, tokenIndex: t, role: 'target' },
          upstreamSpans: [],
        };
      }

      return {
        cell: { ...cell, headIndex: h, groupIndex: g, tokenIndex: t },
        tensorLabel: `SSM State H_t (${formatMambaHeadId(h)})`,
        coordLabel: `H_t[${formatMambaHeadId(h)}, p=${p}, n=${n}] @ ${tokTag(sim, t)}`,
        headBadge: `${formatMambaHeadId(h)} · G${g}`,
        value: currVal,
        isMasked: false,
        valueText: fmtSigned(currVal),
        equation: `H_t = ${decay.toFixed(3)}·(${fmtSigned(prevVal)}) + ${step.toFixed(3)}·(${fmtSigned(xVal)})·(${fmtSigned(bVal)}) = ${fmtSigned(currVal)}`,
        subEquation: `In-place FP32 recurrence H_t = dA·H_{t−1} + Δt·x·B (4×4 mini / 64×128 prod · fixed size T=1→1M)`,
        targetSpan: { tensor: 'h_curr', rowStart: p, rowEnd: p, colStart: n, colEnd: n, headIndex: h, tokenIndex: t, role: 'target' },
        upstreamSpans: [
          { tensor: 'h_prev', rowStart: p, rowEnd: p, colStart: n, colEnd: n, headIndex: h, tokenIndex: t, role: 'operand_a' },
          { tensor: 'x', rowStart: t, rowEnd: t, colStart: p, colEnd: p, headIndex: h, role: 'operand_b' },
          { tensor: 'b', rowStart: t, rowEnd: t, colStart: g * dState + n, colEnd: g * dState + n, groupIndex: g, role: 'operand_b' },
          { tensor: 'dt', rowStart: t, rowEnd: t, colStart: h, colEnd: h, headIndex: h, role: 'operand_a' },
        ],
      };
    }

    case 'y': {
      const t = clampT(cell.row);
      const absT = sim.startTokenOffset + t;
      const c = clampC(cell.col, dInner);
      const h = Math.floor(c / dHead);
      const p = c % dHead;
      const g = mambaGroupForHead(h);
      const ySsm = sim.y_ssm[c * T + t];
      const dSkip = sim.D[h];
      const xVal = sim.x[c * T + t];
      const yVal = sim.y[c * T + t];
      return {
        cell: { ...cell, row: t, col: c, headIndex: h, groupIndex: g, tokenIndex: t },
        tensorLabel: 'SSM Readout y',
        coordLabel: `y[${formatMambaHeadId(h)}, p=${p}, ${tokTag(sim, t)}]`,
        headBadge: `${formatMambaHeadId(h)} · G${g}`,
        value: yVal,
        isMasked: false,
        valueText: fmtSigned(yVal),
        equation: `y[${c}, t=${absT}] = Σₙ H_t[${p},n]·C[G${g},n](${fmtSigned(ySsm)}) + D(${dSkip.toFixed(2)})·x(${fmtSigned(xVal)}) = ${fmtSigned(yVal)}`,
        subEquation: `Readout contracts state dim n=0..3 with group C[G${g}] plus per-head skip D[${formatMambaHeadId(h)}]·x`,
        targetSpan: { tensor: 'y', rowStart: t, rowEnd: t, colStart: c, colEnd: c, headIndex: h, role: 'target' },
        upstreamSpans: [
          { tensor: 'h_curr', rowStart: p, rowEnd: p, colStart: 0, colEnd: dState - 1, headIndex: h, tokenIndex: t, role: 'operand_a' },
          { tensor: 'c', rowStart: t, rowEnd: t, colStart: g * dState, colEnd: (g + 1) * dState - 1, groupIndex: g, role: 'operand_b' },
          { tensor: 'x', rowStart: t, rowEnd: t, colStart: p, colEnd: p, headIndex: h, role: 'operand_b' },
        ],
      };
    }

    case 'y_norm': {
      const t = clampT(cell.row);
      const c = clampC(cell.col, dInner);
      const h = Math.floor(c / dHead);
      const g = mambaGroupForHead(h);
      const yVal = sim.y[c * T + t];
      const zVal = sim.z[c * T + t];
      const sVal = silu(zVal);
      const gVal = sim.gated[c * T + t];
      const rmsG = sim.rms_gate[g * T + t];
      const gam = sim.gamma_gate[c];
      const outVal = sim.y_norm[c * T + t];
      return {
        cell: { ...cell, headIndex: h, groupIndex: g },
        tensorLabel: 'Gate→8-Group Norm',
        coordLabel: `y_norm[c=${c}, G${g}, ${tokTag(sim, t)}]`,
        headBadge: `${formatMambaHeadId(h)} · G${g}`,
        value: outVal,
        isMasked: false,
        valueText: fmtSigned(outVal),
        equation: `y_norm = ((${fmtSigned(yVal)}·SiLU(${fmtSigned(zVal)})) / ${rmsG.toFixed(3)}) · γ(${gam.toFixed(2)}) = ${fmtSigned(outVal)}`,
        subEquation: `norm_before_gate=False: gate FIRST (${fmtSigned(yVal)}·${fmtSigned(sVal)}=${fmtSigned(gVal)}), then G${g} RMSNorm (ch ${g * groupSize}..${(g + 1) * groupSize - 1}, ε=1e−5)`,
        targetSpan: { tensor: 'y_norm', rowStart: t, rowEnd: t, colStart: c, colEnd: c, headIndex: h, role: 'target' },
        upstreamSpans: [
          { tensor: 'y', rowStart: t, rowEnd: t, colStart: g * groupSize, colEnd: (g + 1) * groupSize - 1, role: 'operand_a' },
          { tensor: 'z', rowStart: t, rowEnd: t, colStart: g * groupSize, colEnd: (g + 1) * groupSize - 1, role: 'operand_b' },
        ],
      };
    }

    case 'w_out': {
      const i = clampC(cell.row, d);
      const c = clampC(cell.col, dInner);
      const h = Math.floor(c / dHead);
      const val = sim.W_out[i * dInner + c];
      return {
        cell: { ...cell, headIndex: h, groupIndex: mambaGroupForHead(h) },
        tensorLabel: 'Out-Proj W_out',
        coordLabel: `W_out[out=${i}, in=${c}]`,
        headBadge: `${formatMambaHeadId(h)} · G${mambaGroupForHead(h)}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `W_out[${i}, ${c}] = ${fmtSigned(val)}  (mini 8×64 · prod 2,688×4,096)`,
        subEquation: `Projects gated channel y_norm[${c}] (${formatMambaHeadId(h)}) back into hidden stream channel ${i}`,
        targetSpan: { tensor: 'w_out', rowStart: i, rowEnd: i, colStart: c, colEnd: c, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'residual': {
      const t = clampT(cell.row);
      const absT = sim.startTokenOffset + t;
      const i = clampC(cell.col, d);
      const xVal = sim.X[i * T + t];
      const delta = sim.Mamba_out[i * T + t];
      const yVal = sim.Y[i * T + t];
      const gateVal = sim.residualGateApplied && sim.residualGate ? sim.residualGate[i] : null;
      return {
        cell,
        tensorLabel: '+ Residual Y',
        coordLabel: `Y[ch=${i}, ${tokTag(sim, t)}]`,
        headBadge: null,
        value: yVal,
        isMasked: false,
        valueText: fmtSigned(yVal),
        equation:
          gateVal !== null
            ? `Y[${i}, t=${absT}] = X(${fmtSigned(xVal)}) + gate[${i}](${fmtSigned(gateVal)})·(W_out·y_norm)(${fmtSigned(delta)}) = ${fmtSigned(yVal)}`
            : `Y[${i}, t=${absT}] = X(${fmtSigned(xVal)}) + (W_out·y_norm)(${fmtSigned(delta)}) = ${fmtSigned(yVal)}`,
        subEquation: `Adds Mamba-2 output delta into the continuous 2,688-channel emerald residual stream`,
        targetSpan: { tensor: 'residual', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'target' },
        upstreamSpans: [
          { tensor: 'input', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'operand_a' },
          { tensor: 'w_out', rowStart: i, rowEnd: i, colStart: 0, colEnd: dInner - 1, role: 'operand_b' },
          { tensor: 'y_norm', rowStart: t, rowEnd: t, colStart: 0, colEnd: dInner - 1, role: 'operand_b' },
        ],
      };
    }
  }
}

export const MAMBA_STEPPER_STEPS: readonly StepperStepDef[] = [
  {
    index: 0,
    shortLabel: '1. RMSNorm',
    title: 'Pre-Layer RMSNorm (ε=1e−5)',
    summary: 'Normalize Input X by √(mean(X²)+1e−5) and scale by learned γ_in.',
  },
  {
    index: 1,
    shortLabel: '2. In-Proj',
    title: 'Fused In-Proj → z, xBC, Δt',
    summary: 'Project U into bypass gate z, Conv1D stream xBC_pre, and per-head step Δt_raw.',
  },
  {
    index: 2,
    shortLabel: '3. Causal Conv1D',
    title: 'Depthwise Causal Conv1D (k=4) + SiLU',
    summary: 'Mix last 4 causal taps (always BF16 in prod) + SiLU → split x, B, C.',
  },
  {
    index: 3,
    shortLabel: '4. SSM Update',
    title: 'In-Place SSM Recurrence (H_t)',
    summary: 'Update fixed-size H_t = exp(Δt·A)·H_{t−1} + Δt·x·B in FP32.',
  },
  {
    index: 4,
    shortLabel: '5. Readout + D',
    title: 'State Readout + Skip D·x',
    summary: 'Contract H_t with group C and add per-head skip connection D·x.',
  },
  {
    index: 5,
    shortLabel: '6. Gate→Norm→Out',
    title: 'Gate FIRST → 8-Group RMSNorm → W_out',
    summary: 'Apply y·SiLU(z) first, normalize in 8 groups (ε=1e−5), project via W_out + residual.',
  },
];

/** Map Mamba-2 stepper (step 0..5, subProgress 0..1) to a deterministic live MambaCellRef. */
export function mambaStepperCellForState(
  sim: MambaSimResult,
  activeHead: number,
  step: number,
  subProgress: number,
): MambaCellRef {
  const T = sim.tokens;
  const h = Math.max(0, Math.min(sim.nHeads - 1, activeHead));
  const g = mambaGroupForHead(h);
  const p01 = Math.max(0, Math.min(0.9999, subProgress));
  const t = Math.min(T - 1, Math.floor(p01 * T));

  switch (step) {
    case 0: {
      const i = Math.min(sim.d - 1, Math.floor(p01 * sim.d));
      return { tensor: 'rmsnorm', row: T - 1, col: i };
    }
    case 1: {
      // Alternate across z, xBC_pre, and dt
      if (p01 < 0.34) {
        return { tensor: 'z', row: T - 1, col: h * sim.dHead, headIndex: h, groupIndex: g };
      }
      if (p01 < 0.67) {
        return { tensor: 'xbc_pre', row: T - 1, col: h * sim.dHead, headIndex: h, groupIndex: g };
      }
      return { tensor: 'dt', row: T - 1, col: h, headIndex: h, groupIndex: g };
    }
    case 2: {
      // Show causal Conv1D tap -> post-SiLU x across tokens
      if (p01 < 0.45) {
        const slot = Math.min(sim.convKernel - 1, Math.floor((p01 / 0.45) * sim.convKernel));
        return { tensor: 'conv_buf', row: slot, col: h * sim.dHead, tapIndex: slot, headIndex: h, tokenIndex: T - 1 };
      }
      const p = Math.min(sim.dHead - 1, Math.floor(((p01 - 0.45) / 0.55) * sim.dHead));
      return { tensor: 'x', row: T - 1, col: p, headIndex: h, groupIndex: g, tokenIndex: T - 1 };
    }
    case 3: {
      // Scrub across H_t cells (p, n) for the active head at the latest token T-1
      const total = sim.dHead * sim.dState;
      const idx = Math.min(total - 1, Math.floor(p01 * total));
      const p = Math.floor(idx / sim.dState);
      const n = idx % sim.dState;
      return { tensor: 'h_curr', row: p, col: n, headIndex: h, groupIndex: g, tokenIndex: T - 1 };
    }
    case 4: {
      const p = Math.min(sim.dHead - 1, Math.floor(p01 * sim.dHead));
      return { tensor: 'y', row: t, col: h * sim.dHead + p, headIndex: h, groupIndex: g, tokenIndex: t };
    }
    default: {
      // Gate -> 8-Group RMSNorm -> Residual Output
      if (p01 < 0.55) {
        const p = Math.min(sim.dHead - 1, Math.floor((p01 / 0.55) * sim.dHead));
        return { tensor: 'y_norm', row: T - 1, col: h * sim.dHead + p, headIndex: h, groupIndex: g };
      }
      const i = Math.min(sim.d - 1, Math.floor(((p01 - 0.55) / 0.45) * sim.d));
      return { tensor: 'residual', row: T - 1, col: i };
    }
  }
}
