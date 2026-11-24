/**
 * Deterministic Miniature Float32 GQA Forward-Pass Simulator (Stage A+1 · Increment 2).
 *
 * Explicitly preserves Nemotron 3 Nano's exact 32 Q / 2 KV (16:1) grouping and NoPE
 * (zero positional embeddings, zero linear biases) at educational miniature scale:
 *   d = 8, n_q = 32, n_kv = 2, d_head = 4, concat_dim = 128, T ∈ [1..16].
 *
 * All tensor arithmetic runs in IEEE-754 Float32 for live inspection.
 * Token labels are educational words (not Nemotron tokenizer output or checkpoint inference).
 */

export const EDUCATIONAL_TOKENS = [
  'How',
  'does',
  'Nemotron',
  'use',
  'only',
  'six',
  'attention',
  'layers',
  'across',
  'one',
  'million',
  'tokens',
  'with',
  'shared',
  'KV',
  'heads',
] as const;

export const GQA_SIM_CONFIG = {
  d: 8,
  nQ: 32,
  nKv: 2,
  dHead: 4,
  qConcatDim: 128,
  kvConcatDim: 8,
  maxTokens: 16,
  rmsEps: 1e-6,
} as const;

export type GqaTensorKey =
  | 'input'
  | 'rmsnorm'
  | 'w_q'
  | 'w_k'
  | 'w_v'
  | 'q'
  | 'kv_k'
  | 'kv_v'
  | 'scores'
  | 'softmax'
  | 'v_out'
  | 'concat'
  | 'w_o'
  | 'residual';

export interface GqaCellRef {
  readonly tensor: GqaTensorKey;
  /** Visual sheet row (0 = top / -Z edge). */
  readonly row: number;
  /** Visual sheet column (0 = left / -X edge). */
  readonly col: number;
  /** Query head index 0..31 when applicable (q, scores, softmax, v_out). */
  readonly headIndex?: number;
  /** Shared KV head group 0 | 1 when applicable. */
  readonly kvHeadIndex?: 0 | 1;
  /** Token index 0..T-1 for individual KV slices. */
  readonly tokenIndex?: number;
}

export interface GqaSheetSpan {
  readonly tensor: GqaTensorKey;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  /** Inclusive token range for KV slice tracks (when highlighting K_g or V_g slices). */
  readonly tokenStart?: number;
  readonly tokenEnd?: number;
  readonly rowStart: number;
  readonly rowEnd: number;
  readonly colStart: number;
  readonly colEnd: number;
  readonly role: 'target' | 'operand_a' | 'operand_b';
}

export interface GqaCellFormula {
  readonly cell: GqaCellRef;
  readonly tensorLabel: string;
  readonly coordLabel: string;
  readonly headBadge: string | null;
  readonly value: number;
  readonly isMasked: boolean;
  readonly valueText: string;
  readonly equation: string;
  readonly subEquation: string;
  readonly targetSpan: GqaSheetSpan;
  readonly upstreamSpans: readonly GqaSheetSpan[];
}

export interface GqaSimResult {
  readonly tokens: number;
  readonly tokenLabels: readonly string[];
  readonly d: number;
  readonly nQ: number;
  readonly nKv: number;
  readonly dHead: number;
  readonly qConcatDim: number;
  /** Input hidden state X: [8, T], indexed [i * T + t] */
  readonly X: Float32Array;
  /** Pre-gamma unit-RMS normalized state X_norm: [8, T], indexed [i * T + t] */
  readonly X_norm: Float32Array;
  /** Per-token RMS denominators: [T] */
  readonly rms: Float32Array;
  /** Non-unit learned RMSNorm scale gamma: [8] */
  readonly gamma: Float32Array;
  /** Scaled RMSNorm output U = X_norm * gamma: [8, T], indexed [i * T + t] */
  readonly U: Float32Array;
  /** W_Q: [128, 8], indexed [outRow * 8 + j] where outRow = h * 4 + r */
  readonly W_Q: Float32Array;
  /** W_K: [8, 8], indexed [outRow * 8 + j] where outRow = g * 4 + r */
  readonly W_K: Float32Array;
  /** W_V: [8, 8], indexed [outRow * 8 + j] where outRow = g * 4 + r */
  readonly W_V: Float32Array;
  /** Q: [32, 4, T], indexed [(h * 4 + r) * T + t] */
  readonly Q: Float32Array;
  /** K: [2, 4, T], indexed [(g * 4 + r) * T + t] */
  readonly K: Float32Array;
  /** V: [2, 4, T], indexed [(g * 4 + r) * T + t] */
  readonly V: Float32Array;
  /** Causal scaled attention scores: [32, T, T], indexed [(h * T + q) * T + k]; -Infinity for k > q */
  readonly Scores: Float32Array;
  /** Row-wise causal softmax: [32, T, T], indexed [(h * T + q) * T + k]; 0 for k > q */
  readonly Softmax: Float32Array;
  /** Per-head weighted value outputs: [32, 4, T], indexed [(h * 4 + r) * T + q] */
  readonly V_out: Float32Array;
  /** Concatenated 32 heads: [128, T], indexed [c * T + q] where c = h * 4 + r */
  readonly Concat: Float32Array;
  /** Output projection W_O: [8, 128], indexed [i * 128 + c] */
  readonly W_O: Float32Array;
  /** Projected attention update Attn_out = W_O @ Concat: [8, T], indexed [i * T + q] */
  readonly Attn_out: Float32Array;
  /** Residual merge Y = X + Attn_out: [8, T], indexed [i * T + q] */
  readonly Y: Float32Array;
}

// ============================================================================
// DETERMINISTIC SEEDED PARAMETERS (Mulberry32 + Box-Muller Gaussian)
// ============================================================================

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeGaussian(seed: number): () => number {
  const rng = mulberry32(seed);
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0;
    let v = 0;
    while (u <= 1e-12) u = rng();
    while (v <= 1e-12) v = rng();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    spare = mag * Math.sin(2.0 * Math.PI * v);
    return mag * Math.cos(2.0 * Math.PI * v);
  };
}

interface SimParameters {
  readonly X_full: Float32Array; // [8, 16]
  readonly gamma: Float32Array; // [8]
  readonly W_Q: Float32Array; // [128, 8]
  readonly W_K: Float32Array; // [8, 8]
  readonly W_V: Float32Array; // [8, 8]
  readonly W_O: Float32Array; // [8, 128]
}

function createSimParameters(): SimParameters {
  const { d, nQ, nKv, dHead, qConcatDim, maxTokens } = GQA_SIM_CONFIG;
  const gX = makeGaussian(0x51a7e101);
  const gQ = makeGaussian(0x3f2c9043);
  const gK = makeGaussian(0x7a91d205);
  const gV = makeGaussian(0x28c4b719);
  const gO = makeGaussian(0x9e3779b9);

  // Input hidden state X_full [8, 16]: smooth token-distinct semantic vectors (including slight
  // thematic structure so attention scores exhibit meaningful non-uniform peaks across tokens)
  const X_full = new Float32Array(d * maxTokens);
  for (let t = 0; t < maxTokens; t++) {
    const phase = (t / maxTokens) * Math.PI * 2;
    for (let i = 0; i < d; i++) {
      const base = gX() * 0.72;
      const harmonic = 0.34 * Math.sin(phase * (1 + (i % 3)) + i * 0.85);
      X_full[i * maxTokens + t] = Math.fround(base + harmonic);
    }
  }

  // Deterministic non-unit learned RMSNorm weights near 1.0
  const gamma = new Float32Array([0.92, 1.08, 0.96, 1.05, 0.89, 1.11, 0.97, 1.04]);

  // W_Q [128, 8]: scale chosen so Q · K / sqrt(4) has realistic std ~ 1.1-1.5 across all 32 heads
  const W_Q = new Float32Array(qConcatDim * d);
  for (let h = 0; h < nQ; h++) {
    // Give each head slightly distinct gain & directional bias so Q00..Q31 look clearly unique
    const headGain = 0.72 + ((h * 7) % 11) * 0.032;
    for (let r = 0; r < dHead; r++) {
      const outRow = h * dHead + r;
      for (let j = 0; j < d; j++) {
        W_Q[outRow * d + j] = Math.fround((gQ() * headGain) / Math.sqrt(d * 0.52));
      }
    }
  }

  // W_K [8, 8] & W_V [8, 8] (2 shared KV heads x 4 head_dim)
  const W_K = new Float32Array(nKv * dHead * d);
  const W_V = new Float32Array(nKv * dHead * d);
  for (let g = 0; g < nKv; g++) {
    for (let r = 0; r < dHead; r++) {
      const outRow = g * dHead + r;
      for (let j = 0; j < d; j++) {
        W_K[outRow * d + j] = Math.fround((gK() * 0.82) / Math.sqrt(d * 0.55));
        W_V[outRow * d + j] = Math.fround((gV() * 0.78) / Math.sqrt(d * 0.65));
      }
    }
  }

  // W_O [8, 128]: well-conditioned output projection restoring hidden width 128 -> 8
  const W_O = new Float32Array(d * qConcatDim);
  const woScale = 0.58 / Math.sqrt(qConcatDim);
  for (let i = 0; i < d; i++) {
    for (let c = 0; c < qConcatDim; c++) {
      W_O[i * qConcatDim + c] = Math.fround(gO() * woScale);
    }
  }

  return { X_full, gamma, W_Q, W_K, W_V, W_O };
}

const SIM_PARAMS = createSimParameters();
const simCache = new Map<number, GqaSimResult>();

/**
 * Run (or fetch cached) deterministic Float32 forward pass for `tokens` in [1..16].
 */
export function runGqaSim(tokens: number): GqaSimResult {
  const T = Math.max(1, Math.min(GQA_SIM_CONFIG.maxTokens, Math.round(tokens)));
  const cached = simCache.get(T);
  if (cached) return cached;

  const { d, nQ, nKv, dHead, qConcatDim, maxTokens, rmsEps } = GQA_SIM_CONFIG;
  const { X_full, gamma, W_Q, W_K, W_V, W_O } = SIM_PARAMS;

  // 1. Slice input X [8, T]
  const X = new Float32Array(d * T);
  for (let i = 0; i < d; i++) {
    for (let t = 0; t < T; t++) {
      X[i * T + t] = X_full[i * maxTokens + t];
    }
  }

  // 2. RMSNorm: X_norm [8, T] (unit RMS per token) and U = X_norm * gamma [8, T]
  const X_norm = new Float32Array(d * T);
  const rms = new Float32Array(T);
  const U = new Float32Array(d * T);
  for (let t = 0; t < T; t++) {
    let sumSq = 0;
    for (let i = 0; i < d; i++) {
      const v = X[i * T + t];
      sumSq += v * v;
    }
    const rVal = Math.fround(Math.sqrt(sumSq / d + rmsEps));
    rms[t] = rVal;
    for (let i = 0; i < d; i++) {
      const xn = Math.fround(X[i * T + t] / rVal);
      X_norm[i * T + t] = xn;
      U[i * T + t] = Math.fround(xn * gamma[i]);
    }
  }

  // 3. Linear projections (NoPE, zero linear bias)
  const Q = new Float32Array(nQ * dHead * T);
  for (let outRow = 0; outRow < qConcatDim; outRow++) {
    const wOff = outRow * d;
    const qOff = outRow * T;
    for (let t = 0; t < T; t++) {
      let acc = 0;
      for (let j = 0; j < d; j++) {
        acc += W_Q[wOff + j] * U[j * T + t];
      }
      Q[qOff + t] = Math.fround(acc);
    }
  }

  const K = new Float32Array(nKv * dHead * T);
  const V = new Float32Array(nKv * dHead * T);
  for (let outRow = 0; outRow < nKv * dHead; outRow++) {
    const wOff = outRow * d;
    const kvOff = outRow * T;
    for (let t = 0; t < T; t++) {
      let accK = 0;
      let accV = 0;
      for (let j = 0; j < d; j++) {
        const uVal = U[j * T + t];
        accK += W_K[wOff + j] * uVal;
        accV += W_V[wOff + j] * uVal;
      }
      K[kvOff + t] = Math.fround(accK);
      V[kvOff + t] = Math.fround(accV);
    }
  }

  // 4. Scaled Dot-Product Causal Scores [32, T, T] & Row-Wise Softmax [32, T, T]
  const invSqrtD = 1 / Math.sqrt(dHead); // 1 / 2 = 0.5
  const Scores = new Float32Array(nQ * T * T);
  const Softmax = new Float32Array(nQ * T * T);

  for (let h = 0; h < nQ; h++) {
    const g = h < 16 ? 0 : 1; // Exact 16:1 GQA grouping
    const qHeadOff = h * dHead * T;
    const kHeadOff = g * dHead * T;
    const headScoreOff = h * T * T;

    for (let q = 0; q < T; q++) {
      const rowOff = headScoreOff + q * T;
      let maxScore = -Infinity;

      for (let k = 0; k < T; k++) {
        if (k > q) {
          Scores[rowOff + k] = -Infinity;
        } else {
          let dot = 0;
          for (let r = 0; r < dHead; r++) {
            dot += Q[qHeadOff + r * T + q] * K[kHeadOff + r * T + k];
          }
          const s = Math.fround(dot * invSqrtD);
          Scores[rowOff + k] = s;
          if (s > maxScore) maxScore = s;
        }
      }

      // Numerically stable row-wise softmax over causal keys 0..q
      let sumExp = 0;
      for (let k = 0; k <= q; k++) {
        const e = Math.exp(Scores[rowOff + k] - maxScore);
        Softmax[rowOff + k] = e;
        sumExp += e;
      }
      for (let k = 0; k < T; k++) {
        if (k <= q) {
          Softmax[rowOff + k] = Math.fround(Softmax[rowOff + k] / sumExp);
        } else {
          Softmax[rowOff + k] = 0;
        }
      }
    }
  }

  // 5. Weighted Value Output V_out [32, 4, T] & Concat [128, T]
  const V_out = new Float32Array(nQ * dHead * T);
  const Concat = new Float32Array(qConcatDim * T);

  for (let h = 0; h < nQ; h++) {
    const g = h < 16 ? 0 : 1;
    const vHeadOff = g * dHead * T;
    const headSoftOff = h * T * T;
    const outHeadOff = h * dHead * T;

    for (let q = 0; q < T; q++) {
      const softRowOff = headSoftOff + q * T;
      for (let r = 0; r < dHead; r++) {
        let acc = 0;
        for (let k = 0; k <= q; k++) {
          acc += Softmax[softRowOff + k] * V[vHeadOff + r * T + k];
        }
        const val = Math.fround(acc);
        V_out[outHeadOff + r * T + q] = val;
        Concat[outHeadOff + r * T + q] = val;
      }
    }
  }

  // 6. Output projection W_O [8, 128] @ Concat [128, T] -> Attn_out [8, T] & Residual Y = X + Attn_out
  const Attn_out = new Float32Array(d * T);
  const Y = new Float32Array(d * T);
  for (let i = 0; i < d; i++) {
    const woRowOff = i * qConcatDim;
    for (let q = 0; q < T; q++) {
      let acc = 0;
      for (let c = 0; c < qConcatDim; c++) {
        acc += W_O[woRowOff + c] * Concat[c * T + q];
      }
      const attn = Math.fround(acc);
      Attn_out[i * T + q] = attn;
      Y[i * T + q] = Math.fround(X[i * T + q] + attn);
    }
  }

  const result: GqaSimResult = {
    tokens: T,
    tokenLabels: EDUCATIONAL_TOKENS.slice(0, T),
    d,
    nQ,
    nKv,
    dHead,
    qConcatDim,
    X,
    X_norm,
    rms,
    gamma,
    U,
    W_Q,
    W_K,
    W_V,
    Q,
    K,
    V,
    Scores,
    Softmax,
    V_out,
    Concat,
    W_O,
    Attn_out,
    Y,
  };

  simCache.set(T, result);
  return result;
}

// ============================================================================
// FORMATTING HELPERS & EXACT LIVE FORMULA INSPECTOR
// ============================================================================

export function fmtSigned(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return v < 0 ? '−∞' : '+∞';
  const rounded = Math.abs(v) < 0.5 * Math.pow(10, -digits) ? 0 : v;
  const sign = rounded > 0 ? '+' : rounded < 0 ? '−' : '';
  return `${sign}${Math.abs(rounded).toFixed(digits)}`;
}

export function fmtProb(v: number, digits = 3): string {
  return Math.max(0, v).toFixed(digits);
}

function fmtHead(h: number): string {
  return `Q${String(h).padStart(2, '0')}`;
}

function tokStr(sim: GqaSimResult, idx: number): string {
  const clamped = Math.max(0, Math.min(sim.tokens - 1, idx));
  return `"${sim.tokenLabels[clamped]}"(${clamped})`;
}

/**
 * Clamp a cell reference to valid simulator bounds for the active token count `T`.
 */
export function clampCellToSim(sim: GqaSimResult, cell: GqaCellRef): GqaCellRef {
  const T = sim.tokens;
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
  switch (cell.tensor) {
    case 'input':
    case 'rmsnorm':
    case 'residual':
      return { ...cell, row: clamp(cell.row, 0, T - 1), col: clamp(cell.col, 0, sim.d - 1) };
    case 'w_q':
    case 'w_o':
      return { ...cell, row: clamp(cell.row, 0, sim.d - 1), col: clamp(cell.col, 0, sim.qConcatDim - 1) };
    case 'w_k':
    case 'w_v':
      return { ...cell, row: clamp(cell.row, 0, sim.d - 1), col: clamp(cell.col, 0, sim.nKv * sim.dHead - 1) };
    case 'q': {
      const h = clamp(cell.headIndex ?? 0, 0, sim.nQ - 1);
      return {
        ...cell,
        headIndex: h,
        kvHeadIndex: h < 16 ? 0 : 1,
        row: clamp(cell.row, 0, T - 1),
        col: clamp(cell.col, 0, sim.dHead - 1),
      };
    }
    case 'kv_k':
    case 'kv_v': {
      const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
      const t = clamp(cell.tokenIndex ?? cell.row, 0, T - 1);
      return {
        ...cell,
        kvHeadIndex: g,
        tokenIndex: t,
        row: 0,
        col: clamp(cell.col, 0, sim.dHead - 1),
      };
    }
    case 'scores':
    case 'softmax': {
      const g: 0 | 1 =
        cell.kvHeadIndex !== undefined
          ? cell.kvHeadIndex
          : (cell.headIndex ?? 0) < 16
            ? 0
            : 1;
      const h = clamp(cell.headIndex ?? (g === 0 ? 0 : 16), g * 16, g * 16 + 15);
      return {
        ...cell,
        headIndex: h,
        kvHeadIndex: g,
        row: clamp(cell.row, 0, T - 1),
        col: clamp(cell.col, 0, T - 1),
      };
    }
    case 'v_out': {
      const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
      const col = clamp(cell.col, 0, 16 * sim.dHead - 1);
      const localH = Math.floor(col / sim.dHead);
      const h = g * 16 + localH;
      return {
        ...cell,
        headIndex: h,
        kvHeadIndex: g,
        row: clamp(cell.row, 0, T - 1),
        col,
      };
    }
    case 'concat': {
      const col = clamp(cell.col, 0, sim.qConcatDim - 1);
      const h = Math.floor(col / sim.dHead);
      return {
        ...cell,
        headIndex: h,
        kvHeadIndex: h < 16 ? 0 : 1,
        row: clamp(cell.row, 0, T - 1),
        col,
      };
    }
  }
}

/**
 * Produce an exact live Float32 equation breakdown and upstream sheet dependency spans
 * for any picked or stepper-driven cell in the GQA cutaway.
 */
export function describeCellFormula(sim: GqaSimResult, rawCell: GqaCellRef): GqaCellFormula {
  const cell = clampCellToSim(sim, rawCell);
  const T = sim.tokens;
  const { d, dHead, qConcatDim } = sim;

  switch (cell.tensor) {
    case 'input': {
      const t = cell.row;
      const i = cell.col;
      const val = sim.X[i * T + t];
      return {
        cell,
        tensorLabel: 'Input X',
        coordLabel: `X[ch=${i}, t=${tokStr(sim, t)}]`,
        headBadge: null,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `X[${i}, ${t}] = ${fmtSigned(val)} (incoming residual state before RMSNorm)`,
        subEquation: `Token ${tokStr(sim, t)} · RMS(X[:, ${t}]) = ${sim.rms[t].toFixed(3)} · NoPE (no positional encoding added)`,
        targetSpan: { tensor: 'input', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'rmsnorm': {
      const t = cell.row;
      const i = cell.col;
      const xVal = sim.X[i * T + t];
      const rmsVal = sim.rms[t];
      const xn = sim.X_norm[i * T + t];
      const gVal = sim.gamma[i];
      const uVal = sim.U[i * T + t];
      return {
        cell,
        tensorLabel: 'RMSNorm · U',
        coordLabel: `U[ch=${i}, t=${tokStr(sim, t)}]`,
        headBadge: null,
        value: uVal,
        isMasked: false,
        valueText: fmtSigned(uVal),
        equation: `(X[${i},${t}] / rms) · γ[${i}] = (${fmtSigned(xVal)} / ${rmsVal.toFixed(3)}) · ${gVal.toFixed(2)} = ${fmtSigned(xn)} · ${gVal.toFixed(2)} = ${fmtSigned(uVal)}`,
        subEquation: `Pre-γ unit-RMS X_norm[${i},${t}] = ${fmtSigned(xn)} · rms(X[:,${t}]) = √(mean(x²) + 1e−6) = ${rmsVal.toFixed(3)}`,
        targetSpan: { tensor: 'rmsnorm', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'target' },
        upstreamSpans: [
          { tensor: 'input', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_a' },
        ],
      };
    }

    case 'w_q': {
      const j = cell.row;
      const outRow = cell.col;
      const h = Math.floor(outRow / dHead);
      const r = outRow % dHead;
      const g: 0 | 1 = h < 16 ? 0 : 1;
      const val = sim.W_Q[outRow * d + j];
      return {
        cell,
        tensorLabel: 'W_Q Weight',
        coordLabel: `W_Q[${fmtHead(h)}·ch${r}, in=${j}]`,
        headBadge: `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `W_Q[${outRow}, ${j}] = ${fmtSigned(val)} (projects RMSNorm ch ${j} into ${fmtHead(h)} dim ${r})`,
        subEquation: `Miniature W_Q [128 × 8] · Production W_Q [4,096 × 2,688] (32Q × 128, BF16 §4.2)`,
        targetSpan: { tensor: 'w_q', rowStart: j, rowEnd: j, colStart: outRow, colEnd: outRow, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'w_k':
    case 'w_v': {
      const isK = cell.tensor === 'w_k';
      const j = cell.row;
      const outRow = cell.col;
      const g: 0 | 1 = outRow < dHead ? 0 : 1;
      const r = outRow % dHead;
      const W = isK ? sim.W_K : sim.W_V;
      const val = W[outRow * d + j];
      const name = isK ? 'W_K' : 'W_V';
      return {
        cell,
        tensorLabel: `${name} Weight`,
        coordLabel: `${name}[KV${g}·ch${r}, in=${j}]`,
        headBadge: `Shared KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `${name}[${outRow}, ${j}] = ${fmtSigned(val)} (16× narrower than W_Q; writes shared ${isK ? 'K' : 'V'}${g} dim ${r})`,
        subEquation: `Miniature ${name} [8 × 8] · Production ${name} [256 × 2,688] (2 KV heads × 128)`,
        targetSpan: { tensor: cell.tensor, rowStart: j, rowEnd: j, colStart: outRow, colEnd: outRow, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'q': {
      const h = cell.headIndex ?? 0;
      const g: 0 | 1 = h < 16 ? 0 : 1;
      const t = cell.row;
      const r = cell.col;
      const outRow = h * dHead + r;
      const val = sim.Q[outRow * T + t];
      const terms: string[] = [];
      for (let j = 0; j < 3; j++) {
        terms.push(`${fmtSigned(sim.W_Q[outRow * d + j], 2)}·${fmtSigned(sim.U[j * T + t], 2)}`);
      }
      return {
        cell,
        tensorLabel: `Query ${fmtHead(h)}`,
        coordLabel: `${fmtHead(h)}[t=${tokStr(sim, t)}, d=${r}]`,
        headBadge: `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `Σ_{j=0..7} W_Q[${outRow},j]·U[j,${t}] = ${terms.join(' + ')} + … = ${fmtSigned(val)}`,
        subEquation: `Zero bias · NoPE · ${fmtHead(h)} attends shared KV Head ${g} (Group ${g}: ${g === 0 ? 'Q00–Q15' : 'Q16–Q31'})`,
        targetSpan: {
          tensor: 'q',
          headIndex: h,
          kvHeadIndex: g,
          rowStart: t,
          rowEnd: t,
          colStart: r,
          colEnd: r,
          role: 'target',
        },
        upstreamSpans: [
          { tensor: 'w_q', rowStart: 0, rowEnd: d - 1, colStart: outRow, colEnd: outRow, role: 'operand_a' },
          { tensor: 'rmsnorm', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_b' },
        ],
      };
    }

    case 'kv_k':
    case 'kv_v': {
      const isK = cell.tensor === 'kv_k';
      const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
      const t = cell.tokenIndex ?? 0;
      const r = cell.col;
      const outRow = g * dHead + r;
      const mat = isK ? sim.K : sim.V;
      const W = isK ? sim.W_K : sim.W_V;
      const val = mat[outRow * T + t];
      const terms: string[] = [];
      for (let j = 0; j < 3; j++) {
        terms.push(`${fmtSigned(W[outRow * d + j], 2)}·${fmtSigned(sim.U[j * T + t], 2)}`);
      }
      const name = isK ? `K${g}` : `V${g}`;
      return {
        cell,
        tensorLabel: `Shared ${name} Cache`,
        coordLabel: `${name}[t=${tokStr(sim, t)}, d=${r}]`,
        headBadge: `KV${g} · shared by 16Q`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `Σ_{j=0..7} ${isK ? 'W_K' : 'W_V'}[${outRow},j]·U[j,${t}] = ${terms.join(' + ')} + … = ${fmtSigned(val)}`,
        subEquation: `Shared verbatim across ${g === 0 ? 'Q00–Q15' : 'Q16–Q31'} · 16:1 GQA sharing · Cached at token slot ${t + 1}/${T}`,
        targetSpan: {
          tensor: cell.tensor,
          kvHeadIndex: g,
          tokenStart: t,
          tokenEnd: t,
          rowStart: 0,
          rowEnd: 0,
          colStart: r,
          colEnd: r,
          role: 'target',
        },
        upstreamSpans: [
          {
            tensor: isK ? 'w_k' : 'w_v',
            rowStart: 0,
            rowEnd: d - 1,
            colStart: outRow,
            colEnd: outRow,
            role: 'operand_a',
          },
          { tensor: 'rmsnorm', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_b' },
        ],
      };
    }

    case 'scores': {
      const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
      const h = cell.headIndex ?? (g === 0 ? 0 : 16);
      const q = cell.row;
      const k = cell.col;
      const val = sim.Scores[(h * T + q) * T + k];
      const isMasked = k > q || !Number.isFinite(val);

      if (isMasked) {
        return {
          cell,
          tensorLabel: `Scores · ${fmtHead(h)}`,
          coordLabel: `Scores[${fmtHead(h)}, q=${tokStr(sim, q)}, k=${tokStr(sim, k)}]`,
          headBadge: `${fmtHead(h)} → KV${g}`,
          value: -Infinity,
          isMasked: true,
          valueText: '−∞ (masked)',
          equation: `k=${k} > q=${q} (future token) → Causal Mask sets Score = −∞ → Softmax = 0.000`,
          subEquation: `Autoregressive causality blocks query ${tokStr(sim, q)} from attending future key ${tokStr(sim, k)}`,
          targetSpan: {
            tensor: 'scores',
            headIndex: h,
            kvHeadIndex: g,
            rowStart: q,
            rowEnd: q,
            colStart: k,
            colEnd: k,
            role: 'target',
          },
          upstreamSpans: [],
        };
      }

      const qHeadOff = h * dHead * T;
      const kHeadOff = g * dHead * T;
      const prods: string[] = [];
      let dot = 0;
      for (let r = 0; r < dHead; r++) {
        const qv = sim.Q[qHeadOff + r * T + q];
        const kv = sim.K[kHeadOff + r * T + k];
        dot += qv * kv;
        prods.push(`(${fmtSigned(qv, 2)}·${fmtSigned(kv, 2)})`);
      }
      return {
        cell,
        tensorLabel: `Scores · ${fmtHead(h)}`,
        coordLabel: `Scores[${fmtHead(h)}, q=${tokStr(sim, q)}, k=${tokStr(sim, k)}]`,
        headBadge: `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `(${fmtHead(h)}[:,${q}] · K${g}[:,${k}]) / √4 = [${prods.join(' + ')}] / 2 = ${fmtSigned(dot, 3)} / 2 = ${fmtSigned(val)}`,
        subEquation: `16:1 GQA: ${fmtHead(h)} queries shared K${g} · Post-Softmax weight = ${fmtProb(sim.Softmax[(h * T + q) * T + k])}`,
        targetSpan: {
          tensor: 'scores',
          headIndex: h,
          kvHeadIndex: g,
          rowStart: q,
          rowEnd: q,
          colStart: k,
          colEnd: k,
          role: 'target',
        },
        upstreamSpans: [
          {
            tensor: 'q',
            headIndex: h,
            kvHeadIndex: g,
            rowStart: q,
            rowEnd: q,
            colStart: 0,
            colEnd: dHead - 1,
            role: 'operand_a',
          },
          {
            tensor: 'kv_k',
            kvHeadIndex: g,
            tokenStart: k,
            tokenEnd: k,
            rowStart: 0,
            rowEnd: 0,
            colStart: 0,
            colEnd: dHead - 1,
            role: 'operand_b',
          },
        ],
      };
    }

    case 'softmax': {
      const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
      const h = cell.headIndex ?? (g === 0 ? 0 : 16);
      const q = cell.row;
      const k = cell.col;
      const val = sim.Softmax[(h * T + q) * T + k];
      const scoreVal = sim.Scores[(h * T + q) * T + k];
      const isMasked = k > q;

      if (isMasked) {
        return {
          cell,
          tensorLabel: `Softmax · ${fmtHead(h)}`,
          coordLabel: `Softmax[${fmtHead(h)}, q=${tokStr(sim, q)}, k=${tokStr(sim, k)}]`,
          headBadge: `${fmtHead(h)} → KV${g}`,
          value: 0,
          isMasked: true,
          valueText: '0.000 (masked)',
          equation: `Score = −∞ (k=${k} > q=${q}) → exp(−∞) = 0 → Softmax = 0.000`,
          subEquation: `Upper triangle strictly 0.000 · Valid causal keys for q=${q} are k = 0..${q}`,
          targetSpan: {
            tensor: 'softmax',
            headIndex: h,
            kvHeadIndex: g,
            rowStart: q,
            rowEnd: q,
            colStart: k,
            colEnd: k,
            role: 'target',
          },
          upstreamSpans: [
            {
              tensor: 'scores',
              headIndex: h,
              kvHeadIndex: g,
              rowStart: q,
              rowEnd: q,
              colStart: k,
              colEnd: k,
              role: 'operand_a',
            },
          ],
        };
      }

      let maxS = -Infinity;
      for (let j = 0; j <= q; j++) {
        const s = sim.Scores[(h * T + q) * T + j];
        if (s > maxS) maxS = s;
      }
      let sumExp = 0;
      let rowSum = 0;
      for (let j = 0; j <= q; j++) {
        sumExp += Math.exp(sim.Scores[(h * T + q) * T + j] - maxS);
        rowSum += sim.Softmax[(h * T + q) * T + j];
      }
      const num = Math.exp(scoreVal - maxS);
      return {
        cell,
        tensorLabel: `Softmax · ${fmtHead(h)}`,
        coordLabel: `Softmax[${fmtHead(h)}, q=${tokStr(sim, q)}, k=${tokStr(sim, k)}]`,
        headBadge: `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtProb(val),
        equation: `exp(${fmtSigned(scoreVal)} − ${fmtSigned(maxS)}) / Σ_{j=0..${q}} exp(s_j − max) = ${num.toFixed(3)} / ${sumExp.toFixed(3)} = ${fmtProb(val)}`,
        subEquation: `Causal row ${q} sum Σ_{k=0..${q}} Softmax = ${rowSum.toFixed(4)} (${(val * 100).toFixed(1)}% weight on ${tokStr(sim, k)})`,
        targetSpan: {
          tensor: 'softmax',
          headIndex: h,
          kvHeadIndex: g,
          rowStart: q,
          rowEnd: q,
          colStart: k,
          colEnd: k,
          role: 'target',
        },
        upstreamSpans: [
          {
            tensor: 'scores',
            headIndex: h,
            kvHeadIndex: g,
            rowStart: q,
            rowEnd: q,
            colStart: 0,
            colEnd: q,
            role: 'operand_a',
          },
        ],
      };
    }

    case 'v_out': {
      const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
      const q = cell.row;
      const localH = Math.floor(cell.col / dHead);
      const r = cell.col % dHead;
      const h = g * 16 + localH;
      const val = sim.V_out[(h * dHead + r) * T + q];
      const vHeadOff = g * dHead * T;
      const softRowOff = (h * T + q) * T;
      const terms: string[] = [];
      const showN = Math.min(q + 1, 3);
      for (let k = 0; k < showN; k++) {
        terms.push(`${fmtProb(sim.Softmax[softRowOff + k], 2)}·${fmtSigned(sim.V[vHeadOff + r * T + k], 2)}`);
      }
      const tail = q + 1 > showN ? ` + … (${q + 1} terms)` : '';
      return {
        cell,
        tensorLabel: `V Out · ${fmtHead(h)}`,
        coordLabel: `V_out[${fmtHead(h)}, q=${tokStr(sim, q)}, d=${r}]`,
        headBadge: `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `Σ_{k=0..${q}} Softmax[${fmtHead(h)},${q},k]·V${g}[${r},k] = ${terms.join(' + ')}${tail} = ${fmtSigned(val)}`,
        subEquation: `Weighted blend of cached shared V${g} slices 0..${q} → feeds Concat channel ${h * dHead + r}`,
        targetSpan: {
          tensor: 'v_out',
          headIndex: h,
          kvHeadIndex: g,
          rowStart: q,
          rowEnd: q,
          colStart: cell.col,
          colEnd: cell.col,
          role: 'target',
        },
        upstreamSpans: [
          {
            tensor: 'softmax',
            headIndex: h,
            kvHeadIndex: g,
            rowStart: q,
            rowEnd: q,
            colStart: 0,
            colEnd: q,
            role: 'operand_a',
          },
          {
            tensor: 'kv_v',
            kvHeadIndex: g,
            tokenStart: 0,
            tokenEnd: q,
            rowStart: 0,
            rowEnd: 0,
            colStart: r,
            colEnd: r,
            role: 'operand_b',
          },
        ],
      };
    }

    case 'concat': {
      const q = cell.row;
      const c = cell.col;
      const h = Math.floor(c / dHead);
      const r = c % dHead;
      const g: 0 | 1 = h < 16 ? 0 : 1;
      const localCol = (h % 16) * dHead + r;
      const val = sim.Concat[c * T + q];
      return {
        cell,
        tensorLabel: 'Concat [128]',
        coordLabel: `Concat[ch=${c} (${fmtHead(h)}·d${r}), q=${tokStr(sim, q)}]`,
        headBadge: `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `Concat[${c}, ${q}] = V_out[${fmtHead(h)}, d=${r}, q=${q}] = ${fmtSigned(val)}`,
        subEquation: `32 heads × 4 = 128 channels (mirrors production 32 × 128 = 4,096 ≠ 2,688 before W_O)`,
        targetSpan: { tensor: 'concat', rowStart: q, rowEnd: q, colStart: c, colEnd: c, role: 'target' },
        upstreamSpans: [
          {
            tensor: 'v_out',
            headIndex: h,
            kvHeadIndex: g,
            rowStart: q,
            rowEnd: q,
            colStart: localCol,
            colEnd: localCol,
            role: 'operand_a',
          },
        ],
      };
    }

    case 'w_o': {
      const i = cell.row;
      const c = cell.col;
      const h = Math.floor(c / dHead);
      const r = c % dHead;
      const val = sim.W_O[i * qConcatDim + c];
      return {
        cell,
        tensorLabel: 'W_O Weight',
        coordLabel: `W_O[out=${i}, in=${c} (${fmtHead(h)}·d${r})]`,
        headBadge: `${fmtHead(h)} → ch${i}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `W_O[${i}, ${c}] = ${fmtSigned(val)} (projects ${fmtHead(h)}·d${r} into hidden channel ${i})`,
        subEquation: `Miniature W_O [8 × 128] restores width 128 → 8 · Production W_O [2,688 × 4,096]`,
        targetSpan: { tensor: 'w_o', rowStart: i, rowEnd: i, colStart: c, colEnd: c, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'residual': {
      const t = cell.row;
      const i = cell.col;
      const xVal = sim.X[i * T + t];
      const attnVal = sim.Attn_out[i * T + t];
      const yVal = sim.Y[i * T + t];
      return {
        cell,
        tensorLabel: '+ Residual · Y',
        coordLabel: `Y[ch=${i}, t=${tokStr(sim, t)}]`,
        headBadge: null,
        value: yVal,
        isMasked: false,
        valueText: fmtSigned(yVal),
        equation: `Y[${i},${t}] = X[${i},${t}] + (W_O·Concat)[${i},${t}] = ${fmtSigned(xVal)} + (${fmtSigned(attnVal)}) = ${fmtSigned(yVal)}`,
        subEquation: `Attn_out[${i},${t}] = Σ_{c=0..127} W_O[${i},c]·Concat[c,${t}] = ${fmtSigned(attnVal)}`,
        targetSpan: { tensor: 'residual', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'target' },
        upstreamSpans: [
          { tensor: 'input', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'operand_a' },
          { tensor: 'w_o', rowStart: i, rowEnd: i, colStart: 0, colEnd: qConcatDim - 1, role: 'operand_b' },
          { tensor: 'concat', rowStart: t, rowEnd: t, colStart: 0, colEnd: qConcatDim - 1, role: 'operand_b' },
        ],
      };
    }
  }
}

// ============================================================================
// CAUSAL ATTENTION 6-STEP ANIMATION MAPPING
// ============================================================================

export interface GqaStepperStepDef {
  readonly index: number;
  readonly shortLabel: string;
  readonly title: string;
  readonly summary: string;
}

export const GQA_STEPPER_STEPS: readonly GqaStepperStepDef[] = [
  {
    index: 0,
    shortLabel: '1. RMSNorm',
    title: 'Pre-Attention RMSNorm (NoPE)',
    summary: 'Normalize Input X[:, t] to unit RMS and scale by γ → U[:, t].',
  },
  {
    index: 1,
    shortLabel: '2. Q/K/V',
    title: 'Q / K / V Linear Projections',
    summary: 'Project U[:, t] through W_Q into Q_i and 16× narrower W_K, W_V into shared KV_g.',
  },
  {
    index: 2,
    shortLabel: '3. QKᵀ/√d',
    title: 'Scaled Dot-Product Scores (16:1 GQA)',
    summary: 'Sweep query Q_i[:, q] across shared keys K_g[:, 0..q] (masking k > q to −∞).',
  },
  {
    index: 3,
    shortLabel: '4. Softmax',
    title: 'Causal Mask & Row-Wise Softmax',
    summary: 'Normalize causal scores in row q into probabilities summing to 1.000.',
  },
  {
    index: 4,
    shortLabel: '5. Weighted V',
    title: 'Weighted Shared Value Sum',
    summary: 'Blend shared V_g slices 0..q with Softmax row q into V_out[:, q].',
  },
  {
    index: 5,
    shortLabel: '6. W_O + Res',
    title: 'Concat [128] → W_O [128→8] → + Residual',
    summary: 'Project all 32 concatenated heads back to d=8 and merge onto Input X.',
  },
] as const;

/**
 * Given the active head `0..31`, step index `0..5`, and sub-progress `0..1`,
 * return the exact live Float32 cell highlighted by the causal attention stepper.
 */
export function stepperCellForState(
  sim: GqaSimResult,
  activeHead: number,
  step: number,
  subProgress: number,
): GqaCellRef {
  const h = Math.max(0, Math.min(31, Math.round(activeHead)));
  const g: 0 | 1 = h < 16 ? 0 : 1;
  const T = sim.tokens;
  const p = Math.max(0, Math.min(0.9999, subProgress));
  const lastTok = T - 1;

  switch (Math.max(0, Math.min(5, Math.round(step)))) {
    case 0: {
      // RMSNorm: sweep channels 0..7 on the latest token (or across tokens)
      const ch = Math.floor(p * sim.d);
      return { tensor: 'rmsnorm', row: lastTok, col: ch };
    }
    case 1: {
      // Q/K/V projection: sweep dHead=4 channels of Q_h on the latest token
      const r = Math.floor(p * sim.dHead);
      return { tensor: 'q', headIndex: h, kvHeadIndex: g, row: lastTok, col: r };
    }
    case 2: {
      // QK^T / sqrt(d): sweep key k = 0..lastTok on causal row q = lastTok (with a brief peek at masked if T>1)
      const k = Math.floor(p * T);
      return { tensor: 'scores', headIndex: h, kvHeadIndex: g, row: lastTok, col: k };
    }
    case 3: {
      // Causal Softmax: sweep causal probabilities k = 0..lastTok along row q = lastTok
      const k = Math.floor(p * T);
      return { tensor: 'softmax', headIndex: h, kvHeadIndex: g, row: lastTok, col: k };
    }
    case 4: {
      // Weighted V: sweep head h's 4 channels in V_out bank
      const r = Math.floor(p * sim.dHead);
      const localCol = (h % 16) * sim.dHead + r;
      return { tensor: 'v_out', headIndex: h, kvHeadIndex: g, row: lastTok, col: localCol };
    }
    default: {
      // Concat -> W_O -> + Residual: sweep output channels 0..7 on latest token
      const ch = Math.floor(p * sim.d);
      return { tensor: 'residual', row: lastTok, col: ch };
    }
  }
}
