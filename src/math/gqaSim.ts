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

/**
 * Channel-wise affine AdaLN modulation terms (`shift`, `scale`), each length `d`.
 * Applied as `x * (1 + scale) + shift` in IEEE-754 Float32:
 *   - Pre-RMSNorm for Mamba-2 (`M`) and GQA Attention (`*`) layers
 *   - Post-RMSNorm for Granular MoE (`E`) layers
 */
export interface LayerModulationTerms {
  readonly shift: Float32Array;
  readonly scale: Float32Array;
}

/**
 * Attention visibility mask policy shared across GQA forward pass, formula inspection,
 * and focused-cell stepper accumulation:
 *   - `'causal'`: Nano AR & TwoTower Context Tower (`kGlobal <= prefixLen + qLocal`)
 *   - `'bidirectional_block'`: TwoTower Denoiser Tower (`0 <= kGlobal < keyTokens`)
 */
export type AttentionMaskMode = 'causal' | 'bidirectional_block';

/**
 * Authoritative key visibility predicate shared across GQA score calculation, softmax,
 * weighted V accumulation, `describeCellFormula`, and `computeGqaFocusedAccumulation`.
 */
export function isAttentionKeyVisible(
  mode: AttentionMaskMode,
  prefixLen: number,
  qLocal: number,
  kGlobal: number,
  keyTokens: number,
): boolean {
  if (kGlobal < 0 || kGlobal >= keyTokens) return false;
  if (mode === 'bidirectional_block') return true;
  return kGlobal <= prefixLen + qLocal;
}

/**
 * Inclusive maximum visible key index `kGlobal` for query `qLocal` under `mode`.
 * Returns `-1` when no keys are visible.
 */
export function maxVisibleKeyIndex(
  mode: AttentionMaskMode,
  prefixLen: number,
  qLocal: number,
  keyTokens: number,
): number {
  if (keyTokens <= 0) return -1;
  if (mode === 'bidirectional_block') return keyTokens - 1;
  return Math.max(-1, Math.min(keyTokens - 1, prefixLen + qLocal));
}

/** Validate channel-wise modulation terms (`shift` and `scale` length `d`, all finite). */
export function validateLayerModulation(
  mod: LayerModulationTerms,
  d: number,
  label: string,
): void {
  if (mod.shift.length !== d) {
    throw new RangeError(`${label}.shift length (${mod.shift.length}) !== expected ${d}`);
  }
  if (mod.scale.length !== d) {
    throw new RangeError(`${label}.scale length (${mod.scale.length}) !== expected ${d}`);
  }
  for (let i = 0; i < d; i++) {
    if (!Number.isFinite(mod.shift[i])) {
      throw new RangeError(`Non-finite value (${mod.shift[i]}) in ${label}.shift[${i}]`);
    }
    if (!Number.isFinite(mod.scale[i])) {
      throw new RangeError(`Non-finite value (${mod.scale[i]}) in ${label}.scale[${i}]`);
    }
  }
}

/** Validate channel-wise residual gate (`length === d`, all finite). */
export function validateResidualGate(gate: Float32Array, d: number, label: string): void {
  if (gate.length !== d) {
    throw new RangeError(`${label} length (${gate.length}) !== expected ${d}`);
  }
  for (let i = 0; i < d; i++) {
    if (!Number.isFinite(gate[i])) {
      throw new RangeError(`Non-finite value (${gate[i]}) in ${label}[${i}]`);
    }
  }
}

/** Format an RMSNorm epsilon concisely with typographic minus (e.g. `1e-5` -> `1e−5`). */
export function formatEpsLabel(eps: number): string {
  if (Math.abs(eps - 1e-5) <= 1e-12) return '1e−5';
  if (Math.abs(eps - 1e-6) <= 1e-13) return '1e−6';
  return eps.toExponential().replace('e-', 'e−').replace('e+', 'e+');
}

/**
 * Resolve token IDs and matching display labels for any miniature simulator invocation.
 * Custom token IDs never silently inherit unrelated hardcoded educational labels.
 */
export function resolveSimTokens(
  tokens: number,
  fixtureTokenIds?: readonly number[] | Int32Array,
  fixtureTokenLabels?: readonly string[],
): { readonly tokenIds: Int32Array; readonly tokenLabels: readonly string[] } {
  const tokenIds = new Int32Array(tokens);
  if (fixtureTokenIds) {
    for (let t = 0; t < tokens; t++) {
      tokenIds[t] = fixtureTokenIds[t] ?? t;
    }
    const tokenLabels: string[] = [];
    for (let t = 0; t < tokens; t++) {
      tokenLabels.push(fixtureTokenLabels?.[t] ?? `id:${tokenIds[t]}`);
    }
    return { tokenIds, tokenLabels };
  }
  for (let t = 0; t < tokens; t++) {
    tokenIds[t] = t;
  }
  if (fixtureTokenLabels) {
    const tokenLabels: string[] = [];
    for (let t = 0; t < tokens; t++) {
      tokenLabels.push(fixtureTokenLabels[t] ?? `t${t}`);
    }
    return { tokenIds, tokenLabels };
  }
  return { tokenIds, tokenLabels: EDUCATIONAL_TOKENS.slice(0, tokens) };
}

/**
 * Optional prefix KV cache input for cached incremental GQA attention (`[nQ, S, P+S]`).
 * Positions `0..seqLen-1` are reused verbatim without recomputing `W_K` or `W_V`.
 */
export interface GqaSimKvCacheInput {
  /** Number of populated prefix tokens `P` (`0..maxTokens`). */
  readonly seqLen: number;
  /** Sequence capacity stride used to index `K` and `V`: `[(g * dHead + r) * maxTokens + k]`. */
  readonly maxTokens: number;
  /** Cached prefix keys `[nKv * dHead, maxTokens]`. */
  readonly K: Float32Array;
  /** Cached prefix values `[nKv * dHead, maxTokens]`. */
  readonly V: Float32Array;
  /** Optional prefix token IDs (length `seqLen`) for truthful key coordinate inspection. */
  readonly prefixTokenIds?: readonly number[] | Int32Array;
  /** Optional prefix token labels (length `seqLen`) for truthful key coordinate inspection. */
  readonly prefixTokenLabels?: readonly string[];
}

export interface GqaSimFixture {
  /** Input hidden state X: [d, T], indexed [i * T + t]. */
  readonly X?: Float32Array;
  /** Learned RMSNorm scale gamma: [d]. */
  readonly gamma?: Float32Array;
  /** Query projection W_Q: [qConcatDim, d]. */
  readonly W_Q?: Float32Array;
  /** Key projection W_K: [nKv * dHead, d]. */
  readonly W_K?: Float32Array;
  /** Value projection W_V: [nKv * dHead, d]. */
  readonly W_V?: Float32Array;
  /** Output projection W_O: [d, qConcatDim]. */
  readonly W_O?: Float32Array;
  /** Override RMSNorm epsilon (e.g. 1e-5 in unified 52-layer forward pass). */
  readonly rmsEps?: number;
  /** Actual input token IDs for this execution (length T). */
  readonly tokenIds?: readonly number[] | Int32Array;
  /** Matching display labels for each token position (length T). */
  readonly tokenLabels?: readonly string[];
  /** Optional cached prefix KV state for true rectangular `[nQ, S, P+S]` incremental attention. */
  readonly kvCacheIn?: GqaSimKvCacheInput;
  /** Attention key visibility policy (defaults to `'causal'`). */
  readonly attentionMaskMode?: AttentionMaskMode;
  /** Optional pre-RMSNorm AdaLN modulation (`X_mod = X * (1 + scale) + shift`, then RMSNorm). */
  readonly preNormMod?: LayerModulationTerms;
  /** Optional channel-wise residual gate (`Y = X + gate * Attn_out`, preserving unmodulated `X`). */
  readonly residualGate?: Float32Array;
}

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

/**
 * Resolve the exact query head index (0..31) owned by a GQA cell reference,
 * including multi-head sheets (v_out, concat, w_q, w_o) where the head is
 * encoded in the column index. Returns null for head-agnostic tensors.
 */
export function inferGqaCellHead(cell: GqaCellRef | null | undefined): number | null {
  if (!cell) return null;
  if (cell.headIndex !== undefined && Number.isInteger(cell.headIndex)) {
    return Math.max(0, Math.min(GQA_SIM_CONFIG.nQ - 1, cell.headIndex));
  }
  const { dHead, nQ } = GQA_SIM_CONFIG;
  if (cell.tensor === 'v_out') {
    const g = cell.kvHeadIndex === 1 ? 1 : 0;
    const localH = Math.max(0, Math.min(15, Math.floor(cell.col / dHead)));
    return g * 16 + localH;
  }
  if (cell.tensor === 'concat' || cell.tensor === 'w_q' || cell.tensor === 'w_o') {
    return Math.max(0, Math.min(nQ - 1, Math.floor(cell.col / dHead)));
  }
  return null;
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
  /** Number of newly processed query tokens `S` (`=== keyTokens` when `prefixTokens === 0`). */
  readonly tokens: number;
  /** Number of reused cached prefix KV tokens `P` (`0` in standalone / full-sequence runs). */
  readonly prefixTokens: number;
  /** Total attended KV context length `P + S` (`=== tokens` when `prefixTokens === 0`). */
  readonly keyTokens: number;
  /** Absolute starting position offset (`=== prefixTokens`) for newly processed query tokens. */
  readonly startTokenOffset: number;
  readonly tokenIds: Int32Array;
  readonly tokenLabels: readonly string[];
  /** Key/value token IDs across `0..keyTokens-1` (`=== tokenIds` when `prefixTokens === 0`). */
  readonly keyTokenIds: Int32Array;
  /** Key/value display labels across `0..keyTokens-1` (`=== tokenLabels` when `prefixTokens === 0`). */
  readonly keyTokenLabels: readonly string[];
  readonly d: number;
  readonly nQ: number;
  readonly nKv: number;
  readonly dHead: number;
  readonly qConcatDim: number;
  /** Effective RMSNorm epsilon used for this simulation (1e-6 standalone default; 1e-5 unified). */
  readonly rmsEps: number;
  /** Effective attention visibility mask policy (`'causal'` or `'bidirectional_block'`). */
  readonly attentionMaskMode: AttentionMaskMode;
  /** Input hidden state X: [8, S], indexed [i * S + t] (always untouched original residual). */
  readonly X: Float32Array;
  /** Optional pre-RMSNorm AdaLN modulated input X_mod = X * (1 + scale) + shift: [8, S]. */
  readonly X_mod?: Float32Array;
  /** Pre-gamma unit-RMS normalized state X_norm: [8, S], indexed [i * S + t] */
  readonly X_norm: Float32Array;
  /** Per-token RMS denominators: [S] */
  readonly rms: Float32Array;
  /** Non-unit learned RMSNorm scale gamma: [8] */
  readonly gamma: Float32Array;
  /** Scaled RMSNorm output U = X_norm * gamma: [8, S], indexed [i * S + t] */
  readonly U: Float32Array;
  /** W_Q: [128, 8], indexed [outRow * 8 + j] where outRow = h * 4 + r */
  readonly W_Q: Float32Array;
  /** W_K: [8, 8], indexed [outRow * 8 + j] where outRow = g * 4 + r */
  readonly W_K: Float32Array;
  /** W_V: [8, 8], indexed [outRow * 8 + j] where outRow = g * 4 + r */
  readonly W_V: Float32Array;
  /** Q: [32, 4, S], indexed [(h * 4 + r) * S + t] */
  readonly Q: Float32Array;
  /** K: [2, 4, P+S], indexed [(g * 4 + r) * keyTokens + k] (0..P-1 reused prefix, P..P+S-1 new) */
  readonly K: Float32Array;
  /** V: [2, 4, P+S], indexed [(g * 4 + r) * keyTokens + k] (0..P-1 reused prefix, P..P+S-1 new) */
  readonly V: Float32Array;
  /** Causal scaled attention scores: [32, S, P+S], indexed [(h * S + q) * keyTokens + k]; -Infinity for k > P + q */
  readonly Scores: Float32Array;
  /** Row-wise causal softmax: [32, S, P+S], indexed [(h * S + q) * keyTokens + k]; 0 for k > P + q */
  readonly Softmax: Float32Array;
  /** Per-head weighted value outputs: [32, 4, S], indexed [(h * 4 + r) * S + q] */
  readonly V_out: Float32Array;
  /** Concatenated 32 heads: [128, S], indexed [c * S + q] where c = h * 4 + r */
  readonly Concat: Float32Array;
  /** Output projection W_O: [8, 128], indexed [i * 128 + c] */
  readonly W_O: Float32Array;
  /** Projected attention update Attn_out = W_O @ Concat: [8, S], indexed [i * S + q] */
  readonly Attn_out: Float32Array;
  /** Defensive copy of channel-wise residual gate [8], if supplied. */
  readonly residualGate?: Float32Array;
  /** True when a channel-wise residual gate modulated `Attn_out` before adding to `X`. */
  readonly residualGateApplied: boolean;
  /** Residual merge Y = X + (gate ? gate * Attn_out : Attn_out): [8, S], indexed [i * S + q] */
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
 * Default calls (no fixture) are cached per T with standalone defaults (`rmsEps = 1e-6`).
 * Fixture calls accept external `X`, weights, `rmsEps`, `tokenIds`, and `tokenLabels` without polluting the standalone cache.
 */
export function runGqaSim(tokens: number, fixture?: GqaSimFixture): GqaSimResult {
  const { d, nQ, nKv, dHead, qConcatDim, maxTokens } = GQA_SIM_CONFIG;
  const kvIn = fixture?.kvCacheIn;
  const rawT = Math.round(tokens);
  if (fixture && (!Number.isFinite(tokens) || rawT < 1 || rawT > maxTokens)) {
    throw new RangeError(
      `GQA query length S (${tokens}) must be in [1..${maxTokens}] (independent of prefixLen and kvCacheIn.maxTokens)`,
    );
  }
  const T = Math.max(1, Math.min(maxTokens, rawT));
  if (!fixture) {
    const cached = simCache.get(T);
    if (cached) return cached;
  }

  const attentionMaskMode: AttentionMaskMode = fixture?.attentionMaskMode ?? 'causal';
  const preNormMod = fixture?.preNormMod;
  if (preNormMod) {
    validateLayerModulation(preNormMod, d, 'GQA preNormMod');
  }
  let residualGate: Float32Array | undefined;
  if (fixture?.residualGate) {
    validateResidualGate(fixture.residualGate, d, 'GQA residualGate');
    residualGate = new Float32Array(fixture.residualGate);
  }
  const residualGateApplied = residualGate !== undefined;

  let P = 0;
  if (kvIn) {
    if (!Number.isInteger(kvIn.seqLen) || kvIn.seqLen < 0) {
      throw new RangeError(`Invalid kvCacheIn.seqLen: ${kvIn.seqLen}`);
    }
    if (!Number.isInteger(kvIn.maxTokens) || kvIn.maxTokens < kvIn.seqLen) {
      throw new RangeError(
        `Invalid kvCacheIn.maxTokens (${kvIn.maxTokens}) for seqLen (${kvIn.seqLen})`,
      );
    }
    if (kvIn.seqLen + T > kvIn.maxTokens) {
      throw new RangeError(
        `Incremental GQA step (${kvIn.seqLen} + ${T}) exceeds kvCacheIn.maxTokens (${kvIn.maxTokens})`,
      );
    }
    const minKvLen = nKv * dHead * kvIn.maxTokens;
    if (kvIn.K.length < minKvLen || kvIn.V.length < minKvLen) {
      throw new RangeError(
        `kvCacheIn.K/V length (${kvIn.K.length}/${kvIn.V.length}) < expected ${minKvLen}`,
      );
    }
    P = kvIn.seqLen;
  }

  const keyTokens = P + T;
  const rmsEps = fixture?.rmsEps ?? GQA_SIM_CONFIG.rmsEps;
  const gamma = fixture?.gamma ?? SIM_PARAMS.gamma;
  const W_Q = fixture?.W_Q ?? SIM_PARAMS.W_Q;
  const W_K = fixture?.W_K ?? SIM_PARAMS.W_K;
  const W_V = fixture?.W_V ?? SIM_PARAMS.W_V;
  const W_O = fixture?.W_O ?? SIM_PARAMS.W_O;
  const { tokenIds, tokenLabels } = resolveSimTokens(T, fixture?.tokenIds, fixture?.tokenLabels);

  let keyTokenIds: Int32Array;
  let keyTokenLabels: readonly string[];
  if (P === 0) {
    keyTokenIds = tokenIds;
    keyTokenLabels = tokenLabels;
  } else {
    keyTokenIds = new Int32Array(keyTokens);
    const labels: string[] = [];
    for (let k = 0; k < P; k++) {
      const pid = kvIn?.prefixTokenIds?.[k] ?? -1;
      keyTokenIds[k] = pid;
      labels.push(kvIn?.prefixTokenLabels?.[k] ?? (pid >= 0 ? `id:${pid}` : `cache#${k}`));
    }
    for (let t = 0; t < T; t++) {
      keyTokenIds[P + t] = tokenIds[t];
      labels.push(tokenLabels[t]);
    }
    keyTokenLabels = labels;
  }

  // 1. Slice or copy input X [8, T]
  const X = new Float32Array(d * T);
  if (fixture?.X) {
    X.set(fixture.X.subarray(0, d * T));
  } else {
    for (let i = 0; i < d; i++) {
      for (let t = 0; t < T; t++) {
        X[i * T + t] = SIM_PARAMS.X_full[i * maxTokens + ((P + t) % maxTokens)];
      }
    }
  }

  // 2. Optional Pre-Norm AdaLN Modulation (modulate FIRST, then RMSNorm) -> X_norm [8, T] & U = X_norm * gamma [8, T]
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
  const X_norm = new Float32Array(d * T);
  const rms = new Float32Array(T);
  const U = new Float32Array(d * T);
  for (let t = 0; t < T; t++) {
    let sumSq = 0;
    for (let i = 0; i < d; i++) {
      const v = normInput[i * T + t];
      sumSq += v * v;
    }
    const rVal = Math.fround(Math.sqrt(sumSq / d + rmsEps));
    rms[t] = rVal;
    for (let i = 0; i < d; i++) {
      const xn = Math.fround(normInput[i * T + t] / rVal);
      X_norm[i * T + t] = xn;
      U[i * T + t] = Math.fround(xn * gamma[i]);
    }
  }

  // 3. Linear projections (NoPE, zero linear bias)
  //    Query Q [128, T] is computed ONLY for the T newly arriving tokens.
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

  //    K, V [8, P+T]: copy reused prefix columns 0..P-1 verbatim from kvCacheIn; compute ONLY P..P+T-1.
  const K = new Float32Array(nKv * dHead * keyTokens);
  const V = new Float32Array(nKv * dHead * keyTokens);
  for (let outRow = 0; outRow < nKv * dHead; outRow++) {
    const kvOff = outRow * keyTokens;
    if (kvIn && P > 0) {
      const inOff = outRow * kvIn.maxTokens;
      for (let k = 0; k < P; k++) {
        const kvVal = kvIn.K[inOff + k];
        const vvVal = kvIn.V[inOff + k];
        if (!Number.isFinite(kvVal) || !Number.isFinite(vvVal)) {
          throw new RangeError(`Non-finite prefix KV value at row=${outRow}, k=${k}`);
        }
        K[kvOff + k] = kvVal;
        V[kvOff + k] = vvVal;
      }
    }
    const wOff = outRow * d;
    for (let t = 0; t < T; t++) {
      let accK = 0;
      let accV = 0;
      for (let j = 0; j < d; j++) {
        const uVal = U[j * T + t];
        accK += W_K[wOff + j] * uVal;
        accV += W_V[wOff + j] * uVal;
      }
      K[kvOff + P + t] = Math.fround(accK);
      V[kvOff + P + t] = Math.fround(accV);
    }
  }

  // 4. Scaled Dot-Product Scores [32, T, P+T] & Row-Wise Softmax [32, T, P+T] under authoritative visibility policy
  const invSqrtD = 1 / Math.sqrt(dHead); // 1 / 2 = 0.5
  const Scores = new Float32Array(nQ * T * keyTokens);
  const Softmax = new Float32Array(nQ * T * keyTokens);

  for (let h = 0; h < nQ; h++) {
    const g = h < 16 ? 0 : 1; // Exact 16:1 GQA grouping
    const qHeadOff = h * dHead * T;
    const kHeadOff = g * dHead * keyTokens;
    const headScoreOff = h * T * keyTokens;

    for (let q = 0; q < T; q++) {
      const rowOff = headScoreOff + q * keyTokens;
      const maxVisK = maxVisibleKeyIndex(attentionMaskMode, P, q, keyTokens);
      let maxScore = -Infinity;

      for (let k = 0; k < keyTokens; k++) {
        if (!isAttentionKeyVisible(attentionMaskMode, P, q, k, keyTokens)) {
          Scores[rowOff + k] = -Infinity;
        } else {
          let dot = 0;
          for (let r = 0; r < dHead; r++) {
            dot += Q[qHeadOff + r * T + q] * K[kHeadOff + r * keyTokens + k];
          }
          const s = Math.fround(dot * invSqrtD);
          Scores[rowOff + k] = s;
          if (s > maxScore) maxScore = s;
        }
      }

      // Numerically stable row-wise softmax strictly over visible keys
      let sumExp = 0;
      for (let k = 0; k <= maxVisK; k++) {
        if (!isAttentionKeyVisible(attentionMaskMode, P, q, k, keyTokens)) continue;
        const e = Math.exp(Scores[rowOff + k] - maxScore);
        Softmax[rowOff + k] = e;
        sumExp += e;
      }
      for (let k = 0; k < keyTokens; k++) {
        if (isAttentionKeyVisible(attentionMaskMode, P, q, k, keyTokens)) {
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
    const vHeadOff = g * dHead * keyTokens;
    const headSoftOff = h * T * keyTokens;
    const outHeadOff = h * dHead * T;

    for (let q = 0; q < T; q++) {
      const softRowOff = headSoftOff + q * keyTokens;
      const maxVisK = maxVisibleKeyIndex(attentionMaskMode, P, q, keyTokens);
      for (let r = 0; r < dHead; r++) {
        let acc = 0;
        for (let k = 0; k <= maxVisK; k++) {
          if (!isAttentionKeyVisible(attentionMaskMode, P, q, k, keyTokens)) continue;
          acc += Softmax[softRowOff + k] * V[vHeadOff + r * keyTokens + k];
        }
        const val = Math.fround(acc);
        V_out[outHeadOff + r * T + q] = val;
        Concat[outHeadOff + r * T + q] = val;
      }
    }
  }

  // 6. Output projection W_O [8, 128] @ Concat [128, T] -> Attn_out [8, T] & Residual Y = X + (gate ? gate * Attn_out : Attn_out)
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
      const delta = residualGate ? Math.fround(residualGate[i] * attn) : attn;
      Y[i * T + q] = Math.fround(X[i * T + q] + delta);
    }
  }

  const result: GqaSimResult = {
    tokens: T,
    prefixTokens: P,
    keyTokens,
    startTokenOffset: P,
    tokenIds,
    tokenLabels,
    keyTokenIds,
    keyTokenLabels,
    d,
    nQ,
    nKv,
    dHead,
    qConcatDim,
    rmsEps,
    attentionMaskMode,
    X,
    ...(X_mod ? { X_mod } : {}),
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
    ...(residualGate ? { residualGate } : {}),
    residualGateApplied,
    Y,
  };

  if (!fixture) {
    simCache.set(T, result);
  }
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
  return `"${sim.tokenLabels[clamped]}"(${sim.startTokenOffset + clamped})`;
}

function keyTokStr(sim: GqaSimResult, kIdx: number): string {
  const clamped = Math.max(0, Math.min(sim.keyTokens - 1, kIdx));
  return `"${sim.keyTokenLabels[clamped]}"(${clamped})`;
}

/**
 * Clamp a cell reference to valid simulator bounds for the active query length `S` and KV length `P+S`.
 */
export function clampCellToSim(sim: GqaSimResult, cell: GqaCellRef): GqaCellRef {
  const T = sim.tokens;
  const K = sim.keyTokens;
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
      const k = clamp(cell.tokenIndex ?? cell.row, 0, K - 1);
      return {
        ...cell,
        kvHeadIndex: g,
        tokenIndex: k,
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
        col: clamp(cell.col, 0, K - 1),
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
  const K = sim.keyTokens;
  const P = sim.prefixTokens;
  const { d, dHead, qConcatDim } = sim;

  switch (cell.tensor) {
    case 'input': {
      const t = cell.row;
      const tAbs = P + t;
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
        equation: `X[${i}, ${tAbs}] = ${fmtSigned(val)} (incoming residual state before RMSNorm)`,
        subEquation: `Token ${tokStr(sim, t)} · RMS(X[:, ${tAbs}]) = ${sim.rms[t].toFixed(3)} · NoPE (no positional encoding added)`,
        targetSpan: { tensor: 'input', rowStart: t, rowEnd: t, colStart: i, colEnd: i, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'rmsnorm': {
      const t = cell.row;
      const tAbs = P + t;
      const i = cell.col;
      const normIn = sim.X_mod ?? sim.X;
      const inSym = sim.X_mod ? 'X_mod' : 'X';
      const xVal = normIn[i * T + t];
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
        equation: `(${inSym}[${i},${tAbs}] / rms) · γ[${i}] = (${fmtSigned(xVal)} / ${rmsVal.toFixed(3)}) · ${gVal.toFixed(2)} = ${fmtSigned(xn)} · ${gVal.toFixed(2)} = ${fmtSigned(uVal)}`,
        subEquation: `Pre-γ unit-RMS X_norm[${i},${tAbs}] = ${fmtSigned(xn)} · rms(${inSym}[:,${tAbs}]) = √(mean(x²) + ${formatEpsLabel(sim.rmsEps)}) = ${rmsVal.toFixed(3)}`,
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
      const tAbs = P + t;
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
        equation: `Σ_{j=0..7} W_Q[${outRow},j]·U[j,${tAbs}] = ${terms.join(' + ')} + … = ${fmtSigned(val)}`,
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
      const k = cell.tokenIndex ?? 0;
      const r = cell.col;
      const outRow = g * dHead + r;
      const mat = isK ? sim.K : sim.V;
      const W = isK ? sim.W_K : sim.W_V;
      const val = mat[outRow * K + k];
      const name = isK ? `K${g}` : `V${g}`;

      // Reused prefix KV slice (0..P-1): loaded verbatim from prior KV cache, zero W_K/W_V compute this step
      if (k < P) {
        return {
          cell,
          tensorLabel: `Reused ${name} Cache`,
          coordLabel: `${name}[k=${keyTokStr(sim, k)}, d=${r}]`,
          headBadge: `KV${g} · Reused Prefix Cache (0 compute)`,
          value: val,
          isMasked: false,
          valueText: fmtSigned(val),
          equation: `${name}[${outRow}, k=${k}] = ${fmtSigned(val)}  (loaded from prefix KV cache; ${isK ? 'W_K' : 'W_V'} not recomputed)`,
          subEquation: `Reused prefix slot ${k + 1}/${K} (${P} cached + ${T} new) · Shared across ${g === 0 ? 'Q00–Q15' : 'Q16–Q31'} (16:1 GQA)`,
          targetSpan: {
            tensor: cell.tensor,
            kvHeadIndex: g,
            tokenStart: k,
            tokenEnd: k,
            rowStart: 0,
            rowEnd: 0,
            colStart: r,
            colEnd: r,
            role: 'target',
          },
          upstreamSpans: [],
        };
      }

      const tLocal = k - P;
      const terms: string[] = [];
      for (let j = 0; j < 3; j++) {
        terms.push(`${fmtSigned(W[outRow * d + j], 2)}·${fmtSigned(sim.U[j * T + tLocal], 2)}`);
      }
      return {
        cell,
        tensorLabel: P > 0 ? `New ${name} Append` : `Shared ${name} Cache`,
        coordLabel: `${name}[t=${keyTokStr(sim, k)}, d=${r}]`,
        headBadge: P > 0 ? `KV${g} · Newly Computed (+${T} slot${T > 1 ? 's' : ''})` : `KV${g} · shared by 16Q`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `Σ_{j=0..7} ${isK ? 'W_K' : 'W_V'}[${outRow},j]·U[j,${k}] = ${terms.join(' + ')} + … = ${fmtSigned(val)}`,
        subEquation:
          P > 0
            ? `Newly appended slot ${k + 1}/${K} (${P} reused prefix + ${T} new) · Shared verbatim across ${g === 0 ? 'Q00–Q15' : 'Q16–Q31'}`
            : `Shared verbatim across ${g === 0 ? 'Q00–Q15' : 'Q16–Q31'} · 16:1 GQA sharing · Cached at token slot ${k + 1}/${K}`,
        targetSpan: {
          tensor: cell.tensor,
          kvHeadIndex: g,
          tokenStart: k,
          tokenEnd: k,
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
          { tensor: 'rmsnorm', rowStart: tLocal, rowEnd: tLocal, colStart: 0, colEnd: d - 1, role: 'operand_b' },
        ],
      };
    }

    case 'scores': {
      const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
      const h = cell.headIndex ?? (g === 0 ? 0 : 16);
      const q = cell.row;
      const k = cell.col;
      const qAbs = P + q;
      const val = sim.Scores[(h * T + q) * K + k];
      const isMasked = !isAttentionKeyVisible(sim.attentionMaskMode, P, q, k, K) || !Number.isFinite(val);

      if (isMasked) {
        return {
          cell,
          tensorLabel: `Scores · ${fmtHead(h)}`,
          coordLabel: `Scores[${fmtHead(h)}, q=${tokStr(sim, q)}, k=${keyTokStr(sim, k)}]`,
          headBadge: `${fmtHead(h)} → KV${g}`,
          value: -Infinity,
          isMasked: true,
          valueText: '−∞ (masked)',
          equation: `k=${k} > q=${qAbs} (future token) → Causal Mask sets Score = −∞ → Softmax = 0.000`,
          subEquation: `Autoregressive causality blocks query ${tokStr(sim, q)} from attending future key ${keyTokStr(sim, k)}`,
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
      const kHeadOff = g * dHead * K;
      const prods: string[] = [];
      let dot = 0;
      for (let r = 0; r < dHead; r++) {
        const qv = sim.Q[qHeadOff + r * T + q];
        const kv = sim.K[kHeadOff + r * K + k];
        dot += qv * kv;
        prods.push(`(${fmtSigned(qv, 2)}·${fmtSigned(kv, 2)})`);
      }
      const keyOriginNote =
        P > 0 ? (k < P ? ` · Reused prefix K${g} · Rect [${T}×${K}]` : ` · Newly computed K${g} · Rect [${T}×${K}]`) : '';
      const modeNote = sim.attentionMaskMode === 'bidirectional_block' ? ' · Bidirectional block' : '';
      return {
        cell,
        tensorLabel: `Scores · ${fmtHead(h)}`,
        coordLabel: `Scores[${fmtHead(h)}, q=${tokStr(sim, q)}, k=${keyTokStr(sim, k)}]`,
        headBadge: P > 0 ? `${fmtHead(h)} → KV${g} · [${T}×${K}]` : `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `(${fmtHead(h)}[:,${qAbs}] · K${g}[:,${k}]) / √4 = [${prods.join(' + ')}] / 2 = ${fmtSigned(dot, 3)} / 2 = ${fmtSigned(val)}`,
        subEquation: `16:1 GQA: ${fmtHead(h)} queries shared K${g} · Post-Softmax weight = ${fmtProb(sim.Softmax[(h * T + q) * K + k])}${keyOriginNote}${modeNote}`,
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
      const qAbs = P + q;
      const val = sim.Softmax[(h * T + q) * K + k];
      const scoreVal = sim.Scores[(h * T + q) * K + k];
      const isMasked = !isAttentionKeyVisible(sim.attentionMaskMode, P, q, k, K);

      if (isMasked) {
        return {
          cell,
          tensorLabel: `Softmax · ${fmtHead(h)}`,
          coordLabel: `Softmax[${fmtHead(h)}, q=${tokStr(sim, q)}, k=${keyTokStr(sim, k)}]`,
          headBadge: `${fmtHead(h)} → KV${g}`,
          value: 0,
          isMasked: true,
          valueText: '0.000 (masked)',
          equation: `Score = −∞ (k=${k} > q=${qAbs}) → exp(−∞) = 0 → Softmax = 0.000`,
          subEquation: `Upper triangle strictly 0.000 · Valid causal keys for q=${qAbs} are k = 0..${qAbs}`,
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

      const maxVisK = maxVisibleKeyIndex(sim.attentionMaskMode, P, q, K);
      let maxS = -Infinity;
      for (let j = 0; j <= maxVisK; j++) {
        if (!isAttentionKeyVisible(sim.attentionMaskMode, P, q, j, K)) continue;
        const s = sim.Scores[(h * T + q) * K + j];
        if (s > maxS) maxS = s;
      }
      let sumExp = 0;
      let rowSum = 0;
      for (let j = 0; j <= maxVisK; j++) {
        if (!isAttentionKeyVisible(sim.attentionMaskMode, P, q, j, K)) continue;
        sumExp += Math.exp(sim.Scores[(h * T + q) * K + j] - maxS);
        rowSum += sim.Softmax[(h * T + q) * K + j];
      }
      const num = Math.exp(scoreVal - maxS);
      const rowKindLabel = sim.attentionMaskMode === 'bidirectional_block' ? 'Bidirectional' : 'Causal';
      return {
        cell,
        tensorLabel: `Softmax · ${fmtHead(h)}`,
        coordLabel: `Softmax[${fmtHead(h)}, q=${tokStr(sim, q)}, k=${keyTokStr(sim, k)}]`,
        headBadge: P > 0 ? `${fmtHead(h)} → KV${g} · [${T}×${K}]` : `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtProb(val),
        equation: `exp(${fmtSigned(scoreVal)} − ${fmtSigned(maxS)}) / Σ_{j=0..${maxVisK}} exp(s_j − max) = ${num.toFixed(3)} / ${sumExp.toFixed(3)} = ${fmtProb(val)}`,
        subEquation: `${rowKindLabel} row ${qAbs} sum Σ_{k=0..${maxVisK}} Softmax = ${rowSum.toFixed(4)} (${(val * 100).toFixed(1)}% weight on ${keyTokStr(sim, k)})`,
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
            colEnd: maxVisK,
            role: 'operand_a',
          },
        ],
      };
    }

    case 'v_out': {
      const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
      const q = cell.row;
      const qAbs = P + q;
      const maxVisK = maxVisibleKeyIndex(sim.attentionMaskMode, P, q, K);
      const visCount = maxVisK + 1;
      const localH = Math.floor(cell.col / dHead);
      const r = cell.col % dHead;
      const h = g * 16 + localH;
      const val = sim.V_out[(h * dHead + r) * T + q];
      const vHeadOff = g * dHead * K;
      const softRowOff = (h * T + q) * K;
      const terms: string[] = [];
      const showN = Math.min(visCount, 3);
      for (let k = 0; k < showN; k++) {
        terms.push(`${fmtProb(sim.Softmax[softRowOff + k], 2)}·${fmtSigned(sim.V[vHeadOff + r * K + k], 2)}`);
      }
      const tail = visCount > showN ? ` + … (${visCount} terms)` : '';
      const subEquation =
        sim.attentionMaskMode === 'bidirectional_block'
          ? P > 0
            ? `Bidirectional blend of ${P} reused + ${T} block shared V${g} slices 0..${maxVisK} → Concat ch ${h * dHead + r}`
            : `Bidirectional blend of shared V${g} slices 0..${maxVisK} → feeds Concat channel ${h * dHead + r}`
          : P > 0
            ? `Weighted blend of ${P} reused + ${q + 1} new shared V${g} slices 0..${maxVisK} → Concat ch ${h * dHead + r}`
            : `Weighted blend of cached shared V${g} slices 0..${maxVisK} → feeds Concat channel ${h * dHead + r}`;
      return {
        cell,
        tensorLabel: `V Out · ${fmtHead(h)}`,
        coordLabel: `V_out[${fmtHead(h)}, q=${tokStr(sim, q)}, d=${r}]`,
        headBadge: `${fmtHead(h)} → KV${g}`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val),
        equation: `Σ_{k=0..${maxVisK}} Softmax[${fmtHead(h)},${qAbs},k]·V${g}[${r},k] = ${terms.join(' + ')}${tail} = ${fmtSigned(val)}`,
        subEquation,
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
            colEnd: maxVisK,
            role: 'operand_a',
          },
          {
            tensor: 'kv_v',
            kvHeadIndex: g,
            tokenStart: 0,
            tokenEnd: maxVisK,
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
      const qAbs = P + q;
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
        equation: `Concat[${c}, ${qAbs}] = V_out[${fmtHead(h)}, d=${r}, q=${qAbs}] = ${fmtSigned(val)}`,
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
      const tAbs = P + t;
      const i = cell.col;
      const xVal = sim.X[i * T + t];
      const attnVal = sim.Attn_out[i * T + t];
      const yVal = sim.Y[i * T + t];
      const gateVal = sim.residualGateApplied && sim.residualGate ? sim.residualGate[i] : null;
      return {
        cell,
        tensorLabel: '+ Residual · Y',
        coordLabel: `Y[ch=${i}, t=${tokStr(sim, t)}]`,
        headBadge: null,
        value: yVal,
        isMasked: false,
        valueText: fmtSigned(yVal),
        equation:
          gateVal !== null
            ? `Y[${i},${tAbs}] = X[${i},${tAbs}] + gate[${i}](${fmtSigned(gateVal)})·(W_O·Concat)[${i},${tAbs}] = ${fmtSigned(xVal)} + ${fmtSigned(gateVal)}·(${fmtSigned(attnVal)}) = ${fmtSigned(yVal)}`
            : `Y[${i},${tAbs}] = X[${i},${tAbs}] + (W_O·Concat)[${i},${tAbs}] = ${fmtSigned(xVal)} + (${fmtSigned(attnVal)}) = ${fmtSigned(yVal)}`,
        subEquation: `Attn_out[${i},${tAbs}] = Σ_{c=0..127} W_O[${i},c]·Concat[c,${tAbs}] = ${fmtSigned(attnVal)}`,
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
      // QK^T / sqrt(d): sweep key k = 0..keyTokens-1 on latest query row q = lastTok
      const k = Math.floor(p * sim.keyTokens);
      return { tensor: 'scores', headIndex: h, kvHeadIndex: g, row: lastTok, col: k };
    }
    case 3: {
      // Causal Softmax: sweep causal probabilities k = 0..keyTokens-1 along row q = lastTok
      const k = Math.floor(p * sim.keyTokens);
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
