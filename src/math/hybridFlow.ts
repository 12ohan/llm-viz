/**
 * Unified Deterministic Miniature Forward Pass (`src/math/hybridFlow.ts`)
 *
 * Separates stable, immutable model parameter ownership (`HybridModelParameters`) from
 * individual forward-pass executions (`HybridExecutionResult`) and bounded execution caching
 * (`HybridExecutionCache`).
 *
 * Executes the exact 52-layer NVIDIA Nemotron 3 Nano hybrid motif (`HYBRID_OVERRIDE_PATTERN`:
 * 23 Mamba-2 + 23 Granular MoE 128+1 + 6 NoPE GQA Attention), preceded by Untied Token Embeddings
 * (NoPE) and followed by Final RMSNorm (ε = 1e-5) and an Untied LM Head + Softmax.
 *
 * Epistemic disclaimer:
 *   Educational miniature Float32 simulator (`d = 8`, `V_mini = 32`, deterministic synthetic
 *   seeded weights). NOT 31.6B checkpoint inference and NOT real Nemotron tokenizer logits.
 *   `BF16` vs. `FP8 Selective` remains strictly production metadata.
 */

import { HYBRID_OVERRIDE_PATTERN, NEMOTRON_SPEC, type Kind } from '../model/spec';
import {
  EDUCATIONAL_TOKENS,
  GQA_SIM_CONFIG,
  runGqaSim,
  type GqaSimFixture,
  type GqaSimResult,
} from './gqaSim';
import {
  MAMBA_SIM_CONFIG,
  runMambaSim,
  type MambaSimFixture,
  type MambaSimResult,
} from './mambaSim';
import {
  MOE_SIM_CONFIG,
  runMoeSim,
  type MoeSimFixture,
  type MoeSimResult,
  type RouterPrecisionPolicy,
} from './moeSim';

export type { RouterPrecisionPolicy };

/**
 * Deterministic additive Float32 weight perturbation specification applied during
 * parameter generation (e.g. for deriving an independent TwoTower Denoiser Tower
 * from the Context Tower's base seed without post-hoc weight mutation).
 */
export interface HybridWeightPerturbation {
  readonly seed: number;
  readonly scale: number;
  readonly sourceFingerprint?: string;
}

/**
 * Synthetic 32-word educational miniature vocabulary (`0..15` = input sequence tokens,
 * `16..31` = continuation candidates for the untied miniature LM head).
 */
export const MINI_VOCAB: readonly string[] = Object.freeze([
  ...EDUCATIONAL_TOKENS,
  'efficiently',
  'linearly',
  'context',
  'memory',
  'state',
  'cache',
  'routing',
  'experts',
  'without',
  'growing',
  'constant',
  'recurrent',
  'sparse',
  'parallel',
  'precision',
  'output',
]);

export const DEFAULT_HYBRID_MODEL_SEED = 0x4e454d33; // "NEM3"
export const HYBRID_MINI_D = 8 as const;
export const HYBRID_MINI_VOCAB_SIZE = 32 as const;
export const HYBRID_UNIFIED_RMS_EPS = 1e-5 as const;
/** Default retained bay weight capacity (all 52 miniature bays = ~1.14 MiB; eliminates per-step PRNG thrash). */
export const DEFAULT_MAX_RETAINED_WEIGHT_BAYS = 52 as const;

/** Default primary cutaway bays retained immediately after a forward pass (all 52 bays remain lazily accessible). */
export const DEFAULT_RETAINED_TRACE_BAYS: readonly number[] = Object.freeze([
  1, 2, 6, 42, 43, 52,
]);

// ============================================================================
// DETERMINISTIC PRNG HELPERS (Mulberry32 + Box-Muller, strictly isolated per call)
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

function mixSeed(baseSeed: number, bayId: number, salt: number): number {
  let h = (baseSeed ^ Math.imul(bayId + 1, 0x9e3779b1) ^ Math.imul(salt, 0x85ebca6b)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

function applyDeterministicPerturbation(
  arr: Float32Array,
  rng: () => number,
  scale: number,
): void {
  for (let i = 0; i < arr.length; i++) {
    arr[i] = Math.fround(arr[i] + (rng() * 2 - 1) * scale);
  }
}

/**
 * Compute a deterministic 64-bit hex fingerprint over the COMPLETE model parameterization:
 * seed, full layer pattern, dimensions, vocab, exact IEEE-754 float64 bits of rmsNormEps and
 * residualOutputScale, plus optional routerPrecisionPolicy, perturbation, and generationPolicy.
 * Guarantees distinct fingerprints even if two differently parameterized models share a display
 * `modelId`, while remaining 100% bit-identical to Nano v1.0.0 under default options.
 */
export function computeModelParameterFingerprint(params: {
  readonly seed: number;
  readonly pattern: string;
  readonly d: number;
  readonly vocabSize: number;
  readonly rmsNormEps: number;
  readonly residualOutputScale: number;
  readonly vocab: readonly string[];
  readonly routerPrecisionPolicy?: RouterPrecisionPolicy;
  readonly perturbation?: HybridWeightPerturbation;
  readonly generationPolicy?: string;
}): string {
  let h1 = 0x811c9dc5 >>> 0;
  let h2 = 0x9e3779b9 >>> 0;
  const feedByte = (b: number) => {
    const byte = b & 0xff;
    h1 ^= byte;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= (byte + 0x5a) & 0xff;
    h2 = Math.imul(h2, 0x85ebca6b) >>> 0;
  };
  const feedU32 = (v: number) => {
    const u = v >>> 0;
    feedByte(u);
    feedByte(u >>> 8);
    feedByte(u >>> 16);
    feedByte(u >>> 24);
  };
  const dv = new DataView(new ArrayBuffer(8));
  const feedF64 = (v: number) => {
    dv.setFloat64(0, Object.is(v, -0) ? 0 : v, true);
    feedU32(dv.getUint32(0, true));
    feedU32(dv.getUint32(4, true));
  };
  const feedStr = (s: string) => {
    feedU32(s.length);
    for (let i = 0; i < s.length; i++) {
      const code = s.charCodeAt(i);
      feedByte(code);
      feedByte(code >>> 8);
    }
  };

  feedU32(params.seed);
  feedStr(params.pattern);
  feedU32(params.d);
  feedU32(params.vocabSize);
  feedF64(params.rmsNormEps);
  feedF64(params.residualOutputScale);
  feedU32(params.vocab.length);
  for (const w of params.vocab) {
    feedStr(w);
  }

  // Backward-compatible Nano v1.0.0 fingerprint preservation: only hash extension fields
  // when at least one differs from Nano v1.0.0 defaults.
  const routerPolicy: RouterPrecisionPolicy = params.routerPrecisionPolicy ?? 'float32_nano';
  const hasExtendedConfig =
    routerPolicy !== 'float32_nano' ||
    params.perturbation !== undefined ||
    params.generationPolicy !== undefined;
  if (hasExtendedConfig) {
    feedByte(0xee);
    feedStr(routerPolicy);
    if (params.perturbation !== undefined) {
      feedByte(1);
      feedU32(params.perturbation.seed);
      feedF64(params.perturbation.scale);
      if (params.perturbation.sourceFingerprint !== undefined) {
        feedByte(1);
        feedStr(params.perturbation.sourceFingerprint);
      } else {
        feedByte(0);
      }
    } else {
      feedByte(0);
    }
    if (params.generationPolicy !== undefined) {
      feedByte(1);
      feedStr(params.generationPolicy);
    } else {
      feedByte(0);
    }
  }

  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

// ============================================================================
// 1. STABLE MODEL PARAMETERS (Separated from Execution & Independent of T / tokenIds)
// ============================================================================

export interface HybridModelConfig {
  readonly modelId: string;
  /** Canonical 64-bit fingerprint covering seed, full layer pattern, rmsNormEps, residualOutputScale, vocab, and policy extensions. */
  readonly parameterFingerprint: string;
  readonly seed: number;
  readonly pattern: string;
  readonly d: 8;
  readonly vocabSize: number;
  readonly rmsNormEps: number;
  /** Depth-aware residual branch scaling factor (~1 / sqrt(2 * L)). */
  readonly residualOutputScale: number;
  /** Bounded number of bay weight sets retained in memory inside this parameter instance. */
  readonly maxRetainedWeightBays: number;
  /** MoE router scoring precision policy (defaults to `'float32_nano'`). */
  readonly routerPrecisionPolicy: RouterPrecisionPolicy;
  /** Optional deterministic additive weight perturbation applied at parameter generation time. */
  readonly perturbation?: HybridWeightPerturbation;
  /** Optional generation policy identifier included in fingerprint when present. */
  readonly generationPolicy?: string;
}

export interface MambaBayWeights {
  readonly bayId: number;
  readonly kind: 'M';
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

export interface MoeBayWeights {
  readonly bayId: number;
  readonly kind: 'E';
  readonly gamma_in: Float32Array;
  readonly W_router: Float32Array;
  readonly e_score_correction_bias: Float32Array;
  readonly W_up: Float32Array;
  readonly W_down: Float32Array;
  readonly W_up_s: Float32Array;
  readonly W_down_s: Float32Array;
}

export interface GqaBayWeights {
  readonly bayId: number;
  readonly kind: '*';
  readonly gamma: Float32Array;
  readonly W_Q: Float32Array;
  readonly W_K: Float32Array;
  readonly W_V: Float32Array;
  readonly W_O: Float32Array;
}

export type HybridBayWeights = MambaBayWeights | MoeBayWeights | GqaBayWeights;

export interface HybridModelParameters {
  readonly modelId: string;
  /** Canonical fingerprint over all model parameters; used in cache keys even if display `modelId` is duplicated. */
  readonly parameterFingerprint: string;
  readonly config: HybridModelConfig;
  readonly vocab: readonly string[];
  /** Untied token embedding table [vocabSize, d] (NoPE: zero positional embedding). */
  readonly W_embed: Float32Array;
  /** Learned non-unit Final RMSNorm weights [d]. */
  readonly finalNormGamma: Float32Array;
  /** Strictly untied LM Head projection [vocabSize, d] (W_lm_head !== W_embed). */
  readonly W_lm_head: Float32Array;
  /**
   * Deterministic bay weight accessor. Strictly independent of sequence length `T` and `tokenIds`.
   * Reproducible across calls; retains up to `config.maxRetainedWeightBays` materialized bays.
   */
  getBayWeights(bayId: number): HybridBayWeights;
  /** Exact typed-array byte count currently retained inside this parameter instance. */
  retainedWeightBytes(): number;
}

function bayWeightsByteSize(w: HybridBayWeights): number {
  if (w.kind === 'M') {
    return (
      w.gamma_in.byteLength +
      w.W_in.byteLength +
      w.w_conv.byteLength +
      w.b_conv.byteLength +
      w.A_log.byteLength +
      w.dt_bias.byteLength +
      w.D.byteLength +
      w.gamma_gate.byteLength +
      w.W_out.byteLength
    );
  }
  if (w.kind === 'E') {
    return (
      w.gamma_in.byteLength +
      w.W_router.byteLength +
      w.e_score_correction_bias.byteLength +
      w.W_up.byteLength +
      w.W_down.byteLength +
      w.W_up_s.byteLength +
      w.W_down_s.byteLength
    );
  }
  return (
    w.gamma.byteLength +
    w.W_Q.byteLength +
    w.W_K.byteLength +
    w.W_V.byteLength +
    w.W_O.byteLength
  );
}

function generateMambaBayWeights(
  bayId: number,
  modelSeed: number,
  residualScale: number,
  perturbation?: HybridWeightPerturbation,
): MambaBayWeights {
  const { d, nHeads, dInner, convKernel, convDim, inProjDim } = MAMBA_SIM_CONFIG;
  const rng = mulberry32(mixSeed(modelSeed, bayId, 0x11));
  const gauss = makeGaussian(mixSeed(modelSeed, bayId, 0x12));

  const gamma_in = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    gamma_in[i] = Math.fround(0.90 + 0.20 * rng());
  }

  const W_in = new Float32Array(inProjDim * d);
  const wInScale = 0.56 / Math.sqrt(d);
  for (let r = 0; r < inProjDim; r++) {
    const scale = r >= dInner + convDim ? wInScale * 0.72 : wInScale;
    for (let j = 0; j < d; j++) {
      W_in[r * d + j] = Math.fround(gauss() * scale);
    }
  }

  const w_conv = new Float32Array(convDim * convKernel);
  const b_conv = new Float32Array(convDim);
  const tapScales = [0.22, 0.30, 0.42, 0.58];
  for (let c = 0; c < convDim; c++) {
    for (let s = 0; s < convKernel; s++) {
      w_conv[c * convKernel + s] = Math.fround(
        (s === 3 ? 0.36 : 0.0) + gauss() * tapScales[s],
      );
    }
    b_conv[c] = Math.fround(gauss() * 0.10);
  }

  const A_log = new Float32Array(nHeads);
  const dt_bias = new Float32Array(nHeads);
  const D = new Float32Array(nHeads);
  for (let h = 0; h < nHeads; h++) {
    const frac = h / (nHeads - 1);
    A_log[h] = Math.fround(-1.55 + frac * 1.90 + gauss() * 0.12);
    dt_bias[h] = Math.fround(-0.85 + 0.90 * rng());
    D[h] = Math.fround(0.42 + 0.50 * rng());
  }

  const gamma_gate = new Float32Array(dInner);
  for (let c = 0; c < dInner; c++) {
    gamma_gate[c] = Math.fround(0.86 + 0.28 * rng());
  }

  // Depth-conditioned output projection W_out [d, dInner] so RMS(Mamba_out) ~ 0.14-0.22
  const W_out = new Float32Array(d * dInner);
  const outStd = (1.18 * residualScale) / Math.sqrt(dInner);
  for (let i = 0; i < d; i++) {
    for (let c = 0; c < dInner; c++) {
      W_out[i * dInner + c] = Math.fround(gauss() * outStd);
    }
  }

  if (perturbation && perturbation.scale > 0) {
    const pRng = mulberry32((perturbation.seed ^ Math.imul(bayId, 0x9e3779b1)) >>> 0);
    applyDeterministicPerturbation(gamma_in, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_in, pRng, perturbation.scale);
    applyDeterministicPerturbation(w_conv, pRng, perturbation.scale);
    applyDeterministicPerturbation(b_conv, pRng, perturbation.scale);
    applyDeterministicPerturbation(A_log, pRng, perturbation.scale);
    applyDeterministicPerturbation(dt_bias, pRng, perturbation.scale);
    applyDeterministicPerturbation(D, pRng, perturbation.scale);
    applyDeterministicPerturbation(gamma_gate, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_out, pRng, perturbation.scale);
  }

  return {
    bayId,
    kind: 'M',
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

function generateMoeBayWeights(
  bayId: number,
  modelSeed: number,
  residualScale: number,
  perturbation?: HybridWeightPerturbation,
): MoeBayWeights {
  const { d, nRoutedExperts, dExp, dShared } = MOE_SIM_CONFIG;
  const rng = mulberry32(mixSeed(modelSeed, bayId, 0x21));
  const gauss = makeGaussian(mixSeed(modelSeed, bayId, 0x22));

  const gamma_in = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    gamma_in[i] = Math.fround(0.90 + 0.20 * rng());
  }

  // Unit-normed router prototypes (scaled 1.45) + small e_score_correction_bias for diverse Top-6 routing
  const W_router = new Float32Array(nRoutedExperts * d);
  const e_score_correction_bias = new Float32Array(nRoutedExperts);
  for (let e = 0; e < nRoutedExperts; e++) {
    let normSq = 0;
    for (let i = 0; i < d; i++) {
      const v = gauss();
      W_router[e * d + i] = v;
      normSq += v * v;
    }
    const inv = 1.45 / Math.sqrt(normSq + 1e-6);
    for (let i = 0; i < d; i++) {
      W_router[e * d + i] = Math.fround(W_router[e * d + i] * inv);
    }
    e_score_correction_bias[e] = Math.fround(gauss() * 0.14);
  }

  // Non-gated ReLU² routed experts (2 matrices: W_up + W_down, zero gate_proj)
  const W_up = new Float32Array(nRoutedExperts * dExp * d);
  const W_down = new Float32Array(nRoutedExperts * d * dExp);
  const upStd = 0.72 / Math.sqrt(d);
  const routedDownStd = (0.64 * residualScale) / Math.sqrt(dExp);
  for (let e = 0; e < nRoutedExperts; e++) {
    const upBase = e * dExp * d;
    for (let k = 0; k < dExp * d; k++) {
      W_up[upBase + k] = Math.fround(gauss() * upStd);
    }
    const downBase = e * d * dExp;
    for (let k = 0; k < d * dExp; k++) {
      W_down[downBase + k] = Math.fround(gauss() * routedDownStd);
    }
  }

  // Always-on double-width shared expert (d_shared = 8 = 2 * d_exp, unscaled 1.0x)
  const W_up_s = new Float32Array(dShared * d);
  const W_down_s = new Float32Array(d * dShared);
  const sharedDownStd = (0.76 * residualScale) / Math.sqrt(dShared);
  for (let k = 0; k < dShared * d; k++) {
    W_up_s[k] = Math.fround(gauss() * upStd);
  }
  for (let k = 0; k < d * dShared; k++) {
    W_down_s[k] = Math.fround(gauss() * sharedDownStd);
  }

  if (perturbation && perturbation.scale > 0) {
    const pRng = mulberry32((perturbation.seed ^ Math.imul(bayId, 0x9e3779b1)) >>> 0);
    applyDeterministicPerturbation(gamma_in, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_router, pRng, perturbation.scale);
    applyDeterministicPerturbation(e_score_correction_bias, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_up, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_down, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_up_s, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_down_s, pRng, perturbation.scale);
  }

  return {
    bayId,
    kind: 'E',
    gamma_in,
    W_router,
    e_score_correction_bias,
    W_up,
    W_down,
    W_up_s,
    W_down_s,
  };
}

function generateGqaBayWeights(
  bayId: number,
  modelSeed: number,
  residualScale: number,
  perturbation?: HybridWeightPerturbation,
): GqaBayWeights {
  const { d, nQ, nKv, dHead, qConcatDim } = GQA_SIM_CONFIG;
  const rng = mulberry32(mixSeed(modelSeed, bayId, 0x31));
  const gauss = makeGaussian(mixSeed(modelSeed, bayId, 0x32));

  const gamma = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    gamma[i] = Math.fround(0.90 + 0.20 * rng());
  }

  const W_Q = new Float32Array(qConcatDim * d);
  for (let h = 0; h < nQ; h++) {
    const headGain = 0.76 + ((h * 7 + bayId) % 11) * 0.03;
    const scale = headGain / Math.sqrt(d * 0.55);
    for (let r = 0; r < dHead; r++) {
      const outRow = h * dHead + r;
      for (let j = 0; j < d; j++) {
        W_Q[outRow * d + j] = Math.fround(gauss() * scale);
      }
    }
  }

  const W_K = new Float32Array(nKv * dHead * d);
  const W_V = new Float32Array(nKv * dHead * d);
  const kScale = 0.84 / Math.sqrt(d * 0.55);
  const vScale = 0.80 / Math.sqrt(d * 0.65);
  for (let row = 0; row < nKv * dHead; row++) {
    for (let j = 0; j < d; j++) {
      W_K[row * d + j] = Math.fround(gauss() * kScale);
      W_V[row * d + j] = Math.fround(gauss() * vScale);
    }
  }

  // Depth-conditioned W_O [d, qConcatDim] accounting for 16:1 KV correlation across query heads
  const W_O = new Float32Array(d * qConcatDim);
  const oStd = (0.72 * residualScale) / Math.sqrt(qConcatDim);
  for (let i = 0; i < d; i++) {
    for (let c = 0; c < qConcatDim; c++) {
      W_O[i * qConcatDim + c] = Math.fround(gauss() * oStd);
    }
  }

  if (perturbation && perturbation.scale > 0) {
    const pRng = mulberry32((perturbation.seed ^ Math.imul(bayId, 0x9e3779b1)) >>> 0);
    applyDeterministicPerturbation(gamma, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_Q, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_K, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_V, pRng, perturbation.scale);
    applyDeterministicPerturbation(W_O, pRng, perturbation.scale);
  }

  return { bayId, kind: '*', gamma, W_Q, W_K, W_V, W_O };
}

export interface CreateHybridModelOptions {
  readonly modelId?: string;
  readonly seed?: number;
  readonly pattern?: string;
  readonly rmsNormEps?: number;
  readonly residualOutputScale?: number;
  readonly maxRetainedWeightBays?: number;
  readonly pinnedWeightBays?: readonly number[];
  /** Custom vocabulary table (defaults to 32-token `MINI_VOCAB`). */
  readonly vocab?: readonly string[];
  /** MoE router scoring precision policy (defaults to `'float32_nano'`). */
  readonly routerPrecisionPolicy?: RouterPrecisionPolicy;
  /** Optional deterministic additive perturbation applied to all generated weights. */
  readonly perturbation?: HybridWeightPerturbation;
  /** Optional generation policy tag hashed into `parameterFingerprint`. */
  readonly generationPolicy?: string;
}

/**
 * Create an independent, immutable `HybridModelParameters` instance with deterministic
 * lazy per-bay parameter generation and zero shared mutable globals.
 */
export function createHybridModelParameters(
  options?: CreateHybridModelOptions,
): HybridModelParameters {
  const seed = (options?.seed ?? DEFAULT_HYBRID_MODEL_SEED) >>> 0;
  const pattern = options?.pattern ?? HYBRID_OVERRIDE_PATTERN;
  const numLayers = pattern.length;
  if (numLayers < 1) {
    throw new RangeError('Hybrid model pattern must contain at least 1 layer');
  }
  for (let i = 0; i < numLayers; i++) {
    const ch = pattern[i];
    if (ch !== 'M' && ch !== 'E' && ch !== '*') {
      throw new RangeError(`Unexpected layer kind "${ch}" at pattern index ${i}`);
    }
  }
  const vocab = options?.vocab ? Object.freeze([...options.vocab]) : MINI_VOCAB;
  const vocabSize = vocab.length;
  if (vocabSize < 1) {
    throw new RangeError('Hybrid model vocab must contain at least 1 token');
  }
  const routerPrecisionPolicy: RouterPrecisionPolicy =
    options?.routerPrecisionPolicy ?? 'float32_nano';
  let perturbation: HybridWeightPerturbation | undefined;
  if (options?.perturbation !== undefined) {
    if (!Number.isFinite(options.perturbation.seed)) {
      throw new RangeError(`Invalid perturbation.seed: ${options.perturbation.seed}`);
    }
    if (!Number.isFinite(options.perturbation.scale) || options.perturbation.scale < 0) {
      throw new RangeError(
        `Invalid perturbation.scale (${options.perturbation.scale}): must be finite >= 0`,
      );
    }
    perturbation = Object.freeze({
      seed: options.perturbation.seed >>> 0,
      scale: options.perturbation.scale,
      ...(options.perturbation.sourceFingerprint !== undefined
        ? { sourceFingerprint: options.perturbation.sourceFingerprint }
        : {}),
    });
  }
  const generationPolicy = options?.generationPolicy;
  const rmsNormEps = options?.rmsNormEps ?? HYBRID_UNIFIED_RMS_EPS;
  const residualOutputScale =
    options?.residualOutputScale ?? 1.42 / Math.sqrt(Math.max(2, 2 * numLayers));
  const maxRetainedWeightBays = Math.max(
    1,
    Math.trunc(options?.maxRetainedWeightBays ?? DEFAULT_MAX_RETAINED_WEIGHT_BAYS),
  );
  const parameterFingerprint = computeModelParameterFingerprint({
    seed,
    pattern,
    d: HYBRID_MINI_D,
    vocabSize,
    rmsNormEps,
    residualOutputScale,
    vocab,
    routerPrecisionPolicy,
    perturbation,
    generationPolicy,
  });
  const modelId =
    options?.modelId ??
    `nemotron-nano-mini:s${seed.toString(16)}:L${numLayers}:fp${parameterFingerprint}`;

  const config: HybridModelConfig = Object.freeze({
    modelId,
    parameterFingerprint,
    seed,
    pattern,
    d: HYBRID_MINI_D,
    vocabSize,
    rmsNormEps,
    residualOutputScale,
    maxRetainedWeightBays,
    routerPrecisionPolicy,
    ...(perturbation !== undefined ? { perturbation } : {}),
    ...(generationPolicy !== undefined ? { generationPolicy } : {}),
  });

  // 1. Untied Token Embeddings W_embed [vocabSize, 8], row-normalized to unit RMS (NoPE)
  const gEmbed = makeGaussian(mixSeed(seed, 0, 0x101));
  const W_embed = new Float32Array(vocabSize * HYBRID_MINI_D);
  for (let v = 0; v < vocabSize; v++) {
    let sumSq = 0;
    for (let i = 0; i < HYBRID_MINI_D; i++) {
      const val = gEmbed();
      W_embed[v * HYBRID_MINI_D + i] = val;
      sumSq += val * val;
    }
    const rms = Math.sqrt(sumSq / HYBRID_MINI_D + 1e-8);
    for (let i = 0; i < HYBRID_MINI_D; i++) {
      W_embed[v * HYBRID_MINI_D + i] = Math.fround(W_embed[v * HYBRID_MINI_D + i] / rms);
    }
  }

  // 2. Final RMSNorm learned non-unit scale gamma_final [8]
  const rngFinal = mulberry32(mixSeed(seed, numLayers + 1, 0x202));
  const finalNormGamma = new Float32Array(HYBRID_MINI_D);
  for (let i = 0; i < HYBRID_MINI_D; i++) {
    finalNormGamma[i] = Math.fround(0.90 + 0.22 * rngFinal());
  }

  // 3. Strictly untied LM Head W_lm_head [vocabSize, 8] (independent seed -> W_lm_head !== W_embed)
  const gLm = makeGaussian(mixSeed(seed, numLayers + 2, 0x303));
  const W_lm_head = new Float32Array(vocabSize * HYBRID_MINI_D);
  const lmStd = 1.15 / Math.sqrt(HYBRID_MINI_D);
  for (let k = 0; k < vocabSize * HYBRID_MINI_D; k++) {
    W_lm_head[k] = Math.fround(gLm() * lmStd);
  }

  if (perturbation && perturbation.scale > 0) {
    applyDeterministicPerturbation(
      W_embed,
      mulberry32((perturbation.seed ^ 0x70657201) >>> 0),
      perturbation.scale,
    );
    applyDeterministicPerturbation(
      finalNormGamma,
      mulberry32((perturbation.seed ^ 0x70657202) >>> 0),
      perturbation.scale,
    );
    applyDeterministicPerturbation(
      W_lm_head,
      mulberry32((perturbation.seed ^ 0x70657203) >>> 0),
      perturbation.scale,
    );
  }

  // Bounded LRU cache of materialized bay weights (pinned bays prioritized to avoid regenerating inspectable cutaways)
  const pinnedSet = new Set<number>(options?.pinnedWeightBays ?? DEFAULT_RETAINED_TRACE_BAYS);
  const weightCache = new Map<number, HybridBayWeights>();

  const getBayWeights = (bayId: number): HybridBayWeights => {
    if (!Number.isInteger(bayId) || bayId < 1 || bayId > numLayers) {
      throw new RangeError(`Bay ID ${bayId} out of bounds [1..${numLayers}]`);
    }
    const existing = weightCache.get(bayId);
    if (existing) {
      // Refresh LRU recency
      weightCache.delete(bayId);
      weightCache.set(bayId, existing);
      return existing;
    }

    const kind = pattern[bayId - 1] as Kind;
    const generated: HybridBayWeights = Object.freeze(
      kind === 'M'
        ? generateMambaBayWeights(bayId, seed, residualOutputScale, perturbation)
        : kind === 'E'
          ? generateMoeBayWeights(bayId, seed, residualOutputScale, perturbation)
          : generateGqaBayWeights(bayId, seed, residualOutputScale, perturbation),
    );

    // Always insert newly generated bay into the bounded LRU cache and evict the oldest
    // non-pinned entry first (preserving pinned bays whenever capacity allows).
    weightCache.set(bayId, generated);
    while (weightCache.size > maxRetainedWeightBays) {
      let evictedKey: number | null = null;
      for (const key of weightCache.keys()) {
        if (key !== bayId && !pinnedSet.has(key)) {
          evictedKey = key;
          break;
        }
      }
      if (evictedKey === null) {
        for (const key of weightCache.keys()) {
          if (key !== bayId) {
            evictedKey = key;
            break;
          }
        }
      }
      if (evictedKey === null) break;
      weightCache.delete(evictedKey);
    }

    return generated;
  };

  const retainedWeightBytes = (): number => {
    let total = W_embed.byteLength + finalNormGamma.byteLength + W_lm_head.byteLength;
    for (const w of weightCache.values()) {
      total += bayWeightsByteSize(w);
    }
    return total;
  };

  return {
    modelId,
    parameterFingerprint,
    config,
    vocab,
    W_embed,
    finalNormGamma,
    W_lm_head,
    getBayWeights,
    retainedWeightBytes,
  };
}

/** Build the default Nemotron 3 Nano miniature parameter instance (`seed = DEFAULT_HYBRID_MODEL_SEED`). */
export function createDefaultHybridModel(): HybridModelParameters {
  return createHybridModelParameters();
}

// ============================================================================
// 2. EXECUTION INPUTS, RESULTS, TRACE IDENTITY & BOUNDED CACHING
// ============================================================================

export interface HybridExecutionInput {
  /** Token IDs in `[0..vocabSize-1]`, length `T` in `[1..16]`. */
  readonly tokenIds: readonly number[];
  /** Optional custom display labels aligned 1-to-1 with `tokenIds`. Defaults to `model.vocab[id]`. */
  readonly tokenLabels?: readonly string[];
  /** Override which bay IDs retain full simulator traces eagerly (defaults to `[1, 2, 6, 42, 43, 52]`). */
  readonly retainedTraceBays?: readonly number[];
  /** Maximum number of full layer traces retained inside this execution result (bounded LRU). */
  readonly maxRetainedTraces?: number;
}

export interface HybridBaySummary {
  readonly bayId: number;
  readonly kind: Kind;
  readonly feeder: boolean;
  /** Per-token RMS of incoming residual stream X_{k-1} [T]. */
  readonly rmsIn: Float32Array;
  /** Per-token RMS of layer update Δ_k = Y_k - X_{k-1} [T]. */
  readonly rmsDelta: Float32Array;
  /** Per-token RMS of outgoing residual stream X_k = Y_k [T]. */
  readonly rmsOut: Float32Array;
  /** For MoE (`E`) bays only: Top-6 active expert IDs per token `[T * 6]`. */
  readonly moeTopkIndices?: Int32Array;
}

export interface HybridPrediction {
  readonly tokenId: number;
  readonly label: string;
  readonly logit: number;
  readonly prob: number;
}

export type HybridLayerTrace =
  | { readonly executionId: string; readonly bayId: number; readonly kind: 'M'; readonly sim: MambaSimResult }
  | { readonly executionId: string; readonly bayId: number; readonly kind: 'E'; readonly sim: MoeSimResult }
  | { readonly executionId: string; readonly bayId: number; readonly kind: '*'; readonly sim: GqaSimResult };

export interface HybridFinalNormResult {
  readonly eps: number;
  readonly gamma: Float32Array;
  readonly rms: Float32Array;
  readonly X_norm: Float32Array;
  readonly U_final: Float32Array;
}

export interface HybridLmHeadResult {
  readonly W_lm_head: Float32Array;
  readonly logits: Float32Array; // [vocabSize, T] indexed [v * T + t]
  readonly probs: Float32Array;  // [vocabSize, T] indexed [v * T + t]
  readonly topKByToken: readonly (readonly HybridPrediction[])[];
}

export interface HybridExecutionResult {
  readonly modelId: string;
  readonly parameterFingerprint: string;
  readonly executionId: string;
  readonly tokens: number;
  readonly d: 8;
  readonly vocabSize: number;
  readonly vocab: readonly string[];
  readonly tokenIds: Int32Array;
  readonly tokenLabels: readonly string[];
  /** Untied Embedding output X_0 [d, T] (same values as residualStreams[0]). */
  readonly X_embed: Float32Array;
  /** All `L + 1` (= 53) residual checkpoints: `residualStreams[0] = X_embed`, `residualStreams[k] = Y_k`. */
  readonly residualStreams: readonly Float32Array[];
  /** Per-bay residual telemetry across all `L` (= 52) layers. */
  readonly baySummaries: readonly HybridBaySummary[];
  /** Final RMSNorm (ε = 1e-5) applied to `residualStreams[L]`. */
  readonly finalNorm: HybridFinalNormResult;
  /** Untied LM Head projection & next-token softmax distribution. */
  readonly lmHead: HybridLmHeadResult;
  /** Deterministic access to ANY bay's full simulator trace (`1..L`), lazily reconstructed if not retained. */
  getLayerTrace(bayId: number): HybridLayerTrace;
  /** Typed Mamba-2 trace accessor for a Mamba bay (`kind === 'M'`). */
  getMambaTrace(bayId: number): MambaSimResult;
  /** Typed MoE trace accessor for an MoE bay (`kind === 'E'`). */
  getMoeTrace(bayId: number): MoeSimResult;
  /** Typed GQA trace accessor for a GQA bay (`kind === '*'`). */
  getGqaTrace(bayId: number): GqaSimResult;
  /** IDs of bays whose full traces are currently materialized in memory. */
  retainedTraceBayIds(): readonly number[];
  /** Exact retained typed-array bytes held by this execution result (streams + summaries + retained traces + output). */
  retainedTypedArrayBytes(): number;
}

/** Convenience helper producing default token IDs `[0, 1, ..., T-1]` for length `T in [1..16]`. */
export function defaultTokenIdsForLength(tokens: number): number[] {
  const T = Math.max(1, Math.min(16, Math.round(tokens)));
  const ids: number[] = [];
  for (let t = 0; t < T; t++) ids.push(t);
  return ids;
}

/**
 * Build canonical deterministic execution ID distinguishing model display ID, complete parameter
 * fingerprint (prevents collisions even when display `modelId` is duplicated), token IDs, and labels.
 */
export function computeHybridExecutionId(
  modelId: string,
  tokenIds: readonly number[] | Int32Array,
  tokenLabels: readonly string[],
  parameterFingerprint?: string,
): string {
  const idPart = Array.from(tokenIds).join(',');
  const labelPart = tokenLabels.join('\u001f');
  // Compact FNV-1a hash of labels to keep key concise while strictly collision-safe for label overrides
  let h = 2166136261 >>> 0;
  for (let i = 0; i < labelPart.length; i++) {
    h ^= labelPart.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  const fpPart = parameterFingerprint ? `|fp:${parameterFingerprint}` : '';
  return `${modelId}${fpPart}|t[${idPart}]|l:${h.toString(16)}`;
}

function computeStreamRms(stream: Float32Array, d: number, T: number): Float32Array {
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let sumSq = 0;
    for (let i = 0; i < d; i++) {
      const v = stream[i * T + t];
      sumSq += v * v;
    }
    out[t] = Math.fround(Math.sqrt(sumSq / d));
  }
  return out;
}

function computeDeltaRms(
  prev: Float32Array,
  next: Float32Array,
  d: number,
  T: number,
): Float32Array {
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let sumSq = 0;
    for (let i = 0; i < d; i++) {
      const diff = next[i * T + t] - prev[i * T + t];
      sumSq += diff * diff;
    }
    out[t] = Math.fround(Math.sqrt(sumSq / d));
  }
  return out;
}

function addUniqueArrayBytes(seen: Set<ArrayBufferLike>, arr: ArrayBufferView | undefined): number {
  if (!arr || seen.has(arr.buffer)) return 0;
  seen.add(arr.buffer);
  return arr.buffer.byteLength;
}

function collectTraceUniqueBytes(seen: Set<ArrayBufferLike>, trace: HybridLayerTrace): number {
  if (trace.kind === 'M') {
    const s = trace.sim;
    return (
      addUniqueArrayBytes(seen, s.tokenIds) +
      addUniqueArrayBytes(seen, s.X) +
      addUniqueArrayBytes(seen, s.X_mod) +
      addUniqueArrayBytes(seen, s.X_norm) +
      addUniqueArrayBytes(seen, s.rms_in) +
      addUniqueArrayBytes(seen, s.U) +
      addUniqueArrayBytes(seen, s.z) +
      addUniqueArrayBytes(seen, s.xBC_pre) +
      addUniqueArrayBytes(seen, s.dt_raw) +
      addUniqueArrayBytes(seen, s.conv_buf) +
      addUniqueArrayBytes(seen, s.conv_out) +
      addUniqueArrayBytes(seen, s.xBC) +
      addUniqueArrayBytes(seen, s.x) +
      addUniqueArrayBytes(seen, s.B) +
      addUniqueArrayBytes(seen, s.C) +
      addUniqueArrayBytes(seen, s.A) +
      addUniqueArrayBytes(seen, s.dt) +
      addUniqueArrayBytes(seen, s.dA) +
      addUniqueArrayBytes(seen, s.H_prev) +
      addUniqueArrayBytes(seen, s.H) +
      addUniqueArrayBytes(seen, s.H_history) +
      addUniqueArrayBytes(seen, s.y_ssm) +
      addUniqueArrayBytes(seen, s.y) +
      addUniqueArrayBytes(seen, s.gated) +
      addUniqueArrayBytes(seen, s.rms_gate) +
      addUniqueArrayBytes(seen, s.y_norm_pre) +
      addUniqueArrayBytes(seen, s.y_norm) +
      addUniqueArrayBytes(seen, s.Mamba_out) +
      addUniqueArrayBytes(seen, s.residualGate) +
      addUniqueArrayBytes(seen, s.Y)
    );
  }
  if (trace.kind === 'E') {
    const s = trace.sim;
    return (
      addUniqueArrayBytes(seen, s.tokenIds) +
      addUniqueArrayBytes(seen, s.X) +
      addUniqueArrayBytes(seen, s.rms) +
      addUniqueArrayBytes(seen, s.X_norm) +
      addUniqueArrayBytes(seen, s.U) +
      addUniqueArrayBytes(seen, s.U_mod) +
      addUniqueArrayBytes(seen, s.router_logits) +
      addUniqueArrayBytes(seen, s.scores) +
      addUniqueArrayBytes(seen, s.scores_for_choice) +
      addUniqueArrayBytes(seen, s.router_logits_f64) +
      addUniqueArrayBytes(seen, s.router_scores_f64) +
      addUniqueArrayBytes(seen, s.scores_for_choice_f64) +
      addUniqueArrayBytes(seen, s.topk_indices) +
      addUniqueArrayBytes(seen, s.topk_raw_scores) +
      addUniqueArrayBytes(seen, s.topk_norm_probs) +
      addUniqueArrayBytes(seen, s.topk_weights) +
      addUniqueArrayBytes(seen, s.active_h) +
      addUniqueArrayBytes(seen, s.active_a) +
      addUniqueArrayBytes(seen, s.active_y) +
      addUniqueArrayBytes(seen, s.shared_h) +
      addUniqueArrayBytes(seen, s.shared_a) +
      addUniqueArrayBytes(seen, s.y_shared) +
      addUniqueArrayBytes(seen, s.y_routed) +
      addUniqueArrayBytes(seen, s.y_moe) +
      addUniqueArrayBytes(seen, s.residualGate) +
      addUniqueArrayBytes(seen, s.Y)
    );
  }
  const s = trace.sim;
  return (
    addUniqueArrayBytes(seen, s.tokenIds) +
    addUniqueArrayBytes(seen, s.X) +
    addUniqueArrayBytes(seen, s.X_mod) +
    addUniqueArrayBytes(seen, s.X_norm) +
    addUniqueArrayBytes(seen, s.rms) +
    addUniqueArrayBytes(seen, s.U) +
    addUniqueArrayBytes(seen, s.Q) +
    addUniqueArrayBytes(seen, s.K) +
    addUniqueArrayBytes(seen, s.V) +
    addUniqueArrayBytes(seen, s.Scores) +
    addUniqueArrayBytes(seen, s.Softmax) +
    addUniqueArrayBytes(seen, s.V_out) +
    addUniqueArrayBytes(seen, s.Concat) +
    addUniqueArrayBytes(seen, s.Attn_out) +
    addUniqueArrayBytes(seen, s.residualGate) +
    addUniqueArrayBytes(seen, s.Y)
  );
}

export function runSingleBay(
  model: HybridModelParameters,
  executionId: string,
  bayId: number,
  X_in: Float32Array,
  T: number,
  tokenIds: Int32Array,
  tokenLabels: readonly string[],
): HybridLayerTrace {
  const weights = model.getBayWeights(bayId);
  const eps = model.config.rmsNormEps;

  if (weights.kind === 'M') {
    const fixture: MambaSimFixture = {
      X: X_in,
      gamma_in: weights.gamma_in,
      W_in: weights.W_in,
      w_conv: weights.w_conv,
      b_conv: weights.b_conv,
      A_log: weights.A_log,
      dt_bias: weights.dt_bias,
      D: weights.D,
      gamma_gate: weights.gamma_gate,
      W_out: weights.W_out,
      eps,
      tokenIds,
      tokenLabels,
    };
    return { executionId, bayId, kind: 'M', sim: runMambaSim(T, fixture) };
  }

  if (weights.kind === 'E') {
    const fixture: MoeSimFixture = {
      X: X_in,
      gamma_in: weights.gamma_in,
      W_router: weights.W_router,
      e_score_correction_bias: weights.e_score_correction_bias,
      W_up: weights.W_up,
      W_down: weights.W_down,
      W_up_s: weights.W_up_s,
      W_down_s: weights.W_down_s,
      rmsNormEps: eps,
      routerPrecisionPolicy: model.config.routerPrecisionPolicy,
      tokenIds,
      tokenLabels,
    };
    return { executionId, bayId, kind: 'E', sim: runMoeSim(T, fixture) };
  }

  const fixture: GqaSimFixture = {
    X: X_in,
    gamma: weights.gamma,
    W_Q: weights.W_Q,
    W_K: weights.W_K,
    W_V: weights.W_V,
    W_O: weights.W_O,
    rmsEps: eps,
    tokenIds,
    tokenLabels,
  };
  return { executionId, bayId, kind: '*', sim: runGqaSim(T, fixture) };
}

/**
 * Pure channel-major RMSNorm (`[d, T]`, `eps`), matching NVIDIA `NemotronHRMSNorm`.
 * Exact pre-gamma RMS accounts for `eps`: `mean(X_norm^2) = mean(X^2) / (mean(X^2) + eps)`.
 */
export function applyRmsNorm(
  X: Float32Array,
  gamma: Float32Array,
  d: number,
  T: number,
  eps: number,
): { readonly rms: Float32Array; readonly X_norm: Float32Array; readonly U: Float32Array } {
  const rms = new Float32Array(T);
  const X_norm = new Float32Array(d * T);
  const U = new Float32Array(d * T);
  for (let t = 0; t < T; t++) {
    let sumSq = 0;
    for (let i = 0; i < d; i++) {
      const v = X[i * T + t];
      sumSq += v * v;
    }
    const r = Math.fround(Math.sqrt(sumSq / d + eps));
    rms[t] = r;
    for (let i = 0; i < d; i++) {
      const xn = Math.fround(X[i * T + t] / r);
      X_norm[i * T + t] = xn;
      U[i * T + t] = Math.fround(xn * gamma[i]);
    }
  }
  return { rms, X_norm, U };
}

/** Pure NoPE token embedding lookup: `X_embed[i, t] = W_embed[tokenIds[t], i]`. */
export function embedTokens(
  tokenIds: readonly number[] | Int32Array,
  W_embed: Float32Array,
  d: number,
): Float32Array {
  const T = tokenIds.length;
  const X_embed = new Float32Array(d * T);
  for (let t = 0; t < T; t++) {
    const embRow = tokenIds[t] * d;
    for (let i = 0; i < d; i++) {
      X_embed[i * T + t] = W_embed[embRow + i];
    }
  }
  return X_embed;
}

/** Pure untied LM head linear projection (no bias) + numerically stable vocabulary softmax. */
export function runUntiedLmHead(
  U_final: Float32Array,
  W_lm_head: Float32Array,
  vocab: readonly string[],
  d: number,
  T: number,
  topK = 5,
): HybridLmHeadResult {
  const vocabSize = vocab.length;
  const logits = new Float32Array(vocabSize * T);
  const probs = new Float32Array(vocabSize * T);
  const topKByToken: HybridPrediction[][] = [];

  for (let t = 0; t < T; t++) {
    let maxLogit = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      const wRow = v * d;
      let acc = 0;
      for (let i = 0; i < d; i++) {
        acc += W_lm_head[wRow + i] * U_final[i * T + t];
      }
      const lv = Math.fround(acc);
      logits[v * T + t] = lv;
      if (lv > maxLogit) maxLogit = lv;
    }

    let sumExp = 0;
    for (let v = 0; v < vocabSize; v++) {
      const e = Math.exp(logits[v * T + t] - maxLogit);
      probs[v * T + t] = e;
      sumExp += e;
    }
    const preds: HybridPrediction[] = [];
    for (let v = 0; v < vocabSize; v++) {
      const pv = Math.fround(probs[v * T + t] / sumExp);
      probs[v * T + t] = pv;
      preds.push({
        tokenId: v,
        label: vocab[v],
        logit: logits[v * T + t],
        prob: pv,
      });
    }
    preds.sort((a, b) => (b.logit !== a.logit ? b.logit - a.logit : a.tokenId - b.tokenId));
    topKByToken.push(preds.slice(0, topK));
  }

  return { W_lm_head, logits, probs, topKByToken };
}

/**
 * Execute a full deterministic miniature forward pass over `model` for `input.tokenIds`:
 *   Untied Embedding (NoPE) -> 52-Layer Hybrid Residual Chain -> Final RMSNorm (ε = 1e-5) -> Untied LM Head.
 */
export function runHybridExecution(
  model: HybridModelParameters,
  input: HybridExecutionInput,
): HybridExecutionResult {
  const { d, vocabSize, pattern, rmsNormEps } = model.config;
  const numLayers = pattern.length;
  const rawIds = input.tokenIds;
  if (rawIds.length < 1 || rawIds.length > 16) {
    throw new RangeError(`Expected 1..16 tokenIds, received ${rawIds.length}`);
  }
  const T = rawIds.length;

  const tokenIds = new Int32Array(T);
  const tokenLabels: string[] = [];
  for (let t = 0; t < T; t++) {
    const id = Math.max(0, Math.min(vocabSize - 1, Math.trunc(rawIds[t])));
    tokenIds[t] = id;
    tokenLabels.push(input.tokenLabels?.[t] ?? model.vocab[id] ?? `id:${id}`);
  }
  const frozenLabels = Object.freeze(tokenLabels);
  const executionId = computeHybridExecutionId(
    model.modelId,
    tokenIds,
    frozenLabels,
    model.parameterFingerprint,
  );

  // Stage 0: Untied Token Embedding Lookup (strictly NoPE — zero positional embedding added)
  const X_embed = embedTokens(tokenIds, model.W_embed, d);

  const residualStreams: Float32Array[] = [X_embed];
  const baySummaries: HybridBaySummary[] = [];

  const eagerSet = new Set<number>(input.retainedTraceBays ?? DEFAULT_RETAINED_TRACE_BAYS);
  const maxRetainedTraces = Math.max(eagerSet.size, input.maxRetainedTraces ?? 12);
  const retainedTraces = new Map<number, HybridLayerTrace>();

  // Stages 1..L: Sequential Hybrid Residual Chain (exact HYBRID_OVERRIDE_PATTERN)
  let currentX = X_embed;
  let currentRms = computeStreamRms(currentX, d, T);

  for (let bayId = 1; bayId <= numLayers; bayId++) {
    const trace = runSingleBay(model, executionId, bayId, currentX, T, tokenIds, frozenLabels);
    const nextX = trace.sim.Y;
    const nextRms = computeStreamRms(nextX, d, T);
    const deltaRms = computeDeltaRms(currentX, nextX, d, T);

    const isFeeder =
      bayId <= NEMOTRON_SPEC.bays.length && pattern === HYBRID_OVERRIDE_PATTERN
        ? NEMOTRON_SPEC.bays[bayId - 1].feeder
        : trace.kind === 'M' && bayId < numLayers && pattern[bayId] === '*';

    const summary: HybridBaySummary =
      trace.kind === 'E'
        ? {
            bayId,
            kind: 'E',
            feeder: false,
            rmsIn: currentRms,
            rmsDelta: deltaRms,
            rmsOut: nextRms,
            moeTopkIndices: new Int32Array(trace.sim.topk_indices),
          }
        : {
            bayId,
            kind: trace.kind,
            feeder: isFeeder,
            rmsIn: currentRms,
            rmsDelta: deltaRms,
            rmsOut: nextRms,
          };

    baySummaries.push(summary);
    residualStreams.push(nextX);

    if (eagerSet.has(bayId)) {
      retainedTraces.set(bayId, trace);
    }

    currentX = nextX;
    currentRms = nextRms;
  }

  // Stage L+1: Final RMSNorm (eps = rmsNormEps = 1e-5)
  const finalX = residualStreams[numLayers];
  const { rms: finalRms, X_norm: finalXNorm, U: finalU } = applyRmsNorm(
    finalX,
    model.finalNormGamma,
    d,
    T,
    rmsNormEps,
  );

  // Stage L+2: Strictly Untied LM Head (W_lm_head [32, 8] @ U_final [8, T]) + Softmax
  const lmHead = runUntiedLmHead(finalU, model.W_lm_head, model.vocab, d, T, 5);
  const { logits, probs } = lmHead;

  const getLayerTrace = (bayId: number): HybridLayerTrace => {
    if (!Number.isInteger(bayId) || bayId < 1 || bayId > numLayers) {
      throw new RangeError(`Bay ID ${bayId} out of bounds [1..${numLayers}]`);
    }
    const existing = retainedTraces.get(bayId);
    if (existing) {
      retainedTraces.delete(bayId);
      retainedTraces.set(bayId, existing);
      return existing;
    }
    // Deterministic lazy reconstruction from preserved residual checkpoint residualStreams[bayId - 1]
    const rebuilt = runSingleBay(
      model,
      executionId,
      bayId,
      residualStreams[bayId - 1],
      T,
      tokenIds,
      frozenLabels,
    );
    retainedTraces.set(bayId, rebuilt);
    while (retainedTraces.size > maxRetainedTraces) {
      let evictId: number | null = null;
      for (const k of retainedTraces.keys()) {
        if (!eagerSet.has(k)) {
          evictId = k;
          break;
        }
      }
      if (evictId === null) {
        evictId = retainedTraces.keys().next().value ?? null;
      }
      if (evictId === null || evictId === bayId) break;
      retainedTraces.delete(evictId);
    }
    return rebuilt;
  };

  const getMambaTrace = (bayId: number): MambaSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== 'M') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected Mamba-2 ("M")`);
    }
    return tr.sim;
  };

  const getMoeTrace = (bayId: number): MoeSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== 'E') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected MoE ("E")`);
    }
    return tr.sim;
  };

  const getGqaTrace = (bayId: number): GqaSimResult => {
    const tr = getLayerTrace(bayId);
    if (tr.kind !== '*') {
      throw new TypeError(`Bay ${bayId} has kind "${tr.kind}", expected GQA ("*")`);
    }
    return tr.sim;
  };

  const retainedTraceBayIds = (): readonly number[] => Array.from(retainedTraces.keys()).sort((a, b) => a - b);

  const retainedTypedArrayBytes = (): number => {
    const seen = new Set<ArrayBufferLike>();
    let bytes =
      addUniqueArrayBytes(seen, tokenIds) +
      addUniqueArrayBytes(seen, finalRms) +
      addUniqueArrayBytes(seen, finalXNorm) +
      addUniqueArrayBytes(seen, finalU) +
      addUniqueArrayBytes(seen, logits) +
      addUniqueArrayBytes(seen, probs);
    for (const s of residualStreams) bytes += addUniqueArrayBytes(seen, s);
    for (const b of baySummaries) {
      bytes +=
        addUniqueArrayBytes(seen, b.rmsIn) +
        addUniqueArrayBytes(seen, b.rmsDelta) +
        addUniqueArrayBytes(seen, b.rmsOut) +
        addUniqueArrayBytes(seen, b.moeTopkIndices);
    }
    for (const tr of retainedTraces.values()) {
      bytes += collectTraceUniqueBytes(seen, tr);
    }
    return bytes;
  };

  return {
    modelId: model.modelId,
    parameterFingerprint: model.parameterFingerprint,
    executionId,
    tokens: T,
    d,
    vocabSize,
    vocab: model.vocab,
    tokenIds,
    tokenLabels: frozenLabels,
    X_embed,
    residualStreams,
    baySummaries,
    finalNorm: {
      eps: rmsNormEps,
      gamma: model.finalNormGamma,
      rms: finalRms,
      X_norm: finalXNorm,
      U_final: finalU,
    },
    lmHead,
    getLayerTrace,
    getMambaTrace,
    getMoeTrace,
    getGqaTrace,
    retainedTraceBayIds,
    retainedTypedArrayBytes,
  };
}

/**
 * Bounded LRU cache of `HybridExecutionResult`s keyed by canonical `executionId`
 * (which incorporates `modelId`, `tokenIds`, and `tokenLabels`).
 * Prevents unbounded memory accumulation under arbitrary custom token sequences and
 * guarantees zero collisions across distinct `HybridModelParameters` instances.
 */
export class HybridExecutionCache {
  readonly maxEntries: number;
  private readonly entries = new Map<string, HybridExecutionResult>();

  constructor(maxEntries = 16) {
    this.maxEntries = Math.max(1, Math.trunc(maxEntries));
  }

  get size(): number {
    return this.entries.size;
  }

  keys(): readonly string[] {
    return Array.from(this.entries.keys());
  }

  has(executionId: string): boolean {
    return this.entries.has(executionId);
  }

  get(executionId: string): HybridExecutionResult | undefined {
    const hit = this.entries.get(executionId);
    if (!hit) return undefined;
    // Refresh LRU recency
    this.entries.delete(executionId);
    this.entries.set(executionId, hit);
    return hit;
  }

  set(result: HybridExecutionResult): void {
    if (this.entries.has(result.executionId)) {
      this.entries.delete(result.executionId);
    }
    this.entries.set(result.executionId, result);
    while (this.entries.size > this.maxEntries) {
      const oldestKey = this.entries.keys().next().value;
      if (oldestKey === undefined) break;
      this.entries.delete(oldestKey);
    }
  }

  clear(): void {
    this.entries.clear();
  }

  totalRetainedTypedArrayBytes(): number {
    let total = 0;
    for (const exec of this.entries.values()) {
      total += exec.retainedTypedArrayBytes();
    }
    return total;
  }
}

/**
 * Fetch a cached execution or run and cache it inside the caller-owned bounded `cache`.
 */
export function getOrRunHybridExecution(
  model: HybridModelParameters,
  cache: HybridExecutionCache,
  input: HybridExecutionInput,
): HybridExecutionResult {
  const { vocabSize } = model.config;
  const T = input.tokenIds.length;
  if (T < 1 || T > 16) {
    throw new RangeError(`Expected 1..16 tokenIds, received ${T}`);
  }
  const normalizedIds: number[] = [];
  const resolvedLabels: string[] = [];
  for (let t = 0; t < T; t++) {
    const id = Math.max(0, Math.min(vocabSize - 1, Math.trunc(input.tokenIds[t])));
    normalizedIds.push(id);
    resolvedLabels.push(input.tokenLabels?.[t] ?? model.vocab[id] ?? `id:${id}`);
  }
  const key = computeHybridExecutionId(
    model.modelId,
    normalizedIds,
    resolvedLabels,
    model.parameterFingerprint,
  );
  const hit = cache.get(key);
  if (hit) return hit;

  const exec = runHybridExecution(model, {
    ...input,
    tokenIds: normalizedIds,
    tokenLabels: resolvedLabels,
  });
  cache.set(exec);
  return exec;
}
