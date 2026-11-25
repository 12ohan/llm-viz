/**
 * Independent Miniature TwoTower Parameters (`src/twotower/parameters.ts`)
 *
 * Constructs strictly independent miniature parameter sets for Nemotron-Labs-TwoTower:
 *   1. `contextModel`: causal 52-layer hybrid Context Tower (`V = 34`, `'float64_reference'`).
 *   2. `denoiserModel`: independent 52-layer Denoiser Tower constructed cleanly via deterministic
 *      in-generation additive perturbation (NEVER post-construction mutation), guaranteeing zero
 *      typed-array buffer aliasing with `contextModel`.
 *   3. `timeConditioning`: denoiser-only PixArt-alpha `adaLN-single` timestep conditioning weights
 *      (`d = 8`, `freqDim = 16`, `numLayers = 52`).
 *
 * Educational miniature Float32/Float64 simulator — NOT 63.19B checkpoint weights.
 * Pure headless TypeScript — zero DOM, WebGL, scene, or render imports.
 */

import {
  DEFAULT_HYBRID_MODEL_SEED,
  HYBRID_MINI_D,
  HYBRID_UNIFIED_RMS_EPS,
  MINI_VOCAB,
  computeModelParameterFingerprint,
  createHybridModelParameters,
  type HybridModelParameters,
  type HybridWeightPerturbation,
  type RouterPrecisionPolicy,
} from '../math/hybridFlow';
import { HYBRID_OVERRIDE_PATTERN } from '../model/spec';
import {
  collectTimeConditioningTensors,
  createTimeConditioningWeights,
  type TimeConditioningWeights,
} from './modulation';
import { TWOTOWER_MINI_CONFIG } from './spec';

/**
 * Miniature 34-token TwoTower vocabulary:
 *   - `0..31`: exact `MINI_VOCAB` tokens from Nano v1.0.0
 *   - `32`: `'[MASK]'` masked diffusion sentinel
 *   - `33`: `'<eos>'` post-block stop sentinel
 */
export const TWOTOWER_MINI_VOCAB: readonly string[] = Object.freeze([
  ...MINI_VOCAB,
  '[MASK]',
  '<eos>',
]);

export const TWOTOWER_MINI_MASK_TOKEN_ID = 32 as const;
export const TWOTOWER_MINI_EOS_TOKEN_ID = 33 as const;

export const DEFAULT_TWOTOWER_CONTEXT_SEED = DEFAULT_HYBRID_MODEL_SEED;
export const DEFAULT_TWOTOWER_TIME_COND_SEED = 0x54543031 as const; // "TT01"

export const TWOTOWER_CONTEXT_GENERATION_POLICY = 'twotower-context-v1' as const;
export const TWOTOWER_DENOISER_GENERATION_POLICY = 'twotower-denoiser-v1' as const;

/** Default deterministic additive perturbation deriving independent Denoiser weights from Context seed. */
export const DEFAULT_TWOTOWER_PERTURBATION = Object.freeze({
  seed: 20260701,
  scale: 0.15,
} as const);

export interface TwoTowerCanonicalWeightTensor {
  readonly name: string;
  readonly data: Float32Array;
}

export interface TwoTowerPerturbationConfig {
  readonly seed: number;
  readonly scale: number;
}

export interface CreateTwoTowerModelOptions {
  readonly modelId?: string;
  readonly contextModelId?: string;
  readonly denoiserModelId?: string;
  /** Base PRNG seed shared by Context Tower and unperturbed Denoiser base (defaults to `0x4e454d33`). */
  readonly seed?: number;
  /** Optional explicit alias overriding `seed`. */
  readonly contextSeed?: number;
  /** Layer pattern (defaults to 52-layer `HYBRID_OVERRIDE_PATTERN`). */
  readonly pattern?: string;
  /** Custom vocabulary table (defaults to 34-token `TWOTOWER_MINI_VOCAB`). */
  readonly vocab?: readonly string[];
  /** Mask sentinel token ID (defaults to `32`). */
  readonly maskTokenId?: number;
  /** EOS stop token ID (defaults to `33`). */
  readonly eosTokenId?: number;
  /** MoE router precision policy (defaults to `'float64_reference'`). */
  readonly routerPrecisionPolicy?: RouterPrecisionPolicy;
  /** Bounded retained weight bay count per tower (defaults to `52` / `pattern.length`). */
  readonly maxRetainedWeightBays?: number;
  /** Deterministic Denoiser perturbation seed (defaults to `20260701`). */
  readonly perturbationSeed?: number;
  /** Deterministic Denoiser perturbation scale (defaults to `0.15`; `0` enables identity-weight parity). */
  readonly perturbationScale?: number;
  /** Seed for denoiser-only timestep conditioning MLP + `scale_shift_tables` (defaults to `0x54543031`). */
  readonly timeConditioningSeed?: number;
  /** Sinusoidal frequency embedding width (defaults to `16`). */
  readonly freqDim?: number;
  /** RMSNorm epsilon (defaults to `1e-5`). */
  readonly rmsNormEps?: number;
  /** Optional residual output scale override. */
  readonly residualOutputScale?: number;
  /** Generation policy tag for Context Tower (defaults to `'twotower-context-v1'`). */
  readonly contextGenerationPolicy?: string;
  /** Generation policy tag for Denoiser Tower (defaults to `'twotower-denoiser-v1'`). */
  readonly denoiserGenerationPolicy?: string;
}

export interface TwoTowerModelConfig {
  readonly modelId: string;
  readonly twoTowerFingerprint: string;
  readonly contextSeed: number;
  readonly timeConditioningSeed: number;
  readonly pattern: string;
  readonly d: 8;
  readonly vocabSize: number;
  readonly maskTokenId: number;
  readonly eosTokenId: number;
  readonly freqDim: number;
  readonly routerPrecisionPolicy: RouterPrecisionPolicy;
  readonly perturbation: TwoTowerPerturbationConfig;
  readonly contextGenerationPolicy: string;
  readonly denoiserGenerationPolicy: string;
}

export interface TwoTowerModelParameters {
  readonly modelId: string;
  readonly twoTowerFingerprint: string;
  readonly config: TwoTowerModelConfig;
  readonly vocab: readonly string[];
  readonly maskTokenId: number;
  readonly eosTokenId: number;
  readonly contextModel: HybridModelParameters;
  readonly denoiserModel: HybridModelParameters;
  readonly timeConditioning: TimeConditioningWeights;
  /** Exact typed-array byte count currently retained across Context, Denoiser, and Time Conditioning. */
  retainedWeightBytes(): number;
}

/**
 * Compute a deterministic 64-bit hex fingerprint over the complete dual-tower parameterization:
 * Context fingerprint, Denoiser fingerprint, Time-Conditioning fingerprint, special token IDs,
 * perturbation configuration, and tower generation policies.
 */
export function computeTwoTowerFingerprint(params: {
  readonly contextParameterFingerprint: string;
  readonly denoiserParameterFingerprint: string;
  readonly timeConditioningFingerprint: string;
  readonly maskTokenId: number;
  readonly eosTokenId: number;
  readonly perturbationSeed: number;
  readonly perturbationScale: number;
  readonly contextGenerationPolicy: string;
  readonly denoiserGenerationPolicy: string;
}): string {
  let h1 = 0x811c9dc5 >>> 0;
  let h2 = 0x9e3779b9 >>> 0;
  const dv = new DataView(new ArrayBuffer(8));

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

  feedStr('twotower-model-v1');
  feedStr(params.contextParameterFingerprint);
  feedStr(params.denoiserParameterFingerprint);
  feedStr(params.timeConditioningFingerprint);
  feedU32(params.maskTokenId);
  feedU32(params.eosTokenId);
  feedU32(params.perturbationSeed);
  feedF64(params.perturbationScale);
  feedStr(params.contextGenerationPolicy);
  feedStr(params.denoiserGenerationPolicy);

  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

function collectTowerCanonicalWeights(
  prefix: 'context' | 'denoiser',
  tower: HybridModelParameters,
  out: TwoTowerCanonicalWeightTensor[],
): void {
  out.push(
    { name: `${prefix}.W_embed`, data: tower.W_embed },
    { name: `${prefix}.finalNormGamma`, data: tower.finalNormGamma },
    { name: `${prefix}.W_lm_head`, data: tower.W_lm_head },
  );

  const numLayers = tower.config.pattern.length;
  for (let bayId = 1; bayId <= numLayers; bayId++) {
    const w = tower.getBayWeights(bayId);
    const p = `${prefix}.bay_${String(bayId).padStart(2, '0')}`;
    if (w.kind === 'M') {
      out.push(
        { name: `${p}.gamma_in`, data: w.gamma_in },
        { name: `${p}.W_in`, data: w.W_in },
        { name: `${p}.w_conv`, data: w.w_conv },
        { name: `${p}.b_conv`, data: w.b_conv },
        { name: `${p}.A_log`, data: w.A_log },
        { name: `${p}.dt_bias`, data: w.dt_bias },
        { name: `${p}.D`, data: w.D },
        { name: `${p}.gamma_gate`, data: w.gamma_gate },
        { name: `${p}.W_out`, data: w.W_out },
      );
    } else if (w.kind === 'E') {
      out.push(
        { name: `${p}.gamma_in`, data: w.gamma_in },
        { name: `${p}.W_router`, data: w.W_router },
        { name: `${p}.e_score_correction_bias`, data: w.e_score_correction_bias },
        { name: `${p}.W_up`, data: w.W_up },
        { name: `${p}.W_down`, data: w.W_down },
        { name: `${p}.W_up_s`, data: w.W_up_s },
        { name: `${p}.W_down_s`, data: w.W_down_s },
      );
    } else {
      out.push(
        { name: `${p}.gamma`, data: w.gamma },
        { name: `${p}.W_Q`, data: w.W_Q },
        { name: `${p}.W_K`, data: w.W_K },
        { name: `${p}.W_V`, data: w.W_V },
        { name: `${p}.W_O`, data: w.W_O },
      );
    }
  }
}

/**
 * Canonical ordered list of every Float32 weight tensor owned by a `TwoTowerModelParameters`
 * instance (`context.*`, `denoiser.*`, `time_cond.*`) for deterministic binary export and
 * cryptographic SHA-256 cross-runtime parity verification.
 */
export function collectTwoTowerCanonicalWeights(
  model: TwoTowerModelParameters,
): TwoTowerCanonicalWeightTensor[] {
  const out: TwoTowerCanonicalWeightTensor[] = [];
  collectTowerCanonicalWeights('context', model.contextModel, out);
  collectTowerCanonicalWeights('denoiser', model.denoiserModel, out);
  const timeTensors = collectTimeConditioningTensors(model.timeConditioning);
  for (let i = 0; i < timeTensors.length; i++) {
    out.push({
      name: `time_cond.${timeTensors[i].name}`,
      data: timeTensors[i].data,
    });
  }
  return out;
}

/**
 * Create an independent, immutable-by-contract miniature `TwoTowerModelParameters` instance
 * with strictly non-aliased Context and Denoiser parameter buffers and denoiser-only adaLN-single
 * timestep conditioning weights.
 */
export function createTwoTowerModelParameters(
  options?: CreateTwoTowerModelOptions,
): TwoTowerModelParameters {
  const rawSeed = options?.contextSeed ?? options?.seed ?? DEFAULT_TWOTOWER_CONTEXT_SEED;
  if (!Number.isFinite(rawSeed)) {
    throw new RangeError(`Invalid TwoTower context seed: ${rawSeed}`);
  }
  const contextSeed = rawSeed >>> 0;

  const pattern = options?.pattern ?? HYBRID_OVERRIDE_PATTERN;
  const numLayers = pattern.length;
  if (numLayers < 1) {
    throw new RangeError('TwoTower model pattern must contain at least 1 layer');
  }

  const vocab = options?.vocab ? Object.freeze([...options.vocab]) : TWOTOWER_MINI_VOCAB;
  const vocabSize = vocab.length;
  if (vocabSize < 2) {
    throw new RangeError(`TwoTower vocab must contain at least 2 tokens, received ${vocabSize}`);
  }

  const maskTokenId = options?.maskTokenId ?? TWOTOWER_MINI_MASK_TOKEN_ID;
  const eosTokenId = options?.eosTokenId ?? TWOTOWER_MINI_EOS_TOKEN_ID;
  if (!Number.isInteger(maskTokenId) || maskTokenId < 0 || maskTokenId >= vocabSize) {
    throw new RangeError(
      `Invalid maskTokenId (${maskTokenId}): must be an integer in [0, ${vocabSize - 1}]`,
    );
  }
  if (!Number.isInteger(eosTokenId) || eosTokenId < 0 || eosTokenId >= vocabSize) {
    throw new RangeError(
      `Invalid eosTokenId (${eosTokenId}): must be an integer in [0, ${vocabSize - 1}]`,
    );
  }
  if (maskTokenId === eosTokenId) {
    throw new RangeError(
      `maskTokenId (${maskTokenId}) and eosTokenId (${eosTokenId}) must be distinct`,
    );
  }

  const routerPrecisionPolicy: RouterPrecisionPolicy =
    options?.routerPrecisionPolicy ?? 'float64_reference';
  const maxRetainedWeightBays = Math.max(
    1,
    Math.trunc(options?.maxRetainedWeightBays ?? Math.max(52, numLayers)),
  );

  const rawPertSeed = options?.perturbationSeed ?? DEFAULT_TWOTOWER_PERTURBATION.seed;
  if (!Number.isFinite(rawPertSeed)) {
    throw new RangeError(`Invalid perturbationSeed: ${rawPertSeed}`);
  }
  const perturbationSeed = rawPertSeed >>> 0;

  const perturbationScale = options?.perturbationScale ?? DEFAULT_TWOTOWER_PERTURBATION.scale;
  if (!Number.isFinite(perturbationScale) || perturbationScale < 0) {
    throw new RangeError(
      `Invalid perturbationScale (${perturbationScale}): must be finite >= 0`,
    );
  }

  const rawTimeSeed = options?.timeConditioningSeed ?? DEFAULT_TWOTOWER_TIME_COND_SEED;
  if (!Number.isInteger(rawTimeSeed)) {
    throw new RangeError(`Invalid timeConditioningSeed: ${rawTimeSeed}`);
  }
  const timeConditioningSeed = rawTimeSeed;

  const freqDim = options?.freqDim ?? TWOTOWER_MINI_CONFIG.freqDim;
  const rmsNormEps = options?.rmsNormEps ?? HYBRID_UNIFIED_RMS_EPS;
  const residualOutputScale =
    options?.residualOutputScale ?? 1.42 / Math.sqrt(Math.max(2, 2 * numLayers));

  const contextGenerationPolicy =
    options?.contextGenerationPolicy ?? TWOTOWER_CONTEXT_GENERATION_POLICY;
  const denoiserGenerationPolicy =
    options?.denoiserGenerationPolicy ?? TWOTOWER_DENOISER_GENERATION_POLICY;
  if (contextGenerationPolicy === denoiserGenerationPolicy) {
    throw new RangeError(
      `contextGenerationPolicy and denoiserGenerationPolicy must be distinct ("${contextGenerationPolicy}") to prevent cross-tower cache confusion`,
    );
  }

  const contextFpPreview = computeModelParameterFingerprint({
    seed: contextSeed,
    pattern,
    d: HYBRID_MINI_D,
    vocabSize,
    rmsNormEps,
    residualOutputScale,
    vocab,
    routerPrecisionPolicy,
    generationPolicy: contextGenerationPolicy,
  });
  const contextModelId =
    options?.contextModelId ??
    `twotower-context-mini:s${contextSeed.toString(16)}:L${numLayers}:fp${contextFpPreview}`;

  const contextModel = createHybridModelParameters({
    modelId: contextModelId,
    seed: contextSeed,
    pattern,
    vocab,
    routerPrecisionPolicy,
    maxRetainedWeightBays,
    rmsNormEps,
    residualOutputScale,
    generationPolicy: contextGenerationPolicy,
  });

  const perturbation: HybridWeightPerturbation | undefined =
    perturbationScale > 0
      ? Object.freeze({
          seed: perturbationSeed,
          scale: perturbationScale,
          sourceFingerprint: contextModel.config.parameterFingerprint,
        })
      : undefined;

  const denoiserFpPreview = computeModelParameterFingerprint({
    seed: contextSeed,
    pattern,
    d: HYBRID_MINI_D,
    vocabSize,
    rmsNormEps,
    residualOutputScale,
    vocab,
    routerPrecisionPolicy,
    perturbation,
    generationPolicy: denoiserGenerationPolicy,
  });
  const denoiserModelId =
    options?.denoiserModelId ??
    `twotower-denoiser-mini:s${contextSeed.toString(16)}:L${numLayers}:fp${denoiserFpPreview}`;

  // Denoiser Tower is constructed completely independently inside parameter generation
  // (NEVER via post-construction mutation of Context Tower weights).
  const denoiserModel = createHybridModelParameters({
    modelId: denoiserModelId,
    seed: contextSeed,
    pattern,
    vocab,
    routerPrecisionPolicy,
    maxRetainedWeightBays,
    rmsNormEps,
    residualOutputScale,
    perturbation,
    generationPolicy: denoiserGenerationPolicy,
  });

  const timeConditioning = createTimeConditioningWeights({
    d: HYBRID_MINI_D,
    freqDim,
    numLayers,
    seed: timeConditioningSeed,
  });

  const twoTowerFingerprint = computeTwoTowerFingerprint({
    contextParameterFingerprint: contextModel.config.parameterFingerprint,
    denoiserParameterFingerprint: denoiserModel.config.parameterFingerprint,
    timeConditioningFingerprint: timeConditioning.fingerprint,
    maskTokenId,
    eosTokenId,
    perturbationSeed,
    perturbationScale,
    contextGenerationPolicy,
    denoiserGenerationPolicy,
  });

  const modelId =
    options?.modelId ??
    `nemotron-twotower-mini:s${contextSeed.toString(16)}:L${numLayers}:fp${twoTowerFingerprint}`;

  const config: TwoTowerModelConfig = Object.freeze({
    modelId,
    twoTowerFingerprint,
    contextSeed,
    timeConditioningSeed,
    pattern,
    d: HYBRID_MINI_D,
    vocabSize,
    maskTokenId,
    eosTokenId,
    freqDim,
    routerPrecisionPolicy,
    perturbation: Object.freeze({
      seed: perturbationSeed,
      scale: perturbationScale,
    }),
    contextGenerationPolicy,
    denoiserGenerationPolicy,
  });

  const timeConditioningBytes = collectTimeConditioningTensors(timeConditioning).reduce(
    (sum, t) => sum + t.data.byteLength,
    0,
  );

  const retainedWeightBytes = (): number =>
    contextModel.retainedWeightBytes() +
    denoiserModel.retainedWeightBytes() +
    timeConditioningBytes;

  return Object.freeze({
    modelId,
    twoTowerFingerprint,
    config,
    vocab,
    maskTokenId,
    eosTokenId,
    contextModel,
    denoiserModel,
    timeConditioning,
    retainedWeightBytes,
  });
}

/** Create a fresh default miniature TwoTower parameter bundle (`d = 8`, `V = 34`, `L = 52`). */
export function createDefaultTwoTowerModel(): TwoTowerModelParameters {
  return createTwoTowerModelParameters();
}
