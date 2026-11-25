/**
 * Pure Float32 Timestep Embedding & PixArt-alpha `adaLN-single` Modulation
 * (`src/twotower/modulation.ts`)
 *
 * Clean-room mathematical implementation of Nemotron-Labs-TwoTower's denoiser-only timestep
 * conditioning chain:
 *   1. Sinusoidal frequency embedding with outer scaling `t_scaled = t * maxPeriodScale` (1000),
 *      `omega_k = exp(-ln(maxPeriod) * k / (freqDim / 2))` (maxPeriod = 10000), COS FIRST then SIN.
 *   2. Global 2-layer SiLU MLP (`freqDim -> d -> d`) producing `tRepr`.
 *   3. Shared modulation projection `tEmb = W_blk @ SiLU(tRepr) + b_blk` (`[3 * d]`).
 *   4. Per-layer table addition `scaleShiftTables[l] + tEmb` sliced into `(shift, scale, gate)`.
 *   5. Adaptive modulation `modulate(x, shift, scale) = x * (1 + scale) + shift`.
 *   6. Split layer-dependent normalization ordering:
 *      - Mamba (`M`) & Attention (`*`): `modulate_then_norm`
 *      - MoE (`E`): `norm_then_modulate`
 *
 * Pure headless TypeScript — zero DOM, WebGL, scene, or render imports.
 */

export type BayKind = 'M' | 'E' | '*';
export type ModulationOrder = 'modulate_then_norm' | 'norm_then_modulate';

export const DEFAULT_TIMESTEP_MAX_PERIOD_SCALE = 1000 as const;
export const DEFAULT_TIMESTEP_MAX_PERIOD = 10000 as const;

export interface TimeConditioningConfig {
  readonly d: number;
  readonly freqDim: number;
  readonly numLayers: number;
  readonly seed: number;
  readonly maxPeriodScale?: number;
  readonly maxPeriod?: number;
}

export interface ResolvedTimeConditioningConfig {
  readonly d: number;
  readonly freqDim: number;
  readonly numLayers: number;
  readonly seed: number;
  readonly maxPeriodScale: number;
  readonly maxPeriod: number;
}

export interface TimeConditioningWeights {
  readonly config: ResolvedTimeConditioningConfig;
  /** Row-major `[d, freqDim]` first linear layer weight (`t_embedder.mlp.0.weight`). */
  readonly mlp0Weight: Float32Array;
  /** `[d]` first linear layer bias (`t_embedder.mlp.0.bias`). */
  readonly mlp0Bias: Float32Array;
  /** Row-major `[d, d]` second linear layer weight (`t_embedder.mlp.2.weight`). */
  readonly mlp2Weight: Float32Array;
  /** `[d]` second linear layer bias (`t_embedder.mlp.2.bias`). */
  readonly mlp2Bias: Float32Array;
  /** Row-major `[3 * d, d]` shared modulation linear weight (`t_block.1.weight`). */
  readonly tBlockWeight: Float32Array;
  /** `[3 * d]` shared modulation linear bias (`t_block.1.bias`). */
  readonly tBlockBias: Float32Array;
  /** `numLayers` learned per-layer tables, each `[3 * d]` laid out as `[shift(d), scale(d), gate(d)]`. */
  readonly scaleShiftTables: readonly Float32Array[];
  /** Deterministic 64-bit hex fingerprint covering config and all IEEE-754 Float32 weight bits. */
  readonly fingerprint: string;
}

export interface GlobalTimestepProjection {
  /** Sinusoidal embedding `[freqDim]`, `[cos(0..half-1), sin(0..half-1)]`. */
  readonly phi: Float32Array;
  /** Output of `t_embedder.mlp` `[d]`. */
  readonly tRepr: Float32Array;
  /** Output of `t_block` `[3 * d]` (`[shift_global(d), scale_global(d), gate_global(d)]`). */
  readonly tEmb: Float32Array;
}

export interface LayerModulationParams {
  readonly layerIndex: number;
  readonly shift: Float32Array;
  readonly scale: Float32Array;
  readonly gate: Float32Array;
}

export interface CanonicalTimeConditioningTensor {
  readonly name: string;
  readonly data: Float32Array;
}

// ============================================================================
// DETERMINISTIC PRNG & VALIDATION HELPERS
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

function mixSeed(baseSeed: number, tag: number, salt: number): number {
  let h = (baseSeed ^ Math.imul(tag + 1, 0x9e3779b1) ^ Math.imul(salt, 0x85ebca6b)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

function resolveConfig(config: TimeConditioningConfig): ResolvedTimeConditioningConfig {
  const { d, freqDim, numLayers, seed } = config;
  const maxPeriodScale = config.maxPeriodScale ?? DEFAULT_TIMESTEP_MAX_PERIOD_SCALE;
  const maxPeriod = config.maxPeriod ?? DEFAULT_TIMESTEP_MAX_PERIOD;

  if (!Number.isInteger(d) || d <= 0) {
    throw new RangeError(`d must be a positive integer, received ${d}`);
  }
  if (!Number.isInteger(freqDim) || freqDim <= 0 || freqDim % 2 !== 0) {
    throw new RangeError(`freqDim must be a positive even integer, received ${freqDim}`);
  }
  if (!Number.isInteger(numLayers) || numLayers <= 0) {
    throw new RangeError(`numLayers must be a positive integer, received ${numLayers}`);
  }
  if (!Number.isInteger(seed)) {
    throw new RangeError(`seed must be a finite integer, received ${seed}`);
  }
  if (!Number.isFinite(maxPeriodScale) || maxPeriodScale <= 0) {
    throw new RangeError(`maxPeriodScale must be a positive finite number, received ${maxPeriodScale}`);
  }
  if (!Number.isFinite(maxPeriod) || maxPeriod <= 1) {
    throw new RangeError(`maxPeriod must be a finite number > 1, received ${maxPeriod}`);
  }

  return Object.freeze({
    d,
    freqDim,
    numLayers,
    seed,
    maxPeriodScale,
    maxPeriod,
  });
}

function assertFiniteFloat32(arr: Float32Array, expectedLen: number, label: string): void {
  if (arr.length !== expectedLen) {
    throw new RangeError(`${label} expected length ${expectedLen}, received ${arr.length}`);
  }
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) {
      throw new RangeError(`Non-finite value in ${label} at index ${i}: ${arr[i]}`);
    }
  }
}

function siluF32(x: number): number {
  return Math.fround(x / (1.0 + Math.exp(-x)));
}

function linearF32(
  input: Float32Array,
  weight: Float32Array,
  bias: Float32Array,
  outDim: number,
  inDim: number,
): Float32Array {
  const out = new Float32Array(outDim);
  for (let r = 0; r < outDim; r++) {
    let acc = 0.0;
    const rowOff = r * inDim;
    for (let c = 0; c < inDim; c++) {
      acc += weight[rowOff + c] * input[c];
    }
    const val = Math.fround(acc + bias[r]);
    if (!Number.isFinite(val)) {
      throw new RangeError(`Non-finite linear projection output at index ${r}: ${val}`);
    }
    out[r] = val;
  }
  return out;
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Exact verified layer-dependent normalization vs. adaLN modulation ordering in TwoTower diffusion:
 *   - `'M'` (Mamba-2) & `'*'` (GQA Attention): modulate BEFORE RMSNorm (`modulate_then_norm`)
 *   - `'E'` (MoE): RMSNorm BEFORE modulate (`norm_then_modulate`)
 */
export function modulationOrderForBayKind(kind: BayKind): ModulationOrder {
  switch (kind) {
    case 'M':
    case '*':
      return 'modulate_then_norm';
    case 'E':
      return 'norm_then_modulate';
    default: {
      const exhaustive: never = kind;
      throw new RangeError(`Unsupported bay kind: ${String(exhaustive)}`);
    }
  }
}

/**
 * Canonical ordered list of every Float32 tensor in `TimeConditioningWeights` for deterministic
 * binary export and SHA-256 cross-runtime parity proofs.
 */
export function collectTimeConditioningTensors(
  weights: Pick<
    TimeConditioningWeights,
    | 'mlp0Weight'
    | 'mlp0Bias'
    | 'mlp2Weight'
    | 'mlp2Bias'
    | 'tBlockWeight'
    | 'tBlockBias'
    | 'scaleShiftTables'
  >,
): CanonicalTimeConditioningTensor[] {
  const tensors: CanonicalTimeConditioningTensor[] = [
    { name: 't_embedder.mlp.0.weight', data: weights.mlp0Weight },
    { name: 't_embedder.mlp.0.bias', data: weights.mlp0Bias },
    { name: 't_embedder.mlp.2.weight', data: weights.mlp2Weight },
    { name: 't_embedder.mlp.2.bias', data: weights.mlp2Bias },
    { name: 't_block.1.weight', data: weights.tBlockWeight },
    { name: 't_block.1.bias', data: weights.tBlockBias },
  ];

  for (let l = 0; l < weights.scaleShiftTables.length; l++) {
    tensors.push({
      name: `scale_shift_tables.${String(l).padStart(2, '0')}`,
      data: weights.scaleShiftTables[l],
    });
  }

  return tensors;
}

/**
 * Compute a deterministic 64-bit hex fingerprint over the time-conditioning config and exact
 * little-endian IEEE-754 Float32 bits of every weight tensor.
 */
export function computeTimeConditioningFingerprint(
  config: TimeConditioningConfig,
  weights: Pick<
    TimeConditioningWeights,
    | 'mlp0Weight'
    | 'mlp0Bias'
    | 'mlp2Weight'
    | 'mlp2Bias'
    | 'tBlockWeight'
    | 'tBlockBias'
    | 'scaleShiftTables'
  >,
): string {
  const resolved = resolveConfig(config);
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
  const feedF32 = (v: number) => {
    dv.setFloat32(0, Object.is(v, -0) ? 0 : v, true);
    feedU32(dv.getUint32(0, true));
  };
  const feedStr = (s: string) => {
    feedU32(s.length);
    for (let i = 0; i < s.length; i++) {
      const code = s.charCodeAt(i);
      feedByte(code);
      feedByte(code >>> 8);
    }
  };

  feedStr('twotower-time-cond-v1');
  feedU32(resolved.d);
  feedU32(resolved.freqDim);
  feedU32(resolved.numLayers);
  feedU32(resolved.seed);
  feedF64(resolved.maxPeriodScale);
  feedF64(resolved.maxPeriod);

  const tensors = collectTimeConditioningTensors(weights);
  feedU32(tensors.length);
  for (const t of tensors) {
    feedStr(t.name);
    feedU32(t.data.length);
    for (let i = 0; i < t.data.length; i++) {
      feedF32(t.data[i]);
    }
  }

  return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}`;
}

/**
 * Deterministically synthesize well-conditioned Float32 weights for `t_embedder.mlp`, `t_block`,
 * and all `numLayers` per-layer `scale_shift_tables` (`~randn / sqrt(d)` matching reference init).
 */
export function createTimeConditioningWeights(
  config: TimeConditioningConfig,
): TimeConditioningWeights {
  const resolved = resolveConfig(config);
  const { d, freqDim, numLayers, seed } = resolved;

  const gMlp0 = makeGaussian(mixSeed(seed, 0, 0x51));
  const gMlp2 = makeGaussian(mixSeed(seed, 1, 0x52));
  const gBlock = makeGaussian(mixSeed(seed, 2, 0x53));

  const mlp0Scale = 1.0 / Math.sqrt(freqDim);
  const mlp0Weight = new Float32Array(d * freqDim);
  for (let i = 0; i < mlp0Weight.length; i++) {
    mlp0Weight[i] = Math.fround(gMlp0() * mlp0Scale);
  }
  const mlp0Bias = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    mlp0Bias[i] = Math.fround(0.05 * gMlp0());
  }

  const mlp2Scale = 1.0 / Math.sqrt(d);
  const mlp2Weight = new Float32Array(d * d);
  for (let i = 0; i < mlp2Weight.length; i++) {
    mlp2Weight[i] = Math.fround(gMlp2() * mlp2Scale);
  }
  const mlp2Bias = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    mlp2Bias[i] = Math.fround(0.05 * gMlp2());
  }

  const tBlockScale = 0.35 / Math.sqrt(d);
  const tBlockWeight = new Float32Array(3 * d * d);
  for (let i = 0; i < tBlockWeight.length; i++) {
    tBlockWeight[i] = Math.fround(gBlock() * tBlockScale);
  }
  const tBlockBias = new Float32Array(3 * d);
  for (let i = 0; i < d; i++) {
    tBlockBias[i] = Math.fround(0.05 * gBlock()); // shift bias
    tBlockBias[d + i] = Math.fround(0.05 * gBlock()); // scale bias
    tBlockBias[2 * d + i] = Math.fround(0.35 + 0.1 * gBlock()); // non-degenerate gate bias
  }

  // Reference initialization scale: randn(3, d) / sqrt(d) per layer
  const tableScale = 1.0 / Math.sqrt(d);
  const scaleShiftTables: Float32Array[] = [];
  for (let l = 0; l < numLayers; l++) {
    const gLayer = makeGaussian(mixSeed(seed, l + 16, 0x54));
    const table = new Float32Array(3 * d);
    for (let i = 0; i < 3 * d; i++) {
      table[i] = Math.fround(gLayer() * tableScale);
    }
    scaleShiftTables.push(table);
  }

  const frozenTables = Object.freeze(scaleShiftTables);
  const partial = {
    mlp0Weight,
    mlp0Bias,
    mlp2Weight,
    mlp2Bias,
    tBlockWeight,
    tBlockBias,
    scaleShiftTables: frozenTables,
  };
  const fingerprint = computeTimeConditioningFingerprint(resolved, partial);

  return Object.freeze({
    config: resolved,
    ...partial,
    fingerprint,
  });
}

/**
 * Compute sinusoidal timestep embedding with outer scaling (`t * maxPeriodScale`),
 * IEEE-754 Float32 frequency progression `exp(-ln(maxPeriod) * k / half)` and Float32
 * phase argument `tScaled * omega` (matching PyTorch `TimestepEmbedder` `dtype=torch.float32`),
 * and COS-FIRST ordering:
 *   `phi[0..half-1] = cos(tScaled * omega_k)`
 *   `phi[half..freqDim-1] = sin(tScaled * omega_k)`
 */
export function computeSinusoidalTimestepEmbedding(
  t: number,
  freqDim: number,
  maxPeriodScale: number = DEFAULT_TIMESTEP_MAX_PERIOD_SCALE,
  maxPeriod: number = DEFAULT_TIMESTEP_MAX_PERIOD,
): Float32Array {
  if (!Number.isFinite(t)) {
    throw new RangeError(`Timestep t must be finite, received ${t}`);
  }
  if (!Number.isInteger(freqDim) || freqDim <= 0 || freqDim % 2 !== 0) {
    throw new RangeError(`freqDim must be a positive even integer, received ${freqDim}`);
  }
  if (!Number.isFinite(maxPeriodScale) || maxPeriodScale <= 0) {
    throw new RangeError(`maxPeriodScale must be a positive finite number, received ${maxPeriodScale}`);
  }
  if (!Number.isFinite(maxPeriod) || maxPeriod <= 1) {
    throw new RangeError(`maxPeriod must be a finite number > 1, received ${maxPeriod}`);
  }

  const half = freqDim / 2;
  const tScaledF32 = Math.fround(Math.fround(t) * Math.fround(maxPeriodScale));
  const negLogMaxPeriodF32 = Math.fround(-Math.log(maxPeriod));
  const out = new Float32Array(freqDim);

  for (let k = 0; k < half; k++) {
    const exponentF32 = Math.fround(Math.fround(negLogMaxPeriodF32 * k) / half);
    const omegaF32 = Math.fround(Math.exp(exponentF32));
    const argF32 = Math.fround(tScaledF32 * omegaF32);
    out[k] = Math.fround(Math.cos(argF32));
    out[half + k] = Math.fround(Math.sin(argF32));
  }

  return out;
}

/**
 * Evaluate the global denoiser timestep conditioning backbone once per denoising step:
 *   `phi   = sinusoidal(t)` (`[freqDim]`)
 *   `tRepr = W2 @ SiLU(W1 @ phi + b1) + b2` (`[d]`)
 *   `tEmb  = Wblk @ SiLU(tRepr) + bblk` (`[3 * d]`)
 */
export function computeGlobalTimestepProjection(
  t: number,
  weights: TimeConditioningWeights,
): GlobalTimestepProjection {
  const { d, freqDim, maxPeriodScale, maxPeriod } = weights.config;
  assertFiniteFloat32(weights.mlp0Weight, d * freqDim, 'mlp0Weight');
  assertFiniteFloat32(weights.mlp0Bias, d, 'mlp0Bias');
  assertFiniteFloat32(weights.mlp2Weight, d * d, 'mlp2Weight');
  assertFiniteFloat32(weights.mlp2Bias, d, 'mlp2Bias');
  assertFiniteFloat32(weights.tBlockWeight, 3 * d * d, 'tBlockWeight');
  assertFiniteFloat32(weights.tBlockBias, 3 * d, 'tBlockBias');

  const phi = computeSinusoidalTimestepEmbedding(t, freqDim, maxPeriodScale, maxPeriod);

  const h0 = linearF32(phi, weights.mlp0Weight, weights.mlp0Bias, d, freqDim);
  const h0Act = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    h0Act[i] = siluF32(h0[i]);
  }

  const tRepr = linearF32(h0Act, weights.mlp2Weight, weights.mlp2Bias, d, d);
  const tReprAct = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    tReprAct[i] = siluF32(tRepr[i]);
  }

  const tEmb = linearF32(tReprAct, weights.tBlockWeight, weights.tBlockBias, 3 * d, d);

  return Object.freeze({ phi, tRepr, tEmb });
}

/**
 * Combine global `tEmb` (`[3 * d]`) with layer `layerIndex`'s learned `scaleShiftTables[layerIndex]`
 * in Float32 and slice into `[0..d) -> shift`, `[d..2d) -> scale`, `[2d..3d) -> gate`.
 */
export function computeLayerModulationParams(
  tEmb: Float32Array,
  weights: TimeConditioningWeights,
  layerIndex: number,
): LayerModulationParams {
  const { d } = weights.config;
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= weights.scaleShiftTables.length) {
    throw new RangeError(
      `layerIndex must be an integer in [0, ${weights.scaleShiftTables.length - 1}], received ${layerIndex}`,
    );
  }
  assertFiniteFloat32(tEmb, 3 * d, 'tEmb');
  const table = weights.scaleShiftTables[layerIndex];
  assertFiniteFloat32(table, 3 * d, `scaleShiftTables[${layerIndex}]`);

  const shift = new Float32Array(d);
  const scale = new Float32Array(d);
  const gate = new Float32Array(d);

  for (let i = 0; i < d; i++) {
    const s = Math.fround(table[i] + tEmb[i]);
    const sc = Math.fround(table[d + i] + tEmb[d + i]);
    const g = Math.fround(table[2 * d + i] + tEmb[2 * d + i]);
    if (!Number.isFinite(s) || !Number.isFinite(sc) || !Number.isFinite(g)) {
      throw new RangeError(`Non-finite modulation parameter at layer ${layerIndex}, channel ${i}`);
    }
    shift[i] = s;
    scale[i] = sc;
    gate[i] = g;
  }

  return Object.freeze({ layerIndex, shift, scale, gate });
}

/**
 * Apply adaptive layer modulation over a channel-major `[d, T]` Float32 tensor:
 *   `out[i * T + t] = Math.fround(x[i * T + t] * Math.fround(1 + scale[i]) + shift[i])`
 */
export function applyAdaLnModulate(
  x: Float32Array,
  d: number,
  T: number,
  shift: Float32Array,
  scale: Float32Array,
): Float32Array {
  if (!Number.isInteger(d) || d <= 0) {
    throw new RangeError(`d must be a positive integer, received ${d}`);
  }
  if (!Number.isInteger(T) || T <= 0) {
    throw new RangeError(`T must be a positive integer, received ${T}`);
  }
  assertFiniteFloat32(x, d * T, 'x');
  assertFiniteFloat32(shift, d, 'shift');
  assertFiniteFloat32(scale, d, 'scale');

  const out = new Float32Array(d * T);
  for (let i = 0; i < d; i++) {
    const scaleFactor = Math.fround(1.0 + scale[i]);
    const shiftVal = shift[i];
    const rowOff = i * T;
    for (let t = 0; t < T; t++) {
      const val = Math.fround(x[rowOff + t] * scaleFactor + shiftVal);
      if (!Number.isFinite(val)) {
        throw new RangeError(`Non-finite modulated output at channel ${i}, token ${t}: ${val}`);
      }
      out[rowOff + t] = val;
    }
  }
  return out;
}
