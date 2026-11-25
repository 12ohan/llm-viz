/**
 * Released Confidence-Unmasking Sampler for Nemotron-Labs-TwoTower (`src/twotower/sampler.ts`)
 *
 * Clean-room mathematical implementation of the MDLM / SUBS masked block-diffusion sampling step
 * (zero shift-by-1: logits row `j` directly predicts clean token `x_0[P + j]`):
 *   1. `mdlmConstrainLogProbs`: suppresses `maskTokenId` via `-1e12` pre-softmax (`xTheta[mask] = 0`)
 *      and clamps already-unmasked positions to exact carry-over delta (`log = 0 / -1e12`, `p = 1 / 0`).
 *   2. `sampleBlockPredictions`: greedy argmax when `temperature <= 0`, or deterministic Gumbel-max
 *      using caller-provided uniform variates (`NEVER Math.random()`) when `temperature > 0`.
 *      Locked unmasked positions always preserve `xt[j]`.
 *   3. `runConfidenceUnmaskStep`: scores confidence strictly from UNSCALED `xTheta` even when
 *      `temperature > 0`, enforces strict `>` threshold comparison, progress floor (`>= 1`), and
 *      remaining-step budget floor (`ceil(nMasked / remainingSteps)`), and monotonically remasks
 *      only the lowest-confidence currently-masked slots (never overwriting locked tokens).
 *   4. `blockContainsEos`: post-block EOS check supporting scalar or array EOS IDs.
 *
 * Pure headless TypeScript — zero DOM, WebGL, scene, or render imports.
 */

export const MDLM_MASK_SUPPRESS_LOGIT = -1e12 as const;
export const GUMBEL_UNIFORM_MIN = 1e-10 as const;

export type ConfidenceCommitReason =
  | 'early_completed'
  | 'final_step'
  | 'threshold'
  | 'progress_floor'
  | 'budget_floor';

export interface MdlmConstrainedProbs {
  /** Unscaled constrained log-probabilities `[B, vocabSize]` (row-major). */
  readonly logXTheta: Float32Array;
  /** Unscaled constrained probabilities `[B, vocabSize]` (`xTheta[maskTokenId] === 0`). */
  readonly xTheta: Float32Array;
}

export interface SampleBlockPredictionsResult {
  /** Predicted/sampled token IDs for every slot `[0..B-1]` (locked slots remain `xt[j]`). */
  readonly predicted: readonly number[];
  /** Unscaled constrained log-probabilities `[B, vocabSize]`. */
  readonly logXTheta: Float32Array;
  /** Unscaled constrained probabilities `[B, vocabSize]`. */
  readonly xTheta: Float32Array;
}

export interface ConfidenceUnmaskStepInput {
  /** Current noisy block token IDs of length `B >= 1`. */
  readonly xt: readonly number[];
  /** Unshifted denoiser logits `[B, vocabSize]` (row-major, Float32). */
  readonly logits: Float32Array;
  readonly vocabSize: number;
  readonly maskTokenId: number;
  /** Zero-based step index in `[0, stepsPerBlock - 1]`. */
  readonly stepIndex: number;
  /** Total steps budgeted for this block (`K >= 1`, independently configurable from `B`). */
  readonly stepsPerBlock: number;
  /** Confidence threshold `γ` (strict `>` comparison against unscaled `xTheta`). */
  readonly confidenceThreshold: number;
  /** `<= 0` for greedy argmax; `> 0` for Gumbel-max with explicit `uniformVariates`. */
  readonly temperature: number;
  /** Required when `temperature > 0` and masked positions exist; length `B * vocabSize` in `[0, 1)`. */
  readonly uniformVariates?: Float32Array | readonly number[];
}

export interface ConfidenceUnmaskStepResult {
  readonly xtBefore: readonly number[];
  readonly xtNext: readonly number[];
  readonly predicted: readonly number[];
  /** Unscaled confidence per slot (`xTheta[j, predicted[j]]` if masked, `+Infinity` if already locked). */
  readonly confidence: Float32Array;
  readonly logXTheta: Float32Array;
  /** Unscaled constrained probabilities `[B, vocabSize]` (`xTheta`). */
  readonly unscaledProbs: Float32Array;
  /** Masked fraction `nMaskedBefore / B` fed into timestep conditioning for this step. */
  readonly tModel: number;
  readonly stepIndex: number;
  readonly stepsPerBlock: number;
  readonly remainingSteps: number;
  readonly nMaskedBefore: number;
  readonly nMaskedAfter: number;
  readonly numAboveThreshold: number;
  readonly minCommitFloor: number;
  readonly tokensCommittedCount: number;
  readonly commitReason: ConfidenceCommitReason;
  /** Newly committed block indices (ascending position order). */
  readonly committedIndices: readonly number[];
  /** Subset of `committedIndices` whose unscaled confidence strictly exceeded `confidenceThreshold`. */
  readonly thresholdCommittedIndices: readonly number[];
  /** Subset of `committedIndices` forced by progress floor, budget floor, or final step. */
  readonly floorCommittedIndices: readonly number[];
  /** Currently masked indices remaining masked after this step (ascending position order). */
  readonly remaskedIndices: readonly number[];
  /** True when the block had 0 masked slots (no-op) or finished early before the final step. */
  readonly earlyCompleted: boolean;
  /** True when `nMaskedAfter === 0`. */
  readonly isBlockComplete: boolean;
  /** Convenience post-step EOS check on `xtNext` (supports scalar or array EOS IDs). */
  containsEos(eosTokenIds: number | readonly number[]): boolean;
}

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

function validateLogitsAndTokens(
  logits: Float32Array,
  xt: readonly number[],
  vocabSize: number,
  maskTokenId: number,
): number {
  if (!Number.isInteger(vocabSize) || vocabSize < 2) {
    throw new RangeError(`vocabSize must be an integer >= 2, received ${vocabSize}`);
  }
  if (!Number.isInteger(maskTokenId) || maskTokenId < 0 || maskTokenId >= vocabSize) {
    throw new RangeError(
      `maskTokenId must be an integer in [0, ${vocabSize - 1}], received ${maskTokenId}`,
    );
  }
  const B = xt.length;
  if (B <= 0) {
    throw new RangeError(`xt must contain at least 1 block position, received length ${B}`);
  }
  const expectedLen = B * vocabSize;
  if (logits.length !== expectedLen) {
    throw new RangeError(
      `logits expected length ${expectedLen} (B=${B} * vocabSize=${vocabSize}), received ${logits.length}`,
    );
  }
  for (let j = 0; j < B; j++) {
    const tok = xt[j];
    if (!Number.isInteger(tok) || tok < 0 || tok >= vocabSize) {
      throw new RangeError(
        `xt[${j}] must be an integer token ID in [0, ${vocabSize - 1}], received ${tok}`,
      );
    }
  }
  for (let i = 0; i < logits.length; i++) {
    if (!Number.isFinite(logits[i])) {
      throw new RangeError(`Non-finite logit at index ${i}: ${logits[i]}`);
    }
  }
  return B;
}

function validateUniformVariates(
  uniformVariates: Float32Array | readonly number[] | undefined,
  expectedLen: number,
): Float32Array | readonly number[] {
  if (uniformVariates === undefined) {
    throw new RangeError(
      `uniformVariates of length ${expectedLen} is required when temperature > 0 (Math.random is forbidden)`,
    );
  }
  if (uniformVariates.length !== expectedLen) {
    throw new RangeError(
      `uniformVariates expected length ${expectedLen}, received ${uniformVariates.length}`,
    );
  }
  for (let i = 0; i < expectedLen; i++) {
    const u = uniformVariates[i];
    if (!Number.isFinite(u) || u < 0 || u >= 1) {
      throw new RangeError(
        `uniformVariates[${i}] must be a finite value in [0, 1), received ${u}`,
      );
    }
  }
  return uniformVariates;
}

function computeRowLogSoftmaxF64(
  logits: Float32Array,
  rowStart: number,
  vocabSize: number,
  maskTokenId: number,
  invTemp: number,
): Float64Array {
  const scaled = new Float64Array(vocabSize);
  let maxVal = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    const val = v === maskTokenId ? MDLM_MASK_SUPPRESS_LOGIT : logits[rowStart + v] * invTemp;
    if (!Number.isFinite(val)) {
      throw new RangeError(
        `Temperature-scaled logit overflowed to non-finite value at vocab index ${v}: ${val}`,
      );
    }
    scaled[v] = val;
    if (val > maxVal) {
      maxVal = val;
    }
  }

  let sumExp = 0.0;
  for (let v = 0; v < vocabSize; v++) {
    sumExp += Math.exp(scaled[v] - maxVal);
  }
  const lse = maxVal + Math.log(sumExp);

  const logProbs = new Float64Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    logProbs[v] = scaled[v] - lse;
  }
  return logProbs;
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Apply SUBS / MDLM log-probability constraints (unscaled `temperature = 1.0`):
 *   - Locked unmasked positions (`xt[j] !== maskTokenId`) receive an exact carry-over delta:
 *     `logXTheta = -1e12` everywhere except `0.0` at `xt[j]` (`xTheta = 1.0` at `xt[j]`, `0.0` elsewhere).
 *   - Masked positions (`xt[j] === maskTokenId`) suppress `maskTokenId` to `-1e12` before
 *     numerically stable `log_softmax`, guaranteeing `xTheta[maskTokenId] === 0`.
 */
export function mdlmConstrainLogProbs(
  logits: Float32Array,
  xt: readonly number[],
  vocabSize: number,
  maskTokenId: number,
): MdlmConstrainedProbs {
  const B = validateLogitsAndTokens(logits, xt, vocabSize, maskTokenId);
  const logXTheta = new Float32Array(B * vocabSize);
  const xTheta = new Float32Array(B * vocabSize);
  const suppressF32 = Math.fround(MDLM_MASK_SUPPRESS_LOGIT);

  for (let j = 0; j < B; j++) {
    const rowStart = j * vocabSize;
    const curTok = xt[j];

    if (curTok !== maskTokenId) {
      // Exact carry-over delta for locked unmasked positions
      for (let v = 0; v < vocabSize; v++) {
        logXTheta[rowStart + v] = suppressF32;
        xTheta[rowStart + v] = 0;
      }
      logXTheta[rowStart + curTok] = 0;
      xTheta[rowStart + curTok] = 1;
    } else {
      const logProbs = computeRowLogSoftmaxF64(logits, rowStart, vocabSize, maskTokenId, 1.0);
      for (let v = 0; v < vocabSize; v++) {
        const lp = Math.fround(logProbs[v]);
        logXTheta[rowStart + v] = lp;
        xTheta[rowStart + v] = v === maskTokenId ? 0 : Math.fround(Math.exp(logProbs[v]));
      }
    }
  }

  return Object.freeze({ logXTheta, xTheta });
}

/**
 * Predict block tokens from unshifted denoiser logits while preserving unscaled `logXTheta` & `xTheta`.
 *   - `temperature <= 0`: deterministic greedy `argmax` with lower-index tie-break.
 *   - `temperature > 0`: deterministic Gumbel-max over `log_softmax(masked_logits / temperature)`
 *     using explicit `uniformVariates` of length `B * vocabSize` (NEVER calls `Math.random()`).
 *   - Locked unmasked positions always strictly return `xt[j]`.
 */
export function sampleBlockPredictions(
  logits: Float32Array,
  xt: readonly number[],
  vocabSize: number,
  maskTokenId: number,
  temperature: number,
  uniformVariates?: Float32Array | readonly number[],
): SampleBlockPredictionsResult {
  if (!Number.isFinite(temperature)) {
    throw new RangeError(`temperature must be finite, received ${temperature}`);
  }

  const { logXTheta, xTheta } = mdlmConstrainLogProbs(logits, xt, vocabSize, maskTokenId);
  const B = xt.length;
  const predicted = new Array<number>(B);

  if (temperature <= 0) {
    for (let j = 0; j < B; j++) {
      const curTok = xt[j];
      if (curTok !== maskTokenId) {
        predicted[j] = curTok;
        continue;
      }
      const rowStart = j * vocabSize;
      let bestV = -1;
      let bestVal = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        if (v === maskTokenId) continue;
        const val = logXTheta[rowStart + v];
        if (val > bestVal) {
          bestVal = val;
          bestV = v;
        }
      }
      predicted[j] = bestV;
    }
  } else {
    const uniforms = validateUniformVariates(uniformVariates, B * vocabSize);
    const invTemp = 1.0 / temperature;

    for (let j = 0; j < B; j++) {
      const curTok = xt[j];
      if (curTok !== maskTokenId) {
        // Carry-over delta guarantees locked unmasked positions remain strictly xt[j]
        predicted[j] = curTok;
        continue;
      }
      const rowStart = j * vocabSize;
      const scaledLog = computeRowLogSoftmaxF64(logits, rowStart, vocabSize, maskTokenId, invTemp);
      let bestV = -1;
      let bestScore = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        if (v === maskTokenId) continue;
        const clampedU = Math.max(uniforms[rowStart + v], GUMBEL_UNIFORM_MIN);
        const gumbel = -Math.log(-Math.log(clampedU));
        const score = scaledLog[v] + gumbel;
        if (score > bestScore) {
          bestScore = score;
          bestV = v;
        }
      }
      predicted[j] = bestV;
    }
  }

  return Object.freeze({
    predicted: Object.freeze(predicted),
    logXTheta,
    xTheta,
  });
}

/**
 * Check whether any token in a committed block matches a scalar or array EOS token ID.
 */
export function blockContainsEos(
  tokens: readonly number[],
  eosTokenIds: number | readonly number[],
): boolean {
  if (typeof eosTokenIds === 'number') {
    for (let i = 0; i < tokens.length; i++) {
      if (tokens[i] === eosTokenIds) return true;
    }
    return false;
  }
  if (eosTokenIds.length === 0) return false;
  for (let i = 0; i < tokens.length; i++) {
    const tok = tokens[i];
    for (let k = 0; k < eosTokenIds.length; k++) {
      if (tok === eosTokenIds[k]) return true;
    }
  }
  return false;
}

/**
 * Execute one confidence-based unmasking step on block `xt` using unshifted denoiser `logits`.
 */
export function runConfidenceUnmaskStep(
  input: ConfidenceUnmaskStepInput,
): ConfidenceUnmaskStepResult {
  const {
    xt,
    logits,
    vocabSize,
    maskTokenId,
    stepIndex,
    stepsPerBlock,
    confidenceThreshold,
    temperature,
    uniformVariates,
  } = input;

  const B = validateLogitsAndTokens(logits, xt, vocabSize, maskTokenId);
  if (!Number.isInteger(stepsPerBlock) || stepsPerBlock < 1) {
    throw new RangeError(`stepsPerBlock must be an integer >= 1, received ${stepsPerBlock}`);
  }
  if (!Number.isInteger(stepIndex) || stepIndex < 0 || stepIndex >= stepsPerBlock) {
    throw new RangeError(
      `stepIndex must be an integer in [0, ${stepsPerBlock - 1}], received ${stepIndex}`,
    );
  }
  if (!Number.isFinite(confidenceThreshold)) {
    throw new RangeError(
      `confidenceThreshold must be finite, received ${confidenceThreshold}`,
    );
  }
  if (!Number.isFinite(temperature)) {
    throw new RangeError(`temperature must be finite, received ${temperature}`);
  }

  const xtBefore = Object.freeze([...xt]);
  const isMasked = new Array<boolean>(B);
  const maskedIndices: number[] = [];
  for (let j = 0; j < B; j++) {
    const masked = xt[j] === maskTokenId;
    isMasked[j] = masked;
    if (masked) maskedIndices.push(j);
  }

  const nMaskedBefore = maskedIndices.length;
  const remainingSteps = Math.max(1, stepsPerBlock - stepIndex);
  const tModel = nMaskedBefore / B;

  // Early-exit no-op when the block already has zero masked positions
  if (nMaskedBefore === 0) {
    const { logXTheta, xTheta } = mdlmConstrainLogProbs(logits, xt, vocabSize, maskTokenId);
    const confidence = new Float32Array(B);
    confidence.fill(Infinity);
    const xtNext = Object.freeze([...xt]);
    const predicted = Object.freeze([...xt]);

    return Object.freeze({
      xtBefore,
      xtNext,
      predicted,
      confidence,
      logXTheta,
      unscaledProbs: xTheta,
      tModel: 0,
      stepIndex,
      stepsPerBlock,
      remainingSteps,
      nMaskedBefore: 0,
      nMaskedAfter: 0,
      numAboveThreshold: 0,
      minCommitFloor: 0,
      tokensCommittedCount: 0,
      commitReason: 'early_completed',
      committedIndices: Object.freeze([]),
      thresholdCommittedIndices: Object.freeze([]),
      floorCommittedIndices: Object.freeze([]),
      remaskedIndices: Object.freeze([]),
      earlyCompleted: true,
      isBlockComplete: true,
      containsEos: (eosTokenIds: number | readonly number[]) =>
        blockContainsEos(xtNext, eosTokenIds),
    });
  }

  const { predicted, logXTheta, xTheta } = sampleBlockPredictions(
    logits,
    xt,
    vocabSize,
    maskTokenId,
    temperature,
    uniformVariates,
  );

  // Confidence is ALWAYS gathered from unscaled xTheta even when temperature > 0
  const confidence = new Float32Array(B);
  let numAboveThreshold = 0;
  for (let j = 0; j < B; j++) {
    if (isMasked[j]) {
      const conf = xTheta[j * vocabSize + predicted[j]];
      confidence[j] = conf;
      // Strict '>' comparison: conf === threshold does NOT count as above
      if (conf > confidenceThreshold) {
        numAboveThreshold++;
      }
    } else {
      confidence[j] = Infinity;
    }
  }

  const minCommitFloor = Math.ceil(nMaskedBefore / remainingSteps);
  let tokensToCommit: number;
  let commitReason: ConfidenceCommitReason;

  if (stepIndex >= stepsPerBlock - 1) {
    tokensToCommit = nMaskedBefore;
    commitReason = 'final_step';
  } else {
    const c1 = Math.max(numAboveThreshold, 1);
    tokensToCommit = Math.min(nMaskedBefore, Math.max(c1, minCommitFloor));
    if (numAboveThreshold >= Math.max(1, minCommitFloor)) {
      commitReason = 'threshold';
    } else if (minCommitFloor > 1) {
      commitReason = 'budget_floor';
    } else {
      commitReason = 'progress_floor';
    }
  }

  // Monotonic update: locked unmasked positions are NEVER overwritten
  const nextTokens = new Array<number>(B);
  for (let j = 0; j < B; j++) {
    nextTokens[j] = isMasked[j] ? predicted[j] : xt[j];
  }

  // Sort ONLY currently-masked indices by ascending confidence (deterministic lower-index tie-break)
  const sortedMaskedAsc = [...maskedIndices].sort((a, b) => {
    const diff = confidence[a] - confidence[b];
    if (diff !== 0) return diff;
    return a - b;
  });

  const numToRemask = nMaskedBefore - tokensToCommit;
  const remasked = sortedMaskedAsc.slice(0, numToRemask);
  const committed = sortedMaskedAsc.slice(numToRemask);

  for (let i = 0; i < remasked.length; i++) {
    nextTokens[remasked[i]] = maskTokenId;
  }

  // Sort output index lists in ascending block-slot order for clean deterministic inspection
  remasked.sort((a, b) => a - b);
  committed.sort((a, b) => a - b);

  const thresholdCommitted: number[] = [];
  const floorCommitted: number[] = [];
  for (let i = 0; i < committed.length; i++) {
    const idx = committed[i];
    if (confidence[idx] > confidenceThreshold) {
      thresholdCommitted.push(idx);
    } else {
      floorCommitted.push(idx);
    }
  }

  const nMaskedAfter = nMaskedBefore - tokensToCommit;
  const isBlockComplete = nMaskedAfter === 0;
  const earlyCompleted = isBlockComplete && stepIndex < stepsPerBlock - 1;
  const xtNext = Object.freeze(nextTokens);

  return Object.freeze({
    xtBefore,
    xtNext,
    predicted,
    confidence,
    logXTheta,
    unscaledProbs: xTheta,
    tModel,
    stepIndex,
    stepsPerBlock,
    remainingSteps,
    nMaskedBefore,
    nMaskedAfter,
    numAboveThreshold,
    minCommitFloor,
    tokensCommittedCount: tokensToCommit,
    commitReason,
    committedIndices: Object.freeze(committed),
    thresholdCommittedIndices: Object.freeze(thresholdCommitted),
    floorCommittedIndices: Object.freeze(floorCommitted),
    remaskedIndices: Object.freeze(remasked),
    earlyCompleted,
    isBlockComplete,
    containsEos: (eosTokenIds: number | readonly number[]) =>
      blockContainsEos(xtNext, eosTokenIds),
  });
}
