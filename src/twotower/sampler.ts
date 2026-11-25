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

export type TwoTowerConfidencePresetId =
  | 'mini_default'
  | 'paper_default'
  | 'reference_code_default';

export interface TwoTowerConfidencePreset {
  readonly id: TwoTowerConfidencePresetId;
  readonly threshold: number;
  readonly label: string;
  readonly shortLabel: string;
  readonly isCalibratedMiniatureDefault: boolean;
  readonly provenanceNote: string;
}

/**
 * Authoritative confidence-unmasking threshold (`γ`) presets separating the calibrated
 * educational miniature default (`γ = 0.22`) from the production paper (`0.80`) and
 * reference-code (`0.90`) defaults.
 */
export const TWOTOWER_CONFIDENCE_PRESETS: readonly TwoTowerConfidencePreset[] = Object.freeze([
  Object.freeze({
    id: 'mini_default' as const,
    threshold: 0.22,
    label: 'Miniature Educational Default (γ = 0.22)',
    shortLabel: 'Mini Default (0.22)',
    isCalibratedMiniatureDefault: true,
    provenanceNote:
      'Calibrated for the miniature synthetic d = 8, V = 34 educational simulator so multi-step confidence unmasking is visibly active; NOT a production checkpoint setting.',
  }),
  Object.freeze({
    id: 'paper_default' as const,
    threshold: 0.8,
    label: 'Paper Default (γ = 0.80)',
    shortLabel: 'Paper (0.80)',
    isCalibratedMiniatureDefault: false,
    provenanceNote:
      'Confidence threshold reported in arXiv:2606.26493v2 for production 30B-A3B masked block diffusion.',
  }),
  Object.freeze({
    id: 'reference_code_default' as const,
    threshold: 0.9,
    label: 'Reference Code Default (γ = 0.90)',
    shortLabel: 'Ref Code (0.90)',
    isCalibratedMiniatureDefault: false,
    provenanceNote:
      'Default confidence threshold in the official reference inference code (HF rev 35b64985).',
  }),
]);

export interface DeriveRefinementUniformVariatesParams {
  readonly samplingSeed: number;
  readonly sessionId: string;
  readonly branchId: string;
  readonly blockId: number;
  readonly refinementIndex: number;
  readonly blockSize: number;
  readonly vocabSize: number;
}

export interface DerivedRefinementUniformVariates {
  /** Deterministic Float32 uniform variates strictly in `[0, 1)`, length `blockSize * vocabSize`. */
  readonly variates: Float32Array;
  /** Canonical human-readable derivation key binding every input coordinate. */
  readonly derivationKey: string;
  /** Deterministic 64-bit hex digest covering `derivationKey` and all Float32 variate bits. */
  readonly variateDigest: string;
}

export interface RawLmHeadCandidate {
  readonly tokenId: number;
  readonly tokenLabel: string;
  readonly logit: number;
  readonly prob: number;
}

export interface ConstrainedMdlmCandidate {
  readonly tokenId: number;
  readonly tokenLabel: string;
  readonly logProb: number;
  readonly prob: number;
}

export interface SamplerRawLmHeadView {
  readonly top1TokenId: number;
  readonly top1TokenLabel: string;
  readonly top1Logit: number;
  readonly top1Prob: number;
  readonly maskTokenProb: number;
  readonly topK: readonly RawLmHeadCandidate[];
}

export interface SamplerConstrainedMdlmView {
  readonly maskTokenExcluded: true;
  readonly maskTokenProb: 0;
  readonly isCarryOverDelta: boolean;
  readonly topK: readonly ConstrainedMdlmCandidate[];
}

export type SamplerSelectionMode = 'locked_carry_over' | 'greedy_argmax' | 'gumbel_max';

export interface SamplerSelectionView {
  readonly samplingMode: SamplerSelectionMode;
  readonly temperature: number;
  readonly selectedTokenId: number;
  readonly selectedTokenLabel: string;
}

export type SamplerConfidenceDisplay =
  | {
      readonly kind: 'locked_prior';
      readonly unscaledConfidence: null;
      readonly displayText: 'Locked (prior step)';
      readonly exceedsThreshold: null;
    }
  | {
      readonly kind: 'unscaled_probability';
      readonly unscaledConfidence: number;
      readonly displayText: string;
      readonly exceedsThreshold: boolean;
    };

export type SamplerSlotDecision =
  | 'locked_prior'
  | 'committed_threshold'
  | 'committed_progress_floor'
  | 'committed_budget_floor'
  | 'committed_final_step'
  | 'remasked';

export interface SamplerSlotPresentation {
  readonly slotIndex: number;
  readonly absolutePosition: number | null;
  readonly tokenIdBefore: number;
  readonly tokenLabelBefore: string;
  readonly tokenIdAfter: number;
  readonly tokenLabelAfter: string;
  readonly wasMaskedBefore: boolean;
  readonly isMaskedAfter: boolean;
  readonly rawLmHead: SamplerRawLmHeadView;
  readonly constrainedMdlm: SamplerConstrainedMdlmView;
  readonly selection: SamplerSelectionView;
  readonly confidenceDisplay: SamplerConfidenceDisplay;
  readonly decision: SamplerSlotDecision;
}

export interface SamplerStepPresentation {
  readonly blockSize: number;
  readonly vocabSize: number;
  readonly maskTokenId: number;
  readonly stepIndex: number;
  readonly stepsPerBlock: number;
  readonly remainingSteps: number;
  readonly tModel: number;
  readonly confidenceThreshold: number;
  readonly temperature: number;
  readonly commitReason: ConfidenceCommitReason;
  readonly nMaskedBefore: number;
  readonly nMaskedAfter: number;
  readonly numAboveThreshold: number;
  readonly minCommitFloor: number;
  readonly tokensCommittedCount: number;
  readonly earlyCompleted: boolean;
  readonly isBlockComplete: boolean;
  readonly slots: readonly SamplerSlotPresentation[];
}

export interface BuildSamplerStepPresentationParams {
  /** Unconstrained raw Denoiser logits `[B, vocabSize]` (row-major). */
  readonly rawLogits: Float32Array;
  /** Unconstrained raw Denoiser softmax probabilities `[B, vocabSize]` (row-major). */
  readonly rawProbs: Float32Array;
  /** Step result returned by `runConfidenceUnmaskStep`. */
  readonly unmaskStep: ConfidenceUnmaskStepResult;
  readonly vocab: readonly string[];
  readonly maskTokenId: number;
  readonly confidenceThreshold: number;
  readonly temperature: number;
  /** Optional starting absolute token position `P` so each slot exposes `absolutePosition = P + j`. */
  readonly absoluteTokenStart?: number;
  /** Maximum top-K entries exposed per view (defaults to `5`). */
  readonly topK?: number;
}

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

// ============================================================================
// DETERMINISTIC KEYED UNIFORM VARIATE DERIVATION (ZERO Math.random)
// ============================================================================

function fmix32(h: number): number {
  let x = h >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

function rotl32(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/**
 * Derive deterministic, keyed Float32 uniform variates strictly in `[0, 1)` for a single
 * masked diffusion refinement step (ZERO `Math.random()`).
 *
 * - Bit-exact reproducible for identical `(samplingSeed, sessionId, branchId, blockId,
 *   refinementIndex, blockSize, vocabSize)` inputs.
 * - Changing ANY single input key produces a distinct length-prefixed hash state and
 *   distinct variate stream (`xoshiro128**` projected onto exact Float32 grid `[0, 1)`
 *   via `(u32 >>> 8) / 16777216`).
 */
export function deriveRefinementUniformVariates(
  params: DeriveRefinementUniformVariatesParams,
): DerivedRefinementUniformVariates {
  const {
    samplingSeed,
    sessionId,
    branchId,
    blockId,
    refinementIndex,
    blockSize,
    vocabSize,
  } = params;

  if (!Number.isInteger(samplingSeed)) {
    throw new RangeError(`samplingSeed must be a finite integer, received ${samplingSeed}`);
  }
  if (typeof sessionId !== 'string' || sessionId.length === 0) {
    throw new RangeError('sessionId must be a non-empty string');
  }
  if (typeof branchId !== 'string' || branchId.length === 0) {
    throw new RangeError('branchId must be a non-empty string');
  }
  if (!Number.isInteger(blockId) || blockId < 0) {
    throw new RangeError(`blockId must be a nonnegative integer, received ${blockId}`);
  }
  if (!Number.isInteger(refinementIndex) || refinementIndex < 0) {
    throw new RangeError(
      `refinementIndex must be a nonnegative integer, received ${refinementIndex}`,
    );
  }
  if (!Number.isInteger(blockSize) || blockSize < 1) {
    throw new RangeError(`blockSize must be an integer >= 1, received ${blockSize}`);
  }
  if (!Number.isInteger(vocabSize) || vocabSize < 2) {
    throw new RangeError(`vocabSize must be an integer >= 2, received ${vocabSize}`);
  }

  const derivationKey =
    `twotower-variates-v1|seed=${samplingSeed}|session=${sessionId}|branch=${branchId}` +
    `|block=${blockId}|ref=${refinementIndex}|B=${blockSize}|V=${vocabSize}`;

  // Four independent 32-bit keyed hash lanes with unambiguous length-prefixed framing
  let l0 = 0x811c9dc5 >>> 0;
  let l1 = 0x9e3779b9 >>> 0;
  let l2 = 0x243f6a88 >>> 0;
  let l3 = 0xb7e15162 >>> 0;

  const feedByte = (b: number) => {
    const byte = b & 0xff;
    l0 = Math.imul(l0 ^ byte, 0x01000193) >>> 0;
    l1 = Math.imul(l1 ^ ((byte + 0x5a) & 0xff), 0x85ebca6b) >>> 0;
    l2 = Math.imul(l2 ^ ((byte + 0xa5) & 0xff), 0xc2b2ae35) >>> 0;
    l3 = Math.imul(l3 ^ ((byte ^ 0x3c) & 0xff), 0x27d4eb2d) >>> 0;
  };
  const feedU32 = (v: number) => {
    const u = v >>> 0;
    feedByte(u);
    feedByte(u >>> 8);
    feedByte(u >>> 16);
    feedByte(u >>> 24);
  };
  const feedI32 = (v: number) => feedU32(v | 0);
  const feedStr = (s: string) => {
    feedU32(s.length);
    for (let i = 0; i < s.length; i++) {
      const c = s.charCodeAt(i);
      feedByte(c);
      feedByte(c >>> 8);
    }
  };

  feedStr('twotower-variates-v1');
  feedI32(samplingSeed);
  feedStr(sessionId);
  feedStr(branchId);
  feedU32(blockId);
  feedU32(refinementIndex);
  feedU32(blockSize);
  feedU32(vocabSize);

  let s0 = fmix32(l0 ^ 0x13579bdf);
  let s1 = fmix32(l1 ^ 0x2468ace0);
  let s2 = fmix32(l2 ^ 0xfdb97531);
  let s3 = fmix32(l3 ^ 0x0eca8642);
  if ((s0 | s1 | s2 | s3) === 0) {
    s0 = 0x9e3779b9;
    s1 = 0x85ebca6b;
  }

  const nextU32 = (): number => {
    const result = Math.imul(rotl32(Math.imul(s1, 5) >>> 0, 7), 9) >>> 0;
    const t = (s1 << 9) >>> 0;
    s2 = (s2 ^ s0) >>> 0;
    s3 = (s3 ^ s1) >>> 0;
    s1 = (s1 ^ s2) >>> 0;
    s0 = (s0 ^ s3) >>> 0;
    s2 = (s2 ^ t) >>> 0;
    s3 = rotl32(s3, 11);
    return result;
  };

  // Warm up 4 rounds so initial lanes are fully diffused
  nextU32();
  nextU32();
  nextU32();
  nextU32();

  const totalLen = blockSize * vocabSize;
  const variates = new Float32Array(totalLen);
  const f32Buf = new ArrayBuffer(4);
  const f32View = new Float32Array(f32Buf);
  const u32View = new Uint32Array(f32Buf);

  // Dual 32-bit digest over derivation key + exact Float32 variate bits
  let d1 = l0;
  let d2 = l1;
  const digestByte = (b: number) => {
    const byte = b & 0xff;
    d1 = Math.imul(d1 ^ byte, 0x01000193) >>> 0;
    d2 = Math.imul(d2 ^ ((byte + 0x5a) & 0xff), 0x85ebca6b) >>> 0;
  };
  const digestU32 = (v: number) => {
    const u = v >>> 0;
    digestByte(u);
    digestByte(u >>> 8);
    digestByte(u >>> 16);
    digestByte(u >>> 24);
  };

  for (let i = 0; i < totalLen; i++) {
    // Exact 24-bit Float32 grid in [0, 1): max value is 16777215 / 16777216 < 1.0
    const u = (nextU32() >>> 8) / 16777216;
    variates[i] = u;
    f32View[0] = u;
    digestU32(u32View[0]);
  }

  const variateDigest = `${d1.toString(16).padStart(8, '0')}${d2.toString(16).padStart(8, '0')}`;

  return Object.freeze({
    variates,
    derivationKey,
    variateDigest,
  });
}

// ============================================================================
// PRESENTATION-SAFE SAMPLER VIEW & NO-INFINITY GUARDRAIL
// ============================================================================

/**
 * Recursively verify that a presentation object (and every nested array, typed array, or
 * plain object) contains strictly ZERO `Infinity`, `-Infinity`, or `NaN` numeric values.
 */
export function assertPresentationSafeNoInfinity(
  value: unknown,
  path: string = 'presentation',
): void {
  const visited = new WeakSet<object>();

  const walk = (node: unknown, currentPath: string): void => {
    if (typeof node === 'number') {
      if (!Number.isFinite(node)) {
        throw new RangeError(
          `Presentation-unsafe non-finite number (${String(node)}) at ${currentPath}`,
        );
      }
      return;
    }
    if (node === null || typeof node !== 'object') {
      return;
    }
    if (visited.has(node)) {
      return;
    }
    visited.add(node);

    if (ArrayBuffer.isView(node)) {
      if (
        node instanceof Float32Array ||
        node instanceof Float64Array ||
        node instanceof Int8Array ||
        node instanceof Uint8Array ||
        node instanceof Int16Array ||
        node instanceof Uint16Array ||
        node instanceof Int32Array ||
        node instanceof Uint32Array
      ) {
        for (let i = 0; i < node.length; i++) {
          const elem = node[i];
          if (!Number.isFinite(elem)) {
            throw new RangeError(
              `Presentation-unsafe non-finite typed-array element (${String(elem)}) at ${currentPath}[${i}]`,
            );
          }
        }
      }
      return;
    }

    if (Array.isArray(node)) {
      for (let i = 0; i < node.length; i++) {
        walk(node[i], `${currentPath}[${i}]`);
      }
      return;
    }

    for (const [key, child] of Object.entries(node as Record<string, unknown>)) {
      walk(child, `${currentPath}.${key}`);
    }
  };

  walk(value, path);
}

function resolveVocabLabel(vocab: readonly string[], tokenId: number): string {
  return vocab[tokenId] ?? `id:${tokenId}`;
}

/**
 * Build a strictly presentation-safe `SamplerStepPresentation` separating:
 *   1. Unconstrained raw Denoiser LM-head distribution (`rawLmHead`)
 *   2. MDLM-constrained distribution (`constrainedMdlm`: `[MASK]` excluded, prob `0`, carry-over delta)
 *   3. Selection mode & chosen candidate (`selection`: `locked_carry_over` | `greedy_argmax` | `gumbel_max`)
 *   4. Finite, `Infinity`-free confidence display (`confidenceDisplay`: locked slots report
 *      `kind: 'locked_prior'`, `unscaledConfidence: null`, `displayText: 'Locked (prior step)'`)
 *   5. Per-slot decision classification (`locked_prior`, `committed_threshold`,
 *      `committed_progress_floor`, `committed_budget_floor`, `committed_final_step`, `remasked`).
 */
export function buildSamplerStepPresentation(
  params: BuildSamplerStepPresentationParams,
): SamplerStepPresentation {
  const {
    rawLogits,
    rawProbs,
    unmaskStep,
    vocab,
    maskTokenId,
    confidenceThreshold,
    temperature,
    absoluteTokenStart,
  } = params;

  const vocabSize = vocab.length;
  if (vocabSize < 2) {
    throw new RangeError(`vocab must contain at least 2 entries, received ${vocabSize}`);
  }
  if (!Number.isInteger(maskTokenId) || maskTokenId < 0 || maskTokenId >= vocabSize) {
    throw new RangeError(
      `maskTokenId must be an integer in [0, ${vocabSize - 1}], received ${maskTokenId}`,
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
  if (
    absoluteTokenStart !== undefined &&
    (!Number.isInteger(absoluteTokenStart) || absoluteTokenStart < 0)
  ) {
    throw new RangeError(
      `absoluteTokenStart must be a nonnegative integer when provided, received ${absoluteTokenStart}`,
    );
  }

  const topK = params.topK ?? 5;
  if (!Number.isInteger(topK) || topK < 1) {
    throw new RangeError(`topK must be an integer >= 1, received ${topK}`);
  }

  const B = unmaskStep.xtBefore.length;
  const expectedLen = B * vocabSize;
  if (rawLogits.length !== expectedLen) {
    throw new RangeError(
      `rawLogits expected length ${expectedLen} (B=${B} * V=${vocabSize}), received ${rawLogits.length}`,
    );
  }
  if (rawProbs.length !== expectedLen) {
    throw new RangeError(
      `rawProbs expected length ${expectedLen} (B=${B} * V=${vocabSize}), received ${rawProbs.length}`,
    );
  }
  if (unmaskStep.unscaledProbs.length !== expectedLen) {
    throw new RangeError(
      `unmaskStep.unscaledProbs expected length ${expectedLen}, received ${unmaskStep.unscaledProbs.length}`,
    );
  }
  if (unmaskStep.logXTheta.length !== expectedLen) {
    throw new RangeError(
      `unmaskStep.logXTheta expected length ${expectedLen}, received ${unmaskStep.logXTheta.length}`,
    );
  }

  for (let i = 0; i < expectedLen; i++) {
    if (!Number.isFinite(rawLogits[i])) {
      throw new RangeError(`Non-finite rawLogits[${i}]: ${rawLogits[i]}`);
    }
    if (!Number.isFinite(rawProbs[i])) {
      throw new RangeError(`Non-finite rawProbs[${i}]: ${rawProbs[i]}`);
    }
  }

  const thresholdSet = new Set<number>(unmaskStep.thresholdCommittedIndices);
  const floorSet = new Set<number>(unmaskStep.floorCommittedIndices);
  const remaskedSet = new Set<number>(unmaskStep.remaskedIndices);

  const rawIndices = new Array<number>(vocabSize);
  for (let v = 0; v < vocabSize; v++) rawIndices[v] = v;

  const nonMaskIndices: number[] = [];
  for (let v = 0; v < vocabSize; v++) {
    if (v !== maskTokenId) nonMaskIndices.push(v);
  }

  const kRaw = Math.min(topK, vocabSize);
  const kConstrained = Math.min(topK, nonMaskIndices.length);

  const slots: SamplerSlotPresentation[] = [];

  for (let j = 0; j < B; j++) {
    const rowStart = j * vocabSize;
    const tokenIdBefore = unmaskStep.xtBefore[j];
    const tokenIdAfter = unmaskStep.xtNext[j];
    const wasMaskedBefore = tokenIdBefore === maskTokenId;
    const isMaskedAfter = tokenIdAfter === maskTokenId;

    // 1. Unconstrained raw Denoiser LM head distribution
    const sortedRaw = [...rawIndices].sort((a, b) => {
      const diffLogit = rawLogits[rowStart + b] - rawLogits[rowStart + a];
      if (diffLogit !== 0) return diffLogit;
      const diffProb = rawProbs[rowStart + b] - rawProbs[rowStart + a];
      if (diffProb !== 0) return diffProb;
      return a - b;
    });
    const rawTopK: RawLmHeadCandidate[] = [];
    for (let i = 0; i < kRaw; i++) {
      const v = sortedRaw[i];
      rawTopK.push(
        Object.freeze({
          tokenId: v,
          tokenLabel: resolveVocabLabel(vocab, v),
          logit: rawLogits[rowStart + v],
          prob: rawProbs[rowStart + v],
        }),
      );
    }
    const rawLmHead: SamplerRawLmHeadView = Object.freeze({
      top1TokenId: rawTopK[0].tokenId,
      top1TokenLabel: rawTopK[0].tokenLabel,
      top1Logit: rawTopK[0].logit,
      top1Prob: rawTopK[0].prob,
      maskTokenProb: rawProbs[rowStart + maskTokenId],
      topK: Object.freeze(rawTopK),
    });

    // 2. Constrained MDLM distribution (strictly excludes maskTokenId)
    if (unmaskStep.unscaledProbs[rowStart + maskTokenId] !== 0) {
      throw new RangeError(
        `MDLM mask token probability must be strictly 0 at slot ${j}, received ${unmaskStep.unscaledProbs[rowStart + maskTokenId]}`,
      );
    }
    const isCarryOverDelta = !wasMaskedBefore;
    let constrainedTopK: ConstrainedMdlmCandidate[];
    if (isCarryOverDelta) {
      constrainedTopK = [
        Object.freeze({
          tokenId: tokenIdBefore,
          tokenLabel: resolveVocabLabel(vocab, tokenIdBefore),
          logProb: unmaskStep.logXTheta[rowStart + tokenIdBefore],
          prob: unmaskStep.unscaledProbs[rowStart + tokenIdBefore],
        }),
      ];
    } else {
      const sortedConstrained = [...nonMaskIndices].sort((a, b) => {
        const diffProb =
          unmaskStep.unscaledProbs[rowStart + b] - unmaskStep.unscaledProbs[rowStart + a];
        if (diffProb !== 0) return diffProb;
        const diffLog = unmaskStep.logXTheta[rowStart + b] - unmaskStep.logXTheta[rowStart + a];
        if (diffLog !== 0) return diffLog;
        return a - b;
      });
      constrainedTopK = [];
      for (let i = 0; i < kConstrained; i++) {
        const v = sortedConstrained[i];
        constrainedTopK.push(
          Object.freeze({
            tokenId: v,
            tokenLabel: resolveVocabLabel(vocab, v),
            logProb: unmaskStep.logXTheta[rowStart + v],
            prob: unmaskStep.unscaledProbs[rowStart + v],
          }),
        );
      }
    }
    const constrainedMdlm: SamplerConstrainedMdlmView = Object.freeze({
      maskTokenExcluded: true as const,
      maskTokenProb: 0 as const,
      isCarryOverDelta,
      topK: Object.freeze(constrainedTopK),
    });

    // 3. Selection mode & chosen token
    const selectedTokenId = unmaskStep.predicted[j];
    const samplingMode: SamplerSelectionMode = !wasMaskedBefore
      ? 'locked_carry_over'
      : temperature <= 0
        ? 'greedy_argmax'
        : 'gumbel_max';
    const selection: SamplerSelectionView = Object.freeze({
      samplingMode,
      temperature,
      selectedTokenId,
      selectedTokenLabel: resolveVocabLabel(vocab, selectedTokenId),
    });

    // 4. Presentation-safe confidence display (NEVER leaks Infinity for locked slots!)
    let confidenceDisplay: SamplerConfidenceDisplay;
    if (!wasMaskedBefore) {
      confidenceDisplay = Object.freeze({
        kind: 'locked_prior' as const,
        unscaledConfidence: null,
        displayText: 'Locked (prior step)' as const,
        exceedsThreshold: null,
      });
    } else {
      const conf = unmaskStep.confidence[j];
      if (!Number.isFinite(conf) || conf < 0 || conf > 1) {
        throw new RangeError(
          `Masked slot ${j} confidence must be a finite probability in [0, 1], received ${conf}`,
        );
      }
      confidenceDisplay = Object.freeze({
        kind: 'unscaled_probability' as const,
        unscaledConfidence: conf,
        displayText: `${(conf * 100).toFixed(1)}%`,
        exceedsThreshold: conf > confidenceThreshold,
      });
    }

    // 5. Explicit per-slot decision classification
    let decision: SamplerSlotDecision;
    if (!wasMaskedBefore) {
      decision = 'locked_prior';
    } else if (remaskedSet.has(j) || isMaskedAfter) {
      decision = 'remasked';
    } else if (thresholdSet.has(j)) {
      decision = 'committed_threshold';
    } else if (floorSet.has(j)) {
      if (unmaskStep.commitReason === 'final_step') {
        decision = 'committed_final_step';
      } else if (unmaskStep.commitReason === 'budget_floor') {
        decision = 'committed_budget_floor';
      } else {
        decision = 'committed_progress_floor';
      }
    } else {
      // Defensive fallback matching unmaskStep semantics if indices were externally composed
      const conf = unmaskStep.confidence[j];
      if (conf > confidenceThreshold) {
        decision = 'committed_threshold';
      } else if (unmaskStep.commitReason === 'final_step') {
        decision = 'committed_final_step';
      } else if (unmaskStep.commitReason === 'budget_floor') {
        decision = 'committed_budget_floor';
      } else {
        decision = 'committed_progress_floor';
      }
    }

    slots.push(
      Object.freeze({
        slotIndex: j,
        absolutePosition: absoluteTokenStart !== undefined ? absoluteTokenStart + j : null,
        tokenIdBefore,
        tokenLabelBefore: resolveVocabLabel(vocab, tokenIdBefore),
        tokenIdAfter,
        tokenLabelAfter: resolveVocabLabel(vocab, tokenIdAfter),
        wasMaskedBefore,
        isMaskedAfter,
        rawLmHead,
        constrainedMdlm,
        selection,
        confidenceDisplay,
        decision,
      }),
    );
  }

  const presentation: SamplerStepPresentation = Object.freeze({
    blockSize: B,
    vocabSize,
    maskTokenId,
    stepIndex: unmaskStep.stepIndex,
    stepsPerBlock: unmaskStep.stepsPerBlock,
    remainingSteps: unmaskStep.remainingSteps,
    tModel: unmaskStep.tModel,
    confidenceThreshold,
    temperature,
    commitReason: unmaskStep.commitReason,
    nMaskedBefore: unmaskStep.nMaskedBefore,
    nMaskedAfter: unmaskStep.nMaskedAfter,
    numAboveThreshold: unmaskStep.numAboveThreshold,
    minCommitFloor: unmaskStep.minCommitFloor,
    tokensCommittedCount: unmaskStep.tokensCommittedCount,
    earlyCompleted: unmaskStep.earlyCompleted,
    isBlockComplete: unmaskStep.isBlockComplete,
    slots: Object.freeze(slots),
  });

  assertPresentationSafeNoInfinity(presentation, 'samplerStepPresentation');
  return presentation;
}
