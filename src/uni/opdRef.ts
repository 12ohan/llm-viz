/**
 * Read-Only Cross-Paradigm On-Policy Distillation (OPD) Reference (`src/uni/opdRef.ts`)
 *
 * Clean-room mathematical reference for Section 5.3, `U24`, and `U28`:
 *   1. Understanding OPD (`computeTailKlBuckets`, `evaluateUnderstandingOpdRef`):
 *      - Transitive top-64/128-union `tail_kl` (`|I| = 128` + 1 residual tail bucket = `129`)
 *      - `arOpdLoss = 4.0 * reverse_tail_kl` (`D_KL(p_s || p_t)`)
 *      - `selfConsistencyLoss = forward_tail_kl / (0.5005 * responseLen)`
 *   2. Continuous-Flow Editing OPD (`computeMatchedSigmas`, `evaluateFlowEditingOpdRef`):
 *      - Qwen-Image dynamic time-shift `mu(seqLen)` and endpoint-inverted `matched_sigmas`
 *      - Linear flow noising `x^r = (1 - r) * x^0 + r * delta` (`r = 0.9`)
 *      - `totalEditingOpdLoss = exact_ce + 0.2 * grouped_ce`
 *   3. `U28` Reference-Trainer Bug Diagnostics (`getTrainerBugDiagnosticsU28`):
 *      - Quantifies VLM SFT double-normalization (`~6.915x` avg / `1,000x` spike),
 *        accidental `<|im_end|>` (`id = 11`) unmasking, and `132,101 - 64` `<LOC_*>` tail leak.
 *
 * STRICTLY READ-ONLY & FORWARD-ONLY:
 *   - `0` gradients, `0` backward passes, `0` optimizer steps, `0` `Math.random()`.
 *   - ZERO imports from `src/twotower/`, `src/diffusion/`, or `src/ui/controls.ts`.
 */

import { createXoshiro128ss, nextGaussianBoxMuller } from './prng';
import type {
  UniFlowEditingOpdResult,
  UniMatchedSigmasResult,
  UniTailKlResult,
  UniTrainerBugDiagnosticU28,
  UniUnderstandingOpdResult,
} from './types';

// ============================================================================
// LOCKED OPD REFERENCE CONSTANTS (`U24`, `U28`, §5.3)
// ============================================================================

export const UNI_OPD_TAIL_FLOOR = 1e-5 as const;
export const UNI_OPD_AR_WEIGHT = 4.0 as const;
export const UNI_OPD_NOISE_EPSILON = 0.001 as const;
export const UNI_OPD_GROUPED_CE_WEIGHT = 0.2 as const;

const PROB_SUM_TOLERANCE = 1e-2;

// ============================================================================
// INTERFACES FOR DEMO INPUTS & OPTIONS
// ============================================================================

export interface UniUnderstandingOpdOptions {
  readonly responseLen?: number;
  readonly topKTarget?: number;
  readonly topKStudent?: number;
  readonly arOpdWeight?: number;
  readonly noiseEpsilon?: number;
}

export interface UniFlowEditingOpdOptions {
  readonly r?: number;
  readonly seqLen?: number;
  readonly steps?: number;
  readonly sigmaEnd?: number;
  readonly groupedCeWeight?: number;
}

export interface UniDefaultOpdDemoInputs {
  readonly seed: number;
  readonly vocabSize: number;
  readonly responseLen: number;
  readonly teacherArProbs: Float32Array;
  readonly studentArProbs: Float32Array;
  readonly studentDiffProbs: Float32Array;
  readonly x0: Float32Array;
  readonly noiseDelta: Float32Array;
  readonly exactLogits: Float32Array;
  readonly targetCode: number;
  readonly groupLogits: Float32Array;
  readonly targetGroup: number;
}

// ============================================================================
// INTERNAL VALIDATION & MATH HELPERS
// ============================================================================

function validateProbDistribution(probs: Float32Array, name: string): number {
  let sum = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (!Number.isFinite(p) || p < 0) {
      throw new RangeError(
        `${name}[${i}] must be a finite non-negative probability, got ${String(p)}`,
      );
    }
    sum += p;
  }
  if (!Number.isFinite(sum) || Math.abs(sum - 1) > PROB_SUM_TOLERANCE) {
    throw new RangeError(
      `${name} probabilities must sum to ~1.0 (within ${PROB_SUM_TOLERANCE}), got ${sum}`,
    );
  }
  return sum;
}

function validateFiniteVector(vec: Float32Array, name: string): void {
  if (vec.length === 0) {
    throw new RangeError(`${name} must be non-empty`);
  }
  for (let i = 0; i < vec.length; i++) {
    if (!Number.isFinite(vec[i])) {
      throw new RangeError(`${name}[${i}] must be finite, got ${String(vec[i])}`);
    }
  }
}

function selectTopKIndices(probs: Float32Array, k: number): Int32Array {
  const indices = new Array<number>(probs.length);
  for (let i = 0; i < probs.length; i++) {
    indices[i] = i;
  }
  indices.sort((a, b) => {
    const diff = probs[b] - probs[a];
    return diff !== 0 ? diff : a - b;
  });
  const out = new Int32Array(k);
  for (let i = 0; i < k; i++) {
    out[i] = indices[i];
  }
  return out;
}

function computeCrossEntropyFromLogits(logits: Float32Array, targetIndex: number): number {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    sumExp += Math.exp(logits[i] - maxLogit);
  }
  const logSumExp = maxLogit + Math.log(sumExp);
  return logSumExp - logits[targetIndex];
}

function softmaxToFloat32(logits: Float32Array): Float32Array {
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
    }
  }
  const exps = new Float64Array(logits.length);
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    exps[i] = e;
    sumExp += e;
  }
  const probs = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    probs[i] = Math.fround(exps[i] / sumExp);
  }
  return probs;
}

// ============================================================================
// UNDERSTANDING OPD: DETERMINISTIC TAIL-KL BUCKETING (`U24`, §5.3)
// ============================================================================

/**
 * Compute deterministic top-K union + residual tail-bucket KL divergences (`U24`):
 *
 *   1. Select `kT = min(V, topKTarget)` indices with highest `targetProbs`
 *      (tie-breaking on smaller token index).
 *   2. Select `kS = min(V, topKStudent)` indices with highest `studentProbs`
 *      (tie-breaking on smaller token index).
 *   3. Form union `I` preserving `targetTop[0..kT-1]` first, then appending
 *      unseen indices from `studentTop[0..kS-1]` until `|I| === min(V, topKStudent)`
 *      (yielding `128` union indices + `1` tail bucket = `129` buckets for `V >= 128`).
 *   4. Sort `unionIndices` ascending by token index for canonical cross-runtime comparison.
 *   5. For each bucket `b in 0..|I|-1` (`idx = unionIndices[b]`):
 *        `rawP[b] = max(tailFloor, p[idx])`
 *      For the residual tail bucket `b = |I|`:
 *        `rawP[|I|] = max(tailFloor, 1.0 - sum_{v in I} p[v])`
 *      Normalize each bucket vector `pBuckets[b] = rawP[b] / sum_j rawP[j]`
 *      so both `pTargetBuckets` and `pStudentBuckets` form strictly positive
 *      probability distributions summing to `1.0`.
 *   6. Compute `reverseKl = D_KL(p_s || p_t)` and `forwardKl = D_KL(p_t || p_s)`.
 */
export function computeTailKlBuckets(
  targetProbs: Float32Array,
  studentProbs: Float32Array,
  topKTarget = 64,
  topKStudent = 128,
  tailFloor = UNI_OPD_TAIL_FLOOR,
): UniTailKlResult {
  const vocabSize = targetProbs.length;
  if (vocabSize !== studentProbs.length) {
    throw new RangeError(
      `targetProbs (${vocabSize}) and studentProbs (${studentProbs.length}) must have equal length`,
    );
  }
  if (vocabSize < 2) {
    throw new RangeError(`Vocabulary size must be >= 2, got ${vocabSize}`);
  }
  if (!Number.isInteger(topKTarget) || topKTarget < 1) {
    throw new RangeError(`topKTarget must be a positive integer, got ${String(topKTarget)}`);
  }
  if (!Number.isInteger(topKStudent) || topKStudent < 1) {
    throw new RangeError(`topKStudent must be a positive integer, got ${String(topKStudent)}`);
  }
  if (!Number.isFinite(tailFloor) || tailFloor <= 0 || tailFloor >= 1) {
    throw new RangeError(`tailFloor must be in (0, 1), got ${String(tailFloor)}`);
  }

  validateProbDistribution(targetProbs, 'targetProbs');
  validateProbDistribution(studentProbs, 'studentProbs');

  const kT = Math.min(vocabSize, topKTarget);
  const kS = Math.min(vocabSize, topKStudent);
  const targetUnionLimit = Math.min(vocabSize, topKStudent);

  const targetTop = selectTopKIndices(targetProbs, kT);
  const studentTop = selectTopKIndices(studentProbs, kS);

  const inUnion = new Uint8Array(vocabSize);
  const selected: number[] = [];

  for (let i = 0; i < targetTop.length; i++) {
    const idx = targetTop[i];
    if (inUnion[idx] === 0) {
      inUnion[idx] = 1;
      selected.push(idx);
    }
  }
  for (let i = 0; i < studentTop.length && selected.length < targetUnionLimit; i++) {
    const idx = studentTop[i];
    if (inUnion[idx] === 0) {
      inUnion[idx] = 1;
      selected.push(idx);
    }
  }

  selected.sort((a, b) => a - b);
  const unionIndices = Int32Array.from(selected);
  const unionLen = unionIndices.length;
  const bucketCount = unionLen + 1;

  const rawT = new Float64Array(bucketCount);
  const rawS = new Float64Array(bucketCount);
  let sumUnionT = 0;
  let sumUnionS = 0;

  for (let b = 0; b < unionLen; b++) {
    const idx = unionIndices[b];
    const pT = targetProbs[idx];
    const pS = studentProbs[idx];
    sumUnionT += pT;
    sumUnionS += pS;
    rawT[b] = Math.max(tailFloor, pT);
    rawS[b] = Math.max(tailFloor, pS);
  }

  rawT[unionLen] = Math.max(tailFloor, 1.0 - sumUnionT);
  rawS[unionLen] = Math.max(tailFloor, 1.0 - sumUnionS);

  let normSumT = 0;
  let normSumS = 0;
  for (let b = 0; b < bucketCount; b++) {
    normSumT += rawT[b];
    normSumS += rawS[b];
  }

  const pTargetBuckets = new Float32Array(bucketCount);
  const pStudentBuckets = new Float32Array(bucketCount);
  let reverseKl = 0;
  let forwardKl = 0;

  for (let b = 0; b < bucketCount; b++) {
    const pTNorm = rawT[b] / normSumT;
    const pSNorm = rawS[b] / normSumS;
    pTargetBuckets[b] = Math.fround(pTNorm);
    pStudentBuckets[b] = Math.fround(pSNorm);

    reverseKl += pSNorm * Math.log(pSNorm / pTNorm);
    forwardKl += pTNorm * Math.log(pTNorm / pSNorm);
  }

  return Object.freeze({
    unionIndices,
    bucketCount,
    pTargetBuckets,
    pStudentBuckets,
    pTargetTailMass: pTargetBuckets[unionLen],
    pStudentTailMass: pStudentBuckets[unionLen],
    reverseKl: Math.max(0, reverseKl),
    forwardKl: Math.max(0, forwardKl),
  });
}

/**
 * Evaluate read-only Understanding OPD loss (`U24`, `loss_math.py:18-75`):
 *   - `arTailKl = computeTailKlBuckets(teacherArProbs, studentArProbs, topKTarget, topKStudent)`
 *   - `consistencyTailKl = computeTailKlBuckets(studentArProbs, studentDiffProbs, topKTarget, topKStudent)`
 *   - `expectedMaskDenominator = ((1 + noiseEpsilon) / 2) * responseLen` (`0.5005 * R`)
 *   - `arOpdLoss = arOpdWeight * arTailKl.reverseKl` (`4.0 * reverse_tail_kl`)
 *   - `selfConsistencyLoss = consistencyTailKl.forwardKl / expectedMaskDenominator`
 *   - `totalUnderstandingOpdLoss = arOpdLoss + selfConsistencyLoss`
 */
export function evaluateUnderstandingOpdRef(
  teacherArProbs: Float32Array,
  studentArProbs: Float32Array,
  studentDiffProbs: Float32Array,
  options: UniUnderstandingOpdOptions = {},
): UniUnderstandingOpdResult {
  const responseLen = options.responseLen ?? 32;
  const topKTarget = options.topKTarget ?? 64;
  const topKStudent = options.topKStudent ?? 128;
  const arOpdWeight = options.arOpdWeight ?? UNI_OPD_AR_WEIGHT;
  const noiseEpsilon = options.noiseEpsilon ?? UNI_OPD_NOISE_EPSILON;

  if (!Number.isFinite(responseLen) || responseLen <= 0) {
    throw new RangeError(`responseLen must be positive and finite, got ${String(responseLen)}`);
  }
  if (!Number.isFinite(arOpdWeight) || arOpdWeight < 0) {
    throw new RangeError(`arOpdWeight must be finite and >= 0, got ${String(arOpdWeight)}`);
  }
  if (!Number.isFinite(noiseEpsilon) || noiseEpsilon < 0 || noiseEpsilon >= 1) {
    throw new RangeError(`noiseEpsilon must be in [0, 1), got ${String(noiseEpsilon)}`);
  }

  const arTailKl = computeTailKlBuckets(
    teacherArProbs,
    studentArProbs,
    topKTarget,
    topKStudent,
  );
  const consistencyTailKl = computeTailKlBuckets(
    studentArProbs,
    studentDiffProbs,
    topKTarget,
    topKStudent,
  );

  const expectedMaskDenominator = ((1 + noiseEpsilon) / 2) * responseLen;
  const arOpdLoss = arOpdWeight * arTailKl.reverseKl;
  const selfConsistencyLoss = consistencyTailKl.forwardKl / expectedMaskDenominator;
  const totalUnderstandingOpdLoss = arOpdLoss + selfConsistencyLoss;

  return Object.freeze({
    readOnly: true,
    hasGradients: false,
    arOpdWeight,
    noiseEpsilon,
    responseLen,
    expectedMaskDenominator,
    arTailKl,
    consistencyTailKl,
    arOpdLoss,
    selfConsistencyLoss,
    totalUnderstandingOpdLoss,
  });
}

// ============================================================================
// CONTINUOUS-FLOW EDITING OPD: MATCHED SIGMAS & HYBRID CE (§5.3, `U24`)
// ============================================================================

/**
 * Invert Qwen-Image dynamic shift `timeShift(mu, t) = exp(mu) / (exp(mu) + (1 / t - 1))`
 * between endpoints `r` (`0.9`) and `sigmaEnd` (`0.02`) for `steps` (`40`) points.
 */
export function computeMatchedSigmas(
  r = 0.9,
  seqLen = 1024,
  steps = 40,
  sigmaEnd = 0.02,
): UniMatchedSigmasResult {
  if (!Number.isFinite(r) || !Number.isFinite(sigmaEnd) || !(0 < sigmaEnd && sigmaEnd < r && r < 1)) {
    throw new RangeError(
      `Expected 0 < sigmaEnd < r < 1, got sigmaEnd=${String(sigmaEnd)}, r=${String(r)}`,
    );
  }
  if (!Number.isInteger(steps) || steps < 2) {
    throw new RangeError(`steps must be an integer >= 2, got ${String(steps)}`);
  }
  if (!Number.isFinite(seqLen) || seqLen < 1) {
    throw new RangeError(`seqLen must be >= 1, got ${String(seqLen)}`);
  }

  const frac = Math.min(1, Math.max(0, (seqLen - 256) / (4096 - 256)));
  const mu = 0.5 + frac * (1.15 - 0.5);
  const expMu = Math.exp(mu);

  const tStart = r / (r + expMu * (1 - r));
  const tEnd = sigmaEnd / (sigmaEnd + expMu * (1 - sigmaEnd));

  const sigmas = new Float32Array(steps);
  for (let k = 0; k < steps; k++) {
    const t = tStart + (tEnd - tStart) * (k / (steps - 1));
    sigmas[k] = Math.fround(expMu / (expMu + (1 / t - 1)));
  }

  return Object.freeze({
    r,
    seqLen,
    steps,
    mu,
    sigmaStart: r,
    sigmaEnd,
    sigmas,
  });
}

/**
 * Evaluate read-only Continuous-Flow Editing OPD forward metrics (`§5.3`, `U24`):
 *   - `xNoisyR[i] = Math.fround((1 - r) * x0[i] + r * noiseDelta[i])`
 *   - `exactCe = -logSoftmax(exactLogits)[targetCode]`
 *   - `groupedCe = -logSoftmax(groupLogits)[targetGroup]`
 *   - `totalEditingOpdLoss = exactCe + groupedCeWeight * groupedCe` (`exact_ce + 0.2 * grouped_ce`)
 */
export function evaluateFlowEditingOpdRef(
  x0: Float32Array,
  noiseDelta: Float32Array,
  exactLogits: Float32Array,
  targetCode: number,
  groupLogits: Float32Array,
  targetGroup: number,
  options: UniFlowEditingOpdOptions = {},
): UniFlowEditingOpdResult {
  if (x0.length !== noiseDelta.length) {
    throw new RangeError(
      `x0 (${x0.length}) and noiseDelta (${noiseDelta.length}) must have equal length`,
    );
  }
  validateFiniteVector(x0, 'x0');
  validateFiniteVector(noiseDelta, 'noiseDelta');
  validateFiniteVector(exactLogits, 'exactLogits');
  validateFiniteVector(groupLogits, 'groupLogits');

  if (exactLogits.length < 2) {
    throw new RangeError(`exactLogits length must be >= 2, got ${exactLogits.length}`);
  }
  if (groupLogits.length < 2) {
    throw new RangeError(`groupLogits length must be >= 2, got ${groupLogits.length}`);
  }
  if (!Number.isInteger(targetCode) || targetCode < 0 || targetCode >= exactLogits.length) {
    throw new RangeError(
      `targetCode (${String(targetCode)}) out of range [0, ${exactLogits.length})`,
    );
  }
  if (!Number.isInteger(targetGroup) || targetGroup < 0 || targetGroup >= groupLogits.length) {
    throw new RangeError(
      `targetGroup (${String(targetGroup)}) out of range [0, ${groupLogits.length})`,
    );
  }

  const r = options.r ?? 0.9;
  const seqLen = options.seqLen ?? 1024;
  const steps = options.steps ?? 40;
  const sigmaEnd = options.sigmaEnd ?? 0.02;
  const groupedCeWeight = options.groupedCeWeight ?? UNI_OPD_GROUPED_CE_WEIGHT;

  if (!Number.isFinite(groupedCeWeight) || groupedCeWeight < 0) {
    throw new RangeError(
      `groupedCeWeight must be finite and >= 0, got ${String(groupedCeWeight)}`,
    );
  }

  const matchedSigmas = computeMatchedSigmas(r, seqLen, steps, sigmaEnd);

  const xNoisyR = new Float32Array(x0.length);
  for (let i = 0; i < x0.length; i++) {
    xNoisyR[i] = Math.fround((1 - r) * x0[i] + r * noiseDelta[i]);
  }

  const exactCe = computeCrossEntropyFromLogits(exactLogits, targetCode);
  const groupedCe = computeCrossEntropyFromLogits(groupLogits, targetGroup);
  const totalEditingOpdLoss = exactCe + groupedCeWeight * groupedCe;

  return Object.freeze({
    readOnly: true,
    hasGradients: false,
    r,
    matchedSigmas,
    x0: new Float32Array(x0),
    noiseDelta: new Float32Array(noiseDelta),
    xNoisyR,
    exactCe,
    groupedCe,
    groupedCeWeight,
    totalEditingOpdLoss,
  });
}

// ============================================================================
// `U28` REFERENCE-TRAINER DIAGNOSTICS & SEEDED DEMO INPUTS
// ============================================================================

/**
 * Exact verified quantitative diagnostics for the three `U28` bugs discovered
 * in `minimal_training/` (`objectives.py`, `loss_math.py`), strictly read-only.
 */
export function getTrainerBugDiagnosticsU28(): UniTrainerBugDiagnosticU28 {
  const tMin = 0.001;
  const tMax = 1.0;
  const fullVocabSize = 132_101;
  const teacherVocabSize = 131_072;
  const locExtensionTokens = fullVocabSize - teacherVocabSize; // 1,029
  const topK = 64;
  const leakedTailDenom = fullVocabSize - topK; // 132,037
  const trueTeacherTailDenom = teacherVocabSize - topK; // 131,008

  return Object.freeze({
    readOnly: true,
    sftDoubleNorm: Object.freeze({
      tMin,
      tMax,
      expectedInflationFactor: Math.log(1000) / 0.999,
      peakSpikeAtTMin: 1000,
      formulaNote:
        'objectives.py:142,168,175-176 + loss_math.py:73-75 divide VLM SFT masked CE by both t in [0.001, 1.0] and max(1, masked_count), yielding E[1/t] = ln(1000)/0.999 (~6.915x avg inflation) and 1,000x spike at t = 0.001.',
    }),
    accidentalImEndSupervision: Object.freeze({
      eosTokenId: 11,
      ignoredLabel: -100,
      affectedTurns: 'system and user turns',
      description:
        'objectives.py:98 calls masked_fill(ids.eq(11), 11), unintentionally restoring supervision on system/user <|im_end|> (id=11) tokens previously masked to -100 at line 93.',
    }),
    opdLocTailLeak: Object.freeze({
      fullVocabSize,
      teacherVocabSize,
      locExtensionTokens,
      topK,
      leakedTailDenom,
      trueTeacherTailDenom,
      leakedMassFractionOfTail: locExtensionTokens / leakedTailDenom,
    }),
  });
}

/**
 * Build deterministic seeded demo inputs for read-only OPD inspection using
 * `createXoshiro128ss` (`U23`, zero `Math.random()`).
 *
 * Uses `V = 256` so `topKTarget = 64, topKStudent = 128` exercises the exact
 * `129`-bucket (`128` union + `1` residual tail) `tail_kl` partition.
 */
export function createDefaultOpdDemoInputs(seed = 0x4f504431): UniDefaultOpdDemoInputs {
  const rng = createXoshiro128ss(seed);
  const vocabSize = 256;
  const groupVocabSize = 32;
  const latentSize = 16;
  const responseLen = 32;

  const teacherLogits = new Float32Array(vocabSize);
  const studentArLogits = new Float32Array(vocabSize);
  const studentDiffLogits = new Float32Array(vocabSize);

  for (let i = 0; i < vocabSize; i++) {
    const base = nextGaussianBoxMuller(() => rng.nextUniformOpenUnit()) * 1.4;
    const arDelta = nextGaussianBoxMuller(() => rng.nextUniformOpenUnit()) * 0.45;
    const diffDelta = nextGaussianBoxMuller(() => rng.nextUniformOpenUnit()) * 0.35;
    teacherLogits[i] = Math.fround(base);
    studentArLogits[i] = Math.fround(base * 0.9 + arDelta);
    studentDiffLogits[i] = Math.fround(studentArLogits[i] * 0.95 + diffDelta);
  }

  const teacherArProbs = softmaxToFloat32(teacherLogits);
  const studentArProbs = softmaxToFloat32(studentArLogits);
  const studentDiffProbs = softmaxToFloat32(studentDiffLogits);

  const x0 = new Float32Array(latentSize);
  const noiseDelta = new Float32Array(latentSize);
  for (let i = 0; i < latentSize; i++) {
    x0[i] = Math.fround(nextGaussianBoxMuller(() => rng.nextUniformOpenUnit()) * 0.5);
    noiseDelta[i] = Math.fround(nextGaussianBoxMuller(() => rng.nextUniformOpenUnit()));
  }

  const exactLogits = new Float32Array(vocabSize);
  let targetCode = 0;
  let bestExact = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    const v = Math.fround(nextGaussianBoxMuller(() => rng.nextUniformOpenUnit()) * 1.2);
    exactLogits[i] = v;
    if (v > bestExact) {
      bestExact = v;
      targetCode = i;
    }
  }

  const groupLogits = new Float32Array(groupVocabSize);
  let targetGroup = 0;
  let bestGroup = -Infinity;
  for (let i = 0; i < groupVocabSize; i++) {
    const v = Math.fround(nextGaussianBoxMuller(() => rng.nextUniformOpenUnit()) * 1.2);
    groupLogits[i] = v;
    if (v > bestGroup) {
      bestGroup = v;
      targetGroup = i;
    }
  }

  return Object.freeze({
    seed,
    vocabSize,
    responseLen,
    teacherArProbs,
    studentArProbs,
    studentDiffProbs,
    x0,
    noiseDelta,
    exactLogits,
    targetCode,
    groupLogits,
    targetGroup,
  });
}
