/**
 * Paper §6.2 Self-Speculative Decoding Engine — Contract A (`src/dory/speculative.ts`)
 *
 * Implements `DorySpeculativeModuleContract` strictly matching `D3`, `D5`, `DORY0` §8, and
 * `DORY0_REVIEW_RESPONSE.md` Findings 1–3 & 6:
 *   - **Contract A (`DR23`, `104` Persistent Banks, `per_loop` Only):**
 *     Badged `Reconstructed from Paper §6.2 (Contract A) · Not in Upstream Runtimes · per_loop Only`.
 *     Maintains `8` dedicated persistent `output_shallow` Coda KV banks alongside the `96` deep
 *     `per_loop` banks (`8` Input + `80` Loops 1–4 + `8` `output_deep`), guaranteeing standalone
 *     `k = 1` proposal fidelity across every cycle while preserving 100% Prelude + Loop-1 KV and
 *     `(x, s_1)` activation reuse.
 *   - **Exact 10-Transition `(committedLen = L, pendingToken = d_0)` State Machine:**
 *     - Transition 1 (`prefillDorySelfSpec`): Batched `k = 4` Deep prefill + batched Shallow Coda
 *       on cached `s_1^{(0..P-1)}` commits all `104` banks to `L = P` and samples unconsumed `d_0`.
 *     - Transitions 2–9 (`runDorySelfSpecCycle`):
 *       * `gammaEff = min(gamma, S_max - 1 - L)`; on `gammaEff === 0` (`L = S_max - 1`), runs
 *         1-token Deep `k = 4` decode + 1-token Shallow Coda catch-up and halts.
 *       * Shallow `k = 1` draft (`j = 0 .. gammaEff - 1`, early stop on draft EOS) stages Input,
 *         Loop-1, and `output_shallow` KV + `(x, u_1, s_1)`.
 *       * Batched Deep `k = 4` verify reuses Input + Loop-1 KV & `(x, s_1)` for `[L .. L + gamma' - 1]`,
 *         runs 1-token Prelude + Loop 1 only for bonus slot `L + gamma'` (when non-EOS), then batches
 *         Loops 2–4 and `output_deep` Coda.
 *       * Rejection (`a < gamma'`): `0` extra Shallow Coda work; samples `y* ~ p_res` and rewinds
 *         all `104` banks atomically to `L' = L + a + 1`.
 *       * Full acceptance with draft EOS (`a = gamma'`, `draftEndedWithEos`): commits `L' = L + gamma'`
 *         across all `104` banks and halts cleanly without a bonus token (`nextPendingToken = null`).
 *       * Full non-EOS acceptance (`a = gamma'`): samples `y_bonus ~ p_{gamma'+1}`, runs 1-token
 *         Shallow Coda catch-up on cached `s_1^{(L + gamma')}`, and commits `L' = L + gamma' + 1`.
 *   - **Division-Free Binary64 Rejection Sampling & Residual Correction (Finding 3):**
 *     Operates strictly in IEEE-754 binary64 (`Number`) with identical `(temperature, topK, topP)`
 *     filtering applied to both sampling and acceptance distributions. Asserts `q[d] > 0`
 *     (`DorySpecSupportInvariantError`, zero `1e-38` clamp), accepts unconditionally if `p[d] >= q[d]`,
 *     otherwise accepts iff `r * q[d] < p[d]`. Residual `res[v] = max(0, p[v] - q[v]) / Z` requires
 *     strictly positive `Z > 0` (zero `1e-12` fallback to `p`).
 *   - **Transaction-Local SWA Rollback (`DR19`):**
 *     Enforces `cap_swa = 16 >= W + gamma + 1` with `absPosTag` validation via `rewindDoryKvCache`
 *     inside an active speculative cycle, zeroing rolled-back tags and slots atomically.
 *
 * 100% clean-room original TypeScript — zero ambient RNG, zero DOM/WebGL/Node/sibling imports (`D4`).
 */

import {
  runDoryForwardChunk,
  runDoryInputPrelude,
  runDoryOutputCoda,
  runDoryRecurrentPass,
} from './forward';
import { createDoryKvCache, rewindDoryKvCache } from './kvCache';
import {
  anchorMapJacobianNorm,
  argmaxIndex,
  cosineSimilarity8d,
  l2Distance,
  softmaxFloat64,
  vectorL2Norm,
} from './ngptMath';
import { createDoryPrng, deriveDorySeed } from './prng';
import { buildDoryDualRopeTables } from './rope';
import { computeExecutedDepth } from './spec';
import {
  DORY_DEFAULT_SESSION_SEED,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  DORY_SPEC_DRAFT_LOOPS,
  DORY_SPEC_VERIFY_LOOPS,
  DorySessionImmutableConfigError,
  DorySpecSupportInvariantError,
  type DoryExecutedSublayerRecord,
  type DoryForwardTrace,
  type DoryHaltReason,
  type DoryKvCache,
  type DoryLoopTrajectoryMetrics,
  type DoryMiniConfig,
  type DoryPearsonChiSquaredResult,
  type DoryPrng,
  type DoryRejectionSampleResult,
  type DoryResidualDistributionResult,
  type DorySamplingConfig,
  type DorySelfSpecCycleInput,
  type DorySelfSpecCycleResult,
  type DorySpecAnalyticalMarginalResult,
  type DorySpecCyclePrefillInput,
  type DorySpecCyclePrefillResult,
  type DorySpecCycleSummary,
  type DorySpecDraftStepRecord,
  type DorySpecDraftStepSummary,
  type DorySpecOutcomeKind,
  type DorySpecSingleStepSampleResult,
  type DorySpecTwoStepAnalyticalResult,
  type DorySpecVerifyPositionRecord,
  type DorySpecVerifyPositionSummary,
} from './types';

// ============================================================================
// INTERNAL VALIDATION & TENSOR SLICE HELPERS
// ============================================================================

function isEosTokenId(tokenId: number, config: DoryMiniConfig): boolean {
  const eos = config.eosTokenIds;
  for (let i = 0; i < eos.length; i += 1) {
    if (eos[i] === tokenId) {
      return true;
    }
  }
  return false;
}

function assertValidUniformVariate(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0.0 || value >= 1.0) {
    throw new RangeError(
      `Invalid uniform variate ${label}=${String(value)}: expected finite number in [0, 1).`,
    );
  }
  return value;
}

function validateProbabilityVector(probs: ArrayLike<number>, label: string): number {
  const len = probs.length;
  if (len <= 0) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      `Distribution ${label} must be non-empty; got length ${len}.`,
    );
  }
  let sum = 0.0;
  for (let v = 0; v < len; v += 1) {
    const p = probs[v];
    if (!Number.isFinite(p) || p < 0.0) {
      throw new DorySpecSupportInvariantError(
        'INVALID_DISTRIBUTION',
        `Invalid probability ${label}[${v}]=${String(p)}: expected finite number >= 0.`,
      );
    }
    sum += p;
  }
  if (!(sum > 0.0) || !Number.isFinite(sum)) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      `Distribution ${label} has non-positive total mass ${String(sum)}.`,
    );
  }
  return sum;
}

function sliceFloat32Row(flat: Float32Array, rowIdx: number, width: number): Float32Array {
  const off = rowIdx * width;
  const out = new Float32Array(width);
  for (let i = 0; i < width; i += 1) {
    out[i] = flat[off + i];
  }
  return out;
}

function concatFloat32Vectors(vectors: readonly Float32Array[], dim: number): Float32Array {
  const out = new Float32Array(vectors.length * dim);
  for (let t = 0; t < vectors.length; t += 1) {
    const src = vectors[t];
    const base = t * dim;
    for (let c = 0; c < dim; c += 1) {
      out[base + c] = src[c];
    }
  }
  return out;
}

function computeSingleLoopTokenMetric(
  loopIteration: number,
  prevState: Float32Array,
  xAnchor: Float32Array,
  uEntry: Float32Array,
  sExit: Float32Array,
  tokenOffset: number,
  absTokenPos: number,
  config: DoryMiniConfig,
): DoryLoopTrajectoryMetrics {
  const d = config.hiddenSize;
  const off = tokenOffset * d;
  const prevAnchorCos = cosineSimilarity8d(prevState, xAnchor, off, off, d);
  const clampedCos = Math.max(-1.0, Math.min(1.0, prevAnchorCos));
  const entryAngleRadians = Math.acos(clampedCos);

  return {
    loopIteration,
    tokenOffset,
    absTokenPos,
    uNorm: vectorL2Norm(uEntry, off, d),
    sNorm: vectorL2Norm(sExit, off, d),
    consecutiveCosine: cosineSimilarity8d(sExit, prevState, off, off, d),
    consecutiveDisplacement: l2Distance(sExit, prevState, off, off, d),
    anchorCosine: cosineSimilarity8d(sExit, xAnchor, off, off, d),
    entryAnchorJacobian: anchorMapJacobianNorm(entryAngleRadians, config.l2NormEps),
  };
}

function computeAllLoopMetricsForToken(
  xAnchor: Float32Array,
  loopEntryStates: readonly Float32Array[],
  loopExitStates: readonly Float32Array[],
  tokenOffset: number,
  absTokenPos: number,
  config: DoryMiniConfig,
): DoryLoopTrajectoryMetrics[] {
  const kLoops = loopEntryStates.length;
  const metrics = new Array<DoryLoopTrajectoryMetrics>(kLoops);
  for (let loopIdx = 0; loopIdx < kLoops; loopIdx += 1) {
    const prevState = loopIdx === 0 ? xAnchor : loopExitStates[loopIdx - 1];
    metrics[loopIdx] = computeSingleLoopTokenMetric(
      loopIdx + 1,
      prevState,
      xAnchor,
      loopEntryStates[loopIdx],
      loopExitStates[loopIdx],
      tokenOffset,
      absTokenPos,
      config,
    );
  }
  return metrics;
}

function assertContractAKvCache(kvCache: DoryKvCache, config: DoryMiniConfig): void {
  if (
    kvCache.mode !== 'per_loop' ||
    kvCache.kLoops !== DORY_SPEC_VERIFY_LOOPS ||
    !kvCache.includeShallowCoda ||
    kvCache.outputShallowBanks === null ||
    kvCache.allBanks.length !== DORY_SPEC_CONTRACT_A_BANK_COUNT
  ) {
    throw new DorySessionImmutableConfigError(
      'DR23_SPEC_REQUIRES_PER_LOOP_K4',
      `Contract A self-speculation requires mode='per_loop', kLoops=${DORY_SPEC_VERIFY_LOOPS}, includeShallowCoda=true (${DORY_SPEC_CONTRACT_A_BANK_COUNT} banks); got mode='${kvCache.mode}', kLoops=${kvCache.kLoops}, includeShallowCoda=${String(kvCache.includeShallowCoda)}, banks=${kvCache.allBanks.length}.`,
    );
  }

  const expectedLen = kvCache.committedLen;
  if (
    !Number.isInteger(expectedLen) ||
    expectedLen < 0 ||
    expectedLen > config.maxPositionEmbeddings
  ) {
    throw new RangeError(
      `Invalid Contract A kvCache.committedLen=${String(expectedLen)} (expected integer in 0..${config.maxPositionEmbeddings}).`,
    );
  }

  for (let b = 0; b < kvCache.allBanks.length; b += 1) {
    const bank = kvCache.allBanks[b];
    if (bank.logicalLen !== expectedLen) {
      throw new Error(
        `Contract A cycle-boundary invariant violated: bankSlotId=${bank.bankSlotId} (${bank.bankKind}, p=${bank.physicalLayerIdx}) has logicalLen=${bank.logicalLen} !== kvCache.committedLen=${expectedLen}.`,
      );
    }
  }
}

// ============================================================================
// SAMPLING CONFIG & BINARY64 LOGITS -> FILTERED PROBS (AUDIT §5.6 & §8.2)
// ============================================================================

/**
 * Validates and normalizes partial sampling options into a frozen `DorySamplingConfig`.
 * Neutral identity defaults: `temperature = 1.0`, `topK = 0` (disabled), `topP = 1.0` (disabled).
 */
export function normalizeSamplingConfig(partial?: Partial<DorySamplingConfig>): DorySamplingConfig {
  const rawTemp = partial?.temperature ?? 1.0;
  const rawTopK = partial?.topK ?? 0;
  const rawTopP = partial?.topP ?? 1.0;

  if (!Number.isFinite(rawTemp) || rawTemp < 0.0) {
    throw new RangeError(
      `Invalid sampling temperature=${String(rawTemp)}: expected finite number >= 0.`,
    );
  }
  if (!Number.isInteger(rawTopK) || rawTopK < 0) {
    throw new RangeError(
      `Invalid sampling topK=${String(rawTopK)}: expected integer >= 0 (0 disables top-k).`,
    );
  }
  if (!Number.isFinite(rawTopP) || rawTopP <= 0.0 || rawTopP > 1.0) {
    throw new RangeError(
      `Invalid sampling topP=${String(rawTopP)}: expected finite number in (0, 1].`,
    );
  }

  return Object.freeze({
    temperature: rawTemp === 0 ? 0 : rawTemp,
    topK: rawTopK,
    topP: rawTopP,
  });
}

/**
 * Converts scaled logits into an exact binary64 probability distribution applying
 * `(temperature, topK, topP)` filtering and ascending-order renormalization.
 *
 * Transform Consistency Invariant (`Audit §8.2`, Finding 3):
 * Both draft sampling (`d ~ q`) and rejection/residual evaluation (`(p, q)`) invoke this exact
 * function with identical `DorySamplingConfig`, ensuring sampling and verification operate on
 * identical filtered probability distributions.
 */
export function logitsToSamplingProbsFloat64(
  scaledLogits: ArrayLike<number>,
  offset = 0,
  length: number = scaledLogits.length - offset,
  sampling?: Partial<DorySamplingConfig>,
): Float64Array {
  if (!Number.isInteger(offset) || offset < 0) {
    throw new RangeError(`Invalid logits offset=${String(offset)}: expected integer >= 0.`);
  }
  if (!Number.isInteger(length) || length < 1 || offset + length > scaledLogits.length) {
    throw new RangeError(
      `Invalid logits slice (offset=${offset}, length=${length}) for array length ${scaledLogits.length}.`,
    );
  }

  let hasFinite = false;
  for (let i = 0; i < length; i += 1) {
    const v = scaledLogits[offset + i];
    if (Number.isNaN(v) || v === Infinity) {
      throw new DorySpecSupportInvariantError(
        'INVALID_DISTRIBUTION',
        `Invalid scaled logit at index ${offset + i}: ${String(v)}.`,
      );
    }
    if (Number.isFinite(v)) {
      hasFinite = true;
    }
  }
  if (!hasFinite) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      'All scaled logits in slice are -Infinity; cannot form probability distribution.',
    );
  }

  const cfg = normalizeSamplingConfig(sampling);

  // Greedy limit (T = 0): exact one-hot at deterministic argmax (lowest index on tie)
  if (cfg.temperature === 0) {
    const probs = new Float64Array(length);
    probs[argmaxIndex(scaledLogits, offset, length)] = 1.0;
    return probs;
  }

  // Base binary64 softmax at temperature T > 0
  const probs = softmaxFloat64(scaledLogits, offset, length, cfg.temperature);
  const applyTopK = cfg.topK > 0 && cfg.topK < length;
  const applyTopP = cfg.topP < 1.0;

  if (!applyTopK && !applyTopP) {
    return probs;
  }

  // Sort candidate token indices descending by probability (then raw logit, then ascending index)
  const order = new Int32Array(length);
  for (let i = 0; i < length; i += 1) {
    order[i] = i;
  }
  order.sort((a, b) => {
    const diffProb = probs[b] - probs[a];
    if (diffProb !== 0.0) {
      return diffProb > 0.0 ? 1 : -1;
    }
    const diffLogit = scaledLogits[offset + b] - scaledLogits[offset + a];
    if (diffLogit !== 0.0) {
      return diffLogit > 0.0 ? 1 : -1;
    }
    return a - b;
  });

  let keptLimit = length;

  // Step 1: Top-k cutoff + binary64 ascending renormalization
  if (applyTopK) {
    keptLimit = cfg.topK;
    for (let r = keptLimit; r < length; r += 1) {
      probs[order[r]] = 0.0;
    }
    let sumTopK = 0.0;
    for (let i = 0; i < length; i += 1) {
      sumTopK += probs[i];
    }
    if (!(sumTopK > 0.0) || !Number.isFinite(sumTopK)) {
      throw new DorySpecSupportInvariantError(
        'INVALID_DISTRIBUTION',
        `Top-k filtering (k=${cfg.topK}) yielded non-positive probability mass ${String(sumTopK)}.`,
      );
    }
    for (let i = 0; i < length; i += 1) {
      if (probs[i] > 0.0) {
        probs[i] /= sumTopK;
      }
    }
  }

  // Step 2: Top-p (nucleus) cutoff keeping minimal prefix with cumulative mass >= topP (at least 1)
  if (applyTopP) {
    let cumProb = probs[order[0]];
    let nucleusCount = 1;
    while (nucleusCount < keptLimit && cumProb < cfg.topP) {
      cumProb += probs[order[nucleusCount]];
      nucleusCount += 1;
    }
    for (let r = nucleusCount; r < keptLimit; r += 1) {
      probs[order[r]] = 0.0;
    }
    let sumTopP = 0.0;
    for (let i = 0; i < length; i += 1) {
      sumTopP += probs[i];
    }
    if (!(sumTopP > 0.0) || !Number.isFinite(sumTopP)) {
      throw new DorySpecSupportInvariantError(
        'INVALID_DISTRIBUTION',
        `Top-p filtering (p=${cfg.topP}) yielded non-positive probability mass ${String(sumTopP)}.`,
      );
    }
    for (let i = 0; i < length; i += 1) {
      if (probs[i] > 0.0) {
        probs[i] /= sumTopP;
      }
    }
  }

  return probs;
}

/**
 * Inverse-CDF binary64 categorical sampler in fixed ascending index order `0 .. probs.length - 1`.
 * Strictly guarantees the sampled index carries `probs[idx] > 0` (never falls back to a trailing
 * zero-probability bin under `1e-16` rounding drift).
 */
export function sampleCategoricalFloat64(probs: Float64Array, uniformVariate: number): number {
  assertValidUniformVariate(uniformVariate, 'uniformVariate');
  const len = probs.length;
  if (len <= 0) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      'Cannot sample from an empty probability distribution.',
    );
  }

  let cdf = 0.0;
  let lastPositive = -1;
  let chosen = -1;

  for (let v = 0; v < len; v += 1) {
    const p = probs[v];
    if (!Number.isFinite(p) || p < 0.0) {
      throw new DorySpecSupportInvariantError(
        'INVALID_DISTRIBUTION',
        `Invalid categorical probability probs[${v}]=${String(p)}: expected finite number >= 0.`,
      );
    }
    if (p > 0.0) {
      lastPositive = v;
      cdf += p;
      if (chosen === -1 && uniformVariate < cdf) {
        chosen = v;
      }
    }
  }

  if (lastPositive === -1) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      'Cannot sample from categorical distribution with zero total probability mass.',
    );
  }

  return chosen !== -1 ? chosen : lastPositive;
}

// ============================================================================
// DIVISION-FREE BINARY64 REJECTION SAMPLING & RESIDUAL (AUDIT §8.2, FINDING 3)
// ============================================================================

/**
 * Evaluates exact division-free binary64 rejection acceptance for proposed `draftTokenId`:
 *   - Greedy (`isGreedy === true`): accept iff `draftTokenId === argmax(targetProbs)`.
 *   - Stochastic (`isGreedy === false`):
 *       1. Assert `q[d] > 0` (`DorySpecSupportInvariantError('NON_POSITIVE_PROPOSAL_PROB')`) —
 *          NO `1e-38` denominator clamping!
 *       2. If `p[d] >= q[d]`: accept unconditionally (`uniformVariate: null`, zero rejections on `p === q`).
 *       3. Else (`p[d] < q[d]`): accept iff **`uniformVariate * q[d] < p[d]`** (zero division!).
 */
export function evaluateRejectionAcceptance(
  draftTokenId: number,
  targetProbs: Float64Array,
  proposalProbs: Float64Array,
  uniformVariate: number | null,
  isGreedy = false,
): DoryRejectionSampleResult {
  const V = targetProbs.length;
  if (V <= 0 || proposalProbs.length !== V) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      `Target/proposal length mismatch or empty distribution: target=${V}, proposal=${proposalProbs.length}.`,
    );
  }
  if (!Number.isInteger(draftTokenId) || draftTokenId < 0 || draftTokenId >= V) {
    throw new RangeError(
      `Invalid draftTokenId=${String(draftTokenId)}: expected integer in [0, ${V - 1}].`,
    );
  }

  const pDraft = targetProbs[draftTokenId];
  const qDraft = proposalProbs[draftTokenId];
  if (!Number.isFinite(pDraft) || pDraft < 0.0 || !Number.isFinite(qDraft) || qDraft < 0.0) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      `Invalid probabilities at draftTokenId=${draftTokenId}: p=${String(pDraft)}, q=${String(qDraft)}.`,
    );
  }

  if (isGreedy) {
    const targetBest = argmaxIndex(targetProbs, 0, V);
    const accepted = draftTokenId === targetBest;
    return {
      draftTokenId,
      draftProb: qDraft,
      targetProb: pDraft,
      uniformVariate: null,
      accepted,
      reason: accepted ? 'greedy_match' : 'greedy_mismatch',
    };
  }

  // Hard support invariant: a token sampled from q must carry strictly positive probability
  if (!(qDraft > 0.0)) {
    throw new DorySpecSupportInvariantError(
      'NON_POSITIVE_PROPOSAL_PROB',
      `Proposal probability q[${draftTokenId}]=${String(qDraft)} must be strictly > 0 (zero 1e-38 clamp).`,
    );
  }

  // Unconditional acceptance whenever p[d] >= q[d] (guarantees 0 rejections when p === q)
  if (pDraft >= qDraft) {
    return {
      draftTokenId,
      draftProb: qDraft,
      targetProb: pDraft,
      uniformVariate: null,
      accepted: true,
      reason: 'unconditional_p_ge_q',
    };
  }

  // Division-free product test: accept iff r * q[d] < p[d] (exact at q = 1e-40, p = 5e-41!)
  if (uniformVariate === null) {
    throw new RangeError(
      `Stochastic rejection test for draftTokenId=${draftTokenId} (p=${pDraft} < q=${qDraft}) requires a finite uniformVariate in [0, 1).`,
    );
  }
  assertValidUniformVariate(uniformVariate, 'uniformVariate');

  const accepted = uniformVariate * qDraft < pDraft;
  return {
    draftTokenId,
    draftProb: qDraft,
    targetProb: pDraft,
    uniformVariate,
    accepted,
    reason: accepted ? 'stochastic_accept' : 'stochastic_reject',
  };
}

/**
 * Computes the exact binary64 rejection-sampling residual distribution:
 *   `res[v] = Math.max(0, p[v] - q[v])`, `Z = sum_v res[v]`, `probs[v] = res[v] / Z`.
 *
 * Strictly forbids any `Z <= 1e-12` fallback to `p` (Finding 3): divides whenever `Z > 0`
 * (preserving exact `[1.0, 0.0]` at `Delta = 4e-13`) and throws `DorySpecSupportInvariantError`
 * with `code = 'ZERO_RESIDUAL_MASS'` if `Z <= 0`.
 */
export function computeRejectionResidualFloat64(
  targetProbs: Float64Array,
  proposalProbs: Float64Array,
): DoryResidualDistributionResult {
  const V = targetProbs.length;
  if (V <= 0 || proposalProbs.length !== V) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      `Target/proposal length mismatch or empty distribution: target=${V}, proposal=${proposalProbs.length}.`,
    );
  }

  const probs = new Float64Array(V);
  let massZ = 0.0;

  for (let v = 0; v < V; v += 1) {
    const pv = targetProbs[v];
    const qv = proposalProbs[v];
    if (!Number.isFinite(pv) || pv < 0.0 || !Number.isFinite(qv) || qv < 0.0) {
      throw new DorySpecSupportInvariantError(
        'INVALID_DISTRIBUTION',
        `Invalid probability at index ${v}: p=${String(pv)}, q=${String(qv)}.`,
      );
    }
    const diff = pv - qv;
    if (diff > 0.0) {
      probs[v] = diff;
      massZ += diff;
    } else {
      probs[v] = 0.0;
    }
  }

  if (!(massZ > 0.0) || !Number.isFinite(massZ)) {
    throw new DorySpecSupportInvariantError(
      'ZERO_RESIDUAL_MASS',
      `Rejection residual normalization mass Z=${String(massZ)} must be strictly positive (zero 1e-12 fallback).`,
    );
  }

  for (let v = 0; v < V; v += 1) {
    if (probs[v] > 0.0) {
      probs[v] /= massZ;
    }
  }

  return { probs, massZ };
}

/**
 * Executes one standalone rejection + residual sampling step against `(targetProbs, proposalProbs)`
 * for Gate 2 Layers 3, 4, and 6 verification.
 */
export function sampleSpecSingleStep(
  targetProbs: Float64Array,
  proposalProbs: Float64Array,
  prng?: DoryPrng,
  explicitUniforms?: {
    readonly draftUniform?: number;
    readonly acceptUniform?: number;
    readonly residualUniform?: number;
  },
  isGreedy = false,
): DorySpecSingleStepSampleResult {
  const V = targetProbs.length;
  if (V <= 0 || proposalProbs.length !== V) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      `Target/proposal length mismatch or empty distribution: target=${V}, proposal=${proposalProbs.length}.`,
    );
  }

  if (isGreedy) {
    validateProbabilityVector(targetProbs, 'targetProbs');
    validateProbabilityVector(proposalProbs, 'proposalProbs');
    const draftTokenId = argmaxIndex(proposalProbs, 0, V);
    const decision = evaluateRejectionAcceptance(
      draftTokenId,
      targetProbs,
      proposalProbs,
      null,
      true,
    );
    const emittedTokenId = decision.accepted ? draftTokenId : argmaxIndex(targetProbs, 0, V);
    return {
      draftTokenId,
      emittedTokenId,
      decision,
      residual: null,
    };
  }

  const drawUniform = (explicit: number | undefined, name: string): number => {
    if (explicit !== undefined) {
      return assertValidUniformVariate(explicit, name);
    }
    if (prng === undefined) {
      throw new RangeError(
        `Stochastic sampleSpecSingleStep requires either prng or explicitUniforms.${name}.`,
      );
    }
    return prng.nextFloat64();
  };

  validateProbabilityVector(targetProbs, 'targetProbs');
  const uDraft = drawUniform(explicitUniforms?.draftUniform, 'draftUniform');
  const draftTokenId = sampleCategoricalFloat64(proposalProbs, uDraft);

  const pDraft = targetProbs[draftTokenId];
  const qDraft = proposalProbs[draftTokenId];
  const uAccept =
    pDraft >= qDraft ? null : drawUniform(explicitUniforms?.acceptUniform, 'acceptUniform');

  const decision = evaluateRejectionAcceptance(
    draftTokenId,
    targetProbs,
    proposalProbs,
    uAccept,
    false,
  );

  if (decision.accepted) {
    return {
      draftTokenId,
      emittedTokenId: draftTokenId,
      decision,
      residual: null,
    };
  }

  const residual = computeRejectionResidualFloat64(targetProbs, proposalProbs);
  const uRes = drawUniform(explicitUniforms?.residualUniform, 'residualUniform');
  const emittedTokenId = sampleCategoricalFloat64(residual.probs, uRes);

  return {
    draftTokenId,
    emittedTokenId,
    decision,
    residual,
  };
}

// ============================================================================
// EXACT ANALYTICAL ENUMERATION & PEARSON CHI-SQUARED (GATE 2 LAYERS 1, 2, 6)
// ============================================================================

/**
 * Exact binary64 1-step analytical marginal integration over all draft outcomes `d in 0..V-1`
 * (Gate 2 Layers 1 & 2, zero Monte Carlo variance; asserts `maxAbsError <= 1e-15`).
 */
export function enumerateSpecSingleStepExact(
  targetProbs: Float64Array,
  proposalProbs: Float64Array,
): DorySpecAnalyticalMarginalResult {
  const V = targetProbs.length;
  if (V <= 0 || proposalProbs.length !== V) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      `Target/proposal length mismatch or empty distribution: target=${V}, proposal=${proposalProbs.length}.`,
    );
  }
  validateProbabilityVector(targetProbs, 'targetProbs');
  validateProbabilityVector(proposalProbs, 'proposalProbs');

  const marginalProbs = new Float64Array(V);
  let acceptanceRate = 0.0;
  let rejectionRate = 0.0;

  for (let d = 0; d < V; d += 1) {
    const qd = proposalProbs[d];
    const pd = targetProbs[d];
    if (qd === 0.0) {
      continue;
    }
    if (pd >= qd) {
      // Unconditional acceptance: P(D = d, Accept) = q[d]
      marginalProbs[d] += qd;
      acceptanceRate += qd;
    } else {
      // Division-free acceptance r * q[d] < p[d] has exact probability p[d] / q[d]:
      // P(D = d, Accept) = q[d] * (p[d] / q[d]) = p[d], P(D = d, Reject) = q[d] - p[d]
      marginalProbs[d] += pd;
      acceptanceRate += pd;
      rejectionRate += qd - pd;
    }
  }

  if (rejectionRate > 0.0) {
    const residual = computeRejectionResidualFloat64(targetProbs, proposalProbs);
    for (let v = 0; v < V; v += 1) {
      marginalProbs[v] += rejectionRate * residual.probs[v];
    }
  }

  let maxAbsError = 0.0;
  for (let v = 0; v < V; v += 1) {
    const err = Math.abs(marginalProbs[v] - targetProbs[v]);
    if (err > maxAbsError) {
      maxAbsError = err;
    }
  }

  return {
    marginalProbs,
    acceptanceRate,
    maxAbsError,
  };
}

/**
 * Exact binary64 2-step joint speculative marginal enumeration over all `(d1, accept/reject, d2, accept/reject)`
 * outcomes (`V = 4` Gate 2 Layer 2; asserts `maxAbsError <= 1e-15`).
 */
export function enumerateSpecTwoStepExact(
  targetP1: Float64Array,
  proposalQ1: Float64Array,
  targetP2ByToken1: readonly Float64Array[],
  proposalQ2ByToken1: readonly Float64Array[],
): DorySpecTwoStepAnalyticalResult {
  const V = targetP1.length;
  if (
    V <= 0 ||
    proposalQ1.length !== V ||
    targetP2ByToken1.length !== V ||
    proposalQ2ByToken1.length !== V
  ) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      `Invalid 2-step enumeration dimensions for V=${V}.`,
    );
  }
  validateProbabilityVector(targetP1, 'targetP1');
  validateProbabilityVector(proposalQ1, 'proposalQ1');

  // Precompute exact step-2 speculative marginals conditioned on each emitted v1 in 0..V-1
  const step2ByV1 = new Array<DorySpecAnalyticalMarginalResult>(V);
  for (let v1 = 0; v1 < V; v1 += 1) {
    step2ByV1[v1] = enumerateSpecSingleStepExact(targetP2ByToken1[v1], proposalQ2ByToken1[v1]);
  }

  // Exact step-1 residual distribution (only needed if rejection rate > 0)
  let totalReject1 = 0.0;
  for (let d1 = 0; d1 < V; d1 += 1) {
    const q1 = proposalQ1[d1];
    const p1 = targetP1[d1];
    if (q1 > 0.0 && p1 < q1) {
      totalReject1 += q1 - p1;
    }
  }
  const residual1 =
    totalReject1 > 0.0 ? computeRejectionResidualFloat64(targetP1, proposalQ1) : null;

  const jointProbs = new Float64Array(V * V);

  // Explicitly integrate over every step-1 draft outcome d1 in 0..V-1
  for (let d1 = 0; d1 < V; d1 += 1) {
    const q1 = proposalQ1[d1];
    const p1 = targetP1[d1];
    if (q1 === 0.0) {
      continue;
    }

    if (p1 >= q1) {
      // Branch A1: d1 accepted unconditionally -> v1 = d1, then step-2 speculative outcome
      const step2Probs = step2ByV1[d1].marginalProbs;
      const rowOff = d1 * V;
      for (let v2 = 0; v2 < V; v2 += 1) {
        jointProbs[rowOff + v2] += q1 * step2Probs[v2];
      }
    } else {
      // Branch A2: d1 accepted with probability p1 / q1 -> prob mass p1 on v1 = d1
      const step2AccProbs = step2ByV1[d1].marginalProbs;
      const accRowOff = d1 * V;
      for (let v2 = 0; v2 < V; v2 += 1) {
        jointProbs[accRowOff + v2] += p1 * step2AccProbs[v2];
      }

      // Branch B: d1 rejected with prob mass (q1 - p1) -> resample v1 ~ p_res1, then step-2 spec
      const rejMass = q1 - p1;
      const res1Probs = residual1!.probs;
      for (let v1 = 0; v1 < V; v1 += 1) {
        const w1 = rejMass * res1Probs[v1];
        if (w1 === 0.0) {
          continue;
        }
        const step2RejProbs = step2ByV1[v1].marginalProbs;
        const rejRowOff = v1 * V;
        for (let v2 = 0; v2 < V; v2 += 1) {
          jointProbs[rejRowOff + v2] += w1 * step2RejProbs[v2];
        }
      }
    }
  }

  const targetJointProbs = new Float64Array(V * V);
  let maxAbsError = 0.0;
  for (let v1 = 0; v1 < V; v1 += 1) {
    const p1 = targetP1[v1];
    const p2Row = targetP2ByToken1[v1];
    const rowOff = v1 * V;
    for (let v2 = 0; v2 < V; v2 += 1) {
      const idx = rowOff + v2;
      const targetVal = p1 * p2Row[v2];
      targetJointProbs[idx] = targetVal;
      const err = Math.abs(jointProbs[idx] - targetVal);
      if (err > maxAbsError) {
        maxAbsError = err;
      }
    }
  }

  return {
    vocabSize: V,
    jointProbs,
    targetJointProbs,
    maxAbsError,
  };
}

/**
 * Computes Pearson's chi-squared goodness-of-fit statistic and empirical TVD in binary64
 * (Gate 2 Layer 6; for `V = 64`, `N = 20,000`, `alpha = 0.001`, critical value is `103.51`).
 */
export function computePearsonChiSquared(
  observedCounts: ArrayLike<number>,
  expectedProbs: ArrayLike<number>,
  sampleCount: number,
): DoryPearsonChiSquaredResult {
  const bins = observedCounts.length;
  if (bins < 1 || expectedProbs.length !== bins) {
    throw new RangeError(
      `observedCounts/expectedProbs length mismatch or empty: obs=${bins}, exp=${expectedProbs.length}.`,
    );
  }
  if (!Number.isInteger(sampleCount) || sampleCount < 1) {
    throw new RangeError(`sampleCount must be an integer >= 1; got ${String(sampleCount)}.`);
  }

  let statistic = 0.0;
  let activeBins = 0;
  let minExpectedCount = Infinity;
  let l1Diff = 0.0;

  for (let i = 0; i < bins; i += 1) {
    const obs = observedCounts[i];
    const prob = expectedProbs[i];
    if (!Number.isFinite(obs) || obs < 0.0) {
      throw new RangeError(`Invalid observedCounts[${i}]=${String(obs)}: expected finite >= 0.`);
    }
    if (!Number.isFinite(prob) || prob < 0.0) {
      throw new RangeError(`Invalid expectedProbs[${i}]=${String(prob)}: expected finite >= 0.`);
    }

    const empProb = obs / sampleCount;
    l1Diff += Math.abs(empProb - prob);

    if (prob === 0.0) {
      if (obs > 0.0) {
        statistic = Infinity;
      }
      continue;
    }

    activeBins += 1;
    const expected = sampleCount * prob;
    if (expected < minExpectedCount) {
      minExpectedCount = expected;
    }
    if (Number.isFinite(statistic)) {
      const delta = obs - expected;
      statistic += (delta * delta) / expected;
    }
  }

  if (activeBins === 0) {
    throw new DorySpecSupportInvariantError(
      'INVALID_DISTRIBUTION',
      'expectedProbs has zero positive bins.',
    );
  }

  return {
    statistic,
    degreesOfFreedom: Math.max(0, activeBins - 1),
    sampleCount,
    minExpectedCount,
    totalVariationDistance: 0.5 * l1Diff,
  };
}

// ============================================================================
// CONTRACT A §6.2 SELF-SPECULATION STATE MACHINE (104 BANKS, TRANSITIONS 1–10)
// ============================================================================

/**
 * Contract A Initial Prefill (Transition 1, `Audit §8.1.1` & Finding 1):
 *   1. Allocates a fresh `104`-bank Contract A KV cache (`per_loop`, `kLoops = 4`,
 *      `includeShallowCoda = true`).
 *   2. Runs batched `k = 4` Deep prefill over `promptTokens[0 .. P - 1]` (commits `96` Deep banks
 *      and retains `s_1^{(0..P-1)}`).
 *   3. Runs ONE batched Shallow Coda pass over cached `s_1^{(0..P-1)}` into the `8` persistent
 *      `output_shallow` banks (committing all `104` banks identically to `committedLen = P`).
 *   4. Samples initial unconsumed `pendingToken = d_0` from Deep Coda filtered probabilities at
 *      position `P - 1`, establishing the exact cycle-boundary invariant `(committedLen = P, d_0)`.
 */
export function prefillDorySelfSpec(input: DorySpecCyclePrefillInput): DorySpecCyclePrefillResult {
  const { params } = input;
  const config = params.config;
  const promptLen = input.promptTokens.length;

  if (!Number.isInteger(promptLen) || promptLen < 1) {
    throw new RangeError(`promptTokens must be non-empty; got length ${promptLen}.`);
  }
  if (promptLen > config.maxPositionEmbeddings) {
    throw new RangeError(
      `promptTokens length ${promptLen} exceeds maxPositionEmbeddings=${config.maxPositionEmbeddings}.`,
    );
  }

  const promptTokens = new Int32Array(promptLen);
  for (let i = 0; i < promptLen; i += 1) {
    const tok = input.promptTokens[i];
    if (!Number.isInteger(tok) || tok < 0 || tok >= config.vocabSize) {
      throw new RangeError(
        `Invalid prompt token ID ${String(tok)} at index ${i} (expected integer in 0..${config.vocabSize - 1}).`,
      );
    }
    promptTokens[i] = tok;
  }

  const sampling = normalizeSamplingConfig(input.sampling);
  const isGreedy = sampling.temperature === 0;
  if (!isGreedy && input.initialSampleUniform !== undefined) {
    assertValidUniformVariate(input.initialSampleUniform, 'initialSampleUniform');
  }

  const ropeTables = input.ropeTables ?? buildDoryDualRopeTables(config);
  const sessionBranchId = input.sessionBranchId ?? 'b0';
  const forwardStepId = input.forwardStepId ?? 0;
  const captureTrace = input.captureTrace ?? false;

  // Allocate fresh 104-bank Contract A KV cache (96 Deep per_loop + 8 persistent output_shallow)
  const kvCache = createDoryKvCache({
    mode: 'per_loop',
    kLoops: DORY_SPEC_VERIFY_LOOPS,
    includeShallowCoda: true,
    config,
  });

  // Step 1: Batched k = 4 Deep prefill across [0 .. P - 1] (commits 96 Deep banks)
  const deepPrefill = runDoryForwardChunk({
    tokens: promptTokens,
    params,
    kvCache,
    startPos: 0,
    kLoops: DORY_SPEC_VERIFY_LOOPS,
    kvCacheMode: 'per_loop',
    ropeTables,
    codaKind: 'output_deep',
    writeKv: true,
    captureTrace,
    forwardStepId,
    sessionBranchId,
    role: 'spec_verify_deep',
  });

  // Step 2: Batched Shallow Coda pass over cached s_1^{(0..P-1)} into 8 output_shallow banks
  const shallowCoda = runDoryOutputCoda({
    state: deepPrefill.loopExitStates[0],
    seqLen: promptLen,
    startPos: 0,
    kLoops: DORY_SPEC_DRAFT_LOOPS,
    params,
    kvCache,
    ropeTables,
    codaKind: 'output_shallow',
    writeKv: true,
    captureTrace,
    forwardStepId,
    sessionBranchId,
    role: 'spec_draft',
  });

  const vocabSize = config.vocabSize;
  const lastIdx = promptLen - 1;
  const lastDeepScaledLogits = sliceFloat32Row(deepPrefill.scaledLogits, lastIdx, vocabSize);
  const lastDeepProbs = logitsToSamplingProbsFloat64(lastDeepScaledLogits, 0, vocabSize, sampling);
  const lastShallowScaledLogits = sliceFloat32Row(
    shallowCoda.lmHead.scaledLogits,
    lastIdx,
    vocabSize,
  );
  const lastShallowProbs = logitsToSamplingProbsFloat64(
    lastShallowScaledLogits,
    0,
    vocabSize,
    sampling,
  );

  // Step 3: Sample initial unconsumed pendingToken = d_0 from Deep Coda at pos P - 1
  let pendingToken: number;
  if (isGreedy) {
    pendingToken = argmaxIndex(lastDeepScaledLogits, 0, vocabSize);
  } else {
    const u0 =
      input.initialSampleUniform !== undefined
        ? input.initialSampleUniform
        : (input.prng ?? createDoryPrng(deriveDorySeed(DORY_DEFAULT_SESSION_SEED, forwardStepId))).nextFloat64();
    pendingToken = sampleCategoricalFloat64(lastDeepProbs, u0);
  }

  const emittedTokens = new Int32Array(promptLen + 1);
  emittedTokens.set(promptTokens, 0);
  emittedTokens[promptLen] = pendingToken;

  const lastTokenLoopMetrics = computeAllLoopMetricsForToken(
    deepPrefill.xAnchor,
    deepPrefill.loopEntryStates,
    deepPrefill.loopExitStates,
    lastIdx,
    lastIdx,
    config,
  );

  let haltReason: DoryHaltReason | null = null;
  if (isEosTokenId(pendingToken, config)) {
    haltReason = 'eos';
  } else if (promptLen >= config.maxPositionEmbeddings) {
    haltReason = 'capacity_exhausted';
  }

  return {
    contractBadge: DORY_SPEC_CONTRACT_A_BADGE,
    kvCache,
    promptTokens,
    committedLen: promptLen,
    pendingToken,
    emittedTokens,
    xAnchor: deepPrefill.xAnchor,
    loopEntryStates: deepPrefill.loopEntryStates,
    loopExitStates: deepPrefill.loopExitStates,
    lastTokenLoopMetrics,
    lastDeepScaledLogits,
    lastDeepProbs,
    lastShallowScaledLogits,
    lastShallowProbs,
    haltReason,
    deepTrace: deepPrefill.trace,
    shallowCodaTrace: captureTrace ? shallowCoda.executedSublayers : null,
  };
}

/**
 * Executes one complete Contract A §6.2 self-speculative decoding cycle (Transitions 2–9,
 * `Audit §8.1.1` & Finding 1), preserving the exact 104-bank cycle-boundary invariant:
 *   - All `104` banks end committed strictly through `[0 .. endCommittedLen - 1]`.
 *   - `nextPendingToken` is unconsumed (or `null` on `full_accept_eos`).
 *   - `0` extra Shallow Coda work on rejection (`a < gamma'`); 1-token Shallow Coda catch-up
 *     on cached `s_1^{(L + gamma')}` on full non-EOS acceptance (or final slot).
 *   - Transaction-local `O(1)` SWA rollback (`cap_swa = 16`, `absPosTag`) via `rewindDoryKvCache`.
 */
export function runDorySelfSpecCycle(input: DorySelfSpecCycleInput): DorySelfSpecCycleResult {
  const { kvCache, pendingToken, params } = input;
  const config = params.config;
  const vocabSize = config.vocabSize;
  const hiddenSize = config.hiddenSize;

  // Validate Contract A 104-bank invariants & cycle-boundary synchronization before any mutation
  assertContractAKvCache(kvCache, config);

  const startCommittedLen = kvCache.committedLen;
  if (startCommittedLen >= config.maxPositionEmbeddings) {
    throw new RangeError(
      `Cannot start self-speculative cycle: committedLen=${startCommittedLen} has reached maxPositionEmbeddings=${config.maxPositionEmbeddings}.`,
    );
  }
  if (!Number.isInteger(pendingToken) || pendingToken < 0 || pendingToken >= vocabSize) {
    throw new RangeError(
      `Invalid pendingToken=${String(pendingToken)}: expected integer in 0..${vocabSize - 1}.`,
    );
  }
  if (isEosTokenId(pendingToken, config)) {
    throw new RangeError(
      `Cannot start self-speculative cycle: pendingToken=${pendingToken} is an EOS token (already halted).`,
    );
  }

  const gamma = input.gamma ?? config.speculativeGamma;
  if (!Number.isInteger(gamma) || gamma < 1) {
    throw new RangeError(`Invalid speculative gamma=${String(gamma)}: expected integer >= 1.`);
  }
  if (config.slidingWindow + gamma + 1 > config.swaRingCapacity) {
    throw new RangeError(
      `DR19 SWA rollback horizon violation: W(${config.slidingWindow}) + gamma(${gamma}) + 1 exceeds cap_swa(${config.swaRingCapacity}).`,
    );
  }

  const sampling = normalizeSamplingConfig(input.sampling);
  const isGreedy = sampling.temperature === 0;

  // Validate any explicitly provided uniform variates up front before mutating kvCache
  const explicit = input.explicitVariates;
  if (explicit !== undefined) {
    if (explicit.draftSampleUniforms !== undefined) {
      for (let i = 0; i < explicit.draftSampleUniforms.length; i += 1) {
        assertValidUniformVariate(
          explicit.draftSampleUniforms[i],
          `explicitVariates.draftSampleUniforms[${i}]`,
        );
      }
    }
    if (explicit.acceptUniforms !== undefined) {
      for (let i = 0; i < explicit.acceptUniforms.length; i += 1) {
        assertValidUniformVariate(
          explicit.acceptUniforms[i],
          `explicitVariates.acceptUniforms[${i}]`,
        );
      }
    }
    if (explicit.correctionOrBonusUniform !== undefined) {
      assertValidUniformVariate(
        explicit.correctionOrBonusUniform,
        'explicitVariates.correctionOrBonusUniform',
      );
    }
  }

  const ropeTables = input.ropeTables ?? buildDoryDualRopeTables(config);
  const sessionBranchId = input.sessionBranchId ?? 'b0';
  const forwardStepId = input.forwardStepId ?? kvCache.chunkSizes.length;
  const captureTrace = input.captureTrace ?? false;

  let fallbackPrng: DoryPrng | null = null;
  const nextUniform = (): number => {
    if (input.prng !== undefined) {
      return input.prng.nextFloat64();
    }
    if (fallbackPrng === null) {
      fallbackPrng = createDoryPrng(deriveDorySeed(DORY_DEFAULT_SESSION_SEED, forwardStepId + 1));
    }
    return fallbackPrng.nextFloat64();
  };

  const gammaEffective = Math.min(gamma, config.maxPositionEmbeddings - 1 - startCommittedLen);

  // ==========================================================================
  // SPECIAL TRANSITION: gammaEffective === 0 (final remaining slot L = S_max - 1)
  // ==========================================================================
  if (gammaEffective === 0) {
    const deepRes = runDoryForwardChunk({
      tokens: [pendingToken],
      params,
      kvCache,
      startPos: startCommittedLen,
      kLoops: DORY_SPEC_VERIFY_LOOPS,
      kvCacheMode: 'per_loop',
      ropeTables,
      codaKind: 'output_deep',
      writeKv: true,
      captureTrace,
      forwardStepId,
      sessionBranchId,
      role: 'spec_verify_deep',
    });

    // 1-token Shallow Coda catch-up on cached s_1^{(L)} so all 104 banks commit identically to S_max
    runDoryOutputCoda({
      state: deepRes.loopExitStates[0],
      seqLen: 1,
      startPos: startCommittedLen,
      kLoops: DORY_SPEC_DRAFT_LOOPS,
      params,
      kvCache,
      ropeTables,
      codaKind: 'output_shallow',
      writeKv: true,
      captureTrace: false,
      forwardStepId,
      sessionBranchId,
      role: 'spec_draft',
    });

    const deepScaledLogits = sliceFloat32Row(deepRes.scaledLogits, 0, vocabSize);
    const targetProbs = logitsToSamplingProbsFloat64(deepScaledLogits, 0, vocabSize, sampling);

    let yFinal: number;
    let uFinal: number | null = null;
    if (isGreedy) {
      yFinal = argmaxIndex(deepScaledLogits, 0, vocabSize);
    } else {
      uFinal = explicit?.correctionOrBonusUniform ?? nextUniform();
      yFinal = sampleCategoricalFloat64(targetProbs, uFinal);
    }

    const endCommittedLen = startCommittedLen + 1;
    const haltReason: DoryHaltReason = isEosTokenId(yFinal, config) ? 'eos' : 'capacity_exhausted';
    const lastCommittedLoopMetrics = computeAllLoopMetricsForToken(
      deepRes.xAnchor,
      deepRes.loopEntryStates,
      deepRes.loopExitStates,
      0,
      startCommittedLen,
      config,
    );

    const verifyPositions: DorySpecVerifyPositionRecord[] = [
      {
        verifyOffset: 0,
        absPos: startCommittedLen,
        inputTokenId: pendingToken,
        reusedPreludeAndLoop1: false,
        deepScaledLogits,
        targetProbs,
        rejection: null,
      },
    ];

    return {
      contractBadge: DORY_SPEC_CONTRACT_A_BADGE,
      startCommittedLen,
      endCommittedLen,
      inputPendingToken: pendingToken,
      nextPendingToken: yFinal,
      gammaEffective: 0,
      draftCount: 0,
      draftEndedWithEos: false,
      acceptedDraftCount: 0,
      emittedTokens: Int32Array.of(yFinal),
      outcome: 'final_slot_deep_decode',
      haltReason,
      reusedActivationPositions: 0,
      verifiedPositions: 1,
      shallowCodaCatchupExecuted: true,
      rolledBackPositions: 0,
      draftSteps: [],
      verifyPositions,
      residualDistribution: null,
      correctionOrBonusToken: yFinal,
      correctionOrBonusUniform: uFinal,
      lastCommittedXAnchor: sliceFloat32Row(deepRes.xAnchor, 0, hiddenSize),
      lastCommittedLoopEntryStates: deepRes.loopEntryStates.map((s) =>
        sliceFloat32Row(s, 0, hiddenSize),
      ),
      lastCommittedLoopExitStates: deepRes.loopExitStates.map((s) =>
        sliceFloat32Row(s, 0, hiddenSize),
      ),
      lastCommittedLoopMetrics,
      lastCommittedDeepScaledLogits: deepScaledLogits,
      lastCommittedDeepProbs: targetProbs,
      verifyTrace: deepRes.trace,
    };
  }

  // ==========================================================================
  // TRANSITION 3: SHALLOW DRAFT PHASE (k = 1, j = 0 .. gammaEffective - 1)
  // ==========================================================================
  const tokensSeq: number[] = [pendingToken];
  const reusedXAnchors: Float32Array[] = [];
  const reusedU1States: Float32Array[] = [];
  const reusedS1States: Float32Array[] = [];
  const draftSteps: DorySpecDraftStepRecord[] = [];
  let draftEndedWithEos = false;

  let lastReusedPreludeTrace: readonly DoryExecutedSublayerRecord[] = [];
  let lastReusedLoop1Trace: readonly DoryExecutedSublayerRecord[] = [];

  for (let j = 0; j < gammaEffective; j += 1) {
    const absPos = startCommittedLen + j;
    const inputTokenId = tokensSeq[j];

    // Prelude (0..15) -> writes shared Input KV & caches anchor x^{(L+j)}
    const prelude = runDoryInputPrelude({
      tokens: [inputTokenId],
      startPos: absPos,
      params,
      kvCache,
      ropeTables,
      writeKv: true,
      captureTrace,
      forwardStepId,
      sessionBranchId,
      role: 'spec_verify_reused',
      kLoopsForCoord: DORY_SPEC_VERIFY_LOOPS,
    });

    // Recurrent Loop 1 (16..55, t = 1) -> writes shared Loop-1 KV & caches (u_1, s_1)^{(L+j)}
    const loop1 = runDoryRecurrentPass({
      prevState: prelude.xAnchor,
      xAnchor: prelude.xAnchor,
      seqLen: 1,
      startPos: absPos,
      loopIteration: 1,
      kLoops: DORY_SPEC_VERIFY_LOOPS,
      params,
      kvCache,
      ropeTables,
      writeKv: true,
      captureTrace,
      forwardStepId,
      sessionBranchId,
      role: 'spec_verify_reused',
    });

    if (captureTrace) {
      lastReusedPreludeTrace = prelude.executedSublayers;
      lastReusedLoop1Trace = loop1.executedSublayers;
    }

    // Shallow Coda (56..71) against persistent output_shallow KV -> exact standalone k = 1 logits!
    const shallowCoda = runDoryOutputCoda({
      state: loop1.sExit,
      seqLen: 1,
      startPos: absPos,
      kLoops: DORY_SPEC_DRAFT_LOOPS,
      params,
      kvCache,
      ropeTables,
      codaKind: 'output_shallow',
      writeKv: true,
      captureTrace: false,
      forwardStepId,
      sessionBranchId,
      role: 'spec_draft',
    });

    const shallowScaledLogits = sliceFloat32Row(shallowCoda.lmHead.scaledLogits, 0, vocabSize);
    const proposalProbs = logitsToSamplingProbsFloat64(
      shallowScaledLogits,
      0,
      vocabSize,
      sampling,
    );

    let proposedTokenId: number;
    let sampleUniform: number | null = null;
    if (isGreedy) {
      proposedTokenId = argmaxIndex(shallowScaledLogits, 0, vocabSize);
    } else {
      sampleUniform =
        explicit?.draftSampleUniforms !== undefined && j < explicit.draftSampleUniforms.length
          ? explicit.draftSampleUniforms[j]
          : nextUniform();
      proposedTokenId = sampleCategoricalFloat64(proposalProbs, sampleUniform);
    }

    const proposedIsEos = isEosTokenId(proposedTokenId, config);

    reusedXAnchors.push(prelude.xAnchor);
    reusedU1States.push(loop1.uEntry);
    reusedS1States.push(loop1.sExit);
    tokensSeq.push(proposedTokenId);
    draftSteps.push({
      draftOffset: j,
      absPos,
      inputTokenId,
      proposedTokenId,
      proposedIsEos,
      shallowScaledLogits,
      proposalProbs,
      sampleUniform,
    });

    if (proposedIsEos) {
      draftEndedWithEos = true;
      break;
    }
  }

  const draftCount = draftSteps.length;
  const needBonusSlot =
    !draftEndedWithEos && startCommittedLen + draftCount < config.maxPositionEmbeddings;
  const verifiedPositions = needBonusSlot ? draftCount + 1 : draftCount;

  // ==========================================================================
  // TRANSITION 4: BATCHED DEEP VERIFY PHASE (k = 4, REUSING PRELUDE + LOOP 1)
  // ==========================================================================
  let bonusXAnchor: Float32Array | null = null;
  let bonusU1: Float32Array | null = null;
  let bonusS1: Float32Array | null = null;

  if (needBonusSlot) {
    const bonusPos = startCommittedLen + draftCount;
    const bonusTok = tokensSeq[draftCount];

    // 1-token Prelude + Loop 1 ONLY for bonus position L + gamma' (output_shallow untouched!)
    const bonusPrelude = runDoryInputPrelude({
      tokens: [bonusTok],
      startPos: bonusPos,
      params,
      kvCache,
      ropeTables,
      writeKv: true,
      captureTrace: false,
      forwardStepId,
      sessionBranchId,
      role: 'spec_verify_deep',
      kLoopsForCoord: DORY_SPEC_VERIFY_LOOPS,
    });

    const bonusLoop1 = runDoryRecurrentPass({
      prevState: bonusPrelude.xAnchor,
      xAnchor: bonusPrelude.xAnchor,
      seqLen: 1,
      startPos: bonusPos,
      loopIteration: 1,
      kLoops: DORY_SPEC_VERIFY_LOOPS,
      params,
      kvCache,
      ropeTables,
      writeKv: true,
      captureTrace: false,
      forwardStepId,
      sessionBranchId,
      role: 'spec_verify_deep',
    });

    bonusXAnchor = bonusPrelude.xAnchor;
    bonusU1 = bonusLoop1.uEntry;
    bonusS1 = bonusLoop1.sExit;
  }

  const xAnchorList = needBonusSlot ? [...reusedXAnchors, bonusXAnchor!] : reusedXAnchors;
  const u1List = needBonusSlot ? [...reusedU1States, bonusU1!] : reusedU1States;
  const s1List = needBonusSlot ? [...reusedS1States, bonusS1!] : reusedS1States;

  const batchXAnchor = concatFloat32Vectors(xAnchorList, hiddenSize);
  const batchU1 = concatFloat32Vectors(u1List, hiddenSize);
  const batchS1 = concatFloat32Vectors(s1List, hiddenSize);

  const batchLoopEntryStates: Float32Array[] = [batchU1];
  const batchLoopExitStates: Float32Array[] = [batchS1];

  const verifyTraceSublayers: DoryExecutedSublayerRecord[] = captureTrace
    ? [...lastReusedPreludeTrace, ...lastReusedLoop1Trace]
    : [];
  const verifyTraceLoopMetrics: DoryLoopTrajectoryMetrics[] = [];

  if (captureTrace) {
    for (let offset = 0; offset < verifiedPositions; offset += 1) {
      verifyTraceLoopMetrics.push(
        computeSingleLoopTokenMetric(
          1,
          batchXAnchor,
          batchXAnchor,
          batchU1,
          batchS1,
          offset,
          startCommittedLen + offset,
          config,
        ),
      );
    }
  }

  // Batched Recurrent Loops t = 2, 3, 4 over all verifiedPositions [L .. L + m - 1]
  let batchState: Float32Array = batchS1;
  for (let t = 2; t <= DORY_SPEC_VERIFY_LOOPS; t += 1) {
    const pass = runDoryRecurrentPass({
      prevState: batchState,
      xAnchor: batchXAnchor,
      seqLen: verifiedPositions,
      startPos: startCommittedLen,
      loopIteration: t,
      kLoops: DORY_SPEC_VERIFY_LOOPS,
      params,
      kvCache,
      ropeTables,
      writeKv: true,
      captureTrace,
      forwardStepId,
      sessionBranchId,
      role: 'spec_verify_deep',
    });

    batchLoopEntryStates.push(pass.uEntry);
    batchLoopExitStates.push(pass.sExit);
    batchState = pass.sExit;

    if (captureTrace) {
      for (let i = 0; i < pass.executedSublayers.length; i += 1) {
        verifyTraceSublayers.push(pass.executedSublayers[i]);
      }
      for (let i = 0; i < pass.loopMetrics.length; i += 1) {
        verifyTraceLoopMetrics.push(pass.loopMetrics[i]);
      }
    }
  }

  // Batched Deep Coda (56..71) against committed output_deep KV [0 .. L - 1]
  const deepCoda = runDoryOutputCoda({
    state: batchState,
    seqLen: verifiedPositions,
    startPos: startCommittedLen,
    kLoops: DORY_SPEC_VERIFY_LOOPS,
    params,
    kvCache,
    ropeTables,
    codaKind: 'output_deep',
    writeKv: true,
    captureTrace,
    forwardStepId,
    sessionBranchId,
    role: 'spec_verify_deep',
  });

  if (captureTrace) {
    for (let i = 0; i < deepCoda.executedSublayers.length; i += 1) {
      verifyTraceSublayers.push(deepCoda.executedSublayers[i]);
    }
  }

  // ==========================================================================
  // REJECTION SAMPLING LOOP (j = 0 .. draftCount - 1)
  // ==========================================================================
  const verifyPositions: DorySpecVerifyPositionRecord[] = new Array<DorySpecVerifyPositionRecord>(
    verifiedPositions,
  );
  let acceptedDraftCount = 0;
  let rejectedAtOffset = -1;

  for (let j = 0; j < draftCount; j += 1) {
    const step = draftSteps[j];
    const deepScaledLogits = sliceFloat32Row(deepCoda.lmHead.scaledLogits, j, vocabSize);
    const targetProbs = logitsToSamplingProbsFloat64(deepScaledLogits, 0, vocabSize, sampling);

    let rejection: DoryRejectionSampleResult | null = null;
    if (rejectedAtOffset === -1) {
      if (isGreedy) {
        rejection = evaluateRejectionAcceptance(
          step.proposedTokenId,
          targetProbs,
          step.proposalProbs,
          null,
          true,
        );
      } else {
        const pDraft = targetProbs[step.proposedTokenId];
        const qDraft = step.proposalProbs[step.proposedTokenId];
        let uAccept: number | null = null;
        if (qDraft > 0.0 && pDraft < qDraft) {
          uAccept =
            explicit?.acceptUniforms !== undefined && j < explicit.acceptUniforms.length
              ? explicit.acceptUniforms[j]
              : nextUniform();
        }
        rejection = evaluateRejectionAcceptance(
          step.proposedTokenId,
          targetProbs,
          step.proposalProbs,
          uAccept,
          false,
        );
      }

      if (rejection.accepted) {
        acceptedDraftCount += 1;
      } else {
        rejectedAtOffset = j;
      }
    }

    verifyPositions[j] = {
      verifyOffset: j,
      absPos: startCommittedLen + j,
      inputTokenId: step.inputTokenId,
      reusedPreludeAndLoop1: true,
      deepScaledLogits,
      targetProbs,
      rejection,
    };
  }

  if (needBonusSlot) {
    const deepScaledLogits = sliceFloat32Row(deepCoda.lmHead.scaledLogits, draftCount, vocabSize);
    const targetProbs = logitsToSamplingProbsFloat64(deepScaledLogits, 0, vocabSize, sampling);
    verifyPositions[draftCount] = {
      verifyOffset: draftCount,
      absPos: startCommittedLen + draftCount,
      inputTokenId: tokensSeq[draftCount],
      reusedPreludeAndLoop1: false,
      deepScaledLogits,
      targetProbs,
      rejection: null,
    };
  }

  // ==========================================================================
  // TRANSITIONS 5–8: COMMIT / TRANSACTION-LOCAL REWIND ACROSS ALL 104 BANKS
  // ==========================================================================
  const maxDeepWrittenLen = startCommittedLen + verifiedPositions;

  let endCommittedLen: number;
  let nextPendingToken: number | null;
  let emittedTokens: Int32Array;
  let outcome: DorySpecOutcomeKind;
  let haltReason: DoryHaltReason | null = null;
  let shallowCodaCatchupExecuted = false;
  let residualDistribution: DoryResidualDistributionResult | null = null;
  let correctionOrBonusToken: number | null = null;
  let correctionOrBonusUniform: number | null = null;
  let finalCommittedOffset: number;

  if (rejectedAtOffset !== -1) {
    // Transitions 5 & 6: Zero or partial acceptance (0 <= a < gamma').
    // Zero extra Shallow Coda work: output_shallow already holds [L .. L + a] from Stage A!
    const a = rejectedAtOffset;
    outcome = a === 0 ? 'zero_accept_reject' : 'partial_accept_reject';
    endCommittedLen = startCommittedLen + a + 1;
    finalCommittedOffset = a;

    let yStar: number;
    if (isGreedy) {
      yStar = argmaxIndex(verifyPositions[a].deepScaledLogits, 0, vocabSize);
    } else {
      residualDistribution = computeRejectionResidualFloat64(
        verifyPositions[a].targetProbs,
        draftSteps[a].proposalProbs,
      );
      correctionOrBonusUniform = explicit?.correctionOrBonusUniform ?? nextUniform();
      yStar = sampleCategoricalFloat64(residualDistribution.probs, correctionOrBonusUniform);
    }

    correctionOrBonusToken = yStar;
    nextPendingToken = yStar;

    emittedTokens = new Int32Array(a + 1);
    for (let i = 0; i < a; i += 1) {
      emittedTokens[i] = draftSteps[i].proposedTokenId;
    }
    emittedTokens[a] = yStar;

    if (isEosTokenId(yStar, config)) {
      haltReason = 'eos';
    } else if (endCommittedLen >= config.maxPositionEmbeddings) {
      haltReason = 'capacity_exhausted';
    }

    // Atomic transaction-local O(1)-per-bank rewind across all 104 banks (DR19)
    rewindDoryKvCache(kvCache, endCommittedLen);
  } else if (draftEndedWithEos) {
    // Transition 8: Full acceptance ending on draft EOS -> clean halt without bonus token!
    outcome = 'full_accept_eos';
    endCommittedLen = startCommittedLen + draftCount;
    finalCommittedOffset = draftCount - 1;
    nextPendingToken = null;
    haltReason = 'eos';

    emittedTokens = new Int32Array(draftCount);
    for (let i = 0; i < draftCount; i += 1) {
      emittedTokens[i] = draftSteps[i].proposedTokenId;
    }

    // All 104 banks already stand at L + draftCount; synchronize committedLen & chunkSizes[]
    rewindDoryKvCache(kvCache, endCommittedLen);
  } else {
    // Transition 7: Full non-EOS acceptance (a = gamma') -> bonus token + 1-token Shallow Coda catch-up
    outcome = 'full_accept_bonus';
    endCommittedLen = startCommittedLen + draftCount + 1;
    finalCommittedOffset = draftCount;

    let yBonus: number;
    if (isGreedy) {
      yBonus = argmaxIndex(verifyPositions[draftCount].deepScaledLogits, 0, vocabSize);
    } else {
      correctionOrBonusUniform = explicit?.correctionOrBonusUniform ?? nextUniform();
      yBonus = sampleCategoricalFloat64(
        verifyPositions[draftCount].targetProbs,
        correctionOrBonusUniform,
      );
    }

    correctionOrBonusToken = yBonus;
    nextPendingToken = yBonus;

    emittedTokens = new Int32Array(draftCount + 1);
    for (let i = 0; i < draftCount; i += 1) {
      emittedTokens[i] = draftSteps[i].proposedTokenId;
    }
    emittedTokens[draftCount] = yBonus;

    // 1-token Shallow Coda catch-up on cached s_1^{(L + gamma')} into output_shallow[L + gamma']
    runDoryOutputCoda({
      state: bonusS1!,
      seqLen: 1,
      startPos: startCommittedLen + draftCount,
      kLoops: DORY_SPEC_DRAFT_LOOPS,
      params,
      kvCache,
      ropeTables,
      codaKind: 'output_shallow',
      writeKv: true,
      captureTrace: false,
      forwardStepId,
      sessionBranchId,
      role: 'spec_draft',
    });
    shallowCodaCatchupExecuted = true;

    if (isEosTokenId(yBonus, config)) {
      haltReason = 'eos';
    } else if (endCommittedLen >= config.maxPositionEmbeddings) {
      haltReason = 'capacity_exhausted';
    }

    // All 104 banks now stand at L + draftCount + 1; commit committedLen & chunkSizes[] atomically
    rewindDoryKvCache(kvCache, endCommittedLen);
  }

  const rolledBackPositions = maxDeepWrittenLen - endCommittedLen;
  const finalCommittedAbsPos = endCommittedLen - 1;

  const lastCommittedXAnchor = sliceFloat32Row(batchXAnchor, finalCommittedOffset, hiddenSize);
  const lastCommittedLoopEntryStates = batchLoopEntryStates.map((s) =>
    sliceFloat32Row(s, finalCommittedOffset, hiddenSize),
  );
  const lastCommittedLoopExitStates = batchLoopExitStates.map((s) =>
    sliceFloat32Row(s, finalCommittedOffset, hiddenSize),
  );
  const lastCommittedLoopMetrics = computeAllLoopMetricsForToken(
    batchXAnchor,
    batchLoopEntryStates,
    batchLoopExitStates,
    finalCommittedOffset,
    finalCommittedAbsPos,
    config,
  );
  const lastCommittedDeepScaledLogits = verifyPositions[finalCommittedOffset].deepScaledLogits;
  const lastCommittedDeepProbs = verifyPositions[finalCommittedOffset].targetProbs;

  const verifyTrace: DoryForwardTrace | null = captureTrace
    ? {
        modelWeightId: params.modelWeightId,
        sessionBranchId,
        forwardStepId,
        kLoops: DORY_SPEC_VERIFY_LOOPS,
        kvCacheMode: 'per_loop',
        role: 'spec_verify_deep',
        codaKind: 'output_deep',
        seqLen: verifiedPositions,
        startPos: startCommittedLen,
        executedSublayerCount: computeExecutedDepth(DORY_SPEC_VERIFY_LOOPS, config),
        executedSublayers: verifyTraceSublayers,
        loopMetrics: verifyTraceLoopMetrics,
      }
    : null;

  return {
    contractBadge: DORY_SPEC_CONTRACT_A_BADGE,
    startCommittedLen,
    endCommittedLen,
    inputPendingToken: pendingToken,
    nextPendingToken,
    gammaEffective,
    draftCount,
    draftEndedWithEos,
    acceptedDraftCount,
    emittedTokens,
    outcome,
    haltReason,
    reusedActivationPositions: draftCount,
    verifiedPositions,
    shallowCodaCatchupExecuted,
    rolledBackPositions,
    draftSteps,
    verifyPositions,
    residualDistribution,
    correctionOrBonusToken,
    correctionOrBonusUniform,
    lastCommittedXAnchor,
    lastCommittedLoopEntryStates,
    lastCommittedLoopExitStates,
    lastCommittedLoopMetrics,
    lastCommittedDeepScaledLogits,
    lastCommittedDeepProbs,
    verifyTrace,
  };
}

/**
 * Strips `[V]` logit/probability arrays from `DorySelfSpecCycleResult` into a compact,
 * scalar-only `DorySpecCycleSummary` safe for permanent retention in `DorySessionRecord`s.
 */
export function summarizeSelfSpecCycle(result: DorySelfSpecCycleResult): DorySpecCycleSummary {
  const draftSteps: DorySpecDraftStepSummary[] = new Array<DorySpecDraftStepSummary>(
    result.draftSteps.length,
  );
  for (let i = 0; i < result.draftSteps.length; i += 1) {
    const step = result.draftSteps[i];
    draftSteps[i] = {
      draftOffset: step.draftOffset,
      absPos: step.absPos,
      inputTokenId: step.inputTokenId,
      proposedTokenId: step.proposedTokenId,
      proposedIsEos: step.proposedIsEos,
      proposedProb: step.proposalProbs[step.proposedTokenId],
    };
  }

  const verifyPositions: DorySpecVerifyPositionSummary[] = new Array<DorySpecVerifyPositionSummary>(
    result.verifyPositions.length,
  );
  for (let i = 0; i < result.verifyPositions.length; i += 1) {
    const vp = result.verifyPositions[i];
    const hasDraftAtOffset = vp.verifyOffset < result.draftSteps.length;
    const targetProbForProposed = hasDraftAtOffset
      ? vp.targetProbs[result.draftSteps[vp.verifyOffset].proposedTokenId]
      : null;
    verifyPositions[i] = {
      verifyOffset: vp.verifyOffset,
      absPos: vp.absPos,
      inputTokenId: vp.inputTokenId,
      reusedPreludeAndLoop1: vp.reusedPreludeAndLoop1,
      targetProbForProposed,
      rejection: vp.rejection !== null ? { ...vp.rejection } : null,
    };
  }

  return {
    contractBadge: result.contractBadge,
    startCommittedLen: result.startCommittedLen,
    endCommittedLen: result.endCommittedLen,
    inputPendingToken: result.inputPendingToken,
    nextPendingToken: result.nextPendingToken,
    gammaEffective: result.gammaEffective,
    draftCount: result.draftCount,
    draftEndedWithEos: result.draftEndedWithEos,
    acceptedDraftCount: result.acceptedDraftCount,
    emittedTokens: new Int32Array(result.emittedTokens),
    outcome: result.outcome,
    haltReason: result.haltReason,
    reusedActivationPositions: result.reusedActivationPositions,
    verifiedPositions: result.verifiedPositions,
    shallowCodaCatchupExecuted: result.shallowCodaCatchupExecuted,
    rolledBackPositions: result.rolledBackPositions,
    residualMassZ: result.residualDistribution !== null ? result.residualDistribution.massZ : null,
    correctionOrBonusToken: result.correctionOrBonusToken,
    draftSteps,
    verifyPositions,
  };
}
