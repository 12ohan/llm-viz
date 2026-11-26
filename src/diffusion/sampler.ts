/**
 * Strictly Separated Dual Samplers & Causal Sampler for Nemotron-Labs-Diffusion
 * (`src/diffusion/sampler.ts`)
 *
 * Implements the three mathematically distinct samplers verified in DF0 §5.2:
 *   1. Causal / Inter-Block / Verify sampler (`sampleCausalToken`):
 *      `argmax` at `T <= 0`, `multinomial(softmax(logits / T))` at `T > 0`.
 *   2. Sampler A (`runSamplerATransferStep`, for `dlm` `generate()` `_get_transfer_index`):
 *      Float64 Gumbel-max with overflow guard at `T > 0`, ALWAYS UNSCALED
 *      `softmax(logits)` confidence, `select_index[0]` unconditional floor +
 *      non-strict `conf >= threshold` (or static front-loaded quota when `threshold === null`).
 *   3. Sampler B (`runSamplerBDraftStep`, for `linear_spec_generate()` draft):
 *      `argmax` + unscaled softmax at `T <= 0`, `multinomial(softmax(logits / T))`
 *      + TEMPERATURE-SCALED confidence at `T > 0`, single-pass commit-all at
 *      `threshold <= 0.0`, non-strict `conf >= threshold` with `argmax` fallback at `threshold > 0.0`.
 *
 * ZERO `[MASK]` logit clamping anywhere — raw logits preserved for exact PyTorch parity.
 * Pure headless TypeScript — deterministic, self-contained kernel.
 */

import { softmaxRows } from './attention';
import { dualHashHex } from './prng';
import {
  DIFFUSION_MINI_MASK_TOKEN_ID,
  type CausalSampleResult,
  type SamplerATransferInput,
  type SamplerATransferResult,
  type SamplerBDraftInput,
  type SamplerBDraftResult,
  type SamplerVariateProvenance,
  type Xoshiro128ssGenerator,
} from './types';

const OPEN_UNIT_MIN = 1e-15;
const OPEN_UNIT_MAX = 1 - 1e-15;

function clampOpenUnit(u: number): number {
  if (u < OPEN_UNIT_MIN) return OPEN_UNIT_MIN;
  if (u > OPEN_UNIT_MAX) return OPEN_UNIT_MAX;
  return u;
}

function hashFloat64Variates(values: ArrayLike<number>): string | null {
  const n = values.length;
  if (n === 0) return null;
  const bytes = new Uint8Array(n * 8);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < n; i++) {
    view.setFloat64(i * 8, values[i]!, true);
  }
  return dualHashHex(bytes);
}

function argmaxRow(logits: Float32Array, rowOffset: number, vocabSize: number): number {
  let bestIdx = 0;
  let bestVal = logits[rowOffset]!;
  for (let v = 1; v < vocabSize; v++) {
    const val = logits[rowOffset + v]!;
    if (val > bestVal) {
      bestVal = val;
      bestIdx = v;
    }
  }
  return bestIdx;
}

// ============================================================================
// STATIC QUOTA & INVERSE-CDF CATEGORICAL SAMPLER
// ============================================================================

/**
 * Front-loaded integer division matching upstream `_get_num_transfer_tokens`:
 *   - `B = 4`, `3` masks (`causal_context = true`, `steps = 4`) -> `[1, 1, 1, 0]`
 *   - `B = 4`, `4` masks (`causal_context = false`, `steps = 4`) -> `[1, 1, 1, 1]`
 */
export function computeStaticTransferQuota(numMasked: number, steps: number): Int32Array {
  if (!Number.isInteger(numMasked) || numMasked < 0) {
    throw new RangeError(`numMasked must be a non-negative integer, got ${numMasked}`);
  }
  if (!Number.isInteger(steps) || steps <= 0) {
    throw new RangeError(`steps must be a positive integer, got ${steps}`);
  }

  const base = Math.floor(numMasked / steps);
  const rem = numMasked % steps;
  const quota = new Int32Array(steps);
  for (let i = 0; i < steps; i++) {
    quota[i] = base + (i < rem ? 1 : 0);
  }
  return quota;
}

/**
 * Inverse-CDF multinomial sampling over `probs[0..vocabSize-1]` using `uniform in [0, 1)`.
 */
export function sampleCategoricalFromProbs(
  probs: Float32Array,
  vocabSize: number,
  uniform: number,
): number {
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new RangeError(`vocabSize must be a positive integer, got ${vocabSize}`);
  }
  if (probs.length < vocabSize) {
    throw new RangeError(`probs length (${probs.length}) < vocabSize (${vocabSize})`);
  }

  let total = 0;
  for (let v = 0; v < vocabSize; v++) {
    const p = probs[v]!;
    if (p > 0) total += p;
  }
  if (!(total > 0)) {
    return 0;
  }

  const u = Math.min(Math.max(uniform, 0), OPEN_UNIT_MAX);
  const target = u * total;
  let cdf = 0;
  let lastPositive = 0;
  for (let v = 0; v < vocabSize; v++) {
    const p = probs[v]!;
    if (p > 0) {
      lastPositive = v;
      cdf += p;
      if (target < cdf) {
        return v;
      }
    }
  }
  return lastPositive;
}

// ============================================================================
// CAUSAL / AR / INTER-BLOCK / VERIFY SAMPLER
// ============================================================================

/**
 * Samples a single token from a 1D logit row with zero `[MASK]` clamping:
 *   - `temperature <= 0`: deterministic `argmax` (first tie wins)
 *   - `temperature > 0`: `softmaxRows(logits / temperature)` + inverse-CDF categorical
 */
export function sampleCausalToken(
  logitsRow: Float32Array,
  temperature: number,
  prng?: Xoshiro128ssGenerator,
  explicitUniform?: number,
): CausalSampleResult {
  const vocabSize = logitsRow.length;
  if (vocabSize <= 0) {
    throw new RangeError('logitsRow must be non-empty');
  }

  if (temperature <= 0) {
    const tokenId = argmaxRow(logitsRow, 0, vocabSize);
    const probs = softmaxRows(logitsRow, 1, vocabSize);
    return {
      tokenId,
      probability: probs[tokenId]!,
      temperature,
      variateUsed: null,
      variateProvenance: {
        mode: 'greedy_none',
        variateDigest: null,
        count: 0,
      },
    };
  }

  if (explicitUniform === undefined && prng === undefined) {
    throw new Error(
      'sampleCausalToken requires prng or explicitUniform when temperature > 0',
    );
  }

  const scaled = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    scaled[v] = Math.fround(logitsRow[v]! / temperature);
  }
  const probs = softmaxRows(scaled, 1, vocabSize);
  const u = explicitUniform !== undefined ? explicitUniform : prng!.nextUniformOpenUnit();
  const tokenId = sampleCategoricalFromProbs(probs, vocabSize, u);

  return {
    tokenId,
    probability: probs[tokenId]!,
    temperature,
    variateUsed: u,
    variateProvenance: {
      mode: explicitUniform !== undefined ? 'caller_explicit' : 'prng_xoshiro128ss',
      variateDigest: hashFloat64Variates([u]),
      count: 1,
    },
  };
}

// ============================================================================
// SAMPLER A — `dlm` `generate()` `_get_transfer_index`
// ============================================================================

/**
 * Sampler A (`generate()` `_get_transfer_index`, DF0 §5.2):
 *   - Zero `[MASK]` logit clamping.
 *   - Candidate token `x0[t]`: `argmax(logits[t])` at `T <= 0`, or overflow-guarded
 *     Float64 Gumbel-max `exp(l - maxLogit) / (-ln U)^T` at `T > 0`.
 *   - Confidence `conf[t]`: ALWAYS UNSCALED `softmaxRows(logits)[t, x0[t]]`
 *     for masked slots, `-Infinity` for unmasked slots.
 *   - Selection: static top-`min(numTransferTokens, maskedCount)` when
 *     `threshold === null`, or unconditional rank-0 floor + non-strict
 *     `conf >= threshold` for rank `k >= 1` when `threshold !== null`.
 */
export function runSamplerATransferStep(
  input: SamplerATransferInput,
): SamplerATransferResult {
  const {
    logits,
    blockTokens,
    seqLen,
    vocabSize,
    numTransferTokens,
    config,
    prng,
    explicitGumbelUniforms,
  } = input;

  if (!Number.isInteger(seqLen) || seqLen <= 0) {
    throw new RangeError(`seqLen must be a positive integer, got ${seqLen}`);
  }
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new RangeError(`vocabSize must be a positive integer, got ${vocabSize}`);
  }
  if (logits.length < seqLen * vocabSize) {
    throw new RangeError(
      `logits length (${logits.length}) < seqLen * vocabSize (${seqLen * vocabSize})`,
    );
  }
  if (blockTokens.length < seqLen) {
    throw new RangeError(`blockTokens length (${blockTokens.length}) < seqLen (${seqLen})`);
  }

  const maskTokenId = config.maskTokenId ?? DIFFUSION_MINI_MASK_TOKEN_ID;
  const candidateTokens = new Int32Array(seqLen);
  let variateProvenance: SamplerVariateProvenance;

  // Confidence is ALWAYS UNSCALED softmax over raw logits (even when temperature > 0)
  const unscaledProbs = softmaxRows(logits, seqLen, vocabSize);

  if (config.temperature <= 0) {
    for (let t = 0; t < seqLen; t++) {
      candidateTokens[t] = argmaxRow(logits, t * vocabSize, vocabSize);
    }
    variateProvenance = {
      mode: 'greedy_none',
      variateDigest: null,
      count: 0,
    };
  } else {
    const totalEntries = seqLen * vocabSize;
    if (explicitGumbelUniforms === undefined && prng === undefined) {
      throw new Error(
        'runSamplerATransferStep requires prng or explicitGumbelUniforms when temperature > 0',
      );
    }
    if (
      explicitGumbelUniforms !== undefined &&
      explicitGumbelUniforms.length < totalEntries
    ) {
      throw new RangeError(
        `explicitGumbelUniforms length (${explicitGumbelUniforms.length}) < seqLen * vocabSize (${totalEntries})`,
      );
    }

    const usedVariates = new Float64Array(totalEntries);
    const temp = config.temperature;

    for (let t = 0; t < seqLen; t++) {
      const rowBase = t * vocabSize;
      let maxLogit = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        const l = logits[rowBase + v]!;
        if (l > maxLogit) maxLogit = l;
      }

      let bestV = 0;
      let bestScore = -Infinity;
      for (let v = 0; v < vocabSize; v++) {
        const idx = rowBase + v;
        const rawU =
          explicitGumbelUniforms !== undefined
            ? explicitGumbelUniforms[idx]!
            : prng!.nextUniformOpenUnit();
        const u = clampOpenUnit(rawU);
        usedVariates[idx] = u;

        const denom = Math.pow(-Math.log(u), temp);
        const score = Math.exp(logits[idx]! - maxLogit) / denom;
        if (score > bestScore) {
          bestScore = score;
          bestV = v;
        }
      }
      candidateTokens[t] = bestV;
    }

    variateProvenance = {
      mode: explicitGumbelUniforms !== undefined ? 'caller_explicit' : 'prng_xoshiro128ss',
      variateDigest: hashFloat64Variates(usedVariates),
      count: totalEntries,
    };
  }

  // Gather unscaled confidences for masked slots; -Infinity for unmasked slots
  const confidences = new Float32Array(seqLen);
  const maskedIndices: number[] = [];
  for (let t = 0; t < seqLen; t++) {
    if (blockTokens[t] === maskTokenId) {
      confidences[t] = unscaledProbs[t * vocabSize + candidateTokens[t]!]!;
      maskedIndices.push(t);
    } else {
      confidences[t] = -Infinity;
    }
  }

  // Sort masked positions by descending confidence; earlier index breaks ties deterministically
  maskedIndices.sort((a, b) => {
    const diff = confidences[b]! - confidences[a]!;
    return diff !== 0 ? diff : a - b;
  });

  const transferIndex = new Uint8Array(seqLen);
  if (maskedIndices.length > 0) {
    if (config.threshold === null) {
      const quota = Math.max(0, Math.min(numTransferTokens, maskedIndices.length));
      for (let k = 0; k < quota; k++) {
        transferIndex[maskedIndices[k]!] = 1;
      }
    } else {
      // Unconditional rank-0 floor (`select_index[0]`) + non-strict `>= threshold` for rank >= 1
      transferIndex[maskedIndices[0]!] = 1;
      for (let k = 1; k < maskedIndices.length; k++) {
        const pos = maskedIndices[k]!;
        if (confidences[pos]! >= config.threshold) {
          transferIndex[pos] = 1;
        } else {
          break;
        }
      }
    }
  }

  const updatedBlock = new Int32Array(seqLen);
  let selectedCount = 0;
  let unmaskedMaskTokenCount = 0;
  let remainingMaskCount = 0;

  for (let t = 0; t < seqLen; t++) {
    if (transferIndex[t] === 1) {
      selectedCount++;
      const tok = candidateTokens[t]!;
      updatedBlock[t] = tok;
      if (tok === maskTokenId) {
        unmaskedMaskTokenCount++;
      }
    } else {
      updatedBlock[t] = blockTokens[t]!;
    }
    if (updatedBlock[t] === maskTokenId) {
      remainingMaskCount++;
    }
  }

  return {
    candidateTokens,
    confidences,
    transferIndex,
    updatedBlock,
    selectedCount,
    remainingMaskCount,
    unmaskedMaskTokenCount,
    variateProvenance,
  };
}

// ============================================================================
// SAMPLER B — `linear_spec_generate()` INLINE DRAFT SAMPLER
// ============================================================================

/**
 * Sampler B (`linear_spec_generate()` inline draft, DF0 §5.2):
 *   - Zero `[MASK]` logit clamping.
 *   - At `T <= 0`: `draftTokens[t] = argmax(draftLogits[t])`,
 *     `conf[t] = softmax(draftLogits)[t, draftTokens[t]]`.
 *   - At `T > 0`: `scaledProbs = softmax(draftLogits / T)` (TEMPERATURE-SCALED!),
 *     `draftTokens[t] ~ Categorical(scaledProbs[t])`,
 *     `conf[t] = scaledProbs[t, draftTokens[t]]`.
 *   - Unmasking: `threshold <= 0.0` commits ALL masked slots in a single pass;
 *     `threshold > 0.0` unmasks all masked slots with `conf >= threshold`
 *     (non-strict `>=`) with single-slot `argmax` fallback when none pass.
 */
export function runSamplerBDraftStep(
  input: SamplerBDraftInput,
): SamplerBDraftResult {
  const {
    draftLogits,
    blockTokens,
    seqLen,
    vocabSize,
    config,
    prng,
    explicitUniforms,
  } = input;

  if (!Number.isInteger(seqLen) || seqLen <= 0) {
    throw new RangeError(`seqLen must be a positive integer, got ${seqLen}`);
  }
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new RangeError(`vocabSize must be a positive integer, got ${vocabSize}`);
  }
  if (draftLogits.length < seqLen * vocabSize) {
    throw new RangeError(
      `draftLogits length (${draftLogits.length}) < seqLen * vocabSize (${seqLen * vocabSize})`,
    );
  }
  if (blockTokens.length < seqLen) {
    throw new RangeError(`blockTokens length (${blockTokens.length}) < seqLen (${seqLen})`);
  }

  const maskTokenId = config.maskTokenId ?? DIFFUSION_MINI_MASK_TOKEN_ID;
  const draftTokens = new Int32Array(seqLen);
  const confidences = new Float32Array(seqLen);
  let variateProvenance: SamplerVariateProvenance;

  if (config.temperature <= 0) {
    const probs = softmaxRows(draftLogits, seqLen, vocabSize);
    for (let t = 0; t < seqLen; t++) {
      const tok = argmaxRow(draftLogits, t * vocabSize, vocabSize);
      draftTokens[t] = tok;
      confidences[t] = probs[t * vocabSize + tok]!;
    }
    variateProvenance = {
      mode: 'greedy_none',
      variateDigest: null,
      count: 0,
    };
  } else {
    if (explicitUniforms === undefined && prng === undefined) {
      throw new Error(
        'runSamplerBDraftStep requires prng or explicitUniforms when temperature > 0',
      );
    }
    if (explicitUniforms !== undefined && explicitUniforms.length < seqLen) {
      throw new RangeError(
        `explicitUniforms length (${explicitUniforms.length}) < seqLen (${seqLen})`,
      );
    }

    const totalEntries = seqLen * vocabSize;
    const scaledLogits = new Float32Array(totalEntries);
    const temp = config.temperature;
    for (let i = 0; i < totalEntries; i++) {
      scaledLogits[i] = Math.fround(draftLogits[i]! / temp);
    }
    // TEMPERATURE-SCALED probabilities for both sampling and confidence!
    const scaledProbs = softmaxRows(scaledLogits, seqLen, vocabSize);
    const usedVariates = new Float64Array(seqLen);

    for (let t = 0; t < seqLen; t++) {
      const u =
        explicitUniforms !== undefined
          ? explicitUniforms[t]!
          : prng!.nextUniformOpenUnit();
      usedVariates[t] = u;
      const rowProbs = scaledProbs.subarray(t * vocabSize, (t + 1) * vocabSize);
      const tok = sampleCategoricalFromProbs(rowProbs, vocabSize, u);
      draftTokens[t] = tok;
      confidences[t] = rowProbs[tok]!;
    }

    variateProvenance = {
      mode: explicitUniforms !== undefined ? 'caller_explicit' : 'prng_xoshiro128ss',
      variateDigest: hashFloat64Variates(usedVariates),
      count: seqLen,
    };
  }

  const singlePassCommitAll = config.threshold <= 0.0;
  const unmaskMask = new Uint8Array(seqLen);

  let maskedCount = 0;
  let passedCount = 0;
  let bestMaskedIdx = -1;
  let bestMaskedConf = -Infinity;

  for (let t = 0; t < seqLen; t++) {
    if (blockTokens[t] === maskTokenId) {
      maskedCount++;
      const c = confidences[t]!;
      if (c > bestMaskedConf) {
        bestMaskedConf = c;
        bestMaskedIdx = t;
      }
      if (singlePassCommitAll || c >= config.threshold) {
        unmaskMask[t] = 1;
        passedCount++;
      }
    }
  }

  // Fallback: when threshold > 0 and no masked slot passed >= threshold, select single argmax masked slot
  if (!singlePassCommitAll && maskedCount > 0 && passedCount === 0 && bestMaskedIdx >= 0) {
    unmaskMask[bestMaskedIdx] = 1;
  }

  const updatedBlock = new Int32Array(seqLen);
  let changed = false;
  let remainingMaskCount = 0;

  for (let t = 0; t < seqLen; t++) {
    const nextTok = unmaskMask[t] === 1 ? draftTokens[t]! : blockTokens[t]!;
    updatedBlock[t] = nextTok;
    if (nextTok !== blockTokens[t]!) {
      changed = true;
    }
    if (nextTok === maskTokenId) {
      remainingMaskCount++;
    }
  }

  return {
    draftTokens,
    confidences,
    unmaskMask,
    updatedBlock,
    remainingMaskCount,
    singlePassCommitAll,
    noProgress: !changed,
    variateProvenance,
  };
}
