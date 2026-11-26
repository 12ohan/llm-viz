/**
 * Block-Diffusion Text Decoder & Object Grounding (`src/uni/textSampler.ts`)
 *
 * Clean-room headless TypeScript implementation of:
 *   1. `generateWithPrefixCacheBlockDiff`: FLARE position-shifted (`shift_logits = true`)
 *      block-diffusion text decoder with immutable prefix KV slabs, optional causal
 *      slot-0 pre-seeding (`causalContext`), `U21` `[MASK]` suppression with provenance,
 *      `U22` overflow-safe log-domain Gumbel sampling over `Xoshiro128**` or explicit
 *      uniform variates (`U23`), hard KV capacity enforcement, and per-row EOS cropping (`U30`).
 *   2. `locBinsToPixelBox` & `groundObjects`: `<LOC_0..16>` iterative grounding unmasker
 *      with single-tag / multi-tag template assembly, `U21` `<LOC_*>` domain guard
 *      provenance, exact square-letterbox `pixel_box` inversion (§4.2), and direct
 *      token-ID decoding without `skip_special_tokens=True` stripping (`U27`).
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import { softmaxRows } from './attention';
import { embedUniTextTokens, runUniForwardPass } from './forward';
import {
  appendSingleRowKvsToCache,
  createEmptyUniKvCache,
  cropUniKvCache,
  extractRowPastKvs,
} from './kvCache';
import { createXoshiro128ss } from './prng';
import {
  UNI_MINI_BOX_CLOSE_TOKEN_ID,
  UNI_MINI_BOX_OPEN_TOKEN_ID,
  UNI_MINI_EOS_TOKEN_ID,
  UNI_MINI_LOC_BEGIN_TOKEN_ID,
  UNI_MINI_LOC_END_ID,
  UNI_MINI_LOC_END_TOKEN_ID,
  UNI_MINI_LOC_START_ID,
  UNI_MINI_MASK_TOKEN_ID,
  UNI_MINI_UNK_TOKEN_ID,
  UNI_MINI_VOCAB,
  type UniCoreModelParameters,
  type UniGroundingBox,
  type UniGroundingOptions,
  type UniGroundingResult,
  type UniGroundingStepTrace,
  type UniTextBlockDiffusionOptions,
  type UniTextBlockDiffusionResult,
  type UniTextBlockStepTrace,
} from './types';

const OPEN_UNIT_MIN = 1e-15;
const OPEN_UNIT_MAX = 1 - 1e-15;
const DEFAULT_TEXT_SAMPLER_SEED = 0x554e4954; // "UNIT"

interface UniformStream {
  nextOpenUnit(): number;
}

function createUniformStream(
  seed: number,
  explicitUniformVariates?: ArrayLike<number>,
): UniformStream {
  if (explicitUniformVariates !== undefined) {
    let cursor = 0;
    return {
      nextOpenUnit(): number {
        if (cursor >= explicitUniformVariates.length) {
          throw new RangeError(
            `explicitUniformVariates exhausted at index ${cursor} (length=${explicitUniformVariates.length})`,
          );
        }
        const raw = explicitUniformVariates[cursor++];
        if (!Number.isFinite(raw)) {
          throw new RangeError(`Non-finite explicit uniform variate at index ${cursor - 1}: ${String(raw)}`);
        }
        if (raw < OPEN_UNIT_MIN) return OPEN_UNIT_MIN;
        if (raw > OPEN_UNIT_MAX) return OPEN_UNIT_MAX;
        return raw;
      },
    };
  }

  const rng = createXoshiro128ss(seed >>> 0);
  return {
    nextOpenUnit(): number {
      return rng.nextUniformOpenUnit();
    },
  };
}

/**
 * Strict `>` argmax matching PyTorch `.argmax(-1)` (tie-breaks to lowest index).
 */
function argmaxFinite(values: ArrayLike<number>, length: number): number {
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < length; i++) {
    const v = values[i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * Overflow-safe log-domain Gumbel-Max sampling (`U22`):
 *   `score[v] = (maskedLogits[v] - maxFiniteLogit) / T - log(-log(u_v))`
 *
 * Subtracting `maxFiniteLogit` before dividing by `T` preserves the exact argmax
 * distribution while preventing `exp` or division overflow even for extreme logits
 * (`> 800`, `1e30`, etc.). Always consumes `vocabSize` uniform variates in vocabulary
 * order `v = 0 .. vocabSize - 1` so TS and PyTorch stay 0-ULP aligned (`U23`).
 */
export function sampleLogDomainGumbelArgmax(
  maskedLogits: Float32Array,
  temperature: number,
  uniformStream: UniformStream,
): number {
  const vocabSize = maskedLogits.length;
  let maxLogit = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    const val = maskedLogits[v];
    if (Number.isFinite(val) && val > maxLogit) {
      maxLogit = val;
    }
  }
  if (!Number.isFinite(maxLogit)) {
    throw new RangeError('sampleLogDomainGumbelArgmax requires at least one finite logit');
  }
  if (!(temperature > 0) || !Number.isFinite(temperature)) {
    throw new RangeError(`temperature must be finite and > 0, got ${String(temperature)}`);
  }

  let bestIdx = -1;
  let bestScore = -Infinity;

  for (let v = 0; v < vocabSize; v++) {
    const u = uniformStream.nextOpenUnit();
    const logit = maskedLogits[v];
    if (!Number.isFinite(logit)) {
      continue;
    }
    const gumbelNoise = -Math.log(-Math.log(u));
    const score = (logit - maxLogit) / temperature + gumbelNoise;
    if (score > bestScore) {
      bestScore = score;
      bestIdx = v;
    }
  }

  if (bestIdx < 0) {
    throw new RangeError('No selectable finite token found during Gumbel sampling');
  }
  return bestIdx;
}

interface TextSlotDecodeOutcome {
  readonly rawTop1: number;
  readonly guardIntervened: boolean;
  readonly probs: Float32Array;
  readonly proposed: number;
  readonly confidence: number;
}

function decodeTextLogitsRow(
  rawLogitsRow: Float32Array,
  maskTokenId: number,
  suppressMaskLogit: boolean,
  temperature: number,
  uniformStream: UniformStream,
  consumeStochastic: boolean,
): TextSlotDecodeOutcome {
  const vocabSize = rawLogitsRow.length;
  const rawTop1 = argmaxFinite(rawLogitsRow, vocabSize);
  const guardIntervened = suppressMaskLogit && rawTop1 === maskTokenId;

  const maskedLogits = new Float32Array(rawLogitsRow);
  if (suppressMaskLogit && maskTokenId >= 0 && maskTokenId < vocabSize) {
    maskedLogits[maskTokenId] = -Infinity;
  }

  const probs = softmaxRows(maskedLogits, 1, vocabSize);

  let proposed: number;
  if (consumeStochastic && temperature > 0) {
    proposed = sampleLogDomainGumbelArgmax(maskedLogits, temperature, uniformStream);
  } else {
    proposed = argmaxFinite(maskedLogits, vocabSize);
  }

  return {
    rawTop1,
    guardIntervened,
    probs,
    proposed,
    confidence: probs[proposed],
  };
}

// ============================================================================
// BLOCK-DIFFUSION TEXT DECODER (FLARE SHIFTED LOGITS + PREFIX KV CACHE)
// ============================================================================

/**
 * Run FLARE position-shifted (`shift_logits = true`) block-diffusion text decoding
 * with immutable prefix KV slabs (`UniKvCacheState`).
 *
 * Supports pure text prompts (`promptTokenIds`) as well as multimodal prompts with
 * spliced RADIO-H visual embeddings (`inputsEmbeds`).
 */
export function generateWithPrefixCacheBlockDiff(
  model: UniCoreModelParameters,
  options: UniTextBlockDiffusionOptions,
): UniTextBlockDiffusionResult {
  const cfg = model.config;
  const vocabSize = cfg.vocabSize;
  const hiddenSize = cfg.hiddenSize;
  const promptLen = options.promptTokenIds.length;

  if (promptLen < 1) {
    throw new RangeError('promptTokenIds must be non-empty');
  }
  if (!Number.isInteger(options.maxNewTokens) || options.maxNewTokens < 0) {
    throw new RangeError(
      `maxNewTokens must be a non-negative integer, got ${String(options.maxNewTokens)}`,
    );
  }

  const blockLength = options.blockLength ?? 4;
  if (!Number.isInteger(blockLength) || blockLength < 1) {
    throw new RangeError(`blockLength must be a positive integer, got ${String(blockLength)}`);
  }

  const stepsPerBlock = options.stepsPerBlock ?? blockLength;
  if (!Number.isInteger(stepsPerBlock) || stepsPerBlock < 1) {
    throw new RangeError(
      `stepsPerBlock must be a positive integer, got ${String(stepsPerBlock)}`,
    );
  }

  const confidenceThreshold = options.confidenceThreshold ?? 0.9;
  const temperature = options.temperature ?? 0;
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new RangeError(`temperature must be finite and >= 0, got ${String(temperature)}`);
  }

  const causalContext = options.causalContext ?? true;
  const suppressMaskLogit = options.suppressMaskLogit ?? true;
  const eosTokenId = options.eosTokenId ?? UNI_MINI_EOS_TOKEN_ID;
  const maskTokenId = options.maskTokenId ?? UNI_MINI_MASK_TOKEN_ID;

  if (options.inputsEmbeds !== undefined && options.inputsEmbeds.length !== promptLen * hiddenSize) {
    throw new RangeError(
      `inputsEmbeds length (${options.inputsEmbeds.length}) must equal promptLen * hiddenSize (${promptLen * hiddenSize})`,
    );
  }

  if (promptLen > cfg.kvCapacity) {
    throw new RangeError(
      `promptLen (${promptLen}) exceeds KV cache capacity (${cfg.kvCapacity})`,
    );
  }

  const uniformStream = createUniformStream(
    options.seed ?? DEFAULT_TEXT_SAMPLER_SEED,
    options.explicitUniformVariates,
  );

  let cache = createEmptyUniKvCache({
    batchSize: 1,
    capacity: cfg.kvCapacity,
    prefillMode: 'hf_faithful',
    downstreamTarget: 'u',
  });

  // 1. Causal prompt prefill (U-only, downstreamTarget='u')
  const prefillRouting = new Uint8Array(promptLen);
  const prefill = runUniForwardPass(model, {
    ...(options.inputsEmbeds !== undefined
      ? { inputsEmbeds: options.inputsEmbeds, routing: prefillRouting }
      : { tokenIds: options.promptTokenIds }),
    maskMode: 'causal',
    useCache: true,
    downstreamTarget: 'u',
    computeTextHead: true,
    computeGenHead: false,
  });

  let nfeCount = 1;
  cache = appendSingleRowKvsToCache(cache, prefill.newKvs, prefill.routing);

  const prefillLogits = prefill.textLogits!;
  let nextLogitsContext = new Float32Array(
    prefillLogits.subarray((promptLen - 1) * vocabSize, promptLen * vocabSize),
  );

  const generated: number[] = [];
  const steps: UniTextBlockStepTrace[] = [];
  let stoppedByEos = false;
  let eosOffset: number | null = null;
  let stopReason: 'eos' | 'max_new_tokens' | 'capacity_exhausted' = 'max_new_tokens';
  let totalGuardInterventions = 0;
  let blockIndex = 0;

  // 2. Block-diffusion generation loop
  while (generated.length < options.maxNewTokens) {
    // Hard KV capacity enforcement: never silently shrink blockLength
    if (cache.seqLen + blockLength > cache.capacity) {
      stopReason = 'capacity_exhausted';
      break;
    }

    const blockStartPos = cache.seqLen;
    const blockTokens = new Int32Array(blockLength).fill(maskTokenId);
    let traceStepIndex = 0;

    // Optional causalContext pre-seeding of slot 0 from prior causal commit logits
    if (causalContext) {
      const before = new Int32Array(blockTokens);
      const outcome0 = decodeTextLogitsRow(
        nextLogitsContext,
        maskTokenId,
        suppressMaskLogit,
        temperature,
        uniformStream,
        true,
      );

      blockTokens[0] = outcome0.proposed;

      const shiftedLogits = new Float32Array(blockLength * vocabSize);
      shiftedLogits.set(nextLogitsContext, 0);
      const shiftedProbs = new Float32Array(blockLength * vocabSize);
      shiftedProbs.set(outcome0.probs, 0);

      const slotConfidences = new Float32Array(blockLength);
      slotConfidences[0] = outcome0.confidence;
      const rawTop1TokenIds = new Int32Array(blockLength);
      rawTop1TokenIds[0] = outcome0.rawTop1;

      const guardIntervenedOffsets: number[] = outcome0.guardIntervened ? [0] : [];
      totalGuardInterventions += guardIntervenedOffsets.length;

      steps.push({
        blockIndex,
        stepIndex: traceStepIndex++,
        blockStartPos,
        blockTokensBefore: before,
        blockTokensAfter: new Int32Array(blockTokens),
        unmaskedOffsets: [0],
        slotConfidences,
        rawTop1TokenIds,
        guardIntervenedOffsets,
        shiftedLogits,
        shiftedProbs,
        usedCausalPreSeed: true,
      });
    }

    // Extract read-only prefix KV once for all bidirectional denoising steps in this block
    const { pastKvs, pastRouting } = extractRowPastKvs(cache, 0);

    // Remaining bidirectional denoising steps (at B=1 with causalContext=true, 0 masks remain -> AR!)
    for (let s = 0; s < stepsPerBlock; s++) {
      const maskedOffsets: number[] = [];
      for (let j = 0; j < blockLength; j++) {
        if (blockTokens[j] === maskTokenId) {
          maskedOffsets.push(j);
        }
      }
      if (maskedOffsets.length === 0) {
        break;
      }

      const before = new Int32Array(blockTokens);
      const fwd = runUniForwardPass(model, {
        tokenIds: blockTokens,
        maskMode: 'bidirectional',
        useCache: false,
        downstreamTarget: 'u',
        pastKvs,
        pastRouting,
        computeTextHead: true,
        computeGenHead: false,
      });
      nfeCount += 1;

      const fwdLogits = fwd.textLogits!;
      const shiftedLogits = new Float32Array(blockLength * vocabSize);
      shiftedLogits.set(nextLogitsContext, 0);
      for (let j = 1; j < blockLength; j++) {
        shiftedLogits.set(
          fwdLogits.subarray((j - 1) * vocabSize, j * vocabSize),
          j * vocabSize,
        );
      }

      const shiftedProbs = new Float32Array(blockLength * vocabSize);
      const slotConfidences = new Float32Array(blockLength);
      const rawTop1TokenIds = new Int32Array(blockLength);
      const proposals = new Int32Array(blockLength);
      const guardIntervenedOffsets: number[] = [];

      for (let j = 0; j < blockLength; j++) {
        const isMasked = blockTokens[j] === maskTokenId;
        const rowLogits = shiftedLogits.subarray(j * vocabSize, (j + 1) * vocabSize);
        const outcome = decodeTextLogitsRow(
          rowLogits,
          maskTokenId,
          suppressMaskLogit,
          temperature,
          uniformStream,
          isMasked,
        );

        shiftedProbs.set(outcome.probs, j * vocabSize);
        rawTop1TokenIds[j] = outcome.rawTop1;
        slotConfidences[j] = outcome.confidence;
        proposals[j] = outcome.proposed;

        if (isMasked && outcome.guardIntervened) {
          guardIntervenedOffsets.push(j);
        }
      }

      totalGuardInterventions += guardIntervenedOffsets.length;

      const quota = Math.ceil(maskedOffsets.length / (stepsPerBlock - s));
      const aboveThreshold: number[] = [];
      for (let i = 0; i < maskedOffsets.length; i++) {
        const off = maskedOffsets[i];
        if (slotConfidences[off] >= confidenceThreshold) {
          aboveThreshold.push(off);
        }
      }

      let selected: number[];
      if (aboveThreshold.length >= quota) {
        selected = aboveThreshold;
      } else {
        const ranked = [...maskedOffsets].sort((a, b) => {
          const diff = slotConfidences[b] - slotConfidences[a];
          if (diff !== 0) return diff;
          return a - b;
        });
        selected = ranked.slice(0, quota).sort((a, b) => a - b);
      }

      for (let i = 0; i < selected.length; i++) {
        const off = selected[i];
        blockTokens[off] = proposals[off];
      }

      steps.push({
        blockIndex,
        stepIndex: traceStepIndex++,
        blockStartPos,
        blockTokensBefore: before,
        blockTokensAfter: new Int32Array(blockTokens),
        unmaskedOffsets: selected,
        slotConfidences,
        rawTop1TokenIds,
        guardIntervenedOffsets,
        shiftedLogits,
        shiftedProbs,
        usedCausalPreSeed: false,
      });
    }

    // 3. Causal block commit + prefix KV append
    const commit = runUniForwardPass(model, {
      tokenIds: blockTokens,
      maskMode: 'causal',
      useCache: true,
      downstreamTarget: 'u',
      pastKvs,
      pastRouting,
      computeTextHead: true,
      computeGenHead: false,
    });
    nfeCount += 1;
    cache = appendSingleRowKvsToCache(cache, commit.newKvs, commit.routing);

    const commitLogits = commit.textLogits!;
    nextLogitsContext = new Float32Array(
      commitLogits.subarray((blockLength - 1) * vocabSize, blockLength * vocabSize),
    );

    // 4. Per-row EOS detection & exact KV crop (U30)
    const remainingBudget = options.maxNewTokens - generated.length;
    const limit = Math.min(blockLength, remainingBudget);
    let eosInBlock = -1;
    for (let i = 0; i < limit; i++) {
      if (blockTokens[i] === eosTokenId) {
        eosInBlock = i;
        break;
      }
    }

    if (eosInBlock >= 0) {
      for (let i = 0; i <= eosInBlock; i++) {
        generated.push(blockTokens[i]);
      }
      cache = cropUniKvCache(cache, promptLen + generated.length);
      stoppedByEos = true;
      eosOffset = generated.length - 1;
      stopReason = 'eos';
      break;
    }

    for (let i = 0; i < limit; i++) {
      generated.push(blockTokens[i]);
    }
    if (limit < blockLength) {
      cache = cropUniKvCache(cache, promptLen + generated.length);
      stopReason = 'max_new_tokens';
      break;
    }

    blockIndex += 1;
  }

  const generatedTokenIds = Int32Array.from(generated);
  const fullTokenIds = new Int32Array(promptLen + generatedTokenIds.length);
  for (let i = 0; i < promptLen; i++) {
    fullTokenIds[i] = options.promptTokenIds[i] | 0;
  }
  fullTokenIds.set(generatedTokenIds, promptLen);

  const decodedTokens: string[] = new Array<string>(generatedTokenIds.length);
  for (let i = 0; i < generatedTokenIds.length; i++) {
    const id = generatedTokenIds[i];
    decodedTokens[i] = UNI_MINI_VOCAB[id] ?? '<unk>';
  }

  return {
    promptLen,
    blockLength,
    causalContext,
    generatedTokenIds,
    fullTokenIds,
    decodedTokens,
    stoppedByEos,
    eosOffset,
    stopReason,
    steps,
    finalCache: cache,
    nfeCount,
    totalGuardInterventions,
  };
}

// ============================================================================
// OBJECT GROUNDING (<LOC_0..16> TEMPLATE UNMASKING & PIXEL_BOX INVERSION)
// ============================================================================

/**
 * Invert square-letterboxed `<LOC_0..locBins>` coordinate bins `[bx1, by1, bx2, by2]`
 * back into source image pixel coordinates `[x1, y1, x2, y2]` (§4.2 `pixel_box`):
 *
 *   `side = max(W, H)`
 *   `x = clamp(bx * (side / locBins) - (side - W) / 2, 0, W)`
 *   `y = clamp(by * (side / locBins) - (side - H) / 2, 0, H)`
 */
export function locBinsToPixelBox(
  bins: readonly [number, number, number, number],
  imageWidth: number,
  imageHeight: number,
  locBins = 16,
): readonly [number, number, number, number] {
  if (!Number.isFinite(imageWidth) || imageWidth <= 0) {
    throw new RangeError(`imageWidth must be > 0, got ${String(imageWidth)}`);
  }
  if (!Number.isFinite(imageHeight) || imageHeight <= 0) {
    throw new RangeError(`imageHeight must be > 0, got ${String(imageHeight)}`);
  }
  if (!Number.isFinite(locBins) || locBins <= 0) {
    throw new RangeError(`locBins must be > 0, got ${String(locBins)}`);
  }

  const [bx1, by1, bx2, by2] = bins;
  const side = Math.max(imageWidth, imageHeight);
  const binScale = side / locBins;
  const padX = (side - imageWidth) / 2;
  const padY = (side - imageHeight) / 2;

  const clamp = (v: number, lo: number, hi: number): number =>
    v < lo ? lo : v > hi ? hi : v;

  return Object.freeze([
    clamp(bx1 * binScale - padX, 0, imageWidth),
    clamp(by1 * binScale - padY, 0, imageHeight),
    clamp(bx2 * binScale - padX, 0, imageWidth),
    clamp(by2 * binScale - padY, 0, imageHeight),
  ] as const);
}

function resolveGroundingTagTokenId(tag: string): number {
  const normalized = tag.trim().toLowerCase();
  if (normalized.length === 0) {
    return UNI_MINI_UNK_TOKEN_ID;
  }
  for (let id = 0; id < UNI_MINI_VOCAB.length; id++) {
    if (UNI_MINI_VOCAB[id].toLowerCase() === normalized) {
      return id;
    }
  }
  return UNI_MINI_UNK_TOKEN_ID;
}

function isGroundingSpecialToken(id: number): boolean {
  return id >= UNI_MINI_LOC_START_ID && id <= UNI_MINI_LOC_END_TOKEN_ID;
}

/**
 * Run iterative `<LOC_0..16>` object grounding with FLARE position-shifted logits,
 * optional `U21` `<LOC_*>` domain masking with provenance, exact `pixel_box`
 * un-letterboxing, and direct ID decoding that never strips `<LOC_*>` tokens (`U27`).
 */
export function groundObjects(
  model: UniCoreModelParameters,
  options: UniGroundingOptions,
): UniGroundingResult {
  const cfg = model.config;
  const vocabSize = cfg.vocabSize;
  const hiddenSize = cfg.hiddenSize;
  const promptLen = options.promptTokenIds.length;

  if (promptLen < 1) {
    throw new RangeError('promptTokenIds must be non-empty');
  }
  if (options.tags.length < 1) {
    throw new RangeError('tags must contain at least one object label');
  }
  if (!Number.isFinite(options.imageWidth) || options.imageWidth <= 0) {
    throw new RangeError(`imageWidth must be > 0, got ${String(options.imageWidth)}`);
  }
  if (!Number.isFinite(options.imageHeight) || options.imageHeight <= 0) {
    throw new RangeError(`imageHeight must be > 0, got ${String(options.imageHeight)}`);
  }

  const stepsRequested: 1 | 2 | 4 = options.steps ?? 2;
  if (stepsRequested !== 1 && stepsRequested !== 2 && stepsRequested !== 4) {
    throw new RangeError(`steps must be 1, 2, or 4, got ${String(stepsRequested)}`);
  }

  const enforceLocDomainGuard = options.enforceLocDomainGuard ?? true;

  if (
    options.promptInputsEmbeds !== undefined &&
    options.promptInputsEmbeds.length !== promptLen * hiddenSize
  ) {
    throw new RangeError(
      `promptInputsEmbeds length (${options.promptInputsEmbeds.length}) must equal promptLen * hiddenSize (${promptLen * hiddenSize})`,
    );
  }

  // 1. Assemble single-tag or multi-tag grounding template
  const templateList: number[] = [];
  const slotTemplateIndices: number[] = [];

  if (options.tags.length === 1) {
    templateList.push(UNI_MINI_LOC_BEGIN_TOKEN_ID);
    for (let k = 0; k < 4; k++) {
      slotTemplateIndices.push(templateList.length);
      templateList.push(UNI_MINI_MASK_TOKEN_ID);
    }
    templateList.push(UNI_MINI_LOC_END_TOKEN_ID, UNI_MINI_EOS_TOKEN_ID);
  } else {
    for (let i = 0; i < options.tags.length; i++) {
      const tagId = resolveGroundingTagTokenId(options.tags[i]);
      templateList.push(
        UNI_MINI_BOX_OPEN_TOKEN_ID,
        tagId,
        UNI_MINI_BOX_CLOSE_TOKEN_ID,
        UNI_MINI_LOC_BEGIN_TOKEN_ID,
      );
      for (let k = 0; k < 4; k++) {
        slotTemplateIndices.push(templateList.length);
        templateList.push(UNI_MINI_MASK_TOKEN_ID);
      }
      templateList.push(UNI_MINI_LOC_END_TOKEN_ID);
    }
    templateList.push(UNI_MINI_EOS_TOKEN_ID);
  }

  const assembledTemplateIds = Int32Array.from(templateList);
  const totalLen = promptLen + assembledTemplateIds.length;
  if (totalLen > cfg.kvCapacity) {
    throw new RangeError(
      `Grounding total sequence length (${totalLen}) exceeds KV capacity (${cfg.kvCapacity})`,
    );
  }

  const firstSlotTemplateIdx = slotTemplateIndices[0];
  const firstSlot = promptLen + firstSlotTemplateIdx;
  const totalSlots = slotTemplateIndices.length;

  const slotOffsetsInFullSequence = new Int32Array(totalSlots);
  const slotSuffixIndices = new Int32Array(totalSlots);
  for (let k = 0; k < totalSlots; k++) {
    slotOffsetsInFullSequence[k] = promptLen + slotTemplateIndices[k];
    slotSuffixIndices[k] = slotTemplateIndices[k] - firstSlotTemplateIdx;
  }

  // 2. Causal prefill of [0 .. firstSlot - 1]
  const prefillEmbeds = new Float32Array(firstSlot * hiddenSize);
  if (options.promptInputsEmbeds !== undefined) {
    prefillEmbeds.set(options.promptInputsEmbeds, 0);
  } else {
    prefillEmbeds.set(embedUniTextTokens(options.promptTokenIds, model), 0);
  }
  if (firstSlotTemplateIdx > 0) {
    const templatePrefixEmbeds = embedUniTextTokens(
      assembledTemplateIds.subarray(0, firstSlotTemplateIdx),
      model,
    );
    prefillEmbeds.set(templatePrefixEmbeds, promptLen * hiddenSize);
  }

  const prefill = runUniForwardPass(model, {
    inputsEmbeds: prefillEmbeds,
    routing: new Uint8Array(firstSlot),
    maskMode: 'causal',
    useCache: true,
    downstreamTarget: 'u',
    computeTextHead: true,
    computeGenHead: false,
  });

  let prefillCache = createEmptyUniKvCache({
    batchSize: 1,
    capacity: cfg.kvCapacity,
    prefillMode: 'hf_faithful',
    downstreamTarget: 'u',
  });
  prefillCache = appendSingleRowKvsToCache(prefillCache, prefill.newKvs, prefill.routing);

  const prevLogits = new Float32Array(
    prefill.textLogits!.subarray((firstSlot - 1) * vocabSize, firstSlot * vocabSize),
  );
  const { pastKvs, pastRouting } = extractRowPastKvs(prefillCache, 0);

  // 3. Iterative unmasking of suffix [firstSlot .. totalLen - 1]
  const suffixTokens = new Int32Array(assembledTemplateIds.subarray(firstSlotTemplateIdx));
  const stepTraces: UniGroundingStepTrace[] = [];
  const finalRawTop1 = new Int32Array(totalSlots);
  const finalGuardIntervened = new Uint8Array(totalSlots);

  for (let s = 0; s < stepsRequested; s++) {
    const remainingSlots: number[] = [];
    for (let k = 0; k < totalSlots; k++) {
      if (suffixTokens[slotSuffixIndices[k]] === UNI_MINI_MASK_TOKEN_ID) {
        remainingSlots.push(k);
      }
    }
    if (remainingSlots.length === 0) {
      break;
    }

    const slotTokenIdsBefore = new Int32Array(totalSlots);
    for (let k = 0; k < totalSlots; k++) {
      slotTokenIdsBefore[k] = suffixTokens[slotSuffixIndices[k]];
    }

    const fwd = runUniForwardPass(model, {
      tokenIds: suffixTokens,
      maskMode: 'bidirectional',
      useCache: false,
      downstreamTarget: 'u',
      pastKvs,
      pastRouting,
      computeTextHead: true,
      computeGenHead: false,
    });
    const fwdLogits = fwd.textLogits!;

    const slotConfidences = new Float32Array(totalSlots);
    const rawTop1TokenIds = new Int32Array(totalSlots);
    const proposals = new Int32Array(totalSlots);
    const guardIntervenedSlotIndices: number[] = [];

    for (let k = 0; k < totalSlots; k++) {
      const off = slotSuffixIndices[k];
      const logitsRow =
        off === 0
          ? prevLogits
          : fwdLogits.subarray((off - 1) * vocabSize, off * vocabSize);

      const rawTop1 = argmaxFinite(logitsRow, vocabSize);
      rawTop1TokenIds[k] = rawTop1;

      const isMasked = suffixTokens[off] === UNI_MINI_MASK_TOKEN_ID;
      let proposed: number;
      let conf: number;
      let intervened = false;

      if (enforceLocDomainGuard) {
        const maskedRow = new Float32Array(vocabSize).fill(-Infinity);
        for (let v = UNI_MINI_LOC_START_ID; v <= UNI_MINI_LOC_END_ID; v++) {
          maskedRow[v] = logitsRow[v];
        }
        const probsRow = softmaxRows(maskedRow, 1, vocabSize);
        proposed = argmaxFinite(maskedRow, vocabSize);
        conf = probsRow[proposed];
        intervened = rawTop1 < UNI_MINI_LOC_START_ID || rawTop1 > UNI_MINI_LOC_END_ID;
      } else {
        const probsRow = softmaxRows(logitsRow, 1, vocabSize);
        proposed = rawTop1;
        conf = probsRow[proposed];
      }

      slotConfidences[k] = conf;
      proposals[k] = proposed;
      if (isMasked && intervened) {
        guardIntervenedSlotIndices.push(k);
      }
    }

    const quota = Math.ceil(remainingSlots.length / (stepsRequested - s));
    const ranked = [...remainingSlots].sort((a, b) => {
      const diff = slotConfidences[b] - slotConfidences[a];
      if (diff !== 0) return diff;
      return a - b;
    });
    const unmaskedSlotIndices = ranked.slice(0, quota).sort((a, b) => a - b);

    for (let i = 0; i < unmaskedSlotIndices.length; i++) {
      const k = unmaskedSlotIndices[i];
      suffixTokens[slotSuffixIndices[k]] = proposals[k];
      finalRawTop1[k] = rawTop1TokenIds[k];
      finalGuardIntervened[k] = guardIntervenedSlotIndices.includes(k) ? 1 : 0;
    }

    const slotTokenIdsAfter = new Int32Array(totalSlots);
    for (let k = 0; k < totalSlots; k++) {
      slotTokenIdsAfter[k] = suffixTokens[slotSuffixIndices[k]];
    }

    stepTraces.push({
      stepIndex: s,
      slotTokenIdsBefore,
      slotTokenIdsAfter,
      unmaskedSlotIndices,
      slotConfidences,
      rawTop1TokenIds,
      guardIntervenedSlotIndices,
    });
  }

  // 4. Assemble decoded boxes & U27 direct-decode provenance
  const clampBin = (rawBin: number): number =>
    rawBin < 0 ? 0 : rawBin > 16 ? 16 : rawBin;

  const boxes: UniGroundingBox[] = [];
  let totalGuardInterventions = 0;

  for (let i = 0; i < options.tags.length; i++) {
    const base = i * 4;
    const id0 = suffixTokens[slotSuffixIndices[base]];
    const id1 = suffixTokens[slotSuffixIndices[base + 1]];
    const id2 = suffixTokens[slotSuffixIndices[base + 2]];
    const id3 = suffixTokens[slotSuffixIndices[base + 3]];

    const locTokenIds = Object.freeze([id0, id1, id2, id3] as const);
    const locTokenStrings = Object.freeze([
      UNI_MINI_VOCAB[id0] ?? '<unk>',
      UNI_MINI_VOCAB[id1] ?? '<unk>',
      UNI_MINI_VOCAB[id2] ?? '<unk>',
      UNI_MINI_VOCAB[id3] ?? '<unk>',
    ] as const);
    const normalizedBins = Object.freeze([
      clampBin(id0 - UNI_MINI_LOC_START_ID),
      clampBin(id1 - UNI_MINI_LOC_START_ID),
      clampBin(id2 - UNI_MINI_LOC_START_ID),
      clampBin(id3 - UNI_MINI_LOC_START_ID),
    ] as const);
    const pixelBox = locBinsToPixelBox(
      normalizedBins,
      options.imageWidth,
      options.imageHeight,
      16,
    );
    const rawTop1Ids = Object.freeze([
      finalRawTop1[base],
      finalRawTop1[base + 1],
      finalRawTop1[base + 2],
      finalRawTop1[base + 3],
    ] as const);
    const guardIntervenedCount =
      finalGuardIntervened[base] +
      finalGuardIntervened[base + 1] +
      finalGuardIntervened[base + 2] +
      finalGuardIntervened[base + 3];

    totalGuardInterventions += guardIntervenedCount;
    boxes.push({
      tag: options.tags[i],
      locTokenIds,
      locTokenStrings,
      normalizedBins,
      pixelBox,
      rawTop1Ids,
      guardIntervenedCount,
    });
  }

  // Reconstruct final filled template to count U27 strip-vulnerable grounding tokens
  const filledTemplateIds = new Int32Array(assembledTemplateIds);
  filledTemplateIds.set(suffixTokens, firstSlotTemplateIdx);
  let skipSpecialTokensWouldStripCount = 0;
  for (let i = 0; i < filledTemplateIds.length; i++) {
    if (isGroundingSpecialToken(filledTemplateIds[i])) {
      skipSpecialTokensWouldStripCount += 1;
    }
  }

  const skipSpecialTokensHazardNote =
    `U27: Direct ID lookup via UNI_MINI_VOCAB preserves all ${skipSpecialTokensWouldStripCount} ` +
    `grounding tokens (<LOC_0..16>, <box_p>, </box_p>, <LOC_BEGIN>, <LOC_END>) that ` +
    `skip_special_tokens=True would silently strip.`;

  return {
    imageWidth: options.imageWidth,
    imageHeight: options.imageHeight,
    stepsRequested,
    assembledTemplateIds,
    slotOffsetsInFullSequence,
    boxes,
    stepTraces,
    prefillCache,
    directTokenDecodePreservesLoc: true,
    skipSpecialTokensWouldStripCount,
    skipSpecialTokensHazardNote,
    totalGuardInterventions,
  };
}
