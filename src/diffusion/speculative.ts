/**
 * Linear Self-Speculation Cycle (`linear_spec` & `linear_spec_lora`)
 * (`src/diffusion/speculative.ts`)
 *
 * Implements one full linear self-speculation cycle matching DF0 §§5.1, 5.4, 5.5
 * and Approved Clarifications #1 (`D19` EOS crop provenance) and #2 (`kvCapacity`
 * tail-block boundary — never clamps `B` to `1`):
 *   1. Strict `kvCapacity` & `'token_causal'` provenance / integrity guards.
 *   2. Bidirectional transient draft loop (`useCache: false`, optional `loraEnabled`)
 *      driven by Sampler B with `D18` non-progress / `maxDraftPassesPerCycle` guards.
 *   3. Causal verification pass (`maskMode: 'causal'`, ALWAYS `loraEnabled: false`,
 *      `useCache: true`) appending `B` KVs to a pre-crop snapshot.
 *   4. Raw upstream equality matching (`D11`: `verifyArTokens[m] === draftedBlockTokens[m+1]`),
 *      clean browser post-EOS KV crop (`browserPostEosCropLen`), and exact upstream
 *      pre-EOS crop recording (`upstreamPreEosCropLen`, `D19`).
 *
 * Pure headless TypeScript — deterministic, self-contained kernel.
 */

import { runDiffusionForwardPass } from './forward';
import {
  appendNewKvsToSnapshot,
  assertKvProvenanceCompatible,
  assertKvSnapshotIntegrity,
  cropKvSnapshot,
  extractCroppedSuffixSlabs,
  materializePastKvs,
} from './kvCache';
import { runSamplerBDraftStep, sampleCausalToken } from './sampler';
import { checkKvCapacityForStep, computeSpecCropProvenance } from './spec';
import type {
  BrowserGuardFlags,
  CroppedKvProvenance,
  DiffusionLifecyclePhase,
  DiffusionModelParameters,
  DraftGuardReason,
  ForwardCounters,
  LinearSpecCycleInput,
  LinearSpecCycleResult,
  LinearSpecDraftPassSummary,
} from './types';

function countToken(arr: Int32Array, tokenId: number, start: number, end: number): number {
  let count = 0;
  for (let i = start; i < end; i++) {
    if (arr[i] === tokenId) count++;
  }
  return count;
}

/**
 * Executes a single `linear_spec` or `linear_spec_lora` draft -> causal verify -> crop cycle.
 */
export function runLinearSpecCycle(
  model: DiffusionModelParameters,
  input: LinearSpecCycleInput,
): LinearSpecCycleResult {
  const {
    kvSnapshot,
    seedToken,
    blockLength,
    loraEnabled,
    samplerBConfig,
    verifyTemperature,
    prng,
    explicitDraftUniforms,
    explicitVerifyUniforms,
    retainedTraceLayers,
    overrideDraftForward,
  } = input;

  if (!Number.isInteger(blockLength) || blockLength <= 0) {
    throw new RangeError(`blockLength must be a positive integer, got ${blockLength}`);
  }

  // Provenance and tamper-detection guards (always verified before operating on snapshot)
  assertKvProvenanceCompatible(kvSnapshot, 'token_causal', 'linear_spec_cycle');
  assertKvSnapshotIntegrity(kvSnapshot);

  // Clarification #2 (§5.5): Never silently clamp B down to 1 when remaining slots < B
  const capCheck = checkKvCapacityForStep(
    loraEnabled ? 'linear_spec_lora' : 'linear_spec',
    kvSnapshot.seqLen,
    kvSnapshot.capacity,
    blockLength,
  );
  if (!capCheck.canProceed) {
    const emptyCrop: CroppedKvProvenance = {
      cacheLenBeforeVerify: kvSnapshot.seqLen,
      acceptedCount: 0,
      firstEosOffset: null,
      browserPostEosCropLen: kvSnapshot.seqLen,
      upstreamPreEosCropLen: kvSnapshot.seqLen,
      preCropKvLen: kvSnapshot.seqLen,
      croppedSuffixLen: 0,
    };
    const emptyCounters: ForwardCounters = {
      actualForwardPasses: 0,
      decodeForwardPasses: 0,
      referenceMethodNfe: 0,
    };
    const exhaustedFlags: BrowserGuardFlags = {
      draftGuardTriggered: 'none',
      residualMasksCommittedCount: 0,
      trailingMasksAfterEosCount: 0,
      capacityExhausted: true,
      warnings: Object.freeze([
        `KV capacity exhausted: remaining slots (${capCheck.remainingSlots}) < blockLength (${blockLength}); halted cleanly without clamping B.`,
      ]),
    };

    return {
      capacityExhausted: true,
      lifecyclePhase: 'capacity_exhausted',
      loraEnabled,
      blockLength,
      seedToken,
      draftedBlockTokens: new Int32Array(0),
      draftPasses: Object.freeze([]),
      draftPassCount: 0,
      verifyArTokens: new Int32Array(0),
      matchedDraftCount: 0,
      acceptedCount: 0,
      emittedTokens: new Int32Array(0),
      nextSeedToken: null,
      eosHit: false,
      firstEosOffset: null,
      cropProvenance: emptyCrop,
      postCropKvSnapshot: kvSnapshot,
      croppedSuffixSlabs: Object.freeze([]),
      verifyForward: null,
      guardFlags: exhaustedFlags,
      cycleCounters: emptyCounters,
    };
  }

  const maskTokenId = samplerBConfig.maskTokenId ?? model.config.maskTokenId;
  const eosTokenId = input.eosTokenId ?? model.config.eosTokenId;
  const vocabSize = model.config.vocabSize;

  // Materialize immutable prefix KV slices (never mutated during draft or verify)
  const pastKvs = materializePastKvs(
    kvSnapshot,
    model.config.numLayers,
    model.config.kvProjOutDim,
  );

  // Initialize active block: [seedToken, MASK, ..., MASK] of length B
  let block: Int32Array = new Int32Array(blockLength);
  block[0] = seedToken;
  for (let i = 1; i < blockLength; i++) {
    block[i] = maskTokenId;
  }

  // --------------------------------------------------------------------------
  // 1. BIDIRECTIONAL TRANSIENT DRAFT LOOP (Sampler B, useCache=false)
  // --------------------------------------------------------------------------
  const maxPasses = Math.max(1, samplerBConfig.maxDraftPassesPerCycle ?? blockLength);
  const draftPasses: LinearSpecDraftPassSummary[] = [];
  let draftGuardTriggered: DraftGuardReason = 'none';
  const warnings: string[] = [];

  for (let passIndex = 0; passIndex < maxPasses; passIndex++) {
    if (countToken(block, maskTokenId, 0, blockLength) === 0) {
      break;
    }

    const blockBefore = new Int32Array(block);
    const draftForward = overrideDraftForward
      ? overrideDraftForward(blockBefore, passIndex)
      : runDiffusionForwardPass(model, {
          tokenIds: blockBefore,
          maskMode: 'bidirectional',
          loraEnabled,
          useCache: false,
          pastKvs,
          retainedTraceLayers,
        });

    const draftRes = runSamplerBDraftStep({
      draftLogits: draftForward.logits,
      blockTokens: blockBefore,
      seqLen: blockLength,
      vocabSize,
      config: {
        ...samplerBConfig,
        maskTokenId,
      },
      prng,
      explicitUniforms: explicitDraftUniforms?.[passIndex],
    });

    draftPasses.push(
      Object.freeze({
        passIndex,
        blockBefore,
        blockAfter: draftRes.updatedBlock,
        confidences: draftRes.confidences,
        unmaskMask: draftRes.unmaskMask,
        remainingMaskCount: draftRes.remainingMaskCount,
        noProgress: draftRes.noProgress,
        variateProvenance: draftRes.variateProvenance,
      }),
    );

    block = draftRes.updatedBlock;

    // D18 guard: greedy Sampler B at threshold > 0 predicting [MASK] causes infinite loop upstream
    if (
      samplerBConfig.threshold > 0 &&
      samplerBConfig.temperature <= 0 &&
      draftRes.noProgress
    ) {
      draftGuardTriggered = 'non_progress_mask_loop';
      warnings.push(
        `D18 guard triggered at draft pass ${passIndex}: greedy Sampler B made zero progress (predicted [MASK]); proceeding to causal verify.`,
      );
      break;
    }

    if (draftRes.remainingMaskCount === 0 || samplerBConfig.threshold <= 0.0) {
      break;
    }
  }

  const remainingMasksAfterDraft = countToken(block, maskTokenId, 0, blockLength);
  if (
    draftGuardTriggered === 'none' &&
    samplerBConfig.threshold > 0 &&
    remainingMasksAfterDraft > 0 &&
    draftPasses.length >= maxPasses
  ) {
    draftGuardTriggered = 'max_draft_passes';
    warnings.push(
      `D18 guard triggered: hit maxDraftPassesPerCycle (${maxPasses}) with ${remainingMasksAfterDraft} residual [MASK] token(s); proceeding to causal verify.`,
    );
  }

  const draftedBlockTokens = new Int32Array(block);

  // --------------------------------------------------------------------------
  // 2. CAUSAL VERIFICATION PASS (ALWAYS loraEnabled: false, useCache: true)
  // --------------------------------------------------------------------------
  const verifyForward = runDiffusionForwardPass(model, {
    tokenIds: draftedBlockTokens,
    maskMode: 'causal',
    loraEnabled: false, // Hard invariant: verifier NEVER activates draft LoRA
    useCache: true,
    pastKvs,
    retainedTraceLayers,
  });

  const preCropKvSnapshot = appendNewKvsToSnapshot(
    kvSnapshot,
    verifyForward.newKvs,
    draftedBlockTokens,
    'token_causal',
  );

  const verifyArTokens = new Int32Array(blockLength);
  for (let i = 0; i < blockLength; i++) {
    const logitsRow = verifyForward.logits.subarray(i * vocabSize, (i + 1) * vocabSize);
    const sampled = sampleCausalToken(
      logitsRow,
      verifyTemperature,
      prng,
      explicitVerifyUniforms?.[i],
    );
    verifyArTokens[i] = sampled.tokenId;
  }

  // --------------------------------------------------------------------------
  // 3. EXACT MATCHING (D11) & EOS / KV CROP PROVENANCE (D19 / CLARIFICATION #1)
  // --------------------------------------------------------------------------
  let matchedDraftCount = 0;
  while (
    matchedDraftCount < blockLength - 1 &&
    verifyArTokens[matchedDraftCount] === draftedBlockTokens[matchedDraftCount + 1]
  ) {
    matchedDraftCount++;
  }
  const acceptedCount = matchedDraftCount + 1;

  // Check whether EOS appears inside the accepted block prefix [0 .. acceptedCount - 1]
  let firstEosOffset: number | null = null;
  for (let j = 0; j < acceptedCount; j++) {
    if (draftedBlockTokens[j] === eosTokenId) {
      firstEosOffset = j;
      break;
    }
  }

  let emittedTokens: Int32Array;
  let nextSeedToken: number | null;
  let eosHit: boolean;
  let lifecyclePhase: DiffusionLifecyclePhase;

  if (firstEosOffset !== null) {
    // EOS hit inside draftedBlockTokens[0..matchedDraftCount].
    // Slot 0 (seedToken) was already emitted prior to this cycle, so emit [1 .. firstEosOffset].
    const emitLen = Math.max(0, firstEosOffset);
    emittedTokens = new Int32Array(emitLen);
    for (let i = 0; i < emitLen; i++) {
      emittedTokens[i] = draftedBlockTokens[i + 1]!;
    }
    nextSeedToken = null;
    eosHit = true;
    lifecyclePhase = 'completed_eos';
  } else {
    // No EOS inside accepted draft prefix; emit [draft_1 .. draft_m, ar_m] (length = acceptedCount)
    const correctionOrBonus = verifyArTokens[matchedDraftCount]!;
    emittedTokens = new Int32Array(acceptedCount);
    for (let i = 0; i < matchedDraftCount; i++) {
      emittedTokens[i] = draftedBlockTokens[i + 1]!;
    }
    emittedTokens[matchedDraftCount] = correctionOrBonus;

    if (correctionOrBonus === eosTokenId) {
      nextSeedToken = null;
      eosHit = true;
      lifecyclePhase = 'completed_eos';
    } else {
      nextSeedToken = correctionOrBonus;
      eosHit = false;
      lifecyclePhase = 'running';
    }
  }

  const baseCrop = computeSpecCropProvenance(
    kvSnapshot.seqLen,
    acceptedCount,
    firstEosOffset,
  );
  const postCropKvSnapshot = cropKvSnapshot(
    preCropKvSnapshot,
    baseCrop.browserPostEosCropLen,
  );
  const croppedSuffixSlabs = extractCroppedSuffixSlabs(
    preCropKvSnapshot,
    baseCrop.browserPostEosCropLen,
  );
  const cropProvenance: CroppedKvProvenance = {
    ...baseCrop,
    preCropKvLen: preCropKvSnapshot.seqLen,
    croppedSuffixLen: croppedSuffixSlabs.length,
  };

  // Count any residual [MASK] tokens committed into the post-crop KV cache from this cycle
  const keptCycleTokens = baseCrop.browserPostEosCropLen - kvSnapshot.seqLen;
  const residualMasksCommittedCount = countToken(
    draftedBlockTokens,
    maskTokenId,
    0,
    keptCycleTokens,
  );
  if (residualMasksCommittedCount > 0) {
    warnings.push(
      `Residual [MASK] committed (${residualMasksCommittedCount} token(s); no logit clamp in NLD).`,
    );
  }

  const trailingMasksAfterEosCount =
    firstEosOffset !== null
      ? countToken(draftedBlockTokens, maskTokenId, firstEosOffset + 1, acceptedCount)
      : 0;

  const cyclePassCount = draftPasses.length + 1; // D_c draft passes + 1 causal verify pass
  const cycleCounters: ForwardCounters = {
    actualForwardPasses: cyclePassCount,
    decodeForwardPasses: cyclePassCount,
    referenceMethodNfe: cyclePassCount,
  };

  const guardFlags: BrowserGuardFlags = {
    draftGuardTriggered,
    residualMasksCommittedCount,
    trailingMasksAfterEosCount,
    capacityExhausted: false,
    warnings: Object.freeze(warnings),
  };

  return {
    capacityExhausted: false,
    lifecyclePhase,
    loraEnabled,
    blockLength,
    seedToken,
    draftedBlockTokens,
    draftPasses: Object.freeze(draftPasses),
    draftPassCount: draftPasses.length,
    verifyArTokens,
    matchedDraftCount,
    acceptedCount,
    emittedTokens,
    nextSeedToken,
    eosHit,
    firstEosOffset,
    cropProvenance,
    postCropKvSnapshot,
    croppedSuffixSlabs,
    verifyForward,
    guardFlags,
    cycleCounters,
  };
}
