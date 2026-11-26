/**
 * Unified Four-Path Headless Executor & 3-Counter Forward Accounting
 * (`src/diffusion/execution.ts`)
 *
 * Implements all four released Nemotron-Labs-Diffusion text inference paths:
 *   1. `ar` — Autoregressive causal decoding (`ar_generate`)
 *   2. `dlm` — Block diffusion (`generate`, both `causalContext = true` & `false`)
 *   3. `linear_spec` — Linear self-speculative decoding without LoRA
 *   4. `linear_spec_lora` — Linear self-speculative decoding with draft-only `o_proj` LoRA
 *
 * Preserves exact raw un-suppressed logits (`[MASK]` is never clamped), 3-counter
 * forward accounting (`actualForwardPasses`, `decodeForwardPasses`, `referenceMethodNfe`),
 * Clarification #1 (`D19` browser post-EOS crop + upstream pre-EOS provenance), and
 * Clarification #2 (`kvCapacity` clean halt without clamping `B` to `1`).
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import { runDiffusionForwardPass } from './forward';
import {
  appendNewKvsToSnapshot,
  assertKvProvenanceCompatible,
  assertKvSnapshotIntegrity,
  createEmptyKvSnapshot,
  materializePastKvs,
} from './kvCache';
import { createXoshiro128ss } from './prng';
import {
  computeStaticTransferQuota,
  runSamplerATransferStep,
  sampleCausalToken,
} from './sampler';
import { checkKvCapacityForStep } from './spec';
import { runLinearSpecCycle } from './speculative';
import type {
  BrowserGuardFlags,
  CroppedKvProvenance,
  DiffusionExecutionOptions,
  DiffusionExecutionResult,
  DiffusionForwardResult,
  DiffusionLifecyclePhase,
  DiffusionModelParameters,
  DlmBlockCycleResult,
  DlmDenoiseStepSummary,
  DraftGuardReason,
  ForwardCounters,
  KvMaskProvenance,
  LinearSpecCycleResult,
  PersistentKvSnapshot,
  Xoshiro128ssGenerator,
} from './types';

export const DIFFUSION_DEFAULT_SAMPLING_SEED = 0x4e4c4453 as const; // "NLDS"

const EMPTY_LAYER_SET: ReadonlySet<number> = Object.freeze(new Set<number>());

// ============================================================================
// 3-COUNTER ACCOUNTING & GUARD HELPERS
// ============================================================================

export function createZeroCounters(): ForwardCounters {
  return Object.freeze({
    actualForwardPasses: 0,
    decodeForwardPasses: 0,
    referenceMethodNfe: 0,
  });
}

export function addCounters(a: ForwardCounters, b: ForwardCounters): ForwardCounters {
  return Object.freeze({
    actualForwardPasses: a.actualForwardPasses + b.actualForwardPasses,
    decodeForwardPasses: a.decodeForwardPasses + b.decodeForwardPasses,
    referenceMethodNfe: a.referenceMethodNfe + b.referenceMethodNfe,
  });
}

export function createEmptyGuardFlags(): BrowserGuardFlags {
  return Object.freeze({
    draftGuardTriggered: 'none' as DraftGuardReason,
    residualMasksCommittedCount: 0,
    trailingMasksAfterEosCount: 0,
    capacityExhausted: false,
    warnings: Object.freeze([] as string[]),
  });
}

export function mergeGuardFlags(
  a: BrowserGuardFlags,
  b: BrowserGuardFlags,
): BrowserGuardFlags {
  const draftGuardTriggered: DraftGuardReason =
    b.draftGuardTriggered !== 'none' ? b.draftGuardTriggered : a.draftGuardTriggered;
  const warnings =
    a.warnings.length === 0 && b.warnings.length === 0
      ? Object.freeze([] as string[])
      : Object.freeze([...a.warnings, ...b.warnings]);
  return Object.freeze({
    draftGuardTriggered,
    residualMasksCommittedCount:
      a.residualMasksCommittedCount + b.residualMasksCommittedCount,
    trailingMasksAfterEosCount:
      a.trailingMasksAfterEosCount + b.trailingMasksAfterEosCount,
    capacityExhausted: a.capacityExhausted || b.capacityExhausted,
    warnings,
  });
}

// ============================================================================
// INTERNAL VALIDATION & ARRAY HELPERS
// ============================================================================

interface ValidatedCommonOptions {
  readonly promptTokenIds: Int32Array;
  readonly maxNewTokens: number;
  readonly kvCapacity: number;
  readonly temperature: number;
  readonly prng: Xoshiro128ssGenerator;
}

function validateCommonOptions(
  model: DiffusionModelParameters,
  options: DiffusionExecutionOptions,
): ValidatedCommonOptions {
  if (!Number.isInteger(options.maxNewTokens) || options.maxNewTokens < 1) {
    throw new RangeError(
      `maxNewTokens must be an integer >= 1 (D22 guard), received ${options.maxNewTokens}`,
    );
  }

  const promptLen = options.promptTokenIds.length;
  if (promptLen < 1) {
    throw new RangeError('promptTokenIds must contain at least 1 token');
  }

  const vocabSize = model.config.vocabSize;
  const promptTokenIds = new Int32Array(promptLen);
  for (let i = 0; i < promptLen; i++) {
    const tok = options.promptTokenIds[i];
    if (!Number.isInteger(tok) || tok < 0 || tok >= vocabSize) {
      throw new RangeError(
        `promptTokenIds[${i}] (${tok}) must be an integer in [0, ${vocabSize - 1}]`,
      );
    }
    promptTokenIds[i] = tok;
  }

  const kvCapacity = options.kvCapacity ?? model.config.kvCapacity;
  if (!Number.isInteger(kvCapacity) || kvCapacity < promptLen) {
    throw new RangeError(
      `kvCapacity (${kvCapacity}) must be an integer >= promptTokenIds.length (${promptLen})`,
    );
  }

  const temperature = options.temperature ?? 0;
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new RangeError(
      `temperature must be finite >= 0, received ${temperature}`,
    );
  }

  const samplingSeed = options.samplingSeed ?? DIFFUSION_DEFAULT_SAMPLING_SEED;
  if (!Number.isInteger(samplingSeed)) {
    throw new RangeError(
      `samplingSeed must be an integer, received ${samplingSeed}`,
    );
  }

  return {
    promptTokenIds,
    maxNewTokens: options.maxNewTokens,
    kvCapacity,
    temperature,
    prng: createXoshiro128ss(samplingSeed),
  };
}

function validateBlockLength(blockLength: number): number {
  if (!Number.isInteger(blockLength) || blockLength < 1) {
    throw new RangeError(
      `blockLength must be a positive integer, received ${blockLength}`,
    );
  }
  return blockLength;
}

export function sliceLastLogitsRow(
  forward: DiffusionForwardResult,
  vocabSize: number,
): Float32Array {
  const rowStart = (forward.seqLen - 1) * vocabSize;
  return forward.logits.subarray(rowStart, rowStart + vocabSize);
}

function concatInt32Arrays(a: Int32Array, b: ArrayLike<number>): Int32Array {
  const out = new Int32Array(a.length + b.length);
  out.set(a, 0);
  for (let i = 0; i < b.length; i++) {
    out[a.length + i] = b[i]!;
  }
  return out;
}

function countTokenInRange(
  tokens: Int32Array,
  target: number,
  start: number,
  endExclusive: number,
): number {
  let count = 0;
  for (let i = start; i < endExclusive; i++) {
    if (tokens[i] === target) count += 1;
  }
  return count;
}

function findFirstTokenIndex(tokens: Int32Array, target: number): number {
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === target) return i;
  }
  return -1;
}

function hasMaskBeforeFirstEos(
  block: Int32Array,
  maskTokenId: number,
  eosTokenId: number,
): boolean {
  const firstEos = findFirstTokenIndex(block, eosTokenId);
  const limit = firstEos >= 0 ? firstEos : block.length;
  for (let i = 0; i < limit; i++) {
    if (block[i] === maskTokenId) return true;
  }
  return false;
}

// ============================================================================
// SINGLE DLM BLOCK CYCLE HELPER (SHARED BY EXECUTION & SESSION)
// ============================================================================

export interface DlmBlockCycleInput {
  readonly kvSnapshot: PersistentKvSnapshot;
  readonly blockIndex: number;
  readonly blockLength: number;
  readonly causalContext: boolean;
  readonly initialSeedToken: number | null;
  readonly temperature: number;
  readonly samplerAThreshold: number | null;
  readonly prng?: Xoshiro128ssGenerator;
  readonly explicitBlockGumbelUniforms?: readonly Float64Array[];
  readonly explicitCommitUniform?: number;
  readonly retainedTraceLayers?: ReadonlySet<number>;
}

/**
 * Execute one complete Block-Diffusion (`dlm`) block cycle:
 *   1. Initialize `B`-slot block (`[seed, MASK..MASK]` if `causalContext`, else `[MASK..MASK]`).
 *   2. Up to `B` bidirectional denoising passes (`useCache: false`, 0-ULP prefix immutability)
 *      driven by Sampler A (`runSamplerATransferStep`), breaking early once no `[MASK]` precedes
 *      the first EOS (after pass 0, preserving exact `D21` prefill-EOS parity).
 *   3. Post-block commit pass (`causal` if `causalContext`, else `bidirectional`, `useCache: true`)
 *      committing all `B` tokens to KV and sampling `nextSeedToken` when `causalContext && !eosHit`.
 */
export function runDlmBlockCycle(
  model: DiffusionModelParameters,
  input: DlmBlockCycleInput,
): DlmBlockCycleResult {
  const {
    kvSnapshot,
    blockIndex,
    blockLength,
    causalContext,
    initialSeedToken,
    temperature,
    samplerAThreshold,
    prng,
    explicitBlockGumbelUniforms,
    explicitCommitUniform,
    retainedTraceLayers,
  } = input;

  validateBlockLength(blockLength);
  const kvMaskProvenance: KvMaskProvenance = causalContext
    ? 'token_causal'
    : 'block_bidirectional';
  assertKvProvenanceCompatible(kvSnapshot, kvMaskProvenance, 'dlm_block_cycle');
  assertKvSnapshotIntegrity(kvSnapshot);

  const maskTokenId = model.config.maskTokenId;
  const eosTokenId = model.config.eosTokenId;
  const vocabSize = model.config.vocabSize;

  let block = new Int32Array(blockLength);
  block.fill(maskTokenId);
  if (causalContext && initialSeedToken !== null) {
    block[0] = initialSeedToken;
  }

  const initialMaskedCount = countTokenInRange(block, maskTokenId, 0, blockLength);
  const quotas =
    samplerAThreshold === null
      ? computeStaticTransferQuota(initialMaskedCount, blockLength)
      : null;

  const pastKvs = materializePastKvs(kvSnapshot);
  const denoiseSteps: DlmDenoiseStepSummary[] = [];
  let denoisePassCount = 0;
  let blockCounters = createZeroCounters();

  for (let step = 0; step < blockLength; step++) {
    const currentMaskedCount = countTokenInRange(block, maskTokenId, 0, blockLength);
    if (currentMaskedCount === 0) {
      break;
    }

    const blockBefore = new Int32Array(block);
    const denoiseForward = runDiffusionForwardPass(model, {
      tokenIds: blockBefore,
      maskMode: 'bidirectional',
      loraEnabled: false,
      useCache: false,
      pastKvs,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    denoisePassCount += 1;
    blockCounters = addCounters(blockCounters, {
      actualForwardPasses: 1,
      decodeForwardPasses: 1,
      referenceMethodNfe: 1,
    });

    const quota = quotas !== null ? quotas[step]! : currentMaskedCount;
    const transferRes = runSamplerATransferStep({
      logits: denoiseForward.logits,
      blockTokens: blockBefore,
      seqLen: blockLength,
      vocabSize,
      numTransferTokens: quota,
      config: {
        temperature,
        threshold: samplerAThreshold,
        maskTokenId,
      },
      prng,
      explicitGumbelUniforms: explicitBlockGumbelUniforms?.[step],
    });

    block = new Int32Array(transferRes.updatedBlock);
    denoiseSteps.push(
      Object.freeze({
        stepIndex: step,
        quota,
        blockBefore,
        blockAfter: new Int32Array(block),
        candidateTokens: transferRes.candidateTokens,
        confidences: transferRes.confidences,
        transferIndex: transferRes.transferIndex,
        remainingMaskCount: transferRes.remainingMaskCount,
        variateProvenance: transferRes.variateProvenance,
      }),
    );

    // Mid-block EOS / full-unmask early break (checked AFTER pass 0 -> exact D21 parity)
    if (!hasMaskBeforeFirstEos(block, maskTokenId, eosTokenId)) {
      break;
    }
  }

  // Post-block commit forward: commits all B tokens to KV (causal or bidirectional)
  const finalBlockTokens = new Int32Array(block);
  const commitForward = runDiffusionForwardPass(model, {
    tokenIds: finalBlockTokens,
    maskMode: causalContext ? 'causal' : 'bidirectional',
    loraEnabled: false,
    useCache: true,
    pastKvs,
    retainedTraceLayers,
  });
  const commitKvSnapshot = appendNewKvsToSnapshot(
    kvSnapshot,
    commitForward.newKvs,
    finalBlockTokens,
    kvMaskProvenance,
  );
  blockCounters = addCounters(blockCounters, {
    actualForwardPasses: 1,
    decodeForwardPasses: 1,
    referenceMethodNfe: 1,
  });

  const firstEosIdx = findFirstTokenIndex(finalBlockTokens, eosTokenId);
  const firstEosOffset = firstEosIdx >= 0 ? firstEosIdx : null;
  const eosHit = firstEosOffset !== null;

  const residualMasksCommittedCount =
    firstEosOffset === null
      ? countTokenInRange(finalBlockTokens, maskTokenId, 0, blockLength)
      : countTokenInRange(finalBlockTokens, maskTokenId, 0, firstEosOffset);
  const trailingMasksAfterEosCount =
    firstEosOffset !== null
      ? countTokenInRange(finalBlockTokens, maskTokenId, firstEosOffset + 1, blockLength)
      : 0;

  const warnings: string[] = [];
  if (residualMasksCommittedCount > 0) {
    warnings.push(
      `Residual [MASK] committed (${residualMasksCommittedCount}) — no logit clamp in NLD`,
    );
  }
  if (trailingMasksAfterEosCount > 0) {
    warnings.push(
      `Trailing [MASK] tokens after intra-block EOS (${trailingMasksAfterEosCount}) committed to KV for PyTorch parity`,
    );
  }

  const emittedTokens =
    firstEosOffset !== null
      ? finalBlockTokens.slice(0, firstEosOffset + 1)
      : new Int32Array(finalBlockTokens);

  let nextSeedToken: number | null = null;
  if (causalContext && !eosHit) {
    const nextSample = sampleCausalToken(
      sliceLastLogitsRow(commitForward, vocabSize),
      temperature,
      prng,
      explicitCommitUniform,
    );
    nextSeedToken = nextSample.tokenId;
  }

  return Object.freeze({
    blockIndex,
    causalContext,
    blockLength,
    initialSeedToken,
    denoiseSteps: Object.freeze(denoiseSteps),
    denoisePassCount,
    finalBlockTokens,
    emittedTokens,
    nextSeedToken,
    eosHit,
    firstEosOffset,
    commitKvSnapshot,
    commitForward,
    guardFlags: Object.freeze({
      draftGuardTriggered: 'none' as DraftGuardReason,
      residualMasksCommittedCount,
      trailingMasksAfterEosCount,
      capacityExhausted: false,
      warnings: Object.freeze(warnings),
    }),
    blockCounters,
  });
}

// ============================================================================
// PATH 1: AUTOREGRESSIVE GENERATION (`ar`)
// ============================================================================

export function runArGeneration(
  model: DiffusionModelParameters,
  options: DiffusionExecutionOptions,
): DiffusionExecutionResult {
  const { promptTokenIds, maxNewTokens, kvCapacity, temperature, prng } =
    validateCommonOptions(model, options);

  const vocabSize = model.config.vocabSize;
  const eosTokenId = model.config.eosTokenId;

  let kv = createEmptyKvSnapshot('token_causal', kvCapacity);
  const prefillForward = runDiffusionForwardPass(model, {
    tokenIds: promptTokenIds,
    maskMode: 'causal',
    loraEnabled: false,
    useCache: true,
    retainedTraceLayers: options.retainedTraceLayers,
  });
  kv = appendNewKvsToSnapshot(kv, prefillForward.newKvs, promptTokenIds, 'token_causal');

  let counters: ForwardCounters = Object.freeze({
    actualForwardPasses: 1,
    decodeForwardPasses: 0,
    referenceMethodNfe: 1,
  });

  const prefillSample = sampleCausalToken(
    sliceLastLogitsRow(prefillForward, vocabSize),
    temperature,
    prng,
    options.explicitVariates?.prefillUniform,
  );

  const generated: number[] = [prefillSample.tokenId];
  const arSteps: DiffusionForwardResult[] = [];
  let lifecyclePhase: DiffusionLifecyclePhase = 'running';
  let guardFlags = createEmptyGuardFlags();

  if (prefillSample.tokenId === eosTokenId) {
    lifecyclePhase = 'completed_eos';
  } else if (generated.length >= maxNewTokens) {
    lifecyclePhase = 'completed_max_tokens';
  }

  while (lifecyclePhase === 'running' && generated.length < maxNewTokens) {
    const capCheck = checkKvCapacityForStep('ar', kv.seqLen, kvCapacity, 1);
    if (!capCheck.canProceed) {
      lifecyclePhase = 'capacity_exhausted';
      guardFlags = mergeGuardFlags(guardFlags, {
        draftGuardTriggered: 'none',
        residualMasksCommittedCount: 0,
        trailingMasksAfterEosCount: 0,
        capacityExhausted: true,
        warnings: Object.freeze([
          `KV capacity (${kvCapacity}) exhausted at seqLen=${kv.seqLen} before AR decode step`,
        ]),
      });
      break;
    }

    const prevToken = generated[generated.length - 1]!;
    const stepIndex = arSteps.length;
    const pastKvs = materializePastKvs(kv);
    const stepForward = runDiffusionForwardPass(model, {
      tokenIds: [prevToken],
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
      pastKvs,
      retainedTraceLayers: options.retainedTraceLayers,
    });
    kv = appendNewKvsToSnapshot(kv, stepForward.newKvs, [prevToken], 'token_causal');
    arSteps.push(stepForward);

    counters = addCounters(counters, {
      actualForwardPasses: 1,
      decodeForwardPasses: 1,
      referenceMethodNfe: 1,
    });

    const stepSample = sampleCausalToken(
      sliceLastLogitsRow(stepForward, vocabSize),
      temperature,
      prng,
      options.explicitVariates?.arStepUniforms?.[stepIndex],
    );
    generated.push(stepSample.tokenId);

    if (stepSample.tokenId === eosTokenId) {
      lifecyclePhase = 'completed_eos';
      break;
    }
    if (generated.length >= maxNewTokens) {
      lifecyclePhase = 'completed_max_tokens';
      break;
    }
  }

  const generatedTokenIds = Int32Array.from(generated);
  return Object.freeze({
    mode: 'ar',
    causalContext: true,
    loraEnabledOnDraft: false,
    promptTokenIds,
    generatedTokenIds,
    fullTokenIds: concatInt32Arrays(promptTokenIds, generatedTokenIds),
    lifecyclePhase,
    kvMaskProvenance: 'token_causal',
    finalKvSnapshot: kv,
    counters,
    guardFlags,
    arSteps: Object.freeze(arSteps),
    dlmBlocks: Object.freeze([] as DlmBlockCycleResult[]),
    specCycles: Object.freeze([] as LinearSpecCycleResult[]),
    prefillForward,
    lastCropProvenance: null,
  });
}

// ============================================================================
// PATH 2: BLOCK DIFFUSION GENERATION (`dlm`, causalContext = true | false)
// ============================================================================

export function runDlmGeneration(
  model: DiffusionModelParameters,
  options: DiffusionExecutionOptions,
): DiffusionExecutionResult {
  const { promptTokenIds, maxNewTokens, kvCapacity, temperature, prng } =
    validateCommonOptions(model, options);

  const causalContext = options.causalContext ?? true;
  const blockLength = validateBlockLength(
    options.blockLength ?? model.config.defaultBlockLength,
  );
  const samplerAThreshold =
    options.samplerAThreshold === undefined ? null : options.samplerAThreshold;
  if (samplerAThreshold !== null && !Number.isFinite(samplerAThreshold)) {
    throw new RangeError(
      `samplerAThreshold must be null or finite, received ${samplerAThreshold}`,
    );
  }

  const vocabSize = model.config.vocabSize;
  const kvMaskProvenance: KvMaskProvenance = causalContext
    ? 'token_causal'
    : 'block_bidirectional';

  let kv = createEmptyKvSnapshot(kvMaskProvenance, kvCapacity);
  const prefillForward = runDiffusionForwardPass(model, {
    tokenIds: promptTokenIds,
    maskMode: causalContext ? 'causal' : 'bidirectional',
    loraEnabled: false,
    useCache: true,
    retainedTraceLayers: options.retainedTraceLayers,
  });
  kv = appendNewKvsToSnapshot(kv, prefillForward.newKvs, promptTokenIds, kvMaskProvenance);

  // D12: Upstream generate() excludes prefill from referenceMethodNfe!
  let counters: ForwardCounters = Object.freeze({
    actualForwardPasses: 1,
    decodeForwardPasses: 0,
    referenceMethodNfe: 0,
  });

  let pendingSeed: number | null = null;
  if (causalContext) {
    pendingSeed = sampleCausalToken(
      sliceLastLogitsRow(prefillForward, vocabSize),
      temperature,
      prng,
      options.explicitVariates?.prefillUniform,
    ).tokenId;
  }

  const generated: number[] = [];
  const dlmBlocks: DlmBlockCycleResult[] = [];
  let lifecyclePhase: DiffusionLifecyclePhase = 'running';
  let guardFlags = createEmptyGuardFlags();

  while (lifecyclePhase === 'running' && generated.length < maxNewTokens) {
    // Clarification #2: NEVER clamp B down to 1!
    const capCheck = checkKvCapacityForStep('dlm', kv.seqLen, kvCapacity, blockLength);
    if (!capCheck.canProceed) {
      lifecyclePhase = 'capacity_exhausted';
      guardFlags = mergeGuardFlags(guardFlags, {
        draftGuardTriggered: 'none',
        residualMasksCommittedCount: 0,
        trailingMasksAfterEosCount: 0,
        capacityExhausted: true,
        warnings: Object.freeze([
          `KV capacity (${kvCapacity}) exhausted at seqLen=${kv.seqLen} for DLM blockLength=${blockLength} (B is never clamped)`,
        ]),
      });
      break;
    }

    const blockIndex = dlmBlocks.length;
    const blockResult = runDlmBlockCycle(model, {
      kvSnapshot: kv,
      blockIndex,
      blockLength,
      causalContext,
      initialSeedToken: pendingSeed,
      temperature,
      samplerAThreshold,
      prng,
      explicitBlockGumbelUniforms:
        options.explicitVariates?.dlmBlockGumbelUniforms?.[blockIndex],
      explicitCommitUniform: options.explicitVariates?.dlmCommitUniforms?.[blockIndex],
      retainedTraceLayers: options.retainedTraceLayers,
    });

    dlmBlocks.push(blockResult);
    kv = blockResult.commitKvSnapshot;
    counters = addCounters(counters, blockResult.blockCounters);
    guardFlags = mergeGuardFlags(guardFlags, blockResult.guardFlags);

    for (let i = 0; i < blockResult.emittedTokens.length; i++) {
      generated.push(blockResult.emittedTokens[i]!);
    }

    pendingSeed = blockResult.nextSeedToken;

    if (blockResult.eosHit) {
      lifecyclePhase = 'completed_eos';
      break;
    }
    if (generated.length >= maxNewTokens) {
      lifecyclePhase = 'completed_max_tokens';
      break;
    }
  }

  const generatedTokenIds = Int32Array.from(generated);
  return Object.freeze({
    mode: 'dlm',
    causalContext,
    loraEnabledOnDraft: false,
    promptTokenIds,
    generatedTokenIds,
    fullTokenIds: concatInt32Arrays(promptTokenIds, generatedTokenIds),
    lifecyclePhase,
    kvMaskProvenance,
    finalKvSnapshot: kv,
    counters,
    guardFlags,
    arSteps: Object.freeze([] as DiffusionForwardResult[]),
    dlmBlocks: Object.freeze(dlmBlocks),
    specCycles: Object.freeze([] as LinearSpecCycleResult[]),
    prefillForward,
    lastCropProvenance: null,
  });
}

// ============================================================================
// PATHS 3 & 4: LINEAR SELF-SPECULATION (`linear_spec` | `linear_spec_lora`)
// ============================================================================

export function runLinearSpecGeneration(
  model: DiffusionModelParameters,
  options: DiffusionExecutionOptions,
): DiffusionExecutionResult {
  const { promptTokenIds, maxNewTokens, kvCapacity, temperature, prng } =
    validateCommonOptions(model, options);

  const mode =
    options.mode === 'linear_spec_lora' ? 'linear_spec_lora' : 'linear_spec';
  const loraEnabled = mode === 'linear_spec_lora';
  const blockLength = validateBlockLength(
    options.blockLength ?? model.config.defaultBlockLength,
  );
  const samplerBThreshold = options.samplerBThreshold ?? 0.0;
  if (!Number.isFinite(samplerBThreshold) || samplerBThreshold < 0) {
    throw new RangeError(
      `samplerBThreshold must be finite >= 0, received ${samplerBThreshold}`,
    );
  }
  const maxDraftPassesPerCycle = options.maxDraftPassesPerCycle ?? blockLength;
  if (!Number.isInteger(maxDraftPassesPerCycle) || maxDraftPassesPerCycle < 1) {
    throw new RangeError(
      `maxDraftPassesPerCycle must be an integer >= 1, received ${maxDraftPassesPerCycle}`,
    );
  }

  const vocabSize = model.config.vocabSize;
  const eosTokenId = model.config.eosTokenId;

  let kv = createEmptyKvSnapshot('token_causal', kvCapacity);
  // Prefill is ALWAYS causal and ALWAYS LoRA-OFF
  const prefillForward = runDiffusionForwardPass(model, {
    tokenIds: promptTokenIds,
    maskMode: 'causal',
    loraEnabled: false,
    useCache: true,
    retainedTraceLayers: options.retainedTraceLayers,
  });
  kv = appendNewKvsToSnapshot(kv, prefillForward.newKvs, promptTokenIds, 'token_causal');

  let counters: ForwardCounters = Object.freeze({
    actualForwardPasses: 1,
    decodeForwardPasses: 0,
    referenceMethodNfe: 1,
  });

  const prefillSample = sampleCausalToken(
    sliceLastLogitsRow(prefillForward, vocabSize),
    temperature,
    prng,
    options.explicitVariates?.prefillUniform,
  );

  const generated: number[] = [prefillSample.tokenId];
  let pendingSeed: number | null = prefillSample.tokenId;
  const specCycles: LinearSpecCycleResult[] = [];
  let lastCropProvenance: CroppedKvProvenance | null = null;
  let lifecyclePhase: DiffusionLifecyclePhase = 'running';
  let guardFlags = createEmptyGuardFlags();

  if (prefillSample.tokenId === eosTokenId) {
    lifecyclePhase = 'completed_eos';
    pendingSeed = null;
  } else if (generated.length >= maxNewTokens) {
    lifecyclePhase = 'completed_max_tokens';
  }

  while (
    lifecyclePhase === 'running' &&
    generated.length < maxNewTokens &&
    pendingSeed !== null
  ) {
    const cycleIdx = specCycles.length;
    const cycle = runLinearSpecCycle(model, {
      kvSnapshot: kv,
      seedToken: pendingSeed,
      blockLength,
      loraEnabled,
      samplerBConfig: {
        temperature,
        threshold: samplerBThreshold,
        maxDraftPassesPerCycle,
        maskTokenId: model.config.maskTokenId,
      },
      verifyTemperature: temperature,
      eosTokenId,
      prng,
      explicitDraftUniforms: options.explicitVariates?.specCycleDraftUniforms?.[cycleIdx],
      explicitVerifyUniforms: options.explicitVariates?.specCycleVerifyUniforms?.[cycleIdx],
      retainedTraceLayers: options.retainedTraceLayers,
    });

    guardFlags = mergeGuardFlags(guardFlags, cycle.guardFlags);

    if (cycle.capacityExhausted) {
      lifecyclePhase = 'capacity_exhausted';
      break;
    }

    specCycles.push(cycle);
    counters = addCounters(counters, cycle.cycleCounters);
    kv = cycle.postCropKvSnapshot;
    lastCropProvenance = cycle.cropProvenance;

    for (let i = 0; i < cycle.emittedTokens.length; i++) {
      generated.push(cycle.emittedTokens[i]!);
    }

    pendingSeed = cycle.nextSeedToken;

    if (cycle.eosHit || cycle.lifecyclePhase === 'completed_eos') {
      lifecyclePhase = 'completed_eos';
      break;
    }
    if (generated.length >= maxNewTokens) {
      lifecyclePhase = 'completed_max_tokens';
      break;
    }
  }

  const generatedTokenIds = Int32Array.from(generated);
  return Object.freeze({
    mode,
    causalContext: true,
    loraEnabledOnDraft: loraEnabled,
    promptTokenIds,
    generatedTokenIds,
    fullTokenIds: concatInt32Arrays(promptTokenIds, generatedTokenIds),
    lifecyclePhase,
    kvMaskProvenance: 'token_causal',
    finalKvSnapshot: kv,
    counters,
    guardFlags,
    arSteps: Object.freeze([] as DiffusionForwardResult[]),
    dlmBlocks: Object.freeze([] as DlmBlockCycleResult[]),
    specCycles: Object.freeze(specCycles),
    prefillForward,
    lastCropProvenance,
  });
}

// ============================================================================
// UNIFIED 4-PATH DISPATCHER
// ============================================================================

export function runDiffusionExecution(
  model: DiffusionModelParameters,
  options: DiffusionExecutionOptions,
): DiffusionExecutionResult {
  switch (options.mode) {
    case 'ar':
      return runArGeneration(model, options);
    case 'dlm':
      return runDlmGeneration(model, options);
    case 'linear_spec':
    case 'linear_spec_lora':
      return runLinearSpecGeneration(model, options);
    default: {
      const exhaustive: never = options.mode;
      throw new RangeError(`Unsupported diffusion execution mode: ${String(exhaustive)}`);
    }
  }
}
