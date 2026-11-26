/**
 * Stateless 34-Layer Elastic Mixture-of-Transformers (Elastic MoT) Forward Pass
 * (`src/uni/forward.ts`)
 *
 * Implements the clean-room Nemotron-Labs-Diffusion-Uni 34-layer dual-branch
 * transformer (`L0..17` joint attention + `L18..33` split-tower attention):
 *   - Strictly stateless (`U20`) and non-mutating on `pastKvs` / `pastRouting`
 *   - Supports `inputsEmbeds`, `tokenIds` (`U`), or `genTokenIds` (`G`)
 *   - Implements `D5` `prefillMode = 'hf_faithful' | 'elastic_pruned'`:
 *       * `'hf_faithful'`: executes all 34 layers `L0..33`; records 16 wasted
 *         prefix KV layers when `useCache && canonicalRouting === 'u_only'`
 *       * `'elastic_pruned'`:
 *           - `'u_only'`: executes `L0..17` only, skipping `L18..33`
 *           - `'mixed'`: executes `L0..17` on all tokens, then prunes `U`
 *             compute on `L18..33` (`pruneUnderstandingTokens = true`) so `U`
 *             tokens preserve their `L17` residual while `G` tokens execute
 *             bit-exact identically to `'hf_faithful'`
 *   - Routed final RMSNorm (`encoder.norm` for `U`, `encoder.norm_gen` for `G`),
 *     raw un-suppressed `diffusion_head` (`textLogits`/`textProbs`) and
 *     `gen_predictor` (`genLogits`/`genProbs`).
 *
 * Pure headless TypeScript — zero ambient RNG calls, zero sibling-app imports.
 */

import {
  applyRoutedUniRmsNorm,
  linearProject,
  runUniElasticAttention,
  softmaxRows,
} from './attention';
import { canonicalizeTokenRouting } from './masks';
import { computeUniYarnInvFreq } from './rope';
import { runUniRoutedSwiglu } from './swiglu';
import {
  UNI_ROUTE_G,
  UNI_ROUTE_U,
  type UniAttentionMaskMode,
  type UniCoreModelParameters,
  type UniElasticExecutionTelemetry,
  type UniElasticLayerWeights,
  type UniElasticPrefillMode,
  type UniForwardOptions,
  type UniForwardResult,
  type UniLayerKvSlice,
  type UniLayerTrace,
  type UniMiniConfig,
} from './types';

// ============================================================================
// EMBEDDING LOOKUPS (U TEXT VOCAB & G VISUAL CODEBOOK + SPECIAL ROWS)
// ============================================================================

export function embedUniTextTokens(
  tokenIds: ArrayLike<number>,
  model: UniCoreModelParameters,
): Float32Array {
  const seqLen = tokenIds.length;
  const d = model.config.hiddenSize;
  const vocabSize = model.config.vocabSize;
  const out = new Float32Array(seqLen * d);

  for (let t = 0; t < seqLen; t++) {
    const id = tokenIds[t];
    if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
      throw new RangeError(
        `Text token ID out of range at index ${t}: ${String(id)} (expected 0..${vocabSize - 1})`,
      );
    }
    const srcBase = id * d;
    const dstBase = t * d;
    for (let c = 0; c < d; c++) {
      out[dstBase + c] = model.embedTokens[srcBase + c];
    }
  }
  return out;
}

export function embedUniGenTokens(
  genTokenIds: ArrayLike<number>,
  model: UniCoreModelParameters,
): Float32Array {
  const seqLen = genTokenIds.length;
  const d = model.config.hiddenSize;
  const genRows = model.config.genEmbeddingRows;
  const out = new Float32Array(seqLen * d);

  for (let t = 0; t < seqLen; t++) {
    const id = genTokenIds[t];
    if (!Number.isInteger(id) || id < 0 || id >= genRows) {
      throw new RangeError(
        `Generation token ID out of range at index ${t}: ${String(id)} (expected 0..${genRows - 1})`,
      );
    }
    const srcBase = id * d;
    const dstBase = t * d;
    for (let c = 0; c < d; c++) {
      out[dstBase + c] = model.genEmbedding[srcBase + c];
    }
  }
  return out;
}

// ============================================================================
// SINGLE ELASTIC-MOT TRANSFORMER LAYER (JOINT L0..17 OR SPLIT-TOWER L18..33)
// ============================================================================

export interface RunUniElasticLayerInput {
  readonly hiddenStates: Float32Array;
  readonly seqLen: number;
  readonly routing: Uint8Array;
  readonly layerWeights: UniElasticLayerWeights;
  readonly config: UniMiniConfig;
  readonly invFreq: Float32Array;
  readonly attentionScaling: number;
  readonly positionIds: Int32Array;
  readonly cachePositions?: Int32Array;
  readonly maskMode: UniAttentionMaskMode;
  readonly customMask?: Uint8Array;
  readonly imageBlockStart?: number;
  readonly imageBlockEnd?: number;
  readonly useCache: boolean;
  readonly pastKv?: UniLayerKvSlice | null;
  readonly pastRouting?: Uint8Array | null;
  readonly pruneUnderstandingTokens?: boolean;
}

export function runUniElasticLayer(input: RunUniElasticLayerInput): UniLayerTrace {
  const {
    hiddenStates,
    seqLen,
    routing,
    layerWeights,
    config,
    invFreq,
    attentionScaling,
    positionIds,
    cachePositions,
    maskMode,
    customMask,
    imageBlockStart,
    imageBlockEnd,
    useCache,
    pastKv,
    pastRouting,
  } = input;

  const d = config.hiddenSize;
  if (hiddenStates.length !== seqLen * d) {
    throw new RangeError(
      `hiddenStates length mismatch: got ${hiddenStates.length}, expected ${seqLen * d}`,
    );
  }
  if (routing.length !== seqLen) {
    throw new RangeError(`routing length mismatch: got ${routing.length}, expected ${seqLen}`);
  }

  const splitTower = layerWeights.splitTower;
  const understandingPruned = splitTower && Boolean(input.pruneUnderstandingTokens);

  // 1. Pre-attention routed RMSNorm (U: input_layernorm, G: input_layernorm_gen)
  const inputNorm = applyRoutedUniRmsNorm(
    hiddenStates,
    seqLen,
    d,
    routing,
    layerWeights.u.inputLayerNorm,
    layerWeights.g.inputLayerNorm,
    config.rmsNormEps,
    understandingPruned,
  );

  // 2. Routed Elastic-MoT GQA Attention (8Q/2KV, 1D YaRN RoPE + Llama-4 Q-scale)
  const attention = runUniElasticAttention({
    hiddenStates: inputNorm,
    seqLen,
    routing,
    layerWeights,
    config,
    invFreq,
    attentionScaling,
    positionIds,
    cachePositions,
    maskMode,
    customMask,
    imageBlockStart,
    imageBlockEnd,
    useCache,
    pastKv,
    pastRouting,
    pruneUnderstandingTokens: understandingPruned,
  });

  // 3. First residual connection: h_mid = x + attn_out (U preserved verbatim when pruned)
  const postAttentionResidual = new Float32Array(seqLen * d);
  for (let t = 0; t < seqLen; t++) {
    const base = t * d;
    if (understandingPruned && routing[t] === UNI_ROUTE_U) {
      for (let c = 0; c < d; c++) {
        postAttentionResidual[base + c] = hiddenStates[base + c];
      }
    } else {
      for (let c = 0; c < d; c++) {
        postAttentionResidual[base + c] = Math.fround(
          hiddenStates[base + c] + attention.output[base + c],
        );
      }
    }
  }

  // 4. Pre-MLP routed RMSNorm (U: post_attention_layernorm, G: post_attention_layernorm_gen)
  const postAttentionNorm = applyRoutedUniRmsNorm(
    postAttentionResidual,
    seqLen,
    d,
    routing,
    layerWeights.u.postAttentionLayerNorm,
    layerWeights.g.postAttentionLayerNorm,
    config.rmsNormEps,
    understandingPruned,
  );

  // 5. Routed 3.5x SwiGLU FFN
  const swiglu = runUniRoutedSwiglu(
    postAttentionNorm,
    seqLen,
    routing,
    layerWeights.u.mlp,
    layerWeights.g.mlp,
    config,
    understandingPruned,
  );

  // 6. Second residual connection: out = h_mid + mlp_out (U preserved verbatim when pruned)
  const output = new Float32Array(seqLen * d);
  for (let t = 0; t < seqLen; t++) {
    const base = t * d;
    if (understandingPruned && routing[t] === UNI_ROUTE_U) {
      for (let c = 0; c < d; c++) {
        output[base + c] = postAttentionResidual[base + c];
      }
    } else {
      for (let c = 0; c < d; c++) {
        output[base + c] = Math.fround(postAttentionResidual[base + c] + swiglu.output[base + c]);
      }
    }
  }

  return {
    layerIndex: layerWeights.layerIndex,
    splitTower,
    skipped: false,
    understandingPruned,
    inputNorm,
    attention,
    postAttentionResidual,
    postAttentionNorm,
    swiglu,
    output,
  };
}

// ============================================================================
// FULL 34-LAYER STATELESS ELASTIC-MOT FORWARD PASS
// ============================================================================

function resolveInputsAndRouting(
  model: UniCoreModelParameters,
  options: UniForwardOptions,
): { embeddings: Float32Array; seqLen: number; routing: Uint8Array } {
  const d = model.config.hiddenSize;
  const providedCount =
    (options.inputsEmbeds !== undefined ? 1 : 0) +
    (options.tokenIds !== undefined ? 1 : 0) +
    (options.genTokenIds !== undefined ? 1 : 0);

  if (providedCount !== 1) {
    throw new Error(
      `Provide exactly one of inputsEmbeds, tokenIds, or genTokenIds (received ${providedCount})`,
    );
  }

  let embeddings: Float32Array;
  let seqLen: number;
  let defaultRoute: 0 | 1 | null = null;

  if (options.inputsEmbeds !== undefined) {
    if (options.inputsEmbeds.length === 0 || options.inputsEmbeds.length % d !== 0) {
      throw new RangeError(
        `inputsEmbeds length (${options.inputsEmbeds.length}) must be a positive multiple of hiddenSize (${d})`,
      );
    }
    seqLen = options.inputsEmbeds.length / d;
    embeddings = new Float32Array(options.inputsEmbeds);
    if (options.routing === undefined) {
      throw new Error('routing is required when inputsEmbeds is provided');
    }
  } else if (options.tokenIds !== undefined) {
    seqLen = options.tokenIds.length;
    if (seqLen === 0) {
      throw new RangeError('tokenIds must be non-empty');
    }
    embeddings = embedUniTextTokens(options.tokenIds, model);
    defaultRoute = UNI_ROUTE_U;
  } else {
    const genIds = options.genTokenIds!;
    seqLen = genIds.length;
    if (seqLen === 0) {
      throw new RangeError('genTokenIds must be non-empty');
    }
    embeddings = embedUniGenTokens(genIds, model);
    defaultRoute = UNI_ROUTE_G;
  }

  for (let i = 0; i < embeddings.length; i++) {
    if (!Number.isFinite(embeddings[i])) {
      throw new RangeError(`Non-finite input embedding at index ${i}: ${String(embeddings[i])}`);
    }
  }

  let routing: Uint8Array;
  if (options.routing !== undefined) {
    if (options.routing.length !== seqLen) {
      throw new RangeError(
        `routing length (${options.routing.length}) does not match seqLen (${seqLen})`,
      );
    }
    routing = canonicalizeTokenRouting(options.routing).routing;
  } else {
    routing = new Uint8Array(seqLen);
    if (defaultRoute === UNI_ROUTE_G) {
      routing.fill(UNI_ROUTE_G);
    }
  }

  return { embeddings, seqLen, routing };
}

function resolvePastState(
  model: UniCoreModelParameters,
  options: UniForwardOptions,
): { pastLen: number; pastRouting: Uint8Array | null } {
  const numLayers = model.config.numLayers;
  const pastKvs = options.pastKvs ?? null;
  const rawPastRouting = options.pastRouting ?? null;

  if (pastKvs !== null && pastKvs.length !== numLayers) {
    throw new RangeError(
      `pastKvs length (${pastKvs.length}) must equal numLayers (${numLayers})`,
    );
  }

  let inferredPastLen: number | null = null;
  if (pastKvs !== null) {
    for (let l = 0; l < pastKvs.length; l++) {
      const slice = pastKvs[l];
      if (slice !== null && slice !== undefined) {
        if (inferredPastLen === null) {
          inferredPastLen = slice.seqLen;
        } else if (slice.seqLen !== inferredPastLen) {
          throw new RangeError(
            `Inconsistent pastKvs seqLen at layer ${l}: ${slice.seqLen} !== ${inferredPastLen}`,
          );
        }
      }
    }
  }

  if (rawPastRouting !== null) {
    const copiedRouting = canonicalizeTokenRouting(rawPastRouting).routing;
    if (inferredPastLen !== null && copiedRouting.length !== inferredPastLen) {
      throw new RangeError(
        `pastRouting length (${copiedRouting.length}) does not match pastKvs seqLen (${inferredPastLen})`,
      );
    }
    if (copiedRouting.length > 0 && pastKvs === null) {
      throw new Error('pastKvs must be provided when pastRouting is non-empty');
    }
    return { pastLen: copiedRouting.length, pastRouting: copiedRouting };
  }

  if (inferredPastLen !== null && inferredPastLen > 0) {
    // Default unannotated past KVs to Understanding (U = 0)
    const defaultPastRouting = new Uint8Array(inferredPastLen);
    return { pastLen: inferredPastLen, pastRouting: defaultPastRouting };
  }

  return { pastLen: 0, pastRouting: null };
}

function resolvePositions(
  seqLen: number,
  pastLen: number,
  options: UniForwardOptions,
): { positionIds: Int32Array; cachePositions: Int32Array } {
  const positionIds = new Int32Array(seqLen);
  if (options.positionIds !== undefined) {
    if (options.positionIds.length !== seqLen) {
      throw new RangeError(
        `positionIds length (${options.positionIds.length}) does not match seqLen (${seqLen})`,
      );
    }
    for (let i = 0; i < seqLen; i++) {
      positionIds[i] = options.positionIds[i] | 0;
    }
  } else {
    for (let i = 0; i < seqLen; i++) {
      positionIds[i] = pastLen + i;
    }
  }

  const cachePositions = new Int32Array(seqLen);
  if (options.cachePositions !== undefined) {
    if (options.cachePositions.length !== seqLen) {
      throw new RangeError(
        `cachePositions length (${options.cachePositions.length}) does not match seqLen (${seqLen})`,
      );
    }
    for (let i = 0; i < seqLen; i++) {
      cachePositions[i] = options.cachePositions[i] | 0;
    }
  } else {
    cachePositions.set(positionIds);
  }

  return { positionIds, cachePositions };
}

export function runUniForwardPass(
  model: UniCoreModelParameters,
  options: UniForwardOptions,
): UniForwardResult {
  const cfg = model.config;
  const numLayers = cfg.numLayers;
  const dualTowerLayers = cfg.dualTowerLayers;
  const d = cfg.hiddenSize;

  const { embeddings, seqLen, routing } = resolveInputsAndRouting(model, options);
  const routingPlan = canonicalizeTokenRouting(routing);
  const canonicalRouting = routingPlan.canonical;
  const prefillMode: UniElasticPrefillMode = options.prefillMode ?? 'hf_faithful';

  const { pastLen, pastRouting } = resolvePastState(model, options);
  const kvLen = pastLen + seqLen;
  const { positionIds, cachePositions } = resolvePositions(seqLen, pastLen, options);

  const yarn = computeUniYarnInvFreq(cfg.rope);

  const residualCheckpoints: (Float32Array | null)[] = new Array<Float32Array | null>(
    numLayers + 1,
  ).fill(null);
  residualCheckpoints[0] = new Float32Array(embeddings);

  const newKvs: (UniLayerKvSlice | null)[] = new Array<UniLayerKvSlice | null>(numLayers).fill(
    null,
  );
  const layerTraces = new Map<number, UniLayerTrace>();
  const skippedLayerIndices: number[] = [];
  const droppedCachedUnderstandingLayers: number[] = [];

  const skipSplitLayersEntirely =
    prefillMode === 'elastic_pruned' && canonicalRouting === 'u_only';
  const pruneUnderstandingAtSplit =
    prefillMode === 'elastic_pruned' && canonicalRouting === 'mixed';

  let h = residualCheckpoints[0]!;
  let executedLayerCount = 0;
  let lastLayerExecuted = -1;

  for (let l = 0; l < numLayers; l++) {
    const layerWeights = model.layers[l];
    const isSplit = layerWeights.splitTower;

    if (skipSplitLayersEntirely && isSplit) {
      skippedLayerIndices.push(l);
      residualCheckpoints[l + 1] = null;
      newKvs[l] = null;
      continue;
    }

    const pastKvForLayer = options.pastKvs ? options.pastKvs[l] ?? null : null;
    const trace = runUniElasticLayer({
      hiddenStates: h,
      seqLen,
      routing,
      layerWeights,
      config: cfg,
      invFreq: yarn.invFreq,
      attentionScaling: yarn.attentionScaling,
      positionIds,
      cachePositions,
      maskMode: options.maskMode,
      customMask: options.customMask,
      imageBlockStart: options.imageBlockStart,
      imageBlockEnd: options.imageBlockEnd,
      useCache: options.useCache,
      pastKv: pastKvForLayer,
      pastRouting,
      pruneUnderstandingTokens: pruneUnderstandingAtSplit && isSplit,
    });

    h = trace.output;
    residualCheckpoints[l + 1] = h;
    newKvs[l] = {
      k: trace.attention.kRot,
      v: trace.attention.vNew,
      seqLen,
    };

    if (!options.retainedTraceLayers || options.retainedTraceLayers.has(l)) {
      layerTraces.set(l, trace);
    }
    if (trace.attention.dropCachedUnderstanding) {
      droppedCachedUnderstandingLayers.push(l);
    }

    executedLayerCount += 1;
    lastLayerExecuted = l;
  }

  let finalNorm: Float32Array | null = null;
  let textLogits: Float32Array | null = null;
  let textProbs: Float32Array | null = null;
  let genLogits: Float32Array | null = null;
  let genProbs: Float32Array | null = null;

  if (lastLayerExecuted === numLayers - 1) {
    finalNorm = applyRoutedUniRmsNorm(
      h,
      seqLen,
      d,
      routing,
      model.finalNorm,
      model.finalNormGen,
      cfg.rmsNormEps,
      false,
    );

    const computeTextHead = options.computeTextHead ?? routingPlan.hasUnderstanding;
    const computeGenHead = options.computeGenHead ?? routingPlan.hasGeneration;

    if (computeTextHead) {
      textLogits = linearProject(finalNorm, seqLen, d, model.diffusionHead, cfg.vocabSize);
      textProbs = softmaxRows(textLogits, seqLen, cfg.vocabSize);
    }

    if (computeGenHead) {
      genLogits = linearProject(finalNorm, seqLen, d, model.genPredictor, cfg.vqCodebookSize);
      genProbs = softmaxRows(genLogits, seqLen, cfg.vqCodebookSize);
    }
  }

  const wastedPrefixKvLayers =
    prefillMode === 'hf_faithful' && options.useCache && canonicalRouting === 'u_only'
      ? numLayers - dualTowerLayers
      : 0;

  const understandingTokensPrunedAtSplit =
    skipSplitLayersEntirely || pruneUnderstandingAtSplit
      ? routingPlan.understandingIndices.length
      : 0;

  const telemetry: UniElasticExecutionTelemetry = {
    prefillMode,
    canonicalRouting,
    totalLayers: numLayers,
    executedLayerCount,
    skippedLayerIndices,
    understandingTokensPrunedAtSplit,
    wastedPrefixKvLayers,
    droppedCachedUnderstandingLayers,
  };

  return {
    seqLen,
    pastLen,
    kvLen,
    routing,
    canonicalRouting,
    maskMode: options.maskMode,
    useCache: options.useCache,
    prefillMode,
    positionIds,
    cachePositions,
    embeddings,
    residualCheckpoints,
    layerTraces,
    newKvs,
    finalNorm,
    textLogits,
    textProbs,
    genLogits,
    genProbs,
    telemetry,
  };
}
