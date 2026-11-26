/**
 * Progressive Self-Refining T2I (`B_cfg = 2` / `U30` `B_cfg = 1`) &
 * Dual-CFG Image Editing (`B_cfg = 3`) Sampler (`src/uni/imageSampler.ts`)
 *
 * Clean-room headless TypeScript implementation for Nemotron-Labs-Diffusion-Uni
 * (`UNI0_ARCHITECTURE_AUDIT.md` §5.1, §6 `U9–U18`, `U22–U23`, `U30`, `D2–D5`):
 *   - Unified `UniFullRuntimeBundle` factory combining 766 tensors (`618` core +
 *     `31` RADIO-H vision/projector + `113` Emu3VQ + `4` PackedLinearImageCodec)
 *     plus the deterministic `256`-code `depth = 8` `BalancedSemanticTree`.
 *   - 2-branch T2I CFG (`w_txt = 3.0`) with `U30` `guidanceScale === 0` `B_cfg = 1`
 *     short-circuit (documenting upstream HF's wasted `B = 2` drift).
 *   - 3-branch Dual-CFG Image Editing (`[uncond, vq_dropped, full]`) supporting
 *     all three guidance modes (`legacy` default `edit_mode = 1` per `U16`,
 *     `ip2p`, and `independent`), continuous RADIO-H + discrete packed source-VQ
 *     mixed-routing Stage-1 prefill, `U18` `lockSourcePaddingVq` immutability
 *     across every round, and `cropToSource` letterbox inversion.
 *   - `D5` live `hf_faithful` vs. `elastic_pruned` Stage-1 prefill support with
 *     guaranteed `0`-ULP Stage-2 trajectory equivalence across both T2I and Editing.
 *   - Log-domain Gumbel proposal sampling (`U22`), deterministic `topK`/`topP`
 *     filtering, fixed draw-count consumption (`16 * 256` uniforms + `16` randBits
 *     per round) via `createXoshiro128ss` from `./prng` or explicit variates (`U23`).
 *   - ZERO AdaLN `timesteps` conditioning (`U17`), ZERO `Math.random()`, ZERO DOM,
 *     ZERO imports from `src/twotower/`, `src/diffusion/`, or `src/ui/controls.ts`.
 */

import { softmaxRows } from './attention';
import {
  compressPackedImageEmbeddings,
  computeSourcePaddingVqMask,
  createUniPackedCodecWeights,
  cropPaddedImageToSource,
  embedFixedPrefixTokens,
  expandPackedImageHiddenStates,
  stripGenerationRowMarkers,
} from './codec';
import {
  embedUniGenTokens,
  embedUniTextTokens,
  runUniForwardPass,
} from './forward';
import {
  appendBatchKvsToCache,
  createEmptyUniKvCache,
  extractRowPastKvs,
} from './kvCache';
import { createDefaultUniCoreModel, createUniCoreModel } from './parameters';
import { createXoshiro128ss } from './prng';
import {
  advanceRevisableSemanticTree,
  buildBalancedSemanticTree,
  buildSemanticTreeEmbeddings,
  contentIndexToMarkedGridIndex,
  initSemanticTreeState,
  semanticTreeActivationCounts,
  stratifiedRandom,
} from './tree';
import {
  UNI_MINI_BOS_TOKEN_ID,
  UNI_MINI_CONFIG,
  UNI_MINI_GEN_FIXED_PREFIX_COUNT,
  UNI_MINI_IM_START_TOKEN_ID,
  UNI_MINI_PAD_TOKEN_ID,
  UNI_MINI_UND_PAD_TOKEN_ID,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_MULTIMODAL_SEED,
  UNI_ROUTE_G,
  UNI_STRATIFIED_DEFAULT_SEED,
  type UniBalancedSemanticTree,
  type UniCfgBatchSize,
  type UniCompactCandidateTop4,
  type UniCoreModelParameters,
  type UniEditGuidanceMode,
  type UniElasticPrefillMode,
  type UniEmu3VqWeights,
  type UniGainPresetId,
  type UniImageDenoiseStepRecord,
  type UniImageEditSamplerOptions,
  type UniImageTrajectoryResult,
  type UniKvCacheState,
  type UniLayerKvSlice,
  type UniLetterboxInfo,
  type UniLetterboxResult,
  type UniPackedCodecWeights,
  type UniRgbImage,
  type UniSemanticTreeSchedule,
  type UniSemanticTreeState,
  type UniSourcePaddingVqMask,
  type UniT2iSamplerOptions,
  type UniVisionSubsystemWeights,
} from './types';
import {
  createUniVisionSubsystemWeights,
  fixOnePrompt,
  padToSquareAndResize,
  runMiniVisionPipeline,
  spliceVisualEmbeddingsIntoPrompt,
} from './vision';
import {
  createUniEmu3VqWeights,
  decodeMiniEmu3Vq,
  encodeMiniEmu3Vq,
} from './vqvae';

// ============================================================================
// CONSTANTS & INTERNAL HELPERS
// ============================================================================

const OPEN_UNIT_MIN = 1e-15;
const OPEN_UNIT_MAX = 1 - 1e-15;
const DEFAULT_T2I_SEED = 0x4e4c5533; // "NLU3"
const DEFAULT_EDIT_SEED = 0x4e4c5534; // "NLU4"
const EMPTY_LAYER_SET: ReadonlySet<number> = new Set<number>();

const HF_GUIDANCE_ZERO_DRIFT_NOTE =
  'U30 browser guard: T2I guidanceScale === 0 short-circuited to B_cfg = 1 (cond-only). ' +
  'Upstream HF modeling_nlduni.py passes default image_guidance_scale = 2.0 into _sample, ' +
  'keeping run_cfg_branches = True and wasting an unused B_cfg = 2 uncond pass.';

function clampOpenUnit(u: number): number {
  if (!Number.isFinite(u)) {
    throw new RangeError(`Expected finite open-unit uniform variate, got ${String(u)}`);
  }
  if (u < OPEN_UNIT_MIN) return OPEN_UNIT_MIN;
  if (u > OPEN_UNIT_MAX) return OPEN_UNIT_MAX;
  return u;
}

function buildEqualLengthUncondTokens(
  prefixLen: number,
  explicitUncond?: ArrayLike<number>,
): Int32Array {
  if (!Number.isInteger(prefixLen) || prefixLen < 1) {
    throw new RangeError(`prefixLen must be an integer >= 1, got ${String(prefixLen)}`);
  }

  const out = new Int32Array(prefixLen);
  if (explicitUncond !== undefined) {
    if (explicitUncond.length < 1 || explicitUncond.length > prefixLen) {
      throw new RangeError(
        `uncondPromptTokenIds length (${explicitUncond.length}) must be in [1, ${prefixLen}]`,
      );
    }
    for (let i = 0; i < explicitUncond.length; i++) {
      out[i] = explicitUncond[i] | 0;
    }
    for (let i = explicitUncond.length; i < prefixLen; i++) {
      out[i] = UNI_MINI_PAD_TOKEN_ID;
    }
    return out;
  }

  out[0] = UNI_MINI_BOS_TOKEN_ID;
  if (prefixLen >= 2) {
    out[1] = UNI_MINI_IM_START_TOKEN_ID;
  }
  for (let i = 2; i < prefixLen; i++) {
    out[i] = UNI_MINI_PAD_TOKEN_ID;
  }
  return out;
}

function argmaxRow(values: ArrayLike<number>, length: number): number {
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let v = 0; v < length; v++) {
    const val = values[v];
    if (val > bestVal) {
      bestVal = val;
      bestIdx = v;
    }
  }
  return bestIdx;
}

// ============================================================================
// 1 & 2. FULL RUNTIME BUNDLE (766 TENSORS + BALANCED SEMANTIC TREE)
// ============================================================================

export interface UniFullRuntimeBundle {
  readonly core: UniCoreModelParameters;
  readonly vision: UniVisionSubsystemWeights;
  readonly vqvae: UniEmu3VqWeights;
  readonly codec: UniPackedCodecWeights;
  readonly tree: UniBalancedSemanticTree;
}

/**
 * Construct the complete clean-room NLD-Uni miniature runtime bundle:
 *   - 618 core Elastic-MoT tensors (`createDefaultUniCoreModel` / `createUniCoreModel`)
 *   - 31 Mini RADIO-H + Projector + `image_newline` tensors
 *   - 113 Mini Emu3VQ tensors
 *   - 4 PackedLinearImageCodec tensors
 *   - 256-code `depth = 8` deterministic `BalancedSemanticTree` (`3,068 B`)
 */
export function createDefaultUniRuntimeBundle(options?: {
  backboneSeed?: number;
  multimodalSeed?: number;
}): UniFullRuntimeBundle {
  const core =
    options?.backboneSeed !== undefined
      ? createUniCoreModel({ backboneSeed: options.backboneSeed })
      : createDefaultUniCoreModel();
  const mmSeed = options?.multimodalSeed ?? UNI_MULTIMODAL_SEED;
  const vision = createUniVisionSubsystemWeights(mmSeed);
  const vqvae = createUniEmu3VqWeights(mmSeed);
  const codec = createUniPackedCodecWeights(mmSeed);
  const tree = buildBalancedSemanticTree(
    vqvae.codebook,
    UNI_MINI_VQ_CODEBOOK_SIZE,
    UNI_MINI_CONFIG.vqLatentDim,
  );

  return Object.freeze({
    core,
    vision,
    vqvae,
    codec,
    tree,
  });
}

// ============================================================================
// 3 & 4. T2I CFG & EDITING DUAL-CFG LOGIT MIXERS (`U16`, FP32 EXACT)
// ============================================================================

/**
 * 2-branch T2I Classifier-Free Guidance (`B_cfg = 2`):
 *   `guided[i] = Math.fround((1 + w) * cond[i] - w * uncond[i])`
 */
export function applyT2iCfgLogits(
  uncondLogits: Float32Array,
  condLogits: Float32Array,
  guidanceScale: number,
): Float32Array {
  const len = condLogits.length;
  if (uncondLogits.length !== len) {
    throw new RangeError(
      `uncondLogits (${uncondLogits.length}) and condLogits (${len}) must have equal length`,
    );
  }
  if (!Number.isFinite(guidanceScale)) {
    throw new RangeError(`guidanceScale must be finite, got ${String(guidanceScale)}`);
  }

  const w = Math.fround(guidanceScale);
  const onePlusW = Math.fround(1.0 + w);
  const out = new Float32Array(len);

  for (let i = 0; i < len; i++) {
    out[i] = Math.fround(
      Math.fround(onePlusW * condLogits[i]) - Math.fround(w * uncondLogits[i]),
    );
  }
  return out;
}

/**
 * 3-branch Editing Dual-CFG logit mixer (`[uncond, vq_dropped, full]`, `U16`):
 *   - `'legacy'` (default `edit_mode = 1`):
 *       `cond = (full + w_img * vq_dropped) / (1 + w_img)`
 *       `guided = (1 + w_txt) * cond - w_txt * uncond`
 *   - `'ip2p'`:
 *       `guided = uncond + w_img * (vq_dropped - uncond) + w_txt * (full - vq_dropped)`
 *   - `'independent'`:
 *       `guided = uncond + w_txt * (full - uncond) + w_img * (full - vq_dropped)`
 *
 * Evaluated element-wise in FP32 via `Math.fround` matching PyTorch FP32.
 */
export function applyEditDualCfgLogits(
  uncondLogits: Float32Array,
  vqDroppedLogits: Float32Array,
  fullLogits: Float32Array,
  guidanceMode: UniEditGuidanceMode,
  guidanceScaleText: number,
  guidanceScaleImage: number,
): Float32Array {
  const len = fullLogits.length;
  if (uncondLogits.length !== len || vqDroppedLogits.length !== len) {
    throw new RangeError(
      `uncondLogits (${uncondLogits.length}), vqDroppedLogits (${vqDroppedLogits.length}), and fullLogits (${len}) must have equal length`,
    );
  }
  if (!Number.isFinite(guidanceScaleText) || !Number.isFinite(guidanceScaleImage)) {
    throw new RangeError(
      `guidanceScaleText (${String(guidanceScaleText)}) and guidanceScaleImage (${String(guidanceScaleImage)}) must be finite`,
    );
  }

  const wTxt = Math.fround(guidanceScaleText);
  const wImg = Math.fround(guidanceScaleImage);
  const out = new Float32Array(len);

  if (guidanceMode === 'legacy') {
    const denom = Math.fround(1.0 + wImg);
    if (denom === 0) {
      throw new RangeError('legacy Dual-CFG requires 1 + guidanceScaleImage !== 0');
    }
    const onePlusWTxt = Math.fround(1.0 + wTxt);
    for (let i = 0; i < len; i++) {
      const num = Math.fround(fullLogits[i] + Math.fround(wImg * vqDroppedLogits[i]));
      const cond = Math.fround(num / denom);
      out[i] = Math.fround(
        Math.fround(onePlusWTxt * cond) - Math.fround(wTxt * uncondLogits[i]),
      );
    }
    return out;
  }

  if (guidanceMode === 'ip2p') {
    for (let i = 0; i < len; i++) {
      const u = uncondLogits[i];
      const vq = vqDroppedLogits[i];
      const f = fullLogits[i];
      const dImg = Math.fround(vq - u);
      const dTxt = Math.fround(f - vq);
      out[i] = Math.fround(
        Math.fround(u + Math.fround(wImg * dImg)) + Math.fround(wTxt * dTxt),
      );
    }
    return out;
  }

  if (guidanceMode === 'independent') {
    for (let i = 0; i < len; i++) {
      const u = uncondLogits[i];
      const vq = vqDroppedLogits[i];
      const f = fullLogits[i];
      const dTxt = Math.fround(f - u);
      const dImg = Math.fround(f - vq);
      out[i] = Math.fround(
        Math.fround(u + Math.fround(wTxt * dTxt)) + Math.fround(wImg * dImg),
      );
    }
    return out;
  }

  throw new RangeError(`Unsupported guidanceMode: ${String(guidanceMode)}`);
}

// ============================================================================
// 5, 6 & 7. TOP-K / TOP-P FILTER, LOG-DOMAIN GUMBEL SAMPLING & COMPACT TOP-4
// ============================================================================

/**
 * Deterministic top-k and nucleus (top-p) logit filter returning a fresh `Float32Array`
 * with filtered-out entries masked to `-Infinity`. Always preserves at least the top-1
 * finite logit. Tie-breaks descending sort on smaller token index.
 */
export function applyTopKTopPFilter(
  logitsRow: Float32Array,
  topK: number,
  topP: number,
): Float32Array {
  const vocabSize = logitsRow.length;
  if (vocabSize < 1) {
    throw new RangeError('logitsRow must be non-empty');
  }
  if (!Number.isFinite(topK) || topK < 0) {
    throw new RangeError(`topK must be finite and >= 0, got ${String(topK)}`);
  }
  if (!Number.isFinite(topP) || topP < 0) {
    throw new RangeError(`topP must be finite and >= 0, got ${String(topP)}`);
  }

  const out = new Float32Array(logitsRow);
  const useTopK = topK > 0 && topK < vocabSize;
  const useTopP = topP > 0 && topP < 1.0;
  if (!useTopK && !useTopP) {
    return out;
  }

  const sortedIndices = new Array<number>(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    sortedIndices[v] = v;
  }
  sortedIndices.sort((a, b) => {
    const la = out[a];
    const lb = out[b];
    if (la === lb) return a - b;
    return lb - la;
  });

  let activeCount = vocabSize;
  if (useTopK) {
    const k = Math.max(1, Math.floor(topK));
    for (let rank = k; rank < vocabSize; rank++) {
      out[sortedIndices[rank]] = -Infinity;
    }
    activeCount = k;
  }

  if (useTopP && activeCount > 1) {
    const maxLogit = out[sortedIndices[0]];
    if (Number.isFinite(maxLogit)) {
      const exps = new Float64Array(activeCount);
      let sumExp = 0.0;
      for (let rank = 0; rank < activeCount; rank++) {
        const val = out[sortedIndices[rank]];
        if (Number.isFinite(val)) {
          const e = Math.exp(val - maxLogit);
          exps[rank] = e;
          sumExp += e;
        }
      }
      if (sumExp > 0.0 && Number.isFinite(sumExp)) {
        let cumProb = 0.0;
        for (let rank = 0; rank < activeCount; rank++) {
          const prob = exps[rank] / sumExp;
          // Standard shifted nucleus mask: keep rank 0 and the first token crossing topP;
          // mask subsequent tokens once cumulative probability before this rank >= topP.
          if (rank > 0 && cumProb >= topP) {
            out[sortedIndices[rank]] = -Infinity;
          }
          cumProb += prob;
        }
      }
    }
  }

  return out;
}

/**
 * Sample VQ proposals across a `[gridLen, vocabSize]` guided-logit table (`U22`, `U23`):
 *   - Marker cells (`isMarker?.[i] === 1`): skip RNG draw and write `0`.
 *   - Content cells: apply `applyTopKTopPFilter`, then greedy `argmax` if `temperature <= 0`,
 *     otherwise overflow-safe log-domain Gumbel
 *     `(filtered[v] - maxVal) / temperature - Math.log(-Math.log(u))`
 *     always consuming `vocabSize` draws from `nextUniformOpenUnit()` per content cell
 *     (skipping `-Infinity` entries after drawing) so draw counts stay strictly fixed
 *     (`16 * 256` per round).
 */
export function sampleVqProposals(
  guidedLogits: Float32Array,
  gridLen: number,
  vocabSize: number,
  temperature: number,
  topK: number,
  topP: number,
  nextUniformOpenUnit: () => number,
  isMarker?: Uint8Array,
): Int32Array {
  if (!Number.isInteger(gridLen) || gridLen < 1) {
    throw new RangeError(`gridLen must be an integer >= 1, got ${String(gridLen)}`);
  }
  if (!Number.isInteger(vocabSize) || vocabSize < 1) {
    throw new RangeError(`vocabSize must be an integer >= 1, got ${String(vocabSize)}`);
  }
  if (guidedLogits.length < gridLen * vocabSize) {
    throw new RangeError(
      `guidedLogits length (${guidedLogits.length}) is smaller than gridLen * vocabSize (${gridLen * vocabSize})`,
    );
  }
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new RangeError(`temperature must be finite and >= 0, got ${String(temperature)}`);
  }
  if (isMarker !== undefined && isMarker.length !== gridLen) {
    throw new RangeError(
      `isMarker length (${isMarker.length}) must equal gridLen (${gridLen})`,
    );
  }

  const proposals = new Int32Array(gridLen);

  for (let i = 0; i < gridLen; i++) {
    if (isMarker !== undefined && isMarker[i] === 1) {
      proposals[i] = 0;
      continue;
    }

    const row = guidedLogits.subarray(i * vocabSize, (i + 1) * vocabSize);
    const filtered = applyTopKTopPFilter(row, topK, topP);

    if (temperature <= 0) {
      proposals[i] = argmaxRow(filtered, vocabSize);
      continue;
    }

    let maxVal = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      const val = filtered[v];
      if (Number.isFinite(val) && val > maxVal) {
        maxVal = val;
      }
    }
    if (!Number.isFinite(maxVal)) {
      throw new RangeError(`No finite logit available at cell ${i} during Gumbel sampling`);
    }

    let bestIdx = -1;
    let bestScore = -Infinity;
    for (let v = 0; v < vocabSize; v++) {
      const u = clampOpenUnit(nextUniformOpenUnit());
      const lv = filtered[v];
      if (!Number.isFinite(lv)) {
        continue;
      }
      const score = (lv - maxVal) / temperature - Math.log(-Math.log(u));
      if (score > bestScore) {
        bestScore = score;
        bestIdx = v;
      }
    }

    if (bestIdx < 0) {
      throw new RangeError(`Failed to sample finite proposal at cell ${i}`);
    }
    proposals[i] = bestIdx;
  }

  return proposals;
}

/**
 * Extract compact `[gridLen * 4]` top-4 VQ token IDs and softmax probabilities
 * from guided logits (`[gridLen * vocabSize]`). Deterministic tie-break on lower ID.
 */
export function extractCompactTop4Candidates(
  guidedLogits: Float32Array,
  gridLen: number,
  vocabSize: number,
): UniCompactCandidateTop4 {
  if (!Number.isInteger(gridLen) || gridLen < 1) {
    throw new RangeError(`gridLen must be an integer >= 1, got ${String(gridLen)}`);
  }
  if (!Number.isInteger(vocabSize) || vocabSize < 1) {
    throw new RangeError(`vocabSize must be an integer >= 1, got ${String(vocabSize)}`);
  }
  if (guidedLogits.length < gridLen * vocabSize) {
    throw new RangeError(
      `guidedLogits length (${guidedLogits.length}) is smaller than gridLen * vocabSize (${gridLen * vocabSize})`,
    );
  }

  const probsAll = softmaxRows(guidedLogits, gridLen, vocabSize);
  const kCount = Math.min(4, vocabSize);
  const tokenIds = new Int32Array(gridLen * 4);
  const probs = new Float32Array(gridLen * 4);

  const bestIds = new Int32Array(4);
  const bestProbs = new Float32Array(4);

  for (let i = 0; i < gridLen; i++) {
    const rowBase = i * vocabSize;
    bestIds.fill(0);
    bestProbs.fill(-1.0);

    for (let v = 0; v < vocabSize; v++) {
      const p = probsAll[rowBase + v];
      for (let slot = 0; slot < kCount; slot++) {
        if (p > bestProbs[slot]) {
          for (let shift = kCount - 1; shift > slot; shift--) {
            bestProbs[shift] = bestProbs[shift - 1];
            bestIds[shift] = bestIds[shift - 1];
          }
          bestProbs[slot] = p;
          bestIds[slot] = v;
          break;
        }
      }
    }

    const dstBase = i * 4;
    for (let slot = 0; slot < kCount; slot++) {
      tokenIds[dstBase + slot] = bestIds[slot];
      probs[dstBase + slot] = bestProbs[slot];
    }
  }

  return Object.freeze({
    tokenIds,
    probs,
  });
}

// ============================================================================
// 8, 9 & 10. STAGE-1 PREFIX PREFILL (T2I & DUAL-CFG IMAGE EDITING)
// ============================================================================

export interface UniPreparedGenerationPrefix {
  readonly cfgBatchSize: UniCfgBatchSize;
  readonly cfgShortCircuitApplied: boolean;
  readonly hfGuidanceZeroDriftNote?: string;
  readonly prefixLen: number;
  readonly prefixCache: UniKvCacheState;
  readonly perBranchEmbeds: readonly Float32Array[];
  readonly perBranchRouting: readonly Uint8Array[];
  readonly sourceLetterbox?: UniLetterboxResult;
  readonly sourcePaddingMask?: UniSourcePaddingVqMask;
  readonly sourceVqCodes?: Int32Array;
}

/**
 * Run Stage-1 causal prompt prefill for Text-to-Image generation:
 *   - `guidanceScale === 0`: `U30` short-circuits to `cfgBatchSize = 1` (`[cond]` only)
 *     and populates `hfGuidanceZeroDriftNote`.
 *   - `guidanceScale !== 0`: `cfgBatchSize = 2` (`[uncond, cond]`) equal-length prefix.
 *   - Supports both `D5` prefill modes (`'hf_faithful'` vs `'elastic_pruned'`).
 */
export function prepareT2iPrefix(
  bundle: UniFullRuntimeBundle,
  options: UniT2iSamplerOptions,
): UniPreparedGenerationPrefix {
  const condTokens = Int32Array.from(options.promptTokenIds);
  const prefixLen = condTokens.length;
  if (prefixLen < 1) {
    throw new RangeError('promptTokenIds must be non-empty');
  }

  const guidanceScale = options.guidanceScale ?? 3.0;
  if (!Number.isFinite(guidanceScale)) {
    throw new RangeError(`guidanceScale must be finite, got ${String(guidanceScale)}`);
  }
  const prefillMode: UniElasticPrefillMode = options.prefillMode ?? 'hf_faithful';

  const cfgShortCircuitApplied = guidanceScale === 0;
  const cfgBatchSize: UniCfgBatchSize = cfgShortCircuitApplied ? 1 : 2;

  const perBranchTokens: Int32Array[] = cfgShortCircuitApplied
    ? [condTokens]
    : [buildEqualLengthUncondTokens(prefixLen, options.uncondPromptTokenIds), condTokens];

  const perBranchEmbeds: Float32Array[] = [];
  const perBranchRouting: Uint8Array[] = [];
  const perRowNewKvs: (readonly (UniLayerKvSlice | null)[])[] = [];

  for (let b = 0; b < cfgBatchSize; b++) {
    const emb = embedUniTextTokens(perBranchTokens[b], bundle.core);
    const routing = new Uint8Array(prefixLen); // all UNI_ROUTE_U (0)
    const fwd = runUniForwardPass(bundle.core, {
      inputsEmbeds: emb,
      routing,
      maskMode: 'causal',
      useCache: true,
      prefillMode,
      downstreamTarget: 'g',
      computeTextHead: false,
      computeGenHead: false,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    perBranchEmbeds.push(emb);
    perBranchRouting.push(routing);
    perRowNewKvs.push(fwd.newKvs);
  }

  const emptyCache = createEmptyUniKvCache({
    batchSize: cfgBatchSize,
    prefillMode,
    downstreamTarget: 'g',
  });
  const prefixCache = appendBatchKvsToCache(emptyCache, perRowNewKvs, perBranchRouting);

  const base: UniPreparedGenerationPrefix = {
    cfgBatchSize,
    cfgShortCircuitApplied,
    prefixLen,
    prefixCache,
    perBranchEmbeds: Object.freeze(perBranchEmbeds),
    perBranchRouting: Object.freeze(perBranchRouting),
    ...(cfgShortCircuitApplied ? { hfGuidanceZeroDriftNote: HF_GUIDANCE_ZERO_DRIFT_NOTE } : {}),
  };

  return Object.freeze(base);
}

/**
 * Run Stage-1 causal prompt prefill for 3-branch Dual-CFG Image Editing (`B_cfg = 3`):
 *   - Processes `sourceImage` through BOTH continuous RADIO-H (`U`) and discrete
 *     letterboxed Emu3VQ + `PackedLinearImageCodec(isSourceEdit: true)` (`4` `G` slots).
 *   - Assembles equal-length `prefixLen = [visualSlots + nl] + [4 VQ slots] + [instruction]`:
 *       * Row 0 (`uncond`): all `U` (`BOS, im_start, pad...`) -> `elastic_pruned` skips `L18..33`
 *       * Row 1 (`vq_dropped`): RADIO-H (`U`) + `4` `U` pad tokens (dropping discrete VQ while
 *         retaining RADIO-H per `U16`) + instruction (`U`) -> `elastic_pruned` skips `L18..33`
 *       * Row 2 (`full`): RADIO-H (`U`) + `4` packed source-VQ (`G`) + instruction (`U`)
 *         -> `canonicalRouting = 'mixed'`, `elastic_pruned` prunes `U` and preserves `G` at `L18..33`
 *   - Appends all 3 rows into `UniKvCacheState({ batchSize: 3, prefillMode, downstreamTarget: 'g' })`.
 */
export function prepareImageEditPrefix(
  bundle: UniFullRuntimeBundle,
  options: UniImageEditSamplerOptions,
): UniPreparedGenerationPrefix {
  const instrTokens = Int32Array.from(options.promptTokenIds);
  const instrLen = instrTokens.length;
  if (instrLen < 1) {
    throw new RangeError('promptTokenIds must be non-empty');
  }

  const d = UNI_MINI_CONFIG.hiddenSize; // 8
  const gainPresetId: UniGainPresetId = options.gainPresetId ?? UNI_MINI_CONFIG.defaultGainPreset;
  const prefillMode: UniElasticPrefillMode = options.prefillMode ?? 'hf_faithful';

  // 1. Continuous RADIO-H vision stream (U) + fixOnePrompt row-newline expansion
  const visionTrace = runMiniVisionPipeline(options.sourceImage, bundle.vision, gainPresetId);
  const visRows = visionTrace.projector.tokenGridRows;
  const visCols = visionTrace.projector.tokenGridCols;
  const fixedVisPrompt = fixOnePrompt([UNI_MINI_UND_PAD_TOKEN_ID], visRows, visCols);
  const visBlockLen = fixedVisPrompt.expandedTokenIds.length;
  const visBaseEmbeds = embedUniTextTokens(fixedVisPrompt.expandedTokenIds, bundle.core);
  const visBlockEmbeds = spliceVisualEmbeddingsIntoPrompt(
    fixedVisPrompt.expandedTokenIds,
    visBaseEmbeds,
    fixedVisPrompt.visualSlotIndices,
    visionTrace.projector.visualEmbeddings,
    d,
  );

  // 2. Discrete letterboxed Emu3VQ + PackedLinearImageCodec(isSourceEdit=true) (4 G slots)
  const sourceLetterbox = padToSquareAndResize(
    options.sourceImage,
    UNI_MINI_CONFIG.imageResolution,
  );
  const sourcePaddingMask = computeSourcePaddingVqMask(
    sourceLetterbox.info,
    UNI_MINI_CONFIG.vitPatchSize,
  );
  const vqEnc = encodeMiniEmu3Vq(sourceLetterbox.image, bundle.vqvae);
  const sourceVqCodes = new Int32Array(vqEnc.codes); // [16]
  const srcChildEmb = embedUniGenTokens(sourceVqCodes, bundle.core); // [16, 8]
  const srcPacked = compressPackedImageEmbeddings(
    srcChildEmb,
    UNI_MINI_CONFIG.vqGridSize,
    UNI_MINI_CONFIG.vqGridSize,
    bundle.codec,
    { isSourceEdit: true, gainPresetId },
  ); // [4, 8]
  const vqSlotCount = srcPacked.parentRows * srcPacked.parentCols; // 4

  // 3. Instruction embeddings (U) & 4 U pad embeddings for vq_dropped branch
  const instrEmbeds = embedUniTextTokens(instrTokens, bundle.core);
  const pad4Tokens = new Int32Array(vqSlotCount).fill(UNI_MINI_PAD_TOKEN_ID);
  const pad4Embeds = embedUniTextTokens(pad4Tokens, bundle.core);

  const prefixLen = visBlockLen + vqSlotCount + instrLen;

  // Row 0: uncond (all U)
  const uncondTokens = buildEqualLengthUncondTokens(prefixLen, options.uncondPromptTokenIds);
  const emb0 = embedUniTextTokens(uncondTokens, bundle.core);
  const route0 = new Uint8Array(prefixLen); // all U (0)

  // Row 1: vq_dropped (RADIO-H U + 4 U pad + instruction U) -> all U
  const emb1 = new Float32Array(prefixLen * d);
  emb1.set(visBlockEmbeds, 0);
  emb1.set(pad4Embeds, visBlockLen * d);
  emb1.set(instrEmbeds, (visBlockLen + vqSlotCount) * d);
  const route1 = new Uint8Array(prefixLen); // all U (0)

  // Row 2: full (RADIO-H U + 4 packed source-VQ G + instruction U) -> mixed
  const emb2 = new Float32Array(prefixLen * d);
  emb2.set(visBlockEmbeds, 0);
  emb2.set(srcPacked.output, visBlockLen * d);
  emb2.set(instrEmbeds, (visBlockLen + vqSlotCount) * d);
  const route2 = new Uint8Array(prefixLen);
  for (let i = 0; i < vqSlotCount; i++) {
    route2[visBlockLen + i] = UNI_ROUTE_G;
  }

  const perBranchEmbeds: Float32Array[] = [emb0, emb1, emb2];
  const perBranchRouting: Uint8Array[] = [route0, route1, route2];
  const perRowNewKvs: (readonly (UniLayerKvSlice | null)[])[] = [];

  for (let b = 0; b < 3; b++) {
    const fwd = runUniForwardPass(bundle.core, {
      inputsEmbeds: perBranchEmbeds[b],
      routing: perBranchRouting[b],
      maskMode: 'causal',
      useCache: true,
      prefillMode,
      downstreamTarget: 'g',
      computeTextHead: false,
      computeGenHead: false,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });
    perRowNewKvs.push(fwd.newKvs);
  }

  const emptyCache = createEmptyUniKvCache({
    batchSize: 3,
    prefillMode,
    downstreamTarget: 'g',
  });
  const prefixCache = appendBatchKvsToCache(emptyCache, perRowNewKvs, perBranchRouting);

  return Object.freeze({
    cfgBatchSize: 3 as const,
    cfgShortCircuitApplied: false,
    prefixLen,
    prefixCache,
    perBranchEmbeds: Object.freeze(perBranchEmbeds),
    perBranchRouting: Object.freeze(perBranchRouting),
    sourceLetterbox,
    sourcePaddingMask,
    sourceVqCodes,
  });
}

// ============================================================================
// 11. SINGLE PROGRESSIVE SELF-REFINING DENOISE ROUND
// ============================================================================

export function runSingleImageDenoiseRound(
  bundle: UniFullRuntimeBundle,
  params: {
    prefixCache: UniKvCacheState;
    treeState: UniSemanticTreeState;
    schedule: UniSemanticTreeSchedule;
    unmaskOrder: Int32Array;
    roundIndex: number;
    taskKind: 'text_to_image' | 'image_edit';
    guidanceMode?: UniEditGuidanceMode;
    guidanceScaleText: number;
    guidanceScaleImage: number;
    temperature: number;
    topK: number;
    topP: number;
    gainPresetId: UniGainPresetId;
    sourcePaddingMask?: UniSourcePaddingVqMask;
    sourceVqCodes?: Int32Array;
    nextUniformOpenUnit: () => number;
    nextRandBits: () => number;
    explicitRoundUniforms?: ArrayLike<number>;
    explicitRoundRandBits?: ArrayLike<number>;
    retainFullLogits?: boolean;
  },
): { stepRecord: UniImageDenoiseStepRecord; nextTreeState: UniSemanticTreeState } {
  const {
    prefixCache,
    treeState,
    schedule,
    unmaskOrder,
    roundIndex,
    guidanceMode,
    guidanceScaleText,
    guidanceScaleImage,
    temperature,
    topK,
    topP,
    gainPresetId,
    sourcePaddingMask,
    sourceVqCodes,
    nextUniformOpenUnit,
    nextRandBits,
    explicitRoundUniforms,
    explicitRoundRandBits,
    retainFullLogits = false,
  } = params;

  if (!Number.isInteger(roundIndex) || roundIndex < 0 || roundIndex >= schedule.totalRounds) {
    throw new RangeError(
      `roundIndex (${String(roundIndex)}) out of range [0, ${schedule.totalRounds - 1}]`,
    );
  }

  const rows = UNI_MINI_CONFIG.vqGridSize; // 4
  const contentCols = UNI_MINI_CONFIG.vqGridSize; // 4
  const markedCols = UNI_MINI_CONFIG.vqMarkedGridCols; // 6
  const contentCount = rows * contentCols; // 16
  const gridLen = rows * markedCols; // 24
  const parentRows = UNI_MINI_CONFIG.packedGridRows; // 2
  const parentCols = UNI_MINI_CONFIG.packedGridCols; // 3
  const fixedPrefixCount = UNI_MINI_GEN_FIXED_PREFIX_COUNT; // 4
  const windowLen = fixedPrefixCount + parentRows * parentCols; // 10
  const d = UNI_MINI_CONFIG.hiddenSize; // 8
  const vocabSize = UNI_MINI_VQ_CODEBOOK_SIZE; // 256

  if (treeState.stages.length !== gridLen) {
    throw new RangeError(
      `treeState grid length (${treeState.stages.length}) must equal ${gridLen}`,
    );
  }
  if (unmaskOrder.length !== contentCount) {
    throw new RangeError(
      `unmaskOrder length (${unmaskOrder.length}) must equal ${contentCount}`,
    );
  }

  const stagesBefore = new Int32Array(treeState.stages);
  const representativesBefore = new Int32Array(treeState.representatives);

  // 1. Build [24, 8] progressive self-refining embeddings (U10) & pack to [6, 8]
  const childEmb = buildSemanticTreeEmbeddings(
    treeState.representatives,
    treeState.noiseWeights,
    bundle.core.genEmbedding,
    d,
  );
  const packedTarget = compressPackedImageEmbeddings(
    childEmb,
    rows,
    markedCols,
    bundle.codec,
    { isSourceEdit: false, gainPresetId },
  );

  // 2. Prepend 4 uncompressed fixed-prefix embeddings -> [10, 8] G-only window
  const fixedPrefixEmb = embedFixedPrefixTokens(bundle.core.genEmbedding, d);
  const windowEmb = new Float32Array(windowLen * d);
  windowEmb.set(fixedPrefixEmb, 0);
  windowEmb.set(packedTarget.output, fixedPrefixCount * d);

  const allG = new Uint8Array(windowLen);
  allG.fill(UNI_ROUTE_G);

  const pastLen = prefixCache.seqLen;
  const parentPositionIds = new Int32Array(windowLen);
  for (let i = 0; i < windowLen; i++) {
    parentPositionIds[i] = pastLen + i;
  }

  // 3. Evaluate Stage-2 bidirectional forward across each CFG branch (ZERO AdaLN timesteps: U17)
  const cfgBatchSize = prefixCache.batchSize;
  const perBranchResiduals: (readonly (Float32Array | null)[])[] = [];
  const perBranchFinalNorm: Float32Array[] = [];
  const branchLogits: Float32Array[] = [];

  for (let b = 0; b < cfgBatchSize; b++) {
    const { pastKvs, pastRouting } = extractRowPastKvs(prefixCache, b);
    const fwd = runUniForwardPass(bundle.core, {
      inputsEmbeds: windowEmb,
      routing: allG,
      maskMode: 'bidirectional',
      useCache: false,
      pastKvs,
      pastRouting,
      positionIds: parentPositionIds,
      cachePositions: parentPositionIds,
      computeTextHead: false,
      computeGenHead: false,
      retainedTraceLayers: EMPTY_LAYER_SET,
    });

    const finalNorm = fwd.finalNorm;
    if (finalNorm === null) {
      throw new Error(`Unexpected null finalNorm on Stage-2 G-only forward (branch ${b})`);
    }

    perBranchResiduals.push(fwd.residualCheckpoints);
    perBranchFinalNorm.push(finalNorm);

    const parentHidden = finalNorm.subarray(fixedPrefixCount * d, windowLen * d);
    const expanded = expandPackedImageHiddenStates(
      parentHidden,
      parentRows,
      parentCols,
      bundle.codec,
      bundle.core.genPredictor,
    );
    if (expanded.logits === null) {
      throw new Error(`Unexpected null genPredictor logits (branch ${b})`);
    }
    branchLogits.push(expanded.logits);
  }

  // 4. CFG / Dual-CFG logit mixing
  let guidedLogits: Float32Array;
  if (cfgBatchSize === 1) {
    guidedLogits = branchLogits[0];
  } else if (cfgBatchSize === 2) {
    guidedLogits = applyT2iCfgLogits(branchLogits[0], branchLogits[1], guidanceScaleText);
  } else {
    guidedLogits = applyEditDualCfgLogits(
      branchLogits[0],
      branchLogits[1],
      branchLogits[2],
      guidanceMode ?? 'legacy',
      guidanceScaleText,
      guidanceScaleImage,
    );
  }

  // 5. Determine activated content cells for this round
  const isFlushRound = roundIndex >= schedule.activationSteps;
  const prevCum = roundIndex === 0 ? 0 : schedule.cumulativeActivations[roundIndex - 1];
  const currCum = schedule.cumulativeActivations[roundIndex];
  const actCount = currCum - prevCum;
  const activatedContentIndices = new Int32Array(actCount);
  for (let i = 0; i < actCount; i++) {
    activatedContentIndices[i] = unmaskOrder[prevCum + i];
  }

  // 6. Sample proposals via topK/topP + log-domain Gumbel (or greedy when T <= 0)
  let uniformCursor = 0;
  const uniformProvider =
    explicitRoundUniforms !== undefined
      ? (): number => {
          if (uniformCursor >= explicitRoundUniforms.length) {
            throw new RangeError(
              `explicitRoundUniforms exhausted at index ${uniformCursor} (length=${explicitRoundUniforms.length})`,
            );
          }
          return clampOpenUnit(explicitRoundUniforms[uniformCursor++]);
        }
      : nextUniformOpenUnit;

  const proposals = sampleVqProposals(
    guidedLogits,
    gridLen,
    vocabSize,
    temperature,
    topK,
    topP,
    uniformProvider,
    treeState.isMarker,
  );

  // 7. Draw 16 unsigned bit words for bitwise subtree representative sampling
  let roundRandBits: ArrayLike<number>;
  if (explicitRoundRandBits !== undefined) {
    if (
      explicitRoundRandBits.length !== contentCount &&
      explicitRoundRandBits.length !== gridLen
    ) {
      throw new RangeError(
        `explicitRoundRandBits length (${explicitRoundRandBits.length}) must equal ${contentCount} or ${gridLen}`,
      );
    }
    roundRandBits = explicitRoundRandBits;
  } else {
    const bits = new Uint32Array(contentCount);
    for (let c = 0; c < contentCount; c++) {
      bits[c] = nextRandBits() >>> 0;
    }
    roundRandBits = bits;
  }

  // 8. Advance semantic tree state (immutably) & enforce U18 source-padding lock
  let nextTreeState = advanceRevisableSemanticTree(
    bundle.tree,
    treeState,
    proposals,
    activatedContentIndices,
    roundRandBits,
    sourcePaddingMask?.lockMask,
  );

  if (sourcePaddingMask !== undefined && sourceVqCodes !== undefined) {
    let needsClamp = false;
    for (let c = 0; c < contentCount; c++) {
      if (sourcePaddingMask.lockMask[c] === 1) {
        const g = contentIndexToMarkedGridIndex(c, contentCols, markedCols);
        if (
          nextTreeState.representatives[g] !== sourceVqCodes[c] ||
          nextTreeState.stages[g] !== nextTreeState.refinementSteps ||
          nextTreeState.levels[g] !== bundle.tree.depth ||
          nextTreeState.noiseWeights[g] !== 0.0
        ) {
          needsClamp = true;
          break;
        }
      }
    }
    if (needsClamp) {
      const stages = new Int32Array(nextTreeState.stages);
      const reps = new Int32Array(nextTreeState.representatives);
      const levels = new Int32Array(nextTreeState.levels);
      const noiseWeights = new Float32Array(nextTreeState.noiseWeights);
      for (let c = 0; c < contentCount; c++) {
        if (sourcePaddingMask.lockMask[c] === 1) {
          const g = contentIndexToMarkedGridIndex(c, contentCols, markedCols);
          stages[g] = nextTreeState.refinementSteps;
          reps[g] = sourceVqCodes[c];
          levels[g] = bundle.tree.depth;
          noiseWeights[g] = 0.0;
        }
      }
      nextTreeState = Object.freeze({
        stages,
        representatives: reps,
        levels,
        noiseWeights,
        isMarker: nextTreeState.isMarker,
        refinementSteps: nextTreeState.refinementSteps,
      });
    }
  }

  const lockedCellCount = sourcePaddingMask !== undefined ? sourcePaddingMask.lockedCellCount : 0;
  let committedContentCount = 0;
  for (let g = 0; g < gridLen; g++) {
    if (
      nextTreeState.isMarker[g] === 0 &&
      nextTreeState.stages[g] >= nextTreeState.refinementSteps
    ) {
      committedContentCount++;
    }
  }

  const top4Guided = extractCompactTop4Candidates(guidedLogits, gridLen, vocabSize);

  const stepRecord: UniImageDenoiseStepRecord = Object.freeze({
    roundIndex,
    isFlushRound,
    activatedContentIndices,
    stagesBefore,
    stagesAfter: new Int32Array(nextTreeState.stages),
    representativesBefore,
    representativesAfter: new Int32Array(nextTreeState.representatives),
    proposals,
    levelsAfter: new Int32Array(nextTreeState.levels),
    noiseWeightsAfter: new Float32Array(nextTreeState.noiseWeights),
    lockedCellCount,
    committedContentCount,
    top4Guided,
    perBranchResiduals: Object.freeze(perBranchResiduals),
    perBranchFinalNorm: Object.freeze(perBranchFinalNorm),
    ...(retainFullLogits
      ? {
          perBranchFullLogits: Object.freeze(branchLogits),
          guidedLogits,
        }
      : {}),
    parentPositionIds,
  });

  return { stepRecord, nextTreeState };
}

// ============================================================================
// 12. MARKED CODE DECODING & OPTIONAL SOURCE CROP (`U18`)
// ============================================================================

/**
 * Strip `+2` structural row/end markers (`24 -> 16`), decode through Mini Emu3VQ,
 * and optionally center-crop back to source dimensions using `letterbox`.
 */
export function decodeMarkedCodesToImage(
  bundle: UniFullRuntimeBundle,
  markedCodes: ArrayLike<number>,
  letterbox?: UniLetterboxInfo,
  cropToSource?: boolean,
): {
  contentCodes: Int32Array;
  decodedPaddedImage: UniRgbImage;
  outputImage: UniRgbImage;
} {
  const contentCodes = stripGenerationRowMarkers(
    markedCodes,
    UNI_MINI_CONFIG.vqGridSize,
    UNI_MINI_CONFIG.vqMarkedGridCols,
  );
  const decodedTrace = decodeMiniEmu3Vq(
    contentCodes,
    UNI_MINI_CONFIG.vqGridSize,
    UNI_MINI_CONFIG.vqGridSize,
    bundle.vqvae,
  );
  const decodedPaddedImage = decodedTrace.rgbImage;
  const outputImage =
    cropToSource && letterbox !== undefined
      ? cropPaddedImageToSource(decodedPaddedImage, letterbox)
      : decodedPaddedImage;

  return Object.freeze({
    contentCodes,
    decodedPaddedImage,
    outputImage,
  });
}

// ============================================================================
// 13 & 14. END-TO-END T2I & DUAL-CFG IMAGE EDITING TRAJECTORIES
// ============================================================================

/**
 * Execute a complete end-to-end Text-to-Image progressive self-refining discrete
 * diffusion trajectory (`1` prefill + `steps + N - 1` denoise rounds -> Emu3VQ RGB).
 */
export function runT2iTrajectory(
  bundle: UniFullRuntimeBundle,
  options: UniT2iSamplerOptions,
): UniImageTrajectoryResult {
  const activationSteps = options.activationSteps ?? UNI_MINI_CONFIG.defaultActivationSteps;
  const refinementSteps = options.refinementSteps ?? UNI_MINI_CONFIG.defaultRefinementSteps;
  const shift = options.shift ?? UNI_MINI_CONFIG.defaultShift;
  const guidanceScaleText = options.guidanceScale ?? 3.0;
  const guidanceScaleImage = 0;
  const temperature = options.temperature ?? 0.8;
  const topK = options.topK ?? 0;
  const topP = options.topP ?? 1.0;
  const gainPresetId: UniGainPresetId = options.gainPresetId ?? UNI_MINI_CONFIG.defaultGainPreset;
  const prefillMode: UniElasticPrefillMode = options.prefillMode ?? 'hf_faithful';
  const seed = (options.seed ?? DEFAULT_T2I_SEED) >>> 0;
  const stratifiedSeed = options.stratifiedSeed ?? UNI_STRATIFIED_DEFAULT_SEED;
  const retainFullLogits = options.retainFullLogitsOnSteps ?? false;

  const prepared = prepareT2iPrefix(bundle, {
    ...options,
    guidanceScale: guidanceScaleText,
    prefillMode,
  });

  const contentCount = UNI_MINI_CONFIG.vqGridSize * UNI_MINI_CONFIG.vqGridSize; // 16
  const schedule = semanticTreeActivationCounts(
    contentCount,
    activationSteps,
    refinementSteps,
    shift,
    0,
  );
  const unmaskOrder = stratifiedRandom(UNI_MINI_CONFIG.vqGridSize, stratifiedSeed);
  const rng = createXoshiro128ss(seed);

  let initialContentCodes: ArrayLike<number>;
  if (options.explicitVariates?.initialRootCodes !== undefined) {
    initialContentCodes = options.explicitVariates.initialRootCodes;
  } else {
    const drawn = new Int32Array(contentCount);
    const mask = UNI_MINI_VQ_CODEBOOK_SIZE - 1;
    for (let c = 0; c < contentCount; c++) {
      drawn[c] = rng.nextUint32() & mask;
    }
    initialContentCodes = drawn;
  }

  let treeState = initSemanticTreeState({
    rows: UNI_MINI_CONFIG.vqGridSize,
    contentCols: UNI_MINI_CONFIG.vqGridSize,
    includeMarkers: true,
    depth: bundle.tree.depth,
    refinementSteps,
    initialContentCodes,
  });

  const steps: UniImageDenoiseStepRecord[] = [];
  for (let r = 0; r < schedule.totalRounds; r++) {
    const { stepRecord, nextTreeState } = runSingleImageDenoiseRound(bundle, {
      prefixCache: prepared.prefixCache,
      treeState,
      schedule,
      unmaskOrder,
      roundIndex: r,
      taskKind: 'text_to_image',
      guidanceScaleText,
      guidanceScaleImage,
      temperature,
      topK,
      topP,
      gainPresetId,
      nextUniformOpenUnit: () => rng.nextUniformOpenUnit(),
      nextRandBits: () => rng.nextUint32(),
      explicitRoundUniforms: options.explicitVariates?.roundProposalUniforms?.[r],
      explicitRoundRandBits: options.explicitVariates?.roundRepRandBits?.[r],
      retainFullLogits,
    });
    steps.push(stepRecord);
    treeState = nextTreeState;
  }

  const finalMarkedCodes = new Int32Array(treeState.representatives);
  const {
    contentCodes: finalContentCodes,
    decodedPaddedImage,
    outputImage,
  } = decodeMarkedCodesToImage(bundle, finalMarkedCodes);

  const passCount = 1 + schedule.totalRounds;
  const branchForwardCount = prepared.cfgBatchSize * passCount;

  return Object.freeze({
    taskKind: 'text_to_image' as const,
    cfgBatchSize: prepared.cfgBatchSize,
    cfgShortCircuitApplied: prepared.cfgShortCircuitApplied,
    ...(prepared.hfGuidanceZeroDriftNote !== undefined
      ? { hfGuidanceZeroDriftNote: prepared.hfGuidanceZeroDriftNote }
      : {}),
    guidanceScaleText,
    guidanceScaleImage,
    prefillMode,
    gainPresetId,
    schedule,
    unmaskOrder,
    prefixCache: prepared.prefixCache,
    steps: Object.freeze(steps),
    finalMarkedCodes,
    finalContentCodes,
    decodedPaddedImage,
    outputImage,
    passCount,
    branchForwardCount,
  });
}

/**
 * Execute a complete end-to-end 3-branch Dual-CFG Image Editing trajectory
 * (`B_cfg = 3`, continuous RADIO-H + discrete packed source-VQ, `U18` padding lock
 * and optional `cropToSource` inversion).
 */
export function runImageEditTrajectory(
  bundle: UniFullRuntimeBundle,
  options: UniImageEditSamplerOptions,
): UniImageTrajectoryResult {
  const guidanceMode: UniEditGuidanceMode = options.guidanceMode ?? 'legacy';
  const guidanceScaleText = options.guidanceScaleText ?? 1.6;
  const guidanceScaleImage = options.guidanceScaleImage ?? 2.0;
  const lockSourcePaddingVq = options.lockSourcePaddingVq ?? true;
  const cropToSource = options.cropToSource ?? true;
  const activationSteps = options.activationSteps ?? UNI_MINI_CONFIG.defaultActivationSteps;
  const refinementSteps = options.refinementSteps ?? UNI_MINI_CONFIG.defaultRefinementSteps;
  const shift = options.shift ?? UNI_MINI_CONFIG.defaultShift;
  const temperature = options.temperature ?? 0.8;
  const topK = options.topK ?? 0;
  const topP = options.topP ?? 1.0;
  const gainPresetId: UniGainPresetId = options.gainPresetId ?? UNI_MINI_CONFIG.defaultGainPreset;
  const prefillMode: UniElasticPrefillMode = options.prefillMode ?? 'hf_faithful';
  const seed = (options.seed ?? DEFAULT_EDIT_SEED) >>> 0;
  const stratifiedSeed = options.stratifiedSeed ?? UNI_STRATIFIED_DEFAULT_SEED;
  const retainFullLogits = options.retainFullLogitsOnSteps ?? false;

  const prepared = prepareImageEditPrefix(bundle, {
    ...options,
    gainPresetId,
    prefillMode,
  });

  const contentCount = UNI_MINI_CONFIG.vqGridSize * UNI_MINI_CONFIG.vqGridSize; // 16
  const schedule = semanticTreeActivationCounts(
    contentCount,
    activationSteps,
    refinementSteps,
    shift,
    0,
  );
  const unmaskOrder = stratifiedRandom(UNI_MINI_CONFIG.vqGridSize, stratifiedSeed);
  const rng = createXoshiro128ss(seed);

  let initialContentCodes: ArrayLike<number>;
  if (options.explicitVariates?.initialRootCodes !== undefined) {
    initialContentCodes = options.explicitVariates.initialRootCodes;
  } else {
    const drawn = new Int32Array(contentCount);
    const mask = UNI_MINI_VQ_CODEBOOK_SIZE - 1;
    for (let c = 0; c < contentCount; c++) {
      drawn[c] = rng.nextUint32() & mask;
    }
    initialContentCodes = drawn;
  }

  const activePaddingMask = lockSourcePaddingVq ? prepared.sourcePaddingMask : undefined;
  const activeSourceCodes = lockSourcePaddingVq ? prepared.sourceVqCodes : undefined;

  let treeState = initSemanticTreeState({
    rows: UNI_MINI_CONFIG.vqGridSize,
    contentCols: UNI_MINI_CONFIG.vqGridSize,
    includeMarkers: true,
    depth: bundle.tree.depth,
    refinementSteps,
    initialContentCodes,
    lockedContentMask: activePaddingMask?.lockMask,
    lockedSourceCodes: activeSourceCodes,
  });

  const steps: UniImageDenoiseStepRecord[] = [];
  for (let r = 0; r < schedule.totalRounds; r++) {
    const { stepRecord, nextTreeState } = runSingleImageDenoiseRound(bundle, {
      prefixCache: prepared.prefixCache,
      treeState,
      schedule,
      unmaskOrder,
      roundIndex: r,
      taskKind: 'image_edit',
      guidanceMode,
      guidanceScaleText,
      guidanceScaleImage,
      temperature,
      topK,
      topP,
      gainPresetId,
      sourcePaddingMask: activePaddingMask,
      sourceVqCodes: activeSourceCodes,
      nextUniformOpenUnit: () => rng.nextUniformOpenUnit(),
      nextRandBits: () => rng.nextUint32(),
      explicitRoundUniforms: options.explicitVariates?.roundProposalUniforms?.[r],
      explicitRoundRandBits: options.explicitVariates?.roundRepRandBits?.[r],
      retainFullLogits,
    });
    steps.push(stepRecord);
    treeState = nextTreeState;
  }

  const finalMarkedCodes = new Int32Array(treeState.representatives);
  const {
    contentCodes: finalContentCodes,
    decodedPaddedImage,
    outputImage,
  } = decodeMarkedCodesToImage(
    bundle,
    finalMarkedCodes,
    prepared.sourceLetterbox?.info,
    cropToSource,
  );

  const passCount = 1 + schedule.totalRounds;
  const branchForwardCount = prepared.cfgBatchSize * passCount;

  return Object.freeze({
    taskKind: 'image_edit' as const,
    cfgBatchSize: prepared.cfgBatchSize,
    cfgShortCircuitApplied: prepared.cfgShortCircuitApplied,
    guidanceMode,
    guidanceScaleText,
    guidanceScaleImage,
    prefillMode,
    gainPresetId,
    schedule,
    unmaskOrder,
    prefixCache: prepared.prefixCache,
    sourceLetterbox: prepared.sourceLetterbox,
    sourcePaddingMask: prepared.sourcePaddingMask,
    sourceVqCodes: prepared.sourceVqCodes,
    steps: Object.freeze(steps),
    finalMarkedCodes,
    finalContentCodes,
    decodedPaddedImage,
    outputImage,
    passCount,
    branchForwardCount,
  });
}
