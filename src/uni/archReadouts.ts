/**
 * Headless Architectural Readouts & `D5` Elastic-MoT Comparison (`src/uni/archReadouts.ts`)
 *
 * Clean-room readout builders mapping:
 *   1. All 6 production execution views (`UNI_EXECUTION_VIEWS`) alongside miniature
 *      active float counts (`UNI_MINI_COMPONENT_FOOTPRINT`) and FP32 KV footprints
 *      (`4,352` / `8,704` / `13,056 B/tok`).
 *   2. Exact `D5` HF-Faithful (`34L U` prefill) vs Elastic-Pruned (`18L U` prefill for
 *      T2I, pruned split-`U` for Editing, `34L` unchanged for Understanding/Grounding)
 *      KV comparisons with `stage2UlpDiff: 0`.
 *   3. Dual SI/IEC formatted architecture summary lines using `formatSiIecBytes`
 *      (`./spec`) and `formatTokenCount` (`../model/memory`).
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import { formatTokenCount } from '../model/memory';
import {
  formatSiIecBytes,
  UNI_EXECUTION_VIEWS,
  UNI_MINI_COMPONENT_FOOTPRINT,
  UNI_MINI_CONFIG,
  UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16,
} from './spec';
import type {
  UniElasticD5ComparisonReadout,
  UniElasticPrefillMode,
  UniExecutionViewId,
  UniExecutionViewSpec,
  UniViewArchReadout,
} from './types';

// ============================================================================
// LOCKED PRODUCTION & MINIATURE KV CONSTANTS
// ============================================================================

const PROD_16K_TOKENS = 16_384;
const PROD_KV_BYTES_PER_LAYER_PER_BRANCH =
  UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16 / UNI_MINI_CONFIG.numLayers; // 4,096 B/tok/layer

const MINI_FLOAT32_BYTES = 4;
const MINI_KV_BYTES_PER_LAYER_PER_BRANCH =
  2 * UNI_MINI_CONFIG.kvProjOutDim * MINI_FLOAT32_BYTES; // 128 B/tok/layer
const MINI_KV_BYTES_PER_TOKEN_F32 =
  UNI_MINI_CONFIG.numLayers * MINI_KV_BYTES_PER_LAYER_PER_BRANCH; // 4,352 B/tok

// Derive miniature subsystem float counts directly from `UNI_MINI_COMPONENT_FOOTPRINT`
function getMiniFootprintFloats(id: string): number {
  const entry = UNI_MINI_COMPONENT_FOOTPRINT.find((item) => item.id === id);
  if (!entry) {
    throw new Error(`Missing miniature footprint entry: ${id}`);
  }
  return entry.floatCount;
}

const MINI_U_IO_FLOATS = getMiniFootprintFloats('mini_u_io'); // 1,032
const MINI_EMBED_FLOATS = UNI_MINI_CONFIG.vocabSize * UNI_MINI_CONFIG.hiddenSize; // 512
const MINI_JOINT_HALF_FLOATS = getMiniFootprintFloats('mini_joint_backbone') / 2; // 35,424 (18L, 1 branch)
const MINI_SPLIT_HALF_FLOATS = getMiniFootprintFloats('mini_split_backbone') / 2; // 31,488 (16L, 1 branch)
const MINI_34L_SINGLE_BRANCH_FLOATS = MINI_JOINT_HALF_FLOATS + MINI_SPLIT_HALF_FLOATS; // 66,912
const MINI_G_IO_FLOATS = getMiniFootprintFloats('mini_g_io'); // 4,168
const MINI_CODEC_FLOATS = getMiniFootprintFloats('mini_packed_image_codec'); // 528
const MINI_VISION_FLOATS = getMiniFootprintFloats('mini_radio_vision_and_projector'); // 8,288
const MINI_VQVAE_CODEBOOK_FLOATS =
  UNI_MINI_CONFIG.vqCodebookSize * UNI_MINI_CONFIG.vqLatentDim; // 2,048
const MINI_VQVAE_ENCODER_FLOATS = 19_800; // UNI_MINI_VQVAE_ENCODER_FLOAT_COUNT
const MINI_VQVAE_DECODER_FLOATS = 21_539; // UNI_MINI_VQVAE_DECODER_FLOAT_COUNT

const MINI_ACTIVE_FLOATS_BY_VIEW: Readonly<Record<UniExecutionViewId, number>> = Object.freeze({
  text_understanding: MINI_U_IO_FLOATS + MINI_34L_SINGLE_BRANCH_FLOATS, // 67,944
  visual_understanding:
    MINI_U_IO_FLOATS + MINI_34L_SINGLE_BRANCH_FLOATS + MINI_VISION_FLOATS, // 76,232
  object_grounding:
    MINI_U_IO_FLOATS + MINI_34L_SINGLE_BRANCH_FLOATS + MINI_VISION_FLOATS, // 76,232
  text_to_image:
    MINI_EMBED_FLOATS +
    MINI_JOINT_HALF_FLOATS +
    MINI_34L_SINGLE_BRANCH_FLOATS +
    MINI_G_IO_FLOATS +
    MINI_CODEC_FLOATS +
    MINI_VQVAE_CODEBOOK_FLOATS +
    MINI_VQVAE_DECODER_FLOATS, // 131,131
  image_edit:
    MINI_EMBED_FLOATS +
    MINI_JOINT_HALF_FLOATS +
    MINI_34L_SINGLE_BRANCH_FLOATS +
    MINI_G_IO_FLOATS +
    MINI_CODEC_FLOATS +
    MINI_VQVAE_CODEBOOK_FLOATS +
    MINI_VQVAE_DECODER_FLOATS +
    MINI_VISION_FLOATS +
    MINI_VQVAE_ENCODER_FLOATS, // 159,219
  interleaved_mixed:
    MINI_U_IO_FLOATS +
    2 * MINI_34L_SINGLE_BRANCH_FLOATS +
    MINI_G_IO_FLOATS +
    MINI_CODEC_FLOATS, // 139,552
});

function getExecutionViewSpec(viewId: UniExecutionViewId): UniExecutionViewSpec {
  const spec = UNI_EXECUTION_VIEWS.find((v) => v.id === viewId);
  if (!spec) {
    throw new RangeError(`Unknown UNI execution view: ${String(viewId)}`);
  }
  return spec;
}

function formatCompactParams(params: number): string {
  if (params >= 1_000_000_000) {
    return `${(params / 1_000_000_000).toFixed(2)}B`;
  }
  if (params >= 1_000_000) {
    return `${(params / 1_000_000).toFixed(2)}M`;
  }
  if (params >= 1_000) {
    return `${(params / 1_000).toFixed(2)}K`;
  }
  return `${params}`;
}

// ============================================================================
// PUBLIC ARCHITECTURAL READOUT BUILDERS
// ============================================================================

/**
 * Build architectural & memory readouts for all 6 `UNI_EXECUTION_VIEWS`,
 * pairing verified production parameters / BF16 KV footprints with miniature
 * active float counts and FP32 KV bytes/token (`4,352` / `8,704` / `13,056 B/tok`).
 */
export function buildUniViewArchReadouts(): readonly UniViewArchReadout[] {
  return Object.freeze(
    UNI_EXECUTION_VIEWS.map((view): UniViewArchReadout => {
      const readout: UniViewArchReadout = {
        viewId: view.id,
        label: view.label,
        paperShorthand: view.paperShorthand,
        prodActiveParams: view.activeCausalGraphParams,
        prodEagerHfParams: view.eagerHfExecutedParams,
        prodWastedSplitUParams: view.wastedSplitUParams,
        prodResidentHfBytes: view.residentHfBytes,
        prodResidentSglangBytes: view.residentSglangBytes,
        prodKvBytesPerTokenBf16: view.kvBytesPerTokenBf16,
        prodKvAt16kBytes: view.kvBytesPerTokenBf16 * PROD_16K_TOKENS,
        miniActiveFloats: MINI_ACTIVE_FLOATS_BY_VIEW[view.id],
        miniKvBytesPerTokenF32: MINI_KV_BYTES_PER_TOKEN_F32 * view.cfgBranches,
        cfgBranches: view.cfgBranches,
        releasedEndToEndLoop: view.releasedEndToEndLoop,
        ...(view.badge !== undefined ? { badge: view.badge } : {}),
      };
      return Object.freeze(readout);
    }),
  );
}

/**
 * Compute exact `D5` HF-Faithful (`34L U` prefill) vs Elastic-Pruned KV comparison
 * for a given execution view and prefix token count:
 *   - `text_to_image` (`B_cfg = 2`): `hf_faithful` caches `34L` on both `U` rows;
 *     `elastic_pruned` executes `18L` (`L0..17`), saving `16 / 34` prefix KV bytes.
 *   - `image_edit` (`B_cfg = 3`): `hf_faithful` caches `34L` on all 3 rows (2 pure-`U`
 *     rows + 1 mixed `U+G` row); `elastic_pruned` prunes split `L18..33` `U` computation,
 *     dropping `16L` on the 2 pure-`U` rows (`(2 / 3) * (16 / 34)` KV saved) while
 *     retaining `L18..33` `G` on the full source-VQ row.
 *   - Understanding / Grounding / Interleaved: `34L` unchanged (`0` wasted fraction).
 *   - All views guarantee `stage2UlpDiff: 0`.
 */
export function buildUniElasticD5Comparison(
  viewId: UniExecutionViewId,
  prefixTokens: number,
): UniElasticD5ComparisonReadout {
  if (!Number.isInteger(prefixTokens) || prefixTokens < 0) {
    throw new RangeError(`prefixTokens must be a non-negative integer, got ${String(prefixTokens)}`);
  }

  const view = getExecutionViewSpec(viewId);
  const numLayers = UNI_MINI_CONFIG.numLayers; // 34
  const dualTowerLayers = UNI_MINI_CONFIG.dualTowerLayers; // 18
  const splitLayers = UNI_MINI_CONFIG.splitLayers; // 16
  const cfgBranches = view.cfgBranches;

  const prodHfFaithfulPrefixKvBytes =
    prefixTokens * cfgBranches * numLayers * PROD_KV_BYTES_PER_LAYER_PER_BRANCH;
  const miniHfFaithfulPrefixKvBytes =
    prefixTokens * cfgBranches * numLayers * MINI_KV_BYTES_PER_LAYER_PER_BRANCH;

  let elasticPrunedLayersExecuted: number = numLayers;
  let prodSavedPrefixKvBytes = 0;
  let miniSavedPrefixKvBytes = 0;

  if (viewId === 'text_to_image') {
    elasticPrunedLayersExecuted = dualTowerLayers; // 18L U prefill
    prodSavedPrefixKvBytes =
      prefixTokens * cfgBranches * splitLayers * PROD_KV_BYTES_PER_LAYER_PER_BRANCH;
    miniSavedPrefixKvBytes =
      prefixTokens * cfgBranches * splitLayers * MINI_KV_BYTES_PER_LAYER_PER_BRANCH;
  } else if (viewId === 'image_edit') {
    // Rows 0 (uncond) & 1 (vq_dropped) are pure U and drop L18..33; row 2 (full) keeps G on L18..33.
    elasticPrunedLayersExecuted = numLayers;
    const pureUPrunedBranches = 2;
    prodSavedPrefixKvBytes =
      prefixTokens * pureUPrunedBranches * splitLayers * PROD_KV_BYTES_PER_LAYER_PER_BRANCH;
    miniSavedPrefixKvBytes =
      prefixTokens * pureUPrunedBranches * splitLayers * MINI_KV_BYTES_PER_LAYER_PER_BRANCH;
  }

  const prodElasticPrunedPrefixKvBytes = prodHfFaithfulPrefixKvBytes - prodSavedPrefixKvBytes;
  const miniElasticPrunedPrefixKvBytes = miniHfFaithfulPrefixKvBytes - miniSavedPrefixKvBytes;
  const wastedPrefixKvFraction =
    prodHfFaithfulPrefixKvBytes > 0
      ? prodSavedPrefixKvBytes / prodHfFaithfulPrefixKvBytes
      : 0;

  return Object.freeze({
    viewId,
    prefixTokens,
    cfgBranches,
    hfFaithfulLayersExecuted: numLayers,
    elasticPrunedLayersExecuted,
    prodHfFaithfulPrefixKvBytes,
    prodElasticPrunedPrefixKvBytes,
    prodSavedPrefixKvBytes,
    miniHfFaithfulPrefixKvBytes,
    miniElasticPrunedPrefixKvBytes,
    miniSavedPrefixKvBytes,
    wastedPrefixKvFraction,
    stage2UlpDiff: 0,
  });
}

/**
 * Format dual SI/IEC architectural summary lines for a selected execution view,
 * prefill mode (`hf_faithful` | `elastic_pruned`), and prefix length.
 */
export function formatUniArchSummaryLines(
  viewId: UniExecutionViewId,
  prefillMode: UniElasticPrefillMode,
  prefixTokens = 16,
): readonly string[] {
  const readout = buildUniViewArchReadouts().find((r) => r.viewId === viewId);
  if (!readout) {
    throw new RangeError(`Unknown UNI execution view: ${String(viewId)}`);
  }
  const d5 = buildUniElasticD5Comparison(viewId, prefixTokens);

  const lines: string[] = [
    `${readout.label} (${readout.paperShorthand}) · B_cfg=${readout.cfgBranches}`,
    `Prod Params: Active ${formatCompactParams(readout.prodActiveParams)} · Eager HF ${formatCompactParams(readout.prodEagerHfParams)} · Wasted Split-U ${formatCompactParams(readout.prodWastedSplitUParams)}`,
    `Prod Resident: HF ${formatSiIecBytes(readout.prodResidentHfBytes)}${
      readout.prodResidentSglangBytes !== null
        ? ` · SGLang ${formatSiIecBytes(readout.prodResidentSglangBytes)}`
        : ' · SGLang unsupported'
    }`,
    `Prod BF16 KV: ${formatSiIecBytes(readout.prodKvBytesPerTokenBf16)}/tok · ${formatTokenCount(PROD_16K_TOKENS)} = ${formatSiIecBytes(readout.prodKvAt16kBytes)}`,
    `Mini FP32: ${readout.miniActiveFloats.toLocaleString('en-US')} active floats (${formatSiIecBytes(readout.miniActiveFloats * MINI_FLOAT32_BYTES)}) · KV ${formatSiIecBytes(readout.miniKvBytesPerTokenF32)}/tok`,
  ];

  const activeProdPrefix =
    prefillMode === 'elastic_pruned'
      ? d5.prodElasticPrunedPrefixKvBytes
      : d5.prodHfFaithfulPrefixKvBytes;
  const activeMiniPrefix =
    prefillMode === 'elastic_pruned'
      ? d5.miniElasticPrunedPrefixKvBytes
      : d5.miniHfFaithfulPrefixKvBytes;
  const wastedPct = (d5.wastedPrefixKvFraction * 100).toFixed(1);

  lines.push(
    `D5 Prefill [${prefillMode}] @ ${formatTokenCount(prefixTokens)} tok: Prod ${formatSiIecBytes(activeProdPrefix)} · Mini ${formatSiIecBytes(activeMiniPrefix)} · Saved ${formatSiIecBytes(d5.prodSavedPrefixKvBytes)} (${wastedPct}%) · Stage-2 ${d5.stage2UlpDiff} ULP`,
  );

  if (readout.badge !== undefined) {
    lines.push(`Note: ${readout.badge}`);
  }

  return Object.freeze(lines);
}
