/**
 * Milestone UNI5 — View 5 (`refinement`): Image Refinement Canvas Scene
 * & `128 × 128` `RG32F` Scalar/Mask Atlas (`src/uni/scene/refinementCanvasScene.ts`)
 *
 * Clean-room headless 3D spatial scene builder for Nemotron-Labs-Diffusion-Uni:
 *   - Centerpiece `4×6` (`24`-cell) marked VQ child plates (`4×4` content + `+2` row/end marker cols):
 *       1. `vq_code_stage` (`4×6`): live `codeStage` (`-1` unactivated, `0..N` refining/committed;
 *          `G = 1.0` ONLY on explicitly masked `stage === -1` cells, `G = 0.0` on all active/marker cells)
 *       2. `vq_paper_age` (`4×6`): `U9` reversed polarity `paperAge = max(0, N - max(0, stage))` (`G = 0.0`)
 *       3. `vq_tree_level` (`4×6`): tree split `level ∈ 0..8` / candidate ball size $2^{8-\ell}$ (`G = 0.0`)
 *       4. `vq_mask_noise` (`4×6`): `U10` additive mask-noise weight $\eta_i\,\mathbf{e}(\text{MASK})$ (`G = 0.0`)
 *   - Live `stratified_random(4, seed=42)` unmasking-order plate (`4×4`, `G = 0.0`) with optional
 *     `showStratifiedOrder` framing overlay
 *   - `vq_tree_embeddings` (`24×8`): live `U10` continuous input embeddings
 *     $\mathbf{E}_{\text{gen}}[u_i] + \eta_i\,\mathbf{E}_{\text{gen}}[\text{MASK}]$ (`G = 0.0`)
 *   - Coupled via `2×2` `PackedLinearImageCodec` brackets/ribbons to:
 *       - `parent_tokens` (`6×8`): `2×3` parent transformer tokens (`G = 0.0`)
 *       - `fixed_prefix_tokens` (`4×8`): `4` uncompressed fixed-prefix tokens `259..262` (`G = 0.0`)
 *   - 2-way T2I CFG (`B=2`) / 3-way Editing Dual-CFG (`B=3`) top-32 logit comparison slices
 *     pulled live from `ctx.resolveStepHeadLogits()` (`G = 0.0`, `groupIndex: branchRow`,
 *     `semanticTag` carrying `groupRole=cfg;cfg=${branchRow}` for `pickHitToUniCell`)
 *   - Full trajectory scrubbing across any historical `recordId` (`options?.recordId`)
 *   - Synced `16×16` RGB `decoded` `UniRgbPreviewDescriptor` (`D4`: 0 RGB in `RG32F`,
 *     badged with `UNI_SCENE_VISUAL_HONESTY_BADGE`)
 *   - Graceful deterministic fallback on Understanding / Grounding sessions (never throws;
 *     emits informative muted provenance badge)
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import type { Vec3 } from '../../model/spec';
import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import type { FlowRibbonColor } from '../../render/ribbons';
import {
  FRAME_XZ_AXES,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_DIMMED,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NEUTRAL,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
} from '../../render/sheets';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fAtlasWriter,
  type Rg32fValueAtlas,
} from '../../scene/atlasPacker';
import {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_MOE_AMBER,
  CELL_SSM_CYAN,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../scene/spine';
import {
  compressPackedImageEmbeddings,
  embedFixedPrefixTokens,
} from '../codec';
import { decodeMarkedCodesToImage } from '../imageSampler';
import type { UniSession } from '../session';
import {
  buildSemanticTreeEmbeddings,
  getStagePolarityPair,
  initSemanticTreeState,
  stratifiedRandom,
} from '../tree';
import {
  UNI_MINI_CONFIG,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_STRATIFIED_DEFAULT_SEED,
  type UniCfgBatchSize,
  type UniGainPresetId,
  type UniRgbImage,
} from '../types';
import {
  auditAtlasRegions,
  computeAtlasContentKey,
  packPickBoxFromFrame,
  resolveUniSessionContext,
  UNI_FLOW_COLORS,
  UNI_SCENE_VISUAL_HONESTY_BADGE,
  UniFlatFlowBufferBuilder,
  UniLineBufferBuilder,
  UniRibbonBufferBuilder,
  UniSheetBufferBuilder,
  UniSolidBufferBuilder,
  validateSceneGeometryFinite,
} from './provenance';
import type {
  UniAtlasLayoutAudit,
  UniAtlasRegionBox,
  UniOverlayAnchor,
  UniProvenanceBadge,
  UniRefinementSceneOptions,
  UniResolvedSessionContext,
  UniRgbPreviewDescriptor,
  UniSpatialScene,
} from './types';

// ============================================================================
// CONSTANTS & TYPES
// ============================================================================

export const UNI_REFINEMENT_ATLAS_SIZE = 128 as const;
export const UNI_REFINEMENT_ATLAS_BUDGET_BYTES = 131_072 as const; // 128 * 128 * 8 B
export const UNI_REFINEMENT_REGION_COUNT = 12 as const;
export const UNI_REFINEMENT_TOP_LOGIT_COLS = 32 as const;

export type UniRefinementStageId =
  | 'vq_code_stage'
  | 'vq_paper_age'
  | 'vq_tree_level'
  | 'vq_mask_noise'
  | 'stratified_order'
  | 'vq_tree_embeddings'
  | 'parent_tokens'
  | 'fixed_prefix_tokens'
  | 'cfg_branch_0_logits'
  | 'cfg_branch_1_logits'
  | 'cfg_branch_2_logits'
  | 'cfg_guided_logits';

interface RefinementAtlasSliceEntry {
  readonly id: UniRefinementStageId;
  readonly region: UniAtlasRegionBox;
  readonly slice: AtlasSlice;
}

interface RefinementDataState {
  readonly isFallback: boolean;
  readonly fallbackReason: string | null;
  readonly roundLabel: string;
  readonly roundIndex: number | null;
  readonly isFlushRound: boolean;
  readonly refinementSteps: number;
  readonly stages: Int32Array; // [24]
  readonly paperAges: Int32Array; // [24]
  readonly levels: Int32Array; // [24]
  readonly ballSizes: Int32Array; // [24]
  readonly noiseWeights: Float32Array; // [24]
  readonly representatives: Int32Array; // [24]
  readonly proposals: Int32Array; // [24]
  readonly treeEmbeddings: Float32Array; // [24 * 8]
  readonly parentTokens: Float32Array; // [6 * 8]
  readonly fixedPrefixTokens: Float32Array; // [4 * 8]
  readonly unmaskOrder: Int32Array; // [16]
  readonly stratifiedRankNormalized: Float32Array; // [16]
  readonly activatedContentMask: Uint8Array; // [16]
  readonly lockedCellCount: number;
  readonly committedContentCount: number;
  readonly cfgBatchSize: UniCfgBatchSize;
  readonly branchLabels: readonly [string, string, string];
  readonly cfgSlice0: Float32Array; // [24 * 32]
  readonly cfgSlice1: Float32Array; // [24 * 32]
  readonly cfgSlice2: Float32Array; // [24 * 32]
  readonly cfgGuidedSlice: Float32Array; // [24 * 32]
  readonly selectedVqCodeIds: Int32Array; // [32]
  readonly decodedPaddedImage: UniRgbImage; // 16x16
}

// ============================================================================
// INTERNAL RG32F MATRIX PACKER (NON-EXPORTED TO AVOID COLLISION)
// ============================================================================

interface LocalMatrixPackParams {
  readonly id: UniRefinementStageId;
  readonly x0: number;
  readonly y0: number;
  readonly rows: number;
  readonly cols: number;
  readonly values: ArrayLike<number>;
  readonly mode: 0 | 1;
  readonly isMaskOrScore: boolean;
  readonly gChannelFlags?: ArrayLike<number>;
  readonly explicitScale?: number;
}

function packRefinementMatrixToRg32f(
  writer: Rg32fAtlasWriter,
  params: LocalMatrixPackParams,
): RefinementAtlasSliceEntry {
  const { id, x0, y0, rows, cols, values, mode, isMaskOrScore, gChannelFlags, explicitScale } =
    params;
  const cellCount = rows * cols;
  if (x0 < 0 || y0 < 0 || x0 + cols > writer.width || y0 + rows > writer.height) {
    throw new RangeError(
      `Refinement RG32F region "${id}" (${x0},${y0} ${cols}x${rows}) exceeds ${writer.width}x${writer.height}`,
    );
  }
  if (values.length < cellCount) {
    throw new RangeError(
      `Refinement RG32F region "${id}" values length (${values.length}) < ${cellCount}`,
    );
  }

  let scale = explicitScale;
  if (scale === undefined) {
    if (mode === 1) {
      scale = 1.0;
    } else {
      const buf =
        values instanceof Float32Array ? values.subarray(0, cellCount) : Float32Array.from(values);
      scale = 1.0 / computeMaxAbs(buf, 1e-4);
    }
  }

  for (let r = 0; r < rows; r++) {
    const srcRow = r * cols;
    const dstY = y0 + r;
    for (let c = 0; c < cols; c++) {
      const idx = srcRow + c;
      const rawVal = values[idx]!;
      const finiteVal = Number.isFinite(rawVal) ? rawVal : 0.0;
      const isMasked = gChannelFlags !== undefined && gChannelFlags[idx]! > 0.5;
      if (isMasked) {
        const texIdx = (dstY * writer.width + (x0 + c)) * 2;
        writer.data[texIdx] = finiteVal;
        writer.data[texIdx + 1] = 1.0;
      } else {
        writer.setTexel(x0 + c, dstY, finiteVal, false);
      }
    }
  }

  return Object.freeze({
    id,
    region: Object.freeze({
      id,
      x0,
      y0,
      cols,
      rows,
      mode,
      isMaskOrScore,
    }),
    slice: Object.freeze({
      rect: writer.makeRect(x0, y0, cols, rows),
      scale,
      mode,
    }),
  });
}

// ============================================================================
// DATA EXTRACTION FROM LIVE SESSION (OR GRACEFUL DETERMINISTIC FALLBACK)
// ============================================================================

function selectTopVqCodeColumns(
  referenceLogits: Float32Array,
  representatives: Int32Array,
  gridLen: number,
  vocabSize: number = UNI_MINI_VQ_CODEBOOK_SIZE,
  sliceCols: number = UNI_REFINEMENT_TOP_LOGIT_COLS,
): Int32Array {
  const chosen = new Set<number>();
  for (let i = 0; i < gridLen && chosen.size < sliceCols; i++) {
    const rep = representatives[i]!;
    if (rep >= 0 && rep < vocabSize) {
      chosen.add(rep);
    }
  }

  const peakLogits = new Float32Array(vocabSize);
  peakLogits.fill(-Infinity);
  for (let i = 0; i < gridLen; i++) {
    const base = i * vocabSize;
    for (let k = 0; k < vocabSize; k++) {
      const v = referenceLogits[base + k]!;
      if (Number.isFinite(v) && v > peakLogits[k]!) {
        peakLogits[k] = v;
      }
    }
  }

  const ranked = new Array<number>(vocabSize);
  for (let k = 0; k < vocabSize; k++) ranked[k] = k;
  ranked.sort((a, b) => {
    const diff = peakLogits[b]! - peakLogits[a]!;
    return diff !== 0 ? diff : a - b;
  });

  for (let i = 0; i < ranked.length && chosen.size < sliceCols; i++) {
    chosen.add(ranked[i]!);
  }

  return Int32Array.from(Array.from(chosen).sort((a, b) => a - b).slice(0, sliceCols));
}

function sliceLogitsByColumns(
  fullLogits: Float32Array,
  selectedCols: Int32Array,
  gridLen: number,
  vocabSize: number = UNI_MINI_VQ_CODEBOOK_SIZE,
): Float32Array {
  const sliceCols = selectedCols.length;
  const out = new Float32Array(gridLen * sliceCols);
  for (let r = 0; r < gridLen; r++) {
    const srcBase = r * vocabSize;
    const dstBase = r * sliceCols;
    for (let c = 0; c < sliceCols; c++) {
      const v = fullLogits[srcBase + selectedCols[c]!]!;
      out[dstBase + c] = Number.isFinite(v) ? v : 0.0;
    }
  }
  return out;
}

function resolveRefinementDataState(ctx: UniResolvedSessionContext): RefinementDataState {
  const { bundle, branch, record } = ctx;
  const core = bundle.core;
  const cfg = core.config;
  const d = cfg.hiddenSize; // 8
  const gridRows = cfg.vqGridSize; // 4
  const markedCols = cfg.vqMarkedGridCols; // 6
  const gridLen = gridRows * markedCols; // 24
  const contentCount = gridRows * gridRows; // 16
  const fixedPrefixCount = cfg.genFixedPrefixTokens; // 4
  const parentCount = cfg.packedGridRows * cfg.packedGridCols; // 6
  const vqVocab = cfg.vqCodebookSize; // 256
  const treeDepth = cfg.vqTreeDepth; // 8
  const refinementSteps = branch.refinementSteps;
  const gainPresetId: UniGainPresetId = ctx.gainPresetId;

  const unmaskOrder = branch.unmaskOrder ?? stratifiedRandom(gridRows, UNI_STRATIFIED_DEFAULT_SEED);
  const stratifiedRankNormalized = new Float32Array(contentCount);
  const denomOrder = Math.max(1, contentCount - 1);
  for (let rank = 0; rank < contentCount; rank++) {
    const cellIdx = unmaskOrder[rank]!;
    if (cellIdx >= 0 && cellIdx < contentCount) {
      stratifiedRankNormalized[cellIdx] = rank / denomOrder;
    }
  }

  const isGenTask = branch.taskMode === 'text_to_image' || branch.taskMode === 'image_edit';
  const hasDenoiseState = isGenTask && record.kind === 'image_denoise' && record.imageState !== undefined;

  let stages: Int32Array;
  let levels: Int32Array;
  let noiseWeights: Float32Array;
  let representatives: Int32Array;
  let proposals: Int32Array;
  let lockedCellCount = 0;
  let committedContentCount = 0;
  let roundIndex: number | null = null;
  let isFlushRound = false;
  let roundLabel: string;
  let isFallback = false;
  let fallbackReason: string | null = null;
  const activatedContentMask = new Uint8Array(contentCount);

  if (hasDenoiseState && record.imageState) {
    const imgState = record.imageState;
    stages = new Int32Array(imgState.stages);
    levels = new Int32Array(imgState.levels);
    noiseWeights = new Float32Array(imgState.noiseWeights);
    representatives = new Int32Array(imgState.representatives);
    proposals = new Int32Array(imgState.proposals);
    lockedCellCount = imgState.lockedCellCount;
    committedContentCount = imgState.committedContentCount;
    roundIndex = imgState.roundIndex;
    isFlushRound = imgState.isFlushRound;
    roundLabel = `Round ${imgState.roundIndex + 1}${imgState.isFlushRound ? ' (Flush)' : ''}`;

    // Identify content cells that are currently active (stage >= 0) or newly activated in this round
    const cumActivated = branch.schedule?.cumulativeActivations[imgState.roundIndex];
    if (cumActivated !== undefined) {
      for (let k = 0; k < Math.min(cumActivated, unmaskOrder.length); k++) {
        activatedContentMask[unmaskOrder[k]!] = 1;
      }
    }
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridRows; c++) {
        if (stages[r * markedCols + c]! >= 0) {
          activatedContentMask[r * gridRows + c] = 1;
        }
      }
    }
  } else {
    const initState = initSemanticTreeState({
      refinementSteps,
      lockedContentMask: isGenTask ? branch.sourcePaddingMask?.lockMask : undefined,
      lockedSourceCodes: isGenTask ? branch.sourceVqCodes : undefined,
    });
    stages = new Int32Array(initState.stages);
    levels = new Int32Array(initState.levels);
    noiseWeights = new Float32Array(initState.noiseWeights);
    representatives = new Int32Array(initState.representatives);
    proposals = new Int32Array(initState.representatives);
    if (isGenTask && branch.sourcePaddingMask) {
      lockedCellCount = branch.sourcePaddingMask.lockedCellCount;
      committedContentCount = lockedCellCount;
    }
    if (!isGenTask) {
      isFallback = true;
      fallbackReason = `Refinement Preview Fallback · ${ctx.viewReadout.label} (No Active G Denoise Trajectory)`;
      roundLabel = 'Fallback Sim (Initial Tree State)';
    } else {
      roundLabel = 'Step 0 (Prefill · Unactivated Grid)';
    }
  }

  const paperAges = new Int32Array(gridLen);
  const ballSizes = new Int32Array(gridLen);
  for (let i = 0; i < gridLen; i++) {
    const pair = getStagePolarityPair(stages[i]!, treeDepth, refinementSteps);
    paperAges[i] = pair.paperAge;
    ballSizes[i] = pair.ballSize;
  }

  // U10 continuous additive mask-noise embeddings [24, 8]
  const treeEmbeddings = buildSemanticTreeEmbeddings(
    representatives,
    noiseWeights,
    core.genEmbedding,
    d,
  );

  // Live logits & parent/fixed-prefix transformer hidden states
  const headLogits = ctx.resolveStepHeadLogits();
  const focusBranchIdx = Math.max(0, ctx.cfgBatchSize - 1);
  let fixedPrefixTokens: Float32Array;
  let parentTokens: Float32Array;

  const activeFinalNorm = headLogits.perBranchFinalNorm[focusBranchIdx];
  if (
    hasDenoiseState &&
    activeFinalNorm !== null &&
    activeFinalNorm.length >= (fixedPrefixCount + parentCount) * d
  ) {
    fixedPrefixTokens = new Float32Array(
      activeFinalNorm.subarray(0, fixedPrefixCount * d),
    );
    parentTokens = new Float32Array(
      activeFinalNorm.subarray(fixedPrefixCount * d, (fixedPrefixCount + parentCount) * d),
    );
  } else {
    fixedPrefixTokens = embedFixedPrefixTokens(core.genEmbedding, d).subarray(
      0,
      fixedPrefixCount * d,
    );
    const compressed = compressPackedImageEmbeddings(
      treeEmbeddings,
      gridRows,
      markedCols,
      bundle.codec,
      { isSourceEdit: false, gainPresetId },
    );
    parentTokens = compressed.output.subarray(0, parentCount * d);
  }

  // Build 2-way T2I / 3-way Editing CFG comparison logits ([24, 256] per branch)
  let rawBranch0: Float32Array;
  let rawBranch1: Float32Array;
  let rawBranch2: Float32Array;
  let rawGuided: Float32Array;
  let branchLabels: readonly [string, string, string];

  if (hasDenoiseState && headLogits.guidedGenLogits !== null) {
    rawGuided = headLogits.guidedGenLogits;
    if (ctx.cfgBatchSize === 3) {
      rawBranch0 = headLogits.perBranchGenLogits[0] ?? rawGuided;
      rawBranch1 = headLogits.perBranchGenLogits[1] ?? rawGuided;
      rawBranch2 = headLogits.perBranchGenLogits[2] ?? rawGuided;
      branchLabels = ['CFG Row 0 · Uncond', 'CFG Row 1 · VQ-Dropped', 'CFG Row 2 · Full Cond'];
    } else if (ctx.cfgBatchSize === 2) {
      rawBranch0 = headLogits.perBranchGenLogits[0] ?? rawGuided;
      rawBranch1 = headLogits.perBranchGenLogits[1] ?? rawGuided;
      rawBranch2 = rawGuided;
      branchLabels = ['CFG Row 0 · Uncond', 'CFG Row 1 · Cond', 'T2I CFG · Guided'];
    } else {
      rawBranch0 = headLogits.perBranchGenLogits[0] ?? rawGuided;
      rawBranch1 = rawGuided;
      rawBranch2 = rawGuided;
      branchLabels = ['CFG Row 0 · Cond Only (U30)', 'Guided Mirror', 'Guided Mirror'];
    }
  } else {
    // Deterministic prefill / fallback gen_predictor logits from treeEmbeddings
    const fallbackLogits = new Float32Array(gridLen * vqVocab);
    for (let i = 0; i < gridLen; i++) {
      const eBase = i * d;
      const lBase = i * vqVocab;
      for (let k = 0; k < vqVocab; k++) {
        const wBase = k * d;
        let dot = 0;
        for (let c = 0; c < d; c++) {
          dot += treeEmbeddings[eBase + c]! * core.genPredictor[wBase + c]!;
        }
        fallbackLogits[lBase + k] = Math.fround(dot);
      }
    }
    rawBranch0 = fallbackLogits;
    rawBranch1 = fallbackLogits;
    rawBranch2 = fallbackLogits;
    rawGuided = fallbackLogits;
    branchLabels = isFallback
      ? ['Fallback Logits (Row 0)', 'Fallback Logits (Row 1)', 'Fallback Logits (Row 2)']
      : ['Prefill Uncond Sim', 'Prefill Cond Sim', 'Prefill Guided Sim'];
  }

  const selectedVqCodeIds = selectTopVqCodeColumns(
    rawGuided,
    representatives,
    gridLen,
    vqVocab,
    UNI_REFINEMENT_TOP_LOGIT_COLS,
  );
  const cfgSlice0 = sliceLogitsByColumns(rawBranch0, selectedVqCodeIds, gridLen, vqVocab);
  const cfgSlice1 = sliceLogitsByColumns(rawBranch1, selectedVqCodeIds, gridLen, vqVocab);
  const cfgSlice2 = sliceLogitsByColumns(rawBranch2, selectedVqCodeIds, gridLen, vqVocab);
  const cfgGuidedSlice = sliceLogitsByColumns(rawGuided, selectedVqCodeIds, gridLen, vqVocab);

  const decodedPaddedImage = hasDenoiseState
    ? ctx.session.decodeBranchImage(branch.branchId, record.recordId).decodedPaddedImage
    : decodeMarkedCodesToImage(bundle, representatives).decodedPaddedImage;

  return {
    isFallback,
    fallbackReason,
    roundLabel,
    roundIndex,
    isFlushRound,
    refinementSteps,
    stages,
    paperAges,
    levels,
    ballSizes,
    noiseWeights,
    representatives,
    proposals,
    treeEmbeddings,
    parentTokens,
    fixedPrefixTokens,
    unmaskOrder,
    stratifiedRankNormalized,
    activatedContentMask,
    lockedCellCount,
    committedContentCount,
    cfgBatchSize: ctx.cfgBatchSize,
    branchLabels,
    cfgSlice0,
    cfgSlice1,
    cfgSlice2,
    cfgGuidedSlice,
    selectedVqCodeIds,
    decodedPaddedImage,
  };
}

// ============================================================================
// 128 × 128 RG32F ATLAS PACKING (12 NON-OVERLAPPING RECTS, <= 14)
// ============================================================================

function packRefinementAtlas(
  state: RefinementDataState,
  extraKeyTag: string,
): {
  readonly atlas: Rg32fValueAtlas;
  readonly atlasKey: string;
  readonly regions: readonly UniAtlasRegionBox[];
  readonly audit: UniAtlasLayoutAudit;
  readonly slices: Readonly<Record<UniRefinementStageId, RefinementAtlasSliceEntry>>;
} {
  const W = UNI_REFINEMENT_ATLAS_SIZE;
  const H = UNI_REFINEMENT_ATLAS_SIZE;
  const writer = createRg32fAtlas(W, H);

  const gridLen = 24;
  const codeStageNorm = new Float32Array(gridLen);
  const codeStageG = new Float32Array(gridLen);
  const paperAgeNorm = new Float32Array(gridLen);
  const treeLevelNorm = new Float32Array(gridLen);

  const nSteps = Math.max(1, state.refinementSteps);
  const depth = UNI_MINI_CONFIG.vqTreeDepth;

  for (let i = 0; i < gridLen; i++) {
    const st = state.stages[i]!;
    codeStageNorm[i] = (st + 1) / (nSteps + 1);
    // CRITICAL SheetsPass rule: G = 1.0 ONLY on explicitly masked unactivated stage === -1 cells;
    // G = 0.0 on all active, locked, and +2 structural marker cells!
    codeStageG[i] = st < 0 ? 1.0 : 0.0;
    paperAgeNorm[i] = state.paperAges[i]! / nSteps;
    treeLevelNorm[i] = state.levels[i]! / depth;
  }

  const entries: RefinementAtlasSliceEntry[] = [];
  const sliceMap = {} as Record<UniRefinementStageId, RefinementAtlasSliceEntry>;
  const add = (params: LocalMatrixPackParams) => {
    const entry = packRefinementMatrixToRg32f(writer, params);
    entries.push(entry);
    sliceMap[params.id] = entry;
  };

  // Band 0 (y = 0..23): 3 CFG Branch Top-32 Logit Comparison Plates (24 x 32 each)
  add({
    id: 'cfg_branch_0_logits',
    x0: 0,
    y0: 0,
    rows: 24,
    cols: 32,
    values: state.cfgSlice0,
    mode: 0,
    isMaskOrScore: true,
  });
  add({
    id: 'cfg_branch_1_logits',
    x0: 34,
    y0: 0,
    rows: 24,
    cols: 32,
    values: state.cfgSlice1,
    mode: 0,
    isMaskOrScore: true,
  });
  add({
    id: 'cfg_branch_2_logits',
    x0: 68,
    y0: 0,
    rows: 24,
    cols: 32,
    values: state.cfgSlice2,
    mode: 0,
    isMaskOrScore: true,
  });

  // Band 1 (y = 26..49): Guided Logits (24x32), U10 Tree Embeddings (24x8),
  // Parent Tokens (6x8), Fixed-Prefix Tokens (4x8), and 4x6 / 4x4 Centerpiece Plates
  add({
    id: 'cfg_guided_logits',
    x0: 0,
    y0: 26,
    rows: 24,
    cols: 32,
    values: state.cfgGuidedSlice,
    mode: 0,
    isMaskOrScore: true,
  });
  add({
    id: 'vq_tree_embeddings',
    x0: 34,
    y0: 26,
    rows: 24,
    cols: 8,
    values: state.treeEmbeddings,
    mode: 0,
    isMaskOrScore: false,
  });
  add({
    id: 'parent_tokens',
    x0: 44,
    y0: 26,
    rows: 6,
    cols: 8,
    values: state.parentTokens,
    mode: 0,
    isMaskOrScore: false,
  });
  add({
    id: 'fixed_prefix_tokens',
    x0: 44,
    y0: 34,
    rows: 4,
    cols: 8,
    values: state.fixedPrefixTokens,
    mode: 0,
    isMaskOrScore: false,
  });

  // Centerpiece 4x6 plates + 4x4 stratified order plate (Band 1 right, y = 26..29)
  add({
    id: 'vq_code_stage',
    x0: 54,
    y0: 26,
    rows: 4,
    cols: 6,
    values: codeStageNorm,
    mode: 1,
    isMaskOrScore: true,
    gChannelFlags: codeStageG,
    explicitScale: 1.0,
  });
  add({
    id: 'vq_paper_age',
    x0: 62,
    y0: 26,
    rows: 4,
    cols: 6,
    values: paperAgeNorm,
    mode: 1,
    isMaskOrScore: false,
    explicitScale: 1.0,
  });
  add({
    id: 'vq_tree_level',
    x0: 70,
    y0: 26,
    rows: 4,
    cols: 6,
    values: treeLevelNorm,
    mode: 1,
    isMaskOrScore: false,
    explicitScale: 1.0,
  });
  add({
    id: 'vq_mask_noise',
    x0: 78,
    y0: 26,
    rows: 4,
    cols: 6,
    values: state.noiseWeights,
    mode: 1,
    isMaskOrScore: false,
    explicitScale: 1.0,
  });
  add({
    id: 'stratified_order',
    x0: 86,
    y0: 26,
    rows: 4,
    cols: 4,
    values: state.stratifiedRankNormalized,
    mode: 1,
    isMaskOrScore: false,
    explicitScale: 1.0,
  });

  const regions = Object.freeze(entries.map((e) => e.region));
  const audit = auditAtlasRegions(W, H, regions, UNI_REFINEMENT_ATLAS_BUDGET_BYTES);
  const atlas: Rg32fValueAtlas = Object.freeze({
    width: W,
    height: H,
    data: writer.data,
  });
  const atlasKey = computeAtlasContentKey('refinement', atlas, extraKeyTag);

  return {
    atlas,
    atlasKey,
    regions,
    audit,
    slices: Object.freeze(sliceMap),
  };
}

// ============================================================================
// MAIN VIEW 5 SCENE BUILDER (`buildUniRefinementCanvasScene`)
// ============================================================================

const SHEET_Y = 0.08;
const SHEET_EY = 0.07;
const RIBBON_Y = 0.13;
const FRAME_LINE_Y = 0.01;

const CELL_HIGHLIGHT_FILL: FlowRibbonColor = {
  r: 0.863,
  g: 0.298,
  b: 0.204,
  a: 0.22,
  borderAlpha: 0.92,
};

const MARKER_COL_FILL: FlowRibbonColor = {
  r: 0.82,
  g: 0.62,
  b: 0.18,
  a: 0.14,
  borderAlpha: 0.78,
};

const STRATIFIED_FILL: FlowRibbonColor = {
  r: 0.102,
  g: 0.541,
  b: 0.478,
  a: 0.22,
  borderAlpha: 0.88,
};

function mergeAabb(
  target: { min: [number, number, number]; max: [number, number, number] },
  box: AABB,
): void {
  for (let i = 0; i < 3; i++) {
    if (box.min[i]! < target.min[i]!) target.min[i] = box.min[i]!;
    if (box.max[i]! > target.max[i]!) target.max[i] = box.max[i]!;
  }
}

interface RefinementPlateLayout {
  readonly id: UniRefinementStageId;
  readonly cx: number;
  readonly cz: number;
  readonly ex: number;
  readonly ez: number;
  readonly sheetKind: number;
  readonly role: number;
  readonly baseHighlight: number;
  readonly groupIndex?: number;
  readonly semanticTag: string;
  readonly label: string;
  readonly sublabel: string;
  readonly badge?: string;
}

/**
 * Build the headless 3D spatial scene for **View 5 (`refinement`) — Image Refinement Canvas**.
 */
export function buildUniRefinementCanvasScene(
  session: UniSession,
  options?: UniRefinementSceneOptions,
): UniSpatialScene {
  const ctx = resolveUniSessionContext(session, options?.recordId);
  const state = resolveRefinementDataState(ctx);
  const showStratified = options?.showStratifiedOrder ?? true;
  const focusedCfgRow =
    options?.cfgBranchRow !== undefined
      ? Math.max(0, Math.min(ctx.cfgBatchSize - 1, Math.trunc(options.cfgBranchRow)))
      : ctx.cfgBatchSize - 1;
  const selectedGridCell =
    options?.selectedGridCell !== undefined && options.selectedGridCell !== null
      ? Math.max(0, Math.min(23, Math.trunc(options.selectedGridCell)))
      : null;

  const extraKeyTag = `${ctx.record.recordId}:N${state.refinementSteps}:r${state.roundIndex ?? -1}:${state.isFallback ? 'fb' : 'live'}`;
  const packed = packRefinementAtlas(state, extraKeyTag);

  const solids = new UniSolidBufferBuilder();
  const lines = new UniLineBufferBuilder();
  const ribbons = new UniRibbonBufferBuilder();
  const sheets = new UniSheetBufferBuilder();
  const flow = new UniFlatFlowBufferBuilder();

  const pickBoxes: PickBox[] = [];
  const overlayAnchors: UniOverlayAnchor[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const boundsAcc: { min: [number, number, number]; max: [number, number, number] } = {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
  };

  // Workbench drafting zone frames
  const gStyle = state.isFallback ? LINE_STYLE_GUIDE : LINE_STYLE_ACTIVE;
  lines.addBoxFrameXZ(-6.6, -7.8, 6.6, -1.6, FRAME_LINE_Y, 1.0, gStyle, 0.52); // 4x6 refinement canvas bay
  lines.addBoxFrameXZ(-6.4, -1.2, 6.4, 2.2, FRAME_LINE_Y, 1.0, LINE_STYLE_GUIDE, 0.45); // 2x2 Codec + parent/prefix bay
  lines.addBoxFrameXZ(-6.8, 2.6, 6.8, 8.6, FRAME_LINE_Y, 1.0, LINE_STYLE_GUIDE, 0.50); // CFG Logits + Decoded bay

  // Decoded 16x16 RGB docking plinth (D4: RGB rendered via 2D <canvas> pixelCanvas.ts)
  const rgbDockX = 4.9;
  const rgbDockZ = 7.4;
  lines.addDashedFrameXZ(rgbDockX - 1.3, rgbDockZ - 0.9, rgbDockX + 1.3, rgbDockZ + 0.9, FRAME_LINE_Y + 0.01, gStyle, 0.65);
  solids.push({
    cx: rgbDockX,
    cy: 0.04,
    cz: rgbDockZ,
    ex: 2.2,
    ey: 0.08,
    ez: 1.4,
    cellFamily: state.isFallback ? CELL_EMBED_SLATE : CELL_ACTIVATION_GREEN,
    bayId: 0,
    highlight: 0,
  });
  lines.addBoxEdges(rgbDockX, 0.04, rgbDockZ, 2.2, 0.08, 1.4, 1, 0.9, LINE_STYLE_NEUTRAL, 0.65);

  mergeAabb(boundsAcc, { min: [-7.0, -0.1, -8.0], max: [7.0, 0.9, 8.8] });

  const cfgHl = (row: number) =>
    state.isFallback
      ? SHEET_HL_DIMMED
      : row === focusedCfgRow
        ? SHEET_HL_ACTIVE
        : row < ctx.cfgBatchSize
          ? SHEET_HL_OPERAND_B
          : SHEET_HL_DIMMED;

  const baseG = state.isFallback ? SHEET_HL_DIMMED : SHEET_HL_OPERAND_B;

  const plateLayouts: readonly RefinementPlateLayout[] = [
    // Centerpiece 4x6 VQ Child Plates + 4x4 Stratified Order Plate
    {
      id: 'vq_code_stage',
      cx: -4.2,
      cz: -5.4,
      ex: 3.0,
      ez: 2.0,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SOFTMAX,
      baseHighlight: state.isFallback ? SHEET_HL_DIMMED : SHEET_HL_ACTIVE,
      semanticTag: `branch=g;stage=code_stage;n=${state.refinementSteps};round=${state.roundIndex ?? -1}`,
      label: `4×6 Marked VQ Code Stage (N=${state.refinementSteps})`,
      sublabel: 'stage ∈ {-1 (hatched G=1), 0..N} · 4×4 content + 2 marker cols',
      badge: state.roundLabel,
    },
    {
      id: 'vq_paper_age',
      cx: -0.9,
      cz: -5.4,
      ex: 2.6,
      ez: 1.8,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_NORM,
      baseHighlight: baseG,
      semanticTag: 'branch=g;stage=paper_age;u9=reversed_polarity',
      label: 'U9 Paper Age a_paper = N - max(0, stage)',
      sublabel: 'Reversed polarity side-by-side (G = 0.0 active scalar)',
      badge: 'U9 Dual Polarity',
    },
    {
      id: 'vq_tree_level',
      cx: 2.1,
      cz: -5.4,
      ex: 2.6,
      ez: 1.8,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      baseHighlight: baseG,
      semanticTag: 'branch=g;stage=tree_level;ball_size=2^(8-l)',
      label: 'Tree Level ℓ & Candidate Ball 2^(8−ℓ)',
      sublabel: 'N=4 -> balls [256, 64, 16, 4, 1] across depth-8 BalancedSemanticTree',
    },
    {
      id: 'vq_mask_noise',
      cx: 5.1,
      cz: -5.4,
      ex: 2.6,
      ez: 1.8,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      baseHighlight: baseG,
      semanticTag: 'branch=g;stage=mask_noise;u10=additive_eta_mask',
      label: 'U10 Additive Mask Noise η = 1 − max(0,stage)/N',
      sublabel: 'Weights η_i · e(MASK_257) added continuously (G = 0.0)',
      badge: 'U10 η·e(MASK)',
    },
    {
      id: 'stratified_order',
      cx: -4.4,
      cz: -2.8,
      ex: 2.0,
      ez: 1.8,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      baseHighlight: showStratified ? SHEET_HL_OPERAND_A : SHEET_HL_DIMMED,
      semanticTag: `branch=g;stage=stratified_order;seed=42;visible=${showStratified ? 1 : 0}`,
      label: 'stratified_random(4, seed=42) Order (4×4)',
      sublabel: 'Quadtree-balanced spatial activation permutation (U12)',
      badge: showStratified ? 'Order Overlay ON' : 'Order Overlay OFF',
    },
    {
      id: 'vq_tree_embeddings',
      cx: 1.2,
      cz: -2.8,
      ex: 3.4,
      ez: 1.9,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_RESIDUAL,
      baseHighlight: baseG,
      semanticTag: 'branch=g;stage=u10_tree_embeddings;op=e_u_plus_eta_e_mask',
      label: 'U10 Input Embeddings e(u_i) + η·e(MASK) (24×8)',
      sublabel: 'Continuous 8D vectors entering 2×2 PackedLinearImageCodec',
    },

    // Middle Coupled Codec Layer: 2x3 = 6 Parent Tokens + 4 Fixed-Prefix Tokens
    {
      id: 'fixed_prefix_tokens',
      cx: -3.8,
      cz: 0.5,
      ex: 2.4,
      ez: 1.4,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_NEUTRAL,
      baseHighlight: state.isFallback ? SHEET_HL_DIMMED : SHEET_HL_PATHWAY,
      semanticTag: 'branch=g;stage=fixed_prefix;tokens=259_to_262;u13=4_prefix',
      label: 'Fixed-Prefix Tokens 259..262 (4×8)',
      sublabel: 'Uncompressed G conditioning prefix prepended before 6 parents (U13)',
      badge: '4 Fixed Prefix',
    },
    {
      id: 'parent_tokens',
      cx: 1.4,
      cz: 0.5,
      ex: 3.6,
      ez: 1.7,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_RESIDUAL,
      baseHighlight: baseG,
      semanticTag: 'branch=g;stage=parent_tokens;codec=2x2_packed;grid=2x3',
      label: '2×3 Parent Transformer Tokens (6×8)',
      sublabel: 'Coupled to 4×6 marked VQ children via 2×2 PackedLinearImageCodec',
      badge: '2×2 Codec Coupling',
    },

    // Bottom CFG Logits Comparison Band
    {
      id: 'cfg_branch_0_logits',
      cx: -4.8,
      cz: 4.3,
      ex: 2.8,
      ez: 2.1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      baseHighlight: cfgHl(0),
      groupIndex: 0,
      semanticTag: 'branch=g;groupRole=cfg;cfg=0;stage=cfg_logits_0',
      label: `${state.branchLabels[0]} (24×32)`,
      sublabel: 'Top-32 VQ logit slice from resolveStepHeadLogits() (G = 0.0)',
      badge: 'cfg=0',
    },
    {
      id: 'cfg_branch_1_logits',
      cx: -1.4,
      cz: 4.3,
      ex: 2.8,
      ez: 2.1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      baseHighlight: cfgHl(1),
      groupIndex: 1,
      semanticTag: 'branch=g;groupRole=cfg;cfg=1;stage=cfg_logits_1',
      label: `${state.branchLabels[1]} (24×32)`,
      sublabel: 'Top-32 VQ logit slice from resolveStepHeadLogits() (G = 0.0)',
      badge: 'cfg=1',
    },
    {
      id: 'cfg_branch_2_logits',
      cx: 2.0,
      cz: 4.3,
      ex: 2.8,
      ez: 2.1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      baseHighlight: cfgHl(2),
      groupIndex: 2,
      semanticTag: 'branch=g;groupRole=cfg;cfg=2;stage=cfg_logits_2',
      label: `${state.branchLabels[2]} (24×32)`,
      sublabel: 'Top-32 VQ logit slice from resolveStepHeadLogits() (G = 0.0)',
      badge: 'cfg=2',
    },
    {
      id: 'cfg_guided_logits',
      cx: -1.4,
      cz: 7.2,
      ex: 3.8,
      ez: 1.9,
      sheetKind: SHEET_KIND_CAUSAL_SCORES, // U25 live RG32F bound!
      role: SHEET_ROLE_SCORES,
      baseHighlight: state.isFallback ? SHEET_HL_DIMMED : SHEET_HL_ACTIVE,
      semanticTag: `branch=g;stage=guided_gen_logits;cfg_batch=${ctx.cfgBatchSize};u25=live_rg32f`,
      label: `Guided Gen Logits Top-32 (${ctx.cfgBatchSize}-Branch CFG)`,
      sublabel: 'Mixed CFG distribution -> categorical proposal & representative draw',
      badge: 'U25 Live RG32F',
    },
  ];

  const hovered = options?.hoveredCell ?? null;
  const stageFrames = {} as Record<UniRefinementStageId, SheetFrame3D>;

  for (let i = 0; i < plateLayouts.length; i++) {
    const spec = plateLayouts[i]!;
    const sliceEntry = packed.slices[spec.id];
    const gridCols = sliceEntry.region.cols;
    const gridRows = sliceEntry.region.rows;

    const frame = makeSheetFrame3D(
      [spec.cx, SHEET_Y, spec.cz],
      [spec.ex, SHEET_EY, spec.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    stageFrames[spec.id] = frame;
    mergeAabb(boundsAcc, sheetFrameWorldAabb(frame));

    const isHovered =
      hovered !== null && (hovered.tensorKey === spec.id || hovered.partId === spec.id);
    const highlight = isHovered ? SHEET_HL_ACTIVE : spec.baseHighlight;

    sheets.push({
      cx: spec.cx,
      cy: SHEET_Y,
      cz: spec.cz,
      ex: spec.ex,
      ey: SHEET_EY,
      ez: spec.ez,
      gridCols,
      gridRows,
      sheetKind: spec.sheetKind,
      role: spec.role,
      fp8: 0,
      highlight,
      seed: i * 0.63,
      slice: sliceEntry.slice,
      frame,
    });

    // +2 Structural marker column framing on 4x6 plates (cols 4..5)
    if (gridRows === 4 && gridCols === 6) {
      flow.addCellHighlightOnFrame(
        frame,
        0,
        3,
        4,
        5,
        4,
        6,
        SHEET_EY * 0.5 + 0.008,
        MARKER_COL_FILL,
        UNI_FLOW_COLORS.wasteAmber,
        0.016,
        true,
      );
    }

    // Optional stratified_random(4, seed=42) quadtree / activated cell overlay
    if (showStratified && spec.id === 'stratified_order') {
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          if (state.activatedContentMask[r * 4 + c] === 1) {
            flow.addCellHighlightOnFrame(
              frame,
              r,
              r,
              c,
              c,
              4,
              4,
              SHEET_EY * 0.5 + 0.012,
              STRATIFIED_FILL,
              UNI_FLOW_COLORS.gTeal,
              0.018,
              true,
            );
          }
        }
      }
    }

    // Selected grid cell highlight across 4x6 plates, 24x8 embeddings, 6x8 parents, and CFG logits
    if (selectedGridCell !== null) {
      const selR = Math.floor(selectedGridCell / 6);
      const selC = selectedGridCell % 6;
      if (gridRows === 4 && gridCols === 6) {
        flow.addCellHighlightOnFrame(
          frame,
          selR,
          selR,
          selC,
          selC,
          4,
          6,
          SHEET_EY * 0.5 + 0.016,
          CELL_HIGHLIGHT_FILL,
          UNI_FLOW_COLORS.activeOxide,
          0.024,
          true,
        );
      } else if (gridRows === 24) {
        flow.addCellHighlightOnFrame(
          frame,
          selectedGridCell,
          selectedGridCell,
          0,
          gridCols - 1,
          24,
          gridCols,
          SHEET_EY * 0.5 + 0.014,
          CELL_HIGHLIGHT_FILL,
          UNI_FLOW_COLORS.activeOxide,
          0.02,
          true,
        );
      } else if (spec.id === 'parent_tokens') {
        const parentRow = Math.floor(selR / 2) * 3 + Math.floor(selC / 2);
        flow.addCellHighlightOnFrame(
          frame,
          parentRow,
          parentRow,
          0,
          7,
          6,
          8,
          SHEET_EY * 0.5 + 0.014,
          CELL_HIGHLIGHT_FILL,
          UNI_FLOW_COLORS.activeOxide,
          0.022,
          true,
        );
      }
    }

    // Hover cell highlight
    if (
      isHovered &&
      hovered !== null &&
      hovered.row !== null &&
      hovered.col !== null &&
      hovered.row >= 0 &&
      hovered.row < gridRows &&
      hovered.col >= 0 &&
      hovered.col < gridCols
    ) {
      flow.addCellHighlightOnFrame(
        frame,
        hovered.row,
        hovered.row,
        hovered.col,
        hovered.col,
        gridRows,
        gridCols,
        SHEET_EY * 0.5 + 0.018,
        CELL_HIGHLIGHT_FILL,
        UNI_FLOW_COLORS.activeOxide,
        0.025,
        true,
      );
    }

    pickBoxes.push(
      packPickBoxFromFrame(frame, {
        bayId: 0,
        partId: spec.id,
        tensorKey: spec.id,
        gridCols,
        gridRows,
        ...(spec.groupIndex !== undefined ? { groupIndex: spec.groupIndex } : {}),
        towerRole: 'denoiser',
        semanticTag: spec.semanticTag,
      }),
    );

    overlayAnchors.push(
      Object.freeze({
        id: `anchor_${spec.id}`,
        label: spec.label,
        sublabel: spec.sublabel,
        worldPos: [spec.cx, SHEET_Y + 0.22, spec.cz - spec.ez * 0.5 - 0.16] as Vec3,
        tone: state.isFallback ? 'muted' : spec.groupIndex !== undefined ? 'g_teal' : 'joint',
        ...(spec.badge !== undefined ? { badge: spec.badge } : {}),
      }),
    );
  }

  // 2x2 PackedLinearImageCodec coupling brackets & cubic ribbons (4x6 children -> 2x3 parents)
  const ribbonColor = state.isFallback ? UNI_FLOW_COLORS.ghostSlate : UNI_FLOW_COLORS.gTeal;
  const codeFrame = stageFrames.vq_code_stage;
  const parentFrame = stageFrames.parent_tokens;
  const xz = (x: number, z: number) => ({ x, z });

  for (let pr = 0; pr < 2; pr++) {
    for (let pc = 0; pc < 3; pc++) {
      const childCx =
        codeFrame.center[0] + (((pc * 2 + 1) / 6) - 0.5) * codeFrame.extent[0];
      const childCz =
        codeFrame.center[2] + (((pr * 2 + 1) / 4) - 0.5) * codeFrame.extent[2];
      const parentIdx = pr * 3 + pc;
      const parentCx = parentFrame.center[0] + (((parentIdx + 0.5) / 6) - 0.5) * parentFrame.extent[0] * 0.65;
      const parentTopZ = parentFrame.center[2] - parentFrame.extent[2] * 0.5 - 0.06;

      // Bracket frame on child 2x2 patch
      const bw = codeFrame.extent[0] / 3.2;
      const bh = codeFrame.extent[2] / 2.2;
      lines.addBoxFrameXZ(
        childCx - bw * 0.5,
        childCz - bh * 0.5,
        childCx + bw * 0.5,
        childCz + bh * 0.5,
        SHEET_Y + 0.045,
        1.1,
        LINE_STYLE_ACTIVE,
        state.isFallback ? 0.25 : 0.72,
      );

      flow.addCubicXZ(
        xz(childCx, codeFrame.center[2] + codeFrame.extent[2] * 0.5 + 0.05),
        xz(childCx, parentTopZ - 1.2),
        xz(parentCx, codeFrame.center[2] + codeFrame.extent[2] * 0.5 + 1.4),
        xz(parentCx, parentTopZ),
        RIBBON_Y,
        0.10,
        ribbonColor,
        true,
        20,
      );
    }
  }

  // Coupling from fixed prefix + parent tokens -> CFG branches -> Guided Logits -> RGB Decoded Plinth
  const parentExitZ = parentFrame.center[2] + parentFrame.extent[2] * 0.5 + 0.06;
  const prefixFrame = stageFrames.fixed_prefix_tokens;
  flow.addPolylineXZ(
    [
      xz(prefixFrame.center[0] + prefixFrame.extent[0] * 0.5 + 0.06, prefixFrame.center[2]),
      xz(parentFrame.center[0] - parentFrame.extent[0] * 0.5 - 0.08, parentFrame.center[2]),
    ],
    RIBBON_Y,
    0.16,
    UNI_FLOW_COLORS.jointCyan,
    true,
  );

  const cfgIds: readonly UniRefinementStageId[] = [
    'cfg_branch_0_logits',
    'cfg_branch_1_logits',
    'cfg_branch_2_logits',
  ];
  for (let b = 0; b < cfgIds.length; b++) {
    const f = stageFrames[cfgIds[b]!];
    const bColor =
      b < ctx.cfgBatchSize && !state.isFallback
        ? UNI_FLOW_COLORS.gTeal
        : UNI_FLOW_COLORS.ghostSlate;
    flow.addCubicXZ(
      xz(parentFrame.center[0], parentExitZ),
      xz(parentFrame.center[0], parentExitZ + 0.7),
      xz(f.center[0], f.center[2] - f.extent[2] * 0.5 - 0.8),
      xz(f.center[0], f.center[2] - f.extent[2] * 0.5 - 0.06),
      RIBBON_Y,
      0.13,
      bColor,
      true,
      20,
    );
    flow.addCubicXZ(
      xz(f.center[0], f.center[2] + f.extent[2] * 0.5 + 0.06),
      xz(f.center[0], stageFrames.cfg_guided_logits.center[2] - 0.6),
      xz(stageFrames.cfg_guided_logits.center[0], f.center[2] + f.extent[2] * 0.5 + 0.6),
      xz(
        stageFrames.cfg_guided_logits.center[0],
        stageFrames.cfg_guided_logits.center[2] -
          stageFrames.cfg_guided_logits.extent[2] * 0.5 -
          0.06,
      ),
      RIBBON_Y,
      0.13,
      bColor,
      true,
      20,
    );
  }

  flow.addPolylineXZ(
    [
      xz(
        stageFrames.cfg_guided_logits.center[0] +
          stageFrames.cfg_guided_logits.extent[0] * 0.5 +
          0.08,
        stageFrames.cfg_guided_logits.center[2],
      ),
      xz(rgbDockX - 1.25, rgbDockZ),
    ],
    RIBBON_Y,
    0.18,
    ribbonColor,
    true,
  );

  // Small 3D telemetry plinth next to selected cell
  if (selectedGridCell !== null) {
    const pair = getStagePolarityPair(
      state.stages[selectedGridCell]!,
      UNI_MINI_CONFIG.vqTreeDepth,
      state.refinementSteps,
    );
    solids.push({
      cx: 4.9,
      cy: 0.06,
      cz: 0.5,
      ex: 1.8,
      ey: 0.12,
      ez: 1.0,
      cellFamily: pair.isCommitted ? CELL_ACTIVATION_GREEN : CELL_MOE_AMBER,
      bayId: 0,
      highlight: 1,
    });
    overlayAnchors.push(
      Object.freeze({
        id: 'selected_cell_readout',
        label: `Cell ${selectedGridCell} · u=${state.representatives[selectedGridCell]}`,
        sublabel: `stage=${pair.codeStage} · a_paper=${pair.paperAge} · ℓ=${pair.level} (ball=${pair.ballSize}) · η=${pair.noiseWeight.toFixed(2)}`,
        worldPos: [4.9, 0.35, 0.5] as Vec3,
        tone: 'gold',
        badge: pair.isCommitted ? 'Committed' : 'Refining',
      }),
    );
  } else {
    solids.push({
      cx: 4.9,
      cy: 0.05,
      cz: 0.5,
      ex: 1.8,
      ey: 0.10,
      ez: 1.0,
      cellFamily: CELL_SSM_CYAN,
      bayId: 0,
      highlight: 0,
    });
  }

  // Landmarks & Calipers
  landmarks.push(
    Object.freeze({
      id: 'lm_refinement_canvas',
      label: `4×6 Marked VQ Refinement Canvas (${state.roundLabel})`,
      sublabel: `${state.committedContentCount}/16 content committed · ${state.lockedCellCount}/16 U18 locked`,
      world: [-0.8, 0.48, -7.2] as Vec3,
      tone: state.isFallback ? 'muted' : 'cyan',
      align: 'top',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_decoded_preview',
      label: 'Decoded 16×16 RGB Preview (D4 2D Canvas)',
      sublabel: UNI_SCENE_VISUAL_HONESTY_BADGE,
      world: [rgbDockX, 0.45, rgbDockZ + 0.9] as Vec3,
      tone: 'gold',
      align: 'bottom',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'cal_4x6_marked',
      aWorld: [
        codeFrame.center[0] - codeFrame.extent[0] * 0.5,
        SHEET_Y + 0.05,
        codeFrame.center[2] - codeFrame.extent[2] * 0.5 - 0.12,
      ] as Vec3,
      bWorld: [
        codeFrame.center[0] + codeFrame.extent[0] * 0.5,
        SHEET_Y + 0.05,
        codeFrame.center[2] - codeFrame.extent[2] * 0.5 - 0.12,
      ] as Vec3,
      label: '4×6 (4×4 + 2 markers)',
      partIds: ['vq_code_stage'],
    }),
    Object.freeze({
      id: 'cal_6_parents',
      aWorld: [
        parentFrame.center[0] - parentFrame.extent[0] * 0.5,
        SHEET_Y + 0.05,
        parentFrame.center[2] + parentFrame.extent[2] * 0.5 + 0.12,
      ] as Vec3,
      bWorld: [
        parentFrame.center[0] + parentFrame.extent[0] * 0.5,
        SHEET_Y + 0.05,
        parentFrame.center[2] + parentFrame.extent[2] * 0.5 + 0.12,
      ] as Vec3,
      label: '2×3 = 6 parents (×8)',
      partIds: ['parent_tokens'],
    }),
  );

  // Synced 16x16 RGB `decoded` UniRgbPreviewDescriptor (D4: 0 RGB in RG32F)
  const rgbPreviews: readonly UniRgbPreviewDescriptor[] = Object.freeze([
    Object.freeze({
      id: 'refinement_decoded',
      kind: 'decoded' as const,
      label: `Decoded 16×16 RGB (${state.roundLabel})`,
      caption: state.isFallback
        ? 'Fallback Emu3VQ Decode from Initial Tree State'
        : `Emu3VQ Decode · ${state.committedContentCount}/16 Committed · ${state.lockedCellCount}/16 Locked`,
      width: state.decodedPaddedImage.width,
      height: state.decodedPaddedImage.height,
      rgbChw: new Float32Array(state.decodedPaddedImage.data),
      ...(ctx.branch.sourcePaddingMask !== undefined
        ? { paddingLockMask: ctx.branch.sourcePaddingMask }
        : {}),
      honestyBadge: UNI_SCENE_VISUAL_HONESTY_BADGE,
    }),
  ]);

  // Provenance badges
  const provenanceBadges: UniProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    Object.freeze({
      label: UNI_SCENE_VISUAL_HONESTY_BADGE,
      tone: 'muted' as const,
    }),
    Object.freeze({
      label: `U9: Reversed Polarity (stage ∈ {-1..${state.refinementSteps}} ↔ a_paper ∈ {0..${state.refinementSteps}})`,
      tone: 'g_teal' as const,
    }),
    Object.freeze({
      label: 'U10: Additive Mask Embedding e(u) + η·e(MASK)',
      tone: 'g_teal' as const,
    }),
    Object.freeze({
      label: `U12: stratified_random(4, seed=42) · ${ctx.cfgBatchSize}-Branch CFG`,
      tone: 'muted' as const,
    }),
  ];

  if (state.fallbackReason !== null) {
    provenanceBadges.push(
      Object.freeze({
        label: state.fallbackReason,
        tone: 'muted' as const,
      }),
    );
  }

  // Pack 5-pass GPU buffers
  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flow.pack();

  const fitBounds: AABB = Object.freeze({
    min: [boundsAcc.min[0] - 0.6, boundsAcc.min[1] - 0.2, boundsAcc.min[2] - 0.6] as Vec3,
    max: [boundsAcc.max[0] + 0.6, boundsAcc.max[1] + 1.2, boundsAcc.max[2] + 0.6] as Vec3,
  });

  const scene: UniSpatialScene = Object.freeze({
    viewMode: 'refinement' as const,
    title: 'View 5 · Image Refinement Canvas (BalancedSemanticTree & Dual-CFG)',
    subtitle: `${state.roundLabel} · ${packed.regions.length} RG32F rects (${packed.audit.utilizationPct.toFixed(1)}%)`,
    layerIndex: null,
    executionViewId: ctx.executionViewId,
    prefillMode: ctx.prefillMode,
    solidData: packedSolids.data,
    solidCount: packedSolids.count,
    lineData: packedLines.data,
    lineCount: packedLines.count,
    ribbonData: packedRibbons.data,
    ribbonCount: packedRibbons.count,
    sheetData: packedSheets.sheetData,
    sheetBasisData: packedSheets.sheetBasisData,
    sheetRevealData: packedSheets.sheetRevealData,
    sheetCount: packedSheets.sheetCount,
    flowVertices: packedFlow.flowVertices,
    flowIndices: packedFlow.flowIndices,
    pickBoxes: Object.freeze(pickBoxes),
    cameraPreset: Object.freeze({
      azimuthDeg: 14,
      elevationDeg: 46,
      fitBounds,
    }),
    fitBounds,
    valueAtlas: packed.atlas,
    atlasKey: packed.atlasKey,
    atlasRegions: packed.regions,
    atlasAudit: packed.audit,
    rgbPreviews,
    provenanceBadges: Object.freeze(provenanceBadges),
    overlayAnchors: Object.freeze(overlayAnchors),
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze([]),
    calipers: Object.freeze(calipers),
    d5Comparison: ctx.d5Comparison,
    viewReadout: ctx.viewReadout,
  });

  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `Invalid View 5 (refinement) scene geometry: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
