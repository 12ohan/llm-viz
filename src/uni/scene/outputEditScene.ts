/**
 * Milestone UNI5 — View 6 (`output_edit`): Task-Reactive Output, Grounding & Editing Scene
 * & `128 × 128` `RG32F` Scalar/Mask Atlas (`src/uni/scene/outputEditScene.ts`)
 *
 * Clean-room headless 3D spatial scene builder for Nemotron-Labs-Diffusion-Uni:
 *   - **T2I / Image Editing mode:**
 *       1. `PackedLinearImageCodec.expand` (`24×8`, `G = 0.0`) -> `gen_predictor` top-32 logits
 *          (`24×32`, `G = 0.0`) -> strip `+2` markers (`16×32`, `G = 0.0`)
 *       2. `4×4` `U18` `paddingLockMask` plate (`G = 1.0` on locked border cells, `G = 0.0` on
 *          editable interior cells) + `16×16` scalar luma/decoder activation plate (`G = 0.0`)
 *       3. 3-branch `legacy` / `independent` / `ip2p` Dual-CFG mixer comparison slices (`24×32`,
 *          `G = 0.0`, computed live from `ctx.resolveStepHeadLogits()`, with `groupIndex: branchRow`,
 *          `semanticTag` carrying `groupRole=cfg;cfg=${branchRow}` for `pickHitToUniCell`)
 *       4. Typed `D4` `UniRgbPreviewDescriptor`s (`0` RGB in `RG32F`): `decoded` (`16×16`),
 *          plus in Image Editing: `source`, `letterboxed` (with `paddingLockMask`), and `cropped`
 *          (`16×8` on `harbor_panorama_16x8` when `cropToSource !== false` via `cropPaddedImageToSource`)
 *   - **Understanding / Grounding mode:**
 *       1. Shifted `diffusion_head` block-diffusion logits (`B = 4`, `shift_logits = true`, `G = 0.0`)
 *          and softmax probabilities (`G = 0.0`, `SHEET_KIND_CAUSAL_SOFTMAX` `U25`)
 *       2. Unmasking / confidence state plates (`G = 1.0` ONLY on still-masked `[MASK]` positions)
 *       3. On `object_grounding` (and `visual_understanding` when a source image exists), emits a
 *          `grounding` `UniRgbPreviewDescriptor` with real `<LOC_0..16>` `groundingBoxes` (`pixelBox`)
 *          and explicit **`U27`** provenance badge noting `skip_special_tokens=True` strips `<LOC_*>` tokens!
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import type { Vec3 } from '../../model/spec';
import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
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
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
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
  CELL_WEIGHT_BLUE,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../scene/spine';
import { softmaxRows } from '../attention';
import {
  compressPackedImageEmbeddings,
  cropPaddedImageToSource,
  expandPackedImageHiddenStates,
  stripGenerationRowMarkers,
} from '../codec';
import { embedUniGenTokens, embedUniTextTokens, runUniForwardPass } from '../forward';
import { cropUniKvCache, extractRowPastKvs } from '../kvCache';
import {
  applyEditDualCfgLogits,
  applyT2iCfgLogits,
  decodeMarkedCodesToImage,
} from '../imageSampler';
import type { UniSession } from '../session';
import { groundObjects, locBinsToPixelBox } from '../textSampler';
import { initSemanticTreeState } from '../tree';
import {
  UNI_MINI_LOC_END_ID,
  UNI_MINI_LOC_END_TOKEN_ID,
  UNI_MINI_LOC_START_ID,
  UNI_MINI_MASK_TOKEN_ID,
  UNI_MINI_UND_PAD_TOKEN_ID,
  UNI_MINI_VOCAB,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  type UniEditGuidanceMode,
  type UniGroundingBox,
  type UniRgbImage,
  type UniSourcePaddingVqMask,
} from '../types';
import {
  createUniScenePreset,
  fixOnePrompt,
  padToSquareAndResize,
  runMiniVisionPipeline,
  spliceVisualEmbeddingsIntoPrompt,
} from '../vision';
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
  UniAtlasRegionBox,
  UniOutputEditSceneOptions,
  UniOverlayAnchor,
  UniProvenanceBadge,
  UniResolvedSessionContext,
  UniRgbPreviewDescriptor,
  UniSpatialScene,
} from './types';

// ============================================================================
// CONSTANTS & TYPES
// ============================================================================

export const UNI_OUTPUT_EDIT_ATLAS_SIZE = 128 as const;
export const UNI_OUTPUT_EDIT_ATLAS_BUDGET_BYTES = 131_072 as const; // 128 * 128 * 8 B
export const UNI_OUTPUT_EDIT_TOP_LOGIT_COLS = 32 as const;

export type UniOutputEditGenStageId =
  | 'codec_expand_children'
  | 'gen_predictor_top32'
  | 'stripped_content_logits'
  | 'u18_padding_lock_mask'
  | 'decoded_luma_16x16'
  | 'cfg_mixer_legacy'
  | 'cfg_mixer_independent'
  | 'cfg_mixer_ip2p';

export type UniOutputEditUndStageId =
  | 'shifted_diffusion_logits'
  | 'shifted_diffusion_probs'
  | 'unmask_mask_state'
  | 'token_slot_mask_strip'
  | 'final_norm_hidden'
  | 'source_luma_sheet';

interface OutputEditAtlasSliceEntry {
  readonly id: string;
  readonly region: UniAtlasRegionBox;
  readonly slice: AtlasSlice;
}

// ============================================================================
// INTERNAL RG32F PACKING & NUMERIC HELPERS (NON-EXPORTED)
// ============================================================================

interface LocalOutputMatrixPackParams {
  readonly id: string;
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

function packOutputEditMatrixToRg32f(
  writer: Rg32fAtlasWriter,
  params: LocalOutputMatrixPackParams,
): OutputEditAtlasSliceEntry {
  const { id, x0, y0, rows, cols, values, mode, isMaskOrScore, gChannelFlags, explicitScale } =
    params;
  const cellCount = rows * cols;
  if (x0 < 0 || y0 < 0 || x0 + cols > writer.width || y0 + rows > writer.height) {
    throw new RangeError(
      `Output/Edit RG32F region "${id}" (${x0},${y0} ${cols}x${rows}) exceeds ${writer.width}x${writer.height}`,
    );
  }
  if (values.length < cellCount) {
    throw new RangeError(
      `Output/Edit RG32F region "${id}" values length (${values.length}) < ${cellCount}`,
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

function computeScalarLumaFromRgb(img: UniRgbImage): Float32Array {
  const hw = img.width * img.height;
  const luma = new Float32Array(hw);
  for (let i = 0; i < hw; i++) {
    const r = img.data[i]!;
    const g = img.data[hw + i]!;
    const b = img.data[2 * hw + i]!;
    luma[i] = Math.fround(0.299 * r + 0.587 * g + 0.114 * b);
  }
  return luma;
}

function selectTopVqColumns(
  referenceLogits: Float32Array,
  markedCodes: ArrayLike<number>,
  gridLen: number,
  vocabSize: number = UNI_MINI_VQ_CODEBOOK_SIZE,
  sliceCols: number = UNI_OUTPUT_EDIT_TOP_LOGIT_COLS,
): Int32Array {
  const chosen = new Set<number>();
  for (let i = 0; i < gridLen && chosen.size < sliceCols; i++) {
    const code = markedCodes[i]!;
    if (code >= 0 && code < vocabSize) {
      chosen.add(code);
    }
  }
  const peak = new Float32Array(vocabSize).fill(-Infinity);
  for (let i = 0; i < gridLen; i++) {
    const base = i * vocabSize;
    for (let k = 0; k < vocabSize; k++) {
      const v = referenceLogits[base + k]!;
      if (Number.isFinite(v) && v > peak[k]!) peak[k] = v;
    }
  }
  const ranked = new Array<number>(vocabSize);
  for (let k = 0; k < vocabSize; k++) ranked[k] = k;
  ranked.sort((a, b) => {
    const diff = peak[b]! - peak[a]!;
    return diff !== 0 ? diff : a - b;
  });
  for (let i = 0; i < ranked.length && chosen.size < sliceCols; i++) {
    chosen.add(ranked[i]!);
  }
  return Int32Array.from(Array.from(chosen).sort((a, b) => a - b).slice(0, sliceCols));
}

function sliceVqLogits(
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

function stripMarkedRowsFromLogits(
  markedLogits: Float32Array,
  rows: number,
  markedCols: number,
  contentCols: number,
  sliceCols: number,
): Float32Array {
  const out = new Float32Array(rows * contentCols * sliceCols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < contentCols; c++) {
      const srcBase = (r * markedCols + c) * sliceCols;
      const dstBase = (r * contentCols + c) * sliceCols;
      for (let k = 0; k < sliceCols; k++) {
        out[dstBase + k] = markedLogits[srcBase + k]!;
      }
    }
  }
  return out;
}

// ============================================================================
// GENERATION PIPELINE SCENE (T2I / IMAGE EDITING)
// ============================================================================

const SHEET_Y = 0.08;
const SHEET_EY = 0.07;
const RIBBON_Y = 0.13;
const FRAME_LINE_Y = 0.01;

const LOCK_BORDER_FILL: FlowRibbonColor = {
  r: 0.92,
  g: 0.58,
  b: 0.08,
  a: 0.24,
  borderAlpha: 0.92,
};

const MASKED_TEXT_FILL: FlowRibbonColor = {
  r: 0.56,
  g: 0.60,
  b: 0.67,
  a: 0.24,
  borderAlpha: 0.85,
};

const HOVER_FILL: FlowRibbonColor = {
  r: 0.863,
  g: 0.298,
  b: 0.204,
  a: 0.26,
  borderAlpha: 0.95,
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

function buildGenerationOutputEditScene(
  ctx: UniResolvedSessionContext,
  options?: UniOutputEditSceneOptions,
): UniSpatialScene {
  const { bundle, branch, record } = ctx;
  const core = bundle.core;
  const cfg = core.config;
  const d = cfg.hiddenSize; // 8
  const parentRows = cfg.packedGridRows; // 2
  const parentCols = cfg.packedGridCols; // 3
  const fixedPrefixCount = cfg.genFixedPrefixTokens; // 4
  const parentCount = parentRows * parentCols; // 6
  const gridRows = cfg.vqGridSize; // 4
  const markedCols = cfg.vqMarkedGridCols; // 6
  const gridLen = gridRows * markedCols; // 24
  const contentCount = gridRows * gridRows; // 16
  const vqVocab = cfg.vqCodebookSize; // 256

  const isEdit = branch.taskMode === 'image_edit';
  const activeMode: UniEditGuidanceMode =
    options?.editGuidanceMode ?? branch.guidanceMode ?? 'legacy';
  const showPaddingLockOverlay = options?.showPaddingLockOverlay ?? true;
  const cropToSource = options?.cropToSource ?? true;

  const hasDenoiseState =
    record.kind === 'image_denoise' && record.imageState !== undefined;

  const representatives = hasDenoiseState
    ? record.imageState!.representatives
    : initSemanticTreeState({
        refinementSteps: branch.refinementSteps,
        lockedContentMask: isEdit ? branch.sourcePaddingMask?.lockMask : undefined,
        lockedSourceCodes: isEdit ? branch.sourceVqCodes : undefined,
      }).representatives;

  const contentCodes = stripGenerationRowMarkers(representatives, gridRows, markedCols);

  // Live parent hidden states -> PackedLinearImageCodec.expand (24x8) -> gen_predictor (24x256)
  const headLogits = ctx.resolveStepHeadLogits();
  const focusBranchIdx = Math.max(0, ctx.cfgBatchSize - 1);
  const finalNorm = headLogits.perBranchFinalNorm[focusBranchIdx];

  let parentHidden: Float32Array;
  if (
    hasDenoiseState &&
    finalNorm !== null &&
    finalNorm.length >= (fixedPrefixCount + parentCount) * d
  ) {
    parentHidden = finalNorm.subarray(
      fixedPrefixCount * d,
      (fixedPrefixCount + parentCount) * d,
    );
  } else {
    const childEmb = embedUniGenTokens(representatives, core);
    parentHidden = compressPackedImageEmbeddings(
      childEmb,
      gridRows,
      markedCols,
      bundle.codec,
      { isSourceEdit: false, gainPresetId: ctx.gainPresetId },
    ).output;
  }

  const expanded = expandPackedImageHiddenStates(
    parentHidden,
    parentRows,
    parentCols,
    bundle.codec,
    core.genPredictor,
  );
  const expandedChildren = expanded.unpackedChildren; // [24, 8]
  const expandedLogits = expanded.logits!; // [24, 256]

  // Live 3-branch Dual-CFG mixer logits (`legacy`, `independent`, `ip2p`) or T2I CFG
  const wTxt = branch.guidanceScaleText;
  const wImg = branch.guidanceScaleImage;
  let mixerLegacyFull: Float32Array;
  let mixerIndependentFull: Float32Array;
  let mixerIp2pFull: Float32Array;
  let mixerLabels: readonly [string, string, string];

  if (isEdit && hasDenoiseState && ctx.cfgBatchSize === 3) {
    const uncond = headLogits.perBranchGenLogits[0] ?? expandedLogits;
    const vqDrop = headLogits.perBranchGenLogits[1] ?? expandedLogits;
    const full = headLogits.perBranchGenLogits[2] ?? expandedLogits;
    mixerLegacyFull = applyEditDualCfgLogits(uncond, vqDrop, full, 'legacy', wTxt, wImg);
    mixerIndependentFull = applyEditDualCfgLogits(uncond, vqDrop, full, 'independent', wTxt, wImg);
    mixerIp2pFull = applyEditDualCfgLogits(uncond, vqDrop, full, 'ip2p', wTxt, wImg);
    mixerLabels = [
      `Dual-CFG · legacy (w_txt=${wTxt}, w_img=${wImg})`,
      `Dual-CFG · independent (w_txt=${wTxt}, w_img=${wImg})`,
      `Dual-CFG · ip2p (w_txt=${wTxt}, w_img=${wImg})`,
    ];
  } else if (!isEdit && hasDenoiseState && ctx.cfgBatchSize >= 2) {
    const uncond = headLogits.perBranchGenLogits[0] ?? expandedLogits;
    const cond = headLogits.perBranchGenLogits[1] ?? expandedLogits;
    const guided = headLogits.guidedGenLogits ?? applyT2iCfgLogits(uncond, cond, wTxt);
    // Exact T2I guidance (wImg = 0): legacy and independent both reduce to uncond + wTxt * (cond - uncond)
    mixerLegacyFull = applyEditDualCfgLogits(uncond, uncond, cond, 'legacy', wTxt, 0);
    mixerIndependentFull = applyEditDualCfgLogits(uncond, uncond, cond, 'independent', wTxt, 0);
    mixerIp2pFull = guided;
    mixerLabels = [
      `T2I CFG · legacy (w_txt=${wTxt}, w_img=0)`,
      `T2I CFG · independent (w_txt=${wTxt}, w_img=0)`,
      `T2I CFG · Guided Output (w_txt=${wTxt})`,
    ];
  } else {
    mixerLegacyFull = expandedLogits;
    mixerIndependentFull = applyEditDualCfgLogits(
      expandedLogits,
      expandedLogits,
      expandedLogits,
      'independent',
      wTxt,
      wImg,
    );
    mixerIp2pFull = applyEditDualCfgLogits(
      expandedLogits,
      expandedLogits,
      expandedLogits,
      'ip2p',
      wTxt,
      wImg,
    );
    mixerLabels = [
      `Dual-CFG Sim · legacy`,
      `Dual-CFG Sim · independent`,
      `Dual-CFG Sim · ip2p`,
    ];
  }

  const activeGuidedFull =
    activeMode === 'legacy'
      ? mixerLegacyFull
      : activeMode === 'independent'
        ? mixerIndependentFull
        : mixerIp2pFull;

  const selectedCols = selectTopVqColumns(
    activeGuidedFull,
    representatives,
    gridLen,
    vqVocab,
    UNI_OUTPUT_EDIT_TOP_LOGIT_COLS,
  );
  const genPredictorTop32 = sliceVqLogits(expandedLogits, selectedCols, gridLen, vqVocab);
  const strippedContentLogits = stripMarkedRowsFromLogits(
    genPredictorTop32,
    gridRows,
    markedCols,
    gridRows,
    UNI_OUTPUT_EDIT_TOP_LOGIT_COLS,
  );
  const cfgMixerLegacySlice = sliceVqLogits(mixerLegacyFull, selectedCols, gridLen, vqVocab);
  const cfgMixerIndependentSlice = sliceVqLogits(
    mixerIndependentFull,
    selectedCols,
    gridLen,
    vqVocab,
  );
  const cfgMixerIp2pSlice = sliceVqLogits(mixerIp2pFull, selectedCols, gridLen, vqVocab);

  // Source / letterbox / decoded / cropped RGB previews (D4: 0 RGB in RG32F)
  let sourceImage: UniRgbImage | undefined;
  let letterboxResult: ReturnType<typeof padToSquareAndResize> | undefined;
  let paddingLockMask: UniSourcePaddingVqMask | undefined = branch.sourcePaddingMask;

  if (isEdit) {
    sourceImage = createUniScenePreset(branch.scenePresetId ?? 'harbor_panorama_16x8');
    letterboxResult = padToSquareAndResize(sourceImage, cfg.imageResolution);
    if (!paddingLockMask) {
      paddingLockMask = {
        vqGridRows: gridRows,
        vqGridCols: gridRows,
        letterbox: letterboxResult.info,
        lockMask: new Uint8Array(contentCount),
        lockedCellCount: 0,
        editableCellCount: contentCount,
      };
    }
  }

  const decodedResult = hasDenoiseState
    ? ctx.session.decodeBranchImage(branch.branchId, record.recordId)
    : decodeMarkedCodesToImage(
        bundle,
        representatives,
        letterboxResult?.info,
        cropToSource,
      );
  const decodedPaddedImage = decodedResult.decodedPaddedImage;
  const decodedLuma16x16 = computeScalarLumaFromRgb(decodedPaddedImage);

  const lockGridR = new Float32Array(contentCount);
  const lockGridG = new Float32Array(contentCount);
  for (let i = 0; i < contentCount; i++) {
    lockGridR[i] = contentCodes[i]! / 255.0;
    // CRITICAL SheetsPass rule: G = 1.0 ONLY on U18 locked border cells!
    lockGridG[i] = paddingLockMask && paddingLockMask.lockMask[i] === 1 ? 1.0 : 0.0;
  }

  // Pack 128 x 128 RG32F scalar/mask atlas (8 rects <= 14, 0 overlaps)
  const W = UNI_OUTPUT_EDIT_ATLAS_SIZE;
  const H = UNI_OUTPUT_EDIT_ATLAS_SIZE;
  const writer = createRg32fAtlas(W, H);
  const entries: OutputEditAtlasSliceEntry[] = [];
  const sliceMap = {} as Record<UniOutputEditGenStageId, OutputEditAtlasSliceEntry>;
  const add = (params: LocalOutputMatrixPackParams) => {
    const entry = packOutputEditMatrixToRg32f(writer, params);
    entries.push(entry);
    sliceMap[params.id as UniOutputEditGenStageId] = entry;
  };

  add({
    id: 'codec_expand_children',
    x0: 0,
    y0: 0,
    rows: 24,
    cols: 8,
    values: expandedChildren,
    mode: 0,
    isMaskOrScore: false,
  });
  add({
    id: 'gen_predictor_top32',
    x0: 10,
    y0: 0,
    rows: 24,
    cols: 32,
    values: genPredictorTop32,
    mode: 0,
    isMaskOrScore: true,
  });
  add({
    id: 'stripped_content_logits',
    x0: 44,
    y0: 0,
    rows: 16,
    cols: 32,
    values: strippedContentLogits,
    mode: 0,
    isMaskOrScore: true,
  });
  add({
    id: 'u18_padding_lock_mask',
    x0: 78,
    y0: 0,
    rows: 4,
    cols: 4,
    values: lockGridR,
    mode: 1,
    isMaskOrScore: true,
    gChannelFlags: lockGridG,
    explicitScale: 1.0,
  });
  add({
    id: 'decoded_luma_16x16',
    x0: 84,
    y0: 0,
    rows: 16,
    cols: 16,
    values: decodedLuma16x16,
    mode: 1,
    isMaskOrScore: false,
    explicitScale: 1.0,
  });
  add({
    id: 'cfg_mixer_legacy',
    x0: 0,
    y0: 26,
    rows: 24,
    cols: 32,
    values: cfgMixerLegacySlice,
    mode: 0,
    isMaskOrScore: true,
  });
  add({
    id: 'cfg_mixer_independent',
    x0: 34,
    y0: 26,
    rows: 24,
    cols: 32,
    values: cfgMixerIndependentSlice,
    mode: 0,
    isMaskOrScore: true,
  });
  add({
    id: 'cfg_mixer_ip2p',
    x0: 68,
    y0: 26,
    rows: 24,
    cols: 32,
    values: cfgMixerIp2pSlice,
    mode: 0,
    isMaskOrScore: true,
  });

  const regions = Object.freeze(entries.map((e) => e.region));
  const audit = auditAtlasRegions(W, H, regions, UNI_OUTPUT_EDIT_ATLAS_BUDGET_BYTES);
  const atlas: Rg32fValueAtlas = Object.freeze({ width: W, height: H, data: writer.data });
  const atlasKey = computeAtlasContentKey(
    'output_edit',
    atlas,
    `${record.recordId}:${activeMode}:${cropToSource ? 'crop' : 'pad'}`,
  );

  // Build 5-pass 3D geometry
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

  lines.addBoxFrameXZ(-6.6, -7.2, 6.6, -1.6, FRAME_LINE_Y, 1.0, LINE_STYLE_ACTIVE, 0.5);
  lines.addBoxFrameXZ(-6.8, -1.0, 6.8, 3.0, FRAME_LINE_Y, 1.0, LINE_STYLE_GUIDE, 0.48);
  lines.addBoxFrameXZ(-6.4, 3.4, 6.4, 8.2, FRAME_LINE_Y, 1.0, LINE_STYLE_GUIDE, 0.52);

  // RGB preview plinths (D4)
  const dockZ = 7.0;
  solids.push({
    cx: 3.4,
    cy: 0.04,
    cz: dockZ,
    ex: 4.8,
    ey: 0.08,
    ez: 1.5,
    cellFamily: CELL_ACTIVATION_GREEN,
    bayId: 0,
    highlight: 0,
  });
  lines.addDashedFrameXZ(0.8, dockZ - 0.9, 6.0, dockZ + 0.9, FRAME_LINE_Y + 0.01, LINE_STYLE_ACTIVE, 0.65);
  mergeAabb(boundsAcc, { min: [-7.0, -0.1, -7.6], max: [7.0, 0.9, 8.4] });

  interface GenPlateSpec {
    readonly id: UniOutputEditGenStageId;
    readonly cx: number;
    readonly cz: number;
    readonly ex: number;
    readonly ez: number;
    readonly sheetKind: number;
    readonly role: number;
    readonly highlight: number;
    readonly groupIndex?: number;
    readonly semanticTag: string;
    readonly label: string;
    readonly sublabel: string;
    readonly badge?: string;
  }

  const plateSpecs: readonly GenPlateSpec[] = [
    {
      id: 'codec_expand_children',
      cx: -4.8,
      cz: -4.6,
      ex: 2.2,
      ez: 2.2,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_RESIDUAL,
      highlight: SHEET_HL_OPERAND_B,
      semanticTag: 'branch=g;stage=codec_expand;op=2x2_raster_unpack',
      label: 'PackedLinearImageCodec.expand (24×8)',
      sublabel: 'up_norm -> up_projection -> 2×2 raster unpack (G = 0.0)',
    },
    {
      id: 'gen_predictor_top32',
      cx: -0.9,
      cz: -4.6,
      ex: 3.4,
      ez: 2.2,
      sheetKind: SHEET_KIND_CAUSAL_SCORES, // U25 live RG32F
      role: SHEET_ROLE_SCORES,
      highlight: SHEET_HL_OPERAND_B,
      semanticTag: 'branch=g;stage=gen_predictor_top32;u25=live_rg32f',
      label: 'gen_predictor Top-32 Logits (24×32)',
      sublabel: 'Includes 8 marker rows (cols 4..5) before +2 marker strip (G = 0.0)',
      badge: 'U25 Live RG32F',
    },
    {
      id: 'stripped_content_logits',
      cx: 3.8,
      cz: -4.6,
      ex: 3.4,
      ez: 1.9,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      highlight: SHEET_HL_PATHWAY,
      semanticTag: 'branch=g;stage=strip_markers;grid=4x4x32',
      label: 'Stripped 4×4 Content Logits (16×32)',
      sublabel: 'stripGenerationRowMarkers removes +2 structural marker cols (G = 0.0)',
    },
    {
      id: 'cfg_mixer_legacy',
      cx: -4.4,
      cz: 0.9,
      ex: 3.0,
      ez: 2.1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      highlight: activeMode === 'legacy' ? SHEET_HL_ACTIVE : SHEET_HL_OPERAND_B,
      groupIndex: 0,
      semanticTag: 'branch=g;groupRole=cfg;cfg=0;mode=legacy',
      label: mixerLabels[0],
      sublabel: 'cond = (full + w_img*vq_drop)/(1+w_img); (1+w_txt)*cond - w_txt*uncond (G = 0.0)',
      badge: activeMode === 'legacy' ? 'Active · legacy' : 'cfg=0',
    },
    {
      id: 'cfg_mixer_independent',
      cx: 0.0,
      cz: 0.9,
      ex: 3.0,
      ez: 2.1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      highlight: activeMode === 'independent' ? SHEET_HL_ACTIVE : SHEET_HL_OPERAND_B,
      groupIndex: 1,
      semanticTag: 'branch=g;groupRole=cfg;cfg=1;mode=independent',
      label: mixerLabels[1],
      sublabel: 'uncond + w_txt*(full - uncond) + w_img*(full - vq_drop) (G = 0.0)',
      badge: activeMode === 'independent' ? 'Active · independent' : 'cfg=1',
    },
    {
      id: 'cfg_mixer_ip2p',
      cx: 4.4,
      cz: 0.9,
      ex: 3.0,
      ez: 2.1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      highlight: activeMode === 'ip2p' ? SHEET_HL_ACTIVE : SHEET_HL_OPERAND_B,
      groupIndex: 2,
      semanticTag: 'branch=g;groupRole=cfg;cfg=2;mode=ip2p',
      label: mixerLabels[2],
      sublabel: 'uncond + w_img*(vq_drop - uncond) + w_txt*(full - vq_drop) (G = 0.0)',
      badge: activeMode === 'ip2p' ? 'Active · ip2p' : 'cfg=2',
    },
    {
      id: 'u18_padding_lock_mask',
      cx: -4.2,
      cz: 5.6,
      ex: 2.2,
      ez: 2.2,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SOFTMAX,
      highlight: SHEET_HL_OPERAND_A,
      semanticTag: `branch=g;stage=u18_padding_lock;locked=${paddingLockMask?.lockedCellCount ?? 0}`,
      label: '4×4 U18 paddingLockMask Plate',
      sublabel: `R = code/255 · G = 1.0 on locked border (${paddingLockMask?.lockedCellCount ?? 0}/16 locked)`,
      badge: `U18: ${paddingLockMask?.lockedCellCount ?? 0}/16 Locked`,
    },
    {
      id: 'decoded_luma_16x16',
      cx: -0.6,
      cz: 5.6,
      ex: 2.8,
      ez: 2.8,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      highlight: SHEET_HL_ACTIVE,
      semanticTag: 'branch=g;stage=decoded_luma;op=emu3vq_decode_luma_16x16;d4=scalar_only',
      label: 'Emu3VQ Decoded Scalar Luma (16×16)',
      sublabel: 'Rec.601 scalar luma Y ∈ [0,1] (G = 0.0 active; RGB via 2D Canvas D4)',
      badge: 'Scalar Luma (G=0)',
    },
  ];

  const hovered = options?.hoveredCell ?? null;
  const stageFrames = {} as Record<UniOutputEditGenStageId, SheetFrame3D>;

  for (let i = 0; i < plateSpecs.length; i++) {
    const spec = plateSpecs[i]!;
    const sliceEntry = sliceMap[spec.id];
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
      highlight: isHovered ? SHEET_HL_ACTIVE : spec.highlight,
      seed: i * 0.57,
      slice: sliceEntry.slice,
      frame,
    });

    if (
      showPaddingLockOverlay &&
      spec.id === 'u18_padding_lock_mask' &&
      paddingLockMask !== undefined
    ) {
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) {
          if (paddingLockMask.lockMask[r * 4 + c] === 1) {
            flow.addCellHighlightOnFrame(
              frame,
              r,
              r,
              c,
              c,
              4,
              4,
              SHEET_EY * 0.5 + 0.012,
              LOCK_BORDER_FILL,
              UNI_FLOW_COLORS.wasteAmber,
              0.02,
              true,
            );
          }
        }
      }
    }

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
        SHEET_EY * 0.5 + 0.016,
        HOVER_FILL,
        UNI_FLOW_COLORS.activeOxide,
        0.024,
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
        tone: spec.id === 'u18_padding_lock_mask' ? 'gold' : 'g_teal',
        ...(spec.badge !== undefined ? { badge: spec.badge } : {}),
      }),
    );
  }

  // Flow ribbons connecting expand -> gen_predictor -> strip -> Dual-CFG mixers -> lock mask -> luma -> RGB dock
  const xz = (x: number, z: number) => ({ x, z });
  flow.addPolylineXZ(
    [
      xz(
        stageFrames.codec_expand_children.center[0] +
          stageFrames.codec_expand_children.extent[0] * 0.5 +
          0.06,
        -4.6,
      ),
      xz(
        stageFrames.gen_predictor_top32.center[0] -
          stageFrames.gen_predictor_top32.extent[0] * 0.5 -
          0.06,
        -4.6,
      ),
    ],
    RIBBON_Y,
    0.16,
    UNI_FLOW_COLORS.gTeal,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(
        stageFrames.gen_predictor_top32.center[0] +
          stageFrames.gen_predictor_top32.extent[0] * 0.5 +
          0.06,
        -4.6,
      ),
      xz(
        stageFrames.stripped_content_logits.center[0] -
          stageFrames.stripped_content_logits.extent[0] * 0.5 -
          0.06,
        -4.6,
      ),
    ],
    RIBBON_Y,
    0.16,
    UNI_FLOW_COLORS.gTeal,
    true,
  );

  const activeMixerId: UniOutputEditGenStageId =
    activeMode === 'legacy'
      ? 'cfg_mixer_legacy'
      : activeMode === 'independent'
        ? 'cfg_mixer_independent'
        : 'cfg_mixer_ip2p';
  const activeMixerFrame = stageFrames[activeMixerId];

  flow.addCubicXZ(
    xz(
      stageFrames.stripped_content_logits.center[0],
      stageFrames.stripped_content_logits.center[2] +
        stageFrames.stripped_content_logits.extent[2] * 0.5 +
        0.06,
    ),
    xz(stageFrames.stripped_content_logits.center[0], -1.8),
    xz(activeMixerFrame.center[0], -1.8),
    xz(
      activeMixerFrame.center[0],
      activeMixerFrame.center[2] - activeMixerFrame.extent[2] * 0.5 - 0.06,
    ),
    RIBBON_Y,
    0.16,
    UNI_FLOW_COLORS.gTeal,
    true,
    20,
  );
  flow.addCubicXZ(
    xz(
      activeMixerFrame.center[0],
      activeMixerFrame.center[2] + activeMixerFrame.extent[2] * 0.5 + 0.06,
    ),
    xz(activeMixerFrame.center[0], 3.1),
    xz(stageFrames.decoded_luma_16x16.center[0], 3.1),
    xz(
      stageFrames.decoded_luma_16x16.center[0],
      stageFrames.decoded_luma_16x16.center[2] -
        stageFrames.decoded_luma_16x16.extent[2] * 0.5 -
        0.06,
    ),
    RIBBON_Y,
    0.18,
    UNI_FLOW_COLORS.gTeal,
    true,
    20,
  );
  flow.addPolylineXZ(
    [
      xz(
        stageFrames.u18_padding_lock_mask.center[0] +
          stageFrames.u18_padding_lock_mask.extent[0] * 0.5 +
          0.06,
        5.6,
      ),
      xz(
        stageFrames.decoded_luma_16x16.center[0] -
          stageFrames.decoded_luma_16x16.extent[0] * 0.5 -
          0.06,
        5.6,
      ),
    ],
    RIBBON_Y,
    0.15,
    UNI_FLOW_COLORS.wasteAmber,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(
        stageFrames.decoded_luma_16x16.center[0] +
          stageFrames.decoded_luma_16x16.extent[0] * 0.5 +
          0.06,
        5.9,
      ),
      xz(1.2, dockZ - 0.2),
    ],
    RIBBON_Y,
    0.18,
    UNI_FLOW_COLORS.gTeal,
    true,
  );

  // Typed 3-channel RGB Previews (D4: 0 RGB in RG32F)
  const rgbPreviews: UniRgbPreviewDescriptor[] = [];
  if (isEdit && sourceImage !== undefined && letterboxResult !== undefined) {
    rgbPreviews.push(
      Object.freeze({
        id: 'output_edit_source',
        kind: 'source' as const,
        label: sourceImage.label ?? `Source RGB (${sourceImage.width}×${sourceImage.height})`,
        caption: `Native ${sourceImage.width}×${sourceImage.height} Source Input`,
        width: sourceImage.width,
        height: sourceImage.height,
        rgbChw: new Float32Array(sourceImage.data),
        honestyBadge: UNI_SCENE_VISUAL_HONESTY_BADGE,
      }),
      Object.freeze({
        id: 'output_edit_letterboxed',
        kind: 'letterboxed' as const,
        label: `Letterboxed ${letterboxResult.image.width}×${letterboxResult.image.height}`,
        caption: `U18 paddingLockMask: ${paddingLockMask?.lockedCellCount ?? 0}/16 border cells locked`,
        width: letterboxResult.image.width,
        height: letterboxResult.image.height,
        rgbChw: new Float32Array(letterboxResult.image.data),
        ...(paddingLockMask !== undefined ? { paddingLockMask } : {}),
        honestyBadge: UNI_SCENE_VISUAL_HONESTY_BADGE,
      }),
    );
  }

  rgbPreviews.push(
    Object.freeze({
      id: 'output_edit_decoded',
      kind: 'decoded' as const,
      label: `Decoded ${decodedPaddedImage.width}×${decodedPaddedImage.height} RGB`,
      caption: 'Emu3VQ Decoder Output (16×16 padded canvas)',
      width: decodedPaddedImage.width,
      height: decodedPaddedImage.height,
      rgbChw: new Float32Array(decodedPaddedImage.data),
      ...(paddingLockMask !== undefined ? { paddingLockMask } : {}),
      honestyBadge: UNI_SCENE_VISUAL_HONESTY_BADGE,
    }),
  );

  if (isEdit && cropToSource && letterboxResult !== undefined) {
    const croppedImg = cropPaddedImageToSource(decodedPaddedImage, letterboxResult.info);
    rgbPreviews.push(
      Object.freeze({
        id: 'output_edit_cropped',
        kind: 'cropped' as const,
        label: `Cropped Output (${croppedImg.width}×${croppedImg.height})`,
        caption: `U18 center-cropped back to native ${croppedImg.width}×${croppedImg.height} via cropPaddedImageToSource`,
        width: croppedImg.width,
        height: croppedImg.height,
        rgbChw: new Float32Array(croppedImg.data),
        honestyBadge: UNI_SCENE_VISUAL_HONESTY_BADGE,
      }),
    );
  }

  landmarks.push(
    Object.freeze({
      id: 'lm_output_edit_rgb_bay',
      label: isEdit
        ? `D4 RGB Previews (${cropToSource ? 'source / letterboxed / decoded / cropped' : 'source / letterboxed / decoded'})`
        : 'D4 Decoded 16×16 RGB Preview',
      sublabel: UNI_SCENE_VISUAL_HONESTY_BADGE,
      world: [3.4, 0.45, dockZ + 0.9] as Vec3,
      tone: 'gold',
      align: 'bottom',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'cal_decoded_luma',
      aWorld: [
        stageFrames.decoded_luma_16x16.center[0] - stageFrames.decoded_luma_16x16.extent[0] * 0.5,
        SHEET_Y + 0.05,
        stageFrames.decoded_luma_16x16.center[2] + stageFrames.decoded_luma_16x16.extent[2] * 0.5 + 0.14,
      ] as Vec3,
      bWorld: [
        stageFrames.decoded_luma_16x16.center[0] + stageFrames.decoded_luma_16x16.extent[0] * 0.5,
        SHEET_Y + 0.05,
        stageFrames.decoded_luma_16x16.center[2] + stageFrames.decoded_luma_16x16.extent[2] * 0.5 + 0.14,
      ] as Vec3,
      label: '16×16 Scalar Luma',
      partIds: ['decoded_luma_16x16'],
    }),
  );

  const provenanceBadges: UniProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    Object.freeze({ label: UNI_SCENE_VISUAL_HONESTY_BADGE, tone: 'muted' as const }),
    Object.freeze({
      label: `U16 Dual-CFG Mode: ${activeMode} (w_txt=${wTxt}, w_img=${wImg})`,
      tone: 'g_teal' as const,
    }),
    Object.freeze({
      label: 'D4: 0 RGB in RG32F · Typed 2D Canvas RGB Previews',
      tone: 'muted' as const,
    }),
  ];
  if (isEdit && paddingLockMask !== undefined) {
    provenanceBadges.push(
      Object.freeze({
        label: `U18: ${paddingLockMask.lockedCellCount}/16 Border Cells Locked · cropToSource=${cropToSource}`,
        tone: 'gold' as const,
      }),
    );
  }

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
    viewMode: 'output_edit' as const,
    title: isEdit
      ? 'View 6 · Output, Dual-CFG Editing & U18 Crop Pipeline'
      : 'View 6 · Output & PackedLinearImageCodec Expand Pipeline (T2I)',
    subtitle: `${ctx.viewReadout.label} · ${activeMode} · ${regions.length} RG32F rects (${audit.utilizationPct.toFixed(1)}%)`,
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
    cameraPreset: Object.freeze({ azimuthDeg: 14, elevationDeg: 45, fitBounds }),
    fitBounds,
    valueAtlas: atlas,
    atlasKey,
    atlasRegions: regions,
    atlasAudit: audit,
    rgbPreviews: Object.freeze(rgbPreviews),
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
    throw new Error(`Invalid View 6 (output_edit) scene geometry: ${validation.errors.join('; ')}`);
  }
  return scene;
}

// ============================================================================
// UNDERSTANDING / GROUNDING PIPELINE SCENE
// ============================================================================

const EMPTY_TRACE_LAYERS: ReadonlySet<number> = new Set<number>();

function resolveCausalTailLogits(
  ctx: UniResolvedSessionContext,
  vocabSize: number,
): Float32Array {
  const curStartPos = ctx.record.positionIds[0] ?? 0;
  let cursorId = ctx.record.parentRecordId;
  let hops = 0;
  while (cursorId !== null && hops < 128) {
    const rec = ctx.session.getRecord(cursorId);
    if (!rec) break;
    const recStartPos = rec.positionIds[0] ?? 0;
    // Multi-block Understanding: when stepping back into the immediately preceding committed
    // text_block_step (whose start position strictly precedes the active block), reconstruct
    // that block's causal commit tail logits directly (even when its denoising record has
    // maskMode === 'bidirectional'), matching UniSession.finalizeTextBlockCommit at 0 ULP.
    if (
      rec.kind === 'text_block_step' &&
      rec.textState &&
      recStartPos < curStartPos
    ) {
      if (rec.maskMode === 'causal') {
        const head = ctx.session.reconstructStepHeadLogits(rec.recordId);
        const logits = head.perBranchTextLogits[0];
        if (logits && logits.length >= vocabSize) {
          return new Float32Array(logits.subarray(logits.length - vocabSize, logits.length));
        }
      } else if (recStartPos <= ctx.branch.prefixCache.seqLen) {
        const { pastKvs, pastRouting } = extractRowPastKvs(
          cropUniKvCache(ctx.branch.prefixCache, recStartPos),
          0,
        );
        const commit = runUniForwardPass(ctx.bundle.core, {
          tokenIds: rec.textState.tokenIds,
          maskMode: 'causal',
          useCache: false,
          prefillMode: 'hf_faithful',
          downstreamTarget: 'u',
          pastKvs,
          pastRouting,
          computeTextHead: true,
          computeGenHead: false,
          retainedTraceLayers: EMPTY_TRACE_LAYERS,
        });
        const logits = commit.textLogits;
        if (logits && logits.length >= vocabSize) {
          return new Float32Array(logits.subarray(logits.length - vocabSize, logits.length));
        }
      }
    }
    if (rec.kind === 'prefill' || rec.maskMode === 'causal') {
      const head = ctx.session.reconstructStepHeadLogits(rec.recordId);
      const logits = head.perBranchTextLogits[0];
      if (logits && logits.length >= vocabSize) {
        return new Float32Array(logits.subarray(logits.length - vocabSize, logits.length));
      }
    }
    cursorId = rec.parentRecordId;
    hops += 1;
  }
  const curHead = ctx.resolveStepHeadLogits();
  const curLogits = curHead.perBranchTextLogits[0];
  if (curLogits && curLogits.length >= vocabSize) {
    return new Float32Array(curLogits.subarray(curLogits.length - vocabSize, curLogits.length));
  }
  return new Float32Array(vocabSize);
}

function buildGroundingBoxesFromContext(
  ctx: UniResolvedSessionContext,
  sourceImage: UniRgbImage,
  shiftedLogits: Float32Array,
  slotTokens: Int32Array,
): readonly UniGroundingBox[] {
  const vocabSize = ctx.bundle.core.config.vocabSize;

  // If currently on a grounding_step with live coordinate slots, decode directly from slotTokens + shiftedLogits
  if (ctx.record.kind === 'grounding_step' && slotTokens.length >= 4) {
    const slotIndices: number[] = [];
    for (let i = 0; i < slotTokens.length; i++) {
      const id = slotTokens[i]!;
      if (
        id === UNI_MINI_MASK_TOKEN_ID ||
        (id >= UNI_MINI_LOC_START_ID && id <= UNI_MINI_LOC_END_ID)
      ) {
        slotIndices.push(i);
      }
    }
    const numBoxes = Math.floor(slotIndices.length / 4);
    if (numBoxes >= 1) {
      const boxes: UniGroundingBox[] = [];
      const gtTags = sourceImage.groundTruthBoxes ?? [];
      for (let b = 0; b < numBoxes; b++) {
        const ids: number[] = [];
        const strs: string[] = [];
        const bins: number[] = [];
        const rawIds: number[] = [];
        let intervened = 0;

        for (let k = 0; k < 4; k++) {
          const slotPos = slotIndices[b * 4 + k]!;
          const tok = slotTokens[slotPos]!;
          const rowBase = Math.min(slotPos, Math.floor(shiftedLogits.length / vocabSize) - 1) * vocabSize;
          let rawTop1 = 0;
          let rawBest = -Infinity;
          let locBestId: number = UNI_MINI_LOC_START_ID;
          let locBestVal = -Infinity;
          for (let v = 0; v < vocabSize; v++) {
            const val = shiftedLogits[rowBase + v]!;
            if (val > rawBest) {
              rawBest = val;
              rawTop1 = v;
            }
            if (v >= UNI_MINI_LOC_START_ID && v <= UNI_MINI_LOC_END_ID && val > locBestVal) {
              locBestVal = val;
              locBestId = v;
            }
          }
          const resolvedId =
            tok >= UNI_MINI_LOC_START_ID && tok <= UNI_MINI_LOC_END_ID ? tok : locBestId;
          if (rawTop1 < UNI_MINI_LOC_START_ID || rawTop1 > UNI_MINI_LOC_END_ID) {
            intervened += 1;
          }
          ids.push(resolvedId);
          strs.push(UNI_MINI_VOCAB[resolvedId]!);
          bins.push(resolvedId - UNI_MINI_LOC_START_ID);
          rawIds.push(rawTop1);
        }

        const normBins = [bins[0]!, bins[1]!, bins[2]!, bins[3]!] as const;
        boxes.push(
          Object.freeze({
            tag: gtTags[b]?.tag ?? (b === 0 ? 'cat' : `obj_${b}`),
            locTokenIds: [ids[0]!, ids[1]!, ids[2]!, ids[3]!] as const,
            locTokenStrings: [strs[0]!, strs[1]!, strs[2]!, strs[3]!] as const,
            normalizedBins: normBins,
            pixelBox: locBinsToPixelBox(normBins, sourceImage.width, sourceImage.height, 16),
            rawTop1Ids: [rawIds[0]!, rawIds[1]!, rawIds[2]!, rawIds[3]!] as const,
            guardIntervenedCount: intervened,
          }),
        );
      }
      return Object.freeze(boxes);
    }
  }

  // Otherwise run clean-room `groundObjects` on the source image using Mini RADIO-H features
  const visionTrace = runMiniVisionPipeline(
    sourceImage,
    ctx.bundle.vision,
    ctx.gainPresetId,
  );
  const basePrompt = Int32Array.from([1, 10, UNI_MINI_UND_PAD_TOKEN_ID, 26, 27]);
  const fixed = fixOnePrompt(
    basePrompt,
    visionTrace.projector.tokenGridRows,
    visionTrace.projector.tokenGridCols,
  );
  const textEmbeds = embedUniTextTokens(fixed.expandedTokenIds, ctx.bundle.core);
  const promptEmbeds = spliceVisualEmbeddingsIntoPrompt(
    fixed.expandedTokenIds,
    textEmbeds,
    fixed.visualSlotIndices,
    visionTrace.projector.visualEmbeddings,
    ctx.bundle.core.config.hiddenSize,
  );
  const tags =
    sourceImage.groundTruthBoxes && sourceImage.groundTruthBoxes.length > 0
      ? sourceImage.groundTruthBoxes.map((b) => b.tag)
      : ['cat'];

  const grounded = groundObjects(ctx.bundle.core, {
    promptTokenIds: fixed.expandedTokenIds,
    promptInputsEmbeds: promptEmbeds,
    imageWidth: sourceImage.width,
    imageHeight: sourceImage.height,
    tags,
    steps: 2,
    enforceLocDomainGuard: true,
  });
  return grounded.boxes;
}

function buildUnderstandingOutputScene(
  ctx: UniResolvedSessionContext,
  options?: UniOutputEditSceneOptions,
): UniSpatialScene {
  const { branch, record } = ctx;
  const cfg = ctx.bundle.core.config;
  const vocabSize = cfg.vocabSize; // 64
  const d = cfg.hiddenSize; // 8
  const isGrounding = branch.taskMode === 'object_grounding';
  const hasSourceImage =
    branch.taskMode === 'visual_understanding' || branch.taskMode === 'object_grounding';

  const sourceImage: UniRgbImage | undefined = hasSourceImage
    ? createUniScenePreset(branch.scenePresetId ?? 'sunlit_cat_16x16')
    : undefined;

  const headLogits = ctx.resolveStepHeadLogits();
  const fwdLogits = headLogits.perBranchTextLogits[0] ?? new Float32Array(vocabSize);
  const fwdFinalNorm = headLogits.perBranchFinalNorm[0] ?? new Float32Array(d);
  const prevLogits = resolveCausalTailLogits(ctx, vocabSize);

  // Build shifted FLARE diffusion_head logits (`shift_logits = true`, B = 4 in understanding)
  let slotTokens: Int32Array;
  let blockRows: number;
  if (record.kind === 'text_block_step' && record.textState) {
    blockRows = Math.max(1, Math.min(16, record.textState.tokenIds.length));
    slotTokens = record.textState.tokenIds.subarray(0, blockRows);
  } else if (record.kind === 'grounding_step' && record.textState) {
    blockRows = Math.max(4, Math.min(16, record.textState.tokenIds.length));
    slotTokens = record.textState.tokenIds.subarray(0, blockRows);
  } else if (isGrounding) {
    blockRows = 6;
    slotTokens = Int32Array.from([
      UNI_MINI_MASK_TOKEN_ID,
      UNI_MINI_MASK_TOKEN_ID,
      UNI_MINI_MASK_TOKEN_ID,
      UNI_MINI_MASK_TOKEN_ID,
      UNI_MINI_LOC_END_TOKEN_ID,
      cfg.eosTokenId,
    ]);
  } else {
    blockRows = 4; // B = 4 block-diffusion default
    slotTokens = new Int32Array(blockRows).fill(UNI_MINI_MASK_TOKEN_ID);
  }

  const shiftedLogits = new Float32Array(blockRows * vocabSize);
  shiftedLogits.set(prevLogits.subarray(0, vocabSize), 0);
  const availRows = Math.floor(fwdLogits.length / vocabSize);
  for (let j = 1; j < blockRows; j++) {
    const srcRow = Math.min(j - 1, Math.max(0, availRows - 1));
    shiftedLogits.set(
      fwdLogits.subarray(srcRow * vocabSize, (srcRow + 1) * vocabSize),
      j * vocabSize,
    );
  }

  const shiftedProbs = softmaxRows(shiftedLogits, blockRows, vocabSize);
  const unmaskedSet = new Set<number>(record.textState?.unmaskedOffsets ?? []);

  const unmaskStateR = new Float32Array(blockRows * 4);
  const unmaskStateG = new Float32Array(blockRows * 4);
  const stripR = new Float32Array(blockRows);
  const stripG = new Float32Array(blockRows);
  let stillMaskedCount = 0;

  for (let r = 0; r < blockRows; r++) {
    const tok = slotTokens[r]!;
    const isMasked = tok === UNI_MINI_MASK_TOKEN_ID;
    if (isMasked) stillMaskedCount += 1;
    let bestProb = 0;
    const pBase = r * vocabSize;
    for (let v = 0; v < vocabSize; v++) {
      const p = shiftedProbs[pBase + v]!;
      if (p > bestProb) bestProb = p;
    }
    const base = r * 4;
    unmaskStateR[base] = tok / (vocabSize - 1);
    unmaskStateR[base + 1] = bestProb;
    unmaskStateR[base + 2] = unmaskedSet.has(r) ? 1.0 : 0.0;
    unmaskStateR[base + 3] = isMasked ? 1.0 : 0.0;

    // CRITICAL SheetsPass rule: G = 1.0 ONLY on still-masked [MASK] positions!
    const gVal = isMasked ? 1.0 : 0.0;
    unmaskStateG[base] = gVal;
    unmaskStateG[base + 1] = gVal;
    unmaskStateG[base + 2] = gVal;
    unmaskStateG[base + 3] = gVal;

    stripR[r] = bestProb;
    stripG[r] = gVal;
  }

  const hiddenRows = Math.max(1, Math.min(32, Math.floor(fwdFinalNorm.length / d)));
  const hiddenSlice = fwdFinalNorm.subarray(0, hiddenRows * d);

  const W = UNI_OUTPUT_EDIT_ATLAS_SIZE;
  const H = UNI_OUTPUT_EDIT_ATLAS_SIZE;
  const writer = createRg32fAtlas(W, H);
  const entries: OutputEditAtlasSliceEntry[] = [];
  const sliceMap = {} as Record<UniOutputEditUndStageId, OutputEditAtlasSliceEntry>;
  const add = (params: LocalOutputMatrixPackParams) => {
    const entry = packOutputEditMatrixToRg32f(writer, params);
    entries.push(entry);
    sliceMap[params.id as UniOutputEditUndStageId] = entry;
  };

  add({
    id: 'shifted_diffusion_logits',
    x0: 0,
    y0: 0,
    rows: blockRows,
    cols: vocabSize,
    values: shiftedLogits,
    mode: 0,
    isMaskOrScore: true,
  });
  add({
    id: 'shifted_diffusion_probs',
    x0: 0,
    y0: 18,
    rows: blockRows,
    cols: vocabSize,
    values: shiftedProbs,
    mode: 1,
    isMaskOrScore: true,
    explicitScale: 1.0,
  });
  add({
    id: 'unmask_mask_state',
    x0: 66,
    y0: 0,
    rows: blockRows,
    cols: 4,
    values: unmaskStateR,
    mode: 1,
    isMaskOrScore: true,
    gChannelFlags: unmaskStateG,
    explicitScale: 1.0,
  });
  add({
    id: 'token_slot_mask_strip',
    x0: 66,
    y0: 18,
    rows: 1,
    cols: blockRows,
    values: stripR,
    mode: 1,
    isMaskOrScore: true,
    gChannelFlags: stripG,
    explicitScale: 1.0,
  });
  add({
    id: 'final_norm_hidden',
    x0: 72,
    y0: 0,
    rows: hiddenRows,
    cols: d,
    values: hiddenSlice,
    mode: 0,
    isMaskOrScore: false,
  });

  if (sourceImage !== undefined) {
    const luma = computeScalarLumaFromRgb(sourceImage);
    add({
      id: 'source_luma_sheet',
      x0: 84,
      y0: 0,
      rows: sourceImage.height,
      cols: sourceImage.width,
      values: luma,
      mode: 1,
      isMaskOrScore: false,
      explicitScale: 1.0,
    });
  }

  const regions = Object.freeze(entries.map((e) => e.region));
  const audit = auditAtlasRegions(W, H, regions, UNI_OUTPUT_EDIT_ATLAS_BUDGET_BYTES);
  const atlas: Rg32fValueAtlas = Object.freeze({ width: W, height: H, data: writer.data });
  const atlasKey = computeAtlasContentKey('output_edit', atlas, `${record.recordId}:und`);

  // Build 3D geometry
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

  lines.addBoxFrameXZ(-6.6, -6.2, 6.6, 5.8, FRAME_LINE_Y, 1.0, LINE_STYLE_ACTIVE, 0.5);
  solids.push({
    cx: 0,
    cy: 0.04,
    cz: -5.4,
    ex: 4.0,
    ey: 0.08,
    ez: 0.65,
    cellFamily: CELL_WEIGHT_BLUE,
    bayId: 0,
    highlight: 0,
  });
  mergeAabb(boundsAcc, { min: [-7.0, -0.1, -6.6], max: [7.0, 0.9, 6.4] });

  interface UndPlateSpec {
    readonly id: UniOutputEditUndStageId;
    readonly cx: number;
    readonly cz: number;
    readonly ex: number;
    readonly ez: number;
    readonly sheetKind: number;
    readonly role: number;
    readonly highlight: number;
    readonly semanticTag: string;
    readonly label: string;
    readonly sublabel: string;
    readonly badge?: string;
  }

  const undSpecs: UndPlateSpec[] = [
    {
      id: 'final_norm_hidden',
      cx: -4.6,
      cz: -2.4,
      ex: 2.1,
      ez: 2.2,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_NORM,
      highlight: SHEET_HL_OPERAND_A,
      semanticTag: 'branch=u;stage=final_norm;op=encoder_norm_u',
      label: `encoder.norm (U) (${hiddenRows}×8)`,
      sublabel: 'Final RMSNorm hidden states entering diffusion_head (G = 0.0)',
    },
    {
      id: 'shifted_diffusion_logits',
      cx: 0.4,
      cz: -2.4,
      ex: 5.4,
      ez: 2.2,
      sheetKind: SHEET_KIND_CAUSAL_SCORES, // U25 live RG32F
      role: SHEET_ROLE_SCORES,
      highlight: SHEET_HL_ACTIVE,
      semanticTag: `branch=u;stage=shifted_diffusion_logits;shift_logits=true;B=${blockRows};u25=live_rg32f`,
      label: `Shifted diffusion_head Logits (${blockRows}×64)`,
      sublabel: 'shift_logits = true (FLARE: slot 0 pre-seeded from causal tail, G = 0.0)',
      badge: 'shift_logits=true · U25',
    },
    {
      id: 'shifted_diffusion_probs',
      cx: 0.4,
      cz: 0.9,
      ex: 5.4,
      ez: 2.2,
      sheetKind: SHEET_KIND_CAUSAL_SOFTMAX, // U25 live RG32F
      role: SHEET_ROLE_SOFTMAX,
      highlight: SHEET_HL_OPERAND_A,
      semanticTag: `branch=u;stage=shifted_diffusion_probs;shift_logits=true;B=${blockRows};u25=live_rg32f`,
      label: `Shifted Softmax Probabilities (${blockRows}×64)`,
      sublabel: 'Softmax confidence distribution over V_txt = 64 incl. <LOC_0..16> (G = 0.0)',
      badge: 'U25 Softmax (G=0)',
    },
    {
      id: 'unmask_mask_state',
      cx: -4.4,
      cz: 1.0,
      ex: 2.2,
      ez: 2.0,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      highlight: SHEET_HL_PATHWAY,
      semanticTag: `branch=u;stage=unmask_state;masked=${stillMaskedCount}`,
      label: `Block Unmask State (${blockRows}×4)`,
      sublabel: `G = 1.0 ONLY on ${stillMaskedCount} still-masked [MASK] rows`,
      badge: `${stillMaskedCount} Masked`,
    },
    {
      id: 'token_slot_mask_strip',
      cx: -1.4,
      cz: 3.8,
      ex: 3.4,
      ez: 1.1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SOFTMAX,
      highlight: SHEET_HL_OPERAND_A,
      semanticTag: `branch=u;stage=slot_mask_strip;B=${blockRows}`,
      label: `Active Slot Confidence & [MASK] Strip (1×${blockRows})`,
      sublabel: 'G = 1.0 on remaining [MASK] slots; G = 0.0 on unmasked slots',
    },
  ];

  if (sourceImage !== undefined) {
    undSpecs.push({
      id: 'source_luma_sheet',
      cx: 3.4,
      cz: 3.8,
      ex: 2.6,
      ez: sourceImage.height < sourceImage.width ? 1.4 : 2.6,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      highlight: SHEET_HL_OPERAND_A,
      semanticTag: `branch=u;stage=source_luma;d4=scalar_only;grounding=${isGrounding ? 1 : 0}`,
      label: `Source Scalar Luma (${sourceImage.width}×${sourceImage.height})`,
      sublabel: 'Rec.601 luma scalar plate (G = 0.0; RGB + <LOC_*> boxes on 2D Canvas)',
      badge: isGrounding ? 'Grounding <LOC_*>' : 'Source Luma',
    });
  }

  const hovered = options?.hoveredCell ?? null;
  const stageFrames: Record<string, SheetFrame3D> = {};

  for (let i = 0; i < undSpecs.length; i++) {
    const spec = undSpecs[i]!;
    const sliceEntry = sliceMap[spec.id];
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
      highlight: isHovered ? SHEET_HL_ACTIVE : spec.highlight,
      seed: i * 0.67,
      slice: sliceEntry.slice,
      frame,
    });

    if (spec.id === 'unmask_mask_state') {
      for (let r = 0; r < blockRows; r++) {
        if (slotTokens[r] === UNI_MINI_MASK_TOKEN_ID) {
          flow.addCellHighlightOnFrame(
            frame,
            r,
            r,
            0,
            3,
            blockRows,
            4,
            SHEET_EY * 0.5 + 0.012,
            MASKED_TEXT_FILL,
            UNI_FLOW_COLORS.ghostSlate,
            0.018,
            true,
          );
        }
      }
    }

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
        SHEET_EY * 0.5 + 0.016,
        HOVER_FILL,
        UNI_FLOW_COLORS.activeOxide,
        0.024,
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
        towerRole: 'context',
        semanticTag: spec.semanticTag,
      }),
    );

    overlayAnchors.push(
      Object.freeze({
        id: `anchor_${spec.id}`,
        label: spec.label,
        sublabel: spec.sublabel,
        worldPos: [spec.cx, SHEET_Y + 0.22, spec.cz - spec.ez * 0.5 - 0.16] as Vec3,
        tone: 'u_indigo',
        ...(spec.badge !== undefined ? { badge: spec.badge } : {}),
      }),
    );
  }

  const xz = (x: number, z: number) => ({ x, z });
  flow.addPolylineXZ(
    [
      xz(-3.5, -2.4),
      xz(-2.4, -2.4),
    ],
    RIBBON_Y,
    0.16,
    UNI_FLOW_COLORS.uIndigo,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(0.4, -1.2),
      xz(0.4, -0.25),
    ],
    RIBBON_Y,
    0.16,
    UNI_FLOW_COLORS.uIndigo,
    true,
  );
  flow.addCubicXZ(
    xz(0.4, 2.05),
    xz(0.4, 2.8),
    xz(-1.4, 2.8),
    xz(-1.4, 3.2),
    RIBBON_Y,
    0.16,
    UNI_FLOW_COLORS.uIndigo,
    true,
    18,
  );

  // Grounding RGB preview descriptor (on object_grounding AND visual_understanding with source image)
  const rgbPreviews: UniRgbPreviewDescriptor[] = [];
  let groundingBoxes: readonly UniGroundingBox[] = [];
  if (sourceImage !== undefined) {
    groundingBoxes = buildGroundingBoxesFromContext(
      ctx,
      sourceImage,
      shiftedLogits,
      slotTokens,
    );
    rgbPreviews.push(
      Object.freeze({
        id: 'output_edit_grounding',
        kind: 'grounding' as const,
        label: isGrounding
          ? `Object Grounding <LOC_0..16> (${sourceImage.width}×${sourceImage.height})`
          : `Visual Grounding Overlay (${sourceImage.width}×${sourceImage.height})`,
        caption: `U27: Direct ID decode preserves ${groundingBoxes.length} <LOC_0..16> box(es) (skip_special_tokens=True would strip!)`,
        width: sourceImage.width,
        height: sourceImage.height,
        rgbChw: new Float32Array(sourceImage.data),
        groundingBoxes,
        honestyBadge: UNI_SCENE_VISUAL_HONESTY_BADGE,
      }),
    );
  }

  landmarks.push(
    Object.freeze({
      id: 'lm_und_output_head',
      label: isGrounding
        ? `Grounding diffusion_head · ${groundingBoxes.length} <LOC_0..16> Box(es)`
        : `Block-Diffusion diffusion_head (B=${blockRows}, shift_logits=true)`,
      sublabel: isGrounding
        ? 'U27: Direct token-ID lookup (skip_special_tokens=True strips <LOC_*>)'
        : `${blockRows - stillMaskedCount}/${blockRows} slots unmasked`,
      world: [0.4, 0.46, -4.4] as Vec3,
      tone: 'default',
      align: 'top',
      tier: 'primary',
    }),
  );

  const logitsFrame = stageFrames.shifted_diffusion_logits!;
  calipers.push(
    Object.freeze({
      id: 'cal_shifted_logits',
      aWorld: [
        logitsFrame.center[0] - logitsFrame.extent[0] * 0.5,
        SHEET_Y + 0.05,
        logitsFrame.center[2] - logitsFrame.extent[2] * 0.5 - 0.12,
      ] as Vec3,
      bWorld: [
        logitsFrame.center[0] + logitsFrame.extent[0] * 0.5,
        SHEET_Y + 0.05,
        logitsFrame.center[2] - logitsFrame.extent[2] * 0.5 - 0.12,
      ] as Vec3,
      label: `B=${blockRows} × V_txt=64`,
      partIds: ['shifted_diffusion_logits'],
    }),
  );

  const provenanceBadges: UniProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    Object.freeze({ label: UNI_SCENE_VISUAL_HONESTY_BADGE, tone: 'muted' as const }),
    Object.freeze({
      label: `FLARE Block-Diffusion: B=${blockRows} · shift_logits=true (diffusion_head)`,
      tone: 'u_indigo' as const,
    }),
  ];

  if (sourceImage !== undefined) {
    provenanceBadges.push(
      Object.freeze({
        label:
          'U27: skip_special_tokens=True Strips <LOC_0..16> & <box_p> Tokens — Direct ID Decode Preserved',
        tone: 'warn' as const,
      }),
    );
  }

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
    viewMode: 'output_edit' as const,
    title: isGrounding
      ? 'View 6 · Object Grounding (<LOC_0..16> & U27 Direct Decode)'
      : 'View 6 · Block-Diffusion Output Pipeline (FLARE shift_logits=true)',
    subtitle: `${ctx.viewReadout.label} · B=${blockRows} · ${regions.length} RG32F rects (${audit.utilizationPct.toFixed(1)}%)`,
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
    cameraPreset: Object.freeze({ azimuthDeg: 14, elevationDeg: 45, fitBounds }),
    fitBounds,
    valueAtlas: atlas,
    atlasKey,
    atlasRegions: regions,
    atlasAudit: audit,
    rgbPreviews: Object.freeze(rgbPreviews),
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
    throw new Error(`Invalid View 6 (output_edit) scene geometry: ${validation.errors.join('; ')}`);
  }
  return scene;
}

// ============================================================================
// PUBLIC VIEW 6 DISPATCHER (`buildUniOutputEditScene`)
// ============================================================================

/**
 * Build the task-reactive headless 3D spatial scene for **View 6 (`output_edit`)**:
 * switches cleanly between T2I / Image Editing (`PackedLinearImageCodec.expand`,
 * `U18` `paddingLockMask`, `16×16` scalar luma, 3-branch Dual-CFG mixers, and typed
 * `source`/`letterboxed`/`decoded`/`cropped` RGB previews) and Understanding / Grounding
 * (FLARE `shift_logits = true` `diffusion_head`, `[MASK]` unmask plates, and `<LOC_0..16>`
 * `groundingBoxes` with `U27` provenance).
 */
export function buildUniOutputEditScene(
  session: UniSession,
  options?: UniOutputEditSceneOptions,
): UniSpatialScene {
  const ctx = resolveUniSessionContext(session, options?.recordId);
  if (ctx.branch.taskMode === 'text_to_image' || ctx.branch.taskMode === 'image_edit') {
    return buildGenerationOutputEditScene(ctx, options);
  }
  return buildUnderstandingOutputScene(ctx, options);
}
