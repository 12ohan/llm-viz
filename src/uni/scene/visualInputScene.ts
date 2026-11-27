/**
 * Milestone UNI4 — View 2 (`visual_input`): Dual Visual Input Pipeline Scene
 * & `128 × 128` `RG32F` Scalar/Mask Atlas (`src/uni/scene/visualInputScene.ts`)
 *
 * Clean-room headless 3D scene builder for Nemotron-Labs-Diffusion-Uni:
 *   - Left lane (Continuous `U` Indigo):
 *       1. `4×4` CLIP-normalized patchification (`16×48` or `8×48`)
 *       2. `4` `[CLS/REG]` skip tokens + `2`-block ViT bidirectional softmax (`20×20` or `12×12`,
 *          `G = 0.0`, strict `U25` compliance with live `RG32F` `TexRect`)
 *       3. Skip-token strip `[:, 4:]` spatial hidden states (`16×16` or `8×16`)
 *       4. `2×2` `RadioHProjector` spatial gather (`4×64` or `2×64`)
 *       5. `RadioHProjector` (`RMSNorm` + `SquaredReLU` + `D3` `radio_multiply` gain)
 *          + `-500` / literal `\n` (`fixOnePrompt`) insertion (`6×8` or `3×8`),
 *          explicitly documenting `U8` (`encoder.image_newline` unused on active path)
 *   - Right lane (Discrete `G` Teal):
 *       6. `Emu3VQ` `[4, 4, 8]` conv/attn encoder latents (`16×8`)
 *       7. IBQ dot-product `argmax(z @ E^T)` codebook score slice (`16×64` covering all
 *          winning codes + highest-scoring codebook entries, `U14`)
 *       8. `4×4` quantized code grid & `U18` `paddingLockMask` (`R = codeId / 255`,
 *          `G = 1.0` where border-locked else `0.0`)
 *       9. `PackedLinearImageCodec` `2×2` compression (`4×8` with `D3` generation scale
 *          + additive `2D_SinCos`)
 *   - Center convergence (`U + G` Joint):
 *      10. Assembled `[S, 8]` multimodal sequence plate (`R = token embedding`, `G = 0.0` active
 *          so `SheetsPass` renders live Float32 values for both `U` and `G`; `G` rows framed in
 *          teal via `flow.addCellHighlightOnFrame`) feeding the `L0` backbone rail.
 *
 * Strict `D4` discipline: ZERO 3-channel RGB pixels are packed into `RG32F`; typed
 * `UniRgbPreviewDescriptor`s (`source` + `letterboxed` with `U18` lock mask) are emitted
 * in `scene.rgbPreviews` for `UNI6`'s 2D `<canvas>` `pixelCanvas.ts`.
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
import {
  FRAME_XZ_AXES,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_CAUSAL_SOFTMAX,
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
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../scene/spine';
import {
  compressPackedImageEmbeddings,
  computeSourcePaddingVqMask,
} from '../codec';
import { embedUniGenTokens, embedUniTextTokens } from '../forward';
import type { UniSession } from '../session';
import {
  UNI_GAIN_PRESETS,
  UNI_MINI_BOS_TOKEN_ID,
  UNI_MINI_CONFIG,
  UNI_MINI_IM_START_TOKEN_ID,
  UNI_MINI_NEWLINE_TOKEN_ID,
  UNI_MINI_UND_PAD_TOKEN_ID,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_MINI_VQ_Z_CH,
  UNI_ROUTE_G,
  UNI_ROUTE_U,
  type UniCodecCompressTrace,
  type UniGainPresetId,
  type UniLetterboxResult,
  type UniRgbImage,
  type UniScenePresetId,
  type UniSourcePaddingVqMask,
  type UniVisionPipelineTrace,
  type UniVqEncodeTrace,
} from '../types';
import {
  createUniScenePreset,
  fixOnePrompt,
  padToSquareAndResize,
  runMiniVisionPipeline,
  spliceVisualEmbeddingsIntoPrompt,
} from '../vision';
import { encodeMiniEmu3Vq } from '../vqvae';
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
  UniRgbPreviewDescriptor,
  UniSpatialScene,
  UniVisualInputSceneOptions,
} from './types';

// ============================================================================
// CONSTANTS & EXPORTED HELPER TYPES
// ============================================================================

export const UNI_VISUAL_INPUT_ATLAS_SIZE = 128 as const;
export const UNI_VISUAL_INPUT_ATLAS_BUDGET_BYTES = 131_072 as const; // 128 * 128 * 8 B
export const UNI_VISUAL_INPUT_REGION_COUNT = 10 as const;
export const UNI_VISUAL_INPUT_IBQ_SLICE_COLS = 64 as const;
export const UNI_VISUAL_INPUT_DEFAULT_PRESET: UniScenePresetId = 'sunlit_cat_16x16';

export type UniVisualInputStageId =
  | 'u_patches'
  | 'u_vit_attn'
  | 'u_patch_strip'
  | 'u_projector_gather'
  | 'u_projector_newline'
  | 'g_vq_latents'
  | 'g_ibq_scores'
  | 'g_vq_code_grid'
  | 'g_packed_codec'
  | 'joint_multimodal_seq';

export interface UniVisualInputAtlasSliceEntry {
  readonly id: UniVisualInputStageId;
  readonly region: UniAtlasRegionBox;
  readonly slice: AtlasSlice;
}

export interface UniVisualInputAtlasPackResult {
  readonly atlas: Rg32fValueAtlas;
  readonly atlasKey: string;
  readonly regions: readonly UniAtlasRegionBox[];
  readonly audit: UniAtlasLayoutAudit;
  readonly slices: Readonly<Record<UniVisualInputStageId, UniVisualInputAtlasSliceEntry>>;
  readonly selectedIbqCodeIds: Int32Array;
}

export interface UniVisualInputPipelineRun {
  readonly sourceImage: UniRgbImage;
  readonly letterbox: UniLetterboxResult;
  readonly paddingLockMask: UniSourcePaddingVqMask;
  readonly gainPresetId: UniGainPresetId;
  readonly continuousStreamActive: boolean;
  readonly discreteStreamActive: boolean;
  readonly isTextOnlyFallback: boolean;
  readonly visionTrace: UniVisionPipelineTrace;
  readonly vqTrace: UniVqEncodeTrace;
  readonly packedCodecTrace: UniCodecCompressTrace;
  /** `[rowNewlineSeqLen, 8]`: `RadioHProjector` `D3` output + literal `\n` (`24`) tokens (`U8`). */
  readonly rowNewlineEmbeddings: Float32Array;
  readonly rowNewlineTokenIds: Int32Array;
  readonly rowNewlineSeqLen: number;
  /** `[16, 8]` spatial-major `[r*4 + c, d]` reshape of CHW `vqTrace.zContinuous`. */
  readonly vqLatentsSpatial: Float32Array;
  /** `[16, 64]` IBQ dot-product score slice covering all winning codes + top active codes. */
  readonly ibqScoreSlice: Float32Array;
  /** `[64]` codebook IDs represented by columns `0..63` of `ibqScoreSlice`. */
  readonly selectedIbqCodeIds: Int32Array;
  /** `[S, 8]` assembled multimodal sequence entering `L0` backbone. */
  readonly assembledEmbeddings: Float32Array;
  /** `[S]` routing flags (`0 = U`, `1 = G`) aligned with `assembledEmbeddings`. */
  readonly assembledRouting: Uint8Array;
  readonly assembledSeqLen: number;
  readonly assembledUCount: number;
  readonly assembledGCount: number;
}

export interface PackMatrixToRg32fParams {
  readonly id: string;
  readonly x0: number;
  readonly y0: number;
  readonly rows: number;
  readonly cols: number;
  readonly values: ArrayLike<number>;
  readonly mode: 0 | 1;
  readonly isMaskOrScore: boolean;
  /** Optional explicit per-cell G-channel flag (`0.0` or `1.0`). Preserves finite R value! */
  readonly gChannelFlags?: ArrayLike<number>;
  /** Optional explicit normalization scale; defaults to `1 / computeMaxAbs(values)` (mode 0) or `1.0` (mode 1). */
  readonly explicitScale?: number;
}

// ============================================================================
// RG32F MATRIX PACKER (PRESERVING R VALUE WHEN G-FLAG = 1.0)
// ============================================================================

/**
 * Pack a row-major `[rows, cols]` scalar matrix into an `Rg32fAtlasWriter` at `(x0, y0)`,
 * optionally writing an explicit per-cell `G` channel flag (`0.0` or `1.0`) without zeroing `R`.
 */
function packMatrixToRg32f(
  writer: Rg32fAtlasWriter,
  params: PackMatrixToRg32fParams,
): { readonly region: UniAtlasRegionBox; readonly slice: AtlasSlice } {
  const { id, x0, y0, rows, cols, values, mode, isMaskOrScore, gChannelFlags, explicitScale } =
    params;

  if (
    !Number.isInteger(x0) ||
    !Number.isInteger(y0) ||
    !Number.isInteger(rows) ||
    !Number.isInteger(cols) ||
    rows < 1 ||
    cols < 1
  ) {
    throw new RangeError(`Invalid RG32F region "${id}": (${x0},${y0}) ${cols}x${rows}`);
  }
  if (x0 < 0 || y0 < 0 || x0 + cols > writer.width || y0 + rows > writer.height) {
    throw new RangeError(
      `RG32F region "${id}" (${x0},${y0} + ${cols}x${rows}) exceeds atlas ${writer.width}x${writer.height}`,
    );
  }
  const cellCount = rows * cols;
  if (values.length < cellCount) {
    throw new RangeError(
      `RG32F region "${id}" values length (${values.length}) < rows*cols (${cellCount})`,
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
      const gFlag = gChannelFlags !== undefined ? (gChannelFlags[idx]! > 0.5 ? 1.0 : 0.0) : 0.0;
      if (gFlag > 0.5) {
        // Write directly so both R (scalar value) and G (1.0 mask/routing flag) are preserved.
        const texIdx = (dstY * writer.width + (x0 + c)) * 2;
        writer.data[texIdx] = Number.isFinite(rawVal) ? rawVal : 0.0;
        writer.data[texIdx + 1] = 1.0;
      } else {
        writer.setTexel(x0 + c, dstY, rawVal, false);
      }
    }
  }

  const region: UniAtlasRegionBox = Object.freeze({
    id,
    x0,
    y0,
    cols,
    rows,
    mode,
    isMaskOrScore,
  });

  const slice: AtlasSlice = Object.freeze({
    rect: writer.makeRect(x0, y0, cols, rows),
    scale,
    mode,
  });

  return { region, slice };
}

// ============================================================================
// REAL RUNTIME DUAL-ENCODER PIPELINE EVALUATION
// ============================================================================

/**
 * Select `sliceCols` (64) deterministic codebook IDs from `dotScores [numCells, 256]`,
 * guaranteeing every spatial cell's winning `codes[p]` is included, then filling remaining
 * slots with the highest peak-score codebook entries across the spatial lattice, sorted ascending.
 */
function buildIbqScoreSlice(
  dotScores: Float32Array,
  codes: Int32Array,
  numCells: number,
  codebookSize: number = UNI_MINI_VQ_CODEBOOK_SIZE,
  sliceCols: number = UNI_VISUAL_INPUT_IBQ_SLICE_COLS,
): { readonly slice: Float32Array; readonly selectedCodeIds: Int32Array } {
  const chosen = new Set<number>();
  for (let p = 0; p < numCells; p++) {
    const win = codes[p]!;
    if (win >= 0 && win < codebookSize) {
      chosen.add(win);
    }
  }

  // Peak dot-product score per codebook entry across all spatial cells
  const peakScores = new Float32Array(codebookSize);
  peakScores.fill(-Infinity);
  for (let p = 0; p < numCells; p++) {
    const base = p * codebookSize;
    for (let k = 0; k < codebookSize; k++) {
      const s = dotScores[base + k]!;
      if (s > peakScores[k]!) {
        peakScores[k] = s;
      }
    }
  }

  const ranked = new Array<number>(codebookSize);
  for (let k = 0; k < codebookSize; k++) {
    ranked[k] = k;
  }
  ranked.sort((a, b) => {
    const diff = peakScores[b]! - peakScores[a]!;
    return diff !== 0 ? diff : a - b;
  });

  for (let i = 0; i < ranked.length && chosen.size < sliceCols; i++) {
    chosen.add(ranked[i]!);
  }

  const sortedIds = Array.from(chosen).sort((a, b) => a - b);
  const selectedCodeIds = Int32Array.from(sortedIds.slice(0, sliceCols));

  const slice = new Float32Array(numCells * sliceCols);
  for (let p = 0; p < numCells; p++) {
    const srcBase = p * codebookSize;
    const dstBase = p * sliceCols;
    for (let c = 0; c < sliceCols; c++) {
      slice[dstBase + c] = dotScores[srcBase + selectedCodeIds[c]!]!;
    }
  }

  return { slice, selectedCodeIds };
}

function reshapeVqLatentsChwToSpatial(
  zContinuous: Float32Array,
  vqGridRows: number,
  vqGridCols: number,
  latentDim: number = UNI_MINI_VQ_Z_CH,
): Float32Array {
  const numCells = vqGridRows * vqGridCols;
  const out = new Float32Array(numCells * latentDim);
  for (let p = 0; p < numCells; p++) {
    const dstBase = p * latentDim;
    for (let d = 0; d < latentDim; d++) {
      out[dstBase + d] = zContinuous[d * numCells + p]!;
    }
  }
  return out;
}

/**
 * Execute the 100% live dual visual input pipelines (Mini RADIO-H + Mini RadioHProjector,
 * Mini Emu3VQ + PackedLinearImageCodec, and unified `U + G` multimodal sequence assembly)
 * for the active session and options.
 */
export function runVisualInputPipelines(
  session: UniSession,
  options?: UniVisualInputSceneOptions,
): {
  readonly ctx: ReturnType<typeof resolveUniSessionContext>;
  readonly run: UniVisualInputPipelineRun;
} {
  const ctx = resolveUniSessionContext(session, options?.recordId);
  const gainPresetId: UniGainPresetId = options?.gainPresetId ?? ctx.gainPresetId;
  const taskMode = ctx.branch.taskMode;

  let sourceImage: UniRgbImage;
  let isTextOnlyFallback = false;

  if (options?.sourceImage !== undefined) {
    sourceImage = options.sourceImage;
  } else if (options?.scenePresetId !== undefined) {
    sourceImage = createUniScenePreset(options.scenePresetId);
  } else if (ctx.branch.scenePresetId !== undefined) {
    sourceImage = createUniScenePreset(ctx.branch.scenePresetId);
  } else {
    sourceImage = createUniScenePreset(UNI_VISUAL_INPUT_DEFAULT_PRESET);
    isTextOnlyFallback =
      taskMode === 'text_understanding' || taskMode === 'text_to_image';
  }

  const continuousStreamActive =
    taskMode === 'visual_understanding' ||
    taskMode === 'object_grounding' ||
    taskMode === 'image_edit' ||
    options?.sourceImage !== undefined ||
    options?.scenePresetId !== undefined;

  const discreteStreamActive =
    options?.includeDiscreteStream === true || taskMode === 'image_edit';

  // 1. Continuous Stream: Mini RADIO-H ViT (2L, 4 skip tokens) + RadioHProjector (D3 gain)
  const visionTrace = runMiniVisionPipeline(sourceImage, ctx.bundle.vision, gainPresetId);
  const { tokenGridRows, tokenGridCols } = visionTrace.projector;

  // Row-newline `-500` (`fixOnePrompt`) expansion: `<SPECIAL_23> \n` ->
  //   - `16×16` (`2×2` tokens): `[23, 23, 24, 23, 23, 24]` = `6 × 8` (`2` rows × `2` cols + `2` `\n`)
  //   - `16×8`  (`1×2` tokens): `[23, 23, 24]`             = `3 × 8` (`1` row  × `2` cols + `1` `\n`)
  // Strictly NEVER reads `bundle.vision.imageNewline` (`U8`).
  const fixedVisualBlock = fixOnePrompt(
    [UNI_MINI_UND_PAD_TOKEN_ID, UNI_MINI_NEWLINE_TOKEN_ID],
    tokenGridRows,
    tokenGridCols,
  );
  const baseRowNewlineEmbeds = embedUniTextTokens(
    fixedVisualBlock.expandedTokenIds,
    ctx.bundle.core,
  );
  const rowNewlineEmbeddings = spliceVisualEmbeddingsIntoPrompt(
    fixedVisualBlock.expandedTokenIds,
    baseRowNewlineEmbeds,
    fixedVisualBlock.visualSlotIndices,
    visionTrace.projector.visualEmbeddings,
    UNI_MINI_CONFIG.hiddenSize,
  );
  const rowNewlineSeqLen = fixedVisualBlock.expandedTokenIds.length;

  // 2. Discrete Stream: Square letterbox (`16×16`) -> U18 paddingLockMask -> Emu3VQ -> PackedLinearImageCodec
  const letterbox = padToSquareAndResize(sourceImage, UNI_MINI_CONFIG.imageResolution);
  const paddingLockMask = computeSourcePaddingVqMask(
    letterbox.info,
    UNI_MINI_CONFIG.vitPatchSize,
  );
  const vqTrace = encodeMiniEmu3Vq(letterbox.image, ctx.bundle.vqvae);
  const vqLatentsSpatial = reshapeVqLatentsChwToSpatial(
    vqTrace.zContinuous,
    vqTrace.vqGridRows,
    vqTrace.vqGridCols,
    UNI_MINI_VQ_Z_CH,
  );
  const { slice: ibqScoreSlice, selectedCodeIds: selectedIbqCodeIds } = buildIbqScoreSlice(
    vqTrace.dotScores,
    vqTrace.codes,
    vqTrace.vqGridRows * vqTrace.vqGridCols,
    UNI_MINI_VQ_CODEBOOK_SIZE,
    UNI_VISUAL_INPUT_IBQ_SLICE_COLS,
  );

  const srcChildEmb = embedUniGenTokens(vqTrace.codes, ctx.bundle.core);
  const packedCodecTrace = compressPackedImageEmbeddings(
    srcChildEmb,
    UNI_MINI_CONFIG.vqGridSize,
    UNI_MINI_CONFIG.vqGridSize,
    ctx.bundle.codec,
    { isSourceEdit: true, gainPresetId },
  );

  // 3. Unified Multimodal Sequence Assembly (`[S, 8]` entering L0 backbone)
  const d = UNI_MINI_CONFIG.hiddenSize; // 8
  const prefixIds = new Int32Array([UNI_MINI_BOS_TOKEN_ID, UNI_MINI_IM_START_TOKEN_ID]);
  const prefixEmbeds = embedUniTextTokens(prefixIds, ctx.bundle.core); // [2, 8]
  const tailIds = discreteStreamActive
    ? new Int32Array([30, 26, 27]) // edit the image
    : new Int32Array([26, 27, 28]); // the image shows
  const tailEmbeds = embedUniTextTokens(tailIds, ctx.bundle.core); // [3, 8]

  const gParentCount = discreteStreamActive
    ? packedCodecTrace.parentRows * packedCodecTrace.parentCols
    : 0;
  const assembledSeqLen = prefixIds.length + rowNewlineSeqLen + gParentCount + tailIds.length;
  const assembledEmbeddings = new Float32Array(assembledSeqLen * d);
  const assembledRouting = new Uint8Array(assembledSeqLen);

  let writeRow = 0;
  // Prefix text (U)
  assembledEmbeddings.set(prefixEmbeds, writeRow * d);
  for (let i = 0; i < prefixIds.length; i++) {
    assembledRouting[writeRow + i] = UNI_ROUTE_U;
  }
  writeRow += prefixIds.length;

  // Continuous RADIO-H + row-newline block (U)
  assembledEmbeddings.set(rowNewlineEmbeddings, writeRow * d);
  for (let i = 0; i < rowNewlineSeqLen; i++) {
    assembledRouting[writeRow + i] = UNI_ROUTE_U;
  }
  writeRow += rowNewlineSeqLen;

  // Discrete PackedLinearImageCodec source-VQ parents (G) when active
  if (gParentCount > 0) {
    assembledEmbeddings.set(packedCodecTrace.output, writeRow * d);
    for (let i = 0; i < gParentCount; i++) {
      assembledRouting[writeRow + i] = UNI_ROUTE_G;
    }
    writeRow += gParentCount;
  }

  // Instruction / question tail (U)
  assembledEmbeddings.set(tailEmbeds, writeRow * d);
  for (let i = 0; i < tailIds.length; i++) {
    assembledRouting[writeRow + i] = UNI_ROUTE_U;
  }

  const assembledGCount = gParentCount;
  const assembledUCount = assembledSeqLen - assembledGCount;

  const run: UniVisualInputPipelineRun = Object.freeze({
    sourceImage,
    letterbox,
    paddingLockMask,
    gainPresetId,
    continuousStreamActive,
    discreteStreamActive,
    isTextOnlyFallback,
    visionTrace,
    vqTrace,
    packedCodecTrace,
    rowNewlineEmbeddings,
    rowNewlineTokenIds: fixedVisualBlock.expandedTokenIds,
    rowNewlineSeqLen,
    vqLatentsSpatial,
    ibqScoreSlice,
    selectedIbqCodeIds,
    assembledEmbeddings,
    assembledRouting,
    assembledSeqLen,
    assembledUCount,
    assembledGCount,
  });

  return { ctx, run };
}

// ============================================================================
// 128 × 128 RG32F SCALAR/MASK ATLAS PACKER (EXACTLY 10 NON-OVERLAPPING RECTS)
// ============================================================================

/**
 * Pack all 10 scalar/mask tensors for View 2 (`visual_input`) into a single `128 × 128`
 * `RG32F` atlas (`131,072 B` = `128.00 KiB`, zero overlaps, zero RGB channels per `D4`).
 */
export function packVisualInputAtlas(
  run: UniVisualInputPipelineRun,
): UniVisualInputAtlasPackResult {
  const W = UNI_VISUAL_INPUT_ATLAS_SIZE;
  const H = UNI_VISUAL_INPUT_ATLAS_SIZE;
  const writer = createRg32fAtlas(W, H);

  const { vision, projector } = run.visionTrace;
  const vitLastBlock = vision.blockTraces[vision.blockTraces.length - 1]!;
  const vitSeqLen = vitLastBlock.seqLen; // 20 (16x16) or 12 (16x8)
  // Live Head 0 bidirectional softmax from Block 1: `[vitSeqLen, vitSeqLen]`, G = 0.0 unmasked
  const vitAttnHead0 = vitLastBlock.attnProbs.subarray(0, vitSeqLen * vitSeqLen);

  // Normalized 4x4 quantized code grid (`R = codeId / 255`, `G = 1.0` where border-locked)
  const vqRows = run.vqTrace.vqGridRows;
  const vqCols = run.vqTrace.vqGridCols;
  const vqCellCount = vqRows * vqCols;
  const codeGridR = new Float32Array(vqCellCount);
  const codeGridG = new Float32Array(vqCellCount);
  for (let i = 0; i < vqCellCount; i++) {
    codeGridR[i] = run.vqTrace.codes[i]! / 255.0;
    codeGridG[i] = run.paddingLockMask.lockMask[i] === 1 ? 1.0 : 0.0;
  }

  // Assembled multimodal sequence `[S, 8]`: keep G = 0.0 on all active rows so SheetsPass
  // renders live Float32 embeddings for both U and G tokens (G = 1.0 would hatch cells as masked).
  // G-token rows are instead framed with a teal `flow.addCellHighlightOnFrame(...)` overlay.
  const d = UNI_MINI_CONFIG.hiddenSize;

  // Fixed non-overlapping 3-band shelf layout inside 128 × 128:
  //   Band 0 (y =  0..20): u_patches (0,0), u_vit_attn (50,0), u_patch_strip (72,0), g_vq_latents (90,0)
  //   Band 1 (y = 22..28): u_projector_gather (0,22), u_projector_newline (66,22)
  //   Band 2 (y = 30..62): g_ibq_scores (0,30), g_vq_code_grid (66,30), g_packed_codec (72,30), joint_multimodal_seq (82,30)
  const packedEntries: UniVisualInputAtlasSliceEntry[] = [];
  const sliceMap = {} as Record<UniVisualInputStageId, UniVisualInputAtlasSliceEntry>;

  const pushSlice = (id: UniVisualInputStageId, params: Omit<PackMatrixToRg32fParams, 'id'>) => {
    const packed = packMatrixToRg32f(writer, { id, ...params });
    const entry: UniVisualInputAtlasSliceEntry = Object.freeze({
      id,
      region: packed.region,
      slice: packed.slice,
    });
    packedEntries.push(entry);
    sliceMap[id] = entry;
  };

  // 1. Continuous Stage 1: 4x4 patchification (`16 × 48` or `8 × 48`)
  pushSlice('u_patches', {
    x0: 0,
    y0: 0,
    rows: vision.numPatches,
    cols: 48,
    values: vision.rawPatches,
    mode: 0,
    isMaskOrScore: false,
  });

  // 2. Continuous Stage 2: 4 [CLS/REG] + 2L ViT bidirectional softmax (`20 × 20` or `12 × 12`, G = 0.0)
  pushSlice('u_vit_attn', {
    x0: 50,
    y0: 0,
    rows: vitSeqLen,
    cols: vitSeqLen,
    values: vitAttnHead0,
    mode: 1,
    isMaskOrScore: true,
    explicitScale: 1.0,
  });

  // 3. Continuous Stage 3: Skip-token strip `[:, 4:]` patch hidden states (`16 × 16` or `8 × 16`)
  pushSlice('u_patch_strip', {
    x0: 72,
    y0: 0,
    rows: vision.numPatches,
    cols: 16,
    values: vision.patchFeatures,
    mode: 0,
    isMaskOrScore: false,
  });

  // 4. Continuous Stage 4: 2x2 RadioHProjector spatial gather (`4 × 64` or `2 × 64`)
  pushSlice('u_projector_gather', {
    x0: 0,
    y0: 22,
    rows: projector.numVisualTokens,
    cols: 64,
    values: projector.gathered2x2,
    mode: 0,
    isMaskOrScore: false,
  });

  // 5. Continuous Stage 5: RadioHProjector (D3 gain) + row-newline `-500` insertion (`6 × 8` or `3 × 8`)
  pushSlice('u_projector_newline', {
    x0: 66,
    y0: 22,
    rows: run.rowNewlineSeqLen,
    cols: d,
    values: run.rowNewlineEmbeddings,
    mode: 0,
    isMaskOrScore: false,
  });

  // 6. Discrete Stage 1: Emu3VQ `[4, 4, 8]` conv/attn encoder latents (`16 × 8`)
  pushSlice('g_vq_latents', {
    x0: 90,
    y0: 0,
    rows: vqCellCount,
    cols: UNI_MINI_VQ_Z_CH,
    values: run.vqLatentsSpatial,
    mode: 0,
    isMaskOrScore: false,
  });

  // 7. Discrete Stage 2: IBQ dot-product `argmax(z @ E^T)` codebook score slice (`16 × 64`)
  pushSlice('g_ibq_scores', {
    x0: 0,
    y0: 30,
    rows: vqCellCount,
    cols: UNI_VISUAL_INPUT_IBQ_SLICE_COLS,
    values: run.ibqScoreSlice,
    mode: 0,
    isMaskOrScore: true,
  });

  // 8. Discrete Stage 3: `4 × 4` quantized code grid & U18 paddingLockMask (`R = codeId/255`, `G = lock`)
  pushSlice('g_vq_code_grid', {
    x0: 66,
    y0: 30,
    rows: vqRows,
    cols: vqCols,
    values: codeGridR,
    mode: 1,
    isMaskOrScore: true,
    gChannelFlags: codeGridG,
    explicitScale: 1.0,
  });

  // 9. Discrete Stage 4: PackedLinearImageCodec `2×2` compression (`4 × 8`, D3 gain + 2D_SinCos)
  pushSlice('g_packed_codec', {
    x0: 72,
    y0: 30,
    rows: run.packedCodecTrace.parentRows * run.packedCodecTrace.parentCols,
    cols: d,
    values: run.packedCodecTrace.output,
    mode: 0,
    isMaskOrScore: false,
  });

  // 10. Unified Multimodal Sequence Assembly (`[S, 8]`, R = live L0 embedding, G = 0.0 active)
  pushSlice('joint_multimodal_seq', {
    x0: 82,
    y0: 30,
    rows: run.assembledSeqLen,
    cols: d,
    values: run.assembledEmbeddings,
    mode: 0,
    isMaskOrScore: false,
  });

  const regions = Object.freeze(packedEntries.map((e) => e.region));
  const audit = auditAtlasRegions(W, H, regions, UNI_VISUAL_INPUT_ATLAS_BUDGET_BYTES);

  const atlas: Rg32fValueAtlas = Object.freeze({
    width: W,
    height: H,
    data: writer.data,
  });

  const extraTag = `${run.sourceImage.presetId ?? `${run.sourceImage.width}x${run.sourceImage.height}`}:${run.gainPresetId}:${run.discreteStreamActive ? 'ug' : 'u'}`;
  const atlasKey = computeAtlasContentKey('visual_input', atlas, extraTag);

  return Object.freeze({
    atlas,
    atlasKey,
    regions,
    audit,
    slices: Object.freeze(sliceMap),
    selectedIbqCodeIds: run.selectedIbqCodeIds,
  });
}

// ============================================================================
// TYPED 3-CHANNEL RGB PREVIEW DESCRIPTORS (STRICT D4 — ZERO RGB IN RG32F)
// ============================================================================

function buildRgbPreviewDescriptors(
  run: UniVisualInputPipelineRun,
  taskMode: string,
): readonly UniRgbPreviewDescriptor[] {
  const { sourceImage, letterbox, paddingLockMask } = run;
  const srcLabel = sourceImage.label ?? `Source RGB (${sourceImage.width}×${sourceImage.height})`;
  const previews: UniRgbPreviewDescriptor[] = [
    Object.freeze({
      id: 'visual_input_source',
      kind: 'source' as const,
      label: srcLabel,
      caption: `${sourceImage.width}×${sourceImage.height} CHW RGB [0, 1] · Native Aspect (Mini RADIO-H Input)`,
      width: sourceImage.width,
      height: sourceImage.height,
      rgbChw: new Float32Array(sourceImage.data),
      honestyBadge: UNI_SCENE_VISUAL_HONESTY_BADGE,
    }),
  ];

  const isNonSquare = sourceImage.width !== sourceImage.height;
  const needsLetterboxPreview =
    isNonSquare ||
    taskMode === 'image_edit' ||
    paddingLockMask.lockedCellCount > 0;

  if (needsLetterboxPreview) {
    const info = letterbox.info;
    previews.push(
      Object.freeze({
        id: 'visual_input_letterboxed',
        kind: 'letterboxed' as const,
        label: `Letterboxed ${letterbox.image.width}×${letterbox.image.height} (Emu3VQ Input)`,
        caption: `Pad(L=${info.padLeft}, T=${info.padTop}, R=${info.padRight}, B=${info.padBottom}) · ${paddingLockMask.lockedCellCount}/16 VQ cells locked (U18)`,
        width: letterbox.image.width,
        height: letterbox.image.height,
        rgbChw: new Float32Array(letterbox.image.data),
        paddingLockMask,
        honestyBadge: UNI_SCENE_VISUAL_HONESTY_BADGE,
      }),
    );
  }

  return Object.freeze(previews);
}

// ============================================================================
// MAIN VIEW 2 SCENE BUILDER (`buildUniVisualInputScene`)
// ============================================================================

interface StagePlateSpec {
  readonly id: UniVisualInputStageId;
  readonly branch: 'u' | 'g' | 'joint';
  readonly towerRole: 'context' | 'denoiser' | 'shared';
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ez: number;
  readonly sheetKind: number;
  readonly role: number;
  readonly baseHighlight: number;
  readonly semanticTag: string;
  readonly label: string;
  readonly sublabel: string;
  readonly badge?: string;
}

const SHEET_Y = 0.08;
const SHEET_EY = 0.07;
const RIBBON_Y = 0.13;
const FRAME_LINE_Y = 0.01;
const LEFT_LANE_X = -4.9;
const RIGHT_LANE_X = 4.9;
const CENTER_X = 0.0;

function mergeAabb(target: { min: [number, number, number]; max: [number, number, number] }, box: AABB): void {
  for (let i = 0; i < 3; i++) {
    if (box.min[i]! < target.min[i]!) target.min[i] = box.min[i]!;
    if (box.max[i]! > target.max[i]!) target.max[i] = box.max[i]!;
  }
}

/**
 * Build the complete headless 3D spatial scene for **View 2 (`visual_input`)**:
 * Dual Visual Input Pipeline (`Mini RADIO-H -> U` + `Mini Emu3VQ + PackedLinearImageCodec -> G`
 * -> Unified `U + G` Multimodal Sequence Assembly) with a `128 × 128` `RG32F` scalar/mask atlas
 * (`10` non-overlapping rects, `131,072 B`) and typed `D4` RGB preview descriptors.
 */
export function buildUniVisualInputScene(
  session: UniSession,
  options?: UniVisualInputSceneOptions,
): UniSpatialScene {
  const { ctx, run } = runVisualInputPipelines(session, options);
  const packed = packVisualInputAtlas(run);
  const rgbPreviews = buildRgbPreviewDescriptors(run, ctx.branch.taskMode);
  const gainPreset = UNI_GAIN_PRESETS[run.gainPresetId];

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

  // --- Drafting-table lane plinths & input RGB canvas docking frames (D4: RGB rendered in 2D canvas) ---
  const uLineStyle = run.continuousStreamActive ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE;
  const gLineStyle = run.discreteStreamActive ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE;
  const gLineAlpha = run.discreteStreamActive ? 0.65 : 0.28;

  // Left (U) & Right (G) workbench lane frames
  lines.addBoxFrameXZ(-7.8, -8.8, -2.0, 5.2, FRAME_LINE_Y, 1.0, uLineStyle, 0.48);
  lines.addBoxFrameXZ(2.0, -8.8, 7.8, 4.6, FRAME_LINE_Y, 1.0, gLineStyle, gLineAlpha);
  // Center convergence frame
  lines.addBoxFrameXZ(-2.1, 4.9, 2.1, 9.4, FRAME_LINE_Y, 1.0, LINE_STYLE_GUIDE, 0.55);

  // Input RGB docking plinths (dashed frames indicating 2D `<canvas>` RGB insets per D4)
  lines.addDashedFrameXZ(-6.2, -8.4, -3.6, -6.8, FRAME_LINE_Y + 0.01, uLineStyle, 0.62);
  solids.push({
    cx: LEFT_LANE_X,
    cy: 0.03,
    cz: -7.6,
    ex: 2.2,
    ey: 0.06,
    ez: 1.2,
    cellFamily: CELL_WEIGHT_BLUE,
    bayId: 0,
    highlight: 0,
  });
  lines.addBoxEdges(LEFT_LANE_X, 0.03, -7.6, 2.2, 0.06, 1.2, 1, 0.9, LINE_STYLE_NEUTRAL, 0.55);

  lines.addDashedFrameXZ(3.6, -8.4, 6.2, -6.8, FRAME_LINE_Y + 0.01, gLineStyle, gLineAlpha);
  solids.push({
    cx: RIGHT_LANE_X,
    cy: 0.03,
    cz: -7.6,
    ex: 2.2,
    ey: 0.06,
    ez: 1.2,
    cellFamily: run.discreteStreamActive ? CELL_SSM_CYAN : CELL_NEUTRAL,
    bayId: 0,
    highlight: 0,
  });
  lines.addBoxEdges(
    RIGHT_LANE_X,
    0.03,
    -7.6,
    2.2,
    0.06,
    1.2,
    2,
    0.9,
    LINE_STYLE_NEUTRAL,
    run.discreteStreamActive ? 0.55 : 0.28,
  );

  // L0 Backbone Input Rail solid at the front (+Z)
  const railZ = 8.8;
  solids.push({
    cx: CENTER_X,
    cy: 0.07,
    cz: railZ,
    ex: 3.6,
    ey: 0.14,
    ez: 0.48,
    cellFamily: run.discreteStreamActive ? CELL_ACTIVATION_GREEN : CELL_EMBED_SLATE,
    bayId: 0,
    highlight: 0,
  });
  lines.addBoxEdges(CENTER_X, 0.07, railZ, 3.6, 0.14, 0.48, 3, 1.0, LINE_STYLE_NEUTRAL, 0.72);
  mergeAabb(boundsAcc, { min: [-7.9, -0.1, -8.9], max: [7.9, 0.8, 9.5] });

  // --- Define all 10 stage plates (5 Continuous U + 4 Discrete G + 1 Unified U+G) ---
  const is16x8 = run.sourceImage.height < run.sourceImage.width;
  const gHl = run.discreteStreamActive ? SHEET_HL_OPERAND_B : SHEET_HL_DIMMED;
  const uHl = run.continuousStreamActive ? SHEET_HL_OPERAND_A : SHEET_HL_NORMAL;

  const { vision, projector } = run.visionTrace;
  const vitSeqLen = vision.blockTraces[vision.blockTraces.length - 1]!.seqLen;

  const plateSpecs: readonly StagePlateSpec[] = [
    // Continuous Stream (U Indigo, Left Lane)
    {
      id: 'u_patches',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_LANE_X,
      cy: SHEET_Y,
      cz: -5.4,
      ex: 4.8,
      ez: is16x8 ? 1.15 : 1.65,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      baseHighlight: uHl,
      semanticTag: 'branch=u;stream=continuous;stage=patchify;op=clip_4x4_patches',
      label: `4×4 Patchification (${vision.numPatches}×48)`,
      sublabel: 'CLIP-normalized CHW 4×4 patches (native aspect)',
    },
    {
      id: 'u_vit_attn',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_LANE_X,
      cy: SHEET_Y,
      cz: -2.6,
      ex: is16x8 ? 2.1 : 2.6,
      ez: is16x8 ? 2.1 : 2.6,
      sheetKind: SHEET_KIND_CAUSAL_SOFTMAX, // Live RG32F G=0.0 bidirectional (strict U25!)
      role: SHEET_ROLE_SOFTMAX,
      baseHighlight: uHl,
      semanticTag: 'branch=u;stream=continuous;stage=vit_attn;op=radio_2L_bidirectional;u25=live_rg32f',
      label: `RADIO-H ViT Attn (${vitSeqLen}×${vitSeqLen})`,
      sublabel: '4 [CLS/REG] + 2L ViT · Unmasked Bidirectional Softmax (G=0)',
      badge: 'U25 Live RG32F',
    },
    {
      id: 'u_patch_strip',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_LANE_X,
      cy: SHEET_Y,
      cz: 0.1,
      ex: 2.3,
      ez: is16x8 ? 1.25 : 1.85,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      baseHighlight: uHl,
      semanticTag: 'branch=u;stream=continuous;stage=skip_strip;op=slice_4_to_end',
      label: `Skip-Token Strip [:, 4:] (${vision.numPatches}×16)`,
      sublabel: 'Drops 2 CLS + 2 REG prefix tokens before 2×2 merge',
    },
    {
      id: 'u_projector_gather',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_LANE_X,
      cy: SHEET_Y,
      cz: 2.0,
      ex: 4.8,
      ez: is16x8 ? 0.72 : 0.95,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_NORM,
      baseHighlight: uHl,
      semanticTag: 'branch=u;stream=continuous;stage=projector_gather;op=2x2_ps_v2_gather',
      label: `2×2 RadioHProjector Gather (${projector.numVisualTokens}×64)`,
      sublabel: 'ps_version=v2 spatial concat -> RMSNorm(64) -> SquaredReLU',
    },
    {
      id: 'u_projector_newline',
      branch: 'u',
      towerRole: 'context',
      cx: LEFT_LANE_X,
      cy: SHEET_Y,
      cz: 3.9,
      ex: 1.9,
      ez: is16x8 ? 0.95 : 1.35,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_RESIDUAL,
      baseHighlight: uHl,
      semanticTag: `branch=u;stream=continuous;stage=projector_newline;op=fixOnePrompt_u8;gain=${run.gainPresetId}`,
      label: `RadioHProjector + Row-Newline (${run.rowNewlineSeqLen}×8)`,
      sublabel: `D3 ×${gainPreset.radioProjectorMultiply} + literal \\n (fixOnePrompt · U8 image_newline unused)`,
      badge: 'U8: image_newline unused',
    },

    // Discrete Stream (G Teal, Right Lane)
    {
      id: 'g_vq_latents',
      branch: 'g',
      towerRole: 'denoiser',
      cx: RIGHT_LANE_X,
      cy: SHEET_Y,
      cz: -5.4,
      ex: 2.1,
      ez: 1.95,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      baseHighlight: gHl,
      semanticTag: `branch=g;stream=discrete;stage=vq_latents;op=emu3_conv_attn_enc;active=${run.discreteStreamActive ? 1 : 0}`,
      label: 'Emu3VQ Encoder Latents (16×8)',
      sublabel: '[4, 4, 8] GroupNorm + SiLU + AttnBlock -> quant_conv',
      badge: run.discreteStreamActive ? 'G Active' : 'Inactive (Dimmed)',
    },
    {
      id: 'g_ibq_scores',
      branch: 'g',
      towerRole: 'denoiser',
      cx: RIGHT_LANE_X,
      cy: SHEET_Y,
      cz: -2.4,
      ex: 4.8,
      ez: 1.85,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SCORES,
      baseHighlight: gHl,
      semanticTag: `branch=g;stream=discrete;stage=ibq_scores;op=argmax_z_Et_u14;active=${run.discreteStreamActive ? 1 : 0}`,
      label: 'IBQ Dot-Product Scores (16×64)',
      sublabel: 'argmax(z @ E^T) over top/winning 64 of 256 codes (U14)',
      badge: 'U14: argmax(z @ E^T)',
    },
    {
      id: 'g_vq_code_grid',
      branch: 'g',
      towerRole: 'denoiser',
      cx: RIGHT_LANE_X,
      cy: SHEET_Y,
      cz: 0.4,
      ex: 1.9,
      ez: 1.9,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SOFTMAX,
      baseHighlight: gHl,
      semanticTag: `branch=g;stream=discrete;stage=vq_code_grid;op=u18_padding_lock;locked=${run.paddingLockMask.lockedCellCount}`,
      label: '4×4 Quantized Code Grid + U18 Lock',
      sublabel: `R = codeId/255 · G = paddingLockMask (${run.paddingLockMask.lockedCellCount}/16 locked)`,
      badge:
        run.paddingLockMask.lockedCellCount > 0
          ? `U18: ${run.paddingLockMask.lockedCellCount}/16 Locked`
          : 'U18: 0/16 Locked',
    },
    {
      id: 'g_packed_codec',
      branch: 'g',
      towerRole: 'denoiser',
      cx: RIGHT_LANE_X,
      cy: SHEET_Y,
      cz: 3.1,
      ex: 2.0,
      ez: 1.25,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_RESIDUAL,
      baseHighlight: gHl,
      semanticTag: `branch=g;stream=discrete;stage=packed_codec;op=2x2_compress_sincos;gain=${run.gainPresetId}`,
      label: 'PackedLinearImageCodec 2×2 (4×8)',
      sublabel: `D3 ×${gainPreset.combinedGenerationScale.toPrecision(4)} + 2D_SinCos (marker_col=false)`,
      badge: `D3 ${run.gainPresetId}`,
    },

    // Unified Multimodal Sequence Assembly (U + G -> L0 Backbone)
    {
      id: 'joint_multimodal_seq',
      branch: 'joint',
      towerRole: 'shared',
      cx: CENTER_X,
      cy: SHEET_Y,
      cz: 6.5,
      ex: 2.8,
      ez: 2.5,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_NEUTRAL,
      baseHighlight: SHEET_HL_PATHWAY,
      semanticTag: `branch=joint;stream=unified;stage=multimodal_assembly;u=${run.assembledUCount};g=${run.assembledGCount}`,
      label: `Unified Multimodal Sequence (${run.assembledSeqLen}×8)`,
      sublabel:
        run.assembledGCount > 0
          ? `R = Live L0 Embedding (${run.assembledUCount} U + ${run.assembledGCount} G · G band framed in teal)`
          : `R = Live L0 Embedding (${run.assembledUCount} U-only)`,
      badge: run.assembledGCount > 0 ? 'Mixed U + G -> L0' : 'U-Only -> L0',
    },
  ];

  const hovered = options?.hoveredCell ?? null;
  const stageFrames = {} as Record<UniVisualInputStageId, SheetFrame3D>;

  for (let i = 0; i < plateSpecs.length; i++) {
    const spec = plateSpecs[i]!;
    const sliceEntry = packed.slices[spec.id];
    const gridCols = sliceEntry.region.cols;
    const gridRows = sliceEntry.region.rows;

    const frame = makeSheetFrame3D(
      [spec.cx, spec.cy, spec.cz],
      [spec.ex, SHEET_EY, spec.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    stageFrames[spec.id] = frame;
    mergeAabb(boundsAcc, sheetFrameWorldAabb(frame));

    const isHoveredPlate =
      hovered !== null &&
      (hovered.tensorKey === spec.id || hovered.partId === spec.id);

    const highlight = isHoveredPlate ? SHEET_HL_ACTIVE : spec.baseHighlight;

    sheets.push({
      cx: spec.cx,
      cy: spec.cy,
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
      seed: i * 0.71,
      slice: sliceEntry.slice,
      frame,
    });

    // Teal framing band over contiguous G-routed rows on `joint_multimodal_seq` (keeping G = 0.0
    // in RG32F so SheetsPass renders live Float32 embeddings without cross-hatching them as masked)
    if (spec.id === 'joint_multimodal_seq' && run.assembledGCount > 0) {
      let row = 0;
      while (row < run.assembledSeqLen) {
        if (run.assembledRouting[row] === UNI_ROUTE_G) {
          const runStart = row;
          while (
            row + 1 < run.assembledSeqLen &&
            run.assembledRouting[row + 1] === UNI_ROUTE_G
          ) {
            row += 1;
          }
          flow.addCellHighlightOnFrame(
            frame,
            runStart,
            row,
            0,
            gridCols - 1,
            gridRows,
            gridCols,
            SHEET_EY * 0.5 + 0.012,
            { r: 0.102, g: 0.541, b: 0.478, a: 0.16, borderAlpha: 0.9 },
            UNI_FLOW_COLORS.gTeal,
            0.024,
            true,
          );
        }
        row += 1;
      }
    }

    // Cell hover highlight overlay when a valid cell is hovered on this plate
    if (
      isHoveredPlate &&
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
        SHEET_EY * 0.5 + 0.015,
        { r: 0.863, g: 0.298, b: 0.204, a: 0.34, borderAlpha: 0.95 },
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
        towerRole: spec.towerRole,
        semanticTag: spec.semanticTag,
      }),
    );

    overlayAnchors.push(
      Object.freeze({
        id: `anchor_${spec.id}`,
        label: spec.label,
        sublabel: spec.sublabel,
        worldPos: [spec.cx, spec.cy + 0.22, spec.cz - spec.ez * 0.5 - 0.16] as Vec3,
        tone:
          spec.branch === 'u'
            ? 'u_indigo'
            : spec.branch === 'g'
              ? run.discreteStreamActive
                ? 'g_teal'
                : 'muted'
              : 'joint',
        ...(spec.badge !== undefined ? { badge: spec.badge } : {}),
      }),
    );
  }

  // --- Smooth FlatFlowRibbons connecting each stage and converging into L0 sequence ---
  const uRibbonColor = run.continuousStreamActive
    ? UNI_FLOW_COLORS.uIndigo
    : UNI_FLOW_COLORS.ghostSlate;
  const gRibbonColor = run.discreteStreamActive
    ? UNI_FLOW_COLORS.gTeal
    : UNI_FLOW_COLORS.ghostSlate;
  const jointRibbonColor = run.discreteStreamActive
    ? UNI_FLOW_COLORS.jointCyan
    : UNI_FLOW_COLORS.uIndigo;

  // Left lane sequential flow (U)
  const xz = (x: number, z: number) => ({ x, z });

  flow.addPolylineXZ(
    [
      xz(LEFT_LANE_X, -7.0),
      xz(LEFT_LANE_X, stageFrames.u_patches.center[2] - stageFrames.u_patches.extent[2] * 0.5 - 0.08),
    ],
    RIBBON_Y,
    0.16,
    uRibbonColor,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(LEFT_LANE_X, stageFrames.u_patches.center[2] + stageFrames.u_patches.extent[2] * 0.5 + 0.06),
      xz(LEFT_LANE_X, stageFrames.u_vit_attn.center[2] - stageFrames.u_vit_attn.extent[2] * 0.5 - 0.08),
    ],
    RIBBON_Y,
    0.16,
    uRibbonColor,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(LEFT_LANE_X, stageFrames.u_vit_attn.center[2] + stageFrames.u_vit_attn.extent[2] * 0.5 + 0.06),
      xz(
        LEFT_LANE_X,
        stageFrames.u_patch_strip.center[2] - stageFrames.u_patch_strip.extent[2] * 0.5 - 0.08,
      ),
    ],
    RIBBON_Y,
    0.16,
    uRibbonColor,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(
        LEFT_LANE_X,
        stageFrames.u_patch_strip.center[2] + stageFrames.u_patch_strip.extent[2] * 0.5 + 0.06,
      ),
      xz(
        LEFT_LANE_X,
        stageFrames.u_projector_gather.center[2] -
          stageFrames.u_projector_gather.extent[2] * 0.5 -
          0.08,
      ),
    ],
    RIBBON_Y,
    0.16,
    uRibbonColor,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(
        LEFT_LANE_X,
        stageFrames.u_projector_gather.center[2] +
          stageFrames.u_projector_gather.extent[2] * 0.5 +
          0.06,
      ),
      xz(
        LEFT_LANE_X,
        stageFrames.u_projector_newline.center[2] -
          stageFrames.u_projector_newline.extent[2] * 0.5 -
          0.08,
      ),
    ],
    RIBBON_Y,
    0.16,
    uRibbonColor,
    true,
  );

  // Right lane sequential flow (G)
  flow.addPolylineXZ(
    [
      xz(RIGHT_LANE_X, -7.0),
      xz(
        RIGHT_LANE_X,
        stageFrames.g_vq_latents.center[2] - stageFrames.g_vq_latents.extent[2] * 0.5 - 0.08,
      ),
    ],
    RIBBON_Y,
    0.16,
    gRibbonColor,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(
        RIGHT_LANE_X,
        stageFrames.g_vq_latents.center[2] + stageFrames.g_vq_latents.extent[2] * 0.5 + 0.06,
      ),
      xz(
        RIGHT_LANE_X,
        stageFrames.g_ibq_scores.center[2] - stageFrames.g_ibq_scores.extent[2] * 0.5 - 0.08,
      ),
    ],
    RIBBON_Y,
    0.16,
    gRibbonColor,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(
        RIGHT_LANE_X,
        stageFrames.g_ibq_scores.center[2] + stageFrames.g_ibq_scores.extent[2] * 0.5 + 0.06,
      ),
      xz(
        RIGHT_LANE_X,
        stageFrames.g_vq_code_grid.center[2] - stageFrames.g_vq_code_grid.extent[2] * 0.5 - 0.08,
      ),
    ],
    RIBBON_Y,
    0.16,
    gRibbonColor,
    true,
  );
  flow.addPolylineXZ(
    [
      xz(
        RIGHT_LANE_X,
        stageFrames.g_vq_code_grid.center[2] + stageFrames.g_vq_code_grid.extent[2] * 0.5 + 0.06,
      ),
      xz(
        RIGHT_LANE_X,
        stageFrames.g_packed_codec.center[2] - stageFrames.g_packed_codec.extent[2] * 0.5 - 0.08,
      ),
    ],
    RIBBON_Y,
    0.16,
    gRibbonColor,
    true,
  );

  // Smooth cubic convergence ribbons from both streams into Unified Multimodal Sequence
  const seqTopZ =
    stageFrames.joint_multimodal_seq.center[2] -
    stageFrames.joint_multimodal_seq.extent[2] * 0.5 -
    0.08;
  const uExitZ =
    stageFrames.u_projector_newline.center[2] +
    stageFrames.u_projector_newline.extent[2] * 0.5 +
    0.06;
  const gExitZ =
    stageFrames.g_packed_codec.center[2] + stageFrames.g_packed_codec.extent[2] * 0.5 + 0.06;

  flow.addCubicXZ(
    xz(LEFT_LANE_X, uExitZ),
    xz(LEFT_LANE_X, seqTopZ - 0.35),
    xz(-0.75, uExitZ + 0.35),
    xz(-0.75, seqTopZ),
    RIBBON_Y,
    0.18,
    uRibbonColor,
    true,
    24,
  );

  flow.addCubicXZ(
    xz(RIGHT_LANE_X, gExitZ),
    xz(RIGHT_LANE_X, seqTopZ - 0.25),
    xz(0.75, gExitZ + 0.55),
    xz(0.75, seqTopZ),
    RIBBON_Y,
    0.18,
    gRibbonColor,
    true,
    24,
  );

  // Unified sequence -> L0 Backbone Input Rail
  const seqBottomZ =
    stageFrames.joint_multimodal_seq.center[2] +
    stageFrames.joint_multimodal_seq.extent[2] * 0.5 +
    0.06;
  flow.addPolylineXZ(
    [
      xz(CENTER_X, seqBottomZ),
      xz(CENTER_X, railZ - 0.32),
    ],
    RIBBON_Y,
    0.22,
    jointRibbonColor,
    true,
  );

  // --- Lane Header Overlay Anchors, Landmarks & Dimension Calipers ---
  overlayAnchors.push(
    Object.freeze({
      id: 'lane_continuous_header',
      label: 'Continuous Stream · Mini RADIO-H -> U',
      sublabel: `${run.sourceImage.width}×${run.sourceImage.height} Native RGB -> ${projector.numVisualTokens} U visual tokens`,
      worldPos: [LEFT_LANE_X, 0.34, -8.95] as Vec3,
      tone: 'u_indigo',
      badge: run.continuousStreamActive ? 'Active U Lane' : 'Preview U Lane',
    }),
    Object.freeze({
      id: 'lane_discrete_header',
      label: 'Discrete Stream · Mini Emu3VQ + Codec -> G',
      sublabel: `${run.letterbox.image.width}×${run.letterbox.image.height} Letterbox -> 4×4 IBQ -> 2×2 Packed G Parents`,
      worldPos: [RIGHT_LANE_X, 0.34, -8.95] as Vec3,
      tone: run.discreteStreamActive ? 'g_teal' : 'muted',
      badge: run.discreteStreamActive ? 'Active G Lane' : 'Inactive · Dimmed',
    }),
    Object.freeze({
      id: 'backbone_l0_rail',
      label: 'L0 Elastic-MoT Backbone Input Rail',
      sublabel: `${run.assembledSeqLen} tokens (${run.assembledUCount} U / ${run.assembledGCount} G) · D3 ${run.gainPresetId}`,
      worldPos: [CENTER_X, 0.34, railZ + 0.45] as Vec3,
      tone: 'joint',
    }),
  );

  landmarks.push(
    Object.freeze({
      id: 'lm_continuous_stream',
      label: 'RADIO-H Continuous Stream (U)',
      sublabel: `${vision.numPatches} patches -> ${projector.numVisualTokens} tokens + ${projector.tokenGridRows} \\n`,
      world: [LEFT_LANE_X, 0.45, -8.8] as Vec3,
      tone: 'default',
      align: 'top',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_discrete_stream',
      label: 'Emu3VQ + PackedLinearImageCodec (G)',
      sublabel: run.discreteStreamActive
        ? `16 IBQ codes -> 4 packed G parents (${run.paddingLockMask.lockedCellCount} locked)`
        : 'Inactive in current task mode (dimmed)',
      world: [RIGHT_LANE_X, 0.45, -8.8] as Vec3,
      tone: run.discreteStreamActive ? 'cyan' : 'muted',
      align: 'top',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_multimodal_assembly',
      label: 'Unified Multimodal Sequence -> L0',
      sublabel: `[${run.assembledSeqLen}, 8] · U=${run.assembledUCount} / G=${run.assembledGCount}`,
      world: [CENTER_X, 0.45, railZ + 0.55] as Vec3,
      tone: 'gold',
      align: 'bottom',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'cal_u_patches',
      aWorld: [
        LEFT_LANE_X - stageFrames.u_patches.extent[0] * 0.5,
        SHEET_Y + 0.06,
        stageFrames.u_patches.center[2] - stageFrames.u_patches.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      bWorld: [
        LEFT_LANE_X + stageFrames.u_patches.extent[0] * 0.5,
        SHEET_Y + 0.06,
        stageFrames.u_patches.center[2] - stageFrames.u_patches.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      label: `${vision.numPatches}×48 patches`,
      partIds: ['u_patches'],
    }),
    Object.freeze({
      id: 'cal_g_ibq',
      aWorld: [
        RIGHT_LANE_X - stageFrames.g_ibq_scores.extent[0] * 0.5,
        SHEET_Y + 0.06,
        stageFrames.g_ibq_scores.center[2] - stageFrames.g_ibq_scores.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      bWorld: [
        RIGHT_LANE_X + stageFrames.g_ibq_scores.extent[0] * 0.5,
        SHEET_Y + 0.06,
        stageFrames.g_ibq_scores.center[2] - stageFrames.g_ibq_scores.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      label: '16×64 IBQ z@E^T',
      partIds: ['g_ibq_scores'],
    }),
    Object.freeze({
      id: 'cal_joint_seq',
      aWorld: [
        CENTER_X + stageFrames.joint_multimodal_seq.extent[0] * 0.5 + 0.18,
        SHEET_Y + 0.06,
        stageFrames.joint_multimodal_seq.center[2] -
          stageFrames.joint_multimodal_seq.extent[2] * 0.5,
      ] as Vec3,
      bWorld: [
        CENTER_X + stageFrames.joint_multimodal_seq.extent[0] * 0.5 + 0.18,
        SHEET_Y + 0.06,
        stageFrames.joint_multimodal_seq.center[2] +
          stageFrames.joint_multimodal_seq.extent[2] * 0.5,
      ] as Vec3,
      label: `S=${run.assembledSeqLen}×8`,
      partIds: ['joint_multimodal_seq'],
    }),
  );

  // --- Provenance Badges (Effective D3 Gain, U8, D4, U18, Text-Only Fallback) ---
  const provenanceBadges: UniProvenanceBadge[] = ctx.provenanceBadges.map((b) => {
    if (b.label.startsWith('D3 Gains:')) {
      return Object.freeze({
        label:
          run.gainPresetId === 'prod_exact'
            ? 'D3 Gains: Production-Exact (0.01 / 0.00241)'
            : 'D3 Gains: Mini Default (0.25 / 0.50)',
        tone: 'muted' as const,
      });
    }
    return b;
  });

  provenanceBadges.push(
    Object.freeze({
      label: UNI_SCENE_VISUAL_HONESTY_BADGE,
      tone: 'muted' as const,
    }),
    Object.freeze({
      label: 'U8: encoder.image_newline Unused on Active Path (Literal \\n via fixOnePrompt)',
      tone: 'warn' as const,
    }),
    Object.freeze({
      label: 'D4: 3-Ch RGB via 2D Canvas · 128×128 RG32F Scalar/Mask Only',
      tone: 'muted' as const,
    }),
  );

  if (run.paddingLockMask.lockedCellCount > 0) {
    provenanceBadges.push(
      Object.freeze({
        label: `U18: ${run.paddingLockMask.lockedCellCount}/16 Letterbox VQ Cells Border-Locked`,
        tone: 'g_teal' as const,
      }),
    );
  }

  if (run.isTextOnlyFallback) {
    provenanceBadges.push(
      Object.freeze({
        label: 'Text-Only Session · Previewing Default Visual Input Preset',
        tone: 'muted' as const,
      }),
    );
  }

  if (!run.discreteStreamActive) {
    provenanceBadges.push(
      Object.freeze({
        label: `Discrete Emu3VQ Stream Inactive in ${ctx.viewReadout.label} (Dimmed)`,
        tone: 'muted' as const,
      }),
    );
  }

  // --- Pack 5-pass GPU buffers & assemble UniSpatialScene ---
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
    viewMode: 'visual_input' as const,
    title: 'View 2 · Dual Visual Input Pipeline (RADIO-H & Emu3VQ)',
    subtitle: `${run.sourceImage.label ?? `${run.sourceImage.width}×${run.sourceImage.height}`} · ${gainPreset.label} · ${packed.regions.length} RG32F rects (${packed.audit.utilizationPct.toFixed(1)}%)`,
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
      azimuthDeg: 18,
      elevationDeg: 44,
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
      `Invalid View 2 (visual_input) scene geometry: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
