/**
 * Milestone UNI5 — Slice 5A · View 4 (`ball_tree`): Codebook Geometry & Ball-Tree Inspector
 * (`src/uni/scene/ballTreeScene.ts`)
 *
 * Clean-room headless 3D spatial scene builder for Nemotron-Labs-Diffusion-Uni:
 *   - 3D PCA point cloud of all `256` miniature Emu3 VQ codebook entries positioned via
 *     deterministic Jacobi PCA (`projectCodebookPca3d`) and shaded by `BalancedSemanticTree`
 *     leaf order (`tree.tokenToLeaf[codeId] ∈ 0..255`).
 *   - Overlaid `LinesPass` wireframe bounding shells & clickable depth-hierarchy rings
 *     (`depth = 0..8`) reflecting the `255` monotone `tree.heapRadii` and highlighting the
 *     exact nested power-of-two candidate balls `N_l(c_hat)` across `refinementSteps ∈ {1,2,4,8}`
 *     and `stageIndex ∈ {-1, 0..N}` (`256 -> 64 -> 16 -> 4 -> 1` at `N = 4`;
 *     `256 -> 1` binary MDM reduction at `N = 1`).
 *   - `128 × 128` `RG32F` scalar/mask atlas (`131,072 B` = `128.00 KiB`, EXACTLY `5`
 *     non-overlapping rects, `0` overlaps, `rgbPreviews: []` per `D4`):
 *       1. `codebook_lo` (`128 × 8`, entries `0..127`, `G = 0.0`)
 *       2. `codebook_hi` (`128 × 8`, entries `128..255`, `G = 0.0`)
 *       3. `heap_radii` (`16 × 16`, `255` monotone radii with `G = 0.0`, slot `255` masked `G = 1.0`)
 *       4. `ball_membership` (`64 × 16`, nested candidate-ball funnel + proximity + `U26` strips;
 *          live scalar cells `G = 0.0`, excluded out-of-ball slots `G = 1.0`)
 *       5. `sincos_2d` (`8 × 24`, `compute2dSinCosPosEmbed` for `4×6` marked grid,
 *          `canonicalContentSize = 32`, `scale = 0.0018`, `markerColumn = true`, `G = 0.0`)
 *   - Interactive `U26` production `D = 17` train-vs-inference floor-rounding mismatch stage,
 *     provenance badges, and overlay anchors contrasting miniature `depth = 8` (`0` mismatches)
 *     against production `D = 17` (`trainLevel = [13, 9, 5]` -> `16, 256, 4096` vs
 *     `inferLevel = [12, 8, 4]` -> `32, 512, 8192` at `K = 4`, `2×` wider subtrees).
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import type { Vec3 } from '../../model/spec';
import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
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
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_WEIGHT,
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
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../scene/spine';
import {
  buildUniElasticD5Comparison,
  buildUniViewArchReadouts,
} from '../archReadouts';
import { compute2dSinCosPosEmbed } from '../rope';
import type { UniSession } from '../session';
import {
  analyzePrimeDepthFloorMismatch,
  buildBalancedSemanticTree,
  getStagePolarityPair,
  getTreeBallMembership,
  projectCodebookPca3d,
  semanticTreeLevelsForAges,
} from '../tree';
import {
  UNI_MINI_CONFIG,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_MULTIMODAL_SEED,
  type UniBalancedSemanticTree,
  type UniElasticPrefillMode,
  type UniExecutionViewId,
  type UniPrimeDepthFloorMismatchReport,
  type UniStagePolarityPair,
  type UniTreeBallMembership,
} from '../types';
import { createUniEmu3VqWeights } from '../vqvae';
import {
  auditAtlasRegions,
  computeAtlasContentKey,
  packPickBoxFromAabb,
  packPickBoxFromFrame,
  resolveUniSessionContext,
  UNI_FLOW_COLORS,
  UNI_SCENE_PROJECTION_REVERSAL_BADGE,
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
  UniBallTreeSceneOptions,
  UniOverlayAnchor,
  UniProvenanceBadge,
  UniRefinementStepPreset,
  UniSpatialScene,
} from './types';

// ============================================================================
// CONSTANTS & EXPORTED VIEW 4 TYPES
// ============================================================================

export const UNI_BALL_TREE_ATLAS_SIZE = 128 as const;
export const UNI_BALL_TREE_ATLAS_BUDGET_BYTES = 131_072 as const; // 128 * 128 * 8 B
export const UNI_BALL_TREE_REGION_COUNT = 5 as const;
export const UNI_BALL_TREE_PROD_DEPTH = 17 as const;
export const UNI_BALL_TREE_DEFAULT_CODE_ID = 42 as const;

export type UniBallTreeRegionId =
  | 'codebook_lo'
  | 'codebook_hi'
  | 'heap_radii'
  | 'ball_membership'
  | 'sincos_2d';

export interface UniBallTreeAtlasSliceEntry {
  readonly id: UniBallTreeRegionId;
  readonly region: UniAtlasRegionBox;
  readonly slice: AtlasSlice;
}

export interface UniBallTreeAtlasPackResult {
  readonly atlas: Rg32fValueAtlas;
  readonly atlasKey: string;
  readonly regions: readonly UniAtlasRegionBox[];
  readonly audit: UniAtlasLayoutAudit;
  readonly slices: Readonly<Record<UniBallTreeRegionId, UniBallTreeAtlasSliceEntry>>;
}

interface DefaultCodebookCache {
  readonly codebook: Float32Array;
  readonly tree: UniBalancedSemanticTree;
  readonly pca3d: Float32Array;
  readonly sinCos2d: Float32Array;
}

interface InternalRg32fRegionParams {
  readonly id: UniBallTreeRegionId;
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

interface DepthShellGeometry {
  readonly depth: number;
  readonly membership: UniTreeBallMembership;
  readonly centroid: Vec3;
  readonly radius3d: number;
  readonly pedestalCenter: Vec3;
  readonly isScheduledLevel: boolean;
  readonly isActiveLevel: boolean;
}

// ============================================================================
// LAZY DETERMINISTIC STANDALONE FALLBACK CACHE (ZERO `Math.random()`)
// ============================================================================

let cachedDefaultFallback: DefaultCodebookCache | null = null;
const cachedPcaByCodebook = new WeakMap<Float32Array, Float32Array>();

function getDefaultCodebookAndTree(): DefaultCodebookCache {
  if (cachedDefaultFallback === null) {
    const vqvae = createUniEmu3VqWeights(UNI_MULTIMODAL_SEED);
    const tree = buildBalancedSemanticTree(
      vqvae.codebook,
      UNI_MINI_VQ_CODEBOOK_SIZE,
      UNI_MINI_CONFIG.vqLatentDim,
    );
    const pca3d = projectCodebookPca3d(
      vqvae.codebook,
      UNI_MINI_VQ_CODEBOOK_SIZE,
      UNI_MINI_CONFIG.vqLatentDim,
    );
    const sinCos = compute2dSinCosPosEmbed(
      UNI_MINI_CONFIG.vqGridSize,
      UNI_MINI_CONFIG.vqMarkedGridCols,
      UNI_MINI_CONFIG.hiddenSize,
      true,
      UNI_MINI_CONFIG.sinCos2d,
    );
    cachedPcaByCodebook.set(vqvae.codebook, pca3d);
    cachedDefaultFallback = Object.freeze({
      codebook: vqvae.codebook,
      tree,
      pca3d,
      sinCos2d: sinCos.embeddings,
    });
  }
  return cachedDefaultFallback;
}

function getCachedPca3d(codebook: Float32Array): Float32Array {
  const existing = cachedPcaByCodebook.get(codebook);
  if (existing) return existing;
  const computed = projectCodebookPca3d(
    codebook,
    UNI_MINI_VQ_CODEBOOK_SIZE,
    UNI_MINI_CONFIG.vqLatentDim,
  );
  cachedPcaByCodebook.set(codebook, computed);
  return computed;
}

// ============================================================================
// INTERNAL HELPERS (NON-EXPORTED RG32F PACKER — NO EXPORT NAME COLLISION)
// ============================================================================

function clampInt(value: number, min: number, max: number): number {
  const n = Math.trunc(value);
  if (!Number.isFinite(n)) return min;
  return n < min ? min : n > max ? max : n;
}

function normalizeRefinementSteps(steps: number | undefined): UniRefinementStepPreset {
  if (steps === 1 || steps === 2 || steps === 4 || steps === 8) {
    return steps;
  }
  return UNI_MINI_CONFIG.defaultRefinementSteps;
}

function expandAabb(acc: { min: [number, number, number]; max: [number, number, number] }, box: AABB): void {
  for (let axis = 0; axis < 3; axis++) {
    if (box.min[axis]! < acc.min[axis]!) acc.min[axis] = box.min[axis]!;
    if (box.max[axis]! > acc.max[axis]!) acc.max[axis] = box.max[axis]!;
  }
}

function expandAabbPoint(
  acc: { min: [number, number, number]; max: [number, number, number] },
  x: number,
  y: number,
  z: number,
  pad = 0,
): void {
  if (x - pad < acc.min[0]) acc.min[0] = x - pad;
  if (y - pad < acc.min[1]) acc.min[1] = y - pad;
  if (z - pad < acc.min[2]) acc.min[2] = z - pad;
  if (x + pad > acc.max[0]) acc.max[0] = x + pad;
  if (y + pad > acc.max[1]) acc.max[1] = y + pad;
  if (z + pad > acc.max[2]) acc.max[2] = z + pad;
}

/**
 * Internal non-exported RG32F region packer enforcing strict `SheetsPass` `G`-channel rules:
 *   - `G = 0.0` on ALL live scalar cells (so `SheetsPass` renders `texVal.r` faithfully)
 *   - `G = 1.0` ONLY on genuinely masked/unused/excluded cells (`gChannelFlags[idx] > 0.5`)
 */
function writeBallTreeRegionToRg32f(
  writer: Rg32fAtlasWriter,
  params: InternalRg32fRegionParams,
): UniBallTreeAtlasSliceEntry {
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
    throw new RangeError(`Invalid ball_tree RG32F region "${id}": (${x0},${y0}) ${cols}x${rows}`);
  }
  if (x0 < 0 || y0 < 0 || x0 + cols > writer.width || y0 + rows > writer.height) {
    throw new RangeError(
      `ball_tree RG32F region "${id}" (${x0},${y0} + ${cols}x${rows}) exceeds ${writer.width}x${writer.height}`,
    );
  }

  const cellCount = rows * cols;
  if (values.length < cellCount) {
    throw new RangeError(
      `ball_tree RG32F region "${id}" values length (${values.length}) < ${cellCount}`,
    );
  }

  let scale = explicitScale;
  if (scale === undefined) {
    if (mode === 1) {
      scale = 1.0;
    } else {
      const buf =
        values instanceof Float32Array
          ? values.subarray(0, cellCount)
          : Float32Array.from(values);
      scale = 1.0 / computeMaxAbs(buf, 1e-6);
    }
  }

  for (let r = 0; r < rows; r++) {
    const srcRow = r * cols;
    const dstY = y0 + r;
    for (let c = 0; c < cols; c++) {
      const idx = srcRow + c;
      const rawVal = values[idx]!;
      const masked = gChannelFlags !== undefined && gChannelFlags[idx]! > 0.5;
      if (masked) {
        writer.setTexel(x0 + c, dstY, 0.0, true);
      } else {
        writer.setTexel(x0 + c, dstY, Number.isFinite(rawVal) ? rawVal : 0.0, false);
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

  return Object.freeze({ id, region, slice });
}

/** Compute unit-normalized codebook rows in FP64 for exact squared-L2 leaf-group distances. */
function normalizeCodebookUnitRows(
  codebook: Float32Array,
  vocabSize: number,
  latentDim: number,
): Float64Array {
  const out = new Float64Array(vocabSize * latentDim);
  for (let i = 0; i < vocabSize; i++) {
    const base = i * latentDim;
    let sumSq = 0.0;
    for (let d = 0; d < latentDim; d++) {
      const v = codebook[base + d]!;
      sumSq += v * v;
    }
    const inv = 1.0 / Math.max(1e-12, Math.sqrt(sumSq));
    for (let d = 0; d < latentDim; d++) {
      out[base + d] = codebook[base + d]! * inv;
    }
  }
  return out;
}

// ============================================================================
// 128 × 128 RG32F ATLAS PACKER (EXACTLY 5 NON-OVERLAPPING RECTS, 131,072 B)
// ============================================================================

export function packBallTreeAtlas(params: {
  readonly codebook: Float32Array;
  readonly tree: UniBalancedSemanticTree;
  readonly selectedCodeId: number;
  readonly refinementSteps: UniRefinementStepPreset;
  readonly stageIndex: number;
  readonly activePolarity: UniStagePolarityPair;
  readonly membershipsByDepth: readonly UniTreeBallMembership[];
  readonly u26Mismatch: UniPrimeDepthFloorMismatchReport;
  readonly sinCos2dEmbeddings: Float32Array;
}): UniBallTreeAtlasPackResult {
  const {
    codebook,
    tree,
    selectedCodeId,
    refinementSteps,
    stageIndex,
    activePolarity,
    membershipsByDepth,
    u26Mismatch,
    sinCos2dEmbeddings,
  } = params;

  const writer = createRg32fAtlas(UNI_BALL_TREE_ATLAS_SIZE, UNI_BALL_TREE_ATLAS_SIZE);
  const latentDim = UNI_MINI_CONFIG.vqLatentDim; // 8

  // Shared codebook scale so both halves use identical diverging contrast
  const codebookMaxAbs = computeMaxAbs(codebook, 1e-4);
  const codebookScale = 1.0 / codebookMaxAbs;

  // --- Rect 1: codebook_lo (128 cols x 8 rows at (0, 0), G = 0.0) ---
  // Column c in 0..127 = codebook entry c; row r in 0..7 = latent dim r.
  const codebookLoValues = new Float32Array(128 * latentDim);
  for (let r = 0; r < latentDim; r++) {
    for (let c = 0; c < 128; c++) {
      codebookLoValues[r * 128 + c] = codebook[c * latentDim + r]!;
    }
  }
  const codebookLoEntry = writeBallTreeRegionToRg32f(writer, {
    id: 'codebook_lo',
    x0: 0,
    y0: 0,
    rows: latentDim,
    cols: 128,
    values: codebookLoValues,
    mode: 0,
    isMaskOrScore: false,
    explicitScale: codebookScale,
  });

  // --- Rect 2: codebook_hi (128 cols x 8 rows at (0, 8), G = 0.0) ---
  // Column c in 0..127 = codebook entry 128 + c; row r in 0..7 = latent dim r.
  const codebookHiValues = new Float32Array(128 * latentDim);
  for (let r = 0; r < latentDim; r++) {
    for (let c = 0; c < 128; c++) {
      codebookHiValues[r * 128 + c] = codebook[(128 + c) * latentDim + r]!;
    }
  }
  const codebookHiEntry = writeBallTreeRegionToRg32f(writer, {
    id: 'codebook_hi',
    x0: 0,
    y0: 8,
    rows: latentDim,
    cols: 128,
    values: codebookHiValues,
    mode: 0,
    isMaskOrScore: false,
    explicitScale: codebookScale,
  });

  // --- Rect 3: heap_radii (16 cols x 16 rows at (0, 16)) ---
  // Cells 0..254 carry monotone tree.heapRadii[0..254] with G = 0.0;
  // unused padding slot 255 is explicitly masked with G = 1.0!
  const heapRadiiValues = new Float32Array(256);
  const heapRadiiMasks = new Uint8Array(256);
  let maxRadius = 1e-4;
  for (let h = 0; h < tree.heapRadii.length; h++) {
    const val = tree.heapRadii[h]!;
    heapRadiiValues[h] = val;
    if (val > maxRadius) maxRadius = val;
  }
  heapRadiiValues[255] = 0.0;
  heapRadiiMasks[255] = 1; // Unused padding slot explicitly masked!

  const heapRadiiEntry = writeBallTreeRegionToRg32f(writer, {
    id: 'heap_radii',
    x0: 0,
    y0: 16,
    rows: 16,
    cols: 16,
    values: heapRadiiValues,
    mode: 1,
    isMaskOrScore: false,
    gChannelFlags: heapRadiiMasks,
    explicitScale: 1.0 / maxRadius,
  });

  // --- Rect 4: ball_membership (64 cols x 16 rows at (16, 16)) ---
  // 64 columns correspond to the 64 depth-6 leaf groups (leaves [4c .. 4c+3] in tree leaf order).
  //   Rows 0..8:  Nested candidate ball N_d(selectedCodeId) at split depth d = 0..8
  //               (live in-ball cells G = 0.0; excluded out-of-ball slots G = 1.0)
  //   Row 9:      Unit-normalized codebook proximity to selectedCodeId across all 64 groups (G = 0.0)
  //   Row 10:     Active stage candidate ball N_{l(stage)}(selectedCodeId) (G = 0.0 in-ball, 1.0 excluded)
  //   Row 11:     Miniature D=8 stage-schedule normalized ball width profile (G = 0.0)
  //   Row 12:     Production D=17 train-level normalized ball width profile (G = 0.0)
  //   Row 13:     Production D=17 infer-level normalized ball width profile (G = 0.0)
  //   Row 14:     U26 infer/train ball ratio profile (2x wider = 1.0 vs exact 1x = 0.5, G = 0.0)
  //   Row 15:     Depth-6 subtree radii tree.heapRadii[63..126] / maxRadius (all 64 live, G = 0.0)
  const membershipCols = 64;
  const membershipRows = 16;
  const membershipValues = new Float32Array(membershipRows * membershipCols);
  const membershipMasks = new Uint8Array(membershipRows * membershipCols);

  // Rows 0..8: Exact nested depth membership funnel
  for (let d = 0; d <= tree.depth; d++) {
    const m = membershipsByDepth[d]!;
    const bStart = m.ballStartLeaf;
    const bEnd = bStart + m.ballSize - 1;
    const depthWeight = 0.38 + 0.62 * (d / tree.depth);
    const rowBase = d * membershipCols;

    for (let c = 0; c < membershipCols; c++) {
      const gStart = c * 4;
      const gEnd = gStart + 3;
      const overlapLo = Math.max(gStart, bStart);
      const overlapHi = Math.min(gEnd, bEnd);
      if (overlapLo <= overlapHi) {
        const overlapCount = overlapHi - overlapLo + 1;
        const frac = overlapCount / 4.0;
        membershipValues[rowBase + c] = Math.fround(frac * depthWeight);
        membershipMasks[rowBase + c] = 0;
      } else {
        membershipValues[rowBase + c] = 0.0;
        membershipMasks[rowBase + c] = 1; // Out-of-ball excluded slot
      }
    }
  }

  // Row 9: Normalized codebook proximity (1 - avgDistSq / maxRadius) across all 64 leaf groups
  const normCodebook = normalizeCodebookUnitRows(codebook, UNI_MINI_VQ_CODEBOOK_SIZE, latentDim);
  const selBase = selectedCodeId * latentDim;
  for (let c = 0; c < membershipCols; c++) {
    let avgDistSq = 0.0;
    for (let k = 0; k < 4; k++) {
      const tok = tree.leafToToken[c * 4 + k]!;
      const tBase = tok * latentDim;
      let distSq = 0.0;
      for (let dim = 0; dim < latentDim; dim++) {
        const diff = normCodebook[selBase + dim]! - normCodebook[tBase + dim]!;
        distSq += diff * diff;
      }
      avgDistSq += distSq;
    }
    avgDistSq *= 0.25;
    const prox = Math.max(0.0, Math.min(1.0, 1.0 - avgDistSq / maxRadius));
    membershipValues[9 * membershipCols + c] = Math.fround(prox);
    membershipMasks[9 * membershipCols + c] = 0;
  }

  // Row 10: Active stage candidate ball highlight strip
  const activeBall = membershipsByDepth[activePolarity.level]!;
  const actStart = activeBall.ballStartLeaf;
  const actEnd = actStart + activeBall.ballSize - 1;
  const activeIntensity = Math.fround(0.45 + 0.55 * (1.0 - activePolarity.noiseWeight));
  for (let c = 0; c < membershipCols; c++) {
    const gStart = c * 4;
    const gEnd = gStart + 3;
    const overlapLo = Math.max(gStart, actStart);
    const overlapHi = Math.min(gEnd, actEnd);
    if (overlapLo <= overlapHi) {
      membershipValues[10 * membershipCols + c] = Math.fround(
        activeIntensity * ((overlapHi - overlapLo + 1) / 4.0),
      );
      membershipMasks[10 * membershipCols + c] = 0;
    } else {
      membershipValues[10 * membershipCols + c] = 0.0;
      membershipMasks[10 * membershipCols + c] = 1;
    }
  }

  // Rows 11..14: Miniature D=8 vs Production D=17 schedule comparison strips (all live, G = 0.0)
  const stageCountPlusOne = refinementSteps + 1;
  const stageList = new Int32Array(stageCountPlusOne);
  for (let s = 0; s <= refinementSteps; s++) stageList[s] = s;
  const miniSchedule = semanticTreeLevelsForAges(stageList, tree.depth, refinementSteps);

  for (let c = 0; c < membershipCols; c++) {
    const s = Math.min(refinementSteps, Math.floor((c * stageCountPlusOne) / membershipCols));
    const miniLvl = miniSchedule.levels[s]!;
    const prodTrainLvl =
      s === 0
        ? 0
        : s >= refinementSteps
          ? UNI_BALL_TREE_PROD_DEPTH
          : UNI_BALL_TREE_PROD_DEPTH -
            Math.floor(((refinementSteps - s) * UNI_BALL_TREE_PROD_DEPTH) / refinementSteps);
    const prodInferLvl =
      s === 0
        ? 0
        : s >= refinementSteps
          ? UNI_BALL_TREE_PROD_DEPTH
          : Math.floor((s * UNI_BALL_TREE_PROD_DEPTH) / refinementSteps);

    const miniNormBall = (tree.depth - miniLvl) / tree.depth;
    const prodTrainNormBall = (UNI_BALL_TREE_PROD_DEPTH - prodTrainLvl) / UNI_BALL_TREE_PROD_DEPTH;
    const prodInferNormBall = (UNI_BALL_TREE_PROD_DEPTH - prodInferLvl) / UNI_BALL_TREE_PROD_DEPTH;
    const ratioScaled = prodTrainLvl !== prodInferLvl ? 1.0 : 0.5;

    membershipValues[11 * membershipCols + c] = Math.fround(miniNormBall);
    membershipValues[12 * membershipCols + c] = Math.fround(prodTrainNormBall);
    membershipValues[13 * membershipCols + c] = Math.fround(prodInferNormBall);
    membershipValues[14 * membershipCols + c] = Math.fround(ratioScaled);
    membershipMasks[11 * membershipCols + c] = 0;
    membershipMasks[12 * membershipCols + c] = 0;
    membershipMasks[13 * membershipCols + c] = 0;
    membershipMasks[14 * membershipCols + c] = 0;
  }

  // Row 15: Exact depth-6 subtree radii tree.heapRadii[63..126] normalized by maxRadius (G = 0.0)
  for (let c = 0; c < membershipCols; c++) {
    const heapIdx = 63 + c;
    const rNorm = Math.max(0.0, Math.min(1.0, tree.heapRadii[heapIdx]! / maxRadius));
    membershipValues[15 * membershipCols + c] = Math.fround(rNorm);
    membershipMasks[15 * membershipCols + c] = 0;
  }

  const ballMembershipEntry = writeBallTreeRegionToRg32f(writer, {
    id: 'ball_membership',
    x0: 16,
    y0: 16,
    rows: membershipRows,
    cols: membershipCols,
    values: membershipValues,
    mode: 1,
    isMaskOrScore: true,
    gChannelFlags: membershipMasks,
    explicitScale: 1.0,
  });

  // --- Rect 5: sincos_2d (8 cols x 24 rows at (80, 16), G = 0.0) ---
  // 24 rows = 4x6 marked target VQ grid cells, 8 cols = hiddenSize, markerColumn = true.
  const sinCosRows = UNI_MINI_CONFIG.vqGridSize * UNI_MINI_CONFIG.vqMarkedGridCols; // 24
  const sinCosCols = UNI_MINI_CONFIG.hiddenSize; // 8
  const sinCosMaxAbs = computeMaxAbs(sinCos2dEmbeddings, 1e-6);
  const sinCosEntry = writeBallTreeRegionToRg32f(writer, {
    id: 'sincos_2d',
    x0: 80,
    y0: 16,
    rows: sinCosRows,
    cols: sinCosCols,
    values: sinCos2dEmbeddings,
    mode: 0,
    isMaskOrScore: false,
    explicitScale: 1.0 / sinCosMaxAbs,
  });

  const regions: readonly UniAtlasRegionBox[] = Object.freeze([
    codebookLoEntry.region,
    codebookHiEntry.region,
    heapRadiiEntry.region,
    ballMembershipEntry.region,
    sinCosEntry.region,
  ]);

  const audit = auditAtlasRegions(
    UNI_BALL_TREE_ATLAS_SIZE,
    UNI_BALL_TREE_ATLAS_SIZE,
    regions,
    UNI_BALL_TREE_ATLAS_BUDGET_BYTES,
  );

  const atlas: Rg32fValueAtlas = Object.freeze({
    width: writer.width,
    height: writer.height,
    data: writer.data,
  });

  const extraTag = `c${selectedCodeId}_n${refinementSteps}_s${stageIndex}_u26m${u26Mismatch.mismatchedStageCount}`;
  const atlasKey = computeAtlasContentKey('ball_tree', atlas, extraTag);

  return Object.freeze({
    atlas,
    atlasKey,
    regions,
    audit,
    slices: Object.freeze({
      codebook_lo: codebookLoEntry,
      codebook_hi: codebookHiEntry,
      heap_radii: heapRadiiEntry,
      ball_membership: ballMembershipEntry,
      sincos_2d: sinCosEntry,
    }),
  });
}

// ============================================================================
// WIREFRAME RING & BALL GEOMETRY HELPERS
// ============================================================================

function addWireframeCircle3D(
  lines: UniLineBufferBuilder,
  cx: number,
  cy: number,
  cz: number,
  radius: number,
  plane: 'xy' | 'xz' | 'yz',
  segments: number,
  cssWidth: number,
  style: number,
  alpha: number,
): void {
  const step = (Math.PI * 2) / segments;
  for (let i = 0; i < segments; i++) {
    const a0 = i * step;
    const a1 = (i + 1) * step;
    const c0 = Math.cos(a0) * radius;
    const s0 = Math.sin(a0) * radius;
    const c1 = Math.cos(a1) * radius;
    const s1 = Math.sin(a1) * radius;

    if (plane === 'xy') {
      lines.addSegment(cx + c0, cy + s0, cz, cx + c1, cy + s1, cz, cssWidth, style, alpha);
    } else if (plane === 'xz') {
      lines.addSegment(cx + c0, cy, cz + s0, cx + c1, cy, cz + s1, cssWidth, style, alpha);
    } else {
      lines.addSegment(cx, cy + c0, cz + s0, cx, cy + c1, cz + s1, cssWidth, style, alpha);
    }
  }
}

function leafOrderToCellFamily(leafIndex: number): number {
  if (leafIndex < 64) return CELL_WEIGHT_BLUE;
  if (leafIndex < 128) return CELL_SSM_CYAN;
  if (leafIndex < 192) return CELL_ACTIVATION_GREEN;
  return CELL_MOE_AMBER;
}

function formatU26MismatchSummary(report: UniPrimeDepthFloorMismatchReport): {
  readonly badgeLabel: string;
  readonly anchorLabel: string;
  readonly anchorSublabel: string;
} {
  if (report.rows.length === 0) {
    // N = 1 binary MDM has 0 internal stages; include K = 4 production reference for pedagogy.
    const refK4 = analyzePrimeDepthFloorMismatch(UNI_BALL_TREE_PROD_DEPTH, 4);
    const rev = [...refK4.rows].reverse();
    const trnLvls = rev.map((r) => r.trainLevel).join(',');
    const infLvls = rev.map((r) => r.inferLevel).join(',');
    return {
      badgeLabel: `U26 Prod D=17: 0/0 Internal Stages at N=1 (Binary MDM 256->1; K=4 Train [${trnLvls}] vs Infer [${infLvls}] 2× Wider)`,
      anchorLabel: 'U26 Prod D=17 vs Mini D=8 (N=1 Binary MDM)',
      anchorSublabel: `Mini D=8: 0 mismatches (256->1) · Prod D=17 K=4 ref: trainLevel=[${trnLvls}] (16,256,4096) vs inferLevel=[${infLvls}] (32,512,8192) 2× wider`,
    };
  }

  // Order by trainAge = 1..K-1 (reverse of stageIndex = 1..K-1) so K=4 displays canonical [13,9,5] vs [12,8,4]
  const byTrainAge = [...report.rows].reverse();
  const trainLevelsStr = byTrainAge.map((r) => r.trainLevel).join(', ');
  const inferLevelsStr = byTrainAge.map((r) => r.inferLevel).join(', ');
  const trainBallsStr = byTrainAge.map((r) => r.trainBallSize).join(', ');
  const inferBallsStr = byTrainAge.map((r) => r.inferBallSize).join(', ');

  return {
    badgeLabel: `U26 Prod D=17 (K=${report.refinementSteps}): ${report.mismatchedStageCount}/${report.totalInternalStages} Stages 2× Wider · Train [${trainLevelsStr}] vs Infer [${inferLevelsStr}]`,
    anchorLabel: `U26 Production D=17 Floor Mismatch (${report.mismatchedStageCount}/${report.totalInternalStages} · 2× Wider)`,
    anchorSublabel: `Train [${trainLevelsStr}] -> (${trainBallsStr}) vs Infer [${inferLevelsStr}] -> (${inferBallsStr}) | Mini D=8: 0 mismatches`,
  };
}

// ============================================================================
// MAIN VIEW 4 (`ball_tree`) SPATIAL SCENE BUILDER
// ============================================================================

export function buildUniBallTreeScene(
  session: UniSession | null,
  options: UniBallTreeSceneOptions = {},
): UniSpatialScene {
  // --- 1. Resolve live session or deterministic standalone fallback ---
  const hasLiveRecord =
    session !== null &&
    (options.recordId !== undefined || session.getActiveRecordId() !== null);

  let codebook: Float32Array;
  let tree: UniBalancedSemanticTree;
  let pca3d: Float32Array;
  let executionViewId: UniExecutionViewId = 'text_to_image';
  let prefillMode: UniElasticPrefillMode = 'hf_faithful';
  let baseProvenanceBadges: readonly UniProvenanceBadge[];
  let d5Comparison = buildUniElasticD5Comparison('text_to_image', 16);
  let viewReadout =
    buildUniViewArchReadouts().find((v) => v.viewId === 'text_to_image') ??
    buildUniViewArchReadouts()[0]!;

  let sessionDefaultRefinementSteps: UniRefinementStepPreset =
    UNI_MINI_CONFIG.defaultRefinementSteps;
  let sessionDefaultCodeId: number = UNI_BALL_TREE_DEFAULT_CODE_ID;
  let sessionDefaultStageIndex: number | undefined;

  if (hasLiveRecord && session !== null) {
    const ctx = resolveUniSessionContext(session, options.recordId);
    codebook = ctx.bundle.vqvae.codebook;
    tree = ctx.bundle.tree;
    pca3d = getCachedPca3d(codebook);
    executionViewId = ctx.executionViewId;
    prefillMode = ctx.prefillMode;
    baseProvenanceBadges = ctx.provenanceBadges;
    d5Comparison = ctx.d5Comparison;
    viewReadout = ctx.viewReadout;
    sessionDefaultRefinementSteps = normalizeRefinementSteps(ctx.branch.refinementSteps);

    if (ctx.record.imageState) {
      const { stages, representatives, proposals } = ctx.record.imageState;
      // Pick the first content cell with an active/refining VQ code < 256
      let chosenIdx = -1;
      for (let i = 0; i < stages.length; i++) {
        const rep = representatives[i]!;
        if (rep >= 0 && rep < UNI_MINI_VQ_CODEBOOK_SIZE) {
          if (chosenIdx < 0 || stages[i]! >= 0) {
            chosenIdx = i;
            if (stages[i]! > 0 && stages[i]! < sessionDefaultRefinementSteps) {
              break;
            }
          }
        }
      }
      if (chosenIdx >= 0) {
        const candidateCode =
          proposals[chosenIdx]! >= 0 && proposals[chosenIdx]! < UNI_MINI_VQ_CODEBOOK_SIZE
            ? proposals[chosenIdx]!
            : representatives[chosenIdx]!;
        sessionDefaultCodeId = clampInt(candidateCode, 0, UNI_MINI_VQ_CODEBOOK_SIZE - 1);
        sessionDefaultStageIndex = clampInt(
          stages[chosenIdx]!,
          -1,
          sessionDefaultRefinementSteps,
        );
      }
    } else if (ctx.branch.sourceVqCodes && ctx.branch.sourceVqCodes.length > 0) {
      sessionDefaultCodeId = clampInt(
        ctx.branch.sourceVqCodes[0]!,
        0,
        UNI_MINI_VQ_CODEBOOK_SIZE - 1,
      );
    }
  } else if (session !== null) {
    // Live UniSession provided before startBranch(): use its bundle directly without throwing
    codebook = session.bundle.vqvae.codebook;
    tree = session.bundle.tree;
    pca3d = getCachedPca3d(codebook);
    baseProvenanceBadges = Object.freeze([
      Object.freeze({ label: 'Standalone Bundle · Codebook & Tree', tone: 'g_teal' as const }),
      Object.freeze({ label: UNI_SCENE_PROJECTION_REVERSAL_BADGE, tone: 'muted' as const }),
    ]);
  } else {
    const fallback = getDefaultCodebookAndTree();
    codebook = fallback.codebook;
    tree = fallback.tree;
    pca3d = fallback.pca3d;
    baseProvenanceBadges = Object.freeze([
      Object.freeze({ label: 'Standalone Deterministic Codebook (256 × 8)', tone: 'g_teal' as const }),
      Object.freeze({ label: UNI_SCENE_PROJECTION_REVERSAL_BADGE, tone: 'muted' as const }),
    ]);
  }

  const refinementSteps = normalizeRefinementSteps(
    options.refinementSteps ?? sessionDefaultRefinementSteps,
  );
  const selectedCodeId = clampInt(
    options.selectedCodeId ?? sessionDefaultCodeId,
    0,
    UNI_MINI_VQ_CODEBOOK_SIZE - 1,
  );
  const defaultStage =
    sessionDefaultStageIndex !== undefined
      ? clampInt(sessionDefaultStageIndex, -1, refinementSteps)
      : refinementSteps > 1
        ? 1
        : 0;
  const stageIndex = clampInt(options.stageIndex ?? defaultStage, -1, refinementSteps);
  const showProductionD17Comparison = options.showProductionD17Comparison ?? true;
  const hoveredPartId = options.hoveredCell?.partId ?? null;

  // --- 2. Compute exact Ball-Tree memberships, schedule levels, U9 polarity & U26 mismatch ---
  const activePolarity = getStagePolarityPair(stageIndex, tree.depth, refinementSteps);

  const membershipsByDepth: UniTreeBallMembership[] = [];
  for (let d = 0; d <= tree.depth; d++) {
    membershipsByDepth.push(getTreeBallMembership(tree, selectedCodeId, d));
  }

  const scheduleStages = new Int32Array(refinementSteps + 1);
  for (let s = 0; s <= refinementSteps; s++) scheduleStages[s] = s;
  const scheduleLevelsRes = semanticTreeLevelsForAges(scheduleStages, tree.depth, refinementSteps);
  const scheduledLevelSet = new Set<number>();
  const scheduleBallSizes: number[] = [];
  for (let s = 0; s <= refinementSteps; s++) {
    scheduledLevelSet.add(scheduleLevelsRes.levels[s]!);
    scheduleBallSizes.push(scheduleLevelsRes.ballSizes[s]!);
  }

  const u26Mismatch = analyzePrimeDepthFloorMismatch(UNI_BALL_TREE_PROD_DEPTH, refinementSteps);
  const miniMismatch = analyzePrimeDepthFloorMismatch(tree.depth, refinementSteps);

  // Compute canonical 2D SinCos embeddings for 4×6 marked target grid (canonicalContentSize=32, scale=0.0018, markerColumn=true)
  const sinCos2dResult = compute2dSinCosPosEmbed(
    UNI_MINI_CONFIG.vqGridSize,
    UNI_MINI_CONFIG.vqMarkedGridCols,
    UNI_MINI_CONFIG.hiddenSize,
    true,
    UNI_MINI_CONFIG.sinCos2d,
  );

  // --- 3. Pack 128 × 128 RG32F Scalar/Mask Atlas (5 non-overlapping rects, 131,072 B) ---
  const packedAtlas = packBallTreeAtlas({
    codebook,
    tree,
    selectedCodeId,
    refinementSteps,
    stageIndex,
    activePolarity,
    membershipsByDepth,
    u26Mismatch,
    sinCos2dEmbeddings: sinCos2dResult.embeddings,
  });

  // --- 4. Initialize 5-pass builders, PickBoxes, overlays & bounds accumulator ---
  const solids = new UniSolidBufferBuilder();
  const lines = new UniLineBufferBuilder();
  const ribbons = new UniRibbonBufferBuilder();
  const sheets = new UniSheetBufferBuilder();
  const flow = new UniFlatFlowBufferBuilder();
  const pickBoxes: PickBox[] = [];
  const overlayAnchors: UniOverlayAnchor[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];

  const boundsAcc = {
    min: [Infinity, Infinity, Infinity] as [number, number, number],
    max: [-Infinity, -Infinity, -Infinity] as [number, number, number],
  };

  // --- 5. Left Wing: 3D PCA Codebook Cloud (256 instanced SolidsPass nodes) ---
  const CLOUD_CENTER: Vec3 = [-2.75, 2.55, -0.85];
  const CLOUD_SCALE = 2.05;

  let maxPcaCoord = 1e-4;
  for (let i = 0; i < pca3d.length; i++) {
    const a = Math.abs(pca3d[i]!);
    if (a > maxPcaCoord) maxPcaCoord = a;
  }

  const nodePositions: Vec3[] = new Array<Vec3>(UNI_MINI_VQ_CODEBOOK_SIZE);
  for (let codeId = 0; codeId < UNI_MINI_VQ_CODEBOOK_SIZE; codeId++) {
    const px = CLOUD_CENTER[0] + (pca3d[codeId * 3]! / maxPcaCoord) * CLOUD_SCALE;
    const py = CLOUD_CENTER[1] + (pca3d[codeId * 3 + 1]! / maxPcaCoord) * CLOUD_SCALE;
    const pz = CLOUD_CENTER[2] + (pca3d[codeId * 3 + 2]! / maxPcaCoord) * CLOUD_SCALE;
    nodePositions[codeId] = [px, py, pz];
    expandAabbPoint(boundsAcc, px, py, pz, 0.25);
  }

  const activeBall = membershipsByDepth[activePolarity.level]!;
  const activeMemberSet = new Set<number>();
  for (let i = 0; i < activeBall.memberTokenIds.length; i++) {
    activeMemberSet.add(activeBall.memberTokenIds[i]!);
  }

  // Drafting table plinth & PCA coordinate axes beneath the 3D cloud
  lines.addBoxFrameXZ(
    CLOUD_CENTER[0] - 2.65,
    CLOUD_CENTER[2] - 2.65,
    CLOUD_CENTER[0] + 2.65,
    CLOUD_CENTER[2] + 2.65,
    0.02,
    1.1,
    LINE_STYLE_GUIDE,
    0.45,
  );
  lines.addDashedFrameXZ(
    CLOUD_CENTER[0] - 2.35,
    CLOUD_CENTER[2] - 2.35,
    CLOUD_CENTER[0] + 2.35,
    CLOUD_CENTER[2] + 2.35,
    0.025,
    LINE_STYLE_GUIDE,
    0.3,
  );
  // Subtle PCA reference triad
  lines.addSegment(
    CLOUD_CENTER[0] - 2.35,
    CLOUD_CENTER[1],
    CLOUD_CENTER[2],
    CLOUD_CENTER[0] + 2.35,
    CLOUD_CENTER[1],
    CLOUD_CENTER[2],
    0.85,
    LINE_STYLE_GUIDE,
    0.26,
  );
  lines.addSegment(
    CLOUD_CENTER[0],
    CLOUD_CENTER[1] - 2.25,
    CLOUD_CENTER[2],
    CLOUD_CENTER[0],
    CLOUD_CENTER[1] + 2.25,
    CLOUD_CENTER[2],
    0.85,
    LINE_STYLE_GUIDE,
    0.26,
  );
  lines.addSegment(
    CLOUD_CENTER[0],
    CLOUD_CENTER[1],
    CLOUD_CENTER[2] - 2.35,
    CLOUD_CENTER[0],
    CLOUD_CENTER[1],
    CLOUD_CENTER[2] + 2.35,
    0.85,
    LINE_STYLE_GUIDE,
    0.26,
  );

  // Emit all 256 codebook nodes as instanced SolidsPass cuboids + PickBoxes
  for (let codeId = 0; codeId < UNI_MINI_VQ_CODEBOOK_SIZE; codeId++) {
    const [px, py, pz] = nodePositions[codeId]!;
    const leafIdx = tree.tokenToLeaf[codeId]!;
    const isSelected = codeId === selectedCodeId;
    const isInActiveBall = activeMemberSet.has(codeId);
    const partId = `codebook_node_${codeId}`;
    const isHovered = hoveredPartId === partId;

    const size = isSelected ? 0.22 : isHovered ? 0.18 : isInActiveBall ? 0.135 : 0.095;
    const family = leafOrderToCellFamily(leafIdx);
    const highlight = isSelected || isHovered ? 1 : isInActiveBall ? 2 : 0;
    const fp8 = isSelected || isInActiveBall ? 0 : 1; // Frosted slate for out-of-ball codes

    solids.push({
      cx: px,
      cy: py,
      cz: pz,
      ex: size,
      ey: size,
      ez: size,
      cellFamily: family,
      fp8,
      phase: leafIdx / 255,
      bayId: 0,
      cellScale: 0.12,
      highlight,
    });

    if (isSelected || isHovered) {
      lines.addBoxEdges(
        px,
        py,
        pz,
        size + 0.03,
        size + 0.03,
        size + 0.03,
        -1,
        1.9,
        LINE_STYLE_ACTIVE,
        0.95,
      );
    }

    pickBoxes.push(
      packPickBoxFromAabb(px, py, pz, Math.max(size, 0.14), Math.max(size, 0.14), Math.max(size, 0.14), {
        bayId: 0,
        partId,
        tensorKey: 'codebook_pca_node',
        tokenIndex: codeId,
        tapIndex: activePolarity.level,
        towerRole: 'denoiser',
        semanticTag: `branch=g;tapRole=tree;depth=${activePolarity.level};code=${codeId};leaf=${leafIdx}`,
      }),
    );
  }

  // --- 6. Nested Ball-Tree Wireframe Shells & Clickable Depth Rings (depth = 0..8) ---
  const rootRadius = Math.max(1e-4, tree.heapRadii[0]!);
  const depthShells: DepthShellGeometry[] = [];
  const PEDESTAL_Z = 2.55;
  const PEDESTAL_START_X = -5.55;
  const PEDESTAL_PITCH_X = 0.72;

  for (let d = 0; d <= tree.depth; d++) {
    const m = membershipsByDepth[d]!;
    let sumX = 0;
    let sumY = 0;
    let sumZ = 0;
    const count = m.memberTokenIds.length;
    for (let i = 0; i < count; i++) {
      const p = nodePositions[m.memberTokenIds[i]!]!;
      sumX += p[0];
      sumY += p[1];
      sumZ += p[2];
    }
    const cx = sumX / count;
    const cy = sumY / count;
    const cz = sumZ / count;

    let maxDist3d = 0;
    for (let i = 0; i < count; i++) {
      const p = nodePositions[m.memberTokenIds[i]!]!;
      const dist = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
      if (dist > maxDist3d) maxDist3d = dist;
    }

    const radiusRatio = Math.sqrt(Math.max(0, m.radius) / rootRadius);
    const radius3d =
      d === tree.depth
        ? 0.24
        : Math.max(0.28 + (tree.depth - d) * 0.06, maxDist3d + 0.08 + 0.05 * radiusRatio);

    const isActiveLevel = d === activePolarity.level;
    const isScheduledLevel = scheduledLevelSet.has(d);
    const pedestalCenter: Vec3 = [PEDESTAL_START_X + d * PEDESTAL_PITCH_X, 0.14, PEDESTAL_Z];

    depthShells.push({
      depth: d,
      membership: m,
      centroid: [cx, cy, cz],
      radius3d,
      pedestalCenter,
      isScheduledLevel,
      isActiveLevel,
    });

    expandAabbPoint(boundsAcc, cx, cy, cz, radius3d + 0.1);
    expandAabbPoint(boundsAcc, pedestalCenter[0], pedestalCenter[1], pedestalCenter[2], 0.35);

    // 3 orthogonal wireframe rings in 3D PCA cloud
    const shellStyle = isActiveLevel
      ? LINE_STYLE_ACTIVE
      : isScheduledLevel
        ? d > activePolarity.level
          ? LINE_STYLE_BF16_GOLD
          : LINE_STYLE_NEUTRAL
        : LINE_STYLE_GUIDE;
    const shellWidth = isActiveLevel ? 2.1 : isScheduledLevel ? 1.35 : 0.85;
    const shellAlpha = isActiveLevel ? 0.92 : isScheduledLevel ? 0.56 : 0.22;
    const segments = isActiveLevel ? 32 : 24;

    addWireframeCircle3D(lines, cx, cy, cz, radius3d, 'xy', segments, shellWidth, shellStyle, shellAlpha);
    addWireframeCircle3D(lines, cx, cy, cz, radius3d, 'xz', segments, shellWidth, shellStyle, shellAlpha);
    addWireframeCircle3D(lines, cx, cy, cz, radius3d, 'yz', segments, shellWidth, shellStyle, shellAlpha);

    // Clickable depth-ring pedestal along front-left apron (unobstructed picking for all depths 0..8)
    const ringPartId = `tree_depth_ring_${d}`;
    const isRingHovered = hoveredPartId === ringPartId;
    const pedHeight = 0.14 + 0.09 * (tree.depth - d) * (isScheduledLevel ? 1.0 : 0.55);
    const pedRadius = 0.11 + 0.16 * (d === tree.depth ? 0.15 : radiusRatio);

    solids.push({
      cx: pedestalCenter[0],
      cy: pedHeight * 0.5,
      cz: pedestalCenter[2],
      ex: 0.42,
      ey: pedHeight,
      ez: 0.42,
      cellFamily: isActiveLevel
        ? CELL_ACTIVATION_GREEN
        : isScheduledLevel
          ? CELL_SSM_CYAN
          : CELL_NEUTRAL,
      fp8: isScheduledLevel || isActiveLevel ? 0 : 1,
      phase: d / tree.depth,
      bayId: 0,
      cellScale: 0.14,
      highlight: isActiveLevel || isRingHovered ? 1 : isScheduledLevel ? 2 : 0,
    });

    addWireframeCircle3D(
      lines,
      pedestalCenter[0],
      pedHeight + 0.04,
      pedestalCenter[2],
      pedRadius,
      'xz',
      20,
      isActiveLevel || isRingHovered ? 2.0 : 1.15,
      isActiveLevel || isRingHovered ? LINE_STYLE_ACTIVE : shellStyle,
      isActiveLevel || isRingHovered ? 0.95 : 0.72,
    );

    // Subtle leader line connecting active/scheduled pedestals up to their 3D cloud shell centroid
    if (isActiveLevel || isScheduledLevel) {
      lines.addSegment(
        pedestalCenter[0],
        pedHeight + 0.04,
        pedestalCenter[2],
        cx,
        cy - radius3d * 0.5,
        cz,
        isActiveLevel ? 1.35 : 0.75,
        isActiveLevel ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
        isActiveLevel ? 0.72 : 0.25,
      );
    }

    pickBoxes.push(
      packPickBoxFromAabb(
        pedestalCenter[0],
        pedHeight * 0.5 + 0.04,
        pedestalCenter[2],
        0.52,
        Math.max(0.32, pedHeight + 0.12),
        0.52,
        {
          bayId: 0,
          partId: ringPartId,
          tensorKey: 'tree_depth_ring',
          tapIndex: d,
          towerRole: 'denoiser',
          semanticTag: `branch=g;tapRole=tree;depth=${d};ballSize=${m.ballSize}`,
        },
      ),
    );
  }

  // 3D ancestor trajectory ribbon tracing d=0 -> d=8 centroids inside PCA cloud
  const ancestorPath3d: Vec3[] = depthShells.map((s) => s.centroid);
  flow.addPolyline3D(
    ancestorPath3d,
    [0, 1, 0],
    0.055,
    UNI_FLOW_COLORS.gTeal,
    true,
    0.16,
    0.15,
    true,
  );

  // Subtle top-2 binary tree split hierarchy edges (depth 0 -> 1 -> 2 centroids)
  const computeSubtreeCentroid = (startLeaf: number, count: number): Vec3 => {
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let i = 0; i < count; i++) {
      const p = nodePositions[tree.leafToToken[startLeaf + i]!]!;
      sx += p[0];
      sy += p[1];
      sz += p[2];
    }
    return [sx / count, sy / count, sz / count];
  };
  const rootCentroid = depthShells[0]!.centroid;
  for (let child = 0; child < 2; child++) {
    const c1 = computeSubtreeCentroid(child * 128, 128);
    lines.addSegment(
      rootCentroid[0],
      rootCentroid[1],
      rootCentroid[2],
      c1[0],
      c1[1],
      c1[2],
      1.0,
      LINE_STYLE_GUIDE,
      0.34,
    );
    for (let grand = 0; grand < 2; grand++) {
      const c2 = computeSubtreeCentroid(child * 128 + grand * 64, 64);
      lines.addSegment(
        c1[0],
        c1[1],
        c1[2],
        c2[0],
        c2[1],
        c2[2],
        0.85,
        LINE_STYLE_GUIDE,
        0.24,
      );
    }
  }

  // --- 7. Right Wing: All 5 Live RG32F Atlas Plates (`SheetsPass` + `packPickBoxFromFrame`) ---
  const SHEET_Y = 0.08;
  const SHEET_EY = 0.06;

  interface PlateLayoutSpec {
    readonly id: UniBallTreeRegionId;
    readonly center: Vec3;
    readonly extent: Vec3;
    readonly role: number;
    readonly label: string;
    readonly sublabel: string;
    readonly badge?: string;
  }

  const plateSpecs: readonly PlateLayoutSpec[] = [
    {
      id: 'codebook_lo',
      center: [4.75, SHEET_Y, -3.15],
      extent: [6.2, SHEET_EY, 0.96],
      role: SHEET_ROLE_WEIGHT,
      label: 'VQ Codebook Lo (0..127 × 8)',
      sublabel: 'Emu3 IBQ codebook entries 0..127 · G = 0.0 active',
      badge: selectedCodeId < 128 ? `Selected #${selectedCodeId}` : undefined,
    },
    {
      id: 'codebook_hi',
      center: [4.75, SHEET_Y, -1.85],
      extent: [6.2, SHEET_EY, 0.96],
      role: SHEET_ROLE_WEIGHT,
      label: 'VQ Codebook Hi (128..255 × 8)',
      sublabel: 'Emu3 IBQ codebook entries 128..255 · G = 0.0 active',
      badge: selectedCodeId >= 128 ? `Selected #${selectedCodeId}` : undefined,
    },
    {
      id: 'heap_radii',
      center: [2.75, SHEET_Y, 0.25],
      extent: [2.2, SHEET_EY, 2.2],
      role: SHEET_ROLE_NORM,
      label: 'Monotone Heap Radii (16 × 16)',
      sublabel: `255 internal nodes (G=0.0) · slot 255 masked (G=1.0) · root r=${tree.heapRadii[0]!.toFixed(3)}`,
      badge:
        activeBall.heapNodeIndex !== null
          ? `Heap #${activeBall.heapNodeIndex} (r=${activeBall.radius.toFixed(3)})`
          : 'Leaf (r=0.000)',
    },
    {
      id: 'ball_membership',
      center: [6.05, SHEET_Y, 0.25],
      extent: [3.6, SHEET_EY, 2.2],
      role: SHEET_ROLE_ACTIVATION,
      label: 'Nested Ball Membership & U26 Strips (16 × 64)',
      sublabel: `Depths 0..8 funnel (256->1) · active stage ${stageIndex} (|N|=${activePolarity.ballSize})`,
      badge: `Level ${activePolarity.level} / 8`,
    },
    {
      id: 'sincos_2d',
      center: [8.85, SHEET_Y, -1.25],
      extent: [1.15, SHEET_EY, 3.2],
      role: SHEET_ROLE_RESIDUAL,
      label: '2D SinCos Target Grid (24 × 8)',
      sublabel: '4×6 marked grid · canonical=32 · scale=0.0018 · markerColumn=true',
      badge: 'G = 0.0 Active',
    },
  ];

  const plateFrames = {} as Record<UniBallTreeRegionId, SheetFrame3D>;

  for (let i = 0; i < plateSpecs.length; i++) {
    const spec = plateSpecs[i]!;
    const sliceEntry = packedAtlas.slices[spec.id];
    const frame = makeSheetFrame3D(
      spec.center,
      spec.extent,
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    plateFrames[spec.id] = frame;
    expandAabb(boundsAcc, sheetFrameWorldAabb(frame));

    const isHovered = hoveredPartId === spec.id;
    const isCodebookHit =
      (spec.id === 'codebook_lo' && selectedCodeId < 128) ||
      (spec.id === 'codebook_hi' && selectedCodeId >= 128);
    const hl = isHovered
      ? SHEET_HL_ACTIVE
      : isCodebookHit || spec.id === 'ball_membership'
        ? SHEET_HL_OPERAND_B
        : spec.id === 'heap_radii'
          ? SHEET_HL_PATHWAY
          : SHEET_HL_NORMAL;

    sheets.push({
      cx: spec.center[0],
      cy: spec.center[1],
      cz: spec.center[2],
      ex: spec.extent[0],
      ey: spec.extent[1],
      ez: spec.extent[2],
      gridCols: sliceEntry.region.cols,
      gridRows: sliceEntry.region.rows,
      sheetKind: SHEET_KIND_STANDARD,
      role: spec.role,
      highlight: hl,
      seed: 0.41 * (i + 1),
      slice: sliceEntry.slice,
      frame,
    });

    pickBoxes.push(
      packPickBoxFromFrame(frame, {
        bayId: 0,
        partId: spec.id,
        tensorKey: spec.id,
        gridCols: sliceEntry.region.cols,
        gridRows: sliceEntry.region.rows,
        tapIndex: activePolarity.level,
        towerRole: 'denoiser',
        semanticTag: `branch=g;tapRole=tree;depth=${activePolarity.level};plate=${spec.id}`,
      }),
    );

    overlayAnchors.push(
      Object.freeze({
        id: `anchor_${spec.id}`,
        label: spec.label,
        sublabel: spec.sublabel,
        worldPos: [
          spec.center[0],
          spec.center[1] + 0.24,
          spec.center[2] - spec.extent[2] * 0.5 - 0.14,
        ] as Vec3,
        tone: spec.id === 'ball_membership' || isCodebookHit ? 'g_teal' : 'muted',
        ...(spec.badge ? { badge: spec.badge } : {}),
      }),
    );
  }

  // Colored selection framing over live RG32F values via `addCellHighlightOnFrame` (never via G=1!)
  if (selectedCodeId < 128) {
    flow.addCellHighlightOnFrame(
      plateFrames.codebook_lo,
      0,
      7,
      selectedCodeId,
      selectedCodeId,
      8,
      128,
      0.038,
      UNI_FLOW_COLORS.activeOxide,
      UNI_FLOW_COLORS.activeOxide,
      0.018,
    );
  } else {
    const hiCol = selectedCodeId - 128;
    flow.addCellHighlightOnFrame(
      plateFrames.codebook_hi,
      0,
      7,
      hiCol,
      hiCol,
      8,
      128,
      0.038,
      UNI_FLOW_COLORS.activeOxide,
      UNI_FLOW_COLORS.activeOxide,
      0.018,
    );
  }

  if (activeBall.heapNodeIndex !== null) {
    const hRow = Math.floor(activeBall.heapNodeIndex / 16);
    const hCol = activeBall.heapNodeIndex % 16;
    flow.addCellHighlightOnFrame(
      plateFrames.heap_radii,
      hRow,
      hRow,
      hCol,
      hCol,
      16,
      16,
      0.038,
      UNI_FLOW_COLORS.activeOxide,
      UNI_FLOW_COLORS.activeOxide,
      0.022,
    );
  }

  // Highlight active depth row & active stage row on ball_membership
  flow.addCellHighlightOnFrame(
    plateFrames.ball_membership,
    activePolarity.level,
    activePolarity.level,
    0,
    63,
    16,
    64,
    0.038,
    UNI_FLOW_COLORS.gTeal,
    UNI_FLOW_COLORS.activeOxide,
    0.018,
  );
  flow.addCellHighlightOnFrame(
    plateFrames.ball_membership,
    10,
    10,
    0,
    63,
    16,
    64,
    0.038,
    UNI_FLOW_COLORS.jointCyan,
    UNI_FLOW_COLORS.gTeal,
    0.016,
  );

  // Clean XZ flow connectors across the 5 plates
  const FLOW_Y = SHEET_Y + 0.045;
  flow.addPolylineXZ(
    [
      { x: 2.75, z: -1.32 },
      { x: 2.75, z: -0.9 },
    ],
    FLOW_Y,
    0.14,
    UNI_FLOW_COLORS.gTeal,
    true,
  );
  flow.addPolylineXZ(
    [
      { x: 3.9, z: 0.25 },
      { x: 4.2, z: 0.25 },
    ],
    FLOW_Y,
    0.14,
    UNI_FLOW_COLORS.jointCyan,
    true,
  );
  flow.addPolylineXZ(
    [
      { x: 7.9, z: -1.25 },
      { x: 8.22, z: -1.25 },
    ],
    FLOW_Y,
    0.12,
    UNI_FLOW_COLORS.uIndigo,
    true,
  );

  // --- 8. Front-Right Bay: Interactive U26 Production D=17 Floor-Rounding Mismatch Stage ---
  const U26_BASE_X = 2.05;
  const U26_BASE_Z = 3.15;
  const u26Summary = formatU26MismatchSummary(u26Mismatch);

  const stageSpacing = refinementSteps === 8 ? 0.72 : refinementSteps === 4 ? 1.2 : 1.75;
  const u26StageWidth = stageSpacing * (refinementSteps + 1);
  lines.addBoxFrameXZ(
    U26_BASE_X - 0.35,
    U26_BASE_Z - 0.75,
    U26_BASE_X + u26StageWidth + 0.25,
    U26_BASE_Z + 0.95,
    0.02,
    showProductionD17Comparison ? 1.3 : 0.85,
    u26Mismatch.hasMismatch ? LINE_STYLE_BF16_GOLD : LINE_STYLE_GUIDE,
    showProductionD17Comparison ? 0.78 : 0.35,
  );

  for (let s = 0; s <= refinementSteps; s++) {
    const sx = U26_BASE_X + 0.3 + s * stageSpacing;
    const miniLvl = scheduleLevelsRes.levels[s]!;
    const miniBallLog = tree.depth - miniLvl; // 8 .. 0

    const prodTrainLvl =
      s === 0
        ? 0
        : s >= refinementSteps
          ? UNI_BALL_TREE_PROD_DEPTH
          : UNI_BALL_TREE_PROD_DEPTH -
            Math.floor(((refinementSteps - s) * UNI_BALL_TREE_PROD_DEPTH) / refinementSteps);
    const prodInferLvl =
      s === 0
        ? 0
        : s >= refinementSteps
          ? UNI_BALL_TREE_PROD_DEPTH
          : Math.floor((s * UNI_BALL_TREE_PROD_DEPTH) / refinementSteps);
    const isMismatched = prodTrainLvl !== prodInferLvl;
    const isCurrentStage = s === Math.max(0, stageIndex);

    const miniHeight = 0.14 + 0.95 * (miniBallLog / tree.depth);
    const prodTrainHeight =
      0.14 + 0.95 * ((UNI_BALL_TREE_PROD_DEPTH - prodTrainLvl) / UNI_BALL_TREE_PROD_DEPTH);
    const prodInferHeight =
      0.14 + 0.95 * ((UNI_BALL_TREE_PROD_DEPTH - prodInferLvl) / UNI_BALL_TREE_PROD_DEPTH);

    // Mini D=8 exact bar (back row)
    solids.push({
      cx: sx,
      cy: miniHeight * 0.5,
      cz: U26_BASE_Z - 0.35,
      ex: 0.24,
      ey: miniHeight,
      ez: 0.24,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8: 0,
      phase: s / Math.max(1, refinementSteps),
      bayId: 0,
      cellScale: 0.12,
      highlight: isCurrentStage ? 1 : 0,
    });

    // Production D=17 Train bar (front-left)
    solids.push({
      cx: sx - 0.12,
      cy: prodTrainHeight * 0.5,
      cz: U26_BASE_Z + 0.3,
      ex: 0.2,
      ey: prodTrainHeight,
      ez: 0.24,
      cellFamily: CELL_WEIGHT_BLUE,
      fp8: showProductionD17Comparison ? 0 : 1,
      phase: s / Math.max(1, refinementSteps),
      bayId: 0,
      cellScale: 0.12,
      highlight: isCurrentStage ? 1 : 0,
    });

    // Production D=17 Infer bar (front-right, amber when 2x wider mismatch!)
    solids.push({
      cx: sx + 0.12,
      cy: prodInferHeight * 0.5,
      cz: U26_BASE_Z + 0.3,
      ex: 0.2,
      ey: prodInferHeight,
      ez: 0.24,
      cellFamily: isMismatched ? CELL_MOE_AMBER : CELL_SSM_CYAN,
      fp8: showProductionD17Comparison ? 0 : 1,
      phase: s / Math.max(1, refinementSteps),
      bayId: 0,
      cellScale: 0.12,
      highlight: isMismatched && showProductionD17Comparison ? 2 : isCurrentStage ? 1 : 0,
    });

    if (isMismatched && showProductionD17Comparison) {
      ribbons.push({
        cx: sx + 0.12,
        cy: prodInferHeight + 0.09,
        cz: U26_BASE_Z + 0.3,
        ex: 0.22,
        ey: 0.08,
        ez: 0.26,
        kind: 0,
        colorId: 3, // Amber mismatch cap
        alpha: 0.52,
      });
    }

    expandAabbPoint(boundsAcc, sx, Math.max(miniHeight, prodInferHeight), U26_BASE_Z, 0.55);

    pickBoxes.push(
      packPickBoxFromAabb(sx, 0.55, U26_BASE_Z, 0.54, 1.15, 1.2, {
        bayId: 0,
        partId: `u26_stage_${s}`,
        tensorKey: 'u26_d17_comparison',
        tapIndex: miniLvl,
        towerRole: 'denoiser',
        semanticTag: `branch=g;tapRole=tree;depth=${miniLvl};u26Stage=${s};mismatch=${isMismatched ? 1 : 0}`,
      }),
    );
  }

  // --- 9. Overlay Anchors, Landmarks, Calipers & Provenance Badges ---
  const scheduleSeqLabel = scheduleBallSizes.join(' -> ');

  overlayAnchors.push(
    Object.freeze({
      id: 'anchor_pca_cloud',
      label: `3D PCA Codebook Cloud (256 Codes · Leaf #${tree.tokenToLeaf[selectedCodeId]})`,
      sublabel: `Code #${selectedCodeId} · N=${refinementSteps} (${scheduleSeqLabel}) · Stage ${activePolarity.codeStage} / PaperAge ${activePolarity.paperAge}`,
      worldPos: [CLOUD_CENTER[0], CLOUD_CENTER[1] + CLOUD_SCALE + 0.45, CLOUD_CENTER[2]] as Vec3,
      tone: 'g_teal',
      badge: `|N_${activePolarity.level}| = ${activePolarity.ballSize}`,
    }),
    Object.freeze({
      id: 'anchor_depth_ladder',
      label: 'Depth-8 BalancedSemanticTree Hierarchy (255 Monotone Radii)',
      sublabel: `Active depth ${activePolarity.level} · ballSize=${activePolarity.ballSize} · η=${activePolarity.noiseWeight.toFixed(2)}`,
      worldPos: [CLOUD_CENTER[0], 0.95, PEDESTAL_Z + 0.45] as Vec3,
      tone: 'joint',
      badge: refinementSteps === 1 ? 'N=1 Binary MDM (256 -> 1)' : `N=${refinementSteps} Progressive`,
    }),
    Object.freeze({
      id: 'anchor_u26_d17_mismatch',
      label: u26Summary.anchorLabel,
      sublabel: u26Summary.anchorSublabel,
      worldPos: [
        U26_BASE_X + u26StageWidth * 0.5,
        1.45,
        U26_BASE_Z + 0.2,
      ] as Vec3,
      tone: u26Mismatch.hasMismatch ? 'warn' : 'gold',
      badge: `U26 · D=17 (${u26Mismatch.mismatchedStageCount}/${u26Mismatch.totalInternalStages}) vs D=8 (0/${miniMismatch.totalInternalStages})`,
    }),
  );

  landmarks.push(
    Object.freeze({
      id: 'lm_codebook_pca_cloud',
      label: 'VQ Codebook 3D PCA + Ball-Tree',
      sublabel: `256 codes · depth=8 · |N_${activePolarity.level}(${selectedCodeId})|=${activePolarity.ballSize}`,
      world: [CLOUD_CENTER[0], CLOUD_CENTER[1] + CLOUD_SCALE + 0.65, CLOUD_CENTER[2]] as Vec3,
      tone: 'cyan',
      align: 'top',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_rg32f_tables',
      label: '128×128 RG32F Tree & SinCos Plates',
      sublabel: `5 rects · ${packedAtlas.audit.utilizationPct.toFixed(1)}% util · 0 overlaps`,
      world: [5.2, 0.55, -3.95] as Vec3,
      tone: 'default',
      align: 'top',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_u26_mismatch',
      label: 'U26: Production D=17 Floor-Rounding Mismatch',
      sublabel: u26Mismatch.hasMismatch
        ? `${u26Mismatch.mismatchedStageCount}/${u26Mismatch.totalInternalStages} internal stages 2× wider vs Mini D=8 (0)`
        : 'N=1 Binary MDM (0 internal stages) · K=4 D=17 has 3/3 2× wider',
      world: [U26_BASE_X + u26StageWidth * 0.5, 1.65, U26_BASE_Z + 0.35] as Vec3,
      tone: 'gold',
      align: 'bottom',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'cal_codebook_lo',
      aWorld: [
        plateFrames.codebook_lo.center[0] - plateFrames.codebook_lo.extent[0] * 0.5,
        SHEET_Y + 0.06,
        plateFrames.codebook_lo.center[2] - plateFrames.codebook_lo.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      bWorld: [
        plateFrames.codebook_lo.center[0] + plateFrames.codebook_lo.extent[0] * 0.5,
        SHEET_Y + 0.06,
        plateFrames.codebook_lo.center[2] - plateFrames.codebook_lo.extent[2] * 0.5 - 0.14,
      ] as Vec3,
      label: '128×8 codebook_lo',
      partIds: ['codebook_lo'],
    }),
    Object.freeze({
      id: 'cal_ball_membership',
      aWorld: [
        plateFrames.ball_membership.center[0] - plateFrames.ball_membership.extent[0] * 0.5,
        SHEET_Y + 0.06,
        plateFrames.ball_membership.center[2] + plateFrames.ball_membership.extent[2] * 0.5 + 0.14,
      ] as Vec3,
      bWorld: [
        plateFrames.ball_membership.center[0] + plateFrames.ball_membership.extent[0] * 0.5,
        SHEET_Y + 0.06,
        plateFrames.ball_membership.center[2] + plateFrames.ball_membership.extent[2] * 0.5 + 0.14,
      ] as Vec3,
      label: '16×64 ball_membership',
      partIds: ['ball_membership'],
    }),
  );

  const provenanceBadges: UniProvenanceBadge[] = [
    ...baseProvenanceBadges,
    Object.freeze({
      label: `Tree D=8 (256 codes · 255 heap_radii) · N=${refinementSteps}: ${scheduleSeqLabel}${refinementSteps === 1 ? ' (Binary MDM)' : ''}`,
      tone: 'g_teal' as const,
    }),
    Object.freeze({
      label: `U9 Polarity: Stage=${activePolarity.codeStage} / PaperAge=${activePolarity.paperAge} · Level=${activePolarity.level} (|N|=${activePolarity.ballSize}, η=${activePolarity.noiseWeight.toFixed(2)})`,
      tone: 'joint' as const,
    }),
    Object.freeze({
      label: `Mini D=8 Verification: 0/${miniMismatch.totalInternalStages} Floor Mismatches (Exact Power-of-Two)`,
      tone: 'g_teal' as const,
    }),
    Object.freeze({
      label: u26Summary.badgeLabel,
      tone: u26Mismatch.hasMismatch ? ('warn' as const) : ('gold' as const),
    }),
    Object.freeze({
      label: 'D4: 128×128 RG32F Scalar/Mask Only (5 Rects · 0 RGB · heap_radii[255] Masked)',
      tone: 'muted' as const,
    }),
  ];

  // --- 10. Pack all 5 passes & validate scene geometry ---
  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flow.pack();

  const fitBounds: AABB = Object.freeze({
    min: [boundsAcc.min[0] - 0.6, Math.min(0, boundsAcc.min[1]) - 0.25, boundsAcc.min[2] - 0.6] as Vec3,
    max: [boundsAcc.max[0] + 0.6, boundsAcc.max[1] + 0.85, boundsAcc.max[2] + 0.6] as Vec3,
  });

  const scene: UniSpatialScene = Object.freeze({
    viewMode: 'ball_tree' as const,
    title: 'View 4 · Codebook Geometry & Ball-Tree Inspector',
    subtitle: `Code #${selectedCodeId} (leaf ${tree.tokenToLeaf[selectedCodeId]}) · N=${refinementSteps} (${scheduleSeqLabel}) · Stage ${activePolarity.codeStage} -> |N_${activePolarity.level}|=${activePolarity.ballSize} · 5 RG32F rects (${packedAtlas.audit.utilizationPct.toFixed(1)}%)`,
    layerIndex: null,
    executionViewId,
    prefillMode,
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
      azimuthDeg: 24,
      elevationDeg: 34,
      fitBounds,
    }),
    fitBounds,
    valueAtlas: packedAtlas.atlas,
    atlasKey: packedAtlas.atlasKey,
    atlasRegions: packedAtlas.regions,
    atlasAudit: packedAtlas.audit,
    rgbPreviews: Object.freeze([]),
    provenanceBadges: Object.freeze(provenanceBadges),
    overlayAnchors: Object.freeze(overlayAnchors),
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze([]),
    calipers: Object.freeze(calipers),
    d5Comparison,
    viewReadout,
    u26Mismatch,
  });

  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `Invalid View 4 (ball_tree) scene geometry: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
