import type { HybridModelParameters } from '../math/hybridFlow';
import { NEMOTRON_SPEC, type PrecisionMode, type Vec3 } from '../model/spec';
import {
  DEFAULT_FOV_DEG,
  MAX_CAMERA_DISTANCE,
  MIN_CAMERA_DISTANCE,
  type CameraPose,
} from '../model/state';
import {
  aabbCorners,
  computeCameraBasis,
  dot3,
  type AABB,
  type ViewportMetrics,
} from '../render/camera';
import type { PickBox, PickHit } from '../render/pick';
import { FlowRibbonBuilder } from '../render/ribbons';
import {
  FRAME_XZ_AXES,
  FRAME_YZ_AXES,
  localToWorldOnFrame,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetBasisAxes,
  type SheetFrame3D,
} from '../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_GHOST_RAIL,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NEUTRAL,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_WEIGHT,
} from '../render/sheets';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
} from './atlasPacker';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../ui/overlay';
import type { CutawayPartInfo, CutawaySceneGeometry } from './cutaways';
import {
  CELL_OP_A_FILL,
  CELL_OP_A_FRAME,
  CELL_OP_B_FILL,
  CELL_OP_B_FRAME,
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  computeSheetGroupBounds,
  createSheetFocusSet,
  drawCellHighlight3D,
  drawOperandDependency3D,
  EMPTY_SHEET_FOCUS_SET,
  fmtSigned,
  resolveSheetHighlight,
  RIBBON_ACTIVE,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  RIBBON_RESIDUAL_ACTIVE,
  spanFromSheetGrid,
  type LayerFormulaReadout,
  type SheetFocusEntry,
  type SheetFocusSet,
  type SheetGeomRecord,
  type StepperStepDef,
  type WorldSpan3D,
} from './sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_NEUTRAL,
  CELL_WEIGHT_BLUE,
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_GUIDE,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type LandmarkAnchor,
} from './spine';

/**
 * Live Float32 embedding activation payload supplied to the Embedding (NoPE) cutaway.
 * Accepts `X_embed` directly from `HybridExecutionResult` or `HybridIncrementalStepResult`
 * as `{ X: exec.X_embed }`.
 */
export interface HybridEmbeddingResult {
  /** Embedded activation tensor X₀ [d, T], channel-major indexed [ch * T + t]. */
  readonly X: Float32Array;
  /** Optional hidden dimension override (defaults to `model.config.d` = 8). */
  readonly d?: number;
}

export type EmbeddingTensorKey = 'token_strip' | 'w_embed' | 'x_embed';

export interface EmbeddingCellRef {
  readonly tensor: EmbeddingTensorKey;
  /** Logical sheet row: token index for `token_strip`, vocab ID for `w_embed`, channel for `x_embed`. */
  readonly row: number;
  /** Logical sheet col: 0 for `token_strip`, channel for `w_embed`, token index for `x_embed`. */
  readonly col: number;
  readonly tokenIndex?: number;
  readonly vocabId?: number;
}

export interface EmbeddingSheetSpan {
  readonly tensor: EmbeddingTensorKey;
  readonly rowStart: number;
  readonly rowEnd: number;
  readonly colStart: number;
  readonly colEnd: number;
  readonly role: 'target' | 'operand_a' | 'operand_b';
}

export interface EmbeddingCellFormula extends LayerFormulaReadout {
  readonly cell: EmbeddingCellRef;
  readonly isMasked: false;
  readonly targetSpan: EmbeddingSheetSpan;
  readonly upstreamSpans: readonly EmbeddingSheetSpan[];
}

export interface EmbeddingFormulaInput {
  readonly model: HybridModelParameters;
  readonly embedding: HybridEmbeddingResult;
  readonly tokenIds: readonly number[] | Int32Array;
  readonly tokenLabels: readonly string[];
  readonly startTokenOffset?: number;
  readonly focusToken?: number;
}

export interface BuildEmbeddingCutawayOptions extends EmbeddingFormulaInput {
  readonly precision: PrecisionMode;
  readonly selectedPart: string | null;
  readonly hoveredPart: string | null;
  readonly activeCell?: EmbeddingCellRef | null;
}

export interface SpatialEmbeddingFitOptions {
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly isNarrow?: boolean;
  readonly azimuthDeg?: number;
  readonly elevationDeg?: number;
}

/**
 * Invariant 3D bounding envelope for the Dedicated Embedding (NoPE) Cutaway across
 * T = 1..16 (Full Seq and AR Decode S = 1), BF16 / FP8 Selective, and all inspected cells.
 */
export const EMBEDDING_SPATIAL_FIT_BOUNDS: AABB = {
  min: [-7.10, -0.06, -2.85],
  max: [7.20, 3.10, 2.15],
};

export const EMBEDDING_STEPPER_STEPS: readonly StepperStepDef[] = Object.freeze([
  {
    index: 0,
    shortLabel: 'Tokens',
    title: '1. Input Token Strip',
    summary: 'Discrete token IDs index directly into the untied embedding table with zero positional encoding.',
  },
  {
    index: 1,
    shortLabel: 'W_embed',
    title: '2. Untied W_embed Row Lookup',
    summary: 'Each token ID selects one row of W_embed [V × d] (131,072 × 2,688 prod, untied from W_lm_head).',
  },
  {
    index: 2,
    shortLabel: 'NoPE Copy',
    title: '3. Direct NoPE Copy into X₀',
    summary: 'Looked-up row copies verbatim into column t of X₀ with NO positional embedding (NoPE).',
  },
  {
    index: 3,
    shortLabel: '→ Bay 01',
    title: '4. Residual Highway → Bay 01 Mamba-2',
    summary: 'X₀ enters the emerald Residual Highway bit-exact as Bay 01 Mamba-2 Input X, where Conv1D + SSM inject order.',
  },
]);

/**
 * Exact 8-corner projective camera fitter for the 3D Embedding (NoPE) Cutaway.
 * Keeps all 8 corners of `bounds` strictly inside the unobstructed viewport band.
 */
export function fitSpatialEmbeddingCamera(
  metrics: ViewportMetrics,
  bounds: AABB = EMBEDDING_SPATIAL_FIT_BOUNDS,
  options?: SpatialEmbeddingFitOptions,
): CameraPose {
  const isNarrow = options?.isNarrow ?? metrics.innerWidth < 560;
  const topInset = Math.max(0, options?.topInsetPx ?? (isNarrow ? 12 : 84));
  const bottomInset = Math.max(0, options?.bottomInsetPx ?? (isNarrow ? 12 : 118));
  const azimuthDeg = options?.azimuthDeg ?? (isNarrow ? 17 : 21);
  const elevationDeg = options?.elevationDeg ?? (isNarrow ? 24 : 22.5);
  const fovDeg = DEFAULT_FOV_DEG;
  const fillX = isNarrow ? 0.94 : 0.95;
  const fillY = isNarrow ? 0.90 : 0.92;

  const basis = computeCameraBasis(azimuthDeg, elevationDeg);
  const corners = aabbCorners(bounds);

  let minR = Infinity, maxR = -Infinity;
  let minU = Infinity, maxU = -Infinity;
  let minE = Infinity, maxE = -Infinity;

  for (const p of corners) {
    const r = dot3(p, basis.right);
    const u = dot3(p, basis.up);
    const e = dot3(p, basis.eyeDir);
    if (r < minR) minR = r;
    if (r > maxR) maxR = r;
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    if (e < minE) minE = e;
    if (e > maxE) maxE = e;
  }

  const midR = 0.5 * (minR + maxR);
  const midU = 0.5 * (minU + maxU);
  const midE = 0.5 * (minE + maxE);

  const safeW = Math.max(1, metrics.innerWidth);
  const safeH = Math.max(1, metrics.innerHeight);
  const clampedTop = Math.min(topInset, safeH * 0.42);
  const clampedBottom = Math.min(bottomInset, safeH * 0.42);
  const usableH = Math.max(80, safeH - clampedTop - clampedBottom);
  const usableFracY = Math.max(0.25, Math.min(1, usableH / safeH));

  const aspect = Math.max(0.25, safeW / safeH);
  const tanHalfFovY = Math.tan((fovDeg * 0.5) * (Math.PI / 180));
  const tanHalfFovX = tanHalfFovY * aspect;

  const ndcYCenter = (clampedBottom - clampedTop) / safeH;
  const ndcYHalfSpan = Math.max(0.15, fillY * usableFracY);
  const ndcYMax = ndcYCenter + ndcYHalfSpan;
  const ndcYMin = ndcYCenter - ndcYHalfSpan;

  let requiredDist = MIN_CAMERA_DISTANCE;
  for (const p of corners) {
    const dr = dot3(p, basis.right) - midR;
    const du = dot3(p, basis.up) - midU;
    const de = dot3(p, basis.eyeDir) - midE;

    const dX = de + Math.abs(dr) / (tanHalfFovX * fillX);
    const dYTop = (du / tanHalfFovY + de * ndcYMax) / ndcYHalfSpan;
    const dYBot = (-du / tanHalfFovY - de * ndcYMin) / ndcYHalfSpan;
    const dCorner = Math.max(dX, dYTop, dYBot, de + 2.0);
    if (dCorner > requiredDist) requiredDist = dCorner;
  }

  const distance = Math.max(MIN_CAMERA_DISTANCE, Math.min(MAX_CAMERA_DISTANCE, requiredDist));
  const targetShiftU = -ndcYCenter * distance * tanHalfFovY;

  const target: Vec3 = [
    midR * basis.right[0] + (midU + targetShiftU) * basis.up[0] + midE * basis.eyeDir[0],
    midR * basis.right[1] + (midU + targetShiftU) * basis.up[1] + midE * basis.eyeDir[1],
    midR * basis.right[2] + (midU + targetShiftU) * basis.up[2] + midE * basis.eyeDir[2],
  ];

  return {
    target,
    azimuthDeg,
    elevationDeg,
    distance,
    fovDeg,
  };
}

interface EmbeddingValueAtlas extends Rg32fValueAtlas {
  readonly tokenStrip: AtlasSlice;
  readonly wEmbed: AtlasSlice;
  readonly xEmbed: AtlasSlice;
}

const EMBED_ATLAS_W = 32;
const EMBED_ATLAS_H = 64;

/**
 * Pack exact live Float32 tensors from `model.W_embed`, `embedding.X`, and `tokenIds`
 * into a compact 32×64 RG32F value atlas:
 *   y =  0..31, x = 0..7    : W_embed [32 × 8]  (row = vocabId, col = ch)
 *   y = 32..39, x = 0..T-1  : X₀ [8 × T]        (row = ch, col = token)
 *   y = 40..40+T-1, x = 0   : token_strip [T×1] (row = token, col = 0)
 */
function buildEmbeddingValueAtlas(
  model: HybridModelParameters,
  embedding: HybridEmbeddingResult,
  tokenIds: readonly number[] | Int32Array,
  T: number,
): EmbeddingValueAtlas {
  const d = embedding.d ?? model.config.d;
  const vocabSize = model.config.vocabSize;
  const writer = createRg32fAtlas(EMBED_ATLAS_W, EMBED_ATLAS_H);
  const { data, setTexel, makeRect } = writer;

  // 1. Untied W_embed [32, 8] bit-exact Float32
  for (let v = 0; v < vocabSize; v++) {
    const rowOff = v * d;
    for (let ch = 0; ch < d; ch++) {
      setTexel(ch, v, model.W_embed[rowOff + ch]);
    }
  }

  // 2. X₀ [8, T] bit-exact Float32 (row = ch, col = t)
  for (let ch = 0; ch < d; ch++) {
    for (let t = 0; t < T; t++) {
      setTexel(t, 32 + ch, embedding.X[ch * T + t]);
    }
  }

  // 3. Input Token Strip [T, 1] exact token IDs scaled into [0..1] positive palette
  for (let t = 0; t < T; t++) {
    const id = Math.max(0, Math.min(vocabSize - 1, Math.trunc(tokenIds[t] ?? 0)));
    setTexel(0, 40 + t, id);
  }

  const scaleW = 1 / computeMaxAbs(model.W_embed, 1.0);
  const scaleX = 1 / computeMaxAbs(embedding.X, 1.0);
  const scaleTok = 1 / Math.max(1, vocabSize - 1);

  return {
    width: EMBED_ATLAS_W,
    height: EMBED_ATLAS_H,
    data,
    wEmbed: { rect: makeRect(0, 0, d, vocabSize), scale: scaleW, mode: 0 },
    xEmbed: { rect: makeRect(0, 32, T, d), scale: scaleX, mode: 0 },
    tokenStrip: { rect: makeRect(0, 40, 1, T), scale: scaleTok, mode: 1 },
  };
}

/**
 * Exact mathematical formula & operand provenance for any cell in the Embedding (NoPE) cutaway.
 */
export function describeEmbeddingCellFormula(
  input: EmbeddingFormulaInput,
  cell: EmbeddingCellRef,
): EmbeddingCellFormula {
  const { model, embedding, tokenIds, tokenLabels } = input;
  const d = embedding.d ?? model.config.d;
  const vocabSize = model.config.vocabSize;
  const T = Math.max(1, Math.min(16, tokenIds.length));
  const offset = input.startTokenOffset ?? 0;

  const clampT = (t: number) => Math.max(0, Math.min(T - 1, Math.trunc(t)));
  const clampV = (v: number) => Math.max(0, Math.min(vocabSize - 1, Math.trunc(v)));
  const clampCh = (c: number) => Math.max(0, Math.min(d - 1, Math.trunc(c)));
  const labelForT = (t: number) => tokenLabels[t] ?? model.vocab[tokenIds[t] ?? 0] ?? `t${offset + t}`;
  const labelForV = (v: number) => model.vocab[v] ?? `id:${v}`;

  if (cell.tensor === 'token_strip') {
    const t = clampT(cell.tokenIndex ?? cell.row);
    const absT = offset + t;
    const vid = clampV(cell.vocabId ?? tokenIds[t] ?? 0);
    const word = labelForT(t);
    return {
      cell: { tensor: 'token_strip', row: t, col: 0, tokenIndex: t, vocabId: vid },
      tensorLabel: 'Input Token Strip',
      coordLabel: `t${absT} "${word}" (id=${vid})`,
      headBadge: `NoPE · id ${vid} / ${vocabSize} mini (${NEMOTRON_SPEC.vocabSize.toLocaleString()} prod)`,
      value: vid,
      isMasked: false,
      valueText: `#${vid} ("${word}")`,
      equation: `token[t${absT}] = "${word}" → lookup row #${vid} of untied W_embed[${vid}, 0..${d - 1}]`,
      subEquation: 'No positional embedding added (NoPE) · Order comes from Bay 01 Mamba-2 Conv1D + SSM',
      targetSpan: { tensor: 'token_strip', rowStart: t, rowEnd: t, colStart: 0, colEnd: 0, role: 'target' },
      upstreamSpans: [],
    };
  }

  if (cell.tensor === 'w_embed') {
    const vid = clampV(cell.vocabId ?? cell.row);
    const ch = clampCh(cell.col);
    const word = labelForV(vid);
    const val = model.W_embed[vid * d + ch];

    // If this vocab row is referenced by focusToken (or any token in the strip), link that token slot
    const preferredT = clampT(input.focusToken ?? T - 1);
    let matchedT = tokenIds[preferredT] === vid ? preferredT : -1;
    if (matchedT < 0) {
      for (let t = 0; t < T; t++) {
        if (tokenIds[t] === vid) {
          matchedT = t;
          break;
        }
      }
    }

    return {
      cell: {
        tensor: 'w_embed',
        row: vid,
        col: ch,
        vocabId: vid,
        ...(matchedT >= 0 ? { tokenIndex: matchedT } : {}),
      },
      tensorLabel: 'Untied Embedding Table W_embed',
      coordLabel: `W_embed[v=${vid} "${word}", ch=${ch}]`,
      headBadge: `Untied (≠ W_lm_head) · [${vocabSize}×${d}] mini · [131,072×2,688] prod`,
      value: val,
      isMasked: false,
      valueText: fmtSigned(val),
      equation: `W_embed[id=${vid} ("${word}"), ch${ch}] = ${fmtSigned(val)}`,
      subEquation:
        matchedT >= 0
          ? `Indexed by t${offset + matchedT} ("${labelForT(matchedT)}") · tie_word_embeddings = false (independent of W_lm_head)`
          : 'Learned untied token lookup row (tie_word_embeddings = false) · Unit-RMS normalized',
      targetSpan: { tensor: 'w_embed', rowStart: vid, rowEnd: vid, colStart: ch, colEnd: ch, role: 'target' },
      upstreamSpans:
        matchedT >= 0
          ? [{ tensor: 'token_strip', rowStart: matchedT, rowEnd: matchedT, colStart: 0, colEnd: 0, role: 'operand_a' }]
          : [],
    };
  }

  // tensor === 'x_embed'
  const ch = clampCh(cell.row);
  const t = clampT(cell.tokenIndex ?? cell.col);
  const absT = offset + t;
  const vid = clampV(cell.vocabId ?? tokenIds[t] ?? 0);
  const word = labelForT(t);
  const val = embedding.X[ch * T + t];
  const wVal = model.W_embed[vid * d + ch];

  return {
    cell: { tensor: 'x_embed', row: ch, col: t, tokenIndex: t, vocabId: vid },
    tensorLabel: 'Embedding Output X₀ (NoPE)',
    coordLabel: `X₀[ch=${ch}, t${absT} "${word}"]`,
    headBadge: 'NoPE Direct Copy · X₀ === Bay 01 Input X',
    value: val,
    isMasked: false,
    valueText: fmtSigned(val),
    equation: `X₀[ch${ch}, t${absT}] = W_embed[id=${vid} ("${word}"), ch${ch}] = ${fmtSigned(val)}`,
    subEquation: `Bit-exact row copy (${fmtSigned(wVal)}) + 0 Pos Embed (NoPE) → enters Bay 01 Mamba-2 Residual Highway`,
    targetSpan: { tensor: 'x_embed', rowStart: ch, rowEnd: ch, colStart: t, colEnd: t, role: 'target' },
    upstreamSpans: [
      { tensor: 'token_strip', rowStart: t, rowEnd: t, colStart: 0, colEnd: 0, role: 'operand_a' },
      { tensor: 'w_embed', rowStart: vid, rowEnd: vid, colStart: ch, colEnd: ch, role: 'operand_b' },
    ],
  };
}

/** Resolve a ray-cast PickHit on the Embedding (NoPE) cutaway into an exact `EmbeddingCellRef`. */
export function pickHitToEmbeddingCell(
  hit: PickHit | null,
  tokenIds?: readonly number[] | Int32Array,
): EmbeddingCellRef | null {
  if (!hit || !hit.tensorKey || hit.hitRow === undefined || hit.hitCol === undefined) {
    return null;
  }
  if (hit.tensorKey === 'token_strip') {
    const t = hit.hitRow;
    return {
      tensor: 'token_strip',
      row: t,
      col: 0,
      tokenIndex: t,
      vocabId: tokenIds ? tokenIds[t] : undefined,
    };
  }
  if (hit.tensorKey === 'w_embed') {
    return {
      tensor: 'w_embed',
      row: hit.hitRow,
      col: hit.hitCol,
      vocabId: hit.hitRow,
    };
  }
  if (hit.tensorKey === 'x_embed') {
    const t = hit.hitCol;
    return {
      tensor: 'x_embed',
      row: hit.hitRow,
      col: t,
      tokenIndex: t,
      vocabId: tokenIds ? tokenIds[t] : undefined,
    };
  }
  return null;
}

interface EmbeddingSheetSpec {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly axes: SheetBasisAxes;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly sheetKind: number;
  readonly role: number;
  readonly fp8Flag: number;
  readonly highlight: number;
  readonly seed: number;
  readonly partId: string;
  readonly tensorKey?: string;
  readonly tokenIndex?: number;
  readonly atlasSlice?: AtlasSlice;
  readonly registryKey?: string;
  readonly cellFamily: number;
}

class EmbeddingSceneBuilder {
  readonly sheetFloats: number[] = [];
  readonly basisFloats: number[] = [];
  readonly solidFloats: number[] = [];
  readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();

  addSheet(s: EmbeddingSheetSpec): SheetFrame3D {
    const frame = makeSheetFrame3D(
      [s.cx, s.cy, s.cz],
      [s.ex, s.ey, s.ez],
      s.axes.axisU,
      s.axes.axisN,
    );

    const rect = s.atlasSlice ? s.atlasSlice.rect : ([0, 0, 0, 0] as const);
    const scale = s.atlasSlice ? s.atlasSlice.scale : 1.0;
    const modeFlag = s.atlasSlice ? s.atlasSlice.mode : 0;

    this.sheetFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.gridCols, s.gridRows,
      s.sheetKind,
      s.role,
      s.fp8Flag,
      s.highlight,
      s.seed,
      rect[0], rect[1], rect[2], rect[3],
      scale,
      modeFlag, 0,
    );

    this.basisFloats.push(
      frame.axisU[0], frame.axisU[1], frame.axisU[2],
      frame.axisN[0], frame.axisN[1], frame.axisN[2],
    );

    this.solidFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.cellFamily,
      s.fp8Flag,
      s.seed,
      0, // Embedding bayId = 0
      0.2,
      s.highlight === SHEET_HL_ACTIVE ? 1 : s.highlight === SHEET_HL_BF16_GOLD ? 2 : 0,
    );

    const pickFrame = makeSheetFrame3D(
      frame.center,
      [frame.extent[0], Math.max(frame.extent[1], 0.085), frame.extent[2]],
      frame.axisU,
      frame.axisN,
    );
    const worldAabb = sheetFrameWorldAabb(pickFrame);
    this.pickBoxes.push({
      bayId: 0,
      partId: s.partId,
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
      min: worldAabb.min,
      max: worldAabb.max,
      frame: pickFrame,
    });

    if (s.registryKey) {
      this.sheetRegistry.set(s.registryKey, {
        cx: s.cx,
        cy: s.cy,
        cz: s.cz,
        ex: s.ex,
        ey: s.ey,
        ez: s.ez,
        gridCols: s.gridCols,
        gridRows: s.gridRows,
        frame,
      });
    }

    return frame;
  }

  addGuideLine(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth = 1.0,
    style = LINE_STYLE_GUIDE,
    alpha = 0.5,
  ): void {
    this.lineFloats.push(
      ax, ay, az,
      bx, by, bz,
      0, 1, 0,
      0, 1, 0,
      cssWidth,
      style,
      alpha,
      0, 0, 0,
    );
  }

  pack() {
    const flowPacked = this.flow.pack();
    return {
      sheetData: new Float32Array(this.sheetFloats),
      sheetCount: this.sheetFloats.length / SHEET_INSTANCE_FLOATS,
      sheetBasisData: new Float32Array(this.basisFloats),
      solidData: new Float32Array(this.solidFloats),
      solidCount: this.solidFloats.length / SOLID_INSTANCE_FLOATS,
      lineData: new Float32Array(this.lineFloats),
      lineCount: this.lineFloats.length / LINE_INSTANCE_FLOATS,
      ribbonData: new Float32Array(0),
      ribbonCount: 0,
      flowVertices: flowPacked.vertices,
      flowIndices: flowPacked.indices,
    };
  }
}

function buildEmbeddingFocusSet(formula: EmbeddingCellFormula | null): SheetFocusSet {
  if (!formula) return EMPTY_SHEET_FOCUS_SET;
  const toEntry = (span: EmbeddingSheetSpan): SheetFocusEntry => ({
    tensor: span.tensor,
    rowStart: span.rowStart,
    rowEnd: span.rowEnd,
    colStart: span.colStart,
    colEnd: span.colEnd,
    role: span.role,
  });
  return createSheetFocusSet([
    toEntry(formula.targetSpan),
    ...formula.upstreamSpans.map(toEntry),
  ]);
}

function buildEmbeddingPartCatalog(
  T: number,
  d: number,
  vocabSize: number,
  absFocusT: number,
  focusVocabId: number,
  focusWord: string,
  precision: PrecisionMode,
): Record<string, CutawayPartInfo> {
  const isFp8 = precision === 'fp8_selective';
  return {
    'token-strip': {
      id: 'token-strip',
      title: 'Input Token Strip',
      badge: `1 × ${T} · Focus t${absFocusT} "${focusWord}"`,
      summary:
        'Discrete input token IDs entering the model. Each token indexes exactly one row of the untied embedding matrix with zero positional encoding added.',
      specs: [
        `Active token: t${absFocusT} "${focusWord}" (vocab ID #${focusVocabId})`,
        `Miniature sequence length: T = ${T} · Vocab = ${vocabSize} synthetic (${NEMOTRON_SPEC.vocabSize.toLocaleString()} prod)`,
        'Positional encoding: None (NoPE)',
      ],
    },
    'w-embed': {
      id: 'w-embed',
      title: 'Untied Token Embedding Table (W_embed)',
      badge: isFp8 ? 'BF16 Preserved · Untied' : 'BF16 · Untied',
      summary:
        'Learned input embedding lookup table strictly untied from the final LM head (tie_word_embeddings: false). Each token ID selects its row directly without matrix multiplication.',
      specs: [
        `Production shape: ${NEMOTRON_SPEC.vocabSize.toLocaleString()} × ${NEMOTRON_SPEC.hidden.toLocaleString()} (352.3M params)`,
        `Miniature Float32 table: ${vocabSize} × ${d} (row #${focusVocabId} active)`,
        'tie_word_embeddings: false (W_embed ≠ W_lm_head)',
      ],
    },
    'nope-slot': {
      id: 'nope-slot',
      title: 'Absent Positional Embedding (NoPE)',
      badge: 'NoPE · 0 Params · Zero RoPE',
      summary:
        'Unlike GPT-2 (learned W_pos) or Llama (RoPE), Nemotron 3 Nano omits explicit positional embeddings entirely. Sequence order emerges naturally from Bay 01 Mamba-2’s causal 4-tap Conv1D and selective SSM recurrence.',
      specs: [
        'Positional encoding: NoPE (zero W_pos table, zero RoPE rotation)',
        'Order source: 23 causal Mamba-2 bays (starting immediately in Bay 01)',
        'Benefit: Clean long-context extrapolation up to 1M tokens without RoPE scaling artifacts',
      ],
    },
    'x-embed': {
      id: 'x-embed',
      title: 'Embedding Output Deck (X₀)',
      badge: `[${d} × ${T}] · Bit-Exact Row Copy`,
      summary:
        'Initial residual stream X₀ assembled directly by copying W_embed[tokenIds[t], :] into column t for every token position. Enters Bay 01 Mamba-2 unchanged.',
      specs: [
        `Formula: X₀[ch, t] = W_embed[tokenIds[t], ch] (+ 0 Pos Embed)`,
        `Active column: t${absFocusT} "${focusWord}" ← W_embed[${focusVocabId}, :]`,
        'Bit-exact identity: X₀ === Bay 01 Mamba-2 Input X',
      ],
    },
    'residual-handoff': {
      id: 'residual-handoff',
      title: 'Residual Highway → Bay 01 Mamba-2',
      badge: '2,688-wide Emerald Stream',
      summary:
        'The continuous emerald residual highway carrying X₀ directly into Bay 01 (the first Mamba-2 layer) and onward through all 52 hybrid layers.',
      specs: [
        'Width: d = 8 miniature (2,688 production hidden channels)',
        'Downstream consumer: Bay 01 Mamba-2 Pre-RMSNorm (ε = 1e−5)',
        'Bit-exact hand-off verified across unified 52-layer execution and AR decode',
      ],
    },
  };
}

/**
 * Build the Dedicated 3D Thin-Sheet Embedding (NoPE) Cutaway geometry,
 * backed 100% by live Float32 `model.W_embed` and `embedding.X` tensors packed
 * into an RG32F WebGL2 texture atlas.
 */
export function buildEmbeddingCutaway(
  opts: BuildEmbeddingCutawayOptions,
): CutawaySceneGeometry {
  const {
    model,
    embedding,
    tokenIds,
    tokenLabels,
    precision,
    selectedPart,
    hoveredPart,
    activeCell,
  } = opts;

  const T = Math.max(1, Math.min(16, tokenIds.length));
  const d = embedding.d ?? model.config.d;
  const vocabSize = model.config.vocabSize;
  const offset = opts.startTokenOffset ?? 0;
  const focusTok = Math.max(0, Math.min(T - 1, Math.trunc(opts.focusToken ?? T - 1)));
  const absFocusT = offset + focusTok;
  const focusVocabId = Math.max(0, Math.min(vocabSize - 1, Math.trunc(tokenIds[focusTok] ?? 0)));
  const focusWord = tokenLabels[focusTok] ?? model.vocab[focusVocabId] ?? `t${absFocusT}`;
  const isFp8 = precision === 'fp8_selective';

  const activeFormula = activeCell
    ? describeEmbeddingCellFormula(opts, activeCell)
    : null;
  const focusSet = buildEmbeddingFocusSet(activeFormula);
  const hasCellTrace = focusSet.active;
  const activePart = hoveredPart ?? selectedPart;
  const hasFocus = activePart !== null || hasCellTrace;

  const atlas = buildEmbeddingValueAtlas(model, embedding, tokenIds, T);
  const b = new EmbeddingSceneBuilder();

  const SHEET_THICK = 0.060;
  const BASE_Y = 0.12;

  // Stage X coordinates (left -> right +X)
  const xTokenStrip = -6.05;
  const xWEmbed = -2.45;
  const xNopeSlot = 0.15;
  const xX0 = 2.65;
  const xHandoff = 5.95;

  // Responsive geometric footprints (clean at both T=1 AR Decode and T=8..16 Full Seq)
  const tokenStripW = 1.10;
  const tokenStripD = Math.max(0.66, Math.min(2.60, 0.30 * T + 0.36));

  const wEmbedY = 1.48;
  const wEmbedSpanZ = 1.80;   // 8 hidden channels along -Z
  const wEmbedHeightY = 2.72; // 32 vocab rows top -> bottom along -Y

  const x0WidthX = Math.max(0.76, Math.min(2.64, 0.28 * T + 0.44)); // T token cols along +X
  const x0DepthZ = 1.92;                                             // 8 hidden channel rows along +Z

  const handoffWidthX = Math.max(0.68, Math.min(1.42, 0.14 * T + 0.48));
  const handoffDepthZ = x0DepthZ;

  // -------------------------------------------------------------------------
  // 1. Left: 3D Input Token Strip [1 × T] (horizontal XZ deck)
  // -------------------------------------------------------------------------
  const tokFallbackHl =
    activePart === 'token-strip'
      ? SHEET_HL_ACTIVE
      : activePart === 'w-embed' || activePart === 'x-embed'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;
  const tokHl = resolveSheetHighlight(focusSet, { tensor: 'token_strip' }, tokFallbackHl);

  const tokenStripFrame = b.addSheet({
    cx: xTokenStrip,
    cy: BASE_Y,
    cz: 0,
    ex: tokenStripW,
    ey: SHEET_THICK,
    ez: tokenStripD,
    axes: FRAME_XZ_AXES,
    gridCols: 1,
    gridRows: T,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: tokHl,
    seed: 1.3 + T * 0.1,
    partId: 'token-strip',
    tensorKey: 'token_strip',
    tokenIndex: focusTok,
    atlasSlice: atlas.tokenStrip,
    registryKey: 'token_strip',
    cellFamily: CELL_EMBED_SLATE,
  });

  // -------------------------------------------------------------------------
  // 2. Center-left: Upright Untied W_embed [32 × 8] (FRAME_YZ_AXES)
  // -------------------------------------------------------------------------
  const wFallbackHl =
    activePart === 'w-embed'
      ? SHEET_HL_ACTIVE
      : activePart === 'token-strip' || activePart === 'x-embed'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : isFp8
            ? SHEET_HL_BF16_GOLD
            : SHEET_HL_NORMAL;
  const wHl = resolveSheetHighlight(focusSet, { tensor: 'w_embed' }, wFallbackHl);

  const wEmbedFrame = b.addSheet({
    cx: xWEmbed,
    cy: wEmbedY,
    cz: 0,
    ex: wEmbedSpanZ,
    ey: SHEET_THICK,
    ez: wEmbedHeightY,
    axes: FRAME_YZ_AXES,
    gridCols: d,
    gridRows: vocabSize,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wHl,
    seed: 4.2,
    partId: 'w-embed',
    tensorKey: 'w_embed',
    atlasSlice: atlas.wEmbed,
    registryKey: 'w_embed',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Subtle floor stanchions grounding upright W_embed
  b.addGuideLine(xWEmbed, 0.0, -wEmbedSpanZ * 0.5, xWEmbed, wEmbedY - wEmbedHeightY * 0.5, -wEmbedSpanZ * 0.5, 0.85, LINE_STYLE_GUIDE, 0.35);
  b.addGuideLine(xWEmbed, 0.0, wEmbedSpanZ * 0.5, xWEmbed, wEmbedY - wEmbedHeightY * 0.5, wEmbedSpanZ * 0.5, 0.85, LINE_STYLE_GUIDE, 0.35);

  // -------------------------------------------------------------------------
  // 3. Explicit NoPE Contrast Slot (Ghost Rail beside the copy bridge)
  // -------------------------------------------------------------------------
  const nopeZ = -2.10;
  const nopeW = 1.85;
  const nopeD = 0.92;
  const nopeHl =
    hasCellTrace
      ? SHEET_HL_DIMMED
      : activePart === 'nope-slot'
        ? SHEET_HL_ACTIVE
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;

  b.addSheet({
    cx: xNopeSlot,
    cy: BASE_Y + 0.02,
    cz: nopeZ,
    ex: nopeW,
    ey: 0.048,
    ez: nopeD,
    axes: FRAME_XZ_AXES,
    gridCols: d,
    gridRows: Math.max(2, Math.min(8, T)),
    sheetKind: SHEET_KIND_GHOST_RAIL,
    role: SHEET_ROLE_NEUTRAL,
    fp8Flag: 0,
    highlight: nopeHl,
    seed: 7.7,
    partId: 'nope-slot',
    registryKey: 'nope_slot',
    cellFamily: CELL_NEUTRAL,
  });

  // Delicate diagonal X cross-lines over the NoPE Ghost Rail emphasizing "NO Positional Embedding"
  const nopeY = BASE_Y + 0.056;
  b.addGuideLine(
    xNopeSlot - nopeW * 0.44, nopeY, nopeZ - nopeD * 0.40,
    xNopeSlot + nopeW * 0.44, nopeY, nopeZ + nopeD * 0.40,
    1.1, LINE_STYLE_GUIDE, hasCellTrace ? 0.20 : 0.55,
  );
  b.addGuideLine(
    xNopeSlot - nopeW * 0.44, nopeY, nopeZ + nopeD * 0.40,
    xNopeSlot + nopeW * 0.44, nopeY, nopeZ - nopeD * 0.40,
    1.1, LINE_STYLE_GUIDE, hasCellTrace ? 0.20 : 0.55,
  );

  // -------------------------------------------------------------------------
  // 4. Center-right: Horizontal X₀ [8 × T] Activation Deck
  // -------------------------------------------------------------------------
  const x0FallbackHl =
    activePart === 'x-embed'
      ? SHEET_HL_ACTIVE
      : activePart === 'w-embed' || activePart === 'token-strip' || activePart === 'residual-handoff'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;
  const x0Hl = resolveSheetHighlight(focusSet, { tensor: 'x_embed' }, x0FallbackHl);

  const x0Frame = b.addSheet({
    cx: xX0,
    cy: BASE_Y,
    cz: 0,
    ex: x0WidthX,
    ey: SHEET_THICK,
    ez: x0DepthZ,
    axes: FRAME_XZ_AXES,
    gridCols: T,
    gridRows: d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: x0Hl,
    seed: 9.4 + T * 0.1,
    partId: 'x-embed',
    tensorKey: 'x_embed',
    tokenIndex: focusTok,
    atlasSlice: atlas.xEmbed,
    registryKey: 'x_embed',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // -------------------------------------------------------------------------
  // 5. Right: Emerald Residual Highway Terminal → Bay 01 Mamba-2
  // -------------------------------------------------------------------------
  const handoffHl =
    hasCellTrace
      ? x0Hl !== SHEET_HL_DIMMED
        ? SHEET_HL_PATHWAY
        : SHEET_HL_DIMMED
      : activePart === 'residual-handoff'
        ? SHEET_HL_ACTIVE
        : activePart === 'x-embed'
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : SHEET_HL_NORMAL;

  b.addSheet({
    cx: xHandoff,
    cy: BASE_Y,
    cz: 0,
    ex: handoffWidthX,
    ey: SHEET_THICK,
    ez: handoffDepthZ,
    axes: FRAME_XZ_AXES,
    gridCols: T,
    gridRows: d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: handoffHl,
    seed: 12.1 + T * 0.1,
    partId: 'residual-handoff',
    atlasSlice: atlas.xEmbed, // X₀ === Bay 01 Input X bit-exact
    registryKey: 'residual_handoff',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // -------------------------------------------------------------------------
  // 6. Focus-Token Row Highlight + Smooth 3D Lookup & Emerald Copy Ribbons
  // -------------------------------------------------------------------------
  const tokRec = b.sheetRegistry.get('token_strip')!;
  const wRec = b.sheetRegistry.get('w_embed')!;
  const x0Rec = b.sheetRegistry.get('x_embed')!;

  const focusTokSpan = spanFromSheetGrid(tokRec, focusTok, focusTok, 0, 0);
  const focusWRowSpan = spanFromSheetGrid(wRec, focusVocabId, focusVocabId, 0, d - 1);
  const focusX0ColSpan = spanFromSheetGrid(x0Rec, 0, d - 1, focusTok, focusTok);

  // When no specific single-cell trace overrides it, highlight the active token lookup row in teal/indigo
  if (!hasCellTrace) {
    drawCellHighlight3D(b.flow, focusTokSpan, CELL_OP_A_FILL, CELL_OP_A_FRAME, 0.024, true);
    drawCellHighlight3D(b.flow, focusWRowSpan, CELL_OP_B_FILL, CELL_OP_B_FRAME, 0.024, true);
    drawCellHighlight3D(b.flow, focusX0ColSpan, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.022, true);
  }

  // Exact world endpoints for focus token lookup: Token Strip[focusTok] -> W_embed[focusVocabId, :]
  const tokLz = -0.5 + (focusTok + 0.5) / T;
  const tokExitPt = localToWorldOnFrame(tokenStripFrame, 0.52, 0.35, tokLz);
  const wRowLz = -0.5 + (focusVocabId + 0.5) / vocabSize;
  const wEntryPt = localToWorldOnFrame(wEmbedFrame, 0.0, -0.65, wRowLz);
  const wExitPt = localToWorldOnFrame(wEmbedFrame, 0.0, 0.65, wRowLz);
  const x0ColLx = -0.5 + (focusTok + 0.5) / T;
  const x0EntryTop = localToWorldOnFrame(x0Frame, x0ColLx, 0.55, -0.30);
  const x0EntryBot = localToWorldOnFrame(x0Frame, x0ColLx, 0.55, 0.30);

  const lookupCol = hasCellTrace
    ? RIBBON_DIMMED
    : activePart === 'token-strip' || activePart === 'w-embed'
      ? RIBBON_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_PATHWAY_SOFT;

  // Token Strip -> W_embed looked-up row
  b.flow.addCubic3D(
    tokExitPt,
    [xTokenStrip + 1.45, BASE_Y + 0.35, tokExitPt[2] * 0.65],
    [xWEmbed - 1.35, wEntryPt[1], wEntryPt[2] * 0.5],
    wEntryPt,
    [0, 1, 0],
    hasCellTrace ? 0.052 : 0.088,
    lookupCol,
    true,
    22,
    true,
    0.20,
    0.20,
  );

  // Subtle secondary ribbons from non-focus tokens when T > 1
  if (T > 1) {
    const stepTok = Math.max(1, Math.floor(T / 4));
    for (let t = 0; t < T; t += stepTok) {
      if (t === focusTok) continue;
      const vid = Math.max(0, Math.min(vocabSize - 1, Math.trunc(tokenIds[t] ?? 0)));
      const pStart = localToWorldOnFrame(tokenStripFrame, 0.52, 0.20, -0.5 + (t + 0.5) / T);
      const pEnd = localToWorldOnFrame(wEmbedFrame, 0.0, -0.65, -0.5 + (vid + 0.5) / vocabSize);
      b.flow.addCubic3D(
        pStart,
        [xTokenStrip + 1.2, BASE_Y + 0.20, pStart[2] * 0.7],
        [xWEmbed - 1.1, pEnd[1], pEnd[2] * 0.5],
        pEnd,
        [0, 1, 0],
        0.040,
        hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
        false,
        14,
        false,
      );
    }
  }

  // Curved emerald copy ribbons: W_embed[focusVocabId, :] -> X₀[:, focusTok]
  const copyCol = hasCellTrace
    ? RIBBON_DIMMED
    : activePart === 'w-embed' || activePart === 'x-embed'
      ? RIBBON_RESIDUAL_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL_ACTIVE;

  const wExitA = localToWorldOnFrame(wEmbedFrame, -0.28, 0.65, wRowLz);
  const wExitB = localToWorldOnFrame(wEmbedFrame, 0.28, 0.65, wRowLz);

  b.flow.addCubic3D(
    wExitA,
    [xWEmbed + 1.55, Math.max(wExitPt[1], 0.85), wExitA[2] * 0.7],
    [x0EntryTop[0] - 1.25, BASE_Y + 0.78, x0EntryTop[2]],
    x0EntryTop,
    [0, 1, 0],
    hasCellTrace ? 0.055 : 0.095,
    copyCol,
    true,
    24,
    true,
    0.20,
    0.20,
  );
  b.flow.addCubic3D(
    wExitB,
    [xWEmbed + 1.55, Math.max(wExitPt[1] - 0.06, 0.72), wExitB[2] * 0.7],
    [x0EntryBot[0] - 1.25, BASE_Y + 0.68, x0EntryBot[2]],
    x0EntryBot,
    [0, 1, 0],
    hasCellTrace ? 0.055 : 0.095,
    copyCol,
    true,
    24,
    true,
    0.20,
    0.20,
  );

  // Emerald Residual Highway hand-off ribbon: X₀ -> Bay 01 Mamba-2
  const handoffCol = hasCellTrace
    ? x0Hl !== SHEET_HL_DIMMED
      ? RIBBON_RESIDUAL
      : RIBBON_DIMMED
    : activePart === 'residual-handoff' || activePart === 'x-embed'
      ? RIBBON_RESIDUAL_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL_ACTIVE;

  b.flow.addPolyline3D(
    [
      [xX0 + x0WidthX * 0.5 + 0.06, BASE_Y + 0.04, 0],
      [xHandoff - handoffWidthX * 0.5 - 0.06, BASE_Y + 0.04, 0],
    ],
    [0, 1, 0],
    0.15,
    handoffCol,
    true,
    0.26,
    0.28,
    true,
  );

  // Outgoing arrow stub from Bay 01 hand-off pad toward Bay 01 Mamba-2
  b.flow.addPolyline3D(
    [
      [xHandoff + handoffWidthX * 0.5 + 0.04, BASE_Y + 0.04, 0],
      [Math.min(7.05, xHandoff + handoffWidthX * 0.5 + 0.62), BASE_Y + 0.04, 0],
    ],
    [0, 1, 0],
    0.13,
    handoffCol,
    true,
    0.24,
    0.26,
    true,
  );

  // -------------------------------------------------------------------------
  // 7. Live 3D Cell Highlights & Cross-Plane Dependency Threads
  // -------------------------------------------------------------------------
  const resolveSpan = (span: EmbeddingSheetSpan): WorldSpan3D | null => {
    const rec = b.sheetRegistry.get(span.tensor);
    if (!rec) return null;
    return spanFromSheetGrid(rec, span.rowStart, span.rowEnd, span.colStart, span.colEnd);
  };

  let cellPopover: CellPopoverSpec | null = null;
  let activeOpPartIds: ReadonlySet<string> | undefined;
  if (activeFormula) {
    const mapTensorPart = (t: EmbeddingTensorKey): string =>
      t === 'token_strip' ? 'token-strip' : t === 'w_embed' ? 'w-embed' : 'x-embed';
    const partIds = new Set<string>([mapTensorPart(activeFormula.targetSpan.tensor)]);
    const targetSpan = resolveSpan(activeFormula.targetSpan);
    if (targetSpan) {
      drawCellHighlight3D(b.flow, targetSpan, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.025, true);
      const sourceWorlds: {
        readonly world: readonly [number, number, number];
        readonly role: 'indigo' | 'teal';
      }[] = [];
      let hasOpA = false;
      let hasOpB = false;
      for (const up of activeFormula.upstreamSpans) {
        partIds.add(mapTensorPart(up.tensor));
        const upSpan = resolveSpan(up);
        if (!upSpan) continue;
        const isA = up.role === 'operand_a';
        if (isA) hasOpA = true;
        else hasOpB = true;
        sourceWorlds.push({
          world: upSpan.frontPoint,
          role: isA ? 'indigo' : 'teal',
        });
        drawOperandDependency3D(b.flow, upSpan, targetSpan, isA);
      }
      const operandColors: ('indigo' | 'teal' | 'red')[] = [];
      if (hasOpA) operandColors.push('indigo');
      if (hasOpB) operandColors.push('teal');
      operandColors.push('red');
      const c = activeFormula.cell;
      const popTensorLabel =
        c.tensor === 'w_embed'
          ? 'Embedding W_embed'
          : c.tensor === 'x_embed'
            ? 'Embedding X₀'
            : 'Embedding Tokens';
      const popCoordLabel =
        c.tensor === 'w_embed'
          ? `[v=${c.row} "${model.vocab[c.row] ?? ''}", ch=${c.col}]`
          : c.tensor === 'x_embed'
            ? `[ch=${c.row}, t${offset + c.col}]`
            : `t${offset + c.row} "${tokenLabels[c.row] ?? ''}"`;
      const popBadge =
        c.tensor === 'w_embed' ? 'Untied' : c.tensor === 'x_embed' ? 'NoPE Copy' : 'NoPE';
      const popEq =
        c.tensor === 'w_embed'
          ? `W_embed[${c.row} ("${model.vocab[c.row] ?? ''}"), ch${c.col}]`
          : c.tensor === 'x_embed'
            ? `X₀[ch${c.row}, t${offset + c.col}] ← W_embed[#${c.vocabId ?? tokenIds[c.col] ?? 0}, ch${c.row}]`
            : `Lookup row #${c.vocabId ?? tokenIds[c.row] ?? 0} in W_embed`;
      cellPopover = {
        world: targetSpan.frontPoint,
        tensorLabel: popTensorLabel,
        coordLabel: popCoordLabel,
        badge: popBadge,
        equation: popEq,
        valueText:
          c.tensor === 'token_strip'
            ? activeFormula.valueText
            : `= ${activeFormula.valueText}`,
        operandColors,
        sourceWorlds: sourceWorlds.length > 0 ? sourceWorlds : undefined,
      };
    }
    if (activeFormula.targetSpan.tensor === 'x_embed') {
      partIds.add('residual-handoff');
    }
    activeOpPartIds = partIds;
  }

  // V3 Zoom-adaptive on-cell value grids for X₀ [d × T] and W_embed [V × d]
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const x0TokLabels = Array.from(
    { length: T },
    (_, t) => `t${offset + t} ${tokenLabels[t] ?? model.vocab[tokenIds[t] ?? 0] ?? ''}`.trim(),
  );
  const chanLabels = Array.from({ length: d }, (_, ch) => `ch${ch}`);
  const vocabLabels = Array.from(
    { length: vocabSize },
    (_, v) => `#${v} ${model.vocab[v] ?? ''}`.trim(),
  );

  const x0Sheet = b.sheetRegistry.get('x_embed');
  if (x0Sheet) {
    const frame =
      x0Sheet.frame ??
      makeSheetFrame3D([x0Sheet.cx, x0Sheet.cy, x0Sheet.cz], [x0Sheet.ex, x0Sheet.ey, x0Sheet.ez]);
    const x0Values = new Float32Array(d * T);
    for (let ch = 0; ch < d; ch++) {
      for (let t = 0; t < T; t++) {
        x0Values[ch * T + t] = embedding.X[ch * T + t];
      }
    }
    activeCellGrids.push({
      originWorld: frame.center,
      extent: frame.extent,
      axisU: frame.axisU,
      axisN: frame.axisN,
      cols: T,
      rows: d,
      values: x0Values,
      format: 'signed',
      activeRow: activeFormula?.cell.tensor === 'x_embed' ? activeFormula.cell.row : undefined,
      activeCol: activeFormula?.cell.tensor === 'x_embed' ? activeFormula.cell.col : focusTok,
      colLabels: x0TokLabels,
      rowLabels: chanLabels,
      colDimLabel: T === 1 ? `S = 1 (t${absFocusT})` : `T = ${T}`,
      rowDimLabel: `d = ${d}`,
    });
  }

  const wSheet = b.sheetRegistry.get('w_embed');
  if (wSheet) {
    const frame =
      wSheet.frame ??
      makeSheetFrame3D([wSheet.cx, wSheet.cy, wSheet.cz], [wSheet.ex, wSheet.ey, wSheet.ez]);
    const wValues = new Float32Array(vocabSize * d);
    for (let v = 0; v < vocabSize; v++) {
      for (let ch = 0; ch < d; ch++) {
        wValues[v * d + ch] = model.W_embed[v * d + ch];
      }
    }
    activeCellGrids.push({
      originWorld: frame.center,
      extent: frame.extent,
      axisU: frame.axisU,
      axisN: frame.axisN,
      cols: d,
      rows: vocabSize,
      values: wValues,
      format: 'signed',
      activeRow: activeFormula?.cell.tensor === 'w_embed' ? activeFormula.cell.row : focusVocabId,
      activeCol: activeFormula?.cell.tensor === 'w_embed' ? activeFormula.cell.col : undefined,
      colLabels: chanLabels,
      rowLabels: vocabLabels,
      colDimLabel: `d = ${d}`,
      rowDimLabel: `V = ${vocabSize}`,
    });
  }

  // -------------------------------------------------------------------------
  // 8. 3D Exterior Landmarks & Calipers
  // -------------------------------------------------------------------------
  b.landmarks.push(
    {
      id: 'emb-tokens',
      label: T === 1 ? `Input Token (t${absFocusT} "${focusWord}")` : `Input Token Strip (t${offset}..t${offset + T - 1})`,
      sublabel: `Focus: t${absFocusT} "${focusWord}" · id #${focusVocabId}`,
      world: [xTokenStrip, 0.20, tokenStripD * 0.5 + 0.28],
      partId: 'token-strip',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'emb-w',
      label: `Untied W_embed [${vocabSize}×${d}]`,
      sublabel: '131,072 × 2,688 prod · tie_word_embeddings: false',
      world: [xWEmbed, wEmbedY + wEmbedHeightY * 0.5 + 0.16, 0],
      partId: 'w-embed',
      align: 'top',
      tier: 'primary',
      tone: isFp8 ? 'gold' : 'default',
    },
    {
      id: 'emb-nope',
      label: 'No Pos Embed (NoPE) · Order comes from Bay 01 Mamba-2',
      sublabel: 'Zero W_pos · Zero RoPE',
      world: [xNopeSlot, BASE_Y + 0.18, nopeZ - nopeD * 0.5 - 0.14],
      partId: 'nope-slot',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'emb-x0',
      label: `X₀ [${d}×${T}] · Direct Row Copy`,
      sublabel: `col t${absFocusT} ← W_embed[${focusVocabId}, :]`,
      world: [xX0, 0.20, x0DepthZ * 0.5 + 0.28],
      partId: 'x-embed',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'emb-handoff',
      label: 'Residual Highway → Bay 01 Mamba-2',
      sublabel: 'X₀ === Bay 01 Input X (bit-exact)',
      world: [xHandoff, 0.22, -handoffDepthZ * 0.5 - 0.26],
      partId: 'residual-handoff',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'emb-row-pill',
      label: `row #${focusVocabId} "${focusWord}"`,
      world: [xWEmbed, wEntryPt[1], wEmbedSpanZ * 0.5 + 0.26],
      partId: 'w-embed',
      align: 'right',
      tier: 'detail',
      tone: 'cyan',
    },
  );

  b.calipers.push(
    {
      id: 'cal-emb-vocab',
      aWorld: [xWEmbed, wEmbedY - wEmbedHeightY * 0.5, -wEmbedSpanZ * 0.5 - 0.14],
      bWorld: [xWEmbed, wEmbedY + wEmbedHeightY * 0.5, -wEmbedSpanZ * 0.5 - 0.14],
      label: `V=${vocabSize} mini (131,072 prod)`,
      partIds: ['w-embed'],
      minPixels: 60,
      offsetPx: 10,
    },
    {
      id: 'cal-emb-x0-t',
      aWorld: [xX0 - x0WidthX * 0.5, BASE_Y + 0.04, x0DepthZ * 0.5 + 0.10],
      bWorld: [xX0 + x0WidthX * 0.5, BASE_Y + 0.04, x0DepthZ * 0.5 + 0.10],
      label: T === 1 ? `S = 1 (AR pos t${absFocusT})` : `T = ${T} tokens`,
      partIds: ['x-embed', 'token-strip'],
      minPixels: 48,
      offsetPx: 8,
    },
  );

  const reg = (k: string) => b.sheetRegistry.get(k);
  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('token_strip')], 0.55, 0.35, 0.55, EMBEDDING_SPATIAL_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('token_strip'), reg('w_embed')], 0.50, 0.25, 0.48, EMBEDDING_SPATIAL_FIT_BOUNDS),
    2: computeSheetGroupBounds([reg('w_embed'), reg('nope_slot'), reg('x_embed')], 0.48, 0.25, 0.45, EMBEDDING_SPATIAL_FIT_BOUNDS),
    3: computeSheetGroupBounds([reg('x_embed'), reg('residual_handoff')], 0.55, 0.35, 0.55, EMBEDDING_SPATIAL_FIT_BOUNDS),
  };

  const packed = b.pack();
  return {
    kind: 'cutaway_embedding',
    bayId: 0,
    ...packed,
    fitBounds: EMBEDDING_SPATIAL_FIT_BOUNDS,
    opFocusBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    calipers: b.calipers,
    valueAtlas: atlas,
    activeFormula,
    embeddingFormula: activeFormula,
    activeOpPartIds,
    cellPopover,
    activeCellGrids,
    solidPickBoxes: b.pickBoxes,
    partCatalog: buildEmbeddingPartCatalog(
      T,
      d,
      vocabSize,
      absFocusT,
      focusVocabId,
      focusWord,
      precision,
    ),
  };
}
