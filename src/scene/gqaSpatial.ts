import type {
  GqaCellFormula,
  GqaCellRef,
  GqaSheetSpan,
  GqaSimResult,
} from '../math/gqaSim';
import type { Vec3 } from '../model/spec';
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
import type { PickBox } from '../render/pick';
import { FlowRibbonBuilder, type FlowRibbonColor } from '../render/ribbons';
import {
  FRAME_XY_AXES,
  FRAME_XZ_AXES,
  FRAME_YZ_AXES,
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
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_GHOST_RAIL,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../render/sheets';
import type { AtlasSlice } from './atlasPacker';
import type {
  CutawayPartInfo,
  CutawaySceneGeometry,
} from './cutaways';
import type { GqaValueAtlas } from './gqaAtlas';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  computeSheetGroupBounds,
  createSheetFocusSet,
  drawCellHighlight3D,
  drawOperandDependency3D,
  EMPTY_SHEET_FOCUS_SET,
  resolveSheetFocusRole,
  resolveSheetHighlight,
  RIBBON_ACTIVE,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  RIBBON_RESIDUAL_ACTIVE,
  spanFromSheetGrid,
  type SheetFocusEntry,
  type SheetFocusSet,
  type SheetGeomRecord,
  type WorldSpan3D,
} from './sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_WEIGHT_BLUE,
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from './spine';

/**
 * Invariant 3D bounding envelope for Spatial GQA across T=1..16, BF16/FP8,
 * and all 32 possible selected-head lift states (`Q00..Q31`).
 */
export const GQA_SPATIAL_FIT_BOUNDS: AABB = {
  min: [-8.40, -0.06, -3.78],
  max: [8.90, 3.24, 3.82],
};

export interface SpatialGqaFitOptions {
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly isNarrow?: boolean;
  readonly azimuthDeg?: number;
  readonly elevationDeg?: number;
}

/**
 * Exact 8-corner projective camera fitter for Spatial GQA.
 * Computes the tightest perspective distance that keeps every corner of `bounds`
 * strictly inside the unobstructed viewport band `[0, innerWidth] x [topInset, innerHeight - bottomInset]`,
 * while shifting the target along camera-up so the 3D composition sits dead-center in that band.
 */
export function fitSpatialGqaCamera(
  bounds: AABB,
  metrics: ViewportMetrics,
  options?: SpatialGqaFitOptions,
): CameraPose {
  const isNarrow = options?.isNarrow ?? metrics.innerWidth < 760;
  const topInset = Math.max(0, options?.topInsetPx ?? (isNarrow ? 12 : 84));
  const bottomInset = Math.max(0, options?.bottomInsetPx ?? (isNarrow ? 12 : 118));
  const azimuthDeg = options?.azimuthDeg ?? (isNarrow ? 17 : 20);
  const elevationDeg = options?.elevationDeg ?? (isNarrow ? 25 : 23.5);
  const fovDeg = DEFAULT_FOV_DEG;
  const fillX = isNarrow ? 0.94 : 0.96;
  const fillY = isNarrow ? 0.90 : 0.93;

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

  // Target NDC vertical interval inside [clampedTop, safeH - clampedBottom]
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

/** Horizontal XZ sheet rotated 90° around +Y: cols run along +Z, rows run along -X, normal +Y. */
const FRAME_XZ_ROT90_AXES: SheetBasisAxes = {
  axisU: [0, 0, 1],
  axisN: [0, 1, 0],
  axisV: [-1, 0, 0],
};

// Semantic K / V flow colors so K and V pathways stay unmistakably distinguishable
const RIBBON_K_FLOW: FlowRibbonColor = { r: 0.14, g: 0.54, b: 0.49, a: 0.42, borderAlpha: 0.24 };
const RIBBON_V_FLOW: FlowRibbonColor = { r: 0.18, g: 0.46, b: 0.66, a: 0.42, borderAlpha: 0.24 };
const RIBBON_K_ACTIVE: FlowRibbonColor = { r: 0.12, g: 0.54, b: 0.48, a: 0.78, borderAlpha: 0.40 };
const RIBBON_V_ACTIVE: FlowRibbonColor = { r: 0.16, g: 0.46, b: 0.68, a: 0.78, borderAlpha: 0.40 };

interface SpatialSheetSpec {
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
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly tensorKey?: string;
  readonly tokenIndex?: number;
  readonly atlasSlice?: AtlasSlice;
  readonly registryKey?: string;
  readonly cellFamily: number;
}

class SpatialGqaBuilder {
  readonly sheetFloats: number[] = [];
  readonly basisFloats: number[] = [];
  readonly solidFloats: number[] = [];
  readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly kvDepthTicks: KvDepthTickAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();
  private readonly bayId: number;

  constructor(bayId: number) {
    this.bayId = bayId;
  }

  addSheet(s: SpatialSheetSpec): SheetFrame3D {
    const frame = makeSheetFrame3D(
      [s.cx, s.cy, s.cz],
      [s.ex, s.ey, s.ez],
      s.axes.axisU,
      s.axes.axisN,
    );

    const rect = s.atlasSlice ? s.atlasSlice.rect : ([0, 0, 0, 0] as const);
    const scale = s.atlasSlice ? s.atlasSlice.scale : 1.0;
    const modeFlag = s.atlasSlice ? s.atlasSlice.mode : 0;

    // 1. SheetsPass instance (20 floats — identical layout to XZ sheets)
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

    // 2. Parallel 3D basis buffer (6 floats: axisU, axisN; axisV = U x N in shader)
    this.basisFloats.push(
      frame.axisU[0], frame.axisU[1], frame.axisU[2],
      frame.axisN[0], frame.axisN[1], frame.axisN[2],
    );

    // 3. 1-to-1 mirror in solidData (12 floats)
    this.solidFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.cellFamily,
      s.fp8Flag,
      s.seed,
      this.bayId,
      0.2,
      s.highlight === SHEET_HL_ACTIVE ? 1 : s.highlight === SHEET_HL_BF16_GOLD ? 2 : 0,
    );

    // 4. Exact oriented PickBox (slightly padded thickness along axisN for effortless picking)
    const pickFrame = makeSheetFrame3D(
      frame.center,
      [frame.extent[0], Math.max(frame.extent[1], 0.085), frame.extent[2]],
      frame.axisU,
      frame.axisN,
    );
    const worldAabb = sheetFrameWorldAabb(pickFrame);
    this.pickBoxes.push({
      bayId: this.bayId,
      partId: s.partId,
      headIndex: s.headIndex,
      kvHeadIndex: s.kvHeadIndex,
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
    alpha = 0.6,
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

function kvHeadForQuery(qIdx: number): 0 | 1 {
  return qIdx < 16 ? 0 : 1;
}

function fmtHeadTag(qIdx: number): string {
  return `Q${String(qIdx).padStart(2, '0')}`;
}

function resolveSpatialSpan(
  registry: ReadonlyMap<string, SheetGeomRecord>,
  span: GqaSheetSpan,
  displayedHeads: readonly [number, number],
): WorldSpan3D | null {
  // Multi-token span on shared K or V track (e.g., V_g slices 0..q)
  if (span.tensor === 'kv_k' || span.tensor === 'kv_v') {
    const g = span.kvHeadIndex ?? 0;
    const t0 = span.tokenStart ?? 0;
    const t1 = span.tokenEnd ?? t0;
    const first = registry.get(`${span.tensor}:${g}:${t0}`);
    const last = registry.get(`${span.tensor}:${g}:${t1}`) ?? first;
    if (!first || !last) return null;

    if (t0 === t1) {
      return spanFromSheetGrid(first, 0, 0, span.colStart, span.colEnd);
    }

    // Build synthetic XZ composite span covering token slices t0..t1 along +Z
    const zMin = Math.min(first.cz - first.ez * 0.5, last.cz - last.ez * 0.5);
    const zMax = Math.max(first.cz + first.ez * 0.5, last.cz + last.ez * 0.5);
    const composite: SheetGeomRecord = {
      cx: first.cx,
      cy: first.cy,
      cz: (zMin + zMax) * 0.5,
      ex: first.ex,
      ey: first.ey,
      ez: Math.max(0.05, zMax - zMin),
      gridCols: first.gridCols,
      gridRows: 1,
      frame: makeSheetFrame3D(
        [first.cx, first.cy, (zMin + zMax) * 0.5],
        [first.ex, first.ey, Math.max(0.05, zMax - zMin)],
        FRAME_XZ_AXES.axisU,
        FRAME_XZ_AXES.axisN,
      ),
    };
    return spanFromSheetGrid(composite, 0, 0, span.colStart, span.colEnd);
  }

  let key: string = span.tensor;
  if (span.tensor === 'q') {
    key = `q:${span.headIndex ?? 0}`;
  } else if (span.tensor === 'scores' || span.tensor === 'softmax') {
    const g = span.kvHeadIndex ?? (span.headIndex !== undefined ? kvHeadForQuery(span.headIndex) : 0);
    if (span.headIndex !== undefined && displayedHeads[g] !== span.headIndex) {
      return null;
    }
    key = `${span.tensor}:${g}`;
  } else if (span.tensor === 'v_out') {
    const g = span.kvHeadIndex ?? 0;
    key = `v_out:${g}`;
  }
  const sheet = registry.get(key);
  if (!sheet) return null;
  return spanFromSheetGrid(sheet, span.rowStart, span.rowEnd, span.colStart, span.colEnd);
}

/** Build exact full-identity SheetFocusSet from a live GQA cell formula (shared by Spatial 3/4 and Near-Plan). */
export function buildGqaFocusSet(formula: GqaCellFormula | null): SheetFocusSet {
  if (!formula) return EMPTY_SHEET_FOCUS_SET;
  const toEntry = (span: GqaSheetSpan): SheetFocusEntry => ({
    tensor: span.tensor,
    headIndex: span.headIndex,
    kvHeadIndex: span.kvHeadIndex,
    tokenStart: span.tokenStart,
    tokenEnd: span.tokenEnd,
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

export interface BuildGqaSpatialContext {
  readonly bayId: number;
  readonly clampedT: number;
  readonly isFp8Mode: boolean;
  readonly computeFp8: boolean;
  readonly kvCacheFp8: boolean;
  readonly activePart: string | null;
  readonly activeHeadIdx: number | null;
  readonly activeKvGroup: 0 | 1 | null;
  readonly group0Head: number;
  readonly group1Head: number;
  readonly activeCell: GqaCellRef | null;
  readonly activeFormula: GqaCellFormula | null;
  readonly sim: GqaSimResult;
  readonly atlas: GqaValueAtlas;
  readonly partCatalog: Readonly<Record<string, CutawayPartInfo>>;
}

/**
 * Spatial 3/4 GQA Reference Implementation (Increment 1).
 *
 * Meaningful 3D axis semantics:
 *  - X (left -> right): Forward GQA computation (Input -> RMSNorm -> W_Q/W_K/W_V -> Q/K -> Scores -> Softmax -> V -> V_out -> Concat -> W_O -> +Residual)
 *  - Y (elevation): Separates horizontal activation decks (y=0.04..0.32), upright learned weight walls
 *    (W_Q, W_K, W_V, W_O), lifted active query head (Q_h), and the elevated emerald residual bypass bridge (y=3.18)
 *  - Z (depth): Separates Group 0 (Q00-Q15 + K0/V0, rear z<0) from Group 1 (Q16-Q31 + K1/V1, front z>0),
 *    and fans the 16 query heads per group into parallel 3D architectural cards sharing 1 K rack and 1 V rack.
 */
export function buildGqaSpatialScene(ctx: BuildGqaSpatialContext): CutawaySceneGeometry {
  const {
    bayId,
    clampedT,
    isFp8Mode,
    computeFp8,
    kvCacheFp8,
    activePart,
    activeHeadIdx,
    activeKvGroup,
    group0Head,
    group1Head,
    activeFormula,
    sim,
    atlas,
    partCatalog,
  } = ctx;

  const b = new SpatialGqaBuilder(bayId);

  const focusSet = buildGqaFocusSet(activeFormula);
  const hasCellTrace = focusSet.active;
  const hasFocus = activePart !== null || hasCellTrace;

  const SHEET_THICK = 0.056;
  const BASE_Y = 0.04;
  const DECK_Y = 0.32;

  const stageHighlight = (tensor: string, partId: string, onPath: boolean): number => {
    const fallback =
      activePart === partId
        ? SHEET_HL_ACTIVE
        : onPath && hasFocus
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    return resolveSheetHighlight(focusSet, { tensor }, fallback);
  };

  // X stage coordinates (left -> right)
  const xInput = -7.85;
  const xRms = -6.35;
  const xWeightsIn = -4.75;
  const xQBase = -2.95;
  const xKTrack = -1.05;
  const xScores = 0.65;
  const xSoftmax = 2.45;
  const xVTrack = 3.95;
  const xVOut = 5.20;
  const xConcat = 6.35;
  const xWO = 7.30;
  const xResOut = 8.35;

  // Group Z centers
  const zGrp: readonly [number, number] = [-2.15, 2.15];
  const groupSpanZ = 2.70;
  const qPitchZ = groupSpanZ / 15; // 0.18 spacing (> 3x card thickness)
  const qStepX = 0.04;

  const qHeadPos = (qIdx: number): { x: number; z: number } => {
    const g = kvHeadForQuery(qIdx);
    const localIdx = qIdx % 16;
    return {
      x: xQBase + localIdx * qStepX,
      z: zGrp[g] - groupSpanZ * 0.5 + localIdx * qPitchZ,
    };
  };

  // Shared K / V token rack geometry along Z (1..16 slots)
  const kvPitchZ = groupSpanZ / 16;
  const kvSliceDepth = 0.125;
  const kvSliceWidth = 0.68;
  const kvSliceZ = (g: 0 | 1, tok1Based: number): number =>
    zGrp[g] - groupSpanZ * 0.5 + (tok1Based - 0.5) * kvPitchZ;

  // Square causal T x T size (grows monotonically T=1 -> T=16)
  const scoreSize = 0.88 + (clampedT / 16) * 0.74;

  // ==========================================================================
  // 1. STAGE 1: Input X, Pre-Attention RMSNorm & Overhead Emerald Residual Bridge
  // ==========================================================================
  const ioW = 1.05;
  const ioD = 1.70;

  const resInHl = resolveSheetHighlight(
    focusSet,
    { tensor: 'input' },
    activePart === 'residual-in'
      ? SHEET_HL_ACTIVE
      : activePart === 'residual-out'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: xInput,
    cy: BASE_Y,
    cz: 0,
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioD,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resInHl,
    seed: 1.1 + clampedT * 0.1,
    partId: 'residual-in',
    tensorKey: 'input',
    atlasSlice: atlas.input,
    registryKey: 'input',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  const resOutHl = resolveSheetHighlight(
    focusSet,
    { tensor: 'residual' },
    activePart === 'residual-out'
      ? SHEET_HL_ACTIVE
      : activeHeadIdx !== null
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: xResOut,
    cy: BASE_Y,
    cz: 0,
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioD,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resOutHl,
    seed: 2.3 + clampedT * 0.1,
    partId: 'residual-out',
    tensorKey: 'residual',
    atlasSlice: atlas.residual,
    registryKey: 'residual',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Elevated Emerald Residual Bypass Bridge (arches cleanly over the rear of the attention assembly)
  const bridgeY = 3.16;
  const bridgeZ = -3.68;
  const resCol = hasCellTrace
    ? resInHl !== SHEET_HL_DIMMED || resOutHl !== SHEET_HL_DIMMED
      ? RIBBON_RESIDUAL
      : RIBBON_DIMMED
    : activePart === 'residual-in' || activePart === 'residual-out'
      ? RIBBON_RESIDUAL_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL;
  b.flow.addCubic3D(
    [xInput, BASE_Y + 0.04, -ioD * 0.5],
    [xInput, bridgeY * 0.85, -ioD * 0.5 - 0.8],
    [xInput + 0.3, bridgeY, bridgeZ],
    [xInput + 0.9, bridgeY, bridgeZ],
    [0, 1, 0],
    0.15,
    resCol,
    false,
    14,
    true,
  );
  b.flow.addPolyline3D(
    [
      [xInput + 0.85, bridgeY, bridgeZ],
      [xResOut - 0.85, bridgeY, bridgeZ],
    ],
    [0, 1, 0],
    0.15,
    resCol,
    false,
    0.22,
    0.22,
    true,
  );
  b.flow.addCubic3D(
    [xResOut - 0.9, bridgeY, bridgeZ],
    [xResOut - 0.3, bridgeY, bridgeZ],
    [xResOut, bridgeY * 0.85, -ioD * 0.5 - 0.8],
    [xResOut, BASE_Y + 0.08, -ioD * 0.5 - 0.02],
    [0, 1, 0],
    0.15,
    resCol,
    true,
    14,
    true,
    0.30,
    0.32,
  );

  // Subtle architectural stanchion lines grounding the residual bridge endpoints
  b.addGuideLine(xInput, BASE_Y, -ioD * 0.5, xInput, bridgeY, bridgeZ, 0.8, LINE_STYLE_GUIDE, 0.25);
  b.addGuideLine(xResOut, BASE_Y, -ioD * 0.5, xResOut, bridgeY, bridgeZ, 0.8, LINE_STYLE_GUIDE, 0.25);

  const rmsHl = stageHighlight(
    'rmsnorm',
    'rmsnorm',
    activeHeadIdx !== null || activePart === 'w-q' || activePart === 'k-proj' || activePart === 'v-proj',
  );
  b.addSheet({
    cx: xRms,
    cy: BASE_Y,
    cz: 0,
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioD,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: rmsHl,
    seed: 3.7,
    partId: 'rmsnorm',
    tensorKey: 'rmsnorm',
    atlasSlice: atlas.rmsnorm,
    registryKey: 'rmsnorm',
    cellFamily: CELL_MOE_AMBER,
  });

  // Input -> RMSNorm connector
  b.flow.addPolyline3D(
    [
      [xInput + ioW * 0.5 + 0.02, BASE_Y + 0.04, 0],
      [xRms - ioW * 0.5 - 0.04, BASE_Y + 0.04, 0],
    ],
    [0, 1, 0],
    0.11,
    !hasCellTrace && (rmsHl === SHEET_HL_ACTIVE || activePart === 'residual-in')
      ? RIBBON_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL,
    true,
    0.18,
    0.18,
    false,
  );

  // ==========================================================================
  // 2. STAGE 2: Upright Learned Projection Weight Walls (W_Q, W_K, W_V in YZ plane)
  // ==========================================================================
  const wqY = 2.02;
  const wqSpanZ = 5.40; // spans across both Group 0 & Group 1 -> 16:1 visual contrast vs W_K/W_V
  const wqHeightY = 0.96;
  const wqOnPath = activeHeadIdx !== null || activePart === 'w-q';
  const wqHl = stageHighlight('w_q', 'w-q', wqOnPath);

  b.addSheet({
    cx: xWeightsIn,
    cy: wqY,
    cz: 0,
    ex: wqSpanZ,
    ey: SHEET_THICK,
    ez: wqHeightY,
    axes: FRAME_YZ_AXES,
    gridCols: sim.qConcatDim,
    gridRows: sim.d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wqHl,
    seed: 5.1 + bayId * 0.3,
    partId: 'w-q',
    tensorKey: 'w_q',
    atlasSlice: atlas.wQ,
    registryKey: 'w_q',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  const wkvY = 0.62;
  const wkvSpanZ = 0.92;
  const wkvHeightY = 0.86;
  const wkZ = -1.65;
  const wvZ = 1.65;

  const wkOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'k-proj';
  const wkHl = stageHighlight('w_k', 'k-proj', wkOnPath);
  b.addSheet({
    cx: xWeightsIn,
    cy: wkvY,
    cz: wkZ,
    ex: wkvSpanZ,
    ey: SHEET_THICK,
    ez: wkvHeightY,
    axes: FRAME_YZ_AXES,
    gridCols: sim.nKv * sim.dHead,
    gridRows: sim.d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wkHl,
    seed: 6.4 + bayId * 0.2,
    partId: 'k-proj',
    tensorKey: 'w_k',
    atlasSlice: atlas.wK,
    registryKey: 'w_k',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  const wvOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'v-proj';
  const wvHl = stageHighlight('w_v', 'v-proj', wvOnPath);
  b.addSheet({
    cx: xWeightsIn,
    cy: wkvY,
    cz: wvZ,
    ex: wkvSpanZ,
    ey: SHEET_THICK,
    ez: wkvHeightY,
    axes: FRAME_YZ_AXES,
    gridCols: sim.nKv * sim.dHead,
    gridRows: sim.d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wvHl,
    seed: 7.9 + bayId * 0.2,
    partId: 'v-proj',
    tensorKey: 'w_v',
    atlasSlice: atlas.wV,
    registryKey: 'w_v',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Smooth 3D ribbons: RMSNorm -> upright W_Q, W_K, W_V
  const projFanCol = (onPath: boolean, baseCol: FlowRibbonColor = RIBBON_QUIET): FlowRibbonColor =>
    onPath && hasFocus && !hasCellTrace ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : baseCol;

  // RMSNorm -> W_Q (climbs cleanly to upper W_Q wall)
  b.flow.addCubic3D(
    [xRms + ioW * 0.5 + 0.02, BASE_Y + 0.05, 0],
    [xRms + ioW * 0.5 + 0.55, wqY * 0.65, 0],
    [xWeightsIn - 0.55, wqY, 0],
    [xWeightsIn - 0.06, wqY, 0],
    [0, 1, 0],
    wqOnPath && hasFocus && !hasCellTrace ? 0.105 : 0.08,
    projFanCol(wqOnPath || activePart === 'rmsnorm'),
    true,
    18,
    true,
  );
  // RMSNorm -> W_K (teal-tinted key branch)
  b.flow.addCubic3D(
    [xRms + ioW * 0.5 + 0.02, BASE_Y + 0.05, -0.45],
    [xRms + ioW * 0.5 + 0.50, wkvY * 0.7, wkZ * 0.55],
    [xWeightsIn - 0.55, wkvY, wkZ],
    [xWeightsIn - 0.06, wkvY, wkZ],
    [0, 1, 0],
    wkOnPath && hasFocus && !hasCellTrace ? 0.095 : 0.075,
    projFanCol(wkOnPath || activePart === 'rmsnorm', RIBBON_K_FLOW),
    true,
    18,
    true,
  );
  // RMSNorm -> W_V (steel-cyan value branch)
  b.flow.addCubic3D(
    [xRms + ioW * 0.5 + 0.02, BASE_Y + 0.05, 0.45],
    [xRms + ioW * 0.5 + 0.50, wkvY * 0.7, wvZ * 0.55],
    [xWeightsIn - 0.55, wkvY, wvZ],
    [xWeightsIn - 0.06, wkvY, wvZ],
    [0, 1, 0],
    wvOnPath && hasFocus && !hasCellTrace ? 0.095 : 0.075,
    projFanCol(wvOnPath || activePart === 'rmsnorm', RIBBON_V_FLOW),
    true,
    18,
    true,
  );

  // ==========================================================================
  // 3. STAGE 3A: Twin 16-Head 3D Query Fin Stacks (Q00..Q15 & Q16..Q31, upright XY cards)
  // ==========================================================================
  const qRestY = 0.56;
  const qRestW = 0.76;
  const qRestH = 0.90;
  const qRepLiftY = 1.02;  // subtle reveal for default displayed representative head (Q00 / Q16)
  const qActiveLiftY = 1.88; // full spotlight reveal for actively selected head (e.g. Q18)
  const qActiveW = 0.86;
  const qActiveH = 1.02;

  for (let qIdx = 0; qIdx < 32; qIdx++) {
    const grp = kvHeadForQuery(qIdx);
    const pos = qHeadPos(qIdx);
    const isActiveSelected = activeHeadIdx === qIdx;
    const isDisplayedRep = (grp === 0 && group0Head === qIdx) || (grp === 1 && group1Head === qIdx);

    let fallbackHl = SHEET_HL_NORMAL;
    if (isActiveSelected) {
      fallbackHl = SHEET_HL_ACTIVE;
    } else if (activePart === 'q-proj' || activePart === 'w-q') {
      fallbackHl = SHEET_HL_ACTIVE;
    } else if (activePart === `kv-head-${grp}`) {
      fallbackHl = SHEET_HL_PATHWAY;
    } else if (hasFocus) {
      fallbackHl = SHEET_HL_DIMMED;
    } else if (isFp8Mode) {
      fallbackHl = SHEET_HL_BF16_GOLD;
    }
    const hl = resolveSheetHighlight(
      focusSet,
      { tensor: 'q', headIndex: qIdx, kvHeadIndex: grp },
      fallbackHl,
    );

    const cy = isActiveSelected ? qActiveLiftY : isDisplayedRep ? qRepLiftY : qRestY;
    const ex = isActiveSelected ? qActiveW : qRestW;
    const ez = isActiveSelected ? qActiveH : qRestH;

    b.addSheet({
      cx: pos.x,
      cy,
      cz: pos.z,
      ex,
      ey: 0.044,
      ez,
      axes: FRAME_XY_AXES,
      gridCols: sim.dHead,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: hl,
      seed: bayId * 0.5 + qIdx * 0.73 + clampedT * 0.15,
      partId: 'q-proj',
      headIndex: qIdx,
      kvHeadIndex: grp,
      tensorKey: 'q',
      atlasSlice: atlas.qHeads[qIdx],
      registryKey: `q:${qIdx}`,
      cellFamily: CELL_WEIGHT_BLUE,
    });

    // Crisp vertical slot-anchor line under lifted head so user sees its exact index in the 16-head stack
    if (isActiveSelected || isDisplayedRep) {
      b.addGuideLine(
        pos.x, 0.04, pos.z,
        pos.x, cy - ez * 0.5 - 0.03, pos.z,
        isActiveSelected ? 1.8 : 1.0,
        isActiveSelected ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
        isActiveSelected ? 0.92 : 0.45,
      );
    }
  }

  // Quiet base rail lines beneath each 16-head Q stack
  for (let grp = 0; grp < 2; grp++) {
    const pStart = qHeadPos(grp * 16);
    const pEnd = qHeadPos(grp * 16 + 15);
    const isGrpAct = activeKvGroup === grp;
    b.addGuideLine(
      pStart.x, 0.04, pStart.z - 0.08,
      pEnd.x, 0.04, pEnd.z + 0.08,
      isGrpAct ? 1.6 : 1.0,
      isGrpAct ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
      isGrpAct ? 0.85 : 0.5,
    );
  }

  // Smooth 3D ribbons: W_Q -> Group 0 & Group 1 Q fin stacks (targeted to displayed/selected head)
  for (let grp = 0; grp < 2; grp++) {
    const repHead = grp === 0 ? group0Head : group1Head;
    const isGrpAct = activeKvGroup === grp && !hasCellTrace;
    const isSelHead = activeHeadIdx !== null && kvHeadForQuery(activeHeadIdx) === grp;
    const targetPos = qHeadPos(repHead);
    const targetY = isSelHead ? qActiveLiftY : qRepLiftY;
    const startZ = grp === 0 ? -1.85 : 1.85;

    b.flow.addCubic3D(
      [xWeightsIn + 0.05, wqY, startZ],
      [xWeightsIn + 0.75, wqY, startZ],
      [targetPos.x - 0.85, targetY, targetPos.z],
      [targetPos.x - (isSelHead ? qActiveW : qRestW) * 0.5 - 0.05, targetY, targetPos.z],
      [0, 1, 0],
      isGrpAct || activePart === 'w-q' ? 0.095 : 0.072,
      isGrpAct || activePart === 'w-q'
        ? RIBBON_PATHWAY_SOFT
        : hasFocus
          ? RIBBON_DIMMED
          : RIBBON_QUIET,
      true,
      18,
      true,
    );
  }

  // ==========================================================================
  // 4. STAGE 3B: Explicitly Separate Shared K_g (x = -1.05) & Shared V_g (x = +3.95)
  //    Strictly fed from W_K and W_V respectively — ZERO arrows from Q into K or V!
  // ==========================================================================
  for (let kh = 0; kh < 2; kh++) {
    const grp = kh as 0 | 1;
    const kvPartId = grp === 0 ? 'kv-head-0' : 'kv-head-1';
    const isDirectKv = activePart === kvPartId;
    const isSharedForHead = activeHeadIdx !== null && kvHeadForQuery(activeHeadIdx) === grp;
    const isKvRelevant = isDirectKv || isSharedForHead || activePart === 'k-proj' || activePart === 'v-proj';

    const railFallbackHl = isDirectKv
      ? SHEET_HL_ACTIVE
      : isSharedForHead
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;

    // Ghost 16-slot capacity rails for Shared K_g and Shared V_g
    const kRailHasFocus =
      hasCellTrace && resolveSheetFocusRole(focusSet, { tensor: 'kv_k', kvHeadIndex: grp }) !== null;
    const vRailHasFocus =
      hasCellTrace && resolveSheetFocusRole(focusSet, { tensor: 'kv_v', kvHeadIndex: grp }) !== null;
    for (const [trackX, railHasFocus] of [
      [xKTrack, kRailHasFocus],
      [xVTrack, vRailHasFocus],
    ] as const) {
      const railHl = hasCellTrace
        ? railHasFocus
          ? SHEET_HL_NORMAL
          : SHEET_HL_DIMMED
        : railFallbackHl;
      b.addSheet({
        cx: trackX,
        cy: 0.015,
        cz: zGrp[grp],
        ex: kvSliceWidth + 0.06,
        ey: 0.024,
        ez: groupSpanZ + 0.06,
        axes: FRAME_XZ_AXES,
        gridCols: 1,
        gridRows: 16,
        sheetKind: SHEET_KIND_GHOST_RAIL,
        role: SHEET_ROLE_NORM,
        fp8Flag: 0,
        highlight: railHl,
        seed: grp * 2.1 + (trackX > 0 ? 1.0 : 0.0),
        partId: kvPartId,
        kvHeadIndex: grp,
        cellFamily: CELL_NEUTRAL,
      });
    }

    // Live token slices t = 1..T for Shared K_g and Shared V_g
    for (let t = 1; t <= clampedT; t++) {
      const tokIdx = t - 1;
      const sz = kvSliceZ(grp, t);
      const sliceFallbackHl = isDirectKv
        ? SHEET_HL_ACTIVE
        : isSharedForHead
          ? SHEET_HL_PATHWAY
          : hasFocus && !isKvRelevant
            ? SHEET_HL_DIMMED
            : SHEET_HL_NORMAL;
      const kSliceHl = resolveSheetHighlight(
        focusSet,
        { tensor: 'kv_k', kvHeadIndex: grp, tokenIndex: tokIdx },
        sliceFallbackHl,
      );
      const vSliceHl = resolveSheetHighlight(
        focusSet,
        { tensor: 'kv_v', kvHeadIndex: grp, tokenIndex: tokIdx },
        sliceFallbackHl,
      );
      const isElevated = (h: number) =>
        h === SHEET_HL_ACTIVE ||
        h === SHEET_HL_OPERAND_A ||
        h === SHEET_HL_OPERAND_B ||
        h === SHEET_HL_PATHWAY;

      // Shared K_g slice (teal-sage)
      b.addSheet({
        cx: xKTrack,
        cy: isElevated(kSliceHl) ? 0.072 : 0.056,
        cz: sz,
        ex: kvSliceWidth,
        ey: 0.058,
        ez: kvSliceDepth,
        axes: FRAME_XZ_AXES,
        gridCols: sim.dHead,
        gridRows: 1,
        sheetKind: SHEET_KIND_STANDARD,
        role: SHEET_ROLE_KV_K,
        fp8Flag: kvCacheFp8 ? 1 : 0,
        highlight: kSliceHl,
        seed: t * 1.13 + grp * 4.3 + bayId * 0.1,
        partId: kvPartId,
        kvHeadIndex: grp,
        tensorKey: 'kv_k',
        tokenIndex: tokIdx,
        atlasSlice: atlas.kvK[grp][tokIdx],
        registryKey: `kv_k:${grp}:${tokIdx}`,
        cellFamily: CELL_ACTIVATION_GREEN,
      });

      // Shared V_g slice (steel-cyan)
      b.addSheet({
        cx: xVTrack,
        cy: isElevated(vSliceHl) ? 0.072 : 0.056,
        cz: sz,
        ex: kvSliceWidth,
        ey: 0.058,
        ez: kvSliceDepth,
        axes: FRAME_XZ_AXES,
        gridCols: sim.dHead,
        gridRows: 1,
        sheetKind: SHEET_KIND_STANDARD,
        role: SHEET_ROLE_KV_V,
        fp8Flag: kvCacheFp8 ? 1 : 0,
        highlight: vSliceHl,
        seed: t * 1.29 + grp * 4.7 + bayId * 0.1,
        partId: kvPartId,
        kvHeadIndex: grp,
        tensorKey: 'kv_v',
        tokenIndex: tokIdx,
        atlasSlice: atlas.kvV[grp][tokIdx],
        registryKey: `kv_v:${grp}:${tokIdx}`,
        cellFamily: CELL_ACTIVATION_GREEN,
      });
    }

    // Direct W_K -> Shared K_g ribbon (routes cleanly along low floor lane y=0.05, NEVER via Q!)
    const kActive = (isSharedForHead || isDirectKv || activePart === 'k-proj') && !hasCellTrace;
    const kEntryZ = grp === 0 ? -3.65 : 0.25;
    b.flow.addCubic3D(
      [xWeightsIn + 0.06, wkvY - 0.12, wkZ],
      [xWeightsIn + 1.05, 0.06, kEntryZ],
      [xKTrack - 0.85, 0.06, zGrp[grp] - groupSpanZ * 0.42],
      [xKTrack - kvSliceWidth * 0.5 - 0.05, 0.06, kvSliceZ(grp, Math.max(1, Math.min(clampedT, 3)))],
      [0, 1, 0],
      kActive ? 0.095 : 0.068,
      kActive ? RIBBON_K_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_K_FLOW,
      true,
      22,
      true,
    );

    // Direct W_V -> Shared V_g bypass ribbon (routes cleanly along central/outer floor corridor y=0.04,
    // bypassing Q and Scores entirely to feed V_g at the Weighted-V stage!)
    const vActive = (isSharedForHead || isDirectKv || activePart === 'v-proj') && !hasCellTrace;
    const vCorridorZ = grp === 0 ? -0.18 : 3.74;
    b.flow.addPolyline3D(
      [
        [xWeightsIn + 0.06, wkvY - 0.15, wvZ],
        [xWeightsIn + 0.95, 0.05, vCorridorZ],
        [xVTrack - 0.65, 0.05, vCorridorZ],
        [xVTrack - kvSliceWidth * 0.5 - 0.06, 0.06, zGrp[grp] + (grp === 0 ? 0.45 : 0.45)],
      ],
      [0, 1, 0],
      vActive ? 0.095 : 0.068,
      vActive ? RIBBON_V_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_V_FLOW,
      true,
      0.20,
      0.20,
      true,
    );
  }

  // KV depth ticks (1, 8, 16, active T) along outer left edge of Shared K1 (clear of Scores!)
  const tickSet = new Set<number>([1, 8, 16, clampedT]);
  for (const t of Array.from(tickSet).sort((a, c) => a - c)) {
    b.kvDepthTicks.push({
      token: t,
      world: [xKTrack - kvSliceWidth * 0.5 - 0.16, 0.08, kvSliceZ(1, t)],
    });
  }

  // ==========================================================================
  // 5. STAGE 3C: Causal Scores (T x T) -> Softmax (T x T) -> V_out (2 Groups)
  //    Q_h + Shared K_g converge at Scores; Softmax + Shared V_g converge at V_out!
  // ==========================================================================
  const vOutDepthZ = 2.46; // 16 heads along Z
  const vOutWidthX = 0.82; // T tokens along X

  for (let kh = 0; kh < 2; kh++) {
    const grp = kh as 0 | 1;
    const isGrpActive = activeKvGroup === grp;
    const repHead = grp === 0 ? group0Head : group1Head;
    const repPos = qHeadPos(repHead);
    const isSelHead = activeHeadIdx !== null && kvHeadForQuery(activeHeadIdx) === grp;
    const repY = isSelHead ? qActiveLiftY : qRepLiftY;
    const repW = isSelHead ? qActiveW : qRestW;

    const scoresFallbackHl =
      activePart === 'attn-scores'
        ? SHEET_HL_ACTIVE
        : isGrpActive
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    const scoresHl = resolveSheetHighlight(
      focusSet,
      { tensor: 'scores', headIndex: repHead, kvHeadIndex: grp },
      scoresFallbackHl,
    );

    // Causal Scores = Q_h * K_g^T / sqrt(d_h)
    b.addSheet({
      cx: xScores,
      cy: DECK_Y,
      cz: zGrp[grp],
      ex: scoreSize,
      ey: SHEET_THICK,
      ez: scoreSize,
      axes: FRAME_XZ_AXES,
      gridCols: clampedT,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_CAUSAL_SCORES,
      role: SHEET_ROLE_SCORES,
      fp8Flag: 0,
      highlight: scoresHl,
      seed: 10.2 + grp * 3.1 + clampedT * 0.2,
      partId: 'attn-scores',
      headIndex: repHead,
      kvHeadIndex: grp,
      tensorKey: 'scores',
      atlasSlice: atlas.scores[grp],
      registryKey: `scores:${grp}`,
      cellFamily: CELL_WEIGHT_BLUE,
    });

    const smFallbackHl =
      activePart === 'softmax'
        ? SHEET_HL_ACTIVE
        : isGrpActive
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    const smHl = resolveSheetHighlight(
      focusSet,
      { tensor: 'softmax', headIndex: repHead, kvHeadIndex: grp },
      smFallbackHl,
    );

    // Causal Softmax (row-wise probabilities)
    b.addSheet({
      cx: xSoftmax,
      cy: DECK_Y,
      cz: zGrp[grp],
      ex: scoreSize,
      ey: SHEET_THICK,
      ez: scoreSize,
      axes: FRAME_XZ_AXES,
      gridCols: clampedT,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
      role: SHEET_ROLE_SOFTMAX,
      fp8Flag: 0,
      highlight: smHl,
      seed: 14.5 + grp * 2.7 + clampedT * 0.2,
      partId: 'softmax',
      headIndex: repHead,
      kvHeadIndex: grp,
      tensorKey: 'softmax',
      atlasSlice: atlas.softmax[grp],
      registryKey: `softmax:${grp}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    // Weighted Value Output V_out (cols = 16 heads along +Z, rows = T along -X)
    const vOutFallbackHl =
      activePart === 'weighted-v'
        ? SHEET_HL_ACTIVE
        : isGrpActive
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    const vOutHl = resolveSheetHighlight(
      focusSet,
      { tensor: 'v_out', kvHeadIndex: grp },
      vOutFallbackHl,
    );

    b.addSheet({
      cx: xVOut,
      cy: DECK_Y,
      cz: zGrp[grp],
      ex: vOutDepthZ, // along axisU (+Z)
      ey: SHEET_THICK,
      ez: vOutWidthX, // along axisV (-X)
      axes: FRAME_XZ_ROT90_AXES,
      gridCols: 16 * sim.dHead,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: vOutHl,
      seed: 18.1 + grp * 3.4,
      partId: 'weighted-v',
      kvHeadIndex: grp,
      tensorKey: 'v_out',
      atlasSlice: atlas.vOut[grp],
      registryKey: `v_out:${grp}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    const laneActive = isGrpActive && !hasCellTrace;

    // Convergence 1A: Lifted Q_h -> Scores (arches cleanly OVER Shared K_g without intersecting it!)
    b.flow.addCubic3D(
      [repPos.x + repW * 0.5 + 0.04, repY, repPos.z],
      [xKTrack - 0.1, Math.max(repY + 0.15, 1.35), repPos.z * 0.7 + zGrp[grp] * 0.3],
      [xScores - scoreSize * 0.5 - 0.35, DECK_Y + 0.55, zGrp[grp]],
      [xScores - scoreSize * 0.5 - 0.05, DECK_Y + 0.06, zGrp[grp] - 0.12],
      [0, 1, 0],
      laneActive ? 0.105 : 0.072,
      laneActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
      20,
      true,
    );

    // Convergence 1B: Shared K_g -> Scores (short upward-right climb into Scores!)
    b.flow.addCubic3D(
      [xKTrack + kvSliceWidth * 0.5 + 0.04, 0.08, zGrp[grp] + 0.18],
      [xKTrack + 0.75, 0.18, zGrp[grp] + 0.16],
      [xScores - scoreSize * 0.5 - 0.35, DECK_Y + 0.06, zGrp[grp] + 0.14],
      [xScores - scoreSize * 0.5 - 0.05, DECK_Y + 0.05, zGrp[grp] + 0.14],
      [0, 1, 0],
      laneActive ? 0.095 : 0.068,
      laneActive ? RIBBON_K_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_K_FLOW,
      true,
      16,
      true,
    );

    // Scores -> Softmax
    b.flow.addPolyline3D(
      [
        [xScores + scoreSize * 0.5 + 0.04, DECK_Y + 0.04, zGrp[grp]],
        [xSoftmax - scoreSize * 0.5 - 0.05, DECK_Y + 0.04, zGrp[grp]],
      ],
      [0, 1, 0],
      laneActive || activePart === 'attn-scores' || activePart === 'softmax' ? 0.10 : 0.072,
      laneActive || activePart === 'attn-scores' || activePart === 'softmax'
        ? RIBBON_ACTIVE
        : hasFocus
          ? RIBBON_DIMMED
          : RIBBON_QUIET,
      true,
      0.18,
      0.18,
      false,
    );

    // Convergence 2A: Softmax -> V_out (arches cleanly OVER Shared V_g!)
    b.flow.addCubic3D(
      [xSoftmax + scoreSize * 0.5 + 0.04, DECK_Y + 0.06, zGrp[grp] - 0.14],
      [xVTrack - 0.15, DECK_Y + 0.62, zGrp[grp] - 0.14],
      [xVTrack + 0.35, DECK_Y + 0.55, zGrp[grp] - 0.14],
      [xVOut - vOutWidthX * 0.5 - 0.05, DECK_Y + 0.06, zGrp[grp] - 0.14],
      [0, 1, 0],
      laneActive ? 0.10 : 0.072,
      laneActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
      20,
      true,
    );

    // Convergence 2B: Shared V_g -> V_out (climbs cleanly from V rack into V_out!)
    b.flow.addCubic3D(
      [xVTrack + kvSliceWidth * 0.5 + 0.04, 0.08, zGrp[grp] + 0.18],
      [xVTrack + 0.55, 0.20, zGrp[grp] + 0.18],
      [xVOut - vOutWidthX * 0.5 - 0.25, DECK_Y + 0.05, zGrp[grp] + 0.18],
      [xVOut - vOutWidthX * 0.5 - 0.05, DECK_Y + 0.05, zGrp[grp] + 0.18],
      [0, 1, 0],
      laneActive ? 0.095 : 0.068,
      laneActive ? RIBBON_V_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_V_FLOW,
      true,
      16,
      true,
    );
  }

  // ==========================================================================
  // 6. STAGE 4: Concat [4,096] -> Upright Output Projection Wall W_O -> + Residual
  // ==========================================================================
  const concatDepthZ = 5.20;
  const concatWidthX = 0.86;
  const concatHl = stageHighlight('concat', 'concat', activeHeadIdx !== null || activeKvGroup !== null);

  b.addSheet({
    cx: xConcat,
    cy: DECK_Y,
    cz: 0,
    ex: concatDepthZ, // along +Z (all 32 heads side-by-side)
    ey: SHEET_THICK,
    ez: concatWidthX, // along -X (T tokens)
    axes: FRAME_XZ_ROT90_AXES,
    gridCols: sim.qConcatDim,
    gridRows: clampedT,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: concatHl,
    seed: 22.3 + clampedT * 0.1,
    partId: 'concat',
    tensorKey: 'concat',
    atlasSlice: atlas.concat,
    registryKey: 'concat',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // V_out (Group 0 & 1) -> Concat
  for (let grp = 0; grp < 2; grp++) {
    const laneActive = activeKvGroup === grp && !hasCellTrace;
    const tgtZ = grp === 0 ? -1.45 : 1.45;
    b.flow.addCubic3D(
      [xVOut + vOutWidthX * 0.5 + 0.04, DECK_Y + 0.04, zGrp[grp]],
      [xVOut + vOutWidthX * 0.5 + 0.25, DECK_Y + 0.04, zGrp[grp]],
      [xConcat - concatWidthX * 0.5 - 0.25, DECK_Y + 0.04, tgtZ],
      [xConcat - concatWidthX * 0.5 - 0.05, DECK_Y + 0.04, tgtZ],
      [0, 1, 0],
      laneActive ? 0.10 : 0.072,
      laneActive ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
      14,
      false,
    );
  }

  // Upright W_O Weight Wall (FRAME_YZ_AXES, mirrors W_Q on the output side!)
  const woY = 1.48;
  const woSpanZ = 4.80;
  const woHeightY = 1.00;
  const woHl = stageHighlight('w_o', 'o-proj', activeHeadIdx !== null || activeKvGroup !== null);

  b.addSheet({
    cx: xWO,
    cy: woY,
    cz: 0,
    ex: woSpanZ,
    ey: SHEET_THICK,
    ez: woHeightY,
    axes: FRAME_YZ_AXES,
    gridCols: sim.qConcatDim,
    gridRows: sim.d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: woHl,
    seed: 25.8 + bayId * 0.3,
    partId: 'o-proj',
    tensorKey: 'w_o',
    atlasSlice: atlas.wO,
    registryKey: 'w_o',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Concat -> Upright W_O
  const outAct = (activePart === 'o-proj' || activePart === 'concat' || activeHeadIdx !== null) && !hasCellTrace;
  b.flow.addCubic3D(
    [xConcat + concatWidthX * 0.5 + 0.04, DECK_Y + 0.05, 0],
    [xConcat + concatWidthX * 0.5 + 0.32, woY * 0.72, 0],
    [xWO - 0.32, woY, 0],
    [xWO - 0.06, woY, 0],
    [0, 1, 0],
    outAct ? 0.10 : 0.076,
    outAct ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
    true,
    16,
    true,
  );

  // Upright W_O -> + Residual Output
  b.flow.addCubic3D(
    [xWO + 0.06, woY, 0],
    [xWO + 0.48, woY * 0.75, 0],
    [xResOut - ioW * 0.5 - 0.35, BASE_Y + 0.15, 0],
    [xResOut - ioW * 0.5 - 0.05, BASE_Y + 0.05, 0],
    [0, 1, 0],
    outAct || (!hasCellTrace && activePart === 'residual-out') ? 0.105 : 0.08,
    outAct || (!hasCellTrace && activePart === 'residual-out')
      ? RIBBON_RESIDUAL_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL,
    true,
    16,
    true,
  );

  // ==========================================================================
  // 7. LIVE 3D CELL HIGHLIGHTS & CROSS-PLANE OPERAND DEPENDENCY THREADS
  // ==========================================================================
  if (activeFormula) {
    const displayedHeads: readonly [number, number] = [group0Head, group1Head];
    const targetSpan = resolveSpatialSpan(b.sheetRegistry, activeFormula.targetSpan, displayedHeads);

    if (targetSpan) {
      drawCellHighlight3D(b.flow, targetSpan, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.024, true);

      for (const up of activeFormula.upstreamSpans) {
        const upSpan = resolveSpatialSpan(b.sheetRegistry, up, displayedHeads);
        if (!upSpan) continue;
        drawOperandDependency3D(b.flow, upSpan, targetSpan, up.role === 'operand_a');
      }
    }
  }

  // ==========================================================================
  // 8. 3D EXTERIOR LABELS & DIMENSION CALIPERS (Accurate Head-Specific Naming)
  // ==========================================================================
  const g0Tag = fmtHeadTag(group0Head);
  const g1Tag = fmtHeadTag(group1Head);
  const grp0Selected = activeHeadIdx !== null && activeHeadIdx < 16;
  const grp1Selected = activeHeadIdx !== null && activeHeadIdx >= 16;
  const selectedHeadLabel =
    activeHeadIdx !== null
      ? `${fmtHeadTag(activeHeadIdx)} → KV${kvHeadForQuery(activeHeadIdx)}`
      : null;

  const g0AnchorPos = grp0Selected && activeHeadIdx !== null ? qHeadPos(activeHeadIdx) : qHeadPos(group0Head);
  const g1AnchorPos = grp1Selected && activeHeadIdx !== null ? qHeadPos(activeHeadIdx) : qHeadPos(group1Head);

  b.landmarks.push(
    { id: 'g-res-in', label: 'Input (2,688)', world: [xInput, 0.18, ioD * 0.5 + 0.24], partId: 'residual-in', align: 'bottom', tier: 'primary' },
    { id: 'g-rmsnorm', label: 'RMSNorm', world: [xRms, 0.18, ioD * 0.5 + 0.24], partId: 'rmsnorm', align: 'bottom', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wq', label: 'W_Q', world: [xWeightsIn, wqY + wqHeightY * 0.5 + 0.20, 0], partId: 'w-q', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wk', label: 'W_K', world: [xWeightsIn, wkvY + wkvHeightY * 0.5 + 0.18, wkZ - 0.22], partId: 'k-proj', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wv', label: 'W_V', world: [xWeightsIn, 0.12, wvZ + wkvSpanZ * 0.5 + 0.24], partId: 'v-proj', align: 'bottom', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },

    // Query stacks with active head badge anchored directly above the lifted head sheet
    {
      id: 'g-q0',
      label: grp0Selected ? `Q₀..Q₁₅ · ${selectedHeadLabel}` : `Q₀..Q₁₅ (×16 · ${g0Tag})`,
      world: [
        grp0Selected ? g0AnchorPos.x : xQBase + 0.28,
        grp0Selected ? qActiveLiftY + qActiveH * 0.5 + 0.26 : qRepLiftY + qRestH * 0.5 + 0.20,
        grp0Selected ? g0AnchorPos.z : zGrp[0],
      ],
      partId: grp0Selected ? activePart ?? 'q-proj' : 'q-proj',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'g-q1',
      label: grp1Selected ? `Q₁₆..Q₃₁ · ${selectedHeadLabel}` : `Q₁₆..Q₃₁ (×16 · ${g1Tag})`,
      world: [
        grp1Selected ? g1AnchorPos.x : xQBase + 0.28,
        grp1Selected ? qActiveLiftY + qActiveH * 0.5 + 0.26 : qRepLiftY + qRestH * 0.5 + 0.20,
        grp1Selected ? g1AnchorPos.z : zGrp[1],
      ],
      partId: grp1Selected ? activePart ?? 'q-proj' : 'q-proj',
      align: 'top',
      tier: 'primary',
    },

    // Explicitly distinguished Shared K and Shared V racks!
    {
      id: 'g-kv0-k',
      label: kvCacheFp8 ? `K₀ · FP8 (T=${clampedT})` : `K₀ (T=${clampedT})`,
      world: [xKTrack, 0.16, zGrp[0] - groupSpanZ * 0.5 - 0.22],
      partId: 'kv-head-0',
      align: 'top',
      tier: 'primary',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },
    {
      id: 'g-kv1-k',
      label: kvCacheFp8 ? `K₁ · FP8 (T=${clampedT})` : `K₁ (T=${clampedT})`,
      world: [xKTrack, 0.16, zGrp[1] + groupSpanZ * 0.5 + 0.22],
      partId: 'kv-head-1',
      align: 'bottom',
      tier: 'primary',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },
    {
      id: 'g-kv0-v',
      label: kvCacheFp8 ? `V₀ · FP8 (T=${clampedT})` : `V₀ (T=${clampedT})`,
      world: [xVTrack, 0.16, zGrp[0] - groupSpanZ * 0.5 - 0.22],
      partId: 'kv-head-0',
      align: 'top',
      tier: 'primary',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },
    {
      id: 'g-kv1-v',
      label: kvCacheFp8 ? `V₁ · FP8 (T=${clampedT})` : `V₁ (T=${clampedT})`,
      world: [xVTrack, 0.16, zGrp[1] + groupSpanZ * 0.5 + 0.22],
      partId: 'kv-head-1',
      align: 'bottom',
      tier: 'primary',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },

    // Accurately head-labeled Scores & Softmax matrices
    {
      id: 'g-scores-0',
      label: `Scores · ${g0Tag}×K₀`,
      world: [xScores, DECK_Y + 0.14, zGrp[0] - scoreSize * 0.5 - 0.20],
      partId: 'attn-scores',
      align: 'top',
      tier: grp0Selected ? 'primary' : 'detail',
    },
    {
      id: 'g-scores',
      label: `Scores · ${g1Tag}×K₁ (${clampedT}×${clampedT})`,
      world: [xScores, DECK_Y + 0.14, zGrp[1] + scoreSize * 0.5 + 0.22],
      partId: 'attn-scores',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'g-softmax-0',
      label: `Softmax · ${g0Tag}`,
      world: [xSoftmax, DECK_Y + 0.14, zGrp[0] - scoreSize * 0.5 - 0.20],
      partId: 'softmax',
      align: 'top',
      tier: grp0Selected ? 'primary' : 'detail',
    },
    {
      id: 'g-softmax',
      label: `Softmax · ${g1Tag}`,
      world: [xSoftmax, DECK_Y + 0.14, zGrp[1] + scoreSize * 0.5 + 0.22],
      partId: 'softmax',
      align: 'bottom',
      tier: 'primary',
    },

    { id: 'g-vout', label: 'V Out', world: [xVOut, DECK_Y + 0.14, zGrp[1] + vOutDepthZ * 0.5 + 0.22], partId: 'weighted-v', align: 'bottom', tier: 'primary' },
    { id: 'g-concat', label: 'Concat [4,096]', world: [xConcat, DECK_Y + 0.14, concatDepthZ * 0.5 + 0.22], partId: 'concat', align: 'bottom', tier: 'primary' },
    { id: 'g-oproj', label: 'W_O', world: [xWO, woY + woHeightY * 0.5 + 0.20, 0], partId: 'o-proj', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-res-out', label: '+ Residual', world: [xResOut, 0.18, ioD * 0.5 + 0.24], partId: 'residual-out', align: 'bottom', tier: 'primary' },
    { id: 'g-res-bridge', label: 'Residual Highway (2,688)', world: [0.3, bridgeY + 0.22, bridgeZ], partId: 'residual-in', align: 'top', tier: 'primary' },

    // Progressive detail labels (revealed on hover, selection, or close zoom)
    { id: 'g-nope', label: 'NoPE · order from Mamba-2', world: [xRms, 0.16, -ioD * 0.5 - 0.24], partId: 'rmsnorm', align: 'top', tier: 'detail', tone: 'cyan' },
    { id: 'g-wq-shape', label: isFp8Mode ? '2,688 → 4,096 · BF16' : '2,688 → 4,096', world: [xWeightsIn, wqY - wqHeightY * 0.5 - 0.14, 0], partId: 'w-q', align: 'bottom', tier: 'detail', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wo-shape', label: isFp8Mode ? '4,096 → 2,688 · BF16' : '4,096 → 2,688', world: [xWO, woY - woHeightY * 0.5 - 0.14, 0], partId: 'o-proj', align: 'bottom', tier: 'detail', tone: isFp8Mode ? 'gold' : 'default' },
  );

  // Subtle architectural calipers
  const calGrp: 0 | 1 = activeKvGroup ?? 1;
  b.calipers.push(
    {
      id: 'cal-gqa-ratio',
      aWorld: [xQBase, 0.08, zGrp[calGrp] - groupSpanZ * 0.5],
      bWorld: [xQBase + 15 * qStepX, 0.08, zGrp[calGrp] + groupSpanZ * 0.5],
      label: calGrp === 0 ? '16 Q Heads → 1 Shared K₀ / V₀' : '16 Q Heads → 1 Shared K₁ / V₁',
      partIds: ['q-proj', 'kv-head-0', 'kv-head-1'],
      minPixels: 110,
      offsetPx: 12,
    },
    {
      id: 'cal-scores-t',
      aWorld: [xScores - scoreSize * 0.5, DECK_Y + 0.04, zGrp[calGrp] + scoreSize * 0.5 + 0.10],
      bWorld: [xScores + scoreSize * 0.5, DECK_Y + 0.04, zGrp[calGrp] + scoreSize * 0.5 + 0.10],
      label: `T = ${clampedT}`,
      partIds: ['attn-scores', 'softmax'],
      minPixels: 48,
      offsetPx: 8,
    },
  );

  const packed = b.pack();
  const reg = (k: string) => b.sheetRegistry.get(k);
  const focusGrp: 0 | 1 = activeKvGroup ?? (activeHeadIdx !== null ? kvHeadForQuery(activeHeadIdx) : 0);
  const focusHead = activeHeadIdx ?? (focusGrp === 0 ? group0Head : group1Head);
  const kvKSlices = Array.from({ length: clampedT }, (_, t) => reg(`kv_k:${focusGrp}:${t}`));
  const kvVSlices = Array.from({ length: clampedT }, (_, t) => reg(`kv_v:${focusGrp}:${t}`));

  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('input'), reg('rmsnorm')], 0.48, 0.28, 0.48, GQA_SPATIAL_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('rmsnorm'), reg('w_q'), reg('w_k'), reg('w_v'), reg(`q:${focusHead}`)], 0.48, 0.28, 0.48, GQA_SPATIAL_FIT_BOUNDS),
    2: computeSheetGroupBounds([reg(`q:${focusHead}`), ...kvKSlices, reg(`scores:${focusGrp}`)], 0.48, 0.28, 0.48, GQA_SPATIAL_FIT_BOUNDS),
    3: computeSheetGroupBounds([reg(`scores:${focusGrp}`), reg(`softmax:${focusGrp}`)], 0.48, 0.28, 0.48, GQA_SPATIAL_FIT_BOUNDS),
    4: computeSheetGroupBounds([reg(`softmax:${focusGrp}`), ...kvVSlices, reg(`v_out:${focusGrp}`)], 0.48, 0.28, 0.48, GQA_SPATIAL_FIT_BOUNDS),
    5: computeSheetGroupBounds([reg('concat'), reg('w_o'), reg('residual')], 0.48, 0.28, 0.48, GQA_SPATIAL_FIT_BOUNDS),
  };

  return {
    kind: 'cutaway_gqa',
    bayId,
    ...packed,
    fitBounds: GQA_SPATIAL_FIT_BOUNDS,
    opFocusBounds,
    landmarks: b.landmarks,
    kvDepthTicks: b.kvDepthTicks,
    calipers: b.calipers,
    valueAtlas: atlas,
    gqaSim: sim,
    activeFormula,
    solidPickBoxes: b.pickBoxes,
    partCatalog,
    gqaStats: {
      bayId: bayId === 43 ? 43 : 6,
      qHeadCount: 32,
      qGroupCount: 2,
      qHeadsPerGroup: 16,
      kvHeadCount: 2,
      gqaRatio: 16,
      kvTokenSliceCount: 2 * clampedT,
      computeFp8,
      kvCacheFp8,
      usesSmoothRibbons: true,
      voxelArrowCount: 0,
      selectedQueryHead: activeHeadIdx,
      activeGroup0Head: group0Head,
      activeGroup1Head: group1Head,
    },
  };
}
