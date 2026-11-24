import {
  evaluateSingleRoutedExpert,
  formatExpertId,
  MOE_SIM_CONFIG,
  type MoeCellFormula,
  type MoeCellRef,
  type MoeSheetSpan,
  type MoeSimResult,
  type MoeTensorKey,
} from '../math/moeSim';
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
  FRAME_XZ_AXES,
  FRAME_YZ_AXES,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetBasisAxes,
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
import type { CutawayPartInfo, CutawaySceneGeometry, MoECutawayStats } from './cutaways';
import type { MoeValueAtlas } from './moeAtlas';
import {
  CELL_OP_A_FILL,
  CELL_OP_A_FRAME,
  CELL_OP_B_FILL,
  CELL_OP_B_FRAME,
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  computeSheetGroupBounds,
  drawCellHighlight3D,
  drawOperandDependency3D,
  fmtSigned,
  resolveSheetHighlight,
  RIBBON_ACTIVE,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  RIBBON_RESIDUAL_ACTIVE,
  RIBBON_SSM_LOOP,
  spanFromSheetGrid,
  THREAD_OP_A,
  THREAD_OP_B,
  type SheetCandidateIdentity,
  type SheetFocusSet,
  type SheetGeomRecord,
  type WorldSpan3D,
} from './sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type LandmarkAnchor,
} from './spine';

/**
 * Invariant 3D bounding envelope for Spatial MoE (128+1) across T=1..16, Bay 02/52,
 * BF16/FP8 Selective, focused token t=0..T-1, and active/idle focused experts.
 */
export const MOE_SPATIAL_FIT_BOUNDS: AABB = {
  min: [-8.95, -0.06, -3.90],
  max: [9.25, 3.30, 3.65],
};

export interface SpatialMoeFitOptions {
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly isNarrow?: boolean;
  readonly azimuthDeg?: number;
  readonly elevationDeg?: number;
}

/**
 * Exact 8-corner projective camera fitter for Spatial MoE (128+1).
 * Mirrors `fitSpatialGqaCamera` and `fitSpatialMambaCamera`.
 */
export function fitSpatialMoeCamera(
  bounds: AABB,
  metrics: ViewportMetrics,
  options?: SpatialMoeFitOptions,
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

  let minR = Infinity;
  let maxR = -Infinity;
  let minU = Infinity;
  let maxU = -Infinity;
  let minE = Infinity;
  let maxE = -Infinity;

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

/**
 * Horizontal XZ sheet whose column axis runs along `-Z` (slot 0 at `+Z` front → slot N-1 at `-Z` rear)
 * and row axis runs along `+X`. Strictly right-handed (`cross(U, N) = [+1, 0, 0]`).
 */
const FRAME_XZ_COLS_Z: SheetBasisAxes = {
  axisU: [0, 0, -1],
  axisN: [0, 1, 0],
  axisV: [1, 0, 0],
};

// Semantic branch colors
const RIBBON_SHARED_FLOW: FlowRibbonColor = { r: 0.14, g: 0.56, b: 0.64, a: 0.44, borderAlpha: 0.24 };
const RIBBON_SHARED_ACTIVE: FlowRibbonColor = { r: 0.10, g: 0.54, b: 0.66, a: 0.82, borderAlpha: 0.42 };
const RIBBON_ROUTER_FLOW: FlowRibbonColor = { r: 0.78, g: 0.50, b: 0.16, a: 0.46, borderAlpha: 0.26 };
const RIBBON_ROUTER_ACTIVE: FlowRibbonColor = { r: 0.84, g: 0.44, b: 0.10, a: 0.84, borderAlpha: 0.45 };
const RIBBON_BRANCH_QUIET: FlowRibbonColor = { r: 0.26, g: 0.46, b: 0.68, a: 0.34, borderAlpha: 0.18 };
const RIBBON_BRANCH_FOCUS: FlowRibbonColor = { r: 0.86, g: 0.30, b: 0.20, a: 0.78, borderAlpha: 0.42 };
const TOKEN_ROW_FILL: FlowRibbonColor = { r: 0.14, g: 0.50, b: 0.72, a: 0.14, borderAlpha: 0.0 };
const TOKEN_ROW_FRAME: FlowRibbonColor = { r: 0.14, g: 0.50, b: 0.72, a: 0.72, borderAlpha: 0.25 };

interface SpatialMoeSheetSpec {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly axes: SheetBasisAxes;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly sheetKind?: number;
  readonly role: number;
  readonly fp8Flag: number;
  readonly highlight: number;
  readonly seed: number;
  readonly partId: string;
  readonly headIndex?: number;
  readonly tensorKey?: MoeTensorKey;
  readonly tokenIndex?: number;
  readonly atlasSlice?: AtlasSlice;
  readonly registryKey?: string;
  readonly cellFamily: number;
}

class SpatialMoeBuilder {
  readonly sheetFloats: number[] = [];
  readonly basisFloats: number[] = [];
  readonly solidFloats: number[] = [];
  readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();
  private readonly bayId: number;

  constructor(bayId: number) {
    this.bayId = bayId;
  }

  addSheet(s: SpatialMoeSheetSpec): SheetGeomRecord {
    const frame = makeSheetFrame3D(
      [s.cx, s.cy, s.cz],
      [s.ex, s.ey, s.ez],
      s.axes.axisU,
      s.axes.axisN,
    );

    const rect = s.atlasSlice ? s.atlasSlice.rect : ([0, 0, 0, 0] as const);
    const scale = s.atlasSlice ? s.atlasSlice.scale : 1.0;
    const modeFlag = s.atlasSlice ? s.atlasSlice.mode : 0;

    // 1. SheetsPass instance (20 floats = 80 bytes)
    this.sheetFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.gridCols, s.gridRows,
      s.sheetKind ?? SHEET_KIND_STANDARD,
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

    // 3. 1-to-1 mirror in solidData (12 floats: fp8Flag at +7, seed at +8)
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

    // 4. Oriented PickBox (slightly padded along axisN for crisp 3D picking)
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
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
      min: worldAabb.min,
      max: worldAabb.max,
      frame: pickFrame,
    });

    const rec: SheetGeomRecord = {
      cx: s.cx,
      cy: s.cy,
      cz: s.cz,
      ex: s.ex,
      ey: s.ey,
      ez: s.ez,
      gridCols: s.gridCols,
      gridRows: s.gridRows,
      frame,
    };
    if (s.registryKey) {
      this.sheetRegistry.set(s.registryKey, rec);
    }
    return rec;
  }

  addGuideLine(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth = 1.0,
    style = LINE_STYLE_GUIDE,
    alpha = 0.55,
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

  addBoxFrameXZ(
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    y: number,
    cssWidth = 1.0,
    style = LINE_STYLE_GUIDE,
    alpha = 0.45,
  ): void {
    this.addGuideLine(x0, y, z0, x1, y, z0, cssWidth, style, alpha);
    this.addGuideLine(x1, y, z0, x1, y, z1, cssWidth, style, alpha);
    this.addGuideLine(x1, y, z1, x0, y, z1, cssWidth, style, alpha);
    this.addGuideLine(x0, y, z1, x0, y, z0, cssWidth, style, alpha);
  }

  addDashedFrameXZ(
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    y: number,
    style = LINE_STYLE_GUIDE,
    alpha = 0.45,
    dash = 0.22,
    gap = 0.14,
  ): void {
    const dashedLine = (ax: number, az: number, bx: number, bz: number) => {
      const dx = bx - ax;
      const dz = bz - az;
      const len = Math.hypot(dx, dz);
      if (len < 1e-4) return;
      const ux = dx / len;
      const uz = dz / len;
      for (let d = 0; d < len; d += dash + gap) {
        const dEnd = Math.min(len, d + dash);
        this.addGuideLine(
          ax + ux * d, y, az + uz * d,
          ax + ux * dEnd, y, az + uz * dEnd,
          0.95,
          style,
          alpha,
        );
      }
    };
    dashedLine(x0, z0, x1, z0);
    dashedLine(x1, z0, x1, z1);
    dashedLine(x1, z1, x0, z1);
    dashedLine(x0, z1, x0, z0);
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

export interface BuildMoeSpatialContext {
  readonly bayId: 2 | 52;
  readonly clampedT: number;
  readonly isFp8Mode: boolean;
  readonly activePart: string | null;
  readonly focusToken: number;
  readonly focusExpert: number;
  readonly activeExpertIds: readonly number[];
  readonly activeRankMap: ReadonlyMap<number, number>;
  readonly focusedEval: ReturnType<typeof evaluateSingleRoutedExpert>;
  readonly activeCell: MoeCellRef | null;
  readonly activeFormula: MoeCellFormula | null;
  readonly focusSet: SheetFocusSet;
  readonly stepperStep: number | null;
  readonly sim: MoeSimResult;
  readonly atlas: MoeValueAtlas;
  readonly partCatalog: Readonly<Record<string, CutawayPartInfo>>;
  readonly moeStats: MoECutawayStats;
}

function resolveSpatialMoeSpan(
  registry: ReadonlyMap<string, SheetGeomRecord>,
  bankTileGeom: ReadonlyMap<number, SheetGeomRecord>,
  span: MoeSheetSpan,
  focusToken: number,
  focusExpert: number,
): WorldSpan3D | null {
  if (span.tensor === 'router_bank') {
    if (span.tokenIndex !== undefined && span.tokenIndex !== focusToken) {
      return null;
    }
    const eid =
      span.expertId ??
      Math.max(
        0,
        Math.min(
          MOE_SIM_CONFIG.nRoutedExperts - 1,
          span.rowStart * MOE_SIM_CONFIG.bankCols + span.colStart,
        ),
      );
    const tile = bankTileGeom.get(eid);
    return tile ? spanFromSheetGrid(tile, 0, 0, 0, 0) : null;
  }

  if (
    span.tensor === 'routed_w_up' ||
    span.tensor === 'routed_pre' ||
    span.tensor === 'routed_act' ||
    span.tensor === 'routed_w_down'
  ) {
    if (span.expertId !== undefined && span.expertId !== focusExpert) return null;
    if (span.tokenIndex !== undefined && span.tokenIndex !== focusToken) return null;
  } else if (span.tensor === 'topk_weights' || span.tensor === 'shared_act') {
    if (span.tokenIndex !== undefined && span.tokenIndex !== focusToken) return null;
  }

  const rec = registry.get(span.tensor);
  if (!rec) return null;
  return spanFromSheetGrid(rec, span.rowStart, span.rowEnd, span.colStart, span.colEnd);
}

/**
 * Spatial 3/4 MoE (128+1) Reference Implementation.
 *
 * Meaningful 3D axis semantics:
 *  - +X (left -> right): Forward per-token computation (Input -> RMSNorm -> Parallel Fork -> Merge -> + Residual)
 *  - +Y (elevation): Upright learned weight walls (W_router, W_up, W_down, Shared W_up, Shared W_down),
 *                    6 risen active expert towers above 122 idle floor tiles, elevated Top-6 normalization bridge,
 *                    and overhead emerald Residual Highway bridge
 *  - Z (depth): Center/rear sparse 128-expert routed corridor vs. front parallel always-on 2x-intermediate Shared Expert
 */
export function buildMoeSpatialScene(ctx: BuildMoeSpatialContext): CutawaySceneGeometry {
  const {
    bayId,
    clampedT,
    isFp8Mode,
    activePart,
    focusToken,
    focusExpert,
    activeExpertIds,
    activeRankMap,
    focusedEval,
    activeFormula,
    focusSet,
    stepperStep,
    sim,
    atlas,
    partCatalog,
    moeStats,
  } = ctx;

  const b = new SpatialMoeBuilder(bayId);
  const hasCellTrace = focusSet.active;
  const step = hasCellTrace ? null : stepperStep;
  const hasPartFocus = activePart !== null || step !== null;
  const hasAnyFocus = hasCellTrace || hasPartFocus;

  const focusExpTag = formatExpertId(focusExpert);
  const absFocusToken = focusToken + (sim.startTokenOffset ?? 0);
  const focusTokLabel = sim.tokenLabels[focusToken] ?? `t${absFocusToken}`;
  const focusedRank = activeRankMap.get(focusExpert) ?? null;

  // Exact proportions upholding Safeguard #7: Shared Expert = STRICTLY 2.0x intermediate, 1.0x hidden!
  const HIDDEN_SPAN = 1.10; // d = 8 mini / 2,688 prod
  const ROUTED_INTER_SPAN = 0.62; // d_exp = 4 mini / 1,856 prod
  const SHARED_INTER_SPAN = ROUTED_INTER_SPAN * 2; // 1.24 (exact 2.0x: d_shared = 8 mini / 3,712 prod)

  const BASE_Y = 0.04;
  const SHEET_THICK = 0.064;
  const tFrac = clampedT / MOE_SIM_CONFIG.maxTokens;
  const ioMaxZ = 1.48;
  const ioLiveZ = Math.max(0.24, 0.18 + tFrac * 1.26);
  const zAnchoredCenter = (centerZ: number, maxZ: number, liveZ: number): number =>
    centerZ - maxZ * 0.5 + liveZ * 0.5;

  // X Stage Anchors (left -> right)
  const xInput = -8.35;
  const xRms = -6.75;
  const xWRouter = -5.12;
  const bankCenterX = -2.15;
  const xUnbiased = 0.34;
  const xTopk = 0.96;
  const xRoutedWUp = 2.32;
  const xRoutedPre = 3.06;
  const xRoutedAct = 3.58;
  const xRoutedWDown = 4.30;
  const xBranches = 5.32;
  const xBranchOut = 6.45;
  const xMoeOut = 7.55;
  const xResOut = 8.65;

  // Z Corridor Anchors
  const zRouted = -1.15;
  const zShared = 2.45;
  const zSharedOut = 2.22;
  const zRoutedSum = -1.05;
  const bridgeY = 3.14;
  const bridgeZ = -3.68;

  const sheetHighlight = (
    candidate: SheetCandidateIdentity,
    partIds: readonly string[],
    stepStages: readonly number[],
    idleHighlight = SHEET_HL_NORMAL,
  ): number => {
    const isPartMatch =
      (activePart !== null && partIds.includes(activePart)) ||
      (step !== null && stepStages.includes(step));
    const fallback = isPartMatch
      ? SHEET_HL_ACTIVE
      : hasPartFocus
        ? SHEET_HL_DIMMED
        : idleHighlight;
    return resolveSheetHighlight(focusSet, candidate, fallback);
  };

  const addGhostSeqRail = (
    cx: number,
    cy: number,
    cz: number,
    ex: number,
    ez: number,
    seed: number,
    hl: number,
  ) => {
    const railHl =
      hl === SHEET_HL_DIMMED
        ? SHEET_HL_DIMMED
        : hl === SHEET_HL_ACTIVE || hl === SHEET_HL_OPERAND_A || hl === SHEET_HL_OPERAND_B
          ? SHEET_HL_PATHWAY
          : SHEET_HL_NORMAL;
    b.addSheet({
      cx,
      cy: cy - 0.025,
      cz,
      ex: ex + 0.05,
      ey: 0.022,
      ez: ez + 0.05,
      axes: FRAME_XZ_AXES,
      gridCols: 1,
      gridRows: 16,
      sheetKind: SHEET_KIND_GHOST_RAIL,
      role: SHEET_ROLE_NORM,
      fp8Flag: 0,
      highlight: railHl,
      seed,
      partId: 'seq-rail',
      cellFamily: CELL_NEUTRAL,
    });
  };

  // ==========================================================================
  // 1. STAGE 1: Input X, Pre-MoE RMSNorm & Overhead Emerald Residual Bridge
  // ==========================================================================
  const ioW = 0.88;
  const resInHl = sheetHighlight({ tensor: 'input', isAggregate: true }, ['residual-in'], [0, 5]);
  addGhostSeqRail(xInput, BASE_Y, 0, ioW, ioMaxZ, 0.1, resInHl);
  b.addSheet({
    cx: xInput,
    cy: BASE_Y,
    cz: zAnchoredCenter(0, ioMaxZ, ioLiveZ),
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resInHl,
    seed: 11 + clampedT * 0.1,
    partId: 'residual-in',
    tensorKey: 'input',
    atlasSlice: atlas.input,
    registryKey: 'input',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  const resOutHl = sheetHighlight({ tensor: 'residual', isAggregate: true }, ['residual-out'], [5]);
  addGhostSeqRail(xResOut, BASE_Y, 0, ioW, ioMaxZ, 0.2, resOutHl);
  b.addSheet({
    cx: xResOut,
    cy: BASE_Y,
    cz: zAnchoredCenter(0, ioMaxZ, ioLiveZ),
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resOutHl,
    seed: 19 + clampedT * 0.1,
    partId: 'residual-out',
    tensorKey: 'residual',
    atlasSlice: atlas.residual,
    registryKey: 'residual',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Overhead Emerald Residual Highway Bridge
  const resBridgeCol =
    !hasCellTrace && (activePart === 'residual-in' || activePart === 'residual-out' || step === 5)
      ? RIBBON_RESIDUAL_ACTIVE
      : hasCellTrace && activeFormula?.cell.tensor !== 'residual'
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL;
  b.flow.addCubic3D(
    [xInput, BASE_Y + 0.04, -ioMaxZ * 0.5],
    [xInput, bridgeY * 0.85, -ioMaxZ * 0.5 - 0.82],
    [xInput + 0.35, bridgeY, bridgeZ],
    [xInput + 0.95, bridgeY, bridgeZ],
    [0, 1, 0],
    0.15,
    resBridgeCol,
    false,
    14,
    true,
  );
  b.flow.addPolyline3D(
    [
      [xInput + 0.90, bridgeY, bridgeZ],
      [xResOut - 0.90, bridgeY, bridgeZ],
    ],
    [0, 1, 0],
    0.15,
    resBridgeCol,
    false,
    0.22,
    0.22,
    true,
  );
  b.flow.addCubic3D(
    [xResOut - 0.95, bridgeY, bridgeZ],
    [xResOut - 0.35, bridgeY, bridgeZ],
    [xResOut, bridgeY * 0.85, -ioMaxZ * 0.5 - 0.82],
    [xResOut, BASE_Y + 0.08, -ioMaxZ * 0.5 - 0.02],
    [0, 1, 0],
    0.15,
    resBridgeCol,
    true,
    14,
    true,
    0.30,
    0.32,
  );
  b.addGuideLine(xInput, BASE_Y, -ioMaxZ * 0.5, xInput, bridgeY, bridgeZ, 0.8, LINE_STYLE_GUIDE, 0.24);
  b.addGuideLine(xResOut, BASE_Y, -ioMaxZ * 0.5, xResOut, bridgeY, bridgeZ, 0.8, LINE_STYLE_GUIDE, 0.24);

  // Pre-MoE RMSNorm (U)
  const rmsHl = sheetHighlight({ tensor: 'rmsnorm', isAggregate: true }, ['rmsnorm'], [0]);
  addGhostSeqRail(xRms, BASE_Y, 0, ioW, ioMaxZ, 0.3, rmsHl);
  b.addSheet({
    cx: xRms,
    cy: BASE_Y,
    cz: zAnchoredCenter(0, ioMaxZ, ioLiveZ),
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: rmsHl,
    seed: 23,
    partId: 'rmsnorm',
    tensorKey: 'rmsnorm',
    atlasSlice: atlas.rmsnorm,
    registryKey: 'rmsnorm',
    cellFamily: CELL_MOE_AMBER,
  });

  // Input -> RMSNorm
  b.flow.addPolyline3D(
    [
      [xInput + ioW * 0.5 + 0.04, BASE_Y + 0.04, 0],
      [xRms - ioW * 0.5 - 0.05, BASE_Y + 0.04, 0],
    ],
    [0, 1, 0],
    0.105,
    !hasCellTrace && (rmsHl === SHEET_HL_ACTIVE || activePart === 'residual-in')
      ? RIBBON_ACTIVE
      : hasAnyFocus
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL,
    true,
    0.18,
    0.18,
    false,
  );

  // ==========================================================================
  // 2. SPARSE ROUTED CORRIDOR: Upright FP32 W_router + 8x16 3D Expert Stadium
  // ==========================================================================
  const wRouterY = 1.24;
  const wRouterSpanZ = HIDDEN_SPAN; // d = 8 along Z
  const wRouterHeightY = 1.92; // 128 experts vertically
  const wRouterHl = sheetHighlight(
    { tensor: 'w_router', isAggregate: true },
    ['w-router', 'router'],
    [1],
    isFp8Mode ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: xWRouter,
    cy: wRouterY,
    cz: zRouted,
    ex: wRouterSpanZ,
    ey: SHEET_THICK,
    ez: wRouterHeightY,
    axes: FRAME_YZ_AXES,
    gridCols: sim.d,
    gridRows: sim.nRoutedExperts,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0, // Strictly FP32 router weights even under FP8 Selective (NVIDIA §4.2)
    highlight: wRouterHl,
    seed: 31,
    partId: 'w-router',
    tensorKey: 'w_router',
    atlasSlice: atlas.wRouter,
    registryKey: 'w_router',
    cellFamily: CELL_MOE_AMBER,
  });

  // RMSNorm -> Upright W_router (Rear Sparse Fork)
  const routerForkAct = !hasCellTrace && (wRouterHl === SHEET_HL_ACTIVE || step === 1 || step === 2);
  b.flow.addCubic3D(
    [xRms + ioW * 0.5 + 0.04, BASE_Y + 0.05, -0.35],
    [xRms + ioW * 0.5 + 0.45, wRouterY * 0.55, zRouted],
    [xWRouter - 0.52, wRouterY, zRouted],
    [xWRouter - 0.06, wRouterY, zRouted],
    [0, 1, 0],
    routerForkAct ? 0.105 : 0.082,
    routerForkAct ? RIBBON_ROUTER_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_ROUTER_FLOW,
    true,
    18,
    true,
  );

  // 8x16 3D Expert Stadium (128 experts):
  //   - 122 idle experts rest low on ground tray (y = 0.04, partId: 'inactive-experts')
  //   - 6 active Top-6 experts physically rise as paired 2-matrix plates (W_up + W_down, partId: 'routed-bank')
  const bankCols = MOE_SIM_CONFIG.bankCols; // 16 along X
  const bankRows = MOE_SIM_CONFIG.bankRows; // 8 along Z
  const cellStepX = 0.215;
  const cellStepZ = 0.250;
  const bankW = bankCols * cellStepX; // 3.44
  const bankD = bankRows * cellStepZ; // 2.00
  const bankLeft = bankCenterX - bankW * 0.5;
  const bankTop = zRouted - bankD * 0.5;
  const bankTileGeom = new Map<number, SheetGeomRecord>();

  b.addDashedFrameXZ(
    bankLeft - 0.10,
    bankTop - 0.10,
    bankLeft + bankW + 0.10,
    bankTop + bankD + 0.10,
    0.015,
    LINE_STYLE_GUIDE,
    0.44,
  );

  const bankEnvelope: SheetGeomRecord = {
    cx: bankCenterX,
    cy: 0.34,
    cz: zRouted,
    ex: bankW,
    ey: 0.72,
    ez: bankD,
    gridCols: bankCols,
    gridRows: bankRows,
  };

  for (let e = 0; e < sim.nRoutedExperts; e++) {
    const r = Math.floor(e / bankCols);
    const c = e % bankCols;
    const cx = bankLeft + (c + 0.5) * cellStepX;
    const cz = bankTop + (r + 0.5) * cellStepZ;
    const isActiveWinner = activeRankMap.has(e);
    const isFocused = e === focusExpert;
    const tileW = cellStepX - 0.032;
    const tileD = cellStepZ - 0.034;

    const tileCandidate: SheetCandidateIdentity = {
      tensor: 'router_bank',
      expertId: e,
      tokenIndex: focusToken,
      isAggregate: false,
    };

    if (isActiveWinner) {
      const halfD = (tileD - 0.018) * 0.5;
      const fallbackHl =
        activePart === 'routed-bank' || activePart === `moe-expert-${e}` || step === 1 || step === 2
          ? SHEET_HL_ACTIVE
          : hasPartFocus
            ? SHEET_HL_DIMMED
            : isFocused
              ? SHEET_HL_ACTIVE
              : SHEET_HL_PATHWAY;
      const hl = resolveSheetHighlight(focusSet, tileCandidate, fallbackHl);

      const towerY =
        hl === SHEET_HL_ACTIVE
          ? 0.64
          : hl === SHEET_HL_OPERAND_A
            ? 0.60
            : hl === SHEET_HL_OPERAND_B
              ? 0.50
              : hl === SHEET_HL_PATHWAY
                ? 0.44
                : hl === SHEET_HL_DIMMED
                  ? 0.24
                  : 0.44;

      // Grounding pin anchoring each risen active expert directly to its 8x16 bank cell
      const pinActive =
        hl === SHEET_HL_ACTIVE || hl === SHEET_HL_OPERAND_A || hl === SHEET_HL_OPERAND_B;
      b.addGuideLine(
        cx,
        0.02,
        cz,
        cx,
        towerY - 0.035,
        cz,
        pinActive ? 1.6 : 0.95,
        pinActive ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
        pinActive ? 0.85 : 0.42,
      );

      // Paired 2-matrix micro-plates (W_up + W_down, non-gated ReLU²) -> 12 'routed-bank' pickBoxes!
      b.addSheet({
        cx,
        cy: towerY,
        cz: cz - halfD * 0.5 - 0.009,
        ex: tileW,
        ey: 0.064,
        ez: halfD,
        axes: FRAME_XZ_AXES,
        gridCols: 1,
        gridRows: 1,
        role: SHEET_ROLE_SOFTMAX,
        fp8Flag: isFp8Mode ? 1 : 0,
        highlight: hl,
        seed: 100 + e,
        partId: 'routed-bank',
        headIndex: e,
        tensorKey: 'router_bank',
        tokenIndex: focusToken,
        atlasSlice: atlas.routerBankTiles[e],
        cellFamily: CELL_MOE_AMBER,
      });

      b.addSheet({
        cx,
        cy: towerY,
        cz: cz + halfD * 0.5 + 0.009,
        ex: tileW,
        ey: 0.064,
        ez: halfD,
        axes: FRAME_XZ_AXES,
        gridCols: 1,
        gridRows: 1,
        role: SHEET_ROLE_WEIGHT,
        fp8Flag: isFp8Mode ? 1 : 0,
        highlight: hl,
        seed: 250 + e,
        partId: 'routed-bank',
        headIndex: e,
        tensorKey: 'router_bank',
        tokenIndex: focusToken,
        atlasSlice: atlas.routerBankTiles[e],
        cellFamily: CELL_WEIGHT_BLUE,
      });

      const mergedFrame = makeSheetFrame3D(
        [cx, towerY, cz],
        [tileW, 0.064, tileD],
        FRAME_XZ_AXES.axisU,
        FRAME_XZ_AXES.axisN,
      );
      bankTileGeom.set(e, {
        cx,
        cy: towerY,
        cz,
        ex: tileW,
        ey: 0.064,
        ez: tileD,
        gridCols: 1,
        gridRows: 1,
        frame: mergedFrame,
      });
    } else {
      const idleFallbackHl =
        activePart === 'inactive-experts' || activePart === `moe-expert-${e}`
          ? SHEET_HL_ACTIVE
          : !hasPartFocus && isFocused
            ? SHEET_HL_ACTIVE
            : SHEET_HL_DIMMED;
      const idleHl = resolveSheetHighlight(focusSet, tileCandidate, idleFallbackHl);
      const idleY =
        idleHl === SHEET_HL_ACTIVE
          ? 0.22
          : idleHl === SHEET_HL_OPERAND_A || idleHl === SHEET_HL_OPERAND_B
            ? 0.14
            : BASE_Y;

      const idleGeom = b.addSheet({
        cx,
        cy: idleY,
        cz,
        ex: tileW,
        ey: 0.042,
        ez: tileD,
        axes: FRAME_XZ_AXES,
        gridCols: 1,
        gridRows: 1,
        role: SHEET_ROLE_SCORES,
        fp8Flag: isFp8Mode ? 1 : 0,
        highlight: idleHl,
        seed: 400 + e,
        partId: 'inactive-experts',
        headIndex: e,
        tensorKey: 'router_bank',
        tokenIndex: focusToken,
        atlasSlice: atlas.routerBankTiles[e],
        cellFamily: CELL_NEUTRAL,
      });
      bankTileGeom.set(e, idleGeom);
    }
  }

  // W_router -> 8x16 Expert Stadium
  b.flow.addCubic3D(
    [xWRouter + 0.06, wRouterY, zRouted],
    [xWRouter + 0.55, wRouterY * 0.72, zRouted],
    [bankLeft - 0.45, 0.36, zRouted],
    [bankLeft - 0.12, 0.26, zRouted],
    [0, 1, 0],
    routerForkAct ? 0.095 : 0.075,
    routerForkAct ? RIBBON_ROUTER_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_ROUTER_FLOW,
    true,
    16,
    true,
  );

  // ==========================================================================
  // 3. TWO-TIER TOP-6 NORMALIZATION BRIDGE (Bias Stripped -> Unbiased σ -> ×2.5)
  // ==========================================================================
  const bridgeDeckY = 0.62;
  const topkSpanZ = 2.04; // 6 slots along Z: slot 0 at +Z front (-0.30) -> slot 5 at -Z rear (-2.00)
  const slotPitchZ = topkSpanZ / sim.topK; // 0.34
  const slotZ = (j: number): number => zRouted + topkSpanZ * 0.5 - (j + 0.5) * slotPitchZ;

  // Grounding stanchions & architectural frame for the FP32 Top-6 Bridge
  const bridgeFrameStyle = isFp8Mode ? LINE_STYLE_BF16_GOLD : LINE_STYLE_GUIDE;
  b.addBoxFrameXZ(
    xUnbiased - 0.20,
    zRouted - topkSpanZ * 0.5 - 0.06,
    xTopk + 0.22,
    zRouted + topkSpanZ * 0.5 + 0.06,
    0.018,
    isFp8Mode ? 1.25 : 0.9,
    bridgeFrameStyle,
    isFp8Mode ? 0.68 : 0.40,
  );
  b.addGuideLine(xUnbiased, 0.02, zRouted - topkSpanZ * 0.5, xUnbiased, bridgeDeckY - 0.03, zRouted - topkSpanZ * 0.5, 0.85, bridgeFrameStyle, 0.35);
  b.addGuideLine(xUnbiased, 0.02, zRouted + topkSpanZ * 0.5, xUnbiased, bridgeDeckY - 0.03, zRouted + topkSpanZ * 0.5, 0.85, bridgeFrameStyle, 0.35);
  b.addGuideLine(xTopk, 0.02, zRouted - topkSpanZ * 0.5, xTopk, bridgeDeckY - 0.03, zRouted - topkSpanZ * 0.5, 0.85, bridgeFrameStyle, 0.35);
  b.addGuideLine(xTopk, 0.02, zRouted + topkSpanZ * 0.5, xTopk, bridgeDeckY - 0.03, zRouted + topkSpanZ * 0.5, 0.85, bridgeFrameStyle, 0.35);

  // Tier 1: Unbiased Sigmoid Scores [1 x 6] (e_score_correction_bias stripped before normalization!)
  const isInspectingTopk = activeFormula?.cell.tensor === 'topk_weights';
  const unbiasedHl = isInspectingTopk
    ? SHEET_HL_OPERAND_B
    : hasCellTrace
      ? SHEET_HL_DIMMED
      : activePart === 'router-unbiased' || activePart === 'router' || step === 2
        ? SHEET_HL_PATHWAY
        : hasPartFocus
          ? SHEET_HL_DIMMED
          : isFp8Mode
            ? SHEET_HL_BF16_GOLD
            : SHEET_HL_NORMAL;
  b.addSheet({
    cx: xUnbiased,
    cy: bridgeDeckY,
    cz: zRouted,
    ex: topkSpanZ,
    ey: 0.056,
    ez: 0.30,
    axes: FRAME_XZ_COLS_Z,
    gridCols: sim.topK,
    gridRows: 1,
    role: SHEET_ROLE_SCORES,
    fp8Flag: 0, // Strictly FP32
    highlight: unbiasedHl,
    seed: 39,
    partId: 'router-unbiased',
    tokenIndex: focusToken,
    atlasSlice: atlas.topkUnbiased,
    registryKey: 'topk_unbiased',
    cellFamily: CELL_MOE_AMBER,
  });

  // Tier 2: Normalized ×2.5 Top-6 Weights [1 x 6]
  const topkHl = sheetHighlight(
    { tensor: 'topk_weights', tokenIndex: focusToken, isAggregate: true },
    ['router', 'topk-weights'],
    [2],
    isFp8Mode ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: xTopk,
    cy: topkHl === SHEET_HL_ACTIVE ? bridgeDeckY + 0.035 : bridgeDeckY,
    cz: zRouted,
    ex: topkSpanZ,
    ey: SHEET_THICK,
    ez: 0.36,
    axes: FRAME_XZ_COLS_Z,
    gridCols: sim.topK,
    gridRows: 1,
    role: SHEET_ROLE_SOFTMAX,
    fp8Flag: 0, // Strictly FP32 router weights
    highlight: topkHl,
    seed: 41,
    partId: 'router',
    tensorKey: 'topk_weights',
    tokenIndex: focusToken,
    atlasSlice: atlas.topkWeights,
    registryKey: 'topk_weights',
    cellFamily: CELL_MOE_AMBER,
  });

  // 6 arched 3D ribbons from the 6 risen winning stadium tiles -> Unbiased σ slots -> Top-6 Weights
  for (let j = 0; j < sim.topK; j++) {
    const eid = activeExpertIds[j];
    const tile = bankTileGeom.get(eid);
    if (!tile) continue;
    const sz = slotZ(j);
    const isSlotFocused = !hasCellTrace && (eid === focusExpert || step === 2);
    b.flow.addCubic3D(
      [tile.cx + tile.ex * 0.5 + 0.02, tile.cy + 0.03, tile.cz],
      [tile.cx + 0.60, Math.max(tile.cy, bridgeDeckY) + 0.28, tile.cz * 0.55 + sz * 0.45],
      [xUnbiased - 0.52, bridgeDeckY + 0.16, sz],
      [xUnbiased - 0.17, bridgeDeckY + 0.03, sz],
      [0, 1, 0],
      isSlotFocused ? 0.056 : 0.036,
      isSlotFocused ? RIBBON_ROUTER_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_ROUTER_FLOW,
      true,
      18,
      true,
    );
    b.flow.addPolyline3D(
      [
        [xUnbiased + 0.16, bridgeDeckY + 0.03, sz],
        [xTopk - 0.20, bridgeDeckY + 0.03, sz],
      ],
      [0, 1, 0],
      isSlotFocused ? 0.048 : 0.032,
      isSlotFocused ? RIBBON_ROUTER_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
      0.08,
      0.08,
      false,
    );
  }

  // ==========================================================================
  // 4. FOCUSED ROUTED EXPERT WORKBENCH (Magnified 1 of 6: Non-Gated ReLU², 2 Matrices)
  // ==========================================================================
  const detailIdleHl = focusedEval.isActive ? SHEET_HL_NORMAL : SHEET_HL_DIMMED;
  const expertPartMatch = activePart === `moe-expert-${focusExpert}`;
  const wUpHl = sheetHighlight(
    { tensor: 'routed_w_up', expertId: focusExpert, tokenIndex: focusToken, isAggregate: false },
    expertPartMatch ? ['routed-w-up', activePart!] : ['routed-w-up'],
    [3],
    detailIdleHl,
  );
  const preHl = sheetHighlight(
    { tensor: 'routed_pre', expertId: focusExpert, tokenIndex: focusToken, isAggregate: false },
    expertPartMatch ? ['routed-pre', activePart!] : ['routed-pre'],
    [3],
    detailIdleHl,
  );
  const actHl = sheetHighlight(
    { tensor: 'routed_act', expertId: focusExpert, tokenIndex: focusToken, isAggregate: false },
    expertPartMatch ? ['routed-act', activePart!] : ['routed-act'],
    [3],
    detailIdleHl,
  );
  const wDownHl = sheetHighlight(
    { tensor: 'routed_w_down', expertId: focusExpert, tokenIndex: focusToken, isAggregate: false },
    expertPartMatch ? ['routed-w-down', activePart!] : ['routed-w-down'],
    [3],
    detailIdleHl,
  );

  // Grounded workbench frame highlighting that this is a magnified cutout of 1 of the 6 branches
  const wbX0 = xRoutedWUp - 0.34;
  const wbX1 = xRoutedWDown + 0.34;
  const wbZ0 = zRouted - HIDDEN_SPAN * 0.5 - 0.16;
  const wbZ1 = zRouted + HIDDEN_SPAN * 0.5 + 0.16;
  b.addDashedFrameXZ(
    wbX0,
    wbZ0,
    wbX1,
    wbZ1,
    0.018,
    focusedEval.isActive ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
    focusedEval.isActive ? 0.52 : 0.34,
  );

  // Upright Routed W_up Wall [d_exp=4, d=8] (1.0x hidden x 1.0x intermediate)
  const routedWUpY = 0.86;
  b.addSheet({
    cx: xRoutedWUp,
    cy: routedWUpY,
    cz: zRouted,
    ex: HIDDEN_SPAN, // 1.10 (1.0x hidden)
    ey: SHEET_THICK,
    ez: ROUTED_INTER_SPAN, // 0.62 (1.0x intermediate)
    axes: FRAME_YZ_AXES,
    gridCols: sim.d,
    gridRows: sim.dExp,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: isFp8Mode ? 1 : 0,
    highlight: wUpHl,
    seed: 51,
    partId: 'routed-w-up',
    headIndex: focusExpert,
    tensorKey: 'routed_w_up',
    tokenIndex: focusToken,
    atlasSlice: atlas.routedWUp,
    registryKey: 'routed_w_up',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Horizontal Pre-Activation h [4 x 1]
  b.addSheet({
    cx: xRoutedPre,
    cy: BASE_Y + 0.04,
    cz: zRouted,
    ex: 0.34,
    ey: SHEET_THICK,
    ez: ROUTED_INTER_SPAN, // 0.62 (1.0x intermediate)
    axes: FRAME_XZ_AXES,
    gridCols: 1,
    gridRows: sim.dExp,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: preHl,
    seed: 53,
    partId: 'routed-pre',
    headIndex: focusExpert,
    tensorKey: 'routed_pre',
    tokenIndex: focusToken,
    atlasSlice: atlas.routedPre,
    registryKey: 'routed_pre',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Horizontal ReLU² Activation [4 x 1] (Non-Gated — NO gate_proj!)
  b.addSheet({
    cx: xRoutedAct,
    cy: BASE_Y + 0.04,
    cz: zRouted,
    ex: 0.34,
    ey: SHEET_THICK,
    ez: ROUTED_INTER_SPAN, // 0.62 (1.0x intermediate)
    axes: FRAME_XZ_AXES,
    gridCols: 1,
    gridRows: sim.dExp,
    role: SHEET_ROLE_SOFTMAX,
    fp8Flag: 0,
    highlight: actHl,
    seed: 57,
    partId: 'routed-act',
    headIndex: focusExpert,
    tensorKey: 'routed_act',
    tokenIndex: focusToken,
    atlasSlice: atlas.routedAct,
    registryKey: 'routed_act',
    cellFamily: CELL_MOE_AMBER,
  });

  // Upright Routed W_down Wall [d=8, d_exp=4] (1.0x intermediate x 1.0x hidden)
  const routedWDownY = 1.06;
  b.addSheet({
    cx: xRoutedWDown,
    cy: routedWDownY,
    cz: zRouted,
    ex: ROUTED_INTER_SPAN, // 0.62 (1.0x intermediate)
    ey: SHEET_THICK,
    ez: HIDDEN_SPAN, // 1.10 (1.0x hidden)
    axes: FRAME_YZ_AXES,
    gridCols: sim.dExp,
    gridRows: sim.d,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: isFp8Mode ? 1 : 0,
    highlight: wDownHl,
    seed: 59,
    partId: 'routed-w-down',
    headIndex: focusExpert,
    tensorKey: 'routed_w_down',
    tokenIndex: focusToken,
    atlasSlice: atlas.routedWDown,
    registryKey: 'routed_w_down',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Internal workbench flow ribbons: W_up -> Pre-Act h -> ReLU² -> W_down
  const chainCol =
    hasCellTrace || !focusedEval.isActive
      ? RIBBON_DIMMED
      : step === 3 ||
          wUpHl === SHEET_HL_ACTIVE ||
          preHl === SHEET_HL_ACTIVE ||
          actHl === SHEET_HL_ACTIVE ||
          wDownHl === SHEET_HL_ACTIVE
        ? RIBBON_ACTIVE
        : hasPartFocus
          ? RIBBON_DIMMED
          : RIBBON_PATHWAY_SOFT;
  b.flow.addCubic3D(
    [xRoutedWUp + 0.06, routedWUpY, zRouted],
    [xRoutedWUp + 0.28, routedWUpY * 0.65, zRouted],
    [xRoutedPre - 0.34, BASE_Y + 0.16, zRouted],
    [xRoutedPre - 0.19, BASE_Y + 0.08, zRouted],
    [0, 1, 0],
    0.072,
    chainCol,
    true,
    14,
    true,
  );
  b.flow.addPolyline3D(
    [
      [xRoutedPre + 0.19, BASE_Y + 0.08, zRouted],
      [xRoutedAct - 0.19, BASE_Y + 0.08, zRouted],
    ],
    [0, 1, 0],
    0.072,
    chainCol,
    true,
    0.10,
    0.10,
    false,
  );
  b.flow.addCubic3D(
    [xRoutedAct + 0.19, BASE_Y + 0.08, zRouted],
    [xRoutedAct + 0.34, BASE_Y + 0.24, zRouted],
    [xRoutedWDown - 0.28, routedWDownY * 0.72, zRouted],
    [xRoutedWDown - 0.06, routedWDownY, zRouted],
    [0, 1, 0],
    0.072,
    chainCol,
    true,
    14,
    true,
  );

  // ==========================================================================
  // 5. SIX INDEPENDENT WEIGHTED-OUTPUT BRANCHES (w_0·y_0 .. w_5·y_5) -> Routed Sum
  //    Refinement 1: All 6 active experts visibly execute & converge; workbench magnifies 1 of 6!
  // ==========================================================================
  const branchW = 0.66;
  const branchD = 0.20;
  const branchGeom = new Map<number, SheetGeomRecord>();
  const isInspectingRoutedSum = activeFormula?.cell.tensor === 'routed_sum';

  for (let j = 0; j < sim.topK; j++) {
    const eid = activeExpertIds[j];
    const bz = slotZ(j);
    const isFocusedBranch = eid === focusExpert;

    let bHl: number;
    if (isInspectingRoutedSum) {
      bHl = isFocusedBranch ? SHEET_HL_OPERAND_A : SHEET_HL_OPERAND_B;
    } else if (hasCellTrace) {
      const cellIsFocusedRouted =
        activeFormula?.activeExpertId === eid &&
        (activeFormula.cell.tensor === 'routed_w_up' ||
          activeFormula.cell.tensor === 'routed_pre' ||
          activeFormula.cell.tensor === 'routed_act' ||
          activeFormula.cell.tensor === 'routed_w_down');
      bHl = cellIsFocusedRouted ? SHEET_HL_PATHWAY : SHEET_HL_DIMMED;
    } else if (activePart === 'routed-branch' || activePart === `moe-expert-${eid}`) {
      bHl = SHEET_HL_ACTIVE;
    } else if (step === 3 || activePart === 'routed-sum') {
      bHl = isFocusedBranch ? SHEET_HL_ACTIVE : SHEET_HL_PATHWAY;
    } else if (hasPartFocus) {
      bHl = SHEET_HL_DIMMED;
    } else {
      bHl = isFocusedBranch ? SHEET_HL_ACTIVE : SHEET_HL_NORMAL;
    }

    const by =
      bHl === SHEET_HL_ACTIVE || bHl === SHEET_HL_OPERAND_A
        ? BASE_Y + 0.08
        : bHl === SHEET_HL_OPERAND_B || bHl === SHEET_HL_PATHWAY
          ? BASE_Y + 0.05
          : BASE_Y + 0.02;

    const rec = b.addSheet({
      cx: xBranches,
      cy: by,
      cz: bz,
      ex: branchW,
      ey: 0.048,
      ez: branchD,
      axes: FRAME_XZ_AXES,
      gridCols: sim.d,
      gridRows: 1,
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: bHl,
      seed: 61 + j * 0.7 + focusToken * 0.3,
      partId: 'routed-branch',
      headIndex: eid,
      tokenIndex: focusToken,
      atlasSlice: atlas.activeWeightedOut[j],
      registryKey: `routed_branch:${j}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });
    branchGeom.set(j, rec);

    // Connect Top-6 slot j -> branch strip j:
    //   - Focused active branch routes THROUGH the magnified workbench!
    //   - Other 5 active branches arch cleanly over/beside the workbench as compact independent pipes!
    if (isFocusedBranch) {
      // Top-6[j*] -> Workbench W_up
      b.flow.addCubic3D(
        [xTopk + 0.20, bridgeDeckY + 0.02, bz],
        [xTopk + 0.60, bridgeDeckY + 0.16, bz * 0.5 + zRouted * 0.5],
        [xRoutedWUp - 0.52, routedWUpY, zRouted],
        [xRoutedWUp - 0.06, routedWUpY, zRouted],
        [0, 1, 0],
        0.064,
        !hasCellTrace ? RIBBON_BRANCH_FOCUS : RIBBON_DIMMED,
        true,
        16,
        true,
      );
      // Workbench W_down -> Focused branch strip w_{j*}·y_{j*}
      b.flow.addCubic3D(
        [xRoutedWDown + 0.06, routedWDownY, zRouted],
        [xRoutedWDown + 0.42, routedWDownY * 0.65, zRouted * 0.5 + bz * 0.5],
        [xBranches - branchW * 0.5 - 0.32, by + 0.12, bz],
        [xBranches - branchW * 0.5 - 0.04, by + 0.03, bz],
        [0, 1, 0],
        0.064,
        !hasCellTrace ? RIBBON_BRANCH_FOCUS : RIBBON_DIMMED,
        true,
        16,
        true,
      );
      // Crisp magnification callout leaders connecting workbench corners to focused branch strip
      b.addGuideLine(wbX1, 0.02, wbZ0, xBranches - branchW * 0.5, by, bz - branchD * 0.5, 0.9, LINE_STYLE_ACTIVE, 0.45);
      b.addGuideLine(wbX1, 0.02, wbZ1, xBranches - branchW * 0.5, by, bz + branchD * 0.5, 0.9, LINE_STYLE_ACTIVE, 0.45);
    } else {
      // Compact arched independent execution pipe for each of the other 5 active branches
      const arcY = 1.92 + (j % 3) * 0.11;
      const pipeCol = isInspectingRoutedSum
        ? THREAD_OP_B
        : !hasAnyFocus || step === 3 || activePart === 'routed-sum'
          ? RIBBON_BRANCH_QUIET
          : RIBBON_DIMMED;
      b.flow.addCubic3D(
        [xTopk + 0.20, bridgeDeckY + 0.03, bz],
        [xRoutedWUp - 0.15, arcY, bz],
        [xRoutedWDown + 0.25, arcY, bz],
        [xBranches - branchW * 0.5 - 0.04, by + 0.03, bz],
        [0, 1, 0],
        isInspectingRoutedSum ? 0.036 : 0.028,
        pipeCol,
        true,
        20,
        true,
      );
    }
  }

  // Routed Sum (Σ_{j=1..6} w_j·y_j)
  const routedSumHl = sheetHighlight({ tensor: 'routed_sum', isAggregate: true }, ['routed-sum'], [3, 5]);
  addGhostSeqRail(xBranchOut, BASE_Y, zRoutedSum, ioW, ioMaxZ, 0.4, routedSumHl);
  b.addSheet({
    cx: xBranchOut,
    cy: BASE_Y,
    cz: zAnchoredCenter(zRoutedSum, ioMaxZ, ioLiveZ),
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: routedSumHl,
    seed: 67,
    partId: 'routed-sum',
    tensorKey: 'routed_sum',
    atlasSlice: atlas.routedSum,
    registryKey: 'routed_sum',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // All 6 weighted-output branch strips converge visibly into Routed Sum!
  for (let j = 0; j < sim.topK; j++) {
    const rec = branchGeom.get(j);
    if (!rec) continue;
    const isFoc = activeExpertIds[j] === focusExpert;
    const convCol = isInspectingRoutedSum
      ? isFoc
        ? THREAD_OP_A
        : THREAD_OP_B
      : !hasCellTrace && (routedSumHl === SHEET_HL_ACTIVE || step === 3)
        ? isFoc
          ? RIBBON_BRANCH_FOCUS
          : RIBBON_PATHWAY_SOFT
        : hasAnyFocus
          ? RIBBON_DIMMED
          : isFoc
            ? RIBBON_PATHWAY_SOFT
            : RIBBON_BRANCH_QUIET;
    b.flow.addCubic3D(
      [xBranches + branchW * 0.5 + 0.03, rec.cy + 0.03, rec.cz],
      [xBranches + branchW * 0.5 + 0.28, rec.cy + 0.06, rec.cz],
      [xBranchOut - ioW * 0.5 - 0.30, BASE_Y + 0.06, zRoutedSum],
      [xBranchOut - ioW * 0.5 - 0.04, BASE_Y + 0.04, zRoutedSum],
      [0, 1, 0],
      isFoc || isInspectingRoutedSum ? 0.046 : 0.034,
      convCol,
      true,
      14,
      true,
    );
  }

  // ==========================================================================
  // 6. FRONT PARALLEL CORRIDOR: ALWAYS-ON DOUBLE-WIDTH SHARED EXPERT (3,712 = 2x, 1.0x UNSCALED)
  //    Strictly 2.0x intermediate width (1.24 vs 0.62) and 1.0x hidden width (1.10 vs 1.10)!
  // ==========================================================================
  const xSharedUp = -1.65;
  const xSharedAct = 0.90;
  const xSharedDown = 3.35;

  const sharedUpHl = sheetHighlight({ tensor: 'shared_w_up', isAggregate: true }, ['shared-expert'], [4]);
  const sharedActHl = sheetHighlight(
    { tensor: 'shared_act', tokenIndex: focusToken, isAggregate: false },
    ['shared-act', 'shared-expert'],
    [4],
  );
  const sharedDownHl = sheetHighlight(
    { tensor: 'shared_w_down', isAggregate: true },
    ['shared-w-down', 'shared-expert'],
    [4],
  );
  const sharedOutHl = sheetHighlight(
    { tensor: 'shared_out', isAggregate: true },
    ['shared-out', 'shared-expert'],
    [4, 5],
  );

  // Dashed corridor enclosing the always-on 2x-intermediate parallel Shared Expert
  b.addDashedFrameXZ(
    xSharedUp - 0.45,
    zShared - SHARED_INTER_SPAN * 0.5 - 0.16,
    xSharedDown + 0.45,
    zShared + SHARED_INTER_SPAN * 0.5 + 0.16,
    0.018,
    LINE_STYLE_GUIDE,
    0.46,
  );

  // Upright Shared W_up Wall [d_shared=8, d=8] -> EXACT 1.0x hidden (1.10) x 2.0x intermediate (1.24)!
  const sharedWUpY = 1.17;
  b.addSheet({
    cx: xSharedUp,
    cy: sharedWUpY,
    cz: zShared,
    ex: HIDDEN_SPAN, // 1.10 (EXACT 1.0x hidden!)
    ey: SHEET_THICK,
    ez: SHARED_INTER_SPAN, // 1.24 (EXACT 2.0x intermediate!)
    axes: FRAME_YZ_AXES,
    gridCols: sim.d,
    gridRows: sim.dShared,
    role: SHEET_ROLE_KV_K,
    fp8Flag: isFp8Mode ? 1 : 0,
    highlight: sharedUpHl,
    seed: 71,
    partId: 'shared-expert',
    tensorKey: 'shared_w_up',
    atlasSlice: atlas.sharedWUp,
    registryKey: 'shared_w_up',
    cellFamily: CELL_SSM_CYAN,
  });

  // Horizontal Shared ReLU² [8 x 1] -> EXACT 2.0x intermediate (1.24)!
  b.addSheet({
    cx: xSharedAct,
    cy: BASE_Y + 0.04,
    cz: zShared,
    ex: 0.38,
    ey: SHEET_THICK,
    ez: SHARED_INTER_SPAN, // 1.24 (EXACT 2.0x intermediate!)
    axes: FRAME_XZ_AXES,
    gridCols: 1,
    gridRows: sim.dShared,
    role: SHEET_ROLE_KV_V,
    fp8Flag: 0,
    highlight: sharedActHl,
    seed: 73,
    partId: 'shared-act',
    tensorKey: 'shared_act',
    tokenIndex: focusToken,
    atlasSlice: atlas.sharedAct,
    registryKey: 'shared_act',
    cellFamily: CELL_SSM_CYAN,
  });

  // Upright Shared W_down Wall [d=8, d_shared=8] -> EXACT 2.0x intermediate (1.24) x 1.0x hidden (1.10)!
  const sharedWDownY = 1.06;
  b.addSheet({
    cx: xSharedDown,
    cy: sharedWDownY,
    cz: zShared,
    ex: SHARED_INTER_SPAN, // 1.24 (EXACT 2.0x intermediate!)
    ey: SHEET_THICK,
    ez: HIDDEN_SPAN, // 1.10 (EXACT 1.0x hidden!)
    axes: FRAME_YZ_AXES,
    gridCols: sim.dShared,
    gridRows: sim.d,
    role: SHEET_ROLE_KV_K,
    fp8Flag: isFp8Mode ? 1 : 0,
    highlight: sharedDownHl,
    seed: 79,
    partId: 'shared-w-down',
    tensorKey: 'shared_w_down',
    atlasSlice: atlas.sharedWDown,
    registryKey: 'shared_w_down',
    cellFamily: CELL_SSM_CYAN,
  });

  // Shared Out (Unscaled 1.0x)
  addGhostSeqRail(xBranchOut, BASE_Y, zSharedOut, ioW, ioMaxZ, 0.5, sharedOutHl);
  b.addSheet({
    cx: xBranchOut,
    cy: BASE_Y,
    cz: zAnchoredCenter(zSharedOut, ioMaxZ, ioLiveZ),
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_KV_V,
    fp8Flag: 0,
    highlight: sharedOutHl,
    seed: 83,
    partId: 'shared-out',
    tensorKey: 'shared_out',
    atlasSlice: atlas.sharedOut,
    registryKey: 'shared_out',
    cellFamily: CELL_SSM_CYAN,
  });

  // Parallel Shared Expert flow ribbons: RMSNorm -> Shared W_up -> Shared ReLU² -> Shared W_down -> Shared Out
  const sharedActFlag =
    !hasCellTrace &&
    (step === 4 ||
      sharedUpHl === SHEET_HL_ACTIVE ||
      sharedActHl === SHEET_HL_ACTIVE ||
      sharedDownHl === SHEET_HL_ACTIVE ||
      sharedOutHl === SHEET_HL_ACTIVE);
  const sharedFlowCol = sharedActFlag
    ? RIBBON_SHARED_ACTIVE
    : hasAnyFocus
      ? RIBBON_DIMMED
      : RIBBON_SHARED_FLOW;

  b.flow.addCubic3D(
    [xRms + ioW * 0.5 + 0.04, BASE_Y + 0.05, 0.45],
    [xRms + 1.40, BASE_Y + 0.25, zShared],
    [xSharedUp - 1.35, sharedWUpY, zShared],
    [xSharedUp - 0.06, sharedWUpY, zShared],
    [0, 1, 0],
    sharedActFlag ? 0.105 : 0.082,
    sharedFlowCol,
    true,
    22,
    true,
  );
  b.flow.addCubic3D(
    [xSharedUp + 0.06, sharedWUpY, zShared],
    [xSharedUp + 0.72, sharedWUpY * 0.65, zShared],
    [xSharedAct - 0.65, BASE_Y + 0.18, zShared],
    [xSharedAct - 0.22, BASE_Y + 0.08, zShared],
    [0, 1, 0],
    0.080,
    sharedFlowCol,
    true,
    18,
    true,
  );
  b.flow.addCubic3D(
    [xSharedAct + 0.22, BASE_Y + 0.08, zShared],
    [xSharedAct + 0.68, BASE_Y + 0.22, zShared],
    [xSharedDown - 0.72, sharedWDownY * 0.72, zShared],
    [xSharedDown - 0.06, sharedWDownY, zShared],
    [0, 1, 0],
    0.080,
    sharedFlowCol,
    true,
    18,
    true,
  );
  b.flow.addCubic3D(
    [xSharedDown + 0.06, sharedWDownY, zShared],
    [xSharedDown + 0.85, sharedWDownY * 0.62, zShared],
    [xBranchOut - ioW * 0.5 - 0.75, BASE_Y + 0.14, zSharedOut],
    [xBranchOut - ioW * 0.5 - 0.05, BASE_Y + 0.05, zSharedOut],
    [0, 1, 0],
    0.082,
    sharedFlowCol,
    true,
    20,
    true,
  );

  // ==========================================================================
  // 7. PARALLEL MERGE: Routed Sum + Shared Out -> MoE Out -> + Residual Y
  // ==========================================================================
  const moeOutHl = sheetHighlight(
    { tensor: 'moe_out', isAggregate: true },
    ['moe-out', 'merge-sum'],
    [5],
  );
  addGhostSeqRail(xMoeOut, BASE_Y, 0, ioW, ioMaxZ, 0.6, moeOutHl);
  b.addSheet({
    cx: xMoeOut,
    cy: BASE_Y,
    cz: zAnchoredCenter(0, ioMaxZ, ioLiveZ),
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: moeOutHl,
    seed: 89,
    partId: 'moe-out',
    tensorKey: 'moe_out',
    atlasSlice: atlas.moeOut,
    registryKey: 'moe_out',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  const mergeAct = !hasCellTrace && (moeOutHl === SHEET_HL_ACTIVE || resOutHl === SHEET_HL_ACTIVE || step === 5);
  b.flow.addCubic3D(
    [xBranchOut + ioW * 0.5 + 0.04, BASE_Y + 0.04, zRoutedSum],
    [xBranchOut + ioW * 0.5 + 0.35, BASE_Y + 0.06, zRoutedSum],
    [xMoeOut - ioW * 0.5 - 0.35, BASE_Y + 0.06, -0.25],
    [xMoeOut - ioW * 0.5 - 0.05, BASE_Y + 0.04, -0.20],
    [0, 1, 0],
    0.090,
    mergeAct ? RIBBON_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_PATHWAY_SOFT,
    true,
    14,
    true,
  );
  b.flow.addCubic3D(
    [xBranchOut + ioW * 0.5 + 0.04, BASE_Y + 0.04, zSharedOut],
    [xBranchOut + ioW * 0.5 + 0.40, BASE_Y + 0.06, zSharedOut],
    [xMoeOut - ioW * 0.5 - 0.35, BASE_Y + 0.06, 0.35],
    [xMoeOut - ioW * 0.5 - 0.05, BASE_Y + 0.04, 0.25],
    [0, 1, 0],
    0.090,
    mergeAct ? RIBBON_SHARED_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_SSM_LOOP,
    true,
    16,
    true,
  );
  b.flow.addPolyline3D(
    [
      [xMoeOut + ioW * 0.5 + 0.04, BASE_Y + 0.04, 0],
      [xResOut - ioW * 0.5 - 0.05, BASE_Y + 0.04, 0],
    ],
    [0, 1, 0],
    0.105,
    mergeAct ? RIBBON_RESIDUAL_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_RESIDUAL,
    true,
    0.18,
    0.18,
    false,
  );

  // ==========================================================================
  // 8. LIVE 3D CELL DEPENDENCY THREADS, TOP-6 DENOMINATOR VISUALS & RESIDUAL CARRY-IN
  // ==========================================================================
  // When idle (no cell trace), subtly highlight focused token row on Input / RMSNorm / Routed Sum / Shared Out
  if (!activeFormula) {
    for (const seqKey of ['input', 'rmsnorm', 'routed_sum', 'shared_out'] as const) {
      const rec = b.sheetRegistry.get(seqKey);
      if (!rec) continue;
      const rowSpan = spanFromSheetGrid(rec, focusToken, focusToken, 0, rec.gridCols - 1);
      drawCellHighlight3D(b.flow, rowSpan, TOKEN_ROW_FILL, TOKEN_ROW_FRAME, 0.016, true);
    }
  }

  let residualCarryInfo: { readonly ch: number; readonly tok: number; readonly val: number } | null = null;

  if (activeFormula) {
    const c = activeFormula.cell;
    const targetSpan = resolveSpatialMoeSpan(
      b.sheetRegistry,
      bankTileGeom,
      {
        tensor: c.tensor,
        rowStart: c.row,
        rowEnd: c.row,
        colStart: c.col,
        colEnd: c.col,
        expertId: c.expertId,
        tokenIndex: c.tokenIndex,
      },
      focusToken,
      focusExpert,
    );

    if (targetSpan) {
      drawCellHighlight3D(b.flow, targetSpan, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.024, true);

      // Refinement 2: Special high-clarity visualization for Top-6 Normalization (biased selection vs unbiased σ & 6-term denom!)
      if (c.tensor === 'topk_weights') {
        const slot = Math.max(0, Math.min(sim.topK - 1, c.rankSlot ?? c.col));
        const winnerId = sim.topk_indices[focusToken * sim.topK + slot];
        const unbiasedRec = b.sheetRegistry.get('topk_unbiased');

        if (unbiasedRec) {
          // 1. Teal denominator highlight spanning ALL 6 unbiased sigmoid scores [0..5] (explicitly including slot j itself!)
          const fullDenomSpan = spanFromSheetGrid(unbiasedRec, 0, 0, 0, sim.topK - 1);
          drawCellHighlight3D(b.flow, fullDenomSpan, CELL_OP_B_FILL, CELL_OP_B_FRAME, 0.020, true);

          // 2. Indigo numerator cell highlight on slot j itself (stacked clearly inside the 6-slot denominator!)
          const numSlotSpan = spanFromSheetGrid(unbiasedRec, 0, 0, slot, slot);
          drawCellHighlight3D(b.flow, numSlotSpan, CELL_OP_A_FILL, CELL_OP_A_FRAME, 0.025, true);

          // 3. Bank tiles -> Unbiased σ strip (indigo for numerator expert, teal for other 5 denominator contributors)
          for (let m = 0; m < sim.topK; m++) {
            const eid = activeExpertIds[m];
            const tile = bankTileGeom.get(eid);
            if (!tile) continue;
            const tileSpan = spanFromSheetGrid(tile, 0, 0, 0, 0);
            const mSlotSpan = spanFromSheetGrid(unbiasedRec, 0, 0, m, m);
            const isNumerator = eid === winnerId;
            drawCellHighlight3D(
              b.flow,
              tileSpan,
              isNumerator ? CELL_OP_A_FILL : CELL_OP_B_FILL,
              isNumerator ? CELL_OP_A_FRAME : CELL_OP_B_FRAME,
              isNumerator ? 0.024 : 0.018,
              true,
            );
            drawOperandDependency3D(b.flow, tileSpan, mSlotSpan, isNumerator);
          }

          // 4. Primary indigo numerator thread (σ_j -> w_j) + converging teal denominator arc (Σ_{m=0..5} σ_m -> w_j)
          drawOperandDependency3D(b.flow, numSlotSpan, targetSpan, true);
          b.flow.addCubic3D(
            fullDenomSpan.frontPoint,
            [
              (fullDenomSpan.frontPoint[0] + targetSpan.frontPoint[0]) * 0.5,
              bridgeDeckY + 0.34,
              (fullDenomSpan.frontPoint[2] + targetSpan.frontPoint[2]) * 0.5,
            ],
            [targetSpan.frontPoint[0] - 0.14, bridgeDeckY + 0.22, targetSpan.frontPoint[2]],
            targetSpan.frontPoint,
            [0, 1, 0],
            0.042,
            THREAD_OP_B,
            true,
            16,
            true,
          );
        }
      } else {
        for (let idx = 0; idx < activeFormula.upstreamSpans.length; idx++) {
          const up = activeFormula.upstreamSpans[idx];
          const upSpan = resolveSpatialMoeSpan(
            b.sheetRegistry,
            bankTileGeom,
            up,
            focusToken,
            focusExpert,
          );
          if (!upSpan) continue;

          // Step 6: Route Input X -> + Residual Y along overhead emerald Residual Highway with carry-in badge!
          if (c.tensor === 'residual' && up.tensor === 'input') {
            drawCellHighlight3D(b.flow, upSpan, CELL_OP_A_FILL, CELL_OP_A_FRAME, 0.022, true);
            const threadBridgeY = bridgeY + 0.09;
            b.flow.addCubic3D(
              upSpan.frontPoint,
              [xInput, threadBridgeY * 0.86, -ioMaxZ * 0.5 - 0.78],
              [xInput + 0.35, threadBridgeY, bridgeZ],
              [xInput + 0.95, threadBridgeY, bridgeZ],
              [0, 1, 0],
              0.056,
              THREAD_OP_A,
              false,
              16,
              true,
            );
            b.flow.addPolyline3D(
              [
                [xInput + 0.90, threadBridgeY, bridgeZ],
                [xResOut - 0.90, threadBridgeY, bridgeZ],
              ],
              [0, 1, 0],
              0.056,
              THREAD_OP_A,
              false,
              0.18,
              0.18,
              true,
            );
            b.flow.addCubic3D(
              [xResOut - 0.95, threadBridgeY, bridgeZ],
              [xResOut - 0.25, threadBridgeY, bridgeZ],
              [targetSpan.frontPoint[0], threadBridgeY * 0.76, -ioMaxZ * 0.5 - 0.55],
              targetSpan.frontPoint,
              [0, 1, 0],
              0.056,
              THREAD_OP_A,
              true,
              18,
              true,
              0.16,
              0.16,
            );
            const tok = Math.max(0, Math.min(clampedT - 1, c.row));
            const ch = Math.max(0, Math.min(sim.d - 1, c.col));
            residualCarryInfo = { ch, tok, val: sim.X[ch * clampedT + tok] };
            continue;
          }

          drawOperandDependency3D(b.flow, upSpan, targetSpan, idx === 0);
        }

        // Refinement 1: When inspecting routed_sum, also highlight all 6 weighted-output branch cells for channel ch!
        if (c.tensor === 'routed_sum') {
          const ch = Math.max(0, Math.min(sim.d - 1, c.col));
          for (let j = 0; j < sim.topK; j++) {
            const bRec = branchGeom.get(j);
            if (!bRec) continue;
            const bSpan = spanFromSheetGrid(bRec, 0, 0, ch, ch);
            const isFoc = activeExpertIds[j] === focusExpert;
            drawCellHighlight3D(
              b.flow,
              bSpan,
              isFoc ? CELL_OP_A_FILL : CELL_OP_B_FILL,
              isFoc ? CELL_OP_A_FRAME : CELL_OP_B_FRAME,
              0.020,
              true,
            );
          }
        }
      }
    }
  }

  // ==========================================================================
  // 9. 3D EXTERIOR LABELS & ARCHITECTURAL DIMENSION CALIPERS
  // ==========================================================================
  b.landmarks.push(
    {
      id: 'moe-input',
      label: `Input (2,688 · T=${clampedT})`,
      world: [xInput, 0.16, ioMaxZ * 0.5 + 0.24],
      partId: 'residual-in',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'moe-rms',
      label: 'RMSNorm',
      world: [xRms, 0.16, ioMaxZ * 0.5 + 0.24],
      partId: 'rmsnorm',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'moe-wrouter',
      label: isFp8Mode ? 'W_router · FP32' : 'W_router (128×2,688)',
      world: [xWRouter, wRouterY + wRouterHeightY * 0.5 + 0.22, zRouted],
      partId: 'w-router',
      align: 'top',
      tier: 'primary',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    {
      id: 'moe-bank',
      label: `Router Bank (128 · σ+bias → Top-6 · t${absFocusToken} "${focusTokLabel}")`,
      world: [bankCenterX, 0.92, zRouted - bankD * 0.5 - 0.18],
      partId: 'routed-bank',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-topk',
      label: isFp8Mode ? 'Unbiased σ → Top-6 (×2.5 · FP32)' : 'Unbiased σ → Top-6 (×2.5)',
      world: [(xUnbiased + xTopk) * 0.5, bridgeDeckY + 0.44, zRouted + topkSpanZ * 0.5 + 0.16],
      partId: 'router',
      align: 'top',
      tier: 'primary',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    {
      id: 'moe-routed-up',
      label: isFp8Mode
        ? `${focusExpTag} W_up (1,856) · FP8`
        : `${focusExpTag} W_up (1,856)`,
      world: [xRoutedWUp, routedWUpY + ROUTED_INTER_SPAN * 0.5 + 0.24, zRouted],
      partId: 'routed-w-up',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-routed-act',
      label: 'ReLU² (No gate_proj)',
      world: [(xRoutedPre + xRoutedAct) * 0.5, 0.18, zRouted + ROUTED_INTER_SPAN * 0.5 + 0.22],
      partId: 'routed-act',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'moe-routed-down',
      label: isFp8Mode ? 'W_down · FP8' : 'W_down (2,688)',
      world: [xRoutedWDown, routedWDownY + HIDDEN_SPAN * 0.5 + 0.22, zRouted],
      partId: 'routed-w-down',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-branches',
      label: '6× Active (wᵢ·yᵢ)',
      world: [xBranches, 0.26, zRouted - topkSpanZ * 0.5 - 0.16],
      partId: 'routed-branch',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-routed-sum',
      label: 'Routed Sum (Σ₆)',
      world: [xBranchOut, 0.18, zRoutedSum - ioMaxZ * 0.5 - 0.20],
      partId: 'routed-sum',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-shared-up',
      label: isFp8Mode
        ? 'Shared W_up (3,712 = 2×) · FP8'
        : 'Shared W_up (3,712 = 2×)',
      world: [xSharedUp, 0.14, zShared + HIDDEN_SPAN * 0.5 + 0.24],
      partId: 'shared-expert',
      align: 'bottom',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'moe-shared-act',
      label: 'Shared ReLU² (2×)',
      world: [xSharedAct, 0.16, zShared + SHARED_INTER_SPAN * 0.5 + 0.22],
      partId: 'shared-act',
      align: 'bottom',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'moe-shared-down',
      label: isFp8Mode ? 'Shared W_down (2×) · FP8' : 'Shared W_down (2×)',
      world: [xSharedDown, 0.14, zShared + SHARED_INTER_SPAN * 0.5 + 0.24],
      partId: 'shared-w-down',
      align: 'bottom',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'moe-shared-out',
      label: 'Shared Out (1× Unscaled)',
      world: [xBranchOut, 0.16, zSharedOut + ioMaxZ * 0.5 + 0.22],
      partId: 'shared-out',
      align: 'bottom',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'moe-merge',
      label: 'MoE Out',
      world: [xMoeOut, 0.16, ioMaxZ * 0.5 + 0.24],
      partId: 'moe-out',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'moe-res-out',
      label: '+ Residual',
      world: [xResOut, 0.16, ioMaxZ * 0.5 + 0.24],
      partId: 'residual-out',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'moe-res-bridge',
      label: bayId === 52 ? 'Residual Highway (2,688 · Final Bay 52 · 0 B cache)' : 'Residual Highway (2,688 · 0 B token cache)',
      world: [0.30, bridgeY + 0.22, bridgeZ],
      partId: 'residual-in',
      align: 'top',
      tier: 'primary',
    },
    // Contextual carry-in badge on Step 6 Residual addition (anchored inside Step 6's zoomed frustum)
    ...(residualCarryInfo
      ? [
          {
            id: 'moe-res-carry',
            label: `← Input X[${residualCarryInfo.ch}, t${residualCarryInfo.tok + (sim.startTokenOffset ?? 0)}] = ${fmtSigned(residualCarryInfo.val)}`,
            world: [xResOut - 0.16, 0.46, -1.20] as Vec3,
            partId: 'residual-in',
            align: 'left' as const,
            tier: 'primary' as const,
            tone: 'cyan' as const,
          },
        ]
      : []),
    // Contextual Top-6 normalization note on Step 3
    ...(isInspectingTopk
      ? [
          {
            id: 'moe-topk-norm-note',
            label: 'Bias dropped → w = (σⱼ / Σ₆ σₘ) × 2.5 (all 6 in denom)',
            world: [(xUnbiased + xTopk) * 0.5, bridgeDeckY + 0.72, zRouted - topkSpanZ * 0.5 - 0.10] as Vec3,
            partId: 'router',
            align: 'top' as const,
            tier: 'primary' as const,
            tone: 'gold' as const,
          },
        ]
      : []),
    // Progressive detail callout showing magnification relationship (1 of 6)
    {
      id: 'moe-detail-status',
      label: focusedEval.isActive
        ? `Magnified #${(focusedRank ?? 0) + 1}/6: ${focusExpTag} (w=${focusedEval.appliedWeight.toFixed(2)})`
        : `${focusExpTag} IDLE PREVIEW (Rank #${focusedEval.choiceRank + 1}/128 · 0 contrib)`,
      world: [(xRoutedWUp + xRoutedWDown) * 0.5, 0.14, wbZ1 + 0.18],
      partId: 'routed-w-up',
      align: 'bottom',
      tier: 'detail',
      tone: focusedEval.isActive ? 'default' : 'muted',
    },
  );

  // Dimension caliper visually proving strict 2x intermediate width on Shared Expert
  b.calipers.push({
    id: 'cal-shared-2x',
    aWorld: [xSharedDown, 0.04, zShared - SHARED_INTER_SPAN * 0.5],
    bWorld: [xSharedDown, 0.04, zShared + SHARED_INTER_SPAN * 0.5],
    label: '3,712 = 2× routed intermediate (1,856) · 1× hidden (2,688)',
    partIds: ['shared-expert', 'shared-w-down', 'shared-act', 'shared-out'],
    minPixels: 68,
    offsetPx: 10,
  });

  // ==========================================================================
  // 10. TIGHT PER-OPERATION CAMERA FOCUS BOUNDS (Steps 0..5)
  // ==========================================================================
  const reg = (k: string) => b.sheetRegistry.get(k);
  const allBranches = [0, 1, 2, 3, 4, 5].map((j) => reg(`routed_branch:${j}`));
  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('input'), reg('rmsnorm')], 0.48, 0.28, 0.48, MOE_SPATIAL_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('rmsnorm'), reg('w_router'), bankEnvelope], 0.48, 0.36, 0.48, MOE_SPATIAL_FIT_BOUNDS),
    2: computeSheetGroupBounds([bankEnvelope, reg('topk_unbiased'), reg('topk_weights')], 0.48, 0.36, 0.48, MOE_SPATIAL_FIT_BOUNDS),
    3: computeSheetGroupBounds(
      [reg('routed_w_up'), reg('routed_pre'), reg('routed_act'), reg('routed_w_down'), ...allBranches, reg('routed_sum')],
      0.50,
      0.34,
      0.48,
      MOE_SPATIAL_FIT_BOUNDS,
    ),
    4: computeSheetGroupBounds(
      [reg('shared_w_up'), reg('shared_act'), reg('shared_w_down'), reg('shared_out')],
      0.50,
      0.34,
      0.48,
      MOE_SPATIAL_FIT_BOUNDS,
    ),
    5: computeSheetGroupBounds(
      [reg('routed_sum'), reg('shared_out'), reg('moe_out'), reg('residual')],
      0.52,
      0.36,
      0.52,
      MOE_SPATIAL_FIT_BOUNDS,
    ),
  };

  const packed = b.pack();
  return {
    kind: 'cutaway_moe',
    bayId,
    ...packed,
    calipers: b.calipers,
    valueAtlas: atlas,
    moeSim: sim,
    activeFormula,
    fitBounds: MOE_SPATIAL_FIT_BOUNDS,
    opFocusBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    solidPickBoxes: b.pickBoxes,
    partCatalog,
    moeStats,
  };
}
