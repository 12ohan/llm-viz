import {
  formatMambaHeadId,
  mambaGroupForHead,
  readMambaConvTapValue,
  readMambaPriorSsmValue,
  type MambaCellFormula,
  type MambaCellRef,
  type MambaSheetSpan,
  type MambaSimResult,
  type MambaTensorKey,
} from '../math/mambaSim';
import {
  NEMOTRON_SPEC,
  type Vec3,
} from '../model/spec';
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
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  pushBoxFrameXZ,
  pushFreeGuideLine,
} from '../render/linePacking';
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
  SHEET_KIND_GHOST_RAIL,
  SHEET_KIND_STANDARD,
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../render/sheets';
import {
  computeMambaFocusedAccumulation,
  type FocusedAccumulationResult,
} from './accumulation';
import type { AtlasSlice } from './atlasPacker';
import type {
  ActiveSheetCellGrid,
  CellPopoverSpec,
  CutawayPartInfo,
  CutawaySceneGeometry,
} from './cutaways';
import type { MambaValueAtlas } from './mambaAtlas';
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
  resolveSheetFocusRole,
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
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type LandmarkAnchor,
} from './spine';

/**
 * Invariant 3D bounding envelope for Spatial Mamba-2 across T=1..16, Bay 01/42,
 * BF16/FP8 Selective, all 16 lifted active-head states, and temporal playback t=0..T-1.
 */
export const MAMBA_SPATIAL_FIT_BOUNDS: AABB = {
  min: [-8.75, -0.06, -3.90],
  max: [9.05, 3.30, 3.65],
};

export interface SpatialMambaFitOptions {
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly isNarrow?: boolean;
  readonly azimuthDeg?: number;
  readonly elevationDeg?: number;
}

/**
 * Exact 8-corner projective camera fitter for Spatial Mamba-2.
 * Mirrors `fitSpatialGqaCamera` so every corner of `bounds` sits strictly inside the
 * unobstructed viewport band `[0, innerWidth] x [topInset, innerHeight - bottomInset]`.
 */
export function fitSpatialMambaCamera(
  bounds: AABB,
  metrics: ViewportMetrics,
  options?: SpatialMambaFitOptions,
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
 * Horizontal XZ sheet aligned monotonically with upright YZ weight walls (`FRAME_YZ_AXES`):
 *   - `axisU = [0, 0, -1]` (channel col 0 at +Z front -> col N-1 at -Z rear, matching `W_in`/`W_out`)
 *   - `axisN = [0, 1, 0]`  (sheet normal +Y up)
 *   - `axisV = cross(U, N) = [+1, 0, 0]` (token row 0 at -X left -> newest row T-1 at +X right)
 */
const FRAME_XZ_CH_Z_TOK_X: SheetBasisAxes = {
  axisU: [0, 0, -1],
  axisN: [0, 1, 0],
  axisV: [1, 0, 0],
};

// Semantic branch colors matching the quiet drafting-table language
const RIBBON_GATE_FLOW: FlowRibbonColor = { r: 0.16, g: 0.56, b: 0.45, a: 0.42, borderAlpha: 0.24 };
const RIBBON_GATE_ACTIVE: FlowRibbonColor = { r: 0.12, g: 0.54, b: 0.42, a: 0.78, borderAlpha: 0.42 };
const RIBBON_DT_FLOW: FlowRibbonColor = { r: 0.72, g: 0.50, b: 0.18, a: 0.44, borderAlpha: 0.24 };
const RIBBON_DT_ACTIVE: FlowRibbonColor = { r: 0.78, g: 0.46, b: 0.12, a: 0.80, borderAlpha: 0.42 };
const RIBBON_BC_FLOW: FlowRibbonColor = { r: 0.14, g: 0.52, b: 0.62, a: 0.42, borderAlpha: 0.24 };
const RIBBON_BC_ACTIVE: FlowRibbonColor = { r: 0.10, g: 0.54, b: 0.64, a: 0.78, borderAlpha: 0.42 };
const TOKEN_ROW_FILL: FlowRibbonColor = { r: 0.12, g: 0.52, b: 0.68, a: 0.14, borderAlpha: 0.0 };
const TOKEN_ROW_FRAME: FlowRibbonColor = { r: 0.12, g: 0.52, b: 0.68, a: 0.72, borderAlpha: 0.25 };

interface SpatialMambaSheetSpec {
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
  readonly groupIndex?: number;
  readonly tensorKey?: MambaTensorKey;
  readonly tokenIndex?: number;
  readonly tapIndex?: number;
  readonly stateTokenIndex?: number;
  readonly atlasSlice?: AtlasSlice;
  readonly registryKey?: string;
  readonly cellFamily: number;
}

export interface MambaRevealParams {
  readonly stepperStep: number;
  readonly subProgress: number;
  readonly clampedT: number;
  readonly activeHead: number;
  readonly activeGroup: number;
  readonly stateTokIdx: number;
  readonly activeCell: MambaCellRef | null;
  readonly sim: MambaSimResult;
}

/**
 * Compute per-sheet V2 progressive accumulation reveal tuple
 * `[revealMode, colFrontier, rowFrontier, pulsePhase]` for Mamba-2 Math Stepper steps 0..5.
 *
 * Critical architectural invariants:
 * - Learned weights (`w_in`, `w_out`), `input`, and carried prior state `H_{t-1}` (`h_prev`)
 *   ALWAYS stay `SHEET_REVEAL_NONE` (never ghosted as uncomputed, even in Step 1 or seeded AR Decode).
 * - Step 4 (`stepperStep === 3`, SSM Update) sweeps `H_t` (`h_curr`, 4×4) with `SHEET_REVEAL_ROW_MAJOR`
 *   at `(col = activeCell.col, row = activeCell.row)` while downstream `y`, `y_norm`, `residual`
 *   render as `SHEET_REVEAL_UNCOMPUTED`.
 */
export function computeMambaSheetReveal(
  params: MambaRevealParams,
  sheet: {
    readonly tensorKey?: MambaTensorKey;
    readonly partId: string;
    readonly headIndex?: number;
    readonly sheetKind?: number;
  },
): readonly [number, number, number, number] {
  if (sheet.sheetKind === SHEET_KIND_GHOST_RAIL || sheet.partId === 'seq-rail') {
    return [SHEET_REVEAL_NONE, 0, 0, 0];
  }

  const key = sheet.tensorKey;
  // Input X, learned weights (W_in, W_out), and carried prior SSM state H_{t-1} are always revealed
  if (key === 'input' || key === 'w_in' || key === 'w_out' || key === 'h_prev') {
    return [SHEET_REVEAL_NONE, 0, 0, 0];
  }

  const { stepperStep, subProgress, clampedT, activeHead, activeGroup, stateTokIdx, activeCell, sim } =
    params;
  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  const cT = Math.max(1, clampedT);
  const p01 = Number.isFinite(subProgress) ? Math.max(0, Math.min(0.9999, subProgress)) : 0.25;
  const pulse = Number.isFinite(subProgress) ? ((subProgress * 4) % 1 + 1) % 1 : 0.25;
  const clampIdx = (v: number, max: number) => Math.max(0, Math.min(max - 1, Math.trunc(v)));

  // Step 1 (stepperStep === 0): Pre-Layer RMSNorm
  if (step === 0) {
    if (key === 'rmsnorm') {
      const col =
        activeCell?.tensor === 'rmsnorm'
          ? clampIdx(activeCell.col, sim.d)
          : clampIdx(Math.floor(p01 * sim.d), sim.d);
      const row =
        activeCell?.tensor === 'rmsnorm'
          ? clampIdx(activeCell.row, cT)
          : clampIdx(stateTokIdx, cT);
      return [SHEET_REVEAL_ROW_MAJOR, col, row, pulse];
    }
    return [SHEET_REVEAL_UNCOMPUTED, 0, 0, 0];
  }

  // Step 2 (stepperStep === 1): Fused In-Proj -> z, xBC_pre, Δt
  if (step === 1) {
    if (key === 'rmsnorm') return [SHEET_REVEAL_NONE, 0, 0, 0];
    if (key === 'z') {
      if (activeCell?.tensor === 'z') {
        return [
          SHEET_REVEAL_ROW_MAJOR,
          clampIdx(activeCell.col, sim.dInner),
          clampIdx(activeCell.row, cT),
          pulse,
        ];
      }
      if (activeCell?.tensor === 'xbc_pre' || activeCell?.tensor === 'dt') {
        return [SHEET_REVEAL_NONE, 0, 0, 0];
      }
      return [
        SHEET_REVEAL_ROW_MAJOR,
        clampIdx(Math.floor(p01 * sim.dInner), sim.dInner),
        clampIdx(stateTokIdx, cT),
        pulse,
      ];
    }
    if (key === 'xbc_pre') {
      if (activeCell?.tensor === 'xbc_pre') {
        return [
          SHEET_REVEAL_ROW_MAJOR,
          clampIdx(activeCell.col, sim.convDim),
          clampIdx(activeCell.row, cT),
          pulse,
        ];
      }
      if (activeCell?.tensor === 'dt') {
        return [SHEET_REVEAL_NONE, 0, 0, 0];
      }
      if (activeCell?.tensor === 'z') {
        const scaledCol = Math.floor((activeCell.col / Math.max(1, sim.dInner)) * sim.convDim);
        return [
          SHEET_REVEAL_ROW_MAJOR,
          clampIdx(scaledCol, sim.convDim),
          clampIdx(activeCell.row, cT),
          pulse,
        ];
      }
      return [
        SHEET_REVEAL_ROW_MAJOR,
        clampIdx(Math.floor(p01 * sim.convDim), sim.convDim),
        clampIdx(stateTokIdx, cT),
        pulse,
      ];
    }
    if (key === 'dt') {
      if (activeCell?.tensor === 'dt') {
        return [
          SHEET_REVEAL_ROW_MAJOR,
          clampIdx(activeCell.headIndex ?? activeCell.col, sim.nHeads),
          clampIdx(activeCell.row, cT),
          pulse,
        ];
      }
      const row =
        activeCell && (activeCell.tensor === 'z' || activeCell.tensor === 'xbc_pre')
          ? clampIdx(activeCell.row, cT)
          : clampIdx(stateTokIdx, cT);
      return [SHEET_REVEAL_ROW_MAJOR, clampIdx(activeHead, sim.nHeads), row, pulse];
    }
    // Downstream Conv1D, x/B/C, H_t, 64-head bank, y, y_norm, residual are uncomputed
    return [SHEET_REVEAL_UNCOMPUTED, 0, 0, 0];
  }

  // Step 3 (stepperStep === 2): Depthwise Causal Conv1D (k=4) + SiLU -> x, B, C
  if (step === 2) {
    if (key === 'rmsnorm' || key === 'z' || key === 'xbc_pre' || key === 'dt' || key === 'conv_buf') {
      return [SHEET_REVEAL_NONE, 0, 0, 0];
    }
    const rowF =
      activeCell && (activeCell.tensor === 'x' || activeCell.tensor === 'b' || activeCell.tensor === 'c')
        ? clampIdx(activeCell.row, cT)
        : clampIdx(stateTokIdx, cT);
    const pF =
      activeCell?.tensor === 'x'
        ? activeCell.headIndex !== undefined && activeCell.col >= 0 && activeCell.col < sim.dHead
          ? activeCell.col
          : clampIdx(activeCell.col % sim.dHead, sim.dHead)
        : activeCell?.tensor === 'conv_buf'
          ? clampIdx(activeCell.tapIndex ?? activeCell.row, sim.dHead)
          : clampIdx(Math.floor(p01 * sim.dHead), sim.dHead);

    const resolveGroupCol = (cell: MambaCellRef): number => {
      const isLocalWithGroup =
        cell.groupIndex !== undefined &&
        cell.col >= 0 &&
        cell.col < sim.dState &&
        clampIdx(cell.groupIndex, sim.nGroups) > 0;
      return isLocalWithGroup
        ? clampIdx(cell.groupIndex!, sim.nGroups) * sim.dState + cell.col
        : clampIdx(cell.col, sim.nGroups * sim.dState);
    };

    if (key === 'x') {
      return [SHEET_REVEAL_ROW_MAJOR, pF, rowF, pulse];
    }
    if (key === 'b') {
      const colF =
        activeCell?.tensor === 'b'
          ? resolveGroupCol(activeCell)
          : clampIdx(activeGroup * sim.dState + pF, sim.nGroups * sim.dState);
      return [SHEET_REVEAL_ROW_MAJOR, colF, rowF, pulse];
    }
    if (key === 'c') {
      const colF =
        activeCell?.tensor === 'c'
          ? resolveGroupCol(activeCell)
          : clampIdx(activeGroup * sim.dState + pF, sim.nGroups * sim.dState);
      return [SHEET_REVEAL_ROW_MAJOR, colF, rowF, pulse];
    }
    // Downstream H_t, 64-head bank, y, y_norm, residual are uncomputed
    return [SHEET_REVEAL_UNCOMPUTED, 0, 0, 0];
  }

  // Step 4 (stepperStep === 3): In-Place FP32 SSM State Recurrence H_t = dA·H_{t-1} + Δt·x·B
  if (step === 3) {
    if (
      key === 'rmsnorm' ||
      key === 'z' ||
      key === 'xbc_pre' ||
      key === 'dt' ||
      key === 'conv_buf' ||
      key === 'x' ||
      key === 'b' ||
      key === 'c'
    ) {
      return [SHEET_REVEAL_NONE, 0, 0, 0];
    }
    if (key === 'h_curr') {
      if (activeCell?.tensor === 'h_curr') {
        return [
          SHEET_REVEAL_ROW_MAJOR,
          clampIdx(activeCell.col, sim.dState),
          clampIdx(activeCell.row, sim.dHead),
          pulse,
        ];
      }
      const total = sim.dHead * sim.dState;
      const idx = clampIdx(Math.floor(p01 * total), total);
      const rowF = Math.floor(idx / sim.dState);
      const colF = idx % sim.dState;
      return [SHEET_REVEAL_ROW_MAJOR, colF, rowF, pulse];
    }
    if (sheet.partId === 'ssm-state') {
      if (sheet.headIndex !== undefined && sheet.headIndex > activeHead) {
        return [SHEET_REVEAL_UNCOMPUTED, 0, 0, 0];
      }
      return [SHEET_REVEAL_NONE, 0, 0, 0];
    }
    // Downstream y, y_norm, residual are uncomputed
    return [SHEET_REVEAL_UNCOMPUTED, 0, 0, 0];
  }

  // Step 5 (stepperStep === 4): State Readout + Skip D·x -> y
  if (step === 4) {
    if (key === 'y') {
      if (activeCell?.tensor === 'y') {
        return [
          SHEET_REVEAL_ROW_MAJOR,
          clampIdx(activeCell.col, sim.dInner),
          clampIdx(activeCell.row, cT),
          pulse,
        ];
      }
      const pF = clampIdx(Math.floor(p01 * sim.dHead), sim.dHead);
      return [
        SHEET_REVEAL_ROW_MAJOR,
        clampIdx(activeHead * sim.dHead + pF, sim.dInner),
        clampIdx(stateTokIdx, cT),
        pulse,
      ];
    }
    if (key === 'y_norm' || key === 'residual') {
      return [SHEET_REVEAL_UNCOMPUTED, 0, 0, 0];
    }
    return [SHEET_REVEAL_NONE, 0, 0, 0];
  }

  // Step 6 (stepperStep === 5): Gate FIRST -> 8-Group RMSNorm -> W_out -> + Residual
  if (key === 'y_norm') {
    if (activeCell?.tensor === 'y_norm') {
      return [
        SHEET_REVEAL_ROW_MAJOR,
        clampIdx(activeCell.col, sim.dInner),
        clampIdx(activeCell.row, cT),
        pulse,
      ];
    }
    return [SHEET_REVEAL_NONE, 0, 0, 0];
  }
  if (key === 'residual') {
    if (activeCell?.tensor === 'y_norm') {
      return [SHEET_REVEAL_UNCOMPUTED, 0, 0, 0];
    }
    if (activeCell?.tensor === 'residual') {
      return [
        SHEET_REVEAL_ROW_MAJOR,
        clampIdx(activeCell.col, sim.d),
        clampIdx(activeCell.row, cT),
        pulse,
      ];
    }
    return [
      SHEET_REVEAL_ROW_MAJOR,
      clampIdx(Math.floor(p01 * sim.d), sim.d),
      clampIdx(stateTokIdx, cT),
      pulse,
    ];
  }
  return [SHEET_REVEAL_NONE, 0, 0, 0];
}

/**
 * Resolve RC1 focused-cell running partial-sum state for Mamba-2 during active Stepper
 * playback (`stepperActive && !reducedMotion && stepperStep != null && termProgress < 0.999`).
 * Returns `null` during ordinary unrestricted inspection or when accumulation is complete.
 */
export function resolveMambaFocusedAccumulation(params: {
  readonly sim: MambaSimResult;
  readonly activeCell: MambaCellRef | null;
  readonly stepperStep?: number | null;
  readonly stepperActive?: boolean;
  readonly reducedMotion?: boolean;
  readonly termProgress?: number;
}): FocusedAccumulationResult | null {
  const {
    sim,
    activeCell,
    stepperStep,
    stepperActive = false,
    reducedMotion = false,
    termProgress,
  } = params;
  if (
    !stepperActive ||
    reducedMotion ||
    stepperStep === undefined ||
    stepperStep === null ||
    termProgress === undefined ||
    !Number.isFinite(termProgress) ||
    termProgress >= 0.999 ||
    !activeCell
  ) {
    return null;
  }
  return computeMambaFocusedAccumulation(sim, activeCell, stepperStep, termProgress);
}

function mambaPartIdForTensor(tensor: MambaTensorKey, headIndex?: number): string {
  switch (tensor) {
    case 'input':
      return 'residual-in';
    case 'rmsnorm':
      return 'rmsnorm';
    case 'w_in':
      return 'in-proj';
    case 'z':
      return 'gate-branch';
    case 'xbc_pre':
      return 'xbc-pre';
    case 'conv_buf':
      return 'conv1d';
    case 'dt':
      return 'dt-branch';
    case 'x':
      return `mamba-head-${headIndex ?? 0}`;
    case 'b':
      return 'b-proj';
    case 'c':
      return 'c-proj';
    case 'h_prev':
      return 'ssm-state-prev';
    case 'h_curr':
      return 'ssm-state-curr';
    case 'y':
      return 'ssm-readout';
    case 'y_norm':
      return 'group-norm';
    case 'w_out':
      return 'out-proj';
    case 'residual':
      return 'residual-out';
  }
}

function formatConciseMambaPopoverEquation(formula: MambaCellFormula): string {
  let eq = formula.equation.trim();
  // Strip verbose double-spaced parenthetical metadata notes so floating cards stay crisp
  const parenMetaIdx = eq.indexOf('  (');
  if (parenMetaIdx > 0) {
    eq = eq.slice(0, parenMetaIdx).trim();
  }
  // Avoid repeating the final `= valueText` already displayed in the oxide-red value pill
  const valSuffix = ` = ${formula.valueText}`;
  if (eq.endsWith(valSuffix) && eq.length > valSuffix.length + 2) {
    eq = eq.slice(0, eq.length - valSuffix.length).trim();
  }
  return eq;
}

/**
 * Build a V3 floating contextual cell formula popover anchored at the active/hovered cell's
 * 3D world center with concise formula, formatted value, operand swatches, and provenance targets.
 * When RC1 `focusedAccumulation?.active` is true, renders ONLY the running partial sum and active
 * term expression without prematurely exposing completed final values.
 */
export function buildMambaCellPopover(
  activeFormula: MambaCellFormula | null,
  targetWorld: Vec3 | null,
  sourceWorlds: readonly {
    readonly world: readonly [number, number, number];
    readonly role: 'indigo' | 'teal';
  }[],
  focusedAccumulation?: FocusedAccumulationResult | null,
): CellPopoverSpec | null {
  if (!activeFormula || !targetWorld) return null;

  const hasOpA = activeFormula.upstreamSpans.some((s) => s.role === 'operand_a');
  const hasOpB = activeFormula.upstreamSpans.some((s) => s.role === 'operand_b');
  const operandColors: ('indigo' | 'teal' | 'red')[] = [];
  if (hasOpA) operandColors.push('indigo');
  if (hasOpB) operandColors.push('teal');
  operandColors.push('red');

  // Keep popover header concise so tensorLabel + coordLabel + badge fit cleanly inside max-width 360px
  const compactTensorLabel = activeFormula.tensorLabel.replace(/\s*\(H\d{2}\)\s*$/, '');
  const compactCoordLabel = activeFormula.coordLabel
    .replace(/^[A-Za-z_]+\[/, '[')
    .replace(/\("[^"]*"\)/g, '');

  const isAccumulating = Boolean(focusedAccumulation?.active);

  return {
    world: targetWorld,
    tensorLabel: compactTensorLabel,
    coordLabel: compactCoordLabel,
    ...(isAccumulating
      ? { badge: focusedAccumulation!.badgeText }
      : activeFormula.headBadge
        ? { badge: activeFormula.headBadge }
        : {}),
    equation: isAccumulating
      ? focusedAccumulation!.popoverEquation
      : formatConciseMambaPopoverEquation(activeFormula),
    valueText: isAccumulating
      ? focusedAccumulation!.valueBadgeText
      : `= ${activeFormula.valueText}`,
    ...(isAccumulating ? { isAccumulating: true } : {}),
    operandColors,
    ...(sourceWorlds.length > 0 ? { sourceWorlds } : {}),
  };
}

function sheetGeomToGridBasis(rec: SheetGeomRecord): {
  readonly originWorld: readonly [number, number, number];
  readonly extent: readonly [number, number, number];
  readonly axisU: readonly [number, number, number];
  readonly axisN: readonly [number, number, number];
} {
  if (rec.frame) {
    return {
      originWorld: rec.frame.center,
      extent: rec.frame.extent,
      axisU: rec.frame.axisU,
      axisN: rec.frame.axisN,
    };
  }
  return {
    originWorld: [rec.cx, rec.cy, rec.cz],
    extent: [rec.ex, rec.ey, rec.ez],
    axisU: FRAME_XZ_AXES.axisU,
    axisN: FRAME_XZ_AXES.axisN,
  };
}

/**
 * Build V3 zoom-adaptive `ActiveSheetCellGrid[]` directly from live `sim` Float32 arrays
 * for active `H_t` (`4×4`, `p0..p3` × `n0..n3`), `H_{t-1}` (`4×4`, carried/prior state),
 * and the active target sheet (e.g. `y`, `RMSNorm`, `Δt`, `x`, `Residual`) with absolute
 * token labels (`t${absTok} ${label}`).
 */
export function buildMambaActiveCellGrids(
  registry: ReadonlyMap<string, SheetGeomRecord>,
  sim: MambaSimResult,
  clampedT: number,
  activeHead: number,
  stateTokIdx: number,
  activeFormula: MambaCellFormula | null,
  revealParams: MambaRevealParams | null = null,
  focusedAccumulation: FocusedAccumulationResult | null = null,
): readonly ActiveSheetCellGrid[] {
  const { d, nHeads, nGroups, dHead, dInner, dState, convDim, inProjDim, tokens: T } = sim;
  const rowsT = Math.min(clampedT, T);
  const headTag = formatMambaHeadId(activeHead);
  const stateStride = nHeads * dHead * dState;

  const absTokLabel = (t: number): string => {
    const clamped = Math.max(0, Math.min(T - 1, t));
    const absTok = sim.startTokenOffset + clamped;
    const lbl = sim.tokenLabels[clamped] ?? '';
    return lbl ? `t${absTok} ${lbl}` : `t${absTok}`;
  };
  const tokenRowLabels: readonly string[] = Array.from({ length: rowsT }, (_, t) => absTokLabel(t));
  const pLabels: readonly string[] = Array.from({ length: dHead }, (_, p) => `p${p}`);
  const nLabels: readonly string[] = Array.from({ length: dState }, (_, n) => `n${n}`);
  const chLabels: readonly string[] = Array.from({ length: d }, (_, i) => `ch${i}`);

  const buildHCurrGrid = (): ActiveSheetCellGrid | null => {
    const rec = registry.get('h_curr');
    if (!rec) return null;
    const values = new Float32Array(dHead * dState);
    for (let p = 0; p < dHead; p++) {
      for (let n = 0; n < dState; n++) {
        const localIdx = (activeHead * dHead + p) * dState + n;
        values[p * dState + n] = sim.H_history[stateTokIdx * stateStride + localIdx];
      }
    }
    let activeCol: number | undefined;
    let activeRow: number | undefined;
    if (activeFormula?.targetSpan.tensor === 'h_curr') {
      activeCol = activeFormula.targetSpan.colStart;
      activeRow = activeFormula.targetSpan.rowStart;
    } else {
      const upH = activeFormula?.upstreamSpans.find((s) => s.tensor === 'h_curr');
      if (upH) {
        activeRow = upH.rowStart;
        if (upH.colStart === upH.colEnd) activeCol = upH.colStart;
      }
    }
    return {
      ...sheetGeomToGridBasis(rec),
      cols: dState,
      rows: dHead,
      values,
      format: 'signed',
      activeCol,
      activeRow,
      colLabels: nLabels,
      rowLabels: pLabels,
      colDimLabel: `H_t ${headTag}`,
      rowDimLabel: absTokLabel(stateTokIdx),
    };
  };

  const buildHPrevGrid = (): ActiveSheetCellGrid | null => {
    const rec = registry.get('h_prev');
    if (!rec) return null;
    const values = new Float32Array(dHead * dState);
    for (let p = 0; p < dHead; p++) {
      for (let n = 0; n < dState; n++) {
        const localIdx = (activeHead * dHead + p) * dState + n;
        values[p * dState + n] = readMambaPriorSsmValue(sim, stateTokIdx, localIdx).value;
      }
    }
    let activeCol: number | undefined;
    let activeRow: number | undefined;
    if (activeFormula?.targetSpan.tensor === 'h_prev') {
      activeCol = activeFormula.targetSpan.colStart;
      activeRow = activeFormula.targetSpan.rowStart;
    } else {
      const upPrev = activeFormula?.upstreamSpans.find((s) => s.tensor === 'h_prev');
      if (upPrev) {
        activeRow = upPrev.rowStart;
        activeCol = upPrev.colStart;
      }
    }
    const isSeededPrior = sim.hasInitialState && stateTokIdx === 0;
    const prevTokLabel =
      stateTokIdx > 0
        ? absTokLabel(stateTokIdx - 1)
        : isSeededPrior
          ? `t${Math.max(0, sim.startTokenOffset - 1)} carried`
          : 't−1 zero';
    return {
      ...sheetGeomToGridBasis(rec),
      cols: dState,
      rows: dHead,
      values,
      format: 'signed',
      activeCol,
      activeRow,
      colLabels: nLabels,
      rowLabels: pLabels,
      colDimLabel: isSeededPrior ? `H_{t−1} ${headTag} · Carried` : `H_{t−1} ${headTag}`,
      rowDimLabel: prevTokLabel,
    };
  };

  const buildXHeadGrid = (
    headIdx: number,
    activeCol?: number,
    activeRow?: number,
  ): ActiveSheetCellGrid | null => {
    const rec = registry.get(`x:${headIdx}`);
    if (!rec) return null;
    const values = new Float32Array(rowsT * dHead);
    for (let t = 0; t < rowsT; t++) {
      for (let p = 0; p < dHead; p++) {
        values[t * dHead + p] = sim.x[(headIdx * dHead + p) * T + t];
      }
    }
    return {
      ...sheetGeomToGridBasis(rec),
      cols: dHead,
      rows: rowsT,
      values,
      format: 'signed',
      activeCol,
      activeRow,
      colLabels: pLabels,
      rowLabels: tokenRowLabels,
      colDimLabel: `x ${formatMambaHeadId(headIdx)}`,
      rowDimLabel: `T = ${rowsT}`,
    };
  };

  const buildTargetTensorGrid = (span: MambaSheetSpan): ActiveSheetCellGrid | null => {
    const ac = span.colStart;
    const ar = span.rowStart;
    switch (span.tensor) {
      case 'h_curr':
      case 'h_prev':
        return null;
      case 'x':
        return buildXHeadGrid(span.headIndex ?? activeHead, ac, ar);
      case 'input':
      case 'rmsnorm':
      case 'residual': {
        const rec = registry.get(span.tensor);
        if (!rec) return null;
        const src =
          span.tensor === 'input' ? sim.X : span.tensor === 'rmsnorm' ? sim.U : sim.Y;
        const values = new Float32Array(rowsT * d);
        for (let t = 0; t < rowsT; t++) {
          for (let i = 0; i < d; i++) {
            values[t * d + i] = src[i * T + t];
          }
        }
        return {
          ...sheetGeomToGridBasis(rec),
          cols: d,
          rows: rowsT,
          values,
          format: 'signed',
          activeCol: ac,
          activeRow: ar,
          colLabels: chLabels,
          rowLabels: tokenRowLabels,
          colDimLabel: `d = ${d}`,
          rowDimLabel: `T = ${rowsT}`,
        };
      }
      case 'dt': {
        const rec = registry.get('dt');
        if (!rec) return null;
        const values = new Float32Array(rowsT * nHeads);
        for (let t = 0; t < rowsT; t++) {
          for (let h = 0; h < nHeads; h++) {
            values[t * nHeads + h] = sim.dt[h * T + t];
          }
        }
        return {
          ...sheetGeomToGridBasis(rec),
          cols: nHeads,
          rows: rowsT,
          values,
          format: 'signed',
          activeCol: ac,
          activeRow: ar,
          colLabels: Array.from({ length: nHeads }, (_, h) => formatMambaHeadId(h)),
          rowLabels: tokenRowLabels,
          colDimLabel: `Δt (${nHeads}h)`,
          rowDimLabel: `T = ${rowsT}`,
        };
      }
      case 'b':
      case 'c': {
        const rec = registry.get(span.tensor);
        if (!rec) return null;
        const cols = nGroups * dState;
        const src = span.tensor === 'b' ? sim.B : sim.C;
        const values = new Float32Array(rowsT * cols);
        for (let t = 0; t < rowsT; t++) {
          for (let k = 0; k < cols; k++) {
            values[t * cols + k] = src[k * T + t];
          }
        }
        return {
          ...sheetGeomToGridBasis(rec),
          cols,
          rows: rowsT,
          values,
          format: 'signed',
          activeCol: ac,
          activeRow: ar,
          colLabels: Array.from({ length: cols }, (_, k) => `G${Math.floor(k / dState)}·n${k % dState}`),
          rowLabels: tokenRowLabels,
          colDimLabel: `${span.tensor.toUpperCase()} (${nGroups}×${dState})`,
          rowDimLabel: `T = ${rowsT}`,
        };
      }
      case 'z':
      case 'y':
      case 'y_norm': {
        const rec = registry.get(span.tensor);
        if (!rec) return null;
        const src =
          span.tensor === 'z' ? sim.z : span.tensor === 'y' ? sim.y : sim.y_norm;
        const values = new Float32Array(rowsT * dInner);
        for (let t = 0; t < rowsT; t++) {
          for (let c = 0; c < dInner; c++) {
            values[t * dInner + c] = src[c * T + t];
          }
        }
        return {
          ...sheetGeomToGridBasis(rec),
          cols: dInner,
          rows: rowsT,
          values,
          format: 'signed',
          activeCol: ac,
          activeRow: ar,
          colLabels: Array.from(
            { length: dInner },
            (_, c) => `${formatMambaHeadId(Math.floor(c / dHead))}·p${c % dHead}`,
          ),
          rowLabels: tokenRowLabels,
          colDimLabel: `${span.tensor === 'y_norm' ? 'y_norm' : span.tensor} (${dInner})`,
          rowDimLabel: `T = ${rowsT}`,
        };
      }
      case 'xbc_pre': {
        const rec = registry.get('xbc_pre');
        if (!rec) return null;
        const values = new Float32Array(rowsT * convDim);
        for (let t = 0; t < rowsT; t++) {
          for (let c = 0; c < convDim; c++) {
            values[t * convDim + c] = sim.xBC_pre[c * T + t];
          }
        }
        return {
          ...sheetGeomToGridBasis(rec),
          cols: convDim,
          rows: rowsT,
          values,
          format: 'signed',
          activeCol: ac,
          activeRow: ar,
          colLabels: Array.from({ length: convDim }, (_, c) => `c${c}`),
          rowLabels: tokenRowLabels,
          colDimLabel: `xBC_pre (${convDim})`,
          rowDimLabel: `T = ${rowsT}`,
        };
      }
      case 'conv_buf': {
        const slot = Math.max(0, Math.min(sim.convKernel - 1, span.tapStart ?? 0));
        const rec = registry.get(`conv_buf:${slot}`);
        if (!rec) return null;
        const values = new Float32Array(convDim);
        for (let c = 0; c < convDim; c++) {
          values[c] = readMambaConvTapValue(sim, c, stateTokIdx, slot).value;
        }
        const lag = sim.convKernel - 1 - slot;
        return {
          ...sheetGeomToGridBasis(rec),
          cols: convDim,
          rows: 1,
          values,
          format: 'signed',
          activeCol: ac,
          activeRow: 0,
          colLabels: Array.from({ length: convDim }, (_, c) => `c${c}`),
          rowLabels: [`slot ${slot} (t${sim.startTokenOffset + stateTokIdx - lag})`],
          colDimLabel: `Conv1D (${convDim})`,
          rowDimLabel: absTokLabel(stateTokIdx),
        };
      }
      case 'w_in': {
        const rec = registry.get('w_in');
        if (!rec) return null;
        const values = new Float32Array(d * inProjDim);
        for (let r = 0; r < d; r++) {
          for (let c = 0; c < inProjDim; c++) {
            values[r * inProjDim + c] = sim.W_in[c * d + r];
          }
        }
        return {
          ...sheetGeomToGridBasis(rec),
          cols: inProjDim,
          rows: d,
          values,
          format: 'signed',
          activeCol: ac,
          activeRow: ar,
          colLabels: Array.from({ length: inProjDim }, (_, c) => `o${c}`),
          rowLabels: chLabels,
          colDimLabel: `W_in (${inProjDim}×${d})`,
        };
      }
      case 'w_out': {
        const rec = registry.get('w_out');
        if (!rec) return null;
        const values = new Float32Array(d * dInner);
        for (let r = 0; r < d; r++) {
          for (let c = 0; c < dInner; c++) {
            values[r * dInner + c] = sim.W_out[r * dInner + c];
          }
        }
        return {
          ...sheetGeomToGridBasis(rec),
          cols: dInner,
          rows: d,
          values,
          format: 'signed',
          activeCol: ac,
          activeRow: ar,
          colLabels: Array.from({ length: dInner }, (_, c) => `c${c}`),
          rowLabels: chLabels,
          colDimLabel: `W_out (${d}×${dInner})`,
        };
      }
    }
  };

  const applyRevealAndOverride = (
    grid: ActiveSheetCellGrid | null,
    sheetMeta: {
      readonly tensorKey: MambaTensorKey;
      readonly partId: string;
      readonly headIndex?: number;
    },
    isTargetGrid: boolean,
  ): ActiveSheetCellGrid | null => {
    if (!grid) return null;
    const revealFields =
      revealParams !== null
        ? (() => {
            const [revealMode, revealColFrontier, revealRowFrontier] =
              computeMambaSheetReveal(revealParams, sheetMeta);
            return {
              revealMode,
              revealColFrontier,
              revealRowFrontier,
            };
          })()
        : undefined;
    const overrideFields =
      isTargetGrid && focusedAccumulation?.active
        ? { activeValueOverride: focusedAccumulation.partialSum }
        : undefined;

    if (!revealFields && !overrideFields) return grid;
    return {
      ...grid,
      ...revealFields,
      ...overrideFields,
    };
  };

  const grids: ActiveSheetCellGrid[] = [];
  const hCurrGrid = applyRevealAndOverride(
    buildHCurrGrid(),
    { tensorKey: 'h_curr', partId: 'ssm-state-curr', headIndex: activeHead },
    activeFormula?.targetSpan.tensor === 'h_curr',
  );
  const hPrevGrid = applyRevealAndOverride(
    buildHPrevGrid(),
    { tensorKey: 'h_prev', partId: 'ssm-state-prev', headIndex: activeHead },
    false,
  );

  if (activeFormula) {
    const tgt = activeFormula.targetSpan.tensor;
    if (tgt === 'h_curr') {
      if (hCurrGrid) grids.push(hCurrGrid);
      if (hPrevGrid) grids.push(hPrevGrid);
      const upX = activeFormula.upstreamSpans.find((s) => s.tensor === 'x');
      const xGrid = applyRevealAndOverride(
        buildXHeadGrid(
          activeHead,
          upX?.colStart,
          upX?.rowStart ?? stateTokIdx,
        ),
        { tensorKey: 'x', partId: `mamba-head-${activeHead}`, headIndex: activeHead },
        false,
      );
      if (xGrid) grids.push(xGrid);
    } else if (tgt === 'h_prev') {
      if (hPrevGrid) grids.push(hPrevGrid);
      if (hCurrGrid) grids.push(hCurrGrid);
    } else {
      const span = activeFormula.targetSpan;
      const targetHead = span.headIndex ?? activeHead;
      const targetGrid = applyRevealAndOverride(
        buildTargetTensorGrid(span),
        {
          tensorKey: span.tensor,
          partId: mambaPartIdForTensor(span.tensor, targetHead),
          headIndex: targetHead,
        },
        true,
      );
      if (targetGrid) grids.push(targetGrid);
      if (hCurrGrid) grids.push(hCurrGrid);
      if (hPrevGrid) grids.push(hPrevGrid);
    }
  } else {
    if (hCurrGrid) grids.push(hCurrGrid);
    if (hPrevGrid) grids.push(hPrevGrid);
  }

  return grids;
}

class SpatialMambaBuilder {
  readonly sheetFloats: number[] = [];
  readonly basisFloats: number[] = [];
  readonly revealFloats: number[] = [];
  readonly solidFloats: number[] = [];
  readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();
  readonly activeOpPartIds = new Set<string>();
  private readonly bayId: number;
  private readonly revealParams: MambaRevealParams | null;
  private readonly collectActiveOpParts: boolean;

  constructor(
    bayId: number,
    revealParams: MambaRevealParams | null = null,
    collectActiveOpParts = false,
  ) {
    this.bayId = bayId;
    this.revealParams = revealParams;
    this.collectActiveOpParts = collectActiveOpParts;
  }

  addSheet(s: SpatialMambaSheetSpec): SheetFrame3D {
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

    // 2b. Optional parallel V2 progressive reveal buffer (4 floats/sheet)
    if (this.revealParams !== null) {
      const [mode, colF, rowF, pulse] = computeMambaSheetReveal(this.revealParams, s);
      this.revealFloats.push(mode, colF, rowF, pulse);
    }

    // 2c. Track participating target & operand partIds for V1/V3 label dimming
    if (
      this.collectActiveOpParts &&
      s.partId !== 'seq-rail' &&
      (s.highlight === SHEET_HL_ACTIVE ||
        s.highlight === SHEET_HL_OPERAND_A ||
        s.highlight === SHEET_HL_OPERAND_B)
    ) {
      this.activeOpPartIds.add(s.partId);
    }

    // 3. 1-to-1 mirror in solidData (12 floats: fp8Flag at +7, seed/phase at +8)
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

    // 4. Oriented PickBox (padded slightly along axisN so thin 3D sheets pick effortlessly)
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
      groupIndex: s.groupIndex,
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
      tapIndex: s.tapIndex,
      stateTokenIndex: s.stateTokenIndex,
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
    alpha = 0.55,
  ): void {
    pushFreeGuideLine(this.lineFloats, ax, ay, az, bx, by, bz, cssWidth, style, alpha);
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
    pushBoxFrameXZ(this.lineFloats, x0, z0, x1, z1, y, cssWidth, style, alpha);
  }

  pack() {
    const flowPacked = this.flow.pack();
    const sheetCount = this.sheetFloats.length / SHEET_INSTANCE_FLOATS;
    const sheetRevealData =
      this.revealParams !== null && this.revealFloats.length === sheetCount * SHEET_REVEAL_FLOATS
        ? new Float32Array(this.revealFloats)
        : undefined;
    return {
      sheetData: new Float32Array(this.sheetFloats),
      sheetCount,
      sheetBasisData: new Float32Array(this.basisFloats),
      ...(sheetRevealData ? { sheetRevealData } : {}),
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

export interface BuildMambaSpatialContext {
  readonly bayId: 1 | 42;
  readonly clampedT: number;
  readonly isFeeder: boolean;
  readonly isFp8Mode: boolean;
  readonly computeIsFp8: boolean;
  readonly feederPreservedBf16: boolean;
  readonly activePart: string | null;
  readonly partHeadIdx: number | null;
  readonly activeHead: number;
  readonly activeGroup: number;
  readonly stateTokIdx: number;
  readonly explicitTokenPlayback: boolean;
  readonly activeCell: MambaCellRef | null;
  readonly activeFormula: MambaCellFormula | null;
  readonly focusSet: SheetFocusSet;
  readonly stepperStep?: number | null;
  readonly stepperActive?: boolean;
  readonly stepperSubProgress?: number;
  readonly termProgress?: number;
  readonly reducedMotion?: boolean;
  readonly sim: MambaSimResult;
  readonly atlas: MambaValueAtlas;
  readonly partCatalog: Readonly<Record<string, CutawayPartInfo>>;
}

function resolveSpatialMambaSpans(
  registry: ReadonlyMap<string, SheetGeomRecord>,
  span: MambaSheetSpan,
  displayedHead: number,
  displayedToken: number,
): readonly WorldSpan3D[] {
  if (span.tensor === 'conv_buf') {
    if (span.tokenIndex !== undefined && span.tokenIndex !== displayedToken) {
      return [];
    }
    const t0 = Math.max(0, Math.min(3, span.tapStart ?? 0));
    const t1 = Math.max(t0, Math.min(3, span.tapEnd ?? t0));
    const spans: WorldSpan3D[] = [];
    for (let slot = t0; slot <= t1; slot++) {
      const sheet = registry.get(`conv_buf:${slot}`);
      if (!sheet) continue;
      spans.push(spanFromSheetGrid(sheet, 0, 0, span.colStart, span.colEnd));
    }
    return spans;
  }

  if (span.tensor === 'h_prev' || span.tensor === 'h_curr') {
    if (span.headIndex !== undefined && span.headIndex !== displayedHead) {
      return [];
    }
    if (span.tokenIndex !== undefined && span.tokenIndex !== displayedToken) {
      return [];
    }
    const sheet = registry.get(span.tensor);
    return sheet ? [spanFromSheetGrid(sheet, span.rowStart, span.rowEnd, span.colStart, span.colEnd)] : [];
  }

  const key = span.tensor === 'x' ? `x:${span.headIndex ?? displayedHead}` : span.tensor;
  const sheet = registry.get(key);
  if (!sheet) return [];
  return [spanFromSheetGrid(sheet, span.rowStart, span.rowEnd, span.colStart, span.colEnd)];
}

/**
 * Spatial 3/4 Mamba-2 Reference Implementation.
 *
 * Meaningful 3D axis semantics:
 *  - X (left -> right): Forward Mamba-2 dataflow (Input -> RMSNorm -> Upright W_in -> xBC_pre -> Stepped 4-Tap Conv1D
 *    -> 16-Head x Fin Stack / B / C -> Raised Fixed FP32 SSM Pedestal [H_{t-1} ⇄ H_t + 64-Head Bank] -> Readout y
 *    -> Gate-First 8-Group Norm -> Upright W_out -> + Residual Y)
 *  - Y (elevation): Separates T-growing ground activation decks (y=0.04) inside 16-slot ghost capacity rails,
 *    upright learned weight walls (W_in, W_out), stepped 4-tap fixed BF16 Conv1D shelves (y=0.46..0.70),
 *    raised fixed-size FP32 SSM State Pedestal (y=1.02) with 3D arched recurrence loop, lifted active head fin (H_h),
 *    and the overhead emerald Residual Highway bridge (y=3.16)
 *  - Z (depth): Monotonically aligned with upright W_in columns (+Z front: Gate z bypass | center: Conv1D + SSM trunk
 *    | -Z rear-inner: Δt branch | -Z far rear: overhead Residual Highway) — zero crossing ribbons.
 */
export function buildMambaSpatialScene(ctx: BuildMambaSpatialContext): CutawaySceneGeometry {
  const {
    bayId,
    clampedT,
    isFeeder,
    isFp8Mode,
    computeIsFp8,
    feederPreservedBf16,
    activePart,
    partHeadIdx,
    activeHead,
    activeGroup,
    stateTokIdx,
    explicitTokenPlayback,
    activeCell,
    activeFormula,
    focusSet,
    stepperStep,
    stepperActive = false,
    stepperSubProgress = 0.25,
    termProgress,
    reducedMotion = false,
    sim,
    atlas,
    partCatalog,
  } = ctx;

  const revealParams: MambaRevealParams | null =
    stepperActive && !reducedMotion && stepperStep !== undefined && stepperStep !== null
      ? {
          stepperStep,
          subProgress: stepperSubProgress,
          clampedT,
          activeHead,
          activeGroup,
          stateTokIdx,
          activeCell,
          sim,
        }
      : null;

  const focusedAccumulation = resolveMambaFocusedAccumulation({
    sim,
    activeCell,
    stepperStep,
    stepperActive,
    reducedMotion,
    termProgress,
  });

  const b = new SpatialMambaBuilder(bayId, revealParams, focusSet.active);
  const hasCellTrace = focusSet.active;
  const hasFocus = activePart !== null || hasCellTrace;
  const partHeadFocus = partHeadIdx !== null && !hasCellTrace;

  const SHEET_THICK = 0.056;
  const BASE_Y = 0.04;
  const DECK_Y = 0.24;
  const PEDESTAL_Y = 1.02;

  const resolveCandidateHl = (
    candidate: SheetCandidateIdentity,
    fallback: number,
  ): number => resolveSheetHighlight(focusSet, candidate, fallback);

  const stageHighlight = (
    tensor: MambaTensorKey,
    partId: string,
    onPartPathway: boolean,
    goldWhenFp8Mode: boolean,
  ): number => {
    const fallback =
      activePart === partId
        ? SHEET_HL_ACTIVE
        : partHeadFocus && onPartPathway
          ? SHEET_HL_PATHWAY
          : isFp8Mode && goldWhenFp8Mode && !hasFocus
            ? SHEET_HL_BF16_GOLD
            : hasFocus && !onPartPathway
              ? SHEET_HL_DIMMED
              : SHEET_HL_NORMAL;
    return resolveCandidateHl({ tensor, isAggregate: true }, fallback);
  };

  // --- Key X Stage Coordinates (left -> right) ---
  const xInput = -8.05;
  const xRms = -6.55;
  const xWin = -5.00;
  const xXbcPre = -3.45;
  const xConvStart = -2.28;
  const convStepX = 0.24; // 4 fixed stepped shelves at x = -2.28, -2.04, -1.80, -1.56
  const xXStack = -0.35;
  const xHPrev = 1.25;
  const xHCurr = 2.82;
  const xY = 4.55;
  const xNorm = 5.92;
  const xWout = 7.22;
  const xResOut = 8.40;

  // --- Growing Sequence Extents vs. Fixed Persistent State Extents ---
  // Every sequence deck sits inside an invariant 16-token ghost capacity rail so scrubbing T=1..16
  // visibly grows the sequence stream while Conv1D (4 taps) and SSM State (H_{t-1}, H_t, 64-Head Bank) stay 100% FIXED.
  const tFrac = clampedT / 16;
  const ioMaxZ = 1.80;
  const ioLiveZ = Math.max(0.20, tFrac * ioMaxZ);
  const ioW = 1.02;

  const xSeqMaxX = 0.96;
  const xSeqLiveX = Math.max(0.14, tFrac * xSeqMaxX);

  const branchMaxZ = 1.00;
  const branchLiveZ = Math.max(0.16, tFrac * branchMaxZ);

  // Helper to anchor a growing-along-Z sequence sheet cleanly at the -Z start of its 16-slot ghost rail
  const zAnchoredCenter = (railCenterZ: number, railMaxZ: number, liveZ: number): number =>
    railCenterZ - railMaxZ * 0.5 + liveZ * 0.5;
  // Helper to anchor a growing-along-X sequence sheet cleanly at the -X start of its 16-slot ghost rail
  const xAnchoredCenter = (railCenterX: number, railMaxX: number, liveX: number): number =>
    railCenterX - railMaxX * 0.5 + liveX * 0.5;

  const addGhostSeqRail = (
    cx: number,
    cy: number,
    cz: number,
    ex: number,
    ez: number,
    axes: SheetBasisAxes,
    gridCols: number,
    gridRows: number,
    seed: number,
    hl: number,
  ) => {
    const railHl =
      hasCellTrace &&
      hl !== SHEET_HL_ACTIVE &&
      hl !== SHEET_HL_OPERAND_A &&
      hl !== SHEET_HL_OPERAND_B &&
      hl !== SHEET_HL_PATHWAY
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
      axes,
      gridCols,
      gridRows,
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
  // 1. STAGE 1: Input X, Pre-Layer RMSNorm & Overhead Emerald Residual Bridge
  // ==========================================================================
  const resInHl = stageHighlight('input', 'residual-in', activePart === 'residual-out', false);
  addGhostSeqRail(xInput, BASE_Y, 0, ioW, ioMaxZ, FRAME_XZ_AXES, 1, 16, 0.1, resInHl);
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
    seed: 1.1 + clampedT * 0.1,
    partId: 'residual-in',
    tensorKey: 'input',
    atlasSlice: atlas.input,
    registryKey: 'input',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  const resOutHl = stageHighlight('residual', 'residual-out', activePart === 'out-proj', false);
  addGhostSeqRail(xResOut, BASE_Y, 0, ioW, ioMaxZ, FRAME_XZ_AXES, 1, 16, 0.2, resOutHl);
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
    seed: 2.3 + clampedT * 0.1,
    partId: 'residual-out',
    tensorKey: 'residual',
    atlasSlice: atlas.residual,
    registryKey: 'residual',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Overhead Emerald Residual Bypass Bridge (arches cleanly over the rear of the Mamba-2 assembly)
  const bridgeY = 3.16;
  const bridgeZ = -3.72;
  const resCol =
    !hasCellTrace && (activePart === 'residual-in' || activePart === 'residual-out')
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
    resCol,
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
    resCol,
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
    resCol,
    true,
    14,
    true,
    0.30,
    0.32,
  );
  b.addGuideLine(xInput, BASE_Y, -ioMaxZ * 0.5, xInput, bridgeY, bridgeZ, 0.85, LINE_STYLE_GUIDE, 0.45);
  b.addGuideLine(xResOut, BASE_Y, -ioMaxZ * 0.5, xResOut, bridgeY, bridgeZ, 0.85, LINE_STYLE_GUIDE, 0.45);

  // Pre-Layer RMSNorm (eps = 1e-5)
  const rmsHl = stageHighlight('rmsnorm', 'rmsnorm', activePart === 'in-proj', feederPreservedBf16);
  addGhostSeqRail(xRms, BASE_Y, 0, ioW, ioMaxZ, FRAME_XZ_AXES, 1, 16, 0.3, rmsHl);
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
    fp8Flag: computeIsFp8 ? 1 : 0,
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
      [xInput + ioW * 0.5 + 0.03, BASE_Y + 0.04, 0],
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
  // 2. STAGE 2: Upright Learned Fused Weight Wall W_in [208 x 8] (FRAME_YZ_AXES)
  //    axisU = [0, 0, -1]: cols 0..63 (+Z front) = Gate z | 64..191 (center) = xBC | 192..207 (-Z rear) = Δt
  // ==========================================================================
  const winY = 1.56;
  const winSpanZ = 5.40;
  const winHeightY = 1.00;
  const winHl = stageHighlight('w_in', 'in-proj', false, feederPreservedBf16);

  b.addSheet({
    cx: xWin,
    cy: winY,
    cz: 0,
    ex: winSpanZ,
    ey: SHEET_THICK,
    ez: winHeightY,
    axes: FRAME_YZ_AXES,
    gridCols: sim.inProjDim,
    gridRows: sim.d,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: winHl,
    seed: 5.1 + bayId * 0.3,
    partId: 'in-proj',
    tensorKey: 'w_in',
    atlasSlice: atlas.wIn,
    registryKey: 'w_in',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // RMSNorm -> Upright W_in
  const winAct = !hasCellTrace && (activePart === 'in-proj' || activePart === 'rmsnorm');
  b.flow.addCubic3D(
    [xRms + ioW * 0.5 + 0.03, BASE_Y + 0.05, 0],
    [xRms + ioW * 0.5 + 0.48, winY * 0.65, 0],
    [xWin - 0.50, winY, 0],
    [xWin - 0.06, winY, 0],
    [0, 1, 0],
    winAct ? 0.105 : 0.08,
    winAct ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
    true,
    18,
    true,
  );

  // ==========================================================================
  // 3. STAGE 3: Three Parallel Branches Out of W_in (Monotonic +Z -> -Z, ZERO crossings!)
  //    Front (+Z): Gate z | Center (z=0): xBC_pre -> Stepped 4-Tap Conv1D | Rear (-Z): Δt
  // ==========================================================================
  const branchRibbonCol = (active: boolean, baseCol: FlowRibbonColor = RIBBON_QUIET): FlowRibbonColor =>
    active && !hasCellTrace ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : baseCol;

  // --- 3A. Front Bypass Corridor (+Z): Gate z [64, T] ---
  const gateX = 0.90;
  const gateZ = 2.68;
  const gateW = 2.60;
  const gateOnPath = activePart === 'gate-branch' || activePart === 'group-norm';
  const gateHl = stageHighlight('z', 'gate-branch', gateOnPath, feederPreservedBf16);
  addGhostSeqRail(gateX, BASE_Y, gateZ, gateW, branchMaxZ, FRAME_XZ_AXES, 1, 16, 0.4, gateHl);
  b.addSheet({
    cx: gateX,
    cy: BASE_Y,
    cz: zAnchoredCenter(gateZ, branchMaxZ, branchLiveZ),
    ex: gateW,
    ey: SHEET_THICK,
    ez: branchLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.dInner,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: gateHl,
    seed: 8.3 + clampedT * 0.15,
    partId: 'gate-branch',
    tensorKey: 'z',
    atlasSlice: atlas.z,
    registryKey: 'z',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // W_in (+Z front third, cols 0..63) -> Gate z
  b.flow.addCubic3D(
    [xWin + 0.05, winY, 1.95],
    [xWin + 1.15, winY * 0.72, gateZ],
    [gateX - gateW * 0.5 - 1.10, BASE_Y + 0.14, gateZ],
    [gateX - gateW * 0.5 - 0.05, BASE_Y + 0.05, gateZ],
    [0, 1, 0],
    gateOnPath && !hasCellTrace ? 0.10 : 0.075,
    gateOnPath && !hasCellTrace ? RIBBON_GATE_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_GATE_FLOW,
    true,
    20,
    true,
  );

  // --- 3B. Center Stream: xBC_pre [128, T] (grows along +X) -> Stepped 4-Tap Causal Conv1D (FIXED, ALWAYS BF16!) ---
  const convSpanZ = 2.55;
  const xbcPreHl = stageHighlight('xbc_pre', 'xbc-pre', activePart === 'conv1d' || activePart === 'in-proj', feederPreservedBf16);
  addGhostSeqRail(xXbcPre, BASE_Y, 0, convSpanZ, xSeqMaxX, FRAME_XZ_CH_Z_TOK_X, 1, 16, 0.5, xbcPreHl);
  b.addSheet({
    cx: xAnchoredCenter(xXbcPre, xSeqMaxX, xSeqLiveX),
    cy: BASE_Y,
    cz: 0,
    ex: convSpanZ,
    ey: SHEET_THICK,
    ez: xSeqLiveX,
    axes: FRAME_XZ_CH_Z_TOK_X,
    gridCols: sim.convDim,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: xbcPreHl,
    seed: 10.2 + clampedT * 0.12,
    partId: 'xbc-pre',
    tensorKey: 'xbc_pre',
    atlasSlice: atlas.xbcPre,
    registryKey: 'xbc_pre',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // W_in (center, cols 64..191) -> xBC_pre
  b.flow.addCubic3D(
    [xWin + 0.05, winY, -0.15],
    [xWin + 0.55, winY * 0.68, -0.15],
    [xXbcPre - xSeqMaxX * 0.5 - 0.35, BASE_Y + 0.20, 0],
    [xXbcPre - xSeqMaxX * 0.5 - 0.05, BASE_Y + 0.05, 0],
    [0, 1, 0],
    0.088,
    branchRibbonCol(activePart === 'xbc-pre' || activePart === 'conv1d' || activePart === 'in-proj'),
    true,
    18,
    true,
  );

  // FIXED Stepped 4-Tap Causal Conv1D Rolling Buffer (k=4, ALWAYS BF16, invariant size across T=1..16!)
  const convShelfBaseY = 0.44;
  const convShelfStepY = 0.072;
  const convTapThickX = 0.165;
  const convCenterX = xConvStart + 1.5 * convStepX;
  const convBaseFrameX0 = xConvStart - convTapThickX * 0.5 - 0.08;
  const convBaseFrameX1 = xConvStart + 3 * convStepX + convTapThickX * 0.5 + 0.08;

  // Grounded fixed-footprint architectural frame beneath Conv1D (contrasts with growing xBC_pre beside it!)
  const convFrameStyle = isFp8Mode ? LINE_STYLE_BF16_GOLD : LINE_STYLE_GUIDE;
  b.addBoxFrameXZ(
    convBaseFrameX0,
    -convSpanZ * 0.5 - 0.06,
    convBaseFrameX1,
    convSpanZ * 0.5 + 0.06,
    0.02,
    isFp8Mode ? 1.15 : 0.95,
    convFrameStyle,
    isFp8Mode ? 0.62 : 0.44,
  );

  for (let slot = 0; slot < sim.convKernel; slot++) {
    const sx = xConvStart + slot * convStepX;
    const sy = convShelfBaseY + slot * convShelfStepY;
    const lag = sim.convKernel - 1 - slot;
    const srcT = stateTokIdx - lag;
    const isZeroPadded = srcT < 0;

    const convFallback =
      activePart === 'conv1d'
        ? SHEET_HL_ACTIVE
        : hasFocus
          ? SHEET_HL_DIMMED
          : isZeroPadded
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    const slotHl = resolveCandidateHl(
      {
        tensor: 'conv_buf',
        tapIndex: slot,
        tokenIndex: stateTokIdx,
        isAggregate: false,
      },
      convFallback,
    );

    b.addSheet({
      cx: sx,
      cy: slotHl === SHEET_HL_ACTIVE ? sy + 0.055 : sy,
      cz: 0,
      ex: convSpanZ,
      ey: 0.052,
      ez: convTapThickX,
      axes: FRAME_XZ_CH_Z_TOK_X,
      gridCols: sim.convDim,
      gridRows: 1,
      role: SHEET_ROLE_KV_K,
      fp8Flag: 0, // NEVER quantizes to FP8 (NVIDIA §4.2)
      highlight: slotHl,
      seed: 12.0 + slot * 0.7 + stateTokIdx * 0.35,
      partId: 'conv1d',
      tensorKey: 'conv_buf',
      tokenIndex: slot,
      tapIndex: slot,
      stateTokenIndex: stateTokIdx,
      atlasSlice: atlas.convBufTaps[slot],
      registryKey: `conv_buf:${slot}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    // Subtle vertical stanchions at ends of first & last Conv1D shelf grounding the fixed prism
    if (slot === 0 || slot === sim.convKernel - 1) {
      const convStanchionAlpha = isFp8Mode ? 0.50 : 0.46;
      b.addGuideLine(sx, 0.02, -convSpanZ * 0.5, sx, sy - 0.03, -convSpanZ * 0.5, 0.85, convFrameStyle, convStanchionAlpha);
      b.addGuideLine(sx, 0.02, convSpanZ * 0.5, sx, sy - 0.03, convSpanZ * 0.5, 0.85, convFrameStyle, convStanchionAlpha);
    }
  }

  // xBC_pre -> Conv1D
  b.flow.addCubic3D(
    [xXbcPre + xSeqMaxX * 0.5 + 0.03, BASE_Y + 0.05, 0],
    [xXbcPre + xSeqMaxX * 0.5 + 0.25, BASE_Y + 0.14, 0],
    [xConvStart - convTapThickX * 0.5 - 0.25, convShelfBaseY, 0],
    [xConvStart - convTapThickX * 0.5 - 0.05, convShelfBaseY, 0],
    [0, 1, 0],
    0.085,
    branchRibbonCol(activePart === 'conv1d' || activePart === 'xbc-pre'),
    true,
    14,
    true,
  );

  // --- 3C. Rear-Inner Branch (-Z): Δt = softplus(dt_raw + dt_bias) [16, T] ---
  const dtX = -1.15;
  const dtZ = -2.42;
  const dtW = 1.62;
  const dtOnPath = activePart === 'dt-branch' || activePart === 'ssm-state-curr' || partHeadFocus;
  const dtHl = stageHighlight('dt', 'dt-branch', dtOnPath, feederPreservedBf16);
  addGhostSeqRail(dtX, BASE_Y, dtZ, dtW, branchMaxZ, FRAME_XZ_AXES, 1, 16, 0.6, dtHl);
  b.addSheet({
    cx: dtX,
    cy: BASE_Y,
    cz: zAnchoredCenter(dtZ, branchMaxZ, branchLiveZ),
    ex: dtW,
    ey: SHEET_THICK,
    ez: branchLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.nHeads,
    gridRows: clampedT,
    role: SHEET_ROLE_SOFTMAX,
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: dtHl,
    seed: 15.1 + clampedT * 0.2,
    partId: 'dt-branch',
    tensorKey: 'dt',
    atlasSlice: atlas.dt,
    registryKey: 'dt',
    cellFamily: CELL_MOE_AMBER,
  });

  // W_in (-Z rear edge, cols 192..207) -> Δt
  b.flow.addCubic3D(
    [xWin + 0.05, winY, -2.32],
    [xWin + 0.95, winY * 0.68, dtZ],
    [dtX - dtW * 0.5 - 0.95, BASE_Y + 0.16, dtZ],
    [dtX - dtW * 0.5 - 0.05, BASE_Y + 0.05, dtZ],
    [0, 1, 0],
    dtOnPath && !hasCellTrace ? 0.095 : 0.072,
    dtOnPath && !hasCellTrace ? RIBBON_DT_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_DT_FLOW,
    true,
    18,
    true,
  );

  // ==========================================================================
  // 4. STAGE 4: Post-SiLU Split -> 16-Head 3D x Fin Stack (Active H_h Lifted +Y) + B & C
  // ==========================================================================
  const xSpanZ = 2.40;
  const xPitchZ = xSpanZ / 15; // 0.16 spacing between 16 upright XY fins
  const xStepX = 0.025;
  const xRestY = 0.56;
  const xLiftY = 1.48;
  const xFinW = 0.72;
  const xFinH = Math.max(0.32, 0.30 + tFrac * 0.64); // grows vertically with T (0.34 -> 0.94)
  const xActiveW = 0.80;
  const xActiveH = Math.max(0.40, 0.38 + tFrac * 0.68);

  // Align head 0 at +Z front -> head 15 at -Z rear, matching Conv1D axisU = [0, 0, -1]!
  const xHeadPos = (h: number): { x: number; z: number } => ({
    x: xXStack + (h % 2) * xStepX,
    z: xSpanZ * 0.5 - h * xPitchZ,
  });

  for (let h = 0; h < sim.nHeads; h++) {
    const g = mambaGroupForHead(h);
    const pos = xHeadPos(h);
    const isActiveHead = h === activeHead;
    const partFallback =
      isActiveHead && (partHeadFocus || activePart === 'ssm-state-curr' || activePart === 'ssm-state-prev')
        ? SHEET_HL_ACTIVE
        : g === activeGroup && partHeadFocus
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : feederPreservedBf16
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;

    const hl = resolveCandidateHl(
      { tensor: 'x', headIndex: h, groupIndex: g, isAggregate: false },
      partFallback,
    );

    const isElevated =
      isActiveHead &&
      (hl === SHEET_HL_ACTIVE ||
        hl === SHEET_HL_OPERAND_A ||
        hl === SHEET_HL_OPERAND_B ||
        partHeadFocus ||
        !hasFocus);
    const cy = isElevated ? xLiftY : xRestY;
    const ex = isElevated ? xActiveW : xFinW;
    const ez = isElevated ? xActiveH : xFinH;

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
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: hl,
      seed: 19.0 + h * 0.57 + clampedT * 0.11,
      partId: `mamba-head-${h}`,
      headIndex: h,
      groupIndex: g,
      tensorKey: 'x',
      atlasSlice: atlas.xHeads[h],
      registryKey: `x:${h}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    if (isElevated) {
      const anchorActive =
        hl === SHEET_HL_ACTIVE || hl === SHEET_HL_OPERAND_A || hl === SHEET_HL_OPERAND_B;
      b.addGuideLine(
        pos.x,
        0.04,
        pos.z,
        pos.x,
        cy - ez * 0.5 - 0.03,
        pos.z,
        anchorActive ? 1.65 : 1.0,
        anchorActive ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
        anchorActive ? 0.90 : 0.50,
      );
    }
  }

  // Quiet base rail & 8-group ticks under the 16-head x fin stack
  const xPos0 = xHeadPos(0);
  const xPos15 = xHeadPos(15);
  b.addGuideLine(xXStack, 0.04, xPos0.z + 0.08, xXStack, 0.04, xPos15.z - 0.08, 1.0, LINE_STYLE_GUIDE, 0.48);

  // Conv1D -> Active x head fin
  const activeXPos = xHeadPos(activeHead);
  b.flow.addCubic3D(
    [xConvStart + 3 * convStepX + convTapThickX * 0.5 + 0.04, convShelfBaseY + 3 * convShelfStepY, 0],
    [xXStack - 0.72, 1.05, activeXPos.z * 0.5],
    [activeXPos.x - 0.65, xLiftY, activeXPos.z],
    [activeXPos.x - xActiveW * 0.5 - 0.05, xLiftY, activeXPos.z],
    [0, 1, 0],
    partHeadFocus ? 0.095 : 0.075,
    branchRibbonCol(activePart === 'conv1d' || partHeadFocus),
    true,
    18,
    true,
  );

  // Grouped B (8 groups, near H_t) & Grouped C (8 groups, near Readout y) — clean front-inner deck
  const bX = 1.25;
  const cX = 3.55;
  const bcZ = 1.48;
  const bcW = 1.42;
  const bcMaxZ = 0.88;
  const bcLiveZ = Math.max(0.16, tFrac * bcMaxZ);

  const bOnPath = activePart === 'b-proj' || activePart === 'ssm-state-curr' || partHeadFocus;
  const bHl = stageHighlight('b', 'b-proj', bOnPath, feederPreservedBf16);
  addGhostSeqRail(bX, BASE_Y, bcZ, bcW, bcMaxZ, FRAME_XZ_AXES, 1, 16, 0.7, bHl);
  b.addSheet({
    cx: bX,
    cy: BASE_Y,
    cz: zAnchoredCenter(bcZ, bcMaxZ, bcLiveZ),
    ex: bcW,
    ey: SHEET_THICK,
    ez: bcLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.nGroups * sim.dState,
    gridRows: clampedT,
    role: SHEET_ROLE_KV_K,
    fp8Flag: 0,
    highlight: bHl,
    seed: 24.3 + clampedT * 0.1,
    partId: 'b-proj',
    tensorKey: 'b',
    atlasSlice: atlas.bGroups,
    registryKey: 'b',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  const cOnPath = activePart === 'c-proj' || activePart === 'ssm-readout' || partHeadFocus;
  const cHl = stageHighlight('c', 'c-proj', cOnPath, feederPreservedBf16);
  addGhostSeqRail(cX, BASE_Y, bcZ, bcW, bcMaxZ, FRAME_XZ_AXES, 1, 16, 0.8, cHl);
  b.addSheet({
    cx: cX,
    cy: BASE_Y,
    cz: zAnchoredCenter(bcZ, bcMaxZ, bcLiveZ),
    ex: bcW,
    ey: SHEET_THICK,
    ez: bcLiveZ,
    axes: FRAME_XZ_AXES,
    gridCols: sim.nGroups * sim.dState,
    gridRows: clampedT,
    role: SHEET_ROLE_KV_V,
    fp8Flag: 0,
    highlight: cHl,
    seed: 25.9 + clampedT * 0.1,
    partId: 'c-proj',
    tensorKey: 'c',
    atlasSlice: atlas.cGroups,
    registryKey: 'c',
    cellFamily: CELL_SSM_CYAN,
  });

  // Conv1D -> B and Conv1D -> C ribbons routing cleanly underneath x fins along low floor lane
  b.flow.addCubic3D(
    [xConvStart + 3 * convStepX + convTapThickX * 0.5 + 0.04, convShelfBaseY + 0.08, 0.85],
    [xXStack + 0.10, 0.10, bcZ],
    [bX - bcW * 0.5 - 0.55, BASE_Y + 0.06, bcZ],
    [bX - bcW * 0.5 - 0.05, BASE_Y + 0.05, bcZ],
    [0, 1, 0],
    0.070,
    bOnPath && !hasCellTrace ? RIBBON_BC_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_BC_FLOW,
    true,
    18,
    true,
  );
  b.flow.addPolyline3D(
    [
      [xConvStart + 3 * convStepX + convTapThickX * 0.5 + 0.04, convShelfBaseY + 0.04, 1.15],
      [xXStack + 0.20, 0.05, bcZ + 0.54],
      [cX - bcW * 0.5 - 0.25, 0.05, bcZ + 0.54],
      [cX - bcW * 0.5 - 0.05, BASE_Y + 0.05, bcZ + 0.14],
    ],
    [0, 1, 0],
    0.066,
    cOnPath && !hasCellTrace ? RIBBON_BC_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_BC_FLOW,
    true,
    0.16,
    0.16,
    true,
  );

  // ==========================================================================
  // 5. STAGE 5 CENTERPIECE: Raised Fixed-Size FP32 SSM State Pedestal
  //    (H_{t-1} ⇄ H_t + 64-Head Production Bank — STRICTLY CONSTANT SIZE T=1..16!)
  // ==========================================================================
  const hSize = 1.14; // STRICTLY CONSTANT from T = 1 to T = 16
  const hZ = 0.04;
  const bankCenterX = 2.04;
  const bankCenterZ = -1.42;
  const bankCols = 8; // 8 B/C groups
  const bankRows = 8; // 8 production heads per group
  const bankPitchX = 0.22;
  const bankPitchZ = 0.125;
  const bankStartX = bankCenterX - ((bankCols - 1) * bankPitchX) * 0.5;
  const bankStartZ = bankCenterZ - ((bankRows - 1) * bankPitchZ) * 0.5;

  // Grounded Fixed-Memory Pedestal Plinth (y=0.02 base + 4 stanchions lifting to PEDESTAL_Y=1.02)
  const pedX0 = xHPrev - hSize * 0.5 - 0.16;
  const pedX1 = xHCurr + hSize * 0.5 + 0.16;
  const pedZ0 = bankCenterZ - 0.56;
  const pedZ1 = hZ + hSize * 0.5 + 0.16;
  b.addBoxFrameXZ(pedX0, pedZ0, pedX1, pedZ1, 0.02, 1.0, LINE_STYLE_GUIDE, 0.46);
  b.addBoxFrameXZ(pedX0, pedZ0, pedX1, pedZ1, PEDESTAL_Y - 0.05, 0.9, LINE_STYLE_GUIDE, 0.40);
  for (const [sx, sz] of [
    [pedX0, pedZ0],
    [pedX1, pedZ0],
    [pedX1, pedZ1],
    [pedX0, pedZ1],
  ] as const) {
    b.addGuideLine(sx, 0.02, sz, sx, PEDESTAL_Y - 0.05, sz, 0.9, LINE_STYLE_GUIDE, 0.46);
  }

  // Prior SSM State H_{t-1} (FIXED 4x4 mini / 64x128 prod FP32)
  const hPrevFallback =
    activePart === 'ssm-state-prev'
      ? SHEET_HL_ACTIVE
      : activePart === 'ssm-state-curr' || partHeadFocus
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;
  const hPrevHl = resolveCandidateHl(
    {
      tensor: 'h_prev',
      headIndex: activeHead,
      groupIndex: activeGroup,
      tokenIndex: stateTokIdx,
      isAggregate: false,
    },
    hPrevFallback,
  );
  b.addSheet({
    cx: xHPrev,
    cy: hPrevHl === SHEET_HL_ACTIVE ? PEDESTAL_Y + 0.05 : PEDESTAL_Y,
    cz: hZ,
    ex: hSize,
    ey: SHEET_THICK,
    ez: hSize,
    axes: FRAME_XZ_AXES,
    gridCols: sim.dState,
    gridRows: sim.dHead,
    role: SHEET_ROLE_SCORES,
    fp8Flag: 0, // ALWAYS FP32
    highlight: hPrevHl,
    seed: 31.1 + activeHead * 0.4 + stateTokIdx * 0.6,
    partId: 'ssm-state-prev',
    headIndex: activeHead,
    groupIndex: activeGroup,
    tensorKey: 'h_prev',
    tokenIndex: stateTokIdx,
    atlasSlice: atlas.hPrev,
    registryKey: 'h_prev',
    cellFamily: CELL_SSM_CYAN,
  });

  // Current SSM State H_t (FIXED 4x4 mini / 64x128 prod FP32)
  const hCurrFallback =
    activePart === 'ssm-state-curr'
      ? SHEET_HL_ACTIVE
      : partHeadFocus || activePart === 'ssm-state' || activePart === 'ssm-readout'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;
  const hCurrHl = resolveCandidateHl(
    {
      tensor: 'h_curr',
      headIndex: activeHead,
      groupIndex: activeGroup,
      tokenIndex: stateTokIdx,
      isAggregate: false,
    },
    hCurrFallback,
  );
  b.addSheet({
    cx: xHCurr,
    cy: hCurrHl === SHEET_HL_ACTIVE ? PEDESTAL_Y + 0.05 : PEDESTAL_Y,
    cz: hZ,
    ex: hSize,
    ey: SHEET_THICK,
    ez: hSize,
    axes: FRAME_XZ_AXES,
    gridCols: sim.dState,
    gridRows: sim.dHead,
    role: SHEET_ROLE_KV_V,
    fp8Flag: 0, // ALWAYS FP32
    highlight: hCurrHl,
    seed: 33.7 + activeHead * 0.4 + stateTokIdx * 0.9,
    partId: 'ssm-state-curr',
    headIndex: activeHead,
    groupIndex: activeGroup,
    tensorKey: 'h_curr',
    tokenIndex: stateTokIdx,
    atlasSlice: atlas.hCurr,
    registryKey: 'h_curr',
    cellFamily: CELL_SSM_CYAN,
  });

  // Forward recurrence arrow: H_{t-1} --(dA·H_{t-1} + Δt·x·B)--> H_t
  const ssmRecurrenceActive =
    !hasCellTrace ||
    activeFormula?.cell.tensor === 'h_curr' ||
    activeFormula?.cell.tensor === 'h_prev';
  b.flow.addPolyline3D(
    [
      [xHPrev + hSize * 0.5 + 0.04, PEDESTAL_Y + 0.04, hZ],
      [xHCurr - hSize * 0.5 - 0.05, PEDESTAL_Y + 0.04, hZ],
    ],
    [0, 1, 0],
    0.105,
    !hasCellTrace ? RIBBON_ACTIVE : ssmRecurrenceActive ? RIBBON_PATHWAY_SOFT : RIBBON_DIMMED,
    true,
    0.18,
    0.18,
    false,
  );

  // 3D Arched In-Place Feedback Loop (H_t arches UP over the pedestal back into H_{t-1})
  b.flow.addCubic3D(
    [xHCurr, PEDESTAL_Y + 0.06, hZ - 0.18],
    [xHCurr - 0.10, PEDESTAL_Y + 0.88, hZ - 0.18],
    [xHPrev + 0.10, PEDESTAL_Y + 0.88, hZ - 0.18],
    [xHPrev, PEDESTAL_Y + 0.08, hZ - 0.18],
    [0, 0, 1],
    0.088,
    ssmRecurrenceActive ? RIBBON_SSM_LOOP : RIBBON_DIMMED,
    true,
    24,
    true,
    0.18,
    0.18,
  );

  // Feed ribbons into H_t on pedestal: from active x head fin, B, and Δt
  b.flow.addCubic3D(
    [activeXPos.x + xActiveW * 0.5 + 0.04, xLiftY, activeXPos.z],
    [xHPrev - 0.15, Math.max(xLiftY + 0.18, PEDESTAL_Y + 0.55), activeXPos.z * 0.4 + hZ * 0.6],
    [xHCurr - 0.35, PEDESTAL_Y + 0.45, hZ + 0.12],
    [xHCurr - 0.18, PEDESTAL_Y + 0.08, hZ + 0.18],
    [0, 1, 0],
    partHeadFocus ? 0.095 : 0.072,
    partHeadFocus ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
    true,
    20,
    true,
  );
  b.flow.addCubic3D(
    [bX + 0.15, BASE_Y + 0.06, bcZ - bcMaxZ * 0.5],
    [bX + 0.45, PEDESTAL_Y * 0.55, bcZ - 0.45],
    [xHCurr - 0.18, PEDESTAL_Y + 0.06, hZ + hSize * 0.5 + 0.22],
    [xHCurr - 0.10, PEDESTAL_Y + 0.04, hZ + hSize * 0.5 + 0.04],
    [0, 1, 0],
    0.072,
    bOnPath && !hasCellTrace ? RIBBON_BC_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_BC_FLOW,
    true,
    18,
    true,
  );
  b.flow.addCubic3D(
    [dtX + dtW * 0.5 + 0.04, BASE_Y + 0.06, dtZ],
    [xHCurr + 0.25, BASE_Y + 0.16, dtZ],
    [xHCurr + 0.35, PEDESTAL_Y + 0.14, hZ - hSize * 0.5 - 0.45],
    [xHCurr + 0.22, PEDESTAL_Y + 0.05, hZ - hSize * 0.5 - 0.04],
    [0, 1, 0],
    0.072,
    dtOnPath && !hasCellTrace ? RIBBON_DT_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_DT_FLOW,
    true,
    20,
    true,
  );

  // 64-Head Fixed Production SSM Bank (8 groups x 8 heads = 64 micro-tiles on rear half of pedestal, ALWAYS FP32!)
  const ssmCellActive =
    hasCellTrace &&
    (resolveSheetFocusRole(focusSet, {
      tensor: 'h_curr',
      headIndex: activeHead,
      groupIndex: activeGroup,
      tokenIndex: stateTokIdx,
      isAggregate: false,
    }) !== null ||
      resolveSheetFocusRole(focusSet, {
        tensor: 'h_prev',
        headIndex: activeHead,
        groupIndex: activeGroup,
        tokenIndex: stateTokIdx,
        isAggregate: false,
      }) !== null);

  for (let g = 0; g < bankCols; g++) {
    for (let r = 0; r < bankRows; r++) {
      const tileIdx = g * 8 + r;
      const miniHead = g * 2 + Math.floor(r / 4);
      let tileHl = SHEET_HL_NORMAL;
      if (hasCellTrace) {
        tileHl = ssmCellActive && miniHead === activeHead ? SHEET_HL_PATHWAY : SHEET_HL_DIMMED;
      } else if (activePart === 'ssm-state' || (partHeadFocus && miniHead === activeHead)) {
        tileHl = SHEET_HL_ACTIVE;
      } else if (partHeadFocus && g === activeGroup) {
        tileHl = SHEET_HL_PATHWAY;
      } else if (hasFocus) {
        tileHl = SHEET_HL_DIMMED;
      }

      b.addSheet({
        cx: bankStartX + g * bankPitchX,
        cy: tileHl === SHEET_HL_ACTIVE ? PEDESTAL_Y + 0.02 : PEDESTAL_Y - 0.04,
        cz: bankStartZ + r * bankPitchZ,
        ex: 0.175,
        ey: 0.046,
        ez: 0.098,
        axes: FRAME_XZ_AXES,
        gridCols: 1,
        gridRows: 1,
        role: SHEET_ROLE_KV_V,
        fp8Flag: 0, // ALWAYS FP32!
        highlight: tileHl,
        seed: clampedT * 1.73 + stateTokIdx * 0.41 + tileIdx * 0.29 + bayId * 0.11,
        partId: 'ssm-state',
        headIndex: miniHead,
        groupIndex: g,
        atlasSlice: atlas.ssmBankTiles[tileIdx],
        cellFamily: CELL_SSM_CYAN,
      });
    }
  }

  // Subtle connector from active H_t to the 64-head bank
  b.addGuideLine(
    xHCurr,
    PEDESTAL_Y - 0.02,
    hZ - hSize * 0.5 - 0.04,
    bankCenterX,
    PEDESTAL_Y - 0.04,
    bankCenterZ + 0.48,
    0.9,
    LINE_STYLE_GUIDE,
    0.45,
  );

  // ==========================================================================
  // 6. STAGE 6: Readout y (= H_t·C + D·x) -> Gate-First 8-Grp Norm -> Upright W_out -> + Residual
  // ==========================================================================
  const outSpanZ = 2.48;
  const yOnPath = activePart === 'ssm-readout' || activePart === 'group-norm' || partHeadFocus;
  const yHl = stageHighlight('y', 'ssm-readout', yOnPath, feederPreservedBf16);
  addGhostSeqRail(xY, DECK_Y, 0.04, outSpanZ, xSeqMaxX, FRAME_XZ_CH_Z_TOK_X, 1, 16, 0.9, yHl);
  b.addSheet({
    cx: xAnchoredCenter(xY, xSeqMaxX, xSeqLiveX),
    cy: DECK_Y,
    cz: 0.04,
    ex: outSpanZ,
    ey: SHEET_THICK,
    ez: xSeqLiveX,
    axes: FRAME_XZ_CH_Z_TOK_X,
    gridCols: sim.dInner,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: yHl,
    seed: 41.2 + clampedT * 0.1,
    partId: 'ssm-readout',
    tensorKey: 'y',
    atlasSlice: atlas.y,
    registryKey: 'y',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // H_t -> Readout y & C -> Readout y
  b.flow.addCubic3D(
    [xHCurr + hSize * 0.5 + 0.04, PEDESTAL_Y + 0.04, hZ],
    [xHCurr + hSize * 0.5 + 0.42, PEDESTAL_Y * 0.72, hZ],
    [xY - xSeqMaxX * 0.5 - 0.35, DECK_Y + 0.14, 0.04],
    [xY - xSeqMaxX * 0.5 - 0.05, DECK_Y + 0.05, 0.04],
    [0, 1, 0],
    0.088,
    branchRibbonCol(yOnPath),
    true,
    18,
    true,
  );
  b.flow.addCubic3D(
    [cX + bcW * 0.5 + 0.04, BASE_Y + 0.06, bcZ],
    [cX + bcW * 0.5 + 0.30, DECK_Y * 0.65, bcZ * 0.75],
    [xY - xSeqMaxX * 0.5 - 0.25, DECK_Y + 0.06, 0.58],
    [xY - xSeqMaxX * 0.5 - 0.05, DECK_Y + 0.05, 0.58],
    [0, 1, 0],
    0.072,
    cOnPath && !hasCellTrace ? RIBBON_BC_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_BC_FLOW,
    true,
    16,
    true,
  );

  // Gate-First 8-Group RMSNorm (y_norm: norm_before_gate = False)
  const normOnPath = activePart === 'group-norm' || activePart === 'out-proj';
  const normHl = stageHighlight('y_norm', 'group-norm', normOnPath, feederPreservedBf16);
  addGhostSeqRail(xNorm, DECK_Y, 0.04, outSpanZ, xSeqMaxX, FRAME_XZ_CH_Z_TOK_X, 1, 16, 1.0, normHl);
  b.addSheet({
    cx: xAnchoredCenter(xNorm, xSeqMaxX, xSeqLiveX),
    cy: DECK_Y,
    cz: 0.04,
    ex: outSpanZ,
    ey: SHEET_THICK,
    ez: xSeqLiveX,
    axes: FRAME_XZ_CH_Z_TOK_X,
    gridCols: sim.dInner,
    gridRows: clampedT,
    role: SHEET_ROLE_NORM,
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: normHl,
    seed: 44.6 + clampedT * 0.1,
    partId: 'group-norm',
    tensorKey: 'y_norm',
    atlasSlice: atlas.yNorm,
    registryKey: 'y_norm',
    cellFamily: CELL_MOE_AMBER,
  });

  // y -> Gate→8-Grp Norm
  b.flow.addPolyline3D(
    [
      [xY + xSeqMaxX * 0.5 + 0.04, DECK_Y + 0.04, 0.04],
      [xNorm - xSeqMaxX * 0.5 - 0.05, DECK_Y + 0.04, 0.04],
    ],
    [0, 1, 0],
    0.088,
    branchRibbonCol(normOnPath),
    true,
    0.16,
    0.16,
    false,
  );

  // Front bypass Gate z -> Gate→8-Grp Norm (sweeps cleanly up from front corridor into +Z side!)
  b.flow.addCubic3D(
    [gateX + gateW * 0.5 + 0.04, BASE_Y + 0.06, gateZ],
    [xNorm - 0.45, BASE_Y + 0.10, gateZ],
    [xNorm, DECK_Y + 0.12, outSpanZ * 0.5 + 0.65],
    [xNorm, DECK_Y + 0.05, outSpanZ * 0.5 + 0.05],
    [0, 1, 0],
    gateOnPath && !hasCellTrace ? 0.10 : 0.075,
    gateOnPath && !hasCellTrace ? RIBBON_GATE_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_GATE_FLOW,
    true,
    20,
    true,
  );

  // Upright Learned Output Weight Wall W_out [4,096 -> 2,688] (FRAME_YZ_AXES)
  const woutY = 1.48;
  const woutSpanZ = 4.20;
  const woutHeightY = 0.96;
  const woutOnPath = activePart === 'out-proj' || activePart === 'residual-out';
  const woutHl = stageHighlight('w_out', 'out-proj', woutOnPath, feederPreservedBf16);
  b.addSheet({
    cx: xWout,
    cy: woutY,
    cz: 0,
    ex: woutSpanZ,
    ey: SHEET_THICK,
    ez: woutHeightY,
    axes: FRAME_YZ_AXES,
    gridCols: sim.dInner,
    gridRows: sim.d,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: woutHl,
    seed: 51.3 + bayId * 0.4,
    partId: 'out-proj',
    tensorKey: 'w_out',
    atlasSlice: atlas.wOut,
    registryKey: 'w_out',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Gate→8-Grp Norm -> Upright W_out
  const outAct = !hasCellTrace && (activePart === 'out-proj' || activePart === 'group-norm');
  b.flow.addCubic3D(
    [xNorm + xSeqMaxX * 0.5 + 0.04, DECK_Y + 0.05, 0.04],
    [xNorm + xSeqMaxX * 0.5 + 0.35, woutY * 0.72, 0.04],
    [xWout - 0.35, woutY, 0],
    [xWout - 0.06, woutY, 0],
    [0, 1, 0],
    outAct ? 0.10 : 0.078,
    outAct ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
    true,
    16,
    true,
  );

  // Upright W_out -> + Residual Y
  b.flow.addCubic3D(
    [xWout + 0.06, woutY, 0],
    [xWout + 0.46, woutY * 0.74, 0],
    [xResOut - ioW * 0.5 - 0.32, BASE_Y + 0.15, 0],
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
  // 7. TEMPORAL RECURRENCE TOKEN HIGHLIGHT & LIVE 3D CELL DEPENDENCY THREADS
  // ==========================================================================
  // When user is stepping temporal recurrence t (without an overriding cell trace), subtly highlight
  // token row `stateTokIdx` on x[activeHead], B, C, and Δt plus the sliding 4-tap window on xBC_pre!
  if (explicitTokenPlayback && !activeFormula) {
    const xbcRec = b.sheetRegistry.get('xbc_pre');
    if (xbcRec) {
      const winStart = Math.max(0, stateTokIdx - (sim.convKernel - 1));
      const winSpan = spanFromSheetGrid(xbcRec, winStart, stateTokIdx, 0, xbcRec.gridCols - 1);
      drawCellHighlight3D(b.flow, winSpan, TOKEN_ROW_FILL, TOKEN_ROW_FRAME, 0.018, true);
    }
    const seqKeys = [`x:${activeHead}`, 'b', 'c', 'dt'] as const;
    for (const k of seqKeys) {
      const rec = b.sheetRegistry.get(k);
      if (!rec) continue;
      const rowSpan = spanFromSheetGrid(rec, stateTokIdx, stateTokIdx, 0, rec.gridCols - 1);
      drawCellHighlight3D(b.flow, rowSpan, TOKEN_ROW_FILL, TOKEN_ROW_FRAME, 0.018, true);
    }
  }

  let residualCarryInfo: { readonly ch: number; readonly tok: number; readonly val: number } | null = null;
  let popoverTargetWorld: Vec3 | null = null;
  const popoverSources: {
    readonly world: readonly [number, number, number];
    readonly role: 'indigo' | 'teal';
  }[] = [];

  if (activeFormula) {
    const targetSpans = resolveSpatialMambaSpans(
      b.sheetRegistry,
      activeFormula.targetSpan,
      activeHead,
      stateTokIdx,
    );
    const primaryTarget = targetSpans[0];

    if (primaryTarget) {
      popoverTargetWorld = primaryTarget.frontPoint;
      for (const ts of targetSpans) {
        drawCellHighlight3D(b.flow, ts, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.024, true);
      }

      for (const up of activeFormula.upstreamSpans) {
        const upSpans = resolveSpatialMambaSpans(b.sheetRegistry, up, activeHead, stateTokIdx);
        if (upSpans.length === 0) continue;

        const srcRole: 'indigo' | 'teal' = up.role === 'operand_a' ? 'indigo' : 'teal';

        // Focused visual routing for Input X -> + Residual Y along the overhead Residual Highway bridge!
        if (activeFormula.cell.tensor === 'residual' && up.tensor === 'input') {
          const upSpan = upSpans[0];
          popoverSources.push({ world: upSpan.frontPoint, role: srcRole });
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
            [primaryTarget.frontPoint[0], threadBridgeY * 0.76, -ioMaxZ * 0.5 - 0.55],
            primaryTarget.frontPoint,
            [0, 1, 0],
            0.056,
            THREAD_OP_A,
            true,
            18,
            true,
            0.16,
            0.16,
          );
          const tok = Math.max(0, Math.min(clampedT - 1, activeFormula.cell.row));
          const ch = Math.max(0, Math.min(sim.d - 1, activeFormula.cell.col));
          residualCarryInfo = { ch, tok, val: sim.X[ch * clampedT + tok] };
          continue;
        }

        // For multi-tap Conv1D spans (e.g. 4 causal taps -> post-SiLU x), highlight all 4 taps and thread from newest tap
        if (upSpans.length > 1) {
          popoverSources.push({ world: upSpans[upSpans.length - 1].frontPoint, role: srcRole });
          const isA = up.role === 'operand_a';
          for (let i = 0; i < upSpans.length; i++) {
            if (i === upSpans.length - 1) {
              drawOperandDependency3D(b.flow, upSpans[i], primaryTarget, isA);
            } else {
              drawCellHighlight3D(
                b.flow,
                upSpans[i],
                isA ? CELL_OP_A_FILL : CELL_OP_B_FILL,
                isA ? CELL_OP_A_FRAME : CELL_OP_B_FRAME,
                0.020,
                true,
              );
            }
          }
          continue;
        }

        popoverSources.push({ world: upSpans[0].frontPoint, role: srcRole });
        drawOperandDependency3D(b.flow, upSpans[0], primaryTarget, up.role === 'operand_a');
      }
    }
  }

  const cellPopover = buildMambaCellPopover(
    activeFormula,
    popoverTargetWorld,
    popoverSources,
    focusedAccumulation,
  );
  const activeCellGrids = buildMambaActiveCellGrids(
    b.sheetRegistry,
    sim,
    clampedT,
    activeHead,
    stateTokIdx,
    activeFormula,
    revealParams,
    focusedAccumulation,
  );

  // ==========================================================================
  // 8. 3D EXTERIOR LABELS & ARCHITECTURAL DIMENSION CALIPERS
  // ==========================================================================
  const headTag = formatMambaHeadId(activeHead);
  const absTokIdx = sim.startTokenOffset + stateTokIdx;
  const tokLabel = sim.tokenLabels[stateTokIdx] ?? `t${absTokIdx}`;
  const tokTag =
    stateTokIdx < clampedT - 1 || explicitTokenPlayback || sim.hasInitialState
      ? ` · t${absTokIdx} "${tokLabel}"`
      : '';

  // V1 Label Hygiene: anchor Conv1D (+Z front exterior), Δt (-X left exterior), x: H00–H15
  // (cleanly above lifted active fin), RMSNorm (-Z rear), and H_{t-1} / H_t / 64-Head Bank
  // around separate exterior perimeters with zero overlap across T=1..16 and H00..H15.
  b.landmarks.push(
    { id: 'm-res-in', label: `Input (2,688 · T=${clampedT})`, world: [xInput, 0.16, ioMaxZ * 0.5 + 0.24], partId: 'residual-in', align: 'bottom', tier: 'primary' },
    { id: 'm-rmsnorm', label: 'RMSNorm', world: [xRms, 0.16, -ioMaxZ * 0.5 - 0.26], partId: 'rmsnorm', align: 'top', tier: 'primary', tone: feederPreservedBf16 ? 'gold' : 'default' },
    {
      id: 'm-in-proj',
      label: computeIsFp8 ? 'W_in · FP8' : 'W_in (10,304)',
      world: [xWin, winY + winHeightY * 0.5 + 0.22, 0],
      partId: 'in-proj',
      align: 'top',
      tier: 'primary',
      tone: feederPreservedBf16 ? 'gold' : 'default',
    },
    { id: 'm-gate', label: `Gate z (T=${clampedT})`, world: [gateX, 0.16, gateZ + branchMaxZ * 0.5 + 0.24], partId: 'gate-branch', align: 'bottom', tier: 'primary' },
    {
      id: 'm-conv1d',
      label: isFp8Mode ? `Conv1D (k=4 Fixed · BF16${tokTag})` : `Conv1D (k=4 Fixed${tokTag})`,
      world: [convCenterX - 0.04, 0.16, convSpanZ * 0.5 + 0.32],
      partId: 'conv1d',
      align: 'bottom',
      tier: 'primary',
      tone: isFp8Mode ? 'gold' : 'cyan',
    },
    { id: 'm-dt', label: `Δt (T=${clampedT})`, world: [dtX - dtW * 0.5 - 0.16, 0.18, dtZ], partId: 'dt-branch', align: 'left', tier: 'primary' },
    {
      id: 'm-x-heads',
      label: `x: H00–H15 (${headTag}·G${activeGroup})`,
      world: [activeXPos.x + 0.06, Math.max(2.38, xLiftY + xActiveH * 0.5 + 0.44), activeXPos.z],
      partId: `mamba-head-${activeHead}`,
      align: 'top',
      tier: 'primary',
    },
    { id: 'm-b', label: 'B (8 grps)', world: [bX, 0.16, bcZ + bcMaxZ * 0.5 + 0.22], partId: 'b-proj', align: 'bottom', tier: 'primary' },
    { id: 'm-c', label: 'C (8 grps)', world: [cX, 0.16, bcZ + bcMaxZ * 0.5 + 0.22], partId: 'c-proj', align: 'bottom', tier: 'primary' },
    {
      id: 'm-hprev',
      label: sim.hasInitialState ? `Hₜ₋₁ (${headTag} · Carried)` : `Hₜ₋₁ (${headTag})`,
      world: [xHPrev - 0.23, PEDESTAL_Y + 0.14, hZ + hSize * 0.5 + 0.18],
      partId: 'ssm-state-prev',
      align: 'bottom',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'm-hcurr',
      label: `Hₜ (${headTag} · Fixed${tokTag})`,
      world: [xHCurr + 0.26, PEDESTAL_Y + 0.14, hZ + hSize * 0.5 + 0.23],
      partId: 'ssm-state-curr',
      align: 'bottom',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'm-ssm-bank',
      label: '64-Head Bank (46.0 MiB FP32)',
      world: [bankCenterX + 0.10, PEDESTAL_Y + 0.24, bankCenterZ - 0.62],
      partId: 'ssm-state',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    { id: 'm-y', label: 'y = Hₜ·C + D·x', world: [xY, DECK_Y + 0.14, -outSpanZ * 0.5 - 0.20], partId: 'ssm-readout', align: 'top', tier: 'primary' },
    { id: 'm-gnorm', label: 'Gate→8-Grp Norm', world: [xNorm, DECK_Y + 0.14, outSpanZ * 0.5 + 0.22], partId: 'group-norm', align: 'bottom', tier: 'primary' },
    {
      id: 'm-out-proj',
      label: computeIsFp8 ? 'W_out · FP8' : 'W_out',
      world: [xWout, woutY + woutHeightY * 0.5 + 0.20, 0],
      partId: 'out-proj',
      align: 'top',
      tier: 'primary',
      tone: feederPreservedBf16 ? 'gold' : 'default',
    },
    { id: 'm-res-out', label: '+ Residual', world: [xResOut, 0.16, ioMaxZ * 0.5 + 0.24], partId: 'residual-out', align: 'bottom', tier: 'primary' },
    { id: 'm-res-bridge', label: 'Residual Highway (2,688)', world: [0.3, bridgeY + 0.22, bridgeZ], partId: 'residual-in', align: 'top', tier: 'primary' },
    ...(residualCarryInfo
      ? [
          {
            id: 'm-res-carry',
            label: `← Input X[${residualCarryInfo.ch}, t${sim.startTokenOffset + residualCarryInfo.tok}] = ${fmtSigned(residualCarryInfo.val)}`,
            world: [xResOut - 0.18, bridgeY * 0.80, bridgeZ * 0.62] as Vec3,
            partId: 'residual-in',
            align: 'left' as const,
            tier: 'primary' as const,
            tone: 'cyan' as const,
          },
        ]
      : []),
    // Progressive detail labels (revealed on hover, selection, or close zoom)
    { id: 'm-xbc-pre-sec', label: `xBC_pre (T=${clampedT})`, world: [xXbcPre - 0.10, 0.16, -convSpanZ * 0.5 - 0.24], partId: 'xbc-pre', align: 'top', tier: 'detail', tone: 'muted' },
    { id: 'm-loop-sec', label: 'In-place: Hₜ = dA·Hₜ₋₁ + Δt·x·B', world: [(xHPrev + xHCurr) * 0.5, PEDESTAL_Y + 0.96, hZ - 0.18], partId: 'ssm-state-curr', align: 'top', tier: 'detail', tone: 'cyan' },
    { id: 'm-wout-shape', label: computeIsFp8 ? '4,096 → 2,688 · FP8' : '4,096 → 2,688', world: [xWout, woutY - woutHeightY * 0.5 - 0.14, 0], partId: 'out-proj', align: 'bottom', tier: 'detail', tone: feederPreservedBf16 ? 'gold' : 'default' },
  );

  if (isFeeder) {
    b.landmarks.push({
      id: 'm-feeder-note',
      label: 'Bay 42 Feeder · Preserved in BF16 (NVIDIA §4.2)',
      world: [xWin, winY + winHeightY * 0.5 + 0.52, -1.85],
      align: 'top',
      tier: 'primary',
      tone: 'gold',
    });
  }

  // Calipers highlighting fixed SSM state vs growing T sequence stream
  b.calipers.push(
    {
      id: 'cal-ssm-fixed',
      aWorld: [pedX0, 0.04, pedZ1 + 0.10],
      bWorld: [pedX1, 0.04, pedZ1 + 0.10],
      label: 'Fixed-size FP32 SSM State (46.0 MiB / 23 bays · +1.08 MiB Conv1D = 47.1 MiB)',
      partIds: ['ssm-state-prev', 'ssm-state-curr', 'ssm-state'],
      minPixels: 110,
      offsetPx: 10,
    },
    {
      id: 'cal-seq-growing',
      aWorld: [xXbcPre - xSeqMaxX * 0.5, 0.04, convSpanZ * 0.5 + 0.12],
      bWorld: [xXbcPre - xSeqMaxX * 0.5 + xSeqLiveX, 0.04, convSpanZ * 0.5 + 0.12],
      label: `Sequence T = ${clampedT} (grows)`,
      partIds: ['xbc-pre', 'conv1d'],
      minPixels: 42,
      offsetPx: 8,
    },
  );

  // ==========================================================================
  // 9. TIGHT PER-OPERATION CAMERA FOCUS BOUNDS (Steps 0..5)
  // ==========================================================================
  const reg = (k: string) => b.sheetRegistry.get(k);
  const convTaps = [0, 1, 2, 3].map((slot) => reg(`conv_buf:${slot}`));
  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('input'), reg('rmsnorm')], 0.48, 0.28, 0.48, MAMBA_SPATIAL_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('rmsnorm'), reg('w_in'), reg('z'), reg('xbc_pre'), reg('dt')], 0.48, 0.28, 0.48, MAMBA_SPATIAL_FIT_BOUNDS),
    2: computeSheetGroupBounds([reg('xbc_pre'), ...convTaps, reg(`x:${activeHead}`), reg('b'), reg('c')], 0.48, 0.28, 0.48, MAMBA_SPATIAL_FIT_BOUNDS),
    3: computeSheetGroupBounds([reg('h_prev'), reg('h_curr'), reg(`x:${activeHead}`), reg('b'), reg('dt')], 0.52, 0.38, 0.52, MAMBA_SPATIAL_FIT_BOUNDS),
    4: computeSheetGroupBounds([reg('h_curr'), reg('c'), reg(`x:${activeHead}`), reg('y')], 0.48, 0.30, 0.48, MAMBA_SPATIAL_FIT_BOUNDS),
    5: computeSheetGroupBounds([reg('y'), reg('z'), reg('y_norm'), reg('w_out'), reg('residual')], 0.52, 0.36, 0.52, MAMBA_SPATIAL_FIT_BOUNDS),
  };

  const packed = b.pack();
  const mambaLayerCount = NEMOTRON_SPEC.bays.filter((x) => x.kind === 'M').length;
  const prodConvDim =
    NEMOTRON_SPEC.mamba.heads * NEMOTRON_SPEC.mamba.channels +
    2 * NEMOTRON_SPEC.mamba.groups * NEMOTRON_SPEC.mamba.state;
  const ssmBytesAll23 =
    mambaLayerCount *
    NEMOTRON_SPEC.mamba.heads *
    NEMOTRON_SPEC.mamba.channels *
    NEMOTRON_SPEC.mamba.state *
    4;
  const conv1dBytesAll23 =
    mambaLayerCount *
    prodConvDim *
    NEMOTRON_SPEC.mamba.kernel *
    2;

  return {
    kind: 'cutaway_mamba',
    bayId,
    ...packed,
    calipers: b.calipers,
    valueAtlas: atlas,
    mambaSim: sim,
    activeFormula,
    ...(focusSet.active ? { activeOpPartIds: b.activeOpPartIds } : {}),
    cellPopover,
    activeCellGrids,
    focusedAccumulation,
    fitBounds: MAMBA_SPATIAL_FIT_BOUNDS,
    opFocusBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    solidPickBoxes: b.pickBoxes,
    partCatalog,
    mambaStats: {
      bayId,
      isFeeder,
      conv1dTapCount: 4,
      conv1dFp8: false,
      ssmHeadCount: 64,
      ssmStateFp8: false,
      inProjFp8: computeIsFp8,
      outProjFp8: computeIsFp8,
      activeHead,
      activeGroup,
      stateTokenIndex: stateTokIdx,
      miniatureHeadCount: sim.nHeads,
      miniatureGroupCount: sim.nGroups,
      usesSmoothRibbons: true,
      voxelArrowCount: 0,
      ssmBytesAll23,
      conv1dBytesAll23,
    },
  };
}
