/**
 * Shared thin-sheet inspection types, palette constants, and dependency-thread
 * geometry helpers reused across GQA and Mamba-2 live cutaways (Stage A+1 · Increment 3).
 */
import type { Vec3 } from '../model/spec';
import { add3, dot3, scale3, sub3, type AABB } from '../render/camera';
import type { FlowRibbonBuilder, FlowRibbonColor } from '../render/ribbons';
import {
  localToWorldOnFrame,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetFrame3D,
} from '../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_DIMMED,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
} from '../render/sheets';

/** Layer-agnostic readout contract consumed by the docked Formula Inspector bar. */
export interface LayerFormulaReadout {
  readonly tensorLabel: string;
  readonly coordLabel: string;
  readonly headBadge: string | null;
  readonly value: number;
  readonly isMasked: boolean;
  readonly valueText: string;
  readonly equation: string;
  readonly subEquation: string;
}

/** Layer-agnostic stepper stage definition for the 6-step Math Stepper bar. */
export interface StepperStepDef {
  readonly index: number;
  readonly shortLabel: string;
  readonly title: string;
  readonly summary: string;
}

export interface SheetGeomRecord {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly gridCols: number;
  readonly gridRows: number;
  /** Optional 3D oriented frame. When omitted, defaults to horizontal XZ plane. */
  readonly frame?: SheetFrame3D;
}

export interface WorldSpanRect {
  readonly cx: number;
  readonly cz: number;
  readonly ex: number;
  readonly ez: number;
  readonly topY: number;
}

/** Oriented 3D cell/row/column span anchored on an arbitrary SheetFrame3D. */
export interface WorldSpan3D {
  readonly frame: SheetFrame3D;
  readonly gridRows: number;
  readonly gridCols: number;
  readonly rowStart: number;
  readonly rowEnd: number;
  readonly colStart: number;
  readonly colEnd: number;
  /** World center on the sheet's midplane. */
  readonly center: Vec3;
  /** World center slightly proud of the +N face. */
  readonly frontPoint: Vec3;
  /** World center slightly proud of the -N face. */
  readonly backPoint: Vec3;
  readonly ex: number;
  readonly ez: number;
}

// Restrained drafting-table ribbon palette
export const RIBBON_QUIET: FlowRibbonColor = { r: 0.57, g: 0.62, b: 0.70, a: 0.28, borderAlpha: 0.16 };
export const RIBBON_DIMMED: FlowRibbonColor = { r: 0.68, g: 0.72, b: 0.78, a: 0.13, borderAlpha: 0.08 };
export const RIBBON_RESIDUAL: FlowRibbonColor = { r: 0.25, g: 0.64, b: 0.48, a: 0.40, borderAlpha: 0.26 };
export const RIBBON_RESIDUAL_ACTIVE: FlowRibbonColor = { r: 0.16, g: 0.62, b: 0.44, a: 0.76, borderAlpha: 0.44 };
export const RIBBON_ACTIVE: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.82, borderAlpha: 0.46 };
export const RIBBON_PATHWAY_SOFT: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.48, borderAlpha: 0.28 };
export const RIBBON_SSM_LOOP: FlowRibbonColor = { r: 0.12, g: 0.52, b: 0.68, a: 0.58, borderAlpha: 0.34 };

// Cell-level highlight & dependency thread colors (exact match with SheetsPass shader accents)
export const CELL_TARGET_FILL: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.28, borderAlpha: 0.0 };
export const CELL_TARGET_FRAME: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 1.0, borderAlpha: 0.30 };
export const CELL_OP_A_FILL: FlowRibbonColor = { r: 0.220, g: 0.380, b: 0.741, a: 0.22, borderAlpha: 0.0 };
export const CELL_OP_A_FRAME: FlowRibbonColor = { r: 0.220, g: 0.380, b: 0.741, a: 0.96, borderAlpha: 0.30 };
export const CELL_OP_B_FILL: FlowRibbonColor = { r: 0.102, g: 0.541, b: 0.478, a: 0.22, borderAlpha: 0.0 };
export const CELL_OP_B_FRAME: FlowRibbonColor = { r: 0.102, g: 0.541, b: 0.478, a: 0.96, borderAlpha: 0.30 };
export const THREAD_OP_A: FlowRibbonColor = { r: 0.220, g: 0.380, b: 0.741, a: 0.92, borderAlpha: 0.42 };
export const THREAD_OP_B: FlowRibbonColor = { r: 0.102, g: 0.541, b: 0.478, a: 0.92, borderAlpha: 0.42 };

// ============================================================================
// SHARED FULL-IDENTITY FOCUS ENGINE (exact tensor + head/kv/group/expert/token/tap matching)
// ============================================================================

export type FocusRole = 'target' | 'operand_a' | 'operand_b';

/**
 * Full tensor identity descriptor for exact operand-aware highlighting and dimming.
 * Preserves tensor key, headIndex, kvHeadIndex, groupIndex, expertId, token range,
 * Conv1D tap range, and logical cell/span coordinates where applicable.
 */
export interface SheetFocusIdentity {
  readonly tensor: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly groupIndex?: number;
  readonly expertId?: number;
  /** Inclusive token range (single token when tokenStart === tokenEnd). */
  readonly tokenStart?: number;
  readonly tokenEnd?: number;
  /** Inclusive Conv1D tap range (0..3). */
  readonly tapStart?: number;
  readonly tapEnd?: number;
  /** Optional logical cell/span coordinates. */
  readonly rowStart?: number;
  readonly rowEnd?: number;
  readonly colStart?: number;
  readonly colEnd?: number;
}

export interface SheetFocusEntry extends SheetFocusIdentity {
  readonly role: FocusRole;
}

export interface SheetCandidateIdentity {
  readonly tensor: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly groupIndex?: number;
  readonly expertId?: number;
  readonly tokenIndex?: number;
  readonly tapIndex?: number;
  readonly rowStart?: number;
  readonly rowEnd?: number;
  readonly colStart?: number;
  readonly colEnd?: number;
}

export interface SheetFocusSet {
  readonly active: boolean;
  readonly entries: readonly SheetFocusEntry[];
}

export const EMPTY_SHEET_FOCUS_SET: SheetFocusSet = {
  active: false,
  entries: [],
};

export function createSheetFocusSet(entries: readonly SheetFocusEntry[]): SheetFocusSet {
  return {
    active: entries.length > 0,
    entries,
  };
}

/**
 * Exact full-identity match between a focus entry and a candidate sheet/tile.
 * Never matches by tensor key alone whenever finer-grained identity selectors are present.
 */
export function matchesFocusEntry(
  entry: SheetFocusEntry,
  candidate: SheetCandidateIdentity,
): boolean {
  if (entry.tensor !== candidate.tensor) return false;
  if (
    entry.headIndex !== undefined &&
    candidate.headIndex !== undefined &&
    entry.headIndex !== candidate.headIndex
  ) {
    return false;
  }
  if (
    entry.kvHeadIndex !== undefined &&
    candidate.kvHeadIndex !== undefined &&
    entry.kvHeadIndex !== candidate.kvHeadIndex
  ) {
    return false;
  }
  if (
    entry.groupIndex !== undefined &&
    candidate.groupIndex !== undefined &&
    entry.groupIndex !== candidate.groupIndex
  ) {
    return false;
  }
  if (
    entry.expertId !== undefined &&
    candidate.expertId !== undefined &&
    entry.expertId !== candidate.expertId
  ) {
    return false;
  }
  if (
    candidate.tokenIndex !== undefined &&
    (entry.tokenStart !== undefined || entry.tokenEnd !== undefined)
  ) {
    const t0 = entry.tokenStart ?? entry.tokenEnd!;
    const t1 = entry.tokenEnd ?? entry.tokenStart!;
    if (candidate.tokenIndex < t0 || candidate.tokenIndex > t1) {
      return false;
    }
  }
  if (
    candidate.tapIndex !== undefined &&
    (entry.tapStart !== undefined || entry.tapEnd !== undefined)
  ) {
    const s0 = entry.tapStart ?? entry.tapEnd!;
    const s1 = entry.tapEnd ?? entry.tapStart!;
    if (candidate.tapIndex < s0 || candidate.tapIndex > s1) {
      return false;
    }
  }
  if (
    candidate.rowStart !== undefined &&
    candidate.rowEnd !== undefined &&
    entry.rowStart !== undefined &&
    entry.rowEnd !== undefined
  ) {
    if (candidate.rowEnd < entry.rowStart || candidate.rowStart > entry.rowEnd) {
      return false;
    }
  }
  if (
    candidate.colStart !== undefined &&
    candidate.colEnd !== undefined &&
    entry.colStart !== undefined &&
    entry.colEnd !== undefined
  ) {
    if (candidate.colEnd < entry.colStart || candidate.colStart > entry.colEnd) {
      return false;
    }
  }
  return true;
}

const ROLE_PRIORITY: Record<FocusRole, number> = {
  target: 3,
  operand_a: 2,
  operand_b: 1,
};

/** Resolve the highest-priority focus role for a candidate sheet/tile, or null if unrelated. */
export function resolveSheetFocusRole(
  focus: SheetFocusSet,
  candidate: SheetCandidateIdentity,
): FocusRole | null {
  if (!focus.active) return null;
  let best: FocusRole | null = null;
  for (const entry of focus.entries) {
    if (!matchesFocusEntry(entry, candidate)) continue;
    if (best === null || ROLE_PRIORITY[entry.role] > ROLE_PRIORITY[best]) {
      best = entry.role;
      if (best === 'target') return best;
    }
  }
  return best;
}

/** Map a focus role to its SheetsPass highlight code. */
export function focusRoleToHighlight(role: FocusRole): number {
  if (role === 'target') return SHEET_HL_ACTIVE;
  if (role === 'operand_a') return SHEET_HL_OPERAND_A;
  return SHEET_HL_OPERAND_B;
}

/**
 * Resolve final SheetsPass highlight code for a sheet/tile:
 *  - When a cell/operation focus set is active, ONLY exact full-identity matches highlight
 *    (target -> ACTIVE, operand_a -> OPERAND_A, operand_b -> OPERAND_B) and ALL unrelated
 *    sheets/tiles dim cleanly to SHEET_HL_DIMMED.
 *  - When no focus set is active, falls back to idle/part/BF16-gold highlighting.
 */
export function resolveSheetHighlight(
  focus: SheetFocusSet,
  candidate: SheetCandidateIdentity,
  fallbackHighlight: number,
): number {
  if (!focus.active) return fallbackHighlight;
  const role = resolveSheetFocusRole(focus, candidate);
  return role !== null ? focusRoleToHighlight(role) : SHEET_HL_DIMMED;
}

/** Compute world-space AABB for a SheetGeomRecord (handling both XZ and arbitrary 3D frames). */
export function sheetRecordWorldAabb(sheet: SheetGeomRecord): AABB {
  if (sheet.frame) {
    return sheetFrameWorldAabb(sheet.frame);
  }
  const hx = sheet.ex * 0.5;
  const hy = sheet.ey * 0.5;
  const hz = sheet.ez * 0.5;
  return {
    min: [sheet.cx - hx, sheet.cy - hy, sheet.cz - hz],
    max: [sheet.cx + hx, sheet.cy + hy, sheet.cz + hz],
  };
}

/**
 * Compute a tight, finite, non-degenerate world-space AABB enclosing a set of sheets
 * (typically an operation's inputs + weights + outputs), modestly padded for labels/ribbons
 * and optionally clamped to the scene's outer fitBounds.
 */
export function computeSheetGroupBounds(
  sheets: readonly (SheetGeomRecord | undefined | null)[],
  padX: number,
  padY: number,
  padZ: number,
  clampTo?: AABB,
): AABB {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let count = 0;

  for (const s of sheets) {
    if (!s) continue;
    const b = sheetRecordWorldAabb(s);
    if (b.min[0] < minX) minX = b.min[0];
    if (b.min[1] < minY) minY = b.min[1];
    if (b.min[2] < minZ) minZ = b.min[2];
    if (b.max[0] > maxX) maxX = b.max[0];
    if (b.max[1] > maxY) maxY = b.max[1];
    if (b.max[2] > maxZ) maxZ = b.max[2];
    count++;
  }

  if (count === 0) {
    return clampTo ?? { min: [-1, -0.1, -1], max: [1, 0.2, 1] };
  }

  minX -= padX;
  minY -= padY;
  minZ -= padZ;
  maxX += padX;
  maxY += padY;
  maxZ += padZ;

  if (clampTo) {
    minX = Math.max(clampTo.min[0], minX);
    minY = Math.max(clampTo.min[1], minY);
    minZ = Math.max(clampTo.min[2], minZ);
    maxX = Math.min(clampTo.max[0], maxX);
    maxY = Math.min(clampTo.max[1], maxY);
    maxZ = Math.min(clampTo.max[2], maxZ);
  }

  if (maxX <= minX) maxX = minX + 0.1;
  if (maxY <= minY) maxY = minY + 0.05;
  if (maxZ <= minZ) maxZ = minZ + 0.1;

  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
  };
}

/** Compute world-space XZ bounding rect for an inclusive [rowStart..rowEnd] × [colStart..colEnd] cell span. */
export function rectFromSheetGrid(
  sheet: SheetGeomRecord,
  rowStart: number,
  rowEnd: number,
  colStart: number,
  colEnd: number,
): WorldSpanRect {
  const c0 = Math.max(0, Math.min(sheet.gridCols - 1, colStart));
  const c1 = Math.max(c0, Math.min(sheet.gridCols - 1, colEnd));
  const r0 = Math.max(0, Math.min(sheet.gridRows - 1, rowStart));
  const r1 = Math.max(r0, Math.min(sheet.gridRows - 1, rowEnd));

  const colW = sheet.ex / sheet.gridCols;
  const rowD = sheet.ez / sheet.gridRows;
  const x0 = sheet.cx - sheet.ex * 0.5 + c0 * colW;
  const x1 = sheet.cx - sheet.ex * 0.5 + (c1 + 1) * colW;
  const z0 = sheet.cz - sheet.ez * 0.5 + r0 * rowD;
  const z1 = sheet.cz - sheet.ez * 0.5 + (r1 + 1) * rowD;

  return {
    cx: (x0 + x1) * 0.5,
    cz: (z0 + z1) * 0.5,
    ex: Math.max(0.035, x1 - x0),
    ez: Math.max(0.035, z1 - z0),
    topY: sheet.cy + sheet.ey * 0.5,
  };
}

/** Resolve an oriented 3D cell/span descriptor on any SheetGeomRecord (defaults to XZ frame). */
export function spanFromSheetGrid(
  sheet: SheetGeomRecord,
  rowStart: number,
  rowEnd: number,
  colStart: number,
  colEnd: number,
): WorldSpan3D {
  const frame =
    sheet.frame ??
    makeSheetFrame3D([sheet.cx, sheet.cy, sheet.cz], [sheet.ex, sheet.ey, sheet.ez]);
  const c0 = Math.max(0, Math.min(sheet.gridCols - 1, colStart));
  const c1 = Math.max(c0, Math.min(sheet.gridCols - 1, colEnd));
  const r0 = Math.max(0, Math.min(sheet.gridRows - 1, rowStart));
  const r1 = Math.max(r0, Math.min(sheet.gridRows - 1, rowEnd));

  const lxMid = -0.5 + (c0 + c1 + 1) * 0.5 / sheet.gridCols;
  const lzMid = -0.5 + (r0 + r1 + 1) * 0.5 / sheet.gridRows;
  const surfLy = 0.5 + 0.014 / Math.max(1e-5, frame.extent[1]);

  return {
    frame,
    gridRows: sheet.gridRows,
    gridCols: sheet.gridCols,
    rowStart: r0,
    rowEnd: r1,
    colStart: c0,
    colEnd: c1,
    center: localToWorldOnFrame(frame, lxMid, 0, lzMid),
    frontPoint: localToWorldOnFrame(frame, lxMid, surfLy, lzMid),
    backPoint: localToWorldOnFrame(frame, lxMid, -surfLy, lzMid),
    ex: Math.max(0.035, ((c1 - c0 + 1) / sheet.gridCols) * frame.extent[0]),
    ez: Math.max(0.035, ((r1 - r0 + 1) / sheet.gridRows) * frame.extent[2]),
  };
}

/** Draw an oriented cell highlight on a WorldSpan3D (flush on both +N and -N faces). */
export function drawCellHighlight3D(
  flow: FlowRibbonBuilder,
  span: WorldSpan3D,
  fillColor: FlowRibbonColor,
  frameColor: FlowRibbonColor,
  frameThickness = 0.022,
  doubleSided = true,
): void {
  flow.addCellHighlightOnFrame(
    span.frame,
    span.rowStart,
    span.rowEnd,
    span.colStart,
    span.colEnd,
    span.gridRows,
    span.gridCols,
    0.008,
    fillColor,
    frameColor,
    frameThickness,
    doubleSided,
  );
}

/** Draw upstream oriented highlight + smooth cross-plane 3D cubic dependency thread to target span. */
export function drawOperandDependency3D(
  flow: FlowRibbonBuilder,
  upSpan: WorldSpan3D,
  targetSpan: WorldSpan3D,
  isOperandA: boolean,
): void {
  drawCellHighlight3D(
    flow,
    upSpan,
    isOperandA ? CELL_OP_A_FILL : CELL_OP_B_FILL,
    isOperandA ? CELL_OP_A_FRAME : CELL_OP_B_FRAME,
    0.022,
    true,
  );

  // Pick exit/entry face (+N vs -N) that faces toward the partner span so threads never pierce their own sheet.
  // For horizontal deck sheets (axisN.y > 0.7) whose partner is within deck height, stay on the top (+N) face
  // so threads arch cleanly above the sheets instead of diving underneath.
  const delta = sub3(targetSpan.center, upSpan.center);
  const dist = Math.hypot(delta[0], delta[1], delta[2]);
  const upDot = dot3(delta, upSpan.frame.axisN);
  const tgtDot = dot3(delta, targetSpan.frame.axisN);

  const upIsHoriz = upSpan.frame.axisN[1] > 0.7;
  const tgtIsHoriz = targetSpan.frame.axisN[1] > 0.7;

  const upSign: 1 | -1 = upIsHoriz && delta[1] > -0.85 ? 1 : upDot >= -0.05 ? 1 : -1;
  const tgtSign: 1 | -1 = tgtIsHoriz && delta[1] < 0.85 ? 1 : tgtDot <= 0.05 ? 1 : -1;

  const p0 = upSign > 0 ? upSpan.frontPoint : upSpan.backPoint;
  const p3 = tgtSign > 0 ? targetSpan.frontPoint : targetSpan.backPoint;

  const lift = Math.max(0.24, Math.min(1.05, dist * 0.22));
  const upN = scale3(upSpan.frame.axisN, upSign);
  const tgtN = scale3(targetSpan.frame.axisN, tgtSign);
  const horizPull = scale3(delta, 0.18);

  const p1 = add3(add3(p0, scale3(upN, lift)), horizPull);
  const p2 = add3(add3(p3, scale3(tgtN, lift)), scale3(horizPull, -1));

  const threadCol = isOperandA ? THREAD_OP_A : THREAD_OP_B;
  flow.addCubic3D(p0, p1, p2, p3, [0, 1, 0], 0.050, threadCol, true, 24, true, 0.15, 0.15);
}

/** Draw target cell highlight + upstream operand highlight + crisp curved dependency thread. */
export function drawOperandDependencyXZ(
  flow: FlowRibbonBuilder,
  upRect: WorldSpanRect,
  targetRect: WorldSpanRect,
  isOperandA: boolean,
): void {
  flow.addCellHighlightXZ(
    upRect.cx,
    upRect.cz,
    upRect.ex,
    upRect.ez,
    upRect.topY + 0.008,
    isOperandA ? CELL_OP_A_FILL : CELL_OP_B_FILL,
    isOperandA ? CELL_OP_A_FRAME : CELL_OP_B_FRAME,
    0.025,
  );

  const threadY = Math.max(targetRect.topY, upRect.topY) + 0.020;
  const dz = targetRect.cz - upRect.cz;
  const dx = targetRect.cx - upRect.cx;
  const threadCol = isOperandA ? THREAD_OP_A : THREAD_OP_B;

  if (Math.abs(dz) < 0.25) {
    const startX = upRect.cx + Math.sign(dx || 1) * (upRect.ex * 0.5 + 0.02);
    const endX = targetRect.cx - Math.sign(dx || 1) * (targetRect.ex * 0.5 + 0.03);
    flow.addPolylineXZ(
      [
        { x: startX, z: upRect.cz },
        { x: endX, z: targetRect.cz },
      ],
      threadY,
      0.050,
      threadCol,
      true,
      0.15,
      0.15,
    );
  } else {
    const dirZ = Math.sign(dz || 1);
    const startZ = upRect.cz + dirZ * (upRect.ez * 0.5 + 0.02);
    const endZ = targetRect.cz - dirZ * (targetRect.ez * 0.5 + 0.03);
    const midZ = (upRect.cz + targetRect.cz) * 0.5;
    flow.addCubicXZ(
      { x: upRect.cx, z: startZ },
      { x: upRect.cx, z: midZ },
      { x: targetRect.cx, z: midZ },
      { x: targetRect.cx, z: endZ },
      threadY,
      0.048,
      threadCol,
      true,
      20,
    );
  }
}

export function fmtSigned(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return v < 0 ? '−∞' : '+∞';
  const r = Math.abs(v) < 0.5 * Math.pow(10, -digits) ? 0 : v;
  const sign = r < 0 ? '−' : '+';
  return `${sign}${Math.abs(r).toFixed(digits)}`;
}
