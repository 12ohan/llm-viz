/**
 * Shared thin-sheet inspection types, palette constants, and dependency-thread
 * geometry helpers reused across GQA and Mamba-2 live cutaways (Stage A+1 · Increment 3).
 */
import type { FlowRibbonBuilder, FlowRibbonColor } from '../render/ribbons';

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
}

export interface WorldSpanRect {
  readonly cx: number;
  readonly cz: number;
  readonly ex: number;
  readonly ez: number;
  readonly topY: number;
}

// Restrained drafting-table ribbon palette
export const RIBBON_QUIET: FlowRibbonColor = { r: 0.57, g: 0.62, b: 0.70, a: 0.28, borderAlpha: 0.16 };
export const RIBBON_DIMMED: FlowRibbonColor = { r: 0.68, g: 0.72, b: 0.78, a: 0.13, borderAlpha: 0.08 };
export const RIBBON_RESIDUAL: FlowRibbonColor = { r: 0.25, g: 0.64, b: 0.48, a: 0.40, borderAlpha: 0.26 };
export const RIBBON_RESIDUAL_ACTIVE: FlowRibbonColor = { r: 0.16, g: 0.62, b: 0.44, a: 0.76, borderAlpha: 0.44 };
export const RIBBON_ACTIVE: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.82, borderAlpha: 0.46 };
export const RIBBON_PATHWAY_SOFT: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.48, borderAlpha: 0.28 };
export const RIBBON_SSM_LOOP: FlowRibbonColor = { r: 0.12, g: 0.52, b: 0.68, a: 0.58, borderAlpha: 0.34 };

// Cell-level highlight & dependency thread colors
export const CELL_TARGET_FILL: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.24, borderAlpha: 0.0 };
export const CELL_TARGET_FRAME: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.98, borderAlpha: 0.25 };
export const CELL_OP_A_FILL: FlowRibbonColor = { r: 0.22, g: 0.38, b: 0.74, a: 0.18, borderAlpha: 0.0 };
export const CELL_OP_A_FRAME: FlowRibbonColor = { r: 0.22, g: 0.38, b: 0.74, a: 0.92, borderAlpha: 0.25 };
export const CELL_OP_B_FILL: FlowRibbonColor = { r: 0.10, g: 0.54, b: 0.48, a: 0.18, borderAlpha: 0.0 };
export const CELL_OP_B_FRAME: FlowRibbonColor = { r: 0.10, g: 0.54, b: 0.48, a: 0.92, borderAlpha: 0.25 };
export const THREAD_OP_A: FlowRibbonColor = { r: 0.22, g: 0.38, b: 0.74, a: 0.88, borderAlpha: 0.35 };
export const THREAD_OP_B: FlowRibbonColor = { r: 0.10, g: 0.54, b: 0.48, a: 0.88, borderAlpha: 0.35 };

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
    upRect.topY + 0.007,
    isOperandA ? CELL_OP_A_FILL : CELL_OP_B_FILL,
    isOperandA ? CELL_OP_A_FRAME : CELL_OP_B_FRAME,
    0.022,
  );

  const threadY = Math.max(targetRect.topY, upRect.topY) + 0.018;
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
      0.044,
      threadCol,
      true,
      0.14,
      0.14,
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
      0.042,
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
