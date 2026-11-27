/**
 * Pure 3D-to-2D Drafting-Table Overlay Layout & Cell-Readout Helpers
 * (`src/shared/overlayLayout.ts`)
 *
 * Depends strictly on `src/render/camera.ts` and `src/render/sheetFrame.ts`.
 * Zero imports from `src/ui/`, `src/model/`, `src/math/`, or `src/scene/` so all
 * visualizers can share projected cell sizing, popover collision avoidance,
 * elbow/quadratic SVG leader geometry, and reveal masking without cross-app drag.
 */

import { dot3, projectWorldToHostCss, sub3, type CameraSnapshot } from '../render/camera';
import { localToWorldOnFrame, makeSheetFrame3D } from '../render/sheetFrame';

export interface CellPopoverSpec {
  readonly world: readonly [number, number, number]; // world center of active/hovered 3D cell
  readonly tensorLabel: string;
  readonly coordLabel: string;
  readonly badge?: string;
  readonly equation: string; // concise primary formula
  readonly valueText: string; // e.g. "= +3.294" or "= −∞"
  /** True while actively accumulating a focused-cell running partial sum (toggles `.is-accumulating`). */
  readonly isAccumulating?: boolean;
  readonly operandColors?: readonly ('indigo' | 'teal' | 'red')[];
  readonly sourceWorlds?: readonly {
    readonly world: readonly [number, number, number];
    readonly role: 'indigo' | 'teal';
  }[];
}

export interface ActiveSheetCellGrid {
  readonly originWorld: readonly [number, number, number]; // sheet world center
  readonly extent: readonly [number, number, number]; // (ex, ey, ez)
  readonly axisU: readonly [number, number, number]; // +col unit vector
  readonly axisN: readonly [number, number, number]; // sheet normal unit vector
  readonly cols: number;
  readonly rows: number;
  readonly values: Float32Array; // row-major [row * cols + col]
  readonly masked?: Uint8Array; // optional 1 = masked -inf
  readonly format?: 'signed' | 'prob';
  readonly activeCol?: number;
  readonly activeRow?: number;
  /** Progressive reveal mode matching `SHEET_REVEAL_*`: 0=NONE, 1=ROW_MAJOR, 2=COL_MAJOR, 3=UNCOMPUTED. */
  readonly revealMode?: number;
  /** Inclusive active column frontier for progressive reveal modes (`1` or `2`). */
  readonly revealColFrontier?: number;
  /** Inclusive active row frontier for progressive reveal modes (`1` or `2`). */
  readonly revealRowFrontier?: number;
  /** Running partial-sum override for the actively accumulating frontier cell (`col === activeCol && row === activeRow`). */
  readonly activeValueOverride?: number | null;
  /** Optional formatted string override for the actively accumulating frontier cell (e.g. `Σ1.84`). Takes precedence over numeric formatting when non-null. */
  readonly activeValueTextOverride?: string | null;
  readonly colLabels?: readonly string[];
  readonly rowLabels?: readonly string[];
  readonly colDimLabel?: string; // e.g. "K = 6"
  readonly rowDimLabel?: string; // e.g. "Q = 1"
  /** When true, keeps `colLabels` data intact while omitting exterior col header pills (e.g. twin adjacent strips). */
  readonly suppressColHeaders?: boolean;
  /** When true, keeps `rowLabels` data intact while omitting exterior row header pills (e.g. beside Softmax Agg). */
  readonly suppressRowHeaders?: boolean;
}

export interface OverlayRect {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

export interface PopoverPlacementInput {
  readonly anchorX: number;
  readonly anchorY: number;
  readonly width: number;
  readonly height: number;
  readonly hostWidth: number;
  readonly minY: number;
  readonly maxY: number;
  readonly occupied?: readonly OverlayRect[];
  readonly avoidRects?: readonly OverlayRect[];
  readonly targetSheetRect?: OverlayRect | null;
  readonly marginX?: number;
}

export interface PopoverPlacement {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly side: 'top-right' | 'top-left' | 'top' | 'bottom-right' | 'bottom-left' | 'bottom';
  readonly edgeX: number;
  readonly edgeY: number;
  readonly exitOrientation: 'vertical' | 'horizontal';
}

export interface ProjectedCellSize {
  readonly cellW: number;
  readonly cellH: number;
  readonly colStepPx: number;
  readonly rowStepPx: number;
  readonly visible: boolean;
}

export const MAX_OVERLAY_CELL_NUM_NODES = 64;
export const MAX_OVERLAY_AXIS_HEADER_NODES = 24;
export const CELL_NUM_MIN_WIDTH_PX = 26;
export const CELL_NUM_MIN_HEIGHT_PX = 16;
export const AXIS_HEADER_MIN_WIDTH_PX = 22;
export const AXIS_HEADER_MIN_HEIGHT_PX = 15;
export const LABEL_NUDGE_LEADER_THRESHOLD_PX = 10;
export const DIMMED_LABEL_OPACITY = '0.46';
export const MAX_OVERLAY_WIDTH_CACHE_ENTRIES = 2048;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function rectOverlapArea(a: OverlayRect, b: OverlayRect, pad = 3): number {
  const ox = Math.min(a.x1 + pad, b.x1 + pad) - Math.max(a.x0, b.x0);
  if (ox <= 0) return 0;
  const oy = Math.min(a.y1 + pad, b.y1 + pad) - Math.max(a.y0, b.y0);
  if (oy <= 0) return 0;
  return ox * oy;
}

/**
 * Strict readability threshold for zoom-adaptive on-cell numeric readouts:
 * numbers render ONLY when projected cell dimensions satisfy `cellW >= 26 && cellH >= 16`.
 */
export function shouldShowCellNumbers(cellW: number, cellH: number): boolean {
  return (
    Number.isFinite(cellW) &&
    Number.isFinite(cellH) &&
    cellW >= CELL_NUM_MIN_WIDTH_PX &&
    cellH >= CELL_NUM_MIN_HEIGHT_PX
  );
}

/**
 * Threshold for exterior row/column token & channel axis headers (`cellW >= 22 || cellH >= 15`).
 */
export function shouldShowAxisHeaders(cellW: number, cellH: number): boolean {
  return (
    Number.isFinite(cellW) &&
    Number.isFinite(cellH) &&
    (cellW >= AXIS_HEADER_MIN_WIDTH_PX || cellH >= AXIS_HEADER_MIN_HEIGHT_PX)
  );
}

/**
 * Returns true when logical cell `(col, row)` sits strictly past the sheet's progressive
 * reveal frontier and must NOT expose its final numeric value in `.ov-cell-num`.
 * Matches `SHEETS_FS` progressive reveal modes (`0=NONE, 1=ROW_MAJOR, 2=COL_MAJOR, 3=UNCOMPUTED`).
 */
export function isCellHiddenByReveal(
  col: number,
  row: number,
  grid: Pick<ActiveSheetCellGrid, 'revealMode' | 'revealColFrontier' | 'revealRowFrontier'>,
): boolean {
  const mode = grid.revealMode;
  if (mode === undefined || !Number.isFinite(mode) || mode < 0.5) {
    return false;
  }
  if (mode >= 2.5) {
    return true;
  }
  const colFrontier =
    grid.revealColFrontier !== undefined && Number.isFinite(grid.revealColFrontier)
      ? Math.round(grid.revealColFrontier)
      : 0;
  const rowFrontier =
    grid.revealRowFrontier !== undefined && Number.isFinite(grid.revealRowFrontier)
      ? Math.round(grid.revealRowFrontier)
      : 0;

  if (mode >= 1.5) {
    return col > colFrontier || (col === colFrontier && row > rowFrontier);
  }
  return row > rowFrontier || (row === rowFrontier && col > colFrontier);
}

/**
 * Format a scalar Float32 cell value concisely for in-canvas on-cell numeric readouts
 * (tabular monospace: `−∞`, `0.28` for probabilities, `+1.52` / `−0.98` for signed tensors).
 */
export function formatCellOverlayValue(
  val: number,
  format: 'signed' | 'prob' = 'signed',
  isMasked = false,
): string {
  if (isMasked) return '−∞';
  if (!Number.isFinite(val)) {
    if (Number.isNaN(val)) return '—';
    return val < 0 ? '−∞' : '+∞';
  }
  if (format === 'prob') {
    const clean = Math.abs(val) < 0.005 ? 0 : val;
    return clean.toFixed(2);
  }
  const clean = Math.abs(val) < 0.005 ? 0 : val;
  const sign = clean < 0 ? '−' : '+';
  const mag = Math.abs(clean);
  if (mag >= 100) return `${sign}${mag.toFixed(0)}`;
  if (mag >= 10) return `${sign}${mag.toFixed(1)}`;
  return `${sign}${mag.toFixed(2)}`;
}

/**
 * Compute world-space position of logical cell `(col, row)` on an oriented thin matrix sheet.
 * Matches `SHEETS_VS` right-handed orthonormal basis (`vAxis = normalize(cross(axisU, axisN))`).
 */
export function computeSheetCellWorldPos(
  grid: Pick<ActiveSheetCellGrid, 'originWorld' | 'extent' | 'axisU' | 'axisN' | 'cols' | 'rows'>,
  col: number,
  row: number,
  faceLy = 0.5,
): readonly [number, number, number] {
  const safeCols = Math.max(1, grid.cols);
  const safeRows = Math.max(1, grid.rows);
  const frame = makeSheetFrame3D(grid.originWorld, grid.extent, grid.axisU, grid.axisN);
  const lx = (col + 0.5) / safeCols - 0.5;
  const lz = (row + 0.5) / safeRows - 0.5;
  return localToWorldOnFrame(frame, lx, faceLy, lz);
}

function measureParallelogramCell(
  dux: number,
  duy: number,
  dvx: number,
  dvy: number,
): ProjectedCellSize {
  const lenU = Math.hypot(dux, duy);
  const lenV = Math.hypot(dvx, dvy);
  const area = Math.abs(dux * dvy - duy * dvx);
  const perpU = lenV > 1e-4 ? area / lenV : 0;
  const perpV = lenU > 1e-4 ? area / lenU : 0;
  const colsMoreHorizontal = Math.abs(dux) >= Math.abs(dvx);
  return {
    cellW: colsMoreHorizontal ? perpU : perpV,
    cellH: colsMoreHorizontal ? perpV : perpU,
    colStepPx: perpU,
    rowStepPx: perpV,
    visible: true,
  };
}

/**
 * Project the active cell (or sheet center) and adjacent cell centers `(col+1, row)` and
 * `(col, row+1)` to measure projected cell dimensions `(cellW, cellH)` in CSS pixels.
 */
export function computeProjectedCellSize(
  grid: Pick<
    ActiveSheetCellGrid,
    'originWorld' | 'extent' | 'axisU' | 'axisN' | 'cols' | 'rows' | 'activeCol' | 'activeRow'
  >,
  snapshot: CameraSnapshot,
): ProjectedCellSize {
  const safeCols = Math.max(1, grid.cols);
  const safeRows = Math.max(1, grid.rows);
  const frame = makeSheetFrame3D(grid.originWorld, grid.extent, grid.axisU, grid.axisN);
  const faceLy = dot3(sub3(snapshot.eye, frame.center), frame.axisN) >= 0 ? 0.5 : -0.5;

  const midCol = (safeCols - 1) * 0.5;
  const midRow = (safeRows - 1) * 0.5;
  const hasActive =
    grid.activeCol !== undefined &&
    grid.activeRow !== undefined &&
    grid.activeCol >= 0 &&
    grid.activeCol < safeCols &&
    grid.activeRow >= 0 &&
    grid.activeRow < safeRows;

  const refCandidates: readonly (readonly [number, number])[] = hasActive
    ? [
        [grid.activeCol!, grid.activeRow!],
        [midCol, midRow],
      ]
    : [[midCol, midRow]];

  for (const [rc, rr] of refCandidates) {
    const pCenter = localToWorldOnFrame(
      frame,
      (rc + 0.5) / safeCols - 0.5,
      faceLy,
      (rr + 0.5) / safeRows - 0.5,
    );
    const sCenter = projectWorldToHostCss(pCenter, snapshot);
    if (!sCenter.visible) continue;

    const pU0 = localToWorldOnFrame(frame, rc / safeCols - 0.5, faceLy, (rr + 0.5) / safeRows - 0.5);
    const pU1 = localToWorldOnFrame(
      frame,
      (rc + 1) / safeCols - 0.5,
      faceLy,
      (rr + 0.5) / safeRows - 0.5,
    );
    const pV0 = localToWorldOnFrame(frame, (rc + 0.5) / safeCols - 0.5, faceLy, rr / safeRows - 0.5);
    const pV1 = localToWorldOnFrame(
      frame,
      (rc + 0.5) / safeCols - 0.5,
      faceLy,
      (rr + 1) / safeRows - 0.5,
    );

    const sU0 = projectWorldToHostCss(pU0, snapshot);
    const sU1 = projectWorldToHostCss(pU1, snapshot);
    const sV0 = projectWorldToHostCss(pV0, snapshot);
    const sV1 = projectWorldToHostCss(pV1, snapshot);

    if (!sU0.visible || !sU1.visible || !sV0.visible || !sV1.visible) {
      const sNextU = projectWorldToHostCss(
        localToWorldOnFrame(frame, (rc + 1.5) / safeCols - 0.5, faceLy, (rr + 0.5) / safeRows - 0.5),
        snapshot,
      );
      const sNextV = projectWorldToHostCss(
        localToWorldOnFrame(frame, (rc + 0.5) / safeCols - 0.5, faceLy, (rr + 1.5) / safeRows - 0.5),
        snapshot,
      );
      if (!sNextU.visible || !sNextV.visible) continue;
      return measureParallelogramCell(
        sNextU.x - sCenter.x,
        sNextU.y - sCenter.y,
        sNextV.x - sCenter.x,
        sNextV.y - sCenter.y,
      );
    }

    return measureParallelogramCell(
      sU1.x - sU0.x,
      sU1.y - sU0.y,
      sV1.x - sV0.x,
      sV1.y - sV0.y,
    );
  }

  return { cellW: 0, cellH: 0, colStepPx: 0, rowStepPx: 0, visible: false };
}

export function pickRectBorderPoint(
  rect: OverlayRect,
  targetX: number,
  targetY: number,
): { edgeX: number; edgeY: number; orientation: 'vertical' | 'horizontal' } {
  const w = rect.x1 - rect.x0;
  const h = rect.y1 - rect.y0;
  const insetX = Math.min(16, w * 0.25);
  const insetY = Math.min(12, h * 0.3);

  if (targetY > rect.y1) {
    return {
      edgeX: clamp(targetX, rect.x0 + insetX, rect.x1 - insetX),
      edgeY: rect.y1,
      orientation: 'vertical',
    };
  }
  if (targetY < rect.y0) {
    return {
      edgeX: clamp(targetX, rect.x0 + insetX, rect.x1 - insetX),
      edgeY: rect.y0,
      orientation: 'vertical',
    };
  }
  if (targetX < rect.x0) {
    return {
      edgeX: rect.x0,
      edgeY: clamp(targetY, rect.y0 + insetY, rect.y1 - insetY),
      orientation: 'horizontal',
    };
  }
  if (targetX > rect.x1) {
    return {
      edgeX: rect.x1,
      edgeY: clamp(targetY, rect.y0 + insetY, rect.y1 - insetY),
      orientation: 'horizontal',
    };
  }

  const dTop = Math.abs(targetY - rect.y0);
  const dBot = Math.abs(rect.y1 - targetY);
  const dLeft = Math.abs(targetX - rect.x0);
  const dRight = Math.abs(rect.x1 - targetX);
  const m = Math.min(dTop, dBot, dLeft, dRight);
  if (m === dBot) {
    return {
      edgeX: clamp(targetX, rect.x0 + insetX, rect.x1 - insetX),
      edgeY: rect.y1,
      orientation: 'vertical',
    };
  }
  if (m === dTop) {
    return {
      edgeX: clamp(targetX, rect.x0 + insetX, rect.x1 - insetX),
      edgeY: rect.y0,
      orientation: 'vertical',
    };
  }
  if (m === dLeft) {
    return {
      edgeX: rect.x0,
      edgeY: clamp(targetY, rect.y0 + insetY, rect.y1 - insetY),
      orientation: 'horizontal',
    };
  }
  return {
    edgeX: rect.x1,
    edgeY: clamp(targetY, rect.y0 + insetY, rect.y1 - insetY),
    orientation: 'horizontal',
  };
}

/**
 * Intelligently place a floating contextual cell popover offset above/beside the active 3D
 * cell anchor, avoiding `occupied` label rectangles, `avoidRects`, and clamping strictly inside
 * `[marginX .. hostWidth - width - marginX] × [minY .. maxY - height]`.
 */
export function pickPopoverPlacement(input: PopoverPlacementInput): PopoverPlacement {
  const marginX = input.marginX ?? 8;
  const maxAllowedW = Math.max(40, input.hostWidth - marginX * 2);
  const width = Math.min(input.width, maxAllowedW);
  const maxAllowedH = Math.max(24, input.maxY - input.minY);
  const height = Math.min(input.height, maxAllowedH);

  const minX = marginX;
  const maxX = Math.max(minX, input.hostWidth - width - marginX);
  const minY = input.minY;
  const maxTopY = Math.max(minY, input.maxY - height);
  const occupied = input.occupied ?? [];
  const rawAvoid = input.avoidRects ?? [];
  const targetSheet = input.targetSheetRect ?? null;

  let avoidRects: readonly OverlayRect[] = rawAvoid;
  if (targetSheet) {
    const alreadyPresent = rawAvoid.some(
      (r) =>
        r.x0 === targetSheet.x0 &&
        r.y0 === targetSheet.y0 &&
        r.x1 === targetSheet.x1 &&
        r.y1 === targetSheet.y1,
    );
    if (!alreadyPresent) {
      avoidRects = [...rawAvoid, targetSheet];
    }
  }

  const gapX = 22;
  const gapY = 26;
  const outerGapX = gapX + 48;

  const candidates: {
    readonly side: PopoverPlacement['side'];
    readonly rawX: number;
    readonly rawY: number;
    readonly bias: number;
  }[] = [
    { side: 'top-right', rawX: input.anchorX + gapX, rawY: input.anchorY - height - gapY, bias: 0 },
    { side: 'top-left', rawX: input.anchorX - width - gapX, rawY: input.anchorY - height - gapY, bias: 4 },
    { side: 'top', rawX: input.anchorX - width * 0.5, rawY: input.anchorY - height - gapY - 4, bias: 8 },
    { side: 'bottom-right', rawX: input.anchorX + gapX, rawY: input.anchorY + gapY, bias: 12 },
    { side: 'bottom-left', rawX: input.anchorX - width - gapX, rawY: input.anchorY + gapY, bias: 16 },
    { side: 'bottom', rawX: input.anchorX - width * 0.5, rawY: input.anchorY + gapY + 4, bias: 20 },
    { side: 'top-right', rawX: input.anchorX + outerGapX, rawY: input.anchorY - height - gapY, bias: 26 },
    { side: 'top-left', rawX: input.anchorX - width - outerGapX, rawY: input.anchorY - height - gapY, bias: 30 },
    { side: 'top-right', rawX: input.anchorX + outerGapX, rawY: input.anchorY - height * 0.5, bias: 32 },
    { side: 'bottom-right', rawX: input.anchorX + outerGapX, rawY: input.anchorY + gapY, bias: 34 },
    { side: 'top-left', rawX: input.anchorX - width - outerGapX, rawY: input.anchorY - height * 0.5, bias: 36 },
    { side: 'bottom-left', rawX: input.anchorX - width - outerGapX, rawY: input.anchorY + gapY, bias: 38 },
  ];

  if (targetSheet) {
    const sheetGap = 14;
    const aboveY = targetSheet.y0 - height - sheetGap;
    const belowY = targetSheet.y1 + sheetGap;
    const leftX = targetSheet.x0 - width - sheetGap;
    const rightX = targetSheet.x1 + sheetGap;
    const centeredX = input.anchorX - width * 0.5;
    const centeredY = input.anchorY - height * 0.5;

    candidates.push(
      { side: 'top', rawX: centeredX, rawY: aboveY, bias: 22 },
      { side: 'top-right', rawX: rightX, rawY: centeredY, bias: 24 },
      { side: 'top-right', rawX: targetSheet.x1 - width, rawY: aboveY, bias: 25 },
      { side: 'top-left', rawX: leftX, rawY: centeredY, bias: 26 },
      { side: 'top-left', rawX: targetSheet.x0, rawY: aboveY, bias: 27 },
      { side: 'bottom', rawX: centeredX, rawY: belowY, bias: 28 },
      { side: 'top-right', rawX: rightX, rawY: targetSheet.y0, bias: 29 },
      { side: 'bottom-right', rawX: targetSheet.x1 - width, rawY: belowY, bias: 31 },
      { side: 'top-left', rawX: leftX, rawY: targetSheet.y0, bias: 31 },
      { side: 'bottom-left', rawX: targetSheet.x0, rawY: belowY, bias: 33 },
    );
  }

  let bestX = clamp(candidates[0].rawX, minX, maxX);
  let bestY = clamp(candidates[0].rawY, minY, maxTopY);
  let bestSide: PopoverPlacement['side'] = candidates[0].side;
  let bestScore = Infinity;

  const anchorHalo = 14;

  for (const cand of candidates) {
    const cx = clamp(cand.rawX, minX, maxX);
    const cy = clamp(cand.rawY, minY, maxTopY);
    const rect: OverlayRect = { x0: cx, y0: cy, x1: cx + width, y1: cy + height };

    let score = cand.bias;

    if (
      input.anchorX >= rect.x0 - anchorHalo &&
      input.anchorX <= rect.x1 + anchorHalo &&
      input.anchorY >= rect.y0 - anchorHalo &&
      input.anchorY <= rect.y1 + anchorHalo
    ) {
      score += 10000;
    }

    for (const avoid of avoidRects) {
      const area = rectOverlapArea(rect, avoid, 3);
      if (area > 0) {
        score += 1200 + area * 1.2;
      }
    }

    for (const occ of occupied) {
      const area = rectOverlapArea(rect, occ, 3);
      if (area > 0) {
        score += 450 + area * 0.45;
      }
    }

    const drift = Math.hypot(cx - cand.rawX, cy - cand.rawY);
    score += drift * 1.4;

    if (score < bestScore) {
      bestScore = score;
      bestX = cx;
      bestY = cy;
      bestSide = cand.side;
    }
  }

  const finalRect: OverlayRect = {
    x0: bestX,
    y0: bestY,
    x1: bestX + width,
    y1: bestY + height,
  };
  const border = pickRectBorderPoint(finalRect, input.anchorX, input.anchorY);

  return {
    x: bestX,
    y: bestY,
    width,
    height,
    side: bestSide,
    edgeX: border.edgeX,
    edgeY: border.edgeY,
    exitOrientation: border.orientation,
  };
}

/**
 * Build a crisp SVG quadratic Bezier curve from popover edge `(fromX, fromY)` to 3D cell
 * screen anchor `(toX, toY)`.
 */
export function buildQuadraticPointerPath(
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  exitOrientation: 'vertical' | 'horizontal' | 'auto' = 'auto',
): string {
  const dx = toX - fromX;
  const dy = toY - fromY;
  const dist = Math.hypot(dx, dy);
  if (dist < 2) return '';

  const mode =
    exitOrientation === 'auto' ? (Math.abs(dy) >= Math.abs(dx) ? 'vertical' : 'horizontal') : exitOrientation;

  let ctrlX: number;
  let ctrlY: number;
  if (mode === 'vertical') {
    ctrlX = fromX + dx * 0.16;
    ctrlY = fromY + dy * 0.62;
  } else {
    ctrlX = fromX + dx * 0.62;
    ctrlY = fromY + dy * 0.16;
  }

  return `M${fromX.toFixed(1)},${fromY.toFixed(1)}Q${ctrlX.toFixed(1)},${ctrlY.toFixed(1)} ${toX.toFixed(1)},${toY.toFixed(1)}`;
}

/**
 * Build a subtle drafting-table SVG elbow leader connecting projected world anchor `(anchorX, anchorY)`
 * to the nearest edge of a nudged exterior label rectangle.
 */
export function buildElbowLeaderPath(anchorX: number, anchorY: number, rect: OverlayRect): string {
  const w = rect.x1 - rect.x0;
  const h = rect.y1 - rect.y0;
  const insetX = Math.min(6, w * 0.25);

  let edgeX: number;
  let edgeY: number;
  let horizontalShoulder = false;

  if (anchorX < rect.x0) {
    edgeX = rect.x0;
    edgeY = rect.y0 + h * 0.5;
    horizontalShoulder = true;
  } else if (anchorX > rect.x1) {
    edgeX = rect.x1;
    edgeY = rect.y0 + h * 0.5;
    horizontalShoulder = true;
  } else if (anchorY < rect.y0) {
    edgeX = clamp(anchorX, rect.x0 + insetX, rect.x1 - insetX);
    edgeY = rect.y0;
  } else {
    edgeX = clamp(anchorX, rect.x0 + insetX, rect.x1 - insetX);
    edgeY = rect.y1;
  }

  const dx = edgeX - anchorX;
  const dy = edgeY - anchorY;
  if (Math.hypot(dx, dy) < 2) return '';

  if (Math.abs(dx) <= 4 || Math.abs(dy) <= 4) {
    return `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${edgeX.toFixed(1)},${edgeY.toFixed(1)}`;
  }

  if (horizontalShoulder) {
    const stub = Math.min(Math.abs(dx) * 0.45, 10);
    const kneeX = edgeX - Math.sign(dx) * stub;
    return `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${kneeX.toFixed(1)},${edgeY.toFixed(1)}L${edgeX.toFixed(1)},${edgeY.toFixed(1)}`;
  }

  const stub = Math.min(Math.abs(dy) * 0.45, 8);
  const kneeY = edgeY - Math.sign(dy) * stub;
  return `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${edgeX.toFixed(1)},${kneeY.toFixed(1)}L${edgeX.toFixed(1)},${edgeY.toFixed(1)}`;
}

/**
 * Order `(col, row)` coordinates of a sheet grid prioritizing:
 *  0. the active cell `(activeCol, activeRow)`
 *  1. cells sharing `activeRow` or `activeCol` (nearest first)
 *  2. remaining cells nearest to the active cell (or sheet center when no active cell is set).
 */
export function prioritizeSheetCellCoords(
  cols: number,
  rows: number,
  activeCol?: number,
  activeRow?: number,
): readonly (readonly [number, number])[] {
  const safeCols = Math.max(0, Math.floor(cols));
  const safeRows = Math.max(0, Math.floor(rows));
  if (safeCols === 0 || safeRows === 0) return [];

  const hasActiveCol = activeCol !== undefined && activeCol >= 0 && activeCol < safeCols;
  const hasActiveRow = activeRow !== undefined && activeRow >= 0 && activeRow < safeRows;
  const refCol = hasActiveCol ? activeCol! : (safeCols - 1) * 0.5;
  const refRow = hasActiveRow ? activeRow! : (safeRows - 1) * 0.5;

  interface CellEntry {
    col: number;
    row: number;
    tier: number;
    dist: number;
  }

  const entries: CellEntry[] = [];
  for (let r = 0; r < safeRows; r++) {
    for (let c = 0; c < safeCols; c++) {
      let tier = 2;
      if (hasActiveCol && hasActiveRow && c === activeCol && r === activeRow) {
        tier = 0;
      } else if ((hasActiveRow && r === activeRow) || (hasActiveCol && c === activeCol)) {
        tier = 1;
      }
      entries.push({
        col: c,
        row: r,
        tier,
        dist: Math.abs(c - refCol) + Math.abs(r - refRow),
      });
    }
  }

  entries.sort((a, b) => {
    if (a.tier !== b.tier) return a.tier - b.tier;
    if (a.dist !== b.dist) return a.dist - b.dist;
    if (a.row !== b.row) return a.row - b.row;
    return a.col - b.col;
  });

  return entries.map((e) => [e.col, e.row] as const);
}
