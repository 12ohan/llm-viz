import {
  bayKindLabel,
  formatBayId,
  formatBracketCaption,
  type NemotronSpec,
  type Vec3,
} from '../model/spec';
import type { TocSectionId } from '../model/state';
import { dot3, projectWorldToHostCss, sub3, type CameraSnapshot } from '../render/camera';
import { localToWorldOnFrame, makeSheetFrame3D } from '../render/sheetFrame';
import type { CutawaySceneGeometry } from '../scene/cutaways';
import type { LandmarkAnchor, MacroSceneGeometry } from '../scene/spine';

export interface OverlayUpdateInput {
  readonly spec: NemotronSpec;
  readonly scene: MacroSceneGeometry;
  readonly snapshot: CameraSnapshot;
  readonly section: TocSectionId;
  readonly tokens: number;
  readonly selectedBay: number | null;
  readonly hoveredBay: number | null;
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
}

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

export interface CutawayOverlayUpdateInput {
  readonly scene: CutawaySceneGeometry;
  readonly snapshot: CameraSnapshot;
  readonly tokens: number;
  readonly selectedPart: string | null;
  readonly hoveredPart: string | null;
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  /** When present and non-empty, non-participating sheet labels dim cleanly to opacity 0.46. */
  readonly activeOpPartIds?: ReadonlySet<string> | null;
  /** Floating contextual cell formula popover + curved provenance pointers. */
  readonly cellPopover?: CellPopoverSpec | null;
  /** Active/inspected thin sheets eligible for zoom-adaptive on-cell values & axis headers. */
  readonly activeCellGrids?: readonly ActiveSheetCellGrid[] | null;
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

interface PlacedRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
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

const SVG_NS = 'http://www.w3.org/2000/svg';
const OVERLAY_V3_STYLE_ID = 'nemotron-overlay-v3-defaults';
const MAX_OPERAND_DOTS = 4;

const OPERAND_SWATCH_HEX: Readonly<Record<'indigo' | 'teal' | 'red', string>> = {
  indigo: '#3861BD',
  teal: '#1A8A7A',
  red: '#DC4C34',
};

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function overlapsAny(rects: readonly OverlayRect[], x0: number, y0: number, x1: number, y1: number): boolean {
  const pad = 3;
  for (const r of rects) {
    if (x0 < r.x1 + pad && x1 + pad > r.x0 && y0 < r.y1 + pad && y1 + pad > r.y0) {
      return true;
    }
  }
  return false;
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
    // SHEET_REVEAL_UNCOMPUTED (3): entire sheet is uncomputed future state
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
    // SHEET_REVEAL_COL_MAJOR (2): reveal column by column
    return col > colFrontier || (col === colFrontier && row > rowFrontier);
  }
  // SHEET_REVEAL_ROW_MAJOR (1): reveal row by row
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
 * Fractional or out-of-range coordinates (e.g. `col = -0.5` for exterior row headers) extrapolate
 * linearly along the sheet plane. `faceLy` selects the thickness offset (`0.5` = `+axisN` top/front
 * face, `0` = midplane, `-0.5` = `-axisN` back face).
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

/**
 * Project the active cell (or sheet center) and adjacent cell centers `(col+1, row)` and
 * `(col, row+1)` to measure projected cell dimensions `(cellW, cellH)` in CSS pixels.
 * Uses the projected cell parallelogram's perpendicular spacings so oblique or nearly edge-on
 * views cleanly suppress overlapping numbers.
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

    // Symmetric half-cell span around reference center stays inside single-cell sheets (cols=1/rows=1)
    // while matching exact projected distance to (col+1, row) and (col, row+1).
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
      // Fallback: forward step to (rc+1, rr) and (rc, rr+1)
      const sNextU = projectWorldToHostCss(
        localToWorldOnFrame(frame, (rc + 1.5) / safeCols - 0.5, faceLy, (rr + 0.5) / safeRows - 0.5),
        snapshot,
      );
      const sNextV = projectWorldToHostCss(
        localToWorldOnFrame(frame, (rc + 0.5) / safeCols - 0.5, faceLy, (rr + 1.5) / safeRows - 0.5),
        snapshot,
      );
      if (!sNextU.visible || !sNextV.visible) continue;
      const dux = sNextU.x - sCenter.x;
      const duy = sNextU.y - sCenter.y;
      const dvx = sNextV.x - sCenter.x;
      const dvy = sNextV.y - sCenter.y;
      return measureParallelogramCell(dux, duy, dvx, dvy);
    }

    const dux = sU1.x - sU0.x;
    const duy = sU1.y - sU0.y;
    const dvx = sV1.x - sV0.x;
    const dvy = sV1.y - sV0.y;
    return measureParallelogramCell(dux, duy, dvx, dvy);
  }

  return { cellW: 0, cellH: 0, colStepPx: 0, rowStepPx: 0, visible: false };
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
 * Pick the optimal attachment point on a rectangle's perimeter facing `(targetX, targetY)`.
 */
function pickRectBorderPoint(
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

  // Fallback if target sits inside rect: attach to nearest horizontal or vertical edge
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
 * Intelligently place the floating contextual cell popover offset above/beside the active 3D
 * cell anchor, avoiding `occupied` label rectangles, `avoidRects` (active sheets, upstream
 * operands, and provenance midpoints), and clamping strictly inside
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

  // Ensure targetSheetRect is penalized even when passed without duplicating it in avoidRects
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
    // Outer-ring ±(gapX + 48) horizontal fallbacks
    { side: 'top-right', rawX: input.anchorX + outerGapX, rawY: input.anchorY - height - gapY, bias: 26 },
    { side: 'top-left', rawX: input.anchorX - width - outerGapX, rawY: input.anchorY - height - gapY, bias: 30 },
    { side: 'top-right', rawX: input.anchorX + outerGapX, rawY: input.anchorY - height * 0.5, bias: 32 },
    { side: 'bottom-right', rawX: input.anchorX + outerGapX, rawY: input.anchorY + gapY, bias: 34 },
    { side: 'top-left', rawX: input.anchorX - width - outerGapX, rawY: input.anchorY - height * 0.5, bias: 36 },
    { side: 'bottom-left', rawX: input.anchorX - width - outerGapX, rawY: input.anchorY + gapY, bias: 38 },
  ];

  // Target-sheet-exterior candidates keep the popover cleanly OFF zoomed active matrices
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

    // Heavily penalize covering the active 3D cell itself
    if (
      input.anchorX >= rect.x0 - anchorHalo &&
      input.anchorX <= rect.x1 + anchorHalo &&
      input.anchorY >= rect.y0 - anchorHalo &&
      input.anchorY <= rect.y1 + anchorHalo
    ) {
      score += 10000;
    }

    // Strongly penalize overlapping active sheets, upstream operands, or dataflow midpoints
    for (const avoid of avoidRects) {
      const area = rectOverlapArea(rect, avoid, 3);
      if (area > 0) {
        score += 1200 + area * 1.2;
      }
    }

    // Penalize collisions with existing exterior labels / rects
    for (const occ of occupied) {
      const area = rectOverlapArea(rect, occ, 3);
      if (area > 0) {
        score += 450 + area * 0.45;
      }
    }

    // Penalize excessive clamping displacement from natural offset
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

/**
 * Inject low-precedence (`@layer` + prepended in `<head>`) default styles for V3 floating callouts,
 * curved provenance pointers, zoom-adaptive cell readouts, axis headers, and dimmed labels so the
 * overlay renders cleanly out-of-the-box while remaining 100% overridable by `theme.css`.
 */
function ensureOverlayV3Styles(): void {
  if (typeof document === 'undefined' || !document.head) return;
  if (typeof document.getElementById === 'function' && document.getElementById(OVERLAY_V3_STYLE_ID)) {
    return;
  }
  const style = document.createElement('style');
  style.id = OVERLAY_V3_STYLE_ID;
  style.textContent = `@layer overlay-v3-defaults {
  .ov-sheet-label { transition: opacity 110ms ease; }
  .ov-sheet-label.is-dimmed { opacity: 0.46; }
  @media (prefers-reduced-motion: reduce) {
    .ov-sheet-label { transition: none !important; }
  }
  .overlay-popover-leader { fill: none; stroke: #14181F; stroke-width: 1.5; stroke-linecap: round; }
  .overlay-popover-dot { fill: #DC4C34; stroke: #FFFFFF; stroke-width: 1.4; }
  .overlay-popover-source { fill: none; stroke-width: 1.25; stroke-dasharray: 3 2.5; stroke-linecap: round; stroke-opacity: 0.68; }
  .overlay-popover-source-indigo { stroke: #3861BD; }
  .overlay-popover-source-teal { stroke: #1A8A7A; }
  .ov-cell-popover {
    position: absolute; top: 0; left: 0; pointer-events: none; z-index: 8;
    display: flex; flex-direction: column; gap: 3px; max-width: 396px;
    padding: 5px 8px; border-radius: 6px;
    background-color: rgba(20, 24, 31, 0.95);
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: #F8FAFC;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.24);
    overflow: hidden; box-sizing: border-box;
    will-change: transform;
  }
  .ov-cell-popover-head {
    display: flex; align-items: center; gap: 5px;
    min-width: 0; max-width: 100%; overflow: hidden;
    font-size: 10px; line-height: 13px; white-space: nowrap;
  }
  .ov-cell-popover-tensor {
    flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis;
    font-family: var(--font-sans, system-ui, -apple-system, sans-serif);
    font-weight: 700; color: #FFFFFF;
  }
  .ov-cell-popover-coord {
    flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis;
    font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-weight: 600; color: #94A3B8; font-variant-numeric: tabular-nums;
  }
  .ov-cell-popover-badge {
    flex: 0 1 auto; max-width: 112px; min-width: 0; overflow: hidden; text-overflow: ellipsis;
    margin-left: auto; padding: 0 4px; border-radius: 3px;
    background-color: rgba(255, 255, 255, 0.12); color: #E2E8F0;
    font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 9px; font-weight: 600;
  }
  .ov-cell-popover-body {
    display: flex; align-items: center; gap: 6px;
    min-width: 0; max-width: 100%; overflow: hidden; white-space: nowrap;
  }
  .ov-cell-popover-dots {
    display: inline-flex; align-items: center; gap: 3px; flex-shrink: 0;
  }
  .ov-operand-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
  }
  .ov-cell-popover-eq {
    flex: 1 1 auto; min-width: 0;
    font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 10.5px; line-height: 14px; color: #E2E8F0;
    overflow: hidden; text-overflow: ellipsis;
  }
  .ov-cell-popover-val {
    margin-left: auto; flex-shrink: 0; padding: 1px 5px; border-radius: 3px;
    background-color: #C83E27; color: #FFFFFF;
    font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 10.5px; line-height: 14px; font-weight: 700;
    font-variant-numeric: tabular-nums;
  }
  .ov-cell-popover.is-accumulating {
    border-color: rgba(217, 119, 6, 0.55);
  }
  .ov-cell-popover-val.is-masked {
    background-color: rgba(100, 116, 139, 0.45); color: #E2E8F0;
  }
  .ov-cell-popover-val.is-accumulating {
    background-color: #B45309; color: #FFFFFF;
  }
  .ov-cell-num {
    position: absolute; top: 0; left: 0; height: 13px; padding: 0 3px;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 2px; pointer-events: none; z-index: 2;
    background-color: rgba(248, 250, 252, 0.86); color: #0F172A;
    font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 9px; line-height: 12px; font-weight: 600;
    font-variant-numeric: tabular-nums; white-space: nowrap;
    box-shadow: 0 0 0 0.5px rgba(15, 23, 42, 0.12);
    will-change: transform;
  }
  .ov-cell-num.is-active {
    background-color: #C83E27; color: #FFFFFF; font-weight: 700; z-index: 5;
    box-shadow: 0 1px 3px rgba(200, 62, 39, 0.35);
  }
  .ov-cell-num.is-accumulating {
    background-color: #B45309; color: #FFFFFF; font-weight: 700; z-index: 6;
    box-shadow: 0 1px 3px rgba(180, 83, 9, 0.38);
  }
  .ov-cell-num.is-masked {
    background-color: rgba(241, 245, 249, 0.74); color: #475569; font-weight: 500;
  }
  .ov-axis-header {
    position: absolute; top: 0; left: 0; height: 15px; padding: 0 4px;
    display: inline-flex; align-items: center;
    border-radius: 3px; pointer-events: none; z-index: 3;
    background-color: rgba(248, 249, 251, 0.94);
    border: 1px solid rgba(148, 163, 184, 0.45);
    color: #334155;
    font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 9.5px; line-height: 13px; font-weight: 600;
    font-variant-numeric: tabular-nums; white-space: nowrap;
    will-change: transform;
  }
  .ov-axis-header.is-active {
    background-color: #14181F; border-color: #14181F; color: #FFFFFF; z-index: 5;
  }
  .ov-axis-header.is-dim {
    background-color: rgba(238, 242, 248, 0.94);
    border-color: rgba(56, 97, 189, 0.36); color: #2B4C96;
  }
}`;
  if (typeof document.head.prepend === 'function') {
    document.head.prepend(style);
  } else {
    document.head.appendChild(style);
  }
}

/**
 * Synchronous DOM + SVG overlay updated inside WebGL's requestAnimationFrame pass.
 * Renders crisp llm-viz dark charcoal/black 3D pills (#14181F, white text) for the
 * 52-Layer Tower and thin-sheet exterior labels, SVG elbow leaders, active-op dimming,
 * Bycroft-style floating contextual formula popovers with curved provenance pointers,
 * and zoom-adaptive on-cell numeric readouts + exterior axis headers for exploded cutaways.
 */
export class MacroOverlay {
  private readonly svg: SVGSVGElement;
  private readonly labelLayer: HTMLDivElement;
  private readonly leaderPath: SVGPathElement;
  private readonly phaseBracketPath: SVGPathElement;
  private readonly caliperPath: SVGPathElement;
  private readonly popoverSourceIndigoPath: SVGPathElement;
  private readonly popoverSourceTealPath: SVGPathElement;
  private readonly popoverLeaderPath: SVGPathElement;
  private readonly popoverAnchorDot: SVGCircleElement;

  private readonly landmarkNodes = new Map<string, HTMLDivElement>();
  private readonly cutawayNodes = new Map<string, HTMLDivElement>();
  private readonly caliperNodes = new Map<string, HTMLDivElement>();
  private readonly phaseNodes: HTMLDivElement[] = [];
  private readonly bayMiniNodes = new Map<number, HTMLDivElement>();
  private readonly kvTickNodes = new Map<number, HTMLDivElement>();
  private readonly activePill: HTMLDivElement;

  // V3 pre-allocated pools (zero per-frame DOM creation / GC)
  private readonly cellNumNodes: HTMLDivElement[] = [];
  private readonly axisHeaderNodes: HTMLDivElement[] = [];
  private readonly cellPopover: HTMLDivElement;
  private readonly popoverTensorEl: HTMLSpanElement;
  private readonly popoverCoordEl: HTMLSpanElement;
  private readonly popoverBadgeEl: HTMLSpanElement;
  private readonly popoverDotsWrap: HTMLSpanElement;
  private readonly popoverDotEls: HTMLSpanElement[] = [];
  private readonly popoverEqEl: HTMLSpanElement;
  private readonly popoverValEl: HTMLSpanElement;

  private readonly measureCtx: CanvasRenderingContext2D | null;
  private readonly widthCache = new Map<string, number>();

  constructor(svg: SVGSVGElement, labelLayer: HTMLDivElement) {
    ensureOverlayV3Styles();

    this.svg = svg;
    this.labelLayer = labelLayer;
    this.svg.innerHTML = '';
    this.labelLayer.innerHTML = '';

    const scratch = document.createElement('canvas');
    this.measureCtx = typeof scratch.getContext === 'function' ? scratch.getContext('2d') : null;

    this.phaseBracketPath = this.createSvgPath('overlay-phase-bracket');
    this.caliperPath = this.createSvgPath('overlay-caliper');
    this.leaderPath = this.createSvgPath('overlay-leader');
    this.popoverSourceIndigoPath = this.createSvgPath(
      'overlay-popover-source overlay-popover-source-indigo',
    );
    this.popoverSourceIndigoPath.setAttribute('fill', 'none');
    this.popoverSourceIndigoPath.setAttribute('stroke', OPERAND_SWATCH_HEX.indigo);
    this.popoverSourceIndigoPath.setAttribute('stroke-width', '1.25');
    this.popoverSourceIndigoPath.setAttribute('stroke-dasharray', '3 2.5');
    this.popoverSourceIndigoPath.setAttribute('stroke-opacity', '0.68');

    this.popoverSourceTealPath = this.createSvgPath(
      'overlay-popover-source overlay-popover-source-teal',
    );
    this.popoverSourceTealPath.setAttribute('fill', 'none');
    this.popoverSourceTealPath.setAttribute('stroke', OPERAND_SWATCH_HEX.teal);
    this.popoverSourceTealPath.setAttribute('stroke-width', '1.25');
    this.popoverSourceTealPath.setAttribute('stroke-dasharray', '3 2.5');
    this.popoverSourceTealPath.setAttribute('stroke-opacity', '0.68');

    this.popoverLeaderPath = this.createSvgPath('overlay-popover-leader');
    this.popoverLeaderPath.setAttribute('fill', 'none');
    this.popoverLeaderPath.setAttribute('stroke', '#14181F');
    this.popoverLeaderPath.setAttribute('stroke-width', '1.5');
    this.popoverLeaderPath.setAttribute('stroke-linecap', 'round');

    this.popoverAnchorDot = document.createElementNS(SVG_NS, 'circle');
    this.popoverAnchorDot.setAttribute('class', 'overlay-popover-dot');
    this.popoverAnchorDot.setAttribute('r', '3');
    this.popoverAnchorDot.setAttribute('fill', '#DC4C34');
    this.popoverAnchorDot.setAttribute('stroke', '#FFFFFF');
    this.popoverAnchorDot.setAttribute('stroke-width', '1.4');
    this.popoverAnchorDot.style.display = 'none';

    this.svg.append(
      this.phaseBracketPath,
      this.caliperPath,
      this.leaderPath,
      this.popoverSourceIndigoPath,
      this.popoverSourceTealPath,
      this.popoverLeaderPath,
      this.popoverAnchorDot,
    );

    // Pre-allocate dark landmark pills for tower
    const landmarkIds = [
      'embed',
      'gqa-6',
      'gqa-13',
      'gqa-20',
      'gqa-27',
      'gqa-34',
      'gqa-43',
      'final-norm',
      'output',
    ];
    for (const id of landmarkIds) {
      const el = document.createElement('div');
      el.className = 'ov-pill';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.landmarkNodes.set(id, el);
    }

    // Pre-allocate 4 Figure-2 phase section tags
    for (let i = 0; i < 4; i++) {
      const el = document.createElement('div');
      el.className = 'ov-phase-tag';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.phaseNodes.push(el);
    }

    // Pre-allocate subtle progressive-zoom bay pills (1..52)
    for (let id = 1; id <= 52; id++) {
      const el = document.createElement('div');
      el.className = 'ov-bay-mini';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.bayMiniNodes.set(id, el);
    }

    // Pre-allocate KV depth ticks (1..16)
    for (let t = 1; t <= 16; t++) {
      const el = document.createElement('div');
      el.className = 'ov-kv-tick';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.kvTickNodes.set(t, el);
    }

    // Active hovered / selected bay pill
    this.activePill = document.createElement('div');
    this.activePill.className = 'ov-pill ov-pill-active';
    this.activePill.style.display = 'none';
    this.labelLayer.appendChild(this.activePill);

    // V3: Pre-allocate fixed pool of 64 reusable on-cell numeric readout nodes
    for (let i = 0; i < MAX_OVERLAY_CELL_NUM_NODES; i++) {
      const el = document.createElement('div');
      el.className = 'ov-cell-num';
      el.style.display = 'none';
      el.style.pointerEvents = 'none';
      this.labelLayer.appendChild(el);
      this.cellNumNodes.push(el);
    }

    // V3: Pre-allocate fixed pool of 24 reusable exterior axis header nodes
    for (let i = 0; i < MAX_OVERLAY_AXIS_HEADER_NODES; i++) {
      const el = document.createElement('div');
      el.className = 'ov-axis-header';
      el.style.display = 'none';
      el.style.pointerEvents = 'none';
      this.labelLayer.appendChild(el);
      this.axisHeaderNodes.push(el);
    }

    // V3: Pre-allocate single reusable Bycroft-grade floating cell popover node
    this.cellPopover = document.createElement('div');
    this.cellPopover.className = 'ov-cell-popover';
    this.cellPopover.style.display = 'none';
    this.cellPopover.style.pointerEvents = 'none';
    this.cellPopover.style.zIndex = '8';

    const headRow = document.createElement('div');
    headRow.className = 'ov-cell-popover-head';
    this.popoverTensorEl = document.createElement('span');
    this.popoverTensorEl.className = 'ov-cell-popover-tensor';
    this.popoverCoordEl = document.createElement('span');
    this.popoverCoordEl.className = 'ov-cell-popover-coord';
    this.popoverBadgeEl = document.createElement('span');
    this.popoverBadgeEl.className = 'ov-cell-popover-badge';
    this.popoverBadgeEl.style.display = 'none';
    headRow.append(this.popoverTensorEl, this.popoverCoordEl, this.popoverBadgeEl);

    const bodyRow = document.createElement('div');
    bodyRow.className = 'ov-cell-popover-body';
    this.popoverDotsWrap = document.createElement('span');
    this.popoverDotsWrap.className = 'ov-cell-popover-dots';
    this.popoverDotsWrap.style.display = 'none';
    for (let i = 0; i < MAX_OPERAND_DOTS; i++) {
      const dot = document.createElement('span');
      dot.className = 'ov-operand-dot';
      dot.style.display = 'none';
      this.popoverDotsWrap.appendChild(dot);
      this.popoverDotEls.push(dot);
    }
    this.popoverEqEl = document.createElement('span');
    this.popoverEqEl.className = 'ov-cell-popover-eq';
    this.popoverValEl = document.createElement('span');
    this.popoverValEl.className = 'ov-cell-popover-val';
    bodyRow.append(this.popoverDotsWrap, this.popoverEqEl, this.popoverValEl);

    this.cellPopover.append(headRow, bodyRow);
    this.labelLayer.appendChild(this.cellPopover);
  }

  private createSvgPath(className: string): SVGPathElement {
    const p = document.createElementNS(SVG_NS, 'path');
    p.setAttribute('class', className);
    return p;
  }

  private getOrCreateCutawayNode(id: string): HTMLDivElement {
    let el = this.cutawayNodes.get(id);
    if (!el) {
      el = document.createElement('div');
      el.className = 'ov-pill';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.cutawayNodes.set(id, el);
    }
    return el;
  }

  private getOrCreateCaliperNode(id: string): HTMLDivElement {
    let el = this.caliperNodes.get(id);
    if (!el) {
      el = document.createElement('div');
      el.className = 'ov-caliper-label';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.caliperNodes.set(id, el);
    }
    return el;
  }

  private hideAll(): void {
    for (const el of this.landmarkNodes.values()) el.style.display = 'none';
    for (const el of this.cutawayNodes.values()) {
      el.style.display = 'none';
      el.style.opacity = '1';
      el.classList.remove('is-dimmed');
    }
    for (const el of this.caliperNodes.values()) el.style.display = 'none';
    for (const el of this.phaseNodes) el.style.display = 'none';
    for (const el of this.bayMiniNodes.values()) el.style.display = 'none';
    for (const el of this.kvTickNodes.values()) el.style.display = 'none';
    for (const el of this.cellNumNodes) {
      el.style.display = 'none';
      el.classList.remove('is-accumulating');
    }
    for (const el of this.axisHeaderNodes) el.style.display = 'none';
    this.activePill.style.display = 'none';
    this.cellPopover.style.display = 'none';
    this.cellPopover.classList.remove('is-accumulating');
    this.popoverValEl.classList.remove('is-accumulating');
    this.caliperPath.setAttribute('d', '');
    this.popoverLeaderPath.setAttribute('d', '');
    this.popoverSourceIndigoPath.setAttribute('d', '');
    this.popoverSourceTealPath.setAttribute('d', '');
    this.popoverAnchorDot.style.display = 'none';
  }

  clearMetricCache(): void {
    this.widthCache.clear();
  }

  get metricCacheSize(): number {
    return this.widthCache.size;
  }

  private measureText(text: string, font: string): number {
    const key = `${font}|${text}`;
    const cached = this.widthCache.get(key);
    if (cached !== undefined) return cached;
    let w: number;
    if (this.measureCtx) {
      this.measureCtx.font = font;
      w = Math.ceil(this.measureCtx.measureText(text).width);
    } else {
      w = Math.ceil(text.length * 6.8);
    }
    if (this.widthCache.size >= MAX_OVERLAY_WIDTH_CACHE_ENTRIES) {
      const oldestKey = this.widthCache.keys().next().value;
      if (oldestKey !== undefined) {
        this.widthCache.delete(oldestKey);
      }
    }
    this.widthCache.set(key, w);
    return w;
  }

  /** Tower overlay update (52-Layer Macro Spine). */
  update(input: OverlayUpdateInput): void {
    const { spec, scene, snapshot, section, tokens, selectedBay, hoveredBay } = input;
    const { hostWidth, hostHeight } = snapshot.metrics;
    this.svg.setAttribute('viewBox', `0 0 ${hostWidth} ${hostHeight}`);
    this.hideAll();

    const occupied: PlacedRect[] = [];
    const leaderSegs: string[] = [];
    const phaseSegs: string[] = [];

    // Safe drawing area avoiding floating two-row mode/preset toolbar (~84px desktop, ~12px stacked/compact) and bottom hint (~30px)
    const minY = Math.max(8, input.topInsetPx ?? (hostWidth <= 900 || hostHeight <= 440 ? 12 : 84));
    const maxY = Math.max(minY + 40, hostHeight - Math.max(8, input.bottomInsetPx ?? 30));

    const pillFont = '600 11px system-ui, -apple-system, sans-serif';
    const monoFont = '500 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';

    // --- 1. Hovered / Pinned Bay Active Black Pill (highest priority) ---
    const activeBayId = hoveredBay ?? selectedBay;
    if (activeBayId !== null) {
      const bay = spec.bays[activeBayId - 1];
      const anchor = scene.bayAnchors[activeBayId - 1];
      if (bay && anchor) {
        const pt = projectWorldToHostCss(anchor.pillWorld, snapshot);
        if (pt.visible && pt.x > -40 && pt.x < hostWidth + 40 && pt.y > 0 && pt.y < hostHeight) {
          const role = bay.feeder ? ' · Feeder' : bay.kind === '*' ? ' · 2 KV' : '';
          const text = `${formatBayId(bay.id)} · ${bayKindLabel(bay.kind)}${role}`;
          this.activePill.textContent = text;
          const w = this.measureText(text, pillFont) + 18;
          const h = 22;
          const lx = clamp(pt.x + 14, 10, hostWidth - w - 10);
          const ly = clamp(pt.y - h * 0.5, minY, maxY - h);
          this.activePill.style.display = 'inline-flex';
          this.activePill.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
          occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });

          const anchorX = clamp(pt.x, 4, hostWidth - 4);
          const anchorY = clamp(pt.y, minY, maxY);
          const edgeX = lx > anchorX ? lx : lx + w;
          leaderSegs.push(
            `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${edgeX.toFixed(1)},${(ly + h * 0.5).toFixed(1)}`,
          );
        }
      }
    }

    // --- 2. Black-Pill Landmark Labels (Embedding, 6 GQA bays, Final RMSNorm, Output) ---
    for (const lm of scene.landmarks) {
      if (lm.bayId !== undefined && lm.bayId === activeBayId) continue;

      const el = this.landmarkNodes.get(lm.id);
      if (!el) continue;
      const pt = projectWorldToHostCss(lm.world, snapshot);
      if (!pt.visible || pt.x < 8 || pt.x > hostWidth - 8 || pt.y < minY - 12 || pt.y > maxY + 12) {
        continue;
      }

      const w = this.measureText(lm.label, pillFont) + 16;
      const h = 20;
      const lx = clamp(pt.x + 12, 10, hostWidth - w - 10);
      let ly = clamp(pt.y - h * 0.5, minY, maxY - h);

      if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
        const up = ly - (h + 4);
        const down = ly + (h + 4);
        if (up >= minY && !overlapsAny(occupied, lx, up, lx + w, up + h)) {
          ly = up;
        } else if (down + h <= maxY && !overlapsAny(occupied, lx, down, lx + w, down + h)) {
          ly = down;
        } else {
          continue;
        }
      }

      el.textContent = lm.label;
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });

      const edgeX = lx > pt.x ? lx : lx + w;
      leaderSegs.push(
        `M${pt.x.toFixed(1)},${pt.y.toFixed(1)}L${edgeX.toFixed(1)},${(ly + h * 0.5).toFixed(1)}`,
      );
    }

    // --- 3. Progressive Zoom Bay Labels when zoomed close ---
    const p1 = projectWorldToHostCss(scene.bayAnchors[0].centerWorld, snapshot);
    const p2 = projectWorldToHostCss(scene.bayAnchors[1].centerWorld, snapshot);
    const screenBayPitch = p1.visible && p2.visible ? Math.hypot(p2.x - p1.x, p2.y - p1.y) : 0;

    if (screenBayPitch >= 24) {
      for (const anchor of scene.bayAnchors) {
        if (anchor.id === activeBayId) continue;
        const bay = spec.bays[anchor.id - 1];
        if (!bay || bay.kind === '*') continue;
        const pt = projectWorldToHostCss(anchor.pillWorld, snapshot);
        if (!pt.visible || pt.x < 12 || pt.x > hostWidth - 60 || pt.y < minY || pt.y > maxY) {
          continue;
        }
        const text = `${formatBayId(bay.id)} · ${bay.kind}${bay.feeder ? ' feeder' : ''}`;
        const w = this.measureText(text, monoFont) + 12;
        const h = 18;
        const lx = clamp(pt.x + 10, 8, hostWidth - w - 8);
        const ly = clamp(pt.y - h * 0.5, minY, maxY - h);
        if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) continue;

        const el = this.bayMiniNodes.get(bay.id);
        if (!el) continue;
        el.textContent = text;
        el.classList.toggle('is-feeder', bay.feeder);
        el.style.display = 'inline-flex';
        el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
        occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
      }
    }

    // --- 4. NVIDIA Figure-2 Phase Section Brackets along left side of tower ---
    for (let i = 0; i < scene.phaseSections.length; i++) {
      const sec = scene.phaseSections[i];
      const topPt = projectWorldToHostCss([-6.2, sec.topY, 0], snapshot);
      const botPt = projectWorldToHostCss([-6.2, sec.bottomY, 0], snapshot);
      const midPt = projectWorldToHostCss(sec.midWorld, snapshot);
      if (!midPt.visible || midPt.y < minY || midPt.y > maxY || midPt.x < 24 || midPt.x > hostWidth - 24) {
        continue;
      }

      const caption = formatBracketCaption(spec.brackets[sec.index]);
      const w = this.measureText(caption, monoFont) + 10;
      const h = 17;
      const bx = clamp(midPt.x - 10, 8, hostWidth - 8);
      const lx = clamp(bx - w - 8, 8, hostWidth - w - 8);
      let ly = clamp(midPt.y - h * 0.5, minY, maxY - h);

      if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
        const down = ly + h + 3;
        if (down + h <= maxY && !overlapsAny(occupied, lx, down, lx + w, down + h)) {
          ly = down;
        } else {
          continue;
        }
      }

      const el = this.phaseNodes[i];
      el.textContent = caption;
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });

      if (topPt.visible && botPt.visible) {
        const ty = clamp(topPt.y, minY, maxY);
        const by = clamp(botPt.y, minY, maxY);
        const tickX = clamp(Math.min(topPt.x, botPt.x) - 6, lx + w + 4, hostWidth - 8);
        if (by - ty > 8 && tickX > lx + w + 2) {
          phaseSegs.push(
            `M${(tickX + 4).toFixed(1)},${ty.toFixed(1)}L${tickX.toFixed(1)},${ty.toFixed(1)}L${tickX.toFixed(1)},${by.toFixed(1)}L${(tickX + 4).toFixed(1)},${by.toFixed(1)}`,
          );
        }
      }
    }

    // --- 5. KV Depth Scale Ticks (1, 8, 16, active T) — hidden in Full Overview (Part A #2) ---
    const showKvTicks = section !== 'overview' && screenBayPitch >= 20;
    if (showKvTicks) {
      const orderedTicks = [...scene.kvDepthTicks].sort((a, b) =>
        a.token === tokens ? -1 : b.token === tokens ? 1 : a.token - b.token,
      );
      for (const tick of orderedTicks) {
        const pt = projectWorldToHostCss(tick.world, snapshot);
        if (!pt.visible || pt.x < 12 || pt.x > hostWidth - 34 || pt.y < minY || pt.y > maxY) {
          continue;
        }
        const isCur = tick.token === tokens;
        const text = isCur ? `T=${tick.token}` : `${tick.token}`;
        const w = this.measureText(text, monoFont) + 8;
        const h = 15;
        const lx = clamp(pt.x + 6, 8, hostWidth - w - 8);
        const ly = clamp(pt.y + 6, minY, maxY - h);
        if (!isCur && overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
          continue;
        }
        const el = this.kvTickNodes.get(tick.token);
        if (!el) continue;
        el.textContent = text;
        el.classList.toggle('is-current', isCur);
        el.style.display = 'inline-flex';
        el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
        occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
      }
    }

    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.phaseBracketPath.setAttribute('d', phaseSegs.join(''));
  }

  /** Exploded 3D Layer Cutaway overlay update (Mamba-2, MoE 128+1, GQA Attention, Embedding, Output). */
  updateCutaway(input: CutawayOverlayUpdateInput): void {
    const { scene, snapshot, tokens, selectedPart, hoveredPart } = input;
    const { hostWidth, hostHeight } = snapshot.metrics;
    this.svg.setAttribute('viewBox', `0 0 ${hostWidth} ${hostHeight}`);
    this.hideAll();

    const occupied: PlacedRect[] = [];
    const leaderSegs: string[] = [];

    const minY = input.topInsetPx ?? 84;
    const maxY = Math.max(minY + 40, hostHeight - (input.bottomInsetPx ?? 30));

    const pillFont = '600 11px system-ui, -apple-system, sans-serif';
    const monoFont = '500 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    const activePart = hoveredPart ?? selectedPart;

    // Thin-sheet drafting-table cutaways (GQA / Mamba-2 / MoE / Embedding / Output):
    // exterior edge-aligned labels, SVG elbow leaders on nudged labels, active-op dimming,
    // floating contextual formula callouts, and zoom-adaptive on-cell values & axis headers.
    if (scene.sheetCount && scene.sheetCount > 0) {
      this.updateSheetCutawayOverlay(
        input,
        activePart,
        occupied,
        leaderSegs,
        minY,
        maxY,
      );
      return;
    }

    const activeOpPartIds = input.activeOpPartIds;
    const hasActiveOpFocus = activeOpPartIds != null && activeOpPartIds.size > 0;

    // Prioritize active part pill, gold callouts, and NoPE callout so they never get crowded out
    const orderedLandmarks: LandmarkAnchor[] = [...scene.landmarks].sort((a, b) => {
      const pa = a.partId === activePart ? 3 : a.tone === 'gold' || a.id === 'g-nope' ? 2 : 1;
      const pb = b.partId === activePart ? 3 : b.tone === 'gold' || b.id === 'g-nope' ? 2 : 1;
      return pb - pa;
    });

    for (const lm of orderedLandmarks) {
      const pt = projectWorldToHostCss(lm.world, snapshot);
      if (!pt.visible || pt.x < -30 || pt.x > hostWidth + 30 || pt.y < minY - 24 || pt.y > maxY + 24) {
        continue;
      }

      const w = this.measureText(lm.label, pillFont) + 16;
      const h = 21;
      let lx = clamp(pt.x - w * 0.5, 10, hostWidth - w - 10);
      let ly = clamp(pt.y - h - 6, minY, maxY - h);

      // Try subtle vertical / horizontal nudges if two slab pills collide
      if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
        const candidates: readonly [number, number][] = [
          [lx, clamp(pt.y + 8, minY, maxY - h)],
          [lx, clamp(ly - (h + 5), minY, maxY - h)],
          [clamp(pt.x + 14, 10, hostWidth - w - 10), clamp(pt.y - h * 0.5, minY, maxY - h)],
          [clamp(pt.x - w - 14, 10, hostWidth - w - 10), clamp(pt.y - h * 0.5, minY, maxY - h)],
        ];
        let placed = false;
        for (const [cx, cy] of candidates) {
          if (!overlapsAny(occupied, cx, cy, cx + w, cy + h)) {
            lx = cx;
            ly = cy;
            placed = true;
            break;
          }
        }
        if (!placed && lm.partId !== activePart && lm.tone !== 'gold' && lm.id !== 'g-nope') {
          continue;
        }
      }

      const el = this.getOrCreateCutawayNode(lm.id);
      const isAct = lm.partId !== undefined && lm.partId === activePart;
      const isParticipating =
        !hasActiveOpFocus ||
        isAct ||
        (lm.partId !== undefined && activeOpPartIds!.has(lm.partId)) ||
        activeOpPartIds!.has(lm.id);

      el.className = 'ov-pill';
      if (isAct) el.classList.add('ov-pill-active');
      else if (lm.tone === 'gold') el.classList.add('ov-pill-gold');
      else if (lm.tone === 'cyan') el.classList.add('ov-pill-cyan');
      else if (lm.tone === 'muted') el.classList.add('ov-pill-muted');
      el.classList.toggle('is-dimmed', !isParticipating);
      el.style.opacity = isParticipating ? '1' : DIMMED_LABEL_OPACITY;

      el.textContent = lm.label;
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });

      // Crisp short vertical/diagonal leader connecting pill edge to slab anchor
      const anchorX = clamp(pt.x, 4, hostWidth - 4);
      const anchorY = clamp(pt.y, minY, maxY);
      const targetX = clamp(anchorX, lx + 8, lx + w - 8);
      const targetY = anchorY < ly ? ly : anchorY > ly + h ? ly + h : ly + h * 0.5;
      leaderSegs.push(
        `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${targetX.toFixed(1)},${targetY.toFixed(1)}`,
      );
    }

    // Floating cell popover & zoom-adaptive cell readouts (supported on any cutaway)
    const popoverRect = this.renderCellPopover(
      input.cellPopover,
      snapshot,
      occupied,
      minY,
      maxY,
      input.activeCellGrids,
    );

    // KV depth ticks (1, 8, 16, T=N) in GQA Cutaway
    if (scene.kvDepthTicks.length > 0) {
      const orderedTicks = [...scene.kvDepthTicks].sort((a, b) =>
        a.token === tokens ? -1 : b.token === tokens ? 1 : a.token - b.token,
      );
      for (const tick of orderedTicks) {
        const pt = projectWorldToHostCss(tick.world, snapshot);
        if (!pt.visible || pt.x < 10 || pt.x > hostWidth - 28 || pt.y < minY || pt.y > maxY) {
          continue;
        }
        const isCur = tick.token === tokens;
        const text = isCur ? `T=${tick.token}` : `${tick.token}`;
        const w = this.measureText(text, monoFont) + 8;
        const h = 15;
        const lx = clamp(pt.x - w - 4, 8, hostWidth - w - 8);
        const ly = clamp(pt.y - h * 0.5, minY, maxY - h);
        if (!isCur && overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
          continue;
        }
        const el = this.kvTickNodes.get(tick.token);
        if (!el) continue;
        el.textContent = text;
        el.classList.toggle('is-current', isCur);
        el.style.display = 'inline-flex';
        el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
        occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
      }
    }

    this.renderActiveCellGrids(input.activeCellGrids, snapshot, occupied, popoverRect, minY, maxY);

    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.phaseBracketPath.setAttribute('d', '');
  }

  /** Exterior, zero-occlusion label, elbow-leader, caliper, and V3 callout pass for thin-sheet cutaways. */
  private updateSheetCutawayOverlay(
    input: CutawayOverlayUpdateInput,
    activePart: string | null,
    occupied: PlacedRect[],
    leaderSegs: string[],
    minY: number,
    maxY: number,
  ): void {
    const { scene, snapshot, tokens, activeOpPartIds } = input;
    const { hostWidth } = snapshot.metrics;
    const sheetFont = '600 10.5px system-ui, -apple-system, sans-serif';
    const monoFont = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    const caliperSegs: string[] = [];

    // Estimate zoom level from screen-space width of a 1-unit horizontal span at origin
    const pOrigin = projectWorldToHostCss([0, 0.05, 0], snapshot);
    const pUnit = projectWorldToHostCss([1, 0.05, 0], snapshot);
    const pxPerWorldUnit =
      pOrigin.visible && pUnit.visible ? Math.hypot(pUnit.x - pOrigin.x, pUnit.y - pOrigin.y) : 50;
    const isCloseZoom = pxPerWorldUnit >= 82;

    const isHeadSelected = activePart !== null && activePart.startsWith('q-head-');
    const hasActiveOpFocus = activeOpPartIds != null && activeOpPartIds.size > 0;

    const isLandmarkParticipating = (lm: LandmarkAnchor): boolean => {
      if (!hasActiveOpFocus) return true;
      if (lm.partId !== undefined && lm.partId === activePart) return true;
      if (lm.partId !== undefined && activeOpPartIds!.has(lm.partId)) return true;
      if (activeOpPartIds!.has(lm.id)) return true;
      return false;
    };

    const ordered = [...scene.landmarks].sort((a, b) => {
      const aAct = a.partId !== undefined && a.partId === activePart ? 4 : 0;
      const bAct = b.partId !== undefined && b.partId === activePart ? 4 : 0;
      const aPart = hasActiveOpFocus && isLandmarkParticipating(a) ? 3 : 0;
      const bPart = hasActiveOpFocus && isLandmarkParticipating(b) ? 3 : 0;
      const aTier = a.tier === 'detail' ? 0 : 1;
      const bTier = b.tier === 'detail' ? 0 : 1;
      return Math.max(bAct, bPart, bTier) - Math.max(aAct, aPart, aTier);
    });

    for (const lm of ordered) {
      const isAct = lm.partId !== undefined && lm.partId === activePart;
      const isParticipating = isLandmarkParticipating(lm);
      if (lm.tier === 'detail' && !isAct && !isCloseZoom && !(hasActiveOpFocus && isParticipating)) {
        continue;
      }

      const pt = projectWorldToHostCss(lm.world, snapshot);
      if (!pt.visible || pt.x < -36 || pt.x > hostWidth + 36 || pt.y < minY - 28 || pt.y > maxY + 28) {
        continue;
      }

      const isDetail = lm.tier === 'detail';
      const w = this.measureText(lm.label, isDetail ? monoFont : sheetFont) + (isDetail ? 10 : 12);
      const h = isDetail ? 17 : 18;
      const gap = 4;

      // Strictly respect exterior alignment hint so labels stay OUTSIDE matrix faces
      let idealLx = pt.x - w * 0.5;
      let idealLy = pt.y - h * 0.5;
      if (lm.align === 'left') {
        idealLx = pt.x - w - gap;
        idealLy = pt.y - h * 0.5;
      } else if (lm.align === 'right') {
        idealLx = pt.x + gap;
        idealLy = pt.y - h * 0.5;
      } else if (lm.align === 'top') {
        idealLx = pt.x - w * 0.5;
        idealLy = pt.y - h - gap;
      } else if (lm.align === 'bottom') {
        idealLx = pt.x - w * 0.5;
        idealLy = pt.y + gap;
      }

      const padX = hostWidth < 760 ? 6 : 8;
      const maxX = Math.max(padX, hostWidth - w - padX);
      const maxTopY = Math.max(minY, maxY - h);
      let lx = clamp(idealLx, padX, maxX);
      let ly = clamp(idealLy, minY, maxTopY);

      if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
        const nudges: readonly [number, number][] = [
          [lx, clamp(ly - (h + 3), minY, maxTopY)],
          [lx, clamp(ly + (h + 3), minY, maxTopY)],
          [clamp(lx - 14, padX, maxX), ly],
          [clamp(lx + 14, padX, maxX), ly],
          [lx, clamp(ly - (h * 2 + 6), minY, maxTopY)],
          [lx, clamp(ly + (h * 2 + 6), minY, maxTopY)],
        ];
        let placed = false;
        for (const [cx, cy] of nudges) {
          if (!overlapsAny(occupied, cx, cy, cx + w, cy + h)) {
            lx = cx;
            ly = cy;
            placed = true;
            break;
          }
        }
        if (!placed && !isAct && !(hasActiveOpFocus && isParticipating)) continue;
      }

      const el = this.getOrCreateCutawayNode(lm.id);
      el.className = isDetail ? 'ov-sheet-label ov-sheet-detail' : 'ov-sheet-label';
      if (isAct) el.classList.add('is-active');
      else if (lm.tone === 'gold') el.classList.add('is-gold');
      else if (lm.tone === 'cyan') el.classList.add('is-cyan');
      else if (lm.tone === 'muted') el.classList.add('is-muted');

      // V1 Active-op label dimming: dim secondary/non-participating labels to 0.46
      el.classList.toggle('is-dimmed', !isParticipating);
      el.style.opacity = isParticipating ? '1' : DIMMED_LABEL_OPACITY;

      el.textContent = lm.label;
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;

      const placedRect: PlacedRect = { x0: lx, y0: ly, x1: lx + w, y1: ly + h };
      occupied.push(placedRect);

      // V1 Label Hygiene: draw a subtle SVG elbow leader whenever a label is nudged > 10 px
      // away from its projected world anchor so nudged labels never float ambiguously.
      const nearX = clamp(pt.x, placedRect.x0, placedRect.x1);
      const nearY = clamp(pt.y, placedRect.y0, placedRect.y1);
      const edgeDist = Math.hypot(nearX - pt.x, nearY - pt.y);
      const nudgeDist = Math.hypot(lx - idealLx, ly - idealLy);
      if (
        edgeDist > LABEL_NUDGE_LEADER_THRESHOLD_PX ||
        (nudgeDist > LABEL_NUDGE_LEADER_THRESHOLD_PX && edgeDist > 3)
      ) {
        const anchorX = clamp(pt.x, 4, hostWidth - 4);
        const anchorY = clamp(pt.y, minY, maxY);
        const seg = buildElbowLeaderPath(anchorX, anchorY, placedRect);
        if (seg) leaderSegs.push(seg);
      }
    }

    // V3: Floating contextual formula popover + curved provenance pointers
    const popoverRect = this.renderCellPopover(
      input.cellPopover,
      snapshot,
      occupied,
      minY,
      maxY,
      input.activeCellGrids,
    );

    // Subtle KV depth ticks along right side of KV1
    if (scene.kvDepthTicks.length > 0 && pxPerWorldUnit >= 36) {
      const orderedTicks = [...scene.kvDepthTicks].sort((a, b) =>
        a.token === tokens ? -1 : b.token === tokens ? 1 : a.token - b.token,
      );
      for (const tick of orderedTicks) {
        const pt = projectWorldToHostCss(tick.world, snapshot);
        if (!pt.visible || pt.x < 8 || pt.x > hostWidth - 28 || pt.y < minY || pt.y > maxY) {
          continue;
        }
        const isCur = tick.token === tokens;
        const text = isCur ? `T=${tick.token}` : `${tick.token}`;
        const w = this.measureText(text, monoFont) + 8;
        const h = 15;
        const lx = clamp(pt.x + 2, 8, hostWidth - w - 8);
        const ly = clamp(pt.y - h * 0.5, minY, maxY - h);
        if (!isCur && overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
          continue;
        }
        const el = this.kvTickNodes.get(tick.token);
        if (!el) continue;
        el.textContent = text;
        el.classList.toggle('is-current', isCur);
        el.style.display = 'inline-flex';
        el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
        occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
      }
    }

    // Subtle SVG dimension calipers (revealed on related hover/selection or close zoom)
    if (scene.calipers && scene.calipers.length > 0) {
      for (const cal of scene.calipers) {
        const relatedActive =
          activePart !== null &&
          ((cal.partIds && cal.partIds.includes(activePart)) ||
            (isHeadSelected && cal.partIds?.includes('q-proj')));
        if (!relatedActive && !isCloseZoom) continue;

        const pa = projectWorldToHostCss(cal.aWorld, snapshot);
        const pb = projectWorldToHostCss(cal.bWorld, snapshot);
        if (!pa.visible || !pb.visible) continue;

        const spanPx = Math.hypot(pb.x - pa.x, pb.y - pa.y);
        if (spanPx < (cal.minPixels ?? 64)) continue;

        const off = cal.offsetPx ?? 8;
        const ay = clamp(pa.y + off, minY, maxY);
        const by = clamp(pb.y + off, minY, maxY);
        const ax = clamp(pa.x, 8, hostWidth - 8);
        const bx = clamp(pb.x, 8, hostWidth - 8);

        // Caliper bracket line with crisp 4px end ticks
        caliperSegs.push(
          `M${ax.toFixed(1)},${(ay - 3).toFixed(1)}L${ax.toFixed(1)},${(ay + 3).toFixed(1)}` +
            `M${ax.toFixed(1)},${ay.toFixed(1)}L${bx.toFixed(1)},${by.toFixed(1)}` +
            `M${bx.toFixed(1)},${(by - 3).toFixed(1)}L${bx.toFixed(1)},${(by + 3).toFixed(1)}`,
        );

        const w = this.measureText(cal.label, monoFont) + 8;
        const h = 15;
        const mx = clamp((ax + bx) * 0.5 - w * 0.5, 8, hostWidth - w - 8);
        const my = clamp((ay + by) * 0.5 + 4, minY, maxY - h);
        if (!overlapsAny(occupied, mx, my, mx + w, my + h)) {
          const el = this.getOrCreateCaliperNode(cal.id);
          el.textContent = cal.label;
          el.style.display = 'inline-flex';
          el.style.transform = `translate3d(${mx.toFixed(1)}px, ${my.toFixed(1)}px, 0)`;
          occupied.push({ x0: mx, y0: my, x1: mx + w, y1: my + h });
        }
      }
    }

    // V3: Zoom-adaptive on-cell numeric readouts (<= 64 nodes) & exterior axis headers (<= 24 nodes)
    this.renderActiveCellGrids(input.activeCellGrids, snapshot, occupied, popoverRect, minY, maxY);

    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.caliperPath.setAttribute('d', caliperSegs.join(''));
    this.phaseBracketPath.setAttribute('d', '');
  }

  /**
   * Render the single pre-allocated `.ov-cell-popover` contextual formula card and its curved
   * quadratic Bezier pointer (+ optional indigo/teal provenance curves to `sourceWorlds`).
   */
  private renderCellPopover(
    spec: CellPopoverSpec | null | undefined,
    snapshot: CameraSnapshot,
    occupied: PlacedRect[],
    minY: number,
    maxY: number,
    activeCellGrids?: readonly ActiveSheetCellGrid[] | null,
  ): PlacedRect | null {
    if (!spec) return null;
    const { hostWidth, hostHeight } = snapshot.metrics;
    const pt = projectWorldToHostCss(spec.world, snapshot);
    if (!pt.visible || pt.x < -30 || pt.x > hostWidth + 30 || pt.y < minY - 30 || pt.y > maxY + 30) {
      return null;
    }

    const headFont = '700 10px system-ui, -apple-system, sans-serif';
    const coordFont = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const badgeFont = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    const eqFont = '500 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    const valFont = '700 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';

    this.popoverTensorEl.textContent = spec.tensorLabel;
    this.popoverCoordEl.textContent = spec.coordLabel;
    if (spec.badge) {
      this.popoverBadgeEl.textContent = spec.badge;
      this.popoverBadgeEl.style.display = 'inline-flex';
    } else {
      this.popoverBadgeEl.textContent = '';
      this.popoverBadgeEl.style.display = 'none';
    }

    const operandColors = spec.operandColors ?? [];
    const dotCount = Math.min(operandColors.length, this.popoverDotEls.length);
    if (dotCount > 0) {
      this.popoverDotsWrap.style.display = 'inline-flex';
      for (let i = 0; i < this.popoverDotEls.length; i++) {
        const dotEl = this.popoverDotEls[i];
        if (i < dotCount) {
          dotEl.style.display = 'inline-block';
          dotEl.style.backgroundColor = OPERAND_SWATCH_HEX[operandColors[i]];
        } else {
          dotEl.style.display = 'none';
        }
      }
    } else {
      this.popoverDotsWrap.style.display = 'none';
      for (const dotEl of this.popoverDotEls) dotEl.style.display = 'none';
    }

    this.popoverEqEl.textContent = spec.equation;
    this.popoverValEl.textContent = spec.valueText;
    const isAccumulating = Boolean(spec.isAccumulating);
    const isMaskedVal = spec.valueText.includes('∞') || spec.valueText.toLowerCase().includes('mask');
    this.cellPopover.classList.toggle('is-accumulating', isAccumulating);
    this.popoverValEl.classList.toggle('is-accumulating', isAccumulating);
    this.popoverValEl.classList.toggle('is-masked', isMaskedVal);

    const headW =
      this.measureText(spec.tensorLabel, headFont) +
      this.measureText(spec.coordLabel, coordFont) +
      (spec.badge ? this.measureText(spec.badge, badgeFont) + 14 : 0) +
      34;
    const dotsW = dotCount > 0 ? dotCount * 9 + 6 : 0;
    const bodyW =
      dotsW +
      this.measureText(spec.equation, eqFont) +
      this.measureText(spec.valueText, valFont) +
      48;

    const maxCardW = Math.max(120, Math.min(396, hostWidth - 16));
    const cardW = clamp(Math.max(headW, bodyW), Math.min(148, maxCardW), maxCardW);
    const cardH = bodyW > maxCardW ? 66 : 48;

    const anchorX = clamp(pt.x, 6, hostWidth - 6);
    const anchorY = clamp(pt.y, minY, maxY);

    // Project upstream source points BEFORE placement so popover avoids upstream operands & provenance threads
    const projectedSources: {
      readonly sx: number;
      readonly sy: number;
      readonly role: 'indigo' | 'teal';
    }[] = [];
    const avoidRects: OverlayRect[] = [];
    let targetSheetRect: OverlayRect | null = null;
    let bestTargetNormalDist = Infinity;
    let fallbackTargetArea = Infinity;

    if (activeCellGrids && activeCellGrids.length > 0) {
      const sheetCorners: readonly (readonly [number, number])[] = [
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
      ];
      for (const grid of activeCellGrids) {
        if (grid.cols <= 0 || grid.rows <= 0) continue;
        const frame = makeSheetFrame3D(grid.originWorld, grid.extent, grid.axisU, grid.axisN);
        const faceLy = dot3(sub3(snapshot.eye, frame.center), frame.axisN) >= 0 ? 0.5 : -0.5;
        let sx0 = Infinity;
        let sy0 = Infinity;
        let sx1 = -Infinity;
        let sy1 = -Infinity;
        let visibleCorners = 0;
        for (const [lx, lz] of sheetCorners) {
          const cPt = projectWorldToHostCss(localToWorldOnFrame(frame, lx, faceLy, lz), snapshot);
          if (!cPt.visible) continue;
          visibleCorners++;
          if (cPt.x < sx0) sx0 = cPt.x;
          if (cPt.y < sy0) sy0 = cPt.y;
          if (cPt.x > sx1) sx1 = cPt.x;
          if (cPt.y > sy1) sy1 = cPt.y;
        }
        if (visibleCorners < 2) continue;
        if (sx1 <= 0 || sx0 >= hostWidth || sy1 <= 0 || sy0 >= hostHeight) continue;
        const sheetRect: OverlayRect = { x0: sx0, y0: sy0, x1: sx1, y1: sy1 };
        avoidRects.push(sheetRect);

        // Identify the active target sheet via 3D local-frame containment (with 2D AABB fallback)
        const dVec = sub3(spec.world, frame.center);
        const uNorm = Math.abs(dot3(dVec, frame.axisU)) / Math.max(1e-4, frame.extent[0]);
        const vNorm = Math.abs(dot3(dVec, frame.axisV)) / Math.max(1e-4, frame.extent[2]);
        const nDist = Math.abs(dot3(dVec, frame.axisN));
        if (uNorm <= 0.56 && vNorm <= 0.56 && nDist <= frame.extent[1] * 0.5 + 0.18) {
          if (nDist < bestTargetNormalDist) {
            bestTargetNormalDist = nDist;
            targetSheetRect = sheetRect;
          }
        } else if (
          bestTargetNormalDist === Infinity &&
          anchorX >= sheetRect.x0 - 4 &&
          anchorX <= sheetRect.x1 + 4 &&
          anchorY >= sheetRect.y0 - 4 &&
          anchorY <= sheetRect.y1 + 4
        ) {
          const area = (sheetRect.x1 - sheetRect.x0) * (sheetRect.y1 - sheetRect.y0);
          if (area < fallbackTargetArea) {
            fallbackTargetArea = area;
            targetSheetRect = sheetRect;
          }
        }
      }
    }

    if (spec.sourceWorlds && spec.sourceWorlds.length > 0) {
      const sourcePad = 28;
      for (const src of spec.sourceWorlds) {
        const spt = projectWorldToHostCss(src.world, snapshot);
        if (
          !spt.visible ||
          spt.x < -40 ||
          spt.x > hostWidth + 40 ||
          spt.y < minY - 40 ||
          spt.y > maxY + 40
        ) {
          continue;
        }
        const sx = clamp(spt.x, 4, hostWidth - 4);
        const sy = clamp(spt.y, minY, maxY);
        projectedSources.push({ sx, sy, role: src.role });
        avoidRects.push({
          x0: sx - sourcePad,
          y0: sy - sourcePad,
          x1: sx + sourcePad,
          y1: sy + sourcePad,
        });
        const mx = (sx + anchorX) * 0.5;
        const my = (sy + anchorY) * 0.5;
        avoidRects.push({
          x0: mx - sourcePad,
          y0: my - sourcePad,
          x1: mx + sourcePad,
          y1: my + sourcePad,
        });
      }
    }

    const placed = pickPopoverPlacement({
      anchorX,
      anchorY,
      width: cardW,
      height: cardH,
      hostWidth,
      minY,
      maxY,
      occupied,
      avoidRects,
      targetSheetRect,
      marginX: 8,
    });

    this.cellPopover.style.width = `${placed.width.toFixed(0)}px`;
    this.cellPopover.style.display = 'flex';
    this.cellPopover.style.pointerEvents = 'none';
    this.cellPopover.style.transform = `translate3d(${placed.x.toFixed(1)}px, ${placed.y.toFixed(1)}px, 0)`;

    const placedRect: PlacedRect = {
      x0: placed.x,
      y0: placed.y,
      x1: placed.x + placed.width,
      y1: placed.y + placed.height,
    };
    occupied.push(placedRect);

    // Primary curved quadratic Bezier pointer from popover edge straight into active 3D cell center
    const primaryCurve = buildQuadraticPointerPath(
      placed.edgeX,
      placed.edgeY,
      anchorX,
      anchorY,
      placed.exitOrientation,
    );
    this.popoverLeaderPath.setAttribute('d', primaryCurve);
    this.popoverAnchorDot.setAttribute('cx', anchorX.toFixed(1));
    this.popoverAnchorDot.setAttribute('cy', anchorY.toFixed(1));
    this.popoverAnchorDot.style.display = 'inline';

    // Optional subtle indigo / teal curved provenance pointers to upstream source 3D cells
    if (projectedSources.length > 0) {
      const indigoSegs: string[] = [];
      const tealSegs: string[] = [];
      for (const src of projectedSources) {
        const border = pickRectBorderPoint(placedRect, src.sx, src.sy);
        const curve = buildQuadraticPointerPath(
          border.edgeX,
          border.edgeY,
          src.sx,
          src.sy,
          border.orientation,
        );
        if (!curve) continue;
        if (src.role === 'indigo') indigoSegs.push(curve);
        else tealSegs.push(curve);
      }
      this.popoverSourceIndigoPath.setAttribute('d', indigoSegs.join(''));
      this.popoverSourceTealPath.setAttribute('d', tealSegs.join(''));
    }

    return placedRect;
  }

  /**
   * Render zoom-adaptive on-cell numeric readouts (strictly when `cellW >= 26 && cellH >= 16`,
   * hard-capped at `MAX_OVERLAY_CELL_NUM_NODES = 64`) and exterior row/column axis headers
   * (when `cellW >= 22 || cellH >= 15`, hard-capped at `MAX_OVERLAY_AXIS_HEADER_NODES = 24`).
   */
  private renderActiveCellGrids(
    grids: readonly ActiveSheetCellGrid[] | null | undefined,
    snapshot: CameraSnapshot,
    occupied: PlacedRect[],
    popoverRect: PlacedRect | null,
    minY: number,
    maxY: number,
  ): void {
    if (!grids || grids.length === 0) return;

    const { hostWidth } = snapshot.metrics;
    const cellNumFont = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    const headerFont = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';

    interface PreparedGrid {
      readonly grid: ActiveSheetCellGrid;
      readonly frame: ReturnType<typeof makeSheetFrame3D>;
      readonly faceLy: number;
      readonly size: ProjectedCellSize;
      readonly showNums: boolean;
      readonly showHeaders: boolean;
    }

    const prepared: PreparedGrid[] = [];
    for (const grid of grids) {
      if (grid.cols <= 0 || grid.rows <= 0) continue;
      const size = computeProjectedCellSize(grid, snapshot);
      if (!size.visible) continue;
      const showNums = shouldShowCellNumbers(size.cellW, size.cellH);
      const showHeaders = shouldShowAxisHeaders(size.cellW, size.cellH);
      if (!showNums && !showHeaders) continue;
      const frame = makeSheetFrame3D(grid.originWorld, grid.extent, grid.axisU, grid.axisN);
      const faceLy = dot3(sub3(snapshot.eye, frame.center), frame.axisN) >= 0 ? 0.5 : -0.5;
      prepared.push({ grid, frame, faceLy, size, showNums, showHeaders });
    }

    if (prepared.length === 0) return;

    // --- 1. On-Cell Numeric Readouts (strict 64-node hard cap across all grids) ---
    let cellNumCursor = 0;
    const cellNumRects: PlacedRect[] = [];

    const placeCellNum = (pg: PreparedGrid, col: number, row: number): boolean => {
      if (cellNumCursor >= MAX_OVERLAY_CELL_NUM_NODES) return false;
      const { grid, frame, faceLy } = pg;
      if (isCellHiddenByReveal(col, row, grid)) return false;
      const idx = row * grid.cols + col;
      if (idx < 0 || idx >= grid.values.length) return false;

      const worldPt = localToWorldOnFrame(
        frame,
        (col + 0.5) / grid.cols - 0.5,
        faceLy,
        (row + 0.5) / grid.rows - 0.5,
      );
      const pt = projectWorldToHostCss(worldPt, snapshot);
      if (
        !pt.visible ||
        pt.x < 12 ||
        pt.x > hostWidth - 12 ||
        pt.y < minY + 6 ||
        pt.y > maxY - 6
      ) {
        return false;
      }

      // Skip cells hidden underneath the floating popover card
      if (
        popoverRect &&
        pt.x >= popoverRect.x0 - 4 &&
        pt.x <= popoverRect.x1 + 4 &&
        pt.y >= popoverRect.y0 - 4 &&
        pt.y <= popoverRect.y1 + 4
      ) {
        return false;
      }

      const isActive = col === grid.activeCol && row === grid.activeRow;
      const isMasked = Boolean(grid.masked && idx < grid.masked.length && grid.masked[idx] !== 0);
      const hasTextOverride =
        isActive &&
        !isMasked &&
        grid.activeValueTextOverride !== undefined &&
        grid.activeValueTextOverride !== null;
      const hasNumericOverride =
        isActive &&
        !isMasked &&
        grid.activeValueOverride !== undefined &&
        grid.activeValueOverride !== null &&
        Number.isFinite(grid.activeValueOverride);
      const isAccumulating = hasTextOverride || hasNumericOverride;
      const rawValue = hasNumericOverride ? grid.activeValueOverride! : grid.values[idx];
      const text = hasTextOverride
        ? grid.activeValueTextOverride!
        : formatCellOverlayValue(rawValue, grid.format ?? 'signed', isMasked);
      const w = this.measureText(text, cellNumFont) + 6;
      const h = 13;
      const lx = pt.x - w * 0.5;
      const ly = pt.y - h * 0.5;

      const el = this.cellNumNodes[cellNumCursor++];
      el.textContent = text;
      el.classList.toggle('is-active', isActive);
      el.classList.toggle('is-accumulating', isAccumulating);
      el.classList.toggle('is-masked', isMasked);
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      cellNumRects.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
      return true;
    };

    // Pass A: active cell + active row/col of every eligible grid first so multi-sheet focus never starves
    const remainingPerGrid: { pg: PreparedGrid; coords: readonly (readonly [number, number])[] }[] = [];
    for (const pg of prepared) {
      if (!pg.showNums) continue;
      const ordered = prioritizeSheetCellCoords(
        pg.grid.cols,
        pg.grid.rows,
        pg.grid.activeCol,
        pg.grid.activeRow,
      );
      const hasActiveCol =
        pg.grid.activeCol !== undefined &&
        pg.grid.activeCol >= 0 &&
        pg.grid.activeCol < pg.grid.cols;
      const hasActiveRow =
        pg.grid.activeRow !== undefined &&
        pg.grid.activeRow >= 0 &&
        pg.grid.activeRow < pg.grid.rows;

      const rest: (readonly [number, number])[] = [];
      for (const coord of ordered) {
        const [c, r] = coord;
        const isPriority =
          (hasActiveCol && c === pg.grid.activeCol) || (hasActiveRow && r === pg.grid.activeRow);
        if (isPriority) {
          if (cellNumCursor < MAX_OVERLAY_CELL_NUM_NODES) {
            placeCellNum(pg, c, r);
          }
        } else {
          rest.push(coord);
        }
      }
      remainingPerGrid.push({ pg, coords: rest });
    }

    // Pass B: fill remaining slots up to MAX_OVERLAY_CELL_NUM_NODES (64)
    for (const item of remainingPerGrid) {
      if (cellNumCursor >= MAX_OVERLAY_CELL_NUM_NODES) break;
      for (const [c, r] of item.coords) {
        if (cellNumCursor >= MAX_OVERLAY_CELL_NUM_NODES) break;
        placeCellNum(item.pg, c, r);
      }
    }

    // --- 2. Exterior Row / Column Axis Headers & Dimension Badges (<= 24 nodes, collision-checked) ---
    let headerCursor = 0;

    const placeExteriorHeader = (
      text: string,
      edgeWorld: Vec3,
      insideWorld: Vec3,
      isActive: boolean,
      isDim: boolean,
    ): void => {
      if (headerCursor >= MAX_OVERLAY_AXIS_HEADER_NODES || !text) return;
      const edgePt = projectWorldToHostCss(edgeWorld, snapshot);
      const insidePt = projectWorldToHostCss(insideWorld, snapshot);
      if (
        !edgePt.visible ||
        edgePt.x < -12 ||
        edgePt.x > hostWidth + 12 ||
        edgePt.y < minY - 12 ||
        edgePt.y > maxY + 12
      ) {
        return;
      }

      const ox = insidePt.visible ? edgePt.x - insidePt.x : 0;
      const oy = insidePt.visible ? edgePt.y - insidePt.y : -1;
      const olen = Math.hypot(ox, oy);
      const nx = olen > 1e-3 ? ox / olen : 0;
      const ny = olen > 1e-3 ? oy / olen : -1;

      const w = this.measureText(text, headerFont) + 8;
      const h = 15;
      const gap = 6;
      const ax = edgePt.x + nx * gap;
      const ay = edgePt.y + ny * gap;

      let lx: number;
      let ly: number;
      if (Math.abs(nx) >= Math.abs(ny)) {
        lx = nx < 0 ? ax - w : ax;
        ly = ay - h * 0.5;
      } else {
        lx = ax - w * 0.5;
        ly = ny < 0 ? ay - h : ay;
      }

      lx = clamp(lx, 8, Math.max(8, hostWidth - w - 8));
      ly = clamp(ly, minY, Math.max(minY, maxY - h));

      if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) return;
      if (overlapsAny(cellNumRects, lx, ly, lx + w, ly + h)) return;

      const el = this.axisHeaderNodes[headerCursor++];
      el.textContent = text;
      el.classList.toggle('is-active', isActive);
      el.classList.toggle('is-dim', isDim);
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
    };

    for (const pg of prepared) {
      if (!pg.showHeaders || headerCursor >= MAX_OVERLAY_AXIS_HEADER_NODES) continue;
      const { grid, frame, faceLy } = pg;
      const safeCols = Math.max(1, grid.cols);
      const safeRows = Math.max(1, grid.rows);
      const showColLabels = !grid.suppressColHeaders && Boolean(grid.colLabels && grid.colLabels.length > 0);
      const showRowLabels = !grid.suppressRowHeaders && Boolean(grid.rowLabels && grid.rowLabels.length > 0);

      // Prioritize active row & active column headers first
      if (
        showRowLabels &&
        grid.rowLabels &&
        grid.activeRow !== undefined &&
        grid.activeRow >= 0 &&
        grid.activeRow < grid.rowLabels.length
      ) {
        const r = grid.activeRow;
        placeExteriorHeader(
          grid.rowLabels[r],
          localToWorldOnFrame(frame, -0.5, faceLy, (r + 0.5) / safeRows - 0.5),
          localToWorldOnFrame(frame, 0.5 / safeCols - 0.5, faceLy, (r + 0.5) / safeRows - 0.5),
          true,
          false,
        );
      }

      if (
        showColLabels &&
        grid.colLabels &&
        grid.activeCol !== undefined &&
        grid.activeCol >= 0 &&
        grid.activeCol < grid.colLabels.length
      ) {
        const c = grid.activeCol;
        placeExteriorHeader(
          grid.colLabels[c],
          localToWorldOnFrame(frame, (c + 0.5) / safeCols - 0.5, faceLy, -0.5),
          localToWorldOnFrame(frame, (c + 0.5) / safeCols - 0.5, faceLy, 0.5 / safeRows - 0.5),
          true,
          false,
        );
      }

      // Dimension badges (e.g. "K = 6", "Q = 1") on opposite clean exterior edges
      if (grid.colDimLabel) {
        const edgeLz = grid.colLabels && grid.colLabels.length > 0 ? 0.5 : -0.5;
        placeExteriorHeader(
          grid.colDimLabel,
          localToWorldOnFrame(frame, 0, faceLy, edgeLz),
          localToWorldOnFrame(frame, 0, faceLy, edgeLz * 0.5),
          false,
          true,
        );
      }

      if (grid.rowDimLabel) {
        const edgeLx = grid.rowLabels && grid.rowLabels.length > 0 ? 0.5 : -0.5;
        placeExteriorHeader(
          grid.rowDimLabel,
          localToWorldOnFrame(frame, edgeLx, faceLy, 0),
          localToWorldOnFrame(frame, edgeLx * 0.5, faceLy, 0),
          false,
          true,
        );
      }

      // Remaining row labels (left edge)
      if (showRowLabels && grid.rowLabels) {
        const count = Math.min(grid.rows, grid.rowLabels.length);
        for (let r = 0; r < count && headerCursor < MAX_OVERLAY_AXIS_HEADER_NODES; r++) {
          if (r === grid.activeRow) continue;
          placeExteriorHeader(
            grid.rowLabels[r],
            localToWorldOnFrame(frame, -0.5, faceLy, (r + 0.5) / safeRows - 0.5),
            localToWorldOnFrame(frame, 0.5 / safeCols - 0.5, faceLy, (r + 0.5) / safeRows - 0.5),
            false,
            false,
          );
        }
      }

      // Remaining column labels (top edge)
      if (showColLabels && grid.colLabels) {
        const count = Math.min(grid.cols, grid.colLabels.length);
        for (let c = 0; c < count && headerCursor < MAX_OVERLAY_AXIS_HEADER_NODES; c++) {
          if (c === grid.activeCol) continue;
          placeExteriorHeader(
            grid.colLabels[c],
            localToWorldOnFrame(frame, (c + 0.5) / safeCols - 0.5, faceLy, -0.5),
            localToWorldOnFrame(frame, (c + 0.5) / safeCols - 0.5, faceLy, 0.5 / safeRows - 0.5),
            false,
            false,
          );
        }
      }
    }
  }
}
