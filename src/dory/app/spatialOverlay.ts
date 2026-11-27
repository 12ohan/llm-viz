/**
 * Pooled DOM + SVG 3D-to-2D Drafting-Table Overlay for NVIDIA Nemotron-Labs-Dory1.0-4B
 * (`src/dory/app/spatialOverlay.ts`).
 *
 * Implements `DorySpatialOverlayModuleContract` (`DORY6` Slice 6A):
 *   - Directly reuses verified pure layout, cell-sizing, reveal-frontier, popover collision,
 *     and elbow/quadratic leader helpers from `src/shared/overlayLayout.ts` and projection math
 *     from `src/render/camera.ts` + `src/render/sheetFrame.ts`.
 *   - Pre-allocates ALL DOM and SVG nodes once inside `createDorySpatialOverlay()` — executes
 *     strictly ZERO `document.createElement` or `document.createElementNS` calls during
 *     `update()`, `clear()`, `getStats()`, or `dispose()`.
 *   - Hard DOM pool ceiling: `<= 64` on-cell numeric readout nodes (`DORY_MAX_OVERLAY_CELL_NUM_NODES`)
 *     and `<= 24` exterior axis header nodes (`DORY_MAX_OVERLAY_AXIS_HEADER_NODES`).
 *   - Strict `D4` architectural boundary isolation: zero imports from Gate 4b/4c Nano modules
 *     or sibling visualizer apps; imports strictly from `src/render/*` and `src/shared/overlayLayout.ts`.
 *   - 100% clean-room original TypeScript — zero runtime npm deps, zero ambient RNG.
 */

import {
  dot3,
  projectWorldToHostCss,
  sub3,
  type CameraSnapshot,
} from '../../render/camera';
import type { LandmarkAnchor, Vec3 } from '../../render/primitives';
import { localToWorldOnFrame, makeSheetFrame3D } from '../../render/sheetFrame';
import {
  DIMMED_LABEL_OPACITY,
  LABEL_NUDGE_LEADER_THRESHOLD_PX,
  MAX_OVERLAY_AXIS_HEADER_NODES,
  MAX_OVERLAY_CELL_NUM_NODES,
  MAX_OVERLAY_WIDTH_CACHE_ENTRIES,
  buildElbowLeaderPath,
  buildQuadraticPointerPath,
  computeProjectedCellSize,
  formatCellOverlayValue,
  isCellHiddenByReveal,
  pickPopoverPlacement,
  pickRectBorderPoint,
  prioritizeSheetCellCoords,
  shouldShowAxisHeaders,
  shouldShowCellNumbers,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
  type OverlayRect,
  type ProjectedCellSize,
} from '../../shared/overlayLayout';
import type {
  DoryBadgeTone,
  DoryOverlayAnchor,
  DoryPickIdentity,
  DorySpatialScene,
} from '../scene/types';
import {
  DORY_MAX_OVERLAY_AXIS_HEADER_NODES,
  DORY_MAX_OVERLAY_CELL_NUM_NODES,
  type DoryOverlayRect,
  type DorySpatialOverlayHandle,
  type DorySpatialOverlayModuleContract,
  type DorySpatialOverlayOptions,
  type DorySpatialOverlayStats,
  type DorySpatialOverlayUpdateInput,
} from './types';

export { DORY_MAX_OVERLAY_AXIS_HEADER_NODES, DORY_MAX_OVERLAY_CELL_NUM_NODES };

const SVG_NS = 'http://www.w3.org/2000/svg';
const ANCHOR_PILL_POOL_CAPACITY = 72;
const CALIPER_LABEL_POOL_CAPACITY = 24;
const KV_TICK_POOL_CAPACITY = 36;
const MAX_POPOVER_OPERAND_DOTS = 4;

const DEFAULT_POPOVER_WIDTH_PX = 296;
const DEFAULT_POPOVER_HEIGHT_PX = 124;

const OPERAND_SWATCH_HEX: Readonly<Record<'indigo' | 'teal' | 'red', string>> = Object.freeze({
  indigo: '#3861BD',
  teal: '#1A8A7A',
  red: '#DC4C34',
});

interface MutablePlacedRect {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

// ============================================================================
// PURE BUDGET & COLLISION HELPERS (100% HEADLESS NODE-TESTABLE)
// ============================================================================

/**
 * Clamp requested on-cell numeric overlay count strictly into
 * `[0, Math.min(64, maxNodes ?? 64)]` (`DORY_MAX_OVERLAY_CELL_NUM_NODES`).
 */
export function computeDoryOverlayCellNumBudget(
  requestedCount: number,
  maxNodes?: number,
): number {
  const hardMax = Math.min(DORY_MAX_OVERLAY_CELL_NUM_NODES, MAX_OVERLAY_CELL_NUM_NODES);
  const cap =
    maxNodes === undefined || !Number.isFinite(maxNodes)
      ? hardMax
      : Math.max(0, Math.min(hardMax, Math.trunc(maxNodes)));
  if (!Number.isFinite(requestedCount)) return 0;
  return Math.max(0, Math.min(cap, Math.trunc(requestedCount)));
}

/**
 * Clamp requested exterior axis header count strictly into
 * `[0, Math.min(24, maxNodes ?? 24)]` (`DORY_MAX_OVERLAY_AXIS_HEADER_NODES`).
 */
export function computeDoryOverlayAxisHeaderBudget(
  requestedCount: number,
  maxNodes?: number,
): number {
  const hardMax = Math.min(DORY_MAX_OVERLAY_AXIS_HEADER_NODES, MAX_OVERLAY_AXIS_HEADER_NODES);
  const cap =
    maxNodes === undefined || !Number.isFinite(maxNodes)
      ? hardMax
      : Math.max(0, Math.min(hardMax, Math.trunc(maxNodes)));
  if (!Number.isFinite(requestedCount)) return 0;
  return Math.max(0, Math.min(cap, Math.trunc(requestedCount)));
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function overlapsAny(
  rects: readonly OverlayRect[],
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  pad = 3,
): boolean {
  for (let i = 0; i < rects.length; i += 1) {
    const r = rects[i]!;
    if (x0 < r.x1 + pad && x1 + pad > r.x0 && y0 < r.y1 + pad && y1 + pad > r.y0) {
      return true;
    }
  }
  return false;
}

function pointInsideAny(
  rects: readonly OverlayRect[],
  x: number,
  y: number,
  pad = 4,
): boolean {
  for (let i = 0; i < rects.length; i += 1) {
    const r = rects[i]!;
    if (x >= r.x0 - pad && x <= r.x1 + pad && y >= r.y0 - pad && y <= r.y1 + pad) {
      return true;
    }
  }
  return false;
}

function normalizeLabelTokens(text: string): readonly string[] {
  const cleaned = text.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  if (!cleaned) return [];
  return cleaned.split(/\s+/).filter((tok) => tok.length >= 2);
}

function areLabelsRedundant(a: string, b: string): boolean {
  const normA = a.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const normB = b.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  if (!normA || !normB) return false;
  if (normA === normB) return true;
  if (normA.length >= 14 && normB.includes(normA)) return true;
  if (normB.length >= 14 && normA.includes(normB)) return true;

  const toksA = normalizeLabelTokens(normA);
  const toksB = normalizeLabelTokens(normB);
  const minLen = Math.min(toksA.length, toksB.length);
  if (minLen < 3) return false;
  const setB = new Set(toksB);
  let shared = 0;
  for (let i = 0; i < toksA.length; i += 1) {
    if (setB.has(toksA[i]!)) shared += 1;
  }
  return shared >= 3 && shared / minLen >= 0.72;
}

function toneToCssClass(tone: DoryBadgeTone | LandmarkAnchor['tone'] | undefined): string {
  if (!tone || tone === 'default') return 'dory-ov-tone--default';
  switch (tone) {
    case 'prelude_indigo':
    case 'recurrent_teal':
    case 'coda_amber':
    case 'swa_cyan':
    case 'full_indigo':
    case 'nlerp_emerald':
    case 'spec_oxide':
    case 'warn':
    case 'gold':
    case 'muted':
    case 'readonly':
      return `dory-ov-tone--${tone}`;
    case 'cyan':
      return 'dory-ov-tone--swa_cyan';
  }
}

function findWorldForPartId(scene: DorySpatialScene, partId: string | null): Vec3 | null {
  if (partId) {
    for (let i = 0; i < scene.landmarks.length; i += 1) {
      const lm = scene.landmarks[i]!;
      if (lm.partId === partId || lm.id === partId) {
        return lm.world;
      }
    }
    for (let i = 0; i < scene.overlayAnchors.length; i += 1) {
      const anc = scene.overlayAnchors[i]!;
      if (anc.id === partId) {
        return anc.worldPos;
      }
    }
    for (let i = 0; i < scene.pickBoxes.length; i += 1) {
      const pb = scene.pickBoxes[i]!;
      if (pb.partId === partId) {
        return [
          (pb.min[0] + pb.max[0]) * 0.5,
          (pb.min[1] + pb.max[1]) * 0.5,
          (pb.min[2] + pb.max[2]) * 0.5,
        ];
      }
    }
  }
  return null;
}

function synthesizeFallbackPopover(
  scene: DorySpatialScene,
  hoveredCell: DoryPickIdentity | null | undefined,
  selectedPartId: string | null | undefined,
): CellPopoverSpec | null {
  if (hoveredCell) {
    const partId = hoveredCell.partId ?? selectedPartId ?? null;
    const world = findWorldForPartId(scene, partId);
    if (!world) return null;

    const coordParts: string[] = [];
    if (hoveredCell.physicalLayerIdx !== null) {
      const pairStr =
        hoveredCell.pairedLayerIdx !== null ? ` (Pair L${hoveredCell.pairedLayerIdx})` : '';
      coordParts.push(`Phys L${hoveredCell.physicalLayerIdx}${pairStr}`);
    }
    if (hoveredCell.loopIteration !== null) {
      coordParts.push(
        hoveredCell.loopIteration === 0 ? 'Stage 1x' : `Loop t=${hoveredCell.loopIteration}`,
      );
    }
    if (hoveredCell.qHeadIndex !== null) coordParts.push(`Q${hoveredCell.qHeadIndex}`);
    if (hoveredCell.kvHeadIndex !== null) coordParts.push(`KV${hoveredCell.kvHeadIndex}`);
    if (hoveredCell.rotaryPlane !== null) coordParts.push(`Plane ${hoveredCell.rotaryPlane}`);
    if (hoveredCell.kvBankSlotId !== null) {
      coordParts.push(
        hoveredCell.kvBankKind !== null
          ? `Bank #${hoveredCell.kvBankSlotId} (${hoveredCell.kvBankKind})`
          : `Bank #${hoveredCell.kvBankSlotId}`,
      );
    }
    if (hoveredCell.row !== null && hoveredCell.col !== null) {
      coordParts.push(`[${hoveredCell.row}, ${hoveredCell.col}]`);
    } else if (hoveredCell.absTokenPos !== null) {
      coordParts.push(`pos=${hoveredCell.absTokenPos}`);
    }

    const tensorLabel = hoveredCell.tensorKey ?? hoveredCell.partId ?? `Dory · ${hoveredCell.viewMode}`;
    const coordLabel =
      coordParts.length > 0 ? coordParts.join(' · ') : (hoveredCell.semanticTag ?? hoveredCell.viewMode);
    const badge =
      hoveredCell.kvPolicy !== null
        ? `${hoveredCell.kvPolicy}${hoveredCell.cacheRole ? ` · ${hoveredCell.cacheRole}` : ''}`
        : hoveredCell.stage !== null
          ? hoveredCell.stage.toUpperCase()
          : undefined;

    return {
      world,
      tensorLabel,
      coordLabel,
      ...(badge !== undefined ? { badge } : {}),
      equation: hoveredCell.semanticTag ?? `View: ${scene.viewMode}`,
      valueText: hoveredCell.sublayerKind ?? 'active',
    };
  }

  if (selectedPartId) {
    const world = findWorldForPartId(scene, selectedPartId);
    if (!world) return null;
    return {
      world,
      tensorLabel: selectedPartId,
      coordLabel: `View ${scene.viewMode} · k=${scene.kLoops} · ${scene.kvCacheMode}`,
      equation: 'Selected 3D architecture element',
      valueText: 'selected',
    };
  }

  return null;
}

// ============================================================================
// POOLED 3D-TO-2D OVERLAY IMPLEMENTATION (ZERO POST-MOUNT ELEMENT CREATION)
// ============================================================================

export class DorySpatialOverlay implements DorySpatialOverlayHandle {
  private readonly host: HTMLElement;
  private readonly root: HTMLDivElement;
  private readonly svg: SVGSVGElement;
  private readonly labelLayer: HTMLDivElement;

  private readonly leaderPath: SVGPathElement;
  private readonly caliperPath: SVGPathElement;
  private readonly popoverSourceIndigoPath: SVGPathElement;
  private readonly popoverSourceTealPath: SVGPathElement;
  private readonly popoverLeaderPath: SVGPathElement;
  private readonly popoverAnchorDot: SVGCircleElement;

  private readonly anchorPillPool: HTMLDivElement[] = [];
  private readonly caliperLabelPool: HTMLDivElement[] = [];
  private readonly kvTickPool: HTMLDivElement[] = [];
  private readonly cellNumPool: HTMLDivElement[] = [];
  private readonly axisHeaderPool: HTMLDivElement[] = [];

  private readonly cellPopover: HTMLDivElement;
  private readonly popoverTensorEl: HTMLSpanElement;
  private readonly popoverCoordEl: HTMLSpanElement;
  private readonly popoverBadgeEl: HTMLSpanElement;
  private readonly popoverDotsWrap: HTMLSpanElement;
  private readonly popoverDotEls: HTMLSpanElement[] = [];
  private readonly popoverEqEl: HTMLSpanElement;
  private readonly popoverValEl: HTMLSpanElement;

  private readonly poolCellNumCapacity: number;
  private readonly poolAxisHeaderCapacity: number;
  private readonly defaultPopoverWidthPx: number;
  private readonly defaultPopoverHeightPx: number;

  private readonly measureCtx: CanvasRenderingContext2D | null;
  private readonly widthCache = new Map<string, number>();

  private stats: DorySpatialOverlayStats;
  private disposed = false;

  constructor(host: HTMLElement, options?: DorySpatialOverlayOptions) {
    this.host = host;

    this.poolCellNumCapacity = computeDoryOverlayCellNumBudget(
      DORY_MAX_OVERLAY_CELL_NUM_NODES,
      options?.maxCellNumNodes,
    );
    this.poolAxisHeaderCapacity = computeDoryOverlayAxisHeaderBudget(
      DORY_MAX_OVERLAY_AXIS_HEADER_NODES,
      options?.maxAxisHeaderNodes,
    );
    this.defaultPopoverWidthPx =
      options?.popoverWidthPx !== undefined &&
      Number.isFinite(options.popoverWidthPx) &&
      options.popoverWidthPx >= 140
        ? options.popoverWidthPx
        : DEFAULT_POPOVER_WIDTH_PX;
    this.defaultPopoverHeightPx =
      options?.popoverHeightPx !== undefined &&
      Number.isFinite(options.popoverHeightPx) &&
      options.popoverHeightPx >= 48
        ? options.popoverHeightPx
        : DEFAULT_POPOVER_HEIGHT_PX;

    let ctx: CanvasRenderingContext2D | null = null;
    try {
      const scratch = document.createElement('canvas');
      if (typeof scratch.getContext === 'function') {
        ctx = scratch.getContext('2d');
      }
    } catch {
      ctx = null;
    }
    this.measureCtx = ctx;

    // 1. Root overlay wrapper (pointer-events: none)
    this.root = document.createElement('div');
    this.root.className = 'dory-ov-root';
    this.root.dataset.testid = 'dory-spatial-overlay';
    this.root.style.pointerEvents = 'none';

    // 2. Pre-allocated SVG layer (leaders, calipers, popover Bezier curves)
    this.svg = document.createElementNS(SVG_NS, 'svg');
    this.svg.setAttribute('class', 'dory-ov-svg');
    this.svg.setAttribute('aria-hidden', 'true');
    this.svg.style.pointerEvents = 'none';

    this.leaderPath = document.createElementNS(SVG_NS, 'path');
    this.leaderPath.setAttribute('class', 'dory-ov-leader-line');

    this.caliperPath = document.createElementNS(SVG_NS, 'path');
    this.caliperPath.setAttribute('class', 'dory-ov-caliper-line');

    this.popoverSourceIndigoPath = document.createElementNS(SVG_NS, 'path');
    this.popoverSourceIndigoPath.setAttribute('class', 'dory-ov-pop-source dory-ov-pop-source--indigo');

    this.popoverSourceTealPath = document.createElementNS(SVG_NS, 'path');
    this.popoverSourceTealPath.setAttribute('class', 'dory-ov-pop-source dory-ov-pop-source--teal');

    this.popoverLeaderPath = document.createElementNS(SVG_NS, 'path');
    this.popoverLeaderPath.setAttribute('class', 'dory-ov-pop-leader');

    this.popoverAnchorDot = document.createElementNS(SVG_NS, 'circle');
    this.popoverAnchorDot.setAttribute('class', 'dory-ov-pop-dot');
    this.popoverAnchorDot.setAttribute('r', '3.2');
    this.popoverAnchorDot.style.display = 'none';

    this.svg.append(
      this.leaderPath,
      this.caliperPath,
      this.popoverSourceIndigoPath,
      this.popoverSourceTealPath,
      this.popoverLeaderPath,
      this.popoverAnchorDot,
    );

    // 3. Pre-allocated HTML label layer & fixed pools
    this.labelLayer = document.createElement('div');
    this.labelLayer.className = 'dory-ov-labels';
    this.labelLayer.style.pointerEvents = 'none';

    for (let i = 0; i < ANCHOR_PILL_POOL_CAPACITY; i += 1) {
      const el = document.createElement('div');
      el.className = 'dory-ov-pill';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.anchorPillPool.push(el);
    }

    for (let i = 0; i < CALIPER_LABEL_POOL_CAPACITY; i += 1) {
      const el = document.createElement('div');
      el.className = 'dory-ov-caliper-label';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.caliperLabelPool.push(el);
    }

    for (let i = 0; i < KV_TICK_POOL_CAPACITY; i += 1) {
      const el = document.createElement('div');
      el.className = 'dory-ov-kv-tick';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.kvTickPool.push(el);
    }

    for (let i = 0; i < this.poolCellNumCapacity; i += 1) {
      const el = document.createElement('div');
      el.className = 'dory-ov-cell-num';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.cellNumPool.push(el);
    }

    for (let i = 0; i < this.poolAxisHeaderCapacity; i += 1) {
      const el = document.createElement('div');
      el.className = 'dory-ov-axis-header';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.axisHeaderPool.push(el);
    }

    // 4. Single pre-allocated floating contextual cell popover
    this.cellPopover = document.createElement('div');
    this.cellPopover.className = 'dory-ov-popover';
    this.cellPopover.dataset.testid = 'dory-cell-popover';
    this.cellPopover.style.display = 'none';

    const headRow = document.createElement('div');
    headRow.className = 'dory-ov-popover-head';
    this.popoverTensorEl = document.createElement('span');
    this.popoverTensorEl.className = 'dory-ov-popover-tensor';
    this.popoverCoordEl = document.createElement('span');
    this.popoverCoordEl.className = 'dory-ov-popover-coord';
    this.popoverBadgeEl = document.createElement('span');
    this.popoverBadgeEl.className = 'dory-ov-popover-badge';
    this.popoverBadgeEl.style.display = 'none';
    headRow.append(this.popoverTensorEl, this.popoverCoordEl, this.popoverBadgeEl);

    const bodyRow = document.createElement('div');
    bodyRow.className = 'dory-ov-popover-body';
    this.popoverDotsWrap = document.createElement('span');
    this.popoverDotsWrap.className = 'dory-ov-popover-dots';
    this.popoverDotsWrap.style.display = 'none';
    for (let i = 0; i < MAX_POPOVER_OPERAND_DOTS; i += 1) {
      const dot = document.createElement('span');
      dot.className = 'dory-ov-operand-dot';
      dot.style.display = 'none';
      this.popoverDotsWrap.appendChild(dot);
      this.popoverDotEls.push(dot);
    }
    this.popoverEqEl = document.createElement('span');
    this.popoverEqEl.className = 'dory-ov-popover-eq';
    this.popoverValEl = document.createElement('span');
    this.popoverValEl.className = 'dory-ov-popover-val';
    bodyRow.append(this.popoverDotsWrap, this.popoverEqEl, this.popoverValEl);

    this.cellPopover.append(headRow, bodyRow);
    this.labelLayer.appendChild(this.cellPopover);

    this.root.append(this.svg, this.labelLayer);
    this.host.appendChild(this.root);

    this.stats = Object.freeze({
      renderedCellNumNodes: 0,
      renderedAxisHeaderNodes: 0,
      renderedAnchorPills: 0,
      renderedCalipers: 0,
      popoverVisible: false,
      poolCellNumCapacity: this.poolCellNumCapacity,
    });
  }

  private measureText(text: string, font: string, pxPerCharFallback = 6.2): number {
    const key = `${font}|${text}`;
    const cached = this.widthCache.get(key);
    if (cached !== undefined) return cached;

    let width: number;
    if (this.measureCtx !== null) {
      this.measureCtx.font = font;
      const measured = this.measureCtx.measureText(text).width;
      width = Number.isFinite(measured) && measured > 0 ? Math.ceil(measured) : Math.ceil(text.length * pxPerCharFallback);
    } else {
      width = Math.ceil(text.length * pxPerCharFallback);
    }

    if (this.widthCache.size >= MAX_OVERLAY_WIDTH_CACHE_ENTRIES) {
      const oldest = this.widthCache.keys().next().value;
      if (oldest !== undefined) {
        this.widthCache.delete(oldest);
      }
    }
    this.widthCache.set(key, width);
    return width;
  }

  private hideAll(): void {
    for (let i = 0; i < this.anchorPillPool.length; i += 1) {
      const el = this.anchorPillPool[i]!;
      el.style.display = 'none';
      el.style.opacity = '1';
      el.className = 'dory-ov-pill';
    }
    for (let i = 0; i < this.caliperLabelPool.length; i += 1) {
      this.caliperLabelPool[i]!.style.display = 'none';
    }
    for (let i = 0; i < this.kvTickPool.length; i += 1) {
      const el = this.kvTickPool[i]!;
      el.style.display = 'none';
      el.className = 'dory-ov-kv-tick';
    }
    for (let i = 0; i < this.cellNumPool.length; i += 1) {
      const el = this.cellNumPool[i]!;
      el.style.display = 'none';
      el.className = 'dory-ov-cell-num';
    }
    for (let i = 0; i < this.axisHeaderPool.length; i += 1) {
      const el = this.axisHeaderPool[i]!;
      el.style.display = 'none';
      el.className = 'dory-ov-axis-header';
    }

    this.cellPopover.style.display = 'none';
    this.cellPopover.classList.remove('is-accumulating');
    this.popoverValEl.classList.remove('is-accumulating', 'is-masked');

    this.leaderPath.setAttribute('d', '');
    this.caliperPath.setAttribute('d', '');
    this.popoverSourceIndigoPath.setAttribute('d', '');
    this.popoverSourceTealPath.setAttribute('d', '');
    this.popoverLeaderPath.setAttribute('d', '');
    this.popoverAnchorDot.style.display = 'none';
  }

  clear(): void {
    this.hideAll();
    this.stats = Object.freeze({
      renderedCellNumNodes: 0,
      renderedAxisHeaderNodes: 0,
      renderedAnchorPills: 0,
      renderedCalipers: 0,
      popoverVisible: false,
      poolCellNumCapacity: this.poolCellNumCapacity,
    });
  }

  getStats(): DorySpatialOverlayStats {
    return this.stats;
  }

  update(input: DorySpatialOverlayUpdateInput): DorySpatialOverlayStats {
    if (this.disposed) {
      return this.stats;
    }

    this.hideAll();

    const { scene, snapshot } = input;
    const { hostWidth, hostHeight } = snapshot.metrics;
    this.svg.setAttribute('viewBox', `0 0 ${hostWidth} ${hostHeight}`);

    const minY = Math.max(8, input.topInsetPx ?? 12);
    const maxY = Math.max(minY + 48, hostHeight - Math.max(10, input.bottomInsetPx ?? 16));

    const hudAvoidRects: DoryOverlayRect[] = [];
    const occupied: MutablePlacedRect[] = [];
    if (input.avoidRects) {
      for (let i = 0; i < input.avoidRects.length; i += 1) {
        const r = input.avoidRects[i]!;
        hudAvoidRects.push(r);
        occupied.push({ x0: r.x0, y0: r.y0, x1: r.x1, y1: r.y1 });
      }
    }

    const leaderSegs: string[] = [];
    const caliperSegs: string[] = [];
    const activePart = input.hoveredCell?.partId ?? input.selectedPartId ?? null;
    const isSheetCutaway = scene.sheetCount > 0 && scene.viewMode !== 'macro';

    const pOrigin = projectWorldToHostCss([0, 0.05, 0], snapshot);
    const pUnit = projectWorldToHostCss([1, 0.05, 0], snapshot);
    const pxPerWorldUnit =
      pOrigin.visible && pUnit.visible ? Math.hypot(pUnit.x - pOrigin.x, pUnit.y - pOrigin.y) : 40;
    const isCloseZoom = pxPerWorldUnit >= 64;

    // Pre-scan visible anchor horizontal span so side-flanked overlay anchors clear central 3D plates
    let projMinX = hostWidth;
    let projMaxX = 0;
    let projCount = 0;
    for (let i = 0; i < scene.landmarks.length; i += 1) {
      const p = projectWorldToHostCss(scene.landmarks[i]!.world, snapshot);
      if (p.visible && p.x >= 0 && p.x <= hostWidth && p.y >= minY - 24 && p.y <= maxY + 24) {
        if (p.x < projMinX) projMinX = p.x;
        if (p.x > projMaxX) projMaxX = p.x;
        projCount += 1;
      }
    }
    for (let i = 0; i < scene.overlayAnchors.length; i += 1) {
      const p = projectWorldToHostCss(scene.overlayAnchors[i]!.worldPos, snapshot);
      if (p.visible && p.x >= 0 && p.x <= hostWidth && p.y >= minY - 24 && p.y <= maxY + 24) {
        if (p.x < projMinX) projMinX = p.x;
        if (p.x > projMaxX) projMaxX = p.x;
        projCount += 1;
      }
    }
    const sceneMidX = projCount > 0 ? (projMinX + projMaxX) * 0.5 : hostWidth * 0.5;
    const plateHalfClearance = projCount > 0 ? clamp((projMaxX - projMinX) * 0.24, 32, 92) : 48;

    const pillFont = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const monoFont = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';

    // 1. Floating contextual cell popover (first priority so anchors/numbers route cleanly around it)
    const activePopoverSpec =
      scene.activeCellPopover ??
      synthesizeFallbackPopover(scene, input.hoveredCell, input.selectedPartId);
    const isHoverPopover = Boolean(scene.activeCellPopover || input.hoveredCell);
    const popoverRect = this.renderCellPopover(
      activePopoverSpec,
      isHoverPopover,
      snapshot,
      occupied,
      hudAvoidRects,
      minY,
      maxY,
      scene.activeCellGrids,
    );

    // 2. Pooled Anchor & Landmark Pills (`scene.landmarks` + `scene.overlayAnchors`)
    const placedLabels: { readonly label: string; readonly ax: number; readonly ay: number }[] = [];
    let pillCursor = 0;
    let sideToggle = 0;

    const placePill = (
      label: string,
      sublabel: string | undefined,
      worldPos: Vec3,
      toneClass: string,
      align: LandmarkAnchor['align'] | undefined,
      isActive: boolean,
      isDetail: boolean,
      badge: string | undefined,
      preferSideFlank: boolean,
    ): void => {
      if (pillCursor >= this.anchorPillPool.length) return;
      const pt = projectWorldToHostCss(worldPos, snapshot);
      if (!pt.visible || pt.x < -30 || pt.x > hostWidth + 30 || pt.y < minY - 24 || pt.y > maxY + 24) {
        return;
      }

      const display = badge ? `${label} · ${badge}` : label;
      if (!isActive) {
        for (let i = 0; i < placedLabels.length; i += 1) {
          const prev = placedLabels[i]!;
          if (
            Math.hypot(pt.x - prev.ax, pt.y - prev.ay) <= 145 &&
            areLabelsRedundant(display, prev.label)
          ) {
            return;
          }
        }
      }

      const maxLabelW = Math.max(110, Math.min(440, hostWidth - 20));
      const rawW = this.measureText(display, pillFont, 6.0) + 14;
      const w = Math.min(rawW, maxLabelW);
      const h = isDetail ? 17 : 18;
      const gap = isSheetCutaway ? 7 : 10;

      const padX = hostWidth < 760 ? 6 : 8;
      const maxX = Math.max(padX, hostWidth - w - padX);
      const maxTopY = Math.max(minY, maxY - h);

      let flankLeft = pt.x < sceneMidX - 10;
      if (Math.abs(pt.x - sceneMidX) <= 10) {
        flankLeft = sideToggle % 2 === 0;
        sideToggle += 1;
      }

      const leftFlankX = clamp(Math.min(pt.x - w - gap - 10, projMinX - w - 12), padX, maxX);
      const rightFlankX = clamp(Math.max(pt.x + gap + 10, projMaxX + 12), padX, maxX);
      const nearLeftX = clamp(pt.x - w - plateHalfClearance, padX, maxX);
      const nearRightX = clamp(pt.x + plateHalfClearance, padX, maxX);

      let idealX = pt.x - w * 0.5;
      let idealY = pt.y - h - gap;
      if (preferSideFlank && !isActive) {
        idealX = flankLeft ? nearLeftX : nearRightX;
        idealY = pt.y - h * 0.5;
      } else if (align === 'left') {
        idealX = pt.x - w - gap;
        idealY = pt.y - h * 0.5;
      } else if (align === 'right') {
        idealX = pt.x + gap;
        idealY = pt.y - h * 0.5;
      } else if (align === 'bottom') {
        idealX = pt.x - w * 0.5;
        idealY = pt.y + gap;
      }

      let lx = clamp(idealX, padX, maxX);
      let ly = clamp(idealY, minY, maxTopY);

      if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
        const primaryFlankX = flankLeft ? leftFlankX : rightFlankX;
        const secondaryFlankX = flankLeft ? rightFlankX : leftFlankX;
        const primaryNearX = flankLeft ? nearLeftX : nearRightX;
        const secondaryNearX = flankLeft ? nearRightX : nearLeftX;

        const candidates: readonly (readonly [number, number])[] = [
          [primaryFlankX, clamp(pt.y - h * 0.5, minY, maxTopY)],
          [primaryNearX, clamp(pt.y - h * 0.5, minY, maxTopY)],
          [secondaryFlankX, clamp(pt.y - h * 0.5, minY, maxTopY)],
          [secondaryNearX, clamp(pt.y - h * 0.5, minY, maxTopY)],
          [primaryFlankX, clamp(pt.y - h - 6, minY, maxTopY)],
          [primaryFlankX, clamp(pt.y + 6, minY, maxTopY)],
          [secondaryFlankX, clamp(pt.y - h - 6, minY, maxTopY)],
          [secondaryFlankX, clamp(pt.y + 6, minY, maxTopY)],
          [lx, clamp(ly - (h + 4), minY, maxTopY)],
          [lx, clamp(ly + (h + 4), minY, maxTopY)],
          [clamp(pt.x - w * 0.5, padX, maxX), clamp(pt.y - h - gap - 4, minY, maxTopY)],
          [clamp(pt.x - w * 0.5, padX, maxX), clamp(pt.y + gap + 4, minY, maxTopY)],
        ];

        let placed = false;
        for (let i = 0; i < candidates.length; i += 1) {
          const [cx, cy] = candidates[i]!;
          if (!overlapsAny(occupied, cx, cy, cx + w, cy + h)) {
            lx = cx;
            ly = cy;
            placed = true;
            break;
          }
        }
        if (!placed && !isActive) return;
      }

      const el = this.anchorPillPool[pillCursor]!;
      pillCursor += 1;
      el.className = `dory-ov-pill ${toneClass}${isDetail ? ' is-detail' : ''}${isActive ? ' is-active' : ''}`;
      el.style.opacity = isDetail && !isActive && !isCloseZoom ? DIMMED_LABEL_OPACITY : '1';
      el.style.maxWidth = `${maxLabelW}px`;
      el.textContent = display;
      el.title = sublabel ? `${display} — ${sublabel}` : display;
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;

      const rect: MutablePlacedRect = { x0: lx, y0: ly, x1: lx + w, y1: ly + h };
      occupied.push(rect);
      placedLabels.push({ label: display, ax: pt.x, ay: pt.y });

      const nearX = clamp(pt.x, rect.x0, rect.x1);
      const nearY = clamp(pt.y, rect.y0, rect.y1);
      const edgeDist = Math.hypot(nearX - pt.x, nearY - pt.y);
      const nudgeDist = Math.hypot(lx - idealX, ly - idealY);
      if (
        preferSideFlank ||
        !isSheetCutaway ||
        edgeDist > LABEL_NUDGE_LEADER_THRESHOLD_PX ||
        (nudgeDist > LABEL_NUDGE_LEADER_THRESHOLD_PX && edgeDist > 3)
      ) {
        const anchorX = clamp(pt.x, 4, hostWidth - 4);
        const anchorY = clamp(pt.y, minY, maxY);
        const seg = buildElbowLeaderPath(anchorX, anchorY, rect);
        if (seg) leaderSegs.push(seg);
      }
    };

    // 2a. Scene landmarks (active first, primary before detail)
    const orderedLandmarks = [...scene.landmarks].sort((a, b) => {
      const aAct = a.partId !== undefined && a.partId === activePart ? 3 : a.tier === 'detail' ? 0 : 2;
      const bAct = b.partId !== undefined && b.partId === activePart ? 3 : b.tier === 'detail' ? 0 : 2;
      return bAct - aAct;
    });

    for (let i = 0; i < orderedLandmarks.length; i += 1) {
      const lm = orderedLandmarks[i]!;
      const isAct = Boolean(lm.partId !== undefined && lm.partId === activePart);
      const isDetail = lm.tier === 'detail';
      if (isDetail && !isAct && !isCloseZoom) {
        continue;
      }
      placePill(
        lm.label,
        lm.sublabel,
        lm.world,
        toneToCssClass(lm.tone),
        lm.align,
        isAct,
        isDetail,
        undefined,
        false,
      );
    }

    // 2b. Scene overlayAnchors (side-flanked to keep RG32F heatmaps legible)
    for (let i = 0; i < scene.overlayAnchors.length; i += 1) {
      const anchor: DoryOverlayAnchor = scene.overlayAnchors[i]!;
      const isAct = anchor.id === activePart;
      placePill(
        anchor.label,
        anchor.sublabel,
        anchor.worldPos,
        toneToCssClass(anchor.tone),
        'top',
        isAct,
        false,
        anchor.badge,
        true,
      );
    }

    // 3. Dimension calipers
    let caliperCursor = 0;
    for (let i = 0; i < scene.calipers.length; i += 1) {
      const cal = scene.calipers[i]!;
      const pa = projectWorldToHostCss(cal.aWorld, snapshot);
      const pb = projectWorldToHostCss(cal.bWorld, snapshot);
      if (!pa.visible || !pb.visible) continue;
      if (Math.hypot(pb.x - pa.x, pb.y - pa.y) < (cal.minPixels ?? 48)) continue;

      const off = cal.offsetPx ?? 8;
      const ax = clamp(pa.x, 8, hostWidth - 8);
      const ay = clamp(pa.y + off, minY, maxY);
      const bx = clamp(pb.x, 8, hostWidth - 8);
      const by = clamp(pb.y + off, minY, maxY);

      caliperSegs.push(
        `M${ax.toFixed(1)},${(ay - 3).toFixed(1)}L${ax.toFixed(1)},${(ay + 3).toFixed(1)}` +
          `M${ax.toFixed(1)},${ay.toFixed(1)}L${bx.toFixed(1)},${by.toFixed(1)}` +
          `M${bx.toFixed(1)},${(by - 3).toFixed(1)}L${bx.toFixed(1)},${(by + 3).toFixed(1)}`,
      );

      if (caliperCursor < this.caliperLabelPool.length) {
        const w = this.measureText(cal.label, monoFont, 6.0) + 12;
        const h = 16;
        const mx = clamp((ax + bx) * 0.5 - w * 0.5, 8, Math.max(8, hostWidth - w - 8));
        const my = clamp((ay + by) * 0.5 + 4, minY, Math.max(minY, maxY - h));
        if (!overlapsAny(occupied, mx, my, mx + w, my + h)) {
          const calEl = this.caliperLabelPool[caliperCursor]!;
          caliperCursor += 1;
          calEl.textContent = cal.label;
          calEl.style.display = 'inline-flex';
          calEl.style.transform = `translate3d(${mx.toFixed(1)}px, ${my.toFixed(1)}px, 0)`;
          occupied.push({ x0: mx, y0: my, x1: mx + w, y1: my + h });
        }
      }
    }

    // 4. KV Depth Ticks (`W=8` SWA window & `cap_swa=16` ring boundary highlights)
    if (scene.kvDepthTicks.length > 0 && pxPerWorldUnit >= 18) {
      let kvTickCursor = 0;
      for (
        let i = 0;
        i < scene.kvDepthTicks.length && kvTickCursor < this.kvTickPool.length;
        i += 1
      ) {
        const tick = scene.kvDepthTicks[i]!;
        const pt = projectWorldToHostCss(tick.world, snapshot);
        if (!pt.visible || pt.x < 8 || pt.x > hostWidth - 22 || pt.y < minY || pt.y > maxY) {
          continue;
        }
        const isSwaWindow = tick.token === 8;
        const isSwaCap = tick.token === 16;
        const text = isSwaWindow ? 'W=8' : isSwaCap ? 'cap=16' : `${tick.token}`;
        const w = this.measureText(text, monoFont, 5.8) + 8;
        const h = 14;
        const lx = clamp(pt.x + 3, 8, Math.max(8, hostWidth - w - 8));
        const ly = clamp(pt.y - h * 0.5, minY, Math.max(minY, maxY - h));
        if (!isSwaWindow && !isSwaCap && overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
          continue;
        }
        const tickEl = this.kvTickPool[kvTickCursor]!;
        kvTickCursor += 1;
        tickEl.textContent = text;
        tickEl.className = isSwaWindow
          ? 'dory-ov-kv-tick is-swa-boundary'
          : isSwaCap
            ? 'dory-ov-kv-tick is-cap-boundary'
            : 'dory-ov-kv-tick';
        tickEl.style.display = 'inline-flex';
        tickEl.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
        occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
      }
    }

    // 5. Zoom-adaptive on-cell numeric values (<= 64 hard cap) & exterior axis headers (<= 24)
    const gridStats = this.renderActiveCellGrids(
      scene.activeCellGrids,
      snapshot,
      occupied,
      hudAvoidRects,
      popoverRect,
      minY,
      maxY,
      input.showCellNumbers !== false,
    );

    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.caliperPath.setAttribute('d', caliperSegs.join(''));

    this.stats = Object.freeze({
      renderedCellNumNodes: gridStats.renderedCellNumNodes,
      renderedAxisHeaderNodes: gridStats.renderedAxisHeaderNodes,
      renderedAnchorPills: pillCursor,
      renderedCalipers: caliperCursor,
      popoverVisible: popoverRect !== null,
      poolCellNumCapacity: this.poolCellNumCapacity,
    });
    return this.stats;
  }

  private renderCellPopover(
    spec: CellPopoverSpec | null | undefined,
    isHoverActive: boolean,
    snapshot: CameraSnapshot,
    occupied: MutablePlacedRect[],
    hudAvoidRects: readonly OverlayRect[],
    minY: number,
    maxY: number,
    activeCellGrids: readonly ActiveSheetCellGrid[],
  ): MutablePlacedRect | null {
    if (!spec) return null;
    const { hostWidth, hostHeight } = snapshot.metrics;
    const pt = projectWorldToHostCss(spec.world, snapshot);
    if (!pt.visible || pt.x < -30 || pt.x > hostWidth + 30 || pt.y < minY - 30 || pt.y > maxY + 30) {
      return null;
    }

    const anchorX = clamp(pt.x, 6, hostWidth - 6);
    const anchorY = clamp(pt.y, minY, maxY);

    // Suppress static selection popover when its 3D anchor sits underneath a HUD panel
    if (!isHoverActive && pointInsideAny(hudAvoidRects, anchorX, anchorY, 4)) {
      return null;
    }

    const headFont = '700 10.5px system-ui, -apple-system, sans-serif';
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
      for (let i = 0; i < this.popoverDotEls.length; i += 1) {
        const dotEl = this.popoverDotEls[i]!;
        if (i < dotCount) {
          dotEl.style.display = 'inline-block';
          dotEl.style.backgroundColor = OPERAND_SWATCH_HEX[operandColors[i]!];
        } else {
          dotEl.style.display = 'none';
        }
      }
    } else {
      this.popoverDotsWrap.style.display = 'none';
      for (let i = 0; i < this.popoverDotEls.length; i += 1) {
        this.popoverDotEls[i]!.style.display = 'none';
      }
    }

    this.popoverEqEl.textContent = spec.equation;
    this.popoverValEl.textContent = spec.valueText;

    const isAccumulating = Boolean(spec.isAccumulating);
    const lowerVal = spec.valueText.toLowerCase();
    const isMaskedVal =
      spec.valueText.includes('∞') ||
      lowerVal.includes('-inf') ||
      lowerVal.includes('mask');
    this.cellPopover.classList.toggle('is-accumulating', isAccumulating);
    this.popoverValEl.classList.toggle('is-accumulating', isAccumulating);
    this.popoverValEl.classList.toggle('is-masked', isMaskedVal);

    const headW =
      this.measureText(spec.tensorLabel, headFont, 6.4) +
      this.measureText(spec.coordLabel, coordFont, 6.1) +
      (spec.badge ? this.measureText(spec.badge, badgeFont, 5.6) + 22 : 0) +
      36;
    const dotsW = dotCount > 0 ? dotCount * 9 + 6 : 0;
    const bodyW =
      dotsW +
      this.measureText(spec.equation, eqFont, 6.2) +
      this.measureText(spec.valueText, valFont, 6.4) +
      48;

    const maxCardW = Math.max(180, Math.min(Math.max(440, this.defaultPopoverWidthPx), hostWidth - 20));
    const cardW = clamp(Math.max(headW, bodyW, 200), Math.min(180, maxCardW), maxCardW);
    const estH = 50 + (headW > cardW ? 18 : 0) + (bodyW > cardW ? 18 : 0);
    const cardH = clamp(estH, 48, Math.max(88, this.defaultPopoverHeightPx));

    const avoidRects: OverlayRect[] = [...hudAvoidRects];
    let targetSheetRect: OverlayRect | null = null;
    let bestTargetDist = Infinity;

    if (activeCellGrids.length > 0) {
      const corners: readonly (readonly [number, number])[] = [
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.5, 0.5],
        [-0.5, 0.5],
      ];
      for (let i = 0; i < activeCellGrids.length; i += 1) {
        const grid = activeCellGrids[i]!;
        if (grid.cols <= 0 || grid.rows <= 0) continue;
        const frame = makeSheetFrame3D(grid.originWorld, grid.extent, grid.axisU, grid.axisN);
        const faceLy = dot3(sub3(snapshot.eye, frame.center), frame.axisN) >= 0 ? 0.5 : -0.5;
        let sx0 = Infinity;
        let sy0 = Infinity;
        let sx1 = -Infinity;
        let sy1 = -Infinity;
        let visCount = 0;
        for (let c = 0; c < corners.length; c += 1) {
          const [lx, lz] = corners[c]!;
          const cPt = projectWorldToHostCss(localToWorldOnFrame(frame, lx, faceLy, lz), snapshot);
          if (!cPt.visible) continue;
          visCount += 1;
          if (cPt.x < sx0) sx0 = cPt.x;
          if (cPt.y < sy0) sy0 = cPt.y;
          if (cPt.x > sx1) sx1 = cPt.x;
          if (cPt.y > sy1) sy1 = cPt.y;
        }
        if (visCount < 2 || sx1 <= 0 || sx0 >= hostWidth || sy1 <= 0 || sy0 >= hostHeight) {
          continue;
        }
        const sheetRect: OverlayRect = { x0: sx0, y0: sy0, x1: sx1, y1: sy1 };
        avoidRects.push(sheetRect);

        const dVec = sub3(spec.world, frame.center);
        const uNorm = Math.abs(dot3(dVec, frame.axisU)) / Math.max(1e-4, frame.extent[0]);
        const vNorm = Math.abs(dot3(dVec, frame.axisV)) / Math.max(1e-4, frame.extent[2]);
        const nDist = Math.abs(dot3(dVec, frame.axisN));
        if (
          uNorm <= 0.56 &&
          vNorm <= 0.56 &&
          nDist <= frame.extent[1] * 0.5 + 0.18 &&
          nDist < bestTargetDist
        ) {
          bestTargetDist = nDist;
          targetSheetRect = sheetRect;
        }
      }
    }

    const projectedSources: { readonly sx: number; readonly sy: number; readonly role: 'indigo' | 'teal' }[] = [];
    if (spec.sourceWorlds && spec.sourceWorlds.length > 0) {
      const pad = 24;
      for (let i = 0; i < spec.sourceWorlds.length; i += 1) {
        const src = spec.sourceWorlds[i]!;
        const spt = projectWorldToHostCss(src.world, snapshot);
        if (!spt.visible || spt.x < -40 || spt.x > hostWidth + 40 || spt.y < minY - 40 || spt.y > maxY + 40) {
          continue;
        }
        const sx = clamp(spt.x, 4, hostWidth - 4);
        const sy = clamp(spt.y, minY, maxY);
        projectedSources.push({ sx, sy, role: src.role });
        avoidRects.push({ x0: sx - pad, y0: sy - pad, x1: sx + pad, y1: sy + pad });
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
    this.cellPopover.style.transform = `translate3d(${placed.x.toFixed(1)}px, ${placed.y.toFixed(1)}px, 0)`;

    const placedRect: MutablePlacedRect = {
      x0: placed.x,
      y0: placed.y,
      x1: placed.x + placed.width,
      y1: placed.y + placed.height,
    };
    occupied.push(placedRect);

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

    if (projectedSources.length > 0) {
      const indigoSegs: string[] = [];
      const tealSegs: string[] = [];
      for (let i = 0; i < projectedSources.length; i += 1) {
        const src = projectedSources[i]!;
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

  private renderActiveCellGrids(
    grids: readonly ActiveSheetCellGrid[],
    snapshot: CameraSnapshot,
    occupied: MutablePlacedRect[],
    hudAvoidRects: readonly OverlayRect[],
    popoverRect: MutablePlacedRect | null,
    minY: number,
    maxY: number,
    showCellNumbers: boolean,
  ): { readonly renderedCellNumNodes: number; readonly renderedAxisHeaderNodes: number } {
    if (grids.length === 0) {
      return { renderedCellNumNodes: 0, renderedAxisHeaderNodes: 0 };
    }

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
    for (let i = 0; i < grids.length; i += 1) {
      const grid = grids[i]!;
      if (grid.cols <= 0 || grid.rows <= 0) continue;
      const size = computeProjectedCellSize(grid, snapshot);
      if (!size.visible) continue;
      const showNums =
        showCellNumbers &&
        this.poolCellNumCapacity > 0 &&
        shouldShowCellNumbers(size.cellW, size.cellH);
      const showHeaders =
        this.poolAxisHeaderCapacity > 0 && shouldShowAxisHeaders(size.cellW, size.cellH);
      if (!showNums && !showHeaders) continue;
      const frame = makeSheetFrame3D(grid.originWorld, grid.extent, grid.axisU, grid.axisN);
      const faceLy = dot3(sub3(snapshot.eye, frame.center), frame.axisN) >= 0 ? 0.5 : -0.5;
      prepared.push({ grid, frame, faceLy, size, showNums, showHeaders });
    }

    if (prepared.length === 0) {
      return { renderedCellNumNodes: 0, renderedAxisHeaderNodes: 0 };
    }

    let cellNumCursor = 0;
    const cellNumRects: MutablePlacedRect[] = [];
    const maxCellNodes = computeDoryOverlayCellNumBudget(
      this.poolCellNumCapacity,
      this.cellNumPool.length,
    );

    const placeCellNum = (pg: PreparedGrid, col: number, row: number): boolean => {
      if (cellNumCursor >= maxCellNodes) return false;
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
      if (!pt.visible || pt.x < 12 || pt.x > hostWidth - 12 || pt.y < minY + 6 || pt.y > maxY - 6) {
        return false;
      }
      if (
        popoverRect !== null &&
        pt.x >= popoverRect.x0 - 4 &&
        pt.x <= popoverRect.x1 + 4 &&
        pt.y >= popoverRect.y0 - 4 &&
        pt.y <= popoverRect.y1 + 4
      ) {
        return false;
      }
      if (pointInsideAny(hudAvoidRects, pt.x, pt.y, 2)) {
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

      const rawValue = hasNumericOverride ? grid.activeValueOverride! : grid.values[idx]!;
      const text = hasTextOverride
        ? grid.activeValueTextOverride!
        : formatCellOverlayValue(rawValue, grid.format ?? 'signed', isMasked);

      const w = this.measureText(text, cellNumFont, 5.6) + 6;
      const h = 13;
      const lx = pt.x - w * 0.5;
      const ly = pt.y - h * 0.5;

      const el = this.cellNumPool[cellNumCursor]!;
      cellNumCursor += 1;
      el.textContent = text;
      el.className =
        'dory-ov-cell-num' +
        (isActive ? ' is-active' : '') +
        (isAccumulating ? ' is-accumulating' : '') +
        (isMasked ? ' is-masked' : '');
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      cellNumRects.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
      return true;
    };

    // Pass 1: active cell and active row/col across all visible grids
    const remainingPerGrid: {
      readonly pg: PreparedGrid;
      readonly coords: readonly (readonly [number, number])[];
    }[] = [];

    for (let i = 0; i < prepared.length; i += 1) {
      const pg = prepared[i]!;
      if (!pg.showNums) continue;
      const ordered = prioritizeSheetCellCoords(
        pg.grid.cols,
        pg.grid.rows,
        pg.grid.activeCol,
        pg.grid.activeRow,
      );
      const hasActiveCol =
        pg.grid.activeCol !== undefined && pg.grid.activeCol >= 0 && pg.grid.activeCol < pg.grid.cols;
      const hasActiveRow =
        pg.grid.activeRow !== undefined && pg.grid.activeRow >= 0 && pg.grid.activeRow < pg.grid.rows;
      const rest: (readonly [number, number])[] = [];
      for (let j = 0; j < ordered.length; j += 1) {
        const coord = ordered[j]!;
        const [c, r] = coord;
        const isPriority =
          (hasActiveCol && c === pg.grid.activeCol) || (hasActiveRow && r === pg.grid.activeRow);
        if (isPriority) {
          if (cellNumCursor < maxCellNodes) {
            placeCellNum(pg, c, r);
          }
        } else {
          rest.push(coord);
        }
      }
      remainingPerGrid.push({ pg, coords: rest });
    }

    // Pass 2: fill remaining budget strictly up to maxCellNodes (<= 64)
    for (let i = 0; i < remainingPerGrid.length && cellNumCursor < maxCellNodes; i += 1) {
      const item = remainingPerGrid[i]!;
      for (let j = 0; j < item.coords.length && cellNumCursor < maxCellNodes; j += 1) {
        const [c, r] = item.coords[j]!;
        placeCellNum(item.pg, c, r);
      }
    }

    // Exterior row/column axis header pills (strictly <= poolAxisHeaderCapacity <= 24)
    let headerCursor = 0;
    const maxHeaders = Math.min(this.poolAxisHeaderCapacity, this.axisHeaderPool.length);

    const placeExteriorHeader = (
      text: string,
      edgeWorld: Vec3,
      insideWorld: Vec3,
      isActive: boolean,
      isDim: boolean,
    ): void => {
      if (headerCursor >= maxHeaders || !text) return;
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

      const w = this.measureText(text, headerFont, 5.8) + 8;
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
      lx = clamp(lx, 6, Math.max(6, hostWidth - w - 6));
      ly = clamp(ly, minY, Math.max(minY, maxY - h));

      if (
        overlapsAny(occupied, lx, ly, lx + w, ly + h, 2) ||
        overlapsAny(cellNumRects, lx, ly, lx + w, ly + h, 2)
      ) {
        return;
      }

      const el = this.axisHeaderPool[headerCursor]!;
      headerCursor += 1;
      el.textContent = text;
      el.className =
        'dory-ov-axis-header' + (isActive ? ' is-active' : '') + (isDim ? ' is-dim' : '');
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
    };

    for (let i = 0; i < prepared.length && headerCursor < maxHeaders; i += 1) {
      const pg = prepared[i]!;
      if (!pg.showHeaders) continue;
      const { grid, frame, faceLy, size } = pg;

      // Column headers along top edge (lz = -0.5)
      if (!grid.suppressColHeaders && grid.colLabels && grid.colLabels.length > 0 && size.colStepPx >= 18) {
        const colStride = Math.max(1, Math.ceil(32 / Math.max(1, size.colStepPx)));
        const activeCol = grid.activeCol;
        if (activeCol !== undefined && activeCol >= 0 && activeCol < grid.colLabels.length) {
          const lx = (activeCol + 0.5) / grid.cols - 0.5;
          placeExteriorHeader(
            grid.colLabels[activeCol]!,
            localToWorldOnFrame(frame, lx, faceLy, -0.5),
            localToWorldOnFrame(frame, lx, faceLy, -0.25),
            true,
            false,
          );
        }
        for (let c = 0; c < grid.cols && c < grid.colLabels.length && headerCursor < maxHeaders; c += colStride) {
          if (c === activeCol) continue;
          const lx = (c + 0.5) / grid.cols - 0.5;
          placeExteriorHeader(
            grid.colLabels[c]!,
            localToWorldOnFrame(frame, lx, faceLy, -0.5),
            localToWorldOnFrame(frame, lx, faceLy, -0.25),
            false,
            activeCol !== undefined,
          );
        }
      } else if (grid.colDimLabel && headerCursor < maxHeaders) {
        placeExteriorHeader(
          grid.colDimLabel,
          localToWorldOnFrame(frame, 0, faceLy, -0.5),
          localToWorldOnFrame(frame, 0, faceLy, -0.2),
          false,
          true,
        );
      }

      // Row headers along left edge (lx = -0.5)
      if (!grid.suppressRowHeaders && grid.rowLabels && grid.rowLabels.length > 0 && size.rowStepPx >= 14) {
        const rowStride = Math.max(1, Math.ceil(20 / Math.max(1, size.rowStepPx)));
        const activeRow = grid.activeRow;
        if (activeRow !== undefined && activeRow >= 0 && activeRow < grid.rowLabels.length) {
          const lz = (activeRow + 0.5) / grid.rows - 0.5;
          placeExteriorHeader(
            grid.rowLabels[activeRow]!,
            localToWorldOnFrame(frame, -0.5, faceLy, lz),
            localToWorldOnFrame(frame, -0.25, faceLy, lz),
            true,
            false,
          );
        }
        for (let r = 0; r < grid.rows && r < grid.rowLabels.length && headerCursor < maxHeaders; r += rowStride) {
          if (r === activeRow) continue;
          const lz = (r + 0.5) / grid.rows - 0.5;
          placeExteriorHeader(
            grid.rowLabels[r]!,
            localToWorldOnFrame(frame, -0.5, faceLy, lz),
            localToWorldOnFrame(frame, -0.25, faceLy, lz),
            false,
            activeRow !== undefined,
          );
        }
      } else if (grid.rowDimLabel && headerCursor < maxHeaders) {
        placeExteriorHeader(
          grid.rowDimLabel,
          localToWorldOnFrame(frame, -0.5, faceLy, 0),
          localToWorldOnFrame(frame, -0.2, faceLy, 0),
          false,
          true,
        );
      }
    }

    return {
      renderedCellNumNodes: cellNumCursor,
      renderedAxisHeaderNodes: headerCursor,
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.clear();
    this.widthCache.clear();
    if (this.root.parentNode === this.host) {
      this.host.removeChild(this.root);
    }
  }
}

/**
 * Factory creating a pooled 3D-to-2D Dory spatial overlay mounted into `host`.
 * Pre-allocates all DOM and SVG nodes once at mount (`0` per-frame element allocations).
 */
export function createDorySpatialOverlay(
  host: HTMLElement,
  options?: DorySpatialOverlayOptions,
): DorySpatialOverlayHandle {
  return new DorySpatialOverlay(host, options);
}

// Compile-time verification against frozen Step-0 contract (`src/dory/app/types.ts`)
const _SPATIAL_OVERLAY_CONTRACT_CHECK: DorySpatialOverlayModuleContract = {
  DORY_MAX_OVERLAY_CELL_NUM_NODES,
  computeDoryOverlayCellNumBudget,
  createDorySpatialOverlay,
};
void _SPATIAL_OVERLAY_CONTRACT_CHECK;
