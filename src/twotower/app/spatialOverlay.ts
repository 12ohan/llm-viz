/**
 * TwoTower Projected Editorial Drafting-Table Overlay (`src/twotower/app/spatialOverlay.ts`)
 *
 * Synchronous DOM + SVG overlay updated inside WebGL2's requestAnimationFrame pass for all 7
 * Stage TT3 `TwoTowerSpatialScene` views (`macro`, `paired_gqa`, `paired_mamba`, `timestep`,
 * `moe`, `embedding`, `output`).
 *
 * Directly reuses Nano's verified overlay geometry & placement primitives from `src/ui/overlay.ts`
 * and `src/render/camera.ts` — ZERO duplicated projection, popover placement, reveal-frontier,
 * or cell-sizing math.
 */

import type { Vec3 } from '../../model/spec';
import {
  dot3,
  projectWorldToHostCss,
  sub3,
  type CameraSnapshot,
} from '../../render/camera';
import { localToWorldOnFrame, makeSheetFrame3D } from '../../render/sheetFrame';
import type { LandmarkAnchor } from '../../scene/spine';
import {
  buildElbowLeaderPath,
  buildQuadraticPointerPath,
  computeProjectedCellSize,
  DIMMED_LABEL_OPACITY,
  formatCellOverlayValue,
  isCellHiddenByReveal,
  LABEL_NUDGE_LEADER_THRESHOLD_PX,
  MAX_OVERLAY_AXIS_HEADER_NODES,
  MAX_OVERLAY_CELL_NUM_NODES,
  MAX_OVERLAY_WIDTH_CACHE_ENTRIES,
  pickPopoverPlacement,
  prioritizeSheetCellCoords,
  shouldShowAxisHeaders,
  shouldShowCellNumbers,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
  type OverlayRect,
  type ProjectedCellSize,
} from '../../shared/overlayLayout';
import type { TwoTowerSpatialScene } from '../scene/provenance';

export interface TwoTowerOverlayUpdateInput {
  readonly scene: TwoTowerSpatialScene;
  readonly snapshot: CameraSnapshot;
  readonly selectedPartId?: string | null;
  readonly hoveredPartId?: string | null;
  readonly highlightPartId?: string | null;
  readonly activeTokens?: number;
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly popoverAvoidRects?: readonly OverlayRect[];
}

interface PlacedRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const MAX_OPERAND_DOTS = 4;

const OPERAND_SWATCH_HEX: Readonly<Record<'indigo' | 'teal' | 'red', string>> = {
  indigo: '#3861BD',
  teal: '#1A8A7A',
  red: '#DC4C34',
};

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function overlapsAny(
  rects: readonly OverlayRect[],
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): boolean {
  const pad = 3;
  for (const r of rects) {
    if (x0 < r.x1 + pad && x1 + pad > r.x0 && y0 < r.y1 + pad && y1 + pad > r.y0) {
      return true;
    }
  }
  return false;
}

function pickBorderPoint(
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
  return {
    edgeX: rect.x1,
    edgeY: clamp(targetY, rect.y0 + insetY, rect.y1 - insetY),
    orientation: 'horizontal',
  };
}

/**
 * Editorial drafting-table 3D-to-2D overlay engine for TwoTower Spatial Views.
 * Pre-allocates pooled DOM nodes for zero per-frame allocations/GC.
 */
export class TwoTowerSpatialOverlay {
  private readonly svg: SVGSVGElement;
  private readonly labelLayer: HTMLDivElement;
  private readonly caliperPath: SVGPathElement;
  private readonly leaderPath: SVGPathElement;
  private readonly popoverSourceIndigoPath: SVGPathElement;
  private readonly popoverSourceTealPath: SVGPathElement;
  private readonly popoverLeaderPath: SVGPathElement;
  private readonly popoverAnchorDot: SVGCircleElement;

  private readonly landmarkNodes = new Map<string, HTMLDivElement>();
  private readonly caliperNodes = new Map<string, HTMLDivElement>();
  private readonly kvTickNodes = new Map<number, HTMLDivElement>();
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
    this.svg = svg;
    this.labelLayer = labelLayer;
    this.svg.innerHTML = '';
    this.labelLayer.innerHTML = '';

    const scratch = document.createElement('canvas');
    this.measureCtx = typeof scratch.getContext === 'function' ? scratch.getContext('2d') : null;

    this.caliperPath = this.createSvgPath('tt-ov-caliper-line');
    this.leaderPath = this.createSvgPath('tt-ov-leader-line');
    this.popoverSourceIndigoPath = this.createSvgPath('tt-ov-pop-source tt-ov-pop-source--indigo');
    this.popoverSourceTealPath = this.createSvgPath('tt-ov-pop-source tt-ov-pop-source--teal');
    this.popoverLeaderPath = this.createSvgPath('tt-ov-pop-leader');

    this.popoverAnchorDot = document.createElementNS(SVG_NS, 'circle');
    this.popoverAnchorDot.setAttribute('class', 'tt-ov-pop-dot');
    this.popoverAnchorDot.setAttribute('r', '3');
    this.popoverAnchorDot.style.display = 'none';

    this.svg.append(
      this.caliperPath,
      this.leaderPath,
      this.popoverSourceIndigoPath,
      this.popoverSourceTealPath,
      this.popoverLeaderPath,
      this.popoverAnchorDot,
    );

    // Pre-allocate KV depth ticks (1..24 for TwoTower kvCapacity=24)
    for (let t = 1; t <= 24; t++) {
      const node = document.createElement('div');
      node.className = 'tt-ov-kv-tick';
      node.style.display = 'none';
      this.labelLayer.appendChild(node);
      this.kvTickNodes.set(t, node);
    }

    // Pre-allocate fixed pool of 64 on-cell numeric readout nodes
    for (let i = 0; i < MAX_OVERLAY_CELL_NUM_NODES; i++) {
      const node = document.createElement('div');
      node.className = 'tt-ov-cell-num';
      node.style.display = 'none';
      this.labelLayer.appendChild(node);
      this.cellNumNodes.push(node);
    }

    // Pre-allocate fixed pool of 24 exterior axis header nodes
    for (let i = 0; i < MAX_OVERLAY_AXIS_HEADER_NODES; i++) {
      const node = document.createElement('div');
      node.className = 'tt-ov-axis-header';
      node.style.display = 'none';
      this.labelLayer.appendChild(node);
      this.axisHeaderNodes.push(node);
    }

    // Single reusable floating contextual cell formula popover
    this.cellPopover = document.createElement('div');
    this.cellPopover.className = 'tt-ov-popover';
    this.cellPopover.dataset.testid = 'spatial-cell-popover';
    this.cellPopover.style.display = 'none';

    const headRow = document.createElement('div');
    headRow.className = 'tt-ov-popover-head';
    this.popoverTensorEl = document.createElement('span');
    this.popoverTensorEl.className = 'tt-ov-popover-tensor';
    this.popoverCoordEl = document.createElement('span');
    this.popoverCoordEl.className = 'tt-ov-popover-coord';
    this.popoverBadgeEl = document.createElement('span');
    this.popoverBadgeEl.className = 'tt-ov-popover-badge';
    this.popoverBadgeEl.style.display = 'none';
    headRow.append(this.popoverTensorEl, this.popoverCoordEl, this.popoverBadgeEl);

    const bodyRow = document.createElement('div');
    bodyRow.className = 'tt-ov-popover-body';
    this.popoverDotsWrap = document.createElement('span');
    this.popoverDotsWrap.className = 'tt-ov-popover-dots';
    this.popoverDotsWrap.style.display = 'none';
    for (let i = 0; i < MAX_OPERAND_DOTS; i++) {
      const dot = document.createElement('span');
      dot.className = 'tt-ov-operand-dot';
      dot.style.display = 'none';
      this.popoverDotsWrap.appendChild(dot);
      this.popoverDotEls.push(dot);
    }
    this.popoverEqEl = document.createElement('span');
    this.popoverEqEl.className = 'tt-ov-popover-eq';
    this.popoverValEl = document.createElement('span');
    this.popoverValEl.className = 'tt-ov-popover-val';
    bodyRow.append(this.popoverDotsWrap, this.popoverEqEl, this.popoverValEl);

    this.cellPopover.append(headRow, bodyRow);
    this.labelLayer.appendChild(this.cellPopover);
  }

  private createSvgPath(className: string): SVGPathElement {
    const p = document.createElementNS(SVG_NS, 'path');
    p.setAttribute('class', className);
    return p;
  }

  private getOrCreateLandmarkNode(id: string): HTMLDivElement {
    let node = this.landmarkNodes.get(id);
    if (!node) {
      node = document.createElement('div');
      node.className = 'tt-ov-pill';
      node.style.display = 'none';
      this.labelLayer.appendChild(node);
      this.landmarkNodes.set(id, node);
    }
    return node;
  }

  private getOrCreateCaliperNode(id: string): HTMLDivElement {
    let node = this.caliperNodes.get(id);
    if (!node) {
      node = document.createElement('div');
      node.className = 'tt-ov-caliper-label';
      node.style.display = 'none';
      this.labelLayer.appendChild(node);
      this.caliperNodes.set(id, node);
    }
    return node;
  }

  clearMetricCache(): void {
    this.widthCache.clear();
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
      const oldest = this.widthCache.keys().next().value;
      if (oldest !== undefined) this.widthCache.delete(oldest);
    }
    this.widthCache.set(key, w);
    return w;
  }

  private hideAll(): void {
    for (const el of this.landmarkNodes.values()) {
      el.style.display = 'none';
      el.style.opacity = '1';
      el.classList.remove('is-dimmed');
    }
    for (const el of this.caliperNodes.values()) {
      el.style.display = 'none';
    }
    for (const el of this.kvTickNodes.values()) {
      el.style.display = 'none';
    }
    for (const el of this.cellNumNodes) {
      el.style.display = 'none';
      el.classList.remove('is-active', 'is-accumulating', 'is-masked');
    }
    for (const el of this.axisHeaderNodes) {
      el.style.display = 'none';
      el.classList.remove('is-active', 'is-dim');
    }
    this.cellPopover.style.display = 'none';
    this.cellPopover.classList.remove('is-accumulating');
    this.popoverValEl.classList.remove('is-accumulating', 'is-masked');
    this.caliperPath.setAttribute('d', '');
    this.leaderPath.setAttribute('d', '');
    this.popoverLeaderPath.setAttribute('d', '');
    this.popoverSourceIndigoPath.setAttribute('d', '');
    this.popoverSourceTealPath.setAttribute('d', '');
    this.popoverAnchorDot.style.display = 'none';
  }

  update(input: TwoTowerOverlayUpdateInput): void {
    const { scene, snapshot } = input;
    const { hostWidth, hostHeight } = snapshot.metrics;
    this.svg.setAttribute('viewBox', `0 0 ${hostWidth} ${hostHeight}`);
    this.hideAll();

    const occupied: PlacedRect[] = [];
    const leaderSegs: string[] = [];
    const caliperSegs: string[] = [];

    const minY = Math.max(8, input.topInsetPx ?? 40);
    const maxY = Math.max(minY + 48, hostHeight - Math.max(12, input.bottomInsetPx ?? 64));
    const activePart =
      input.hoveredPartId ?? input.selectedPartId ?? input.highlightPartId ?? null;
    const activeOpPartIds = scene.activeOpPartIds;
    const hasActiveOpFocus = activeOpPartIds != null && activeOpPartIds.size > 0;
    const isSheetCutaway = Boolean(scene.sheetCount && scene.sheetCount > 0 && scene.viewMode !== 'macro');

    const sheetFont = '600 10.5px system-ui, -apple-system, sans-serif';
    const pillFont = '600 11px system-ui, -apple-system, sans-serif';
    const monoFont = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';

    // Estimate zoom scale in CSS pixels per world unit
    const pOrigin = projectWorldToHostCss([0, 0.05, 0], snapshot);
    const pUnit = projectWorldToHostCss([1, 0.05, 0], snapshot);
    const pxPerWorldUnit =
      pOrigin.visible && pUnit.visible ? Math.hypot(pUnit.x - pOrigin.x, pUnit.y - pOrigin.y) : 40;
    const isCloseZoom = pxPerWorldUnit >= 72;

    const isLandmarkParticipating = (lm: LandmarkAnchor): boolean => {
      if (!hasActiveOpFocus) return true;
      if (lm.partId !== undefined && lm.partId === activePart) return true;
      if (lm.partId !== undefined && activeOpPartIds!.has(lm.partId)) return true;
      if (activeOpPartIds!.has(lm.id)) return true;
      return false;
    };

    // Prioritize active part, participating op landmarks, gold/important badges, and primary tier
    const orderedLandmarks = [...scene.landmarks].sort((a, b) => {
      const aAct = a.partId !== undefined && a.partId === activePart ? 5 : 0;
      const bAct = b.partId !== undefined && b.partId === activePart ? 5 : 0;
      const aPart = hasActiveOpFocus && isLandmarkParticipating(a) ? 4 : 0;
      const bPart = hasActiveOpFocus && isLandmarkParticipating(b) ? 4 : 0;
      const aTone = a.tone === 'gold' ? 3 : a.tone === 'cyan' ? 2 : 1;
      const bTone = b.tone === 'gold' ? 3 : b.tone === 'cyan' ? 2 : 1;
      const aTier = a.tier === 'detail' ? 0 : 2;
      const bTier = b.tier === 'detail' ? 0 : 2;
      return Math.max(bAct, bPart, bTier) * 10 + bTone - (Math.max(aAct, aPart, aTier) * 10 + aTone);
    });

    for (const lm of orderedLandmarks) {
      const isAct = lm.partId !== undefined && lm.partId === activePart;
      const isParticipating = isLandmarkParticipating(lm);
      if (lm.tier === 'detail' && !isAct && !isCloseZoom && !(hasActiveOpFocus && isParticipating)) {
        continue;
      }

      const pt = projectWorldToHostCss(lm.world, snapshot);
      if (!pt.visible || pt.x < -40 || pt.x > hostWidth + 40 || pt.y < minY - 28 || pt.y > maxY + 28) {
        continue;
      }

      const isDetail = lm.tier === 'detail';
      const font = isSheetCutaway ? (isDetail ? monoFont : sheetFont) : pillFont;
      const maxLabelW = Math.max(180, Math.min(540, hostWidth - 20));
      const rawW = this.measureText(lm.label, font) + (isSheetCutaway ? (isDetail ? 16 : 18) : 22);
      const w = Math.min(rawW, maxLabelW);
      const h = isSheetCutaway ? (isDetail ? 17 : 18) : 20;
      const gap = isSheetCutaway ? 4 : 6;

      let idealLx = pt.x - w * 0.5;
      let idealLy = pt.y - h - gap;
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
          [clamp(lx - 18, padX, maxX), ly],
          [clamp(lx + 18, padX, maxX), ly],
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
        if (!placed && !isAct && !(hasActiveOpFocus && isParticipating)) {
          continue;
        }
      }

      const el = this.getOrCreateLandmarkNode(lm.id);
      el.className = isSheetCutaway
        ? isDetail
          ? 'tt-ov-sheet-label tt-ov-sheet-detail'
          : 'tt-ov-sheet-label'
        : 'tt-ov-pill';
      if (isAct) el.classList.add('is-active');
      else if (lm.tone === 'gold') el.classList.add('is-gold');
      else if (lm.tone === 'cyan') el.classList.add('is-cyan');
      else if (lm.tone === 'muted') el.classList.add('is-muted');

      el.classList.toggle('is-dimmed', !isParticipating);
      el.style.opacity = isParticipating ? '1' : DIMMED_LABEL_OPACITY;
      el.style.maxWidth = `${maxLabelW}px`;
      el.textContent = lm.label;
      el.style.display = 'inline-block';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;

      const placedRect: PlacedRect = { x0: lx, y0: ly, x1: lx + w, y1: ly + h };
      occupied.push(placedRect);

      const nearX = clamp(pt.x, placedRect.x0, placedRect.x1);
      const nearY = clamp(pt.y, placedRect.y0, placedRect.y1);
      const edgeDist = Math.hypot(nearX - pt.x, nearY - pt.y);
      const nudgeDist = Math.hypot(lx - idealLx, ly - idealLy);
      if (
        !isSheetCutaway ||
        edgeDist > LABEL_NUDGE_LEADER_THRESHOLD_PX ||
        (nudgeDist > LABEL_NUDGE_LEADER_THRESHOLD_PX && edgeDist > 3)
      ) {
        const anchorX = clamp(pt.x, 4, hostWidth - 4);
        const anchorY = clamp(pt.y, minY, maxY);
        const seg = buildElbowLeaderPath(anchorX, anchorY, placedRect);
        if (seg) leaderSegs.push(seg);
      }
    }

    // Floating contextual cell formula popover + curved provenance pointers
    const popoverRect = this.renderCellPopover(
      scene.cellPopover,
      snapshot,
      occupied,
      minY,
      maxY,
      scene.activeCellGrids,
      input.popoverAvoidRects,
    );

    // KV Depth Ticks (1..24)
    const activeTokens = input.activeTokens ?? 0;
    if (scene.kvDepthTicks.length > 0 && pxPerWorldUnit >= 28) {
      const orderedTicks = [...scene.kvDepthTicks].sort((a, b) =>
        a.token === activeTokens ? -1 : b.token === activeTokens ? 1 : a.token - b.token,
      );
      for (const tick of orderedTicks) {
        const pt = projectWorldToHostCss(tick.world, snapshot);
        if (!pt.visible || pt.x < 10 || pt.x > hostWidth - 28 || pt.y < minY || pt.y > maxY) {
          continue;
        }
        const isCur = tick.token === activeTokens;
        const text = isCur ? `P=${tick.token}` : `${tick.token}`;
        const w = this.measureText(text, monoFont) + 8;
        const h = 15;
        const lx = clamp(pt.x + 3, 8, hostWidth - w - 8);
        const ly = clamp(pt.y - h * 0.5, minY, maxY - h);
        if (!isCur && overlapsAny(occupied, lx, ly, lx + w, ly + h)) continue;
        const node = this.kvTickNodes.get(tick.token);
        if (!node) continue;
        node.textContent = text;
        node.classList.toggle('is-current', isCur);
        node.style.display = 'inline-flex';
        node.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
        occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
      }
    }

    // Dimension calipers
    if (scene.calipers && scene.calipers.length > 0) {
      for (const cal of scene.calipers) {
        const relatedActive =
          activePart !== null && Boolean(cal.partIds && cal.partIds.includes(activePart));
        if (!relatedActive && !isCloseZoom && scene.viewMode === 'macro') continue;

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

    // Zoom-adaptive on-cell numeric values & exterior axis headers
    this.renderActiveCellGrids(scene.activeCellGrids, snapshot, occupied, popoverRect, minY, maxY);

    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.caliperPath.setAttribute('d', caliperSegs.join(''));
  }

  private renderCellPopover(
    spec: CellPopoverSpec | null | undefined,
    snapshot: CameraSnapshot,
    occupied: PlacedRect[],
    minY: number,
    maxY: number,
    activeCellGrids?: readonly ActiveSheetCellGrid[] | null,
    extraAvoidRects?: readonly OverlayRect[] | null,
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
      (spec.badge ? this.measureText(spec.badge, badgeFont) + 24 : 0) +
      38;
    const dotsW = dotCount > 0 ? dotCount * 9 + 6 : 0;
    const bodyW =
      dotsW +
      this.measureText(spec.equation, eqFont) +
      this.measureText(spec.valueText, valFont) +
      52;

    const maxCardW = Math.max(180, Math.min(460, hostWidth - 24));
    const cardW = clamp(Math.max(headW, bodyW), Math.min(180, maxCardW), maxCardW);
    const cardH = 48 + (headW > cardW ? 16 : 0) + (bodyW > cardW ? 16 : 0);

    const anchorX = clamp(pt.x, 6, hostWidth - 6);
    const anchorY = clamp(pt.y, minY, maxY);

    const projectedSources: { sx: number; sy: number; role: 'indigo' | 'teal' }[] = [];
    const avoidRects: OverlayRect[] = extraAvoidRects ? [...extraAvoidRects] : [];
    const popoverOccupied: PlacedRect[] = extraAvoidRects ? [...occupied, ...extraAvoidRects] : occupied;
    let targetSheetRect: OverlayRect | null = null;
    let bestTargetDist = Infinity;

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
        if (visibleCorners < 2 || sx1 <= 0 || sx0 >= hostWidth || sy1 <= 0 || sy0 >= hostHeight) {
          continue;
        }
        const sheetRect: OverlayRect = { x0: sx0, y0: sy0, x1: sx1, y1: sy1 };
        avoidRects.push(sheetRect);

        const dVec = sub3(spec.world, frame.center);
        const uNorm = Math.abs(dot3(dVec, frame.axisU)) / Math.max(1e-4, frame.extent[0]);
        const vNorm = Math.abs(dot3(dVec, frame.axisV)) / Math.max(1e-4, frame.extent[2]);
        const nDist = Math.abs(dot3(dVec, frame.axisN));
        if (uNorm <= 0.56 && vNorm <= 0.56 && nDist <= frame.extent[1] * 0.5 + 0.18 && nDist < bestTargetDist) {
          bestTargetDist = nDist;
          targetSheetRect = sheetRect;
        }
      }
    }

    if (spec.sourceWorlds && spec.sourceWorlds.length > 0) {
      const pad = 26;
      for (const src of spec.sourceWorlds) {
        const spt = projectWorldToHostCss(src.world, snapshot);
        if (!spt.visible || spt.x < -40 || spt.x > hostWidth + 40 || spt.y < minY - 40 || spt.y > maxY + 40) {
          continue;
        }
        const sx = clamp(spt.x, 4, hostWidth - 4);
        const sy = clamp(spt.y, minY, maxY);
        projectedSources.push({ sx, sy, role: src.role });
        avoidRects.push({ x0: sx - pad, y0: sy - pad, x1: sx + pad, y1: sy + pad });
        const mx = (sx + anchorX) * 0.5;
        const my = (sy + anchorY) * 0.5;
        avoidRects.push({ x0: mx - pad, y0: my - pad, x1: mx + pad, y1: my + pad });
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
      occupied: popoverOccupied,
      avoidRects,
      targetSheetRect,
      marginX: 8,
    });

    this.cellPopover.style.width = `${placed.width.toFixed(0)}px`;
    this.cellPopover.style.display = 'flex';
    this.cellPopover.style.transform = `translate3d(${placed.x.toFixed(1)}px, ${placed.y.toFixed(1)}px, 0)`;

    const placedRect: PlacedRect = {
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
      for (const src of projectedSources) {
        const border = pickBorderPoint(placedRect, src.sx, src.sy);
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
      if (!pt.visible || pt.x < 12 || pt.x > hostWidth - 12 || pt.y < minY + 6 || pt.y > maxY - 6) {
        return false;
      }
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
        pg.grid.activeCol !== undefined && pg.grid.activeCol >= 0 && pg.grid.activeCol < pg.grid.cols;
      const hasActiveRow =
        pg.grid.activeRow !== undefined && pg.grid.activeRow >= 0 && pg.grid.activeRow < pg.grid.rows;
      const rest: (readonly [number, number])[] = [];
      for (const coord of ordered) {
        const [c, r] = coord;
        const isPriority =
          (hasActiveCol && c === pg.grid.activeCol) || (hasActiveRow && r === pg.grid.activeRow);
        if (isPriority) {
          if (cellNumCursor < MAX_OVERLAY_CELL_NUM_NODES) placeCellNum(pg, c, r);
        } else {
          rest.push(coord);
        }
      }
      remainingPerGrid.push({ pg, coords: rest });
    }

    for (const item of remainingPerGrid) {
      if (cellNumCursor >= MAX_OVERLAY_CELL_NUM_NODES) break;
      for (const [c, r] of item.coords) {
        if (cellNumCursor >= MAX_OVERLAY_CELL_NUM_NODES) break;
        placeCellNum(item.pg, c, r);
      }
    }

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
