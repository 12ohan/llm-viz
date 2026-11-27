/**
 * Pooled DOM + SVG 3D-to-2D Projected Drafting-Table Overlay (`src/uni/app/spatialOverlay.ts`)
 *
 * Pre-allocates fixed DOM/SVG pools once in the constructor (ZERO per-frame
 * `document.createElement` or `document.createElementNS` calls during `update()`!):
 *   - Anchor / landmark pills pool
 *   - Caliper label pool + pointer-events-none SVG leader/caliper stroke layer
 *   - Callout / hover badge pill for active `hoveredCell` / `selectedPartId`
 *   - Strictly bounded cell-number node pool (`poolSize <= 64`, enforced via
 *     `UNI_MAX_OVERLAY_CELL_NUM_NODES`)
 *
 * Strictly isolated from `src/ui/controls.ts` and `src/ui/overlay.ts` (D4).
 * Imports only pure projection math from `../../render/camera`.
 */

import { projectWorldToHostCss, type CameraSnapshot } from '../../render/camera';
import type { LandmarkAnchor } from '../../scene/spine';
import type { UniOverlayAnchor, UniPickIdentity, UniSpatialScene } from '../scene/types';
import type {
  UniOverlayRect,
  UniSpatialOverlayOptions,
  UniSpatialOverlayUpdateInput,
} from './types';

export const UNI_MAX_OVERLAY_CELL_NUM_NODES = 64 as const;

const PILL_POOL_CAPACITY = 64;
const CALIPER_POOL_CAPACITY = 16;
const SVG_NS = 'http://www.w3.org/2000/svg';

/**
 * Clamp requested overlay cell-number node count strictly into
 * `[0, Math.min(UNI_MAX_OVERLAY_CELL_NUM_NODES, maxNodes ?? UNI_MAX_OVERLAY_CELL_NUM_NODES)]`.
 */
export function computeOverlayCellNumBudget(
  requestedCount: number,
  maxNodes?: number,
): number {
  const cap =
    maxNodes === undefined || !Number.isFinite(maxNodes)
      ? UNI_MAX_OVERLAY_CELL_NUM_NODES
      : Math.max(0, Math.min(UNI_MAX_OVERLAY_CELL_NUM_NODES, Math.trunc(maxNodes)));
  if (!Number.isFinite(requestedCount)) return 0;
  return Math.max(0, Math.min(cap, Math.trunc(requestedCount)));
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function overlapsAny(
  rects: readonly UniOverlayRect[],
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  pad = 3,
): boolean {
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i]!;
    if (x0 < r.x1 + pad && x1 + pad > r.x0 && y0 < r.y1 + pad && y1 + pad > r.y0) {
      return true;
    }
  }
  return false;
}

function estimateTextWidth(text: string, pxPerChar = 6.4): number {
  return Math.ceil(text.length * pxPerChar);
}

function buildElbowPath(
  anchorX: number,
  anchorY: number,
  rect: UniOverlayRect,
): string {
  const cx = (rect.x0 + rect.x1) * 0.5;
  const cy = (rect.y0 + rect.y1) * 0.5;
  const edgeX = clamp(anchorX, rect.x0 + 4, rect.x1 - 4);
  const edgeY = anchorY <= cy ? rect.y0 : rect.y1;
  const midY = (anchorY + edgeY) * 0.5;
  if (Math.hypot(edgeX - anchorX, edgeY - anchorY) < 4) {
    return '';
  }
  return `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${anchorX.toFixed(1)},${midY.toFixed(1)}L${edgeX.toFixed(1)},${edgeY.toFixed(1)}M${cx.toFixed(1)},${edgeY.toFixed(1)}`;
}

function formatCalloutLines(
  hoveredCell: UniPickIdentity | null | undefined,
  selectedPartId: string | null | undefined,
): { title: string; detail: string; partId: string | null } | null {
  if (hoveredCell) {
    const parts: string[] = [];
    if (hoveredCell.layerIndex !== null) parts.push(`L${hoveredCell.layerIndex}`);
    if (hoveredCell.branch !== null) parts.push(hoveredCell.branch.toUpperCase());
    if (hoveredCell.qHeadIndex !== null) parts.push(`Q${hoveredCell.qHeadIndex}`);
    if (hoveredCell.kvHeadIndex !== null) parts.push(`KV${hoveredCell.kvHeadIndex}`);
    if (hoveredCell.cfgBatchRow !== null) parts.push(`CFG${hoveredCell.cfgBatchRow}`);
    if (hoveredCell.rotaryPlane !== null) parts.push(`P${hoveredCell.rotaryPlane}`);
    if (hoveredCell.treeDepth !== null) parts.push(`D${hoveredCell.treeDepth}`);
    if (hoveredCell.row !== null && hoveredCell.col !== null) {
      parts.push(`[${hoveredCell.row},${hoveredCell.col}]`);
    } else if (hoveredCell.tokenIndex !== null) {
      parts.push(`t${hoveredCell.tokenIndex}`);
    }
    const title = hoveredCell.tensorKey ?? hoveredCell.partId ?? hoveredCell.viewMode;
    const detail = parts.length > 0 ? parts.join(' · ') : (hoveredCell.semanticTag ?? '');
    return {
      title,
      detail,
      partId: hoveredCell.partId ?? selectedPartId ?? null,
    };
  }
  if (selectedPartId) {
    return {
      title: selectedPartId,
      detail: 'Selected',
      partId: selectedPartId,
    };
  }
  return null;
}

function findCalloutWorldPos(
  scene: UniSpatialScene,
  partId: string | null,
): readonly [number, number, number] | null {
  if (partId) {
    for (let i = 0; i < scene.landmarks.length; i++) {
      const lm = scene.landmarks[i]!;
      if (lm.partId === partId || lm.id === partId) {
        return lm.world;
      }
    }
    for (let i = 0; i < scene.pickBoxes.length; i++) {
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
  if (scene.overlayAnchors.length > 0) {
    return scene.overlayAnchors[0]!.worldPos;
  }
  return null;
}

/**
 * Pooled DOM + SVG 3D-to-2D overlay mounted inside `host`.
 * Zero per-frame element creation during `update()`.
 */
export class UniSpatialOverlay {
  private readonly host: HTMLElement;
  private readonly root: HTMLDivElement;
  private readonly svg: SVGSVGElement;
  private readonly labelLayer: HTMLDivElement;

  private readonly leaderPath: SVGPathElement;
  private readonly caliperPath: SVGPathElement;
  private readonly calloutLeaderPath: SVGPathElement;
  private readonly calloutAnchorDot: SVGCircleElement;

  private readonly pillPool: HTMLDivElement[] = [];
  private readonly caliperLabelPool: HTMLDivElement[] = [];
  private readonly cellNumPool: HTMLDivElement[] = [];

  private readonly calloutPill: HTMLDivElement;
  private readonly calloutTitleEl: HTMLSpanElement;
  private readonly calloutDetailEl: HTMLSpanElement;

  private activeCellNumNodeCount = 0;
  private disposed = false;

  constructor(host: HTMLElement, options?: UniSpatialOverlayOptions) {
    this.host = host;

    this.root = document.createElement('div');
    this.root.className = 'uni-ov-root';
    this.root.dataset.testid = 'uni-spatial-overlay';
    this.root.style.pointerEvents = 'none';

    this.svg = document.createElementNS(SVG_NS, 'svg');
    this.svg.setAttribute('class', 'uni-ov-svg');
    this.svg.setAttribute('aria-hidden', 'true');
    this.svg.style.pointerEvents = 'none';

    this.leaderPath = document.createElementNS(SVG_NS, 'path');
    this.leaderPath.setAttribute('class', 'uni-ov-leader-line');

    this.caliperPath = document.createElementNS(SVG_NS, 'path');
    this.caliperPath.setAttribute('class', 'uni-ov-caliper-line');

    this.calloutLeaderPath = document.createElementNS(SVG_NS, 'path');
    this.calloutLeaderPath.setAttribute('class', 'uni-ov-callout-leader');

    this.calloutAnchorDot = document.createElementNS(SVG_NS, 'circle');
    this.calloutAnchorDot.setAttribute('class', 'uni-ov-callout-dot');
    this.calloutAnchorDot.setAttribute('r', '3.2');
    this.calloutAnchorDot.style.display = 'none';

    this.svg.append(
      this.leaderPath,
      this.caliperPath,
      this.calloutLeaderPath,
      this.calloutAnchorDot,
    );

    this.labelLayer = document.createElement('div');
    this.labelLayer.className = 'uni-ov-labels';
    this.labelLayer.style.pointerEvents = 'none';

    for (let i = 0; i < PILL_POOL_CAPACITY; i++) {
      const pill = document.createElement('div');
      pill.className = 'uni-ov-pill';
      pill.style.display = 'none';
      this.labelLayer.appendChild(pill);
      this.pillPool.push(pill);
    }

    for (let i = 0; i < CALIPER_POOL_CAPACITY; i++) {
      const cal = document.createElement('div');
      cal.className = 'uni-ov-caliper-label';
      cal.style.display = 'none';
      this.labelLayer.appendChild(cal);
      this.caliperLabelPool.push(cal);
    }

    const cellNumCapacity = computeOverlayCellNumBudget(
      UNI_MAX_OVERLAY_CELL_NUM_NODES,
      options?.maxCellNumNodes,
    );
    for (let i = 0; i < cellNumCapacity; i++) {
      const cellEl = document.createElement('div');
      cellEl.className = 'uni-ov-cell-num';
      cellEl.style.display = 'none';
      this.labelLayer.appendChild(cellEl);
      this.cellNumPool.push(cellEl);
    }

    this.calloutPill = document.createElement('div');
    this.calloutPill.className = 'uni-ov-callout';
    this.calloutPill.dataset.testid = 'uni-ov-callout';
    this.calloutPill.style.display = 'none';

    this.calloutTitleEl = document.createElement('span');
    this.calloutTitleEl.className = 'uni-ov-callout-title';
    this.calloutDetailEl = document.createElement('span');
    this.calloutDetailEl.className = 'uni-ov-callout-detail';
    this.calloutPill.append(this.calloutTitleEl, this.calloutDetailEl);
    this.labelLayer.appendChild(this.calloutPill);

    this.root.append(this.svg, this.labelLayer);
    this.host.appendChild(this.root);
  }

  getActiveCellNumNodeCount(): number {
    return this.activeCellNumNodeCount;
  }

  getCellNumPoolCapacity(): number {
    return this.cellNumPool.length;
  }

  hideAll(): void {
    this.activeCellNumNodeCount = 0;

    for (let i = 0; i < this.pillPool.length; i++) {
      const el = this.pillPool[i]!;
      el.style.display = 'none';
      el.className = 'uni-ov-pill';
    }
    for (let i = 0; i < this.caliperLabelPool.length; i++) {
      const el = this.caliperLabelPool[i]!;
      el.style.display = 'none';
    }
    for (let i = 0; i < this.cellNumPool.length; i++) {
      const el = this.cellNumPool[i]!;
      el.style.display = 'none';
      el.className = 'uni-ov-cell-num';
    }

    this.calloutPill.style.display = 'none';
    this.leaderPath.setAttribute('d', '');
    this.caliperPath.setAttribute('d', '');
    this.calloutLeaderPath.setAttribute('d', '');
    this.calloutAnchorDot.style.display = 'none';
  }

  update(input: UniSpatialOverlayUpdateInput): void {
    if (this.disposed) return;
    this.hideAll();

    const { scene, snapshot } = input;
    const { hostWidth, hostHeight } = snapshot.metrics;
    this.svg.setAttribute('viewBox', `0 0 ${hostWidth} ${hostHeight}`);

    const minY = Math.max(8, input.topInsetPx ?? 12);
    const maxY = Math.max(minY + 48, hostHeight - Math.max(10, input.bottomInsetPx ?? 16));
    const occupied: UniOverlayRect[] = [];
    if (input.avoidRects) {
      for (let i = 0; i < input.avoidRects.length; i++) {
        occupied.push(input.avoidRects[i]!);
      }
    }

    const leaderSegs: string[] = [];
    const caliperSegs: string[] = [];
    const activePart = input.hoveredCell?.partId ?? input.selectedPartId ?? null;

    let pillCursor = 0;

    const placePill = (
      label: string,
      sublabel: string | undefined,
      worldPos: readonly [number, number, number],
      toneClass: string,
      align: LandmarkAnchor['align'] | undefined,
      isActive: boolean,
      badge?: string,
    ): void => {
      if (pillCursor >= this.pillPool.length) return;
      const pt = projectWorldToHostCss([worldPos[0], worldPos[1], worldPos[2]], snapshot);
      if (!pt.visible || pt.x < -30 || pt.x > hostWidth + 30 || pt.y < minY - 24 || pt.y > maxY + 24) {
        return;
      }

      const display = badge ? `${label} · ${badge}` : label;
      const w = Math.min(Math.max(120, hostWidth - 24), estimateTextWidth(display, 6.4) + 18);
      const h = 20;
      const gap = 6;

      let idealX = pt.x - w * 0.5;
      let idealY = pt.y - h - gap;
      if (align === 'left') {
        idealX = pt.x - w - gap;
        idealY = pt.y - h * 0.5;
      } else if (align === 'right') {
        idealX = pt.x + gap;
        idealY = pt.y - h * 0.5;
      } else if (align === 'bottom') {
        idealX = pt.x - w * 0.5;
        idealY = pt.y + gap;
      }

      const maxX = Math.max(8, hostWidth - w - 8);
      const maxTopY = Math.max(minY, maxY - h);
      let lx = clamp(idealX, 8, maxX);
      let ly = clamp(idealY, minY, maxTopY);

      if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
        const candidates: readonly (readonly [number, number])[] = [
          [lx, clamp(ly - (h + 4), minY, maxTopY)],
          [lx, clamp(ly + (h + 4), minY, maxTopY)],
          [clamp(lx - 20, 8, maxX), ly],
          [clamp(lx + 20, 8, maxX), ly],
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
        if (!placed && !isActive) return;
      }

      const el = this.pillPool[pillCursor++]!;
      el.className = `uni-ov-pill ${toneClass}${isActive ? ' is-active' : ''}`;
      el.textContent = display;
      el.title = sublabel ? `${display} — ${sublabel}` : display;
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;

      const rect: UniOverlayRect = { x0: lx, y0: ly, x1: lx + w, y1: ly + h };
      occupied.push(rect);

      const seg = buildElbowPath(clamp(pt.x, 4, hostWidth - 4), clamp(pt.y, minY, maxY), rect);
      if (seg) leaderSegs.push(seg);
    };

    // 1. Project scene landmarks (primary first)
    const landmarks = [...scene.landmarks].sort((a, b) => {
      const aAct = a.partId && a.partId === activePart ? 2 : a.tier === 'detail' ? 0 : 1;
      const bAct = b.partId && b.partId === activePart ? 2 : b.tier === 'detail' ? 0 : 1;
      return bAct - aAct;
    });
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i]!;
      const isAct = Boolean(lm.partId && lm.partId === activePart);
      const toneClass = lm.tone ? `uni-ov-tone--${lm.tone}` : 'uni-ov-tone--default';
      placePill(lm.label, lm.sublabel, lm.world, toneClass, lm.align, isAct);
    }

    // 2. Project scene overlayAnchors
    for (let i = 0; i < scene.overlayAnchors.length; i++) {
      const anchor: UniOverlayAnchor = scene.overlayAnchors[i]!;
      const isAct = anchor.id === activePart;
      const toneClass = anchor.tone ? `uni-ov-tone--${anchor.tone}` : 'uni-ov-tone--default';
      placePill(
        anchor.label,
        anchor.sublabel,
        anchor.worldPos,
        toneClass,
        'top',
        isAct,
        anchor.badge,
      );
    }

    // 3. Project callout / hover badge pill
    const callout = formatCalloutLines(input.hoveredCell, input.selectedPartId);
    if (callout) {
      const worldTarget = findCalloutWorldPos(scene, callout.partId);
      const pt = worldTarget
        ? projectWorldToHostCss([worldTarget[0], worldTarget[1], worldTarget[2]], snapshot)
        : null;
      const titleW = estimateTextWidth(callout.title, 6.8);
      const detailW = estimateTextWidth(callout.detail, 6.2);
      const cardW = clamp(Math.max(titleW, detailW) + 24, 140, Math.max(160, hostWidth - 24));
      const cardH = callout.detail ? 38 : 24;

      const anchorX = pt && pt.visible ? clamp(pt.x, 12, hostWidth - 12) : hostWidth - 24;
      const anchorY = pt && pt.visible ? clamp(pt.y, minY, maxY) : minY + 12;

      const cx = clamp(anchorX + 14, 10, Math.max(10, hostWidth - cardW - 10));
      const cy = clamp(anchorY - cardH - 12, minY, Math.max(minY, maxY - cardH));

      this.calloutTitleEl.textContent = callout.title;
      this.calloutDetailEl.textContent = callout.detail;
      this.calloutDetailEl.style.display = callout.detail ? 'inline' : 'none';
      this.calloutPill.style.display = 'flex';
      this.calloutPill.style.transform = `translate3d(${cx.toFixed(1)}px, ${cy.toFixed(1)}px, 0)`;

      const calloutRect: UniOverlayRect = { x0: cx, y0: cy, x1: cx + cardW, y1: cy + cardH };
      occupied.push(calloutRect);

      if (pt && pt.visible) {
        const seg = buildElbowPath(anchorX, anchorY, calloutRect);
        if (seg) this.calloutLeaderPath.setAttribute('d', seg);
        this.calloutAnchorDot.setAttribute('cx', anchorX.toFixed(1));
        this.calloutAnchorDot.setAttribute('cy', anchorY.toFixed(1));
        this.calloutAnchorDot.style.display = 'inline';
      }
    }

    // 4. Project dimension calipers
    let caliperCursor = 0;
    for (let i = 0; i < scene.calipers.length; i++) {
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
        const w = estimateTextWidth(cal.label, 6.0) + 12;
        const h = 16;
        const mx = clamp((ax + bx) * 0.5 - w * 0.5, 8, Math.max(8, hostWidth - w - 8));
        const my = clamp((ay + by) * 0.5 + 4, minY, Math.max(minY, maxY - h));
        if (!overlapsAny(occupied, mx, my, mx + w, my + h)) {
          const calEl = this.caliperLabelPool[caliperCursor++]!;
          calEl.textContent = cal.label;
          calEl.style.display = 'inline-flex';
          calEl.style.transform = `translate3d(${mx.toFixed(1)}px, ${my.toFixed(1)}px, 0)`;
          occupied.push({ x0: mx, y0: my, x1: mx + w, y1: my + h });
        }
      }
    }

    // 5. Project bounded cell-number / KV-tick readout nodes (strictly <= 64)
    const maxBudget = computeOverlayCellNumBudget(
      scene.kvDepthTicks.length,
      this.cellNumPool.length,
    );
    let cellNumCursor = 0;
    for (let i = 0; i < scene.kvDepthTicks.length && cellNumCursor < maxBudget; i++) {
      const tick = scene.kvDepthTicks[i]!;
      const pt = projectWorldToHostCss(tick.world, snapshot);
      if (!pt.visible || pt.x < 8 || pt.x > hostWidth - 20 || pt.y < minY || pt.y > maxY) {
        continue;
      }
      const isBoundary = tick.token === 16;
      const text = isBoundary ? 'p=16·L4' : `${tick.token}`;
      const w = estimateTextWidth(text, 6.0) + 8;
      const h = 14;
      const lx = clamp(pt.x + 3, 8, Math.max(8, hostWidth - w - 8));
      const ly = clamp(pt.y - h * 0.5, minY, Math.max(minY, maxY - h));
      if (!isBoundary && overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
        continue;
      }
      const cellEl = this.cellNumPool[cellNumCursor++]!;
      cellEl.textContent = text;
      cellEl.className = isBoundary ? 'uni-ov-cell-num is-boundary' : 'uni-ov-cell-num';
      cellEl.style.display = 'inline-flex';
      cellEl.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
    }
    this.activeCellNumNodeCount = cellNumCursor;

    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.caliperPath.setAttribute('d', caliperSegs.join(''));
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.hideAll();
    if (this.root.parentNode === this.host) {
      this.host.removeChild(this.root);
    }
  }
}

export type { CameraSnapshot };
