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
import type { LandmarkAnchor } from '../../render/primitives';
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

function estimateTextWidth(text: string, pxPerChar = 6.0): number {
  return Math.ceil(text.length * pxPerChar);
}

function tokenizeOverlayLabel(text: string): readonly string[] {
  const cleaned = text.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  if (!cleaned) return [];
  return cleaned.split(/\s+/).filter((tok) => tok.length >= 2);
}

function areOverlayLabelsRedundant(a: string, b: string): boolean {
  const normA = a.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const normB = b.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  if (!normA || !normB) return false;
  if (normA === normB) return true;
  if (normA.length >= 14 && normB.includes(normA)) return true;
  if (normB.length >= 14 && normA.includes(normB)) return true;

  const toksA = tokenizeOverlayLabel(normA);
  const toksB = tokenizeOverlayLabel(normB);
  const minLen = Math.min(toksA.length, toksB.length);
  if (minLen < 3) return false;
  const setB = new Set(toksB);
  let shared = 0;
  for (let i = 0; i < toksA.length; i++) {
    if (setB.has(toksA[i]!)) shared += 1;
  }
  return shared >= 3 && shared / minLen >= 0.7;
}

function rectOverlapArea(
  rects: readonly UniOverlayRect[],
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  pad = 3,
): number {
  let area = 0;
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i]!;
    const ox = Math.min(x1 + pad, r.x1 + pad) - Math.max(x0 - pad, r.x0 - pad);
    const oy = Math.min(y1 + pad, r.y1 + pad) - Math.max(y0 - pad, r.y0 - pad);
    if (ox > 0 && oy > 0) {
      area += ox * oy;
    }
  }
  return area;
}

function pointInsideRects(
  rects: readonly UniOverlayRect[],
  x: number,
  y: number,
  pad = 4,
): boolean {
  for (let i = 0; i < rects.length; i++) {
    const r = rects[i]!;
    if (x >= r.x0 - pad && x <= r.x1 + pad && y >= r.y0 - pad && y <= r.y1 + pad) {
      return true;
    }
  }
  return false;
}

function buildElbowPath(
  anchorX: number,
  anchorY: number,
  rect: UniOverlayRect,
): string {
  const cy = (rect.y0 + rect.y1) * 0.5;
  // Side-attached leader when pill sits cleanly left or right of its 3D anchor
  if (rect.x1 + 4 < anchorX) {
    const midX = rect.x1 + Math.min(18, (anchorX - rect.x1) * 0.45);
    if (Math.hypot(rect.x1 - anchorX, cy - anchorY) < 4) return '';
    return `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${midX.toFixed(1)},${cy.toFixed(1)}L${rect.x1.toFixed(1)},${cy.toFixed(1)}`;
  }
  if (rect.x0 - 4 > anchorX) {
    const midX = rect.x0 - Math.min(18, (rect.x0 - anchorX) * 0.45);
    if (Math.hypot(rect.x0 - anchorX, cy - anchorY) < 4) return '';
    return `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${midX.toFixed(1)},${cy.toFixed(1)}L${rect.x0.toFixed(1)},${cy.toFixed(1)}`;
  }
  const edgeX = clamp(anchorX, rect.x0 + 4, rect.x1 - 4);
  const edgeY = anchorY <= cy ? rect.y0 : rect.y1;
  const midY = (anchorY + edgeY) * 0.5;
  if (Math.hypot(edgeX - anchorX, edgeY - anchorY) < 4) {
    return '';
  }
  return `M${anchorX.toFixed(1)},${anchorY.toFixed(1)}L${anchorX.toFixed(1)},${midY.toFixed(1)}L${edgeX.toFixed(1)},${edgeY.toFixed(1)}`;
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
    const hudAvoidRects: UniOverlayRect[] = [];
    const occupied: UniOverlayRect[] = [];
    if (input.avoidRects) {
      for (let i = 0; i < input.avoidRects.length; i++) {
        const r = input.avoidRects[i]!;
        hudAvoidRects.push(r);
        occupied.push(r);
      }
    }

    const leaderSegs: string[] = [];
    const caliperSegs: string[] = [];
    const activePart = input.hoveredCell?.partId ?? input.selectedPartId ?? null;

    // Pre-scan visible anchor horizontal span so side-flanked labels clear central 3D RG32F plates
    let projMinX = hostWidth;
    let projMaxX = 0;
    let projCount = 0;
    for (let i = 0; i < scene.landmarks.length; i++) {
      const p = projectWorldToHostCss(scene.landmarks[i]!.world, snapshot);
      if (p.visible && p.x >= 0 && p.x <= hostWidth && p.y >= minY - 24 && p.y <= maxY + 24) {
        if (p.x < projMinX) projMinX = p.x;
        if (p.x > projMaxX) projMaxX = p.x;
        projCount += 1;
      }
    }
    for (let i = 0; i < scene.overlayAnchors.length; i++) {
      const w = scene.overlayAnchors[i]!.worldPos;
      const p = projectWorldToHostCss([w[0], w[1], w[2]], snapshot);
      if (p.visible && p.x >= 0 && p.x <= hostWidth && p.y >= minY - 24 && p.y <= maxY + 24) {
        if (p.x < projMinX) projMinX = p.x;
        if (p.x > projMaxX) projMaxX = p.x;
        projCount += 1;
      }
    }
    const sceneMidX = projCount > 0 ? (projMinX + projMaxX) * 0.5 : hostWidth * 0.5;
    const plateHalfClearance = projCount > 0 ? clamp((projMaxX - projMinX) * 0.24, 34, 96) : 52;

    const placedLabels: { readonly label: string; readonly ax: number; readonly ay: number }[] = [];
    let pillCursor = 0;
    let sideToggle = 0;

    const placePill = (
      label: string,
      sublabel: string | undefined,
      worldPos: readonly [number, number, number],
      toneClass: string,
      align: LandmarkAnchor['align'] | undefined,
      isActive: boolean,
      badge?: string,
      preferSideFlank = false,
    ): void => {
      if (pillCursor >= this.pillPool.length) return;
      const pt = projectWorldToHostCss([worldPos[0], worldPos[1], worldPos[2]], snapshot);
      if (!pt.visible || pt.x < -30 || pt.x > hostWidth + 30 || pt.y < minY - 24 || pt.y > maxY + 24) {
        return;
      }

      const display = badge ? `${label} · ${badge}` : label;

      // Deduplicate near-identical overlapping pill labels (e.g., landmark + overlayAnchor on same plate)
      if (!isActive) {
        for (let i = 0; i < placedLabels.length; i++) {
          const prev = placedLabels[i]!;
          if (
            Math.hypot(pt.x - prev.ax, pt.y - prev.ay) <= 150 &&
            areOverlayLabelsRedundant(display, prev.label)
          ) {
            return;
          }
        }
      }

      const w = Math.min(Math.max(110, hostWidth - 24), estimateTextWidth(display, 6.0) + 14);
      const h = 18;
      const gap = 10;

      const maxX = Math.max(8, hostWidth - w - 8);
      const maxTopY = Math.max(minY, maxY - h);

      // Determine side preference for flanking away from central 3D tensor plates
      let flankLeft = pt.x < sceneMidX - 10;
      if (Math.abs(pt.x - sceneMidX) <= 10) {
        flankLeft = sideToggle % 2 === 0;
        sideToggle += 1;
      }

      const leftFlankX = clamp(
        Math.min(pt.x - w - gap - 10, projMinX - w - 12),
        8,
        maxX,
      );
      const rightFlankX = clamp(
        Math.max(pt.x + gap + 10, projMaxX + 12),
        8,
        maxX,
      );
      const nearLeftX = clamp(pt.x - w - plateHalfClearance, 8, maxX);
      const nearRightX = clamp(pt.x + plateHalfClearance, 8, maxX);

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

      let lx = clamp(idealX, 8, maxX);
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
          [lx, clamp(ly - (h + 5), minY, maxTopY)],
          [lx, clamp(ly + (h + 5), minY, maxTopY)],
          [clamp(pt.x - w * 0.5, 8, maxX), clamp(pt.y - h - gap - 4, minY, maxTopY)],
          [clamp(pt.x - w * 0.5, 8, maxX), clamp(pt.y + gap + 4, minY, maxTopY)],
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
      placedLabels.push({ label: display, ax: pt.x, ay: pt.y });

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

    // 2. Project scene overlayAnchors (side-flanked so central RG32F heatmaps stay clearly readable)
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
        true,
      );
    }

    // 3. Project callout / hover badge pill (strict HUD + pill collision avoidance)
    const callout = formatCalloutLines(input.hoveredCell, input.selectedPartId);
    if (callout) {
      const isHoverActive = Boolean(input.hoveredCell);
      const worldTarget = findCalloutWorldPos(scene, callout.partId);
      const pt = worldTarget
        ? projectWorldToHostCss([worldTarget[0], worldTarget[1], worldTarget[2]], snapshot)
        : null;
      const titleW = estimateTextWidth(callout.title, 6.6);
      const detailW = estimateTextWidth(callout.detail, 6.0);
      const cardW = clamp(Math.max(titleW, detailW) + 22, 132, Math.max(160, hostWidth - 24));
      const cardH = callout.detail ? 36 : 22;

      const anchorX = pt && pt.visible ? clamp(pt.x, 12, hostWidth - 12) : hostWidth - 24;
      const anchorY = pt && pt.visible ? clamp(pt.y, minY, maxY) : minY + 12;

      const maxCardX = Math.max(10, hostWidth - cardW - 10);
      const maxCardY = Math.max(minY, maxY - cardH);

      // Suppress static "Selected" callout if its anchor itself is occluded under a HUD panel (e.g. Tour HUD)
      const anchorHiddenByHud =
        !isHoverActive && pt && pt.visible && pointInsideRects(hudAvoidRects, anchorX, anchorY, 4);

      if (!anchorHiddenByHud) {
        const candidates: readonly (readonly [number, number])[] = [
          // Left of anchor
          [clamp(anchorX - cardW - 14, 10, maxCardX), clamp(anchorY - cardH * 0.5, minY, maxCardY)],
          [clamp(anchorX - cardW - 14, 10, maxCardX), clamp(anchorY - cardH - 10, minY, maxCardY)],
          // Below anchor
          [clamp(anchorX - cardW * 0.5, 10, maxCardX), clamp(anchorY + 14, minY, maxCardY)],
          [clamp(anchorX - cardW - 10, 10, maxCardX), clamp(anchorY + 14, minY, maxCardY)],
          // Above anchor
          [clamp(anchorX - cardW * 0.5, 10, maxCardX), clamp(anchorY - cardH - 14, minY, maxCardY)],
          // Right of anchor
          [clamp(anchorX + 14, 10, maxCardX), clamp(anchorY - cardH - 12, minY, maxCardY)],
          [clamp(anchorX + 14, 10, maxCardX), clamp(anchorY - cardH * 0.5, minY, maxCardY)],
          [clamp(anchorX + 14, 10, maxCardX), clamp(anchorY + 14, minY, maxCardY)],
        ];

        let chosen: readonly [number, number] | null = null;
        // Pass 1: zero overlap with ALL occupied rects (HUD panels + placed pills)
        for (const [tx, ty] of candidates) {
          if (!overlapsAny(occupied, tx, ty, tx + cardW, ty + cardH, 4)) {
            chosen = [tx, ty];
            break;
          }
        }
        // Pass 2: zero overlap with HUD avoidRects (acceptable to touch a pill only when user hovers, or fallback)
        if (!chosen && isHoverActive) {
          for (const [tx, ty] of candidates) {
            if (!overlapsAny(hudAvoidRects, tx, ty, tx + cardW, ty + cardH, 4)) {
              chosen = [tx, ty];
              break;
            }
          }
        }
        // Pass 3: live hover always renders at minimum HUD-overlap spot; static "Selected" suppresses cleanly
        if (!chosen && isHoverActive) {
          let bestArea = Infinity;
          for (const [tx, ty] of candidates) {
            const area = rectOverlapArea(hudAvoidRects, tx, ty, tx + cardW, ty + cardH, 4);
            if (area < bestArea) {
              bestArea = area;
              chosen = [tx, ty];
            }
          }
        }

        if (chosen) {
          const [cx, cy] = chosen;
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
