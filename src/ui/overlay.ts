import {
  bayKindLabel,
  formatBayId,
  formatBracketCaption,
  STATIC_MACRO_BAYS,
  type NemotronSpec,
} from '../model/spec';
import {
  projectWorldToHostCss,
  projectWorldToInnerCss,
  type CameraSnapshot,
} from '../render/camera';
import type { MacroSceneGeometry } from '../scene/spine';

const SVG_NS = 'http://www.w3.org/2000/svg';
const STATIC_BAY_SET = new Set<number>(STATIC_MACRO_BAYS);

const BAY_BOX_W = 26;
const BAY_BOX_H = 16;
const BAY_HALF_W = BAY_BOX_W * 0.5;
const BAY_CLEARANCE = 4;
const BAY_MARGIN_X = 15;
const TOP_ROW_OFFSETS = [-9, -27, -45] as const; // relative to inner viewport top (§4)

const BRACKET_ROW_SPACING = 15;
const BRACKET_MAX_ROWS = 4;
const BRACKET_CLEARANCE = 4;

interface OccupiedBox {
  readonly cx: number;
  readonly halfW: number;
}

interface PlacedLabel {
  readonly cx: number;
  readonly cy: number;
  readonly halfW: number;
  readonly row: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function rowFitsAt(occupied: readonly OccupiedBox[], cx: number, halfW: number, clearance: number): boolean {
  for (const box of occupied) {
    if (Math.abs(cx - box.cx) < halfW + box.halfW + clearance) {
      return false;
    }
  }
  return true;
}

/**
 * Find the nearest valid center X in [minCx, maxCx] that avoids all occupied boxes in a row.
 * Ties break to the left (§4).
 */
function findNearestFreeX(
  occupied: readonly OccupiedBox[],
  targetCx: number,
  halfW: number,
  clearance: number,
  minCx: number,
  maxCx: number,
): number | null {
  if (minCx > maxCx) return null;

  // Build blocked intervals for cx
  const blocked: [number, number][] = occupied
    .map((b): [number, number] => [b.cx - b.halfW - halfW - clearance, b.cx + b.halfW + halfW + clearance])
    .sort((a, b) => a[0] - b[0]);

  // Merge overlapping blocked intervals
  const merged: [number, number][] = [];
  for (const seg of blocked) {
    if (merged.length === 0 || seg[0] > merged[merged.length - 1][1]) {
      merged.push([seg[0], seg[1]]);
    } else if (seg[1] > merged[merged.length - 1][1]) {
      merged[merged.length - 1][1] = seg[1];
    }
  }

  // Collect free intervals within [minCx, maxCx]
  const free: [number, number][] = [];
  let cursor = minCx;
  for (const [b0, b1] of merged) {
    if (b1 <= minCx) continue;
    if (b0 >= maxCx) break;
    if (b0 > cursor) {
      free.push([cursor, Math.min(b0, maxCx)]);
    }
    cursor = Math.max(cursor, b1);
  }
  if (cursor <= maxCx) {
    free.push([cursor, maxCx]);
  }

  let bestX: number | null = null;
  let bestDist = Infinity;
  for (const [f0, f1] of free) {
    if (f0 > f1) continue;
    const cand = clamp(targetCx, f0, f1);
    const dist = Math.abs(cand - targetCx);
    if (dist < bestDist - 1e-6 || (Math.abs(dist - bestDist) <= 1e-6 && bestX !== null && cand < bestX)) {
      bestDist = dist;
      bestX = cand;
    }
  }
  return bestX;
}

function placeInRows(
  rows: OccupiedBox[][],
  rowYs: readonly number[],
  naturalCx: number,
  halfW: number,
  clearance: number,
  minCx: number,
  maxCx: number,
  allowShift: boolean,
): PlacedLabel | null {
  const clampedCx = clamp(naturalCx, minCx, maxCx);

  // First try natural X in each row nearest drawing first
  for (let r = 0; r < rows.length; r++) {
    if (rowFitsAt(rows[r], clampedCx, halfW, clearance)) {
      rows[r].push({ cx: clampedCx, halfW });
      return { cx: clampedCx, cy: rowYs[r], halfW, row: r };
    }
  }

  if (!allowShift) return null;

  // All rows collided at natural X: choose first row with a free interval, shift least distance (ties left)
  for (let r = 0; r < rows.length; r++) {
    const shifted = findNearestFreeX(rows[r], clampedCx, halfW, clearance, minCx, maxCx);
    if (shifted !== null) {
      rows[r].push({ cx: shifted, halfW });
      return { cx: shifted, cy: rowYs[r], halfW, row: r };
    }
  }

  // Emergency fallback if viewport is extraordinarily narrow: place in top row clamped
  const fallbackRow = rows.length - 1;
  rows[fallbackRow].push({ cx: clampedCx, halfW });
  return { cx: clampedCx, cy: rowYs[fallbackRow], halfW, row: fallbackRow };
}

/** Clip 2D line segment (x0,y0)-(x1,y1) horizontally to [minX, maxX]. */
function clipSegmentX(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  minX: number,
  maxX: number,
): readonly [number, number, number, number] | null {
  let ax = x0, ay = y0, bx = x1, by = y1;
  if (ax > bx) {
    [ax, bx] = [bx, ax];
    [ay, by] = [by, ay];
  }
  if (bx < minX || ax > maxX) return null;
  const dx = bx - ax;
  if (Math.abs(dx) < 1e-6) {
    return [ax, ay, bx, by];
  }
  let t0 = 0;
  let t1 = 1;
  if (ax < minX) t0 = (minX - ax) / dx;
  if (bx > maxX) t1 = (maxX - ax) / dx;
  return [
    ax + dx * t0,
    ay + (by - ay) * t0,
    ax + dx * t1,
    ay + (by - ay) * t1,
  ];
}

export interface OverlayUpdateInput {
  readonly spec: NemotronSpec;
  readonly scene: MacroSceneGeometry;
  readonly snapshot: CameraSnapshot;
  readonly tokens: number;
  readonly selectedBay: number | null;
  readonly hoveredBay: number | null;
}

/**
 * Synchronous DOM/SVG annotation renderer updated directly inside WebGL's draw pass (§4).
 * Zero React state updates, zero CSS transitions, cached text metrics.
 */
export class MacroOverlay {
  private readonly svg: SVGSVGElement;
  private readonly labelLayer: HTMLDivElement;
  private readonly leaderPath: SVGPathElement;
  private readonly activeLeaderPath: SVGPathElement;
  private readonly bracketPath: SVGPathElement;
  private readonly bracketLeaderPath: SVGPathElement;
  private readonly measureCtx: CanvasRenderingContext2D | null;
  private readonly textWidthCache = new Map<string, number>();

  // Pooled DOM nodes to avoid per-frame allocations during pan/zoom
  private readonly bayNodes = new Map<number, HTMLDivElement>();
  private readonly bracketNodes: HTMLDivElement[] = [];
  private readonly rulerNodes = new Map<number, HTMLDivElement>();
  private readonly bookendNodes = new Map<string, HTMLDivElement>();
  private readonly hoverTag: HTMLDivElement;
  private readonly rulerTitle: HTMLDivElement;

  constructor(svg: SVGSVGElement, labelLayer: HTMLDivElement) {
    this.svg = svg;
    this.labelLayer = labelLayer;

    const measureCanvas = document.createElement('canvas');
    this.measureCtx = measureCanvas.getContext('2d');

    this.bracketPath = this.createSvgPath('overlay-bracket-path');
    this.bracketLeaderPath = this.createSvgPath('overlay-bracket-leader-path');
    this.leaderPath = this.createSvgPath('overlay-leader-path');
    this.activeLeaderPath = this.createSvgPath('overlay-leader-active-path');

    this.svg.append(
      this.bracketPath,
      this.bracketLeaderPath,
      this.leaderPath,
      this.activeLeaderPath,
    );

    // Pre-create 52 bay label elements (hidden when unused)
    for (let id = 1; id <= 52; id++) {
      const el = document.createElement('div');
      el.className = STATIC_BAY_SET.has(id)
        ? 'ov-bay-label ov-bay-mandatory'
        : 'ov-bay-label ov-bay-extra';
      el.textContent = formatBayId(id);
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.bayNodes.set(id, el);
    }

    // Pre-create 4 Figure-2 bracket caption elements
    for (let i = 0; i < 4; i++) {
      const el = document.createElement('div');
      el.className = 'ov-bracket-label';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.bracketNodes.push(el);
    }

    // Pre-create ruler tick labels (1..16)
    for (let t = 1; t <= 16; t++) {
      const el = document.createElement('div');
      el.className = 'ov-ruler-label';
      el.textContent = String(t);
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.rulerNodes.set(t, el);
    }

    this.rulerTitle = document.createElement('div');
    this.rulerTitle.className = 'ov-ruler-title';
    this.rulerTitle.textContent = 'tokens';
    this.rulerTitle.style.display = 'none';
    this.labelLayer.appendChild(this.rulerTitle);

    for (const id of ['embed', 'output']) {
      const el = document.createElement('div');
      el.className = 'ov-bookend-label';
      el.textContent = id === 'embed' ? 'Embed' : 'Output';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.bookendNodes.set(id, el);
    }

    this.hoverTag = document.createElement('div');
    this.hoverTag.className = 'ov-hover-tag';
    this.hoverTag.style.display = 'none';
    this.labelLayer.appendChild(this.hoverTag);
  }

  private createSvgPath(className: string): SVGPathElement {
    const p = document.createElementNS(SVG_NS, 'path');
    p.setAttribute('class', className);
    return p;
  }

  clearMetricCache(): void {
    this.textWidthCache.clear();
  }

  private measureMonospaceWidth(text: string, fontSizePx: number): number {
    const key = `${fontSizePx}:${text}`;
    const cached = this.textWidthCache.get(key);
    if (cached !== undefined) return cached;

    let width: number;
    if (this.measureCtx) {
      this.measureCtx.font = `500 ${fontSizePx}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace`;
      width = Math.ceil(this.measureCtx.measureText(text).width);
    } else {
      width = Math.ceil(text.length * fontSizePx * 0.62);
    }
    this.textWidthCache.set(key, width);
    return width;
  }

  /** Project all anchors, solve collision rules, and batch SVG + DOM writes synchronously. */
  update(input: OverlayUpdateInput): void {
    const { spec, scene, snapshot, tokens, selectedBay, hoveredBay } = input;
    const { hostWidth, hostHeight, insetTop, innerWidth, innerHeight } = snapshot.metrics;
    const zoomed = snapshot.pose.zoom > 1.001;

    this.svg.setAttribute('viewBox', `0 0 ${hostWidth} ${hostHeight}`);

    // Hide all pooled labels first; visible ones will be toggled on in this pass
    for (const el of this.bayNodes.values()) el.style.display = 'none';
    for (const el of this.rulerNodes.values()) el.style.display = 'none';
    for (const el of this.bracketNodes) el.style.display = 'none';
    for (const el of this.bookendNodes.values()) el.style.display = 'none';
    this.hoverTag.style.display = 'none';
    this.rulerTitle.style.display = 'none';

    const leaderSegs: string[] = [];
    const activeLeaderSegs: string[] = [];

    // --- 1. Top band bay labels (§4 rules 1 & 2) ---
    const topRowYs = TOP_ROW_OFFSETS.map((off) => insetTop + off);
    const topRows: OccupiedBox[][] = [[], [], []];

    // Project all bay top anchors once
    const projectedTops = scene.bayAnchors.map((a) => ({
      anchor: a,
      inner: projectWorldToInnerCss(a.topWorld, snapshot),
      host: projectWorldToHostCss(a.topWorld, snapshot),
    }));

    // Mandatory 8 static bay IDs, sorted ascending X then ID
    const mandatoryItems = projectedTops
      .filter((item) => STATIC_BAY_SET.has(item.anchor.id))
      .sort((a, b) => a.host.x - b.host.x || a.anchor.id - b.anchor.id);

    const placeBayLabel = (
      item: (typeof projectedTops)[number],
      isMandatory: boolean,
    ) => {
      const { id } = item.anchor;
      const { x: ax, y: ay } = item.host;
      if (zoomed) {
        if (ax < 0 || ax > hostWidth || item.inner.y < 0 || item.inner.y > innerHeight) {
          return;
        }
      }

      const placed = placeInRows(
        topRows,
        topRowYs,
        ax,
        BAY_HALF_W,
        BAY_CLEARANCE,
        BAY_MARGIN_X,
        hostWidth - BAY_MARGIN_X,
        isMandatory,
      );
      if (!placed) return;

      const el = this.bayNodes.get(id);
      if (!el) return;
      const isSelected = selectedBay === id;
      const isHovered = hoveredBay === id;
      el.classList.toggle('is-active', isSelected || isHovered);
      el.style.display = 'flex';
      el.style.transform = `translate3d(${(placed.cx - BAY_HALF_W).toFixed(1)}px, ${(placed.cy - BAY_BOX_H * 0.5).toFixed(1)}px, 0)`;

      // Elbow leader: from projected anchor to (anchorX, rowY + 8) then nearest bottom edge of box
      const bottomY = placed.cy + BAY_BOX_H * 0.5;
      const clampedStartX = clamp(ax, 1, hostWidth - 1);
      const startY = clamp(ay, insetTop, insetTop + innerHeight);
      const targetX = clamp(clampedStartX, placed.cx - BAY_HALF_W + 2, placed.cx + BAY_HALF_W - 2);
      const pathStr =
        Math.abs(targetX - clampedStartX) > 0.5
          ? `M${clampedStartX.toFixed(1)},${startY.toFixed(1)}L${clampedStartX.toFixed(1)},${bottomY.toFixed(1)}L${targetX.toFixed(1)},${bottomY.toFixed(1)}`
          : `M${clampedStartX.toFixed(1)},${startY.toFixed(1)}L${clampedStartX.toFixed(1)},${bottomY.toFixed(1)}`;

      if (isSelected || isHovered) {
        activeLeaderSegs.push(pathStr);
      } else {
        leaderSegs.push(pathStr);
      }
    };

    for (const item of mandatoryItems) {
      placeBayLabel(item, true);
    }

    // Progressive reveal on zoom when projected bay pitch >= 30 CSS px (§4 rule 2)
    const pBay1 = projectWorldToInnerCss([0, 0, 0], snapshot);
    const pBay2 = projectWorldToInnerCss([1, 0, 0], snapshot);
    const projectedPitch = Math.hypot(pBay2.x - pBay1.x, pBay2.y - pBay1.y);

    if (projectedPitch >= 30) {
      const extraItems = projectedTops
        .filter((item) => !STATIC_BAY_SET.has(item.anchor.id))
        .sort((a, b) => a.anchor.id - b.anchor.id);
      for (const item of extraItems) {
        placeBayLabel(item, false);
      }
    }

    // --- 2. Bookend subtle labels (Embed / Output) ---
    for (const bk of scene.bookendAnchors) {
      if (bk.id === 'norm') continue;
      const el = this.bookendNodes.get(bk.id);
      if (!el) continue;
      const pt = projectWorldToHostCss(bk.world, snapshot);
      const innerPt = projectWorldToInnerCss(bk.world, snapshot);
      if (
        innerPt.x < 12 ||
        innerPt.x > innerWidth - 12 ||
        innerPt.y < 0 ||
        innerPt.y > innerHeight
      ) {
        continue;
      }
      const lx = clamp(pt.x + (bk.id === 'output' ? -4 : 0), 28, hostWidth - 28);
      const ly = Math.max(insetTop + 4, pt.y - (bk.id === 'output' ? 20 : 16));
      el.style.display = 'block';
      el.style.transform = `translate3d(${(lx - 24).toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
    }

    // --- 3. Depth ruler tick labels (1, 8, 16, and current T when distinct) ---
    const majorTokens = [1, 8, 16];
    const currentTick = scene.rulerTicks[tokens - 1];
    const placedRulerCoords: { x: number; y: number }[] = [];

    const placeRulerTick = (t: number, isCurrent: boolean) => {
      const tick = scene.rulerTicks[t - 1];
      if (!tick) return;
      const pt = projectWorldToHostCss(tick.labelWorld, snapshot);
      const innerPt = projectWorldToInnerCss(tick.labelWorld, snapshot);
      if (
        innerPt.x < 0 ||
        innerPt.x > innerWidth - 4 ||
        innerPt.y < 0 ||
        innerPt.y > innerHeight
      ) {
        return;
      }
      if (!isCurrent) {
        for (const occ of placedRulerCoords) {
          if (Math.abs(pt.x - occ.x) < 16 && Math.abs(pt.y - occ.y) < 10) {
            return;
          }
        }
      }
      const el = this.rulerNodes.get(t);
      if (!el) return;
      el.classList.toggle('is-current', isCurrent);
      el.style.display = 'block';
      const tx = clamp(pt.x + 4, 4, hostWidth - 20);
      const ty = clamp(pt.y - 7, insetTop + 2, insetTop + innerHeight - 14);
      el.style.transform = `translate3d(${tx.toFixed(1)}px, ${ty.toFixed(1)}px, 0)`;
      placedRulerCoords.push({ x: pt.x, y: pt.y });
    };

    // Place current T first so it never gets suppressed
    if (currentTick) {
      placeRulerTick(tokens, true);
    }
    for (const mt of majorTokens) {
      if (mt !== tokens) {
        placeRulerTick(mt, false);
      }
    }

    // Ruler unit header ("tokens") near tick 16 end
    const backTick = scene.rulerTicks[15];
    if (backTick) {
      const pt = projectWorldToHostCss(backTick.labelWorld, snapshot);
      const innerPt = projectWorldToInnerCss(backTick.labelWorld, snapshot);
      if (innerPt.x > 0 && innerPt.x < innerWidth && innerPt.y > 14 && innerPt.y < innerHeight) {
        const tx = clamp(pt.x - 10, 4, hostWidth - 44);
        const ty = clamp(pt.y - 20, insetTop + 2, insetTop + innerHeight - 16);
        this.rulerTitle.style.display = 'block';
        this.rulerTitle.style.transform = `translate3d(${tx.toFixed(1)}px, ${ty.toFixed(1)}px, 0)`;
      }
    }

    // --- 4. Transient hover tag (clamped inside drawing below top tiers) ---
    if (hoveredBay !== null) {
      const anchor = scene.bayAnchors[hoveredBay - 1];
      const bay = spec.bays[hoveredBay - 1];
      if (anchor && bay) {
        const pt = projectWorldToHostCss(anchor.topWorld, snapshot);
        const tagText = `${formatBayId(bay.id)} · ${bay.kind} ${bayKindLabel(bay.kind)}${bay.feeder ? ' (feeder)' : ''}`;
        this.hoverTag.textContent = tagText;
        const tagW = this.measureMonospaceWidth(tagText, 11) + 14;
        const tx = clamp(pt.x - tagW * 0.5, 6, hostWidth - tagW - 6);
        const ty = clamp(pt.y - 24, insetTop + 6, insetTop + innerHeight - 24);
        this.hoverTag.style.display = 'block';
        this.hoverTag.style.transform = `translate3d(${tx.toFixed(1)}px, ${ty.toFixed(1)}px, 0)`;
      }
    }

    // --- 5. Bottom band: four Figure-2 brackets (§4 rule 3) ---
    const bracketSegs: string[] = [];
    const bracketLeaderSegs: string[] = [];
    const projectedBrackets = scene.bracketAnchors.map((b) => ({
      anchor: b,
      caption: formatBracketCaption(spec.brackets[b.index]),
      left: projectWorldToHostCss(b.leftWorld, snapshot),
      right: projectWorldToHostCss(b.rightWorld, snapshot),
      mid: projectWorldToHostCss(b.midWorld, snapshot),
    }));

    let lowestEndpointY = insetTop + innerHeight;
    for (const pb of projectedBrackets) {
      const ly = pb.left.y + 8;
      const ry = pb.right.y + 8;
      if (ly > lowestEndpointY) lowestEndpointY = ly;
      if (ry > lowestEndpointY) lowestEndpointY = ry;

      const clipped = clipSegmentX(pb.left.x, ly, pb.right.x, ry, 0, hostWidth);
      if (clipped) {
        const [x0, y0, x1, y1] = clipped;
        bracketSegs.push(`M${x0.toFixed(1)},${y0.toFixed(1)}L${x1.toFixed(1)},${y1.toFixed(1)}`);
      }
      // 4 px end ticks pointing toward residual baseline
      if (pb.left.x >= 0 && pb.left.x <= hostWidth) {
        bracketSegs.push(
          `M${pb.left.x.toFixed(1)},${ly.toFixed(1)}L${pb.left.x.toFixed(1)},${(ly - 4).toFixed(1)}`,
        );
      }
      if (pb.right.x >= 0 && pb.right.x <= hostWidth) {
        bracketSegs.push(
          `M${pb.right.x.toFixed(1)},${ry.toFixed(1)}L${pb.right.x.toFixed(1)},${(ry - 4).toFixed(1)}`,
        );
      }
    }

    // Place four captions in span order in 15 px rows starting 18 px below lowest projected bracket endpoint
    const maxBaseRowY = Math.max(insetTop + innerHeight + 8, hostHeight - (BRACKET_MAX_ROWS * BRACKET_ROW_SPACING) + 2);
    const baseRowY = Math.min(maxBaseRowY, lowestEndpointY + 18);
    const bracketRowYs = [
      baseRowY,
      baseRowY + BRACKET_ROW_SPACING,
      baseRowY + BRACKET_ROW_SPACING * 2,
      baseRowY + BRACKET_ROW_SPACING * 3,
    ];
    const bracketRows: OccupiedBox[][] = [[], [], [], []];

    for (let i = 0; i < projectedBrackets.length; i++) {
      const pb = projectedBrackets[i];
      // Show caption only if midpoint remains visible (§4 rule 3)
      if (pb.mid.x < 0 || pb.mid.x > hostWidth) continue;

      const capW = this.measureMonospaceWidth(pb.caption, 11) + 8;
      const halfW = capW * 0.5;
      const minCx = Math.min(hostWidth * 0.5, halfW + 4);
      const maxCx = Math.max(hostWidth * 0.5, hostWidth - halfW - 4);

      const placed = placeInRows(
        bracketRows,
        bracketRowYs,
        pb.mid.x,
        halfW,
        BRACKET_CLEARANCE,
        minCx,
        maxCx,
        true,
      );
      if (!placed) continue;

      const el = this.bracketNodes[i];
      el.textContent = pb.caption;
      el.style.display = 'block';
      el.style.transform = `translate3d(${(placed.cx - halfW).toFixed(1)}px, ${(placed.cy - 7).toFixed(1)}px, 0)`;

      // Leader from bracket midpoint down to caption top
      const midBarY = pb.mid.y + 8;
      const capTopY = placed.cy - 7;
      const targetX = clamp(pb.mid.x, placed.cx - halfW + 4, placed.cx + halfW - 4);
      if (capTopY > midBarY + 2) {
        if (Math.abs(targetX - pb.mid.x) > 0.5) {
          bracketLeaderSegs.push(
            `M${pb.mid.x.toFixed(1)},${midBarY.toFixed(1)}L${pb.mid.x.toFixed(1)},${capTopY.toFixed(1)}L${targetX.toFixed(1)},${capTopY.toFixed(1)}`,
          );
        } else {
          bracketLeaderSegs.push(
            `M${pb.mid.x.toFixed(1)},${midBarY.toFixed(1)}L${pb.mid.x.toFixed(1)},${capTopY.toFixed(1)}`,
          );
        }
      }
    }

    // Batch SVG path attribute writes
    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.activeLeaderPath.setAttribute('d', activeLeaderSegs.join(''));
    this.bracketPath.setAttribute('d', bracketSegs.join(''));
    this.bracketLeaderPath.setAttribute('d', bracketLeaderSegs.join(''));
  }
}
