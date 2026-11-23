import {
  bayKindLabel,
  formatBayId,
  formatBracketCaption,
  type NemotronSpec,
} from '../model/spec';
import { projectWorldToHostCss, type CameraSnapshot } from '../render/camera';
import type { MacroSceneGeometry } from '../scene/spine';

const SVG_NS = 'http://www.w3.org/2000/svg';

interface PlacedRect {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function overlapsAny(rects: readonly PlacedRect[], x0: number, y0: number, x1: number, y1: number): boolean {
  for (const r of rects) {
    if (x0 < r.x1 + 4 && x1 > r.x0 - 4 && y0 < r.y1 + 3 && y1 > r.y0 - 3) {
      return true;
    }
  }
  return false;
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
 * Synchronous llm-viz-style dark charcoal/black pill overlay & Figure-2 phase brackets.
 * Updated directly inside WebGL's draw frame via DOM/SVG refs (zero framework state updates).
 */
export class MacroOverlay {
  private readonly svg: SVGSVGElement;
  private readonly labelLayer: HTMLDivElement;
  private readonly leaderPath: SVGPathElement;
  private readonly phaseBracketPath: SVGPathElement;
  private readonly measureCtx: CanvasRenderingContext2D | null;
  private readonly widthCache = new Map<string, number>();

  private readonly landmarkNodes = new Map<string, HTMLDivElement>();
  private readonly phaseNodes: HTMLDivElement[] = [];
  private readonly bayMiniNodes = new Map<number, HTMLDivElement>();
  private readonly kvTickNodes = new Map<number, HTMLDivElement>();
  private readonly activePill: HTMLDivElement;

  constructor(svg: SVGSVGElement, labelLayer: HTMLDivElement) {
    this.svg = svg;
    this.labelLayer = labelLayer;

    const c = document.createElement('canvas');
    this.measureCtx = c.getContext('2d');

    this.phaseBracketPath = this.createSvgPath('ov-phase-bracket-path');
    this.leaderPath = this.createSvgPath('ov-leader-path');
    this.svg.append(this.phaseBracketPath, this.leaderPath);

    // Pre-create landmark pills
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
      el.className = id.startsWith('gqa-') ? 'ov-pill ov-pill-gqa' : 'ov-pill ov-pill-bookend';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.landmarkNodes.set(id, el);
    }

    // Pre-create 4 Figure-2 phase tags
    for (let i = 0; i < 4; i++) {
      const el = document.createElement('div');
      el.className = 'ov-phase-tag';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.phaseNodes.push(el);
    }

    // Pre-create zoom-revealed individual bay pills (1..52)
    for (let id = 1; id <= 52; id++) {
      const el = document.createElement('div');
      el.className = 'ov-bay-mini';
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.bayMiniNodes.set(id, el);
    }

    // Pre-create KV depth tick badges (1..16)
    for (let t = 1; t <= 16; t++) {
      const el = document.createElement('div');
      el.className = 'ov-kv-tick';
      el.textContent = `T=${t}`;
      el.style.display = 'none';
      this.labelLayer.appendChild(el);
      this.kvTickNodes.set(t, el);
    }

    // Active hovered/pinned bay black pill
    this.activePill = document.createElement('div');
    this.activePill.className = 'ov-pill ov-pill-active';
    this.activePill.style.display = 'none';
    this.labelLayer.appendChild(this.activePill);
  }

  private createSvgPath(className: string): SVGPathElement {
    const p = document.createElementNS(SVG_NS, 'path');
    p.setAttribute('class', className);
    return p;
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
    this.widthCache.set(key, w);
    return w;
  }

  update(input: OverlayUpdateInput): void {
    const { spec, scene, snapshot, tokens, selectedBay, hoveredBay } = input;
    const { hostWidth, hostHeight } = snapshot.metrics;
    this.svg.setAttribute('viewBox', `0 0 ${hostWidth} ${hostHeight}`);

    for (const el of this.landmarkNodes.values()) el.style.display = 'none';
    for (const el of this.phaseNodes) el.style.display = 'none';
    for (const el of this.bayMiniNodes.values()) el.style.display = 'none';
    for (const el of this.kvTickNodes.values()) el.style.display = 'none';
    this.activePill.style.display = 'none';

    const occupied: PlacedRect[] = [];
    const leaderSegs: string[] = [];
    const phaseSegs: string[] = [];

    // Safe drawing area avoiding floating top toolbar (~52px) and bottom hint (~28px)
    const minY = 52;
    const maxY = Math.max(minY + 40, hostHeight - 28);

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
      // Skip GQA landmark pill if that exact bay already shows the active pill
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

      // Gentle vertical stagger if two landmarks crowd on narrow screens
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

    // --- 3. Progressive Zoom Bay Labels when zoomed close (bay vertical pitch >= 26 CSS px) ---
    const p1 = projectWorldToHostCss(scene.bayAnchors[0].centerWorld, snapshot);
    const p2 = projectWorldToHostCss(scene.bayAnchors[1].centerWorld, snapshot);
    const screenBayPitch = p1.visible && p2.visible ? Math.hypot(p2.x - p1.x, p2.y - p1.y) : 0;

    if (screenBayPitch >= 24) {
      for (const anchor of scene.bayAnchors) {
        if (anchor.id === activeBayId) continue;
        const bay = spec.bays[anchor.id - 1];
        if (!bay || bay.kind === '*') continue; // GQA bays already have landmark pills
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

      // Subtle vertical phase bracket with top/bottom ticks
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

    // --- 5. KV Depth Scale Ticks (1, 8, 16, active T) beside GQA bay 06 when zoomed in ---
    if (screenBayPitch >= 18) {
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
        const lx = clamp(pt.x + 2, 8, hostWidth - w - 8);
        const ly = clamp(pt.y + 4, minY, maxY - h);
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
}
