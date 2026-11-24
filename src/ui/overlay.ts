import {
  bayKindLabel,
  formatBayId,
  formatBracketCaption,
  type NemotronSpec,
} from '../model/spec';
import type { TocSectionId } from '../model/state';
import { projectWorldToHostCss, type CameraSnapshot } from '../render/camera';
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
}

export interface CutawayOverlayUpdateInput {
  readonly scene: CutawaySceneGeometry;
  readonly snapshot: CameraSnapshot;
  readonly tokens: number;
  readonly selectedPart: string | null;
  readonly hoveredPart: string | null;
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
}

interface PlacedRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const SVG_NS = 'http://www.w3.org/2000/svg';

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function overlapsAny(rects: readonly PlacedRect[], x0: number, y0: number, x1: number, y1: number): boolean {
  const pad = 3;
  for (const r of rects) {
    if (x0 < r.x1 + pad && x1 + pad > r.x0 && y0 < r.y1 + pad && y1 + pad > r.y0) {
      return true;
    }
  }
  return false;
}

/**
 * Synchronous DOM + SVG overlay updated inside WebGL's requestAnimationFrame pass.
 * Renders crisp llm-viz dark charcoal/black 3D pills (#14181F, white text) for both
 * the 52-Layer Tower and the 3 exploded layer cutaways.
 */
export class MacroOverlay {
  private readonly svg: SVGSVGElement;
  private readonly labelLayer: HTMLDivElement;
  private readonly leaderPath: SVGPathElement;
  private readonly phaseBracketPath: SVGPathElement;
  private readonly caliperPath: SVGPathElement;

  private readonly landmarkNodes = new Map<string, HTMLDivElement>();
  private readonly cutawayNodes = new Map<string, HTMLDivElement>();
  private readonly caliperNodes = new Map<string, HTMLDivElement>();
  private readonly phaseNodes: HTMLDivElement[] = [];
  private readonly bayMiniNodes = new Map<number, HTMLDivElement>();
  private readonly kvTickNodes = new Map<number, HTMLDivElement>();
  private readonly activePill: HTMLDivElement;

  private readonly measureCtx: CanvasRenderingContext2D | null;
  private readonly widthCache = new Map<string, number>();

  constructor(svg: SVGSVGElement, labelLayer: HTMLDivElement) {
    this.svg = svg;
    this.labelLayer = labelLayer;
    this.svg.innerHTML = '';
    this.labelLayer.innerHTML = '';

    const scratch = document.createElement('canvas');
    this.measureCtx = scratch.getContext('2d');

    this.phaseBracketPath = this.createSvgPath('overlay-phase-bracket');
    this.caliperPath = this.createSvgPath('overlay-caliper');
    this.leaderPath = this.createSvgPath('overlay-leader');
    this.svg.append(this.phaseBracketPath, this.caliperPath, this.leaderPath);

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
    for (const el of this.cutawayNodes.values()) el.style.display = 'none';
    for (const el of this.caliperNodes.values()) el.style.display = 'none';
    for (const el of this.phaseNodes) el.style.display = 'none';
    for (const el of this.bayMiniNodes.values()) el.style.display = 'none';
    for (const el of this.kvTickNodes.values()) el.style.display = 'none';
    this.activePill.style.display = 'none';
    this.caliperPath.setAttribute('d', '');
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

  /** Tower overlay update (52-Layer Macro Spine). */
  update(input: OverlayUpdateInput): void {
    const { spec, scene, snapshot, section, tokens, selectedBay, hoveredBay } = input;
    const { hostWidth, hostHeight } = snapshot.metrics;
    this.svg.setAttribute('viewBox', `0 0 ${hostWidth} ${hostHeight}`);
    this.hideAll();

    const occupied: PlacedRect[] = [];
    const leaderSegs: string[] = [];
    const phaseSegs: string[] = [];

    // Safe drawing area avoiding floating two-row mode/preset toolbar (~84px) and bottom hint (~30px)
    const minY = 84;
    const maxY = Math.max(minY + 40, hostHeight - 30);

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

  /** Exploded 3D Layer Cutaway overlay update (Mamba-2, MoE 128+1, GQA Attention). */
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

    // Thin-sheet drafting-table cutaways (GQA Increment 1+): exterior edge-aligned labels,
    // zero matrix-face occlusion, progressive detail reveal, and subtle SVG dimension calipers.
    if (scene.sheetCount && scene.sheetCount > 0) {
      this.updateSheetCutawayOverlay(scene, snapshot, tokens, activePart, occupied, leaderSegs, minY, maxY);
      return;
    }

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
      el.className = 'ov-pill';
      if (isAct) el.classList.add('ov-pill-active');
      else if (lm.tone === 'gold') el.classList.add('ov-pill-gold');
      else if (lm.tone === 'cyan') el.classList.add('ov-pill-cyan');
      else if (lm.tone === 'muted') el.classList.add('ov-pill-muted');

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

    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.phaseBracketPath.setAttribute('d', '');
  }

  /** Exterior, zero-occlusion label & dimension-caliper pass for thin-sheet cutaways (GQA). */
  private updateSheetCutawayOverlay(
    scene: CutawaySceneGeometry,
    snapshot: CameraSnapshot,
    tokens: number,
    activePart: string | null,
    occupied: PlacedRect[],
    leaderSegs: string[],
    minY: number,
    maxY: number,
  ): void {
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

    const ordered = [...scene.landmarks].sort((a, b) => {
      const aAct = a.partId !== undefined && a.partId === activePart ? 3 : a.tier === 'detail' ? 0 : 1;
      const bAct = b.partId !== undefined && b.partId === activePart ? 3 : b.tier === 'detail' ? 0 : 1;
      return bAct - aAct;
    });

    for (const lm of ordered) {
      const isAct = lm.partId !== undefined && lm.partId === activePart;
      if (lm.tier === 'detail' && !isAct && !isCloseZoom) {
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
      let lx = pt.x - w * 0.5;
      let ly = pt.y - h * 0.5;
      if (lm.align === 'left') {
        lx = pt.x - w - gap;
        ly = pt.y - h * 0.5;
      } else if (lm.align === 'right') {
        lx = pt.x + gap;
        ly = pt.y - h * 0.5;
      } else if (lm.align === 'top') {
        lx = pt.x - w * 0.5;
        ly = pt.y - h - gap;
      } else if (lm.align === 'bottom') {
        lx = pt.x - w * 0.5;
        ly = pt.y + gap;
      }

      const padX = hostWidth < 760 ? 6 : 8;
      lx = clamp(lx, padX, Math.max(padX, hostWidth - w - padX));
      ly = clamp(ly, minY, Math.max(minY, maxY - h));

      if (overlapsAny(occupied, lx, ly, lx + w, ly + h)) {
        const nudges: readonly [number, number][] = [
          [lx, clamp(ly - (h + 3), minY, Math.max(minY, maxY - h))],
          [lx, clamp(ly + (h + 3), minY, Math.max(minY, maxY - h))],
          [clamp(lx - 14, padX, Math.max(padX, hostWidth - w - padX)), ly],
          [clamp(lx + 14, padX, Math.max(padX, hostWidth - w - padX)), ly],
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
        if (!placed && !isAct) continue;
      }

      const el = this.getOrCreateCutawayNode(lm.id);
      el.className = isDetail ? 'ov-sheet-label ov-sheet-detail' : 'ov-sheet-label';
      if (isAct) el.classList.add('is-active');
      else if (lm.tone === 'gold') el.classList.add('is-gold');
      else if (lm.tone === 'cyan') el.classList.add('is-cyan');
      else if (lm.tone === 'muted') el.classList.add('is-muted');

      el.textContent = lm.label;
      el.style.display = 'inline-flex';
      el.style.transform = `translate3d(${lx.toFixed(1)}px, ${ly.toFixed(1)}px, 0)`;
      occupied.push({ x0: lx, y0: ly, x1: lx + w, y1: ly + h });
    }

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

    this.leaderPath.setAttribute('d', leaderSegs.join(''));
    this.caliperPath.setAttribute('d', caliperSegs.join(''));
    this.phaseBracketPath.setAttribute('d', '');
  }
}
