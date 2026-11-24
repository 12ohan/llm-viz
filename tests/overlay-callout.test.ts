import { afterEach, describe, expect, it } from 'vitest';

import { createPerspectivePose } from '../src/model/state';
import {
  buildCameraSnapshot,
  makeViewportMetrics,
  type AABB,
} from '../src/render/camera';
import {
  FRAME_XY_AXES,
  FRAME_XZ_AXES,
  FRAME_YZ_AXES,
} from '../src/render/sheetFrame';
import type { CutawaySceneGeometry } from '../src/scene/cutaways';
import {
  buildElbowLeaderPath,
  buildQuadraticPointerPath,
  computeProjectedCellSize,
  computeSheetCellWorldPos,
  DIMMED_LABEL_OPACITY,
  formatCellOverlayValue,
  MacroOverlay,
  MAX_OVERLAY_AXIS_HEADER_NODES,
  MAX_OVERLAY_CELL_NUM_NODES,
  pickPopoverPlacement,
  prioritizeSheetCellCoords,
  shouldShowAxisHeaders,
  shouldShowCellNumbers,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
  type OverlayRect,
} from '../src/ui/overlay';

// Minimal deterministic DOM shim so MacroOverlay can be exercised end-to-end under Node vitest
class MockClassList {
  private readonly set = new Set<string>();

  setFromString(value: string): void {
    this.set.clear();
    for (const tok of value.split(/\s+/)) {
      if (tok) this.set.add(tok);
    }
  }

  toString(): string {
    return [...this.set].join(' ');
  }

  add(...tokens: string[]): void {
    for (const t of tokens) if (t) this.set.add(t);
  }

  remove(...tokens: string[]): void {
    for (const t of tokens) this.set.delete(t);
  }

  toggle(token: string, force?: boolean): boolean {
    if (force === undefined) {
      if (this.set.has(token)) {
        this.set.delete(token);
        return false;
      }
      this.set.add(token);
      return true;
    }
    if (force) this.set.add(token);
    else this.set.delete(token);
    return force;
  }

  contains(token: string): boolean {
    return this.set.has(token);
  }
}

class MockElement {
  readonly tagName: string;
  id = '';
  textContent = '';
  readonly style: Record<string, string> = {};
  readonly classList = new MockClassList();
  readonly attributes = new Map<string, string>();
  readonly children: MockElement[] = [];

  constructor(tagName: string) {
    this.tagName = tagName;
  }

  get className(): string {
    return this.classList.toString();
  }

  set className(val: string) {
    this.classList.setFromString(val);
    this.attributes.set('class', val);
  }

  set innerHTML(val: string) {
    if (val === '') {
      this.children.length = 0;
    }
  }

  get innerHTML(): string {
    return '';
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    if (name === 'class') this.classList.setFromString(value);
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  appendChild(child: MockElement): MockElement {
    this.children.push(child);
    return child;
  }

  append(...nodes: MockElement[]): void {
    for (const n of nodes) this.children.push(n);
  }

  prepend(...nodes: MockElement[]): void {
    this.children.unshift(...nodes);
  }

  getContext(): null {
    return null;
  }
}

function installMockDocument(): {
  svg: MockElement;
  labelLayer: MockElement;
  createdCounts: Record<string, number>;
} {
  const head = new MockElement('head');
  const createdCounts: Record<string, number> = {};
  const record = (tag: string): void => {
    createdCounts[tag] = (createdCounts[tag] ?? 0) + 1;
  };
  const mockDoc = {
    head,
    getElementById(id: string): MockElement | null {
      return head.children.find((c) => c.id === id) ?? null;
    },
    createElement(tag: string): MockElement {
      record(tag);
      return new MockElement(tag);
    },
    createElementNS(_ns: string, tag: string): MockElement {
      record(tag);
      return new MockElement(tag);
    },
  };
  (globalThis as { document?: unknown }).document = mockDoc;
  return {
    svg: new MockElement('svg'),
    labelLayer: new MockElement('div'),
    createdCounts,
  };
}

function makeSheetScene(landmarks: CutawaySceneGeometry['landmarks']): CutawaySceneGeometry {
  return {
    kind: 'cutaway_gqa',
    bayId: 6,
    solidData: new Float32Array(0),
    solidCount: 0,
    lineData: new Float32Array(0),
    lineCount: 0,
    ribbonData: new Float32Array(0),
    ribbonCount: 0,
    sheetData: new Float32Array(20),
    sheetCount: 1,
    solidPickBoxes: [],
    landmarks,
    kvDepthTicks: [],
    fitBounds: { min: [-4, -1, -4], max: [4, 2, 4] },
    partCatalog: {},
    gqaStats: {
      bayId: 6,
      qHeadCount: 32,
      qGroupCount: 2,
      qHeadsPerGroup: 16,
      kvHeadCount: 2,
      gqaRatio: 16,
      kvTokenSliceCount: 16,
      computeFp8: false,
      kvCacheFp8: false,
    },
  };
}

const SCENE_BOUNDS: AABB = { min: [-6, -2, -6], max: [6, 4, 6] };

describe('V1/V3 Overlay Pure Helpers & Cutaway Callouts', () => {
  afterEach(() => {
    delete (globalThis as { document?: unknown }).document;
  });

  describe('shouldShowCellNumbers & shouldShowAxisHeaders', () => {
    it('enforces strict cellW >= 26 && cellH >= 16 readability threshold', () => {
      // Zoomed out
      expect(shouldShowCellNumbers(12, 10)).toBe(false);
      expect(shouldShowCellNumbers(12, 24)).toBe(false);
      expect(shouldShowCellNumbers(25.9, 16)).toBe(false);
      expect(shouldShowCellNumbers(26, 15.9)).toBe(false);
      expect(shouldShowCellNumbers(NaN, 20)).toBe(false);
      expect(shouldShowCellNumbers(34, Infinity)).toBe(false);

      // Exact boundary & zoomed in
      expect(shouldShowCellNumbers(26, 16)).toBe(true);
      expect(shouldShowCellNumbers(34, 20)).toBe(true);
      expect(shouldShowCellNumbers(80, 22)).toBe(true);
    });

    it('enables exterior axis headers when cellW >= 22 || cellH >= 15', () => {
      expect(shouldShowAxisHeaders(18, 12)).toBe(false);
      expect(shouldShowAxisHeaders(22, 10)).toBe(true);
      expect(shouldShowAxisHeaders(14, 15)).toBe(true);
      expect(shouldShowAxisHeaders(34, 20)).toBe(true);
    });
  });

  describe('formatCellOverlayValue', () => {
    it('formats masked cells as −∞ regardless of underlying float', () => {
      expect(formatCellOverlayValue(-1e9, 'signed', true)).toBe('−∞');
      expect(formatCellOverlayValue(0, 'prob', true)).toBe('−∞');
      expect(formatCellOverlayValue(-Infinity, 'signed', false)).toBe('−∞');
      expect(formatCellOverlayValue(Infinity, 'signed', false)).toBe('+∞');
      expect(formatCellOverlayValue(Number.NaN, 'signed', false)).toBe('—');
    });

    it('formats signed values with crisp tabular sign and avoids −0.00', () => {
      expect(formatCellOverlayValue(1.523, 'signed', false)).toBe('+1.52');
      expect(formatCellOverlayValue(-0.981, 'signed', false)).toBe('−0.98');
      expect(formatCellOverlayValue(-0.001, 'signed', false)).toBe('+0.00');
      expect(formatCellOverlayValue(0, 'signed', false)).toBe('+0.00');
      expect(formatCellOverlayValue(12.36, 'signed', false)).toBe('+12.4');
      expect(formatCellOverlayValue(-142.8, 'signed', false)).toBe('−143');
    });

    it('formats probabilities cleanly to 2 decimal places', () => {
      expect(formatCellOverlayValue(0.284, 'prob', false)).toBe('0.28');
      expect(formatCellOverlayValue(0.001, 'prob', false)).toBe('0.00');
      expect(formatCellOverlayValue(1.0, 'prob', false)).toBe('1.00');
    });
  });

  describe('computeSheetCellWorldPos & computeProjectedCellSize', () => {
    it('matches SHEETS_VS right-handed basis across XZ, XY, and YZ sheet frames', () => {
      const xzGrid: ActiveSheetCellGrid = {
        originWorld: [1, 0, 2],
        extent: [4, 0.08, 2],
        axisU: FRAME_XZ_AXES.axisU,
        axisN: FRAME_XZ_AXES.axisN,
        cols: 4,
        rows: 2,
        values: new Float32Array(8),
      };
      // Cell (0,0): x = 1 + (-0.5 + 0.125)*4 = -0.5, y = 0 + 0.5*0.08 = 0.04, z = 2 + (-0.5 + 0.25)*2 = 1.5
      const xz00 = computeSheetCellWorldPos(xzGrid, 0, 0);
      expect(xz00[0]).toBeCloseTo(-0.5, 5);
      expect(xz00[1]).toBeCloseTo(0.04, 5);
      expect(xz00[2]).toBeCloseTo(1.5, 5);

      // Cell (3,1): x = 2.5, y = 0.04, z = 2.5
      const xz31 = computeSheetCellWorldPos(xzGrid, 3, 1);
      expect(xz31[0]).toBeCloseTo(2.5, 5);
      expect(xz31[1]).toBeCloseTo(0.04, 5);
      expect(xz31[2]).toBeCloseTo(2.5, 5);

      // Upright XY wall: axisU = +X, axisN = +Z => axisV = cross(+X, +Z) = -Y (row 0 at top)
      const xyGrid: ActiveSheetCellGrid = {
        originWorld: [0, 1, 0],
        extent: [2, 0.06, 4],
        axisU: FRAME_XY_AXES.axisU,
        axisN: FRAME_XY_AXES.axisN,
        cols: 2,
        rows: 4,
        values: new Float32Array(8),
      };
      const xyRow0 = computeSheetCellWorldPos(xyGrid, 0, 0);
      const xyRow3 = computeSheetCellWorldPos(xyGrid, 0, 3);
      expect(xyRow0[0]).toBeCloseTo(-0.5, 5);
      expect(xyRow0[1]).toBeCloseTo(2.5, 5); // row 0 at top (+Y)
      expect(xyRow3[1]).toBeCloseTo(-0.5, 5); // row 3 at bottom (-Y)
      expect(xyRow0[2]).toBeCloseTo(0.03, 5); // +N front surface

      // Standing YZ fin: axisU = -Z, axisN = +X => axisV = cross(-Z, +X) = -Y
      const yzGrid: ActiveSheetCellGrid = {
        originWorld: [0, 0, 0],
        extent: [2, 0.06, 2],
        axisU: FRAME_YZ_AXES.axisU,
        axisN: FRAME_YZ_AXES.axisN,
        cols: 2,
        rows: 2,
        values: new Float32Array(4),
      };
      const yz00 = computeSheetCellWorldPos(yzGrid, 0, 0);
      const yz10 = computeSheetCellWorldPos(yzGrid, 1, 0);
      expect(yz00[0]).toBeCloseTo(0.03, 5);
      expect(yz00[1]).toBeCloseTo(0.5, 5);
      expect(yz00[2]).toBeCloseTo(0.5, 5); // col 0 at +0.5 along Z because axisU = -Z
      expect(yz10[2]).toBeCloseTo(-0.5, 5); // col 1 steps along -Z
    });

    it('scales projected cellW and cellH with camera zoom distance', () => {
      const grid: ActiveSheetCellGrid = {
        originWorld: [0, 0, 0],
        extent: [2.4, 0.06, 1.6],
        axisU: FRAME_XZ_AXES.axisU,
        axisN: FRAME_XZ_AXES.axisN,
        cols: 4,
        rows: 4,
        values: new Float32Array(16),
      };
      const metrics = makeViewportMetrics(0, 0, 1440, 900, 1);

      // Far zoomed-out overview camera -> cells well below 26x16 threshold
      const farSnap = buildCameraSnapshot(
        createPerspectivePose([0, 0, 0], 26, 18, 35, 36),
        SCENE_BOUNDS,
        metrics,
      );
      const farSize = computeProjectedCellSize(grid, farSnap);
      expect(farSize.visible).toBe(true);
      expect(shouldShowCellNumbers(farSize.cellW, farSize.cellH)).toBe(false);

      // Close-up inspection camera -> cells comfortably above 26x16 threshold
      const closeSnap = buildCameraSnapshot(
        createPerspectivePose([0, 0, 0], 4.2, 12, 48, 36),
        SCENE_BOUNDS,
        metrics,
      );
      const closeSize = computeProjectedCellSize(grid, closeSnap);
      expect(closeSize.visible).toBe(true);
      expect(closeSize.cellW).toBeGreaterThan(26);
      expect(closeSize.cellH).toBeGreaterThan(16);
      expect(shouldShowCellNumbers(closeSize.cellW, closeSize.cellH)).toBe(true);
    });
  });

  describe('pickPopoverPlacement viewport boundary clamping & collision avoidance', () => {
    it('clamps strictly inside viewport margins at 1440×900 and 1024×800 across corners', () => {
      const viewports = [
        { hostWidth: 1440, minY: 84, maxY: 780 },
        { hostWidth: 1024, minY: 76, maxY: 660 },
      ];
      const anchors: readonly [number, number][] = [
        [10, 86], // top-left corner
        [1430, 86], // top-right corner
        [10, 775], // bottom-left corner
        [1430, 775], // bottom-right corner
        [512, 360], // center
      ];

      for (const vp of viewports) {
        for (const [rawAx, rawAy] of anchors) {
          const anchorX = Math.min(vp.hostWidth - 6, Math.max(6, rawAx));
          const anchorY = Math.min(vp.maxY, Math.max(vp.minY, rawAy));
          const placed = pickPopoverPlacement({
            anchorX,
            anchorY,
            width: 260,
            height: 48,
            hostWidth: vp.hostWidth,
            minY: vp.minY,
            maxY: vp.maxY,
          });

          expect(placed.x).toBeGreaterThanOrEqual(8);
          expect(placed.x + placed.width).toBeLessThanOrEqual(vp.hostWidth - 8);
          expect(placed.y).toBeGreaterThanOrEqual(vp.minY);
          expect(placed.y + placed.height).toBeLessThanOrEqual(vp.maxY);

          // Leader path from popover edge to anchor is valid quadratic Bezier
          const path = buildQuadraticPointerPath(
            placed.edgeX,
            placed.edgeY,
            anchorX,
            anchorY,
            placed.exitOrientation,
          );
          expect(path.startsWith('M')).toBe(true);
          expect(path.includes('Q')).toBe(true);
        }
      }
    });

    it('avoids occupied label rectangles when choosing popover quadrant', () => {
      const occupied: OverlayRect[] = [
        // Block the preferred top-right spot around anchor (500, 400)
        { x0: 515, y0: 320, x1: 775, y1: 382 },
      ];
      const placed = pickPopoverPlacement({
        anchorX: 500,
        anchorY: 400,
        width: 240,
        height: 48,
        hostWidth: 1440,
        minY: 84,
        maxY: 780,
        occupied,
      });

      // Should relocate away from top-right obstacle
      expect(placed.side).not.toBe('top-right');
      const intersectsObstacle =
        placed.x < occupied[0].x1 &&
        placed.x + placed.width > occupied[0].x0 &&
        placed.y < occupied[0].y1 &&
        placed.y + placed.height > occupied[0].y0;
      expect(intersectsObstacle).toBe(false);
    });
  });

  describe('buildElbowLeaderPath & prioritizeSheetCellCoords', () => {
    it('builds crisp elbow leader paths from nudged anchors to label edges', () => {
      const seg = buildElbowLeaderPath(100, 220, { x0: 128, y0: 170, x1: 210, y1: 188 });
      expect(seg.startsWith('M100.0,220.0')).toBe(true);
      expect(seg.endsWith('L128.0,179.0')).toBe(true);
    });

    it('prioritizes active cell first, then active row/col, then nearby cells', () => {
      const coords = prioritizeSheetCellCoords(6, 4, 2, 1);
      expect(coords).toHaveLength(24);
      expect(coords[0]).toEqual([2, 1]);
      // Next 8 entries are the remaining cells in row 1 (5) + col 2 (3)
      const rowOrColSlice = coords.slice(1, 9);
      for (const [c, r] of rowOrColSlice) {
        expect(c === 2 || r === 1).toBe(true);
      }
    });
  });

  describe('MacroOverlay end-to-end V1/V3 integration (zero-GC pools)', () => {
    it('dims non-participating labels, draws elbow leaders when nudged, renders popover + capped <= 64 cell values', () => {
      const { svg, labelLayer, createdCounts } = installMockDocument();
      const overlay = new MacroOverlay(
        svg as unknown as SVGSVGElement,
        labelLayer as unknown as HTMLDivElement,
      );

      // Verify exact pre-allocated pool sizes on construction
      const cellNumEls = labelLayer.children.filter((el) => el.classList.contains('ov-cell-num'));
      const headerEls = labelLayer.children.filter((el) => el.classList.contains('ov-axis-header'));
      const popoverEls = labelLayer.children.filter((el) => el.classList.contains('ov-cell-popover'));
      expect(cellNumEls).toHaveLength(MAX_OVERLAY_CELL_NUM_NODES);
      expect(headerEls).toHaveLength(MAX_OVERLAY_AXIS_HEADER_NODES);
      expect(popoverEls).toHaveLength(1);
      expect(popoverEls[0].style.pointerEvents).toBe('none');

      // Two landmarks anchored at the exact same world point force a collision nudge > 10px
      const scene = makeSheetScene([
        { id: 'lm-q', label: 'Q00 Active', world: [0, 0.1, 0], partId: 'q-head-0', align: 'top' },
        { id: 'lm-k', label: 'KV0 Shared', world: [0, 0.1, 0], partId: 'kv-head-0', align: 'top' },
        { id: 'lm-idle', label: 'W_O Idle', world: [1.1, 0.1, 0], partId: 'o-proj', align: 'right' },
      ]);

      const metrics = makeViewportMetrics(0, 0, 1440, 900, 1);
      const closeSnap = buildCameraSnapshot(
        createPerspectivePose([0, 0, 0], 4.2, 10, 50, 36),
        SCENE_BOUNDS,
        metrics,
      );

      // 16×10 = 160 cells (> 64 cap!) to assert strict <= 64 hard cap + active prioritization
      const cols = 16;
      const rows = 10;
      const values = new Float32Array(cols * rows);
      const masked = new Uint8Array(cols * rows);
      for (let i = 0; i < values.length; i++) {
        values[i] = (i % 7) * 0.45 - 1.2;
      }
      masked[0 * cols + 1] = 1; // mask cell (1, 0)

      const grid: ActiveSheetCellGrid = {
        originWorld: [0, 0, 0],
        extent: [9.6, 0.06, 4.0],
        axisU: FRAME_XZ_AXES.axisU,
        axisN: FRAME_XZ_AXES.axisN,
        cols,
        rows,
        values,
        masked,
        format: 'signed',
        activeCol: 8,
        activeRow: 5,
        colLabels: Array.from({ length: cols }, (_, c) => `k${c}`),
        rowLabels: Array.from({ length: rows }, (_, r) => `t${r} tok`),
        colDimLabel: 'K = 16',
        rowDimLabel: 'Q = 10',
      };

      const popoverSpec: CellPopoverSpec = {
        world: computeSheetCellWorldPos(grid, 8, 5),
        tensorLabel: 'Scores',
        coordLabel: '[Q00, t5→t8]',
        badge: 'BF16',
        equation: 'dot(Q00, K0) / √4',
        valueText: '= +3.294',
        operandColors: ['indigo', 'teal', 'red'],
        sourceWorlds: [
          { world: [-1.5, 0.04, 0], role: 'indigo' },
          { world: [1.5, 0.04, -0.5], role: 'teal' },
        ],
      };

      // First update creates the 3 lazy cutaway label nodes; record creation baseline after 1st frame
      overlay.updateCutaway({
        scene,
        snapshot: closeSnap,
        tokens: 8,
        selectedPart: 'q-head-0',
        hoveredPart: null,
        activeOpPartIds: new Set(['q-head-0', 'kv-head-0']),
        cellPopover: popoverSpec,
        activeCellGrids: [grid],
      });

      // V1: Active-op dimming applied cleanly
      const sheetLabels = labelLayer.children.filter((el) => el.classList.contains('ov-sheet-label'));
      const idleLabel = sheetLabels.find((el) => el.textContent === 'W_O Idle');
      const activeLabel = sheetLabels.find((el) => el.textContent === 'Q00 Active');
      expect(idleLabel?.classList.contains('is-dimmed')).toBe(true);
      expect(idleLabel?.style.opacity).toBe(DIMMED_LABEL_OPACITY);
      expect(activeLabel?.classList.contains('is-dimmed')).toBe(false);
      expect(activeLabel?.style.opacity).toBe('1');

      // V1: Nudged colliding label emits an elbow leader into leaderPath
      const leaderPath = svg.children.find((el) => el.classList.contains('overlay-leader'));
      expect(leaderPath?.getAttribute('d')).toBeTruthy();

      // V3: Floating cell popover & curved quadratic Bezier provenance pointers rendered
      expect(popoverEls[0].style.display).toBe('flex');
      const popLeader = svg.children.find((el) => el.classList.contains('overlay-popover-leader'));
      const popIndigo = svg.children.find((el) =>
        el.classList.contains('overlay-popover-source-indigo'),
      );
      const popTeal = svg.children.find((el) => el.classList.contains('overlay-popover-source-teal'));
      expect(popLeader?.getAttribute('d')).toContain('Q');
      expect(popIndigo?.getAttribute('d')).toContain('Q');
      expect(popTeal?.getAttribute('d')).toContain('Q');

      // V3: On-cell numbers hard-capped at <= 64 and active cell highlighted
      const visibleCellNums = cellNumEls.filter((el) => el.style.display !== 'none');
      expect(visibleCellNums.length).toBeGreaterThan(0);
      expect(visibleCellNums.length).toBeLessThanOrEqual(MAX_OVERLAY_CELL_NUM_NODES);
      expect(visibleCellNums.some((el) => el.classList.contains('is-active'))).toBe(true);

      // V3: Exterior axis headers rendered within <= 24 cap
      const visibleHeaders = headerEls.filter((el) => el.style.display !== 'none');
      expect(visibleHeaders.length).toBeGreaterThan(0);
      expect(visibleHeaders.length).toBeLessThanOrEqual(MAX_OVERLAY_AXIS_HEADER_NODES);

      // Zero per-frame DOM node creation on subsequent updates
      const beforeSecondFrame = { ...createdCounts };
      overlay.updateCutaway({
        scene,
        snapshot: closeSnap,
        tokens: 8,
        selectedPart: 'q-head-0',
        hoveredPart: null,
        activeOpPartIds: null, // clearing activeOpPartIds restores opacity '1'
        cellPopover: null,
        activeCellGrids: [grid],
      });
      expect(createdCounts).toEqual(beforeSecondFrame);

      // Opacity restored to '1' and popover hidden when cleared
      expect(idleLabel?.classList.contains('is-dimmed')).toBe(false);
      expect(idleLabel?.style.opacity).toBe('1');
      expect(popoverEls[0].style.display).toBe('none');
      expect(popLeader?.getAttribute('d')).toBe('');

      // Zoomed-out camera suppresses all on-cell numeric readouts (strict >= 26x16 threshold)
      const farSnap = buildCameraSnapshot(
        createPerspectivePose([0, 0, 0], 38, 18, 35, 36),
        SCENE_BOUNDS,
        metrics,
      );
      overlay.updateCutaway({
        scene,
        snapshot: farSnap,
        tokens: 8,
        selectedPart: null,
        hoveredPart: null,
        activeCellGrids: [grid],
      });
      const visibleWhenFar = cellNumEls.filter((el) => el.style.display !== 'none');
      expect(visibleWhenFar).toHaveLength(0);

      // Backward-compatible legacy updateCutaway call (all optional fields omitted)
      overlay.updateCutaway({
        scene,
        snapshot: closeSnap,
        tokens: 8,
        selectedPart: null,
        hoveredPart: null,
      });
      expect(cellNumEls.filter((el) => el.style.display !== 'none')).toHaveLength(0);
      expect(headerEls.filter((el) => el.style.display !== 'none')).toHaveLength(0);
      expect(popoverEls[0].style.display).toBe('none');
    });

    it('renders masked cells as −∞ with is-masked class when zoomed in', () => {
      const { svg, labelLayer } = installMockDocument();
      const overlay = new MacroOverlay(
        svg as unknown as SVGSVGElement,
        labelLayer as unknown as HTMLDivElement,
      );

      const scene = makeSheetScene([]);
      const metrics = makeViewportMetrics(0, 0, 1440, 900, 1);
      const closeSnap = buildCameraSnapshot(
        createPerspectivePose([0, 0, 0], 4.0, 8, 52, 36),
        SCENE_BOUNDS,
        metrics,
      );

      const grid: ActiveSheetCellGrid = {
        originWorld: [0, 0, 0],
        extent: [2.0, 0.06, 1.4],
        axisU: FRAME_XZ_AXES.axisU,
        axisN: FRAME_XZ_AXES.axisN,
        cols: 3,
        rows: 2,
        values: new Float32Array([0.12, -1e9, -1e9, 0.35, 0.65, -1e9]),
        masked: new Uint8Array([0, 1, 1, 0, 0, 1]),
        format: 'prob',
        activeCol: 0,
        activeRow: 1,
      };

      overlay.updateCutaway({
        scene,
        snapshot: closeSnap,
        tokens: 3,
        selectedPart: null,
        hoveredPart: null,
        activeCellGrids: [grid],
      });

      const visibleCellNums = labelLayer.children.filter(
        (el) => el.classList.contains('ov-cell-num') && el.style.display !== 'none',
      );
      expect(visibleCellNums).toHaveLength(6);
      const maskedEls = visibleCellNums.filter((el) => el.classList.contains('is-masked'));
      expect(maskedEls).toHaveLength(3);
      for (const m of maskedEls) {
        expect(m.textContent).toBe('−∞');
      }
      const activeEl = visibleCellNums.find((el) => el.classList.contains('is-active'));
      expect(activeEl?.textContent).toBe('0.35');
    });
  });
});
