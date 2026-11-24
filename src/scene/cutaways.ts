import type { GqaSimResult } from '../math/gqaSim';
import type { MambaSimResult } from '../math/mambaSim';
import type { PrecisionMode, Vec3 } from '../model/spec';
import type { AABB } from '../render/camera';
import type { PickBox } from '../render/pick';
import type { Rg32fValueAtlas } from './atlasPacker';
import type { LayerFormulaReadout } from './sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_WEIGHT_BLUE,
  FACE_NEG_X,
  FACE_NEG_Y,
  FACE_NEG_Z,
  FACE_POS_X,
  FACE_POS_Y,
  FACE_POS_Z,
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_INSTANCE_FLOATS,
  RIBBON_KIND_ARROW_NEG_X,
  RIBBON_KIND_ARROW_NEG_Y,
  RIBBON_KIND_ARROW_POS_X,
  RIBBON_KIND_BOX,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from './spine';
export { buildGqaCutaway, kvHeadForQueryHead, parseQueryHeadPartId, formatQueryHeadId, GQA_FIT_BOUNDS } from './gqaCutaway';
export { buildMambaCutaway, parseMambaHeadPartId, MAMBA_FIT_BOUNDS } from './mambaCutaway';
export { formatMambaHeadId, mambaGroupForHead } from '../math/mambaSim';

export interface CutawayPartInfo {
  readonly id: string;
  readonly title: string;
  readonly badge: string;
  readonly summary: string;
  readonly specs: readonly string[];
}

export interface MambaCutawayStats {
  readonly bayId: 1 | 42;
  readonly isFeeder: boolean;
  readonly conv1dTapCount: number;
  readonly conv1dFp8: boolean;
  readonly ssmHeadCount: number;
  readonly ssmStateFp8: boolean;
  readonly inProjFp8: boolean;
  readonly outProjFp8: boolean;
  readonly activeHead?: number;
  readonly activeGroup?: number;
  readonly miniatureHeadCount?: number;
  readonly miniatureGroupCount?: number;
  readonly usesSmoothRibbons?: boolean;
  readonly voxelArrowCount?: number;
  readonly ssmBytesAll23?: number;
  readonly conv1dBytesAll23?: number;
}

export interface MoECutawayStats {
  readonly bayId: number;
  readonly routedTotal: number;
  readonly routedActiveCount: number;
  readonly activeExpertIds: readonly number[];
  readonly sharedExpertCount: number;
  readonly routedWidth: number;
  readonly sharedWidth: number;
  readonly matricesPerExpert: number;
  readonly hasGateProj: boolean;
  readonly cacheBytes: number;
  readonly weightsFp8: boolean;
}

export interface GqaCutawayStats {
  readonly bayId: 6 | 43;
  readonly qHeadCount: number;
  readonly qGroupCount: number;
  readonly qHeadsPerGroup: number;
  readonly kvHeadCount: number;
  readonly gqaRatio: number;
  readonly kvTokenSliceCount: number; // 2 * T paired K+V slices across 2 KV heads
  readonly computeFp8: boolean;
  readonly kvCacheFp8: boolean;
  readonly usesSmoothRibbons?: boolean;
  readonly voxelArrowCount?: number;
  readonly selectedQueryHead?: number | null;
  readonly activeGroup0Head?: number;
  readonly activeGroup1Head?: number;
}

export interface CutawaySceneGeometry {
  readonly kind: 'cutaway_mamba' | 'cutaway_moe' | 'cutaway_gqa';
  readonly bayId: number;
  readonly solidData: Float32Array;
  readonly solidCount: number;
  readonly lineData: Float32Array;
  readonly lineCount: number;
  readonly ribbonData: Float32Array;
  readonly ribbonCount: number;
  readonly sheetData?: Float32Array;
  readonly sheetCount?: number;
  readonly flowVertices?: Float32Array;
  readonly flowIndices?: Uint32Array;
  readonly calipers?: readonly DimensionCaliper[];
  readonly valueAtlas?: Rg32fValueAtlas;
  readonly gqaSim?: GqaSimResult;
  readonly mambaSim?: MambaSimResult;
  readonly activeFormula?: LayerFormulaReadout | null;
  readonly fitBounds: AABB;
  readonly landmarks: readonly LandmarkAnchor[];
  readonly kvDepthTicks: readonly KvDepthTickAnchor[];
  readonly solidPickBoxes: readonly PickBox[];
  readonly partCatalog: Readonly<Record<string, CutawayPartInfo>>;
  readonly mambaStats?: MambaCutawayStats;
  readonly moeStats?: MoECutawayStats;
  readonly gqaStats?: GqaCutawayStats;
}

interface SolidSpec {
  cx: number;
  cy: number;
  cz: number;
  ex: number;
  ey: number;
  ez: number;
  cellFamily: number;
  fp8Flag: number;
  phase: number;
  bayId: number;
  cellScale: number;
  highlightFlag: number;
  lineStyle: number;
  cssWidth: number;
  lineAlpha: number;
  pickPart?: string;
}

interface LineSpec {
  ax: number;
  ay: number;
  az: number;
  bx: number;
  by: number;
  bz: number;
  cssWidth: number;
  style: number;
  alpha: number;
  ownerId: number;
  face0: number;
  face1: number;
  ocx: number;
  ocy: number;
  ocz: number;
  oex: number;
  oey: number;
  oez: number;
}

interface RibbonSpec {
  cx: number;
  cy: number;
  cz: number;
  ex: number;
  ey: number;
  ez: number;
  kind: number;
  colorId: number;
  alpha: number;
}

function quant(v: number): number {
  return Math.round(v * 10000);
}

function segKey(ax: number, ay: number, az: number, bx: number, by: number, bz: number): string {
  const qax = quant(ax), qay = quant(ay), qaz = quant(az);
  const qbx = quant(bx), qby = quant(by), qbz = quant(bz);
  const aFirst =
    qax < qbx || (qax === qbx && (qay < qby || (qay === qby && qaz <= qbz)));
  return aFirst
    ? `${qax},${qay},${qaz}|${qbx},${qby},${qbz}`
    : `${qbx},${qby},${qbz}|${qax},${qay},${qaz}`;
}

class CutawayBuilder {
  readonly solids: SolidSpec[] = [];
  readonly dedupLines = new Map<string, LineSpec>();
  readonly freeLines: LineSpec[] = [];
  readonly ribbons: RibbonSpec[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly kvDepthTicks: KvDepthTickAnchor[] = [];

  private readonly bayId: number;

  constructor(bayId: number) {
    this.bayId = bayId;
  }

  addSolid(s: SolidSpec, pickable = true): void {
    this.solids.push(s);
    this.emitEdges(s);
    if (pickable && s.pickPart) {
      const hx = s.ex * 0.5;
      const hy = s.ey * 0.5;
      const hz = s.ez * 0.5;
      this.pickBoxes.push({
        bayId: this.bayId,
        partId: s.pickPart,
        min: [s.cx - hx, s.cy - hy, s.cz - hz],
        max: [s.cx + hx, s.cy + hy, s.cz + hz],
      });
    }
  }

  addRibbon(r: RibbonSpec): void {
    this.ribbons.push(r);
  }

  addGuideLine(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth = 0.9,
    style = LINE_STYLE_GUIDE,
    alpha = 0.55,
  ): void {
    this.freeLines.push({
      ax, ay, az, bx, by, bz,
      cssWidth, style, alpha,
      ownerId: 0,
      face0: -1, face1: -1,
      ocx: 0, ocy: 0, ocz: 0,
      oex: 0, oey: 0, oez: 0,
    });
  }

  /**
   * Emits a smooth translucent 3D quadratic-Bezier ribbon curve made of overlapping
   * axis-aligned ribbon segments plus an optional directional 3D arrowhead at p2.
   */
  addCurvedRibbon(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    steps: number,
    thickness: number,
    colorId: number,
    alpha: number,
    endArrowKind?: number,
  ): void {
    const evalPt = (t: number): Vec3 => {
      const u = 1 - t;
      return [
        u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
        u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1],
        u * u * p0[2] + 2 * u * t * p1[2] + t * t * p2[2],
      ];
    };

    for (let i = 0; i < steps; i++) {
      const a = evalPt(i / steps);
      const b = evalPt((i + 1) / steps);
      const cx = (a[0] + b[0]) * 0.5;
      const cy = (a[1] + b[1]) * 0.5;
      const cz = (a[2] + b[2]) * 0.5;
      const dx = Math.abs(b[0] - a[0]);
      const dy = Math.abs(b[1] - a[1]);
      const dz = Math.abs(b[2] - a[2]);
      const isLast = i === steps - 1 && endArrowKind !== undefined;
      this.ribbons.push({
        cx,
        cy,
        cz,
        ex: Math.max(thickness, dx + thickness * 0.45),
        ey: Math.max(thickness, dy + thickness * 0.45),
        ez: Math.max(thickness, dz + thickness * 0.45),
        kind: isLast ? endArrowKind : RIBBON_KIND_BOX,
        colorId,
        alpha: isLast ? Math.min(0.68, alpha + 0.14) : alpha,
      });
    }
  }

  private emitEdges(s: SolidSpec): void {
    const hx = s.ex * 0.5;
    const hy = s.ey * 0.5;
    const hz = s.ez * 0.5;
    const x0 = s.cx - hx, x1 = s.cx + hx;
    const y0 = s.cy - hy, y1 = s.cy + hy;
    const z0 = s.cz - hz, z1 = s.cz + hz;

    const rawEdges: readonly [number, number, number, number, number, number, number, number][] = [
      [x0, y0, z0, x1, y0, z0, FACE_NEG_Y, FACE_NEG_Z],
      [x0, y1, z0, x1, y1, z0, FACE_POS_Y, FACE_NEG_Z],
      [x0, y0, z1, x1, y0, z1, FACE_NEG_Y, FACE_POS_Z],
      [x0, y1, z1, x1, y1, z1, FACE_POS_Y, FACE_POS_Z],
      [x0, y0, z0, x0, y1, z0, FACE_NEG_X, FACE_NEG_Z],
      [x1, y0, z0, x1, y1, z0, FACE_POS_X, FACE_NEG_Z],
      [x0, y0, z1, x0, y1, z1, FACE_NEG_X, FACE_POS_Z],
      [x1, y0, z1, x1, y1, z1, FACE_POS_X, FACE_POS_Z],
      [x0, y0, z0, x0, y0, z1, FACE_NEG_X, FACE_NEG_Y],
      [x1, y0, z0, x1, y0, z1, FACE_POS_X, FACE_NEG_Y],
      [x0, y1, z0, x0, y1, z1, FACE_NEG_X, FACE_POS_Y],
      [x1, y1, z0, x1, y1, z1, FACE_POS_X, FACE_POS_Y],
    ];

    for (const [ax, ay, az, bx, by, bz, face0, face1] of rawEdges) {
      const key = segKey(ax, ay, az, bx, by, bz);
      const existing = this.dedupLines.get(key);
      const cand: LineSpec = {
        ax, ay, az, bx, by, bz,
        cssWidth: s.cssWidth,
        style: s.lineStyle,
        alpha: s.lineAlpha,
        ownerId: s.bayId,
        face0, face1,
        ocx: s.cx, ocy: s.cy, ocz: s.cz,
        oex: s.ex, oey: s.ey, oez: s.ez,
      };
      if (!existing) {
        this.dedupLines.set(key, cand);
      } else {
        const candPri =
          cand.style === LINE_STYLE_ACTIVE ? 3 : cand.style === LINE_STYLE_BF16_GOLD ? 2 : 1;
        const existPri =
          existing.style === LINE_STYLE_ACTIVE ? 3 : existing.style === LINE_STYLE_BF16_GOLD ? 2 : 1;
        if (candPri > existPri || (candPri === existPri && cand.cssWidth > existing.cssWidth)) {
          this.dedupLines.set(key, cand);
        }
      }
    }
  }

  pack(): {
    solidData: Float32Array;
    solidCount: number;
    lineData: Float32Array;
    lineCount: number;
    ribbonData: Float32Array;
    ribbonCount: number;
  } {
    const solidData = new Float32Array(this.solids.length * SOLID_INSTANCE_FLOATS);
    for (let i = 0; i < this.solids.length; i++) {
      const s = this.solids[i];
      const o = i * SOLID_INSTANCE_FLOATS;
      solidData[o + 0] = s.cx;
      solidData[o + 1] = s.cy;
      solidData[o + 2] = s.cz;
      solidData[o + 3] = s.ex;
      solidData[o + 4] = s.ey;
      solidData[o + 5] = s.ez;
      solidData[o + 6] = s.cellFamily;
      solidData[o + 7] = s.fp8Flag;
      solidData[o + 8] = s.phase;
      solidData[o + 9] = s.bayId;
      solidData[o + 10] = s.cellScale;
      solidData[o + 11] = s.highlightFlag;
    }

    const orderedLines: LineSpec[] = [
      ...this.freeLines,
      ...Array.from(this.dedupLines.values()),
    ];
    const lineData = new Float32Array(orderedLines.length * LINE_INSTANCE_FLOATS);
    for (let i = 0; i < orderedLines.length; i++) {
      const l = orderedLines[i];
      const o = i * LINE_INSTANCE_FLOATS;
      lineData[o + 0] = l.ax;
      lineData[o + 1] = l.ay;
      lineData[o + 2] = l.az;
      lineData[o + 3] = l.bx;
      lineData[o + 4] = l.by;
      lineData[o + 5] = l.bz;
      lineData[o + 6] = l.cssWidth;
      lineData[o + 7] = l.style;
      lineData[o + 8] = l.alpha;
      lineData[o + 9] = l.ownerId;
      lineData[o + 10] = l.face0;
      lineData[o + 11] = l.face1;
      lineData[o + 12] = l.ocx;
      lineData[o + 13] = l.ocy;
      lineData[o + 14] = l.ocz;
      lineData[o + 15] = l.oex;
      lineData[o + 16] = l.oey;
      lineData[o + 17] = l.oez;
    }

    const ribbonData = new Float32Array(this.ribbons.length * RIBBON_INSTANCE_FLOATS);
    for (let i = 0; i < this.ribbons.length; i++) {
      const r = this.ribbons[i];
      const o = i * RIBBON_INSTANCE_FLOATS;
      ribbonData[o + 0] = r.cx;
      ribbonData[o + 1] = r.cy;
      ribbonData[o + 2] = r.cz;
      ribbonData[o + 3] = r.ex;
      ribbonData[o + 4] = r.ey;
      ribbonData[o + 5] = r.ez;
      ribbonData[o + 6] = r.kind;
      ribbonData[o + 7] = r.colorId;
      ribbonData[o + 8] = r.alpha;
      ribbonData[o + 9] = 0;
      ribbonData[o + 10] = 0;
      ribbonData[o + 11] = 0;
    }

    return {
      solidData,
      solidCount: this.solids.length,
      lineData,
      lineCount: orderedLines.length,
      ribbonData,
      ribbonCount: this.ribbons.length,
    };
  }
}

/**
 * Deterministic Top-6 sigmoid routed expert selection out of 128 experts for token T.
 * Spreads the 6 active winners across distinct rows/columns of the 8×16 bank and
 * visibly shifts which 6 experts activate when T steps from 1 to 16.
 */
export function activeRoutedExpertIndices(tokens: number): readonly number[] {
  const t = Math.max(1, Math.min(16, Math.round(tokens)));
  const picks = new Set<number>();
  // Deterministic prime-stride scatter across the 8x16 bank
  const seeds = [7, 29, 53, 79, 103, 121];
  for (let i = 0; i < seeds.length; i++) {
    let idx = (seeds[i] + t * (13 + i * 7)) % 128;
    while (picks.has(idx)) {
      idx = (idx + 5) % 128;
    }
    picks.add(idx);
  }
  return Array.from(picks).sort((a, b) => a - b);
}

// Mamba-2 cutaway is now implemented in ./mambaCutaway.ts (Stage A+1 · Increment 3).
// Re-exported at the top of this module.

// ============================================================================
// 2. CUTAWAY: GRANULAR MoE 128+1 LAYER (Bay 02 — Parallel Routing)
// ============================================================================

export function buildMoECutaway(opts: {
  tokens: number;
  precision: PrecisionMode;
  selectedPart: string | null;
  hoveredPart: string | null;
}): CutawaySceneGeometry {
  const bayId = 2;
  const clampedT = Math.max(1, Math.min(16, Math.round(opts.tokens)));
  const isFp8Mode = opts.precision === 'fp8_selective';
  const weightsFp8 = isFp8Mode; // Bay 02 is FP8-eligible in FP8 Selective

  const activeExpertIds = activeRoutedExpertIndices(clampedT);
  const activeSet = new Set(activeExpertIds);

  const b = new CutawayBuilder(bayId);
  const activePart = opts.hoveredPart ?? opts.selectedPart;
  const isPartActive = (id: string) => activePart === id;

  // --- Right Vertical Emerald Residual Stream ---
  const resX = 8.4;
  b.addRibbon({
    cx: resX, cy: -0.2, cz: 0,
    ex: 0.42, ey: 15.0, ez: 0.42,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.34,
  });

  // Residual In
  b.addSolid({
    cx: resX, cy: 6.4, cz: 0,
    ex: 0.85, ey: 1.50, ez: 1.15,
    cellFamily: CELL_ACTIVATION_GREEN,
    fp8Flag: 0,
    phase: clampedT * 0.6 + 1,
    bayId,
    cellScale: 0.22,
    highlightFlag: isPartActive('residual-in') ? 1 : 0,
    lineStyle: isPartActive('residual-in') ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: isPartActive('residual-in') ? 1.85 : 0.95,
    lineAlpha: isPartActive('residual-in') ? 1.0 : 0.62,
    pickPart: 'residual-in',
  });

  // Residual Out
  b.addSolid({
    cx: resX, cy: -6.8, cz: 0,
    ex: 0.85, ey: 1.50, ez: 1.15,
    cellFamily: CELL_ACTIVATION_GREEN,
    fp8Flag: 0,
    phase: clampedT * 0.6 + 11,
    bayId,
    cellScale: 0.22,
    highlightFlag: isPartActive('residual-out') ? 1 : 0,
    lineStyle: isPartActive('residual-out') ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: isPartActive('residual-out') ? 1.85 : 0.95,
    lineAlpha: isPartActive('residual-out') ? 1.0 : 0.62,
    pickPart: 'residual-out',
  });

  // Pre-RMSNorm (2,688)
  b.addSolid({
    cx: 5.55, cy: 5.65, cz: 0,
    ex: 0.56, ey: 1.30, ez: 1.10,
    cellFamily: CELL_MOE_AMBER,
    fp8Flag: weightsFp8 ? 1 : 0,
    phase: 3 + clampedT * 0.3,
    bayId,
    cellScale: 0.20,
    highlightFlag: isPartActive('rmsnorm') ? 1 : 0,
    lineStyle: isPartActive('rmsnorm') ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: isPartActive('rmsnorm') ? 1.85 : 0.95,
    lineAlpha: isPartActive('rmsnorm') ? 1.0 : weightsFp8 ? 0.42 : 0.62,
    pickPart: 'rmsnorm',
  });
  // Residual In -> RMSNorm tap ribbon
  b.addCurvedRibbon(
    [resX - 0.5, 6.2, 0],
    [6.9, 6.0, 0],
    [5.95, 5.65, 0],
    5,
    0.26,
    RIBBON_COLOR_EMERALD,
    0.42,
    RIBBON_KIND_ARROW_NEG_X,
  );

  // Parallel Split ribbons:
  // 1. Right parallel stream straight down into Shared Expert (always active!)
  b.addCurvedRibbon(
    [5.35, 4.95, 0],
    [5.15, 3.85, 0],
    [5.05, 2.55, 0],
    5,
    0.28,
    RIBBON_COLOR_EMERALD,
    0.46,
    RIBBON_KIND_ARROW_NEG_Y,
  );
  // 2. Left parallel stream into MLP Router & 128-Expert Bank
  b.addCurvedRibbon(
    [5.20, 5.55, 0],
    [1.6, 5.25, 0],
    [-2.0, 4.35, 0],
    7,
    0.28,
    RIBBON_COLOR_EMERALD,
    0.44,
    RIBBON_KIND_ARROW_NEG_Y,
  );

  // --- Branch 1A: MLP Router (2,688 -> 128) + Sigmoid Top-6 Score Bar ---
  {
    const act = isPartActive('router');
    b.addSolid({
      cx: -2.0, cy: 3.55, cz: 0,
      ex: 4.8, ey: 0.72, ez: 1.05,
      cellFamily: CELL_MOE_AMBER,
      fp8Flag: weightsFp8 ? 1 : 0,
      phase: clampedT * 1.4,
      bayId,
      cellScale: 0.21,
      highlightFlag: act ? 1 : 0,
      lineStyle: act ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
      cssWidth: act ? 1.85 : 1.1,
      lineAlpha: act ? 1.0 : 0.72,
      pickPart: 'router',
    });

    // 16 miniature sigmoid score columns atop the router bar, with 6 tall bright-amber winners
    const winnerBuckets = new Set(activeExpertIds.map((id) => id % 16));
    let winnersDrawn = 0;
    for (let s = 0; s < 16; s++) {
      const isWinner = winnerBuckets.has(s) && winnersDrawn < 6;
      if (isWinner) winnersDrawn++;
      const barH = isWinner ? 0.68 : 0.20;
      b.addSolid({
        cx: -4.25 + s * 0.30,
        cy: 3.92 + barH * 0.5,
        cz: 0.28,
        ex: 0.16,
        ey: barH,
        ez: 0.22,
        cellFamily: isWinner ? CELL_MOE_AMBER : CELL_NEUTRAL,
        fp8Flag: 0,
        phase: clampedT + s,
        bayId,
        cellScale: 0.16,
        highlightFlag: isWinner ? 2 : 0,
        lineStyle: isWinner ? LINE_STYLE_BF16_GOLD : LINE_STYLE_NEUTRAL,
        cssWidth: isWinner ? 1.0 : 0.6,
        lineAlpha: isWinner ? 0.90 : 0.32,
        pickPart: 'router',
      });
    }
  }

  // --- Branch 1B: 128 Routed Experts in 8 rows (Z) × 16 columns (X) ---
  // Active 6 experts rise cleanly and display TWO distinct non-gated ReLU² weight plates:
  //   up_proj (2,688 -> 1,856) and down_proj (1,856 -> 2,688) — NO gate_proj!
  // Inactive 122 experts sit dimmed in quiet slate (95.3% skipped per token).
  const bankCols = 16;
  const bankRows = 8;
  const pitchX = 0.58;
  const pitchZ = 0.62;
  const bankStartX = -6.55;
  const bankStartZ = -((bankRows - 1) * pitchZ) * 0.5;
  const bankBaseY = -0.75;

  const routedBankActive = isPartActive('routed-bank');
  const inactiveSelected = isPartActive('inactive-experts');

  for (let r = 0; r < bankRows; r++) {
    for (let c = 0; c < bankCols; c++) {
      const expertId = r * bankCols + c;
      const ex = bankStartX + c * pitchX;
      const ez = bankStartZ + r * pitchZ;
      const isWinner = activeSet.has(expertId);

      if (isWinner) {
        // Active expert: elevated with TWO non-gated ReLU² plates (up_proj + down_proj)
        const liftY = bankBaseY + 0.62;
        const borderStyle = routedBankActive ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD;
        const borderWidth = routedBankActive ? 1.65 : 1.25;

        // Plate 1: up_proj (2,688 -> 1,856) — vivid amber/gold matrix cells
        b.addSolid({
          cx: ex,
          cy: liftY + 0.34,
          cz: ez,
          ex: 0.44,
          ey: 0.52,
          ez: 0.44,
          cellFamily: CELL_MOE_AMBER,
          fp8Flag: weightsFp8 ? 1 : 0,
          phase: clampedT * 1.3 + expertId * 0.17,
          bayId,
          cellScale: 0.14,
          highlightFlag: routedBankActive ? 1 : 2,
          lineStyle: borderStyle,
          cssWidth: borderWidth,
          lineAlpha: 0.96,
          pickPart: 'routed-bank',
        });

        // Plate 2: down_proj (1,856 -> 2,688) — periwinkle weight matrix (NO gate_proj!)
        b.addSolid({
          cx: ex,
          cy: liftY - 0.24,
          cz: ez,
          ex: 0.44,
          ey: 0.48,
          ez: 0.44,
          cellFamily: CELL_WEIGHT_BLUE,
          fp8Flag: weightsFp8 ? 1 : 0,
          phase: clampedT * 1.1 + expertId * 0.31,
          bayId,
          cellScale: 0.14,
          highlightFlag: routedBankActive ? 1 : 2,
          lineStyle: borderStyle,
          cssWidth: borderWidth,
          lineAlpha: 0.94,
          pickPart: 'routed-bank',
        });

        // Translucent emerald fan-out ribbon from Router -> Active Expert
        b.addCurvedRibbon(
          [-2.0, 3.15, 0],
          [(ex - 2.0) * 0.5, 1.95, ez * 0.65],
          [ex, liftY + 0.68, ez],
          5,
          0.14,
          RIBBON_COLOR_EMERALD,
          0.44,
          RIBBON_KIND_ARROW_NEG_Y,
        );

        // Translucent emerald fan-in ribbon from Active Expert -> Weighted Sum (+)
        b.addCurvedRibbon(
          [ex, liftY - 0.55, ez],
          [(ex + 1.8) * 0.5, -3.55, ez * 0.45],
          [1.8, -5.05, 0],
          5,
          0.14,
          RIBBON_COLOR_EMERALD,
          0.42,
        );
      } else {
        // Inactive expert (122 of 128): dimmed quiet slate, skipped on this token
        b.addSolid({
          cx: ex,
          cy: bankBaseY - 0.18,
          cz: ez,
          ex: 0.40,
          ey: 0.44,
          ez: 0.40,
          cellFamily: CELL_NEUTRAL,
          fp8Flag: weightsFp8 ? 1 : 0,
          phase: expertId * 0.1,
          bayId,
          cellScale: 0.25,
          highlightFlag: inactiveSelected ? 1 : 0,
          lineStyle: inactiveSelected ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
          cssWidth: inactiveSelected ? 1.2 : 0.55,
          lineAlpha: inactiveSelected ? 0.85 : 0.26,
          pickPart: 'inactive-experts',
        });
      }
    }
  }

  // --- Branch 2: Parallel Shared Expert (3,712 = 2 × 1,856, always active, 2× wider!) ---
  {
    const act = isPartActive('shared-expert');
    const sx = 5.05;
    const style = act ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL;
    const width = act ? 1.85 : 1.05;

    // Shared up_proj (2,688 -> 3,712) — visibly 2× wider than routed experts
    b.addSolid({
      cx: sx, cy: 1.35, cz: 0,
      ex: 2.25, ey: 1.25, ez: 1.35,
      cellFamily: CELL_WEIGHT_BLUE,
      fp8Flag: weightsFp8 ? 1 : 0,
      phase: 21 + clampedT * 0.4,
      bayId,
      cellScale: 0.22,
      highlightFlag: act ? 1 : 0,
      lineStyle: style,
      cssWidth: width,
      lineAlpha: act ? 1.0 : weightsFp8 ? 0.44 : 0.68,
      pickPart: 'shared-expert',
    });

    // Shared ReLU² activation wafer (3,712)
    b.addSolid({
      cx: sx, cy: -0.15, cz: 0,
      ex: 2.15, ey: 0.46, ez: 1.25,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8Flag: 0,
      phase: clampedT * 1.2 + 5,
      bayId,
      cellScale: 0.19,
      highlightFlag: act ? 1 : 0,
      lineStyle: act ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      cssWidth: act ? 1.7 : 0.95,
      lineAlpha: act ? 1.0 : 0.62,
      pickPart: 'shared-expert',
    });

    // Shared down_proj (3,712 -> 2,688) — non-gated ReLU² (2 matrices total)
    b.addSolid({
      cx: sx, cy: -1.65, cz: 0,
      ex: 2.25, ey: 1.25, ez: 1.35,
      cellFamily: CELL_WEIGHT_BLUE,
      fp8Flag: weightsFp8 ? 1 : 0,
      phase: 37 + clampedT * 0.4,
      bayId,
      cellScale: 0.22,
      highlightFlag: act ? 1 : 0,
      lineStyle: style,
      cssWidth: width,
      lineAlpha: act ? 1.0 : weightsFp8 ? 0.44 : 0.68,
      pickPart: 'shared-expert',
    });

    // Periwinkle matmul arrows inside shared expert
    b.addRibbon({
      cx: sx, cy: 0.42, cz: 0,
      ex: 0.45, ey: 0.45, ez: 0.42,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_PERIWINKLE,
      alpha: weightsFp8 ? 0.30 : 0.50,
    });
    b.addRibbon({
      cx: sx, cy: -0.72, cz: 0,
      ex: 0.45, ey: 0.45, ez: 0.42,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_PERIWINKLE,
      alpha: weightsFp8 ? 0.30 : 0.50,
    });

    // Emerald output ribbon Shared Expert -> Weighted Sum (+)
    b.addCurvedRibbon(
      [sx, -2.45, 0],
      [4.9, -4.35, 0],
      [3.35, -5.25, 0],
      6,
      0.28,
      RIBBON_COLOR_EMERALD,
      0.46,
      RIBBON_KIND_ARROW_NEG_X,
    );
  }

  // --- Weighted Sum Merge (+) ---
  {
    const act = isPartActive('merge-sum');
    b.addSolid({
      cx: 2.25, cy: -5.35, cz: 0,
      ex: 1.95, ey: 1.05, ez: 1.15,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8Flag: 0,
      phase: clampedT * 0.9 + 15,
      bayId,
      cellScale: 0.20,
      highlightFlag: act ? 1 : 0,
      lineStyle: act ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      cssWidth: act ? 1.85 : 1.0,
      lineAlpha: act ? 1.0 : 0.65,
      pickPart: 'merge-sum',
    });
    // Merge -> Residual Out
    b.addCurvedRibbon(
      [3.35, -5.45, 0],
      [5.85, -6.15, 0],
      [resX - 0.55, -6.8, 0],
      6,
      0.30,
      RIBBON_COLOR_EMERALD,
      0.48,
      RIBBON_KIND_ARROW_POS_X,
    );
  }

  // --- Black-Pill 3D Slab Labels ---
  b.landmarks.push(
    { id: 'e-res-in', label: 'Residual In (2,688)', world: [resX + 0.55, 7.15, 0], partId: 'residual-in' },
    { id: 'e-rmsnorm', label: 'RMSNorm → Parallel Split', world: [5.55, 6.65, 0], partId: 'rmsnorm' },
    {
      id: 'e-router',
      label: `MLP Router (2,688→128) · Top-6 Sigmoid (T=${clampedT})`,
      world: [-2.0, 4.95, 0.3],
      partId: 'router',
      tone: 'gold',
    },
    {
      id: 'e-active',
      label: '6 Active Routed Experts (ReLU² · up + down)',
      world: [-2.55, 1.35, 2.4],
      partId: 'routed-bank',
      tone: 'gold',
    },
    {
      id: 'e-inactive',
      label: '122 Inactive Experts (95.3% skipped)',
      world: [-6.6, -1.75, 2.3],
      partId: 'inactive-experts',
      tone: 'muted',
    },
    {
      id: 'e-shared',
      label: 'Shared Expert (3,712 = 2×1,856 · Always Active)',
      world: [5.85, 2.45, 0.8],
      partId: 'shared-expert',
      tone: 'default',
    },
    { id: 'e-merge', label: 'Weighted Sum + (6 Routed + Shared)', world: [2.25, -6.25, 0.7], partId: 'merge-sum' },
    { id: 'e-res-out', label: '+ Residual Out (2,688)', world: [resX + 0.55, -7.65, 0], partId: 'residual-out' },
  );

  const activeListText = activeExpertIds.map((id) => `#${id}`).join(', ');
  const partCatalog: Record<string, CutawayPartInfo> = {
    'residual-in': {
      id: 'residual-in',
      title: 'Residual Stream Input (2,688)',
      badge: 'Activation · Emerald',
      summary:
        'Hidden vector arriving from the preceding Mamba-2 or GQA bay. Immediately splits in parallel after RMSNorm so the sparse routed bank and shared expert execute concurrently.',
      specs: [
        'Shape: [batch, 2,688] per token',
        'Parallel overlap: moe_shared_expert_overlap = true',
      ],
    },
    rmsnorm: {
      id: 'rmsnorm',
      title: 'Pre-MoE RMSNorm + Parallel Split',
      badge: weightsFp8 ? 'FP8 Selective' : 'BF16',
      summary:
        'Normalizes the 2,688-channel token vector and fans it simultaneously into the 128-expert routed branch (left) and the always-active double-width shared expert (right).',
      specs: [
        'Experts execute in PARALLEL — never serialized',
        '1.30B params/layer × 23 = 29.8B (~94% of 31.6B weights) · 0 B token cache',
      ],
    },
    router: {
      id: 'router',
      title: 'MLP Router (2,688 → 128) + Sigmoid Gate',
      badge: `Top-6 Active · T=${clampedT}`,
      summary:
        'Scores all 128 routed experts using a learned linear gate with sigmoid normalization (not softmax), selecting the top-6 highest-scoring experts per token.',
      specs: [
        `Token T=${clampedT} routed winners: Experts ${activeListText}`,
        'Routing weights scale each winning expert output before summation',
        'Scrubbing T = 1 → 16 shifts which 6 of 128 experts fire',
      ],
    },
    'routed-bank': {
      id: 'routed-bank',
      title: '6 Active Routed Experts (ReLU²)',
      badge: weightsFp8 ? 'FP8 Weights · 6 / 128 Active' : 'BF16 · 6 / 128 Active',
      summary:
        'Each active routed expert runs a non-gated two-matrix MLP (up_proj 2,688→1,856, ReLU², down_proj 1,856→2,688 — NO gate_proj). Only 6 of 128 experts execute on any token.',
      specs: [
        `Active on T=${clampedT}: Experts ${activeListText} (6 of 128 = 4.7%)`,
        'Matrices per expert: 2 (up_proj + down_proj, non-gated ReLU²)',
        '1.30B params/layer × 23 = 29.8B (~94% of 31.6B weights) · 0 B token cache',
      ],
    },
    'inactive-experts': {
      id: 'inactive-experts',
      title: '122 Idle Routed Experts (95.3% Skipped)',
      badge: 'Zero Compute This Token',
      summary:
        'These 122 experts sit completely idle for the current token, decoupling total model capacity (31.6B stored weights) from per-token compute (3.2B active parameters).',
      specs: [
        '122 / 128 = 95.3% of routed FFN weights skipped per token',
        'Stateless feed-forward bank: 0 B KV or recurrent token cache',
      ],
    },
    'shared-expert': {
      id: 'shared-expert',
      title: 'Parallel Shared Expert (3,712 = 2×1,856)',
      badge: weightsFp8 ? 'FP8 · Always Active' : 'BF16 · Always Active',
      summary:
        'Runs in parallel beside the routed bank on every token. Shared expert width 3,712 (= 2 × 1,856 in NVIDIA paper, fused in checkpoint) captures common cross-token transformations.',
      specs: [
        'Shared expert width 3,712 (= 2 × 1,856 in NVIDIA paper, fused in checkpoint)',
        'Two non-gated ReLU² weight plates: up_proj (2,688→3,712) + down_proj (3,712→2,688)',
        'Always active (100% of tokens) · overlaps compute with routed bank',
      ],
    },
    'merge-sum': {
      id: 'merge-sum',
      title: 'Weighted Sum Merge (+)',
      badge: '6 Routed + 1 Shared',
      summary:
        'Combines the sigmoid-weighted outputs of the 6 active routed experts with the parallel shared expert output, then adds the result back into the continuous residual stream.',
      specs: [
        'y = Shared(x) + Σ_{i ∈ Top6} g_i · Expert_i(x)',
        '1.30B params/layer × 23 = 29.8B (~94% of 31.6B weights) · 0 B token cache',
      ],
    },
    'residual-out': {
      id: 'residual-out',
      title: 'Residual Stream Merge (+ 2,688)',
      badge: 'Activation · Emerald',
      summary:
        'Adds the combined MoE output back into the 2,688-channel residual stream without allocating any per-token cache memory.',
      specs: [
        'Token cache added: 0 B',
      ],
    },
  };

  const packed = b.pack();
  const fitBounds: AABB = {
    min: [-7.0, -7.8, -2.9],
    max: [9.2, 7.3, 2.9],
  };

  return {
    kind: 'cutaway_moe',
    bayId,
    ...packed,
    fitBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    solidPickBoxes: b.pickBoxes,
    partCatalog,
    moeStats: {
      bayId,
      routedTotal: 128,
      routedActiveCount: 6,
      activeExpertIds,
      sharedExpertCount: 1,
      routedWidth: 1856,
      sharedWidth: 3712,
      matricesPerExpert: 2,
      hasGateProj: false,
      cacheBytes: 0,
      weightsFp8,
    },
  };
}

// GQA Cutaway is implemented in ./gqaCutaway.ts (Stage A+1 Increment 1) and re-exported above.
