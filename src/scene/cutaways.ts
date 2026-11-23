import {
  formatBayId,
  NEMOTRON_SPEC,
  type PrecisionMode,
  type Vec3,
} from '../model/spec';
import type { AABB } from '../render/camera';
import type { PickBox } from '../render/pick';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
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
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_INSTANCE_FLOATS,
  RIBBON_KIND_ARROW_NEG_X,
  RIBBON_KIND_ARROW_NEG_Y,
  RIBBON_KIND_ARROW_POS_X,
  RIBBON_KIND_ARROW_POS_Z,
  RIBBON_KIND_BOX,
  SOLID_INSTANCE_FLOATS,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from './spine';

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

// ============================================================================
// 1. CUTAWAY: MAMBA-2 LAYER (Bay 01 Ordinary | Bay 42 Feeder)
// ============================================================================

export function buildMambaCutaway(opts: {
  bayId: 1 | 42;
  tokens: number;
  precision: PrecisionMode;
  selectedPart: string | null;
  hoveredPart: string | null;
}): CutawaySceneGeometry {
  const { bayId, precision, selectedPart, hoveredPart } = opts;
  const clampedT = Math.max(1, Math.min(16, Math.round(opts.tokens)));
  const bay = NEMOTRON_SPEC.bays[bayId - 1];
  const isFeeder = bay.feeder;
  const isFp8Mode = precision === 'fp8_selective';
  const computeIsFp8 = isFp8Mode && !bay.sensitive; // Bay 01 -> true, Bay 42 -> false!
  const feederPreservedBf16 = isFp8Mode && bay.sensitive; // Bay 42 -> true!

  const b = new CutawayBuilder(bayId);
  const activePart = hoveredPart ?? selectedPart;
  const isPartActive = (id: string) => activePart === id;

  const styleFor = (partId: string, fp8: boolean, goldWhenFp8Mode: boolean) => {
    const act = isPartActive(partId);
    const gold = !act && goldWhenFp8Mode && isFp8Mode;
    return {
      fp8Flag: fp8 ? 1 : 0,
      highlightFlag: act ? 1 : gold ? 2 : 0,
      lineStyle: act ? LINE_STYLE_ACTIVE : gold ? LINE_STYLE_BF16_GOLD : LINE_STYLE_NEUTRAL,
      cssWidth: act ? 1.85 : gold ? 1.45 : 0.95,
      lineAlpha: act ? 1.0 : gold ? 0.96 : fp8 ? 0.36 : 0.62,
    };
  };

  // --- Right Vertical Emerald Residual Stream ---
  const resX = 7.4;
  b.addRibbon({
    cx: resX, cy: -0.3, cz: 0,
    ex: 0.42, ey: 15.2, ez: 0.42,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.34,
  });

  // Residual In (2,688)
  {
    const st = styleFor('residual-in', false, false);
    b.addSolid({
      cx: resX, cy: 6.4, cz: 0,
      ex: 0.85, ey: 1.55, ez: 1.15,
      cellFamily: CELL_ACTIVATION_GREEN,
      phase: clampedT * 0.7 + 1,
      bayId,
      cellScale: 0.22,
      pickPart: 'residual-in',
      ...st,
    });
  }

  // Residual Out (2,688)
  {
    const st = styleFor('residual-out', false, false);
    b.addSolid({
      cx: resX, cy: -7.0, cz: 0,
      ex: 0.85, ey: 1.55, ez: 1.15,
      cellFamily: CELL_ACTIVATION_GREEN,
      phase: clampedT * 0.7 + 9,
      bayId,
      cellScale: 0.22,
      pickPart: 'residual-out',
      ...st,
    });
  }

  // --- Pre-RMSNorm (2,688) + gamma bar ---
  {
    const st = styleFor('rmsnorm', computeIsFp8, feederPreservedBf16);
    b.addSolid({
      cx: 4.5, cy: 5.5, cz: 0,
      ex: 0.56, ey: 1.35, ez: 1.15,
      cellFamily: CELL_MOE_AMBER,
      phase: 2 + clampedT * 0.2,
      bayId,
      cellScale: 0.20,
      pickPart: 'rmsnorm',
      ...st,
    });
    // gamma scale strip beside RMSNorm
    b.addSolid({
      cx: 3.85, cy: 5.5, cz: 0,
      ex: 0.24, ey: 1.35, ez: 1.05,
      cellFamily: CELL_WEIGHT_BLUE,
      phase: 3,
      bayId,
      cellScale: 0.18,
      pickPart: 'rmsnorm',
      ...st,
    });
    // Tap ribbon from Residual In -> RMSNorm
    b.addCurvedRibbon(
      [resX - 0.5, 6.2, 0],
      [6.0, 6.0, 0],
      [4.95, 5.5, 0],
      5,
      0.26,
      RIBBON_COLOR_EMERALD,
      0.40,
      RIBBON_KIND_ARROW_NEG_X,
    );
  }

  // --- In-Proj Weights (2,688 -> 10,304) ---
  {
    const st = styleFor('in-proj', computeIsFp8, feederPreservedBf16);
    b.addSolid({
      cx: 1.0, cy: 5.2, cz: 0,
      ex: 3.2, ey: 1.25, ez: 1.35,
      cellFamily: CELL_WEIGHT_BLUE,
      phase: bayId * 1.3,
      bayId,
      cellScale: 0.24,
      pickPart: 'in-proj',
      ...st,
    });
    // Periwinkle weight arrow from RMSNorm -> In-Proj
    b.addRibbon({
      cx: 3.15, cy: 5.35, cz: 0,
      ex: 0.92, ey: 0.42, ez: 0.36,
      kind: RIBBON_KIND_ARROW_NEG_X,
      colorId: RIBBON_COLOR_PERIWINKLE,
      alpha: computeIsFp8 ? 0.26 : 0.48,
    });
  }

  // --- Branch 1: Gate branch (z, 4,096) bypassing recurrence to final gate ---
  {
    const st = styleFor('gate-branch', false, feederPreservedBf16);
    b.addSolid({
      cx: 4.1, cy: 0.1, cz: -1.45,
      ex: 0.85, ey: 2.35, ez: 0.95,
      cellFamily: CELL_ACTIVATION_GREEN,
      phase: clampedT * 0.9 + 4,
      bayId,
      cellScale: 0.22,
      pickPart: 'gate-branch',
      ...st,
    });
    // Ribbon In-Proj -> Gate z
    b.addCurvedRibbon(
      [2.2, 4.5, -0.45],
      [4.0, 3.6, -1.35],
      [4.1, 1.45, -1.45],
      6,
      0.24,
      RIBBON_COLOR_EMERALD,
      0.38,
      RIBBON_KIND_ARROW_NEG_Y,
    );
    // Ribbon Gate z -> Gate × Group RMSNorm
    b.addCurvedRibbon(
      [4.1, -1.2, -1.45],
      [4.0, -3.5, -1.2],
      [2.5, -4.6, -0.25],
      6,
      0.24,
      RIBBON_COLOR_EMERALD,
      0.40,
      RIBBON_KIND_ARROW_NEG_X,
    );
  }

  // --- Branch 2: Main branch (x, B, C, 6,144) -> Causal Conv1D (k=4) ALWAYS BF16 -> SiLU ---
  {
    // Conv1D NEVER switches to FP8 — persistent gold rim in FP8 Selective!
    const st = styleFor('conv1d', false, true);
    const tapOffsetsZ = [-0.90, -0.30, 0.30, 0.90];
    for (let k = 0; k < 4; k++) {
      b.addSolid({
        cx: -1.55,
        cy: 2.55,
        cz: tapOffsetsZ[k],
        ex: 2.35,
        ey: 0.88,
        ez: 0.36,
        cellFamily: CELL_ACTIVATION_GREEN,
        phase: clampedT + k * 0.65,
        bayId,
        cellScale: 0.19,
        pickPart: 'conv1d',
        ...st,
      });
    }
    // Ribbon In-Proj -> Conv1D
    b.addCurvedRibbon(
      [0.2, 4.5, 0.15],
      [-1.3, 4.0, 0.15],
      [-1.55, 3.15, 0.0],
      5,
      0.24,
      RIBBON_COLOR_EMERALD,
      0.40,
      RIBBON_KIND_ARROW_NEG_Y,
    );

    // SiLU activation strip
    const stSilu = styleFor('silu', false, feederPreservedBf16);
    b.addSolid({
      cx: -1.55,
      cy: 1.05,
      cz: 0,
      ex: 2.25,
      ey: 0.44,
      ez: 1.35,
      cellFamily: CELL_ACTIVATION_GREEN,
      phase: clampedT * 1.1 + 5,
      bayId,
      cellScale: 0.19,
      pickPart: 'silu',
      ...stSilu,
    });
    b.addRibbon({
      cx: -1.55, cy: 1.68, cz: 0,
      ex: 0.36, ey: 0.62, ez: 0.36,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: 0.44,
    });
  }

  // --- Branch 3: Step / Decay branch (Δt, 64 heads) ---
  {
    const st = styleFor('dt-branch', computeIsFp8, feederPreservedBf16);
    b.addSolid({
      cx: -5.45,
      cy: 1.95,
      cz: 0.55,
      ex: 1.05,
      ey: 1.65,
      ez: 0.95,
      cellFamily: CELL_MOE_AMBER,
      phase: clampedT * 0.5 + 7,
      bayId,
      cellScale: 0.20,
      pickPart: 'dt-branch',
      ...st,
    });
    // Ribbon In-Proj -> Δt
    b.addCurvedRibbon(
      [-0.5, 4.8, 0.45],
      [-4.6, 4.4, 0.55],
      [-5.45, 2.95, 0.55],
      7,
      0.22,
      RIBBON_COLOR_EMERALD,
      0.36,
      RIBBON_KIND_ARROW_NEG_Y,
    );
    // Cyan ribbon Δt -> SSM State recurrence
    b.addCurvedRibbon(
      [-5.45, 1.0, 0.55],
      [-5.2, -0.8, 0.65],
      [-4.15, -1.45, 0.45],
      6,
      0.22,
      RIBBON_COLOR_CYAN,
      0.42,
      RIBBON_KIND_ARROW_POS_X,
    );
  }

  // --- Centerpiece: SSM Recurrent State (64 heads = 8 groups × 8 heads, FP32, FIXED SIZE!) ---
  {
    // SSM state is ALWAYS FP32 (never FP8!)
    const st = styleFor('ssm-state', false, feederPreservedBf16);
    const cols = 8; // 8 groups along X
    const rows = 8; // 8 heads per group along Z
    const pitchX = 0.68;
    const pitchZ = 0.46;
    const startX = -1.55 - ((cols - 1) * pitchX) * 0.5;
    const startZ = -((rows - 1) * pitchZ) * 0.5;

    for (let g = 0; g < cols; g++) {
      for (let h = 0; h < rows; h++) {
        const headIdx = g * 8 + h;
        b.addSolid({
          cx: startX + g * pitchX,
          cy: -1.45,
          cz: startZ + h * pitchZ,
          ex: 0.52,
          ey: 1.38,
          ez: 0.30,
          cellFamily: CELL_SSM_CYAN,
          // Hash phase updates IN PLACE as T steps 1 -> 16 while geometry stays 100% fixed!
          phase: clampedT * 1.73 + headIdx * 0.29 + bayId * 0.11,
          bayId,
          cellScale: 0.16,
          pickPart: 'ssm-state',
          ...st,
          cssWidth: isPartActive('ssm-state') ? 1.45 : feederPreservedBf16 ? 1.1 : 0.72,
          lineAlpha: isPartActive('ssm-state') ? 0.95 : feederPreservedBf16 ? 0.80 : 0.44,
        });
      }
    }

    // Downward input arrow from SiLU -> SSM state
    b.addRibbon({
      cx: -1.55, cy: 0.05, cz: 0,
      ex: 0.42, ey: 1.05, ez: 0.42,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: 0.44,
    });

    // Curved in-place recurrence feedback loop arrow: h_t = A·h_{t-1} + B·x_t
    // Arcs out from the front-bottom of the 64-head array, loops left & upward, and re-enters top-front
    b.addCurvedRibbon(
      [-1.55, -2.35, 2.05],
      [-4.95, -2.55, 2.35],
      [-5.10, -1.45, 2.15],
      7,
      0.24,
      RIBBON_COLOR_CYAN,
      0.44,
    );
    b.addCurvedRibbon(
      [-5.10, -1.45, 2.15],
      [-5.05, -0.15, 2.15],
      [-3.55, -0.55, 1.85],
      7,
      0.24,
      RIBBON_COLOR_EMERALD,
      0.46,
      RIBBON_KIND_ARROW_POS_X,
    );
  }

  // --- Output: Gate × Group RMSNorm (8 groups) ---
  {
    const st = styleFor('group-norm', computeIsFp8, feederPreservedBf16);
    for (let g = 0; g < 8; g++) {
      const gx = -1.65 + g * 0.50;
      b.addSolid({
        cx: gx,
        cy: -4.60,
        cz: 0,
        ex: 0.36,
        ey: 0.96,
        ez: 1.05,
        cellFamily: CELL_MOE_AMBER,
        phase: clampedT * 0.4 + g * 1.1,
        bayId,
        cellScale: 0.17,
        pickPart: 'group-norm',
        ...st,
      });
    }
    // Ribbon SSM -> Group RMSNorm
    b.addCurvedRibbon(
      [-1.55, -2.35, 0],
      [-1.0, -3.55, 0],
      [0.10, -4.05, 0],
      5,
      0.28,
      RIBBON_COLOR_EMERALD,
      0.42,
      RIBBON_KIND_ARROW_NEG_Y,
    );
  }

  // --- Out-Proj Weights (4,096 -> 2,688) ---
  {
    const st = styleFor('out-proj', computeIsFp8, feederPreservedBf16);
    b.addSolid({
      cx: 3.45,
      cy: -6.15,
      cz: 0,
      ex: 2.65,
      ey: 1.18,
      ez: 1.25,
      cellFamily: CELL_WEIGHT_BLUE,
      phase: bayId * 2.9,
      bayId,
      cellScale: 0.23,
      pickPart: 'out-proj',
      ...st,
    });
    // Periwinkle arrow Group RMSNorm -> Out-Proj
    b.addCurvedRibbon(
      [1.95, -4.75, 0],
      [2.85, -5.25, 0],
      [3.45, -5.45, 0],
      4,
      0.28,
      RIBBON_COLOR_PERIWINKLE,
      computeIsFp8 ? 0.28 : 0.50,
      RIBBON_KIND_ARROW_NEG_Y,
    );
    // Emerald + merge ribbon Out-Proj -> Residual Out
    b.addCurvedRibbon(
      [4.85, -6.25, 0],
      [5.95, -6.65, 0],
      [resX - 0.55, -7.0, 0],
      5,
      0.30,
      RIBBON_COLOR_EMERALD,
      0.48,
      RIBBON_KIND_ARROW_POS_X,
    );
  }

  // --- Black-Pill 3D Slab Landmarks ---
  b.landmarks.push(
    { id: 'm-res-in', label: 'Residual In (2,688)', world: [resX + 0.55, 7.15, 0], partId: 'residual-in' },
    { id: 'm-rmsnorm', label: 'RMSNorm + γ', world: [4.5, 6.45, 0], partId: 'rmsnorm' },
    {
      id: 'm-in-proj',
      label: computeIsFp8 ? 'In-Proj · FP8' : 'In-Proj (2,688 → 10,304)',
      world: [1.0, 6.15, 0],
      partId: 'in-proj',
      tone: feederPreservedBf16 ? 'gold' : 'default',
    },
    { id: 'm-gate', label: 'Gate Branch (z · 4,096)', world: [4.65, 1.35, -1.45], partId: 'gate-branch' },
    {
      id: 'm-conv1d',
      label: isFp8Mode ? 'Causal Conv1D (k=4) · Always BF16' : 'Causal Conv1D (k=4, 6,144)',
      world: [-1.55, 3.35, 0.95],
      partId: 'conv1d',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    { id: 'm-dt', label: 'Δt Step / Decay (64 heads)', world: [-5.65, 3.05, 0.55], partId: 'dt-branch' },
    {
      id: 'm-ssm',
      label: `SSM State (64×64×128 · FP32 · T=${clampedT})`,
      world: [-1.55, -0.55, 2.05],
      partId: 'ssm-state',
      tone: 'cyan',
    },
    {
      id: 'm-loop',
      label: 'h_t = A·h_{t−1} + B·x_t (fixed size)',
      world: [-5.15, -2.55, 2.35],
      partId: 'ssm-state',
      tone: 'muted',
    },
    { id: 'm-gnorm', label: 'Gate × Group RMSNorm (8 grps)', world: [-1.75, -5.45, 0.65], partId: 'group-norm' },
    {
      id: 'm-out-proj',
      label: computeIsFp8 ? 'Out-Proj (4,096→2,688) · FP8' : 'Out-Proj (4,096 → 2,688)',
      world: [3.45, -7.10, 0.65],
      partId: 'out-proj',
      tone: feederPreservedBf16 ? 'gold' : 'default',
    },
    { id: 'm-res-out', label: '+ Residual Out (2,688)', world: [resX + 0.55, -7.85, 0], partId: 'residual-out' },
  );

  if (isFeeder) {
    b.landmarks.push({
      id: 'm-feeder-note',
      label: 'Bay 42 Feeder · Preserved in BF16 (NVIDIA §4.2)',
      world: [-3.9, 5.45, 0],
      tone: 'gold',
    });
  }

  const partCatalog: Record<string, CutawayPartInfo> = {
    'residual-in': {
      id: 'residual-in',
      title: 'Residual Stream Input (2,688)',
      badge: 'Activation · Emerald',
      summary:
        'Incoming hidden state from the continuous 2,688-channel residual highway. A copy taps left into pre-RMSNorm while the main residual stream passes straight down to the final + merge.',
      specs: [
        'Shape: [batch, 2,688] per token',
        'Pre-RMSNorm architecture keeps the residual path unscaled',
      ],
    },
    rmsnorm: {
      id: 'rmsnorm',
      title: 'Pre-Layer RMSNorm + γ',
      badge: computeIsFp8 ? 'FP8 Selective' : 'BF16',
      summary:
        'Normalizes the 2,688-channel hidden vector by its root-mean-square magnitude and applies learned per-channel scale vector γ before entering the Mamba-2 input projection.',
      specs: [
        'Input / Output width: 2,688 · No mean subtraction, no bias',
        `Bay ${formatBayId(bayId)} compute: ${computeIsFp8 ? 'FP8 quantized' : isFeeder && isFp8Mode ? 'BF16 preserved (Feeder §4.2)' : 'BF16'}`,
      ],
    },
    'in-proj': {
      id: 'in-proj',
      title: 'Mamba-2 Input Projection',
      badge: computeIsFp8 ? 'FP8 Weights' : feederPreservedBf16 ? 'BF16 Preserved (§4.2)' : 'BF16 Weights',
      summary:
        'Single fused linear projection mapping 2,688 channels into 10,304 outputs, immediately split into three parallel branches: Gate z (4,096), Conv1D stream x/B/C (6,144), and per-head step Δt (64).',
      specs: [
        'Dimensions: 2,688 → 4,096 (z) + 4,096 (x) + 1,024 (B) + 1,024 (C) + 64 (Δt) = 10,304',
        isFeeder
          ? 'Bay 42 precedes Bay 43 GQA: NVIDIA §4.2 keeps this entire Mamba-2 block in BF16.'
          : 'Bay 01 is an ordinary Mamba-2 bay: In-Proj switches to FP8 in FP8 Selective mode.',
      ],
    },
    'gate-branch': {
      id: 'gate-branch',
      title: 'Bypass Gate Branch (z · 4,096)',
      badge: 'Multiplicative Gate',
      summary:
        'Carries 4,096 gated channels around the state-space recurrence directly into the Group RMSNorm output gate, modulating the SSM output before down-projection.',
      specs: [
        'Width: 4,096 (64 heads × 64 head_dim)',
        'Bypasses Conv1D and SSM recurrence completely',
      ],
    },
    conv1d: {
      id: 'conv1d',
      title: 'Causal Conv1D (k=4, 6,144)',
      badge: 'Always BF16',
      summary:
        'Short depthwise causal convolution across the last 4 token slots (t−3, t−2, t−1, t) over the 6,144-channel (x, B, C) stream. Supplies local token ordering and stays BF16 in every precision mode.',
      specs: [
        'State shape: 6,144 channels × 4 slots = 49,152 B (48 KiB) per Mamba-2 bay',
        'Precision override: ALWAYS BF16 even on quantized FP8 bays (NVIDIA §4.2)',
      ],
    },
    silu: {
      id: 'silu',
      title: 'SiLU Nonlinearity',
      badge: '6,144 channels',
      summary:
        'Applies elementwise SiLU (x · σ(x)) to the Causal Conv1D output before splitting x (4,096), B (8 groups × 128), and C (8 groups × 128) into the 64-head state-space update.',
      specs: [
        'Splits 6,144 → x (4,096) + B (1,024) + C (1,024)',
        'Zero persistent state storage',
      ],
    },
    'dt-branch': {
      id: 'dt-branch',
      title: 'Step / Decay Branch (Δt · 64 heads)',
      badge: '64 SSM Heads',
      summary:
        'Computes per-head discretization step Δt (softplus + bias) controlling how strongly each of the 64 SSM heads decays prior state h_{t−1} versus writes new input B·x_t.',
      specs: [
        '64 scalar decay gates per token (1 per Mamba-2 head)',
        'Discretizes continuous decay A into A_bar = exp(Δt · A)',
      ],
    },
    'ssm-state': {
      id: 'ssm-state',
      title: 'SSM Recurrent State (64 heads · FP32)',
      badge: '2.05 MiB Fixed · Never Grows',
      summary:
        'Eight groups × eight heads of 64 × 128 FP32 state matrices updated in place via h_t = A·h_{t−1} + B·x_t. Scrubbing T = 1 → 16 refreshes cell values in place while physical memory stays strictly constant.',
      specs: [
        'Per-layer SSM state: 64 × 64 × 128 × 4 B = 2,097,152 B (2.00 MiB) + 48 KiB Conv1D = 2.05 MiB',
        'All 23 Mamba-2 layers combined: 49,364,992 B (47.1 MiB) fixed at any context length',
        'Storage dtype: FP32 across both BF16 and FP8 Selective modes',
      ],
    },
    'group-norm': {
      id: 'group-norm',
      title: 'Gate × Group RMSNorm (8 Groups)',
      badge: '8 Groups × 512',
      summary:
        'Multiplies the 4,096-channel SSM output by SiLU(z) from the bypass gate branch and normalizes independently across 8 groups of 512 channels.',
      specs: [
        '8 normalization groups matching the 8 B/C Mamba-2 groups',
        'Stabilizes recurrent magnitudes prior to Out-Proj',
      ],
    },
    'out-proj': {
      id: 'out-proj',
      title: 'Mamba-2 Output Projection (4,096 → 2,688)',
      badge: computeIsFp8 ? 'FP8 Weights' : feederPreservedBf16 ? 'BF16 Preserved (§4.2)' : 'BF16 Weights',
      summary:
        'Projects the gated 4,096-channel Mamba-2 output back down to the 2,688-channel model hidden dimension before adding into the continuous emerald residual stream.',
      specs: [
        'Matrix shape: 4,096 × 2,688 (~11.0M parameters)',
        isFeeder
          ? 'Bay 42 Feeder: preserved in BF16 alongside Bay 43 GQA.'
          : 'Bay 01: quantized to FP8 in FP8 Selective mode.',
      ],
    },
    'residual-out': {
      id: 'residual-out',
      title: 'Residual Stream Merge (+ 2,688)',
      badge: 'Activation · Emerald',
      summary:
        'Adds the Mamba-2 layer delta back into the untouched 2,688-channel residual stream and passes straight down to the next layer.',
      specs: [
        'Operation: h_out = h_in + Mamba2(RMSNorm(h_in))',
      ],
    },
  };

  const packed = b.pack();
  const fitBounds: AABB = {
    min: [-6.4, -7.9, -2.3],
    max: [8.2, 7.3, 2.6],
  };

  return {
    kind: 'cutaway_mamba',
    bayId,
    ...packed,
    fitBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    solidPickBoxes: b.pickBoxes,
    partCatalog,
    mambaStats: {
      bayId,
      isFeeder,
      conv1dTapCount: 4,
      conv1dFp8: false,
      ssmHeadCount: 64,
      ssmStateFp8: false,
      inProjFp8: computeIsFp8,
      outProjFp8: computeIsFp8,
    },
  };
}

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

// ============================================================================
// 3. CUTAWAY: GQA ATTENTION LAYER (Bay 06 | Bay 43 — 32Q / 2KV, NoPE, Live KV)
// ============================================================================

export function buildGqaCutaway(opts: {
  bayId: 6 | 43;
  tokens: number;
  precision: PrecisionMode;
  selectedPart: string | null;
  hoveredPart: string | null;
}): CutawaySceneGeometry {
  const { bayId, precision, selectedPart, hoveredPart } = opts;
  const clampedT = Math.max(1, Math.min(16, Math.round(opts.tokens)));
  const isFp8Mode = precision === 'fp8_selective';
  // NVIDIA §4.2: ALL 6 GQA compute bodies stay BF16 (gold border in FP8 Selective),
  // while ONLY the KV-cache token slices switch to FP8!
  const computeFp8 = false;
  const kvCacheFp8 = isFp8Mode;

  const b = new CutawayBuilder(bayId);
  const activePart = hoveredPart ?? selectedPart;
  const isPartActive = (id: string) => activePart === id;

  const computeStyle = (partId: string) => {
    const act = isPartActive(partId);
    const gold = !act && isFp8Mode;
    return {
      fp8Flag: 0,
      highlightFlag: act ? 1 : gold ? 2 : 0,
      lineStyle: act ? LINE_STYLE_ACTIVE : gold ? LINE_STYLE_BF16_GOLD : LINE_STYLE_NEUTRAL,
      cssWidth: act ? 1.85 : gold ? 1.40 : 0.95,
      lineAlpha: act ? 1.0 : gold ? 0.95 : 0.62,
    };
  };

  // --- Right Vertical Emerald Residual Stream ---
  const resX = 8.0;
  b.addRibbon({
    cx: resX, cy: -0.2, cz: 0,
    ex: 0.42, ey: 15.2, ez: 0.42,
    kind: RIBBON_KIND_BOX,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.34,
  });

  // Residual In (2,688)
  b.addSolid({
    cx: resX, cy: 6.4, cz: 0,
    ex: 0.85, ey: 1.50, ez: 1.15,
    cellFamily: CELL_ACTIVATION_GREEN,
    fp8Flag: 0,
    phase: clampedT * 0.8 + 1,
    bayId,
    cellScale: 0.22,
    highlightFlag: isPartActive('residual-in') ? 1 : 0,
    lineStyle: isPartActive('residual-in') ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: isPartActive('residual-in') ? 1.85 : 0.95,
    lineAlpha: isPartActive('residual-in') ? 1.0 : 0.62,
    pickPart: 'residual-in',
  });

  // Residual Out (2,688)
  b.addSolid({
    cx: resX, cy: -6.9, cz: 0,
    ex: 0.85, ey: 1.50, ez: 1.15,
    cellFamily: CELL_ACTIVATION_GREEN,
    fp8Flag: 0,
    phase: clampedT * 0.8 + 13,
    bayId,
    cellScale: 0.22,
    highlightFlag: isPartActive('residual-out') ? 1 : 0,
    lineStyle: isPartActive('residual-out') ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    cssWidth: isPartActive('residual-out') ? 1.85 : 0.95,
    lineAlpha: isPartActive('residual-out') ? 1.0 : 0.62,
    pickPart: 'residual-out',
  });

  // Pre-RMSNorm (2,688) — stays BF16 (gold rim in FP8 Selective)
  {
    const st = computeStyle('rmsnorm');
    b.addSolid({
      cx: 5.2, cy: 5.55, cz: 0,
      ex: 0.56, ey: 1.32, ez: 1.12,
      cellFamily: CELL_MOE_AMBER,
      phase: 2 + clampedT * 0.2,
      bayId,
      cellScale: 0.20,
      pickPart: 'rmsnorm',
      ...st,
    });
    b.addCurvedRibbon(
      [resX - 0.5, 6.2, 0],
      [6.6, 6.0, 0],
      [5.6, 5.55, 0],
      5,
      0.26,
      RIBBON_COLOR_EMERALD,
      0.42,
      RIBBON_KIND_ARROW_NEG_X,
    );
  }

  // --- Stacked Q / K / V Projection Plates (ALL BF16, gold rim in FP8 Selective) ---
  // Q Proj (2,688 -> 32 × 128 = 4,096): 2 groups × 16 stacked query slabs!
  {
    const stQ = computeStyle('q-proj');
    const groupCentersX = [2.65, 0.70]; // Group 0 (Q0..Q15), Group 1 (Q16..Q31)
    for (let grp = 0; grp < 2; grp++) {
      const gx = groupCentersX[grp];
      for (let h = 0; h < 16; h++) {
        const hz = -1.50 + h * 0.20;
        b.addSolid({
          cx: gx,
          cy: 3.10,
          cz: hz,
          ex: 1.35,
          ey: 1.18,
          ez: 0.13,
          cellFamily: CELL_WEIGHT_BLUE,
          phase: bayId * 0.7 + grp * 16 + h * 0.25,
          bayId,
          cellScale: 0.18,
          pickPart: 'q-proj',
          ...stQ,
          cssWidth: isPartActive('q-proj') ? 1.35 : isFp8Mode ? 1.05 : 0.72,
          lineAlpha: isPartActive('q-proj') ? 0.96 : isFp8Mode ? 0.84 : 0.45,
        });
      }
    }

    // Periwinkle matmul arrow RMSNorm -> Q Proj
    b.addCurvedRibbon(
      [4.85, 5.35, 0],
      [3.85, 4.75, 0],
      [2.85, 3.85, 0],
      5,
      0.28,
      RIBBON_COLOR_PERIWINKLE,
      0.48,
      RIBBON_KIND_ARROW_NEG_Y,
    );
  }

  // K Proj (2,688 -> 2 × 128 = 256) and V Proj (2,688 -> 2 × 128 = 256) — visibly 16× thinner!
  {
    const stK = computeStyle('k-proj');
    const stV = computeStyle('v-proj');
    const kvHeadX = [-1.65, -4.15]; // KV Head 0, KV Head 1

    for (let kh = 0; kh < 2; kh++) {
      // K head plate (emerald-tinted weight or blue weight)
      b.addSolid({
        cx: kvHeadX[kh] + 0.36,
        cy: 3.10,
        cz: -0.25,
        ex: 0.54,
        ey: 1.18,
        ez: 0.48,
        cellFamily: CELL_WEIGHT_BLUE,
        phase: bayId + kh * 5 + 1,
        bayId,
        cellScale: 0.17,
        pickPart: 'k-proj',
        ...stK,
      });
      // V head plate
      b.addSolid({
        cx: kvHeadX[kh] - 0.36,
        cy: 3.10,
        cz: -0.25,
        ex: 0.54,
        ey: 1.18,
        ez: 0.48,
        cellFamily: CELL_WEIGHT_BLUE,
        phase: bayId + kh * 5 + 9,
        bayId,
        cellScale: 0.17,
        pickPart: 'v-proj',
        ...stV,
      });
    }

    // Periwinkle arrows fanning from RMSNorm across to K/V projections
    b.addCurvedRibbon(
      [4.85, 5.55, -0.25],
      [0.8, 5.35, -0.25],
      [-1.65, 3.90, -0.25],
      7,
      0.22,
      RIBBON_COLOR_PERIWINKLE,
      0.44,
      RIBBON_KIND_ARROW_NEG_Y,
    );
    b.addCurvedRibbon(
      [4.85, 5.55, -0.25],
      [-0.8, 5.65, -0.25],
      [-4.15, 3.90, -0.25],
      8,
      0.22,
      RIBBON_COLOR_PERIWINKLE,
      0.44,
      RIBBON_KIND_ARROW_NEG_Y,
    );
  }

  // --- Twin Live KV-Cache Token Stacks (KV Head 0 & KV Head 1, t = 1..T along +Z) ---
  const kvHeadCentersX = [-1.65, -4.15];
  const kvTrackStartZ = 1.15;
  const kvSlicePitch = 0.44;
  const kvSliceDepth = 0.30;
  const kvTrackLen = 16 * kvSlicePitch; // 7.04
  const kvCenterY = 0.45;

  const kvSliceZ = (tok: number) => kvTrackStartZ + (tok - 1) * kvSlicePitch + kvSliceDepth * 0.5;
  const kvActiveTipZ = kvSliceZ(clampedT);

  for (let kh = 0; kh < 2; kh++) {
    const partId = kh === 0 ? 'kv-head-0' : 'kv-head-1';
    const act = isPartActive(partId);
    const hx = kvHeadCentersX[kh];

    // Two ghost 16-slot capacity rails per KV head (K rail + V rail)
    for (const dx of [0.34, -0.34]) {
      b.addSolid(
        {
          cx: hx + dx,
          cy: kvCenterY - 0.56,
          cz: kvTrackStartZ + kvTrackLen * 0.5,
          ex: 0.12,
          ey: 0.08,
          ez: kvTrackLen,
          cellFamily: CELL_NEUTRAL,
          fp8Flag: 0,
          phase: 0,
          bayId,
          cellScale: 0.5,
          highlightFlag: 0,
          lineStyle: LINE_STYLE_GUIDE,
          cssWidth: 0.85,
          lineAlpha: 0.48,
          pickPart: partId,
        },
        true,
      );
    }

    // Periwinkle write arrows from K/V projections into KV Head track
    b.addCurvedRibbon(
      [hx, 2.40, -0.10],
      [hx, 1.45, 0.45],
      [hx, kvCenterY + 0.55, kvTrackStartZ + 0.10],
      5,
      0.22,
      RIBBON_COLOR_PERIWINKLE,
      0.42,
      RIBBON_KIND_ARROW_POS_Z,
    );

    // Explicit individually separable live token slices t = 1..T
    // Each of the 2 KV heads has T paired K (emerald) + V (cyan) slices -> 2 * T live KV slice units!
    for (let t = 1; t <= clampedT; t++) {
      const cz = kvSliceZ(t);
      const isNewest = t === clampedT;

      // K slice (emerald in BF16, frosted FP8 in FP8 Selective!)
      b.addSolid({
        cx: hx + 0.34,
        cy: kvCenterY,
        cz,
        ex: 0.50,
        ey: 0.96,
        ez: kvSliceDepth,
        cellFamily: CELL_ACTIVATION_GREEN,
        fp8Flag: kvCacheFp8 ? 1 : 0,
        phase: t * 1.15 + kh * 3.1 + bayId * 0.2,
        bayId,
        cellScale: 0.17,
        highlightFlag: act ? 1 : 0,
        lineStyle: act ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
        cssWidth: act ? 1.45 : isNewest ? 1.1 : 0.85,
        lineAlpha: act ? 0.95 : kvCacheFp8 ? 0.42 : 0.60,
        pickPart: partId,
      });

      // V slice (cyan in BF16, frosted FP8 in FP8 Selective!)
      b.addSolid({
        cx: hx - 0.34,
        cy: kvCenterY,
        cz,
        ex: 0.50,
        ey: 0.96,
        ez: kvSliceDepth,
        cellFamily: CELL_SSM_CYAN,
        fp8Flag: kvCacheFp8 ? 1 : 0,
        phase: t * 1.41 + kh * 5.7 + bayId * 0.3,
        bayId,
        cellScale: 0.17,
        highlightFlag: act ? 1 : 0,
        lineStyle: act ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
        cssWidth: act ? 1.45 : isNewest ? 1.1 : 0.85,
        lineAlpha: act ? 0.95 : kvCacheFp8 ? 0.42 : 0.60,
        pickPart: partId,
      });
    }
  }

  // Depth scale guide wire & ticks (1, 8, 16, active T) cleanly offset on the far-left of KV Head 1
  const scaleX = -5.35;
  const scaleY = kvCenterY - 0.56;
  b.addGuideLine(scaleX, scaleY, kvTrackStartZ, scaleX, scaleY, kvTrackStartZ + kvTrackLen, 0.95, LINE_STYLE_GUIDE, 0.60);
  const tickSet = new Set<number>([1, 8, 16, clampedT]);
  for (const tk of Array.from(tickSet).sort((a, c) => a - c)) {
    const z = kvSliceZ(tk);
    b.addGuideLine(scaleX - 0.16, scaleY, z, scaleX + 0.16, scaleY, z, tk === clampedT ? 1.45 : 0.95, tk === clampedT ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE, 0.82);
    b.kvDepthTicks.push({
      token: tk,
      world: [scaleX - 0.32, scaleY, z],
    });
  }

  // --- Translucent Emerald Curved Attention Fan Ribbons (16:1 GQA Grouping!) ---
  // Group 0 (16 Q heads at x=+2.65) sweeps across to KV Head 0's 1..T history (x=-1.65)
  b.addCurvedRibbon(
    [2.65, 2.45, 0.15],
    [1.15, 2.05, Math.min(3.5, kvActiveTipZ * 0.75)],
    [-1.05, kvCenterY + 0.58, kvActiveTipZ * 0.65 + 0.4],
    7,
    0.26,
    RIBBON_COLOR_EMERALD,
    0.42,
    RIBBON_KIND_ARROW_NEG_X,
  );
  // Group 1 (16 Q heads at x=+0.70) sweeps across to KV Head 1's 1..T history (x=-4.15)
  b.addCurvedRibbon(
    [0.70, 2.45, 0.55],
    [-1.45, 1.95, Math.min(4.8, kvActiveTipZ * 0.95 + 0.5)],
    [-3.55, kvCenterY + 0.58, kvActiveTipZ * 0.8 + 0.45],
    8,
    0.26,
    RIBBON_COLOR_EMERALD,
    0.42,
    RIBBON_KIND_ARROW_NEG_X,
  );

  // --- Attn Scores (32 × T) -> Softmax (Width physically scales with T!) ---
  {
    const stScores = computeStyle('attn-scores');
    const scoresWidth = 0.95 + clampedT * 0.24; // visibly expands as T grows 1 -> 16!
    b.addSolid({
      cx: -1.85,
      cy: -2.45,
      cz: 0.45,
      ex: scoresWidth,
      ey: 1.25,
      ez: 0.88,
      cellFamily: CELL_ACTIVATION_GREEN,
      phase: clampedT * 1.3 + bayId,
      bayId,
      cellScale: 0.18,
      pickPart: 'attn-scores',
      ...stScores,
    });

    // Softmax amber agg strip beside Attn Scores
    b.addSolid({
      cx: -1.85 + scoresWidth * 0.5 + 0.42,
      cy: -2.45,
      cz: 0.45,
      ex: 0.44,
      ey: 1.25,
      ez: 0.88,
      cellFamily: CELL_MOE_AMBER,
      phase: clampedT * 0.9 + 4,
      bayId,
      cellScale: 0.18,
      pickPart: 'attn-scores',
      ...stScores,
    });

    // Ribbons from both KV heads -> Attn Scores
    b.addCurvedRibbon(
      [-1.65, kvCenterY - 0.55, kvActiveTipZ * 0.5 + 0.5],
      [-1.65, -0.95, 0.95],
      [-1.45, -1.70, 0.55],
      5,
      0.24,
      RIBBON_COLOR_EMERALD,
      0.40,
      RIBBON_KIND_ARROW_NEG_Y,
    );
    b.addCurvedRibbon(
      [-4.15, kvCenterY - 0.55, kvActiveTipZ * 0.5 + 0.5],
      [-3.55, -1.05, 0.95],
      [-2.35, -1.70, 0.55],
      6,
      0.24,
      RIBBON_COLOR_CYAN,
      0.40,
      RIBBON_KIND_ARROW_NEG_Y,
    );
  }

  // --- Weighted V Out (32 × 128 = 4,096) -> O Proj (4,096 -> 2,688) ---
  {
    const stVOut = computeStyle('weighted-v');
    b.addSolid({
      cx: 1.65,
      cy: -4.25,
      cz: 0.15,
      ex: 2.35,
      ey: 1.10,
      ez: 1.05,
      cellFamily: CELL_ACTIVATION_GREEN,
      phase: clampedT * 0.7 + 19,
      bayId,
      cellScale: 0.21,
      pickPart: 'weighted-v',
      ...stVOut,
    });

    // Ribbon Softmax -> Weighted V Out
    b.addCurvedRibbon(
      [-0.15, -2.95, 0.45],
      [0.65, -3.45, 0.30],
      [1.35, -3.62, 0.15],
      5,
      0.26,
      RIBBON_COLOR_EMERALD,
      0.44,
      RIBBON_KIND_ARROW_NEG_Y,
    );

    const stO = computeStyle('o-proj');
    b.addSolid({
      cx: 4.15,
      cy: -6.05,
      cz: 0,
      ex: 2.65,
      ey: 1.22,
      ez: 1.25,
      cellFamily: CELL_WEIGHT_BLUE,
      phase: bayId * 3.1,
      bayId,
      cellScale: 0.23,
      pickPart: 'o-proj',
      ...stO,
    });

    // Periwinkle arrow Weighted V -> O Proj
    b.addCurvedRibbon(
      [2.65, -4.75, 0.10],
      [3.55, -5.15, 0.05],
      [4.05, -5.35, 0],
      4,
      0.28,
      RIBBON_COLOR_PERIWINKLE,
      0.50,
      RIBBON_KIND_ARROW_NEG_Y,
    );

    // Emerald + merge ribbon O Proj -> Residual Out
    b.addCurvedRibbon(
      [5.55, -6.15, 0],
      [6.55, -6.55, 0],
      [resX - 0.55, -6.9, 0],
      5,
      0.30,
      RIBBON_COLOR_EMERALD,
      0.48,
      RIBBON_KIND_ARROW_POS_X,
    );
  }

  // --- Black-Pill 3D Slab Labels ---
  b.landmarks.push(
    { id: 'g-res-in', label: 'Residual In (2,688)', world: [resX + 0.55, 7.15, 0], partId: 'residual-in' },
    {
      id: 'g-nope',
      label: 'NoPE · No Positional Embeddings (Order from Mamba-2)',
      world: [2.1, 6.75, 0],
      partId: 'rmsnorm',
      tone: 'cyan',
    },
    {
      id: 'g-qproj',
      label: isFp8Mode ? 'Q Proj (32 Heads = 2×16) · BF16' : 'Q Proj (32 Heads = 2×16 · 4,096)',
      world: [1.85, 4.15, -1.15],
      partId: 'q-proj',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    {
      id: 'g-kvproj',
      label: isFp8Mode ? 'K & V Proj (2 KV Heads · 16× Thinner) · BF16' : 'K & V Proj (2 KV Heads · 16× Thinner)',
      world: [-2.90, 4.15, -0.45],
      partId: 'k-proj',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    {
      id: 'g-kv0',
      label: kvCacheFp8 ? `KV Head 0 · FP8 (${clampedT} tok)` : `KV Head 0 · BF16 (${clampedT} tok)`,
      world: [-1.45, kvCenterY + 1.35, kvActiveTipZ + 0.2],
      partId: 'kv-head-0',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },
    {
      id: 'g-kv1',
      label: kvCacheFp8 ? `KV Head 1 · FP8 (${clampedT} tok)` : `KV Head 1 · BF16 (${clampedT} tok)`,
      world: [-4.15, kvCenterY + 1.35, kvActiveTipZ + 0.2],
      partId: 'kv-head-1',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },
    {
      id: 'g-scores',
      label: `Attn Scores (32 × ${clampedT}) → Softmax`,
      world: [-1.85, -3.55, 0.95],
      partId: 'attn-scores',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    { id: 'g-vout', label: 'Weighted V Out (4,096)', world: [1.65, -5.20, 0.85], partId: 'weighted-v' },
    {
      id: 'g-oproj',
      label: isFp8Mode ? 'O Proj (4,096→2,688) · BF16' : 'O Proj (4,096 → 2,688)',
      world: [4.15, -7.05, 0.75],
      partId: 'o-proj',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    { id: 'g-res-out', label: '+ Residual Out (2,688)', world: [resX + 0.55, -7.75, 0], partId: 'residual-out' },
  );

  const perBayKiB = ((kvCacheFp8 ? 512 : 1024) * clampedT) / 1024;
  const perBaySlope = kvCacheFp8 ? '+0.5 KiB/tok (FP8)' : '+1 KiB/tok (BF16)';

  const partCatalog: Record<string, CutawayPartInfo> = {
    'residual-in': {
      id: 'residual-in',
      title: 'Residual Stream Input (2,688)',
      badge: `Bay ${formatBayId(bayId)} · GQA`,
      summary:
        'Incoming 2,688-channel hidden state already enriched with sequential token ordering by the preceding Mamba-2 feeder bay.',
      specs: [
        `Preceding BF16 feeder: Bay ${formatBayId(bayId - 1)} (Mamba-2)`,
        'NoPE: no RoPE or learned positional embedding needed here',
      ],
    },
    rmsnorm: {
      id: 'rmsnorm',
      title: 'Pre-Attention RMSNorm · NoPE',
      badge: isFp8Mode ? 'BF16 Preserved (§4.2)' : 'BF16',
      summary:
        'Normalizes the 2,688-channel input without applying RoPE. Because Nemotron 3 Nano uses NoPE (No Positional Embeddings), token order is supplied organically by the preceding Mamba-2 causal layers.',
      specs: [
        'NoPE improves long-context extrapolation without rotary frequency decay',
        'Compute stays BF16 in FP8 Selective mode',
      ],
    },
    'q-proj': {
      id: 'q-proj',
      title: 'Q Projection (32 Heads = 2 Groups × 16)',
      badge: isFp8Mode ? 'BF16 Preserved (§4.2)' : 'BF16 Weights',
      summary:
        'Projects 2,688 hidden channels into 32 query heads of dimension 128 (4,096 total), organized into 2 groups of 16 query heads that share the 2 KV heads (16:1 GQA ratio).',
      specs: [
        'Group 0 (Q0..Q15) attends to KV Head 0 · Group 1 (Q16..Q31) attends to KV Head 1',
        'All Q/K/V/O attention weights & activations remain BF16 in FP8 Selective mode',
      ],
    },
    'k-proj': {
      id: 'k-proj',
      title: 'K Projection (2 KV Heads · 16× Thinner)',
      badge: isFp8Mode ? 'Compute BF16 · Cache FP8' : 'BF16',
      summary:
        'Projects 2,688 channels down to just 2 key heads × 128 dim = 256 values (16× thinner than Q), appending one new key vector per head onto the live KV-cache tracks.',
      specs: [
        'Dimensions: 2,688 → 2 × 128 = 256',
        'K projection matmul stays BF16; stored K cache switches to FP8 in FP8 Selective',
      ],
    },
    'v-proj': {
      id: 'v-proj',
      title: 'V Projection (2 KV Heads · 16× Thinner)',
      badge: isFp8Mode ? 'Compute BF16 · Cache FP8' : 'BF16',
      summary:
        'Projects 2,688 channels down to 2 value heads × 128 dim = 256 values, appending one new value vector per head onto the live KV-cache tracks.',
      specs: [
        'Dimensions: 2,688 → 2 × 128 = 256',
        'V projection matmul stays BF16; stored V cache switches to FP8 in FP8 Selective',
      ],
    },
    'kv-head-0': {
      id: 'kv-head-0',
      title: `KV Head 0 Stack (t = 1 … ${clampedT})`,
      badge: kvCacheFp8 ? 'FP8 Cache (Halved)' : 'BF16 Cache',
      summary:
        'Live token history stack for KV Head 0 (paired emerald K0 + cyan V0 vectors of dim 128) shared by Group 0’s 16 query heads. Scrubbing T physically appends or removes slices.',
      specs: [
        `This GQA bay total (2 KV heads): ${perBayKiB} KiB at T=${clampedT} (${perBaySlope})`,
        'In FP8 Selective mode, ONLY the KV-cache token slices switch to FP8 while Q/K/V/O compute stays BF16',
      ],
    },
    'kv-head-1': {
      id: 'kv-head-1',
      title: `KV Head 1 Stack (t = 1 … ${clampedT})`,
      badge: kvCacheFp8 ? 'FP8 Cache (Halved)' : 'BF16 Cache',
      summary:
        'Live token history stack for KV Head 1 (paired emerald K1 + cyan V1 vectors of dim 128) shared by Group 1’s 16 query heads.',
      specs: [
        `This GQA bay total (2 KV heads): ${perBayKiB} KiB at T=${clampedT} (${perBaySlope})`,
        'All 6 GQA bays combined: +6 KiB/tok (BF16) or +3 KiB/tok (FP8 Selective)',
      ],
    },
    'attn-scores': {
      id: 'attn-scores',
      title: `Attention Scores (32 × ${clampedT}) → Softmax`,
      badge: isFp8Mode ? 'BF16 Preserved (§4.2)' : 'BF16',
      summary:
        'Computes scaled dot products Q·K^T / √128 across all 32 query heads against all T cached keys, followed by row-wise Softmax. Notice how the matrix physically widens with T.',
      specs: [
        `Score matrix shape for current query token: 32 heads × ${clampedT} cached tokens`,
        'Scaling factor: 1 / √128 ≈ 0.0884',
      ],
    },
    'weighted-v': {
      id: 'weighted-v',
      title: 'Weighted Value Output (32 × 128 = 4,096)',
      badge: isFp8Mode ? 'BF16 Preserved' : 'BF16',
      summary:
        'Multiplies Softmax attention probabilities against the cached V stacks (KV Head 0 for Q0..Q15, KV Head 1 for Q16..Q31) and concatenates all 32 heads into a 4,096-channel vector.',
      specs: [
        'Concatenated head output width: 32 × 128 = 4,096',
      ],
    },
    'o-proj': {
      id: 'o-proj',
      title: 'Output Projection O (4,096 → 2,688)',
      badge: isFp8Mode ? 'BF16 Preserved (§4.2)' : 'BF16 Weights',
      summary:
        'Mixes all 32 concatenated attention heads back down into the 2,688-channel model residual dimension before summing into the emerald residual stream.',
      specs: [
        'Matrix shape: 4,096 × 2,688 (~11.0M parameters)',
        'Preserved in BF16 in FP8 Selective mode (NVIDIA §4.2)',
      ],
    },
    'residual-out': {
      id: 'residual-out',
      title: 'Residual Stream Merge (+ 2,688)',
      badge: 'Activation · Emerald',
      summary:
        'Adds the GQA attention update back into the continuous 2,688-channel residual stream.',
      specs: [
        'Operation: h_out = h_in + GQA(RMSNorm(h_in))',
      ],
    },
  };

  const packed = b.pack();
  const fitBounds: AABB = {
    min: [-5.6, -7.8, -1.8],
    max: [8.6, 7.3, kvTrackStartZ + kvTrackLen + 0.1],
  };

  return {
    kind: 'cutaway_gqa',
    bayId,
    ...packed,
    fitBounds,
    landmarks: b.landmarks,
    kvDepthTicks: b.kvDepthTicks,
    solidPickBoxes: b.pickBoxes,
    partCatalog,
    gqaStats: {
      bayId,
      qHeadCount: 32,
      qGroupCount: 2,
      qHeadsPerGroup: 16,
      kvHeadCount: 2,
      gqaRatio: 16,
      kvTokenSliceCount: 2 * clampedT,
      computeFp8,
      kvCacheFp8,
    },
  };
}
