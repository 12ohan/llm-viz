import {
  formatBayId,
  type PrecisionMode,
} from '../model/spec';
import type { AABB } from '../render/camera';
import type { PickBox } from '../render/pick';
import {
  FlowRibbonBuilder,
  type FlowRibbonColor,
} from '../render/ribbons';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_GHOST_RAIL,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../render/sheets';
import type { CutawayPartInfo, CutawaySceneGeometry } from './cutaways';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from './spine';

/** Exact 16:1 GQA head grouping: Q0..Q15 -> KV Head 0, Q16..Q31 -> KV Head 1. */
export function kvHeadForQueryHead(qHeadIndex: number): 0 | 1 {
  return qHeadIndex < 16 ? 0 : 1;
}

/** Parse `'q-head-0'` .. `'q-head-31'` into integer head index `0..31`, or `null`. */
export function parseQueryHeadPartId(partId: string | null | undefined): number | null {
  if (!partId || !partId.startsWith('q-head-')) return null;
  const n = Number.parseInt(partId.slice('q-head-'.length), 10);
  if (!Number.isInteger(n) || n < 0 || n > 31) return null;
  return n;
}

export function formatQueryHeadId(qHeadIndex: number): string {
  return `Q${String(qHeadIndex).padStart(2, '0')}`;
}

export interface GqaCutawayOptions {
  readonly bayId: 6 | 43;
  readonly tokens: number;
  readonly precision: PrecisionMode;
  readonly selectedPart: string | null;
  readonly hoveredPart: string | null;
}

interface SheetSpec {
  cx: number;
  cy: number;
  cz: number;
  ex: number;
  ey: number;
  ez: number;
  gridCols: number;
  gridRows: number;
  sheetKind: number;
  role: number;
  fp8Flag: number;
  highlight: number;
  seed: number;
  partId: string;
  headIndex?: number;
  kvHeadIndex?: 0 | 1;
  cellFamily: number;
}

class GqaSheetBuilder {
  private readonly bayId: number;
  private readonly sheetFloats: number[] = [];
  private readonly solidFloats: number[] = [];
  private readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly kvDepthTicks: KvDepthTickAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();

  constructor(bayId: number) {
    this.bayId = bayId;
  }

  addSheet(s: SheetSpec): void {
    // 1. Primary thin-sheet instance (20 floats = 80 bytes)
    this.sheetFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.gridCols, s.gridRows,
      s.sheetKind,
      s.role,
      s.fp8Flag,
      s.highlight,
      s.seed,
      0, 0, 0, 0, // iTexRect (reserved for Increment 2 live Float32 texture)
      1.0,        // iValueScale (Increment 2)
      0, 0,       // iPad
    );

    // 2. 1-to-1 mirror in solidData (12 floats, fp8Flag at +7) for test & picking contract parity
    this.solidFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.cellFamily,
      s.fp8Flag,
      s.seed,
      this.bayId,
      0.2,
      s.highlight === SHEET_HL_ACTIVE ? 1 : s.highlight === SHEET_HL_BF16_GOLD ? 2 : 0,
    );

    // 3. Pick AABB (slightly padded vertically so thin sheets are effortlessly hoverable/clickable)
    const hx = s.ex * 0.5;
    const hy = Math.max(s.ey * 0.5, 0.06);
    const hz = s.ez * 0.5;
    this.pickBoxes.push({
      bayId: this.bayId,
      partId: s.partId,
      headIndex: s.headIndex,
      kvHeadIndex: s.kvHeadIndex,
      min: [s.cx - hx, s.cy - hy, s.cz - hz],
      max: [s.cx + hx, s.cy + hy, s.cz + hz],
    });
  }

  addGuideLine(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth = 0.9,
    style = LINE_STYLE_GUIDE,
    alpha = 0.55,
  ): void {
    this.lineFloats.push(
      ax, ay, az,
      bx, by, bz,
      0, 1, 0,
      0, 1, 0,
      cssWidth,
      style,
      alpha,
      0, 0, 0,
    );
  }

  pack(): {
    sheetData: Float32Array;
    sheetCount: number;
    solidData: Float32Array;
    solidCount: number;
    lineData: Float32Array;
    lineCount: number;
    ribbonData: Float32Array;
    ribbonCount: number;
    flowVertices: Float32Array;
    flowIndices: Uint32Array;
  } {
    const flowPacked = this.flow.pack();
    return {
      sheetData: new Float32Array(this.sheetFloats),
      sheetCount: this.sheetFloats.length / SHEET_INSTANCE_FLOATS,
      solidData: new Float32Array(this.solidFloats),
      solidCount: this.solidFloats.length / SOLID_INSTANCE_FLOATS,
      lineData: new Float32Array(this.lineFloats),
      lineCount: this.lineFloats.length / LINE_INSTANCE_FLOATS,
      // Explicitly ZERO voxel-bead / cube-chain arrows in GQA!
      ribbonData: new Float32Array(0),
      ribbonCount: 0,
      flowVertices: flowPacked.vertices,
      flowIndices: flowPacked.indices,
    };
  }
}

// Restrained ribbon palette (quiet neutral by default, crisp oxide-red only on active pathway)
const RIBBON_QUIET: FlowRibbonColor = { r: 0.57, g: 0.62, b: 0.70, a: 0.30, borderAlpha: 0.18 };
const RIBBON_DIMMED: FlowRibbonColor = { r: 0.68, g: 0.72, b: 0.78, a: 0.14, borderAlpha: 0.08 };
const RIBBON_RESIDUAL: FlowRibbonColor = { r: 0.25, g: 0.64, b: 0.48, a: 0.42, borderAlpha: 0.28 };
const RIBBON_RESIDUAL_ACTIVE: FlowRibbonColor = { r: 0.16, g: 0.62, b: 0.44, a: 0.78, borderAlpha: 0.45 };
const RIBBON_ACTIVE: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.86, borderAlpha: 0.50 };
const RIBBON_PATHWAY_SOFT: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.56, borderAlpha: 0.34 };

/** Invariant GQA cutaway bounding box across T=1..16 and BF16/FP8 — prevents camera jump. */
export const GQA_FIT_BOUNDS: AABB = {
  min: [-6.65, -0.15, -6.10],
  max: [7.05, 0.25, 6.45],
};

/**
 * Rebuilt Stage A+1 Increment 1 GQA Cutaway (Bay 06 & Bay 43):
 * Thin dimension-aware flat matrix sheets in a calm XZ drafting-table layout,
 * individually selectable Q0..Q31 heads grouped 16:1 onto shared KV Head 0 & KV Head 1,
 * growing square T x T causal Scores & Softmax sheets, and smooth continuous flat flow ribbons.
 */
export function buildGqaCutaway(opts: GqaCutawayOptions): CutawaySceneGeometry {
  const { bayId, precision, selectedPart, hoveredPart } = opts;
  const clampedT = Math.max(1, Math.min(16, Math.round(opts.tokens)));
  const isFp8Mode = precision === 'fp8_selective';
  // Production precision metadata (NVIDIA §4.2): GQA compute bodies stay BF16; KV cache stores FP8.
  const computeFp8 = false;
  const kvCacheFp8 = isFp8Mode;

  const b = new GqaSheetBuilder(bayId);
  const activePart = hoveredPart ?? selectedPart;
  const activeHeadIdx = parseQueryHeadPartId(activePart);
  const activeKvGroup: 0 | 1 | null =
    activeHeadIdx !== null
      ? kvHeadForQueryHead(activeHeadIdx)
      : activePart === 'kv-head-0'
        ? 0
        : activePart === 'kv-head-1'
          ? 1
          : null;

  const hasFocus = activePart !== null;

  // Sheet elevation: thin flat sheets (thickness 0.07) lying cleanly on the XZ drafting plane
  const BASE_Y = 0.035;
  const LIFT_Y = 0.068;
  const SHEET_THICK = 0.070;
  const RIBBON_Y_BG = 0.022;
  const RIBBON_Y_ACT = 0.052;

  const stageHighlight = (partId: string, onHeadPathway: boolean, isComputeBf16 = true): number => {
    if (activePart === partId) return SHEET_HL_ACTIVE;
    if (activeHeadIdx !== null && onHeadPathway) return SHEET_HL_PATHWAY;
    if (activePart === 'kv-head-0' || activePart === 'kv-head-1') {
      if (onHeadPathway) return SHEET_HL_PATHWAY;
    }
    if (isFp8Mode && isComputeBf16 && !hasFocus) return SHEET_HL_BF16_GOLD;
    if (hasFocus && !onHeadPathway) return SHEET_HL_DIMMED;
    return SHEET_HL_NORMAL;
  };

  // --- Key Spatial Anchors (symmetric two-lane XZ schematic) ---
  const resX = 6.50;
  const zRow0 = -5.62; // Input + RMSNorm (generous breathing room above W_K / W_V labels)
  const zRow1 = -3.95; // W_Q, W_K, W_V projections
  const zQComb = -1.55; // Q0..Q15 & Q16..Q31 combs
  const kvStartZ = -2.45;
  const kvSlicePitch = 0.172;
  const kvSliceDepth = 0.130;
  const kvTrackLen = 16 * kvSlicePitch; // 2.752 -> ends at z = +0.302
  const kvSliceZ = (tok: number) => kvStartZ + (tok - 1) * kvSlicePitch + kvSliceDepth * 0.5;
  const kvActiveTipZ = kvSliceZ(clampedT);
  const zScores = 1.70; // Causal T x T Scores & Softmax
  const zVOut = 3.55;   // Weighted V Output (2 lanes)
  const zConcat = 4.65; // Concat [4,096]
  const zWO = 5.82;     // W_O [4,096 -> 2,688] & + Residual

  // Two 16-head Q combs + paired shared KV tracks
  const qPitch = 0.222;
  const qWidth = 0.178;
  const qDepth = 1.18;
  const qGroupStartX: readonly [number, number] = [-6.28, 0.12];
  const qHeadCenterX = (qIdx: number): number => {
    const grp = kvHeadForQueryHead(qIdx);
    const local = qIdx % 16;
    return qGroupStartX[grp] + local * qPitch;
  };
  const qGroupCenterX: readonly [number, number] = [
    qGroupStartX[0] + 7.5 * qPitch, // -4.615
    qGroupStartX[1] + 7.5 * qPitch, // +1.785
  ];

  // KV tracks: Group 0 (K0, V0) & Group 1 (K1, V1)
  const kvKCenterX: readonly [number, number] = [-1.88, 4.48];
  const kvVCenterX: readonly [number, number] = [-1.22, 5.14];
  const kvPairCenterX: readonly [number, number] = [-1.55, 4.81];
  const kvSliceWidth = 0.54;

  // Causal T x T Scores & Softmax square size scales visibly from T=1 to T=16
  const scoreSize = 0.54 + (clampedT / 16) * 1.42; // 0.63 at T=1 -> 1.25 at T=8 -> 1.96 at T=16
  const scoresX: readonly [number, number] = [-4.62, 1.72];
  const softmaxX: readonly [number, number] = [-2.12, 4.22];

  // V Out per lane
  const vOutX: readonly [number, number] = [-3.38, 2.96];
  const vOutW = 3.25;

  // ==========================================================================
  // 1. RIGHT RESIDUAL HIGHWAY: Input (2,688) ---> + Residual (2,688)
  // ==========================================================================
  const resInHl =
    activePart === 'residual-in'
      ? SHEET_HL_ACTIVE
      : activePart === 'residual-out'
        ? SHEET_HL_PATHWAY
        : SHEET_HL_NORMAL;
  b.addSheet({
    cx: resX,
    cy: resInHl === SHEET_HL_ACTIVE ? LIFT_Y : BASE_Y,
    cz: zRow0,
    ex: 0.72,
    ey: SHEET_THICK,
    ez: 1.08,
    gridCols: 4,
    gridRows: 16,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resInHl,
    seed: 1.1 + clampedT * 0.1,
    partId: 'residual-in',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  const resOutHl =
    activePart === 'residual-out'
      ? SHEET_HL_ACTIVE
      : activeHeadIdx !== null || activePart === 'o-proj'
        ? SHEET_HL_PATHWAY
        : SHEET_HL_NORMAL;
  b.addSheet({
    cx: resX,
    cy: resOutHl === SHEET_HL_ACTIVE ? LIFT_Y : BASE_Y,
    cz: zWO,
    ex: 0.72,
    ey: SHEET_THICK,
    ez: 1.08,
    gridCols: 4,
    gridRows: 16,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resOutHl,
    seed: 2.3 + clampedT * 0.1,
    partId: 'residual-out',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Smooth continuous emerald residual bypass ribbon
  const resRibbonCol =
    activePart === 'residual-in' || activePart === 'residual-out'
      ? RIBBON_RESIDUAL_ACTIVE
      : RIBBON_RESIDUAL;
  b.flow.addPolylineXZ(
    [
      { x: resX, z: zRow0 + 0.56 },
      { x: resX, z: zWO - 0.56 },
    ],
    RIBBON_Y_BG,
    0.18,
    resRibbonCol,
    true,
    0.34,
    0.36,
  );

  // ==========================================================================
  // 2. ROW 0: Pre-Attention RMSNorm (2,688)
  // ==========================================================================
  const rmsX = 0.60;
  const rmsW = 2.60;
  const rmsD = 0.52;
  const rmsHl = stageHighlight('rmsnorm', activeHeadIdx !== null || activePart === 'w-q' || activePart === 'k-proj' || activePart === 'v-proj');
  b.addSheet({
    cx: rmsX,
    cy: rmsHl === SHEET_HL_ACTIVE ? LIFT_Y : BASE_Y,
    cz: zRow0,
    ex: rmsW,
    ey: SHEET_THICK,
    ez: rmsD,
    gridCols: 24,
    gridRows: 4,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: rmsHl,
    seed: 3.7,
    partId: 'rmsnorm',
    cellFamily: CELL_MOE_AMBER,
  });

  // Flow Input -> RMSNorm
  b.flow.addPolylineXZ(
    [
      { x: resX - 0.38, z: zRow0 },
      { x: rmsX + rmsW * 0.5 + 0.04, z: zRow0 },
    ],
    rmsHl === SHEET_HL_ACTIVE || activePart === 'residual-in' ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.13,
    rmsHl === SHEET_HL_ACTIVE || activePart === 'residual-in' ? RIBBON_ACTIVE : RIBBON_RESIDUAL,
    true,
  );

  // ==========================================================================
  // 3. ROW 1: Q / K / V Projection Weight Sheets (Logical 16:1 Width Encoding)
  // ==========================================================================
  // W_Q (2,688 -> 4,096): visibly wide weight sheet
  const wqX = -1.90;
  const wqW = 4.60;
  const wqD = 0.86;
  const wqOnPath = activeHeadIdx !== null || activePart === 'w-q';
  const wqHl = stageHighlight('w-q', wqOnPath);
  b.addSheet({
    cx: wqX,
    cy: wqHl === SHEET_HL_ACTIVE ? LIFT_Y : BASE_Y,
    cz: zRow1,
    ex: wqW,
    ey: SHEET_THICK,
    ez: wqD,
    gridCols: 32,
    gridRows: 12,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wqHl,
    seed: 5.1 + bayId * 0.3,
    partId: 'w-q',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // W_K (2,688 -> 256) & W_V (2,688 -> 256): visibly narrow (~8.5x narrower per sheet -> 16:1 ratio!)
  const wkX = 1.65;
  const wvX = 2.65;
  const wkvW = 0.54;
  const wkvD = 0.86;

  const wkOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'k-proj';
  const wkHl = stageHighlight('k-proj', wkOnPath);
  b.addSheet({
    cx: wkX,
    cy: wkHl === SHEET_HL_ACTIVE ? LIFT_Y : BASE_Y,
    cz: zRow1,
    ex: wkvW,
    ey: SHEET_THICK,
    ez: wkvD,
    gridCols: 4,
    gridRows: 12,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wkHl,
    seed: 6.4 + bayId * 0.2,
    partId: 'k-proj',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  const wvOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'v-proj';
  const wvHl = stageHighlight('v-proj', wvOnPath);
  b.addSheet({
    cx: wvX,
    cy: wvHl === SHEET_HL_ACTIVE ? LIFT_Y : BASE_Y,
    cz: zRow1,
    ex: wkvW,
    ey: SHEET_THICK,
    ez: wkvD,
    gridCols: 4,
    gridRows: 12,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wvHl,
    seed: 7.9 + bayId * 0.2,
    partId: 'v-proj',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Smooth curves RMSNorm -> W_Q, W_K, W_V
  const projFanColor = (onPath: boolean): FlowRibbonColor =>
    onPath && hasFocus ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET;

  b.flow.addCubicXZ(
    { x: rmsX - 0.55, z: zRow0 + rmsD * 0.5 + 0.02 },
    { x: rmsX - 0.55, z: -4.75 },
    { x: wqX, z: -4.75 },
    { x: wqX, z: zRow1 - wqD * 0.5 - 0.04 },
    wqOnPath && hasFocus ? RIBBON_Y_ACT : RIBBON_Y_BG,
    wqOnPath && hasFocus ? 0.13 : 0.095,
    projFanColor(wqOnPath || activePart === 'rmsnorm'),
    true,
  );
  b.flow.addCubicXZ(
    { x: rmsX + 0.35, z: zRow0 + rmsD * 0.5 + 0.02 },
    { x: rmsX + 0.35, z: -4.82 },
    { x: wkX, z: -4.82 },
    { x: wkX, z: zRow1 - wkvD * 0.5 - 0.04 },
    wkOnPath && hasFocus ? RIBBON_Y_ACT : RIBBON_Y_BG,
    wkOnPath && hasFocus ? 0.11 : 0.085,
    projFanColor(wkOnPath || activePart === 'rmsnorm'),
    true,
  );
  b.flow.addCubicXZ(
    { x: rmsX + 0.85, z: zRow0 + rmsD * 0.5 + 0.02 },
    { x: rmsX + 0.85, z: -4.88 },
    { x: wvX, z: -4.88 },
    { x: wvX, z: zRow1 - wkvD * 0.5 - 0.04 },
    wvOnPath && hasFocus ? RIBBON_Y_ACT : RIBBON_Y_BG,
    wvOnPath && hasFocus ? 0.11 : 0.085,
    projFanColor(wvOnPath || activePart === 'rmsnorm'),
    true,
  );

  // ==========================================================================
  // 4. ROW 2: 32 Individually Selectable Q Heads (2x16) & 2 Shared KV Tracks
  // ==========================================================================
  for (let qIdx = 0; qIdx < 32; qIdx++) {
    const grp = kvHeadForQueryHead(qIdx);
    const hx = qHeadCenterX(qIdx);

    let hl = SHEET_HL_NORMAL;
    if (activeHeadIdx === qIdx) {
      hl = SHEET_HL_ACTIVE;
    } else if (activePart === 'q-proj' || activePart === 'w-q') {
      hl = SHEET_HL_ACTIVE;
    } else if (activePart === `kv-head-${grp}`) {
      // Clicking KV Head 0 highlights all 16 query heads Q0..Q15 sharing it!
      hl = SHEET_HL_PATHWAY;
    } else if (activeHeadIdx !== null && kvHeadForQueryHead(activeHeadIdx) === grp) {
      // Sibling heads in same 16:1 group stay gently legible while non-group recedes
      hl = SHEET_HL_NORMAL;
    } else if (hasFocus) {
      hl = SHEET_HL_DIMMED;
    } else if (isFp8Mode) {
      hl = SHEET_HL_BF16_GOLD;
    }

    b.addSheet({
      cx: hx,
      cy: hl === SHEET_HL_ACTIVE ? LIFT_Y : BASE_Y,
      cz: zQComb,
      ex: qWidth,
      ey: SHEET_THICK,
      ez: qDepth,
      gridCols: 2,
      gridRows: 16,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: hl,
      seed: bayId * 0.5 + qIdx * 0.73 + clampedT * 0.15,
      partId: 'q-proj',
      headIndex: qIdx,
      kvHeadIndex: grp,
      cellFamily: CELL_WEIGHT_BLUE,
    });
  }

  // Quiet bank bracket lines behind Group 0 (Q0..Q15) and Group 1 (Q16..Q31)
  for (let grp = 0; grp < 2; grp++) {
    const x0 = qGroupStartX[grp] - qWidth * 0.5;
    const x1 = qGroupStartX[grp] + 15 * qPitch + qWidth * 0.5;
    const bz = zQComb - qDepth * 0.5 - 0.14;
    const isGrpAct = activeKvGroup === grp;
    b.addGuideLine(
      x0, 0.02, bz,
      x1, 0.02, bz,
      isGrpAct ? 1.5 : 0.9,
      isGrpAct ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
      isGrpAct ? 0.9 : 0.55,
    );
  }

  // Smooth ribbons W_Q -> Group 0 & Group 1 Q combs
  for (let grp = 0; grp < 2; grp++) {
    const isActiveLane = activeKvGroup === grp;
    const targetX =
      activeHeadIdx !== null && kvHeadForQueryHead(activeHeadIdx) === grp
        ? qHeadCenterX(activeHeadIdx)
        : qGroupCenterX[grp];
    const startX = grp === 0 ? wqX - 1.15 : wqX + 1.15;
    b.flow.addCubicXZ(
      { x: startX, z: zRow1 + wqD * 0.5 + 0.02 },
      { x: startX, z: -2.92 },
      { x: targetX, z: -2.92 },
      { x: targetX, z: zQComb - qDepth * 0.5 - 0.06 },
      isActiveLane || activePart === 'w-q' ? RIBBON_Y_ACT : RIBBON_Y_BG,
      isActiveLane || activePart === 'w-q' ? 0.12 : 0.085,
      isActiveLane || activePart === 'w-q'
        ? RIBBON_PATHWAY_SOFT
        : hasFocus
          ? RIBBON_DIMMED
          : RIBBON_QUIET,
      true,
    );
  }

  // Twin Shared KV Cache Tracks (KV Head 0 & KV Head 1: 2 ghost rails + 2*T live slices each)
  for (let kh = 0; kh < 2; kh++) {
    const kvPartId = kh === 0 ? 'kv-head-0' : 'kv-head-1';
    const isDirectKv = activePart === kvPartId;
    const isSharedForHead = activeHeadIdx !== null && kvHeadForQueryHead(activeHeadIdx) === kh;
    const isKvRelevant = isDirectKv || isSharedForHead || activePart === 'k-proj' || activePart === 'v-proj';

    const railHl = isDirectKv
      ? SHEET_HL_ACTIVE
      : isSharedForHead
        ? SHEET_HL_PATHWAY
        : SHEET_HL_NORMAL;

    // 2 ghost capacity rails per KV head (K rail + V rail, 1..16 slots) -> fp8Flag = 0
    for (const trackX of [kvKCenterX[kh], kvVCenterX[kh]]) {
      b.addSheet({
        cx: trackX,
        cy: 0.012,
        cz: kvStartZ + kvTrackLen * 0.5,
        ex: kvSliceWidth + 0.04,
        ey: 0.024,
        ez: kvTrackLen + 0.04,
        gridCols: 1,
        gridRows: 16,
        sheetKind: SHEET_KIND_GHOST_RAIL,
        role: SHEET_ROLE_NORM,
        fp8Flag: 0,
        highlight: railHl,
        seed: kh * 2.1,
        partId: kvPartId,
        kvHeadIndex: kh as 0 | 1,
        cellFamily: CELL_NEUTRAL,
      });
    }

    // Live token slices t = 1..T (K slice + V slice per token -> 2*T live slices per KV head!)
    for (let t = 1; t <= clampedT; t++) {
      const sz = kvSliceZ(t);
      const sliceHl = isDirectKv
        ? SHEET_HL_ACTIVE
        : isSharedForHead
          ? SHEET_HL_PATHWAY
          : hasFocus && !isKvRelevant
            ? SHEET_HL_DIMMED
            : SHEET_HL_NORMAL;

      // K slice
      b.addSheet({
        cx: kvKCenterX[kh],
        cy: sliceHl === SHEET_HL_ACTIVE || sliceHl === SHEET_HL_PATHWAY ? LIFT_Y : BASE_Y + 0.014,
        cz: sz,
        ex: kvSliceWidth,
        ey: 0.062,
        ez: kvSliceDepth,
        gridCols: 8,
        gridRows: 1,
        sheetKind: SHEET_KIND_STANDARD,
        role: SHEET_ROLE_KV_K,
        fp8Flag: kvCacheFp8 ? 1 : 0,
        highlight: sliceHl,
        seed: t * 1.13 + kh * 4.3 + bayId * 0.1,
        partId: kvPartId,
        kvHeadIndex: kh as 0 | 1,
        cellFamily: CELL_ACTIVATION_GREEN,
      });

      // V slice
      b.addSheet({
        cx: kvVCenterX[kh],
        cy: sliceHl === SHEET_HL_ACTIVE || sliceHl === SHEET_HL_PATHWAY ? LIFT_Y : BASE_Y + 0.014,
        cz: sz,
        ex: kvSliceWidth,
        ey: 0.062,
        ez: kvSliceDepth,
        gridCols: 8,
        gridRows: 1,
        sheetKind: SHEET_KIND_STANDARD,
        role: SHEET_ROLE_KV_V,
        fp8Flag: kvCacheFp8 ? 1 : 0,
        highlight: sliceHl,
        seed: t * 1.37 + kh * 6.7 + bayId * 0.15,
        partId: kvPartId,
        kvHeadIndex: kh as 0 | 1,
        cellFamily: CELL_SSM_CYAN,
      });
    }

    // Smooth write ribbons W_K -> K_kh and W_V -> V_kh
    const kWriteActive = activePart === 'k-proj' || isDirectKv || isSharedForHead;
    b.flow.addCubicXZ(
      { x: wkX, z: zRow1 + wkvD * 0.5 + 0.02 },
      { x: wkX, z: -3.02 },
      { x: kvKCenterX[kh], z: -3.02 },
      { x: kvKCenterX[kh], z: kvStartZ - 0.05 },
      kWriteActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      kWriteActive ? 0.095 : 0.070,
      kWriteActive ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );

    const vWriteActive = activePart === 'v-proj' || isDirectKv || isSharedForHead;
    b.flow.addCubicXZ(
      { x: wvX, z: zRow1 + wkvD * 0.5 + 0.02 },
      { x: wvX, z: -2.92 },
      { x: kvVCenterX[kh], z: -2.92 },
      { x: kvVCenterX[kh], z: kvStartZ - 0.05 },
      vWriteActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      vWriteActive ? 0.095 : 0.070,
      vWriteActive ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );
  }

  // KV depth scale ticks (1, 8, 16, active T) neatly along outer right edge of KV Head 1
  const tickScaleX = kvVCenterX[1] + kvSliceWidth * 0.5 + 0.22;
  b.addGuideLine(tickScaleX, 0.02, kvStartZ, tickScaleX, 0.02, kvStartZ + kvTrackLen, 0.85, LINE_STYLE_GUIDE, 0.50);
  const tickSet = new Set<number>([1, 8, 16, clampedT]);
  for (const tk of Array.from(tickSet).sort((a, c) => a - c)) {
    const tz = kvSliceZ(tk);
    b.addGuideLine(
      tickScaleX - 0.09, 0.02, tz,
      tickScaleX + 0.09, 0.02, tz,
      tk === clampedT ? 1.4 : 0.85,
      tk === clampedT ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
      0.8,
    );
    b.kvDepthTicks.push({
      token: tk,
      world: [tickScaleX + 0.18, 0.02, tz],
    });
  }

  // ==========================================================================
  // 5. ROW 3: Causal T x T Scores (QK^T / sqrt(128)) & Causal Mask + Softmax
  // ==========================================================================
  for (let grp = 0; grp < 2; grp++) {
    const isGrpActive = activeKvGroup === grp;
    const scoresHl =
      activePart === 'attn-scores'
        ? SHEET_HL_ACTIVE
        : isGrpActive
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;

    b.addSheet({
      cx: scoresX[grp],
      cy: scoresHl === SHEET_HL_ACTIVE || scoresHl === SHEET_HL_PATHWAY ? LIFT_Y : BASE_Y,
      cz: zScores,
      ex: scoreSize,
      ey: SHEET_THICK,
      ez: scoreSize,
      gridCols: clampedT,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_CAUSAL_SCORES,
      role: SHEET_ROLE_SCORES,
      fp8Flag: 0,
      highlight: scoresHl,
      seed: 11.3 + grp * 3.7 + clampedT * 0.4,
      partId: 'attn-scores',
      kvHeadIndex: grp as 0 | 1,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    const softmaxHl =
      activePart === 'softmax'
        ? SHEET_HL_ACTIVE
        : isGrpActive || activePart === 'attn-scores'
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;

    b.addSheet({
      cx: softmaxX[grp],
      cy: softmaxHl === SHEET_HL_ACTIVE || softmaxHl === SHEET_HL_PATHWAY ? LIFT_Y : BASE_Y,
      cz: zScores,
      ex: scoreSize,
      ey: SHEET_THICK,
      ez: scoreSize,
      gridCols: clampedT,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
      role: SHEET_ROLE_SOFTMAX,
      fp8Flag: 0,
      highlight: softmaxHl,
      seed: 17.5 + grp * 5.1 + clampedT * 0.3,
      partId: 'softmax',
      kvHeadIndex: grp as 0 | 1,
      cellFamily: CELL_MOE_AMBER,
    });

    // Smooth curve Q_comb (or specific selected Q_i!) -> Scores_grp
    const qSourceX =
      activeHeadIdx !== null && kvHeadForQueryHead(activeHeadIdx) === grp
        ? qHeadCenterX(activeHeadIdx)
        : qGroupCenterX[grp];
    const qToScoresActive = isGrpActive || activePart === 'attn-scores';
    b.flow.addCubicXZ(
      { x: qSourceX, z: zQComb + qDepth * 0.5 + 0.03 },
      { x: qSourceX, z: 0.10 },
      { x: scoresX[grp] - 0.16, z: 0.15 },
      { x: scoresX[grp] - 0.16, z: zScores - scoreSize * 0.5 - 0.06 },
      qToScoresActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      activeHeadIdx !== null && isGrpActive ? 0.13 : 0.09,
      activeHeadIdx !== null && isGrpActive
        ? RIBBON_ACTIVE
        : qToScoresActive
          ? RIBBON_PATHWAY_SOFT
          : hasFocus
            ? RIBBON_DIMMED
            : RIBBON_QUIET,
      true,
    );

    // Smooth curve Shared K_grp (from active 1..T tip!) -> Scores_grp
    const kToScoresActive = isGrpActive || activePart === 'attn-scores';
    b.flow.addCubicXZ(
      { x: kvKCenterX[grp], z: kvActiveTipZ + kvSliceDepth * 0.5 + 0.04 },
      { x: kvKCenterX[grp], z: Math.max(kvActiveTipZ + 0.45, 0.42) },
      { x: scoresX[grp] + 0.18, z: 0.35 },
      { x: scoresX[grp] + 0.18, z: zScores - scoreSize * 0.5 - 0.06 },
      kToScoresActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      kToScoresActive ? 0.12 : 0.08,
      kToScoresActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );

    // Smooth horizontal ribbon Scores_grp -> Softmax_grp
    const s2sActive = isGrpActive || activePart === 'attn-scores' || activePart === 'softmax';
    b.flow.addPolylineXZ(
      [
        { x: scoresX[grp] + scoreSize * 0.5 + 0.04, z: zScores },
        { x: softmaxX[grp] - scoreSize * 0.5 - 0.05, z: zScores },
      ],
      s2sActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      s2sActive ? 0.12 : 0.085,
      s2sActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );
  }

  // ==========================================================================
  // 6. ROW 4: Weighted V Output (2 lanes x 16*128) -> Concat [4,096]
  // ==========================================================================
  for (let grp = 0; grp < 2; grp++) {
    const isGrpActive = activeKvGroup === grp;
    const vOutHl =
      activePart === 'weighted-v'
        ? SHEET_HL_ACTIVE
        : isGrpActive || activePart === 'softmax'
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;

    b.addSheet({
      cx: vOutX[grp],
      cy: vOutHl === SHEET_HL_ACTIVE || vOutHl === SHEET_HL_PATHWAY ? LIFT_Y : BASE_Y,
      cz: zVOut,
      ex: vOutW,
      ey: SHEET_THICK,
      ez: 0.54,
      gridCols: 16,
      gridRows: 4,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: vOutHl,
      seed: 23.1 + grp * 2.9 + clampedT * 0.2,
      partId: 'weighted-v',
      kvHeadIndex: grp as 0 | 1,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    // Ribbon Softmax_grp -> V Out_grp
    const sm2vActive = isGrpActive || activePart === 'softmax' || activePart === 'weighted-v';
    b.flow.addCubicXZ(
      { x: softmaxX[grp], z: zScores + scoreSize * 0.5 + 0.03 },
      { x: softmaxX[grp], z: 2.95 },
      { x: vOutX[grp] + 0.35, z: 2.95 },
      { x: vOutX[grp] + 0.35, z: zVOut - 0.30 },
      sm2vActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      sm2vActive ? 0.12 : 0.085,
      sm2vActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );

    // Smooth curved V-cache bypass ribbon from Shared V_grp -> V Out_grp (sweeping cleanly outside Softmax)
    const vBypassX = grp === 0 ? -0.82 : 5.56;
    const v2vActive = isGrpActive || activePart === 'weighted-v';
    b.flow.addRoundedOrthoXZ(
      [
        { x: kvVCenterX[grp], z: kvActiveTipZ + kvSliceDepth * 0.5 + 0.03 },
        { x: kvVCenterX[grp], z: Math.max(kvActiveTipZ + 0.32, 0.45) },
        { x: vBypassX, z: Math.max(kvActiveTipZ + 0.32, 0.45) },
        { x: vBypassX, z: 3.04 },
        { x: vOutX[grp] + 1.05, z: 3.04 },
        { x: vOutX[grp] + 1.05, z: zVOut - 0.30 },
      ],
      0.24,
      v2vActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      v2vActive ? 0.105 : 0.072,
      v2vActive ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );
  }

  // Concat [4,096]
  const concatX = -0.20;
  const concatW = 5.60;
  const concatD = 0.50;
  const concatOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'weighted-v' || activePart === 'o-proj';
  const concatHl = stageHighlight('concat', concatOnPath);
  b.addSheet({
    cx: concatX,
    cy: concatHl === SHEET_HL_ACTIVE || concatHl === SHEET_HL_PATHWAY ? LIFT_Y : BASE_Y,
    cz: zConcat,
    ex: concatW,
    ey: SHEET_THICK,
    ez: concatD,
    gridCols: 32,
    gridRows: 4,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: concatHl,
    seed: 29.4 + clampedT * 0.2,
    partId: 'concat',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  for (let grp = 0; grp < 2; grp++) {
    const laneOn = activeKvGroup === grp || activePart === 'weighted-v' || activePart === 'concat';
    const destX = grp === 0 ? concatX - 1.40 : concatX + 1.40;
    b.flow.addCubicXZ(
      { x: vOutX[grp], z: zVOut + 0.29 },
      { x: vOutX[grp], z: 4.10 },
      { x: destX, z: 4.10 },
      { x: destX, z: zConcat - concatD * 0.5 - 0.04 },
      laneOn ? RIBBON_Y_ACT : RIBBON_Y_BG,
      laneOn ? 0.12 : 0.085,
      laneOn ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );
  }

  // ==========================================================================
  // 7. ROW 5: W_O [4,096 -> 2,688] -> + Residual
  // ==========================================================================
  const woX = 1.60;
  const woW = 3.55; // Visibly narrower than Concat [4,096] (5.60 -> 3.55) encoding 4,096 -> 2,688!
  const woD = 0.84;
  const woOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'concat' || activePart === 'residual-out';
  const woHl = stageHighlight('o-proj', woOnPath);
  b.addSheet({
    cx: woX,
    cy: woHl === SHEET_HL_ACTIVE || woHl === SHEET_HL_PATHWAY ? LIFT_Y : BASE_Y,
    cz: zWO,
    ex: woW,
    ey: SHEET_THICK,
    ez: woD,
    gridCols: 24,
    gridRows: 16,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: woHl,
    seed: 37.2 + bayId * 0.4,
    partId: 'o-proj',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  const c2oActive = woOnPath || activePart === 'o-proj';
  b.flow.addCubicXZ(
    { x: concatX + 0.80, z: zConcat + concatD * 0.5 + 0.02 },
    { x: concatX + 0.80, z: 5.16 },
    { x: woX, z: 5.16 },
    { x: woX, z: zWO - woD * 0.5 - 0.04 },
    c2oActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
    c2oActive ? 0.125 : 0.09,
    c2oActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
    true,
  );

  // W_O -> + Residual
  const o2rActive = activePart === 'o-proj' || activePart === 'residual-out' || activeHeadIdx !== null;
  b.flow.addPolylineXZ(
    [
      { x: woX + woW * 0.5 + 0.04, z: zWO },
      { x: resX - 0.40, z: zWO },
    ],
    o2rActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
    o2rActive ? 0.14 : 0.11,
    o2rActive ? RIBBON_ACTIVE : RIBBON_RESIDUAL,
    true,
  );

  // ==========================================================================
  // 8. EXTERIOR PROGRESSIVE LABELS (Short Primary Labels Outside Sheet Edges)
  // ==========================================================================
  const selectedHeadLabel =
    activeHeadIdx !== null
      ? `${formatQueryHeadId(activeHeadIdx)} → KV${kvHeadForQueryHead(activeHeadIdx)}`
      : null;

  b.landmarks.push(
    // Primary exterior labels (1-3 words, strictly outside sheet edges, zero face occlusion)
    { id: 'g-res-in', label: 'Input (2,688)', world: [resX, 0.08, zRow0 - 0.68], partId: 'residual-in', align: 'top', tier: 'primary' },
    { id: 'g-rmsnorm', label: 'RMSNorm', world: [rmsX, 0.08, zRow0 - 0.40], partId: 'rmsnorm', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wq', label: 'W_Q', world: [wqX - wqW * 0.5 - 0.18, 0.08, zRow1], partId: 'w-q', align: 'left', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wk', label: 'W_K', world: [wkX, 0.08, zRow1 - 0.48], partId: 'k-proj', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wv', label: 'W_V', world: [wvX, 0.08, zRow1 - 0.48], partId: 'v-proj', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },

    {
      id: 'g-q0',
      label: activeHeadIdx !== null && activeHeadIdx < 16 ? `Q₀..Q₁₅ · ${selectedHeadLabel}` : 'Q₀..Q₁₅ (×16)',
      world: [qGroupCenterX[0], 0.08, zQComb - qDepth * 0.5 - 0.24],
      partId: activeHeadIdx !== null && activeHeadIdx < 16 ? activePart ?? 'q-proj' : 'q-proj',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'g-q1',
      label: activeHeadIdx !== null && activeHeadIdx >= 16 ? `Q₁₆..Q₃₁ · ${selectedHeadLabel}` : 'Q₁₆..Q₃₁ (×16)',
      world: [qGroupCenterX[1], 0.08, zQComb - qDepth * 0.5 - 0.24],
      partId: activeHeadIdx !== null && activeHeadIdx >= 16 ? activePart ?? 'q-proj' : 'q-proj',
      align: 'top',
      tier: 'primary',
    },

    {
      id: 'g-kv0',
      label: kvCacheFp8 ? `KV₀ · FP8 (T=${clampedT})` : `KV₀ (T=${clampedT})`,
      world: [kvPairCenterX[0], 0.08, kvStartZ - 0.18],
      partId: 'kv-head-0',
      align: 'top',
      tier: 'primary',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },
    {
      id: 'g-kv1',
      label: kvCacheFp8 ? `KV₁ · FP8 (T=${clampedT})` : `KV₁ (T=${clampedT})`,
      world: [kvPairCenterX[1], 0.08, kvStartZ - 0.18],
      partId: 'kv-head-1',
      align: 'top',
      tier: 'primary',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },

    {
      id: 'g-scores',
      label: `Scores (${clampedT}×${clampedT})`,
      world: [scoresX[0] - scoreSize * 0.5 - 0.16, 0.08, zScores],
      partId: 'attn-scores',
      align: 'left',
      tier: 'primary',
    },
    {
      id: 'g-softmax',
      label: 'Softmax',
      world: [5.66, 0.08, zScores],
      partId: 'softmax',
      align: 'right',
      tier: 'primary',
    },

    { id: 'g-vout', label: 'V Out', world: [vOutX[0] - vOutW * 0.5 - 0.16, 0.08, zVOut], partId: 'weighted-v', align: 'left', tier: 'primary' },
    { id: 'g-concat', label: 'Concat [4,096]', world: [concatX - concatW * 0.5 - 0.16, 0.08, zConcat], partId: 'concat', align: 'left', tier: 'primary' },
    { id: 'g-oproj', label: 'W_O', world: [woX - woW * 0.5 - 0.16, 0.08, zWO], partId: 'o-proj', align: 'left', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-res-out', label: '+ Residual', world: [resX, 0.08, zWO + 0.68], partId: 'residual-out', align: 'bottom', tier: 'primary' },

    // Progressive detail labels (revealed on hover, selection, or close zoom!)
    {
      id: 'g-nope',
      label: 'NoPE · order from Mamba-2',
      world: [rmsX + rmsW * 0.5 + 0.25, 0.08, zRow0 - 0.45],
      partId: 'rmsnorm',
      align: 'right',
      tier: 'detail',
      tone: 'cyan',
    },
    {
      id: 'g-wq-shape',
      label: isFp8Mode ? '2,688 → 4,096 · BF16' : '2,688 → 4,096',
      world: [wqX, 0.08, zRow1 - wqD * 0.5 - 0.16],
      partId: 'w-q',
      align: 'top',
      tier: 'detail',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    {
      id: 'g-wo-shape',
      label: isFp8Mode ? '4,096 → 2,688 · BF16' : '4,096 → 2,688',
      world: [woX, 0.08, zWO + woD * 0.5 + 0.18],
      partId: 'o-proj',
      align: 'bottom',
      tier: 'detail',
      tone: isFp8Mode ? 'gold' : 'default',
    },
  );

  // Subtle architectural dimension calipers (dynamically attached to active KV group when focused)
  const calGrp: 0 | 1 = activeKvGroup ?? 0;
  b.calipers.push(
    {
      id: 'cal-gqa-ratio',
      aWorld: [qGroupStartX[calGrp] - qWidth * 0.5, 0.06, zQComb + qDepth * 0.5 + 0.16],
      bWorld: [kvVCenterX[calGrp] + kvSliceWidth * 0.5, 0.06, zQComb + qDepth * 0.5 + 0.16],
      label: calGrp === 0 ? '16:1 GQA (Q00–Q15 → KV₀)' : '16:1 GQA (Q16–Q31 → KV₁)',
      partIds: ['q-proj', 'kv-head-0', 'kv-head-1'],
      minPixels: 130,
      offsetPx: 10,
    },
    {
      id: 'cal-scores-t',
      aWorld: [scoresX[calGrp] - scoreSize * 0.5, 0.06, zScores + scoreSize * 0.5 + 0.12],
      bWorld: [scoresX[calGrp] + scoreSize * 0.5, 0.06, zScores + scoreSize * 0.5 + 0.12],
      label: `T = ${clampedT}`,
      partIds: ['attn-scores', 'softmax'],
      minPixels: 52,
      offsetPx: 8,
    },
    {
      id: 'cal-wo-taper',
      aWorld: [woX - woW * 0.5, 0.06, zWO + woD * 0.5 + 0.12],
      bWorld: [woX + woW * 0.5, 0.06, zWO + woD * 0.5 + 0.12],
      label: '4,096 → 2,688',
      partIds: ['o-proj', 'concat'],
      minPixels: 110,
      offsetPx: 10,
    },
  );

  // ==========================================================================
  // 9. PART CATALOG (All Stages + Individual Q00..Q31 Entries, Honest Precision)
  // ==========================================================================
  const perBayKiB = ((kvCacheFp8 ? 512 : 1024) * clampedT) / 1024;
  const perBaySlope = kvCacheFp8 ? '+0.5 KiB/tok (FP8 KV)' : '+1 KiB/tok (BF16 KV)';
  const precisionMetaLine = isFp8Mode
    ? 'Production FP8 Selective metadata (§4.2): GQA weights & compute stay BF16; only KV-cache storage uses FP8'
    : 'Production BF16 metadata: GQA weights, activations, and KV-cache storage all in BF16';

  const partCatalog: Record<string, CutawayPartInfo> = {
    'residual-in': {
      id: 'residual-in',
      title: 'Residual Input (2,688)',
      badge: `Bay ${formatBayId(bayId)} · GQA`,
      summary:
        'Incoming 2,688-channel residual stream already carrying sequential order from the preceding Mamba-2 feeder layer.',
      specs: [
        `Preceding BF16 feeder: Bay ${formatBayId(bayId - 1)} (Mamba-2)`,
        'NoPE: no RoPE or learned positional embedding required',
      ],
    },
    rmsnorm: {
      id: 'rmsnorm',
      title: 'Pre-Attention RMSNorm · NoPE',
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16',
      summary:
        'Normalizes the 2,688-channel activation vector before Q/K/V projections. Nemotron 3 Nano uses NoPE (no rotary position embeddings) because causal Mamba-2 layers supply positional context.',
      specs: [
        'Shape: 2,688 channels (pre-norm residual branch)',
        precisionMetaLine,
      ],
    },
    'w-q': {
      id: 'w-q',
      title: 'Query Projection Matrix W_Q',
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16 Weights',
      summary:
        'Projects the 2,688-channel normalized hidden state into 32 query heads × 128 dim = 4,096 channels, split across Group 0 (Q00–Q15) and Group 1 (Q16–Q31).',
      specs: [
        'Shape: 2,688 → 4,096 (32 heads × d_head = 128)',
        '16× wider output than W_K or W_V (16:1 GQA ratio)',
        precisionMetaLine,
      ],
    },
    'q-proj': {
      id: 'q-proj',
      title: 'Query Heads Q₀₀..Q₃₁ (2 Groups × 16)',
      badge: '32 Heads · 16:1 GQA',
      summary:
        'Thirty-two individually addressable query heads (d_head = 128 each). Click any single query strip or choose Q00–Q31 below to trace its exact shared KV head and causal attention path.',
      specs: [
        'Group 0 (Q00–Q15) shares KV Head 0 · Group 1 (Q16–Q31) shares KV Head 1',
        precisionMetaLine,
      ],
    },
    'k-proj': {
      id: 'k-proj',
      title: 'Key Projection Matrix W_K (16× Narrower)',
      badge: isFp8Mode ? 'Compute BF16 · KV FP8' : 'BF16 Weights',
      summary:
        'Projects 2,688 channels down to just 2 key heads × 128 dim = 256 values — 16× narrower than W_Q — writing one new key vector per token into KV Head 0 and KV Head 1.',
      specs: [
        'Shape: 2,688 → 256 (2 KV heads × d_head = 128)',
        precisionMetaLine,
      ],
    },
    'v-proj': {
      id: 'v-proj',
      title: 'Value Projection Matrix W_V (16× Narrower)',
      badge: isFp8Mode ? 'Compute BF16 · KV FP8' : 'BF16 Weights',
      summary:
        'Projects 2,688 channels down to 2 value heads × 128 dim = 256 values, appending one new value vector per token onto both shared KV-cache tracks.',
      specs: [
        'Shape: 2,688 → 256 (2 KV heads × d_head = 128)',
        precisionMetaLine,
      ],
    },
    'kv-head-0': {
      id: 'kv-head-0',
      title: `Shared KV Head 0 (t = 1 … ${clampedT})`,
      badge: kvCacheFp8 ? 'FP8 KV Storage' : 'BF16 KV Storage',
      summary:
        'Shared Key (K₀) and Value (V₀) history track of dimension 128 per token, attended by all 16 query heads in Group 0 (Q00–Q15).',
      specs: [
        `Shared by 16 query heads: Q00–Q15 (16:1 GQA sharing)`,
        `This GQA bay (KV₀ + KV₁): ${perBayKiB} KiB at T=${clampedT} (${perBaySlope})`,
        precisionMetaLine,
      ],
    },
    'kv-head-1': {
      id: 'kv-head-1',
      title: `Shared KV Head 1 (t = 1 … ${clampedT})`,
      badge: kvCacheFp8 ? 'FP8 KV Storage' : 'BF16 KV Storage',
      summary:
        'Shared Key (K₁) and Value (V₁) history track of dimension 128 per token, attended by all 16 query heads in Group 1 (Q16–Q31).',
      specs: [
        `Shared by 16 query heads: Q16–Q31 (16:1 GQA sharing)`,
        `This GQA bay (KV₀ + KV₁): ${perBayKiB} KiB at T=${clampedT} (${perBaySlope})`,
        precisionMetaLine,
      ],
    },
    'attn-scores': {
      id: 'attn-scores',
      title: `Causal Attention Scores QKᵀ / √128 (${clampedT} × ${clampedT})`,
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16',
      summary:
        'Per-head scaled dot-product score matrix growing quadratically (T × T) with cached context length. The shaded lower triangle holds causal past-and-present scores; the muted upper triangle is masked to −∞.',
      specs: [
        `Logical shape per head: ${clampedT} × ${clampedT} · scale 1 / √128 ≈ 0.0884`,
        'Causal mask blocks future tokens (col > row → −∞)',
        precisionMetaLine,
      ],
    },
    softmax: {
      id: 'softmax',
      title: `Causal Mask & Row-Wise Softmax (${clampedT} × ${clampedT})`,
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16',
      summary:
        'Applies row-wise Softmax over the unmasked causal lower triangle so each query token distributes probability mass summing to 1.0 across tokens 1..t (0.0 in masked upper triangle).',
      specs: [
        `Logical shape per head: ${clampedT} × ${clampedT} probability weights`,
        precisionMetaLine,
      ],
    },
    'weighted-v': {
      id: 'weighted-v',
      title: 'Weighted Value Output (2 Banks × 16 × 128)',
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16',
      summary:
        'Multiplies each head’s Softmax attention weights by its group’s shared cached V track (V₀ for Q00–Q15, V₁ for Q16–Q31), producing 128-dim head outputs.',
      specs: [
        'Each bank outputs 16 heads × 128 = 2,048 channels',
        precisionMetaLine,
      ],
    },
    concat: {
      id: 'concat',
      title: 'Concatenate 32 Heads [4,096]',
      badge: '32 × 128 = 4,096',
      summary:
        'Concatenates all 32 per-head weighted value vectors (Group 0 + Group 1) side-by-side into a single 4,096-channel vector entering the output projection.',
      specs: [
        'Shape: 32 heads × 128 dim = 4,096 channels',
        precisionMetaLine,
      ],
    },
    'o-proj': {
      id: 'o-proj',
      title: 'Output Projection W_O (4,096 → 2,688)',
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16 Weights',
      summary:
        'Projects the 4,096-channel concatenated attention output back down to the 2,688-channel model residual dimension before merging into the residual highway.',
      specs: [
        'Matrix shape: 4,096 × 2,688 (~11.0M parameters)',
        precisionMetaLine,
      ],
    },
    'residual-out': {
      id: 'residual-out',
      title: '+ Residual Stream Merge (2,688)',
      badge: 'h_out = h_in + Attn',
      summary:
        'Adds the 2,688-channel projected GQA update back onto the continuous residual highway.',
      specs: [
        'Residual update: h_out = h_in + W_O(Concat(Attn(RMSNorm(h_in))))',
      ],
    },
  };

  // Dynamic individual entries for all 32 query heads (Q00..Q31)
  for (let i = 0; i < 32; i++) {
    const kv = kvHeadForQueryHead(i);
    const label = formatQueryHeadId(i);
    const groupRange = kv === 0 ? 'Q00–Q15' : 'Q16–Q31';
    partCatalog[`q-head-${i}`] = {
      id: `q-head-${i}`,
      title: `Query Head ${label} → Shared KV Head ${kv}`,
      badge: `Group ${kv} · 16:1 GQA`,
      summary:
        `Individual query head ${label} (d_head = 128) attends across all ${clampedT} cached tokens in Shared KV Head ${kv} (shared with ${groupRange}), computes a causal ${clampedT}×${clampedT} score & Softmax sheet, and writes its 128-dim slice into Concat [4,096] → W_O.`,
      specs: [
        `Mapping: ${label} → KV Head ${kv} (K${kv}, V${kv} · ${clampedT} live slices, ${kvCacheFp8 ? 'FP8 storage' : 'BF16 storage'})`,
        `Causal scores & Softmax: ${clampedT} × ${clampedT} lower-triangular attention`,
        `Output slice: channels [${i * 128}..${(i + 1) * 128 - 1}] of Concat [4,096] → W_O [4,096 → 2,688]`,
        precisionMetaLine,
      ],
    };
  }

  const packed = b.pack();

  return {
    kind: 'cutaway_gqa',
    bayId,
    ...packed,
    fitBounds: GQA_FIT_BOUNDS,
    landmarks: b.landmarks,
    kvDepthTicks: b.kvDepthTicks,
    calipers: b.calipers,
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
      usesSmoothRibbons: true,
      voxelArrowCount: 0,
      selectedQueryHead: activeHeadIdx,
    },
  };
}
