import {
  describeCellFormula,
  formatEpsLabel,
  inferGqaCellHead,
  runGqaSim,
  type GqaCellFormula,
  type GqaCellRef,
  type GqaSheetSpan,
  type GqaSimResult,
  type GqaTensorKey,
} from '../math/gqaSim';
import {
  formatBayId,
  type PrecisionMode,
} from '../model/spec';
import type { GqaViewAngle } from '../model/state';
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
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
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
import { buildGqaValueAtlas, type GqaAtlasSlice } from './gqaAtlas';
import {
  buildGqaFocusSet,
  buildGqaSpatialScene,
  fitSpatialGqaCamera,
  GQA_SPATIAL_FIT_BOUNDS,
} from './gqaSpatial';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  computeSheetGroupBounds,
  drawOperandDependencyXZ,
  resolveSheetFocusRole,
  resolveSheetHighlight,
  RIBBON_ACTIVE,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  RIBBON_RESIDUAL_ACTIVE,
  type SheetGeomRecord,
  type WorldSpanRect,
} from './sheetInspector';

export { fitSpatialGqaCamera, GQA_SPATIAL_FIT_BOUNDS };
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
  readonly activeGroup0Head?: number;
  readonly activeGroup1Head?: number;
  readonly activeCell?: GqaCellRef | null;
  /** Layout mode: `'three_quarter'` builds the 3D Spatial GQA architecture; `'near_plan'` (default) keeps the coplanar XZ schematic. */
  readonly viewAngle?: GqaViewAngle;
  /** Optional authoritative simulator result from a unified `HybridExecutionResult`. */
  readonly simOverride?: GqaSimResult;
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
  tensorKey?: GqaTensorKey;
  tokenIndex?: number;
  atlasSlice?: GqaAtlasSlice;
  registryKey?: string;
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
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();

  constructor(bayId: number) {
    this.bayId = bayId;
  }

  addSheet(s: SheetSpec): void {
    const rect = s.atlasSlice?.rect ?? [0, 0, 0, 0];
    const scale = s.atlasSlice?.scale ?? 1.0;
    const modeFlag = s.atlasSlice?.mode ?? 0;

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
      rect[0], rect[1], rect[2], rect[3], // iTexRect in RG32F value atlas
      scale,                              // iValueScale
      modeFlag, 0,                        // iFlags (x: 0=signed, 1=prob)
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
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
      min: [s.cx - hx, s.cy - hy, s.cz - hz],
      max: [s.cx + hx, s.cy + hy, s.cz + hz],
    });

    if (s.registryKey) {
      this.sheetRegistry.set(s.registryKey, {
        cx: s.cx,
        cy: s.cy,
        cz: s.cz,
        ex: s.ex,
        ey: s.ey,
        ez: s.ez,
        gridCols: s.gridCols,
        gridRows: s.gridRows,
      });
    }
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

/** Invariant GQA cutaway bounding box across T=1..16 and BF16/FP8 — prevents camera jump. */
export const GQA_FIT_BOUNDS: AABB = {
  min: [-6.65, -0.15, -6.10],
  max: [7.05, 0.25, 6.45],
};

function resolveSpanWorldRect(
  registry: ReadonlyMap<string, SheetGeomRecord>,
  span: GqaSheetSpan,
  displayedHeads: readonly [number, number],
): WorldSpanRect | null {
  // Multi-token span on shared KV track (e.g. V_g slices 0..q)
  if (span.tensor === 'kv_k' || span.tensor === 'kv_v') {
    const g = span.kvHeadIndex ?? 0;
    const t0 = span.tokenStart ?? 0;
    const t1 = span.tokenEnd ?? t0;
    const first = registry.get(`${span.tensor}:${g}:${t0}`);
    const last = registry.get(`${span.tensor}:${g}:${t1}`) ?? first;
    if (!first || !last) return null;

    const colW = first.ex / first.gridCols;
    const x0 = first.cx - first.ex * 0.5 + span.colStart * colW;
    const x1 = first.cx - first.ex * 0.5 + (span.colEnd + 1) * colW;
    const z0 = Math.min(first.cz - first.ez * 0.5, last.cz - last.ez * 0.5);
    const z1 = Math.max(first.cz + first.ez * 0.5, last.cz + last.ez * 0.5);
    return {
      cx: (x0 + x1) * 0.5,
      cz: (z0 + z1) * 0.5,
      ex: Math.max(0.04, x1 - x0),
      ez: Math.max(0.04, z1 - z0),
      topY: Math.max(first.cy + first.ey * 0.5, last.cy + last.ey * 0.5),
    };
  }

  let key: string = span.tensor;
  if (span.tensor === 'q') {
    key = `q:${span.headIndex ?? 0}`;
  } else if (span.tensor === 'scores' || span.tensor === 'softmax') {
    const grp: 0 | 1 =
      span.headIndex !== undefined
        ? kvHeadForQueryHead(span.headIndex)
        : span.kvHeadIndex === 1
          ? 1
          : 0;
    // Never highlight a head's Scores/Softmax row onto a sheet displaying a different head
    if (span.headIndex !== undefined && displayedHeads[grp] !== span.headIndex) {
      return null;
    }
    key = `${span.tensor}:${grp}`;
  } else if (span.tensor === 'v_out') {
    key = `${span.tensor}:${span.kvHeadIndex ?? 0}`;
  }

  const sheet = registry.get(key);
  if (!sheet) return null;

  const colW = sheet.ex / sheet.gridCols;
  const rowD = sheet.ez / sheet.gridRows;
  const x0 = sheet.cx - sheet.ex * 0.5 + span.colStart * colW;
  const x1 = sheet.cx - sheet.ex * 0.5 + (span.colEnd + 1) * colW;
  const z0 = sheet.cz - sheet.ez * 0.5 + span.rowStart * rowD;
  const z1 = sheet.cz - sheet.ez * 0.5 + (span.rowEnd + 1) * rowD;

  return {
    cx: (x0 + x1) * 0.5,
    cz: (z0 + z1) * 0.5,
    ex: Math.max(0.035, x1 - x0),
    ez: Math.max(0.035, z1 - z0),
    topY: sheet.cy + sheet.ey * 0.5,
  };
}

/**
 * Stage A+1 Increment 2+ GQA Cutaway (Bay 06 & Bay 43):
 * Thin dimension-aware flat matrix sheets backed by a deterministic miniature simulator
 * with Float32 tensor storage and rounded outputs (d=8, 32Q/2KV, d_h=4, NoPE), RG32F value atlas,
 * head-synchronized cell-level picking, upstream operand highlights, dependency threads,
 * and causal attention stepping.
 */
export function buildGqaCutaway(opts: GqaCutawayOptions): CutawaySceneGeometry {
  const { bayId, precision, selectedPart, hoveredPart } = opts;
  const clampedT = opts.simOverride
    ? opts.simOverride.tokens
    : Math.max(1, Math.min(16, Math.round(opts.tokens)));
  const isFp8Mode = precision === 'fp8_selective';
  // Production precision metadata (NVIDIA §4.2): GQA compute bodies stay BF16; KV cache stores FP8.
  const computeFp8 = false;
  const kvCacheFp8 = isFp8Mode;

  const activePart = hoveredPart ?? selectedPart;
  const partHeadIdx = parseQueryHeadPartId(activePart);
  const cellHeadIdx = inferGqaCellHead(opts.activeCell);
  // Inspected cell head takes precedence over stale part head so V_out/Concat cells for Q_h
  // always synchronize their KV group's displayed Scores & Softmax sheets to Q_h.
  const activeHeadIdx = cellHeadIdx ?? partHeadIdx;

  // Determine which query head's Scores/Softmax live Float32 matrices are displayed in each group
  const group0Head =
    activeHeadIdx !== null && activeHeadIdx < 16
      ? activeHeadIdx
      : Math.max(0, Math.min(15, opts.activeGroup0Head ?? 0));
  const group1Head =
    activeHeadIdx !== null && activeHeadIdx >= 16
      ? activeHeadIdx
      : Math.max(16, Math.min(31, opts.activeGroup1Head ?? 16));
  const displayedHeads: readonly [number, number] = [group0Head, group1Head];

  // Use authoritative unified execution trace when supplied, or standalone miniature simulator
  const sim = opts.simOverride ?? runGqaSim(clampedT);
  const atlas = buildGqaValueAtlas(sim, group0Head, group1Head);
  const activeFormula: GqaCellFormula | null = opts.activeCell
    ? describeCellFormula(sim, opts.activeCell)
    : null;

  const activeKvGroup: 0 | 1 | null =
    activeHeadIdx !== null
      ? kvHeadForQueryHead(activeHeadIdx)
      : opts.activeCell?.kvHeadIndex !== undefined
        ? opts.activeCell.kvHeadIndex
        : activePart === 'kv-head-0'
          ? 0
          : activePart === 'kv-head-1'
            ? 1
            : null;

  const partCatalog = buildGqaPartCatalog(
    bayId,
    clampedT,
    sim.prefixTokens,
    sim.keyTokens,
    isFp8Mode,
    kvCacheFp8,
    group0Head,
    group1Head,
    sim.rmsEps,
  );

  if (opts.viewAngle === 'three_quarter') {
    return buildGqaSpatialScene({
      bayId,
      clampedT,
      isFp8Mode,
      computeFp8,
      kvCacheFp8,
      activePart,
      activeHeadIdx,
      activeKvGroup,
      group0Head,
      group1Head,
      activeCell: opts.activeCell ?? null,
      activeFormula,
      sim,
      atlas,
      partCatalog,
    });
  }

  const b = new GqaSheetBuilder(bayId);
  const focusSet = buildGqaFocusSet(activeFormula);
  const hasCellTrace = focusSet.active;
  const hasFocus = activePart !== null || hasCellTrace;

  // Sheet elevation: thin flat sheets (thickness 0.07) lying cleanly on the XZ drafting plane
  const BASE_Y = 0.035;
  const OPERAND_LIFT_Y = 0.052;
  const LIFT_Y = 0.068;
  const SHEET_THICK = 0.070;
  const RIBBON_Y_BG = 0.022;
  const RIBBON_Y_ACT = 0.052;

  const sheetY = (hl: number, liftOnPathway = false): number =>
    hl === SHEET_HL_ACTIVE || (liftOnPathway && hl === SHEET_HL_PATHWAY)
      ? LIFT_Y
      : hl === SHEET_HL_OPERAND_A || hl === SHEET_HL_OPERAND_B
        ? OPERAND_LIFT_Y
        : BASE_Y;

  const stageHighlight = (
    tensor: GqaTensorKey,
    partId: string,
    onHeadPathway: boolean,
    isComputeBf16 = true,
  ): number => {
    let fallback = SHEET_HL_NORMAL;
    if (activePart === partId) {
      fallback = SHEET_HL_ACTIVE;
    } else if (activeHeadIdx !== null && onHeadPathway) {
      fallback = SHEET_HL_PATHWAY;
    } else if ((activePart === 'kv-head-0' || activePart === 'kv-head-1') && onHeadPathway) {
      fallback = SHEET_HL_PATHWAY;
    } else if (isFp8Mode && isComputeBf16 && !hasFocus) {
      fallback = SHEET_HL_BF16_GOLD;
    } else if (hasFocus && !onHeadPathway) {
      fallback = SHEET_HL_DIMMED;
    }
    return resolveSheetHighlight(focusSet, { tensor }, fallback);
  };

  // Query length S (= clampedT), reused prefix KV count P, and total attended KV length K = P + S
  const keyCount = sim.keyTokens;
  const prefixCount = sim.prefixTokens;
  const isIncremental = prefixCount > 0;

  // --- Key Spatial Anchors (symmetric two-lane XZ schematic) ---
  const resX = 6.50;
  const zRow0 = -5.62; // Input + RMSNorm
  const zRow1 = -3.95; // W_Q, W_K, W_V projections
  const zQComb = -1.55; // Q0..Q15 & Q16..Q31 combs
  const kvStartZ = -2.45;
  const kvSlicePitch = 0.172;
  const kvSliceDepth = 0.130;
  const kvTrackLen = 16 * kvSlicePitch; // 2.752 -> ends at z = +0.302
  const kvSliceZ = (tok: number) => kvStartZ + (tok - 1) * kvSlicePitch + kvSliceDepth * 0.5;
  const kvActiveTipZ = kvSliceZ(keyCount);
  const zScores = 1.70; // Causal Scores & Softmax ([T×T] or rectangular [S×(P+S)])
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

  // Causal Scores & Softmax extent: square T×T when P === 0; rectangular S×(P+S) when P > 0
  const scoreWidthX = 0.54 + (keyCount / 16) * 1.42;
  const scoreDepthZ = isIncremental
    ? Math.max(0.28, (clampedT / Math.max(1, keyCount)) * scoreWidthX)
    : scoreWidthX;
  const scoresX: readonly [number, number] = [-4.62, 1.72];
  const softmaxX: readonly [number, number] = [-2.12, 4.22];

  // V Out per lane
  const vOutX: readonly [number, number] = [-3.38, 2.96];
  const vOutW = 3.25;

  // ==========================================================================
  // 1. RIGHT RESIDUAL HIGHWAY: Input (X) ---> + Residual (Y)
  // ==========================================================================
  const resInHl = resolveSheetHighlight(
    focusSet,
    { tensor: 'input' },
    activePart === 'residual-in'
      ? SHEET_HL_ACTIVE
      : activePart === 'residual-out'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: resX,
    cy: sheetY(resInHl),
    cz: zRow0,
    ex: 0.72,
    ey: SHEET_THICK,
    ez: 1.08,
    gridCols: sim.d,
    gridRows: clampedT,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resInHl,
    seed: 1.1 + clampedT * 0.1,
    partId: 'residual-in',
    tensorKey: 'input',
    atlasSlice: atlas.input,
    registryKey: 'input',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  const resOutHl = resolveSheetHighlight(
    focusSet,
    { tensor: 'residual' },
    activePart === 'residual-out'
      ? SHEET_HL_ACTIVE
      : activeHeadIdx !== null || activePart === 'o-proj'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: resX,
    cy: sheetY(resOutHl),
    cz: zWO,
    ex: 0.72,
    ey: SHEET_THICK,
    ez: 1.08,
    gridCols: sim.d,
    gridRows: clampedT,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resOutHl,
    seed: 2.3 + clampedT * 0.1,
    partId: 'residual-out',
    tensorKey: 'residual',
    atlasSlice: atlas.residual,
    registryKey: 'residual',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Smooth continuous emerald residual bypass ribbon
  const resRibbonCol = hasCellTrace
    ? resInHl !== SHEET_HL_DIMMED || resOutHl !== SHEET_HL_DIMMED
      ? RIBBON_RESIDUAL
      : RIBBON_DIMMED
    : activePart === 'residual-in' || activePart === 'residual-out'
      ? RIBBON_RESIDUAL_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
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
  // 2. ROW 0: Pre-Attention RMSNorm (U = X_norm * gamma)
  // ==========================================================================
  const rmsX = 0.60;
  const rmsW = 2.60;
  const rmsD = 0.52;
  const rmsHl = stageHighlight(
    'rmsnorm',
    'rmsnorm',
    activeHeadIdx !== null || activePart === 'w-q' || activePart === 'k-proj' || activePart === 'v-proj',
  );
  b.addSheet({
    cx: rmsX,
    cy: sheetY(rmsHl),
    cz: zRow0,
    ex: rmsW,
    ey: SHEET_THICK,
    ez: rmsD,
    gridCols: sim.d,
    gridRows: clampedT,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: rmsHl,
    seed: 3.7,
    partId: 'rmsnorm',
    tensorKey: 'rmsnorm',
    atlasSlice: atlas.rmsnorm,
    registryKey: 'rmsnorm',
    cellFamily: CELL_MOE_AMBER,
  });

  // Flow Input -> RMSNorm
  const in2RmsActive = !hasCellTrace && (rmsHl === SHEET_HL_ACTIVE || activePart === 'residual-in');
  b.flow.addPolylineXZ(
    [
      { x: resX - 0.38, z: zRow0 },
      { x: rmsX + rmsW * 0.5 + 0.04, z: zRow0 },
    ],
    in2RmsActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.13,
    in2RmsActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_RESIDUAL,
    true,
  );

  // ==========================================================================
  // 3. ROW 1: Q / K / V Projection Weight Sheets (Logical 16:1 Width Encoding)
  // ==========================================================================
  const wqX = -1.90;
  const wqW = 4.60;
  const wqD = 0.86;
  const wqOnPath = activeHeadIdx !== null || activePart === 'w-q';
  const wqHl = stageHighlight('w_q', 'w-q', wqOnPath);
  b.addSheet({
    cx: wqX,
    cy: sheetY(wqHl),
    cz: zRow1,
    ex: wqW,
    ey: SHEET_THICK,
    ez: wqD,
    gridCols: sim.qConcatDim,
    gridRows: sim.d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wqHl,
    seed: 5.1 + bayId * 0.3,
    partId: 'w-q',
    tensorKey: 'w_q',
    atlasSlice: atlas.wQ,
    registryKey: 'w_q',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  const wkX = 1.65;
  const wvX = 2.65;
  const wkvW = 0.54;
  const wkvD = 0.86;

  const wkOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'k-proj';
  const wkHl = stageHighlight('w_k', 'k-proj', wkOnPath);
  b.addSheet({
    cx: wkX,
    cy: sheetY(wkHl),
    cz: zRow1,
    ex: wkvW,
    ey: SHEET_THICK,
    ez: wkvD,
    gridCols: sim.nKv * sim.dHead,
    gridRows: sim.d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wkHl,
    seed: 6.4 + bayId * 0.2,
    partId: 'k-proj',
    tensorKey: 'w_k',
    atlasSlice: atlas.wK,
    registryKey: 'w_k',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  const wvOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'v-proj';
  const wvHl = stageHighlight('w_v', 'v-proj', wvOnPath);
  b.addSheet({
    cx: wvX,
    cy: sheetY(wvHl),
    cz: zRow1,
    ex: wkvW,
    ey: SHEET_THICK,
    ez: wkvD,
    gridCols: sim.nKv * sim.dHead,
    gridRows: sim.d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wvHl,
    seed: 7.9 + bayId * 0.2,
    partId: 'v-proj',
    tensorKey: 'w_v',
    atlasSlice: atlas.wV,
    registryKey: 'w_v',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Smooth curves RMSNorm -> W_Q, W_K, W_V
  const projFanColor = (onPath: boolean): FlowRibbonColor =>
    onPath && hasFocus && !hasCellTrace ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET;

  b.flow.addCubicXZ(
    { x: rmsX - 0.55, z: zRow0 + rmsD * 0.5 + 0.02 },
    { x: rmsX - 0.55, z: -4.75 },
    { x: wqX, z: -4.75 },
    { x: wqX, z: zRow1 - wqD * 0.5 - 0.04 },
    wqOnPath && hasFocus && !hasCellTrace ? RIBBON_Y_ACT : RIBBON_Y_BG,
    wqOnPath && hasFocus && !hasCellTrace ? 0.12 : 0.09,
    projFanColor(wqOnPath || activePart === 'rmsnorm'),
    true,
  );
  b.flow.addCubicXZ(
    { x: rmsX + 0.35, z: zRow0 + rmsD * 0.5 + 0.02 },
    { x: rmsX + 0.35, z: -4.82 },
    { x: wkX, z: -4.82 },
    { x: wkX, z: zRow1 - wkvD * 0.5 - 0.04 },
    wkOnPath && hasFocus && !hasCellTrace ? RIBBON_Y_ACT : RIBBON_Y_BG,
    wkOnPath && hasFocus && !hasCellTrace ? 0.105 : 0.08,
    projFanColor(wkOnPath || activePart === 'rmsnorm'),
    true,
  );
  b.flow.addCubicXZ(
    { x: rmsX + 0.85, z: zRow0 + rmsD * 0.5 + 0.02 },
    { x: rmsX + 0.85, z: -4.88 },
    { x: wvX, z: -4.88 },
    { x: wvX, z: zRow1 - wkvD * 0.5 - 0.04 },
    wvOnPath && hasFocus && !hasCellTrace ? RIBBON_Y_ACT : RIBBON_Y_BG,
    wvOnPath && hasFocus && !hasCellTrace ? 0.105 : 0.08,
    projFanColor(wvOnPath || activePart === 'rmsnorm'),
    true,
  );

  // ==========================================================================
  // 4. ROW 2: 32 Individually Selectable Q Heads (2x16) & 2 Shared KV Tracks
  // ==========================================================================
  for (let qIdx = 0; qIdx < 32; qIdx++) {
    const grp = kvHeadForQueryHead(qIdx);
    const hx = qHeadCenterX(qIdx);

    let fallbackHl = SHEET_HL_NORMAL;
    if (activeHeadIdx === qIdx) {
      fallbackHl = SHEET_HL_ACTIVE;
    } else if (activePart === 'q-proj' || activePart === 'w-q') {
      fallbackHl = SHEET_HL_ACTIVE;
    } else if (activePart === `kv-head-${grp}`) {
      fallbackHl = SHEET_HL_PATHWAY;
    } else if (hasFocus) {
      fallbackHl = SHEET_HL_DIMMED;
    } else if (isFp8Mode) {
      fallbackHl = SHEET_HL_BF16_GOLD;
    }
    const hl = resolveSheetHighlight(
      focusSet,
      { tensor: 'q', headIndex: qIdx, kvHeadIndex: grp },
      fallbackHl,
    );

    b.addSheet({
      cx: hx,
      cy: sheetY(hl),
      cz: zQComb,
      ex: qWidth,
      ey: SHEET_THICK,
      ez: qDepth,
      gridCols: sim.dHead,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: hl,
      seed: bayId * 0.5 + qIdx * 0.73 + clampedT * 0.15,
      partId: 'q-proj',
      headIndex: qIdx,
      kvHeadIndex: grp,
      tensorKey: 'q',
      atlasSlice: atlas.qHeads[qIdx],
      registryKey: `q:${qIdx}`,
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
    const isActiveLane = activeKvGroup === grp && !hasCellTrace;
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
      isActiveLane || activePart === 'w-q' ? 0.115 : 0.08,
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

    const railFallbackHl = isDirectKv
      ? SHEET_HL_ACTIVE
      : isSharedForHead
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;

    // 2 ghost capacity rails per KV head (K rail + V rail, 1..16 slots) -> fp8Flag = 0
    const khTyped = kh as 0 | 1;
    const kRailHasFocus =
      hasCellTrace && resolveSheetFocusRole(focusSet, { tensor: 'kv_k', kvHeadIndex: khTyped }) !== null;
    const vRailHasFocus =
      hasCellTrace && resolveSheetFocusRole(focusSet, { tensor: 'kv_v', kvHeadIndex: khTyped }) !== null;
    for (const [trackX, railHasFocus] of [
      [kvKCenterX[kh], kRailHasFocus],
      [kvVCenterX[kh], vRailHasFocus],
    ] as const) {
      const railHl = hasCellTrace
        ? railHasFocus
          ? SHEET_HL_NORMAL
          : SHEET_HL_DIMMED
        : railFallbackHl;
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
        kvHeadIndex: khTyped,
        cellFamily: CELL_NEUTRAL,
      });
    }

    // Live token slices k = 1..K (0..P-1 reused prefix + P..K-1 newly computed)
    for (let t = 1; t <= keyCount; t++) {
      const tokIdx = t - 1;
      const isNewKvSlice = isIncremental && tokIdx >= prefixCount;
      const sz = kvSliceZ(t);
      const sliceFallbackHl = isDirectKv
        ? SHEET_HL_ACTIVE
        : isSharedForHead
          ? SHEET_HL_PATHWAY
          : hasFocus && !isKvRelevant
            ? SHEET_HL_DIMMED
            : isNewKvSlice
              ? SHEET_HL_PATHWAY
              : SHEET_HL_NORMAL;
      const kSliceHl = resolveSheetHighlight(
        focusSet,
        { tensor: 'kv_k', kvHeadIndex: khTyped, tokenIndex: tokIdx },
        sliceFallbackHl,
      );
      const vSliceHl = resolveSheetHighlight(
        focusSet,
        { tensor: 'kv_v', kvHeadIndex: khTyped, tokenIndex: tokIdx },
        sliceFallbackHl,
      );
      const kvSliceY = (h: number) =>
        h === SHEET_HL_ACTIVE || h === SHEET_HL_PATHWAY
          ? LIFT_Y
          : h === SHEET_HL_OPERAND_A || h === SHEET_HL_OPERAND_B
            ? OPERAND_LIFT_Y + 0.008
            : BASE_Y + 0.014;

      // K slice
      b.addSheet({
        cx: kvKCenterX[kh],
        cy: kvSliceY(kSliceHl),
        cz: sz,
        ex: kvSliceWidth,
        ey: 0.062,
        ez: kvSliceDepth,
        gridCols: sim.dHead,
        gridRows: 1,
        sheetKind: SHEET_KIND_STANDARD,
        role: SHEET_ROLE_KV_K,
        fp8Flag: kvCacheFp8 ? 1 : 0,
        highlight: kSliceHl,
        seed: t * 1.13 + kh * 4.3 + bayId * 0.1,
        partId: kvPartId,
        kvHeadIndex: khTyped,
        tensorKey: 'kv_k',
        tokenIndex: tokIdx,
        atlasSlice: atlas.kvK[kh][tokIdx],
        registryKey: `kv_k:${kh}:${tokIdx}`,
        cellFamily: CELL_ACTIVATION_GREEN,
      });

      // V slice
      b.addSheet({
        cx: kvVCenterX[kh],
        cy: kvSliceY(vSliceHl),
        cz: sz,
        ex: kvSliceWidth,
        ey: 0.062,
        ez: kvSliceDepth,
        gridCols: sim.dHead,
        gridRows: 1,
        sheetKind: SHEET_KIND_STANDARD,
        role: SHEET_ROLE_KV_V,
        fp8Flag: kvCacheFp8 ? 1 : 0,
        highlight: vSliceHl,
        seed: t * 1.37 + kh * 6.7 + bayId * 0.15,
        partId: kvPartId,
        kvHeadIndex: khTyped,
        tensorKey: 'kv_v',
        tokenIndex: tokIdx,
        atlasSlice: atlas.kvV[kh][tokIdx],
        registryKey: `kv_v:${kh}:${tokIdx}`,
        cellFamily: CELL_SSM_CYAN,
      });
    }

    // Smooth write ribbons W_K -> K_kh and W_V -> V_kh
    const kWriteActive = !hasCellTrace && (activePart === 'k-proj' || isDirectKv || isSharedForHead);
    b.flow.addCubicXZ(
      { x: wkX, z: zRow1 + wkvD * 0.5 + 0.02 },
      { x: wkX, z: -3.02 },
      { x: kvKCenterX[kh], z: -3.02 },
      { x: kvKCenterX[kh], z: kvStartZ - 0.05 },
      kWriteActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      kWriteActive ? 0.095 : 0.068,
      kWriteActive ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );

    const vWriteActive = !hasCellTrace && (activePart === 'v-proj' || isDirectKv || isSharedForHead);
    b.flow.addCubicXZ(
      { x: wvX, z: zRow1 + wkvD * 0.5 + 0.02 },
      { x: wvX, z: -2.92 },
      { x: kvVCenterX[kh], z: -2.92 },
      { x: kvVCenterX[kh], z: kvStartZ - 0.05 },
      vWriteActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      vWriteActive ? 0.095 : 0.068,
      vWriteActive ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );
  }

  // KV depth scale ticks (1, 8, 16, active K) neatly along outer right edge of KV Head 1
  const tickScaleX = kvVCenterX[1] + kvSliceWidth * 0.5 + 0.22;
  b.addGuideLine(tickScaleX, 0.02, kvStartZ, tickScaleX, 0.02, kvStartZ + kvTrackLen, 0.85, LINE_STYLE_GUIDE, 0.50);
  const tickSet = new Set<number>([1, 8, 16, keyCount]);
  for (const tk of Array.from(tickSet).sort((a, c) => a - c)) {
    const tz = kvSliceZ(tk);
    b.addGuideLine(
      tickScaleX - 0.09, 0.02, tz,
      tickScaleX + 0.09, 0.02, tz,
      tk === keyCount ? 1.4 : 0.85,
      tk === keyCount ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
      0.8,
    );
    b.kvDepthTicks.push({
      token: tk,
      world: [tickScaleX + 0.18, 0.02, tz],
    });
  }

  // ==========================================================================
  // 5. ROW 3: Causal Scores ([T×T] or [S×(P+S)]) & Causal Mask + Softmax
  // ==========================================================================
  const groupHeads: readonly [number, number] = [group0Head, group1Head];

  for (let grp = 0; grp < 2; grp++) {
    const grpTyped = grp as 0 | 1;
    const isGrpActive = activeKvGroup === grp;
    const grpHead = groupHeads[grp];
    const scoresFallbackHl =
      activePart === 'attn-scores'
        ? SHEET_HL_ACTIVE
        : isGrpActive
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    const scoresHl = resolveSheetHighlight(
      focusSet,
      { tensor: 'scores', headIndex: grpHead, kvHeadIndex: grpTyped },
      scoresFallbackHl,
    );

    b.addSheet({
      cx: scoresX[grp],
      cy: sheetY(scoresHl, true),
      cz: zScores,
      ex: scoreWidthX,
      ey: SHEET_THICK,
      ez: scoreDepthZ,
      gridCols: keyCount,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_CAUSAL_SCORES,
      role: SHEET_ROLE_SCORES,
      fp8Flag: 0,
      highlight: scoresHl,
      seed: 11.3 + grp * 3.7 + keyCount * 0.4,
      partId: 'attn-scores',
      headIndex: grpHead,
      kvHeadIndex: grpTyped,
      tensorKey: 'scores',
      atlasSlice: atlas.scores[grp],
      registryKey: `scores:${grp}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    const softmaxFallbackHl =
      activePart === 'softmax'
        ? SHEET_HL_ACTIVE
        : isGrpActive || activePart === 'attn-scores'
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    const softmaxHl = resolveSheetHighlight(
      focusSet,
      { tensor: 'softmax', headIndex: grpHead, kvHeadIndex: grpTyped },
      softmaxFallbackHl,
    );

    b.addSheet({
      cx: softmaxX[grp],
      cy: sheetY(softmaxHl, true),
      cz: zScores,
      ex: scoreWidthX,
      ey: SHEET_THICK,
      ez: scoreDepthZ,
      gridCols: keyCount,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
      role: SHEET_ROLE_SOFTMAX,
      fp8Flag: 0,
      highlight: softmaxHl,
      seed: 17.5 + grp * 5.1 + keyCount * 0.3,
      partId: 'softmax',
      headIndex: grpHead,
      kvHeadIndex: grpTyped,
      tensorKey: 'softmax',
      atlasSlice: atlas.softmax[grp],
      registryKey: `softmax:${grp}`,
      cellFamily: CELL_MOE_AMBER,
    });

    // Smooth curve Q_comb (or specific selected Q_i!) -> Scores_grp
    const qSourceX =
      activeHeadIdx !== null && kvHeadForQueryHead(activeHeadIdx) === grp
        ? qHeadCenterX(activeHeadIdx)
        : qGroupCenterX[grp];
    const qToScoresActive = !hasCellTrace && (isGrpActive || activePart === 'attn-scores');
    b.flow.addCubicXZ(
      { x: qSourceX, z: zQComb + qDepth * 0.5 + 0.03 },
      { x: qSourceX, z: 0.10 },
      { x: scoresX[grp] - 0.16, z: 0.15 },
      { x: scoresX[grp] - 0.16, z: zScores - scoreDepthZ * 0.5 - 0.06 },
      qToScoresActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      activeHeadIdx !== null && qToScoresActive ? 0.12 : 0.085,
      activeHeadIdx !== null && qToScoresActive
        ? RIBBON_ACTIVE
        : qToScoresActive
          ? RIBBON_PATHWAY_SOFT
          : hasFocus
            ? RIBBON_DIMMED
            : RIBBON_QUIET,
      true,
    );

    // Smooth curve Shared K_grp (from active 1..K tip!) -> Scores_grp
    const kToScoresActive = !hasCellTrace && (isGrpActive || activePart === 'attn-scores');
    b.flow.addCubicXZ(
      { x: kvKCenterX[grp], z: kvActiveTipZ + kvSliceDepth * 0.5 + 0.04 },
      { x: kvKCenterX[grp], z: Math.max(kvActiveTipZ + 0.45, 0.42) },
      { x: scoresX[grp] + 0.18, z: 0.35 },
      { x: scoresX[grp] + 0.18, z: zScores - scoreDepthZ * 0.5 - 0.06 },
      kToScoresActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      kToScoresActive ? 0.115 : 0.075,
      kToScoresActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );

    // Smooth horizontal ribbon Scores_grp -> Softmax_grp
    const s2sActive = !hasCellTrace && (isGrpActive || activePart === 'attn-scores' || activePart === 'softmax');
    b.flow.addPolylineXZ(
      [
        { x: scoresX[grp] + scoreWidthX * 0.5 + 0.04, z: zScores },
        { x: softmaxX[grp] - scoreWidthX * 0.5 - 0.05, z: zScores },
      ],
      s2sActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      s2sActive ? 0.115 : 0.08,
      s2sActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );
  }

  // ==========================================================================
  // 6. ROW 4: Weighted V Output (2 lanes x 16*4) -> Concat [128]
  // ==========================================================================
  for (let grp = 0; grp < 2; grp++) {
    const grpTyped = grp as 0 | 1;
    const isGrpActive = activeKvGroup === grp;
    const vOutFallbackHl =
      activePart === 'weighted-v'
        ? SHEET_HL_ACTIVE
        : isGrpActive || activePart === 'softmax'
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : isFp8Mode
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    const vOutHl = resolveSheetHighlight(
      focusSet,
      { tensor: 'v_out', kvHeadIndex: grpTyped },
      vOutFallbackHl,
    );

    b.addSheet({
      cx: vOutX[grp],
      cy: sheetY(vOutHl, true),
      cz: zVOut,
      ex: vOutW,
      ey: SHEET_THICK,
      ez: 0.54,
      gridCols: 16 * sim.dHead,
      gridRows: clampedT,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: vOutHl,
      seed: 23.1 + grp * 2.9 + clampedT * 0.2,
      partId: 'weighted-v',
      kvHeadIndex: grpTyped,
      tensorKey: 'v_out',
      atlasSlice: atlas.vOut[grp],
      registryKey: `v_out:${grp}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    // Ribbon Softmax_grp -> V Out_grp
    const sm2vActive = !hasCellTrace && (isGrpActive || activePart === 'softmax' || activePart === 'weighted-v');
    b.flow.addCubicXZ(
      { x: softmaxX[grp], z: zScores + scoreDepthZ * 0.5 + 0.03 },
      { x: softmaxX[grp], z: 2.95 },
      { x: vOutX[grp] + 0.35, z: 2.95 },
      { x: vOutX[grp] + 0.35, z: zVOut - 0.30 },
      sm2vActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
      sm2vActive ? 0.115 : 0.08,
      sm2vActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );

    // Smooth curved V-cache bypass ribbon from Shared V_grp -> V Out_grp
    const vBypassX = grp === 0 ? -0.82 : 5.56;
    const v2vActive = !hasCellTrace && (isGrpActive || activePart === 'weighted-v');
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
      v2vActive ? 0.10 : 0.07,
      v2vActive ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );
  }

  // Concat [4,096] (miniature [128, T])
  const concatX = -0.20;
  const concatW = 5.60;
  const concatD = 0.50;
  const concatOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'weighted-v' || activePart === 'o-proj';
  const concatHl = stageHighlight('concat', 'concat', concatOnPath);
  b.addSheet({
    cx: concatX,
    cy: sheetY(concatHl, true),
    cz: zConcat,
    ex: concatW,
    ey: SHEET_THICK,
    ez: concatD,
    gridCols: sim.qConcatDim,
    gridRows: clampedT,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: concatHl,
    seed: 29.4 + clampedT * 0.2,
    partId: 'concat',
    tensorKey: 'concat',
    atlasSlice: atlas.concat,
    registryKey: 'concat',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  for (let grp = 0; grp < 2; grp++) {
    const laneOn = !hasCellTrace && (activeKvGroup === grp || activePart === 'weighted-v' || activePart === 'concat');
    const destX = grp === 0 ? concatX - 1.40 : concatX + 1.40;
    b.flow.addCubicXZ(
      { x: vOutX[grp], z: zVOut + 0.29 },
      { x: vOutX[grp], z: 4.10 },
      { x: destX, z: 4.10 },
      { x: destX, z: zConcat - concatD * 0.5 - 0.04 },
      laneOn ? RIBBON_Y_ACT : RIBBON_Y_BG,
      laneOn ? 0.115 : 0.08,
      laneOn ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
      true,
    );
  }

  // ==========================================================================
  // 7. ROW 5: W_O [4,096 -> 2,688] (miniature [8, 128]) -> + Residual
  // ==========================================================================
  const woX = 1.60;
  const woW = 3.55;
  const woD = 0.84;
  const woOnPath = activeHeadIdx !== null || activeKvGroup !== null || activePart === 'concat' || activePart === 'residual-out';
  const woHl = stageHighlight('w_o', 'o-proj', woOnPath);
  b.addSheet({
    cx: woX,
    cy: sheetY(woHl, true),
    cz: zWO,
    ex: woW,
    ey: SHEET_THICK,
    ez: woD,
    gridCols: sim.qConcatDim,
    gridRows: sim.d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: woHl,
    seed: 37.2 + bayId * 0.4,
    partId: 'o-proj',
    tensorKey: 'w_o',
    atlasSlice: atlas.wO,
    registryKey: 'w_o',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  const c2oActive = !hasCellTrace && (woOnPath || activePart === 'o-proj');
  b.flow.addCubicXZ(
    { x: concatX + 0.80, z: zConcat + concatD * 0.5 + 0.02 },
    { x: concatX + 0.80, z: 5.16 },
    { x: woX, z: 5.16 },
    { x: woX, z: zWO - woD * 0.5 - 0.04 },
    c2oActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
    c2oActive ? 0.12 : 0.085,
    c2oActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
    true,
  );

  // W_O -> + Residual
  const o2rActive = !hasCellTrace && (activePart === 'o-proj' || activePart === 'residual-out' || activeHeadIdx !== null);
  b.flow.addPolylineXZ(
    [
      { x: woX + woW * 0.5 + 0.04, z: zWO },
      { x: resX - 0.40, z: zWO },
    ],
    o2rActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
    o2rActive ? 0.135 : 0.105,
    o2rActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_RESIDUAL,
    true,
  );

  // ==========================================================================
  // 8. LIVE CELL-LEVEL HIGHLIGHTS & UPSTREAM DEPENDENCY THREADS (Increment 2)
  // ==========================================================================
  if (activeFormula) {
    const targetRect = resolveSpanWorldRect(b.sheetRegistry, activeFormula.targetSpan, displayedHeads);
    if (targetRect) {
      // Crisp oxide-red frame around the exact picked/stepped cell
      b.flow.addCellHighlightXZ(
        targetRect.cx,
        targetRect.cz,
        targetRect.ex,
        targetRect.ez,
        targetRect.topY + 0.009,
        CELL_TARGET_FILL,
        CELL_TARGET_FRAME,
        0.028,
      );

      // Highlight each upstream operand row/column/cell and connect with thin crisp dependency threads
      for (const upSpan of activeFormula.upstreamSpans) {
        const upRect = resolveSpanWorldRect(b.sheetRegistry, upSpan, displayedHeads);
        if (!upRect) continue;
        drawOperandDependencyXZ(b.flow, upRect, targetRect, upSpan.role === 'operand_a');
      }
    }
  }

  // ==========================================================================
  // 9. EXTERIOR PROGRESSIVE LABELS (with Increment 1 Polish #3 active head tags)
  // ==========================================================================
  const selectedHeadLabel =
    activeHeadIdx !== null
      ? `${formatQueryHeadId(activeHeadIdx)} → KV${kvHeadForQueryHead(activeHeadIdx)}`
      : null;

  const grp0Selected = activeHeadIdx !== null && activeHeadIdx < 16;
  const grp1Selected = activeHeadIdx !== null && activeHeadIdx >= 16;

  const g0Tag = formatQueryHeadId(group0Head);
  const g1Tag = formatQueryHeadId(group1Head);

  b.landmarks.push(
    // Primary exterior labels (1-3 words, strictly outside sheet edges, zero face occlusion)
    { id: 'g-res-in', label: 'Input (2,688)', world: [resX, 0.08, zRow0 - 0.68], partId: 'residual-in', align: 'top', tier: 'primary' },
    { id: 'g-rmsnorm', label: 'RMSNorm', world: [rmsX, 0.08, zRow0 - 0.40], partId: 'rmsnorm', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wq', label: 'W_Q', world: [wqX - wqW * 0.5 - 0.18, 0.08, zRow1], partId: 'w-q', align: 'left', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wk', label: 'W_K', world: [wkX, 0.08, zRow1 - 0.48], partId: 'k-proj', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },
    { id: 'g-wv', label: 'W_V', world: [wvX, 0.08, zRow1 - 0.48], partId: 'v-proj', align: 'top', tier: 'primary', tone: isFp8Mode ? 'gold' : 'default' },

    {
      id: 'g-q0',
      label: grp0Selected ? `Q₀..Q₁₅ · ${selectedHeadLabel}` : 'Q₀..Q₁₅ (×16)',
      world: [qGroupCenterX[0], 0.08, zQComb - qDepth * 0.5 - 0.24],
      partId: grp0Selected ? activePart ?? 'q-proj' : 'q-proj',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'g-q1',
      label: grp1Selected ? `Q₁₆..Q₃₁ · ${selectedHeadLabel}` : 'Q₁₆..Q₃₁ (×16)',
      world: [qGroupCenterX[1], 0.08, zQComb - qDepth * 0.5 - 0.24],
      partId: grp1Selected ? activePart ?? 'q-proj' : 'q-proj',
      align: 'top',
      tier: 'primary',
    },

    {
      id: 'g-kv0',
      label: isIncremental
        ? kvCacheFp8
          ? `KV₀ · FP8 (${prefixCount}+${clampedT})`
          : `KV₀ (${prefixCount} cached + ${clampedT})`
        : kvCacheFp8
          ? `KV₀ · FP8 (T=${clampedT})`
          : `KV₀ (T=${clampedT})`,
      world: [kvPairCenterX[0], 0.08, kvStartZ - 0.18],
      partId: 'kv-head-0',
      align: 'top',
      tier: 'primary',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },
    {
      id: 'g-kv1',
      label: isIncremental
        ? kvCacheFp8
          ? `KV₁ · FP8 (${prefixCount}+${clampedT})`
          : `KV₁ (${prefixCount} cached + ${clampedT})`
        : kvCacheFp8
          ? `KV₁ · FP8 (T=${clampedT})`
          : `KV₁ (T=${clampedT})`,
      world: [kvPairCenterX[1], 0.08, kvStartZ - 0.18],
      partId: 'kv-head-1',
      align: 'top',
      tier: 'primary',
      tone: kvCacheFp8 ? 'muted' : 'cyan',
    },

    // Active head tags on Scores & Softmax (rectangular S×K in incremental decode)
    {
      id: 'g-scores',
      label: grp0Selected
        ? `Scores · ${g0Tag} (${clampedT}×${keyCount})`
        : `Scores (${clampedT}×${keyCount})`,
      world: [scoresX[0] - scoreWidthX * 0.5 - 0.16, 0.08, zScores],
      partId: 'attn-scores',
      align: 'left',
      tier: 'primary',
    },
    {
      id: 'g-softmax',
      label: grp1Selected ? `Softmax · ${g1Tag}` : 'Softmax',
      world: [5.62, 0.08, zScores],
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

  // Additional concise active-group head tag beside the focused group's Scores / Softmax sheet
  if (grp0Selected) {
    b.landmarks.push({
      id: 'g-softmax-active',
      label: `Softmax · ${g0Tag}`,
      world: [softmaxX[0], 0.08, zScores - scoreDepthZ * 0.5 - 0.14],
      partId: 'softmax',
      align: 'top',
      tier: 'primary',
    });
  } else if (grp1Selected) {
    b.landmarks.push({
      id: 'g-scores-active',
      label: `Scores · ${g1Tag} (${clampedT}×${keyCount})`,
      world: [scoresX[1], 0.08, zScores - scoreDepthZ * 0.5 - 0.14],
      partId: 'attn-scores',
      align: 'top',
      tier: 'primary',
    });
  }

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
      aWorld: [scoresX[calGrp] - scoreWidthX * 0.5, 0.06, zScores + scoreDepthZ * 0.5 + 0.12],
      bWorld: [scoresX[calGrp] + scoreWidthX * 0.5, 0.06, zScores + scoreDepthZ * 0.5 + 0.12],
      label: isIncremental ? `Q=${clampedT} × K=${keyCount}` : `T = ${clampedT}`,
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

  const packed = b.pack();
  const reg = (k: string) => b.sheetRegistry.get(k);
  const focusGrp: 0 | 1 = activeKvGroup ?? (activeHeadIdx !== null ? kvHeadForQueryHead(activeHeadIdx) : 0);
  const focusHead = activeHeadIdx ?? (focusGrp === 0 ? group0Head : group1Head);
  const kvKSlices = Array.from({ length: keyCount }, (_, t) => reg(`kv_k:${focusGrp}:${t}`));
  const kvVSlices = Array.from({ length: keyCount }, (_, t) => reg(`kv_v:${focusGrp}:${t}`));

  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('input'), reg('rmsnorm')], 0.48, 0.10, 0.48, GQA_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('rmsnorm'), reg('w_q'), reg('w_k'), reg('w_v'), reg(`q:${focusHead}`)], 0.48, 0.10, 0.48, GQA_FIT_BOUNDS),
    2: computeSheetGroupBounds([reg(`q:${focusHead}`), ...kvKSlices, reg(`scores:${focusGrp}`)], 0.48, 0.10, 0.48, GQA_FIT_BOUNDS),
    3: computeSheetGroupBounds([reg(`scores:${focusGrp}`), reg(`softmax:${focusGrp}`)], 0.48, 0.10, 0.48, GQA_FIT_BOUNDS),
    4: computeSheetGroupBounds([reg(`softmax:${focusGrp}`), ...kvVSlices, reg(`v_out:${focusGrp}`)], 0.48, 0.10, 0.48, GQA_FIT_BOUNDS),
    5: computeSheetGroupBounds([reg('concat'), reg('w_o'), reg('residual')], 0.48, 0.10, 0.48, GQA_FIT_BOUNDS),
  };

  return {
    kind: 'cutaway_gqa',
    bayId,
    ...packed,
    fitBounds: GQA_FIT_BOUNDS,
    opFocusBounds,
    landmarks: b.landmarks,
    kvDepthTicks: b.kvDepthTicks,
    calipers: b.calipers,
    valueAtlas: atlas,
    gqaSim: sim,
    activeFormula,
    solidPickBoxes: b.pickBoxes,
    partCatalog,
    gqaStats: {
      bayId,
      qHeadCount: 32,
      qGroupCount: 2,
      qHeadsPerGroup: 16,
      kvHeadCount: 2,
      gqaRatio: 16,
      kvTokenSliceCount: 2 * keyCount,
      computeFp8,
      kvCacheFp8,
      usesSmoothRibbons: true,
      voxelArrowCount: 0,
      selectedQueryHead: activeHeadIdx,
      activeGroup0Head: group0Head,
      activeGroup1Head: group1Head,
    },
  };
}

function buildGqaPartCatalog(
  bayId: 6 | 43,
  clampedT: number,
  prefixTokens: number,
  keyTokens: number,
  isFp8Mode: boolean,
  kvCacheFp8: boolean,
  group0Head: number,
  group1Head: number,
  rmsEps: number,
): Record<string, CutawayPartInfo> {
  const g0Tag = formatQueryHeadId(group0Head);
  const g1Tag = formatQueryHeadId(group1Head);
  const epsText = formatEpsLabel(rmsEps);
  const isIncremental = prefixTokens > 0;
  const perBayKiB = ((kvCacheFp8 ? 512 : 1024) * keyTokens) / 1024;
  const perBaySlope = kvCacheFp8 ? '+0.5 KiB/tok (FP8 KV)' : '+1 KiB/tok (BF16 KV)';
  const kvRangeLabel = isIncremental
    ? `${prefixTokens} reused + ${clampedT} new = ${keyTokens}`
    : `t = 1 … ${keyTokens}`;
  const attnShapeLabel = `${clampedT} × ${keyTokens}`;
  const precisionMetaLine = isFp8Mode
    ? 'Production precision metadata (§4.2): GQA compute BF16, KV storage FP8 · Live cells: miniature Float32 sim (d=8, 32Q/2KV, d_h=4)'
    : 'Production precision metadata: BF16 weights, activations & KV · Live cells: miniature Float32 sim (d=8, 32Q/2KV, d_h=4)';

  const partCatalog: Record<string, CutawayPartInfo> = {
    'residual-in': {
      id: 'residual-in',
      title: 'Residual Input (2,688)',
      badge: `Bay ${formatBayId(bayId)} · GQA`,
      summary:
        'Incoming 2,688-channel residual stream (d=8 in miniature Float32 simulator) already carrying sequential order from the preceding Mamba-2 feeder layer.',
      specs: [
        `Preceding BF16 feeder: Bay ${formatBayId(bayId - 1)} (Mamba-2)`,
        'NoPE: no RoPE or learned positional embedding added',
        precisionMetaLine,
      ],
    },
    rmsnorm: {
      id: 'rmsnorm',
      title: `Pre-Attention RMSNorm (ε = ${epsText}) · NoPE`,
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16',
      summary:
        `Normalizes each token vector by √(mean(X²) + ${epsText}) (X_norm) then multiplies by non-unit learned scale γ → U before Q/K/V projections. Hover any cell to inspect exact Float32 math.`,
      specs: [
        `Production width: 2,688 · Miniature live sim: d = 8 · Effective ε = ${epsText}`,
        precisionMetaLine,
      ],
    },
    'w-q': {
      id: 'w-q',
      title: 'Query Projection Matrix W_Q',
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16 Weights',
      summary:
        'Projects the normalized hidden state into 32 query heads (production 2,688 → 4,096; miniature Float32 sim 8 → 128), split across Group 0 (Q00–Q15) and Group 1 (Q16–Q31).',
      specs: [
        '16× wider output than W_K or W_V (exact 16:1 GQA ratio)',
        precisionMetaLine,
      ],
    },
    'q-proj': {
      id: 'q-proj',
      title: 'Query Heads Q₀₀..Q₃₁ (2 Groups × 16)',
      badge: '32 Heads · 16:1 GQA',
      summary:
        'Thirty-two individually addressable query heads. Click any head or cell to update its group’s live Float32 Scores & Softmax textures and inspect exact dot-product equations.',
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
        'Projects hidden state down to just 2 key heads (production 2,688 → 256; miniature sim 8 → 8) — 16× narrower than W_Q — writing one shared key slice per token.',
      specs: [
        'Shape: production 2,688 → 256 (2 KV × 128) · live sim 8 → 8 (2 KV × 4)',
        precisionMetaLine,
      ],
    },
    'v-proj': {
      id: 'v-proj',
      title: 'Value Projection Matrix W_V (16× Narrower)',
      badge: isFp8Mode ? 'Compute BF16 · KV FP8' : 'BF16 Weights',
      summary:
        'Projects hidden state down to 2 value heads (production 2,688 → 256; miniature sim 8 → 8), appending one shared value slice per token onto KV₀ and KV₁.',
      specs: [
        'Shape: production 2,688 → 256 (2 KV × 128) · live sim 8 → 8 (2 KV × 4)',
        precisionMetaLine,
      ],
    },
    'kv-head-0': {
      id: 'kv-head-0',
      title: `Shared KV Head 0 (${kvRangeLabel})`,
      badge: kvCacheFp8 ? 'FP8 KV Metadata' : 'BF16 KV Metadata',
      summary:
        isIncremental
          ? `Shared Key (K₀) and Value (V₀) track reusing ${prefixTokens} cached prefix slots (0 compute) plus ${clampedT} newly appended slot(s), attended by all 16 query heads in Group 0 (Q00–Q15).`
          : 'Shared Key (K₀) and Value (V₀) history track attended by all 16 query heads in Group 0 (Q00–Q15). Hover any token slice cell to inspect its exact Float32 projection.',
      specs: [
        `Shared by 16 query heads: Q00–Q15 (16:1 GQA sharing)`,
        `This GQA bay (KV₀ + KV₁): ${perBayKiB} KiB at K=${keyTokens} (${perBaySlope})`,
        precisionMetaLine,
      ],
    },
    'kv-head-1': {
      id: 'kv-head-1',
      title: `Shared KV Head 1 (${kvRangeLabel})`,
      badge: kvCacheFp8 ? 'FP8 KV Metadata' : 'BF16 KV Metadata',
      summary:
        isIncremental
          ? `Shared Key (K₁) and Value (V₁) track reusing ${prefixTokens} cached prefix slots (0 compute) plus ${clampedT} newly appended slot(s), attended by all 16 query heads in Group 1 (Q16–Q31).`
          : 'Shared Key (K₁) and Value (V₁) history track attended by all 16 query heads in Group 1 (Q16–Q31). Hover any token slice cell to inspect its exact Float32 projection.',
      specs: [
        `Shared by 16 query heads: Q16–Q31 (16:1 GQA sharing)`,
        `This GQA bay (KV₀ + KV₁): ${perBayKiB} KiB at K=${keyTokens} (${perBaySlope})`,
        precisionMetaLine,
      ],
    },
    'attn-scores': {
      id: 'attn-scores',
      title: `Causal Attention Scores QKᵀ / √d (${attnShapeLabel})`,
      badge: `${g0Tag} | ${g1Tag}`,
      summary:
        isIncremental
          ? `Live rectangular [${clampedT}×${keyTokens}] Float32 score matrix: ${clampedT} newly computed query row(s) attending across ${prefixTokens} reused prefix keys + ${clampedT} new key(s).`
          : 'Live Float32 scaled dot-product score matrices for the active head in each group. Hover or click any cell (q, k) to trace Q[:, q] · K[:, k] / √d or inspect masked −∞ upper-triangle cells.',
      specs: [
        `Showing live matrices for Group 0: ${g0Tag} and Group 1: ${g1Tag}`,
        isIncremental
          ? `Cached incremental attention: ${clampedT} query row(s) × ${keyTokens} keys (no recomputed prefix rows)`
          : 'Causal mask strictly sets col > row (future tokens) to −∞',
        precisionMetaLine,
      ],
    },
    softmax: {
      id: 'softmax',
      title: `Causal Mask & Row-Wise Softmax (${attnShapeLabel})`,
      badge: `${g0Tag} | ${g1Tag}`,
      summary:
        'Live numerically stable row-wise Softmax over causal keys 0..q so each row sums to 1.000 (upper triangle strictly 0.000). Hover any cell to inspect exact exponential weights.',
      specs: [
        `Showing live probabilities for Group 0: ${g0Tag} and Group 1: ${g1Tag}`,
        precisionMetaLine,
      ],
    },
    'weighted-v': {
      id: 'weighted-v',
      title: 'Weighted Value Output (2 Banks × 16 Heads)',
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16',
      summary:
        'Multiplies each head’s Softmax weights by its group’s shared cached V track (V₀ for Q00–Q15, V₁ for Q16–Q31). Hover any cell to trace the weighted sum across cached tokens 0..q.',
      specs: [
        'Production: 16 × 128 = 2,048/bank · Live sim: 16 × 4 = 64/bank',
        precisionMetaLine,
      ],
    },
    concat: {
      id: 'concat',
      title: 'Concatenate 32 Heads [4,096]',
      badge: '32 Heads Concat',
      summary:
        'Concatenates all 32 per-head weighted value outputs side-by-side (production 32×128 = 4,096 ≠ 2,688; live sim 32×4 = 128 ≠ 8) before down-projecting through W_O.',
      specs: [
        'Explicitly wider than model residual width until W_O restores d',
        precisionMetaLine,
      ],
    },
    'o-proj': {
      id: 'o-proj',
      title: 'Output Projection W_O (4,096 → 2,688)',
      badge: isFp8Mode ? 'BF16 (§4.2)' : 'BF16 Weights',
      summary:
        'Projects the concatenated 32-head output back to model residual width (production 4,096 → 2,688; live sim 128 → 8) before merging onto Input X.',
      specs: [
        'Production shape: 4,096 × 2,688 (~11.0M params) · Live sim: 8 × 128',
        precisionMetaLine,
      ],
    },
    'residual-out': {
      id: 'residual-out',
      title: '+ Residual Stream Merge (Y = X + Attn_out)',
      badge: 'Y = X + Attn',
      summary:
        'Adds the projected GQA update back onto the continuous residual highway. Hover any cell (ch, t) to see exact X[ch,t] + Attn_out[ch,t] = Y[ch,t] Float32 values.',
      specs: [
        'Operation: Y = X + W_O(Concat(Attn(RMSNorm(X))))',
        precisionMetaLine,
      ],
    },
  };

  for (let i = 0; i < 32; i++) {
    const kv = kvHeadForQueryHead(i);
    const label = formatQueryHeadId(i);
    const groupRange = kv === 0 ? 'Q00–Q15' : 'Q16–Q31';
    partCatalog[`q-head-${i}`] = {
      id: `q-head-${i}`,
      title: `Query Head ${label} → Shared KV Head ${kv}`,
      badge: `Group ${kv} · 16:1 GQA`,
      summary:
        `Query head ${label} attends across all ${keyTokens} cached tokens in Shared KV Head ${kv} (shared with ${groupRange}). Group ${kv}’s Scores & Softmax sheets now display ${label}’s live Float32 matrices.`,
      specs: [
        `Mapping: ${label} → KV Head ${kv} (K${kv}, V${kv} · ${keyTokens} live slices, ${kvCacheFp8 ? 'FP8 metadata' : 'BF16 metadata'})`,
        `Live Scores & Softmax: ${attnShapeLabel} ${isIncremental ? 'rectangular cached' : 'lower-triangular'} Float32 attention`,
        `Output slice: production [${i * 128}..${(i + 1) * 128 - 1}] · live sim [${i * 4}..${(i + 1) * 4 - 1}]`,
        precisionMetaLine,
      ],
    };
  }

  return partCatalog;
}
