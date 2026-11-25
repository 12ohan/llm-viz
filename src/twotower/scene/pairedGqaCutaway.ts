/**
 * Asymmetric Paired 35/65 TwoTower GQA Cutaway (`src/twotower/scene/pairedGqaCutaway.ts`)
 *
 * Builds a unified `TwoTowerSpatialScene` for any of the 6 aligned GQA bays
 * (`{6, 13, 20, 27, 34, 43}`, 0-based layers `{5, 12, 19, 26, 33, 42}`):
 *   - Left Wing (35% width, `x ∈ [-13.5, -3.5]`): Context Tower provenance (`context_computed` or
 *     `context_reconstructed`, or persistent Context KV cache + explicit pruned notice when
 *     historical Context traces were pruned — NEVER substituting standalone Nano fixtures).
 *   - Center Logical Conditioning Bridge (`x ∈ [-3.5, -1.8]`): Read-only alignment ribbons carrying
 *     `KV[0..P-1]` (diffusion) or `KV[0..P-2]` (Mock-AR) from Context's persistent KV cache into
 *     Denoiser's concatenated KV bank — explicitly labeled as logical read-only prefix conditioning
 *     (`ownerRole: 'context_tower'`), NOT requiring a physical inter-device copy.
 *   - Right Wing (65% width, `x ∈ [-1.8, +13.5]`): Active Denoiser Tower computation
 *     (`denoiser_scratch`) with independent Denoiser `W_Q`/`W_K`/`W_V`/`W_O`, pre-RMSNorm AdaLN
 *     (`X → X_mod → U`), concatenated KV (inherited `0..P-1` prefix + newly projected `P..P+B-1`
 *     block + unused `P+B..23` ghost capacity rail), 32 Q heads, `[B × (P+B)]` Scores & Softmax,
 *     Weighted V, Concat, `W_O`, and Residual Gate output.
 *   - Supports `'paired'`, `'context_solo'`, and `'denoiser_solo'` focus modes.
 *
 * Pure headless TypeScript — zero DOM, zero WebGL2, fully deterministic.
 */

import type { GqaLayerKvCache } from '../../math/cacheContracts';
import {
  clampCellToSim,
  describeCellFormula,
  fmtProb,
  GQA_STEPPER_STEPS,
  inferGqaCellHead,
  isAttentionKeyVisible,
  maxVisibleKeyIndex,
  type AttentionMaskMode,
  type GqaCellFormula,
  type GqaCellRef,
  type GqaSheetSpan,
  type GqaSimResult,
  type GqaTensorKey,
} from '../../math/gqaSim';
import type { CameraPose } from '../../model/state';
import type { AABB, ViewportMetrics } from '../../render/camera';
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  pushBoxFrameXZ,
  pushFreeGuideLine,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import { FlowRibbonBuilder, type FlowRibbonColor } from '../../render/ribbons';
import {
  FRAME_XZ_AXES,
  FRAME_YZ_AXES,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetBasisAxes,
  type SheetFrame3D,
} from '../../render/sheetFrame';
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
  SHEET_REVEAL_FLOATS,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_UNCOMPUTED,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NEUTRAL,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import {
  computeGqaFocusedAccumulation,
  type FocusedAccumulationResult,
} from '../../scene/accumulation';
import { createRg32fAtlas, type AtlasSlice, type Rg32fValueAtlas } from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  buildGqaFocusSet,
  computeGqaSheetReveal,
  fitSpatialGqaCamera,
  type GqaRevealContext,
  type SpatialGqaFitOptions,
} from '../../scene/gqaSpatial';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  drawCellHighlight3D,
  drawOperandDependency3D,
  EMPTY_SHEET_FOCUS_SET,
  resolveSheetHighlight,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  spanFromSheetGrid,
  type SheetFocusSet,
  type SheetGeomRecord,
  type WorldSpan3D,
} from '../../scene/sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from '../../scene/spine';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../ui/overlay';
import type { TwoTowerTraceOrigin } from '../contracts';
import type { TwoTowerSession } from '../session';
import {
  TWOTOWER_MINI_MODEL_INVENTORY,
  buildTwoTowerSelectionIdentity,
  resolveNearestBayOfKind,
  resolveTwoTowerSessionViewState,
  type TwoTowerPairedFocus,
  type TwoTowerProvenanceBadge,
  type TwoTowerSelectionIdentity,
  type TwoTowerSpatialScene,
  type TwoTowerTowerRole,
} from './provenance';
import {
  buildTwoTowerGqaValueAtlas,
  TWOTOWER_GQA_ATLAS_HEIGHT,
  TWOTOWER_GQA_ATLAS_WIDTH,
  type TwoTowerGqaPrimaryAtlasSlices,
  type TwoTowerGqaSecondaryAtlasSlices,
  type TwoTowerGqaValueAtlas,
} from './twoTowerGqaAtlas';

// ============================================================================
// BOUNDS, OPTIONS & DIAGNOSTIC TYPES
// ============================================================================

export const PAIRED_GQA_FIT_BOUNDS: AABB = {
  min: [-13.85, -0.10, -4.10],
  max: [13.95, 3.35, 4.10],
};

export const CONTEXT_SOLO_GQA_FIT_BOUNDS: AABB = {
  min: [-9.25, -0.10, -3.95],
  max: [9.25, 3.20, 3.95],
};

export const DENOISER_SOLO_GQA_FIT_BOUNDS: AABB = {
  min: [-9.85, -0.10, -4.05],
  max: [9.85, 3.35, 4.05],
};

export interface BuildPairedGqaOptions {
  readonly bayId?: number;
  readonly towerFocus?: TwoTowerPairedFocus;
  readonly group0Head?: number;
  readonly group1Head?: number;
  readonly selectedCell?: { readonly tower: 'context' | 'denoiser'; readonly cell: GqaCellRef } | null;
  readonly hoveredCell?: { readonly tower: 'context' | 'denoiser'; readonly cell: GqaCellRef } | null;
  readonly stepperStep?: number | null;
  readonly stepProgress?: number;
  readonly termProgress?: number;
  readonly reducedMotion?: boolean;
}

export interface PairedGqaLengthDiagnostics {
  readonly queryLen: number;
  readonly contextPrefixKvLen: number;
  readonly blockKvLen: number;
  readonly allocatedKvCapacity: number;
  readonly totalAttendedKvLen: number;
  readonly unusedKvCapacity: number;
}

export interface PairedGqaKvRevealRecord {
  readonly sheetIndex: number;
  readonly kvHeadIndex: 0 | 1;
  readonly kind: 'kv_k' | 'kv_v';
  readonly tokenIndex: number;
  readonly traceOrigin: TwoTowerTraceOrigin;
  readonly revealMode: number;
}

export interface PairedGqaDiagnostics {
  readonly bayId: number;
  readonly layerIndex: number;
  readonly towerFocus: TwoTowerPairedFocus;
  readonly modeAtRecord: string;
  readonly attentionMaskMode: AttentionMaskMode;
  readonly queryLen: number;
  readonly contextPrefixKvLen: number;
  readonly blockKvLen: number;
  readonly allocatedKvCapacity: number;
  readonly lengths: PairedGqaLengthDiagnostics;
  readonly contextTraceStatus: 'available' | 'pruned_unavailable';
  readonly contextTraceOrigin: 'context_computed' | 'context_reconstructed' | 'persistent_cache';
  readonly denoiserPassStatus: 'active_on_record' | 'historical_prior_record' | 'none_yet';
  readonly currentContextPrefixLen: number;
  readonly denoiserConditioningPrefixLen: number | null;
  readonly denoiserSourceContextCacheDigest: string | null;
  readonly denoiserSourceRecordIndex: number | null;
  readonly denoiserSourceBlockId: number | null;
  readonly denoiserSourceRefinementIndex: number | null;
  readonly isHistoricalDenoiserInspection: boolean;
  readonly matchesCurrentContextBoundary: boolean;
  readonly historicalProvenanceWarning: string | null;
  readonly hasDenoiserPreNormMod: boolean;
  readonly hasDenoiserResidualGate: boolean;
  readonly conditioningOwnerRole: 'context_tower';
  readonly conditioningConsumerRole: 'denoiser_tower';
  readonly conditioningTransportSemantics: 'logical_read_only_no_physical_copy';
  readonly conditioningNote: string;
  readonly contextParameterFingerprint: string;
  readonly denoiserParameterFingerprint: string;
  readonly prefixKvReveals: readonly PairedGqaKvRevealRecord[];
  readonly blockKvReveals: readonly PairedGqaKvRevealRecord[];
  readonly modelGqaInventory: {
    readonly queryHeads: number;
    readonly kvHeads: number;
    readonly queriesPerKvGroup: number;
    readonly headDim: number;
    readonly displayedScoreHeads: readonly [number, number];
    readonly displayedScoreHeadCount: 2;
  };
  readonly atlasDimensions: readonly [number, number];
  readonly atlasPrimaryMaskedCells: number;
  readonly atlasSecondaryMaskedCells: number;
}

/** Reuse Nano's exact 8-corner projective camera fitter for paired & solo TwoTower GQA views. */
export function fitPairedGqaCamera(
  bounds: AABB,
  metrics: ViewportMetrics,
  options?: SpatialGqaFitOptions,
): CameraPose {
  return fitSpatialGqaCamera(bounds, metrics, options);
}

// ============================================================================
// INTERNAL BUILDER & PALETTE HELPERS
// ============================================================================

const RIBBON_BRIDGE_KV0: FlowRibbonColor = { r: 0.11, g: 0.56, b: 0.52, a: 0.78, borderAlpha: 0.44 };
const RIBBON_BRIDGE_KV1: FlowRibbonColor = { r: 0.16, g: 0.46, b: 0.72, a: 0.78, borderAlpha: 0.44 };
const RIBBON_K_FLOW: FlowRibbonColor = { r: 0.14, g: 0.54, b: 0.49, a: 0.42, borderAlpha: 0.24 };
const RIBBON_V_FLOW: FlowRibbonColor = { r: 0.18, g: 0.46, b: 0.66, a: 0.42, borderAlpha: 0.24 };
const RIBBON_CONTEXT_QUIET: FlowRibbonColor = { r: 0.36, g: 0.48, b: 0.62, a: 0.36, borderAlpha: 0.20 };

interface PairedGqaSheetSpec {
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly axes: SheetBasisAxes;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly sheetKind: number;
  readonly role: number;
  readonly fp8Flag: number;
  readonly highlight: number;
  readonly seed: number;
  readonly partId: string;
  readonly towerRole: TwoTowerTowerRole;
  readonly executionId: string;
  readonly traceOrigin: TwoTowerTraceOrigin;
  readonly semanticTag?: string;
  readonly absoluteTokenOffset?: number;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly tensorKey?: GqaTensorKey;
  readonly tokenIndex?: number;
  readonly atlasSlice?: AtlasSlice;
  readonly registryKey?: string;
  readonly cellFamily: number;
  readonly revealOverride?: readonly [number, number, number, number];
}

class PairedGqaBuilder {
  readonly sheetFloats: number[] = [];
  readonly basisFloats: number[] = [];
  readonly revealFloats: number[] = [];
  readonly solidFloats: number[] = [];
  readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly kvDepthTicks: KvDepthTickAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();
  readonly activeOpPartIds = new Set<string>();
  readonly prefixKvReveals: PairedGqaKvRevealRecord[] = [];
  readonly blockKvReveals: PairedGqaKvRevealRecord[] = [];

  private readonly bayId: number;
  private readonly revealEnabled: boolean;
  private readonly collectActiveOpParts: boolean;

  constructor(bayId: number, revealEnabled: boolean, collectActiveOpParts: boolean) {
    this.bayId = bayId;
    this.revealEnabled = revealEnabled;
    this.collectActiveOpParts = collectActiveOpParts;
  }

  addSheet(
    s: PairedGqaSheetSpec,
    revealCtx: GqaRevealContext | null,
    isDenoiserKvPrefix = false,
    isDenoiserKvBlock = false,
  ): { readonly frame: SheetFrame3D; readonly sheetIndex: number; readonly revealMode: number } {
    const sheetIndex = this.sheetFloats.length / SHEET_INSTANCE_FLOATS;
    const frame = makeSheetFrame3D(
      [s.cx, s.cy, s.cz],
      [s.ex, s.ey, s.ez],
      s.axes.axisU,
      s.axes.axisN,
    );

    const rect = s.atlasSlice ? s.atlasSlice.rect : ([0, 0, 0, 0] as const);
    const scale = s.atlasSlice ? s.atlasSlice.scale : 1.0;
    const modeFlag = s.atlasSlice ? s.atlasSlice.mode : 0;

    this.sheetFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.gridCols, s.gridRows,
      s.sheetKind,
      s.role,
      s.fp8Flag,
      s.highlight,
      s.seed,
      rect[0], rect[1], rect[2], rect[3],
      scale,
      modeFlag, 0,
    );

    this.basisFloats.push(
      frame.axisU[0], frame.axisU[1], frame.axisU[2],
      frame.axisN[0], frame.axisN[1], frame.axisN[2],
    );

    let revealMode: number = SHEET_REVEAL_NONE;
    if (this.revealEnabled) {
      let tuple: readonly [number, number, number, number];
      if (s.revealOverride) {
        tuple = s.revealOverride;
      } else if (isDenoiserKvPrefix) {
        // CRITICAL INVARIANT: inherited Context prefix KV is NEVER ghosted during reveal!
        tuple = [SHEET_REVEAL_NONE, 0, 0, 0];
      } else if (revealCtx) {
        tuple = computeGqaSheetReveal(revealCtx, {
          sheetKind: s.sheetKind,
          tensorKey: s.tensorKey,
          headIndex: s.headIndex,
          kvHeadIndex: s.kvHeadIndex,
          tokenIndex: s.tokenIndex,
        });
      } else {
        tuple = [SHEET_REVEAL_NONE, 0, 0, 0];
      }
      revealMode = tuple[0];
      this.revealFloats.push(tuple[0], tuple[1], tuple[2], tuple[3]);
    }

    if (
      (s.tensorKey === 'kv_k' || s.tensorKey === 'kv_v') &&
      s.kvHeadIndex !== undefined &&
      s.tokenIndex !== undefined
    ) {
      if (isDenoiserKvPrefix) {
        this.prefixKvReveals.push({
          sheetIndex,
          kvHeadIndex: s.kvHeadIndex,
          kind: s.tensorKey,
          tokenIndex: s.tokenIndex,
          traceOrigin: s.traceOrigin,
          revealMode,
        });
      } else if (isDenoiserKvBlock) {
        this.blockKvReveals.push({
          sheetIndex,
          kvHeadIndex: s.kvHeadIndex,
          kind: s.tensorKey,
          tokenIndex: s.tokenIndex,
          traceOrigin: s.traceOrigin,
          revealMode,
        });
      }
    }

    if (
      this.collectActiveOpParts &&
      (s.highlight === SHEET_HL_ACTIVE ||
        s.highlight === SHEET_HL_OPERAND_A ||
        s.highlight === SHEET_HL_OPERAND_B)
    ) {
      this.activeOpPartIds.add(s.partId);
    }

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

    const pickFrame = makeSheetFrame3D(
      frame.center,
      [frame.extent[0], Math.max(frame.extent[1], 0.085), frame.extent[2]],
      frame.axisU,
      frame.axisN,
    );
    const worldAabb = sheetFrameWorldAabb(pickFrame);
    this.pickBoxes.push({
      bayId: this.bayId,
      partId: s.partId,
      headIndex: s.headIndex,
      kvHeadIndex: s.kvHeadIndex,
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
      towerRole: s.towerRole,
      executionId: s.executionId,
      traceOrigin: s.traceOrigin,
      semanticTag: s.semanticTag,
      absoluteTokenOffset: s.absoluteTokenOffset,
      min: worldAabb.min,
      max: worldAabb.max,
      frame: pickFrame,
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
        frame,
      });
    }

    return { frame, sheetIndex, revealMode };
  }

  addGuideLine(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth = 1.0,
    style = LINE_STYLE_GUIDE,
    alpha = 0.55,
  ): void {
    pushFreeGuideLine(this.lineFloats, ax, ay, az, bx, by, bz, cssWidth, style, alpha);
  }

  addBoxFrameXZ(
    x0: number, z0: number, x1: number, z1: number, y: number,
    cssWidth = 1.0, style = LINE_STYLE_GUIDE, alpha = 0.45,
  ): void {
    pushBoxFrameXZ(this.lineFloats, x0, z0, x1, z1, y, cssWidth, style, alpha);
  }

  pack() {
    const flowPacked = this.flow.pack();
    const sheetCount = this.sheetFloats.length / SHEET_INSTANCE_FLOATS;
    return {
      sheetData: new Float32Array(this.sheetFloats),
      sheetCount,
      sheetBasisData: new Float32Array(this.basisFloats),
      ...(this.revealEnabled && this.revealFloats.length === sheetCount * SHEET_REVEAL_FLOATS
        ? { sheetRevealData: new Float32Array(this.revealFloats) }
        : {}),
      solidData: new Float32Array(this.solidFloats),
      solidCount: this.solidFloats.length / SOLID_INSTANCE_FLOATS,
      lineData: new Float32Array(this.lineFloats),
      lineCount: this.lineFloats.length / LINE_INSTANCE_FLOATS,
      ribbonData: new Float32Array(0),
      ribbonCount: 0,
      flowVertices: flowPacked.vertices,
      flowIndices: flowPacked.indices,
    };
  }
}

function fmtHead(h: number): string {
  return `Q${String(h).padStart(2, '0')}`;
}

/** Stepper cell mapper that smoothly sweeps across all query rows 0..T-1 during projection step 1. */
function twoTowerGqaStepperCell(
  sim: GqaSimResult,
  activeHead: number,
  step: number,
  stepProgress: number,
): GqaCellRef {
  const h = Math.max(0, Math.min(31, Math.round(activeHead)));
  const g: 0 | 1 = h < 16 ? 0 : 1;
  const T = sim.tokens;
  const K = sim.keyTokens;
  const p = Math.max(0, Math.min(0.9999, Number.isFinite(stepProgress) ? stepProgress : 0.25));
  const lastTok = Math.max(0, T - 1);

  switch (Math.max(0, Math.min(5, Math.round(step)))) {
    case 0:
      return { tensor: 'rmsnorm', row: Math.min(lastTok, Math.floor(p * T)), col: Math.floor(((p * T) % 1) * sim.d) };
    case 1: {
      // Sweep block query rows 0..T-1 and head channels 0..dHead-1 so new block KV reveals progressively
      const total = T * sim.dHead;
      const idx = Math.min(total - 1, Math.floor(p * total));
      const row = Math.floor(idx / sim.dHead);
      const col = idx % sim.dHead;
      return { tensor: 'q', headIndex: h, kvHeadIndex: g, row, col };
    }
    case 2: {
      const k = Math.min(K - 1, Math.floor(p * K));
      return { tensor: 'scores', headIndex: h, kvHeadIndex: g, row: lastTok, col: k };
    }
    case 3: {
      const k = Math.min(K - 1, Math.floor(p * K));
      return { tensor: 'softmax', headIndex: h, kvHeadIndex: g, row: lastTok, col: k };
    }
    case 4: {
      const r = Math.min(sim.dHead - 1, Math.floor(p * sim.dHead));
      return { tensor: 'v_out', headIndex: h, kvHeadIndex: g, row: lastTok, col: (h % 16) * sim.dHead + r };
    }
    default:
      return { tensor: 'residual', row: lastTok, col: Math.min(sim.d - 1, Math.floor(p * sim.d)) };
  }
}

/** Augment Nano's exact Float32 formula with TwoTower tower role & prefix-vs-block key provenance. */
function augmentGqaFormulaWithProvenance(
  sim: GqaSimResult,
  base: GqaCellFormula,
  tower: 'context' | 'denoiser',
  bayId: number,
  historicalPrefixNote?: string | null,
): GqaCellFormula {
  const towerLabel = tower === 'denoiser' ? 'Denoiser Tower' : 'Context Tower';
  const P = sim.prefixTokens;
  let provenanceTag = '';
  const histQualifier = historicalPrefixNote ? ` [${historicalPrefixNote}]` : '';

  if (base.cell.tensor === 'kv_k' || base.cell.tensor === 'kv_v') {
    const k = base.cell.tokenIndex ?? 0;
    provenanceTag =
      tower === 'denoiser'
        ? k < P
          ? `Inherited Context Prefix KV (pos ${k}${histQualifier})`
          : `Denoiser Scratch Block KV (pos ${k}${histQualifier})`
        : k < P
          ? `Inherited Context Prefix KV (pos ${k})`
          : `Context Segment KV (pos ${k})`;
  } else if (base.cell.tensor === 'scores' || base.cell.tensor === 'softmax') {
    const k = base.cell.col;
    const keyOrigin =
      tower === 'denoiser'
        ? k < P
          ? `Inherited Context Prefix KV (pos ${k}${histQualifier})`
          : `Denoiser Scratch Block KV (pos ${k}${histQualifier})`
        : k < P
          ? `Inherited Context Prefix KV (pos ${k})`
          : `Context Segment KV (pos ${k})`;
    const maskDesc =
      sim.attentionMaskMode === 'bidirectional_block'
        ? 'Bidirectional block visibility (0 masked cells)'
        : 'Causal mask (k <= P + q)';
    provenanceTag = `${keyOrigin} · ${maskDesc}`;
  } else if (base.cell.tensor === 'rmsnorm' && sim.X_mod) {
    provenanceTag = `Pre-RMSNorm AdaLN modulated (X → X_mod → RMSNorm)${histQualifier}`;
  } else if (base.cell.tensor === 'residual' && sim.residualGateApplied) {
    provenanceTag = `Outer AdaLN residual gate active (unmodulated X preserved)${histQualifier}`;
  } else if (historicalPrefixNote && tower === 'denoiser') {
    provenanceTag = historicalPrefixNote;
  }

  return {
    ...base,
    tensorLabel: `${tower === 'denoiser' ? 'Denoiser' : 'Context'} · ${base.tensorLabel}`,
    headBadge: base.headBadge ? `${towerLabel} · ${base.headBadge}` : `${towerLabel} · Bay ${String(bayId).padStart(2, '0')}`,
    subEquation: provenanceTag
      ? `${provenanceTag} · ${base.subEquation}`
      : `${towerLabel} (Bay ${String(bayId).padStart(2, '0')}) · ${base.subEquation}`,
  };
}

/** Resolve a GQA sheet span on the targeted wing (`ctx:` or `den:`). */
function resolvePairedGqaSpan(
  registry: ReadonlyMap<string, SheetGeomRecord>,
  prefix: 'ctx' | 'den',
  span: GqaSheetSpan,
  displayedHeads: readonly [number, number],
): WorldSpan3D | null {
  if (span.tensor === 'kv_k' || span.tensor === 'kv_v') {
    const g = span.kvHeadIndex ?? 0;
    const t0 = span.tokenStart ?? 0;
    const t1 = span.tokenEnd ?? t0;
    const first = registry.get(`${prefix}:${span.tensor}:${g}:${t0}`);
    const last = registry.get(`${prefix}:${span.tensor}:${g}:${t1}`) ?? first;
    if (!first || !last) return null;
    if (t0 === t1) {
      return spanFromSheetGrid(first, 0, 0, span.colStart, span.colEnd);
    }
    const zMin = Math.min(first.cz - first.ez * 0.5, last.cz - last.ez * 0.5);
    const zMax = Math.max(first.cz + first.ez * 0.5, last.cz + last.ez * 0.5);
    const composite: SheetGeomRecord = {
      cx: first.cx,
      cy: first.cy,
      cz: (zMin + zMax) * 0.5,
      ex: first.ex,
      ey: first.ey,
      ez: Math.max(0.05, zMax - zMin),
      gridCols: first.gridCols,
      gridRows: 1,
      frame: makeSheetFrame3D(
        [first.cx, first.cy, (zMin + zMax) * 0.5],
        [first.ex, first.ey, Math.max(0.05, zMax - zMin)],
        FRAME_XZ_AXES.axisU,
        FRAME_XZ_AXES.axisN,
      ),
    };
    return spanFromSheetGrid(composite, 0, 0, span.colStart, span.colEnd);
  }

  let key = `${prefix}:${span.tensor}`;
  if (span.tensor === 'q') {
    key = `${prefix}:q:${span.headIndex ?? 0}`;
  } else if (span.tensor === 'scores' || span.tensor === 'softmax') {
    const g = span.kvHeadIndex ?? ((span.headIndex ?? 0) < 16 ? 0 : 1);
    if (span.headIndex !== undefined && displayedHeads[g] !== span.headIndex) {
      return null;
    }
    key = `${prefix}:${span.tensor}:${g}`;
  } else if (span.tensor === 'v_out') {
    const g = span.kvHeadIndex ?? 0;
    key = `${prefix}:v_out:${g}`;
  }
  const sheet = registry.get(key);
  if (!sheet) return null;
  return spanFromSheetGrid(sheet, span.rowStart, span.rowEnd, span.colStart, span.colEnd);
}

/** Build visibility-consistent `CellPopoverSpec` using `isAttentionKeyVisible` & `maxVisibleKeyIndex`. */
function buildPairedGqaCellPopover(
  sim: GqaSimResult,
  formula: GqaCellFormula,
  tower: 'context' | 'denoiser',
  targetWorld: readonly [number, number, number],
  sourceWorlds: readonly {
    readonly world: readonly [number, number, number];
    readonly role: 'indigo' | 'teal';
  }[],
  focusedAccumulation?: FocusedAccumulationResult | null,
  historicalPrefixTag?: string | null,
): CellPopoverSpec {
  const cell = clampCellToSim(sim, formula.cell);
  const P = sim.prefixTokens;
  const K = sim.keyTokens;
  const { dHead } = sim;
  const histSuffix = historicalPrefixTag && tower === 'denoiser' ? ` · ${historicalPrefixTag}` : '';

  let coordLabel = formula.coordLabel;
  let equation = formula.equation;
  let valueText = `= ${formula.valueText}`;

  if (cell.tensor === 'scores' || cell.tensor === 'softmax') {
    const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
    const h = cell.headIndex ?? (g === 0 ? 0 : 16);
    const q = cell.row;
    const k = cell.col;
    const qAbs = P + q;
    const visible = isAttentionKeyVisible(sim.attentionMaskMode, P, q, k, K);
    const maxVisK = maxVisibleKeyIndex(sim.attentionMaskMode, P, q, K);
    const keySourceTag =
      tower === 'denoiser'
        ? k < P
          ? `Inherited Context Prefix KV (pos ${k}${histSuffix})`
          : `Denoiser Scratch Block KV (pos ${k}${histSuffix})`
        : `Context KV (pos ${k})`;
    coordLabel = `[q=t${qAbs}, k=t${k} · ${keySourceTag}]`;
    if (!visible) {
      equation =
        cell.tensor === 'scores'
          ? `k=${k} > q=${qAbs} · Causal mask (−∞)`
          : `Score=−∞ (k=${k} > q=${qAbs}) → exp(−∞)`;
      valueText = cell.tensor === 'scores' ? '= −∞' : '= 0.000 (masked)';
    } else if (cell.tensor === 'scores') {
      equation = `(${fmtHead(h)}[:,t${qAbs}] · K${g}[:,t${k}]) / √${dHead}`;
    } else {
      equation = `exp(s − max) / Σ_{j=0..${maxVisK}} exp(s_j − max)`;
      valueText = `= ${fmtProb(formula.value)}`;
    }
  } else if (cell.tensor === 'v_out') {
    const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
    const q = cell.row;
    const qAbs = P + q;
    const maxVisK = maxVisibleKeyIndex(sim.attentionMaskMode, P, q, K);
    const h = g * 16 + Math.floor(cell.col / dHead);
    const r = cell.col % dHead;
    coordLabel = `[${fmtHead(h)}, q=t${qAbs}, d=${r}]`;
    equation = `Σ_{k=0..${maxVisK}} Softmax[${fmtHead(h)},t${qAbs},k]·V${g}[${r},k]`;
  } else if (cell.tensor === 'kv_k' || cell.tensor === 'kv_v') {
    const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
    const k = cell.tokenIndex ?? 0;
    const r = cell.col;
    const name = cell.tensor === 'kv_k' ? `K${g}` : `V${g}`;
    const srcLabel =
      k < P
        ? `Inherited Context Prefix KV (pos ${k}${histSuffix})`
        : tower === 'denoiser'
          ? `Denoiser Scratch Block KV (pos ${k}${histSuffix})`
          : `Context Segment KV (pos ${k})`;
    coordLabel = `[${name} t${k}, d=${r}]`;
    equation = k < P ? `${srcLabel} (read-only · 0 compute)` : `${srcLabel} = W_${cell.tensor === 'kv_k' ? 'K' : 'V'}·U`;
  }

  const hasOpA = formula.upstreamSpans.some((s) => s.role === 'operand_a');
  const hasOpB = formula.upstreamSpans.some((s) => s.role === 'operand_b');
  const operandColors: ('indigo' | 'teal' | 'red')[] = [];
  if (hasOpA) operandColors.push('indigo');
  if (hasOpB) operandColors.push('teal');
  operandColors.push('red');

  const isAccumulating = Boolean(focusedAccumulation?.active);
  return {
    world: targetWorld,
    tensorLabel: formula.tensorLabel,
    coordLabel,
    badge: isAccumulating ? focusedAccumulation!.badgeText : (formula.headBadge ?? undefined),
    equation: isAccumulating ? focusedAccumulation!.popoverEquation : equation,
    valueText: isAccumulating ? focusedAccumulation!.valueBadgeText : valueText,
    ...(isAccumulating ? { isAccumulating: true } : {}),
    operandColors,
    ...(sourceWorlds.length > 0 ? { sourceWorlds } : {}),
  };
}

/** Build visibility-consistent `ActiveSheetCellGrid[]` strictly honoring `isAttentionKeyVisible`. */
function buildPairedGqaActiveCellGrids(
  sim: GqaSimResult,
  registry: ReadonlyMap<string, SheetGeomRecord>,
  prefix: 'ctx' | 'den',
  formula: GqaCellFormula | null,
  group0Head: number,
  group1Head: number,
  revealCtx: GqaRevealContext | null,
  focusedAccumulation: FocusedAccumulationResult | null,
): readonly ActiveSheetCellGrid[] {
  const T = sim.tokens;
  const K = sim.keyTokens;
  const P = sim.prefixTokens;
  const { d, dHead } = sim;
  const displayedHeads: readonly [number, number] = [group0Head, group1Head];

  const queryTokLabels = Array.from(
    { length: T },
    (_, q) => `t${P + q} ${sim.tokenLabels[q] ?? ''}`.trim(),
  );
  const keyTokLabels = Array.from(
    { length: K },
    (_, k) => `t${k} ${sim.keyTokenLabels[k] ?? ''}`.trim(),
  );
  const headDimLabels = Array.from({ length: dHead }, (_, r) => `d${r}`);
  const chanLabels = Array.from({ length: d }, (_, i) => `ch${i}`);

  const grids: ActiveSheetCellGrid[] = [];
  const emitted = new Set<string>();

  const pushGrid = (
    regKey: string,
    values: Float32Array,
    sheetInfo: {
      readonly sheetKind: number;
      readonly tensorKey: GqaTensorKey;
      readonly headIndex?: number;
      readonly kvHeadIndex?: 0 | 1;
      readonly tokenIndex?: number;
    },
    opts: {
      readonly masked?: Uint8Array;
      readonly format?: 'signed' | 'prob';
      readonly activeCol?: number;
      readonly activeRow?: number;
      readonly activeValueOverride?: number | null;
      readonly colLabels?: readonly string[];
      readonly rowLabels?: readonly string[];
      readonly colDimLabel?: string;
      readonly rowDimLabel?: string;
    },
  ): void => {
    if (emitted.has(regKey)) return;
    const sheet = registry.get(regKey);
    if (!sheet) return;
    emitted.add(regKey);
    const frame =
      sheet.frame ??
      makeSheetFrame3D([sheet.cx, sheet.cy, sheet.cz], [sheet.ex, sheet.ey, sheet.ez]);

    let revealFields:
      | Pick<
          ActiveSheetCellGrid,
          'revealMode' | 'revealColFrontier' | 'revealRowFrontier' | 'suppressColHeaders' | 'suppressRowHeaders'
        >
      | undefined;
    if (revealCtx?.enabled) {
      const [revealMode, revealColFrontier, revealRowFrontier] = computeGqaSheetReveal(
        revealCtx,
        sheetInfo,
      );
      revealFields = {
        revealMode,
        revealColFrontier,
        revealRowFrontier,
        ...(revealMode === SHEET_REVEAL_UNCOMPUTED
          ? { suppressColHeaders: true, suppressRowHeaders: true }
          : {}),
      };
    }

    grids.push({
      originWorld: frame.center,
      extent: frame.extent,
      axisU: frame.axisU,
      axisN: frame.axisN,
      cols: sheet.gridCols,
      rows: sheet.gridRows,
      values,
      ...(opts.masked ? { masked: opts.masked } : {}),
      format: opts.format ?? 'signed',
      activeCol: opts.activeCol,
      activeRow: opts.activeRow,
      ...revealFields,
      ...(opts.activeValueOverride !== undefined && opts.activeValueOverride !== null
        ? { activeValueOverride: opts.activeValueOverride }
        : {}),
      colLabels: opts.colLabels,
      rowLabels: opts.rowLabels,
      colDimLabel: opts.colDimLabel,
      rowDimLabel: opts.rowDimLabel,
    });
  };

  const emitScoresOrSoftmax = (
    kind: 'scores' | 'softmax',
    g: 0 | 1,
    activeCol?: number,
    activeRow?: number,
    activeValueOverride?: number | null,
  ): void => {
    const h = displayedHeads[g];
    const values = new Float32Array(T * K);
    const masked = new Uint8Array(T * K);
    const src = kind === 'scores' ? sim.Scores : sim.Softmax;
    const headOff = h * T * K;
    for (let q = 0; q < T; q++) {
      for (let k = 0; k < K; k++) {
        const idx = q * K + k;
        values[idx] = src[headOff + idx];
        masked[idx] = isAttentionKeyVisible(sim.attentionMaskMode, P, q, k, K) ? 0 : 1;
      }
    }
    const sheetKind =
      sim.attentionMaskMode === 'bidirectional_block'
        ? SHEET_KIND_STANDARD
        : kind === 'scores'
          ? SHEET_KIND_CAUSAL_SCORES
          : SHEET_KIND_CAUSAL_SOFTMAX;
    pushGrid(
      `${prefix}:${kind}:${g}`,
      values,
      { sheetKind, tensorKey: kind, headIndex: h, kvHeadIndex: g },
      {
        masked,
        format: kind === 'softmax' ? 'prob' : 'signed',
        activeCol,
        activeRow,
        activeValueOverride,
        colLabels: keyTokLabels,
        rowLabels: queryTokLabels,
        colDimLabel: P > 0 ? `K = ${K} (${P}+${T})` : `K = ${K}`,
        rowDimLabel: `Q = ${T}`,
      },
    );
  };

  const emitQHead = (h: number, activeCol?: number, activeRow?: number): void => {
    const values = new Float32Array(T * dHead);
    const headOff = h * dHead * T;
    for (let t = 0; t < T; t++) {
      for (let r = 0; r < dHead; r++) {
        values[t * dHead + r] = sim.Q[headOff + r * T + t];
      }
    }
    pushGrid(
      `${prefix}:q:${h}`,
      values,
      { sheetKind: SHEET_KIND_STANDARD, tensorKey: 'q', headIndex: h, kvHeadIndex: h < 16 ? 0 : 1 },
      {
        format: 'signed',
        activeCol,
        activeRow,
        colLabels: headDimLabels,
        rowLabels: queryTokLabels,
        colDimLabel: `d_h = ${dHead}`,
        rowDimLabel: `${fmtHead(h)} · Q = ${T}`,
      },
    );
  };

  const emitVOut = (
    g: 0 | 1,
    activeCol?: number,
    activeRow?: number,
    activeValueOverride?: number | null,
  ): void => {
    const cols = 16 * dHead;
    const values = new Float32Array(T * cols);
    const baseChan = g * cols;
    for (let q = 0; q < T; q++) {
      for (let c = 0; c < cols; c++) {
        values[q * cols + c] = sim.V_out[(baseChan + c) * T + q];
      }
    }
    pushGrid(
      `${prefix}:v_out:${g}`,
      values,
      { sheetKind: SHEET_KIND_STANDARD, tensorKey: 'v_out', kvHeadIndex: g },
      {
        format: 'signed',
        activeCol,
        activeRow,
        activeValueOverride,
        rowLabels: queryTokLabels,
        colDimLabel: `16Q×${dHead} = ${cols}`,
        rowDimLabel: `Q = ${T}`,
      },
    );
  };

  const emitTokenChannel = (
    kind: 'input' | 'rmsnorm' | 'residual',
    activeCol?: number,
    activeRow?: number,
  ): void => {
    const src = kind === 'input' ? sim.X : kind === 'rmsnorm' ? sim.U : sim.Y;
    const values = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < d; i++) {
        values[t * d + i] = src[i * T + t];
      }
    }
    pushGrid(
      `${prefix}:${kind}`,
      values,
      { sheetKind: SHEET_KIND_STANDARD, tensorKey: kind },
      {
        format: 'signed',
        activeCol,
        activeRow,
        colLabels: chanLabels,
        rowLabels: queryTokLabels,
        colDimLabel: `d = ${d}`,
        rowDimLabel: `Q = ${T}`,
      },
    );
  };

  const activeOverride = focusedAccumulation?.active ? focusedAccumulation.partialSum : null;
  if (formula) {
    const cell = clampCellToSim(sim, formula.cell);
    const g: 0 | 1 = cell.kvHeadIndex === 1 ? 1 : 0;
    if (cell.tensor === 'scores') {
      emitScoresOrSoftmax('scores', g, cell.col, cell.row, activeOverride);
      emitScoresOrSoftmax('softmax', g, cell.col, cell.row);
    } else if (cell.tensor === 'softmax') {
      emitScoresOrSoftmax('softmax', g, cell.col, cell.row);
      emitScoresOrSoftmax('scores', g, cell.col, cell.row);
    } else if (cell.tensor === 'v_out') {
      emitVOut(g, cell.col, cell.row, activeOverride);
      emitScoresOrSoftmax('softmax', g, undefined, cell.row);
    } else if (cell.tensor === 'q') {
      emitQHead(cell.headIndex ?? displayedHeads[g], cell.col, cell.row);
      emitScoresOrSoftmax('scores', g, undefined, cell.row);
    } else if (cell.tensor === 'input' || cell.tensor === 'rmsnorm' || cell.tensor === 'residual') {
      emitTokenChannel(cell.tensor, cell.col, cell.row);
    } else {
      emitScoresOrSoftmax('scores', 0);
      emitScoresOrSoftmax('softmax', 0);
    }
  } else {
    emitScoresOrSoftmax('scores', 0);
    emitScoresOrSoftmax('softmax', 0);
  }

  return grids;
}

// ============================================================================
// MAIN PAIRED 35/65 GQA SCENE BUILDER
// ============================================================================

export function buildTwoTowerPairedGqaScene(
  session: TwoTowerSession,
  options?: BuildPairedGqaOptions,
): TwoTowerSpatialScene {
  const view = resolveTwoTowerSessionViewState(session);
  const bayId = resolveNearestBayOfKind(options?.bayId ?? 6, '*');
  const layerIndex = bayId - 1;
  const towerFocus: TwoTowerPairedFocus = options?.towerFocus ?? 'paired';

  const group0Head = Math.max(0, Math.min(15, Math.round(options?.group0Head ?? 0)));
  const group1Head = Math.max(16, Math.min(31, Math.round(options?.group1Head ?? 16)));

  const ctxLookup = view.resolveContextBayTrace(bayId);
  const contextSim: GqaSimResult | null =
    ctxLookup.status === 'available' && ctxLookup.trace?.kind === '*' ? ctxLookup.trace.sim : null;
  const ctxKvCache: GqaLayerKvCache | null =
    view.contextSnapshot.ctxState.gqaCaches.get(bayId) ?? null;

  const denoiserTrace = view.resolveDenoiserBayTrace(bayId);
  const denoiserSim: GqaSimResult | null =
    denoiserTrace?.kind === '*' ? denoiserTrace.sim : null;

  // Build dedicated 128×128 TwoTower GQA value atlas without ever inventing Nano fixtures
  let gqaAtlas: TwoTowerGqaValueAtlas | null = null;
  let fallbackCacheAtlas: Rg32fValueAtlas | null = null;
  let primaryRole: 'denoiser' | 'context' | 'cache_only' = 'cache_only';

  if (denoiserSim !== null) {
    primaryRole = 'denoiser';
    gqaAtlas = buildTwoTowerGqaValueAtlas({
      primarySim: denoiserSim,
      secondarySim: contextSim,
      secondaryKvCacheFallback: contextSim === null ? ctxKvCache : null,
      group0Head,
      group1Head,
    });
  } else if (contextSim !== null) {
    primaryRole = 'context';
    gqaAtlas = buildTwoTowerGqaValueAtlas({
      primarySim: contextSim,
      secondarySim: null,
      group0Head,
      group1Head,
    });
  } else if (ctxKvCache !== null) {
    // Pruned Context-only state: pack only the persistent KV cache without running any fake sim
    const writer = createRg32fAtlas(TWOTOWER_GQA_ATLAS_WIDTH, TWOTOWER_GQA_ATLAS_HEIGHT);
    const P = Math.min(24, ctxKvCache.seqLen);
    for (let g = 0; g < 2; g++) {
      const kBaseX = g === 0 ? 96 : 104;
      const vBaseX = g === 0 ? 100 : 108;
      for (let k = 0; k < P; k++) {
        for (let r = 0; r < 4; r++) {
          const row = g * 4 + r;
          writer.setTexel(kBaseX + r, 88 + k, ctxKvCache.K[row * ctxKvCache.maxTokens + k]);
          writer.setTexel(vBaseX + r, 88 + k, ctxKvCache.V[row * ctxKvCache.maxTokens + k]);
        }
      }
    }
    fallbackCacheAtlas = {
      width: TWOTOWER_GQA_ATLAS_WIDTH,
      height: TWOTOWER_GQA_ATLAS_HEIGHT,
      data: writer.data,
    };
  }

  const dl = view.denoiserLookup;
  const isHistoricalDenoiser = dl.isHistoricalInspection;
  const matchesCurrentBoundary = dl.matchesCurrentContextBoundary;
  const historicalBoundaryMismatch = isHistoricalDenoiser && !matchesCurrentBoundary;
  const currentContextPrefixLen = view.committedPrefixLen;
  const denoiserConditioningPrefixLen = dl.sourceContextPrefixLen;

  // Explicit 4-length distinction (NEVER relabeling a prior Denoiser pass's P_cond as current Context P!)
  const allocatedKvCapacity = view.kvCapacity;
  const queryLen =
    denoiserSim !== null
      ? denoiserSim.tokens
      : contextSim !== null
        ? contextSim.tokens
        : 0;
  const contextPrefixKvLen =
    denoiserSim !== null
      ? denoiserSim.prefixTokens
      : currentContextPrefixLen;
  const blockKvLen = denoiserSim !== null ? denoiserSim.tokens : 0;
  const totalAttendedKvLen =
    denoiserSim !== null
      ? denoiserSim.keyTokens
      : contextSim !== null
        ? contextSim.keyTokens
        : currentContextPrefixLen;
  const unusedKvCapacity = Math.max(0, allocatedKvCapacity - totalAttendedKvLen);
  const historicalTag =
    isHistoricalDenoiser && dl.pass !== null
      ? `Historical Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, ${dl.pass.passExecutionId}, P_cond=${denoiserConditioningPrefixLen ?? contextPrefixKvLen})`
      : null;

  // Resolve active inspection target & stepper state
  const stepperStep =
    options?.stepperStep !== undefined && options.stepperStep !== null
      ? Math.max(0, Math.min(5, Math.round(options.stepperStep)))
      : null;
  const stepProgress = options?.stepProgress ?? 0.25;
  const termProgress = options?.termProgress;
  const reducedMotion = options?.reducedMotion ?? false;
  const revealEnabled = stepperStep !== null && !reducedMotion;

  const requestedCell = options?.hoveredCell ?? options?.selectedCell ?? null;
  let activeTower: 'context' | 'denoiser' =
    requestedCell?.tower ??
    (towerFocus === 'context_solo'
      ? 'context'
      : denoiserSim !== null
        ? 'denoiser'
        : 'context');
  if (activeTower === 'denoiser' && denoiserSim === null && contextSim !== null) {
    activeTower = 'context';
  }
  if (activeTower === 'context' && contextSim === null && denoiserSim !== null) {
    activeTower = 'denoiser';
  }

  const activeSim: GqaSimResult | null =
    activeTower === 'denoiser' ? denoiserSim : contextSim;

  let activeCell: GqaCellRef | null = null;
  if (requestedCell && requestedCell.tower === activeTower && activeSim !== null) {
    activeCell = clampCellToSim(activeSim, requestedCell.cell);
  } else if (stepperStep !== null && activeSim !== null) {
    activeCell = twoTowerGqaStepperCell(activeSim, group0Head, stepperStep, stepProgress);
  }

  // Keep group0Head / group1Head aligned with inspected cell head if applicable
  const inferredHead = inferGqaCellHead(activeCell);
  const effG0 = inferredHead !== null && inferredHead < 16 ? inferredHead : group0Head;
  const effG1 = inferredHead !== null && inferredHead >= 16 ? inferredHead : group1Head;
  const displayedHeads: readonly [number, number] = [effG0, effG1];

  // Rebuild atlas only if activeCell shifted the displayed group head away from options
  if (gqaAtlas !== null && (effG0 !== gqaAtlas.group0Head || effG1 !== gqaAtlas.group1Head)) {
    gqaAtlas =
      primaryRole === 'denoiser' && denoiserSim !== null
        ? buildTwoTowerGqaValueAtlas({
            primarySim: denoiserSim,
            secondarySim: contextSim,
            secondaryKvCacheFallback: contextSim === null ? ctxKvCache : null,
            group0Head: effG0,
            group1Head: effG1,
          })
        : contextSim !== null
          ? buildTwoTowerGqaValueAtlas({
              primarySim: contextSim,
              secondarySim: null,
              group0Head: effG0,
              group1Head: effG1,
            })
          : gqaAtlas;
  }

  const rawFormula =
    activeSim !== null && activeCell !== null ? describeCellFormula(activeSim, activeCell) : null;
  const activeFormula =
    rawFormula !== null && activeSim !== null
      ? augmentGqaFormulaWithProvenance(
          activeSim,
          rawFormula,
          activeTower,
          bayId,
          historicalBoundaryMismatch && activeTower === 'denoiser'
            ? `Historical P_cond=${contextPrefixKvLen} ≠ current Context P=${currentContextPrefixLen}, Rec #${dl.sourceRecordIndex}`
            : null,
        )
      : null;

  const focusedAccumulation: FocusedAccumulationResult | null =
    activeSim !== null &&
    activeCell !== null &&
    stepperStep !== null &&
    !reducedMotion &&
    termProgress !== undefined
      ? computeGqaFocusedAccumulation(activeSim, activeCell, stepperStep, termProgress)
      : null;

  const focusSet: SheetFocusSet = buildGqaFocusSet(activeFormula);
  const activeFocusGrp: 0 | 1 =
    activeCell?.kvHeadIndex === 1 || (inferredHead !== null && inferredHead >= 16) ? 1 : 0;
  const activeFocusHead = inferredHead ?? displayedHeads[activeFocusGrp];

  const denoiserRevealCtx: GqaRevealContext | null =
    revealEnabled && activeTower === 'denoiser' && denoiserSim !== null
      ? {
          enabled: true,
          step: stepperStep!,
          sim: denoiserSim,
          activeCell,
          activeHead: activeFocusHead,
          activeGroup: activeFocusGrp,
          pulsePhase: ((stepProgress * 4) % 1 + 1) % 1,
        }
      : null;

  const contextRevealCtx: GqaRevealContext | null =
    revealEnabled && activeTower === 'context' && contextSim !== null
      ? {
          enabled: true,
          step: stepperStep!,
          sim: contextSim,
          activeCell,
          activeHead: activeFocusHead,
          activeGroup: activeFocusGrp,
          pulsePhase: ((stepProgress * 4) % 1 + 1) % 1,
        }
      : null;

  const b = new PairedGqaBuilder(bayId, revealEnabled, focusSet.active);
  const SHEET_THICK = 0.056;
  const BASE_Y = 0.04;
  const DECK_Y = 0.30;

  const ctxExecId =
    ctxLookup.segment?.stepExecutionId ?? view.contextSnapshot.ctxState.stateDigest;
  const denExecId =
    view.denoiserLookup.pass?.passExecutionId ?? view.identity.recordId;

  const showContextWing = towerFocus !== 'denoiser_solo';
  const showDenoiserWing = towerFocus !== 'context_solo';

  // Coordinate mapping helper: maps normalized [0..1] wing coordinate into [xMin, xMax]
  const ctxXMin = towerFocus === 'context_solo' ? -8.8 : -13.5;
  const ctxXMax = towerFocus === 'context_solo' ? 8.8 : -3.5;
  const ctxX = (u: number): number => ctxXMin + u * (ctxXMax - ctxXMin);
  const ctxSpan = ctxXMax - ctxXMin;

  const denXMin = towerFocus === 'denoiser_solo' ? -9.2 : -1.8;
  const denXMax = towerFocus === 'denoiser_solo' ? 9.2 : 13.5;
  const denX = (u: number): number => denXMin + u * (denXMax - denXMin);
  const denSpan = denXMax - denXMin;

  const zGrp: readonly [number, number] = [-1.95, 1.95];
  const groupSpanZ = 2.60;
  const kvPitchZ = groupSpanZ / Math.max(1, allocatedKvCapacity);
  const kvSliceDepth = Math.min(0.095, kvPitchZ * 0.78);
  const kvSliceZ = (g: 0 | 1, kZeroBased: number): number =>
    zGrp[g] - groupSpanZ * 0.5 + (kZeroBased + 0.5) * kvPitchZ;

  // Track bridge anchors
  const ctxBridgeOutK: [number, number, number][] = [];
  const ctxBridgeOutV: [number, number, number][] = [];
  const denBridgeInK: [number, number, number][] = [];
  const denBridgeInV: [number, number, number][] = [];

  // ==========================================================================
  // 1. LEFT WING: CONTEXT TOWER PROVENANCE (`x ∈ [-13.5, -3.5]` in paired mode)
  // ==========================================================================
  if (showContextWing) {
    const ctxSlicesPrimary: TwoTowerGqaPrimaryAtlasSlices | null =
      primaryRole === 'context' && gqaAtlas ? gqaAtlas.primary : null;
    const ctxSlicesSec: TwoTowerGqaSecondaryAtlasSlices | null =
      primaryRole === 'denoiser' && gqaAtlas ? gqaAtlas.secondary : null;
    const ctxFocus = activeTower === 'context' ? focusSet : EMPTY_SHEET_FOCUS_SET;
    const ctxAbsOffset = ctxLookup.segment?.tokenRange[0] ?? 0;

    // Wing boundary frame on drafting table
    b.addBoxFrameXZ(ctxXMin, -3.65, ctxXMax, 3.65, BASE_Y - 0.03, 1.0, LINE_STYLE_GUIDE, 0.42);
    b.landmarks.push({
      id: 'ctx-wing-header',
      label: historicalBoundaryMismatch
        ? `Context Tower Provenance · Current P=${currentContextPrefixLen} (≠ Prior Denoiser P_cond=${contextPrefixKvLen})`
        : 'Context Tower Provenance (Causal)',
      sublabel:
        ctxLookup.status === 'available'
          ? `origin=${ctxLookup.traceOrigin} · KV[0..${Math.max(0, currentContextPrefixLen - 1)}] (${currentContextPrefixLen}/${allocatedKvCapacity} slots) · Model: 32Q/2KV dHead=4 (2 score slices: ${fmtHead(effG0)}, ${fmtHead(effG1)})`
          : `Trace pruned · Persistent KV cache [0..${Math.max(0, currentContextPrefixLen - 1)}] authoritative`,
      world: [ctxX(0.5), 2.85, -3.65],
      bayId,
      partId: 'ctx-kv-cache',
      tone: historicalBoundaryMismatch ? 'gold' : ctxLookup.status === 'available' ? 'cyan' : 'gold',
      align: 'center',
      tier: 'primary',
    });

    if (contextSim !== null) {
      const cT = contextSim.tokens;
      const cK = contextSim.keyTokens;
      const cP = contextSim.prefixTokens;
      const cScaleX = ctxSpan / 10.0;

      const xIn = ctxX(0.08);
      const xRms = ctxX(0.22);
      const xQ = ctxX(0.36);
      const xScores = ctxX(0.54);
      const xSoft = ctxX(0.70);
      const xKv = ctxX(0.86);
      const xOut = ctxX(0.95);

      const getCtxSlice = (
        key: 'input' | 'rmsnorm' | 'residual' | 'wK' | 'wV',
      ): AtlasSlice | undefined => {
        if (ctxSlicesPrimary) return ctxSlicesPrimary[key];
        return ctxSlicesSec?.[key] ?? undefined;
      };

      const hlCtx = (tensor: string, fallback: number): number =>
        resolveSheetHighlight(ctxFocus, { tensor }, fallback);

      // Input X, RMSNorm U, Residual Y
      b.addSheet(
        {
          cx: xIn, cy: BASE_Y, cz: 0,
          ex: 0.72 * cScaleX, ey: SHEET_THICK, ez: 1.45,
          axes: FRAME_XZ_AXES,
          gridCols: contextSim.d, gridRows: cT,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_RESIDUAL, fp8Flag: 0,
          highlight: hlCtx('input', SHEET_HL_NORMAL),
          seed: 11.1, partId: 'ctx-input',
          towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
          semanticTag: 'context_input', absoluteTokenOffset: ctxAbsOffset,
          tensorKey: 'input', atlasSlice: getCtxSlice('input'), registryKey: 'ctx:input',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        contextRevealCtx,
      );

      b.addSheet(
        {
          cx: xRms, cy: BASE_Y, cz: 0,
          ex: 0.72 * cScaleX, ey: SHEET_THICK, ez: 1.45,
          axes: FRAME_XZ_AXES,
          gridCols: contextSim.d, gridRows: cT,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_NORM, fp8Flag: 0,
          highlight: hlCtx('rmsnorm', SHEET_HL_NORMAL),
          seed: 12.1, partId: 'ctx-rmsnorm',
          towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
          semanticTag: 'context_rmsnorm', absoluteTokenOffset: ctxAbsOffset,
          tensorKey: 'rmsnorm', atlasSlice: getCtxSlice('rmsnorm'), registryKey: 'ctx:rmsnorm',
          cellFamily: CELL_MOE_AMBER,
        },
        contextRevealCtx,
      );

      // Active Q heads for Group 0 & Group 1 + Scores/Softmax (strictly causal) + Persistent KV racks
      for (let g = 0 as 0 | 1; g < 2; g = (g + 1) as 0 | 1) {
        const h = displayedHeads[g];
        const qSlice = ctxSlicesPrimary
          ? ctxSlicesPrimary.qHeads[h]
          : ctxSlicesSec?.activeQ?.[g];
        const scSlice = ctxSlicesPrimary
          ? ctxSlicesPrimary.scores[g]
          : ctxSlicesSec?.scores?.[g];
        const smSlice = ctxSlicesPrimary
          ? ctxSlicesPrimary.softmax[g]
          : ctxSlicesSec?.softmax?.[g];

        b.addSheet(
          {
            cx: xQ, cy: DECK_Y + 0.35, cz: zGrp[g],
            ex: 0.56 * cScaleX, ey: SHEET_THICK, ez: 0.92,
            axes: FRAME_XZ_AXES,
            gridCols: contextSim.dHead, gridRows: cT,
            sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
            highlight: resolveSheetHighlight(
              ctxFocus,
              { tensor: 'q', headIndex: h, kvHeadIndex: g, isAggregate: false },
              SHEET_HL_NORMAL,
            ),
            seed: 13.0 + g, partId: `ctx-q-${h}`,
            towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
            semanticTag: 'context_query', absoluteTokenOffset: ctxAbsOffset,
            headIndex: h, kvHeadIndex: g, tensorKey: 'q',
            atlasSlice: qSlice, registryKey: `ctx:q:${h}`,
            cellFamily: CELL_WEIGHT_BLUE,
          },
          contextRevealCtx,
        );

        const scW = Math.min(1.18 * cScaleX, 0.62 * cScaleX + (cK / allocatedKvCapacity) * 0.56 * cScaleX);
        const scD = Math.max(0.32, Math.min(1.15, (cT / Math.max(1, cK)) * scW));

        b.addSheet(
          {
            cx: xScores, cy: DECK_Y, cz: zGrp[g],
            ex: scW, ey: SHEET_THICK, ez: scD,
            axes: FRAME_XZ_AXES,
            gridCols: cK, gridRows: cT,
            sheetKind: SHEET_KIND_CAUSAL_SCORES, role: SHEET_ROLE_SCORES, fp8Flag: 0,
            highlight: resolveSheetHighlight(
              ctxFocus,
              { tensor: 'scores', headIndex: h, kvHeadIndex: g, isAggregate: false },
              SHEET_HL_NORMAL,
            ),
            seed: 14.0 + g, partId: `ctx-scores-g${g}`,
            towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
            semanticTag: 'context_causal_scores', absoluteTokenOffset: ctxAbsOffset,
            headIndex: h, kvHeadIndex: g, tensorKey: 'scores',
            atlasSlice: scSlice, registryKey: `ctx:scores:${g}`,
            cellFamily: CELL_WEIGHT_BLUE,
          },
          contextRevealCtx,
        );

        b.addSheet(
          {
            cx: xSoft, cy: DECK_Y, cz: zGrp[g],
            ex: scW, ey: SHEET_THICK, ez: scD,
            axes: FRAME_XZ_AXES,
            gridCols: cK, gridRows: cT,
            sheetKind: SHEET_KIND_CAUSAL_SOFTMAX, role: SHEET_ROLE_SOFTMAX, fp8Flag: 0,
            highlight: resolveSheetHighlight(
              ctxFocus,
              { tensor: 'softmax', headIndex: h, kvHeadIndex: g, isAggregate: false },
              SHEET_HL_NORMAL,
            ),
            seed: 15.0 + g, partId: `ctx-softmax-g${g}`,
            towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
            semanticTag: 'context_causal_softmax', absoluteTokenOffset: ctxAbsOffset,
            headIndex: h, kvHeadIndex: g, tensorKey: 'softmax',
            atlasSlice: smSlice, registryKey: `ctx:softmax:${g}`,
            cellFamily: CELL_ACTIVATION_GREEN,
          },
          contextRevealCtx,
        );

        // Context Persistent KV Cache rack (K & V side by side + ghost capacity rail up to 24)
        const kvRailW = 0.42 * cScaleX;
        const xKvK = xKv - kvRailW * 0.56;
        const xKvV = xKv + kvRailW * 0.56;

        b.addSheet(
          {
            cx: xKv, cy: BASE_Y - 0.02, cz: zGrp[g],
            ex: kvRailW * 2.25, ey: 0.022, ez: groupSpanZ + 0.06,
            axes: FRAME_XZ_AXES,
            gridCols: 2, gridRows: allocatedKvCapacity,
            sheetKind: SHEET_KIND_GHOST_RAIL, role: SHEET_ROLE_NORM, fp8Flag: 0,
            highlight: SHEET_HL_NORMAL,
            seed: 16.0 + g, partId: `ctx-kv-rail-g${g}`,
            towerRole: 'context', executionId: ctxExecId, traceOrigin: 'persistent_cache',
            semanticTag: 'context_kv_capacity_rail',
            cellFamily: CELL_NEUTRAL,
          },
          null,
        );

        for (let k = 0; k < cK; k++) {
          const kz = kvSliceZ(g, k);
          const kSlice = ctxSlicesPrimary ? ctxSlicesPrimary.kvK[g][k] : ctxSlicesSec?.kvK[g][k];
          const vSlice = ctxSlicesPrimary ? ctxSlicesPrimary.kvV[g][k] : ctxSlicesSec?.kvV[g][k];
          const sliceOrigin: TwoTowerTraceOrigin = k < cP ? 'persistent_cache' : ctxLookup.traceOrigin;

          b.addSheet(
            {
              cx: xKvK, cy: BASE_Y + 0.04, cz: kz,
              ex: kvRailW, ey: SHEET_THICK, ez: kvSliceDepth,
              axes: FRAME_XZ_AXES,
              gridCols: contextSim.dHead, gridRows: 1,
              sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_K, fp8Flag: 0,
              highlight: resolveSheetHighlight(
                ctxFocus,
                { tensor: 'kv_k', kvHeadIndex: g, tokenIndex: k, isAggregate: false },
                SHEET_HL_NORMAL,
              ),
              seed: 17.0 + g * 30 + k, partId: 'ctx-kv-cache',
              towerRole: 'context', executionId: ctxExecId, traceOrigin: sliceOrigin,
              semanticTag: 'context_persistent_kv_k', absoluteTokenOffset: 0,
              kvHeadIndex: g, tokenIndex: k, tensorKey: 'kv_k',
              atlasSlice: kSlice, registryKey: `ctx:kv_k:${g}:${k}`,
              cellFamily: CELL_SSM_CYAN,
            },
            contextRevealCtx,
          );

          b.addSheet(
            {
              cx: xKvV, cy: BASE_Y + 0.04, cz: kz,
              ex: kvRailW, ey: SHEET_THICK, ez: kvSliceDepth,
              axes: FRAME_XZ_AXES,
              gridCols: contextSim.dHead, gridRows: 1,
              sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_V, fp8Flag: 0,
              highlight: resolveSheetHighlight(
                ctxFocus,
                { tensor: 'kv_v', kvHeadIndex: g, tokenIndex: k, isAggregate: false },
                SHEET_HL_NORMAL,
              ),
              seed: 18.0 + g * 30 + k, partId: 'ctx-kv-cache',
              towerRole: 'context', executionId: ctxExecId, traceOrigin: sliceOrigin,
              semanticTag: 'context_persistent_kv_v', absoluteTokenOffset: 0,
              kvHeadIndex: g, tokenIndex: k, tensorKey: 'kv_v',
              atlasSlice: vSlice, registryKey: `ctx:kv_v:${g}:${k}`,
              cellFamily: CELL_SSM_CYAN,
            },
            contextRevealCtx,
          );
        }

        const midPrefixIdx = Math.max(0, Math.min(cK - 1, Math.floor(Math.max(0, contextPrefixKvLen - 1) * 0.5)));
        ctxBridgeOutK[g] = [xKvV + kvRailW * 0.55, BASE_Y + 0.14, kvSliceZ(g, midPrefixIdx)];
        ctxBridgeOutV[g] = [xKvV + kvRailW * 0.55, BASE_Y + 0.28, kvSliceZ(g, Math.min(cK - 1, midPrefixIdx + 1))];

        // Subtle internal context flow
        b.flow.addPolyline3D(
          [
            [xRms + 0.36 * cScaleX, BASE_Y + 0.06, 0],
            [xQ - 0.30 * cScaleX, DECK_Y + 0.35, zGrp[g]],
          ],
          [0, 1, 0], 0.06, RIBBON_CONTEXT_QUIET, true, 0.14, 0.14, false,
        );
        b.flow.addPolyline3D(
          [
            [xScores + scW * 0.52, DECK_Y + 0.04, zGrp[g]],
            [xSoft - scW * 0.52, DECK_Y + 0.04, zGrp[g]],
          ],
          [0, 1, 0], 0.06, RIBBON_CONTEXT_QUIET, true, 0.14, 0.14, false,
        );
      }

      // Context Residual Output Y
      b.addSheet(
        {
          cx: xOut, cy: BASE_Y, cz: 0,
          ex: 0.68 * cScaleX, ey: SHEET_THICK, ez: 1.45,
          axes: FRAME_XZ_AXES,
          gridCols: contextSim.d, gridRows: cT,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_RESIDUAL, fp8Flag: 0,
          highlight: hlCtx('residual', SHEET_HL_NORMAL),
          seed: 19.1, partId: 'ctx-residual',
          towerRole: 'context', executionId: ctxExecId, traceOrigin: ctxLookup.traceOrigin,
          semanticTag: 'context_residual', absoluteTokenOffset: ctxAbsOffset,
          tensorKey: 'residual', atlasSlice: getCtxSlice('residual'), registryKey: 'ctx:residual',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        contextRevealCtx,
      );
    } else {
      // PRUNED CONTEXT TRACE FAIL-SAFE: render ONLY persistent KV cache + explicit notice plaque
      const cP = ctxKvCache?.seqLen ?? view.committedPrefixLen;
      const cScaleX = ctxSpan / 10.0;
      const xNotice = ctxX(0.32);
      const xKv = ctxX(0.78);
      const kvRailW = 0.52 * cScaleX;

      b.addSheet(
        {
          cx: xNotice, cy: BASE_Y + 0.16, cz: 0,
          ex: 3.4 * cScaleX, ey: SHEET_THICK, ez: 2.2,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 2,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_NEUTRAL, fp8Flag: 0,
          highlight: SHEET_HL_BF16_GOLD,
          seed: 10.5, partId: 'ctx-pruned-notice',
          towerRole: 'context', executionId: ctxExecId, traceOrigin: 'persistent_cache',
          semanticTag: 'context_trace_pruned',
          cellFamily: CELL_NEUTRAL,
        },
        null,
      );

      for (let g = 0 as 0 | 1; g < 2; g = (g + 1) as 0 | 1) {
        const xKvK = xKv - kvRailW * 0.56;
        const xKvV = xKv + kvRailW * 0.56;
        b.addSheet(
          {
            cx: xKv, cy: BASE_Y - 0.02, cz: zGrp[g],
            ex: kvRailW * 2.25, ey: 0.022, ez: groupSpanZ + 0.06,
            axes: FRAME_XZ_AXES,
            gridCols: 2, gridRows: allocatedKvCapacity,
            sheetKind: SHEET_KIND_GHOST_RAIL, role: SHEET_ROLE_NORM, fp8Flag: 0,
            highlight: SHEET_HL_NORMAL,
            seed: 16.0 + g, partId: `ctx-kv-rail-g${g}`,
            towerRole: 'context', executionId: ctxExecId, traceOrigin: 'persistent_cache',
            semanticTag: 'context_kv_capacity_rail',
            cellFamily: CELL_NEUTRAL,
          },
          null,
        );

        for (let k = 0; k < cP; k++) {
          const kz = kvSliceZ(g, k);
          const kSlice = ctxSlicesSec?.kvK[g][k];
          const vSlice = ctxSlicesSec?.kvV[g][k];
          b.addSheet(
            {
              cx: xKvK, cy: BASE_Y + 0.04, cz: kz,
              ex: kvRailW, ey: SHEET_THICK, ez: kvSliceDepth,
              axes: FRAME_XZ_AXES,
              gridCols: 4, gridRows: 1,
              sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_K, fp8Flag: 0,
              highlight: SHEET_HL_NORMAL,
              seed: 17.0 + g * 30 + k, partId: 'ctx-kv-cache',
              towerRole: 'context', executionId: ctxExecId, traceOrigin: 'persistent_cache',
              semanticTag: 'context_persistent_kv_k', absoluteTokenOffset: 0,
              kvHeadIndex: g, tokenIndex: k, tensorKey: 'kv_k',
              atlasSlice: kSlice, registryKey: `ctx:kv_k:${g}:${k}`,
              cellFamily: CELL_SSM_CYAN,
            },
            null,
          );
          b.addSheet(
            {
              cx: xKvV, cy: BASE_Y + 0.04, cz: kz,
              ex: kvRailW, ey: SHEET_THICK, ez: kvSliceDepth,
              axes: FRAME_XZ_AXES,
              gridCols: 4, gridRows: 1,
              sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_V, fp8Flag: 0,
              highlight: SHEET_HL_NORMAL,
              seed: 18.0 + g * 30 + k, partId: 'ctx-kv-cache',
              towerRole: 'context', executionId: ctxExecId, traceOrigin: 'persistent_cache',
              semanticTag: 'context_persistent_kv_v', absoluteTokenOffset: 0,
              kvHeadIndex: g, tokenIndex: k, tensorKey: 'kv_v',
              atlasSlice: vSlice, registryKey: `ctx:kv_v:${g}:${k}`,
              cellFamily: CELL_SSM_CYAN,
            },
            null,
          );
        }

        const midIdx = Math.max(0, Math.min(Math.max(0, cP - 1), Math.floor(Math.max(0, contextPrefixKvLen - 1) * 0.5)));
        ctxBridgeOutK[g] = [xKvV + kvRailW * 0.55, BASE_Y + 0.14, kvSliceZ(g, midIdx)];
        ctxBridgeOutV[g] = [xKvV + kvRailW * 0.55, BASE_Y + 0.28, kvSliceZ(g, midIdx)];
      }
    }
  }

  // ==========================================================================
  // 2. RIGHT WING: ACTIVE DENOISER TOWER COMPUTATION (`x ∈ [-1.8, +13.5]`)
  // ==========================================================================
  if (showDenoiserWing) {
    const denFocus = activeTower === 'denoiser' ? focusSet : EMPTY_SHEET_FOCUS_SET;
    const denSlices: TwoTowerGqaPrimaryAtlasSlices | null =
      primaryRole === 'denoiser' && gqaAtlas ? gqaAtlas.primary : null;

    b.addBoxFrameXZ(denXMin, -3.75, denXMax, 3.75, BASE_Y - 0.03, 1.0, LINE_STYLE_GUIDE, 0.48);

    if (denoiserSim !== null && denSlices !== null) {
      const T = denoiserSim.tokens;
      const P = denoiserSim.prefixTokens;
      const K = denoiserSim.keyTokens;
      const isBidirectional = denoiserSim.attentionMaskMode === 'bidirectional_block';
      const dScaleX = denSpan / 15.3;

      const xIn = denX(0.04);
      const xMod = denX(0.11);
      const xRms = denX(0.18);
      const xWIn = denX(0.27);
      const xQBase = denX(0.36);
      const xKTrack = denX(0.47);
      const xScores = denX(0.57);
      const xSoft = denX(0.68);
      const xVTrack = denX(0.77);
      const xVOut = denX(0.84);
      const xConcat = denX(0.895);
      const xWO = denX(0.935);
      const xOut = denX(0.975);

      const hlDen = (tensor: string, fallback: number): number =>
        resolveSheetHighlight(denFocus, { tensor }, fallback);

      // 2a. Input X -> optional X_mod (Pre-RMSNorm AdaLN) -> RMSNorm U
      const ioW = 0.74 * dScaleX;
      const ioD = 1.48;
      b.addSheet(
        {
          cx: xIn, cy: BASE_Y, cz: 0,
          ex: ioW, ey: SHEET_THICK, ez: ioD,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.d, gridRows: T,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_RESIDUAL, fp8Flag: 0,
          highlight: hlDen('input', SHEET_HL_NORMAL),
          seed: 21.1, partId: 'den-input',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_input_x', absoluteTokenOffset: P,
          tensorKey: 'input', atlasSlice: denSlices.input, registryKey: 'den:input',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealCtx,
      );

      if (denoiserSim.X_mod && denSlices.modulatedInput) {
        b.addSheet(
          {
            cx: xMod, cy: BASE_Y + 0.08, cz: 0,
            ex: ioW * 0.92, ey: SHEET_THICK, ez: ioD,
            axes: FRAME_XZ_AXES,
            gridCols: denoiserSim.d, gridRows: T,
            sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_NORM, fp8Flag: 0,
            highlight: SHEET_HL_BF16_GOLD,
            seed: 21.6, partId: 'den-adaln-pre-norm',
            towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
            semanticTag: 'denoiser_pre_norm_adaln', absoluteTokenOffset: P,
            atlasSlice: denSlices.modulatedInput, registryKey: 'den:x_mod',
            cellFamily: CELL_MOE_AMBER,
          },
          denoiserRevealCtx,
        );
      }

      b.addSheet(
        {
          cx: xRms, cy: BASE_Y, cz: 0,
          ex: ioW, ey: SHEET_THICK, ez: ioD,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.d, gridRows: T,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_NORM, fp8Flag: 0,
          highlight: hlDen('rmsnorm', SHEET_HL_NORMAL),
          seed: 22.1, partId: 'den-rmsnorm',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_rmsnorm_u', absoluteTokenOffset: P,
          tensorKey: 'rmsnorm', atlasSlice: denSlices.rmsnorm, registryKey: 'den:rmsnorm',
          cellFamily: CELL_MOE_AMBER,
        },
        denoiserRevealCtx,
      );

      // 2b. Independent Denoiser Weight Walls (W_Q, W_K, W_V)
      b.addSheet(
        {
          cx: xWIn, cy: 1.98, cz: 0,
          ex: 5.10, ey: SHEET_THICK, ez: 0.90,
          axes: FRAME_YZ_AXES,
          gridCols: denoiserSim.qConcatDim, gridRows: denoiserSim.d,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_WEIGHT, fp8Flag: 0,
          highlight: hlDen('w_q', SHEET_HL_NORMAL),
          seed: 23.1, partId: 'den-w-q',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_weight_wq',
          tensorKey: 'w_q', atlasSlice: denSlices.wQ, registryKey: 'den:w_q',
          cellFamily: CELL_WEIGHT_BLUE,
        },
        denoiserRevealCtx,
      );

      b.addSheet(
        {
          cx: xWIn, cy: 0.72, cz: -1.15,
          ex: 0.95, ey: SHEET_THICK, ez: 0.64,
          axes: FRAME_YZ_AXES,
          gridCols: 8, gridRows: denoiserSim.d,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_K, fp8Flag: 0,
          highlight: hlDen('w_k', SHEET_HL_NORMAL),
          seed: 23.2, partId: 'den-w-k',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_weight_wk',
          tensorKey: 'w_k', atlasSlice: denSlices.wK, registryKey: 'den:w_k',
          cellFamily: CELL_SSM_CYAN,
        },
        denoiserRevealCtx,
      );

      b.addSheet(
        {
          cx: xWIn, cy: 0.72, cz: 1.15,
          ex: 0.95, ey: SHEET_THICK, ez: 0.64,
          axes: FRAME_YZ_AXES,
          gridCols: 8, gridRows: denoiserSim.d,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_V, fp8Flag: 0,
          highlight: hlDen('w_v', SHEET_HL_NORMAL),
          seed: 23.3, partId: 'den-w-v',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_weight_wv',
          tensorKey: 'w_v', atlasSlice: denSlices.wV, registryKey: 'den:w_v',
          cellFamily: CELL_SSM_CYAN,
        },
        denoiserRevealCtx,
      );

      // 2c. All 32 Denoiser Q Heads (2 groups × 16 heads)
      const qPitchZ = groupSpanZ / 15;
      const qStepX = 0.032 * dScaleX;
      for (let h = 0; h < 32; h++) {
        const g: 0 | 1 = h < 16 ? 0 : 1;
        const localIdx = h % 16;
        const isLifted = h === displayedHeads[g];
        const qx = xQBase + localIdx * qStepX;
        const qz = zGrp[g] - groupSpanZ * 0.5 + localIdx * qPitchZ;
        const qy = isLifted ? 1.56 : 0.94;
        const qHl = resolveSheetHighlight(
          denFocus,
          { tensor: 'q', headIndex: h, kvHeadIndex: g, isAggregate: false },
          isLifted ? SHEET_HL_PATHWAY : SHEET_HL_NORMAL,
        );

        b.addSheet(
          {
            cx: qx, cy: qy, cz: qz,
            ex: 0.54 * dScaleX, ey: SHEET_THICK * 0.85, ez: 0.135,
            axes: FRAME_XZ_AXES,
            gridCols: denoiserSim.dHead, gridRows: T,
            sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
            highlight: qHl,
            seed: 30 + h * 0.2, partId: `den-q-${h}`,
            towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
            semanticTag: 'denoiser_query_head', absoluteTokenOffset: P,
            headIndex: h, kvHeadIndex: g, tensorKey: 'q',
            atlasSlice: denSlices.qHeads[h], registryKey: `den:q:${h}`,
            cellFamily: CELL_WEIGHT_BLUE,
          },
          denoiserRevealCtx,
        );
      }

      // 2d. Concatenated KV Bank (0..P-1 Inherited Prefix + P..P+B-1 Newly Projected Block + P+B..23 Ghost Rail)
      //     + Scores[g0/g1] & Softmax[g0/g1] + V_out[g0/g1]
      const kvSliceW = 0.58 * dScaleX;
      const scoreW = Math.min(1.42 * dScaleX, (0.76 + (K / allocatedKvCapacity) * 0.66) * dScaleX);
      const scoreD = Math.max(0.36, Math.min(1.25, (T / Math.max(1, K)) * scoreW * 1.15));

      for (let g = 0 as 0 | 1; g < 2; g = (g + 1) as 0 | 1) {
        const h = displayedHeads[g];

        // Ghost capacity rails showing full allocatedKvCapacity = 24 slots (P+B..23 visibly empty!)
        b.addSheet(
          {
            cx: xKTrack, cy: BASE_Y - 0.02, cz: zGrp[g],
            ex: kvSliceW + 0.06, ey: 0.022, ez: groupSpanZ + 0.06,
            axes: FRAME_XZ_AXES,
            gridCols: denoiserSim.dHead, gridRows: allocatedKvCapacity,
            sheetKind: SHEET_KIND_GHOST_RAIL, role: SHEET_ROLE_KV_K, fp8Flag: 0,
            highlight: SHEET_HL_NORMAL,
            seed: 40 + g, partId: `den-kv-capacity-k-g${g}`,
            towerRole: 'shared', executionId: denExecId, traceOrigin: 'persistent_cache',
            semanticTag: 'allocated_kv_capacity_rail',
            cellFamily: CELL_NEUTRAL,
          },
          null,
        );
        b.addSheet(
          {
            cx: xVTrack, cy: BASE_Y - 0.02, cz: zGrp[g],
            ex: kvSliceW + 0.06, ey: 0.022, ez: groupSpanZ + 0.06,
            axes: FRAME_XZ_AXES,
            gridCols: denoiserSim.dHead, gridRows: allocatedKvCapacity,
            sheetKind: SHEET_KIND_GHOST_RAIL, role: SHEET_ROLE_KV_V, fp8Flag: 0,
            highlight: SHEET_HL_NORMAL,
            seed: 42 + g, partId: `den-kv-capacity-v-g${g}`,
            towerRole: 'shared', executionId: denExecId, traceOrigin: 'persistent_cache',
            semanticTag: 'allocated_kv_capacity_rail',
            cellFamily: CELL_NEUTRAL,
          },
          null,
        );

        // Separator line between inherited Context prefix [0..P-1] and newly projected Denoiser block [P..P+B-1]
        if (P > 0 && P < K) {
          const splitZ = zGrp[g] - groupSpanZ * 0.5 + P * kvPitchZ;
          b.addGuideLine(
            xKTrack - kvSliceW * 0.65, BASE_Y + 0.08, splitZ,
            xKTrack + kvSliceW * 0.65, BASE_Y + 0.08, splitZ,
            1.4, LINE_STYLE_ACTIVE, 0.85,
          );
          b.addGuideLine(
            xVTrack - kvSliceW * 0.65, BASE_Y + 0.08, splitZ,
            xVTrack + kvSliceW * 0.65, BASE_Y + 0.08, splitZ,
            1.4, LINE_STYLE_ACTIVE, 0.85,
          );
        }

        for (let k = 0; k < K; k++) {
          const kz = kvSliceZ(g, k);
          const isPrefix = k < P;
          const sliceOrigin: TwoTowerTraceOrigin = isPrefix ? 'persistent_cache' : 'denoiser_scratch';
          const semTagK = isPrefix ? 'inherited_context_prefix_kv' : 'denoiser_block_kv';
          const semTagV = isPrefix ? 'inherited_context_prefix_kv' : 'denoiser_block_kv';
          const partIdK = isPrefix ? 'den-prefix-kv' : 'den-block-kv';
          const partIdV = isPrefix ? 'den-prefix-kv' : 'den-block-kv';

          b.addSheet(
            {
              cx: xKTrack, cy: BASE_Y + (isPrefix ? 0.04 : 0.08), cz: kz,
              ex: kvSliceW, ey: SHEET_THICK, ez: kvSliceDepth,
              axes: FRAME_XZ_AXES,
              gridCols: denoiserSim.dHead, gridRows: 1,
              sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_K, fp8Flag: 0,
              highlight: resolveSheetHighlight(
                denFocus,
                { tensor: 'kv_k', kvHeadIndex: g, tokenIndex: k, isAggregate: false },
                isPrefix ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
              ),
              seed: 50 + g * 32 + k, partId: partIdK,
              towerRole: 'denoiser', executionId: denExecId, traceOrigin: sliceOrigin,
              semanticTag: semTagK, absoluteTokenOffset: 0,
              kvHeadIndex: g, tokenIndex: k, tensorKey: 'kv_k',
              atlasSlice: denSlices.kvK[g][k], registryKey: `den:kv_k:${g}:${k}`,
              cellFamily: CELL_SSM_CYAN,
            },
            denoiserRevealCtx,
            isPrefix,
            !isPrefix,
          );

          b.addSheet(
            {
              cx: xVTrack, cy: BASE_Y + (isPrefix ? 0.04 : 0.08), cz: kz,
              ex: kvSliceW, ey: SHEET_THICK, ez: kvSliceDepth,
              axes: FRAME_XZ_AXES,
              gridCols: denoiserSim.dHead, gridRows: 1,
              sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_V, fp8Flag: 0,
              highlight: resolveSheetHighlight(
                denFocus,
                { tensor: 'kv_v', kvHeadIndex: g, tokenIndex: k, isAggregate: false },
                isPrefix ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
              ),
              seed: 60 + g * 32 + k, partId: partIdV,
              towerRole: 'denoiser', executionId: denExecId, traceOrigin: sliceOrigin,
              semanticTag: semTagV, absoluteTokenOffset: 0,
              kvHeadIndex: g, tokenIndex: k, tensorKey: 'kv_v',
              atlasSlice: denSlices.kvV[g][k], registryKey: `den:kv_v:${g}:${k}`,
              cellFamily: CELL_SSM_CYAN,
            },
            denoiserRevealCtx,
            isPrefix,
            !isPrefix,
          );
        }

        const midPrefix = Math.max(0, Math.min(Math.max(0, P - 1), Math.floor(Math.max(0, P - 1) * 0.5)));
        denBridgeInK[g] = [xKTrack - kvSliceW * 0.55, BASE_Y + 0.14, kvSliceZ(g, midPrefix)];
        denBridgeInV[g] = [xVTrack - kvSliceW * 0.55, BASE_Y + 0.28, kvSliceZ(g, midPrefix)];

        // Scores [B × (P+B)] & Softmax [B × (P+B)]
        b.addSheet(
          {
            cx: xScores, cy: DECK_Y, cz: zGrp[g],
            ex: scoreW, ey: SHEET_THICK, ez: scoreD,
            axes: FRAME_XZ_AXES,
            gridCols: K, gridRows: T,
            sheetKind: isBidirectional ? SHEET_KIND_STANDARD : SHEET_KIND_CAUSAL_SCORES,
            role: SHEET_ROLE_SCORES, fp8Flag: 0,
            highlight: resolveSheetHighlight(
              denFocus,
              { tensor: 'scores', headIndex: h, kvHeadIndex: g, isAggregate: false },
              SHEET_HL_NORMAL,
            ),
            seed: 70 + g, partId: `den-scores-g${g}`,
            towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
            semanticTag: isBidirectional ? 'denoiser_bidirectional_scores' : 'denoiser_causal_scores',
            absoluteTokenOffset: P,
            headIndex: h, kvHeadIndex: g, tensorKey: 'scores',
            atlasSlice: denSlices.scores[g], registryKey: `den:scores:${g}`,
            cellFamily: CELL_WEIGHT_BLUE,
          },
          denoiserRevealCtx,
        );

        b.addSheet(
          {
            cx: xSoft, cy: DECK_Y, cz: zGrp[g],
            ex: scoreW, ey: SHEET_THICK, ez: scoreD,
            axes: FRAME_XZ_AXES,
            gridCols: K, gridRows: T,
            sheetKind: isBidirectional ? SHEET_KIND_STANDARD : SHEET_KIND_CAUSAL_SOFTMAX,
            role: SHEET_ROLE_SOFTMAX, fp8Flag: 0,
            highlight: resolveSheetHighlight(
              denFocus,
              { tensor: 'softmax', headIndex: h, kvHeadIndex: g, isAggregate: false },
              SHEET_HL_NORMAL,
            ),
            seed: 72 + g, partId: `den-softmax-g${g}`,
            towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
            semanticTag: isBidirectional ? 'denoiser_bidirectional_softmax' : 'denoiser_causal_softmax',
            absoluteTokenOffset: P,
            headIndex: h, kvHeadIndex: g, tensorKey: 'softmax',
            atlasSlice: denSlices.softmax[g], registryKey: `den:softmax:${g}`,
            cellFamily: CELL_ACTIVATION_GREEN,
          },
          denoiserRevealCtx,
        );

        // Weighted V_out [16×4 = 64 cols × T rows]
        b.addSheet(
          {
            cx: xVOut, cy: DECK_Y, cz: zGrp[g],
            ex: 0.72 * dScaleX, ey: SHEET_THICK, ez: 1.55,
            axes: FRAME_XZ_AXES,
            gridCols: 64, gridRows: T,
            sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
            highlight: resolveSheetHighlight(
              denFocus,
              { tensor: 'v_out', kvHeadIndex: g, isAggregate: true },
              SHEET_HL_NORMAL,
            ),
            seed: 74 + g, partId: `den-v-out-g${g}`,
            towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
            semanticTag: 'denoiser_v_out', absoluteTokenOffset: P,
            kvHeadIndex: g, tensorKey: 'v_out',
            atlasSlice: denSlices.vOut[g], registryKey: `den:v_out:${g}`,
            cellFamily: CELL_ACTIVATION_GREEN,
          },
          denoiserRevealCtx,
        );

        // Internal Denoiser flow ribbons
        b.flow.addPolyline3D(
          [
            [xKTrack + kvSliceW * 0.52, BASE_Y + 0.08, zGrp[g]],
            [xScores - scoreW * 0.52, DECK_Y + 0.03, zGrp[g]],
          ],
          [0, 1, 0], 0.08, RIBBON_K_FLOW, true, 0.15, 0.15, false,
        );
        b.flow.addPolyline3D(
          [
            [xScores + scoreW * 0.52, DECK_Y + 0.03, zGrp[g]],
            [xSoft - scoreW * 0.52, DECK_Y + 0.03, zGrp[g]],
          ],
          [0, 1, 0], 0.08, RIBBON_QUIET, true, 0.15, 0.15, false,
        );
        b.flow.addPolyline3D(
          [
            [xVTrack + kvSliceW * 0.52, BASE_Y + 0.08, zGrp[g]],
            [xVOut - 0.38 * dScaleX, DECK_Y + 0.03, zGrp[g]],
          ],
          [0, 1, 0], 0.08, RIBBON_V_FLOW, true, 0.15, 0.15, false,
        );

        // KV Depth Ticks along Group 0 (0, P-1, K-1, capacity 24)
        if (g === 0) {
          b.kvDepthTicks.push({ token: 1, world: [xKTrack, BASE_Y + 0.14, kvSliceZ(0, 0)] });
          if (P > 0) {
            b.kvDepthTicks.push({ token: P, world: [xKTrack, BASE_Y + 0.14, kvSliceZ(0, P - 1)] });
          }
          if (K > P) {
            b.kvDepthTicks.push({ token: K, world: [xKTrack, BASE_Y + 0.14, kvSliceZ(0, K - 1)] });
          }
          b.kvDepthTicks.push({
            token: allocatedKvCapacity,
            world: [xKTrack, BASE_Y + 0.14, kvSliceZ(0, allocatedKvCapacity - 1)],
          });
        }
      }

      // 2e. Concat [128 × T], Upright W_O [8 × 128], + Residual Gate Output Y [8 × T]
      b.addSheet(
        {
          cx: xConcat, cy: DECK_Y, cz: 0,
          ex: 0.52 * dScaleX, ey: SHEET_THICK, ez: 3.15,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.qConcatDim, gridRows: T,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_ACTIVATION, fp8Flag: 0,
          highlight: hlDen('concat', SHEET_HL_NORMAL),
          seed: 80.1, partId: 'den-concat',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_concat', absoluteTokenOffset: P,
          tensorKey: 'concat', atlasSlice: denSlices.concat, registryKey: 'den:concat',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealCtx,
      );

      b.addSheet(
        {
          cx: xWO, cy: 1.50, cz: 0,
          ex: 3.80, ey: SHEET_THICK, ez: 0.78,
          axes: FRAME_YZ_AXES,
          gridCols: denoiserSim.qConcatDim, gridRows: denoiserSim.d,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_WEIGHT, fp8Flag: 0,
          highlight: hlDen('w_o', SHEET_HL_NORMAL),
          seed: 81.1, partId: 'den-w-o',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_weight_wo',
          tensorKey: 'w_o', atlasSlice: denSlices.wO, registryKey: 'den:w_o',
          cellFamily: CELL_WEIGHT_BLUE,
        },
        denoiserRevealCtx,
      );

      b.addSheet(
        {
          cx: xOut, cy: BASE_Y, cz: 0,
          ex: ioW, ey: SHEET_THICK, ez: ioD,
          axes: FRAME_XZ_AXES,
          gridCols: denoiserSim.d, gridRows: T,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_RESIDUAL, fp8Flag: 0,
          highlight: hlDen('residual', SHEET_HL_NORMAL),
          seed: 82.1, partId: 'den-residual',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: denoiserSim.residualGateApplied ? 'denoiser_gated_residual' : 'denoiser_residual',
          absoluteTokenOffset: P,
          tensorKey: 'residual', atlasSlice: denSlices.residual, registryKey: 'den:residual',
          cellFamily: CELL_ACTIVATION_GREEN,
        },
        denoiserRevealCtx,
      );

      // Overhead Emerald Residual Bypass (unmodulated Input X -> gated Residual Output Y)
      const bridgeY = 3.08;
      const bridgeZ = -3.58;
      b.flow.addCubic3D(
        [xIn, BASE_Y + 0.04, -ioD * 0.5],
        [xIn, bridgeY * 0.85, -ioD * 0.5 - 0.7],
        [xIn + 0.4, bridgeY, bridgeZ],
        [xOut - 0.4, bridgeY, bridgeZ],
        [0, 1, 0], 0.13, RIBBON_RESIDUAL, false, 18, true,
      );
      b.flow.addCubic3D(
        [xOut - 0.4, bridgeY, bridgeZ],
        [xOut, bridgeY, bridgeZ],
        [xOut, bridgeY * 0.85, -ioD * 0.5 - 0.7],
        [xOut, BASE_Y + 0.08, -ioD * 0.5 - 0.02],
        [0, 1, 0], 0.13, RIBBON_RESIDUAL, true, 12, true, 0.24, 0.24,
      );

      // Denoiser Landmarks & 4-Length Calipers
      b.landmarks.push(
        {
          id: 'den-wing-header',
          label: historicalBoundaryMismatch
            ? `HISTORICAL Denoiser Tower (Rec #${dl.sourceRecordIndex}, P_cond=${contextPrefixKvLen} ≠ P=${currentContextPrefixLen})`
            : `Denoiser Tower (${isBidirectional ? 'Bidirectional Block' : 'Mock-AR Causal'})`,
          sublabel:
            (historicalBoundaryMismatch ? `${historicalTag} · ` : '') +
            `Q=${queryLen} · Prefix KV=${contextPrefixKvLen} · Block KV=${blockKvLen} · Capacity=${allocatedKvCapacity} · Model 32Q/2KV (2 score slices: ${fmtHead(effG0)}, ${fmtHead(effG1)})`,
          world: [denX(0.5), 2.95, -3.75],
          bayId,
          partId: 'den-scores-g0',
          tone: historicalBoundaryMismatch ? 'gold' : 'default',
          align: 'center',
          tier: 'primary',
        },
        {
          id: 'den-kv-bank-landmark',
          label: historicalBoundaryMismatch
            ? `Historical Concatenated KV (Prior Prefix KV[0..${Math.max(0, P - 1)}] P_cond=${contextPrefixKvLen} ≠ P=${currentContextPrefixLen} + ${blockKvLen} Block / ${allocatedKvCapacity} Cap)`
            : `Concatenated KV (${contextPrefixKvLen} Prefix + ${blockKvLen} Block / ${allocatedKvCapacity} Cap)`,
          sublabel: historicalBoundaryMismatch
            ? `Prior pass prefix [0..${Math.max(0, P - 1)}] from Rec #${dl.sourceRecordIndex} (NOT current Context P=${currentContextPrefixLen}) · Block [${P}..${K - 1}] denoiser_scratch`
            : `Prefix [0..${Math.max(0, P - 1)}] inherited read-only · Block [${P}..${K - 1}] denoiser_scratch · [${K}..${allocatedKvCapacity - 1}] unused`,
          world: [xKTrack, 1.35, -3.35],
          bayId,
          partId: 'den-prefix-kv',
          tone: 'gold',
          align: 'center',
          tier: 'primary',
        },
      );

      // 4 explicit dimension calipers distinguishing queryLen, contextPrefixKvLen, blockKvLen, allocatedKvCapacity
      b.calipers.push(
        {
          id: 'caliper-query-len',
          aWorld: [xScores - scoreW * 0.55, DECK_Y + 0.08, zGrp[0] - scoreD * 0.5],
          bWorld: [xScores - scoreW * 0.55, DECK_Y + 0.08, zGrp[0] + scoreD * 0.5],
          label: `queryLen = ${queryLen}`,
          partIds: ['den-scores-g0'],
        },
        {
          id: 'caliper-prefix-kv-len',
          aWorld: [xKTrack - kvSliceW * 0.65, BASE_Y + 0.14, zGrp[0] - groupSpanZ * 0.5],
          bWorld: [xKTrack - kvSliceW * 0.65, BASE_Y + 0.14, zGrp[0] - groupSpanZ * 0.5 + Math.max(1, P) * kvPitchZ],
          label: `contextPrefixKvLen = ${contextPrefixKvLen}`,
          partIds: ['den-prefix-kv'],
        },
        {
          id: 'caliper-block-kv-len',
          aWorld: [xKTrack + kvSliceW * 0.65, BASE_Y + 0.14, zGrp[0] - groupSpanZ * 0.5 + P * kvPitchZ],
          bWorld: [xKTrack + kvSliceW * 0.65, BASE_Y + 0.14, zGrp[0] - groupSpanZ * 0.5 + K * kvPitchZ],
          label: `blockKvLen = ${blockKvLen}`,
          partIds: ['den-block-kv'],
        },
        {
          id: 'caliper-allocated-kv-cap',
          aWorld: [xKTrack, BASE_Y - 0.04, zGrp[1] - groupSpanZ * 0.5],
          bWorld: [xKTrack, BASE_Y - 0.04, zGrp[1] + groupSpanZ * 0.5],
          label: `allocatedKvCapacity = ${allocatedKvCapacity} (${unusedKvCapacity} unused)`,
          partIds: ['den-kv-capacity-k-g1'],
        },
      );
    } else {
      // Context-only state: Denoiser standby card (never invents fake Nano sim)
      b.addSheet(
        {
          cx: denX(0.5), cy: BASE_Y + 0.16, cz: 0,
          ex: 4.2, ey: SHEET_THICK, ez: 2.2,
          axes: FRAME_XZ_AXES,
          gridCols: 4, gridRows: 2,
          sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_NEUTRAL, fp8Flag: 0,
          highlight: SHEET_HL_DIMMED,
          seed: 90.1, partId: 'den-standby',
          towerRole: 'denoiser', executionId: denExecId, traceOrigin: 'denoiser_scratch',
          semanticTag: 'denoiser_awaiting_pass',
          cellFamily: CELL_NEUTRAL,
        },
        null,
      );
      b.landmarks.push({
        id: 'den-wing-header',
        label: 'Denoiser Tower (Awaiting Pass)',
        sublabel: `Context-only state · Committed Prefix KV = ${contextPrefixKvLen} / ${allocatedKvCapacity}`,
        world: [denX(0.5), 2.10, -2.5],
        bayId,
        partId: 'den-standby',
        tone: 'muted',
        align: 'center',
        tier: 'primary',
      });
    }
  }

  // ==========================================================================
  // 3. CENTER LOGICAL CONDITIONING BRIDGE (`x ∈ [-3.5, -1.8]` in paired mode)
  // ==========================================================================
  const prefixSliceDesc =
    dl.pass?.mode === 'mock_ar' || (dl.pass === null && view.modeAtRecord === 'mock_ar')
      ? `KV[0..${Math.max(0, contextPrefixKvLen - 1)}] (KV[:-1] Mock-AR)`
      : `KV[0..${Math.max(0, contextPrefixKvLen - 1)}]`;
  const bridgeNote = historicalBoundaryMismatch
    ? `Historical logical read-only prefix conditioning (${prefixSliceDesc}, P_cond=${contextPrefixKvLen} from ${historicalTag}, ownerRole: 'context_tower', consumerRole: 'denoiser_tower') — does NOT match current Context boundary P=${currentContextPrefixLen}; NOT requiring a physical inter-device copy.`
    : `Logical read-only prefix conditioning (${prefixSliceDesc}, ownerRole: 'context_tower', consumerRole: 'denoiser_tower') — NOT requiring a physical inter-device copy.`;

  if (towerFocus === 'paired') {
    // Divider lines & bridge plaque in x ∈ [-3.5, -1.8]
    b.addGuideLine(-3.5, BASE_Y, -3.75, -3.5, BASE_Y, 3.75, 1.1, LINE_STYLE_GUIDE, 0.60);
    b.addGuideLine(-1.8, BASE_Y, -3.75, -1.8, BASE_Y, 3.75, 1.1, LINE_STYLE_GUIDE, 0.60);

    b.addSheet(
      {
        cx: -2.65, cy: BASE_Y + 0.02, cz: 0,
        ex: 1.35, ey: 0.036, ez: 1.15,
        axes: FRAME_XZ_AXES,
        gridCols: 2, gridRows: 2,
        sheetKind: SHEET_KIND_STANDARD, role: SHEET_ROLE_KV_K, fp8Flag: 0,
        highlight: SHEET_HL_BF16_GOLD,
        seed: 95.1, partId: 'conditioning-bridge',
        towerRole: 'shared', executionId: ctxExecId, traceOrigin: 'persistent_cache',
        semanticTag: 'gqa_kv_bridge',
        cellFamily: CELL_SSM_CYAN,
      },
      null,
    );

    for (let g = 0 as 0 | 1; g < 2; g = (g + 1) as 0 | 1) {
      const startK = ctxBridgeOutK[g] ?? [-3.55, BASE_Y + 0.14, zGrp[g]];
      const endK = denBridgeInK[g] ?? [-1.65, BASE_Y + 0.14, zGrp[g]];
      const midX = -2.65;
      b.flow.addCubic3D(
        startK,
        [midX - 0.25, BASE_Y + 0.34, startK[2]],
        [midX + 0.25, BASE_Y + 0.34, endK[2]],
        endK,
        [0, 1, 0],
        0.10,
        g === 0 ? RIBBON_BRIDGE_KV0 : RIBBON_BRIDGE_KV1,
        true,
        18,
        true,
        0.22,
        0.22,
      );

      const startV = ctxBridgeOutV[g] ?? [-3.55, BASE_Y + 0.28, zGrp[g] + 0.16];
      const endV = denBridgeInV[g] ?? [-1.65, BASE_Y + 0.28, zGrp[g] + 0.16];
      b.flow.addCubic3D(
        startV,
        [midX - 0.15, BASE_Y + 0.56, startV[2]],
        [midX + 0.15, BASE_Y + 0.56, endV[2]],
        endV,
        [0, 1, 0],
        0.08,
        g === 0 ? RIBBON_BRIDGE_KV0 : RIBBON_BRIDGE_KV1,
        true,
        18,
        true,
        0.18,
        0.18,
      );
    }

    b.landmarks.push({
      id: 'conditioning-bridge-landmark',
      label: historicalBoundaryMismatch
        ? `Historical Read-Only Prefix Bridge (${prefixSliceDesc} · P_cond=${contextPrefixKvLen} ≠ P=${currentContextPrefixLen})`
        : `Logical Read-Only Prefix Bridge (${prefixSliceDesc})`,
      sublabel: historicalBoundaryMismatch
        ? `${historicalTag} · Prior conditioning does NOT belong to current Context P=${currentContextPrefixLen}`
        : "ownerRole: 'context_tower' · Zero physical inter-device copy required",
      world: [-2.65, 1.35, 0],
      bayId,
      partId: 'conditioning-bridge',
      tone: 'gold',
      align: 'center',
      tier: 'primary',
    });
  }

  // ==========================================================================
  // 4. CELL HIGHLIGHTS, DEPENDENCY RIBBONS, POPOVER & ACTIVE CELL GRIDS
  // ==========================================================================
  let cellPopover: CellPopoverSpec | null = null;
  const activeWingPrefix: 'ctx' | 'den' = activeTower === 'denoiser' ? 'den' : 'ctx';

  if (activeFormula && activeSim) {
    const targetSpan3D = resolvePairedGqaSpan(
      b.sheetRegistry,
      activeWingPrefix,
      activeFormula.targetSpan,
      displayedHeads,
    );
    if (targetSpan3D) {
      drawCellHighlight3D(b.flow, targetSpan3D, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.024, true);
      const sourceWorlds: { readonly world: readonly [number, number, number]; readonly role: 'indigo' | 'teal' }[] = [];

      for (const up of activeFormula.upstreamSpans) {
        const upSpan3D = resolvePairedGqaSpan(b.sheetRegistry, activeWingPrefix, up, displayedHeads);
        if (!upSpan3D) continue;
        const isOpA = up.role === 'operand_a';
        drawOperandDependency3D(b.flow, upSpan3D, targetSpan3D, isOpA);
        sourceWorlds.push({ world: upSpan3D.frontPoint, role: isOpA ? 'indigo' : 'teal' });
      }

      cellPopover = buildPairedGqaCellPopover(
        activeSim,
        activeFormula,
        activeTower,
        targetSpan3D.frontPoint,
        sourceWorlds,
        focusedAccumulation,
        historicalBoundaryMismatch
          ? `Historical P_cond=${contextPrefixKvLen} ≠ P=${currentContextPrefixLen}, Rec #${dl.sourceRecordIndex}`
          : null,
      );
    }
  }

  const activeCellGrids: readonly ActiveSheetCellGrid[] =
    activeSim !== null
      ? buildPairedGqaActiveCellGrids(
          activeSim,
          b.sheetRegistry,
          activeWingPrefix,
          activeFormula,
          effG0,
          effG1,
          activeTower === 'denoiser' ? denoiserRevealCtx : contextRevealCtx,
          focusedAccumulation,
        )
      : [];

  // ==========================================================================
  // 5. PART CATALOG, SELECTION IDENTITY, BADGES, WARNINGS & DIAGNOSTICS
  // ==========================================================================
  const miniGqa = TWOTOWER_MINI_MODEL_INVENTORY.gqa;
  const partCatalog: Record<string, CutawayPartInfo> = {
    'ctx-kv-cache': {
      id: 'ctx-kv-cache',
      title: 'Context Persistent KV Cache',
      badge: 'ownerRole: context_tower',
      summary: `Persistent causal KV cache owned exclusively by Context Tower (${currentContextPrefixLen}/${allocatedKvCapacity} populated slots).`,
      specs: [
        `Current Context prefix: KV[0..${Math.max(0, currentContextPrefixLen - 1)}] (P = ${currentContextPrefixLen})`,
        `Denoiser pass conditioning prefix: contextPrefixKvLen = ${contextPrefixKvLen}${historicalBoundaryMismatch ? ` (historical Rec #${dl.sourceRecordIndex} ≠ current P=${currentContextPrefixLen})` : ''}`,
        `Complete miniature GQA inventory: ${miniGqa.queryHeads} query heads / ${miniGqa.kvHeads} KV heads, head width dHead = ${miniGqa.headDim} (16:1 grouping)`,
        `allocatedKvCapacity = ${allocatedKvCapacity}`,
        `Context fingerprint: ${view.model.contextModel.parameterFingerprint}`,
      ],
    },
    'conditioning-bridge': {
      id: 'conditioning-bridge',
      title: 'Logical Read-Only Conditioning Bridge',
      badge: historicalBoundaryMismatch ? 'Historical Conditioning · Mismatch' : 'Zero Physical Copy',
      summary: bridgeNote,
      specs: [
        `Exposed prefix slice: ${prefixSliceDesc} (P_cond=${contextPrefixKvLen}, current Context P=${currentContextPrefixLen})`,
        ...(historicalTag ? [`Historical pass: ${historicalTag}`] : []),
        "ownerRole: 'context_tower' → consumerRole: 'denoiser_tower'",
        'Logical read-only view — no inter-device physical KV duplication required',
      ],
    },
    'den-prefix-kv': {
      id: 'den-prefix-kv',
      title: historicalBoundaryMismatch
        ? `Historical Inherited Prefix KV [0..${Math.max(0, contextPrefixKvLen - 1)}] (P_cond=${contextPrefixKvLen} ≠ P=${currentContextPrefixLen})`
        : `Inherited Context Prefix KV [0..${Math.max(0, contextPrefixKvLen - 1)}]`,
      badge: historicalBoundaryMismatch ? 'Historical Prefix · Prior Boundary' : 'persistent_cache · Read-Only',
      summary: historicalBoundaryMismatch
        ? `Read-only Context prefix KV[0..${Math.max(0, contextPrefixKvLen - 1)}] read by ${historicalTag}; does NOT belong to current Context boundary P=${currentContextPrefixLen}.`
        : 'Read-only Context prefix KV slices referenced by Denoiser attention; NEVER recomputed or ghosted during projection reveal.',
      specs: [
        `contextPrefixKvLen = ${contextPrefixKvLen} (current Context P = ${currentContextPrefixLen})`,
        ...(dl.sourceContextCacheDigest ? [`Source Context cache digest: ${dl.sourceContextCacheDigest}`] : []),
        'Reveal policy: SHEET_REVEAL_NONE across all steps 0..5',
      ],
    },
    'den-block-kv': {
      id: 'den-block-kv',
      title: `Denoiser Scratch Block KV [${contextPrefixKvLen}..${Math.max(contextPrefixKvLen, totalAttendedKvLen - 1)}]`,
      badge: historicalBoundaryMismatch ? 'Historical Scratch · Prior Pass' : 'denoiser_scratch · Ephemeral',
      summary: 'Newly projected Denoiser K/V slices computed with independent Denoiser W_K/W_V weights and discarded between refinements.',
      specs: [
        `queryLen = ${queryLen}`,
        `blockKvLen = ${blockKvLen}`,
        ...(historicalTag ? [`Source pass: ${historicalTag}`] : []),
        `Denoiser fingerprint: ${view.model.denoiserModel.parameterFingerprint}`,
      ],
    },
    'den-scores-g0': {
      id: 'den-scores-g0',
      title: `Denoiser Attention Scores (${fmtHead(effG0)} — Selected Slice 1 of 2 Displayed / 32 Model Q Heads)`,
      badge: denoiserSim?.attentionMaskMode === 'bidirectional_block' ? 'Bidirectional Block' : 'Causal',
      summary:
        denoiserSim?.attentionMaskMode === 'bidirectional_block'
          ? `Full bidirectional block attention over [0..${totalAttendedKvLen - 1}] (0 masked cells).`
          : `Causal attention over [0..${totalAttendedKvLen - 1}] (k > P + q masked to −∞).`,
      specs: [
        `Model inventory: ${miniGqa.queryHeads} miniature Q heads / ${miniGqa.kvHeads} KV heads, head width ${miniGqa.headDim}; 128×128 atlas shows 2 selected Scores/Softmax head slices (${fmtHead(effG0)}, ${fmtHead(effG1)})`,
        `Shape: [Q=${queryLen} × K=${totalAttendedKvLen}] (Prefix ${contextPrefixKvLen} + Block ${blockKvLen})`,
        `Allocated KV capacity: ${allocatedKvCapacity} (${unusedKvCapacity} unused)`,
      ],
    },
  };

  const activeSelection: TwoTowerSelectionIdentity = buildTwoTowerSelectionIdentity(view, {
    towerRole: activeTower,
    bayId,
    mixerKind: '*',
    tensorRole: activeCell?.tensor ?? 'scores',
    traceOrigin:
      activeTower === 'denoiser'
        ? activeCell && (activeCell.tensor === 'kv_k' || activeCell.tensor === 'kv_v') && (activeCell.tokenIndex ?? 0) < contextPrefixKvLen
          ? 'persistent_cache'
          : 'denoiser_scratch'
        : ctxLookup.traceOrigin,
    executionId: activeTower === 'denoiser' ? denExecId : ctxExecId,
    absoluteTokenPosition:
      activeCell && activeSim
        ? activeCell.tensor === 'kv_k' || activeCell.tensor === 'kv_v'
          ? (activeCell.tokenIndex ?? 0)
          : activeSim.startTokenOffset + activeCell.row
        : null,
    headIndex: activeFocusHead,
    kvHeadIndex: activeFocusGrp,
    provenanceNote:
      `Paired GQA Bay ${String(bayId).padStart(2, '0')} (layer ${layerIndex}) · ${activeTower.toUpperCase()} · ` +
      `Model 32Q/2KV (dHead=4; score slices ${fmtHead(effG0)}, ${fmtHead(effG1)}) · ` +
      `Q=${queryLen}, Prefix=${contextPrefixKvLen}, Block=${blockKvLen}, Cap=${allocatedKvCapacity}` +
      (historicalTag ? ` · ${historicalTag}` : ''),
  });

  const warnings: string[] = [];
  if (!view.snapshotIntegrity.valid) {
    warnings.push(`Context snapshot integrity check failed: ${view.snapshotIntegrity.error}`);
  }
  if (ctxLookup.status === 'pruned_unavailable') {
    warnings.push(ctxLookup.reason);
  }
  if (dl.historicalProvenanceWarning !== null) {
    warnings.push(dl.historicalProvenanceWarning);
  }
  const warningBanner = warnings.length > 0 ? warnings.join(' | ') : null;

  const provenanceBadges: TwoTowerProvenanceBadge[] = [
    {
      label: `Bay ${String(bayId).padStart(2, '0')} (Layer ${layerIndex}) · GQA 32Q/2KV dHead=4 (Score slices: ${fmtHead(effG0)}, ${fmtHead(effG1)})`,
      tone: 'shared',
    },
    {
      label: `Context: ${ctxLookup.traceOrigin} (P=${currentContextPrefixLen})`,
      tone: ctxLookup.status === 'available' ? 'context' : 'warn',
      origin: ctxLookup.traceOrigin,
    },
    {
      label: denoiserSim ? `Denoiser: ${denoiserSim.attentionMaskMode}` : 'Denoiser: standby',
      tone: denoiserSim ? 'denoiser' : 'muted',
      origin: 'denoiser_scratch',
    },
    {
      label: `Q=${queryLen} · Prefix=${contextPrefixKvLen} · Block=${blockKvLen} · Cap=${allocatedKvCapacity}`,
      tone: historicalBoundaryMismatch ? 'warn' : 'gold',
    },
  ];
  if (isHistoricalDenoiser) {
    provenanceBadges.push({
      label: `Historical Denoiser Pass · Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, P_cond=${contextPrefixKvLen} vs P=${currentContextPrefixLen})`,
      tone: 'warn',
      origin: 'denoiser_scratch',
    });
  }

  const fitBounds: AABB =
    towerFocus === 'context_solo'
      ? CONTEXT_SOLO_GQA_FIT_BOUNDS
      : towerFocus === 'denoiser_solo'
        ? DENOISER_SOLO_GQA_FIT_BOUNDS
        : PAIRED_GQA_FIT_BOUNDS;

  const lengths: PairedGqaLengthDiagnostics = Object.freeze({
    queryLen,
    contextPrefixKvLen,
    blockKvLen,
    allocatedKvCapacity,
    totalAttendedKvLen,
    unusedKvCapacity,
  });

  const diagnostics: PairedGqaDiagnostics = Object.freeze({
    bayId,
    layerIndex,
    towerFocus,
    modeAtRecord: view.modeAtRecord,
    attentionMaskMode: denoiserSim?.attentionMaskMode ?? contextSim?.attentionMaskMode ?? 'causal',
    queryLen,
    contextPrefixKvLen,
    blockKvLen,
    allocatedKvCapacity,
    lengths,
    contextTraceStatus: ctxLookup.status,
    contextTraceOrigin: ctxLookup.traceOrigin,
    denoiserPassStatus: dl.status,
    currentContextPrefixLen,
    denoiserConditioningPrefixLen,
    denoiserSourceContextCacheDigest: dl.sourceContextCacheDigest,
    denoiserSourceRecordIndex: dl.sourceRecordIndex,
    denoiserSourceBlockId: dl.sourceBlockId,
    denoiserSourceRefinementIndex: dl.sourceRefinementIndex,
    isHistoricalDenoiserInspection: isHistoricalDenoiser,
    matchesCurrentContextBoundary: matchesCurrentBoundary,
    historicalProvenanceWarning: dl.historicalProvenanceWarning,
    hasDenoiserPreNormMod: Boolean(denoiserSim?.X_mod),
    hasDenoiserResidualGate: Boolean(denoiserSim?.residualGateApplied),
    conditioningOwnerRole: 'context_tower',
    conditioningConsumerRole: 'denoiser_tower',
    conditioningTransportSemantics: 'logical_read_only_no_physical_copy',
    conditioningNote: bridgeNote,
    contextParameterFingerprint: view.model.contextModel.parameterFingerprint,
    denoiserParameterFingerprint: view.model.denoiserModel.parameterFingerprint,
    prefixKvReveals: Object.freeze(b.prefixKvReveals),
    blockKvReveals: Object.freeze(b.blockKvReveals),
    modelGqaInventory: Object.freeze({
      queryHeads: miniGqa.queryHeads,
      kvHeads: miniGqa.kvHeads,
      queriesPerKvGroup: miniGqa.queriesPerKvGroup,
      headDim: miniGqa.headDim,
      displayedScoreHeads: Object.freeze([effG0, effG1] as const),
      displayedScoreHeadCount: 2 as const,
    }),
    atlasDimensions: [TWOTOWER_GQA_ATLAS_WIDTH, TWOTOWER_GQA_ATLAS_HEIGHT] as const,
    atlasPrimaryMaskedCells: gqaAtlas?.primary.maskedScoreCellCount ?? 0,
    atlasSecondaryMaskedCells: gqaAtlas?.secondary?.maskedScoreCellCount ?? 0,
  });

  const packed = b.pack();
  const valueAtlas: Rg32fValueAtlas | undefined = gqaAtlas ?? fallbackCacheAtlas ?? undefined;
  const atlasKey = `tt-gqa:${view.identity.recordId}:${ctxExecId}:${denExecId}:b${bayId}:f${towerFocus}:g0-${effG0}:g1-${effG1}`;

  const displayFormula =
    activeFormula && focusedAccumulation?.active
      ? (() => {
          const provTag = activeFormula.subEquation.split(' · ')[0] ?? '';
          return {
            ...activeFormula,
            value: focusedAccumulation.partialSum,
            valueText: focusedAccumulation.valueBadgeText,
            equation: focusedAccumulation.partialEquation,
            subEquation: provTag
              ? `${provTag} · ${focusedAccumulation.partialSubEquation}`
              : focusedAccumulation.partialSubEquation,
            headBadge: focusedAccumulation.badgeText,
          };
        })()
      : activeFormula;

  return {
    viewMode: 'paired_gqa',
    title: `Paired GQA Cutaway · Bay ${String(bayId).padStart(2, '0')} (Layer ${layerIndex})`,
    subtitle:
      (historicalBoundaryMismatch ? `HISTORICAL Denoiser (${historicalTag} ≠ P=${currentContextPrefixLen}) · ` : '') +
      `35/65 Context Provenance → Logical Read-Only Prefix Bridge → Denoiser ` +
      `(Q=${queryLen}, Prefix=${contextPrefixKvLen}, Block=${blockKvLen}, Cap=${allocatedKvCapacity} · 32Q/2KV, slices ${fmtHead(effG0)}/${fmtHead(effG1)})`,
    bayId,
    layerIndex,
    activeTowerFocus: towerFocus,
    ...packed,
    ...(valueAtlas ? { valueAtlas, atlasKey } : {}),
    fitBounds,
    landmarks: b.landmarks,
    kvDepthTicks: b.kvDepthTicks,
    calipers: b.calipers,
    solidPickBoxes: b.pickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    ...(b.activeOpPartIds.size > 0 ? { activeOpPartIds: b.activeOpPartIds } : {}),
    activeCellGrids,
    cellPopover,
    activeFormula: displayFormula,
    focusedAccumulation,
    activeSelection,
    stepperSteps: GQA_STEPPER_STEPS,
    provenanceBadges,
    epistemicNote:
      'Synthetic miniature Float32 TwoTower simulation (d=8, 32 miniature GQA query heads / 2 KV heads, head width 4, kvCapacity=24; not 63.19B checkpoint weights). ' +
      'The 128×128 atlas packs all 32 Q heads and both KV heads while displaying 2 selected Q-head slices (1 per KV group) on Scores/Softmax plates. ' +
      'Prefix KV conditioning is a logical read-only view of Context Tower memory (no physical copy required).',
    warningBanner,
    diagnostics: diagnostics as unknown as Readonly<Record<string, unknown>>,
  };
}
