/**
 * Stage A+1 · Increment 3: Mamba-2 Cutaway Foundation & Live Mathematics
 * (Bay 01 · Ordinary Mamba-2 & Bay 42 · Feeder before Bay 43 GQA)
 *
 * Built in the Increment 1+2 thin-sheet XZ drafting-table visual language:
 *   - SheetsPass thin flat matrix sheets (ey ≈ 0.07) backed by a 256×64 RG32F live value atlas
 *   - FlatFlowRibbonsPass smooth vector pathways + dependency threads (zero voxel-bead arrows)
 *   - Invariant MAMBA_FIT_BOUNDS across T=1..16 and BF16/FP8 Selective
 *   - Fixed-size active-head H_{t-1} ⇄ H_t recurrence + 64-head fixed FP32 SSM bank
 *     juxtaposed directly against T-growing sequence tensors (xBC_pre, z, x, B, C, Δt)
 *   - Individual miniature head selection H00..H15 across 8 B/C groups G0..G7
 *   - Honest Bay 01 vs Bay 42 Feeder FP8 Selective precision metadata (NVIDIA §4.2)
 */

import {
  describeMambaCellFormula,
  formatMambaHeadId,
  inferMambaCellHead,
  inferMambaCellToken,
  MAMBA_SIM_CONFIG,
  mambaGroupForHead,
  runMambaSim,
  type MambaCellFormula,
  type MambaCellRef,
  type MambaSheetSpan,
  type MambaSimResult,
  type MambaTensorKey,
} from '../math/mambaSim';
import {
  formatBayId,
  NEMOTRON_SPEC,
  type PrecisionMode,
} from '../model/spec';
import type { GqaViewAngle } from '../model/state';
import type { AABB } from '../render/camera';
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  pushFreeGuideLine,
} from '../render/linePacking';
import type { PickBox } from '../render/pick';
import { FlowRibbonBuilder, type FlowRibbonColor } from '../render/ribbons';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_STANDARD,
  SHEET_REVEAL_FLOATS,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../render/sheets';
import type { AtlasSlice } from './atlasPacker';
import type { CutawayPartInfo, CutawaySceneGeometry } from './cutaways';
import { buildMambaValueAtlas } from './mambaAtlas';
import {
  buildMambaActiveCellGrids,
  buildMambaCellPopover,
  buildMambaSpatialScene,
  computeMambaSheetReveal,
  fitSpatialMambaCamera,
  MAMBA_SPATIAL_FIT_BOUNDS,
  resolveMambaFocusedAccumulation,
  type MambaRevealParams,
} from './mambaSpatial';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  computeSheetGroupBounds,
  createSheetFocusSet,
  drawOperandDependencyXZ,
  EMPTY_SHEET_FOCUS_SET,
  rectFromSheetGrid,
  resolveSheetFocusRole,
  resolveSheetHighlight,
  RIBBON_ACTIVE,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  RIBBON_RESIDUAL_ACTIVE,
  RIBBON_SSM_LOOP,
  type SheetFocusEntry,
  type SheetFocusSet,
  type SheetGeomRecord,
  type WorldSpanRect,
} from './sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type LandmarkAnchor,
} from './spine';

export { fitSpatialMambaCamera, MAMBA_SPATIAL_FIT_BOUNDS };

/** Parse `'mamba-head-0'` .. `'mamba-head-15'` into integer head index `0..15`, or `null`. */
export function parseMambaHeadPartId(partId: string | null | undefined): number | null {
  if (!partId || !partId.startsWith('mamba-head-')) return null;
  const n = Number.parseInt(partId.slice('mamba-head-'.length), 10);
  if (!Number.isInteger(n) || n < 0 || n >= MAMBA_SIM_CONFIG.nHeads) return null;
  return n;
}

export interface MambaCutawayOptions {
  readonly bayId: 1 | 42;
  readonly tokens: number;
  readonly precision: PrecisionMode;
  readonly selectedPart: string | null;
  readonly hoveredPart: string | null;
  readonly activeHead?: number;
  readonly activeCell?: MambaCellRef | null;
  /** Optional active Math Stepper stage index `0..5` (used for V2 progressive accumulation reveal). */
  readonly stepperStep?: number | null;
  /** When true (and `!reducedMotion` with a valid `stepperStep`), emits V2 progressive `sheetRevealData`. Defaults to false. */
  readonly stepperActive?: boolean;
  /** Optional stepper sub-progress `0..1` for active frontier cell pulse animation. */
  readonly stepperSubProgress?: number;
  /** Optional focused-cell term accumulation progress `0..1` (default `1` = complete). */
  readonly termProgress?: number;
  /** Honors `prefers-reduced-motion`: omits `sheetRevealData` so sheets remain 100% revealed statically. Defaults to false. */
  readonly reducedMotion?: boolean;
  /** Layout mode: `'three_quarter'` builds the 3D Spatial Mamba-2 architecture; `'near_plan'` (default) keeps the coplanar XZ schematic. */
  readonly viewAngle?: GqaViewAngle;
  /** Optional explicit recurrence token index `0..T-1` for temporal playback (overridden when activeCell specifies a token). */
  readonly stateTokenIndex?: number | null;
  /** Optional authoritative simulator result from a unified `HybridExecutionResult`. */
  readonly simOverride?: MambaSimResult;
}

interface MambaSheetSpec {
  cx: number;
  cy: number;
  cz: number;
  ex: number;
  ey: number;
  ez: number;
  gridCols: number;
  gridRows: number;
  role: number;
  fp8Flag: number;
  highlight: number;
  seed: number;
  partId: string;
  headIndex?: number;
  groupIndex?: number;
  tensorKey?: MambaTensorKey;
  tokenIndex?: number;
  tapIndex?: number;
  stateTokenIndex?: number;
  atlasSlice?: AtlasSlice;
  registryKey?: string;
  cellFamily: number;
}

class MambaSheetBuilder {
  private readonly bayId: number;
  private readonly sheetFloats: number[] = [];
  private readonly revealFloats: number[] = [];
  private readonly solidFloats: number[] = [];
  private readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();
  readonly activeOpPartIds = new Set<string>();
  private readonly revealParams: MambaRevealParams | null;
  private readonly collectActiveOpParts: boolean;

  constructor(
    bayId: number,
    revealParams: MambaRevealParams | null = null,
    collectActiveOpParts = false,
  ) {
    this.bayId = bayId;
    this.revealParams = revealParams;
    this.collectActiveOpParts = collectActiveOpParts;
  }

  addSheet(s: MambaSheetSpec): void {
    const rect = s.atlasSlice?.rect ?? [0, 0, 0, 0];
    const scale = s.atlasSlice?.scale ?? 1.0;
    const modeFlag = s.atlasSlice?.mode ?? 0;

    // 1. Primary thin-sheet instance (20 floats = 80 bytes)
    this.sheetFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.gridCols, s.gridRows,
      SHEET_KIND_STANDARD,
      s.role,
      s.fp8Flag,
      s.highlight,
      s.seed,
      rect[0], rect[1], rect[2], rect[3],
      scale,
      modeFlag, 0,
    );

    if (this.revealParams !== null) {
      const [mode, colF, rowF, pulse] = computeMambaSheetReveal(this.revealParams, s);
      this.revealFloats.push(mode, colF, rowF, pulse);
    }

    if (
      this.collectActiveOpParts &&
      s.partId !== 'seq-rail' &&
      (s.highlight === SHEET_HL_ACTIVE ||
        s.highlight === SHEET_HL_OPERAND_A ||
        s.highlight === SHEET_HL_OPERAND_B)
    ) {
      this.activeOpPartIds.add(s.partId);
    }

    // 2. 1-to-1 mirror in solidData (12 floats: fp8Flag at +7, phase/seed at +8)
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

    // 3. Pick AABB (slightly padded vertically so thin sheets pick effortlessly)
    const hx = s.ex * 0.5;
    const hy = Math.max(s.ey * 0.5, 0.06);
    const hz = s.ez * 0.5;
    this.pickBoxes.push({
      bayId: this.bayId,
      partId: s.partId,
      headIndex: s.headIndex,
      groupIndex: s.groupIndex,
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
      tapIndex: s.tapIndex,
      stateTokenIndex: s.stateTokenIndex,
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
    pushFreeGuideLine(this.lineFloats, ax, ay, az, bx, by, bz, cssWidth, style, alpha);
  }

  pack() {
    const flowPacked = this.flow.pack();
    const sheetCount = this.sheetFloats.length / SHEET_INSTANCE_FLOATS;
    const sheetRevealData =
      this.revealParams !== null && this.revealFloats.length === sheetCount * SHEET_REVEAL_FLOATS
        ? new Float32Array(this.revealFloats)
        : undefined;
    return {
      sheetData: new Float32Array(this.sheetFloats),
      sheetCount,
      ...(sheetRevealData ? { sheetRevealData } : {}),
      solidData: new Float32Array(this.solidFloats),
      solidCount: this.solidFloats.length / SOLID_INSTANCE_FLOATS,
      lineData: new Float32Array(this.lineFloats),
      lineCount: this.lineFloats.length / LINE_INSTANCE_FLOATS,
      // Explicitly ZERO voxel-bead arrows in the rebuilt Mamba-2 cutaway
      ribbonData: new Float32Array(0),
      ribbonCount: 0,
      flowVertices: flowPacked.vertices,
      flowIndices: flowPacked.indices,
    };
  }
}

/** Invariant Mamba-2 cutaway bounds across T=1..16 and BF16/FP8 — prevents camera jump. */
export const MAMBA_FIT_BOUNDS: AABB = {
  min: [-6.65, -0.15, -6.35],
  max: [7.05, 0.25, 6.40],
};

function resolveMambaSpanRect(
  registry: ReadonlyMap<string, SheetGeomRecord>,
  span: MambaSheetSpan,
  displayedHead: number,
  displayedToken: number,
): WorldSpanRect | null {
  if (span.tensor === 'conv_buf') {
    if (span.tokenIndex !== undefined && span.tokenIndex !== displayedToken) {
      return null;
    }
    const t0 = span.tapStart ?? 0;
    const t1 = span.tapEnd ?? t0;
    const first = registry.get(`conv_buf:${t0}`);
    const last = registry.get(`conv_buf:${t1}`) ?? first;
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

  if (span.tensor === 'h_prev' || span.tensor === 'h_curr') {
    if (span.headIndex !== undefined && span.headIndex !== displayedHead) {
      return null;
    }
    if (span.tokenIndex !== undefined && span.tokenIndex !== displayedToken) {
      return null;
    }
    const sheet = registry.get(span.tensor);
    return sheet ? rectFromSheetGrid(sheet, span.rowStart, span.rowEnd, span.colStart, span.colEnd) : null;
  }

  const key = span.tensor === 'x' ? `x:${span.headIndex ?? displayedHead}` : span.tensor;
  const sheet = registry.get(key);
  if (!sheet) return null;
  return rectFromSheetGrid(sheet, span.rowStart, span.rowEnd, span.colStart, span.colEnd);
}

function buildMambaFocusSet(formula: MambaCellFormula | null): SheetFocusSet {
  if (!formula) return EMPTY_SHEET_FOCUS_SET;
  const toEntry = (span: MambaSheetSpan): SheetFocusEntry => ({
    tensor: span.tensor,
    headIndex: span.headIndex,
    groupIndex: span.groupIndex,
    tokenStart: span.tokenIndex,
    tokenEnd: span.tokenIndex,
    tapStart: span.tapStart,
    tapEnd: span.tapEnd,
    rowStart: span.rowStart,
    rowEnd: span.rowEnd,
    colStart: span.colStart,
    colEnd: span.colEnd,
    role: span.role,
  });
  return createSheetFocusSet([
    toEntry(formula.targetSpan),
    ...formula.upstreamSpans.map(toEntry),
  ]);
}

function buildMambaPartCatalog(params: {
  readonly bayId: 1 | 42;
  readonly clampedT: number;
  readonly isFeeder: boolean;
  readonly computeIsFp8: boolean;
  readonly feederPreservedBf16: boolean;
  readonly activeHead: number;
  readonly activeGroup: number;
  readonly stateTokIdx: number;
  readonly sim: MambaSimResult;
}): Record<string, CutawayPartInfo> {
  const {
    bayId,
    clampedT,
    isFeeder,
    computeIsFp8,
    feederPreservedBf16,
    activeHead,
    activeGroup,
    stateTokIdx,
    sim,
  } = params;
  const headTag = formatMambaHeadId(activeHead);
  const absTokIdx = sim.startTokenOffset + stateTokIdx;
  const tokLabel = sim.tokenLabels[stateTokIdx] ?? `t${absTokIdx}`;
  const zeroPads = Math.max(0, 3 - absTokIdx);
  const partCatalog: Record<string, CutawayPartInfo> = {
    'residual-in': {
      id: 'residual-in',
      title: 'Residual Stream Input (2,688)',
      badge: 'Activation · Emerald',
      summary:
        'Incoming hidden state from the continuous 2,688-channel emerald residual spine (miniature d=8). Taps left into Pre-RMSNorm while the untouched residual passes straight to the final + merge.',
      specs: [
        'Production shape: [T, 2,688] · Miniature live sim: [T, 8] (Float32 storage)',
        'Hover or click any cell to inspect its value and RMS magnitude.',
      ],
    },
    rmsnorm: {
      id: 'rmsnorm',
      title: 'Pre-Layer RMSNorm (ε = 1e−5)',
      badge: computeIsFp8 ? 'FP8 Selective (prod)' : feederPreservedBf16 ? 'BF16 Preserved §4.2' : 'BF16 (prod)',
      summary:
        'Normalizes each token vector by √(mean(X²) + 1e−5) and multiplies by learned non-unit scale γ_in before the fused Mamba-2 input projection.',
      specs: [
        'Matches NVIDIA NemotronH rms_norm_eps = 1e−5 (no bias, no mean shift)',
        `Bay ${formatBayId(bayId)} production metadata: ${computeIsFp8 ? 'FP8 compute' : feederPreservedBf16 ? 'BF16 preserved (Feeder §4.2)' : 'BF16'}`,
      ],
    },
    'in-proj': {
      id: 'in-proj',
      title: 'Fused Mamba-2 Input Projection (W_in)',
      badge: computeIsFp8 ? 'FP8 Weights (prod)' : feederPreservedBf16 ? 'BF16 Preserved §4.2' : 'BF16 Weights (prod)',
      summary:
        'Single fused projection splitting into three parallel branches: bypass gate z (4,096), Conv1D stream xBC_pre (6,144), and per-head discretization step logits Δt_raw (64).',
      specs: [
        'Production: 2,688 → 4,096(z) + 6,144(xBC) + 64(Δt) = 10,304 · Miniature: 8 → 64 + 128 + 16 = 208',
        isFeeder
          ? 'Bay 42 Feeder (before Bay 43 GQA): preserved in BF16 under NVIDIA §4.2 (gold rim).'
          : 'Bay 01 (ordinary Mamba-2): switches to FP8 metadata in FP8 Selective mode (live sim stays Float32).',
      ],
    },
    'gate-branch': {
      id: 'gate-branch',
      title: 'Bypass Gate Branch (z · 4,096)',
      badge: 'Gate FIRST (norm_before_gate=False)',
      summary:
        'Bypasses Conv1D and SSM recurrence completely, entering MambaRMSNormGated where y · SiLU(z) is applied FIRST before 8-group RMSNorm.',
      specs: [
        'Production width: 4,096 (64 × 64) · Miniature: 64 (16 × 4) × T rows',
        'Matches NemotronH MambaRMSNormGated(norm_before_gate=False)',
      ],
    },
    'xbc-pre': {
      id: 'xbc-pre',
      title: 'Pre-Conv xBC Stream (6,144 × T)',
      badge: `Sequence Tensor · T=${clampedT}`,
      summary:
        'Sequence activation stream feeding the depthwise causal Conv1D kernel. Unlike the fixed 4-tap rolling inference buffer beside it, this sequence activation strip scales with token count T.',
      specs: [
        'Production channels: 4,096(x) + 1,024(B) + 1,024(C) = 6,144 · Miniature: 128',
        'Each token column is pushed into the 4-slot rolling Conv1D buffer.',
      ],
    },
    conv1d: {
      id: 'conv1d',
      title: 'Causal Conv1D Rolling Buffer (k = 4)',
      badge: 'Always BF16 · ~1.08 MiB / 23 bays',
      summary:
        'Fixed 4-tap depthwise causal convolution buffer holding the last four token vectors (t−3, t−2, t−1, t; zero-padded for t−lag < 0) followed by SiLU. Supplies local ordering and NEVER quantizes to FP8.',
      specs: [
        `Active recurrence snapshot: t${absTokIdx} "${tokLabel}" (${
          zeroPads > 0
            ? `${zeroPads} zero-padded tap${zeroPads === 1 ? '' : 's'}`
            : sim.hasInitialState
              ? 'carried 4-tap Conv1D state + new token'
              : '0 zero-padded taps'
        })`,
        'Persistent Conv1D storage: 6,144 × 4 × 2 B = 48 KiB/bay (~1.08 MiB BF16 across 23 bays)',
        'Precision override: ALWAYS BF16 even on FP8 bays (NVIDIA §4.2 gold rim)',
      ],
    },
    'dt-branch': {
      id: 'dt-branch',
      title: 'Step / Decay Branch (Δt = softplus)',
      badge: '16 mini / 64 prod heads',
      summary:
        'Per-head discretization step Δt = softplus(Δt_raw + Δt_bias) > 0 controlling continuous decay A = −exp(A_log) < 0 via dA = exp(Δt · A) ∈ (0, 1).',
      specs: [
        'Uses numerically stable softplus: max(v, 0) + log1p(exp(−|v|))',
        'Balances retention (dA · Hₜ₋₁) against new write energy (Δt · x · B)',
      ],
    },
    'b-proj': {
      id: 'b-proj',
      title: 'Grouped Input Projection B (8 Groups)',
      badge: '8 Groups · Shared across heads',
      summary:
        'Post-Conv1D+SiLU write keys shared across the 8 Mamba-2 groups (2 miniature heads/group; 8 production heads/group). Outer-products with Δt · x to write into Hₜ.',
      specs: [
        'Production: 8 groups × 128 state dim = 1,024 · Miniature: 8 × 4 = 32',
        `Active head ${headTag} reads group G${activeGroup} columns.`,
      ],
    },
    'c-proj': {
      id: 'c-proj',
      title: 'Grouped Readout Projection C (8 Groups)',
      badge: '8 Groups · Shared across heads',
      summary:
        'Post-Conv1D+SiLU readout queries shared across the 8 Mamba-2 groups. Contracts the updated SSM state Hₜ along state dimension n to produce y_ssm.',
      specs: [
        'Production: 8 groups × 128 state dim = 1,024 · Miniature: 8 × 4 = 32',
        `Active head ${headTag} contracts Hₜ with C[G${activeGroup}].`,
      ],
    },
    'ssm-state-prev': {
      id: 'ssm-state-prev',
      title: `Prior SSM State Hₜ₋₁ (${headTag} · G${activeGroup})`,
      badge: sim.hasInitialState ? 'Carried FP32 Cache' : 'Fixed Size · FP32',
      summary:
        sim.hasInitialState
          ? 'Externally carried FP32 SSM state H_{t−1} from the prior cached decode step (strictly fixed size — zero token-history allocation).'
          : 'State matrix carried from the previous token (all zeros at t = 0). In production inference only the single active state buffer is kept and overwritten in place; H_history is browser-only playback history.',
      specs: [
        `Active miniature head: ${headTag} (4×4) at t${absTokIdx} "${tokLabel}" · Production head: 64×128 FP32`,
        'Decays elementwise by dA = exp(Δt · A) ∈ (0, 1) into Hₜ.',
      ],
    },
    'ssm-state-curr': {
      id: 'ssm-state-curr',
      title: `SSM State Hₜ (${headTag} · G${activeGroup})`,
      badge: 'Fixed Size · 46.0 MiB FP32 / 23 bays',
      summary:
        'Updated recurrent state Hₜ = dA · Hₜ₋₁ + Δt · x · B. Stepping t = 0 → T−1 or scrubbing T = 1 → 16 refreshes values inside this exact fixed-size matrix without allocating a single extra byte of SSM memory.',
      specs: [
        'Persistent SSM state: 64×64×128×4 B = 2.00 MiB/bay (46.0 MiB FP32 across 23 bays)',
        'Combined persistent Mamba state: 46.0 MiB SSM + ~1.08 MiB Conv1D ≈ 47.1 MiB fixed (T=1 → 1M)',
        'H_history is browser-only educational playback history (NOT stored in production inference).',
      ],
    },
    'ssm-state': {
      id: 'ssm-state',
      title: '64-Head Production SSM State Bank (FP32)',
      badge: '46.0 MiB SSM + 1.08 MiB Conv1D',
      summary:
        'All 64 production SSM heads (8 groups × 8 heads, each 64×128 FP32) remain strictly fixed in size from 1 to 1M tokens. Click any tile to inspect its mapped miniature head H00–H15 in Hₜ₋₁ ⇄ Hₜ.',
      specs: [
        'Persistent SSM storage: 2.00 MiB/layer × 23 = 46.0 MiB FP32 (NEVER FP8)',
        'Persistent Conv1D storage: 48 KiB/layer × 23 = ~1.08 MiB BF16 → ~47.1 MiB total fixed',
        'H_history is browser-only educational playback history.',
      ],
    },
    'ssm-readout': {
      id: 'ssm-readout',
      title: 'SSM State Readout + Skip (y = Hₜ·C + D·x)',
      badge: 'State Readout + Per-Head D',
      summary:
        'Contracts each head’s updated state Hₜ with its group’s C vector and adds the learned per-head skip connection D[h] · x.',
      specs: [
        'Equation: y[h, p, t] = Σₙ Hₜ[h, p, n] · C[g, n, t] + D[h] · x[h, p, t]',
        'Feeds directly into MambaRMSNormGated.',
      ],
    },
    'group-norm': {
      id: 'group-norm',
      title: 'MambaRMSNormGated (Gate FIRST → 8-Group Norm)',
      badge: 'norm_before_gate = False · ε = 1e−5',
      summary:
        'Applies multiplicative SiLU(z) gating FIRST (gated = y · SiLU(z)), then normalizes independently across 8 groups with ε = 1e−5, and scales by learned γ_gate.',
      specs: [
        'Production: 8 groups × 512 channels = 4,096 · Miniature: 8 groups × 8 channels = 64',
        'Verified against NVIDIA NemotronH MambaRMSNormGated(norm_before_gate=False).',
      ],
    },
    'out-proj': {
      id: 'out-proj',
      title: 'Mamba-2 Output Projection (W_out)',
      badge: computeIsFp8 ? 'FP8 Weights (prod)' : feederPreservedBf16 ? 'BF16 Preserved §4.2' : 'BF16 Weights (prod)',
      summary:
        'Projects the gated-and-normalized 4,096-channel vector back to the 2,688-channel residual stream width (miniature 64 → 8).',
      specs: [
        'Production matrix: 4,096 × 2,688 (~11.0M params) · Miniature: 64 × 8',
        isFeeder
          ? 'Bay 42 Feeder: preserved in BF16 alongside Bay 43 GQA (NVIDIA §4.2).'
          : 'Bay 01: FP8 production metadata in FP8 Selective mode (live sim stays Float32).',
      ],
    },
    'residual-out': {
      id: 'residual-out',
      title: 'Residual Stream Merge (+ 2,688)',
      badge: 'Activation · Emerald',
      summary:
        'Adds the Mamba-2 output delta back into the untouched emerald residual stream: Y = X + Mamba2(RMSNorm(X)).',
      specs: [
        'Hover or click any cell to see exact X + Mamba_out = Y values.',
      ],
    },
  };

  for (let h = 0; h < sim.nHeads; h++) {
    const g = mambaGroupForHead(h);
    const hId = formatMambaHeadId(h);
    partCatalog[`mamba-head-${h}`] = {
      id: `mamba-head-${h}`,
      title: `Mamba-2 Head ${hId} (Group G${g})`,
      badge: `A = ${sim.A[h].toFixed(2)} · D = ${sim.D[h].toFixed(2)}`,
      summary:
        `Miniature head ${hId} shares B/C group G${g} with ${formatMambaHeadId(g * 2)}–${formatMambaHeadId(g * 2 + 1)} (mirroring 8 production heads per group across 64 heads). Selecting ${hId} updates the live Hₜ₋₁ ⇄ Hₜ state sheets.`,
      specs: [
        `Continuous decay A[${hId}] = −exp(${sim.A_log[h].toFixed(2)}) = ${sim.A[h].toFixed(3)} · t${stateTokIdx} dA = ${sim.dA[h * clampedT + stateTokIdx].toFixed(3)}`,
        `Per-head skip D[${hId}] = ${sim.D[h].toFixed(3)} · t${stateTokIdx} Δt = ${sim.dt[h * clampedT + stateTokIdx].toFixed(3)}`,
        'Persistent storage: 46.0 MiB SSM (FP32) + ~1.08 MiB Conv1D (BF16) across 23 bays.',
      ],
    };
  }

  return partCatalog;
}

/**
 * Builds the Mamba-2 cutaway for Bay 01 (Ordinary) or Bay 42 (Feeder):
 *   - `viewAngle === 'three_quarter'`: true 3D multi-plane Spatial Mamba-2 architecture (`mambaSpatial.ts`)
 *   - `viewAngle === 'near_plan'` (default): preserved coplanar XZ drafting-table schematic
 */
export function buildMambaCutaway(opts: MambaCutawayOptions): CutawaySceneGeometry {
  const { bayId, precision, selectedPart, hoveredPart } = opts;
  const clampedT = opts.simOverride
    ? opts.simOverride.tokens
    : Math.max(1, Math.min(16, Math.round(opts.tokens)));
  const bay = NEMOTRON_SPEC.bays[bayId - 1];
  const isFeeder = bay.feeder;
  const isFp8Mode = precision === 'fp8_selective';
  // NVIDIA §4.2 production precision metadata: Bay 01 compute is FP8; Bay 42 Feeder stays BF16.
  // Conv1D is ALWAYS BF16 and SSM State is ALWAYS FP32 on every Mamba-2 bay!
  const computeIsFp8 = isFp8Mode && !bay.sensitive;
  const feederPreservedBf16 = isFp8Mode && bay.sensitive;

  const activePart = hoveredPart ?? selectedPart;
  const partHeadIdx = parseMambaHeadPartId(activePart);
  const cellHeadIdx = inferMambaCellHead(opts.activeCell);
  // Inspected cell head takes precedence so H_{t-1}/H_t sheets, labels, and threads stay synchronized
  const activeHead = Math.max(
    0,
    Math.min(
      MAMBA_SIM_CONFIG.nHeads - 1,
      cellHeadIdx ?? partHeadIdx ?? opts.activeHead ?? 0,
    ),
  );
  const activeGroup = mambaGroupForHead(activeHead);

  const sim = opts.simOverride ?? runMambaSim(clampedT);
  const cellTokIdx = inferMambaCellToken(opts.activeCell, clampedT);
  const explicitTokIdx =
    opts.stateTokenIndex !== undefined && opts.stateTokenIndex !== null
      ? Math.max(0, Math.min(clampedT - 1, Math.round(opts.stateTokenIndex)))
      : null;
  const stateTokIdx = cellTokIdx ?? explicitTokIdx ?? clampedT - 1;
  const explicitTokenPlayback = explicitTokIdx !== null;

  const atlas = buildMambaValueAtlas(sim, activeHead, stateTokIdx);
  const activeFormula: MambaCellFormula | null = opts.activeCell
    ? describeMambaCellFormula(sim, opts.activeCell)
    : null;
  const focusSet = buildMambaFocusSet(activeFormula);

  const partCatalog = buildMambaPartCatalog({
    bayId,
    clampedT,
    isFeeder,
    computeIsFp8,
    feederPreservedBf16,
    activeHead,
    activeGroup,
    stateTokIdx,
    sim,
  });

  if (opts.viewAngle === 'three_quarter') {
    return buildMambaSpatialScene({
      bayId,
      clampedT,
      isFeeder,
      isFp8Mode,
      computeIsFp8,
      feederPreservedBf16,
      activePart,
      partHeadIdx,
      activeHead,
      activeGroup,
      stateTokIdx,
      explicitTokenPlayback,
      activeCell: opts.activeCell ?? null,
      activeFormula,
      focusSet,
      stepperStep: opts.stepperStep,
      stepperActive: opts.stepperActive ?? false,
      stepperSubProgress: opts.stepperSubProgress,
      termProgress: opts.termProgress,
      reducedMotion: opts.reducedMotion ?? false,
      sim,
      atlas,
      partCatalog,
    });
  }

  const revealParams: MambaRevealParams | null =
    opts.stepperActive &&
    !opts.reducedMotion &&
    opts.stepperStep !== undefined &&
    opts.stepperStep !== null
      ? {
          stepperStep: opts.stepperStep,
          subProgress: opts.stepperSubProgress ?? 0.25,
          clampedT,
          activeHead,
          activeGroup,
          stateTokIdx,
          activeCell: opts.activeCell ?? null,
          sim,
        }
      : null;

  const focusedAccumulation = resolveMambaFocusedAccumulation({
    sim,
    activeCell: opts.activeCell ?? null,
    stepperStep: opts.stepperStep,
    stepperActive: opts.stepperActive ?? false,
    reducedMotion: opts.reducedMotion ?? false,
    termProgress: opts.termProgress,
  });

  const b = new MambaSheetBuilder(bayId, revealParams, focusSet.active);
  const hasCellTrace = focusSet.active;
  const hasFocus = activePart !== null || hasCellTrace;
  // Part-only head selection (no cell trace) highlights only that head's SSM recurrence pathway
  const partHeadFocus = partHeadIdx !== null && !hasCellTrace;

  const BASE_Y = 0.035;
  const OPERAND_LIFT_Y = 0.052;
  const LIFT_Y = 0.068;
  const SHEET_THICK = 0.070;
  const RIBBON_Y_BG = 0.022;
  const RIBBON_Y_ACT = 0.052;

  const sheetY = (hl: number): number =>
    hl === SHEET_HL_ACTIVE
      ? LIFT_Y
      : hl === SHEET_HL_OPERAND_A || hl === SHEET_HL_OPERAND_B
        ? OPERAND_LIFT_Y
        : BASE_Y;

  const stageHighlight = (
    tensor: MambaTensorKey,
    partId: string,
    onPartPathway: boolean,
    goldWhenFp8Mode: boolean,
  ): number => {
    const fallback =
      activePart === partId
        ? SHEET_HL_ACTIVE
        : partHeadFocus && onPartPathway
          ? SHEET_HL_PATHWAY
          : isFp8Mode && goldWhenFp8Mode && !hasFocus
            ? SHEET_HL_BF16_GOLD
            : hasFocus && !onPartPathway
              ? SHEET_HL_DIMMED
              : SHEET_HL_NORMAL;
    return resolveSheetHighlight(focusSet, { tensor, isAggregate: true }, fallback);
  };

  // --- Spatial Anchors (XZ drafting table, top -Z to bottom +Z) ---
  const resX = 6.45;
  const zRow0 = -5.85; // Input (2,688) & RMSNorm
  const zRow1 = -4.60; // Fused W_in [208×8]
  const zXbcPre = -3.32; // Pre-Conv xBC [128, T]
  const zConvTop = -2.46; // 4 fixed rolling Conv1D taps
  const convTapPitch = 0.175;
  const convTapDepth = 0.125;
  const convTapZ = (slot: number) => zConvTop + slot * convTapPitch;
  const zConvBottom = convTapZ(3) + convTapDepth * 0.5;
  const zSplit = -0.25; // Post-SiLU x (16 heads in 8 groups) + B (8 grps) + C (8 grps)
  const zSsm = 1.65; // Fixed-size H_{t-1} ⇄ H_t + 64-Head Production SSM Bank
  const zY = 3.48; // Readout y = H_t·C + D·x
  const zNorm = 4.55; // Gate → 8-Group RMSNorm (y · SiLU(z))
  const zOut = 5.75; // W_out [4,096→2,688] & + Residual

  // Token-length scaling for sequence sheets (clearly grows T=1 -> T=16 while SSM state stays 100% FIXED!)
  const tFrac = clampedT / 16;
  const seqDepthSmall = 0.30 + tFrac * 0.36; // 0.32..0.66
  const seqDepthMed = 0.40 + tFrac * 0.52; // 0.43..0.92
  const seqDepthTall = 0.52 + tFrac * 0.88; // 0.575..1.40

  // ==========================================================================
  // 1. RIGHT RESIDUAL HIGHWAY: Input (X) ---> + Residual (Y)
  // ==========================================================================
  const resInHl = resolveSheetHighlight(
    focusSet,
    { tensor: 'input', isAggregate: true },
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
    ez: 1.04,
    gridCols: sim.d,
    gridRows: clampedT,
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
    { tensor: 'residual', isAggregate: true },
    activePart === 'residual-out'
      ? SHEET_HL_ACTIVE
      : activePart === 'out-proj'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: resX,
    cy: sheetY(resOutHl),
    cz: zOut,
    ex: 0.72,
    ey: SHEET_THICK,
    ez: 1.04,
    gridCols: sim.d,
    gridRows: clampedT,
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

  const resRibbonCol = hasCellTrace
    ? resOutHl === SHEET_HL_ACTIVE || resInHl === SHEET_HL_ACTIVE
      ? RIBBON_RESIDUAL
      : RIBBON_DIMMED
    : activePart === 'residual-in' || activePart === 'residual-out'
      ? RIBBON_RESIDUAL_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL;
  b.flow.addPolylineXZ(
    [
      { x: resX, z: zRow0 + 0.54 },
      { x: resX, z: zOut - 0.54 },
    ],
    RIBBON_Y_BG,
    0.18,
    resRibbonCol,
    true,
    0.34,
    0.36,
  );

  // ==========================================================================
  // 2. ROW 0: Pre-Layer RMSNorm (eps = 1e-5)
  // ==========================================================================
  const rmsX = 0.55;
  const rmsW = 2.40;
  const rmsD = 0.48;
  const rmsHl = stageHighlight(
    'rmsnorm',
    'rmsnorm',
    activePart === 'in-proj',
    feederPreservedBf16,
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
    role: SHEET_ROLE_NORM,
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: rmsHl,
    seed: 3.7,
    partId: 'rmsnorm',
    tensorKey: 'rmsnorm',
    atlasSlice: atlas.rmsnorm,
    registryKey: 'rmsnorm',
    cellFamily: CELL_MOE_AMBER,
  });

  const inToRmsActive = !hasCellTrace && (rmsHl === SHEET_HL_ACTIVE || activePart === 'residual-in');
  b.flow.addPolylineXZ(
    [
      { x: resX - 0.38, z: zRow0 },
      { x: rmsX + rmsW * 0.5 + 0.04, z: zRow0 },
    ],
    inToRmsActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.13,
    inToRmsActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_RESIDUAL,
    true,
  );

  // ==========================================================================
  // 3. ROW 1: Fused Input Projection W_in [208 × 8] (prod 10,304 × 2,688)
  // ==========================================================================
  const winX = -0.35;
  const winW = 6.20;
  const winD = 0.78;
  const winHl = stageHighlight('w_in', 'in-proj', false, feederPreservedBf16);
  b.addSheet({
    cx: winX,
    cy: sheetY(winHl),
    cz: zRow1,
    ex: winW,
    ey: SHEET_THICK,
    ez: winD,
    gridCols: sim.inProjDim,
    gridRows: sim.d,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: winHl,
    seed: 5.1 + bayId * 0.3,
    partId: 'in-proj',
    tensorKey: 'w_in',
    atlasSlice: atlas.wIn,
    registryKey: 'w_in',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  b.flow.addCubicXZ(
    { x: rmsX, z: zRow0 + rmsD * 0.5 + 0.02 },
    { x: rmsX, z: -4.98 },
    { x: winX + 0.45, z: -4.98 },
    { x: winX + 0.45, z: zRow1 - winD * 0.5 - 0.04 },
    winHl === SHEET_HL_ACTIVE && !hasCellTrace ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.11,
    winHl === SHEET_HL_ACTIVE && !hasCellTrace ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET,
    true,
  );

  // ==========================================================================
  // 4. THREE PARALLEL BRANCHES: Left Gate z | Center Conv1D | Right Δt
  // ==========================================================================
  const branchColor = (active: boolean): FlowRibbonColor =>
    active && !hasCellTrace ? RIBBON_PATHWAY_SOFT : hasFocus ? RIBBON_DIMMED : RIBBON_QUIET;

  // --- 4A. Left Bypass Gate Branch: z [64, T] ---
  const gateX = -5.45;
  const gateZ = -0.20;
  const gateW = 1.68;
  const gateD = seqDepthTall;
  const gateOnPath = activePart === 'gate-branch' || activePart === 'group-norm';
  const gateHl = stageHighlight('z', 'gate-branch', gateOnPath, feederPreservedBf16);
  b.addSheet({
    cx: gateX,
    cy: sheetY(gateHl),
    cz: gateZ,
    ex: gateW,
    ey: SHEET_THICK,
    ez: gateD,
    gridCols: sim.dInner,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: gateHl,
    seed: 8.3 + clampedT * 0.15,
    partId: 'gate-branch',
    tensorKey: 'z',
    atlasSlice: atlas.z,
    registryKey: 'z',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // W_in (left third) -> Gate z
  b.flow.addCubicXZ(
    { x: winX - 2.15, z: zRow1 + winD * 0.5 + 0.02 },
    { x: winX - 2.15, z: -3.20 },
    { x: gateX, z: -3.20 },
    { x: gateX, z: gateZ - gateD * 0.5 - 0.05 },
    gateOnPath && !hasCellTrace ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.10,
    branchColor(gateOnPath),
    true,
  );

  // --- 4B. Center Stream: Pre-Conv xBC [128, T] -> Fixed 4-Tap Causal Conv1D (ALWAYS BF16!) ---
  const convX = -0.55;
  const convW = 4.40;
  const xbcPreHl = stageHighlight(
    'xbc_pre',
    'xbc-pre',
    activePart === 'conv1d' || activePart === 'in-proj',
    feederPreservedBf16,
  );
  b.addSheet({
    cx: convX,
    cy: sheetY(xbcPreHl),
    cz: zXbcPre,
    ex: convW,
    ey: SHEET_THICK,
    ez: seqDepthSmall,
    gridCols: sim.convDim,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: xbcPreHl,
    seed: 10.2 + clampedT * 0.12,
    partId: 'xbc-pre',
    tensorKey: 'xbc_pre',
    atlasSlice: atlas.xbcPre,
    registryKey: 'xbc_pre',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // W_in (center) -> xBC_pre
  b.flow.addPolylineXZ(
    [
      { x: convX, z: zRow1 + winD * 0.5 + 0.02 },
      { x: convX, z: zXbcPre - seqDepthSmall * 0.5 - 0.04 },
    ],
    !hasCellTrace && (activePart === 'xbc-pre' || activePart === 'conv1d') ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.10,
    branchColor(activePart === 'xbc-pre' || activePart === 'conv1d'),
    true,
  );

  // Fixed 4-Tap Rolling Causal Conv1D Buffer (ALWAYS BF16 — gold rim in FP8 Selective!)
  const convFallbackHl =
    activePart === 'conv1d'
      ? SHEET_HL_ACTIVE
      : isFp8Mode && !hasFocus
        ? SHEET_HL_BF16_GOLD
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;
  for (let slot = 0; slot < sim.convKernel; slot++) {
    const cz = convTapZ(slot);
    const tapHl = resolveSheetHighlight(
      focusSet,
      { tensor: 'conv_buf', tapIndex: slot, tokenIndex: stateTokIdx, isAggregate: false },
      convFallbackHl,
    );
    b.addSheet({
      cx: convX,
      cy: sheetY(tapHl),
      cz,
      ex: convW,
      ey: 0.062,
      ez: convTapDepth,
      gridCols: sim.convDim,
      gridRows: 1,
      role: SHEET_ROLE_KV_K,
      fp8Flag: 0, // NEVER switches to FP8 (NVIDIA §4.2)
      highlight: tapHl,
      seed: 12.0 + slot * 0.7 + clampedT * 0.35,
      partId: 'conv1d',
      tensorKey: 'conv_buf',
      tokenIndex: slot,
      tapIndex: slot,
      stateTokenIndex: stateTokIdx,
      atlasSlice: atlas.convBufTaps[slot],
      registryKey: `conv_buf:${slot}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });
  }

  // xBC_pre -> Conv1D taps
  b.flow.addPolylineXZ(
    [
      { x: convX, z: zXbcPre + seqDepthSmall * 0.5 + 0.02 },
      { x: convX, z: zConvTop - convTapDepth * 0.5 - 0.04 },
    ],
    !hasCellTrace && activePart === 'conv1d' ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.10,
    branchColor(activePart === 'conv1d'),
    true,
  );

  // --- 4C. Right-Inner Branch: Δt = softplus(dt_raw + dt_bias) [16, T] ---
  const dtX = 4.25;
  const dtZ = -2.25;
  const dtW = 1.62;
  const dtD = seqDepthMed;
  const dtOnPath = activePart === 'dt-branch' || activePart === 'ssm-state-curr' || partHeadFocus;
  const dtHl = stageHighlight('dt', 'dt-branch', dtOnPath, feederPreservedBf16);
  b.addSheet({
    cx: dtX,
    cy: sheetY(dtHl),
    cz: dtZ,
    ex: dtW,
    ey: SHEET_THICK,
    ez: dtD,
    gridCols: sim.nHeads,
    gridRows: clampedT,
    role: SHEET_ROLE_SOFTMAX, // positive softplus intensity
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: dtHl,
    seed: 15.1 + clampedT * 0.2,
    partId: 'dt-branch',
    tensorKey: 'dt',
    atlasSlice: atlas.dt,
    registryKey: 'dt',
    cellFamily: CELL_MOE_AMBER,
  });

  // W_in (right) -> Δt
  b.flow.addCubicXZ(
    { x: winX + 2.45, z: zRow1 + winD * 0.5 + 0.02 },
    { x: winX + 2.45, z: -3.28 },
    { x: dtX, z: -3.28 },
    { x: dtX, z: dtZ - dtD * 0.5 - 0.05 },
    dtOnPath && !hasCellTrace ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.095,
    branchColor(dtOnPath),
    true,
  );

  // ==========================================================================
  // 5. ROW 3: Post-SiLU Split -> x (16 Selectable Heads in 8 Groups) + B + C
  // ==========================================================================
  const headWidth = 0.20;
  const headInnerPitch = 0.235;
  const groupGap = 0.085;
  const groupSpan = headInnerPitch + groupGap;
  const xStart = -3.86;
  const headCenterX = (h: number): number => {
    const g = Math.floor(h / 2);
    const local = h % 2;
    return xStart + g * (2 * headInnerPitch + groupGap) + local * headInnerPitch;
  };
  const xBankCenterX = (headCenterX(0) + headCenterX(15)) * 0.5;
  const splitDepth = seqDepthMed;

  for (let h = 0; h < sim.nHeads; h++) {
    const g = mambaGroupForHead(h);
    const hx = headCenterX(h);
    let fallbackHl = SHEET_HL_NORMAL;
    if (h === activeHead && (partHeadFocus || activePart === 'ssm-state-curr' || activePart === 'ssm-state-prev')) {
      fallbackHl = SHEET_HL_ACTIVE;
    } else if (hasFocus) {
      fallbackHl = SHEET_HL_DIMMED;
    } else if (feederPreservedBf16) {
      fallbackHl = SHEET_HL_BF16_GOLD;
    }
    const hl = resolveSheetHighlight(
      focusSet,
      { tensor: 'x', headIndex: h, groupIndex: g, isAggregate: false },
      fallbackHl,
    );

    b.addSheet({
      cx: hx,
      cy: sheetY(hl),
      cz: zSplit,
      ex: headWidth,
      ey: SHEET_THICK,
      ez: splitDepth,
      gridCols: sim.dHead,
      gridRows: clampedT,
      role: SHEET_ROLE_ACTIVATION,
      fp8Flag: 0,
      highlight: hl,
      seed: 19.0 + h * 0.57 + clampedT * 0.11,
      partId: `mamba-head-${h}`,
      headIndex: h,
      groupIndex: g,
      tensorKey: 'x',
      atlasSlice: atlas.xHeads[h],
      registryKey: `x:${h}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });
  }

  // Quiet 8-group pairing brackets behind x heads (G0..G7)
  for (let g = 0; g < sim.nGroups; g++) {
    const x0 = headCenterX(g * 2) - headWidth * 0.5;
    const x1 = headCenterX(g * 2 + 1) + headWidth * 0.5;
    const bz = zSplit - splitDepth * 0.5 - 0.11;
    const isActG = g === activeGroup && (partHeadFocus || hasCellTrace);
    b.addGuideLine(
      x0, 0.02, bz,
      x1, 0.02, bz,
      isActG ? 1.5 : 0.85,
      isActG ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
      isActG ? 0.92 : 0.52,
    );
  }
  void groupSpan;

  // B (8 groups × d_state=4) and C (8 groups × d_state=4)
  const bX = 1.62;
  const cX = 3.38;
  const bcW = 1.46;
  const bOnPath = activePart === 'b-proj' || activePart === 'ssm-state-curr' || partHeadFocus;
  const bHl = stageHighlight('b', 'b-proj', bOnPath, feederPreservedBf16);
  b.addSheet({
    cx: bX,
    cy: sheetY(bHl),
    cz: zSplit,
    ex: bcW,
    ey: SHEET_THICK,
    ez: splitDepth,
    gridCols: sim.nGroups * sim.dState,
    gridRows: clampedT,
    role: SHEET_ROLE_KV_K,
    fp8Flag: 0,
    highlight: bHl,
    seed: 24.3 + clampedT * 0.1,
    partId: 'b-proj',
    tensorKey: 'b',
    atlasSlice: atlas.bGroups,
    registryKey: 'b',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  const cOnPath = activePart === 'c-proj' || activePart === 'ssm-readout' || partHeadFocus;
  const cHl = stageHighlight('c', 'c-proj', cOnPath, feederPreservedBf16);
  b.addSheet({
    cx: cX,
    cy: sheetY(cHl),
    cz: zSplit,
    ex: bcW,
    ey: SHEET_THICK,
    ez: splitDepth,
    gridCols: sim.nGroups * sim.dState,
    gridRows: clampedT,
    role: SHEET_ROLE_KV_V,
    fp8Flag: 0,
    highlight: cHl,
    seed: 25.9 + clampedT * 0.1,
    partId: 'c-proj',
    tensorKey: 'c',
    atlasSlice: atlas.cGroups,
    registryKey: 'c',
    cellFamily: CELL_SSM_CYAN,
  });

  // Conv1D + SiLU -> x, B, C fan-out ribbons
  const xTargetX = partHeadFocus || hasCellTrace ? headCenterX(activeHead) : xBankCenterX;
  b.flow.addCubicXZ(
    { x: convX - 1.05, z: zConvBottom + 0.02 },
    { x: convX - 1.05, z: -1.16 },
    { x: xTargetX, z: -1.16 },
    { x: xTargetX, z: zSplit - splitDepth * 0.5 - 0.05 },
    !hasCellTrace && partHeadFocus ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.095,
    branchColor(activePart === 'conv1d' || partHeadFocus),
    true,
  );
  b.flow.addCubicXZ(
    { x: convX + 0.70, z: zConvBottom + 0.02 },
    { x: convX + 0.70, z: -1.18 },
    { x: bX, z: -1.18 },
    { x: bX, z: zSplit - splitDepth * 0.5 - 0.05 },
    !hasCellTrace && bOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.085,
    branchColor(bOnPath),
    true,
  );
  b.flow.addCubicXZ(
    { x: convX + 1.55, z: zConvBottom + 0.02 },
    { x: convX + 1.55, z: -1.24 },
    { x: cX, z: -1.24 },
    { x: cX, z: zSplit - splitDepth * 0.5 - 0.05 },
    !hasCellTrace && cOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.085,
    branchColor(cOnPath),
    true,
  );

  // ==========================================================================
  // 6. ROW 4 CENTERPIECE: Fixed-Size SSM State (H_{t-1} ⇄ H_t + 64-Head Bank)
  // ==========================================================================
  const hPrevX = -2.45;
  const hCurrX = -0.35;
  const hSize = 1.38; // STRICTLY CONSTANT from T = 1 to T = 16!

  const hPrevFallback =
    activePart === 'ssm-state-prev'
      ? SHEET_HL_ACTIVE
      : partHeadFocus || activePart === 'ssm-state-curr'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;
  const hPrevHl = resolveSheetHighlight(
    focusSet,
    { tensor: 'h_prev', headIndex: activeHead, groupIndex: activeGroup, tokenIndex: stateTokIdx, isAggregate: false },
    hPrevFallback,
  );
  b.addSheet({
    cx: hPrevX,
    cy: sheetY(hPrevHl),
    cz: zSsm,
    ex: hSize,
    ey: SHEET_THICK,
    ez: hSize,
    gridCols: sim.dState,
    gridRows: sim.dHead,
    role: SHEET_ROLE_SCORES,
    fp8Flag: 0, // ALWAYS FP32
    highlight: hPrevHl,
    seed: 31.1 + activeHead * 0.4 + clampedT * 0.6,
    partId: 'ssm-state-prev',
    headIndex: activeHead,
    groupIndex: activeGroup,
    tensorKey: 'h_prev',
    tokenIndex: stateTokIdx,
    atlasSlice: atlas.hPrev,
    registryKey: 'h_prev',
    cellFamily: CELL_SSM_CYAN,
  });

  const hCurrFallback =
    activePart === 'ssm-state-curr'
      ? SHEET_HL_ACTIVE
      : partHeadFocus || activePart === 'ssm-state' || activePart === 'ssm-readout'
        ? SHEET_HL_PATHWAY
        : hasFocus
          ? SHEET_HL_DIMMED
          : SHEET_HL_NORMAL;
  const hCurrHl = resolveSheetHighlight(
    focusSet,
    { tensor: 'h_curr', headIndex: activeHead, groupIndex: activeGroup, tokenIndex: stateTokIdx, isAggregate: false },
    hCurrFallback,
  );
  b.addSheet({
    cx: hCurrX,
    cy: sheetY(hCurrHl) === BASE_Y && hCurrHl === SHEET_HL_PATHWAY ? LIFT_Y : sheetY(hCurrHl),
    cz: zSsm,
    ex: hSize,
    ey: SHEET_THICK,
    ez: hSize,
    gridCols: sim.dState,
    gridRows: sim.dHead,
    role: SHEET_ROLE_KV_V,
    fp8Flag: 0, // ALWAYS FP32
    highlight: hCurrHl,
    seed: 33.7 + activeHead * 0.4 + clampedT * 0.9,
    partId: 'ssm-state-curr',
    headIndex: activeHead,
    groupIndex: activeGroup,
    tensorKey: 'h_curr',
    tokenIndex: stateTokIdx,
    atlasSlice: atlas.hCurr,
    registryKey: 'h_curr',
    cellFamily: CELL_SSM_CYAN,
  });

  // Forward recurrence arrow: H_{t-1} --(dA·H_{t-1} + Δt·x·B)--> H_t
  b.flow.addPolylineXZ(
    [
      { x: hPrevX + hSize * 0.5 + 0.04, z: zSsm },
      { x: hCurrX - hSize * 0.5 - 0.05, z: zSsm },
    ],
    !hasCellTrace && (!hasFocus || partHeadFocus || activePart === 'ssm-state-curr') ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.115,
    !hasCellTrace && (!hasFocus || partHeadFocus || activePart === 'ssm-state-curr') ? RIBBON_ACTIVE : RIBBON_DIMMED,
    true,
  );

  // Smooth in-place recurrence feedback loop ribbon: H_t loops beneath back to H_{t-1}
  b.flow.addCubicXZ(
    { x: hCurrX, z: zSsm + hSize * 0.5 + 0.03 },
    { x: hCurrX - 0.15, z: zSsm + hSize * 0.5 + 0.58 },
    { x: hPrevX + 0.15, z: zSsm + hSize * 0.5 + 0.58 },
    { x: hPrevX, z: zSsm + hSize * 0.5 + 0.06 },
    RIBBON_Y_BG + 0.008,
    0.095,
    hasCellTrace
      ? hCurrHl !== SHEET_HL_DIMMED || hPrevHl !== SHEET_HL_DIMMED
        ? RIBBON_SSM_LOOP
        : RIBBON_DIMMED
      : hasFocus && !partHeadFocus && activePart !== 'ssm-state-curr' && activePart !== 'ssm-state-prev'
        ? RIBBON_DIMMED
        : RIBBON_SSM_LOOP,
    true,
    24,
  );

  // Feed ribbons into H_t from x (active head), B, and Δt
  b.flow.addCubicXZ(
    { x: xTargetX, z: zSplit + splitDepth * 0.5 + 0.02 },
    { x: xTargetX, z: 0.58 },
    { x: hCurrX - 0.28, z: 0.58 },
    { x: hCurrX - 0.28, z: zSsm - hSize * 0.5 - 0.05 },
    !hasCellTrace && partHeadFocus ? RIBBON_Y_ACT : RIBBON_Y_BG,
    partHeadFocus && !hasCellTrace ? 0.11 : 0.085,
    partHeadFocus && !hasCellTrace ? RIBBON_ACTIVE : branchColor(false),
    true,
  );
  b.flow.addCubicXZ(
    { x: bX, z: zSplit + splitDepth * 0.5 + 0.02 },
    { x: bX, z: 0.56 },
    { x: hCurrX + 0.18, z: 0.56 },
    { x: hCurrX + 0.18, z: zSsm - hSize * 0.5 - 0.05 },
    !hasCellTrace && bOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.085,
    branchColor(bOnPath),
    true,
  );
  b.flow.addRoundedOrthoXZ(
    [
      { x: dtX, z: dtZ + dtD * 0.5 + 0.03 },
      { x: dtX, z: -1.06 },
      { x: 4.95, z: -1.06 },
      { x: 4.95, z: 0.62 },
      { x: hCurrX + 0.48, z: 0.62 },
      { x: hCurrX + 0.48, z: zSsm - hSize * 0.5 - 0.05 },
    ],
    0.20,
    !hasCellTrace && dtOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.075,
    branchColor(dtOnPath),
    true,
  );

  // 64-Head Fixed Production SSM Bank (8 groups × 8 heads = 64 micro-tiles, ALWAYS FP32, FIXED SIZE)
  // Clicking any tile selects its mapped miniature head H00..H15; preserves 64 'ssm-state' contract!
  const bankCenterX = 2.48;
  const bankCols = 8; // 8 B/C groups
  const bankRows = 8; // 8 production heads per group
  const bankPitchX = 0.225;
  const bankPitchZ = 0.175;
  const bankStartX = bankCenterX - ((bankCols - 1) * bankPitchX) * 0.5;
  const bankStartZ = zSsm - ((bankRows - 1) * bankPitchZ) * 0.5;

  for (let g = 0; g < bankCols; g++) {
    for (let r = 0; r < bankRows; r++) {
      const tileIdx = g * 8 + r;
      const miniHead = g * 2 + Math.floor(r / 4);
      let tileHl: number;
      if (hasCellTrace) {
        const matchesStateHead =
          resolveSheetFocusRole(focusSet, {
            tensor: 'h_curr',
            headIndex: miniHead,
            groupIndex: g,
            tokenIndex: stateTokIdx,
            isAggregate: false,
          }) !== null ||
          resolveSheetFocusRole(focusSet, {
            tensor: 'h_prev',
            headIndex: miniHead,
            groupIndex: g,
            tokenIndex: stateTokIdx,
            isAggregate: false,
          }) !== null;
        tileHl = matchesStateHead ? SHEET_HL_PATHWAY : SHEET_HL_DIMMED;
      } else {
        const isTileActive = activePart === 'ssm-state' || (partHeadFocus && miniHead === activeHead);
        tileHl = isTileActive
          ? SHEET_HL_ACTIVE
          : hasFocus
            ? SHEET_HL_DIMMED
            : SHEET_HL_NORMAL;
      }

      b.addSheet({
        cx: bankStartX + g * bankPitchX,
        cy: tileHl === SHEET_HL_ACTIVE ? LIFT_Y : BASE_Y,
        cz: bankStartZ + r * bankPitchZ,
        ex: 0.18,
        ey: 0.055,
        ez: 0.135,
        gridCols: 1,
        gridRows: 1,
        role: SHEET_ROLE_KV_V,
        fp8Flag: 0, // ALWAYS FP32!
        highlight: tileHl,
        // Live token-dependent phase updates IN PLACE as T or temporal playback t steps while geometry stays fixed
        seed: clampedT * 1.73 + stateTokIdx * 0.41 + tileIdx * 0.29 + bayId * 0.11,
        partId: 'ssm-state',
        headIndex: miniHead,
        groupIndex: g,
        atlasSlice: atlas.ssmBankTiles[tileIdx],
        cellFamily: CELL_SSM_CYAN,
      });
    }
  }

  // Subtle dashed guide connector linking active H_t to the 64-head bank
  b.addGuideLine(
    hCurrX + hSize * 0.5 + 0.08, 0.03, zSsm,
    bankStartX - 0.14, 0.03, zSsm,
    0.9,
    LINE_STYLE_GUIDE,
    0.55,
  );

  // ==========================================================================
  // 7. ROW 5: SSM Readout y = Σ_n H_t·C + D·x  ->  ROW 6: Gate→8-Group RMSNorm
  // ==========================================================================
  const yX = -0.35;
  const yW = 4.40;
  const yD = 0.44;
  const yOnPath = activePart === 'ssm-readout' || activePart === 'group-norm' || partHeadFocus;
  const yHl = stageHighlight('y', 'ssm-readout', yOnPath, feederPreservedBf16);
  b.addSheet({
    cx: yX,
    cy: sheetY(yHl),
    cz: zY,
    ex: yW,
    ey: SHEET_THICK,
    ez: yD,
    gridCols: sim.dInner,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: yHl,
    seed: 41.2 + clampedT * 0.1,
    partId: 'ssm-readout',
    tensorKey: 'y',
    atlasSlice: atlas.y,
    registryKey: 'y',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // H_t -> y readout ribbon
  b.flow.addPolylineXZ(
    [
      { x: hCurrX, z: zSsm + hSize * 0.5 + 0.04 },
      { x: yX, z: zY - yD * 0.5 - 0.04 },
    ],
    !hasCellTrace && yOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.105,
    branchColor(yOnPath),
    true,
  );

  // C -> y readout ribbon routing cleanly around the right side of the 64-head bank
  b.flow.addRoundedOrthoXZ(
    [
      { x: cX, z: zSplit + splitDepth * 0.5 + 0.03 },
      { x: cX, z: 0.42 },
      { x: 4.18, z: 0.42 },
      { x: 4.18, z: 2.78 },
      { x: yX + 1.35, z: 2.78 },
      { x: yX + 1.35, z: zY - yD * 0.5 - 0.04 },
    ],
    0.22,
    !hasCellTrace && cOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.08,
    branchColor(cOnPath),
    true,
  );

  // Row 6: Gate → 8-Group RMSNorm (norm_before_gate = False: y * SiLU(z) FIRST!)
  const normX = -0.85;
  const normW = 4.80;
  const normD = 0.48;
  const normOnPath = activePart === 'group-norm' || activePart === 'out-proj';
  const normHl = stageHighlight('y_norm', 'group-norm', normOnPath, feederPreservedBf16);
  b.addSheet({
    cx: normX,
    cy: sheetY(normHl),
    cz: zNorm,
    ex: normW,
    ey: SHEET_THICK,
    ez: normD,
    gridCols: sim.dInner,
    gridRows: clampedT,
    role: SHEET_ROLE_NORM,
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: normHl,
    seed: 44.6 + clampedT * 0.1,
    partId: 'group-norm',
    tensorKey: 'y_norm',
    atlasSlice: atlas.yNorm,
    registryKey: 'y_norm',
    cellFamily: CELL_MOE_AMBER,
  });

  // y -> Gate→8-Group Norm
  b.flow.addCubicXZ(
    { x: yX, z: zY + yD * 0.5 + 0.02 },
    { x: yX, z: 4.02 },
    { x: normX + 0.35, z: 4.02 },
    { x: normX + 0.35, z: zNorm - normD * 0.5 - 0.04 },
    !hasCellTrace && normOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.105,
    branchColor(normOnPath),
    true,
  );

  // Left bypass Gate z -> Gate→8-Group Norm (enters from left side!)
  b.flow.addRoundedOrthoXZ(
    [
      { x: gateX, z: gateZ + gateD * 0.5 + 0.03 },
      { x: gateX, z: zNorm },
      { x: normX - normW * 0.5 - 0.05, z: zNorm },
    ],
    0.34,
    !hasCellTrace && gateOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.105,
    branchColor(gateOnPath),
    true,
  );

  // ==========================================================================
  // 8. ROW 7: Output Projection W_out [4,096 → 2,688] -> + Residual
  // ==========================================================================
  const woutX = 1.60;
  const woutW = 3.60;
  const woutD = 0.78;
  const woutOnPath = activePart === 'out-proj' || activePart === 'residual-out';
  const woutHl = stageHighlight('w_out', 'out-proj', woutOnPath, feederPreservedBf16);
  b.addSheet({
    cx: woutX,
    cy: sheetY(woutHl),
    cz: zOut,
    ex: woutW,
    ey: SHEET_THICK,
    ez: woutD,
    gridCols: sim.dInner,
    gridRows: sim.d,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: computeIsFp8 ? 1 : 0,
    highlight: woutHl,
    seed: 51.3 + bayId * 0.4,
    partId: 'out-proj',
    tensorKey: 'w_out',
    atlasSlice: atlas.wOut,
    registryKey: 'w_out',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Gate→8-Group Norm -> W_out
  b.flow.addCubicXZ(
    { x: normX + 0.90, z: zNorm + normD * 0.5 + 0.02 },
    { x: normX + 0.90, z: 5.10 },
    { x: woutX, z: 5.10 },
    { x: woutX, z: zOut - woutD * 0.5 - 0.04 },
    !hasCellTrace && woutOnPath ? RIBBON_Y_ACT : RIBBON_Y_BG,
    0.11,
    branchColor(woutOnPath),
    true,
  );

  // W_out -> + Residual
  const o2rActive = !hasCellTrace && (activePart === 'out-proj' || activePart === 'residual-out');
  b.flow.addPolylineXZ(
    [
      { x: woutX + woutW * 0.5 + 0.04, z: zOut },
      { x: resX - 0.40, z: zOut },
    ],
    o2rActive ? RIBBON_Y_ACT : RIBBON_Y_BG,
    o2rActive ? 0.135 : 0.105,
    o2rActive ? RIBBON_ACTIVE : hasFocus ? RIBBON_DIMMED : RIBBON_RESIDUAL,
    true,
  );

  // ==========================================================================
  // 9. LIVE CELL-LEVEL HIGHLIGHTS & UPSTREAM DEPENDENCY THREADS
  // ==========================================================================
  let popoverTargetWorld: [number, number, number] | null = null;
  const popoverSources: {
    readonly world: readonly [number, number, number];
    readonly role: 'indigo' | 'teal';
  }[] = [];

  if (activeFormula) {
    const targetRect = resolveMambaSpanRect(b.sheetRegistry, activeFormula.targetSpan, activeHead, stateTokIdx);
    if (targetRect) {
      popoverTargetWorld = [targetRect.cx, targetRect.topY + 0.01, targetRect.cz];
      b.flow.addCellHighlightXZ(
        targetRect.cx,
        targetRect.cz,
        targetRect.ex,
        targetRect.ez,
        targetRect.topY + 0.009,
        CELL_TARGET_FILL,
        CELL_TARGET_FRAME,
        0.026,
      );
      for (const upSpan of activeFormula.upstreamSpans) {
        const upRect = resolveMambaSpanRect(b.sheetRegistry, upSpan, activeHead, stateTokIdx);
        if (!upRect) continue;
        popoverSources.push({
          world: [upRect.cx, upRect.topY + 0.01, upRect.cz],
          role: upSpan.role === 'operand_a' ? 'indigo' : 'teal',
        });
        drawOperandDependencyXZ(b.flow, upRect, targetRect, upSpan.role === 'operand_a');
      }
    }
  }

  const cellPopover = buildMambaCellPopover(
    activeFormula,
    popoverTargetWorld,
    popoverSources,
    focusedAccumulation,
  );
  const activeCellGrids = buildMambaActiveCellGrids(
    b.sheetRegistry,
    sim,
    clampedT,
    activeHead,
    stateTokIdx,
    activeFormula,
    revealParams,
    focusedAccumulation,
  );

  // ==========================================================================
  // 10. EXTERIOR PROGRESSIVE LABELS (strictly outside sheet faces, zero clutter)
  // ==========================================================================
  const headTag = formatMambaHeadId(activeHead);
  const histTokTag =
    stateTokIdx < clampedT - 1 || explicitTokenPlayback ? ` · t${stateTokIdx}` : '';
  b.landmarks.push(
    { id: 'm-res-in', label: 'Input (2,688)', world: [resX, 0.08, zRow0 - 0.66], partId: 'residual-in', align: 'top', tier: 'primary' },
    { id: 'm-rmsnorm', label: 'RMSNorm', world: [rmsX, 0.08, zRow0 - 0.36], partId: 'rmsnorm', align: 'top', tier: 'primary', tone: feederPreservedBf16 ? 'gold' : 'default' },
    {
      id: 'm-in-proj',
      label: computeIsFp8 ? 'W_in · FP8' : 'W_in (10,304)',
      world: [winX - winW * 0.5 - 0.22, 0.08, zRow1],
      partId: 'in-proj',
      align: 'left',
      tier: 'primary',
      tone: feederPreservedBf16 ? 'gold' : 'default',
    },
    { id: 'm-gate', label: `Gate z (T=${clampedT})`, world: [gateX, 0.08, gateZ - gateD * 0.5 - 0.18], partId: 'gate-branch', align: 'top', tier: 'primary' },
    {
      id: 'm-conv1d',
      label: isFp8Mode ? `Conv1D (k=4) · BF16${histTokTag}` : `Conv1D (k=4)${histTokTag}`,
      world: [convX + convW * 0.5 + 0.14, 0.08, (zConvTop + zConvBottom) * 0.5],
      partId: 'conv1d',
      align: 'right',
      tier: 'primary',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    { id: 'm-dt', label: 'Δt = softplus', world: [dtX, 0.08, dtZ - dtD * 0.5 - 0.18], partId: 'dt-branch', align: 'top', tier: 'primary' },
    {
      id: 'm-x-heads',
      label: `x: H00–H15 (${headTag}·G${activeGroup})`,
      world: [xBankCenterX, 0.08, zSplit - splitDepth * 0.5 - 0.20],
      partId: `mamba-head-${activeHead}`,
      align: 'top',
      tier: 'primary',
    },
    { id: 'm-b', label: 'B (8 grps)', world: [bX, 0.08, zSplit - splitDepth * 0.5 - 0.18], partId: 'b-proj', align: 'top', tier: 'primary' },
    { id: 'm-c', label: 'C (8 grps)', world: [cX, 0.08, zSplit - splitDepth * 0.5 - 0.18], partId: 'c-proj', align: 'top', tier: 'primary' },
    {
      id: 'm-hprev',
      label: `Hₜ₋₁ (${headTag}${histTokTag})`,
      world: [hPrevX, 0.08, zSsm - hSize * 0.5 - 0.18],
      partId: 'ssm-state-prev',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'm-hcurr',
      label: `Hₜ (${headTag}${histTokTag} · Fixed)`,
      world: [hCurrX, 0.08, zSsm - hSize * 0.5 - 0.18],
      partId: 'ssm-state-curr',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'm-ssm-bank',
      label: '64-Head Bank (46.0 MiB FP32)',
      world: [bankCenterX, 0.08, zSsm - 0.90],
      partId: 'ssm-state',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    { id: 'm-y', label: 'y = Hₜ·C + D·x', world: [yX - yW * 0.5 - 0.18, 0.08, zY], partId: 'ssm-readout', align: 'left', tier: 'primary' },
    { id: 'm-gnorm', label: 'Gate→8-Grp Norm', world: [normX - normW * 0.5 - 0.22, 0.08, zNorm], partId: 'group-norm', align: 'left', tier: 'primary' },
    {
      id: 'm-out-proj',
      label: computeIsFp8 ? 'W_out · FP8' : 'W_out (4,096→2,688)',
      world: [woutX - woutW * 0.5 - 0.20, 0.08, zOut],
      partId: 'out-proj',
      align: 'left',
      tier: 'primary',
      tone: feederPreservedBf16 ? 'gold' : 'default',
    },
    { id: 'm-res-out', label: '+ Residual', world: [resX, 0.08, zOut + 0.66], partId: 'residual-out', align: 'bottom', tier: 'primary' },
    // Detail zoom-revealed callouts
    { id: 'm-xbc-pre-sec', label: `xBC_pre (T=${clampedT})`, world: [convX + convW * 0.5 + 0.18, 0.08, zXbcPre], partId: 'xbc-pre', align: 'right', tier: 'detail', tone: 'muted' },
    { id: 'm-loop-sec', label: 'In-place update: Hₜ = dA·Hₜ₋₁ + Δt·x·B', world: [(hPrevX + hCurrX) * 0.5, 0.08, zSsm + hSize * 0.5 + 0.62], partId: 'ssm-state-curr', align: 'bottom', tier: 'detail', tone: 'cyan' },
  );

  if (isFeeder) {
    b.landmarks.push({
      id: 'm-feeder-note',
      label: 'Bay 42 Feeder · Preserved in BF16 (NVIDIA §4.2)',
      world: [-4.65, 0.08, zRow0 - 0.12],
      align: 'top',
      tier: 'primary',
      tone: 'gold',
    });
  }

  // Calipers highlighting fixed SSM state vs growing T sequence stream
  b.calipers.push(
    {
      id: 'cal-ssm-fixed',
      aWorld: [hPrevX - hSize * 0.5, 0.02, zSsm + hSize * 0.5 + 0.18],
      bWorld: [hCurrX + hSize * 0.5, 0.02, zSsm + hSize * 0.5 + 0.18],
      label: 'Fixed-size state (46.0 MiB FP32)',
      partIds: ['ssm-state-prev', 'ssm-state-curr', 'ssm-state'],
    },
  );

  const packed = b.pack();
  const reg = (k: string) => b.sheetRegistry.get(k);
  const convTaps = [0, 1, 2, 3].map((slot) => reg(`conv_buf:${slot}`));
  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('input'), reg('rmsnorm')], 0.50, 0.10, 0.50, MAMBA_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('rmsnorm'), reg('w_in'), reg('z'), reg('xbc_pre'), reg('dt')], 0.50, 0.10, 0.50, MAMBA_FIT_BOUNDS),
    2: computeSheetGroupBounds([reg('xbc_pre'), ...convTaps, reg(`x:${activeHead}`), reg('b'), reg('c')], 0.50, 0.10, 0.50, MAMBA_FIT_BOUNDS),
    3: computeSheetGroupBounds([reg('h_prev'), reg('h_curr'), reg(`x:${activeHead}`), reg('b'), reg('dt')], 0.50, 0.10, 0.50, MAMBA_FIT_BOUNDS),
    4: computeSheetGroupBounds([reg('h_curr'), reg('c'), reg(`x:${activeHead}`), reg('y')], 0.50, 0.10, 0.50, MAMBA_FIT_BOUNDS),
    5: computeSheetGroupBounds([reg('y'), reg('z'), reg('y_norm'), reg('w_out'), reg('residual')], 0.50, 0.10, 0.50, MAMBA_FIT_BOUNDS),
  };

  const mambaLayerCount = NEMOTRON_SPEC.bays.filter((x) => x.kind === 'M').length;
  const prodConvDim =
    NEMOTRON_SPEC.mamba.heads * NEMOTRON_SPEC.mamba.channels +
    2 * NEMOTRON_SPEC.mamba.groups * NEMOTRON_SPEC.mamba.state;
  const ssmBytesAll23 =
    mambaLayerCount *
    NEMOTRON_SPEC.mamba.heads *
    NEMOTRON_SPEC.mamba.channels *
    NEMOTRON_SPEC.mamba.state *
    4;
  const conv1dBytesAll23 =
    mambaLayerCount *
    prodConvDim *
    NEMOTRON_SPEC.mamba.kernel *
    2;

  return {
    kind: 'cutaway_mamba',
    bayId,
    ...packed,
    calipers: b.calipers,
    valueAtlas: atlas,
    mambaSim: sim,
    activeFormula,
    ...(focusSet.active ? { activeOpPartIds: b.activeOpPartIds } : {}),
    cellPopover,
    activeCellGrids,
    focusedAccumulation,
    fitBounds: MAMBA_FIT_BOUNDS,
    opFocusBounds,
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
      activeHead,
      activeGroup,
      stateTokenIndex: stateTokIdx,
      miniatureHeadCount: sim.nHeads,
      miniatureGroupCount: sim.nGroups,
      usesSmoothRibbons: true,
      voxelArrowCount: 0,
      ssmBytesAll23,
      conv1dBytesAll23,
    },
  };
}
