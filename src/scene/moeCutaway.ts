/**
 * Stage A+1 · Increment 4: Granular MoE (128+1) Cutaway Foundation & Live Mathematics
 * (`Bay 02 · Granular MoE 128+1` and optional quick switch `Bay 52 · Final MoE`).
 *
 * Uses the shared thin-sheet XZ drafting-table rendering architecture:
 *   - SheetsPass (RG32F live value atlas + zoom-adaptive cell grid)
 *   - FlatFlowRibbonsPass (smooth Bezier/polyline pathways, 0 voxel arrows)
 *   - LinesPass (crisp architectural guide frames)
 *   - 1-to-1 solidData mirror so existing solidFp8FlagsByPart contract checks pass identically
 *
 * Spatial parallel topology (top -Z to bottom +Z):
 *   Right Residual Highway: Input (2,688) ─────────────────────────► + Residual
 *   Top: RMSNorm (U) ──► clean parallel fork into two side-by-side branches:
 *     LEFT / CENTER (Sparse Routed Path, Focused Token t):
 *       W_router [FP32] ─► Router Bank (8×16 = 128 · Top-6 active, 122 idle)
 *         ─► Top-6 Weights (×2.5)
 *         ─► Focused Routed Expert E_e (Non-Gated ReLU², 2 matrices: W_up → Pre-Act h → ReLU² → W_down)
 *         ─► Routed Sum (Σ wᵢ·yᵢ)
 *     RIGHT PARALLEL (Always-On Double-Width Shared Expert, 3,712 = 2×1,856, unscaled 1×):
 *       Shared W_up (2×) ─► Shared ReLU² (2×) ─► Shared W_down (2×) ─► Shared Out (1×)
 *   Bottom Merge: Routed Sum + Shared Out ─► MoE Out ─► + Residual
 */

import {
  describeMoeCellFormula,
  evaluateSingleRoutedExpert,
  formatExpertId,
  inferMoeCellExpert,
  MOE_SIM_CONFIG,
  runMoeSim,
  type MoeCellFormula,
  type MoeCellRef,
  type MoeSheetSpan,
  type MoeSimResult,
  type MoeTensorKey,
} from '../math/moeSim';
import { normalizeMoeWeightCell } from '../model/moePlayback';
import type { PrecisionMode } from '../model/spec';
import type { GqaViewAngle } from '../model/state';
import type { AABB } from '../render/camera';
import type { PickBox } from '../render/pick';
import { FlowRibbonBuilder } from '../render/ribbons';
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
  SHEET_REVEAL_NONE,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../render/sheets';
import {
  computeMoeFocusedAccumulation,
  type FocusedAccumulationResult,
} from './accumulation';
import type { AtlasSlice } from './atlasPacker';
import type { CutawayPartInfo, CutawaySceneGeometry, MoECutawayStats } from './cutaways';
import { buildMoeValueAtlas } from './moeAtlas';
import {
  buildMoeActiveCellGrids,
  buildMoeActiveOpPartIds,
  buildMoeCellPopover,
  buildMoeSpatialScene,
  computeMoeSheetReveal,
  fitSpatialMoeCamera,
  MOE_SPATIAL_FIT_BOUNDS,
  type MoeRevealContext,
  type MoeRevealTag,
  type SheetRevealTuple,
} from './moeSpatial';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  computeSheetGroupBounds,
  createSheetFocusSet,
  drawOperandDependencyXZ,
  EMPTY_SHEET_FOCUS_SET,
  rectFromSheetGrid,
  resolveSheetHighlight,
  RIBBON_ACTIVE,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  RIBBON_RESIDUAL_ACTIVE,
  RIBBON_SSM_LOOP,
  type SheetCandidateIdentity,
  type SheetFocusEntry,
  type SheetFocusSet,
  type SheetGeomRecord,
  type WorldSpanRect,
} from './sheetInspector';
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
  type LandmarkAnchor,
} from './spine';

const SHEET_T = 0.068;
const LIFT_Y = 0.032;

export { fitSpatialMoeCamera, MOE_SPATIAL_FIT_BOUNDS };

/** Invariant framing bounds across T=1..16, Bay 02/52, and BF16/FP8 Selective. */
export const MOE_FIT_BOUNDS: AABB = {
  min: [-8.0, -0.35, -7.2],
  max: [7.9, 0.85, 7.8],
};

/** Parse `moe-expert-0..127` part IDs into expert indices, or null. */
export function parseMoeExpertPartId(partId: string | null | undefined): number | null {
  if (!partId || !partId.startsWith('moe-expert-')) return null;
  const idx = Number.parseInt(partId.slice('moe-expert-'.length), 10);
  if (!Number.isInteger(idx) || idx < 0 || idx >= MOE_SIM_CONFIG.nRoutedExperts) return null;
  return idx;
}

export interface BuildMoECutawayOptions {
  readonly tokens: number;
  readonly precision: PrecisionMode;
  readonly bayId?: 2 | 52;
  readonly selectedPart: string | null;
  readonly hoveredPart: string | null;
  readonly focusToken?: number;
  readonly activeExpert?: number;
  readonly activeCell?: MoeCellRef | null;
  readonly stepperStep?: number | null;
  /** V2: Enable progressive cell-by-cell accumulation reveal when Math Stepper is active. Defaults to false. */
  readonly stepperActive?: boolean;
  /** V2: Suppress progressive reveal masking under prefers-reduced-motion. Defaults to false. */
  readonly reducedMotion?: boolean;
  /** Optional sub-step progress [0..1) for smooth frontier pulse phase. */
  readonly stepperSubProgress?: number;
  /** RC1: Focused-cell term-by-term accumulation progress [0..1]; defaults to 1 (unrestricted). */
  readonly termProgress?: number;
  /** Layout mode: `'three_quarter'` builds the 3D Spatial MoE architecture; `'near_plan'` (default) keeps the coplanar XZ schematic. */
  readonly viewAngle?: GqaViewAngle;
  /** Optional authoritative simulator result from a unified `HybridExecutionResult`. */
  readonly simOverride?: MoeSimResult;
}

interface MoeSheetSpec {
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
  tensorKey?: MoeTensorKey;
  tokenIndex?: number;
  atlasSlice?: AtlasSlice;
  registryKey?: string;
  cellFamily: number;
  reveal?: SheetRevealTuple;
}

class MoeSheetBuilder {
  private readonly bayId: number;
  private readonly revealEnabled: boolean;
  private readonly sheetFloats: number[] = [];
  private readonly revealFloats: number[] = [];
  private readonly solidFloats: number[] = [];
  private readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();

  constructor(bayId: number, revealEnabled = false) {
    this.bayId = bayId;
    this.revealEnabled = revealEnabled;
  }

  addSheet(s: MoeSheetSpec): SheetGeomRecord {
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

    // 1b. Optional parallel V2 progressive reveal buffer (4 floats/sheet)
    if (this.revealEnabled) {
      const rv = s.reveal ?? ([SHEET_REVEAL_NONE, 0, 0, 0] as const);
      this.revealFloats.push(rv[0], rv[1], rv[2], rv[3]);
    }

    // 2. 1-to-1 mirror in solidData (12 floats: fp8Flag at +7, seed at +8)
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

    // 3. Pick AABB (slightly padded vertically for effortless picking)
    const hx = s.ex * 0.5;
    const hy = Math.max(s.ey * 0.5, 0.06);
    const hz = s.ez * 0.5;
    this.pickBoxes.push({
      bayId: this.bayId,
      partId: s.partId,
      headIndex: s.headIndex,
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
      min: [s.cx - hx, s.cy - hy, s.cz - hz],
      max: [s.cx + hx, s.cy + hy, s.cz + hz],
    });

    const rec: SheetGeomRecord = {
      cx: s.cx,
      cy: s.cy,
      cz: s.cz,
      ex: s.ex,
      ey: s.ey,
      ez: s.ez,
      gridCols: s.gridCols,
      gridRows: s.gridRows,
    };
    if (s.registryKey) {
      this.sheetRegistry.set(s.registryKey, rec);
    }
    return rec;
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

  addDashedFrameXZ(
    x0: number,
    z0: number,
    x1: number,
    z1: number,
    y: number,
    style = LINE_STYLE_GUIDE,
    alpha = 0.45,
    dash = 0.22,
    gap = 0.14,
  ): void {
    const dashedLine = (ax: number, az: number, bx: number, bz: number) => {
      const dx = bx - ax;
      const dz = bz - az;
      const len = Math.hypot(dx, dz);
      if (len < 1e-4) return;
      const ux = dx / len;
      const uz = dz / len;
      for (let d = 0; d < len; d += dash + gap) {
        const dEnd = Math.min(len, d + dash);
        this.addGuideLine(
          ax + ux * d, y, az + uz * d,
          ax + ux * dEnd, y, az + uz * dEnd,
          0.95,
          style,
          alpha,
        );
      }
    };
    dashedLine(x0, z0, x1, z0);
    dashedLine(x1, z0, x1, z1);
    dashedLine(x1, z1, x0, z1);
    dashedLine(x0, z1, x0, z0);
  }

  pack() {
    const flowPacked = this.flow.pack();
    return {
      sheetData: new Float32Array(this.sheetFloats),
      sheetCount: this.sheetFloats.length / SHEET_INSTANCE_FLOATS,
      ...(this.revealEnabled ? { sheetRevealData: new Float32Array(this.revealFloats) } : {}),
      solidData: new Float32Array(this.solidFloats),
      solidCount: this.solidFloats.length / SOLID_INSTANCE_FLOATS,
      lineData: new Float32Array(this.lineFloats),
      lineCount: this.lineFloats.length / LINE_INSTANCE_FLOATS,
      // Explicitly ZERO voxel-bead arrows in the rebuilt MoE cutaway
      ribbonData: new Float32Array(0),
      ribbonCount: 0,
      flowVertices: flowPacked.vertices,
      flowIndices: flowPacked.indices,
    };
  }
}

function buildMoeFocusSet(
  formula: MoeCellFormula | null,
  sim: MoeSimResult,
  focusToken: number,
): SheetFocusSet {
  if (!formula) return EMPTY_SHEET_FOCUS_SET;
  const c = formula.cell;
  const targetEntry: SheetFocusEntry = {
    tensor: c.tensor,
    expertId: c.expertId,
    tokenStart: c.tokenIndex,
    tokenEnd: c.tokenIndex,
    rowStart: c.row,
    rowEnd: c.row,
    colStart: c.col,
    colEnd: c.col,
    role: 'target',
  };
  const upEntries: SheetFocusEntry[] = formula.upstreamSpans.map((up, idx) => ({
    tensor: up.tensor,
    expertId: up.expertId,
    tokenStart: up.tokenIndex,
    tokenEnd: up.tokenIndex,
    rowStart: up.rowStart,
    rowEnd: up.rowEnd,
    colStart: up.colStart,
    colEnd: up.colEnd,
    role: idx === 0 ? 'operand_a' : 'operand_b',
  }));

  // Top-6 weight normalization refinement: w_j = (σ_j / Σ_{m=0..5} σ_m) × 2.5.
  // The selected expert contributes the numerator (operand_a) AND joins all 6 selected unbiased
  // sigmoid scores in the denominator; highlight the other 5 selected Top-6 experts as operand_b!
  if (c.tensor === 'topk_weights') {
    const tok = Math.max(0, Math.min(sim.tokens - 1, c.tokenIndex ?? focusToken));
    const slot = Math.max(0, Math.min(sim.topK - 1, c.rankSlot ?? c.col));
    const winnerId = sim.topk_indices[tok * sim.topK + slot];
    for (let m = 0; m < sim.topK; m++) {
      const eid = sim.topk_indices[tok * sim.topK + m];
      if (eid === winnerId) continue;
      const bankRow = Math.floor(eid / MOE_SIM_CONFIG.bankCols);
      const bankCol = eid % MOE_SIM_CONFIG.bankCols;
      upEntries.push({
        tensor: 'router_bank',
        expertId: eid,
        tokenStart: tok,
        tokenEnd: tok,
        rowStart: bankRow,
        rowEnd: bankRow,
        colStart: bankCol,
        colEnd: bankCol,
        role: 'operand_b',
      });
    }
  }

  return createSheetFocusSet([targetEntry, ...upEntries]);
}

function buildMoePartCatalog(params: {
  readonly bayId: 2 | 52;
  readonly isFp8Mode: boolean;
  readonly focusToken: number;
  readonly focusTokLabel: string;
  readonly focusExpert: number;
  readonly focusExpTag: string;
  readonly activeExpertIds: readonly number[];
  readonly focusedEval: ReturnType<typeof evaluateSingleRoutedExpert>;
}): Record<string, CutawayPartInfo> {
  const {
    bayId,
    isFp8Mode,
    focusToken,
    focusTokLabel,
    focusExpert,
    focusExpTag,
    activeExpertIds,
    focusedEval,
  } = params;
  const partCatalog: Record<string, CutawayPartInfo> = {
    'residual-in': {
      id: 'residual-in',
      title: 'Residual Stream Input (2,688)',
      badge: 'Activation · 0 B Token Cache',
      summary:
        'Incoming hidden state from the 2,688-channel emerald residual spine (miniature d=8). MoE layers are strictly per-token independent and allocate 0 B of sequence cache across 1 → 1M tokens.',
      specs: [
        'Production shape: [T, 2,688] · Miniature live sim: [T, 8] (Float32 storage)',
        'Hover or click any cell to trace per-token RMSNorm, routing, and residual merge.',
      ],
    },
    rmsnorm: {
      id: 'rmsnorm',
      title: 'Pre-MoE RMSNorm (ε = 1e−5)',
      badge: isFp8Mode ? 'FP8 Selective (prod)' : 'BF16 (prod)',
      summary:
        'Normalizes each token vector by √(mean(X²) + 1e−5) × γ_in before forking in parallel to BOTH the sparse 128-expert router path (rear/center) and the always-on double-width shared expert (front).',
      specs: [
        'Matches NVIDIA NemotronH rms_norm_eps = 1e−5 (learned non-unit γ_in, no bias)',
        'Strictly per-token independent: mutating any other token t′ ≠ t leaves U[:, t] unchanged.',
      ],
    },
    'w-router': {
      id: 'w-router',
      title: 'Router Gate Linear (W_router · 128 × 2,688)',
      badge: 'Always FP32 (Never FP8)',
      summary:
        'Computes 128 expert logits per token without linear bias, applies sigmoid, and adds learned e_score_correction_bias for Top-6 selection only. Kept strictly FP32 in production even under FP8 Selective (NVIDIA §4.2).',
      specs: [
        'Production: 128 × 2,688 (344K params, FP32) · Miniature: 128 × 8 (no bias)',
        'e_score_correction_bias guides load-balanced selection without distorting expert weights.',
      ],
    },
    'router-unbiased': {
      id: 'router-unbiased',
      title: 'Unbiased Sigmoid Scores (Bias Stripped)',
      badge: 'Always FP32 · All 6 in Denom',
      summary:
        'After Top-6 winners are selected using biased choice scores (σ + e_score_correction_bias), the bias is discarded. All six selected UNBIASED sigmoid scores σ_m (including the numerator expert itself) sum into the normalization denominator Σ₆ σ_m.',
      specs: [
        `Active token t${focusToken} "${focusTokLabel}": ${activeExpertIds.map(formatExpertId).join(', ')}`,
        'Selection uses (σ + bias); weight normalization strictly uses unbiased σ.',
      ],
    },
    router: {
      id: 'router',
      title: 'Top-6 Router Weights (Unbiased σ → ×2.5)',
      badge: 'Always FP32 · Σw = 2.500',
      summary:
        'Normalizes the 6 UNBIASED sigmoid scores (excluding e_score_correction_bias) by their 6-term sum Σ_{m=1..6} σ_m (including the numerator expert itself) and multiplies by routed_scaling_factor = 2.5.',
      specs: [
        `Active token t${focusToken} "${focusTokLabel}": ${activeExpertIds.map(formatExpertId).join(', ')}`,
        'Equation: w_j = (σ_j / Σ_{m=1..6} σ_m) × 2.5 · Σw = 2.500.',
      ],
    },
    'routed-bank': {
      id: 'routed-bank',
      title: `Active Top-6 Routed Experts (${focusExpTag})`,
      badge: isFp8Mode ? 'FP8 Weights (prod) · 6/128 Active' : 'BF16 Weights (prod) · 6/128 Active',
      summary:
        'Only 6 of 128 routed experts execute per token (95.3% sparse) as paired non-gated ReLU² MLPs (W_up + W_down only, zero gate_proj). Click any active tile or Top-6 slot to magnify that expert in the detail workbench.',
      specs: [
        'Production: 128 experts × 2 matrices (1,856 width) = 1.30B params/layer × 23 = 29.8B (~94% of model)',
        'Each active tile renders as 2 paired micro-plates representing non-gated W_up and W_down.',
      ],
    },
    'routed-branch': {
      id: 'routed-branch',
      title: `Six Independent Routed Outputs (wᵢ·yᵢ · ${focusExpTag})`,
      badge: '6 Parallel Branches → Σ₆',
      summary:
        'All six active Top-6 experts evaluate independently in parallel and scale their outputs by their normalized ×2.5 router weights before summing into Routed Sum. The detail workbench magnifies 1 of these 6 branches at a time.',
      specs: [
        `Active 6 for t${focusToken} "${focusTokLabel}": ${activeExpertIds.map(formatExpertId).join(', ')}`,
        'Click any of the 6 weighted-output strips to magnify that expert inside W_up → ReLU² → W_down.',
      ],
    },
    'inactive-experts': {
      id: 'inactive-experts',
      title: `Idle Routed Experts (122 / 128 Skipped · ${focusExpTag})`,
      badge: '95.3% Sparse · 0 Contribution',
      summary:
        'These 122 routed experts ranked outside the Top-6 for the focused token and contribute exact 0 to the layer output. Clicking any idle tile previews its weights and hypothetical ReLU² activations on demand.',
      specs: [
        'Idle experts are never multiplied into Routed Sum (appliedWeight = 0).',
        'Cycle or select tokens t0..t15 to watch different Top-6 expert subsets rise cleanly.',
      ],
    },
    'routed-w-up': {
      id: 'routed-w-up',
      title: `Focused Expert ${focusExpTag} · W_up (1,856)`,
      badge: isFp8Mode ? 'FP8 Weights (prod)' : 'BF16 Weights (prod)',
      summary:
        'First of only 2 matrices in NemotronHMLP (no SwiGLU gate_proj, no linear bias). Projects normalized token U into intermediate hidden width 1,856 (miniature d_exp = 4).',
      specs: [
        focusedEval.isActive
          ? `${focusExpTag} is ACTIVE (#${(focusedEval.activeRank ?? 0) + 1} · w = ${focusedEval.appliedWeight.toFixed(3)}) for t${focusToken}.`
          : `${focusExpTag} is IDLE (Rank #${focusedEval.choiceRank + 1}/128 · 0 contribution) for t${focusToken}.`,
        'Production precision metadata: FP8 in FP8 Selective mode (live simulator stays Float32).',
      ],
    },
    'routed-pre': {
      id: 'routed-pre',
      title: `Focused Expert ${focusExpTag} · Pre-Activation h`,
      badge: 'h = W_up @ U',
      summary:
        'Linear pre-activation vector before ReLU². Negative or zero values are zeroed cleanly in the next step; positive entries are squared.',
      specs: ['Miniature width: d_exp = 4 · Production width: moe_intermediate_size = 1,856'],
    },
    'routed-act': {
      id: 'routed-act',
      title: `Focused Expert ${focusExpTag} · ReLU² Activation`,
      badge: 'max(0, h)² · Non-Gated',
      summary:
        'Applies squared ReLU (exact 0 for h ≤ 0, h² for h > 0) without a third gate_proj matrix, eliminating 33% of MLP weight matrices compared to SwiGLU.',
      specs: ['Exact equation: a[i] = h[i] > 0 ? h[i] * h[i] : 0'],
    },
    'routed-w-down': {
      id: 'routed-w-down',
      title: `Focused Expert ${focusExpTag} · W_down (2,688)`,
      badge: isFp8Mode ? 'FP8 Weights (prod)' : 'BF16 Weights (prod)',
      summary:
        'Second of the 2 expert matrices, projecting sparse ReLU² activations back to the 2,688-channel residual width before scaling by the expert’s Top-6 router weight.',
      specs: ['Production: 1,856 → 2,688 · Miniature: 4 → 8 (no bias)'],
    },
    'routed-sum': {
      id: 'routed-sum',
      title: 'Parallel Routed Sum (Σ wᵢ·yᵢ)',
      badge: '6 Active · ×2.5 Scaled',
      summary:
        'Weighted parallel sum of all 6 independently evaluated routed experts (scaled by routed_scaling_factor = 2.5). All 122 idle experts contribute exact 0.',
      specs: ['Equation: y_routed[:, t] = Σ_{j=0..5} topk_weights[t, j] · y_{topk_indices[t, j]}[:, t]'],
    },
    'shared-expert': {
      id: 'shared-expert',
      title: 'Always-On Double-Width Shared Expert (3,712 = 2×)',
      badge: isFp8Mode ? 'FP8 Weights (prod) · Unscaled 1×' : 'BF16 Weights (prod) · Unscaled 1×',
      summary:
        'Runs in parallel for EVERY token alongside the 128 routed experts. Its intermediate width is strictly 2× (3,712 = 2 × 1,856) while hidden input/output width remains 1× (2,688).',
      specs: [
        'Exact 2× intermediate width (miniature d_shared = 8 = 2 × d_exp) and 1× hidden width (d = 8).',
        'NOT multiplied by routed_scaling_factor = 2.5 (added directly at 1.0× weight).',
      ],
    },
    'shared-act': {
      id: 'shared-act',
      title: 'Shared Expert ReLU² (3,712 Width)',
      badge: 'Always-On · 2× Intermediate',
      summary:
        'Non-gated ReLU² activation across all 3,712 shared intermediate channels (miniature 8 channels), providing dense always-on capacity in parallel with sparse routing.',
      specs: ['Zero gate_proj, zero router gating, zero sequence cache.'],
    },
    'shared-w-down': {
      id: 'shared-w-down',
      title: 'Shared Expert W_down (3,712 → 2,688)',
      badge: isFp8Mode ? 'FP8 Weights (prod) · 2× Inter' : 'BF16 Weights (prod) · 2× Inter',
      summary:
        'Double-intermediate-width down projection mapping 3,712 shared ReLU² activations back to the 2,688-channel hidden space (miniature 8 → 8).',
      specs: ['Strictly 2× intermediate width (3,712 vs 1,856) and 1× hidden width (2,688).'],
    },
    'shared-out': {
      id: 'shared-out',
      title: 'Shared Expert Output (Unscaled 1.0×)',
      badge: 'Parallel Branch · Always On',
      summary:
        'Output of the always-on double-width shared expert, merged directly with the 2.5×-scaled Top-6 Routed Sum.',
      specs: ['Does NOT receive routed_scaling_factor = 2.5.'],
    },
    'moe-out': {
      id: 'moe-out',
      title: 'MoE Parallel Merge (Routed + Shared)',
      badge: 'y_routed + y_shared',
      summary:
        'Combines the sparse Top-6 routed sum and the always-on double-width shared output before adding back to the untouched emerald residual stream.',
      specs: ['Equation: y_moe = Σ_{j=1..6} wⱼ·yⱼ + y_shared'],
    },
    'residual-out': {
      id: 'residual-out',
      title: 'Residual Stream Merge (+ 2,688)',
      badge: 'Activation · Emerald',
      summary:
        'Adds y_moe back into the continuous 2,688-channel emerald residual stream: Y = X + y_routed + y_shared.',
      specs: [
        bayId === 52
          ? 'Bay 52 (Final MoE): feeds final RMSNorm and the untied LM Head (131,072 vocab).'
          : 'Bay 02 (Granular MoE 128+1): feeds Bay 03 Mamba-2 along the hybrid spine.',
      ],
    },
  };

  partCatalog[`moe-expert-${focusExpert}`] = {
    id: `moe-expert-${focusExpert}`,
    title: focusedEval.isActive
      ? `Routed Expert ${focusExpTag} · Active Top-6 #${(focusedEval.activeRank ?? 0) + 1}`
      : `Routed Expert ${focusExpTag} · Idle (Rank #${focusedEval.choiceRank + 1}/128)`,
    badge: focusedEval.isActive
      ? `w = ${focusedEval.appliedWeight.toFixed(3)} · ReLU²`
      : 'Skipped · 0 Contribution',
    summary: focusedEval.isActive
      ? `Selected in Top-6 for token t${focusToken} "${focusTokLabel}" via choice score ${focusedEval.choiceScore.toFixed(3)} (unbiased σ = ${focusedEval.unbiasedScore.toFixed(3)}, e_bias = ${focusedEval.bias >= 0 ? '+' : '−'}${Math.abs(focusedEval.bias).toFixed(3)}).`
      : `Ranked #${focusedEval.choiceRank + 1} of 128 for token t${focusToken} "${focusTokLabel}" (choice score ${focusedEval.choiceScore.toFixed(3)}). Not executed in layer sum (0 contribution); detail workbench previews its hypothetical ReLU² weights on demand.`,
    specs: [
      `Non-gated 2-matrix MLP: W_up (2,688→1,856) + ReLU² + W_down (1,856→2,688), zero gate_proj.`,
      `Production weights: ${isFp8Mode ? 'FP8 Selective (router remains FP32)' : 'BF16'} · 0 B token cache.`,
    ],
  };

  return partCatalog;
}

export function buildMoECutaway(opts: BuildMoECutawayOptions): CutawaySceneGeometry {
  const bayId: 2 | 52 = opts.bayId === 52 ? 52 : 2;
  const clampedT = opts.simOverride
    ? opts.simOverride.tokens
    : Math.max(1, Math.min(MOE_SIM_CONFIG.maxTokens, Math.round(opts.tokens)));
  const isFp8Mode = opts.precision === 'fp8_selective';
  const sim = opts.simOverride ?? runMoeSim(clampedT);

  // Token-independent weight cells (w_router, shared_w_up, shared_w_down) must never pin focusToken
  const effectiveActiveCell = opts.activeCell ? normalizeMoeWeightCell(opts.activeCell) : null;
  const focusToken = Math.max(
    0,
    Math.min(clampedT - 1, effectiveActiveCell?.tokenIndex ?? opts.focusToken ?? clampedT - 1),
  );
  const activePart = opts.hoveredPart ?? opts.selectedPart;

  // Gather the 6 winning experts for the focused token
  const activeExpertIds: number[] = [];
  const activeRankMap = new Map<number, number>();
  for (let j = 0; j < sim.topK; j++) {
    const eid = sim.topk_indices[focusToken * sim.topK + j];
    activeExpertIds.push(eid);
    activeRankMap.set(eid, j);
  }

  // Synchronize focused expert ID from inspected cell > hovered/selected part > explicit activeExpert > rank-0 winner
  const partExpert = parseMoeExpertPartId(activePart);
  const rawFallbackExpert =
    partExpert ??
    (opts.activeExpert !== undefined
      ? Math.max(0, Math.min(MOE_SIM_CONFIG.nRoutedExperts - 1, opts.activeExpert))
      : activeExpertIds[0]);
  const inspectedExpert = inferMoeCellExpert(effectiveActiveCell, sim, rawFallbackExpert);
  const focusExpert = inspectedExpert ?? rawFallbackExpert;

  const focusedEval = evaluateSingleRoutedExpert(sim, focusExpert, focusToken);
  const focusExpTag = formatExpertId(focusExpert);
  const absFocusToken = focusToken + (sim.startTokenOffset ?? 0);
  const focusTokLabel = sim.tokenLabels[focusToken] ?? `t${absFocusToken}`;

  const atlas = buildMoeValueAtlas(sim, focusToken, focusExpert);

  // Compute live formula readout if a cell is hovered/selected/stepped
  const activeFormula: MoeCellFormula | null = effectiveActiveCell
    ? describeMoeCellFormula(sim, effectiveActiveCell, focusToken, focusExpert)
    : null;
  const focusSet = buildMoeFocusSet(activeFormula, sim, focusToken);

  const partCatalog = buildMoePartCatalog({
    bayId,
    isFp8Mode,
    focusToken: absFocusToken,
    focusTokLabel,
    focusExpert,
    focusExpTag,
    activeExpertIds,
    focusedEval,
  });

  const moeStats: MoECutawayStats = {
    bayId,
    routedTotal: sim.nRoutedExperts,
    routedActiveCount: sim.topK,
    activeExpertIds,
    sharedExpertCount: 1,
    routedWidth: MOE_SIM_CONFIG.prodRoutedWidth,
    sharedWidth: MOE_SIM_CONFIG.prodSharedWidth,
    matricesPerExpert: 2,
    hasGateProj: false,
    cacheBytes: 0,
    weightsFp8: isFp8Mode,
    routerFp8: false,
    focusToken,
    focusExpert,
    usesSmoothRibbons: true,
    voxelArrowCount: 0,
  };

  const stepperActive = opts.stepperActive ?? false;
  const reducedMotion = opts.reducedMotion ?? false;
  const stepperStep = opts.stepperStep ?? null;
  const termProgress = opts.termProgress;

  if (opts.viewAngle === 'three_quarter') {
    return buildMoeSpatialScene({
      bayId,
      clampedT,
      isFp8Mode,
      activePart,
      focusToken,
      focusExpert,
      activeExpertIds,
      activeRankMap,
      focusedEval,
      activeCell: effectiveActiveCell,
      activeFormula,
      focusSet,
      stepperStep,
      stepperActive,
      reducedMotion,
      stepperSubProgress: opts.stepperSubProgress,
      termProgress,
      sim,
      atlas,
      partCatalog,
      moeStats,
    });
  }

  const revealEnabled =
    stepperActive && !reducedMotion && stepperStep !== null && stepperStep !== undefined;
  const focusedAccumulation: FocusedAccumulationResult | null =
    stepperActive &&
    !reducedMotion &&
    stepperStep !== null &&
    stepperStep !== undefined &&
    termProgress !== undefined &&
    Number.isFinite(termProgress) &&
    termProgress < 0.999 &&
    effectiveActiveCell
      ? computeMoeFocusedAccumulation(
          sim,
          effectiveActiveCell,
          stepperStep,
          termProgress,
          focusToken,
          focusExpert,
        )
      : null;
  const pulsePhase =
    opts.stepperSubProgress !== undefined && Number.isFinite(opts.stepperSubProgress)
      ? ((opts.stepperSubProgress * 6) % 1 + 1) % 1
      : 0.25;
  const revealCtx: MoeRevealContext = {
    enabled: revealEnabled,
    step: stepperStep ?? 0,
    activeCell: effectiveActiveCell,
    focusToken,
    focusExpert,
    activeRankMap,
    sim,
    pulsePhase,
  };
  const sheetReveal = (
    tag: MoeRevealTag,
    revealOpts?: { readonly expertId?: number; readonly branchRank?: number },
  ): SheetRevealTuple => computeMoeSheetReveal(revealCtx, tag, revealOpts);

  const b = new MoeSheetBuilder(bayId, revealEnabled);
  const hasCellTrace = focusSet.active;
  const step = hasCellTrace ? null : stepperStep;
  const hasPartFocus = activePart !== null || step !== null;
  const hasAnyFocus = hasCellTrace || hasPartFocus;

  const sheetY = (hl: number): number =>
    hl === SHEET_HL_ACTIVE
      ? LIFT_Y
      : hl === SHEET_HL_OPERAND_A || hl === SHEET_HL_OPERAND_B
        ? LIFT_Y * 0.55
        : 0;

  const sheetHighlight = (
    candidate: SheetCandidateIdentity,
    partIds: readonly string[],
    stepStages: readonly number[],
    idleHighlight = SHEET_HL_NORMAL,
  ): number => {
    const isPartMatch =
      (activePart !== null && partIds.includes(activePart)) ||
      (step !== null && stepStages.includes(step));
    const fallback = isPartMatch
      ? SHEET_HL_ACTIVE
      : hasPartFocus
        ? SHEET_HL_DIMMED
        : idleHighlight;
    return resolveSheetHighlight(focusSet, candidate, fallback);
  };

  // Layout anchors (XZ architectural drafting table)
  const seqDepth = Math.min(1.42, 0.44 + clampedT * 0.062);
  const resX = 6.20;
  const rmsX = 0.80;
  const zTop = -5.85;
  const zRouter = -3.45;
  const zTopk = -1.15;
  const zDetail = 1.55;
  const zBranchOut = 4.05;
  const zBottom = 5.85;

  const sharedX = 3.55;
  const routedCenterX = -1.35;

  const bankTileGeom = new Map<number, SheetGeomRecord>();

  // ==========================================================================
  // 1. RIGHT RESIDUAL HIGHWAY: Input X (2,688) ──────────────► + Residual
  // ==========================================================================
  const resInHl = sheetHighlight({ tensor: 'input', isAggregate: true }, ['residual-in'], [0, 5]);
  b.addSheet({
    cx: resX,
    cy: sheetY(resInHl),
    cz: zTop,
    ex: 1.44,
    ey: SHEET_T,
    ez: seqDepth,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resInHl,
    seed: 11,
    partId: 'residual-in',
    tensorKey: 'input',
    atlasSlice: atlas.input,
    registryKey: 'input',
    cellFamily: CELL_ACTIVATION_GREEN,
    reveal: sheetReveal('input'),
  });

  const resOutHl = sheetHighlight({ tensor: 'residual', isAggregate: true }, ['residual-out'], [5]);
  b.addSheet({
    cx: resX,
    cy: sheetY(resOutHl),
    cz: zBottom,
    ex: 1.44,
    ey: SHEET_T,
    ez: seqDepth,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resOutHl,
    seed: 19,
    partId: 'residual-out',
    tensorKey: 'residual',
    atlasSlice: atlas.residual,
    registryKey: 'residual',
    cellFamily: CELL_ACTIVATION_GREEN,
    reveal: sheetReveal('residual'),
  });

  // Straight emerald residual highway ribbon on the right
  const resRibbonCol = hasCellTrace
    ? resInHl !== SHEET_HL_DIMMED || resOutHl !== SHEET_HL_DIMMED
      ? RIBBON_RESIDUAL
      : RIBBON_DIMMED
    : resInHl === SHEET_HL_ACTIVE || resOutHl === SHEET_HL_ACTIVE
      ? RIBBON_RESIDUAL_ACTIVE
      : hasAnyFocus
        ? RIBBON_DIMMED
        : RIBBON_RESIDUAL;
  b.flow.addPolylineXZ(
    [
      { x: resX, z: zTop + seqDepth * 0.5 + 0.08 },
      { x: resX, z: zBottom - seqDepth * 0.5 - 0.08 },
    ],
    0.008,
    0.16,
    resRibbonCol,
    true,
    0.26,
    0.26,
  );

  // ==========================================================================
  // 2. TOP PRE-NORM: RMSNorm (U) ──► Parallel Fork
  // ==========================================================================
  const rmsHl = sheetHighlight({ tensor: 'rmsnorm', isAggregate: true }, ['rmsnorm'], [0]);
  b.addSheet({
    cx: rmsX,
    cy: sheetY(rmsHl),
    cz: zTop,
    ex: 1.72,
    ey: SHEET_T,
    ez: seqDepth,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: rmsHl,
    seed: 23,
    partId: 'rmsnorm',
    tensorKey: 'rmsnorm',
    atlasSlice: atlas.rmsnorm,
    registryKey: 'rmsnorm',
    cellFamily: CELL_MOE_AMBER,
    reveal: sheetReveal('rmsnorm'),
  });

  // Input -> RMSNorm horizontal tap
  b.flow.addPolylineXZ(
    [
      { x: resX - 0.72 - 0.06, z: zTop },
      { x: rmsX + 0.86 + 0.08, z: zTop },
    ],
    0.008,
    0.13,
    !hasCellTrace && rmsHl === SHEET_HL_ACTIVE
      ? RIBBON_ACTIVE
      : hasAnyFocus
        ? RIBBON_DIMMED
        : RIBBON_QUIET,
    true,
    0.22,
    0.22,
  );

  // ==========================================================================
  // 3. LEFT/CENTER BRANCH: FP32 W_router + 8×16 Router Bank (128 · Top-6)
  // ==========================================================================
  const wRouterX = -5.75;
  const wRouterW = 1.24;
  const wRouterD = 2.56;
  const wRouterHl = sheetHighlight(
    { tensor: 'w_router', isAggregate: true },
    ['w-router', 'router'],
    [1],
    isFp8Mode ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: wRouterX,
    cy: sheetY(wRouterHl),
    cz: zRouter,
    ex: wRouterW,
    ey: SHEET_T,
    ez: wRouterD,
    gridCols: sim.d,
    gridRows: sim.nRoutedExperts,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0, // Strictly FP32 metadata even under FP8 Selective (NVIDIA §4.2)
    highlight: wRouterHl,
    seed: 31,
    partId: 'w-router',
    tensorKey: 'w_router',
    atlasSlice: atlas.wRouter,
    registryKey: 'w_router',
    cellFamily: CELL_MOE_AMBER,
    reveal: sheetReveal('weight'),
  });

  // 8×16 Router Expert Bank (128 experts):
  //   - 122 idle experts: 1 flat tile each (partId: 'inactive-experts', headIndex: e)
  //   - 6 active experts: 2 paired micro-plates (top=W_up, bottom=W_down, partId: 'routed-bank', headIndex: e)
  //     preserving exact 12 solidPickBoxes for 'routed-bank' and 122 for 'inactive-experts' + fp8Flag=1 under FP8!
  const bankCols = MOE_SIM_CONFIG.bankCols;
  const bankRows = MOE_SIM_CONFIG.bankRows;
  const cellStepX = 0.36;
  const cellStepZ = 0.34;
  const bankW = bankCols * cellStepX; // 5.76
  const bankD = bankRows * cellStepZ; // 2.72
  const bankLeft = routedCenterX - bankW * 0.5;
  const bankTop = zRouter - bankD * 0.5;
  const bankEnvelope: SheetGeomRecord = {
    cx: routedCenterX,
    cy: 0,
    cz: zRouter,
    ex: bankW,
    ey: SHEET_T,
    ez: bankD,
    gridCols: bankCols,
    gridRows: bankRows,
  };

  b.addDashedFrameXZ(
    bankLeft - 0.12,
    bankTop - 0.12,
    bankLeft + bankW + 0.12,
    bankTop + bankD + 0.12,
    0.004,
    LINE_STYLE_GUIDE,
    0.42,
  );

  for (let e = 0; e < sim.nRoutedExperts; e++) {
    const r = Math.floor(e / bankCols);
    const c = e % bankCols;
    const cx = bankLeft + (c + 0.5) * cellStepX;
    const cz = bankTop + (r + 0.5) * cellStepZ;
    const isActiveWinner = activeRankMap.has(e);
    const isFocused = e === focusExpert;
    const tileW = cellStepX - 0.048;
    const tileD = cellStepZ - 0.048;
    const tileCandidate: SheetCandidateIdentity = {
      tensor: 'router_bank',
      expertId: e,
      tokenIndex: focusToken,
      isAggregate: false,
    };

    if (isActiveWinner) {
      // Paired 2-matrix micro-plates (W_up + W_down, non-gated ReLU²)
      const halfD = (tileD - 0.022) * 0.5;
      const fallbackHl =
        activePart === 'routed-bank' || activePart === `moe-expert-${e}` || step === 1 || step === 2
          ? SHEET_HL_ACTIVE
          : hasPartFocus
            ? SHEET_HL_DIMMED
            : isFocused
              ? SHEET_HL_ACTIVE
              : SHEET_HL_PATHWAY;
      const hl = resolveSheetHighlight(focusSet, tileCandidate, fallbackHl);
      const plateY =
        hl === SHEET_HL_ACTIVE
          ? LIFT_Y + 0.014
          : hl === SHEET_HL_OPERAND_A || hl === SHEET_HL_OPERAND_B || hl === SHEET_HL_PATHWAY
            ? LIFT_Y * 0.72
            : 0;

      const activeTileReveal = sheetReveal('router_bank', { expertId: e });
      const topPlate = b.addSheet({
        cx,
        cy: plateY,
        cz: cz - halfD * 0.5 - 0.011,
        ex: tileW,
        ey: SHEET_T * 1.12,
        ez: halfD,
        gridCols: 1,
        gridRows: 1,
        role: SHEET_ROLE_SOFTMAX,
        fp8Flag: isFp8Mode ? 1 : 0,
        highlight: hl,
        seed: 100 + e,
        partId: 'routed-bank',
        headIndex: e,
        tensorKey: 'router_bank',
        tokenIndex: focusToken,
        atlasSlice: atlas.routerBankTiles[e],
        cellFamily: CELL_MOE_AMBER,
        reveal: activeTileReveal,
      });

      b.addSheet({
        cx,
        cy: plateY,
        cz: cz + halfD * 0.5 + 0.011,
        ex: tileW,
        ey: SHEET_T * 1.12,
        ez: halfD,
        gridCols: 1,
        gridRows: 1,
        role: SHEET_ROLE_WEIGHT,
        fp8Flag: isFp8Mode ? 1 : 0,
        highlight: hl,
        seed: 250 + e,
        partId: 'routed-bank',
        headIndex: e,
        tensorKey: 'router_bank',
        tokenIndex: focusToken,
        atlasSlice: atlas.routerBankTiles[e],
        cellFamily: CELL_WEIGHT_BLUE,
        reveal: activeTileReveal,
      });

      bankTileGeom.set(e, {
        cx,
        cy: plateY,
        cz,
        ex: tileW,
        ey: topPlate.ey,
        ez: tileD,
        gridCols: 1,
        gridRows: 1,
      });
    } else {
      const idleFallbackHl =
        activePart === 'inactive-experts' || activePart === `moe-expert-${e}`
          ? SHEET_HL_ACTIVE
          : !hasPartFocus && isFocused
            ? SHEET_HL_ACTIVE
            : SHEET_HL_DIMMED;
      const idleHl = resolveSheetHighlight(focusSet, tileCandidate, idleFallbackHl);
      const idleGeom = b.addSheet({
        cx,
        cy: idleHl === SHEET_HL_ACTIVE ? LIFT_Y * 0.8 : idleHl === SHEET_HL_OPERAND_A || idleHl === SHEET_HL_OPERAND_B ? LIFT_Y * 0.5 : 0,
        cz,
        ex: tileW,
        ey: SHEET_T * 0.72,
        ez: tileD,
        gridCols: 1,
        gridRows: 1,
        role: SHEET_ROLE_SCORES,
        fp8Flag: isFp8Mode ? 1 : 0,
        highlight: idleHl,
        seed: 400 + e,
        partId: 'inactive-experts',
        headIndex: e,
        tensorKey: 'router_bank',
        tokenIndex: focusToken,
        atlasSlice: atlas.routerBankTiles[e],
        cellFamily: CELL_NEUTRAL,
        reveal: sheetReveal('router_bank', { expertId: e }),
      });
      bankTileGeom.set(e, idleGeom);
    }
  }

  // Parallel Fork Ribbons from RMSNorm U -> Left Routed Path
  const routedForkColor = hasCellTrace
    ? RIBBON_DIMMED
    : step === 1 || step === 2 || wRouterHl === SHEET_HL_ACTIVE
      ? RIBBON_ACTIVE
      : hasPartFocus
        ? RIBBON_DIMMED
        : RIBBON_PATHWAY_SOFT;
  b.flow.addCubicXZ(
    { x: rmsX - 0.42, z: zTop + seqDepth * 0.5 + 0.06 },
    { x: rmsX - 0.80, z: (zTop + zRouter - bankD * 0.5) * 0.5 },
    { x: routedCenterX, z: (zTop + zRouter - bankD * 0.5) * 0.5 },
    { x: routedCenterX, z: bankTop - 0.14 },
    0.008,
    0.13,
    routedForkColor,
    true,
    22,
  );
  b.flow.addPolylineXZ(
    [
      { x: wRouterX + wRouterW * 0.5 + 0.06, z: zRouter },
      { x: bankLeft - 0.14, z: zRouter },
    ],
    0.008,
    0.12,
    !hasCellTrace && (wRouterHl === SHEET_HL_ACTIVE || step === 1)
      ? RIBBON_ACTIVE
      : hasAnyFocus
        ? RIBBON_DIMMED
        : RIBBON_QUIET,
    true,
    0.20,
    0.20,
  );

  // ==========================================================================
  // 4. TOP-6 WEIGHTS (×2.5) SHEET
  // ==========================================================================
  const topkW = 4.32;
  const topkD = 0.66;
  const topkHl = sheetHighlight(
    { tensor: 'topk_weights', tokenIndex: focusToken, isAggregate: true },
    ['router', 'topk-weights'],
    [2],
    isFp8Mode ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
  );
  b.addSheet({
    cx: routedCenterX,
    cy: sheetY(topkHl),
    cz: zTopk,
    ex: topkW,
    ey: SHEET_T,
    ez: topkD,
    gridCols: sim.topK,
    gridRows: 1,
    role: SHEET_ROLE_SOFTMAX,
    fp8Flag: 0, // Always FP32 router weights/probs
    highlight: topkHl,
    seed: 41,
    partId: 'router',
    tensorKey: 'topk_weights',
    tokenIndex: focusToken,
    atlasSlice: atlas.topkWeights,
    registryKey: 'topk_weights',
    cellFamily: CELL_MOE_AMBER,
    reveal: sheetReveal('topk_weights'),
  });

  // 6 smooth fan-in ribbons from the 6 active expert tiles -> Top-6 Weights slots
  for (let j = 0; j < sim.topK; j++) {
    const eid = activeExpertIds[j];
    const tile = bankTileGeom.get(eid);
    if (!tile) continue;
    const slotX = routedCenterX - topkW * 0.5 + (j + 0.5) * (topkW / sim.topK);
    const isSlotFocused = !hasCellTrace && (eid === focusExpert || step === 2);
    b.flow.addCubicXZ(
      { x: tile.cx, z: tile.cz + tile.ez * 0.5 + 0.04 },
      { x: tile.cx, z: (tile.cz + zTopk) * 0.5 + 0.18 },
      { x: slotX, z: (bankTop + bankD + zTopk) * 0.5 },
      { x: slotX, z: zTopk - topkD * 0.5 - 0.05 },
      isSlotFocused ? 0.014 : 0.008,
      isSlotFocused ? 0.072 : 0.044,
      isSlotFocused ? RIBBON_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_PATHWAY_SOFT,
      true,
      18,
    );
  }

  // ==========================================================================
  // 5. FOCUSED ROUTED EXPERT DETAIL (Non-Gated ReLU²: W_up -> h -> ReLU² -> W_down)
  // ==========================================================================
  const wUpX = -5.05;
  const preX = -3.35;
  const actX = -2.25;
  const wDownX = -0.55;

  const wUpW = 1.68;
  const wUpD = 0.92; // proportional to d_exp = 4 (shared W_up is 1.84 = exact 2×!)
  const vecW = 0.48;
  const vecD = 0.92;
  const wDownW = 1.04; // proportional to d_exp = 4 (shared W_down is 2.08 = exact 2×!)
  const wDownD = 1.68;

  const detailIdleHl = focusedEval.isActive ? SHEET_HL_NORMAL : SHEET_HL_DIMMED;
  const expertPartMatch = activePart === `moe-expert-${focusExpert}`;
  const wUpHl = sheetHighlight(
    { tensor: 'routed_w_up', expertId: focusExpert, tokenIndex: focusToken, isAggregate: false },
    expertPartMatch ? ['routed-w-up', activePart!] : ['routed-w-up'],
    [3],
    detailIdleHl,
  );
  const preHl = sheetHighlight(
    { tensor: 'routed_pre', expertId: focusExpert, tokenIndex: focusToken, isAggregate: false },
    expertPartMatch ? ['routed-pre', activePart!] : ['routed-pre'],
    [3],
    detailIdleHl,
  );
  const actHl = sheetHighlight(
    { tensor: 'routed_act', expertId: focusExpert, tokenIndex: focusToken, isAggregate: false },
    expertPartMatch ? ['routed-act', activePart!] : ['routed-act'],
    [3],
    detailIdleHl,
  );
  const wDownHl = sheetHighlight(
    { tensor: 'routed_w_down', expertId: focusExpert, tokenIndex: focusToken, isAggregate: false },
    expertPartMatch ? ['routed-w-down', activePart!] : ['routed-w-down'],
    [3],
    detailIdleHl,
  );

  b.addDashedFrameXZ(
    wUpX - wUpW * 0.5 - 0.20,
    zDetail - wDownD * 0.5 - 0.22,
    wDownX + wDownW * 0.5 + 0.20,
    zDetail + wDownD * 0.5 + 0.20,
    0.004,
    focusedEval.isActive ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
    focusedEval.isActive ? 0.44 : 0.36,
  );

  b.addSheet({
    cx: wUpX,
    cy: sheetY(wUpHl),
    cz: zDetail,
    ex: wUpW,
    ey: SHEET_T,
    ez: wUpD,
    gridCols: sim.d,
    gridRows: sim.dExp,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: isFp8Mode ? 1 : 0,
    highlight: wUpHl,
    seed: 51,
    partId: 'routed-w-up',
    headIndex: focusExpert,
    tensorKey: 'routed_w_up',
    tokenIndex: focusToken,
    atlasSlice: atlas.routedWUp,
    registryKey: 'routed_w_up',
    cellFamily: CELL_WEIGHT_BLUE,
    reveal: sheetReveal('weight'),
  });

  b.addSheet({
    cx: preX,
    cy: sheetY(preHl),
    cz: zDetail,
    ex: vecW,
    ey: SHEET_T,
    ez: vecD,
    gridCols: 1,
    gridRows: sim.dExp,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: preHl,
    seed: 53,
    partId: 'routed-pre',
    headIndex: focusExpert,
    tensorKey: 'routed_pre',
    tokenIndex: focusToken,
    atlasSlice: atlas.routedPre,
    registryKey: 'routed_pre',
    cellFamily: CELL_ACTIVATION_GREEN,
    reveal: sheetReveal('routed_pre'),
  });

  b.addSheet({
    cx: actX,
    cy: sheetY(actHl),
    cz: zDetail,
    ex: vecW,
    ey: SHEET_T,
    ez: vecD,
    gridCols: 1,
    gridRows: sim.dExp,
    role: SHEET_ROLE_SOFTMAX,
    fp8Flag: 0,
    highlight: actHl,
    seed: 57,
    partId: 'routed-act',
    headIndex: focusExpert,
    tensorKey: 'routed_act',
    tokenIndex: focusToken,
    atlasSlice: atlas.routedAct,
    registryKey: 'routed_act',
    cellFamily: CELL_MOE_AMBER,
    reveal: sheetReveal('routed_act'),
  });

  b.addSheet({
    cx: wDownX,
    cy: sheetY(wDownHl),
    cz: zDetail,
    ex: wDownW,
    ey: SHEET_T,
    ez: wDownD,
    gridCols: sim.dExp,
    gridRows: sim.d,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: isFp8Mode ? 1 : 0,
    highlight: wDownHl,
    seed: 59,
    partId: 'routed-w-down',
    headIndex: focusExpert,
    tensorKey: 'routed_w_down',
    tokenIndex: focusToken,
    atlasSlice: atlas.routedWDown,
    registryKey: 'routed_w_down',
    cellFamily: CELL_WEIGHT_BLUE,
    reveal: sheetReveal('weight'),
  });

  // Horizontal chain ribbons inside Focused Routed Expert: W_up -> Pre-Act h -> ReLU² -> W_down
  const chainCol =
    hasCellTrace || !focusedEval.isActive
      ? RIBBON_DIMMED
      : step === 3 ||
          wUpHl === SHEET_HL_ACTIVE ||
          preHl === SHEET_HL_ACTIVE ||
          actHl === SHEET_HL_ACTIVE ||
          wDownHl === SHEET_HL_ACTIVE
        ? RIBBON_ACTIVE
        : hasPartFocus
          ? RIBBON_DIMMED
          : RIBBON_PATHWAY_SOFT;
  b.flow.addPolylineXZ(
    [
      { x: wUpX + wUpW * 0.5 + 0.05, z: zDetail },
      { x: preX - vecW * 0.5 - 0.05, z: zDetail },
    ],
    0.009,
    0.10,
    chainCol,
    true,
    0.16,
    0.16,
  );
  b.flow.addPolylineXZ(
    [
      { x: preX + vecW * 0.5 + 0.05, z: zDetail },
      { x: actX - vecW * 0.5 - 0.05, z: zDetail },
    ],
    0.009,
    0.10,
    chainCol,
    true,
    0.16,
    0.16,
  );
  b.flow.addPolylineXZ(
    [
      { x: actX + vecW * 0.5 + 0.05, z: zDetail },
      { x: wDownX - wDownW * 0.5 - 0.05, z: zDetail },
    ],
    0.009,
    0.10,
    chainCol,
    true,
    0.16,
    0.16,
  );

  // Top-6 Weights -> Focused Expert chain connector
  b.flow.addCubicXZ(
    { x: routedCenterX - 1.30, z: zTopk + topkD * 0.5 + 0.05 },
    { x: routedCenterX - 1.60, z: (zTopk + zDetail) * 0.5 },
    { x: wUpX, z: (zTopk + zDetail) * 0.5 - 0.15 },
    { x: wUpX, z: zDetail - wUpD * 0.5 - 0.10 },
    0.008,
    0.09,
    !hasCellTrace && (step === 2 || step === 3) ? RIBBON_ACTIVE : hasAnyFocus ? RIBBON_DIMMED : RIBBON_QUIET,
    true,
    18,
  );

  // ==========================================================================
  // 6. ROUTED SUM (Σ_{j=1..6} wⱼ·yⱼ)
  // ==========================================================================
  const routedSumHl = sheetHighlight({ tensor: 'routed_sum', isAggregate: true }, ['routed-sum'], [3, 5]);
  b.addSheet({
    cx: routedCenterX,
    cy: sheetY(routedSumHl),
    cz: zBranchOut,
    ex: 1.92,
    ey: SHEET_T,
    ez: seqDepth,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: routedSumHl,
    seed: 67,
    partId: 'routed-sum',
    tensorKey: 'routed_sum',
    atlasSlice: atlas.routedSum,
    registryKey: 'routed_sum',
    cellFamily: CELL_ACTIVATION_GREEN,
    reveal: sheetReveal('routed_sum'),
  });

  b.flow.addCubicXZ(
    { x: wDownX, z: zDetail + wDownD * 0.5 + 0.06 },
    { x: wDownX, z: (zDetail + zBranchOut) * 0.5 + 0.18 },
    { x: routedCenterX + 0.30, z: (zDetail + zBranchOut) * 0.5 + 0.10 },
    { x: routedCenterX + 0.30, z: zBranchOut - seqDepth * 0.5 - 0.08 },
    0.009,
    0.12,
    !hasCellTrace && (routedSumHl === SHEET_HL_ACTIVE || step === 3)
      ? RIBBON_ACTIVE
      : hasAnyFocus
        ? RIBBON_DIMMED
        : RIBBON_PATHWAY_SOFT,
    true,
    18,
  );

  // ==========================================================================
  // 7. RIGHT PARALLEL BRANCH: ALWAYS-ON DOUBLE-WIDTH SHARED EXPERT (3,712 = 2×)
  // ==========================================================================
  const zSharedUp = -2.70;
  const zSharedAct = -0.35;
  const zSharedDown = 1.85;

  const sharedWUpW = 1.68;
  const sharedWUpD = 1.84; // EXACT 2.0× routed W_up depth (0.92 -> 1.84)!
  const sharedActW = 0.54;
  const sharedActD = 1.44; // visibly 2× routed ReLU²!
  const sharedWDownW = 2.08; // EXACT 2.0× routed W_down width (1.04 -> 2.08)!
  const sharedWDownD = 1.68;

  b.addDashedFrameXZ(
    sharedX - sharedWDownW * 0.5 - 0.24,
    zSharedUp - sharedWUpD * 0.5 - 0.20,
    sharedX + sharedWDownW * 0.5 + 0.24,
    zBranchOut + seqDepth * 0.5 + 0.20,
    0.004,
    LINE_STYLE_GUIDE,
    0.48,
  );

  const sharedUpHl = sheetHighlight({ tensor: 'shared_w_up', isAggregate: true }, ['shared-expert'], [4]);
  const sharedActHl = sheetHighlight(
    { tensor: 'shared_act', tokenIndex: focusToken, isAggregate: false },
    ['shared-act', 'shared-expert'],
    [4],
  );
  const sharedDownHl = sheetHighlight({ tensor: 'shared_w_down', isAggregate: true }, ['shared-w-down', 'shared-expert'], [4]);
  const sharedOutHl = sheetHighlight({ tensor: 'shared_out', isAggregate: true }, ['shared-out', 'shared-expert'], [4, 5]);

  b.addSheet({
    cx: sharedX,
    cy: sheetY(sharedUpHl),
    cz: zSharedUp,
    ex: sharedWUpW,
    ey: SHEET_T,
    ez: sharedWUpD,
    gridCols: sim.d,
    gridRows: sim.dShared,
    role: SHEET_ROLE_KV_K,
    fp8Flag: isFp8Mode ? 1 : 0,
    highlight: sharedUpHl,
    seed: 71,
    partId: 'shared-expert',
    tensorKey: 'shared_w_up',
    atlasSlice: atlas.sharedWUp,
    registryKey: 'shared_w_up',
    cellFamily: CELL_SSM_CYAN,
    reveal: sheetReveal('weight'),
  });

  b.addSheet({
    cx: sharedX,
    cy: sheetY(sharedActHl),
    cz: zSharedAct,
    ex: sharedActW,
    ey: SHEET_T,
    ez: sharedActD,
    gridCols: 1,
    gridRows: sim.dShared,
    role: SHEET_ROLE_KV_V,
    fp8Flag: 0,
    highlight: sharedActHl,
    seed: 73,
    partId: 'shared-act',
    tensorKey: 'shared_act',
    tokenIndex: focusToken,
    atlasSlice: atlas.sharedAct,
    registryKey: 'shared_act',
    cellFamily: CELL_SSM_CYAN,
    reveal: sheetReveal('shared_act'),
  });

  b.addSheet({
    cx: sharedX,
    cy: sheetY(sharedDownHl),
    cz: zSharedDown,
    ex: sharedWDownW,
    ey: SHEET_T,
    ez: sharedWDownD,
    gridCols: sim.dShared,
    gridRows: sim.d,
    role: SHEET_ROLE_KV_K,
    fp8Flag: isFp8Mode ? 1 : 0,
    highlight: sharedDownHl,
    seed: 79,
    partId: 'shared-w-down',
    tensorKey: 'shared_w_down',
    atlasSlice: atlas.sharedWDown,
    registryKey: 'shared_w_down',
    cellFamily: CELL_SSM_CYAN,
    reveal: sheetReveal('weight'),
  });

  b.addSheet({
    cx: sharedX,
    cy: sheetY(sharedOutHl),
    cz: zBranchOut,
    ex: 1.92,
    ey: SHEET_T,
    ez: seqDepth,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_KV_V,
    fp8Flag: 0,
    highlight: sharedOutHl,
    seed: 83,
    partId: 'shared-out',
    tensorKey: 'shared_out',
    atlasSlice: atlas.sharedOut,
    registryKey: 'shared_out',
    cellFamily: CELL_SSM_CYAN,
    reveal: sheetReveal('shared_out'),
  });

  // Right parallel fork from RMSNorm U -> Shared W_up -> Shared ReLU² -> Shared W_down -> Shared Out
  const sharedFlowCol = hasCellTrace
    ? RIBBON_DIMMED
    : step === 4 ||
        sharedUpHl === SHEET_HL_ACTIVE ||
        sharedActHl === SHEET_HL_ACTIVE ||
        sharedDownHl === SHEET_HL_ACTIVE ||
        sharedOutHl === SHEET_HL_ACTIVE
      ? RIBBON_ACTIVE
      : hasPartFocus
        ? RIBBON_DIMMED
        : RIBBON_SSM_LOOP;

  b.flow.addCubicXZ(
    { x: rmsX + 0.42, z: zTop + seqDepth * 0.5 + 0.06 },
    { x: rmsX + 0.90, z: (zTop + zSharedUp - sharedWUpD * 0.5) * 0.5 },
    { x: sharedX, z: (zTop + zSharedUp - sharedWUpD * 0.5) * 0.5 },
    { x: sharedX, z: zSharedUp - sharedWUpD * 0.5 - 0.08 },
    0.008,
    0.13,
    sharedFlowCol,
    true,
    22,
  );
  b.flow.addPolylineXZ(
    [
      { x: sharedX, z: zSharedUp + sharedWUpD * 0.5 + 0.05 },
      { x: sharedX, z: zSharedAct - sharedActD * 0.5 - 0.05 },
    ],
    0.008,
    0.11,
    sharedFlowCol,
    true,
    0.18,
    0.18,
  );
  b.flow.addPolylineXZ(
    [
      { x: sharedX, z: zSharedAct + sharedActD * 0.5 + 0.05 },
      { x: sharedX, z: zSharedDown - sharedWDownD * 0.5 - 0.05 },
    ],
    0.008,
    0.11,
    sharedFlowCol,
    true,
    0.18,
    0.18,
  );
  b.flow.addPolylineXZ(
    [
      { x: sharedX, z: zSharedDown + sharedWDownD * 0.5 + 0.05 },
      { x: sharedX, z: zBranchOut - seqDepth * 0.5 - 0.06 },
    ],
    0.008,
    0.11,
    sharedFlowCol,
    true,
    0.18,
    0.18,
  );

  // ==========================================================================
  // 8. BOTTOM PARALLEL MERGE: Routed Sum + Shared Out -> MoE Out -> + Residual
  // ==========================================================================
  const moeOutX = 1.65;
  const moeOutHl = sheetHighlight({ tensor: 'moe_out', isAggregate: true }, ['moe-out', 'merge-sum'], [5]);
  b.addSheet({
    cx: moeOutX,
    cy: sheetY(moeOutHl),
    cz: zBottom,
    ex: 1.92,
    ey: SHEET_T,
    ez: seqDepth,
    gridCols: sim.d,
    gridRows: clampedT,
    role: SHEET_ROLE_ACTIVATION,
    fp8Flag: 0,
    highlight: moeOutHl,
    seed: 89,
    partId: 'moe-out',
    tensorKey: 'moe_out',
    atlasSlice: atlas.moeOut,
    registryKey: 'moe_out',
    cellFamily: CELL_ACTIVATION_GREEN,
    reveal: sheetReveal('moe_out'),
  });

  const mergeCol = hasCellTrace
    ? RIBBON_DIMMED
    : moeOutHl === SHEET_HL_ACTIVE || resOutHl === SHEET_HL_ACTIVE || step === 5
      ? RIBBON_ACTIVE
      : hasPartFocus
        ? RIBBON_DIMMED
        : RIBBON_PATHWAY_SOFT;
  b.flow.addCubicXZ(
    { x: routedCenterX, z: zBranchOut + seqDepth * 0.5 + 0.06 },
    { x: routedCenterX, z: (zBranchOut + zBottom) * 0.5 + 0.25 },
    { x: moeOutX - 0.40, z: (zBranchOut + zBottom) * 0.5 },
    { x: moeOutX - 0.40, z: zBottom - seqDepth * 0.5 - 0.08 },
    0.008,
    0.13,
    mergeCol,
    true,
    20,
  );
  b.flow.addCubicXZ(
    { x: sharedX, z: zBranchOut + seqDepth * 0.5 + 0.06 },
    { x: sharedX, z: (zBranchOut + zBottom) * 0.5 + 0.25 },
    { x: moeOutX + 0.40, z: (zBranchOut + zBottom) * 0.5 },
    { x: moeOutX + 0.40, z: zBottom - seqDepth * 0.5 - 0.08 },
    0.008,
    0.13,
    hasCellTrace
      ? RIBBON_DIMMED
      : moeOutHl === SHEET_HL_ACTIVE || step === 5
        ? RIBBON_ACTIVE
        : hasPartFocus
          ? RIBBON_DIMMED
          : RIBBON_SSM_LOOP,
    true,
    20,
  );

  // MoE Out -> + Residual horizontal merge
  b.flow.addPolylineXZ(
    [
      { x: moeOutX + 0.96 + 0.06, z: zBottom },
      { x: resX - 0.72 - 0.08, z: zBottom },
    ],
    0.009,
    0.14,
    hasCellTrace
      ? RIBBON_DIMMED
      : resOutHl === SHEET_HL_ACTIVE || step === 5
        ? RIBBON_RESIDUAL_ACTIVE
        : hasPartFocus
          ? RIBBON_DIMMED
          : RIBBON_RESIDUAL,
    true,
    0.22,
    0.22,
  );

  // ==========================================================================
  // 9. CELL-LEVEL HIGHLIGHT & DEPENDENCY THREADS (Synced to Inspected Cell)
  // ==========================================================================
  const resolveSpanRect = (span: MoeSheetSpan): WorldSpanRect | null => {
    if (span.tensor === 'router_bank') {
      if (span.tokenIndex !== undefined && span.tokenIndex !== focusToken) {
        return null;
      }
      const eid =
        span.expertId ??
        Math.max(
          0,
          Math.min(
            sim.nRoutedExperts - 1,
            span.rowStart * MOE_SIM_CONFIG.bankCols + span.colStart,
          ),
        );
      const tile = bankTileGeom.get(eid);
      if (!tile) return null;
      return {
        cx: tile.cx,
        cz: tile.cz,
        ex: tile.ex,
        ez: tile.ez,
        topY: tile.cy + tile.ey * 0.5,
      };
    }
    if (
      span.tensor === 'routed_w_up' ||
      span.tensor === 'routed_pre' ||
      span.tensor === 'routed_act' ||
      span.tensor === 'routed_w_down'
    ) {
      if (span.expertId !== undefined && span.expertId !== focusExpert) {
        return null;
      }
      if (span.tokenIndex !== undefined && span.tokenIndex !== focusToken) {
        return null;
      }
    } else if (span.tensor === 'topk_weights' || span.tensor === 'shared_act') {
      if (span.tokenIndex !== undefined && span.tokenIndex !== focusToken) {
        return null;
      }
    }
    const geom = b.sheetRegistry.get(span.tensor);
    if (!geom) return null;
    return rectFromSheetGrid(geom, span.rowStart, span.rowEnd, span.colStart, span.colEnd);
  };

  let popoverTargetWorld: readonly [number, number, number] | null = null;
  const popoverSources: {
    readonly world: readonly [number, number, number];
    readonly role: 'indigo' | 'teal';
  }[] = [];

  if (activeFormula) {
    const c = activeFormula.cell;
    const targetRect = resolveSpanRect({
      tensor: c.tensor,
      rowStart: c.row,
      rowEnd: c.row,
      colStart: c.col,
      colEnd: c.col,
      expertId: c.expertId,
      tokenIndex: c.tokenIndex,
    });
    if (targetRect) {
      popoverTargetWorld = [targetRect.cx, targetRect.topY, targetRect.cz];
      b.flow.addCellHighlightXZ(
        targetRect.cx,
        targetRect.cz,
        targetRect.ex,
        targetRect.ez,
        targetRect.topY + 0.008,
        CELL_TARGET_FILL,
        CELL_TARGET_FRAME,
        0.026,
      );
      activeFormula.upstreamSpans.forEach((up, idx) => {
        const upRect = resolveSpanRect(up);
        if (!upRect) return;
        popoverSources.push({
          world: [upRect.cx, upRect.topY, upRect.cz],
          role: idx === 0 ? 'indigo' : 'teal',
        });
        drawOperandDependencyXZ(b.flow, upRect, targetRect, idx === 0);
      });
      // Refinement 2: In Near-Plan too, show all 5 other Top-6 unbiased denominator contributors on topk_weights!
      if (c.tensor === 'topk_weights') {
        const slot = Math.max(0, Math.min(sim.topK - 1, c.rankSlot ?? c.col));
        const winnerId = sim.topk_indices[focusToken * sim.topK + slot];
        let addedDenomSource = false;
        for (const eid of activeExpertIds) {
          if (eid === winnerId) continue;
          const denomRect = resolveSpanRect({
            tensor: 'router_bank',
            rowStart: Math.floor(eid / MOE_SIM_CONFIG.bankCols),
            rowEnd: Math.floor(eid / MOE_SIM_CONFIG.bankCols),
            colStart: eid % MOE_SIM_CONFIG.bankCols,
            colEnd: eid % MOE_SIM_CONFIG.bankCols,
            expertId: eid,
            tokenIndex: focusToken,
          });
          if (denomRect) {
            if (!addedDenomSource) {
              popoverSources.push({
                world: [denomRect.cx, denomRect.topY, denomRect.cz],
                role: 'teal',
              });
              addedDenomSource = true;
            }
            drawOperandDependencyXZ(b.flow, denomRect, targetRect, false);
          }
        }
      }
    }
  }

  // ==========================================================================
  // 10. NON-OCCLUDING EXTERIOR LABELS & CALIPERS
  // ==========================================================================
  b.landmarks.push(
    {
      id: 'moe-input',
      label: `Input (2,688 · T=${clampedT})`,
      world: [resX, 0.08, zTop - seqDepth * 0.5 - 0.22],
      partId: 'residual-in',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-rms',
      label: 'RMSNorm (ε=1e−5)',
      world: [rmsX, 0.08, zTop - seqDepth * 0.5 - 0.22],
      partId: 'rmsnorm',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-wrouter',
      label: isFp8Mode ? 'W_router · FP32' : 'W_router (128×2,688)',
      world: [wRouterX, 0.08, zRouter - wRouterD * 0.5 - 0.22],
      partId: 'w-router',
      align: 'top',
      tier: 'primary',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    {
      id: 'moe-bank',
      label: `Router Bank (8×16 = 128 · Top-6 · t${absFocusToken} "${focusTokLabel}")`,
      world: [routedCenterX - 0.52, 0.08, bankTop - 0.28],
      partId: 'routed-bank',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-topk',
      label: 'Top-6 Weights (×2.5)',
      world: [routedCenterX - topkW * 0.5 - 0.18, 0.08, zTopk],
      partId: 'router',
      align: 'left',
      tier: 'primary',
      tone: isFp8Mode ? 'gold' : 'default',
    },
    {
      id: 'moe-routed-up',
      label: isFp8Mode
        ? `${focusExpTag} W_up (1,856) · FP8`
        : `${focusExpTag} W_up (1,856)`,
      world: [wUpX, 0.08, zDetail - wUpD * 0.5 - 0.22],
      partId: 'routed-w-up',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-routed-act',
      label: 'ReLU² (No gate_proj)',
      world: [(preX + actX) * 0.5, 0.08, zDetail - vecD * 0.5 - 0.34],
      partId: 'routed-act',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-routed-down',
      label: isFp8Mode ? 'W_down · FP8' : 'W_down (2,688)',
      world: [wDownX, 0.08, zDetail - wDownD * 0.5 - 0.22],
      partId: 'routed-w-down',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'moe-routed-sum',
      label: 'Routed Sum (Σ wᵢ·yᵢ)',
      world: [routedCenterX - 0.96 - 0.18, 0.08, zBranchOut],
      partId: 'routed-sum',
      align: 'left',
      tier: 'primary',
    },
    {
      id: 'moe-shared-up',
      label: isFp8Mode
        ? 'Shared W_up (3,712 = 2×) · FP8'
        : 'Shared W_up (3,712 = 2×)',
      world: [sharedX + 0.38, 0.08, zSharedUp - sharedWUpD * 0.5 - 0.26],
      partId: 'shared-expert',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'moe-shared-act',
      label: 'Shared ReLU² (2×)',
      world: [sharedX + sharedActW * 0.5 + 0.18, 0.08, zSharedAct],
      partId: 'shared-act',
      align: 'right',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'moe-shared-down',
      label: isFp8Mode ? 'Shared W_down (2×) · FP8' : 'Shared W_down (2×)',
      world: [sharedX, 0.08, zSharedDown - sharedWDownD * 0.5 - 0.20],
      partId: 'shared-w-down',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'moe-shared-out',
      label: 'Shared Out (1× Unscaled)',
      world: [sharedX + 0.96 + 0.18, 0.08, zBranchOut],
      partId: 'shared-out',
      align: 'right',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'moe-merge',
      label: 'MoE Out (Routed + Shared)',
      world: [moeOutX, 0.08, zBottom + seqDepth * 0.5 + 0.26],
      partId: 'moe-out',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'moe-res-out',
      label: '+ Residual',
      world: [resX, 0.08, zBottom + seqDepth * 0.5 + 0.26],
      partId: 'residual-out',
      align: 'bottom',
      tier: 'primary',
    },
    // Detail zoom-revealed callout
    {
      id: 'moe-detail-status',
      label: focusedEval.isActive
        ? `${focusExpTag} ACTIVE (#${(focusedEval.activeRank ?? 0) + 1} · w=${focusedEval.appliedWeight.toFixed(2)})`
        : `${focusExpTag} IDLE (Rank #${focusedEval.choiceRank + 1}/128 · 0 contrib)`,
      world: [(wUpX + wDownX) * 0.5, 0.08, zDetail + wDownD * 0.5 + 0.30],
      partId: 'routed-w-up',
      align: 'bottom',
      tier: 'detail',
      tone: focusedEval.isActive ? 'default' : 'muted',
    },
  );

  b.calipers.push({
    id: 'cal-shared-2x',
    aWorld: [sharedX - sharedWDownW * 0.5, 0.02, zSharedDown + sharedWDownD * 0.5 + 0.16],
    bWorld: [sharedX + sharedWDownW * 0.5, 0.02, zSharedDown + sharedWDownD * 0.5 + 0.16],
    label: '3,712 = 2× routed (1,856)',
    partIds: ['shared-expert', 'shared-w-down', 'shared-act', 'shared-out'],
  });

  const focusedRank = activeRankMap.get(focusExpert) ?? null;
  const activeOpPartIds = buildMoeActiveOpPartIds(focusSet, activeFormula, activeRankMap);
  const cellPopover = buildMoeCellPopover({
    activeFormula,
    sim,
    focusToken,
    focusExpert,
    targetWorld: popoverTargetWorld,
    sourceWorlds: popoverSources,
    focusedAccumulation,
  });
  const activeCellGrids = buildMoeActiveCellGrids({
    sheetRegistry: b.sheetRegistry,
    sim,
    focusToken,
    focusExpert,
    activeExpertIds,
    focusedRank,
    focusedEval,
    activeFormula,
    revealCtx,
    focusedAccumulation,
  });

  const packed = b.pack();
  const reg = (k: string) => b.sheetRegistry.get(k);
  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('input'), reg('rmsnorm')], 0.50, 0.12, 0.50, MOE_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('rmsnorm'), reg('w_router'), bankEnvelope], 0.50, 0.12, 0.50, MOE_FIT_BOUNDS),
    2: computeSheetGroupBounds([bankEnvelope, reg('topk_weights')], 0.50, 0.12, 0.50, MOE_FIT_BOUNDS),
    3: computeSheetGroupBounds([reg('routed_w_up'), reg('routed_pre'), reg('routed_act'), reg('routed_w_down'), reg('routed_sum')], 0.50, 0.12, 0.50, MOE_FIT_BOUNDS),
    4: computeSheetGroupBounds([reg('shared_w_up'), reg('shared_act'), reg('shared_w_down'), reg('shared_out')], 0.50, 0.12, 0.50, MOE_FIT_BOUNDS),
    5: computeSheetGroupBounds([reg('routed_sum'), reg('shared_out'), reg('moe_out'), reg('residual')], 0.50, 0.12, 0.50, MOE_FIT_BOUNDS),
  };

  return {
    kind: 'cutaway_moe',
    bayId,
    ...packed,
    calipers: b.calipers,
    valueAtlas: atlas,
    moeSim: sim,
    activeFormula,
    activeOpPartIds,
    cellPopover,
    activeCellGrids,
    focusedAccumulation,
    fitBounds: MOE_FIT_BOUNDS,
    opFocusBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    solidPickBoxes: b.pickBoxes,
    partCatalog,
    moeStats,
  };
}
