/**
 * Dedicated Timestep Conditioning & Split AdaLN-Single Spatial Cutaway
 * (`src/twotower/scene/timestepCutaway.ts`)
 *
 * Pure headless TypeScript (zero DOM, WebGL2, or `Math.random()`) visualizing EVERY stage of
 * Nemotron-Labs-TwoTower's denoiser-only PixArt-alpha `adaLN-single` conditioning chain
 * (`src/twotower/modulation.ts`) on the shared 3D thin-sheet editorial drafting table:
 *
 *   1. Masked Fraction `t = nMasked / B` (explicitly BYPASSED in Context AR / Mock-AR with `t = null`).
 *   2. Sinusoidal Timestep Embedding `φ(t)` (`[16]` FP32, `t * 1000`, COS-FIRST `[cos(0..7); sin(0..7)]`).
 *   3. Timestep MLP (`W_0 [8×16]` → `SiLU` → `W_2 [8×8]` → `tRepr [8]`).
 *   4. Global `3d` Modulation Projection (`SiLU(tRepr)` → `W_time_block [24×8]` → `tEmb [24]`).
 *   5. Per-Layer Learned `scale_shift_tables[layerIndex]` (`[24]`) + elementwise sum → `(shift, scale, gate)`.
 *   6. Selected Layer's Actual Modulation Application (pulled from real `denoiserPass.getLayerTrace(bayId)`):
 *      - `'M'` (Mamba-2) & `'*'` (GQA): `modulate_then_norm` (`X → X_mod → RMSNorm(X_mod) → Mixer → gate ⊙ out → + X`)
 *      - `'E'` (Granular MoE): `norm_then_modulate` (`X → RMSNorm(X) → U_mod → MoE → gate ⊙ out → + X`)
 *      - Dedicated untouched emerald residual bypass highway (`X`) separate from modulated mixer input
 *      - Explicit contrast between outer channel-wise AdaLN residual gate (`d = 8`) vs Mamba internal `SiLU(z)` gate (`d_inner = 64`)
 *      - Explicit reference callout confirming `norm_f` (Final RMSNorm) is strictly unmodulated.
 */

import type { Kind, Vec3 } from '../../model/spec';
import {
  DEFAULT_FOV_DEG,
  MAX_CAMERA_DISTANCE,
  MIN_CAMERA_DISTANCE,
  type CameraPose,
} from '../../model/state';
import {
  aabbCorners,
  computeCameraBasis,
  dot3,
  type AABB,
  type ViewportMetrics,
} from '../../render/camera';
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  pushDashedFrameXZ,
  pushFreeGuideLine,
} from '../../render/linePacking';
import type { PickBox, PickHit } from '../../render/pick';
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
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_GHOST_RAIL,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NEUTRAL,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
} from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  computeSheetGroupBounds,
  createSheetFocusSet,
  drawCellHighlight3D,
  drawOperandDependency3D,
  EMPTY_SHEET_FOCUS_SET,
  fmtSigned,
  resolveSheetHighlight,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  RIBBON_RESIDUAL_ACTIVE,
  spanFromSheetGrid,
  type LayerFormulaReadout,
  type SheetFocusEntry,
  type SheetFocusSet,
  type SheetGeomRecord,
  type StepperStepDef,
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
  type LandmarkAnchor,
} from '../../scene/spine';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../ui/overlay';
import type { TwoTowerTraceOrigin } from '../contracts';
import type { TwoTowerDenoiserLayerTrace } from '../execution';
import {
  DEFAULT_TIMESTEP_MAX_PERIOD,
  DEFAULT_TIMESTEP_MAX_PERIOD_SCALE,
  modulationOrderForBayKind,
  type ModulationOrder,
  type TimeConditioningWeights,
} from '../modulation';
import type { TwoTowerSession } from '../session';
import {
  buildTwoTowerSelectionIdentity,
  getTwoTowerBayKind,
  resolveTwoTowerSessionViewState,
  type TwoTowerDenoiserPassLookup,
  type TwoTowerProvenanceBadge,
  type TwoTowerSpatialScene,
  type TwoTowerTimestepConditioningMode,
  type TwoTowerTimestepStatus,
} from './provenance';

export type TimestepTensorKey =
  | 't_scalar'
  | 'phi'
  | 'mlp0_w'
  | 'mlp_h'
  | 'mlp2_w'
  | 't_repr'
  | 't_block_w'
  | 't_emb'
  | 'scale_shift_table'
  | 'layer_mod'
  | 'residual_x'
  | 'stage1'
  | 'stage2'
  | 'mixer_out'
  | 'gated_delta'
  | 'residual_y';

export interface TimestepCellRef {
  readonly tensor: TimestepTensorKey;
  readonly row: number;
  readonly col: number;
}

export interface BuildTimestepCutawayOptions {
  /** Selected 1-based layer bay ID (`1..52`); defaults to `6` (GQA Bay 06). */
  readonly bayId?: number;
  /** Local block/sequence token index to inspect on the selected layer application track (defaults to `0`). */
  readonly localTokenIndex?: number;
  readonly selectedPartId?: string | null;
  readonly hoveredPartId?: string | null;
  readonly selectedCell?: TimestepCellRef | null;
  /** Optional 6-stage math stepper stage (`0..5`). */
  readonly stepperStep?: number | null;
}

export interface TimestepCameraFitOptions {
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly isNarrow?: boolean;
  readonly azimuthDeg?: number;
  readonly elevationDeg?: number;
}

export interface TimestepCutawayDiagnostics {
  readonly bypassed: boolean;
  readonly inactiveNoPassYet: boolean;
  readonly conditioningMode: TwoTowerTimestepConditioningMode;
  readonly timestepStatus: TwoTowerTimestepStatus;
  readonly bypassReason: string | null;
  readonly statusLabel: string;
  readonly tModel: number | null;
  readonly nMasked: number | null;
  readonly blockSize: number | null;
  readonly empiricalMaskFraction: number | null;
  readonly isDiagnosticOverride: boolean;
  readonly overrideValue: number | null;
  readonly isHistoricalDenoiserInspection: boolean;
  readonly matchesCurrentContextBoundary: boolean;
  readonly sourceRecordIndex: number | null;
  readonly sourceBlockId: number | null;
  readonly sourceRefinementIndex: number | null;
  readonly sourceContextPrefixLen: number | null;
  readonly sourceContextCacheDigest: string | null;
  readonly historicalProvenanceWarning: string | null;
  readonly bayId: number;
  readonly layerIndex: number;
  readonly mixerKind: Kind;
  readonly modulationOrder: ModulationOrder;
  readonly modulationOrderEquation: string;
  readonly shiftL2: number;
  readonly scaleL2: number;
  readonly gateL2: number;
  readonly outerGateDim: 8;
  readonly mambaInternalSiluGateDim: 64;
  readonly outerVsInternalGateNote: string;
  readonly finalNormModulated: false;
  readonly finalNormNote: string;
  readonly residualPreservedUntouched: true;
  readonly localTokenIndex: number;
  readonly absoluteTokenPosition: number | null;
  readonly tokenLabel: string;
  readonly hasLiveDenoiserTrace: boolean;
  readonly timeConditioningFingerprint: string;
  readonly phi: readonly number[] | null;
  readonly tRepr: readonly number[] | null;
  readonly tEmb: readonly number[] | null;
  readonly shift: readonly number[] | null;
  readonly scale: readonly number[] | null;
  readonly gate: readonly number[] | null;
  readonly scaleShiftTable: readonly number[];
}

export const TIMESTEP_SPATIAL_FIT_BOUNDS: AABB = {
  min: [-10.0, -0.06, -3.45],
  max: [10.6, 3.30, 3.45],
};

export const TIMESTEP_STEPPER_STEPS: readonly StepperStepDef[] = Object.freeze([
  {
    index: 0,
    shortLabel: '1. t = nM/B',
    title: '1. Masked Fraction t = nMasked / B',
    summary:
      'Active diffusion computes t = nMasked / B ∈ [0, 1]; Context AR and Mock-AR bypass timestep conditioning completely (t = null).',
  },
  {
    index: 1,
    shortLabel: '2. φ(t)',
    title: '2. Sinusoidal Embedding φ(t) [16]',
    summary:
      'Scales t × 1000 in FP32 with log-spaced frequencies (maxPeriod = 10000), COS-FIRST [cos(0..7); sin(0..7)].',
  },
  {
    index: 2,
    shortLabel: '3. MLP → tRepr',
    title: '3. Timestep MLP (W₀ → SiLU → W₂ → tRepr)',
    summary:
      'Two-layer SiLU MLP maps 16-dim sinusoidal embedding φ(t) into shared d = 8 timestep representation tRepr.',
  },
  {
    index: 3,
    shortLabel: '4. t_block [3d]',
    title: '4. Global 3d Projection (tEmb = [shift_g, scale_g, gate_g])',
    summary:
      'Applies SiLU(tRepr) then W_time_block [24×8] + b_blk to produce shared global shift, scale, and gate vectors.',
  },
  {
    index: 4,
    shortLabel: '5. Layer Table',
    title: '5. Per-Layer scale_shift_tables[l] + tEmb',
    summary:
      'Adds layer l’s learned 24-element table to global tEmb, producing this bay’s exact shift[8], scale[8], and outer gate[8].',
  },
  {
    index: 5,
    shortLabel: '6. Split AdaLN',
    title: '6. Asymmetric Modulation, Outer Gate & Untouched Residual X',
    summary:
      'Applies modulate→RMSNorm on Mamba/GQA vs RMSNorm→modulate on MoE, gates Mixer output by outer gate[8], and adds untouched residual X.',
  },
]);

/**
 * Exact 8-corner projective camera fitter for the 3D Timestep Conditioning Cutaway.
 */
export function fitTimestepCamera(
  metrics: ViewportMetrics,
  bounds: AABB = TIMESTEP_SPATIAL_FIT_BOUNDS,
  options?: TimestepCameraFitOptions,
): CameraPose {
  const isNarrow = options?.isNarrow ?? metrics.innerWidth < 680;
  const topInset = Math.max(0, options?.topInsetPx ?? (isNarrow ? 12 : 84));
  const bottomInset = Math.max(0, options?.bottomInsetPx ?? (isNarrow ? 12 : 118));
  const azimuthDeg = options?.azimuthDeg ?? (isNarrow ? 14 : 17);
  const elevationDeg = options?.elevationDeg ?? (isNarrow ? 26 : 24);
  const fovDeg = DEFAULT_FOV_DEG;
  const fillX = isNarrow ? 0.94 : 0.96;
  const fillY = isNarrow ? 0.90 : 0.93;

  const basis = computeCameraBasis(azimuthDeg, elevationDeg);
  const corners = aabbCorners(bounds);

  let minR = Infinity;
  let maxR = -Infinity;
  let minU = Infinity;
  let maxU = -Infinity;
  let minE = Infinity;
  let maxE = -Infinity;

  for (const p of corners) {
    const r = dot3(p, basis.right);
    const u = dot3(p, basis.up);
    const e = dot3(p, basis.eyeDir);
    if (r < minR) minR = r;
    if (r > maxR) maxR = r;
    if (u < minU) minU = u;
    if (u > maxU) maxU = u;
    if (e < minE) minE = e;
    if (e > maxE) maxE = e;
  }

  const midR = 0.5 * (minR + maxR);
  const midU = 0.5 * (minU + maxU);
  const midE = 0.5 * (minE + maxE);

  const safeW = Math.max(1, metrics.innerWidth);
  const safeH = Math.max(1, metrics.innerHeight);
  const clampedTop = Math.min(topInset, safeH * 0.42);
  const clampedBottom = Math.min(bottomInset, safeH * 0.42);
  const usableH = Math.max(80, safeH - clampedTop - clampedBottom);
  const usableFracY = Math.max(0.25, Math.min(1, usableH / safeH));

  const aspect = Math.max(0.25, safeW / safeH);
  const tanHalfFovY = Math.tan((fovDeg * 0.5) * (Math.PI / 180));
  const tanHalfFovX = tanHalfFovY * aspect;

  const ndcYCenter = (clampedBottom - clampedTop) / safeH;
  const ndcYHalfSpan = Math.max(0.15, fillY * usableFracY);
  const ndcYMax = ndcYCenter + ndcYHalfSpan;
  const ndcYMin = ndcYCenter - ndcYHalfSpan;

  let requiredDist = MIN_CAMERA_DISTANCE;
  for (const p of corners) {
    const dr = dot3(p, basis.right) - midR;
    const du = dot3(p, basis.up) - midU;
    const de = dot3(p, basis.eyeDir) - midE;

    const dX = de + Math.abs(dr) / (tanHalfFovX * fillX);
    const dYTop = (du / tanHalfFovY + de * ndcYMax) / ndcYHalfSpan;
    const dYBot = (-du / tanHalfFovY - de * ndcYMin) / ndcYHalfSpan;
    const dCorner = Math.max(dX, dYTop, dYBot, de + 2.0);
    if (dCorner > requiredDist) requiredDist = dCorner;
  }

  const distance = Math.max(MIN_CAMERA_DISTANCE, Math.min(MAX_CAMERA_DISTANCE, requiredDist));
  const targetShiftU = -ndcYCenter * distance * tanHalfFovY;

  const target: Vec3 = [
    midR * basis.right[0] + (midU + targetShiftU) * basis.up[0] + midE * basis.eyeDir[0],
    midR * basis.right[1] + (midU + targetShiftU) * basis.up[1] + midE * basis.eyeDir[1],
    midR * basis.right[2] + (midU + targetShiftU) * basis.up[2] + midE * basis.eyeDir[2],
  ];

  return {
    target,
    azimuthDeg,
    elevationDeg,
    distance,
    fovDeg,
  };
}

const TIMESTEP_TENSOR_KEYS = new Set<string>([
  't_scalar',
  'phi',
  'mlp0_w',
  'mlp_h',
  'mlp2_w',
  't_repr',
  't_block_w',
  't_emb',
  'scale_shift_table',
  'layer_mod',
  'residual_x',
  'stage1',
  'stage2',
  'mixer_out',
  'gated_delta',
  'residual_y',
]);

/** Resolve a ray-cast `PickHit` on the Timestep Conditioning cutaway into a `TimestepCellRef`. */
export function pickHitToTimestepCell(hit: PickHit | null): TimestepCellRef | null {
  if (
    !hit ||
    !hit.tensorKey ||
    !TIMESTEP_TENSOR_KEYS.has(hit.tensorKey) ||
    hit.hitRow === undefined ||
    hit.hitCol === undefined
  ) {
    return null;
  }
  return {
    tensor: hit.tensorKey as TimestepTensorKey,
    row: hit.hitRow,
    col: hit.hitCol,
  };
}

// ============================================================================
// PURE FLOAT32 INTERMEDIATE HELPERS (Matches src/twotower/modulation.ts bit-for-bit)
// ============================================================================

function siluF32(x: number): number {
  return Math.fround(x / (1.0 + Math.exp(-x)));
}

function l2Norm(vec: Float32Array): number {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) {
    sumSq += vec[i] * vec[i];
  }
  return Math.sqrt(sumSq);
}

function computeMlpHiddenAct(phi: Float32Array, w: TimeConditioningWeights): Float32Array {
  const { d, freqDim } = w.config;
  const out = new Float32Array(d);
  for (let r = 0; r < d; r++) {
    let acc = 0.0;
    const rowOff = r * freqDim;
    for (let c = 0; c < freqDim; c++) {
      acc += w.mlp0Weight[rowOff + c] * phi[c];
    }
    out[r] = siluF32(Math.fround(acc + w.mlp0Bias[r]));
  }
  return out;
}

function computeSinusoidalArgF32(t: number, k: number, freqDim: number): {
  readonly omega: number;
  readonly arg: number;
} {
  const half = freqDim / 2;
  const tScaledF32 = Math.fround(Math.fround(t) * Math.fround(DEFAULT_TIMESTEP_MAX_PERIOD_SCALE));
  const negLogMaxPeriodF32 = Math.fround(-Math.log(DEFAULT_TIMESTEP_MAX_PERIOD));
  const exponentF32 = Math.fround(Math.fround(negLogMaxPeriodF32 * k) / half);
  const omega = Math.fround(Math.exp(exponentF32));
  const arg = Math.fround(tScaledF32 * omega);
  return { omega, arg };
}

function bayKindDisplayName(kind: Kind): string {
  return kind === 'M' ? 'Mamba-2' : kind === '*' ? 'GQA Attention' : 'Granular MoE (128+1)';
}

interface LayerApplicationVectors {
  readonly x: Float32Array;
  readonly stage1: Float32Array;
  readonly stage2: Float32Array;
  readonly mixerOut: Float32Array;
  readonly gatedDelta: Float32Array;
  readonly y: Float32Array;
  readonly stage1Label: string;
  readonly stage2Label: string;
  readonly mixerOutLabel: string;
  readonly hasLiveTrace: boolean;
}

function extractTokenColumn(mat: Float32Array, d: number, seqLen: number, tok: number): Float32Array {
  const out = new Float32Array(d);
  const clampedTok = Math.max(0, Math.min(seqLen - 1, tok));
  for (let i = 0; i < d; i++) {
    out[i] = mat[i * seqLen + clampedTok];
  }
  return out;
}

function extractLayerApplicationVectors(params: {
  readonly d: number;
  readonly mixerKind: Kind;
  readonly modulationOrder: ModulationOrder;
  readonly bypassed: boolean;
  readonly trace: TwoTowerDenoiserLayerTrace | null;
  readonly fallbackTraceSim: { readonly X: Float32Array; readonly U: Float32Array; readonly Y: Float32Array; readonly tokens: number; readonly outVec: Float32Array } | null;
  readonly localTok: number;
  readonly gate: Float32Array | null;
}): LayerApplicationVectors {
  const { d, mixerKind, modulationOrder, bypassed, trace, fallbackTraceSim, localTok, gate } = params;
  const mixerOutLabel =
    mixerKind === 'M' ? 'Mamba_out' : mixerKind === '*' ? 'Attn_out' : 'MoE_out';
  const stage1Label =
    modulationOrder === 'modulate_then_norm'
      ? bypassed
        ? 'X (Bypassed)'
        : 'X_mod = X·(1+s)+sh'
      : 'U = RMSNorm(X)';
  const stage2Label =
    modulationOrder === 'modulate_then_norm'
      ? bypassed
        ? 'U = RMSNorm(X)'
        : 'U = RMSNorm(X_mod)'
      : bypassed
        ? 'U (Bypassed)'
        : 'U_mod = U·(1+s)+sh';

  if (trace !== null) {
    const T = trace.sim.tokens;
    const x = extractTokenColumn(trace.sim.X, d, T, localTok);
    const y = extractTokenColumn(trace.sim.Y, d, T, localTok);
    let stage1: Float32Array;
    let stage2: Float32Array;
    let mixerOut: Float32Array;

    if (trace.kind === 'M') {
      const sim = trace.sim;
      stage1 = sim.X_mod ? extractTokenColumn(sim.X_mod, d, T, localTok) : new Float32Array(x);
      stage2 = extractTokenColumn(sim.U, d, T, localTok);
      mixerOut = extractTokenColumn(sim.Mamba_out, d, T, localTok);
    } else if (trace.kind === '*') {
      const sim = trace.sim;
      stage1 = sim.X_mod ? extractTokenColumn(sim.X_mod, d, T, localTok) : new Float32Array(x);
      stage2 = extractTokenColumn(sim.U, d, T, localTok);
      mixerOut = extractTokenColumn(sim.Attn_out, d, T, localTok);
    } else {
      const sim = trace.sim;
      stage1 = extractTokenColumn(sim.U, d, T, localTok);
      stage2 = sim.U_mod ? extractTokenColumn(sim.U_mod, d, T, localTok) : new Float32Array(stage1);
      mixerOut = extractTokenColumn(sim.y_moe, d, T, localTok);
    }

    const gatedDelta = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      gatedDelta[i] =
        !bypassed && gate !== null ? Math.fround(gate[i] * mixerOut[i]) : mixerOut[i];
    }

    return {
      x,
      stage1,
      stage2,
      mixerOut,
      gatedDelta,
      y,
      stage1Label,
      stage2Label,
      mixerOutLabel,
      hasLiveTrace: true,
    };
  }

  if (fallbackTraceSim !== null) {
    const T = fallbackTraceSim.tokens;
    const x = extractTokenColumn(fallbackTraceSim.X, d, T, localTok);
    const u = extractTokenColumn(fallbackTraceSim.U, d, T, localTok);
    const mixerOut = extractTokenColumn(fallbackTraceSim.outVec, d, T, localTok);
    const y = extractTokenColumn(fallbackTraceSim.Y, d, T, localTok);
    return {
      x,
      stage1: modulationOrder === 'modulate_then_norm' ? new Float32Array(x) : u,
      stage2: new Float32Array(u),
      mixerOut,
      gatedDelta: new Float32Array(mixerOut),
      y,
      stage1Label,
      stage2Label,
      mixerOutLabel,
      hasLiveTrace: false,
    };
  }

  const zeros = () => new Float32Array(d);
  return {
    x: zeros(),
    stage1: zeros(),
    stage2: zeros(),
    mixerOut: zeros(),
    gatedDelta: zeros(),
    y: zeros(),
    stage1Label,
    stage2Label,
    mixerOutLabel,
    hasLiveTrace: false,
  };
}

// ============================================================================
// RG32F VALUE ATLAS PACKER (64×64, strictly zero texel collisions)
// ============================================================================

interface TimestepValueAtlas extends Rg32fValueAtlas {
  readonly tScalar: AtlasSlice;
  readonly phi: AtlasSlice;
  readonly mlp0W: AtlasSlice;
  readonly mlpH: AtlasSlice;
  readonly mlp2W: AtlasSlice;
  readonly tRepr: AtlasSlice;
  readonly tBlockW: AtlasSlice;
  readonly tEmb: AtlasSlice;
  readonly scaleShiftTable: AtlasSlice;
  readonly layerMod: AtlasSlice;
  readonly residualX: AtlasSlice;
  readonly stage1: AtlasSlice;
  readonly stage2: AtlasSlice;
  readonly mixerOut: AtlasSlice;
  readonly gatedDelta: AtlasSlice;
  readonly residualY: AtlasSlice;
}

const TIME_ATLAS_W = 64;
const TIME_ATLAS_H = 64;

function buildTimestepValueAtlas(params: {
  readonly weights: TimeConditioningWeights;
  readonly layerIndex: number;
  readonly bypassed: boolean;
  readonly tModel: number | null;
  readonly phi: Float32Array;
  readonly mlpH: Float32Array;
  readonly tRepr: Float32Array;
  readonly tEmb: Float32Array;
  readonly layerModFlat: Float32Array;
  readonly appVecs: LayerApplicationVectors;
}): TimestepValueAtlas {
  const {
    weights,
    layerIndex,
    bypassed,
    tModel,
    phi,
    mlpH,
    tRepr,
    tEmb,
    layerModFlat,
    appVecs,
  } = params;
  const { d, freqDim } = weights.config;
  const half = freqDim / 2;
  const writer = createRg32fAtlas(TIME_ATLAS_W, TIME_ATLAS_H);
  const { data, setTexel, makeRect } = writer;

  // y = 0, x = 0: scalar t
  setTexel(0, 0, tModel ?? 0, bypassed);

  // y = 2..3, x = 0..7: phi [2 × 8] (row 0 = cos, row 1 = sin)
  for (let k = 0; k < half; k++) {
    setTexel(k, 2, phi[k], bypassed);
    setTexel(k, 3, phi[half + k], bypassed);
  }

  // y = 6..13, x = 0..15: W_0 [8 × 16]
  for (let r = 0; r < d; r++) {
    for (let c = 0; c < freqDim; c++) {
      setTexel(c, 6 + r, weights.mlp0Weight[r * freqDim + c]);
    }
  }

  // y = 15, x = 0..7: SiLU(h0) [1 × 8]
  for (let i = 0; i < d; i++) {
    setTexel(i, 15, mlpH[i], bypassed);
  }

  // y = 17..24, x = 0..7: W_2 [8 × 8]
  for (let r = 0; r < d; r++) {
    for (let c = 0; c < d; c++) {
      setTexel(c, 17 + r, weights.mlp2Weight[r * d + c]);
    }
  }

  // y = 26, x = 0..7: tRepr [1 × 8]
  for (let i = 0; i < d; i++) {
    setTexel(i, 26, tRepr[i], bypassed);
  }

  // y = 0..23, x = 20..27: W_time_block [24 × 8]
  for (let r = 0; r < 3 * d; r++) {
    for (let c = 0; c < d; c++) {
      setTexel(20 + c, r, weights.tBlockWeight[r * d + c]);
    }
  }

  // y = 26..28, x = 20..27: tEmb [3 × 8]
  for (let band = 0; band < 3; band++) {
    for (let c = 0; c < d; c++) {
      setTexel(20 + c, 26 + band, tEmb[band * d + c], bypassed);
    }
  }

  // y = 30..32, x = 20..27: scale_shift_tables[layerIndex] [3 × 8]
  const table = weights.scaleShiftTables[layerIndex];
  for (let band = 0; band < 3; band++) {
    for (let c = 0; c < d; c++) {
      setTexel(20 + c, 30 + band, table[band * d + c]);
    }
  }

  // y = 34..36, x = 20..27: layerMod [3 × 8]
  for (let band = 0; band < 3; band++) {
    for (let c = 0; c < d; c++) {
      setTexel(20 + c, 34 + band, layerModFlat[band * d + c], bypassed);
    }
  }

  // y = 40..50 (even rows), x = 0..7: selected layer application vectors
  for (let c = 0; c < d; c++) {
    setTexel(c, 40, appVecs.x[c]);
    setTexel(c, 42, appVecs.stage1[c]);
    setTexel(c, 44, appVecs.stage2[c]);
    setTexel(c, 46, appVecs.mixerOut[c]);
    setTexel(c, 48, appVecs.gatedDelta[c]);
    setTexel(c, 50, appVecs.y[c]);
  }

  return {
    width: TIME_ATLAS_W,
    height: TIME_ATLAS_H,
    data,
    tScalar: { rect: makeRect(0, 0, 1, 1), scale: 1.0, mode: 1 },
    phi: { rect: makeRect(0, 2, half, 2), scale: 1.0, mode: 0 },
    mlp0W: { rect: makeRect(0, 6, freqDim, d), scale: 1 / computeMaxAbs(weights.mlp0Weight, 0.5), mode: 0 },
    mlpH: { rect: makeRect(0, 15, d, 1), scale: 1 / computeMaxAbs(mlpH, 0.5), mode: 0 },
    mlp2W: { rect: makeRect(0, 17, d, d), scale: 1 / computeMaxAbs(weights.mlp2Weight, 0.5), mode: 0 },
    tRepr: { rect: makeRect(0, 26, d, 1), scale: 1 / computeMaxAbs(tRepr, 0.5), mode: 0 },
    tBlockW: { rect: makeRect(20, 0, d, 3 * d), scale: 1 / computeMaxAbs(weights.tBlockWeight, 0.5), mode: 0 },
    tEmb: { rect: makeRect(20, 26, d, 3), scale: 1 / computeMaxAbs(tEmb, 0.5), mode: 0 },
    scaleShiftTable: { rect: makeRect(20, 30, d, 3), scale: 1 / computeMaxAbs(table, 0.5), mode: 0 },
    layerMod: { rect: makeRect(20, 34, d, 3), scale: 1 / computeMaxAbs(layerModFlat, 0.5), mode: 0 },
    residualX: { rect: makeRect(0, 40, d, 1), scale: 1 / computeMaxAbs(appVecs.x, 0.5), mode: 0 },
    stage1: { rect: makeRect(0, 42, d, 1), scale: 1 / computeMaxAbs(appVecs.stage1, 0.5), mode: 0 },
    stage2: { rect: makeRect(0, 44, d, 1), scale: 1 / computeMaxAbs(appVecs.stage2, 0.5), mode: 0 },
    mixerOut: { rect: makeRect(0, 46, d, 1), scale: 1 / computeMaxAbs(appVecs.mixerOut, 0.5), mode: 0 },
    gatedDelta: { rect: makeRect(0, 48, d, 1), scale: 1 / computeMaxAbs(appVecs.gatedDelta, 0.5), mode: 0 },
    residualY: { rect: makeRect(0, 50, d, 1), scale: 1 / computeMaxAbs(appVecs.y, 0.5), mode: 0 },
  };
}

// ============================================================================
// FORMULA & OPERAND PROVENANCE
// ============================================================================

interface TimestepSheetSpan {
  readonly tensor: TimestepTensorKey;
  readonly rowStart: number;
  readonly rowEnd: number;
  readonly colStart: number;
  readonly colEnd: number;
  readonly role: 'target' | 'operand_a' | 'operand_b';
}

interface TimestepCellFormula extends LayerFormulaReadout {
  readonly cell: TimestepCellRef;
  readonly targetSpan: TimestepSheetSpan;
  readonly upstreamSpans: readonly TimestepSheetSpan[];
}

const BAND_NAMES = ['shift', 'scale', 'gate'] as const;

function clampInt(v: number, lo: number, hi: number): number {
  const n = Math.trunc(v);
  return n < lo ? lo : n > hi ? hi : n;
}

function describeTimestepCellFormula(params: {
  readonly cell: TimestepCellRef;
  readonly weights: TimeConditioningWeights;
  readonly bayId: number;
  readonly layerIndex: number;
  readonly mixerKind: Kind;
  readonly modulationOrder: ModulationOrder;
  readonly bypassed: boolean;
  readonly sheetsDimmed: boolean;
  readonly timestepStatus: TwoTowerTimestepStatus;
  readonly denoiserLookup: TwoTowerDenoiserPassLookup;
  readonly currentContextPrefixLen: number;
  readonly tModel: number | null;
  readonly phi: Float32Array;
  readonly mlpH: Float32Array;
  readonly tRepr: Float32Array;
  readonly tEmb: Float32Array;
  readonly layerModFlat: Float32Array;
  readonly appVecs: LayerApplicationVectors;
  readonly tokTag: string;
}): TimestepCellFormula {
  const {
    cell,
    weights,
    bayId,
    layerIndex,
    mixerKind,
    modulationOrder,
    bypassed,
    sheetsDimmed,
    timestepStatus,
    denoiserLookup,
    currentContextPrefixLen,
    tModel,
    phi,
    mlpH,
    tRepr,
    tEmb,
    layerModFlat,
    appVecs,
    tokTag,
  } = params;
  const { d, freqDim } = weights.config;
  const half = freqDim / 2;
  const table = weights.scaleShiftTables[layerIndex];
  const bayTag = `Bay ${String(bayId).padStart(2, '0')} (L${layerIndex} · ${bayKindDisplayName(mixerKind)})`;
  const condMode = timestepStatus.conditioningMode;
  const inactiveNoPassYet = condMode === 'inactive_no_pass_yet';
  const isHistoricalPass = condMode === 'historical_prior_pass';
  const inactiveBadge =
    condMode === 'bypassed_ar'
      ? 'BYPASSED in Context AR (t = null)'
      : condMode === 'bypassed_mock_ar'
        ? 'BYPASSED in Mock-AR (t = null)'
        : 'INACTIVE · No diffusion pass yet';
  const inactiveValueText = bypassed ? 'bypassed' : 'inactive (no pass yet)';
  const inactiveEqSuffix =
    condMode === 'bypassed_ar'
      ? 'BYPASSED in Context AR (t = null)'
      : condMode === 'bypassed_mock_ar'
        ? 'BYPASSED in Mock-AR (t = null)'
        : 'INACTIVE (no mask_diffusion pass executed yet)';

  switch (cell.tensor) {
    case 't_scalar': {
      if (sheetsDimmed || tModel === null || timestepStatus.executedPass === null) {
        const nextNote = timestepStatus.nextEligibleStep
          ? ` Next eligible pass will use canonical t = nMasked / B = ${timestepStatus.nextEligibleStep.nextNMasked} / ${timestepStatus.nextEligibleStep.blockSize} = ${timestepStatus.nextEligibleStep.nextTModel.toFixed(4)} (never r / K).`
          : ` ${timestepStatus.blockCompletionNote}`;
        return {
          cell: { tensor: 't_scalar', row: 0, col: 0 },
          tensorLabel: 'Masked Fraction t',
          coordLabel: 't = null',
          headBadge: inactiveBadge,
          value: 0,
          isMasked: true,
          valueText: bypassed ? 'null (BYPASSED)' : 'null (no pass yet)',
          equation: bypassed
            ? `${inactiveBadge} — all AdaLN shift/scale/gate disabled`
            : `INACTIVE · No diffusion pass executed yet (t = null; NOT an AR/Mock-AR bypass)`,
          subEquation: bypassed
            ? 'Timestep conditioning runs ONLY during Denoiser mask_diffusion refinements (canonical t = nMasked / B, never r / K).'
            : `Mask-diffusion mode ready.${nextNote}`,
          targetSpan: { tensor: 't_scalar', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0, role: 'target' },
          upstreamSpans: [],
        };
      }
      const ep = timestepStatus.executedPass;
      const histPrefix = isHistoricalPass
        ? `HISTORICAL Rec #${ep.sourceRecordIndex} (b${ep.sourceBlockId ?? '-'} r${ep.sourceRefinementIndex ?? '-'}, ${ep.passExecutionId}, P_cond=${denoiserLookup.sourceContextPrefixLen ?? 'n/a'} vs P=${currentContextPrefixLen})`
        : `Denoiser Diffusion Active (Rec #${ep.sourceRecordIndex}, b${ep.sourceBlockId ?? '-'} r${ep.sourceRefinementIndex ?? '-'})`;
      const eq = ep.isDiagnosticOverride
        ? `Diagnostic override t_override = ${ep.overrideValue!.toFixed(4)} (empirical input nMasked / B = ${ep.nMaskedInput} / ${ep.blockSize} = ${ep.empiricalMaskFraction.toFixed(4)}) → t_scaled = ${(tModel * 1000).toFixed(1)}`
        : `t = nMasked / B = ${ep.nMaskedInput} / ${ep.blockSize} = ${tModel.toFixed(4)} → t_scaled = ${(tModel * 1000).toFixed(1)}`;
      const stageNote =
        timestepStatus.currentBlockMask &&
        timestepStatus.currentBlockMask.currentUnresolvedCount !== ep.nMaskedInput
          ? ` Post-step block state: ${timestepStatus.currentBlockMask.currentUnresolvedCount}/${timestepStatus.currentBlockMask.blockSize} masked (${timestepStatus.blockCompletionNote})`
          : '';
      return {
        cell: { tensor: 't_scalar', row: 0, col: 0 },
        tensorLabel: 'Masked Fraction t',
        coordLabel: ep.isDiagnosticOverride
          ? `t_override = ${ep.overrideValue!.toFixed(4)} (empirical ${ep.nMaskedInput}/${ep.blockSize})`
          : `t = ${ep.nMaskedInput}/${ep.blockSize}`,
        headBadge: histPrefix,
        value: tModel,
        isMasked: false,
        valueText: tModel.toFixed(4),
        equation: eq,
        subEquation: `Canonical t = nMasked / B (never r / K); outer scaling t_scaled = t × 1000 feeds FP32 sinusoidal frequencies.${stageNote}`,
        targetSpan: { tensor: 't_scalar', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'phi': {
      const row = clampInt(cell.row, 0, 1);
      const k = clampInt(cell.col, 0, half - 1);
      const flatIdx = row * half + k;
      const fnName = row === 0 ? 'cos' : 'sin';
      const val = phi[flatIdx];
      const { omega, arg } = computeSinusoidalArgF32(tModel ?? 0, k, freqDim);
      return {
        cell: { tensor: 'phi', row, col: k },
        tensorLabel: 'Sinusoidal Embedding φ(t)',
        coordLabel: `φ[${flatIdx}] = ${fnName}(k=${k})`,
        headBadge: sheetsDimmed ? inactiveBadge : 'FP32 · COS-FIRST [cos; sin]',
        value: val,
        isMasked: sheetsDimmed,
        valueText: sheetsDimmed ? inactiveValueText : fmtSigned(val, 4),
        equation: sheetsDimmed
          ? `φ[${flatIdx}] ${inactiveEqSuffix}`
          : `φ[${flatIdx}] = ${fnName}(t·1000 · ω_${k}) = ${fnName}(${arg.toFixed(3)}) = ${fmtSigned(val, 4)}`,
        subEquation: `ω_${k} = exp(-ln(10000)·${k}/${half}) = ${omega.toFixed(5)} · COS occupies [0..7], SIN occupies [8..15].`,
        targetSpan: { tensor: 'phi', rowStart: row, rowEnd: row, colStart: k, colEnd: k, role: 'target' },
        upstreamSpans: [{ tensor: 't_scalar', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: 0, role: 'operand_a' }],
      };
    }

    case 'mlp0_w': {
      const r = clampInt(cell.row, 0, d - 1);
      const c = clampInt(cell.col, 0, freqDim - 1);
      const val = weights.mlp0Weight[r * freqDim + c];
      return {
        cell: { tensor: 'mlp0_w', row: r, col: c },
        tensorLabel: 'Timestep MLP W₀ [8×16]',
        coordLabel: `W₀[ch=${r}, f=${c}]`,
        headBadge: 't_embedder.mlp.0.weight',
        value: val,
        isMasked: sheetsDimmed,
        valueText: fmtSigned(val, 4),
        equation: `W₀[${r}, ${c}] = ${fmtSigned(val, 4)} (b₀[${r}] = ${fmtSigned(weights.mlp0Bias[r], 4)})`,
        subEquation: 'Denoiser-only global MLP layer 0 weight (mini [8×16], prod [2688×256]).',
        targetSpan: { tensor: 'mlp0_w', rowStart: r, rowEnd: r, colStart: c, colEnd: c, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'mlp_h': {
      const r = clampInt(cell.col, 0, d - 1);
      const val = mlpH[r];
      return {
        cell: { tensor: 'mlp_h', row: 0, col: r },
        tensorLabel: 'Timestep MLP SiLU(h₀)',
        coordLabel: `h₀_act[ch=${r}]`,
        headBadge: sheetsDimmed ? inactiveBadge : 'SiLU(W₀·φ + b₀)',
        value: val,
        isMasked: sheetsDimmed,
        valueText: sheetsDimmed ? inactiveValueText : fmtSigned(val, 4),
        equation: sheetsDimmed
          ? `SiLU(h₀[${r}]) ${inactiveEqSuffix}`
          : `h₀_act[${r}] = SiLU(Σ W₀[${r},:]·φ + b₀[${r}]) = ${fmtSigned(val, 4)}`,
        subEquation: 'Intermediate SiLU activation feeding W₂ [8×8] inside t_embedder.mlp.',
        targetSpan: { tensor: 'mlp_h', rowStart: 0, rowEnd: 0, colStart: r, colEnd: r, role: 'target' },
        upstreamSpans: [
          { tensor: 'phi', rowStart: 0, rowEnd: 1, colStart: 0, colEnd: half - 1, role: 'operand_a' },
          { tensor: 'mlp0_w', rowStart: r, rowEnd: r, colStart: 0, colEnd: freqDim - 1, role: 'operand_b' },
        ],
      };
    }

    case 'mlp2_w': {
      const r = clampInt(cell.row, 0, d - 1);
      const c = clampInt(cell.col, 0, d - 1);
      const val = weights.mlp2Weight[r * d + c];
      return {
        cell: { tensor: 'mlp2_w', row: r, col: c },
        tensorLabel: 'Timestep MLP W₂ [8×8]',
        coordLabel: `W₂[ch=${r}, in=${c}]`,
        headBadge: 't_embedder.mlp.2.weight',
        value: val,
        isMasked: sheetsDimmed,
        valueText: fmtSigned(val, 4),
        equation: `W₂[${r}, ${c}] = ${fmtSigned(val, 4)} (b₂[${r}] = ${fmtSigned(weights.mlp2Bias[r], 4)})`,
        subEquation: 'Denoiser-only global MLP layer 2 weight (mini [8×8], prod [2688×2688]).',
        targetSpan: { tensor: 'mlp2_w', rowStart: r, rowEnd: r, colStart: c, colEnd: c, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 't_repr': {
      const r = clampInt(cell.col, 0, d - 1);
      const val = tRepr[r];
      return {
        cell: { tensor: 't_repr', row: 0, col: r },
        tensorLabel: 'Timestep Representation tRepr',
        coordLabel: `tRepr[ch=${r}]`,
        headBadge: sheetsDimmed ? inactiveBadge : 'Global MLP Output [8]',
        value: val,
        isMasked: sheetsDimmed,
        valueText: sheetsDimmed ? inactiveValueText : fmtSigned(val, 4),
        equation: sheetsDimmed
          ? `tRepr[${r}] ${inactiveEqSuffix}`
          : `tRepr[${r}] = W₂[${r},:] · h₀_act + b₂[${r}] = ${fmtSigned(val, 4)}`,
        subEquation: 'Shared timestep vector evaluated once per denoising step; feeds SiLU → W_time_block [24×8].',
        targetSpan: { tensor: 't_repr', rowStart: 0, rowEnd: 0, colStart: r, colEnd: r, role: 'target' },
        upstreamSpans: [
          { tensor: 'mlp_h', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: d - 1, role: 'operand_a' },
          { tensor: 'mlp2_w', rowStart: r, rowEnd: r, colStart: 0, colEnd: d - 1, role: 'operand_b' },
        ],
      };
    }

    case 't_block_w': {
      const r = clampInt(cell.row, 0, 3 * d - 1);
      const c = clampInt(cell.col, 0, d - 1);
      const band = Math.floor(r / d);
      const ch = r % d;
      const val = weights.tBlockWeight[r * d + c];
      return {
        cell: { tensor: 't_block_w', row: r, col: c },
        tensorLabel: 'Global Projection W_time_block [24×8]',
        coordLabel: `${BAND_NAMES[band]}_g[ch=${ch}] × in${c}`,
        headBadge: 't_block.1.weight (3d × d)',
        value: val,
        isMasked: sheetsDimmed,
        valueText: fmtSigned(val, 4),
        equation: `W_time_block[${r} (${BAND_NAMES[band]}[${ch}]), ${c}] = ${fmtSigned(val, 4)}`,
        subEquation: 'Shared PixArt-alpha adaLN-single projection mapping SiLU(tRepr) into [shift_g(8), scale_g(8), gate_g(8)].',
        targetSpan: { tensor: 't_block_w', rowStart: r, rowEnd: r, colStart: c, colEnd: c, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 't_emb': {
      const band = clampInt(cell.row, 0, 2);
      const ch = clampInt(cell.col, 0, d - 1);
      const r = band * d + ch;
      const val = tEmb[r];
      return {
        cell: { tensor: 't_emb', row: band, col: ch },
        tensorLabel: 'Global Modulation tEmb [3×8]',
        coordLabel: `${BAND_NAMES[band]}_g[ch=${ch}]`,
        headBadge: sheetsDimmed ? inactiveBadge : 'Shared Across All 52 Layers',
        value: val,
        isMasked: sheetsDimmed,
        valueText: sheetsDimmed ? inactiveValueText : fmtSigned(val, 4),
        equation: sheetsDimmed
          ? `tEmb[${BAND_NAMES[band]}_g, ch${ch}] ${inactiveEqSuffix}`
          : `${BAND_NAMES[band]}_g[${ch}] = W_time_block[${r},:] · SiLU(tRepr) + b_blk[${r}] = ${fmtSigned(val, 4)}`,
        subEquation: 'Broadcast to all 52 layers and added elementwise to each layer’s learned scale_shift_tables[l].',
        targetSpan: { tensor: 't_emb', rowStart: band, rowEnd: band, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans: [
          { tensor: 't_repr', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: d - 1, role: 'operand_a' },
          { tensor: 't_block_w', rowStart: r, rowEnd: r, colStart: 0, colEnd: d - 1, role: 'operand_b' },
        ],
      };
    }

    case 'scale_shift_table': {
      const band = clampInt(cell.row, 0, 2);
      const ch = clampInt(cell.col, 0, d - 1);
      const val = table[band * d + ch];
      return {
        cell: { tensor: 'scale_shift_table', row: band, col: ch },
        tensorLabel: `Learned scale_shift_tables[${layerIndex}]`,
        coordLabel: `table[L${layerIndex}][${BAND_NAMES[band]}, ch=${ch}]`,
        headBadge: `${bayTag} · Learned [3×8]`,
        value: val,
        isMasked: sheetsDimmed,
        valueText: fmtSigned(val, 4),
        equation: `scale_shift_tables[${layerIndex}][${BAND_NAMES[band]}, ${ch}] = ${fmtSigned(val, 4)}`,
        subEquation: 'Per-layer learned AdaLN offset (~randn/√d init) added to global tEmb.',
        targetSpan: { tensor: 'scale_shift_table', rowStart: band, rowEnd: band, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'layer_mod': {
      const band = clampInt(cell.row, 0, 2);
      const ch = clampInt(cell.col, 0, d - 1);
      const idx = band * d + ch;
      const val = layerModFlat[idx];
      const gVal = tEmb[idx];
      const tVal = table[idx];
      const gateExtra =
        band === 2
          ? ' · Outer AdaLN residual gate (d=8), distinct from Mamba internal SiLU(z) gate (d_inner=64)!'
          : '';
      return {
        cell: { tensor: 'layer_mod', row: band, col: ch },
        tensorLabel: `${bayTag} · Layer ${BAND_NAMES[band]}`,
        coordLabel: `${BAND_NAMES[band]}[ch=${ch}]`,
        headBadge: sheetsDimmed ? inactiveBadge : `tEmb + scale_shift_tables[${layerIndex}]`,
        value: val,
        isMasked: sheetsDimmed,
        valueText: sheetsDimmed ? inactiveValueText : fmtSigned(val, 4),
        equation: sheetsDimmed
          ? `${BAND_NAMES[band]}[${ch}] ${inactiveEqSuffix}`
          : `${BAND_NAMES[band]}[${ch}] = table(${fmtSigned(tVal, 3)}) + ${BAND_NAMES[band]}_g(${fmtSigned(gVal, 3)}) = ${fmtSigned(val, 4)}`,
        subEquation: `Active order for ${bayTag}: ${modulationOrder}${gateExtra}${inactiveNoPassYet ? ' · No pass executed yet.' : ''}`,
        targetSpan: { tensor: 'layer_mod', rowStart: band, rowEnd: band, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans: [
          { tensor: 't_emb', rowStart: band, rowEnd: band, colStart: ch, colEnd: ch, role: 'operand_a' },
          { tensor: 'scale_shift_table', rowStart: band, rowEnd: band, colStart: ch, colEnd: ch, role: 'operand_b' },
        ],
      };
    }

    case 'residual_x': {
      const ch = clampInt(cell.col, 0, d - 1);
      const val = appVecs.x[ch];
      return {
        cell: { tensor: 'residual_x', row: 0, col: ch },
        tensorLabel: 'Untouched Original Residual X',
        coordLabel: `X[ch=${ch}, ${tokTag}]`,
        headBadge: 'Untouched Residual Bypass Highway',
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 4),
        equation: `X[ch${ch}, ${tokTag}] = ${fmtSigned(val, 4)} (carried untouched to Y = X + gate ⊙ Mixer_out)`,
        subEquation: 'Residual bypass highway NEVER carries shift/scale modulation; only the mixer branch is modulated.',
        targetSpan: { tensor: 'residual_x', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans: [],
      };
    }

    case 'stage1': {
      const ch = clampInt(cell.col, 0, d - 1);
      const val = appVecs.stage1[ch];
      const xv = appVecs.x[ch];
      const sh = layerModFlat[ch];
      const sc = layerModFlat[d + ch];
      const eq =
        modulationOrder === 'modulate_then_norm'
          ? sheetsDimmed
            ? `X[${ch}] = ${fmtSigned(val, 4)} (Pre-RMSNorm modulation ${bypassed ? 'BYPASSED' : 'INACTIVE — no pass yet'})`
            : `X_mod[${ch}] = X(${fmtSigned(xv, 3)})·(1 + ${fmtSigned(sc, 3)}) + ${fmtSigned(sh, 3)} = ${fmtSigned(val, 4)}`
          : `U[${ch}] = RMSNorm(X)[${ch}] = ${fmtSigned(val, 4)} (MoE normalizes BEFORE AdaLN modulation)`;
      return {
        cell: { tensor: 'stage1', row: 0, col: ch },
        tensorLabel: appVecs.stage1Label,
        coordLabel: `ch=${ch} · ${tokTag}`,
        headBadge: modulationOrder,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 4),
        equation: eq,
        subEquation:
          modulationOrder === 'modulate_then_norm'
            ? `${bayTag} uses modulate_then_norm (X_mod feeds RMSNorm).`
            : `${bayTag} uses norm_then_modulate (RMSNorm(X) feeds AdaLN modulation).`,
        targetSpan: { tensor: 'stage1', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans:
          modulationOrder === 'modulate_then_norm' && !sheetsDimmed
            ? [
                { tensor: 'residual_x', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'operand_a' },
                { tensor: 'layer_mod', rowStart: 0, rowEnd: 1, colStart: ch, colEnd: ch, role: 'operand_b' },
              ]
            : [{ tensor: 'residual_x', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: d - 1, role: 'operand_a' }],
      };
    }

    case 'stage2': {
      const ch = clampInt(cell.col, 0, d - 1);
      const val = appVecs.stage2[ch];
      const s1 = appVecs.stage1[ch];
      const sh = layerModFlat[ch];
      const sc = layerModFlat[d + ch];
      const eq =
        modulationOrder === 'modulate_then_norm'
          ? `U[${ch}] = RMSNorm(${sheetsDimmed ? 'X' : 'X_mod'})[${ch}] = ${fmtSigned(val, 4)}`
          : sheetsDimmed
            ? `U[${ch}] = ${fmtSigned(val, 4)} (Post-RMSNorm modulation ${bypassed ? 'BYPASSED' : 'INACTIVE — no pass yet'})`
            : `U_mod[${ch}] = U(${fmtSigned(s1, 3)})·(1 + ${fmtSigned(sc, 3)}) + ${fmtSigned(sh, 3)} = ${fmtSigned(val, 4)}`;
      return {
        cell: { tensor: 'stage2', row: 0, col: ch },
        tensorLabel: appVecs.stage2Label,
        coordLabel: `ch=${ch} · ${tokTag}`,
        headBadge: modulationOrder,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 4),
        equation: eq,
        subEquation: `Feeds ${bayKindDisplayName(mixerKind)} mixer directly.`,
        targetSpan: { tensor: 'stage2', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans:
          modulationOrder === 'norm_then_modulate' && !sheetsDimmed
            ? [
                { tensor: 'stage1', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'operand_a' },
                { tensor: 'layer_mod', rowStart: 0, rowEnd: 1, colStart: ch, colEnd: ch, role: 'operand_b' },
              ]
            : [{ tensor: 'stage1', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: d - 1, role: 'operand_a' }],
      };
    }

    case 'mixer_out': {
      const ch = clampInt(cell.col, 0, d - 1);
      const val = appVecs.mixerOut[ch];
      return {
        cell: { tensor: 'mixer_out', row: 0, col: ch },
        tensorLabel: appVecs.mixerOutLabel,
        coordLabel: `ch=${ch} · ${tokTag}`,
        headBadge: `${bayKindDisplayName(mixerKind)} Raw Output`,
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 4),
        equation: `${appVecs.mixerOutLabel}[${ch}, ${tokTag}] = ${fmtSigned(val, 4)}`,
        subEquation:
          mixerKind === 'M'
            ? 'Already applied Mamba internal SiLU(z) gate (d_inner=64) before W_out; outer AdaLN gate[8] applies next.'
            : 'Pre-residual-gate mixer delta entering outer AdaLN residual gate[8].',
        targetSpan: { tensor: 'mixer_out', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans: [{ tensor: 'stage2', rowStart: 0, rowEnd: 0, colStart: 0, colEnd: d - 1, role: 'operand_a' }],
      };
    }

    case 'gated_delta': {
      const ch = clampInt(cell.col, 0, d - 1);
      const val = appVecs.gatedDelta[ch];
      const mVal = appVecs.mixerOut[ch];
      const gVal = layerModFlat[2 * d + ch];
      return {
        cell: { tensor: 'gated_delta', row: 0, col: ch },
        tensorLabel: 'Outer AdaLN Residual Gate (d = 8)',
        coordLabel: `gate[${ch}] ⊙ ${appVecs.mixerOutLabel}[${ch}]`,
        headBadge: sheetsDimmed
          ? bypassed
            ? 'Gate BYPASSED (1×)'
            : 'Gate INACTIVE (no pass yet)'
          : 'Outer Residual Gate (d=8 ≠ Mamba d_inner=64)',
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 4),
        equation: sheetsDimmed
          ? `Δ[${ch}] = ${appVecs.mixerOutLabel}(${fmtSigned(mVal, 3)}) (outer AdaLN gate ${bypassed ? 'BYPASSED' : 'INACTIVE'})`
          : `Δ[${ch}] = gate[${ch}](${fmtSigned(gVal, 3)}) · ${appVecs.mixerOutLabel}(${fmtSigned(mVal, 3)}) = ${fmtSigned(val, 4)}`,
        subEquation: 'Channel-wise outer AdaLN residual gate (d=8), strictly separate from Mamba’s internal d_inner=64 SiLU(z) gate.',
        targetSpan: { tensor: 'gated_delta', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans: sheetsDimmed
          ? [{ tensor: 'mixer_out', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'operand_a' }]
          : [
              { tensor: 'mixer_out', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'operand_a' },
              { tensor: 'layer_mod', rowStart: 2, rowEnd: 2, colStart: ch, colEnd: ch, role: 'operand_b' },
            ],
      };
    }

    case 'residual_y':
    default: {
      const ch = clampInt(cell.col, 0, d - 1);
      const val = appVecs.y[ch];
      const xv = appVecs.x[ch];
      const dv = appVecs.gatedDelta[ch];
      return {
        cell: { tensor: 'residual_y', row: 0, col: ch },
        tensorLabel: 'Residual Output Y',
        coordLabel: `Y[ch=${ch}, ${tokTag}]`,
        headBadge: sheetsDimmed ? 'Y = X + Mixer_out' : 'Y = X + gate ⊙ Mixer_out',
        value: val,
        isMasked: false,
        valueText: fmtSigned(val, 4),
        equation: `Y[${ch}, ${tokTag}] = X_untouched(${fmtSigned(xv, 3)}) + Δ(${fmtSigned(dv, 3)}) = ${fmtSigned(val, 4)}`,
        subEquation: `Untouched residual X merges with ${sheetsDimmed ? 'unmodulated' : 'outer-gated'} ${appVecs.mixerOutLabel}. Final norm_f remains strictly unmodulated.`,
        targetSpan: { tensor: 'residual_y', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'target' },
        upstreamSpans: [
          { tensor: 'residual_x', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'operand_a' },
          { tensor: 'gated_delta', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'operand_b' },
        ],
      };
    }
  }
}

// ============================================================================
// SCENE BUILDER
// ============================================================================

interface TimestepSheetSpec {
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
  readonly highlight: number;
  readonly seed: number;
  readonly partId: string;
  readonly tensorKey?: TimestepTensorKey;
  readonly atlasSlice?: AtlasSlice;
  readonly registryKey?: string;
  readonly cellFamily: number;
}

class TimestepSceneBuilder {
  readonly sheetFloats: number[] = [];
  readonly basisFloats: number[] = [];
  readonly solidFloats: number[] = [];
  readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();

  private readonly bayId: number;
  private readonly towerRole: 'denoiser' | 'shared';
  private readonly executionId: string;
  private readonly traceOrigin: TwoTowerTraceOrigin;
  private readonly absTokenOffset: number;

  constructor(params: {
    readonly bayId: number;
    readonly towerRole: 'denoiser' | 'shared';
    readonly executionId: string;
    readonly traceOrigin: TwoTowerTraceOrigin;
    readonly absTokenOffset: number;
  }) {
    this.bayId = params.bayId;
    this.towerRole = params.towerRole;
    this.executionId = params.executionId;
    this.traceOrigin = params.traceOrigin;
    this.absTokenOffset = params.absTokenOffset;
  }

  addSheet(s: TimestepSheetSpec): SheetFrame3D {
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
      0,
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

    this.solidFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.cellFamily,
      0,
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
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      towerRole: this.towerRole,
      executionId: this.executionId,
      traceOrigin: this.traceOrigin,
      semanticTag: 'timestep',
      absoluteTokenOffset: this.absTokenOffset,
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

    return frame;
  }

  addGuideLine(
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cssWidth = 1.0,
    style = LINE_STYLE_GUIDE,
    alpha = 0.5,
  ): void {
    pushFreeGuideLine(this.lineFloats, ax, ay, az, bx, by, bz, cssWidth, style, alpha);
  }

  addDashedFrameXZ(
    x0: number, z0: number, x1: number, z1: number, y: number,
    style = LINE_STYLE_GUIDE, alpha = 0.48,
  ): void {
    pushDashedFrameXZ(this.lineFloats, x0, z0, x1, z1, y, style, alpha, 0.20, 0.12, 0.9);
  }

  pack() {
    const flowPacked = this.flow.pack();
    return {
      sheetData: new Float32Array(this.sheetFloats),
      sheetCount: this.sheetFloats.length / SHEET_INSTANCE_FLOATS,
      sheetBasisData: new Float32Array(this.basisFloats),
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

const RIBBON_TIME_GOLD: FlowRibbonColor = { r: 0.78, g: 0.52, b: 0.14, a: 0.62, borderAlpha: 0.32 };
const RIBBON_GATE_CORAL: FlowRibbonColor = { r: 0.84, g: 0.34, b: 0.22, a: 0.72, borderAlpha: 0.38 };

function partIdForTimestepTensor(t: TimestepTensorKey): string {
  switch (t) {
    case 't_scalar':
      return 't-scalar';
    case 'phi':
      return 'phi-sinusoid';
    case 'mlp0_w':
      return 'mlp0-w';
    case 'mlp_h':
      return 'mlp-h';
    case 'mlp2_w':
      return 'mlp2-w';
    case 't_repr':
      return 't-repr';
    case 't_block_w':
      return 't-block-w';
    case 't_emb':
      return 't-emb';
    case 'scale_shift_table':
      return 'scale-shift-table';
    case 'layer_mod':
      return 'layer-mod';
    case 'residual_x':
      return 'residual-x';
    case 'stage1':
      return 'layer-stage1';
    case 'stage2':
      return 'layer-stage2';
    case 'mixer_out':
      return 'mixer-out';
    case 'gated_delta':
      return 'gated-delta';
    case 'residual_y':
      return 'residual-y';
  }
}

function defaultCellForTimestepStepper(step: number): TimestepCellRef {
  switch (clampInt(step, 0, 5)) {
    case 0:
      return { tensor: 't_scalar', row: 0, col: 0 };
    case 1:
      return { tensor: 'phi', row: 0, col: 0 };
    case 2:
      return { tensor: 't_repr', row: 0, col: 0 };
    case 3:
      return { tensor: 't_emb', row: 0, col: 0 };
    case 4:
      return { tensor: 'layer_mod', row: 2, col: 0 };
    case 5:
    default:
      return { tensor: 'residual_y', row: 0, col: 0 };
  }
}

/**
 * Build the dedicated 3D Timestep Conditioning & Split AdaLN-Single Cutaway scene directly from
 * a `TwoTowerSession`'s active or historical record (zero DOM/WebGL/mutation).
 */
export function buildTwoTowerTimestepScene(
  session: TwoTowerSession,
  options?: BuildTimestepCutawayOptions,
): TwoTowerSpatialScene {
  const view = resolveTwoTowerSessionViewState(session);
  const bayId = options?.bayId ?? 6;
  const mixerKind = getTwoTowerBayKind(bayId);
  const layerIndex = bayId - 1;
  const modulationOrder = modulationOrderForBayKind(mixerKind);
  const weights = view.model.timeConditioning;
  const { d, freqDim } = weights.config;
  const half = freqDim / 2;

  const ts = view.timestepStatus;
  const dl = view.denoiserLookup;
  const denoiserPass = dl.pass;
  const conditioningMode = ts.conditioningMode;
  const bypassed =
    conditioningMode === 'bypassed_ar' || conditioningMode === 'bypassed_mock_ar';
  const inactiveNoPassYet = conditioningMode === 'inactive_no_pass_yet';
  const isHistoricalPass = conditioningMode === 'historical_prior_pass';

  const hasExecutedDiffusionPass =
    (conditioningMode === 'active_diffusion' || conditioningMode === 'historical_prior_pass') &&
    ts.executedPass !== null &&
    denoiserPass !== null &&
    denoiserPass.timestepProjection !== null;
  const sheetsDimmed = !hasExecutedDiffusionPass;

  const tModel: number | null = hasExecutedDiffusionPass ? ts.executedPass!.tModel : null;
  const nMasked: number | null = hasExecutedDiffusionPass ? ts.executedPass!.nMaskedInput : null;
  const blockSize: number | null = hasExecutedDiffusionPass ? ts.executedPass!.blockSize : null;
  const empiricalMaskFraction: number | null = hasExecutedDiffusionPass
    ? ts.executedPass!.empiricalMaskFraction
    : null;
  const isDiagnosticOverride = hasExecutedDiffusionPass && ts.executedPass!.isDiagnosticOverride;
  const overrideValue: number | null = hasExecutedDiffusionPass
    ? ts.executedPass!.overrideValue
    : null;

  const bypassReason: string | null =
    conditioningMode === 'bypassed_ar'
      ? 'BYPASSED in Context AR (t = null) — Context Tower never applies timestep conditioning.'
      : conditioningMode === 'bypassed_mock_ar'
        ? 'BYPASSED in Mock-AR (t = null) — Mock-AR runs Denoiser causally without AdaLN modulation.'
        : null;

  const statusLabel =
    conditioningMode === 'bypassed_ar'
      ? 'BYPASSED in Context AR (t = null)'
      : conditioningMode === 'bypassed_mock_ar'
        ? 'BYPASSED in Mock-AR (t = null)'
        : conditioningMode === 'inactive_no_pass_yet'
          ? ts.summaryLabel
          : isHistoricalPass && ts.executedPass !== null
            ? isDiagnosticOverride
              ? `HISTORICAL Diffusion Pass (Rec #${ts.executedPass.sourceRecordIndex}, b${ts.executedPass.sourceBlockId ?? '-'} r${ts.executedPass.sourceRefinementIndex ?? '-'}, t_override = ${tModel!.toFixed(3)}; empirical ${nMasked}/${blockSize} = ${empiricalMaskFraction!.toFixed(3)})`
              : `HISTORICAL Diffusion Pass (Rec #${ts.executedPass.sourceRecordIndex}, b${ts.executedPass.sourceBlockId ?? '-'} r${ts.executedPass.sourceRefinementIndex ?? '-'}, t = ${tModel!.toFixed(3)}, input ${nMasked}/${blockSize})`
            : isDiagnosticOverride
              ? `ACTIVE Diffusion Conditioning (t_override = ${tModel!.toFixed(3)}; empirical ${nMasked}/${blockSize} = ${empiricalMaskFraction!.toFixed(3)})`
              : `ACTIVE Diffusion Conditioning (t = ${tModel!.toFixed(3)}, input ${nMasked}/${blockSize} masked)`;

  // Extract REAL timestep conditioning tensors (detached copies, never mutating session buffers)
  const phi =
    hasExecutedDiffusionPass && denoiserPass?.timestepProjection
      ? new Float32Array(denoiserPass.timestepProjection.phi)
      : new Float32Array(freqDim);
  const mlpH = hasExecutedDiffusionPass ? computeMlpHiddenAct(phi, weights) : new Float32Array(d);
  const tRepr =
    hasExecutedDiffusionPass && denoiserPass?.timestepProjection
      ? new Float32Array(denoiserPass.timestepProjection.tRepr)
      : new Float32Array(d);
  const tEmb =
    hasExecutedDiffusionPass && denoiserPass?.timestepProjection
      ? new Float32Array(denoiserPass.timestepProjection.tEmb)
      : new Float32Array(3 * d);

  const layerMod = hasExecutedDiffusionPass ? denoiserPass!.getLayerModulation(bayId) : null;
  const layerModFlat = new Float32Array(3 * d);
  if (layerMod) {
    layerModFlat.set(layerMod.shift, 0);
    layerModFlat.set(layerMod.scale, d);
    layerModFlat.set(layerMod.gate, 2 * d);
  }

  const shiftL2 = layerMod ? l2Norm(layerMod.shift) : 0;
  const scaleL2 = layerMod ? l2Norm(layerMod.scale) : 0;
  const gateL2 = layerMod ? l2Norm(layerMod.gate) : 0;

  // Pull REAL selected layer trace (Denoiser trace when available, else latest Context bay trace for bypassed/inactive view)
  const denoiserTrace = denoiserPass ? view.resolveDenoiserBayTrace(bayId) : null;
  let fallbackTraceSim: {
    readonly X: Float32Array;
    readonly U: Float32Array;
    readonly Y: Float32Array;
    readonly tokens: number;
    readonly outVec: Float32Array;
  } | null = null;
  let seqTokens = denoiserTrace ? denoiserTrace.sim.tokens : 1;
  let absTokenOffset = denoiserPass ? denoiserPass.absoluteTokenStart : 0;
  let tokenLabels: readonly string[] = denoiserPass ? denoiserPass.tokenLabels : [];
  let executionId = denoiserPass ? denoiserPass.passExecutionId : view.identity.recordId;
  let traceOrigin: TwoTowerTraceOrigin = denoiserPass ? 'denoiser_scratch' : 'persistent_cache';

  if (!denoiserTrace) {
    const ctxLookup = view.resolveContextBayTrace(bayId);
    if (ctxLookup.status === 'available' && ctxLookup.trace && ctxLookup.segment) {
      const tr = ctxLookup.trace;
      seqTokens = tr.sim.tokens;
      absTokenOffset = ctxLookup.segment.tokenRange[0];
      tokenLabels = ctxLookup.segment.tokenLabels;
      executionId = ctxLookup.segment.stepExecutionId;
      traceOrigin = tr.traceOrigin;
      fallbackTraceSim = {
        X: tr.sim.X,
        U: tr.sim.U,
        Y: tr.sim.Y,
        tokens: tr.sim.tokens,
        outVec:
          tr.kind === 'M'
            ? tr.sim.Mamba_out
            : tr.kind === '*'
              ? tr.sim.Attn_out
              : tr.sim.y_moe,
      };
    }
  }

  const localTokenIndex = clampInt(options?.localTokenIndex ?? 0, 0, Math.max(0, seqTokens - 1));
  const absoluteTokenPosition =
    denoiserTrace || fallbackTraceSim ? absTokenOffset + localTokenIndex : null;
  const tokenLabel =
    tokenLabels[localTokenIndex] ??
    (absoluteTokenPosition !== null ? `t${absoluteTokenPosition}` : 'n/a');
  const tokTag =
    absoluteTokenPosition !== null
      ? `t${absoluteTokenPosition} "${tokenLabel}"`
      : tokenLabel;

  const appVecs = extractLayerApplicationVectors({
    d,
    mixerKind,
    modulationOrder,
    bypassed: sheetsDimmed,
    trace: denoiserTrace,
    fallbackTraceSim,
    localTok: localTokenIndex,
    gate: layerMod ? layerMod.gate : null,
  });

  const atlas = buildTimestepValueAtlas({
    weights,
    layerIndex,
    bypassed: sheetsDimmed,
    tModel,
    phi,
    mlpH,
    tRepr,
    tEmb,
    layerModFlat,
    appVecs,
  });

  const stepperStep =
    options?.stepperStep !== undefined && options?.stepperStep !== null
      ? clampInt(options.stepperStep, 0, 5)
      : null;
  const activeCell: TimestepCellRef | null =
    options?.selectedCell ?? (stepperStep !== null ? defaultCellForTimestepStepper(stepperStep) : null);

  const activeFormula: TimestepCellFormula | null = activeCell
    ? describeTimestepCellFormula({
        cell: activeCell,
        weights,
        bayId,
        layerIndex,
        mixerKind,
        modulationOrder,
        bypassed,
        sheetsDimmed,
        timestepStatus: ts,
        denoiserLookup: dl,
        currentContextPrefixLen: view.committedPrefixLen,
        tModel,
        phi,
        mlpH,
        tRepr,
        tEmb,
        layerModFlat,
        appVecs,
        tokTag,
      })
    : null;

  const focusSet: SheetFocusSet = activeFormula
    ? createSheetFocusSet([
        {
          tensor: activeFormula.targetSpan.tensor,
          rowStart: activeFormula.targetSpan.rowStart,
          rowEnd: activeFormula.targetSpan.rowEnd,
          colStart: activeFormula.targetSpan.colStart,
          colEnd: activeFormula.targetSpan.colEnd,
          role: 'target',
        },
        ...activeFormula.upstreamSpans.map(
          (u): SheetFocusEntry => ({
            tensor: u.tensor,
            rowStart: u.rowStart,
            rowEnd: u.rowEnd,
            colStart: u.colStart,
            colEnd: u.colEnd,
            role: u.role,
          }),
        ),
      ])
    : EMPTY_SHEET_FOCUS_SET;

  const activePart = options?.hoveredPartId ?? options?.selectedPartId ?? null;
  const hasCellTrace = focusSet.active;
  const hasFocus = activePart !== null || hasCellTrace || stepperStep !== null;

  const b = new TimestepSceneBuilder({
    bayId,
    towerRole: denoiserPass ? 'denoiser' : 'shared',
    executionId,
    traceOrigin,
    absTokenOffset,
  });

  const BASE_Y = 0.12;
  const SHEET_THICK = 0.058;
  const WALL_Y = 1.46;

  // Helper resolving highlight state: dims modulation sheets visibly whenever bypassed or inactive (no pass yet)!
  const modSheetHl = (tensor: TimestepTensorKey, partId: string, stepIdx: number, isGold = false): number => {
    if (sheetsDimmed) {
      if (activePart === partId || activeFormula?.cell.tensor === tensor) {
        return SHEET_HL_PATHWAY;
      }
      return SHEET_HL_DIMMED;
    }
    const fallback =
      activePart === partId || stepperStep === stepIdx
        ? SHEET_HL_ACTIVE
        : hasFocus
          ? SHEET_HL_DIMMED
          : isGold
            ? SHEET_HL_BF16_GOLD
            : SHEET_HL_NORMAL;
    return resolveSheetHighlight(focusSet, { tensor }, fallback);
  };

  const appSheetHl = (tensor: TimestepTensorKey, partId: string, defaultGold = false): number => {
    const fallback =
      activePart === partId || stepperStep === 5
        ? SHEET_HL_ACTIVE
        : hasFocus
          ? SHEET_HL_DIMMED
          : defaultGold
            ? SHEET_HL_BF16_GOLD
            : SHEET_HL_NORMAL;
    return resolveSheetHighlight(focusSet, { tensor }, fallback);
  };

  // =========================================================================
  // STAGE 0..4: Global Timestep Backbone -> Per-Layer Table -> (shift,scale,gate)
  // =========================================================================
  const xScalar = -8.95;
  const xPhi = -7.05;
  const xW0 = -5.15;
  const xMlpH = -3.75;
  const xW2 = -2.45;
  const xTRepr = -1.15;
  const xTBlockW = 0.25;
  const xTEmb = 1.75;
  const xLayerMod = 3.65;

  // Stage 0: Scalar t = nMasked / B
  b.addSheet({
    cx: xScalar, cy: BASE_Y, cz: 0,
    ex: 0.96, ey: SHEET_THICK, ez: 0.96,
    axes: FRAME_XZ_AXES,
    gridCols: 1, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_SOFTMAX,
    highlight: modSheetHl('t_scalar', 't-scalar', 0, true),
    seed: 1.1,
    partId: 't-scalar',
    tensorKey: 't_scalar',
    atlasSlice: atlas.tScalar,
    registryKey: 't_scalar',
    cellFamily: CELL_MOE_AMBER,
  });

  // Stage 1: Sinusoidal φ(t) [2 × 8] (row 0 = cos, row 1 = sin)
  b.addSheet({
    cx: xPhi, cy: BASE_Y, cz: 0,
    ex: 1.92, ey: SHEET_THICK, ez: 0.96,
    axes: FRAME_XZ_AXES,
    gridCols: half, gridRows: 2,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_SCORES,
    highlight: modSheetHl('phi', 'phi-sinusoid', 1),
    seed: 2.3,
    partId: 'phi-sinusoid',
    tensorKey: 'phi',
    atlasSlice: atlas.phi,
    registryKey: 'phi',
    cellFamily: CELL_SSM_CYAN,
  });

  // Stage 2: MLP W_0 [8×16] (upright), SiLU(h0) [1×8], W_2 [8×8] (upright), tRepr [1×8]
  b.addSheet({
    cx: xW0, cy: WALL_Y, cz: 0,
    ex: 1.80, ey: SHEET_THICK, ez: 1.56,
    axes: FRAME_YZ_AXES,
    gridCols: freqDim, gridRows: d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: modSheetHl('mlp0_w', 'mlp0-w', 2),
    seed: 3.5,
    partId: 'mlp0-w',
    tensorKey: 'mlp0_w',
    atlasSlice: atlas.mlp0W,
    registryKey: 'mlp0_w',
    cellFamily: CELL_WEIGHT_BLUE,
  });
  b.addSheet({
    cx: xMlpH, cy: BASE_Y, cz: 0,
    ex: 1.48, ey: SHEET_THICK, ez: 0.48,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: modSheetHl('mlp_h', 'mlp-h', 2),
    seed: 4.1,
    partId: 'mlp-h',
    tensorKey: 'mlp_h',
    atlasSlice: atlas.mlpH,
    registryKey: 'mlp_h',
    cellFamily: CELL_ACTIVATION_GREEN,
  });
  b.addSheet({
    cx: xW2, cy: WALL_Y, cz: 0,
    ex: 1.42, ey: SHEET_THICK, ez: 1.42,
    axes: FRAME_YZ_AXES,
    gridCols: d, gridRows: d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: modSheetHl('mlp2_w', 'mlp2-w', 2),
    seed: 5.2,
    partId: 'mlp2-w',
    tensorKey: 'mlp2_w',
    atlasSlice: atlas.mlp2W,
    registryKey: 'mlp2_w',
    cellFamily: CELL_WEIGHT_BLUE,
  });
  b.addSheet({
    cx: xTRepr, cy: BASE_Y, cz: 0,
    ex: 1.48, ey: SHEET_THICK, ez: 0.48,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: modSheetHl('t_repr', 't-repr', 2),
    seed: 6.4,
    partId: 't-repr',
    tensorKey: 't_repr',
    atlasSlice: atlas.tRepr,
    registryKey: 't_repr',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Stage 3: W_time_block [24×8] (upright) -> tEmb [3×8]
  b.addSheet({
    cx: xTBlockW, cy: WALL_Y + 0.14, cz: 0,
    ex: 1.56, ey: SHEET_THICK, ez: 2.38,
    axes: FRAME_YZ_AXES,
    gridCols: d, gridRows: 3 * d,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: modSheetHl('t_block_w', 't-block-w', 3),
    seed: 7.3,
    partId: 't-block-w',
    tensorKey: 't_block_w',
    atlasSlice: atlas.tBlockW,
    registryKey: 't_block_w',
    cellFamily: CELL_WEIGHT_BLUE,
  });
  b.addSheet({
    cx: xTEmb, cy: BASE_Y, cz: 0,
    ex: 1.58, ey: SHEET_THICK, ez: 1.16,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 3,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: modSheetHl('t_emb', 't-emb', 3, true),
    seed: 8.1,
    partId: 't-emb',
    tensorKey: 't_emb',
    atlasSlice: atlas.tEmb,
    registryKey: 't_emb',
    cellFamily: CELL_MOE_AMBER,
  });

  // Stage 4: Per-layer learned scale_shift_tables[l] [3×8] (rear -Z) + summed layerMod [3×8]
  const zTable = -2.05;
  b.addSheet({
    cx: xLayerMod, cy: BASE_Y + 0.08, cz: zTable,
    ex: 1.58, ey: SHEET_THICK, ez: 1.16,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 3,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: modSheetHl('scale_shift_table', 'scale-shift-table', 4),
    seed: 9.2,
    partId: 'scale-shift-table',
    tensorKey: 'scale_shift_table',
    atlasSlice: atlas.scaleShiftTable,
    registryKey: 'scale_shift_table',
    cellFamily: CELL_WEIGHT_BLUE,
  });
  b.addSheet({
    cx: xLayerMod, cy: BASE_Y, cz: 0,
    ex: 1.58, ey: SHEET_THICK, ez: 1.16,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 3,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: modSheetHl('layer_mod', 'layer-mod', 4, true),
    seed: 10.4,
    partId: 'layer-mod',
    tensorKey: 'layer_mod',
    atlasSlice: atlas.layerMod,
    registryKey: 'layer_mod',
    cellFamily: CELL_MOE_AMBER,
  });

  // =========================================================================
  // STAGE 5: Selected Layer Application (Asymmetric Ordering + Untouched Residual X)
  // =========================================================================
  const xResX = 5.45;
  const xStage1 = 6.20;
  const xStage2 = 7.35;
  const xMixerOut = 8.45;
  const xGated = 9.35;
  const xResY = 9.95;

  const zResBypass = -2.15;
  const zAppTrack = 0.0;

  // Dedicated untouched emerald Residual Bypass Highway (X -> Y)
  b.addSheet({
    cx: xResX, cy: BASE_Y + 0.04, cz: zResBypass,
    ex: 1.44, ey: SHEET_THICK, ez: 0.50,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    highlight: appSheetHl('residual_x', 'residual-x'),
    seed: 11.1,
    partId: 'residual-x',
    tensorKey: 'residual_x',
    atlasSlice: atlas.residualX,
    registryKey: 'residual_x',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Stage 1 & Stage 2 of asymmetric ordering (M/* modulate->norm vs E norm->modulate)
  b.addSheet({
    cx: xStage1, cy: BASE_Y, cz: zAppTrack,
    ex: 1.02, ey: SHEET_THICK, ez: 0.46,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: modulationOrder === 'modulate_then_norm' ? SHEET_ROLE_ACTIVATION : SHEET_ROLE_NORM,
    highlight: appSheetHl('stage1', 'layer-stage1'),
    seed: 12.1,
    partId: 'layer-stage1',
    tensorKey: 'stage1',
    atlasSlice: atlas.stage1,
    registryKey: 'stage1',
    cellFamily: modulationOrder === 'modulate_then_norm' ? CELL_MOE_AMBER : CELL_SSM_CYAN,
  });
  b.addSheet({
    cx: xStage2, cy: BASE_Y, cz: zAppTrack,
    ex: 1.02, ey: SHEET_THICK, ez: 0.46,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: modulationOrder === 'modulate_then_norm' ? SHEET_ROLE_NORM : SHEET_ROLE_ACTIVATION,
    highlight: appSheetHl('stage2', 'layer-stage2'),
    seed: 13.1,
    partId: 'layer-stage2',
    tensorKey: 'stage2',
    atlasSlice: atlas.stage2,
    registryKey: 'stage2',
    cellFamily: modulationOrder === 'modulate_then_norm' ? CELL_SSM_CYAN : CELL_MOE_AMBER,
  });

  // Mixer raw output -> Outer residual gate[8] -> Output Residual Y
  b.addSheet({
    cx: xMixerOut, cy: BASE_Y, cz: zAppTrack,
    ex: 0.96, ey: SHEET_THICK, ez: 0.46,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: appSheetHl('mixer_out', 'mixer-out'),
    seed: 14.1,
    partId: 'mixer-out',
    tensorKey: 'mixer_out',
    atlasSlice: atlas.mixerOut,
    registryKey: 'mixer_out',
    cellFamily: CELL_ACTIVATION_GREEN,
  });
  b.addSheet({
    cx: xGated, cy: BASE_Y + 0.04, cz: zAppTrack + 0.92,
    ex: 1.12, ey: SHEET_THICK, ez: 0.48,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: appSheetHl('gated_delta', 'gated-delta', !sheetsDimmed),
    seed: 15.1,
    partId: 'gated-delta',
    tensorKey: 'gated_delta',
    atlasSlice: atlas.gatedDelta,
    registryKey: 'gated_delta',
    cellFamily: CELL_MOE_AMBER,
  });
  b.addSheet({
    cx: xResY, cy: BASE_Y + 0.04, cz: -0.78,
    ex: 1.06, ey: SHEET_THICK, ez: 0.50,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    highlight: appSheetHl('residual_y', 'residual-y'),
    seed: 16.1,
    partId: 'residual-y',
    tensorKey: 'residual_y',
    atlasSlice: atlas.residualY,
    registryKey: 'residual_y',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Reference Contrast Panels (+Z front):
  // 1. Outer Residual Gate (d=8) vs Mamba Internal SiLU(z) Gate (d_inner=64)
  b.addSheet({
    cx: 4.55, cy: BASE_Y + 0.02, cz: 2.25,
    ex: 3.10, ey: 0.044, ez: 0.78,
    axes: FRAME_XZ_AXES,
    gridCols: 8, gridRows: 2,
    sheetKind: SHEET_KIND_GHOST_RAIL,
    role: SHEET_ROLE_NEUTRAL,
    highlight: activePart === 'gate-contrast' ? SHEET_HL_ACTIVE : SHEET_HL_NORMAL,
    seed: 17.7,
    partId: 'gate-contrast',
    registryKey: 'gate_contrast',
    cellFamily: CELL_NEUTRAL,
  });

  // 2. Final RMSNorm (norm_f) Strictly Unmodulated Callout
  b.addSheet({
    cx: 8.35, cy: BASE_Y + 0.02, cz: 2.25,
    ex: 2.80, ey: 0.044, ez: 0.78,
    axes: FRAME_XZ_AXES,
    gridCols: d, gridRows: 1,
    sheetKind: SHEET_KIND_GHOST_RAIL,
    role: SHEET_ROLE_NEUTRAL,
    highlight: activePart === 'final-norm-unmodulated' ? SHEET_HL_ACTIVE : SHEET_HL_NORMAL,
    seed: 18.8,
    partId: 'final-norm-unmodulated',
    registryKey: 'final_norm_unmodulated',
    cellFamily: CELL_NEUTRAL,
  });

  // Architectural dashed frame around selected layer application box
  b.addDashedFrameXZ(4.65, -2.75, 10.50, 1.55, 0.02, sheetsDimmed ? LINE_STYLE_GUIDE : LINE_STYLE_ACTIVE, 0.45);
  // Stanchions grounding upright weight walls
  for (const wx of [xW0, xW2, xTBlockW]) {
    b.addGuideLine(wx, 0.0, -0.75, wx, 0.45, -0.75, 0.9, LINE_STYLE_GUIDE, 0.45);
    b.addGuideLine(wx, 0.0, 0.75, wx, 0.45, 0.75, 0.9, LINE_STYLE_GUIDE, 0.45);
  }

  // =========================================================================
  // SMOOTH 3D FLOW RIBBONS (Modulation Chain + Emerald Untouched Residual Highway)
  // =========================================================================
  const chainCol = sheetsDimmed ? RIBBON_DIMMED : hasCellTrace ? RIBBON_DIMMED : RIBBON_TIME_GOLD;
  const gateCol = sheetsDimmed ? RIBBON_DIMMED : hasCellTrace ? RIBBON_DIMMED : RIBBON_GATE_CORAL;
  const resCol = hasCellTrace ? RIBBON_RESIDUAL : RIBBON_RESIDUAL_ACTIVE;

  const connectHorizontal = (xA: number, xB: number, z: number, col: FlowRibbonColor, w = 0.085) => {
    b.flow.addPolyline3D(
      [
        [xA, BASE_Y + 0.04, z],
        [xB, BASE_Y + 0.04, z],
      ],
      [0, 1, 0],
      w,
      col,
      true,
      0.16,
      0.18,
      true,
    );
  };

  connectHorizontal(xScalar + 0.52, xPhi - 1.00, 0, chainCol);
  connectHorizontal(xPhi + 1.00, xW0 - 0.14, 0, chainCol, 0.075);
  connectHorizontal(xW0 + 0.14, xMlpH - 0.78, 0, chainCol, 0.075);
  connectHorizontal(xMlpH + 0.78, xW2 - 0.14, 0, chainCol, 0.075);
  connectHorizontal(xW2 + 0.14, xTRepr - 0.78, 0, chainCol, 0.075);
  connectHorizontal(xTRepr + 0.78, xTBlockW - 0.14, 0, chainCol, 0.075);
  connectHorizontal(xTBlockW + 0.14, xTEmb - 0.84, 0, chainCol, 0.075);
  connectHorizontal(xTEmb + 0.84, xLayerMod - 0.84, 0, chainCol, 0.088);

  // Per-layer table -> layerMod elementwise add ribbon
  b.flow.addPolyline3D(
    [
      [xLayerMod, BASE_Y + 0.08, zTable + 0.62],
      [xLayerMod, BASE_Y + 0.04, -0.62],
    ],
    [0, 1, 0],
    0.082,
    chainCol,
    true,
    0.16,
    0.16,
    true,
  );

  // Selected layer's (shift, scale) branch feeding stage1 (M/*) or stage2 (E)
  const modTargetX = modulationOrder === 'modulate_then_norm' ? xStage1 : xStage2;
  b.flow.addCubic3D(
    [xLayerMod + 0.82, BASE_Y + 0.04, -0.25],
    [xLayerMod + 1.45, BASE_Y + 0.36, -0.45],
    [modTargetX - 0.65, BASE_Y + 0.36, -0.35],
    [modTargetX, BASE_Y + 0.08, -0.26],
    [0, 1, 0],
    0.078,
    chainCol,
    true,
    20,
    true,
    0.16,
    0.18,
  );

  // Outer channel-wise residual gate[8] ribbon directly from layerMod row 2 -> gated_delta
  b.flow.addCubic3D(
    [xLayerMod + 0.82, BASE_Y + 0.04, 0.40],
    [xLayerMod + 2.20, BASE_Y + 0.32, 1.05],
    [xGated - 1.50, BASE_Y + 0.32, 1.05],
    [xGated - 0.60, BASE_Y + 0.06, 0.92],
    [0, 1, 0],
    0.086,
    gateCol,
    true,
    22,
    true,
    0.16,
    0.18,
  );

  // Untouched emerald Residual Highway: X bypasses modulation completely and merges at Y!
  b.flow.addCubic3D(
    [xResX + 0.76, BASE_Y + 0.08, zResBypass],
    [xResX + 2.10, BASE_Y + 0.24, zResBypass],
    [xResY - 0.95, BASE_Y + 0.24, zResBypass * 0.75],
    [xResY, BASE_Y + 0.08, -1.06],
    [0, 1, 0],
    0.13,
    resCol,
    true,
    24,
    true,
    0.22,
    0.24,
  );

  // Fork from untouched X into stage1
  b.flow.addCubic3D(
    [xResX + 0.15, BASE_Y + 0.06, zResBypass + 0.28],
    [xResX + 0.35, BASE_Y + 0.16, -1.20],
    [xStage1 - 0.25, BASE_Y + 0.16, -0.75],
    [xStage1, BASE_Y + 0.06, -0.26],
    [0, 1, 0],
    0.075,
    resCol,
    true,
    18,
    true,
    0.14,
    0.16,
  );

  // Stage1 -> Stage2 -> Mixer_out -> Gated Delta -> Residual Y
  connectHorizontal(xStage1 + 0.54, xStage2 - 0.54, zAppTrack, hasCellTrace ? RIBBON_DIMMED : RIBBON_QUIET, 0.072);
  connectHorizontal(xStage2 + 0.54, xMixerOut - 0.52, zAppTrack, hasCellTrace ? RIBBON_DIMMED : RIBBON_QUIET, 0.072);
  b.flow.addCubic3D(
    [xMixerOut + 0.48, BASE_Y + 0.04, zAppTrack + 0.10],
    [xMixerOut + 0.72, BASE_Y + 0.14, 0.50],
    [xGated - 0.45, BASE_Y + 0.14, 0.80],
    [xGated - 0.58, BASE_Y + 0.06, 0.90],
    [0, 1, 0],
    0.075,
    gateCol,
    true,
    16,
    true,
    0.14,
    0.16,
  );
  b.flow.addCubic3D(
    [xGated + 0.58, BASE_Y + 0.06, 0.85],
    [xGated + 0.85, BASE_Y + 0.20, 0.20],
    [xResY + 0.10, BASE_Y + 0.20, -0.25],
    [xResY, BASE_Y + 0.08, -0.50],
    [0, 1, 0],
    0.088,
    hasCellTrace ? RIBBON_DIMMED : RIBBON_PATHWAY_SOFT,
    true,
    18,
    true,
    0.16,
    0.18,
  );

  // =========================================================================
  // CELL HIGHLIGHTS, OPERAND THREADS, POPOVER & ACTIVE CELL GRIDS
  // =========================================================================
  let cellPopover: CellPopoverSpec | null = null;
  let activeOpPartIds: ReadonlySet<string> | undefined;

  const resolveSpan = (span: TimestepSheetSpan): WorldSpan3D | null => {
    const rec = b.sheetRegistry.get(span.tensor);
    if (!rec) return null;
    return spanFromSheetGrid(rec, span.rowStart, span.rowEnd, span.colStart, span.colEnd);
  };

  if (activeFormula) {
    const partIds = new Set<string>([partIdForTimestepTensor(activeFormula.targetSpan.tensor)]);
    const targetSpan = resolveSpan(activeFormula.targetSpan);
    if (targetSpan) {
      drawCellHighlight3D(b.flow, targetSpan, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.024, true);
      const sourceWorlds: {
        readonly world: readonly [number, number, number];
        readonly role: 'indigo' | 'teal';
      }[] = [];
      let hasOpA = false;
      let hasOpB = false;
      for (const up of activeFormula.upstreamSpans) {
        partIds.add(partIdForTimestepTensor(up.tensor));
        const upSpan = resolveSpan(up);
        if (!upSpan) continue;
        const isA = up.role === 'operand_a';
        if (isA) hasOpA = true;
        else hasOpB = true;
        sourceWorlds.push({ world: upSpan.frontPoint, role: isA ? 'indigo' : 'teal' });
        drawOperandDependency3D(b.flow, upSpan, targetSpan, isA);
      }
      const operandColors: ('indigo' | 'teal' | 'red')[] = [];
      if (hasOpA) operandColors.push('indigo');
      if (hasOpB) operandColors.push('teal');
      operandColors.push('red');
      cellPopover = {
        world: targetSpan.frontPoint,
        tensorLabel: activeFormula.tensorLabel,
        coordLabel: activeFormula.coordLabel,
        badge: activeFormula.headBadge ?? undefined,
        equation: activeFormula.equation,
        valueText: activeFormula.valueText.startsWith('=')
          ? activeFormula.valueText
          : `= ${activeFormula.valueText}`,
        operandColors,
        sourceWorlds: sourceWorlds.length > 0 ? sourceWorlds : undefined,
      };
    }
    activeOpPartIds = partIds;
  }

  // Zoom-adaptive cell grids for key live tensors
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const chLabels = Array.from({ length: d }, (_, ch) => `c${ch}`);
  const bandLabels = ['shift', 'scale', 'gate'];

  const pushGrid = (
    key: TimestepTensorKey,
    values: Float32Array,
    optsGrid: {
      readonly format?: 'signed' | 'prob';
      readonly colLabels?: readonly string[];
      readonly rowLabels?: readonly string[];
      readonly colDimLabel?: string;
      readonly rowDimLabel?: string;
    },
  ) => {
    const rec = b.sheetRegistry.get(key);
    if (!rec) return;
    const frame =
      rec.frame ?? makeSheetFrame3D([rec.cx, rec.cy, rec.cz], [rec.ex, rec.ey, rec.ez]);
    activeCellGrids.push({
      originWorld: frame.center,
      extent: frame.extent,
      axisU: frame.axisU,
      axisN: frame.axisN,
      cols: rec.gridCols,
      rows: rec.gridRows,
      values,
      format: optsGrid.format ?? 'signed',
      activeRow: activeFormula?.cell.tensor === key ? activeFormula.cell.row : undefined,
      activeCol: activeFormula?.cell.tensor === key ? activeFormula.cell.col : undefined,
      colLabels: optsGrid.colLabels,
      rowLabels: optsGrid.rowLabels,
      colDimLabel: optsGrid.colDimLabel,
      rowDimLabel: optsGrid.rowDimLabel,
    });
  };

  pushGrid('layer_mod', new Float32Array(layerModFlat), {
    colLabels: chLabels,
    rowLabels: bandLabels,
    colDimLabel: `L${layerIndex} (shift/scale/gate)`,
  });
  pushGrid('phi', new Float32Array(phi), {
    colLabels: Array.from({ length: half }, (_, k) => `k${k}`),
    rowLabels: ['cos', 'sin'],
    colDimLabel: 'φ(t) [16]',
  });
  pushGrid('t_emb', new Float32Array(tEmb), {
    colLabels: chLabels,
    rowLabels: ['sh_g', 'sc_g', 'gt_g'],
    colDimLabel: 'tEmb [24]',
  });
  pushGrid('residual_x', new Float32Array(appVecs.x), {
    colLabels: chLabels,
    rowLabels: [tokTag],
    colDimLabel: 'Untouched X',
  });
  pushGrid('gated_delta', new Float32Array(appVecs.gatedDelta), {
    colLabels: chLabels,
    rowLabels: [tokTag],
    colDimLabel: 'gate[8] ⊙ Mixer',
  });

  // =========================================================================
  // LANDMARKS, CALIPERS, PART CATALOG & OP FOCUS BOUNDS
  // =========================================================================
  const bayStr = `Bay ${String(bayId).padStart(2, '0')} (L${layerIndex} · ${bayKindDisplayName(mixerKind)})`;
  const modEq =
    modulationOrder === 'modulate_then_norm'
      ? 'X → X_mod = X·(1+scale)+shift → RMSNorm(X_mod) → Mixer → gate ⊙ Mixer_out → + X'
      : 'X → RMSNorm(X) → U → U_mod = U·(1+scale)+shift → MoE → gate ⊙ MoE_out → + X';

  const tScalarLandmarkLabel =
    conditioningMode === 'bypassed_ar'
      ? 'BYPASSED in Context AR (t = null)'
      : conditioningMode === 'bypassed_mock_ar'
        ? 'BYPASSED in Mock-AR (t = null)'
        : inactiveNoPassYet
          ? ts.nextEligibleStep
            ? `INACTIVE · No pass yet (Next t = ${ts.nextEligibleStep.nextNMasked}/${ts.nextEligibleStep.blockSize} = ${ts.nextEligibleStep.nextTModel.toFixed(3)})`
            : 'INACTIVE · No diffusion pass yet (t = null)'
          : isDiagnosticOverride
            ? `${isHistoricalPass ? `HISTORICAL Rec #${ts.executedPass!.sourceRecordIndex} · ` : ''}t_override = ${overrideValue!.toFixed(3)} (empirical ${nMasked}/${blockSize} = ${empiricalMaskFraction!.toFixed(3)})`
            : `${isHistoricalPass ? `HISTORICAL Rec #${ts.executedPass!.sourceRecordIndex} · ` : ''}t = nMasked / B = ${nMasked}/${blockSize} (${tModel!.toFixed(3)})`;
  const tScalarLandmarkSublabel = bypassed
    ? 'All AdaLN modulation sheets dimmed (AR / Mock-AR bypass)'
    : inactiveNoPassYet
      ? 'Mask-diffusion ready · Canonical t = nMasked / B activates on pass 0'
      : isHistoricalPass
        ? `Prior pass (b${ts.executedPass!.sourceBlockId ?? '-'} r${ts.executedPass!.sourceRefinementIndex ?? '-'}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'} ≠ P=${view.committedPrefixLen}) · t_scaled = ${(tModel! * 1000).toFixed(1)}`
        : `t_scaled = ${(tModel! * 1000).toFixed(1)} · ${ts.blockCompletionNote}`;

  b.landmarks.push(
    {
      id: 'time-t-scalar',
      label: tScalarLandmarkLabel,
      sublabel: tScalarLandmarkSublabel,
      world: [xScalar, 0.24, 0.74],
      partId: 't-scalar',
      align: 'bottom',
      tier: 'primary',
      tone: sheetsDimmed ? 'muted' : 'gold',
    },
    {
      id: 'time-phi',
      label: 'Sinusoidal φ(t) [16]',
      sublabel: 'COS-FIRST: cos(0..7), sin(0..7) · FP32',
      world: [xPhi, 0.24, -0.78],
      partId: 'phi-sinusoid',
      align: 'top',
      tier: 'primary',
      tone: sheetsDimmed ? 'muted' : 'cyan',
    },
    {
      id: 'time-mlp',
      label: 'Timestep MLP (W₀ → SiLU → W₂ → tRepr)',
      sublabel: '[16 → 8 → 8] mini ([256 → 2688 → 2688] prod)',
      world: [(xW0 + xTRepr) * 0.5, WALL_Y + 0.96, 0],
      partId: 't-repr',
      align: 'top',
      tier: 'primary',
      tone: sheetsDimmed ? 'muted' : 'default',
    },
    {
      id: 'time-tblock',
      label: 'Global t_block [24×8] → tEmb [3d]',
      sublabel: 'SiLU(tRepr) → [shift_g(8), scale_g(8), gate_g(8)]',
      world: [(xTBlockW + xTEmb) * 0.5, WALL_Y + 1.48, 0],
      partId: 't-emb',
      align: 'top',
      tier: 'primary',
      tone: sheetsDimmed ? 'muted' : 'gold',
    },
    {
      id: 'time-layer-table',
      label: `Learned scale_shift_tables[${layerIndex}]`,
      sublabel: `+ tEmb → ${bayStr}`,
      world: [xLayerMod, 0.26, zTable - 0.82],
      partId: 'scale-shift-table',
      align: 'top',
      tier: 'primary',
      tone: sheetsDimmed ? 'muted' : 'default',
    },
    {
      id: 'time-layer-mod',
      label: `${bayStr} · (shift, scale, gate) [3×8]`,
      sublabel: bypassed
        ? conditioningMode === 'bypassed_ar'
          ? 'BYPASSED in Context AR (t = null)'
          : 'BYPASSED in Mock-AR (t = null)'
        : inactiveNoPassYet
          ? 'INACTIVE (no diffusion pass yet)'
          : `‖sh‖=${shiftL2.toFixed(2)} · ‖sc‖=${scaleL2.toFixed(2)} · ‖gt‖=${gateL2.toFixed(2)}`,
      world: [xLayerMod, 0.24, 0.92],
      partId: 'layer-mod',
      align: 'bottom',
      tier: 'primary',
      tone: sheetsDimmed ? 'muted' : 'gold',
    },
    {
      id: 'time-residual-highway',
      label: 'Untouched Original Residual Highway (X → + Y)',
      sublabel: 'Never modulated by shift/scale',
      world: [(xResX + xResY) * 0.5, 0.36, zResBypass - 0.46],
      partId: 'residual-x',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'time-order-banner',
      label: `${bayStr}: ${modulationOrder}`,
      sublabel: modEq,
      world: [(xStage1 + xMixerOut) * 0.5, 0.28, -0.82],
      partId: 'layer-stage1',
      align: 'top',
      tier: 'primary',
      tone: modulationOrder === 'norm_then_modulate' ? 'gold' : 'cyan',
    },
    {
      id: 'time-outer-gate',
      label: 'Outer Gate (d = 8) vs Mamba SiLU(z) (d_inner = 64)',
      sublabel: 'Y = X + gate[8] ⊙ Mixer_out · Distinct from Mamba norm_before_gate=False SiLU(z)',
      world: [4.55, 0.20, 2.86],
      partId: 'gate-contrast',
      align: 'bottom',
      tier: 'primary',
      tone: 'gold',
    },
    {
      id: 'time-final-norm-ref',
      label: 'Final RMSNorm (norm_f): Strictly UNMODULATED',
      sublabel: 'No shift, scale, or gate after Bay 52 in either tower',
      world: [8.35, 0.20, 2.86],
      partId: 'final-norm-unmodulated',
      align: 'bottom',
      tier: 'primary',
      tone: 'cyan',
    },
  );

  b.calipers.push({
    id: 'cal-outer-vs-inner-gate',
    aWorld: [xGated - 0.56, 0.04, 1.32],
    bWorld: [xGated + 0.56, 0.04, 1.32],
    label: 'Outer AdaLN Gate d = 8 (prod 2,688) ≠ Mamba internal SiLU(z) d_inner = 64 (prod 4,096)',
    partIds: ['gated-delta', 'gate-contrast'],
    minPixels: 64,
    offsetPx: 10,
  });

  const partCatalog: Record<string, CutawayPartInfo> = {
    't-scalar': {
      id: 't-scalar',
      title: 'Masked Fraction Scalar t = nMasked / B',
      badge: bypassed
        ? conditioningMode === 'bypassed_ar'
          ? 'BYPASSED · AR (t = null)'
          : 'BYPASSED · Mock-AR (t = null)'
        : inactiveNoPassYet
          ? 'INACTIVE · No Pass Yet'
          : isDiagnosticOverride
            ? `t_override = ${overrideValue!.toFixed(4)} (emp ${empiricalMaskFraction!.toFixed(4)})`
            : `t = ${tModel!.toFixed(4)}`,
      summary: bypassed
        ? `${bypassReason}`
        : inactiveNoPassYet
          ? `Mask-diffusion conditioning is INACTIVE because no Denoiser pass has executed yet (NOT an AR/Mock-AR bypass). ${ts.blockCompletionNote}`
          : 'Canonical fraction of masked slots entering the executed Denoiser pass (`t = nMasked / B`, NEVER `r / K`). Scaled by maxPeriodScale = 1000 before sinusoidal frequency encoding.',
      specs: [
        `Conditioning mode: ${conditioningMode} · Block state: ${ts.blockCompletionState}`,
        ...(hasExecutedDiffusionPass && ts.executedPass
          ? [
              isDiagnosticOverride
                ? `Diagnostic override t_override = ${overrideValue!.toFixed(4)} (empirical input ${nMasked}/${blockSize} = ${empiricalMaskFraction!.toFixed(4)})`
                : `Executed input masks: nMasked = ${nMasked}, B = ${blockSize} → t = ${tModel!.toFixed(4)}`,
              `Source pass: Rec #${ts.executedPass.sourceRecordIndex} (b${ts.executedPass.sourceBlockId ?? '-'}, r${ts.executedPass.sourceRefinementIndex ?? '-'}, ${ts.executedPass.passExecutionId}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'})`,
            ]
          : [`Status: ${statusLabel}`]),
        ...(ts.currentBlockMask
          ? [
              `Current block state: ${ts.currentBlockMask.currentUnresolvedCount}/${ts.currentBlockMask.blockSize} masked (${ts.blockCompletionState})`,
            ]
          : []),
        ...(ts.nextEligibleStep
          ? [
              `Next eligible pass: t = ${ts.nextEligibleStep.nextNMasked}/${ts.nextEligibleStep.blockSize} = ${ts.nextEligibleStep.nextTModel.toFixed(4)} (${ts.nextEligibleStep.remainingSteps} steps left)`,
            ]
          : []),
        'Outer scale: 1000 · Max period: 10000 (IEEE-754 Float32)',
      ],
    },
    'phi-sinusoid': {
      id: 'phi-sinusoid',
      title: 'Sinusoidal Timestep Embedding φ(t)',
      badge: bypassed ? 'BYPASSED' : inactiveNoPassYet ? 'INACTIVE' : 'FP32 · [16] mini ([256] prod)',
      summary:
        'Log-spaced sinusoidal frequency embedding in strict COS-FIRST ordering: φ[0..7] = cos(t·1000·ω_k), φ[8..15] = sin(t·1000·ω_k).',
      specs: [
        'ω_k = exp(-ln(10000) · k / 8) in IEEE-754 Float32',
        'Matches official Nemotron-Labs-TwoTower TimestepEmbedder',
      ],
    },
    'mlp0-w': {
      id: 'mlp0-w',
      title: 'Timestep MLP Linear 0 (W₀)',
      badge: '[8×16] mini ([2,688×256] prod)',
      summary: 'First linear projection of t_embedder.mlp followed by SiLU.',
      specs: [`Time conditioning fingerprint: ${weights.fingerprint}`],
    },
    'mlp-h': {
      id: 'mlp-h',
      title: 'Timestep MLP Hidden SiLU(h₀)',
      badge: 'SiLU(W₀·φ + b₀)',
      summary: 'Non-linear hidden state inside the global 2-layer timestep MLP.',
      specs: ['Width: d = 8 miniature (2,688 production)'],
    },
    'mlp2-w': {
      id: 'mlp2-w',
      title: 'Timestep MLP Linear 2 (W₂)',
      badge: '[8×8] mini ([2,688×2,688] prod)',
      summary: 'Second linear projection of t_embedder.mlp producing global tRepr.',
      specs: ['Shared across all 52 Denoiser layers'],
    },
    't-repr': {
      id: 't-repr',
      title: 'Global Timestep Representation (tRepr)',
      badge: '[8] mini ([2,688] prod)',
      summary: 'Evaluated once per denoising step and shared by all 52 Denoiser layers.',
      specs: ['Feeds SiLU(tRepr) → W_time_block [24×8]'],
    },
    't-block-w': {
      id: 't-block-w',
      title: 'Global 3d Modulation Projection (W_time_block)',
      badge: '[24×8] mini ([8,064×2,688] prod)',
      summary: 'PixArt-alpha adaLN-single shared linear projection generating global [shift_g, scale_g, gate_g].',
      specs: ['Production params: 21.68M (t_block.1.weight + bias)'],
    },
    't-emb': {
      id: 't-emb',
      title: 'Global 3d Modulation Vector (tEmb)',
      badge: '[shift_g(8), scale_g(8), gate_g(8)]',
      summary: 'Shared global modulation vector added elementwise to each layer’s learned scale_shift_tables[l].',
      specs: ['Evaluated once per denoising step for all 52 layers'],
    },
    'scale-shift-table': {
      id: 'scale-shift-table',
      title: `Per-Layer Learned Table scale_shift_tables[${layerIndex}]`,
      badge: `${bayStr} · [3×8]`,
      summary: 'Layer-specific learned modulation offsets (~randn / √d init) specializing global tEmb for this bay.',
      specs: ['52 independent learned tables (one per Denoiser layer, 0 in Context Tower)'],
    },
    'layer-mod': {
      id: 'layer-mod',
      title: `${bayStr} · Effective (shift, scale, gate)`,
      badge: bypassed ? 'BYPASSED' : inactiveNoPassYet ? 'INACTIVE' : `${modulationOrder}`,
      summary:
        'Elementwise sum tEmb + scale_shift_tables[l], sliced into shift[8], scale[8], and outer residual gate[8].',
      specs: [
        `‖shift‖₂ = ${shiftL2.toFixed(4)} · ‖scale‖₂ = ${scaleL2.toFixed(4)} · ‖gate‖₂ = ${gateL2.toFixed(4)}`,
        `Ordering: ${modEq}`,
      ],
    },
    'residual-x': {
      id: 'residual-x',
      title: 'Untouched Original Residual Highway (X)',
      badge: 'Original Residual Provenance',
      summary:
        'Original layer input X bypasses shift/scale modulation completely and merges untouched into Y = X + gate ⊙ Mixer_out.',
      specs: [
        'Preserves clean residual identity across all 52 layers',
        'Modulation alters only the branch entering the mixer, never the skip path X',
      ],
    },
    'layer-stage1': {
      id: 'layer-stage1',
      title: appVecs.stage1Label,
      badge: modulationOrder,
      summary:
        modulationOrder === 'modulate_then_norm'
          ? 'Mamba-2 (M) and GQA (*) apply AdaLN modulation BEFORE RMSNorm: X_mod = X·(1+scale)+shift.'
          : 'Granular MoE (E) applies RMSNorm BEFORE AdaLN modulation: U = RMSNorm(X).',
      specs: [`Selected bay: ${bayStr}`, `Equation: ${modEq}`],
    },
    'layer-stage2': {
      id: 'layer-stage2',
      title: appVecs.stage2Label,
      badge: modulationOrder,
      summary:
        modulationOrder === 'modulate_then_norm'
          ? 'Mamba-2 (M) and GQA (*) apply RMSNorm AFTER AdaLN modulation: U = RMSNorm(X_mod).'
          : 'Granular MoE (E) applies AdaLN modulation AFTER RMSNorm: U_mod = U·(1+scale)+shift.',
      specs: [`Feeds ${bayKindDisplayName(mixerKind)} mixer directly`],
    },
    'mixer-out': {
      id: 'mixer-out',
      title: `${bayKindDisplayName(mixerKind)} Output (${appVecs.mixerOutLabel})`,
      badge: `d = ${d} · ${tokTag}`,
      summary: 'Raw mixer output prior to the outer channel-wise AdaLN residual gate.',
      specs: [`Pulled from ${appVecs.hasLiveTrace ? 'live Denoiser bay trace' : 'Context fallback / idle state'}`],
    },
    'gated-delta': {
      id: 'gated-delta',
      title: 'Outer Channel-Wise Residual Gate (gate ⊙ Mixer_out)',
      badge: 'Outer AdaLN Gate (d = 8)',
      summary:
        'Multiplies the mixer output channel-wise by outer gate[8] before adding to untouched residual X. Strictly distinct from Mamba-2’s internal d_inner = 64 SiLU(z) gate.',
      specs: [
        'Outer AdaLN gate dim: d = 8 mini (2,688 prod), applied on ALL 52 Denoiser bays (M, *, E)',
        'Mamba internal gate dim: d_inner = 64 mini (4,096 prod), applied inside MambaRMSNormGated (norm_before_gate = False)',
      ],
    },
    'residual-y': {
      id: 'residual-y',
      title: 'Residual Output (Y = X + gate ⊙ Mixer_out)',
      badge: 'Emerald Residual Stream',
      summary: 'Adds the outer-gated mixer update back into untouched residual X.',
      specs: [`Focused token: ${tokTag}`],
    },
    'gate-contrast': {
      id: 'gate-contrast',
      title: 'Outer Residual Gate (d=8) vs Mamba Internal SiLU(z) Gate (d_inner=64)',
      badge: 'Two Distinct Gates',
      summary:
        'In TwoTower Denoiser Mamba-2 layers, TWO distinct gates operate at different widths and positions: (1) Mamba’s internal d_inner=64 SiLU(z) gate before group RMSNorm inside the mixer, and (2) the outer d=8 AdaLN residual gate scaling Mamba_out before adding to untouched X.',
      specs: [
        'Outer AdaLN residual gate: d = 8 (2,688 prod) — present on all 52 layers (M, *, E)',
        'Mamba internal SiLU(z) gate: d_inner = 64 (4,096 prod) — inside 23 Mamba-2 mixers only',
      ],
    },
    'final-norm-unmodulated': {
      id: 'final-norm-unmodulated',
      title: 'Final RMSNorm (norm_f) — Strictly Unmodulated',
      badge: 'finalNormModulated = false',
      summary:
        'After Bay 52, Final RMSNorm (norm_f, ε = 1e-5) is strictly UNMODULATED in both Context and Denoiser towers — no shift, scale, or gate is applied before W_lm_head.',
      specs: ['Verified directly against official checkpoint safetensors (52 scale_shift_tables, zero norm_f modulation)'],
    },
  };

  const reg = (k: string) => b.sheetRegistry.get(k);
  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('t_scalar')], 0.75, 0.45, 0.75, TIMESTEP_SPATIAL_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('t_scalar'), reg('phi')], 0.60, 0.40, 0.65, TIMESTEP_SPATIAL_FIT_BOUNDS),
    2: computeSheetGroupBounds([reg('mlp0_w'), reg('mlp_h'), reg('mlp2_w'), reg('t_repr')], 0.55, 0.35, 0.55, TIMESTEP_SPATIAL_FIT_BOUNDS),
    3: computeSheetGroupBounds([reg('t_repr'), reg('t_block_w'), reg('t_emb')], 0.55, 0.35, 0.55, TIMESTEP_SPATIAL_FIT_BOUNDS),
    4: computeSheetGroupBounds([reg('t_emb'), reg('scale_shift_table'), reg('layer_mod')], 0.55, 0.35, 0.55, TIMESTEP_SPATIAL_FIT_BOUNDS),
    5: computeSheetGroupBounds(
      [reg('layer_mod'), reg('residual_x'), reg('stage1'), reg('stage2'), reg('mixer_out'), reg('gated_delta'), reg('residual_y')],
      0.55, 0.35, 0.55, TIMESTEP_SPATIAL_FIT_BOUNDS,
    ),
  };

  const provenanceBadges: TwoTowerProvenanceBadge[] = [
    {
      label: `Timestep Conditioning: ${statusLabel}`,
      tone:
        conditioningMode === 'active_diffusion'
          ? 'gold'
          : isHistoricalPass
            ? 'warn'
            : 'muted',
      origin: traceOrigin,
    },
    {
      label: `${bayStr} · ${modulationOrder}`,
      tone: modulationOrder === 'norm_then_modulate' ? 'gold' : 'denoiser',
    },
    {
      label: 'Outer Gate d=8 ≠ Mamba SiLU(z) d_inner=64',
      tone: 'shared',
    },
    {
      label: 'Final RMSNorm norm_f: Unmodulated',
      tone: 'shared',
    },
  ];
  if (dl.isHistoricalInspection) {
    provenanceBadges.push({
      label: `Historical Pass · Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'} vs P=${view.committedPrefixLen})`,
      tone: 'warn',
      origin: 'denoiser_scratch',
    });
  }

  const activeSelection = buildTwoTowerSelectionIdentity(view, {
    towerRole: denoiserPass ? 'denoiser' : 'shared',
    bayId,
    mixerKind: 'time_cond',
    tensorRole: activeCell?.tensor ?? 'layer_mod',
    traceOrigin,
    executionId,
    absoluteTokenPosition,
    provenanceNote: `${statusLabel} · ${bayStr} · ${modulationOrder}`,
  });

  const diagnostics: TimestepCutawayDiagnostics = Object.freeze({
    bypassed,
    inactiveNoPassYet,
    conditioningMode,
    timestepStatus: ts,
    bypassReason,
    statusLabel,
    tModel,
    nMasked,
    blockSize,
    empiricalMaskFraction,
    isDiagnosticOverride,
    overrideValue,
    isHistoricalDenoiserInspection: dl.isHistoricalInspection,
    matchesCurrentContextBoundary: dl.matchesCurrentContextBoundary,
    sourceRecordIndex: dl.sourceRecordIndex,
    sourceBlockId: dl.sourceBlockId,
    sourceRefinementIndex: dl.sourceRefinementIndex,
    sourceContextPrefixLen: dl.sourceContextPrefixLen,
    sourceContextCacheDigest: dl.sourceContextCacheDigest,
    historicalProvenanceWarning: dl.historicalProvenanceWarning,
    bayId,
    layerIndex,
    mixerKind,
    modulationOrder,
    modulationOrderEquation: modEq,
    shiftL2,
    scaleL2,
    gateL2,
    outerGateDim: 8 as const,
    mambaInternalSiluGateDim: 64 as const,
    outerVsInternalGateNote:
      'Outer AdaLN residual gate operates on d=8 residual channels (Y = X + gate ⊙ Mixer_out across all 52 layers), strictly distinct from Mamba-2 internal SiLU(z) gate operating on d_inner=64 inside MambaRMSNormGated.',
    finalNormModulated: false as const,
    finalNormNote: 'Final RMSNorm (norm_f, ε=1e-5) is strictly unmodulated in both Context and Denoiser towers.',
    residualPreservedUntouched: true as const,
    localTokenIndex,
    absoluteTokenPosition,
    tokenLabel,
    hasLiveDenoiserTrace: appVecs.hasLiveTrace,
    timeConditioningFingerprint: weights.fingerprint,
    phi: sheetsDimmed ? null : Object.freeze(Array.from(phi)),
    tRepr: sheetsDimmed ? null : Object.freeze(Array.from(tRepr)),
    tEmb: sheetsDimmed ? null : Object.freeze(Array.from(tEmb)),
    shift: layerMod ? Object.freeze(Array.from(layerMod.shift)) : null,
    scale: layerMod ? Object.freeze(Array.from(layerMod.scale)) : null,
    gate: layerMod ? Object.freeze(Array.from(layerMod.gate)) : null,
    scaleShiftTable: Object.freeze(Array.from(weights.scaleShiftTables[layerIndex])),
  });

  const warningParts: string[] = [];
  if (!view.snapshotIntegrity.valid) {
    warningParts.push(`Context snapshot integrity error: ${view.snapshotIntegrity.error}`);
  }
  if (dl.historicalProvenanceWarning !== null) {
    warningParts.push(dl.historicalProvenanceWarning);
  }
  if (bypassReason !== null) {
    warningParts.push(bypassReason);
  }
  const warningBanner = warningParts.length > 0 ? warningParts.join(' ') : null;

  const packed = b.pack();
  const atlasKey =
    `tt-time|${view.model.twoTowerFingerprint}|${executionId}|${conditioningMode}|src${dl.sourceRecordIndex ?? 'none'}|b${bayId}|tok${localTokenIndex}|` +
    `${sheetsDimmed ? conditioningMode : tModel!.toFixed(6)}`;

  return {
    viewMode: 'timestep',
    title: `Timestep Conditioning & Split AdaLN-Single · ${bayStr}`,
    subtitle: `${statusLabel} · ${modulationOrder}`,
    bayId,
    layerIndex,
    activeTowerFocus: 'denoiser_solo',
    ...packed,
    valueAtlas: atlas,
    atlasKey,
    fitBounds: TIMESTEP_SPATIAL_FIT_BOUNDS,
    opFocusBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    calipers: b.calipers,
    solidPickBoxes: b.pickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeOpPartIds,
    activeCellGrids,
    cellPopover,
    activeFormula,
    focusedAccumulation: null,
    activeSelection,
    stepperSteps: TIMESTEP_STEPPER_STEPS,
    provenanceBadges,
    epistemicNote:
      'Educational miniature TwoTower Float32 simulator (d=8, freqDim=16, 52 layers) preserving exact PixArt-alpha adaLN-single split modulation ordering (M/* modulate→norm, E norm→modulate, unmodulated norm_f); NOT 63.19B checkpoint weights.',
    warningBanner,
    diagnostics: diagnostics as unknown as Readonly<Record<string, unknown>>,
  };
}
