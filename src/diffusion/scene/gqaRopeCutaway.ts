/**
 * 32Q/8KV RoPE + YaRN + Llama-4 + Draft-Only LoRA GQA Cutaway
 * (`src/diffusion/scene/gqaRopeCutaway.ts`)
 *
 * Interactive 5-stage drafting-table spatial cutaway for any of the 40 layers:
 *   1. Input `X` & Pre-Attn `RMSNorm U` -> `W_Q [8×256]`, `W_K [8×64]`, `W_V [8×64]`
 *      (with explicit caliper & badge for the `8→256/64` vs `5120→4096/1024` projection reversal)
 *   2. Half-split YaRN RoPE (`Q_pre / K_pre -> Q_rot / K_rot`) highlighting all 4 dyadic
 *      rotary planes `(0,4)..(3,7)` (`1.0, 0.3125, 0.0625, 0.03125`) + `rope_cos/sin/inv_freq`
 *   3. Live Llama-4 query-only temperature scaling (`×1.000` for `p<16` -> `×1.069` for `p>=16`)
 *   4. 32Q/8KV (4:1) Attention: 8 KV-group `Scores` & `Softmax` sheets (ALL D24-bound to live
 *      `RG32F` slices with `G ∈ {0, 1}`) + `K_full`/`V_full` bank (`groupIndex: 0..7`)
 *   5. `o_proj` + Optional Draft-Only LoRA Delta Branch (`+4.0·B(A·a)`) -> `attn_out` + residual
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`,
 * or `src/twotower/` / `src/math/gqaSim` / `src/math/moeSim` imports.
 */

import type { LayerFormulaReadout, StepperStepDef } from '../../math/formulaFormat';
import { fmtSigned } from '../../shared/format';
import type { Vec3 } from '../../render/primitives';
import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import { FlowRibbonBuilder, type FlowRibbonColor } from '../../render/ribbons';
import {
  FRAME_XZ_AXES,
  localToWorldOnFrame,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
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
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import type { AtlasSlice } from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  RIBBON_ACTIVE,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  RIBBON_RESIDUAL_ACTIVE,
  RIBBON_SSM_LOOP,
} from '../../scene/sheetInspector';
import type { DimensionCaliper, LandmarkAnchor } from '../../render/primitives';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../shared/overlayLayout';
import { formatDiffusionAttentionCellFormula, linearProject } from '../attention';
import {
  computeLlama4AttnScale,
  computeYarnInvFreq,
  formatRopeCellFormula,
} from '../rope';
import type {
  DiffusionLayerTrace,
  DiffusionMiniConfig,
} from '../types';
import {
  buildDiffusionGqaRopeAtlas,
  DIFFUSION_GQA_MAX_KEY,
} from './gqaRopeAtlas';
import {
  DIFFUSION_EPISTEMIC_NOTE,
  DIFFUSION_LORA_RAIL_BADGE,
  DIFFUSION_PROJECTION_REVERSAL_BADGE,
  LineBufferBuilder,
  RibbonBufferBuilder,
  SheetBufferBuilder,
  SolidBufferBuilder,
  validateSceneGeometryFinite,
} from './provenance';
import type {
  DiffusionFocusState,
  DiffusionPickIdentity,
  DiffusionProvenanceBadge,
  DiffusionResolvedSessionContext,
  DiffusionRopePhasorTelemetry,
  DiffusionSpatialScene,
} from './types';

const SHEET_THICKNESS = 0.08;
const RIBBON_Y = 0.048;

const RIBBON_LORA_ACTIVE: FlowRibbonColor = {
  r: 0.855,
  g: 0.494,
  b: 0.118,
  a: 0.86,
  borderAlpha: 0.45,
};
const RIBBON_LLAMA4_GOLD: FlowRibbonColor = {
  r: 0.765,
  g: 0.545,
  b: 0.145,
  a: 0.84,
  borderAlpha: 0.44,
};

const GQA_ROPE_STEPPER_STEPS: readonly StepperStepDef[] = Object.freeze([
  Object.freeze({
    index: 0,
    shortLabel: 'RMSNorm & QKV',
    title: 'Pre-Attention RMSNorm & Expanding Mini QKV Projections',
    summary:
      'Normalizes hidden states x [S×8] into u [S×8], then projects into W_Q [8×256], W_K [8×64], and W_V [8×64] (mini expands 8→256/64; prod 14B contracts 5120→4096/1024).',
  }),
  Object.freeze({
    index: 1,
    shortLabel: 'YaRN RoPE',
    title: 'Half-Split Dyadic YaRN Rotary Position Embedding',
    summary:
      'Rotates 4 half-split planes (0,4)..(3,7) with exact dyadic frequencies [1.0, 0.3125, 0.0625, 0.03125] spanning original, blended, and interpolated YaRN zones.',
  }),
  Object.freeze({
    index: 2,
    shortLabel: 'Llama-4 Q Scale',
    title: 'Post-RoPE Llama-4 Query-Only Temperature Scaling',
    summary:
      'Scales rotated queries by 1 + 0.1·ln(1 + ⌊cache_pos / 16⌋) (×1.000 for pos < 16, ×1.0693 for pos >= 16). Keys are NEVER scaled.',
  }),
  Object.freeze({
    index: 3,
    shortLabel: '32Q/8KV Scores',
    title: '4:1 Grouped-Query Scaled Dot-Product & Explicit Mask',
    summary:
      'Each of the 8 KV heads serves 4 Q heads. Masked cells carry −∞ in scores with explicit RG32F channel G=1.0 (D24 anti-contamination).',
  }),
  Object.freeze({
    index: 4,
    shortLabel: 'Softmax & V',
    title: 'Row-Wise Softmax & Value Aggregation into 256-dim Concat',
    summary:
      'Softmax turns finite scores into row probabilities (exact 0.0 on masked cells) and aggregates V_full [K×64] into attn_concat [S×256].',
  }),
  Object.freeze({
    index: 5,
    shortLabel: 'o_proj + LoRA',
    title: 'Output Projection, Optional Draft-Only LoRA Delta & Residual',
    summary:
      'Projects attn_concat through W_O [8×256] and adds +4.0·B(A·a) ONLY on linear_spec_lora draft passes (0 ULP invariance on causal verify).',
  }),
]);

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(v)));
}

// ============================================================================
// 2D UNIT-CIRCLE ROPE PHASOR TELEMETRY
// ============================================================================

export interface ComputeRopePhasorOptions {
  readonly layerIndex: number;
  readonly qHeadIndex: number;
  readonly kvHeadIndex: number;
  readonly rotaryPlane: number;
  readonly queryRow: number;
  readonly keyCol: number;
  readonly positionIds: Int32Array;
  readonly cachePositions: Int32Array;
  readonly keyPositionIds: Int32Array;
}

export function computeRopePhasorTelemetry(
  trace: DiffusionLayerTrace,
  config: DiffusionMiniConfig,
  options: ComputeRopePhasorOptions,
): DiffusionRopePhasorTelemetry {
  const attn = trace.attention;
  const headDim = config.headDim;
  const halfDim = headDim / 2;

  const layerIndex = clampInt(options.layerIndex, 0, config.numLayers - 1);
  const qHeadIndex = clampInt(options.qHeadIndex, 0, config.numAttentionHeads - 1);
  const kvHeadIndex = clampInt(options.kvHeadIndex, 0, config.numKeyValueHeads - 1);
  const rotaryPlane = clampInt(options.rotaryPlane, 0, halfDim - 1);

  const qRow = clampInt(options.queryRow, 0, Math.max(0, attn.seqLen - 1));
  const kCol = clampInt(options.keyCol, 0, Math.max(0, attn.kvLen - 1));

  const yarn = computeYarnInvFreq(config.rope);
  const invFreq = yarn.invFreq[rotaryPlane]!;
  const yarnZone = yarn.zones[rotaryPlane]!;
  const allPlaneInvFreqs: readonly [number, number, number, number] = Object.freeze([
    yarn.invFreq[0]!,
    yarn.invFreq[1]!,
    yarn.invFreq[2]!,
    yarn.invFreq[3]!,
  ] as const);

  const queryPosition =
    qRow < options.positionIds.length ? options.positionIds[qRow]! : qRow;
  const queryCachePos =
    qRow < options.cachePositions.length
      ? options.cachePositions[qRow]!
      : queryPosition;
  const keyPosition =
    kCol < options.keyPositionIds.length ? options.keyPositionIds[kCol]! : kCol;

  const angleRad = queryPosition * invFreq;
  const relAngleRad = (queryPosition - keyPosition) * invFreq;
  const llama4Scale = computeLlama4AttnScale(
    queryCachePos,
    config.rope.llama4Beta,
    config.rope.originalMaxPositionEmbeddings,
  );

  // Extract Q plane pair (p, p + halfDim)
  const qBase = qRow * config.qProjOutDim + qHeadIndex * headDim;
  const qPrePair: readonly [number, number] = Object.freeze([
    attn.qPre[qBase + rotaryPlane] ?? 0,
    attn.qPre[qBase + rotaryPlane + halfDim] ?? 0,
  ] as const);
  const qRotPair: readonly [number, number] = Object.freeze([
    attn.qRot[qBase + rotaryPlane] ?? 0,
    attn.qRot[qBase + rotaryPlane + halfDim] ?? 0,
  ] as const);
  const qScaledPair: readonly [number, number] = Object.freeze([
    attn.qScaled[qBase + rotaryPlane] ?? 0,
    attn.qScaled[qBase + rotaryPlane + halfDim] ?? 0,
  ] as const);

  // Extract K plane pair (p, p + halfDim)
  let kPrePair: readonly [number, number];
  let kRotPair: readonly [number, number];
  if (kCol >= attn.pastLen && kCol - attn.pastLen < attn.seqLen) {
    const localKRow = kCol - attn.pastLen;
    const kBase = localKRow * config.kvProjOutDim + kvHeadIndex * headDim;
    kPrePair = Object.freeze([
      attn.kPre[kBase + rotaryPlane] ?? 0,
      attn.kPre[kBase + rotaryPlane + halfDim] ?? 0,
    ] as const);
    kRotPair = Object.freeze([
      attn.kRot[kBase + rotaryPlane] ?? 0,
      attn.kRot[kBase + rotaryPlane + halfDim] ?? 0,
    ] as const);
  } else {
    const kFullBase = kCol * config.kvProjOutDim + kvHeadIndex * headDim;
    const kr1 = attn.kFull[kFullBase + rotaryPlane] ?? 0;
    const kr2 = attn.kFull[kFullBase + rotaryPlane + halfDim] ?? 0;
    kRotPair = Object.freeze([kr1, kr2] as const);
    // Exact inverse rotation on orthogonal half-split RoPE plane for cached past keys
    const keyAngle = Math.fround(keyPosition * invFreq);
    const invScale = yarn.attentionScaling > 0 ? 1.0 / yarn.attentionScaling : 1.0;
    const c = Math.cos(keyAngle) * invScale;
    const s = Math.sin(keyAngle) * invScale;
    kPrePair = Object.freeze([
      Math.fround(kr1 * c + kr2 * s),
      Math.fround(kr2 * c - kr1 * s),
    ] as const);
  }

  return {
    layerIndex,
    qHeadIndex,
    kvHeadIndex,
    rotaryPlane,
    invFreq,
    yarnZone,
    queryPosition,
    keyPosition,
    angleRad,
    relAngleRad,
    llama4Scale,
    origMaxPos: config.rope.originalMaxPositionEmbeddings,
    qPrePair,
    qRotPair,
    qScaledPair,
    kPrePair,
    kRotPair,
    allPlaneInvFreqs,
    allPlaneZones: yarn.zones,
  };
}

// ============================================================================
// INTERNAL SHEET PLACEMENT HELPER
// ============================================================================

interface RegisteredSheet {
  readonly partId: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly frame: SheetFrame3D;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly headIndex?: number;
  readonly groupIndex?: number;
  readonly tapIndex?: number;
}

// ============================================================================
// MAIN CUTAWAY BUILDER
// ============================================================================

export function buildDiffusionGqaRopeCutaway(
  ctx: DiffusionResolvedSessionContext,
  focus: DiffusionFocusState,
): DiffusionSpatialScene {
  const config = ctx.model.config;
  const layerIndex = clampInt(focus.selectedLayerIndex, 0, config.numLayers - 1);
  const trace = ctx.resolveLayerTrace(layerIndex);
  const layerWeights = ctx.model.layers[layerIndex]!;
  const ckpt = ctx.record.primaryPassCheckpoint;
  const attn = trace.attention;

  const selectedQHead = clampInt(focus.selectedQHead, 0, config.numAttentionHeads - 1);
  const selectedKvHead = clampInt(focus.selectedKvHead, 0, config.numKeyValueHeads - 1);
  const selectedRotaryPlane = clampInt(focus.selectedRotaryPlane, 0, config.headDim / 2 - 1);
  const selectedTokenRow = clampInt(focus.selectedTokenRow, 0, Math.max(0, attn.seqLen - 1));
  const activeStep = clampInt(focus.activeStepperStep, 0, GQA_ROPE_STEPPER_STEPS.length - 1);
  const loraActive = attn.loraEnabled || ctx.loraActiveOnDraft;

  const positionIds = ckpt.positionIds;
  const cachePositions = ckpt.cachePositions;

  // Build exact keyPositionIds [0..kvLen-1]
  const keyPositionIds = new Int32Array(attn.kvLen);
  for (let k = 0; k < attn.kvLen; k++) {
    if (k < attn.pastLen) {
      keyPositionIds[k] = ctx.kvSnapshot.slabs[k]?.position ?? k;
    } else {
      const localIdx = k - attn.pastLen;
      keyPositionIds[k] =
        localIdx < positionIds.length ? positionIds[localIdx]! : k;
    }
  }

  // Build live 256×256 RG32F atlas (throws on overlap/OOB)
  const inputCheckpoint = ckpt.residualCheckpoints[layerIndex];
  const atlasResult = buildDiffusionGqaRopeAtlas(trace, layerWeights, config, {
    layerIndex,
    selectedQHead,
    selectedKvHead,
    positionIds,
    cachePositions,
    keyPositionIds,
    maskMode: attn.maskMode,
    loraEnabled: loraActive,
    snapshotDigest: ctx.kvSnapshot.snapshotDigest,
    inputHiddenStates: inputCheckpoint,
  });

  const S = atlasResult.queryTokens;
  const K = atlasResult.keyTokens;
  const focusRow = clampInt(selectedTokenRow, 0, S - 1);
  const activeCell = focus.pinnedCell ?? focus.hoveredCell;
  const focusKeyCol =
    activeCell &&
    (activeCell.tensorKey === 'scores' || activeCell.tensorKey === 'softmax') &&
    activeCell.col !== null
      ? clampInt(activeCell.col, 0, K - 1)
      : Math.max(0, K - 1);

  const ropePhasor = computeRopePhasorTelemetry(trace, config, {
    layerIndex,
    qHeadIndex: selectedQHead,
    kvHeadIndex: selectedKvHead,
    rotaryPlane: selectedRotaryPlane,
    queryRow: focusRow,
    keyCol: focusKeyCol,
    positionIds,
    cachePositions,
    keyPositionIds,
  });

  const solids = new SolidBufferBuilder();
  const lines = new LineBufferBuilder();
  const ribbons = new RibbonBufferBuilder();
  const sheets = new SheetBufferBuilder();
  const flow = new FlowRibbonBuilder();

  const solidPickBoxes: PickBox[] = [];
  const fasciaPickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const partCatalog: Record<string, CutawayPartInfo> = {};
  const registeredByPart = new Map<string, RegisteredSheet>();
  const registeredByTensor = new Map<string, RegisteredSheet>();

  const absTokenOffset = positionIds.length > 0 ? positionIds[0]! : 0;
  const bayId = layerIndex + 1;

  const isPartFocused = (partId: string): boolean =>
    focus.selectedPartId === partId ||
    focus.hoveredPartId === partId ||
    activeCell?.partId === partId;

  const addSheet = (spec: {
    readonly partId: string;
    readonly tensorKey: string;
    readonly label: string;
    readonly sublabel?: string;
    readonly cx: number;
    readonly cz: number;
    readonly ex: number;
    readonly ez: number;
    readonly gridCols: number;
    readonly gridRows: number;
    readonly sheetKind?: number;
    readonly role: number;
    readonly highlight: number;
    readonly slice: AtlasSlice;
    readonly headIndex?: number;
    readonly groupIndex?: number;
    readonly tapIndex?: number;
    readonly tokenIndex?: number;
    readonly semanticTag?: string;
    readonly badge: string;
    readonly summary: string;
    readonly specs: readonly string[];
    readonly landmarkAlign?: LandmarkAnchor['align'];
    readonly landmarkTier?: LandmarkAnchor['tier'];
    readonly landmarkTone?: LandmarkAnchor['tone'];
  }): RegisteredSheet => {
    const frame = makeSheetFrame3D(
      [spec.cx, 0, spec.cz],
      [spec.ex, SHEET_THICKNESS, spec.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    const aabb = sheetFrameWorldAabb(frame);

    sheets.push({
      cx: spec.cx,
      cy: 0,
      cz: spec.cz,
      ex: spec.ex,
      ey: SHEET_THICKNESS,
      ez: spec.ez,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      sheetKind: spec.sheetKind ?? SHEET_KIND_STANDARD,
      role: spec.role,
      highlight: isPartFocused(spec.partId) ? SHEET_HL_ACTIVE : spec.highlight,
      slice: spec.slice,
      frame,
    });

    solidPickBoxes.push({
      bayId,
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      ...(spec.headIndex !== undefined ? { headIndex: spec.headIndex } : {}),
      ...(spec.groupIndex !== undefined ? { groupIndex: spec.groupIndex } : {}),
      ...(spec.tapIndex !== undefined ? { tapIndex: spec.tapIndex } : {}),
      ...(spec.tokenIndex !== undefined ? { tokenIndex: spec.tokenIndex } : {}),
      absoluteTokenOffset: absTokenOffset,
      semanticTag: spec.semanticTag ?? spec.tensorKey,
      min: aabb.min,
      max: aabb.max,
      frame,
    });

    landmarks.push({
      id: `lm-${spec.partId}`,
      label: spec.label,
      ...(spec.sublabel ? { sublabel: spec.sublabel } : {}),
      world: [spec.cx, 0.22, spec.cz - spec.ez * 0.5 - 0.26],
      bayId,
      partId: spec.partId,
      align: spec.landmarkAlign ?? 'bottom',
      tier: spec.landmarkTier ?? 'primary',
      tone: spec.landmarkTone ?? 'default',
    });

    partCatalog[spec.partId] = {
      id: spec.partId,
      title: spec.label,
      badge: spec.badge,
      summary: spec.summary,
      specs: spec.specs,
    };

    const reg: RegisteredSheet = {
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      label: spec.label,
      frame,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      headIndex: spec.headIndex,
      groupIndex: spec.groupIndex,
      tapIndex: spec.tapIndex,
    };
    registeredByPart.set(spec.partId, reg);
    if (!registeredByTensor.has(spec.tensorKey)) {
      registeredByTensor.set(spec.tensorKey, reg);
    }
    return reg;
  };

  // ==========================================================================
  // STAGE 1: Input X & Pre-Attn RMSNorm U -> W_Q / W_K / W_V (Reversal Caliper)
  // ==========================================================================
  lines.addDashedFrameXZ(-17.4, -5.0, -6.6, 5.0, -0.05, LINE_STYLE_GUIDE, 0.34);

  addSheet({
    partId: 'input_x',
    tensorKey: 'input_x',
    label: `X [${S}×8]`,
    sublabel: `Layer ${layerIndex + 1} input`,
    cx: -16.0,
    cz: 0.0,
    ex: 1.7,
    ez: 1.45,
    gridCols: config.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_RESIDUAL,
    highlight: activeStep === 0 ? SHEET_HL_OPERAND_A : SHEET_HL_NORMAL,
    slice: atlasResult.slices['input_x']!,
    badge: `Hidden [${S}×8]`,
    summary: `Residual stream entering Layer ${layerIndex + 1} before pre-attention RMSNorm.`,
    specs: [`Shape: [${S}, 8]`, `Positions: [${Array.from(positionIds).join(', ')}]`],
  });

  addSheet({
    partId: 'rmsnorm_u',
    tensorKey: 'rmsnorm_u',
    label: `RMSNorm U [${S}×8]`,
    sublabel: 'ε = 1e-5',
    cx: -13.4,
    cz: 0.0,
    ex: 1.7,
    ez: 1.45,
    gridCols: config.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_NORM,
    highlight: activeStep === 0 ? SHEET_HL_ACTIVE : SHEET_HL_NORMAL,
    slice: atlasResult.slices['rmsnorm_u']!,
    badge: 'Pre-Attn RMSNorm',
    summary: 'Ministral-3 RMSNorm: (x · rsqrt(mean(x²) + 1e-5)) · γ_in.',
    specs: [`Shape: [${S}, 8]`, `eps = ${config.rmsNormEps}`],
  });

  addSheet({
    partId: 'w_q',
    tensorKey: 'w_q',
    label: 'W_Q [8×256]',
    sublabel: '32 Q heads × dHead=8',
    cx: -9.4,
    cz: -3.2,
    ex: 4.4,
    ez: 1.15,
    gridCols: config.qProjOutDim,
    gridRows: config.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: activeStep === 0 ? SHEET_HL_OPERAND_B : SHEET_HL_NORMAL,
    slice: atlasResult.slices['w_q']!,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    badge: 'Mini 8→256 · Prod 5120→4096',
    summary: `Bias-free query projection weight. ${DIFFUSION_PROJECTION_REVERSAL_BADGE}.`,
    specs: [
      'Mini: [256, 8] (expands 8→256)',
      'Prod 14B: [4096, 5120] (contracts 5120→4096)',
      '32 Q heads × headDim=8 (mini) / 128 (prod)',
    ],
  });

  addSheet({
    partId: 'w_k',
    tensorKey: 'w_k',
    label: 'W_K [8×64]',
    sublabel: '8 KV heads × dHead=8',
    cx: -9.4,
    cz: 0.0,
    ex: 3.2,
    ez: 1.15,
    gridCols: config.kvProjOutDim,
    gridRows: config.hiddenSize,
    role: SHEET_ROLE_KV_K,
    highlight: activeStep === 0 ? SHEET_HL_OPERAND_B : SHEET_HL_NORMAL,
    slice: atlasResult.slices['w_k']!,
    groupIndex: selectedKvHead,
    badge: 'Mini 8→64 · Prod 5120→1024',
    summary: 'Bias-free key projection weight feeding 8 shared KV heads (4:1 GQA).',
    specs: ['Mini: [64, 8]', 'Prod 14B: [1024, 5120]', '8 KV heads × 4 Q/group'],
  });

  addSheet({
    partId: 'w_v',
    tensorKey: 'w_v',
    label: 'W_V [8×64]',
    sublabel: '8 KV heads × dHead=8',
    cx: -9.4,
    cz: 3.2,
    ex: 3.2,
    ez: 1.15,
    gridCols: config.kvProjOutDim,
    gridRows: config.hiddenSize,
    role: SHEET_ROLE_KV_V,
    highlight: activeStep === 0 ? SHEET_HL_OPERAND_B : SHEET_HL_NORMAL,
    slice: atlasResult.slices['w_v']!,
    groupIndex: selectedKvHead,
    badge: 'Mini 8→64 · Prod 5120→1024',
    summary: 'Bias-free value projection weight (no RoPE, no Llama-4 scale).',
    specs: ['Mini: [64, 8]', 'Prod 14B: [1024, 5120]'],
  });

  calipers.push({
    id: 'cal-projection-reversal',
    aWorld: [-13.4, 0.35, -4.55],
    bWorld: [-7.2, 0.35, -4.55],
    label: DIFFUSION_PROJECTION_REVERSAL_BADGE,
    partIds: ['rmsnorm_u', 'w_q', 'w_k', 'w_v'],
  });

  // Stage 1 flow ribbons
  flow.addPolylineXZ(
    [
      { x: -15.15, z: 0 },
      { x: -14.25, z: 0 },
    ],
    RIBBON_Y,
    0.16,
    activeStep === 0 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );
  flow.addRoundedOrthoXZ(
    [
      { x: -12.55, z: 0 },
      { x: -12.0, z: 0 },
      { x: -12.0, z: -3.2 },
      { x: -11.6, z: -3.2 },
    ],
    0.28,
    RIBBON_Y,
    0.14,
    activeStep === 0 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );
  flow.addPolylineXZ(
    [
      { x: -12.55, z: 0 },
      { x: -11.0, z: 0 },
    ],
    RIBBON_Y,
    0.14,
    activeStep === 0 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );
  flow.addRoundedOrthoXZ(
    [
      { x: -12.55, z: 0 },
      { x: -12.0, z: 0 },
      { x: -12.0, z: 3.2 },
      { x: -11.0, z: 3.2 },
    ],
    0.28,
    RIBBON_Y,
    0.14,
    activeStep === 0 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );

  // ==========================================================================
  // STAGE 2: Half-Split YaRN RoPE (`Q_pre/K_pre -> Q_rot/K_rot`, 4 dyadic planes)
  // ==========================================================================
  lines.addDashedFrameXZ(-6.2, -5.0, 6.2, 2.5, -0.05, LINE_STYLE_BF16_GOLD, 0.38);

  const qPreSheet = addSheet({
    partId: 'q_pre',
    tensorKey: 'q_pre',
    label: `Q_pre [${S}×256]`,
    sublabel: 'Pre-RoPE 32 Q heads',
    cx: -3.6,
    cz: -3.2,
    ex: 4.4,
    ez: 1.55,
    gridCols: config.qProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: activeStep === 1 ? SHEET_HL_OPERAND_A : SHEET_HL_NORMAL,
    slice: atlasResult.slices['q_pre']!,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    tapIndex: selectedRotaryPlane,
    badge: `Pre-RoPE Q [${S}×256]`,
    summary: 'Unrotated query projections before half-split YaRN RoPE.',
    specs: [
      `Active Q head: h${selectedQHead} (cols ${selectedQHead * 8}..${selectedQHead * 8 + 7})`,
      `Active rotary plane: (${selectedRotaryPlane}, ${selectedRotaryPlane + 4})`,
    ],
  });

  const kPreSheet = addSheet({
    partId: 'k_pre',
    tensorKey: 'k_pre',
    label: `K_pre [${S}×64]`,
    sublabel: 'Pre-RoPE 8 KV heads',
    cx: -3.6,
    cz: 0.0,
    ex: 3.2,
    ez: 1.55,
    gridCols: config.kvProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_KV_K,
    highlight: activeStep === 1 ? SHEET_HL_OPERAND_A : SHEET_HL_NORMAL,
    slice: atlasResult.slices['k_pre']!,
    groupIndex: selectedKvHead,
    tapIndex: selectedRotaryPlane,
    badge: `Pre-RoPE K [${S}×64]`,
    summary: 'Unrotated key projections before half-split YaRN RoPE.',
    specs: [
      `Active KV head: kv${selectedKvHead} (cols ${selectedKvHead * 8}..${selectedKvHead * 8 + 7})`,
    ],
  });

  const ropeCosSheet = addSheet({
    partId: 'rope_cos',
    tensorKey: 'rope_cos',
    label: 'cos(p·ω) [32×4]',
    sublabel: '4 dyadic planes',
    cx: -0.5,
    cz: -1.6,
    ex: 1.05,
    ez: 1.75,
    gridCols: config.headDim / 2,
    gridRows: DIFFUSION_GQA_MAX_KEY,
    role: SHEET_ROLE_NORM,
    highlight: activeStep === 1 ? SHEET_HL_OPERAND_B : SHEET_HL_NORMAL,
    slice: atlasResult.slices['rope_cos']!,
    tapIndex: selectedRotaryPlane,
    badge: 'YaRN cos [32×4]',
    summary: 'Half-split YaRN cosine table across positions 0..31 and 4 dyadic rotary planes.',
    specs: [
      `Plane 0 (original): ω=1.0`,
      `Plane 1 (blended): ω=0.3125`,
      `Plane 2 (interpolated): ω=0.0625`,
      `Plane 3 (interpolated): ω=0.03125`,
    ],
    landmarkTier: 'detail',
  });

  addSheet({
    partId: 'rope_sin',
    tensorKey: 'rope_sin',
    label: 'sin(p·ω) [32×4]',
    sublabel: 'rotate_half',
    cx: 0.75,
    cz: -1.6,
    ex: 1.05,
    ez: 1.75,
    gridCols: config.headDim / 2,
    gridRows: DIFFUSION_GQA_MAX_KEY,
    role: SHEET_ROLE_NORM,
    highlight: activeStep === 1 ? SHEET_HL_OPERAND_B : SHEET_HL_NORMAL,
    slice: atlasResult.slices['rope_sin']!,
    tapIndex: selectedRotaryPlane,
    badge: 'YaRN sin [32×4]',
    summary: 'Half-split YaRN sine table paired with rotate_half([-x2, x1]).',
    specs: [
      `Active plane ${selectedRotaryPlane}: zone=${ropePhasor.yarnZone}, ω=${ropePhasor.invFreq}`,
      `Angle φ = p·ω = ${ropePhasor.angleRad.toFixed(4)} rad`,
    ],
    landmarkTier: 'detail',
  });

  const invFreqSheet = addSheet({
    partId: 'yarn_inv_freq',
    tensorKey: 'yarn_inv_freq',
    label: 'YaRN ω [4×1]',
    sublabel: '1 · 5/16 · 1/16 · 1/32',
    cx: 0.12,
    cz: 1.35,
    ex: 0.68,
    ez: 1.35,
    gridCols: 1,
    gridRows: config.headDim / 2,
    role: SHEET_ROLE_NORM,
    highlight: activeStep === 1 ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
    slice: atlasResult.slices['yarn_inv_freq']!,
    tapIndex: selectedRotaryPlane,
    badge: 'Dyadic YaRN Zones',
    summary:
      'Exact dyadic inverse frequencies [1.0, 0.3125, 0.0625, 0.03125] covering original (0), blended (1), and interpolated (2,3) zones.',
    specs: [
      'Plane 0 (0,4): 1.00000 · original',
      'Plane 1 (1,5): 0.31250 · blended',
      'Plane 2 (2,6): 0.06250 · interpolated',
      'Plane 3 (3,7): 0.03125 · interpolated',
    ],
    landmarkTone: 'gold',
  });

  const qRotSheet = addSheet({
    partId: 'q_rot',
    tensorKey: 'q_rot',
    label: `Q_rot [${S}×256]`,
    sublabel: 'Post-YaRN RoPE Q',
    cx: 3.6,
    cz: -3.2,
    ex: 4.4,
    ez: 1.55,
    gridCols: config.qProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: activeStep === 1 ? SHEET_HL_ACTIVE : activeStep === 2 ? SHEET_HL_OPERAND_A : SHEET_HL_NORMAL,
    slice: atlasResult.slices['q_rot']!,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    tapIndex: selectedRotaryPlane,
    badge: `Post-RoPE Q [${S}×256]`,
    summary: 'Rotated queries after applying half-split YaRN RoPE across all 4 rotary planes.',
    specs: [
      `q_rot[p=${ropePhasor.queryPosition}, h=${selectedQHead}, plane=${selectedRotaryPlane}] = (${fmtSigned(ropePhasor.qRotPair[0])}, ${fmtSigned(ropePhasor.qRotPair[1])})`,
    ],
  });

  const kRotSheet = addSheet({
    partId: 'k_rot',
    tensorKey: 'k_rot',
    label: `K_rot [${S}×64]`,
    sublabel: 'Post-RoPE K (unscaled!)',
    cx: 3.6,
    cz: 0.0,
    ex: 3.2,
    ez: 1.55,
    gridCols: config.kvProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_KV_K,
    highlight: activeStep === 1 ? SHEET_HL_ACTIVE : SHEET_HL_NORMAL,
    slice: atlasResult.slices['k_rot']!,
    groupIndex: selectedKvHead,
    tapIndex: selectedRotaryPlane,
    badge: `Post-RoPE K [${S}×64]`,
    summary: 'Rotated keys written into K_full. Keys NEVER receive Llama-4 temperature scaling.',
    specs: [
      `k_rot[p=${ropePhasor.keyPosition}, kv=${selectedKvHead}, plane=${selectedRotaryPlane}] = (${fmtSigned(ropePhasor.kRotPair[0])}, ${fmtSigned(ropePhasor.kRotPair[1])})`,
    ],
  });

  // Highlight active rotary plane on yarn_inv_freq, rope_cos, q_pre, q_rot, k_pre, k_rot
  flow.addCellHighlightOnFrame(
    invFreqSheet.frame,
    selectedRotaryPlane,
    selectedRotaryPlane,
    0,
    0,
    4,
    1,
    0.015,
    CELL_TARGET_FILL,
    CELL_TARGET_FRAME,
    0.028,
  );
  flow.addCellHighlightOnFrame(
    ropeCosSheet.frame,
    clampInt(ropePhasor.queryPosition, 0, 31),
    clampInt(ropePhasor.queryPosition, 0, 31),
    selectedRotaryPlane,
    selectedRotaryPlane,
    32,
    4,
    0.015,
    CELL_TARGET_FILL,
    CELL_TARGET_FRAME,
    0.024,
  );
  // Highlight selected Q head span in q_pre and q_rot
  flow.addCellHighlightOnFrame(
    qPreSheet.frame,
    focusRow,
    focusRow,
    selectedQHead * 8,
    selectedQHead * 8 + 7,
    S,
    config.qProjOutDim,
    0.015,
    CELL_TARGET_FILL,
    CELL_TARGET_FRAME,
    0.022,
  );
  flow.addCellHighlightOnFrame(
    qRotSheet.frame,
    focusRow,
    focusRow,
    selectedQHead * 8,
    selectedQHead * 8 + 7,
    S,
    config.qProjOutDim,
    0.015,
    CELL_TARGET_FILL,
    CELL_TARGET_FRAME,
    0.022,
  );
  flow.addCellHighlightOnFrame(
    kPreSheet.frame,
    focusRow,
    focusRow,
    selectedKvHead * 8,
    selectedKvHead * 8 + 7,
    S,
    config.kvProjOutDim,
    0.015,
    CELL_TARGET_FILL,
    CELL_TARGET_FRAME,
    0.022,
  );
  flow.addCellHighlightOnFrame(
    kRotSheet.frame,
    focusRow,
    focusRow,
    selectedKvHead * 8,
    selectedKvHead * 8 + 7,
    S,
    config.kvProjOutDim,
    0.015,
    CELL_TARGET_FILL,
    CELL_TARGET_FRAME,
    0.022,
  );

  // Stage 1 -> Stage 2 ribbons
  flow.addPolylineXZ(
    [
      { x: -7.2, z: -3.2 },
      { x: -5.8, z: -3.2 },
    ],
    RIBBON_Y,
    0.15,
    activeStep <= 1 ? RIBBON_PATHWAY_SOFT : RIBBON_QUIET,
  );
  flow.addPolylineXZ(
    [
      { x: -7.8, z: 0.0 },
      { x: -5.2, z: 0.0 },
    ],
    RIBBON_Y,
    0.15,
    activeStep <= 1 ? RIBBON_PATHWAY_SOFT : RIBBON_QUIET,
  );
  // Q_pre -> Q_rot and K_pre -> K_rot ribbons
  flow.addPolylineXZ(
    [
      { x: -1.4, z: -3.2 },
      { x: 1.4, z: -3.2 },
    ],
    RIBBON_Y,
    0.16,
    activeStep === 1 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );
  flow.addPolylineXZ(
    [
      { x: -2.0, z: 0.0 },
      { x: 2.0, z: 0.0 },
    ],
    RIBBON_Y,
    0.16,
    activeStep === 1 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );

  // ==========================================================================
  // STAGE 3: Live Llama-4 Query Scale (Queries ONLY — Never Keys!)
  // ==========================================================================
  const llama4Active = ropePhasor.llama4Scale > 1.0001;

  const llama4Sheet = addSheet({
    partId: 'llama4_scale',
    tensorKey: 'llama4_scale',
    label: `Llama-4 s(p) [32×1]`,
    sublabel: llama4Active
      ? `ACTIVE ×${ropePhasor.llama4Scale.toFixed(4)} (p=${ropePhasor.queryPosition} ≥ 16)`
      : `×1.0000 (p=${ropePhasor.queryPosition} < 16 · queries only)`,
    cx: 7.4,
    cz: -5.25,
    ex: 0.72,
    ez: 2.0,
    gridCols: 1,
    gridRows: DIFFUSION_GQA_MAX_KEY,
    role: SHEET_ROLE_NORM,
    highlight:
      activeStep === 2 || llama4Active ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
    slice: atlasResult.slices['llama4_scale']!,
    badge: llama4Active
      ? `×${ropePhasor.llama4Scale.toFixed(4)} ACTIVE`
      : '×1.000 (p<16)',
    summary:
      'Llama-4 post-RoPE query temperature multiplier 1 + 0.1·ln(1 + ⌊cache_pos / 16⌋). Applied strictly to queries (Q_rot -> Q_scaled); keys remain completely unscaled.',
    specs: [
      `cache_pos 0..15 -> ×1.0000`,
      `cache_pos 16..31 -> ×1.0693`,
      `Active query row ${focusRow} (pos ${ropePhasor.queryPosition}): ×${ropePhasor.llama4Scale.toFixed(6)}`,
    ],
    landmarkTone: llama4Active ? 'gold' : 'default',
  });

  flow.addCellHighlightOnFrame(
    llama4Sheet.frame,
    clampInt(ropePhasor.queryPosition, 0, 31),
    clampInt(ropePhasor.queryPosition, 0, 31),
    0,
    0,
    32,
    1,
    0.015,
    CELL_TARGET_FILL,
    CELL_TARGET_FRAME,
    0.026,
  );

  addSheet({
    partId: 'q_scaled',
    tensorKey: 'q_scaled',
    label: `Q_scaled [${S}×256]`,
    sublabel: `Queries ×${ropePhasor.llama4Scale.toFixed(3)}`,
    cx: 9.8,
    cz: -3.2,
    ex: 4.4,
    ez: 1.55,
    gridCols: config.qProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight:
      activeStep === 2
        ? SHEET_HL_ACTIVE
        : llama4Active
          ? SHEET_HL_BF16_GOLD
          : activeStep === 3
            ? SHEET_HL_OPERAND_A
            : SHEET_HL_NORMAL,
    slice: atlasResult.slices['q_scaled']!,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    tapIndex: selectedRotaryPlane,
    badge: `Scaled Q [${S}×256]`,
    summary: 'Post-RoPE queries scaled by Llama-4 query temperature multiplier.',
    specs: [
      `q_scaled[h=${selectedQHead}, plane=${selectedRotaryPlane}] = (${fmtSigned(ropePhasor.qScaledPair[0])}, ${fmtSigned(ropePhasor.qScaledPair[1])})`,
    ],
  });

  // Q_rot -> Q_scaled + Llama-4 scale injection arrow
  flow.addPolylineXZ(
    [
      { x: 5.8, z: -3.2 },
      { x: 7.6, z: -3.2 },
    ],
    RIBBON_Y,
    0.16,
    activeStep === 2 || llama4Active ? RIBBON_LLAMA4_GOLD : RIBBON_QUIET,
  );
  flow.addRoundedOrthoXZ(
    [
      { x: 7.4, z: -4.25 },
      { x: 7.4, z: -3.65 },
      { x: 8.6, z: -3.65 },
    ],
    0.22,
    RIBBON_Y,
    0.13,
    activeStep === 2 || llama4Active ? RIBBON_LLAMA4_GOLD : RIBBON_QUIET,
  );

  // ==========================================================================
  // STAGE 4: 32Q/8KV (4:1) Attention — K_full, V_full & 8 KV-Group Scores/Softmax
  // ==========================================================================
  lines.addDashedFrameXZ(12.4, -6.6, 26.8, 6.8, -0.05, LINE_STYLE_GUIDE, 0.34);

  addSheet({
    partId: 'k_full',
    tensorKey: 'k_full',
    label: `K_full [${K}×64]`,
    sublabel: `Past ${attn.pastLen} + New ${S} · 8 KV`,
    cx: 14.3,
    cz: 1.4,
    ex: 3.1,
    ez: 2.4,
    gridCols: config.kvProjOutDim,
    gridRows: K,
    role: SHEET_ROLE_KV_K,
    highlight: activeStep === 3 ? SHEET_HL_OPERAND_B : SHEET_HL_NORMAL,
    slice: atlasResult.slices['k_full']!,
    groupIndex: selectedKvHead,
    badge: `KV Bank K [${K}×64]`,
    summary: `Concatenated keys (${attn.pastLen} cached + ${S} current) across 8 KV heads (unscaled by Llama-4).`,
    specs: [
      `Shape: [${K}, 64] (8 KV heads × 8 headDim)`,
      `useCache: ${attn.useCache ? 'true (committed)' : 'false (transient concat)'}`,
    ],
  });

  addSheet({
    partId: 'v_full',
    tensorKey: 'v_full',
    label: `V_full [${K}×64]`,
    sublabel: `Past ${attn.pastLen} + New ${S} · 8 KV`,
    cx: 14.3,
    cz: 4.9,
    ex: 3.1,
    ez: 2.4,
    gridCols: config.kvProjOutDim,
    gridRows: K,
    role: SHEET_ROLE_KV_V,
    highlight: activeStep === 4 ? SHEET_HL_OPERAND_B : SHEET_HL_NORMAL,
    slice: atlasResult.slices['v_full']!,
    groupIndex: selectedKvHead,
    badge: `KV Bank V [${K}×64]`,
    summary: 'Concatenated values across 8 KV heads weighted by Softmax probabilities.',
    specs: [`Shape: [${K}, 64]`, `4:1 GQA broadcast across 32 Q heads`],
  });

  // Bypass ribbon K_rot -> K_full (keys never scaled) & W_V -> V_full
  flow.addRoundedOrthoXZ(
    [
      { x: 5.2, z: 0.0 },
      { x: 11.8, z: 0.0 },
      { x: 11.8, z: 1.4 },
      { x: 12.75, z: 1.4 },
    ],
    0.35,
    RIBBON_Y,
    0.15,
    RIBBON_SSM_LOOP,
  );
  flow.addRoundedOrthoXZ(
    [
      { x: -7.8, z: 3.2 },
      { x: 11.6, z: 3.2 },
      { x: 11.6, z: 4.9 },
      { x: 12.75, z: 4.9 },
    ],
    0.45,
    RIBBON_Y,
    0.13,
    RIBBON_QUIET,
  );

  // 8 KV-Group Scores & Softmax plates (D24 live RG32F bound, PickBox.groupIndex = 0..7)
  const scoreGridXs = [18.1, 20.3];
  const softmaxGridXs = [23.0, 25.2];
  const groupZs = [-5.0, -2.5, 0.0, 2.5];

  let activeScoreSheet: RegisteredSheet | null = null;
  let activeSoftmaxSheet: RegisteredSheet | null = null;

  for (let g = 0; g < config.numKeyValueHeads; g++) {
    const colSlot = g % 2;
    const rowSlot = Math.floor(g / 2);
    const qHead = atlasResult.selectedQHeadsPerGroup[g]!;
    const isSelGroup = g === selectedKvHead;

    const scoreSheet = addSheet({
      partId: `scores_g${g}`,
      tensorKey: 'scores',
      label: `Scores KV${g} (Q${qHead})`,
      sublabel: isSelGroup ? `[${S}×${K}] · ${attn.maskMode}` : undefined,
      cx: scoreGridXs[colSlot]!,
      cz: groupZs[rowSlot]!,
      ex: 1.78,
      ez: 1.65,
      gridCols: K,
      gridRows: S,
      sheetKind: SHEET_KIND_CAUSAL_SCORES,
      role: SHEET_ROLE_SCORES,
      highlight: isSelGroup
        ? activeStep === 3
          ? SHEET_HL_ACTIVE
          : SHEET_HL_PATHWAY
        : activeStep === 3
          ? SHEET_HL_NORMAL
          : SHEET_HL_DIMMED,
      slice: atlasResult.scoreSlices[g]!,
      headIndex: qHead,
      groupIndex: g,
      semanticTag: 'gqa_scores',
      badge: `KV${g} · Q${qHead} [${S}×${K}]`,
      summary: `Scaled dot-product scores ⟨Q_scaled[h${qHead}], K_rot[kv${g}]⟩ / √8 with explicit D24 RG32F mask channel G ∈ {0, 1}.`,
      specs: [
        `KV group ${g} serves Q heads ${g * 4}..${g * 4 + 3}`,
        `Displayed Q head: h${qHead}`,
        `Mask mode: ${attn.maskMode}`,
      ],
      landmarkTier: isSelGroup ? 'primary' : 'detail',
      landmarkTone: isSelGroup ? 'gold' : 'muted',
    });

    const softmaxSheet = addSheet({
      partId: `softmax_g${g}`,
      tensorKey: 'softmax',
      label: `Softmax KV${g} (Q${qHead})`,
      sublabel: isSelGroup ? `[${S}×${K}] probs` : undefined,
      cx: softmaxGridXs[colSlot]!,
      cz: groupZs[rowSlot]!,
      ex: 1.78,
      ez: 1.65,
      gridCols: K,
      gridRows: S,
      sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
      role: SHEET_ROLE_SOFTMAX,
      highlight: isSelGroup
        ? activeStep === 4
          ? SHEET_HL_ACTIVE
          : SHEET_HL_PATHWAY
        : activeStep === 4
          ? SHEET_HL_NORMAL
          : SHEET_HL_DIMMED,
      slice: atlasResult.softmaxSlices[g]!,
      headIndex: qHead,
      groupIndex: g,
      semanticTag: 'gqa_softmax',
      badge: `KV${g} · Q${qHead} Probs`,
      summary: `Row-wise softmax attention weights for KV group ${g} (Q head ${qHead}); masked entries are exact 0.0 with RG32F G=1.0.`,
      specs: [
        `KV group ${g} -> Q head ${qHead}`,
        `Shape: [${S}, ${K}]`,
      ],
      landmarkTier: isSelGroup ? 'primary' : 'detail',
      landmarkTone: isSelGroup ? 'cyan' : 'muted',
    });

    if (isSelGroup) {
      activeScoreSheet = scoreSheet;
      activeSoftmaxSheet = softmaxSheet;
    }
  }

  // Q_scaled -> Scores & Scores -> Softmax -> attn_concat ribbons
  flow.addRoundedOrthoXZ(
    [
      { x: 12.0, z: -3.2 },
      { x: 16.5, z: -3.2 },
      { x: 16.5, z: groupZs[Math.floor(selectedKvHead / 2)]! },
      { x: 17.2, z: groupZs[Math.floor(selectedKvHead / 2)]! },
    ],
    0.25,
    RIBBON_Y,
    0.16,
    activeStep === 3 ? RIBBON_ACTIVE : RIBBON_PATHWAY_SOFT,
  );
  flow.addPolylineXZ(
    [
      { x: 21.25, z: groupZs[Math.floor(selectedKvHead / 2)]! },
      { x: 22.1, z: groupZs[Math.floor(selectedKvHead / 2)]! },
    ],
    RIBBON_Y,
    0.16,
    activeStep === 3 || activeStep === 4 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );

  // ==========================================================================
  // STAGE 5: attn_concat -> W_O + Optional Draft-Only LoRA Delta -> attn_out + Residual
  // ==========================================================================
  lines.addDashedFrameXZ(27.2, -4.6, 46.2, 4.4, -0.05, LINE_STYLE_GUIDE, 0.34);

  addSheet({
    partId: 'attn_concat',
    tensorKey: 'attn_concat',
    label: `Attn Concat [${S}×256]`,
    sublabel: '32 heads concatenated',
    cx: 29.8,
    cz: -1.4,
    ex: 4.3,
    ez: 1.55,
    gridCols: config.qProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: activeStep === 4 ? SHEET_HL_ACTIVE : activeStep === 5 ? SHEET_HL_OPERAND_A : SHEET_HL_NORMAL,
    slice: atlasResult.slices['attn_concat']!,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    badge: `Concat [${S}×256]`,
    summary: 'Concatenated 32-head attention output feeding both W_O and the optional draft-only o_proj LoRA branch.',
    specs: [`Shape: [${S}, 256] (32 heads × 8 headDim)`],
  });

  flow.addRoundedOrthoXZ(
    [
      { x: 26.1, z: groupZs[Math.floor(selectedKvHead / 2)]! },
      { x: 26.9, z: groupZs[Math.floor(selectedKvHead / 2)]! },
      { x: 26.9, z: -1.4 },
      { x: 27.65, z: -1.4 },
    ],
    0.25,
    RIBBON_Y,
    0.16,
    activeStep === 4 ? RIBBON_ACTIVE : RIBBON_PATHWAY_SOFT,
  );

  // Base o_proj path
  addSheet({
    partId: 'w_o',
    tensorKey: 'w_o',
    label: 'W_O [8×256]',
    sublabel: 'Mini 256→8 · Prod 4096→5120',
    cx: 35.0,
    cz: -1.4,
    ex: 4.2,
    ez: 1.15,
    gridCols: config.qProjOutDim,
    gridRows: config.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: activeStep === 5 ? SHEET_HL_OPERAND_B : SHEET_HL_NORMAL,
    slice: atlasResult.slices['w_o']!,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    badge: 'Output Proj W_O',
    summary: 'Backbone output projection mapping 32 concatenated heads back to hidden size d=8.',
    specs: ['Mini: [8, 256]', 'Prod 14B: [5120, 4096]'],
  });

  addSheet({
    partId: 'o_proj_base',
    tensorKey: 'o_proj_base',
    label: `W_O·a [${S}×8]`,
    sublabel: 'Base o_proj output',
    cx: 39.2,
    cz: -1.4,
    ex: 1.85,
    ez: 1.45,
    gridCols: config.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: activeStep === 5 ? SHEET_HL_PATHWAY : SHEET_HL_NORMAL,
    slice: atlasResult.slices['o_proj_base']!,
    badge: `Base o_proj [${S}×8]`,
    summary: 'Pure backbone attention output W_O·a before optional draft-only LoRA delta.',
    specs: [`Shape: [${S}, 8]`, 'Always identical on AR / DLM / Verify passes (0 ULP)'],
  });

  // Parallel draft-only o_proj LoRA branch (+4.0·B(A·a))
  const loraHl = loraActive
    ? activeStep === 5
      ? SHEET_HL_ACTIVE
      : SHEET_HL_BF16_GOLD
    : SHEET_HL_DIMMED;

  lines.addDashedFrameXZ(
    31.0,
    1.15,
    41.4,
    3.65,
    -0.04,
    loraActive ? LINE_STYLE_BF16_GOLD : LINE_STYLE_GUIDE,
    loraActive ? 0.78 : 0.28,
  );

  addSheet({
    partId: 'lora_a',
    tensorKey: 'lora_a',
    label: 'LoRA A [4×256]',
    sublabel: loraActive ? 'ACTIVE (draft)' : 'BYPASSED (OFF)',
    cx: 33.2,
    cz: 2.4,
    ex: 3.6,
    ez: 0.88,
    gridCols: config.qProjOutDim,
    gridRows: config.loraRank,
    role: SHEET_ROLE_WEIGHT,
    highlight: loraHl,
    slice: atlasResult.slices['lora_a']!,
    badge: loraActive ? 'LoRA A · ACTIVE' : 'LoRA A · OFF',
    summary: 'Low-rank down-projection A [r=4, 256] for draft-only o_proj LoRA.',
    specs: [`Rank r=${config.loraRank}, α=${config.loraAlpha}, scale=${config.loraScale.toFixed(1)}`],
    landmarkTone: loraActive ? 'gold' : 'muted',
    landmarkTier: loraActive ? 'primary' : 'detail',
  });

  addSheet({
    partId: 'lora_hidden',
    tensorKey: 'lora_hidden',
    label: `A·a [${S}×4]`,
    cx: 36.1,
    cz: 2.4,
    ex: 1.1,
    ez: 1.4,
    gridCols: config.loraRank,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: loraHl,
    slice: atlasResult.slices['lora_hidden']!,
    badge: `LoRA Hidden [${S}×4]`,
    summary: 'Rank-4 intermediate activation A·a.',
    specs: [`Shape: [${S}, 4]`],
    landmarkTone: loraActive ? 'gold' : 'muted',
    landmarkTier: 'detail',
  });

  addSheet({
    partId: 'lora_b',
    tensorKey: 'lora_b',
    label: 'LoRA B [8×4]',
    cx: 37.8,
    cz: 2.4,
    ex: 1.1,
    ez: 1.15,
    gridCols: config.loraRank,
    gridRows: config.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: loraHl,
    slice: atlasResult.slices['lora_b']!,
    badge: 'LoRA B [8×4]',
    summary: 'Low-rank up-projection B [8, r=4] scaled by α/r = 4.0.',
    specs: [`Shape: [8, 4]`, `Scale: α/r = 16/4 = 4.0`],
    landmarkTone: loraActive ? 'gold' : 'muted',
    landmarkTier: 'detail',
  });

  addSheet({
    partId: 'lora_delta',
    tensorKey: 'lora_delta',
    label: `ΔLoRA +4·B(A·a) [${S}×8]`,
    sublabel: loraActive ? 'Draft-only delta ACTIVE' : 'Bypassed (0.0)',
    cx: 40.0,
    cz: 2.4,
    ex: 1.85,
    ez: 1.45,
    gridCols: config.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: loraHl,
    slice: atlasResult.slices['lora_delta']!,
    badge: loraActive ? 'LoRA Δ · ACTIVE' : 'LoRA Δ · BYPASSED',
    summary:
      'Scaled LoRA delta +4.0·B(A·a) added to W_O·a strictly during linear_spec_lora draft passes.',
    specs: [
      `Status: ${loraActive ? 'ACTIVE on draft (verify remains OFF)' : 'BYPASSED (LoRA OFF)'}`,
      DIFFUSION_LORA_RAIL_BADGE,
    ],
    landmarkTone: loraActive ? 'gold' : 'muted',
  });

  addSheet({
    partId: 'attn_out',
    tensorKey: 'attn_out',
    label: `Attn Out [${S}×8]`,
    sublabel: attn.loraEnabled ? 'W_O·a + ΔLoRA' : 'W_O·a (verify/base)',
    cx: 42.4,
    cz: -0.4,
    ex: 1.85,
    ez: 1.45,
    gridCols: config.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: activeStep === 5 ? SHEET_HL_ACTIVE : SHEET_HL_NORMAL,
    slice: atlasResult.slices['attn_out']!,
    badge: `Attn Out [${S}×8]`,
    summary: 'Final attention sublayer output entering the post-attention residual add.',
    specs: [
      `Shape: [${S}, 8]`,
      `Pass LoRA applied: ${attn.loraEnabled ? 'YES (draft)' : 'NO (base/verify)'}`,
    ],
  });

  addSheet({
    partId: 'post_attn_resid',
    tensorKey: 'post_attn_resid',
    label: `Post-Attn Resid [${S}×8]`,
    sublabel: 'X + Attn Out',
    cx: 45.0,
    cz: -0.4,
    ex: 1.85,
    ez: 1.45,
    gridCols: config.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_RESIDUAL,
    highlight: activeStep === 5 ? SHEET_HL_PATHWAY : SHEET_HL_NORMAL,
    slice: atlasResult.slices['post_attn_resid']!,
    badge: `Residual [${S}×8]`,
    summary: 'Residual sum h\' = X + Attn(RMSNorm(X)) feeding the dense SwiGLU MLP.',
    specs: [`Shape: [${S}, 8]`],
  });

  // Stage 5 flow ribbons
  flow.addPolylineXZ(
    [
      { x: 31.95, z: -1.4 },
      { x: 32.9, z: -1.4 },
    ],
    RIBBON_Y,
    0.16,
    activeStep === 5 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );
  flow.addPolylineXZ(
    [
      { x: 37.1, z: -1.4 },
      { x: 38.25, z: -1.4 },
    ],
    RIBBON_Y,
    0.16,
    activeStep === 5 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );
  flow.addRoundedOrthoXZ(
    [
      { x: 40.15, z: -1.4 },
      { x: 40.8, z: -1.4 },
      { x: 40.8, z: -0.4 },
      { x: 41.45, z: -0.4 },
    ],
    0.22,
    RIBBON_Y,
    0.16,
    activeStep === 5 ? RIBBON_ACTIVE : RIBBON_QUIET,
  );
  flow.addPolylineXZ(
    [
      { x: 43.35, z: -0.4 },
      { x: 44.05, z: -0.4 },
    ],
    RIBBON_Y,
    0.16,
    activeStep === 5 ? RIBBON_RESIDUAL_ACTIVE : RIBBON_RESIDUAL,
  );

  // LoRA branch ribbons (highlighted active when loraActive, dimmed when bypassed)
  const loraRibbonCol = loraActive ? RIBBON_LORA_ACTIVE : RIBBON_DIMMED;
  flow.addRoundedOrthoXZ(
    [
      { x: 29.8, z: -0.62 },
      { x: 29.8, z: 2.4 },
      { x: 31.4, z: 2.4 },
    ],
    0.30,
    RIBBON_Y,
    loraActive ? 0.16 : 0.10,
    loraRibbonCol,
  );
  flow.addPolylineXZ(
    [
      { x: 35.0, z: 2.4 },
      { x: 35.55, z: 2.4 },
    ],
    RIBBON_Y,
    loraActive ? 0.15 : 0.09,
    loraRibbonCol,
  );
  flow.addPolylineXZ(
    [
      { x: 36.65, z: 2.4 },
      { x: 37.25, z: 2.4 },
    ],
    RIBBON_Y,
    loraActive ? 0.15 : 0.09,
    loraRibbonCol,
  );
  flow.addPolylineXZ(
    [
      { x: 38.35, z: 2.4 },
      { x: 39.05, z: 2.4 },
    ],
    RIBBON_Y,
    loraActive ? 0.15 : 0.09,
    loraRibbonCol,
  );
  flow.addRoundedOrthoXZ(
    [
      { x: 40.95, z: 2.4 },
      { x: 42.4, z: 2.4 },
      { x: 42.4, z: 0.35 },
    ],
    0.30,
    RIBBON_Y,
    loraActive ? 0.16 : 0.09,
    loraRibbonCol,
  );

  // Residual skip ribbon curving across top from X -> Post-Attn Resid
  flow.addRoundedOrthoXZ(
    [
      { x: -16.0, z: -0.75 },
      { x: -16.0, z: -7.15 },
      { x: 45.0, z: -7.15 },
      { x: 45.0, z: -1.15 },
    ],
    0.65,
    RIBBON_Y,
    0.16,
    activeStep === 5 ? RIBBON_RESIDUAL_ACTIVE : RIBBON_RESIDUAL,
  );

  // ==========================================================================
  // FORMULA INSPECTOR, CELL POPOVER & ACTIVE CELL GRIDS
  // ==========================================================================
  const activeFormula = buildGqaRopeFormulaReadout({
    trace,
    config,
    layerIndex,
    selectedQHead,
    selectedKvHead,
    selectedRotaryPlane,
    focusRow,
    focusKeyCol,
    activeStep,
    activeCell,
    ropePhasor,
    loraActive,
    layerWeights,
  });

  const cellPopover = buildCellPopover(
    activeCell,
    activeFormula,
    registeredByPart,
    registeredByTensor,
  );

  const activeCellGrids = buildActiveCellGrids({
    trace,
    config,
    S,
    K,
    selectedQHead,
    selectedKvHead,
    selectedRotaryPlane,
    focusRow,
    focusKeyCol,
    activeScoreSheet,
    activeSoftmaxSheet,
    llama4Sheet,
    invFreqSheet,
    ropePhasor,
  });

  // ==========================================================================
  // PACK BUFFERS & FINALIZE SCENE
  // ==========================================================================
  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flow.pack();

  const provenanceBadges: DiffusionProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    { label: DIFFUSION_PROJECTION_REVERSAL_BADGE, tone: 'gold' },
    {
      label: DIFFUSION_LORA_RAIL_BADGE,
      tone: loraActive ? 'lora' : 'muted',
    },
  ];

  const fitBounds: AABB = {
    min: [-17.8, -0.6, -7.8],
    max: [46.6, 1.8, 7.2],
  };

  const scene: DiffusionSpatialScene = {
    viewMode: 'gqa_rope',
    title: `Layer ${String(layerIndex + 1).padStart(2, '0')} · 32Q/8KV RoPE + YaRN + LoRA GQA`,
    subtitle: `Q${selectedQHead}→KV${selectedKvHead} · plane (${selectedRotaryPlane},${selectedRotaryPlane + 4}) ${ropePhasor.yarnZone} · Llama-4 ×${ropePhasor.llama4Scale.toFixed(4)} · ${attn.maskMode}`,
    layerIndex,
    solidData: packedSolids.data,
    solidCount: packedSolids.count,
    lineData: packedLines.data,
    lineCount: packedLines.count,
    ribbonData: packedRibbons.data,
    ribbonCount: packedRibbons.count,
    sheetData: packedSheets.sheetData,
    sheetCount: packedSheets.sheetCount,
    sheetBasisData: packedSheets.sheetBasisData,
    sheetRevealData: packedSheets.sheetRevealData,
    flowVertices: packedFlow.vertices,
    flowIndices: packedFlow.indices,
    valueAtlas: atlasResult.atlas,
    atlasKey: atlasResult.atlasKey,
    atlasAudit: atlasResult.audit,
    fitBounds,
    defaultCameraAzimuthDeg: 0,
    defaultCameraElevationDeg: 52,
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze([]),
    calipers: Object.freeze(calipers),
    solidPickBoxes: Object.freeze(solidPickBoxes),
    fasciaPickBoxes: Object.freeze(fasciaPickBoxes),
    partCatalog: Object.freeze(partCatalog),
    activeCellGrids: Object.freeze(activeCellGrids),
    cellPopover,
    activeFormula,
    stepperSteps: GQA_ROPE_STEPPER_STEPS,
    provenanceBadges: Object.freeze(provenanceBadges),
    epistemicNote: DIFFUSION_EPISTEMIC_NOTE,
    warningBanner: ctx.warningBanner,
    ropePhasor,
    diagnostics: Object.freeze({
      layerIndex,
      selectedQHead,
      selectedKvHead,
      selectedRotaryPlane,
      queryTokens: S,
      keyTokens: K,
      maskMode: attn.maskMode,
      loraActive,
      maskedScoreCellCount: atlasResult.maskedScoreCellCount,
      maskedSoftmaxCellCount: atlasResult.maskedSoftmaxCellCount,
      atlasUtilizationPct: atlasResult.audit.utilizationPct,
      llama4Scale: ropePhasor.llama4Scale,
    }),
  };

  const finiteCheck = validateSceneGeometryFinite(scene);
  if (!finiteCheck.valid) {
    throw new Error(
      `Invalid gqa_rope scene geometry: ${finiteCheck.errors.join('; ')}`,
    );
  }

  return scene;
}

// ============================================================================
// FORMULA READOUT, POPOVER & ACTIVE CELL GRID HELPERS
// ============================================================================

function buildGqaRopeFormulaReadout(params: {
  readonly trace: DiffusionLayerTrace;
  readonly config: DiffusionMiniConfig;
  readonly layerIndex: number;
  readonly selectedQHead: number;
  readonly selectedKvHead: number;
  readonly selectedRotaryPlane: number;
  readonly focusRow: number;
  readonly focusKeyCol: number;
  readonly activeStep: number;
  readonly activeCell: DiffusionPickIdentity | null;
  readonly ropePhasor: DiffusionRopePhasorTelemetry;
  readonly loraActive: boolean;
  readonly layerWeights: { readonly lora: { readonly loraA: Float32Array; readonly loraB: Float32Array } };
}): LayerFormulaReadout {
  const {
    trace,
    config,
    selectedQHead,
    selectedKvHead,
    selectedRotaryPlane,
    focusRow,
    focusKeyCol,
    activeStep,
    activeCell,
    ropePhasor,
    loraActive,
    layerWeights,
  } = params;
  const attn = trace.attention;
  const tensor = activeCell?.tensorKey ?? null;

  // Scores / Softmax readout
  if (
    tensor === 'scores' ||
    tensor === 'softmax' ||
    (tensor === null && (activeStep === 3 || activeStep === 4))
  ) {
    const isProb = tensor === 'softmax' || (tensor === null && activeStep === 4);
    const qHead =
      activeCell?.qHeadIndex !== null && activeCell?.qHeadIndex !== undefined
        ? clampInt(activeCell.qHeadIndex, 0, config.numAttentionHeads - 1)
        : selectedQHead;
    const kvHead =
      activeCell?.kvHeadIndex !== null && activeCell?.kvHeadIndex !== undefined
        ? clampInt(activeCell.kvHeadIndex, 0, config.numKeyValueHeads - 1)
        : Math.floor(qHead / config.kvGroupSize);
    const qRow =
      activeCell?.row !== null && activeCell?.row !== undefined
        ? clampInt(activeCell.row, 0, attn.seqLen - 1)
        : focusRow;
    const kCol =
      activeCell?.col !== null && activeCell?.col !== undefined
        ? clampInt(activeCell.col, 0, attn.kvLen - 1)
        : focusKeyCol;

    const idx = qHead * attn.seqLen * attn.kvLen + qRow * attn.kvLen + kCol;
    const rawScore = attn.scores[idx]!;
    const isMasked = !Number.isFinite(rawScore);
    const val = isProb ? attn.probs[idx]! : rawScore;

    const eq = formatDiffusionAttentionCellFormula({
      stage: isProb ? 'prob' : 'score',
      qHead,
      kvHead,
      qRow,
      kCol,
      headDim: config.headDim,
      maskMode: attn.maskMode,
      isMasked,
      value: val,
    });

    return {
      tensorLabel: isProb ? 'Attention Softmax' : 'GQA Attention Score',
      coordLabel: `h${qHead}→kv${kvHead} [q=${qRow}, k=${kCol}]`,
      headBadge: `Q${qHead} · KV${kvHead} (4:1)`,
      value: isMasked ? (isProb ? 0 : -Infinity) : val,
      isMasked,
      valueText: isMasked
        ? isProb
          ? '= 0.000 (masked)'
          : '= −∞ (masked)'
        : `= ${fmtSigned(val, isProb ? 4 : 3)}`,
      equation: eq,
      subEquation: `Mask: ${attn.maskMode} · D24 RG32F G=${isMasked ? '1.0' : '0.0'}`,
    };
  }

  // o_proj / LoRA readout
  if (
    tensor === 'o_proj_base' ||
    tensor === 'lora_delta' ||
    tensor === 'attn_out' ||
    tensor === 'w_o' ||
    tensor === 'lora_a' ||
    tensor === 'lora_b' ||
    tensor === 'lora_hidden' ||
    (tensor === null && activeStep === 5)
  ) {
    const qRow =
      activeCell?.row !== null && activeCell?.row !== undefined
        ? clampInt(activeCell.row, 0, attn.seqLen - 1)
        : focusRow;
    const outDim =
      activeCell?.col !== null && activeCell?.col !== undefined
        ? clampInt(activeCell.col, 0, config.hiddenSize - 1)
        : 0;
    const idx = qRow * config.hiddenSize + outDim;
    const baseVal = attn.oProjBase[idx] ?? 0;

    let deltaVal: number | null = null;
    if (attn.loraDelta) {
      deltaVal = attn.loraDelta[idx] ?? 0;
    } else if (loraActive) {
      const hidden = linearProject(
        attn.attnConcat.subarray(qRow * config.qProjOutDim, (qRow + 1) * config.qProjOutDim),
        1,
        config.qProjOutDim,
        layerWeights.lora.loraA,
        config.loraRank,
      );
      const raw = linearProject(
        hidden,
        1,
        config.loraRank,
        layerWeights.lora.loraB,
        config.hiddenSize,
      );
      deltaVal = Math.fround(config.loraScale * (raw[outDim] ?? 0));
    }
    const outVal = attn.loraEnabled
      ? (attn.output[idx] ?? baseVal)
      : deltaVal !== null
        ? Math.fround(baseVal + deltaVal)
        : baseVal;

    const eq = formatDiffusionAttentionCellFormula({
      stage: 'o_proj',
      qHead: selectedQHead,
      kvHead: selectedKvHead,
      qRow,
      outDimIndex: outDim,
      headDim: config.headDim,
      maskMode: attn.maskMode,
      value: outVal,
      oProjBaseVal: baseVal,
      loraDeltaVal: deltaVal,
      loraScale: config.loraScale,
    });

    return {
      tensorLabel: loraActive ? 'o_proj + Draft LoRA' : 'o_proj (Backbone)',
      coordLabel: `t=${qRow}, d=${outDim}`,
      headBadge: loraActive ? 'LoRA r=4 · scale=4.0' : 'LoRA OFF',
      value: outVal,
      isMasked: false,
      valueText: `= ${fmtSigned(outVal)}`,
      equation: eq,
      subEquation: DIFFUSION_LORA_RAIL_BADGE,
    };
  }

  // Default / RoPE + YaRN + Llama-4 readout
  const role: 'Q' | 'K' =
    tensor === 'k_pre' || tensor === 'k_rot' || tensor === 'k_full' ? 'K' : 'Q';
  const plane =
    activeCell?.rotaryPlane !== null && activeCell?.rotaryPlane !== undefined
      ? clampInt(activeCell.rotaryPlane, 0, config.headDim / 2 - 1)
      : selectedRotaryPlane;
  const isSecondHalf =
    activeCell?.col !== null && activeCell?.col !== undefined
      ? activeCell.col % config.headDim >= config.headDim / 2
      : false;
  const dimIndex = isSecondHalf ? plane + config.headDim / 2 : plane;

  const angle = Math.fround(
    (role === 'Q' ? ropePhasor.queryPosition : ropePhasor.keyPosition) *
      ropePhasor.invFreq,
  );
  const cosVal = Math.fround(Math.cos(angle));
  const sinVal = Math.fround(Math.sin(angle));

  const x1 = role === 'Q' ? ropePhasor.qPrePair[0] : ropePhasor.kPrePair[0];
  const x2 = role === 'Q' ? ropePhasor.qPrePair[1] : ropePhasor.kPrePair[1];
  const rotVal =
    role === 'Q'
      ? isSecondHalf
        ? ropePhasor.qRotPair[1]
        : ropePhasor.qRotPair[0]
      : isSecondHalf
        ? ropePhasor.kRotPair[1]
        : ropePhasor.kRotPair[0];
  const scaledVal =
    role === 'Q'
      ? isSecondHalf
        ? ropePhasor.qScaledPair[1]
        : ropePhasor.qScaledPair[0]
      : undefined;

  const eq = formatRopeCellFormula({
    role,
    tokenIndex: focusRow,
    positionId: role === 'Q' ? ropePhasor.queryPosition : ropePhasor.keyPosition,
    headIndex: role === 'Q' ? selectedQHead : selectedKvHead,
    dimIndex,
    headDim: config.headDim,
    invFreq: ropePhasor.invFreq,
    zone: ropePhasor.yarnZone,
    x1,
    x2,
    cosVal,
    sinVal,
    rotatedVal: rotVal,
    llama4Scale: role === 'Q' ? ropePhasor.llama4Scale : undefined,
    scaledVal,
  });

  const shownVal = role === 'Q' && scaledVal !== undefined ? scaledVal : rotVal;
  return {
    tensorLabel: role === 'Q' ? 'YaRN RoPE + Llama-4 Q' : 'YaRN RoPE K (Unscaled)',
    coordLabel: `p=${role === 'Q' ? ropePhasor.queryPosition : ropePhasor.keyPosition}, h=${role === 'Q' ? selectedQHead : selectedKvHead}, d=${dimIndex}`,
    headBadge: `Plane (${plane},${plane + 4}) · ${ropePhasor.yarnZone}`,
    value: shownVal,
    isMasked: false,
    valueText: `= ${fmtSigned(shownVal)}`,
    equation: eq,
    subEquation:
      role === 'Q'
        ? `ω=${ropePhasor.invFreq} · Llama-4 s(p)=${ropePhasor.llama4Scale.toFixed(4)} · ${DIFFUSION_PROJECTION_REVERSAL_BADGE}`
        : `ω=${ropePhasor.invFreq} · Keys untouched by Llama-4 scaling`,
  };
}

function buildCellPopover(
  activeCell: DiffusionPickIdentity | null,
  formula: LayerFormulaReadout,
  byPart: ReadonlyMap<string, RegisteredSheet>,
  byTensor: ReadonlyMap<string, RegisteredSheet>,
): CellPopoverSpec | null {
  if (!activeCell) return null;
  const reg =
    (activeCell.partId ? byPart.get(activeCell.partId) : undefined) ??
    (activeCell.tensorKey ? byTensor.get(activeCell.tensorKey) : undefined);
  if (!reg) return null;

  const col = clampInt(activeCell.col ?? 0, 0, reg.gridCols - 1);
  const row = clampInt(activeCell.row ?? 0, 0, reg.gridRows - 1);
  const lx = -0.5 + (col + 0.5) / reg.gridCols;
  const lz = -0.5 + (row + 0.5) / reg.gridRows;
  const world: Vec3 = localToWorldOnFrame(reg.frame, lx, 0.55, lz);

  return {
    world,
    tensorLabel: formula.tensorLabel,
    coordLabel: formula.coordLabel,
    badge: formula.headBadge ?? undefined,
    equation: formula.equation,
    valueText: formula.valueText,
  };
}

function buildActiveCellGrids(params: {
  readonly trace: DiffusionLayerTrace;
  readonly config: DiffusionMiniConfig;
  readonly S: number;
  readonly K: number;
  readonly selectedQHead: number;
  readonly selectedKvHead: number;
  readonly selectedRotaryPlane: number;
  readonly focusRow: number;
  readonly focusKeyCol: number;
  readonly activeScoreSheet: RegisteredSheet | null;
  readonly activeSoftmaxSheet: RegisteredSheet | null;
  readonly llama4Sheet: RegisteredSheet;
  readonly invFreqSheet: RegisteredSheet;
  readonly ropePhasor: DiffusionRopePhasorTelemetry;
}): ActiveSheetCellGrid[] {
  const {
    trace,
    config,
    S,
    K,
    selectedQHead,
    selectedRotaryPlane,
    focusRow,
    focusKeyCol,
    activeScoreSheet,
    activeSoftmaxSheet,
    llama4Sheet,
    invFreqSheet,
    ropePhasor,
  } = params;
  const attn = trace.attention;
  const grids: ActiveSheetCellGrid[] = [];

  if (activeScoreSheet && activeSoftmaxSheet) {
    const scoreVals = new Float32Array(S * K);
    const probVals = new Float32Array(S * K);
    const masked = new Uint8Array(S * K);
    const headBase = selectedQHead * attn.seqLen * attn.kvLen;
    for (let r = 0; r < S; r++) {
      for (let c = 0; c < K; c++) {
        const sVal = attn.scores[headBase + r * attn.kvLen + c]!;
        const isM = !Number.isFinite(sVal);
        masked[r * K + c] = isM ? 1 : 0;
        scoreVals[r * K + c] = isM ? 0 : sVal;
        probVals[r * K + c] = isM ? 0 : attn.probs[headBase + r * attn.kvLen + c]!;
      }
    }

    grids.push({
      originWorld: activeScoreSheet.frame.center,
      extent: activeScoreSheet.frame.extent,
      axisU: activeScoreSheet.frame.axisU,
      axisN: activeScoreSheet.frame.axisN,
      cols: K,
      rows: S,
      values: scoreVals,
      masked,
      format: 'signed',
      activeCol: focusKeyCol,
      activeRow: focusRow,
      colDimLabel: `K=${K}`,
      rowDimLabel: `S=${S}`,
    });

    grids.push({
      originWorld: activeSoftmaxSheet.frame.center,
      extent: activeSoftmaxSheet.frame.extent,
      axisU: activeSoftmaxSheet.frame.axisU,
      axisN: activeSoftmaxSheet.frame.axisN,
      cols: K,
      rows: S,
      values: probVals,
      masked,
      format: 'prob',
      activeCol: focusKeyCol,
      activeRow: focusRow,
      colDimLabel: `K=${K}`,
      rowDimLabel: `S=${S}`,
    });
  }

  // Dyadic YaRN inv_freq [4×1]
  grids.push({
    originWorld: invFreqSheet.frame.center,
    extent: invFreqSheet.frame.extent,
    axisU: invFreqSheet.frame.axisU,
    axisN: invFreqSheet.frame.axisN,
    cols: 1,
    rows: 4,
    values: Float32Array.from(ropePhasor.allPlaneInvFreqs),
    format: 'prob',
    activeCol: 0,
    activeRow: selectedRotaryPlane,
    rowLabels: ['(0,4)', '(1,5)', '(2,6)', '(3,7)'],
    colDimLabel: 'ω',
  });

  // Llama-4 scale [32×1]
  const l4Vals = new Float32Array(DIFFUSION_GQA_MAX_KEY);
  for (let p = 0; p < DIFFUSION_GQA_MAX_KEY; p++) {
    l4Vals[p] = Math.fround(
      computeLlama4AttnScale(
        p,
        config.rope.llama4Beta,
        config.rope.originalMaxPositionEmbeddings,
      ),
    );
  }
  grids.push({
    originWorld: llama4Sheet.frame.center,
    extent: llama4Sheet.frame.extent,
    axisU: llama4Sheet.frame.axisU,
    axisN: llama4Sheet.frame.axisN,
    cols: 1,
    rows: DIFFUSION_GQA_MAX_KEY,
    values: l4Vals,
    format: 'prob',
    activeCol: 0,
    activeRow: clampInt(ropePhasor.queryPosition, 0, DIFFUSION_GQA_MAX_KEY - 1),
    colDimLabel: 's_L4',
  });

  return grids;
}
