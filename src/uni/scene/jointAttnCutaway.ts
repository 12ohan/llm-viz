/**
 * View 3 (`joint_attn`): Joint vs. Split-Tower Attention, D5 Prefix-KV & Dual-SwiGLU Cutaway
 * (`src/uni/scene/jointAttnCutaway.ts`)
 *
 * Interactive 5-pass WebGL2 spatial scene for any of the 34 Elastic-MoT layers (`L0..L33`,
 * presets `L0, L17, L18, L33`, CFG batch row `0..cfgBatchSize-1`, Q head `0..7`):
 *   - Routed `U` (indigo) & `G` (teal) QKV projection plates with smooth
 *     `UniFlatFlowBufferBuilder` cubic scatter ribbons into unified `[S, 64]` `Q_rot`
 *     and `[S, 16]` `K_rot, V_new`
 *   - 1D dyadic YaRN RoPE plane-zone indicators (`original` / `blended` / `interpolated`
 *     via `computeUniYarnInvFreq` & `UNI_DYADIC_YARN_INV_FREQ`), Llama-4 query temperature
 *     scale (`1.000` vs `1.069` via `computeLlama4AttnScale`), and `2D SinCos` badge on
 *     `G` visual tokens
 *   - Live `RG32F` attention `scores` & `softmax` (`SHEET_KIND_CAUSAL_SCORES` /
 *     `SHEET_KIND_CAUSAL_SOFTMAX`, strictly `U25`-compliant) showing the `L17 -> L18`
 *     joint-to-split mask transition + `D5` prefix-KV overlay (`hf_faithful` dead/dropped
 *     `U` prefix slices vs `elastic_pruned` pruned prefix with `0`-ULP Stage-2 equivalence)
 *   - Routed `O_proj` & `3.5×` SwiGLU (`d=8 -> d_ff=28 -> d=8`) -> `layer_out`
 *   - Rich `PickBox`es via `packPickBoxFromFrame` decoding `layerIndex`, `isSplitLayer`,
 *     `qHeadIndex`, `kvHeadIndex`, `cfgBatchRow`, and `rotaryPlane` in `pickHitToUniCell`
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`,
 * or `src/twotower/` / `src/diffusion/` / `src/ui/controls.ts` imports.
 */

import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
} from '../../render/linePacking';
import type { PickBox } from '../../render/pick';
import type { FlowRibbonColor } from '../../render/ribbons';
import {
  FRAME_XZ_AXES,
  makeSheetFrame3D,
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
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_KIND_BOX,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
} from '../../render/primitives';
import { runUniElasticLayer } from '../forward';
import { canonicalizeTokenRouting } from '../masks';
import {
  computeLlama4AttnScale,
  computeUniYarnInvFreq,
} from '../rope';
import type { UniSession } from '../session';
import { formatSiIecBytes } from '../spec';
import {
  UNI_2D_SINCOS_CANONICAL_CONTENT_SIZE,
  UNI_2D_SINCOS_SCALE,
  UNI_DYADIC_YARN_INV_FREQ,
  UNI_ROUTE_G,
  UNI_ROUTE_U,
  type UniElasticPrefillMode,
  type UniLayerKvSlice,
  type UniLayerTrace,
} from '../types';
import { packUniJointAttnAtlas } from './jointAttnAtlas';
import {
  computeAtlasContentKey,
  packPickBoxFromFrame,
  resolveUniSessionContext,
  UNI_FLOW_COLORS,
  UNI_SCENE_INTERLEAVED_BADGE,
  UniFlatFlowBufferBuilder,
  UniLineBufferBuilder,
  UniRibbonBufferBuilder,
  UniSheetBufferBuilder,
  UniSolidBufferBuilder,
  validateSceneGeometryFinite,
} from './provenance';
import type {
  UniJointAttnSceneOptions,
  UniOverlayAnchor,
  UniProvenanceBadge,
  UniResolvedSessionContext,
  UniSpatialScene,
} from './types';

export const UNI_JOINT_ATTN_LAYER_PRESETS: readonly [0, 17, 18, 33] = Object.freeze([
  0, 17, 18, 33,
] as const);

const SHEET_THICKNESS = 0.08;
const RIBBON_Y = 0.048;

const CELL_HIGHLIGHT_FILL: FlowRibbonColor = {
  r: 0.863,
  g: 0.298,
  b: 0.204,
  a: 0.16,
  borderAlpha: 0.88,
};

const CELL_HIGHLIGHT_FRAME: FlowRibbonColor = {
  r: 0.863,
  g: 0.298,
  b: 0.204,
  a: 0.92,
  borderAlpha: 0.92,
};

const LLAMA4_GOLD_RIBBON: FlowRibbonColor = {
  r: 0.765,
  g: 0.545,
  b: 0.145,
  a: 0.84,
  borderAlpha: 0.88,
};

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(v)));
}

function resolveBatchRowRouting(
  ctx: UniResolvedSessionContext,
  batchRow: number,
): Uint8Array {
  const { record, branch } = ctx;
  if (
    record.kind === 'prefill' &&
    record.cfgBatchSize > 1 &&
    branch.prefixCache.seqLen >= record.routing.length
  ) {
    const out = new Uint8Array(record.routing.length);
    for (let t = 0; t < out.length; t++) {
      out[t] = branch.prefixCache.slabs[t]!.routing[batchRow]!;
    }
    return out;
  }
  return record.routing;
}

function resolveBatchRowPastRouting(
  ctx: UniResolvedSessionContext,
  batchRow: number,
  pastLen: number,
): Uint8Array {
  const out = new Uint8Array(pastLen);
  const slabs = ctx.branch.prefixCache.slabs;
  const n = Math.min(pastLen, slabs.length);
  for (let t = 0; t < n; t++) {
    out[t] = slabs[t]!.routing[batchRow]!;
  }
  return out;
}

function resolveLayerTraceSafely(
  ctx: UniResolvedSessionContext,
  layerIndex: number,
  batchRow: number,
  rowRouting: Uint8Array,
): {
  readonly trace: UniLayerTrace;
  readonly hiddenIn: Float32Array;
  readonly layerSkippedByElasticPrune: boolean;
} {
  const rowResiduals = ctx.record.perBranchResiduals[batchRow]!;
  const inputResidual = rowResiduals[layerIndex];
  const outputResidual = rowResiduals[layerIndex + 1];

  if (inputResidual !== null && outputResidual !== null) {
    return {
      trace: ctx.resolveLayerTrace(layerIndex, batchRow),
      hiddenIn: inputResidual,
      layerSkippedByElasticPrune: false,
    };
  }

  // Fallback for U-only elastic_pruned prefill records when inspecting skipped L18..33:
  // reconstruct a pruned split-layer trace from the preserved L17 residual (`rowResiduals[18]`).
  const cfg = ctx.bundle.core.config;
  const fallbackResidual =
    rowResiduals[cfg.dualTowerLayers] ?? rowResiduals[0]!;
  const yarn = computeUniYarnInvFreq(cfg.rope);
  let pastKv: UniLayerKvSlice | null = null;
  let pastRouting: Uint8Array | null = null;
  if (ctx.record.kind === 'image_denoise' && ctx.branch.prefixCache.seqLen > 0) {
    pastRouting = resolveBatchRowPastRouting(
      ctx,
      batchRow,
      ctx.branch.prefixCache.seqLen,
    );
  }
  const trace = runUniElasticLayer({
    hiddenStates: fallbackResidual,
    seqLen: ctx.record.routing.length,
    routing: rowRouting,
    layerWeights: ctx.bundle.core.layers[layerIndex]!,
    config: cfg,
    invFreq: yarn.invFreq,
    attentionScaling: yarn.attentionScaling,
    positionIds: ctx.record.positionIds,
    cachePositions: ctx.record.positionIds,
    maskMode: ctx.record.maskMode,
    useCache: false,
    pastKv,
    pastRouting,
    pruneUnderstandingTokens: true,
  });

  return {
    trace: {
      ...trace,
      skipped: true,
    },
    hiddenIn: fallbackResidual,
    layerSkippedByElasticPrune: true,
  };
}

/**
 * Build the complete headless View 3 (`joint_attn`) spatial cutaway scene.
 */
export function buildUniJointAttnScene(
  session: UniSession,
  options?: UniJointAttnSceneOptions,
): UniSpatialScene {
  const ctx = resolveUniSessionContext(session, options?.recordId);
  const cfg = ctx.bundle.core.config;

  const layerIndex = clampInt(
    options?.layerIndex ?? cfg.dualTowerLayers, // defaults to L18 (split boundary)
    0,
    cfg.numLayers - 1,
  );
  const batchRow = clampInt(
    options?.batchRow ?? ctx.cfgBatchSize - 1,
    0,
    ctx.cfgBatchSize - 1,
  );
  const prefillMode: UniElasticPrefillMode = options?.prefillMode ?? ctx.prefillMode;
  const selectedQHead = clampInt(
    options?.selectedQHead ?? 0,
    0,
    cfg.numAttentionHeads - 1,
  );
  const selectedKvHead = Math.floor(selectedQHead / cfg.kvGroupSize) as 0 | 1;
  const isSplitLayer = layerIndex >= cfg.dualTowerLayers;

  const rowRouting = resolveBatchRowRouting(ctx, batchRow);
  const routingPlan = canonicalizeTokenRouting(rowRouting);
  const { trace, hiddenIn, layerSkippedByElasticPrune } = resolveLayerTraceSafely(
    ctx,
    layerIndex,
    batchRow,
    rowRouting,
  );
  const layerWeights = ctx.bundle.core.layers[layerIndex]!;

  const atlasPack = packUniJointAttnAtlas({
    trace,
    layerWeights,
    config: cfg,
    selectedQHead,
    hiddenIn,
    cacheTag: `${ctx.record.recordId}|b${batchRow}|${prefillMode}`,
  });

  const S = atlasPack.queryTokens;
  const K = atlasPack.keyTokens;
  const focusRow = clampInt(options?.selectedTokenRow ?? 0, 0, S - 1);
  const absTokenIdx = atlasPack.qOffset + focusRow;
  const focusTokenRoute = rowRouting[absTokenIdx] ?? UNI_ROUTE_U;
  const focusPosId =
    absTokenIdx < ctx.record.positionIds.length
      ? ctx.record.positionIds[absTokenIdx]!
      : absTokenIdx;

  // Resolve active rotary plane from hoveredCell (if any) or default plane 0
  const hoveredPlane =
    options?.hoveredCell?.rotaryPlane !== null &&
    options?.hoveredCell?.rotaryPlane !== undefined
      ? clampInt(options.hoveredCell.rotaryPlane, 0, cfg.headDim / 2 - 1)
      : 0;

  // Live 1D dyadic YaRN RoPE & Llama-4 query temperature telemetry
  const yarn = computeUniYarnInvFreq(cfg.rope);
  const llama4ActiveScale = computeLlama4AttnScale(
    focusPosId,
    cfg.rope.llama4Beta,
    cfg.rope.originalMaxPositionEmbeddings,
  );
  const llama4ExampleHi = computeLlama4AttnScale(
    cfg.rope.originalMaxPositionEmbeddings,
    cfg.rope.llama4Beta,
    cfg.rope.originalMaxPositionEmbeddings,
  );
  const llama4HighActive = llama4ActiveScale > 1.0001;

  // Determine active U/G visibility for projection plates
  const uActive =
    routingPlan.hasUnderstanding &&
    !layerSkippedByElasticPrune &&
    !(isSplitLayer && trace.understandingPruned);
  const gActive = routingPlan.hasGeneration;

  const solids = new UniSolidBufferBuilder();
  const lines = new UniLineBufferBuilder();
  const ribbons = new UniRibbonBufferBuilder();
  const sheets = new UniSheetBufferBuilder();
  const flow = new UniFlatFlowBufferBuilder();
  const pickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const kvDepthTicks: KvDepthTickAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const overlayAnchors: UniOverlayAnchor[] = [];

  const bayId = layerIndex + 1;
  const hoveredPartId = options?.hoveredCell?.partId ?? null;

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
    readonly kvHeadIndex?: 0 | 1;
    readonly tapIndex?: number;
    readonly towerRole?: 'context' | 'denoiser' | 'shared';
    readonly semanticTag: string;
    readonly landmarkTier?: LandmarkAnchor['tier'];
    readonly landmarkTone?: LandmarkAnchor['tone'];
  }): SheetFrame3D => {
    const frame = makeSheetFrame3D(
      [spec.cx, 0, spec.cz],
      [spec.ex, SHEET_THICKNESS, spec.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );

    const isHovered = hoveredPartId === spec.partId;
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
      highlight: isHovered ? SHEET_HL_ACTIVE : spec.highlight,
      slice: spec.slice,
      frame,
    });

    pickBoxes.push(
      packPickBoxFromFrame(frame, {
        bayId,
        partId: spec.partId,
        tensorKey: spec.tensorKey,
        gridCols: spec.gridCols,
        gridRows: spec.gridRows,
        ...(spec.headIndex !== undefined ? { headIndex: spec.headIndex } : {}),
        ...(spec.groupIndex !== undefined ? { groupIndex: spec.groupIndex } : {}),
        ...(spec.kvHeadIndex !== undefined ? { kvHeadIndex: spec.kvHeadIndex } : {}),
        ...(spec.tapIndex !== undefined ? { tapIndex: spec.tapIndex } : {}),
        ...(spec.towerRole !== undefined ? { towerRole: spec.towerRole } : {}),
        tokenIndex: focusRow,
        absoluteTokenOffset: focusPosId,
        semanticTag: spec.semanticTag,
      }),
    );

    landmarks.push({
      id: `lm-${spec.partId}`,
      label: spec.label,
      ...(spec.sublabel ? { sublabel: spec.sublabel } : {}),
      world: [spec.cx, 0.22, spec.cz - spec.ez * 0.5 - 0.22],
      bayId,
      partId: spec.partId,
      align: 'bottom',
      tier: spec.landmarkTier ?? 'primary',
      tone: spec.landmarkTone ?? 'default',
    });

    return frame;
  };

  // ==========================================================================
  // STAGE 1: Input Residual, Pre-Attn Norm & Routed U/G QKV Projection Plates
  // ==========================================================================
  lines.addDashedFrameXZ(-22.5, -5.8, -11.5, 5.8, -0.05, LINE_STYLE_GUIDE, 0.34);

  addSheet({
    partId: 'hidden_in',
    tensorKey: 'hidden_in',
    label: `Hidden In [${S}×8]`,
    sublabel: `L${layerIndex} · CFG row ${batchRow}`,
    cx: -21.0,
    cz: 0.0,
    ex: 1.7,
    ez: 1.5,
    gridCols: cfg.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.hiddenIn,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'normed_attn_in',
    tensorKey: 'normed_attn_in',
    label: `RMSNorm In [${S}×8]`,
    sublabel: 'ε = 1e-5 · routed γ',
    cx: -18.4,
    cz: 0.0,
    ex: 1.7,
    ez: 1.5,
    gridCols: cfg.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.normedAttnIn,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });

  const uHl = uActive ? SHEET_HL_OPERAND_A : SHEET_HL_DIMMED;
  const gHl = gActive ? SHEET_HL_OPERAND_B : SHEET_HL_DIMMED;

  // U Projection Plates (Indigo, z < 0) — omit static head/kv stamps so pickHitToUniCell infers from hitCol
  addSheet({
    partId: 'w_q_u',
    tensorKey: 'w_q_u',
    label: 'W_Q^U [8×64]',
    sublabel: uActive ? 'Understanding Q (Indigo)' : 'U inactive/pruned',
    cx: -14.4,
    cz: -4.4,
    ex: 3.4,
    ez: 1.05,
    gridCols: cfg.qProjOutDim,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: uHl,
    slice: atlasPack.slices.wQU,
    towerRole: 'context',
    semanticTag: `branch=u;cfg=${batchRow}`,
  });
  addSheet({
    partId: 'w_k_u',
    tensorKey: 'w_k_u',
    label: 'W_K^U [8×16]',
    cx: -14.4,
    cz: -2.9,
    ex: 2.3,
    ez: 1.0,
    gridCols: cfg.kvProjOutDim,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_KV_K,
    highlight: uHl,
    slice: atlasPack.slices.wKU,
    towerRole: 'context',
    semanticTag: `branch=u;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'w_v_u',
    tensorKey: 'w_v_u',
    label: 'W_V^U [8×16]',
    cx: -14.4,
    cz: -1.5,
    ex: 2.3,
    ez: 1.0,
    gridCols: cfg.kvProjOutDim,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_KV_V,
    highlight: uHl,
    slice: atlasPack.slices.wVU,
    towerRole: 'context',
    semanticTag: `branch=u;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });

  // G Projection Plates (Teal, z > 0) — omit static head/kv stamps so pickHitToUniCell infers from hitCol
  addSheet({
    partId: 'w_v_g',
    tensorKey: 'w_v_g',
    label: 'W_V^G [8×16]',
    cx: -14.4,
    cz: 1.5,
    ex: 2.3,
    ez: 1.0,
    gridCols: cfg.kvProjOutDim,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_KV_V,
    highlight: gHl,
    slice: atlasPack.slices.wVG,
    towerRole: 'denoiser',
    semanticTag: `branch=g;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'w_k_g',
    tensorKey: 'w_k_g',
    label: 'W_K^G [8×16]',
    cx: -14.4,
    cz: 2.9,
    ex: 2.3,
    ez: 1.0,
    gridCols: cfg.kvProjOutDim,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_KV_K,
    highlight: gHl,
    slice: atlasPack.slices.wKG,
    towerRole: 'denoiser',
    semanticTag: `branch=g;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'w_q_g',
    tensorKey: 'w_q_g',
    label: 'W_Q^G [8×64]',
    sublabel: gActive ? 'Generation Q (Teal)' : 'G inactive',
    cx: -14.4,
    cz: 4.4,
    ex: 3.4,
    ez: 1.05,
    gridCols: cfg.qProjOutDim,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: gHl,
    slice: atlasPack.slices.wQG,
    towerRole: 'denoiser',
    semanticTag: `branch=g;cfg=${batchRow}`,
  });

  // Flow ribbons: hidden_in -> normed_attn_in -> U/G projection banks
  flow.addPolylineXZ(
    [
      { x: -20.1, z: 0.0 },
      { x: -19.3, z: 0.0 },
    ],
    RIBBON_Y,
    0.14,
    UNI_FLOW_COLORS.jointCyan,
  );
  flow.addCubicXZ(
    { x: -17.5, z: -0.2 },
    { x: -16.8, z: -0.2 },
    { x: -16.8, z: -4.4 },
    { x: -16.15, z: -4.4 },
    RIBBON_Y,
    0.13,
    uActive ? UNI_FLOW_COLORS.uIndigo : UNI_FLOW_COLORS.ghostSlate,
  );
  flow.addCubicXZ(
    { x: -17.5, z: 0.2 },
    { x: -16.8, z: 0.2 },
    { x: -16.8, z: 4.4 },
    { x: -16.15, z: 4.4 },
    RIBBON_Y,
    0.13,
    gActive ? UNI_FLOW_COLORS.gTeal : UNI_FLOW_COLORS.ghostSlate,
  );

  // ==========================================================================
  // STAGE 2: Scatter into Unified [S×64] Q_pre/Q_rot/Q_scaled & [S×16] K_rot/V
  // ==========================================================================
  lines.addDashedFrameXZ(-11.1, -5.8, -0.2, 5.8, -0.05, LINE_STYLE_BF16_GOLD, 0.36);

  const qPreFrame = addSheet({
    partId: 'q_pre',
    tensorKey: 'q_pre',
    label: `Q_pre [${S}×64]`,
    sublabel: 'Scattered U+G queries',
    cx: -9.1,
    cz: -2.2,
    ex: 3.1,
    ez: 1.45,
    gridCols: cfg.qProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.qPre,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow}`,
  });

  const qRotFrame = addSheet({
    partId: 'q_rot',
    tensorKey: 'q_rot',
    label: `Q_rot [${S}×64]`,
    sublabel: '1D Dyadic YaRN RoPE',
    cx: -5.4,
    cz: -2.2,
    ex: 3.1,
    ez: 1.45,
    gridCols: cfg.qProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: atlasPack.slices.qRot,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow}`,
  });

  const qScaledFrame = addSheet({
    partId: 'q_scaled',
    tensorKey: 'q_scaled',
    label: `Q_scaled [${S}×64]`,
    sublabel: `Llama-4 ×${llama4ActiveScale.toFixed(3)}`,
    cx: -1.7,
    cz: -2.2,
    ex: 3.1,
    ez: 1.45,
    gridCols: cfg.qProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: llama4HighActive ? SHEET_HL_BF16_GOLD : SHEET_HL_PATHWAY,
    slice: atlasPack.slices.qScaled,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow}`,
    landmarkTone: llama4HighActive ? 'gold' : 'default',
  });

  const kRotFrame = addSheet({
    partId: 'k_rot',
    tensorKey: 'k_rot',
    label: `K_rot [${S}×16]`,
    sublabel: '2 KV heads · YaRN unscaled',
    cx: -5.4,
    cz: 1.6,
    ex: 2.3,
    ez: 1.45,
    gridCols: cfg.kvProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_KV_K,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.kRot,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'v_new',
    tensorKey: 'v_new',
    label: `V_new [${S}×16]`,
    sublabel: '2 KV heads',
    cx: -5.4,
    cz: 3.8,
    ex: 2.3,
    ez: 1.45,
    gridCols: cfg.kvProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_KV_V,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.vNew,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'rope_llama4_table',
    tensorKey: 'rope_llama4_table',
    label: 'YaRN & Llama-4 [32×10]',
    sublabel: 'cos/sin/s_L4/ω',
    cx: -9.1,
    cz: 1.6,
    ex: 1.8,
    ez: 1.65,
    gridCols: 10,
    gridRows: 32,
    role: SHEET_ROLE_NORM,
    highlight: llama4HighActive ? SHEET_HL_BF16_GOLD : SHEET_HL_NORMAL,
    slice: atlasPack.slices.ropeLlama4Table,
    tapIndex: hoveredPlane,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow};plane=${hoveredPlane}`,
    landmarkTier: 'detail',
    landmarkTone: 'gold',
  });

  // Smooth cubic scatter ribbons from U/G projection plates -> unified Q_pre, K_rot, V_new
  flow.addCubicXZ(
    { x: -12.65, z: -4.4 },
    { x: -11.7, z: -4.4 },
    { x: -11.5, z: -2.4 },
    { x: -10.7, z: -2.4 },
    RIBBON_Y,
    0.14,
    uActive ? UNI_FLOW_COLORS.uIndigo : UNI_FLOW_COLORS.ghostSlate,
  );
  flow.addCubicXZ(
    { x: -12.65, z: 4.4 },
    { x: -11.3, z: 4.4 },
    { x: -11.4, z: -2.0 },
    { x: -10.7, z: -2.0 },
    RIBBON_Y,
    0.14,
    gActive ? UNI_FLOW_COLORS.gTeal : UNI_FLOW_COLORS.ghostSlate,
  );
  flow.addCubicXZ(
    { x: -13.2, z: -2.9 },
    { x: -11.4, z: -2.9 },
    { x: -8.6, z: 1.4 },
    { x: -6.6, z: 1.4 },
    RIBBON_Y,
    0.11,
    uActive ? UNI_FLOW_COLORS.uIndigo : UNI_FLOW_COLORS.ghostSlate,
  );
  flow.addCubicXZ(
    { x: -13.2, z: 2.9 },
    { x: -11.0, z: 2.9 },
    { x: -8.5, z: 1.8 },
    { x: -6.6, z: 1.8 },
    RIBBON_Y,
    0.11,
    gActive ? UNI_FLOW_COLORS.gTeal : UNI_FLOW_COLORS.ghostSlate,
  );
  flow.addCubicXZ(
    { x: -13.2, z: 1.5 },
    { x: -10.8, z: 1.5 },
    { x: -8.6, z: 3.8 },
    { x: -6.6, z: 3.8 },
    RIBBON_Y,
    0.11,
    gActive ? UNI_FLOW_COLORS.gTeal : UNI_FLOW_COLORS.ghostSlate,
  );

  // Q_pre -> Q_rot -> Q_scaled ribbons
  flow.addPolylineXZ(
    [
      { x: -7.5, z: -2.2 },
      { x: -7.0, z: -2.2 },
    ],
    RIBBON_Y,
    0.14,
    UNI_FLOW_COLORS.jointCyan,
  );
  flow.addPolylineXZ(
    [
      { x: -3.8, z: -2.2 },
      { x: -3.3, z: -2.2 },
    ],
    RIBBON_Y,
    0.14,
    llama4HighActive ? LLAMA4_GOLD_RIBBON : UNI_FLOW_COLORS.jointCyan,
  );

  // Cell highlights for active Q head & KV head
  flow.addCellHighlightOnFrame(
    qPreFrame,
    focusRow,
    focusRow,
    selectedQHead * cfg.headDim,
    selectedQHead * cfg.headDim + cfg.headDim - 1,
    S,
    cfg.qProjOutDim,
    0.015,
    CELL_HIGHLIGHT_FILL,
    CELL_HIGHLIGHT_FRAME,
  );
  flow.addCellHighlightOnFrame(
    qRotFrame,
    focusRow,
    focusRow,
    selectedQHead * cfg.headDim,
    selectedQHead * cfg.headDim + cfg.headDim - 1,
    S,
    cfg.qProjOutDim,
    0.015,
    CELL_HIGHLIGHT_FILL,
    CELL_HIGHLIGHT_FRAME,
  );
  flow.addCellHighlightOnFrame(
    qScaledFrame,
    focusRow,
    focusRow,
    selectedQHead * cfg.headDim,
    selectedQHead * cfg.headDim + cfg.headDim - 1,
    S,
    cfg.qProjOutDim,
    0.015,
    CELL_HIGHLIGHT_FILL,
    CELL_HIGHLIGHT_FRAME,
  );
  flow.addCellHighlightOnFrame(
    kRotFrame,
    focusRow,
    focusRow,
    selectedKvHead * cfg.headDim,
    selectedKvHead * cfg.headDim + cfg.headDim - 1,
    S,
    cfg.kvProjOutDim,
    0.015,
    CELL_HIGHLIGHT_FILL,
    CELL_HIGHLIGHT_FRAME,
  );

  // ==========================================================================
  // STAGE 3: Live RG32F Scores/Softmax + L17->L18 Boundary + D5 Prefix-KV Overlay
  // ==========================================================================
  lines.addDashedFrameXZ(
    0.3,
    -5.8,
    10.1,
    5.8,
    -0.05,
    isSplitLayer ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
    0.42,
  );

  const scoresFrame = addSheet({
    partId: 'scores_selected',
    tensorKey: 'scores',
    label: `Scores Q${selectedQHead}→KV${selectedKvHead} [${S}×${K}]`,
    sublabel: isSplitLayer
      ? `L${layerIndex} SPLIT (U⊥G · G=1 masked)`
      : `L${layerIndex} JOINT (U↔G cross-modal)`,
    cx: 3.1,
    cz: -2.0,
    ex: 2.5,
    ez: 2.2,
    gridCols: K,
    gridRows: S,
    sheetKind: SHEET_KIND_CAUSAL_SCORES, // HARD U25: bound to live RG32F slice
    role: SHEET_ROLE_SCORES,
    highlight: isSplitLayer ? SHEET_HL_ACTIVE : SHEET_HL_PATHWAY,
    slice: atlasPack.slices.scoresSelected,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    kvHeadIndex: selectedKvHead,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow};kv=${selectedKvHead}`,
    landmarkTone: isSplitLayer ? 'cyan' : 'default',
  });

  const softmaxFrame = addSheet({
    partId: 'softmax_selected',
    tensorKey: 'softmax',
    label: `Softmax Q${selectedQHead} [${S}×${K}]`,
    sublabel: `${atlasPack.maskedSoftmaxCellCount} masked (G=1.0)`,
    cx: 6.4,
    cz: -2.0,
    ex: 2.5,
    ez: 2.2,
    gridCols: K,
    gridRows: S,
    sheetKind: SHEET_KIND_CAUSAL_SOFTMAX, // HARD U25: bound to live RG32F slice
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_PATHWAY,
    slice: atlasPack.slices.softmaxSelected,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    kvHeadIndex: selectedKvHead,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow};kv=${selectedKvHead}`,
  });

  addSheet({
    partId: 'softmax_overview',
    tensorKey: 'softmax_overview',
    label: `8-Head Softmax Overview [${S}×${8 * K}]`,
    sublabel: 'h0..h7 (4:1 GQA)',
    cx: 4.75,
    cz: 1.2,
    ex: 5.6,
    ez: 1.35,
    gridCols: cfg.numAttentionHeads * K,
    gridRows: S,
    sheetKind: SHEET_KIND_CAUSAL_SOFTMAX, // HARD U25: bound to live RG32F slice
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.softmaxOverview,
    headIndex: selectedQHead,
    groupIndex: selectedKvHead,
    kvHeadIndex: selectedKvHead,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow};kv=${selectedKvHead}`,
    landmarkTier: 'detail',
  });

  flow.addCellHighlightOnFrame(
    scoresFrame,
    focusRow,
    focusRow,
    0,
    K - 1,
    S,
    K,
    0.016,
    CELL_HIGHLIGHT_FILL,
    CELL_HIGHLIGHT_FRAME,
  );
  flow.addCellHighlightOnFrame(
    softmaxFrame,
    focusRow,
    focusRow,
    0,
    K - 1,
    S,
    K,
    0.016,
    CELL_HIGHLIGHT_FILL,
    CELL_HIGHLIGHT_FRAME,
  );

  // D5 Prefix-KV Cache Overlay on L18..33 (and active prefix KV rail on L0..17)
  const prefixSeqLen = ctx.branch.prefixCache.seqLen;
  const pastRouting = resolveBatchRowPastRouting(ctx, batchRow, prefixSeqLen);
  const shownPrefixSlots = Math.min(Math.max(1, prefixSeqLen), 16);
  const prefixPitchX = 5.4 / shownPrefixSlots;
  const prefixStartX = 2.05;
  const prefixZ = 3.85;

  let droppedPrefixUCount = 0;
  let activePrefixGCount = 0;
  let activePrefixUCount = 0;

  for (let t = 0; t < shownPrefixSlots; t++) {
    const r = t < pastRouting.length ? pastRouting[t]! : UNI_ROUTE_U;
    const isU = r === UNI_ROUTE_U;
    const isG = r === UNI_ROUTE_G;
    const cx = prefixStartX + (t + 0.5) * prefixPitchX;
    const width = Math.max(0.16, prefixPitchX * 0.72);

    const isDeadOrPrunedU =
      isSplitLayer &&
      isU &&
      (ctx.executionViewId === 'text_to_image' ||
        ctx.executionViewId === 'image_edit' ||
        trace.attention.dropCachedUnderstanding ||
        trace.understandingPruned);

    if (isDeadOrPrunedU) {
      droppedPrefixUCount += 1;
    } else if (isG) {
      activePrefixGCount += 1;
    } else {
      activePrefixUCount += 1;
    }

    let cellFamily = CELL_WEIGHT_BLUE;
    let ribbonColorId = RIBBON_COLOR_PERIWINKLE;
    let cy = 0.14;
    let ey = 0.24;
    let highlight = 0.0;

    if (isDeadOrPrunedU) {
      if (prefillMode === 'hf_faithful') {
        cellFamily = CELL_MOE_AMBER; // Allocated in DynamicCache, dead/dropped by split-tower G
        ribbonColorId = RIBBON_COLOR_AMBER;
        cy = 0.14;
        ey = 0.22;
        highlight = 1.0;
      } else {
        cellFamily = CELL_NEUTRAL; // Pruned in elastic_pruned (0 wasted bytes)
        ribbonColorId = RIBBON_COLOR_CYAN;
        cy = 0.06;
        ey = 0.07;
        highlight = 0.25;
      }
    } else if (isG) {
      cellFamily = CELL_ACTIVATION_GREEN;
      ribbonColorId = RIBBON_COLOR_EMERALD;
      cy = 0.16;
      ey = 0.28;
      highlight = 0.7;
    } else {
      cellFamily = CELL_SSM_CYAN;
      ribbonColorId = RIBBON_COLOR_PERIWINKLE;
      cy = 0.14;
      ey = 0.24;
      highlight = 0.4;
    }

    solids.push({
      cx,
      cy,
      cz: prefixZ,
      ex: width,
      ey,
      ez: 0.72,
      cellFamily,
      bayId,
      highlight,
    });
    lines.addBoxEdges(
      cx,
      cy,
      prefixZ,
      width,
      ey,
      0.72,
      bayId,
      isDeadOrPrunedU && prefillMode === 'hf_faithful' ? 1.2 : 0.85,
      isDeadOrPrunedU && prefillMode === 'hf_faithful'
        ? LINE_STYLE_BF16_GOLD
        : LINE_STYLE_NEUTRAL,
      isDeadOrPrunedU && prefillMode === 'elastic_pruned' ? 0.32 : 0.74,
    );
    ribbons.push({
      cx,
      cy: cy + ey * 0.5 + 0.04,
      cz: prefixZ,
      ex: width * 0.7,
      ey: 0.05,
      ez: 0.45,
      kind: RIBBON_KIND_BOX,
      colorId: ribbonColorId,
      alpha: isDeadOrPrunedU && prefillMode === 'elastic_pruned' ? 0.18 : 0.42,
    });

    kvDepthTicks.push({
      token: t + 1,
      world: [cx, cy + ey * 0.5 + 0.14, prefixZ + 0.46],
    });
  }

  const d5RailFrame = makeSheetFrame3D(
    [4.75, 0.12, prefixZ],
    [5.6, 0.32, 0.9],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  pickBoxes.push(
    packPickBoxFromFrame(d5RailFrame, {
      bayId,
      partId: 'd5_prefix_kv',
      tensorKey: 'd5_prefix_kv',
      gridCols: shownPrefixSlots,
      gridRows: 1,
      groupIndex: batchRow,
      towerRole: 'shared',
      semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow};d5=${prefillMode}`,
    }),
  );

  // Caliper showing D5 Prefix KV waste vs 0-ULP equivalence
  calipers.push({
    id: 'cal-d5-prefix-kv',
    aWorld: [prefixStartX, 0.36, prefixZ + 0.65],
    bWorld: [prefixStartX + 5.4, 0.36, prefixZ + 0.65],
    label: isSplitLayer
      ? prefillMode === 'hf_faithful'
        ? `D5 HF-Faithful L${layerIndex}: ${droppedPrefixUCount} dead/dropped U prefix KV · Wasted ${formatSiIecBytes(ctx.d5Comparison.prodSavedPrefixKvBytes)}`
        : `D5 Elastic-Pruned L${layerIndex}: ${droppedPrefixUCount} U prefix KV pruned · 0 ULP Stage-2 equiv`
      : `Joint L${layerIndex}: ${activePrefixUCount}U + ${activePrefixGCount}G prefix KV active (cross-modal)`,
    partIds: ['d5_prefix_kv', 'scores_selected', 'softmax_selected'],
  });

  // Connect Q_scaled -> Scores -> Softmax
  flow.addPolylineXZ(
    [
      { x: -0.1, z: -2.0 },
      { x: 1.8, z: -2.0 },
    ],
    RIBBON_Y,
    0.15,
    UNI_FLOW_COLORS.jointCyan,
  );
  flow.addPolylineXZ(
    [
      { x: 4.4, z: -2.0 },
      { x: 5.1, z: -2.0 },
    ],
    RIBBON_Y,
    0.15,
    UNI_FLOW_COLORS.jointCyan,
  );
  flow.addCubicXZ(
    { x: -4.2, z: 1.6 },
    { x: -1.6, z: 1.6 },
    { x: -0.2, z: -1.4 },
    { x: 1.8, z: -1.4 },
    RIBBON_Y,
    0.12,
    UNI_FLOW_COLORS.gTeal,
  );

  // ==========================================================================
  // STAGE 4: Attn Out [S×64] -> Routed W_O^U / W_O^G -> O_proj Out & Residual
  // ==========================================================================
  lines.addDashedFrameXZ(10.5, -5.0, 20.6, 5.0, -0.05, LINE_STYLE_GUIDE, 0.34);

  addSheet({
    partId: 'attn_out',
    tensorKey: 'attn_concat',
    label: `Attn Out [${S}×64]`,
    sublabel: '8 Q heads concatenated',
    cx: 12.2,
    cz: 0.0,
    ex: 2.9,
    ez: 1.45,
    gridCols: cfg.qProjOutDim,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: atlasPack.slices.attnOut,
    towerRole: 'shared',
    semanticTag: `branch=joint;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'w_o_u',
    tensorKey: 'w_o_u',
    label: 'W_O^U [8×64]',
    cx: 15.6,
    cz: -2.5,
    ex: 2.9,
    ez: 1.05,
    gridCols: cfg.qProjOutDim,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: uHl,
    slice: atlasPack.slices.wOU,
    towerRole: 'context',
    semanticTag: `branch=u;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'w_o_g',
    tensorKey: 'w_o_g',
    label: 'W_O^G [8×64]',
    cx: 15.6,
    cz: 2.5,
    ex: 2.9,
    ez: 1.05,
    gridCols: cfg.qProjOutDim,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: gHl,
    slice: atlasPack.slices.wOG,
    towerRole: 'denoiser',
    semanticTag: `branch=g;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'o_proj_out',
    tensorKey: 'o_proj_out',
    label: `O_proj Out [${S}×8]`,
    cx: 18.2,
    cz: 0.0,
    ex: 1.5,
    ez: 1.45,
    gridCols: cfg.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.oProjOut,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'post_attn_resid',
    tensorKey: 'post_attn_resid',
    label: `Post-Attn Resid [${S}×8]`,
    sublabel: 'X + O_proj Out',
    cx: 20.0,
    cz: 0.0,
    ex: 1.5,
    ez: 1.45,
    gridCols: cfg.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.postAttnResid,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });

  flow.addCubicXZ(
    { x: 7.7, z: -2.0 },
    { x: 9.2, z: -2.0 },
    { x: 9.4, z: 0.0 },
    { x: 10.7, z: 0.0 },
    RIBBON_Y,
    0.15,
    UNI_FLOW_COLORS.jointCyan,
  );
  flow.addCubicXZ(
    { x: 13.7, z: -0.2 },
    { x: 13.9, z: -0.2 },
    { x: 13.9, z: -2.5 },
    { x: 14.1, z: -2.5 },
    RIBBON_Y,
    0.12,
    uActive ? UNI_FLOW_COLORS.uIndigo : UNI_FLOW_COLORS.ghostSlate,
  );
  flow.addCubicXZ(
    { x: 13.7, z: 0.2 },
    { x: 13.9, z: 0.2 },
    { x: 13.9, z: 2.5 },
    { x: 14.1, z: 2.5 },
    RIBBON_Y,
    0.12,
    gActive ? UNI_FLOW_COLORS.gTeal : UNI_FLOW_COLORS.ghostSlate,
  );
  flow.addPolylineXZ(
    [
      { x: 19.0, z: 0.0 },
      { x: 19.25, z: 0.0 },
    ],
    RIBBON_Y,
    0.14,
    UNI_FLOW_COLORS.jointCyan,
  );

  // ==========================================================================
  // STAGE 5: Routed 3.5× SwiGLU (`d=8 -> d_ff=28 -> d=8`) -> `layer_out`
  // ==========================================================================
  lines.addDashedFrameXZ(21.2, -5.8, 37.8, 5.8, -0.05, LINE_STYLE_GUIDE, 0.34);

  addSheet({
    partId: 'mlp_norm_in',
    tensorKey: 'mlp_norm_in',
    label: `MLP Norm In [${S}×8]`,
    sublabel: 'post_attention_layernorm',
    cx: 22.2,
    cz: 0.0,
    ex: 1.5,
    ez: 1.45,
    gridCols: cfg.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.mlpNormIn,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });

  // Routed SwiGLU U/G Weights
  addSheet({
    partId: 'w_gate_u',
    tensorKey: 'w_gate_u',
    label: 'W_gate^U [8×28]',
    cx: 24.8,
    cz: -4.4,
    ex: 2.3,
    ez: 0.95,
    gridCols: cfg.intermediateSize,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: uHl,
    slice: atlasPack.slices.wGateU,
    towerRole: 'context',
    semanticTag: `branch=u;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'w_up_u',
    tensorKey: 'w_up_u',
    label: 'W_up^U [8×28]',
    cx: 24.8,
    cz: -3.1,
    ex: 2.3,
    ez: 0.95,
    gridCols: cfg.intermediateSize,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: uHl,
    slice: atlasPack.slices.wUpU,
    towerRole: 'context',
    semanticTag: `branch=u;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'w_up_g',
    tensorKey: 'w_up_g',
    label: 'W_up^G [8×28]',
    cx: 24.8,
    cz: 3.1,
    ex: 2.3,
    ez: 0.95,
    gridCols: cfg.intermediateSize,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: gHl,
    slice: atlasPack.slices.wUpG,
    towerRole: 'denoiser',
    semanticTag: `branch=g;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'w_gate_g',
    tensorKey: 'w_gate_g',
    label: 'W_gate^G [8×28]',
    cx: 24.8,
    cz: 4.4,
    ex: 2.3,
    ez: 0.95,
    gridCols: cfg.intermediateSize,
    gridRows: cfg.hiddenSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: gHl,
    slice: atlasPack.slices.wGateG,
    towerRole: 'denoiser',
    semanticTag: `branch=g;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });

  // Intermediate SwiGLU Activations (`[S, 28]`)
  addSheet({
    partId: 'gate_pre',
    tensorKey: 'gate_pre',
    label: `Gate Pre [${S}×28]`,
    cx: 27.9,
    cz: -1.6,
    ex: 2.3,
    ez: 1.2,
    gridCols: cfg.intermediateSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.gatePre,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'gate_silu',
    tensorKey: 'gate_silu',
    label: `SiLU(Gate) [${S}×28]`,
    cx: 27.9,
    cz: 0.0,
    ex: 2.3,
    ez: 1.2,
    gridCols: cfg.intermediateSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: atlasPack.slices.gateSilu,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });
  addSheet({
    partId: 'up_val',
    tensorKey: 'up_val',
    label: `Up [${S}×28]`,
    cx: 27.9,
    cz: 1.6,
    ex: 2.3,
    ez: 1.2,
    gridCols: cfg.intermediateSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.upVal,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'gated',
    tensorKey: 'gated',
    label: `Gated [${S}×28]`,
    sublabel: 'SiLU(gate) ⊙ up · 3.5×',
    cx: 30.6,
    cz: 0.0,
    ex: 2.3,
    ez: 1.35,
    gridCols: cfg.intermediateSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: atlasPack.slices.gated,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'w_down_u',
    tensorKey: 'w_down_u',
    label: 'W_down^U [28×8]',
    cx: 33.0,
    cz: -2.2,
    ex: 1.2,
    ez: 1.9,
    gridCols: cfg.hiddenSize,
    gridRows: cfg.intermediateSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: uHl,
    slice: atlasPack.slices.wDownU,
    towerRole: 'context',
    semanticTag: `branch=u;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });
  addSheet({
    partId: 'w_down_g',
    tensorKey: 'w_down_g',
    label: 'W_down^G [28×8]',
    cx: 33.0,
    cz: 2.2,
    ex: 1.2,
    ez: 1.9,
    gridCols: cfg.hiddenSize,
    gridRows: cfg.intermediateSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: gHl,
    slice: atlasPack.slices.wDownG,
    towerRole: 'denoiser',
    semanticTag: `branch=g;cfg=${batchRow}`,
    landmarkTier: 'detail',
  });

  addSheet({
    partId: 'mlp_out',
    tensorKey: 'mlp_out',
    label: `MLP Out [${S}×8]`,
    cx: 34.8,
    cz: 0.0,
    ex: 1.5,
    ez: 1.45,
    gridCols: cfg.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_NORMAL,
    slice: atlasPack.slices.mlpOut,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });

  addSheet({
    partId: 'layer_out',
    tensorKey: 'layer_out',
    label: `Layer Out [${S}×8]`,
    sublabel: `Post-MLP residual · L${layerIndex}`,
    cx: 36.8,
    cz: 0.0,
    ex: 1.5,
    ez: 1.45,
    gridCols: cfg.hiddenSize,
    gridRows: S,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_ACTIVE,
    slice: atlasPack.slices.layerOut,
    groupIndex: batchRow,
    towerRole: 'shared',
    semanticTag: `branch=joint;groupRole=cfg;cfg=${batchRow}`,
  });

  flow.addPolylineXZ(
    [
      { x: 20.8, z: 0.0 },
      { x: 21.4, z: 0.0 },
    ],
    RIBBON_Y,
    0.14,
    UNI_FLOW_COLORS.jointCyan,
  );
  flow.addPolylineXZ(
    [
      { x: 29.1, z: 0.0 },
      { x: 29.4, z: 0.0 },
    ],
    RIBBON_Y,
    0.14,
    UNI_FLOW_COLORS.jointCyan,
  );
  flow.addPolylineXZ(
    [
      { x: 35.6, z: 0.0 },
      { x: 36.0, z: 0.0 },
    ],
    RIBBON_Y,
    0.15,
    UNI_FLOW_COLORS.jointCyan,
  );

  // ==========================================================================
  // PROVENANCE BADGES & EXPLICIT OVERLAY ANCHORS (YaRN, Llama-4, 2D SinCos, D5)
  // ==========================================================================
  const yarnSummary = yarn.zones
    .map((z, p) => `p${p}:${UNI_DYADIC_YARN_INV_FREQ[p]}[${z}]`)
    .join(' · ');

  overlayAnchors.push(
    {
      id: 'anchor-yarn-rope',
      label: `1D Dyadic YaRN RoPE (4 planes: ${yarn.zones.join(' / ')})`,
      sublabel: yarnSummary,
      worldPos: [-5.4, 0.55, -2.2],
      tone: 'joint',
      badge: `Plane ${hoveredPlane}: ω=${UNI_DYADIC_YARN_INV_FREQ[hoveredPlane]} (${yarn.zones[hoveredPlane]})`,
    },
    {
      id: 'anchor-llama4-scale',
      label: `Llama-4 Q Temperature Scale: ×${llama4ActiveScale.toFixed(3)} (p=${focusPosId})`,
      sublabel: `pos<16: ×1.000 vs pos≥16: ×${llama4ExampleHi.toFixed(3)} (queries only; keys unscaled)`,
      worldPos: [-1.7, 0.55, -2.2],
      tone: llama4HighActive ? 'gold' : 'muted',
      badge: llama4HighActive ? `ACTIVE ×${llama4ActiveScale.toFixed(3)}` : '×1.000 (p<16)',
    },
    {
      id: 'anchor-2d-sincos-g',
      label: `2D SinCos Spatial Embed on G Visual Tokens (canonicalContentSize=${UNI_2D_SINCOS_CANONICAL_CONTENT_SIZE}, scale=${UNI_2D_SINCOS_SCALE})`,
      sublabel: `Additive at G input embedding · marker_column = !isSourceEdit · 1D YaRN RoPE still applied at every layer`,
      worldPos: [-14.4, 0.55, 4.4],
      tone: 'g_teal',
      badge: `2D SinCos · scale=${UNI_2D_SINCOS_SCALE}`,
    },
    {
      id: 'anchor-l17-l18-boundary',
      label: isSplitLayer
        ? `Split-Tower Layer L${layerIndex} (U⊥G Isolation · L18..33)`
        : `Joint-Attention Layer L${layerIndex} (U↔G Cross-Modal · L0..17)`,
      sublabel: `D5 [${prefillMode}]: ${droppedPrefixUCount} U prefix KV ${prefillMode === 'hf_faithful' ? 'dead/dropped' : 'pruned'} · Stage-2 0 ULP`,
      worldPos: [4.75, 0.58, -2.0],
      tone: isSplitLayer ? 'split' : 'joint',
      badge: isSplitLayer ? 'SPLIT L18..33' : 'JOINT L0..17',
    },
  );

  const provenanceBadges: UniProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    {
      label: isSplitLayer
        ? `L${layerIndex} Split-Tower (U⊥G · dual_tower_layers=18)`
        : `L${layerIndex} Joint Attention (U↔G · L0..17)`,
      tone: isSplitLayer ? 'split' : 'joint',
    },
    {
      label: `1D YaRN RoPE: [${yarn.zones.join(' · ')}] (${UNI_DYADIC_YARN_INV_FREQ.join(', ')})`,
      tone: 'muted',
    },
    {
      label: `Llama-4 Q Scale: ×${llama4ActiveScale.toFixed(3)} (1.000 vs ${llama4ExampleHi.toFixed(3)})`,
      tone: llama4HighActive ? 'gold' : 'muted',
    },
    {
      label: `2D SinCos on G Visual Tokens (canonicalContentSize=${UNI_2D_SINCOS_CANONICAL_CONTENT_SIZE}, scale=${UNI_2D_SINCOS_SCALE}, marker_column = !isSourceEdit)`,
      tone: 'g_teal',
    },
    {
      label:
        prefillMode === 'hf_faithful'
          ? `D5 Overlay [hf_faithful]: ${droppedPrefixUCount} dead/dropped U prefix KV slices · Stage-2 0 ULP`
          : `D5 Overlay [elastic_pruned]: ${droppedPrefixUCount} pruned U prefix KV slices · Stage-2 0 ULP`,
      tone: prefillMode === 'elastic_pruned' ? 'g_teal' : 'warn',
    },
  ];
  if (ctx.executionViewId === 'interleaved_mixed') {
    provenanceBadges.push({
      label: UNI_SCENE_INTERLEAVED_BADGE,
      tone: 'warn',
    });
  }

  const fitBounds: AABB = Object.freeze({
    min: Object.freeze([-23.0, -0.4, -6.4] as const),
    max: Object.freeze([38.2, 1.2, 6.4] as const),
  });

  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flow.pack();

  const atlasKey = computeAtlasContentKey(
    'joint_attn',
    atlasPack.atlas,
    `${ctx.record.recordId}|L${layerIndex}|B${batchRow}|Q${selectedQHead}|${prefillMode}`,
  );

  const tokenBranchLabel = focusTokenRoute === UNI_ROUTE_G ? 'G' : 'U';
  const scene: UniSpatialScene = {
    viewMode: 'joint_attn',
    title: `View 3 · L${layerIndex} ${isSplitLayer ? 'Split-Tower (U⊥G)' : 'Joint Attention (U↔G)'} & Dual-SwiGLU`,
    subtitle: `${ctx.viewReadout.label} · CFG row ${batchRow}/${ctx.cfgBatchSize - 1} · Q${selectedQHead}→KV${selectedKvHead} · tok[${focusRow}]=${tokenBranchLabel} (pos ${focusPosId}) · D5 ${prefillMode} (0 ULP)`,
    layerIndex,
    executionViewId: ctx.executionViewId,
    prefillMode,
    solidData: packedSolids.data,
    solidCount: packedSolids.count,
    lineData: packedLines.data,
    lineCount: packedLines.count,
    ribbonData: packedRibbons.data,
    ribbonCount: packedRibbons.count,
    sheetData: packedSheets.sheetData,
    sheetBasisData: packedSheets.sheetBasisData,
    sheetRevealData: packedSheets.sheetRevealData,
    sheetCount: packedSheets.sheetCount,
    flowVertices: packedFlow.flowVertices,
    flowIndices: packedFlow.flowIndices,
    pickBoxes: Object.freeze(pickBoxes),
    cameraPreset: Object.freeze({
      azimuthDeg: 28,
      elevationDeg: 42,
      fitBounds,
    }),
    fitBounds,
    valueAtlas: atlasPack.atlas,
    atlasKey,
    atlasRegions: atlasPack.audit.regions,
    atlasAudit: atlasPack.audit,
    rgbPreviews: Object.freeze([]),
    provenanceBadges: Object.freeze(provenanceBadges),
    overlayAnchors: Object.freeze(overlayAnchors),
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze(kvDepthTicks),
    calipers: Object.freeze(calipers),
    d5Comparison: ctx.d5Comparison,
    viewReadout: ctx.viewReadout,
  };

  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `Invalid joint_attn scene geometry: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
