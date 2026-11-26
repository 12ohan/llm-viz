/**
 * Input Embedding & Output `diffusion_head` Bookend Cutaways
 * (`src/diffusion/scene/bookendCutaways.ts`)
 *
 * Implements the two architectural bookends of `nvidia/Nemotron-Labs-Diffusion-14B`
 * (miniature `V = 36, d = 8`), each with its own non-overlapping `64×64` `RG32F` atlas:
 *
 *   1. `buildDiffusionEmbeddingCutaway`:
 *      - Packs untied `encoder.embed_tokens [36, 8]`, gathered active sequence embeddings
 *        `[S, 8]`, and dedicated special-token extract `[4, 8]` (`32 [MASK]`, `33 <|im_end|>`,
 *        `34 <s>`, `35 </think>`).
 *      - Explicitly discloses that positional information is injected exclusively via per-layer
 *        YaRN RoPE inside attention (zero learned positional embeddings).
 *
 *   2. `buildDiffusionOutputCutaway`:
 *      - Packs `finalNorm [S, 8]`, untied `diffusion_head [36, 8]`, raw un-suppressed
 *        `logits [S, 36]`, raw unscaled `probs [S, 36]` (`[MASK]` id `32` remains live!),
 *        and temperature-scaled Sampler B comparison `probs(logits / T) [S, 36]`.
 *      - Explicitly contrasts Causal Shifted (`+1`: row `i` predicts `p + 1`) vs. Diffusion
 *        Same-Position (`+0`: row `i` predicts `p`) alignment on the same shared `diffusion_head`.
 *      - Visualizes Sampler A (`unscaled softmax`, `threshold = null | 0.22 | 0.90`, Float64
 *        Gumbel-max) vs. Sampler B (`temp-scaled softmax`, `threshold = 0.0 | 0.22 | 0.90`,
 *        multinomial) threshold planes and live `[MASK]` probability readout.
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import { fmtSigned, type LayerFormulaReadout, type StepperStepDef } from '../../math/formulaFormat';
import type { AABB } from '../../render/camera';
import {
  LINE_STYLE_ACTIVE,
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
} from '../../render/sheetFrame';
import {
  SHEET_HL_BF16_GOLD,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import {
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fAtlasWriter,
} from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  drawCellHighlight3D,
  RIBBON_ACTIVE,
  RIBBON_QUIET,
  spanFromSheetGrid,
  type SheetGeomRecord,
} from '../../scene/sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../scene/spine';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../ui/overlay';
import { softmaxRows } from '../attention';
import {
  DIFFUSION_MINI_BOS_TOKEN_ID,
  DIFFUSION_MINI_END_THINK_TOKEN_ID,
  DIFFUSION_MINI_EOS_TOKEN_ID,
  DIFFUSION_MINI_MASK_TOKEN_ID,
} from '../types';
import {
  auditAtlasRegions,
  DIFFUSION_EPISTEMIC_NOTE,
  LineBufferBuilder,
  RibbonBufferBuilder,
  SheetBufferBuilder,
  SolidBufferBuilder,
} from './provenance';
import type {
  DiffusionAtlasRegionBox,
  DiffusionFocusState,
  DiffusionPickIdentity,
  DiffusionProvenanceBadge,
  DiffusionResolvedSessionContext,
  DiffusionSpatialScene,
} from './types';

export const DIFFUSION_BOOKEND_ATLAS_SIZE = 64 as const;
const MAX_BOOKEND_SEQ = 16;

export const DIFFUSION_EMBED_STEPPER_STEPS: readonly StepperStepDef[] = Object.freeze([
  Object.freeze({
    index: 0,
    shortLabel: 'Token IDs',
    title: 'Active Pass Input Token IDs (V = 36)',
    summary: '32 educational words (0..31) plus 4 special tokens: 32 [MASK], 33 <|im_end|>, 34 <s>, 35 </think>.',
  }),
  Object.freeze({
    index: 1,
    shortLabel: 'Table Lookup',
    title: 'Untied encoder.embed_tokens [36 × 8]',
    summary: 'Row gather h_0[t] = E[token_id[t]] (prod 14B: [131072, 5120] = 671.09M BF16 params, untied from diffusion_head).',
  }),
  Object.freeze({
    index: 2,
    shortLabel: 'Zero Pos Emb',
    title: 'No Learned Positional Embeddings',
    summary: 'Positional structure is injected exclusively via per-layer YaRN RoPE + Llama-4 query scale inside attention.',
  }),
]);

export const DIFFUSION_OUTPUT_STEPPER_STEPS: readonly StepperStepDef[] = Object.freeze([
  Object.freeze({
    index: 0,
    shortLabel: 'Final Norm',
    title: 'Final Ministral3RMSNorm (encoder.norm)',
    summary: 'Normalize Layer 40 output h^(40) with final RMSNorm γ (ε = 1e-5, zero bias).',
  }),
  Object.freeze({
    index: 1,
    shortLabel: 'diffusion_head',
    title: 'Shared Untied diffusion_head [36 × 8]',
    summary: 'Single bias-free projection shared across AR (+1 shifted), DLM (+0 unshifted), and Linear Spec.',
  }),
  Object.freeze({
    index: 2,
    shortLabel: 'Raw Logits',
    title: 'Un-Suppressed Logits ([MASK] Live!)',
    summary: 'Released NLD never clamps logits[MASK] (id 32 mini / 100 prod); exact raw softmax feeds both samplers.',
  }),
  Object.freeze({
    index: 3,
    shortLabel: 'Sampler A vs B',
    title: 'Dual Sampler Threshold Planes',
    summary: 'Sampler A uses unscaled softmax + F64 Gumbel-max (thr=null|0.22|0.90); Sampler B uses temp-scaled softmax + multinomial (thr=0.0|0.22|0.90).',
  }),
]);

const SHIFTED_CAUSAL_COLOR: FlowRibbonColor = {
  r: 0.220,
  g: 0.380,
  b: 0.741,
  a: 0.88,
  borderAlpha: 0.42,
};

const SAME_POS_DLM_COLOR: FlowRibbonColor = {
  r: 0.102,
  g: 0.541,
  b: 0.478,
  a: 0.88,
  borderAlpha: 0.42,
};

function clampInt(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(v)));
}

function packMatrix(
  writer: Rg32fAtlasWriter,
  regions: DiffusionAtlasRegionBox[],
  id: string,
  x0: number,
  y0: number,
  cols: number,
  rows: number,
  src: Float32Array,
  mode: 0 | 1,
): AtlasSlice {
  let maxAbs = mode === 1 ? 1.0 : 0.25;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = src[r * cols + c]!;
      writer.setTexel(x0 + c, y0 + r, val, false);
      if (mode === 0 && Number.isFinite(val)) {
        const a = Math.abs(val);
        if (a > maxAbs) maxAbs = a;
      }
    }
  }
  regions.push({
    id,
    x0,
    y0,
    cols,
    rows,
    mode,
    isMaskOrScore: false,
  });
  return {
    rect: writer.makeRect(x0, y0, cols, rows),
    scale: mode === 1 ? 1.0 : 1.0 / maxAbs,
    mode,
  };
}

// ============================================================================
// 1. INPUT EMBEDDING CUTAWAY
// ============================================================================

export function buildDiffusionEmbeddingCutaway(
  ctx: DiffusionResolvedSessionContext,
  focus: DiffusionFocusState,
): DiffusionSpatialScene {
  const { model, record } = ctx;
  const d = model.config.hiddenSize; // 8
  const V = model.config.vocabSize; // 36
  const cp = record.primaryPassCheckpoint;
  const S = Math.max(1, Math.min(cp.inputTokenIds.length, MAX_BOOKEND_SEQ));

  const size = DIFFUSION_BOOKEND_ATLAS_SIZE;
  const writer = createRg32fAtlas(size, size);
  const regions: DiffusionAtlasRegionBox[] = [];

  // Active embeddings [S × 8] from checkpoint 0
  const activeEmb = cp.residualCheckpoints[0]!.subarray(0, S * d);
  // Extract 4 special tokens [32..35] × 8
  const specialIds = [
    DIFFUSION_MINI_MASK_TOKEN_ID,
    DIFFUSION_MINI_EOS_TOKEN_ID,
    DIFFUSION_MINI_BOS_TOKEN_ID,
    DIFFUSION_MINI_END_THINK_TOKEN_ID,
  ] as const;
  const specialEmb = new Float32Array(specialIds.length * d);
  for (let i = 0; i < specialIds.length; i++) {
    const id = specialIds[i]!;
    specialEmb.set(model.embedTokens.subarray(id * d, (id + 1) * d), i * d);
  }

  const embedSlice = packMatrix(writer, regions, 'embed_tokens', 0, 0, d, V, model.embedTokens, 0);
  const activeSlice = packMatrix(writer, regions, 'active_embeddings', 12, 0, d, S, activeEmb, 0);
  const specialSlice = packMatrix(writer, regions, 'special_embeddings', 12, 20, d, 4, specialEmb, 0);

  const audit = auditAtlasRegions(size, size, regions);
  if (audit.hasOverlap || audit.outOfBoundsIds.length > 0) {
    throw new Error(
      `Embedding 64×64 atlas layout error: overlap=${String(audit.overlapPair)}, oob=${audit.outOfBoundsIds.join(',')}`,
    );
  }

  const solids = new SolidBufferBuilder();
  const lines = new LineBufferBuilder();
  const ribbons = new RibbonBufferBuilder();
  const sheets = new SheetBufferBuilder();
  const flow = new FlowRibbonBuilder();

  const solidPickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];

  lines.addBoxFrameXZ(-6.8, -4.8, 6.8, 4.8, -0.05, 1.0, LINE_STYLE_GUIDE, 0.34);

  // Full untied encoder.embed_tokens [36 × 8] on the left
  const tableEx = 2.4;
  const tableEz = 7.92; // 36 rows * 0.22
  const tableCx = -3.4;
  const tableCz = 0.0;
  const tableFrame = makeSheetFrame3D(
    [tableCx, 0.0, tableCz],
    [tableEx, 0.072, tableEz],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  const tableGeom: SheetGeomRecord = {
    cx: tableCx,
    cy: 0.0,
    cz: tableCz,
    ex: tableEx,
    ey: 0.072,
    ez: tableEz,
    gridCols: d,
    gridRows: V,
    frame: tableFrame,
  };
  sheets.push({
    cx: tableCx,
    cy: 0.0,
    cz: tableCz,
    ex: tableEx,
    ey: 0.072,
    ez: tableEz,
    gridCols: d,
    gridRows: V,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_NORMAL,
    slice: embedSlice,
    frame: tableFrame,
  });
  const tableAabb = sheetFrameWorldAabb(tableFrame);
  solidPickBoxes.push({
    bayId: 0,
    partId: 'embed_tokens',
    tensorKey: 'embed_tokens',
    gridCols: d,
    gridRows: V,
    min: tableAabb.min,
    max: tableAabb.max,
    frame: tableFrame,
  });

  // Active sequence embeddings [S × 8] on upper right
  const actEx = 2.6;
  const actEz = Math.max(0.65, S * 0.34);
  const actCx = 2.9;
  const actCz = -1.6;
  const actFrame = makeSheetFrame3D(
    [actCx, 0.0, actCz],
    [actEx, 0.072, actEz],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  const actGeom: SheetGeomRecord = {
    cx: actCx,
    cy: 0.0,
    cz: actCz,
    ex: actEx,
    ey: 0.072,
    ez: actEz,
    gridCols: d,
    gridRows: S,
    frame: actFrame,
  };
  sheets.push({
    cx: actCx,
    cy: 0.0,
    cz: actCz,
    ex: actEx,
    ey: 0.072,
    ez: actEz,
    gridCols: d,
    gridRows: S,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: activeSlice,
    frame: actFrame,
  });
  const actAabb = sheetFrameWorldAabb(actFrame);
  solidPickBoxes.push({
    bayId: 0,
    partId: 'active_embeddings',
    tensorKey: 'active_embeddings',
    gridCols: d,
    gridRows: S,
    absoluteTokenOffset: cp.positionIds[0] ?? 0,
    min: actAabb.min,
    max: actAabb.max,
    frame: actFrame,
  });

  // Special-tokens extract [4 × 8] on lower right highlighting 32 [MASK], 33 <|im_end|>, 34 <s>, 35 </think>
  const specEx = 2.6;
  const specEz = 1.68;
  const specCx = 2.9;
  const specCz = 2.45;
  const specFrame = makeSheetFrame3D(
    [specCx, 0.0, specCz],
    [specEx, 0.072, specEz],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  sheets.push({
    cx: specCx,
    cy: 0.0,
    cz: specCz,
    ex: specEx,
    ey: 0.072,
    ez: specEz,
    gridCols: d,
    gridRows: 4,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: specialSlice,
    frame: specFrame,
  });
  const specAabb = sheetFrameWorldAabb(specFrame);
  solidPickBoxes.push({
    bayId: 0,
    partId: 'special_embeddings',
    tensorKey: 'special_embeddings',
    gridCols: d,
    gridRows: 4,
    min: specAabb.min,
    max: specAabb.max,
    frame: specFrame,
  });

  // Lookup ribbons from each active pass token's row in embed_tokens to its row in active_embeddings
  const selRow = clampInt(focus.selectedTokenRow, 0, S - 1);
  for (let t = 0; t < S; t++) {
    const tokId = cp.inputTokenIds[t]!;
    const srcZ = tableCz - tableEz * 0.5 + ((tokId + 0.5) / V) * tableEz;
    const dstZ = actCz - actEz * 0.5 + ((t + 0.5) / S) * actEz;
    const isSel = t === selRow;
    flow.addCubicXZ(
      { x: tableCx + tableEx * 0.5 + 0.06, z: srcZ },
      { x: -0.6, z: srcZ },
      { x: 0.3, z: dstZ },
      { x: actCx - actEx * 0.5 - 0.06, z: dstZ },
      0.06,
      isSel ? 0.14 : 0.07,
      isSel ? RIBBON_ACTIVE : RIBBON_QUIET,
      isSel,
      20,
    );
  }

  // Highlight special rows 32..35 in embed_tokens with gold bracket + ribbon to special extract
  const specRowStartZ = tableCz - tableEz * 0.5 + (32 / V) * tableEz;
  const specRowEndZ = tableCz + tableEz * 0.5;
  lines.addBoxFrameXZ(
    tableCx - tableEx * 0.5 - 0.06,
    specRowStartZ,
    tableCx + tableEx * 0.5 + 0.06,
    specRowEndZ,
    0.06,
    1.5,
    LINE_STYLE_BF16_GOLD,
    0.9,
  );
  flow.addCubicXZ(
    { x: tableCx + tableEx * 0.5 + 0.08, z: (specRowStartZ + specRowEndZ) * 0.5 },
    { x: -0.3, z: 3.55 },
    { x: 0.6, z: specCz },
    { x: specCx - specEx * 0.5 - 0.08, z: specCz },
    0.06,
    0.12,
    { r: 0.812, g: 0.569, b: 0.145, a: 0.85, borderAlpha: 0.4 },
    true,
    18,
  );

  // Landmarks & calipers
  const activeRowLabels = Array.from({ length: S }, (_, i) => {
    const id = cp.inputTokenIds[i]!;
    return `p${cp.positionIds[i] ?? i}:${model.vocab[id] ?? id}`;
  });
  const specialLabels = [
    '32 [MASK] (prod 100)',
    '33 <|im_end|> (prod 11)',
    '34 <s> (prod 1)',
    '35 </think> (prod 13)',
  ];
  const dLabels = Array.from({ length: d }, (_, i) => `d${i}`);

  landmarks.push(
    {
      id: 'embed-lm-table',
      label: `Untied encoder.embed_tokens [${V}×${d}]`,
      sublabel: 'Prod 14B: [131,072 × 5,120] = 671.09M BF16 params',
      world: [tableCx, 0.16, tableCz - tableEz * 0.5 - 0.30],
      partId: 'embed_tokens',
      tier: 'primary',
    },
    {
      id: 'embed-lm-active',
      label: `Active Sequence Embeddings h_0 [${S}×${d}]`,
      sublabel: 'Zero learned pos embeddings — YaRN RoPE injects positions inside attention',
      world: [actCx, 0.16, actCz - actEz * 0.5 - 0.28],
      partId: 'active_embeddings',
      tone: 'cyan',
      tier: 'primary',
    },
    {
      id: 'embed-lm-specials',
      label: 'Special Tokens: 32 [MASK] · 33 <|im_end|> · 34 <s> · 35 </think>',
      sublabel: 'D5 verified production IDs: MASK=100, EOS=11, BOS=1, </think>=13',
      world: [specCx, 0.16, specCz - specEz * 0.5 - 0.24],
      partId: 'special_embeddings',
      tone: 'gold',
      tier: 'primary',
    },
  );

  calipers.push(
    {
      id: 'embed-cal-vocab',
      aWorld: [tableCx - tableEx * 0.5 - 0.26, 0.08, tableCz - tableEz * 0.5],
      bWorld: [tableCx - tableEx * 0.5 - 0.26, 0.08, tableCz + tableEz * 0.5],
      label: `V = ${V} (0..31 words + 4 special · prod 131,072)`,
      partIds: ['embed_tokens'],
    },
    {
      id: 'embed-cal-specials',
      aWorld: [tableCx + tableEx * 0.5 + 0.16, 0.08, specRowStartZ],
      bWorld: [tableCx + tableEx * 0.5 + 0.16, 0.08, specRowEndZ],
      label: '32 [MASK]..35 </think>',
      partIds: ['embed_tokens', 'special_embeddings'],
    },
  );

  activeCellGrids.push(
    {
      originWorld: [actCx, 0.0, actCz],
      extent: [actEx, 0.072, actEz],
      axisU: actFrame.axisU,
      axisN: actFrame.axisN,
      cols: d,
      rows: S,
      values: activeEmb,
      format: 'signed',
      colLabels: dLabels,
      rowLabels: activeRowLabels,
      colDimLabel: `d=${d}`,
      rowDimLabel: `S=${S}`,
    },
    {
      originWorld: [specCx, 0.0, specCz],
      extent: [specEx, 0.072, specEz],
      axisU: specFrame.axisU,
      axisN: specFrame.axisN,
      cols: d,
      rows: 4,
      values: specialEmb,
      format: 'signed',
      colLabels: dLabels,
      rowLabels: specialLabels,
      colDimLabel: `d=${d}`,
      rowDimLabel: 'Specials',
    },
  );

  const activeCell: DiffusionPickIdentity | null =
    focus.pinnedCell && focus.pinnedCell.viewMode === 'embedding'
      ? focus.pinnedCell
      : focus.hoveredCell && focus.hoveredCell.viewMode === 'embedding'
        ? focus.hoveredCell
        : null;

  const focusTokRow = clampInt(
    activeCell?.tensorKey === 'active_embeddings' ? (activeCell.row ?? selRow) : selRow,
    0,
    S - 1,
  );
  const focusCol = clampInt(activeCell?.col ?? 0, 0, d - 1);
  const focusTokId = cp.inputTokenIds[focusTokRow]!;
  const val = activeEmb[focusTokRow * d + focusCol] ?? 0;

  if (activeCell !== null) {
    const geom = activeCell.tensorKey === 'embed_tokens' ? tableGeom : actGeom;
    const r = clampInt(activeCell.row ?? focusTokRow, 0, geom.gridRows - 1);
    const c = clampInt(activeCell.col ?? focusCol, 0, geom.gridCols - 1);
    drawCellHighlight3D(flow, spanFromSheetGrid(geom, r, r, c, c), CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.022, true);
  }

  const equation = `h_0[t=${focusTokRow}, d=${focusCol}] = embed_tokens[id=${focusTokId} (${model.vocab[focusTokId]}), ${focusCol}] = ${fmtSigned(val)}`;
  const activeFormula: LayerFormulaReadout = {
    tensorLabel: 'encoder.embed_tokens Lookup',
    coordLabel: `${activeRowLabels[focusTokRow]}, d=${focusCol}`,
    headBadge: 'Untied · Zero Pos Emb',
    value: val,
    isMasked: false,
    valueText: `= ${fmtSigned(val)}`,
    equation,
    subEquation: 'Positional encoding is injected exclusively via per-layer YaRN RoPE inside attention (zero learned positional table).',
  };

  const cellPopover: CellPopoverSpec | null =
    activeCell !== null
      ? {
          world: localToWorldOnFrame(
            activeCell.tensorKey === 'embed_tokens' ? tableFrame : actFrame,
            -0.5 + (focusCol + 0.5) / d,
            0.5 + 0.08 / 0.072,
            -0.5 + ((activeCell.row ?? focusTokRow) + 0.5) / (activeCell.tensorKey === 'embed_tokens' ? V : S),
          ),
          tensorLabel: activeCell.tensorKey === 'embed_tokens' ? 'encoder.embed_tokens' : 'Active Embeddings h_0',
          coordLabel: `id=${focusTokId} (${model.vocab[focusTokId]}), d=${focusCol}`,
          badge: 'Untied',
          equation,
          valueText: `= ${fmtSigned(val)}`,
          operandColors: ['indigo'],
        }
      : null;

  const partCatalog: Record<string, CutawayPartInfo> = {
    embed_tokens: {
      id: 'embed_tokens',
      title: 'Untied encoder.embed_tokens [36 × 8]',
      badge: 'Untied · Zero Pos Emb',
      summary: 'Input vocabulary lookup table separate from diffusion_head. No positional embeddings are added here; YaRN RoPE rotates Q and K inside every attention layer.',
      specs: [
        `Mini: [36, 8] (288 floats) · Prod 14B: [131072, 5120] (671,088,640 BF16 params = 1.2500 GiB)`,
        'Special tokens: 32 [MASK], 33 <|im_end|>, 34 <s>, 35 </think>',
      ],
    },
    active_embeddings: {
      id: 'active_embeddings',
      title: `Active Sequence Embeddings h_0 [${S} × ${d}]`,
      badge: `S = ${S} token(s)`,
      summary: 'Gathered input vectors feeding Layer 01 pre-attention RMSNorm (residualCheckpoints[0]).',
      specs: [
        `Positions: ${Array.from(cp.positionIds.subarray(0, S)).join(', ')}`,
        'Zero learned positional table; RoPE + Llama-4 query scale apply inside each layer',
      ],
    },
    special_embeddings: {
      id: 'special_embeddings',
      title: 'Special Control Token Rows (32..35)',
      badge: 'D5 Verified Mapping',
      summary: 'Dedicated rows for diffusion masking, sequence boundaries, and thinking-budget termination.',
      specs: [
        '32 [MASK] ↔ prod 100 (<SPECIAL_100>) · never clamped in output logits',
        '33 <|im_end|> ↔ prod 11 (Instruct EOS) · 34 <s> ↔ prod 1 (BOS) · 35 </think> ↔ prod 13',
      ],
    },
  };

  const badges: DiffusionProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    { label: 'Untied embed_tokens · Zero Learned Pos Emb (YaRN RoPE in Attn)', tone: 'causal' },
  ];

  const solidPacked = solids.pack();
  const linePacked = lines.pack();
  const ribbonPacked = ribbons.pack();
  const sheetPacked = sheets.pack();
  const flowPacked = flow.pack();

  const fitBounds: AABB = {
    min: [-7.0, -0.4, -5.0],
    max: [7.0, 1.6, 5.0],
  };

  return {
    viewMode: 'embedding',
    title: 'Input Embedding Bookend Cutaway (encoder.embed_tokens)',
    subtitle: `Untied V=${V} × d=${d} lookup highlighting 32 [MASK], 33 <|im_end|>, 34 <s>, 35 </think> · Positional encoding via per-layer YaRN RoPE only`,
    layerIndex: null,
    solidData: solidPacked.data,
    solidCount: solidPacked.count,
    lineData: linePacked.data,
    lineCount: linePacked.count,
    ribbonData: ribbonPacked.data,
    ribbonCount: ribbonPacked.count,
    sheetData: sheetPacked.sheetData,
    sheetCount: sheetPacked.sheetCount,
    sheetBasisData: sheetPacked.sheetBasisData,
    sheetRevealData: sheetPacked.sheetRevealData,
    flowVertices: flowPacked.vertices,
    flowIndices: flowPacked.indices,
    valueAtlas: {
      width: size,
      height: size,
      data: writer.data,
    },
    atlasKey: `embedding|rec=${record.recordId}|S=${S}`,
    atlasAudit: audit,
    fitBounds,
    defaultCameraAzimuthDeg: 22,
    defaultCameraElevationDeg: 44,
    landmarks,
    kvDepthTicks: [],
    calipers,
    solidPickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeCellGrids,
    cellPopover,
    activeFormula,
    stepperSteps: DIFFUSION_EMBED_STEPPER_STEPS,
    provenanceBadges: Object.freeze(badges),
    epistemicNote: DIFFUSION_EPISTEMIC_NOTE,
    warningBanner: ctx.warningBanner,
    diagnostics: Object.freeze({
      vocabSize: V,
      hiddenSize: d,
      activeTokens: S,
      specialTokenIds: specialIds,
      atlasUtilizationPct: audit.utilizationPct,
    }),
  };
}

// ============================================================================
// 2. OUTPUT `diffusion_head` CUTAWAY
// ============================================================================

export function buildDiffusionOutputCutaway(
  ctx: DiffusionResolvedSessionContext,
  focus: DiffusionFocusState,
): DiffusionSpatialScene {
  const { model, record } = ctx;
  const d = model.config.hiddenSize; // 8
  const V = model.config.vocabSize; // 36
  const cp = record.primaryPassCheckpoint;
  const S = Math.max(1, Math.min(cp.inputTokenIds.length, MAX_BOOKEND_SEQ));
  const cfg = record.configAtStep;

  const size = DIFFUSION_BOOKEND_ATLAS_SIZE;
  const writer = createRg32fAtlas(size, size);
  const regions: DiffusionAtlasRegionBox[] = [];

  const finalNorm = cp.finalNorm.subarray(0, S * d);
  const rawLogits = cp.logits.subarray(0, S * V);
  const rawProbs = cp.probs.subarray(0, S * V);

  // Sampler B temperature-scaled softmax comparison (`T = cfg.temperature > 0 ? cfg.temperature : 0.7`)
  const compareTemp = cfg.temperature > 0 ? cfg.temperature : 0.7;
  const scaledLogits = new Float32Array(S * V);
  for (let i = 0; i < S * V; i++) {
    scaledLogits[i] = Math.fround(rawLogits[i]! / compareTemp);
  }
  const tempScaledProbs = softmaxRows(scaledLogits, S, V);

  // Disjoint 64×64 packing:
  //   final_norm        [8 × S]   at (0,  0)
  //   diffusion_head    [8 × 36]  at (10, 0)
  //   logits            [36 × S]  at (20, 0)
  //   probs             [36 × S]  at (20, 20)
  //   temp_scaled_probs [36 × S]  at (20, 40)
  const normSlice = packMatrix(writer, regions, 'final_norm', 0, 0, d, S, finalNorm, 0);
  const headSlice = packMatrix(writer, regions, 'diffusion_head', 10, 0, d, V, model.diffusionHead, 0);
  const logitsSlice = packMatrix(writer, regions, 'logits', 20, 0, V, S, rawLogits, 0);
  const probsSlice = packMatrix(writer, regions, 'probs', 20, 20, V, S, rawProbs, 1);
  const tempProbsSlice = packMatrix(writer, regions, 'temp_scaled_probs', 20, 40, V, S, tempScaledProbs, 1);

  const audit = auditAtlasRegions(size, size, regions);
  if (audit.hasOverlap || audit.outOfBoundsIds.length > 0) {
    throw new Error(
      `Output 64×64 atlas layout error: overlap=${String(audit.overlapPair)}, oob=${audit.outOfBoundsIds.join(',')}`,
    );
  }

  const solids = new SolidBufferBuilder();
  const lines = new LineBufferBuilder();
  const ribbons = new RibbonBufferBuilder();
  const sheets = new SheetBufferBuilder();
  const flow = new FlowRibbonBuilder();

  const solidPickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];

  lines.addBoxFrameXZ(-9.6, -5.4, 9.6, 5.8, -0.05, 1.0, LINE_STYLE_GUIDE, 0.34);

  const seqEz = Math.max(0.60, S * 0.28);
  const normEx = 2.0;
  const headEx = 2.0;
  const headEz = 6.84; // 36 * 0.19
  const vocabEx = 7.92; // 36 * 0.22

  // 1. Final RMSNorm [S × 8]
  const normCx = -7.6;
  const normCz = -1.1;
  const normFrame = makeSheetFrame3D(
    [normCx, 0.0, normCz],
    [normEx, 0.068, seqEz],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  sheets.push({
    cx: normCx,
    cy: 0.0,
    cz: normCz,
    ex: normEx,
    ey: 0.068,
    ez: seqEz,
    gridCols: d,
    gridRows: S,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_NORMAL,
    slice: normSlice,
    frame: normFrame,
  });
  const normAabb = sheetFrameWorldAabb(normFrame);
  solidPickBoxes.push({
    bayId: 0,
    partId: 'final_norm',
    tensorKey: 'final_norm',
    gridCols: d,
    gridRows: S,
    min: normAabb.min,
    max: normAabb.max,
    frame: normFrame,
  });

  // 2. Shared Untied diffusion_head [36 × 8]
  const headCx = -4.2;
  const headCz = -0.3;
  const headFrame = makeSheetFrame3D(
    [headCx, 0.0, headCz],
    [headEx, 0.072, headEz],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  sheets.push({
    cx: headCx,
    cy: 0.0,
    cz: headCz,
    ex: headEx,
    ey: 0.072,
    ez: headEz,
    gridCols: d,
    gridRows: V,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_PATHWAY,
    slice: headSlice,
    frame: headFrame,
  });
  const headAabb = sheetFrameWorldAabb(headFrame);
  solidPickBoxes.push({
    bayId: 0,
    partId: 'diffusion_head',
    tensorKey: 'diffusion_head',
    gridCols: d,
    gridRows: V,
    min: headAabb.min,
    max: headAabb.max,
    frame: headFrame,
  });

  // 3. Raw Un-Suppressed Logits [S × 36]
  const logitsCx = 3.2;
  const logitsCz = -3.1;
  const logitsFrame = makeSheetFrame3D(
    [logitsCx, 0.0, logitsCz],
    [vocabEx, 0.068, seqEz],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  const logitsGeom: SheetGeomRecord = {
    cx: logitsCx,
    cy: 0.0,
    cz: logitsCz,
    ex: vocabEx,
    ey: 0.068,
    ez: seqEz,
    gridCols: V,
    gridRows: S,
    frame: logitsFrame,
  };
  sheets.push({
    cx: logitsCx,
    cy: 0.0,
    cz: logitsCz,
    ex: vocabEx,
    ey: 0.068,
    ez: seqEz,
    gridCols: V,
    gridRows: S,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_NORMAL,
    slice: logitsSlice,
    frame: logitsFrame,
  });
  const logitsAabb = sheetFrameWorldAabb(logitsFrame);
  solidPickBoxes.push({
    bayId: 0,
    partId: 'logits',
    tensorKey: 'logits',
    gridCols: V,
    gridRows: S,
    min: logitsAabb.min,
    max: logitsAabb.max,
    frame: logitsFrame,
  });

  // 4. Sampler A Unscaled Softmax Probs [S × 36] (with [MASK] id 32 live!)
  const probsCx = 3.2;
  const probsCz = -0.3;
  const probsFrame = makeSheetFrame3D(
    [probsCx, 0.0, probsCz],
    [vocabEx, 0.068, seqEz],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  const probsGeom: SheetGeomRecord = {
    cx: probsCx,
    cy: 0.0,
    cz: probsCz,
    ex: vocabEx,
    ey: 0.068,
    ez: seqEz,
    gridCols: V,
    gridRows: S,
    frame: probsFrame,
  };
  sheets.push({
    cx: probsCx,
    cy: 0.0,
    cz: probsCz,
    ex: vocabEx,
    ey: 0.068,
    ez: seqEz,
    gridCols: V,
    gridRows: S,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_OPERAND_A,
    slice: probsSlice,
    frame: probsFrame,
  });
  const probsAabb = sheetFrameWorldAabb(probsFrame);
  solidPickBoxes.push({
    bayId: 0,
    partId: 'probs',
    tensorKey: 'probs',
    gridCols: V,
    gridRows: S,
    min: probsAabb.min,
    max: probsAabb.max,
    frame: probsFrame,
  });

  // 5. Sampler B Temperature-Scaled Softmax Probs [S × 36]
  const tempCx = 3.2;
  const tempCz = 2.0;
  const tempFrame = makeSheetFrame3D(
    [tempCx, 0.0, tempCz],
    [vocabEx, 0.068, seqEz],
    FRAME_XZ_AXES.axisU,
    FRAME_XZ_AXES.axisN,
  );
  sheets.push({
    cx: tempCx,
    cy: 0.0,
    cz: tempCz,
    ex: vocabEx,
    ey: 0.068,
    ez: seqEz,
    gridCols: V,
    gridRows: S,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_OPERAND_B,
    slice: tempProbsSlice,
    frame: tempFrame,
  });
  const tempAabb = sheetFrameWorldAabb(tempFrame);
  solidPickBoxes.push({
    bayId: 0,
    partId: 'temp_scaled_probs',
    tensorKey: 'temp_scaled_probs',
    gridCols: V,
    gridRows: S,
    min: tempAabb.min,
    max: tempAabb.max,
    frame: tempFrame,
  });

  // --------------------------------------------------------------------------
  // 3D PROBABILITY BAR HISTOGRAM + SAMPLER A vs B THRESHOLD PLANES (z = 4.25)
  // --------------------------------------------------------------------------
  const selRow = clampInt(focus.selectedTokenRow, 0, S - 1);
  const barZ = 4.25;
  const barPitch = vocabEx / V;
  const barStartX = probsCx - vocabEx * 0.5 + barPitch * 0.5;
  const barMaxH = 1.9;

  let topVocabId = 0;
  let topProb = -1;
  for (let v = 0; v < V; v++) {
    const pA = rawProbs[selRow * V + v]!;
    if (pA > topProb) {
      topProb = pA;
      topVocabId = v;
    }
    const x = barStartX + v * barPitch;
    const h = Math.max(0.03, pA * barMaxH);
    const isMask = v === DIFFUSION_MINI_MASK_TOKEN_ID;
    const isEos = v === DIFFUSION_MINI_EOS_TOKEN_ID;
    solids.push({
      cx: x,
      cy: h * 0.5,
      cz: barZ,
      ex: barPitch * 0.72,
      ey: h,
      ez: 0.34,
      cellFamily: isMask
        ? CELL_MOE_AMBER
        : isEos
          ? CELL_SSM_CYAN
          : pA >= (cfg.samplerAThreshold ?? 0.22)
            ? CELL_ACTIVATION_GREEN
            : CELL_WEIGHT_BLUE,
      highlight: isMask ? 1 : 0,
    });
  }

  // Sampler A vs Sampler B Threshold Planes across the probability bars (0.22 calibrated & 0.90 prod)
  const thrLowY = 0.22 * barMaxH;
  const thrHighY = 0.90 * barMaxH;
  lines.addBoxFrameXZ(
    probsCx - vocabEx * 0.5 - 0.1,
    barZ - 0.24,
    probsCx + vocabEx * 0.5 + 0.1,
    barZ + 0.24,
    thrLowY,
    1.5,
    LINE_STYLE_BF16_GOLD,
    0.85,
  );
  lines.addBoxFrameXZ(
    probsCx - vocabEx * 0.5 - 0.1,
    barZ - 0.24,
    probsCx + vocabEx * 0.5 + 0.1,
    barZ + 0.24,
    thrHighY,
    1.5,
    LINE_STYLE_ACTIVE,
    0.85,
  );
  // Subtle translucent ribbon plane at active Sampler A threshold
  const activeThr = cfg.samplerAThreshold ?? 0.22;
  ribbons.push({
    cx: probsCx,
    cy: Math.min(barMaxH, Math.max(0.05, activeThr * barMaxH)),
    cz: barZ,
    ex: vocabEx + 0.18,
    ey: 0.02,
    ez: 0.44,
    kind: 0,
    colorId: 3, // amber
    alpha: 0.22,
  });

  // Dual alignment ribbons emerging from shared diffusion_head:
  //   1. Causal Shifted (+1): row i predicts p + 1 (indigo)
  //   2. Diffusion Same-Position (+0): row i predicts p (teal)
  flow.addPolylineXZ(
    [
      { x: normCx + normEx * 0.5 + 0.08, z: normCz },
      { x: headCx - headEx * 0.5 - 0.08, z: normCz },
    ],
    0.06,
    0.16,
    RIBBON_QUIET,
    true,
  );
  flow.addCubicXZ(
    { x: headCx + headEx * 0.5 + 0.08, z: -1.4 },
    { x: -2.0, z: -1.4 },
    { x: -1.6, z: logitsCz },
    { x: logitsCx - vocabEx * 0.5 - 0.08, z: logitsCz },
    0.06,
    0.16,
    SHIFTED_CAUSAL_COLOR,
    true,
  );
  flow.addCubicXZ(
    { x: headCx + headEx * 0.5 + 0.08, z: -0.3 },
    { x: -1.8, z: -0.3 },
    { x: -1.5, z: probsCz },
    { x: probsCx - vocabEx * 0.5 - 0.08, z: probsCz },
    0.06,
    0.16,
    SAME_POS_DLM_COLOR,
    true,
  );
  flow.addCubicXZ(
    { x: headCx + headEx * 0.5 + 0.08, z: 1.2 },
    { x: -1.8, z: 1.2 },
    { x: -1.5, z: tempCz },
    { x: tempCx - vocabEx * 0.5 - 0.08, z: tempCz },
    0.06,
    0.14,
    SAME_POS_DLM_COLOR,
    true,
  );

  // Highlight live [MASK] column (id 32) across logits & probs sheets
  const maskColX = probsCx - vocabEx * 0.5 + ((DIFFUSION_MINI_MASK_TOKEN_ID + 0.5) / V) * vocabEx;
  lines.addSegment(maskColX, 0.08, logitsCz - seqEz * 0.5 - 0.12, maskColX, 0.08, barZ + 0.25, 1.5, LINE_STYLE_BF16_GOLD, 0.80);

  const maskProb = rawProbs[selRow * V + DIFFUSION_MINI_MASK_TOKEN_ID]!;
  const maskLogit = rawLogits[selRow * V + DIFFUSION_MINI_MASK_TOKEN_ID]!;
  const selAbsPos = cp.positionIds[selRow] ?? selRow;
  const isCausalPass = cp.maskMode === 'causal';

  const rowLabels = Array.from({ length: S }, (_, i) => {
    const pos = cp.positionIds[i] ?? i;
    return isCausalPass ? `r${i}(p${pos}→p${pos + 1})` : `r${i}(p${pos}→p${pos})`;
  });

  landmarks.push(
    {
      id: 'out-lm-norm',
      label: `Final RMSNorm [${S}×${d}]`,
      sublabel: 'encoder.norm.weight (ε=1e-5)',
      world: [normCx, 0.16, normCz - seqEz * 0.5 - 0.24],
      partId: 'final_norm',
      tier: 'detail',
    },
    {
      id: 'out-lm-head',
      label: `Shared Untied diffusion_head [${V}×${d}]`,
      sublabel: 'Causal Shifted (+1: row i → p+1) vs. Diffusion Same-Pos (+0: row i → p)',
      world: [headCx, 0.16, headCz - headEz * 0.5 - 0.26],
      partId: 'diffusion_head',
      tone: 'gold',
      tier: 'primary',
    },
    {
      id: 'out-lm-logits',
      label: `Raw Un-Suppressed Logits [${S}×${V}]`,
      sublabel: `Row ${selRow} [MASK] id=32 logit = ${fmtSigned(maskLogit)} (NEVER clamped)`,
      world: [logitsCx, 0.16, logitsCz - seqEz * 0.5 - 0.22],
      partId: 'logits',
      tier: 'primary',
    },
    {
      id: 'out-lm-probs-a',
      label: `Sampler A: Unscaled Softmax (thr=${cfg.samplerAThreshold === null ? 'null/quota' : cfg.samplerAThreshold.toFixed(2)})`,
      sublabel: `F64 Gumbel-max · live P([MASK])=${(maskProb * 100).toFixed(2)}% · top=${model.vocab[topVocabId]} (${(topProb * 100).toFixed(1)}%)`,
      world: [probsCx, 0.16, probsCz - seqEz * 0.5 - 0.22],
      partId: 'probs',
      tone: 'default',
      tier: 'primary',
    },
    {
      id: 'out-lm-probs-b',
      label: `Sampler B: Temp-Scaled Softmax (T=${compareTemp.toFixed(2)}, thr=${cfg.samplerBThreshold.toFixed(2)})`,
      sublabel: 'Multinomial(softmax(logits / T)) · single-pass commit-all at thr=0.0',
      world: [tempCx, 0.16, tempCz - seqEz * 0.5 - 0.22],
      partId: 'temp_scaled_probs',
      tone: 'cyan',
      tier: 'primary',
    },
    {
      id: 'out-lm-thr-planes',
      label: 'Threshold Planes: 0.22 (mini) · 0.90 (prod CLI)',
      sublabel: `Amber bar = live [MASK] (id 32 · ${(maskProb * 100).toFixed(2)}%)`,
      world: [maskColX, thrHighY + 0.18, barZ],
      partId: 'probs',
      tone: 'gold',
      tier: 'primary',
    },
  );

  calipers.push(
    {
      id: 'out-cal-vocab',
      aWorld: [probsCx - vocabEx * 0.5, 0.08, barZ + 0.42],
      bWorld: [probsCx + vocabEx * 0.5, 0.08, barZ + 0.42],
      label: `V = ${V} (0..31 words · 32 [MASK] live · 33 <|im_end|> · 34 <s> · 35 </think>)`,
      partIds: ['logits', 'probs', 'temp_scaled_probs'],
    },
  );

  activeCellGrids.push(
    {
      originWorld: [logitsCx, 0.0, logitsCz],
      extent: [vocabEx, 0.068, seqEz],
      axisU: logitsFrame.axisU,
      axisN: logitsFrame.axisN,
      cols: V,
      rows: S,
      values: rawLogits,
      format: 'signed',
      colLabels: model.vocab,
      rowLabels,
      colDimLabel: `V=${V}`,
      rowDimLabel: `S=${S}`,
    },
    {
      originWorld: [probsCx, 0.0, probsCz],
      extent: [vocabEx, 0.068, seqEz],
      axisU: probsFrame.axisU,
      axisN: probsFrame.axisN,
      cols: V,
      rows: S,
      values: rawProbs,
      format: 'prob',
      colLabels: model.vocab,
      rowLabels,
      colDimLabel: `V=${V}`,
      rowDimLabel: `S=${S}`,
    },
  );

  const activeCell: DiffusionPickIdentity | null =
    focus.pinnedCell && focus.pinnedCell.viewMode === 'output'
      ? focus.pinnedCell
      : focus.hoveredCell && focus.hoveredCell.viewMode === 'output'
        ? focus.hoveredCell
        : null;

  const inspectRow = clampInt(
    activeCell && (activeCell.tensorKey === 'logits' || activeCell.tensorKey === 'probs' || activeCell.tensorKey === 'temp_scaled_probs')
      ? (activeCell.row ?? selRow)
      : selRow,
    0,
    S - 1,
  );
  const inspectCol = clampInt(
    activeCell && (activeCell.tensorKey === 'logits' || activeCell.tensorKey === 'probs' || activeCell.tensorKey === 'temp_scaled_probs')
      ? (activeCell.col ?? DIFFUSION_MINI_MASK_TOKEN_ID)
      : topVocabId,
    0,
    V - 1,
  );
  const cellLogit = rawLogits[inspectRow * V + inspectCol]!;
  const cellProbA = rawProbs[inspectRow * V + inspectCol]!;
  const cellProbB = tempScaledProbs[inspectRow * V + inspectCol]!;

  if (activeCell !== null && (activeCell.tensorKey === 'logits' || activeCell.tensorKey === 'probs')) {
    const geom = activeCell.tensorKey === 'logits' ? logitsGeom : probsGeom;
    drawCellHighlight3D(
      flow,
      spanFromSheetGrid(geom, inspectRow, inspectRow, inspectCol, inspectCol),
      CELL_TARGET_FILL,
      CELL_TARGET_FRAME,
      0.022,
      true,
    );
  }

  const alignNote = isCausalPass
    ? `Causal Shifted (+1): row ${inspectRow} at p=${selAbsPos} predicts p=${selAbsPos + 1}`
    : `Diffusion Same-Position (+0): row ${inspectRow} at p=${selAbsPos} predicts p=${selAbsPos}`;

  const equation = `tok=${inspectCol} (${model.vocab[inspectCol]}): logit=${fmtSigned(cellLogit)}, P_A=${(cellProbA * 100).toFixed(2)}%, P_B(T=${compareTemp.toFixed(2)})=${(cellProbB * 100).toFixed(2)}%`;
  const activeFormula: LayerFormulaReadout = {
    tensorLabel: 'Untied diffusion_head Output',
    coordLabel: `row=${inspectRow} (p=${selAbsPos}), vocab=${inspectCol} (${model.vocab[inspectCol]})`,
    headBadge: isCausalPass ? 'Causal Shifted (+1)' : 'Diffusion Same-Pos (+0)',
    value: cellProbA,
    isMasked: false,
    valueText: `P_A = ${(cellProbA * 100).toFixed(2)}%`,
    equation,
    subEquation: `${alignNote} · Live P([MASK] id=32) = ${(maskProb * 100).toFixed(2)}% (un-suppressed)`,
  };

  const cellPopover: CellPopoverSpec | null =
    activeCell !== null && (activeCell.tensorKey === 'logits' || activeCell.tensorKey === 'probs')
      ? {
          world: localToWorldOnFrame(
            activeCell.tensorKey === 'logits' ? logitsFrame : probsFrame,
            -0.5 + (inspectCol + 0.5) / V,
            0.5 + 0.08 / 0.068,
            -0.5 + (inspectRow + 0.5) / S,
          ),
          tensorLabel: activeCell.tensorKey === 'logits' ? 'Raw Logits' : 'Unscaled Softmax (Sampler A)',
          coordLabel: `r=${inspectRow}, ${model.vocab[inspectCol]} (id=${inspectCol})`,
          badge: inspectCol === DIFFUSION_MINI_MASK_TOKEN_ID ? '[MASK] Live' : undefined,
          equation,
          valueText: activeCell.tensorKey === 'logits' ? `= ${fmtSigned(cellLogit)}` : `= ${(cellProbA * 100).toFixed(2)}%`,
          operandColors: ['indigo', 'teal'],
        }
      : null;

  const partCatalog: Record<string, CutawayPartInfo> = {
    final_norm: {
      id: 'final_norm',
      title: 'Final RMSNorm (encoder.norm)',
      badge: `S=${S} × d=${d} · ε=1e-5`,
      summary: 'Final Ministral3RMSNorm normalizing Layer 40 output before untied diffusion_head.',
      specs: ['Prod 14B: encoder.norm.weight [5120] (10.00 KiB BF16)'],
    },
    diffusion_head: {
      id: 'diffusion_head',
      title: 'Shared Untied diffusion_head [36 × 8]',
      badge: 'Causal (+1) & DLM (+0)',
      summary:
        'Single bias-free output head shared across all 4 execution modes: AR & Spec Verify read shifted (+1) logits while DLM denoise & Spec Draft read same-position (+0) logits.',
      specs: [
        'Prod 14B: [131072, 5120] = 671,088,640 BF16 params (untied from encoder.embed_tokens)',
        alignNote,
      ],
    },
    logits: {
      id: 'logits',
      title: 'Raw Un-Suppressed Logits',
      badge: `[MASK] id=32 Live (${fmtSigned(maskLogit)})`,
      summary: 'Exact linear projection finalNorm · diffusion_head^T with zero −1e12 [MASK] logit suppression.',
      specs: ['Preserves exact PyTorch parity with upstream modeling_nemotron_labs_diffusion.py'],
    },
    probs: {
      id: 'probs',
      title: 'Sampler A: Unscaled Softmax & F64 Gumbel-Max',
      badge: `thr=${cfg.samplerAThreshold === null ? 'null (quota)' : cfg.samplerAThreshold.toFixed(2)}`,
      summary: 'Used by DLM generate() _get_transfer_index: ALWAYS computes confidence from UNSCALED softmax(logits), with non-strict conf >= threshold and select_index[0] floor.',
      specs: [
        `Live [MASK] prob (row ${selRow}): ${(maskProb * 100).toFixed(2)}%`,
        'Threshold presets: null (static quota), 0.22 (mini), 0.90 (prod CLI)',
      ],
    },
    temp_scaled_probs: {
      id: 'temp_scaled_probs',
      title: 'Sampler B: Temp-Scaled Softmax & Multinomial',
      badge: `T=${compareTemp.toFixed(2)} · thr=${cfg.samplerBThreshold.toFixed(2)}`,
      summary: 'Used by linear_spec_generate() draft: computes confidence from temperature-scaled softmax(logits / T) and samples via multinomial.',
      specs: [
        'Threshold 0.0 = single-pass commit-all; > 0 = iterative draft with D18 maxDraftPassesPerCycle guard',
      ],
    },
  };

  const badges: DiffusionProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    {
      label: isCausalPass ? 'Alignment: Causal Shifted (+1)' : 'Alignment: Diffusion Same-Pos (+0)',
      tone: isCausalPass ? 'causal' : 'bidir',
    },
    {
      label: `[MASK] Live P = ${(maskProb * 100).toFixed(2)}% (Zero Clamping)`,
      tone: 'gold',
    },
  ];

  const solidPacked = solids.pack();
  const linePacked = lines.pack();
  const ribbonPacked = ribbons.pack();
  const sheetPacked = sheets.pack();
  const flowPacked = flow.pack();

  const fitBounds: AABB = {
    min: [-9.8, -0.4, -5.6],
    max: [9.8, 2.5, 6.0],
  };

  return {
    viewMode: 'output',
    title: 'Output diffusion_head Bookend Cutaway',
    subtitle: `Untied diffusion_head [${V}×${d}] · Causal (+1) vs Diffusion (+0) alignment · Sampler A (unscaled) vs Sampler B (T-scaled) · Live [MASK]=${(maskProb * 100).toFixed(2)}%`,
    layerIndex: null,
    solidData: solidPacked.data,
    solidCount: solidPacked.count,
    lineData: linePacked.data,
    lineCount: linePacked.count,
    ribbonData: ribbonPacked.data,
    ribbonCount: ribbonPacked.count,
    sheetData: sheetPacked.sheetData,
    sheetCount: sheetPacked.sheetCount,
    sheetBasisData: sheetPacked.sheetBasisData,
    sheetRevealData: sheetPacked.sheetRevealData,
    flowVertices: flowPacked.vertices,
    flowIndices: flowPacked.indices,
    valueAtlas: {
      width: size,
      height: size,
      data: writer.data,
    },
    atlasKey: `output|rec=${record.recordId}|S=${S}|T=${compareTemp.toFixed(3)}`,
    atlasAudit: audit,
    fitBounds,
    defaultCameraAzimuthDeg: 24,
    defaultCameraElevationDeg: 40,
    landmarks,
    kvDepthTicks: [],
    calipers,
    solidPickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeCellGrids,
    cellPopover,
    activeFormula,
    stepperSteps: DIFFUSION_OUTPUT_STEPPER_STEPS,
    provenanceBadges: Object.freeze(badges),
    epistemicNote: DIFFUSION_EPISTEMIC_NOTE,
    warningBanner: ctx.warningBanner,
    diagnostics: Object.freeze({
      vocabSize: V,
      activeTokens: S,
      alignmentMode: isCausalPass ? 'causal_shifted_plus_1' : 'diffusion_same_pos_plus_0',
      maskTokenLiveProb: maskProb,
      maskTokenRawLogit: maskLogit,
      topVocabId,
      topProb,
      compareTemp,
      atlasUtilizationPct: audit.utilizationPct,
    }),
  };
}
