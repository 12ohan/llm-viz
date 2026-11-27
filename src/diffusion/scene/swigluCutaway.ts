/**
 * Dense SwiGLU Cutaway Spatial View (`src/diffusion/scene/swigluCutaway.ts`)
 *
 * Renders the homogeneous dense Ministral 3 SwiGLU FFN block for any layer `0..39`:
 *   `post_attn_resid` -> `Norm₂` -> parallel `W_gate -> SiLU` (indigo operand A)
 *   & `W_up` (teal operand B) -> Hadamard `⊙` (`gated`) -> `W_down` -> `swiglu_out`
 *   -> Residual Add (`post_attn_resid + swiglu_out = layer_out`).
 *
 * Features:
 *   - Lazy layer trace resolution via `ctx.resolveLayerTrace(focus.selectedLayerIndex)`
 *   - `64×64` RG32F live value atlas (`32 KiB`, `0` overlaps)
 *   - Progressive row-major accumulation reveal via `focus.swigluAccumReveal < 0.999`
 *   - Live cell-hover formula (`formatDiffusionSwigluCellFormula`), `cellPopover`,
 *     exact `SheetFrame3D` OBB `solidPickBoxes`, and `activeCellGrids`
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import type { LayerFormulaReadout, StepperStepDef } from '../../math/formulaFormat';
import { fmtSigned } from '../../shared/format';
import type { AABB } from '../../render/camera';
import { LINE_STYLE_GUIDE, LINE_STYLE_NEUTRAL } from '../../render/linePacking';
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
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_STANDARD,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  drawCellHighlight3D,
  drawOperandDependency3D,
  RIBBON_ACTIVE,
  RIBBON_QUIET,
  RIBBON_RESIDUAL,
  spanFromSheetGrid,
  type SheetGeomRecord,
} from '../../scene/sheetInspector';
import type { DimensionCaliper, LandmarkAnchor } from '../../render/primitives';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../shared/overlayLayout';
import { formatDiffusionSwigluCellFormula } from '../swiglu';
import {
  DIFFUSION_EPISTEMIC_NOTE,
  LineBufferBuilder,
  RibbonBufferBuilder,
  SheetBufferBuilder,
  SolidBufferBuilder,
} from './provenance';
import { buildDiffusionSwigluAtlas } from './swigluAtlas';
import type {
  DiffusionFocusState,
  DiffusionPickIdentity,
  DiffusionResolvedSessionContext,
  DiffusionSpatialScene,
} from './types';

export const DIFFUSION_SWIGLU_STEPPER_STEPS: readonly StepperStepDef[] = Object.freeze([
  Object.freeze({
    index: 0,
    shortLabel: 'Norm₂',
    title: 'Post-Attention RMSNorm',
    summary: 'Normalize residual hidden states h′ ∈ ℝ^{S×8} in FP32 with γ₂ (ε = 1e-5, zero bias).',
  }),
  Object.freeze({
    index: 1,
    shortLabel: 'Gate→SiLU',
    title: 'W_gate Projection & SiLU Activation',
    summary: 'Expand 8 → 24 (prod 5120 → 16384 = 3.2×) via W_gate and apply SiLU(x) = x / (1 + e^{-x}).',
  }),
  Object.freeze({
    index: 2,
    shortLabel: 'W_up',
    title: 'Parallel W_up Linear Branch',
    summary: 'Compute parallel un-activated linear projection up = u · W_up^T ∈ ℝ^{S×24}.',
  }),
  Object.freeze({
    index: 3,
    shortLabel: 'Hadamard ⊙',
    title: 'Elementwise Gating Product',
    summary: 'Multiply gate_silu ⊙ up_val elementwise across all 24 intermediate channels.',
  }),
  Object.freeze({
    index: 4,
    shortLabel: 'W_down',
    title: 'Down Projection Back to d',
    summary: 'Contract 24 → 8 via W_down (prod 16384 → 5120) to produce swiglu_out.',
  }),
  Object.freeze({
    index: 5,
    shortLabel: 'Resid +',
    title: 'Second Residual Addition',
    summary: 'Add post_attn_resid + swiglu_out = layer_out into the persistent 40-layer stream.',
  }),
]);

const SWIGLU_OP_PARTS: readonly (readonly string[])[] = Object.freeze([
  Object.freeze(['post_attn_resid', 'norm2']),
  Object.freeze(['norm2', 'w_gate', 'gate_pre', 'gate_silu']),
  Object.freeze(['norm2', 'w_up', 'up_val']),
  Object.freeze(['gate_silu', 'up_val', 'gated']),
  Object.freeze(['gated', 'w_down', 'swiglu_out']),
  Object.freeze(['post_attn_resid', 'swiglu_out', 'layer_out']),
]);

const GATE_RIBBON_COL: FlowRibbonColor = {
  r: 0.220,
  g: 0.380,
  b: 0.741,
  a: 0.72,
  borderAlpha: 0.36,
};

const UP_RIBBON_COL: FlowRibbonColor = {
  r: 0.102,
  g: 0.541,
  b: 0.478,
  a: 0.72,
  borderAlpha: 0.36,
};

interface SwigluSheetPlacement {
  readonly id: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly cx: number;
  readonly cy: number;
  readonly cz: number;
  readonly ex: number;
  readonly ey: number;
  readonly ez: number;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly role: number;
  readonly values: Float32Array;
  readonly progressive: boolean;
  readonly branchRole?: 'operand_a' | 'operand_b' | 'target';
  readonly colDimLabel: string;
  readonly rowDimLabel: string;
}

function clampInt(val: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, Math.trunc(val)));
}

function transposeWeight(
  src: Float32Array,
  outDim: number,
  inDim: number,
): Float32Array {
  // Returns row-major [inDim × outDim] so row=inIdx, col=outIdx
  const out = new Float32Array(inDim * outDim);
  for (let r = 0; r < inDim; r++) {
    for (let c = 0; c < outDim; c++) {
      out[r * outDim + c] = src[c * inDim + r]!;
    }
  }
  return out;
}

function buildRowTokenLabels(
  tokenIds: Int32Array,
  positionIds: Int32Array,
  vocab: readonly string[],
  count: number,
): readonly string[] {
  const labels: string[] = [];
  for (let i = 0; i < count; i++) {
    const id = tokenIds[i] ?? 0;
    const pos = positionIds[i] ?? i;
    const tok = vocab[id] ?? `#${id}`;
    labels.push(`p${pos}:${tok}`);
  }
  return Object.freeze(labels);
}

function makeRangeLabels(prefix: string, count: number): readonly string[] {
  const out: string[] = [];
  for (let i = 0; i < count; i++) {
    out.push(`${prefix}${i}`);
  }
  return Object.freeze(out);
}

export function buildDiffusionSwigluCutaway(
  ctx: DiffusionResolvedSessionContext,
  focus: DiffusionFocusState,
): DiffusionSpatialScene {
  const { model, record } = ctx;
  const config = model.config;
  const d = config.hiddenSize; // 8
  const dFf = config.intermediateSize; // 24
  const layerIndex = clampInt(focus.selectedLayerIndex, 0, config.numLayers - 1);
  const layerWeights = model.layers[layerIndex]!;
  const trace = ctx.resolveLayerTrace(layerIndex);

  const atlasResult = buildDiffusionSwigluAtlas(trace, layerWeights, config, {
    layerIndex,
    snapshotDigest: ctx.kvSnapshot.snapshotDigest,
  });
  const S = atlasResult.queryTokens;

  const cp = record.primaryPassCheckpoint;
  const tokenLabels = buildRowTokenLabels(cp.inputTokenIds, cp.positionIds, model.vocab, S);
  const dLabels = makeRangeLabels('d', d);
  const fLabels = makeRangeLabels('f', dFf);

  // Dimensions tuned for clean editorial drafting-table proportions
  const cellS = 0.24;
  const sheetH = 0.068;
  const seqEz = Math.max(0.42, S * cellS);
  const dEx = d * cellS; // 1.92
  const fEx = dFf * cellS * 0.66; // 3.80

  const wGateT = transposeWeight(layerWeights.mlp.gateProj, dFf, d);
  const wUpT = transposeWeight(layerWeights.mlp.upProj, dFf, d);
  const wDownT = transposeWeight(layerWeights.mlp.downProj, d, dFf);

  // Spatial layout along X (left -> right pipeline), Z splits into Gate (-Z, indigo) & Up (+Z, teal)
  const placements: readonly SwigluSheetPlacement[] = [
    {
      id: 'post_attn_resid',
      tensorKey: 'post_attn_resid',
      label: 'Post-Attn Residual h′',
      cx: -8.6,
      cy: 0.0,
      cz: 0.0,
      ex: dEx,
      ey: sheetH,
      ez: seqEz,
      gridCols: d,
      gridRows: S,
      role: SHEET_ROLE_RESIDUAL,
      values: trace.postAttentionResidual.subarray(0, S * d),
      progressive: false,
      colDimLabel: `d=${d}`,
      rowDimLabel: `S=${S}`,
    },
    {
      id: 'norm2',
      tensorKey: 'norm2',
      label: 'Post-Attn RMSNorm u',
      cx: -5.8,
      cy: 0.0,
      cz: 0.0,
      ex: dEx,
      ey: sheetH,
      ez: seqEz,
      gridCols: d,
      gridRows: S,
      role: SHEET_ROLE_NORM,
      values: trace.postAttentionNorm.subarray(0, S * d),
      progressive: false,
      colDimLabel: `d=${d}`,
      rowDimLabel: `S=${S}`,
    },
    // Upper Branch (-Z): W_gate -> gate_pre -> gate_silu (Indigo Operand A)
    {
      id: 'w_gate',
      tensorKey: 'w_gate',
      label: 'W_gate [8×24]',
      cx: -2.4,
      cy: 0.0,
      cz: -3.55,
      ex: fEx,
      ey: sheetH,
      ez: d * 0.16,
      gridCols: dFf,
      gridRows: d,
      role: SHEET_ROLE_WEIGHT,
      values: wGateT,
      progressive: false,
      branchRole: 'operand_a',
      colDimLabel: `dFf=${dFf}`,
      rowDimLabel: `d=${d}`,
    },
    {
      id: 'gate_pre',
      tensorKey: 'gate_pre',
      label: 'gate_pre = u·W_gate^T',
      cx: -2.4,
      cy: 0.0,
      cz: -1.45,
      ex: fEx,
      ey: sheetH,
      ez: seqEz,
      gridCols: dFf,
      gridRows: S,
      role: SHEET_ROLE_ACTIVATION,
      values: trace.swiglu.gatePre.subarray(0, S * dFf),
      progressive: true,
      branchRole: 'operand_a',
      colDimLabel: `dFf=${dFf}`,
      rowDimLabel: `S=${S}`,
    },
    {
      id: 'gate_silu',
      tensorKey: 'gate_silu',
      label: 'SiLU(gate_pre)',
      cx: 2.35,
      cy: 0.0,
      cz: -1.45,
      ex: fEx,
      ey: sheetH,
      ez: seqEz,
      gridCols: dFf,
      gridRows: S,
      role: SHEET_ROLE_ACTIVATION,
      values: trace.swiglu.gateSilu.subarray(0, S * dFf),
      progressive: true,
      branchRole: 'operand_a',
      colDimLabel: `dFf=${dFf}`,
      rowDimLabel: `S=${S}`,
    },
    // Lower Branch (+Z): W_up -> up_val (Teal Operand B)
    {
      id: 'w_up',
      tensorKey: 'w_up',
      label: 'W_up [8×24]',
      cx: -2.4,
      cy: 0.0,
      cz: 3.55,
      ex: fEx,
      ey: sheetH,
      ez: d * 0.16,
      gridCols: dFf,
      gridRows: d,
      role: SHEET_ROLE_WEIGHT,
      values: wUpT,
      progressive: false,
      branchRole: 'operand_b',
      colDimLabel: `dFf=${dFf}`,
      rowDimLabel: `d=${d}`,
    },
    {
      id: 'up_val',
      tensorKey: 'up_val',
      label: 'up_val = u·W_up^T',
      cx: 0.1,
      cy: 0.0,
      cz: 1.45,
      ex: fEx,
      ey: sheetH,
      ez: seqEz,
      gridCols: dFf,
      gridRows: S,
      role: SHEET_ROLE_ACTIVATION,
      values: trace.swiglu.upVal.subarray(0, S * dFf),
      progressive: true,
      branchRole: 'operand_b',
      colDimLabel: `dFf=${dFf}`,
      rowDimLabel: `S=${S}`,
    },
    // Merge: Hadamard product -> W_down -> swiglu_out -> Residual Add -> layer_out
    {
      id: 'gated',
      tensorKey: 'gated',
      label: 'gated = SiLU(gate) ⊙ up',
      cx: 6.85,
      cy: 0.0,
      cz: 0.0,
      ex: fEx,
      ey: sheetH,
      ez: seqEz,
      gridCols: dFf,
      gridRows: S,
      role: SHEET_ROLE_ACTIVATION,
      values: trace.swiglu.gated.subarray(0, S * dFf),
      progressive: true,
      branchRole: 'target',
      colDimLabel: `dFf=${dFf}`,
      rowDimLabel: `S=${S}`,
    },
    {
      id: 'w_down',
      tensorKey: 'w_down',
      label: 'W_down [24×8]',
      cx: 10.65,
      cy: 0.0,
      cz: -2.95,
      ex: dEx,
      ey: sheetH,
      ez: dFf * 0.13,
      gridCols: d,
      gridRows: dFf,
      role: SHEET_ROLE_WEIGHT,
      values: wDownT,
      progressive: false,
      colDimLabel: `d=${d}`,
      rowDimLabel: `dFf=${dFf}`,
    },
    {
      id: 'swiglu_out',
      tensorKey: 'swiglu_out',
      label: 'swiglu_out = gated·W_down^T',
      cx: 10.65,
      cy: 0.0,
      cz: 0.0,
      ex: dEx,
      ey: sheetH,
      ez: seqEz,
      gridCols: d,
      gridRows: S,
      role: SHEET_ROLE_ACTIVATION,
      values: trace.swiglu.output.subarray(0, S * d),
      progressive: true,
      colDimLabel: `d=${d}`,
      rowDimLabel: `S=${S}`,
    },
    {
      id: 'layer_out',
      tensorKey: 'layer_out',
      label: 'Layer Output h′ + swiglu_out',
      cx: 13.55,
      cy: 0.0,
      cz: 0.0,
      ex: dEx,
      ey: sheetH,
      ez: seqEz,
      gridCols: d,
      gridRows: S,
      role: SHEET_ROLE_RESIDUAL,
      values: trace.output.subarray(0, S * d),
      progressive: false,
      colDimLabel: `d=${d}`,
      rowDimLabel: `S=${S}`,
    },
  ];

  // Resolve active cell (pinned > hovered > default focus cell on `gated`)
  const activeCell: DiffusionPickIdentity | null =
    focus.pinnedCell && focus.pinnedCell.viewMode === 'swiglu'
      ? focus.pinnedCell
      : focus.hoveredCell && focus.hoveredCell.viewMode === 'swiglu'
        ? focus.hoveredCell
        : null;

  const activePartId =
    activeCell?.partId ?? focus.selectedPartId ?? focus.hoveredPartId ?? null;
  const stepIdx = clampInt(focus.activeStepperStep, 0, DIFFUSION_SWIGLU_STEPPER_STEPS.length - 1);
  const activeOpPartIds = new Set<string>(SWIGLU_OP_PARTS[stepIdx] ?? []);
  if (activePartId) {
    activeOpPartIds.add(activePartId);
  }

  // Progressive accumulation reveal state (`focus.swigluAccumReveal < 0.999`)
  const clampedReveal = Math.max(0, Math.min(1, focus.swigluAccumReveal));
  const useReveal = clampedReveal < 0.999;

  const solids = new SolidBufferBuilder();
  const lines = new LineBufferBuilder();
  const ribbons = new RibbonBufferBuilder();
  const sheets = new SheetBufferBuilder();
  const flow = new FlowRibbonBuilder();

  const solidPickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const sheetMap = new Map<string, { placement: SwigluSheetPlacement; frame: SheetFrame3D; geom: SheetGeomRecord }>();

  // Subtle architectural drafting table border lines
  lines.addBoxFrameXZ(-10.1, -5.0, 15.1, 5.0, -0.05, 1.0, LINE_STYLE_GUIDE, 0.35);
  lines.addDashedFrameXZ(-4.7, -4.45, 4.7, -0.45, -0.04, LINE_STYLE_NEUTRAL, 0.30);
  lines.addDashedFrameXZ(-4.7, 0.45, 4.7, 4.45, -0.04, LINE_STYLE_NEUTRAL, 0.30);

  for (let idx = 0; idx < placements.length; idx++) {
    const p = placements[idx]!;
    const frame = makeSheetFrame3D(
      [p.cx, p.cy, p.cz],
      [p.ex, p.ey, p.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    const geom: SheetGeomRecord = {
      cx: p.cx,
      cy: p.cy,
      cz: p.cz,
      ex: p.ex,
      ey: p.ey,
      ez: p.ez,
      gridCols: p.gridCols,
      gridRows: p.gridRows,
      frame,
    };
    sheetMap.set(p.id, { placement: p, frame, geom });

    // Determine highlight role
    let hl = SHEET_HL_NORMAL;
    if (activeCell && activeCell.tensorKey === p.tensorKey) {
      hl = SHEET_HL_ACTIVE;
    } else if (activePartId === p.id) {
      hl = SHEET_HL_ACTIVE;
    } else if (p.branchRole === 'operand_a') {
      hl = activeOpPartIds.size > 0 && !activeOpPartIds.has(p.id) && activeCell === null
        ? SHEET_HL_DIMMED
        : SHEET_HL_OPERAND_A;
    } else if (p.branchRole === 'operand_b') {
      hl = activeOpPartIds.size > 0 && !activeOpPartIds.has(p.id) && activeCell === null
        ? SHEET_HL_DIMMED
        : SHEET_HL_OPERAND_B;
    } else if (activeOpPartIds.has(p.id)) {
      hl = SHEET_HL_PATHWAY;
    }

    // Progressive row-major reveal frontier on SwiGLU computed sheets
    let revealMode = SHEET_REVEAL_NONE;
    let revealColFrontier = p.gridCols - 1;
    let revealRowFrontier = p.gridRows - 1;
    let revealPulsePhase = 0;

    if (useReveal && p.progressive) {
      const totalCells = p.gridCols * p.gridRows;
      const cellCursor = Math.min(totalCells - 1, Math.floor(clampedReveal * totalCells));
      revealMode = SHEET_REVEAL_ROW_MAJOR;
      revealRowFrontier = Math.floor(cellCursor / p.gridCols);
      revealColFrontier = cellCursor % p.gridCols;
      revealPulsePhase = (clampedReveal * totalCells) % 1;
    }

    const slice = atlasResult.slices[p.tensorKey]!;
    sheets.push({
      cx: p.cx,
      cy: p.cy,
      cz: p.cz,
      ex: p.ex,
      ey: p.ey,
      ez: p.ez,
      gridCols: p.gridCols,
      gridRows: p.gridRows,
      sheetKind: SHEET_KIND_STANDARD,
      role: p.role,
      highlight: hl,
      slice,
      frame,
      revealMode,
      revealColFrontier,
      revealRowFrontier,
      revealPulsePhase,
    });

    const aabb = sheetFrameWorldAabb(frame);
    solidPickBoxes.push({
      bayId: layerIndex + 1,
      partId: p.id,
      tensorKey: p.tensorKey,
      gridCols: p.gridCols,
      gridRows: p.gridRows,
      absoluteTokenOffset: cp.positionIds[0] ?? 0,
      min: aabb.min,
      max: aabb.max,
      frame,
    });

    landmarks.push({
      id: `swiglu-lm-${p.id}`,
      label: p.label,
      sublabel: `${p.gridRows}×${p.gridCols}`,
      world: [p.cx, p.cy + 0.14, p.cz - p.ez * 0.5 - 0.18],
      bayId: layerIndex + 1,
      partId: p.id,
      tone:
        p.branchRole === 'operand_a'
          ? 'default'
          : p.branchRole === 'operand_b'
            ? 'cyan'
            : p.role === SHEET_ROLE_RESIDUAL
              ? 'gold'
              : 'default',
      tier:
        p.id === 'norm2' || p.id === 'gate_silu' || p.id === 'up_val' || p.id === 'gated' || p.id === 'layer_out'
          ? 'primary'
          : 'detail',
    });

    const isFocusedSheet =
      activeCell?.tensorKey === p.tensorKey ||
      activePartId === p.id ||
      p.id === 'gated' ||
      p.id === 'swiglu_out';
    if (isFocusedSheet) {
      const colLabels = p.gridCols === dFf ? fLabels : dLabels;
      const rowLabels =
        p.gridRows === S
          ? tokenLabels
          : p.gridRows === dFf
            ? fLabels
            : dLabels;
      activeCellGrids.push({
        originWorld: [p.cx, p.cy, p.cz],
        extent: [p.ex, p.ey, p.ez],
        axisU: frame.axisU,
        axisN: frame.axisN,
        cols: p.gridCols,
        rows: p.gridRows,
        values: p.values,
        format: 'signed',
        activeCol: activeCell?.tensorKey === p.tensorKey ? (activeCell.col ?? undefined) : undefined,
        activeRow: activeCell?.tensorKey === p.tensorKey ? (activeCell.row ?? undefined) : undefined,
        revealMode,
        revealColFrontier,
        revealRowFrontier,
        colLabels,
        rowLabels,
        colDimLabel: p.colDimLabel,
        rowDimLabel: p.rowDimLabel,
      });
    }
  }

  // Flow ribbons connecting the SwiGLU graph cleanly
  const flowY = 0.06;
  // post_attn_resid -> norm2
  flow.addPolylineXZ(
    [
      { x: -8.6 + dEx * 0.5 + 0.08, z: 0 },
      { x: -5.8 - dEx * 0.5 - 0.08, z: 0 },
    ],
    flowY,
    0.16,
    RIBBON_QUIET,
    true,
  );
  // norm2 -> gate_pre (indigo operand A)
  flow.addCubicXZ(
    { x: -5.8 + dEx * 0.5 + 0.06, z: -0.15 },
    { x: -4.6, z: -0.4 },
    { x: -4.6, z: -1.45 },
    { x: -2.4 - fEx * 0.5 - 0.06, z: -1.45 },
    flowY,
    0.16,
    GATE_RIBBON_COL,
    true,
  );
  // w_gate -> gate_pre
  flow.addPolylineXZ(
    [
      { x: -2.4, z: -3.55 + d * 0.08 + 0.08 },
      { x: -2.4, z: -1.45 - seqEz * 0.5 - 0.08 },
    ],
    flowY,
    0.13,
    GATE_RIBBON_COL,
    true,
  );
  // gate_pre -> gate_silu
  flow.addPolylineXZ(
    [
      { x: -2.4 + fEx * 0.5 + 0.08, z: -1.45 },
      { x: 2.35 - fEx * 0.5 - 0.08, z: -1.45 },
    ],
    flowY,
    0.16,
    GATE_RIBBON_COL,
    true,
  );
  // norm2 -> up_val (teal operand B)
  flow.addCubicXZ(
    { x: -5.8 + dEx * 0.5 + 0.06, z: 0.15 },
    { x: -4.2, z: 0.7 },
    { x: -2.9, z: 1.45 },
    { x: 0.1 - fEx * 0.5 - 0.06, z: 1.45 },
    flowY,
    0.16,
    UP_RIBBON_COL,
    true,
  );
  // w_up -> up_val
  flow.addCubicXZ(
    { x: -2.4, z: 3.55 - d * 0.08 - 0.08 },
    { x: -2.4, z: 2.45 },
    { x: 0.1, z: 2.45 },
    { x: 0.1, z: 1.45 + seqEz * 0.5 + 0.08 },
    flowY,
    0.13,
    UP_RIBBON_COL,
    true,
  );
  // gate_silu -> gated (⊙ operand A)
  flow.addCubicXZ(
    { x: 2.35 + fEx * 0.5 + 0.06, z: -1.45 },
    { x: 4.55, z: -1.45 },
    { x: 4.65, z: -0.22 },
    { x: 6.85 - fEx * 0.5 - 0.08, z: -0.22 },
    flowY,
    0.16,
    GATE_RIBBON_COL,
    true,
  );
  // up_val -> gated (⊙ operand B)
  flow.addCubicXZ(
    { x: 0.1 + fEx * 0.5 + 0.06, z: 1.45 },
    { x: 3.85, z: 1.45 },
    { x: 4.45, z: 0.22 },
    { x: 6.85 - fEx * 0.5 - 0.08, z: 0.22 },
    flowY,
    0.16,
    UP_RIBBON_COL,
    true,
  );
  // gated -> swiglu_out
  flow.addPolylineXZ(
    [
      { x: 6.85 + fEx * 0.5 + 0.08, z: 0 },
      { x: 10.65 - dEx * 0.5 - 0.08, z: 0 },
    ],
    flowY,
    0.16,
    RIBBON_ACTIVE,
    true,
  );
  // w_down -> swiglu_out
  flow.addPolylineXZ(
    [
      { x: 10.65, z: -2.95 + dFf * 0.065 + 0.08 },
      { x: 10.65, z: -seqEz * 0.5 - 0.08 },
    ],
    flowY,
    0.13,
    RIBBON_QUIET,
    true,
  );
  // swiglu_out -> layer_out
  flow.addPolylineXZ(
    [
      { x: 10.65 + dEx * 0.5 + 0.08, z: 0 },
      { x: 13.55 - dEx * 0.5 - 0.08, z: 0 },
    ],
    flowY,
    0.16,
    RIBBON_ACTIVE,
    true,
  );
  // Residual bypass highway from post_attn_resid around +Z edge to layer_out
  flow.addRoundedOrthoXZ(
    [
      { x: -8.6, z: seqEz * 0.5 + 0.14 },
      { x: -8.6, z: 4.55 },
      { x: 13.55, z: 4.55 },
      { x: 13.55, z: seqEz * 0.5 + 0.14 },
    ],
    0.45,
    flowY,
    0.18,
    RIBBON_RESIDUAL,
    true,
  );

  // Dimension calipers
  calipers.push(
    {
      id: 'swiglu-cal-dff',
      aWorld: [6.85 - fEx * 0.5, 0.10, seqEz * 0.5 + 0.32],
      bWorld: [6.85 + fEx * 0.5, 0.10, seqEz * 0.5 + 0.32],
      label: `dFf = ${dFf} (3.0× mini · prod 16,384 = 3.2×H)`,
      partIds: ['gated', 'gate_pre', 'gate_silu', 'up_val'],
    },
    {
      id: 'swiglu-cal-d',
      aWorld: [13.55 - dEx * 0.5, 0.10, -seqEz * 0.5 - 0.32],
      bWorld: [13.55 + dEx * 0.5, 0.10, -seqEz * 0.5 - 0.32],
      label: `d = ${d} (prod H = 5,120)`,
      partIds: ['layer_out', 'swiglu_out', 'norm2'],
    },
  );

  // Cell-level formula inspector, popover & 3D dependency threads
  const defaultRow = clampInt(focus.selectedTokenRow, 0, S - 1);
  const targetTensorKey = activeCell?.tensorKey ?? 'gated';
  const targetEntry = sheetMap.get(targetTensorKey) ?? sheetMap.get('gated')!;
  const targetRow = clampInt(
    activeCell?.row ?? defaultRow,
    0,
    targetEntry.placement.gridRows - 1,
  );
  const targetCol = clampInt(
    activeCell?.col ?? 0,
    0,
    targetEntry.placement.gridCols - 1,
  );
  const targetVal =
    targetEntry.placement.values[targetRow * targetEntry.placement.gridCols + targetCol] ?? 0;

  let equation = '';
  let subEquation = `Layer ${layerIndex + 1}/${config.numLayers} · Dense SwiGLU (Ministral3MLP, zero bias)`;
  const operandColors: ('indigo' | 'teal' | 'red')[] = ['red'];
  const sourceWorlds: { readonly world: readonly [number, number, number]; readonly role: 'indigo' | 'teal' }[] = [];

  const targetSpan = spanFromSheetGrid(
    targetEntry.geom,
    targetRow,
    targetRow,
    targetCol,
    targetCol,
  );
  if (activeCell !== null) {
    drawCellHighlight3D(flow, targetSpan, CELL_TARGET_FILL, CELL_TARGET_FRAME, 0.024, true);
  }

  if (targetTensorKey === 'gate_pre') {
    equation = formatDiffusionSwigluCellFormula({
      stage: 'gate_pre',
      tokenIndex: targetRow,
      channelIndex: targetCol,
      value: targetVal,
      dFf,
    });
    const normSpan = spanFromSheetGrid(sheetMap.get('norm2')!.geom, targetRow, targetRow, 0, d - 1);
    const wSpan = spanFromSheetGrid(sheetMap.get('w_gate')!.geom, 0, d - 1, targetCol, targetCol);
    if (activeCell !== null) {
      drawOperandDependency3D(flow, normSpan, targetSpan, true);
      drawOperandDependency3D(flow, wSpan, targetSpan, false);
    }
    operandColors.push('indigo', 'teal');
    sourceWorlds.push(
      { world: normSpan.frontPoint, role: 'indigo' },
      { world: wSpan.frontPoint, role: 'teal' },
    );
  } else if (targetTensorKey === 'gate_silu') {
    const preVal = trace.swiglu.gatePre[targetRow * dFf + targetCol] ?? 0;
    equation = formatDiffusionSwigluCellFormula({
      stage: 'gate_silu',
      tokenIndex: targetRow,
      channelIndex: targetCol,
      value: targetVal,
      gatePreVal: preVal,
      dFf,
    });
    const preSpan = spanFromSheetGrid(
      sheetMap.get('gate_pre')!.geom,
      targetRow,
      targetRow,
      targetCol,
      targetCol,
    );
    if (activeCell !== null) {
      drawOperandDependency3D(flow, preSpan, targetSpan, true);
    }
    operandColors.push('indigo');
    sourceWorlds.push({ world: preSpan.frontPoint, role: 'indigo' });
  } else if (targetTensorKey === 'up_val') {
    equation = formatDiffusionSwigluCellFormula({
      stage: 'up',
      tokenIndex: targetRow,
      channelIndex: targetCol,
      value: targetVal,
      dFf,
    });
    const normSpan = spanFromSheetGrid(sheetMap.get('norm2')!.geom, targetRow, targetRow, 0, d - 1);
    const wSpan = spanFromSheetGrid(sheetMap.get('w_up')!.geom, 0, d - 1, targetCol, targetCol);
    if (activeCell !== null) {
      drawOperandDependency3D(flow, normSpan, targetSpan, true);
      drawOperandDependency3D(flow, wSpan, targetSpan, false);
    }
    operandColors.push('indigo', 'teal');
    sourceWorlds.push(
      { world: normSpan.frontPoint, role: 'indigo' },
      { world: wSpan.frontPoint, role: 'teal' },
    );
  } else if (targetTensorKey === 'gated') {
    const siluVal = trace.swiglu.gateSilu[targetRow * dFf + targetCol] ?? 0;
    const upV = trace.swiglu.upVal[targetRow * dFf + targetCol] ?? 0;
    equation = formatDiffusionSwigluCellFormula({
      stage: 'gated',
      tokenIndex: targetRow,
      channelIndex: targetCol,
      value: targetVal,
      gateSiluVal: siluVal,
      upVal: upV,
      dFf,
    });
    const siluSpan = spanFromSheetGrid(
      sheetMap.get('gate_silu')!.geom,
      targetRow,
      targetRow,
      targetCol,
      targetCol,
    );
    const upSpan = spanFromSheetGrid(
      sheetMap.get('up_val')!.geom,
      targetRow,
      targetRow,
      targetCol,
      targetCol,
    );
    if (activeCell !== null) {
      drawOperandDependency3D(flow, siluSpan, targetSpan, true);
      drawOperandDependency3D(flow, upSpan, targetSpan, false);
    }
    operandColors.push('indigo', 'teal');
    sourceWorlds.push(
      { world: siluSpan.frontPoint, role: 'indigo' },
      { world: upSpan.frontPoint, role: 'teal' },
    );
  } else if (targetTensorKey === 'swiglu_out') {
    equation = formatDiffusionSwigluCellFormula({
      stage: 'output',
      tokenIndex: targetRow,
      channelIndex: targetCol,
      value: targetVal,
      dFf,
    });
    const gatedSpan = spanFromSheetGrid(
      sheetMap.get('gated')!.geom,
      targetRow,
      targetRow,
      0,
      dFf - 1,
    );
    const wDownSpan = spanFromSheetGrid(
      sheetMap.get('w_down')!.geom,
      0,
      dFf - 1,
      targetCol,
      targetCol,
    );
    if (activeCell !== null) {
      drawOperandDependency3D(flow, gatedSpan, targetSpan, true);
      drawOperandDependency3D(flow, wDownSpan, targetSpan, false);
    }
    operandColors.push('indigo', 'teal');
    sourceWorlds.push(
      { world: gatedSpan.frontPoint, role: 'indigo' },
      { world: wDownSpan.frontPoint, role: 'teal' },
    );
  } else if (targetTensorKey === 'layer_out') {
    const rVal = trace.postAttentionResidual[targetRow * d + targetCol] ?? 0;
    const mVal = trace.swiglu.output[targetRow * d + targetCol] ?? 0;
    equation = `layer_out[t=${targetRow}, d=${targetCol}] = ${fmtSigned(rVal)} + ${fmtSigned(mVal)} = ${fmtSigned(targetVal)}`;
    const residSpan = spanFromSheetGrid(
      sheetMap.get('post_attn_resid')!.geom,
      targetRow,
      targetRow,
      targetCol,
      targetCol,
    );
    const mlpSpan = spanFromSheetGrid(
      sheetMap.get('swiglu_out')!.geom,
      targetRow,
      targetRow,
      targetCol,
      targetCol,
    );
    if (activeCell !== null) {
      drawOperandDependency3D(flow, residSpan, targetSpan, true);
      drawOperandDependency3D(flow, mlpSpan, targetSpan, false);
    }
    operandColors.push('indigo', 'teal');
    sourceWorlds.push(
      { world: residSpan.frontPoint, role: 'indigo' },
      { world: mlpSpan.frontPoint, role: 'teal' },
    );
  } else if (targetTensorKey === 'norm2') {
    const rVal = trace.postAttentionResidual[targetRow * d + targetCol] ?? 0;
    const gVal = layerWeights.postAttentionLayerNorm[targetCol] ?? 1;
    equation = `norm2[t=${targetRow}, d=${targetCol}] = RMSNorm(${fmtSigned(rVal)}) · γ(${fmtSigned(gVal)}) = ${fmtSigned(targetVal)}`;
    const residSpan = spanFromSheetGrid(
      sheetMap.get('post_attn_resid')!.geom,
      targetRow,
      targetRow,
      0,
      d - 1,
    );
    if (activeCell !== null) {
      drawOperandDependency3D(flow, residSpan, targetSpan, true);
    }
    operandColors.push('indigo');
    sourceWorlds.push({ world: residSpan.frontPoint, role: 'indigo' });
  } else {
    equation = `${targetTensorKey}[r=${targetRow}, c=${targetCol}] = ${fmtSigned(targetVal)}`;
  }

  if (useReveal) {
    subEquation = `Accumulation reveal: ${(clampedReveal * 100).toFixed(0)}% · Layer ${layerIndex + 1}/${config.numLayers}`;
  }

  const coordLabel =
    targetEntry.placement.gridRows === S
      ? `${tokenLabels[targetRow] ?? `t=${targetRow}`}, ch=${targetCol}`
      : `r=${targetRow}, c=${targetCol}`;

  const activeFormula: LayerFormulaReadout = {
    tensorLabel: targetEntry.placement.label,
    coordLabel,
    headBadge: `L${String(layerIndex + 1).padStart(2, '0')} SwiGLU`,
    value: targetVal,
    isMasked: false,
    valueText: `= ${fmtSigned(targetVal)}`,
    equation,
    subEquation,
  };

  const cellPopover: CellPopoverSpec | null =
    activeCell !== null
      ? {
          world: localToWorldOnFrame(
            targetEntry.frame,
            -0.5 + (targetCol + 0.5) / targetEntry.placement.gridCols,
            0.5 + 0.08 / Math.max(0.02, targetEntry.frame.extent[1]),
            -0.5 + (targetRow + 0.5) / targetEntry.placement.gridRows,
          ),
          tensorLabel: targetEntry.placement.label,
          coordLabel,
          badge: `L${String(layerIndex + 1).padStart(2, '0')} · SwiGLU`,
          equation,
          valueText: `= ${fmtSigned(targetVal)}`,
          isAccumulating: useReveal,
          operandColors,
          sourceWorlds,
        }
      : null;

  const partCatalog: Record<string, CutawayPartInfo> = {
    post_attn_resid: {
      id: 'post_attn_resid',
      title: 'Post-Attention Residual Stream h′',
      badge: `L${layerIndex + 1} · [${S}×${d}]`,
      summary: 'Residual state h^{(ℓ-1)} + Attn(u) feeding Pre-MLP RMSNorm and the bypass add.',
      specs: [
        `Mini shape: [${S}, ${d}] · Prod 14B: [S, 5120]`,
        'Bypasses SwiGLU directly into layer_out = h′ + swiglu_out',
      ],
    },
    norm2: {
      id: 'norm2',
      title: 'Post-Attention RMSNorm (Norm₂)',
      badge: 'FP32 var · ε=1e-5',
      summary: 'Ministral3RMSNorm normalizing each token row before parallel W_gate and W_up.',
      specs: [
        `gamma: encoder.layers.${layerIndex}.post_attention_layernorm.weight [${d}]`,
        'Zero bias · shared across AR, DLM, and Linear Spec modes',
      ],
    },
    w_gate: {
      id: 'w_gate',
      title: 'Gate Projection Weight W_gate',
      badge: `Operand A · [${d}×${dFf}]`,
      summary: 'Bias-free expansion weight producing pre-activation channels for SiLU gating.',
      specs: [
        `Stored [${dFf}, ${d}] · displayed [${d}, ${dFf}] aligned with u·W_gate^T`,
        'Prod 14B: [16384, 5120] (83,886,080 BF16 params)',
      ],
    },
    gate_pre: {
      id: 'gate_pre',
      title: 'Gate Pre-Activation (u · W_gate^T)',
      badge: `Indigo Branch · [${S}×${dFf}]`,
      summary: 'Linear expansion into the 24-channel intermediate space prior to SiLU nonlinearity.',
      specs: [
        `Mini dFf = ${dFf} (3.0×d) · Prod 14B I = 16,384 (3.2×H)`,
        'Supports progressive row-major accumulation reveal',
      ],
    },
    gate_silu: {
      id: 'gate_silu',
      title: 'SiLU Activated Gate',
      badge: 'SiLU(x) = x·σ(x)',
      summary: 'Smooth self-gated nonlinearity acting as Operand A in the SwiGLU Hadamard product.',
      specs: [
        `Shape: [${S}, ${dFf}]`,
        'Exact Float32 SiLU matching PyTorch F.silu',
      ],
    },
    w_up: {
      id: 'w_up',
      title: 'Up Projection Weight W_up',
      badge: `Operand B · [${d}×${dFf}]`,
      summary: 'Parallel bias-free linear expansion weight paired with W_gate in SwiGLU.',
      specs: [
        `Stored [${dFf}, ${d}] · displayed [${d}, ${dFf}]`,
        'Prod 14B: [16384, 5120] (83,886,080 BF16 params)',
      ],
    },
    up_val: {
      id: 'up_val',
      title: 'Up Branch Activation (u · W_up^T)',
      badge: `Teal Branch · [${S}×${dFf}]`,
      summary: 'Un-activated linear value stream acting as Operand B in the elementwise gate.',
      specs: [`Shape: [${S}, ${dFf}]`, 'Multiplied elementwise with SiLU(gate_pre)'],
    },
    gated: {
      id: 'gated',
      title: 'Hadamard Gated Intermediate (⊙)',
      badge: `SiLU(gate) ⊙ up · [${S}×${dFf}]`,
      summary: 'Elementwise product of indigo SiLU gate and teal up projection.',
      specs: [
        `Shape: [${S}, ${dFf}]`,
        'Feeds W_down contraction back to hidden dimension d',
      ],
    },
    w_down: {
      id: 'w_down',
      title: 'Down Projection Weight W_down',
      badge: `[${dFf}×${d}]`,
      summary: 'Bias-free contraction projecting 24 gated channels back to hidden dimension 8.',
      specs: [
        `Stored [${d}, ${dFf}] · depth-scaled 1/sqrt(4L) in miniature init`,
        'Prod 14B: [5120, 16384] (83,886,080 BF16 params)',
      ],
    },
    swiglu_out: {
      id: 'swiglu_out',
      title: 'SwiGLU Delta Output',
      badge: `[${S}×${d}]`,
      summary: 'Contracted MLP update added onto the post-attention residual stream.',
      specs: [`Shape: [${S}, ${d}]`, 'Zero MoE routing — 100% dense homogeneous FFN on all 40 layers'],
    },
    layer_out: {
      id: 'layer_out',
      title: `Layer ${layerIndex + 1} Output State`,
      badge: `h′ + swiglu_out · [${S}×${d}]`,
      summary: 'Final residual hidden state forwarded to Layer ' + (layerIndex + 2 <= config.numLayers ? `${layerIndex + 2}` : 'Final RMSNorm'),
      specs: [
        `Shape: [${S}, ${d}]`,
        'Checkpointed in residualCheckpoints for 0-ULP lazy reconstruction',
      ],
    },
  };

  const solidPacked = solids.pack();
  const linePacked = lines.pack();
  const ribbonPacked = ribbons.pack();
  const sheetPacked = sheets.pack();
  const flowPacked = flow.pack();

  const fitBounds: AABB = {
    min: [-10.4, -0.4, -5.2],
    max: [15.4, 1.8, 5.2],
  };

  return {
    viewMode: 'swiglu',
    title: `Layer ${String(layerIndex + 1).padStart(2, '0')} Dense SwiGLU Cutaway`,
    subtitle: `Norm₂ → parallel W_gate→SiLU (indigo) & W_up (teal) → Hadamard ⊙ → W_down → Residual Add · S=${S}, d=${d}, dFf=${dFf}`,
    layerIndex,
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
    valueAtlas: atlasResult.atlas,
    atlasKey: atlasResult.atlasKey,
    atlasAudit: atlasResult.audit,
    fitBounds,
    defaultCameraAzimuthDeg: 28,
    defaultCameraElevationDeg: 38,
    landmarks,
    kvDepthTicks: [],
    calipers,
    solidPickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeOpPartIds,
    activeCellGrids,
    cellPopover,
    activeFormula,
    stepperSteps: DIFFUSION_SWIGLU_STEPPER_STEPS,
    provenanceBadges: ctx.provenanceBadges,
    epistemicNote: DIFFUSION_EPISTEMIC_NOTE,
    warningBanner: ctx.warningBanner,
    diagnostics: Object.freeze({
      layerIndex,
      queryTokens: S,
      atlasUtilizationPct: atlasResult.audit.utilizationPct,
      revealProgress: clampedReveal,
      useReveal,
    }),
  };
}
