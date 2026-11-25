import { formatEpsLabel } from '../math/gqaSim';
import type {
  HybridFinalNormResult,
  HybridLmHeadResult,
  HybridModelParameters,
  HybridPrediction,
} from '../math/hybridFlow';
import { NEMOTRON_SPEC, type PrecisionMode, type Vec3 } from '../model/spec';
import {
  DEFAULT_FOV_DEG,
  MAX_CAMERA_DISTANCE,
  MIN_CAMERA_DISTANCE,
  type CameraPose,
} from '../model/state';
import {
  aabbCorners,
  computeCameraBasis,
  dot3,
  type AABB,
  type ViewportMetrics,
} from '../render/camera';
import {
  LINE_INSTANCE_FLOATS,
  LINE_STYLE_GUIDE,
  pushFreeGuideLine,
} from '../render/linePacking';
import type { PickBox, PickHit } from '../render/pick';
import { FlowRibbonBuilder, type FlowRibbonColor } from '../render/ribbons';
import {
  FRAME_XY_AXES,
  FRAME_XZ_AXES,
  FRAME_YZ_AXES,
  localToWorldOnFrame,
  makeSheetFrame3D,
  sheetFrameWorldAabb,
  type SheetBasisAxes,
  type SheetFrame3D,
} from '../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_DIMMED,
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../render/sheets';
import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
} from './atlasPacker';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../ui/overlay';
import type { CutawayPartInfo, CutawaySceneGeometry } from './cutaways';
import {
  CELL_OP_A_FILL,
  CELL_OP_A_FRAME,
  CELL_OP_B_FILL,
  CELL_OP_B_FRAME,
  CELL_TARGET_FILL,
  CELL_TARGET_FRAME,
  computeSheetGroupBounds,
  createSheetFocusSet,
  drawCellHighlight3D,
  drawOperandDependency3D,
  EMPTY_SHEET_FOCUS_SET,
  fmtSigned,
  resolveSheetHighlight,
  RIBBON_ACTIVE,
  RIBBON_DIMMED,
  RIBBON_PATHWAY_SOFT,
  RIBBON_QUIET,
  RIBBON_RESIDUAL_ACTIVE,
  RIBBON_SSM_LOOP,
  spanFromSheetGrid,
  type LayerFormulaReadout,
  type SheetFocusEntry,
  type SheetFocusSet,
  type SheetGeomRecord,
  type StepperStepDef,
  type WorldSpan3D,
} from './sheetInspector';
import {
  CELL_ACTIVATION_GREEN,
  CELL_MOE_AMBER,
  CELL_WEIGHT_BLUE,
  SOLID_INSTANCE_FLOATS,
  type DimensionCaliper,
  type LandmarkAnchor,
} from './spine';

export type OutputTensorKey =
  | 'residual_in'
  | 'final_norm'
  | 'w_lm_head'
  | 'logits'
  | 'probs';

export type OutputAuxTensorKey = OutputTensorKey | 'rms_agg' | 'gamma' | 'softmax_agg';

export interface OutputCellRef {
  readonly tensor: OutputTensorKey;
  /** Logical sheet row: token for residual_in/final_norm; vocabId for w_lm_head/logits/probs. */
  readonly row: number;
  /** Logical sheet col: channel for residual_in/final_norm/w_lm_head; 0 for logits/probs. */
  readonly col: number;
  readonly tokenIndex?: number;
  readonly vocabId?: number;
}

export interface OutputSheetSpan {
  readonly tensor: OutputAuxTensorKey;
  readonly rowStart: number;
  readonly rowEnd: number;
  readonly colStart: number;
  readonly colEnd: number;
  readonly role: 'target' | 'operand_a' | 'operand_b';
}

export interface OutputCellFormula extends LayerFormulaReadout {
  readonly cell: OutputCellRef;
  readonly isMasked: false;
  readonly targetSpan: OutputSheetSpan;
  readonly upstreamSpans: readonly OutputSheetSpan[];
}

export interface OutputFormulaInput {
  readonly model: HybridModelParameters;
  readonly residualIn: Float32Array;
  readonly finalNorm: HybridFinalNormResult;
  readonly lmHead: HybridLmHeadResult;
  readonly tokenLabels: readonly string[];
  readonly startTokenOffset?: number;
  readonly focusToken?: number;
  readonly isArDecode?: boolean;
}

export interface BuildOutputCutawayOptions extends OutputFormulaInput {
  readonly precision: PrecisionMode;
  readonly selectedPart: string | null;
  readonly hoveredPart: string | null;
  readonly activeCell?: OutputCellRef | null;
  readonly selectedCandidateTokenId?: number | null;
}

export interface SpatialOutputFitOptions {
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly isNarrow?: boolean;
  readonly azimuthDeg?: number;
  readonly elevationDeg?: number;
}

/**
 * Invariant 3D bounding envelope for the Dedicated Output & Untied LM Head Cutaway
 * across T = 1..16, Full Seq & AR Decode, BF16 / FP8 Selective, and all Top-5 selections.
 */
export const OUTPUT_SPATIAL_FIT_BOUNDS: AABB = {
  min: [-8.35, -0.06, -1.95],
  max: [7.65, 3.40, 1.85],
};

export const OUTPUT_STEPPER_STEPS: readonly StepperStepDef[] = Object.freeze([
  {
    index: 0,
    shortLabel: 'Bay 52 Y',
    title: '1. Bay 52 Residual Stream',
    summary: 'Final residual output Y₅₂ exits Bay 52 MoE along the emerald Residual Highway.',
  },
  {
    index: 1,
    shortLabel: 'Final Norm',
    title: '2. Final RMSNorm (ε=1e−5) + γ',
    summary: 'Per-token RMS normalization (ε = 1e−5) followed by learned per-channel scale γ.',
  },
  {
    index: 2,
    shortLabel: 'W_lm_head',
    title: '3. Untied LM Head Projection',
    summary: 'Independent W_lm_head [32 × 8] mini (131,072 × 2,688 prod, ≠ W_embed) projects U_final into vocabulary space.',
  },
  {
    index: 3,
    shortLabel: 'Logits',
    title: '4. Vocabulary Logits [32 × 1]',
    summary: 'Linear dot products z[v] = W_lm_head[v, :] · U_final[:, t] across all V vocabulary entries.',
  },
  {
    index: 4,
    shortLabel: 'Softmax',
    title: '5. Softmax Agg → Probs [32 × 1]',
    summary: 'Numerically stable softmax subtracts max(z) and divides by Σexp(z − max) to form next-token probabilities.',
  },
  {
    index: 5,
    shortLabel: 'Top-5 / AR',
    title: '6. Top-5 Candidates & AR Loop',
    summary: 'Top-5 candidate bars elevate cleanly for inspection or interactive AR next-token selection.',
  },
]);

/**
 * Exact 8-corner projective camera fitter for the 3D Output & Untied LM Head Cutaway.
 * Keeps all 8 corners of `bounds` strictly inside the unobstructed viewport band.
 */
export function fitSpatialOutputCamera(
  metrics: ViewportMetrics,
  bounds: AABB = OUTPUT_SPATIAL_FIT_BOUNDS,
  options?: SpatialOutputFitOptions,
): CameraPose {
  const isNarrow = options?.isNarrow ?? metrics.innerWidth < 560;
  const topInset = Math.max(0, options?.topInsetPx ?? (isNarrow ? 12 : 84));
  const bottomInset = Math.max(0, options?.bottomInsetPx ?? (isNarrow ? 12 : 118));
  const azimuthDeg = options?.azimuthDeg ?? (isNarrow ? 16 : 19);
  const elevationDeg = options?.elevationDeg ?? (isNarrow ? 22 : 20.5);
  const fovDeg = DEFAULT_FOV_DEG;
  const fillX = isNarrow ? 0.94 : 0.95;
  const fillY = isNarrow ? 0.90 : 0.92;

  const basis = computeCameraBasis(azimuthDeg, elevationDeg);
  const corners = aabbCorners(bounds);

  let minR = Infinity, maxR = -Infinity;
  let minU = Infinity, maxU = -Infinity;
  let minE = Infinity, maxE = -Infinity;

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

/** Parse `'output-cand-<vocabId>'` pick part ID into integer vocabId, or null if not a candidate bar. */
export function parseOutputCandidatePartId(partId: string | null | undefined): number | null {
  if (!partId || !partId.startsWith('output-cand-')) return null;
  const parsed = Number.parseInt(partId.slice('output-cand-'.length), 10);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

interface SoftmaxSummary {
  readonly maxLogit: number;
  readonly sumExp: number;
  readonly ranks: Int32Array; // 1-based rank per vocabId
}

function computeSoftmaxSummary(
  logits: Float32Array,
  vocabSize: number,
  T: number,
  tok: number,
): SoftmaxSummary {
  let maxLogit = -Infinity;
  const order: number[] = [];
  for (let v = 0; v < vocabSize; v++) {
    const lv = logits[v * T + tok];
    if (lv > maxLogit) maxLogit = lv;
    order.push(v);
  }
  let sumExp = 0;
  for (let v = 0; v < vocabSize; v++) {
    sumExp += Math.exp(logits[v * T + tok] - maxLogit);
  }
  order.sort((a, b) => {
    const la = logits[a * T + tok];
    const lb = logits[b * T + tok];
    return lb !== la ? lb - la : a - b;
  });
  const ranks = new Int32Array(vocabSize);
  for (let r = 0; r < order.length; r++) {
    ranks[order[r]] = r + 1;
  }
  return { maxLogit, sumExp, ranks };
}

interface OutputValueAtlas extends Rg32fValueAtlas {
  readonly residualIn: AtlasSlice;
  readonly finalNorm: AtlasSlice;
  readonly rmsAgg: AtlasSlice;
  readonly gamma: AtlasSlice;
  readonly wLmHead: AtlasSlice;
  readonly logits: AtlasSlice;
  readonly softmaxAgg: AtlasSlice;
  readonly probs: AtlasSlice;
  readonly probSingleRow: (vocabId: number) => AtlasSlice;
}

const OUT_ATLAS_W = 32;
const OUT_ATLAS_H = 64;

/**
 * Pack every live Float32 tensor for the Output & Untied LM Head cutaway into a single
 * 32×64 RG32F WebGL2 texture atlas (zero fake textures, bit-exact Float32):
 *   y =  0..31, x =  0..7  : W_lm_head [32 × 8]     (row = vocabId, col = ch)
 *   y =  0..31, x =  8     : Logits [32 × 1]        (row = vocabId, col = 0)
 *   y =  0..31, x =  9     : Probs [32 × 1]         (row = vocabId, col = 0, mode = 1)
 *   y =  0..1,  x = 10     : Softmax Agg [2 × 1]    (row 0 = maxLogit, row 1 = sumExp)
 *   y = 32..32+T-1, x=0..7 : Bay 52 Residual Y [T×8](row = t, col = ch)
 *   y = 32..32+T-1, x=8..15: Final Norm U_final[T×8](row = t, col = ch)
 *   y = 32..32+T-1, x=16   : RMS Agg [T × 1]        (row = t, col = 0)
 *   y = 48,     x =  0..7  : Final Norm γ [1 × 8]   (row = 0, col = ch)
 */
function buildOutputValueAtlas(
  model: HybridModelParameters,
  residualIn: Float32Array,
  finalNorm: HybridFinalNormResult,
  lmHead: HybridLmHeadResult,
  T: number,
  focusTok: number,
  smSummary: SoftmaxSummary,
): OutputValueAtlas {
  const d = model.config.d;
  const vocabSize = model.config.vocabSize;
  const writer = createRg32fAtlas(OUT_ATLAS_W, OUT_ATLAS_H);
  const { data, setTexel, makeRect } = writer;

  // 1. Untied W_lm_head [32, 8], Logits [32, 1], Probs [32, 1]
  let maxAbsLogit = 0.5;
  for (let v = 0; v < vocabSize; v++) {
    const wOff = v * d;
    for (let ch = 0; ch < d; ch++) {
      setTexel(ch, v, lmHead.W_lm_head[wOff + ch]);
    }
    const lv = lmHead.logits[v * T + focusTok];
    const pv = lmHead.probs[v * T + focusTok];
    if (Number.isFinite(lv) && Math.abs(lv) > maxAbsLogit) maxAbsLogit = Math.abs(lv);
    setTexel(8, v, lv);
    setTexel(9, v, pv);
  }

  // 2. Softmax Agg [2, 1]: row 0 = maxLogit, row 1 = sumExp
  setTexel(10, 0, Math.fround(smSummary.maxLogit));
  setTexel(10, 1, Math.fround(smSummary.sumExp));
  const scaleSmAgg = 1 / Math.max(1.0, Math.abs(smSummary.maxLogit), Math.abs(smSummary.sumExp));

  // 3. Bay 52 Residual Y [T, 8], Final RMSNorm U_final [T, 8], RMS Agg [T, 1]
  for (let t = 0; t < T; t++) {
    for (let ch = 0; ch < d; ch++) {
      setTexel(ch, 32 + t, residualIn[ch * T + t]);
      setTexel(8 + ch, 32 + t, finalNorm.U_final[ch * T + t]);
    }
    setTexel(16, 32 + t, finalNorm.rms[t]);
  }

  // 4. Learned Final RMSNorm scale γ [1, 8]
  for (let ch = 0; ch < d; ch++) {
    setTexel(ch, 48, finalNorm.gamma[ch]);
  }

  const scaleRes = 1 / computeMaxAbs(residualIn, 1.0);
  const scaleNorm = 1 / computeMaxAbs(finalNorm.U_final, 1.0);
  const scaleRms = 1 / computeMaxAbs(finalNorm.rms, 1.0);
  const scaleGamma = 1 / computeMaxAbs(finalNorm.gamma, 1.0);
  const scaleWLm = 1 / computeMaxAbs(lmHead.W_lm_head, 0.5);
  const scaleLogits = 1 / Math.max(maxAbsLogit, 1.0);

  return {
    width: OUT_ATLAS_W,
    height: OUT_ATLAS_H,
    data,
    residualIn: { rect: makeRect(0, 32, d, T), scale: scaleRes, mode: 0 },
    finalNorm: { rect: makeRect(8, 32, d, T), scale: scaleNorm, mode: 0 },
    rmsAgg: { rect: makeRect(16, 32, 1, T), scale: scaleRms, mode: 0 },
    gamma: { rect: makeRect(0, 48, d, 1), scale: scaleGamma, mode: 0 },
    wLmHead: { rect: makeRect(0, 0, d, vocabSize), scale: scaleWLm, mode: 0 },
    logits: { rect: makeRect(8, 0, 1, vocabSize), scale: scaleLogits, mode: 0 },
    softmaxAgg: { rect: makeRect(10, 0, 1, 2), scale: scaleSmAgg, mode: 0 },
    probs: { rect: makeRect(9, 0, 1, vocabSize), scale: 1.0, mode: 1 },
    probSingleRow: (vocabId: number): AtlasSlice => ({
      rect: makeRect(9, Math.max(0, Math.min(vocabSize - 1, vocabId)), 1, 1),
      scale: 1.0,
      mode: 1,
    }),
  };
}

/**
 * Exact mathematical formula & operand provenance for any cell in the Output & Untied LM Head cutaway.
 */
export function describeOutputCellFormula(
  input: OutputFormulaInput,
  cell: OutputCellRef,
): OutputCellFormula {
  const { model, residualIn, finalNorm, lmHead, tokenLabels } = input;
  const d = model.config.d;
  const vocabSize = model.config.vocabSize;
  const T = Math.max(1, Math.min(16, tokenLabels.length));
  const offset = input.startTokenOffset ?? 0;
  const epsLabel = formatEpsLabel(finalNorm.eps);
  const defaultFocusT = Math.max(0, Math.min(T - 1, Math.trunc(input.focusToken ?? T - 1)));

  const clampT = (t: number) => Math.max(0, Math.min(T - 1, Math.trunc(t)));
  const clampV = (v: number) => Math.max(0, Math.min(vocabSize - 1, Math.trunc(v)));
  const clampCh = (c: number) => Math.max(0, Math.min(d - 1, Math.trunc(c)));
  const labelForT = (t: number) => tokenLabels[t] ?? `t${offset + t}`;
  const labelForV = (v: number) => model.vocab[v] ?? `id:${v}`;

  if (cell.tensor === 'residual_in') {
    const t = clampT(cell.tokenIndex ?? cell.row);
    const ch = clampCh(cell.col);
    const absT = offset + t;
    const word = labelForT(t);
    const val = residualIn[ch * T + t];
    const rms = finalNorm.rms[t];
    return {
      cell: { tensor: 'residual_in', row: t, col: ch, tokenIndex: t },
      tensorLabel: 'Bay 52 Residual Output Y',
      coordLabel: `Y₅₂[t${absT} "${word}", ch=${ch}]`,
      headBadge: `Bay 52 Y → Final RMSNorm · d=${d} mini (${NEMOTRON_SPEC.hidden.toLocaleString()} prod)`,
      value: val,
      isMasked: false,
      valueText: fmtSigned(val),
      equation: `Y₅₂[ch${ch}, t${absT}] = ${fmtSigned(val)} (exits Bay 52 MoE → Final RMSNorm)`,
      subEquation: `Per-token rms(t${absT}) = ${rms.toFixed(4)} (ε = ${epsLabel}) · Bit-exact Bay 52 residual stream`,
      targetSpan: { tensor: 'residual_in', rowStart: t, rowEnd: t, colStart: ch, colEnd: ch, role: 'target' },
      upstreamSpans: [],
    };
  }

  if (cell.tensor === 'final_norm') {
    const t = clampT(cell.tokenIndex ?? cell.row);
    const ch = clampCh(cell.col);
    const absT = offset + t;
    const word = labelForT(t);
    const xVal = residualIn[ch * T + t];
    const rms = finalNorm.rms[t];
    const gVal = finalNorm.gamma[ch];
    const uVal = finalNorm.U_final[ch * T + t];
    return {
      cell: { tensor: 'final_norm', row: t, col: ch, tokenIndex: t },
      tensorLabel: 'Final RMSNorm U_final',
      coordLabel: `U_final[t${absT} "${word}", ch=${ch}]`,
      headBadge: `Final RMSNorm · ε = ${epsLabel}`,
      value: uVal,
      isMasked: false,
      valueText: fmtSigned(uVal),
      equation: `U_final[ch${ch}, t${absT}] = (${fmtSigned(xVal)} / ${rms.toFixed(3)}) · γ[${fmtSigned(gVal)}] = ${fmtSigned(uVal)}`,
      subEquation: `rms(t${absT}) = √(mean(Y₅₂²) + ${epsLabel}) = ${rms.toFixed(4)} · learned γ[${ch}] = ${fmtSigned(gVal)}`,
      targetSpan: { tensor: 'final_norm', rowStart: t, rowEnd: t, colStart: ch, colEnd: ch, role: 'target' },
      upstreamSpans: [
        { tensor: 'residual_in', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_a' },
        { tensor: 'gamma', rowStart: 0, rowEnd: 0, colStart: ch, colEnd: ch, role: 'operand_b' },
      ],
    };
  }

  if (cell.tensor === 'w_lm_head') {
    const vid = clampV(cell.vocabId ?? cell.row);
    const ch = clampCh(cell.col);
    const word = labelForV(vid);
    const wVal = lmHead.W_lm_head[vid * d + ch];
    const embVal = model.W_embed[vid * d + ch];
    return {
      cell: { tensor: 'w_lm_head', row: vid, col: ch, vocabId: vid },
      tensorLabel: 'Untied LM Head W_lm_head',
      coordLabel: `W_lm_head[v=${vid} "${word}", ch=${ch}]`,
      headBadge: `Untied Weights ≠ W_embed · [${vocabSize}×${d}] mini · [131,072×2,688] prod`,
      value: wVal,
      isMasked: false,
      valueText: fmtSigned(wVal),
      equation: `W_lm_head[id=${vid} ("${word}"), ch${ch}] = ${fmtSigned(wVal)} (≠ W_embed = ${fmtSigned(embVal)})`,
      subEquation: 'Independent output head (tie_word_embeddings = false, no bias) · Synthetic V=32 mini (131,072 prod)',
      targetSpan: { tensor: 'w_lm_head', rowStart: vid, rowEnd: vid, colStart: ch, colEnd: ch, role: 'target' },
      upstreamSpans: [],
    };
  }

  if (cell.tensor === 'logits') {
    const vid = clampV(cell.vocabId ?? cell.row);
    const t = clampT(cell.tokenIndex ?? defaultFocusT);
    const absT = offset + t;
    const tokWord = labelForT(t);
    const candWord = labelForV(vid);
    const lv = lmHead.logits[vid * T + t];

    const termParts: string[] = [];
    for (let i = 0; i < Math.min(3, d); i++) {
      termParts.push(
        `(${fmtSigned(lmHead.W_lm_head[vid * d + i], 2)}·${fmtSigned(finalNorm.U_final[i * T + t], 2)})`,
      );
    }
    const preview = `${termParts.join(' + ')} + … (${d} terms)`;

    return {
      cell: { tensor: 'logits', row: vid, col: 0, tokenIndex: t, vocabId: vid },
      tensorLabel: 'Vocabulary Logits',
      coordLabel: `logit[v=${vid} "${candWord}" | t${absT} "${tokWord}"]`,
      headBadge: `z = W_lm_head · U_final · V=${vocabSize} mini (${NEMOTRON_SPEC.vocabSize.toLocaleString()} prod)`,
      value: lv,
      isMasked: false,
      valueText: fmtSigned(lv),
      equation: `z["${candWord}" (#${vid}), t${absT}] = Σ_{i=0..${d - 1}} W_lm_head[${vid}, i] · U_final[i, t${absT}] = ${fmtSigned(lv)}`,
      subEquation: `${preview} = ${fmtSigned(lv)} · Synthetic Float32 mini (not 31.6B checkpoint)`,
      targetSpan: { tensor: 'logits', rowStart: vid, rowEnd: vid, colStart: 0, colEnd: 0, role: 'target' },
      upstreamSpans: [
        { tensor: 'final_norm', rowStart: t, rowEnd: t, colStart: 0, colEnd: d - 1, role: 'operand_a' },
        { tensor: 'w_lm_head', rowStart: vid, rowEnd: vid, colStart: 0, colEnd: d - 1, role: 'operand_b' },
      ],
    };
  }

  // tensor === 'probs'
  const vid = clampV(cell.vocabId ?? cell.row);
  const t = clampT(cell.tokenIndex ?? defaultFocusT);
  const absT = offset + t;
  const tokWord = labelForT(t);
  const candWord = labelForV(vid);
  const lv = lmHead.logits[vid * T + t];
  const pv = lmHead.probs[vid * T + t];
  const sm = computeSoftmaxSummary(lmHead.logits, vocabSize, T, t);
  const rank = sm.ranks[vid];

  return {
    cell: { tensor: 'probs', row: vid, col: 0, tokenIndex: t, vocabId: vid },
    tensorLabel: 'Next-Token Softmax Probability',
    coordLabel: `P(next="${candWord}" #${vid} | t${absT} "${tokWord}")`,
    headBadge: `Rank #${rank} · ${(pv * 100).toFixed(1)}% · V=${vocabSize} mini (${NEMOTRON_SPEC.vocabSize.toLocaleString()} prod)`,
    value: pv,
    isMasked: false,
    valueText: `${(pv * 100).toFixed(2)}% (${pv.toFixed(4)})`,
    equation: `P("${candWord}" | t${absT}) = exp(${fmtSigned(lv)} − ${fmtSigned(sm.maxLogit)}) / Σexp(${sm.sumExp.toFixed(3)}) = ${pv.toFixed(4)}`,
    subEquation:
      `Stable softmax over V=${vocabSize} synthetic vocab (max=${fmtSigned(sm.maxLogit)}, Σexp=${sm.sumExp.toFixed(3)})` +
      (input.isArDecode ? ' · Click a Top-5 bar to stage next AR token' : ''),
    targetSpan: { tensor: 'probs', rowStart: vid, rowEnd: vid, colStart: 0, colEnd: 0, role: 'target' },
    upstreamSpans: [
      { tensor: 'logits', rowStart: vid, rowEnd: vid, colStart: 0, colEnd: 0, role: 'operand_a' },
      { tensor: 'softmax_agg', rowStart: 0, rowEnd: 1, colStart: 0, colEnd: 0, role: 'operand_b' },
    ],
  };
}

/** Resolve a ray-cast PickHit on the Output & Untied LM Head cutaway into an exact `OutputCellRef`. */
export function pickHitToOutputCell(
  hit: PickHit | null,
  focusToken: number,
): OutputCellRef | null {
  if (!hit) return null;
  const candId = parseOutputCandidatePartId(hit.partId);
  if (candId !== null) {
    return {
      tensor: 'probs',
      row: candId,
      col: 0,
      tokenIndex: hit.tokenIndex ?? focusToken,
      vocabId: candId,
    };
  }
  if (!hit.tensorKey || hit.hitRow === undefined || hit.hitCol === undefined) {
    return null;
  }
  if (hit.tensorKey === 'residual_in' || hit.tensorKey === 'final_norm') {
    return {
      tensor: hit.tensorKey,
      row: hit.hitRow,
      col: hit.hitCol,
      tokenIndex: hit.hitRow,
    };
  }
  if (hit.tensorKey === 'w_lm_head') {
    return {
      tensor: 'w_lm_head',
      row: hit.hitRow,
      col: hit.hitCol,
      vocabId: hit.hitRow,
    };
  }
  if (hit.tensorKey === 'logits' || hit.tensorKey === 'probs') {
    return {
      tensor: hit.tensorKey,
      row: hit.hitRow,
      col: 0,
      tokenIndex: hit.tokenIndex ?? focusToken,
      vocabId: hit.hitRow,
    };
  }
  return null;
}

interface OutputSheetSpec {
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
  readonly headIndex?: number;
  readonly tensorKey?: string;
  readonly tokenIndex?: number;
  readonly atlasSlice?: AtlasSlice;
  readonly registryKey?: string;
  readonly cellFamily: number;
}

class OutputSceneBuilder {
  readonly sheetFloats: number[] = [];
  readonly basisFloats: number[] = [];
  readonly solidFloats: number[] = [];
  readonly lineFloats: number[] = [];
  readonly pickBoxes: PickBox[] = [];
  readonly landmarks: LandmarkAnchor[] = [];
  readonly calipers: DimensionCaliper[] = [];
  readonly flow = new FlowRibbonBuilder();
  readonly sheetRegistry = new Map<string, SheetGeomRecord>();

  addSheet(s: OutputSheetSpec): SheetFrame3D {
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

    this.solidFloats.push(
      s.cx, s.cy, s.cz,
      s.ex, s.ey, s.ez,
      s.cellFamily,
      s.fp8Flag,
      s.seed,
      53, // Output bookend bayId = 53 (immediately after Bay 52)
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
      bayId: 53,
      partId: s.partId,
      headIndex: s.headIndex,
      tensorKey: s.tensorKey,
      gridCols: s.tensorKey ? s.gridCols : undefined,
      gridRows: s.tensorKey ? s.gridRows : undefined,
      tokenIndex: s.tokenIndex,
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

function buildOutputFocusSet(formula: OutputCellFormula | null): SheetFocusSet {
  if (!formula) return EMPTY_SHEET_FOCUS_SET;
  const toEntry = (span: OutputSheetSpan): SheetFocusEntry => ({
    tensor: span.tensor,
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

const RIBBON_CAND_LINK: FlowRibbonColor = { r: 0.12, g: 0.56, b: 0.44, a: 0.58, borderAlpha: 0.30 };
const RIBBON_CAND_ACTIVE: FlowRibbonColor = { r: 0.863, g: 0.298, b: 0.204, a: 0.84, borderAlpha: 0.44 };
const RIBBON_NORM_FLOW: FlowRibbonColor = { r: 0.68, g: 0.50, b: 0.20, a: 0.52, borderAlpha: 0.26 };

function buildOutputPartCatalog(
  d: number,
  vocabSize: number,
  epsLabel: string,
  absFocusT: number,
  focusWord: string,
  top5: readonly HybridPrediction[],
  smSummary: SoftmaxSummary,
  precision: PrecisionMode,
  isArDecode: boolean,
): Record<string, CutawayPartInfo> {
  const isFp8 = precision === 'fp8_selective';
  const top1 = top5[0];
  const catalog: Record<string, CutawayPartInfo> = {
    'residual-in': {
      id: 'residual-in',
      title: 'Bay 52 Residual Output (Y₅₂)',
      badge: `Emerald Residual Highway · t${absFocusT} "${focusWord}"`,
      summary:
        'Final residual state exiting Bay 52 (the 23rd MoE layer) after traversing all 52 hybrid layers. Enters Final RMSNorm before next-token projection.',
      specs: [
        `Hidden width: d = ${d} miniature (${NEMOTRON_SPEC.hidden.toLocaleString()} production)`,
        `Focus token: t${absFocusT} "${focusWord}"`,
        'Bit-exact continuity with Bay 52 MoE Residual Output Y',
      ],
    },
    'rms-agg': {
      id: 'rms-agg',
      title: 'Final RMS Aggregator',
      badge: `ε = ${epsLabel}`,
      summary:
        'Computes per-token root-mean-square denominator √(mean(Y₅₂²) + ε) across the hidden dimension prior to channel rescaling.',
      specs: [
        `RMSNorm epsilon: ε = ${epsLabel} (NemotronHRMSNorm)`,
        'One scalar RMS denominator per token position',
      ],
    },
    'final-gamma': {
      id: 'final-gamma',
      title: 'Final RMSNorm Learned Scale (γ)',
      badge: isFp8 ? 'BF16 Preserved · [1 × d]' : 'BF16 · [1 × d]',
      summary:
        'Learned non-unit per-channel weight vector γ multiplying the normalized hidden activations before the untied LM head.',
      specs: [
        `Shape: [${d}] miniature ([${NEMOTRON_SPEC.hidden.toLocaleString()}] production)`,
        'Zero additive bias (RMSNorm, not LayerNorm)',
      ],
    },
    'final-norm': {
      id: 'final-norm',
      title: `Final RMSNorm Output (U_final)`,
      badge: `ε = ${epsLabel} · Pre-LM Head`,
      summary:
        'Normalized, γ-scaled hidden state ready for linear projection into vocabulary logits.',
      specs: [
        `Formula: U_final = (Y₅₂ / √(mean(Y₅₂²) + ${epsLabel})) · γ`,
        `Focus token: t${absFocusT} "${focusWord}"`,
      ],
    },
    'w-lm-head': {
      id: 'w-lm-head',
      title: 'Untied LM Head Projection (W_lm_head)',
      badge: isFp8 ? 'Untied ≠ W_embed · BF16' : 'Untied ≠ W_embed',
      summary:
        'Strictly untied vocabulary projection matrix (tie_word_embeddings: false). Keeps output classification weights completely independent from input W_embed.',
      specs: [
        `Production shape: ${NEMOTRON_SPEC.hidden.toLocaleString()} → ${NEMOTRON_SPEC.vocabSize.toLocaleString()} (352.3M params)`,
        `Miniature Float32 table: ${vocabSize} × ${d} (synthetic, ≠ W_embed)`,
        'No additive bias term',
      ],
    },
    'logits': {
      id: 'logits',
      title: `Vocabulary Logits [${vocabSize} × 1]`,
      badge: `Focus t${absFocusT} · V=${vocabSize} mini (131,072 prod)`,
      summary:
        'Unnormalized vocabulary scores computed as z[v] = W_lm_head[v, :] · U_final[:, t] for the active token position.',
      specs: [
        `Max logit: ${fmtSigned(smSummary.maxLogit)} ("${top1?.label ?? '?'}")`,
        `Synthetic miniature vocabulary (V = ${vocabSize} · 131,072 prod)`,
      ],
    },
    'softmax-agg': {
      id: 'softmax-agg',
      title: 'Softmax Aggregator (max, Σexp)',
      badge: `max = ${fmtSigned(smSummary.maxLogit, 2)} · Σexp = ${smSummary.sumExp.toFixed(2)}`,
      summary:
        'Computes the shift-invariant maximum logit and partition sum Σexp(z − max) for numerically stable vocabulary softmax.',
      specs: [
        `max(z) = ${fmtSigned(smSummary.maxLogit, 4)}`,
        `Σexp(z − max) = ${smSummary.sumExp.toFixed(4)}`,
      ],
    },
    'probs': {
      id: 'probs',
      title: `Next-Token Probabilities [${vocabSize} × 1]`,
      badge: `Top-1: "${top1?.label ?? '?'}" (${((top1?.prob ?? 0) * 100).toFixed(1)}%)`,
      summary:
        'Normalized probability distribution over the vocabulary. The Top-5 candidate rows elevate into clickable 3D bars on the right.',
      specs: [
        `Greedy argmax: "${top1?.label ?? '?'}" (#${top1?.tokenId ?? 0}) · ${((top1?.prob ?? 0) * 100).toFixed(2)}%`,
        'V=32 synthetic Float32 mini · 131,072 prod (not 31.6B checkpoint logits)',
        isArDecode
          ? 'AR Decode active: click any Top-5 bar to stage/feed that candidate into the next step'
          : 'Switch to AR Decode to feed a Top-5 candidate back through all 52 layers',
      ],
    },
  };

  for (let rank = 0; rank < top5.length; rank++) {
    const c = top5[rank];
    const partId = `output-cand-${c.tokenId}`;
    catalog[partId] = {
      id: partId,
      title: `Top-5 Candidate #${rank + 1}: "${c.label}"`,
      badge: `id #${c.tokenId} · ${(c.prob * 100).toFixed(1)}% · z=${fmtSigned(c.logit, 2)}`,
      summary: isArDecode
        ? `Ranked #${rank + 1} synthetic continuation for t${absFocusT}. Clicking stages "${c.label}" (#${c.tokenId}) for the next cached AR decode step.`
        : `Ranked #${rank + 1} synthetic continuation for t${absFocusT} from the untied miniature LM head (V=32 mini · 131,072 prod).`,
      specs: [
        `Logit z[#${c.tokenId}] = ${fmtSigned(c.logit, 4)}`,
        `Probability P("${c.label}") = ${(c.prob * 100).toFixed(2)}% (${c.prob.toFixed(4)})`,
        rank === 0 ? 'Greedy argmax candidate' : `Alternative branch candidate (rank #${rank + 1})`,
      ],
    };
  }

  return catalog;
}

/**
 * Build the Dedicated 3D Thin-Sheet Output & Untied LM Head Cutaway geometry,
 * backed 100% by live Float32 `residualIn` (Bay 52 Y), `finalNorm`, and `lmHead`
 * tensors in BOTH Full-Sequence (`T=8`) and AR Decode (`S=1`) modes.
 */
export function buildOutputCutaway(
  opts: BuildOutputCutawayOptions,
): CutawaySceneGeometry {
  const {
    model,
    residualIn,
    finalNorm,
    lmHead,
    tokenLabels,
    precision,
    selectedPart,
    hoveredPart,
    activeCell,
    isArDecode = false,
    selectedCandidateTokenId = null,
  } = opts;

  const T = Math.max(1, Math.min(16, tokenLabels.length));
  const d = model.config.d;
  const vocabSize = model.config.vocabSize;
  const offset = opts.startTokenOffset ?? 0;
  const focusTok = Math.max(0, Math.min(T - 1, Math.trunc(opts.focusToken ?? T - 1)));
  const absFocusT = offset + focusTok;
  const focusWord = tokenLabels[focusTok] ?? `t${absFocusT}`;
  const epsLabel = formatEpsLabel(finalNorm.eps);
  const isFp8 = precision === 'fp8_selective';

  const top5: readonly HybridPrediction[] = lmHead.topKByToken[focusTok] ?? [];
  const top5Set = new Set<number>(top5.map((c) => c.tokenId));
  const smSummary = computeSoftmaxSummary(lmHead.logits, vocabSize, T, focusTok);

  const activeFormula = activeCell
    ? describeOutputCellFormula(opts, activeCell)
    : null;
  const focusSet = buildOutputFocusSet(activeFormula);
  const hasCellTrace = focusSet.active;
  const activePart = hoveredPart ?? selectedPart;
  const hasFocus = activePart !== null || hasCellTrace;

  const atlas = buildOutputValueAtlas(model, residualIn, finalNorm, lmHead, T, focusTok, smSummary);
  const b = new OutputSceneBuilder();

  const SHEET_THICK = 0.058;
  const BASE_Y = 0.12;
  const WALL_Y = 1.56;
  const WALL_H = 2.76; // y = 0.18 .. 2.94 for 32 vocab rows

  // Left -> right +X coordinates
  const xResIn = -7.25;
  const xRmsAgg = -5.82;
  const xFinalNorm = -4.35;
  const xWLm = -1.75;
  const xLogits = 0.70;
  const xSoftAgg = 2.05;
  const xProbs = 3.40;
  const xTop5Left = 5.10;

  const ioW = 1.24;
  const ioD = Math.max(0.62, Math.min(2.15, 0.24 * T + 0.34));

  const stageHl = (tensor: OutputAuxTensorKey, partId: string, onPath: boolean, goldIfFp8 = false): number => {
    const fallback =
      activePart === partId
        ? SHEET_HL_ACTIVE
        : onPath && hasFocus
          ? SHEET_HL_PATHWAY
          : hasFocus
            ? SHEET_HL_DIMMED
            : goldIfFp8 && isFp8
              ? SHEET_HL_BF16_GOLD
              : SHEET_HL_NORMAL;
    return resolveSheetHighlight(focusSet, { tensor }, fallback);
  };

  // -------------------------------------------------------------------------
  // 0. Far-left: Bay 52 Residual Y [T × 8] entering from emerald Residual Highway
  // -------------------------------------------------------------------------
  const resInHl = stageHl('residual_in', 'residual-in', activePart === 'final-norm' || activePart === 'rms-agg');
  b.addSheet({
    cx: xResIn,
    cy: BASE_Y,
    cz: 0,
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioD,
    axes: FRAME_XZ_AXES,
    gridCols: d,
    gridRows: T,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    fp8Flag: 0,
    highlight: resInHl,
    seed: 1.5 + T * 0.1,
    partId: 'residual-in',
    tensorKey: 'residual_in',
    tokenIndex: focusTok,
    atlasSlice: atlas.residualIn,
    registryKey: 'residual_in',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Incoming Residual Highway ribbon from Bay 52
  b.flow.addPolyline3D(
    [
      [-8.25, BASE_Y + 0.04, 0],
      [xResIn - ioW * 0.5 - 0.06, BASE_Y + 0.04, 0],
    ],
    [0, 1, 0],
    0.15,
    hasCellTrace && resInHl === SHEET_HL_DIMMED ? RIBBON_DIMMED : RIBBON_RESIDUAL_ACTIVE,
    true,
    0.22,
    0.24,
    true,
  );

  // -------------------------------------------------------------------------
  // 1. Step 1: Final RMSNorm (ε=1e−5) + RMS Agg strip + Upright γ scale bar
  // -------------------------------------------------------------------------
  const rmsAggHl = stageHl('rms_agg', 'rms-agg', activePart === 'final-norm' || activePart === 'residual-in');
  b.addSheet({
    cx: xRmsAgg,
    cy: BASE_Y,
    cz: 0,
    ex: 0.44,
    ey: SHEET_THICK,
    ez: ioD,
    axes: FRAME_XZ_AXES,
    gridCols: 1,
    gridRows: T,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: rmsAggHl,
    seed: 3.1,
    partId: 'rms-agg',
    tokenIndex: focusTok,
    atlasSlice: atlas.rmsAgg,
    registryKey: 'rms_agg',
    cellFamily: CELL_MOE_AMBER,
  });

  const gammaY = 1.24;
  const gammaZ = -0.35;
  const gammaHl = stageHl('gamma', 'final-gamma', activePart === 'final-norm', true);
  b.addSheet({
    cx: xFinalNorm,
    cy: gammaY,
    cz: gammaZ,
    ex: ioW,
    ey: SHEET_THICK,
    ez: 0.36,
    axes: FRAME_XY_AXES,
    gridCols: d,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: gammaHl,
    seed: 4.4,
    partId: 'final-gamma',
    atlasSlice: atlas.gamma,
    registryKey: 'gamma',
    cellFamily: CELL_MOE_AMBER,
  });

  const normHl = stageHl(
    'final_norm',
    'final-norm',
    activePart === 'w-lm-head' || activePart === 'logits' || activePart === 'rms-agg' || activePart === 'final-gamma',
    true,
  );
  const finalNormFrame = b.addSheet({
    cx: xFinalNorm,
    cy: BASE_Y,
    cz: 0,
    ex: ioW,
    ey: SHEET_THICK,
    ez: ioD,
    axes: FRAME_XZ_AXES,
    gridCols: d,
    gridRows: T,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: normHl,
    seed: 5.2 + T * 0.1,
    partId: 'final-norm',
    tensorKey: 'final_norm',
    tokenIndex: focusTok,
    atlasSlice: atlas.finalNorm,
    registryKey: 'final_norm',
    cellFamily: CELL_MOE_AMBER,
  });

  // Residual Y -> RMS Agg -> Final Norm + γ -> Final Norm connectors
  const normFlowCol = hasCellTrace
    ? RIBBON_DIMMED
    : activePart === 'final-norm' || activePart === 'rms-agg' || activePart === 'final-gamma'
      ? RIBBON_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_NORM_FLOW;

  b.flow.addPolyline3D(
    [
      [xResIn + ioW * 0.5 + 0.04, BASE_Y + 0.04, 0],
      [xRmsAgg - 0.22 - 0.04, BASE_Y + 0.04, 0],
    ],
    [0, 1, 0],
    0.09,
    normFlowCol,
    true,
    0.16,
    0.16,
    false,
  );
  b.flow.addPolyline3D(
    [
      [xRmsAgg + 0.22 + 0.04, BASE_Y + 0.04, 0],
      [xFinalNorm - ioW * 0.5 - 0.04, BASE_Y + 0.04, 0],
    ],
    [0, 1, 0],
    0.09,
    normFlowCol,
    true,
    0.16,
    0.16,
    false,
  );
  b.flow.addCubic3D(
    [xFinalNorm, gammaY - 0.20, gammaZ],
    [xFinalNorm, gammaY * 0.55, gammaZ * 0.6],
    [xFinalNorm, BASE_Y + 0.32, -0.08],
    [xFinalNorm, BASE_Y + 0.06, 0],
    [1, 0, 0],
    0.070,
    normFlowCol,
    true,
    14,
    true,
    0.16,
    0.16,
  );

  // -------------------------------------------------------------------------
  // 2. Step 2: Upright Untied W_lm_head [32 × 8] (FRAME_YZ_AXES)
  // -------------------------------------------------------------------------
  const wLmSpanZ = 1.65;
  const wLmHl = stageHl(
    'w_lm_head',
    'w-lm-head',
    activePart === 'logits' || activePart === 'final-norm',
    true,
  );
  b.addSheet({
    cx: xWLm,
    cy: WALL_Y,
    cz: 0,
    ex: wLmSpanZ,
    ey: 0.062,
    ez: WALL_H,
    axes: FRAME_YZ_AXES,
    gridCols: d,
    gridRows: vocabSize,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    fp8Flag: 0,
    highlight: wLmHl,
    seed: 8.6,
    partId: 'w-lm-head',
    tensorKey: 'w_lm_head',
    atlasSlice: atlas.wLmHead,
    registryKey: 'w_lm_head',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // Grounding stanchions under W_lm_head
  b.addGuideLine(xWLm, 0.0, -wLmSpanZ * 0.5, xWLm, WALL_Y - WALL_H * 0.5, -wLmSpanZ * 0.5, 0.9, LINE_STYLE_GUIDE, 0.48);
  b.addGuideLine(xWLm, 0.0, wLmSpanZ * 0.5, xWLm, WALL_Y - WALL_H * 0.5, wLmSpanZ * 0.5, 0.9, LINE_STYLE_GUIDE, 0.48);

  // Final Norm [focusTok] -> Upright W_lm_head
  const normFocusPt = localToWorldOnFrame(
    finalNormFrame,
    0.52,
    0.35,
    -0.5 + (focusTok + 0.5) / T,
  );
  b.flow.addCubic3D(
    normFocusPt,
    [xFinalNorm + 1.35, BASE_Y + 0.55, normFocusPt[2] * 0.55],
    [xWLm - 0.85, WALL_Y, 0],
    [xWLm - 0.06, WALL_Y, 0],
    [0, 1, 0],
    hasCellTrace ? 0.055 : 0.095,
    hasCellTrace
      ? RIBBON_DIMMED
      : activePart === 'w-lm-head' || activePart === 'final-norm'
        ? RIBBON_ACTIVE
        : hasFocus
          ? RIBBON_DIMMED
          : RIBBON_PATHWAY_SOFT,
    true,
    20,
    true,
    0.20,
    0.20,
  );

  // -------------------------------------------------------------------------
  // 3. Step 3: Logits [32 × 1] -> Softmax Agg (max, Σexp) -> Probs [32 × 1]
  // -------------------------------------------------------------------------
  const colW = 0.64;
  const logitsHl = stageHl(
    'logits',
    'logits',
    activePart === 'w-lm-head' || activePart === 'softmax-agg' || activePart === 'probs',
  );
  b.addSheet({
    cx: xLogits,
    cy: WALL_Y,
    cz: 0,
    ex: colW,
    ey: SHEET_THICK,
    ez: WALL_H,
    axes: FRAME_XY_AXES,
    gridCols: 1,
    gridRows: vocabSize,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_SCORES,
    fp8Flag: 0,
    highlight: logitsHl,
    seed: 11.4,
    partId: 'logits',
    tensorKey: 'logits',
    tokenIndex: focusTok,
    atlasSlice: atlas.logits,
    registryKey: 'logits',
    cellFamily: CELL_WEIGHT_BLUE,
  });

  // W_lm_head -> Logits
  b.flow.addCubic3D(
    [xWLm + 0.06, WALL_Y, 0],
    [xWLm + 0.85, WALL_Y, 0],
    [xLogits - colW * 0.5 - 0.55, WALL_Y, 0],
    [xLogits - colW * 0.5 - 0.05, WALL_Y, 0],
    [0, 1, 0],
    hasCellTrace ? 0.055 : 0.095,
    hasCellTrace
      ? RIBBON_DIMMED
      : activePart === 'w-lm-head' || activePart === 'logits'
        ? RIBBON_ACTIVE
        : hasFocus
          ? RIBBON_DIMMED
          : RIBBON_QUIET,
    true,
    16,
    true,
    0.18,
    0.18,
  );

  // Softmax Agg (max, Σexp)
  const smAggW = 0.56;
  const smAggH = 0.88;
  const smAggHl = stageHl('softmax_agg', 'softmax-agg', activePart === 'logits' || activePart === 'probs');
  b.addSheet({
    cx: xSoftAgg,
    cy: WALL_Y,
    cz: 0,
    ex: smAggW,
    ey: SHEET_THICK,
    ez: smAggH,
    axes: FRAME_XY_AXES,
    gridCols: 1,
    gridRows: 2,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    fp8Flag: 0,
    highlight: smAggHl,
    seed: 13.7,
    partId: 'softmax-agg',
    tokenIndex: focusTok,
    atlasSlice: atlas.softmaxAgg,
    registryKey: 'softmax_agg',
    cellFamily: CELL_MOE_AMBER,
  });

  // Probs [32 × 1] (probability palette flags.x = 1!)
  const probsHl = stageHl(
    'probs',
    'probs',
    activePart === 'softmax-agg' || parseOutputCandidatePartId(activePart) !== null,
  );
  const probsFrame = b.addSheet({
    cx: xProbs,
    cy: WALL_Y,
    cz: 0,
    ex: colW,
    ey: SHEET_THICK,
    ez: WALL_H,
    axes: FRAME_XY_AXES,
    gridCols: 1,
    gridRows: vocabSize,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_SOFTMAX,
    fp8Flag: 0,
    highlight: probsHl,
    seed: 16.2,
    partId: 'probs',
    tensorKey: 'probs',
    tokenIndex: focusTok,
    atlasSlice: atlas.probs, // mode = 1 -> flags.x = 1 probability emerald palette
    registryKey: 'probs',
    cellFamily: CELL_ACTIVATION_GREEN,
  });

  // Logits -> Softmax Agg -> Probs ribbons
  const smFlowCol = hasCellTrace
    ? RIBBON_DIMMED
    : activePart === 'logits' || activePart === 'softmax-agg' || activePart === 'probs'
      ? RIBBON_ACTIVE
      : hasFocus
        ? RIBBON_DIMMED
        : RIBBON_QUIET;
  b.flow.addPolyline3D(
    [
      [xLogits + colW * 0.5 + 0.04, WALL_Y, 0],
      [xSoftAgg - smAggW * 0.5 - 0.05, WALL_Y, 0],
    ],
    [0, 1, 0],
    0.088,
    smFlowCol,
    true,
    0.16,
    0.16,
    true,
  );
  b.flow.addPolyline3D(
    [
      [xSoftAgg + smAggW * 0.5 + 0.04, WALL_Y, 0],
      [xProbs - colW * 0.5 - 0.05, WALL_Y, 0],
    ],
    [0, 1, 0],
    0.088,
    smFlowCol,
    true,
    0.16,
    0.16,
    true,
  );

  // -------------------------------------------------------------------------
  // 4. Elevated Top-5 Candidate Rows & Clickable Top-5 Next-Token Strip
  // -------------------------------------------------------------------------
  const probsRec = b.sheetRegistry.get('probs')!;
  const maxTopProb = Math.max(0.05, top5[0]?.prob ?? 0.25);
  const top5StartY = 2.48;
  const top5PitchY = 0.46;
  const top5BarH = 0.30;

  let chosenBarEndPt: Vec3 | null = null;
  let chosenCandLabel = top5[0]?.label ?? '?';

  for (let rank = 0; rank < top5.length; rank++) {
    const cand = top5[rank];
    const partId = `output-cand-${cand.tokenId}`;
    const isSelectedCand = selectedCandidateTokenId === cand.tokenId;
    const isHoveredOrSelectedPart = activePart === partId;
    const isCellMatch =
      activeCell?.tensor === 'probs' &&
      (activeCell.vocabId ?? activeCell.row) === cand.tokenId;

    const barY = top5StartY - rank * top5PitchY;
    const probFrac = Math.max(0.08, Math.min(1.0, cand.prob / maxTopProb));
    const barW = 0.58 + probFrac * 1.40; // x in [5.10, 7.08]
    const barCx = xTop5Left + barW * 0.5;
    // Visibly elevate Top-5 bars forward along +Z (selected or rank-0 pops proudest)
    const barZ = isSelectedCand || isHoveredOrSelectedPart ? 0.30 : rank === 0 ? 0.24 : 0.16;

    const candHl =
      hasCellTrace
        ? isCellMatch
          ? SHEET_HL_ACTIVE
          : SHEET_HL_DIMMED
        : isSelectedCand || isHoveredOrSelectedPart
          ? SHEET_HL_ACTIVE
          : rank === 0
            ? SHEET_HL_PATHWAY
            : hasFocus
              ? SHEET_HL_DIMMED
              : SHEET_HL_OPERAND_B;

    b.addSheet({
      cx: barCx,
      cy: barY,
      cz: barZ,
      ex: barW,
      ey: 0.068,
      ez: top5BarH,
      axes: FRAME_XY_AXES,
      gridCols: 1,
      gridRows: 1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_SOFTMAX,
      fp8Flag: 0,
      highlight: candHl,
      seed: 20.0 + rank * 1.7,
      partId,
      headIndex: cand.tokenId,
      tensorKey: 'probs',
      tokenIndex: focusTok,
      atlasSlice: atlas.probSingleRow(cand.tokenId), // mode = 1, exact Float32 prob!
      registryKey: `top5:${rank}`,
      cellFamily: CELL_ACTIVATION_GREEN,
    });

    // Subtle elevated highlight directly on the matching vocab row of Probs [32×1]
    if (!hasCellTrace) {
      const rowSpan = spanFromSheetGrid(probsRec, cand.tokenId, cand.tokenId, 0, 0);
      drawCellHighlight3D(
        b.flow,
        rowSpan,
        isSelectedCand || rank === 0 ? CELL_TARGET_FILL : CELL_OP_B_FILL,
        isSelectedCand || rank === 0 ? CELL_TARGET_FRAME : CELL_OP_B_FRAME,
        0.018,
        true,
      );
    }

    // Smooth curved connector from Probs[cand.tokenId] -> elevated Top-5 candidate bar
    const probRowPt = localToWorldOnFrame(
      probsFrame,
      0.52,
      0.35,
      -0.5 + (cand.tokenId + 0.5) / vocabSize,
    );
    const barEntryPt: Vec3 = [xTop5Left - 0.05, barY, barZ];
    const linkCol = hasCellTrace
      ? isCellMatch
        ? RIBBON_CAND_ACTIVE
        : RIBBON_DIMMED
      : isSelectedCand || isHoveredOrSelectedPart || rank === 0
        ? RIBBON_CAND_ACTIVE
        : hasFocus
          ? RIBBON_DIMMED
          : RIBBON_CAND_LINK;

    b.flow.addCubic3D(
      probRowPt,
      [xProbs + 0.85, probRowPt[1], barZ * 0.45],
      [xTop5Left - 0.65, barY, barZ * 0.90],
      barEntryPt,
      [0, 1, 0],
      isSelectedCand || rank === 0 ? 0.072 : 0.048,
      linkCol,
      true,
      18,
      true,
      0.15,
      0.15,
    );

    if (isSelectedCand || (chosenBarEndPt === null && rank === 0)) {
      chosenBarEndPt = [xTop5Left + barW + 0.06, barY, barZ];
      chosenCandLabel = cand.label;
    }

    b.landmarks.push({
      id: `out-top5-${rank}`,
      label: `#${rank + 1} "${cand.label}" · ${(cand.prob * 100).toFixed(1)}%`,
      world: [xTop5Left + barW + 0.16, barY, barZ],
      partId,
      align: 'right',
      tier: rank < 3 || isSelectedCand ? 'primary' : 'detail',
      tone: isSelectedCand || rank === 0 ? 'cyan' : 'default',
    });
  }

  // Highlight focus token row on Final Norm when idle
  if (!hasCellTrace) {
    const fnRec = b.sheetRegistry.get('final_norm')!;
    const fnFocusSpan = spanFromSheetGrid(fnRec, focusTok, focusTok, 0, d - 1);
    drawCellHighlight3D(b.flow, fnFocusSpan, CELL_OP_A_FILL, CELL_OP_A_FRAME, 0.022, true);
  }

  // -------------------------------------------------------------------------
  // 5. Curved AR Decode Feedback Loop Arrow (when isArDecode is true)
  // -------------------------------------------------------------------------
  if (isArDecode && chosenBarEndPt) {
    const loopY = 3.24;
    const loopZ = -1.35;
    const loopCol = hasCellTrace ? RIBBON_DIMMED : RIBBON_SSM_LOOP;

    b.flow.addCubic3D(
      chosenBarEndPt,
      [7.42, chosenBarEndPt[1] + 0.45, chosenBarEndPt[2] - 0.35],
      [7.35, loopY, loopZ + 0.45],
      [6.45, loopY, loopZ],
      [0, 1, 0],
      0.115,
      loopCol,
      false,
      16,
      true,
    );
    b.flow.addPolyline3D(
      [
        [6.48, loopY, loopZ],
        [-6.45, loopY, loopZ],
      ],
      [0, 1, 0],
      0.115,
      loopCol,
      false,
      0.20,
      0.20,
      true,
    );
    b.flow.addCubic3D(
      [-6.42, loopY, loopZ],
      [-7.45, loopY, loopZ + 0.35],
      [-7.65, 1.45, -0.85],
      [-7.65, BASE_Y + 0.38, -0.45],
      [0, 1, 0],
      0.115,
      loopCol,
      true,
      16,
      true,
      0.26,
      0.28,
    );

    b.landmarks.push({
      id: 'out-ar-loop',
      label: `→ AR Next Step: feed "${chosenCandLabel}" as t${absFocusT + 1}`,
      sublabel: 'Cached decode · Zero prefix recompute',
      world: [-0.35, loopY + 0.14, loopZ],
      partId: selectedCandidateTokenId !== null ? `output-cand-${selectedCandidateTokenId}` : top5[0] ? `output-cand-${top5[0].tokenId}` : 'probs',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    });
  }

  // -------------------------------------------------------------------------
  // 6. Live 3D Cell Highlights & Cross-Plane Operand Dependency Threads
  // -------------------------------------------------------------------------
  const resolveSpan = (span: OutputSheetSpan): WorldSpan3D | null => {
    const rec = b.sheetRegistry.get(span.tensor);
    if (!rec) return null;
    return spanFromSheetGrid(rec, span.rowStart, span.rowEnd, span.colStart, span.colEnd);
  };

  let cellPopover: CellPopoverSpec | null = null;
  let activeOpPartIds: ReadonlySet<string> | undefined;
  if (activeFormula) {
    const mapTensorPart = (t: OutputAuxTensorKey): string => {
      switch (t) {
        case 'residual_in':
          return 'residual-in';
        case 'rms_agg':
          return 'rms-agg';
        case 'gamma':
          return 'final-gamma';
        case 'final_norm':
          return 'final-norm';
        case 'w_lm_head':
          return 'w-lm-head';
        case 'logits':
          return 'logits';
        case 'softmax_agg':
          return 'softmax-agg';
        case 'probs':
          return 'probs';
      }
    };
    const partIds = new Set<string>([mapTensorPart(activeFormula.targetSpan.tensor)]);
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
        partIds.add(mapTensorPart(up.tensor));
        const upSpan = resolveSpan(up);
        if (!upSpan) continue;
        const isA = up.role === 'operand_a';
        if (isA) hasOpA = true;
        else hasOpB = true;
        sourceWorlds.push({
          world: upSpan.frontPoint,
          role: isA ? 'indigo' : 'teal',
        });
        drawOperandDependency3D(b.flow, upSpan, targetSpan, isA);
      }
      const operandColors: ('indigo' | 'teal' | 'red')[] = [];
      if (hasOpA) operandColors.push('indigo');
      if (hasOpB) operandColors.push('teal');
      operandColors.push('red');
      const c = activeFormula.cell;
      const cTok = Math.max(0, Math.min(T - 1, c.tokenIndex ?? focusTok));
      const cAbsT = offset + cTok;
      const cVocab = Math.max(0, Math.min(vocabSize - 1, c.vocabId ?? c.row));
      const cWord = model.vocab[cVocab] ?? `id:${cVocab}`;
      const popTensorLabel =
        c.tensor === 'residual_in'
          ? 'Bay 52 Y'
          : c.tensor === 'final_norm'
            ? 'Final RMSNorm'
            : c.tensor === 'w_lm_head'
              ? 'Untied W_lm_head'
              : c.tensor === 'logits'
                ? 'Output Logits'
                : 'Output Probs';
      const popCoordLabel =
        c.tensor === 'residual_in' || c.tensor === 'final_norm'
          ? `[t${cAbsT}, ch=${c.col}]`
          : c.tensor === 'w_lm_head'
            ? `[v=${cVocab} "${cWord}", ch=${c.col}]`
            : `[v=${cVocab} "${cWord}" | t${cAbsT}]`;
      const popBadge =
        c.tensor === 'residual_in'
          ? 'Bay 52'
          : c.tensor === 'final_norm'
            ? `ε=${epsLabel}`
            : c.tensor === 'w_lm_head'
              ? 'Untied ≠ W_emb'
              : c.tensor === 'logits'
                ? 'W_lm · U'
                : `Rank #${smSummary.ranks[cVocab] ?? 1}`;
      const popEq =
        c.tensor === 'residual_in'
          ? `Y₅₂[ch${c.col}, t${cAbsT}] · rms=${finalNorm.rms[cTok].toFixed(2)}`
          : c.tensor === 'final_norm'
            ? `(${fmtSigned(residualIn[c.col * T + cTok], 2)} / ${finalNorm.rms[cTok].toFixed(2)}) · γ[${c.col}]`
            : c.tensor === 'w_lm_head'
              ? `W_lm_head[${cVocab}, ch${c.col}] (≠ W_embed)`
              : c.tensor === 'logits'
                ? `Σ_{i=0..${d - 1}} W_lm_head[${cVocab}, i] · U_final[i, t${cAbsT}]`
                : `exp(${fmtSigned(lmHead.logits[cVocab * T + cTok], 2)} − ${fmtSigned(smSummary.maxLogit, 2)}) / Σexp`;
      cellPopover = {
        world: targetSpan.frontPoint,
        tensorLabel: popTensorLabel,
        coordLabel: popCoordLabel,
        badge: popBadge,
        equation: popEq,
        valueText: `= ${activeFormula.valueText}`,
        operandColors,
        sourceWorlds: sourceWorlds.length > 0 ? sourceWorlds : undefined,
      };
    }
    if (activeFormula.cell.tensor === 'probs' && activeFormula.cell.vocabId !== undefined) {
      partIds.add(`output-cand-${activeFormula.cell.vocabId}`);
    }
    activeOpPartIds = partIds;
  }

  // V3 Zoom-adaptive on-cell value grids for Probs, Logits, Final Norm (plus inspected target)
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const vocabLabels = Array.from(
    { length: vocabSize },
    (_, v) => `#${v} ${model.vocab[v] ?? ''}`.trim(),
  );
  const chanLabels = Array.from({ length: d }, (_, ch) => `ch${ch}`);
  const outTokLabels = Array.from(
    { length: T },
    (_, t) => `t${offset + t} ${tokenLabels[t] ?? ''}`.trim(),
  );
  const pushOutputGrid = (
    regKey: string,
    values: Float32Array,
    optsGrid: {
      readonly format?: 'signed' | 'prob';
      readonly activeRow?: number;
      readonly activeCol?: number;
      readonly colLabels?: readonly string[];
      readonly rowLabels?: readonly string[];
      readonly colDimLabel?: string;
      readonly rowDimLabel?: string;
      readonly suppressRowHeaders?: boolean;
    },
  ): void => {
    const rec = b.sheetRegistry.get(regKey);
    if (!rec) return;
    const frame =
      rec.frame ??
      makeSheetFrame3D([rec.cx, rec.cy, rec.cz], [rec.ex, rec.ey, rec.ez]);
    activeCellGrids.push({
      originWorld: frame.center,
      extent: frame.extent,
      axisU: frame.axisU,
      axisN: frame.axisN,
      cols: rec.gridCols,
      rows: rec.gridRows,
      values,
      format: optsGrid.format ?? 'signed',
      activeRow: optsGrid.activeRow,
      activeCol: optsGrid.activeCol,
      colLabels: optsGrid.colLabels,
      rowLabels: optsGrid.rowLabels,
      colDimLabel: optsGrid.colDimLabel,
      rowDimLabel: optsGrid.rowDimLabel,
      ...(optsGrid.suppressRowHeaders ? { suppressRowHeaders: true } : {}),
    });
  };

  const probColValues = new Float32Array(vocabSize);
  const logitColValues = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    probColValues[v] = lmHead.probs[v * T + focusTok];
    logitColValues[v] = lmHead.logits[v * T + focusTok];
  }
  pushOutputGrid('probs', probColValues, {
    format: 'prob',
    activeCol: 0,
    activeRow: activeFormula?.cell.tensor === 'probs' ? activeFormula.cell.row : (top5[0]?.tokenId ?? 0),
    colLabels: [`t${absFocusT} ${focusWord}`],
    rowLabels: vocabLabels,
    suppressRowHeaders: true,
    colDimLabel: 'P',
    rowDimLabel: `V = ${vocabSize}`,
  });
  pushOutputGrid('logits', logitColValues, {
    format: 'signed',
    activeCol: 0,
    activeRow: activeFormula?.cell.tensor === 'logits' ? activeFormula.cell.row : (top5[0]?.tokenId ?? 0),
    colLabels: [`t${absFocusT} ${focusWord}`],
    rowLabels: vocabLabels,
    colDimLabel: 'z',
    rowDimLabel: `V = ${vocabSize}`,
  });

  const finalNormValues = new Float32Array(T * d);
  for (let t = 0; t < T; t++) {
    for (let ch = 0; ch < d; ch++) {
      finalNormValues[t * d + ch] = finalNorm.U_final[ch * T + t];
    }
  }
  pushOutputGrid('final_norm', finalNormValues, {
    format: 'signed',
    activeRow: activeFormula?.cell.tensor === 'final_norm' ? activeFormula.cell.row : focusTok,
    activeCol: activeFormula?.cell.tensor === 'final_norm' ? activeFormula.cell.col : undefined,
    colLabels: chanLabels,
    rowLabels: outTokLabels,
    colDimLabel: `d = ${d}`,
    rowDimLabel: T === 1 ? `S = 1 (t${absFocusT})` : `T = ${T}`,
  });

  if (activeFormula?.cell.tensor === 'residual_in') {
    const resValues = new Float32Array(T * d);
    for (let t = 0; t < T; t++) {
      for (let ch = 0; ch < d; ch++) {
        resValues[t * d + ch] = residualIn[ch * T + t];
      }
    }
    pushOutputGrid('residual_in', resValues, {
      format: 'signed',
      activeRow: activeFormula.cell.row,
      activeCol: activeFormula.cell.col,
      colLabels: chanLabels,
      rowLabels: outTokLabels,
      colDimLabel: `d = ${d}`,
      rowDimLabel: T === 1 ? `S = 1 (t${absFocusT})` : `T = ${T}`,
    });
  } else if (activeFormula?.cell.tensor === 'w_lm_head') {
    const wLmValues = new Float32Array(vocabSize * d);
    for (let v = 0; v < vocabSize; v++) {
      for (let ch = 0; ch < d; ch++) {
        wLmValues[v * d + ch] = lmHead.W_lm_head[v * d + ch];
      }
    }
    pushOutputGrid('w_lm_head', wLmValues, {
      format: 'signed',
      activeRow: activeFormula.cell.row,
      activeCol: activeFormula.cell.col,
      colLabels: chanLabels,
      rowLabels: vocabLabels,
      colDimLabel: `d = ${d}`,
      rowDimLabel: `V = ${vocabSize}`,
    });
  }

  // -------------------------------------------------------------------------
  // 7. 3D Exterior Landmarks & Calipers
  // -------------------------------------------------------------------------
  b.landmarks.push(
    {
      id: 'out-res-in',
      label: 'Bay 52 Residual Y',
      sublabel: `t${absFocusT} "${focusWord}" · d=${d} mini (2,688 prod)`,
      world: [xResIn, 0.18, ioD * 0.5 + 0.28],
      partId: 'residual-in',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'out-rms-agg',
      label: 'RMS Agg',
      sublabel: `rms = ${finalNorm.rms[focusTok].toFixed(2)}`,
      world: [xRmsAgg, 0.16, -ioD * 0.5 - 0.26],
      partId: 'rms-agg',
      align: 'top',
      tier: 'detail',
    },
    {
      id: 'out-gamma',
      label: 'γ Scale',
      world: [xFinalNorm, gammaY + 0.30, gammaZ],
      partId: 'final-gamma',
      align: 'top',
      tier: 'detail',
      tone: isFp8 ? 'gold' : 'default',
    },
    {
      id: 'out-final-norm',
      label: `Final RMSNorm (ε=${epsLabel})`,
      world: [xFinalNorm, 0.18, ioD * 0.5 + 0.28],
      partId: 'final-norm',
      align: 'bottom',
      tier: 'primary',
      tone: isFp8 ? 'gold' : 'default',
    },
    {
      id: 'out-w-lm',
      label: `Untied W_lm_head [${vocabSize}×${d}]`,
      sublabel: 'Untied Weights ≠ W_embed · 2,688 → 131,072 prod',
      world: [xWLm, WALL_Y + WALL_H * 0.5 + 0.16, 0],
      partId: 'w-lm-head',
      align: 'top',
      tier: 'primary',
      tone: isFp8 ? 'gold' : 'default',
    },
    {
      id: 'out-logits',
      label: `Logits [${vocabSize}×1]`,
      world: [xLogits, WALL_Y - WALL_H * 0.5 - 0.14, 0],
      partId: 'logits',
      align: 'bottom',
      tier: 'primary',
    },
    {
      id: 'out-sm-agg',
      label: 'Softmax Agg (max, Σexp)',
      world: [xSoftAgg, WALL_Y + smAggH * 0.5 + 0.20, 0],
      partId: 'softmax-agg',
      align: 'top',
      tier: 'primary',
    },
    {
      id: 'out-probs',
      label: `Probs [${vocabSize}×1]`,
      sublabel: `Top-${top5Set.size} highlighted`,
      world: [xProbs, WALL_Y - WALL_H * 0.5 - 0.14, 0],
      partId: 'probs',
      align: 'bottom',
      tier: 'primary',
      tone: 'cyan',
    },
    {
      id: 'out-top5-header',
      label: 'Top-5 Next-Token Strip',
      sublabel: `V=${vocabSize} mini · ${NEMOTRON_SPEC.vocabSize.toLocaleString()} prod`,
      world: [xTop5Left + 0.85, top5StartY + 0.36, 0.24],
      partId: top5[0] ? `output-cand-${top5[0].tokenId}` : 'probs',
      align: 'top',
      tier: 'primary',
      tone: 'cyan',
    },
  );

  b.calipers.push(
    {
      id: 'cal-out-vocab',
      aWorld: [xWLm, WALL_Y - WALL_H * 0.5, -wLmSpanZ * 0.5 - 0.14],
      bWorld: [xWLm, WALL_Y + WALL_H * 0.5, -wLmSpanZ * 0.5 - 0.14],
      label: `V=${vocabSize} mini · 131,072 prod`,
      partIds: ['w-lm-head', 'logits', 'probs'],
      minPixels: 60,
      offsetPx: 10,
    },
  );

  const reg = (k: string) => b.sheetRegistry.get(k);
  const top5Recs = top5.map((_, r) => reg(`top5:${r}`));
  const opFocusBounds: Record<number, AABB> = {
    0: computeSheetGroupBounds([reg('residual_in')], 0.52, 0.35, 0.52, OUTPUT_SPATIAL_FIT_BOUNDS),
    1: computeSheetGroupBounds([reg('residual_in'), reg('rms_agg'), reg('gamma'), reg('final_norm')], 0.45, 0.28, 0.45, OUTPUT_SPATIAL_FIT_BOUNDS),
    2: computeSheetGroupBounds([reg('final_norm'), reg('w_lm_head')], 0.48, 0.25, 0.48, OUTPUT_SPATIAL_FIT_BOUNDS),
    3: computeSheetGroupBounds([reg('w_lm_head'), reg('logits')], 0.48, 0.25, 0.48, OUTPUT_SPATIAL_FIT_BOUNDS),
    4: computeSheetGroupBounds([reg('logits'), reg('softmax_agg'), reg('probs')], 0.48, 0.25, 0.48, OUTPUT_SPATIAL_FIT_BOUNDS),
    5: computeSheetGroupBounds([reg('probs'), ...top5Recs], 0.48, 0.25, 0.48, OUTPUT_SPATIAL_FIT_BOUNDS),
  };

  const packed = b.pack();
  return {
    kind: 'cutaway_output',
    bayId: 53,
    ...packed,
    fitBounds: OUTPUT_SPATIAL_FIT_BOUNDS,
    opFocusBounds,
    landmarks: b.landmarks,
    kvDepthTicks: [],
    calipers: b.calipers,
    valueAtlas: atlas,
    activeFormula,
    outputFormula: activeFormula,
    activeOpPartIds,
    cellPopover,
    activeCellGrids,
    solidPickBoxes: b.pickBoxes,
    partCatalog: buildOutputPartCatalog(
      d,
      vocabSize,
      epsLabel,
      absFocusT,
      focusWord,
      top5,
      smSummary,
      precision,
      isArDecode,
    ),
  };
}
