/**
 * View 4 (`swiglu`): Normalized SwiGLU Mixer Cutaway Spatial Scene
 * (`src/dory/scene/swigluScene.ts`)
 *
 * Implements DORY4 Slice 4A (Locked Decisions `D1`, `D2`, `D4`, `D5`, `D7`; Guards `DR4`,
 * `DR7`, `DR9`, `DR21`):
 *
 *   - Inspects any odd physical MLP sublayer `p in {1, 3, ..., 71}` (default `p = 17`, first
 *     Recurrent Core MLP sublayer) and recurrent pass `t in 1..k` (or `t = 0` for Prelude/Coda):
 *       1. Bias-free `gate_proj` and `up_proj` expanding `d = 8 -> d_ff = 24` (`3.00x` mini,
 *          explicitly disclosing production `d = 2,560 -> d_ff = 7,168 = 2.80x`).
 *       2. Separate **linear signed** Family 4 SwiGLU scales `suv_gate * sqrt(d)` and
 *          `suv_up * sqrt(d)` (raw init `1.0`, NOT `initializer_range = 0.0177`, and **NO `.abs()`** —
 *          visually highlighting negative signed entries in contrast to Family 1 `|mlp_alpha|`).
 *       3. `SiLU(U_g) * U_u` gated product followed strictly BEFORE `down_proj` by weightless
 *          `ffn_sub_norm = N_{24}(..., eps = 1e-6)` over all `d_ff = 24` intermediate channels.
 *       4. Bias-free `down_proj` (`24 -> 8`) with interactive progressive **`Accum Reveal`**
 *          slider (`accumReveal in [0, 1]`, revealing `0..24` intermediate dot-product terms
 *          `M[row, c] * W_down[o, c]` via `SHEET_REVEAL_*` and exact binary64 partial sums).
 *       5. Channel-wise nGPT NLERP residual blend (`DR7`, `dory-nlerp-residual`, NOT
 *          trigonometric SLERP) with dual raw/effective `|mlp_alpha * 0.05 / 0.0177|` strips.
 *   - Compact dynamically-sized `RG32F` atlas (`width = 128`, `16..80 KiB <= 128 KiB <= 512 KiB`),
 *     `0` overlaps, `0` out-of-bounds, `0` NaN/Inf, `DR21` pick encoding (zero edits to
 *     `src/render/pick.ts`), zero ambient RNG, and strict `D4` architectural boundary isolation.
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL execution, Node built-ins, or ambient RNG.
 */

import type { AABB } from '../../render/camera';
import type { PickBox } from '../../render/pick';
import {
  CELL_MOE_AMBER,
  CELL_SSM_CYAN,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_EMERALD,
  RIBBON_KIND_ARROW_POS_X,
  type DimensionCaliper,
  type LandmarkAnchor,
} from '../../render/primitives';
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
  SHEET_HL_NORMAL,
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_STANDARD,
  SHEET_REVEAL_NONE,
  SHEET_REVEAL_ROW_MAJOR,
  SHEET_REVEAL_UNCOMPUTED,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NEUTRAL,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import type { AtlasSlice } from '../../scene/atlasPacker';
import { fmtSigned } from '../../shared/format';
import {
  computeSheetCellWorldPos,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
} from '../../shared/overlayLayout';
import { vectorL2Norm } from '../ngptMath';
import {
  DORY_PRODUCTION_SPEC,
  decodeDoryToken,
  getDoryPhysicalLayerSpec,
  getDoryStage,
} from '../spec';
import {
  DORY_MINI_CONFIG,
  DORY_NLERP_MARKER,
  type DoryForwardTrace,
  type DoryMlpSublayerTrace,
  type DoryMlpSublayerWeights,
  type DoryReconstructedStepResult,
  type DorySession,
} from '../types';
import {
  DORY_ANTI_SLERP_PLAQUE,
  DORY_FLOW_COLORS,
  DoryAtlasPacker,
  DoryFlatFlowBufferBuilder,
  DoryLineBufferBuilder,
  DoryRibbonBufferBuilder,
  DorySheetBufferBuilder,
  DorySolidBufferBuilder,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_GUIDE,
  computeBoundsFromPoints,
  encodeDoryPickMetadata,
  ensureDorySessionContext,
  mergeAabbs,
  packPickBoxFromAabb,
  packPickBoxFromFrame,
  validateSceneGeometryFinite,
} from './provenance';
import {
  DORY_SPATIAL_VIEW_MODES,
  type DoryBadgeTone,
  type DoryOverlayAnchor,
  type DoryPickIdentity,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DorySceneCallout,
  type DorySpatialScene,
  type DorySwigluSceneOptions,
  type DorySwigluViewMetadata,
} from './types';

// ============================================================================
// VIEW 4 CONSTANTS & PURE HELPERS
// ============================================================================

export const DORY_SWIGLU_ATLAS_WIDTH = 128 as const;
export const DORY_SWIGLU_ATLAS_BUDGET_BYTES = 131_072 as const; // 128.00 KiB <= 512.00 KiB
export const DORY_SWIGLU_DEFAULT_LAYER = 17 as const;

const VIEW_DESCRIPTOR = DORY_SPATIAL_VIEW_MODES[3]!; // 'swiglu'

function clampInt(val: number, lo: number, hi: number): number {
  if (!Number.isFinite(val)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(val)));
}

function clampUnit(val: number): number {
  if (!Number.isFinite(val)) return 1.0;
  return Math.max(0, Math.min(1, val));
}

/**
 * Resolves any requested physical sublayer index into a valid odd MLP physical layer
 * `p in {1, 3, ..., 71}` (snapping even Attention indices cleanly to their paired MLP sublayer).
 */
export function resolveMlpPhysicalLayerIdx(requested?: number): number {
  if (requested === undefined || !Number.isFinite(requested)) {
    return DORY_SWIGLU_DEFAULT_LAYER;
  }
  const clamped = clampInt(requested, 0, DORY_MINI_CONFIG.numHiddenLayers - 1);
  return clamped % 2 === 1 ? clamped : Math.min(DORY_MINI_CONFIG.numHiddenLayers - 1, clamped + 1);
}

/**
 * Computes the exact binary64 progressive `down_proj` partial sum (fixed ascending `0..dFf-1`
 * reduction order, matching `D5`) for one `(selectedTokenRow, selectedOutputDim)` output cell.
 */
export function computeSwigluDownProjPartialSum(
  ffnSubNorm: Float32Array,
  rowIdx: number,
  downProjWeights: Float32Array,
  outputDim: number,
  revealedCount: number,
  dFf: number = DORY_MINI_CONFIG.intermediateSize,
): {
  readonly partialSum: number;
  readonly fullSum: number;
  readonly contributions: Float32Array;
} {
  const rowBase = rowIdx * dFf;
  const wBase = outputDim * dFf;
  const contributions = new Float32Array(dFf);
  const clampedReveal = clampInt(revealedCount, 0, dFf);

  let partialSum = 0;
  let fullSum = 0;
  for (let c = 0; c < dFf; c += 1) {
    const term = (ffnSubNorm[rowBase + c] ?? 0) * (downProjWeights[wBase + c] ?? 0);
    contributions[c] = term;
    fullSum += term;
    if (c < clampedReveal) {
      partialSum += term;
    }
  }
  return { partialSum, fullSum, contributions };
}

function resolveTraceTokens(
  ctx: DoryResolvedSessionContext,
  reconstructed: DoryReconstructedStepResult,
  trace: DoryForwardTrace,
): readonly number[] {
  const tokens: number[] = new Array<number>(trace.seqLen);
  for (let i = 0; i < trace.seqLen; i += 1) {
    let tok: number | undefined;
    if (reconstructed.forwardResult !== null && i < reconstructed.forwardResult.tokens.length) {
      tok = reconstructed.forwardResult.tokens[i];
    } else if (
      reconstructed.specPrefillResult !== null &&
      i < reconstructed.specPrefillResult.promptTokens.length
    ) {
      tok = reconstructed.specPrefillResult.promptTokens[i];
    } else if (
      reconstructed.specCycleResult !== null &&
      i < reconstructed.specCycleResult.verifyPositions.length
    ) {
      tok = reconstructed.specCycleResult.verifyPositions[i]!.inputTokenId;
    } else {
      const absPos = trace.startPos + i;
      tok =
        absPos < ctx.record.committedSequence.length
          ? ctx.record.committedSequence[absPos]
          : ctx.record.inputTokens[i];
    }
    tokens[i] = tok ?? 0;
  }
  return tokens;
}

interface PlacedSwigluSheet {
  readonly partId: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly frame: SheetFrame3D;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly values: Float32Array;
  readonly grid: ActiveSheetCellGrid;
}

function buildSwigluHoveredPopover(
  hovered: DoryPickIdentity | null | undefined,
  sheetsByPartId: ReadonlyMap<string, PlacedSwigluSheet>,
  mlpTrace: DoryMlpSublayerTrace,
  weights: DoryMlpSublayerWeights,
  metadata: DorySwigluViewMetadata,
  tokenLabels: readonly string[],
  accumInfo: {
    readonly partialSum: number;
    readonly fullSum: number;
    readonly contributions: Float32Array;
  },
): CellPopoverSpec | null {
  if (!hovered || !hovered.partId) return null;

  // Hovering a raised negative-signed `suv` marker (`swiglu_neg_suv_gate_c` / `swiglu_neg_suv_up_c`)
  const negSuvMatch = /^swiglu_neg_suv_(gate|up)_(\d+)$/.exec(hovered.partId);
  if (negSuvMatch !== null) {
    const branch = negSuvMatch[1] as 'gate' | 'up';
    const ch = clampInt(Number.parseInt(negSuvMatch[2]!, 10), 0, DORY_MINI_CONFIG.intermediateSize - 1);
    const sheetRec = sheetsByPartId.get(
      branch === 'gate' ? 'swiglu_suv_gate_eff' : 'swiglu_suv_up_eff',
    );
    if (!sheetRec) return null;
    const raw = branch === 'gate' ? (weights.suvGate[ch] ?? 0) : (weights.suvUp[ch] ?? 0);
    const eff =
      branch === 'gate'
        ? (mlpTrace.effectiveSuvGate[ch] ?? 0)
        : (mlpTrace.effectiveSuvUp[ch] ?? 0);
    const pos = computeSheetCellWorldPos(sheetRec.grid, ch, 0, 0.5);
    return Object.freeze({
      world: Object.freeze([pos[0], pos[1] + 0.34, pos[2]] as const),
      tensorLabel: `Signed suv_${branch}[${ch}] * sqrt(8)`,
      coordLabel: `ch c=${ch} · NEGATIVE SIGNED SCALE (NO .abs())`,
      badge: 'FAMILY 4 LINEAR SIGNED · RAW INIT 1.0',
      equation: `${fmtSigned(raw, 4)} * sqrt(8) (preserves negative sign!)`,
      valueText: `= ${fmtSigned(eff, 4)}`,
    });
  }

  const rec = sheetsByPartId.get(hovered.partId);
  if (!rec) return null;

  const row = clampInt(hovered.row ?? 0, 0, rec.gridRows - 1);
  const col = clampInt(hovered.col ?? 0, 0, rec.gridCols - 1);
  const world = computeSheetCellWorldPos(rec.grid, col, row, 0.5);
  const idx = row * rec.gridCols + col;
  const val = rec.values[idx] ?? 0;
  const tokLabel = tokenLabels[row] ?? `t${row}`;
  const dFf = DORY_MINI_CONFIG.intermediateSize;

  if (rec.partId === 'swiglu_suv_gate_raw' || rec.partId === 'swiglu_suv_gate_eff') {
    const raw = weights.suvGate[col] ?? 0;
    const eff = mlpTrace.effectiveSuvGate[col] ?? 0;
    return Object.freeze({
      world,
      tensorLabel:
        rec.partId === 'swiglu_suv_gate_raw'
          ? `Raw suv_gate[${col}]`
          : `Effective suv_gate[${col}] * sqrt(8)`,
      coordLabel: `ch c=${col} · ${eff < 0 ? 'NEGATIVE SIGNED' : 'POSITIVE SIGNED'} (NO .abs())`,
      badge: eff < 0 ? 'SIGNED NEGATIVE · NO .abs()' : 'FAMILY 4 · INIT 1.0 (NO sigma_0)',
      equation: `${fmtSigned(raw, 4)} * sqrt(8) = ${fmtSigned(eff, 4)}`,
      valueText:
        rec.partId === 'swiglu_suv_gate_raw' ? `raw = ${fmtSigned(raw, 4)}` : `= ${fmtSigned(eff, 4)}`,
    });
  }

  if (rec.partId === 'swiglu_suv_up_raw' || rec.partId === 'swiglu_suv_up_eff') {
    const raw = weights.suvUp[col] ?? 0;
    const eff = mlpTrace.effectiveSuvUp[col] ?? 0;
    return Object.freeze({
      world,
      tensorLabel:
        rec.partId === 'swiglu_suv_up_raw'
          ? `Raw suv_up[${col}]`
          : `Effective suv_up[${col}] * sqrt(8)`,
      coordLabel: `ch c=${col} · ${eff < 0 ? 'NEGATIVE SIGNED' : 'POSITIVE SIGNED'} (NO .abs())`,
      badge: eff < 0 ? 'SIGNED NEGATIVE · NO .abs()' : 'FAMILY 4 · INIT 1.0 (NO sigma_0)',
      equation: `${fmtSigned(raw, 4)} * sqrt(8) = ${fmtSigned(eff, 4)}`,
      valueText:
        rec.partId === 'swiglu_suv_up_raw' ? `raw = ${fmtSigned(raw, 4)}` : `= ${fmtSigned(eff, 4)}`,
    });
  }

  if (rec.partId === 'swiglu_product') {
    const siluVal = mlpTrace.gateSilu[idx] ?? 0;
    const upVal = mlpTrace.upScaled[idx] ?? 0;
    const siluSheet = sheetsByPartId.get('swiglu_gate_silu');
    const upSheet = sheetsByPartId.get('swiglu_up_scaled');
    return Object.freeze({
      world,
      tensorLabel: 'SwiGLU Product SiLU(U_g) * U_u',
      coordLabel: `${tokLabel} · ch c=${col}`,
      badge: 'PRE-ffn_sub_norm PRODUCT',
      equation: `${fmtSigned(siluVal, 3)} * ${fmtSigned(upVal, 3)}`,
      valueText: `= ${fmtSigned(val, 4)}`,
      operandColors: Object.freeze(['indigo', 'teal'] as const),
      sourceWorlds:
        siluSheet && upSheet
          ? Object.freeze([
              { world: computeSheetCellWorldPos(siluSheet.grid, col, row, 0.5), role: 'indigo' as const },
              { world: computeSheetCellWorldPos(upSheet.grid, col, row, 0.5), role: 'teal' as const },
            ])
          : undefined,
    });
  }

  if (rec.partId === 'swiglu_ffn_sub_norm') {
    const rowNorm = vectorL2Norm(mlpTrace.ffnSubNorm, row * dFf, dFf);
    return Object.freeze({
      world,
      tensorLabel: 'ffn_sub_norm M = N_24(SiLU(U_g) * U_u)',
      coordLabel: `${tokLabel} · ch c=${col} · ||M||_2 = ${rowNorm.toFixed(4)}`,
      badge: 'PRE-down_proj L2NORM (24-DIM, eps=1e-6)',
      equation: `prod[${row},${col}] / sqrt(||prod||_2^2 + 1e-6)`,
      valueText: `= ${fmtSigned(val, 4)}`,
    });
  }

  if (rec.partId === 'swiglu_accum_terms') {
    const mVal = mlpTrace.ffnSubNorm[metadata.selectedTokenRow * dFf + col] ?? 0;
    const wVal = weights.downProj[metadata.selectedOutputDim * dFf + col] ?? 0;
    const isRevealed = col < metadata.revealedIntermediateCount;
    const isAccum = metadata.revealedIntermediateCount < dFf;
    return Object.freeze({
      world,
      tensorLabel: `down_proj Term c=${col} -> out d${metadata.selectedOutputDim}`,
      coordLabel: `pos ${metadata.selectedAbsTokenPos} · reveal ${metadata.revealedIntermediateCount}/${dFf}`,
      badge: isRevealed ? 'ACCUMULATED TERM' : 'UNREVEALED (SLIDE ACCUM REVEAL)',
      equation: `M[${metadata.selectedTokenRow},${col}](${fmtSigned(mVal, 3)}) * W_down[${metadata.selectedOutputDim},${col}](${fmtSigned(wVal, 3)})`,
      valueText: isRevealed
        ? `= ${fmtSigned(val, 4)} (Σ=${fmtSigned(accumInfo.partialSum, 3)})`
        : `(hidden; full Σ=${fmtSigned(accumInfo.fullSum, 3)})`,
      isAccumulating: isAccum,
    });
  }

  if (rec.partId === 'swiglu_down_proj_out') {
    const isFocusedOut = row === metadata.selectedTokenRow && col === metadata.selectedOutputDim;
    const isAccum = isFocusedOut && metadata.revealedIntermediateCount < dFf;
    return Object.freeze({
      world,
      tensorLabel: `down_proj O = M W_down^T (24 -> 8)`,
      coordLabel: `${tokLabel} · out ch d${col}`,
      badge: isAccum
        ? `ACCUM REVEAL ${metadata.revealedIntermediateCount}/${dFf}`
        : 'BIAS-FREE DOWN PROJECTION',
      equation: isAccum
        ? `Σ_{c=0..${Math.max(0, metadata.revealedIntermediateCount - 1)}} M[${row},c] * W_down[${col},c]`
        : `Σ_{c=0..23} M[${row},c] * W_down[${col},c]`,
      valueText: isAccum
        ? `Σ = ${fmtSigned(accumInfo.partialSum, 4)} (full ${fmtSigned(accumInfo.fullSum, 4)})`
        : `= ${fmtSigned(val, 4)}`,
      isAccumulating: isAccum,
    });
  }

  if (
    rec.partId === 'swiglu_mlp_alpha_raw' ||
    rec.partId === 'swiglu_mlp_alpha_eff' ||
    rec.partId === 'swiglu_nlerp_output'
  ) {
    const raw = weights.mlpAlpha[col] ?? 0;
    const eff = mlpTrace.nlerp.effectiveAlpha[col] ?? 0;
    return Object.freeze({
      world,
      tensorLabel:
        rec.partId === 'swiglu_mlp_alpha_raw'
          ? `Raw mlp_alpha[${col}]`
          : rec.partId === 'swiglu_mlp_alpha_eff'
            ? `Effective |mlp_alpha[${col}]|`
            : `NLERP Residual Output x_next`,
      coordLabel:
        rec.partId === 'swiglu_nlerp_output'
          ? `${tokLabel} · ch d${col}`
          : `ch d${col} · sigma_0=0.0177`,
      badge: DORY_NLERP_MARKER,
      equation: `|${fmtSigned(raw, 4)} * 0.05 / 0.0177| = ${eff.toFixed(4)} (NLERP, NOT SLERP)`,
      valueText: `= ${fmtSigned(val, 4)}`,
    });
  }

  return Object.freeze({
    world,
    tensorLabel: rec.label,
    coordLabel: `${tokLabel} · ch ${col}`,
    badge: `Phys L${metadata.physicalLayerIdx} · Pair L${metadata.pairedLayerIdx} · MLP`,
    equation: `${rec.tensorKey}[${row}, ${col}]`,
    valueText: `= ${fmtSigned(val, 4)}`,
  });
}

// ============================================================================
// MAIN VIEW 4 SCENE BUILDER (`buildDorySwigluScene`)
// ============================================================================

/**
 * Builds the complete headless 5-pass WebGL2 spatial scene for **View 4 (`swiglu`)**:
 * Normalized SwiGLU Mixer Cutaway (`d=8 -> d_ff=24 [3.00x]`, linear signed `suv_gate` /
 * `suv_up` with NO `.abs()`, `ffn_sub_norm`, progressive `Accum Reveal`, and channel-wise
 * nGPT NLERP residual blend).
 *
 * Strictly read-only — never mutates `DorySession`.
 */
export function buildDorySwigluScene(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  options: DorySwigluSceneOptions = {},
): DorySpatialScene {
  const ctx = ensureDorySessionContext(sessionOrCtx, options.recordId);
  const config = ctx.params.config;
  const d = config.hiddenSize; // 8
  const dFf = config.intermediateSize; // 24

  const reconstructed = ctx.resolveStepTrace();
  const trace = ctx.resolveForwardTrace();

  const physicalLayerIdx = resolveMlpPhysicalLayerIdx(options.physicalLayerIdx);
  const layerSpec = getDoryPhysicalLayerSpec(physicalLayerIdx);
  const stage = getDoryStage(physicalLayerIdx);
  const loopIteration =
    stage === 'recurrent' ? clampInt(options.loopIteration ?? 1, 1, ctx.kLoops) : 0;

  const sublayerRecord = ctx.resolveSublayerRecord(physicalLayerIdx, loopIteration);
  if (sublayerRecord.trace.kind !== 'mlp') {
    throw new Error(
      `View 4 swigluScene expected mlp trace at physicalLayerIdx=${physicalLayerIdx}, got '${sublayerRecord.trace.kind}'.`,
    );
  }
  const mlpTrace: DoryMlpSublayerTrace = sublayerRecord.trace;
  const layerWeights = ctx.params.layers[physicalLayerIdx]!;
  if (layerWeights.kind !== 'mlp') {
    throw new Error(`Expected mlp weights at physicalLayerIdx=${physicalLayerIdx}.`);
  }

  const seqLen = mlpTrace.seqLen;
  const selectedTokenRow = clampInt(
    options.selectedTokenRow ?? ctx.resolveDefaultTokenOffset(),
    0,
    Math.max(0, seqLen - 1),
  );
  const selectedAbsTokenPos = trace.startPos + selectedTokenRow;
  const traceTokens = resolveTraceTokens(ctx, reconstructed, trace);
  const selectedTokenId = traceTokens[selectedTokenRow] ?? 0;
  const selectedTokenText = decodeDoryToken(selectedTokenId);

  const selectedChannel = clampInt(options.selectedChannel ?? 0, 0, dFf - 1);
  const selectedOutputDim = clampInt(options.selectedOutputDim ?? 0, 0, d - 1);
  const accumReveal = clampUnit(options.accumReveal ?? 1.0);
  const revealedIntermediateCount = clampInt(Math.round(accumReveal * dFf), 0, dFf);

  // Count signed negative suv entries (verifying NO .abs() on Family 4 scales!)
  const negativeSuvGateChannels: number[] = [];
  const negativeSuvUpChannels: number[] = [];
  for (let c = 0; c < dFf; c += 1) {
    if (mlpTrace.effectiveSuvGate[c]! < 0) negativeSuvGateChannels.push(c);
    if (mlpTrace.effectiveSuvUp[c]! < 0) negativeSuvUpChannels.push(c);
  }

  const ffnSubNormL2Norm = vectorL2Norm(mlpTrace.ffnSubNorm, selectedTokenRow * dFf, dFf);
  const accumInfo = computeSwigluDownProjPartialSum(
    mlpTrace.ffnSubNorm,
    selectedTokenRow,
    layerWeights.downProj,
    selectedOutputDim,
    revealedIntermediateCount,
    dFf,
  );

  const mlpExpansionRatioMini = dFf / d; // 3.0
  const mlpExpansionRatioProd =
    DORY_PRODUCTION_SPEC.intermediateSize / DORY_PRODUCTION_SPEC.hiddenSize; // 2.8
  const suvMultiplierMini = Math.sqrt(d); // sqrt(8)
  const suvMultiplierProd = Math.sqrt(DORY_PRODUCTION_SPEC.hiddenSize); // sqrt(2560)
  const mlpExpansionDisclosure =
    `Mini d=${d} -> d_ff=${dFf} (${mlpExpansionRatioMini.toFixed(2)}x) vs. ` +
    `Prod d=${DORY_PRODUCTION_SPEC.hiddenSize} -> d_ff=${DORY_PRODUCTION_SPEC.intermediateSize} (${mlpExpansionRatioProd.toFixed(2)}x) · Bias-Free Normalized SwiGLU`;

  const metadata: DorySwigluViewMetadata = Object.freeze({
    physicalLayerIdx,
    pairedLayerIdx: layerSpec.pairedLayerIdx,
    stage,
    loopIteration,
    seqLen,
    startPos: trace.startPos,
    selectedTokenRow,
    selectedAbsTokenPos,
    selectedTokenId,
    selectedTokenText,
    selectedChannel,
    selectedOutputDim,
    accumReveal,
    revealedIntermediateCount,
    mlpExpansionDisclosure,
    mlpExpansionRatioMini,
    mlpExpansionRatioProd,
    suvMultiplierMini,
    suvMultiplierProd,
    ffnSubNormAxisDim: dFf,
    ffnSubNormL2Norm,
    hasSignedNegativeSuv:
      negativeSuvGateChannels.length > 0 || negativeSuvUpChannels.length > 0,
    negativeSuvGateCount: negativeSuvGateChannels.length,
    negativeSuvUpCount: negativeSuvUpChannels.length,
    extrapolatedAlphaCount: mlpTrace.nlerp.extrapolatedChannelCount,
    maxEffectiveAlpha: mlpTrace.nlerp.maxEffectiveAlpha,
    antiSlerpMarker: DORY_NLERP_MARKER,
  });

  // ==========================================================================
  // 1. PACK LIVE RG32F ATLAS (<= 128 KiB <= 512 KiB, 0 OVERLAPS)
  // ==========================================================================
  const neededRows = seqLen + Math.max(seqLen, d) + 4;
  const atlasHeight = Math.min(128, Math.max(16, Math.ceil(neededRows / 16) * 16));
  const atlasPacker = new DoryAtlasPacker(
    DORY_SWIGLU_ATLAS_WIDTH,
    atlasHeight,
    DORY_SWIGLU_ATLAS_BUDGET_BYTES,
  );

  // Shelf 1 (width = 24*5 + 8 = 128 exact)
  const sliceGatePre = atlasPacker.packMatrix('swiglu_gate_pre', mlpTrace.gatePre, seqLen, dFf, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceUpPre = atlasPacker.packMatrix('swiglu_up_pre', mlpTrace.upPre, seqLen, dFf, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceGateScaled = atlasPacker.packMatrix(
    'swiglu_gate_scaled',
    mlpTrace.gateScaled,
    seqLen,
    dFf,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceUpScaled = atlasPacker.packMatrix(
    'swiglu_up_scaled',
    mlpTrace.upScaled,
    seqLen,
    dFf,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceGateSilu = atlasPacker.packMatrix(
    'swiglu_gate_silu',
    mlpTrace.gateSilu,
    seqLen,
    dFf,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceHIn = atlasPacker.packMatrix('swiglu_h_in', mlpTrace.nlerp.aNorm, seqLen, d, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });

  // Shelf 2 (width = 24*2 + 8*2 + 24 + 24 = 112 <= 128)
  const sliceProduct = atlasPacker.packMatrix(
    'swiglu_product',
    mlpTrace.swigluProduct,
    seqLen,
    dFf,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceFfnSubNorm = atlasPacker.packMatrix(
    'swiglu_ffn_sub_norm',
    mlpTrace.ffnSubNorm,
    seqLen,
    dFf,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceDownProjOut = atlasPacker.packMatrix(
    'swiglu_down_proj_out',
    mlpTrace.downProjOut,
    seqLen,
    d,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceNlerpOut = atlasPacker.packMatrix(
    'swiglu_nlerp_output',
    mlpTrace.nlerp.output,
    seqLen,
    d,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceDownProjWeight = atlasPacker.packMatrix(
    'swiglu_down_proj_weight',
    layerWeights.downProj,
    d,
    dFf,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceAccumTerms = atlasPacker.packMatrix(
    'swiglu_accum_terms',
    accumInfo.contributions,
    1,
    dFf,
    { mode: 0, minFloor: 0.05, physicalLayerIdx, loopIteration },
  );

  // Shelf 3 (1-row scale strips: width = 24*4 + 8*2 = 112 <= 128)
  const sliceSuvGateRaw = atlasPacker.packMatrix(
    'swiglu_suv_gate_raw',
    layerWeights.suvGate,
    1,
    dFf,
    { mode: 0, minFloor: 0.5, physicalLayerIdx, loopIteration },
  );
  const sliceSuvGateEff = atlasPacker.packMatrix(
    'swiglu_suv_gate_eff',
    mlpTrace.effectiveSuvGate,
    1,
    dFf,
    { mode: 0, minFloor: 1.0, physicalLayerIdx, loopIteration },
  );
  const sliceSuvUpRaw = atlasPacker.packMatrix(
    'swiglu_suv_up_raw',
    layerWeights.suvUp,
    1,
    dFf,
    { mode: 0, minFloor: 0.5, physicalLayerIdx, loopIteration },
  );
  const sliceSuvUpEff = atlasPacker.packMatrix(
    'swiglu_suv_up_eff',
    mlpTrace.effectiveSuvUp,
    1,
    dFf,
    { mode: 0, minFloor: 1.0, physicalLayerIdx, loopIteration },
  );
  const sliceMlpAlphaRaw = atlasPacker.packMatrix(
    'swiglu_mlp_alpha_raw',
    layerWeights.mlpAlpha,
    1,
    d,
    { mode: 0, minFloor: 0.02, physicalLayerIdx, loopIteration },
  );
  const sliceMlpAlphaEff = atlasPacker.packMatrix(
    'swiglu_mlp_alpha_eff',
    mlpTrace.nlerp.effectiveAlpha,
    1,
    d,
    {
      mode: 1,
      fixedScale: 1.0 / Math.max(1.0, mlpTrace.nlerp.maxEffectiveAlpha),
      physicalLayerIdx,
      loopIteration,
    },
  );

  const atlasExtraTag = `b=${ctx.branch.branchId}:r=${ctx.record.recordId}:p=${physicalLayerIdx}:t=${loopIteration}:row=${selectedTokenRow}:out=${selectedOutputDim}`;
  const packedAtlas = atlasPacker.build('swiglu', atlasExtraTag);

  // ==========================================================================
  // 2. BUILD 5-PASS 3D EDITORIAL DRAFTING-TABLE GEOMETRY
  // ==========================================================================
  const solids = new DorySolidBufferBuilder();
  const lines = new DoryLineBufferBuilder();
  const ribbons = new DoryRibbonBufferBuilder();
  const sheets = new DorySheetBufferBuilder();
  const flatFlow = new DoryFlatFlowBufferBuilder();

  const pickBoxes: PickBox[] = [];
  const aabbs: AABB[] = [];
  const overlayAnchors: DoryOverlayAnchor[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const sheetsByPartId = new Map<string, PlacedSwigluSheet>();

  const tokenLabels = Object.freeze(
    traceTokens.map((tokId, i) => `p${trace.startPos + i}:${decodeDoryToken(tokId)}`),
  );
  const dLabels = Object.freeze(Array.from({ length: d }, (_, c) => `d${c}`));
  const ffLabels = Object.freeze(Array.from({ length: dFf }, (_, c) => `f${c}`));

  const sheetH = 0.068;
  const ex8 = d * 0.26; // 2.08
  const ex24 = dFf * 0.145; // 3.48
  const seqEz = Math.max(0.62, Math.min(3.1, seqLen * 0.18));
  const stripEz = 0.36;
  const weightEz = d * 0.17; // 1.36

  const zGate = -2.8;
  const zUp = 2.8;

  const accumRevealMode =
    revealedIntermediateCount === 0
      ? SHEET_REVEAL_UNCOMPUTED
      : revealedIntermediateCount < dFf
        ? SHEET_REVEAL_ROW_MAJOR
        : SHEET_REVEAL_NONE;
  const accumColFrontier = Math.max(0, revealedIntermediateCount - 1);

  const registerSheet = (spec: {
    readonly partId: string;
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
    readonly highlight: number;
    readonly slice: AtlasSlice;
    readonly values: Float32Array;
    readonly format: 'signed' | 'prob';
    readonly colLabels: readonly string[];
    readonly rowLabels: readonly string[];
    readonly colDimLabel: string;
    readonly rowDimLabel: string;
    readonly activeRow?: number;
    readonly activeCol?: number;
    readonly revealMode?: number;
    readonly revealColFrontier?: number;
    readonly revealRowFrontier?: number;
    readonly revealPulsePhase?: number;
    readonly activeValueOverride?: number | null;
    readonly activeValueTextOverride?: string | null;
    readonly includeActiveGrid?: boolean;
    readonly extraTags?: Readonly<Record<string, string | number | boolean>>;
  }): PlacedSwigluSheet => {
    const frame = makeSheetFrame3D(
      [spec.cx, spec.cy, spec.cz],
      [spec.ex, spec.ey, spec.ez],
      FRAME_XZ_AXES.axisU,
      FRAME_XZ_AXES.axisN,
    );
    sheets.push({
      cx: spec.cx,
      cy: spec.cy,
      cz: spec.cz,
      ex: spec.ex,
      ey: spec.ey,
      ez: spec.ez,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      sheetKind: SHEET_KIND_STANDARD,
      role: spec.role,
      highlight: spec.highlight,
      slice: spec.slice,
      frame,
      revealMode: spec.revealMode,
      revealColFrontier: spec.revealColFrontier,
      revealRowFrontier: spec.revealRowFrontier,
      revealPulsePhase: spec.revealPulsePhase,
    });

    const meta = encodeDoryPickMetadata({
      physicalLayerIdx,
      executedLayerIdx: sublayerRecord.executedLayerIdx,
      stage,
      loopIteration,
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      absTokenPos: selectedAbsTokenPos,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      modelWeightId: ctx.params.modelWeightId,
      kvPolicy: ctx.kvCacheMode,
      cacheRole: sublayerRecord.coord.cacheOwnership.role,
      groupRole: 'loop',
      tapRole: 'loop',
      extraTags: {
        view: 'swiglu',
        ...(spec.extraTags ?? {}),
      },
    });
    pickBoxes.push(packPickBoxFromFrame(frame, meta));
    aabbs.push(sheetFrameWorldAabb(frame));

    const grid: ActiveSheetCellGrid = Object.freeze({
      originWorld: frame.center,
      extent: frame.extent,
      axisU: frame.axisU,
      axisN: frame.axisN,
      cols: spec.gridCols,
      rows: spec.gridRows,
      values: spec.values,
      format: spec.format,
      activeRow: spec.activeRow,
      activeCol: spec.activeCol,
      ...(spec.revealMode !== undefined ? { revealMode: spec.revealMode } : {}),
      ...(spec.revealColFrontier !== undefined
        ? { revealColFrontier: spec.revealColFrontier }
        : {}),
      ...(spec.revealRowFrontier !== undefined
        ? { revealRowFrontier: spec.revealRowFrontier }
        : {}),
      ...(spec.activeValueOverride !== undefined
        ? { activeValueOverride: spec.activeValueOverride }
        : {}),
      ...(spec.activeValueTextOverride !== undefined
        ? { activeValueTextOverride: spec.activeValueTextOverride }
        : {}),
      colLabels: spec.colLabels,
      rowLabels: spec.rowLabels,
      colDimLabel: spec.colDimLabel,
      rowDimLabel: spec.rowDimLabel,
    });
    if (spec.includeActiveGrid !== false) {
      activeCellGrids.push(grid);
    }

    const placed: PlacedSwigluSheet = {
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      label: spec.label,
      frame,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      values: spec.values,
      grid,
    };
    sheetsByPartId.set(spec.partId, placed);
    return placed;
  };

  // Drafting-table zone frames
  lines.addBoxFrameXZ(-16.4, -7.2, 2.2, 6.8, -0.06, 1.0, LINE_STYLE_GUIDE, 0.34);
  lines.addBoxFrameXZ(2.7, -7.2, 12.2, 6.8, -0.06, 1.0, LINE_STYLE_GUIDE, 0.36);
  lines.addBoxFrameXZ(12.7, -7.2, 20.6, 6.8, -0.06, 1.0, LINE_STYLE_GUIDE, 0.34);

  // --- COL 0: Input Residual State ---
  const sheetHIn = registerSheet({
    partId: 'swiglu_h_in',
    tensorKey: 'nlerp_a_norm',
    label: 'h_in = N_8(x_res) [S, 8]',
    cx: -14.8,
    cy: 0,
    cz: 0,
    ex: ex8,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_OPERAND_A,
    slice: sliceHIn,
    values: mlpTrace.nlerp.aNorm,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
  });

  // --- COL 1: Bias-Free Gate & Up Projections (`d=8 -> d_ff=24 [3.00x]`) ---
  const sheetGatePre = registerSheet({
    partId: 'swiglu_gate_pre',
    tensorKey: 'gate_proj',
    label: 'gate_pre = h W_gate^T (8->24, 3.00x)',
    cx: -10.2,
    cy: 0,
    cz: zGate,
    ex: ex24,
    ey: sheetH,
    ez: seqEz,
    gridCols: dFf,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_A,
    slice: sliceGatePre,
    values: mlpTrace.gatePre,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d_ff=${dFf} (3.0x)`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedChannel,
    includeActiveGrid: false,
  });

  const sheetUpPre = registerSheet({
    partId: 'swiglu_up_pre',
    tensorKey: 'up_proj',
    label: 'up_pre = h W_up^T (8->24, 3.00x)',
    cx: -10.2,
    cy: 0,
    cz: zUp,
    ex: ex24,
    ey: sheetH,
    ez: seqEz,
    gridCols: dFf,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceUpPre,
    values: mlpTrace.upPre,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d_ff=${dFf} (3.0x)`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedChannel,
    includeActiveGrid: false,
  });

  // --- COL 2: Linear Signed `suv_gate` & `suv_up` (`* sqrt(8)`, NO `.abs()`) ---
  const sheetSuvGateRaw = registerSheet({
    partId: 'swiglu_suv_gate_raw',
    tensorKey: 'suv_gate',
    label: 'Raw suv_gate (init 1.0, signed)',
    cx: -5.2,
    cy: 0,
    cz: -5.85,
    ex: ex24,
    ey: sheetH,
    ez: stripEz,
    gridCols: dFf,
    gridRows: 1,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_NORMAL,
    slice: sliceSuvGateRaw,
    values: layerWeights.suvGate,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: Object.freeze(['raw']),
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  const sheetSuvGateEff = registerSheet({
    partId: 'swiglu_suv_gate_eff',
    tensorKey: 'suv_gate',
    label: 'Signed suv_gate * sqrt(8) (NO .abs()!)',
    cx: -5.2,
    cy: 0,
    cz: -5.2,
    ex: ex24,
    ey: sheetH,
    ez: stripEz,
    gridCols: dFf,
    gridRows: 1,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: sliceSuvGateEff,
    values: mlpTrace.effectiveSuvGate,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: Object.freeze(['*sqrt(8)']),
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: '1',
    activeCol: selectedChannel,
    extraTags: { signedNoAbs: true },
  });

  const sheetGateScaled = registerSheet({
    partId: 'swiglu_gate_scaled',
    tensorKey: 'gate_scaled',
    label: 'U_g = gate_pre * (suv_gate * sqrt(8))',
    cx: -5.2,
    cy: 0,
    cz: zGate,
    ex: ex24,
    ey: sheetH,
    ez: seqEz,
    gridCols: dFf,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceGateScaled,
    values: mlpTrace.gateScaled,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedChannel,
    includeActiveGrid: false,
  });

  const sheetUpScaled = registerSheet({
    partId: 'swiglu_up_scaled',
    tensorKey: 'up_scaled',
    label: 'U_u = up_pre * (suv_up * sqrt(8))',
    cx: -5.2,
    cy: 0,
    cz: zUp,
    ex: ex24,
    ey: sheetH,
    ez: seqEz,
    gridCols: dFf,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceUpScaled,
    values: mlpTrace.upScaled,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedChannel,
    includeActiveGrid: false,
  });

  const sheetSuvUpEff = registerSheet({
    partId: 'swiglu_suv_up_eff',
    tensorKey: 'suv_up',
    label: 'Signed suv_up * sqrt(8) (NO .abs()!)',
    cx: -5.2,
    cy: 0,
    cz: 5.2,
    ex: ex24,
    ey: sheetH,
    ez: stripEz,
    gridCols: dFf,
    gridRows: 1,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: sliceSuvUpEff,
    values: mlpTrace.effectiveSuvUp,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: Object.freeze(['*sqrt(8)']),
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: '1',
    activeCol: selectedChannel,
    extraTags: { signedNoAbs: true },
  });

  const sheetSuvUpRaw = registerSheet({
    partId: 'swiglu_suv_up_raw',
    tensorKey: 'suv_up',
    label: 'Raw suv_up (init 1.0, signed)',
    cx: -5.2,
    cy: 0,
    cz: 5.85,
    ex: ex24,
    ey: sheetH,
    ez: stripEz,
    gridCols: dFf,
    gridRows: 1,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_NORMAL,
    slice: sliceSuvUpRaw,
    values: layerWeights.suvUp,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: Object.freeze(['raw']),
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  // 3D markers above negative signed `suv_gate` & `suv_up` entries (visual proof of NO `.abs()`!)
  const emitSignedNegativeSuvMarkers = (
    branch: 'gate' | 'up',
    effSheet: PlacedSwigluSheet,
    negChannels: readonly number[],
    effValues: Float32Array,
  ): void => {
    for (let i = 0; i < negChannels.length; i += 1) {
      const ch = negChannels[i]!;
      const mag = Math.abs(effValues[ch] ?? 0);
      const barH = Math.min(0.46, Math.max(0.18, mag * 0.095));
      const pos = computeSheetCellWorldPos(effSheet.grid, ch, 0, 0.5);
      const cy = pos[1] + barH * 0.5 + 0.03;

      solids.push({
        cx: pos[0],
        cy,
        cz: pos[2],
        ex: 0.1,
        ey: barH,
        ez: 0.14,
        cellFamily: branch === 'gate' ? CELL_MOE_AMBER : CELL_SSM_CYAN,
        bayId: physicalLayerIdx + 1,
        cellScale: 0.12,
        highlight: ch === selectedChannel ? 1 : 2,
      });
      lines.addBoxEdges(
        pos[0],
        cy,
        pos[2],
        0.1,
        barH,
        0.14,
        physicalLayerIdx + 1,
        0.95,
        LINE_STYLE_ACTIVE,
        0.8,
      );
      const meta = encodeDoryPickMetadata({
        physicalLayerIdx,
        executedLayerIdx: sublayerRecord.executedLayerIdx,
        stage,
        loopIteration,
        partId: `swiglu_neg_suv_${branch}_${ch}`,
        tensorKey: branch === 'gate' ? 'suv_gate' : 'suv_up',
        absTokenPos: selectedAbsTokenPos,
        modelWeightId: ctx.params.modelWeightId,
        extraTags: { view: 'swiglu', channel: ch, signedNegative: true },
      });
      pickBoxes.push(packPickBoxFromAabb(pos[0], cy, pos[2], 0.13, barH, 0.17, meta));
      aabbs.push({
        min: [pos[0] - 0.08, cy - barH * 0.5, pos[2] - 0.09],
        max: [pos[0] + 0.08, cy + barH * 0.5, pos[2] + 0.09],
      });
    }
  };

  emitSignedNegativeSuvMarkers('gate', sheetSuvGateEff, negativeSuvGateChannels, mlpTrace.effectiveSuvGate);
  emitSignedNegativeSuvMarkers('up', sheetSuvUpEff, negativeSuvUpChannels, mlpTrace.effectiveSuvUp);

  // --- COL 3..5: SiLU(U_g) -> Hadamard Product -> `ffn_sub_norm` (24-dim) + Progressive `Accum Reveal` ---
  const sheetGateSilu = registerSheet({
    partId: 'swiglu_gate_silu',
    tensorKey: 'gate_silu',
    label: 'SiLU(U_g) = U_g * sigma(U_g)',
    cx: -0.2,
    cy: 0,
    cz: zGate,
    ex: ex24,
    ey: sheetH,
    ez: seqEz,
    gridCols: dFf,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_A,
    slice: sliceGateSilu,
    values: mlpTrace.gateSilu,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedChannel,
    includeActiveGrid: false,
  });

  const sheetProduct = registerSheet({
    partId: 'swiglu_product',
    tensorKey: 'swiglu_product',
    label: 'SwiGLU Product = SiLU(U_g) * U_u',
    cx: 4.8,
    cy: 0,
    cz: 0,
    ex: ex24,
    ey: sheetH,
    ez: seqEz,
    gridCols: dFf,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceProduct,
    values: mlpTrace.swigluProduct,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedChannel,
  });

  const sheetFfnSubNorm = registerSheet({
    partId: 'swiglu_ffn_sub_norm',
    tensorKey: 'ffn_sub_norm',
    label: 'ffn_sub_norm M = N_24(prod) (Pre-down_proj)',
    cx: 9.8,
    cy: 0,
    cz: 0,
    ex: ex24,
    ey: sheetH,
    ez: seqEz,
    gridCols: dFf,
    gridRows: seqLen,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: sliceFfnSubNorm,
    values: mlpTrace.ffnSubNorm,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedChannel,
  });

  const sheetDownWeight = registerSheet({
    partId: 'swiglu_down_proj_weight',
    tensorKey: 'down_proj',
    label: 'W_down [8, 24] (Row-Unit-Norm, Bias-Free)',
    cx: 9.8,
    cy: 0,
    cz: 3.35,
    ex: ex24,
    ey: sheetH,
    ez: weightEz,
    gridCols: dFf,
    gridRows: d,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_NORMAL,
    slice: sliceDownProjWeight,
    values: layerWeights.downProj,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: dLabels,
    colDimLabel: `d_ff=${dFf}`,
    rowDimLabel: `d=${d}`,
    activeRow: selectedOutputDim,
    activeCol: selectedChannel,
    includeActiveGrid: false,
  });

  const sheetAccumTerms = registerSheet({
    partId: 'swiglu_accum_terms',
    tensorKey: 'down_proj_accum',
    label: `Accum Reveal (${revealedIntermediateCount}/${dFf}): M[t] * W_down[d${selectedOutputDim}]`,
    cx: 9.8,
    cy: 0,
    cz: 4.85,
    ex: ex24,
    ey: sheetH,
    ez: stripEz,
    gridCols: dFf,
    gridRows: 1,
    role: SHEET_ROLE_NEUTRAL,
    highlight: revealedIntermediateCount < dFf ? SHEET_HL_ACTIVE : SHEET_HL_PATHWAY,
    slice: sliceAccumTerms,
    values: accumInfo.contributions,
    format: 'signed',
    colLabels: ffLabels,
    rowLabels: Object.freeze([`d${selectedOutputDim}`]),
    colDimLabel: `reveal=${revealedIntermediateCount}/${dFf}`,
    rowDimLabel: '1',
    activeCol: accumColFrontier,
    revealMode: accumRevealMode,
    revealColFrontier: accumColFrontier,
    revealRowFrontier: 0,
    revealPulsePhase: accumReveal,
    extraTags: { accumReveal: revealedIntermediateCount },
  });

  // --- COL 6..7: `down_proj` Output (`24 -> 8`) + Channel-Wise nGPT NLERP Residual ---
  const isPartialAccum = revealedIntermediateCount < dFf;
  const sheetDownOut = registerSheet({
    partId: 'swiglu_down_proj_out',
    tensorKey: 'down_proj_out',
    label: 'O = M W_down^T [S, 8]',
    cx: 14.6,
    cy: 0,
    cz: 0,
    ex: ex8,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceDownProjOut,
    values: mlpTrace.downProjOut,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedOutputDim,
    activeValueOverride: isPartialAccum ? accumInfo.partialSum : null,
    activeValueTextOverride: isPartialAccum ? `Σ${fmtSigned(accumInfo.partialSum, 2)}` : null,
  });

  const sheetMlpAlphaRaw = registerSheet({
    partId: 'swiglu_mlp_alpha_raw',
    tensorKey: 'mlp_alpha',
    label: 'Raw mlp_alpha (init 0.0177)',
    cx: 18.8,
    cy: 0,
    cz: -3.2,
    ex: ex8,
    ey: sheetH,
    ez: stripEz,
    gridCols: d,
    gridRows: 1,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_NORMAL,
    slice: sliceMlpAlphaRaw,
    values: layerWeights.mlpAlpha,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: Object.freeze(['raw']),
    colDimLabel: `d=${d}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  const sheetMlpAlphaEff = registerSheet({
    partId: 'swiglu_mlp_alpha_eff',
    tensorKey: 'mlp_alpha',
    label: '|mlp_alpha * 0.05 / 0.0177| (Uses .abs())',
    cx: 18.8,
    cy: 0,
    cz: -2.55,
    ex: ex8,
    ey: sheetH,
    ez: stripEz,
    gridCols: d,
    gridRows: 1,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: sliceMlpAlphaEff,
    values: mlpTrace.nlerp.effectiveAlpha,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: Object.freeze(['|alpha|']),
    colDimLabel: `d=${d}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  const sheetNlerpOut = registerSheet({
    partId: 'swiglu_nlerp_output',
    tensorKey: 'nlerp_output',
    label: 'x_next = N_8(a + |alpha_hat|*(b-a))',
    cx: 18.8,
    cy: 0,
    cz: 0,
    ex: ex8,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_ACTIVE,
    slice: sliceNlerpOut,
    values: mlpTrace.nlerp.output,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenRow,
    activeCol: selectedOutputDim,
  });

  // Highlight selected `(selectedTokenRow, selectedChannel)` cell on product & ffn_sub_norm,
  // and `(selectedTokenRow, selectedOutputDim)` cell on down_proj_out & nlerp_output
  const activeFill = { r: 0.863, g: 0.298, b: 0.204, a: 0.18, borderAlpha: 0.9 };
  flatFlow.addCellHighlightOnFrame(
    sheetFfnSubNorm.frame,
    selectedTokenRow,
    selectedTokenRow,
    selectedChannel,
    selectedChannel,
    seqLen,
    dFf,
    0.044,
    activeFill,
    DORY_FLOW_COLORS.activeOxide,
    0.018,
    true,
  );
  flatFlow.addCellHighlightOnFrame(
    sheetDownOut.frame,
    selectedTokenRow,
    selectedTokenRow,
    selectedOutputDim,
    selectedOutputDim,
    seqLen,
    d,
    0.044,
    activeFill,
    DORY_FLOW_COLORS.activeOxide,
    0.02,
    true,
  );
  // Highlight active W_down output row `selectedOutputDim` feeding Accum Reveal
  flatFlow.addCellHighlightOnFrame(
    sheetDownWeight.frame,
    selectedOutputDim,
    selectedOutputDim,
    0,
    dFf - 1,
    d,
    dFf,
    0.042,
    { r: 0.765, g: 0.545, b: 0.145, a: 0.16, borderAlpha: 0.86 },
    DORY_FLOW_COLORS.anchorGold,
    0.018,
    true,
  );

  // ==========================================================================
  // 3. FLOW RIBBONS & DIRECTIONAL ARROWS
  // ==========================================================================
  const flowY = -0.025;

  // Fan-out h_in -> gate_pre & up_pre
  for (const [target, color] of [
    [sheetGatePre, DORY_FLOW_COLORS.preludeIndigo],
    [sheetUpPre, DORY_FLOW_COLORS.recurrentTeal],
  ] as const) {
    flatFlow.addRoundedRouteXZ(
      [
        { x: sheetHIn.frame.center[0] + ex8 * 0.5 + 0.08, z: 0 },
        { x: -12.85, z: 0 },
        { x: -12.85, z: target.frame.center[2] },
        { x: target.frame.center[0] - ex24 * 0.5 - 0.08, z: target.frame.center[2] },
      ],
      flowY,
      0.11,
      color,
      0.26,
      true,
    );
  }

  // Gate path: gate_pre -> gate_scaled -> gate_silu -> swiglu_product
  flatFlow.addPolylineXZ(
    [
      { x: sheetGatePre.frame.center[0] + ex24 * 0.5 + 0.08, z: zGate },
      { x: sheetGateScaled.frame.center[0] - ex24 * 0.5 - 0.08, z: zGate },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.preludeIndigo,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetGateScaled.frame.center[0] + ex24 * 0.5 + 0.08, z: zGate },
      { x: sheetGateSilu.frame.center[0] - ex24 * 0.5 - 0.08, z: zGate },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.preludeIndigo,
    true,
  );
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetGateSilu.frame.center[0] + ex24 * 0.5 + 0.08, z: zGate },
      { x: 2.35, z: zGate },
      { x: 2.35, z: -0.35 },
      { x: sheetProduct.frame.center[0] - ex24 * 0.5 - 0.08, z: -0.35 },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.preludeIndigo,
    0.26,
    true,
  );

  // Up path: up_pre -> up_scaled -> swiglu_product
  flatFlow.addPolylineXZ(
    [
      { x: sheetUpPre.frame.center[0] + ex24 * 0.5 + 0.08, z: zUp },
      { x: sheetUpScaled.frame.center[0] - ex24 * 0.5 - 0.08, z: zUp },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetUpScaled.frame.center[0] + ex24 * 0.5 + 0.08, z: zUp },
      { x: 2.35, z: zUp },
      { x: 2.35, z: 0.35 },
      { x: sheetProduct.frame.center[0] - ex24 * 0.5 - 0.08, z: 0.35 },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.recurrentTeal,
    0.26,
    true,
  );

  // Signed suv_gate & suv_up scale injections
  flatFlow.addPolylineXZ(
    [
      { x: sheetSuvGateRaw.frame.center[0], z: sheetSuvGateRaw.frame.center[2] + stripEz * 0.5 + 0.04 },
      { x: sheetSuvGateEff.frame.center[0], z: sheetSuvGateEff.frame.center[2] - stripEz * 0.5 - 0.04 },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetSuvGateEff.frame.center[0], z: sheetSuvGateEff.frame.center[2] + stripEz * 0.5 + 0.04 },
      { x: sheetGateScaled.frame.center[0], z: zGate - seqEz * 0.5 - 0.08 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetSuvUpRaw.frame.center[0], z: sheetSuvUpRaw.frame.center[2] - stripEz * 0.5 - 0.04 },
      { x: sheetSuvUpEff.frame.center[0], z: sheetSuvUpEff.frame.center[2] + stripEz * 0.5 + 0.04 },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetSuvUpEff.frame.center[0], z: sheetSuvUpEff.frame.center[2] - stripEz * 0.5 - 0.04 },
      { x: sheetUpScaled.frame.center[0], z: zUp + seqEz * 0.5 + 0.08 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );

  // Product -> ffn_sub_norm -> down_proj_out -> NLERP output
  flatFlow.addPolylineXZ(
    [
      { x: sheetProduct.frame.center[0] + ex24 * 0.5 + 0.08, z: 0 },
      { x: sheetFfnSubNorm.frame.center[0] - ex24 * 0.5 - 0.08, z: 0 },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetFfnSubNorm.frame.center[0] + ex24 * 0.5 + 0.08, z: 0 },
      { x: sheetDownOut.frame.center[0] - ex8 * 0.5 - 0.08, z: 0 },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetDownOut.frame.center[0] + ex8 * 0.5 + 0.08, z: 0 },
      { x: sheetNlerpOut.frame.center[0] - ex8 * 0.5 - 0.08, z: 0 },
    ],
    flowY,
    0.13,
    DORY_FLOW_COLORS.nlerpEmerald,
    true,
  );

  // Progressive Accum Reveal route: ffn_sub_norm & W_down -> accum_terms -> down_proj_out
  flatFlow.addPolylineXZ(
    [
      { x: sheetFfnSubNorm.frame.center[0], z: seqEz * 0.5 + 0.08 },
      { x: sheetDownWeight.frame.center[0], z: sheetDownWeight.frame.center[2] - weightEz * 0.5 - 0.08 },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetDownWeight.frame.center[0], z: sheetDownWeight.frame.center[2] + weightEz * 0.5 + 0.06 },
      { x: sheetAccumTerms.frame.center[0], z: sheetAccumTerms.frame.center[2] - stripEz * 0.5 - 0.06 },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.activeOxide,
    true,
  );
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetAccumTerms.frame.center[0] + ex24 * 0.5 + 0.08, z: sheetAccumTerms.frame.center[2] },
      { x: sheetDownOut.frame.center[0], z: sheetAccumTerms.frame.center[2] },
      { x: sheetDownOut.frame.center[0], z: seqEz * 0.5 + 0.08 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.activeOxide,
    0.26,
    true,
  );

  // mlp_alpha raw -> effective -> NLERP output + overhead emerald residual skip bus
  flatFlow.addPolylineXZ(
    [
      { x: sheetMlpAlphaRaw.frame.center[0], z: sheetMlpAlphaRaw.frame.center[2] + stripEz * 0.5 + 0.04 },
      { x: sheetMlpAlphaEff.frame.center[0], z: sheetMlpAlphaEff.frame.center[2] - stripEz * 0.5 - 0.04 },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetMlpAlphaEff.frame.center[0], z: sheetMlpAlphaEff.frame.center[2] + stripEz * 0.5 + 0.04 },
      { x: sheetNlerpOut.frame.center[0], z: -seqEz * 0.5 - 0.08 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetHIn.frame.center[0], z: -seqEz * 0.5 - 0.1 },
      { x: sheetHIn.frame.center[0], z: -6.75 },
      { x: sheetNlerpOut.frame.center[0] + 0.65, z: -6.75 },
      { x: sheetNlerpOut.frame.center[0] + 0.65, z: -seqEz * 0.5 - 0.1 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.nlerpEmerald,
    0.28,
    true,
  );

  ribbons.push({
    cx: 7.3,
    cy: 0.14,
    cz: 0,
    ex: 0.68,
    ey: 0.07,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_AMBER,
    alpha: 0.44,
  });
  ribbons.push({
    cx: 16.7,
    cy: 0.14,
    cz: 0,
    ex: 0.68,
    ey: 0.07,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.44,
  });

  // ==========================================================================
  // 4. OVERLAY ANCHORS, LANDMARKS, CALIPERS, BADGES & CALLOUTS
  // ==========================================================================
  const stageTone: DoryBadgeTone =
    stage === 'input'
      ? 'prelude_indigo'
      : stage === 'recurrent'
        ? 'recurrent_teal'
        : 'coda_amber';

  overlayAnchors.push(
    Object.freeze({
      id: 'anchor_swiglu_expand',
      label: `Gate / Up Expansion (d=${d} -> d_ff=${dFf})`,
      sublabel: '3.00x mini vs. 2.80x prod (2,560 -> 7,168) · Bias-Free',
      worldPos: localToWorldOnFrame(sheetGatePre.frame, 0, 0.72, -0.64),
      tone: 'prelude_indigo',
      badge: '3.00x MINI · 2.80x PROD',
    }),
    Object.freeze({
      id: 'anchor_swiglu_suv_gate',
      label: 'Signed suv_gate * sqrt(8) (NO .abs())',
      sublabel: `Init 1.0 (NOT 0.0177) · ${negativeSuvGateChannels.length} negative ch preserved`,
      worldPos: localToWorldOnFrame(sheetSuvGateRaw.frame, 0, 0.75, -0.88),
      tone: 'gold',
      badge: 'LINEAR SIGNED',
    }),
    Object.freeze({
      id: 'anchor_swiglu_suv_up',
      label: 'Signed suv_up * sqrt(8) (NO .abs())',
      sublabel: `Init 1.0 (NOT 0.0177) · ${negativeSuvUpChannels.length} negative ch preserved`,
      worldPos: localToWorldOnFrame(sheetSuvUpRaw.frame, 0, 0.75, 0.88),
      tone: 'gold',
      badge: 'LINEAR SIGNED',
    }),
    Object.freeze({
      id: 'anchor_swiglu_ffn_sub_norm',
      label: 'ffn_sub_norm = N_24(SiLU(U_g) * U_u)',
      sublabel: `||M||_2 = ${ffnSubNormL2Norm.toFixed(4)} across all d_ff=24 dims BEFORE down_proj`,
      worldPos: localToWorldOnFrame(sheetFfnSubNorm.frame, 0, 0.75, -0.64),
      tone: 'gold',
      badge: 'PRE-down_proj N_24',
    }),
    Object.freeze({
      id: 'anchor_swiglu_accum',
      label: `Progressive Accum Reveal (${revealedIntermediateCount}/${dFf})`,
      sublabel: `Out d${selectedOutputDim}: Σ=${fmtSigned(accumInfo.partialSum, 3)} / full=${fmtSigned(accumInfo.fullSum, 3)}`,
      worldPos: localToWorldOnFrame(sheetAccumTerms.frame, 0, 0.75, 0.9),
      tone: isPartialAccum ? 'spec_oxide' : 'recurrent_teal',
      badge: isPartialAccum ? 'ACCUMULATING' : 'COMPLETE',
    }),
    Object.freeze({
      id: 'anchor_swiglu_nlerp',
      label: 'nGPT NLERP Residual (NOT SLERP)',
      sublabel: `N_8(a + |mlp_alpha * 0.05/0.0177|*(b-a)) · ${DORY_NLERP_MARKER}`,
      worldPos: localToWorldOnFrame(sheetNlerpOut.frame, 0, 0.75, 0.64),
      tone: 'nlerp_emerald',
      badge: 'DR7',
    }),
  );

  landmarks.push(
    Object.freeze({
      id: 'lm_swiglu_dual_branch',
      label: 'Bias-Free SwiGLU Expansion + Signed suv',
      sublabel: `d=8 -> d_ff=24 (3.00x) · Linear Signed * sqrt(8), NO .abs()`,
      world: Object.freeze([-7.6, 0.9, -6.9] as const),
      partId: 'swiglu_suv_gate_eff',
      tone: 'gold',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_swiglu_subnorm_accum',
      label: 'ffn_sub_norm (N_24) & Accum Reveal',
      sublabel: `Pre-down_proj L2Norm -> Progressive 24->8 Dot Product`,
      world: Object.freeze([7.4, 0.9, -4.6] as const),
      partId: 'swiglu_ffn_sub_norm',
      tone: 'cyan',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_swiglu_nlerp',
      label: 'Channel-Wise nGPT NLERP Residual',
      sublabel: `Uses |mlp_alpha| (.abs(), unclamped) · NOT SLERP (DR7)`,
      world: Object.freeze([16.8, 0.9, -4.6] as const),
      partId: 'swiglu_nlerp_output',
      tone: 'default',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'caliper_swiglu_dff',
      aWorld: localToWorldOnFrame(sheetFfnSubNorm.frame, -0.5, 0.5, -0.58),
      bWorld: localToWorldOnFrame(sheetFfnSubNorm.frame, 0.5, 0.5, -0.58),
      label: `d_ff = ${dFf} (3.00x mini · prod 7,168 = 2.80x)`,
      partIds: Object.freeze(['swiglu_ffn_sub_norm', 'swiglu_product']),
    }),
    Object.freeze({
      id: 'caliper_swiglu_d',
      aWorld: localToWorldOnFrame(sheetNlerpOut.frame, -0.5, 0.5, 0.58),
      bWorld: localToWorldOnFrame(sheetNlerpOut.frame, 0.5, 0.5, 0.58),
      label: `d = ${d} (prod 2,560)`,
      partIds: Object.freeze(['swiglu_down_proj_out', 'swiglu_nlerp_output']),
    }),
  );

  const provenanceBadges: DoryProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    Object.freeze({
      label: `Phys L${physicalLayerIdx} · ${layerSpec.transformersPairBadge}`,
      tone: stageTone,
      discrepancyId: 'DR4',
    }),
    Object.freeze({
      label: `d=8 -> d_ff=24 (3.00x mini vs. 2.80x prod) · Bias-Free SwiGLU`,
      tone: 'prelude_indigo',
    }),
    Object.freeze({
      label: `Linear Signed suv_gate/suv_up (*sqrt(8), init 1.0, NO .abs() · ${negativeSuvGateChannels.length}+${negativeSuvUpChannels.length} neg ch)`,
      tone: 'gold',
      discrepancyId: 'DR9',
    }),
    Object.freeze({
      label: `ffn_sub_norm = N_24 (||M||_2 = ${ffnSubNormL2Norm.toFixed(4)}) BEFORE down_proj`,
      tone: 'recurrent_teal',
    }),
    Object.freeze({
      label: DORY_ANTI_SLERP_PLAQUE,
      tone: 'nlerp_emerald',
      discrepancyId: 'DR7',
    }),
  ];

  const callouts: DorySceneCallout[] = [
    Object.freeze({
      id: 'callout_swiglu_signed_suv',
      title: 'Family 4 SwiGLU Scales: Linear Signed (init 1.0, NO .abs(), NO sigma_0)',
      body:
        `Unlike Families 1, 2, 3, and 5 (which divide by initializer_range = 0.0177), suv_gate and suv_up initialize around 1.0 ` +
        `and multiply linearly by sqrt(d) = ${suvMultiplierMini.toFixed(4)} (prod sqrt(2560) = ${suvMultiplierProd.toFixed(2)}) ` +
        `with ZERO .abs() clamping — preserving ${negativeSuvGateChannels.length} negative gate channels and ` +
        `${negativeSuvUpChannels.length} negative up channels in Phys L${physicalLayerIdx}.`,
      tone: 'gold',
      worldPos: sheetSuvGateEff.frame.center,
      badge: 'FAMILY 4 SIGNED · NO .abs()',
      discrepancyIds: Object.freeze(['DR9'] as const),
    }),
    Object.freeze({
      id: 'callout_swiglu_ffn_sub_norm',
      title: 'Pre-down_proj ffn_sub_norm (24-Dim Unit-Sphere L2Norm)',
      body:
        `${mlpExpansionDisclosure}. Immediately after SiLU(U_g) * U_u, weightless ffn_sub_norm projects the 24-channel ` +
        `intermediate vector onto S^23 (||M||_2 = ${ffnSubNormL2Norm.toFixed(4)}, eps = 1e-6) strictly BEFORE down_proj (24 -> 8).`,
      tone: 'recurrent_teal',
      worldPos: sheetFfnSubNorm.frame.center,
      badge: 'ffn_sub_norm N_24',
      discrepancyIds: Object.freeze(['DR1'] as const),
    }),
    Object.freeze({
      id: 'callout_swiglu_accum_reveal',
      title: `Progressive down_proj Accum Reveal (${revealedIntermediateCount} / ${dFf} channels)`,
      body:
        `Fixed ascending D5 reduction (0..23) for pos ${selectedAbsTokenPos} "${selectedTokenText}", output channel d${selectedOutputDim}: ` +
        `partial sum over [0..${Math.max(0, revealedIntermediateCount - 1)}] = ${fmtSigned(accumInfo.partialSum, 4)} ` +
        `(full 24-channel dot product = ${fmtSigned(accumInfo.fullSum, 4)}).`,
      tone: isPartialAccum ? 'spec_oxide' : 'swa_cyan',
      worldPos: sheetAccumTerms.frame.center,
      badge: `REVEAL ${revealedIntermediateCount}/${dFf}`,
    }),
    Object.freeze({
      id: 'callout_swiglu_nlerp',
      title: DORY_ANTI_SLERP_PLAQUE,
      body:
        `down_proj output O re-joins the unit hypersphere via channel-wise NLERP N_8(a + |mlp_alpha * 0.05 / 0.0177| * (b - a)) ` +
        `(${DORY_NLERP_MARKER}, max |alpha_hat| = ${mlpTrace.nlerp.maxEffectiveAlpha.toFixed(3)}). ` +
        `Notice the contrast: Family 1 mlp_alpha applies .abs() (unclamped >= 0), whereas Family 4 suv_gate/suv_up stay linear signed.`,
      tone: 'nlerp_emerald',
      worldPos: sheetNlerpOut.frame.center,
      badge: DORY_NLERP_MARKER,
      discrepancyIds: Object.freeze(['DR7', 'DR9'] as const),
    }),
  ];

  // Active cell popover
  const activeCellPopover = buildSwigluHoveredPopover(
    options.hoveredCell,
    sheetsByPartId,
    mlpTrace,
    layerWeights,
    metadata,
    tokenLabels,
    accumInfo,
  );

  const stageBoundaryAabb = computeBoundsFromPoints(
    [
      [-16.4, -0.1, -7.2],
      [20.6, 1.2, 6.8],
    ],
    0.4,
  );
  const fitBounds = mergeAabbs([...aabbs, stageBoundaryAabb]);

  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flatFlow.pack();

  const loopText =
    stage === 'recurrent' ? `Loop ${loopIteration}/${ctx.kLoops}` : `${stage} (1x)`;

  const scene: DorySpatialScene = Object.freeze({
    viewMode: 'swiglu',
    title: VIEW_DESCRIPTOR.label,
    subtitle:
      `Phys L${physicalLayerIdx} (${layerSpec.transformersPairBadge}) · ${loopText} · ` +
      `d=8 -> d_ff=24 (3.00x mini / 2.80x prod) · Linear Signed suv · ffn_sub_norm · Reveal ${revealedIntermediateCount}/${dFf}`,
    modelWeightId: ctx.params.modelWeightId,
    branchId: ctx.branch.branchId,
    recordId: ctx.record.recordId,
    kLoops: ctx.kLoops,
    kvCacheMode: ctx.kvCacheMode,
    executionMode: ctx.executionMode,
    physicalLayerIdx,
    loopIteration,
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
      azimuthDeg: VIEW_DESCRIPTOR.defaultCameraAzimuthDeg,
      elevationDeg: VIEW_DESCRIPTOR.defaultCameraElevationDeg,
      fitBounds,
    }),
    fitBounds,
    valueAtlas: packedAtlas.valueAtlas,
    atlasKey: packedAtlas.atlasKey,
    atlasRegions: packedAtlas.atlasRegions,
    atlasAudit: packedAtlas.atlasAudit,
    provenanceBadges: Object.freeze(provenanceBadges),
    callouts: Object.freeze(callouts),
    overlayAnchors: Object.freeze(overlayAnchors),
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze([]),
    calipers: Object.freeze(calipers),
    activeCellGrids: Object.freeze(activeCellGrids),
    activeCellPopover,
    swigluMetadata: metadata,
  });

  // Self-audit geometry finiteness and atlas budget/non-overlap before returning
  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `View 4 swigluScene validation failed: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
