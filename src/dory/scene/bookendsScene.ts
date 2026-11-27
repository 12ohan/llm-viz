/**
 * View 7 (`bookends`): Untied Unit-Norm Bookends, 4-Format Production Readouts &
 * Strictly Read-Only TBPTL Depth-1 Spatial Scene (`src/dory/scene/bookendsScene.ts`).
 *
 * Implements DORY5 Slice 5A (Locked Decisions `D1`, `D2`, `D4`, `D5`, `D7`;
 * Guards `DR1`, `DR4`, `DR5`, `DR9`, `DR14`, `DR15`, `DR16`, `DR21`):
 *
 *   - **Zone A — Input Bookend (`backbone.embeddings.weight`, Untied `[64 x 8]`):**
 *     Displays the full `[64 x 8]` unit-norm synthetic embedding table (`||e_v||_2 ≈ 1.0`,
 *     matching `vLLM`/`llama.cpp` Layer-0 raw-embedding entry per `DR5`), the active chunk's
 *     looked-up `[seqLen x 8]` token embedding sheet, and the focused token's `[1 x 8]` vector.
 *   - **Zone B — Output Bookend (`lm_head.weight` Untied + Strictly Separated 4-Stage Pipeline):**
 *     Proves constructively that `backbone.embeddings.weight !== lm_head.weight` (`max|Δ| > 0`,
 *     `tie_word_embeddings: false`), then separates every stage cleanly in 3D space:
 *       1. Final Coda hidden state `h_out` (`[seqLen x 8]`) & untied `lm_head.weight` (`[64 x 8]`)
 *       2. Raw logits `z_raw = h_out W_lm^T` (`[1 x 64]`)
 *       3. Family 5 FP32 `lm_head.logit_scale / 0.0177` (`raw` vs. `effective` `[1 x 64]`)
 *          -> Scaled logits `z_scaled = z_raw * (s_z / 0.0177)` (`[1 x 64]`)
 *       4. Binary64 `softmax(z_scaled)` probabilities (`[1 x 64]`) + Top-12 multi-stage matrix
 *          (`[4 x 12]`) + raised 3D argmax / sampled token probability bars.
 *   - **Zone C — Read-Only 4-Checkpoint Production Pedestals (`BF16 / MXFP8 / NVFP4 / Q5_K_M`):**
 *     Proportional 3D footprint columns displaying exact `0 B` diff safetensors reconciliation
 *     (`219` unquantized BF16 tensors vs. `288` quantized matrices; `DR14`/`DR15`/`DR16`) with
 *     **ZERO fake quantization toggles**.
 *   - **Zone D — Strictly `READ-ONLY · NO GRADIENTS` TBPTL Depth-1 Schematic Pedestal:**
 *     Sculptural architectural diagram of Truncated Backprop Through Loops (`depth = 1`,
 *     stop-gradient barrier severing `s_{k-1} -> u_k`, Prelude gradients flowing exclusively via
 *     final-loop anchor `+x`, `0` early-exit losses, `0.1` ELR rank override, `4:1:1:1` GRPO RLVR).
 *   - Compact `128 x 80` (`80.00 KiB <= 128.00 KiB <= 512.00 KiB`) live `RG32F` atlas,
 *     `0` overlaps, `0` out-of-bounds, `0` NaN/Inf, `DR21` pick metadata encoding (zero edits
 *     to `src/render/pick.ts`), zero ambient RNG, and strict `D4` architectural boundary isolation.
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL execution, Node built-ins, or ambient RNG.
 */

import type { AABB } from '../../render/camera';
import type { PickBox } from '../../render/pick';
import {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_MOE_AMBER,
  CELL_NEUTRAL,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
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
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SOFTMAX,
  SHEET_ROLE_WEIGHT,
} from '../../render/sheets';
import type { AtlasSlice } from '../../scene/atlasPacker';
import { fmtSigned } from '../../shared/format';
import {
  computeSheetCellWorldPos,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
} from '../../shared/overlayLayout';
import {
  buildDoryCheckpointReadouts,
  buildDoryTbptlSchematicReadout,
  formatDoryDualBytes,
} from '../archReadouts';
import { runDoryLmHead } from '../forward';
import {
  argmaxIndex,
  effectiveLogitScale,
  softmaxFloat64,
  vectorL2Norm,
} from '../ngptMath';
import {
  DORY_CHECKPOINT_FORMATS,
  DORY_PRODUCTION_SPEC,
  DORY_SCALING_DISCLOSURE,
  computeEffectivePairedDepth,
  computeExecutedDepth,
  decodeDoryToken,
} from '../spec';
import {
  DORY_BOS_TOKEN_ID,
  DORY_END_THINK_TOKEN_ID,
  DORY_END_TOOL_CALL_TOKEN_ID,
  DORY_END_TOOL_RESPONSE_TOKEN_ID,
  DORY_EOS_TOKEN_ID,
  DORY_IM_END_TOKEN_ID,
  DORY_IM_START_TOKEN_ID,
  DORY_MINI_CONFIG,
  DORY_THINK_TOKEN_ID,
  DORY_TOOL_CALL_TOKEN_ID,
  DORY_TOOL_RESPONSE_TOKEN_ID,
  DORY_UNK_TOKEN_ID,
  type DoryForwardTrace,
  type DoryModelParameters,
  type DoryReconstructedStepResult,
  type DorySession,
} from '../types';
import type { DoryCheckpointFormatReadoutRow } from '../story/types';
import {
  DORY_FLOW_COLORS,
  DORY_TBPTL_READONLY_BADGE,
  DoryAtlasPacker,
  DoryFlatFlowBufferBuilder,
  DoryLineBufferBuilder,
  DoryRibbonBufferBuilder,
  DorySheetBufferBuilder,
  DorySolidBufferBuilder,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
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
  type DoryBookendsLogitRow,
  type DoryBookendsSceneOptions,
  type DoryBookendsViewMetadata,
  type DoryCheckpointFormatId,
  type DoryOverlayAnchor,
  type DoryPickIdentity,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DorySceneCallout,
  type DorySpatialScene,
} from './types';

// ============================================================================
// VIEW 7 CONSTANTS & PURE HELPERS
// ============================================================================

export const DORY_BOOKENDS_ATLAS_WIDTH = 128 as const;
export const DORY_BOOKENDS_ATLAS_HEIGHT = 80 as const;
export const DORY_BOOKENDS_ATLAS_BUDGET_BYTES = 131_072 as const; // 128.00 KiB <= 512.00 KiB
export const DORY_BOOKENDS_TOP_LOGIT_COUNT = 12 as const;

const VIEW_DESCRIPTOR = DORY_SPATIAL_VIEW_MODES[6]!; // 'bookends'

const SPECIAL_TOKEN_ID_SET: ReadonlySet<number> = new Set([
  DORY_UNK_TOKEN_ID,
  DORY_BOS_TOKEN_ID,
  DORY_EOS_TOKEN_ID,
  DORY_IM_START_TOKEN_ID,
  DORY_IM_END_TOKEN_ID,
  DORY_THINK_TOKEN_ID,
  DORY_END_THINK_TOKEN_ID,
  DORY_TOOL_CALL_TOKEN_ID,
  DORY_END_TOOL_CALL_TOKEN_ID,
  DORY_TOOL_RESPONSE_TOKEN_ID,
  DORY_END_TOOL_RESPONSE_TOKEN_ID,
]);

const VALID_CHECKPOINT_IDS: ReadonlySet<DoryCheckpointFormatId> = new Set([
  'bf16',
  'mxfp8',
  'nvfp4',
  'q5_k_m',
]);

function clampInt(val: number, lo: number, hi: number): number {
  if (!Number.isFinite(val)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(val)));
}

/**
 * Constructive numerical proof that `backbone.embeddings.weight` and `lm_head.weight` are
 * untied (`tie_word_embeddings: false`), returning exact `max_i |E[i] - W_lm[i]|`.
 */
export function computeUntiedBookendsMaxAbsDiff(
  embeddings: Float32Array,
  lmHead: Float32Array,
): number {
  const len = Math.min(embeddings.length, lmHead.length);
  let maxDiff = 0;
  for (let i = 0; i < len; i += 1) {
    const diff = Math.abs(embeddings[i]! - lmHead[i]!);
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }
  return maxDiff;
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

function lookupTokenEmbeddings(
  tokens: readonly number[],
  params: DoryModelParameters,
): Float32Array {
  const d = params.config.hiddenSize;
  const out = new Float32Array(tokens.length * d);
  for (let t = 0; t < tokens.length; t += 1) {
    const tok = clampInt(tokens[t] ?? 0, 0, params.config.vocabSize - 1);
    const srcOff = tok * d;
    const dstOff = t * d;
    for (let c = 0; c < d; c += 1) {
      out[dstOff + c] = params.embeddings[srcOff + c]!;
    }
  }
  return out;
}

interface ResolvedBookendsTensors {
  readonly tokens: readonly number[];
  readonly embeddings: Float32Array;
  readonly codaHidden: Float32Array;
  readonly rawLogits: Float32Array;
  readonly scaledLogits: Float32Array;
  readonly effectiveLogitScale: Float32Array;
}

function resolveBookendsTensors(
  ctx: DoryResolvedSessionContext,
  reconstructed: DoryReconstructedStepResult,
  trace: DoryForwardTrace,
): ResolvedBookendsTensors {
  const params = ctx.params;
  const tokens = resolveTraceTokens(ctx, reconstructed, trace);

  if (reconstructed.forwardResult !== null) {
    const effScale = effectiveLogitScale(params.logitScale, params.config);
    return {
      tokens,
      embeddings: reconstructed.forwardResult.embeddings,
      codaHidden: reconstructed.forwardResult.codaHidden,
      rawLogits: reconstructed.forwardResult.rawLogits,
      scaledLogits: reconstructed.forwardResult.scaledLogits,
      effectiveLogitScale: effScale,
    };
  }

  // Contract A `spec_prefill` or `spec_cycle`: recover exact embeddings and Coda final hidden state
  const embeddings = lookupTokenEmbeddings(tokens, params);
  const lastCodaRec = trace.executedSublayers[trace.executedSublayers.length - 1];
  if (lastCodaRec === undefined) {
    throw new Error(
      `View 7 bookendsScene expected non-empty executedSublayers in trace for record '${ctx.record.recordId}'.`,
    );
  }
  const codaHidden = lastCodaRec.trace.nlerp.output;
  const lmHeadRes = runDoryLmHead(codaHidden, trace.seqLen, params);
  return {
    tokens,
    embeddings,
    codaHidden,
    rawLogits: lmHeadRes.rawLogits,
    scaledLogits: lmHeadRes.scaledLogits,
    effectiveLogitScale: lmHeadRes.effectiveLogitScale,
  };
}

function sliceRowF32(src: Float32Array, rowIdx: number, rowWidth: number): Float32Array {
  const out = new Float32Array(rowWidth);
  const off = rowIdx * rowWidth;
  for (let i = 0; i < rowWidth; i += 1) {
    out[i] = src[off + i]!;
  }
  return out;
}

interface PlacedBookendSheet {
  readonly partId: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly frame: SheetFrame3D;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly values: Float32Array;
  readonly grid: ActiveSheetCellGrid;
}

// ============================================================================
// HOVERED CELL / PEDESTAL POPOVER BUILDER
// ============================================================================

function buildBookendsHoveredPopover(
  hovered: DoryPickIdentity | null | undefined,
  ctx: DoryResolvedSessionContext,
  sheetsByPartId: ReadonlyMap<string, PlacedBookendSheet>,
  checkpointRows: readonly DoryCheckpointFormatReadoutRow[],
  checkpointCenters: ReadonlyMap<DoryCheckpointFormatId, readonly [number, number, number]>,
  tbptlCenters: ReadonlyMap<string, readonly [number, number, number]>,
  params: DoryModelParameters,
  rawLogitsRow: Float32Array,
  effLogitScale: Float32Array,
  scaledLogitsRow: Float32Array,
  probsRowF32: Float32Array,
  metadata: DoryBookendsViewMetadata,
  tokenLabels: readonly string[],
): CellPopoverSpec | null {
  if (!hovered || !hovered.partId) return null;

  // Hovering the central 72-sublayer looped backbone bridge prism
  if (hovered.partId === 'bookend_backbone_bridge') {
    const executedDepth = computeExecutedDepth(ctx.kLoops, params.config);
    const effectivePaired = computeEffectivePairedDepth(ctx.kLoops, params.config);
    return Object.freeze({
      world: Object.freeze([-4.8, 0.72, -1.4] as const),
      tensorLabel: `Looped Backbone Bridge (k=${ctx.kLoops} -> ${executedDepth}L executed)`,
      coordLabel: `16 Prelude + ${ctx.kLoops}x40 Shared Core + 16 Coda · ${effectivePaired} paired · ${ctx.kvCacheMode}`,
      badge: '72 PHYSICAL SUBLAYERS · 0 LOOP ADAPTERS (D1)',
      equation: `D_sub(${ctx.kLoops}) = 32 + 40*${ctx.kLoops} = ${executedDepth}L · u_t = N(s_{t-1} + x) · ${ctx.executionMode}`,
      valueText: `pos ${metadata.selectedTokenPos} "${metadata.selectedTokenText}"`,
    });
  }

  // Hovering a 4-checkpoint production pedestal (strictly read-only, zero fake quant toggles)
  const ckptMatch = /^bookend_ckpt_(bf16|mxfp8|nvfp4|q5_k_m)$/.exec(hovered.partId);
  if (ckptMatch !== null) {
    const fmtId = ckptMatch[1] as DoryCheckpointFormatId;
    const row = checkpointRows.find((r) => r.id === fmtId);
    const center = checkpointCenters.get(fmtId);
    if (!row || !center) return null;
    return Object.freeze({
      world: Object.freeze([center[0], center[1] + 0.35, center[2]] as const),
      tensorLabel: `${row.label} · ${row.repoId}`,
      coordLabel: `SHA ${row.shortSha} · ${row.license} · ${row.zeroDiffVerified ? '0 B DIFF VERIFIED' : 'PINNED HF LFS'}`,
      badge: 'READ-ONLY METADATA · ZERO FAKE QUANT TOGGLES',
      equation:
        row.safetensorsContainerExact === true
          ? `8 + ${row.headerJsonBytes!.toLocaleString('en-US')} hdr + ${row.payloadBytes!.toLocaleString('en-US')} payload = ${row.totalFileBytes.toLocaleString('en-US')} B`
          : `${row.quantScheme} · ${row.totalFileBytes.toLocaleString('en-US')} B`,
      valueText: row.totalFormatted.dualLabel,
    });
  }

  // Hovering a READ-ONLY TBPTL depth-1 pedestal block
  if (hovered.partId.startsWith('bookend_tbptl_')) {
    const center = tbptlCenters.get(hovered.partId) ?? ([11.8, 1.0, 3.6] as const);
    const isStopGrad = hovered.partId === 'bookend_tbptl_stop_grad';
    const isAnchor = hovered.partId === 'bookend_tbptl_anchor_grad';
    return Object.freeze({
      world: Object.freeze([center[0], center[1] + 0.35, center[2]] as const),
      tensorLabel: isStopGrad
        ? 'TBPTL Depth-1 Stop-Gradient Wall (sg[s_{k-1}])'
        : isAnchor
          ? 'Final-Loop Persistent Anchor Gradient Path (+x -> Prelude)'
          : 'Dory1.0-4B Training & TBPTL Depth-1 Schematic',
      coordLabel: 'depth=1 · 0 early-exit losses · ELR 0.1 · GRPO 4:1:1:1 (alpha=0.3)',
      badge: DORY_TBPTL_READONLY_BADGE,
      equation: isStopGrad
        ? 'u_k = N(stop_grad(s_{k-1}) + x) — severs loop-to-loop BPTT'
        : isAnchor
          ? 'dL/dx flows to Prelude (0..15) solely through +x in final loop u_k'
          : '4.5T NVFP4 -> 1.5T BF16 Looped -> 300B 256K -> SFT -> GRPO RLVR',
      valueText: 'NO GRADIENTS EXECUTED',
    });
  }

  // Hovering a raised top-12 probability bar
  const barMatch = /^bookend_top_bar_(\d+)$/.exec(hovered.partId);
  if (barMatch !== null) {
    const rank = clampInt(
      Number.parseInt(barMatch[1]!, 10),
      0,
      Math.max(0, metadata.topLogitRows.length - 1),
    );
    const entry = metadata.topLogitRows[rank];
    const topSheet = sheetsByPartId.get('bookend_top12_stages');
    if (!entry || !topSheet) return null;
    const pos = computeSheetCellWorldPos(topSheet.grid, rank, 3, 0.5);
    return Object.freeze({
      world: Object.freeze([pos[0], pos[1] + 0.55, pos[2]] as const),
      tensorLabel: `Rank #${rank + 1} Token #${entry.tokenId} "${entry.tokenText}"`,
      coordLabel: `raw=${fmtSigned(entry.rawLogit, 3)} -> scaled=${fmtSigned(entry.scaledLogit, 3)}`,
      badge: entry.isArgMax
        ? 'ARGMAX · FAMILY 5 FP32 LOGIT SCALE'
        : entry.isSampled
          ? 'SAMPLED NEXT TOKEN'
          : 'SOFTMAX PROBABILITY (BINARY64)',
      equation: `${fmtSigned(entry.rawLogit, 3)} * (${fmtSigned(entry.rawLogitScale, 4)} / 0.0177) = ${fmtSigned(entry.scaledLogit, 3)}`,
      valueText: `p = ${entry.prob.toFixed(4)}`,
    });
  }

  const rec = sheetsByPartId.get(hovered.partId);
  if (!rec) return null;

  const row = clampInt(hovered.row ?? 0, 0, rec.gridRows - 1);
  const col = clampInt(hovered.col ?? 0, 0, rec.gridCols - 1);
  const world = computeSheetCellWorldPos(rec.grid, col, row, 0.5);
  const idx = row * rec.gridCols + col;
  const val = rec.values[idx] ?? 0;

  if (rec.partId === 'bookend_embed_weight') {
    const tokText = decodeDoryToken(row);
    const norm = vectorL2Norm(params.embeddings, row * DORY_MINI_CONFIG.hiddenSize, DORY_MINI_CONFIG.hiddenSize);
    return Object.freeze({
      world,
      tensorLabel: `backbone.embeddings.weight[${row}, d${col}]`,
      coordLabel: `tok #${row} "${tokText}" · ||e||_2 = ${norm.toFixed(4)} (UNTIED, DR5)`,
      badge: 'UNTIED INPUT EMBEDDING (UNIT-NORM, BF16 ON DISK)',
      equation: `Raw embedding enters Layer 0 directly (no pre-norm, DR5); prod [131072, 2560]`,
      valueText: `= ${fmtSigned(val, 4)}`,
    });
  }

  if (rec.partId === 'bookend_lm_head_weight') {
    const tokText = decodeDoryToken(row);
    const norm = vectorL2Norm(params.lmHead, row * DORY_MINI_CONFIG.hiddenSize, DORY_MINI_CONFIG.hiddenSize);
    const embVal = params.embeddings[row * DORY_MINI_CONFIG.hiddenSize + col]!;
    return Object.freeze({
      world,
      tensorLabel: `lm_head.weight[${row}, d${col}] (UNTIED)`,
      coordLabel: `tok #${row} "${tokText}" · ||w_lm||_2 = ${norm.toFixed(4)}`,
      badge: 'tie_word_embeddings: false (SEPARATE FROM EMBEDDINGS)',
      equation: `lm_head=${fmtSigned(val, 4)} vs. embed=${fmtSigned(embVal, 4)} (max|Δ|=${metadata.embeddingVsLmHeadMaxAbsDiff.toFixed(3)})`,
      valueText: `= ${fmtSigned(val, 4)}`,
    });
  }

  if (
    rec.partId === 'bookend_raw_logits' ||
    rec.partId === 'bookend_logit_scale_raw' ||
    rec.partId === 'bookend_logit_scale_eff' ||
    rec.partId === 'bookend_scaled_logits' ||
    rec.partId === 'bookend_softmax_probs'
  ) {
    const tokId = col;
    const tokText = decodeDoryToken(tokId);
    const zRaw = rawLogitsRow[tokId] ?? 0;
    const rawS = params.logitScale[tokId] ?? 0;
    const effS = effLogitScale[tokId] ?? 0;
    const zScaled = scaledLogitsRow[tokId] ?? 0;
    const prob = probsRowF32[tokId] ?? 0;
    return Object.freeze({
      world,
      tensorLabel: `${rec.label} — tok #${tokId} "${tokText}"`,
      coordLabel: `pos ${metadata.selectedTokenPos} · Family 5 (1 / 0.0177 = ${metadata.logitScaleMultiplier.toFixed(2)})`,
      badge:
        rec.partId === 'bookend_softmax_probs'
          ? 'STAGE 4 · BINARY64 SOFTMAX'
          : rec.partId === 'bookend_scaled_logits'
            ? 'STAGE 3 · FP32 SCALED LOGITS'
            : rec.partId === 'bookend_raw_logits'
              ? 'STAGE 1 · RAW LOGITS z_raw = h W_lm^T'
              : 'STAGE 2 · FAMILY 5 FP32 LOGIT SCALE (DR9/DR14)',
      equation: `${fmtSigned(zRaw, 3)} * (${fmtSigned(rawS, 4)} / 0.0177 = ${fmtSigned(effS, 3)}) = ${fmtSigned(zScaled, 3)} -> p=${prob.toFixed(4)}`,
      valueText:
        rec.partId === 'bookend_softmax_probs'
          ? `p = ${prob.toFixed(4)}`
          : `= ${fmtSigned(val, 4)}`,
    });
  }

  if (rec.partId === 'bookend_top12_stages') {
    const entry = metadata.topLogitRows[col];
    if (!entry) return null;
    const stageName =
      row === 0
        ? 'Raw Logit z_raw'
        : row === 1
          ? 'FP32 Eff Scale s_z / 0.0177'
          : row === 2
            ? 'Scaled Logit z_scaled'
            : 'Binary64 Softmax Prob';
    return Object.freeze({
      world,
      tensorLabel: `${stageName} — #${entry.tokenId} "${entry.tokenText}"`,
      coordLabel: `Rank #${col + 1}${entry.isArgMax ? ' · ARGMAX' : ''}${entry.isSampled ? ' · SAMPLED' : ''}`,
      badge: '4-STAGE UNTIED LM HEAD PIPELINE',
      equation: `z_raw(${fmtSigned(entry.rawLogit, 3)}) * s_eff(${fmtSigned(entry.effectiveLogitScale, 3)}) = ${fmtSigned(entry.scaledLogit, 3)} -> p=${entry.prob.toFixed(4)}`,
      valueText: row === 3 ? `p = ${entry.prob.toFixed(4)}` : `= ${fmtSigned(val, 4)}`,
    });
  }

  const rowLabel = tokenLabels[row] ?? `r${row}`;
  return Object.freeze({
    world,
    tensorLabel: rec.label,
    coordLabel: `${rowLabel} · ch d${col}`,
    badge: 'UNTIED BOOKEND TENSOR',
    equation: `${rec.tensorKey}[${row}, ${col}]`,
    valueText: `= ${fmtSigned(val, 4)}`,
  });
}

// ============================================================================
// MAIN VIEW 7 SCENE BUILDER (`buildDoryBookendsScene`)
// ============================================================================

/**
 * Builds the complete headless 5-pass WebGL2 spatial scene for **View 7 (`bookends`)**:
 * Untied Unit-Norm Embeddings + Untied LM Head 4-Stage Pipeline (`raw -> FP32 logit_scale/0.0177
 * -> scaled -> softmax`), `0 B` diff 4-Checkpoint Production Pedestals (`BF16 / MXFP8 / NVFP4 /
 * Q5_K_M`, zero fake quantization toggles), and strictly `READ-ONLY · NO GRADIENTS` TBPTL depth-1
 * schematic.
 *
 * Strictly read-only — never mutates `DorySession`.
 */
export function buildDoryBookendsScene(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  options: DoryBookendsSceneOptions = {},
): DorySpatialScene {
  const ctx = ensureDorySessionContext(sessionOrCtx, options.recordId);
  const { params } = ctx;
  const config = params.config;
  const d = config.hiddenSize; // 8
  const vocabSize = config.vocabSize; // 64

  const selectedCheckpointFormat: DoryCheckpointFormatId =
    options.selectedCheckpointFormat ?? 'bf16';
  if (!VALID_CHECKPOINT_IDS.has(selectedCheckpointFormat)) {
    throw new RangeError(
      `Invalid View 7 selectedCheckpointFormat: ${String(selectedCheckpointFormat)}.`,
    );
  }

  const reconstructed = ctx.resolveStepTrace();
  const trace = ctx.resolveForwardTrace();
  const seqLen = trace.seqLen;

  const bookendTensors = resolveBookendsTensors(ctx, reconstructed, trace);
  const defaultTokenOffset = ctx.resolveDefaultTokenOffset();
  const selectedTokenOffset =
    options.selectedTokenOffset !== undefined && Number.isFinite(options.selectedTokenOffset)
      ? clampInt(Math.trunc(options.selectedTokenOffset), 0, Math.max(0, seqLen - 1))
      : options.selectedTokenPos !== undefined && Number.isFinite(options.selectedTokenPos)
        ? clampInt(Math.trunc(options.selectedTokenPos) - trace.startPos, 0, Math.max(0, seqLen - 1))
        : defaultTokenOffset;
  const selectedTokenPos = trace.startPos + selectedTokenOffset;
  const selectedTokenId = clampInt(
    bookendTensors.tokens[selectedTokenOffset] ?? 0,
    0,
    vocabSize - 1,
  );
  const selectedTokenText = decodeDoryToken(selectedTokenId);

  // Extract 4-stage LM Head vectors strictly for `selectedTokenOffset`
  const rawLogitsRow = sliceRowF32(bookendTensors.rawLogits, selectedTokenOffset, vocabSize);
  const effLogitScale = bookendTensors.effectiveLogitScale;
  const scaledLogitsRow = sliceRowF32(bookendTensors.scaledLogits, selectedTokenOffset, vocabSize);
  const probsRowF64 = softmaxFloat64(scaledLogitsRow, 0, vocabSize, 1.0);
  const probsRowF32 = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v += 1) {
    probsRowF32[v] = probsRowF64[v]!;
  }

  const argMaxTokenId = argmaxIndex(scaledLogitsRow, 0, vocabSize);
  const argMaxTokenText = decodeDoryToken(argMaxTokenId);

  // Determine emitted / sampled next token at `selectedTokenPos`
  let sampledTokenId: number | null = null;
  if (selectedTokenPos === ctx.record.committedLen - 1) {
    sampledTokenId =
      ctx.record.pendingToken ??
      (ctx.record.emittedTokens.length > 0
        ? ctx.record.emittedTokens[ctx.record.emittedTokens.length - 1]!
        : null);
  } else if (selectedTokenPos + 1 < ctx.record.committedSequence.length) {
    sampledTokenId = ctx.record.committedSequence[selectedTokenPos + 1]!;
  }
  const sampledTokenText = sampledTokenId !== null ? decodeDoryToken(sampledTokenId) : null;

  const selectedVocabTokenId =
    options.selectedVocabTokenId !== undefined && options.selectedVocabTokenId !== null
      ? clampInt(options.selectedVocabTokenId, 0, vocabSize - 1)
      : argMaxTokenId;
  const selectedVocabTokenText = decodeDoryToken(selectedVocabTokenId);

  // Constructive untied verification & Family 5 logit_scale stats
  const embeddingRowL2Norm = vectorL2Norm(params.embeddings, selectedTokenId * d, d);
  const lmHeadRowL2Norm = vectorL2Norm(params.lmHead, selectedVocabTokenId * d, d);
  const embeddingVsLmHeadMaxAbsDiff = computeUntiedBookendsMaxAbsDiff(
    params.embeddings,
    params.lmHead,
  );
  if (!(embeddingVsLmHeadMaxAbsDiff > 0)) {
    throw new Error(
      'Invariant violation: expected untied backbone.embeddings.weight !== lm_head.weight (maxAbsDiff > 0).',
    );
  }

  let minEffScale = Infinity;
  let maxEffScale = -Infinity;
  for (let v = 0; v < vocabSize; v += 1) {
    const s = effLogitScale[v]!;
    if (s < minEffScale) minEffScale = s;
    if (s > maxEffScale) maxEffScale = s;
  }
  const logitScaleMultiplier = 1 / config.initializerRange; // 1 / 0.0177 ≈ 56.497175
  const logitScaleRange: readonly [number, number] = Object.freeze([minEffScale, maxEffScale]);

  // Rank top-12 vocab tokens descending by `scaledLogit` (ties broken by ascending tokenId)
  const sortedVocabIds = Array.from({ length: vocabSize }, (_, v) => v);
  sortedVocabIds.sort((a, b) => {
    const diff = scaledLogitsRow[b]! - scaledLogitsRow[a]!;
    if (diff !== 0) return diff;
    return a - b;
  });

  const topCount = Math.min(DORY_BOOKENDS_TOP_LOGIT_COUNT, vocabSize);
  const topLogitRows: DoryBookendsLogitRow[] = [];
  const top12StageMatrix = new Float32Array(4 * topCount);
  for (let rank = 0; rank < topCount; rank += 1) {
    const tokId = sortedVocabIds[rank]!;
    const rawLogit = rawLogitsRow[tokId]!;
    const rawLogitScale = params.logitScale[tokId]!;
    const effectiveScale = effLogitScale[tokId]!;
    const scaledLogit = scaledLogitsRow[tokId]!;
    const prob = probsRowF64[tokId]!;
    topLogitRows.push(
      Object.freeze({
        tokenId: tokId,
        tokenText: decodeDoryToken(tokId),
        rawLogit,
        rawLogitScale,
        effectiveLogitScale: effectiveScale,
        scaledLogit,
        prob,
        isArgMax: tokId === argMaxTokenId,
        isSampled: sampledTokenId !== null && tokId === sampledTokenId,
        isSpecialToken: SPECIAL_TOKEN_ID_SET.has(tokId),
      }),
    );
    top12StageMatrix[rank] = rawLogit;
    top12StageMatrix[topCount + rank] = effectiveScale;
    top12StageMatrix[2 * topCount + rank] = scaledLogit;
    top12StageMatrix[3 * topCount + rank] = prob;
  }

  // Reconcile 0 B diff production checkpoints & read-only TBPTL data via `archReadouts.ts`
  const checkpointRows = buildDoryCheckpointReadouts();
  const tbptlReadout = buildDoryTbptlSchematicReadout();
  const allSafetensorsZeroDiff =
    checkpointRows.filter((r) => r.safetensorsContainerExact !== null).length === 3 &&
    checkpointRows
      .filter((r) => r.safetensorsContainerExact !== null)
      .every((r) => r.zeroDiffVerified && r.safetensorsContainerExact === true);
  if (!allSafetensorsZeroDiff) {
    throw new Error('View 7 safetensors 0 B diff reconciliation invariant failed.');
  }

  const metadata: DoryBookendsViewMetadata = Object.freeze({
    readOnly: true,
    hasGradients: false,
    zeroFakeQuantToggles: true,
    tbptlBadgeLabel: DORY_TBPTL_READONLY_BADGE,
    untiedVerified: true,
    safetensorsZeroByteDiffVerified: true,
    selectedCheckpointFormat,
    seqLen,
    startPos: trace.startPos,
    selectedTokenOffset,
    selectedTokenPos,
    selectedTokenId,
    selectedTokenText,
    selectedVocabTokenId,
    selectedVocabTokenText,
    embeddingRowL2Norm,
    lmHeadRowL2Norm,
    embeddingVsLmHeadMaxAbsDiff,
    logitScaleMultiplier,
    logitScaleRange,
    argMaxTokenId,
    argMaxTokenText,
    sampledTokenId,
    sampledTokenText,
    tbptlDepth: tbptlReadout.tbptlDepth,
    tbptlStopGradientActive: tbptlReadout.stopGradientBetweenLoops,
    tbptlEarlyExitLosses: tbptlReadout.earlyExitLosses,
    tbptlPreludeGradientViaFinalAnchorOnly: true,
    topLogitRows: Object.freeze(topLogitRows),
    checkpointFormats: DORY_CHECKPOINT_FORMATS,
    scalingDisclosure: DORY_SCALING_DISCLOSURE,
  });

  // ==========================================================================
  // 1. PACK COMPACT 128 x 80 LIVE RG32F ATLAS (80.00 KiB <= 128 KiB <= 512 KiB)
  // ==========================================================================
  const atlasPacker = new DoryAtlasPacker(
    DORY_BOOKENDS_ATLAS_WIDTH,
    DORY_BOOKENDS_ATLAS_HEIGHT,
    DORY_BOOKENDS_ATLAS_BUDGET_BYTES,
  );

  const activeEmbedRow = sliceRowF32(params.embeddings, selectedTokenId, d);
  const activeLmHeadRow = sliceRowF32(params.lmHead, selectedVocabTokenId, d);

  // Shelf 1 (y = 0..63, total width = 8 + 8 + 8 + 8 + 64 + 12 + 8 + 8 = 124 <= 128)
  const sliceEmbedWeight = atlasPacker.packMatrix(
    'bookend_embed_weight',
    params.embeddings,
    vocabSize,
    d,
    { mode: 0, minFloor: 0.25 },
  );
  const sliceLmHeadWeight = atlasPacker.packMatrix(
    'bookend_lm_head_weight',
    params.lmHead,
    vocabSize,
    d,
    { mode: 0, minFloor: 0.25 },
  );
  const sliceEmbedTokens = atlasPacker.packMatrix(
    'bookend_embed_tokens',
    bookendTensors.embeddings,
    seqLen,
    d,
    { mode: 0, minFloor: 0.25 },
  );
  const sliceCodaHidden = atlasPacker.packMatrix(
    'bookend_coda_hidden',
    bookendTensors.codaHidden,
    seqLen,
    d,
    { mode: 0, minFloor: 0.25 },
  );
  const sliceRawLogits = atlasPacker.packMatrix(
    'bookend_raw_logits',
    rawLogitsRow,
    1,
    vocabSize,
    { mode: 0, minFloor: 0.25 },
  );
  const sliceTop12Stages = atlasPacker.packMatrix(
    'bookend_top12_stages',
    top12StageMatrix,
    4,
    topCount,
    { mode: 0, minFloor: 0.5 },
  );
  const sliceActiveEmbedRow = atlasPacker.packMatrix(
    'bookend_embed_active_row',
    activeEmbedRow,
    1,
    d,
    { mode: 0, minFloor: 0.25 },
  );
  const sliceActiveLmHeadRow = atlasPacker.packMatrix(
    'bookend_lm_head_active_row',
    activeLmHeadRow,
    1,
    d,
    { mode: 0, minFloor: 0.25 },
  );

  // Shelf 2 (y = 64, total width = 64 + 64 = 128 <= 128)
  const sliceLogitScaleRaw = atlasPacker.packMatrix(
    'bookend_logit_scale_raw',
    params.logitScale,
    1,
    vocabSize,
    { mode: 0, minFloor: 0.02 },
  );
  const sliceLogitScaleEff = atlasPacker.packMatrix(
    'bookend_logit_scale_eff',
    effLogitScale,
    1,
    vocabSize,
    { mode: 0, minFloor: 0.5 },
  );

  // Shelf 3 (y = 65, total width = 64 + 64 = 128 <= 128)
  const sliceScaledLogits = atlasPacker.packMatrix(
    'bookend_scaled_logits',
    scaledLogitsRow,
    1,
    vocabSize,
    { mode: 0, minFloor: 0.25 },
  );
  const sliceSoftmaxProbs = atlasPacker.packMatrix(
    'bookend_softmax_probs',
    probsRowF32,
    1,
    vocabSize,
    { mode: 1, fixedScale: 1.0 },
  );

  const atlasExtraTag = `b=${ctx.branch.branchId}:r=${ctx.record.recordId}:pos=${selectedTokenPos}:v=${selectedVocabTokenId}:ckpt=${selectedCheckpointFormat}`;
  const packedAtlas = atlasPacker.build('bookends', atlasExtraTag);

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
  const sheetsByPartId = new Map<string, PlacedBookendSheet>();
  const checkpointCenters = new Map<DoryCheckpointFormatId, readonly [number, number, number]>();
  const tbptlCenters = new Map<string, readonly [number, number, number]>();

  const dLabels = Object.freeze(Array.from({ length: d }, (_, c) => `d${c}`));
  const vocabLabels = Object.freeze(
    Array.from({ length: vocabSize }, (_, v) => `#${v}:${decodeDoryToken(v)}`),
  );
  const tokenLabels = Object.freeze(
    bookendTensors.tokens.map((tokId, i) => `p${trace.startPos + i}:${decodeDoryToken(tokId)}`),
  );
  const top12ColLabels = Object.freeze(
    topLogitRows.map((row) => `#${row.tokenId}:${row.tokenText}`),
  );
  const top12RowLabels = Object.freeze(['z_raw', 's_z/σ₀', 'z_scaled', 'softmax'] as const);

  const sheetH = 0.068;
  const ex8 = d * 0.25; // 2.00
  const weightEz64 = 5.6;
  const seqEz = Math.max(0.62, Math.min(2.8, seqLen * 0.18));
  const vocabStripEx = 6.4;
  const vocabStripEz = 0.44;
  const top12Ex = 5.2;
  const top12Ez = 1.68;

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
    readonly includeActiveGrid?: boolean;
    readonly extraTags?: Readonly<Record<string, string | number | boolean>>;
  }): PlacedBookendSheet => {
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
    });

    const meta = encodeDoryPickMetadata({
      physicalLayerIdx: null,
      bayId: 0,
      stage: 'bookend',
      loopIteration: 0,
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      tokenIndex: selectedTokenOffset,
      absTokenPos: selectedTokenPos,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      modelWeightId: ctx.params.modelWeightId,
      kvPolicy: ctx.kvCacheMode,
      extraTags: {
        view: 'bookends',
        untied: true,
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
      ...(spec.activeCol !== undefined ? { activeCol: spec.activeCol } : {}),
      ...(spec.activeRow !== undefined ? { activeRow: spec.activeRow } : {}),
      colLabels: spec.colLabels,
      rowLabels: spec.rowLabels,
      colDimLabel: spec.colDimLabel,
      rowDimLabel: spec.rowDimLabel,
    });

    if (spec.includeActiveGrid) {
      activeCellGrids.push(grid);
    }

    const placed: PlacedBookendSheet = {
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

  // --- Zone A: Input Bookend (Untied Unit-Norm Embeddings) ---
  lines.addBoxFrameXZ(-13.8, -5.8, -6.2, 5.6, 0.0, 1.0, LINE_STYLE_GUIDE, 0.42);

  const sheetEmbedWeight = registerSheet({
    partId: 'bookend_embed_weight',
    tensorKey: 'embeddings',
    label: 'backbone.embeddings.weight [64 x 8]',
    cx: -11.9,
    cy: 0.04,
    cz: -0.4,
    ex: ex8,
    ey: sheetH,
    ez: weightEz64,
    gridCols: d,
    gridRows: vocabSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_OPERAND_A,
    slice: sliceEmbedWeight,
    values: params.embeddings,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: vocabLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `V=${vocabSize}`,
    activeRow: selectedTokenId,
    activeCol: 0,
  });

  const sheetEmbedTokens = registerSheet({
    partId: 'bookend_embed_tokens',
    tensorKey: 'embed_lookup',
    label: `Token Embeddings [${seqLen} x 8] (Unit-Norm DR5)`,
    cx: -8.3,
    cy: 0.04,
    cz: -1.4,
    ex: ex8,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_ACTIVE,
    slice: sliceEmbedTokens,
    values: bookendTensors.embeddings,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenOffset,
    activeCol: 0,
    includeActiveGrid: true,
  });

  const sheetActiveEmbedRow = registerSheet({
    partId: 'bookend_embed_active_row',
    tensorKey: 'embed_active_row',
    label: `e[#${selectedTokenId} "${selectedTokenText}"] (||e||_2=${embeddingRowL2Norm.toFixed(3)})`,
    cx: -8.3,
    cy: 0.04,
    cz: 2.3,
    ex: ex8,
    ey: sheetH,
    ez: 0.42,
    gridCols: d,
    gridRows: 1,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceActiveEmbedRow,
    values: activeEmbedRow,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: Object.freeze([`#${selectedTokenId}:${selectedTokenText}`]),
    colDimLabel: `d=${d}`,
    rowDimLabel: 'unit S^7',
    activeRow: 0,
    activeCol: 0,
    includeActiveGrid: true,
  });

  // --- Backbone Bridge Prism (Prelude -> k x Core -> Coda) ---
  const executedDepth = computeExecutedDepth(ctx.kLoops, config);
  solids.push({
    cx: -4.8,
    cy: 0.24,
    cz: -1.4,
    ex: 1.5,
    ey: 0.48,
    ez: 2.2,
    cellFamily: CELL_SSM_CYAN,
    bayId: 0,
    highlight: 0.35,
  });
  lines.addBoxEdges(-4.8, 0.24, -1.4, 1.5, 0.48, 2.2, 0, 1.0, LINE_STYLE_NEUTRAL, 0.65);
  pickBoxes.push(
    packPickBoxFromAabb(
      -4.8,
      0.24,
      -1.4,
      1.5,
      0.48,
      2.2,
      encodeDoryPickMetadata({
        physicalLayerIdx: null,
        bayId: 0,
        stage: 'bookend',
        loopIteration: ctx.kLoops,
        partId: 'bookend_backbone_bridge',
        modelWeightId: ctx.params.modelWeightId,
        extraTags: { view: 'bookends', executedDepth, kLoops: ctx.kLoops },
      }),
    ),
  );

  // --- Zone B: Output Bookend — Untied LM Head & 4-Stage Pipeline ---
  lines.addBoxFrameXZ(-3.2, -5.8, 7.6, 5.6, 0.0, 1.0, LINE_STYLE_GUIDE, 0.42);

  const sheetCodaHidden = registerSheet({
    partId: 'bookend_coda_hidden',
    tensorKey: 'coda_hidden',
    label: `Coda Output h_out [${seqLen} x 8]`,
    cx: -1.5,
    cy: 0.04,
    cz: -1.4,
    ex: ex8,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceCodaHidden,
    values: bookendTensors.codaHidden,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedTokenOffset,
    activeCol: 0,
  });

  const sheetLmHeadWeight = registerSheet({
    partId: 'bookend_lm_head_weight',
    tensorKey: 'lm_head',
    label: 'lm_head.weight [64 x 8] (Untied)',
    cx: -1.5,
    cy: 0.04,
    cz: 2.35,
    ex: ex8,
    ey: sheetH,
    ez: weightEz64 * 0.72,
    gridCols: d,
    gridRows: vocabSize,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceLmHeadWeight,
    values: params.lmHead,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: vocabLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `V=${vocabSize}`,
    activeRow: selectedVocabTokenId,
    activeCol: 0,
  });

  const sheetActiveLmHeadRow = registerSheet({
    partId: 'bookend_lm_head_active_row',
    tensorKey: 'lm_head_active_row',
    label: `w_lm[#${selectedVocabTokenId} "${selectedVocabTokenText}"]`,
    cx: -1.5,
    cy: 0.04,
    cz: -4.25,
    ex: ex8,
    ey: sheetH,
    ez: 0.42,
    gridCols: d,
    gridRows: 1,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceActiveLmHeadRow,
    values: activeLmHeadRow,
    format: 'signed',
    colLabels: dLabels,
    rowLabels: Object.freeze([`#${selectedVocabTokenId}:${selectedVocabTokenText}`]),
    colDimLabel: `d=${d}`,
    rowDimLabel: 'untied row',
    activeRow: 0,
    activeCol: 0,
    includeActiveGrid: true,
  });

  // Strict 4-stage LM Head Pipeline strips stacked cleanly along Z (X = +3.6)
  const pipeX = 3.6;
  const sheetRawLogits = registerSheet({
    partId: 'bookend_raw_logits',
    tensorKey: 'raw_logits',
    label: 'Stage 1: Raw Logits z_raw = h_out W_lm^T [64]',
    cx: pipeX,
    cy: 0.04,
    cz: -4.5,
    ex: vocabStripEx,
    ey: sheetH,
    ez: vocabStripEz,
    gridCols: vocabSize,
    gridRows: 1,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_A,
    slice: sliceRawLogits,
    values: rawLogitsRow,
    format: 'signed',
    colLabels: vocabLabels,
    rowLabels: Object.freeze(['z_raw']),
    colDimLabel: `V=${vocabSize}`,
    rowDimLabel: 'raw',
    activeRow: 0,
    activeCol: selectedVocabTokenId,
  });

  const sheetLogitScaleRaw = registerSheet({
    partId: 'bookend_logit_scale_raw',
    tensorKey: 'logit_scale_raw',
    label: 'Stage 2a: Raw lm_head.logit_scale (init 0.0177) [64]',
    cx: pipeX,
    cy: 0.04,
    cz: -3.2,
    ex: vocabStripEx,
    ey: sheetH,
    ez: vocabStripEz,
    gridCols: vocabSize,
    gridRows: 1,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_NORMAL,
    slice: sliceLogitScaleRaw,
    values: params.logitScale,
    format: 'signed',
    colLabels: vocabLabels,
    rowLabels: Object.freeze(['s_z raw']),
    colDimLabel: `V=${vocabSize}`,
    rowDimLabel: 'BF16 disk',
    activeRow: 0,
    activeCol: selectedVocabTokenId,
  });

  const sheetLogitScaleEff = registerSheet({
    partId: 'bookend_logit_scale_eff',
    tensorKey: 'logit_scale_eff',
    label: 'Stage 2b: FP32 Effective Scale s_z / 0.0177 [64]',
    cx: pipeX,
    cy: 0.04,
    cz: -2.05,
    ex: vocabStripEx,
    ey: sheetH,
    ez: vocabStripEz,
    gridCols: vocabSize,
    gridRows: 1,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: sliceLogitScaleEff,
    values: effLogitScale,
    format: 'signed',
    colLabels: vocabLabels,
    rowLabels: Object.freeze(['s_z / σ₀']),
    colDimLabel: `V=${vocabSize}`,
    rowDimLabel: 'FP32 x56.5',
    activeRow: 0,
    activeCol: selectedVocabTokenId,
  });

  const sheetScaledLogits = registerSheet({
    partId: 'bookend_scaled_logits',
    tensorKey: 'scaled_logits',
    label: 'Stage 3: Scaled Logits z_scaled = z_raw * (s_z / 0.0177) [64]',
    cx: pipeX,
    cy: 0.04,
    cz: -0.75,
    ex: vocabStripEx,
    ey: sheetH,
    ez: vocabStripEz,
    gridCols: vocabSize,
    gridRows: 1,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceScaledLogits,
    values: scaledLogitsRow,
    format: 'signed',
    colLabels: vocabLabels,
    rowLabels: Object.freeze(['z_scaled']),
    colDimLabel: `V=${vocabSize}`,
    rowDimLabel: 'scaled',
    activeRow: 0,
    activeCol: argMaxTokenId,
  });

  const sheetSoftmaxProbs = registerSheet({
    partId: 'bookend_softmax_probs',
    tensorKey: 'softmax_probs',
    label: 'Stage 4: Binary64 Softmax Probabilities p(v) [64]',
    cx: pipeX,
    cy: 0.04,
    cz: 0.55,
    ex: vocabStripEx,
    ey: sheetH,
    ez: vocabStripEz,
    gridCols: vocabSize,
    gridRows: 1,
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_ACTIVE,
    slice: sliceSoftmaxProbs,
    values: probsRowF32,
    format: 'prob',
    colLabels: vocabLabels,
    rowLabels: Object.freeze(['p(v)']),
    colDimLabel: `V=${vocabSize}`,
    rowDimLabel: 'prob [0,1]',
    activeRow: 0,
    activeCol: argMaxTokenId,
  });

  // Focused Top-12 4-Stage Comparison Matrix (`[4 x 12]`) + Raised 3D Probability Bars
  const sheetTop12Stages = registerSheet({
    partId: 'bookend_top12_stages',
    tensorKey: 'top12_stages',
    label: 'Top-12 Tokens: Raw -> Scale -> Scaled -> Softmax',
    cx: pipeX,
    cy: 0.04,
    cz: 3.15,
    ex: top12Ex,
    ey: sheetH,
    ez: top12Ez,
    gridCols: topCount,
    gridRows: 4,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_ACTIVE,
    slice: sliceTop12Stages,
    values: top12StageMatrix,
    format: 'signed',
    colLabels: top12ColLabels,
    rowLabels: top12RowLabels,
    colDimLabel: `Top ${topCount}`,
    rowDimLabel: '4 stages',
    activeRow: 3,
    activeCol: 0,
    includeActiveGrid: true,
  });

  const maxTopProb = Math.max(0.05, topLogitRows[0]?.prob ?? 0.05);
  for (let rank = 0; rank < topCount; rank += 1) {
    const entry = topLogitRows[rank]!;
    const cellWorld = computeSheetCellWorldPos(sheetTop12Stages.grid, rank, 3, 0.5);
    const barH = Math.max(0.08, (entry.prob / maxTopProb) * 1.15);
    const barCy = cellWorld[1] + barH * 0.5 + 0.04;
    const barCz = cellWorld[2] + 0.56;
    const barFamily = entry.isArgMax
      ? CELL_ACTIVATION_GREEN
      : entry.isSampled
        ? CELL_MOE_AMBER
        : entry.isSpecialToken
          ? CELL_WEIGHT_BLUE
          : CELL_SSM_CYAN;

    solids.push({
      cx: cellWorld[0],
      cy: barCy,
      cz: barCz,
      ex: 0.28,
      ey: barH,
      ez: 0.24,
      cellFamily: barFamily,
      bayId: 0,
      highlight: entry.isArgMax || entry.isSampled ? 0.9 : 0.2,
    });
    if (entry.isArgMax || entry.isSampled) {
      lines.addBoxEdges(
        cellWorld[0],
        barCy,
        barCz,
        0.28,
        barH,
        0.24,
        0,
        1.1,
        entry.isArgMax ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
        0.92,
      );
    }
    pickBoxes.push(
      packPickBoxFromAabb(
        cellWorld[0],
        barCy,
        barCz,
        0.3,
        barH,
        0.28,
        encodeDoryPickMetadata({
          physicalLayerIdx: null,
          bayId: 0,
          stage: 'bookend',
          loopIteration: 0,
          partId: `bookend_top_bar_${rank}`,
          tokenIndex: rank,
          absTokenPos: selectedTokenPos,
          modelWeightId: ctx.params.modelWeightId,
          extraTags: {
            view: 'bookends',
            rank,
            vocabTokenId: entry.tokenId,
            isArgMax: entry.isArgMax,
            isSampled: entry.isSampled,
          },
        }),
      ),
    );
  }

  // Highlight selected rows on embedding & untied lm_head sheets
  flatFlow.addCellHighlightOnFrame(
    sheetEmbedWeight.frame,
    selectedTokenId,
    selectedTokenId,
    0,
    d - 1,
    vocabSize,
    d,
    0.025,
    DORY_FLOW_COLORS.preludeIndigo,
    DORY_FLOW_COLORS.activeOxide,
  );
  flatFlow.addCellHighlightOnFrame(
    sheetLmHeadWeight.frame,
    selectedVocabTokenId,
    selectedVocabTokenId,
    0,
    d - 1,
    vocabSize,
    d,
    0.025,
    DORY_FLOW_COLORS.codaAmber,
    DORY_FLOW_COLORS.activeOxide,
  );

  // --- Zone C: Read-Only 4-Checkpoint Production Footprint Pedestals (Zero Fake Quant Toggles) ---
  lines.addBoxFrameXZ(8.4, -5.8, 15.6, 0.2, 0.0, 1.0, LINE_STYLE_BF16_GOLD, 0.52);

  const bf16TotalBytes = checkpointRows[0]!.totalFileBytes; // 7,950,515,472 B
  const maxBarHeight = 2.6;
  const ckptStartX = 9.5;
  const ckptStepX = 1.65;
  const ckptZ = -2.8;

  for (let i = 0; i < checkpointRows.length; i += 1) {
    const row = checkpointRows[i]!;
    const cx = ckptStartX + i * ckptStepX;
    const isSelected = row.id === selectedCheckpointFormat;
    const totalH = Math.max(0.35, (row.totalFileBytes / bf16TotalBytes) * maxBarHeight);

    if (row.unquantizedBf16Bytes !== null && row.quantizedPayloadBytes !== null && row.quantizedPayloadBytes > 0) {
      // Split column: unquantized 219 BF16 base + quantized 288 linear matrices cap
      const baseRatio = row.unquantizedBf16Bytes / row.totalFileBytes;
      const baseH = Math.max(0.18, totalH * baseRatio);
      const quantH = Math.max(0.16, totalH - baseH);

      solids.push({
        cx,
        cy: baseH * 0.5,
        cz: ckptZ,
        ex: 1.02,
        ey: baseH,
        ez: 1.25,
        cellFamily: CELL_WEIGHT_BLUE,
        bayId: 0,
        highlight: isSelected ? 0.75 : 0.2,
      });
      solids.push({
        cx,
        cy: baseH + quantH * 0.5,
        cz: ckptZ,
        ex: 1.02,
        ey: quantH,
        ez: 1.25,
        cellFamily: row.id === 'nvfp4' ? CELL_ACTIVATION_GREEN : CELL_SSM_CYAN,
        fp8: 1,
        bayId: 0,
        highlight: isSelected ? 0.92 : 0.35,
      });
    } else {
      solids.push({
        cx,
        cy: totalH * 0.5,
        cz: ckptZ,
        ex: 1.02,
        ey: totalH,
        ez: 1.25,
        cellFamily: row.id === 'bf16' ? CELL_WEIGHT_BLUE : CELL_MOE_AMBER,
        bayId: 0,
        highlight: isSelected ? 0.9 : 0.3,
      });
    }

    lines.addBoxEdges(
      cx,
      totalH * 0.5,
      ckptZ,
      1.02,
      totalH,
      1.25,
      0,
      isSelected ? 1.4 : 0.95,
      isSelected ? LINE_STYLE_BF16_GOLD : LINE_STYLE_NEUTRAL,
      isSelected ? 0.96 : 0.65,
    );

    const topCenter: readonly [number, number, number] = Object.freeze([cx, totalH, ckptZ]);
    checkpointCenters.set(row.id, topCenter);

    pickBoxes.push(
      packPickBoxFromAabb(
        cx,
        totalH * 0.5,
        ckptZ,
        1.08,
        totalH,
        1.32,
        encodeDoryPickMetadata({
          physicalLayerIdx: null,
          bayId: 0,
          stage: 'bookend',
          loopIteration: 0,
          partId: `bookend_ckpt_${row.id}`,
          modelWeightId: ctx.params.modelWeightId,
          extraTags: {
            view: 'bookends',
            checkpoint: row.id,
            readOnly: true,
            zeroFakeQuantToggles: true,
            zeroDiff: row.zeroDiffVerified,
          },
        }),
      ),
    );
  }

  // --- Zone D: Strictly READ-ONLY · NO GRADIENTS TBPTL Depth-1 Schematic Pedestal ---
  lines.addDashedFrameXZ(8.4, 1.1, 15.6, 5.6, 0.0, LINE_STYLE_ACTIVE, 0.62);

  const tbptlZ = 3.35;
  const tbptlBlocks: readonly {
    readonly partId: string;
    readonly cx: number;
    readonly ex: number;
    readonly ey: number;
    readonly ez: number;
    readonly cellFamily: number;
    readonly style: number;
  }[] = [
    {
      partId: 'bookend_tbptl_prelude',
      cx: 9.45,
      ex: 1.1,
      ey: 0.62,
      ez: 1.35,
      cellFamily: CELL_WEIGHT_BLUE,
      style: LINE_STYLE_NEUTRAL,
    },
    {
      partId: 'bookend_tbptl_loops_prev',
      cx: 10.95,
      ex: 1.05,
      ey: 0.54,
      ez: 1.35,
      cellFamily: CELL_NEUTRAL,
      style: LINE_STYLE_GUIDE,
    },
    {
      partId: 'bookend_tbptl_stop_grad',
      cx: 11.88,
      ex: 0.22,
      ey: 1.18,
      ez: 1.75,
      cellFamily: CELL_MOE_AMBER,
      style: LINE_STYLE_ACTIVE,
    },
    {
      partId: 'bookend_tbptl_loop_final',
      cx: 12.82,
      ex: 1.05,
      ey: 0.72,
      ez: 1.35,
      cellFamily: CELL_SSM_CYAN,
      style: LINE_STYLE_ACTIVE,
    },
    {
      partId: 'bookend_tbptl_coda_loss',
      cx: 14.35,
      ex: 1.15,
      ey: 0.68,
      ez: 1.35,
      cellFamily: CELL_EMBED_SLATE,
      style: LINE_STYLE_BF16_GOLD,
    },
  ];

  for (let i = 0; i < tbptlBlocks.length; i += 1) {
    const blk = tbptlBlocks[i]!;
    const cy = blk.ey * 0.5;
    solids.push({
      cx: blk.cx,
      cy,
      cz: tbptlZ,
      ex: blk.ex,
      ey: blk.ey,
      ez: blk.ez,
      cellFamily: blk.cellFamily,
      bayId: 0,
      highlight: blk.partId === 'bookend_tbptl_stop_grad' ? 0.95 : 0.35,
    });
    lines.addBoxEdges(
      blk.cx,
      cy,
      tbptlZ,
      blk.ex,
      blk.ey,
      blk.ez,
      0,
      blk.partId === 'bookend_tbptl_stop_grad' ? 1.3 : 0.95,
      blk.style,
      0.85,
    );
    tbptlCenters.set(blk.partId, Object.freeze([blk.cx, blk.ey, tbptlZ]));
    pickBoxes.push(
      packPickBoxFromAabb(
        blk.cx,
        cy,
        tbptlZ,
        blk.ex + 0.08,
        blk.ey,
        blk.ez + 0.08,
        encodeDoryPickMetadata({
          physicalLayerIdx: null,
          bayId: 0,
          stage: 'bookend',
          loopIteration: 0,
          partId: blk.partId,
          modelWeightId: ctx.params.modelWeightId,
          extraTags: {
            view: 'bookends',
            readOnly: true,
            hasGradients: false,
            tbptlDepth: 1,
          },
        }),
      ),
    );
  }

  // Sculptural anchor-gradient return bridge (+x bypassing stop-gradient wall straight to Prelude)
  const anchorGradCenter: readonly [number, number, number] = Object.freeze([11.15, 0.22, 1.9]);
  tbptlCenters.set('bookend_tbptl_anchor_grad', anchorGradCenter);
  pickBoxes.push(
    packPickBoxFromAabb(
      anchorGradCenter[0],
      anchorGradCenter[1],
      anchorGradCenter[2],
      3.5,
      0.24,
      0.42,
      encodeDoryPickMetadata({
        physicalLayerIdx: null,
        bayId: 0,
        stage: 'bookend',
        loopIteration: 0,
        partId: 'bookend_tbptl_anchor_grad',
        modelWeightId: ctx.params.modelWeightId,
        extraTags: {
          view: 'bookends',
          readOnly: true,
          hasGradients: false,
          tbptlDepth: 1,
          anchorOnly: true,
        },
      }),
    ),
  );

  // ==========================================================================
  // 3. FLOW RIBBONS & ARCHITECTURAL BUSES
  // ==========================================================================
  const flowY = 0.095;

  // Input Bookend: embeddings[V,d] -> token lookup -> backbone bridge -> codaHidden
  flatFlow.addPolylineXZ(
    [
      { x: sheetEmbedWeight.frame.center[0] + ex8 * 0.5 + 0.08, z: -1.4 },
      { x: sheetEmbedTokens.frame.center[0] - ex8 * 0.5 - 0.08, z: -1.4 },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.preludeIndigo,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetEmbedTokens.frame.center[0], z: sheetEmbedTokens.frame.center[2] + seqEz * 0.5 + 0.08 },
      { x: sheetActiveEmbedRow.frame.center[0], z: sheetActiveEmbedRow.frame.center[2] - 0.25 },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.nlerpEmerald,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetEmbedTokens.frame.center[0] + ex8 * 0.5 + 0.08, z: -1.4 },
      { x: -5.6, z: -1.4 },
    ],
    flowY,
    0.13,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: -4.0, z: -1.4 },
      { x: sheetCodaHidden.frame.center[0] - ex8 * 0.5 - 0.08, z: -1.4 },
    ],
    flowY,
    0.13,
    DORY_FLOW_COLORS.codaAmber,
    true,
  );

  // Output Bookend: codaHidden + untied lm_head -> Stage 1 Raw -> Stage 2 Scale -> Stage 3 Scaled -> Stage 4 Softmax -> Top-12
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetCodaHidden.frame.center[0] + ex8 * 0.5 + 0.08, z: -1.4 },
      { x: 0.0, z: -1.4 },
      { x: 0.0, z: -4.5 },
      { x: sheetRawLogits.frame.center[0] - vocabStripEx * 0.5 - 0.08, z: -4.5 },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.codaAmber,
    0.24,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: pipeX, z: sheetRawLogits.frame.center[2] + vocabStripEz * 0.5 + 0.06 },
      { x: pipeX, z: sheetLogitScaleRaw.frame.center[2] - vocabStripEz * 0.5 - 0.06 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.preludeIndigo,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: pipeX, z: sheetLogitScaleRaw.frame.center[2] + vocabStripEz * 0.5 + 0.06 },
      { x: pipeX, z: sheetLogitScaleEff.frame.center[2] - vocabStripEz * 0.5 - 0.06 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: pipeX, z: sheetLogitScaleEff.frame.center[2] + vocabStripEz * 0.5 + 0.06 },
      { x: pipeX, z: sheetScaledLogits.frame.center[2] - vocabStripEz * 0.5 - 0.06 },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: pipeX, z: sheetScaledLogits.frame.center[2] + vocabStripEz * 0.5 + 0.06 },
      { x: pipeX, z: sheetSoftmaxProbs.frame.center[2] - vocabStripEz * 0.5 - 0.06 },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.nlerpEmerald,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: pipeX, z: sheetSoftmaxProbs.frame.center[2] + vocabStripEz * 0.5 + 0.06 },
      { x: pipeX, z: sheetTop12Stages.frame.center[2] - top12Ez * 0.5 - 0.08 },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.activeOxide,
    true,
  );

  // TBPTL depth-1 forward arrows + golden anchor-only gradient bypass route (strictly read-only schematic)
  flatFlow.addPolylineXZ(
    [
      { x: 10.02, z: tbptlZ },
      { x: 10.4, z: tbptlZ },
    ],
    flowY,
    0.08,
    DORY_FLOW_COLORS.ghostSlate,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: 13.38, z: tbptlZ },
      { x: 13.76, z: tbptlZ },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );
  // Golden bypass route: Final Loop u_k (+x) back to Prelude (bypassing stop-gradient wall!)
  flatFlow.addRoundedRouteXZ(
    [
      { x: 12.82, z: tbptlZ - 0.72 },
      { x: 12.82, z: 1.9 },
      { x: 9.45, z: 1.9 },
      { x: 9.45, z: tbptlZ - 0.72 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.anchorGold,
    0.22,
    true,
  );

  // Selected untied lm_head row -> Stage 1 Raw Logits
  flatFlow.addPolylineXZ(
    [
      {
        x: sheetActiveLmHeadRow.frame.center[0] + ex8 * 0.5 + 0.08,
        z: sheetActiveLmHeadRow.frame.center[2],
      },
      {
        x: sheetRawLogits.frame.center[0] - vocabStripEx * 0.5 - 0.08,
        z: sheetActiveLmHeadRow.frame.center[2],
      },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.codaAmber,
    true,
  );

  ribbons.push({
    cx: -6.4,
    cy: 0.14,
    cz: -1.4,
    ex: 0.62,
    ey: 0.07,
    ez: 0.3,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: 0.44,
  });
  ribbons.push({
    cx: -3.25,
    cy: 0.14,
    cz: -1.4,
    ex: 0.62,
    ey: 0.07,
    ez: 0.3,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_AMBER,
    alpha: 0.44,
  });
  ribbons.push({
    cx: 7.95,
    cy: 0.14,
    cz: -2.8,
    ex: 0.56,
    ey: 0.06,
    ez: 0.28,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_CYAN,
    alpha: 0.34,
  });
  ribbons.push({
    cx: 12.82,
    cy: 0.16,
    cz: 1.9,
    ex: 0.52,
    ey: 0.06,
    ez: 0.24,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.34,
  });

  // ==========================================================================
  // 4. OVERLAY ANCHORS, LANDMARKS, CALIPERS, BADGES & CALLOUTS
  // ==========================================================================
  const selectedCkptRow = checkpointRows.find((r) => r.id === selectedCheckpointFormat)!;

  overlayAnchors.push(
    Object.freeze({
      id: 'anchor_bookend_embeddings',
      label: 'Untied Input Embeddings [64 x 8]',
      sublabel: `||e[#${selectedTokenId}]||_2 = ${embeddingRowL2Norm.toFixed(4)} (DR5) · Prod [131,072 x 2,560] = 640.00 MiB BF16`,
      worldPos: localToWorldOnFrame(sheetEmbedWeight.frame, 0, 0.72, -0.58),
      tone: 'prelude_indigo',
      badge: 'UNTIED · UNIT-NORM',
    }),
    Object.freeze({
      id: 'anchor_bookend_lm_head',
      label: 'Untied LM Head [64 x 8] (tie_word_embeddings: false)',
      sublabel: `max|E - W_lm| = ${embeddingVsLmHeadMaxAbsDiff.toFixed(4)} > 0 · Prod 640.00 MiB BF16 (unquantized)`,
      worldPos: localToWorldOnFrame(sheetLmHeadWeight.frame, 0, 0.72, 0.58),
      tone: 'coda_amber',
      badge: 'UNTIED VERIFIED',
    }),
    Object.freeze({
      id: 'anchor_bookend_logit_scale',
      label: 'Family 5 FP32 lm_head.logit_scale / 0.0177',
      sublabel: `Range [${logitScaleRange[0].toFixed(3)}, ${logitScaleRange[1].toFixed(3)}] · x${logitScaleMultiplier.toFixed(2)} (FP32 in all 3 runtimes)`,
      worldPos: localToWorldOnFrame(sheetLogitScaleEff.frame, 0, 0.72, -0.7),
      tone: 'gold',
      badge: 'FAMILY 5 · DR9/DR14',
    }),
    Object.freeze({
      id: 'anchor_bookend_softmax',
      label: `Binary64 Softmax -> argmax #${argMaxTokenId} "${argMaxTokenText}"`,
      sublabel:
        sampledTokenText !== null
          ? `Sampled next: #${sampledTokenId} "${sampledTokenText}" · Top prob ${(topLogitRows[0]?.prob ?? 0).toFixed(4)}`
          : `Top prob ${(topLogitRows[0]?.prob ?? 0).toFixed(4)} · Strictly separated raw -> scaled -> prob`,
      worldPos: localToWorldOnFrame(sheetTop12Stages.frame, 0, 0.78, 0.65),
      tone: 'nlerp_emerald',
      badge: '4-STAGE HEAD',
    }),
    Object.freeze({
      id: 'anchor_bookend_checkpoints',
      label: `4-Checkpoint Production Footprints (${selectedCkptRow.label})`,
      sublabel: `${selectedCkptRow.totalFormatted.dualLabel} · 0 B diff safetensors · Zero fake quant toggles`,
      worldPos: Object.freeze([12.0, maxBarHeight + 0.45, ckptZ] as const),
      tone: 'readonly',
      badge: '0 B DIFF · READ-ONLY',
    }),
    Object.freeze({
      id: 'anchor_bookend_tbptl',
      label: 'TBPTL Depth-1 & 5-Stage Training Schematic',
      sublabel: 'Stop-gradient s_{k-1} -> u_k · Prelude grad via +x only · 0 early-exit losses',
      worldPos: Object.freeze([11.9, 1.55, tbptlZ] as const),
      tone: 'readonly',
      badge: DORY_TBPTL_READONLY_BADGE,
    }),
  );

  landmarks.push(
    Object.freeze({
      id: 'lm_bookends_untied',
      label: 'Untied Unit-Norm Bookends (E != W_lm)',
      sublabel: 'backbone.embeddings.weight & lm_head.weight kept unquantized BF16 (1.250 GiB prod)',
      world: Object.freeze([-10.1, 0.95, -5.9] as const),
      partId: 'bookend_embed_weight',
      tone: 'default',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_bookends_lm_head_pipeline',
      label: '4-Stage Scaled LM Head Pipeline',
      sublabel: 'Raw z -> FP32 s_z / 0.0177 -> Scaled z -> Binary64 Softmax -> Sample',
      world: Object.freeze([3.6, 0.95, -5.9] as const),
      partId: 'bookend_scaled_logits',
      tone: 'gold',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_bookends_checkpoints',
      label: '4-Format Production Checkpoints (0 B Diff)',
      sublabel: 'BF16 / MXFP8-PTQ / NVFP4-QAD / Q5_K_M-QAD · Read-Only (No Fake Quant Toggles)',
      world: Object.freeze([12.0, 1.05, -5.9] as const),
      partId: `bookend_ckpt_${selectedCheckpointFormat}`,
      tone: 'cyan',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_bookends_tbptl',
      label: `TBPTL Depth-1 Schematic (${DORY_TBPTL_READONLY_BADGE})`,
      sublabel: '4.5T NVFP4 -> 1.5T BF16 Looped -> 300B 256K -> SFT -> GRPO 4:1:1:1',
      world: Object.freeze([12.0, 1.05, 5.75] as const),
      partId: 'bookend_tbptl_stop_grad',
      tone: 'muted',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'caliper_bookends_untied_sep',
      aWorld: localToWorldOnFrame(sheetEmbedWeight.frame, 0.5, 0.5, 0.52),
      bWorld: localToWorldOnFrame(sheetLmHeadWeight.frame, -0.5, 0.5, 0.52),
      label: `Untied: max|E - W_lm| = ${embeddingVsLmHeadMaxAbsDiff.toFixed(3)} (tie_word_embeddings: false)`,
      partIds: Object.freeze(['bookend_embed_weight', 'bookend_lm_head_weight']),
    }),
    Object.freeze({
      id: 'caliper_bookends_vocab',
      aWorld: localToWorldOnFrame(sheetRawLogits.frame, -0.5, 0.5, -0.62),
      bWorld: localToWorldOnFrame(sheetRawLogits.frame, 0.5, 0.5, -0.62),
      label: `V = ${vocabSize} mini (11 exact special IDs) · Prod V = ${DORY_PRODUCTION_SPEC.vocabSize.toLocaleString('en-US')}`,
      partIds: Object.freeze(['bookend_raw_logits', 'bookend_scaled_logits', 'bookend_softmax_probs']),
    }),
  );

  const provenanceBadges: DoryProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    Object.freeze({
      label: `Untied Bookends Verified (max|E - W_lm| = ${embeddingVsLmHeadMaxAbsDiff.toFixed(3)})`,
      tone: 'prelude_indigo',
      discrepancyId: 'DR1',
    }),
    Object.freeze({
      label: `Family 5 FP32 logit_scale / 0.0177 (x${logitScaleMultiplier.toFixed(2)})`,
      tone: 'gold',
      discrepancyId: 'DR9',
    }),
    Object.freeze({
      label: '0 B Diff 4-Checkpoint Readouts (BF16/MXFP8/NVFP4/Q5_K_M) · Zero Fake Quant Toggles',
      tone: 'readonly',
      discrepancyId: 'DR14',
    }),
    Object.freeze({
      label: DORY_TBPTL_READONLY_BADGE,
      tone: 'readonly',
    }),
  ];

  const callouts: DorySceneCallout[] = [
    Object.freeze({
      id: 'callout_bookends_untied',
      title: 'Untied Unit-Norm Input & Output Bookends (tie_word_embeddings: false)',
      body:
        `backbone.embeddings.weight [V, d] and lm_head.weight [V, d] are strictly distinct tensors ` +
        `(max|E - W_lm| = ${embeddingVsLmHeadMaxAbsDiff.toFixed(4)} > 0). Synthetic embedding rows are unit-normalized ` +
        `(||e[#${selectedTokenId}]||_2 = ${embeddingRowL2Norm.toFixed(4)}) so vLLM/llama.cpp raw Layer-0 entry matches transformers to ~5e-7 (DR5). ` +
        `In production both [131,072, 2,560] matrices remain unquantized BF16 (640.00 MiB each).`,
      tone: 'prelude_indigo',
      worldPos: sheetEmbedWeight.frame.center,
      badge: 'UNTIED · DR1/DR5',
      discrepancyIds: Object.freeze(['DR1', 'DR5'] as const),
    }),
    Object.freeze({
      id: 'callout_bookends_logit_scale',
      title: 'Strict 4-Stage LM Head: Raw Logits -> FP32 logit_scale / 0.0177 -> Softmax',
      body:
        `Raw logits z_raw = h_out W_lm^T are multiplied element-wise in FP32 by Family 5 scale s_z / 0.0177 ` +
        `(multiplier ${logitScaleMultiplier.toFixed(2)}, effective range [${logitScaleRange[0].toFixed(3)}, ${logitScaleRange[1].toFixed(3)}]; ` +
        `FP32 in ALL 3 upstream runtimes per DR9/DR14) before binary64 softmax. At pos ${selectedTokenPos} "${selectedTokenText}", ` +
        `argmax is #${argMaxTokenId} "${argMaxTokenText}" (p=${(topLogitRows[0]?.prob ?? 0).toFixed(4)}).`,
      tone: 'gold',
      worldPos: sheetLogitScaleEff.frame.center,
      badge: 'FAMILY 5 · FP32',
      discrepancyIds: Object.freeze(['DR9', 'DR14'] as const),
    }),
    Object.freeze({
      id: 'callout_bookends_four_checkpoints',
      title: '0 B Diff 4-Checkpoint Production Readouts (Zero Fake Quant Toggles)',
      body:
        `BF16 (${formatDoryDualBytes(checkpointRows[0]!.totalFileBytes).dualLabel}), ` +
        `MXFP8-PTQ (${formatDoryDualBytes(checkpointRows[1]!.totalFileBytes).dualLabel}, config.json MSE E8M0 search DR15), ` +
        `NVFP4-QAD (${formatDoryDualBytes(checkpointRows[2]!.totalFileBytes).dualLabel}), and ` +
        `GGUF Q5_K_M (${formatDoryDualBytes(checkpointRows[3]!.totalFileBytes).dualLabel}). ` +
        `All 3 safetensors headers reconcile to exact 0 B diff vs. HF LFS with 219 unquantized BF16 tensors (1.252 GiB) and 288 quantized linear matrices. ` +
        `Governing license: OpenMDW-1.1 (DR16).`,
      tone: 'readonly',
      worldPos: Object.freeze([12.0, maxBarHeight * 0.65, ckptZ] as const),
      badge: '0 B DIFF · READ-ONLY',
      discrepancyIds: Object.freeze(['DR14', 'DR15', 'DR16'] as const),
    }),
    Object.freeze({
      id: 'callout_bookends_tbptl_readonly',
      title: `${DORY_TBPTL_READONLY_BADGE} — TBPTL Depth-1 & 5-Stage Training Concepts`,
      body:
        `${tbptlReadout.preludeGradientPath} Training spans 4.5T NVFP4 -> 1.5T BF16 Looped (0.1 ELR rank override) -> ` +
        `300B 256K -> SFT -> GRPO RLVR (4:1:1:1 difficulty curriculum, alpha = 0.3). ` +
        `Zero backpropagation or RLVR loops run in the browser.`,
      tone: 'readonly',
      worldPos: Object.freeze([11.88, 1.15, tbptlZ] as const),
      badge: DORY_TBPTL_READONLY_BADGE,
    }),
  ];

  // Active cell popover
  const activeCellPopover = buildBookendsHoveredPopover(
    options.hoveredCell,
    ctx,
    sheetsByPartId,
    checkpointRows,
    checkpointCenters,
    tbptlCenters,
    params,
    rawLogitsRow,
    effLogitScale,
    scaledLogitsRow,
    probsRowF32,
    metadata,
    tokenLabels,
  );

  const stageBoundaryAabb = computeBoundsFromPoints(
    [
      [-14.0, -0.1, -6.0],
      [15.8, maxBarHeight + 0.6, 5.9],
    ],
    0.4,
  );
  const fitBounds = mergeAabbs([...aabbs, stageBoundaryAabb]);

  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flatFlow.pack();

  const scene: DorySpatialScene = Object.freeze({
    viewMode: 'bookends',
    title: VIEW_DESCRIPTOR.label,
    subtitle:
      `Untied Bookends (pos ${selectedTokenPos} "${selectedTokenText}" -> argmax "${argMaxTokenText}") · ` +
      `FP32 logit_scale/0.0177 · 0 B Diff 4-Format Readouts (${selectedCkptRow.label}) · ${DORY_TBPTL_READONLY_BADGE}`,
    modelWeightId: ctx.params.modelWeightId,
    branchId: ctx.branch.branchId,
    recordId: ctx.record.recordId,
    kLoops: ctx.kLoops,
    kvCacheMode: ctx.kvCacheMode,
    executionMode: ctx.executionMode,
    physicalLayerIdx: null,
    loopIteration: null,
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
    bookendsMetadata: metadata,
  });

  // Self-audit geometry finiteness and atlas budget/non-overlap before returning
  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `View 7 bookendsScene validation failed: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
