/**
 * View 5 (`kv_compare`): Dual-Axis KV Cache & `last_loop` Divergence Inspector
 * (`src/dory/scene/kvCompareScene.ts`)
 *
 * Implements DORY4 Slice 4B (Locked Decisions `D1`, `D2`, `D4`, `D5`, `D6`, `D7`;
 * Guards `DR4`, `DR17`, `DR18`, `DR19`, `DR20`, `DR21`, `DR24`):
 *
 *   - **Strictly Isolated Dual-Session Comparison (`D6`, `DR18`):**
 *     Runs fresh, non-aliased `per_loop` (`16 + 20k` persistent banks) and `last_loop`
 *     (flat `36` persistent banks at any `k`) caches side by side over the identical
 *     token sequence and singleton `params` without mutating the caller's `DorySession`.
 *
 *   - **Live Prefill-Partition Divergence Probe (`Audit §7.4`, `DR17`):**
 *     Evaluates all three partition regimes (`'single_shot'`, `'chunked'`, `'one_by_one'`)
 *     on isolated `last_loop` caches against the canonical `per_loop` reference:
 *       * Theorem 1 (Empty-cache single-shot equivalence): `<= 1e-6`.
 *       * Theorem 2 (`k = 1` universal equivalence): `<= 1e-6` across all partitions.
 *       * Theorem 3 (`k >= 2` partition non-invariance): `<= 1e-6` on chunk 0, diverging
 *         materially (`> 1e-2` on non-degenerate weights) once cached final-loop history exists!
 *
 *   - **`cap_swa = 16` Ring Buffer & `absPosTag` Inspector (`DR19`, `DR24`):**
 *     Visualizes all `16` physical SWA ring slots (`pos & 15`) with live `absPosTag` and
 *     `writerLoopTag`, distinguishing active sliding-window slots (`W = 8`), rollback-retention
 *     slots evicted from `W = 8` (`[L - 16 .. L - 9]`), and empty slots (`-1`).
 *
 *   - **Live `RG32F` Causal Attention Plates (`DR20`):**
 *     Every SWA/Full score (`sheetKind = 1`) and softmax (`sheetKind = 2`) plate binds a live
 *     `RG32F` `G`-channel visibility mask (`<= 128 KiB <= 256 KiB <= 512 KiB`, `0` overlaps).
 *
 *   - **Production `1 -> 256K` BF16 Scaling Descriptor (`Audit §7.2`):**
 *     Provides exact stacked `SWA` vs. `Full` BF16 curves for `per_loop` (`kLoops`) and
 *     `last_loop` (`36` banks), exposing both the `S = 1,024` `4x` marginal slope knee and
 *     the strictly `S`-invariant `(4 + 5k) / 9` ratio.
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL execution, Node built-ins,
 * ambient RNG, or Gate 4b/4c / sibling-app imports (`D4`).
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
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_KIND_ARROW_POS_X,
  type DimensionCaliper,
  type KvDepthTickAnchor,
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
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_NEUTRAL,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
} from '../../render/sheets';
import { fmtSigned, formatBytesAdaptive, formatKiB } from '../../shared/format';
import {
  computeSheetCellWorldPos,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
} from '../../shared/overlayLayout';
import { runDoryForwardChunk } from '../forward';
import { createDoryKvCache, getDoryInputBank, getDoryOutputBank, getDoryRecurrentBank } from '../kvCache';
import {
  DORY_PHYSICAL_LAYERS,
  DORY_PRODUCTION_SPEC,
  computeExecutedLayerIdx,
  computeKvBankCounts,
  computeMiniKvCacheBytes,
  computeProductionKvBytesBf16,
  decodeDoryToken,
  getDoryPhysicalLayerSpec,
  validateDoryRecurrentLoops,
} from '../spec';
import {
  DORY_DEFAULT_SESSION_PROMPT,
  type DoryAttentionSublayerTrace,
  type DoryDualRopeTables,
  type DoryForwardTrace,
  type DoryKvBank,
  type DoryKvBankKind,
  type DoryKvCache,
  type DoryKvCacheMode,
  type DoryMiniConfig,
  type DoryModelParameters,
  type DoryProductionKvPoint,
  type DorySession,
} from '../types';
import {
  DORY_FLOW_COLORS,
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
  type DoryKvComparePartitionMode,
  type DoryKvCompareSceneOptions,
  type DoryKvCompareViewMetadata,
  type DoryKvPositionDivergencePoint,
  type DoryOverlayAnchor,
  type DoryPickIdentity,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DorySceneCallout,
  type DorySpatialScene,
} from './types';

// ============================================================================
// VIEW 5 CONSTANTS & PRODUCTION 1 -> 256K SCALING MILESTONES
// ============================================================================

export const DORY_KV_COMPARE_ATLAS_WIDTH = 128 as const;
export const DORY_KV_COMPARE_ATLAS_HEIGHT = 128 as const;
export const DORY_KV_COMPARE_ATLAS_BUDGET_BYTES = 262_144 as const; // 256.00 KiB <= 512 KiB

/**
 * Canonical context lengths spanning `1 -> 256K` (`262,144`) including every milestone
 * from Audit §7.2 (`1, 512, 1024 [SWA knee], 2048, 8192, 32768, 131072, 262144`).
 */
export const DORY_PRODUCTION_KV_SCALING_LENGTHS: readonly number[] = Object.freeze([
  1, 64, 256, 512, 1024, 2048, 4096, 8192, 32768, 65536, 131072, 262144,
]);

const VIEW_DESCRIPTOR = DORY_SPATIAL_VIEW_MODES[4]!; // 'kv_compare'

// ============================================================================
// PURE PARTITION, SCALING & ISOLATED DUAL-POLICY SIMULATION HELPERS
// ============================================================================

function clampInt(val: number, lo: number, hi: number): number {
  if (!Number.isFinite(val)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(val)));
}

/**
 * Resolves a valid canonical chunked prefill partition for `promptLen` tokens:
 *   - `promptLen === 1` -> `[1]`
 *   - `2 <= promptLen <= 4` -> `[floor(promptLen / 2), ceil(promptLen / 2)]`
 *   - `promptLen > 4` -> `[4, promptLen - 4]` (matches Audit §7.4 `[4, 8]` at `S = 12`).
 */
export function buildDefaultKvChunkPartition(promptLen: number): readonly number[] {
  if (!Number.isInteger(promptLen) || promptLen < 1) {
    throw new RangeError(`Invalid promptLen=${String(promptLen)}: expected integer >= 1.`);
  }
  if (promptLen === 1) {
    return Object.freeze([1]);
  }
  if (promptLen <= 4) {
    const first = Math.floor(promptLen / 2);
    return Object.freeze([first, promptLen - first]);
  }
  return Object.freeze([4, promptLen - 4]);
}

function validateExplicitChunkPartition(
  partition: readonly number[],
  promptLen: number,
): readonly number[] {
  if (partition.length < 1) {
    throw new RangeError('chunkPartition must contain at least one positive chunk size.');
  }
  let sum = 0;
  const out: number[] = [];
  for (let i = 0; i < partition.length; i += 1) {
    const sz = partition[i]!;
    if (!Number.isInteger(sz) || sz < 1) {
      throw new RangeError(
        `Invalid chunkPartition[${i}]=${String(sz)}: expected positive integer.`,
      );
    }
    sum += sz;
    out.push(sz);
  }
  if (sum !== promptLen) {
    throw new RangeError(
      `chunkPartition sum (${sum}) must equal comparison prompt length (${promptLen}).`,
    );
  }
  return Object.freeze(out);
}

/**
 * Generates exact analytical `1 -> 256K` BF16 production KV memory curves for both
 * `per_loop` (`16 + 20k` banks) and `last_loop` (`36` banks), preserving the exact
 * `S`-invariant `(4 + 5k) / 9` ratio across every context length (`Audit §7.2`).
 */
export function buildProductionKvScalingCurves(kLoops: number): {
  readonly perLoopCurve: readonly DoryProductionKvPoint[];
  readonly lastLoopCurve: readonly DoryProductionKvPoint[];
  readonly sInvariantRatio: number;
} {
  const k = validateDoryRecurrentLoops(kLoops);
  const perLoopCurve: DoryProductionKvPoint[] = [];
  const lastLoopCurve: DoryProductionKvPoint[] = [];
  for (let i = 0; i < DORY_PRODUCTION_KV_SCALING_LENGTHS.length; i += 1) {
    const s = DORY_PRODUCTION_KV_SCALING_LENGTHS[i]!;
    perLoopCurve.push(computeProductionKvBytesBf16(s, k, 'per_loop'));
    lastLoopCurve.push(computeProductionKvBytesBf16(s, k, 'last_loop'));
  }
  return Object.freeze({
    perLoopCurve: Object.freeze(perLoopCurve),
    lastLoopCurve: Object.freeze(lastLoopCurve),
    sInvariantRatio: (4 + 5 * k) / 9,
  });
}

interface PartitionRunOutput {
  readonly logits: Float32Array;
  readonly kvCache: DoryKvCache;
  readonly lastChunkTrace: DoryForwardTrace | null;
}

function executeIsolatedPartitionedRun(
  tokens: readonly number[],
  partition: readonly number[],
  mode: DoryKvCacheMode,
  kLoops: number,
  params: DoryModelParameters,
  ropeTables: DoryDualRopeTables,
  captureLastChunkTrace: boolean,
  branchTag: string,
): PartitionRunOutput {
  const config = params.config;
  const vocabSize = config.vocabSize;
  const kvCache = createDoryKvCache({
    mode,
    kLoops,
    includeShallowCoda: false,
    config,
  });
  const logits = new Float32Array(tokens.length * vocabSize);
  let cursor = 0;
  let lastChunkTrace: DoryForwardTrace | null = null;

  for (let c = 0; c < partition.length; c += 1) {
    const chunkLen = partition[c]!;
    const slice = tokens.slice(cursor, cursor + chunkLen);
    const isLast = c === partition.length - 1;
    const res = runDoryForwardChunk({
      tokens: slice,
      params,
      kvCache,
      startPos: cursor,
      kLoops,
      kvCacheMode: mode,
      ropeTables,
      writeKv: true,
      captureTrace: captureLastChunkTrace && isLast,
      forwardStepId: c,
      sessionBranchId: branchTag,
      role: 'standard',
    });
    logits.set(res.scaledLogits, cursor * vocabSize);
    if (isLast) {
      lastChunkTrace = res.trace;
    }
    cursor += chunkLen;
  }

  return {
    logits,
    kvCache,
    lastChunkTrace,
  };
}

function computeMaxAbsDiffAll(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let maxDiff = 0;
  for (let i = 0; i < n; i += 1) {
    const d = Math.abs(a[i]! - b[i]!);
    if (d > maxDiff) {
      maxDiff = d;
    }
  }
  return maxDiff;
}

function computePositionMaxAbsDiff(
  a: Float32Array,
  b: Float32Array,
  pos: number,
  vocabSize: number,
): number {
  const off = pos * vocabSize;
  let maxDiff = 0;
  for (let v = 0; v < vocabSize; v += 1) {
    const d = Math.abs(a[off + v]! - b[off + v]!);
    if (d > maxDiff) {
      maxDiff = d;
    }
  }
  return maxDiff;
}

function resolveComparisonTokens(
  ctx: DoryResolvedSessionContext,
  explicitTokens: readonly number[] | undefined,
  config: DoryMiniConfig,
): readonly number[] {
  const raw =
    explicitTokens !== undefined
      ? explicitTokens
      : ctx.record.committedSequence.length > 0
        ? ctx.record.committedSequence
        : ctx.branch.promptTokens.length > 0
          ? ctx.branch.promptTokens
          : DORY_DEFAULT_SESSION_PROMPT;

  if (raw.length < 1) {
    throw new RangeError('comparisonTokens must be non-empty.');
  }
  if (raw.length > config.maxPositionEmbeddings) {
    throw new RangeError(
      `comparisonTokens length (${raw.length}) exceeds maxPositionEmbeddings (${config.maxPositionEmbeddings}).`,
    );
  }
  const out: number[] = new Array<number>(raw.length);
  for (let i = 0; i < raw.length; i += 1) {
    const tok = raw[i]!;
    if (!Number.isInteger(tok) || tok < 0 || tok >= config.vocabSize) {
      throw new RangeError(
        `Invalid comparisonTokens[${i}]=${String(tok)}: expected integer in 0..${config.vocabSize - 1}.`,
      );
    }
    out[i] = tok;
  }
  return Object.freeze(out);
}

function extractAttentionSublayerTrace(
  trace: DoryForwardTrace,
  physicalLayerIdx: number,
  loopIteration: number,
  kLoops: number,
): DoryAttentionSublayerTrace {
  const execIdx = computeExecutedLayerIdx(physicalLayerIdx, loopIteration, kLoops);
  const direct = trace.executedSublayers[execIdx];
  if (
    direct !== undefined &&
    direct.coord.physicalLayerIdx === physicalLayerIdx &&
    direct.coord.loopIteration === loopIteration &&
    direct.trace.kind === 'attention'
  ) {
    return direct.trace;
  }
  const found = trace.executedSublayers.find(
    (rec) =>
      rec.coord.physicalLayerIdx === physicalLayerIdx &&
      rec.coord.loopIteration === loopIteration &&
      rec.trace.kind === 'attention',
  );
  if (found === undefined || found.trace.kind !== 'attention') {
    throw new Error(
      `Missing attention sublayer trace for physicalLayerIdx=${physicalLayerIdx}, loopIteration=${loopIteration} (kLoops=${kLoops}).`,
    );
  }
  return found.trace;
}

function resolveKvBankForLayer(
  cache: DoryKvCache,
  physicalLayerIdx: number,
  loopIteration: number,
): DoryKvBank {
  const spec = getDoryPhysicalLayerSpec(physicalLayerIdx);
  if (spec.stage === 'input') {
    return getDoryInputBank(cache, physicalLayerIdx);
  }
  if (spec.stage === 'recurrent') {
    return getDoryRecurrentBank(cache, physicalLayerIdx, Math.max(1, loopIteration));
  }
  return getDoryOutputBank(cache, physicalLayerIdx, 'output_deep');
}

function invertMaskVisibleToMasked(maskVisible: Uint8Array): Uint8Array {
  const masked = new Uint8Array(maskVisible.length);
  for (let i = 0; i < maskVisible.length; i += 1) {
    masked[i] = maskVisible[i]! > 0.5 ? 0 : 1;
  }
  return masked;
}

function sliceHeadMatrix(flat: Float32Array, headIdx: number, rows: number, cols: number): Float32Array {
  const count = rows * cols;
  const off = headIdx * count;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = flat[off + i]!;
  }
  return out;
}

function extractBankKeyMatrix(bank: DoryKvBank, kvLen: number, absPositions: Int32Array): Float32Array {
  const kvDim = bank.kvDim;
  const out = new Float32Array(kvLen * kvDim);
  const cap = bank.capacity;
  const isSwa = bank.attentionKind === 'sliding_attention';
  for (let i = 0; i < kvLen; i += 1) {
    const pos = absPositions[i]!;
    const slot = isSwa ? pos & (cap - 1) : pos;
    if (slot >= 0 && slot < cap && bank.absPosTag[slot] === pos) {
      const srcOff = slot * kvDim;
      const dstOff = i * kvDim;
      for (let c = 0; c < kvDim; c += 1) {
        out[dstOff + c] = bank.k[srcOff + c]!;
      }
    }
  }
  return out;
}

interface PlacedSheetRecord {
  readonly partId: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly frame: SheetFrame3D;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly values: Float32Array;
  readonly masked?: Uint8Array;
  readonly grid: ActiveSheetCellGrid;
}

interface SwaRingSlotInspectInfo {
  readonly slotIndex: number;
  readonly absPos: number;
  readonly writerLoop: number;
  readonly status: 'empty' | 'window_active' | 'rollback_retained';
}

function inspectSwaRingSlots(
  bank: DoryKvBank,
  promptLen: number,
  slidingWindow: number,
): readonly SwaRingSlotInspectInfo[] {
  const cap = bank.capacity;
  const winStart = Math.max(0, promptLen - slidingWindow);
  const out: SwaRingSlotInspectInfo[] = [];
  for (let s = 0; s < cap; s += 1) {
    const absPos = bank.absPosTag[s]!;
    const writerLoop = bank.writerLoopTag[s]!;
    let status: 'empty' | 'window_active' | 'rollback_retained';
    if (absPos < 0) {
      status = 'empty';
    } else if (absPos >= winStart) {
      status = 'window_active';
    } else {
      status = 'rollback_retained';
    }
    out.push(Object.freeze({ slotIndex: s, absPos, writerLoop, status }));
  }
  return Object.freeze(out);
}

// ============================================================================
// HOVERED CELL POPOVER BUILDER
// ============================================================================

function buildKvComparePopover(
  hovered: DoryPickIdentity | null | undefined,
  sheetsByPartId: ReadonlyMap<string, PlacedSheetRecord>,
  perLoopAttn: DoryAttentionSublayerTrace,
  lastLoopAttn: DoryAttentionSublayerTrace,
  swaRingInfo: readonly SwaRingSlotInspectInfo[],
  perPositionDivergence: readonly DoryKvPositionDivergencePoint[],
  kLoops: number,
  partitionMode: DoryKvComparePartitionMode,
): CellPopoverSpec | null {
  if (!hovered || !hovered.partId) return null;

  // Hovering an individual 3D SWA ring slot solid
  const slotMatch = /^swa_ring_slot_(\d+)$/.exec(hovered.partId);
  if (slotMatch !== null) {
    const s = clampInt(Number.parseInt(slotMatch[1]!, 10), 0, swaRingInfo.length - 1);
    const info = swaRingInfo[s]!;
    const tagSheet = sheetsByPartId.get('swa_ring_tags');
    const world = tagSheet
      ? computeSheetCellWorldPos(tagSheet.grid, s, 0, 0.5)
      : Object.freeze([0, 0.6, 0] as const);
    return Object.freeze({
      world: Object.freeze([world[0], world[1] + 0.55, world[2]] as const),
      tensorLabel: `SWA Ring Slot #${s} (pos & 15)`,
      coordLabel:
        info.absPos >= 0
          ? `absPosTag=${info.absPos} · writerLoop=${info.writerLoop}`
          : 'absPosTag=-1 (empty)',
      badge:
        info.status === 'window_active'
          ? 'ACTIVE SWA WINDOW (W=8)'
          : info.status === 'rollback_retained'
            ? 'ROLLBACK BUFFER (cap_swa=16, DR19)'
            : 'EMPTY SLOT',
      equation:
        info.absPos >= 0
          ? `slot = ${info.absPos} & 15 = ${s} · ${info.status === 'window_active' ? 'visible to latest query' : 'evicted from W=8, retained for O(1) rollback'}`
          : 'Unwritten physical ring slot (absPosTag = -1 rejects stale reads)',
      valueText: info.absPos >= 0 ? `pos = ${info.absPos}` : 'empty',
    });
  }

  // Hovering a 3D divergence bar
  const barMatch = /^div_bar_(\d+)$/.exec(hovered.partId);
  if (barMatch !== null) {
    const pos = clampInt(Number.parseInt(barMatch[1]!, 10), 0, perPositionDivergence.length - 1);
    const pt = perPositionDivergence[pos]!;
    const divSheet = sheetsByPartId.get('partition_divergence_matrix');
    const world = divSheet
      ? computeSheetCellWorldPos(divSheet.grid, pos, 1, 0.5)
      : Object.freeze([0, 0.6, 0] as const);
    return Object.freeze({
      world: Object.freeze([world[0], world[1] + 0.65, world[2]] as const),
      tensorLabel: `Position ${pt.absPos} "${pt.tokenText}" Divergence`,
      coordLabel: `chunk #${pt.chunkIndex} · mode=${partitionMode} · k=${kLoops}`,
      badge:
        pt.maxAbsLogitDiff > 1e-2
          ? 'THEOREM 3 DIVERGENCE (> 1e-2 · DR17)'
          : kLoops === 1
            ? 'THEOREM 2 (k=1 EQUIVALENT)'
            : 'ZERO-DIVERGENCE PREFIX',
      equation: `max_v |logit_last_loop[${pt.absPos},v] - logit_per_loop[${pt.absPos},v]|${pt.beyondSwaWindow ? ' (pos >= W=8 evicted)' : ''}`,
      valueText: `= ${pt.maxAbsLogitDiff.toExponential(3)}`,
    });
  }

  const rec = sheetsByPartId.get(hovered.partId);
  if (!rec) return null;

  const row = clampInt(hovered.row ?? 0, 0, rec.gridRows - 1);
  const col = clampInt(hovered.col ?? 0, 0, rec.gridCols - 1);
  const world = computeSheetCellWorldPos(rec.grid, col, row, 0.5);
  const idx = row * rec.gridCols + col;
  const val = rec.values[idx] ?? 0;
  const isMasked = rec.masked !== undefined && (rec.masked[idx] ?? 0) > 0.5;

  if (
    rec.partId === 'per_loop_attn_probs' ||
    rec.partId === 'per_loop_attn_scores' ||
    rec.partId === 'last_loop_attn_probs' ||
    rec.partId === 'last_loop_attn_scores'
  ) {
    const isLastLoop = rec.partId.startsWith('last_loop');
    const isScores = rec.partId.endsWith('scores');
    const attn = isLastLoop ? lastLoopAttn : perLoopAttn;
    const qPos = attn.startPos + row;
    const kPos = attn.kvAbsPositions[col] ?? col;
    const writer = attn.kvWriterLoops[col] ?? 0;
    const isPastKey = col < attn.pastLen;
    return Object.freeze({
      world,
      tensorLabel: `${isLastLoop ? 'last_loop' : 'per_loop'} ${isScores ? 'Causal Scores' : 'Causal Softmax'} (h0)`,
      coordLabel: `q_pos=${qPos}, k_pos=${kPos} · writerLoop=t${writer}${isPastKey ? ' (cached past)' : ' (current chunk)'}`,
      badge: isMasked
        ? 'DR20 RG32F G=1 MASKED'
        : isLastLoop && !attn.writeKv && isPastKey
          ? `DoryPastReader: READS FINAL LOOP t=${writer}`
          : `DR20 RG32F G=0 VISIBLE (${attn.attentionKind === 'sliding_attention' ? 'SWA W=8' : 'Full'})`,
      equation: isMasked
        ? `Masked out (${attn.attentionKind === 'sliding_attention' ? 'q - k < 0 or q - k >= 8' : 'k > q'})`
        : isScores
          ? `d_h^-0.5 <sqk^2 * N(RoPE(Q[${qPos}])), N(RoPE(K[${kPos}]^(t${writer})))>`
          : `softmax(scores)[q=${qPos}, k=${kPos}] (writeKv=${String(attn.writeKv)})`,
      valueText: isMasked ? '= -inf' : isScores ? `= ${fmtSigned(val, 4)}` : `= ${val.toFixed(4)}`,
    });
  }

  if (rec.partId === 'attn_probs_diff') {
    const qPos = lastLoopAttn.startPos + row;
    const kPos = lastLoopAttn.kvAbsPositions[col] ?? col;
    const pPl = perLoopAttn.probs[idx] ?? 0;
    const pLl = lastLoopAttn.probs[idx] ?? 0;
    return Object.freeze({
      world,
      tensorLabel: 'Attention Shift (last_loop - per_loop)',
      coordLabel: `q_pos=${qPos}, k_pos=${kPos} · w_ll=t${lastLoopAttn.kvWriterLoops[col] ?? 0} vs w_pl=t${perLoopAttn.kvWriterLoops[col] ?? 0}`,
      badge: 'DR17 HISTORICAL KV PROVENANCE SHIFT',
      equation: `${pLl.toFixed(4)} - ${pPl.toFixed(4)}`,
      valueText: `= ${fmtSigned(val, 4)}`,
    });
  }

  if (rec.partId === 'per_loop_writer_loops' || rec.partId === 'last_loop_writer_loops') {
    const isLastLoop = rec.partId.startsWith('last_loop');
    const attn = isLastLoop ? lastLoopAttn : perLoopAttn;
    const kPos = attn.kvAbsPositions[col] ?? col;
    const writer = attn.kvWriterLoops[col] ?? 0;
    const isPastKey = col < attn.pastLen;
    return Object.freeze({
      world,
      tensorLabel: `${isLastLoop ? 'last_loop' : 'per_loop'} KV Writer-Loop Provenance`,
      coordLabel: `kv_col=${col} -> absPos=${kPos} (${isPastKey ? 'historical cache' : 'current chunk'})`,
      badge: isLastLoop && isPastKey ? `FINAL-LOOP HISTORY (t=${writer})` : `SAME-LOOP KEY (t=${writer})`,
      equation: `writerLoopTag[pos=${kPos}] = ${writer}`,
      valueText: `t = ${writer}`,
    });
  }

  if (rec.partId === 'swa_ring_tags') {
    const info = swaRingInfo[col]!;
    return Object.freeze({
      world,
      tensorLabel: row === 0 ? `SWA Ring absPosTag[slot ${col}]` : `SWA Ring writerLoopTag[slot ${col}]`,
      coordLabel: `ring slot ${col} (pos & 15) · ${info.status}`,
      badge: 'cap_swa = 16 RING (DR19)',
      equation: `absPosTag=${info.absPos}, writerLoopTag=${info.writerLoop}`,
      valueText: row === 0 ? `pos = ${info.absPos}` : `loop = ${info.writerLoop}`,
    });
  }

  if (rec.partId === 'partition_divergence_matrix') {
    const modeLabel = row === 0 ? 'single_shot (Thm 1)' : row === 1 ? 'chunked (Thm 3)' : 'one_by_one (Thm 3)';
    const pt = perPositionDivergence[col];
    return Object.freeze({
      world,
      tensorLabel: `Partition Divergence: ${modeLabel}`,
      coordLabel: `absPos=${col}${pt ? ` "${pt.tokenText}"` : ''}`,
      badge: row === 0 ? 'THEOREM 1 (EMPTY CACHE = 0)' : 'THEOREM 3 (PARTITION-DEPENDENT)',
      equation: `max_v |last_loop(${modeLabel}) - per_loop| at pos ${col}`,
      valueText: `= ${val.toExponential(3)}`,
    });
  }

  return Object.freeze({
    world,
    tensorLabel: rec.label,
    coordLabel: `r=${row}, c=${col}`,
    equation: `${rec.tensorKey}[${row}, ${col}]`,
    valueText: `= ${fmtSigned(val, 4)}`,
  });
}

// ============================================================================
// MAIN VIEW 5 SCENE BUILDER (`buildDoryKvCompareScene`)
// ============================================================================

/**
 * Builds the headless 5-pass WebGL2 spatial scene for **View 5 (`kv_compare`)**:
 * Dual-Axis KV Cache & `last_loop` Divergence Inspector.
 *
 * Strictly read-only on `sessionOrCtx`: runs isolated non-aliased `per_loop` and `last_loop`
 * caches for all partition regimes without mutating any caller branch, KV cache, or LRU pool.
 */
export function buildDoryKvCompareScene(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  options: DoryKvCompareSceneOptions = {},
): DorySpatialScene {
  const ctx = ensureDorySessionContext(sessionOrCtx, options.recordId);
  const { params, ropeTables } = ctx;
  const config = params.config;
  const vocabSize = config.vocabSize;

  const kLoops =
    options.kLoopsOverride !== undefined
      ? validateDoryRecurrentLoops(Math.trunc(options.kLoopsOverride))
      : ctx.kLoops;

  const comparisonTokens = resolveComparisonTokens(ctx, options.comparisonTokens, config);
  const promptLen = comparisonTokens.length;

  const partitionMode: DoryKvComparePartitionMode = options.partitionMode ?? 'chunked';
  const chunkedSchedule =
    options.chunkPartition !== undefined
      ? validateExplicitChunkPartition(options.chunkPartition, promptLen)
      : buildDefaultKvChunkPartition(promptLen);

  const singleShotSchedule: readonly number[] = Object.freeze([promptLen]);
  const oneByOneSchedule: readonly number[] = Object.freeze(
    Array.from({ length: promptLen }, () => 1),
  );
  const activePartition: readonly number[] =
    partitionMode === 'single_shot'
      ? singleShotSchedule
      : partitionMode === 'one_by_one'
        ? oneByOneSchedule
        : chunkedSchedule;

  // Resolve focused even physical attention layer (`0, 2, ..., 70`; default `16` first Recurrent SWA)
  const rawPhys = options.physicalLayerIdx ?? 16;
  const clampedPhys = clampInt(rawPhys, 0, config.numHiddenLayers - 1);
  const selectedPhysicalLayerIdx = clampedPhys % 2 === 0 ? clampedPhys : Math.max(0, clampedPhys - 1);
  const layerSpec = getDoryPhysicalLayerSpec(selectedPhysicalLayerIdx);
  const selectedAttentionKind = layerSpec.attentionKind!;
  const selectedLoopIteration =
    layerSpec.stage === 'recurrent'
      ? clampInt(options.loopIteration ?? 1, 1, kLoops)
      : 0;

  // 1. Isolated dual-policy forward executions (ZERO mutation to caller `DorySession` — D6/DR18)
  const perLoopSingleShot = executeIsolatedPartitionedRun(
    comparisonTokens,
    singleShotSchedule,
    'per_loop',
    kLoops,
    params,
    ropeTables,
    partitionMode === 'single_shot',
    'kv_compare_per_loop_single',
  );
  const captureLastLoopSingleShotTrace =
    partitionMode === 'single_shot' ||
    (partitionMode === 'chunked' && chunkedSchedule.length === 1) ||
    (partitionMode === 'one_by_one' && promptLen === 1);

  const lastLoopSingleShot = executeIsolatedPartitionedRun(
    comparisonTokens,
    singleShotSchedule,
    'last_loop',
    kLoops,
    params,
    ropeTables,
    captureLastLoopSingleShotTrace,
    'kv_compare_last_loop_single',
  );

  const lastLoopChunked =
    chunkedSchedule.length === 1
      ? lastLoopSingleShot
      : executeIsolatedPartitionedRun(
          comparisonTokens,
          chunkedSchedule,
          'last_loop',
          kLoops,
          params,
          ropeTables,
          partitionMode === 'chunked',
          'kv_compare_last_loop_chunked',
        );

  const lastLoopOneByOne =
    promptLen === 1
      ? lastLoopSingleShot
      : executeIsolatedPartitionedRun(
          comparisonTokens,
          oneByOneSchedule,
          'last_loop',
          kLoops,
          params,
          ropeTables,
          partitionMode === 'one_by_one',
          'kv_compare_last_loop_1by1',
        );

  // Align per_loop final-chunk trace with activePartition so side-by-side causal plates match 1:1
  const perLoopActive =
    partitionMode === 'single_shot'
      ? perLoopSingleShot
      : executeIsolatedPartitionedRun(
          comparisonTokens,
          activePartition,
          'per_loop',
          kLoops,
          params,
          ropeTables,
          true,
          `kv_compare_per_loop_${partitionMode}`,
        );

  const lastLoopActive =
    partitionMode === 'single_shot'
      ? lastLoopSingleShot
      : partitionMode === 'one_by_one'
        ? lastLoopOneByOne
        : lastLoopChunked;

  const perLoopTrace = perLoopActive.lastChunkTrace!;
  const lastLoopTrace = lastLoopActive.lastChunkTrace!;
  const perLoopAttn = extractAttentionSublayerTrace(
    perLoopTrace,
    selectedPhysicalLayerIdx,
    selectedLoopIteration,
    kLoops,
  );
  const lastLoopAttn = extractAttentionSublayerTrace(
    lastLoopTrace,
    selectedPhysicalLayerIdx,
    selectedLoopIteration,
    kLoops,
  );

  // 2. Compute exact Theorem 1, Theorem 2, Theorem 3 & per-position divergence metrics
  const theorem2K1Equivalence = kLoops === 1;
  const theorem1SingleShotMaxAbsDiff = computeMaxAbsDiffAll(
    lastLoopSingleShot.logits,
    perLoopSingleShot.logits,
  );
  const theorem3ChunkedVsPerLoopMaxAbsDiff = computeMaxAbsDiffAll(
    lastLoopChunked.logits,
    perLoopSingleShot.logits,
  );
  const theorem3ChunkedVsOneByOneMaxAbsDiff = computeMaxAbsDiffAll(
    lastLoopChunked.logits,
    lastLoopOneByOne.logits,
  );
  const activePartitionVsPerLoopMaxAbsDiff = computeMaxAbsDiffAll(
    lastLoopActive.logits,
    perLoopSingleShot.logits,
  );

  // Map each absPos -> chunkIndex in activePartition
  const chunkIndexByPos = new Int32Array(promptLen);
  const chunkBoundaryPositions: number[] = [];
  {
    let posCursor = 0;
    for (let c = 0; c < activePartition.length; c += 1) {
      if (c > 0) {
        chunkBoundaryPositions.push(posCursor);
      }
      const len = activePartition[c]!;
      for (let j = 0; j < len; j += 1) {
        chunkIndexByPos[posCursor + j] = c;
      }
      posCursor += len;
    }
  }

  const perPositionDivergence: DoryKvPositionDivergencePoint[] = [];
  const divMatrix = new Float32Array(3 * promptLen);
  for (let i = 0; i < promptLen; i += 1) {
    const dSingle = computePositionMaxAbsDiff(
      lastLoopSingleShot.logits,
      perLoopSingleShot.logits,
      i,
      vocabSize,
    );
    const dChunked = computePositionMaxAbsDiff(
      lastLoopChunked.logits,
      perLoopSingleShot.logits,
      i,
      vocabSize,
    );
    const dOneByOne = computePositionMaxAbsDiff(
      lastLoopOneByOne.logits,
      perLoopSingleShot.logits,
      i,
      vocabSize,
    );
    divMatrix[i] = dSingle;
    divMatrix[promptLen + i] = dChunked;
    divMatrix[2 * promptLen + i] = dOneByOne;

    const activeDiff =
      partitionMode === 'single_shot'
        ? dSingle
        : partitionMode === 'one_by_one'
          ? dOneByOne
          : dChunked;

    const tokId = comparisonTokens[i]!;
    perPositionDivergence.push(
      Object.freeze({
        absPos: i,
        tokenId: tokId,
        tokenText: decodeDoryToken(tokId),
        chunkIndex: chunkIndexByPos[i]!,
        maxAbsLogitDiff: activeDiff,
        beyondSwaWindow: i >= config.slidingWindow,
      }),
    );
  }

  // 3. Exact bank counts, miniature payload bytes & production 1 -> 256K scaling curves
  const perLoopBankSummary = computeKvBankCounts(kLoops, 'per_loop', false);
  const lastLoopBankSummary = computeKvBankCounts(kLoops, 'last_loop', false);
  const perLoopMiniMem = computeMiniKvCacheBytes(kLoops, 'per_loop', false, config);
  const lastLoopMiniMem = computeMiniKvCacheBytes(kLoops, 'last_loop', false, config);
  const { perLoopCurve, lastLoopCurve, sInvariantRatio } = buildProductionKvScalingCurves(kLoops);

  // Inspect SWA ring buffer (`cap_swa = 16`, `absPosTag`, `writerLoopTag`) on selected SWA layer or Phys L16
  const swaRingLayerIdx =
    selectedAttentionKind === 'sliding_attention' ? selectedPhysicalLayerIdx : 16;
  const swaRingLoop =
    getDoryPhysicalLayerSpec(swaRingLayerIdx).stage === 'recurrent'
      ? Math.max(1, selectedLoopIteration)
      : 0;
  const swaRingBank = resolveKvBankForLayer(lastLoopActive.kvCache, swaRingLayerIdx, swaRingLoop);
  const swaRingInfo = inspectSwaRingSlots(swaRingBank, promptLen, config.slidingWindow);

  // 4. Pack Live RG32F Atlas (128 x 128 = 128.00 KiB <= 256 KiB <= 512 KiB, 0 overlaps)
  const atlasPacker = new DoryAtlasPacker(
    DORY_KV_COMPARE_ATLAS_WIDTH,
    DORY_KV_COMPARE_ATLAS_HEIGHT,
    DORY_KV_COMPARE_ATLAS_BUDGET_BYTES,
  );

  const seqLen = perLoopAttn.seqLen;
  const kvLen = perLoopAttn.kvLen;
  const kvDim = config.kvProjOutDim; // 16

  const perLoopProbsH0 = sliceHeadMatrix(perLoopAttn.probs, 0, seqLen, kvLen);
  const perLoopScoresH0 = sliceHeadMatrix(perLoopAttn.scores, 0, seqLen, kvLen);
  const lastLoopProbsH0 = sliceHeadMatrix(lastLoopAttn.probs, 0, seqLen, kvLen);
  const lastLoopScoresH0 = sliceHeadMatrix(lastLoopAttn.scores, 0, seqLen, kvLen);

  const attnProbsDiff = new Float32Array(seqLen * kvLen);
  for (let i = 0; i < seqLen * kvLen; i += 1) {
    attnProbsDiff[i] =
      perLoopAttn.maskVisible[i]! > 0.5 ? lastLoopProbsH0[i]! - perLoopProbsH0[i]! : 0;
  }

  const perLoopMaskedFlags = invertMaskVisibleToMasked(perLoopAttn.maskVisible);
  const lastLoopMaskedFlags = invertMaskVisibleToMasked(lastLoopAttn.maskVisible);

  // DR20: live RG32F G-channel masks packed strictly via packCausalMatrix
  const perLoopProbsSlice = atlasPacker.packCausalMatrix(
    'per_loop_attn_probs',
    perLoopProbsH0,
    perLoopAttn.maskVisible,
    seqLen,
    kvLen,
    1,
    {
      fixedScale: 1.0,
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );
  const perLoopScoresSlice = atlasPacker.packCausalMatrix(
    'per_loop_attn_scores',
    perLoopScoresH0,
    perLoopAttn.maskVisible,
    seqLen,
    kvLen,
    0,
    {
      minFloor: 0.5,
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );
  const lastLoopProbsSlice = atlasPacker.packCausalMatrix(
    'last_loop_attn_probs',
    lastLoopProbsH0,
    lastLoopAttn.maskVisible,
    seqLen,
    kvLen,
    1,
    {
      fixedScale: 1.0,
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );
  const lastLoopScoresSlice = atlasPacker.packCausalMatrix(
    'last_loop_attn_scores',
    lastLoopScoresH0,
    lastLoopAttn.maskVisible,
    seqLen,
    kvLen,
    0,
    {
      minFloor: 0.5,
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );
  const probsDiffSlice = atlasPacker.packMatrix('attn_probs_diff', attnProbsDiff, seqLen, kvLen, {
    mode: 0,
    masked: perLoopMaskedFlags,
    minFloor: 0.05,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
  });

  // Pack post-RoPE post-L2Norm Key matrices for both caches
  const perLoopBank = resolveKvBankForLayer(
    perLoopActive.kvCache,
    selectedPhysicalLayerIdx,
    selectedLoopIteration,
  );
  const lastLoopBank = resolveKvBankForLayer(
    lastLoopActive.kvCache,
    selectedPhysicalLayerIdx,
    selectedLoopIteration,
  );
  const perLoopKvK = extractBankKeyMatrix(perLoopBank, kvLen, perLoopAttn.kvAbsPositions);
  const lastLoopKvK = extractBankKeyMatrix(lastLoopBank, kvLen, lastLoopAttn.kvAbsPositions);

  const perLoopKvKSlice = atlasPacker.packMatrix('per_loop_kv_k', perLoopKvK, kvLen, kvDim, {
    mode: 0,
    minFloor: 0.25,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
  });
  const lastLoopKvKSlice = atlasPacker.packMatrix('last_loop_kv_k', lastLoopKvK, kvLen, kvDim, {
    mode: 0,
    minFloor: 0.25,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
  });

  // Pack writer-loop provenance strips [1 x kvLen]
  const perLoopWriterArr = new Float32Array(kvLen);
  const lastLoopWriterArr = new Float32Array(kvLen);
  const loopScaleDenom = Math.max(1, kLoops);
  for (let i = 0; i < kvLen; i += 1) {
    perLoopWriterArr[i] = perLoopAttn.kvWriterLoops[i]!;
    lastLoopWriterArr[i] = lastLoopAttn.kvWriterLoops[i]!;
  }
  const perLoopWriterSlice = atlasPacker.packMatrix(
    'per_loop_writer_loops',
    perLoopWriterArr,
    1,
    kvLen,
    {
      mode: 1,
      fixedScale: 1 / loopScaleDenom,
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );
  const lastLoopWriterSlice = atlasPacker.packMatrix(
    'last_loop_writer_loops',
    lastLoopWriterArr,
    1,
    kvLen,
    {
      mode: 1,
      fixedScale: 1 / loopScaleDenom,
      physicalLayerIdx: selectedPhysicalLayerIdx,
      loopIteration: selectedLoopIteration,
    },
  );

  // Pack cap_swa = 16 ring tag strip [2 x 16] (row 0 = absPosTag, row 1 = writerLoopTag)
  const swaRingTagsArr = new Float32Array(2 * config.swaRingCapacity);
  for (let s = 0; s < config.swaRingCapacity; s += 1) {
    swaRingTagsArr[s] = swaRingInfo[s]!.absPos;
    swaRingTagsArr[config.swaRingCapacity + s] = swaRingInfo[s]!.writerLoop;
  }
  const swaRingTagsSlice = atlasPacker.packMatrix(
    'swa_ring_tags',
    swaRingTagsArr,
    2,
    config.swaRingCapacity,
    {
      mode: 0,
      minFloor: Math.max(4, promptLen),
      physicalLayerIdx: swaRingLayerIdx,
      loopIteration: swaRingLoop,
    },
  );

  // Pack 3 x promptLen partition divergence matrix [single_shot, chunked, one_by_one]
  const maxDivForScale = Math.max(
    0.05,
    theorem1SingleShotMaxAbsDiff,
    theorem3ChunkedVsPerLoopMaxAbsDiff,
    theorem3ChunkedVsOneByOneMaxAbsDiff,
    activePartitionVsPerLoopMaxAbsDiff,
  );
  const divMatrixSlice = atlasPacker.packMatrix(
    'partition_divergence_matrix',
    divMatrix,
    3,
    promptLen,
    {
      mode: 1,
      fixedScale: 1 / maxDivForScale,
    },
  );

  const atlasExtraTag = `k=${kLoops}:mode=${partitionMode}:part=${activePartition.join('-')}:p=${selectedPhysicalLayerIdx}:t=${selectedLoopIteration}:S=${promptLen}`;
  const packedAtlas = atlasPacker.build('kv_compare', atlasExtraTag);

  // 5. Build 5-Pass 3D Editorial Drafting-Table Scene Geometry
  const solids = new DorySolidBufferBuilder();
  const lines = new DoryLineBufferBuilder();
  const ribbons = new DoryRibbonBufferBuilder();
  const sheets = new DorySheetBufferBuilder();
  const flatFlow = new DoryFlatFlowBufferBuilder();

  const pickBoxes: PickBox[] = [];
  const aabbs: AABB[] = [];
  const overlayAnchors: DoryOverlayAnchor[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const kvDepthTicks: KvDepthTickAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const sheetsByPartId = new Map<string, PlacedSheetRecord>();

  let ownerCounter = 1;

  const queryLabels = Object.freeze(
    Array.from({ length: seqLen }, (_, r) => {
      const pos = perLoopAttn.startPos + r;
      return `q${pos}:${decodeDoryToken(comparisonTokens[pos] ?? 0)}`;
    }),
  );
  const keyLabels = Object.freeze(
    Array.from({ length: kvLen }, (_, c) => {
      const pos = perLoopAttn.kvAbsPositions[c] ?? c;
      return `k${pos}`;
    }),
  );
  const kvDimLabels = Object.freeze(Array.from({ length: kvDim }, (_, c) => `kv${c}`));
  const ringSlotLabels = Object.freeze(
    Array.from({ length: config.swaRingCapacity }, (_, s) => `s${s}`),
  );
  const promptPosLabels = Object.freeze(
    Array.from({ length: promptLen }, (_, i) => `p${i}`),
  );

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
    readonly sheetKind: number;
    readonly role: number;
    readonly highlight: number;
    readonly slice: typeof perLoopProbsSlice;
    readonly values: Float32Array;
    readonly masked?: Uint8Array;
    readonly format: 'signed' | 'prob';
    readonly colLabels: readonly string[];
    readonly rowLabels: readonly string[];
    readonly colDimLabel: string;
    readonly rowDimLabel: string;
    readonly kvPolicy: DoryKvCacheMode | null;
    readonly kvBankSlotId?: number | null;
    readonly kvBankKind?: DoryKvBankKind | null;
    readonly physicalLayerIdx?: number | null;
    readonly loopIteration?: number | null;
    readonly includeInCellGrid?: boolean;
    readonly extraTags?: Readonly<Record<string, string | number | boolean>>;
  }): PlacedSheetRecord => {
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
      sheetKind: spec.sheetKind,
      role: spec.role,
      highlight: spec.highlight,
      slice: spec.slice,
      frame,
    });

    const meta = encodeDoryPickMetadata({
      physicalLayerIdx:
        spec.physicalLayerIdx !== undefined ? spec.physicalLayerIdx : selectedPhysicalLayerIdx,
      loopIteration:
        spec.loopIteration !== undefined ? spec.loopIteration : selectedLoopIteration,
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      kvBankSlotId: spec.kvBankSlotId ?? null,
      kvBankKind: spec.kvBankKind ?? null,
      kvPolicy: spec.kvPolicy,
      cacheRole: 'standard',
      modelWeightId: params.modelWeightId,
      extraTags: {
        view: 'kv_compare',
        partition: partitionMode,
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
      ...(spec.masked !== undefined ? { masked: spec.masked } : {}),
      format: spec.format,
      colLabels: spec.colLabels,
      rowLabels: spec.rowLabels,
      colDimLabel: spec.colDimLabel,
      rowDimLabel: spec.rowDimLabel,
    });
    if (spec.includeInCellGrid !== false) {
      activeCellGrids.push(grid);
    }

    const placed: PlacedSheetRecord = {
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      label: spec.label,
      frame,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      values: spec.values,
      ...(spec.masked !== undefined ? { masked: spec.masked } : {}),
      grid,
    };
    sheetsByPartId.set(spec.partId, placed);
    return placed;
  };

  // Layout parameters
  const sheetH = 0.068;
  const plateEx = Math.max(2.4, Math.min(4.6, kvLen * 0.24));
  const plateEz = Math.max(1.2, Math.min(3.6, seqLen * 0.24));
  const writerEz = 0.42;
  const kvKEx = 2.6;
  const kvKEz = Math.max(1.2, Math.min(3.4, kvLen * 0.16));

  // Helper: emit compact 3D KV Bank Matrix for either `per_loop` (`16+20k`) or `last_loop` (`36`)
  const emitBankMatrix = (
    cache: DoryKvCache,
    originX: number,
    originZ: number,
    policy: DoryKvCacheMode,
  ): { readonly minX: number; readonly maxX: number; readonly minZ: number; readonly maxZ: number } => {
    const colCount = policy === 'per_loop' ? kLoops : 1;
    const colPitchX = colCount > 4 ? 0.46 : 0.62;
    const rowPitchZ = 0.22;
    const bankEx = colPitchX * 0.72;
    const bankEz = 0.15;

    let minX = Infinity;
    let maxX = -Infinity;
    let minZ = Infinity;
    let maxZ = -Infinity;

    for (let b = 0; b < cache.allBanks.length; b += 1) {
      const bank = cache.allBanks[b]!;
      const layer = DORY_PHYSICAL_LAYERS[bank.physicalLayerIdx]!;
      const ord = layer.stageAttentionOrdinal ?? 0;
      const isFull = bank.attentionKind === 'full_attention';

      let colIdx = 0;
      let rowIdx = 0;
      if (bank.bankKind === 'input') {
        colIdx = 0;
        rowIdx = ord; // 0..7
      } else if (bank.bankKind === 'recurrent') {
        colIdx = policy === 'per_loop' ? Math.max(0, bank.loopSlot - 1) : 0;
        rowIdx = 9 + ord; // 9..28
      } else {
        colIdx = policy === 'per_loop' ? Math.max(0, colCount - 1) : 0;
        rowIdx = 30 + ord; // 30..37
      }

      const cx = originX + colIdx * colPitchX;
      const cz = originZ + rowIdx * rowPitchZ;
      const ey = isFull ? 0.34 : 0.22;
      const cy = ey * 0.5;
      const isFocused =
        bank.physicalLayerIdx === selectedPhysicalLayerIdx &&
        (bank.bankKind !== 'recurrent' ||
          policy === 'last_loop' ||
          bank.loopSlot === selectedLoopIteration);

      const cellFamily = isFull
        ? CELL_WEIGHT_BLUE
        : bank.bankKind === 'input'
          ? CELL_SSM_CYAN
          : bank.bankKind === 'output_deep'
            ? CELL_MOE_AMBER
            : CELL_ACTIVATION_GREEN;

      solids.push({
        cx,
        cy,
        cz,
        ex: bankEx,
        ey,
        ez: bankEz,
        cellFamily,
        bayId: bank.physicalLayerIdx + 1,
        cellScale: 0.12,
        highlight: isFocused ? 1 : policy === 'last_loop' && bank.bankKind === 'recurrent' ? 2 : 0,
      });

      if (isFocused || isFull) {
        lines.addBoxEdges(
          cx,
          cy,
          cz,
          bankEx,
          ey,
          bankEz,
          ownerCounter++,
          isFocused ? 1.4 : 0.85,
          isFocused
            ? LINE_STYLE_ACTIVE
            : policy === 'last_loop' && bank.bankKind === 'recurrent'
              ? LINE_STYLE_BF16_GOLD
              : LINE_STYLE_NEUTRAL,
          isFocused ? 0.96 : 0.62,
        );
      }

      const pbMeta = encodeDoryPickMetadata({
        physicalLayerIdx: bank.physicalLayerIdx,
        stage: bank.stage,
        loopIteration: bank.loopSlot,
        partId: `${policy}_bank_${bank.bankSlotId}`,
        tensorKey: 'kv_bank_matrix',
        kvBankSlotId: bank.bankSlotId,
        kvBankKind: bank.bankKind,
        kvPolicy: policy,
        cacheRole: 'standard',
        modelWeightId: params.modelWeightId,
        groupRole: 'bank',
        tapRole: 'loop',
        extraTags: {
          view: 'kv_compare',
          attnKind: bank.attentionKind,
          overwrite: policy === 'last_loop' && bank.bankKind === 'recurrent',
          logicalLen: bank.logicalLen,
        },
      });
      pickBoxes.push(packPickBoxFromAabb(cx, cy, cz, bankEx, ey, bankEz, pbMeta));

      if (cx - bankEx * 0.5 < minX) minX = cx - bankEx * 0.5;
      if (cx + bankEx * 0.5 > maxX) maxX = cx + bankEx * 0.5;
      if (cz - bankEz * 0.5 < minZ) minZ = cz - bankEz * 0.5;
      if (cz + bankEz * 0.5 > maxZ) maxZ = cz + bankEz * 0.5;
    }

    // For last_loop at k >= 2: draw gold overwrite frame + dashed ghost outline of avoided columns
    if (policy === 'last_loop') {
      const recZ0 = originZ + 9 * rowPitchZ - 0.14;
      const recZ1 = originZ + 28 * rowPitchZ + 0.14;
      lines.addBoxFrameXZ(
        originX - bankEx * 0.72,
        recZ0,
        originX + bankEx * 0.72,
        recZ1,
        0.02,
        1.25,
        LINE_STYLE_BF16_GOLD,
        0.88,
      );
      if (kLoops >= 2) {
        const ghostCols = kLoops - 1;
        const ghostPitch = kLoops > 4 ? 0.46 : 0.62;
        const gx0 = originX + ghostPitch - bankEx * 0.5;
        const gx1 = originX + ghostCols * ghostPitch + bankEx * 0.5;
        lines.addDashedFrameXZ(
          gx0,
          recZ0,
          gx1,
          recZ1,
          0.02,
          LINE_STYLE_GUIDE,
          0.52,
          0.22,
          0.14,
          0.9,
        );
        if (gx1 > maxX) maxX = gx1;
      }
    }

    aabbs.push({
      min: [minX - 0.2, -0.05, minZ - 0.2],
      max: [maxX + 0.2, 0.55, maxZ + 0.2],
    });
    return { minX, maxX, minZ, maxZ };
  };

  // --- LEFT STAGE: `per_loop` ISOLATED SESSION (`16 + 20k` BANKS) ---
  const leftPlatesX = -7.2;
  const leftPlatesLeftEdge = leftPlatesX - plateEx * 0.5;
  const perLoopKvKX = leftPlatesLeftEdge - kvKEx * 0.5 - 0.9;
  const perLoopColPitch = kLoops > 4 ? 0.46 : 0.62;
  const perLoopBankSpanX = (kLoops - 1) * perLoopColPitch + perLoopColPitch * 0.72;
  const leftBankOriginX = perLoopKvKX - kvKEx * 0.5 - 1.15 - perLoopBankSpanX;
  const leftBankBounds = emitBankMatrix(perLoopActive.kvCache, leftBankOriginX, -4.4, 'per_loop');
  const leftStageFrameX0 = leftBankBounds.minX - 0.6;
  const leftStageFrameX1 = leftPlatesX + plateEx * 0.5 + 0.85;
  const stageZMin = -5.6;
  const stageZMax = 5.6;

  lines.addBoxFrameXZ(
    leftStageFrameX0,
    stageZMin,
    leftStageFrameX1,
    stageZMax,
    -0.06,
    1.05,
    LINE_STYLE_GUIDE,
    0.4,
  );

  const sheetPerLoopProbs = registerSheet({
    partId: 'per_loop_attn_probs',
    tensorKey: 'attn_probs',
    label: `per_loop Softmax (Phys L${selectedPhysicalLayerIdx} t=${selectedLoopIteration})`,
    cx: leftPlatesX,
    cy: 0.0,
    cz: -2.4,
    ex: plateEx,
    ey: sheetH,
    ez: plateEz,
    gridCols: kvLen,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_OPERAND_A,
    slice: perLoopProbsSlice,
    values: perLoopProbsH0,
    masked: perLoopMaskedFlags,
    format: 'prob',
    colLabels: keyLabels,
    rowLabels: queryLabels,
    colDimLabel: `K=${kvLen}`,
    rowDimLabel: `Q=${seqLen}`,
    kvPolicy: 'per_loop',
    kvBankSlotId: perLoopAttn.bankSlotId,
    kvBankKind: perLoopAttn.bankKind,
    extraTags: { dr20LiveGMask: true, attnKind: selectedAttentionKind },
  });

  const sheetPerLoopScores = registerSheet({
    partId: 'per_loop_attn_scores',
    tensorKey: 'attn_scores',
    label: `per_loop Scores (Phys L${selectedPhysicalLayerIdx} t=${selectedLoopIteration})`,
    cx: leftPlatesX,
    cy: 0.0,
    cz: 1.6,
    ex: plateEx,
    ey: sheetH,
    ez: plateEz,
    gridCols: kvLen,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_CAUSAL_SCORES,
    role: SHEET_ROLE_SCORES,
    highlight: SHEET_HL_PATHWAY,
    slice: perLoopScoresSlice,
    values: perLoopScoresH0,
    masked: perLoopMaskedFlags,
    format: 'signed',
    colLabels: keyLabels,
    rowLabels: queryLabels,
    colDimLabel: `K=${kvLen}`,
    rowDimLabel: `Q=${seqLen}`,
    kvPolicy: 'per_loop',
    kvBankSlotId: perLoopAttn.bankSlotId,
    kvBankKind: perLoopAttn.bankKind,
    includeInCellGrid: false,
    extraTags: { dr20LiveGMask: true, attnKind: selectedAttentionKind },
  });

  const sheetPerLoopWriters = registerSheet({
    partId: 'per_loop_writer_loops',
    tensorKey: 'kv_writer_loops',
    label: 'per_loop Writer Loop Tags',
    cx: leftPlatesX,
    cy: 0.0,
    cz: 4.35,
    ex: plateEx,
    ey: sheetH,
    ez: writerEz,
    gridCols: kvLen,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: perLoopWriterSlice,
    values: perLoopWriterArr,
    format: 'prob',
    colLabels: keyLabels,
    rowLabels: Object.freeze(['writer_t']),
    colDimLabel: `K=${kvLen}`,
    rowDimLabel: '1',
    kvPolicy: 'per_loop',
    kvBankSlotId: perLoopAttn.bankSlotId,
    kvBankKind: perLoopAttn.bankKind,
    includeInCellGrid: false,
  });

  const sheetPerLoopKvK = registerSheet({
    partId: 'per_loop_kv_k',
    tensorKey: 'k_cache',
    label: 'per_loop Cached Keys (Post-RoPE L2Norm)',
    cx: perLoopKvKX,
    cy: 0.0,
    cz: 0.0,
    ex: kvKEx,
    ey: sheetH,
    ez: kvKEz,
    gridCols: kvDim,
    gridRows: kvLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_K,
    highlight: SHEET_HL_NORMAL,
    slice: perLoopKvKSlice,
    values: perLoopKvK,
    format: 'signed',
    colLabels: kvDimLabels,
    rowLabels: keyLabels,
    colDimLabel: `kv=${kvDim}`,
    rowDimLabel: `K=${kvLen}`,
    kvPolicy: 'per_loop',
    kvBankSlotId: perLoopAttn.bankSlotId,
    kvBankKind: perLoopAttn.bankKind,
    includeInCellGrid: false,
  });

  // --- RIGHT STAGE: `last_loop` ISOLATED SESSION (FLAT `36` BANKS) ---
  const rightPlatesX = 7.2;
  const rightPlatesRightEdge = rightPlatesX + plateEx * 0.5;
  const lastLoopKvKX = rightPlatesRightEdge + kvKEx * 0.5 + 0.9;
  const rightBankOriginX = lastLoopKvKX + kvKEx * 0.5 + 1.25;
  const rightBankBounds = emitBankMatrix(lastLoopActive.kvCache, rightBankOriginX, -4.4, 'last_loop');
  const rightStageFrameX0 = rightPlatesX - plateEx * 0.5 - 0.85;
  const rightStageFrameX1 = rightBankBounds.maxX + 0.65;

  lines.addBoxFrameXZ(
    rightStageFrameX0,
    stageZMin,
    rightStageFrameX1,
    stageZMax,
    -0.06,
    1.05,
    LINE_STYLE_BF16_GOLD,
    0.45,
  );

  const sheetLastLoopProbs = registerSheet({
    partId: 'last_loop_attn_probs',
    tensorKey: 'attn_probs',
    label: `last_loop Softmax (Phys L${selectedPhysicalLayerIdx} t=${selectedLoopIteration})`,
    cx: rightPlatesX,
    cy: 0.0,
    cz: -2.4,
    ex: plateEx,
    ey: sheetH,
    ez: plateEz,
    gridCols: kvLen,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_BF16_GOLD,
    slice: lastLoopProbsSlice,
    values: lastLoopProbsH0,
    masked: lastLoopMaskedFlags,
    format: 'prob',
    colLabels: keyLabels,
    rowLabels: queryLabels,
    colDimLabel: `K=${kvLen}`,
    rowDimLabel: `Q=${seqLen}`,
    kvPolicy: 'last_loop',
    kvBankSlotId: lastLoopAttn.bankSlotId,
    kvBankKind: lastLoopAttn.bankKind,
    extraTags: {
      dr20LiveGMask: true,
      attnKind: selectedAttentionKind,
      writeKv: lastLoopAttn.writeKv,
    },
  });

  const sheetLastLoopScores = registerSheet({
    partId: 'last_loop_attn_scores',
    tensorKey: 'attn_scores',
    label: `last_loop Scores (Phys L${selectedPhysicalLayerIdx} t=${selectedLoopIteration})`,
    cx: rightPlatesX,
    cy: 0.0,
    cz: 1.6,
    ex: plateEx,
    ey: sheetH,
    ez: plateEz,
    gridCols: kvLen,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_CAUSAL_SCORES,
    role: SHEET_ROLE_SCORES,
    highlight: SHEET_HL_OPERAND_B,
    slice: lastLoopScoresSlice,
    values: lastLoopScoresH0,
    masked: lastLoopMaskedFlags,
    format: 'signed',
    colLabels: keyLabels,
    rowLabels: queryLabels,
    colDimLabel: `K=${kvLen}`,
    rowDimLabel: `Q=${seqLen}`,
    kvPolicy: 'last_loop',
    kvBankSlotId: lastLoopAttn.bankSlotId,
    kvBankKind: lastLoopAttn.bankKind,
    includeInCellGrid: false,
    extraTags: {
      dr20LiveGMask: true,
      attnKind: selectedAttentionKind,
      writeKv: lastLoopAttn.writeKv,
    },
  });

  const sheetLastLoopWriters = registerSheet({
    partId: 'last_loop_writer_loops',
    tensorKey: 'kv_writer_loops',
    label: 'last_loop Writer Loop Tags (Hist=k vs Cur=t)',
    cx: rightPlatesX,
    cy: 0.0,
    cz: 4.35,
    ex: plateEx,
    ey: sheetH,
    ez: writerEz,
    gridCols: kvLen,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: lastLoopAttn.pastLen > 0 && !lastLoopAttn.writeKv ? SHEET_HL_ACTIVE : SHEET_HL_BF16_GOLD,
    slice: lastLoopWriterSlice,
    values: lastLoopWriterArr,
    format: 'prob',
    colLabels: keyLabels,
    rowLabels: Object.freeze(['writer_t']),
    colDimLabel: `K=${kvLen}`,
    rowDimLabel: '1',
    kvPolicy: 'last_loop',
    kvBankSlotId: lastLoopAttn.bankSlotId,
    kvBankKind: lastLoopAttn.bankKind,
    includeInCellGrid: false,
  });

  const sheetLastLoopKvK = registerSheet({
    partId: 'last_loop_kv_k',
    tensorKey: 'k_cache',
    label: 'last_loop Cached Keys (Final-Loop History)',
    cx: lastLoopKvKX,
    cy: 0.0,
    cz: 0.0,
    ex: kvKEx,
    ey: sheetH,
    ez: kvKEz,
    gridCols: kvDim,
    gridRows: kvLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_K,
    highlight: SHEET_HL_BF16_GOLD,
    slice: lastLoopKvKSlice,
    values: lastLoopKvK,
    format: 'signed',
    colLabels: kvDimLabels,
    rowLabels: keyLabels,
    colDimLabel: `kv=${kvDim}`,
    rowDimLabel: `K=${kvLen}`,
    kvPolicy: 'last_loop',
    kvBankSlotId: lastLoopAttn.bankSlotId,
    kvBankKind: lastLoopAttn.bankKind,
    includeInCellGrid: false,
  });

  // --- CENTER BRIDGE: ATTENTION DIFF PLATE, `cap_swa = 16` RING & PARTITION DIVERGENCE PROBE ---
  const centerX = 0.0;
  const centerHalfW = 3.55;
  lines.addBoxFrameXZ(
    centerX - centerHalfW,
    stageZMin,
    centerX + centerHalfW,
    stageZMax,
    -0.06,
    1.0,
    LINE_STYLE_GUIDE,
    0.38,
  );

  // Center Top: Attention Probability Shift Plate (`last_loop - per_loop`)
  const diffEx = Math.min(4.8, plateEx);
  const diffEz = Math.min(2.6, plateEz);
  const sheetProbsDiff = registerSheet({
    partId: 'attn_probs_diff',
    tensorKey: 'attn_probs_diff',
    label: 'Attention Shift (last_loop - per_loop)',
    cx: centerX,
    cy: 0.0,
    cz: -3.2,
    ex: diffEx,
    ey: sheetH,
    ez: diffEz,
    gridCols: kvLen,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NEUTRAL,
    highlight: activePartitionVsPerLoopMaxAbsDiff > 1e-2 ? SHEET_HL_ACTIVE : SHEET_HL_NORMAL,
    slice: probsDiffSlice,
    values: attnProbsDiff,
    masked: perLoopMaskedFlags,
    format: 'signed',
    colLabels: keyLabels,
    rowLabels: queryLabels,
    colDimLabel: `K=${kvLen}`,
    rowDimLabel: `Q=${seqLen}`,
    kvPolicy: null,
    includeInCellGrid: false,
  });

  // Center Middle: `cap_swa = 16` Ring Buffer & `absPosTag` Visualizer (`DR19`)
  const ringZ = 0.25;
  const ringEx = 5.76;
  const ringSlotW = ringEx / config.swaRingCapacity;
  const sheetSwaRingTags = registerSheet({
    partId: 'swa_ring_tags',
    tensorKey: 'swa_ring_tags',
    label: `cap_swa=16 Ring (Phys L${swaRingLayerIdx}: absPosTag / writerLoop)`,
    cx: centerX,
    cy: 0.0,
    cz: ringZ + 0.55,
    ex: ringEx,
    ey: sheetH,
    ez: 0.68,
    gridCols: config.swaRingCapacity,
    gridRows: 2,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_K,
    highlight: SHEET_HL_PATHWAY,
    slice: swaRingTagsSlice,
    values: swaRingTagsArr,
    format: 'signed',
    colLabels: ringSlotLabels,
    rowLabels: Object.freeze(['absPos', 'loop']),
    colDimLabel: `cap_swa=${config.swaRingCapacity}`,
    rowDimLabel: '2',
    kvPolicy: 'last_loop',
    physicalLayerIdx: swaRingLayerIdx,
    loopIteration: swaRingLoop,
    includeInCellGrid: true,
    extraTags: { capSwa: config.swaRingCapacity, windowW: config.slidingWindow },
  });

  for (let s = 0; s < config.swaRingCapacity; s += 1) {
    const info = swaRingInfo[s]!;
    const sx = centerX - ringEx * 0.5 + (s + 0.5) * ringSlotW;
    const barH =
      info.status === 'window_active' ? 0.52 : info.status === 'rollback_retained' ? 0.34 : 0.14;
    const cy = barH * 0.5;
    const cellFamily =
      info.status === 'window_active'
        ? CELL_SSM_CYAN
        : info.status === 'rollback_retained'
          ? CELL_MOE_AMBER
          : CELL_NEUTRAL;

    solids.push({
      cx: sx,
      cy,
      cz: ringZ - 0.3,
      ex: ringSlotW * 0.78,
      ey: barH,
      ez: 0.34,
      cellFamily,
      bayId: swaRingLayerIdx + 1,
      cellScale: 0.14,
      highlight: info.status === 'window_active' ? 2 : 0,
    });
    lines.addBoxEdges(
      sx,
      cy,
      ringZ - 0.3,
      ringSlotW * 0.78,
      barH,
      0.34,
      ownerCounter++,
      info.status === 'window_active' ? 1.15 : 0.8,
      info.status === 'rollback_retained' ? LINE_STYLE_BF16_GOLD : LINE_STYLE_NEUTRAL,
      0.78,
    );
    pickBoxes.push(
      packPickBoxFromAabb(
        sx,
        cy,
        ringZ - 0.3,
        ringSlotW * 0.84,
        barH + 0.04,
        0.38,
        encodeDoryPickMetadata({
          physicalLayerIdx: swaRingLayerIdx,
          loopIteration: swaRingLoop,
          partId: `swa_ring_slot_${s}`,
          tensorKey: 'swa_ring_slot',
          absTokenPos: info.absPos >= 0 ? info.absPos : null,
          kvPolicy: 'last_loop',
          modelWeightId: params.modelWeightId,
          extraTags: {
            slot: s,
            absPosTag: info.absPos,
            writerLoopTag: info.writerLoop,
            ringStatus: info.status,
          },
        }),
      ),
    );
    aabbs.push({
      min: [sx - ringSlotW * 0.5, 0, ringZ - 0.5],
      max: [sx + ringSlotW * 0.5, barH + 0.05, ringZ - 0.1],
    });
  }

  // Center Bottom: Live Prefill-Partition Divergence Matrix [3 x promptLen] + 3D Divergence Bars
  const divZ = 3.65;
  const divEx = Math.max(2.6, Math.min(6.0, promptLen * 0.36));
  const sheetDivMatrix = registerSheet({
    partId: 'partition_divergence_matrix',
    tensorKey: 'partition_divergence',
    label: `Prefill-Partition Divergence Probe (Single / Chunked / 1-by-1 · k=${kLoops})`,
    cx: centerX,
    cy: 0.0,
    cz: divZ + 0.65,
    ex: divEx,
    ey: sheetH,
    ez: 0.96,
    gridCols: promptLen,
    gridRows: 3,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight:
      theorem3ChunkedVsPerLoopMaxAbsDiff > 1e-2 ? SHEET_HL_ACTIVE : SHEET_HL_PATHWAY,
    slice: divMatrixSlice,
    values: divMatrix,
    format: 'prob',
    colLabels: promptPosLabels,
    rowLabels: Object.freeze(['single', 'chunked', '1-by-1']),
    colDimLabel: `S=${promptLen}`,
    rowDimLabel: '3 regimes',
    kvPolicy: 'last_loop',
    physicalLayerIdx: null,
    loopIteration: kLoops,
    includeInCellGrid: false,
  });

  const barStepX = divEx / promptLen;
  const barZ = divZ - 0.55;
  for (let i = 0; i < promptLen; i += 1) {
    const pt = perPositionDivergence[i]!;
    const bx = centerX - divEx * 0.5 + (i + 0.5) * barStepX;
    const normMag = Math.min(1.0, pt.maxAbsLogitDiff / Math.max(0.05, maxDivForScale));
    const barH = 0.08 + normMag * 1.15;
    const isDivergent = pt.maxAbsLogitDiff > 1e-2;
    const cellFamily = isDivergent
      ? CELL_MOE_AMBER
      : pt.beyondSwaWindow
        ? CELL_SSM_CYAN
        : CELL_EMBED_SLATE;

    solids.push({
      cx: bx,
      cy: barH * 0.5,
      cz: barZ,
      ex: Math.min(0.28, barStepX * 0.72),
      ey: barH,
      ez: 0.26,
      cellFamily,
      bayId: 0,
      cellScale: 0.14,
      highlight: isDivergent ? 1 : 0,
    });
    lines.addBoxEdges(
      bx,
      barH * 0.5,
      barZ,
      Math.min(0.28, barStepX * 0.72),
      barH,
      0.26,
      ownerCounter++,
      isDivergent ? 1.25 : 0.75,
      isDivergent ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      0.8,
    );
    pickBoxes.push(
      packPickBoxFromAabb(
        bx,
        barH * 0.5,
        barZ,
        Math.min(0.32, barStepX * 0.82),
        barH + 0.06,
        0.32,
        encodeDoryPickMetadata({
          physicalLayerIdx: null,
          loopIteration: kLoops,
          partId: `div_bar_${i}`,
          tensorKey: 'partition_divergence_bar',
          absTokenPos: i,
          kvPolicy: 'last_loop',
          modelWeightId: params.modelWeightId,
          extraTags: {
            chunkIndex: pt.chunkIndex,
            beyondSwaWindow: pt.beyondSwaWindow,
            divergent: isDivergent,
          },
        }),
      ),
    );
    aabbs.push({
      min: [bx - 0.16, 0, barZ - 0.16],
      max: [bx + 0.16, barH + 0.08, barZ + 0.16],
    });

    // KV depth ticks at pos 0, SWA eviction knee W=8, and final position
    if (i === 0 || i === config.slidingWindow || i === promptLen - 1) {
      kvDepthTicks.push(Object.freeze({ token: i, world: Object.freeze([bx, 0.04, barZ - 0.35] as const) }));
    }
  }

  // Chunk boundary markers (gold vertical lines between chunks)
  for (let i = 0; i < chunkBoundaryPositions.length; i += 1) {
    const boundaryPos = chunkBoundaryPositions[i]!;
    const bx = centerX - divEx * 0.5 + boundaryPos * barStepX;
    lines.addSegment(bx, 0.02, barZ - 0.38, bx, 1.38, barZ - 0.38, 1.5, LINE_STYLE_BF16_GOLD, 0.92);
    lines.addSegment(bx, 0.02, barZ - 0.38, bx, 0.02, divZ + 1.2, 1.35, LINE_STYLE_BF16_GOLD, 0.85);
  }
  // SWA eviction boundary marker at pos = W = 8 when prompt crosses window
  if (promptLen > config.slidingWindow) {
    const swaX = centerX - divEx * 0.5 + config.slidingWindow * barStepX;
    lines.addSegment(swaX, 0.02, barZ - 0.44, swaX, 1.45, barZ - 0.44, 1.4, LINE_STYLE_ACTIVE, 0.85);
  }

  // Flow ribbons connecting left (`per_loop`) and right (`last_loop`) causal plates into center diff
  const flowY = -0.025;
  flatFlow.addPolylineXZ(
    [
      { x: sheetPerLoopProbs.frame.center[0] + plateEx * 0.5 + 0.12, z: sheetPerLoopProbs.frame.center[2] },
      { x: sheetProbsDiff.frame.center[0] - diffEx * 0.5 - 0.12, z: sheetProbsDiff.frame.center[2] },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.swaCyan,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetLastLoopProbs.frame.center[0] - plateEx * 0.5 - 0.12, z: sheetLastLoopProbs.frame.center[2] },
      { x: sheetProbsDiff.frame.center[0] + diffEx * 0.5 + 0.12, z: sheetProbsDiff.frame.center[2] },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  ribbons.push({
    cx: -3.55,
    cy: 0.14,
    cz: -2.8,
    ex: 0.65,
    ey: 0.07,
    ez: 0.32,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_CYAN,
    alpha: 0.38,
  });
  ribbons.push({
    cx: 3.55,
    cy: 0.14,
    cz: -2.8,
    ex: 0.65,
    ey: 0.07,
    ez: 0.32,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_AMBER,
    alpha: 0.38,
  });
  ribbons.push({
    cx: leftPlatesX,
    cy: 0.12,
    cz: -0.4,
    ex: 0.42,
    ey: 0.06,
    ez: 0.42,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: 0.28,
  });

  // 6. Overlay Anchors, Landmarks, Calipers, Badges & Callouts
  const prod256kPerLoop = perLoopCurve[perLoopCurve.length - 1]!;
  const prod256kLastLoop = lastLoopCurve[lastLoopCurve.length - 1]!;
  const prodKneePerLoop = perLoopCurve.find((pt) => pt.seqLen === DORY_PRODUCTION_SPEC.slidingWindow)!;

  overlayAnchors.push(
    Object.freeze({
      id: 'anchor_per_loop_banks',
      label: `per_loop (${perLoopBankSummary.totalBanks} Banks)`,
      sublabel: `${perLoopBankSummary.totalSwaBanks} SWA + ${perLoopBankSummary.totalFullBanks} Full · ${formatKiB(perLoopMiniMem.kvPayloadBytes)} mini`,
      worldPos: Object.freeze([(leftBankBounds.minX + leftBankBounds.maxX) * 0.5, 0.72, leftBankBounds.minZ - 0.55] as const),
      tone: 'swa_cyan',
    }),
    Object.freeze({
      id: 'anchor_last_loop_banks',
      label: `last_loop (${lastLoopBankSummary.totalBanks} Banks Flat)`,
      sublabel: `27 SWA + 9 Full · ${formatKiB(lastLoopMiniMem.kvPayloadBytes)} mini · overwrite at t=k`,
      worldPos: Object.freeze([(rightBankBounds.minX + rightBankBounds.maxX) * 0.5, 0.72, rightBankBounds.minZ - 0.55] as const),
      tone: 'gold',
      badge: 'DR17',
    }),
    Object.freeze({
      id: 'anchor_per_loop_probs',
      label: `per_loop Softmax (Phys L${selectedPhysicalLayerIdx} t=${selectedLoopIteration})`,
      sublabel: `writeKv=${String(perLoopAttn.writeKv)} · pastLen=${perLoopAttn.pastLen} · DR20 live G-mask`,
      worldPos: localToWorldOnFrame(sheetPerLoopProbs.frame, 0, 0.7, -0.62),
      tone: 'swa_cyan',
    }),
    Object.freeze({
      id: 'anchor_last_loop_probs',
      label: `last_loop Softmax (Phys L${selectedPhysicalLayerIdx} t=${selectedLoopIteration})`,
      sublabel: `writeKv=${String(lastLoopAttn.writeKv)} (${lastLoopAttn.writeKv ? 'commit' : 'DoryPastReader'}) · DR20 live G-mask`,
      worldPos: localToWorldOnFrame(sheetLastLoopProbs.frame, 0, 0.7, -0.62),
      tone: 'gold',
    }),
    Object.freeze({
      id: 'anchor_probs_diff',
      label: 'Attention Weight Shift (last_loop - per_loop)',
      sublabel: `Active [${partitionMode}] max logit diff = ${activePartitionVsPerLoopMaxAbsDiff.toExponential(2)}`,
      worldPos: localToWorldOnFrame(sheetProbsDiff.frame, 0, 0.7, -0.62),
      tone: activePartitionVsPerLoopMaxAbsDiff > 1e-2 ? 'warn' : 'muted',
    }),
    Object.freeze({
      id: 'anchor_swa_ring',
      label: `cap_swa=${config.swaRingCapacity} SWA Ring & absPosTag`,
      sublabel: `W=${config.slidingWindow} active + ${config.swaRingCapacity - config.slidingWindow} rollback slots (pos & 15)`,
      worldPos: localToWorldOnFrame(sheetSwaRingTags.frame, 0, 0.78, 0.72),
      tone: 'swa_cyan',
      badge: 'DR19',
    }),
    Object.freeze({
      id: 'anchor_partition_probe',
      label: 'Prefill-Partition Divergence Probe',
      sublabel: `Thm 1=${theorem1SingleShotMaxAbsDiff.toExponential(1)} · Thm 3 chunked=${theorem3ChunkedVsPerLoopMaxAbsDiff.toExponential(2)}`,
      worldPos: localToWorldOnFrame(sheetDivMatrix.frame, 0, 0.78, 0.72),
      tone: theorem3ChunkedVsPerLoopMaxAbsDiff > 1e-2 ? 'warn' : 'nlerp_emerald',
      badge: 'DR17',
    }),
    Object.freeze({
      id: 'anchor_per_loop_scores',
      label: 'per_loop Causal Scores (h0)',
      worldPos: localToWorldOnFrame(sheetPerLoopScores.frame, 0, 0.65, 0.62),
      tone: 'prelude_indigo',
    }),
    Object.freeze({
      id: 'anchor_last_loop_scores',
      label: 'last_loop Causal Scores (h0)',
      worldPos: localToWorldOnFrame(sheetLastLoopScores.frame, 0, 0.65, 0.62),
      tone: 'gold',
    }),
    Object.freeze({
      id: 'anchor_per_loop_writers',
      label: 'per_loop Writer Loops',
      worldPos: localToWorldOnFrame(sheetPerLoopWriters.frame, 0, 0.65, 0.85),
      tone: 'swa_cyan',
    }),
    Object.freeze({
      id: 'anchor_last_loop_writers',
      label: 'last_loop Writer Loops',
      worldPos: localToWorldOnFrame(sheetLastLoopWriters.frame, 0, 0.65, 0.85),
      tone: 'gold',
    }),
    Object.freeze({
      id: 'anchor_per_loop_kv_k',
      label: 'per_loop Key Cache',
      worldPos: localToWorldOnFrame(sheetPerLoopKvK.frame, 0, 0.65, 0.65),
      tone: 'muted',
    }),
    Object.freeze({
      id: 'anchor_last_loop_kv_k',
      label: 'last_loop Key Cache',
      worldPos: localToWorldOnFrame(sheetLastLoopKvK.frame, 0, 0.65, 0.65),
      tone: 'muted',
    }),
  );

  landmarks.push(
    Object.freeze({
      id: 'lm_kv_per_loop',
      label: `per_loop Session (${perLoopBankSummary.totalBanks} Banks)`,
      sublabel: `${sInvariantRatio.toFixed(3)}x last_loop (S-invariant)`,
      world: Object.freeze([leftPlatesX, 0.95, stageZMin - 0.35] as const),
      partId: 'per_loop_attn_probs',
      tone: 'cyan',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_kv_last_loop',
      label: 'last_loop Session (36 Banks Flat)',
      sublabel: `Partition-dependent for k >= 2 (DR17)`,
      world: Object.freeze([rightPlatesX, 0.95, stageZMin - 0.35] as const),
      partId: 'last_loop_attn_probs',
      tone: 'gold',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_kv_probe',
      label: 'SWA Ring (cap=16) & Partition Probe',
      sublabel: `256K Prod: ${formatBytesAdaptive(prod256kPerLoop.totalBytes)} vs ${formatBytesAdaptive(prod256kLastLoop.totalBytes)}`,
      world: Object.freeze([centerX, 0.95, stageZMin - 0.35] as const),
      partId: 'partition_divergence_matrix',
      tone: 'default',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'caliper_swa_ring_cap',
      aWorld: Object.freeze([centerX - ringEx * 0.5, 0.65, ringZ - 0.65] as const),
      bWorld: Object.freeze([centerX + ringEx * 0.5, 0.65, ringZ - 0.65] as const),
      label: `cap_swa = ${config.swaRingCapacity} (W=${config.slidingWindow} + gamma=4 + 1 = 13 <= 16)`,
      partIds: Object.freeze(['swa_ring_tags']),
    }),
    Object.freeze({
      id: 'caliper_prompt_len',
      aWorld: Object.freeze([centerX - divEx * 0.5, 0.1, divZ + 1.35] as const),
      bWorld: Object.freeze([centerX + divEx * 0.5, 0.1, divZ + 1.35] as const),
      label: `S = ${promptLen} tokens · partition [${activePartition.join(', ')}]`,
      partIds: Object.freeze(['partition_divergence_matrix']),
    }),
  );

  const provenanceBadges: DoryProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    Object.freeze({
      label: `Isolated Dual-Session: per_loop (${perLoopBankSummary.totalBanks} banks) vs last_loop (${lastLoopBankSummary.totalBanks} banks)`,
      tone: 'swa_cyan',
      discrepancyId: 'DR18',
    }),
    Object.freeze({
      label: `S-Invariant Ratio (4 + 5k)/9 = ${sInvariantRatio.toFixed(3)}x · SWA Knee S=1,024 (4x Slope Drop)`,
      tone: 'recurrent_teal',
    }),
    Object.freeze({
      label: theorem2K1Equivalence
        ? 'Theorem 2 (k=1): last_loop === per_loop across all partitions'
        : `Theorem 1 Single-Shot = ${theorem1SingleShotMaxAbsDiff.toExponential(1)} · Theorem 3 Chunked = ${theorem3ChunkedVsPerLoopMaxAbsDiff.toExponential(2)}`,
      tone: theorem3ChunkedVsPerLoopMaxAbsDiff > 1e-2 ? 'warn' : 'nlerp_emerald',
      discrepancyId: 'DR17',
    }),
    Object.freeze({
      label: `cap_swa=${config.swaRingCapacity} Ring + absPosTag (DR19 · DR24 Long-Chunk Workspace)`,
      tone: 'gold',
      discrepancyId: 'DR19',
    }),
  ];

  const callouts: DorySceneCallout[] = [
    Object.freeze({
      id: 'callout_kv_isolated_dual_session',
      title: 'Isolated Immutable Dual-Session Comparison (D6 / DR18)',
      body:
        `Runs brand-new non-aliased per_loop (${perLoopBankSummary.totalBanks} banks = ${formatKiB(perLoopMiniMem.kvPayloadBytes)}) ` +
        `and last_loop (${lastLoopBankSummary.totalBanks} banks = ${formatKiB(lastLoopMiniMem.kvPayloadBytes)}) caches at k=${kLoops} ` +
        `over identical S=${promptLen} tokens without mutating the active DorySession.`,
      tone: 'swa_cyan',
      worldPos: sheetPerLoopProbs.frame.center,
      badge: 'ZERO SESSION MUTATION',
      discrepancyIds: Object.freeze(['DR18'] as const),
    }),
    Object.freeze({
      id: 'callout_kv_theorems_1_2_3',
      title: 'last_loop Partition Dependence (Theorems 1–3, DR17)',
      body:
        `Theorem 1 (empty-cache single-shot [${promptLen}]): max |diff| = ${theorem1SingleShotMaxAbsDiff.toExponential(2)} (exact identity in real math). ` +
        (theorem2K1Equivalence
          ? `Theorem 2 (k=1): every loop is the final loop, so chunked (${theorem3ChunkedVsPerLoopMaxAbsDiff.toExponential(2)}) and 1-by-1 match universally.`
          : `Theorem 3 (k=${kLoops} >= 2): chunked [${chunkedSchedule.join(', ')}] vs per_loop = ${theorem3ChunkedVsPerLoopMaxAbsDiff.toExponential(2)}, ` +
            `chunked vs 1-by-1 = ${theorem3ChunkedVsOneByOneMaxAbsDiff.toExponential(2)}. On non-final loops t < k, DoryPastReader reads committed final-loop KV (t=${kLoops}) ` +
            `for past tokens while attending to same-loop KV (t=${selectedLoopIteration}) for the current chunk.`),
      tone: theorem3ChunkedVsPerLoopMaxAbsDiff > 1e-2 ? 'warn' : 'nlerp_emerald',
      worldPos: sheetDivMatrix.frame.center,
      badge: 'DR17 · DR24',
      discrepancyIds: Object.freeze(['DR17', 'DR24'] as const),
    }),
    Object.freeze({
      id: 'callout_kv_swa_ring',
      title: `cap_swa = ${config.swaRingCapacity} Ring Buffer & absPosTag (DR19 / DR24)`,
      body:
        `Sliding-window layers (W=${config.slidingWindow}) retain up to ${config.swaRingCapacity} ring slots (pos & 15) tagged with exact absPosTag ` +
        `so speculative gamma=4 rollback (W+gamma+1=13 <= 16) never overwrites history. Chunks L > 16 execute over a transient [past_visible || chunk] workspace before committing.`,
      tone: 'swa_cyan',
      worldPos: sheetSwaRingTags.frame.center,
      badge: 'cap_swa=16 · absPosTag',
      discrepancyIds: Object.freeze(['DR19', 'DR24'] as const),
    }),
    Object.freeze({
      id: 'callout_kv_production_scaling',
      title: `1 -> 256K Production BF16 Scaling & S-Invariant (4+5k)/9 Ratio`,
      body:
        `Because all 3 stages preserve exact 3:1 SWA:Full, per_loop / last_loop = (4 + 5*${kLoops})/9 = ${sInvariantRatio.toFixed(3)}x is strictly invariant to context S. ` +
        `At the S=1,024 SWA knee (${formatBytesAdaptive(prodKneePerLoop.totalBytes)}, SWA=75%), marginal growth drops 4x; by 256K, SWA shrinks to ` +
        `${(prod256kPerLoop.swaShare * 100).toFixed(2)}% (${formatBytesAdaptive(prod256kPerLoop.totalBytes)} per_loop vs ${formatBytesAdaptive(prod256kLastLoop.totalBytes)} last_loop).`,
      tone: 'recurrent_teal',
      worldPos: sheetLastLoopProbs.frame.center,
      badge: '1 -> 256K BF16',
    }),
  ];

  const activeCellPopover = buildKvComparePopover(
    options.hoveredCell,
    sheetsByPartId,
    perLoopAttn,
    lastLoopAttn,
    swaRingInfo,
    perPositionDivergence,
    kLoops,
    partitionMode,
  );

  const stageBoundaryAabb = computeBoundsFromPoints(
    [
      [leftStageFrameX0, -0.1, stageZMin],
      [rightStageFrameX1, 1.5, stageZMax],
    ],
    0.5,
  );
  const fitBounds = mergeAabbs([...aabbs, stageBoundaryAabb]);

  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flatFlow.pack();

  const kvCompareMetadata: DoryKvCompareViewMetadata = Object.freeze({
    kLoops,
    promptLen,
    partitionMode,
    chunkPartition: activePartition,
    selectedPhysicalLayerIdx,
    selectedPairedLayerIdx: layerSpec.pairedLayerIdx,
    selectedAttentionKind,
    selectedLoopIteration,
    perLoopBankCount: perLoopBankSummary.totalBanks,
    perLoopSwaBanks: perLoopBankSummary.totalSwaBanks,
    perLoopFullBanks: perLoopBankSummary.totalFullBanks,
    lastLoopBankCount: lastLoopBankSummary.totalBanks,
    lastLoopSwaBanks: lastLoopBankSummary.totalSwaBanks,
    lastLoopFullBanks: lastLoopBankSummary.totalFullBanks,
    perLoopMiniBytes: perLoopMiniMem.kvPayloadBytes,
    lastLoopMiniBytes: lastLoopMiniMem.kvPayloadBytes,
    swaWindowMini: config.slidingWindow,
    swaRingCapacityMini: config.swaRingCapacity,
    swaWindowProd: DORY_PRODUCTION_SPEC.slidingWindow,
    swaKneeSlopeDropFactor: 4 as const,
    sInvariantRatio,
    theorem2K1Equivalence,
    theorem1SingleShotMaxAbsDiff,
    theorem3ChunkedVsPerLoopMaxAbsDiff,
    theorem3ChunkedVsOneByOneMaxAbsDiff,
    activePartitionVsPerLoopMaxAbsDiff,
    isolatedDualSessionVerified: true as const,
    hasLiveCausalMaskGChannel: true as const,
    perPositionDivergence: Object.freeze(perPositionDivergence),
    productionScalingCurve: perLoopCurve,
    productionLastLoopCurve: lastLoopCurve,
  });

  const scene: DorySpatialScene = Object.freeze({
    viewMode: 'kv_compare',
    title: VIEW_DESCRIPTOR.label,
    subtitle: `k=${kLoops} · per_loop (${perLoopBankSummary.totalBanks} banks) vs last_loop (36 banks) · [${partitionMode}: ${activePartition.join('+')}] · Phys L${selectedPhysicalLayerIdx} (${layerSpec.transformersPairBadge})`,
    modelWeightId: params.modelWeightId,
    branchId: ctx.branch.branchId,
    recordId: ctx.record.recordId,
    kLoops,
    kvCacheMode: ctx.kvCacheMode,
    executionMode: ctx.executionMode,
    physicalLayerIdx: selectedPhysicalLayerIdx,
    loopIteration: selectedLoopIteration,
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
    kvDepthTicks: Object.freeze(kvDepthTicks),
    calipers: Object.freeze(calipers),
    activeCellGrids: Object.freeze(activeCellGrids),
    activeCellPopover,
    kvCompareMetadata,
  });

  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `View 5 kvCompareScene validation failed: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
