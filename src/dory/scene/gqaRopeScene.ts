/**
 * View 3 (`gqa_rope`): Gated GQA & Dual-RoPE Cutaway Spatial Scene
 * (`src/dory/scene/gqaRopeScene.ts`)
 *
 * Implements DORY4 Slice 4A (Locked Decisions `D1`, `D2`, `D4`, `D5`, `D7`; Guards `DR3`,
 * `DR4`, `DR5`, `DR6`, `DR7`, `DR9`, `DR17`, `DR20`, `DR21`, `DR24`):
 *
 *   - Contrasts Sliding-Window Attention (`W_mini = 8`, `theta_S = 16`, dyadic `inv_freq =
 *     [1, 1/2, 1/4, 1/8]`) against Full Attention (`theta_F = 1,024`, full causal visibility)
 *     across any even physical attention sublayer `p in {0, 2, ..., 70}` and recurrent pass
 *     `t in 1..k` (or `t = 0` for Prelude/Coda).
 *   - Explicit miniature vs. production projection scaling disclosure (`D2`):
 *       * Miniature: `d = 8 -> q_dim = 64` (`8.0x`), `kv_dim = 16` (`2.0x`), `4:1` GQA (`h_q=8, h_kv=2, d_h=8`).
 *       * Production: `d = 2,560 -> q_dim = 4,096` (`1.60x`), `kv_dim = 1,024` (`0.40x`), `4:1` GQA (`h_q=16, h_kv=4, d_h=256`).
 *   - Canonical Full-nGPT Gated GQA dataflow (`DR5`, `DR6`, `DR3`, `DR7`):
 *       1. Bias-free `Q, K, V, Gate` projections (NO Layer-0 pre-norm; `V` and `G` unnormalized).
 *       2. 100% NeoX `rotate_half` Dual-RoPE (`4` rotary planes `(j, j + 4)`) strictly BEFORE
 *          per-head Q/K L2Norm (`DR6`, matching `vLLM`/`llama.cpp`/TeX), paired with a 3D 4-plane
 *          phasor pedestal and exact `DoryRopePhasorPlane[]` metadata.
 *       3. Per-head unit-sphere L2Norm (`eps = 1e-6`) on `Q` and `K`, plus **Q-only** squared
 *          `sqk^2 = (sqk * sqrt(d_h) / 0.0177)^2` scaling (cached `K` stays unscaled unit-norm!).
 *       4. All `8` query heads grouped `4:1` under `KV0 (Q0..Q3)` & `KV1 (Q4..Q7)`, emitting
 *          `8` score plates (`SHEET_KIND_CAUSAL_SCORES = 1`) and `8` softmax plates
 *          (`SHEET_KIND_CAUSAL_SOFTMAX = 2`), EVERY SINGLE ONE carrying a live `RG32F` `G`-channel
 *          causal/SWA mask (`DR20` — zero `sheets.ts:351` geometric fallback!) and explicit
 *          historical KV loop-author provenance (`kvAbsPositions`, `kvWriterLoops`, `writeKv`).
 *       5. Input-dependent sigmoid output gate `C * sigmoid(G * s_gate^2)` honoring `g_0` (`DR3`),
 *          concatenated `attn_sub_norm = N_{64}(...)` over ALL `q_dim = h_q * d_h = 64` channels
 *          (NOT per-head `8`), bias-free `o_proj` (`64 -> 8`), and channel-wise nGPT NLERP
 *          residual blend (`DR7`, NOT trigonometric SLERP, unclamped `|alpha_hat| >= 0`).
 *   - Compact dynamically-sized `RG32F` atlas (`width = 256`, `32..416 KiB <= 512 KiB`),
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
  CELL_WEIGHT_BLUE,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_KIND_ARROW_POS_X,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
  type Vec3,
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
  SHEET_ROLE_KV_V,
  SHEET_ROLE_NORM,
  SHEET_ROLE_RESIDUAL,
  SHEET_ROLE_SCORES,
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
import { vectorL2Norm } from '../ngptMath';
import {
  DORY_PRODUCTION_SPEC,
  decodeDoryToken,
  getDoryPhysicalLayerSpec,
  getDoryStage,
} from '../spec';
import {
  DORY_MAX_ATLAS_BYTES,
  DORY_MINI_CONFIG,
  DORY_NLERP_MARKER,
  type DoryAttentionSublayerTrace,
  type DoryAttentionSublayerWeights,
  type DoryForwardTrace,
  type DoryKvBank,
  type DoryKvCache,
  type DoryReconstructedStepResult,
  type DoryRopeTable,
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
  type DoryBadgeTone,
  type DoryGqaRopeSceneOptions,
  type DoryGqaRopeViewMetadata,
  type DoryOverlayAnchor,
  type DoryPickIdentity,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DoryRopePhasorPlane,
  type DorySceneCallout,
  type DorySpatialScene,
} from './types';

// ============================================================================
// VIEW 3 CONSTANTS & PURE HELPERS
// ============================================================================

export const DORY_GQA_ROPE_ATLAS_WIDTH = 256 as const;
export const DORY_GQA_ROPE_ATLAS_BUDGET_BYTES = DORY_MAX_ATLAS_BYTES; // 524,288 B = 512.00 KiB
export const DORY_GQA_ROPE_DEFAULT_SWA_LAYER = 16 as const;
export const DORY_GQA_ROPE_DEFAULT_FULL_LAYER = 22 as const;

const VIEW_DESCRIPTOR = DORY_SPATIAL_VIEW_MODES[2]!; // 'gqa_rope'
const PHASOR_RING_SEGMENTS = 32;

function clampInt(val: number, lo: number, hi: number): number {
  if (!Number.isFinite(val)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(val)));
}

/**
 * Resolves any requested physical sublayer index into a valid even Attention physical layer
 * `p in {0, 2, ..., 70}` (snapping odd MLP indices cleanly to their paired attention sublayer).
 */
export function resolveAttentionPhysicalLayerIdx(requested?: number): number {
  if (requested === undefined || !Number.isFinite(requested)) {
    return DORY_GQA_ROPE_DEFAULT_SWA_LAYER;
  }
  const clamped = clampInt(requested, 0, DORY_MINI_CONFIG.numHiddenLayers - 1);
  return clamped % 2 === 0 ? clamped : Math.max(0, clamped - 1);
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

function findKvBankForTrace(
  kvCache: DoryKvCache,
  attnTrace: DoryAttentionSublayerTrace,
): DoryKvBank | null {
  for (let i = 0; i < kvCache.allBanks.length; i += 1) {
    const bank = kvCache.allBanks[i]!;
    if (bank.bankSlotId === attnTrace.bankSlotId) {
      return bank;
    }
  }
  return null;
}

function computeArrayMinMax(arr: Float32Array): readonly [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < arr.length; i += 1) {
    const v = arr[i]!;
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    return Object.freeze([0, 0] as const);
  }
  return Object.freeze([lo, hi] as const);
}

/**
 * Computes exact 4-plane NeoX `rotate_half` Dual-RoPE phasor descriptors for the focused
 * `(selectedQHead, selectedKvHead, selectedQueryRow, selectedKeyCol)` pair (`D2`, `DR6`).
 *
 * For current-chunk keys (`selectedKeyCol >= pastLen`), reads exact `kPre` and `kRot` directly
 * from `DoryAttentionSublayerTrace`. For historical cached keys (`selectedKeyCol < pastLen`),
 * reads post-RoPE cached `K` from `postStepKvCache` when resident and applies exact inverse
 * NeoX rotation `R(-keyAngle)` to recover the pre-RoPE direction cleanly.
 */
export function computeDoryRopePhasorPlanes(params: {
  readonly attnTrace: DoryAttentionSublayerTrace;
  readonly ropeTable: DoryRopeTable;
  readonly postStepKvCache: DoryKvCache;
  readonly selectedQHead: number;
  readonly selectedKvHead: number;
  readonly selectedQueryRow: number;
  readonly selectedKeyCol: number;
}): readonly DoryRopePhasorPlane[] {
  const {
    attnTrace,
    ropeTable,
    postStepKvCache,
    selectedQHead,
    selectedKvHead,
    selectedQueryRow,
    selectedKeyCol,
  } = params;
  const dH = ropeTable.headDim;
  const halfDim = ropeTable.halfDim; // 4 rotary planes
  const qDim = DORY_MINI_CONFIG.qProjOutDim;
  const kvDim = DORY_MINI_CONFIG.kvProjOutDim;

  const qAbsPos = attnTrace.startPos + selectedQueryRow;
  const kAbsPos = attnTrace.kvAbsPositions[selectedKeyCol] ?? qAbsPos;
  const qHeadBase = selectedQueryRow * qDim + selectedQHead * dH;

  const isKeyInCurrentChunk = selectedKeyCol >= attnTrace.pastLen;
  const chunkKeyRow = selectedKeyCol - attnTrace.pastLen;
  const kvBank = isKeyInCurrentChunk ? null : findKvBankForTrace(postStepKvCache, attnTrace);

  // Resolve post-RoPE historical key vector from resident bank slot if available
  let histKCacheOffset = -1;
  if (kvBank !== null) {
    const slot =
      kvBank.attentionKind === 'sliding_attention'
        ? kAbsPos & (kvBank.capacity - 1)
        : kAbsPos;
    if (slot >= 0 && slot < kvBank.capacity && kvBank.absPosTag[slot] === kAbsPos) {
      histKCacheOffset = slot * kvDim + selectedKvHead * dH;
    }
  }

  const planes: DoryRopePhasorPlane[] = [];
  for (let j = 0; j < halfDim; j += 1) {
    const invFreq = ropeTable.invFreq[j]!;
    const periodTokens = invFreq > 0 ? (2 * Math.PI) / invFreq : 0;
    const queryAngleRad = qAbsPos * invFreq;
    const keyAngleRad = kAbsPos * invFreq;
    const deltaAngleRad = (qAbsPos - kAbsPos) * invFreq;

    const qPrePair: readonly [number, number] = Object.freeze([
      attnTrace.qPre[qHeadBase + j] ?? 0,
      attnTrace.qPre[qHeadBase + j + halfDim] ?? 0,
    ] as const);
    const qRotPair: readonly [number, number] = Object.freeze([
      attnTrace.qRot[qHeadBase + j] ?? 0,
      attnTrace.qRot[qHeadBase + j + halfDim] ?? 0,
    ] as const);

    let kPrePair: readonly [number, number];
    let kRotPair: readonly [number, number];

    if (isKeyInCurrentChunk && chunkKeyRow >= 0 && chunkKeyRow < attnTrace.seqLen) {
      const kHeadBase = chunkKeyRow * kvDim + selectedKvHead * dH;
      kPrePair = Object.freeze([
        attnTrace.kPre[kHeadBase + j] ?? 0,
        attnTrace.kPre[kHeadBase + j + halfDim] ?? 0,
      ] as const);
      kRotPair = Object.freeze([
        attnTrace.kRot[kHeadBase + j] ?? 0,
        attnTrace.kRot[kHeadBase + j + halfDim] ?? 0,
      ] as const);
    } else if (kvBank !== null && histKCacheOffset >= 0) {
      const y0 = kvBank.k[histKCacheOffset + j] ?? 0;
      const y1 = kvBank.k[histKCacheOffset + j + halfDim] ?? 0;
      const c = Math.cos(keyAngleRad);
      const s = Math.sin(keyAngleRad);
      // Exact inverse NeoX rotate_half: R(-keyAngle) * (y0, y1)
      kPrePair = Object.freeze([y0 * c + y1 * s, -y0 * s + y1 * c] as const);
      kRotPair = Object.freeze([y0, y1] as const);
    } else {
      // Fallback if historical slot was overwritten by a long L > 16 chunk after attention
      const c = Math.cos(keyAngleRad);
      const s = Math.sin(keyAngleRad);
      kPrePair = Object.freeze([1, 0] as const);
      kRotPair = Object.freeze([c, s] as const);
    }

    planes.push(
      Object.freeze({
        planeIdx: j,
        invFreq,
        periodTokens,
        queryAngleRad,
        keyAngleRad,
        deltaAngleRad,
        qPrePair,
        qRotPair,
        kPrePair,
        kRotPair,
      }),
    );
  }

  return Object.freeze(planes);
}

/**
 * Selects the default key column for `(selectedQHead, selectedQueryRow)`: picks the highest-probability
 * visible key column (guaranteed visible since masked probabilities are 0 and visible probabilities
 * sum to 1), falling back to the query's own key column if needed.
 */
function resolveDefaultKeyCol(
  attnTrace: DoryAttentionSublayerTrace,
  selectedQHead: number,
  selectedQueryRow: number,
): number {
  const { seqLen, kvLen, probs, maskVisible } = attnTrace;
  const rowBase = (selectedQHead * seqLen + selectedQueryRow) * kvLen;
  const maskBase = selectedQueryRow * kvLen;
  let bestCol = Math.max(0, Math.min(kvLen - 1, attnTrace.pastLen + selectedQueryRow));
  let bestProb = -1;
  for (let ki = 0; ki < kvLen; ki += 1) {
    if (maskVisible[maskBase + ki] === 0) continue;
    const p = probs[rowBase + ki] ?? 0;
    if (p > bestProb) {
      bestProb = p;
      bestCol = ki;
    }
  }
  return bestCol;
}

interface PlacedGqaSheet {
  readonly partId: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly frame: SheetFrame3D;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly values: Float32Array;
  readonly masked?: Uint8Array;
  readonly grid: ActiveSheetCellGrid;
  readonly qHeadIndex?: number;
  readonly kvHeadIndex?: number;
}

function buildGqaHoveredPopover(
  hovered: DoryPickIdentity | null | undefined,
  sheetsByPartId: ReadonlyMap<string, PlacedGqaSheet>,
  attnTrace: DoryAttentionSublayerTrace,
  weights: DoryAttentionSublayerWeights,
  metadata: DoryGqaRopeViewMetadata,
  tokenLabels: readonly string[],
  kvColLabels: readonly string[],
): CellPopoverSpec | null {
  if (!hovered || !hovered.partId) return null;

  // Hovering a 3D Dual-RoPE phasor plane pedestal (`gqa_rope_plane_0..3`)
  const planeMatch = /^gqa_rope_plane_(\d+)$/.exec(hovered.partId);
  if (planeMatch !== null) {
    const planeIdx = clampInt(
      Number.parseInt(planeMatch[1]!, 10),
      0,
      metadata.ropePhasors.length - 1,
    );
    const plane = metadata.ropePhasors[planeIdx]!;
    const pedestalRec = sheetsByPartId.get('gqa_k_rot');
    const world: Vec3 = pedestalRec
      ? [
          pedestalRec.frame.center[0] + (planeIdx - 1.5) * 1.12,
          0.35,
          2.65,
        ]
      : [-5.9 + (planeIdx - 1.5) * 1.12, 0.35, 2.65];
    const deltaDeg = (plane.deltaAngleRad * 180) / Math.PI;
    return Object.freeze({
      world,
      tensorLabel: `Dual-RoPE Plane j=${planeIdx} (${metadata.ropeRegime === 'sliding' ? 'SWA' : 'Full'})`,
      coordLabel: `pair (${planeIdx}, ${planeIdx + 4}) · pos q=${metadata.selectedQueryAbsPos}, k=${metadata.selectedKeyAbsPos}`,
      badge: `theta=${metadata.ropeThetaMini} (prod ${metadata.ropeThetaProd.toLocaleString('en-US')}) · DR6`,
      equation: `inv_freq=${plane.invFreq.toFixed(4)} · T=${plane.periodTokens.toFixed(1)} tok · delta_phi=${fmtSigned(deltaDeg, 1)} deg`,
      valueText: `Q_rot=(${fmtSigned(plane.qRotPair[0], 2)}, ${fmtSigned(plane.qRotPair[1], 2)})`,
    });
  }

  const rec = sheetsByPartId.get(hovered.partId);
  if (!rec) return null;

  const row = clampInt(hovered.row ?? 0, 0, rec.gridRows - 1);
  const col = clampInt(hovered.col ?? 0, 0, rec.gridCols - 1);
  const world = computeSheetCellWorldPos(rec.grid, col, row, 0.5);
  const idx = row * rec.gridCols + col;
  const val = rec.values[idx] ?? 0;
  const tokLabel = tokenLabels[row] ?? `q${row}`;
  const dH = DORY_MINI_CONFIG.headDim;

  // Score & Softmax plates (`gqa_scores_h0..7`, `gqa_probs_h0..7`)
  if (rec.partId.startsWith('gqa_scores_h') || rec.partId.startsWith('gqa_probs_h')) {
    const qh = rec.qHeadIndex ?? 0;
    const kvh = rec.kvHeadIndex ?? Math.floor(qh / DORY_MINI_CONFIG.kvGroupSize);
    const isScore = rec.partId.startsWith('gqa_scores_h');
    const isMasked = rec.masked !== undefined && (rec.masked[idx] ?? 0) > 0.5;
    const qAbs = attnTrace.startPos + row;
    const kAbs = attnTrace.kvAbsPositions[col] ?? col;
    const diff = qAbs - kAbs;
    const writerLoop = attnTrace.kvWriterLoops[col] ?? 0;
    const kLabel = kvColLabels[col] ?? `k${kAbs}`;
    const kvProv =
      col < attnTrace.pastLen
        ? `past KV (writer t=${writerLoop})`
        : `chunk KV (t=${attnTrace.loopIteration})`;
    const qScaledSheet = sheetsByPartId.get('gqa_q_scaled');
    const kCacheSheet = sheetsByPartId.get('gqa_k_cache');
    const maskReason =
      diff < 0
        ? `future key (q=${qAbs} < k=${kAbs})`
        : `outside SWA window W=${DORY_MINI_CONFIG.slidingWindow} (q-k=${diff} >= 8)`;

    return Object.freeze({
      world,
      tensorLabel: isScore
        ? `GQA Scores Q${qh} (KV${kvh})`
        : `GQA Softmax P Q${qh} (KV${kvh})`,
      coordLabel: `${tokLabel} -> ${kLabel} · ${kvProv}`,
      badge: isMasked
        ? `DR20 LIVE RG32F G=1 MASKED · ${maskReason}`
        : `DR20 LIVE RG32F G=0 · ${metadata.attentionKind === 'sliding_attention' ? 'SWA W=8' : 'Full Causal'}`,
      equation: isMasked
        ? isScore
          ? `masked -> -inf (G-channel = 1.0, diff=${diff})`
          : `masked -> 0.0000 (G-channel = 1.0, diff=${diff})`
        : isScore
          ? `d_h^(-1/2) * <sqk^2 * N(RoPE(q)), N(RoPE(k))> (bank #${attnTrace.bankSlotId})`
          : `softmax(score[${row}, :]) over visible ${metadata.attentionKind === 'sliding_attention' ? 'W<=8' : 'causal'} keys`,
      valueText: isMasked
        ? isScore
          ? '= -inf'
          : '= 0.0000 (masked)'
        : isScore
          ? `= ${fmtSigned(val, 4)}`
          : `= ${val.toFixed(4)}`,
      operandColors: isMasked ? undefined : Object.freeze(['indigo', 'teal'] as const),
      sourceWorlds:
        !isMasked && qScaledSheet && kCacheSheet
          ? Object.freeze([
              {
                world: computeSheetCellWorldPos(qScaledSheet.grid, qh * dH, row, 0.5),
                role: 'indigo' as const,
              },
              {
                world: computeSheetCellWorldPos(
                  kCacheSheet.grid,
                  kvh * dH,
                  Math.max(0, Math.min(attnTrace.seqLen - 1, col - attnTrace.pastLen)),
                  0.5,
                ),
                role: 'teal' as const,
              },
            ])
          : undefined,
    });
  }

  if (rec.partId === 'gqa_sqk_raw' || rec.partId === 'gqa_sqk_eff_sq') {
    const qh = Math.floor(col / dH);
    const raw = weights.sqk[col] ?? 0;
    const effSq = attnTrace.effectiveSqkSq[col] ?? 0;
    return Object.freeze({
      world,
      tensorLabel: rec.partId === 'gqa_sqk_raw' ? `Raw sqk[${col}]` : `Effective sqk^2[${col}] (Q-Only)`,
      coordLabel: `Q-head h=${qh}, dim m=${col % dH} · sigma_0=0.0177`,
      badge: 'FAMILY 2 · SQUARED ON Q ONLY (K UNSCALED)',
      equation: `(${fmtSigned(raw, 4)} * sqrt(8) / 0.0177)^2 = ${effSq.toFixed(3)}`,
      valueText: rec.partId === 'gqa_sqk_raw' ? `raw = ${fmtSigned(raw, 4)}` : `= ${effSq.toFixed(4)}`,
    });
  }

  if (rec.partId === 'gqa_s_gate_raw' || rec.partId === 'gqa_s_gate_eff_sq') {
    const qh = Math.floor(col / dH);
    const raw = weights.sGate[col] ?? 0;
    const effSq = attnTrace.effectiveSGateSq[col] ?? 0;
    return Object.freeze({
      world,
      tensorLabel:
        rec.partId === 'gqa_s_gate_raw' ? `Raw s_gate[${col}]` : `Effective s_gate^2[${col}]`,
      coordLabel: `Q-head h=${qh}, dim m=${col % dH} · g_0=sqrt(8)`,
      badge: 'FAMILY 3 · SQUARED INSIDE SIGMOID (DR3)',
      equation: `(${fmtSigned(raw, 4)} * 8^(1/4) / 0.0177)^2 = ${effSq.toFixed(3)}`,
      valueText: rec.partId === 'gqa_s_gate_raw' ? `raw = ${fmtSigned(raw, 4)}` : `= ${effSq.toFixed(4)}`,
    });
  }

  if (rec.partId === 'gqa_gate_values') {
    const gRaw = attnTrace.gPre[idx] ?? 0;
    const sGateSq = attnTrace.effectiveSGateSq[col] ?? 0;
    return Object.freeze({
      world,
      tensorLabel: 'Output Gate sigma(G * s_gate^2)',
      coordLabel: `${tokLabel} · Q${Math.floor(col / dH)}[m=${col % dH}] (ch ${col})`,
      badge: 'INPUT-DEPENDENT SIGMOID GATE (DR3)',
      equation: `sigmoid(${fmtSigned(gRaw, 3)} * ${sGateSq.toFixed(2)})`,
      valueText: `= ${val.toFixed(4)}`,
    });
  }

  if (rec.partId === 'gqa_attn_sub_norm') {
    const rowNorm = vectorL2Norm(attnTrace.attnSubNorm, row * DORY_MINI_CONFIG.qProjOutDim, DORY_MINI_CONFIG.qProjOutDim);
    return Object.freeze({
      world,
      tensorLabel: 'attn_sub_norm = N_64(C * sigma)',
      coordLabel: `${tokLabel} · ch ${col} (Q${Math.floor(col / dH)}[m=${col % dH}])`,
      badge: 'FULL 64-DIM CONCAT L2NORM (NOT PER-HEAD!)',
      equation: `Y_gated / sqrt(||Y_gated||_2^2 + 1e-6) · ||Y_norm||_2 = ${rowNorm.toFixed(4)}`,
      valueText: `= ${fmtSigned(val, 4)}`,
    });
  }

  if (rec.partId === 'gqa_nlerp_output' || rec.partId === 'gqa_attn_alpha_eff') {
    const eff = attnTrace.nlerp.effectiveAlpha[col] ?? 0;
    const isExtrap = eff > 1.0;
    return Object.freeze({
      world,
      tensorLabel:
        rec.partId === 'gqa_attn_alpha_eff'
          ? `Effective |attn_alpha[${col}]|`
          : `NLERP Residual Output x_next`,
      coordLabel:
        rec.partId === 'gqa_attn_alpha_eff'
          ? `ch c=${col} · ${isExtrap ? 'UNCLAMPED > 1 EXTRAPOLATION' : 'INTERPOLATION'}`
          : `${tokLabel} · ch c=${col}`,
      badge: isExtrap ? 'DR7 UNCLAMPED |alpha_hat| > 1' : DORY_NLERP_MARKER,
      equation: `N_8(a + |alpha_hat|*(b - a)) · |alpha_hat[${col}]|=${eff.toFixed(4)}`,
      valueText: rec.partId === 'gqa_attn_alpha_eff' ? `= ${eff.toFixed(4)}` : `= ${fmtSigned(val, 4)}`,
    });
  }

  const headDesc =
    rec.gridCols === DORY_MINI_CONFIG.qProjOutDim
      ? `Q${Math.floor(col / dH)}[m=${col % dH}]`
      : rec.gridCols === DORY_MINI_CONFIG.kvProjOutDim
        ? `KV${Math.floor(col / dH)}[m=${col % dH}]`
        : `ch c=${col}`;

  return Object.freeze({
    world,
    tensorLabel: rec.label,
    coordLabel: `${tokLabel} · ${headDesc}`,
    badge:
      rec.partId === 'gqa_q_rot' || rec.partId === 'gqa_k_rot'
        ? `NEOX DUAL-ROPE BEFORE L2NORM (DR6 · theta=${metadata.ropeThetaMini})`
        : rec.partId === 'gqa_k_cache'
          ? 'POST-ROPE PER-HEAD L2NORM · UNSCALED CACHE KEY'
          : rec.partId === 'gqa_v_new' || rec.partId === 'gqa_g_pre'
            ? 'UNNORMALIZED PROJECTION (NO L2NORM)'
            : `Phys L${metadata.physicalLayerIdx} · Pair L${metadata.pairedLayerIdx}`,
    equation: `${rec.tensorKey}[${row}, ${col}]`,
    valueText: `= ${fmtSigned(val, 4)}`,
  });
}

// ============================================================================
// MAIN VIEW 3 SCENE BUILDER (`buildDoryGqaRopeScene`)
// ============================================================================

/**
 * Builds the complete headless 5-pass WebGL2 spatial scene for **View 3 (`gqa_rope`)**:
 * Gated GQA & Dual-RoPE Cutaway (`SWA W=8, theta_S=16` vs. `Full, theta_F=1024`).
 *
 * Strictly read-only — never mutates `DorySession` or its KV caches. Enforces `DR20`
 * live `RG32F` `G`-channel masking on every one of the 16 score/softmax plates and
 * self-audits finiteness, zero overlaps, and `<= 512 KiB` before returning.
 */
export function buildDoryGqaRopeScene(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  options: DoryGqaRopeSceneOptions = {},
): DorySpatialScene {
  const ctx = ensureDorySessionContext(sessionOrCtx, options.recordId);
  const config = ctx.params.config;
  const d = config.hiddenSize; // 8
  const hQ = config.numAttentionHeads; // 8
  const hKv = config.numKeyValueHeads; // 2
  const kvGroupSize = config.kvGroupSize; // 4
  const dH = config.headDim; // 8
  const qDim = config.qProjOutDim; // 64
  const kvDim = config.kvProjOutDim; // 16

  const reconstructed = ctx.resolveStepTrace();
  const trace = ctx.resolveForwardTrace();

  const physicalLayerIdx = resolveAttentionPhysicalLayerIdx(options.physicalLayerIdx);
  const layerSpec = getDoryPhysicalLayerSpec(physicalLayerIdx);
  const stage = getDoryStage(physicalLayerIdx);
  const loopIteration =
    stage === 'recurrent' ? clampInt(options.loopIteration ?? 1, 1, ctx.kLoops) : 0;

  const sublayerRecord = ctx.resolveSublayerRecord(physicalLayerIdx, loopIteration);
  if (sublayerRecord.trace.kind !== 'attention') {
    throw new Error(
      `View 3 gqaRopeScene expected attention trace at physicalLayerIdx=${physicalLayerIdx}, got '${sublayerRecord.trace.kind}'.`,
    );
  }
  const attnTrace: DoryAttentionSublayerTrace = sublayerRecord.trace;
  const layerWeights = ctx.params.layers[physicalLayerIdx]!;
  if (layerWeights.kind !== 'attention') {
    throw new Error(`Expected attention weights at physicalLayerIdx=${physicalLayerIdx}.`);
  }

  const seqLen = attnTrace.seqLen;
  const kvLen = attnTrace.kvLen;
  const isSwa = attnTrace.attentionKind === 'sliding_attention';
  const ropeTable =
    attnTrace.ropeRegime === 'sliding' ? ctx.ropeTables.sliding : ctx.ropeTables.full;

  const selectedQHead = clampInt(options.selectedQHead ?? 0, 0, hQ - 1);
  const selectedKvHead = Math.floor(selectedQHead / kvGroupSize);
  const selectedQueryRow = clampInt(
    options.selectedQueryRow ?? ctx.resolveDefaultTokenOffset(),
    0,
    Math.max(0, seqLen - 1),
  );
  const selectedKeyCol = clampInt(
    options.selectedKeyCol ?? resolveDefaultKeyCol(attnTrace, selectedQHead, selectedQueryRow),
    0,
    Math.max(0, kvLen - 1),
  );
  const selectedQueryAbsPos = attnTrace.startPos + selectedQueryRow;
  const selectedKeyAbsPos = attnTrace.kvAbsPositions[selectedKeyCol] ?? selectedQueryAbsPos;
  const selectedRotaryPlane = clampInt(options.selectedRotaryPlane ?? 0, 0, config.rotaryPlanes - 1);

  const traceTokens = resolveTraceTokens(ctx, reconstructed, trace);
  const tokenLabels = Object.freeze(
    traceTokens.map((tokId, i) => `p${attnTrace.startPos + i}:${decodeDoryToken(tokId)}`),
  );
  const kvColLabels = Object.freeze(
    Array.from({ length: kvLen }, (_, ki) => {
      const absPos = attnTrace.kvAbsPositions[ki] ?? ki;
      const writer = attnTrace.kvWriterLoops[ki] ?? 0;
      const tokId =
        absPos >= 0 && absPos < ctx.record.committedSequence.length
          ? ctx.record.committedSequence[absPos]!
          : ki >= attnTrace.pastLen
            ? (traceTokens[ki - attnTrace.pastLen] ?? 0)
            : 0;
      const loopSuffix = stage === 'recurrent' ? ` t${writer}` : '';
      return `k${absPos}:${decodeDoryToken(tokId)}${loopSuffix}`;
    }),
  );
  const qChannelLabels = Object.freeze(
    Array.from({ length: qDim }, (_, c) => `h${Math.floor(c / dH)}.${c % dH}`),
  );
  const kvChannelLabels = Object.freeze(
    Array.from({ length: kvDim }, (_, c) => `kv${Math.floor(c / dH)}.${c % dH}`),
  );
  const dChannelLabels = Object.freeze(Array.from({ length: d }, (_, c) => `d${c}`));

  // Visibility counts & inverted mask for ActiveSheetCellGrid (`1` = masked -inf)
  let visibleCellCount = 0;
  let maskedCellCount = 0;
  const maskInverted = new Uint8Array(seqLen * kvLen);
  for (let i = 0; i < seqLen * kvLen; i += 1) {
    if (attnTrace.maskVisible[i] === 1) {
      visibleCellCount += 1;
      maskInverted[i] = 0;
    } else {
      maskedCellCount += 1;
      maskInverted[i] = 1;
    }
  }

  // Exact 4-plane NeoX Dual-RoPE phasors
  const ropePhasors = computeDoryRopePhasorPlanes({
    attnTrace,
    ropeTable,
    postStepKvCache: reconstructed.postStepKvCache,
    selectedQHead,
    selectedKvHead,
    selectedQueryRow,
    selectedKeyCol,
  });

  const sqkScaleRange = computeArrayMinMax(attnTrace.effectiveSqkSq);
  const sGateScaleRange = computeArrayMinMax(attnTrace.effectiveSGateSq);
  const attnSubNormL2Norm = vectorL2Norm(attnTrace.attnSubNorm, selectedQueryRow * qDim, qDim);

  const qExpansionRatioMini = config.qProjOutDim / config.hiddenSize; // 8.0
  const qExpansionRatioProd = DORY_PRODUCTION_SPEC.qProjOutDim / DORY_PRODUCTION_SPEC.hiddenSize; // 1.6
  const kvDimRatioMini = config.kvProjOutDim / config.hiddenSize; // 2.0
  const kvDimRatioProd = DORY_PRODUCTION_SPEC.kvProjOutDim / DORY_PRODUCTION_SPEC.hiddenSize; // 0.4
  const qExpansionDisclosure =
    `Mini q_dim=${qDim} (${qExpansionRatioMini.toFixed(1)}x d=${d}), kv_dim=${kvDim} (${kvDimRatioMini.toFixed(1)}x d=${d}) ` +
    `vs. Prod q_dim=${DORY_PRODUCTION_SPEC.qProjOutDim} (${qExpansionRatioProd.toFixed(2)}x d=${DORY_PRODUCTION_SPEC.hiddenSize}), ` +
    `kv_dim=${DORY_PRODUCTION_SPEC.kvProjOutDim} (${kvDimRatioProd.toFixed(2)}x d=${DORY_PRODUCTION_SPEC.hiddenSize}) · 4:1 GQA`;

  const metadata: DoryGqaRopeViewMetadata = Object.freeze({
    physicalLayerIdx,
    pairedLayerIdx: layerSpec.pairedLayerIdx,
    stage,
    loopIteration,
    attentionKind: attnTrace.attentionKind,
    ropeRegime: attnTrace.ropeRegime,
    ropeThetaMini: layerSpec.ropeThetaMini ?? config.ropeThetaSliding,
    ropeThetaProd: layerSpec.ropeThetaProd ?? DORY_PRODUCTION_SPEC.ropeThetaSliding,
    slidingWindowMini: isSwa ? config.slidingWindow : null,
    slidingWindowProd: isSwa ? DORY_PRODUCTION_SPEC.slidingWindow : null,
    selectedQHead,
    selectedKvHead,
    selectedQueryRow,
    selectedKeyCol,
    selectedQueryAbsPos,
    selectedKeyAbsPos,
    selectedRotaryPlane,
    seqLen,
    startPos: attnTrace.startPos,
    pastLen: attnTrace.pastLen,
    kvLen,
    bankSlotId: attnTrace.bankSlotId,
    bankKind: attnTrace.bankKind,
    writeKv: attnTrace.writeKv,
    visibleCellCount,
    maskedCellCount,
    qExpansionDisclosure,
    qExpansionRatioMini,
    qExpansionRatioProd,
    kvDimRatioMini,
    kvDimRatioProd,
    sqkScaleRange,
    sGateScaleRange,
    attnSubNormAxisDim: qDim,
    attnSubNormL2Norm,
    ropeBeforeQkNorm: true,
    extrapolatedAlphaCount: attnTrace.nlerp.extrapolatedChannelCount,
    ropePhasors,
    kvAbsPositions: Object.freeze(Array.from(attnTrace.kvAbsPositions)),
    kvWriterLoops: Object.freeze(Array.from(attnTrace.kvWriterLoops)),
    hasLiveCausalMaskGChannel: true,
  });

  // ==========================================================================
  // 1. PACK LIVE RG32F ATLAS (<= 512 KiB, 0 OVERLAPS, LIVE G-CHANNEL DR20)
  // ==========================================================================
  const platesPerShelf = Math.max(1, Math.floor(DORY_GQA_ROPE_ATLAS_WIDTH / Math.max(1, kvLen)));
  const causalShelfCount = Math.ceil((2 * hQ) / platesPerShelf);
  const neededRows = 3 * seqLen + causalShelfCount * seqLen + 4;
  const atlasHeight = Math.min(256, Math.max(16, Math.ceil(neededRows / 16) * 16));
  const atlasPacker = new DoryAtlasPacker(
    DORY_GQA_ROPE_ATLAS_WIDTH,
    atlasHeight,
    DORY_GQA_ROPE_ATLAS_BUDGET_BYTES,
  );

  // Pack all height-seqLen activation matrices first (packs tightly into exactly 3 shelves of width 256)
  const sliceQPre = atlasPacker.packMatrix('gqa_q_pre', attnTrace.qPre, seqLen, qDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceQRot = atlasPacker.packMatrix('gqa_q_rot', attnTrace.qRot, seqLen, qDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceQNorm = atlasPacker.packMatrix('gqa_q_norm', attnTrace.qNorm, seqLen, qDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceQScaled = atlasPacker.packMatrix('gqa_q_scaled', attnTrace.qScaled, seqLen, qDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });

  const sliceGPre = atlasPacker.packMatrix('gqa_g_pre', attnTrace.gPre, seqLen, qDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceContext = atlasPacker.packMatrix(
    'gqa_attn_context',
    attnTrace.attnContext,
    seqLen,
    qDim,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceGateValues = atlasPacker.packMatrix(
    'gqa_gate_values',
    attnTrace.gateValues,
    seqLen,
    qDim,
    { mode: 1, fixedScale: 1.0, physicalLayerIdx, loopIteration },
  );
  const sliceGatedContext = atlasPacker.packMatrix(
    'gqa_gated_context',
    attnTrace.gatedContext,
    seqLen,
    qDim,
    { mode: 0, physicalLayerIdx, loopIteration },
  );

  const sliceAttnSubNorm = atlasPacker.packMatrix(
    'gqa_attn_sub_norm',
    attnTrace.attnSubNorm,
    seqLen,
    qDim,
    { mode: 0, physicalLayerIdx, loopIteration },
  );
  const sliceKPre = atlasPacker.packMatrix('gqa_k_pre', attnTrace.kPre, seqLen, kvDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceKRot = atlasPacker.packMatrix('gqa_k_rot', attnTrace.kRot, seqLen, kvDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceKCache = atlasPacker.packMatrix('gqa_k_cache', attnTrace.kCache, seqLen, kvDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceVNew = atlasPacker.packMatrix('gqa_v_new', attnTrace.vNew, seqLen, kvDim, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceHIn = atlasPacker.packMatrix('gqa_h_in', attnTrace.nlerp.aNorm, seqLen, d, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceOProj = atlasPacker.packMatrix('gqa_o_proj_out', attnTrace.oProjOut, seqLen, d, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceNlerpOut = atlasPacker.packMatrix('gqa_nlerp_output', attnTrace.nlerp.output, seqLen, d, {
    mode: 0,
    physicalLayerIdx,
    loopIteration,
  });

  // Pack all 8 query heads' Causal Score & Softmax plates via packCausalMatrix (DR20 G-channel mask!)
  const headScoreSlices: AtlasSlice[] = [];
  const headProbSlices: AtlasSlice[] = [];
  const headScoreArrays: Float32Array[] = [];
  const headProbArrays: Float32Array[] = [];
  const headMatrixLen = seqLen * kvLen;

  for (let qh = 0; qh < hQ; qh += 1) {
    const start = qh * headMatrixLen;
    const scoreArr = attnTrace.scores.subarray(start, start + headMatrixLen);
    const probArr = attnTrace.probs.subarray(start, start + headMatrixLen);
    headScoreArrays.push(scoreArr);
    headProbArrays.push(probArr);

    headScoreSlices.push(
      atlasPacker.packCausalMatrix(
        `gqa_scores_h${qh}`,
        scoreArr,
        attnTrace.maskVisible,
        seqLen,
        kvLen,
        0,
        { physicalLayerIdx, loopIteration },
      ),
    );
    headProbSlices.push(
      atlasPacker.packCausalMatrix(
        `gqa_probs_h${qh}`,
        probArr,
        attnTrace.maskVisible,
        seqLen,
        kvLen,
        1,
        { fixedScale: 1.0, physicalLayerIdx, loopIteration },
      ),
    );
  }

  // Pack 1-row nGPT scale strips (raw & effective squared sqk^2, s_gate^2, and |attn_alpha|)
  const sliceSqkRaw = atlasPacker.packMatrix('gqa_sqk_raw', layerWeights.sqk, 1, qDim, {
    mode: 0,
    minFloor: 0.02,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceSqkEffSq = atlasPacker.packMatrix(
    'gqa_sqk_eff_sq',
    attnTrace.effectiveSqkSq,
    1,
    qDim,
    {
      mode: 1,
      fixedScale: 1.0 / Math.max(1.0, sqkScaleRange[1]),
      physicalLayerIdx,
      loopIteration,
    },
  );
  const sliceSGateRaw = atlasPacker.packMatrix('gqa_s_gate_raw', layerWeights.sGate, 1, qDim, {
    mode: 0,
    minFloor: 0.02,
    physicalLayerIdx,
    loopIteration,
  });
  const sliceSGateEffSq = atlasPacker.packMatrix(
    'gqa_s_gate_eff_sq',
    attnTrace.effectiveSGateSq,
    1,
    qDim,
    {
      mode: 1,
      fixedScale: 1.0 / Math.max(1.0, sGateScaleRange[1]),
      physicalLayerIdx,
      loopIteration,
    },
  );
  const sliceAttnAlphaEff = atlasPacker.packMatrix(
    'gqa_attn_alpha_eff',
    attnTrace.nlerp.effectiveAlpha,
    1,
    d,
    {
      mode: 1,
      fixedScale: 1.0 / Math.max(1.0, attnTrace.nlerp.maxEffectiveAlpha),
      physicalLayerIdx,
      loopIteration,
    },
  );

  const atlasExtraTag = `b=${ctx.branch.branchId}:r=${ctx.record.recordId}:p=${physicalLayerIdx}:t=${loopIteration}:h=${selectedQHead}`;
  const packedAtlas = atlasPacker.build('gqa_rope', atlasExtraTag);

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
  const kvDepthTicks: KvDepthTickAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const sheetsByPartId = new Map<string, PlacedGqaSheet>();

  const sheetH = 0.068;
  const ex8 = d * 0.24; // 1.92
  const ex16 = kvDim * 0.16; // 2.56
  const ex64 = qDim * 0.068; // 4.352
  const seqEz = Math.max(0.56, Math.min(2.6, seqLen * 0.16));
  const stripEz = 0.34;

  const plateEx = Math.max(0.78, Math.min(1.24, kvLen * 0.085));
  const plateEz = Math.max(0.62, Math.min(1.06, seqLen * 0.075));
  const headPitchZ = plateEz + 0.36;
  const kvGroupPitchX = plateEx + 0.38;

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
    readonly slice: AtlasSlice;
    readonly values: Float32Array;
    readonly masked?: Uint8Array;
    readonly format: 'signed' | 'prob';
    readonly colLabels: readonly string[];
    readonly rowLabels: readonly string[];
    readonly colDimLabel: string;
    readonly rowDimLabel: string;
    readonly activeRow?: number;
    readonly activeCol?: number;
    readonly qHeadIndex?: number;
    readonly kvHeadIndex?: number;
    readonly includeActiveGrid?: boolean;
    readonly extraTags?: Readonly<Record<string, string | number | boolean>>;
  }): PlacedGqaSheet => {
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
      physicalLayerIdx,
      executedLayerIdx: sublayerRecord.executedLayerIdx,
      stage,
      loopIteration,
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      qHeadIndex: spec.qHeadIndex ?? null,
      kvHeadIndex: spec.kvHeadIndex ?? null,
      kvBankSlotId: attnTrace.bankSlotId,
      kvBankKind: attnTrace.bankKind,
      kvPolicy: ctx.kvCacheMode,
      cacheRole: sublayerRecord.coord.cacheOwnership.role,
      absTokenPos: selectedQueryAbsPos,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      modelWeightId: ctx.params.modelWeightId,
      groupRole: spec.kvHeadIndex !== undefined ? 'kv_head' : 'bank',
      tapRole: 'loop',
      extraTags: {
        view: 'gqa_rope',
        attentionKind: attnTrace.attentionKind,
        ropeRegime: attnTrace.ropeRegime,
        writeKv: attnTrace.writeKv,
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
      activeRow: spec.activeRow,
      activeCol: spec.activeCol,
      colLabels: spec.colLabels,
      rowLabels: spec.rowLabels,
      colDimLabel: spec.colDimLabel,
      rowDimLabel: spec.rowDimLabel,
    });
    if (spec.includeActiveGrid !== false) {
      activeCellGrids.push(grid);
    }

    const placed: PlacedGqaSheet = {
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      label: spec.label,
      frame,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      values: spec.values,
      masked: spec.masked,
      grid,
      qHeadIndex: spec.qHeadIndex,
      kvHeadIndex: spec.kvHeadIndex,
    };
    sheetsByPartId.set(spec.partId, placed);
    return placed;
  };

  // Drafting-table stage boundary guides
  lines.addBoxFrameXZ(-17.0, -9.2, 2.4, 7.8, -0.06, 1.0, LINE_STYLE_GUIDE, 0.34);
  lines.addBoxFrameXZ(2.9, -9.2, 12.1, 7.8, -0.06, 1.0, LINE_STYLE_GUIDE, 0.36);
  lines.addBoxFrameXZ(12.6, -9.2, 26.4, 7.8, -0.06, 1.0, LINE_STYLE_GUIDE, 0.34);

  // --- COL 0: Input Residual State ---
  const sheetHIn = registerSheet({
    partId: 'gqa_h_in',
    tensorKey: 'nlerp_a_norm',
    label: physicalLayerIdx === 0 ? 'h_0 (Raw Unit Embed · DR5)' : 'h_in (Unit Sphere)',
    cx: -15.6,
    cy: 0,
    cz: 0,
    ex: ex8,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_OPERAND_A,
    slice: sliceHIn,
    values: attnTrace.nlerp.aNorm,
    format: 'signed',
    colLabels: dChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
  });

  // --- COL 1: Bias-Free Projections (Q: 8->64 [8x], K/V: 8->16 [2x], Gate: 8->64 [8x]) ---
  const zQ = -5.8;
  const zK = -1.8;
  const zV = 1.8;
  const zG = 5.8;

  const sheetQPre = registerSheet({
    partId: 'gqa_q_pre',
    tensorKey: 'q_pre',
    label: 'Q_pre = h W_q^T (8->64, 8.0x mini / 1.60x prod)',
    cx: -11.4,
    cy: 0,
    cz: zQ,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_A,
    slice: sliceQPre,
    values: attnTrace.qPre,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim} (8.0x)`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    activeCol: selectedQHead * dH + selectedRotaryPlane,
    includeActiveGrid: false,
  });

  const sheetKPre = registerSheet({
    partId: 'gqa_k_pre',
    tensorKey: 'k_pre',
    label: 'K_pre = h W_k^T (8->16, 2.0x mini / 0.40x prod)',
    cx: -11.4,
    cy: 0,
    cz: zK,
    ex: ex16,
    ey: sheetH,
    ez: seqEz,
    gridCols: kvDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_K,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceKPre,
    values: attnTrace.kPre,
    format: 'signed',
    colLabels: kvChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `kv_dim=${kvDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    activeCol: selectedKvHead * dH + selectedRotaryPlane,
    includeActiveGrid: false,
  });

  const sheetVNew = registerSheet({
    partId: 'gqa_v_new',
    tensorKey: 'v_new',
    label: 'V = h W_v^T (Unnormalized)',
    cx: -11.4,
    cy: 0,
    cz: zV,
    ex: ex16,
    ey: sheetH,
    ez: seqEz,
    gridCols: kvDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_V,
    highlight: SHEET_HL_NORMAL,
    slice: sliceVNew,
    values: attnTrace.vNew,
    format: 'signed',
    colLabels: kvChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `kv_dim=${kvDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    includeActiveGrid: false,
  });

  const sheetGPre = registerSheet({
    partId: 'gqa_g_pre',
    tensorKey: 'g_pre',
    label: 'G = h W_g^T (Unnormalized Gate Pre-Act)',
    cx: -11.4,
    cy: 0,
    cz: zG,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_NORMAL,
    slice: sliceGPre,
    values: attnTrace.gPre,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    includeActiveGrid: false,
  });

  // --- COL 2: Dual-RoPE (100% NeoX rotate_half strictly BEFORE Q/K L2Norm — DR6) ---
  const sheetQRot = registerSheet({
    partId: 'gqa_q_rot',
    tensorKey: 'q_rot',
    label: `Q_rot = RoPE(Q_pre, theta=${metadata.ropeThetaMini})`,
    cx: -5.9,
    cy: 0,
    cz: zQ,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceQRot,
    values: attnTrace.qRot,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    activeCol: selectedQHead * dH + selectedRotaryPlane,
    includeActiveGrid: false,
  });

  const sheetKRot = registerSheet({
    partId: 'gqa_k_rot',
    tensorKey: 'k_rot',
    label: `K_rot = RoPE(K_pre, theta=${metadata.ropeThetaMini})`,
    cx: -5.9,
    cy: 0,
    cz: zK,
    ex: ex16,
    ey: sheetH,
    ez: seqEz,
    gridCols: kvDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_K,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceKRot,
    values: attnTrace.kRot,
    format: 'signed',
    colLabels: kvChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `kv_dim=${kvDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    activeCol: selectedKvHead * dH + selectedRotaryPlane,
    includeActiveGrid: false,
  });

  // 3D 4-Plane Dual-RoPE Phasor Pedestal (z = +2.65, x = -5.9)
  const phasorBaseX = -5.9;
  const phasorZ = 2.65;
  const phasorY = 0.03;
  const phasorPitchX = 1.12;
  const phasorR = 0.42;
  const ropeRingStyle = isSwa ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD;

  lines.addDashedFrameXZ(
    phasorBaseX - 2.38,
    phasorZ - 0.72,
    phasorBaseX + 2.38,
    phasorZ + 0.72,
    0.01,
    ropeRingStyle,
    0.58,
    0.18,
    0.12,
    0.95,
  );

  for (let j = 0; j < ropePhasors.length; j += 1) {
    const plane = ropePhasors[j]!;
    const px = phasorBaseX + (j - 1.5) * phasorPitchX;
    const isSelectedPlane = j === selectedRotaryPlane;

    for (let seg = 0; seg < PHASOR_RING_SEGMENTS; seg += 1) {
      const a0 = (seg / PHASOR_RING_SEGMENTS) * Math.PI * 2;
      const a1 = ((seg + 1) / PHASOR_RING_SEGMENTS) * Math.PI * 2;
      lines.addSegment(
        px + Math.cos(a0) * phasorR,
        phasorY,
        phasorZ - Math.sin(a0) * phasorR,
        px + Math.cos(a1) * phasorR,
        phasorY,
        phasorZ - Math.sin(a1) * phasorR,
        isSelectedPlane ? 1.35 : 0.9,
        isSelectedPlane ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
        isSelectedPlane ? 0.88 : 0.56,
      );
    }

    // Unit-normalized query & key rotated phasor arms inside plane j
    const qNormLen = Math.hypot(plane.qRotPair[0], plane.qRotPair[1]);
    const qUx = qNormLen > 1e-8 ? plane.qRotPair[0] / qNormLen : Math.cos(plane.queryAngleRad);
    const qUy = qNormLen > 1e-8 ? plane.qRotPair[1] / qNormLen : Math.sin(plane.queryAngleRad);
    const kNormLen = Math.hypot(plane.kRotPair[0], plane.kRotPair[1]);
    const kUx = kNormLen > 1e-8 ? plane.kRotPair[0] / kNormLen : Math.cos(plane.keyAngleRad);
    const kUy = kNormLen > 1e-8 ? plane.kRotPair[1] / kNormLen : Math.sin(plane.keyAngleRad);

    const qTipX = px + qUx * phasorR * 0.9;
    const qTipZ = phasorZ - qUy * phasorR * 0.9;
    const kTipX = px + kUx * phasorR * 0.78;
    const kTipZ = phasorZ - kUy * phasorR * 0.78;

    lines.addSegment(px, phasorY + 0.04, phasorZ, qTipX, phasorY + 0.04, qTipZ, 1.35, LINE_STYLE_ACTIVE, 0.92);
    lines.addSegment(px, phasorY + 0.03, phasorZ, kTipX, phasorY + 0.03, kTipZ, 1.2, LINE_STYLE_BF16_GOLD, 0.86);

    solids.push({
      cx: qTipX,
      cy: phasorY + 0.09,
      cz: qTipZ,
      ex: 0.11,
      ey: 0.11,
      ez: 0.11,
      cellFamily: isSwa ? CELL_SSM_CYAN : CELL_WEIGHT_BLUE,
      bayId: physicalLayerIdx + 1,
      cellScale: 0.12,
      highlight: isSelectedPlane ? 1 : 0,
    });
    solids.push({
      cx: kTipX,
      cy: phasorY + 0.08,
      cz: kTipZ,
      ex: 0.095,
      ey: 0.095,
      ez: 0.095,
      cellFamily: CELL_MOE_AMBER,
      bayId: physicalLayerIdx + 1,
      cellScale: 0.12,
      highlight: isSelectedPlane ? 2 : 0,
    });

    const planeMeta = encodeDoryPickMetadata({
      physicalLayerIdx,
      executedLayerIdx: sublayerRecord.executedLayerIdx,
      stage,
      loopIteration,
      partId: `gqa_rope_plane_${j}`,
      tensorKey: 'rope_cos',
      qHeadIndex: selectedQHead,
      kvHeadIndex: selectedKvHead,
      rotaryPlane: j,
      absTokenPos: selectedQueryAbsPos,
      modelWeightId: ctx.params.modelWeightId,
      groupRole: 'kv_head',
      tapRole: 'plane',
      extraTags: {
        view: 'gqa_rope',
        ropeRegime: attnTrace.ropeRegime,
        theta: metadata.ropeThetaMini,
      },
    });
    pickBoxes.push(
      packPickBoxFromAabb(px, phasorY + 0.08, phasorZ, phasorR * 2.05, 0.24, phasorR * 2.05, planeMeta),
    );
    aabbs.push({
      min: [px - phasorR - 0.1, 0, phasorZ - phasorR - 0.1],
      max: [px + phasorR + 0.1, 0.26, phasorZ + phasorR + 0.1],
    });
  }

  // --- COL 3: Per-Head L2Norm (`Q_norm`, `K_cache` Unscaled) + Q-Only `sqk` / `sqk^2` Strips ---
  const sheetSqkRaw = registerSheet({
    partId: 'gqa_sqk_raw',
    tensorKey: 'sqk',
    label: 'Raw sqk (init 0.0177)',
    cx: -0.4,
    cy: 0,
    cz: -8.35,
    ex: ex64,
    ey: sheetH,
    ez: stripEz,
    gridCols: qDim,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_NORMAL,
    slice: sliceSqkRaw,
    values: layerWeights.sqk,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: Object.freeze(['raw']),
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  const sheetSqkEffSq = registerSheet({
    partId: 'gqa_sqk_eff_sq',
    tensorKey: 'sqk',
    label: 'Effective sqk^2 = (sqk * sqrt(8) / 0.0177)^2 (Q-Only!)',
    cx: -0.4,
    cy: 0,
    cz: -7.72,
    ex: ex64,
    ey: sheetH,
    ez: stripEz,
    gridCols: qDim,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: sliceSqkEffSq,
    values: attnTrace.effectiveSqkSq,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: Object.freeze(['sqk^2']),
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  const sheetQNorm = registerSheet({
    partId: 'gqa_q_norm',
    tensorKey: 'q_norm',
    label: 'Q_norm = N_8(Q_rot) per-head',
    cx: -0.4,
    cy: 0,
    cz: zQ,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_OPERAND_A,
    slice: sliceQNorm,
    values: attnTrace.qNorm,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    includeActiveGrid: false,
  });

  const sheetKCache = registerSheet({
    partId: 'gqa_k_cache',
    tensorKey: 'k_cache',
    label: 'K_cache = N_8(K_rot) (Unit Norm · No sqk)',
    cx: -0.4,
    cy: 0,
    cz: zK,
    ex: ex16,
    ey: sheetH,
    ez: seqEz,
    gridCols: kvDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_K,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceKCache,
    values: attnTrace.kCache,
    format: 'signed',
    colLabels: kvChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `kv_dim=${kvDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
  });

  // --- COL 4: Q-Scaled + All 8 Head Causal Score Plates (LIVE RG32F G-Channel DR20) ---
  const scoresBankCx = 5.0;
  const probsBankCx = 9.8;

  const sheetQScaled = registerSheet({
    partId: 'gqa_q_scaled',
    tensorKey: 'q_scaled',
    label: 'Q_scaled = Q_norm * sqk^2',
    cx: scoresBankCx,
    cy: 0,
    cz: zQ,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_ACTIVE,
    slice: sliceQScaled,
    values: attnTrace.qScaled,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    activeCol: selectedQHead * dH,
  });

  const scoreSheets: PlacedGqaSheet[] = [];
  const probSheets: PlacedGqaSheet[] = [];

  for (let qh = 0; qh < hQ; qh += 1) {
    const kvh = Math.floor(qh / kvGroupSize); // 0 or 1
    const inGroupRow = qh % kvGroupSize; // 0..3
    const groupOffsetX = (kvh - 0.5) * kvGroupPitchX;
    const plateZ = (inGroupRow - 1.5) * headPitchZ;
    const isFocusedHead = qh === selectedQHead;
    const isSameKvGroup = kvh === selectedKvHead;

    const scoreSheet = registerSheet({
      partId: `gqa_scores_h${qh}`,
      tensorKey: 'scores',
      label: `Scores Q${qh} (KV${kvh})`,
      cx: scoresBankCx + groupOffsetX,
      cy: 0,
      cz: plateZ,
      ex: plateEx,
      ey: sheetH,
      ez: plateEz,
      gridCols: kvLen,
      gridRows: seqLen,
      sheetKind: SHEET_KIND_CAUSAL_SCORES,
      role: SHEET_ROLE_SCORES,
      highlight: isFocusedHead
        ? SHEET_HL_ACTIVE
        : isSameKvGroup
          ? SHEET_HL_PATHWAY
          : SHEET_HL_NORMAL,
      slice: headScoreSlices[qh]!,
      values: headScoreArrays[qh]!,
      masked: maskInverted,
      format: 'signed',
      colLabels: kvColLabels,
      rowLabels: tokenLabels,
      colDimLabel: `KV=${kvLen}`,
      rowDimLabel: `Q=${seqLen}`,
      activeRow: isFocusedHead ? selectedQueryRow : undefined,
      activeCol: isFocusedHead ? selectedKeyCol : undefined,
      qHeadIndex: qh,
      kvHeadIndex: kvh,
      includeActiveGrid: isFocusedHead,
      extraTags: { liveMaskG: true, headGroup: kvh },
    });
    scoreSheets.push(scoreSheet);

    const probSheet = registerSheet({
      partId: `gqa_probs_h${qh}`,
      tensorKey: 'probs',
      label: `Softmax Q${qh} (KV${kvh})`,
      cx: probsBankCx + groupOffsetX,
      cy: 0,
      cz: plateZ,
      ex: plateEx,
      ey: sheetH,
      ez: plateEz,
      gridCols: kvLen,
      gridRows: seqLen,
      sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
      role: SHEET_ROLE_SOFTMAX,
      highlight: isFocusedHead
        ? SHEET_HL_ACTIVE
        : isSameKvGroup
          ? SHEET_HL_PATHWAY
          : SHEET_HL_NORMAL,
      slice: headProbSlices[qh]!,
      values: headProbArrays[qh]!,
      masked: maskInverted,
      format: 'prob',
      colLabels: kvColLabels,
      rowLabels: tokenLabels,
      colDimLabel: `KV=${kvLen}`,
      rowDimLabel: `Q=${seqLen}`,
      activeRow: isFocusedHead ? selectedQueryRow : undefined,
      activeCol: isFocusedHead ? selectedKeyCol : undefined,
      qHeadIndex: qh,
      kvHeadIndex: kvh,
      includeActiveGrid: isFocusedHead,
      extraTags: { liveMaskG: true, headGroup: kvh },
    });
    probSheets.push(probSheet);
  }

  // Highlight active `(selectedQueryRow, selectedKeyCol)` cell on focused Score & Softmax plates
  const activeScoreSheet = scoreSheets[selectedQHead]!;
  const activeProbSheet = probSheets[selectedQHead]!;
  const cellFill = { r: 0.863, g: 0.298, b: 0.204, a: 0.2, borderAlpha: 0.92 };
  for (const activePlate of [activeScoreSheet, activeProbSheet]) {
    flatFlow.addCellHighlightOnFrame(
      activePlate.frame,
      selectedQueryRow,
      selectedQueryRow,
      selectedKeyCol,
      selectedKeyCol,
      seqLen,
      kvLen,
      0.044,
      cellFill,
      DORY_FLOW_COLORS.activeOxide,
      0.018,
      true,
    );
  }

  // Dashed KV-group frames around KV0 (Q0..Q3) and KV1 (Q4..Q7) on both Score & Softmax banks
  const bankHalfZ = 1.5 * headPitchZ + plateEz * 0.5 + 0.2;
  for (const bankCx of [scoresBankCx, probsBankCx]) {
    for (let kvh = 0; kvh < hKv; kvh += 1) {
      const gx = bankCx + (kvh - 0.5) * kvGroupPitchX;
      lines.addDashedFrameXZ(
        gx - plateEx * 0.5 - 0.12,
        -bankHalfZ,
        gx + plateEx * 0.5 + 0.12,
        bankHalfZ,
        0.01,
        kvh === selectedKvHead ? LINE_STYLE_ACTIVE : LINE_STYLE_GUIDE,
        kvh === selectedKvHead ? 0.62 : 0.36,
        0.16,
        0.1,
        0.95,
      );
    }
  }

  // KV depth ticks along focused softmax plate columns (marking absolute KV token positions)
  const tickStep = kvLen <= 10 ? 1 : kvLen <= 20 ? 2 : 4;
  for (let ki = 0; ki < kvLen; ki += tickStep) {
    const absPos = attnTrace.kvAbsPositions[ki] ?? ki;
    const tickWorld = computeSheetCellWorldPos(activeProbSheet.grid, ki, 0, 0.5);
    kvDepthTicks.push(
      Object.freeze({
        token: absPos,
        world: Object.freeze([tickWorld[0], tickWorld[1] + 0.08, tickWorld[2] - 0.24] as const),
      }),
    );
  }

  // --- COL 6..8: Context -> Sigmoid Gate -> `attn_sub_norm` (64-dim) -> `o_proj` -> NLERP ---
  const sheetContext = registerSheet({
    partId: 'gqa_attn_context',
    tensorKey: 'attn_context',
    label: 'Context C = P V [S, 64]',
    cx: 14.8,
    cy: 0,
    cz: -2.2,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceContext,
    values: attnTrace.attnContext,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    includeActiveGrid: false,
  });

  const sheetGateValues = registerSheet({
    partId: 'gqa_gate_values',
    tensorKey: 'gate_values',
    label: 'Gate = sigma(G * s_gate^2) [S, 64]',
    cx: 14.8,
    cy: 0,
    cz: 2.2,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceGateValues,
    values: attnTrace.gateValues,
    format: 'prob',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    includeActiveGrid: false,
  });

  const sheetSGateRaw = registerSheet({
    partId: 'gqa_s_gate_raw',
    tensorKey: 's_gate',
    label: 'Raw s_gate (init 0.0177)',
    cx: 14.8,
    cy: 0,
    cz: 4.55,
    ex: ex64,
    ey: sheetH,
    ez: stripEz,
    gridCols: qDim,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_WEIGHT,
    highlight: SHEET_HL_NORMAL,
    slice: sliceSGateRaw,
    values: layerWeights.sGate,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: Object.freeze(['raw']),
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  const sheetSGateEffSq = registerSheet({
    partId: 'gqa_s_gate_eff_sq',
    tensorKey: 's_gate',
    label: 'Effective s_gate^2 (g_0=sqrt(8), DR3)',
    cx: 14.8,
    cy: 0,
    cz: 5.2,
    ex: ex64,
    ey: sheetH,
    ez: stripEz,
    gridCols: qDim,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: sliceSGateEffSq,
    values: attnTrace.effectiveSGateSq,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: Object.freeze(['s_gate^2']),
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  const sheetGatedContext = registerSheet({
    partId: 'gqa_gated_context',
    tensorKey: 'gated_context',
    label: 'Y_gated = C * sigma(G * s_gate^2)',
    cx: 20.3,
    cy: 0,
    cz: -2.2,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: sliceGatedContext,
    values: attnTrace.gatedContext,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
    includeActiveGrid: false,
  });

  const sheetAttnSubNorm = registerSheet({
    partId: 'gqa_attn_sub_norm',
    tensorKey: 'attn_sub_norm',
    label: 'attn_sub_norm = N_64(Y_gated) (All Heads Concat)',
    cx: 20.3,
    cy: 0,
    cz: 2.2,
    ex: ex64,
    ey: sheetH,
    ez: seqEz,
    gridCols: qDim,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: SHEET_HL_BF16_GOLD,
    slice: sliceAttnSubNorm,
    values: attnTrace.attnSubNorm,
    format: 'signed',
    colLabels: qChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `q_dim=${qDim} (concat!)`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
  });

  const sheetOProj = registerSheet({
    partId: 'gqa_o_proj_out',
    tensorKey: 'o_proj',
    label: 'O = Y_norm W_o^T (64->8)',
    cx: 24.8,
    cy: 0,
    cz: -2.4,
    ex: ex8,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_OPERAND_B,
    slice: sliceOProj,
    values: attnTrace.oProjOut,
    format: 'signed',
    colLabels: dChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
  });

  const hasExtrapAlpha = attnTrace.nlerp.extrapolatedChannelCount > 0;
  const sheetAttnAlphaEff = registerSheet({
    partId: 'gqa_attn_alpha_eff',
    tensorKey: 'attn_alpha',
    label: '|attn_alpha * 0.05 / 0.0177|',
    cx: 24.8,
    cy: 0,
    cz: 0.2,
    ex: ex8,
    ey: sheetH,
    ez: stripEz,
    gridCols: d,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_NORM,
    highlight: hasExtrapAlpha ? SHEET_HL_ACTIVE : SHEET_HL_BF16_GOLD,
    slice: sliceAttnAlphaEff,
    values: attnTrace.nlerp.effectiveAlpha,
    format: 'signed',
    colLabels: dChannelLabels,
    rowLabels: Object.freeze(['|alpha|']),
    colDimLabel: `d=${d}`,
    rowDimLabel: '1',
    includeActiveGrid: false,
  });

  const sheetNlerpOut = registerSheet({
    partId: 'gqa_nlerp_output',
    tensorKey: 'nlerp_output',
    label: 'x_next = N_8(a + |alpha_hat|*(b-a))',
    cx: 24.8,
    cy: 0,
    cz: 2.6,
    ex: ex8,
    ey: sheetH,
    ez: seqEz,
    gridCols: d,
    gridRows: seqLen,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    highlight: SHEET_HL_ACTIVE,
    slice: sliceNlerpOut,
    values: attnTrace.nlerp.output,
    format: 'signed',
    colLabels: dChannelLabels,
    rowLabels: tokenLabels,
    colDimLabel: `d=${d}`,
    rowDimLabel: `S=${seqLen}`,
    activeRow: selectedQueryRow,
  });

  // ==========================================================================
  // 3. FLOW RIBBONS & DIRECTIONAL ARROWS
  // ==========================================================================
  const flowY = -0.025;
  const regimeFlowColor = isSwa ? DORY_FLOW_COLORS.swaCyan : DORY_FLOW_COLORS.fullIndigo;

  // Fan-out from h_in -> Q, K, V, Gate
  for (const [targetSheet, color] of [
    [sheetQPre, DORY_FLOW_COLORS.preludeIndigo],
    [sheetKPre, DORY_FLOW_COLORS.recurrentTeal],
    [sheetVNew, DORY_FLOW_COLORS.swaCyan],
    [sheetGPre, DORY_FLOW_COLORS.codaAmber],
  ] as const) {
    flatFlow.addRoundedRouteXZ(
      [
        { x: sheetHIn.frame.center[0] + ex8 * 0.5 + 0.08, z: sheetHIn.frame.center[2] },
        { x: -14.1, z: sheetHIn.frame.center[2] },
        { x: -14.1, z: targetSheet.frame.center[2] },
        {
          x: targetSheet.frame.center[0] - targetSheet.frame.extent[0] * 0.5 - 0.08,
          z: targetSheet.frame.center[2],
        },
      ],
      flowY,
      0.1,
      color,
      0.24,
      true,
    );
  }

  // Q path: q_pre -> q_rot -> q_norm -> q_scaled -> scores
  flatFlow.addPolylineXZ(
    [
      { x: sheetQPre.frame.center[0] + ex64 * 0.5 + 0.08, z: zQ },
      { x: sheetQRot.frame.center[0] - ex64 * 0.5 - 0.08, z: zQ },
    ],
    flowY,
    0.11,
    regimeFlowColor,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetQRot.frame.center[0] + ex64 * 0.5 + 0.08, z: zQ },
      { x: sheetQNorm.frame.center[0] - ex64 * 0.5 - 0.08, z: zQ },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.preludeIndigo,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetQNorm.frame.center[0] + ex64 * 0.5 + 0.08, z: zQ },
      { x: sheetQScaled.frame.center[0] - ex64 * 0.5 - 0.08, z: zQ },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  // sqk raw -> sqk^2 -> q_scaled
  flatFlow.addPolylineXZ(
    [
      { x: sheetSqkRaw.frame.center[0], z: sheetSqkRaw.frame.center[2] + stripEz * 0.5 + 0.04 },
      { x: sheetSqkEffSq.frame.center[0], z: sheetSqkEffSq.frame.center[2] - stripEz * 0.5 - 0.04 },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetSqkEffSq.frame.center[0] + ex64 * 0.5 + 0.08, z: sheetSqkEffSq.frame.center[2] },
      { x: scoresBankCx, z: sheetSqkEffSq.frame.center[2] },
      { x: scoresBankCx, z: zQ - seqEz * 0.5 - 0.08 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.anchorGold,
    0.24,
    true,
  );
  // q_scaled -> scores bank
  flatFlow.addPolylineXZ(
    [
      { x: scoresBankCx, z: zQ + seqEz * 0.5 + 0.08 },
      { x: scoresBankCx, z: -bankHalfZ - 0.08 },
    ],
    flowY,
    0.12,
    regimeFlowColor,
    true,
  );

  // K path: k_pre -> k_rot -> k_cache -> scores bank
  flatFlow.addPolylineXZ(
    [
      { x: sheetKPre.frame.center[0] + ex16 * 0.5 + 0.08, z: zK },
      { x: sheetKRot.frame.center[0] - ex16 * 0.5 - 0.08, z: zK },
    ],
    flowY,
    0.11,
    regimeFlowColor,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetKRot.frame.center[0] + ex16 * 0.5 + 0.08, z: zK },
      { x: sheetKCache.frame.center[0] - ex16 * 0.5 - 0.08, z: zK },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetKCache.frame.center[0] + ex16 * 0.5 + 0.08, z: zK },
      { x: scoresBankCx - kvGroupPitchX * 0.5 - plateEx * 0.5 - 0.16, z: zK },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );

  // Scores -> Softmax -> Context
  flatFlow.addPolylineXZ(
    [
      { x: scoresBankCx + kvGroupPitchX * 0.5 + plateEx * 0.5 + 0.16, z: 0 },
      { x: probsBankCx - kvGroupPitchX * 0.5 - plateEx * 0.5 - 0.16, z: 0 },
    ],
    flowY,
    0.13,
    DORY_FLOW_COLORS.nlerpEmerald,
    true,
  );
  ribbons.push({
    cx: (scoresBankCx + probsBankCx) * 0.5,
    cy: 0.14,
    cz: 0,
    ex: 0.72,
    ey: 0.07,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: isSwa ? RIBBON_COLOR_CYAN : RIBBON_COLOR_PERIWINKLE,
    alpha: 0.44,
  });

  flatFlow.addRoundedRouteXZ(
    [
      { x: probsBankCx + kvGroupPitchX * 0.5 + plateEx * 0.5 + 0.16, z: 0 },
      { x: 12.15, z: 0 },
      { x: 12.15, z: sheetContext.frame.center[2] },
      { x: sheetContext.frame.center[0] - ex64 * 0.5 - 0.08, z: sheetContext.frame.center[2] },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.nlerpEmerald,
    0.24,
    true,
  );

  // V bypass route -> Context (unnormalized V joins P * V)
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetVNew.frame.center[0], z: sheetVNew.frame.center[2] + seqEz * 0.5 + 0.08 },
      { x: sheetVNew.frame.center[0], z: 4.15 },
      { x: 12.3, z: 4.15 },
      { x: 12.3, z: sheetContext.frame.center[2] + 0.45 },
      {
        x: sheetContext.frame.center[0] - ex64 * 0.5 - 0.08,
        z: sheetContext.frame.center[2] + 0.45,
      },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.swaCyan,
    0.26,
    true,
  );

  // Gate bypass route: g_pre -> gate_values (with s_gate^2)
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetGPre.frame.center[0] + ex64 * 0.5 + 0.08, z: zG },
      { x: 12.45, z: zG },
      { x: 12.45, z: sheetGateValues.frame.center[2] },
      {
        x: sheetGateValues.frame.center[0] - ex64 * 0.5 - 0.08,
        z: sheetGateValues.frame.center[2],
      },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.codaAmber,
    0.26,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetSGateRaw.frame.center[0], z: sheetSGateRaw.frame.center[2] - stripEz * 0.5 - 0.04 },
      { x: sheetGateValues.frame.center[0], z: sheetGateValues.frame.center[2] + seqEz * 0.5 + 0.08 },
    ],
    flowY,
    0.09,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetSGateEffSq.frame.center[0], z: sheetSGateEffSq.frame.center[2] - stripEz * 0.5 - 0.04 },
      { x: sheetSGateRaw.frame.center[0], z: sheetSGateRaw.frame.center[2] + stripEz * 0.5 + 0.04 },
    ],
    flowY,
    0.08,
    DORY_FLOW_COLORS.anchorGold,
    false,
  );

  // Context & Gate -> Gated Context -> attn_sub_norm -> o_proj -> NLERP output
  flatFlow.addPolylineXZ(
    [
      { x: sheetContext.frame.center[0] + ex64 * 0.5 + 0.08, z: sheetContext.frame.center[2] },
      { x: sheetGatedContext.frame.center[0] - ex64 * 0.5 - 0.08, z: sheetGatedContext.frame.center[2] },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetGateValues.frame.center[0] + ex64 * 0.5 + 0.08, z: sheetGateValues.frame.center[2] },
      { x: 17.55, z: sheetGateValues.frame.center[2] },
      { x: 17.55, z: sheetGatedContext.frame.center[2] + 0.35 },
      {
        x: sheetGatedContext.frame.center[0] - ex64 * 0.5 - 0.08,
        z: sheetGatedContext.frame.center[2] + 0.35,
      },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.codaAmber,
    0.24,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetGatedContext.frame.center[0], z: sheetGatedContext.frame.center[2] + seqEz * 0.5 + 0.08 },
      { x: sheetAttnSubNorm.frame.center[0], z: sheetAttnSubNorm.frame.center[2] - seqEz * 0.5 - 0.08 },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.anchorGold,
    true,
  );
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetAttnSubNorm.frame.center[0] + ex64 * 0.5 + 0.08, z: sheetAttnSubNorm.frame.center[2] },
      { x: 23.15, z: sheetAttnSubNorm.frame.center[2] },
      { x: 23.15, z: sheetOProj.frame.center[2] },
      { x: sheetOProj.frame.center[0] - ex8 * 0.5 - 0.08, z: sheetOProj.frame.center[2] },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.anchorGold,
    0.24,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetOProj.frame.center[0], z: sheetOProj.frame.center[2] + seqEz * 0.5 + 0.08 },
      { x: sheetAttnAlphaEff.frame.center[0], z: sheetAttnAlphaEff.frame.center[2] - stripEz * 0.5 - 0.06 },
    ],
    flowY,
    0.11,
    DORY_FLOW_COLORS.nlerpEmerald,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: sheetAttnAlphaEff.frame.center[0], z: sheetAttnAlphaEff.frame.center[2] + stripEz * 0.5 + 0.06 },
      { x: sheetNlerpOut.frame.center[0], z: sheetNlerpOut.frame.center[2] - seqEz * 0.5 - 0.08 },
    ],
    flowY,
    0.12,
    hasExtrapAlpha ? DORY_FLOW_COLORS.extrapolationCoral : DORY_FLOW_COLORS.nlerpEmerald,
    true,
  );
  // Residual skip bus from h_in along z = +7.35 -> NLERP output
  flatFlow.addRoundedRouteXZ(
    [
      { x: sheetHIn.frame.center[0], z: sheetHIn.frame.center[2] + seqEz * 0.5 + 0.1 },
      { x: sheetHIn.frame.center[0], z: 7.35 },
      { x: sheetNlerpOut.frame.center[0], z: 7.35 },
      { x: sheetNlerpOut.frame.center[0], z: sheetNlerpOut.frame.center[2] + seqEz * 0.5 + 0.1 },
    ],
    flowY,
    0.1,
    DORY_FLOW_COLORS.nlerpEmerald,
    0.28,
    true,
  );
  ribbons.push({
    cx: 23.15,
    cy: 0.14,
    cz: sheetOProj.frame.center[2],
    ex: 0.64,
    ey: 0.07,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_AMBER,
    alpha: 0.42,
  });
  ribbons.push({
    cx: 17.55,
    cy: 0.14,
    cz: sheetGatedContext.frame.center[2],
    ex: 0.64,
    ey: 0.07,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.42,
  });

  // ==========================================================================
  // 4. OVERLAY ANCHORS, LANDMARKS, CALIPERS, BADGES & CALLOUTS
  // ==========================================================================
  const regimeTone: DoryBadgeTone = isSwa ? 'swa_cyan' : 'full_indigo';

  overlayAnchors.push(
    Object.freeze({
      id: 'anchor_gqa_q_exp',
      label: `Q / Gate Expansion (${qDim} = 8.0x mini)`,
      sublabel: 'Prod: 2,560 -> 4,096 (1.60x) · 4:1 GQA',
      worldPos: localToWorldOnFrame(sheetQPre.frame, 0, 0.72, -0.64),
      tone: 'prelude_indigo',
      badge: '8.0x MINI · 1.60x PROD',
    }),
    Object.freeze({
      id: 'anchor_gqa_rope_pedestal',
      label: `Dual-RoPE 4-Plane Phasors (${isSwa ? 'theta_S=16' : 'theta_F=1024'})`,
      sublabel: '100% NeoX rotate_half BEFORE Q/K L2Norm (DR6)',
      worldPos: Object.freeze([phasorBaseX, 0.68, phasorZ + 0.95] as const),
      tone: regimeTone,
      badge: 'DR6 ROPE -> L2NORM',
    }),
    Object.freeze({
      id: 'anchor_gqa_sqk',
      label: 'Q-Only Squared sqk^2',
      sublabel: `[${sqkScaleRange[0].toFixed(2)}, ${sqkScaleRange[1].toFixed(2)}] · Cached K stays unit-norm`,
      worldPos: localToWorldOnFrame(sheetSqkRaw.frame, 0, 0.75, -0.9),
      tone: 'gold',
      badge: 'Q-ONLY',
    }),
    Object.freeze({
      id: 'anchor_gqa_scores_bank',
      label: `8-Head GQA Scores (${isSwa ? 'SWA W=8' : 'Full Causal'})`,
      sublabel: `Live RG32F G-Channel: ${visibleCellCount} vis / ${maskedCellCount} masked · Bank #${attnTrace.bankSlotId}`,
      worldPos: Object.freeze([scoresBankCx, 0.78, bankHalfZ + 0.48] as const),
      tone: regimeTone,
      badge: 'DR20 LIVE G-MASK',
    }),
    Object.freeze({
      id: 'anchor_gqa_probs_bank',
      label: `8-Head Softmax P (KV0: Q0..3, KV1: Q4..7)`,
      sublabel: attnTrace.writeKv
        ? `writeKv=true · ${attnTrace.pastLen} past + ${seqLen} chunk`
        : `writeKv=false (DoryPastReader, DR17)`,
      worldPos: Object.freeze([probsBankCx, 0.78, bankHalfZ + 0.48] as const),
      tone: attnTrace.writeKv ? 'nlerp_emerald' : 'gold',
    }),
    Object.freeze({
      id: 'anchor_gqa_gate',
      label: 'Sigmoid Output Gate sigma(G * s_gate^2)',
      sublabel: `s_gate^2 in [${sGateScaleRange[0].toFixed(2)}, ${sGateScaleRange[1].toFixed(2)}] · g_0=sqrt(8) (DR3)`,
      worldPos: localToWorldOnFrame(sheetSGateEffSq.frame, 0, 0.75, 0.9),
      tone: 'coda_amber',
      badge: 'DR3',
    }),
    Object.freeze({
      id: 'anchor_gqa_sub_norm',
      label: 'attn_sub_norm (64-dim Concat)',
      sublabel: `||Y_norm||_2 = ${attnSubNormL2Norm.toFixed(4)} across all 8 heads (NOT per-head!)`,
      worldPos: localToWorldOnFrame(sheetAttnSubNorm.frame, 0, 0.75, 0.65),
      tone: 'gold',
      badge: 'CONCAT N_64',
    }),
    Object.freeze({
      id: 'anchor_gqa_nlerp',
      label: 'nGPT NLERP Residual (NOT SLERP)',
      sublabel: `Phys L${physicalLayerIdx} (${layerSpec.transformersPairBadge}) · ${DORY_NLERP_MARKER}`,
      worldPos: localToWorldOnFrame(sheetNlerpOut.frame, 0, 0.75, 0.65),
      tone: 'nlerp_emerald',
      badge: 'DR7',
    }),
  );

  landmarks.push(
    Object.freeze({
      id: 'lm_gqa_projections',
      label: 'Projections + Dual-RoPE -> Q/K L2Norm',
      sublabel: `q_dim=${qDim} (8.0x mini / 1.60x prod) · theta=${metadata.ropeThetaMini}`,
      world: Object.freeze([-6.0, 0.9, -8.8] as const),
      partId: 'gqa_q_rot',
      tone: isSwa ? 'cyan' : 'default',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_gqa_causal_plates',
      label: isSwa ? 'SWA W=8 Banded GQA Plates (4:1)' : 'Full Causal GQA Plates (4:1)',
      sublabel: `Live RG32F G-Channel Mask (DR20) · Q${selectedQHead} -> KV${selectedKvHead}`,
      world: Object.freeze([7.4, 0.95, -bankHalfZ - 0.45] as const),
      partId: activeScoreSheet.partId,
      tone: isSwa ? 'cyan' : 'gold',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_gqa_gate_subnorm',
      label: 'Sigmoid Gate + attn_sub_norm + NLERP',
      sublabel: `64-dim Concat Norm -> o_proj (64->8) -> NLERP`,
      world: Object.freeze([19.8, 0.9, -4.6] as const),
      partId: 'gqa_attn_sub_norm',
      tone: 'gold',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'caliper_gqa_qdim',
      aWorld: localToWorldOnFrame(sheetQPre.frame, -0.5, 0.5, -0.58),
      bWorld: localToWorldOnFrame(sheetQPre.frame, 0.5, 0.5, -0.58),
      label: `q_dim = ${qDim} (8.0x mini · prod 4,096 = 1.60x)`,
      partIds: Object.freeze(['gqa_q_pre', 'gqa_q_scaled']),
    }),
    Object.freeze({
      id: 'caliper_gqa_kvdim',
      aWorld: localToWorldOnFrame(sheetKCache.frame, -0.5, 0.5, 0.58),
      bWorld: localToWorldOnFrame(sheetKCache.frame, 0.5, 0.5, 0.58),
      label: `kv_dim = ${kvDim} (2.0x mini · prod 1,024 = 0.40x)`,
      partIds: Object.freeze(['gqa_k_pre', 'gqa_k_cache']),
    }),
    Object.freeze({
      id: 'caliper_gqa_attn_subnorm',
      aWorld: localToWorldOnFrame(sheetAttnSubNorm.frame, -0.5, 0.5, 0.58),
      bWorld: localToWorldOnFrame(sheetAttnSubNorm.frame, 0.5, 0.5, 0.58),
      label: `attn_sub_norm axis = ${qDim} (all 8 heads concat)`,
      partIds: Object.freeze(['gqa_attn_sub_norm']),
    }),
  );

  const provenanceBadges: DoryProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    Object.freeze({
      label: `Phys L${physicalLayerIdx} · ${layerSpec.transformersPairBadge}`,
      tone: regimeTone,
      discrepancyId: 'DR4',
    }),
    Object.freeze({
      label: isSwa
        ? `SWA W=8 · Dual-RoPE theta_S=16 (prod 10,000, W=1,024)`
        : `Full Attention · Dual-RoPE theta_F=1,024 (prod 10,000,000)`,
      tone: regimeTone,
      discrepancyId: 'DR6',
    }),
    Object.freeze({
      label: `q_dim/d = 8.0x mini vs 1.60x prod · kv_dim/d = 2.0x mini vs 0.40x prod`,
      tone: 'muted',
    }),
    Object.freeze({
      label: `Live RG32F G-Channel Mask (${visibleCellCount} vis / ${maskedCellCount} masked · DR20)`,
      tone: 'nlerp_emerald',
      discrepancyId: 'DR20',
    }),
    Object.freeze({
      label: `Q-only sqk^2 · Sigmoid Gate (DR3) · 64-dim attn_sub_norm`,
      tone: 'gold',
      discrepancyId: 'DR3',
    }),
  ];

  if (!attnTrace.writeKv) {
    provenanceBadges.push(
      Object.freeze({
        label: `last_loop Non-Final Pass (t=${loopIteration} < k=${ctx.kLoops}) · writeKv=false (DoryPastReader)`,
        tone: 'gold',
        discrepancyId: 'DR17',
      }),
    );
  }
  if (hasExtrapAlpha) {
    provenanceBadges.push(
      Object.freeze({
        label: `Unclamped |attn_alpha| > 1 on ${attnTrace.nlerp.extrapolatedChannelCount} ch (DR7)`,
        tone: 'warn',
        discrepancyId: 'DR7',
      }),
    );
  }

  const callouts: DorySceneCallout[] = [
    Object.freeze({
      id: 'callout_gqa_dual_rope_mask',
      title: isSwa
        ? `Sliding-Window GQA (W=8, theta_S=16) & Live RG32F G-Channel Mask`
        : `Full Causal GQA (theta_F=1,024) & Live RG32F G-Channel Mask`,
      body:
        `Phys L${physicalLayerIdx} (${layerSpec.transformersPairBadge}) applies 100% NeoX rotate_half Dual-RoPE ` +
        `(theta=${metadata.ropeThetaMini} mini vs. ${metadata.ropeThetaProd.toLocaleString('en-US')} prod across 4 rotary planes) ` +
        `strictly BEFORE per-head Q/K L2Norm (DR6). Every one of the 16 score/softmax plates carries a live RG32F ` +
        `G-channel mask (DR20: ${visibleCellCount} visible, ${maskedCellCount} masked cells across ${seqLen}x${kvLen}).`,
      tone: regimeTone,
      worldPos: activeScoreSheet.frame.center,
      badge: 'DR6 · DR20',
      discrepancyIds: Object.freeze(['DR6', 'DR20'] as const),
    }),
    Object.freeze({
      id: 'callout_gqa_scaling_disclosure',
      title: 'Attention Projection Expansion Disclosure (Mini 8.0x vs. Prod 1.60x)',
      body:
        `${qExpansionDisclosure}. Dory breaks the classic d = h_q * d_h identity (production d/h_q = 160 != d_h = 256), ` +
        `so q_proj and g_proj expand hidden width while o_proj contracts back to d. Bias-free throughout; ` +
        `${physicalLayerIdx === 0 ? 'Layer 0 has NO pre-norm (DR5), relying on unit-norm embeddings.' : 'V and Gate projections are never L2-normalized.'}`,
      tone: 'prelude_indigo',
      worldPos: sheetQPre.frame.center,
      badge: 'D2 SCALING DISCLOSURE',
      discrepancyIds: Object.freeze(['DR1', 'DR5'] as const),
    }),
    Object.freeze({
      id: 'callout_gqa_sqk_gate_subnorm',
      title: 'Q-Only sqk^2, Sigmoid Gate (s_gate^2) & Full-Concat attn_sub_norm',
      body:
        `Squared query scale sqk^2 in [${sqkScaleRange[0].toFixed(2)}, ${sqkScaleRange[1].toFixed(2)}] multiplies Q ONLY ` +
        `so cached K stays unit-norm. Attention context C = P V is gated by sigma(G * s_gate^2) ` +
        `(s_gate^2 in [${sGateScaleRange[0].toFixed(2)}, ${sGateScaleRange[1].toFixed(2)}], g_0=sqrt(8), DR3), then ` +
        `attn_sub_norm L2-normalizes across ALL q_dim=${qDim} concatenated channels (||Y_norm||_2 = ${attnSubNormL2Norm.toFixed(4)}) ` +
        `before o_proj and channel-wise NLERP (${DORY_ANTI_SLERP_PLAQUE}).`,
      tone: 'gold',
      worldPos: sheetAttnSubNorm.frame.center,
      badge: 'DR3 · DR7 · N_64',
      discrepancyIds: Object.freeze(['DR3', 'DR7', 'DR9'] as const),
    }),
    Object.freeze({
      id: 'callout_gqa_kv_provenance',
      title: `KV Bank #${attnTrace.bankSlotId} (${attnTrace.bankKind}) Provenance`,
      body:
        `Transient KV workspace spans ${attnTrace.pastLen} historical + ${seqLen} current-chunk positions (kvLen=${kvLen}, DR24). ` +
        `Policy=${ctx.kvCacheMode}, writeKv=${String(attnTrace.writeKv)}. ` +
        (attnTrace.writeKv
          ? `Chunk commits post-RoPE post-L2Norm K and raw V after attention completes.`
          : `Non-final last_loop pass reads historical final-loop KVs without advancing ring pointers (DR17).`),
      tone: attnTrace.writeKv ? 'recurrent_teal' : 'gold',
      worldPos: activeProbSheet.frame.center,
      badge: attnTrace.writeKv ? `BANK #${attnTrace.bankSlotId}` : 'READ-ONLY PASS',
      discrepancyIds: Object.freeze(['DR17', 'DR24'] as const),
    }),
  ];

  // Active cell popover
  const activeCellPopover = buildGqaHoveredPopover(
    options.hoveredCell,
    sheetsByPartId,
    attnTrace,
    layerWeights,
    metadata,
    tokenLabels,
    kvColLabels,
  );

  const stageBoundaryAabb = computeBoundsFromPoints(
    [
      [-17.0, -0.1, -9.2],
      [26.4, 1.3, 7.8],
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
    viewMode: 'gqa_rope',
    title: VIEW_DESCRIPTOR.label,
    subtitle:
      `Phys L${physicalLayerIdx} (${layerSpec.transformersPairBadge}) · ${loopText} · ` +
      `${isSwa ? `SWA W=8, theta_S=16` : `Full, theta_F=1024`} · Q${selectedQHead}/KV${selectedKvHead} · ` +
      `q_dim=64 (8.0x mini / 1.60x prod)`,
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
    kvDepthTicks: Object.freeze(kvDepthTicks),
    calipers: Object.freeze(calipers),
    activeCellGrids: Object.freeze(activeCellGrids),
    activeCellPopover,
    gqaRopeMetadata: metadata,
  });

  // Self-audit finiteness, atlas budget/non-overlap, and DR20 live G-channel mask before returning
  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `View 3 gqaRopeScene validation failed: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
