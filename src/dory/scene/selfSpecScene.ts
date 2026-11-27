/**
 * View 6 (`self_spec`): Contract A Paper §6.2 Self-Speculative Execution Inspector
 * (`src/dory/scene/selfSpecScene.ts`)
 *
 * Implements DORY4 Slice 4B (Locked Decisions `D1`, `D2`, `D3`, `D4`, `D5`, `D7`;
 * Guards `DR19`, `DR20`, `DR21`, `DR23`):
 *
 *   - **Explicit Epistemic & Reconstruction Disclosure (`DR23`):**
 *     Badged everywhere with the exact locked plaque:
 *       `'Reconstructed from Paper §6.2 (Contract A) · Not in Upstream Runtimes · per_loop Only'`
 *     Paper §6.2 is absent from upstream `transformers`, `vLLM`, and `llama.cpp`, and omits
 *     Output/Coda KV history handling across cycles; Contract A allocates `8` dedicated
 *     persistent `output_shallow` Coda KV banks alongside the `96` deep `per_loop` banks.
 *
 *   - **Exact 104-Bank Contract A Architecture Rail (`8 + 20 + 60 + 8 + 8 = 104`):**
 *       * `0..7` (`8` banks): Shared **Input Prelude** (`6` SWA + `2` Full) — **REUSED** across draft & verify!
 *       * `8..27` (`20` banks): Shared **Recurrent Loop 1** (`15` SWA + `5` Full) — **REUSED** `(x, s_1)` & KV!
 *       * `28..87` (`60` banks): Verifier-only **Recurrent Loops 2–4** (`45` SWA + `15` Full).
 *       * `88..95` (`8` banks): Verifier-only persistent **`output_deep` Coda KV** (`6` SWA + `2` Full).
 *       * `96..103` (`8` banks): Drafter-only persistent **`output_shallow` Coda KV** (`+20,480 B` = `+20.0 KiB`,
 *         total `266,240 B` = `260.0 KiB` payload across all `104` banks), showing `0` extra Shallow Coda work
 *         on rejection (`a < gamma'`) vs. `1`-token catch-up on `s_1^{(L + gamma')}` upon full acceptance!
 *
 *   - **Division-Free Binary64 Rejection Sampling Bars & Residual Correction (`Audit §8.2`, Finding 3):**
 *     Visualizes per-position shallow proposal `q[d]` vs. deep target `p[d]` bars under
 *     `p[d] >= q[d] || r * q[d] < p[d]` (zero `1e-38` denominator clamping, zero `1e-12` fallback),
 *     accepted/rejected/bonus/rolled-back status markers, and exact top residual correction tokens
 *     `p_res[v] = max(0, p[v] - q[v]) / Z` whenever rejection occurs.
 *
 *   - **Transaction-Local `cap_swa = 16` Rollback Markers (`DR19`):**
 *     Visualizes all `16` physical SWA ring slots (`pos & 15`) distinguishing `committed_past`,
 *     `committed_new`, `rolled_back` (`absPosTag` atomically invalidated to `-1`), `evicted_from_window`,
 *     and `empty` slots — proving zero historical window overwrite for `gamma <= 4` (`W + gamma + 1 = 13 <= 16`).
 *
 *   - **Live `RG32F` Causal Attention Plates (`DR20`):**
 *     Every SWA/Full verify score (`sheetKind = 1`) and softmax (`sheetKind = 2`) plate binds a live
 *     `RG32F` `G`-channel mask inside a compact `64 x 64` (`32.00 KiB <= 256 KiB <= 512 KiB`, `0` overlaps) atlas.
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
  RIBBON_COLOR_EMERALD,
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
  SHEET_HL_OPERAND_A,
  SHEET_HL_OPERAND_B,
  SHEET_HL_PATHWAY,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_KV_K,
  SHEET_ROLE_SCORES,
  SHEET_ROLE_SOFTMAX,
} from '../../render/sheets';
import { fmtSigned, formatKiB } from '../../shared/format';
import {
  computeSheetCellWorldPos,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
} from '../../shared/overlayLayout';
import { getDoryInputBank, getDoryOutputBank, getDoryRecurrentBank } from '../kvCache';
import { createDoryPrng, deriveDorySeed } from '../prng';
import {
  DORY_PHYSICAL_LAYERS,
  computeExecutedLayerIdx,
  computeMiniKvCacheBytes,
  decodeDoryToken,
  getDoryPhysicalLayerSpec,
} from '../spec';
import {
  normalizeSamplingConfig,
  prefillDorySelfSpec,
  runDorySelfSpecCycle,
  summarizeSelfSpecCycle,
} from '../speculative';
import {
  DORY_DEFAULT_SESSION_PROMPT,
  DORY_DEFAULT_SESSION_SEED,
  DORY_EOS_TOKEN_IDS,
  DORY_MINI_CONFIG,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  DORY_SPEC_VERIFY_LOOPS,
  type DoryAttentionSublayerTrace,
  type DoryCacheRole,
  type DoryForwardTrace,
  type DoryKvBank,
  type DoryKvCache,
  type DoryMiniConfig,
  type DorySelfSpecCycleResult,
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
  type DoryOverlayAnchor,
  type DoryPickIdentity,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DorySceneCallout,
  type DorySelfSpecSceneOptions,
  type DorySelfSpecViewMetadata,
  type DorySpatialScene,
  type DorySpecPositionBarDescriptor,
  type DorySpecResidualTopToken,
  type DorySpecSwaRingSlotState,
  type DorySpecSwaRingSlotStatus,
} from './types';

// ============================================================================
// VIEW 6 CONSTANTS & CONTRACT A INVARIANTS
// ============================================================================

export const DORY_SELF_SPEC_ATLAS_WIDTH = 64 as const;
export const DORY_SELF_SPEC_ATLAS_HEIGHT = 64 as const;
export const DORY_SELF_SPEC_ATLAS_BUDGET_BYTES = 262_144 as const; // 256.00 KiB <= 512 KiB
export const DORY_DIVISION_FREE_BINARY64_RULE = 'p[d] >= q[d] || r * q[d] < p[d]' as const;

const VIEW_DESCRIPTOR = DORY_SPATIAL_VIEW_MODES[5]!; // 'self_spec'
const TOP_PROB_DISPLAY_COUNT = 8;

// ============================================================================
// PURE HELPERS: SWA RING STATUS, CYCLE EXTRACTION & PREVIEW SYNTHESIS
// ============================================================================

function clampInt(val: number, lo: number, hi: number): number {
  if (!Number.isFinite(val)) return lo;
  return Math.max(lo, Math.min(hi, Math.trunc(val)));
}

function isEosToken(tok: number): boolean {
  for (let i = 0; i < DORY_EOS_TOKEN_IDS.length; i += 1) {
    if (DORY_EOS_TOKEN_IDS[i] === tok) return true;
  }
  return false;
}

function findBestNonEosToken(probs: Float64Array): number {
  let bestIdx = 6; // fallback non-EOS token 'the'
  let bestProb = -1;
  for (let v = 0; v < probs.length; v += 1) {
    if (isEosToken(v)) continue;
    const p = probs[v]!;
    if (p > bestProb) {
      bestProb = p;
      bestIdx = v;
    }
  }
  return bestIdx;
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

/**
 * Classifies all `cap_swa` (`16`) physical SWA ring slots (`pos & 15`) after one Contract A
 * speculative cycle, explicitly separating `committed_past`, `committed_new`, `rolled_back`
 * (invalidated to `absPosTag = -1` by `rewindDoryKvCache`), `evicted_from_window`, and `empty`.
 */
export function computeSpecSwaRingSlotStates(
  swaBank: DoryKvBank,
  startCommittedLen: number,
  endCommittedLen: number,
  maxStagedLen: number,
  slidingWindow: number = DORY_MINI_CONFIG.slidingWindow,
): readonly DorySpecSwaRingSlotState[] {
  const cap = swaBank.capacity;
  const mask = cap - 1;
  const winStart = Math.max(0, endCommittedLen - slidingWindow);

  // Map rolled-back positions [endCommittedLen .. maxStagedLen - 1] to their physical ring slots
  const rolledBackBySlot = new Map<number, number>();
  for (let rp = endCommittedLen; rp < maxStagedLen; rp += 1) {
    rolledBackBySlot.set(rp & mask, rp);
  }

  const out: DorySpecSwaRingSlotState[] = [];
  for (let slotIndex = 0; slotIndex < cap; slotIndex += 1) {
    const tagPos = swaBank.absPosTag[slotIndex]!;
    const rolledPos = rolledBackBySlot.get(slotIndex);

    let absPos: number | null;
    let status: DorySpecSwaRingSlotStatus;

    if (tagPos < 0 && rolledPos !== undefined) {
      absPos = rolledPos;
      status = 'rolled_back';
    } else if (tagPos >= 0) {
      absPos = tagPos;
      if (tagPos < winStart) {
        status = 'evicted_from_window';
      } else if (tagPos >= startCommittedLen) {
        status = 'committed_new';
      } else {
        status = 'committed_past';
      }
    } else {
      absPos = null;
      status = 'empty';
    }

    out.push(Object.freeze({ slotIndex, absPos, status }));
  }

  return Object.freeze(out);
}

function statusToRingCode(status: DorySpecSwaRingSlotStatus): number {
  switch (status) {
    case 'empty':
      return 0;
    case 'evicted_from_window':
      return 1;
    case 'committed_past':
      return 2;
    case 'committed_new':
      return 3;
    case 'rolled_back':
      return 4;
  }
}

interface ResolvedSpecCycleContext {
  readonly cycleResult: DorySelfSpecCycleResult;
  readonly postStepKvCache: DoryKvCache;
  readonly isSynthesizedPreview: boolean;
}

/**
 * Resolves either the active live `spec_cycle` record (via 0-ULP `resolveStepTrace`) OR
 * synthesizes an isolated, deterministic, read-only Contract A preview cycle without ever
 * mutating the caller's `DorySession`.
 */
function resolveOrSynthesizeSpecCycle(
  ctx: DoryResolvedSessionContext,
  options: DorySelfSpecSceneOptions,
): ResolvedSpecCycleContext {
  const { params, ropeTables, record, branch } = ctx;
  const config = params.config;

  if (record.kind === 'spec_cycle') {
    const reconstructed = ctx.resolveStepTrace();
    if (reconstructed.specCycleResult !== null) {
      return {
        cycleResult: reconstructed.specCycleResult,
        postStepKvCache: reconstructed.postStepKvCache,
        isSynthesizedPreview: false,
      };
    }
  }

  // Synthesize an isolated read-only preview cycle (NEVER mutates caller session!)
  const sourceSeq =
    record.committedSequence.length > 0
      ? record.committedSequence
      : branch.promptTokens.length > 0
        ? branch.promptTokens
        : DORY_DEFAULT_SESSION_PROMPT;

  // Leave enough room for a full gamma=4 + bonus cycle (<= maxPositionEmbeddings - 5)
  const maxPreviewPromptLen = Math.max(1, config.maxPositionEmbeddings - config.speculativeGamma - 1);
  const previewPromptLen = Math.min(sourceSeq.length, maxPreviewPromptLen);
  const previewPrompt: number[] = new Array<number>(previewPromptLen);
  for (let i = 0; i < previewPromptLen; i += 1) {
    previewPrompt[i] = sourceSeq[i]!;
  }

  const sampling = normalizeSamplingConfig(options.previewSampling ?? branch.config.sampling);
  const previewPrng = createDoryPrng(
    deriveDorySeed(branch.config.seed ^ DORY_DEFAULT_SESSION_SEED, record.stepIndex + 0x51ec),
  );

  const prefill = prefillDorySelfSpec({
    promptTokens: previewPrompt,
    params,
    sampling,
    prng: previewPrng,
    ropeTables,
    sessionBranchId: 'spec_preview',
    forwardStepId: 0,
    captureTrace: false,
  });

  const safePendingToken =
    prefill.haltReason !== null || isEosToken(prefill.pendingToken)
      ? findBestNonEosToken(prefill.lastDeepProbs)
      : prefill.pendingToken;

  const cycleResult = runDorySelfSpecCycle({
    kvCache: prefill.kvCache,
    pendingToken: safePendingToken,
    params,
    gamma: config.speculativeGamma,
    sampling,
    prng: previewPrng,
    ropeTables,
    sessionBranchId: 'spec_preview',
    forwardStepId: 1,
    captureTrace: true,
  });

  return {
    cycleResult,
    postStepKvCache: prefill.kvCache,
    isSynthesizedPreview: true,
  };
}

function extractVerifyAttentionTrace(
  verifyTrace: DoryForwardTrace,
  physicalLayerIdx: number,
  config: DoryMiniConfig,
): DoryAttentionSublayerTrace {
  const spec = getDoryPhysicalLayerSpec(physicalLayerIdx);
  const loopIter = spec.stage === 'recurrent' ? DORY_SPEC_VERIFY_LOOPS : 0;
  const execIdx = computeExecutedLayerIdx(physicalLayerIdx, loopIter, DORY_SPEC_VERIFY_LOOPS);
  const direct = verifyTrace.executedSublayers[execIdx];
  if (
    direct !== undefined &&
    direct.coord.physicalLayerIdx === physicalLayerIdx &&
    direct.coord.loopIteration === loopIter &&
    direct.trace.kind === 'attention'
  ) {
    return direct.trace;
  }
  const found = verifyTrace.executedSublayers.find(
    (rec) =>
      rec.coord.physicalLayerIdx === physicalLayerIdx &&
      rec.coord.loopIteration === loopIter &&
      rec.trace.kind === 'attention',
  );
  if (found === undefined || found.trace.kind !== 'attention') {
    throw new Error(
      `Missing verify attention trace for physicalLayerIdx=${physicalLayerIdx}, loopIteration=${loopIter} (config.numHiddenLayers=${config.numHiddenLayers}).`,
    );
  }
  return found.trace;
}

function resolveContractABankForLayer(
  kvCache: DoryKvCache,
  physicalLayerIdx: number,
): DoryKvBank {
  const spec = getDoryPhysicalLayerSpec(physicalLayerIdx);
  if (spec.stage === 'input') {
    return getDoryInputBank(kvCache, physicalLayerIdx);
  }
  if (spec.stage === 'recurrent') {
    return getDoryRecurrentBank(kvCache, physicalLayerIdx, 1);
  }
  return getDoryOutputBank(kvCache, physicalLayerIdx, 'output_deep');
}

function resolveBankCacheRole(bank: DoryKvBank): DoryCacheRole {
  if (bank.bankKind === 'input' || (bank.bankKind === 'recurrent' && bank.loopSlot === 1)) {
    return 'spec_verify_reused';
  }
  if (bank.bankKind === 'output_shallow') {
    return 'spec_draft';
  }
  return 'spec_verify_deep';
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

// ============================================================================
// HOVERED CELL POPOVER BUILDER
// ============================================================================

function buildSelfSpecPopover(
  hovered: DoryPickIdentity | null | undefined,
  sheetsByPartId: ReadonlyMap<string, PlacedSheetRecord>,
  verifyAttn: DoryAttentionSublayerTrace,
  positionBars: readonly DorySpecPositionBarDescriptor[],
  residualTopTokens: readonly DorySpecResidualTopToken[],
  swaRingSlots: readonly DorySpecSwaRingSlotState[],
  topCompTokenIds: readonly number[],
  cycleResult: DorySelfSpecCycleResult,
  selectedVerifyOffset: number,
): CellPopoverSpec | null {
  if (!hovered || !hovered.partId) return null;

  // Hovering a 3D position acceptance bar
  const posBarMatch = /^spec_pos_bar_(\d+)_(q|p)$/.exec(hovered.partId);
  if (posBarMatch !== null) {
    const offset = clampInt(Number.parseInt(posBarMatch[1]!, 10), 0, positionBars.length - 1);
    const bar = positionBars[offset]!;
    const isQ = posBarMatch[2] === 'q';
    const qpSheet = sheetsByPartId.get('spec_qp_bars');
    const world = qpSheet
      ? computeSheetCellWorldPos(qpSheet.grid, offset, isQ ? 0 : 1, 0.5)
      : Object.freeze([0, 0.7, 0] as const);
    return Object.freeze({
      world: Object.freeze([world[0], world[1] + 0.65, world[2]] as const),
      tensorLabel: isQ
        ? `Shallow Draft q[d_${offset + 1}] (k=1 · output_shallow)`
        : `Deep Verify p[d_${offset + 1}] (k=4 · output_deep)`,
      coordLabel: `offset=${offset} · absPos=${bar.absPos} · in="${bar.inputTokenText}" -> prop="${bar.proposedTokenText ?? 'bonus'}"`,
      badge: bar.rolledBack
        ? 'ROLLED BACK (DR19)'
        : bar.isBonusSlot
          ? 'BONUS SLOT (FRESH PRELUDE+LOOP1)'
          : bar.accepted === true
            ? `ACCEPTED (${bar.rejectionReason ?? 'accept'})`
            : bar.accepted === false
              ? `REJECTED (${bar.rejectionReason ?? 'reject'})`
              : 'DEEP DECODE SLOT',
      equation:
        bar.proposalProb !== null && bar.targetProb !== null
          ? `${DORY_DIVISION_FREE_BINARY64_RULE} · q=${bar.proposalProb.toFixed(4)}, p=${bar.targetProb.toFixed(4)}${bar.uniformVariate !== null ? `, r=${bar.uniformVariate.toFixed(4)}` : ''}`
          : 'Batched k=4 target distribution (no draft proposal on bonus/final slot)',
      valueText: isQ
        ? `q = ${(bar.proposalProb ?? 0).toFixed(4)}`
        : `p = ${(bar.targetProb ?? 0).toFixed(4)}`,
    });
  }

  // Hovering a 3D SWA ring rollback slot
  const ringMatch = /^spec_swa_slot_(\d+)$/.exec(hovered.partId);
  if (ringMatch !== null) {
    const s = clampInt(Number.parseInt(ringMatch[1]!, 10), 0, swaRingSlots.length - 1);
    const slot = swaRingSlots[s]!;
    const strip = sheetsByPartId.get('spec_swa_ring_strip');
    const world = strip
      ? computeSheetCellWorldPos(strip.grid, s, 0, 0.5)
      : Object.freeze([0, 0.5, 0] as const);
    return Object.freeze({
      world: Object.freeze([world[0], world[1] + 0.55, world[2]] as const),
      tensorLabel: `Contract A SWA Ring Slot #${s} (pos & 15)`,
      coordLabel:
        slot.absPos !== null ? `absPos=${slot.absPos} · status=${slot.status}` : 'empty (-1)',
      badge:
        slot.status === 'rolled_back'
          ? 'ATOMICALLY REWOUND (absPosTag=-1, DR19)'
          : slot.status === 'committed_new'
            ? 'NEWLY COMMITTED IN CYCLE'
            : slot.status === 'committed_past'
              ? 'PRESERVED PAST HISTORY (ZERO OVERWRITE)'
              : slot.status === 'evicted_from_window'
                ? 'RETAINED OUTSIDE W=8 WINDOW'
                : 'EMPTY SLOT',
      equation: `cap_swa=16 >= W(8)+gamma(4)+1=13 · committed [${cycleResult.startCommittedLen} -> ${cycleResult.endCommittedLen}]`,
      valueText: slot.absPos !== null ? `pos = ${slot.absPos}` : 'empty',
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

  if (rec.partId === 'spec_verify_attn_probs' || rec.partId === 'spec_verify_attn_scores') {
    const isScores = rec.partId === 'spec_verify_attn_scores';
    const qPos = verifyAttn.startPos + row;
    const kPos = verifyAttn.kvAbsPositions[col] ?? col;
    const reused = row < cycleResult.draftCount;
    return Object.freeze({
      world,
      tensorLabel: `Batched k=4 Verify ${isScores ? 'Causal Scores' : 'Causal Softmax'} (h0)`,
      coordLabel: `q_pos=${qPos} (${reused ? 'reused x,s_1' : 'fresh bonus'}), k_pos=${kPos}`,
      badge: isMasked ? 'DR20 RG32F G=1 MASKED' : 'DR20 RG32F G=0 VISIBLE · per_loop',
      equation: isMasked
        ? `Masked out (${verifyAttn.attentionKind === 'sliding_attention' ? 'SWA W=8' : 'Full causal'})`
        : isScores
          ? `d_h^-0.5 <sqk^2 * N(RoPE(Q[${qPos}])), N(RoPE(K[${kPos}]))>`
          : `Batched verify attention weight at (q=${qPos}, k=${kPos})`,
      valueText: isMasked ? '= -inf' : isScores ? `= ${fmtSigned(val, 4)}` : `= ${val.toFixed(4)}`,
    });
  }

  if (rec.partId === 'spec_prob_comparison') {
    const tokId = topCompTokenIds[col] ?? 0;
    const tokText = decodeDoryToken(tokId);
    const isShallowRow = row === 0;
    return Object.freeze({
      world,
      tensorLabel: isShallowRow
        ? `Shallow Draft q (k=1 · output_shallow)`
        : `Deep Target p (k=4 · output_deep)`,
      coordLabel: `verify offset ${selectedVerifyOffset} · tok #${tokId} "${tokText}"`,
      badge: DORY_DIVISION_FREE_BINARY64_RULE,
      equation: `Compare standalone k=1 proposal q[${tokId}] vs k=4 verifier target p[${tokId}]`,
      valueText: `prob = ${val.toFixed(4)}`,
    });
  }

  if (rec.partId === 'spec_residual_dist' && col < residualTopTokens.length) {
    const rt = residualTopTokens[col]!;
    const rowLabel = row === 0 ? 'Deep Target p' : row === 1 ? 'Shallow Proposal q' : 'Residual p_res = max(0,p-q)/Z';
    return Object.freeze({
      world,
      tensorLabel: `${rowLabel} — "${rt.tokenText}" (#${rt.tokenId})`,
      coordLabel: `Z = ${(cycleResult.residualDistribution?.massZ ?? 0).toFixed(5)} (binary64, zero 1e-12 fallback)`,
      badge: 'EXACT BINARY64 RESIDUAL CORRECTION',
      equation: `max(0, ${rt.targetProb.toFixed(4)} - ${rt.proposalProb.toFixed(4)}) / Z = ${rt.residualProb.toFixed(4)}`,
      valueText: `= ${val.toFixed(4)}`,
    });
  }

  return Object.freeze({
    world,
    tensorLabel: rec.label,
    coordLabel: `r=${row}, c=${col}`,
    equation: `${rec.tensorKey}[${row}, ${col}]`,
    valueText: `= ${val.toFixed(4)}`,
  });
}

// ============================================================================
// MAIN VIEW 6 SCENE BUILDER (`buildDorySelfSpecScene`)
// ============================================================================

/**
 * Builds the headless 5-pass WebGL2 spatial scene for **View 6 (`self_spec`)**:
 * Contract A Paper §6.2 Self-Speculative Execution Inspector.
 *
 * Strictly non-mutating on `sessionOrCtx`: inspects live `spec_cycle` records via 0-ULP
 * lazy reconstruction or synthesizes an isolated read-only Contract A preview cycle when
 * invoked on a standard or `spec_prefill` record.
 */
export function buildDorySelfSpecScene(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  options: DorySelfSpecSceneOptions = {},
): DorySpatialScene {
  const ctx = ensureDorySessionContext(sessionOrCtx, options.recordId);
  const { params } = ctx;
  const config = params.config;

  // 1. Resolve live spec_cycle or synthesize isolated read-only preview cycle
  const { cycleResult, postStepKvCache, isSynthesizedPreview } = resolveOrSynthesizeSpecCycle(
    ctx,
    options,
  );
  const cycleSummary = summarizeSelfSpecCycle(cycleResult);

  if (cycleResult.verifyTrace === null) {
    throw new Error('Self-speculative cycle result is missing verifyTrace required for View 6.');
  }

  // Resolve focused verify offset & even attention layer (`0..70`, default `16` SWA)
  const selectedVerifyOffset = clampInt(
    options.selectedVerifyOffset ??
      (cycleResult.acceptedDraftCount < cycleResult.draftCount
        ? cycleResult.acceptedDraftCount
        : Math.max(0, cycleResult.verifiedPositions - 1)),
    0,
    Math.max(0, cycleResult.verifiedPositions - 1),
  );
  const rawAttnLayer = options.selectedAttentionLayerIdx ?? 16;
  const clampedAttnLayer = clampInt(rawAttnLayer, 0, config.numHiddenLayers - 1);
  const selectedAttentionLayerIdx =
    clampedAttnLayer % 2 === 0 ? clampedAttnLayer : Math.max(0, clampedAttnLayer - 1);
  const attnSpec = getDoryPhysicalLayerSpec(selectedAttentionLayerIdx);

  const verifyAttn = extractVerifyAttentionTrace(
    cycleResult.verifyTrace,
    selectedAttentionLayerIdx,
    config,
  );

  // 2. Contract A 104-bank payload byte accounting (`245,760 + 20,480 = 266,240 B = 260 KiB`)
  const specMem = computeMiniKvCacheBytes(DORY_SPEC_VERIFY_LOOPS, 'per_loop', true, config);
  const deepOnlyMem = computeMiniKvCacheBytes(DORY_SPEC_VERIFY_LOOPS, 'per_loop', false, config);
  const shallowCodaExtraBytes = specMem.kvPayloadBytes - deepOnlyMem.kvPayloadBytes; // 20,480 B
  const totalKvPayloadBytes = specMem.kvPayloadBytes; // 266,240 B

  // 3. Build per-position rejection sampling bar descriptors
  const positionBars: DorySpecPositionBarDescriptor[] = [];
  for (let j = 0; j < cycleResult.verifyPositions.length; j += 1) {
    const vp = cycleResult.verifyPositions[j]!;
    const draft = j < cycleResult.draftSteps.length ? cycleResult.draftSteps[j]! : null;
    const isBonusSlot = draft === null && cycleResult.draftCount > 0;
    const proposedTokenId = draft !== null ? draft.proposedTokenId : null;
    const proposedTokenText = proposedTokenId !== null ? decodeDoryToken(proposedTokenId) : null;
    const proposalProb = draft !== null ? draft.proposalProbs[draft.proposedTokenId]! : null;
    const targetProb = draft !== null ? vp.targetProbs[draft.proposedTokenId]! : null;
    const rolledBack = vp.absPos >= cycleResult.endCommittedLen;

    positionBars.push(
      Object.freeze({
        verifyOffset: vp.verifyOffset,
        absPos: vp.absPos,
        inputTokenId: vp.inputTokenId,
        inputTokenText: decodeDoryToken(vp.inputTokenId),
        proposedTokenId,
        proposedTokenText,
        isBonusSlot,
        reusedPreludeAndLoop1: vp.reusedPreludeAndLoop1,
        proposalProb,
        targetProb,
        uniformVariate: vp.rejection !== null ? vp.rejection.uniformVariate : null,
        accepted: vp.rejection !== null ? vp.rejection.accepted : null,
        rejectionReason: vp.rejection !== null ? vp.rejection.reason : null,
        rolledBack,
      }),
    );
  }

  // 4. Build exact binary64 residual correction top-token list (non-empty on stochastic rejection)
  const residualTopTokens: DorySpecResidualTopToken[] = [];
  if (
    cycleResult.residualDistribution !== null &&
    cycleResult.acceptedDraftCount < cycleResult.draftSteps.length &&
    cycleResult.acceptedDraftCount < cycleResult.verifyPositions.length
  ) {
    const rejIdx = cycleResult.acceptedDraftCount;
    const pVec = cycleResult.verifyPositions[rejIdx]!.targetProbs;
    const qVec = cycleResult.draftSteps[rejIdx]!.proposalProbs;
    const resVec = cycleResult.residualDistribution.probs;
    const positiveIndices: number[] = [];
    for (let v = 0; v < resVec.length; v += 1) {
      if (resVec[v]! > 0) {
        positiveIndices.push(v);
      }
    }
    positiveIndices.sort((a, b) => {
      const diff = resVec[b]! - resVec[a]!;
      if (diff !== 0) return diff;
      return a - b;
    });
    const topN = Math.min(TOP_PROB_DISPLAY_COUNT, positiveIndices.length);
    for (let i = 0; i < topN; i += 1) {
      const tokId = positiveIndices[i]!;
      residualTopTokens.push(
        Object.freeze({
          tokenId: tokId,
          tokenText: decodeDoryToken(tokId),
          residualProb: resVec[tokId]!,
          targetProb: pVec[tokId]!,
          proposalProb: qVec[tokId]!,
        }),
      );
    }
  }

  // 5. Inspect transaction-local `cap_swa = 16` SWA ring rollback state (`DR19`)
  const swaRingLayerIdx =
    attnSpec.attentionKind === 'sliding_attention' ? selectedAttentionLayerIdx : 16;
  const swaRingBank = resolveContractABankForLayer(postStepKvCache, swaRingLayerIdx);
  const maxStagedLen = cycleResult.startCommittedLen + cycleResult.verifiedPositions;
  const swaRingSlots = computeSpecSwaRingSlotStates(
    swaRingBank,
    cycleResult.startCommittedLen,
    cycleResult.endCommittedLen,
    maxStagedLen,
    config.slidingWindow,
  );

  // 6. Pack compact 64 x 64 Live RG32F Atlas (32.00 KiB <= 256 KiB <= 512 KiB, 0 overlaps)
  const atlasPacker = new DoryAtlasPacker(
    DORY_SELF_SPEC_ATLAS_WIDTH,
    DORY_SELF_SPEC_ATLAS_HEIGHT,
    DORY_SELF_SPEC_ATLAS_BUDGET_BYTES,
  );

  const verifyRows = verifyAttn.seqLen;
  const verifyCols = verifyAttn.kvLen;
  const verifyProbsH0 = sliceHeadMatrix(verifyAttn.probs, 0, verifyRows, verifyCols);
  const verifyScoresH0 = sliceHeadMatrix(verifyAttn.scores, 0, verifyRows, verifyCols);
  const verifyMaskedFlags = invertMaskVisibleToMasked(verifyAttn.maskVisible);

  // DR20: Live RG32F G-channel masks on BOTH causal softmax (kind=2) and causal scores (kind=1)
  const verifyProbsSlice = atlasPacker.packCausalMatrix(
    'spec_verify_attn_probs',
    verifyProbsH0,
    verifyAttn.maskVisible,
    verifyRows,
    verifyCols,
    1,
    {
      fixedScale: 1.0,
      physicalLayerIdx: selectedAttentionLayerIdx,
      loopIteration: attnSpec.stage === 'recurrent' ? DORY_SPEC_VERIFY_LOOPS : 0,
    },
  );
  const verifyScoresSlice = atlasPacker.packCausalMatrix(
    'spec_verify_attn_scores',
    verifyScoresH0,
    verifyAttn.maskVisible,
    verifyRows,
    verifyCols,
    0,
    {
      minFloor: 0.5,
      physicalLayerIdx: selectedAttentionLayerIdx,
      loopIteration: attnSpec.stage === 'recurrent' ? DORY_SPEC_VERIFY_LOOPS : 0,
    },
  );

  // Per-position [2 x verifiedPositions] shallow q vs deep p probability bar sheet
  const verifiedCount = Math.max(1, positionBars.length);
  const qpBarsArr = new Float32Array(2 * verifiedCount);
  for (let j = 0; j < verifiedCount; j += 1) {
    const bar = positionBars[j]!;
    qpBarsArr[j] = bar.proposalProb ?? 0;
    qpBarsArr[verifiedCount + j] = bar.targetProb ?? 0;
  }
  const qpBarsSlice = atlasPacker.packMatrix('spec_qp_bars', qpBarsArr, 2, verifiedCount, {
    mode: 1,
    fixedScale: 1.0,
  });

  // Focused position top-8 vocab comparison [2 x 8] (Row 0 = shallow q, Row 1 = deep p)
  const focusVp = cycleResult.verifyPositions[selectedVerifyOffset]!;
  const focusDraft =
    selectedVerifyOffset < cycleResult.draftSteps.length
      ? cycleResult.draftSteps[selectedVerifyOffset]!
      : null;
  const focusP = focusVp.targetProbs;
  const focusQ = focusDraft !== null ? focusDraft.proposalProbs : focusP;

  const vocabOrder = Array.from({ length: config.vocabSize }, (_, v) => v);
  vocabOrder.sort((a, b) => {
    const scoreA = Math.max(focusP[a]!, focusQ[a]!);
    const scoreB = Math.max(focusP[b]!, focusQ[b]!);
    if (scoreB !== scoreA) return scoreB - scoreA;
    return a - b;
  });
  const topCompTokenIds = Object.freeze(vocabOrder.slice(0, TOP_PROB_DISPLAY_COUNT));
  const probCompArr = new Float32Array(2 * topCompTokenIds.length);
  for (let i = 0; i < topCompTokenIds.length; i += 1) {
    const tokId = topCompTokenIds[i]!;
    probCompArr[i] = focusQ[tokId]!;
    probCompArr[topCompTokenIds.length + i] = focusP[tokId]!;
  }
  const probCompSlice = atlasPacker.packMatrix(
    'spec_prob_comparison',
    probCompArr,
    2,
    topCompTokenIds.length,
    {
      mode: 1,
      fixedScale: 1.0,
    },
  );

  // Optional [3 x N] binary64 residual distribution sheet when rejection occurred
  let residualDistArr: Float32Array | null = null;
  let residualDistSlice = null;
  if (residualTopTokens.length > 0) {
    const cols = residualTopTokens.length;
    residualDistArr = new Float32Array(3 * cols);
    for (let i = 0; i < cols; i += 1) {
      const rt = residualTopTokens[i]!;
      residualDistArr[i] = rt.targetProb;
      residualDistArr[cols + i] = rt.proposalProb;
      residualDistArr[2 * cols + i] = rt.residualProb;
    }
    residualDistSlice = atlasPacker.packMatrix('spec_residual_dist', residualDistArr, 3, cols, {
      mode: 1,
      fixedScale: 1.0,
    });
  }

  // [2 x 16] SWA ring rollback strip (Row 0 = absPos or -1, Row 1 = status code 0..4)
  const swaRingStripArr = new Float32Array(2 * config.swaRingCapacity);
  for (let s = 0; s < config.swaRingCapacity; s += 1) {
    const slot = swaRingSlots[s]!;
    swaRingStripArr[s] = slot.absPos ?? -1;
    swaRingStripArr[config.swaRingCapacity + s] = statusToRingCode(slot.status);
  }
  const swaRingStripSlice = atlasPacker.packMatrix(
    'spec_swa_ring_strip',
    swaRingStripArr,
    2,
    config.swaRingCapacity,
    {
      mode: 0,
      minFloor: Math.max(4, maxStagedLen),
      physicalLayerIdx: swaRingLayerIdx,
      loopIteration: 1,
    },
  );

  const atlasExtraTag = `b=${ctx.branch.branchId}:r=${ctx.record.recordId}:prev=${isSynthesizedPreview ? 1 : 0}:p=${selectedAttentionLayerIdx}:v=${selectedVerifyOffset}:out=${cycleResult.outcome}`;
  const packedAtlas = atlasPacker.build('self_spec', atlasExtraTag);

  // 7. Build 5-Pass 3D Editorial Drafting-Table Scene Geometry
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
  const sheetH = 0.068;

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
    readonly slice: typeof verifyProbsSlice;
    readonly values: Float32Array;
    readonly masked?: Uint8Array;
    readonly format: 'signed' | 'prob';
    readonly colLabels: readonly string[];
    readonly rowLabels: readonly string[];
    readonly colDimLabel: string;
    readonly rowDimLabel: string;
    readonly physicalLayerIdx?: number | null;
    readonly loopIteration?: number | null;
    readonly cacheRole?: DoryCacheRole;
    readonly specOffset?: number | null;
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
        spec.physicalLayerIdx !== undefined ? spec.physicalLayerIdx : selectedAttentionLayerIdx,
      loopIteration: spec.loopIteration !== undefined ? spec.loopIteration : DORY_SPEC_VERIFY_LOOPS,
      partId: spec.partId,
      tensorKey: spec.tensorKey,
      gridCols: spec.gridCols,
      gridRows: spec.gridRows,
      kvPolicy: 'per_loop',
      cacheRole: spec.cacheRole ?? 'spec_verify_deep',
      specOffset: spec.specOffset ?? null,
      modelWeightId: params.modelWeightId,
      extraTags: {
        view: 'self_spec',
        contractA: true,
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

  // --- LEFT STAGE: EXACT 104-BANK CONTRACT A DUAL-CODA ARCHITECTURE RAIL ---
  // Column layout for the 104 banks (`8 + 20 + 60 + 8 + 8 = 104`):
  //   Col 0 (x = -12.6): Input Prelude (`0..7`, 8 banks, REUSED)
  //   Col 1 (x = -11.4): Recurrent Loop 1 (`8..27`, 20 banks, REUSED)
  //   Col 2 (x = -10.1): Recurrent Loop 2 (`28..47`, 20 banks, Verifier)
  //   Col 3 (x = -9.1):  Recurrent Loop 3 (`48..67`, 20 banks, Verifier)
  //   Col 4 (x = -8.1):  Recurrent Loop 4 (`68..87`, 20 banks, Verifier)
  //   Col 5 (x = -6.8):  Output Deep Coda (`88..95`, 8 banks, Verifier persistent)
  //   Col 6 (x = -5.5):  Output Shallow Coda (`96..103`, 8 banks, Drafter persistent, +20 KiB!)
  const bankColX = {
    input: -12.6,
    loop1: -11.4,
    loop2: -10.1,
    loop3: -9.1,
    loop4: -8.1,
    output_deep: -6.8,
    output_shallow: -5.5,
  };
  const bankRowPitchZ = 0.34;
  const bankOriginZ = -3.6;

  if (postStepKvCache.allBanks.length !== DORY_SPEC_CONTRACT_A_BANK_COUNT) {
    throw new Error(
      `Contract A selfSpecScene expected exactly ${DORY_SPEC_CONTRACT_A_BANK_COUNT} persistent KV banks, got ${postStepKvCache.allBanks.length}.`,
    );
  }

  for (let b = 0; b < postStepKvCache.allBanks.length; b += 1) {
    const bank = postStepKvCache.allBanks[b]!;
    const layer = DORY_PHYSICAL_LAYERS[bank.physicalLayerIdx]!;
    const ord = layer.stageAttentionOrdinal ?? 0;
    const isFull = bank.attentionKind === 'full_attention';
    const role = resolveBankCacheRole(bank);

    let cx: number;
    let cz: number;
    if (bank.bankKind === 'input') {
      cx = bankColX.input;
      cz = bankOriginZ + (ord + 6) * bankRowPitchZ;
    } else if (bank.bankKind === 'recurrent') {
      cx =
        bank.loopSlot === 1
          ? bankColX.loop1
          : bank.loopSlot === 2
            ? bankColX.loop2
            : bank.loopSlot === 3
              ? bankColX.loop3
              : bankColX.loop4;
      cz = bankOriginZ + ord * bankRowPitchZ;
    } else if (bank.bankKind === 'output_deep') {
      cx = bankColX.output_deep;
      cz = bankOriginZ + (ord + 6) * bankRowPitchZ;
    } else {
      cx = bankColX.output_shallow;
      cz = bankOriginZ + (ord + 6) * bankRowPitchZ;
    }

    const ex = 0.66;
    const ey = isFull ? 0.42 : 0.28;
    const ez = 0.22;
    const cy = ey * 0.5;

    const isFocused =
      bank.physicalLayerIdx === selectedAttentionLayerIdx &&
      (bank.bankKind !== 'recurrent' || bank.loopSlot === DORY_SPEC_VERIFY_LOOPS);

    const cellFamily =
      bank.bankKind === 'output_shallow'
        ? CELL_MOE_AMBER
        : bank.bankKind === 'output_deep'
          ? CELL_WEIGHT_BLUE
          : role === 'spec_verify_reused'
            ? CELL_SSM_CYAN
            : isFull
              ? CELL_WEIGHT_BLUE
              : CELL_ACTIVATION_GREEN;

    solids.push({
      cx,
      cy,
      cz,
      ex,
      ey,
      ez,
      cellFamily,
      bayId: bank.physicalLayerIdx + 1,
      cellScale: 0.14,
      highlight:
        isFocused
          ? 1
          : bank.bankKind === 'output_shallow' || role === 'spec_verify_reused'
            ? 2
            : 0,
    });

    if (isFocused || bank.bankKind === 'output_shallow' || bank.bankKind === 'output_deep' || isFull) {
      lines.addBoxEdges(
        cx,
        cy,
        cz,
        ex,
        ey,
        ez,
        ownerCounter++,
        isFocused ? 1.45 : bank.bankKind === 'output_shallow' ? 1.1 : 0.8,
        isFocused
          ? LINE_STYLE_ACTIVE
          : bank.bankKind === 'output_shallow'
            ? LINE_STYLE_BF16_GOLD
            : LINE_STYLE_NEUTRAL,
        isFocused ? 0.96 : 0.72,
      );
    }

    const bankMeta = encodeDoryPickMetadata({
      physicalLayerIdx: bank.physicalLayerIdx,
      stage: bank.stage,
      loopIteration: bank.loopSlot,
      partId: `spec_bank_${bank.bankSlotId}`,
      tensorKey: `kv_bank_${bank.bankKind}`,
      kvBankSlotId: bank.bankSlotId,
      kvBankKind: bank.bankKind,
      kvPolicy: 'per_loop',
      cacheRole: role,
      modelWeightId: params.modelWeightId,
      groupRole: 'bank',
      tapRole: 'loop',
      extraTags: {
        view: 'self_spec',
        contractA: true,
        reused: role === 'spec_verify_reused',
        attnKind: bank.attentionKind,
        logicalLen: bank.logicalLen,
      },
    });
    pickBoxes.push(packPickBoxFromAabb(cx, cy, cz, ex, ey, ez, bankMeta));
    aabbs.push({
      min: [cx - ex * 0.5, 0, cz - ez * 0.5],
      max: [cx + ex * 0.5, ey, cz + ez * 0.5],
    });
  }

  // Distinct editorial frames highlighting REUSED (Prelude + Loop-1), Verifier Deep, and Separate Shallow Coda
  const bankStageZ0 = bankOriginZ - 0.55;
  const bankStageZ1 = bankOriginZ + 19 * bankRowPitchZ + 0.55;
  // Shared Reused Input + Loop 1 frame (`28` banks: `0..27`)
  lines.addBoxFrameXZ(
    bankColX.input - 0.55,
    bankStageZ0,
    bankColX.loop1 + 0.55,
    bankStageZ1,
    0.015,
    1.3,
    LINE_STYLE_ACTIVE,
    0.82,
  );
  // Verifier-only Loops 2–4 + Deep Coda frame (`68` banks: `28..95`)
  lines.addBoxFrameXZ(
    bankColX.loop2 - 0.5,
    bankStageZ0,
    bankColX.output_deep + 0.52,
    bankStageZ1,
    0.015,
    1.05,
    LINE_STYLE_GUIDE,
    0.65,
  );
  // Dedicated Contract A persistent `output_shallow` Coda frame (`8` banks: `96..103`, `+20 KiB`)
  lines.addDashedFrameXZ(
    bankColX.output_shallow - 0.52,
    bankOriginZ + 6 * bankRowPitchZ - 0.45,
    bankColX.output_shallow + 0.52,
    bankOriginZ + 13 * bankRowPitchZ + 0.45,
    0.02,
    LINE_STYLE_BF16_GOLD,
    0.92,
    0.2,
    0.12,
    1.35,
  );

  // Flow arrows inside 104-bank rail: Prelude -> Loop 1 -> Loops 2..4 -> Deep Coda, plus Loop 1 -> Shallow Coda
  const flowY = -0.025;
  flatFlow.addPolylineXZ(
    [
      { x: bankColX.input + 0.36, z: bankOriginZ + 9.5 * bankRowPitchZ },
      { x: bankColX.loop1 - 0.36, z: bankOriginZ + 9.5 * bankRowPitchZ },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.swaCyan,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: bankColX.loop1 + 0.36, z: bankOriginZ + 5.0 * bankRowPitchZ },
      { x: bankColX.loop2 - 0.36, z: bankOriginZ + 5.0 * bankRowPitchZ },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: bankColX.loop4 + 0.36, z: bankOriginZ + 9.5 * bankRowPitchZ },
      { x: bankColX.output_deep - 0.36, z: bankOriginZ + 9.5 * bankRowPitchZ },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.preludeIndigo,
    true,
  );
  // Loop 1 `s_1` bypass route directly to persistent `output_shallow` Coda (Contract A!)
  flatFlow.addRoundedRouteXZ(
    [
      { x: bankColX.loop1, z: bankOriginZ + 19 * bankRowPitchZ + 0.32 },
      { x: bankColX.loop1, z: bankStageZ1 + 0.55 },
      { x: bankColX.output_shallow, z: bankStageZ1 + 0.55 },
      { x: bankColX.output_shallow, z: bankOriginZ + 13 * bankRowPitchZ + 0.48 },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.anchorGold,
    0.26,
    true,
  );

  // --- CENTER STAGE: REJECTION SAMPLING BARS, TOP-8 PROBS & RESIDUAL CORRECTION ---
  const centerX = 0.6;
  lines.addBoxFrameXZ(centerX - 3.6, bankStageZ0, centerX + 3.6, bankStageZ1, -0.06, 1.0, LINE_STYLE_GUIDE, 0.38);

  const posBarEx = Math.max(2.6, verifiedCount * 1.05);
  const posBarZ = -2.25;
  const posColLabels = Object.freeze(
    positionBars.map((b) =>
      b.isBonusSlot ? `p${b.absPos}:bonus` : `p${b.absPos}:${b.proposedTokenText ?? b.inputTokenText}`,
    ),
  );
  const sheetQpBars = registerSheet({
    partId: 'spec_qp_bars',
    tensorKey: 'spec_qp_bars',
    label: 'Per-Position Shallow q[d] vs Deep p[d]',
    cx: centerX,
    cy: 0.0,
    cz: posBarZ + 0.72,
    ex: posBarEx,
    ey: sheetH,
    ez: 0.78,
    gridCols: verifiedCount,
    gridRows: 2,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_ACTIVATION,
    highlight: SHEET_HL_PATHWAY,
    slice: qpBarsSlice,
    values: qpBarsArr,
    format: 'prob',
    colLabels: posColLabels,
    rowLabels: Object.freeze(['q (k=1)', 'p (k=4)']),
    colDimLabel: `m=${verifiedCount}`,
    rowDimLabel: 'q vs p',
    physicalLayerIdx: null,
    loopIteration: DORY_SPEC_VERIFY_LOOPS,
    cacheRole: 'spec_verify_deep',
    specOffset: selectedVerifyOffset,
    includeInCellGrid: true,
  });

  // Twin 3D probability bars (q shallow cyan/amber vs p deep emerald/coral) + rollback markers per verify offset
  const slotPitchX = posBarEx / verifiedCount;
  for (let j = 0; j < positionBars.length; j += 1) {
    const bar = positionBars[j]!;
    const bx = centerX - posBarEx * 0.5 + (j + 0.5) * slotPitchX;
    const qVal = bar.proposalProb ?? 0;
    const pVal = bar.targetProb ?? 0;
    const qH = 0.08 + Math.min(1.0, qVal) * 1.45;
    const pH = 0.08 + Math.min(1.0, pVal) * 1.45;
    const barZ = posBarZ - 0.35;
    const barW = Math.min(0.28, slotPitchX * 0.34);

    // Shallow q bar
    solids.push({
      cx: bx - barW * 0.62,
      cy: qH * 0.5,
      cz: barZ,
      ex: barW,
      ey: qH,
      ez: 0.28,
      cellFamily: bar.rolledBack ? CELL_NEUTRAL : CELL_MOE_AMBER,
      bayId: 0,
      cellScale: 0.14,
      highlight: j === selectedVerifyOffset ? 1 : bar.reusedPreludeAndLoop1 ? 2 : 0,
    });
    lines.addBoxEdges(
      bx - barW * 0.62,
      qH * 0.5,
      barZ,
      barW,
      qH,
      0.28,
      ownerCounter++,
      j === selectedVerifyOffset ? 1.3 : 0.85,
      bar.rolledBack ? LINE_STYLE_GUIDE : LINE_STYLE_BF16_GOLD,
      0.85,
    );
    pickBoxes.push(
      packPickBoxFromAabb(
        bx - barW * 0.62,
        qH * 0.5,
        barZ,
        barW + 0.04,
        qH + 0.06,
        0.34,
        encodeDoryPickMetadata({
          physicalLayerIdx: null,
          loopIteration: 1,
          partId: `spec_pos_bar_${j}_q`,
          tensorKey: 'spec_proposal_q',
          absTokenPos: bar.absPos,
          specOffset: j,
          kvPolicy: 'per_loop',
          cacheRole: 'spec_draft',
          modelWeightId: params.modelWeightId,
          tapRole: 'spec',
          extraTags: {
            reused: bar.reusedPreludeAndLoop1,
            accepted: bar.accepted ?? 'na',
            rolledBack: bar.rolledBack,
          },
        }),
      ),
    );

    // Deep p bar
    const pFamily = bar.rolledBack
      ? CELL_EMBED_SLATE
      : bar.accepted === false
        ? CELL_MOE_AMBER
        : bar.isBonusSlot
          ? CELL_WEIGHT_BLUE
          : CELL_ACTIVATION_GREEN;
    solids.push({
      cx: bx + barW * 0.62,
      cy: pH * 0.5,
      cz: barZ,
      ex: barW,
      ey: pH,
      ez: 0.28,
      cellFamily: pFamily,
      bayId: 0,
      cellScale: 0.14,
      highlight: bar.accepted === false || j === selectedVerifyOffset ? 1 : 2,
    });
    lines.addBoxEdges(
      bx + barW * 0.62,
      pH * 0.5,
      barZ,
      barW,
      pH,
      0.28,
      ownerCounter++,
      bar.accepted === false || j === selectedVerifyOffset ? 1.35 : 0.9,
      bar.accepted === false || bar.rolledBack ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      0.88,
    );
    pickBoxes.push(
      packPickBoxFromAabb(
        bx + barW * 0.62,
        pH * 0.5,
        barZ,
        barW + 0.04,
        pH + 0.06,
        0.34,
        encodeDoryPickMetadata({
          physicalLayerIdx: null,
          loopIteration: DORY_SPEC_VERIFY_LOOPS,
          partId: `spec_pos_bar_${j}_p`,
          tensorKey: 'spec_target_p',
          absTokenPos: bar.absPos,
          specOffset: j,
          kvPolicy: 'per_loop',
          cacheRole: 'spec_verify_deep',
          modelWeightId: params.modelWeightId,
          tapRole: 'spec',
          extraTags: {
            reused: bar.reusedPreludeAndLoop1,
            accepted: bar.accepted ?? 'na',
            rolledBack: bar.rolledBack,
          },
        }),
      ),
    );

    // Visual X-cross rollback marker on rolled-back verify positions
    if (bar.rolledBack) {
      const mY = Math.max(qH, pH) + 0.16;
      lines.addSegment(bx - 0.24, mY, barZ - 0.14, bx + 0.24, mY + 0.28, barZ + 0.14, 1.5, LINE_STYLE_ACTIVE, 0.94);
      lines.addSegment(bx - 0.24, mY + 0.28, barZ - 0.14, bx + 0.24, mY, barZ + 0.14, 1.5, LINE_STYLE_ACTIVE, 0.94);
    }

    aabbs.push({
      min: [bx - slotPitchX * 0.45, 0, barZ - 0.25],
      max: [bx + slotPitchX * 0.45, Math.max(qH, pH) + 0.48, barZ + 0.25],
    });
  }

  // Focused position top-8 vocab comparison sheet (Shallow q vs Deep p)
  const topCompLabels = Object.freeze(topCompTokenIds.map((id) => decodeDoryToken(id)));
  const sheetProbComp = registerSheet({
    partId: 'spec_prob_comparison',
    tensorKey: 'spec_prob_comparison',
    label: `Top-8 Probs at Offset ${selectedVerifyOffset} (q vs p)`,
    cx: centerX,
    cy: 0.0,
    cz: 0.35,
    ex: 5.4,
    ey: sheetH,
    ez: 0.82,
    gridCols: topCompTokenIds.length,
    gridRows: 2,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_OPERAND_A,
    slice: probCompSlice,
    values: probCompArr,
    format: 'prob',
    colLabels: topCompLabels,
    rowLabels: Object.freeze(['q (k=1)', 'p (k=4)']),
    colDimLabel: `top-${topCompTokenIds.length}`,
    rowDimLabel: '2',
    physicalLayerIdx: null,
    loopIteration: DORY_SPEC_VERIFY_LOOPS,
    specOffset: selectedVerifyOffset,
    includeInCellGrid: false,
  });

  // Optional binary64 residual correction sheet when rejection occurred
  let sheetResidual: PlacedSheetRecord | null = null;
  if (residualTopTokens.length > 0 && residualDistArr !== null && residualDistSlice !== null) {
    const resCols = residualTopTokens.length;
    sheetResidual = registerSheet({
      partId: 'spec_residual_dist',
      tensorKey: 'spec_residual_dist',
      label: `Binary64 Residual p_res = max(0, p - q) / Z (Z=${(cycleResult.residualDistribution?.massZ ?? 0).toFixed(4)})`,
      cx: centerX,
      cy: 0.0,
      cz: 2.3,
      ex: Math.max(2.4, Math.min(5.4, resCols * 0.68)),
      ey: sheetH,
      ez: 1.05,
      gridCols: resCols,
      gridRows: 3,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      highlight: SHEET_HL_ACTIVE,
      slice: residualDistSlice,
      values: residualDistArr,
      format: 'prob',
      colLabels: Object.freeze(residualTopTokens.map((rt) => rt.tokenText)),
      rowLabels: Object.freeze(['p (k=4)', 'q (k=1)', 'p_res']),
      colDimLabel: `top-${resCols}`,
      rowDimLabel: '3',
      physicalLayerIdx: null,
      loopIteration: DORY_SPEC_VERIFY_LOOPS,
      specOffset: cycleResult.acceptedDraftCount,
      includeInCellGrid: false,
    });
  }

  // --- RIGHT STAGE: LIVE DR20 BATCHED VERIFY ATTENTION PLATES & `cap_swa = 16` ROLLBACK RING ---
  const rightX = 8.4;
  const attnEx = Math.max(2.6, Math.min(5.4, verifyCols * 0.24));
  const attnEz = Math.max(1.0, Math.min(2.4, verifyRows * 0.34));
  lines.addBoxFrameXZ(rightX - 3.6, bankStageZ0, rightX + 3.6, bankStageZ1, -0.06, 1.0, LINE_STYLE_GUIDE, 0.38);

  const verifyQueryLabels = Object.freeze(
    Array.from({ length: verifyRows }, (_, r) => {
      const pos = verifyAttn.startPos + r;
      const vp = cycleResult.verifyPositions[r];
      return `q${pos}:${vp ? decodeDoryToken(vp.inputTokenId) : ''}`;
    }),
  );
  const verifyKeyLabels = Object.freeze(
    Array.from({ length: verifyCols }, (_, c) => `k${verifyAttn.kvAbsPositions[c] ?? c}`),
  );

  // DR20 Live RG32F Causal Softmax Plate (sheetKind = 2)
  const sheetVerifyProbs = registerSheet({
    partId: 'spec_verify_attn_probs',
    tensorKey: 'verify_attn_probs',
    label: `Batched k=4 Verify Softmax (Phys L${selectedAttentionLayerIdx} ${attnSpec.transformersPairBadge})`,
    cx: rightX,
    cy: 0.0,
    cz: -2.25,
    ex: attnEx,
    ey: sheetH,
    ez: attnEz,
    gridCols: verifyCols,
    gridRows: verifyRows,
    sheetKind: SHEET_KIND_CAUSAL_SOFTMAX,
    role: SHEET_ROLE_SOFTMAX,
    highlight: SHEET_HL_OPERAND_B,
    slice: verifyProbsSlice,
    values: verifyProbsH0,
    masked: verifyMaskedFlags,
    format: 'prob',
    colLabels: verifyKeyLabels,
    rowLabels: verifyQueryLabels,
    colDimLabel: `K=${verifyCols}`,
    rowDimLabel: `m=${verifyRows}`,
    physicalLayerIdx: selectedAttentionLayerIdx,
    loopIteration: attnSpec.stage === 'recurrent' ? DORY_SPEC_VERIFY_LOOPS : 0,
    cacheRole: 'spec_verify_deep',
    includeInCellGrid: true,
    extraTags: { dr20LiveGMask: true, attnKind: verifyAttn.attentionKind },
  });

  // DR20 Live RG32F Causal Score Plate (sheetKind = 1)
  const sheetVerifyScores = registerSheet({
    partId: 'spec_verify_attn_scores',
    tensorKey: 'verify_attn_scores',
    label: `Batched k=4 Verify Scores (Phys L${selectedAttentionLayerIdx})`,
    cx: rightX,
    cy: 0.0,
    cz: 0.15,
    ex: attnEx,
    ey: sheetH,
    ez: attnEz,
    gridCols: verifyCols,
    gridRows: verifyRows,
    sheetKind: SHEET_KIND_CAUSAL_SCORES,
    role: SHEET_ROLE_SCORES,
    highlight: SHEET_HL_PATHWAY,
    slice: verifyScoresSlice,
    values: verifyScoresH0,
    masked: verifyMaskedFlags,
    format: 'signed',
    colLabels: verifyKeyLabels,
    rowLabels: verifyQueryLabels,
    colDimLabel: `K=${verifyCols}`,
    rowDimLabel: `m=${verifyRows}`,
    physicalLayerIdx: selectedAttentionLayerIdx,
    loopIteration: attnSpec.stage === 'recurrent' ? DORY_SPEC_VERIFY_LOOPS : 0,
    cacheRole: 'spec_verify_deep',
    includeInCellGrid: false,
    extraTags: { dr20LiveGMask: true, attnKind: verifyAttn.attentionKind },
  });

  // Transaction-local `cap_swa = 16` Rollback Ring Slots (`DR19`)
  const ringZ = 2.6;
  const ringEx = 5.76;
  const ringSlotW = ringEx / config.swaRingCapacity;
  const ringSlotLabels = Object.freeze(
    Array.from({ length: config.swaRingCapacity }, (_, s) => `s${s}`),
  );
  const sheetSwaRing = registerSheet({
    partId: 'spec_swa_ring_strip',
    tensorKey: 'spec_swa_ring',
    label: `Transaction-Local cap_swa=16 SWA Rollback Ring (Phys L${swaRingLayerIdx})`,
    cx: rightX,
    cy: 0.0,
    cz: ringZ + 0.58,
    ex: ringEx,
    ey: sheetH,
    ez: 0.68,
    gridCols: config.swaRingCapacity,
    gridRows: 2,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_KV_K,
    highlight: cycleResult.rolledBackPositions > 0 ? SHEET_HL_ACTIVE : SHEET_HL_PATHWAY,
    slice: swaRingStripSlice,
    values: swaRingStripArr,
    format: 'signed',
    colLabels: ringSlotLabels,
    rowLabels: Object.freeze(['absPos', 'state']),
    colDimLabel: `cap_swa=${config.swaRingCapacity}`,
    rowDimLabel: '2',
    physicalLayerIdx: swaRingLayerIdx,
    loopIteration: 1,
    includeInCellGrid: false,
  });

  for (let s = 0; s < config.swaRingCapacity; s += 1) {
    const slot = swaRingSlots[s]!;
    const sx = rightX - ringEx * 0.5 + (s + 0.5) * ringSlotW;
    const barZ = ringZ - 0.28;
    const barH =
      slot.status === 'committed_new'
        ? 0.58
        : slot.status === 'committed_past'
          ? 0.44
          : slot.status === 'rolled_back'
            ? 0.36
            : slot.status === 'evicted_from_window'
              ? 0.24
              : 0.12;
    const cellFamily =
      slot.status === 'committed_new'
        ? CELL_ACTIVATION_GREEN
        : slot.status === 'committed_past'
          ? CELL_SSM_CYAN
          : slot.status === 'rolled_back'
            ? CELL_MOE_AMBER
            : slot.status === 'evicted_from_window'
              ? CELL_EMBED_SLATE
              : CELL_NEUTRAL;

    solids.push({
      cx: sx,
      cy: barH * 0.5,
      cz: barZ,
      ex: ringSlotW * 0.78,
      ey: barH,
      ez: 0.34,
      cellFamily,
      bayId: swaRingLayerIdx + 1,
      cellScale: 0.14,
      highlight: slot.status === 'rolled_back' ? 1 : slot.status === 'committed_new' ? 2 : 0,
    });
    lines.addBoxEdges(
      sx,
      barH * 0.5,
      barZ,
      ringSlotW * 0.78,
      barH,
      0.34,
      ownerCounter++,
      slot.status === 'rolled_back' ? 1.35 : 0.85,
      slot.status === 'rolled_back' ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      0.82,
    );
    if (slot.status === 'rolled_back') {
      lines.addSegment(sx - 0.12, barH + 0.06, barZ, sx + 0.12, barH + 0.26, barZ, 1.4, LINE_STYLE_ACTIVE, 0.94);
      lines.addSegment(sx - 0.12, barH + 0.26, barZ, sx + 0.12, barH + 0.06, barZ, 1.4, LINE_STYLE_ACTIVE, 0.94);
    }

    pickBoxes.push(
      packPickBoxFromAabb(
        sx,
        barH * 0.5,
        barZ,
        ringSlotW * 0.84,
        barH + 0.08,
        0.38,
        encodeDoryPickMetadata({
          physicalLayerIdx: swaRingLayerIdx,
          loopIteration: 1,
          partId: `spec_swa_slot_${s}`,
          tensorKey: 'spec_swa_ring_slot',
          absTokenPos: slot.absPos,
          kvPolicy: 'per_loop',
          cacheRole: 'spec_verify_reused',
          modelWeightId: params.modelWeightId,
          extraTags: {
            slot: s,
            ringStatus: slot.status,
            rolledBack: slot.status === 'rolled_back',
          },
        }),
      ),
    );
    aabbs.push({
      min: [sx - ringSlotW * 0.5, 0, barZ - 0.2],
      max: [sx + ringSlotW * 0.5, barH + 0.3, barZ + 0.2],
    });

    if (
      slot.absPos !== null &&
      (slot.absPos === cycleResult.startCommittedLen ||
        slot.absPos === Math.max(0, cycleResult.endCommittedLen - 1))
    ) {
      kvDepthTicks.push(
        Object.freeze({ token: slot.absPos, world: Object.freeze([sx, 0.04, barZ - 0.36] as const) }),
      );
    }
  }

  // Stage-to-stage flow ribbons (104-bank rail -> Rejection Sampling -> Batched Verify Plates)
  flatFlow.addPolylineXZ(
    [
      { x: bankColX.output_shallow + 0.65, z: -2.25 },
      { x: centerX - 3.7, z: -2.25 },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.specOxide,
    true,
  );
  flatFlow.addPolylineXZ(
    [
      { x: centerX + 3.7, z: -2.25 },
      { x: rightX - 3.7, z: -2.25 },
    ],
    flowY,
    0.12,
    DORY_FLOW_COLORS.recurrentTeal,
    true,
  );
  ribbons.push({
    cx: -3.9,
    cy: 0.14,
    cz: -2.25,
    ex: 0.68,
    ey: 0.07,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_AMBER,
    alpha: 0.42,
  });
  ribbons.push({
    cx: 4.5,
    cy: 0.14,
    cz: -2.25,
    ex: 0.68,
    ey: 0.07,
    ez: 0.34,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_EMERALD,
    alpha: 0.42,
  });
  ribbons.push({
    cx: bankColX.output_deep,
    cy: 0.12,
    cz: bankStageZ0 - 0.35,
    ex: 0.44,
    ey: 0.06,
    ez: 0.36,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: 0.3,
  });
  ribbons.push({
    cx: bankColX.loop1,
    cy: 0.12,
    cz: bankStageZ0 - 0.35,
    ex: 0.44,
    ey: 0.06,
    ez: 0.36,
    kind: RIBBON_KIND_ARROW_POS_X,
    colorId: RIBBON_COLOR_CYAN,
    alpha: 0.34,
  });

  // 8. Overlay Anchors, Landmarks, Calipers, Badges & Callouts
  overlayAnchors.push(
    Object.freeze({
      id: 'anchor_spec_reused_banks',
      label: 'Shared Reused Banks (0..27)',
      sublabel: `8 Input + 20 Loop-1 · Reused (x, s_1) & KV for ${cycleResult.reusedActivationPositions}/${cycleResult.verifiedPositions} pos`,
      worldPos: Object.freeze([(bankColX.input + bankColX.loop1) * 0.5, 0.78, bankStageZ0 - 0.35] as const),
      tone: 'swa_cyan',
      badge: 'PRELUDE+LOOP-1 REUSE',
    }),
    Object.freeze({
      id: 'anchor_spec_deep_banks',
      label: 'Verifier Deep Banks (28..95)',
      sublabel: '60 Loops 2–4 + 8 output_deep Coda',
      worldPos: Object.freeze([(bankColX.loop2 + bankColX.output_deep) * 0.5, 0.78, bankStageZ0 - 0.35] as const),
      tone: 'prelude_indigo',
    }),
    Object.freeze({
      id: 'anchor_spec_shallow_coda',
      label: 'output_shallow Coda (96..103)',
      sublabel: `+${formatKiB(shallowCodaExtraBytes)} Contract A · catch-up=${cycleResult.shallowCodaCatchupExecuted ? '1-tok ran' : '0 extra work'}`,
      worldPos: Object.freeze([bankColX.output_shallow, 0.78, bankOriginZ + 13 * bankRowPitchZ + 0.9] as const),
      tone: 'spec_oxide',
      badge: 'CONTRACT A DUAL-CODA',
    }),
    Object.freeze({
      id: 'anchor_spec_qp_bars',
      label: 'Division-Free Binary64 Rejection Bars',
      sublabel: `${DORY_DIVISION_FREE_BINARY64_RULE} · acc=${cycleResult.acceptedDraftCount}/${cycleResult.draftCount}`,
      worldPos: localToWorldOnFrame(sheetQpBars.frame, 0, 0.78, 0.75),
      tone: 'spec_oxide',
      badge: cycleResult.outcome,
    }),
    Object.freeze({
      id: 'anchor_spec_prob_comp',
      label: `Top-8 Probs (Offset ${selectedVerifyOffset})`,
      sublabel: 'Shallow q (k=1) vs Deep p (k=4)',
      worldPos: localToWorldOnFrame(sheetProbComp.frame, 0, 0.7, 0.75),
      tone: 'prelude_indigo',
    }),
    Object.freeze({
      id: 'anchor_spec_verify_probs',
      label: `Batched k=4 Verify Softmax (L${selectedAttentionLayerIdx})`,
      sublabel: `m=${verifyRows} pos · K=${verifyCols} · DR20 live G-mask`,
      worldPos: localToWorldOnFrame(sheetVerifyProbs.frame, 0, 0.72, -0.68),
      tone: 'recurrent_teal',
    }),
    Object.freeze({
      id: 'anchor_spec_verify_scores',
      label: `Batched k=4 Verify Scores (L${selectedAttentionLayerIdx})`,
      worldPos: localToWorldOnFrame(sheetVerifyScores.frame, 0, 0.68, 0.68),
      tone: 'muted',
    }),
    Object.freeze({
      id: 'anchor_spec_swa_ring',
      label: `cap_swa=16 SWA Rollback Ring`,
      sublabel: `rolledBack=${cycleResult.rolledBackPositions} · L=${cycleResult.startCommittedLen} -> ${cycleResult.endCommittedLen}`,
      worldPos: localToWorldOnFrame(sheetSwaRing.frame, 0, 0.78, 0.78),
      tone: cycleResult.rolledBackPositions > 0 ? 'warn' : 'swa_cyan',
      badge: 'DR19',
    }),
  );
  if (sheetResidual !== null) {
    overlayAnchors.push(
      Object.freeze({
        id: 'anchor_spec_residual',
        label: 'Binary64 Residual p_res = max(0, p-q)/Z',
        sublabel: `Z = ${(cycleResult.residualDistribution?.massZ ?? 0).toFixed(4)} · zero 1e-12 fallback`,
        worldPos: localToWorldOnFrame(sheetResidual.frame, 0, 0.72, 0.78),
        tone: 'warn',
      }),
    );
  }

  landmarks.push(
    Object.freeze({
      id: 'lm_spec_104_banks',
      label: 'Contract A 104-Bank Topology (Dual Coda)',
      sublabel: '8 Input + 20 Loop-1 + 60 L2–4 + 8 Deep + 8 Shallow Coda',
      world: Object.freeze([-9.0, 0.95, bankStageZ0 - 0.55] as const),
      partId: 'spec_bank_96',
      tone: 'gold',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_spec_rejection',
      label: 'Paper §6.2 Rejection & Residual Sampling',
      sublabel: `${DORY_SPEC_CONTRACT_A_BADGE}`,
      world: Object.freeze([centerX, 0.95, bankStageZ0 - 0.55] as const),
      partId: 'spec_qp_bars',
      tone: 'default',
      tier: 'primary',
    }),
    Object.freeze({
      id: 'lm_spec_verify_ring',
      label: 'Batched k=4 Verify & SWA Rollback Ring',
      sublabel: `cap_swa=${config.swaRingCapacity} · ${cycleResult.rolledBackPositions} rolled back`,
      world: Object.freeze([rightX, 0.95, bankStageZ0 - 0.55] as const),
      partId: 'spec_verify_attn_probs',
      tone: 'cyan',
      tier: 'primary',
    }),
  );

  calipers.push(
    Object.freeze({
      id: 'caliper_spec_104_banks',
      aWorld: Object.freeze([bankColX.input - 0.45, 0.1, bankStageZ1 + 0.85] as const),
      bWorld: Object.freeze([bankColX.output_shallow + 0.45, 0.1, bankStageZ1 + 0.85] as const),
      label: `104 Persistent Banks = ${formatKiB(totalKvPayloadBytes)} (96 Deep + 8 Shallow Coda [+${formatKiB(shallowCodaExtraBytes)}])`,
      partIds: Object.freeze(['spec_bank_0', 'spec_bank_96']),
    }),
    Object.freeze({
      id: 'caliper_spec_swa_rollback',
      aWorld: Object.freeze([rightX - ringEx * 0.5, 0.65, ringZ - 0.68] as const),
      bWorld: Object.freeze([rightX + ringEx * 0.5, 0.65, ringZ - 0.68] as const),
      label: `cap_swa=${config.swaRingCapacity} >= W(${config.slidingWindow})+gamma(${config.speculativeGamma})+1=13`,
      partIds: Object.freeze(['spec_swa_ring_strip']),
    }),
  );

  const provenanceBadges: DoryProvenanceBadge[] = [
    ...ctx.provenanceBadges.filter((b) => b.label !== DORY_SPEC_CONTRACT_A_BADGE),
    Object.freeze({
      label: DORY_SPEC_CONTRACT_A_BADGE,
      tone: 'spec_oxide',
      discrepancyId: 'DR23',
    }),
    Object.freeze({
      label: `104 Banks: 8 Input + 20 Loop-1 Reused + 60 L2–4 + 8 output_deep + 8 output_shallow (+${formatKiB(shallowCodaExtraBytes)})`,
      tone: 'gold',
      discrepancyId: 'DR23',
    }),
    Object.freeze({
      label: `Binary64 Division-Free: ${DORY_DIVISION_FREE_BINARY64_RULE}`,
      tone: 'nlerp_emerald',
    }),
    Object.freeze({
      label: isSynthesizedPreview
        ? 'Isolated Read-Only Preview Cycle (Caller Session Untouched)'
        : `Live Contract A Cycle (${ctx.record.recordId} · L=${cycleResult.startCommittedLen} -> ${cycleResult.endCommittedLen})`,
      tone: isSynthesizedPreview ? 'muted' : 'recurrent_teal',
    }),
  ];

  const callouts: DorySceneCallout[] = [
    Object.freeze({
      id: 'callout_spec_contract_a_provenance',
      title: DORY_SPEC_CONTRACT_A_BADGE,
      body:
        `Paper §6.2 self-speculation (same checkpoint drafting at k=1, verifying at k=4, gamma=${config.speculativeGamma}) ` +
        `is absent from upstream transformers, vLLM, and llama.cpp, and omits Coda KV history across cycles (DR23). ` +
        `Contract A allocates 8 separate persistent output_shallow Coda KV banks (+${formatKiB(shallowCodaExtraBytes)}, total 104 banks = ${formatKiB(totalKvPayloadBytes)}) ` +
        `so shallow drafting matches standalone k=1 Dory while preserving 100% Prelude + Loop-1 (x, s_1) & KV reuse.`,
      tone: 'spec_oxide',
      worldPos: Object.freeze([bankColX.output_shallow, 0.4, bankOriginZ + 9 * bankRowPitchZ] as const),
      badge: 'DR23 · CONTRACT A',
      discrepancyIds: Object.freeze(['DR23'] as const),
    }),
    Object.freeze({
      id: 'callout_spec_reuse_and_coda',
      title: 'Prelude + Loop-1 Reuse & Dual-Coda Catch-Up Rule',
      body:
        `Batched k=4 verification reused Prelude + Loop-1 KV and (x, s_1) activations for ${cycleResult.reusedActivationPositions}/${cycleResult.verifiedPositions} positions ` +
        `(running fresh 1-token Prelude + Loop-1 only for the bonus slot when non-EOS). ` +
        `Outcome '${cycleResult.outcome}' accepted ${cycleResult.acceptedDraftCount}/${cycleResult.draftCount} drafts -> ` +
        (cycleResult.shallowCodaCatchupExecuted
          ? `1-token Shallow Coda catch-up ran on cached s_1 so all 104 banks commit at L'=${cycleResult.endCommittedLen}.`
          : `0 extra Shallow Coda work needed because output_shallow already staged [L..L+a]; all 104 banks commit at L'=${cycleResult.endCommittedLen}.`),
      tone: 'swa_cyan',
      worldPos: Object.freeze([bankColX.loop1, 0.4, bankOriginZ + 9 * bankRowPitchZ] as const),
      badge: `REUSED ${cycleResult.reusedActivationPositions} POS`,
    }),
    Object.freeze({
      id: 'callout_spec_binary64_rejection',
      title: 'Division-Free Binary64 Rejection Sampling & Residual Correction',
      body:
        `Acceptance evaluates '${DORY_DIVISION_FREE_BINARY64_RULE}' strictly in IEEE-754 binary64 with zero 1e-38 denominator clamping and zero 1e-12 fallback. ` +
        (cycleResult.residualDistribution !== null
          ? `Rejected at offset ${cycleResult.acceptedDraftCount}: exact residual mass Z=${cycleResult.residualDistribution.massZ.toFixed(5)} yielded correction token "${decodeDoryToken(cycleResult.correctionOrBonusToken ?? 0)}".`
          : cycleResult.correctionOrBonusToken !== null
            ? `Emitted bonus/correction token "${decodeDoryToken(cycleResult.correctionOrBonusToken)}" with exact target marginal preservation.`
            : `Terminated cleanly on accepted draft EOS without sampling a bonus token.`),
      tone: 'nlerp_emerald',
      worldPos: sheetQpBars.frame.center,
      badge: 'BINARY64 EXACT',
    }),
    Object.freeze({
      id: 'callout_spec_swa_rollback',
      title: `Transaction-Local cap_swa=${config.swaRingCapacity} SWA Rollback (DR19)`,
      body:
        `Every SWA bank holds cap_swa=${config.swaRingCapacity} >= W(${config.slidingWindow}) + gamma(${config.speculativeGamma}) + 1 = 13 slots with per-slot absPosTag. ` +
        `Rewinding ${cycleResult.rolledBackPositions} speculative position(s) from L_staged=${maxStagedLen} back to L'=${cycleResult.endCommittedLen} ` +
        `invalidates only rolled-back tags (-1) with zero historical window overwrite.`,
      tone: cycleResult.rolledBackPositions > 0 ? 'warn' : 'swa_cyan',
      worldPos: sheetSwaRing.frame.center,
      badge: `ROLLED BACK ${cycleResult.rolledBackPositions}`,
      discrepancyIds: Object.freeze(['DR19'] as const),
    }),
  ];

  const activeCellPopover = buildSelfSpecPopover(
    options.hoveredCell,
    sheetsByPartId,
    verifyAttn,
    positionBars,
    residualTopTokens,
    swaRingSlots,
    topCompTokenIds,
    cycleResult,
    selectedVerifyOffset,
  );

  const stageBoundaryAabb = computeBoundsFromPoints(
    [
      [bankColX.input - 0.85, -0.1, bankStageZ0 - 0.6],
      [rightX + 3.8, 1.85, bankStageZ1 + 1.1],
    ],
    0.5,
  );
  const fitBounds = mergeAabbs([...aabbs, stageBoundaryAabb]);

  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlow = flatFlow.pack();

  const selfSpecMetadata: DorySelfSpecViewMetadata = Object.freeze({
    contractBadge: DORY_SPEC_CONTRACT_A_BADGE,
    totalBankCount: DORY_SPEC_CONTRACT_A_BANK_COUNT,
    inputBankCount: 8 as const,
    loop1SharedBankCount: 20 as const,
    loops2To4VerifierBankCount: 60 as const,
    outputDeepBankCount: 8 as const,
    outputShallowBankCount: 8 as const,
    shallowCodaExtraBytes,
    totalKvPayloadBytes,
    isSynthesizedPreview,
    selectedVerifyOffset,
    selectedAttentionLayerIdx,
    shallowCodaCatchupExecuted: cycleResult.shallowCodaCatchupExecuted,
    divisionFreeBinary64Rule: DORY_DIVISION_FREE_BINARY64_RULE,
    hasLiveCausalMaskGChannel: true as const,
    cycleSummary,
    positionBars: Object.freeze(positionBars),
    residualTopTokens: Object.freeze(residualTopTokens),
    swaRingSlots,
    swaRingCapacity: config.swaRingCapacity,
    slidingWindow: config.slidingWindow,
    rolledBackPositions: cycleResult.rolledBackPositions,
  });

  const scene: DorySpatialScene = Object.freeze({
    viewMode: 'self_spec',
    title: VIEW_DESCRIPTOR.label,
    subtitle: `${DORY_SPEC_CONTRACT_A_BADGE} · 104 banks · acc ${cycleResult.acceptedDraftCount}/${cycleResult.draftCount} (${cycleResult.outcome}) · rolled back ${cycleResult.rolledBackPositions}`,
    modelWeightId: params.modelWeightId,
    branchId: ctx.branch.branchId,
    recordId: ctx.record.recordId,
    kLoops: DORY_SPEC_VERIFY_LOOPS,
    kvCacheMode: 'per_loop',
    executionMode: 'self_spec',
    physicalLayerIdx: selectedAttentionLayerIdx,
    loopIteration: DORY_SPEC_VERIFY_LOOPS,
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
    selfSpecMetadata,
  });

  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(
      `View 6 selfSpecScene validation failed: ${validation.errors.join('; ')}`,
    );
  }

  return scene;
}
