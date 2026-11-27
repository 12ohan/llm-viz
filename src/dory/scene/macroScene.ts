/**
 * View 1 (`macro`): Folded & Unrolled Macro Architecture Spatial Scene
 * (`src/dory/scene/macroScene.ts`)
 *
 * Clean-room headless 3D WebGL2 scene builder for NVIDIA Nemotron-Labs-Dory1.0-4B
 * (`#F7F8FA` editorial drafting-table aesthetic, locked Decisions `D1`, `D2`, `D3`, `D4`, `D7`):
 *
 *   - Canonical Physical-72 Sublayer Topology (`0..71` <-> Paired `0..35` `DR4`):
 *       * Input Prelude (`0..15`): `6` SWA + `2` Full + `8` MLP (`Pair L0..L7`, runs `1x`)
 *       * Shared Recurrent Core (`16..55`): `15` SWA + `5` Full + `20` MLP (`Pair L8..L27`, runs `k`x)
 *       * Output Coda (`56..71`): `6` SWA + `2` Full + `8` MLP (`Pair L28..L35`, runs `1x`, no outer skip)
 *     Every stage preserves the exact repeating 8-sublayer motif
 *     `[SWA, MLP, SWA, MLP, SWA, MLP, Full, MLP] x 9` (`3:1` SWA:Full across all 3 stages).
 *
 *   - Instant Toggle Between Folded & Unrolled Views:
 *       * `'folded'`: 1 physical Prelude (`0..15`), 1 physical Shared Recurrent Core (`16..55`)
 *         with sweeping `-X` latent feedback arc (`s_t -> +`) and persistent golden input-anchor
 *         rail (`x = s_0` feeding `u_t = N(s_{t-1} + x)` incl. literal `u_1 = N(2x)`, `DR8`),
 *         1 physical Coda (`56..71`), and a side-mounted `(1 + k) x 8` live `RG32F` trajectory
 *         strip showing distinct per-loop states `s_1 .. s_k`.
 *       * `'unrolled'`: Full `32 + 40k` executed sublayers arrayed across `k` recurrent pass
 *         columns (`t = 1..k`, supporting trained `k in {1..4}` and extrapolation `k in {6, 8}`),
 *         visually linked by horizontal Shared Weight-ID Provenance Rails (`modelWeightId`,
 *         zero weight cloning, zero loop adapters) while each pass carries distinct live `RG32F`
 *         8-channel `u_t` entry and `s_t` exit activation sheets plus inter-loop `s_{t-1} -> u_t`
 *         flow ribbons and a persistent `+ x` anchor bus.
 *
 *   - Parallel Dual-Axis KV-Bank Rail (`36..176` banks, or `104` banks under Contract A):
 *       * `per_loop`: visibly widens from `36` (`k=1`) -> `56` (`k=2`) -> `76` (`k=3`) -> `96` (`k=4`)
 *         -> `136` (`k=6`) -> `176` (`k=8`) persistent banks (`(12 + 15k)` SWA + `(4 + 5k)` Full),
 *         preserving the exact `S`-invariant `(4 + 5k) / 9` ratio.
 *       * `last_loop`: stays flat at `36` banks (`27` SWA + `9` Full) at any `k` with an explicit
 *         gold overwrite frame (`DoryPastReader` read-only for `t < k`, commit on `t = k`, `DR17`)
 *         and ghost outlines showing the `20 * (k - 1)` avoided recurrent bank columns.
 *       * Contract A Self-Speculation (`executionMode === 'self_spec'` / `includeShallowCoda`):
 *         renders the `+8` dedicated persistent `output_shallow` Coda KV banks (`96..103` at `k=4`,
 *         `104` banks total) badged `Reconstructed from Paper §6.2 (Contract A)` (`DR23`).
 *
 *   - Strictly `D4`-compliant: imports leaf primitives solely from `src/render/*`,
 *     `src/shared/overlayLayout.ts`, `src/dory/{types,spec}.ts`, and
 *     `src/dory/scene/{types,provenance}.ts`. Zero DOM, WebGL calls, ambient RNG,
 *     or Gate 4b/4c/sibling imports.
 */

import type { AABB } from '../../render/camera';
import type { PickBox } from '../../render/pick';
import {
  CELL_ACTIVATION_GREEN,
  CELL_EMBED_SLATE,
  CELL_MOE_AMBER,
  CELL_SSM_CYAN,
  CELL_WEIGHT_BLUE,
  RIBBON_COLOR_AMBER,
  RIBBON_COLOR_CYAN,
  RIBBON_COLOR_EMERALD,
  RIBBON_COLOR_PERIWINKLE,
  RIBBON_KIND_ARROW_NEG_Y,
  RIBBON_KIND_ARROW_POS_X,
  RIBBON_KIND_ARROW_POS_Y,
  type DimensionCaliper,
  type KvDepthTickAnchor,
  type LandmarkAnchor,
  type Vec3,
} from '../../render/primitives';
import {
  FRAME_XY_AXES,
  makeSheetFrame3D,
  type SheetFrame3D,
} from '../../render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_BF16_GOLD,
  SHEET_HL_NORMAL,
  SHEET_HL_PATHWAY,
  SHEET_KIND_STANDARD,
  SHEET_ROLE_ACTIVATION,
  SHEET_ROLE_RESIDUAL,
} from '../../render/sheets';
import {
  computeSheetCellWorldPos,
  type ActiveSheetCellGrid,
  type CellPopoverSpec,
} from '../../shared/overlayLayout';
import {
  DORY_PHYSICAL_LAYERS,
  computeEffectivePairedDepth,
  computeExecutedDepth,
  computeExecutedLayerIdx,
  computeKvBankCounts,
  computeMiniKvCacheBytes,
  decodeDoryToken,
  getDoryPhysicalLayerSpec,
  validateDoryRecurrentLoops,
} from '../spec';
import {
  DORY_MINI_CONFIG,
  DORY_SPEC_CONTRACT_A_BADGE,
  type DoryCacheRole,
  type DoryKvBankKind,
  type DoryKvCacheMode,
  type DoryPhysicalLayerSpec,
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
  packPickBoxFromAabb,
  packPickBoxFromFrame,
  resolveDorySessionContext,
  validateSceneGeometryFinite,
} from './provenance';
import {
  DORY_SPATIAL_VIEW_MODES,
  type DoryMacroDisplayMode,
  type DoryMacroKvRailSummary,
  type DoryMacroSceneOptions,
  type DoryMacroViewMetadata,
  type DoryOverlayAnchor,
  type DoryPickIdentity,
  type DoryProvenanceBadge,
  type DoryResolvedSessionContext,
  type DorySceneCallout,
  type DorySharedWeightBinding,
  type DorySpatialScene,
} from './types';

// ============================================================================
// MACRO GEOMETRY CONSTANTS & DETERMINISTIC LAYER COORDINATES (D1, D2, D7)
// ============================================================================

const NUM_PHYSICAL_LAYERS = DORY_MINI_CONFIG.numHiddenLayers; // 72
const NUM_PAIRED_LAYERS = DORY_MINI_CONFIG.numPairedLayers; // 36
const N_INPUT_LAYERS = DORY_MINI_CONFIG.nInputLayers; // 16 (8 paired)
const N_RECURRENT_LAYERS = DORY_MINI_CONFIG.nRecurrentLayers; // 40 (20 paired)
const N_OUTPUT_LAYERS = DORY_MINI_CONFIG.nOutputLayers; // 16 (8 paired)
const INPUT_PAIRS = N_INPUT_LAYERS >> 1; // 8
const RECURRENT_PAIRS = N_RECURRENT_LAYERS >> 1; // 20
const RECURRENT_END_PAIR = INPUT_PAIRS + RECURRENT_PAIRS; // 28
const HIDDEN_SIZE = DORY_MINI_CONFIG.hiddenSize; // 8

const PAIR_PITCH_Y = 0.86;
const MOTIF_GAP_Y = 0.28;
const STAGE_GAP_Y = 3.10;
const SUBLAYER_STAGGER_Y = 0.045;

/** Deterministic Y coordinate for paired layer index `0..35` (descending top-to-bottom along `-Y`). */
export function doryMacroPairRowY(pairedLayerIdx: number): number {
  const pair = Math.max(0, Math.min(NUM_PAIRED_LAYERS - 1, Math.trunc(pairedLayerIdx)));
  const completedMotifs = Math.floor(pair / 4);
  let y = -pair * PAIR_PITCH_Y - completedMotifs * MOTIF_GAP_Y;
  if (pair >= INPUT_PAIRS) {
    y -= STAGE_GAP_Y;
  }
  if (pair >= RECURRENT_END_PAIR) {
    y -= STAGE_GAP_Y;
  }
  return y;
}

/**
 * Deterministic Y coordinate for canonical physical sublayer `0..71` (`D1`):
 * even indices (`Attn`: SWA or Full) sit subtly above odd indices (`MLP`) within their pair row.
 */
export function doryMacroPhysicalLayerY(physicalLayerIdx: number): number {
  const p = Math.max(0, Math.min(NUM_PHYSICAL_LAYERS - 1, Math.trunc(physicalLayerIdx)));
  const pairY = doryMacroPairRowY(p >> 1);
  return p % 2 === 0 ? pairY + SUBLAYER_STAGGER_Y : pairY - SUBLAYER_STAGGER_Y;
}

const PRELUDE_TOP_Y = doryMacroPairRowY(0);
const PRELUDE_BOT_Y = doryMacroPairRowY(INPUT_PAIRS - 1);
const CORE_TOP_Y = doryMacroPairRowY(INPUT_PAIRS);
const CORE_BOT_Y = doryMacroPairRowY(RECURRENT_END_PAIR - 1);
const CODA_TOP_Y = doryMacroPairRowY(RECURRENT_END_PAIR);
const CODA_BOT_Y = doryMacroPairRowY(NUM_PAIRED_LAYERS - 1);

const EMBED_BOOKEND_Y = PRELUDE_TOP_Y + 2.55;
const ANCHOR_SHEET_Y = PRELUDE_BOT_Y - 1.38;
const LOOP_ENTRY_Y = CORE_TOP_Y + 1.38;
const LOOP_EXIT_Y = CORE_BOT_Y - 1.48;
const LM_HEAD_Y = CODA_BOT_Y - 2.45;
const LOGIT_SCALE_Y = LM_HEAD_Y - 1.28;

const MACRO_VIEW_DESCRIPTOR = DORY_SPATIAL_VIEW_MODES[0]!; // 'macro'

export const DORY_MACRO_LAYOUT = Object.freeze({
  numPhysicalLayers: NUM_PHYSICAL_LAYERS,
  numPairedLayers: NUM_PAIRED_LAYERS,
  nInputLayers: N_INPUT_LAYERS,
  nRecurrentLayers: N_RECURRENT_LAYERS,
  nOutputLayers: N_OUTPUT_LAYERS,
  motifPeriod: DORY_MINI_CONFIG.motifPeriod,
  pairPitchY: PAIR_PITCH_Y,
  motifGapY: MOTIF_GAP_Y,
  stageGapY: STAGE_GAP_Y,
  sublayerStaggerY: SUBLAYER_STAGGER_Y,
  preludeTopY: PRELUDE_TOP_Y,
  preludeBotY: PRELUDE_BOT_Y,
  coreTopY: CORE_TOP_Y,
  coreBotY: CORE_BOT_Y,
  codaTopY: CODA_TOP_Y,
  codaBotY: CODA_BOT_Y,
  embedBookendY: EMBED_BOOKEND_Y,
  anchorSheetY: ANCHOR_SHEET_Y,
  loopEntryY: LOOP_ENTRY_Y,
  loopExitY: LOOP_EXIT_Y,
  lmHeadY: LM_HEAD_Y,
  logitScaleY: LOGIT_SCALE_Y,
  foldedPassPitchX: 5.40,
  compactUnrolledPassPitchX: 4.55,
  feedbackArcX: -5.45,
  anchorRailX: -3.45,
  foldedTrajectoryStripX: -7.65,
  kvRailGapX: 2.85,
  kvColPitchStandard: 1.16,
  kvColPitchCompact: 0.94,
  weightProvenanceRailZ: -0.78,
  sheetFrontZ: 0.28,
});

const CHANNEL_LABELS_8: readonly string[] = Object.freeze([
  'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7',
]);

// ============================================================================
// PURE METADATA & SHARED-WEIGHT PROVENANCE HELPERS
// ============================================================================

/**
 * Computes exact parallel KV-bank rail counts for View 1 (`36..176` standard banks,
 * or `104` banks under Contract A self-speculation `includeShallowCoda = true`).
 */
export function buildDoryMacroKvRailSummary(
  kLoops: number,
  kvCacheMode: DoryKvCacheMode,
  includeShallowCoda: boolean,
): DoryMacroKvRailSummary {
  const counts = computeKvBankCounts(kLoops, kvCacheMode, includeShallowCoda);
  return Object.freeze({
    kvCacheMode: counts.kvCacheMode,
    includeShallowCoda: counts.includeShallowCoda,
    totalBanks: counts.totalBanks,
    inputBanks: counts.inputSwaBanks + counts.inputFullBanks,
    recurrentBanks: counts.recurrentSwaBanks + counts.recurrentFullBanks,
    outputDeepBanks: counts.outputDeepSwaBanks + counts.outputDeepFullBanks,
    outputShallowBanks: counts.outputShallowSwaBanks + counts.outputShallowFullBanks,
    totalSwaBanks: counts.totalSwaBanks,
    totalFullBanks: counts.totalFullBanks,
  });
}

/**
 * Builds the `k`-element shared-weight binding ledger proving that every recurrent pass
 * `t = 1..kLoops` binds the identical physical layers `16..55` and singleton `modelWeightId`
 * (zero weight cloning, zero loop adapters), while mapping either to distinct `per_loop`
 * KV bank ranges (`8 + (t-1)*20 .. 7 + t*20`) or the shared `last_loop` overwrite range (`8..27`).
 */
export function buildDorySharedWeightBindings(
  kLoops: number,
  kvCacheMode: DoryKvCacheMode,
  modelWeightId: string,
): readonly DorySharedWeightBinding[] {
  const k = validateDoryRecurrentLoops(kLoops);
  const bindings: DorySharedWeightBinding[] = [];
  const isSharedOverwriteBank = kvCacheMode === 'last_loop';
  for (let t = 1; t <= k; t += 1) {
    const executedLayerStart = computeExecutedLayerIdx(N_INPUT_LAYERS, t, k);
    const executedLayerEnd = computeExecutedLayerIdx(N_INPUT_LAYERS + N_RECURRENT_LAYERS - 1, t, k);
    const kvBankRangeStart = isSharedOverwriteBank ? 8 : 8 + (t - 1) * 20;
    const kvBankRangeEnd = kvBankRangeStart + 19;
    bindings.push(
      Object.freeze({
        loopIteration: t,
        executedLayerStart,
        executedLayerEnd,
        physicalLayerStart: 16 as const,
        physicalLayerEnd: 55 as const,
        modelWeightId,
        kvBankRangeStart,
        kvBankRangeEnd,
        isSharedOverwriteBank,
      }),
    );
  }
  return Object.freeze(bindings);
}

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

function isResolvedSessionContext(
  value: DoryResolvedSessionContext | DorySession,
): value is DoryResolvedSessionContext {
  return 'resolveForwardTrace' in value && typeof value.resolveForwardTrace === 'function';
}

function resolveContext(
  ctxOrSession: DoryResolvedSessionContext | DorySession,
  recordId?: string,
): DoryResolvedSessionContext {
  if (isResolvedSessionContext(ctxOrSession)) {
    if (recordId !== undefined && recordId !== ctxOrSession.record.recordId) {
      return resolveDorySessionContext(ctxOrSession.session, recordId);
    }
    return ctxOrSession;
  }
  return resolveDorySessionContext(ctxOrSession, recordId);
}

function shortWeightDigest(modelWeightId: string): string {
  return modelWeightId.length > 12 ? `${modelWeightId.slice(0, 12)}…` : modelWeightId;
}

function formatFloatSigned(v: number): string {
  if (!Number.isFinite(v)) return '0.000';
  const clean = Math.abs(v) < 5e-4 ? 0 : v;
  const sign = clean < 0 ? '-' : '+';
  return `${sign}${Math.abs(clean).toFixed(3)}`;
}

function computeBankSlotIdForSublayer(
  spec: DoryPhysicalLayerSpec,
  loopIteration: number,
  kLoops: number,
  kvCacheMode: DoryKvCacheMode,
  codaKind: 'output_deep' | 'output_shallow' = 'output_deep',
): number | null {
  if (spec.stageAttentionOrdinal === null) return null;
  const recurrentBankCount = kvCacheMode === 'per_loop' ? 20 * kLoops : 20;
  if (spec.stage === 'input') {
    return spec.stageAttentionOrdinal; // 0..7
  }
  if (spec.stage === 'recurrent') {
    const loopOffset = kvCacheMode === 'per_loop' ? Math.max(0, loopIteration - 1) * 20 : 0;
    return 8 + loopOffset + spec.stageAttentionOrdinal;
  }
  const deepBase = 8 + recurrentBankCount;
  if (codaKind === 'output_shallow') {
    return deepBase + 8 + spec.stageAttentionOrdinal;
  }
  return deepBase + spec.stageAttentionOrdinal;
}

function resolveBankCacheRole(
  executionMode: 'standard' | 'self_spec',
  bankKind: DoryKvBankKind,
  loopSlot: number,
): DoryCacheRole {
  if (executionMode !== 'self_spec') {
    return 'standard';
  }
  if (bankKind === 'input' || (bankKind === 'recurrent' && loopSlot === 1)) {
    return 'spec_verify_reused';
  }
  if (bankKind === 'output_shallow') {
    return 'spec_draft';
  }
  return 'spec_verify_deep';
}

function pushBoundsPoint(points: Vec3[], cx: number, cy: number, cz: number, ex: number, ey: number, ez: number): void {
  const hx = ex * 0.5;
  const hy = ey * 0.5;
  const hz = ez * 0.5;
  points.push([cx - hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz + hz]);
}

function safeStateVector8(vec: Float32Array | undefined): Float32Array {
  if (vec !== undefined && vec.length >= HIDDEN_SIZE) {
    return vec.subarray(0, HIDDEN_SIZE);
  }
  const out = new Float32Array(HIDDEN_SIZE);
  if (vec !== undefined) {
    for (let i = 0; i < Math.min(vec.length, HIDDEN_SIZE); i += 1) {
      out[i] = vec[i]!;
    }
  }
  return out;
}

interface PlacedSheetRecord {
  readonly id: string;
  readonly tensorKey: string;
  readonly label: string;
  readonly formula: string;
  readonly loopIteration: number;
  readonly frame: SheetFrame3D;
  readonly values: Float32Array;
  readonly rows: number;
  readonly cols: number;
  readonly rowLabels: readonly string[];
}

// ============================================================================
// MAIN VIEW 1 BUILDER: FOLDED / UNROLLED MACRO ARCHITECTURE + KV-BANK RAIL
// ============================================================================

/**
 * Builds the headless 5-pass WebGL2 spatial scene for View 1 (`macro` — Folded / Unrolled
 * Macro Architecture with physical `0..71` sublayers, shared weight-ID provenance rails,
 * live `RG32F` 8-channel latent state strips, and parallel `36..176` / `104` Contract A
 * KV-bank rail). Strictly read-only — never mutates `DorySession`.
 */
export function buildDoryMacroScene(
  ctxOrSession: DoryResolvedSessionContext | DorySession,
  options: DoryMacroSceneOptions = {},
): DorySpatialScene {
  const ctx = resolveContext(ctxOrSession, options.recordId);
  const { params, branch, record, kLoops, kvCacheMode, executionMode } = ctx;
  const displayMode: DoryMacroDisplayMode = options.displayMode ?? 'folded';
  const includeShallowCoda =
    branch.config.includeShallowCoda || executionMode === 'self_spec';

  const selectedPhysicalLayerIdx: number | null =
    options.selectedPhysicalLayerIdx === null
      ? null
      : options.selectedPhysicalLayerIdx !== undefined
        ? Math.max(0, Math.min(NUM_PHYSICAL_LAYERS - 1, Math.trunc(options.selectedPhysicalLayerIdx)))
        : 16; // default focus: p=16 (first Recurrent Core SWA sublayer, planted >1 NLERP layer)

  const selectedSpec =
    selectedPhysicalLayerIdx !== null
      ? getDoryPhysicalLayerSpec(selectedPhysicalLayerIdx)
      : null;

  const selectedLoopIteration: number | null =
    options.selectedLoopIteration === null
      ? null
      : options.selectedLoopIteration !== undefined
        ? Math.max(0, Math.min(kLoops, Math.trunc(options.selectedLoopIteration)))
        : selectedSpec?.stage === 'recurrent'
          ? 1
          : 0;

  const activeRecurrentLoop =
    selectedLoopIteration !== null && selectedLoopIteration >= 1 && selectedLoopIteration <= kLoops
      ? selectedLoopIteration
      : kLoops;

  const hoveredPartId = options.hoveredPartId ?? null;
  const hoveredCell: DoryPickIdentity | null = options.hoveredCell ?? null;

  const modelWeightId = params.modelWeightId;
  const shortWeightId = shortWeightDigest(modelWeightId);
  const executedSublayerCount = computeExecutedDepth(kLoops, params.config);
  const effectivePairedDepth = computeEffectivePairedDepth(kLoops, params.config);
  const kvRail = buildDoryMacroKvRailSummary(kLoops, kvCacheMode, includeShallowCoda);
  const sharedRecurrentWeightBindings = buildDorySharedWeightBindings(
    kLoops,
    kvCacheMode,
    modelWeightId,
  );
  const kvMemorySummary = computeMiniKvCacheBytes(
    kLoops,
    kvCacheMode,
    includeShallowCoda,
    params.config,
  );

  // Committed token summary for annotations & state-strip provenance
  const committedLen = Math.max(1, record.committedLen);
  const lastCommittedPos = committedLen - 1;
  const lastTokenId =
    record.committedSequence.length > 0
      ? record.committedSequence[record.committedSequence.length - 1]!
      : DORY_MINI_CONFIG.bosTokenId;
  const lastTokenText = decodeDoryToken(lastTokenId);

  // Horizontal layout parameters adapting smoothly across Folded & Unrolled k=1..8
  const isUnrolled = displayMode === 'unrolled';
  const passCount = isUnrolled ? kLoops : 1;
  const compactColumns = isUnrolled && kLoops > 4;
  const passPitchX = compactColumns
    ? DORY_MACRO_LAYOUT.compactUnrolledPassPitchX
    : DORY_MACRO_LAYOUT.foldedPassPitchX;

  const attnOffsetX = compactColumns ? -0.92 : -1.16;
  const mlpOffsetX = compactColumns ? 0.92 : 1.16;
  const swaEx = compactColumns ? 1.58 : 1.98;
  const fullEx = compactColumns ? 1.72 : 2.12;
  const mlpEx = compactColumns ? 1.58 : 1.98;
  const sublayerEy = 0.54;
  const fullAttnEy = 0.58;
  const sublayerEz = 0.84;
  const fullAttnEz = 1.04;
  const colHalfW = Math.abs(attnOffsetX) + fullEx * 0.5;

  const passCenterX = (loopT: number): number =>
    isUnrolled ? (loopT - 1) * passPitchX : 0;

  const preludeX = passCenterX(1);
  const codaX = passCenterX(passCount);
  const rightmostSpineX = codaX + colHalfW;

  // Parallel KV-Bank Rail layout along +X right beside the backbone spine
  const kvColCount = kvCacheMode === 'per_loop' ? kLoops : 1;
  const kvColPitch =
    kvCacheMode === 'per_loop' && kLoops > 4
      ? DORY_MACRO_LAYOUT.kvColPitchCompact
      : DORY_MACRO_LAYOUT.kvColPitchStandard;
  const kvRailStartX = rightmostSpineX + DORY_MACRO_LAYOUT.kvRailGapX;
  const kvSwaEx = kvColPitch * 0.68;
  const kvFullEx = kvColPitch * 0.74;
  const kvSwaEy = 0.44;
  const kvFullEy = 0.48;
  const kvSwaEz = 0.72;
  const kvFullEz = 1.28;
  const kvColumnX = (colIndex0: number): number => kvRailStartX + colIndex0 * kvColPitch;
  const shallowCodaBankX = kvRailStartX + kvColPitch * 1.45;
  const kvRailRightEdgeX = Math.max(
    kvColumnX(kvColCount - 1) + kvFullEx * 0.5,
    includeShallowCoda ? shallowCodaBankX + kvFullEx * 0.5 : -Infinity,
  );

  // 5-Pass Builders + Compact Live RG32F Atlas (<= 64 KiB View 1 Budget)
  const solids = new DorySolidBufferBuilder();
  const lines = new DoryLineBufferBuilder();
  const ribbons = new DoryRibbonBufferBuilder();
  const sheets = new DorySheetBufferBuilder();
  const flows = new DoryFlatFlowBufferBuilder();
  const pickBoxes: PickBox[] = [];
  const landmarks: LandmarkAnchor[] = [];
  const overlayAnchors: DoryOverlayAnchor[] = [];
  const calipers: DimensionCaliper[] = [];
  const kvDepthTicks: KvDepthTickAnchor[] = [];
  const boundsPoints: Vec3[] = [];
  const placedSheets: PlacedSheetRecord[] = [];

  // Dynamically sized power-of-16 atlas height staying strictly inside View 1's 64 KiB budget
  // (64 x 32 RG32F = 16,384 B = 16.00 KiB for k <= 8; expands gracefully for higher k).
  const atlasWidth = 64;
  const minRowsNeeded = Math.max(32, Math.ceil((kLoops * 3 + 12) / 16) * 16);
  const atlasHeight = Math.min(128, minRowsNeeded);
  const atlasPacker = new DoryAtlasPacker(
    atlasWidth,
    atlasHeight,
    MACRO_VIEW_DESCRIPTOR.atlasBudgetBytes,
  );

  // Pack live 8-channel hypersphere state vectors directly from immutable DorySessionRecord
  const xAnchorVec = safeStateVector8(record.lastTokenXAnchor);
  const xAnchorSlice = atlasPacker.packMatrix('x_anchor_8ch', xAnchorVec, 1, HIDDEN_SIZE, {
    mode: 0,
    minFloor: 0.35,
  });

  const uSlices = [];
  const sSlices = [];
  const uVecs: Float32Array[] = [];
  const sVecs: Float32Array[] = [];
  const trajectoryMatrix = new Float32Array((1 + kLoops) * HIDDEN_SIZE);
  trajectoryMatrix.set(xAnchorVec, 0);

  for (let t = 1; t <= kLoops; t += 1) {
    const uVec = safeStateVector8(record.lastTokenLoopEntryStates[t - 1]);
    const sVec = safeStateVector8(record.lastTokenLoopExitStates[t - 1]);
    uVecs.push(uVec);
    sVecs.push(sVec);
    trajectoryMatrix.set(sVec, t * HIDDEN_SIZE);

    uSlices.push(
      atlasPacker.packMatrix(`u_loop_${t}_8ch`, uVec, 1, HIDDEN_SIZE, {
        mode: 0,
        minFloor: 0.35,
        loopIteration: t,
      }),
    );
    sSlices.push(
      atlasPacker.packMatrix(`s_loop_${t}_8ch`, sVec, 1, HIDDEN_SIZE, {
        mode: 0,
        minFloor: 0.35,
        loopIteration: t,
      }),
    );
  }

  const trajectorySlice = atlasPacker.packMatrix(
    'latent_trajectory_8ch',
    trajectoryMatrix,
    1 + kLoops,
    HIDDEN_SIZE,
    {
      mode: 0,
      minFloor: 0.35,
    },
  );

  let ownerCounter = 1;

  // Helper: emit one physical/executed sublayer cuboid + crisp edges + DR21 PickBox
  const emitSublayerBlock = (
    spec: DoryPhysicalLayerSpec,
    colX: number,
    loopIter: number,
    execIdx: number,
    emitFullEdges: boolean,
  ): void => {
    const isAttn = spec.attentionKind !== null;
    const isFull = spec.sublayerKind === 'full_attention';
    const isSwa = spec.sublayerKind === 'sliding_attention';
    const cx = colX + (isAttn ? attnOffsetX : mlpOffsetX);
    const cy = doryMacroPhysicalLayerY(spec.physicalLayerIdx);
    const cz = 0;
    const ex = isFull ? fullEx : isSwa ? swaEx : mlpEx;
    const ey = isFull ? fullAttnEy : sublayerEy;
    const ez = isFull ? fullAttnEz : sublayerEz;

    const partId = isUnrolled
      ? `exec_e${execIdx}_p${spec.physicalLayerIdx}_t${loopIter}`
      : `phys_l${spec.physicalLayerIdx}`;

    const isPhysMatch =
      selectedPhysicalLayerIdx === spec.physicalLayerIdx ||
      hoveredCell?.physicalLayerIdx === spec.physicalLayerIdx;
    const isExactLoopMatch =
      !isUnrolled ||
      spec.stage !== 'recurrent' ||
      selectedLoopIteration === null ||
      selectedLoopIteration === 0 ||
      selectedLoopIteration === loopIter;
    const isSelected = (isPhysMatch && isExactLoopMatch) || hoveredPartId === partId;
    const isSharedPeerHighlight =
      isUnrolled && spec.stage === 'recurrent' && isPhysMatch && !isSelected;

    const cellFamily = isFull
      ? CELL_WEIGHT_BLUE
      : isSwa
        ? CELL_SSM_CYAN
        : CELL_ACTIVATION_GREEN;

    solids.push({
      cx,
      cy,
      cz,
      ex,
      ey,
      ez,
      cellFamily,
      fp8: 0,
      phase: spec.physicalLayerIdx * 0.19 + (isAttn ? 0 : 0.45),
      bayId: spec.physicalLayerIdx + 1,
      cellScale: compactColumns ? 0.19 : 0.21,
      highlight: isSelected ? 1 : isSharedPeerHighlight || isFull ? 2 : 0,
    });
    pushBoundsPoint(boundsPoints, cx, cy, cz, ex, ey, ez);

    if (emitFullEdges || isSelected || isSharedPeerHighlight || isFull) {
      lines.addBoxEdges(
        cx,
        cy,
        cz,
        ex,
        ey,
        ez,
        ownerCounter++,
        isSelected ? 1.55 : isSharedPeerHighlight ? 1.25 : isFull ? 1.1 : 0.82,
        isSelected
          ? LINE_STYLE_ACTIVE
          : isSharedPeerHighlight
            ? LINE_STYLE_BF16_GOLD
            : LINE_STYLE_NEUTRAL,
        isSelected ? 0.96 : isSharedPeerHighlight ? 0.88 : isFull ? 0.78 : 0.56,
      );
    }

    // Intra-pair coupler ribbon from Attention (even p) -> SwiGLU MLP (odd p+1)
    if (isAttn) {
      const attnRightX = cx + ex * 0.5;
      const mlpLeftX = colX + mlpOffsetX - mlpEx * 0.5;
      const gapX = mlpLeftX - attnRightX;
      if (gapX > 0.04) {
        ribbons.push({
          cx: 0.5 * (attnRightX + mlpLeftX),
          cy: doryMacroPairRowY(spec.pairedLayerIdx),
          cz: 0,
          ex: Math.max(0.12, gapX - 0.04),
          ey: 0.16,
          ez: 0.18,
          kind: RIBBON_KIND_ARROW_POS_X,
          colorId: isFull ? RIBBON_COLOR_PERIWINKLE : RIBBON_COLOR_CYAN,
          alpha: isSelected ? 0.72 : 0.34,
        });
      }
    }

    const kvBankSlotId = computeBankSlotIdForSublayer(
      spec,
      loopIter,
      kLoops,
      kvCacheMode,
      'output_deep',
    );
    const kvBankKind: DoryKvBankKind | null =
      kvBankSlotId === null
        ? null
        : spec.stage === 'output'
          ? 'output_deep'
          : spec.stage;
    const cacheRole: DoryCacheRole | null =
      kvBankKind === null
        ? null
        : resolveBankCacheRole(
            executionMode,
            kvBankKind,
            spec.stage === 'recurrent' ? (kvCacheMode === 'per_loop' ? loopIter : 1) : 0,
          );

    const pickMeta = encodeDoryPickMetadata({
      physicalLayerIdx: spec.physicalLayerIdx,
      executedLayerIdx: execIdx,
      stage: spec.stage,
      loopIteration: loopIter,
      partId,
      tensorKey: isAttn ? 'attn_sublayer' : 'mlp_sublayer',
      kvBankSlotId,
      kvBankKind,
      kvPolicy: kvBankSlotId !== null ? kvCacheMode : null,
      cacheRole,
      modelWeightId,
      groupRole: kvBankSlotId !== null ? 'bank' : 'loop',
      tapRole: 'loop',
      extraTags: {
        motif: spec.motifSlot,
        kind: spec.sublayerKind,
        display: displayMode,
        sharedWeight: spec.stage === 'recurrent',
      },
    });
    pickBoxes.push(packPickBoxFromAabb(cx, cy, cz, ex, ey, ez, pickMeta));
  };

  // ==========================================================================
  // 1. TOP BOOKEND: `backbone.embeddings.weight [64, 8]` (Untied, Unit-Norm DR5)
  // ==========================================================================
  const embedEx = colHalfW * 2.02;
  const embedEy = 1.04;
  const embedEz = 1.06;
  const embedActive =
    hoveredPartId === 'bookend_embeddings' || hoveredCell?.partId === 'bookend_embeddings';

  solids.push({
    cx: preludeX,
    cy: EMBED_BOOKEND_Y,
    cz: 0,
    ex: embedEx,
    ey: embedEy,
    ez: embedEz,
    cellFamily: CELL_EMBED_SLATE,
    fp8: 0,
    phase: 0,
    bayId: 0,
    cellScale: 0.22,
    highlight: embedActive ? 1 : 0,
  });
  pushBoundsPoint(boundsPoints, preludeX, EMBED_BOOKEND_Y, 0, embedEx, embedEy, embedEz);
  lines.addBoxEdges(
    preludeX,
    EMBED_BOOKEND_Y,
    0,
    embedEx,
    embedEy,
    embedEz,
    ownerCounter++,
    embedActive ? 1.55 : 1.05,
    embedActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    embedActive ? 0.94 : 0.74,
  );
  pickBoxes.push(
    packPickBoxFromAabb(
      preludeX,
      EMBED_BOOKEND_Y,
      0,
      embedEx,
      embedEy,
      embedEz,
      encodeDoryPickMetadata({
        physicalLayerIdx: null,
        bayId: 0,
        stage: 'bookend',
        loopIteration: 0,
        partId: 'bookend_embeddings',
        tensorKey: 'embeddings',
        modelWeightId,
        extraTags: { untied: true, unitNorm: true },
      }),
    ),
  );
  landmarks.push({
    id: 'lm-dory-embeddings',
    label: 'backbone.embeddings [64, 8]',
    sublabel: 'Untied · unit-norm rows · no Layer-0 pre-norm (DR5)',
    world: [preludeX - embedEx * 0.5 - 0.55, EMBED_BOOKEND_Y, 0],
    bayId: 0,
    partId: 'bookend_embeddings',
    align: 'right',
    tier: 'primary',
  });

  // Downward arrow: embeddings -> Prelude p=0
  const embedToPreludeMidY = 0.5 * (EMBED_BOOKEND_Y - embedEy * 0.5 + (PRELUDE_TOP_Y + fullAttnEy * 0.5));
  const embedToPreludeSpanY = Math.max(
    0.35,
    EMBED_BOOKEND_Y - embedEy * 0.5 - (PRELUDE_TOP_Y + fullAttnEy * 0.5) - 0.16,
  );
  ribbons.push({
    cx: preludeX,
    cy: embedToPreludeMidY,
    cz: 0,
    ex: 0.36,
    ey: embedToPreludeSpanY,
    ez: 0.24,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: 0.54,
  });

  // ==========================================================================
  // 2. STAGE 1 — INPUT PRELUDE (`p = 0..15`, 8 Pairs = 6 SWA + 2 Full + 8 MLP)
  // ==========================================================================
  for (let p = 0; p < N_INPUT_LAYERS; p += 1) {
    const spec = DORY_PHYSICAL_LAYERS[p]!;
    emitSublayerBlock(spec, preludeX, 0, p, true);
  }
  // Subtle stage framing envelope around Input Prelude (0..15)
  lines.addDashedFrameXZ(
    preludeX - colHalfW - 0.32,
    -fullAttnEz * 0.5 - 0.24,
    preludeX + colHalfW + 0.32,
    fullAttnEz * 0.5 + 0.24,
    PRELUDE_TOP_Y + fullAttnEy * 0.5 + 0.18,
    LINE_STYLE_GUIDE,
    0.42,
  );
  landmarks.push({
    id: 'lm-dory-prelude',
    label: 'Input Prelude (0..15)',
    sublabel: '16 sublayers (Pair L0–L7) · 6 SWA + 2 Full + 8 MLP · 1x',
    world: [preludeX - colHalfW - 0.65, 0.5 * (PRELUDE_TOP_Y + PRELUDE_BOT_Y), 0],
    bayId: 1,
    partId: isUnrolled ? 'exec_e0_p0_t0' : 'phys_l0',
    align: 'right',
    tier: 'primary',
  });

  // ==========================================================================
  // 3. PERSISTENT INPUT ANCHOR `x = s_0` LIVE SHEET + ANCHOR RAIL
  // ==========================================================================
  const anchorNodeEx = 1.48;
  const anchorNodeEy = 0.34;
  const anchorNodeEz = 0.52;
  const anchorActive =
    hoveredPartId === 'anchor_x_s0' || hoveredCell?.tensorKey === 'x_anchor';

  solids.push({
    cx: preludeX,
    cy: ANCHOR_SHEET_Y,
    cz: -0.22,
    ex: anchorNodeEx,
    ey: anchorNodeEy,
    ez: anchorNodeEz,
    cellFamily: CELL_MOE_AMBER,
    fp8: 0,
    phase: 1.5,
    bayId: 16,
    cellScale: 0.18,
    highlight: anchorActive ? 1 : 2,
  });
  pushBoundsPoint(boundsPoints, preludeX, ANCHOR_SHEET_Y, -0.22, anchorNodeEx, anchorNodeEy, anchorNodeEz);
  lines.addBoxEdges(
    preludeX,
    ANCHOR_SHEET_Y,
    -0.22,
    anchorNodeEx,
    anchorNodeEy,
    anchorNodeEz,
    ownerCounter++,
    anchorActive ? 1.5 : 1.15,
    anchorActive ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
    0.92,
  );

  // Live 1x8 RG32F x_anchor sheet facing +Z right at the Prelude -> Core handoff
  const anchorSheetW = compactColumns ? 2.60 : 3.04;
  const anchorSheetH = 0.44;
  const anchorFrame = makeSheetFrame3D(
    [preludeX, ANCHOR_SHEET_Y, DORY_MACRO_LAYOUT.sheetFrontZ],
    [anchorSheetW, 0.06, anchorSheetH],
    FRAME_XY_AXES.axisU,
    FRAME_XY_AXES.axisN,
  );
  sheets.push({
    cx: anchorFrame.center[0],
    cy: anchorFrame.center[1],
    cz: anchorFrame.center[2],
    ex: anchorFrame.extent[0],
    ey: anchorFrame.extent[1],
    ez: anchorFrame.extent[2],
    gridCols: HIDDEN_SIZE,
    gridRows: 1,
    sheetKind: SHEET_KIND_STANDARD,
    role: SHEET_ROLE_RESIDUAL,
    highlight: anchorActive ? SHEET_HL_ACTIVE : SHEET_HL_BF16_GOLD,
    slice: xAnchorSlice,
    frame: anchorFrame,
  });
  pushBoundsPoint(
    boundsPoints,
    anchorFrame.center[0],
    anchorFrame.center[1],
    anchorFrame.center[2],
    anchorFrame.extent[0],
    anchorFrame.extent[2],
    0.12,
  );
  pickBoxes.push(
    packPickBoxFromFrame(
      anchorFrame,
      encodeDoryPickMetadata({
        physicalLayerIdx: 15,
        executedLayerIdx: 15,
        stage: 'input',
        loopIteration: 0,
        partId: 'anchor_x_s0',
        tensorKey: 'x_anchor',
        gridCols: HIDDEN_SIZE,
        gridRows: 1,
        absTokenPos: lastCommittedPos,
        modelWeightId,
        extraTags: { anchor: true, unitSphere: true },
      }),
    ),
  );
  placedSheets.push({
    id: 'anchor_x_s0',
    tensorKey: 'x_anchor',
    label: `Anchor x = s_0 (pos ${lastCommittedPos} "${lastTokenText}")`,
    formula: 'x = s_0 = Prelude(E[tok]) in S^7 (||x||_2 ≈ 1)',
    loopIteration: 0,
    frame: anchorFrame,
    values: xAnchorVec,
    rows: 1,
    cols: HIDDEN_SIZE,
    rowLabels: ['x = s_0'],
  });

  // Downward arrow: Prelude exit (p=15) -> Anchor x_anchor
  ribbons.push({
    cx: preludeX,
    cy: 0.5 * (PRELUDE_BOT_Y - sublayerEy * 0.5 + (ANCHOR_SHEET_Y + anchorSheetH * 0.5)),
    cz: 0,
    ex: 0.34,
    ey: Math.max(0.28, PRELUDE_BOT_Y - sublayerEy * 0.5 - (ANCHOR_SHEET_Y + anchorSheetH * 0.5) - 0.08),
    ez: 0.22,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_AMBER,
    alpha: 0.62,
  });

  // ==========================================================================
  // 4. STAGE 2 — SHARED RECURRENT CORE (`p = 16..55`, FOLDED vs. UNROLLED)
  // ==========================================================================
  const emitFullCoreEdges = !isUnrolled || kLoops <= 4;

  for (let passIdx = 0; passIdx < passCount; passIdx += 1) {
    const loopT = isUnrolled ? passIdx + 1 : activeRecurrentLoop;
    const colX = passCenterX(passIdx + 1);
    const isFocusedPass =
      !isUnrolled || selectedLoopIteration === null || selectedLoopIteration === loopT;

    // 4a. Loop Entry Junction `u_t = N(s_{t-1} + x)` + live 1x8 RG32F sheet
    const entryPartId = `loop_entry_u_t${loopT}`;
    const entryActive =
      hoveredPartId === entryPartId ||
      (hoveredCell?.tensorKey === 'u_entry' && hoveredCell.loopIteration === loopT);
    const entryW = compactColumns ? 2.52 : 2.96;
    const entryH = 0.40;
    const entryFrame = makeSheetFrame3D(
      [colX, LOOP_ENTRY_Y, DORY_MACRO_LAYOUT.sheetFrontZ],
      [entryW, 0.06, entryH],
      FRAME_XY_AXES.axisU,
      FRAME_XY_AXES.axisN,
    );

    solids.push({
      cx: colX,
      cy: LOOP_ENTRY_Y,
      cz: -0.22,
      ex: 1.28,
      ey: 0.30,
      ez: 0.46,
      cellFamily: CELL_MOE_AMBER,
      fp8: 0,
      phase: 2.0 + loopT * 0.3,
      bayId: 17,
      cellScale: 0.18,
      highlight: entryActive ? 1 : isFocusedPass ? 2 : 0,
    });
    pushBoundsPoint(boundsPoints, colX, LOOP_ENTRY_Y, -0.22, 1.28, 0.30, 0.46);
    lines.addBoxEdges(
      colX,
      LOOP_ENTRY_Y,
      -0.22,
      1.28,
      0.30,
      0.46,
      ownerCounter++,
      entryActive ? 1.45 : 1.0,
      entryActive ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
      0.86,
    );

    sheets.push({
      cx: entryFrame.center[0],
      cy: entryFrame.center[1],
      cz: entryFrame.center[2],
      ex: entryFrame.extent[0],
      ey: entryFrame.extent[1],
      ez: entryFrame.extent[2],
      gridCols: HIDDEN_SIZE,
      gridRows: 1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_ACTIVATION,
      highlight: entryActive
        ? SHEET_HL_ACTIVE
        : isFocusedPass
          ? SHEET_HL_PATHWAY
          : SHEET_HL_NORMAL,
      slice: uSlices[loopT - 1]!,
      frame: entryFrame,
    });
    pushBoundsPoint(
      boundsPoints,
      entryFrame.center[0],
      entryFrame.center[1],
      entryFrame.center[2],
      entryFrame.extent[0],
      entryFrame.extent[2],
      0.12,
    );
    pickBoxes.push(
      packPickBoxFromFrame(
        entryFrame,
        encodeDoryPickMetadata({
          physicalLayerIdx: 16,
          executedLayerIdx: computeExecutedLayerIdx(16, loopT, kLoops),
          stage: 'recurrent',
          loopIteration: loopT,
          partId: entryPartId,
          tensorKey: 'u_entry',
          gridCols: HIDDEN_SIZE,
          gridRows: 1,
          absTokenPos: lastCommittedPos,
          modelWeightId,
          extraTags: { formula: loopT === 1 ? 'N(2x)' : `N(s_${loopT - 1}+x)` },
        }),
      ),
    );
    placedSheets.push({
      id: entryPartId,
      tensorKey: 'u_entry',
      label: loopT === 1 ? 'Loop 1 Entry u_1 = N(2x)' : `Loop ${loopT} Entry u_${loopT}`,
      formula:
        loopT === 1
          ? 'u_1 = N(s_0 + x) = N(2x) (literal DR8)'
          : `u_${loopT} = N(s_${loopT - 1} + x) in S^7`,
      loopIteration: loopT,
      frame: entryFrame,
      values: uVecs[loopT - 1]!,
      rows: 1,
      cols: HIDDEN_SIZE,
      rowLabels: [`u_${loopT}`],
    });

    // Entry arrow into physical p=16
    ribbons.push({
      cx: colX,
      cy: 0.5 * (LOOP_ENTRY_Y - entryH * 0.5 + (CORE_TOP_Y + fullAttnEy * 0.5)),
      cz: 0,
      ex: 0.30,
      ey: Math.max(0.24, LOOP_ENTRY_Y - entryH * 0.5 - (CORE_TOP_Y + fullAttnEy * 0.5) - 0.06),
      ez: 0.20,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: isFocusedPass ? 0.62 : 0.38,
    });

    // 4b. All 40 Shared Recurrent Core sublayers (`p = 16..55`, exact 3:1 motif x 5)
    for (let p = N_INPUT_LAYERS; p < N_INPUT_LAYERS + N_RECURRENT_LAYERS; p += 1) {
      const spec = DORY_PHYSICAL_LAYERS[p]!;
      const execIdx = computeExecutedLayerIdx(p, loopT, kLoops);
      emitSublayerBlock(spec, colX, loopT, execIdx, emitFullCoreEdges);
    }

    // 4c. Loop Exit Post-Norm Junction `s_t = N(F_theta(u_t))` + live 1x8 RG32F sheet
    const exitPartId = `loop_exit_s_t${loopT}`;
    const exitActive =
      hoveredPartId === exitPartId ||
      (hoveredCell?.tensorKey === 's_exit' && hoveredCell.loopIteration === loopT);
    const exitFrame = makeSheetFrame3D(
      [colX, LOOP_EXIT_Y, DORY_MACRO_LAYOUT.sheetFrontZ],
      [entryW, 0.06, entryH],
      FRAME_XY_AXES.axisU,
      FRAME_XY_AXES.axisN,
    );

    solids.push({
      cx: colX,
      cy: LOOP_EXIT_Y,
      cz: -0.22,
      ex: 1.28,
      ey: 0.30,
      ez: 0.46,
      cellFamily: CELL_ACTIVATION_GREEN,
      fp8: 0,
      phase: 3.0 + loopT * 0.3,
      bayId: 56,
      cellScale: 0.18,
      highlight: exitActive ? 1 : isFocusedPass ? 2 : 0,
    });
    pushBoundsPoint(boundsPoints, colX, LOOP_EXIT_Y, -0.22, 1.28, 0.30, 0.46);
    lines.addBoxEdges(
      colX,
      LOOP_EXIT_Y,
      -0.22,
      1.28,
      0.30,
      0.46,
      ownerCounter++,
      exitActive ? 1.45 : 1.0,
      exitActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
      0.84,
    );

    sheets.push({
      cx: exitFrame.center[0],
      cy: exitFrame.center[1],
      cz: exitFrame.center[2],
      ex: exitFrame.extent[0],
      ey: exitFrame.extent[1],
      ez: exitFrame.extent[2],
      gridCols: HIDDEN_SIZE,
      gridRows: 1,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_RESIDUAL,
      highlight: exitActive
        ? SHEET_HL_ACTIVE
        : isFocusedPass
          ? SHEET_HL_PATHWAY
          : SHEET_HL_NORMAL,
      slice: sSlices[loopT - 1]!,
      frame: exitFrame,
    });
    pushBoundsPoint(
      boundsPoints,
      exitFrame.center[0],
      exitFrame.center[1],
      exitFrame.center[2],
      exitFrame.extent[0],
      exitFrame.extent[2],
      0.12,
    );
    pickBoxes.push(
      packPickBoxFromFrame(
        exitFrame,
        encodeDoryPickMetadata({
          physicalLayerIdx: 55,
          executedLayerIdx: computeExecutedLayerIdx(55, loopT, kLoops),
          stage: 'recurrent',
          loopIteration: loopT,
          partId: exitPartId,
          tensorKey: 's_exit',
          gridCols: HIDDEN_SIZE,
          gridRows: 1,
          absTokenPos: lastCommittedPos,
          modelWeightId,
          extraTags: { formula: `N(F_theta(u_${loopT}))`, unitSphere: true },
        }),
      ),
    );
    placedSheets.push({
      id: exitPartId,
      tensorKey: 's_exit',
      label: `Loop ${loopT} Exit s_${loopT} (post-norm)`,
      formula: `s_${loopT} = N(F_theta(u_${loopT})) in S^7 (literal DR8)`,
      loopIteration: loopT,
      frame: exitFrame,
      values: sVecs[loopT - 1]!,
      rows: 1,
      cols: HIDDEN_SIZE,
      rowLabels: [`s_${loopT}`],
    });

    // Exit arrow from p=55 -> s_t
    ribbons.push({
      cx: colX,
      cy: 0.5 * (CORE_BOT_Y - sublayerEy * 0.5 + (LOOP_EXIT_Y + entryH * 0.5)),
      cz: 0,
      ex: 0.30,
      ey: Math.max(0.24, CORE_BOT_Y - sublayerEy * 0.5 - (LOOP_EXIT_Y + entryH * 0.5) - 0.06),
      ez: 0.20,
      kind: RIBBON_KIND_ARROW_NEG_Y,
      colorId: RIBBON_COLOR_EMERALD,
      alpha: isFocusedPass ? 0.62 : 0.38,
    });

    // Unrolled per-pass landmark
    if (isUnrolled) {
      const binding = sharedRecurrentWeightBindings[passIdx]!;
      landmarks.push({
        id: `lm-dory-pass-t${loopT}`,
        label: `Loop t=${loopT} · Exec ${binding.executedLayerStart}..${binding.executedLayerEnd}`,
        sublabel: `Phys 16..55 · Shared ID ${shortWeightId} · KV [${binding.kvBankRangeStart}..${binding.kvBankRangeEnd}]`,
        world: [colX, LOOP_ENTRY_Y + 0.68, 0.35],
        bayId: 17,
        partId: entryPartId,
        tone: isFocusedPass ? 'cyan' : 'default',
        align: 'bottom',
        tier: loopT === 1 || loopT === kLoops || isFocusedPass ? 'primary' : 'detail',
      });
    }
  }

  // ==========================================================================
  // 4d. MODE-SPECIFIC FLOW TOPOLOGY: FOLDED FEEDBACK ARC vs. UNROLLED SHARED RAILS
  // ==========================================================================
  if (!isUnrolled) {
    // --- FOLDED MODE ---
    const anchorRailX = DORY_MACRO_LAYOUT.anchorRailX;
    const feedbackX = DORY_MACRO_LAYOUT.feedbackArcX;
    const feedbackActive = kLoops >= 2;

    // 1. Golden Persistent Anchor Rail (`x = s_0` feeding `+ / N` at LOOP_ENTRY_Y)
    flows.addPolyline3D(
      [
        [preludeX - anchorSheetW * 0.5, ANCHOR_SHEET_Y, 0.10],
        [anchorRailX, ANCHOR_SHEET_Y, 0.10],
        [anchorRailX, LOOP_ENTRY_Y, 0.10],
        [preludeX - 1.48, LOOP_ENTRY_Y, 0.10],
      ],
      [0, 0, 1],
      0.18,
      DORY_FLOW_COLORS.anchorGold,
      true,
      0.26,
      0.26,
    );
    pushBoundsPoint(boundsPoints, anchorRailX, 0.5 * (ANCHOR_SHEET_Y + LOOP_ENTRY_Y), 0.10, 0.4, 2.0, 0.4);

    // 2. Sweeping `-X` Recurrent Feedback Arc (`s_t -> +`) from LOOP_EXIT_Y back up to LOOP_ENTRY_Y
    const arcColor = feedbackActive
      ? DORY_FLOW_COLORS.recurrentTeal
      : DORY_FLOW_COLORS.ghostSlate;

    flows.addCubic3D(
      [preludeX - 1.52, LOOP_EXIT_Y, 0.08],
      [feedbackX + 0.4, LOOP_EXIT_Y - 0.4, 0.08],
      [feedbackX, LOOP_EXIT_Y + 2.2, 0.08],
      [feedbackX, CORE_BOT_Y + 1.0, 0.08],
      [0, 0, 1],
      feedbackActive ? 0.24 : 0.14,
      arcColor,
      false,
      16,
    );
    flows.addPolyline3D(
      [
        [feedbackX, CORE_BOT_Y + 1.0, 0.08],
        [feedbackX, CORE_TOP_Y - 1.0, 0.08],
      ],
      [0, 0, 1],
      feedbackActive ? 0.24 : 0.14,
      arcColor,
      false,
    );
    flows.addCubic3D(
      [feedbackX, CORE_TOP_Y - 1.0, 0.08],
      [feedbackX, LOOP_ENTRY_Y - 1.6, 0.08],
      [feedbackX + 0.4, LOOP_ENTRY_Y + 0.35, 0.08],
      [preludeX - 1.56, LOOP_ENTRY_Y + 0.05, 0.08],
      [0, 0, 1],
      feedbackActive ? 0.24 : 0.14,
      arcColor,
      true,
      16,
      true,
      0.30,
      0.30,
    );
    pushBoundsPoint(boundsPoints, feedbackX, 0.5 * (CORE_TOP_Y + CORE_BOT_Y), 0.08, 0.8, 20, 0.8);

    // Bold upward directional 3D arrows along the return feedback spine
    const feedbackArrowYs = [
      CORE_BOT_Y + 4.2,
      0.5 * (CORE_TOP_Y + CORE_BOT_Y),
      CORE_TOP_Y - 4.2,
    ];
    for (let i = 0; i < feedbackArrowYs.length; i += 1) {
      ribbons.push({
        cx: feedbackX,
        cy: feedbackArrowYs[i]!,
        cz: 0.14,
        ex: 0.36,
        ey: 1.05,
        ez: 0.22,
        kind: RIBBON_KIND_ARROW_POS_Y,
        colorId: feedbackActive ? RIBBON_COLOR_CYAN : RIBBON_COLOR_PERIWINKLE,
        alpha: feedbackActive ? 0.68 : 0.24,
      });
    }

    // Visual loop-pass count ticks (`t = 1..k`) along the feedback rail
    const tickSpanY = Math.min(12.0, Math.max(2.4, kLoops * 1.4));
    const tickTopY = 0.5 * (CORE_TOP_Y + CORE_BOT_Y) + tickSpanY * 0.5;
    for (let t = 1; t <= kLoops; t += 1) {
      const frac = kLoops === 1 ? 0.5 : (t - 1) / (kLoops - 1);
      const ty = tickTopY - frac * tickSpanY;
      const isActiveTick = t === activeRecurrentLoop;
      lines.addSegment(
        feedbackX - 0.55,
        ty,
        0.16,
        feedbackX + 0.55,
        ty,
        0.16,
        isActiveTick ? 1.9 : 1.15,
        isActiveTick ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
        isActiveTick ? 0.96 : 0.72,
      );
    }

    // 3. Side-Mounted Stacked `(1 + k) x 8` Live RG32F Latent Trajectory Strip (`[x; s_1..s_k]`)
    const trajX = DORY_MACRO_LAYOUT.foldedTrajectoryStripX;
    const trajMidY = 0.5 * (CORE_TOP_Y + CORE_BOT_Y);
    const trajW = 2.56;
    const trajH = Math.min(8.0, Math.max(1.4, (1 + kLoops) * 0.56));
    const trajFrame = makeSheetFrame3D(
      [trajX, trajMidY, DORY_MACRO_LAYOUT.sheetFrontZ],
      [trajW, 0.06, trajH],
      FRAME_XY_AXES.axisU,
      FRAME_XY_AXES.axisN,
    );
    const trajActive =
      hoveredPartId === 'folded_latent_trajectory' ||
      hoveredCell?.tensorKey === 'latent_trajectory';

    sheets.push({
      cx: trajFrame.center[0],
      cy: trajFrame.center[1],
      cz: trajFrame.center[2],
      ex: trajFrame.extent[0],
      ey: trajFrame.extent[1],
      ez: trajFrame.extent[2],
      gridCols: HIDDEN_SIZE,
      gridRows: 1 + kLoops,
      sheetKind: SHEET_KIND_STANDARD,
      role: SHEET_ROLE_RESIDUAL,
      highlight: trajActive ? SHEET_HL_ACTIVE : SHEET_HL_PATHWAY,
      slice: trajectorySlice,
      frame: trajFrame,
    });
    pushBoundsPoint(
      boundsPoints,
      trajFrame.center[0],
      trajFrame.center[1],
      trajFrame.center[2],
      trajFrame.extent[0],
      trajFrame.extent[2],
      0.20,
    );

    const trajRowLabels: string[] = ['x (s_0)'];
    for (let t = 1; t <= kLoops; t += 1) {
      trajRowLabels.push(`s_${t}`);
    }
    pickBoxes.push(
      packPickBoxFromFrame(
        trajFrame,
        encodeDoryPickMetadata({
          physicalLayerIdx: 55,
          stage: 'recurrent',
          loopIteration: activeRecurrentLoop,
          partId: 'folded_latent_trajectory',
          tensorKey: 'latent_trajectory',
          gridCols: HIDDEN_SIZE,
          gridRows: 1 + kLoops,
          absTokenPos: lastCommittedPos,
          modelWeightId,
          extraTags: { rows: 1 + kLoops, distinctActivations: true },
        }),
      ),
    );
    placedSheets.push({
      id: 'folded_latent_trajectory',
      tensorKey: 'latent_trajectory',
      label: `Distinct Per-Loop States [x; s_1..s_${kLoops}]`,
      formula: 'Row 0 = x_anchor, Row t = s_t = N(F_theta(u_t)) (distinct states, shared weights)',
      loopIteration: activeRecurrentLoop,
      frame: trajFrame,
      values: trajectoryMatrix,
      rows: 1 + kLoops,
      cols: HIDDEN_SIZE,
      rowLabels: trajRowLabels,
    });

    // Subtle leader connecting feedback arc to stacked trajectory strip
    lines.addSegment(
      trajX + trajW * 0.5 + 0.12,
      trajMidY,
      0.16,
      feedbackX - 0.18,
      trajMidY,
      0.16,
      1.05,
      LINE_STYLE_GUIDE,
      0.62,
    );

    landmarks.push(
      {
        id: 'lm-dory-core-folded',
        label: `Shared Recurrent Core (16..55 · ×${kLoops})`,
        sublabel: `40 physical sublayers (${40 * kLoops} exec) · Weight ID ${shortWeightId} · 0 loop adapters`,
        world: [feedbackX - 0.45, CORE_TOP_Y - 1.6, 0.20],
        bayId: 17,
        partId: 'phys_l16',
        tone: 'cyan',
        align: 'right',
        tier: 'primary',
      },
      {
        id: 'lm-dory-feedback-arc',
        label:
          kLoops >= 2
            ? `Latent Feedback s_t -> u_{t+1} = N(s_t + x) (${kLoops - 1} returns)`
            : 'Single Pass k=1 (u_1 = N(2x) · 0 returns)',
        sublabel: 'Distinct unit-sphere states s_1..s_k through identical physical weights',
        world: [trajX, trajMidY + trajH * 0.5 + 0.65, 0.30],
        bayId: 56,
        partId: 'folded_latent_trajectory',
        tone: kLoops >= 2 ? 'gold' : 'muted',
        align: 'bottom',
        tier: 'primary',
      },
    );
  } else {
    // --- UNROLLED MODE (`t = 1..k`) ---
    const provZ = DORY_MACRO_LAYOUT.weightProvenanceRailZ;
    const firstPassX = passCenterX(1);
    const lastPassX = passCenterX(kLoops);

    // 1. Golden Persistent Anchor Bus (`x = s_0` feeding EVERY pass's `u_t = N(s_{t-1} + x)`)
    const anchorBusY = 0.5 * (ANCHOR_SHEET_Y + LOOP_ENTRY_Y) + 0.12;
    if (kLoops >= 2) {
      flows.addPolyline3D(
        [
          [firstPassX, ANCHOR_SHEET_Y - anchorSheetH * 0.5, 0.14],
          [firstPassX, anchorBusY, 0.14],
          [lastPassX, anchorBusY, 0.14],
        ],
        [0, 0, 1],
        0.18,
        DORY_FLOW_COLORS.anchorGold,
        false,
      );
      for (let t = 1; t <= kLoops; t += 1) {
        const tx = passCenterX(t);
        flows.addPolyline3D(
          [
            [tx, anchorBusY, 0.14],
            [tx, LOOP_ENTRY_Y + 0.24, 0.14],
          ],
          [0, 0, 1],
          0.14,
          DORY_FLOW_COLORS.anchorGold,
          true,
          0.22,
          0.22,
        );
      }
    } else {
      flows.addPolyline3D(
        [
          [firstPassX, ANCHOR_SHEET_Y - anchorSheetH * 0.5, 0.14],
          [firstPassX, LOOP_ENTRY_Y + 0.24, 0.14],
        ],
        [0, 0, 1],
        0.18,
        DORY_FLOW_COLORS.anchorGold,
        true,
        0.24,
        0.24,
      );
    }

    // 2. Inter-Loop State Transfer Ribbons (`s_t` bottom of Pass t -> `u_{t+1}` top of Pass t+1)
    for (let t = 1; t < kLoops; t += 1) {
      const xFrom = passCenterX(t);
      const xTo = passCenterX(t + 1);
      const corridorX = 0.5 * (xFrom + xTo);
      flows.addPolyline3D(
        [
          [xFrom + colHalfW * 0.55, LOOP_EXIT_Y, 0.14],
          [corridorX, LOOP_EXIT_Y, 0.14],
          [corridorX, LOOP_ENTRY_Y, 0.14],
          [xTo - colHalfW * 0.55, LOOP_ENTRY_Y, 0.14],
        ],
        [0, 0, 1],
        0.17,
        DORY_FLOW_COLORS.recurrentTeal,
        true,
        0.24,
        0.24,
      );
      ribbons.push({
        cx: corridorX,
        cy: 0.5 * (CORE_TOP_Y + CORE_BOT_Y),
        cz: 0.18,
        ex: 0.26,
        ey: 0.96,
        ez: 0.18,
        kind: RIBBON_KIND_ARROW_POS_Y,
        colorId: RIBBON_COLOR_CYAN,
        alpha: 0.56,
      });
    }

    // 3. Horizontal Shared Weight-ID Provenance Rails linking identical physical layers `16..55`
    if (kLoops >= 2) {
      const railLayers = new Set<number>([16, 22, 30, 38, 46, 54, 55]);
      if (
        selectedPhysicalLayerIdx !== null &&
        selectedPhysicalLayerIdx >= N_INPUT_LAYERS &&
        selectedPhysicalLayerIdx < N_INPUT_LAYERS + N_RECURRENT_LAYERS
      ) {
        railLayers.add(selectedPhysicalLayerIdx);
      }
      const sortedRailLayers = Array.from(railLayers).sort((a, b) => a - b);
      for (let i = 0; i < sortedRailLayers.length; i += 1) {
        const p = sortedRailLayers[i]!;
        const y = doryMacroPhysicalLayerY(p);
        const isSel = p === selectedPhysicalLayerIdx;
        lines.addSegment(
          firstPassX - colHalfW - 0.18,
          y,
          provZ,
          lastPassX + colHalfW + 0.18,
          y,
          provZ,
          isSel ? 1.85 : 1.15,
          isSel ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
          isSel ? 0.96 : 0.68,
        );
      }
      pushBoundsPoint(
        boundsPoints,
        0.5 * (firstPassX + lastPassX),
        0.5 * (CORE_TOP_Y + CORE_BOT_Y),
        provZ,
        lastPassX - firstPassX + colHalfW * 2,
        CORE_TOP_Y - CORE_BOT_Y,
        0.2,
      );
    }
  }

  // ==========================================================================
  // 5. STAGE 3 — OUTPUT CODA (`p = 56..71`, 8 Pairs = 6 SWA + 2 Full + 8 MLP)
  // ==========================================================================
  // Exit arrow from final loop `s_k` straight into Coda `p = 56` (NO outer skip!)
  ribbons.push({
    cx: codaX,
    cy: 0.5 * (LOOP_EXIT_Y - 0.22 + (CODA_TOP_Y + fullAttnEy * 0.5)),
    cz: 0,
    ex: 0.36,
    ey: Math.max(0.32, LOOP_EXIT_Y - 0.22 - (CODA_TOP_Y + fullAttnEy * 0.5) - 0.12),
    ez: 0.24,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_AMBER,
    alpha: 0.60,
  });

  for (let p = N_INPUT_LAYERS + N_RECURRENT_LAYERS; p < NUM_PHYSICAL_LAYERS; p += 1) {
    const spec = DORY_PHYSICAL_LAYERS[p]!;
    const execIdx = computeExecutedLayerIdx(p, 0, kLoops);
    emitSublayerBlock(spec, codaX, 0, execIdx, true);
  }
  lines.addDashedFrameXZ(
    codaX - colHalfW - 0.32,
    -fullAttnEz * 0.5 - 0.24,
    codaX + colHalfW + 0.32,
    fullAttnEz * 0.5 + 0.24,
    CODA_TOP_Y + fullAttnEy * 0.5 + 0.18,
    LINE_STYLE_GUIDE,
    0.42,
  );
  landmarks.push({
    id: 'lm-dory-coda',
    label: 'Output Coda (56..71)',
    sublabel: '16 sublayers (Pair L28–L35) · 6 SWA + 2 Full + 8 MLP · No outer skip',
    world: [codaX - colHalfW - 0.65, 0.5 * (CODA_TOP_Y + CODA_BOT_Y), 0],
    bayId: 57,
    partId: isUnrolled
      ? `exec_e${computeExecutedLayerIdx(56, 0, kLoops)}_p56_t0`
      : 'phys_l56',
    tone: 'gold',
    align: 'right',
    tier: 'primary',
  });

  // ==========================================================================
  // 6. BOTTOM BOOKENDS: `lm_head.weight [64, 8]` + FP32 `logit_scale / 0.0177`
  // ==========================================================================
  const lmHeadActive =
    hoveredPartId === 'bookend_lm_head' || hoveredCell?.partId === 'bookend_lm_head';
  const logitScaleActive =
    hoveredPartId === 'bookend_logit_scale' || hoveredCell?.partId === 'bookend_logit_scale';

  // Arrow Coda exit (p=71) -> lm_head
  ribbons.push({
    cx: codaX,
    cy: 0.5 * (CODA_BOT_Y - sublayerEy * 0.5 + (LM_HEAD_Y + embedEy * 0.5)),
    cz: 0,
    ex: 0.36,
    ey: Math.max(0.32, CODA_BOT_Y - sublayerEy * 0.5 - (LM_HEAD_Y + embedEy * 0.5) - 0.12),
    ez: 0.24,
    kind: RIBBON_KIND_ARROW_NEG_Y,
    colorId: RIBBON_COLOR_PERIWINKLE,
    alpha: 0.54,
  });

  // Untied lm_head.weight
  solids.push({
    cx: codaX,
    cy: LM_HEAD_Y,
    cz: 0,
    ex: embedEx,
    ey: embedEy,
    ez: embedEz,
    cellFamily: CELL_EMBED_SLATE,
    fp8: 0,
    phase: 2.5,
    bayId: 73,
    cellScale: 0.22,
    highlight: lmHeadActive ? 1 : 0,
  });
  pushBoundsPoint(boundsPoints, codaX, LM_HEAD_Y, 0, embedEx, embedEy, embedEz);
  lines.addBoxEdges(
    codaX,
    LM_HEAD_Y,
    0,
    embedEx,
    embedEy,
    embedEz,
    ownerCounter++,
    lmHeadActive ? 1.55 : 1.05,
    lmHeadActive ? LINE_STYLE_ACTIVE : LINE_STYLE_NEUTRAL,
    lmHeadActive ? 0.94 : 0.74,
  );
  pickBoxes.push(
    packPickBoxFromAabb(
      codaX,
      LM_HEAD_Y,
      0,
      embedEx,
      embedEy,
      embedEz,
      encodeDoryPickMetadata({
        physicalLayerIdx: null,
        bayId: 73,
        stage: 'bookend',
        loopIteration: 0,
        partId: 'bookend_lm_head',
        tensorKey: 'lm_head',
        modelWeightId,
        extraTags: { untied: true },
      }),
    ),
  );

  // FP32 lm_head.logit_scale / 0.0177 strip
  const scaleEx = embedEx * 0.92;
  const scaleEy = 0.44;
  const scaleEz = 0.86;
  solids.push({
    cx: codaX,
    cy: LOGIT_SCALE_Y,
    cz: 0,
    ex: scaleEx,
    ey: scaleEy,
    ez: scaleEz,
    cellFamily: CELL_MOE_AMBER,
    fp8: 0,
    phase: 4.0,
    bayId: 74,
    cellScale: 0.19,
    highlight: logitScaleActive ? 1 : 2,
  });
  pushBoundsPoint(boundsPoints, codaX, LOGIT_SCALE_Y, 0, scaleEx, scaleEy, scaleEz);
  lines.addBoxEdges(
    codaX,
    LOGIT_SCALE_Y,
    0,
    scaleEx,
    scaleEy,
    scaleEz,
    ownerCounter++,
    logitScaleActive ? 1.5 : 1.05,
    logitScaleActive ? LINE_STYLE_ACTIVE : LINE_STYLE_BF16_GOLD,
    0.88,
  );
  pickBoxes.push(
    packPickBoxFromAabb(
      codaX,
      LOGIT_SCALE_Y,
      0,
      scaleEx,
      scaleEy,
      scaleEz,
      encodeDoryPickMetadata({
        physicalLayerIdx: null,
        bayId: 74,
        stage: 'bookend',
        loopIteration: 0,
        partId: 'bookend_logit_scale',
        tensorKey: 'logit_scale',
        modelWeightId,
        extraTags: { fp32Scale: true, divisor: '0.0177' },
      }),
    ),
  );
  landmarks.push({
    id: 'lm-dory-lm-head',
    label: 'lm_head [64, 8] + FP32 logit_scale / 0.0177',
    sublabel: 'Untied raw logits -> FP32 scaled logits -> binary64 softmax',
    world: [codaX - embedEx * 0.5 - 0.55, LM_HEAD_Y - 0.45, 0],
    bayId: 73,
    partId: 'bookend_lm_head',
    align: 'right',
    tier: 'primary',
  });

  // ==========================================================================
  // 7. PARALLEL DUAL-AXIS KV-BANK RAIL (`36..176` BANKS / `104` CONTRACT A)
  // ==========================================================================
  const emitKvBankBlock = (
    spec: DoryPhysicalLayerSpec,
    bankSlotId: number,
    bankKind: DoryKvBankKind,
    cx: number,
    loopForMeta: number,
    isOverwriteBank: boolean,
  ): void => {
    const isFull = spec.sublayerKind === 'full_attention';
    const isShallow = bankKind === 'output_shallow';
    const cy = doryMacroPhysicalLayerY(spec.physicalLayerIdx);
    const cz = 0;
    const ex = isFull ? kvFullEx : kvSwaEx;
    const ey = isFull ? kvFullEy : kvSwaEy;
    const ez = isFull ? kvFullEz : kvSwaEz;
    const partId = `kv_bank_${bankSlotId}`;

    const isSelectedBank =
      hoveredPartId === partId ||
      hoveredCell?.kvBankSlotId === bankSlotId ||
      (selectedPhysicalLayerIdx === spec.physicalLayerIdx &&
        (bankKind !== 'recurrent' ||
          isOverwriteBank ||
          selectedLoopIteration === null ||
          selectedLoopIteration === 0 ||
          selectedLoopIteration === loopForMeta));

    const cellFamily = isShallow
      ? CELL_MOE_AMBER
      : isFull
        ? CELL_WEIGHT_BLUE
        : CELL_SSM_CYAN;

    solids.push({
      cx,
      cy,
      cz,
      ex,
      ey,
      ez,
      cellFamily,
      fp8: 0,
      phase: bankSlotId * 0.11 + (isShallow ? 3.5 : 0),
      bayId: spec.physicalLayerIdx + 1,
      cellScale: 0.17,
      highlight: isSelectedBank ? 1 : isOverwriteBank || isShallow ? 2 : 0,
    });
    pushBoundsPoint(boundsPoints, cx, cy, cz, ex, ey, ez);

    // Keep edge rendering ultra-crisp across k=1..8 without dark ink clutter
    if (isSelectedBank || isFull || isOverwriteBank || isShallow || kvColCount <= 4) {
      lines.addBoxEdges(
        cx,
        cy,
        cz,
        ex,
        ey,
        ez,
        ownerCounter++,
        isSelectedBank ? 1.45 : isOverwriteBank || isShallow ? 1.15 : isFull ? 0.95 : 0.72,
        isSelectedBank
          ? LINE_STYLE_ACTIVE
          : isOverwriteBank || isShallow
            ? LINE_STYLE_BF16_GOLD
            : LINE_STYLE_NEUTRAL,
        isSelectedBank ? 0.95 : isOverwriteBank || isShallow ? 0.88 : 0.60,
      );
    }

    const cacheRole = resolveBankCacheRole(
      executionMode,
      bankKind,
      bankKind === 'recurrent' ? (isOverwriteBank ? 1 : loopForMeta) : 0,
    );

    pickBoxes.push(
      packPickBoxFromAabb(
        cx,
        cy,
        cz,
        ex,
        ey,
        ez,
        encodeDoryPickMetadata({
          physicalLayerIdx: spec.physicalLayerIdx,
          stage: spec.stage,
          loopIteration: loopForMeta,
          partId,
          tensorKey: isFull ? 'kv_bank_full' : 'kv_bank_swa',
          kvBankSlotId: bankSlotId,
          kvBankKind: bankKind,
          kvPolicy: kvCacheMode,
          cacheRole,
          modelWeightId,
          groupRole: 'bank',
          tapRole: 'loop',
          extraTags: {
            attn: spec.attentionKind ?? 'none',
            swa: !isFull,
            overwrite: isOverwriteBank,
          },
        }),
      ),
    );
  };

  // Collect physical attention specs per stage in canonical ascending order
  const inputAttnSpecs: DoryPhysicalLayerSpec[] = [];
  const recurrentAttnSpecs: DoryPhysicalLayerSpec[] = [];
  const outputAttnSpecs: DoryPhysicalLayerSpec[] = [];
  for (let p = 0; p < NUM_PHYSICAL_LAYERS; p += 1) {
    const spec = DORY_PHYSICAL_LAYERS[p]!;
    if (spec.attentionKind === null) continue;
    if (spec.stage === 'input') inputAttnSpecs.push(spec);
    else if (spec.stage === 'recurrent') recurrentAttnSpecs.push(spec);
    else outputAttnSpecs.push(spec);
  }

  let nextBankSlotId = 0;

  // 7a. Input Prelude KV Banks (`0..7`: 6 SWA + 2 Full)
  for (let i = 0; i < inputAttnSpecs.length; i += 1) {
    emitKvBankBlock(inputAttnSpecs[i]!, nextBankSlotId, 'input', kvColumnX(0), 0, false);
    nextBankSlotId += 1;
  }

  // 7b. Recurrent Core KV Banks (`per_loop`: 20*k banks across k columns; `last_loop`: 20 shared overwrite banks)
  if (kvCacheMode === 'per_loop') {
    for (let t = 1; t <= kLoops; t += 1) {
      const bx = kvColumnX(t - 1);
      for (let i = 0; i < recurrentAttnSpecs.length; i += 1) {
        emitKvBankBlock(recurrentAttnSpecs[i]!, nextBankSlotId, 'recurrent', bx, t, false);
        nextBankSlotId += 1;
      }
      // Loop column tick at top of recurrent bank column
      kvDepthTicks.push({
        token: t,
        world: [bx, CORE_TOP_Y + 0.85, 0.25],
      });
    }
  } else {
    // `last_loop`: ONE shared overwrite column (banks 8..27, writerLoop = kLoops)
    const bx = kvColumnX(0);
    for (let i = 0; i < recurrentAttnSpecs.length; i += 1) {
      emitKvBankBlock(recurrentAttnSpecs[i]!, nextBankSlotId, 'recurrent', bx, kLoops, true);
      nextBankSlotId += 1;
    }
    kvDepthTicks.push({
      token: kLoops,
      world: [bx, CORE_TOP_Y + 0.85, 0.25],
    });

    // Gold overwrite bracket around the single shared recurrent KV column
    const owTopY = doryMacroPhysicalLayerY(16) + kvFullEy * 0.5 + 0.28;
    const owBotY = doryMacroPhysicalLayerY(54) - kvFullEy * 0.5 - 0.28;
    const owLeftX = bx - kvFullEx * 0.5 - 0.20;
    const owRightX = bx + kvFullEx * 0.5 + 0.20;
    lines.addSegment(owLeftX, owTopY, 0.68, owRightX, owTopY, 0.68, 1.5, LINE_STYLE_BF16_GOLD, 0.94);
    lines.addSegment(owRightX, owTopY, 0.68, owRightX, owBotY, 0.68, 1.5, LINE_STYLE_BF16_GOLD, 0.94);
    lines.addSegment(owRightX, owBotY, 0.68, owLeftX, owBotY, 0.68, 1.5, LINE_STYLE_BF16_GOLD, 0.94);
    lines.addSegment(owLeftX, owBotY, 0.68, owLeftX, owTopY, 0.68, 1.5, LINE_STYLE_BF16_GOLD, 0.94);

    // Ghosted dashed rectangles showing the `k - 1` avoided recurrent columns vs `per_loop`
    for (let ghostCol = 1; ghostCol < kLoops; ghostCol += 1) {
      const gx = kvColumnX(ghostCol);
      const gx0 = gx - kvSwaEx * 0.45;
      const gx1 = gx + kvSwaEx * 0.45;
      lines.addSegment(gx0, owTopY, 0, gx1, owTopY, 0, 0.9, LINE_STYLE_GUIDE, 0.34);
      lines.addSegment(gx1, owTopY, 0, gx1, owBotY, 0, 0.9, LINE_STYLE_GUIDE, 0.34);
      lines.addSegment(gx1, owBotY, 0, gx0, owBotY, 0, 0.9, LINE_STYLE_GUIDE, 0.34);
      lines.addSegment(gx0, owBotY, 0, gx0, owTopY, 0, 0.9, LINE_STYLE_GUIDE, 0.34);
      pushBoundsPoint(boundsPoints, gx, 0.5 * (owTopY + owBotY), 0, kvSwaEx, owTopY - owBotY, 0.2);
    }
  }

  // 7c. Output Deep Coda KV Banks (`output_deep`: 8 banks = 6 SWA + 2 Full)
  for (let i = 0; i < outputAttnSpecs.length; i += 1) {
    emitKvBankBlock(outputAttnSpecs[i]!, nextBankSlotId, 'output_deep', kvColumnX(0), 0, false);
    nextBankSlotId += 1;
  }

  // 7d. Optional Contract A Persistent Shallow Coda KV Rail (`output_shallow`: +8 banks -> 104 total)
  if (includeShallowCoda) {
    for (let i = 0; i < outputAttnSpecs.length; i += 1) {
      emitKvBankBlock(
        outputAttnSpecs[i]!,
        nextBankSlotId,
        'output_shallow',
        shallowCodaBankX,
        0,
        false,
      );
      nextBankSlotId += 1;
    }
    const scTopY = doryMacroPhysicalLayerY(56) + kvFullEy * 0.5 + 0.24;
    const scBotY = doryMacroPhysicalLayerY(70) - kvFullEy * 0.5 - 0.24;
    const scX0 = shallowCodaBankX - kvFullEx * 0.5 - 0.16;
    const scX1 = shallowCodaBankX + kvFullEx * 0.5 + 0.16;
    lines.addSegment(scX0, scTopY, 0.68, scX1, scTopY, 0.68, 1.5, LINE_STYLE_ACTIVE, 0.92);
    lines.addSegment(scX1, scTopY, 0.68, scX1, scBotY, 0.68, 1.5, LINE_STYLE_ACTIVE, 0.92);
    lines.addSegment(scX1, scBotY, 0.68, scX0, scBotY, 0.68, 1.5, LINE_STYLE_ACTIVE, 0.92);
    lines.addSegment(scX0, scBotY, 0.68, scX0, scTopY, 0.68, 1.5, LINE_STYLE_ACTIVE, 0.92);

    // Connector from Loop-1 Recurrent KV exit to Shallow Coda KV rail (k=1 draft path)
    flows.addPolyline3D(
      [
        [kvColumnX(0), CORE_BOT_Y - 0.6, 0.52],
        [shallowCodaBankX, CORE_BOT_Y - 0.6, 0.52],
        [shallowCodaBankX, scTopY + 0.08, 0.52],
      ],
      [0, 0, 1],
      0.14,
      DORY_FLOW_COLORS.specOxide,
      true,
      0.22,
      0.22,
    );

    landmarks.push({
      id: 'lm-dory-shallow-coda-rail',
      label: `output_shallow Coda KV (+8 banks -> ${kvRail.totalBanks} total)`,
      sublabel: DORY_SPEC_CONTRACT_A_BADGE,
      world: [scX1 + 0.55, 0.5 * (scTopY + scBotY), 0.35],
      bayId: 57,
      partId: `kv_bank_${kvRail.totalBanks - 8}`,
      tone: 'gold',
      align: 'left',
      tier: 'primary',
    });
  }

  // Horizontal bridge ribbons connecting Backbone Spine -> Parallel KV-Bank Rail
  const bridgeFromPreludeX = preludeX + colHalfW + 0.14;
  const bridgeToKvX = kvColumnX(0) - kvFullEx * 0.5 - 0.14;
  const bridgeFromCoreX = rightmostSpineX + 0.14;
  const bridgeFromCodaX = codaX + colHalfW + 0.14;

  // Prelude -> Input KV banks
  flows.addPolyline3D(
    [
      [bridgeFromPreludeX, 0.5 * (PRELUDE_TOP_Y + PRELUDE_BOT_Y), 0],
      [bridgeToKvX, 0.5 * (PRELUDE_TOP_Y + PRELUDE_BOT_Y), 0],
    ],
    [0, 0, 1],
    0.16,
    DORY_FLOW_COLORS.preludeIndigo,
    true,
    0.24,
    0.22,
  );
  // Recurrent Core -> Recurrent KV banks
  flows.addPolyline3D(
    [
      [bridgeFromCoreX, 0.5 * (CORE_TOP_Y + CORE_BOT_Y), 0],
      [bridgeToKvX, 0.5 * (CORE_TOP_Y + CORE_BOT_Y), 0],
    ],
    [0, 0, 1],
    0.18,
    kvCacheMode === 'per_loop' ? DORY_FLOW_COLORS.swaCyan : DORY_FLOW_COLORS.anchorGold,
    true,
    0.26,
    0.24,
  );
  // Output Coda -> Output Deep Coda KV banks
  flows.addPolyline3D(
    [
      [bridgeFromCodaX, 0.5 * (CODA_TOP_Y + CODA_BOT_Y), 0],
      [bridgeToKvX, 0.5 * (CODA_TOP_Y + CODA_BOT_Y), 0],
    ],
    [0, 0, 1],
    0.16,
    DORY_FLOW_COLORS.codaAmber,
    true,
    0.24,
    0.22,
  );

  // Subtle vertical collector rail behind the first KV column
  lines.addSegment(
    kvColumnX(0) - kvFullEx * 0.5 - 0.28,
    PRELUDE_TOP_Y + 0.5,
    0,
    kvColumnX(0) - kvFullEx * 0.5 - 0.28,
    CODA_BOT_Y - 0.5,
    0,
    1.0,
    LINE_STYLE_GUIDE,
    0.45,
  );

  // KV-Bank Rail top landmark
  const kvMidX = 0.5 * (kvColumnX(0) + kvRailRightEdgeX);
  landmarks.push({
    id: 'lm-dory-kv-rail',
    label: `KV-Bank Rail (${kvRail.totalBanks} banks · ${kvRail.totalSwaBanks} SWA + ${kvRail.totalFullBanks} Full)`,
    sublabel:
      kvCacheMode === 'per_loop'
        ? includeShallowCoda
          ? `per_loop k=${kLoops} + 8 output_shallow (Contract A) · ${(kvMemorySummary.kvPayloadBytes / 1024).toFixed(0)} KiB payload`
          : `per_loop k=${kLoops} (${kvRail.recurrentBanks} rec banks) · ${(kvMemorySummary.kvPayloadBytes / 1024).toFixed(0)} KiB payload`
        : `last_loop flat (${kvRail.totalBanks} banks at any k · overwrite t=k · DR17)`,
    world: [kvMidX, PRELUDE_TOP_Y + 1.55, 0.35],
    bayId: 1,
    partId: 'kv_bank_0',
    tone: kvCacheMode === 'per_loop' ? 'cyan' : 'gold',
    align: 'bottom',
    tier: 'primary',
  });

  // Selected physical layer landmark
  if (selectedSpec !== null) {
    const selColX =
      selectedSpec.stage === 'input'
        ? preludeX
        : selectedSpec.stage === 'output'
          ? codaX
          : passCenterX(isUnrolled ? activeRecurrentLoop : 1);
    const selY = doryMacroPhysicalLayerY(selectedSpec.physicalLayerIdx);
    const selExecIdx = computeExecutedLayerIdx(
      selectedSpec.physicalLayerIdx,
      selectedSpec.stage === 'recurrent' ? activeRecurrentLoop : 0,
      kLoops,
    );
    landmarks.push({
      id: `lm-dory-selected-p${selectedSpec.physicalLayerIdx}`,
      label: `Phys L${selectedSpec.physicalLayerIdx} (${selectedSpec.transformersPairBadge})`,
      sublabel:
        selectedSpec.stage === 'recurrent'
          ? `Recurrent · Exec L${selExecIdx} (t=${activeRecurrentLoop}/${kLoops}) · Motif slot ${selectedSpec.motifSlot}/8`
          : `${selectedSpec.stage === 'input' ? 'Prelude' : 'Coda'} · Exec L${selExecIdx} · Motif slot ${selectedSpec.motifSlot}/8`,
      world: [selColX + colHalfW + 0.45, selY, 0.35],
      bayId: selectedSpec.physicalLayerIdx + 1,
      partId: isUnrolled
        ? `exec_e${selExecIdx}_p${selectedSpec.physicalLayerIdx}_t${selectedSpec.stage === 'recurrent' ? activeRecurrentLoop : 0}`
        : `phys_l${selectedSpec.physicalLayerIdx}`,
      tone: 'gold',
      align: 'left',
      tier: 'detail',
    });
  }

  // ==========================================================================
  // 8. OVERLAY ANCHORS, DIMENSION CALIPERS, CALLOUTS, BADGES & CELL GRIDS
  // ==========================================================================
  const ratioPerLoopToLastLoop = (4 + 5 * kLoops) / 9;

  overlayAnchors.push(
    {
      id: 'ov-macro-anchor-x',
      label: 'Anchor x = s_0',
      sublabel: 'Re-injected at every loop: u_t = N(s_{t-1} + x) (u_1 = N(2x))',
      worldPos: [preludeX, ANCHOR_SHEET_Y, 0.48],
      tone: 'gold',
      badge: 'DR8',
    },
    {
      id: 'ov-macro-weight-id',
      label: isUnrolled
        ? `Shared Weight ID across t=1..${kLoops}`
        : `Shared Recurrent Core (×${kLoops} loops)`,
      sublabel: `${shortWeightId} · 40 physical sublayers (16..55) · 0 weight clones`,
      worldPos: [
        isUnrolled ? 0.5 * (preludeX + codaX) : 0,
        0.5 * (CORE_TOP_Y + CORE_BOT_Y),
        0.48,
      ],
      tone: 'recurrent_teal',
      badge: isUnrolled ? `32+40×${kLoops}=${executedSublayerCount}L` : '72L Physical',
    },
    {
      id: 'ov-macro-kv-rail',
      label: `KV Rail: ${kvRail.totalBanks} Banks`,
      sublabel:
        kvCacheMode === 'per_loop'
          ? `${kvRail.totalSwaBanks} SWA (W=8, cap=16) + ${kvRail.totalFullBanks} Full (S=32) · ${ratioPerLoopToLastLoop.toFixed(2)}× vs last_loop`
          : `Flat 36 banks (${kvRail.totalSwaBanks} SWA + ${kvRail.totalFullBanks} Full) · overwrites t=k`,
      worldPos: [kvMidX, 0.5 * (CORE_TOP_Y + CORE_BOT_Y), 0.55],
      tone: kvCacheMode === 'per_loop' ? 'swa_cyan' : 'gold',
      badge: includeShallowCoda ? 'Contract A · 104 Banks' : kvCacheMode,
    },
  );

  if (includeShallowCoda) {
    overlayAnchors.push({
      id: 'ov-macro-contract-a',
      label: 'Shallow Coda KV (+8 banks)',
      sublabel: DORY_SPEC_CONTRACT_A_BADGE,
      worldPos: [shallowCodaBankX, 0.5 * (CODA_TOP_Y + CODA_BOT_Y), 0.55],
      tone: 'spec_oxide',
      badge: 'DR23',
    });
  }

  // Dimension calipers along outer drafting margins
  const leftCaliperX = isUnrolled
    ? preludeX - colHalfW - 2.40
    : DORY_MACRO_LAYOUT.foldedTrajectoryStripX - 2.25;
  const rightCaliperX = kvRailRightEdgeX + 1.55;

  calipers.push(
    {
      id: 'cal-dory-prelude',
      aWorld: [rightCaliperX, PRELUDE_TOP_Y + fullAttnEy * 0.5, 0],
      bWorld: [rightCaliperX, PRELUDE_BOT_Y - sublayerEy * 0.5, 0],
      label: '16L Input Prelude (0..15 · 6 SWA, 2 Full, 8 MLP)',
      minPixels: 32,
      offsetPx: 14,
    },
    {
      id: 'cal-dory-recurrent',
      aWorld: [rightCaliperX, CORE_TOP_Y + fullAttnEy * 0.5, 0],
      bWorld: [rightCaliperX, CORE_BOT_Y - sublayerEy * 0.5, 0],
      label: `40L Shared Core (16..55 · ×${kLoops} = ${40 * kLoops}L exec)`,
      minPixels: 40,
      offsetPx: 14,
    },
    {
      id: 'cal-dory-coda',
      aWorld: [rightCaliperX, CODA_TOP_Y + fullAttnEy * 0.5, 0],
      bWorld: [rightCaliperX, CODA_BOT_Y - sublayerEy * 0.5, 0],
      label: '16L Output Coda (56..71 · 6 SWA, 2 Full, 8 MLP)',
      minPixels: 32,
      offsetPx: 14,
    },
    {
      id: 'cal-dory-total-depth',
      aWorld: [leftCaliperX, PRELUDE_TOP_Y + fullAttnEy * 0.5, 0],
      bWorld: [leftCaliperX, CODA_BOT_Y - sublayerEy * 0.5, 0],
      label: `72L Physical -> ${executedSublayerCount}L Executed (${effectivePairedDepth} Paired at k=${kLoops})`,
      minPixels: 48,
      offsetPx: 16,
    },
  );

  if (isUnrolled && kLoops >= 2) {
    calipers.push({
      id: 'cal-dory-unrolled-span',
      aWorld: [preludeX - colHalfW, CORE_TOP_Y + 2.15, 0],
      bWorld: [codaX + colHalfW, CORE_TOP_Y + 2.15, 0],
      label: `Unrolled k=${kLoops} Passes · Identical Weights (${shortWeightId})`,
      minPixels: 48,
      offsetPx: 12,
    });
  }

  calipers.push({
    id: 'cal-dory-kv-rail-width',
    aWorld: [kvColumnX(0) - kvFullEx * 0.5, CODA_BOT_Y - 1.35, 0],
    bWorld: [kvRailRightEdgeX, CODA_BOT_Y - 1.35, 0],
    label: `${kvRail.totalBanks} KV Banks (${kvCacheMode}${includeShallowCoda ? ' + 8 Shallow Coda' : ''})`,
    minPixels: 36,
    offsetPx: 12,
  });

  // Provenance Badges
  const provenanceBadges: DoryProvenanceBadge[] = [
    ...ctx.provenanceBadges,
    {
      label: isUnrolled
        ? `Unrolled ${executedSublayerCount}L (${effectivePairedDepth} paired · Shared Weight ID across t=1..${kLoops})`
        : 'Folded Physical 72L (16 Prelude + 40 Shared Core + 16 Coda · 0 Loop Adapters)',
      tone: isUnrolled ? 'recurrent_teal' : 'prelude_indigo',
      discrepancyId: 'DR4',
    },
    {
      label: 'Motif [SWA, MLP, SWA, MLP, SWA, MLP, Full, MLP] × 9 (3:1 SWA:Full in all 3 stages)',
      tone: 'swa_cyan',
    },
    {
      label: `Weight ID ${shortWeightId} · 507 tensors (0 cloned, 0 norm weights, 0 biases)`,
      tone: 'gold',
    },
  ];

  // Editorial Callouts
  const callouts: DorySceneCallout[] = [
    {
      id: 'callout-macro-shared-weights',
      title: 'Shared Physical Core vs. Distinct Per-Loop Activations',
      body:
        `All k=${kLoops} recurrent passes execute the identical 40 physical sublayers ` +
        `(backbone.layers.16..55, singleton weight ID ${shortWeightId}, zero weight cloning, zero loop adapters), ` +
        `while each pass computes distinct unit-sphere states u_t = N(s_{t-1} + x) (literal u_1 = N(2x)) and s_t = N(F_theta(u_t)).`,
      tone: 'recurrent_teal',
      worldPos: [isUnrolled ? 0.5 * (preludeX + codaX) : 0, 0.5 * (CORE_TOP_Y + CORE_BOT_Y), 0],
      badge: `k=${kLoops} · ${executedSublayerCount}L exec`,
      discrepancyIds: ['DR4', 'DR8'],
    },
    {
      id: 'callout-macro-motif-3to1',
      title: 'Exact 3:1 SWA:Full Motif × 9 & S-Invariant KV Ratio',
      body:
        `Every 8-sublayer block repeats [SWA, MLP, SWA, MLP, SWA, MLP, Full, MLP] across Input (6:2), ` +
        `Recurrent (15:5), and Output (6:2). Because all 3 stages preserve 3:1 SWA:Full, ` +
        `per_loop / last_loop = (4 + 5k) / 9 = ${ratioPerLoopToLastLoop.toFixed(3)}× is strictly invariant to context length S.`,
      tone: 'swa_cyan',
      worldPos: [preludeX, 0.5 * (PRELUDE_TOP_Y + PRELUDE_BOT_Y), 0],
      badge: `(4 + 5×${kLoops})/9 = ${ratioPerLoopToLastLoop.toFixed(2)}×`,
    },
    {
      id: 'callout-macro-kv-rail',
      title: includeShallowCoda
        ? 'Contract A 104-Bank KV Rail (Reconstructed from Paper §6.2)'
        : kvCacheMode === 'per_loop'
          ? `Expanding per_loop KV Rail (${kvRail.totalBanks} Banks)`
          : 'Flat 36-Bank last_loop KV Rail (Partition-Dependent)',
      body: includeShallowCoda
        ? `${DORY_SPEC_CONTRACT_A_BADGE}. Allocates 8 Input + 80 Loops 1–4 + 8 Deep Coda + 8 persistent ` +
          `output_shallow Coda banks (${kvRail.totalBanks} banks, ${(kvMemorySummary.kvPayloadBytes / 1024).toFixed(0)} KiB mini payload) ` +
          `so k=1 drafting preserves standalone k=1 fidelity across cycles.`
        : kvCacheMode === 'per_loop'
          ? `Retains 16 + 20k = ${kvRail.totalBanks} logical banks (${kvRail.totalSwaBanks} SWA ring cap=16 + ` +
            `${kvRail.totalFullBanks} Full cap=32; ${(kvMemorySummary.kvPayloadBytes / 1024).toFixed(0)} KiB mini payload). ` +
            `Invariant to prefill chunking (within D5 tolerance).`
          : `Freezes KV footprint at 36 banks (27 SWA + 9 Full, ${(kvMemorySummary.kvPayloadBytes / 1024).toFixed(0)} KiB mini payload) ` +
            `for any k: loops t < k read committed final-loop history (DoryPastReader) and commit only at t = k, ` +
            `making last_loop (k >= 2) not generally invariant to prefill partitioning (DR17).`,
      tone: includeShallowCoda ? 'spec_oxide' : kvCacheMode === 'per_loop' ? 'swa_cyan' : 'gold',
      worldPos: [kvMidX, 0.5 * (CORE_TOP_Y + CORE_BOT_Y), 0],
      badge: `${kvRail.totalBanks} banks`,
      discrepancyIds: includeShallowCoda ? ['DR23', 'DR19'] : kvCacheMode === 'last_loop' ? ['DR17'] : ['DR19'],
    },
    {
      id: 'callout-macro-compute-caveat',
      title: 'Body Compute Ratio vs. Latency Scope Caveat',
      body:
        `Linear body depth scales as (4 + 5k) / 9 = ${ratioPerLoopToLastLoop.toFixed(3)}× vs. k=1 ` +
        `(${executedSublayerCount} vs. 72 executed sublayers), while untied embeddings & LM head are k-invariant. ` +
        `Depth multipliers are NOT end-to-end latency or throughput multipliers; synthetic miniature weights do not claim reasoning gains.`,
      tone: 'muted',
      badge: 'Scope Guard',
    },
  ];

  // Active Cell Grids (`<= 64` cell-number overlay budget handled by DORY6 spatialOverlay.ts)
  const activeCellGrids: ActiveSheetCellGrid[] = [];
  const primarySheetsForOverlay = isUnrolled
    ? [
        placedSheets[0]!, // x_anchor
        placedSheets.find((s) => s.id === `loop_exit_s_t${activeRecurrentLoop}`) ??
          placedSheets[placedSheets.length - 1]!,
      ]
    : [
        placedSheets[0]!, // x_anchor
        placedSheets.find((s) => s.id === 'folded_latent_trajectory') ??
          placedSheets[placedSheets.length - 1]!,
      ];

  for (let i = 0; i < primarySheetsForOverlay.length; i += 1) {
    const ps = primarySheetsForOverlay[i]!;
    const isHoveredThisSheet =
      hoveredCell !== null &&
      (hoveredCell.partId === ps.id ||
        (hoveredCell.tensorKey === ps.tensorKey &&
          (hoveredCell.loopIteration === null || hoveredCell.loopIteration === ps.loopIteration)));
    activeCellGrids.push({
      originWorld: ps.frame.center,
      extent: ps.frame.extent,
      axisU: ps.frame.axisU,
      axisN: ps.frame.axisN,
      cols: ps.cols,
      rows: ps.rows,
      values: ps.values,
      format: 'signed',
      ...(isHoveredThisSheet && hoveredCell?.col !== null ? { activeCol: hoveredCell.col } : {}),
      ...(isHoveredThisSheet && hoveredCell?.row !== null ? { activeRow: hoveredCell.row } : {}),
      colLabels: CHANNEL_LABELS_8,
      rowLabels: ps.rowLabels,
      colDimLabel: 'd = 8',
      rowDimLabel: ps.rows === 1 ? `pos ${lastCommittedPos}` : `1+k=${ps.rows}`,
    });
  }

  // Contextual Cell Popover when hovering a live 8-channel sheet cell
  let activeCellPopover: CellPopoverSpec | null = null;
  if (hoveredCell !== null && hoveredCell.col !== null && hoveredCell.row !== null) {
    const matched = placedSheets.find(
      (ps) =>
        ps.id === hoveredCell.partId ||
        (ps.tensorKey === hoveredCell.tensorKey &&
          (hoveredCell.loopIteration === null || hoveredCell.loopIteration === ps.loopIteration)),
    );
    if (matched !== undefined) {
      const col = Math.max(0, Math.min(matched.cols - 1, hoveredCell.col));
      const row = Math.max(0, Math.min(matched.rows - 1, hoveredCell.row));
      const val = matched.values[row * matched.cols + col] ?? 0;
      const rowTag = matched.rowLabels[row] ?? `r${row}`;
      const cellWorld = computeSheetCellWorldPos(
        {
          originWorld: matched.frame.center,
          extent: matched.frame.extent,
          axisU: matched.frame.axisU,
          axisN: matched.frame.axisN,
          cols: matched.cols,
          rows: matched.rows,
        },
        col,
        row,
        0.5,
      );
      activeCellPopover = Object.freeze({
        world: cellWorld,
        tensorLabel: matched.label,
        coordLabel: `${rowTag} · ch ${col} (pos ${lastCommittedPos})`,
        badge: `ID ${shortWeightId}`,
        equation: matched.formula,
        valueText: `= ${formatFloatSigned(val)}`,
      });
    }
  }

  // Finalize Atlas & Buffers
  const atlasExtraTag = `${displayMode}:k${kLoops}:${kvCacheMode}:${executionMode}:${record.recordId}`;
  const atlasBuilt = atlasPacker.build('macro', atlasExtraTag);

  const packedSolids = solids.pack();
  const packedLines = lines.pack();
  const packedRibbons = ribbons.pack();
  const packedSheets = sheets.pack();
  const packedFlows = flows.pack();

  const fitBounds: AABB = computeBoundsFromPoints(boundsPoints, 1.35);

  const macroMetadata: DoryMacroViewMetadata = Object.freeze({
    displayMode,
    modelWeightId,
    kLoops,
    kvCacheMode,
    executionMode,
    physicalSublayerCount: 72 as const,
    executedSublayerCount,
    pairedLayerCount: 36 as const,
    effectivePairedDepth,
    kvRail,
    sharedRecurrentWeightBindings,
    selectedPhysicalLayerIdx,
    selectedLoopIteration,
  });

  const title = isUnrolled
    ? `Unrolled Macro Architecture (${executedSublayerCount}L Executed · k=${kLoops})`
    : 'Folded Macro Architecture (72L Physical · 16 + 40×k + 16)';

  const subtitle =
    `${isUnrolled ? 'Unrolled' : 'Folded'} · 72L phys (${effectivePairedDepth} paired exec at k=${kLoops}) · ` +
    `Weight ID ${shortWeightId} · ` +
    `${kvRail.totalBanks} KV banks (${kvCacheMode}${includeShallowCoda ? ' + 8 Shallow Coda Contract A' : ''})`;

  const scene: DorySpatialScene = Object.freeze({
    viewMode: 'macro',
    title,
    subtitle,
    modelWeightId,
    branchId: branch.branchId,
    recordId: record.recordId,
    kLoops,
    kvCacheMode,
    executionMode,
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
    flowVertices: packedFlows.flowVertices,
    flowIndices: packedFlows.flowIndices,
    pickBoxes: Object.freeze(pickBoxes),
    cameraPreset: Object.freeze({
      azimuthDeg: MACRO_VIEW_DESCRIPTOR.defaultCameraAzimuthDeg,
      elevationDeg: MACRO_VIEW_DESCRIPTOR.defaultCameraElevationDeg,
      fitBounds,
    }),
    fitBounds,
    valueAtlas: atlasBuilt.valueAtlas,
    atlasKey: atlasBuilt.atlasKey,
    atlasRegions: atlasBuilt.atlasRegions,
    atlasAudit: atlasBuilt.atlasAudit,
    provenanceBadges: Object.freeze(provenanceBadges),
    callouts: Object.freeze(callouts),
    overlayAnchors: Object.freeze(overlayAnchors),
    landmarks: Object.freeze(landmarks),
    kvDepthTicks: Object.freeze(kvDepthTicks),
    calipers: Object.freeze(calipers),
    activeCellGrids: Object.freeze(activeCellGrids),
    activeCellPopover,
    macroMetadata,
  });

  // Self-audit geometry finiteness, atlas budget/non-overlap, and DR20 compliance before returning
  const validation = validateSceneGeometryFinite(scene);
  if (!validation.valid) {
    throw new Error(`View 1 macroScene validation failed: ${validation.errors.join('; ')}`);
  }

  return scene;
}
