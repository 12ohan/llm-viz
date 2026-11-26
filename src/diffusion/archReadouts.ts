/**
 * DF4 Headless Architecture Readouts, Context KV Scaling & D1–D24 Discrepancy Ledger
 * (`src/diffusion/archReadouts.ts`)
 *
 * Pure headless TypeScript — zero DOM (`document`/`window`), zero WebGL,
 * zero `Math.random()`, zero `src/twotower/` imports, and strictly `Infinity`/`NaN`-free.
 */

import { computeLlama4AttnScale } from './rope';
import {
  DIFFUSION_EPISTEMIC_NOTE,
  DIFFUSION_PROJECTION_REVERSAL_BADGE,
} from './scene/provenance';
import {
  DIFFUSION_14B_COMPONENT_BREAKDOWN,
  DIFFUSION_ATLAS_BUDGETS,
  DIFFUSION_CANONICAL_REFS,
  DIFFUSION_CONTEXT_TIERS,
  DIFFUSION_DISCREPANCY_LEDGER,
  DIFFUSION_FAMILY_SPECS,
  DIFFUSION_LFS_INVENTORY,
  DIFFUSION_PRODUCTION_KV_TABLE,
  verifyDiffusionLfsInventory,
} from './spec';
import type {
  DiffusionArchReadoutsBundle,
  DiffusionArchReadoutsInput,
  DiffusionCanonical14BSummary,
  DiffusionComponentReadoutRow,
  DiffusionContextKvScalingRow,
  DiffusionDiscrepancyCategory,
  DiffusionDiscrepancyFilterOptions,
  DiffusionDiscrepancyTableModel,
  DiffusionDiscrepancyViewRow,
  DiffusionDualByteReadout,
  DiffusionLfsTableRow,
  DiffusionMemoryBoundaryRow,
  DiffusionNfeTpfReadout,
  DiffusionProductionVariantRow,
} from './story/types';
import {
  DEFAULT_DIFFUSION_SESSION_BOUNDS,
  DIFFUSION_KV_SLAB_BYTES_PER_TOKEN,
  DIFFUSION_MINI_CONFIG,
  type DiffusionDiscrepancyEntry,
  type DiffusionExecutionMode,
  type DiffusionSessionMemoryReport,
  type ForwardCounters,
} from './types';

// ============================================================================
// LOCKED CONSTANTS
// ============================================================================

export const DIFFUSION_CONTEXT_SCALING_PRESETS: readonly number[] = Object.freeze([
  4096,
  16384,
  20480,
  65536,
  131072,
  262144,
]);

export const DIFFUSION_BACKBONE_MINI_BYTES = 229_064 * 4; // 916,256 B = 894.78 KiB
export const DIFFUSION_LORA_MINI_BYTES = 42_240 * 4; // 168,960 B = 165.00 KiB

const SI_KB = 1_000;
const SI_MB = 1_000_000;
const SI_GB = 1_000_000_000;

const IEC_KIB = 1024;
const IEC_MIB = 1024 * 1024;
const IEC_GIB = 1024 * 1024 * 1024;

const PROD_14B_KV_ENTRY = DIFFUSION_PRODUCTION_KV_TABLE.find(
  (entry) => entry.variantId === '14B',
)!;
const PROD_14B_KV_BYTES_PER_TOKEN = PROD_14B_KV_ENTRY.kvBytesPerToken; // 163,840 B/tok
const PROD_14B_TEMP_BLOCK32_BYTES = PROD_14B_KV_ENTRY.tempBlock32Bytes; // 5,242,880 B (5.00 MiB)

const PROD_14B_ORIG_MAX_POS = 16_384;
const PROD_14B_MAX_POS = 262_144;
const PROD_14B_LLAMA4_BETA = 0.1;

// ============================================================================
// FORMATTING HELPERS
// ============================================================================

function sanitizeNonNegativeFinite(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  return value;
}

function formatPlainBytes(bytes: number): string {
  return Number.isInteger(bytes) ? `${bytes} B` : `${bytes.toFixed(2)} B`;
}

function formatSiBytes(bytes: number): string {
  if (bytes >= SI_GB) return `${(bytes / SI_GB).toFixed(2)} GB`;
  if (bytes >= SI_MB) return `${(bytes / SI_MB).toFixed(2)} MB`;
  if (bytes >= SI_KB) return `${(bytes / SI_KB).toFixed(2)} KB`;
  return formatPlainBytes(bytes);
}

function formatIecBytes(bytes: number): string {
  if (bytes >= IEC_GIB) return `${(bytes / IEC_GIB).toFixed(2)} GiB`;
  if (bytes >= IEC_MIB) return `${(bytes / IEC_MIB).toFixed(2)} MiB`;
  if (bytes >= IEC_KIB) return `${(bytes / IEC_KIB).toFixed(2)} KiB`;
  return formatPlainBytes(bytes);
}

/**
 * Format a byte count into decimal SI (`B/KB/MB/GB`), binary IEC (`B/KiB/MiB/GiB`),
 * and a combined `dualLabel` such as `"25.16 GiB (27.01 GB)"` (or `"512 B"` below 1000 B).
 */
export function formatDualSiIecBytes(bytes: number): DiffusionDualByteReadout {
  const safeBytes = sanitizeNonNegativeFinite(bytes);
  const siLabel = formatSiBytes(safeBytes);
  const iecLabel = formatIecBytes(safeBytes);
  const dualLabel = iecLabel === siLabel ? iecLabel : `${iecLabel} (${siLabel})`;
  return Object.freeze({
    bytes: safeBytes,
    siLabel,
    iecLabel,
    dualLabel,
  });
}

/**
 * Format parameter counts compactly (`13.51B`, `47.19M`, `229.06K`), guarding non-finite/negative inputs.
 */
export function formatParameterCount(params: number): string {
  const safe = sanitizeNonNegativeFinite(params);
  if (safe >= 1e9) return `${(safe / 1e9).toFixed(2)}B`;
  if (safe >= 1e6) return `${(safe / 1e6).toFixed(2)}M`;
  if (safe >= 1e3) return `${(safe / 1e3).toFixed(2)}K`;
  return Number.isInteger(safe) ? `${safe}` : safe.toFixed(2);
}

function formatTokenLabel(tokens: number): string {
  if (tokens === 1_048_576) return '1M (1,048,576)';
  if (tokens >= 1024 && tokens % 1024 === 0) {
    return `${tokens / 1024}K (${tokens.toLocaleString('en-US')})`;
  }
  return tokens.toLocaleString('en-US');
}

// ============================================================================
// 3-COUNTER NFE & THROUGHPUT READOUT (D12)
// ============================================================================

const D12_COMPARISON_NOTE =
  'D12 NFE Accounting: ar_generate includes prefill in nfe = G; generate (DLM) excludes prefill (nfe = sum(D_b) + K); linear_spec_generate includes prefill (nfe = 1 + sum(D_c) + C). All 3 counters stay side-by-side so throughput ratios never mix conventions.';

function describeExecutionMode(
  mode: DiffusionExecutionMode,
  causalContext?: boolean,
): { readonly modeLabel: string; readonly methodNote: string } {
  switch (mode) {
    case 'ar':
      return {
        modeLabel: 'AR (Autoregressive Causal)',
        methodNote:
          'ar_generate: includes prefill in referenceMethodNfe (nfe = G); actualForwardPasses = 1 + decodeForwardPasses.',
      };
    case 'dlm':
      return {
        modeLabel:
          causalContext === false
            ? 'DLM (Block Diffusion · Bidir Context)'
            : causalContext === true
              ? 'DLM (Block Diffusion · Causal Context)'
              : 'DLM (Block Diffusion)',
        methodNote:
          'generate (DLM): excludes prefill from referenceMethodNfe (nfe = sum(D_b) + K); actualForwardPasses includes prefill (+1).',
      };
    case 'linear_spec':
      return {
        modeLabel: 'Linear Spec (Draft + AR Verify)',
        methodNote:
          'linear_spec_generate: includes prefill in referenceMethodNfe (nfe = 1 + sum(D_c) + C); matches actualForwardPasses.',
      };
    case 'linear_spec_lora':
      return {
        modeLabel: 'Linear Spec + o_proj LoRA',
        methodNote:
          'linear_spec_generate (o_proj LoRA on draft only; OFF on verify): includes prefill in referenceMethodNfe (nfe = 1 + sum(D_c) + C).',
      };
  }
}

function computeTpfMetric(
  generatedTokens: number,
  denominator: number,
): { readonly label: string; readonly ratioValue: number | null } {
  if (
    !Number.isFinite(generatedTokens) ||
    !Number.isFinite(denominator) ||
    denominator <= 0
  ) {
    return { label: '—', ratioValue: null };
  }
  const ratio = generatedTokens / denominator;
  if (!Number.isFinite(ratio)) {
    return { label: '—', ratioValue: null };
  }
  return {
    label: `${ratio.toFixed(2)} tok/pass`,
    ratioValue: ratio,
  };
}

/**
 * Build an `Infinity`/`NaN`-free 3-counter NFE and tokens-per-forward-pass readout
 * explaining `D12` method accounting across all 4 execution modes.
 */
export function buildDiffusionNfeTpfReadout(
  mode: DiffusionExecutionMode,
  counters: ForwardCounters,
  generatedTokens: number,
  causalContext?: boolean,
): DiffusionNfeTpfReadout {
  const actualForwardPasses = sanitizeNonNegativeFinite(counters.actualForwardPasses);
  const decodeForwardPasses = sanitizeNonNegativeFinite(counters.decodeForwardPasses);
  const referenceMethodNfe = sanitizeNonNegativeFinite(counters.referenceMethodNfe);
  const safeGenerated = sanitizeNonNegativeFinite(generatedTokens);

  const actualMetric = computeTpfMetric(safeGenerated, actualForwardPasses);
  const decodeMetric = computeTpfMetric(safeGenerated, decodeForwardPasses);
  const refMetric = computeTpfMetric(safeGenerated, referenceMethodNfe);
  const { modeLabel, methodNote } = describeExecutionMode(mode, causalContext);

  return Object.freeze({
    mode,
    modeLabel,
    actualForwardPasses,
    decodeForwardPasses,
    referenceMethodNfe,
    generatedTokens: safeGenerated,
    tpfActualLabel: actualMetric.label,
    tpfDecodeLabel: decodeMetric.label,
    tpfRefLabel: refMetric.label,
    actualRatioValue: actualMetric.ratioValue,
    decodeRatioValue: decodeMetric.ratioValue,
    refRatioValue: refMetric.ratioValue,
    methodNote,
    d12ComparisonNote: D12_COMPARISON_NOTE,
  });
}

// ============================================================================
// 4-TIER CONTEXT KV SCALING & LLAMA-4 BOUNDARY
// ============================================================================

function classifyContextTier(tokens: number): 1 | 2 | 3 | 4 {
  if (tokens <= PROD_14B_ORIG_MAX_POS) {
    return 1;
  }
  if (tokens === 20_480 || tokens === 65_536) {
    return 2;
  }
  if (tokens <= PROD_14B_MAX_POS) {
    return 3;
  }
  return 4;
}

/**
 * Compute production `14B` BF16 KV scaling rows, miniature `40L` FP32 KV rows,
 * 4-tier provenance classification, and exact Llama-4 tail scaling across context lengths.
 */
export function buildDiffusionContextKvScalingRows(
  tokenCounts: readonly number[] = DIFFUSION_CONTEXT_SCALING_PRESETS,
): readonly DiffusionContextKvScalingRow[] {
  const rows: DiffusionContextKvScalingRow[] = [];
  const tierById = new Map(DIFFUSION_CONTEXT_TIERS.map((spec) => [spec.tier, spec]));

  for (const rawTokens of tokenCounts) {
    const tokens = Number.isFinite(rawTokens)
      ? Math.max(0, Math.trunc(rawTokens))
      : 0;
    const tier = classifyContextTier(tokens);
    const tierSpec = tierById.get(tier)!;

    const prodKvBytes = tokens * PROD_14B_KV_BYTES_PER_TOKEN;
    const prodWithTempBytes = prodKvBytes + PROD_14B_TEMP_BLOCK32_BYTES;
    const miniKvBytes = tokens * DIFFUSION_KV_SLAB_BYTES_PER_TOKEN;

    const tailPos = Math.max(0, tokens - 1);
    const bucket = Math.floor(tailPos / PROD_14B_ORIG_MAX_POS);
    const llama4ScaleAtTail = computeLlama4AttnScale(
      tailPos,
      PROD_14B_LLAMA4_BETA,
      PROD_14B_ORIG_MAX_POS,
    );
    const llama4ScaleLabel = `${llama4ScaleAtTail.toFixed(4)}×`;
    const crossesOrigMaxPos = tokens > PROD_14B_ORIG_MAX_POS;

    let transitionNote: string;
    if (!crossesOrigMaxPos) {
      transitionNote = `Within original_max_position_embeddings (16,384); tail pos=${tailPos} stays in Llama-4 bucket 0 (1.0000×).`;
    } else if (tokens <= PROD_14B_MAX_POS) {
      transitionNote = `Crosses orig_max_pos (16,384); tail pos=${tailPos.toLocaleString('en-US')} enters Llama-4 bucket ${bucket} (${llama4ScaleLabel}).`;
    } else {
      transitionNote = `Exceeds 256K max_position_embeddings (262,144); tail pos=${tailPos.toLocaleString('en-US')} in analytical bucket ${bucket} (${llama4ScaleLabel}).`;
    }

    rows.push(
      Object.freeze({
        tokens,
        tokenLabel: formatTokenLabel(tokens),
        tier,
        tierBadge: tierSpec.badge,
        tierProvenance: tierSpec.provenance,
        prodBf16Kv: formatDualSiIecBytes(prodKvBytes),
        prodWithTempBlock32: formatDualSiIecBytes(prodWithTempBytes),
        miniFp32Kv: formatDualSiIecBytes(miniKvBytes),
        llama4ScaleAtTail,
        llama4ScaleLabel,
        crossesOrigMaxPos,
        transitionNote,
      }),
    );
  }

  return Object.freeze(rows);
}

// ============================================================================
// 4-WAY MEMORY BOUNDARY AUDIT
// ============================================================================

interface ActiveAtlasInput {
  readonly bytes?: number;
  readonly width?: number;
  readonly height?: number;
  readonly name?: string;
}

function resolveAtlasAuthorizedCap(activeAtlas?: ActiveAtlasInput): {
  readonly capBytes: number;
  readonly capNote: string;
} {
  const gqaCap =
    DIFFUSION_ATLAS_BUDGETS.gqaRopeLora.maxAuthorizedBytes ?? 524_288;
  const smallCap = DIFFUSION_ATLAS_BUDGETS.swiglu.totalBytes; // 32,768 (64x64 RG32F)

  const lowerName = activeAtlas?.name?.toLowerCase() ?? '';
  if (lowerName.includes('swiglu') || lowerName.includes('mask')) {
    return {
      capBytes: smallCap,
      capNote: '64×64 RG32F authorized atlas cap (32 KiB)',
    };
  }
  if (
    lowerName.includes('gqa') ||
    lowerName.includes('rope') ||
    lowerName.includes('lora')
  ) {
    return {
      capBytes: gqaCap,
      capNote: '256×256 RG32F authorized DF3 headroom cap (512 KiB)',
    };
  }
  if (activeAtlas?.width === 64 && activeAtlas?.height === 64) {
    return {
      capBytes: smallCap,
      capNote: '64×64 RG32F authorized atlas cap (32 KiB)',
    };
  }
  return {
    capBytes: gqaCap,
    capNote: '256×256 RG32F authorized DF3 headroom cap (512 KiB)',
  };
}

function resolveAtlasBytes(activeAtlas?: ActiveAtlasInput): number {
  if (!activeAtlas) return 0;
  if (activeAtlas.bytes !== undefined) {
    return sanitizeNonNegativeFinite(activeAtlas.bytes);
  }
  if (activeAtlas.width !== undefined && activeAtlas.height !== undefined) {
    const w = sanitizeNonNegativeFinite(activeAtlas.width);
    const h = sanitizeNonNegativeFinite(activeAtlas.height);
    return w * h * 8;
  }
  return 0;
}

/**
 * Build the 4-way miniature memory boundary table:
 * `backbone_weights`, `lora_weights`, `session_retained`, and `active_atlas`.
 */
export function buildDiffusionMemoryBoundaryRows(
  memoryReport: DiffusionSessionMemoryReport,
  activeAtlas?: ActiveAtlasInput,
): readonly DiffusionMemoryBoundaryRow[] {
  const sessionRetainedBytes = sanitizeNonNegativeFinite(
    memoryReport.totalRetainedBytes,
  );
  const sessionCapBytes =
    Number.isFinite(memoryReport.maxRetainedBytes) &&
    memoryReport.maxRetainedBytes > 0
      ? memoryReport.maxRetainedBytes
      : DEFAULT_DIFFUSION_SESSION_BOUNDS.maxRetainedBytes;
  const sessionUtilPct = Number(
    ((sessionRetainedBytes / sessionCapBytes) * 100).toFixed(2),
  );
  const sessionCapFormatted = formatDualSiIecBytes(sessionCapBytes).dualLabel;

  const atlasBytes = resolveAtlasBytes(activeAtlas);
  const { capBytes: atlasCapBytes, capNote: atlasCapNote } =
    resolveAtlasAuthorizedCap(activeAtlas);
  const atlasUtilPct = Number(((atlasBytes / atlasCapBytes) * 100).toFixed(2));
  const atlasCapFormatted = formatDualSiIecBytes(atlasCapBytes).dualLabel;

  const rows: DiffusionMemoryBoundaryRow[] = [
    Object.freeze({
      id: 'backbone_weights' as const,
      label: 'Miniature Backbone Weights (40L FP32)',
      bytes: DIFFUSION_BACKBONE_MINI_BYTES,
      formatted: formatDualSiIecBytes(DIFFUSION_BACKBONE_MINI_BYTES),
      capBytes: null,
      capFormatted: null,
      utilizationPct: null,
      withinCap: true,
      ownership: 'Shared immutable model parameters (363 tensors · 229,064 floats)',
      note: `Deterministic FP32 miniature backbone (seed 0x${DIFFUSION_MINI_CONFIG.backboneSeed.toString(16).toUpperCase()}), shared across all branches.`,
    }),
    Object.freeze({
      id: 'lora_weights' as const,
      label: 'Draft-Only o_proj LoRA Adapter (40L FP32)',
      bytes: DIFFUSION_LORA_MINI_BYTES,
      formatted: formatDualSiIecBytes(DIFFUSION_LORA_MINI_BYTES),
      capBytes: null,
      capFormatted: null,
      utilizationPct: null,
      withinCap: true,
      ownership: `Shared immutable LoRA adapter (80 tensors · 42,240 floats · r=${DIFFUSION_MINI_CONFIG.loraRank}, α=${DIFFUSION_MINI_CONFIG.loraAlpha}, scale=${DIFFUSION_MINI_CONFIG.loraScale.toFixed(1)})`,
      note: 'Active ONLY during linear_spec_lora draft passes; disabled on verify, DLM, and AR modes (D17).',
    }),
    Object.freeze({
      id: 'session_retained' as const,
      label: 'Session Retained History & KV Slabs',
      bytes: sessionRetainedBytes,
      formatted: formatDualSiIecBytes(sessionRetainedBytes),
      capBytes: sessionCapBytes,
      capFormatted: sessionCapFormatted,
      utilizationPct: sessionUtilPct,
      withinCap: sessionRetainedBytes <= sessionCapBytes && memoryReport.withinBudget,
      ownership: 'DiffusionSession deduplicated ArrayBuffers across branches & records',
      note: `${memoryReport.branchCount} branch(es), ${memoryReport.recordCount} record(s), ${memoryReport.uniqueSlabCount} unique KV slab(s) (${formatDualSiIecBytes(memoryReport.committedKvSlabBytes).iecLabel} KV + ${formatDualSiIecBytes(memoryReport.activeEagerTracesBytes).iecLabel} active traces).`,
    }),
    Object.freeze({
      id: 'active_atlas' as const,
      label: activeAtlas?.name
        ? `Active GPU RG32F Atlas (${activeAtlas.name})`
        : 'Active GPU RG32F Atlas',
      bytes: atlasBytes,
      formatted: formatDualSiIecBytes(atlasBytes),
      capBytes: atlasCapBytes,
      capFormatted: atlasCapFormatted,
      utilizationPct: atlasUtilPct,
      withinCap: atlasBytes <= atlasCapBytes,
      ownership: 'WebGL2 RG32F live data texture (8 bytes/texel)',
      note:
        activeAtlas?.width !== undefined && activeAtlas?.height !== undefined
          ? `${activeAtlas.width}×${activeAtlas.height} RG32F (${atlasCapNote}).`
          : `${atlasCapNote}.`,
    }),
  ];

  return Object.freeze(rows);
}

// ============================================================================
// D1–D24 VERIFIED DISCREPANCY LEDGER CLASSIFICATION & FILTERING
// ============================================================================

type DiscrepancyGroup = Exclude<DiffusionDiscrepancyCategory, 'all'>;

const DISCREPANCY_CATEGORY_META: Readonly<
  Record<DiscrepancyGroup, { readonly label: string }>
> = Object.freeze({
  architecture_specs: Object.freeze({ label: 'Architecture & Specs' }),
  rope_masks: Object.freeze({ label: 'RoPE & Masks' }),
  execution_kv: Object.freeze({ label: 'Execution & KV Cache' }),
  samplers_guards: Object.freeze({ label: 'Samplers & Guards' }),
});

const DISCREPANCY_CATEGORY_BY_ID: Readonly<Record<string, DiscrepancyGroup>> =
  Object.freeze({
    D1: 'architecture_specs',
    D2: 'architecture_specs',
    D3: 'rope_masks',
    D4: 'rope_masks',
    D5: 'architecture_specs',
    D6: 'execution_kv',
    D7: 'execution_kv',
    D8: 'rope_masks',
    D9: 'samplers_guards',
    D10: 'samplers_guards',
    D11: 'samplers_guards',
    D12: 'execution_kv',
    D13: 'architecture_specs',
    D14: 'architecture_specs',
    D15: 'architecture_specs',
    D16: 'execution_kv',
    D17: 'execution_kv',
    D18: 'samplers_guards',
    D19: 'execution_kv',
    D20: 'execution_kv',
    D21: 'execution_kv',
    D22: 'samplers_guards',
    D23: 'rope_masks',
    D24: 'samplers_guards',
  });

function toDiscrepancyViewRow(
  entry: DiffusionDiscrepancyEntry,
): DiffusionDiscrepancyViewRow {
  const numericId = Number.parseInt(entry.id.replace(/^D/i, ''), 10);
  const category: DiscrepancyGroup =
    DISCREPANCY_CATEGORY_BY_ID[entry.id] ?? 'architecture_specs';
  return Object.freeze({
    ...entry,
    numericId: Number.isFinite(numericId) ? numericId : 0,
    category,
    categoryLabel: DISCREPANCY_CATEGORY_META[category].label,
  });
}

const ALL_DISCREPANCY_VIEW_ROWS: readonly DiffusionDiscrepancyViewRow[] =
  Object.freeze(DIFFUSION_DISCREPANCY_LEDGER.map(toDiscrepancyViewRow));

/**
 * Filter the 24-entry `D1–D24` discrepancy ledger by category, case-insensitive
 * substring query, and optional ID whitelist.
 */
export function filterDiffusionDiscrepancyLedger(
  options?: DiffusionDiscrepancyFilterOptions,
): DiffusionDiscrepancyTableModel {
  const query = (options?.query ?? '').trim();
  const normalizedQuery = query.toLowerCase();
  const category: DiffusionDiscrepancyCategory = options?.category ?? 'all';
  const idFilter =
    options?.ids !== undefined
      ? new Set(options.ids.map((id) => id.trim().toUpperCase()))
      : null;

  const filtered = ALL_DISCREPANCY_VIEW_ROWS.filter((row) => {
    if (category !== 'all' && row.category !== category) {
      return false;
    }
    if (idFilter !== null && !idFilter.has(row.id.toUpperCase())) {
      return false;
    }
    if (normalizedQuery.length > 0) {
      const haystack = [
        row.id,
        row.topic,
        row.claimedOrExpected,
        row.verifiedGroundTruth,
        row.resolution,
        row.sourceRef,
        row.categoryLabel,
      ]
        .join(' ')
        .toLowerCase();
      if (!haystack.includes(normalizedQuery)) {
        return false;
      }
    }
    return true;
  });

  return Object.freeze({
    totalCount: ALL_DISCREPANCY_VIEW_ROWS.length,
    filteredCount: filtered.length,
    query,
    category,
    rows: Object.freeze(filtered),
  });
}

// ============================================================================
// CANONICAL 14B SUMMARY, COMPONENTS, VARIANTS & LFS INVENTORY BUILDERS
// ============================================================================

function buildCanonical14BSummary(): DiffusionCanonical14BSummary {
  const { canonicalModel, paper, github, outOfScopeSisterRepo } =
    DIFFUSION_CANONICAL_REFS;
  const breakdown = DIFFUSION_14B_COMPONENT_BREAKDOWN;
  const lfsCheck = verifyDiffusionLfsInventory();
  const lfsTotalDiffBytes = DIFFUSION_LFS_INVENTORY.reduce(
    (sum, entry) => sum + entry.diffBytes,
    0,
  );
  const loraSharePct = (
    (breakdown.lora.parameters / breakdown.backboneParameters) *
    100
  ).toFixed(2);

  return Object.freeze({
    repoId: canonicalModel.repoId,
    revision: canonicalModel.revision,
    shortRevision: canonicalModel.revision.slice(0, 8),
    paperArxivId: paper.arxivId,
    githubRepo: github.repo,
    backboneTensors: breakdown.backboneTensorCount,
    backboneParameters: breakdown.backboneParameters,
    backboneParamsFormatted: formatParameterCount(breakdown.backboneParameters),
    backbonePayload: formatDualSiIecBytes(breakdown.backbonePayloadBytes),
    loraTensors: breakdown.lora.tensorCount,
    loraParameters: breakdown.lora.parameters,
    loraParamsFormatted: formatParameterCount(breakdown.lora.parameters),
    loraShareOfBackbonePct: `+${loraSharePct}%`,
    loraPayload: formatDualSiIecBytes(breakdown.lora.f32Bytes),
    lfsFileCount: lfsCheck.entryCount,
    lfsTotalDiffBytes,
    lfsAllZeroDiff: lfsCheck.valid && lfsTotalDiffBytes === 0,
    projectionReversalNote: DIFFUSION_PROJECTION_REVERSAL_BADGE,
    epistemicNote: DIFFUSION_EPISTEMIC_NOTE,
    outOfScopeNote: `Out of scope: ${outOfScopeSisterRepo.repo} (${outOfScopeSisterRepo.arxivId}) — ${outOfScopeSisterRepo.note}`,
  });
}

function buildComponentRows14B(): readonly DiffusionComponentReadoutRow[] {
  const breakdown = DIFFUSION_14B_COMPONENT_BREAKDOWN;
  const backboneTotal = breakdown.backboneParameters;

  const rows: DiffusionComponentReadoutRow[] = breakdown.backbone.map((entry) =>
    Object.freeze({
      component: entry.component,
      tensorPattern: entry.tensorPattern,
      shape: entry.shape,
      tensorCount: entry.tensorCount,
      parameters: entry.parameters,
      parametersFormatted: formatParameterCount(entry.parameters),
      shareOfBackbonePct: `${((entry.parameters / backboneTotal) * 100).toFixed(2)}%`,
      dtype: 'BF16' as const,
      payload: formatDualSiIecBytes(entry.bf16Bytes),
    }),
  );

  rows.push(
    Object.freeze({
      component: breakdown.lora.component,
      tensorPattern: breakdown.lora.tensorPattern,
      shape: breakdown.lora.shape,
      tensorCount: breakdown.lora.tensorCount,
      parameters: breakdown.lora.parameters,
      parametersFormatted: formatParameterCount(breakdown.lora.parameters),
      shareOfBackbonePct: `+${((breakdown.lora.parameters / backboneTotal) * 100).toFixed(2)}%`,
      dtype: 'F32' as const,
      payload: formatDualSiIecBytes(breakdown.lora.f32Bytes),
    }),
  );

  return Object.freeze(rows);
}

function buildFamilyVariantRows(): readonly DiffusionProductionVariantRow[] {
  return Object.freeze(
    DIFFUSION_FAMILY_SPECS.map((v) => {
      const ropeThetaLabel =
        v.ropeTheta >= 1e9
          ? '1e9'
          : v.ropeTheta >= 1e6
            ? '1e6'
            : String(v.ropeTheta);
      const eosSummary = Array.isArray(v.configEosTokenId)
        ? `cfg=[${v.configEosTokenId.join(',')}] · tok=${v.tokenizerEosTokenId}`
        : `cfg=${v.configEosTokenId} · tok=${v.tokenizerEosTokenId}`;

      return Object.freeze({
        id: v.id,
        repoId: v.repoId,
        shortRevision: v.revision.slice(0, 8),
        isCanonical: v.isCanonical,
        layersLabel:
          v.vitLayers !== null ? `${v.numLayers}L + ${v.vitLayers}L ViT` : `${v.numLayers}L`,
        hiddenSize: v.hiddenSize,
        intermediateSize: v.intermediateSize,
        gqaLabel: `${v.numAttentionHeads}Q / ${v.numKeyValueHeads}KV (dHead=${v.headDim})`,
        ropeThetaLabel,
        yarnLabel: `factor=${v.yarnFactor} · orig=${v.originalMaxPositionEmbeddings.toLocaleString('en-US')} · β=${v.llama4Beta}`,
        maxPositionEmbeddings: v.maxPositionEmbeddings,
        eosSummary,
        backboneParameters: v.backboneParameters,
        backboneParamsFormatted: formatParameterCount(v.backboneParameters),
        backbonePayload: formatDualSiIecBytes(v.backbonePayloadBytes),
        loraParameters: v.loraParameters,
        loraSummary:
          v.loraParameters !== null
            ? `${formatParameterCount(v.loraParameters)} F32 (o_proj)`
            : 'None',
        loraPayload:
          v.loraPayloadBytes !== null
            ? formatDualSiIecBytes(v.loraPayloadBytes)
            : null,
      });
    }),
  );
}

function buildLfsTableRows(): readonly DiffusionLfsTableRow[] {
  return Object.freeze(
    DIFFUSION_LFS_INVENTORY.map((entry) =>
      Object.freeze({
        repoId: entry.repoId,
        shortRepoLabel: entry.repoId.replace(/^nvidia\//, ''),
        revision: entry.revision,
        shortRevision: entry.revision.slice(0, 8),
        file: entry.file,
        dtype: entry.dtype,
        isLora: entry.isLora,
        tensors: entry.tensors,
        parameters: entry.parameters,
        parametersFormatted: formatParameterCount(entry.parameters),
        headerBytes: entry.headerBytes,
        payload: formatDualSiIecBytes(entry.payloadBytes),
        total: formatDualSiIecBytes(entry.totalBytes),
        hfLfsSizeBytes: entry.hfLfsSizeBytes,
        diffBytes: entry.diffBytes,
        lfsSha256Prefix: entry.lfsSha256Prefix,
      }),
    ),
  );
}

const CACHED_CANONICAL_14B = buildCanonical14BSummary();
const CACHED_COMPONENTS_14B = buildComponentRows14B();
const CACHED_FAMILY_VARIANTS = buildFamilyVariantRows();
const CACHED_LFS_INVENTORY = buildLfsTableRows();
const CACHED_CONTEXT_KV_SCALING = buildDiffusionContextKvScalingRows();

/**
 * Assemble the complete frozen DF4 headless architecture readouts bundle from
 * live execution counters, `DiffusionSession.measureMemory()`, optional atlas
 * telemetry, and optional discrepancy ledger filters.
 */
export function buildDiffusionArchReadouts(
  input: DiffusionArchReadoutsInput,
): DiffusionArchReadoutsBundle {
  const hasAtlasInfo =
    input.activeAtlasBytes !== undefined ||
    input.activeAtlasWidth !== undefined ||
    input.activeAtlasHeight !== undefined ||
    input.activeAtlasName !== undefined;

  const memoryBoundaries = buildDiffusionMemoryBoundaryRows(
    input.memoryReport,
    hasAtlasInfo
      ? {
          bytes: input.activeAtlasBytes,
          width: input.activeAtlasWidth,
          height: input.activeAtlasHeight,
          name: input.activeAtlasName,
        }
      : undefined,
  );

  const memoryTotalMiniBytes = memoryBoundaries.reduce(
    (sum, row) => sum + row.bytes,
    0,
  );

  return Object.freeze({
    canonical14B: CACHED_CANONICAL_14B,
    components14B: CACHED_COMPONENTS_14B,
    familyVariants: CACHED_FAMILY_VARIANTS,
    lfsInventory: CACHED_LFS_INVENTORY,
    contextKvScaling: CACHED_CONTEXT_KV_SCALING,
    memoryBoundaries,
    memoryTotalMiniBytes,
    memoryTotalMiniFormatted: formatDualSiIecBytes(memoryTotalMiniBytes),
    nfeTpf: buildDiffusionNfeTpfReadout(
      input.mode,
      input.counters,
      input.generatedTokens,
      input.causalContext,
    ),
    discrepancyLedger: filterDiffusionDiscrepancyLedger(input.discrepancyFilter),
  });
}

// ============================================================================
// UI NUMERIC STRING SANITY VALIDATOR
// ============================================================================

const INVALID_NUMERIC_STRING_PATTERNS: readonly {
  readonly label: string;
  readonly test: (value: string) => boolean;
}[] = [
  {
    label: 'Infinity',
    test: (s) => /-?Infinity\b/.test(s),
  },
  {
    label: 'NaN',
    test: (s) => /\bNaN\b/.test(s),
  },
  {
    label: 'undefined',
    test: (s) => /\bundefined\b/.test(s),
  },
  {
    label: 'null',
    test: (s) => {
      const trimmed = s.trim();
      return (
        trimmed === 'null' ||
        /^null(?:\s*(?:B|KB|MB|GB|KiB|MiB|GiB|tok\/pass|%|×))$/.test(trimmed)
      );
    },
  },
];

/**
 * Recursively inspect strings, numbers, arrays, and objects for invalid UI numeric
 * strings (`Infinity`, `-Infinity`, `NaN`, standalone `undefined`, or stringified `null`)
 * and non-finite numeric values. Returns human-readable paths/matches (`[]` when clean).
 */
export function findInvalidUiNumericStrings(value: unknown): readonly string[] {
  const issues: string[] = [];
  const visited = new WeakSet<object>();

  const walk = (node: unknown, path: string): void => {
    if (typeof node === 'number') {
      if (!Number.isFinite(node)) {
        issues.push(`${path}: non-finite number (${String(node)})`);
      }
      return;
    }

    if (typeof node === 'string') {
      for (const pattern of INVALID_NUMERIC_STRING_PATTERNS) {
        if (pattern.test(node)) {
          issues.push(`${path}: contains ${pattern.label} (${JSON.stringify(node)})`);
          break;
        }
      }
      return;
    }

    if (node === null || typeof node !== 'object') {
      return;
    }

    if (visited.has(node)) {
      return;
    }
    visited.add(node);

    if (Array.isArray(node)) {
      for (let i = 0; i < node.length; i++) {
        walk(node[i], `${path}[${i}]`);
      }
      return;
    }

    if (ArrayBuffer.isView(node)) {
      if (
        node instanceof Float32Array ||
        node instanceof Float64Array
      ) {
        for (let i = 0; i < node.length; i++) {
          if (!Number.isFinite(node[i])) {
            issues.push(`${path}[${i}]: non-finite typed array element (${String(node[i])})`);
            break;
          }
        }
      }
      return;
    }

    if (node instanceof Set) {
      let idx = 0;
      for (const item of node) {
        walk(item, `${path}[set:${idx++}]`);
      }
      return;
    }

    if (node instanceof Map) {
      for (const [k, v] of node.entries()) {
        walk(v, `${path}[${String(k)}]`);
      }
      return;
    }

    for (const [key, val] of Object.entries(node)) {
      walk(val, `${path}.${key}`);
    }
  };

  walk(value, 'root');
  return Object.freeze(issues);
}
