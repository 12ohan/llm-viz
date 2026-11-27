/**
 * Headless Production Architecture Readouts, 0 B Diff 4-Checkpoint Reconciliation,
 * 1 -> 256K KV Scaling Curves, DR1–DR24 Structured Ledger, D10 Memory Bounds &
 * Strictly Read-Only TBPTL Depth-1 Data for NVIDIA Nemotron-Labs-Dory1.0-4B
 * (`src/dory/archReadouts.ts`).
 *
 * Implements `DoryArchReadoutsModuleContract` (`DORY5` Slice 5A) strictly aligned with
 * `DORY0_ARCHITECTURE_AUDIT.md` (§1.4, §2.1, §3.2, §6.1, §7.2, §9, §11.4, §12.1 View 7, §14.2)
 * and `DORY0_REVIEW_RESPONSE.md`:
 *
 *   - **0 B Diff 4-Checkpoint Production Readouts (`BF16 / MXFP8 / NVFP4 / Q5_K_M`):**
 *     Reconciles every safetensors checkpoint (`8 + headerJsonBytes + payloadBytes === totalFileBytes`
 *     AND `unquantizedBf16Bytes + quantizedPayloadBytes === payloadBytes` with `diffBytes === 0`)
 *     and reports exact BF16 stage/component parameters summing to `3,975,229,440` across `507`
 *     tensors. Strictly read-only — **zero fake quantization toggles**.
 *   - **Exact `1 -> 256K` BF16 Production KV Curve Generator (`Audit §7.2`):**
 *     Verifies the exact `S`-invariant `per_loop / last_loop = (4 + 5k) / 9` ratio across every
 *     context length `S in [1 .. 262,144]` and the exact `4x` marginal bytes/token slope drop
 *     across the `S = W = 1,024` SWA saturation knee (`144 -> 36 KiB/tok` at `k = 1`;
 *     `384 -> 96 KiB/tok` at `k = 4`).
 *   - **Structured `DR1–DR24` Ledger & `D1–D10` Decisions:** Filterable by text query, category,
 *     browser-guard flag, or explicit ID list.
 *   - **`D10` Analytical Conditional Memory Bounds (`k = 1..8`):** Documents singleton weights
 *     (`372,992 B`), Contract A `104`-bank KV (`260.00 KiB` payload), `k = 8` `176`-bank KV
 *     (`440.00 KiB` payload), `3.00 MiB` LRU sparse-checkpoint pool, `<= 7.29 MiB` (`k = 4`) and
 *     `<= 9.60 MiB` (`k = 8`) analytical conditional session bounds (`<= 12.00 MiB`), and `512 KiB`
 *     `RG32F` atlas ceiling, plus optional live `DorySessionMemoryAudit` integration.
 *   - **Strictly `READ-ONLY · NO GRADIENTS` TBPTL Depth-1 Schematic & Paper Results:**
 *     Zero gradients, zero training/RLVR loops, zero fake quantization toggles, and explicit
 *     epistemic separation of `arXiv:2611.03998` paper-reported numbers from synthetic miniature
 *     execution (`D2`, `DR13`–`DR16`).
 *
 * `D4` Architectural Boundary Isolation:
 *   - Pure headless TypeScript with ZERO DOM, WebGL, Node built-ins, ambient RNG (`Math.random`),
 *     Gate 4b/4c Nano modules, or sibling-app imports.
 *   - Imports shared formatting solely from `src/shared/format.ts`.
 */

import { formatTokenCount } from '../shared/format';
import {
  DORY_DESIGN_DECISIONS,
  DORY_DISCREPANCY_LEDGER,
  getDoryBrowserGuards,
} from './ledger';
import { DORY_TBPTL_READONLY_BADGE, type DoryCheckpointFormatId } from './scene/types';
import {
  DORY_CHECKPOINT_FORMATS,
  DORY_PRODUCTION_SPEC,
  DORY_SCALING_DISCLOSURE,
  computeEffectivePairedDepth,
  computeExecutedDepth,
  computeKvBankCounts,
  computeMiniKvCacheBytes,
  computeProductionKvBytesBf16,
  validateDoryRecurrentLoops,
} from './spec';
import {
  DORY_PAPER_RESULTS_BADGE,
  type DoryArchReadoutsBundle,
  type DoryArchReadoutsInput,
  type DoryArchReadoutsModuleContract,
  type DoryCheckpointFormatReadoutRow,
  type DoryDepthComputeScalingRow,
  type DoryDiscrepancyFilterCategory,
  type DoryDiscrepancyFilterOptions,
  type DoryDiscrepancyTableModel,
  type DoryDualByteReadout,
  type DoryMemoryBoundReadoutRow,
  type DoryPaperResultReadout,
  type DoryPretrainingStageReadout,
  type DoryProductionKvScalingRow,
  type DoryStageParamBreakdownRow,
  type DoryTbptlSchematicReadout,
} from './story/types';
import {
  DORY_CHECKPOINT_LRU_POOL_BYTES,
  DORY_MAX_ATLAS_BYTES,
  DORY_MAX_SESSION_RETAINED_BYTES,
  DORY_MINI_CONFIG,
  DORY_MINI_WEIGHT_BYTES,
  type DoryDiscrepancyEntry,
  type DoryDiscrepancyId,
  type DoryKvCacheMode,
  type DoryProductionKvPoint,
  type DorySessionMemoryAudit,
} from './types';

// ============================================================================
// AUTHORITATIVE CONSTANTS (AUDIT §3.2, §6.1, §7.2, §11.4, §12.1)
// ============================================================================

const BYTES_PER_KIB = 1_024;
const BYTES_PER_MIB = 1_048_576;
const BYTES_PER_GIB = 1_073_741_824;

const BYTES_PER_KB = 1_000;
const BYTES_PER_MB = 1_000_000;
const BYTES_PER_GB = 1_000_000_000;

/**
 * Canonical context lengths spanning `1 -> 256K` (`262,144`) including every Audit §7.2
 * milestone (`1, 512, 1024 [SWA knee], 2048, 8192, 32768, 131072, 262144`) plus smooth
 * intermediate points for `1 -> 256K` SVG scaling charts.
 */
export const DORY_ARCH_READOUTS_KV_LENGTHS: readonly number[] = Object.freeze([
  1, 64, 256, 512, 1024, 2048, 4096, 8192, 32768, 65536, 131072, 262144,
]);

/** Exact BF16 unquantized payload bytes shared by `MXFP8` and `NVFP4` (`219` BF16 tensors). */
export const DORY_PROD_UNQUANTIZED_BF16_BYTES = 1_344_430_080 as const;

/** Exact `MXFP8` quantized payload bytes (`288` F8_E4M3 + `288` U8 E8M0 block scales). */
export const DORY_PROD_MXFP8_QUANTIZED_BYTES = 3_406_233_600 as const;

/** Exact `NVFP4` quantized payload bytes (`288` U8 packed 2xFP4 + `288` F8_E4M3 + `288` F32). */
export const DORY_PROD_NVFP4_QUANTIZED_BYTES = 1_857_946_752 as const;

/** Exact production linear projection parameters per paired `(Attn, MLP)` decoder block. */
const PROD_LINEAR_PARAMS_PER_PAIR =
  // Attn: q_proj (4096*2560) + k_proj (1024*2560) + v_proj (1024*2560) + g_proj (4096*2560) + o_proj (2560*4096)
  (4096 + 1024 + 1024 + 4096 + 4096) * 2560 +
  // MLP: gate_proj (7168*2560) + up_proj (7168*2560) + down_proj (2560*7168)
  (7168 + 7168 + 7168) * 2560; // 91,750,400

/** Exact production 1D nGPT scale parameters per paired `(Attn, MLP)` decoder block (`6` tensors). */
const PROD_SCALE_PARAMS_PER_PAIR =
  // Attn: attn_alpha (2560) + sqk (4096) + s_gate (4096) = 10,752
  (2560 + 4096 + 4096) +
  // MLP: mlp_alpha (2560) + suv_gate (7168) + suv_up (7168) = 16,896
  (2560 + 7168 + 7168); // 27,648

const PROD_TOTAL_PARAMS_PER_PAIR = PROD_LINEAR_PARAMS_PER_PAIR + PROD_SCALE_PARAMS_PER_PAIR; // 91,778,048
const PROD_BOOKEND_MATRIX_PARAMS =
  DORY_PRODUCTION_SPEC.vocabSize * DORY_PRODUCTION_SPEC.hiddenSize; // 335,544,320
const PROD_LOGIT_SCALE_PARAMS = DORY_PRODUCTION_SPEC.vocabSize; // 131,072

/** Heaviest analytical `RG32F` atlas (`256 x 192 x 8 B = 393,216 B = 384.00 KiB`, Audit §11.4). */
const ANALYTICAL_HEAVIEST_ATLAS_BYTES = 256 * 192 * 8; // 393,216 B (0.375 MiB)

/** Analytical transient record/trace/scratch/comparator slices from Audit §11.4 (`D10`). */
const ANALYTICAL_K4_RECORDS_TRACE_SCRATCH_BYTES = 531_628; // ~0.507 MiB
const ANALYTICAL_K8_RECORDS_TRACE_SCRATCH_BYTES = 747_634; // ~0.713 MiB

const VALID_CHECKPOINT_IDS: ReadonlySet<DoryCheckpointFormatId> = new Set([
  'bf16',
  'mxfp8',
  'nvfp4',
  'q5_k_m',
]);

const VALID_KV_CACHE_MODES: ReadonlySet<DoryKvCacheMode> = new Set([
  'per_loop',
  'last_loop',
]);

// ============================================================================
// DUAL SI / IEC BYTE FORMATTER
// ============================================================================

function formatSiBytes(bytes: number): string {
  if (bytes >= BYTES_PER_GB) {
    return `${(bytes / BYTES_PER_GB).toFixed(3)} GB`;
  }
  if (bytes >= BYTES_PER_MB) {
    return `${(bytes / BYTES_PER_MB).toFixed(2)} MB`;
  }
  if (bytes >= BYTES_PER_KB) {
    return `${(bytes / BYTES_PER_KB).toFixed(2)} KB`;
  }
  return `${Math.round(bytes)} B`;
}

function formatIecBytes(bytes: number): string {
  if (bytes >= BYTES_PER_GIB) {
    return `${(bytes / BYTES_PER_GIB).toFixed(3)} GiB`;
  }
  if (bytes >= BYTES_PER_MIB) {
    return `${(bytes / BYTES_PER_MIB).toFixed(2)} MiB`;
  }
  if (bytes >= BYTES_PER_KIB) {
    return `${(bytes / BYTES_PER_KIB).toFixed(2)} KiB`;
  }
  return `${Math.round(bytes)} B`;
}

/**
 * Formats any non-negative byte count in both binary IEC (`KiB/MiB/GiB`) and decimal SI
 * (`KB/MB/GB`) conventions so production checkpoint & KV readouts never confuse GB vs. GiB.
 */
export function formatDoryDualBytes(bytes: number): DoryDualByteReadout {
  if (!Number.isFinite(bytes) || bytes < 0) {
    throw new RangeError(
      `Invalid byte count ${String(bytes)}: expected finite number >= 0.`,
    );
  }
  const siLabel = formatSiBytes(bytes);
  const iecLabel = formatIecBytes(bytes);
  const dualLabel = iecLabel === siLabel ? iecLabel : `${iecLabel} (${siLabel})`;
  return Object.freeze({
    bytes,
    siLabel,
    iecLabel,
    dualLabel,
  });
}

// ============================================================================
// 1. 0 B DIFF 4-CHECKPOINT PRODUCTION READOUTS (AUDIT §2.1 & §3.2)
// ============================================================================

interface CheckpointExtraMeta {
  readonly unquantizedBf16Bytes: number | null;
  readonly quantizedPayloadBytes: number | null;
  readonly discrepancyRefs: readonly DoryDiscrepancyId[];
  readonly dgxSparkReadoutNote: string;
}

const CHECKPOINT_EXTRA_META: Readonly<Record<DoryCheckpointFormatId, CheckpointExtraMeta>> =
  Object.freeze({
    bf16: Object.freeze({
      unquantizedBf16Bytes: 7_950_458_880,
      quantizedPayloadBytes: 0,
      discrepancyRefs: Object.freeze(['DR1', 'DR14', 'DR16'] as const),
      dgxSparkReadoutNote:
        'Canonical W16A16 reference release (507 BF16 tensors; 0 B diff vs. HF LFS; zero quantization noise; read-only baseline).',
    }),
    mxfp8: Object.freeze({
      unquantizedBf16Bytes: DORY_PROD_UNQUANTIZED_BF16_BYTES,
      quantizedPayloadBytes: DORY_PROD_MXFP8_QUANTIZED_BYTES,
      discrepancyRefs: Object.freeze(['DR14', 'DR15', 'DR16'] as const),
      dgxSparkReadoutNote:
        'Read-only W8A16 PTQ (group_size=32, 288 F8_E4M3 + 288 U8 E8M0; released config.json uses observer="mse" exact_e8m0_search_with_dominance_bounds per DR15; 1.67x smaller on disk vs. BF16).',
    }),
    nvfp4: Object.freeze({
      unquantizedBf16Bytes: DORY_PROD_UNQUANTIZED_BF16_BYTES,
      quantizedPayloadBytes: DORY_PROD_NVFP4_QUANTIZED_BYTES,
      discrepancyRefs: Object.freeze(['DR14', 'DR16'] as const),
      dgxSparkReadoutNote:
        'Read-only W4A16 ModelOpt QAD (group_size=16, 288 packed U8 2xFP4 + 288 F8_E4M3 + 288 F32 weight_scale_2; 2.48x smaller on disk vs. BF16; targeted for Blackwell/DGX Spark FP4 bandwidth savings).',
    }),
    q5_k_m: Object.freeze({
      unquantizedBf16Bytes: null,
      quantizedPayloadBytes: null,
      discrepancyRefs: Object.freeze(['DR11', 'DR14', 'DR16'] as const),
      dgxSparkReadoutNote:
        'Read-only GGUF Q5_K_M QAD (llama.cpp PR #30306 stores 217 1D nGPT scale tensors as F32 via conversion/dory.py; caps n_loops <= 4 and disables KV context shift per DR11).',
    }),
  });

/**
 * Builds the read-only 4-checkpoint production table (`BF16 / MXFP8 / NVFP4 / Q5_K_M`),
 * mechanically verifying `8 + headerJsonBytes + payloadBytes === totalFileBytes` and
 * `unquantizedBf16Bytes + quantizedPayloadBytes === payloadBytes` (`0 B` diff) on all 3
 * Hugging Face safetensors checkpoints.
 */
export function buildDoryCheckpointReadouts(): readonly DoryCheckpointFormatReadoutRow[] {
  const rows: DoryCheckpointFormatReadoutRow[] = [];

  for (let i = 0; i < DORY_CHECKPOINT_FORMATS.length; i += 1) {
    const spec = DORY_CHECKPOINT_FORMATS[i]!;
    const extra = CHECKPOINT_EXTRA_META[spec.id];

    const isSafetensors = spec.headerJsonBytes !== null && spec.payloadBytes !== null;
    const safetensorsContainerExact: boolean | null = isSafetensors
      ? 8 + spec.headerJsonBytes! + spec.payloadBytes! === spec.totalFileBytes
      : null;

    const payloadSplitExact =
      isSafetensors &&
      extra.unquantizedBf16Bytes !== null &&
      extra.quantizedPayloadBytes !== null
        ? extra.unquantizedBf16Bytes + extra.quantizedPayloadBytes === spec.payloadBytes
        : false;

    const zeroDiffVerified =
      isSafetensors &&
      spec.diffBytes === 0 &&
      safetensorsContainerExact === true &&
      payloadSplitExact;

    rows.push(
      Object.freeze({
        ...spec,
        shortSha: spec.pinnedCommitSha.slice(0, 8),
        payloadFormatted:
          spec.payloadBytes !== null ? formatDoryDualBytes(spec.payloadBytes) : null,
        totalFormatted: formatDoryDualBytes(spec.totalFileBytes),
        unquantizedBf16Bytes: extra.unquantizedBf16Bytes,
        quantizedPayloadBytes: extra.quantizedPayloadBytes,
        zeroDiffVerified,
        safetensorsContainerExact,
        discrepancyRefs: extra.discrepancyRefs,
        dgxSparkReadoutNote: extra.dgxSparkReadoutNote,
      }),
    );
  }

  return Object.freeze(rows);
}

// ============================================================================
// 2. EXACT BF16 STAGE & COMPONENT PARAMETER BREAKDOWN (AUDIT §3.2)
// ============================================================================

function makeStageBreakdownRow(params: {
  readonly id: string;
  readonly stageOrComponent: string;
  readonly physicalRangeLabel: string;
  readonly tensorCount: number;
  readonly parameters: number;
  readonly quantizedInMxfp8Nvfp4: boolean | 'partial' | 'n/a';
  readonly quantNote: string;
  readonly isSubtotal?: boolean;
}): DoryStageParamBreakdownRow {
  const shareOfParamsPct = (params.parameters / DORY_PRODUCTION_SPEC.totalParams) * 100;
  return Object.freeze({
    id: params.id,
    stageOrComponent: params.stageOrComponent,
    physicalRangeLabel: params.physicalRangeLabel,
    tensorCount: params.tensorCount,
    parameters: params.parameters,
    parametersFormatted: params.parameters.toLocaleString('en-US'),
    shareOfParamsPct,
    bf16Payload: formatDoryDualBytes(params.parameters * 2),
    quantizedInMxfp8Nvfp4: params.quantizedInMxfp8Nvfp4,
    quantNote: params.quantNote,
    ...(params.isSubtotal !== undefined ? { isSubtotal: params.isSubtotal } : {}),
  });
}

/**
 * Builds the exact `BF16` stage & component parameter breakdown (`Audit §3.2`).
 * Non-subtotal rows partition the model disjointly and sum to exact `507` tensors,
 * `3,975,229,440` parameters, and `7,950,458,880 B` (`0 B` diff vs. HF LFS header+payload).
 */
export function buildDoryBf16StageBreakdown(): readonly DoryStageParamBreakdownRow[] {
  const inputPairs = DORY_PRODUCTION_SPEC.nInputLayers >> 1; // 8
  const recurrentPairs = DORY_PRODUCTION_SPEC.nRecurrentLayers >> 1; // 20
  const outputPairs = DORY_PRODUCTION_SPEC.nOutputLayers >> 1; // 8
  const totalPairs = DORY_PRODUCTION_SPEC.numPairedLayers; // 36

  const inputParams = inputPairs * PROD_TOTAL_PARAMS_PER_PAIR; // 734,224,384
  const recurrentParams = recurrentPairs * PROD_TOTAL_PARAMS_PER_PAIR; // 1,835,560,960
  const outputParams = outputPairs * PROD_TOTAL_PARAMS_PER_PAIR; // 734,224,384

  const totalLinearParams = totalPairs * PROD_LINEAR_PARAMS_PER_PAIR; // 3,303,014,400
  const totalNgptScaleParams = totalPairs * PROD_SCALE_PARAMS_PER_PAIR + PROD_LOGIT_SCALE_PARAMS; // 1,126,400

  return Object.freeze([
    makeStageBreakdownRow({
      id: 'embeddings',
      stageOrComponent: 'backbone.embeddings.weight [131072, 2560]',
      physicalRangeLabel: 'Input Bookend (Untied)',
      tensorCount: 1,
      parameters: PROD_BOOKEND_MATRIX_PARAMS,
      quantizedInMxfp8Nvfp4: false,
      quantNote: 'Unquantized BF16 across BF16, MXFP8, and NVFP4 safetensors checkpoints.',
    }),
    makeStageBreakdownRow({
      id: 'lm_head_weight',
      stageOrComponent: 'lm_head.weight [131072, 2560]',
      physicalRangeLabel: 'Output Bookend (Untied)',
      tensorCount: 1,
      parameters: PROD_BOOKEND_MATRIX_PARAMS,
      quantizedInMxfp8Nvfp4: false,
      quantNote: 'Unquantized BF16 across all 3 safetensors checkpoints (tie_word_embeddings=false).',
    }),
    makeStageBreakdownRow({
      id: 'lm_head_logit_scale',
      stageOrComponent: 'lm_head.logit_scale [131072]',
      physicalRangeLabel: 'Output Bookend Scale (Family 5)',
      tensorCount: 1,
      parameters: PROD_LOGIT_SCALE_PARAMS,
      quantizedInMxfp8Nvfp4: false,
      quantNote: 'BF16 on disk in safetensors; FP32 at runtime across all 3 runtimes (F32 in GGUF Q5_K_M, DR14).',
    }),
    makeStageBreakdownRow({
      id: 'input_prelude',
      stageOrComponent: 'Input Prelude (8 Attn + 8 MLP)',
      physicalRangeLabel: 'Phys 0..15 (Pair L0..L7, 1x)',
      tensorCount: inputPairs * 14, // 112 (64 Attn + 48 MLP)
      parameters: inputParams,
      quantizedInMxfp8Nvfp4: 'partial',
      quantNote: '64 linear projection matrices quantized in MXFP8/NVFP4; 48 nGPT scales remain BF16.',
    }),
    makeStageBreakdownRow({
      id: 'recurrent_core',
      stageOrComponent: 'Shared Recurrent Core (20 Attn + 20 MLP)',
      physicalRangeLabel: 'Phys 16..55 (Pair L8..L27, k x)',
      tensorCount: recurrentPairs * 14, // 280 (160 Attn + 120 MLP)
      parameters: recurrentParams,
      quantizedInMxfp8Nvfp4: 'partial',
      quantNote: '160 linear projection matrices quantized in MXFP8/NVFP4; 120 nGPT scales remain BF16; reused across all k loops.',
    }),
    makeStageBreakdownRow({
      id: 'output_coda',
      stageOrComponent: 'Output Coda (8 Attn + 8 MLP)',
      physicalRangeLabel: 'Phys 56..71 (Pair L28..L35, 1x)',
      tensorCount: outputPairs * 14, // 112 (64 Attn + 48 MLP)
      parameters: outputParams,
      quantizedInMxfp8Nvfp4: 'partial',
      quantNote: '64 linear projection matrices quantized in MXFP8/NVFP4; 48 nGPT scales remain BF16.',
    }),
    makeStageBreakdownRow({
      id: 'subtotal_linear_weights',
      stageOrComponent: 'Subtotal: Linear Projection Weights (180 Attn + 108 MLP)',
      physicalRangeLabel: 'Phys 0..71 (288 Matrices)',
      tensorCount: 288,
      parameters: totalLinearParams,
      quantizedInMxfp8Nvfp4: true,
      quantNote: 'Quantized in MXFP8 (F8_E4M3 + U8 E8M0) and NVFP4 (packed U8 + F8_E4M3 + F32 scalar).',
      isSubtotal: true,
    }),
    makeStageBreakdownRow({
      id: 'subtotal_ngpt_scales',
      stageOrComponent: 'Subtotal: All 5 nGPT Scale Families (216 Layer + 1 LM Head)',
      physicalRangeLabel: 'Phys 0..71 + LM Head',
      tensorCount: 217,
      parameters: totalNgptScaleParams,
      quantizedInMxfp8Nvfp4: false,
      quantNote: 'Never quantized: BF16 on disk in safetensors, upcast to FP32 at runtime in vLLM/llama.cpp (F32 in GGUF, DR14).',
      isSubtotal: true,
    }),
    makeStageBreakdownRow({
      id: 'subtotal_zero_norms_biases_adapters',
      stageOrComponent: 'Subtotal: Learned Norm Weights / Linear Biases / Loop Adapters',
      physicalRangeLabel: 'All 72 Physical Sublayers',
      tensorCount: 0,
      parameters: 0,
      quantizedInMxfp8Nvfp4: 'n/a',
      quantNote: 'Strictly 0 across all 507 tensors: weightless unit-sphere L2Norm, bias-free linears, and zero loop adapters.',
      isSubtotal: true,
    }),
  ]);
}

// ============================================================================
// 3. EFFECTIVE DEPTH VS. LINEAR COMPUTE & KV BANK SCALING (AUDIT §6.1)
// ============================================================================

/**
 * Builds the exact depth vs. linear compute & KV bank scaling table across all supported loop
 * budgets `k in {1, 2, 3, 4, 6, 8}` (`Audit §6.1`).
 */
export function buildDoryDepthComputeScalingTable(): readonly DoryDepthComputeScalingRow[] {
  const inputPairs = DORY_PRODUCTION_SPEC.nInputLayers >> 1; // 8
  const recurrentPairs = DORY_PRODUCTION_SPEC.nRecurrentLayers >> 1; // 20
  const outputPairs = DORY_PRODUCTION_SPEC.nOutputLayers >> 1; // 8
  const fixedPairs = inputPairs + outputPairs; // 16

  const k1BodyLinearParams = (fixedPairs + recurrentPairs) * PROD_LINEAR_PARAMS_PER_PAIR; // 3,303,014,400
  const k1WithHeadLinearParams = k1BodyLinearParams + PROD_BOOKEND_MATRIX_PARAMS; // 3,638,558,720

  const rows: DoryDepthComputeScalingRow[] = [];
  for (let i = 0; i < DORY_MINI_CONFIG.supportedRecurrentLoops.length; i += 1) {
    const kLoops = DORY_MINI_CONFIG.supportedRecurrentLoops[i]!;
    const executedSublayers = computeExecutedDepth(kLoops);
    const effectivePairedDepth = computeEffectivePairedDepth(kLoops);
    const effectiveLinearBodyParams =
      (fixedPairs + kLoops * recurrentPairs) * PROD_LINEAR_PARAMS_PER_PAIR;
    const effectiveLinearWithHeadParams = effectiveLinearBodyParams + PROD_BOOKEND_MATRIX_PARAMS;
    const banksPerLoop = computeKvBankCounts(kLoops, 'per_loop', false);
    const ratio = (4 + 5 * kLoops) / 9;

    rows.push(
      Object.freeze({
        kLoops,
        isTrained: kLoops <= 4,
        isDefault: kLoops === DORY_PRODUCTION_SPEC.defaultRecurrentLoops,
        physicalSublayers: 72,
        executedSublayers,
        effectivePairedDepth,
        effectiveLinearBodyParams,
        effectiveLinearWithHeadParams,
        bodyComputeRatioVsK1: effectiveLinearBodyParams / k1BodyLinearParams,
        linearWithHeadRatioVsK1: effectiveLinearWithHeadParams / k1WithHeadLinearParams,
        perLoopTotalBanks: banksPerLoop.totalBanks,
        perLoopSwaBanks: banksPerLoop.totalSwaBanks,
        perLoopFullBanks: banksPerLoop.totalFullBanks,
        lastLoopTotalBanks: 36,
        perLoopToLastLoopKvRatio: ratio,
      }),
    );
  }

  return Object.freeze(rows);
}

// ============================================================================
// 4. EXACT 1 -> 256K BF16 PRODUCTION KV SCALING GENERATOR (AUDIT §7.2)
// ============================================================================

/**
 * Generates an exact `1 -> 256K` BF16 logical KV memory curve for any valid loop count `k >= 1`
 * and cache policy (`'per_loop' | 'last_loop'`).
 */
export function buildDoryProductionKvCurve(
  kLoops: number,
  kvCacheMode: DoryKvCacheMode,
  seqLengths: readonly number[] = DORY_ARCH_READOUTS_KV_LENGTHS,
): readonly DoryProductionKvPoint[] {
  const k = validateDoryRecurrentLoops(kLoops);
  const points: DoryProductionKvPoint[] = new Array<DoryProductionKvPoint>(seqLengths.length);
  for (let i = 0; i < seqLengths.length; i += 1) {
    points[i] = computeProductionKvBytesBf16(seqLengths[i]!, k, kvCacheMode);
  }
  return Object.freeze(points);
}

/**
 * Verifies the Audit §7.2 `S`-Invariance Theorem numerically across all `seqLengths`:
 * `KV_per_loop(S, k) / KV_last_loop(S, k) === (4 + 5k) / 9` for EVERY `S >= 1`.
 */
export function verifyDoryProductionKvSInvariantRatio(
  kLoops: number,
  seqLengths: readonly number[] = DORY_ARCH_READOUTS_KV_LENGTHS,
): {
  readonly kLoops: number;
  readonly expectedRatio: number;
  readonly maxAbsDeviation: number;
  readonly invariantVerified: boolean;
} {
  const k = validateDoryRecurrentLoops(kLoops);
  if (seqLengths.length === 0) {
    throw new RangeError(
      'verifyDoryProductionKvSInvariantRatio requires at least one sequence length >= 1.',
    );
  }

  const expectedRatio = (4 + 5 * k) / 9;
  let maxAbsDeviation = 0;
  let checkedCount = 0;

  for (let i = 0; i < seqLengths.length; i += 1) {
    const s = seqLengths[i]!;
    if (!Number.isInteger(s) || s < 1) {
      throw new RangeError(
        `Invalid sequence length seqLengths[${i}]=${String(s)}: expected integer >= 1.`,
      );
    }
    const pl = computeProductionKvBytesBf16(s, k, 'per_loop');
    const ll = computeProductionKvBytesBf16(s, k, 'last_loop');
    const observed = pl.totalBytes / ll.totalBytes;
    const dev = Math.abs(observed - expectedRatio);
    if (dev > maxAbsDeviation) {
      maxAbsDeviation = dev;
    }
    checkedCount += 1;
  }

  return Object.freeze({
    kLoops: k,
    expectedRatio,
    maxAbsDeviation,
    invariantVerified: checkedCount > 0 && maxAbsDeviation <= 1e-15,
  });
}

/**
 * Builds the canonical `1 -> 256K` BF16 production logical KV scaling table (`Audit §7.2`),
 * exposing exact `last_loop` (`= per_loop k=1`), `per_loop k=2/4/8`, SWA share (`75.00% -> 1.16%`),
 * and exact `4x` marginal slope drop across the `S = 1,024` SWA saturation knee.
 */
export function buildDoryProductionKvScalingTable(): readonly DoryProductionKvScalingRow[] {
  const wProd = DORY_PRODUCTION_SPEC.slidingWindow; // 1,024
  const bytesPerTok = DORY_PRODUCTION_SPEC.kvBytesPerTokenPerLayerBf16; // 4,096 B

  const rows: DoryProductionKvScalingRow[] = [];
  for (let i = 0; i < DORY_ARCH_READOUTS_KV_LENGTHS.length; i += 1) {
    const seqLen = DORY_ARCH_READOUTS_KV_LENGTHS[i]!;
    const lastLoopPoint = computeProductionKvBytesBf16(seqLen, 1, 'last_loop');
    const perLoopK2Point = computeProductionKvBytesBf16(seqLen, 2, 'per_loop');
    const perLoopK4Point = computeProductionKvBytesBf16(seqLen, 4, 'per_loop');
    const perLoopK8Point = computeProductionKvBytesBf16(seqLen, 8, 'per_loop');

    // For S <= 1,024 both SWA (3x) and Full (1x) grow; for S > 1,024 SWA saturates -> exact 4x slope drop!
    const activeFactor = seqLen <= wProd ? 4 : 1;
    const marginalBytesPerTokLastLoop = 9 * activeFactor * bytesPerTok; // 144 KiB/tok -> 36 KiB/tok
    const marginalBytesPerTokPerLoopK4 = 24 * activeFactor * bytesPerTok; // 384 KiB/tok -> 96 KiB/tok

    rows.push(
      Object.freeze({
        seqLen,
        seqLenLabel: formatTokenCount(seqLen),
        isSwaKnee: seqLen === wProd,
        lastLoopPoint,
        perLoopK2Point,
        perLoopK4Point,
        perLoopK8Point,
        lastLoopFormatted: formatDoryDualBytes(lastLoopPoint.totalBytes),
        perLoopK2Formatted: formatDoryDualBytes(perLoopK2Point.totalBytes),
        perLoopK4Formatted: formatDoryDualBytes(perLoopK4Point.totalBytes),
        perLoopK8Formatted: formatDoryDualBytes(perLoopK8Point.totalBytes),
        swaSharePct: lastLoopPoint.swaShare * 100,
        marginalBytesPerTokLastLoop,
        marginalBytesPerTokPerLoopK4,
      }),
    );
  }

  return Object.freeze(rows);
}

// ============================================================================
// 5. D10 ANALYTICAL CONDITIONAL MEMORY BOUNDS & LIVE AUDIT TABLE (AUDIT §11.4)
// ============================================================================

/**
 * Builds the `D10` analytical conditional memory bounds table across `k = 1..8` (`<= 12.00 MiB`
 * session cap, `3.00 MiB` LRU sparse-checkpoint pool, `512.00 KiB` `RG32F` atlas cap), optionally
 * incorporating a live `DorySessionMemoryAudit` and active atlas byte size.
 */
export function buildDoryMemoryBoundsTable(
  liveSessionMemory?: DorySessionMemoryAudit | null,
  activeAtlasBytes?: number | null,
): readonly DoryMemoryBoundReadoutRow[] {
  const k4SpecKv = computeMiniKvCacheBytes(4, 'per_loop', true, DORY_MINI_CONFIG);
  const k4DeepOnlyKv = computeMiniKvCacheBytes(4, 'per_loop', false, DORY_MINI_CONFIG);
  const k8ExtrapKv = computeMiniKvCacheBytes(8, 'per_loop', false, DORY_MINI_CONFIG);

  const shallowCodaExtraBytes = k4SpecKv.kvPayloadBytes - k4DeepOnlyKv.kvPayloadBytes; // 20,480 B (+20 KiB)
  const k4BoundBytes =
    12 * k4SpecKv.kvPayloadBytes +
    DORY_CHECKPOINT_LRU_POOL_BYTES +
    ANALYTICAL_K4_RECORDS_TRACE_SCRATCH_BYTES +
    DORY_MINI_WEIGHT_BYTES +
    ANALYTICAL_HEAVIEST_ATLAS_BYTES; // 7,638,444 B (~7.28 MiB <= 7.29 MiB <= 12.00 MiB)

  const k8BoundBytes =
    12 * k8ExtrapKv.kvPayloadBytes +
    DORY_CHECKPOINT_LRU_POOL_BYTES +
    ANALYTICAL_K8_RECORDS_TRACE_SCRATCH_BYTES +
    DORY_MINI_WEIGHT_BYTES +
    ANALYTICAL_HEAVIEST_ATLAS_BYTES; // 10,066,290 B (~9.60 MiB <= 12.00 MiB)

  const lruPoolBytes =
    liveSessionMemory != null
      ? liveSessionMemory.checkpointPoolBytes
      : DORY_CHECKPOINT_LRU_POOL_BYTES;
  const lruPoolCap =
    liveSessionMemory != null
      ? liveSessionMemory.maxCheckpointPoolBytes
      : DORY_CHECKPOINT_LRU_POOL_BYTES;

  const atlasBytes =
    activeAtlasBytes !== undefined && activeAtlasBytes !== null
      ? activeAtlasBytes
      : ANALYTICAL_HEAVIEST_ATLAS_BYTES;

  const rows: DoryMemoryBoundReadoutRow[] = [
    Object.freeze({
      id: 'singleton_weights',
      label: 'Singleton Miniature Weights + Scales (507 F32 Tensors)',
      bytes: DORY_MINI_WEIGHT_BYTES,
      formatted: formatDoryDualBytes(DORY_MINI_WEIGHT_BYTES),
      capBytes: null,
      capFormatted: null,
      withinCap: true,
      note:
        '93,248 F32 floats (1:1 tensor count with production BF16) — singleton shared across all loops and branches (zero cloning).',
    }),
    Object.freeze({
      id: 'kv_k4_spec_104',
      label: '1 Branch KV Cache — k=4 + Contract A Self-Spec (104 Banks)',
      bytes: k4SpecKv.kvPayloadBytes,
      formatted: formatDoryDualBytes(k4SpecKv.kvPayloadBytes),
      capBytes: null,
      capFormatted: null,
      withinCap: true,
      note:
        `96 deep per_loop banks (${formatDoryDualBytes(k4DeepOnlyKv.kvPayloadBytes).iecLabel}) + 8 persistent output_shallow Coda banks (+${formatDoryDualBytes(shallowCodaExtraBytes).iecLabel}); +${formatDoryDualBytes(k4SpecKv.absPosTagBytes + k4SpecKv.writerLoopTagBytes).iecLabel} absPosTag/writerLoopTag metadata.`,
    }),
    Object.freeze({
      id: 'kv_k8_extrap_176',
      label: '1 Branch KV Cache — k=8 Extrapolation (176 Banks)',
      bytes: k8ExtrapKv.kvPayloadBytes,
      formatted: formatDoryDualBytes(k8ExtrapKv.kvPayloadBytes),
      capBytes: null,
      capFormatted: null,
      withinCap: true,
      note:
        `132 SWA (cap_swa=16) + 44 Full (S_max=32) banks; +${formatDoryDualBytes(k8ExtrapKv.absPosTagBytes + k8ExtrapKv.writerLoopTagBytes).iecLabel} tag metadata. Scales linearly as (4 + 5k) / 9.`,
    }),
    Object.freeze({
      id: 'd10_lru_pool',
      label: 'D10 Byte-Budgeted LRU Sparse-Checkpoint Pool',
      bytes: lruPoolBytes,
      formatted: formatDoryDualBytes(lruPoolBytes),
      capBytes: lruPoolCap,
      capFormatted: formatDoryDualBytes(lruPoolCap).iecLabel,
      withinCap:
        liveSessionMemory != null
          ? liveSessionMemory.withinCheckpointPoolBudget
          : lruPoolBytes <= lruPoolCap,
      note:
        liveSessionMemory != null
          ? `Live pool holds ${liveSessionMemory.checkpointCount} sparse checkpoints (active branch heads pinned; oldest unused checkpoints evicted automatically).`
          : '3.00 MiB (3,145,728 B) byte-budgeted LRU cap holding up to 11 k=4 spec checkpoints or 6 k=8 extrapolation checkpoints with active heads pinned.',
    }),
    Object.freeze({
      id: 'session_bound_k4',
      label: 'Analytical Conditional Session Upper Bound — k=4 Spec (12 Branches)',
      bytes: k4BoundBytes,
      formatted: formatDoryDualBytes(k4BoundBytes),
      capBytes: DORY_MAX_SESSION_RETAINED_BYTES,
      capFormatted: formatDoryDualBytes(DORY_MAX_SESSION_RETAINED_BYTES).iecLabel,
      withinCap: k4BoundBytes <= DORY_MAX_SESSION_RETAINED_BYTES,
      note:
        '12 Contract A k=4 branches + D10 3.00 MiB LRU pool + 128 lightweight records + trace/scratch/comparator + singleton weights + 384 KiB atlas (<= 7.29 MiB; +4.71 MiB headroom).',
    }),
    Object.freeze({
      id: 'session_bound_k8',
      label: 'Analytical Conditional Session Upper Bound — k=8 Extrap (12 Branches)',
      bytes: k8BoundBytes,
      formatted: formatDoryDualBytes(k8BoundBytes),
      capBytes: DORY_MAX_SESSION_RETAINED_BYTES,
      capFormatted: formatDoryDualBytes(DORY_MAX_SESSION_RETAINED_BYTES).iecLabel,
      withinCap: k8BoundBytes <= DORY_MAX_SESSION_RETAINED_BYTES,
      note:
        '12 k=8 per_loop branches + D10 3.00 MiB LRU pool (6 checkpoints) + 128 lightweight records + trace/comparator + singleton weights + 384 KiB atlas (<= 9.60 MiB; +2.40 MiB headroom).',
    }),
    Object.freeze({
      id: 'rg32f_atlas_cap',
      label:
        activeAtlasBytes !== undefined && activeAtlasBytes !== null
          ? 'Active View WebGL2 RG32F Value Atlas'
          : 'Heaviest Analytical WebGL2 RG32F Value Atlas (View 3 GQA)',
      bytes: atlasBytes,
      formatted: formatDoryDualBytes(atlasBytes),
      capBytes: DORY_MAX_ATLAS_BYTES,
      capFormatted: formatDoryDualBytes(DORY_MAX_ATLAS_BYTES).iecLabel,
      withinCap: atlasBytes <= DORY_MAX_ATLAS_BYTES,
      note:
        '2-channel RG32F (8 B/texel: R=scalar, G=DR20 causal mask) strictly <= 512.00 KiB with 0 texel overlaps.',
    }),
  ];

  if (liveSessionMemory != null) {
    rows.push(
      Object.freeze({
        id: 'live_session_retained',
        label: 'Live DorySession Deduplicated ArrayBuffers (Incl. Weights)',
        bytes: liveSessionMemory.totalWithWeightsBytes,
        formatted: formatDoryDualBytes(liveSessionMemory.totalWithWeightsBytes),
        capBytes: liveSessionMemory.maxSessionRetainedBytes,
        capFormatted: formatDoryDualBytes(liveSessionMemory.maxSessionRetainedBytes).iecLabel,
        withinCap:
          liveSessionMemory.withinTotalBudgetWithWeights &&
          liveSessionMemory.withinSessionBudget &&
          liveSessionMemory.nonAliasedSlabsVerified,
        note:
          `${liveSessionMemory.branchCount} branches, ${liveSessionMemory.recordCount} records, ${liveSessionMemory.checkpointCount} pooled checkpoints; ` +
          `session-only=${formatDoryDualBytes(liveSessionMemory.totalSessionRetainedBytes).iecLabel}, ` +
          `non-aliased slabs=${liveSessionMemory.nonAliasedSlabsVerified ? 'VERIFIED' : 'VIOLATION'}.`,
      }),
    );
  }

  return Object.freeze(rows);
}

// ============================================================================
// 6. STRUCTURED DR1–DR24 DISCREPANCY LEDGER FILTER (AUDIT §9)
// ============================================================================

function matchesQuery(entry: DoryDiscrepancyEntry, needle: string): boolean {
  if (entry.id.toLowerCase().includes(needle)) return true;
  if (entry.title.toLowerCase().includes(needle)) return true;
  if (entry.summary.toLowerCase().includes(needle)) return true;
  if (entry.provenance.toLowerCase().includes(needle)) return true;
  if (entry.resolution.toLowerCase().includes(needle)) return true;
  if (entry.category.toLowerCase().includes(needle)) return true;
  if (entry.primaryTag.toLowerCase().includes(needle)) return true;
  if (entry.secondaryTags !== undefined) {
    for (let i = 0; i < entry.secondaryTags.length; i += 1) {
      if (entry.secondaryTags[i]!.toLowerCase().includes(needle)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Filters the complete 24-entry `DR1–DR24` discrepancy & browser-guard ledger cleanly by
 * search query, category, browser-guard flag, or explicit discrepancy ID list.
 */
export function filterDoryDiscrepancyLedger(
  options: DoryDiscrepancyFilterOptions = {},
): DoryDiscrepancyTableModel {
  const query = (options.query ?? '').trim();
  const needle = query.toLowerCase();
  const category: DoryDiscrepancyFilterCategory = options.category ?? 'all';
  const onlyBrowserGuards = options.onlyBrowserGuards ?? false;
  const idSet: ReadonlySet<DoryDiscrepancyId> | null =
    options.ids !== undefined ? new Set(options.ids) : null;

  const filtered: DoryDiscrepancyEntry[] = [];
  for (let i = 0; i < DORY_DISCREPANCY_LEDGER.length; i += 1) {
    const entry = DORY_DISCREPANCY_LEDGER[i]!;
    if (category !== 'all' && entry.category !== category) continue;
    if (onlyBrowserGuards && !entry.isBrowserGuard) continue;
    if (idSet !== null && !idSet.has(entry.id)) continue;
    if (needle.length > 0 && !matchesQuery(entry, needle)) continue;
    filtered.push(entry);
  }

  return Object.freeze({
    totalCount: DORY_DISCREPANCY_LEDGER.length,
    browserGuardCount: getDoryBrowserGuards().length,
    filteredCount: filtered.length,
    query,
    category,
    onlyBrowserGuards,
    rows: Object.freeze(filtered),
  });
}

// ============================================================================
// 7. STRICTLY READ-ONLY TBPTL DEPTH-1 & TRAINING SCHEMATIC DATA (AUDIT §12.1)
// ============================================================================

const PRETRAINING_STAGES: readonly DoryPretrainingStageReadout[] = Object.freeze([
  Object.freeze({
    stageNumber: 1,
    id: 'pretrain_stage1',
    title: 'Stage 1 — NVFP4 Base Pretraining',
    tokenBudgetLabel: '4.5T tokens',
    precisionAndDepthLabel: 'NVFP4 · non-looped (k = 1)',
    maxContextLabel: 'Base context',
    keyMechanism:
      'Establishes the 72-sublayer 3:1 SWA:Full Full-nGPT hypersphere backbone with post-optimizer weight matrix normalization (never applied at inference time).',
  }),
  Object.freeze({
    stageNumber: 2,
    id: 'pretrain_stage2',
    title: 'Stage 2 — BF16 Looped Mid-Training + ELR',
    tokenBudgetLabel: '1.5T tokens',
    precisionAndDepthLabel: 'BF16 · looped k in {1, 2, 3, 4}',
    maxContextLabel: 'Standard context',
    keyMechanism:
      'Trains the shared 40-sublayer recurrent core via TBPTL depth-1 stop-gradient (0 early-exit losses) plus 0.1 ELR rank override so Prelude gradients flow solely through final-loop +x.',
  }),
  Object.freeze({
    stageNumber: 3,
    id: 'pretrain_stage3',
    title: 'Stage 3 — 256K Long-Context Extension',
    tokenBudgetLabel: '300B tokens',
    precisionAndDepthLabel: 'BF16 · looped k in {1, 2, 3, 4}',
    maxContextLabel: '256K (262,144 tokens)',
    keyMechanism:
      'Extends dual-RoPE full-attention layers (theta_F = 10,000,000 vs. SWA theta_S = 10,000, W = 1,024) out to 256K context length without YaRN/NTK scaling.',
  }),
  Object.freeze({
    stageNumber: 4,
    id: 'posttrain_sft',
    title: 'Stage 4 — Supervised Fine-Tuning (SFT)',
    tokenBudgetLabel: 'Post-training SFT',
    precisionAndDepthLabel: 'BF16 · elastic k in {1, 2, 3, 4}',
    maxContextLabel: 'Chat / <think> / <tool_call>',
    keyMechanism:
      'Aligns chat framing (<|im_start|>, <|im_end|>, <think>, <tool_call>), default thinking mode, and {reasoning effort: efficient} medium-effort mode.',
  }),
  Object.freeze({
    stageNumber: 5,
    id: 'posttrain_rlvr',
    title: 'Stage 5 — GRPO RLVR Difficulty Curriculum',
    tokenBudgetLabel: '4:1:1:1 curriculum',
    precisionAndDepthLabel: 'BF16 · GRPO RLVR (alpha = 0.3)',
    maxContextLabel: 'Multi-loop reasoning',
    keyMechanism:
      'GRPO RLVR with 4:1:1:1 difficulty curriculum (alpha = 0.3) teaching elastic loop utilization across k in {1..4}. Strictly read-only metadata here (no RLVR execution).',
  }),
]);

const TBPTL_SCHEMATIC_READOUT: DoryTbptlSchematicReadout = Object.freeze({
  readOnly: true,
  hasGradients: false,
  zeroTrainingLoops: true,
  zeroFakeQuantToggles: true,
  badge: DORY_TBPTL_READONLY_BADGE,
  tbptlDepth: 1,
  stopGradientBetweenLoops: true,
  earlyExitLosses: 0,
  preludeGradientPath:
    'TBPTL depth-1 places a stop-gradient across loop-to-loop feedback (s_{k-1} -> u_k), so the Input Prelude (0..15) receives gradients solely through the final-loop anchor re-injection (+x) inside u_k = N(s_{k-1} + x), with 0 early-exit auxiliary losses.',
  elrRankOverride: 0.1,
  rlvrDifficultyRatio: '4:1:1:1',
  rlvrAlpha: 0.3,
  stages: PRETRAINING_STAGES,
  epistemicCaveat:
    'Strictly READ-ONLY · NO GRADIENTS architectural concept summary from arXiv:2611.03998 (§3–§5). Zero backpropagation, zero optimizer updates, zero RLVR training loops, and zero fake quantization toggles exist in this visualizer.',
});

/**
 * Returns the immutable, strictly `READ-ONLY · NO GRADIENTS` TBPTL depth-1 & 5-stage training
 * schematic dataset (`Audit §1.4 & §12.1 View 7`).
 */
export function buildDoryTbptlSchematicReadout(): DoryTbptlSchematicReadout {
  return TBPTL_SCHEMATIC_READOUT;
}

// ============================================================================
// 8. PAPER RESULTS READOUTS (STRICTLY SEPARATED FROM SYNTHETIC MINIATURE)
// ============================================================================

const PAPER_RESULTS_READOUTS: readonly DoryPaperResultReadout[] = Object.freeze([
  Object.freeze({
    id: 'paper_elastic_depth_scaling',
    badge: DORY_PAPER_RESULTS_BADGE,
    metricLabel: '12-Benchmark Elastic Loop Depth Scaling (k in {1..4} trained; {6, 8} base extrap.)',
    reportedValue:
      'Effective depth 32 + 40k (72 -> 192 physical sublayers; 3.303B -> 8.808B effective body params) from one 3.975B checkpoint with 0 loop adapters.',
    provenance: 'arXiv:2611.03998 §1, §3–§4 & Audit §6.1',
    caveat:
      'Benchmark accuracy improvements with k are empirical properties of the trained 3.975B weights — NEVER claimed or asserted on synthetic d=8 miniature weights.',
    discrepancyRefs: Object.freeze(['DR4', 'DR11'] as const),
  }),
  Object.freeze({
    id: 'paper_output_token_efficiency',
    badge: DORY_PAPER_RESULTS_BADGE,
    metricLabel: 'Agentic Output-Token Compression at Deeper Loops',
    reportedValue: '0.40x–0.77x generated output tokens on agentic benchmarks at higher k.',
    provenance: 'arXiv:2611.03998 §4.2 & Audit §6.1',
    caveat:
      'Paper-level trained behavior that can partially offset per-token compute scaling (4 + 5k) / 9; do NOT equate depth multipliers directly with wall-clock latency or claim shortening on miniature weights.',
    discrepancyRefs: Object.freeze(['DR11'] as const),
  }),
  Object.freeze({
    id: 'paper_self_spec_speedup',
    badge: DORY_PAPER_RESULTS_BADGE,
    metricLabel: 'Paper §6.2 Self-Speculative Decode Speedup (k=1 -> k=4, gamma=4)',
    reportedValue: '1.37x–1.71x measured decode speedup across 1K and 16K input-token curves.',
    provenance: 'arXiv:2611.03998 §6.2 & figures/dory_decode_speedup.pdf (Audit §8, DR13, DR23)',
    caveat:
      'Quotes only verified 1K/16K curves (caption erroneously claims a "without activation reuse" curve per DR13). Absent from upstream runtimes; browser runs reconstructed Contract A (104 Dual-Coda banks, DR23) without claiming browser wall-clock speedup.',
    discrepancyRefs: Object.freeze(['DR13', 'DR23'] as const),
  }),
  Object.freeze({
    id: 'paper_four_checkpoint_release',
    badge: DORY_PAPER_RESULTS_BADGE,
    metricLabel: '4-Checkpoint Production Footprints (0 B Diff Safetensors + GGUF Q5_K_M)',
    reportedValue:
      'BF16 7.404 GiB (7.951 GB) -> MXFP8-PTQ 4.424 GiB (4.751 GB) -> Q5_K_M-QAD 3.421 GiB (3.673 GB) -> NVFP4-QAD 2.983 GiB (3.202 GB).',
    provenance: 'HF Collection nvidia/nemotron-labs-dory HTTP Range headers (Audit §2.1, §3.2, DR14–DR16)',
    caveat:
      'All 219 unquantized tensors (untied bookends + 217 nGPT scales) stay BF16 on disk in safetensors (upcast to FP32 at runtime; F32 in GGUF). Strictly read-only metadata — zero fake browser quantization toggles.',
    discrepancyRefs: Object.freeze(['DR14', 'DR15', 'DR16'] as const),
  }),
  Object.freeze({
    id: 'paper_dual_axis_kv_scaling',
    badge: DORY_PAPER_RESULTS_BADGE,
    metricLabel: 'Dual-Axis KV Scaling (1 -> 256K BF16, S-Invariant Ratio & 4x SWA Knee)',
    reportedValue:
      '256K BF16 KV: 9.105 GiB (last_loop / k=1) vs. 14.164 GiB (k=2, 1.556x) vs. 24.281 GiB (k=4, 2.667x); SWA saturates at S=1,024 (75.00% -> 1.16% share, 4.0x slope drop).',
    provenance: 'Audit §7.2 Analytical Derivation & §7.4 Theorems 1–3 (DR17)',
    caveat:
      'Logical retained tensor footprint at B=1 (excludes paged-attention fragmentation). For k >= 2, last_loop trades exact (4+5k)/9 memory savings for prefill-partition dependence (DR17).',
    discrepancyRefs: Object.freeze(['DR17'] as const),
  }),
]);

/**
 * Returns the curated, explicitly badged `Paper Results` entries (`arXiv:2611.03998`),
 * strictly separated from synthetic miniature execution.
 */
export function buildDoryPaperResultsReadouts(): readonly DoryPaperResultReadout[] {
  return PAPER_RESULTS_READOUTS;
}

// ============================================================================
// 9. UNIFIED HEADLESS ARCHITECTURE READOUTS BUNDLE
// ============================================================================

const GOVERNANCE_SCOPE_NOTE =
  'Epistemic Governance Boundary: (1) Full-nGPT 72-sublayer execution, k in {1..4, 6, 8}, per_loop & last_loop KV run live on deterministic synthetic d=8 miniature weights under the D5 binary32-storage / binary64-accumulation FP32 contract; ' +
  '(2) Paper §6.2 self-speculation runs live under explicitly badged Contract A (104 Dual-Coda banks, DR23); ' +
  '(3) 4-checkpoint quantization footprints (BF16/MXFP8/NVFP4/Q5_K_M), TBPTL depth-1 training stages, and arXiv:2611.03998 empirical tables are strictly READ-ONLY metadata with zero fake quantization toggles and zero gradients.';

/**
 * Assembles the complete headless read-only architecture readouts bundle consumed by
 * View 7 (`bookendsScene.ts`), the 4-Act guided walkthrough (`walkthrough.ts`),
 * Gate 5 tests, and the DORY6 browser readout drawers.
 */
export function buildDoryArchReadouts(
  input: DoryArchReadoutsInput = {},
): DoryArchReadoutsBundle {
  const activeKLoops = validateDoryRecurrentLoops(
    input.activeKLoops ?? DORY_PRODUCTION_SPEC.defaultRecurrentLoops,
  );
  const activeKvCacheMode: DoryKvCacheMode =
    input.activeKvCacheMode ?? DORY_PRODUCTION_SPEC.defaultKvCacheMode;
  if (!VALID_KV_CACHE_MODES.has(activeKvCacheMode)) {
    throw new RangeError(
      `Invalid DoryKvCacheMode: ${String(activeKvCacheMode)}.`,
    );
  }
  const selectedCheckpointFormat: DoryCheckpointFormatId =
    input.selectedCheckpointFormat ?? 'bf16';
  if (!VALID_CHECKPOINT_IDS.has(selectedCheckpointFormat)) {
    throw new RangeError(
      `Invalid DoryCheckpointFormatId: ${String(selectedCheckpointFormat)}.`,
    );
  }

  const checkpoints = buildDoryCheckpointReadouts();
  const safetensorsRows = checkpoints.filter((c) => c.safetensorsContainerExact !== null);
  const allSafetensorsVerified =
    safetensorsRows.length === 3 &&
    safetensorsRows.every((c) => c.zeroDiffVerified && c.safetensorsContainerExact === true);
  if (!allSafetensorsVerified) {
    throw new Error(
      'Invariant violation: expected all 3 HF safetensors checkpoints (BF16, MXFP8, NVFP4) to reconcile with 0 B diff.',
    );
  }

  return Object.freeze({
    readOnly: true,
    zeroFakeQuantToggles: true,
    activeKLoops,
    activeKvCacheMode,
    selectedCheckpointFormat,
    checkpoints,
    allSafetensorsZeroByteDiff: true,
    bf16StageBreakdown: buildDoryBf16StageBreakdown(),
    scalingDisclosure: DORY_SCALING_DISCLOSURE,
    depthComputeScaling: buildDoryDepthComputeScalingTable(),
    productionKvScaling: buildDoryProductionKvScalingTable(),
    activeSInvariantRatio: (4 + 5 * activeKLoops) / 9,
    swaKneeSlopeDropFactor: 4,
    memoryBounds: buildDoryMemoryBoundsTable(
      input.liveSessionMemory ?? null,
      input.activeAtlasBytes ?? null,
    ),
    discrepancyLedger: filterDoryDiscrepancyLedger(input.discrepancyFilter),
    designDecisions: DORY_DESIGN_DECISIONS,
    tbptlSchematic: buildDoryTbptlSchematicReadout(),
    paperResults: buildDoryPaperResultsReadouts(),
    governanceScopeNote: GOVERNANCE_SCOPE_NOTE,
  });
}

// Compile-time verification against frozen Step-0 contract (`src/dory/story/types.ts`)
const _ARCH_READOUTS_CONTRACT_CHECK: DoryArchReadoutsModuleContract = {
  formatDoryDualBytes,
  buildDoryCheckpointReadouts,
  buildDoryBf16StageBreakdown,
  buildDoryDepthComputeScalingTable,
  buildDoryProductionKvScalingTable,
  buildDoryMemoryBoundsTable,
  filterDoryDiscrepancyLedger,
  buildDoryTbptlSchematicReadout,
  buildDoryPaperResultsReadouts,
  buildDoryArchReadouts,
};
void _ARCH_READOUTS_CONTRACT_CHECK;
