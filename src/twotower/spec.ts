/**
 * Nemotron-Labs-TwoTower Production Specification, Discrepancy Ledger & Memory Accounting
 * (`src/twotower/spec.ts`)
 *
 * Composes Nano v1.0.0's `NEMOTRON_SPEC` and `memoryFor` without modifying any existing Nano
 * files. Clearly separates:
 *   1. `directlyInspectedMetadata`: verbatim fields read from Hugging Face metadata/index files
 *      (`config.json`, `generation_config.json`, `tokenizer_config.json`, `model.safetensors.index.json`).
 *   2. `derivedInventoryCalculations`: analytical parameter counts derived from backbone config
 *      shapes + index key counts, proving 0-byte agreement with `metadata.total_size`.
 *   3. `TWOTOWER_DISCREPANCY_LEDGER`: verified discrepancies across paper, model card, config,
 *      index, and official reference code.
 *   4. `TWOTOWER_MINI_CONFIG`: approved browser-simulation miniature dimensions (`d = 8`, `V = 34`,
 *      `kvCapacity = 24`, `B ∈ {4, 8}`, independently configurable `stepsPerBlock`).
 *   5. `twoTowerMemoryFor`: exact production cache + dual-tower weight accounting across
 *      `'ar'`, `'mock_ar'`, and `'mask_diffusion'` execution modes.
 *
 * Clean-room TypeScript implementation — zero external/proprietary code copied.
 * Pure headless module — zero DOM, WebGL, scene, or render imports.
 */

import { memoryFor, type MemoryReadout } from '../model/memory';
import {
  NEMOTRON_SPEC,
  type BayDescriptor,
  type Kind,
  type NemotronSpec,
} from '../model/spec';

export type TwoTowerExecutionMode = 'ar' | 'mock_ar' | 'mask_diffusion';
export type TwoTowerLayerTransferKind = 'mamba_state' | 'gqa_kv' | 'none';
export type TwoTowerModulationOrdering = 'modulate_then_norm' | 'norm_then_modulate';

export interface TwoTowerProvenance {
  readonly hfRepo: 'nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16';
  readonly hfRevision: '35b6498508be8856d380cdfe4a900a79b918d1ea';
  readonly revisionCheckedDate: '2026-11-25';
  readonly revisionPinDecision: string;
  readonly hfCreatedAt: '2026-04-11T00:58:51Z';
  readonly hfLastModified: '2026-07-01T00:34:48Z';
  readonly paperArxivId: '2606.26493v2';
  readonly paperTitle: string;
  readonly nanoBaselineRevision: string;
}

export interface TwoTowerDirectlyInspectedMetadata {
  readonly provenanceSources: readonly string[];
  readonly architectureClass: 'NemotronHTwoTowerForCausalLM';
  readonly numTowers: 2;
  readonly numHiddenLayersPerTower: 52;
  readonly hybridOverridePattern: string;
  readonly hiddenSize: 2688;
  readonly vocabSize: 131072;
  readonly tieWordEmbeddings: false;
  readonly crossTowerWeightSharing: false;
  readonly positionalEncoding: 'NoPE';
  readonly maxPositionEmbeddings: 262144;
  readonly layerNormEpsilon: 1e-5;
  readonly mambaChunkSize: 128;
  readonly nSharedExperts: 1;
  readonly moeSharedExpertIntermediateSize: 3712;
  readonly moeRoutedScalingFactor: 2.5;
  readonly padTokenId: 0;
  readonly bosTokenId: 1;
  readonly configEosTokenId: 2;
  readonly tokenizerEosTokenId: 11;
  readonly generationConfigEosTokenIds: readonly [2, 11];
  readonly maskTokenId: 3;
  readonly maskTokenSurfaceForm: '[INST]';
  readonly tokenizerMaskTokenField: null;
  readonly timeEmbedderFreqDim: 256;
  readonly timeEmbedderMaxPeriodScale: 1000;
  readonly timeEmbedderMaxPeriod: 10000;
  readonly safetensorsTotalPayloadBytes: 126_371_817_472;
  readonly safetensorsDiskBytes: 126_373_497_008;
  readonly safetensorsTensorCount: 12_544;
  readonly safetensorsShardCount: 24;
  readonly contextTowerTensorCount: 6_243;
  readonly denoiserTowerTensorCount: 6_243;
  readonly timeConditioningTensorCount: 58;
  readonly fp32ScoreCorrectionBiasTensorCount: 46;
  readonly fp32ScoreCorrectionBiasParams: 5_888;
}

export interface TwoTowerDerivedInventoryCalculations {
  readonly methodologyNote: string;
  readonly singleTowerEmbeddingsParams: number;
  readonly singleTowerLmHeadParams: number;
  readonly singleTowerFinalNormParams: number;
  readonly mambaLayerParams: number;
  readonly mambaAllLayersParams: number;
  readonly gqaLayerParams: number;
  readonly gqaAllLayersParams: number;
  readonly moeLayerParams: number;
  readonly moeAllLayersParams: number;
  readonly singleTowerParams: number;
  readonly dualTowerBackboneParams: number;
  readonly timeEmbedderMlp0Params: number;
  readonly timeEmbedderMlp2Params: number;
  readonly timeEmbedderMlpParams: number;
  readonly timeBlockParams: number;
  readonly scaleShiftTablesParams: number;
  readonly timeConditioningTotalParams: number;
  readonly grandTotalParams: number;
  readonly singleTowerActiveParamsExclEmbHead: number;
  readonly singleTowerActiveParamsInclHeadExclEmb: number;
  readonly singleTowerActiveParamsInclEmbHead: number;
  readonly bf16BaseBytes: number;
  readonly fp32ScoreCorrectionBiasExtraBytes: number;
  readonly derivedPayloadBytes: number;
  readonly payloadByteDiscrepancy: number;
  readonly contextTowerPayloadBytes: number;
  readonly denoiserTowerBackbonePayloadBytes: number;
  readonly timeConditioningPayloadBytes: number;
  readonly denoiserTowerTotalPayloadBytes: number;
}

export interface TwoTowerLayerAlignmentEntry {
  /** Zero-based layer index (`0..51`). */
  readonly layerIndex: number;
  /** One-based bay ID (`1..52`), matching `NEMOTRON_SPEC.bays`. */
  readonly bayId: number;
  readonly kind: Kind;
  readonly transferKind: TwoTowerLayerTransferKind;
  readonly transfersState: boolean;
  readonly denoiserModulationOrder: TwoTowerModulationOrdering;
  readonly bay: BayDescriptor;
}

export interface TwoTowerLayerAlignmentSummary {
  readonly totalLayers: 52;
  readonly alignedTransferBayCount: 29;
  readonly gqaTransferBayCount: 6;
  readonly mambaTransferBayCount: 23;
  readonly statelessMoeBayCount: 23;
  /** One-based bay IDs (`{6, 13, 20, 27, 34, 43}`). */
  readonly gqaTransferBayIds: readonly number[];
  /** Zero-based layer indices (`{5, 12, 19, 26, 33, 42}`). */
  readonly gqaTransferLayerIndices: readonly number[];
  /** One-based bay IDs for the 23 Mamba-2 transfer bays. */
  readonly mambaTransferBayIds: readonly number[];
  /** Zero-based layer indices for the 23 Mamba-2 transfer bays. */
  readonly mambaTransferLayerIndices: readonly number[];
  /** One-based bay IDs for the 23 stateless MoE bays. */
  readonly statelessMoeBayIds: readonly number[];
  /** Zero-based layer indices for the 23 stateless MoE bays. */
  readonly statelessMoeLayerIndices: readonly number[];
  readonly entries: readonly TwoTowerLayerAlignmentEntry[];
}

export interface TwoTowerProductionDefaults {
  readonly blockSize: 16;
  readonly stepsPerBlock: 16;
  readonly maskTokenId: 3;
  readonly eosTokenIds: readonly [2, 11];
  readonly configEosTokenId: 2;
  readonly tokenizerEosTokenId: 11;
  readonly codeConfidenceThreshold: 0.9;
  readonly paperConfidenceThreshold: 0.8;
  readonly codeTemperature: 0.0;
  readonly cardTemperature: 0.1;
  readonly primaryPrecision: 'bf16';
  readonly mambaScanPrecision: 'fp32';
  readonly moeRouterPrecision: 'fp64';
  readonly epistemicNote: string;
}

export interface TwoTowerDiscrepancyEntry {
  readonly id: string;
  readonly topic: string;
  readonly paperClaim: string;
  readonly modelCardClaim: string;
  readonly configOrIndexFact: string;
  readonly codeFact: string;
  readonly authoritativeResolution: string;
}

export interface TwoTowerMiniConfig {
  readonly d: 8;
  readonly vocabSize: 34;
  readonly maskTokenId: 32;
  readonly eosTokenId: 33;
  readonly freqDim: 16;
  readonly kvCapacity: 24;
  readonly defaultPromptLen: 8;
  readonly defaultBlockSize: 4;
  readonly supportedBlockSizes: readonly [4, 8];
  /** Independently configurable from `blockSize` (e.g. `K < B`, `K = B`, or `K > B`). */
  readonly defaultStepsPerBlock: 4;
  readonly provisionalDemoThreshold: 0.22;
  readonly presetThresholds: {
    readonly miniDemo: 0.22;
    readonly paper: 0.8;
    readonly referenceCode: 0.9;
  };
  readonly gqaAtlasWidth: 128;
  readonly gqaAtlasHeight: 128;
}

export interface CommunityNvfp4ReferenceNote {
  readonly isOfficialCheckpoint: false;
  readonly label: string;
  readonly quantizedScope: 'routed_moe_experts_only';
  readonly preservedInBf16OrFp32: readonly string[];
  readonly approxTotalWeightGiB: number;
  readonly approxPerTowerWeightGiB: number;
  readonly summary: string;
}

export interface TwoTowerSpec {
  readonly backbone: NemotronSpec;
  readonly provenance: TwoTowerProvenance;
  readonly directlyInspectedMetadata: TwoTowerDirectlyInspectedMetadata;
  readonly derivedInventoryCalculations: TwoTowerDerivedInventoryCalculations;
  readonly layerAlignment: TwoTowerLayerAlignmentSummary;
  readonly productionDefaults: TwoTowerProductionDefaults;
  readonly discrepancyLedger: readonly TwoTowerDiscrepancyEntry[];
  readonly communityNvfp4Reference: CommunityNvfp4ReferenceNote;
}

export interface TwoTowerMemoryReadout {
  readonly prefixTokens: number;
  readonly blockSize: number;
  readonly mode: TwoTowerExecutionMode;
  readonly precision: 'bf16';
  /** Underlying single-tower BF16 readout from Nano's `memoryFor(NEMOTRON_SPEC, prefixTokens, 'bf16')`. */
  readonly contextPrefixCache: MemoryReadout;
  /** Persistent Context Mamba-2 fixed state (`49,364,992 B` = `47.08 MiB`). */
  readonly persistentContextMambaBytes: number;
  /** Persistent Context GQA KV cache (`prefixTokens * 6,144 B` = `prefixTokens * 6 KiB`). */
  readonly persistentContextKvBytes: number;
  /** Total persistent Context prefix cache (`persistentContextMambaBytes + persistentContextKvBytes`). */
  readonly persistentContextTotalBytes: number;
  /** Per-token BF16 KV slope (`6,144 B/tok` = `6 KiB/tok`). */
  readonly kvBytesPerToken: number;
  /**
   * Ephemeral Denoiser block KV (`blockSize * 6 KiB` in `'mask_diffusion'`, `0` otherwise).
   * Concatenated read-only with Context prefix KV during a denoising step and discarded between steps.
   */
  readonly ephemeralDenoiserBlockKvBytes: number;
  /**
   * Extra `mamba_s2` snapshot (`49,364,992 B` = `47.08 MiB` in `'mock_ar'`, `0` otherwise).
   * Holds Context's Mamba-2 state prior to the last token so Denoiser can re-run `last_token`.
   */
  readonly mockArMambaS2SnapshotBytes: number;
  /** Total active logical cache bytes for this mode (`persistent + ephemeral + mockArMambaS2`). */
  readonly totalActiveCacheBytes: number;
  /** Cross-GPU read-only mirror payload transferred to Denoiser when split across 2 GPUs. */
  readonly denoiserReadOnlyMirrorBytes: number;
  /** Dual-tower official BF16 checkpoint weight payload (`126,371,817,472 B` = `117.69 GiB`). */
  readonly dualTowerBf16WeightBytes: number;
  /** Average BF16 weight bytes per tower (`~58.85 GiB/tower`). */
  readonly averagePerTowerBf16WeightBytes: number;
  /** Exact Context tower weight bytes (`63,155,886,464 B` = `58.82 GiB`). */
  readonly contextTowerBf16WeightBytes: number;
  /** Exact Denoiser tower + time-conditioning weight bytes (`63,215,931,008 B` = `58.87 GiB`). */
  readonly denoiserTowerBf16WeightBytes: number;
  /** Time-conditioning BF16 weight bytes (`60,044,544 B` = `57.26 MiB`, denoiser-only). */
  readonly timeConditioningBf16WeightBytes: number;
  /** Active model weight bytes required by `mode` (1 tower for `'ar'`, 2 towers for `'mock_ar'` / `'mask_diffusion'`). */
  readonly activeModelWeightBytes: number;
  /** Distinct community expert-only NVFP4 reference metadata (non-official). */
  readonly communityNvfp4Reference: CommunityNvfp4ReferenceNote;
}

export const TWOTOWER_HF_REPO = 'nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16' as const;
export const TWOTOWER_HF_REVISION = '35b6498508be8856d380cdfe4a900a79b918d1ea' as const;
export const TWOTOWER_REVISION_CHECKED_DATE = '2026-11-25' as const;
export const TWOTOWER_PAPER_ARXIV_ID = '2606.26493v2' as const;

const DIRECTLY_INSPECTED_METADATA: TwoTowerDirectlyInspectedMetadata = Object.freeze({
  provenanceSources: Object.freeze([
    'config.json',
    'generation_config.json',
    'tokenizer_config.json',
    'model.safetensors.index.json',
  ]),
  architectureClass: 'NemotronHTwoTowerForCausalLM',
  numTowers: 2,
  numHiddenLayersPerTower: 52,
  hybridOverridePattern: NEMOTRON_SPEC.pattern,
  hiddenSize: 2688,
  vocabSize: 131072,
  tieWordEmbeddings: false,
  crossTowerWeightSharing: false,
  positionalEncoding: 'NoPE',
  maxPositionEmbeddings: 262144,
  layerNormEpsilon: 1e-5,
  mambaChunkSize: 128,
  nSharedExperts: 1,
  moeSharedExpertIntermediateSize: 3712,
  moeRoutedScalingFactor: 2.5,
  padTokenId: 0,
  bosTokenId: 1,
  configEosTokenId: 2,
  tokenizerEosTokenId: 11,
  generationConfigEosTokenIds: Object.freeze([2, 11] as const),
  maskTokenId: 3,
  maskTokenSurfaceForm: '[INST]',
  tokenizerMaskTokenField: null,
  timeEmbedderFreqDim: 256,
  timeEmbedderMaxPeriodScale: 1000,
  timeEmbedderMaxPeriod: 10000,
  safetensorsTotalPayloadBytes: 126_371_817_472,
  safetensorsDiskBytes: 126_373_497_008,
  safetensorsTensorCount: 12_544,
  safetensorsShardCount: 24,
  contextTowerTensorCount: 6_243,
  denoiserTowerTensorCount: 6_243,
  timeConditioningTensorCount: 58,
  fp32ScoreCorrectionBiasTensorCount: 46,
  fp32ScoreCorrectionBiasParams: 5_888,
});

function computeDerivedInventory(
  backbone: NemotronSpec,
  inspected: TwoTowerDirectlyInspectedMetadata,
): TwoTowerDerivedInventoryCalculations {
  let mambaCount = 0;
  let gqaCount = 0;
  let moeCount = 0;
  for (const bay of backbone.bays) {
    if (bay.kind === 'M') mambaCount++;
    else if (bay.kind === '*') gqaCount++;
    else moeCount++;
  }

  const d = backbone.hidden;
  const vocab = backbone.vocabSize;
  const numLayers = backbone.bays.length;

  // Single-tower untied embeddings, LM head, and final norm
  const singleTowerEmbeddingsParams = vocab * d;
  const singleTowerLmHeadParams = vocab * d;
  const singleTowerFinalNormParams = d;

  // Mamba-2 per-layer parameter inventory (exact config.json projection shapes)
  const mambaInner = backbone.mamba.heads * backbone.mamba.channels; // 4096
  const mambaBcDim = 2 * backbone.mamba.groups * backbone.mamba.state; // 2048
  const mambaConvDim = mambaInner + mambaBcDim; // 6144
  const mambaInProjOut = 2 * mambaInner + mambaBcDim + backbone.mamba.heads; // 10304
  const mambaLayerParams =
    d + // pre-mixer RMSNorm
    mambaInProjOut * d + // in_proj (no bias)
    mambaConvDim * backbone.mamba.kernel + // depthwise conv1d.weight
    mambaConvDim + // conv1d.bias
    backbone.mamba.heads * 3 + // A_log, D, dt_bias
    mambaInner + // mixer.norm (gated group RMSNorm)
    d * mambaInner; // out_proj (no bias)
  const mambaAllLayersParams = mambaCount * mambaLayerParams;

  // GQA Attention per-layer parameter inventory (32Q / 2KV, dHead=128, no bias)
  const qProjOut = backbone.gqa.q * backbone.gqa.headDim; // 4096
  const kvProjOut = backbone.gqa.kv * backbone.gqa.headDim; // 256
  const gqaLayerParams =
    d + // pre-mixer RMSNorm
    qProjOut * d + // q_proj
    kvProjOut * d + // k_proj
    kvProjOut * d + // v_proj
    d * qProjOut; // o_proj
  const gqaAllLayersParams = gqaCount * gqaLayerParams;

  // Granular MoE per-layer parameter inventory (128 routed + 1 double-wide shared, no bias)
  const singleRoutedExpertParams = 2 * d * backbone.moe.width;
  const routedAllExpertsParams = backbone.moe.routed * singleRoutedExpertParams;
  const sharedExpertParams = backbone.moe.shared * 2 * d * backbone.moe.sharedWidth;
  const moeLayerParams =
    d + // pre-mixer RMSNorm
    backbone.moe.routed * d + // router gate.weight
    backbone.moe.routed + // FP32 e_score_correction_bias
    routedAllExpertsParams +
    sharedExpertParams;
  const moeAllLayersParams = moeCount * moeLayerParams;

  const singleTowerParams =
    singleTowerEmbeddingsParams +
    singleTowerLmHeadParams +
    singleTowerFinalNormParams +
    mambaAllLayersParams +
    gqaAllLayersParams +
    moeAllLayersParams;
  const dualTowerBackboneParams = 2 * singleTowerParams;

  // Denoiser-only PixArt-alpha adaLN-single time conditioning
  const freqDim = inspected.timeEmbedderFreqDim;
  const timeEmbedderMlp0Params = d * freqDim + d; // (2688, 256) + (2688,) = 690,816
  const timeEmbedderMlp2Params = d * d + d; // (2688, 2688) + (2688,) = 7,228,032
  const timeEmbedderMlpParams = timeEmbedderMlp0Params + timeEmbedderMlp2Params; // 7,918,848
  const timeBlockParams = 3 * d * d + 3 * d; // (8064, 2688) + (8064,) = 21,684,096
  const scaleShiftTablesParams = numLayers * 3 * d; // 52 * 3 * 2688 = 419,328
  const timeConditioningTotalParams =
    timeEmbedderMlpParams + timeBlockParams + scaleShiftTablesParams; // 30,022,272

  const grandTotalParams = dualTowerBackboneParams + timeConditioningTotalParams; // 63,185,902,848

  // Single-tower active parameters per token (top-6 routed + 1 shared + Mamba + GQA)
  const moeActiveLayerParams =
    d +
    backbone.moe.routed * d +
    backbone.moe.routed +
    backbone.moe.active * singleRoutedExpertParams +
    sharedExpertParams;
  const singleTowerActiveParamsExclEmbHead =
    singleTowerFinalNormParams +
    mambaAllLayersParams +
    gqaAllLayersParams +
    moeCount * moeActiveLayerParams;
  const singleTowerActiveParamsInclHeadExclEmb =
    singleTowerActiveParamsExclEmbHead + singleTowerLmHeadParams;
  const singleTowerActiveParamsInclEmbHead =
    singleTowerActiveParamsInclHeadExclEmb + singleTowerEmbeddingsParams;

  // Exact byte payload proof: every parameter is 2 B (BF16), plus +2 B for each FP32 bias scalar
  const bf16BaseBytes = grandTotalParams * 2;
  const fp32BiasPerTower = moeCount * backbone.moe.routed; // 23 * 128 = 2,944
  const fp32ScoreCorrectionBiasExtraBytes = 2 * fp32BiasPerTower * 2; // 5,888 * 2 = 11,776
  const derivedPayloadBytes = bf16BaseBytes + fp32ScoreCorrectionBiasExtraBytes;
  const payloadByteDiscrepancy = derivedPayloadBytes - inspected.safetensorsTotalPayloadBytes;

  const contextTowerPayloadBytes = singleTowerParams * 2 + fp32BiasPerTower * 2;
  const denoiserTowerBackbonePayloadBytes = contextTowerPayloadBytes;
  const timeConditioningPayloadBytes = timeConditioningTotalParams * 2;
  const denoiserTowerTotalPayloadBytes =
    denoiserTowerBackbonePayloadBytes + timeConditioningPayloadBytes;

  return Object.freeze({
    methodologyNote:
      'Derived analytically from config.json tensor shapes, NEMOTRON_SPEC 52-layer hybrid pattern, and model.safetensors.index.json key structure without downloading weight shards; cross-checked to 0 bytes against metadata.total_size.',
    singleTowerEmbeddingsParams,
    singleTowerLmHeadParams,
    singleTowerFinalNormParams,
    mambaLayerParams,
    mambaAllLayersParams,
    gqaLayerParams,
    gqaAllLayersParams,
    moeLayerParams,
    moeAllLayersParams,
    singleTowerParams,
    dualTowerBackboneParams,
    timeEmbedderMlp0Params,
    timeEmbedderMlp2Params,
    timeEmbedderMlpParams,
    timeBlockParams,
    scaleShiftTablesParams,
    timeConditioningTotalParams,
    grandTotalParams,
    singleTowerActiveParamsExclEmbHead,
    singleTowerActiveParamsInclHeadExclEmb,
    singleTowerActiveParamsInclEmbHead,
    bf16BaseBytes,
    fp32ScoreCorrectionBiasExtraBytes,
    derivedPayloadBytes,
    payloadByteDiscrepancy,
    contextTowerPayloadBytes,
    denoiserTowerBackbonePayloadBytes,
    timeConditioningPayloadBytes,
    denoiserTowerTotalPayloadBytes,
  });
}

function buildLayerAlignment(backbone: NemotronSpec): TwoTowerLayerAlignmentSummary {
  const entries: TwoTowerLayerAlignmentEntry[] = [];
  const gqaTransferBayIds: number[] = [];
  const gqaTransferLayerIndices: number[] = [];
  const mambaTransferBayIds: number[] = [];
  const mambaTransferLayerIndices: number[] = [];
  const statelessMoeBayIds: number[] = [];
  const statelessMoeLayerIndices: number[] = [];

  for (let layerIndex = 0; layerIndex < backbone.bays.length; layerIndex++) {
    const bay = backbone.bays[layerIndex];
    let transferKind: TwoTowerLayerTransferKind;
    let denoiserModulationOrder: TwoTowerModulationOrdering;

    if (bay.kind === '*') {
      transferKind = 'gqa_kv';
      denoiserModulationOrder = 'modulate_then_norm';
      gqaTransferBayIds.push(bay.id);
      gqaTransferLayerIndices.push(layerIndex);
    } else if (bay.kind === 'M') {
      transferKind = 'mamba_state';
      denoiserModulationOrder = 'modulate_then_norm';
      mambaTransferBayIds.push(bay.id);
      mambaTransferLayerIndices.push(layerIndex);
    } else {
      transferKind = 'none';
      denoiserModulationOrder = 'norm_then_modulate';
      statelessMoeBayIds.push(bay.id);
      statelessMoeLayerIndices.push(layerIndex);
    }

    entries.push(
      Object.freeze({
        layerIndex,
        bayId: bay.id,
        kind: bay.kind,
        transferKind,
        transfersState: transferKind !== 'none',
        denoiserModulationOrder,
        bay,
      }),
    );
  }

  return Object.freeze({
    totalLayers: 52,
    alignedTransferBayCount: 29,
    gqaTransferBayCount: 6,
    mambaTransferBayCount: 23,
    statelessMoeBayCount: 23,
    gqaTransferBayIds: Object.freeze(gqaTransferBayIds),
    gqaTransferLayerIndices: Object.freeze(gqaTransferLayerIndices),
    mambaTransferBayIds: Object.freeze(mambaTransferBayIds),
    mambaTransferLayerIndices: Object.freeze(mambaTransferLayerIndices),
    statelessMoeBayIds: Object.freeze(statelessMoeBayIds),
    statelessMoeLayerIndices: Object.freeze(statelessMoeLayerIndices),
    entries: Object.freeze(entries),
  });
}

export const TWOTOWER_DISCREPANCY_LEDGER: readonly TwoTowerDiscrepancyEntry[] = Object.freeze([
  Object.freeze({
    id: 'adaln_param_count',
    topic: 'adaLN-single time-conditioning parameter count',
    paperClaim: '1.5M parameters (arXiv:2606.26493v2 §2.1, §3.2)',
    modelCardClaim: '≈1.5M parameters (README.md)',
    configOrIndexFact:
      '30,022,272 parameters (30.02M) across 58 tensors in shard 24 of model.safetensors.index.json',
    codeFact:
      't_embedder.mlp (7,918,848) + t_block (21,684,096) + 52 scale_shift_tables (419,328) = 30,022,272',
    authoritativeResolution:
      'Report 30,022,272 (30.02M) actual checkpoint parameters and explicitly note the paper/card 1.5M discrepancy.',
  }),
  Object.freeze({
    id: 'shared_experts_count',
    topic: 'MoE shared expert count vs. double-wide width',
    paperClaim: 'References Nemotron 3 Nano 30B-A3B backbone (§2.1)',
    modelCardClaim: '"2 shared experts" (README.md)',
    configOrIndexFact:
      'n_shared_experts: 1, moe_shared_expert_intermediate_size: 3712 (2 × 1856); 2 tensors/layer in index.json',
    codeFact: 'Instantiates 1 shared MLP of intermediate_size = 3712 (identical to Nano)',
    authoritativeResolution:
      'Model as 1 double-wide shared expert (width 3712 = 2 × 1856) matching config.json, index.json, and Nano.',
  }),
  Object.freeze({
    id: 'total_parameter_count',
    topic: 'Total dual-tower parameter count and BF16 payload size',
    paperClaim: 'Two copies of the 30B Nemotron 3 Nano backbone',
    modelCardClaim: '~60B total parameters (README.md)',
    configOrIndexFact:
      '63,185,902,848 total parameters (63.19B); 126,371,817,472 B (117.69 GiB) payload in index.json',
    codeFact: '2 × 31,577,940,288 independent towers + 30,022,272 denoiser time-conditioning params',
    authoritativeResolution:
      'Report exact 63,185,902,848 (63.19B) total parameters and 117.69 GiB BF16 payload (~58.85 GiB/tower).',
  }),
  Object.freeze({
    id: 'max_context_length',
    topic: 'Maximum sequence length (128K vs. 256K)',
    paperClaim: 'Inherits Nano backbone context architecture',
    modelCardClaim: '128K tokens context length (README.md)',
    configOrIndexFact: 'max_position_embeddings: 262144 (256K) in config.json and tokenizer_config.json',
    codeFact: 'Uses config.max_position_embeddings = 262144 (with NoPE)',
    authoritativeResolution:
      'Record config.json value 262,144 (256K) while documenting the model card 128K operational note.',
  }),
  Object.freeze({
    id: 'confidence_threshold_and_temperature',
    topic: 'Default confidence threshold γ and sampling temperature',
    paperClaim: 'γ = 0.8 default operating point (§3.1, §3.6)',
    modelCardClaim: 'γ = 0.8, temperature = 0.1 (README.md usage examples)',
    configOrIndexFact: 'Not specified in config.json or generation_config.json',
    codeFact: 'confidence_threshold = 0.9, temperature = 0.0 (greedy default in inference & modeling code)',
    authoritativeResolution:
      'Expose configurable γ and temperature with explicit presets for miniature demo (0.22), paper (0.80), and reference code (0.90).',
  }),
  Object.freeze({
    id: 'adaln_rmsnorm_ordering',
    topic: 'Split adaLN-single modulate vs. RMSNorm ordering across bay kinds',
    paperClaim: 'Generic PixArt-α adaLN-single conditioning description (§2.1)',
    modelCardClaim: 'Generic adaLN-single description (README.md)',
    configOrIndexFact: 'Not encoded in config.json',
    codeFact:
      'Diffusion forward applies modulate→RMSNorm for Mamba (M) & Attention (*) layers, but RMSNorm→modulate for MoE (E) / MLP layers; all 52 layers apply gate * mixer_out residual gating and unmodulated norm_f.',
    authoritativeResolution:
      'Enforce exact split ordering (M/* modulate_then_norm vs. E norm_then_modulate) in both TypeScript kernels and PyTorch parity tests.',
  }),
  Object.freeze({
    id: 'moe_router_precision',
    topic: 'MoE router matmul precision (FP64 vs. FP32)',
    paperClaim: 'Backbone MoE routing (§2.1)',
    modelCardClaim: 'Backbone MoE (README.md)',
    configOrIndexFact: 'gate.weight stored in BF16; 2 × 23 × 128 = 5,888 e_score_correction_bias params in FP32',
    codeFact:
      'TwoTower HF router casts hidden states and router weights to FP64 for matmul + sigmoid (matching Megatron-Core --moe-router-dtype fp64).',
    authoritativeResolution:
      'Record production MoE router compute precision as FP64 (distinct from Nano v1 FP32) and support explicit router precision policy in simulation.',
  }),
  Object.freeze({
    id: 'special_tokens_mask_and_eos',
    topic: 'Repurposed [MASK] token ID and EOS token ID discrepancy',
    paperClaim: 'Symbolic [MASK] token for masked block diffusion',
    modelCardClaim: 'mask_token_id = 3 and tokenizer.eos_token_id (README.md)',
    configOrIndexFact:
      'tokenizer_config.json has mask_token: null (ID 3 is [INST]); config.json eos_token_id = 2 (</s>); generation_config.json = [2, 11]; tokenizer_config.json = 11 (<|im_end|>)',
    codeFact:
      'Hardcodes mask_token_id = 3 and checks scalar eos_token_id (default 11 in CLI) only after full block commit.',
    authoritativeResolution:
      'Document mask_token_id = 3 ([INST] repurposed) and both EOS IDs [2, 11]; support scalar or array EOS IDs in clean-room sampler.',
  }),
  Object.freeze({
    id: 'mamba_chunk_size_and_training_tokens',
    topic: 'Mamba chunk_size (16 vs. 128) and training token count (1.4T vs. 2.1T)',
    paperClaim: 'Matched Mamba chunk_size = 16 in training; cites ~2.1T (Abstract/§1/§3.1) and ~1.4T (§2.3)',
    modelCardClaim: '~2.1T training tokens (README.md)',
    configOrIndexFact: 'chunk_size: 128 in config.json',
    codeFact: 'Inference chunk-scan uses config.chunk_size = 128 in FP32',
    authoritativeResolution:
      'Record chunk_size = 128 at inference (noting S = 16 training match) and ~2.1T for the released 3-stage checkpoint (noting §2.3 1.4T).',
  }),
]);

export const TWOTOWER_COMMUNITY_NVFP4_REFERENCE: CommunityNvfp4ReferenceNote = Object.freeze({
  isOfficialCheckpoint: false,
  label: 'Community Expert-Only NVFP4 / W4A16 Reference (Non-Official)',
  quantizedScope: 'routed_moe_experts_only',
  preservedInBf16OrFp32: Object.freeze([
    'All 23 Mamba-2 layers (projections, Conv1D, FP32 SSM scan)',
    'All 6 GQA Attention layers & KV cache',
    'MoE router weights (BF16), FP32 e_score_correction_bias, FP64 router compute',
    'All 23 MoE shared experts (width 3712)',
    'Context & Denoiser embeddings and untied LM heads',
    'Denoiser adaLN-single time conditioning (t_embedder, t_block, 52 scale_shift_tables)',
  ]),
  approxTotalWeightGiB: 42,
  approxPerTowerWeightGiB: 21,
  summary:
    'Community expert-only NVFP4/W4A16 quantization compresses routed MoE experts (~98% of weights) from ~117.69 GiB BF16 to ~42 GiB (~21 GiB/tower) without degrading diffusion NFE when Mamba-2, GQA, routers, shared experts, embeddings/heads, and adaLN stay in BF16/FP32. Distinct from NVIDIA official BF16 checkpoint.',
});

const DERIVED_INVENTORY = computeDerivedInventory(NEMOTRON_SPEC, DIRECTLY_INSPECTED_METADATA);
const LAYER_ALIGNMENT = buildLayerAlignment(NEMOTRON_SPEC);

export const TWOTOWER_SPEC: TwoTowerSpec = Object.freeze({
  backbone: NEMOTRON_SPEC,
  provenance: Object.freeze({
    hfRepo: TWOTOWER_HF_REPO,
    hfRevision: TWOTOWER_HF_REVISION,
    revisionCheckedDate: TWOTOWER_REVISION_CHECKED_DATE,
    revisionPinDecision:
      'Retain pinned revision 35b6498508be8856d380cdfe4a900a79b918d1ea; verified current on Hugging Face on 2026-11-25 with zero newer revisions.',
    hfCreatedAt: '2026-04-11T00:58:51Z',
    hfLastModified: '2026-07-01T00:34:48Z',
    paperArxivId: TWOTOWER_PAPER_ARXIV_ID,
    paperTitle:
      'Nemotron-Labs-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context',
    nanoBaselineRevision: NEMOTRON_SPEC.revision,
  }),
  directlyInspectedMetadata: DIRECTLY_INSPECTED_METADATA,
  derivedInventoryCalculations: DERIVED_INVENTORY,
  layerAlignment: LAYER_ALIGNMENT,
  productionDefaults: Object.freeze({
    blockSize: 16,
    stepsPerBlock: 16,
    maskTokenId: 3,
    eosTokenIds: Object.freeze([2, 11] as const),
    configEosTokenId: 2,
    tokenizerEosTokenId: 11,
    codeConfidenceThreshold: 0.9,
    paperConfidenceThreshold: 0.8,
    codeTemperature: 0.0,
    cardTemperature: 0.1,
    primaryPrecision: 'bf16',
    mambaScanPrecision: 'fp32',
    moeRouterPrecision: 'fp64',
    epistemicNote:
      'Production metadata reflects the official 63.19B-parameter BF16 checkpoint (FP32 Mamba-2 SSM scan & MoE bias, FP64 MoE router compute). Browser execution runs an educational miniature Float32/Float64 simulator (d = 8, V = 34, deterministic synthetic weights) — NOT 63B checkpoint weights or real Nemotron tokenizer logits.',
  }),
  discrepancyLedger: TWOTOWER_DISCREPANCY_LEDGER,
  communityNvfp4Reference: TWOTOWER_COMMUNITY_NVFP4_REFERENCE,
});

export const TWOTOWER_MINI_CONFIG: TwoTowerMiniConfig = Object.freeze({
  d: 8,
  vocabSize: 34,
  maskTokenId: 32,
  eosTokenId: 33,
  freqDim: 16,
  kvCapacity: 24,
  defaultPromptLen: 8,
  defaultBlockSize: 4,
  supportedBlockSizes: Object.freeze([4, 8] as const),
  defaultStepsPerBlock: 4,
  provisionalDemoThreshold: 0.22,
  presetThresholds: Object.freeze({
    miniDemo: 0.22,
    paper: 0.8,
    referenceCode: 0.9,
  }),
  gqaAtlasWidth: 128,
  gqaAtlasHeight: 128,
});

/**
 * Compute exact production TwoTower cache and BF16 weight footprint for a given Context prefix
 * length, denoising block size, and execution mode (`'ar'`, `'mock_ar'`, or `'mask_diffusion'`).
 *
 * Composes Nano's `memoryFor(NEMOTRON_SPEC, prefixTokens, 'bf16')` for the single persistent
 * Context prefix cache (`47.08 MiB` Mamba + `6 KiB/tok` BF16 KV), explicitly separating:
 *   - Ephemeral Denoiser block KV (`blockSize * 6 KiB`, discarded between denoising steps)
 *   - Mock-AR's extra `mamba_s2` snapshot (`47.08 MiB`)
 *   - Official dual-tower BF16 weights (`117.69 GiB`, `~58.85 GiB/tower`) vs. non-official
 *     community expert-only NVFP4 metadata.
 */
export function twoTowerMemoryFor(
  prefixTokens: number,
  blockSize: number = TWOTOWER_SPEC.productionDefaults.blockSize,
  mode: TwoTowerExecutionMode = 'mask_diffusion',
): TwoTowerMemoryReadout {
  if (!Number.isInteger(prefixTokens) || prefixTokens < 0) {
    throw new RangeError(`prefixTokens must be a nonnegative integer, received ${prefixTokens}`);
  }
  if (!Number.isInteger(blockSize) || blockSize < 0) {
    throw new RangeError(`blockSize must be a nonnegative integer, received ${blockSize}`);
  }
  if (mode !== 'ar' && mode !== 'mock_ar' && mode !== 'mask_diffusion') {
    throw new RangeError(`Unsupported TwoTower execution mode: ${String(mode)}`);
  }

  const contextPrefixCache = memoryFor(NEMOTRON_SPEC, prefixTokens, 'bf16');
  const persistentContextMambaBytes = contextPrefixCache.fixedMambaBytes;
  const persistentContextKvBytes = contextPrefixCache.kvBytes;
  const persistentContextTotalBytes = contextPrefixCache.hybridTotalBytes;
  const kvBytesPerToken = contextPrefixCache.kvBytesPerToken;

  const ephemeralDenoiserBlockKvBytes =
    mode === 'mask_diffusion' ? blockSize * kvBytesPerToken : 0;
  const mockArMambaS2SnapshotBytes = mode === 'mock_ar' ? persistentContextMambaBytes : 0;
  const totalActiveCacheBytes =
    persistentContextTotalBytes + ephemeralDenoiserBlockKvBytes + mockArMambaS2SnapshotBytes;

  let denoiserReadOnlyMirrorBytes = 0;
  if (mode === 'mask_diffusion') {
    // Denoiser holds a read-only clone of Context's S-1 Mamba state + full prefix KV [0..P-1]
    denoiserReadOnlyMirrorBytes = persistentContextTotalBytes;
  } else if (mode === 'mock_ar') {
    // Mock-AR transfers mamba_s2 + KV[:-1] ([0..P-2]) so Denoiser evaluates token P-1 itself
    const mockArKvTokens = Math.max(0, prefixTokens - 1);
    denoiserReadOnlyMirrorBytes =
      persistentContextMambaBytes + mockArKvTokens * kvBytesPerToken;
  }

  const inv = TWOTOWER_SPEC.derivedInventoryCalculations;
  const dualTowerBf16WeightBytes = inv.derivedPayloadBytes;
  const averagePerTowerBf16WeightBytes = dualTowerBf16WeightBytes / 2;
  const contextTowerBf16WeightBytes = inv.contextTowerPayloadBytes;
  const denoiserTowerBf16WeightBytes = inv.denoiserTowerTotalPayloadBytes;
  const timeConditioningBf16WeightBytes = inv.timeConditioningPayloadBytes;
  const activeModelWeightBytes =
    mode === 'ar' ? contextTowerBf16WeightBytes : dualTowerBf16WeightBytes;

  return Object.freeze({
    prefixTokens,
    blockSize,
    mode,
    precision: 'bf16',
    contextPrefixCache,
    persistentContextMambaBytes,
    persistentContextKvBytes,
    persistentContextTotalBytes,
    kvBytesPerToken,
    ephemeralDenoiserBlockKvBytes,
    mockArMambaS2SnapshotBytes,
    totalActiveCacheBytes,
    denoiserReadOnlyMirrorBytes,
    dualTowerBf16WeightBytes,
    averagePerTowerBf16WeightBytes,
    contextTowerBf16WeightBytes,
    denoiserTowerBf16WeightBytes,
    timeConditioningBf16WeightBytes,
    activeModelWeightBytes,
    communityNvfp4Reference: TWOTOWER_COMMUNITY_NVFP4_REFERENCE,
  });
}
