/**
 * Shared Headless TypeScript Contracts for Nemotron-Labs-Diffusion-Uni (`src/uni/types.ts`)
 *
 * Pinned Step-0 architectural contract for Milestone UNI1 shared across all disjoint slices:
 *   - Slice 1A: `spec.ts`, `prng.ts`
 *   - Slice 1B: `rope.ts`, `masks.ts`, `attention.ts`, `swiglu.ts`
 *   - Slice 1C: `parameters.ts`, `forward.ts`, PyTorch 2.15.0 reference exporter & parity suite
 *
 * Clean-room first-principles specification (arXiv:2610.36466v1 / nvidia/NLD-Uni).
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

// ============================================================================
// MINIATURE TEXT & GROUNDING VOCABULARY (V_txt = 64)
// ============================================================================

/**
 * Miniature 64-token Understanding & Grounding vocabulary (`V_txt = 64`):
 *   - `15` control/special tokens matching production structural IDs where compact:
 *     `0`: `<unk>`, `1`: `<s>` (BOS), `2`: `</s>`, `10`: `<|im_start|>`,
 *     `11`: `<|im_end|>` (EOS/PAD), `12`: `<think>`, `13`: `</think>`,
 *     `18`: `<SPECIAL_18>` (target gen slot), `19`: `<SPECIAL_19>` (source edit VQ slot),
 *     `20`: `<SPECIAL_20>` (FIM reserve), `21`: `<SPECIAL_21>` (gen image start),
 *     `22`: `<SPECIAL_22>` (gen image end), `23`: `<SPECIAL_23>` (RADIO-H image pad),
 *     `24`: `\n` (visual row delimiter), `25`: `[MASK]` (prod `<SPECIAL_100>`, id 100)
 *   - `28` readable educational words (`3..9`, `14..17`, `26..42`)
 *   - `17` normalized grounding coordinate bins (`43..59`: `<LOC_0>` .. `<LOC_16>`)
 *   - `4` grounding delimiter tags (`60..63`: `<box_p>`, `</box_p>`, `<LOC_BEGIN>`, `<LOC_END>`)
 */
export const UNI_MINI_VOCAB: readonly string[] = Object.freeze([
  '<unk>',        // 0
  '<s>',          // 1  (BOS)
  '</s>',         // 2
  'a',            // 3
  'red',          // 4
  'blue',         // 5
  'cat',          // 6
  'dog',          // 7
  'sun',          // 8
  'sky',          // 9
  '<|im_start|>', // 10
  '<|im_end|>',   // 11 (EOS / PAD)
  '<think>',      // 12
  '</think>',     // 13
  'on',           // 14
  'in',           // 15
  'with',         // 16
  'near',         // 17
  '<SPECIAL_18>', // 18 (target generation placeholder)
  '<SPECIAL_19>', // 19 (source-image VQ placeholder in editing)
  '<SPECIAL_20>', // 20 (FIM reserve)
  '<SPECIAL_21>', // 21 (gen_im_start_token)
  '<SPECIAL_22>', // 22 (gen_im_end_token)
  '<SPECIAL_23>', // 23 (und_im_pad_token)
  '\n',           // 24 (visual row separator)
  '[MASK]',       // 25 (prod <SPECIAL_100>, id 100)
  'the',          // 26
  'image',        // 27
  'shows',        // 28
  'generate',     // 29
  'edit',         // 30
  'photo',        // 31
  'tree',         // 32
  'house',        // 33
  'water',        // 34
  'green',        // 35
  'bright',       // 36
  'night',        // 37
  'left',         // 38
  'right',        // 39
  'top',          // 40
  'center',       // 41
  'sure',         // 42
  '<LOC_0>',      // 43
  '<LOC_1>',      // 44
  '<LOC_2>',      // 45
  '<LOC_3>',      // 46
  '<LOC_4>',      // 47
  '<LOC_5>',      // 48
  '<LOC_6>',      // 49
  '<LOC_7>',      // 50
  '<LOC_8>',      // 51
  '<LOC_9>',      // 52
  '<LOC_10>',     // 53
  '<LOC_11>',     // 54
  '<LOC_12>',     // 55
  '<LOC_13>',     // 56
  '<LOC_14>',     // 57
  '<LOC_15>',     // 58
  '<LOC_16>',     // 59
  '<box_p>',      // 60
  '</box_p>',     // 61
  '<LOC_BEGIN>',  // 62
  '<LOC_END>',    // 63
]);

export const UNI_MINI_UNK_TOKEN_ID = 0 as const;
export const UNI_MINI_BOS_TOKEN_ID = 1 as const;
export const UNI_MINI_IM_START_TOKEN_ID = 10 as const;
export const UNI_MINI_EOS_TOKEN_ID = 11 as const;
export const UNI_MINI_PAD_TOKEN_ID = 11 as const;
export const UNI_MINI_THINK_TOKEN_ID = 12 as const;
export const UNI_MINI_END_THINK_TOKEN_ID = 13 as const;
export const UNI_MINI_GEN_SLOT_TOKEN_ID = 18 as const;
export const UNI_MINI_ENC_SLOT_TOKEN_ID = 19 as const;
export const UNI_MINI_FIM_TOKEN_ID = 20 as const;
export const UNI_MINI_GEN_START_TOKEN_ID = 21 as const;
export const UNI_MINI_GEN_END_TOKEN_ID = 22 as const;
export const UNI_MINI_UND_PAD_TOKEN_ID = 23 as const;
export const UNI_MINI_NEWLINE_TOKEN_ID = 24 as const;
export const UNI_MINI_MASK_TOKEN_ID = 25 as const;

export const UNI_MINI_BASE_VOCAB_SIZE = 43 as const;
export const UNI_MINI_LOC_START_ID = 43 as const;
export const UNI_MINI_LOC_END_ID = 59 as const;
export const UNI_MINI_LOC_BIN_COUNT = 17 as const;
export const UNI_MINI_BOX_OPEN_TOKEN_ID = 60 as const;
export const UNI_MINI_BOX_CLOSE_TOKEN_ID = 61 as const;
export const UNI_MINI_LOC_BEGIN_TOKEN_ID = 62 as const;
export const UNI_MINI_LOC_END_TOKEN_ID = 63 as const;

// ============================================================================
// GENERATION VISUAL VOCABULARY & SPECIAL ROW IDS (V_vq = 256, V_gen_emb = 264)
// ============================================================================

export const UNI_MINI_VQ_CODEBOOK_SIZE = 256 as const;
export const UNI_MINI_VQ_TREE_DEPTH = 8 as const;
export const UNI_MINI_GEN_ROW_MARKER_ID = 256 as const;
export const UNI_MINI_GEN_IMAGE_MASK_ID = 257 as const;
export const UNI_MINI_GEN_IMAGE_END_ID = 258 as const;
export const UNI_MINI_GEN_FIXED_PREFIX_START_ID = 259 as const;
export const UNI_MINI_GEN_FIXED_PREFIX_COUNT = 4 as const;
export const UNI_MINI_GEN_SPARE_ROW_ID = 263 as const;
export const UNI_MINI_GEN_RESERVED_ROWS = 8 as const;
export const UNI_MINI_GEN_EMBEDDING_ROWS = 264 as const;

// ============================================================================
// LOCKED SEEDS, TENSOR COUNTS & EXACT DYADIC / GAIN CONSTANTS (D1–D5)
// ============================================================================

export const UNI_BACKBONE_SEED = 0x4e4c5531 as const; // "NLU1"

/**
 * Core Elastic-MoT backbone + U/G IO tensors implemented in UNI1:
 *   - Understanding IO: 3 tensors (1,032 floats)
 *   - Joint Backbone L0..17: 324 tensors (162 U + 162 G = 70,848 floats)
 *   - Split Backbone L18..33: 288 tensors (144 U + 144 G = 62,976 floats)
 *   - Generation IO: 3 tensors (norm_gen, gen_embedding, gen_predictor = 4,168 floats)
 * Total UNI1 Core: 618 tensors = 139,024 floats = 556,096 B.
 */
export const UNI_MINI_CORE_TENSOR_COUNT = 618 as const;
export const UNI_MINI_CORE_FLOAT_COUNT = 139_024 as const;
export const UNI_MINI_CORE_WEIGHT_BYTES = 556_096 as const;

/**
 * Full miniature model counts across UNI1 + UNI2 (matching UNI0 §8.3A):
 *   618 core + 4 PackedLinearImageCodec + 31 (Mini RADIO-H + Projector + image_newline)
 *   + 113 Mini Emu3VQ = 766 tensors = 191,235 floats = 764,940 B (+ 3,068 B tree = 768,008 B).
 */
export const UNI_MINI_FULL_TENSOR_COUNT = 766 as const;
export const UNI_MINI_FULL_FLOAT_COUNT = 191_235 as const;
export const UNI_MINI_FULL_WEIGHT_BYTES = 764_940 as const;
export const UNI_MINI_TREE_BUFFER_BYTES = 3_068 as const;
export const UNI_MINI_TOTAL_RESIDENT_BYTES = 768_008 as const;
export const UNI_MAX_SESSION_RETAINED_BYTES = 12_582_912 as const; // 12.00 MiB

/**
 * Exact 0-ULP dyadic YaRN inverse frequencies `[1, 9/32, 1/32, 1/64]` for
 * `headDim = 8, ropeTheta = 16.0, origMaxPos = 16, factor = 8.0, betaFast = 32.0,
 * betaSlow = 1.0, truncate = true` (`extrap = [1.0, 0.5, 0.0, 0.0]`).
 */
export const UNI_DYADIC_YARN_INV_FREQ: readonly [number, number, number, number] =
  Object.freeze([1.0, 0.28125, 0.03125, 0.015625] as const);

export const UNI_2D_SINCOS_CANONICAL_CONTENT_SIZE = 32.0 as const;
export const UNI_2D_SINCOS_SCALE = 0.0018 as const;

export const UNI_PROD_GEN_EMBEDDING_SCALE = 0.005357294692673304 as const;
export const UNI_PROD_GEN_DOWNSAMPLE_SCALE = 0.4498539166224965 as const;
export const UNI_PROD_RADIO_MULTIPLY = 0.01 as const;

/** D3: Locked dual gain presets (miniature interactive default vs. production-exact parity). */
export type UniGainPresetId = 'mini_default' | 'prod_exact';

export interface UniGainPreset {
  readonly id: UniGainPresetId;
  readonly label: string;
  readonly generationEmbeddingScale: number;
  readonly generationDownsampleScale: number;
  readonly combinedGenerationScale: number;
  readonly radioProjectorMultiply: number;
}

export const UNI_GAIN_PRESETS: Readonly<Record<UniGainPresetId, UniGainPreset>> =
  Object.freeze({
    mini_default: Object.freeze({
      id: 'mini_default',
      label: 'Miniature Well-Conditioned (1.0 / 0.5 & 0.25)',
      generationEmbeddingScale: 1.0,
      generationDownsampleScale: 0.5,
      combinedGenerationScale: 0.5,
      radioProjectorMultiply: 0.25,
    }),
    prod_exact: Object.freeze({
      id: 'prod_exact',
      label: 'Production Exact (0.0053573 / 0.4498539 & 0.01)',
      generationEmbeddingScale: UNI_PROD_GEN_EMBEDDING_SCALE,
      generationDownsampleScale: UNI_PROD_GEN_DOWNSAMPLE_SCALE,
      combinedGenerationScale:
        UNI_PROD_GEN_EMBEDDING_SCALE * UNI_PROD_GEN_DOWNSAMPLE_SCALE,
      radioProjectorMultiply: UNI_PROD_RADIO_MULTIPLY,
    }),
  });

// ============================================================================
// MINIATURE MODEL CONFIGURATION (D1, D2, D3)
// ============================================================================

export interface UniYarnRopeConfig {
  readonly ropeType: 'yarn';
  readonly headDim: number;
  readonly ropeTheta: number;
  readonly originalMaxPositionEmbeddings: number;
  readonly maxPositionEmbeddings: number;
  readonly factor: number;
  readonly betaFast: number;
  readonly betaSlow: number;
  readonly mscale: number;
  readonly mscaleAllDim: number;
  readonly truncate: boolean;
  readonly llama4Beta: number;
}

export interface Uni2dSinCosConfig {
  readonly canonicalContentSize: number;
  readonly scale: number;
  readonly baseTheta: number;
}

export interface UniMiniConfig {
  readonly numLayers: 34;
  readonly dualTowerLayers: 18;
  readonly splitLayers: 16;
  readonly hiddenSize: 8;
  readonly intermediateSize: 28;
  readonly numAttentionHeads: 8;
  readonly numKeyValueHeads: 2;
  readonly kvGroupSize: 4;
  readonly headDim: 8;
  readonly qProjOutDim: 64;
  readonly kvProjOutDim: 16;
  readonly rmsNormEps: 1e-5;
  readonly codecNormEps: 1e-6;
  readonly vocabSize: 64;
  readonly baseVocabSize: 43;
  readonly maskTokenId: 25;
  readonly eosTokenId: 11;
  readonly padTokenId: 11;
  readonly bosTokenId: 1;
  readonly vqCodebookSize: 256;
  readonly vqLatentDim: 8;
  readonly vqTreeDepth: 8;
  readonly genEmbeddingRows: 264;
  readonly genFixedPrefixTokens: 4;
  readonly imageResolution: 16;
  readonly vitPatchSize: 4;
  readonly vqGridSize: 4;
  readonly vqMarkedGridCols: 6;
  readonly packedGridRows: 2;
  readonly packedGridCols: 3;
  readonly defaultActivationSteps: 8;
  readonly supportedActivationSteps: readonly [4, 8, 16];
  readonly defaultRefinementSteps: 4;
  readonly supportedRefinementSteps: readonly [1, 2, 4, 8];
  readonly defaultShift: 5.0;
  readonly kvCapacity: 64;
  readonly backboneSeed: number;
  readonly defaultGainPreset: UniGainPresetId;
  readonly maxSessionRetainedBytes: 12_582_912;
  readonly rope: UniYarnRopeConfig;
  readonly sinCos2d: Uni2dSinCosConfig;
}

export const UNI_MINI_CONFIG: UniMiniConfig = Object.freeze({
  numLayers: 34,
  dualTowerLayers: 18,
  splitLayers: 16,
  hiddenSize: 8,
  intermediateSize: 28,
  numAttentionHeads: 8,
  numKeyValueHeads: 2,
  kvGroupSize: 4,
  headDim: 8,
  qProjOutDim: 64,
  kvProjOutDim: 16,
  rmsNormEps: 1e-5,
  codecNormEps: 1e-6,
  vocabSize: 64,
  baseVocabSize: UNI_MINI_BASE_VOCAB_SIZE,
  maskTokenId: UNI_MINI_MASK_TOKEN_ID,
  eosTokenId: UNI_MINI_EOS_TOKEN_ID,
  padTokenId: UNI_MINI_PAD_TOKEN_ID,
  bosTokenId: UNI_MINI_BOS_TOKEN_ID,
  vqCodebookSize: UNI_MINI_VQ_CODEBOOK_SIZE,
  vqLatentDim: 8,
  vqTreeDepth: UNI_MINI_VQ_TREE_DEPTH,
  genEmbeddingRows: UNI_MINI_GEN_EMBEDDING_ROWS,
  genFixedPrefixTokens: UNI_MINI_GEN_FIXED_PREFIX_COUNT,
  imageResolution: 16,
  vitPatchSize: 4,
  vqGridSize: 4,
  vqMarkedGridCols: 6,
  packedGridRows: 2,
  packedGridCols: 3,
  defaultActivationSteps: 8,
  supportedActivationSteps: Object.freeze([4, 8, 16] as const),
  defaultRefinementSteps: 4,
  supportedRefinementSteps: Object.freeze([1, 2, 4, 8] as const),
  defaultShift: 5.0,
  kvCapacity: 64,
  backboneSeed: UNI_BACKBONE_SEED,
  defaultGainPreset: 'mini_default',
  maxSessionRetainedBytes: UNI_MAX_SESSION_RETAINED_BYTES,
  rope: Object.freeze({
    ropeType: 'yarn',
    headDim: 8,
    ropeTheta: 16.0,
    originalMaxPositionEmbeddings: 16,
    maxPositionEmbeddings: 128,
    factor: 8.0,
    betaFast: 32.0,
    betaSlow: 1.0,
    mscale: 1.0,
    mscaleAllDim: 1.0,
    truncate: true,
    llama4Beta: 0.1,
  }),
  sinCos2d: Object.freeze({
    canonicalContentSize: UNI_2D_SINCOS_CANONICAL_CONTENT_SIZE,
    scale: UNI_2D_SINCOS_SCALE,
    baseTheta: 10_000.0,
  }),
});

// ============================================================================
// ROUTING, EXECUTION VIEW & ELASTIC PREFILL ENUMS (D5, U5, U6, U20)
// ============================================================================

/** Per-token branch assignment: `0` = Understanding (`U`), `1` = Generation (`G`). */
export type UniTokenRoute = 0 | 1;
export const UNI_ROUTE_U: UniTokenRoute = 0;
export const UNI_ROUTE_G: UniTokenRoute = 1;

export type UniBranchId = 'u' | 'g';

export type UniCanonicalRouting = 'u_only' | 'g_only' | 'mixed';

/**
 * D5: Live execution comparison between:
 *   - `'hf_faithful'`: executes all 34 `U` layers during T2I/Edit prompt prefill
 *     (caching 16 layers `L18..33` of `U` prefix KV that split-tower drops).
 *   - `'elastic_pruned'`: prunes causally disconnected `U` computation at `L18..33`
 *     during generation prefill, yielding 0-ULP bit-exact downstream `G` outputs.
 */
export type UniElasticPrefillMode = 'hf_faithful' | 'elastic_pruned';

export type UniExecutionViewId =
  | 'text_understanding'
  | 'visual_understanding'
  | 'object_grounding'
  | 'text_to_image'
  | 'image_edit'
  | 'interleaved_mixed';

export type UniAttentionMaskMode = 'causal' | 'bidirectional' | 'sbd_image' | 'custom';

export type UniMaskExplorerMode =
  | 'causal'
  | 'bidirectional'
  | 'sbd_image'
  | 'split_tower_mixed'
  | 'opd_dual_stream';

export type UniYarnPlaneZone = 'original' | 'blended' | 'interpolated';

export type UniCoreTensorRole =
  | 'embed_tokens'
  | 'input_layernorm'
  | 'q_proj'
  | 'k_proj'
  | 'v_proj'
  | 'o_proj'
  | 'post_attention_layernorm'
  | 'gate_proj'
  | 'up_proj'
  | 'down_proj'
  | 'final_norm'
  | 'diffusion_head'
  | 'final_norm_gen'
  | 'gen_embedding'
  | 'gen_predictor';

// ============================================================================
// CANONICAL CHECKPOINT TENSOR NAMING HELPERS
// ============================================================================

export const UNI_EMBED_TENSOR_NAME = 'encoder.embed_tokens.weight' as const;
export const UNI_FINAL_NORM_TENSOR_NAME = 'encoder.norm.weight' as const;
export const UNI_DIFFUSION_HEAD_TENSOR_NAME = 'diffusion_head.weight' as const;
export const UNI_FINAL_NORM_GEN_TENSOR_NAME = 'encoder.norm_gen.weight' as const;
export const UNI_GEN_EMBEDDING_TENSOR_NAME = 'encoder.gen_embedding.weight' as const;
export const UNI_GEN_PREDICTOR_TENSOR_NAME = 'encoder.gen_predictor.weight' as const;
export const UNI_IMAGE_NEWLINE_TENSOR_NAME = 'encoder.image_newline' as const;

export function uniLayerNormWeightName(
  layer: number,
  kind: 'input' | 'post_attention',
  branch: UniBranchId,
): string {
  const suffix = branch === 'g' ? '_gen' : '';
  return `encoder.layers.${layer}.${kind}_layernorm${suffix}.weight`;
}

export function uniLayerAttnWeightName(
  layer: number,
  proj: 'q_proj' | 'k_proj' | 'v_proj' | 'o_proj',
  branch: UniBranchId,
): string {
  const suffix = branch === 'g' ? '_gen' : '';
  return `encoder.layers.${layer}.self_attn.${proj}${suffix}.weight`;
}

export function uniLayerMlpWeightName(
  layer: number,
  proj: 'gate_proj' | 'up_proj' | 'down_proj',
  branch: UniBranchId,
): string {
  const suffix = branch === 'g' ? '_gen' : '';
  return `encoder.layers.${layer}.mlp.${proj}${suffix}.weight`;
}

// ============================================================================
// PRNG INTERFACES (Slice 1A: `prng.ts`)
// ============================================================================

export type UniXoshiro128ssState = readonly [number, number, number, number];

export interface UniXoshiro128ssGenerator {
  nextUint32(): number;
  nextFloat32(): number;
  nextFloat64(): number;
  /** Strictly open-interval uniform sample in `(0, 1)` to prevent `log(0)` underflow (U22). */
  nextUniformOpenUnit(): number;
  getState(): UniXoshiro128ssState;
  clone(): UniXoshiro128ssGenerator;
}

// ============================================================================
// SPEC, 8-SHARD LFS INVENTORY & U1–U30 LEDGER INTERFACES (Slice 1A: `spec.ts`)
// ============================================================================

export interface UniCheckpointShardEntry {
  readonly file: string;
  readonly shardIndex: number;
  readonly totalShards: number;
  readonly tensorCount: number;
  readonly headerJsonBytes: number;
  readonly headerFramingBytes: number; // 8 + headerJsonBytes
  readonly payloadBytes: number;
  readonly totalFileBytes: number;
  readonly diffBytes: number;
  readonly sha256Prefix: string;
}

export interface UniLfsReconciliationSummary {
  readonly shardCount: number;
  readonly totalTensors: number;
  readonly totalElements: number;
  readonly totalHeaderFramingBytes: number;
  readonly totalPayloadBytes: number;
  readonly totalFileBytes: number;
  readonly indexTotalSizeBytes: number;
  readonly payloadDiffBytes: number;
  readonly fileDiffBytes: number;
  readonly allZeroDiff: boolean;
}

export interface UniAuxiliaryLfsEntry {
  readonly file: string;
  readonly sizeBytes: number;
  readonly sha256Prefix: string;
  readonly role: string;
  readonly redistributed: false;
}

export interface UniComponentBreakdownEntry {
  readonly id: string;
  readonly label: string;
  readonly tensorCount: number;
  readonly elementCount: number;
  readonly dtype: 'BF16' | 'BF16+I64';
  readonly payloadBytes: number;
  readonly branch: 'u' | 'g' | 'shared_io' | 'vision' | 'vqvae' | 'dead';
  readonly unusedInForward?: boolean;
}

export interface UniExecutionViewSpec {
  readonly id: UniExecutionViewId;
  readonly label: string;
  readonly paperShorthand: string;
  readonly decoderBlocksSummary: string;
  readonly decoderBlockParams: number;
  readonly activeCausalGraphParams: number;
  readonly eagerHfExecutedParams: number;
  readonly wastedSplitUParams: number;
  readonly residentHfBytes: number;
  readonly residentSglangBytes: number | null;
  readonly cfgBranches: 1 | 2 | 3;
  readonly kvBytesPerTokenBf16: number;
  readonly releasedEndToEndLoop: boolean;
  readonly badge?: string;
}

export interface UniContextTierSpec {
  readonly tier: number;
  readonly label: string;
  readonly tokens: number;
  readonly source: string;
  readonly notes: string;
}

export interface UniDiscrepancyEntry {
  readonly id: `U${number}`;
  readonly numericId: number;
  readonly topic: string;
  readonly category:
    | 'provenance_license'
    | 'architecture_elastic'
    | 'diffusion_tree'
    | 'multimodal_codec'
    | 'browser_guard'
    | 'training_serving';
  readonly isBrowserGuard: boolean;
  readonly paperOrConfigClaim: string;
  readonly verifiedGroundTruth: string;
  readonly browserResolution: string;
  readonly epistemicStatus: 'VERIFIED' | 'DERIVED';
}

export interface UniMiniComponentFootprintEntry {
  readonly id: string;
  readonly label: string;
  readonly milestone: 'UNI1' | 'UNI2' | 'UNI3';
  readonly tensorCount: number;
  readonly floatCount: number;
  readonly bytes: number;
  readonly unusedInForward?: boolean;
}

export interface UniAtlasBudgetSpec {
  readonly viewId: string;
  readonly label: string;
  readonly width: number;
  readonly height: number;
  readonly bytes: number;
  readonly rectCount: number;
  readonly scalarMaskOnly: true;
}

// ============================================================================
// ROPE, LLAMA-4 SCALE & 2D SINCOS INTERFACES (Slice 1B: `rope.ts`)
// ============================================================================

export interface UniYarnInvFreqResult {
  readonly invFreq: Float32Array;
  readonly attentionScaling: number;
  readonly low: number;
  readonly high: number;
  readonly extrapolationFactor: Float32Array;
  readonly zones: readonly UniYarnPlaneZone[];
}

export interface UniRopeCosSin {
  readonly cos: Float32Array;
  readonly sin: Float32Array;
  readonly seqLen: number;
  readonly headDim: number;
}

export interface Uni2dSinCosResult {
  /** Row-major `[height * width, hiddenSize]` Float32 additive spatial embedding. */
  readonly embeddings: Float32Array;
  readonly height: number;
  readonly width: number;
  readonly contentWidth: number;
  readonly hiddenSize: number;
  readonly markerColumn: boolean;
  readonly columnStride: number;
  readonly rowStride: number;
  readonly frequencies: Float32Array;
}

// ============================================================================
// MASK & ROUTING PLAN INTERFACES (Slice 1B: `masks.ts`)
// ============================================================================

export interface UniTokenRoutingPlan {
  readonly routing: Uint8Array;
  readonly canonical: UniCanonicalRouting;
  readonly understandingIndices: Int32Array;
  readonly generationIndices: Int32Array;
  readonly hasUnderstanding: boolean;
  readonly hasGeneration: boolean;
}

export interface UniAttentionMaskMatrix {
  readonly mode: UniMaskExplorerMode;
  readonly qLen: number;
  readonly kvLen: number;
  /** Row-major `[qLen * kvLen]` binary visibility (`1` = visible, `0` = masked). */
  readonly visible: Uint8Array;
  readonly splitTowerApplied: boolean;
  readonly layerIndex: number | null;
  readonly isReadOnlyReference: boolean;
  readonly badge: string;
}

// ============================================================================
// ATTENTION, SWIGLU & LAYER WEIGHT/TRACE INTERFACES (Slices 1B & 1C)
// ============================================================================

export interface UniBranchAttentionWeights {
  /** `[qProjOutDim, hiddenSize]` = `[64, 8]`. */
  readonly qProj: Float32Array;
  /** `[kvProjOutDim, hiddenSize]` = `[16, 8]`. */
  readonly kProj: Float32Array;
  /** `[kvProjOutDim, hiddenSize]` = `[16, 8]`. */
  readonly vProj: Float32Array;
  /** `[hiddenSize, qProjOutDim]` = `[8, 64]`. */
  readonly oProj: Float32Array;
}

export interface UniBranchSwigluWeights {
  /** `[intermediateSize, hiddenSize]` = `[28, 8]`. */
  readonly gateProj: Float32Array;
  /** `[intermediateSize, hiddenSize]` = `[28, 8]`. */
  readonly upProj: Float32Array;
  /** `[hiddenSize, intermediateSize]` = `[8, 28]`. */
  readonly downProj: Float32Array;
}

export interface UniBranchLayerWeights {
  /** `[hiddenSize]` = `[8]`. */
  readonly inputLayerNorm: Float32Array;
  readonly selfAttn: UniBranchAttentionWeights;
  /** `[hiddenSize]` = `[8]`. */
  readonly postAttentionLayerNorm: Float32Array;
  readonly mlp: UniBranchSwigluWeights;
}

export interface UniElasticLayerWeights {
  readonly layerIndex: number;
  /** `false` for layers `0..17` (joint), `true` for layers `18..33` (split-tower). */
  readonly splitTower: boolean;
  /** Understanding branch weights (`r = 0`). */
  readonly u: UniBranchLayerWeights;
  /** Generation branch weights (`r = 1`, `_gen` suffix in checkpoint). */
  readonly g: UniBranchLayerWeights;
}

export interface UniLayerKvSlice {
  /** `[seqLen, kvProjOutDim]` = `[seqLen, 16]` post-RoPE keys. */
  readonly k: Float32Array;
  /** `[seqLen, kvProjOutDim]` = `[seqLen, 16]` values. */
  readonly v: Float32Array;
  readonly seqLen: number;
}

export interface UniElasticAttentionInput {
  /** Pre-normed `[seqLen, hiddenSize]` routed inputs. */
  readonly hiddenStates: Float32Array;
  readonly seqLen: number;
  readonly routing: Uint8Array;
  readonly layerWeights: UniElasticLayerWeights;
  readonly config: UniMiniConfig;
  readonly invFreq: Float32Array;
  readonly attentionScaling: number;
  readonly positionIds: Int32Array;
  readonly cachePositions?: Int32Array;
  readonly maskMode: UniAttentionMaskMode;
  readonly customMask?: Uint8Array;
  /** Required when `maskMode === 'sbd_image'` on a single unified sequence. */
  readonly imageBlockStart?: number;
  readonly imageBlockEnd?: number;
  readonly useCache: boolean;
  /** Read-only cached K/V slice for this layer (`null` allowed at `L18..33` when pruned/dropped). */
  readonly pastKv?: UniLayerKvSlice | null;
  /** Read-only cached token routing (`[pastLen]`) aligned with `pastKv`. */
  readonly pastRouting?: Uint8Array | null;
  /** D5: When true on a split layer (`>= 18`), skip computing `U` query rows (`r = 0`). */
  readonly pruneUnderstandingTokens?: boolean;
}

export interface UniElasticAttentionTrace {
  readonly layerIndex: number;
  readonly splitTower: boolean;
  readonly seqLen: number;
  readonly pastLen: number;
  readonly kvLen: number;
  readonly canonicalRouting: UniCanonicalRouting;
  readonly cachedHasGeneration: boolean;
  readonly dropCachedUnderstanding: boolean;
  readonly understandingPruned: boolean;
  /** `[seqLen, qProjOutDim]` pre-RoPE Q. */
  readonly qPre: Float32Array;
  /** `[seqLen, kvProjOutDim]` pre-RoPE K. */
  readonly kPre: Float32Array;
  /** `[seqLen, kvProjOutDim]` V. */
  readonly vNew: Float32Array;
  /** `[seqLen, qProjOutDim]` post-RoPE Q. */
  readonly qRot: Float32Array;
  /** `[seqLen, kvProjOutDim]` post-RoPE K. */
  readonly kRot: Float32Array;
  /** `[seqLen]` per-token Llama-4 query temperature scale. */
  readonly llama4Scales: Float32Array;
  /** `[seqLen, qProjOutDim]` post-RoPE Llama-4 scaled Q. */
  readonly qScaled: Float32Array;
  /** `[seqLen * kvLen]` final binary mask after joint/split-tower routing. */
  readonly effectiveMask: Uint8Array;
  /** `[nQ * seqLen * kvLen]` scaled dot-product logits (`-Infinity` where masked). */
  readonly scores: Float32Array;
  /** `[nQ * seqLen * kvLen]` softmax probabilities (`0.0` where masked). */
  readonly probs: Float32Array;
  /** `[seqLen, qProjOutDim]` concatenated head outputs before `o_proj`. */
  readonly attnConcat: Float32Array;
  /** `[seqLen, hiddenSize]` routed `o_proj` / `o_proj_gen` output. */
  readonly output: Float32Array;
}

export interface UniSwigluTrace {
  readonly gatePre: Float32Array;
  readonly gateSilu: Float32Array;
  readonly upVal: Float32Array;
  readonly gated: Float32Array;
  readonly output: Float32Array;
  readonly seqLen: number;
  readonly d: number;
  readonly dFf: number;
}

export interface UniLayerTrace {
  readonly layerIndex: number;
  readonly splitTower: boolean;
  readonly skipped: boolean;
  readonly understandingPruned: boolean;
  readonly inputNorm: Float32Array;
  readonly attention: UniElasticAttentionTrace;
  readonly postAttentionResidual: Float32Array;
  readonly postAttentionNorm: Float32Array;
  readonly swiglu: UniSwigluTrace;
  readonly output: Float32Array;
}

// ============================================================================
// CORE MODEL PARAMETERS & FORWARD PASS INTERFACES (Slice 1C)
// ============================================================================

export interface UniCanonicalWeightTensor {
  readonly name: string;
  readonly role: UniCoreTensorRole;
  readonly branch: 'u' | 'g';
  readonly layerIndex: number | null;
  readonly splitTower: boolean;
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

export interface UniCoreModelParameters {
  readonly modelId: string;
  readonly backboneSeed: number;
  readonly backboneFingerprint: string;
  readonly config: UniMiniConfig;
  readonly vocab: readonly string[];
  /** `encoder.embed_tokens.weight`: `[64, 8]`. */
  readonly embedTokens: Float32Array;
  /** `34` dual-branch Elastic-MoT layers (`0..17` joint, `18..33` split). */
  readonly layers: readonly UniElasticLayerWeights[];
  /** `encoder.norm.weight`: `[8]`. */
  readonly finalNorm: Float32Array;
  /** `diffusion_head.weight`: `[64, 8]`. */
  readonly diffusionHead: Float32Array;
  /** `encoder.norm_gen.weight`: `[8]`. */
  readonly finalNormGen: Float32Array;
  /** `encoder.gen_embedding.weight`: `[264, 8]`. */
  readonly genEmbedding: Float32Array;
  /** `encoder.gen_predictor.weight`: `[256, 8]`. */
  readonly genPredictor: Float32Array;
}

export interface CreateUniCoreModelOptions {
  readonly modelId?: string;
  readonly backboneSeed?: number;
}

export interface UniElasticExecutionTelemetry {
  readonly prefillMode: UniElasticPrefillMode;
  readonly canonicalRouting: UniCanonicalRouting;
  readonly totalLayers: number;
  readonly executedLayerCount: number;
  readonly skippedLayerIndices: readonly number[];
  readonly understandingTokensPrunedAtSplit: number;
  readonly wastedPrefixKvLayers: number;
  readonly droppedCachedUnderstandingLayers: readonly number[];
}

export interface UniForwardOptions {
  /** Provide one of `inputsEmbeds`, `tokenIds` (text `U`), or `genTokenIds` (visual `G`). */
  readonly inputsEmbeds?: Float32Array;
  readonly tokenIds?: ArrayLike<number>;
  readonly genTokenIds?: ArrayLike<number>;
  /** Required when `inputsEmbeds` is provided; defaults to all `U` for `tokenIds` or all `G` for `genTokenIds`. */
  readonly routing?: ArrayLike<number>;
  readonly maskMode: UniAttentionMaskMode;
  readonly customMask?: Uint8Array;
  readonly imageBlockStart?: number;
  readonly imageBlockEnd?: number;
  readonly useCache: boolean;
  /** Optional D5 prefill mode (default `'hf_faithful'`). */
  readonly prefillMode?: UniElasticPrefillMode;
  /** Read-only per-layer past KV slices (`length === 34`; `L18..33` may be `null` under pruned T2I). */
  readonly pastKvs?: readonly (UniLayerKvSlice | null)[] | null;
  /** Read-only cached token routing (`length === pastLen`) paired with `pastKvs`. */
  readonly pastRouting?: ArrayLike<number> | null;
  /** Optional explicit 1D RoPE position IDs (`length === seqLen`). Defaults to `[pastLen .. pastLen + seqLen - 1]`. */
  readonly positionIds?: ArrayLike<number>;
  /** Optional explicit Llama-4 cache positions (`length === seqLen`). Defaults to `positionIds`. */
  readonly cachePositions?: ArrayLike<number>;
  /** Optional subset of layer indices to retain detailed `layerTraces` for (retains all executed layers if omitted). */
  readonly retainedTraceLayers?: ReadonlySet<number>;
  /** Whether to compute text `diffusion_head` logits/probs (default: true if any `U` token present). */
  readonly computeTextHead?: boolean;
  /** Whether to compute visual `gen_predictor` logits/probs (default: true if any `G` token present). */
  readonly computeGenHead?: boolean;
}

export interface UniForwardResult {
  readonly seqLen: number;
  readonly pastLen: number;
  readonly kvLen: number;
  readonly routing: Uint8Array;
  readonly canonicalRouting: UniCanonicalRouting;
  readonly maskMode: UniAttentionMaskMode;
  readonly useCache: boolean;
  readonly prefillMode: UniElasticPrefillMode;
  readonly positionIds: Int32Array;
  readonly cachePositions: Int32Array;
  readonly embeddings: Float32Array;
  /** `35` checkpoints `[h_0, h_1, ..., h_34]` (`null` for `h_{l+1}` when layer `l` was fully skipped in `elastic_pruned`). */
  readonly residualCheckpoints: readonly (Float32Array | null)[];
  readonly layerTraces: ReadonlyMap<number, UniLayerTrace>;
  /** `34` newly computed KV slices (`null` for layers skipped in `elastic_pruned`). */
  readonly newKvs: readonly (UniLayerKvSlice | null)[];
  /** Routed final RMSNorm output (`encoder.norm` for `U`, `encoder.norm_gen` for `G`), or `null` if `L33` was fully skipped. */
  readonly finalNorm: Float32Array | null;
  /** `[seqLen, 64]` raw un-suppressed text logits (when computed). */
  readonly textLogits: Float32Array | null;
  /** `[seqLen, 64]` softmax text probabilities (when computed). */
  readonly textProbs: Float32Array | null;
  /** `[seqLen, 256]` raw visual VQ logits (when computed). */
  readonly genLogits: Float32Array | null;
  /** `[seqLen, 256]` softmax visual VQ probabilities (when computed). */
  readonly genProbs: Float32Array | null;
  readonly telemetry: UniElasticExecutionTelemetry;
}
