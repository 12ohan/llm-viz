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
  /**
   * Optional task-aware downstream consumer hint (`'u'` for Understanding/Grounding where
   * `L18..33` `U` KVs are read; `'g'` for T2I/Editing where split-tower `G` drops `L18..33` `U` KVs).
   */
  readonly downstreamTarget?: 'u' | 'g';
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


// ============================================================================
// UNI2 MULTIMODAL CONSTANTS, EXACT TENSOR COUNTS & SEEDS (D2, D3, U8, U13–U18)
// ============================================================================

export const UNI_MULTIMODAL_SEED = 0x4e4c5532 as const; // "NLU2"

export const UNI_CLIP_MEAN: readonly [number, number, number] = Object.freeze([
  0.48145466,
  0.4578275,
  0.40821073,
] as const);

export const UNI_CLIP_STD: readonly [number, number, number] = Object.freeze([
  0.26862954,
  0.26130258,
  0.27577711,
] as const);

// Miniature RADIO-H ViT-2L + RadioHProjector dimensions
export const UNI_MINI_VIT_EMBED_DIM = 16 as const;
export const UNI_MINI_VIT_NUM_HEADS = 4 as const;
export const UNI_MINI_VIT_HEAD_DIM = 4 as const;
export const UNI_MINI_VIT_MLP_DIM = 64 as const;
export const UNI_MINI_VIT_NUM_LAYERS = 2 as const;
export const UNI_MINI_VIT_CLS_TOKENS = 2 as const;
export const UNI_MINI_VIT_REG_TOKENS = 2 as const;
export const UNI_MINI_VIT_PREFIX_TOKENS = 4 as const;
export const UNI_MINI_VIT_PATCH_SIZE = 4 as const;
export const UNI_MINI_VIT_PATCH_DIM = 48 as const; // 3 * 4 * 4
export const UNI_MINI_VIT_BASE_GRID = 4 as const; // 4x4 = 16 base patches for 16x16
export const UNI_MINI_VIT_LAYER_NORM_EPS = 1e-6 as const;
export const UNI_MINI_PROJECTOR_IN_DIM = 64 as const; // 2x2 * 16
export const UNI_MINI_PROJECTOR_NORM_EPS = 1e-5 as const;

// Miniature Emu3VQ dimensions (2 down/up stages: 16x16 <-> 8x8 <-> 4x4)
export const UNI_MINI_VQ_BASE_CH = 8 as const;
export const UNI_MINI_VQ_MID_CH = 16 as const;
export const UNI_MINI_VQ_Z_CH = 8 as const;
export const UNI_MINI_VQ_NUM_GROUPS = 4 as const;
export const UNI_MINI_VQ_GROUP_NORM_EPS = 1e-6 as const;

// Exact UNI2 tensor & float counts reconciling UNI1 (618 / 139,024) + UNI2 (148 / 52,211) = 766 / 191,235
export const UNI_MINI_VISION_TOWER_TENSOR_COUNT = 27 as const;
export const UNI_MINI_VISION_TOWER_FLOAT_COUNT = 7_648 as const;
export const UNI_MINI_PROJECTOR_TENSOR_COUNT = 3 as const;
export const UNI_MINI_PROJECTOR_FLOAT_COUNT = 640 as const;
export const UNI_MINI_IMAGE_NEWLINE_TENSOR_COUNT = 1 as const;
export const UNI_MINI_IMAGE_NEWLINE_FLOAT_COUNT = 8 as const;
export const UNI_MINI_VISION_SUBSYSTEM_TENSOR_COUNT = 31 as const;
export const UNI_MINI_VISION_SUBSYSTEM_FLOAT_COUNT = 8_296 as const;

export const UNI_MINI_CODEC_TENSOR_COUNT = 4 as const;
export const UNI_MINI_CODEC_FLOAT_COUNT = 528 as const;

export const UNI_MINI_VQVAE_CODEBOOK_TENSOR_COUNT = 1 as const;
export const UNI_MINI_VQVAE_CODEBOOK_FLOAT_COUNT = 2_048 as const;
export const UNI_MINI_VQVAE_ENCODER_TENSOR_COUNT = 56 as const;
export const UNI_MINI_VQVAE_ENCODER_FLOAT_COUNT = 19_800 as const;
export const UNI_MINI_VQVAE_DECODER_TENSOR_COUNT = 56 as const;
export const UNI_MINI_VQVAE_DECODER_FLOAT_COUNT = 21_539 as const;
export const UNI_MINI_VQVAE_TOTAL_TENSOR_COUNT = 113 as const;
export const UNI_MINI_VQVAE_TOTAL_FLOAT_COUNT = 43_387 as const;

export const UNI_MINI_MULTIMODAL_TENSOR_COUNT = 148 as const;
export const UNI_MINI_MULTIMODAL_FLOAT_COUNT = 52_211 as const;
export const UNI_MINI_MULTIMODAL_WEIGHT_BYTES = 208_844 as const;

export type UniMultimodalSubsystem =
  | 'vision_tower'
  | 'mm_projector'
  | 'image_newline'
  | 'packed_image_codec'
  | 'vqvae_codebook'
  | 'vqvae_encoder'
  | 'vqvae_decoder';

export interface UniMultimodalTensorEntry {
  readonly name: string;
  readonly subsystem: UniMultimodalSubsystem;
  readonly shape: readonly number[];
  readonly data: Float32Array;
  readonly unusedInForward?: boolean;
}

// ============================================================================
// SLICE 2A: PROCEDURAL RGB SCENES, MINI RADIO-H VISION & PROJECTOR (`vision.ts`)
// ============================================================================

export type UniScenePresetId =
  | 'sunlit_cat_16x16'
  | 'harbor_panorama_16x8'
  | 'geometric_blocks_16x16';

export interface UniSceneGroundTruthBox {
  readonly tag: string;
  /** `[x1, y1, x2, y2]` in source pixel coordinates. */
  readonly pixelBox: readonly [number, number, number, number];
}

export interface UniRgbImage {
  readonly presetId?: UniScenePresetId;
  readonly label?: string;
  readonly width: number;
  readonly height: number;
  /** Row-major CHW `[3, height, width]` Float32 pixels in `[0, 1]`. */
  readonly data: Float32Array;
  readonly groundTruthBoxes?: readonly UniSceneGroundTruthBox[];
}

export interface UniLetterboxInfo {
  readonly sourceWidth: number;
  readonly sourceHeight: number;
  readonly targetSize: number;
  readonly scale: number;
  readonly resizedWidth: number;
  readonly resizedHeight: number;
  readonly padLeft: number;
  readonly padTop: number;
  readonly padRight: number;
  readonly padBottom: number;
  readonly exactIntegerCentered: boolean;
}

export interface UniLetterboxResult {
  readonly image: UniRgbImage;
  readonly info: UniLetterboxInfo;
}

export interface UniRadioVitBlockWeights {
  readonly blockIndex: number;
  /** `[16]` */
  readonly norm1Weight: Float32Array;
  /** `[16]` */
  readonly norm1Bias: Float32Array;
  /** `[48, 16]` */
  readonly qkvWeight: Float32Array;
  /** `[48]` */
  readonly qkvBias: Float32Array;
  /** `[16, 16]` */
  readonly projWeight: Float32Array;
  /** `[16]` */
  readonly projBias: Float32Array;
  /** `[16]` */
  readonly norm2Weight: Float32Array;
  /** `[16]` */
  readonly norm2Bias: Float32Array;
  /** `[64, 16]` */
  readonly fc1Weight: Float32Array;
  /** `[64]` */
  readonly fc1Bias: Float32Array;
  /** `[16, 64]` */
  readonly fc2Weight: Float32Array;
  /** `[16]` */
  readonly fc2Bias: Float32Array;
}

export interface UniRadioVisionTowerWeights {
  /** `encoder.vision_tower.patch_generator.embedder.weight`: `[16, 48]` (bias-free). */
  readonly patchEmbedWeight: Float32Array;
  /** `encoder.vision_tower.pos_embed`: `[16, 16]` (`4x4` base patch grid). */
  readonly posEmbed: Float32Array;
  /** `encoder.vision_tower.cls_token`: `[4, 16]` (`2` CLS + `2` register tokens). */
  readonly clsToken: Float32Array;
  /** `2` timm ViT blocks (`24` tensors). */
  readonly blocks: readonly UniRadioVitBlockWeights[];
}

export interface UniRadioProjectorWeights {
  /** `encoder.mm_projector.norm.weight`: `[64]` (`RMSNorm(64, eps=1e-5)`). */
  readonly normWeight: Float32Array;
  /** `encoder.mm_projector.fc1.weight`: `[8, 64]` (bias-free). */
  readonly fc1Weight: Float32Array;
  /** `encoder.mm_projector.fc2.weight`: `[8, 8]` (bias-free). */
  readonly fc2Weight: Float32Array;
}

export interface UniVisionSubsystemWeights {
  readonly seed: number;
  readonly visionTower: UniRadioVisionTowerWeights;
  readonly projector: UniRadioProjectorWeights;
  /** `encoder.image_newline`: `[8]` — dead parameter (`unusedInForward: true`, `U8`). */
  readonly imageNewline: Float32Array;
}

export interface UniRadioVitBlockTrace {
  readonly blockIndex: number;
  readonly seqLen: number;
  readonly norm1: Float32Array;
  /** `[numHeads, seqLen, seqLen]` bidirectional ViT attention weights. */
  readonly attnProbs: Float32Array;
  readonly postAttn: Float32Array;
  readonly norm2: Float32Array;
  readonly output: Float32Array;
}

export interface UniRadioVisionTrace {
  readonly patchGridRows: number;
  readonly patchGridCols: number;
  readonly numPatches: number;
  readonly numPrefixTokens: number;
  /** `[3, H, W]` CLIP-normalized image. */
  readonly clipNormalized: Float32Array;
  /** `[numPatches, 48]` flattened CHW `4x4` patches. */
  readonly rawPatches: Float32Array;
  /** `[numPatches, 16]` interpolated/native positional embedding. */
  readonly interpolatedPosEmbed: Float32Array;
  /** `[numPrefixTokens + numPatches, 16]` tokens entering block 0. */
  readonly vitInputTokens: Float32Array;
  readonly blockTraces: readonly UniRadioVitBlockTrace[];
  /** `[numPatches, 16]` spatial patch outputs after `[:, 4:]` prefix strip. */
  readonly patchFeatures: Float32Array;
}

export interface UniRadioProjectorTrace {
  readonly gainPresetId: UniGainPresetId;
  readonly radioMultiply: number;
  readonly tokenGridRows: number;
  readonly tokenGridCols: number;
  readonly numVisualTokens: number;
  /** `[numVisualTokens, 64]` `2x2` `ps_version='v2'` gathered features. */
  readonly gathered2x2: Float32Array;
  /** `[numVisualTokens, 64]` FP32 `RMSNorm(64, eps=1e-5)` output. */
  readonly normed: Float32Array;
  /** `[numVisualTokens, 8]` after `fc1`. */
  readonly fc1Out: Float32Array;
  /** `[numVisualTokens, 8]` after `SquaredReLU(max(0, x)^2)`. */
  readonly squaredReluOut: Float32Array;
  /** `[numVisualTokens, 8]` after `fc2` (pre-gain). */
  readonly fc2Out: Float32Array;
  /** `[numVisualTokens, 8]` final projected embeddings (`fc2Out * radioMultiply`). */
  readonly visualEmbeddings: Float32Array;
}

export interface UniVisionPipelineTrace {
  readonly vision: UniRadioVisionTrace;
  readonly projector: UniRadioProjectorTrace;
  /** Confirms `encoder.image_newline` is never read in forward (`U8`). */
  readonly imageNewlineUnused: true;
}

export interface UniFixedPromptResult {
  readonly originalTokenIds: Int32Array;
  readonly expandedTokenIds: Int32Array;
  readonly visualSlotIndices: Int32Array;
  readonly newlineIndices: Int32Array;
  readonly tokenGridRows: number;
  readonly tokenGridCols: number;
  readonly usedDeadImageNewline: false;
}

// ============================================================================
// SLICE 2B: MINI EMU3 VQ-VAE & PACKED LINEAR IMAGE CODEC (`vqvae.ts`, `codec.ts`)
// ============================================================================

export interface UniConv2dWeights {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly kernelSize: 1 | 3;
  /** `[outChannels, inChannels, kernelSize, kernelSize]` */
  readonly weight: Float32Array;
  /** `[outChannels]` */
  readonly bias: Float32Array;
}

export interface UniGroupNormWeights {
  readonly numChannels: number;
  readonly numGroups: number;
  readonly weight: Float32Array;
  readonly bias: Float32Array;
}

export interface UniEmu3ResBlockWeights {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly norm1: UniGroupNormWeights;
  readonly conv1: UniConv2dWeights;
  readonly norm2: UniGroupNormWeights;
  readonly conv2: UniConv2dWeights;
  /** Present iff `inChannels !== outChannels` (`1x1` conv). */
  readonly ninShortcut: UniConv2dWeights | null;
}

export interface UniEmu3AttnBlockWeights {
  readonly channels: number;
  readonly norm: UniGroupNormWeights;
  readonly q: UniConv2dWeights;
  readonly k: UniConv2dWeights;
  readonly v: UniConv2dWeights;
  readonly projOut: UniConv2dWeights;
}

export interface UniEmu3EncoderWeights {
  readonly convIn: UniConv2dWeights;       // 3 -> 8, 3x3
  readonly down0Block: UniEmu3ResBlockWeights; // 8 -> 8
  readonly down0Conv: UniConv2dWeights;    // 8 -> 8, 3x3 stride 2 (asym pad 0,1,0,1)
  readonly down1Block: UniEmu3ResBlockWeights; // 8 -> 16 (with ninShortcut)
  readonly down1Conv: UniConv2dWeights;    // 16 -> 16, 3x3 stride 2 (asym pad 0,1,0,1)
  readonly midBlock1: UniEmu3ResBlockWeights;  // 16 -> 16
  readonly midAttn1: UniEmu3AttnBlockWeights;  // 16
  readonly midBlock2: UniEmu3ResBlockWeights;  // 16 -> 16
  readonly normOut: UniGroupNormWeights;   // 16
  readonly convOut: UniConv2dWeights;      // 16 -> 8, 3x3
  readonly quantConv: UniConv2dWeights;    // 8 -> 8, 1x1
}

export interface UniEmu3DecoderWeights {
  readonly postQuantConv: UniConv2dWeights;    // 8 -> 8, 1x1
  readonly convIn: UniConv2dWeights;           // 8 -> 16, 3x3
  readonly midBlock1: UniEmu3ResBlockWeights;  // 16 -> 16
  readonly midAttn1: UniEmu3AttnBlockWeights;  // 16
  readonly midBlock2: UniEmu3ResBlockWeights;  // 16 -> 16
  readonly up1Block: UniEmu3ResBlockWeights;   // 16 -> 16
  readonly up1Conv: UniConv2dWeights;          // 16 -> 16, 3x3 (after 2x nearest)
  readonly up0Block: UniEmu3ResBlockWeights;   // 16 -> 8 (with ninShortcut)
  readonly up0Conv: UniConv2dWeights;          // 8 -> 8, 3x3 (after 2x nearest)
  readonly normOut: UniGroupNormWeights;       // 8
  readonly convOut: UniConv2dWeights;          // 8 -> 3, 3x3
}

export interface UniEmu3VqWeights {
  readonly seed: number;
  /** `encoder.vqvae.quantize.embedding.weight`: `[256, 8]` IBQ codebook. */
  readonly codebook: Float32Array;
  readonly encoder: UniEmu3EncoderWeights;
  readonly decoder: UniEmu3DecoderWeights;
}

export interface UniVqEncodeTrace {
  readonly inputWidth: number;
  readonly inputHeight: number;
  readonly vqGridRows: number;
  readonly vqGridCols: number;
  /** `[3, H, W]` mapped to `[-1, 1]`. */
  readonly normalizedInput: Float32Array;
  /** `[8, vqGridRows, vqGridCols]` continuous encoder latents after `quant_conv`. */
  readonly zContinuous: Float32Array;
  /** `[vqGridRows * vqGridCols, 256]` exact IBQ dot-product scores `z @ E^T` (`U14`). */
  readonly dotScores: Float32Array;
  /** `[vqGridRows * vqGridCols]` quantized indices `argmax(z @ E^T)` (`U14`). */
  readonly codes: Int32Array;
  /** `[8, vqGridRows, vqGridCols]` looked-up codebook embeddings. */
  readonly zQuantized: Float32Array;
}

export interface UniVqDecodeTrace {
  readonly vqGridRows: number;
  readonly vqGridCols: number;
  readonly outputWidth: number;
  readonly outputHeight: number;
  readonly codes: Int32Array;
  /** `[8, vqGridRows, vqGridCols]` codebook embeddings. */
  readonly zQuantized: Float32Array;
  /** `[3, outputHeight, outputWidth]` raw decoder output before `clamp((x + 1) * 0.5, 0, 1)`. */
  readonly rawOutput: Float32Array;
  /** Clamped `[0, 1]` CHW RGB image. */
  readonly rgbImage: UniRgbImage;
}

export interface UniPackedCodecWeights {
  readonly seed: number;
  /** `encoder.packed_image_codec.down_norm.weight`: `[8]` (`eps=1e-6`). */
  readonly downNorm: Float32Array;
  /** `encoder.packed_image_codec.down_projection.weight`: `[8, 32]`. */
  readonly downProjection: Float32Array;
  /** `encoder.packed_image_codec.up_norm.weight`: `[8]` (`eps=1e-6`). */
  readonly upNorm: Float32Array;
  /** `encoder.packed_image_codec.up_projection.weight`: `[32, 8]`. */
  readonly upProjection: Float32Array;
}

export interface UniCodecCompressTrace {
  readonly childRows: number;
  readonly childCols: number;
  readonly parentRows: number;
  readonly parentCols: number;
  readonly isSourceEdit: boolean;
  readonly markerColumn: boolean;
  readonly gainPresetId: UniGainPresetId;
  readonly combinedScale: number;
  /** `[childRows * childCols, 8]` after per-child `down_norm` (`eps=1e-6`). */
  readonly normedChildren: Float32Array;
  /** `[parentRows * parentCols, 32]` `2x2` row-major concatenated children. */
  readonly packedConcat: Float32Array;
  /** `[parentRows * parentCols, 8]` after `down_projection`. */
  readonly projected: Float32Array;
  /** `[parentRows * parentCols, 8]` after `combinedScale`. */
  readonly scaled: Float32Array;
  readonly sinCos2d: Uni2dSinCosResult;
  /** `[parentRows * parentCols, 8]` final `scaled + sinCos2d.embeddings`. */
  readonly output: Float32Array;
}

export interface UniCodecExpandTrace {
  readonly parentRows: number;
  readonly parentCols: number;
  readonly childRows: number;
  readonly childCols: number;
  /** `[parentRows * parentCols, 8]` after `up_norm` (`eps=1e-6`). */
  readonly normedParents: Float32Array;
  /** `[parentRows * parentCols, 32]` after `up_projection`. */
  readonly upProjected: Float32Array;
  /** `[childRows * childCols, 8]` `2x2` raster unpacked child states. */
  readonly unpackedChildren: Float32Array;
  /** Optional `[childRows * childCols, 256]` logits when `genPredictor` is provided. */
  readonly logits: Float32Array | null;
}

export interface UniSourcePaddingVqMask {
  readonly vqGridRows: number;
  readonly vqGridCols: number;
  readonly letterbox: UniLetterboxInfo;
  /** Row-major `[vqGridRows * vqGridCols]` binary mask (`1` = locked padding, `0` = editable). */
  readonly lockMask: Uint8Array;
  readonly lockedCellCount: number;
  readonly editableCellCount: number;
}

// ============================================================================
// SLICE 2C: IMMUTABLE KV SLABS, BLOCK-DIFFUSION TEXT & GROUNDING
// ============================================================================

export type UniCfgBatchSize = 1 | 2 | 3;

export interface UniLayerBatchKv {
  readonly batchSize: UniCfgBatchSize;
  /** `[batchSize * kvProjOutDim]` (`batchSize * 16`) post-RoPE keys at one position. */
  readonly k: Float32Array;
  /** `[batchSize * kvProjOutDim]` (`batchSize * 16`) values at one position. */
  readonly v: Float32Array;
}

/** Immutable snapshot of one cached sequence position across all 34 layers (`B_cfg in {1,2,3}`). */
export interface UniKvPositionSlab {
  readonly positionIndex: number;
  readonly batchSize: UniCfgBatchSize;
  /** `[batchSize]` per-branch token routing (`0 = U`, `1 = G`) mirroring `_nlduni_dual_tower_routing`. */
  readonly routing: Uint8Array;
  /** True if any CFG row at this position has `G` routing. */
  readonly hasGeneration: boolean;
  readonly prefillMode: UniElasticPrefillMode;
  /** `34` layer entries (`L18..33` may be `null` under `elastic_pruned` T2I). */
  readonly layers: readonly (UniLayerBatchKv | null)[];
  readonly allocatedLayerCount: number;
  readonly byteLength: number;
}

export interface UniKvWasteTelemetry {
  readonly prefillMode: UniElasticPrefillMode;
  readonly downstreamTarget: 'u' | 'g';
  readonly batchSize: UniCfgBatchSize;
  readonly seqLen: number;
  readonly allocatedBytes: number;
  readonly activeBytes: number;
  readonly wastedSplitUBytes: number;
  readonly wastedSplitULayerCount: number;
  readonly wastedFraction: number;
}

export interface UniKvCacheState {
  readonly batchSize: UniCfgBatchSize;
  readonly capacity: number;
  readonly seqLen: number;
  readonly slabs: readonly UniKvPositionSlab[];
  /** Mirrors `_nlduni_dual_tower_has_generation` on HF `DynamicCache`. */
  readonly dualTowerHasGeneration: boolean;
  readonly prefillMode: UniElasticPrefillMode;
  readonly downstreamTarget: 'u' | 'g';
  readonly wasteTelemetry: UniKvWasteTelemetry;
}

export interface UniTextBlockDiffusionOptions {
  /** Text prompt token IDs (used to build embeddings when `inputsEmbeds` is omitted). */
  readonly promptTokenIds: ArrayLike<number>;
  /** Optional pre-assembled multimodal embeddings (e.g., RADIO-H visual embeddings spliced into `<SPECIAL_23>`). */
  readonly inputsEmbeds?: Float32Array;
  /** Block length `B` (miniature default `4`; production `32`). */
  readonly blockLength?: number;
  /** Maximum number of newly generated tokens. */
  readonly maxNewTokens: number;
  /** Maximum denoising steps per block (default `blockLength`). */
  readonly stepsPerBlock?: number;
  /** Confidence threshold for parallel unmasking (default `0.9`). */
  readonly confidenceThreshold?: number;
  /** Sampling temperature (`0` = greedy argmax; `> 0` = log-domain Gumbel per `U22`). */
  readonly temperature?: number;
  /** When true, pre-seeds slot 0 of each block from prior causal commit logits (`chat_utils.py:242-255`). */
  readonly causalContext?: boolean;
  /** Browser guard `U21`: mask `[MASK]` logit to `-Infinity` before argmax/sampling (default `true`). */
  readonly suppressMaskLogit?: boolean;
  /** Seed for `Xoshiro128**` when `temperature > 0` (`U23`). */
  readonly seed?: number;
  /** Optional explicit uniform variates in `(0, 1)` for 0-ULP stochastic parity (`U23`). */
  readonly explicitUniformVariates?: ArrayLike<number>;
  readonly eosTokenId?: number;
  readonly maskTokenId?: number;
}

export interface UniTextBlockStepTrace {
  readonly blockIndex: number;
  readonly stepIndex: number;
  readonly blockStartPos: number;
  readonly blockTokensBefore: Int32Array;
  readonly blockTokensAfter: Int32Array;
  /** Relative offsets within `[0 .. blockLength - 1]` unmasked on this step. */
  readonly unmaskedOffsets: readonly number[];
  /** Per-slot top-1 confidence (after optional mask suppression). */
  readonly slotConfidences: Float32Array;
  /** Unconstrained raw top-1 token ID per slot BEFORE `U21` `[MASK]` suppression. */
  readonly rawTop1TokenIds: Int32Array;
  /** Offsets where raw top-1 was `[MASK]` and the `U21` guard intervened. */
  readonly guardIntervenedOffsets: readonly number[];
  /** Shifted FLARE logits `[blockLength, 64]` predicting slots `0 .. B-1`. */
  readonly shiftedLogits: Float32Array;
  /** Shifted probabilities `[blockLength, 64]`. */
  readonly shiftedProbs: Float32Array;
  readonly usedCausalPreSeed: boolean;
}

export interface UniTextBlockDiffusionResult {
  readonly promptLen: number;
  readonly blockLength: number;
  readonly causalContext: boolean;
  readonly generatedTokenIds: Int32Array;
  readonly fullTokenIds: Int32Array;
  readonly decodedTokens: readonly string[];
  readonly stoppedByEos: boolean;
  readonly eosOffset: number | null;
  readonly stopReason: 'eos' | 'max_new_tokens' | 'capacity_exhausted';
  readonly steps: readonly UniTextBlockStepTrace[];
  readonly finalCache: UniKvCacheState;
  /** Number of forward evaluations (1 prefill + denoise steps + commit passes). */
  readonly nfeCount: number;
  readonly totalGuardInterventions: number;
}

export interface UniGroundingOptions {
  /** Prompt prefix token IDs before the assistant grounding template. */
  readonly promptTokenIds: ArrayLike<number>;
  /** Optional multimodal embeddings for the prompt prefix (`[promptLen, 8]`). */
  readonly promptInputsEmbeds?: Float32Array;
  readonly imageWidth: number;
  readonly imageHeight: number;
  /** Object tags to ground (`1..4` in miniature; single tag uses compact template). */
  readonly tags: readonly string[];
  /** Iterative unmasking steps (`1`, `2`, or `4`; default `2`). */
  readonly steps?: 1 | 2 | 4;
  /** Browser guard `U21`: constrain coordinate slots to `<LOC_0..16>` (default `true`). */
  readonly enforceLocDomainGuard?: boolean;
}

export interface UniGroundingStepTrace {
  readonly stepIndex: number;
  readonly slotTokenIdsBefore: Int32Array;
  readonly slotTokenIdsAfter: Int32Array;
  readonly unmaskedSlotIndices: readonly number[];
  readonly slotConfidences: Float32Array;
  readonly rawTop1TokenIds: Int32Array;
  readonly guardIntervenedSlotIndices: readonly number[];
}

export interface UniGroundingBox {
  readonly tag: string;
  /** Exact token IDs in `[43..59]` (`<LOC_0>..<LOC_16>`). */
  readonly locTokenIds: readonly [number, number, number, number];
  /** Exact token strings (`'<LOC_k>'`), never stripped (`U27`). */
  readonly locTokenStrings: readonly [string, string, string, string];
  /** Integer normalized bins in `[0..16]`. */
  readonly normalizedBins: readonly [number, number, number, number];
  /** Un-letterboxed `[x1, y1, x2, y2]` in source image pixels (`pixel_box`). */
  readonly pixelBox: readonly [number, number, number, number];
  /** Unconstrained raw top-1 token IDs before `U21` `<LOC_*>` domain mask. */
  readonly rawTop1Ids: readonly [number, number, number, number];
  readonly guardIntervenedCount: number;
}

export interface UniGroundingResult {
  readonly imageWidth: number;
  readonly imageHeight: number;
  readonly stepsRequested: 1 | 2 | 4;
  readonly assembledTemplateIds: Int32Array;
  readonly slotOffsetsInFullSequence: Int32Array;
  readonly boxes: readonly UniGroundingBox[];
  readonly stepTraces: readonly UniGroundingStepTrace[];
  readonly prefillCache: UniKvCacheState;
  /** Provenance (`U27`): confirms direct ID decoding preserves all `<LOC_*>` tokens. */
  readonly directTokenDecodePreservesLoc: true;
  /** Count of `<LOC_*>` / `<box_p>` / `<LOC_BEGIN/END>` tokens `skip_special_tokens=True` would strip (`U27`). */
  readonly skipSpecialTokensWouldStripCount: number;
  readonly skipSpecialTokensHazardNote: string;
  readonly totalGuardInterventions: number;
}
