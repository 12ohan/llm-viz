/**
 * Shared Headless TypeScript Contracts for Nemotron-Labs-Diffusion (`src/diffusion/types.ts`)
 *
 * Pinned Step-0 architectural contract shared across DF1 slices (`spec`, `prng`, `masks`,
 * `rope`, `attention`, `swiglu`, `parameters`, `forward`, and PyTorch 2.15.0 parity scripts).
 *
 * Pure headless TypeScript — zero DOM, WebGL, scene, render, or `src/twotower/` imports.
 * Zero `Math.random()`.
 */

// ============================================================================
// MINIATURE VOCABULARY & SPECIAL TOKENS
// ============================================================================

/**
 * Miniature 36-token vocabulary for Nemotron-Labs-Diffusion (`V = 36`):
 *   - `0..31`: 32 readable educational words
 *   - `32`: `'[MASK]'` (production `<SPECIAL_100>`, id `100`)
 *   - `33`: `'<|im_end|>'` (production Instruct EOS, id `11`)
 *   - `34`: `'<s>'` (production BOS, id `1`)
 *   - `35`: `'</think>'` (production thinking-budget sentinel, id `13`)
 */
export const DIFFUSION_MINI_VOCAB: readonly string[] = Object.freeze([
  'the',
  'model',
  'uses',
  'shared',
  'weights',
  'across',
  'forty',
  'dense',
  'layers',
  'with',
  'yarn',
  'rope',
  'scaling',
  'grouped',
  'query',
  'heads',
  'block',
  'masks',
  'decode',
  'tokens',
  'draft',
  'verify',
  'cache',
  'prefix',
  'causal',
  'bidir',
  'adapter',
  'delta',
  'exact',
  'parity',
  'token',
  'stream',
  '[MASK]',
  '<|im_end|>',
  '<s>',
  '</think>',
]);

export const DIFFUSION_MINI_MASK_TOKEN_ID = 32 as const;
export const DIFFUSION_MINI_EOS_TOKEN_ID = 33 as const;
export const DIFFUSION_MINI_BOS_TOKEN_ID = 34 as const;
export const DIFFUSION_MINI_END_THINK_TOKEN_ID = 35 as const;

// ============================================================================
// LOCKED MINIATURE TOPOLOGY & SEED CONSTANTS
// ============================================================================

export const DIFFUSION_BACKBONE_SEED = 0x4e4c4431 as const; // "NLD1"
export const DIFFUSION_LORA_SEED = 0x4c4f5241 as const; // "LORA"

export const DIFFUSION_BACKBONE_TENSOR_COUNT = 363 as const;
export const DIFFUSION_LORA_TENSOR_COUNT = 80 as const;
export const DIFFUSION_TOTAL_TENSOR_COUNT = 443 as const;

export const DIFFUSION_BACKBONE_FLOAT_COUNT = 229_064 as const;
export const DIFFUSION_LORA_FLOAT_COUNT = 42_240 as const;
export const DIFFUSION_TOTAL_FLOAT_COUNT = 271_304 as const;

/** Exact dyadic binary fractions `[1, 5/16, 1/16, 1/32]` for `theta=16, origMaxPos=16, factor=4, truncate=true`. */
export const DIFFUSION_DYADIC_YARN_INV_FREQ: readonly [number, number, number, number] =
  Object.freeze([1.0, 0.3125, 0.0625, 0.03125] as const);

export interface DiffusionYarnRopeConfig {
  readonly ropeType: 'yarn';
  readonly headDim: number;
  readonly ropeTheta: number;
  readonly originalMaxPositionEmbeddings: number;
  readonly factor: number;
  readonly betaFast: number;
  readonly betaSlow: number;
  readonly mscale: number;
  readonly mscaleAllDim: number;
  readonly truncate: boolean;
  readonly llama4Beta: number;
}

export interface DiffusionMiniConfig {
  readonly numLayers: 40;
  readonly hiddenSize: 8;
  readonly intermediateSize: 24;
  readonly numAttentionHeads: 32;
  readonly numKeyValueHeads: 8;
  readonly kvGroupSize: 4;
  readonly headDim: 8;
  readonly qProjOutDim: 256;
  readonly kvProjOutDim: 64;
  readonly vocabSize: 36;
  readonly maskTokenId: 32;
  readonly eosTokenId: 33;
  readonly bosTokenId: 34;
  readonly endThinkTokenId: 35;
  readonly rmsNormEps: 1e-5;
  readonly loraRank: 4;
  readonly loraAlpha: 16;
  readonly loraScale: 4.0;
  readonly defaultPromptLen: 8;
  readonly defaultBlockLength: 4;
  readonly supportedBlockLengths: readonly [4, 8];
  readonly kvCapacity: 32;
  readonly backboneSeed: number;
  readonly loraSeed: number;
  readonly maxSessionRetainedBytes: 12_582_912;
  readonly rope: DiffusionYarnRopeConfig;
}

export const DIFFUSION_MINI_CONFIG: DiffusionMiniConfig = Object.freeze({
  numLayers: 40,
  hiddenSize: 8,
  intermediateSize: 24,
  numAttentionHeads: 32,
  numKeyValueHeads: 8,
  kvGroupSize: 4,
  headDim: 8,
  qProjOutDim: 256,
  kvProjOutDim: 64,
  vocabSize: 36,
  maskTokenId: DIFFUSION_MINI_MASK_TOKEN_ID,
  eosTokenId: DIFFUSION_MINI_EOS_TOKEN_ID,
  bosTokenId: DIFFUSION_MINI_BOS_TOKEN_ID,
  endThinkTokenId: DIFFUSION_MINI_END_THINK_TOKEN_ID,
  rmsNormEps: 1e-5,
  loraRank: 4,
  loraAlpha: 16,
  loraScale: 4.0,
  defaultPromptLen: 8,
  defaultBlockLength: 4,
  supportedBlockLengths: Object.freeze([4, 8] as const),
  kvCapacity: 32,
  backboneSeed: DIFFUSION_BACKBONE_SEED,
  loraSeed: DIFFUSION_LORA_SEED,
  maxSessionRetainedBytes: 12_582_912, // 12.00 MiB hard session retention cap
  rope: Object.freeze({
    ropeType: 'yarn',
    headDim: 8,
    ropeTheta: 16.0,
    originalMaxPositionEmbeddings: 16,
    factor: 4.0,
    betaFast: 32.0,
    betaSlow: 1.0,
    mscale: 1.0,
    mscaleAllDim: 1.0,
    truncate: true,
    llama4Beta: 0.1,
  }),
});

// ============================================================================
// ENUMS & DISCRIMINATED UNION TYPES
// ============================================================================

export type DiffusionExecutionMode = 'ar' | 'dlm' | 'linear_spec' | 'linear_spec_lora';

export type DiffusionAttentionMaskMode = 'causal' | 'bidirectional' | 'custom';

export type DiffusionExplorerMaskMode =
  | 'joint_training'
  | 'ar_causal'
  | 'block_bidir'
  | 'causal_commit_verify'
  | 'vlm_quadratic_ss';

export type JointTrainingMaskRegion = 'none' | 'M_BD' | 'M_OBC' | 'M_BC';

export type KvMaskProvenance = 'token_causal' | 'block_bidirectional';

export type DiffusionLifecyclePhase =
  | 'ready'
  | 'running'
  | 'completed_eos'
  | 'completed_max_tokens'
  | 'capacity_exhausted';

export type YarnPlaneZone = 'original' | 'blended' | 'interpolated';

export type DiffusionTensorRole =
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
  | 'lora_A'
  | 'lora_B';

// ============================================================================
// CANONICAL TENSOR NAMING HELPERS
// ============================================================================

export const DIFFUSION_EMBED_TENSOR_NAME = 'encoder.embed_tokens.weight' as const;
export const DIFFUSION_FINAL_NORM_TENSOR_NAME = 'encoder.norm.weight' as const;
export const DIFFUSION_HEAD_TENSOR_NAME = 'diffusion_head.weight' as const;

export function layerNormWeightName(
  layer: number,
  kind: 'input' | 'post_attention',
): string {
  return `encoder.layers.${layer}.${kind}_layernorm.weight`;
}

export function layerAttnWeightName(
  layer: number,
  proj: 'q_proj' | 'k_proj' | 'v_proj' | 'o_proj',
): string {
  return `encoder.layers.${layer}.self_attn.${proj}.weight`;
}

export function layerMlpWeightName(
  layer: number,
  proj: 'gate_proj' | 'up_proj' | 'down_proj',
): string {
  return `encoder.layers.${layer}.mlp.${proj}.weight`;
}

export function layerLoraWeightName(
  layer: number,
  ab: 'lora_A' | 'lora_B',
): string {
  return `base_model.model.encoder.layers.${layer}.self_attn.o_proj.${ab}.weight`;
}

// ============================================================================
// PROVENANCE, 3-COORDINATE & MASK INTERFACES
// ============================================================================

/** Clarification #1 (`D19`): Tracks both clean post-EOS browser KV crop and upstream pre-EOS crop. */
export interface DiffusionSpecCropProvenance {
  readonly cacheLenBeforeVerify: number;
  readonly acceptedCount: number;
  readonly firstEosOffset: number | null;
  readonly browserPostEosCropLen: number;
  readonly upstreamPreEosCropLen: number;
}

export interface DiffusionThreeCoordinates {
  readonly seqIdx: Int32Array;
  readonly ropePositionIds: Int32Array;
  readonly cachePositions: Int32Array;
}

export interface DiffusionMaskMatrix {
  readonly mode: DiffusionExplorerMaskMode;
  readonly qLen: number;
  readonly kvLen: number;
  /** Row-major `[qLen * kvLen]` binary mask (`1` = visible, `0` = masked). */
  readonly visible: Uint8Array;
  readonly queryCoords: DiffusionThreeCoordinates;
  readonly kvRopePositionIds: Int32Array;
  readonly kvCachePositions: Int32Array;
  readonly isReadOnlyReference: boolean;
  readonly badge: string;
  /** Optional per-cell region classification `[qLen * kvLen]` for `joint_training`. */
  readonly regions?: readonly JointTrainingMaskRegion[];
}

// ============================================================================
// SPEC INVENTORY & DISCREPANCY LEDGER INTERFACES (for `spec.ts`)
// ============================================================================

export interface DiffusionCheckpointLfsEntry {
  readonly repoId: string;
  readonly revision: string;
  readonly file: string;
  readonly dtype: 'BF16' | 'F32';
  readonly isLora: boolean;
  readonly tensors: number;
  readonly parameters: number;
  readonly headerBytes: number;
  readonly payloadBytes: number;
  readonly totalBytes: number;
  readonly hfLfsSizeBytes: number;
  readonly diffBytes: number;
  readonly lfsSha256Prefix: string;
}

export interface DiffusionVariantSpec {
  readonly id: '14B' | '14B-Base' | '8B' | '8B-Base' | '3B' | '3B-Base' | 'VLM-8B';
  readonly repoId: string;
  readonly revision: string;
  readonly isCanonical: boolean;
  readonly numLayers: number;
  readonly vitLayers: number | null;
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly vocabSize: number;
  readonly numAttentionHeads: number;
  readonly numKeyValueHeads: number;
  readonly headDim: number;
  readonly qProjOutDim: number;
  readonly kvProjOutDim: number;
  readonly rmsNormEps: number;
  readonly ropeTheta: number;
  readonly yarnFactor: number;
  readonly originalMaxPositionEmbeddings: number;
  readonly llama4Beta: number;
  readonly maxPositionEmbeddings: number;
  readonly bosTokenId: number;
  readonly configEosTokenId: number | readonly number[];
  readonly tokenizerEosTokenId: number;
  readonly maskTokenId: number;
  readonly padTokenId: number | null;
  readonly dlmLossWeight: number | null;
  readonly backboneParameters: number;
  readonly backbonePayloadBytes: number;
  readonly loraParameters: number | null;
  readonly loraPayloadBytes: number | null;
}

export interface DiffusionComponentBreakdownEntry {
  readonly component: string;
  readonly tensorPattern: string;
  readonly shape: string;
  readonly tensorCount: number;
  readonly parameters: number;
  readonly bf16Bytes: number;
}

export interface DiffusionProductionKvEntry {
  readonly variantId: '14B' | '8B' | '3B';
  readonly numLayers: number;
  readonly kvBytesPerToken: number;
  readonly tempBlock32Bytes: number;
  readonly kvAt4KBytes: number;
  readonly kvAt16KBytes: number;
  readonly kvAt20KBytes: number;
  readonly kvAt64KBytes: number;
  readonly kvAt256KBytes: number;
}

export interface DiffusionContextTierSpec {
  readonly tier: 1 | 2 | 3 | 4;
  readonly label: string;
  readonly badge: string;
  readonly minTokens: number;
  readonly maxTokens: number | null;
  readonly referencePoints: readonly number[];
  readonly provenance: string;
}

export interface DiffusionDiscrepancyEntry {
  readonly id: string;
  readonly topic: string;
  readonly claimedOrExpected: string;
  readonly verifiedGroundTruth: string;
  readonly resolution: string;
  readonly sourceRef: string;
}

export interface DiffusionAtlasBudgetSpec {
  readonly name: string;
  readonly width: number;
  readonly height: number;
  readonly format: 'RG32F';
  readonly bytesPerTexel: 8;
  readonly totalBytes: number;
  readonly maxAuthorizedWidth?: number;
  readonly maxAuthorizedHeight?: number;
  readonly maxAuthorizedBytes?: number;
  readonly notes: string;
}

// ============================================================================
// PRNG INTERFACES (for `prng.ts`)
// ============================================================================

export type Xoshiro128ssState = readonly [number, number, number, number];

export interface Xoshiro128ssGenerator {
  nextUint32(): number;
  nextFloat32(): number;
  nextFloat64(): number;
  nextUniformOpenUnit(): number;
  getState(): Xoshiro128ssState;
  clone(): Xoshiro128ssGenerator;
}

// ============================================================================
// ROPE, WEIGHTS, ATTENTION, SWIGLU & FORWARD TRACE INTERFACES
// ============================================================================

export interface DiffusionYarnInvFreqResult {
  readonly invFreq: Float32Array;
  readonly attentionScaling: number;
  readonly low: number;
  readonly high: number;
  readonly extrapolationFactor: Float32Array;
  readonly zones: readonly YarnPlaneZone[];
}

export interface DiffusionRopeCosSin {
  readonly cos: Float32Array;
  readonly sin: Float32Array;
  readonly seqLen: number;
  readonly headDim: number;
}

export interface DiffusionAttentionWeights {
  readonly qProj: Float32Array; // [256, 8]
  readonly kProj: Float32Array; // [64, 8]
  readonly vProj: Float32Array; // [64, 8]
  readonly oProj: Float32Array; // [8, 256]
}

export interface DiffusionSwigluWeights {
  readonly gateProj: Float32Array; // [24, 8]
  readonly upProj: Float32Array; // [24, 8]
  readonly downProj: Float32Array; // [8, 24]
}

export interface DiffusionLoraWeights {
  readonly loraA: Float32Array; // [4, 256]
  readonly loraB: Float32Array; // [8, 4]
}

export interface DiffusionLayerWeights {
  readonly layerIndex: number;
  readonly inputLayerNorm: Float32Array; // [8]
  readonly selfAttn: DiffusionAttentionWeights;
  readonly postAttentionLayerNorm: Float32Array; // [8]
  readonly mlp: DiffusionSwigluWeights;
  readonly lora: DiffusionLoraWeights;
}

export interface DiffusionCanonicalWeightTensor {
  readonly name: string;
  readonly role: DiffusionTensorRole;
  readonly layerIndex: number | null;
  readonly isLora: boolean;
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

export interface CreateDiffusionModelOptions {
  readonly modelId?: string;
  readonly backboneSeed?: number;
  readonly loraSeed?: number;
  readonly config?: DiffusionMiniConfig;
  readonly vocab?: readonly string[];
}

export interface DiffusionModelParameters {
  readonly modelId: string;
  readonly config: DiffusionMiniConfig;
  readonly vocab: readonly string[];
  readonly backboneSeed: number;
  readonly loraSeed: number;
  readonly backboneFingerprint: string;
  readonly loraFingerprint: string;
  readonly combinedFingerprint: string;
  readonly embedTokens: Float32Array; // [36, 8]
  readonly layers: readonly DiffusionLayerWeights[]; // 40 layers
  readonly finalNorm: Float32Array; // [8]
  readonly diffusionHead: Float32Array; // [36, 8]
}

/**
 * Per-layer KV slice with shape `[seqLen, nKv * dHead]` (`[seqLen, 64]`),
 * stored in row-major `[t * 64 + kvHead * 8 + dim]` layout.
 */
export interface DiffusionLayerKvSlice {
  readonly k: Float32Array;
  readonly v: Float32Array;
  readonly seqLen: number;
}

export interface DiffusionAttentionInput {
  readonly hiddenStates: Float32Array; // [seqLen, d]
  readonly seqLen: number;
  readonly layerWeights: DiffusionLayerWeights;
  readonly config: DiffusionMiniConfig;
  readonly invFreq: Float32Array;
  readonly attentionScaling: number;
  readonly positionIds: Int32Array;
  readonly cachePositions?: Int32Array;
  readonly maskMode: DiffusionAttentionMaskMode;
  /** Row-major `[seqLen * kvLen]` binary mask (`1` visible, `0` masked) required when `maskMode === 'custom'`. */
  readonly customMask?: Uint8Array;
  readonly loraEnabled: boolean;
  readonly useCache: boolean;
  readonly pastKv?: DiffusionLayerKvSlice | null;
}

export interface DiffusionAttentionTrace {
  readonly qPre: Float32Array; // [seqLen, 256]
  readonly kPre: Float32Array; // [seqLen, 64]
  readonly vNew: Float32Array; // [seqLen, 64]
  readonly qRot: Float32Array; // [seqLen, 256]
  readonly kRot: Float32Array; // [seqLen, 64]
  readonly llama4Scales: Float32Array; // [seqLen]
  readonly qScaled: Float32Array; // [seqLen, 256]
  readonly kFull: Float32Array; // [kvLen, 64]
  readonly vFull: Float32Array; // [kvLen, 64]
  readonly scores: Float32Array; // [nQ, seqLen, kvLen]
  readonly probs: Float32Array; // [nQ, seqLen, kvLen]
  readonly attnConcat: Float32Array; // [seqLen, 256]
  readonly oProjBase: Float32Array; // [seqLen, 8]
  readonly loraHidden: Float32Array | null; // [seqLen, 4] | null
  readonly loraDelta: Float32Array | null; // [seqLen, 8] | null
  readonly output: Float32Array; // [seqLen, 8]
  readonly pastLen: number;
  readonly kvLen: number;
  readonly seqLen: number;
  readonly maskMode: DiffusionAttentionMaskMode;
  readonly loraEnabled: boolean;
  readonly useCache: boolean;
}

export interface DiffusionSwigluTrace {
  readonly gatePre: Float32Array; // [seqLen, dFf]
  readonly gateSilu: Float32Array; // [seqLen, dFf]
  readonly upVal: Float32Array; // [seqLen, dFf]
  readonly gated: Float32Array; // [seqLen, dFf]
  readonly output: Float32Array; // [seqLen, d]
  readonly seqLen: number;
  readonly d: number;
  readonly dFf: number;
}

export interface DiffusionLayerInput {
  readonly hiddenStates: Float32Array; // [seqLen, d]
  readonly seqLen: number;
  readonly layerWeights: DiffusionLayerWeights;
  readonly config: DiffusionMiniConfig;
  readonly invFreq: Float32Array;
  readonly attentionScaling: number;
  readonly positionIds: Int32Array;
  readonly cachePositions: Int32Array;
  readonly maskMode: DiffusionAttentionMaskMode;
  readonly customMask?: Uint8Array;
  readonly loraEnabled: boolean;
  readonly useCache: boolean;
  readonly pastKv?: DiffusionLayerKvSlice | null;
}

export interface DiffusionLayerTrace {
  readonly layerIndex: number;
  readonly inputNorm: Float32Array; // [seqLen, d]
  readonly attention: DiffusionAttentionTrace;
  readonly postAttentionResidual: Float32Array; // [seqLen, d]
  readonly postAttentionNorm: Float32Array; // [seqLen, d]
  readonly swiglu: DiffusionSwigluTrace;
  readonly output: Float32Array; // [seqLen, d]
}

export interface DiffusionForwardOptions {
  readonly tokenIds: ArrayLike<number>;
  readonly maskMode: DiffusionAttentionMaskMode;
  readonly customMask?: Uint8Array;
  readonly loraEnabled: boolean;
  readonly useCache: boolean;
  readonly positionIds?: Int32Array;
  readonly cachePositions?: Int32Array;
  readonly pastKvs?: readonly (DiffusionLayerKvSlice | null)[] | null;
  readonly retainedTraceLayers?: ReadonlySet<number>;
}

export interface DiffusionForwardResult {
  readonly tokenIds: Int32Array;
  readonly seqLen: number;
  readonly pastLen: number;
  readonly kvLen: number;
  readonly maskMode: DiffusionAttentionMaskMode;
  readonly loraEnabled: boolean;
  readonly useCache: boolean;
  readonly positionIds: Int32Array;
  readonly cachePositions: Int32Array;
  readonly embeddings: Float32Array; // [seqLen, d]
  readonly residualCheckpoints: readonly Float32Array[]; // 41 Float32Arrays of [seqLen, d]
  readonly layerTraces: ReadonlyMap<number, DiffusionLayerTrace>;
  readonly newKvs: readonly DiffusionLayerKvSlice[]; // 40 slices
  readonly finalNorm: Float32Array; // [seqLen, d]
  readonly logits: Float32Array; // [seqLen, vocabSize]
  readonly probs: Float32Array; // [seqLen, vocabSize]
}

// ============================================================================
// DF2 EXECUTION, KV CACHE, SAMPLER & SESSION CONTRACTS
// ============================================================================

export const DIFFUSION_KV_SLAB_FLOATS_PER_LAYER = 64 as const; // 8 KV heads * 8 headDim
export const DIFFUSION_KV_SLAB_FLOATS_PER_TOKEN = 5_120 as const; // 40 layers * 2 * 64
export const DIFFUSION_KV_SLAB_BYTES_PER_TOKEN = 20_480 as const; // 20.00 KiB / tok

export interface KvPositionSlab {
  readonly position: number;
  readonly tokenId: number;
  readonly maskProvenance: KvMaskProvenance;
  readonly numLayers: number;
  readonly kvProjOutDim: number;
  /** Row-major `[numLayers * kvProjOutDim]` (`2560` floats), layer `l` at `[l*64 .. (l+1)*64)`. */
  readonly k: Float32Array;
  /** Row-major `[numLayers * kvProjOutDim]` (`2560` floats), layer `l` at `[l*64 .. (l+1)*64)`. */
  readonly v: Float32Array;
  readonly slabDigest: string;
}

export interface PersistentKvSnapshot {
  readonly seqLen: number;
  readonly capacity: number;
  readonly kvMaskProvenance: KvMaskProvenance;
  readonly committedTokenIds: readonly number[];
  readonly slabs: readonly KvPositionSlab[];
  readonly snapshotDigest: string;
}

/** Clarification #1 (`D19`): Tracks both clean browser post-EOS crop and upstream pre-EOS crop. */
export interface CroppedKvProvenance extends DiffusionSpecCropProvenance {
  readonly preCropKvLen: number;
  readonly croppedSuffixLen: number;
}

export interface ForwardCounters {
  readonly actualForwardPasses: number;
  readonly decodeForwardPasses: number;
  readonly referenceMethodNfe: number;
}

export type DraftGuardReason = 'none' | 'non_progress_mask_loop' | 'max_draft_passes';

export interface BrowserGuardFlags {
  readonly draftGuardTriggered: DraftGuardReason;
  readonly residualMasksCommittedCount: number;
  readonly trailingMasksAfterEosCount: number;
  readonly capacityExhausted: boolean;
  readonly warnings: readonly string[];
}

export interface SamplerVariateProvenance {
  readonly mode: 'greedy_none' | 'prng_xoshiro128ss' | 'caller_explicit';
  readonly variateDigest: string | null;
  readonly count: number;
}

export interface CausalSampleResult {
  readonly tokenId: number;
  readonly probability: number;
  readonly temperature: number;
  readonly variateUsed: number | null;
  readonly variateProvenance: SamplerVariateProvenance;
}

export interface SamplerAConfig {
  readonly temperature: number;
  /** `null` enables static front-loaded quotas (`[1,..,1,0]` or `[1,..,1]`); numeric uses `conf >= threshold`. */
  readonly threshold: number | null;
  readonly maskTokenId?: number;
}

export interface SamplerATransferInput {
  readonly logits: Float32Array;
  readonly blockTokens: Int32Array;
  readonly seqLen: number;
  readonly vocabSize: number;
  readonly numTransferTokens: number;
  readonly config: SamplerAConfig;
  readonly prng?: Xoshiro128ssGenerator;
  readonly explicitGumbelUniforms?: Float64Array;
}

export interface SamplerATransferResult {
  readonly candidateTokens: Int32Array;
  readonly confidences: Float32Array;
  readonly transferIndex: Uint8Array;
  readonly updatedBlock: Int32Array;
  readonly selectedCount: number;
  readonly remainingMaskCount: number;
  readonly unmaskedMaskTokenCount: number;
  readonly variateProvenance: SamplerVariateProvenance;
}

export interface SamplerBConfig {
  readonly temperature: number;
  /** `0.0` performs single-pass commit-all; `> 0` performs confidence-thresholded iterative drafting. */
  readonly threshold: number;
  readonly maxDraftPassesPerCycle?: number;
  readonly maskTokenId?: number;
}

export interface SamplerBDraftInput {
  readonly draftLogits: Float32Array;
  readonly blockTokens: Int32Array;
  readonly seqLen: number;
  readonly vocabSize: number;
  readonly config: SamplerBConfig;
  readonly prng?: Xoshiro128ssGenerator;
  readonly explicitUniforms?: Float64Array;
}

export interface SamplerBDraftResult {
  readonly draftTokens: Int32Array;
  readonly confidences: Float32Array;
  readonly unmaskMask: Uint8Array;
  readonly updatedBlock: Int32Array;
  readonly remainingMaskCount: number;
  readonly singlePassCommitAll: boolean;
  readonly noProgress: boolean;
  readonly variateProvenance: SamplerVariateProvenance;
}

export interface LinearSpecDraftPassSummary {
  readonly passIndex: number;
  readonly blockBefore: Int32Array;
  readonly blockAfter: Int32Array;
  readonly confidences: Float32Array;
  readonly unmaskMask: Uint8Array;
  readonly remainingMaskCount: number;
  readonly noProgress: boolean;
  readonly variateProvenance: SamplerVariateProvenance;
}

export interface LinearSpecCycleInput {
  readonly kvSnapshot: PersistentKvSnapshot;
  readonly seedToken: number;
  readonly blockLength: number;
  readonly loraEnabled: boolean;
  readonly samplerBConfig: SamplerBConfig;
  readonly verifyTemperature: number;
  readonly eosTokenId?: number;
  readonly prng?: Xoshiro128ssGenerator;
  readonly explicitDraftUniforms?: readonly Float64Array[];
  readonly explicitVerifyUniforms?: Float64Array;
  readonly retainedTraceLayers?: ReadonlySet<number>;
  /** Optional deterministic hook for testing `D18` `[MASK]` non-progress guard without mutating parameters. */
  readonly overrideDraftForward?: (block: Int32Array, passIndex: number) => DiffusionForwardResult;
}

export interface LinearSpecCycleResult {
  readonly capacityExhausted: boolean;
  readonly lifecyclePhase: DiffusionLifecyclePhase;
  readonly loraEnabled: boolean;
  readonly blockLength: number;
  readonly seedToken: number;
  readonly draftedBlockTokens: Int32Array;
  readonly draftPasses: readonly LinearSpecDraftPassSummary[];
  readonly draftPassCount: number;
  readonly verifyArTokens: Int32Array;
  readonly matchedDraftCount: number;
  readonly acceptedCount: number;
  readonly emittedTokens: Int32Array;
  readonly nextSeedToken: number | null;
  readonly eosHit: boolean;
  readonly firstEosOffset: number | null;
  readonly cropProvenance: CroppedKvProvenance;
  readonly postCropKvSnapshot: PersistentKvSnapshot;
  readonly croppedSuffixSlabs: readonly KvPositionSlab[];
  readonly verifyForward: DiffusionForwardResult | null;
  readonly guardFlags: BrowserGuardFlags;
  readonly cycleCounters: ForwardCounters;
}

export interface DlmDenoiseStepSummary {
  readonly stepIndex: number;
  readonly quota: number;
  readonly blockBefore: Int32Array;
  readonly blockAfter: Int32Array;
  readonly candidateTokens: Int32Array;
  readonly confidences: Float32Array;
  readonly transferIndex: Uint8Array;
  readonly remainingMaskCount: number;
  readonly variateProvenance: SamplerVariateProvenance;
}

export interface DlmBlockCycleResult {
  readonly blockIndex: number;
  readonly causalContext: boolean;
  readonly blockLength: number;
  readonly initialSeedToken: number | null;
  readonly denoiseSteps: readonly DlmDenoiseStepSummary[];
  readonly denoisePassCount: number;
  readonly finalBlockTokens: Int32Array;
  readonly emittedTokens: Int32Array;
  readonly nextSeedToken: number | null;
  readonly eosHit: boolean;
  readonly firstEosOffset: number | null;
  readonly commitKvSnapshot: PersistentKvSnapshot;
  readonly commitForward: DiffusionForwardResult;
  readonly guardFlags: BrowserGuardFlags;
  readonly blockCounters: ForwardCounters;
}

export interface DiffusionExplicitVariates {
  readonly prefillUniform?: number;
  readonly arStepUniforms?: readonly number[];
  readonly dlmBlockGumbelUniforms?: readonly (readonly Float64Array[])[];
  readonly dlmCommitUniforms?: readonly number[];
  readonly specCycleDraftUniforms?: readonly (readonly Float64Array[])[];
  readonly specCycleVerifyUniforms?: readonly Float64Array[];
}

export interface DiffusionExecutionOptions {
  readonly mode: DiffusionExecutionMode;
  readonly promptTokenIds: ArrayLike<number>;
  readonly maxNewTokens: number;
  readonly blockLength?: number;
  readonly causalContext?: boolean;
  readonly temperature?: number;
  readonly samplerAThreshold?: number | null;
  readonly samplerBThreshold?: number;
  readonly maxDraftPassesPerCycle?: number;
  readonly samplingSeed?: number;
  readonly explicitVariates?: DiffusionExplicitVariates;
  readonly retainedTraceLayers?: ReadonlySet<number>;
  readonly kvCapacity?: number;
}

export interface DiffusionExecutionResult {
  readonly mode: DiffusionExecutionMode;
  readonly causalContext: boolean;
  readonly loraEnabledOnDraft: boolean;
  readonly promptTokenIds: Int32Array;
  readonly generatedTokenIds: Int32Array;
  readonly fullTokenIds: Int32Array;
  readonly lifecyclePhase: DiffusionLifecyclePhase;
  readonly kvMaskProvenance: KvMaskProvenance;
  readonly finalKvSnapshot: PersistentKvSnapshot;
  readonly counters: ForwardCounters;
  readonly guardFlags: BrowserGuardFlags;
  readonly arSteps: readonly DiffusionForwardResult[];
  readonly dlmBlocks: readonly DlmBlockCycleResult[];
  readonly specCycles: readonly LinearSpecCycleResult[];
  readonly prefillForward: DiffusionForwardResult;
  readonly lastCropProvenance: CroppedKvProvenance | null;
}

export type DiffusionSessionRecordKind = 'prefill' | 'ar_step' | 'dlm_block' | 'spec_cycle';

export interface DiffusionSessionBounds {
  readonly maxBranches: number;
  readonly maxRecords: number;
  readonly kvCapacity: number;
  readonly maxRetainedBytes: number;
  readonly eagerTraceLayers: readonly number[];
}

export const DEFAULT_DIFFUSION_SESSION_BOUNDS: DiffusionSessionBounds = Object.freeze({
  maxBranches: 12,
  maxRecords: 128,
  kvCapacity: 32,
  maxRetainedBytes: 12_582_912, // 12.00 MiB
  eagerTraceLayers: Object.freeze([0, 19, 39] as const),
});

export interface DiffusionSessionConfig {
  readonly mode: DiffusionExecutionMode;
  readonly causalContext: boolean;
  readonly blockLength: 4 | 8;
  readonly maxNewTokens: number;
  readonly temperature: number;
  readonly samplerAThreshold: number | null;
  readonly samplerBThreshold: number;
  readonly maxDraftPassesPerCycle: number;
  readonly samplingSeed: number;
}

export interface DiffusionSessionConfigPatch {
  readonly mode?: DiffusionExecutionMode;
  readonly causalContext?: boolean;
  readonly blockLength?: 4 | 8;
  readonly maxNewTokens?: number;
  readonly temperature?: number;
  readonly samplerAThreshold?: number | null;
  readonly samplerBThreshold?: number;
  readonly maxDraftPassesPerCycle?: number;
  readonly samplingSeed?: number;
  readonly allowExplicitRePrefill?: boolean;
}

export interface DiffusionPassCheckpointData {
  readonly inputTokenIds: Int32Array;
  readonly positionIds: Int32Array;
  readonly cachePositions: Int32Array;
  readonly maskMode: DiffusionAttentionMaskMode;
  readonly loraEnabled: boolean;
  readonly useCache: boolean;
  readonly residualCheckpoints: readonly Float32Array[];
  readonly finalNorm: Float32Array;
  readonly logits: Float32Array;
  readonly probs: Float32Array;
}

export interface DiffusionSessionRecord {
  readonly recordId: string;
  readonly parentRecordId: string | null;
  readonly branchId: string;
  readonly recordIndex: number;
  readonly kind: DiffusionSessionRecordKind;
  readonly configAtStep: DiffusionSessionConfig;
  readonly lifecyclePhase: DiffusionLifecyclePhase;
  readonly kvMaskProvenance: KvMaskProvenance;
  readonly kvSnapshot: PersistentKvSnapshot;
  readonly generatedTokenIds: Int32Array;
  readonly stepEmittedTokenIds: Int32Array;
  readonly pendingSeedToken: number | null;
  readonly counters: ForwardCounters;
  readonly guardFlags: BrowserGuardFlags;
  readonly cropProvenance: CroppedKvProvenance | null;
  readonly prngStateAfter: Xoshiro128ssState;
  readonly primaryPassCheckpoint: DiffusionPassCheckpointData;
  readonly specMeta?: {
    readonly draftedBlockTokens: Int32Array;
    readonly draftPassCount: number;
    readonly verifyArTokens: Int32Array;
    readonly matchedDraftCount: number;
    readonly acceptedCount: number;
    readonly loraEnabled: boolean;
  };
  readonly dlmMeta?: {
    readonly denoisePassCount: number;
    readonly finalBlockTokens: Int32Array;
    readonly causalContext: boolean;
  };
  readonly eagerLayerTraces: ReadonlyMap<number, DiffusionLayerTrace>;
  readonly activeCroppedSuffixSlabs: readonly KvPositionSlab[] | null;
}

export interface DiffusionSessionMemoryReport {
  readonly totalRetainedBytes: number;
  readonly maxRetainedBytes: number;
  readonly withinBudget: boolean;
  readonly committedKvSlabBytes: number;
  readonly activeCroppedSuffixBytes: number;
  readonly checkpointsAndLogitsBytes: number;
  readonly activeEagerTracesBytes: number;
  readonly uniqueBufferCount: number;
  readonly uniqueSlabCount: number;
  readonly branchCount: number;
  readonly recordCount: number;
  readonly activeRecordId: string | null;
}
