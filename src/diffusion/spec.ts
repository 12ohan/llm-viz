/**
 * Canonical Specification, Verified 0-Byte LFS Inventory, 7-Variant Family Matrix,
 * 4-Tier Context Honesty Contract, and 24-Entry Discrepancy Ledger (`D1–D24`)
 * for Nemotron-Labs-Diffusion (`src/diffusion/spec.ts`).
 *
 * Pure headless TypeScript — zero DOM, WebGL, Math.random(), or `src/twotower/` imports.
 */

import {
  DIFFUSION_BACKBONE_SEED,
  DIFFUSION_DYADIC_YARN_INV_FREQ,
  DIFFUSION_LORA_SEED,
  DIFFUSION_MINI_BOS_TOKEN_ID,
  DIFFUSION_MINI_CONFIG,
  DIFFUSION_MINI_END_THINK_TOKEN_ID,
  DIFFUSION_MINI_EOS_TOKEN_ID,
  DIFFUSION_MINI_MASK_TOKEN_ID,
  DIFFUSION_MINI_VOCAB,
  type DiffusionAtlasBudgetSpec,
  type DiffusionCheckpointLfsEntry,
  type DiffusionComponentBreakdownEntry,
  type DiffusionContextTierSpec,
  type DiffusionDiscrepancyEntry,
  type DiffusionExecutionMode,
  type DiffusionLifecyclePhase,
  type DiffusionProductionKvEntry,
  type DiffusionSpecCropProvenance,
  type DiffusionVariantSpec,
} from './types';

// Re-export miniature topology, vocab, special tokens, and seeds from `./types`
export {
  DIFFUSION_BACKBONE_SEED,
  DIFFUSION_DYADIC_YARN_INV_FREQ,
  DIFFUSION_LORA_SEED,
  DIFFUSION_MINI_BOS_TOKEN_ID,
  DIFFUSION_MINI_CONFIG,
  DIFFUSION_MINI_END_THINK_TOKEN_ID,
  DIFFUSION_MINI_EOS_TOKEN_ID,
  DIFFUSION_MINI_MASK_TOKEN_ID,
  DIFFUSION_MINI_VOCAB,
};

// ============================================================================
// CANONICAL REFERENCES & OUT-OF-SCOPE SISTER REPO GUARDRAIL
// ============================================================================

export const DIFFUSION_CANONICAL_REFS = Object.freeze({
  canonicalModel: Object.freeze({
    variantId: '14B' as const,
    repoId: 'nvidia/Nemotron-Labs-Diffusion-14B',
    revision: 'f8c3e2c078e193599b8882d965b1001c456ba738',
    commitTimestamp: '2026-06-03T17:58:02Z',
    url: 'https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-14B',
    collectionUrl: 'https://huggingface.co/collections/nvidia/nemotron-labs-diffusion',
    invalidPrefixNote:
      'nvidia/NVIDIA-Nemotron-Labs-Diffusion-* returns HTTP 401; all 7 public checkpoints omit the NVIDIA- prefix.',
  }),
  github: Object.freeze({
    repo: 'NVlabs/Nemotron-Labs-Diffusion',
    commit: '2aabcf75ecf70264d4047ff090d842450853f2e4',
    commitTimestamp: '2026-06-03T18:14:25Z',
    url: 'https://github.com/NVlabs/Nemotron-Labs-Diffusion',
  }),
  paper: Object.freeze({
    arxivId: 'arXiv:2607.05722v1',
    url: 'https://arxiv.org/abs/2607.05722',
    date: '2026-07-07',
    pages: 21,
  }),
  outOfScopeSisterRepo: Object.freeze({
    repo: 'NVlabs/Nemotron-Labs-Diffusion-Uni',
    commit: '16916949402ca738dcc74a7a37e0c43facafe347',
    commitTimestamp: '2026-10-28',
    arxivId: 'arXiv:2610.36466',
    url: 'https://github.com/NVlabs/Nemotron-Labs-Diffusion-Uni',
    paperUrl: 'https://arxiv.org/abs/2610.36466',
    inScope: false as const,
    note: 'Separate later unified visual understanding + visual generation DLLM repository (2026-10-27/28); explicitly out of scope for NLD v1.0 (arXiv:2607.05722v1).',
  }),
  sharedPythonSha256Prefixes: Object.freeze({
    configurationNemotronLabsDiffusion: '2b18216e1b4e',
    modelingNemotronLabsDiffusion: '29d73c5709e9',
    modelingMinistralText: '021afd8a72e2',
    modelingMinistralVlm8b: '21f6d710f395',
  }),
  licensing: Object.freeze({
    codeLicense: 'Apache-2.0',
    weightsLicense: 'NVIDIA Open Model License (Oct 24, 2025)',
    cleanRoomPolicy:
      'Zero NVIDIA Python source code, weights, LoRA safetensors, or tokenizer files are copied or redistributed.',
  }),
});

// ============================================================================
// APPROVED SPEC CLARIFICATIONS (DF1–DF3)
// ============================================================================

export function computeSpecCropProvenance(
  cacheLenBeforeVerify: number,
  acceptedCount: number,
  firstEosOffset: number | null,
): DiffusionSpecCropProvenance {
  const upstreamPreEosCropLen = cacheLenBeforeVerify + acceptedCount;
  const browserPostEosCropLen =
    firstEosOffset !== null && firstEosOffset >= 0 && firstEosOffset < acceptedCount
      ? cacheLenBeforeVerify + firstEosOffset + 1
      : upstreamPreEosCropLen;
  return {
    cacheLenBeforeVerify,
    acceptedCount,
    firstEosOffset,
    browserPostEosCropLen,
    upstreamPreEosCropLen,
  };
}

export interface KvCapacityCheckResult {
  readonly canProceed: boolean;
  readonly lifecyclePhase: DiffusionLifecyclePhase;
  readonly remainingSlots: number;
  readonly requiredSlots: number;
}

/**
 * Clarification #2 (§5.5): Never silently clamp `B` down to `1` when approaching `kvCapacity`.
 * Halt cleanly with `lifecyclePhase = 'capacity_exhausted'` whenever `kvCapacity - cacheLen < B`
 * (and `< 1` for `ar`).
 */
export function checkKvCapacityForStep(
  mode: DiffusionExecutionMode,
  cacheLen: number,
  kvCapacity: number,
  blockSizeB: number,
): KvCapacityCheckResult {
  const remainingSlots = Math.max(0, kvCapacity - cacheLen);
  const requiredSlots = mode === 'ar' ? 1 : blockSizeB;
  const canProceed = remainingSlots >= requiredSlots;
  return {
    canProceed,
    lifecyclePhase: canProceed ? 'running' : 'capacity_exhausted',
    remainingSlots,
    requiredSlots,
  };
}

export const DIFFUSION_SPEC_CLARIFICATIONS = Object.freeze({
  eosCropParity: Object.freeze({
    id: 'C1',
    ref: 'D19 / DF0 §5.5 vs Cat 9',
    rule: 'Keep clean browser post-EOS KV crop (`browserPostEosCropLen = cache_len + first_eos + 1`) while recording `upstreamPreEosCropLen = cache_len + accepted` in provenance.',
  }),
  kvCapacityTailBlockBoundary: Object.freeze({
    id: 'C2',
    ref: 'DF0 §5.5',
    rule: "Never silently clamp B down to 1. Halt cleanly with `lifecyclePhase = 'capacity_exhausted'` whenever `kvCapacity - cache_len < B` (and `< 1` for `ar`).",
  }),
  atlasHeadroom: Object.freeze({
    id: 'C3',
    ref: 'DF0 §8.3 (for DF3)',
    rule: '256x128 (256 KiB) is used when row-packing stays clean, and DF3 is explicitly authorized up to 256x256 (512 KiB) for the GQA+RoPE+LoRA RG32F atlas.',
  }),
});

// ============================================================================
// EXACT 0 B SAFETENSORS + LFS INVENTORY (ALL 13 FILES ACROSS 7 REPOS + 3 LORAS)
// ============================================================================

export const DIFFUSION_LFS_INVENTORY: readonly DiffusionCheckpointLfsEntry[] = Object.freeze([
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-14B',
    revision: 'f8c3e2c078e193599b8882d965b1001c456ba738',
    file: 'model.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 363,
    parameters: 13_506_073_600,
    headerBytes: 43_512,
    payloadBytes: 27_012_147_200,
    totalBytes: 27_012_190_712,
    hfLfsSizeBytes: 27_012_190_712,
    diffBytes: 0,
    lfsSha256Prefix: '272e78c39809',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-14B',
    revision: 'f8c3e2c078e193599b8882d965b1001c456ba738',
    file: 'linear_spec_lora/adapter_model.safetensors',
    dtype: 'F32',
    isLora: true,
    tensors: 80,
    parameters: 47_185_920,
    headerBytes: 11_120,
    payloadBytes: 188_743_680,
    totalBytes: 188_754_800,
    hfLfsSizeBytes: 188_754_800,
    diffBytes: 0,
    lfsSha256Prefix: 'cb29e2595fe0',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-14B-Base',
    revision: 'ca30a6db6274c0b01d4eb21e6f1cb28222ea0c63',
    file: 'model.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 363,
    parameters: 13_506_073_600,
    headerBytes: 43_512,
    payloadBytes: 27_012_147_200,
    totalBytes: 27_012_190_712,
    hfLfsSizeBytes: 27_012_190_712,
    diffBytes: 0,
    lfsSha256Prefix: 'f43537460484',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-8B',
    revision: '16c67f0560b912e93e0cabb6e0c4f5c3086d95fc',
    file: 'model.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 309,
    parameters: 8_489_553_920,
    headerBytes: 36_880,
    payloadBytes: 16_979_107_840,
    totalBytes: 16_979_144_720,
    hfLfsSizeBytes: 16_979_144_720,
    diffBytes: 0,
    lfsSha256Prefix: '14de6f8fee56',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-8B',
    revision: '16c67f0560b912e93e0cabb6e0c4f5c3086d95fc',
    file: 'linear_spec_lora/adapter_model.safetensors',
    dtype: 'F32',
    isLora: true,
    tensors: 68,
    parameters: 35_651_584,
    headerBytes: 9_432,
    payloadBytes: 142_606_336,
    totalBytes: 142_615_768,
    hfLfsSizeBytes: 142_615_768,
    diffBytes: 0,
    lfsSha256Prefix: '5893f4aec2f4',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-8B-Base',
    revision: '59ff0ffee284112fc6ccf37493ced41a11031434',
    file: 'model.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 309,
    parameters: 8_489_553_920,
    headerBytes: 36_880,
    payloadBytes: 16_979_107_840,
    totalBytes: 16_979_144_720,
    hfLfsSizeBytes: 16_979_144_720,
    diffBytes: 0,
    lfsSha256Prefix: '73af2cd1c982',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-3B',
    revision: '0d51902da1f8869f83413ce642fab402fa5641e0',
    file: 'model.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 237,
    parameters: 3_831_659_520,
    headerBytes: 27_960,
    payloadBytes: 7_663_319_040,
    totalBytes: 7_663_347_000,
    hfLfsSizeBytes: 7_663_347_000,
    diffBytes: 0,
    lfsSha256Prefix: '8c4beff3a8a2',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-3B',
    revision: '0d51902da1f8869f83413ce642fab402fa5641e0',
    file: 'linear_spec_lora/adapter_model.safetensors',
    dtype: 'F32',
    isLora: true,
    tensors: 52,
    parameters: 23_855_104,
    headerBytes: 7_184,
    payloadBytes: 95_420_416,
    totalBytes: 95_427_600,
    hfLfsSizeBytes: 95_427_600,
    diffBytes: 0,
    lfsSha256Prefix: '897ef67dff8a',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-3B-Base',
    revision: '4bd78816ba988ca8b71b8f432b00b682572b236c',
    file: 'model.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 237,
    parameters: 3_831_659_520,
    headerBytes: 27_960,
    payloadBytes: 7_663_319_040,
    totalBytes: 7_663_347_000,
    hfLfsSizeBytes: 7_663_347_000,
    diffBytes: 0,
    lfsSha256Prefix: 'cf9ef7826f4d',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-VLM-8B',
    revision: 'adca93d16471c1e07d594ae444d23e1876f6b365',
    file: 'model-00001-of-00004.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 286,
    parameters: 2_492_131_328,
    headerBytes: 39_432,
    payloadBytes: 4_984_262_656,
    totalBytes: 4_984_302_088,
    hfLfsSizeBytes: 4_984_302_088,
    diffBytes: 0,
    lfsSha256Prefix: 'd6bdc8ac3f1b',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-VLM-8B',
    revision: 'adca93d16471c1e07d594ae444d23e1876f6b365',
    file: 'model-00002-of-00004.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 104,
    parameters: 2_499_895_296,
    headerBytes: 12_312,
    payloadBytes: 4_999_790_592,
    totalBytes: 4_999_802_904,
    hfLfsSizeBytes: 4_999_802_904,
    diffBytes: 0,
    lfsSha256Prefix: '6e4c3f9cac91',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-VLM-8B',
    revision: 'adca93d16471c1e07d594ae444d23e1876f6b365',
    file: 'model-00003-of-00004.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 100,
    parameters: 2_457_952_256,
    headerBytes: 11_864,
    payloadBytes: 4_915_904_512,
    totalBytes: 4_915_916_376,
    hfLfsSizeBytes: 4_915_916_376,
    diffBytes: 0,
    lfsSha256Prefix: '562159c86be4',
  }),
  Object.freeze({
    repoId: 'nvidia/Nemotron-Labs-Diffusion-VLM-8B',
    revision: 'adca93d16471c1e07d594ae444d23e1876f6b365',
    file: 'model-00004-of-00004.safetensors',
    dtype: 'BF16',
    isLora: false,
    tensors: 41,
    parameters: 1_468_055_552,
    headerBytes: 4_864,
    payloadBytes: 2_936_111_104,
    totalBytes: 2_936_115_968,
    hfLfsSizeBytes: 2_936_115_968,
    diffBytes: 0,
    lfsSha256Prefix: '7323f1fdacbf',
  }),
]);

export interface DiffusionLfsVerificationResult {
  readonly valid: boolean;
  readonly entryCount: number;
  readonly mismatches: readonly string[];
}

export function verifyDiffusionLfsInventory(
  entries: readonly DiffusionCheckpointLfsEntry[] = DIFFUSION_LFS_INVENTORY,
): DiffusionLfsVerificationResult {
  const mismatches: string[] = [];
  for (const entry of entries) {
    const label = `${entry.repoId}/${entry.file}`;
    const expectedPayload = entry.parameters * (entry.dtype === 'BF16' ? 2 : 4);
    if (entry.payloadBytes !== expectedPayload) {
      mismatches.push(`${label}: payloadBytes ${entry.payloadBytes} !== expected ${expectedPayload}`);
    }
    if (entry.headerBytes + entry.payloadBytes !== entry.totalBytes) {
      mismatches.push(
        `${label}: headerBytes+payloadBytes (${entry.headerBytes + entry.payloadBytes}) !== totalBytes (${entry.totalBytes})`,
      );
    }
    if (entry.totalBytes !== entry.hfLfsSizeBytes) {
      mismatches.push(`${label}: totalBytes (${entry.totalBytes}) !== hfLfsSizeBytes (${entry.hfLfsSizeBytes})`);
    }
    if (entry.diffBytes !== 0) {
      mismatches.push(`${label}: diffBytes (${entry.diffBytes}) !== 0`);
    }
    if (entry.lfsSha256Prefix.length !== 12) {
      mismatches.push(`${label}: lfsSha256Prefix length (${entry.lfsSha256Prefix.length}) !== 12`);
    }
  }
  return {
    valid: mismatches.length === 0,
    entryCount: entries.length,
    mismatches,
  };
}

// ============================================================================
// 7-VARIANT FAMILY MATRIX (DF0 §3.1)
// ============================================================================

export const DIFFUSION_FAMILY_SPECS: readonly DiffusionVariantSpec[] = Object.freeze([
  Object.freeze({
    id: '14B',
    repoId: 'nvidia/Nemotron-Labs-Diffusion-14B',
    revision: 'f8c3e2c078e193599b8882d965b1001c456ba738',
    isCanonical: true,
    numLayers: 40,
    vitLayers: null,
    hiddenSize: 5120,
    intermediateSize: 16384,
    vocabSize: 131072,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 128,
    qProjOutDim: 4096,
    kvProjOutDim: 1024,
    rmsNormEps: 1e-5,
    ropeTheta: 1e9,
    yarnFactor: 16.0,
    originalMaxPositionEmbeddings: 16384,
    llama4Beta: 0.1,
    maxPositionEmbeddings: 262144,
    bosTokenId: 1,
    configEosTokenId: 11,
    tokenizerEosTokenId: 11,
    maskTokenId: 100,
    padTokenId: null,
    dlmLossWeight: null,
    backboneParameters: 13_506_073_600,
    backbonePayloadBytes: 27_012_147_200,
    loraParameters: 47_185_920,
    loraPayloadBytes: 188_743_680,
  }),
  Object.freeze({
    id: '14B-Base',
    repoId: 'nvidia/Nemotron-Labs-Diffusion-14B-Base',
    revision: 'ca30a6db6274c0b01d4eb21e6f1cb28222ea0c63',
    isCanonical: false,
    numLayers: 40,
    vitLayers: null,
    hiddenSize: 5120,
    intermediateSize: 16384,
    vocabSize: 131072,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 128,
    qProjOutDim: 4096,
    kvProjOutDim: 1024,
    rmsNormEps: 1e-5,
    ropeTheta: 1e9,
    yarnFactor: 16.0,
    originalMaxPositionEmbeddings: 16384,
    llama4Beta: 0.1,
    maxPositionEmbeddings: 262144,
    bosTokenId: 1,
    configEosTokenId: 2,
    tokenizerEosTokenId: 11,
    maskTokenId: 100,
    padTokenId: null,
    dlmLossWeight: null,
    backboneParameters: 13_506_073_600,
    backbonePayloadBytes: 27_012_147_200,
    loraParameters: null,
    loraPayloadBytes: null,
  }),
  Object.freeze({
    id: '8B',
    repoId: 'nvidia/Nemotron-Labs-Diffusion-8B',
    revision: '16c67f0560b912e93e0cabb6e0c4f5c3086d95fc',
    isCanonical: false,
    numLayers: 34,
    vitLayers: null,
    hiddenSize: 4096,
    intermediateSize: 14336,
    vocabSize: 131072,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 128,
    qProjOutDim: 4096,
    kvProjOutDim: 1024,
    rmsNormEps: 1e-5,
    ropeTheta: 1e6,
    yarnFactor: 16.0,
    originalMaxPositionEmbeddings: 16384,
    llama4Beta: 0.1,
    maxPositionEmbeddings: 262144,
    bosTokenId: 1,
    configEosTokenId: 11,
    tokenizerEosTokenId: 11,
    maskTokenId: 100,
    padTokenId: null,
    dlmLossWeight: null,
    backboneParameters: 8_489_553_920,
    backbonePayloadBytes: 16_979_107_840,
    loraParameters: 35_651_584,
    loraPayloadBytes: 142_606_336,
  }),
  Object.freeze({
    id: '8B-Base',
    repoId: 'nvidia/Nemotron-Labs-Diffusion-8B-Base',
    revision: '59ff0ffee284112fc6ccf37493ced41a11031434',
    isCanonical: false,
    numLayers: 34,
    vitLayers: null,
    hiddenSize: 4096,
    intermediateSize: 14336,
    vocabSize: 131072,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 128,
    qProjOutDim: 4096,
    kvProjOutDim: 1024,
    rmsNormEps: 1e-5,
    ropeTheta: 1e6,
    yarnFactor: 0.25,
    originalMaxPositionEmbeddings: 16384,
    llama4Beta: 0.1,
    maxPositionEmbeddings: 4096,
    bosTokenId: 1,
    configEosTokenId: 2,
    tokenizerEosTokenId: 11,
    maskTokenId: 100,
    padTokenId: null,
    dlmLossWeight: null,
    backboneParameters: 8_489_553_920,
    backbonePayloadBytes: 16_979_107_840,
    loraParameters: null,
    loraPayloadBytes: null,
  }),
  Object.freeze({
    id: '3B',
    repoId: 'nvidia/Nemotron-Labs-Diffusion-3B',
    revision: '0d51902da1f8869f83413ce642fab402fa5641e0',
    isCanonical: false,
    numLayers: 26,
    vitLayers: null,
    hiddenSize: 3072,
    intermediateSize: 9216,
    vocabSize: 131072,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 128,
    qProjOutDim: 4096,
    kvProjOutDim: 1024,
    rmsNormEps: 1e-5,
    ropeTheta: 1e6,
    yarnFactor: 16.0,
    originalMaxPositionEmbeddings: 16384,
    llama4Beta: 0.1,
    maxPositionEmbeddings: 262144,
    bosTokenId: 1,
    configEosTokenId: 11,
    tokenizerEosTokenId: 11,
    maskTokenId: 100,
    padTokenId: null,
    dlmLossWeight: null,
    backboneParameters: 3_831_659_520,
    backbonePayloadBytes: 7_663_319_040,
    loraParameters: 23_855_104,
    loraPayloadBytes: 95_420_416,
  }),
  Object.freeze({
    id: '3B-Base',
    repoId: 'nvidia/Nemotron-Labs-Diffusion-3B-Base',
    revision: '4bd78816ba988ca8b71b8f432b00b682572b236c',
    isCanonical: false,
    numLayers: 26,
    vitLayers: null,
    hiddenSize: 3072,
    intermediateSize: 9216,
    vocabSize: 131072,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 128,
    qProjOutDim: 4096,
    kvProjOutDim: 1024,
    rmsNormEps: 1e-5,
    ropeTheta: 1e6,
    yarnFactor: 0.25,
    originalMaxPositionEmbeddings: 16384,
    llama4Beta: 0.1,
    maxPositionEmbeddings: 4096,
    bosTokenId: 1,
    configEosTokenId: 2,
    tokenizerEosTokenId: 11,
    maskTokenId: 100,
    padTokenId: null,
    dlmLossWeight: null,
    backboneParameters: 3_831_659_520,
    backbonePayloadBytes: 7_663_319_040,
    loraParameters: null,
    loraPayloadBytes: null,
  }),
  Object.freeze({
    id: 'VLM-8B',
    repoId: 'nvidia/Nemotron-Labs-Diffusion-VLM-8B',
    revision: 'adca93d16471c1e07d594ae444d23e1876f6b365',
    isCanonical: false,
    numLayers: 34,
    vitLayers: 24,
    hiddenSize: 4096,
    intermediateSize: 14336,
    vocabSize: 131073,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 128,
    qProjOutDim: 4096,
    kvProjOutDim: 1024,
    rmsNormEps: 1e-5,
    ropeTheta: 1e6,
    yarnFactor: 16.0,
    originalMaxPositionEmbeddings: 16384,
    llama4Beta: 0.1,
    maxPositionEmbeddings: 262144,
    bosTokenId: 1,
    configEosTokenId: Object.freeze([11, 2] as const),
    tokenizerEosTokenId: 11,
    maskTokenId: 100,
    padTokenId: 11,
    dlmLossWeight: 0.5,
    backboneParameters: 8_918_034_432,
    backbonePayloadBytes: 17_836_068_864,
    loraParameters: null,
    loraPayloadBytes: null,
  }),
]);

// ============================================================================
// EXACT 14B COMPONENT BREAKDOWN (363 BF16 BACKBONE + 80 F32 LORA)
// ============================================================================

export interface Diffusion14BLoraBreakdownEntry {
  readonly component: string;
  readonly tensorPattern: string;
  readonly shape: string;
  readonly tensorCount: number;
  readonly parameters: number;
  readonly f32Bytes: number;
}

export interface Diffusion14BComponentBreakdown {
  readonly backbone: readonly DiffusionComponentBreakdownEntry[];
  readonly backboneTensorCount: 363;
  readonly backboneParameters: 13_506_073_600;
  readonly backbonePayloadBytes: 27_012_147_200;
  readonly lora: Diffusion14BLoraBreakdownEntry;
}

export const DIFFUSION_14B_COMPONENT_BREAKDOWN: Diffusion14BComponentBreakdown = Object.freeze({
  backbone: Object.freeze([
    Object.freeze({
      component: 'Input Embeddings',
      tensorPattern: 'encoder.embed_tokens.weight',
      shape: '[131072, 5120]',
      tensorCount: 1,
      parameters: 671_088_640,
      bf16Bytes: 1_342_177_280,
    }),
    Object.freeze({
      component: 'Pre-Attention RMSNorm (x40)',
      tensorPattern: 'encoder.layers.*.input_layernorm.weight',
      shape: '[5120]',
      tensorCount: 40,
      parameters: 204_800,
      bf16Bytes: 409_600,
    }),
    Object.freeze({
      component: 'GQA Q Projection (x40, contracting 5120->4096)',
      tensorPattern: 'encoder.layers.*.self_attn.q_proj.weight',
      shape: '[4096, 5120]',
      tensorCount: 40,
      parameters: 838_860_800,
      bf16Bytes: 1_677_721_600,
    }),
    Object.freeze({
      component: 'GQA K Projection (x40, 8 KV heads)',
      tensorPattern: 'encoder.layers.*.self_attn.k_proj.weight',
      shape: '[1024, 5120]',
      tensorCount: 40,
      parameters: 209_715_200,
      bf16Bytes: 419_430_400,
    }),
    Object.freeze({
      component: 'GQA V Projection (x40, 8 KV heads)',
      tensorPattern: 'encoder.layers.*.self_attn.v_proj.weight',
      shape: '[1024, 5120]',
      tensorCount: 40,
      parameters: 209_715_200,
      bf16Bytes: 419_430_400,
    }),
    Object.freeze({
      component: 'GQA O Projection (x40, expanding 4096->5120)',
      tensorPattern: 'encoder.layers.*.self_attn.o_proj.weight',
      shape: '[5120, 4096]',
      tensorCount: 40,
      parameters: 838_860_800,
      bf16Bytes: 1_677_721_600,
    }),
    Object.freeze({
      component: 'Pre-MLP RMSNorm (x40)',
      tensorPattern: 'encoder.layers.*.post_attention_layernorm.weight',
      shape: '[5120]',
      tensorCount: 40,
      parameters: 204_800,
      bf16Bytes: 409_600,
    }),
    Object.freeze({
      component: 'SwiGLU Gate Projection (x40)',
      tensorPattern: 'encoder.layers.*.mlp.gate_proj.weight',
      shape: '[16384, 5120]',
      tensorCount: 40,
      parameters: 3_355_443_200,
      bf16Bytes: 6_710_886_400,
    }),
    Object.freeze({
      component: 'SwiGLU Up Projection (x40)',
      tensorPattern: 'encoder.layers.*.mlp.up_proj.weight',
      shape: '[16384, 5120]',
      tensorCount: 40,
      parameters: 3_355_443_200,
      bf16Bytes: 6_710_886_400,
    }),
    Object.freeze({
      component: 'SwiGLU Down Projection (x40)',
      tensorPattern: 'encoder.layers.*.mlp.down_proj.weight',
      shape: '[5120, 16384]',
      tensorCount: 40,
      parameters: 3_355_443_200,
      bf16Bytes: 6_710_886_400,
    }),
    Object.freeze({
      component: 'Final RMSNorm',
      tensorPattern: 'encoder.norm.weight',
      shape: '[5120]',
      tensorCount: 1,
      parameters: 5_120,
      bf16Bytes: 10_240,
    }),
    Object.freeze({
      component: 'Untied Diffusion Head',
      tensorPattern: 'diffusion_head.weight',
      shape: '[131072, 5120]',
      tensorCount: 1,
      parameters: 671_088_640,
      bf16Bytes: 1_342_177_280,
    }),
  ]),
  backboneTensorCount: 363,
  backboneParameters: 13_506_073_600,
  backbonePayloadBytes: 27_012_147_200,
  lora: Object.freeze({
    component: 'linear_spec_lora (Draft-Only o_proj, r=128, alpha=512, scale=4.0)',
    tensorPattern: 'base_model.model.encoder.layers.*.self_attn.o_proj.lora_{A,B}.weight',
    shape: 'lora_A:[128, 4096], lora_B:[5120, 128] (x40)',
    tensorCount: 80,
    parameters: 47_185_920,
    f32Bytes: 188_743_680,
  }),
});

// ============================================================================
// PRODUCTION BF16 KV FOOTPRINT & 4-TIER CONTEXT HONESTY CONTRACT (DF0 §3.3)
// ============================================================================

export const DIFFUSION_PRODUCTION_KV_TABLE: readonly DiffusionProductionKvEntry[] = Object.freeze([
  Object.freeze({
    variantId: '14B',
    numLayers: 40,
    kvBytesPerToken: 163_840, // 160.00 KiB
    tempBlock32Bytes: 5_242_880, // 5.000 MiB
    kvAt4KBytes: 671_088_640, // 0.625 GiB / 0.671 GB
    kvAt16KBytes: 2_684_354_560, // 2.500 GiB / 2.684 GB
    kvAt20KBytes: 3_355_443_200, // 3.125 GiB / 3.355 GB
    kvAt64KBytes: 10_737_418_240, // 10.000 GiB / 10.737 GB
    kvAt256KBytes: 42_949_672_960, // 40.000 GiB / 42.950 GB
  }),
  Object.freeze({
    variantId: '8B',
    numLayers: 34,
    kvBytesPerToken: 139_264, // 136.00 KiB
    tempBlock32Bytes: 4_456_448, // 4.250 MiB
    kvAt4KBytes: 570_425_344, // 0.53125 GiB / 0.570 GB
    kvAt16KBytes: 2_281_701_376, // 2.125 GiB / 2.282 GB
    kvAt20KBytes: 2_852_126_720, // 2.65625 GiB / 2.852 GB
    kvAt64KBytes: 9_126_805_504, // 8.500 GiB / 9.127 GB
    kvAt256KBytes: 36_507_222_016, // 34.000 GiB / 36.507 GB
  }),
  Object.freeze({
    variantId: '3B',
    numLayers: 26,
    kvBytesPerToken: 106_496, // 104.00 KiB
    tempBlock32Bytes: 3_407_872, // 3.250 MiB
    kvAt4KBytes: 436_207_616, // 0.40625 GiB / 0.436 GB
    kvAt16KBytes: 1_744_830_464, // 1.625 GiB / 1.745 GB
    kvAt20KBytes: 2_181_038_080, // 2.03125 GiB / 2.181 GB
    kvAt64KBytes: 6_979_321_856, // 6.500 GiB / 6.979 GB
    kvAt256KBytes: 27_917_287_424, // 26.000 GiB / 27.917 GB
  }),
]);

export const DIFFUSION_CONTEXT_TIERS: readonly DiffusionContextTierSpec[] = Object.freeze([
  Object.freeze({
    tier: 1,
    label: '4,096 (Base Stage 2) · 16,384 (Instruct SFT)',
    badge: 'Verified Training',
    minTokens: 1,
    maxTokens: 16_384,
    referencePoints: Object.freeze([4096, 16384] as const),
    provenance:
      'Paper §5.1 (seq_len=4096) & §5.2 (seq_len=16k), matching original_max_position_embeddings=16384 where Llama-4 query scale = 1.0.',
  }),
  Object.freeze({
    tier: 2,
    label: '512 (CLI) · 1,024 (SPEED-Bench) · 2,048 (SGLang Spark) · 20,480 (eval.sh dLM/Spec) · 65,536 (eval.sh AR)',
    badge: 'Harness / Benchmarked Cap',
    minTokens: 512,
    maxTokens: 65_536,
    referencePoints: Object.freeze([512, 1024, 2048, 20480, 65536] as const),
    provenance:
      'Verified in Paper §6.5 (1024), chat/*.py & evaluate.py (512, max_thinking_tokens=6000), sglang_spark/launch_server.sh (CTX_LEN=2048), and eval.sh L91-104 (20480 dlm/linear_spec, 65536 ar). Zero published long-context evals exist in arXiv:2607.05722v1.',
  }),
  Object.freeze({
    tier: 3,
    label: '16,385 .. 262,144 (256K)',
    badge: 'Configured YaRN Ceiling (Unverified)',
    minTokens: 16_385,
    maxTokens: 262_144,
    referencePoints: Object.freeze([16385, 131072, 262144] as const),
    provenance:
      'Inherited from Ministral 3 YaRN factor=16.0 + llama_4_scaling_beta=0.1 in config.json & tokenizer_config.json; not empirically validated in paper.',
  }),
  Object.freeze({
    tier: 4,
    label: '> 262,144 (512K, 1M)',
    badge: 'Analytical Extrapolation',
    minTokens: 262_145,
    maxTokens: null,
    referencePoints: Object.freeze([524288, 1048576] as const),
    provenance:
      'Pure linear KV scaling (+160 KiB/tok on 14B) beyond configured positional ceiling (262,144).',
  }),
]);

// ============================================================================
// 24-ENTRY VERIFIED DISCREPANCY LEDGER (D1–D24, DF0 §4.1–§4.2 + CLARIFICATIONS)
// ============================================================================

export const DIFFUSION_DISCREPANCY_LEDGER: readonly DiffusionDiscrepancyEntry[] = Object.freeze([
  Object.freeze({
    id: 'D1',
    topic: 'Base Backbone Family',
    claimedOrExpected: 'Prior AI Studio dossier claimed Qwen3-14B Base.',
    verifiedGroundTruth:
      'Paper §5.1 & modeling_ministral.py: all NLD variants derive from Ministral 3 (Ministral3Model).',
    resolution: 'Cite Ministral 3 exclusively across code, specs, and readouts.',
    sourceRef: 'arXiv:2607.05722v1 §5.1; modeling_ministral.py',
  }),
  Object.freeze({
    id: 'D2',
    topic: 'Canonical HF Repo IDs',
    claimedOrExpected: 'Prompt & dossier cited nvidia/NVIDIA-Nemotron-Labs-Diffusion-14B-Base.',
    verifiedGroundTruth:
      'nvidia/NVIDIA-* returns HTTP 401; all 7 public repos omit NVIDIA- (canonical 14B Instruct: nvidia/Nemotron-Labs-Diffusion-14B).',
    resolution: 'Pin public repo IDs + exact 40-char commit SHAs.',
    sourceRef: 'Hugging Face nvidia/nemotron-labs-diffusion collection',
  }),
  Object.freeze({
    id: 'D3',
    topic: 'RoPE Type, rope_theta & Llama-4 Scale',
    claimedOrExpected: 'Dossier claimed standard RoPE theta=10,000 with no query scaling.',
    verifiedGroundTruth:
      'All models use YaRN RoPE (truncate=True) + Llama-4 query temperature scale (beta=0.1, orig=16384); 3B/8B use theta=1e6, 14B uses theta=1e9.',
    resolution: 'Implement exact YaRN (truncate=True) + Llama-4 query scaling.',
    sourceRef: 'config.json; modeling_ministral.py L104-106, L304-369',
  }),
  Object.freeze({
    id: 'D4',
    topic: '3B-Base / 8B-Base factor=0.25',
    claimedOrExpected: 'Expected max_position_embeddings=262,144 and yarn factor=16.0.',
    verifiedGroundTruth:
      '3B-Base & 8B-Base config.json export max_pos=4096, factor=0.25 (contradicting tokenizer.model_max_length=262144 and 14B-Base 262144/16.0).',
    resolution: 'Document verbatim in family table; canonical 14B Instruct is clean (262144/16.0).',
    sourceRef: '3B-Base/8B-Base config.json & tokenizer_config.json',
  }),
  Object.freeze({
    id: 'D5',
    topic: 'Special Token IDs',
    claimedOrExpected: 'Dossier claimed mask=131066, bos=131064, eos=131065.',
    verifiedGroundTruth:
      'bos=1 (<s>), mask=100 (<SPECIAL_100>), eos=11 (<|im_end|> Instruct) vs 2 (</s> Base config vs 11 Base tokenizer), pad=None; VLM-8B adds 131072 (|<MASK>|) yet keeps config.mask_token_id=100.',
    resolution: 'Report verified production IDs; map miniature V=36 to MASK=32, EOS=33, BOS=34, </think>=35.',
    sourceRef: 'config.json; tokenizer_config.json',
  }),
  Object.freeze({
    id: 'D6',
    topic: 'Block-Diffusion KV Caching',
    claimedOrExpected: 'Dossier claimed _CropableDynamicCache cropped every denoising step + bidirectional commit.',
    verifiedGroundTruth:
      'Denoising uses use_cache=False transient cat([old_k, K]) (no crop); commit is causal (use_cache=True, use_causal_mask=True) when causal_context=True; _crop_dynamic_cache is used in linear_spec_generate.',
    resolution: 'Model transient concat, causal/bidir commit, and speculative crop distinctly.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L477-519; modeling_ministral.py L159-162',
  }),
  Object.freeze({
    id: 'D7',
    topic: 'AR-Seeded Slot 0 in DLM',
    claimedOrExpected: 'Paper §3.2 & dossier assumed all B slots start as [MASK].',
    verifiedGroundTruth:
      'generate() L429-443 & L513-520: when causal_context=True (default), slot 0 is seeded from prior causal logits[:, -1], leaving only B-1 masks.',
    resolution: 'Implement both causal_context=True and causal_context=False explicitly.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L429-443, L513-520',
  }),
  Object.freeze({
    id: 'D8',
    topic: 'Latent ar_generate Missing use_causal_mask=True',
    claimedOrExpected: 'Docstring claims identical causal masking to MistralForCausalLM.',
    verifiedGroundTruth:
      'modeling_nemotron_labs_diffusion.py L567-573 omit use_causal_mask=True -> causal_mask=None; sdpa uses self.is_causal=True, while eager ignores is_causal and runs bidirectional prefill.',
    resolution: 'Always apply explicit causal mask in TS & PyTorch reference; document upstream bug.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L567-573; modeling_ministral.py L92-94',
  }),
  Object.freeze({
    id: 'D9',
    topic: 'Trained 4.8M Diffusion Sampler',
    claimedOrExpected: 'Paper §3.2 / App. A describe a 4-layer d=384 Transformer sampler.',
    verifiedGroundTruth:
      'Not shipped in any HF repo or GitHub (xp/dlm_api/dlm_generate/utils/sampler.py has only heuristic rules).',
    resolution: 'Documentation-only; implement released samplers.',
    sourceRef: 'arXiv:2607.05722v1 §3.2/App. A; xp/dlm_api/dlm_generate/utils/sampler.py',
  }),
  Object.freeze({
    id: 'D10',
    topic: 'Quadratic SS & Ensemble Verifier',
    claimedOrExpected: 'Paper §3.4 / App. C present Quadratic SS across 3B/8B/14B.',
    verifiedGroundTruth:
      'Absent from all 6 text checkpoints; shipped only in VLM-8B (sbd_inference_diffusion_quadratic / self_spec_generate).',
    resolution: 'Live text engine covers 4 released text paths; badged 5th Mask Explorer ref tab from VLM-8B.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py vs modeling_nemotron_labs_diffusion_vlm.py',
  }),
  Object.freeze({
    id: 'D11',
    topic: 'Stochastic Verification Coupling',
    claimedOrExpected: 'Paper cites Leviathan et al. speculative decoding.',
    verifiedGroundTruth:
      'linear_spec_generate L754-769 sample ar_tokens via multinomial and check raw equality ar_tokens[i] == block[i+1] (sum p_v q_v acceptance vs maximal coupling sum min(p_v, q_v)).',
    resolution: 'Implement exact upstream equality rule; document theoretical coupling gap.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L754-769',
  }),
  Object.freeze({
    id: 'D12',
    topic: 'NFE Counting Inconsistency',
    claimedOrExpected: 'All 3 methods return (output_ids, nfe) on a uniform basis.',
    verifiedGroundTruth:
      'ar_generate counts prefill in nfe=G; generate excludes prefill (nfe=sum(D_b)+K); linear_spec_generate includes prefill (nfe=1+sum(D_c)+C).',
    resolution: 'Expose 3 separate counters: actualForwardPasses, decodeForwardPasses, referenceMethodNfe.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L424-511, L581, L684-775',
  }),
  Object.freeze({
    id: 'D13',
    topic: 'Model Card Copy-Paste Errors',
    claimedOrExpected: 'Base READMEs cite subfolder="linear_spec_lora"; all READMEs use inputs.input_ids.',
    verifiedGroundTruth:
      '-Base repos lack linear_spec_lora/; variable on line 96 is prompt_ids, not inputs.',
    resolution: 'Cite verified chat/*.py scripts instead.',
    sourceRef: 'HF README.md; NVlabs/Nemotron-Labs-Diffusion chat/*.py',
  }),
  Object.freeze({
    id: 'D14',
    topic: 'dlm_paradigm & Stale Docstring',
    claimedOrExpected: 'Class docstring describes dual encoder A + causal decoder B.',
    verifiedGroundTruth:
      'All 6 config.json ship dlm_paradigm="bidirectional" (instantiating Ministral3Attention, not FlexAttention); only one self.encoder + one self.diffusion_head exist.',
    resolution: 'Document single-backbone architecture accurately.',
    sourceRef: 'config.json; modeling_nemotron_labs_diffusion.py',
  }),
  Object.freeze({
    id: 'D15',
    topic: '14B Param Count & dlm_loss_weight',
    claimedOrExpected: 'Dossier claimed 13,505,664,000 params; Paper Eq. 3 sets alpha=0.3.',
    verifiedGroundTruth:
      'Safetensors exact = 13,506,073,600 (+409,600 = 80 RMSNorm vectors). Text config.json sets dlm_loss_weight: null (implicit 1.0 in forward() L346); VLM-8B sets 0.5.',
    resolution: 'Use 0-diff safetensors counts; record 0.3 / null / 0.5 mismatch.',
    sourceRef: '14B model.safetensors header; config.json; forward() L346',
  }),
  Object.freeze({
    id: 'D16',
    topic: 'Default block_length, Overshoot, kvCapacity Boundary & Harness Hooks',
    claimedOrExpected: 'chat/ uses B=32; evaluate.py & eval.sh set dlm to B=8; tail-block clamping might shrink B to 1.',
    verifiedGroundTruth:
      'linear_spec_generate L771-794 can overshoot max_new_tokens by B-1; base.py L193-208 guards left-padding (ArNativeGeneration defaults shift_logits=False at ar_native.py:127); nemotron_mixed.py hooks unreleased NEMOTRON_CODE_REPO.',
    resolution:
      "Enforce kvCapacity=32 hard cap in browser and Clarification #2: never silently clamp B down to 1 — halt cleanly with lifecyclePhase='capacity_exhausted' whenever kvCapacity - cache_len < B (and < 1 for ar).",
    sourceRef: 'modeling_nemotron_labs_diffusion.py L771-794; base.py L193-208; ar_native.py:127',
  }),
  Object.freeze({
    id: 'D17',
    topic: 'Stateful diffusion_lm & _disable_adapters Mutation Across Calls',
    claimedOrExpected: 'Successive generate/ar_generate/linear_spec_generate calls are independent.',
    verifiedGroundTruth:
      'ar_generate() & linear_spec_generate() exit with self_attn.diffusion_lm=False; generate(..., causal_context=False) omits toggling diffusion_lm and runs CAUSAL attention during denoising; ar_generate()/generate() never disable attached PEFT LoRA.',
    resolution: 'Stateless TS execution passes explicit maskMode and loraEnabled per forward call.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L422-427, L502-514, L553-555, L674, L742',
  }),
  Object.freeze({
    id: 'D18',
    topic: 'linear_spec_generate(threshold > 0) Infinite while True Loop',
    claimedOrExpected: 'Iterative drafting terminates monotonically.',
    verifiedGroundTruth:
      'Because draft_logits[MASK] is never suppressed, argmax predicting token_mask_id at T=0 writes MASK back to MASK, leaving is_mask unchanged in an infinite while True loop.',
    resolution: 'Enforce badged maxDraftPassesPerCycle=B guard + non-progress detection without clamping raw MASK logits.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L707-736',
  }),
  Object.freeze({
    id: 'D19',
    topic: 'Verify-Crop-Before-EOS Cache Length Desync (Clarification #1)',
    claimedOrExpected: 'past_key_values length matches truncated generated sequence on EOS.',
    verifiedGroundTruth:
      '_crop_dynamic_cache(past_key_values, cache_len + accepted) runs at L775 before EOS truncation at L778-784, leaving accepted-(first_eos+1) post-EOS KVs in cache.',
    resolution:
      'Clarification #1: Keep clean browser post-EOS KV crop (browserPostEosCropLen = cache_len + first_eos + 1) while recording upstreamPreEosCropLen = cache_len + accepted in provenance.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L775-784',
  }),
  Object.freeze({
    id: 'D20',
    topic: 'Thinking-Budget </think> Output-vs-KV Desync & Seed Clobber',
    claimedOrExpected: 'Thinking-budget cutoff keeps emitted tokens and KV aligned.',
    verifiedGroundTruth:
      'In linear_spec_generate L787-791, next_token=end_think_token_id overwrites next_token without updating generated[-1][0,-1]; in generate() L461-465, offset=0 overwrites AR-seeded slot 0.',
    resolution: 'Document both in ledger; keep emitted tokens and committed KV strictly synchronized if toggled.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L787-791, L461-465',
  }),
  Object.freeze({
    id: 'D21',
    topic: 'DLM Prefill-EOS Waste (nfe=2) & batch_size>1 Premature .any() Break',
    claimedOrExpected: 'Prefill EOS halts immediately; batch rows finish independently.',
    verifiedGroundTruth:
      'If prefill next_token==eos, generate() still runs 1 denoise + 1 commit pass (actual=3, ref_nfe=2); L497 breaks on .any() rather than .all().',
    resolution: 'Single-stream (batch_size=1) in browser; document exact nfe=2 prefill-EOS behavior.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L429-445, L494-498, L504',
  }),
  Object.freeze({
    id: 'D22',
    topic: 'Zero-Token Boundary Crashes/Overruns',
    claimedOrExpected: 'max_new_tokens=0 returns empty sequence cleanly.',
    verifiedGroundTruth:
      'ar_generate(max_new_tokens=0) crashes at torch.cat([], dim=1) (L621); linear_spec_generate(max_new_tokens=0) still generates 1 prefill token.',
    resolution: 'Enforce maxNewTokens >= 1 validation cleanly.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L577-621, L686-698',
  }),
  Object.freeze({
    id: 'D23',
    topic: 'Training block_diff Asymmetric Llama-4 Scale & Ignored loss_mask',
    claimedOrExpected: 'x_t and x_0 share identical positional scaling during dual-stream training.',
    verifiedGroundTruth:
      'Noisy+clean concatenation doubles length to 2L so cache_position=0..2L-1 while position_ids=0..L-1; at L=16384, x_0 gets Llama-4 scale 1.069315 while x_t gets 1.0; ar_loss ignores loss_mask and L358 lacks zero-guard.',
    resolution: 'Surface 3-coordinate divergence (seq_idx vs ropePositionIds vs cachePositions) in Mask Explorer.',
    sourceRef: 'modeling_nemotron_labs_diffusion.py L115-134, L282-295, L350-365',
  }),
  Object.freeze({
    id: 'D24',
    topic: 'VLM Quadratic SS No-Op / temperature + Local Shader Fallback Trap',
    claimedOrExpected: 'VLM temperature scales Quadratic SS sampling; sheets render dynamic masks without textures.',
    verifiedGroundTruth:
      'VLM self_spec_generate L1344-1346 divides logits/temperature then calls argmax (always greedy); locally src/render/sheets.ts:351 falls back to geometric cellCol>cellRow when hasLiveTex==false.',
    resolution: 'Document VLM greedy-only bug in Quadratic Ref tab; always bind live RG32F G-channel textures.',
    sourceRef: 'modeling_nemotron_labs_diffusion_vlm.py L1343-1346; src/render/sheets.ts:347-361',
  }),
]);

// ============================================================================
// WEBGL2 RG32F TEXTURE ATLAS BUDGETS (DF0 §8.3 + CLARIFICATION #3)
// ============================================================================

export const DIFFUSION_ATLAS_BUDGETS: {
  readonly gqaRopeLora: DiffusionAtlasBudgetSpec;
  readonly swiglu: DiffusionAtlasBudgetSpec;
  readonly maskExplorer: DiffusionAtlasBudgetSpec;
} = Object.freeze({
  gqaRopeLora: Object.freeze({
    name: 'GQA+RoPE+LoRA',
    width: 256,
    height: 128,
    format: 'RG32F',
    bytesPerTexel: 8,
    totalBytes: 262_144, // 256 KiB
    maxAuthorizedWidth: 256,
    maxAuthorizedHeight: 256,
    maxAuthorizedBytes: 524_288, // 512 KiB authorized headroom for DF3
    notes:
      '256x128 (256 KiB) preferred when row-packing stays clean; DF3 explicitly authorized up to 256x256 (512 KiB) for GQA+RoPE+LoRA.',
  }),
  swiglu: Object.freeze({
    name: 'SwiGLU',
    width: 64,
    height: 64,
    format: 'RG32F',
    bytesPerTexel: 8,
    totalBytes: 32_768, // 32 KiB
    notes: '64x64 RG32F atlas for dense SwiGLU cutaway (gate/SiLU/up/hadamard/down/residual).',
  }),
  maskExplorer: Object.freeze({
    name: 'Mask Explorer',
    width: 64,
    height: 64,
    format: 'RG32F',
    bytesPerTexel: 8,
    totalBytes: 32_768, // 32 KiB
    notes: '64x64 RG32F atlas binding live G-channel mask visibility across all 5 mask modes.',
  }),
});
