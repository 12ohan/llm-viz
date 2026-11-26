/**
 * Canonical Architectural Specification, `0 B` LFS Reconciliation, Execution Views,
 * Context Honesty Tiers & `U1–U30` Discrepancy Ledger for Nemotron-Labs-Diffusion-Uni
 * (`src/uni/spec.ts`)
 *
 * Clean-room first-principles specification derived from:
 *   - Primary Paper: `arXiv:2610.36466v1` (`2026-10-26`, CC BY 4.0)
 *   - Canonical Checkpoint: `nvidia/NLD-Uni` @ `5db755a2eb0ab84654e3dcf80c7d0245bfe36a50`
 *   - Official Repository: `NVlabs/Nemotron-Labs-Diffusion-Uni` @ `16916949402ca738dcc74a7a37e0c43facafe347`
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import { formatGiB, formatKiB, formatMiB } from '../model/memory';
import type {
  UniAtlasBudgetSpec,
  UniAuxiliaryLfsEntry,
  UniCheckpointShardEntry,
  UniComponentBreakdownEntry,
  UniContextTierSpec,
  UniDiscrepancyEntry,
  UniExecutionViewSpec,
  UniLfsReconciliationSummary,
  UniMiniComponentFootprintEntry,
} from './types';

// ============================================================================
// RE-EXPORTS OF PINNED CORE CONSTANTS FROM `./types`
// ============================================================================

export {
  UNI_2D_SINCOS_CANONICAL_CONTENT_SIZE,
  UNI_2D_SINCOS_SCALE,
  UNI_BACKBONE_SEED,
  UNI_DYADIC_YARN_INV_FREQ,
  UNI_GAIN_PRESETS,
  UNI_MAX_SESSION_RETAINED_BYTES,
  UNI_MINI_BASE_VOCAB_SIZE,
  UNI_MINI_BOS_TOKEN_ID,
  UNI_MINI_BOX_CLOSE_TOKEN_ID,
  UNI_MINI_BOX_OPEN_TOKEN_ID,
  UNI_MINI_CONFIG,
  UNI_MINI_CORE_FLOAT_COUNT,
  UNI_MINI_CORE_TENSOR_COUNT,
  UNI_MINI_CORE_WEIGHT_BYTES,
  UNI_MINI_ENC_SLOT_TOKEN_ID,
  UNI_MINI_END_THINK_TOKEN_ID,
  UNI_MINI_EOS_TOKEN_ID,
  UNI_MINI_FIM_TOKEN_ID,
  UNI_MINI_FULL_FLOAT_COUNT,
  UNI_MINI_FULL_TENSOR_COUNT,
  UNI_MINI_FULL_WEIGHT_BYTES,
  UNI_MINI_GEN_EMBEDDING_ROWS,
  UNI_MINI_GEN_END_TOKEN_ID,
  UNI_MINI_GEN_FIXED_PREFIX_COUNT,
  UNI_MINI_GEN_FIXED_PREFIX_START_ID,
  UNI_MINI_GEN_IMAGE_END_ID,
  UNI_MINI_GEN_IMAGE_MASK_ID,
  UNI_MINI_GEN_RESERVED_ROWS,
  UNI_MINI_GEN_ROW_MARKER_ID,
  UNI_MINI_GEN_SLOT_TOKEN_ID,
  UNI_MINI_GEN_SPARE_ROW_ID,
  UNI_MINI_GEN_START_TOKEN_ID,
  UNI_MINI_IM_START_TOKEN_ID,
  UNI_MINI_LOC_BEGIN_TOKEN_ID,
  UNI_MINI_LOC_BIN_COUNT,
  UNI_MINI_LOC_END_ID,
  UNI_MINI_LOC_END_TOKEN_ID,
  UNI_MINI_LOC_START_ID,
  UNI_MINI_MASK_TOKEN_ID,
  UNI_MINI_NEWLINE_TOKEN_ID,
  UNI_MINI_PAD_TOKEN_ID,
  UNI_MINI_THINK_TOKEN_ID,
  UNI_MINI_TOTAL_RESIDENT_BYTES,
  UNI_MINI_TREE_BUFFER_BYTES,
  UNI_MINI_UND_PAD_TOKEN_ID,
  UNI_MINI_UNK_TOKEN_ID,
  UNI_MINI_VOCAB,
  UNI_MINI_VQ_CODEBOOK_SIZE,
  UNI_MINI_VQ_TREE_DEPTH,
  UNI_PROD_GEN_DOWNSAMPLE_SCALE,
  UNI_PROD_GEN_EMBEDDING_SCALE,
  UNI_PROD_RADIO_MULTIPLY,
} from './types';

// ============================================================================
// CANONICAL REFERENCES & 3-TIER LICENSE BREAKDOWN (§1.1 & §1.2)
// ============================================================================

export const UNI_CANONICAL_REFS = Object.freeze({
  primaryPaper: Object.freeze({
    id: 'arXiv:2610.36466v1',
    url: 'https://arxiv.org/abs/2610.36466',
    title: 'Nemotron-Labs-Diffusion-Uni (NLD-Uni)',
    date: '2026-10-26',
    license: 'CC BY 4.0',
  }),
  checkpoint: Object.freeze({
    repoId: 'nvidia/NLD-Uni',
    url: 'https://huggingface.co/nvidia/NLD-Uni',
    commit: '5db755a2eb0ab84654e3dcf80c7d0245bfe36a50',
    timestamp: '2026-10-28T00:45:27Z',
    releaseManifestTag: 'generation-only-step-10000',
    license: 'NVIDIA Non-Commercial License (§3.3)',
  }),
  githubRepo: Object.freeze({
    repoId: 'NVlabs/Nemotron-Labs-Diffusion-Uni',
    url: 'https://github.com/NVlabs/Nemotron-Labs-Diffusion-Uni',
    commit: '16916949402ca738dcc74a7a37e0c43facafe347',
    timestamp: '2026-10-28T00:43:07Z',
    license: 'NVIDIA Non-Commercial License (§3.3)',
  }),
  lineage: Object.freeze([
    Object.freeze({
      id: 'nld',
      name: 'Nemotron-Labs-Diffusion (NLD-8B / VLM-8B)',
      arxiv: 'arXiv:2607.05722v1',
      url: 'https://arxiv.org/abs/2607.05722',
      role: 'Base masked block-diffusion LLM / VLM lineage',
    }),
    Object.freeze({
      id: 'lavida_o',
      name: 'LaViDa-O (Elastic-MoT)',
      arxiv: 'arXiv:2509.19244',
      url: 'https://arxiv.org/abs/2509.19244',
      role: 'Elastic Mixture-of-Transformers architecture & upstream sampling lineage',
    }),
    Object.freeze({
      id: 'nld_image',
      name: 'Nemotron-Labs-Diffusion-Image (NLD-Image)',
      arxiv: 'arXiv:2606.29814',
      url: 'https://arxiv.org/abs/2606.29814',
      role: 'Predecessor discrete image diffusion baseline',
    }),
    Object.freeze({
      id: 'radio_h',
      name: 'RADIO-H (ViT-Huge/16)',
      citation: 'heinrich2025radiov2',
      role: 'Continuous semantic vision encoder (32L ViT, 1280-dim, 4 CLS + 6 registers)',
    }),
    Object.freeze({
      id: 'emu3_5_ibq',
      name: 'Emu3.5 IBQ VQ-VAE',
      arxiv: 'arXiv:2510.26583',
      url: 'https://arxiv.org/abs/2510.26583',
      role: 'Discrete visual tokenizer (stride 16, 131,072-entry 256-dim IBQ codebook)',
    }),
    Object.freeze({
      id: 'ministral_3',
      name: 'Ministral 3',
      arxiv: 'arXiv:2601.08584',
      url: 'https://arxiv.org/abs/2601.08584',
      role: '34-layer GQA (32Q/8KV) + 3.5x SwiGLU + YaRN/Llama-4 backbone template',
    }),
    Object.freeze({
      id: 'nemotron_3_nano_omni',
      name: 'Nemotron-3-Nano-Omni',
      arxiv: 'arXiv:2604.24954',
      url: 'https://arxiv.org/abs/2604.24954',
      role: 'Multimodal training lineage reference',
    }),
    Object.freeze({
      id: 'qwen_image',
      name: 'Qwen-Image',
      arxiv: 'arXiv:2508.02324',
      url: 'https://arxiv.org/abs/2508.02324',
      role: 'Continuous flow teacher for editing On-Policy Distillation',
    }),
  ] as const),
  licenseBreakdown: Object.freeze({
    modelWeightsAndArtifacts: Object.freeze({
      license: 'NVIDIA Non-Commercial License (§3.3)',
      licenseFileBytes: 4_115,
      useRestriction: 'Research or evaluation purposes only; non-commercial.',
      coveredArtifacts: Object.freeze([
        'model-00001-of-00008.safetensors .. model-00008-of-00008.safetensors',
        'pca_tree.pt',
        'tokenizer.json',
        'minimal_training/assets/image_groups.pt',
      ] as const),
    }),
    implementationCode: Object.freeze({
      rootLicense: 'NVIDIA Non-Commercial License (§3.3)',
      adobeConfidentialFiles: Object.freeze([
        'sampling.py',
        'positions.py',
        'pmj.py',
        'schedules.py',
        'image_utils.py',
        'utils.py',
      ] as const),
      dualHeaderNotes:
        'positions.py carries both # ADOBE CONFIDENTIAL (L1-11) and Apache-2.0 Haotian Liu (L13-25) headers.',
      nvidiaProprietaryFiles: Object.freeze([
        'radio_cls.py',
        'radio_patch.py',
      ] as const),
      derivedUpstreamFiles: Object.freeze(['vq_model.py'] as const),
    }),
    researchPapers: Object.freeze({
      license: 'CC BY 4.0',
      coveredPapers: Object.freeze([
        'arXiv:2610.36466v1',
        'arXiv:2607.05722v1',
        'arXiv:2509.19244',
        'arXiv:2606.29814',
      ] as const),
    }),
    cleanRoomPolicy:
      '100% clean-room original TypeScript (MIT repo) and clean-room PyTorch 2.15.0 verification scripts derived from mathematical specifications; zero remote source lines, weights, tokenizer payloads, or pca_tree.pt binaries copied or redistributed.',
  }),
});

// ============================================================================
// EXACT 8-SHARD SAFETENSORS INVENTORY & `0 B` LFS VERIFIER (§2.2A & §2.2B)
// ============================================================================

export const UNI_INDEX_TOTAL_SIZE_BYTES = 36_535_104_542 as const;
export const UNI_TOTAL_CHECKPOINT_TENSORS = 1_557 as const;
export const UNI_TOTAL_CHECKPOINT_ELEMENTS = 18_267_552_262 as const;
export const UNI_TOTAL_HEADER_FRAMING_BYTES = 199_680 as const;
export const UNI_TOTAL_FILE_BYTES = 36_535_304_222 as const;

export const UNI_CHECKPOINT_SHARDS: readonly UniCheckpointShardEntry[] = Object.freeze([
  Object.freeze({
    file: 'model-00001-of-00008.safetensors',
    shardIndex: 1,
    totalShards: 8,
    tensorCount: 84,
    headerJsonBytes: 10_056,
    headerFramingBytes: 10_064,
    payloadBytes: 4_974_624_768,
    totalFileBytes: 4_974_634_832,
    diffBytes: 0,
    sha256Prefix: 'c4d490e01047',
  }),
  Object.freeze({
    file: 'model-00002-of-00008.safetensors',
    shardIndex: 2,
    totalShards: 8,
    tensorCount: 106,
    headerJsonBytes: 12_688,
    headerFramingBytes: 12_696,
    payloadBytes: 4_999_806_976,
    totalFileBytes: 4_999_819_672,
    diffBytes: 0,
    sha256Prefix: '933c4b2f44f9',
  }),
  Object.freeze({
    file: 'model-00003-of-00008.safetensors',
    shardIndex: 3,
    totalShards: 8,
    tensorCount: 95,
    headerJsonBytes: 11_448,
    headerFramingBytes: 11_456,
    payloadBytes: 4_949_442_560,
    totalFileBytes: 4_949_454_016,
    diffBytes: 0,
    sha256Prefix: '6893a316ec66',
  }),
  Object.freeze({
    file: 'model-00004-of-00008.safetensors',
    shardIndex: 4,
    totalShards: 8,
    tensorCount: 106,
    headerJsonBytes: 12_760,
    headerFramingBytes: 12_768,
    payloadBytes: 4_999_806_976,
    totalFileBytes: 4_999_819_744,
    diffBytes: 0,
    sha256Prefix: '858fab295f10',
  }),
  Object.freeze({
    file: 'model-00005-of-00008.safetensors',
    shardIndex: 5,
    totalShards: 8,
    tensorCount: 106,
    headerJsonBytes: 12_792,
    headerFramingBytes: 12_800,
    payloadBytes: 4_999_806_976,
    totalFileBytes: 4_999_819_776,
    diffBytes: 0,
    sha256Prefix: '0d3160809362',
  }),
  Object.freeze({
    file: 'model-00006-of-00008.safetensors',
    shardIndex: 6,
    totalShards: 8,
    tensorCount: 102,
    headerJsonBytes: 12_288,
    headerFramingBytes: 12_296,
    payloadBytes: 4_999_806_976,
    totalFileBytes: 4_999_819_272,
    diffBytes: 0,
    sha256Prefix: '4356b837d798',
  }),
  Object.freeze({
    file: 'model-00007-of-00008.safetensors',
    shardIndex: 7,
    totalShards: 8,
    tensorCount: 956,
    headerJsonBytes: 127_344,
    headerFramingBytes: 127_352,
    payloadBytes: 4_455_896_094,
    totalFileBytes: 4_456_023_446,
    diffBytes: 0,
    sha256Prefix: '003f5a6c4c72',
  }),
  Object.freeze({
    file: 'model-00008-of-00008.safetensors',
    shardIndex: 8,
    totalShards: 8,
    tensorCount: 2,
    headerJsonBytes: 240,
    headerFramingBytes: 248,
    payloadBytes: 2_155_913_216,
    totalFileBytes: 2_155_913_464,
    diffBytes: 0,
    sha256Prefix: '299783fae2f5',
  }),
]);

export const UNI_AUXILIARY_LFS_ARTIFACTS: readonly UniAuxiliaryLfsEntry[] = Object.freeze([
  Object.freeze({
    file: 'pca_tree.pt',
    sizeBytes: 1_052_237,
    sha256Prefix: '4b7ee94fbb10',
    role: 'BalancedSemanticTree (depth=17, 131,072-code leaf permutation & heap radii)',
    redistributed: false,
  }),
  Object.freeze({
    file: 'tokenizer.json',
    sizeBytes: 17_268_792,
    sha256Prefix: 'f9ad3482369b',
    role: 'Ministral-3 BPE tokenizer extended to 132,101 tokens (+1,029 grounding tokens)',
    redistributed: false,
  }),
]);

export const UNI_COMPONENT_BREAKDOWN: readonly UniComponentBreakdownEntry[] = Object.freeze([
  Object.freeze({
    id: 'embed_tokens',
    label: 'encoder.embed_tokens.weight ([132101, 4096])',
    tensorCount: 1,
    elementCount: 541_085_696,
    dtype: 'BF16',
    payloadBytes: 1_082_171_392,
    branch: 'shared_io',
  }),
  Object.freeze({
    id: 'u_joint_layers',
    label: 'encoder.layers.0..17 — Understanding (U Joint, 18L)',
    tensorCount: 162,
    elementCount: 3_926_016_000,
    dtype: 'BF16',
    payloadBytes: 7_852_032_000,
    branch: 'u',
  }),
  Object.freeze({
    id: 'u_split_layers',
    label: 'encoder.layers.18..33 — Understanding (U Split, 16L)',
    tensorCount: 144,
    elementCount: 3_489_792_000,
    dtype: 'BF16',
    payloadBytes: 6_979_584_000,
    branch: 'u',
  }),
  Object.freeze({
    id: 'u_norm_and_head',
    label: 'encoder.norm.weight ([4096]) + diffusion_head.weight ([132101, 4096])',
    tensorCount: 2,
    elementCount: 541_089_792,
    dtype: 'BF16',
    payloadBytes: 1_082_179_584,
    branch: 'u',
  }),
  Object.freeze({
    id: 'g_joint_layers',
    label: 'encoder.layers.0..17 — Generation (G Joint, 18L)',
    tensorCount: 162,
    elementCount: 3_926_016_000,
    dtype: 'BF16',
    payloadBytes: 7_852_032_000,
    branch: 'g',
  }),
  Object.freeze({
    id: 'g_split_layers',
    label: 'encoder.layers.18..33 — Generation (G Split, 16L)',
    tensorCount: 144,
    elementCount: 3_489_792_000,
    dtype: 'BF16',
    payloadBytes: 6_979_584_000,
    branch: 'g',
  }),
  Object.freeze({
    id: 'g_norm_embed_predictor',
    label: 'encoder.norm_gen ([4096]) + gen_embedding ([131328, 4096]) + gen_predictor ([131072, 4096])',
    tensorCount: 3,
    elementCount: 1_074_794_496,
    dtype: 'BF16',
    payloadBytes: 2_149_588_992,
    branch: 'g',
  }),
  Object.freeze({
    id: 'packed_image_codec',
    label: 'encoder.packed_image_codec (down_norm, down_proj [4096,16384], up_norm, up_proj [16384,4096])',
    tensorCount: 4,
    elementCount: 134_225_920,
    dtype: 'BF16',
    payloadBytes: 268_451_840,
    branch: 'g',
  }),
  Object.freeze({
    id: 'vision_tower',
    label: 'encoder.vision_tower (RADIO-H 32L ViT: 387 BF16 + 1 I64 summary_idxs [3])',
    tensorCount: 388,
    elementCount: 651_645_443,
    dtype: 'BF16+I64',
    payloadBytes: 1_303_290_904,
    branch: 'vision',
  }),
  Object.freeze({
    id: 'mm_projector',
    label: 'encoder.mm_projector (RMSNorm [5120], Linear [4096,5120], Linear [4096,4096])',
    tensorCount: 3,
    elementCount: 37_753_856,
    dtype: 'BF16',
    payloadBytes: 75_507_712,
    branch: 'vision',
  }),
  Object.freeze({
    id: 'image_newline',
    label: 'encoder.image_newline ([4096] — dead parameter, never read in forward, U8)',
    tensorCount: 1,
    elementCount: 4_096,
    dtype: 'BF16',
    payloadBytes: 8_192,
    branch: 'dead',
    unusedInForward: true,
  }),
  Object.freeze({
    id: 'vqvae_encoder',
    label: 'encoder.vqvae.encoder + quant_conv (Emu3VQ encoder)',
    tensorCount: 246,
    elementCount: 184_599_552,
    dtype: 'BF16',
    payloadBytes: 369_199_104,
    branch: 'vqvae',
  }),
  Object.freeze({
    id: 'vqvae_codebook',
    label: 'encoder.vqvae.quantize.embedding ([131072, 256] IBQ codebook)',
    tensorCount: 1,
    elementCount: 33_554_432,
    dtype: 'BF16',
    payloadBytes: 67_108_864,
    branch: 'vqvae',
  }),
  Object.freeze({
    id: 'vqvae_decoder',
    label: 'encoder.vqvae.post_quant_conv + vqvae.decoder (Emu3VQ decoder)',
    tensorCount: 296,
    elementCount: 237_182_979,
    dtype: 'BF16',
    payloadBytes: 474_365_958,
    branch: 'vqvae',
  }),
]);

/**
 * Reconciles all 8 safetensors shard headers against `model.safetensors.index.json`
 * (`total_size: 36,535,104,542`) and the 14-row architectural component breakdown,
 * verifying exact `0 B` discrepancy across every shard and aggregate total.
 */
export function verifyUniLfsReconciliation(): UniLfsReconciliationSummary {
  let shardTensors = 0;
  let totalHeaderFramingBytes = 0;
  let totalPayloadBytes = 0;
  let totalFileBytes = 0;
  let shardFileMismatch = 0;
  let perShardOk = true;

  for (const shard of UNI_CHECKPOINT_SHARDS) {
    const expectedFraming = 8 + shard.headerJsonBytes;
    const expectedFileBytes = shard.headerFramingBytes + shard.payloadBytes;
    const computedDiff = shard.totalFileBytes - expectedFileBytes;

    if (
      shard.headerFramingBytes !== expectedFraming ||
      shard.diffBytes !== 0 ||
      computedDiff !== 0
    ) {
      perShardOk = false;
    }

    shardTensors += shard.tensorCount;
    totalHeaderFramingBytes += shard.headerFramingBytes;
    totalPayloadBytes += shard.payloadBytes;
    totalFileBytes += shard.totalFileBytes;
    shardFileMismatch += Math.abs(computedDiff);
  }

  let componentTensors = 0;
  let totalElements = 0;
  let componentPayloadBytes = 0;

  for (const comp of UNI_COMPONENT_BREAKDOWN) {
    componentTensors += comp.tensorCount;
    totalElements += comp.elementCount;
    componentPayloadBytes += comp.payloadBytes;
  }

  const payloadDiffBytes = totalPayloadBytes - UNI_INDEX_TOTAL_SIZE_BYTES;
  const fileDiffBytes = totalFileBytes - (totalHeaderFramingBytes + totalPayloadBytes);
  const componentMatchesShards =
    componentTensors === shardTensors &&
    componentPayloadBytes === totalPayloadBytes &&
    shardTensors === UNI_TOTAL_CHECKPOINT_TENSORS &&
    totalElements === UNI_TOTAL_CHECKPOINT_ELEMENTS;

  const allZeroDiff =
    perShardOk &&
    shardFileMismatch === 0 &&
    payloadDiffBytes === 0 &&
    fileDiffBytes === 0 &&
    componentMatchesShards;

  return Object.freeze({
    shardCount: UNI_CHECKPOINT_SHARDS.length,
    totalTensors: shardTensors,
    totalElements,
    totalHeaderFramingBytes,
    totalPayloadBytes,
    totalFileBytes,
    indexTotalSizeBytes: UNI_INDEX_TOTAL_SIZE_BYTES,
    payloadDiffBytes,
    fileDiffBytes,
    allZeroDiff,
  });
}

// ============================================================================
// PRODUCTION EXECUTION VIEWS & KV FOOTPRINTS (§2.2C & §3.1)
// ============================================================================

/** Single-branch Ministral-3 layer params: 41,943,040 (GQA) + 176,160,768 (SwiGLU) + 8,192 (2×RMSNorm). */
export const UNI_PRODUCTION_SINGLE_LAYER_BRANCH_PARAMS = 218_112_000 as const;

/** Single-stream BF16 KV cache bytes/token across 34 layers (34 * 2 * 8 * 128 * 2 B = 136.00 KiB/tok). */
export const UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16 = 139_264 as const;

/**
 * Parameters executed in `U` prefill during eager HF T2I / Image Editing that are causally
 * disconnected from `G` denoising (`U6`):
 *   - 16 split `U` decoder layers (`L18..33`): `16 * 218,112,000 = 3,489,792,000`
 *   - Final `encoder.norm` (`[4096]`): `4,096`
 * Total eager-HF wasted `U` parameters: `3,489,796,096`.
 */
export const UNI_WASTED_SPLIT_U_DECODER_PARAMS = 3_489_792_000 as const;
export const UNI_WASTED_SPLIT_U_TOTAL_PARAMS = 3_489_796_096 as const;

/** Full interleaved multimodal active params including Vision (`RADIO-H` + `Projector`) + `Emu3VQ` (excl. 4,099 dead elements). */
export const UNI_INTERLEAVED_FULL_MULTIMODAL_PARAMS = 18_267_548_163 as const;

/** SGLang VLM-only resident weight bytes (`9,187,382,784 BF16 params * 2 B = 18.37 GB / 17.11 GiB`). */
export const UNI_SGLANG_VLM_RESIDENT_BYTES = 18_374_765_568 as const;

const UNI_PCA_TREE_BYTES = 1_052_237;

export const UNI_EXECUTION_VIEWS: readonly UniExecutionViewSpec[] = Object.freeze([
  Object.freeze({
    id: 'text_understanding',
    label: 'Text-Only Understanding',
    paperShorthand: '"8B"',
    decoderBlocksSummary: '34U (7.42B)',
    decoderBlockParams: 7_415_808_000,
    activeCausalGraphParams: 8_497_983_488,
    eagerHfExecutedParams: 8_497_983_488,
    wastedSplitUParams: 0,
    residentHfBytes: UNI_INDEX_TOTAL_SIZE_BYTES,
    residentSglangBytes: UNI_SGLANG_VLM_RESIDENT_BYTES,
    cfgBranches: 1,
    kvBytesPerTokenBf16: UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16,
    releasedEndToEndLoop: true,
  }),
  Object.freeze({
    id: 'visual_understanding',
    label: 'Visual Understanding',
    paperShorthand: '"8B VLM" (Fig. 2)',
    decoderBlocksSummary: '34U (7.42B)',
    decoderBlockParams: 7_415_808_000,
    activeCausalGraphParams: 9_187_382_784,
    eagerHfExecutedParams: 9_187_382_784,
    wastedSplitUParams: 0,
    residentHfBytes: UNI_INDEX_TOTAL_SIZE_BYTES,
    residentSglangBytes: UNI_SGLANG_VLM_RESIDENT_BYTES,
    cfgBranches: 1,
    kvBytesPerTokenBf16: UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16,
    releasedEndToEndLoop: true,
  }),
  Object.freeze({
    id: 'object_grounding',
    label: 'Object Grounding',
    paperShorthand: '"8B VLM" (Fig. 2)',
    decoderBlocksSummary: '34U (7.42B)',
    decoderBlockParams: 7_415_808_000,
    activeCausalGraphParams: 9_187_382_784,
    eagerHfExecutedParams: 9_187_382_784,
    wastedSplitUParams: 0,
    residentHfBytes: UNI_INDEX_TOTAL_SIZE_BYTES,
    residentSglangBytes: UNI_SGLANG_VLM_RESIDENT_BYTES,
    cfgBranches: 1,
    kvBytesPerTokenBf16: UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16,
    releasedEndToEndLoop: true,
  }),
  Object.freeze({
    id: 'text_to_image',
    label: 'Text-to-Image (T2I)',
    paperShorthand: '"12B*" (4B U + 8B G, Fig. 2 & Tab. 2)',
    decoderBlocksSummary: '18U + 34G (52L, 11.34B)',
    decoderBlockParams: 11_341_824_000,
    activeCausalGraphParams: 13_362_667_523,
    eagerHfExecutedParams: 16_852_463_619,
    wastedSplitUParams: UNI_WASTED_SPLIT_U_TOTAL_PARAMS,
    residentHfBytes: UNI_INDEX_TOTAL_SIZE_BYTES + UNI_PCA_TREE_BYTES,
    residentSglangBytes: null,
    cfgBranches: 2,
    kvBytesPerTokenBf16: UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16 * 2,
    releasedEndToEndLoop: true,
  }),
  Object.freeze({
    id: 'image_edit',
    label: 'Image Editing',
    paperShorthand: 'T2I + RADIO-H + Emu3VQ Encoder',
    decoderBlocksSummary: '18U + 34G (52L, 11.34B)',
    decoderBlockParams: 11_341_824_000,
    activeCausalGraphParams: 14_236_666_371,
    eagerHfExecutedParams: 17_726_462_467,
    wastedSplitUParams: UNI_WASTED_SPLIT_U_TOTAL_PARAMS,
    residentHfBytes: UNI_INDEX_TOTAL_SIZE_BYTES + UNI_PCA_TREE_BYTES,
    residentSglangBytes: null,
    cfgBranches: 3,
    kvBytesPerTokenBf16: UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16 * 3,
    releasedEndToEndLoop: true,
  }),
  Object.freeze({
    id: 'interleaved_mixed',
    label: 'Interleaved Multimodal',
    paperShorthand: '"16B" (Fig. 2 & App. E.1)',
    decoderBlocksSummary: '34U + 34G (68L, 14.83B)',
    decoderBlockParams: 14_831_616_000,
    activeCausalGraphParams: 17_122_811_904,
    eagerHfExecutedParams: 17_122_811_904,
    wastedSplitUParams: 0,
    residentHfBytes: UNI_INDEX_TOTAL_SIZE_BYTES,
    residentSglangBytes: null,
    cfgBranches: 1,
    kvBytesPerTokenBf16: UNI_PRODUCTION_KV_BYTES_PER_TOKEN_BF16,
    releasedEndToEndLoop: false,
    badge: 'Kernel-Supported · No Released Multi-Turn Outer Loop',
  }),
]);

// ============================================================================
// 4-TIER CONTEXT HONESTY TABLE (`U3`)
// ============================================================================

export const UNI_CONTEXT_TIERS: readonly UniContextTierSpec[] = Object.freeze([
  Object.freeze({
    tier: 1,
    label: 'Tokenizer Config Default Ceiling',
    tokens: 8_192,
    source: 'nvidia/NLD-Uni tokenizer_config.json:164 (`model_max_length`)',
    notes:
      'Conservative tokenizer truncation limit; task-dependent effective limits noted in HF README.md:105.',
  }),
  Object.freeze({
    tier: 2,
    label: 'Verified SGLang Serving & Llama-4 Scale Unit',
    tokens: 16_384,
    source:
      'config.json (`original_max_position_embeddings: 16384`) & sglang/demo_sglang.py:50 (`context_length: 16384`)',
    notes:
      'Base pre-YaRN RoPE context and denominator of Llama-4 query temperature scaling `1 + 0.1 * ln(1 + floor(pos / 16384))`.',
  }),
  Object.freeze({
    tier: 3,
    label: 'Configured YaRN Maximum Position Ceiling',
    tokens: 131_072,
    source: 'nvidia/NLD-Uni config.json:9,14 (`max_position_embeddings: 131072`, `factor: 8.0`)',
    notes:
      '8x YaRN extension over 16,384 base context; single-stream BF16 KV cache reaches 17.000 GiB at 128K tokens.',
  }),
  Object.freeze({
    tier: 4,
    label: 'Lineage Contrast: NLD-8B YaRN Ceiling',
    tokens: 262_144,
    source: 'nvidia/Nemotron-Labs-Diffusion-8B (`max_position_embeddings: 262144`, `factor: 16.0`)',
    notes:
      'NLD-Uni reduces YaRN factor from 16.0 to 8.0 relative to NLD-8B; do not conflate NLD-8B 256K with NLD-Uni 128K.',
  }),
]);

// ============================================================================
// COMPLETE 30-ENTRY DISCREPANCY & BROWSER-GUARD LEDGER (`U1–U30`, §6)
// ============================================================================

export const UNI_DISCREPANCY_LEDGER: readonly UniDiscrepancyEntry[] = Object.freeze([
  Object.freeze({
    id: 'U1',
    numericId: 1,
    topic: 'Stale Paper Draft Title, Webpage Links & Card Heading',
    category: 'provenance_license',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'main.tex:163 titles NLD-Uni (arXiv:2610.36466), while main.tex:130 retains NLD-Image subtitle; HF/README.md:11,69 says -8B & links NL-Diffusion-Image/LICENSE; pages/index.html:84,94 links 2606.29814.',
    verifiedGroundTruth:
      'Overwritten LaTeX macro + un-updated project-page template (main.tex:130,163; pages/nl-diffusion-uni/index.html:84-115).',
    browserResolution:
      'Pin canonical arXiv:2610.36466v1, nvidia/NLD-Uni @ 5db755a2eb0ab84654e3dcf80c7d0245bfe36a50, and NVlabs/Nemotron-Labs-Diffusion-Uni @ 16916949402ca738dcc74a7a37e0c43facafe347.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U2',
    numericId: 2,
    topic: '3-Tier License & # ADOBE CONFIDENTIAL Headers',
    category: 'provenance_license',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Repo LICENSE is NVIDIA Non-Commercial (§3.3); HF README.md:69 calls it "NVIDIA Open Source License Agreement".',
    verifiedGroundTruth:
      'sampling.py:1-11, positions.py:1-25, pmj.py:1-11, schedules.py:1-11, image_utils.py:1-11, utils.py:1-11 carry # ADOBE CONFIDENTIAL; radio_cls/patch.py:1-7 carry proprietary NVIDIA headers.',
    browserResolution:
      '100% clean-room TypeScript + clean-room PyTorch 2.15.0 math from equations; zero source lines, weights, or pca_tree.pt redistributed.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U3',
    numericId: 3,
    topic: 'Context Ceiling Mismatch (131,072 vs. 8,192 vs. 16,384 vs. NLD 262,144)',
    category: 'provenance_license',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'config.json:9,14 sets max_position_embeddings=131072, factor=8.0 (vs. NLD-8B 262144, factor=16.0).',
    verifiedGroundTruth:
      'tokenizer_config.json:164 sets model_max_length=8192; sglang/demo_sglang.py:50 sets context_length=16384; HF/README.md:105 notes limits are task-dependent.',
    browserResolution:
      'Surface 4-Tier Context Honesty table (16K verified SGLang/Llama-4 unit scale vs 128K configured YaRN ceiling).',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U4',
    numericId: 4,
    topic: 'vocab_size=132,101 vs. Rollout/Teacher [:131072] Slice & Forbidden IDs',
    category: 'architecture_elastic',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'config.json:8 exports vocab_size = 132101 (+1,029 <LOC_*>/box tokens).',
    verifiedGroundTruth:
      'objectives.py:120-121 & loss_math.py:6 slice logits[:, :131072] and mask 18..29, 100 in OPD rollouts, whereas chat_utils.py:199,285 leaves all 132,101 logits unmasked at inference.',
    browserResolution:
      'Model full V_txt = 64 (with 17 <LOC_0..16> + 4 box/loc tags) and expose explicit logit-restriction provenance.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U5',
    numericId: 5,
    topic: 'dual_tower_layers = 18 (18 Joint + 16 Split) vs. Paper "Half / 4B"',
    category: 'architecture_elastic',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Paper (intro.tex:30, appendix.tex:453, t2i.tex:226) describes decoupling "the final half" (4B + 8B).',
    verifiedGroundTruth:
      'core.py:70-79 hardcodes configured = 18 (L0..17 joint = 3.926B [52.94%], L18..33 split = 3.490B [47.06%]); omitted from config.json & configuration_nlduni.py.',
    browserResolution:
      'Preserve exact 34-layer (18 joint + 16 split) topology in UNI_MINI_CONFIG and Macro 3D view.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U6',
    numericId: 6,
    topic: 'Selective Elastic Loading Claim vs. Full 18.27B Allocation & L18..33 Prefill Waste',
    category: 'architecture_elastic',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Paper Fig. 2 claims "Each view loads only the parameters needed for its task" (8B VLM, 12B T2I).',
    verifiedGroundTruth:
      'modeling_nlduni.py:82-91 loads all 1,557 tensors (18.27B = 34.03 GiB), and sampling.py:686 runs all 34 U layers + encoder.norm on the prompt in step 0 (3,489,796,096 wasted params), caching L18..33 U KVs (16/34 = 47.1% of prefix KV) that attention.py:81,129 drops; only SGLang (nlduni_radio_h.py:1806) skips non-U weights (9.19B).',
    browserResolution:
      'Distinguish Active Causal Graph (13.36B T2I / 14.24B Edit) vs Eager HF Executed (16.85B / 17.73B) vs Resident (18.27B); expose D5 live HF-Faithful vs Elastic-Pruned toggle verified to 0-ULP output equivalence.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U7',
    numericId: 7,
    topic: 'Paper "16B Interleaved View" vs. Missing End-to-End Inference Loop',
    category: 'architecture_elastic',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Paper Fig. 2 & App. E.1 present an interleaved 16B execution view.',
    verifiedGroundTruth:
      'core.py:59-68, 147-164 & attention.py:151-153 implement full mixed U+G routing, and Editing prefill uses it, but no public method implements multi-turn text+image interleaved decoding (sampling.py:392 plan="" is unexposed).',
    browserResolution:
      'Provide live kernel-faithful Mixed-Routing execution view badged "Kernel-Supported · No Released Multi-Turn Outer Loop".',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U8',
    numericId: 8,
    topic: 'Dead encoder.image_newline ([4096] BF16) & summary_idxs ([3] I64)',
    category: 'architecture_elastic',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Registered in modeling_nlduni.py:30 and vision.py:22 (8,216 B combined).',
    verifiedGroundTruth:
      'Never read in any forward method; fix_one_prompt (image_utils.py:45-52) tokenizes literal "\\n" instead; nlduni_radio_h.py:1634,1806 ignores both.',
    browserResolution:
      'Account for all 4,099 dead elements in 0 B safetensors reconciliation; retain image_newline in miniature parameter table with unusedInForward: true provenance.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U9',
    numericId: 9,
    topic: 'Reversed Token Age Polarity (age=0 Noisy -> age=N Clean in Inference)',
    category: 'diffusion_tree',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Paper (method.tex:18) & trainer (corruption.py:27) set a=0 clean and a=K masked.',
    verifiedGroundTruth:
      'Inference (tree.py:238-248, 409-414, sampling.py:534,772-788) sets unstarted stage = -1, active 0 -> ... -> N, where stage = N (committed) is clean (level=17, noise=0.0).',
    browserResolution:
      'Display both code stage (stage in {-1, 0..N}) and paper age (a_paper = N - max(0, stage)) side-by-side in UI formula readouts.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U10',
    numericId: 10,
    topic: 'Undocumented Additive Mask Embedding e(u) + (1 - a/N) * e(MASK)',
    category: 'diffusion_tree',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Paper §3.2 defines discrete state (c, a) via ball-tree replacement only.',
    verifiedGroundTruth:
      'Both tree.py:381-382 and corruption.py:70-71 add (1 - age/N) * gen_embedding(131073) in continuous embedding space whenever noise > 0.',
    browserResolution:
      'Implement exact build_semantic_tree_embeddings equation in TypeScript & PyTorch parity fixtures and visualize both terms on the Refinement Canvas.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U11',
    numericId: 11,
    topic: 'Tree depth = 17 vs. Paper D = 18 & Default refinement_steps = 4 vs. Paper N = 8',
    category: 'diffusion_tree',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Paper (method.tex:47, appendix.tex:99,384) uses 1-indexed node depth D=18 (K=17) and claims N=8 is default at 64 steps.',
    verifiedGroundTruth:
      'tree_metadata.json:6 & tree.py:16 use split depth 17; configuration_nlduni.py:22, modeling_nlduni.py:143, tree.py:17, and gradio_demo.py:47 all default to refinement_steps = 4.',
    browserResolution:
      'Default miniature to N = 4 (matching checkpoint default) with interactive selector N in {1, 2, 4, 8} on depth = 8 (256-code) tree.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U12',
    numericId: 12,
    topic: 'Paper Stochastic (tau_i, tau_i + eps) vs. Shift Quotas + seed=42 Stratified + N - 1 Flush Rounds',
    category: 'diffusion_tree',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Paper §3.2 / App. A.2.2 presamples tau_i ~ U([0, 1-eps]) at inference.',
    verifiedGroundTruth:
      'sampling.py:545-581 uses deterministic shift=5.0 quotas + hardcoded stratified_random(seed=42) (ignoring caller seed) + N - 1 flush rounds (64 + 3 = 67 denoise passes + 1 prefill = 68 passes).',
    browserResolution:
      'Implement exact semantic_tree_activation_counts + stratified_random + N - 1 flush schedule in live inference while testing continuous ages_at_time (Eq. 16) in parity suite.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U13',
    numericId: 13,
    topic: 'Undocumented PackedLinearImageCodec (2x2), 128 Fixed Prefix Tokens, +2 Row Markers & Scales',
    category: 'multimodal_codec',
    isBrowserGuard: false,
    paperOrConfigClaim: 'Omitted from paper architecture prose (method.tex:11-13).',
    verifiedGroundTruth:
      'modeling_nlduni.py:33-67, bridge.py:7-82, markers.py:58-182, positions.py:31-74 pack 64x66 -> 32x33 (1,056 tokens), scale by 0.0053573 x 0.4498539, add 0.0018 x 2D_SinCos, and prepend 128 uncompressed prefix tokens (131075..131202), all receiving 1D YaRN RoPE + Llama-4 scale.',
    browserResolution:
      'Implement every operator (+2 markers, 2x2 PackedLinearImageCodec, 4 fixed-prefix tokens, dual gain presets, 2D_SinCos + 1D YaRN) faithfully in miniature.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U14',
    numericId: 14,
    topic: 'IBQ Dot-Product Quantizer (argmax(z @ E^T)) vs. Squared-L2 Tree Metric',
    category: 'multimodal_codec',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Standard VQ-VAEs use Euclidean argmin ||z - e||_2^2; tree.py:14 uses normalized_squared_l2_tree_ultrametric.',
    verifiedGroundTruth:
      'vq.py:17-20 quantizes encoder latents via inner product (part @ self.embedding.weight.T).argmax(-1) to match IBQ softmax argmax.',
    browserResolution:
      'Use exact argmax(z @ E^T) in MiniEmu3VQ.encode and normalized_squared_l2 centered-PCA median splits in BalancedSemanticTree.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U15',
    numericId: 15,
    topic: 'Asymmetric SBD Windowing (ministral_full_attention_approx = True)',
    category: 'diffusion_tree',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'chat_utils.py:61-100 defines exact 512-padded FlexAttention SBD block mask.',
    verifiedGroundTruth:
      'sampling.py:289,468-474,703-706 defaults ministral_full_attention_approx = True: caches [0:start] causally, drops trailing <SPECIAL_22><|im_end|>, and runs bidirectional attention over [start:end].',
    browserResolution:
      'Support exact default windowing [0:start] causal prefix + [start:end] bidirectional image block and surface trailing-token exclusion in Mask Cutaway.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U16',
    numericId: 16,
    topic: 'Editing legacy edit_mode=1 Weighted Average vs. Extrapolative Dual-CFG',
    category: 'diffusion_tree',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Standard IP2P / independent CFG extrapolates away from dropped conditions.',
    verifiedGroundTruth:
      'modeling_nlduni.py:156,162 & sampling.py:199-205 (legacy, edit_mode=1) drop only discrete VQ in row 1 (retaining continuous RADIO-H), average (full + w_img * vq_drop) / (1 + w_img) (w_img=2.0), then extrapolate text (1 + w_txt) * cond - w_txt * uncond (w_txt=1.6).',
    browserResolution:
      'Implement all 3 upstream guidance modes (legacy default, independent, ip2p) with live 3-branch logit comparison in UI.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U17',
    numericId: 17,
    topic: 'Ignored timesteps in call_gen_predictor & Dead sch_temperatures',
    category: 'diffusion_tree',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'sampling.py:551-564, 616, 706 computes refinement_mask_fraction timesteps and cosine2 temperatures.',
    verifiedGroundTruth:
      'modeling_nlduni.py:64-67 ignores timesteps completely (no AdaLN exists), and sch_temperatures is never read when confidence_policy == "stratified".',
    browserResolution:
      'Document dead parameters; never fabricate nonexistent AdaLN gates.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U18',
    numericId: 18,
    topic: 'Source-Padding VQ Lock (lock_source_padding_vq) & Lanczos Un-Pad',
    category: 'multimodal_codec',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Non-square edits pad to square 1024x1024 (image_utils.py:19-22).',
    verifiedGroundTruth:
      'padding.py:30-90 & sampling.py:524-530,669-674,820-821 lock letterbox VQ cells to source codes/embeddings at every step; modeling_nlduni.py:167-171 crops & LANCZOS-resizes back.',
    browserResolution:
      'Include a non-square 16x8 miniature edit preset exercising _source_padding_vq_mask locking + center crop.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U19',
    numericId: 19,
    topic: 'No Speculative / LoRA Path & Conversation vs. chat_template.jinja Split',
    category: 'training_serving',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'NLD-14B had Tri-Mode + LoRA; HF/chat_template.jinja:205 opens <think>\\n.',
    verifiedGroundTruth:
      'NLD-Uni ships zero LoRA/speculative code; modeling_nlduni.py:183-190 uses Conversation (conversation.py:3-18, no <think>), whereas objectives.py:44 & chat_template.jinja inject <think>.',
    browserResolution:
      'Keep NLD-Uni text engine strictly aligned to released generate_with_prefix_cache_block_diff (no transplanted NLD LoRA/speculative modes).',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U20',
    numericId: 20,
    topic: '[Browser Guard] Stateful self_attn.diffusion_lm / mode Mutation Leak',
    category: 'browser_guard',
    isBrowserGuard: true,
    paperOrConfigClaim:
      'Sequential generate() / text_to_image() calls should be side-effect-free.',
    verifiedGroundTruth:
      'chat_utils.py:229-240, 312-321, 338-339 mutates layer.self_attn.mode and diffusion_lm without try...finally (leaking diffusion_lm=True on exception); core.py:218 ignores is_causal in eager mode when attention_mask=None.',
    browserResolution:
      'Stateless TypeScript forwardUniModel takes explicit maskMode & routing per call — zero mutable module flags.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U21',
    numericId: 21,
    topic: '[Browser Guard] Unsuppressed [MASK] (100) & Non-LOC Logits',
    category: 'browser_guard',
    isBrowserGuard: true,
    paperOrConfigClaim:
      'Block diffusion & grounding assume masks resolve cleanly to text / <LOC_*>.',
    verifiedGroundTruth:
      'chat_utils.py:141-186 never masks logits[..., mask_id] = -inf; grounding.py:83-86,127 runs unconstrained max(dim=-1) over 132,101 tokens and crashes with RuntimeError if a non-<LOC_*> token wins.',
    browserResolution:
      'Enforce safe browser guards (suppressMaskLogit in block decode & <LOC_0..16> domain mask in miniature grounding) while recording raw top-1 provenance if guard intervenes.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U22',
    numericId: 22,
    topic: '[Browser Guard] Float64 logits.exp() Overflow & log(0) in Gumbel Sampling',
    category: 'browser_guard',
    isBrowserGuard: true,
    paperOrConfigClaim:
      'chat_utils.py:136-139 & sampling.py:95-97 inject Gumbel noise.',
    verifiedGroundTruth:
      'logits.to(float64).exp() / (-log(rand))**T overflows for logits > 709.78, and uniform_(0, 1) includes 0.0 (-log(0) = +inf).',
    browserResolution:
      'Use log-domain Gumbel logits / T - log(-log(u)) (T > 0) over open-interval nextUniformOpenUnit() in [1e-15, 1 - 1e-15].',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U23',
    numericId: 23,
    topic: '[Browser Guard] Ambient Global RNG vs. Explicit Deterministic Variates',
    category: 'browser_guard',
    isBrowserGuard: true,
    paperOrConfigClaim:
      'text_to_image(seed=42) forks global CUDA/CPU RNG, while pmj.py:20 hardcodes random.Random(42).',
    verifiedGroundTruth:
      'sampling.py:507,739,780 pass generator=None, coupling draw order to tensor shapes/devices.',
    browserResolution:
      'Drive all browser randomness from seeded Xoshiro128** (zero ambient JS random) and support explicitVariates in TS + PyTorch 2.15.0 for 0-ULP stochastic transition parity.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U24',
    numericId: 24,
    topic: 'Idealized Paper OPD Equations vs. tail_kl (128+tail) & 0.2 * grouped_ce',
    category: 'training_serving',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Paper Eq. 2 & Eq. 19 show full-vocab KL, 1/t weighting, and plain VQ CE.',
    verifiedGroundTruth:
      'loss_math.py:18-75,92-98 & objectives.py:175-178,227-232 use 4.0 * reverse_tail_kl_129 + forward_tail_kl_129 / (0.5005 * R) and exact_ce + 0.2 * grouped_ce (16,384 clusters).',
    browserResolution:
      'Show both paper Eq. 2/19 and exact minimal_training formulas in the badged Read-Only OPD Explorer (0 gradients).',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U25',
    numericId: 25,
    topic: '[Browser Guard] Local sheets.ts:351 Fallback Trap & pick.ts:39 kvHeadIndex: 0|1',
    category: 'browser_guard',
    isBrowserGuard: true,
    paperOrConfigClaim:
      'src/render/sheets.ts falls back to cellCol > cellRow when !hasLiveTex; src/render/pick.ts types kvHeadIndex?: 0 | 1.',
    verifiedGroundTruth:
      'Any unbound sheetKind in {1, 2} silently corrupts bidirectional, rectangular [S, P+S], or split-tower masks.',
    browserResolution:
      'UniSheetBufferBuilder throws if sheetKind in {1, 2} has empty TexRect and always binds live RG32F G in {0.0, 1.0}; pickHitToUniCell uses groupIndex/headIndex/tapIndex/semanticTag with zero edits to src/render/.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U26',
    numericId: 26,
    topic: 'Prime-Depth D = 17 Train-vs-Inference 2x Ball-Tree Floor-Rounding Mismatch',
    category: 'diffusion_tree',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Training (corruption.py) and inference (tree.py) are assumed to use identical candidate ball sizes at each intermediate stage 1..K-1.',
    verifiedGroundTruth:
      'Because D = 17 is prime, minimal_training/corruption.py:58 computes 17 - floor(a_trn * 17 / 4) = [13, 9, 5] (balls 16, 256, 4096) whereas tree.py:243 computes floor(a_inf * 17 / 4) = [12, 8, 4] (balls 32, 512, 8192) — 2x wider candidate subtrees at every internal stage; tree.py:50-54 also hardcodes default 4.',
    browserResolution:
      'Add dedicated production-D = 17 integer floor-mismatch unit test in tests/uni-semantic-tree.test.ts and expose educational callout/overlay in View 4 (ballTreeScene.ts).',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U27',
    numericId: 27,
    topic: '<LOC_*> Grounding Tokens Silently Deleted by skip_special_tokens=True',
    category: 'multimodal_codec',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Calling model.generate() on grounding prompts should return <LOC_x1><LOC_y1><LOC_x2><LOC_y2> coordinates.',
    verifiedGroundTruth:
      'modeling_nlduni.py:204 decodes with skip_special_tokens=True while tokenizer_config.json & special_tokens_map.json mark all 1,029 tokens (131072..132100) as special: true, silently stripping bounding boxes unless return_token_ids=True or grounding.py:124 convert_ids_to_tokens is used.',
    browserResolution:
      'Browser grounding decodes token IDs directly via structured <LOC_0..16> lookup (never stripping grounding tokens) and surfaces the skip_special_tokens=True hazard in View 6.',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U28',
    numericId: 28,
    topic: 'minimal_training/ VLM SFT Double-Norm (~6.9x/1000x), Accidental <|im_end|> Supervision & OPD Tail Leak',
    category: 'training_serving',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'Reference trainer (minimal_training/) implements paper Eq. 2 & SFT masked-diffusion loss.',
    verifiedGroundTruth:
      '(a) objectives.py:142,168,175-176 + loss_math.py:73-75 divide VLM SFT masked CE by both time in [0.001, 1] and max(1, masked_count) (~6.915x avg inflation, up to 1,000x spike at t=0.001); (b) objectives.py:98 masked_fill(ids.eq(11), 11) unmasks system/user <|im_end|>; (c) loss_math.py:63 divides 131,072-vocab teacher tail mass by 132101 - 64 = 132,037, leaking tail mass onto 1,029 <LOC_*>/box tokens.',
    browserResolution:
      'Surface all three reference-trainer bugs in View 7 (Read-Only OPD Explorer) while keeping opdRef.ts mathematically clean and strictly forward-only (0 gradients).',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U29',
    numericId: 29,
    topic: 'SGLang install_nlduni_scheduler_metrics() NameError & Uninstalled NLDUniFastDiffuser',
    category: 'training_serving',
    isBrowserGuard: false,
    paperOrConfigClaim:
      'sglang/ provides a runnable NLD-Uni fast-diffusion serving demo & scheduler metric hook.',
    verifiedGroundTruth:
      'sglang/sglang_nlduni/algorithm.py:134-139 calls undefined _record_dllm_prefill_finished (immediate NameError); sglang/demo_sglang.py:35-55 passes dllm_algorithm="NLDUniFastDiffuser" (L53) without calling install_nlduni_fastdiffuser().',
    browserResolution:
      'Document in spec.ts runtime-support provenance (SGLang covers 9.19B VLM understanding only and requires patching both hooks).',
    epistemicStatus: 'VERIFIED',
  }),
  Object.freeze({
    id: 'U30',
    numericId: 30,
    topic: '[Browser Guard] Minor HF Edge-Case Drift (Padding/Crop Rounding, B > 1 EOS .any() Desync, T2I guidance_scale = 0 Wasted B = 2 CFG)',
    category: 'browser_guard',
    isBrowserGuard: true,
    paperOrConfigClaim:
      'Letterboxing, batched block diffusion, and unguided T2I (guidance_scale = 0) should behave consistently.',
    verifiedGroundTruth:
      '(a) Letterbox rounding drifts across image_utils.py:20, padding.py:38-39, modeling_nlduni.py:168-171, and grounding.py:103-104; (b) chat_utils.py:305-310 breaks on .any(), aborting unresolved rows when B > 1; (c) modeling_nlduni.py:143,152,158 passes default image_guidance_scale=2.0 into _sample, keeping run_cfg_branches True and running wasted B = 2 CFG when T2I sets guidance_scale=0.0.',
    browserResolution:
      'Miniature engine uses exact integer-centered even-dimension 16x8 -> 16x16 letterbox, enforces per-row EOS completion (B = 1 primary), and short-circuits T2I guidance_scale === 0 to B = 1.',
    epistemicStatus: 'VERIFIED',
  }),
]);

// ============================================================================
// MINIATURE RESIDENT FOOTPRINT & `RG32F` ATLAS BUDGETS (§8.3A & §8.3C)
// ============================================================================

export const UNI_MINI_COMPONENT_FOOTPRINT: readonly UniMiniComponentFootprintEntry[] =
  Object.freeze([
    Object.freeze({
      id: 'mini_u_io',
      label: 'Understanding IO (embed_tokens, norm, diffusion_head)',
      milestone: 'UNI1',
      tensorCount: 3,
      floatCount: 1_032,
      bytes: 4_128,
    }),
    Object.freeze({
      id: 'mini_joint_backbone',
      label: 'Backbone L0..17 Joint (162 U + 162 G)',
      milestone: 'UNI1',
      tensorCount: 324,
      floatCount: 70_848,
      bytes: 283_392,
    }),
    Object.freeze({
      id: 'mini_split_backbone',
      label: 'Backbone L18..33 Split (144 U + 144 G)',
      milestone: 'UNI1',
      tensorCount: 288,
      floatCount: 62_976,
      bytes: 251_904,
    }),
    Object.freeze({
      id: 'mini_g_io',
      label: 'Generation IO (norm_gen, gen_embedding, gen_predictor)',
      milestone: 'UNI1',
      tensorCount: 3,
      floatCount: 4_168,
      bytes: 16_672,
    }),
    Object.freeze({
      id: 'mini_packed_image_codec',
      label: 'PackedLinearImageCodec (down_norm, down_proj, up_norm, up_proj)',
      milestone: 'UNI2',
      tensorCount: 4,
      floatCount: 528,
      bytes: 2_112,
    }),
    Object.freeze({
      id: 'mini_radio_vision_and_projector',
      label: 'Mini RADIO-H (2L ViT) + RadioHProjector',
      milestone: 'UNI2',
      tensorCount: 30,
      floatCount: 8_288,
      bytes: 33_152,
    }),
    Object.freeze({
      id: 'mini_image_newline',
      label: 'encoder.image_newline ([8] — dead parameter, U8)',
      milestone: 'UNI2',
      tensorCount: 1,
      floatCount: 8,
      bytes: 32,
      unusedInForward: true,
    }),
    Object.freeze({
      id: 'mini_emu3_vqvae',
      label: 'Mini Emu3VQ (Codebook 1 + Encoder 56 + Decoder 56)',
      milestone: 'UNI2',
      tensorCount: 113,
      floatCount: 43_387,
      bytes: 173_548,
    }),
    Object.freeze({
      id: 'mini_semantic_tree_buffers',
      label: 'BalancedSemanticTree Buffers (leaf_to_token, token_to_leaf, heap_radii)',
      milestone: 'UNI3',
      tensorCount: 0,
      floatCount: 0,
      bytes: 3_068,
    }),
  ]);

export const UNI_MAX_ATLAS_BYTES = 524_288 as const; // 512.00 KiB

export const UNI_ATLAS_BUDGETS: readonly UniAtlasBudgetSpec[] = Object.freeze([
  Object.freeze({
    viewId: 'visual_input',
    label: 'View 2: Visual Input Pipeline (RADIO-H + VQ + Assembly)',
    width: 128,
    height: 128,
    bytes: 131_072,
    rectCount: 10,
    scalarMaskOnly: true,
  }),
  Object.freeze({
    viewId: 'joint_attn',
    label: 'View 3: Joint Attention + Dual-Branch SwiGLU Cutaway',
    width: 256,
    height: 256,
    bytes: 524_288,
    rectCount: 35,
    scalarMaskOnly: true,
  }),
  Object.freeze({
    viewId: 'ball_tree',
    label: 'View 4: Codebook Geometry & Ball-Tree Inspector',
    width: 128,
    height: 128,
    bytes: 131_072,
    rectCount: 5,
    scalarMaskOnly: true,
  }),
  Object.freeze({
    viewId: 'refinement_output_edit',
    label: 'Views 5 & 6: Image Refinement Canvas & Output/Editing',
    width: 128,
    height: 128,
    bytes: 131_072,
    rectCount: 14,
    scalarMaskOnly: true,
  }),
  Object.freeze({
    viewId: 'opd_readonly',
    label: 'View 7: Cross-Paradigm OPD Reference (Read-Only)',
    width: 128,
    height: 64,
    bytes: 65_536,
    rectCount: 5,
    scalarMaskOnly: true,
  }),
]);

// ============================================================================
// DUAL SI / IEC BYTE FORMATTER
// ============================================================================

const BYTES_PER_KIB = 1_024;
const BYTES_PER_MIB = 1_048_576;
const BYTES_PER_GIB = 1_073_741_824;

/**
 * Format a nonnegative byte count with both SI (base-10: KB/MB/GB) and IEC (base-2: KiB/MiB/GiB)
 * representations, delegating IEC unit formatting to `../model/memory`.
 */
export function formatSiIecBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    throw new RangeError(`bytes must be a finite nonnegative number, received ${bytes}`);
  }
  if (bytes < 1_000) {
    return `${Math.round(bytes)} B`;
  }

  let si: string;
  if (bytes >= 1_000_000_000) {
    si = `${(bytes / 1_000_000_000).toFixed(3)} GB`;
  } else if (bytes >= 1_000_000) {
    si = `${(bytes / 1_000_000).toFixed(2)} MB`;
  } else {
    si = `${(bytes / 1_000).toFixed(2)} KB`;
  }

  let iec: string;
  if (bytes >= BYTES_PER_GIB) {
    iec = formatGiB(bytes, 3);
  } else if (bytes >= BYTES_PER_MIB) {
    iec = formatMiB(bytes, 2);
  } else if (bytes >= BYTES_PER_KIB) {
    const kib = bytes / BYTES_PER_KIB;
    iec = Number.isInteger(kib) ? formatKiB(bytes) : `${kib.toFixed(2)} KiB`;
  } else {
    iec = `${Math.round(bytes)} B`;
  }

  return `${si} (${iec})`;
}

// ============================================================================
// DISCLOSED SCALING BADGES & SGLANG RUNTIME PROVENANCE (§8.2 & U29)
// ============================================================================

export const UNI_PROJECTION_REVERSAL_BADGE = Object.freeze({
  id: 'uni_projection_reversal',
  label: 'Miniature GQA Projection Expansion (8 -> 64 Q / 16 KV)',
  miniatureSummary: 'd_model = 8 expands to 64 Q (8 heads × headDim 8) and 16 KV (2 heads × headDim 8)',
  productionSummary: 'd_model = 4096 maps to 4096 Q (32 heads × headDim 128) and 1024 KV (8 heads × headDim 128)',
  rationale:
    'Preserves exact 4:1 GQA ratio and 4 YaRN rotary frequency planes (headDim = 8) while keeping the residual stream compact (d = 8) for 256x256 RG32F cutaway atlases.',
});

export const UNI_VISUAL_QUALITY_HONESTY_BADGE = Object.freeze({
  id: 'uni_visual_quality_honesty',
  label: 'Miniature 16×16 Mechanics Sim',
  summary:
    'Deterministic seeded miniature RADIO-H / Emu3VQ weights execute real convolutions, IBQ dot-product quantization, and ball-tree self-refinement math — never presented as photorealistic synthesis.',
  resolution: '16×16 RGB (4×4 VQ grid, 256-code depth-8 tree)',
  productionResolution: '1024×1024 RGB (64×64 VQ grid, 131,072-code depth-17 tree)',
});

export const UNI_SGLANG_PROVENANCE = Object.freeze({
  supportedViews: Object.freeze([
    'text_understanding',
    'visual_understanding',
    'object_grounding',
  ] as const),
  unsupportedViews: Object.freeze([
    'text_to_image',
    'image_edit',
    'interleaved_mixed',
  ] as const),
  loadedParams: 9_187_382_784,
  residentBytes: UNI_SGLANG_VLM_RESIDENT_BYTES,
  loaderReference: 'sglang/sglang_nlduni/nlduni_radio_h.py:1806',
  skippedSubsystems: Object.freeze([
    'encoder.layers.*.*_gen',
    'encoder.norm_gen',
    'encoder.gen_embedding',
    'encoder.gen_predictor',
    'encoder.packed_image_codec',
    'encoder.vqvae',
    'encoder.image_newline',
    'encoder.vision_tower.summary_idxs',
  ] as const),
  contextLengthDefault: 16_384,
  knownUpstreamBugs: Object.freeze([
    Object.freeze({
      discrepancyId: 'U29',
      location: 'sglang/sglang_nlduni/algorithm.py:134-139',
      issue:
        'install_nlduni_scheduler_metrics() references undefined _record_dllm_prefill_finished (immediate NameError).',
    }),
    Object.freeze({
      discrepancyId: 'U29',
      location: 'sglang/demo_sglang.py:35-55',
      issue:
        'Passes dllm_algorithm="NLDUniFastDiffuser" (L53) without calling install_nlduni_fastdiffuser().',
    }),
  ] as const),
});
