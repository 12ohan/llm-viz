/**
 * Architecture Specification, Physical-72 Layer Table & 4-Checkpoint Reconciliation
 * for NVIDIA Nemotron-Labs-Dory1.0-4B (`src/dory/spec.ts`).
 *
 * Implements `DorySpecModuleContract` (`D1`, `D2`, `D3`, `D10`, `DR1–DR4`, `DR11`, `DR14–DR16`).
 * 100% clean-room original TypeScript — zero DOM, WebGL, Node built-ins, or sibling imports.
 */

import {
  DORY_MINI_CONFIG,
  DORY_MINI_VOCAB,
  DORY_MOTIF_PATTERN,
  type DoryAttentionKind,
  type DoryCheckpointFormatSpec,
  type DoryKvBankCountSummary,
  type DoryKvCacheMode,
  type DoryLayerStage,
  type DoryMiniConfig,
  type DoryMiniKvMemorySummary,
  type DoryPhysicalLayerSpec,
  type DoryProductionKvPoint,
  type DoryProductionSpec,
  type DoryRopeRegime,
  type DoryScalingRatioEntry,
  type DorySublayerKind,
} from './types';

export { DORY_MINI_CONFIG } from './types';

// ============================================================================
// PRODUCTION MODEL SPECIFICATION (`BF16/config.json` @ `ee517dd3`)
// ============================================================================

export const DORY_PRODUCTION_SPEC: DoryProductionSpec = Object.freeze({
  modelName: 'NVIDIA-Nemotron-Labs-Dory1.0-4B',
  architectureName: 'DoryForCausalLM',
  modelType: 'dory',
  totalParams: 3_975_229_440,
  tensorCountBf16: 507,
  numHiddenLayers: 72,
  numPairedLayers: 36,
  nInputLayers: 16,
  nRecurrentLayers: 40,
  nOutputLayers: 16,
  hiddenSize: 2560,
  intermediateSize: 7168,
  numAttentionHeads: 16,
  numKeyValueHeads: 4,
  headDim: 256,
  qProjOutDim: 4096,
  kvProjOutDim: 1024,
  vocabSize: 131072,
  slidingWindow: 1024,
  maxPositionEmbeddings: 262144,
  ropeThetaSliding: 10_000,
  ropeThetaFull: 10_000_000,
  initializerRange: 0.0177,
  ngptAlphaInit: 0.05,
  attentionOutputGateLogitScaleInit: null,
  l2NormEps: 1e-6,
  defaultRecurrentLoops: 2,
  defaultKvCacheMode: 'per_loop',
  kvBytesPerTokenPerLayerBf16: 4096,
  weightLicense: 'OpenMDW-1.1',
});

// ============================================================================
// 0-BYTE DIFF 4-CHECKPOINT RECONCILIATION CONSTANTS (AUDIT §2.1 & §3.2)
// ============================================================================

export const DORY_CHECKPOINT_FORMATS: readonly DoryCheckpointFormatSpec[] = Object.freeze([
  Object.freeze({
    id: 'bf16',
    label: 'BF16 (Unquantized)',
    repoId: 'nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16',
    pinnedCommitSha: 'ee517dd388a9b348890ae1e00dacc5eb4d0c750d',
    quantScheme: 'W16A16 BF16 unquantized safetensors (507 BF16 tensors)',
    tensorCount: 507,
    unquantizedBf16TensorCount: 507,
    quantizedLinearMatrixCount: 0,
    logicalParams: 3_975_229_440,
    storedElements: 3_975_229_440,
    headerJsonBytes: 56_584,
    payloadBytes: 7_950_458_880,
    totalFileBytes: 7_950_515_472,
    diffBytes: 0,
    lfsSha256Prefix: '207483664dfe',
    license: 'OpenMDW-1.1',
    scaleStorageNote:
      'All 217 learned nGPT scale tensors stored as BF16 on disk; upcast to FP32 at runtime in vLLM/llama.cpp (DR14).',
  }),
  Object.freeze({
    id: 'mxfp8',
    label: 'MXFP8 (PTQ W8A16)',
    repoId: 'nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8',
    pinnedCommitSha: 'f57fc2d5353e0a3847ae0544c33e5601672f4065',
    quantScheme:
      'W8A16 compressed-tensors (group_size=32, 288 F8_E4M3 matrices + 288 U8 E8M0 scales; observer=mse exact_e8m0_search_with_dominance_bounds)',
    tensorCount: 795,
    unquantizedBf16TensorCount: 219,
    quantizedLinearMatrixCount: 288,
    logicalParams: 3_975_229_440,
    storedElements: 4_078_448_640,
    headerJsonBytes: 91_616,
    payloadBytes: 4_750_663_680,
    totalFileBytes: 4_750_755_304,
    diffBytes: 0,
    lfsSha256Prefix: '2dd3ff9600a6',
    license: 'OpenMDW-1.1',
    scaleStorageNote:
      '219 unquantized BF16 tensors (embeddings, lm_head, 217 nGPT scales); 288 linear matrices quantized via 2-candidate MSE E8M0 search (DR14, DR15).',
  }),
  Object.freeze({
    id: 'nvfp4',
    label: 'NVFP4 (QAD W4A16)',
    repoId: 'nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4',
    pinnedCommitSha: '17065a6849266dd478221735be7747689a0a4331',
    quantScheme:
      'W4A16 modelopt QAD (group_size=16, 288 packed U8 2xFP4 + 288 F8_E4M3 block scales + 288 F32 weight_scale_2)',
    tensorCount: 1_083,
    unquantizedBf16TensorCount: 219,
    quantizedLinearMatrixCount: 288,
    logicalParams: 3_975_229_440,
    storedElements: 2_530_160_928,
    headerJsonBytes: 121_192,
    payloadBytes: 3_202_376_832,
    totalFileBytes: 3_202_498_032,
    diffBytes: 0,
    lfsSha256Prefix: 'ca08dec5d70b',
    license: 'OpenMDW-1.1',
    scaleStorageNote:
      '219 unquantized BF16 tensors (embeddings, lm_head, 217 nGPT scales); 288 linear matrices quantized in NVFP4 QAD (DR14).',
  }),
  Object.freeze({
    id: 'q5_k_m',
    label: 'GGUF Q5_K_M (QAD)',
    repoId: 'nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF',
    pinnedCommitSha: 'df53755cb8741843af8e019e088819fbef17e29b',
    quantScheme: 'GGUF Q5_K_M linear weights + F32 1D nGPT scale tensors (conversion/dory.py)',
    tensorCount: null,
    unquantizedBf16TensorCount: null,
    quantizedLinearMatrixCount: 288,
    logicalParams: 3_975_229_440,
    storedElements: null,
    headerJsonBytes: null,
    payloadBytes: null,
    totalFileBytes: 3_673_288_864,
    diffBytes: null,
    lfsSha256Prefix: '96080be49ffd',
    license: 'OpenMDW-1.1',
    scaleStorageNote:
      'GGUF container stores all 217 1D nGPT scale tensors directly in F32 (conversion/dory.py:L13-19, DR14).',
  }),
]);

// ============================================================================
// EXPLICIT MINIATURE-TO-PRODUCTION SCALING RATIO DISCLOSURE (D2, AUDIT §11.1-11.2)
// ============================================================================

export const DORY_SCALING_DISCLOSURE: readonly DoryScalingRatioEntry[] = Object.freeze([
  Object.freeze({
    id: 'physical_topology',
    label: 'Physical Sublayer Topology',
    productionValue: '72 sublayers (16 + 40 + 16), 507 tensors',
    miniatureValue: '72 sublayers (16 + 40 + 16), 507 tensors',
    productionRatio: '1.00x',
    miniatureRatio: '1.00x (1:1 topology & tensor count)',
    disclosureNote:
      'Full 72-sublayer physical depth and 3:1 SWA:Full repeating motif are preserved 1:1 with production BF16.',
  }),
  Object.freeze({
    id: 'hidden_size',
    label: 'Hidden Width (d)',
    productionValue: '2,560',
    miniatureValue: '8',
    productionRatio: '1.00x d',
    miniatureRatio: '1/320x prod (8-channel visual grammar)',
    disclosureNote:
      'Matches the 8-channel residual sheet grammar shared across Nano, TwoTower, Diffusion-14B, and NLD-Uni.',
  }),
  Object.freeze({
    id: 'q_proj_expansion',
    label: 'Attention Query/Gate Expansion (q_dim = h_q * d_h)',
    productionValue: '4,096 (h_q=16, d_h=256)',
    miniatureValue: '64 (h_q=8, d_h=8)',
    productionRatio: '1.60x d',
    miniatureRatio: '8.00x d',
    disclosureNote:
      'Preserves qualitative Q/Gate expansion (q_dim > d) and 4:1 GQA with 4 rotary planes; miniature q_dim/d is 8.00x vs. production 1.60x.',
  }),
  Object.freeze({
    id: 'kv_proj_width',
    label: 'Attention KV Width (kv_dim = h_kv * d_h)',
    productionValue: '1,024 (h_kv=4, d_h=256)',
    miniatureValue: '16 (h_kv=2, d_h=8)',
    productionRatio: '0.40x d',
    miniatureRatio: '2.00x d',
    disclosureNote:
      'Preserves exact 4:1 GQA head ratio (8:2) and d_h=8 rotary resolution; miniature kv_dim/d is 2.00x vs. production 0.40x contraction.',
  }),
  Object.freeze({
    id: 'mlp_ratio',
    label: 'SwiGLU MLP Intermediate (d_ff)',
    productionValue: '7,168',
    miniatureValue: '24',
    productionRatio: '2.80x d',
    miniatureRatio: '3.00x d',
    disclosureNote:
      'Compact multiple of 8 (3.00x d vs. production 2.80x d) matching Diffusion-14B miniature geometry.',
  }),
  Object.freeze({
    id: 'dual_rope_bases',
    label: 'Dual-RoPE Frequency Regimes (100% NeoX rotate_half)',
    productionValue: 'theta_S = 10,000, theta_F = 10,000,000 (128 planes)',
    miniatureValue: 'theta_S = 16, theta_F = 1,024 (4 planes)',
    productionRatio: '1,000x theta_F/theta_S',
    miniatureRatio: '64x theta_F/theta_S (dyadic SWA [1, 1/2, 1/4, 1/8])',
    disclosureNote:
      'Calibrated for S_max=32 / W=8 so sliding frequencies cycle within local context while full frequencies remain quasi-static.',
  }),
  Object.freeze({
    id: 'context_and_window',
    label: 'Max Context & Sliding Window (S_max, W)',
    productionValue: 'S_max = 262,144, W = 1,024',
    miniatureValue: 'S_max = 32, W = 8, cap_swa = 16',
    productionRatio: 'S_max / W = 256x',
    miniatureRatio: 'S_max / W = 4x (eviction begins at pos 8)',
    disclosureNote:
      'Trajectories cross SWA eviction after 8 tokens and exercise cap_swa=16 transaction-local rollback + DR24 L>16 transient workspaces.',
  }),
  Object.freeze({
    id: 'vocab_size',
    label: 'Untied Vocabulary Size (V)',
    productionValue: '131,072 (untied)',
    miniatureValue: '64 (untied)',
    productionRatio: '1.00x',
    miniatureRatio: '1/2048x (preserves all 11 special token IDs)',
    disclosureNote:
      'Retains exact production IDs for 0=<unk>, 1=<s>, 2=</s>, 10=<|im_start|>, 11=<|im_end|>, 12..17 think/tool tags.',
  }),
  Object.freeze({
    id: 'ngpt_constants',
    label: 'Active nGPT Runtime Constants',
    productionValue: 'sigma_0=0.0177, alpha_init=0.05, g_0=sqrt(d), eps=1e-6',
    miniatureValue: 'sigma_0=0.0177, alpha_init=0.05, g_0=sqrt(8), eps=1e-6',
    productionRatio: '1.00x',
    miniatureRatio: '1.00x (unmodified production constants)',
    disclosureNote:
      'Zero magic simplifications: initializer_range=0.0177 remains a live inference-time divisor across 4 of the 5 scale families.',
  }),
]);

// ============================================================================
// CANONICAL PHYSICAL-72 SUBLAYER TABLE (`0..71` <-> PAIRED `0..35`, D1 / DR4)
// ============================================================================

function buildPhysicalLayersTable(): readonly DoryPhysicalLayerSpec[] {
  const layers: DoryPhysicalLayerSpec[] = [];
  for (let p = 0; p < DORY_MINI_CONFIG.numHiddenLayers; p++) {
    const pairedLayerIdx = p >> 1;
    const stage: DoryLayerStage =
      p < DORY_MINI_CONFIG.nInputLayers
        ? 'input'
        : p < DORY_MINI_CONFIG.nInputLayers + DORY_MINI_CONFIG.nRecurrentLayers
          ? 'recurrent'
          : 'output';
    const motifSlot = p % DORY_MINI_CONFIG.motifPeriod;
    const sublayerKind: DorySublayerKind = DORY_MOTIF_PATTERN[motifSlot];
    const attentionKind: DoryAttentionKind | null =
      sublayerKind === 'mlp' ? null : sublayerKind;
    const ropeRegime: DoryRopeRegime | null =
      sublayerKind === 'sliding_attention'
        ? 'sliding'
        : sublayerKind === 'full_attention'
          ? 'full'
          : null;
    const ropeThetaMini: number | null =
      sublayerKind === 'sliding_attention'
        ? DORY_MINI_CONFIG.ropeThetaSliding
        : sublayerKind === 'full_attention'
          ? DORY_MINI_CONFIG.ropeThetaFull
          : null;
    const ropeThetaProd: number | null =
      sublayerKind === 'sliding_attention'
        ? DORY_PRODUCTION_SPEC.ropeThetaSliding
        : sublayerKind === 'full_attention'
          ? DORY_PRODUCTION_SPEC.ropeThetaFull
          : null;

    let stageAttentionOrdinal: number | null = null;
    if (attentionKind !== null) {
      if (stage === 'input') {
        stageAttentionOrdinal = p >> 1;
      } else if (stage === 'recurrent') {
        stageAttentionOrdinal = (p - DORY_MINI_CONFIG.nInputLayers) >> 1;
      } else {
        stageAttentionOrdinal =
          (p - (DORY_MINI_CONFIG.nInputLayers + DORY_MINI_CONFIG.nRecurrentLayers)) >> 1;
      }
    }

    const globalAttentionOrdinal: number | null = attentionKind !== null ? p >> 1 : null;
    const kindBadge =
      sublayerKind === 'sliding_attention'
        ? 'SWA'
        : sublayerKind === 'full_attention'
          ? 'Full'
          : 'MLP';
    const transformersPairBadge = `Pair L${pairedLayerIdx} · ${kindBadge}`;

    layers.push(
      Object.freeze({
        physicalLayerIdx: p,
        pairedLayerIdx,
        stage,
        motifSlot,
        sublayerKind,
        attentionKind,
        ropeRegime,
        ropeThetaMini,
        ropeThetaProd,
        stageAttentionOrdinal,
        globalAttentionOrdinal,
        transformersPairBadge,
      }),
    );
  }
  return Object.freeze(layers);
}

export const DORY_PHYSICAL_LAYERS: readonly DoryPhysicalLayerSpec[] = buildPhysicalLayersTable();

// ============================================================================
// LAYER, DEPTH, KV MEMORY & TOKEN DECODING HELPERS
// ============================================================================

function assertValidPhysicalLayerIdx(physicalLayerIdx: number): void {
  if (
    !Number.isInteger(physicalLayerIdx) ||
    physicalLayerIdx < 0 ||
    physicalLayerIdx >= DORY_MINI_CONFIG.numHiddenLayers
  ) {
    throw new RangeError(
      `Invalid Dory physicalLayerIdx ${String(physicalLayerIdx)}: expected integer in [0, ${DORY_MINI_CONFIG.numHiddenLayers - 1}].`,
    );
  }
}

export function getDoryPhysicalLayerSpec(physicalLayerIdx: number): DoryPhysicalLayerSpec {
  assertValidPhysicalLayerIdx(physicalLayerIdx);
  return DORY_PHYSICAL_LAYERS[physicalLayerIdx];
}

export function getDoryStage(physicalLayerIdx: number): DoryLayerStage {
  return getDoryPhysicalLayerSpec(physicalLayerIdx).stage;
}

export function getDorySublayerKind(physicalLayerIdx: number): DorySublayerKind {
  return getDoryPhysicalLayerSpec(physicalLayerIdx).sublayerKind;
}

export function isDoryAttentionLayer(physicalLayerIdx: number): boolean {
  assertValidPhysicalLayerIdx(physicalLayerIdx);
  return physicalLayerIdx % 2 === 0;
}

export function isDorySlidingAttentionLayer(physicalLayerIdx: number): boolean {
  return getDorySublayerKind(physicalLayerIdx) === 'sliding_attention';
}

export function isDoryFullAttentionLayer(physicalLayerIdx: number): boolean {
  return getDorySublayerKind(physicalLayerIdx) === 'full_attention';
}

export function isDoryMlpLayer(physicalLayerIdx: number): boolean {
  return getDorySublayerKind(physicalLayerIdx) === 'mlp';
}

/**
 * Validates recurrent loop budget `kLoops >= 1` (`DR11`: matches `vLLM`/`transformers`
 * accepting any positive integer `k >= 1`, including trained `1..4` and extrapolation `6, 8`).
 */
export function validateDoryRecurrentLoops(kLoops: number): number {
  if (!Number.isInteger(kLoops) || kLoops < 1) {
    throw new RangeError(
      `Invalid Dory recurrent loop count ${String(kLoops)}: expected integer >= 1.`,
    );
  }
  return kLoops;
}

export function computeExecutedDepth(
  kLoops: number,
  config: DoryMiniConfig = DORY_MINI_CONFIG,
): number {
  validateDoryRecurrentLoops(kLoops);
  return config.nInputLayers + kLoops * config.nRecurrentLayers + config.nOutputLayers;
}

export function computeEffectivePairedDepth(
  kLoops: number,
  config: DoryMiniConfig = DORY_MINI_CONFIG,
): number {
  return computeExecutedDepth(kLoops, config) / 2;
}

/**
 * Maps `(physicalLayerIdx, loopIteration, kLoops)` to executed layer index `e in 0..(31 + 40*k)`:
 *   - Prelude (`0..15`): requires `loopIteration === 0` -> `e = p`.
 *   - Recurrent (`16..55`): requires `loopIteration in 1..kLoops` -> `e = 16 + (t - 1) * 40 + (p - 16)`.
 *   - Coda (`56..71`): requires `loopIteration === 0` -> `e = 16 + 40 * kLoops + (p - 56)`.
 */
export function computeExecutedLayerIdx(
  physicalLayerIdx: number,
  loopIteration: number,
  kLoops: number,
): number {
  assertValidPhysicalLayerIdx(physicalLayerIdx);
  validateDoryRecurrentLoops(kLoops);

  const nInput = DORY_MINI_CONFIG.nInputLayers;
  const nRecurrent = DORY_MINI_CONFIG.nRecurrentLayers;
  const recurrentEnd = nInput + nRecurrent;

  if (physicalLayerIdx < nInput) {
    if (loopIteration !== 0) {
      throw new RangeError(
        `Input Prelude layer ${physicalLayerIdx} requires loopIteration === 0, got ${String(loopIteration)}.`,
      );
    }
    return physicalLayerIdx;
  }

  if (physicalLayerIdx < recurrentEnd) {
    if (!Number.isInteger(loopIteration) || loopIteration < 1 || loopIteration > kLoops) {
      throw new RangeError(
        `Recurrent layer ${physicalLayerIdx} requires loopIteration in [1, ${kLoops}], got ${String(loopIteration)}.`,
      );
    }
    return nInput + (loopIteration - 1) * nRecurrent + (physicalLayerIdx - nInput);
  }

  if (loopIteration !== 0) {
    throw new RangeError(
      `Output Coda layer ${physicalLayerIdx} requires loopIteration === 0, got ${String(loopIteration)}.`,
    );
  }
  return nInput + kLoops * nRecurrent + (physicalLayerIdx - recurrentEnd);
}

export function computeKvBankCounts(
  kLoops: number,
  kvCacheMode: DoryKvCacheMode,
  includeShallowCoda = false,
): DoryKvBankCountSummary {
  validateDoryRecurrentLoops(kLoops);

  const recurrentMult = kvCacheMode === 'per_loop' ? kLoops : 1;
  const inputSwaBanks = 6 as const;
  const inputFullBanks = 2 as const;
  const recurrentSwaBanks = 15 * recurrentMult;
  const recurrentFullBanks = 5 * recurrentMult;
  const outputDeepSwaBanks = 6 as const;
  const outputDeepFullBanks = 2 as const;
  const outputShallowSwaBanks: 0 | 6 = includeShallowCoda ? 6 : 0;
  const outputShallowFullBanks: 0 | 2 = includeShallowCoda ? 2 : 0;

  const totalSwaBanks =
    inputSwaBanks + recurrentSwaBanks + outputDeepSwaBanks + outputShallowSwaBanks;
  const totalFullBanks =
    inputFullBanks + recurrentFullBanks + outputDeepFullBanks + outputShallowFullBanks;
  const totalBanks = totalSwaBanks + totalFullBanks;

  return Object.freeze({
    kLoops,
    kvCacheMode,
    includeShallowCoda,
    inputSwaBanks,
    inputFullBanks,
    recurrentSwaBanks,
    recurrentFullBanks,
    outputDeepSwaBanks,
    outputDeepFullBanks,
    outputShallowSwaBanks,
    outputShallowFullBanks,
    totalSwaBanks,
    totalFullBanks,
    totalBanks,
  });
}

export function computeMiniKvCacheBytes(
  kLoops: number,
  kvCacheMode: DoryKvCacheMode,
  includeShallowCoda = false,
  config: DoryMiniConfig = DORY_MINI_CONFIG,
): DoryMiniKvMemorySummary {
  const banks = computeKvBankCounts(kLoops, kvCacheMode, includeShallowCoda);
  // K + V in Float32Array (4 bytes per float)
  const bytesPerTokenPerBank = 2 * config.kvProjOutDim * 4;
  const swaBankPayloadBytes = config.swaRingCapacity * bytesPerTokenPerBank;
  const fullBankPayloadBytes = config.maxPositionEmbeddings * bytesPerTokenPerBank;
  const kvPayloadBytes =
    banks.totalSwaBanks * swaBankPayloadBytes + banks.totalFullBanks * fullBankPayloadBytes;

  const totalAllocatedSlots =
    banks.totalSwaBanks * config.swaRingCapacity +
    banks.totalFullBanks * config.maxPositionEmbeddings;
  const absPosTagBytes = totalAllocatedSlots * 4;
  const writerLoopTagBytes = totalAllocatedSlots * 4;
  const totalAllocatedBytes = kvPayloadBytes + absPosTagBytes + writerLoopTagBytes;

  return Object.freeze({
    banks,
    bytesPerTokenPerBank,
    swaBankPayloadBytes,
    fullBankPayloadBytes,
    kvPayloadBytes,
    absPosTagBytes,
    writerLoopTagBytes,
    totalAllocatedBytes,
  });
}

/**
 * Analytical production BF16 logical KV memory at context length `seqLen` (`B = 1`,
 * visible steady-state `min(S, W)` SWA slots, `4,096 B` per token per attention bank).
 * Preserves exact `S`-invariant `per_loop / last_loop = (4 + 5k) / 9` ratio (Audit §7.2).
 */
export function computeProductionKvBytesBf16(
  seqLen: number,
  kLoops: number,
  kvCacheMode: DoryKvCacheMode,
): DoryProductionKvPoint {
  if (!Number.isInteger(seqLen) || seqLen < 0) {
    throw new RangeError(
      `Invalid sequence length ${String(seqLen)}: expected integer >= 0.`,
    );
  }
  validateDoryRecurrentLoops(kLoops);

  const banks = computeKvBankCounts(kLoops, kvCacheMode, false);
  const swaVisibleSlots = Math.min(seqLen, DORY_PRODUCTION_SPEC.slidingWindow);
  const bytesPerTok = DORY_PRODUCTION_SPEC.kvBytesPerTokenPerLayerBf16;

  const swaBytes = banks.totalSwaBanks * swaVisibleSlots * bytesPerTok;
  const fullBytes = banks.totalFullBanks * seqLen * bytesPerTok;
  const totalBytes = swaBytes + fullBytes;
  const swaShare = totalBytes > 0 ? swaBytes / totalBytes : 0;
  const perLoopToLastLoopRatio = (4 + 5 * kLoops) / 9;

  return Object.freeze({
    seqLen,
    kLoops,
    kvCacheMode,
    swaBanks: banks.totalSwaBanks,
    fullBanks: banks.totalFullBanks,
    swaBytes,
    fullBytes,
    totalBytes,
    swaShare,
    perLoopToLastLoopRatio,
  });
}

export function decodeDoryToken(tokenId: number): string {
  if (Number.isInteger(tokenId) && tokenId >= 0 && tokenId < DORY_MINI_VOCAB.length) {
    return DORY_MINI_VOCAB[tokenId];
  }
  return DORY_MINI_VOCAB[0];
}

export function decodeDoryTokens(tokenIds: ArrayLike<number>): string[] {
  const out: string[] = new Array<string>(tokenIds.length);
  for (let i = 0; i < tokenIds.length; i++) {
    out[i] = decodeDoryToken(tokenIds[i]);
  }
  return out;
}
