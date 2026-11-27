/**
 * Shared Headless TypeScript Contracts for NVIDIA Nemotron-Labs-Dory1.0-4B (`src/dory/types.ts`)
 *
 * Frozen Step-0 contracts for Milestones `DORY1` & `DORY2` (with forward-compatible hooks for
 * `DORY3`–`DORY5`) enabling strictly disjoint parallel implementation across:
 *   - DORY1 Slice 1A: `spec.ts`, `ledger.ts`, `prng.ts`, `parameters.ts`, `ngptMath.ts`, `rope.ts`
 *   - DORY1 Slice 1B: `kvCache.ts`, `attention.ts`, `swiglu.ts`, `forward.ts`, `index.ts`
 *   - DORY1 Slice 1C: `scripts/reference/{export_dory_params.mjs,generate_dory_pytorch_fixtures.py}`,
 *                     `tests/fixtures/pytorch-dory-backbone.json`, `tests/dory-math-parity.test.ts`
 *   - DORY2 Slice 2A: `speculative.ts` (Contract A §6.2 self-speculation, 104 banks, division-free binary64)
 *   - DORY2 Slice 2B: `session.ts` (immutable non-aliased CoW branches, DR18 guard, D10 3.00 MiB LRU pool, DR24 replay)
 *   - DORY2 Slice 2C: `scripts/reference/generate_dory_spec_fixtures.py`, `tests/fixtures/pytorch-dory-spec.json`,
 *                     `tests/dory-speculative-parity.test.ts`, `tests/dory-session-stress.test.ts`
 *
 * Locked Design Decisions (`D1–D10`) & Discrepancy Guards (`DR1–DR24`):
 *   - `D1` Physical `0..71` single-sublayer indexing (`16` Input + `40` Recurrent + `16` Output),
 *     paired `transformers` badge `Pair L{p >> 1}` (`0..35`).
 *   - `D2` Miniature geometry: `d=8`, `d_ff=24`, `h_q=8`, `h_kv=2` (`4:1` GQA), `d_h=8`
 *     (`q_dim=64`, `kv_dim=16`), `V=64` (untied), `S_max=32`, `W=8`, `cap_swa=16` (`+ absPosTag`),
 *     dual RoPE `(θ_S=16, θ_F=1024)` 100% NeoX `rotate_half`, `507` tensors / `93,248` floats.
 *   - `D3` Contract A self-speculation (`output_shallow` + `output_deep` Coda KV, `104` banks).
 *   - `D4` Architectural Boundary Isolation: ZERO imports from Gate 4b/4c Nano modules or sibling
 *     apps. Zero runtime npm dependencies and strictly seeded deterministic PRNG (zero ambient RNG).
 *   - `D5` Canonical Browser Numerical Contract: `Float32Array` (binary32) storage with JS native
 *     binary64 (`Number`) accumulation & transcendentals, rounded strictly at named tensor
 *     write-back boundaries (no per-op `Math.fround()`), fixed ascending reduction order,
 *     matching `vLLM`/`llama.cpp`/TeX FP32 semantics (`RoPE -> Q/K L2Norm`, literal
 *     `u_t = N(s_{t-1} + x)` incl. `u_1 = N(2x)` and `s_t = N(F_θ(u_t))`, no Layer-0 pre-norm,
 *     unit-norm synthetic embeddings).
 *   - `D10` `3.00 MiB` (`3,145,728 B`) byte-budgeted LRU sparse-checkpoint pool (`k in {1..4, 6, 8}`).
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL, Node built-ins, or sibling imports.
 */

// ============================================================================
// MINIATURE VOCABULARY & PRODUCTION-ALIGNED SPECIAL TOKENS (V = 64, UNTIED)
// ============================================================================

/**
 * Miniature 64-token vocabulary (`V = 64`, untied embeddings & LM head):
 *   - Preserves all 11 production special token IDs (`BF16/tokenizer_config.json`):
 *     `0: <unk>`, `1: <s>` (BOS), `2: </s>` (EOS #1), `10: <|im_start|>`,
 *     `11: <|im_end|>` (EOS #2 / PAD), `12: <think>`, `13: </think>`,
 *     `14: <tool_call>`, `15: </tool_call>`, `16: <tool_response>`, `17: </tool_response>`.
 *   - 53 compact reasoning, architecture, code, numeric, and punctuation tokens (`3..9`, `18..63`).
 */
export const DORY_MINI_VOCAB: readonly string[] = Object.freeze([
  '<unk>',           // 0  (UNK)
  '<s>',             // 1  (BOS)
  '</s>',            // 2  (EOS #1)
  'system',          // 3
  'user',            // 4
  'assistant',       // 5
  'the',             // 6
  'of',              // 7
  'to',              // 8
  'in',              // 9
  '<|im_start|>',    // 10 (chat turn start)
  '<|im_end|>',      // 11 (EOS #2 / PAD)
  '<think>',         // 12 (reasoning open)
  '</think>',        // 13 (reasoning close)
  '<tool_call>',     // 14 (tool call open)
  '</tool_call>',    // 15 (tool call close)
  '<tool_response>', // 16 (tool response open)
  '</tool_response>',// 17 (tool response close)
  '\n',              // 18
  'a',               // 19
  'is',              // 20
  'and',             // 21
  'for',             // 22
  'with',            // 23
  'step',            // 24
  'loop',            // 25
  'depth',           // 26
  'state',           // 27
  'norm',            // 28
  'sphere',          // 29
  'anchor',          // 30
  'cache',           // 31
  'window',          // 32
  'full',            // 33
  'gate',            // 34
  'scale',           // 35
  'draft',           // 36
  'verify',          // 37
  'accept',          // 38
  'reject',          // 39
  'solve',           // 40
  'check',           // 41
  'proof',           // 42
  'answer',          // 43
  'sum',             // 44
  'ratio',           // 45
  'true',            // 46
  'false',           // 47
  'fn',              // 48
  'let',             // 49
  'return',          // 50
  'query',           // 51
  'result',          // 52
  'error',           // 53
  '0',               // 54
  '1',               // 55
  '2',               // 56
  '4',               // 57
  '8',               // 58
  '=',               // 59
  '+',               // 60
  '-',               // 61
  ':',               // 62
  '.',               // 63
]);

export const DORY_UNK_TOKEN_ID = 0 as const;
export const DORY_BOS_TOKEN_ID = 1 as const;
export const DORY_EOS_TOKEN_ID = 2 as const;
export const DORY_IM_START_TOKEN_ID = 10 as const;
export const DORY_IM_END_TOKEN_ID = 11 as const;
export const DORY_PAD_TOKEN_ID = 11 as const;
export const DORY_THINK_TOKEN_ID = 12 as const;
export const DORY_END_THINK_TOKEN_ID = 13 as const;
export const DORY_TOOL_CALL_TOKEN_ID = 14 as const;
export const DORY_END_TOOL_CALL_TOKEN_ID = 15 as const;
export const DORY_TOOL_RESPONSE_TOKEN_ID = 16 as const;
export const DORY_END_TOOL_RESPONSE_TOKEN_ID = 17 as const;
export const DORY_EOS_TOKEN_IDS: readonly [2, 11] = Object.freeze([2, 11] as const);

// ============================================================================
// LOCKED SEEDS, TENSOR COUNTS, MEMORY BUDGETS & NUMERICAL CONTRACT (D2, D5, D10)
// ============================================================================

/** Deterministic default PRNG seed shared across TS and Python/PyTorch 2.15.0 parity. */
export const DORY_BACKBONE_SEED = 0xd0411004 as const;

/** Exact canonical tensor and float accounting (1:1 tensor count with production BF16!). */
export const DORY_MINI_TENSOR_COUNT = 507 as const;
export const DORY_MINI_LINEAR_WEIGHT_TENSOR_COUNT = 288 as const; // 180 Attn + 108 MLP
export const DORY_MINI_NGPT_SCALE_TENSOR_COUNT = 217 as const;    // 108 Attn + 108 MLP + 1 LM head
export const DORY_MINI_BOOKEND_MATRIX_TENSOR_COUNT = 2 as const;  // embeddings + lm_head
export const DORY_MINI_NORM_WEIGHT_TENSOR_COUNT = 0 as const;     // weightless L2Norm
export const DORY_MINI_BIAS_TENSOR_COUNT = 0 as const;            // bias-free projections
export const DORY_MINI_LOOP_ADAPTER_TENSOR_COUNT = 0 as const;    // zero loop adapters

/**
 * Exact miniature Float32 parameter count:
 *   - Bookends (3 tensors): embeddings [64,8] (512) + lm_head [64,8] (512) + logit_scale [64] (64) = 1,088
 *   - 36 Attn sublayers (288 tensors): 36 * (8 + 512 + 128 + 128 + 512 + 512 + 64 + 64) = 36 * 1,928 = 69,408
 *   - 36 MLP sublayers (216 tensors):  36 * (8 + 192 + 192 + 192 + 24 + 24)             = 36 * 632   = 22,752
 *   - Total: 507 tensors = 93,248 F32 floats = 372,992 B (364.25 KiB = 0.356 MiB).
 */
export const DORY_MINI_FLOAT_COUNT = 93_248 as const;
export const DORY_MINI_WEIGHT_BYTES = 372_992 as const;

/** Analytical session memory caps (D10). */
export const DORY_MAX_SESSION_RETAINED_BYTES = 12_582_912 as const;  // 12.00 MiB
export const DORY_CHECKPOINT_LRU_POOL_BYTES = 3_145_728 as const;    // 3.00 MiB (D10)
export const DORY_MAX_ATLAS_BYTES = 524_288 as const;                // 512.00 KiB RG32F

/** Exact dyadic SWA RoPE inverse frequencies for `theta_S = 16`, `headDim = 8`. */
export const DORY_MINI_ROPE_SLIDING_INV_FREQ: readonly [number, number, number, number] =
  Object.freeze([1.0, 0.5, 0.25, 0.125] as const);

/**
 * Planted unclamped NLERP extrapolation channels (`DR7`, Audit §11.3):
 * In physical layer `16` (first Recurrent Core SWA sublayer), channels `6` and `7` of `attn_alpha`
 * are explicitly initialized so effective `|alpha_hat| in (1.05, 1.25)` (`1.12` and `1.22`),
 * directly exercising the unclamped `> 1` extrapolation contract without harming stability.
 */
export const DORY_PLANTED_EXTRAPOLATION_SPEC = Object.freeze({
  physicalLayerIdx: 16 as const,
  channelIndices: Object.freeze([6, 7] as const),
  effectiveTargets: Object.freeze([1.12, 1.22] as const),
  minEffectiveAlpha: 1.05 as const,
  maxEffectiveAlpha: 1.25 as const,
});

/**
 * Canonical binary framing magic header for `computeDoryWeightDigestHex()` (SHA-256).
 * Both `src/dory/parameters.ts` (Slice 1A) and `scripts/reference/generate_dory_pytorch_fixtures.py`
 * (Slice 1C) hash this exact byte stream in canonical `listDoryCanonicalTensors()` order:
 *   1. UTF-8 bytes of `DORY_WEIGHT_DIGEST_MAGIC`
 *   2. For each of the 507 tensors in canonical order:
 *      - UTF-8 bytes of `tensor.name` followed by `0x00`
 *      - `uint32LE(tensor.shape.length)`
 *      - `uint32LE(dim)` for each dimension in `tensor.shape`
 *      - Raw little-endian IEEE-754 binary32 bytes of `tensor.data` (`data.length * 4` bytes)
 */
export const DORY_WEIGHT_DIGEST_MAGIC = 'DORY-MINI-F32-V1\n' as const;

/** Unified D5 numerical parity tolerance table (TS vs. PyTorch 2.15.0 FP32). */
export const DORY_NUMERICAL_TOLERANCES = Object.freeze({
  singleOpMaxAbs: 1e-6 as const,
  algebraicIdentityMaxAbs: 1e-6 as const,
  sublayerMaxAbs: 1e-5 as const,
  fullForwardMaxAbs: 2e-5 as const,
  lastLoopDivergenceMinAbs: 1e-2 as const,
  binary64SpecEnumMaxAbs: 1e-15 as const,
});

/** Canonical anti-SLERP terminology marker (also listed in `FORBIDDEN_DORY_MARKERS`). */
export const DORY_NLERP_MARKER = 'dory-nlerp-residual' as const;

// ============================================================================
// CORE DOMAIN ENUMS & PHYSICAL-72 LAYER INDEXING (D1)
// ============================================================================

export type DoryLayerStage = 'input' | 'recurrent' | 'output';
export type DorySublayerKind = 'sliding_attention' | 'full_attention' | 'mlp';
export type DoryAttentionKind = 'sliding_attention' | 'full_attention';
export type DoryRopeRegime = 'sliding' | 'full';
export type DoryKvCacheMode = 'per_loop' | 'last_loop';
export type DoryKvBankKind = 'input' | 'recurrent' | 'output_deep' | 'output_shallow';
export type DoryCodaBankKind = 'output_deep' | 'output_shallow';
export type DoryCacheRole =
  | 'standard'
  | 'spec_draft'
  | 'spec_verify_reused'
  | 'spec_verify_deep';

export type DoryTrainedLoopCount = 1 | 2 | 3 | 4;
export type DoryExtrapolationLoopCount = 6 | 8;
export type DorySupportedLoopCount = DoryTrainedLoopCount | DoryExtrapolationLoopCount;

export type DoryScaleFamilyId =
  | 'residual_alpha'
  | 'sqk'
  | 's_gate'
  | 'suv'
  | 'logit_scale';

export type DoryTensorRole =
  | 'embeddings'
  | 'attn_alpha'
  | 'q_proj'
  | 'k_proj'
  | 'v_proj'
  | 'g_proj'
  | 'o_proj'
  | 'sqk'
  | 's_gate'
  | 'mlp_alpha'
  | 'gate_proj'
  | 'up_proj'
  | 'down_proj'
  | 'suv_gate'
  | 'suv_up'
  | 'lm_head'
  | 'logit_scale';

/**
 * Repeating 8-sublayer motif (`×9` across physical indices `0..71`):
 *   `[SWA, MLP, SWA, MLP, SWA, MLP, Full, MLP]`
 * Exact 3:1 SWA:Full ratio preserved inside Input (`6:2`), Recurrent (`15:5`), and Output (`6:2`).
 */
export const DORY_MOTIF_PATTERN: readonly [
  'sliding_attention',
  'mlp',
  'sliding_attention',
  'mlp',
  'sliding_attention',
  'mlp',
  'full_attention',
  'mlp',
] = Object.freeze([
  'sliding_attention',
  'mlp',
  'sliding_attention',
  'mlp',
  'sliding_attention',
  'mlp',
  'full_attention',
  'mlp',
] as const);

// ============================================================================
// MINIATURE CONFIGURATION (`DORY_MINI_CONFIG`, LOCKED D2)
// ============================================================================

export interface DoryMiniConfig {
  readonly modelType: 'dory';
  readonly architectureName: 'DoryForCausalLM';
  /** Physical single-sublayer count (`72 = 16 + 40 + 16`). */
  readonly numHiddenLayers: 72;
  /** Paired `(Attn, MLP)` decoder-layer correspondence (`36 = 8 + 20 + 8`). */
  readonly numPairedLayers: 36;
  readonly nInputLayers: 16;
  readonly nRecurrentLayers: 40;
  readonly nOutputLayers: 16;
  readonly motifPeriod: 8;
  readonly hiddenSize: 8;
  readonly intermediateSize: 24;
  readonly numAttentionHeads: 8;
  readonly numKeyValueHeads: 2;
  readonly kvGroupSize: 4;
  readonly headDim: 8;
  readonly rotaryPlanes: 4;
  /** Attention query/gate expansion width `h_q * d_h = 64` (`8.0× d`, prod `1.60× d`). */
  readonly qProjOutDim: 64;
  /** Attention key/value width `h_kv * d_h = 16` (`2.0× d`, prod `0.40× d`). */
  readonly kvProjOutDim: 16;
  readonly vocabSize: 64;
  readonly maxPositionEmbeddings: 32;
  readonly slidingWindow: 8;
  /** Power-of-2 SWA ring capacity (`16 >= W + gamma + 1 = 13`), bitmask `pos & 15` (`DR19`). */
  readonly swaRingCapacity: 16;
  /** Max transient SWA chunk workspace length `W - 1 + S_max = 39` (`DR24`). */
  readonly maxTransientSwaWorkspaceLen: 39;
  readonly ropeType: 'default';
  readonly partialRotaryFactor: 1.0;
  readonly ropeThetaSliding: 16;
  readonly ropeThetaFull: 1024;
  readonly hiddenAct: 'silu';
  readonly attentionBias: false;
  readonly mlpBias: false;
  readonly tieWordEmbeddings: false;
  /** Active runtime inference divisor `sigma_0 = 0.0177` (unmodified production constant!). */
  readonly initializerRange: 0.0177;
  /** Target initial effective residual rate `alpha_init = 0.05`. */
  readonly ngptAlphaInit: 0.05;
  /** Output-gate init target `g_0`; `null` resolves to `sqrt(hiddenSize)` (`DR3`). */
  readonly attentionOutputGateLogitScaleInit: number | null;
  /** Weightless unit-sphere L2Norm epsilon `1e-6` across all 7 normalization sites. */
  readonly l2NormEps: 1e-6;
  readonly defaultRecurrentLoops: 2;
  readonly trainedRecurrentLoops: readonly [1, 2, 3, 4];
  readonly extrapolationRecurrentLoops: readonly [6, 8];
  readonly supportedRecurrentLoops: readonly [1, 2, 3, 4, 6, 8];
  readonly defaultKvCacheMode: DoryKvCacheMode;
  readonly speculativeDraftLoops: 1;
  readonly speculativeVerifyLoops: 4;
  readonly speculativeGamma: 4;
  readonly bosTokenId: 1;
  readonly eosTokenIds: readonly [2, 11];
  readonly padTokenId: 11;
  readonly unkTokenId: 0;
  readonly backboneSeed: number;
  readonly maxSessionRetainedBytes: 12_582_912;
  readonly checkpointLruPoolBytes: 3_145_728;
}

export const DORY_MINI_CONFIG: DoryMiniConfig = Object.freeze({
  modelType: 'dory',
  architectureName: 'DoryForCausalLM',
  numHiddenLayers: 72,
  numPairedLayers: 36,
  nInputLayers: 16,
  nRecurrentLayers: 40,
  nOutputLayers: 16,
  motifPeriod: 8,
  hiddenSize: 8,
  intermediateSize: 24,
  numAttentionHeads: 8,
  numKeyValueHeads: 2,
  kvGroupSize: 4,
  headDim: 8,
  rotaryPlanes: 4,
  qProjOutDim: 64,
  kvProjOutDim: 16,
  vocabSize: 64,
  maxPositionEmbeddings: 32,
  slidingWindow: 8,
  swaRingCapacity: 16,
  maxTransientSwaWorkspaceLen: 39,
  ropeType: 'default',
  partialRotaryFactor: 1.0,
  ropeThetaSliding: 16,
  ropeThetaFull: 1024,
  hiddenAct: 'silu',
  attentionBias: false,
  mlpBias: false,
  tieWordEmbeddings: false,
  initializerRange: 0.0177,
  ngptAlphaInit: 0.05,
  attentionOutputGateLogitScaleInit: null,
  l2NormEps: 1e-6,
  defaultRecurrentLoops: 2,
  trainedRecurrentLoops: Object.freeze([1, 2, 3, 4] as const),
  extrapolationRecurrentLoops: Object.freeze([6, 8] as const),
  supportedRecurrentLoops: Object.freeze([1, 2, 3, 4, 6, 8] as const),
  defaultKvCacheMode: 'per_loop',
  speculativeDraftLoops: 1,
  speculativeVerifyLoops: 4,
  speculativeGamma: 4,
  bosTokenId: DORY_BOS_TOKEN_ID,
  eosTokenIds: DORY_EOS_TOKEN_IDS,
  padTokenId: DORY_PAD_TOKEN_ID,
  unkTokenId: DORY_UNK_TOKEN_ID,
  backboneSeed: DORY_BACKBONE_SEED,
  maxSessionRetainedBytes: DORY_MAX_SESSION_RETAINED_BYTES,
  checkpointLruPoolBytes: DORY_CHECKPOINT_LRU_POOL_BYTES,
});

// ============================================================================
// CANONICAL CHECKPOINT & PAIRED TRANSFORMERS TENSOR NAMING HELPERS (D1, DR4)
// ============================================================================

export const DORY_EMBEDDINGS_TENSOR_NAME = 'backbone.embeddings.weight' as const;
export const DORY_LM_HEAD_TENSOR_NAME = 'lm_head.weight' as const;
export const DORY_LM_HEAD_LOGIT_SCALE_TENSOR_NAME = 'lm_head.logit_scale' as const;

export function doryAttnAlphaTensorName(physicalLayerIdx: number): string {
  return `backbone.layers.${physicalLayerIdx}.attn_alpha`;
}

export function doryAttnProjTensorName(
  physicalLayerIdx: number,
  proj: 'q_proj' | 'k_proj' | 'v_proj' | 'g_proj' | 'o_proj',
): string {
  return `backbone.layers.${physicalLayerIdx}.mixer.${proj}.weight`;
}

export function doryAttnScaleTensorName(
  physicalLayerIdx: number,
  scale: 'sqk' | 's_gate',
): string {
  return `backbone.layers.${physicalLayerIdx}.mixer.${scale}`;
}

export function doryMlpAlphaTensorName(physicalLayerIdx: number): string {
  return `backbone.layers.${physicalLayerIdx}.mlp_alpha`;
}

export function doryMlpProjTensorName(
  physicalLayerIdx: number,
  proj: 'gate_proj' | 'up_proj' | 'down_proj',
): string {
  return `backbone.layers.${physicalLayerIdx}.mixer.${proj}.weight`;
}

export function doryMlpScaleTensorName(
  physicalLayerIdx: number,
  scale: 'suv_gate' | 'suv_up',
): string {
  return `backbone.layers.${physicalLayerIdx}.mixer.${scale}`;
}

/**
 * Maps canonical physical-72 checkpoint tensor names (`backbone.layers.{0..71}...`)
 * to paired-36 Hugging Face `transformers` (`LayerPairCollapse`) correspondence names (`DR4`).
 */
export function doryToTransformersPairedTensorName(canonicalName: string): string {
  if (canonicalName === DORY_EMBEDDINGS_TENSOR_NAME) {
    return 'model.embed_tokens.weight';
  }
  if (
    canonicalName === DORY_LM_HEAD_TENSOR_NAME ||
    canonicalName === DORY_LM_HEAD_LOGIT_SCALE_TENSOR_NAME
  ) {
    return canonicalName;
  }
  const match = /^backbone\.layers\.(\d+)\.(.+)$/.exec(canonicalName);
  if (!match) {
    return canonicalName;
  }
  const p = Number(match[1]);
  const suffix = match[2];
  const pairIdx = p >> 1;
  if (p % 2 === 0) {
    if (suffix === 'attn_alpha') {
      return `model.layers.${pairIdx}.lambda_1`;
    }
    if (suffix === 'mixer.g_proj.weight') {
      return `model.layers.${pairIdx}.self_attn.gate_proj.weight`;
    }
    if (suffix.startsWith('mixer.')) {
      return `model.layers.${pairIdx}.self_attn.${suffix.slice('mixer.'.length)}`;
    }
  } else {
    if (suffix === 'mlp_alpha') {
      return `model.layers.${pairIdx}.lambda_2`;
    }
    if (suffix.startsWith('mixer.')) {
      return `model.layers.${pairIdx}.mlp.${suffix.slice('mixer.'.length)}`;
    }
  }
  return canonicalName;
}

// ============================================================================
// IMMUTABLE 8-TUPLE EXECUTION COORDINATE SYSTEM (AUDIT §6.3)
// ============================================================================

export interface DoryCacheOwnership {
  readonly policy: DoryKvCacheMode;
  readonly bankKind: DoryKvBankKind;
  /** Logical KV bank ID (`0..95` std `per_loop`; `0..103` Contract A spec; `0..35` `last_loop`). */
  readonly bankSlotId: number;
  /** Which loop authored the historical KV being read (`0` for input/output, `1..k` recurrent). */
  readonly writerLoop: number;
  readonly role: DoryCacheRole;
}

/**
 * Immutable 8-tuple execution coordinate separating shared physical parameter identity
 * from per-loop activation & cache identity (`DORY0_ARCHITECTURE_AUDIT.md` §6.3).
 */
export interface DoryExecutionCoord {
  /** (1) Singleton SHA-256 weight identity (invariant across loops & branches). */
  readonly modelWeightId: string;
  /** (2) Physical layer index `p in 0..71` (even = Attn, odd = MLP; `pairedLayerIdx = p >> 1`). */
  readonly physicalLayerIdx: number;
  /** (3) Executed layer index `e in 0..(31 + 40*k)`. */
  readonly executedLayerIdx: number;
  /** (4) Loop iteration `t = 0` (Prelude/Coda), `t in 1..k` (Recurrent pass). */
  readonly loopIteration: number;
  /** (5) Global sequence position (`0..S_max - 1`). */
  readonly absTokenPos: number;
  /** (6) Monotonic forward-call / chunk ID within branch. */
  readonly forwardStepId: number;
  /** (7) Immutable session + branch ID (e.g. `"b0"`, `"b3"`). */
  readonly sessionBranchId: string;
  /** (8) Explicit KV policy & ownership provenance. */
  readonly cacheOwnership: DoryCacheOwnership;
}

// ============================================================================
// SPEC, PRODUCTION METADATA & LAYER TABLE INTERFACES (Slice 1A: `spec.ts`)
// ============================================================================

export interface DoryPhysicalLayerSpec {
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly stage: DoryLayerStage;
  readonly motifSlot: number;
  readonly sublayerKind: DorySublayerKind;
  readonly attentionKind: DoryAttentionKind | null;
  readonly ropeRegime: DoryRopeRegime | null;
  readonly ropeThetaMini: number | null;
  readonly ropeThetaProd: number | null;
  /** `0..7` (input), `0..19` (recurrent), `0..7` (output), or `null` for MLP. */
  readonly stageAttentionOrdinal: number | null;
  /** `0..35` across all attention sublayers, or `null` for MLP. */
  readonly globalAttentionOrdinal: number | null;
  /** Human-readable paired badge e.g. `"Pair L0 · SWA"` or `"Pair L8 · MLP"`. */
  readonly transformersPairBadge: string;
}

export interface DoryCheckpointFormatSpec {
  readonly id: 'bf16' | 'mxfp8' | 'nvfp4' | 'q5_k_m';
  readonly label: string;
  readonly repoId: string;
  readonly pinnedCommitSha: string;
  readonly quantScheme: string;
  readonly tensorCount: number | null;
  readonly unquantizedBf16TensorCount: number | null;
  readonly quantizedLinearMatrixCount: number;
  readonly logicalParams: number;
  readonly storedElements: number | null;
  readonly headerJsonBytes: number | null;
  readonly payloadBytes: number | null;
  readonly totalFileBytes: number;
  readonly diffBytes: number | null;
  readonly lfsSha256Prefix: string;
  readonly license: 'OpenMDW-1.1';
  readonly scaleStorageNote: string;
}

export interface DoryScalingRatioEntry {
  readonly id: string;
  readonly label: string;
  readonly productionValue: string;
  readonly miniatureValue: string;
  readonly productionRatio: string;
  readonly miniatureRatio: string;
  readonly disclosureNote: string;
}

export interface DoryProductionSpec {
  readonly modelName: 'NVIDIA-Nemotron-Labs-Dory1.0-4B';
  readonly architectureName: 'DoryForCausalLM';
  readonly modelType: 'dory';
  readonly totalParams: 3_975_229_440;
  readonly tensorCountBf16: 507;
  readonly numHiddenLayers: 72;
  readonly numPairedLayers: 36;
  readonly nInputLayers: 16;
  readonly nRecurrentLayers: 40;
  readonly nOutputLayers: 16;
  readonly hiddenSize: 2560;
  readonly intermediateSize: 7168;
  readonly numAttentionHeads: 16;
  readonly numKeyValueHeads: 4;
  readonly headDim: 256;
  readonly qProjOutDim: 4096;
  readonly kvProjOutDim: 1024;
  readonly vocabSize: 131072;
  readonly slidingWindow: 1024;
  readonly maxPositionEmbeddings: 262144;
  readonly ropeThetaSliding: 10_000;
  readonly ropeThetaFull: 10_000_000;
  readonly initializerRange: 0.0177;
  readonly ngptAlphaInit: 0.05;
  readonly attentionOutputGateLogitScaleInit: null;
  readonly l2NormEps: 1e-6;
  readonly defaultRecurrentLoops: 2;
  readonly defaultKvCacheMode: 'per_loop';
  readonly kvBytesPerTokenPerLayerBf16: 4096;
  readonly weightLicense: 'OpenMDW-1.1';
}

export interface DoryKvBankCountSummary {
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly includeShallowCoda: boolean;
  readonly inputSwaBanks: 6;
  readonly inputFullBanks: 2;
  readonly recurrentSwaBanks: number;
  readonly recurrentFullBanks: number;
  readonly outputDeepSwaBanks: 6;
  readonly outputDeepFullBanks: 2;
  readonly outputShallowSwaBanks: 0 | 6;
  readonly outputShallowFullBanks: 0 | 2;
  readonly totalSwaBanks: number;
  readonly totalFullBanks: number;
  readonly totalBanks: number;
}

export interface DoryMiniKvMemorySummary {
  readonly banks: DoryKvBankCountSummary;
  readonly bytesPerTokenPerBank: number;
  readonly swaBankPayloadBytes: number;
  readonly fullBankPayloadBytes: number;
  readonly kvPayloadBytes: number;
  readonly absPosTagBytes: number;
  readonly writerLoopTagBytes: number;
  readonly totalAllocatedBytes: number;
}

export interface DoryProductionKvPoint {
  readonly seqLen: number;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly swaBanks: number;
  readonly fullBanks: number;
  readonly swaBytes: number;
  readonly fullBytes: number;
  readonly totalBytes: number;
  readonly swaShare: number;
  /** Exact `S`-invariant ratio `(4 + 5k) / 9` for `per_loop` vs. `last_loop`. */
  readonly perLoopToLastLoopRatio: number;
}

// ============================================================================
// DISCREPANCY & DESIGN-DECISION LEDGER INTERFACES (Slice 1A: `ledger.ts`)
// ============================================================================

export type DoryDiscrepancyId =
  | 'DR1'  | 'DR2'  | 'DR3'  | 'DR4'  | 'DR5'  | 'DR6'
  | 'DR7'  | 'DR8'  | 'DR9'  | 'DR10' | 'DR11' | 'DR12'
  | 'DR13' | 'DR14' | 'DR15' | 'DR16' | 'DR17' | 'DR18'
  | 'DR19' | 'DR20' | 'DR21' | 'DR22' | 'DR23' | 'DR24';

export type DoryDesignDecisionId =
  | 'D1' | 'D2' | 'D3' | 'D4' | 'D5'
  | 'D6' | 'D7' | 'D8' | 'D9' | 'D10';

export type DoryEpistemicTag =
  | 'VERIFIED-CANONICAL'
  | 'DERIVED'
  | 'IMPL-SPECIFIC'
  | 'SECONDARY'
  | 'UNRESOLVED-DISCREPANCY';

export type DoryDiscrepancyCategory =
  | 'spec_config'
  | 'architecture_math'
  | 'runtime_precision'
  | 'provenance_license'
  | 'kv_cache'
  | 'speculative'
  | 'browser_guard';

export interface DoryEvidenceAxes {
  readonly origin: 'paper_tex' | 'hf_checkpoint' | 'upstream_runtime' | 'repo_audit' | 'browser_contract';
  readonly verificationMethod: 'direct_inspection' | 'algebraic_derivation' | 'numerical_experiment';
  readonly scope: 'canonical_model' | 'runtime_specific' | 'browser_miniature';
  readonly resolutionStatus: 'locked_canonical' | 'adjudicated_browser_contract' | 'documented_divergence';
}

export interface DoryDiscrepancyEntry {
  readonly id: DoryDiscrepancyId;
  readonly numericId: number;
  readonly category: DoryDiscrepancyCategory;
  readonly title: string;
  readonly summary: string;
  readonly provenance: string;
  readonly primaryTag: DoryEpistemicTag;
  readonly secondaryTags?: readonly DoryEpistemicTag[];
  readonly evidenceAxes: DoryEvidenceAxes;
  readonly resolution: string;
  readonly isBrowserGuard: boolean;
}

export interface DoryDesignDecisionEntry {
  readonly id: DoryDesignDecisionId;
  readonly numericId: number;
  readonly title: string;
  readonly summary: string;
  readonly status: 'APPROVED' | 'APPROVED_HARDENED' | 'RESOLVED';
  readonly relatedDiscrepancies: readonly DoryDiscrepancyId[];
}

// ============================================================================
// DETERMINISTIC PRNG INTERFACE (Slice 1A: `prng.ts` — SEEDED DETERMINISTIC RNG)
// ============================================================================

export interface DoryPrng {
  /** Unsigned 32-bit integer in `[0, 2^32 - 1]`. */
  nextUint32(): number;
  /** Uniform binary64 sample in `[0, 1)`. */
  nextFloat64(): number;
  /** Strictly open uniform sample in `(0, 1)` safe for `Math.log` / Box-Muller. */
  nextOpenUnitFloat64(): number;
  /** Deterministic standard normal sample `N(0, 1)` via Box-Muller transform. */
  nextGaussian(): number;
  /** Uniform binary64 sample in `[min, max)`. */
  nextRange(min: number, max: number): number;
  /** Current 32-bit internal PRNG state word. */
  getState(): number;
  /** Non-aliasing clone at the exact same internal state. */
  clone(): DoryPrng;
}

// ============================================================================
// 507-TENSOR / 93,248-FLOAT PARAMETER INTERFACES (Slice 1A: `parameters.ts`)
// ============================================================================

export interface DoryAttentionSublayerWeights {
  readonly kind: 'attention';
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly stage: DoryLayerStage;
  readonly attentionKind: DoryAttentionKind;
  readonly ropeRegime: DoryRopeRegime;
  /** `backbone.layers.{p}.attn_alpha`: raw `[d]` (`[8]`), init around `0.0177`. */
  readonly attnAlpha: Float32Array;
  /** `backbone.layers.{p}.mixer.q_proj.weight`: `[q_dim, d]` (`[64, 8]`), row-unit-norm. */
  readonly qProj: Float32Array;
  /** `backbone.layers.{p}.mixer.k_proj.weight`: `[kv_dim, d]` (`[16, 8]`), row-unit-norm. */
  readonly kProj: Float32Array;
  /** `backbone.layers.{p}.mixer.v_proj.weight`: `[kv_dim, d]` (`[16, 8]`), row-unit-norm. */
  readonly vProj: Float32Array;
  /** `backbone.layers.{p}.mixer.g_proj.weight`: `[q_dim, d]` (`[64, 8]`), row-unit-norm. */
  readonly gProj: Float32Array;
  /** `backbone.layers.{p}.mixer.o_proj.weight`: `[d, q_dim]` (`[8, 64]`), row-unit-norm. */
  readonly oProj: Float32Array;
  /** `backbone.layers.{p}.mixer.sqk`: raw `[q_dim]` (`[64]`), init around `0.0177`. */
  readonly sqk: Float32Array;
  /** `backbone.layers.{p}.mixer.s_gate`: raw `[q_dim]` (`[64]`), init around `0.0177`. */
  readonly sGate: Float32Array;
}

export interface DoryMlpSublayerWeights {
  readonly kind: 'mlp';
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly stage: DoryLayerStage;
  /** `backbone.layers.{p}.mlp_alpha`: raw `[d]` (`[8]`), init around `0.0177`. */
  readonly mlpAlpha: Float32Array;
  /** `backbone.layers.{p}.mixer.gate_proj.weight`: `[d_ff, d]` (`[24, 8]`), row-unit-norm. */
  readonly gateProj: Float32Array;
  /** `backbone.layers.{p}.mixer.up_proj.weight`: `[d_ff, d]` (`[24, 8]`), row-unit-norm. */
  readonly upProj: Float32Array;
  /** `backbone.layers.{p}.mixer.down_proj.weight`: `[d, d_ff]` (`[8, 24]`), row-unit-norm. */
  readonly downProj: Float32Array;
  /** `backbone.layers.{p}.mixer.suv_gate`: raw `[d_ff]` (`[24]`), init around `1.0` (signed!). */
  readonly suvGate: Float32Array;
  /** `backbone.layers.{p}.mixer.suv_up`: raw `[d_ff]` (`[24]`), init around `1.0` (signed!). */
  readonly suvUp: Float32Array;
}

export type DorySublayerWeights = DoryAttentionSublayerWeights | DoryMlpSublayerWeights;

export interface DoryCanonicalWeightTensor {
  readonly name: string;
  readonly transformersPairedName: string;
  readonly role: DoryTensorRole;
  readonly physicalLayerIdx: number | null;
  readonly pairedLayerIdx: number | null;
  readonly stage: DoryLayerStage | 'bookend';
  readonly scaleFamily: DoryScaleFamilyId | null;
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

export interface DoryModelParameters {
  /** Singleton SHA-256 hex digest over all 507 canonical tensors (`modelWeightId` in Coord #1). */
  readonly modelWeightId: string;
  readonly backboneSeed: number;
  readonly config: DoryMiniConfig;
  readonly vocab: readonly string[];
  /** `backbone.embeddings.weight`: `[V, d]` (`[64, 8]`), row-unit-norm (`DR5`). */
  readonly embeddings: Float32Array;
  /** All `72` physical sublayers (`0..71`), even = Attention, odd = MLP. */
  readonly layers: readonly DorySublayerWeights[];
  /** `lm_head.weight`: `[V, d]` (`[64, 8]`), untied, row-unit-norm. */
  readonly lmHead: Float32Array;
  /** `lm_head.logit_scale`: raw `[V]` (`[64]`), init around `0.0177`. */
  readonly logitScale: Float32Array;
}

export interface DoryParameterCountSummary {
  readonly totalTensors: number;
  readonly totalFloats: number;
  readonly totalBytes: number;
  readonly linearWeightTensors: number;
  readonly ngptScaleTensors: number;
  readonly bookendMatrixTensors: number;
  readonly normWeightTensors: 0;
  readonly biasTensors: 0;
  readonly loopAdapterTensors: 0;
  readonly inputStageTensors: number;
  readonly inputStageFloats: number;
  readonly recurrentStageTensors: number;
  readonly recurrentStageFloats: number;
  readonly outputStageTensors: number;
  readonly outputStageFloats: number;
  readonly bookendTensors: number;
  readonly bookendFloats: number;
}

export interface DorySerializedTensorEntry {
  readonly name: string;
  readonly shape: readonly number[];
  /** Exact IEEE-754 float32 values represented as JS numbers (lossless round-trip for binary32). */
  readonly data: readonly number[];
}

export interface DorySerializedParametersPayload {
  readonly format: typeof DORY_WEIGHT_DIGEST_MAGIC;
  readonly modelWeightId: string;
  readonly backboneSeed: number;
  readonly config: DoryMiniConfig;
  readonly tensorCount: number;
  readonly floatCount: number;
  readonly tensors: readonly DorySerializedTensorEntry[];
}

// ============================================================================
// FULL-nGPT MATH & DUAL-ROPE INTERFACES (Slice 1A: `ngptMath.ts`, `rope.ts`)
// ============================================================================

export interface DoryScaleFamilyMultipliers {
  /** `alpha_init / sigma_0 = 0.05 / 0.0177`. */
  readonly alphaMultiplier: number;
  /** `sqrt(d_h) / sigma_0 = sqrt(8) / 0.0177` (squared after multiplying raw `sqk`). */
  readonly sqkLinearMultiplier: number;
  /** Resolved `g_0 = attentionOutputGateLogitScaleInit ?? sqrt(d)`. */
  readonly sGateInitValue: number;
  /** `sqrt(g_0) / sigma_0` (squared after multiplying raw `s_gate`). */
  readonly sGateLinearMultiplier: number;
  /** `sqrt(d) = sqrt(8)` (linear, signed — NO `.abs()`). */
  readonly suvMultiplier: number;
  /** `1 / sigma_0 = 1 / 0.0177` (linear, signed). */
  readonly logitScaleMultiplier: number;
}

export interface DoryNlerpTrace {
  readonly seqLen: number;
  readonly hiddenSize: number;
  /** `a = N_d(residual)` (`[seqLen, d]`). */
  readonly aNorm: Float32Array;
  /** `b = N_d(sublayerOut)` (`[seqLen, d]`). */
  readonly bNorm: Float32Array;
  /** `b - a` (`[seqLen, d]`). */
  readonly delta: Float32Array;
  /** Effective unclamped `lambda = |alpha * (alpha_init / sigma_0)|` (`[d]`). */
  readonly effectiveAlpha: Float32Array;
  /** `a + lambda * (b - a)` prior to final unit-sphere re-projection (`[seqLen, d]`). */
  readonly blendedPreNorm: Float32Array;
  /** `N_d(blendedPreNorm)` (`[seqLen, d]`). */
  readonly output: Float32Array;
  /** Number of channels in `effectiveAlpha` strictly exceeding `1.0` (`DR7`). */
  readonly extrapolatedChannelCount: number;
  readonly maxEffectiveAlpha: number;
}

export interface DoryAnchorJacobianMetrics {
  readonly angleRadians: number;
  readonly angleDegrees: number;
  /** `2 * cos(theta / 2) = ||s + x||_2` for unit vectors. */
  readonly sumNormIdeal: number;
  /** Ideal `1 / (2 * cos(theta / 2))` (`Infinity` at antipodal `pi`). */
  readonly idealJacobianNorm: number;
  /** Stabilized `1 / sqrt(4 * cos^2(theta / 2) + eps)` (capped at `1 / sqrt(eps) = 1000`). */
  readonly epsStabilizedJacobianNorm: number;
  /** True iff `theta < 120°` (`2*pi/3`), local boundary-map regime ONLY (not core convergence!). */
  readonly isLocalContractiveRegime: boolean;
}

export interface DoryRopeTable {
  readonly regime: DoryRopeRegime;
  readonly theta: number;
  readonly headDim: number;
  readonly halfDim: number;
  readonly maxPositions: number;
  /** `[halfDim]` inverse frequencies `theta^(-2j / headDim)`. */
  readonly invFreq: Float32Array;
  /** `[maxPositions * halfDim]` row-major `cos(pos * invFreq[j])`. */
  readonly cos: Float32Array;
  /** `[maxPositions * halfDim]` row-major `sin(pos * invFreq[j])`. */
  readonly sin: Float32Array;
}

export interface DoryDualRopeTables {
  readonly sliding: DoryRopeTable;
  readonly full: DoryRopeTable;
}

// ============================================================================
// DUAL-AXIS KV CACHE & LONG-CHUNK WORKSPACE INTERFACES (Slice 1B: `kvCache.ts`)
// ============================================================================

export interface DoryKvBank {
  readonly bankSlotId: number;
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly stage: DoryLayerStage;
  readonly bankKind: DoryKvBankKind;
  readonly attentionKind: DoryAttentionKind;
  /** `0` for input/output; `1..k` for `per_loop` recurrent; `1` for shared `last_loop` recurrent. */
  readonly loopSlot: number;
  /** `16` (`cap_swa`) for SWA, `32` (`S_max`) for Full. */
  readonly capacity: number;
  /** `8` (`W`) for SWA, `null` for Full. */
  readonly windowSize: number | null;
  /** `16` (`h_kv * d_h`). */
  readonly kvDim: number;
  /** `[capacity * kvDim]` post-RoPE, post-L2Norm keys. */
  readonly k: Float32Array;
  /** `[capacity * kvDim]` unnormalized values. */
  readonly v: Float32Array;
  /** `[capacity]` exact absolute position residing in each physical slot (`-1` if empty; `DR19`). */
  readonly absPosTag: Int32Array;
  /** `[capacity]` loop iteration `t` that authored the entry in each slot (`-1` if empty). */
  readonly writerLoopTag: Int32Array;
  /** Number of committed logical sequence positions `[0 .. logicalLen - 1]`. */
  logicalLen: number;
}

/**
 * Transient current-chunk attention workspace (`DR24`):
 * Holds `[past_visible (<= W - 1 for SWA, startPos for Full) || current_chunk (L)]` so even
 * long chunks `L in (16, 32]` execute exact SWA attention with ZERO ring-buffer overwrite,
 * committing only the newest `min(endPos, capacity)` positions into the persistent bank afterward.
 */
export interface DoryTransientKvWorkspace {
  readonly pastLen: number;
  readonly chunkLen: number;
  readonly totalLen: number;
  readonly kvDim: number;
  /** `[totalLen * kvDim]` visible keys (`[past_visible || current_chunk]`). */
  readonly k: Float32Array;
  /** `[totalLen * kvDim]` visible values (`[past_visible || current_chunk]`). */
  readonly v: Float32Array;
  /** `[totalLen]` exact absolute token position for each KV entry. */
  readonly absPositions: Int32Array;
  /** `[totalLen]` exact loop iteration that authored each KV entry (`k` for historical `last_loop`). */
  readonly writerLoops: Int32Array;
}

export interface DoryKvCacheCreateOptions {
  readonly mode: DoryKvCacheMode;
  readonly kLoops: number;
  /** Allocate `8` persistent `output_shallow` Coda KV banks (Contract A, `D3`/`DR23`). */
  readonly includeShallowCoda?: boolean;
  readonly config?: DoryMiniConfig;
}

export interface DoryKvCache {
  readonly mode: DoryKvCacheMode;
  readonly kLoops: number;
  readonly includeShallowCoda: boolean;
  readonly config: DoryMiniConfig;
  /** Keyed by physical layer index `p in {0, 2, 4, 6, 8, 10, 12, 14}`. */
  readonly inputBanks: ReadonlyMap<number, DoryKvBank>;
  /**
   * Indexed by `loopIndex = (mode === 'per_loop' ? t - 1 : 0)`.
   * Each map is keyed by physical layer index `p in {16, 18, ..., 54}`.
   */
  readonly recurrentBanks: readonly ReadonlyMap<number, DoryKvBank>[];
  /** Keyed by physical layer index `p in {56, 58, 60, 62, 64, 66, 68, 70}`. */
  readonly outputDeepBanks: ReadonlyMap<number, DoryKvBank>;
  /** Non-null iff `includeShallowCoda === true` (Contract A persistent Shallow Coda KV). */
  readonly outputShallowBanks: ReadonlyMap<number, DoryKvBank> | null;
  /** All allocated banks in ascending `bankSlotId` order (`36`, `16 + 20k`, or `104`). */
  readonly allBanks: readonly DoryKvBank[];
  /** Global committed sequence length across all banks (`0..S_max`). */
  committedLen: number;
  /** Exact recorded forward chunk-size partition schedule for `last_loop` replay (`DR24`). */
  readonly chunkSizes: number[];
}

export interface DoryKvCacheByteUsage {
  readonly bankCount: number;
  readonly kvPayloadBytes: number;
  readonly tagMetadataBytes: number;
  readonly totalUniqueBufferBytes: number;
}

// ============================================================================
// ATTENTION & SWIGLU SUBLAYER INTERFACES (Slice 1B: `attention.ts`, `swiglu.ts`)
// ============================================================================

export interface DoryAttentionSublayerInput {
  readonly hiddenStates: Float32Array;
  readonly seqLen: number;
  readonly startPos: number;
  readonly physicalLayerIdx: number;
  /** `0` for Prelude/Coda, `1..k` for Recurrent passes. */
  readonly loopIteration: number;
  readonly weights: DoryAttentionSublayerWeights;
  readonly ropeTables: DoryDualRopeTables;
  readonly kvBank: DoryKvBank;
  /**
   * True to commit `[startPos..startPos+seqLen-1]` into `kvBank` after attention;
   * False for non-final `last_loop` passes (`t < k`, `DoryPastReader` semantics) or read-only
   * counterfactual Coda evaluation.
   */
  readonly writeKv: boolean;
  readonly config: DoryMiniConfig;
  readonly captureTrace?: boolean;
}

export interface DoryAttentionSublayerTrace {
  readonly kind: 'attention';
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly stage: DoryLayerStage;
  readonly loopIteration: number;
  readonly attentionKind: DoryAttentionKind;
  readonly ropeRegime: DoryRopeRegime;
  readonly ropeTheta: number;
  readonly seqLen: number;
  readonly startPos: number;
  readonly pastLen: number;
  readonly kvLen: number;
  readonly writeKv: boolean;
  readonly bankSlotId: number;
  readonly bankKind: DoryKvBankKind;
  readonly kvAbsPositions: Int32Array;
  readonly kvWriterLoops: Int32Array;
  /** `[seqLen, q_dim]` pre-RoPE Q. */
  readonly qPre: Float32Array;
  /** `[seqLen, kv_dim]` pre-RoPE K. */
  readonly kPre: Float32Array;
  /** `[seqLen, kv_dim]` V (unnormalized). */
  readonly vNew: Float32Array;
  /** `[seqLen, q_dim]` Gate projection `G = h W_g^T` (unnormalized). */
  readonly gPre: Float32Array;
  /** `[seqLen, q_dim]` post-RoPE Q. */
  readonly qRot: Float32Array;
  /** `[seqLen, kv_dim]` post-RoPE K. */
  readonly kRot: Float32Array;
  /** `[seqLen, q_dim]` post-RoPE per-head L2Norm Q. */
  readonly qNorm: Float32Array;
  /** `[seqLen, kv_dim]` post-RoPE per-head L2Norm K stored in cache. */
  readonly kCache: Float32Array;
  /** `[q_dim]` effective squared `sqk^2`. */
  readonly effectiveSqkSq: Float32Array;
  /** `[seqLen, q_dim]` `qNorm * effectiveSqkSq`. */
  readonly qScaled: Float32Array;
  /** `[seqLen * kvLen]` binary visibility (`1` = visible, `0` = masked; feeds `DR20` `RG32F` G). */
  readonly maskVisible: Uint8Array;
  /** `[h_q * seqLen * kvLen]` scaled dot-product logits (`-Infinity` where masked). */
  readonly scores: Float32Array;
  /** `[h_q * seqLen * kvLen]` softmax probabilities (`0.0` where masked). */
  readonly probs: Float32Array;
  /** `[seqLen, q_dim]` concatenated head context `C = P V`. */
  readonly attnContext: Float32Array;
  /** `[q_dim]` effective squared `s_gate^2`. */
  readonly effectiveSGateSq: Float32Array;
  /** `[seqLen, q_dim]` `sigmoid(G * effectiveSGateSq)`. */
  readonly gateValues: Float32Array;
  /** `[seqLen, q_dim]` `C * gateValues`. */
  readonly gatedContext: Float32Array;
  /** `[seqLen, q_dim]` `attn_sub_norm = N_{q_dim}(gatedContext)` over ALL concatenated heads! */
  readonly attnSubNorm: Float32Array;
  /** `[seqLen, d]` `O = attnSubNorm W_o^T` prior to NLERP residual. */
  readonly oProjOut: Float32Array;
  readonly nlerp: DoryNlerpTrace;
}

export interface DoryAttentionSublayerResult {
  /** Post-NLERP residual hidden state `[seqLen, d]`. */
  readonly output: Float32Array;
  /** Pre-NLERP sublayer projection `O = Y_norm W_o^T` `[seqLen, d]`. */
  readonly sublayerOut: Float32Array;
  readonly trace: DoryAttentionSublayerTrace | null;
}

export interface DoryMlpSublayerInput {
  readonly hiddenStates: Float32Array;
  readonly seqLen: number;
  readonly physicalLayerIdx: number;
  readonly loopIteration: number;
  readonly weights: DoryMlpSublayerWeights;
  readonly config: DoryMiniConfig;
  readonly captureTrace?: boolean;
}

export interface DoryMlpSublayerTrace {
  readonly kind: 'mlp';
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly stage: DoryLayerStage;
  readonly loopIteration: number;
  readonly seqLen: number;
  /** `[seqLen, d_ff]` raw `h W_gate^T`. */
  readonly gatePre: Float32Array;
  /** `[seqLen, d_ff]` raw `h W_up^T`. */
  readonly upPre: Float32Array;
  /** `[d_ff]` signed `suv_gate * sqrt(d)` (NO `.abs()`). */
  readonly effectiveSuvGate: Float32Array;
  /** `[d_ff]` signed `suv_up * sqrt(d)` (NO `.abs()`). */
  readonly effectiveSuvUp: Float32Array;
  /** `[seqLen, d_ff]` `gatePre * effectiveSuvGate`. */
  readonly gateScaled: Float32Array;
  /** `[seqLen, d_ff]` `upPre * effectiveSuvUp`. */
  readonly upScaled: Float32Array;
  /** `[seqLen, d_ff]` `SiLU(gateScaled)`. */
  readonly gateSilu: Float32Array;
  /** `[seqLen, d_ff]` `gateSilu * upScaled`. */
  readonly swigluProduct: Float32Array;
  /** `[seqLen, d_ff]` `ffn_sub_norm = N_{d_ff}(swigluProduct)`. */
  readonly ffnSubNorm: Float32Array;
  /** `[seqLen, d]` `O = ffnSubNorm W_down^T` prior to NLERP residual. */
  readonly downProjOut: Float32Array;
  readonly nlerp: DoryNlerpTrace;
}

export interface DoryMlpSublayerResult {
  /** Post-NLERP residual hidden state `[seqLen, d]`. */
  readonly output: Float32Array;
  /** Pre-NLERP sublayer projection `O = M W_down^T` `[seqLen, d]`. */
  readonly sublayerOut: Float32Array;
  readonly trace: DoryMlpSublayerTrace | null;
}

export type DorySublayerTrace = DoryAttentionSublayerTrace | DoryMlpSublayerTrace;

// ============================================================================
// FULL 72-SUBLAYER FORWARD PASS & STAGE-LEVEL INTERFACES (Slice 1B: `forward.ts`)
// ============================================================================

export interface DoryExecutedSublayerRecord {
  readonly executedLayerIdx: number;
  readonly coord: DoryExecutionCoord;
  readonly trace: DorySublayerTrace;
}

export interface DoryLoopTrajectoryMetrics {
  readonly loopIteration: number;
  readonly tokenOffset: number;
  readonly absTokenPos: number;
  /** `||u_t||_2` in original 8D. */
  readonly uNorm: number;
  /** `||s_t||_2` in original 8D. */
  readonly sNorm: number;
  /** `cos(s_t, s_{t-1})` in original 8D (`s_0 = x_anchor`). */
  readonly consecutiveCosine: number;
  /** `||s_t - s_{t-1}||_2` in original 8D. */
  readonly consecutiveDisplacement: number;
  /** `cos(s_t, x_anchor)` in original 8D. */
  readonly anchorCosine: number;
  /** Local boundary-map Jacobian evaluated at angle between `s_{t-1}` and `x_anchor` (Finding 9). */
  readonly entryAnchorJacobian: DoryAnchorJacobianMetrics;
}

export interface DoryLmHeadResult {
  /** `[seqLen, V]` raw `h_out W_lm^T`. */
  readonly rawLogits: Float32Array;
  /** `[seqLen, V]` FP32 scaled `rawLogits * (logit_scale / sigma_0)`. */
  readonly scaledLogits: Float32Array;
  /** `[V]` effective `logit_scale / sigma_0`. */
  readonly effectiveLogitScale: Float32Array;
  /** `[V]` binary64 softmax of the final token (`seqLen - 1`) at `temperature = 1.0`. */
  readonly lastTokenProbs: Float64Array;
}

export interface DoryForwardTrace {
  readonly modelWeightId: string;
  readonly sessionBranchId: string;
  readonly forwardStepId: number;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly role: DoryCacheRole;
  readonly codaKind: DoryCodaBankKind;
  readonly seqLen: number;
  readonly startPos: number;
  readonly executedSublayerCount: number;
  readonly executedSublayers: readonly DoryExecutedSublayerRecord[];
  readonly loopMetrics: readonly DoryLoopTrajectoryMetrics[];
}

export interface DoryStagePreludeInput {
  readonly tokens: ArrayLike<number>;
  readonly startPos: number;
  readonly params: DoryModelParameters;
  readonly kvCache: DoryKvCache;
  readonly ropeTables?: DoryDualRopeTables;
  readonly writeKv?: boolean;
  readonly captureTrace?: boolean;
  readonly forwardStepId?: number;
  readonly sessionBranchId?: string;
  readonly role?: DoryCacheRole;
  readonly kLoopsForCoord?: number;
}

export interface DoryStagePreludeResult {
  readonly embeddings: Float32Array;
  readonly xAnchor: Float32Array;
  readonly executedSublayers: readonly DoryExecutedSublayerRecord[];
}

export interface DoryStageRecurrentPassInput {
  readonly prevState: Float32Array;
  readonly xAnchor: Float32Array;
  readonly seqLen: number;
  readonly startPos: number;
  readonly loopIteration: number;
  readonly kLoops: number;
  readonly params: DoryModelParameters;
  readonly kvCache: DoryKvCache;
  readonly ropeTables?: DoryDualRopeTables;
  /** Defaults to `kvCache.mode === 'per_loop' || loopIteration === kLoops`. */
  readonly writeKv?: boolean;
  readonly captureTrace?: boolean;
  readonly forwardStepId?: number;
  readonly sessionBranchId?: string;
  readonly role?: DoryCacheRole;
}

export interface DoryStageRecurrentPassResult {
  /** Literal `u_t = N(prevState + xAnchor)` (incl. `u_1 = N(2x)`). */
  readonly uEntry: Float32Array;
  /** Literal `s_t = N(F_theta(u_t))`. */
  readonly sExit: Float32Array;
  readonly executedSublayers: readonly DoryExecutedSublayerRecord[];
  readonly loopMetrics: readonly DoryLoopTrajectoryMetrics[];
}

export interface DoryStageCodaInput {
  readonly state: Float32Array;
  readonly seqLen: number;
  readonly startPos: number;
  readonly kLoops: number;
  readonly params: DoryModelParameters;
  readonly kvCache: DoryKvCache;
  readonly ropeTables?: DoryDualRopeTables;
  readonly codaKind?: DoryCodaBankKind;
  /** Set `false` for View 2 read-only Diagnostic Counterfactual logit-lens! */
  readonly writeKv?: boolean;
  readonly captureTrace?: boolean;
  readonly forwardStepId?: number;
  readonly sessionBranchId?: string;
  readonly role?: DoryCacheRole;
}

export interface DoryStageCodaResult {
  readonly codaHidden: Float32Array;
  readonly lmHead: DoryLmHeadResult;
  readonly executedSublayers: readonly DoryExecutedSublayerRecord[];
}

export interface DoryForwardChunkInput {
  readonly tokens: ArrayLike<number>;
  readonly params: DoryModelParameters;
  /** Optional persistent KV cache; if omitted, an ephemeral cache is created starting at `startPos = 0`. */
  readonly kvCache?: DoryKvCache;
  /** Explicit starting absolute position; defaults to `kvCache?.committedLen ?? 0`. */
  readonly startPos?: number;
  /** Loop budget `k >= 1`; defaults to `kvCache?.kLoops ?? params.config.defaultRecurrentLoops`. */
  readonly kLoops?: number;
  /** KV cache policy; defaults to `kvCache?.mode ?? params.config.defaultKvCacheMode`. */
  readonly kvCacheMode?: DoryKvCacheMode;
  readonly ropeTables?: DoryDualRopeTables;
  readonly codaKind?: DoryCodaBankKind;
  readonly writeKv?: boolean;
  readonly captureTrace?: boolean;
  readonly forwardStepId?: number;
  readonly sessionBranchId?: string;
  readonly role?: DoryCacheRole;
}

export interface DoryForwardChunkResult {
  readonly tokens: Int32Array;
  readonly seqLen: number;
  readonly startPos: number;
  readonly endPos: number;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executedSublayerCount: number;
  readonly kvCache: DoryKvCache;
  /** `[seqLen, d]` token embeddings (unit-norm synthetic rows). */
  readonly embeddings: Float32Array;
  /** `[seqLen, d]` Prelude output `x_anchor = s_0`. */
  readonly xAnchor: Float32Array;
  /** Length `kLoops`: `[u_1, ..., u_k]`, each `[seqLen, d]`. */
  readonly loopEntryStates: readonly Float32Array[];
  /** Length `kLoops`: `[s_1, ..., s_k]`, each `[seqLen, d]`. */
  readonly loopExitStates: readonly Float32Array[];
  /** `[seqLen, d]` Coda output before LM head. */
  readonly codaHidden: Float32Array;
  /** `[seqLen, V]` raw LM head logits `h_out W_lm^T`. */
  readonly rawLogits: Float32Array;
  /** `[seqLen, V]` FP32 scaled logits `rawLogits * (s_z / sigma_0)`. */
  readonly scaledLogits: Float32Array;
  /** `[V]` binary64 softmax probabilities of final token in chunk. */
  readonly lastTokenProbs: Float64Array;
  /** Detailed execution trace (populated iff `captureTrace === true`). */
  readonly trace: DoryForwardTrace | null;
}

// ============================================================================
// FROZEN SLICE 1A & SLICE 1B MODULE EXPORT CONTRACTS (ZERO IMPORT MISMATCHES)
// ============================================================================
//
// Slice 1A MUST export these exact symbol names and signatures from each module so
// Slice 1B (`kvCache.ts`, `attention.ts`, `swiglu.ts`, `forward.ts`, `index.ts`) and
// Slice 1C (`export_dory_params.mjs`, `tests/dory-math-parity.test.ts`) import verbatim:
//
//   1. `src/dory/spec.ts`        -> implements `DorySpecModuleContract`
//      Exports: `DORY_MINI_CONFIG` (re-export), `DORY_PRODUCTION_SPEC`,
//      `DORY_CHECKPOINT_FORMATS`, `DORY_SCALING_DISCLOSURE`, `DORY_PHYSICAL_LAYERS`,
//      `getDoryPhysicalLayerSpec`, `getDoryStage`, `getDorySublayerKind`,
//      `isDoryAttentionLayer`, `isDorySlidingAttentionLayer`, `isDoryFullAttentionLayer`,
//      `isDoryMlpLayer`, `computeExecutedDepth`, `computeEffectivePairedDepth`,
//      `computeExecutedLayerIdx`, `computeKvBankCounts`, `computeMiniKvCacheBytes`,
//      `computeProductionKvBytesBf16`, `validateDoryRecurrentLoops`,
//      `decodeDoryToken`, `decodeDoryTokens`.
//
//   2. `src/dory/ledger.ts`      -> implements `DoryLedgerModuleContract`
//      Exports: `DORY_DISCREPANCY_LEDGER`, `DORY_DESIGN_DECISIONS`,
//      `getDoryDiscrepancyById`, `getDoryBrowserGuards`.
//
//   3. `src/dory/prng.ts`        -> implements `DoryPrngModuleContract`
//      Exports: `splitMix32`, `deriveDorySeed`, `createDoryPrng`.
//
//   4. `src/dory/parameters.ts`  -> implements `DoryParametersModuleContract`
//      Exports: `createDoryModelParameters`, `getDefaultDoryModelParameters`,
//      `listDoryCanonicalTensors`, `computeDoryWeightDigestHex`,
//      `serializeDoryParametersJson`, `countDoryParameterStats`.
//
//   5. `src/dory/ngptMath.ts`    -> implements `DoryNgptMathModuleContract`
//      Exports: `l2NormVector`, `l2NormSliceInto`, `l2NormRows`, `l2NormHeadsInto`,
//      `resolveOutputGateInitScale`, `computeScaleFamilyMultipliers`,
//      `effectiveResidualAlpha`, `effectiveSqkSquared`, `effectiveSGateSquared`,
//      `effectiveSuvScale`, `effectiveLogitScale`,
//      `ngptNlerpResidualRows`, `ngptNlerpResidualRowsDetailed`,
//      `loopEntryAnchorInject`, `loopExitNormalize`,
//      `linearProject`, `linearProjectInto`,
//      `sigmoidScalar`, `siluScalar`, `softmaxFloat64`, `argmaxIndex`,
//      `cosineSimilarity8d`, `l2Distance`, `vectorL2Norm`, `anchorMapJacobianNorm`.
//
//   6. `src/dory/rope.ts`        -> implements `DoryRopeModuleContract`
//      Exports: `computeDoryRopeInvFreq`, `buildDoryRopeTable`, `buildDoryDualRopeTables`,
//      `applyNeoXRotateHalf`, `applyNeoXRotateHalfAtPositions`, `applyNeoXRotateHalfInto`.
//
// Slice 1B exports:
//   7. `src/dory/kvCache.ts`     -> implements `DoryKvCacheModuleContract`
//   8. `src/dory/attention.ts`   -> implements `DoryAttentionModuleContract`
//   9. `src/dory/swiglu.ts`      -> implements `DorySwigluModuleContract`
//  10. `src/dory/forward.ts`     -> implements `DoryForwardModuleContract`
//  11. `src/dory/index.ts`       -> barrel re-exporting all DORY1 modules cleanly.

export interface DorySpecModuleContract {
  readonly DORY_MINI_CONFIG: DoryMiniConfig;
  readonly DORY_PRODUCTION_SPEC: DoryProductionSpec;
  readonly DORY_CHECKPOINT_FORMATS: readonly DoryCheckpointFormatSpec[];
  readonly DORY_SCALING_DISCLOSURE: readonly DoryScalingRatioEntry[];
  readonly DORY_PHYSICAL_LAYERS: readonly DoryPhysicalLayerSpec[];
  getDoryPhysicalLayerSpec(physicalLayerIdx: number): DoryPhysicalLayerSpec;
  getDoryStage(physicalLayerIdx: number): DoryLayerStage;
  getDorySublayerKind(physicalLayerIdx: number): DorySublayerKind;
  isDoryAttentionLayer(physicalLayerIdx: number): boolean;
  isDorySlidingAttentionLayer(physicalLayerIdx: number): boolean;
  isDoryFullAttentionLayer(physicalLayerIdx: number): boolean;
  isDoryMlpLayer(physicalLayerIdx: number): boolean;
  computeExecutedDepth(kLoops: number, config?: DoryMiniConfig): number;
  computeEffectivePairedDepth(kLoops: number, config?: DoryMiniConfig): number;
  computeExecutedLayerIdx(physicalLayerIdx: number, loopIteration: number, kLoops: number): number;
  computeKvBankCounts(
    kLoops: number,
    kvCacheMode: DoryKvCacheMode,
    includeShallowCoda?: boolean,
  ): DoryKvBankCountSummary;
  computeMiniKvCacheBytes(
    kLoops: number,
    kvCacheMode: DoryKvCacheMode,
    includeShallowCoda?: boolean,
    config?: DoryMiniConfig,
  ): DoryMiniKvMemorySummary;
  computeProductionKvBytesBf16(
    seqLen: number,
    kLoops: number,
    kvCacheMode: DoryKvCacheMode,
  ): DoryProductionKvPoint;
  validateDoryRecurrentLoops(kLoops: number): number;
  decodeDoryToken(tokenId: number): string;
  decodeDoryTokens(tokenIds: ArrayLike<number>): string[];
}

export interface DoryLedgerModuleContract {
  readonly DORY_DISCREPANCY_LEDGER: readonly DoryDiscrepancyEntry[];
  readonly DORY_DESIGN_DECISIONS: readonly DoryDesignDecisionEntry[];
  getDoryDiscrepancyById(id: DoryDiscrepancyId): DoryDiscrepancyEntry;
  getDoryBrowserGuards(): readonly DoryDiscrepancyEntry[];
}

export interface DoryPrngModuleContract {
  splitMix32(seed: number): number;
  deriveDorySeed(baseSeed: number, salt: number): number;
  createDoryPrng(seed: number): DoryPrng;
}

export interface DoryParametersModuleContract {
  createDoryModelParameters(seed?: number, config?: DoryMiniConfig): DoryModelParameters;
  getDefaultDoryModelParameters(): DoryModelParameters;
  listDoryCanonicalTensors(params: DoryModelParameters): readonly DoryCanonicalWeightTensor[];
  computeDoryWeightDigestHex(params: DoryModelParameters): string;
  serializeDoryParametersJson(params: DoryModelParameters): DorySerializedParametersPayload;
  countDoryParameterStats(params: DoryModelParameters): DoryParameterCountSummary;
}

export interface DoryNgptMathModuleContract {
  l2NormVector(input: Float32Array, eps?: number): Float32Array;
  l2NormSliceInto(
    src: Float32Array,
    srcOffset: number,
    dst: Float32Array,
    dstOffset: number,
    length: number,
    eps?: number,
  ): void;
  l2NormRows(input: Float32Array, rows: number, cols: number, eps?: number): Float32Array;
  l2NormHeadsInto(
    src: Float32Array,
    dst: Float32Array,
    seqLen: number,
    numHeads: number,
    headDim: number,
    eps?: number,
  ): void;
  resolveOutputGateInitScale(
    config: Pick<DoryMiniConfig, 'hiddenSize' | 'attentionOutputGateLogitScaleInit'>,
  ): number;
  computeScaleFamilyMultipliers(
    config?: Pick<
      DoryMiniConfig,
      | 'hiddenSize'
      | 'headDim'
      | 'initializerRange'
      | 'ngptAlphaInit'
      | 'attentionOutputGateLogitScaleInit'
    >,
  ): DoryScaleFamilyMultipliers;
  effectiveResidualAlpha(
    rawAlpha: Float32Array,
    config?: Pick<DoryMiniConfig, 'ngptAlphaInit' | 'initializerRange'>,
  ): Float32Array;
  effectiveSqkSquared(
    rawSqk: Float32Array,
    config?: Pick<DoryMiniConfig, 'headDim' | 'initializerRange'>,
  ): Float32Array;
  effectiveSGateSquared(
    rawSGate: Float32Array,
    config?: Pick<
      DoryMiniConfig,
      'hiddenSize' | 'attentionOutputGateLogitScaleInit' | 'initializerRange'
    >,
  ): Float32Array;
  effectiveSuvScale(
    rawSuv: Float32Array,
    config?: Pick<DoryMiniConfig, 'hiddenSize'>,
  ): Float32Array;
  effectiveLogitScale(
    rawLogitScale: Float32Array,
    config?: Pick<DoryMiniConfig, 'initializerRange'>,
  ): Float32Array;
  ngptNlerpResidualRows(
    residual: Float32Array,
    sublayerOut: Float32Array,
    rawAlpha: Float32Array,
    seqLen: number,
    hiddenSize: number,
    config?: Pick<DoryMiniConfig, 'ngptAlphaInit' | 'initializerRange' | 'l2NormEps'>,
  ): Float32Array;
  ngptNlerpResidualRowsDetailed(
    residual: Float32Array,
    sublayerOut: Float32Array,
    rawAlpha: Float32Array,
    seqLen: number,
    hiddenSize: number,
    config?: Pick<DoryMiniConfig, 'ngptAlphaInit' | 'initializerRange' | 'l2NormEps'>,
  ): DoryNlerpTrace;
  loopEntryAnchorInject(
    prevState: Float32Array,
    xAnchor: Float32Array,
    seqLen: number,
    hiddenSize: number,
    eps?: number,
  ): Float32Array;
  loopExitNormalize(
    coreOut: Float32Array,
    seqLen: number,
    hiddenSize: number,
    eps?: number,
  ): Float32Array;
  linearProject(
    input: Float32Array,
    weight: Float32Array,
    seqLen: number,
    inDim: number,
    outDim: number,
  ): Float32Array;
  linearProjectInto(
    input: Float32Array,
    weight: Float32Array,
    dst: Float32Array,
    seqLen: number,
    inDim: number,
    outDim: number,
  ): void;
  sigmoidScalar(x: number): number;
  siluScalar(x: number): number;
  softmaxFloat64(
    logits: ArrayLike<number>,
    offset?: number,
    length?: number,
    temperature?: number,
  ): Float64Array;
  argmaxIndex(values: ArrayLike<number>, offset?: number, length?: number): number;
  cosineSimilarity8d(
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    aOffset?: number,
    bOffset?: number,
    length?: number,
  ): number;
  l2Distance(
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    aOffset?: number,
    bOffset?: number,
    length?: number,
  ): number;
  vectorL2Norm(v: ArrayLike<number>, offset?: number, length?: number): number;
  anchorMapJacobianNorm(angleRadians: number, eps?: number): DoryAnchorJacobianMetrics;
}

export interface DoryRopeModuleContract {
  computeDoryRopeInvFreq(headDim: number, theta: number): Float32Array;
  buildDoryRopeTable(
    maxPositions: number,
    headDim: number,
    theta: number,
    regime: DoryRopeRegime,
  ): DoryRopeTable;
  buildDoryDualRopeTables(
    config?: Pick<
      DoryMiniConfig,
      'maxPositionEmbeddings' | 'headDim' | 'ropeThetaSliding' | 'ropeThetaFull'
    >,
  ): DoryDualRopeTables;
  applyNeoXRotateHalf(
    input: Float32Array,
    seqLen: number,
    numHeads: number,
    headDim: number,
    startPos: number,
    table: DoryRopeTable,
  ): Float32Array;
  applyNeoXRotateHalfAtPositions(
    input: Float32Array,
    positionIds: ArrayLike<number>,
    numHeads: number,
    headDim: number,
    table: DoryRopeTable,
  ): Float32Array;
  applyNeoXRotateHalfInto(
    src: Float32Array,
    dst: Float32Array,
    seqLen: number,
    numHeads: number,
    headDim: number,
    startPos: number,
    table: DoryRopeTable,
  ): void;
}

export interface DoryKvCacheModuleContract {
  createDoryKvCache(options: DoryKvCacheCreateOptions): DoryKvCache;
  cloneDoryKvCache(cache: DoryKvCache): DoryKvCache;
  getDoryInputBank(cache: DoryKvCache, physicalLayerIdx: number): DoryKvBank;
  getDoryRecurrentBank(
    cache: DoryKvCache,
    physicalLayerIdx: number,
    loopIteration: number,
  ): DoryKvBank;
  getDoryOutputBank(
    cache: DoryKvCache,
    physicalLayerIdx: number,
    codaKind?: DoryCodaBankKind,
  ): DoryKvBank;
  buildTransientKvWorkspace(
    bank: DoryKvBank,
    startPos: number,
    currentK: Float32Array,
    currentV: Float32Array,
    chunkLen: number,
    currentWriterLoop: number,
  ): DoryTransientKvWorkspace;
  commitChunkToKvBank(
    bank: DoryKvBank,
    startPos: number,
    currentK: Float32Array,
    currentV: Float32Array,
    chunkLen: number,
    writerLoop: number,
  ): void;
  rewindDoryKvCache(cache: DoryKvCache, targetLen: number): void;
  measureDoryKvCacheBytes(cache: DoryKvCache): DoryKvCacheByteUsage;
}

export interface DoryAttentionModuleContract {
  runDoryAttentionSublayer(input: DoryAttentionSublayerInput): DoryAttentionSublayerResult;
}

export interface DorySwigluModuleContract {
  runDoryMlpSublayer(input: DoryMlpSublayerInput): DoryMlpSublayerResult;
}

export interface DoryForwardModuleContract {
  runDoryInputPrelude(input: DoryStagePreludeInput): DoryStagePreludeResult;
  runDoryRecurrentPass(input: DoryStageRecurrentPassInput): DoryStageRecurrentPassResult;
  runDoryOutputCoda(input: DoryStageCodaInput): DoryStageCodaResult;
  runDoryLmHead(
    hiddenStates: Float32Array,
    seqLen: number,
    params: DoryModelParameters,
  ): DoryLmHeadResult;
  runDoryForwardChunk(input: DoryForwardChunkInput): DoryForwardChunkResult;
}

// ============================================================================
// DORY2 CONSTANTS, INVARIANT ERRORS & SAMPLING CONTRACTS (D3, D6, D10, DR18/23/24)
// ============================================================================

/**
 * Mandatory provenance disclosure badge for Contract A Paper §6.2 self-speculation (`D3`, `DR23`).
 * Paper §6.2 is absent from upstream `transformers`, `vLLM`, and `llama.cpp`, and omits Output/Coda
 * KV management across cycles; Contract A allocates `8` dedicated persistent `output_shallow` banks
 * alongside the `96` deep `per_loop` banks (`104` banks total).
 */
export const DORY_SPEC_CONTRACT_A_BADGE =
  'Reconstructed from Paper §6.2 (Contract A) · Not in Upstream Runtimes · per_loop Only' as const;

/** Exact Contract A persistent bank count (`8` Input + `80` Loops 1–4 + `8` Deep Coda + `8` Shallow Coda). */
export const DORY_SPEC_CONTRACT_A_BANK_COUNT = 104 as const;
export const DORY_SPEC_DRAFT_LOOPS = 1 as const;
export const DORY_SPEC_VERIFY_LOOPS = 4 as const;
export const DORY_SPEC_DEFAULT_GAMMA = 4 as const;

/** Gate 2 Layer 6 Pearson chi-squared critical value (`df = 63`, `alpha = 0.001`). */
export const DORY_SPEC_PEARSON_CHI2_DF63_ALPHA_001 = 103.51 as const;

/** View 2 read-only intermediate-loop logit-lens disclosure badge (Audit §12.1, Finding 9). */
export const DORY_COUNTERFACTUAL_CODA_BADGE =
  'DIAGNOSTIC COUNTERFACTUAL · CODA NOT RUN AT INTERMEDIATE LOOPS' as const;

/** Default deterministic session PRNG seed (distinct from backbone weight seed). */
export const DORY_DEFAULT_SESSION_SEED = 0xd0412002 as const;

/**
 * Default non-EOS 8-token miniature session prompt:
 * `['<s>', '<|im_start|>', 'user', '\n', 'solve', 'loop', 'depth', '4']`.
 */
export const DORY_DEFAULT_SESSION_PROMPT: readonly number[] = Object.freeze([
  1, 10, 4, 18, 40, 25, 26, 57,
]);

export type DoryExecutionMode = 'standard' | 'self_spec';

export type DoryHaltReason = 'eos' | 'capacity_exhausted' | 'max_new_tokens';

export type DorySpecOutcomeKind =
  | 'zero_accept_reject'
  | 'partial_accept_reject'
  | 'full_accept_bonus'
  | 'full_accept_eos'
  | 'final_slot_deep_decode';

export type DoryRejectionReason =
  | 'greedy_match'
  | 'greedy_mismatch'
  | 'unconditional_p_ge_q'
  | 'stochastic_accept'
  | 'stochastic_reject';

export type DorySessionRecordKind =
  | 'prefill'
  | 'chunk'
  | 'decode_step'
  | 'spec_prefill'
  | 'spec_cycle';

export type DorySessionForkReason = 'initial' | 'cursor_fork' | 'dr18_re_prefill';

/**
 * Shared binary64 sampling configuration (`Audit §5.6 & §8.2`).
 * Transform consistency invariant: the identical `(temperature, topK, topP)` filtering MUST be
 * applied both when sampling draft token `d` from `q` and when computing `(q, p)` passed to
 * division-free rejection sampling and residual correction.
 */
export interface DorySamplingConfig {
  /** Sampling temperature (`>= 0`; `0` = deterministic greedy argmax). */
  readonly temperature: number;
  /** Top-k cutoff (`0` = disabled, matching production `top_k = 0`; positive integer `1..V`). */
  readonly topK: number;
  /** Nucleus cumulative probability threshold in `(0, 1]` (`1.0` = disabled). */
  readonly topP: number;
}

export const DORY_DEFAULT_GREEDY_SAMPLING: DorySamplingConfig = Object.freeze({
  temperature: 0,
  topK: 0,
  topP: 1.0,
});

export const DORY_DEFAULT_STOCHASTIC_SAMPLING: DorySamplingConfig = Object.freeze({
  temperature: 1.0,
  topK: 0,
  topP: 0.95,
});

/**
 * Hard invariant error thrown by binary64 rejection sampling when `q[d] <= 0` or when residual
 * mass `Z = sum_v max(0, p[v] - q[v]) <= 0` under malformed distributions (`Audit §8.2`, Finding 3).
 * Never silently clamp tiny denominators (`1e-38`) or fall back to `p` (`1e-12`).
 */
export class DorySpecSupportInvariantError extends Error {
  readonly code: 'NON_POSITIVE_PROPOSAL_PROB' | 'ZERO_RESIDUAL_MASS' | 'INVALID_DISTRIBUTION';

  constructor(
    code: 'NON_POSITIVE_PROPOSAL_PROB' | 'ZERO_RESIDUAL_MASS' | 'INVALID_DISTRIBUTION',
    message: string,
  ) {
    super(message);
    this.name = 'DorySpecSupportInvariantError';
    this.code = code;
  }
}

/**
 * Hard `DR18` / `D6` / `DR23` guard error thrown when a caller attempts to mutate `kLoops`,
 * `kvCacheMode`, or `executionMode` in place on a populated branch without an explicit clean
 * fork/re-prefill transition, or requests Contract A `self_spec` under `last_loop` / `kLoops !== 4`.
 */
export class DorySessionImmutableConfigError extends Error {
  readonly code:
    | 'DR18_INCOMPATIBLE_BRANCH_CONFIG'
    | 'DR18_REPREFILL_REQUIRED'
    | 'DR23_SPEC_REQUIRES_PER_LOOP_K4';

  constructor(
    code:
      | 'DR18_INCOMPATIBLE_BRANCH_CONFIG'
      | 'DR18_REPREFILL_REQUIRED'
      | 'DR23_SPEC_REQUIRES_PER_LOOP_K4',
    message: string,
  ) {
    super(message);
    this.name = 'DorySessionImmutableConfigError';
    this.code = code;
  }
}

// ============================================================================
// CONTRACT A §6.2 SELF-SPECULATIVE DECODING INTERFACES (Slice 2A: `speculative.ts`)
// ============================================================================

/** Result of one division-free binary64 acceptance decision (`Audit §8.2`). */
export interface DoryRejectionSampleResult {
  readonly draftTokenId: number;
  /** `q[draftTokenId]` in binary64 (asserted strictly `> 0` in stochastic mode). */
  readonly draftProb: number;
  /** `p[draftTokenId]` in binary64. */
  readonly targetProb: number;
  /** Uniform variate `r in [0, 1)` used when `p[d] < q[d]`, or `null` on unconditional accept / greedy. */
  readonly uniformVariate: number | null;
  /** True iff accepted (`p[d] >= q[d]` unconditionally, `r * q[d] < p[d]` division-free, or greedy match). */
  readonly accepted: boolean;
  readonly reason: DoryRejectionReason;
}

/** Exact binary64 residual correction distribution `p_res[v] = max(0, p[v] - q[v]) / Z` (`Audit §8.2`). */
export interface DoryResidualDistributionResult {
  /** Normalized residual distribution (`[V]`). */
  readonly probs: Float64Array;
  /** Strictly positive binary64 normalization sum `Z = sum_v max(0, p[v] - q[v]) > 0`. */
  readonly massZ: number;
}

/** Result of a standalone single-position rejection + residual draw (Gate 2 Layers 3, 4, 6). */
export interface DorySpecSingleStepSampleResult {
  readonly draftTokenId: number;
  readonly emittedTokenId: number;
  readonly decision: DoryRejectionSampleResult;
  readonly residual: DoryResidualDistributionResult | null;
}

/** Exact analytical 1-step marginal verification result (Gate 2 Layers 1 & 2, `<= 1e-15`). */
export interface DorySpecAnalyticalMarginalResult {
  /** Exact marginal probability `P_spec(v)` over all draft outcomes (`[V]`). */
  readonly marginalProbs: Float64Array;
  /** Exact total acceptance probability `alpha = sum_v min(p[v], q[v])`. */
  readonly acceptanceRate: number;
  /** Exact `max_v |P_spec(v) - p(v)|` in binary64 (must be `<= 1e-15`). */
  readonly maxAbsError: number;
}

/** Exact analytical 2-step joint speculative marginal result (`V = 4`, Gate 2 Layer 2, `<= 1e-15`). */
export interface DorySpecTwoStepAnalyticalResult {
  readonly vocabSize: number;
  /** Row-major `[V * V]` exact speculative joint distribution `P_spec(v1, v2)`. */
  readonly jointProbs: Float64Array;
  /** Row-major `[V * V]` exact target joint distribution `p_1(v1) * p_2(v2 | v1)`. */
  readonly targetJointProbs: Float64Array;
  /** Exact `max_{v1,v2} |P_spec(v1, v2) - P_target(v1, v2)|` in binary64 (must be `<= 1e-15`). */
  readonly maxAbsError: number;
}

/** Pearson chi-squared goodness-of-fit summary (Gate 2 Layer 6). */
export interface DoryPearsonChiSquaredResult {
  readonly statistic: number;
  readonly degreesOfFreedom: number;
  readonly sampleCount: number;
  readonly minExpectedCount: number;
  readonly totalVariationDistance: number;
}

/** Explicit uniform variates for deterministic Gate 2 Layer 3 & 4 boundary tests. */
export interface DorySpecExplicitVariates {
  /** Optional explicit draft sampling uniforms `[j]` in `[0, 1)` for `j = 0 .. gammaEff - 1`. */
  readonly draftSampleUniforms?: readonly number[];
  /** Optional explicit rejection-test uniforms `[j]` in `[0, 1)` for `j = 0 .. gammaPrime - 1`. */
  readonly acceptUniforms?: readonly number[];
  /** Optional explicit uniform in `[0, 1)` for residual correction (`y*`) or bonus (`y_bonus`) sampling. */
  readonly correctionOrBonusUniform?: number;
}

/** Per-step shallow (`k = 1`, `output_shallow` Coda) draft record inside one Contract A cycle. */
export interface DorySpecDraftStepRecord {
  /** Draft step offset `j in 0 .. draftCount - 1`. */
  readonly draftOffset: number;
  /** Absolute KV position `L + j` consumed in this draft step. */
  readonly absPos: number;
  /** Input token `d_j` consumed at `absPos`. */
  readonly inputTokenId: number;
  /** Proposed next token `d_{j+1}` sampled from `q_{j+1}`. */
  readonly proposedTokenId: number;
  /** True iff `proposedTokenId` is in `DORY_EOS_TOKEN_IDS`. */
  readonly proposedIsEos: boolean;
  /** Shallow `k = 1` scaled logits (`[V]`). */
  readonly shallowScaledLogits: Float32Array;
  /** Shallow `k = 1` filtered binary64 proposal distribution `q_{j+1}` (`[V]`). */
  readonly proposalProbs: Float64Array;
  /** Uniform variate used to sample `proposedTokenId` (`null` in greedy mode). */
  readonly sampleUniform: number | null;
}

/** Per-position deep (`k = 4`, `output_deep` Coda) batched verify record inside one Contract A cycle. */
export interface DorySpecVerifyPositionRecord {
  /** Verify offset `j in 0 .. verifiedPositions - 1`. */
  readonly verifyOffset: number;
  /** Absolute KV position `L + j`. */
  readonly absPos: number;
  /** Token `d_j` verified at `absPos`. */
  readonly inputTokenId: number;
  /** True for `[L .. L + draftCount - 1]` (reused Prelude + Loop-1 KV & `(x, s_1)`); false for bonus slot `L + draftCount`. */
  readonly reusedPreludeAndLoop1: boolean;
  /** Deep `k = 4` scaled logits (`[V]`). */
  readonly deepScaledLogits: Float32Array;
  /** Deep `k = 4` filtered binary64 target distribution `p_{j+1}` (`[V]`). */
  readonly targetProbs: Float64Array;
  /** Rejection decision for `d_{j+1}` (`null` for the extra bonus slot `j === draftCount` or final-slot decode). */
  readonly rejection: DoryRejectionSampleResult | null;
}

export interface DorySpecCyclePrefillInput {
  readonly promptTokens: ArrayLike<number>;
  readonly params: DoryModelParameters;
  readonly sampling?: Partial<DorySamplingConfig>;
  readonly prng?: DoryPrng;
  /** Optional explicit uniform in `[0, 1)` for sampling initial `pendingToken = d_0`. */
  readonly initialSampleUniform?: number;
  readonly ropeTables?: DoryDualRopeTables;
  readonly sessionBranchId?: string;
  readonly forwardStepId?: number;
  readonly captureTrace?: boolean;
}

/**
 * Contract A initial prefill output (`Audit §8.1.1` Transition 1):
 * Commits all `104` banks (`8` Input + `80` Loops 1–4 + `8` Deep Coda + `8` Shallow Coda on cached
 * `s_1`) to `committedLen = P`, and samples initial unconsumed `pendingToken = d_0` from Deep Coda
 * at position `P - 1`.
 */
export interface DorySpecCyclePrefillResult {
  readonly contractBadge: typeof DORY_SPEC_CONTRACT_A_BADGE;
  readonly kvCache: DoryKvCache;
  readonly promptTokens: Int32Array;
  readonly committedLen: number;
  /** Initial emitted token `d_0` sampled from Deep Coda at pos `P - 1` (unconsumed). */
  readonly pendingToken: number;
  /** Full emitted sequence `[...promptTokens, pendingToken]`. */
  readonly emittedTokens: Int32Array;
  /** `[P, d]` Prelude anchor `x`. */
  readonly xAnchor: Float32Array;
  /** Length `4`: `[u_1, u_2, u_3, u_4]`, each `[P, d]`. */
  readonly loopEntryStates: readonly Float32Array[];
  /** Length `4`: `[s_1, s_2, s_3, s_4]`, each `[P, d]`. */
  readonly loopExitStates: readonly Float32Array[];
  /** Length `4`: exact 8D loop metrics for final prompt token `P - 1`. */
  readonly lastTokenLoopMetrics: readonly DoryLoopTrajectoryMetrics[];
  /** Deep `k = 4` scaled logits at pos `P - 1` (`[V]`). */
  readonly lastDeepScaledLogits: Float32Array;
  /** Deep `k = 4` filtered binary64 probabilities at pos `P - 1` (`[V]`). */
  readonly lastDeepProbs: Float64Array;
  /** Shallow `k = 1` scaled logits at pos `P - 1` (`[V]`). */
  readonly lastShallowScaledLogits: Float32Array;
  /** Shallow `k = 1` filtered binary64 probabilities at pos `P - 1` (`[V]`). */
  readonly lastShallowProbs: Float64Array;
  readonly haltReason: DoryHaltReason | null;
  readonly deepTrace: DoryForwardTrace | null;
  readonly shallowCodaTrace: readonly DoryExecutedSublayerRecord[] | null;
}

export interface DorySelfSpecCycleInput {
  /** Contract A `per_loop`, `kLoops = 4`, `includeShallowCoda = true` (`104` banks) cache at `committedLen = L`. */
  readonly kvCache: DoryKvCache;
  /** Unconsumed emitted token `d_0` entering at position `L = kvCache.committedLen`. */
  readonly pendingToken: number;
  readonly params: DoryModelParameters;
  /** Draft horizon `gamma` (defaults to `config.speculativeGamma = 4`). */
  readonly gamma?: number;
  readonly sampling?: Partial<DorySamplingConfig>;
  /** Dedicated deterministic PRNG stream for draft/accept/residual/bonus sampling (zero ambient RNG). */
  readonly prng?: DoryPrng;
  /** Optional explicit uniform variates for Gate 2 Layer 3 & 4 deterministic testing. */
  readonly explicitVariates?: DorySpecExplicitVariates;
  readonly ropeTables?: DoryDualRopeTables;
  readonly sessionBranchId?: string;
  readonly forwardStepId?: number;
  readonly captureTrace?: boolean;
}

/**
 * Full execution result of one Contract A §6.2 self-speculative cycle (`Audit §8.1.1` Transitions 2–9).
 * Preserves the cycle-boundary invariant: all `104` banks end committed strictly through
 * `[0 .. endCommittedLen - 1]`, with `nextPendingToken` unconsumed (or `null` on accepted draft EOS).
 */
export interface DorySelfSpecCycleResult {
  readonly contractBadge: typeof DORY_SPEC_CONTRACT_A_BADGE;
  readonly startCommittedLen: number;
  readonly endCommittedLen: number;
  readonly inputPendingToken: number;
  /** Next unconsumed emitted token `d_0'` for the following cycle (or `null` on `full_accept_eos`). */
  readonly nextPendingToken: number | null;
  /** Effective draft budget `min(gamma, S_max - 1 - L)`. */
  readonly gammaEffective: number;
  /** Actual drafts proposed `gamma' <= gammaEffective` (may stop early on draft EOS). */
  readonly draftCount: number;
  readonly draftEndedWithEos: boolean;
  /** Number of consecutive draft tokens accepted (`0 <= a <= draftCount`). */
  readonly acceptedDraftCount: number;
  /** Tokens newly emitted in this cycle (`[d_1..d_a, y*]`, `[d_1..d_{gamma'}, y_bonus]`, `[d_1..d_{gamma'}]`, or `[y_final]`). */
  readonly emittedTokens: Int32Array;
  readonly outcome: DorySpecOutcomeKind;
  readonly haltReason: DoryHaltReason | null;
  /** Count of positions `[L .. L + draftCount - 1]` whose Prelude + Loop-1 KV & `(x, s_1)` were reused. */
  readonly reusedActivationPositions: number;
  /** Total positions evaluated in the batched `k = 4` verify pass (`draftCount`, `draftCount + 1`, or `1` on final slot). */
  readonly verifiedPositions: number;
  /** True iff 1-token Shallow Coda catch-up ran on cached `s_1^{(L + draftCount)}` (full non-EOS accept or final slot). */
  readonly shallowCodaCatchupExecuted: boolean;
  /** Speculative KV positions rewound atomically across all `104` banks via `cap_swa = 16` + `absPosTag`. */
  readonly rolledBackPositions: number;
  /** Per-step shallow `k = 1` draft records (`length === draftCount`). */
  readonly draftSteps: readonly DorySpecDraftStepRecord[];
  /** Per-position deep `k = 4` verify records (`length === verifiedPositions`). */
  readonly verifyPositions: readonly DorySpecVerifyPositionRecord[];
  /** Exact binary64 residual correction distribution `p_res` on rejection (`null` on full accept or final slot). */
  readonly residualDistribution: DoryResidualDistributionResult | null;
  /** Correction token `y*` (rejection), bonus token `y_bonus` (full non-EOS accept), or final-slot token (`null` on `full_accept_eos`). */
  readonly correctionOrBonusToken: number | null;
  /** Uniform variate used to sample `correctionOrBonusToken` (`null` in greedy mode or when none sampled). */
  readonly correctionOrBonusUniform: number | null;
  /** `[d]` Prelude anchor `x` for the final committed position `endCommittedLen - 1`. */
  readonly lastCommittedXAnchor: Float32Array;
  /** Length `4`: `[u_1..u_4]` (`[d]` each) for `endCommittedLen - 1`. */
  readonly lastCommittedLoopEntryStates: readonly Float32Array[];
  /** Length `4`: `[s_1..s_4]` (`[d]` each) for `endCommittedLen - 1`. */
  readonly lastCommittedLoopExitStates: readonly Float32Array[];
  /** Length `4`: exact 8D trajectory metrics for `endCommittedLen - 1`. */
  readonly lastCommittedLoopMetrics: readonly DoryLoopTrajectoryMetrics[];
  /** Deep `k = 4` scaled logits (`[V]`) at `endCommittedLen - 1`. */
  readonly lastCommittedDeepScaledLogits: Float32Array;
  /** Deep `k = 4` filtered binary64 probabilities (`[V]`) at `endCommittedLen - 1`. */
  readonly lastCommittedDeepProbs: Float64Array;
  /** Optional batched verify trace (`captureTrace === true`) for View 6 & lazy reconstruction. */
  readonly verifyTrace: DoryForwardTrace | null;
}

/** Lightweight scalar-only draft summary retained in `DorySessionRecord` (zero `[V]` array retention). */
export interface DorySpecDraftStepSummary {
  readonly draftOffset: number;
  readonly absPos: number;
  readonly inputTokenId: number;
  readonly proposedTokenId: number;
  readonly proposedIsEos: boolean;
  readonly proposedProb: number;
}

/** Lightweight scalar-only verify summary retained in `DorySessionRecord` (zero `[V]` array retention). */
export interface DorySpecVerifyPositionSummary {
  readonly verifyOffset: number;
  readonly absPos: number;
  readonly inputTokenId: number;
  readonly reusedPreludeAndLoop1: boolean;
  readonly targetProbForProposed: number | null;
  readonly rejection: DoryRejectionSampleResult | null;
}

/** Compact Contract A cycle summary retained in historical `DorySessionRecord`s. */
export interface DorySpecCycleSummary {
  readonly contractBadge: typeof DORY_SPEC_CONTRACT_A_BADGE;
  readonly startCommittedLen: number;
  readonly endCommittedLen: number;
  readonly inputPendingToken: number;
  readonly nextPendingToken: number | null;
  readonly gammaEffective: number;
  readonly draftCount: number;
  readonly draftEndedWithEos: boolean;
  readonly acceptedDraftCount: number;
  readonly emittedTokens: Int32Array;
  readonly outcome: DorySpecOutcomeKind;
  readonly haltReason: DoryHaltReason | null;
  readonly reusedActivationPositions: number;
  readonly verifiedPositions: number;
  readonly shallowCodaCatchupExecuted: boolean;
  readonly rolledBackPositions: number;
  readonly residualMassZ: number | null;
  readonly correctionOrBonusToken: number | null;
  readonly draftSteps: readonly DorySpecDraftStepSummary[];
  readonly verifyPositions: readonly DorySpecVerifyPositionSummary[];
}

// ============================================================================
// BOUNDED <= 12.00 MiB DORY SESSION & D10 LRU POOL INTERFACES (Slice 2B: `session.ts`)
// ============================================================================

export interface DorySessionBounds {
  /** Hard cap on live retained branches (default `12`). */
  readonly maxBranches: number;
  /** Hard cap on retained lightweight history records (default `128`). */
  readonly maxRecords: number;
  /** Hard `D10` byte budget for the LRU sparse-checkpoint pool (default `3,145,728 B` = `3.00 MiB`). */
  readonly maxCheckpointPoolBytes: number;
  /** Hard ceiling on total session retained `Set<ArrayBufferLike>` bytes (default `12,582,912 B` = `12.00 MiB`). */
  readonly maxSessionRetainedBytes: number;
  /** Sparse checkpoint stride in steps per branch (default `4`). */
  readonly checkpointStride: number;
}

export const DORY_DEFAULT_SESSION_BOUNDS: DorySessionBounds = Object.freeze({
  maxBranches: 12,
  maxRecords: 128,
  maxCheckpointPoolBytes: DORY_CHECKPOINT_LRU_POOL_BYTES,
  maxSessionRetainedBytes: DORY_MAX_SESSION_RETAINED_BYTES,
  checkpointStride: 4,
});

export interface DoryCheckpointLruEntryMeta {
  readonly checkpointKey: string;
  readonly branchId: string;
  readonly recordId: string;
  readonly committedLen: number;
  readonly chunkSizes: readonly number[];
  /** Exact `measureDoryKvCacheBytes(kvCache).totalUniqueBufferBytes` (payload + tag metadata). */
  readonly bytes: number;
  readonly lastUsedSeq: number;
}

export interface DoryCheckpointLruStats {
  readonly maxPoolBytes: number;
  readonly usedBytes: number;
  readonly entryCount: number;
  readonly hitCount: number;
  readonly missCount: number;
  readonly evictionCount: number;
  readonly entries: readonly DoryCheckpointLruEntryMeta[];
}

/**
 * Deduplicated `Set<ArrayBufferLike>` session memory audit (`D10`, `DR18`, `DR19`).
 * Accounts for ALL unique underlying `ArrayBuffer`s across live branch head KV caches (including
 * `absPosTag` & `writerLoopTag`), `D10` LRU sparse-checkpoint pool entries, and compact records.
 */
export interface DorySessionMemoryAudit {
  readonly branchCount: number;
  readonly recordCount: number;
  readonly checkpointCount: number;
  readonly branchKvBytes: number;
  readonly checkpointPoolBytes: number;
  readonly recordBytes: number;
  /** Total unique `ArrayBuffer` bytes retained by `DorySession` (excluding singleton model weights). */
  readonly totalSessionRetainedBytes: number;
  /** Singleton model weight bytes (`372,992 B`, never cloned). */
  readonly singletonWeightBytes: number;
  /** Deduplicated sum of `totalSessionRetainedBytes` and `singletonWeightBytes`. */
  readonly totalWithWeightsBytes: number;
  readonly maxCheckpointPoolBytes: number;
  readonly maxSessionRetainedBytes: number;
  readonly withinCheckpointPoolBudget: boolean;
  readonly withinSessionBudget: boolean;
  readonly withinTotalBudgetWithWeights: boolean;
  /** True iff zero `ArrayBuffer`s are aliased across any two branches or pooled checkpoints (`DR18`/`DR19`). */
  readonly nonAliasedSlabsVerified: boolean;
}

export interface DorySessionBranchConfig {
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  readonly includeShallowCoda: boolean;
  readonly speculativeGamma: number;
  readonly sampling: DorySamplingConfig;
  readonly seed: number;
}

/** Immutable public view of a session branch (never exposes mutable `DoryKvCache` handles). */
export interface DorySessionBranchSnapshot {
  readonly branchId: string;
  readonly label: string;
  readonly parentBranchId: string | null;
  readonly forkRecordId: string | null;
  readonly forkReason: DorySessionForkReason;
  readonly config: DorySessionBranchConfig;
  readonly promptTokens: Int32Array;
  /** Exact recorded cumulative forward chunk-size partition schedule (`DR24`). */
  readonly chunkSizes: readonly number[];
  /** Consumed & committed tokens residing in KV `[0 .. committedLen - 1]`. */
  readonly committedTokens: Int32Array;
  readonly committedLen: number;
  /** Next unconsumed emitted token `d_0`, or `null` when halted without pending token. */
  readonly pendingToken: number | null;
  /** Full emitted sequence (`committedTokens` + `pendingToken` if non-null). */
  readonly emittedTokens: Int32Array;
  readonly recordIds: readonly string[];
  readonly headRecordId: string;
  readonly haltReason: DoryHaltReason | null;
  readonly kvBankCount: number;
  /** Exact `measureDoryKvCacheBytes(branchKv).totalUniqueBufferBytes` (payload + tag metadata). */
  readonly kvBytes: number;
  readonly createdAtSeq: number;
  readonly lastActiveSeq: number;
}

/**
 * Compact immutable history record retained in `DorySession` (`<= 128` records).
 * NEVER retains full sublayer traces or aliased `DoryKvCache` slabs; full step traces are
 * reconstructed on demand at 0-ULP via `reconstructStepTrace(recordId)` replaying exact `chunkSizes[]`.
 */
export interface DorySessionRecord {
  readonly recordId: string;
  readonly branchId: string;
  readonly parentRecordId: string | null;
  readonly stepIndex: number;
  readonly forwardStepId: number;
  readonly kind: DorySessionRecordKind;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  readonly sampling: DorySamplingConfig;
  readonly startPos: number;
  readonly endPos: number;
  readonly committedLen: number;
  readonly pendingToken: number | null;
  /** Tokens consumed during this step / cycle (`[startPos .. endPos - 1]` or `[L .. L + a]`). */
  readonly inputTokens: Int32Array;
  /** Tokens newly emitted by this step / cycle. */
  readonly emittedTokens: Int32Array;
  /** Cumulative committed tokens `[0 .. committedLen - 1]` immediately after this record. */
  readonly committedSequence: Int32Array;
  /** Exact cumulative chunk-size partition schedule immediately BEFORE this record (`DR24`). */
  readonly chunkSizesBefore: readonly number[];
  /** Exact cumulative chunk-size partition schedule immediately AFTER this record (`DR24`). */
  readonly chunkSizesAfter: readonly number[];
  /** Compact `[V]` scaled logits of the final committed position (`256 B`). */
  readonly lastTokenScaledLogits: Float32Array;
  /** Compact `[V]` binary64 probabilities of the final committed position (`512 B`). */
  readonly lastTokenProbs: Float64Array;
  /** Compact `[d]` Prelude anchor `x` for the final committed position (`32 B`). */
  readonly lastTokenXAnchor: Float32Array;
  /** Compact `k` `[d]` loop-entry states `u_1..u_k` for the final committed position (`32*k B`). */
  readonly lastTokenLoopEntryStates: readonly Float32Array[];
  /** Compact `k` `[d]` loop-exit states `s_1..s_k` for the final committed position (`32*k B`). */
  readonly lastTokenLoopExitStates: readonly Float32Array[];
  /** Exact 8D loop trajectory metrics (`k` entries) for the final committed position. */
  readonly lastTokenLoopMetrics: readonly DoryLoopTrajectoryMetrics[];
  /** Compact Contract A cycle summary (`non-null` iff `kind === 'spec_cycle'`). */
  readonly specSummary: DorySpecCycleSummary | null;
  readonly haltReason: DoryHaltReason | null;
  readonly createdAtSeq: number;
}

/** On-demand 0-ULP lazy step-trace reconstruction result (not permanently retained by `DorySession`). */
export interface DoryReconstructedStepResult {
  readonly recordId: string;
  readonly branchId: string;
  readonly kind: DorySessionRecordKind;
  /** True iff reconstruction started from a pooled `D10` LRU checkpoint; false if replayed from branch root. */
  readonly usedCheckpoint: boolean;
  /** Record ID of the checkpoint hit (or `null` on cold replay from empty cache). */
  readonly checkpointHitRecordId: string | null;
  /** Exact chunk sizes replayed prior to executing the target step (`DR24`). */
  readonly replayedChunkSizes: readonly number[];
  /** Full traced forward chunk result (`non-null` for `'prefill' | 'chunk' | 'decode_step'`). */
  readonly forwardResult: DoryForwardChunkResult | null;
  /** Full traced Contract A prefill result (`non-null` for `'spec_prefill'`). */
  readonly specPrefillResult: DorySpecCyclePrefillResult | null;
  /** Full traced Contract A speculative cycle result (`non-null` for `'spec_cycle'`). */
  readonly specCycleResult: DorySelfSpecCycleResult | null;
  /** Non-aliased post-step `DoryKvCache` snapshot (`committedLen === record.committedLen`). */
  readonly postStepKvCache: DoryKvCache;
}

/**
 * View 2 read-only `Diagnostic Counterfactual` Coda evaluation at intermediate loop `t in 1..k`
 * evaluated strictly against committed historical Deep Coda KV (`writeKv: false`, zero KV mutation).
 */
export interface DoryCounterfactualCodaResult {
  readonly badge: typeof DORY_COUNTERFACTUAL_CODA_BADGE;
  readonly recordId: string;
  readonly loopIteration: number;
  readonly kLoops: number;
  readonly absTokenPos: number;
  readonly codaHidden: Float32Array;
  readonly rawLogits: Float32Array;
  readonly scaledLogits: Float32Array;
  readonly probs: Float64Array;
}

export interface DorySessionStartBranchOptions {
  readonly label?: string;
  readonly promptTokens?: ArrayLike<number>;
  /**
   * Optional explicit prefill chunk partition summing to `promptTokens.length`
   * (e.g. `[4, 8]` for `DR17`/`DR24` `last_loop` partition divergence & lazy replay).
   * Valid only when `executionMode === 'standard'`.
   */
  readonly prefillChunkSizes?: readonly number[];
  readonly kLoops?: number;
  readonly kvCacheMode?: DoryKvCacheMode;
  readonly executionMode?: DoryExecutionMode;
  readonly speculativeGamma?: number;
  readonly sampling?: Partial<DorySamplingConfig>;
  readonly seed?: number;
  /** Optional explicit uniform in `[0, 1)` for sampling the initial `pendingToken = d_0`. */
  readonly initialSampleUniform?: number;
}

export interface DorySessionCreateOptions {
  readonly params?: DoryModelParameters;
  readonly ropeTables?: DoryDualRopeTables;
  readonly bounds?: Partial<DorySessionBounds>;
  readonly initialBranch?: DorySessionStartBranchOptions;
  /** Defaults to `true` (starts an initial branch immediately); set `false` for empty session. */
  readonly autoStartInitialBranch?: boolean;
}

export interface DorySessionStepOptions {
  /** Optional explicit branch ID to step from its tip; defaults to `activeBranchId` (copy-on-step if rewound). */
  readonly branchId?: string;
  /** Optional explicit uniform in `[0, 1)` for standard decode token sampling. */
  readonly sampleUniform?: number;
  /** Optional explicit uniform variates for Contract A speculative cycle testing. */
  readonly specExplicitVariates?: DorySpecExplicitVariates;
}

export interface DorySessionAppendChunkOptions {
  readonly branchId?: string;
  readonly sampleUniform?: number;
}

export interface DorySessionForkOptions {
  readonly label?: string;
  readonly kLoops?: number;
  readonly kvCacheMode?: DoryKvCacheMode;
  readonly executionMode?: DoryExecutionMode;
  readonly speculativeGamma?: number;
  readonly sampling?: Partial<DorySamplingConfig>;
  readonly seed?: number;
  /**
   * Required (`true`) whenever `(kLoops, kvCacheMode, executionMode)` differs from the source
   * branch (`DR18` / `D6`): spawns a fresh re-prefilled branch from prompt tokens & seed with
   * non-aliased KV slabs without ever mutating the source branch.
   */
  readonly allowRePrefill?: boolean;
  /** Optional prefill chunk schedule when `allowRePrefill === true` and `executionMode === 'standard'`. */
  readonly prefillChunkSizes?: readonly number[];
}

export interface DorySessionSwitchConfigOptions {
  readonly branchId?: string;
  readonly label?: string;
  readonly kLoops?: number;
  readonly kvCacheMode?: DoryKvCacheMode;
  readonly executionMode?: DoryExecutionMode;
  readonly speculativeGamma?: number;
  readonly sampling?: Partial<DorySamplingConfig>;
  readonly seed?: number;
  readonly prefillChunkSizes?: readonly number[];
}

export interface DorySessionSnapshot {
  readonly modelWeightId: string;
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
  readonly bounds: DorySessionBounds;
  readonly branches: readonly DorySessionBranchSnapshot[];
  readonly recordCount: number;
  readonly checkpointStats: DoryCheckpointLruStats;
}

export interface DorySession {
  readonly params: DoryModelParameters;
  readonly ropeTables: DoryDualRopeTables;
  readonly bounds: DorySessionBounds;
  getSnapshot(): DorySessionSnapshot;
  getActiveBranch(): DorySessionBranchSnapshot | null;
  getActiveRecord(): DorySessionRecord | null;
  getBranch(branchId: string): DorySessionBranchSnapshot | null;
  getRecord(recordId: string): DorySessionRecord | null;
  listBranches(): readonly DorySessionBranchSnapshot[];
  listBranchRecords(branchId: string): readonly DorySessionRecord[];
  setActiveBranch(branchId: string): DorySessionBranchSnapshot;
  seekRecord(recordId: string): DorySessionRecord;
  /** Starts a fresh root branch with non-aliased KV slabs (`DR18`/`DR19`). */
  startBranch(options?: DorySessionStartBranchOptions): DorySessionRecord;
  /**
   * Appends an explicit token chunk to a `'standard'` branch (recording exact chunk size in
   * `chunkSizes[]` for `DR24` replay). Throws if `executionMode !== 'standard'` or capacity exceeded.
   */
  appendChunk(
    chunkTokens: ArrayLike<number>,
    options?: DorySessionAppendChunkOptions,
  ): DorySessionRecord;
  /**
   * Advances the target branch by one decode step (`'standard'`) or one Contract A cycle (`'self_spec'`).
   * Stepping the active branch while `activeRecordId` is rewound before its tip automatically performs
   * a non-aliased copy-on-step fork (`forkReason: 'cursor_fork'`), leaving the source branch immutable.
   * Throws `RangeError` if the target branch is already halted (`eos` / `capacity_exhausted`).
   */
  step(options?: DorySessionStepOptions): DorySessionRecord;
  /**
   * Forks an immutable branch from `fromRecordId`:
   *   - Compatible `(kLoops, kvCacheMode, executionMode)`: clones the reconstructed KV cache at
   *     `fromRecordId` via `cloneDoryKvCache` (strictly non-aliased `ArrayBuffer`s — `DR18`/`DR19`).
   *   - Incompatible `(kLoops, kvCacheMode, executionMode)`: throws `DorySessionImmutableConfigError`
   *     unless `allowRePrefill === true`, in which case spawns a clean re-prefilled branch from
   *     prompt tokens & seed without mutating the source branch (`D6`).
   */
  forkBranch(fromRecordId: string, options?: DorySessionForkOptions): DorySessionRecord;
  /**
   * Explicit `DR18` / `D6` mid-session depth/policy transition: NEVER mutates the source branch's
   * KV slabs; always forks/re-prefills a clean branch with non-aliased slabs from prompt tokens & seed.
   */
  switchDepthOrPolicy(options: DorySessionSwitchConfigOptions): DorySessionRecord;
  /**
   * Enforces `DR18`: throws `DorySessionImmutableConfigError` if `requested` conflicts with the
   * branch's immutable `(kLoops, kvCacheMode, executionMode)`.
   */
  assertCompatibleBranchConfig(
    branchId: string,
    requested: Partial<Pick<DorySessionBranchConfig, 'kLoops' | 'kvCacheMode' | 'executionMode'>>,
  ): void;
  /**
   * Reconstructs a strictly non-aliased 0-ULP post-step `DoryKvCache` at any historical `recordId`,
   * consulting the `D10` `3.00 MiB` LRU pool first and replaying exact recorded `chunkSizes[]` (`DR24`).
   */
  reconstructKvCacheAtRecord(recordId: string): DoryKvCache;
  /**
   * Lazily reconstructs the exact 0-ULP full traced forward pass or Contract A speculative cycle for
   * any historical `recordId`, replaying exact `chunkSizes[]` under `last_loop` (`DR24`).
   */
  reconstructStepTrace(recordId: string): DoryReconstructedStepResult;
  /**
   * View 2 read-only `Diagnostic Counterfactual` Coda evaluation at intermediate loop `t in 1..k`
   * against committed historical Deep Coda KV (`writeKv: false`, zero KV/session mutation).
   */
  evaluateCounterfactualIntermediateCoda(
    recordId: string,
    loopIteration: number,
  ): DoryCounterfactualCodaResult;
  getCheckpointLruStats(): DoryCheckpointLruStats;
  measureMemory(): DorySessionMemoryAudit;
  reset(options?: DorySessionStartBranchOptions): DorySessionRecord | null;
}

// ============================================================================
// FROZEN DORY2 SLICE 2A & SLICE 2B MODULE EXPORT CONTRACTS
// ============================================================================
//
// Slice 2A (`src/dory/speculative.ts`) MUST implement `DorySpeculativeModuleContract`.
// Slice 2B (`src/dory/session.ts`)     MUST implement `DorySessionModuleContract`.
// Once merged, `src/dory/index.ts` re-exports both modules cleanly.

export interface DorySpeculativeModuleContract {
  normalizeSamplingConfig(partial?: Partial<DorySamplingConfig>): DorySamplingConfig;
  logitsToSamplingProbsFloat64(
    scaledLogits: ArrayLike<number>,
    offset?: number,
    length?: number,
    sampling?: Partial<DorySamplingConfig>,
  ): Float64Array;
  sampleCategoricalFloat64(probs: Float64Array, uniformVariate: number): number;
  evaluateRejectionAcceptance(
    draftTokenId: number,
    targetProbs: Float64Array,
    proposalProbs: Float64Array,
    uniformVariate: number | null,
    isGreedy?: boolean,
  ): DoryRejectionSampleResult;
  computeRejectionResidualFloat64(
    targetProbs: Float64Array,
    proposalProbs: Float64Array,
  ): DoryResidualDistributionResult;
  sampleSpecSingleStep(
    targetProbs: Float64Array,
    proposalProbs: Float64Array,
    prng?: DoryPrng,
    explicitUniforms?: {
      readonly draftUniform?: number;
      readonly acceptUniform?: number;
      readonly residualUniform?: number;
    },
    isGreedy?: boolean,
  ): DorySpecSingleStepSampleResult;
  enumerateSpecSingleStepExact(
    targetProbs: Float64Array,
    proposalProbs: Float64Array,
  ): DorySpecAnalyticalMarginalResult;
  enumerateSpecTwoStepExact(
    targetP1: Float64Array,
    proposalQ1: Float64Array,
    targetP2ByToken1: readonly Float64Array[],
    proposalQ2ByToken1: readonly Float64Array[],
  ): DorySpecTwoStepAnalyticalResult;
  computePearsonChiSquared(
    observedCounts: ArrayLike<number>,
    expectedProbs: ArrayLike<number>,
    sampleCount: number,
  ): DoryPearsonChiSquaredResult;
  prefillDorySelfSpec(input: DorySpecCyclePrefillInput): DorySpecCyclePrefillResult;
  runDorySelfSpecCycle(input: DorySelfSpecCycleInput): DorySelfSpecCycleResult;
  summarizeSelfSpecCycle(result: DorySelfSpecCycleResult): DorySpecCycleSummary;
}

export interface DorySessionModuleContract {
  createDorySession(options?: DorySessionCreateOptions): DorySession;
  measureDorySessionMemory(session: DorySession): DorySessionMemoryAudit;
}
