/**
 * Discrepancy & Browser-Guard Ledger (`DR1–DR24`) and Locked Design Decisions (`D1–D10`)
 * for NVIDIA Nemotron-Labs-Dory1.0-4B (`src/dory/ledger.ts`).
 *
 * Implements `DoryLedgerModuleContract` strictly aligned with `DORY0_ARCHITECTURE_AUDIT.md`
 * (§1.3, §9, §14.2) and `DORY0_REVIEW_RESPONSE.md`.
 * 100% clean-room original TypeScript — zero DOM, WebGL, Node built-ins, or sibling imports.
 */

import type {
  DoryDesignDecisionEntry,
  DoryDiscrepancyEntry,
  DoryDiscrepancyId,
  DoryEpistemicTag,
} from './types';

function freezeTags(tags: readonly DoryEpistemicTag[]): readonly DoryEpistemicTag[] {
  return Object.freeze(tags);
}

function freezeIds(ids: readonly DoryDiscrepancyId[]): readonly DoryDiscrepancyId[] {
  return Object.freeze(ids);
}

// ============================================================================
// NUMBERED DISCREPANCY & BROWSER-GUARD LEDGER (`DR1–DR24`, AUDIT §9)
// ============================================================================

export const DORY_DISCREPANCY_LEDGER: readonly DoryDiscrepancyEntry[] = Object.freeze([
  Object.freeze({
    id: 'DR1',
    numericId: 1,
    category: 'spec_config',
    title: 'Paper TeX Numeric Omissions',
    summary:
      'Paper TeX uses symbolic d, d_ff, d_h only and omits numeric d=2560, d_ff=7168, d_h=256, V=131072, bias=false, and tie_word_embeddings=false.',
    provenance: 'dory_architecture.tex (symbolic only) vs. BF16/config.json:L6-24,L121-122',
    primaryTag: 'VERIFIED-CANONICAL',
    evidenceAxes: Object.freeze({
      origin: 'hf_checkpoint',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'locked_canonical',
    }),
    resolution:
      'Pin exact production dimensions directly from released BF16/config.json and safetensors headers (0 B diff).',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR2',
    numericId: 2,
    category: 'spec_config',
    title: 'Upstream DoryConfig Class Default Mismatches',
    summary:
      'DoryConfig() class defaults disagree with released config.json: vLLM defaults initializer_range=0.02, n_input=0, n_output=0, n_recurrent_loops=1, max_pos=8192, rope_theta=10000 (both); transformers defaults num_hidden_layers=36, max_pos=8192, rope_theta=10000 (both).',
    provenance:
      'vllm/.../configs/dory.py:L55-76; transformers/.../configuration_dory.py:L52-77 vs. BF16/config.json',
    primaryTag: 'VERIFIED-CANONICAL',
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'locked_canonical',
    }),
    resolution:
      'Never rely on upstream DoryConfig() defaults; explicitly pin released 0.0177, 16/40/16, 262144, and dual RoPE 10^4 / 10^7.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR3',
    numericId: 3,
    category: 'spec_config',
    title: 'Dropped attention_output_gate_logit_scale_init in transformers',
    summary:
      'transformers drops attention_output_gate_logit_scale_init and hardcodes hidden_size**0.25 / initializer_range, whereas vLLM and llama.cpp honor custom g_0 values (defaulting null -> sqrt(d)).',
    provenance:
      'transformers/.../modeling_dory.py:L274 vs. vllm/.../models/dory.py:L96-99 & llamacpp/conversion/dory.py:L32-36',
    primaryTag: 'IMPL-SPECIFIC',
    secondaryTags: freezeTags(['VERIFIED-CANONICAL']),
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'runtime_specific',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Honor attentionOutputGateLogitScaleInit (null -> sqrt(d)) matching vLLM and llama.cpp.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR4',
    numericId: 4,
    category: 'architecture_math',
    title: 'Physical 72-Sublayer vs. Paired 36-Layer Indexing',
    summary:
      'Checkpoint keys, Paper TeX, vLLM, and llama.cpp use 72 physical single-sublayer blocks (backbone.layers.0..71, 16+40+16), while transformers LayerPairCollapse and the HF model card caption pair them into 36 decoder layers (8+20+8).',
    provenance:
      'BF16/safetensors; pretraining.tex:L19,39; vllm/.../dory.py:L246-277 vs. transformers/.../configuration_dory.py:L85-92, core_model_loading.py:L1344-1370',
    primaryTag: 'IMPL-SPECIFIC',
    secondaryTags: freezeTags(['VERIFIED-CANONICAL']),
    evidenceAxes: Object.freeze({
      origin: 'hf_checkpoint',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Adopt physical 0..71 indexing as canonical throughout src/dory/ with Pair L{p >> 1} badges in inspectors (D1).',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR5',
    numericId: 5,
    category: 'architecture_math',
    title: 'Layer-0 Pre-Normalization Divergence',
    summary:
      'transformers applies input_layernorm before Layer 0 self_attn, whereas vLLM, llama.cpp, and Paper TikZ feed raw embeddings(ids) directly into Layer 0 q/k/v/g_proj.',
    provenance:
      'transformers/.../modeling_dory.py:L377 vs. vllm/.../dory.py:L261,266, dory.cpp:L143, dory_architecture.tex:L54-55',
    primaryTag: 'UNRESOLVED-DISCREPANCY',
    secondaryTags: freezeTags(['VERIFIED-CANONICAL', 'IMPL-SPECIFIC']),
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'runtime_specific',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Adopt vLLM/llama.cpp/TeX semantics (no Layer-0 pre-norm) and unit-normalize synthetic embedding rows so both variants agree to ~5e-7 (D5).',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR6',
    numericId: 6,
    category: 'architecture_math',
    title: 'RoPE vs. Q/K L2Norm Ordering',
    summary:
      'vLLM, llama.cpp, Paper TikZ, and dory-mlx apply RoPE before Q/K L2Norm; transformers applies Q/K L2Norm before RoPE. Exact real arithmetic is identical because RoPE is 2D block-orthogonal, with minor finite-precision drift.',
    provenance:
      'vllm/.../dory.py:L140-150, dory.cpp:L175-179, dory_architecture.tex:L110-119 vs. transformers/.../modeling_dory.py:L293-298',
    primaryTag: 'IMPL-SPECIFIC',
    secondaryTags: freezeTags(['VERIFIED-CANONICAL']),
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'runtime_specific',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Adopt RoPE -> per-head Q/K L2Norm order matching vLLM, llama.cpp, and Paper TikZ (D5).',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR7',
    numericId: 7,
    category: 'architecture_math',
    title: 'Anti-SLERP Residual Update Terminology & Extrapolation Guard',
    summary:
      'vLLM names its residual helper slerp_norm, risking confusion with trigonometric SLERP, but all Tier-1 implementations compute unclamped channel-wise NLERP N(a + |alpha_hat| * (b - a)) with |alpha_hat| >= 0.',
    provenance:
      'vllm/.../dory.py:L43-52; dory_architecture.tex:L166-185; transformers/.../modeling_dory.py:L386-392',
    primaryTag: 'VERIFIED-CANONICAL',
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'locked_canonical',
    }),
    resolution:
      'Label strictly as nGPT Channel-Wise Normalized Residual Interpolation (NLERP), forbid sin/arccos SLERP and [0, 1] clamping, and plant 2 channels with |alpha_hat| in (1.05, 1.25) at physical layer 16.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR8',
    numericId: 8,
    category: 'architecture_math',
    title: 'Literal Loop-Boundary N(2x) and Post-Loop N(s_t)',
    summary:
      'Paper, vLLM, and llama.cpp execute literal u_t = N(s_{t-1} + x) (so u_1 = N(2x)) and explicit post-loop s_t = N(F_theta(u_t)); transformers runs self.norm(h) twice at t=1 and omits post-loop self.norm before Coda.',
    provenance:
      'pretraining.tex:L41-43, vllm/.../dory.py:L267-276, dory.cpp:L156-161,192-194 vs. transformers/.../modeling_dory.py:L541-544',
    primaryTag: 'IMPL-SPECIFIC',
    secondaryTags: freezeTags(['VERIFIED-CANONICAL']),
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'runtime_specific',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Execute literal u_t = N(s_{t-1} + x) (never eliding N(2x)) and literal s_t = N(F_theta(u_t)) (D5).',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR9',
    numericId: 9,
    category: 'runtime_precision',
    title: 'FP32 vs. BF16 Scale, Gate, SwiGLU & Residual Compute Precision',
    summary:
      'vLLM and llama.cpp upcast all 5 scale families, sigmoid gate, SwiGLU, and NLERP residual blending to FP32; transformers leaves sqk, s_gate/sigmoid, suv/SiLU, and NLERP residual in native BF16.',
    provenance:
      'vllm/.../dory.py:L46-52,145-194,327-329; dory.cpp:L150-203 vs. transformers/.../modeling_dory.py:L299-392',
    primaryTag: 'IMPL-SPECIFIC',
    secondaryTags: freezeTags(['VERIFIED-CANONICAL']),
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'runtime_specific',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Enforce D5 browser contract: Float32Array (binary32) storage with JS native binary64 accumulation rounded strictly at named write-back boundaries, matching vLLM/llama.cpp FP32 semantics.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR10',
    numericId: 10,
    category: 'provenance_license',
    title: 'transformers PR #50993 Commit Title vs. Actual Diff',
    summary:
      'Commit 7ad60adc9e33 is titled "add update(..., write=False), use it for Dory last_loop (#50993)" but merged 0 lines in cache_utils.py and instead added DoryPastReader(DummyLayer) in modeling_dory.py.',
    provenance: 'transformers commit 7ad60adc9e33 diff (modeling_dory.py:L130-163)',
    primaryTag: 'VERIFIED-CANONICAL',
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'runtime_specific',
      resolutionStatus: 'documented_divergence',
    }),
    resolution:
      'Rely on verified DoryPastReader read-only past-cache semantics rather than the commit title.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR11',
    numericId: 11,
    category: 'kv_cache',
    title: 'llama.cpp Loop Cap (<= 4), Shift Disable & Pinned SHA Lag',
    summary:
      'llama.cpp PR #30306 hard-caps n_loops <= 4 and disables KV context shifting (get_can_shift() = false), while vLLM/transformers allow any n_recurrent_loops >= 1; the HF model card pins an older commit than PR #30306 head.',
    provenance:
      'dory.cpp:L30; llama-kv-cache.cpp:L1224; vllm/tests/models/test_dory.py:L135; BF16/README.md:L288',
    primaryTag: 'IMPL-SPECIFIC',
    secondaryTags: freezeTags(['VERIFIED-CANONICAL']),
    evidenceAxes: Object.freeze({
      origin: 'upstream_runtime',
      verificationMethod: 'direct_inspection',
      scope: 'runtime_specific',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Support k in {1, 2, 3, 4} primary plus optional k in {6, 8} extrapolation (and any integer k >= 1 in core math), with zero KV context shifting (D2, D10).',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR12',
    numericId: 12,
    category: 'architecture_math',
    title: 'Paper Figure 1 TikZ vs. Section 3.2 Symbol Collision',
    summary:
      'Fig. 1 TikZ calls the Prelude output u (s_0 = u), while Section 3.2 calls the Prelude output x, loop entry u_t = N(s_{t-1} + x), and loop exit state s_t.',
    provenance: 'dory_architecture.tex:L68,L73 vs. pretraining.tex:L39-43',
    primaryTag: 'UNRESOLVED-DISCREPANCY',
    evidenceAxes: Object.freeze({
      origin: 'paper_tex',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Adopt unambiguous Section 3.2 notation (x_anchor, u_t, s_t) consistently across code, traces, and UI.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR13',
    numericId: 13,
    category: 'provenance_license',
    title: 'Paper Section 6.2 Speedup Figure Caption Mismatch',
    summary:
      'Caption in inference.tex:L59 claims curves compare KV reuse with vs. without activation reuse, but dory_decode_speedup.pdf plots 1K Input Tokens vs. 16K Input Tokens.',
    provenance: 'sections/inference.tex:L59 vs. figures/dory_decode_speedup.pdf',
    primaryTag: 'UNRESOLVED-DISCREPANCY',
    evidenceAxes: Object.freeze({
      origin: 'paper_tex',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'documented_divergence',
    }),
    resolution:
      'Quote only the verified 1.37-1.71x speedup curves (1K and 16K input tokens) in Paper Results readouts.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR14',
    numericId: 14,
    category: 'runtime_precision',
    title: 'Paper "High Precision" Scales vs. On-Disk BF16 Storage',
    summary:
      'Paper Section 5 states all learned vectors and scalars remain in high precision, while all 3 HF safetensors checkpoints store all 217 scale vectors as BF16 on disk (upcast to FP32 at runtime; stored as F32 in GGUF Q5_K_M).',
    provenance:
      'quantization.tex:L11 vs. BF16/MXFP8/NVFP4 safetensors headers & conversion/dory.py:L13-19',
    primaryTag: 'VERIFIED-CANONICAL',
    evidenceAxes: Object.freeze({
      origin: 'hf_checkpoint',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'locked_canonical',
    }),
    resolution:
      'Document both explicitly: on-disk BF16 (unquantized relative to FP8/FP4) and runtime FP32 upcast.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR15',
    numericId: 15,
    category: 'runtime_precision',
    title: 'MXFP8 PTQ Scaling Text vs. Released config.json Observer',
    summary:
      'Paper Section 5 describes MXFP8 PTQ as using standard maximum scaling, whereas released MXFP8/config.json specifies observer="mse" with exact_e8m0_search_with_dominance_bounds.',
    provenance: 'quantization.tex:L17 vs. MXFP8/config.json:L323-335',
    primaryTag: 'UNRESOLVED-DISCREPANCY',
    evidenceAxes: Object.freeze({
      origin: 'hf_checkpoint',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'documented_divergence',
    }),
    resolution:
      'Report both facts accurately in the 4-checkpoint readout table, pinning released config.json MSE 2-candidate E8M0 search.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR16',
    numericId: 16,
    category: 'provenance_license',
    title: 'HF License Subcard Lag (OpenMDW-1.1 vs. NVIDIA Open Model License)',
    summary:
      'All 4 HF README.md cards and the paper declare OpenMDW-1.1, while BF16/explainability.md:L12 still references NVIDIA Open Model License Agreement.',
    provenance: 'BF16/README.md:L51,81 vs. BF16/explainability.md:L12',
    primaryTag: 'UNRESOLVED-DISCREPANCY',
    evidenceAxes: Object.freeze({
      origin: 'hf_checkpoint',
      verificationMethod: 'direct_inspection',
      scope: 'canonical_model',
      resolutionStatus: 'documented_divergence',
    }),
    resolution:
      'Treat OpenMDW-1.1 (YAML frontmatter, README, paper) as the governing weight license and note the stale subcard.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR17',
    numericId: 17,
    category: 'kv_cache',
    title: 'last_loop (k >= 2) Non-Invariance to Prefill Partitioning',
    summary:
      'For k >= 2, last_loop retains historical tokens final-loop KV so chunk boundaries alter historical KV provenance read by loops t < k; non-degenerate weights diverge materially across partitions.',
    provenance:
      'sections/inference.tex:L10; llama.cpp PR #30306; Audit §7.4 Theorems 1-3 & EXP 3 measurement',
    primaryTag: 'DERIVED',
    secondaryTags: freezeTags(['VERIFIED-CANONICAL']),
    evidenceAxes: Object.freeze({
      origin: 'browser_contract',
      verificationMethod: 'numerical_experiment',
      scope: 'canonical_model',
      resolutionStatus: 'locked_canonical',
    }),
    resolution:
      'Never assert last_loop === per_loop for k >= 2 after prefill; test each partition independently against PyTorch plus constructive > 1e-2 divergence fixture.',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR18',
    numericId: 18,
    category: 'browser_guard',
    title: 'Mid-Session Depth or Cache-Policy Switch Guard',
    summary:
      'Mutating k or switching per_loop <-> last_loop on a populated KV session corrupts missing loop banks or mixes incompatible KV depth provenances.',
    provenance: 'Audit §7.5 architectural invariant (D6)',
    primaryTag: 'DERIVED',
    evidenceAxes: Object.freeze({
      origin: 'browser_contract',
      verificationMethod: 'algebraic_derivation',
      scope: 'browser_miniature',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Depth or KV-policy changes always fork/re-prefill a fresh branch with non-aliased CoW slabs from prompt tokens and seed; never mutate populated KV slabs (D6).',
    isBrowserGuard: true,
  }),
  Object.freeze({
    id: 'DR19',
    numericId: 19,
    category: 'browser_guard',
    title: 'Self-Speculative SWA Eviction-Safe Rollback Guard',
    summary:
      'A W-slot SWA ring overwrites committed history during gamma=4 speculative verification; suffix cropping cannot restore evicted entries.',
    provenance: 'Audit §8.3 analysis; transformers/cache_utils.py:L288-293',
    primaryTag: 'DERIVED',
    evidenceAxes: Object.freeze({
      origin: 'browser_contract',
      verificationMethod: 'algebraic_derivation',
      scope: 'browser_miniature',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Allocate cap_swa = 16 >= W + gamma + 1 = 13 plus per-slot absPosTag for transaction-local O(1) rollback, with non-aliased CoW slabs on branch forks/checkpoints.',
    isBrowserGuard: true,
  }),
  Object.freeze({
    id: 'DR20',
    numericId: 20,
    category: 'browser_guard',
    title: 'SheetsPass Live RG32F G-Channel Mask Guard (sheets.ts:351)',
    summary:
      'SheetsPass falls back to geometric cellCol > cellRow whenever sheetKind in {1, 2} lacks a live RG32F TexRect (hasLiveTex === false), breaking SWA diagonal-band masks.',
    provenance: 'src/render/sheets.ts:L347-368',
    primaryTag: 'VERIFIED-CANONICAL',
    evidenceAxes: Object.freeze({
      origin: 'repo_audit',
      verificationMethod: 'direct_inspection',
      scope: 'browser_miniature',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Hard-throw in DorySheetBufferBuilder.push() and validateSceneGeometryFinite() if any SWA/Full causal sheet lacks a live RG32F G-channel mask.',
    isBrowserGuard: true,
  }),
  Object.freeze({
    id: 'DR21',
    numericId: 21,
    category: 'browser_guard',
    title: 'PickBox kvHeadIndex Typing & Shared-Renderer Isolation',
    summary:
      'PickBox.kvHeadIndex in src/render/pick.ts:26 is typed 0 | 1 for Nano; modifying pick.ts violates shared-renderer zero-edit isolation.',
    provenance: 'src/render/pick.ts:L26,L50',
    primaryTag: 'VERIFIED-CANONICAL',
    evidenceAxes: Object.freeze({
      origin: 'repo_audit',
      verificationMethod: 'direct_inspection',
      scope: 'browser_miniature',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Never edit src/render/pick.ts; encode loop, KV-head, and stage provenance via groupIndex, tapIndex, and semanticTag in src/dory/scene/provenance.ts.',
    isBrowserGuard: true,
  }),
  Object.freeze({
    id: 'DR22',
    numericId: 22,
    category: 'browser_guard',
    title: 'Gate 4b/4c Architectural Boundary Isolation & Nano Budget Guard',
    summary:
      'Importing Nano UI or non-hybrid core modules (ui/{controls,overlay,memoryChart}, scene/{spine,sheetInspector}, model/{spec,memory,state,rcPlayback}, math/*, story/walkthrough) fails Gate 4b/4c and threatens Nano 215.00 KiB gzip ceiling.',
    provenance: 'scripts/verify_bundle_isolation.mjs:L86-93; live 3ed28e2 dist/ audit',
    primaryTag: 'VERIFIED-CANONICAL',
    evidenceAxes: Object.freeze({
      origin: 'repo_audit',
      verificationMethod: 'direct_inspection',
      scope: 'browser_miniature',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Import leaf primitives solely from src/render/primitives.ts and shared helpers from src/shared/*; enforce nanoUI=0, nonHybridCore=0, and 0 sibling imports (D4).',
    isBrowserGuard: true,
  }),
  Object.freeze({
    id: 'DR23',
    numericId: 23,
    category: 'speculative',
    title: 'Paper Section 6.2 Self-Speculative Coda KV History Omission',
    summary:
      'Paper Section 6.2 specifies Prelude + Loop-1 KV and (x, s_1) activation reuse for k=1 -> k=4 self-speculation (absent from upstream transformers/vLLM/llama.cpp), while omitting how Output/Coda KV history is maintained across cycles.',
    provenance: 'sections/inference.tex:L50-52; Audit §8.1 & DORY0_REVIEW_RESPONSE Finding 1',
    primaryTag: 'UNRESOLVED-DISCREPANCY',
    secondaryTags: freezeTags(['DERIVED', 'IMPL-SPECIFIC']),
    evidenceAxes: Object.freeze({
      origin: 'paper_tex',
      verificationMethod: 'algebraic_derivation',
      scope: 'browser_miniature',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Adopt Contract A: 8 dedicated persistent output_shallow + 8 output_deep Coda KV banks (104 total, +20.00 KiB), explicitly badged "Reconstructed from Paper §6.2 (Contract A) · Not in Upstream Runtimes · per_loop Only" (D3).',
    isBrowserGuard: false,
  }),
  Object.freeze({
    id: 'DR24',
    numericId: 24,
    category: 'browser_guard',
    title: 'Long-Chunk SWA Workspace (L > 16) & last_loop Chunk-Schedule Replay',
    summary:
      'A persistent cap_swa=16 ring cannot hold an L > 16 chunk (up to S_max=32) simultaneously during attention; lazy trace reconstruction under last_loop (k >= 2) diverges if recorded chunk boundaries are coalesced.',
    provenance: 'Audit §7.1, §7.5, §8.3; DORY0_REVIEW_RESPONSE Finding 6',
    primaryTag: 'DERIVED',
    evidenceAxes: Object.freeze({
      origin: 'browser_contract',
      verificationMethod: 'algebraic_derivation',
      scope: 'browser_miniature',
      resolutionStatus: 'adjudicated_browser_contract',
    }),
    resolution:
      'Execute SWA attention over a transient [past_visible (<= W-1) || current_chunk (L)] workspace before committing the newest min(endPos, 16) slots; record and replay exact chunkSizes[] during last_loop lazy trace reconstruction.',
    isBrowserGuard: true,
  }),
]);

// ============================================================================
// LOCKED DESIGN DECISIONS (`D1–D10`, AUDIT §14.2)
// ============================================================================

export const DORY_DESIGN_DECISIONS: readonly DoryDesignDecisionEntry[] = Object.freeze([
  Object.freeze({
    id: 'D1',
    numericId: 1,
    title: 'Canonical Physical-72 Single-Sublayer Indexing',
    summary:
      'Adopt physical 0..71 sublayer indexing (16 Input + 40 Recurrent + 16 Output) with paired transformers correspondence badges (Pair L{p >> 1}) in inspectors.',
    status: 'APPROVED',
    relatedDiscrepancies: freezeIds(['DR4']),
  }),
  Object.freeze({
    id: 'D2',
    numericId: 2,
    title: 'Locked Miniature Geometry (d=8, d_ff=24, 72L, 507 Tensors)',
    summary:
      'Full 72 physical sublayers, 507 tensors / 93,248 F32 floats (0.356 MiB), d=8, d_ff=24, h_q=8, h_kv=2 (4:1 GQA), d_h=8 (q_dim=64, kv_dim=16), V=64 untied, S_max=32, W=8, cap_swa=16, dual RoPE (theta_S=16, theta_F=1024), k in {1..4} (+6, 8).',
    status: 'APPROVED',
    relatedDiscrepancies: freezeIds(['DR1', 'DR2', 'DR11']),
  }),
  Object.freeze({
    id: 'D3',
    numericId: 3,
    title: 'Contract A Self-Speculative Decoding (104 Persistent Banks)',
    summary:
      'Live deterministic k=1 -> k=4 (gamma=4, per_loop only) self-speculation with Prelude+Loop-1 reuse, separate persistent output_shallow & output_deep Coda KV (104 banks, +20.00 KiB), division-free binary64 rejection sampling, and 6-layer verification + 2 failing negative controls.',
    status: 'APPROVED_HARDENED',
    relatedDiscrepancies: freezeIds(['DR13', 'DR19', 'DR23']),
  }),
  Object.freeze({
    id: 'D4',
    numericId: 4,
    title: 'Strict Gate 4b/4c Architectural Boundary Isolation',
    summary:
      'Zero imports from Gate 4b/4c Nano modules or sibling apps; import leaf primitives solely from src/render/primitives.ts and shared helpers from src/shared/*, sharing only shared(all5) (atlasPacker).',
    status: 'APPROVED_HARDENED',
    relatedDiscrepancies: freezeIds(['DR20', 'DR21', 'DR22']),
  }),
  Object.freeze({
    id: 'D5',
    numericId: 5,
    title: 'Canonical Browser Numerical Contract (Binary32 Storage / Binary64 Accumulation)',
    summary:
      'Float32Array binary32 tensor storage with JS native binary64 accumulation & transcendentals rounded strictly at named tensor write-back boundaries (no per-op Math.fround()), fixed ascending 0..dim-1 reduction order, RoPE -> Q/K L2Norm, literal N(2x) & post-loop N(s_t), no Layer-0 pre-norm, unit-norm synthetic embeddings.',
    status: 'APPROVED_HARDENED',
    relatedDiscrepancies: freezeIds(['DR3', 'DR5', 'DR6', 'DR7', 'DR8', 'DR9']),
  }),
  Object.freeze({
    id: 'D6',
    numericId: 6,
    title: 'Immutable Fork / Re-Prefill on Depth or Cache-Policy Switch',
    summary:
      'Changing k or per_loop <-> last_loop on a populated session never mutates existing KV slabs; always forks a clean branch with non-aliased CoW slabs from prompt tokens and seed.',
    status: 'APPROVED',
    relatedDiscrepancies: freezeIds(['DR18']),
  }),
  Object.freeze({
    id: 'D7',
    numericId: 7,
    title: 'Seven Editorial WebGL2 Spatial Views (#F7F8FA)',
    summary:
      'Macro Folded/Unrolled, nGPT Hypersphere & Trajectory (Exact 2D Spanning-Subspace + Exploratory 2D PCA with 8D-only metrics & local 120-deg caveat), Gated GQA & Dual-RoPE, SwiGLU, Dual-Axis KV & last_loop, Self-Speculation Contract A, and Bookends + 4-Format/TBPTL Readouts.',
    status: 'APPROVED_HARDENED',
    relatedDiscrepancies: freezeIds(['DR7', 'DR12', 'DR14', 'DR15', 'DR17', 'DR23']),
  }),
  Object.freeze({
    id: 'D8',
    numericId: 8,
    title: 'Seven Gated Milestones (DORY1-DORY7)',
    summary:
      'Keep DORY6 standalone browser shell + dev-only Rollup/WebGL2 dry-run separate from DORY7 5-way MPA wiring, modelRegistry registration, 20-pair isolation, and full 5-app E2E.',
    status: 'APPROVED_HARDENED',
    relatedDiscrepancies: freezeIds(['DR22']),
  }),
  Object.freeze({
    id: 'D9',
    numericId: 9,
    title: 'CI/Deploy N-Way MPA Readiness (Resolved in 3ed28e2)',
    summary:
      '.github/workflows/{ci,deploy}.yml already trigger on feat/** and use N-way MPA wording on main (3ed28e2 / uni-v1.0.0); zero workflow edits required.',
    status: 'RESOLVED',
    relatedDiscrepancies: freezeIds([]),
  }),
  Object.freeze({
    id: 'D10',
    numericId: 10,
    title: 'Byte-Budgeted 3.00 MiB LRU Sparse-Checkpoint Pool (k = 1..8)',
    summary:
      'Cap sparse historical checkpoints with a 3.00 MiB (3,145,728 B) LRU pool (pinning active branch heads) so 12-branch / 128-record sessions stay strictly <= 12.00 MiB at both k=4 (<= 7.29 MiB) and k=8 (<= 9.60 MiB).',
    status: 'APPROVED_HARDENED',
    relatedDiscrepancies: freezeIds(['DR11', 'DR18', 'DR19']),
  }),
]);

// ============================================================================
// LOOKUP HELPERS
// ============================================================================

const DISCREPANCY_BY_ID: ReadonlyMap<DoryDiscrepancyId, DoryDiscrepancyEntry> = new Map(
  DORY_DISCREPANCY_LEDGER.map((entry) => [entry.id, entry]),
);

const BROWSER_GUARDS: readonly DoryDiscrepancyEntry[] = Object.freeze(
  DORY_DISCREPANCY_LEDGER.filter((entry) => entry.isBrowserGuard),
);

export function getDoryDiscrepancyById(id: DoryDiscrepancyId): DoryDiscrepancyEntry {
  const entry = DISCREPANCY_BY_ID.get(id);
  if (!entry) {
    throw new RangeError(`Unknown Dory discrepancy ID: ${String(id)}`);
  }
  return entry;
}

export function getDoryBrowserGuards(): readonly DoryDiscrepancyEntry[] {
  return BROWSER_GUARDS;
}
