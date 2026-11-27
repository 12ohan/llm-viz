/**
 * Milestone UNI6 — Slice 6A: 5-Act, 28-Beat Guided Pedagogical Tour
 * (`src/uni/story/walkthrough.ts`)
 *
 * Interactive 5-Act, 28-Beat pedagogical tour for Nemotron-Labs-Diffusion-Uni
 * (`UNI0_ARCHITECTURE_AUDIT.md` §§6, 8, 9):
 *   - Act I   (`act1`, beats `0..4`,   5 beats): Unified Multimodal Challenge & Elastic MoT
 *   - Act II  (`act2`, beats `5..9`,   5 beats): Dual Visual Pipelines — RADIO-H vs. Emu3 IBQ
 *   - Act III (`act3`, beats `10..15`, 6 beats): Joint vs. Split-Tower Attention & Coordinates
 *   - Act IV  (`act4`, beats `16..22`, 7 beats): Ball-Tree Geometry, Progressive Self-Refinement & Dual-CFG Editing
 *   - Act V   (`act5`, beats `23..27`, 5 beats): Understanding, Grounding & Cross-Paradigm OPD
 *
 * Isolation guarantee: `createUniGuidedTour` owns a strictly private `UniSession`
 * (`7` lazily cached deterministic demo branches, `<= 45` records total, well within
 * `maxBranches = 12`, `maxRecords = 128`, `<= 12.00 MiB`), so entering, stepping all
 * 28 beats, and exiting never mutates the user's Free-Explore `UniSession`.
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG, or sibling-app /
 * shared-control imports.
 */

import {
  createDefaultUniRuntimeBundle,
  type UniFullRuntimeBundle,
} from '../imageSampler';
import { buildUniSpatialScene } from '../scene/index';
import {
  DEFAULT_UNI_SPATIAL_FOCUS_STATE,
  type UniSpatialFocusState,
  type UniSpatialScene,
} from '../scene/types';
import {
  UniSession,
  type UniSessionStartBranchOptions,
} from '../session';
import {
  UNI_TOUR_TOTAL_ACTS,
  UNI_TOUR_TOTAL_BEATS,
  type UniFreeExploreRestoreState,
  type UniTourActId,
  type UniTourActSummary,
  type UniTourBeatSpec,
  type UniTourController,
  type UniTourDemoKey,
  type UniTourState,
} from './types';

// ============================================================================
// 5-ACT STRUCTURE (EXACT 5 ACTS · 5 + 5 + 6 + 7 + 5 = 28 BEATS)
// ============================================================================

export const UNI_TOUR_ACTS: readonly UniTourActSummary[] = Object.freeze([
  Object.freeze({
    id: 'act1',
    actNumber: 1,
    title: 'Unified Multimodal Challenge & Elastic MoT',
    shortTitle: 'Elastic MoT',
    subtitle:
      'Twin 34-layer U/G towers (18 joint L0..17 + 16 split L18..33), D5 HF-Faithful vs. Elastic-Pruned KV waste, and task-reactive dimming.',
    startBeatIndex: 0,
    beatCount: 5,
  }),
  Object.freeze({
    id: 'act2',
    actNumber: 2,
    title: 'Dual Visual Pipelines: RADIO-H vs. Emu3 IBQ',
    shortTitle: 'Visual Inputs',
    subtitle:
      '16×16 native vs. 16×8 panorama letterboxing, U18 border-lock mask (8/16 cells), continuous RADIO-H ViT + projector, and discrete Emu3 IBQ + 2×2 codec.',
    startBeatIndex: 5,
    beatCount: 5,
  }),
  Object.freeze({
    id: 'act3',
    actNumber: 3,
    title: 'Joint vs. Split-Tower Attention & Coordinates',
    shortTitle: 'Joint / Split',
    subtitle:
      'Routed U/G QKV/O & 3.5× SwiGLU, L17 joint → L18 split (U⊥G) decoupling, 1D dyadic YaRN + Llama-4 Q scale + 2D SinCos-G, and D5 prefix KV slices.',
    startBeatIndex: 10,
    beatCount: 6,
  }),
  Object.freeze({
    id: 'act4',
    actNumber: 4,
    title: 'Ball-Tree Geometry, Progressive Self-Refinement & Dual-CFG Editing',
    shortTitle: 'Tree & Edit',
    subtitle:
      '256-code depth-8 PCA ball-tree (N=4 funnel & N=1 MDM limit), U26 D=17 floor mismatch, 4×6 VQ canvas (U9/U10), and 3-branch Dual-CFG panorama crop-back.',
    startBeatIndex: 16,
    beatCount: 7,
  }),
  Object.freeze({
    id: 'act5',
    actNumber: 5,
    title: 'Understanding, Grounding & Cross-Paradigm OPD',
    shortTitle: 'VLM & OPD',
    subtitle:
      'FLARE B=4 block-diffusion understanding, <LOC_0..16> grounding (U27), Read-Only OPD 129-bucket tail_kl (U28), and continuous-flow matched_sigmas(0.9).',
    startBeatIndex: 23,
    beatCount: 5,
  }),
]);

// ============================================================================
// 28-BEAT PEDAGOGICAL TOUR SPECIFICATION (INDICES 0..27)
// ============================================================================

export const UNI_TOUR_BEATS: readonly UniTourBeatSpec[] = Object.freeze([
  // --------------------------------------------------------------------------
  // ACT I (Beats 0..4) — Unified Multimodal Challenge & Elastic MoT
  // --------------------------------------------------------------------------
  Object.freeze({
    index: 0,
    id: 'act1_twin_34l_elastic_mot',
    actId: 'act1',
    actNumber: 1,
    beatInAct: 1,
    title: 'Twin 34-Layer Elastic Mixture-of-Transformers',
    takeaway:
      'Understanding (U) and Generation (G) own 100% separate weights across all 34 layers inside one 18.27B (36.535 GB) checkpoint.',
    narrative:
      'Unifying visual understanding and image synthesis inside a single dense backbone forces gradients from semantic text prediction and discrete pixel modeling to fight over shared weights. NLD-Uni solves this with an Elastic Mixture-of-Transformers: parallel 34-layer U (mineral indigo) and G (emerald-teal) towers whose RMSNorm, 32Q/8KV GQA projections, and 3.5× SwiGLU MLPs share zero parameters—only the non-parametric YaRN RoPE inv_freq buffer is shared across all 1,557 BF16 checkpoint tensors.',
    equation:
      'P_ckpt = 18,267,552,262 params (36.535 GB / 34.026 GiB)  |  P_layer,1-branch = 218.11M',
    ledgerRefs: Object.freeze(['U1', 'U2', 'D1']),
    viewMode: 'macro',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    demoKey: 't2i_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      executionViewId: 'text_to_image',
      prefillMode: 'elastic_pruned',
      selectedLayerIndex: 0,
      hoveredPartId: 'joint_coupler_0',
    }),
    highlightPartId: 'joint_coupler_0',
    cameraOverride: Object.freeze({ azimuthDeg: 28, elevationDeg: 18 }),
  }),

  Object.freeze({
    index: 1,
    id: 'act1_18_joint_16_split_topology',
    actId: 'act1',
    actNumber: 1,
    beatInAct: 2,
    title: '18 Joint Layers (L0..17) + 16 Split Layers (L18..33)',
    takeaway:
      'Released code hardcodes dual_tower_layers = 18 (52.94% joint + 47.06% split), refining the paper’s "final half" shorthand (U5).',
    narrative:
      'While the paper describes decoupling "the final half" of layers, core.py:70-79 hardcodes dual_tower_layers = 18 (absent from config.json, U5). Layers L0..17 (18 layers, 3.926B per tower) are bridged by luminous joint-attention cross-couplers so G queries read U prompt semantics; at the L17 | L18 divider rail, layers L18..33 (16 layers, 3.490B per tower) split apart into strictly decoupled U⊥G towers.',
    equation:
      'L0..17: 18 Joint (split_tower = False, 52.94%)  |  L18..33: 16 Split (split_tower = True, 47.06%)',
    ledgerRefs: Object.freeze(['U5', 'D1']),
    viewMode: 'macro',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    demoKey: 't2i_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      executionViewId: 'text_to_image',
      prefillMode: 'elastic_pruned',
      selectedLayerIndex: 17,
      hoveredPartId: 'joint_coupler_17',
    }),
    highlightPartId: 'joint_coupler_17',
    cameraOverride: Object.freeze({ azimuthDeg: 22, elevationDeg: 14 }),
  }),

  Object.freeze({
    index: 2,
    id: 'act1_d5_hf_faithful_kv_waste',
    actId: 'act1',
    actNumber: 1,
    beatInAct: 3,
    title: 'D5 HF-Faithful Prefill: 16L Wasted U Prefix KV',
    takeaway:
      'Eager HF prefill runs all 34 U layers and caches L18..33 U KVs (47.1% T2I / 31.4% Edit) that Stage-2 drops unread (U6).',
    narrative:
      'In HF-Faithful mode (sampling.py:686), Stage-1 prompt prefill executes all 34 Understanding layers, highlighting U L18..33 in amber with protruding wasted-KV slabs. During Stage-2 image denoising, attention.py:81 sets drop_cached_understanding = True in split layers L18..33—meaning 16/34 = 47.1% of T2I prefix KV cache (and 32/102 = 31.4% across 3-branch Editing) plus 3.49B executed U parameters (16.85B eager vs 13.36B active) are computed and discarded.',
    equation:
      'KV_waste(T2I) = 16 / 34 = 47.1%  |  KV_waste(Edit, B_cfg = 3) = 32 / 102 = 31.4%',
    ledgerRefs: Object.freeze(['U6', 'D5']),
    viewMode: 'macro',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'hf_faithful',
    demoKey: 't2i_hf_faithful',
    targetStepOffset: 0,
    focusPatch: Object.freeze({
      executionViewId: 'text_to_image',
      prefillMode: 'hf_faithful',
      selectedLayerIndex: 18,
      hoveredPartId: 'u_layer_18',
    }),
    highlightPartId: 'u_layer_18',
    cameraOverride: Object.freeze({ azimuthDeg: -26, elevationDeg: 16 }),
  }),

  Object.freeze({
    index: 3,
    id: 'act1_d5_elastic_pruned_zero_ulp',
    actId: 'act1',
    actNumber: 1,
    beatInAct: 4,
    title: 'D5 Elastic-Pruned: 0 Wasted KV & 0-ULP Stage-2 Equivalence',
    takeaway:
      'Pruning U prefill at L17 (18L U + 34L G) eliminates 100% of split-layer U KV waste while matching Stage-2 outputs to 0 ULP.',
    narrative:
      'Switching the D5 toggle to Elastic-Pruned ghosts U L18..33 immediately during T2I and Editing prefill: zero split-layer U KV slabs are allocated, reducing the causal graph cleanly to 13.36B (T2I) or 14.24B (Edit). Because L18..33 G queries never read cached U keys, our PyTorch 2.15.0 parity suite verifies exact 0-ULP equivalence between Elastic-Pruned and eager 34-layer HF-Faithful Stage-2 trajectories.',
    equation:
      'Elastic-Pruned(T2I) = 18L U + 34L G (0 wasted KV)  ⇒  max|Stage2_pruned − Stage2_HF| = 0 ULP',
    ledgerRefs: Object.freeze(['U6', 'D5']),
    viewMode: 'macro',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    demoKey: 't2i_elastic',
    targetStepOffset: 0,
    focusPatch: Object.freeze({
      executionViewId: 'text_to_image',
      prefillMode: 'elastic_pruned',
      selectedLayerIndex: 18,
      hoveredPartId: 'u_layer_18',
    }),
    highlightPartId: 'u_layer_18',
    cameraOverride: Object.freeze({ azimuthDeg: -22, elevationDeg: 16 }),
  }),

  Object.freeze({
    index: 4,
    id: 'act1_task_reactive_tower_dimming',
    actId: 'act1',
    actNumber: 1,
    beatInAct: 5,
    title: 'Task-Reactive Tower Dimming Across Execution Views',
    takeaway:
      'Each task lights only its causal-graph towers—8.50B Text, 9.19B VLM, 13.36B T2I, 14.24B Edit, and 18.27B Interleaved (U7).',
    narrative:
      'NLD-Uni adapts its active compute graph by task: Understanding & Grounding dim the entire 34-layer G tower (9.19B active, matching SGLang’s selective loader; U29); T2I & Editing prune U L18..33 while lighting G L0..33; and Interleaved Mixed routing activates all 68 tower blocks (17.12B dual backbone / 18.27B full). Notice the explicit U7 honesty badge: core.py supports mixed U+G routing, while no released outer loop yet alternates multi-turn text and image decoding.',
    equation:
      'Active Params: 8.50B Text | 9.19B VLM | 13.36B T2I | 14.24B Edit | 18.27B Mixed (U7)',
    ledgerRefs: Object.freeze(['U6', 'U7', 'U29']),
    viewMode: 'macro',
    taskMode: 'visual_understanding',
    executionViewId: 'interleaved_mixed',
    prefillMode: 'elastic_pruned',
    demoKey: 'visual_understanding',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      executionViewId: 'interleaved_mixed',
      prefillMode: 'elastic_pruned',
      selectedLayerIndex: 17,
      hoveredPartId: 'diffusion_head',
    }),
    highlightPartId: 'diffusion_head',
    cameraOverride: Object.freeze({ azimuthDeg: 32, elevationDeg: 20 }),
  }),

  // --------------------------------------------------------------------------
  // ACT II (Beats 5..9) — Dual Visual Pipelines: RADIO-H vs. Emu3 IBQ
  // --------------------------------------------------------------------------
  Object.freeze({
    index: 5,
    id: 'act2_aspect_preserving_letterbox',
    actId: 'act2',
    actNumber: 2,
    beatInAct: 1,
    title: '16×16 Native vs. 16×8 Panorama Aspect-Preserving Letterbox',
    takeaway:
      'RADIO-H keeps native aspect ratios dynamically, whereas Emu3 IBQ square-pads 16×8 panoramas to 16×16 before VQ encoding.',
    narrative:
      'View 2 places both visual front-ends side-by-side on the non-square 16×8 harbor panorama preset (rendered via our dedicated 2D RGB canvas inset per D4, never stuffed into 2-channel RG32F). The continuous RADIO-H lane preserves native 16×8 geometry without multi-crop tiling, while the discrete Emu3 IBQ editing lane centers the panorama inside a 16×16 square via pad_to_square_and_resize—using exact even-dimension integer centering to eliminate upstream U30 rounding drift.',
    equation:
      'RADIO-H: native (16×8) → 4×2 patches  |  Emu3VQ: pad_to_square(16×8 → 16×16) → 4×4 VQ',
    ledgerRefs: Object.freeze(['U18', 'U30', 'D2', 'D4']),
    viewMode: 'visual_input',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: 0,
    focusPatch: Object.freeze({
      scenePresetId: 'harbor_panorama_16x8',
      gainPresetId: 'mini_default',
      includeDiscreteStream: true,
    }),
    highlightPartId: 'u_patches',
  }),

  Object.freeze({
    index: 6,
    id: 'act2_u18_border_lock_mask',
    actId: 'act2',
    actNumber: 2,
    beatInAct: 2,
    title: 'U18 Source-Padding Border-Lock Mask (8/16 Locked Cells)',
    takeaway:
      '8 of 16 VQ cells on the 16×8 letterboxed panorama lie on padding borders and are locked immutable across every edit round.',
    narrative:
      'When editing a non-square image inside a square VQ canvas, hallucinating content into the letterbox bars corrupts the final center-crop. computeSourcePaddingVqMask (padding.py:30-90, U18) marks the top and bottom letterbox rows (rows 0 and 3: 8/16 cells) with G = 1.0 on the 4×4 quantized code grid and hatched borders on the RGB inset, guaranteeing lock_source_padding_vq pins those source codes untouched at every denoising step.',
    equation:
      'lockMask[r, c] = 1 iff VQ cell (r, c) overlaps letterbox border  (8 / 16 cells locked on 16×8)',
    ledgerRefs: Object.freeze(['U18', 'D4']),
    viewMode: 'visual_input',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: 0,
    focusPatch: Object.freeze({
      scenePresetId: 'harbor_panorama_16x8',
      gainPresetId: 'mini_default',
      includeDiscreteStream: true,
    }),
    highlightPartId: 'g_vq_code_grid',
  }),

  Object.freeze({
    index: 7,
    id: 'act2_radio_h_continuous_vit_projector',
    actId: 'act2',
    actNumber: 2,
    beatInAct: 3,
    title: 'Continuous RADIO-H 4×4 Patches, ViT, 2×2 Projector & Literal "\\n" (U8)',
    takeaway:
      'RADIO-H strips prefix [CLS/REG] tokens, applies 2×2 SquaredReLU projection, and inserts literal "\\n" while encoder.image_newline sits dead.',
    narrative:
      'On the 16×16 sunlit cat preset, the continuous Understanding lane splits RGB pixels into 16 (4×4) CLIP-normalized patches, prepends 4 [CLS/REG] skip tokens (production: 4 CLS + 6 registers = 10), runs 2 ViT self-attention blocks, strips [:, 4:], and merges 2×2 spatial neighborhoods through RadioHProjector (RMSNorm → Linear → SquaredReLU → Linear × gain; D3 0.25 mini / 0.01 prod). fix_one_prompt expands -500 (<SPECIAL_23>) with literal "\\n" tokens—leaving encoder.image_newline unread in every forward pass (U8).',
    equation:
      'y_proj = g_radio · W_2(max(0, W_1 · RMSNorm(gather_{2×2}(ViT[:, 4:])))^2)',
    ledgerRefs: Object.freeze(['U8', 'D3', 'U25']),
    viewMode: 'visual_input',
    taskMode: 'visual_understanding',
    executionViewId: 'visual_understanding',
    prefillMode: 'elastic_pruned',
    demoKey: 'visual_understanding',
    targetStepOffset: 0,
    focusPatch: Object.freeze({
      scenePresetId: 'sunlit_cat_16x16',
      gainPresetId: 'mini_default',
      includeDiscreteStream: false,
    }),
    highlightPartId: 'u_projector_newline',
  }),

  Object.freeze({
    index: 8,
    id: 'act2_emu3_ibq_codebook_lookup',
    actId: 'act2',
    actNumber: 2,
    beatInAct: 4,
    title: 'Discrete Emu3 IBQ Encoder & Dot-Product Codebook Lookup (U14)',
    takeaway:
      'Mini Emu3VQ downsamples 16×16 → 4×4 latents via GroupNorm+SiLU+AttnBlock and quantizes with maximum inner product argmax(z @ E^T).',
    narrative:
      'Unlike standard VQ-VAEs that minimize Euclidean distance ||z − e||_2^2, Emu3/IBQ (vq.py:17-20, U14) quantizes encoder latents via raw maximum inner product argmax(z @ E^T) over the codebook (256 entries in miniature, 131,072 = 2^17 in production) to match IBQ softmax training. The RG32F IBQ score sheet visualizes the live dot-product landscape across winning and top competing codebook entries.',
    equation:
      'c_i = argmax_{k ∈ 0..V_vq−1} (z_i · E_k^T)   [IBQ maximum inner product, U14]',
    ledgerRefs: Object.freeze(['U14', 'D2']),
    viewMode: 'visual_input',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: 0,
    focusPatch: Object.freeze({
      scenePresetId: 'sunlit_cat_16x16',
      gainPresetId: 'mini_default',
      includeDiscreteStream: true,
    }),
    highlightPartId: 'g_ibq_scores',
  }),

  Object.freeze({
    index: 9,
    id: 'act2_packed_codec_and_ug_assembly',
    actId: 'act2',
    actNumber: 2,
    beatInAct: 5,
    title: 'PackedLinearImageCodec 2×2 Compression & Unified U+G Assembly',
    takeaway:
      '2×2 spatial packing shrinks 4×4 source VQ codes into 4 G tokens merged with RADIO-H (U) and instruction (U) tokens for L0.',
    narrative:
      'Omitted from paper prose (U13), PackedLinearImageCodec (bridge.py:7-82) applies per-child RMSNorm (eps = 1e-6), concatenates 2×2 neighborhoods, projects 32 → 8 (production: 16,384 → 4,096), applies the factorized D3 generation gain, and adds 2D SinCos coordinates (marker_column = False for source edits). Both streams converge into one unified [S, 8] sequence ribbon carrying per-token binary routing r ∈ {0 (U), 1 (G)} into layer L0.',
    equation:
      'h_src = s_gen · W_down(concat_{2×2}(RMSNorm(E_gen[c]))) + 0.0018 · SinCos2D(marker_col = False)',
    ledgerRefs: Object.freeze(['U13', 'D3']),
    viewMode: 'visual_input',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: 0,
    focusPatch: Object.freeze({
      scenePresetId: 'harbor_panorama_16x8',
      gainPresetId: 'mini_default',
      includeDiscreteStream: true,
    }),
    highlightPartId: 'joint_multimodal_seq',
  }),

  // --------------------------------------------------------------------------
  // ACT III (Beats 10..15) — Joint vs. Split-Tower Attention & Coordinates
  // --------------------------------------------------------------------------
  Object.freeze({
    index: 10,
    id: 'act3_routed_qkv_o_and_swiglu',
    actId: 'act3',
    actNumber: 3,
    beatInAct: 1,
    title: 'Routed U/G QKV/O Projections & 3.5× SwiGLU FFN',
    takeaway:
      'Mixed U+G editing prefill routes each token to its modality-specific 4:1 GQA matrices and 3.5× SwiGLU before scattering back.',
    narrative:
      'Inspecting layer L0 of the full editing branch (batchRow = 2, mixed U+G routing) reveals exact Elastic-MoT dispatch: U tokens flow through indigo input_layernorm, W_Q/K/V^U, W_O^U, and 3.5× SwiGLU_U (d=8 → d_ff=28), while G source-VQ tokens flow through parallel teal _gen matrices and index-copy scatter into shared sequence order. The provenance badge explicitly discloses our miniature 8 → 64/16 projection-width reversal so headDim = 8 retains all 4 YaRN planes.',
    equation:
      'H_norm = route_UG(H, r, Norm_U, Norm_G)  |  SwiGLU(x) = W_down(SiLU(W_gate x) ⊙ W_up x),  d_ff = 3.5d',
    ledgerRefs: Object.freeze(['D1', 'U5']),
    viewMode: 'joint_attn',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'hf_faithful',
    demoKey: 'edit_panorama_hf',
    targetStepOffset: 0,
    focusPatch: Object.freeze({
      jointLayerIndex: 0,
      batchRow: 2,
      prefillMode: 'hf_faithful',
      selectedQHead: 0,
      selectedTokenRow: 0,
    }),
    highlightPartId: 'w_q_g',
  }),

  Object.freeze({
    index: 11,
    id: 'act3_l17_joint_cross_modal_attention',
    actId: 'act3',
    actNumber: 3,
    beatInAct: 2,
    title: 'Layer 17: Joint Cross-Modal Attention (G = 0 Across Modalities)',
    takeaway:
      'At L17 (split_tower = False), target G tokens attend freely (G = 0) across cached RADIO-H, source VQ, and text instruction K/Vs.',
    narrative:
      'In the final joint layer L17 during Editing Stage-2 denoising, the live RG32F attention plates (strictly U25-compliant with live G-channel masking) show G = 0 across the entire prefix and current target window: every target generation query absorbs continuous RADIO-H semantics (U), discrete source-VQ structure (G), and text edit instructions (U) under the asymmetric SBD window (U15).',
    equation:
      'L17 (split_tower = False): eff_mask(q, k) = base_mask(q, k)   [G = 0 across U and G keys]',
    ledgerRefs: Object.freeze(['U5', 'U15', 'U25']),
    viewMode: 'joint_attn',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'hf_faithful',
    demoKey: 'edit_panorama_hf',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      jointLayerIndex: 17,
      batchRow: 2,
      prefillMode: 'hf_faithful',
      selectedQHead: 0,
      selectedTokenRow: 4,
    }),
    highlightPartId: 'softmax_selected',
  }),

  Object.freeze({
    index: 12,
    id: 'act3_l18_split_tower_isolation',
    actId: 'act3',
    actNumber: 3,
    beatInAct: 3,
    title: 'Layer 18: Split-Tower Isolation (U ⊥ G, G = 1 Across Modalities)',
    takeaway:
      'Advancing L17 → L18 masks out all cached U keys (G = 1.0): target G queries attend strictly to cached source-VQ G keys and themselves.',
    narrative:
      'Crossing the elastic boundary into L18 (split_tower = True) enforces eff_mask &= (r_q == r_k). Watch the RG32F score and softmax plates partition instantly: every cached U prefix column (RADIO-H + instruction) turns masked (G = 1.0), leaving target G tokens attending solely to the 4 cached is_gen_enc source-VQ tokens (r_kv = 1) plus the bidirectional target block.',
    equation:
      'L18..33 (split_tower = True): eff_mask(q, k) = base_mask(q, k) ∧ (r_q == r_k)   [U ⊥ G]',
    ledgerRefs: Object.freeze(['U5', 'U15', 'U25']),
    viewMode: 'joint_attn',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'hf_faithful',
    demoKey: 'edit_panorama_hf',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      jointLayerIndex: 18,
      batchRow: 2,
      prefillMode: 'hf_faithful',
      selectedQHead: 0,
      selectedTokenRow: 4,
    }),
    highlightPartId: 'scores_selected',
  }),

  Object.freeze({
    index: 13,
    id: 'act3_1d_yarn_rope_and_llama4_scale',
    actId: 'act3',
    actNumber: 3,
    beatInAct: 4,
    title: '1D Dyadic YaRN RoPE & Llama-4 Query Temperature Scale',
    takeaway:
      'All U and G tokens share 1D YaRN (inv_freq = [1, 0.28125, 0.03125, 0.015625]) and Llama-4 Q scaling (1.000 → 1.069 at pos ≥ 16).',
    narrative:
      'Both towers share the exact Ministral-3 1D YaRN rotary schedule (factor = 8.0, miniature ropeTheta = 16, origMaxPos = 16 giving exact 0-ULP dyadic frequencies [1, 9/32, 1/32, 1/64] spanning original, blended, and interpolated zones). After rotating Q and K, only query_states are scaled by Llama-4’s cache-position temperature σ(c_t): positions 0..15 stay at 1.000 while this Stage-2 token (pos ≥ 16) activates 1.069 live (U3 documents the 16K verified vs 128K configured context ceiling).',
    equation:
      'inv_freq = [1, 0.28125, 0.03125, 0.015625]  |  σ_Llama4(c_t) = 1 + 0.1·ln(1 + ⌊c_t/16⌋) ∈ {1.000, 1.069}',
    ledgerRefs: Object.freeze(['D1', 'U3']),
    viewMode: 'joint_attn',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      jointLayerIndex: 17,
      batchRow: 2,
      prefillMode: 'elastic_pruned',
      selectedQHead: 0,
      selectedTokenRow: 9,
    }),
    highlightPartId: 'rope_llama4_table',
  }),

  Object.freeze({
    index: 14,
    id: 'act3_2d_sincos_g_coordinates',
    actId: 'act3',
    actNumber: 3,
    beatInAct: 5,
    title: 'Additive 2D SinCos-G Spatial Coordinates (marker_column = !isSourceEdit)',
    takeaway:
      '2D spatial geometry is added once at G input embedding (canonicalContentSize = 32, scale = 0.0018), aligning source & target columns.',
    narrative:
      'Instead of modifying RoPE into rotary M-RoPE, NLD-Uni injects 2D spatial coordinates once at G embedding time (positions.py:31-74, U13): canonicalContentSize = 32, scale = 0.0018, with marker_column = !isSourceEdit. Because target generation subtracts the +1 packed marker column (W_content = W − 1), content column strides match source-edit grids bit-for-bit on every content coordinate while 1D YaRN RoPE still orders the flat sequence.',
    equation:
      'c̃ = c · (32 / W_content),  r̃ = r · (32 / H),  scale = 0.0018  |  marker_column = !isSourceEdit',
    ledgerRefs: Object.freeze(['U13', 'D1']),
    viewMode: 'joint_attn',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      jointLayerIndex: 18,
      batchRow: 2,
      prefillMode: 'elastic_pruned',
      selectedQHead: 0,
      selectedTokenRow: 4,
    }),
    highlightPartId: 'hidden_in',
  }),

  Object.freeze({
    index: 15,
    id: 'act3_d5_dropped_prefix_kv_slices',
    actId: 'act3',
    actNumber: 3,
    beatInAct: 6,
    title: 'D5 Prefix-KV Cutaway: Dead L18..33 U Slices vs. Elastic-Pruned',
    takeaway:
      'In T2I at L18, HF-Faithful holds dead U prefix KVs ignored by every G query; Elastic-Pruned omits them with 0-ULP output diff.',
    narrative:
      'Switching to T2I at split layer L18 exposes the D5 waste directly inside the attention cutaway: because T2I’s prefix contains zero G tokens (cached_has_generation = False), every HF-Faithful L18..33 U prefix KV slice sits dead behind a wall of G = 1 masks. Under Elastic-Pruned, those 16 U prefix KV layers are never allocated, yet the attention output, O-proj, and SwiGLU residual match HF-Faithful to 0 ULP.',
    equation:
      'T2I L18..33: cached_has_generation = False ⇒ drop_cached_understanding = True (0-ULP prune)',
    ledgerRefs: Object.freeze(['U6', 'D5', 'U25']),
    viewMode: 'joint_attn',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'hf_faithful',
    demoKey: 't2i_hf_faithful',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      jointLayerIndex: 18,
      batchRow: 1,
      prefillMode: 'hf_faithful',
      selectedQHead: 0,
      selectedTokenRow: 4,
    }),
    highlightPartId: 'd5_prefix_kv',
  }),

  // --------------------------------------------------------------------------
  // ACT IV (Beats 16..22) — Ball-Tree Geometry, Self-Refinement & Dual-CFG Edit
  // --------------------------------------------------------------------------
  Object.freeze({
    index: 16,
    id: 'act4_ball_tree_pca_n4_funnel',
    actId: 'act4',
    actNumber: 4,
    beatInAct: 1,
    title: '256-Code Depth-8 PCA Ball-Tree & N = 4 Refinement Funnel',
    takeaway:
      'BalancedSemanticTree splits 256 VQ codes by recursive centered PCA into nested balls shrinking 256 → 64 → 16 → 4 → 1 at N = 4.',
    narrative:
      'View 4 projects all 256 Emu3 VQ codebook vectors into 3D PCA space, colored by leaf order 0..255 and wrapped in wireframe shells from the 255 monotone heap_radii (recursive_centered_pca_rank_median_v1). At checkpoint default refinement_steps = 4 (U11; paper claimed N = 8), each code narrows monotonically through power-of-two subtrees—256 → 64 → 16 → 4 → 1—sampled in O(1) via leaf-index bit shifts while allowing unconstrained self-correction across coarse commits.',
    equation:
      'ℓ(a) = ⌊a · D / N⌋,  |N_ℓ(ĉ)| = 2^{D − ℓ}  →  [256, 64, 16, 4, 1]  (D = 8, N = 4)',
    ledgerRefs: Object.freeze(['U11', 'U14', 'D2']),
    viewMode: 'ball_tree',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    demoKey: 't2i_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      selectedCodeId: 42,
      refinementSteps: 4,
      stageIndex: 2,
      showProductionD17Comparison: false,
    }),
    highlightPartId: 'tree_ring_4',
  }),

  Object.freeze({
    index: 17,
    id: 'act4_n1_binary_mdm_limit',
    actId: 'act4',
    actNumber: 4,
    beatInAct: 2,
    title: 'N = 1 Limit: Exact Reduction to Binary Masked Diffusion (MDM)',
    takeaway:
      'Setting refinement_steps = 1 collapses the ball hierarchy straight from 256 → 1 in one step, recovering standard binary MDM.',
    narrative:
      'When N = 1, every unactivated token sits at the root ball (|N_0| = 256, η = 1.0) and commits directly to a singleton leaf (|N_8| = 1, η = 0.0) in one pass with N − 1 = 0 flush rounds. Progressive self-refining diffusion therefore strictly generalizes binary Masked Diffusion Modeling: increasing N ∈ {1, 2, 4, 8} trades extra coarse-to-fine revisable passes for smoother semantic trajectories.',
    equation:
      'N = 1: ℓ(0) = 0 (|N| = 256, η = 1)  →  ℓ(1) = D (|N| = 1, η = 0)  ≡  Binary MDM',
    ledgerRefs: Object.freeze(['U11', 'D2']),
    viewMode: 'ball_tree',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    demoKey: 't2i_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      selectedCodeId: 42,
      refinementSteps: 1,
      stageIndex: 0,
      showProductionD17Comparison: false,
    }),
    highlightPartId: 'tree_ring_0',
  }),

  Object.freeze({
    index: 18,
    id: 'act4_u26_production_d17_floor_mismatch',
    actId: 'act4',
    actNumber: 4,
    beatInAct: 3,
    title: 'U26 Production D = 17 Prime-Depth Floor-Rounding Mismatch',
    takeaway:
      'Because D = 17 is prime, production inference [12, 8, 4] draws from 2× wider subtrees than training [13, 9, 5] at every internal stage.',
    narrative:
      'Enabling the U26 comparison reveals a subtle verified bug in production’s prime depth D = 17: minimal_training/corruption.py:58 computes train split levels 17 − ⌊a_trn · 17 / 4⌋ = [13, 9, 5] (ball sizes 16, 256, 4096), whereas tree.py:243 computes inference split levels ⌊a_inf · 17 / 4⌋ = [12, 8, 4] (ball sizes 32, 512, 8192)—2× wider candidate subtrees at 3/3 internal stages for K = 4 (7/7 for K = 8, 15/15 for K = 16). Our divisible D = 8 miniature exhibits 0 mismatch.',
    equation:
      'D = 17, K = 4: train 17 − ⌊a·17/4⌋ = [13, 9, 5]  ≠  infer ⌊a·17/4⌋ = [12, 8, 4]  (2× wider subtrees)',
    ledgerRefs: Object.freeze(['U26', 'U11']),
    viewMode: 'ball_tree',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    demoKey: 't2i_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      selectedCodeId: 42,
      refinementSteps: 4,
      stageIndex: 2,
      showProductionD17Comparison: true,
    }),
    highlightPartId: 'u26_stage_1',
  }),

  Object.freeze({
    index: 19,
    id: 'act4_refinement_canvas_u9_u10',
    actId: 'act4',
    actNumber: 4,
    beatInAct: 4,
    title: '4×6 Marked VQ Canvas: Stage/Age Polarity (U9) & Additive Mask Noise (U10)',
    takeaway:
      'Inference reverses paper age polarity (stage -1 → N clean) and injects continuous additive mask noise e(u) + η·e(MASK) with zero AdaLN.',
    narrative:
      'View 5 inspects the live 4×6 marked VQ child canvas (4×4 content + 2 structural row/end marker columns) mid-trajectory. Two critical paper-vs-code findings sit side-by-side: U9 reverses age polarity (unstarted stage = -1, refining 0..N-1, finalized stage = N clean vs paper a = N − max(0, stage)), and U10 adds continuous mask-noise weight η_i = 1 − max(stage_i, 0)/N times E_gen[IMAGE_MASK] directly in embedding space—the sole noise-level signal, since call_gen_predictor ignores timesteps with zero AdaLN gates (U17).',
    equation:
      'a_paper = N − max(0, stage)  (U9)  |  ẽ_i = E_gen[u_i] + (1 − max(stage_i, 0)/N) · E_gen[MASK]  (U10)',
    ledgerRefs: Object.freeze(['U9', 'U10', 'U17']),
    viewMode: 'refinement',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    demoKey: 't2i_elastic',
    targetStepOffset: 2,
    focusPatch: Object.freeze({
      selectedGridCell: 0,
      showStratifiedOrder: true,
      cfgBranchRow: 1,
    }),
    highlightPartId: 'vq_mask_noise',
  }),

  Object.freeze({
    index: 20,
    id: 'act4_stratified_order_codec_brackets',
    actId: 'act4',
    actNumber: 4,
    beatInAct: 5,
    title: 'Stratified Quadtree Order (4, 42), 2×2 Codec Brackets & Flush Progression',
    takeaway:
      'Deterministic stratified_random(4, 42) + shift=5.0 quotas + N-1 flush rounds couple 24 child cells to 6 parents + 4 fixed prefix tokens.',
    narrative:
      'Advancing to the final flush round shows how every content cell reached stage = N (immutable once committed; edit_threshold = -1). Rather than stochastic τ_i intervals, released inference uses flipped shift = 5.0 quotas, hardcoded stratified_random(4, seed=42) quadtree spatial ordering, and N − 1 zero-activation flush rounds (U12, U23). 2×2 PackedLinearImageCodec brackets bridge the 24 child cells into 6 (2×3) parent tokens prepended by 4 fixed-prefix tokens (10 G tokens total, U13), with overflow-safe log-domain Gumbel proposals (U22) and U30 B=1 short-circuit when w_txt = 0.',
    equation:
      '24 marked cells (4×6) → 6 parents (2×3) + 4 fixed prefix = 10 G tokens  |  rounds = S_act + (N − 1)',
    ledgerRefs: Object.freeze(['U12', 'U13', 'U22', 'U23', 'U30']),
    viewMode: 'refinement',
    taskMode: 'text_to_image',
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    demoKey: 't2i_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      selectedGridCell: 5,
      showStratifiedOrder: true,
      cfgBranchRow: 1,
    }),
    highlightPartId: 'parent_tokens',
  }),

  Object.freeze({
    index: 21,
    id: 'act4_dual_cfg_3branch_editing',
    actId: 'act4',
    actNumber: 4,
    beatInAct: 6,
    title: 'Dual-CFG 3-Branch Editing: legacy vs. independent vs. ip2p (U16)',
    takeaway:
      'Row 1 drops only discrete source VQ (keeping RADIO-H!), and default legacy mode averages before extrapolating text guidance.',
    narrative:
      'View 6 enters Image Editing mode on the 3-branch Dual-CFG batch [uncond, vq_dropped, full]. Contrary to standard InstructPix2Pix extrapolation, default legacy mode (edit_mode = 1, w_txt = 1.6, w_img = 2.0, U16) keeps continuous RADIO-H active in row 1, forms the weighted average (full + w_img · vq_dropped) / (1 + w_img), and then extrapolates away from uncond. Live RG32F top-32 logit plates compare legacy, independent, and ip2p side-by-side from exact 0-ULP reconstructed branch logits.',
    equation:
      'legacy (U16): ℓ_cond = (ℓ_full + w_img·ℓ_vq_drop)/(1 + w_img),   ℓ_guided = (1 + w_txt)ℓ_cond − w_txt·ℓ_unc',
    ledgerRefs: Object.freeze(['U16', 'D4']),
    viewMode: 'output_edit',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      editGuidanceMode: 'legacy',
      cropToSource: true,
      showPaddingLockOverlay: true,
    }),
    highlightPartId: 'cfg_mixer_legacy',
  }),

  Object.freeze({
    index: 22,
    id: 'act4_u18_lock_and_panorama_crop_back',
    actId: 'act4',
    actNumber: 4,
    beatInAct: 7,
    title: 'U18 Border-Lock Verification & 16×8 Panorama Crop-Back',
    takeaway:
      'After expanding parents and stripping +2 row markers, Emu3VQ decodes 16×16 RGB with 8/16 locked border cells before cropping back to 16×8.',
    narrative:
      'At the end of the editing pipeline, PackedLinearImageCodec.expand unpacks the 6 parent states back to 24 child logits, gen_predictor scores all 256 VQ codes, stripGenerationRowMarkers removes the +2 marker columns, and MiniEmu3VQ.decode reconstructs the 16×16 RGB preview (badged "Miniature 16×16 Mechanics Sim"). The U18 lock plate confirms all 8/16 padding cells remained locked to source VQ codes before cropPaddedImageToSource restores the exact native 16×8 panorama aspect.',
    equation:
      'expand(6 → 24) → gen_predictor → strip(+2 cols → 16) → Emu3VQ.decode(16×16) → crop(16×8)',
    ledgerRefs: Object.freeze(['U18', 'U30', 'D4']),
    viewMode: 'output_edit',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      editGuidanceMode: 'ip2p',
      cropToSource: true,
      showPaddingLockOverlay: true,
    }),
    highlightPartId: 'u18_padding_lock_mask',
  }),

  // --------------------------------------------------------------------------
  // ACT V (Beats 23..27) — Understanding, Grounding & Cross-Paradigm OPD
  // --------------------------------------------------------------------------
  Object.freeze({
    index: 23,
    id: 'act5_flare_block_diffusion_understanding',
    actId: 'act5',
    actNumber: 5,
    beatInAct: 1,
    title: 'FLARE Block-Diffusion Understanding (B = 4, shift_logits = true)',
    takeaway:
      'Understanding runs pure U block diffusion (0 LoRA/speculative paths, U19), pre-seeding slot 0 from the prior causal tail.',
    narrative:
      'View 6 switches reactively to Visual Understanding: the 34-layer U tower (stateless forward, U20; zero LoRA or linear speculative modes, U19) unmasks B = 4 token blocks via generate_with_prefix_cache_block_diff. Under FLARE shifted logits (shift_logits = true), slot 0 is predicted from the prior causal context tail output.logits[:, -1, :], high-confidence tokens (p ≥ 0.9) unmask in parallel with our U21 [MASK]-suppression guard, and finished blocks commit causally into the KV cache.',
    equation:
      'ℓ_shifted[0] = ℓ_ctx_tail,  ℓ_shifted[1..B−1] = ℓ_block[0..B−2]  |  commit KV causally on block end',
    ledgerRefs: Object.freeze(['U19', 'U20', 'U21', 'U25']),
    viewMode: 'output_edit',
    taskMode: 'visual_understanding',
    executionViewId: 'visual_understanding',
    prefillMode: 'elastic_pruned',
    demoKey: 'visual_understanding',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      cropToSource: false,
      showPaddingLockOverlay: false,
    }),
    highlightPartId: 'shifted_diffusion_probs',
  }),

  Object.freeze({
    index: 24,
    id: 'act5_loc_grounding_and_u27_hazard',
    actId: 'act5',
    actNumber: 5,
    beatInAct: 2,
    title: '<LOC_0..16> 2D Object Grounding & U27 skip_special_tokens Hazard',
    takeaway:
      'Grounding unmasks 4 <LOC_*> bins into pixelBox overlays—and exposes U27, where skip_special_tokens=True silently deletes all boxes.',
    narrative:
      'Object Grounding runs the identical 34-layer U path to iteratively unmask <LOC_BEGIN>[MASK]*4<LOC_END> slots into <LOC_0..16> coordinates (production: 1,025 <LOC_0..1024> bins extending vocab_size to 132,101, U4) with a domain guard preventing U21 crashes, then un-letterboxes each box onto the RGB inset via pixel_box. Crucially, U27 documents a verified production trap: tokenizer_config.json marks all 1,029 <LOC_*>/box tokens as "special": true, so model.generate()’s skip_special_tokens=True silently erases every predicted bounding box unless token IDs are read directly.',
    equation:
      'pixel_box(v) = clip(v · max(W, H)/16 − (max(W, H) − D_axis)/2, 0, D_axis)   [U27: read raw IDs]',
    ledgerRefs: Object.freeze(['U4', 'U21', 'U27']),
    viewMode: 'output_edit',
    taskMode: 'object_grounding',
    executionViewId: 'object_grounding',
    prefillMode: 'elastic_pruned',
    demoKey: 'object_grounding',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      cropToSource: false,
      showPaddingLockOverlay: false,
    }),
    highlightPartId: 'unmask_mask_state',
  }),

  Object.freeze({
    index: 25,
    id: 'act5_opd_readonly_cross_paradigm',
    actId: 'act5',
    actNumber: 5,
    beatInAct: 3,
    title: 'Read-Only Cross-Paradigm OPD Explorer (0 Gradients)',
    takeaway:
      'Strictly forward-only (READ-ONLY · NO GRADIENTS): inspects how AR VLM and Continuous-Flow teachers distill into one discrete student.',
    narrative:
      'View 7 is strictly quarantined from live inference and badged READ-ONLY · NO GRADIENTS: it executes zero backward passes, zero optimizer steps, and loads zero teacher checkpoints. Instead, it evaluates exact closed-form distillation math across two bays—Left Bay (Indigo U): AR VLM teacher → unified discrete DLLM student via dual-stream SBD masking; Right Bay (Teal G): continuous-flow editing teacher (Qwen-Image) → discrete VQ student via partial flow noising and ODE re-quantization.',
    equation:
      'Left (U): AR Teacher → Student AR → stopgrad → Student Diff  |  Right (G): Flow ODE → Re-Quant VQ',
    ledgerRefs: Object.freeze(['U24', 'U25']),
    viewMode: 'opd_readonly',
    taskMode: 'text_understanding',
    executionViewId: 'text_understanding',
    prefillMode: 'elastic_pruned',
    demoKey: 'text_understanding',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      opdSeed: 0x4f504431,
      opdR: 0.9,
      opdSteps: 40,
      opdSeqLen: 1024,
      opdSelectedBay: 'both',
    }),
    highlightPartId: 'sbd_dual_stream_mask',
  }),

  Object.freeze({
    index: 26,
    id: 'act5_transitive_tail_kl_and_u28',
    actId: 'act5',
    actNumber: 5,
    beatInAct: 4,
    title: 'Transitive 129-Bucket tail_kl, "New Francisco" & U28 Trainer Diagnostics',
    takeaway:
      'Transitive AR→Diff distillation prevents "New Francisco" context mismatch and exposes 3 verified minimal_training/ trainer bugs (U28).',
    narrative:
      'Copying an autoregressive teacher’s next-token score directly onto a bidirectional masked diffusion student fails on coupled phrases ("New [MASK]" vs "[MASK] Francisco" yielding "New Francisco"). NLD-Uni breaks the bridge into two 129-bucket tail_kl steps (Top-64 ∪ Top-128 + residual tail, U24): 4.0× reverse KL from AR teacher to student’s AR view, plus stop-gradient forward self-consistency KL into the dual-stream SBD diffusion view. The diagnostic card surfaces all 3 verified U28 bugs in minimal_training/: ~6.915× (1,000× spike) SFT double-normalization, accidental <|im_end|> unmasking, and 132,101 − 64 <LOC_*> tail leakage.',
    equation:
      'L_OPD = 4.0 · KL_129(p_s^AR ∥ sg(p_t^AR)) + KL_129(sg(p_s^AR) ∥ p_s^Diff) / (0.5005 · R)',
    ledgerRefs: Object.freeze(['U24', 'U28']),
    viewMode: 'opd_readonly',
    taskMode: 'text_understanding',
    executionViewId: 'text_understanding',
    prefillMode: 'elastic_pruned',
    demoKey: 'text_understanding',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      opdSeed: 0x4f504431,
      opdR: 0.9,
      opdSteps: 40,
      opdSeqLen: 1024,
      opdSelectedBay: 'understanding',
    }),
    highlightPartId: 'opd_u28_diagnostics_card',
  }),

  Object.freeze({
    index: 27,
    id: 'act5_continuous_flow_editing_opd',
    actId: 'act5',
    actNumber: 5,
    beatInAct: 5,
    title: 'Continuous-Flow Editing OPD: matched_sigmas(0.9) & x^r ODE Distillation',
    takeaway:
      'Student VQ rollouts are partially noised to x^r (r = 0.9), refined by a continuous-flow ODE teacher, and re-quantized for hybrid CE.',
    narrative:
      'Closing in the right OPD bay, NLD-Uni bridges continuous-flow editing teachers into its discrete ball-tree student without differentiable VQ relaxation: decode the student’s discrete rollout into RGB, encode into continuous VAE latent x^0, mix to x^r = (1 − r)x^0 + r·δ at r = 0.9, invert Qwen-Image’s dynamic shift schedule via matched_sigmas(0.9, seqLen = 1024, steps = 40) (0.9 → 0.02), run multi-step teacher ODE denoising, and re-quantize into stop-gradient discrete targets supervised by exact_ce + 0.2 · grouped_ce (U24).',
    equation:
      'x^r = (1 − r)x^0 + r·δ  (r = 0.9, matched_sigmas → 0.02)  ⇒  L_edit = exact_ce + 0.2 · grouped_ce',
    ledgerRefs: Object.freeze(['U24']),
    viewMode: 'opd_readonly',
    taskMode: 'image_edit',
    executionViewId: 'image_edit',
    prefillMode: 'elastic_pruned',
    demoKey: 'edit_panorama_elastic',
    targetStepOffset: -1,
    focusPatch: Object.freeze({
      opdSeed: 0x4f504431,
      opdR: 0.9,
      opdSteps: 40,
      opdSeqLen: 1024,
      opdSelectedBay: 'editing',
    }),
    highlightPartId: 'opd_matched_sigmas',
  }),
]);

// ============================================================================
// DETERMINISTIC PRIVATE TOUR DEMO BRANCH DEFINITIONS (7 BRANCHES <= 12 CAP)
// ============================================================================

interface TourDemoBranchConfig {
  readonly startOptions: UniSessionStartBranchOptions;
  readonly maxCompletionSteps: number;
}

const TOUR_DEMO_CONFIGS: Readonly<Record<UniTourDemoKey, TourDemoBranchConfig>> = Object.freeze({
  t2i_elastic: Object.freeze({
    startOptions: Object.freeze({
      taskMode: 'text_to_image',
      label: 'Tour · T2I Elastic-Pruned',
      prefillMode: 'elastic_pruned',
      gainPresetId: 'mini_default',
      guidanceScaleText: 3.0,
      activationSteps: 4,
      refinementSteps: 4,
      shift: 5.0,
      temperature: 0.8,
      seed: 0x544f5501,
    }),
    maxCompletionSteps: 12,
  }),
  t2i_hf_faithful: Object.freeze({
    startOptions: Object.freeze({
      taskMode: 'text_to_image',
      label: 'Tour · T2I HF-Faithful (D5)',
      prefillMode: 'hf_faithful',
      gainPresetId: 'mini_default',
      guidanceScaleText: 3.0,
      activationSteps: 4,
      refinementSteps: 4,
      shift: 5.0,
      temperature: 0.8,
      seed: 0x544f5501,
    }),
    maxCompletionSteps: 12,
  }),
  edit_panorama_elastic: Object.freeze({
    startOptions: Object.freeze({
      taskMode: 'image_edit',
      label: 'Tour · Edit 16×8 Panorama (Elastic)',
      scenePresetId: 'harbor_panorama_16x8',
      prefillMode: 'elastic_pruned',
      gainPresetId: 'mini_default',
      guidanceMode: 'legacy',
      guidanceScaleText: 1.6,
      guidanceScaleImage: 2.0,
      activationSteps: 4,
      refinementSteps: 4,
      shift: 5.0,
      temperature: 0.8,
      lockSourcePaddingVq: true,
      cropToSource: true,
      seed: 0x544f5502,
    }),
    maxCompletionSteps: 12,
  }),
  edit_panorama_hf: Object.freeze({
    startOptions: Object.freeze({
      taskMode: 'image_edit',
      label: 'Tour · Edit 16×8 Panorama (HF-Faithful)',
      scenePresetId: 'harbor_panorama_16x8',
      prefillMode: 'hf_faithful',
      gainPresetId: 'mini_default',
      guidanceMode: 'legacy',
      guidanceScaleText: 1.6,
      guidanceScaleImage: 2.0,
      activationSteps: 4,
      refinementSteps: 4,
      shift: 5.0,
      temperature: 0.8,
      lockSourcePaddingVq: true,
      cropToSource: true,
      seed: 0x544f5502,
    }),
    maxCompletionSteps: 12,
  }),
  visual_understanding: Object.freeze({
    startOptions: Object.freeze({
      taskMode: 'visual_understanding',
      label: 'Tour · Visual Understanding (RADIO-H)',
      scenePresetId: 'sunlit_cat_16x16',
      prefillMode: 'elastic_pruned',
      gainPresetId: 'mini_default',
      blockLength: 4,
      stepsPerBlock: 4,
      maxNewTokens: 4,
      confidenceThreshold: 0.9,
      causalContext: true,
      suppressMaskLogit: true,
      seed: 0x544f5503,
    }),
    maxCompletionSteps: 8,
  }),
  text_understanding: Object.freeze({
    startOptions: Object.freeze({
      taskMode: 'text_understanding',
      label: 'Tour · Text Understanding (FLARE B=4)',
      prefillMode: 'elastic_pruned',
      gainPresetId: 'mini_default',
      blockLength: 4,
      stepsPerBlock: 4,
      maxNewTokens: 4,
      confidenceThreshold: 0.9,
      causalContext: true,
      suppressMaskLogit: true,
      seed: 0x544f5504,
    }),
    maxCompletionSteps: 8,
  }),
  object_grounding: Object.freeze({
    startOptions: Object.freeze({
      taskMode: 'object_grounding',
      label: 'Tour · Object Grounding (<LOC_0..16>)',
      scenePresetId: 'sunlit_cat_16x16',
      prefillMode: 'elastic_pruned',
      gainPresetId: 'mini_default',
      groundingTags: Object.freeze(['cat', 'sun']),
      groundingSteps: 2,
      enforceLocDomainGuard: true,
      seed: 0x544f5505,
    }),
    maxCompletionSteps: 4,
  }),
});

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

const ACTS_BY_ID: ReadonlyMap<UniTourActId, UniTourActSummary> = new Map(
  UNI_TOUR_ACTS.map((act) => [act.id, act]),
);

function clampBeatIndex(rawIndex: number, fallbackIndex: number): number {
  if (!Number.isFinite(rawIndex)) {
    return fallbackIndex;
  }
  const truncated = Math.trunc(rawIndex);
  if (truncated < 0) return 0;
  if (truncated >= UNI_TOUR_TOTAL_BEATS) return UNI_TOUR_TOTAL_BEATS - 1;
  return truncated;
}

function resolveRecordIdForOffset(
  recordIds: readonly string[],
  targetStepOffset: number,
): string {
  if (recordIds.length === 0) {
    throw new RangeError('Tour demo branch produced zero records');
  }
  const idx =
    targetStepOffset < 0
      ? Math.max(0, recordIds.length + Math.trunc(targetStepOffset))
      : Math.min(recordIds.length - 1, Math.trunc(targetStepOffset));
  return recordIds[idx]!;
}

function buildBeatFocusState(
  beat: UniTourBeatSpec,
  recordId: string | null,
): UniSpatialFocusState {
  const hoveredPartId =
    beat.focusPatch.hoveredPartId !== undefined
      ? beat.focusPatch.hoveredPartId
      : (beat.highlightPartId ?? null);

  return Object.freeze({
    ...DEFAULT_UNI_SPATIAL_FOCUS_STATE,
    executionViewId: beat.executionViewId,
    prefillMode: beat.prefillMode,
    ...beat.focusPatch,
    hoveredPartId,
    ...(recordId !== null ? { recordId } : {}),
  });
}

// ============================================================================
// ISOLATED PRIVATE-SESSION GUIDED TOUR CONTROLLER FACTORY
// ============================================================================

/**
 * Create the 5-Act, 28-Beat guided pedagogical tour controller backed by a strictly
 * isolated private `UniSession`.
 *
 * Entering, stepping through all 28 beats, and exiting never touches any external
 * Free-Explore `UniSession`, and returns the saved `UniFreeExploreRestoreState`
 * untouched on `exit()`.
 */
export function createUniGuidedTour(bundle?: UniFullRuntimeBundle): UniTourController {
  const tourSession = new UniSession(bundle ?? createDefaultUniRuntimeBundle());
  const cachedBranchIds = new Map<UniTourDemoKey, string>();

  let active = false;
  let currentBeatIndex = 0;
  let savedRestoreState: UniFreeExploreRestoreState | null = null;

  const ensureDemoBranchRecordId = (beat: UniTourBeatSpec): string => {
    let branchId = cachedBranchIds.get(beat.demoKey);
    if (branchId === undefined) {
      const demoCfg = TOUR_DEMO_CONFIGS[beat.demoKey];
      const branch = tourSession.startBranch(demoCfg.startOptions);
      tourSession.runBranchToCompletion(branch.branchId, demoCfg.maxCompletionSteps);
      branchId = branch.branchId;
      cachedBranchIds.set(beat.demoKey, branchId);
    }

    const branch = tourSession.getBranch(branchId);
    if (!branch) {
      throw new RangeError(
        `Missing cached tour branch "${branchId}" for demoKey "${beat.demoKey}"`,
      );
    }
    return resolveRecordIdForOffset(branch.recordIds, beat.targetStepOffset);
  };

  const peekCachedRecordId = (beat: UniTourBeatSpec): string | null => {
    const branchId = cachedBranchIds.get(beat.demoKey);
    if (branchId === undefined) {
      return null;
    }
    const branch = tourSession.getBranch(branchId);
    if (!branch || branch.recordIds.length === 0) {
      return null;
    }
    return resolveRecordIdForOffset(branch.recordIds, beat.targetStepOffset);
  };

  const buildTourState = (recordId: string | null): UniTourState => {
    const beat = UNI_TOUR_BEATS[currentBeatIndex]!;
    const act = ACTS_BY_ID.get(beat.actId)!;
    return Object.freeze({
      active,
      beatIndex: currentBeatIndex,
      beatCount: UNI_TOUR_TOTAL_BEATS,
      actIndex: act.actNumber - 1,
      actCount: UNI_TOUR_TOTAL_ACTS,
      currentAct: act,
      currentBeat: beat,
      activeRecordId: recordId,
      focus: buildBeatFocusState(beat, recordId),
    });
  };

  const activateBeat = (targetIndex: number): UniTourState => {
    currentBeatIndex = clampBeatIndex(targetIndex, currentBeatIndex);
    const beat = UNI_TOUR_BEATS[currentBeatIndex]!;
    const recordId = ensureDemoBranchRecordId(beat);
    if (tourSession.getActiveRecordId() !== recordId) {
      tourSession.setActiveRecord(recordId);
    }
    return buildTourState(recordId);
  };

  return Object.freeze({
    acts: UNI_TOUR_ACTS,
    beats: UNI_TOUR_BEATS,
    totalBeats: UNI_TOUR_TOTAL_BEATS,
    tourSession,

    getState(): UniTourState {
      const beat = UNI_TOUR_BEATS[currentBeatIndex]!;
      const recordId = active
        ? ensureDemoBranchRecordId(beat)
        : peekCachedRecordId(beat);
      return buildTourState(recordId);
    },

    isActive(): boolean {
      return active;
    },

    start(
      restoreState?: UniFreeExploreRestoreState | null,
      initialBeatIndex = 0,
    ): UniTourState {
      if (!active || restoreState !== undefined) {
        savedRestoreState = restoreState ?? null;
      }
      active = true;
      return activateBeat(clampBeatIndex(initialBeatIndex, 0));
    },

    goToBeat(beatIndex: number): UniTourState {
      return activateBeat(beatIndex);
    },

    nextBeat(): UniTourState {
      return activateBeat(currentBeatIndex + 1);
    },

    prevBeat(): UniTourState {
      return activateBeat(currentBeatIndex - 1);
    },

    goToAct(actId: UniTourActId): UniTourState {
      const act = ACTS_BY_ID.get(actId);
      if (!act) {
        throw new RangeError(`Unknown UniTourActId: "${String(actId)}"`);
      }
      return activateBeat(act.startBeatIndex);
    },

    buildBeatScene(beatIndex?: number): UniSpatialScene {
      const resolvedIndex =
        beatIndex === undefined
          ? currentBeatIndex
          : clampBeatIndex(beatIndex, currentBeatIndex);
      const beat = UNI_TOUR_BEATS[resolvedIndex]!;
      const recordId = ensureDemoBranchRecordId(beat);
      const beatFocus = buildBeatFocusState(beat, recordId);
      const scene = buildUniSpatialScene(beat.viewMode, tourSession, beatFocus);

      if (!beat.cameraOverride) {
        return scene;
      }
      return Object.freeze({
        ...scene,
        cameraPreset: Object.freeze({
          ...scene.cameraPreset,
          azimuthDeg: beat.cameraOverride.azimuthDeg,
          elevationDeg: beat.cameraOverride.elevationDeg,
        }),
      });
    },

    exit(): UniFreeExploreRestoreState | null {
      active = false;
      const restored = savedRestoreState;
      savedRestoreState = null;
      return restored;
    },
  });
}
