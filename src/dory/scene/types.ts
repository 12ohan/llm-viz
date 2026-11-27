/**
 * Shared Headless Spatial Scene, Focus, Pick, Atlas, Badge & Callout Contracts
 * for NVIDIA Nemotron-Labs-Dory1.0-4B (`src/dory/scene/types.ts`)
 *
 * Frozen Step-0 contract shared across all 7 D7 Editorial WebGL2 spatial views:
 *   - DORY3: View 1 `macro` (Folded/Unrolled Macro + KV-Bank Rail)
 *            View 2 `hypersphere` (nGPT Hyperspherical Residual NLERP & Latent Trajectory)
 *   - DORY4: View 3 `gqa_rope` (Gated GQA & Dual-RoPE Cutaway: SWA W=8 vs. Full)
 *            View 4 `swiglu` (Normalized SwiGLU Mixer Cutaway)
 *            View 5 `kv_compare` (Dual-Axis KV Cache & `last_loop` Divergence Inspector)
 *            View 6 `self_spec` (Contract A Paper §6.2 Self-Speculative Execution Inspector)
 *   - DORY5: View 7 `bookends` (Untied Bookends, 4-Format Readouts & Read-Only TBPTL)
 *   - DORY6: Standalone isolated browser app shell (`src/dory/app/*`)
 *
 * Locked Decisions & Guards Enforced Here:
 *   - `D1` Physical `0..71` sublayer indexing (`16 + 40 + 16`) with paired `L{p >> 1}` badges.
 *   - `D3`/`DR23` Contract A `104`-bank self-speculation badged `Reconstructed from Paper §6.2`.
 *   - `D4` Architectural Boundary Isolation: ZERO imports from Gate 4b/4c Nano modules
 *     (`src/ui/*`, `src/scene/spine.ts`, `src/scene/sheetInspector.ts`, `src/model/*`,
 *     `src/math/*`, `src/story/*`, `src/main.ts`) or sibling apps (`twotower`, `diffusion`, `uni`).
 *     Leaf primitives imported strictly from `src/render/primitives.ts` and `src/scene/atlasPacker.ts`.
 *   - `DR7` Anti-SLERP guard (`dory-nlerp-residual`, channel-wise NLERP — NOT trigonometric SLERP).
 *   - `DR20` Live `RG32F` `G`-channel mask required on every `sheetKind in {1, 2}` causal plate.
 *   - `DR21` Zero edits to `src/render/pick.ts`: loop, KV-head, bank, rotary-plane, and stage
 *     metadata encoded via `groupIndex`, `tapIndex`, `stateTokenIndex`, and `semanticTag`.
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL execution, Node built-ins, or ambient RNG.
 */

import type { AABB } from '../../render/camera';
import type { PickBox } from '../../render/pick';
import type {
  DimensionCaliper,
  KvDepthTickAnchor,
  LandmarkAnchor,
  Vec3,
} from '../../render/primitives';
import type { Rg32fValueAtlas } from '../../scene/atlasPacker';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../shared/overlayLayout';
import type {
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_NLERP_MARKER,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  DoryAttentionKind,
  DoryCacheRole,
  DoryCheckpointFormatSpec,
  DoryCounterfactualCodaResult,
  DoryDiscrepancyId,
  DoryDualRopeTables,
  DoryExecutedSublayerRecord,
  DoryExecutionMode,
  DoryForwardTrace,
  DoryKvBankKind,
  DoryKvCacheMode,
  DoryLayerStage,
  DoryLoopTrajectoryMetrics,
  DoryModelParameters,
  DoryProductionKvPoint,
  DoryReconstructedStepResult,
  DoryRopeRegime,
  DorySamplingConfig,
  DoryScalingRatioEntry,
  DorySession,
  DorySessionBranchSnapshot,
  DorySessionRecord,
  DorySpecCycleSummary,
  DorySublayerKind,
} from '../types';

// ============================================================================
// AUTHORITATIVE EPISTEMIC PLAQUES & DISCLOSURE STRINGS (D3, D7, DR7, FINDING 9)
// ============================================================================

/** Mandatory Anti-SLERP terminology plaque (`DR7`, Audit §5.4 & §12.1 View 2). */
export const DORY_ANTI_SLERP_PLAQUE =
  'Channel-Wise Normalized Residual Interpolation (nGPT NLERP) — NOT Trigonometric SLERP' as const;

/** Mandatory View 2 Exploratory 2D PCA disclosure badge (Audit §12.1 View 2, Finding 9). */
export const DORY_EXPLORATORY_PCA_BADGE =
  'EXPLORATORY 2D PCA PROJECTION OF 8D TRACE (PROJECTED NORMS <= 1; ALL NUMERIC METRICS COMPUTED STRICTLY IN 8D)' as const;

/** Mandatory View 2 local 120-degree anchor-map Jacobian caveat (Audit §6.2.1, Finding 9). */
export const DORY_ANCHOR_MAP_CAVEAT_BADGE =
  'Local anchor-map Jacobian < 1 for theta < 120 deg ONLY — does NOT imply global core contraction or convergence' as const;

/** Mandatory miniature vs. production scaling-ratio disclosure (`D2`, Audit §11.1–§11.2). */
export const DORY_SCALING_DISCLOSURE_BADGE =
  'Mini d=8, q=64 (8.0x), kv=16 (2.0x), d_ff=24 (3.0x) · Prod d=2560, q=4096 (1.60x), kv=1024 (0.40x), d_ff=7168 (2.80x)' as const;

/** Mandatory View 7 strictly read-only training/TBPTL schematic badge (Audit §1.4 & §12.1). */
export const DORY_TBPTL_READONLY_BADGE = 'READ-ONLY · NO GRADIENTS' as const;

// ============================================================================
// SEVEN ARCHITECTURE-SPECIFIC DORY1.0-4B SPATIAL VIEW MODES (D7, AUDIT §12.1)
// ============================================================================

export type DorySpatialViewMode =
  | 'macro'
  | 'hypersphere'
  | 'gqa_rope'
  | 'swiglu'
  | 'kv_compare'
  | 'self_spec'
  | 'bookends';

export interface DorySpatialViewDescriptor {
  readonly id: DorySpatialViewMode;
  readonly viewNumber: 1 | 2 | 3 | 4 | 5 | 6 | 7;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
  /** Maximum `RG32F` atlas byte budget (`<= 524,288 B` = `512.00 KiB`; `0` if procedural-only). */
  readonly atlasBudgetBytes: number;
  readonly defaultCameraAzimuthDeg: number;
  readonly defaultCameraElevationDeg: number;
}

export const DORY_SPATIAL_VIEW_MODES: readonly DorySpatialViewDescriptor[] = Object.freeze([
  Object.freeze({
    id: 'macro',
    viewNumber: 1,
    label: 'Folded & Unrolled Macro Architecture',
    shortLabel: 'Macro 72L',
    description:
      'Physical 72-sublayer (16 Prelude + 40 Shared Recurrent Core + 16 Coda) topology toggling between Folded loop feedback and Unrolled 32+40k depth with parallel KV-bank rail (36..176 / 104 Contract A banks).',
    atlasBudgetBytes: 65_536,
    defaultCameraAzimuthDeg: -32,
    defaultCameraElevationDeg: 26,
  }),
  Object.freeze({
    id: 'hypersphere',
    viewNumber: 2,
    label: 'nGPT Hypersphere & Latent Trajectory',
    shortLabel: 'nGPT Sphere',
    description:
      'Channel-wise NLERP residual cutaway (anti-SLERP DR7, raw vs. effective |alpha_hat| incl. >1 extrapolation), Exact 2D Spanning-Subspace diagram, Exploratory 2D PCA with 8D-only metrics, local 120-deg anchor-map caveat, and read-only Diagnostic Counterfactual Coda.',
    atlasBudgetBytes: 131_072,
    defaultCameraAzimuthDeg: -26,
    defaultCameraElevationDeg: 28,
  }),
  Object.freeze({
    id: 'gqa_rope',
    viewNumber: 3,
    label: 'Gated GQA & Dual-RoPE Cutaway',
    shortLabel: 'GQA & RoPE',
    description:
      'SWA (theta_S=16, W=8) vs. Full (theta_F=1024) attention cutaway: Q/K/V/Gate projections (8->64 mini vs. 1.60x prod), 4-plane NeoX dual-RoPE, per-head Q/K L2Norm, Q-only sqk^2, live RG32F G-channel causal/banded masks (DR20), sigmoid output gate, and attn_sub_norm.',
    atlasBudgetBytes: 524_288,
    defaultCameraAzimuthDeg: -24,
    defaultCameraElevationDeg: 32,
  }),
  Object.freeze({
    id: 'swiglu',
    viewNumber: 4,
    label: 'Normalized SwiGLU Mixer Cutaway',
    shortLabel: 'SwiGLU',
    description:
      'Bias-free gate_proj and up_proj (8->24, 3.00x mini vs. 2.80x prod), separate signed suv_gate/suv_up (*sqrt(d)), SiLU product, pre-down_proj ffn_sub_norm, down_proj, NLERP residual blend, and progressive accumulation reveal.',
    atlasBudgetBytes: 131_072,
    defaultCameraAzimuthDeg: -28,
    defaultCameraElevationDeg: 28,
  }),
  Object.freeze({
    id: 'kv_compare',
    viewNumber: 5,
    label: 'Dual-Axis KV & last_loop Divergence',
    shortLabel: 'KV & last_loop',
    description:
      'Side-by-side immutable per_loop vs. last_loop comparison, SWA eviction boundary (S>8), live prefill-partition divergence probe (Theorems 1–3, DR17), and 1->256K BF16 production scaling with S-invariant (4+5k)/9 ratio.',
    atlasBudgetBytes: 262_144,
    defaultCameraAzimuthDeg: -22,
    defaultCameraElevationDeg: 30,
  }),
  Object.freeze({
    id: 'self_spec',
    viewNumber: 6,
    label: 'Self-Speculative Execution (Contract A)',
    shortLabel: 'Self-Spec §6.2',
    description:
      'Reconstructed Paper §6.2 (Contract A, 104 banks, per_loop only): shallow k=1 drafting with persistent output_shallow Coda KV, batched k=4 verification reusing Prelude + Loop-1 (x, s_1) & KV, binary64 rejection/residual, and transaction-local cap_swa=16 rollback.',
    atlasBudgetBytes: 262_144,
    defaultCameraAzimuthDeg: -26,
    defaultCameraElevationDeg: 29,
  }),
  Object.freeze({
    id: 'bookends',
    viewNumber: 7,
    label: 'Bookends, 4-Format Readouts & TBPTL',
    shortLabel: 'Bookends',
    description:
      'Untied unit-norm embeddings & lm_head separating raw logits -> FP32 logit_scale/0.0177 -> probabilities, 0-byte-diff 4-checkpoint production readouts (BF16/MXFP8/NVFP4/Q5_K_M), and strictly READ-ONLY · NO GRADIENTS TBPTL depth-1 schematic.',
    atlasBudgetBytes: 131_072,
    defaultCameraAzimuthDeg: -28,
    defaultCameraElevationDeg: 26,
  }),
]);

// ============================================================================
// PROVENANCE BADGES, CALLOUTS, OVERLAY ANCHORS & ATLAS AUDIT CONTRACTS
// ============================================================================

export type DoryBadgeTone =
  | 'prelude_indigo'
  | 'recurrent_teal'
  | 'coda_amber'
  | 'swa_cyan'
  | 'full_indigo'
  | 'nlerp_emerald'
  | 'spec_oxide'
  | 'warn'
  | 'gold'
  | 'muted'
  | 'readonly';

export interface DoryProvenanceBadge {
  readonly label: string;
  readonly tone: DoryBadgeTone;
  readonly discrepancyId?: DoryDiscrepancyId;
}

export interface DorySceneCallout {
  readonly id: string;
  readonly title: string;
  readonly body: string;
  readonly tone: DoryBadgeTone;
  readonly worldPos?: Vec3;
  readonly badge?: string;
  readonly discrepancyIds?: readonly DoryDiscrepancyId[];
}

export interface DoryOverlayAnchor {
  readonly id: string;
  readonly label: string;
  readonly sublabel?: string;
  readonly worldPos: Vec3;
  readonly tone?: DoryBadgeTone;
  readonly badge?: string;
}

export interface DoryAtlasRegionBox {
  readonly id: string;
  readonly x0: number;
  readonly y0: number;
  readonly cols: number;
  readonly rows: number;
  /** `0` = diverging signed palette, `1` = positive/probability `[0..1]` palette. */
  readonly mode: 0 | 1;
  /** True for SWA/Full causal score or softmax plates carrying live `G`-channel mask (`DR20`). */
  readonly isMaskOrScore: boolean;
  readonly physicalLayerIdx?: number;
  readonly loopIteration?: number;
}

export interface DoryAtlasLayoutAudit {
  readonly width: number;
  readonly height: number;
  readonly totalTexels: number;
  readonly usedTexels: number;
  readonly utilizationPct: number;
  readonly byteLength: number;
  readonly maxBudgetBytes: number;
  readonly withinBudget: boolean;
  readonly regions: readonly DoryAtlasRegionBox[];
  readonly hasOverlap: boolean;
  readonly overlapPair: readonly [string, string] | null;
  readonly outOfBoundsIds: readonly string[];
}

export interface DoryCameraPreset {
  readonly azimuthDeg: number;
  readonly elevationDeg: number;
  readonly fitBounds: AABB;
}

export interface DorySceneValidationOptions {
  /**
   * When `true` (default), throws `DoryDr20CausalMaskError` immediately if any sheet with
   * `sheetKind in {1, 2}` lacks a live `RG32F` atlas or valid positive `TexRect` (`DR20`).
   * Pass `{ throwOnDr20: false }` to collect `DR20` violations in `errors` without throwing.
   */
  readonly throwOnDr20?: boolean;
}

export interface DorySceneValidationResult {
  readonly valid: boolean;
  readonly errors: readonly string[];
}

// ============================================================================
// DR21 PICK METADATA ENCODING & DECODING CONTRACTS (ZERO EDITS TO pick.ts)
// ============================================================================

/** Meaning carried in `PickBox.groupIndex` when encoding Dory spatial metadata (`DR21`). */
export type DoryPickGroupRole = 'kv_head' | 'bank' | 'loop';

/** Meaning carried in `PickBox.tapIndex` when encoding Dory spatial metadata (`DR21`). */
export type DoryPickTapRole = 'loop' | 'plane' | 'spec';

/** High-level typed input accepted by `encodeDoryPickMetadata()` (`DR21`). */
export interface DoryPickEncodeInput {
  /** Canonical physical sublayer `0..71` (`bayId = physicalLayerIdx + 1`), or `null` (`bayId = 0`). */
  readonly physicalLayerIdx?: number | null;
  /** Optional explicit `bayId` override when `physicalLayerIdx` is `null`. */
  readonly bayId?: number;
  readonly stage?: DoryLayerStage | 'bookend';
  readonly executedLayerIdx?: number | null;
  readonly loopIteration?: number | null;
  readonly partId?: string;
  readonly tensorKey?: string;
  readonly qHeadIndex?: number | null;
  readonly kvHeadIndex?: number | null;
  readonly rotaryPlane?: number | null;
  readonly kvBankSlotId?: number | null;
  readonly kvBankKind?: DoryKvBankKind | null;
  readonly kvPolicy?: DoryKvCacheMode | null;
  readonly cacheRole?: DoryCacheRole | null;
  readonly specOffset?: number | null;
  readonly tokenIndex?: number | null;
  readonly absTokenPos?: number | null;
  readonly gridCols?: number;
  readonly gridRows?: number;
  readonly modelWeightId?: string;
  readonly traceOrigin?: string;
  /** Optional override for which coordinate `groupIndex` carries (`'kv_head' | 'bank' | 'loop'`). */
  readonly groupRole?: DoryPickGroupRole;
  /** Optional override for which coordinate `tapIndex` carries (`'loop' | 'plane' | 'spec'`). */
  readonly tapRole?: DoryPickTapRole;
  /** Optional extra `key=value` suffix tags appended to `semanticTag`. */
  readonly extraTags?: Readonly<Record<string, string | number | boolean>>;
}

/** `PickBox` metadata subset (without world `min`/`max`/`frame`) produced by `encodeDoryPickMetadata`. */
export interface DoryPickBoxMeta {
  readonly bayId: number;
  readonly partId?: string;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly groupIndex?: number;
  readonly tensorKey?: string;
  readonly gridCols?: number;
  readonly gridRows?: number;
  readonly tokenIndex?: number;
  readonly tapIndex?: number;
  readonly stateTokenIndex?: number;
  readonly towerRole?: 'context' | 'denoiser' | 'shared';
  readonly executionId?: string;
  readonly traceOrigin?: string;
  readonly semanticTag?: string;
  readonly absoluteTokenOffset?: number;
}

/**
 * Full Dory1.0-4B pick identity decoded from shared `PickHit` with ZERO edits to `src/render/pick.ts` (`DR21`).
 * `groupIndex` carries KV head (`0..1`), KV bank slot (`0..175`), or loop iteration (`0..8`);
 * `tapIndex` carries loop iteration (`0..8`), rotary plane (`0..3`), or speculative offset (`0..4`);
 * `semanticTag` carries structured `stage=...;loop=...;kv=...;bank=...;bankKind=...;policy=...;role=...` provenance.
 */
export interface DoryPickIdentity {
  readonly viewMode: DorySpatialViewMode;
  readonly physicalLayerIdx: number | null;
  readonly pairedLayerIdx: number | null;
  readonly executedLayerIdx: number | null;
  readonly stage: DoryLayerStage | 'bookend' | null;
  readonly sublayerKind: DorySublayerKind | null;
  readonly loopIteration: number | null;
  readonly partId: string | null;
  readonly tensorKey: string | null;
  readonly qHeadIndex: number | null;
  readonly kvHeadIndex: number | null;
  readonly rotaryPlane: number | null;
  readonly kvBankSlotId: number | null;
  readonly kvBankKind: DoryKvBankKind | null;
  readonly kvPolicy: DoryKvCacheMode | null;
  readonly cacheRole: DoryCacheRole | null;
  readonly specOffset: number | null;
  readonly tokenIndex: number | null;
  readonly absTokenPos: number | null;
  readonly row: number | null;
  readonly col: number | null;
  readonly modelWeightId: string | null;
  readonly semanticTag: string | null;
}

// ============================================================================
// VIEW-SPECIFIC ANALYTICAL & INSET METADATA DESCRIPTORS (ALL 7 D7 VIEWS)
// ============================================================================

/** View 1 (`macro`) display mode: physical 72L Folded loop vs. Unrolled `32 + 40k` executed depth. */
export type DoryMacroDisplayMode = 'folded' | 'unrolled';

/** Proves every unrolled recurrent pass `t = 1..k` shares identical physical parameters (`modelWeightId`, `16..55`). */
export interface DorySharedWeightBinding {
  readonly loopIteration: number;
  readonly executedLayerStart: number;
  readonly executedLayerEnd: number;
  readonly physicalLayerStart: 16;
  readonly physicalLayerEnd: 55;
  readonly modelWeightId: string;
  readonly kvBankRangeStart: number;
  readonly kvBankRangeEnd: number;
  readonly isSharedOverwriteBank: boolean;
}

export interface DoryMacroKvRailSummary {
  readonly kvCacheMode: DoryKvCacheMode;
  readonly includeShallowCoda: boolean;
  readonly totalBanks: number;
  readonly inputBanks: number;
  readonly recurrentBanks: number;
  readonly outputDeepBanks: number;
  readonly outputShallowBanks: number;
  readonly totalSwaBanks: number;
  readonly totalFullBanks: number;
}

export interface DoryMacroViewMetadata {
  readonly displayMode: DoryMacroDisplayMode;
  readonly modelWeightId: string;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  readonly physicalSublayerCount: 72;
  readonly executedSublayerCount: number;
  readonly pairedLayerCount: 36;
  readonly effectivePairedDepth: number;
  readonly kvRail: DoryMacroKvRailSummary;
  readonly sharedRecurrentWeightBindings: readonly DorySharedWeightBinding[];
  readonly selectedPhysicalLayerIdx: number | null;
  readonly selectedLoopIteration: number | null;
}

/** Sample point along the uniform-rate NLERP arc inside the exact 2D spanning subspace `span(a, b)`. */
export interface DorySpanningSubspaceCurvePoint {
  readonly lambda: number;
  readonly x: number;
  readonly y: number;
  readonly isExtrapolation: boolean;
}

/**
 * View 2 Diagram (a) — Exact 2D Spanning-Subspace Diagram (`span(a, b)` for one selected NLERP update).
 * Strictly distinct from the Exploratory 2D PCA trajectory projection (Finding 9).
 */
export interface DorySpanningSubspaceDiagramData {
  readonly diagramTitle: string;
  readonly antiSlerpCaption: string;
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly loopIteration: number;
  readonly tokenOffset: number;
  readonly absTokenPos: number;
  /** Exact unit vector `a = N(x_res)` mapped to `(1, 0)` in `span(a, b)`. */
  readonly a2d: readonly [number, number];
  /** Exact unit vector `b = N(x_sub)` mapped to `(cos(omega), sin(omega))` in `span(a, b)`. */
  readonly b2d: readonly [number, number];
  readonly omegaRadians: number;
  readonly omegaDegrees: number;
  readonly cosineAB: number;
  /** Exact uniform-scalar NLERP curve `N(a + lambda*(b - a))` over `lambda in [0, maxLambda]`. */
  readonly uniformAlphaCurve: readonly DorySpanningSubspaceCurvePoint[];
  /** Orthogonal projection of per-channel pre-norm `a + |alpha_hat|*(b - a)` onto `span(a, b)`. */
  readonly projectedBlendedPreNorm2d: readonly [number, number];
  /** Orthogonal projection of actual 8D NLERP output `N(a + |alpha_hat|*(b - a))` onto `span(a, b)`. */
  readonly projectedOutput2d: readonly [number, number];
  /** Out-of-plane component norm `||x_next - Pi_{span(a,b)}(x_next)||_2` induced by per-channel rates. */
  readonly outOfPlaneNorm: number;
  readonly rawAlpha: Float32Array;
  readonly effectiveAlpha: Float32Array;
  readonly minEffectiveAlpha: number;
  readonly meanEffectiveAlpha: number;
  readonly maxEffectiveAlpha: number;
  /** Channel indices with `|alpha_hat_c| > 1.0` (`DR7` unclamped extrapolation). */
  readonly extrapolatedChannels: readonly number[];
}

export interface DoryPcaTrajectoryPoint {
  readonly id: string;
  readonly label: string;
  readonly kind: 'anchor' | 'loop_entry' | 'loop_exit';
  readonly loopIteration: number;
  /** 2D PCA coordinates (always satisfy `hypot(x, y) <= 1.0 + 1e-6` for unit-sphere states). */
  readonly x: number;
  readonly y: number;
  readonly projectedNorm: number;
  /** Exact metrics computed STRICTLY in original 8D (`Finding 9`). */
  readonly exact8dNorm: number;
  readonly exact8dAnchorCosine: number;
  readonly exact8dAnchorAngleDeg: number;
  readonly exact8dConsecutiveCosine: number | null;
  readonly exact8dConsecutiveDisplacement: number | null;
}

/**
 * View 2 Diagram (b) — Exploratory 2D PCA Trajectory Projection of actual 8D latent states
 * `{x, u_1, s_1, ..., u_k, s_k}` with ALL numeric metrics computed strictly in original 8D.
 */
export interface DoryExploratoryPcaTrajectoryData {
  readonly disclosureBadge: typeof DORY_EXPLORATORY_PCA_BADGE;
  readonly metricsComputedIn8d: true;
  readonly allProjectedNormsLeqOne: boolean;
  readonly explainedVarianceRatio: readonly [number, number];
  readonly mean8d: Float32Array;
  readonly pc1Axis8d: Float32Array;
  readonly pc2Axis8d: Float32Array;
  readonly points: readonly DoryPcaTrajectoryPoint[];
}

export interface DoryCounterfactualTopToken {
  readonly tokenId: number;
  readonly tokenText: string;
  readonly prob: number;
  readonly scaledLogit: number;
}

/** Read-only View 2 intermediate-loop Coda logit-lens preview (`writeKv: false`, zero KV mutation). */
export interface DoryCounterfactualCodaPreview {
  readonly badge: typeof DORY_COUNTERFACTUAL_CODA_BADGE;
  readonly recordId: string;
  readonly loopIteration: number;
  readonly kLoops: number;
  readonly absTokenPos: number;
  readonly readOnlyZeroKvMutation: true;
  readonly topTokens: readonly DoryCounterfactualTopToken[];
  /** KL divergence `D_KL(p_final || p_counterfactual)` in nats computed in binary64. */
  readonly klFromFinalLoopNats: number;
  readonly argMaxTokenId: number;
  readonly finalLoopArgMaxTokenId: number;
  readonly argMaxChanged: boolean;
  readonly rawResult: DoryCounterfactualCodaResult;
}

export interface DoryHypersphereViewMetadata {
  readonly antiSlerpPlaque: typeof DORY_ANTI_SLERP_PLAQUE;
  readonly antiSlerpMarker: typeof DORY_NLERP_MARKER;
  readonly anchorMapCaveatBadge: typeof DORY_ANCHOR_MAP_CAVEAT_BADGE;
  readonly selectedPhysicalLayerIdx: number;
  readonly selectedPairedLayerIdx: number;
  readonly selectedLoopIteration: number;
  readonly selectedTokenOffset: number;
  readonly selectedAbsTokenPos: number;
  readonly selectedTokenId: number;
  readonly selectedTokenText: string;
  readonly loopMetrics8d: readonly DoryLoopTrajectoryMetrics[];
  readonly spanningSubspace: DorySpanningSubspaceDiagramData;
  readonly exploratoryPca: DoryExploratoryPcaTrajectoryData;
  readonly counterfactualCoda: DoryCounterfactualCodaPreview | null;
}

/** View 3 (`gqa_rope`) 4-plane NeoX dual-RoPE phasor inset descriptor. */
export interface DoryRopePhasorPlane {
  readonly planeIdx: number;
  readonly invFreq: number;
  readonly periodTokens: number;
  readonly queryAngleRad: number;
  readonly keyAngleRad: number;
  readonly deltaAngleRad: number;
  readonly qPrePair: readonly [number, number];
  readonly qRotPair: readonly [number, number];
  readonly kPrePair: readonly [number, number];
  readonly kRotPair: readonly [number, number];
}

export interface DoryGqaRopeViewMetadata {
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly stage: DoryLayerStage;
  readonly loopIteration: number;
  readonly attentionKind: DoryAttentionKind;
  readonly ropeRegime: DoryRopeRegime;
  readonly ropeThetaMini: number;
  readonly ropeThetaProd: number;
  readonly slidingWindowMini: number | null;
  readonly slidingWindowProd: number | null;
  readonly selectedQHead: number;
  readonly selectedKvHead: number;
  readonly selectedQueryRow: number;
  readonly selectedKeyCol: number;
  readonly selectedQueryAbsPos: number;
  readonly selectedKeyAbsPos: number;
  readonly selectedRotaryPlane: number;
  readonly seqLen: number;
  readonly startPos: number;
  readonly pastLen: number;
  readonly kvLen: number;
  readonly bankSlotId: number;
  readonly bankKind: DoryKvBankKind;
  /** False on non-final `last_loop` passes (`t < k`, `DoryPastReader` read-only semantics). */
  readonly writeKv: boolean;
  readonly visibleCellCount: number;
  readonly maskedCellCount: number;
  readonly qExpansionDisclosure: string;
  /** Miniature `q_dim / d` (`64 / 8 = 8.0`) vs. production `4096 / 2560 = 1.6`. */
  readonly qExpansionRatioMini: number;
  readonly qExpansionRatioProd: number;
  /** Miniature `kv_dim / d` (`16 / 8 = 2.0`) vs. production `1024 / 2560 = 0.4`. */
  readonly kvDimRatioMini: number;
  readonly kvDimRatioProd: number;
  /** `[min, max]` of effective Q-only squared scale `sqk^2 = (raw * sqrt(d_h) / sigma_0)^2`. */
  readonly sqkScaleRange: readonly [number, number];
  /** `[min, max]` of effective output-gate squared scale `s_gate^2 = (raw * sqrt(g_0) / sigma_0)^2`. */
  readonly sGateScaleRange: readonly [number, number];
  /** Axis width of `attn_sub_norm`: full concatenated `h_q * d_h = 64` (NOT per-head `d_h = 8`). */
  readonly attnSubNormAxisDim: number;
  /** Verified L2 norm (`~1.0`) of `attn_sub_norm` on `selectedQueryRow` across all `q_dim` channels. */
  readonly attnSubNormL2Norm: number;
  /** Canonical `DR6` order: NeoX dual-RoPE applied BEFORE per-head Q/K L2Norm. */
  readonly ropeBeforeQkNorm: true;
  /** Count of channels with unclamped `|alpha_hat_c| > 1` on this attention layer's NLERP (`DR7`). */
  readonly extrapolatedAlphaCount: number;
  readonly ropePhasors: readonly DoryRopePhasorPlane[];
  readonly kvAbsPositions: readonly number[];
  readonly kvWriterLoops: readonly number[];
  readonly hasLiveCausalMaskGChannel: true;
}

/** View 4 (`swiglu`) Normalized SwiGLU cutaway descriptor. */
export interface DorySwigluViewMetadata {
  readonly physicalLayerIdx: number;
  readonly pairedLayerIdx: number;
  readonly stage: DoryLayerStage;
  readonly loopIteration: number;
  readonly seqLen: number;
  readonly startPos: number;
  readonly selectedTokenRow: number;
  readonly selectedAbsTokenPos: number;
  readonly selectedTokenId: number;
  readonly selectedTokenText: string;
  readonly selectedChannel: number;
  readonly selectedOutputDim: number;
  readonly accumReveal: number;
  /** Number of intermediate channels (`0..d_ff`) revealed in progressive `down_proj` accumulation. */
  readonly revealedIntermediateCount: number;
  readonly mlpExpansionDisclosure: string;
  /** Miniature `d_ff / d` (`24 / 8 = 3.0`) vs. production `7168 / 2560 = 2.8`. */
  readonly mlpExpansionRatioMini: number;
  readonly mlpExpansionRatioProd: number;
  /** Family 4 runtime multiplier `sqrt(d)` (NO `sigma_0` divisor, NO `.abs()`). */
  readonly suvMultiplierMini: number;
  readonly suvMultiplierProd: number;
  /** Axis width of `ffn_sub_norm` (`d_ff = 24`) applied strictly BEFORE `down_proj`. */
  readonly ffnSubNormAxisDim: number;
  /** Verified L2 norm (`~1.0`) of `ffn_sub_norm` on `selectedTokenRow` across all `d_ff` channels. */
  readonly ffnSubNormL2Norm: number;
  readonly hasSignedNegativeSuv: boolean;
  readonly negativeSuvGateCount: number;
  readonly negativeSuvUpCount: number;
  readonly extrapolatedAlphaCount: number;
  readonly maxEffectiveAlpha: number;
  readonly antiSlerpMarker: typeof DORY_NLERP_MARKER;
}

/** View 5 (`kv_compare`) Dual-Axis KV & `last_loop` prefill-partition divergence descriptor. */
export type DoryKvComparePartitionMode = 'single_shot' | 'chunked' | 'one_by_one';

export interface DoryKvPositionDivergencePoint {
  readonly absPos: number;
  readonly tokenId: number;
  readonly tokenText: string;
  readonly chunkIndex: number;
  readonly maxAbsLogitDiff: number;
  readonly beyondSwaWindow: boolean;
}

export interface DoryKvCompareViewMetadata {
  readonly kLoops: number;
  readonly promptLen: number;
  readonly partitionMode: DoryKvComparePartitionMode;
  readonly chunkPartition: readonly number[];
  readonly selectedPhysicalLayerIdx: number;
  readonly selectedPairedLayerIdx: number;
  readonly selectedAttentionKind: DoryAttentionKind;
  readonly selectedLoopIteration: number;
  readonly perLoopBankCount: number;
  readonly perLoopSwaBanks: number;
  readonly perLoopFullBanks: number;
  readonly lastLoopBankCount: number;
  readonly lastLoopSwaBanks: number;
  readonly lastLoopFullBanks: number;
  readonly perLoopMiniBytes: number;
  readonly lastLoopMiniBytes: number;
  readonly swaWindowMini: number;
  readonly swaRingCapacityMini: number;
  readonly swaWindowProd: number;
  /** Exact `4x` marginal bytes/token slope drop once `S > W_prod = 1,024` (`Audit §7.2`). */
  readonly swaKneeSlopeDropFactor: 4;
  /** Exact `S`-invariant ratio `(4 + 5k) / 9` (`Audit §7.2`). */
  readonly sInvariantRatio: number;
  /** True iff `kLoops === 1` (`Theorem 2`: universal `last_loop === per_loop` equivalence). */
  readonly theorem2K1Equivalence: boolean;
  readonly theorem1SingleShotMaxAbsDiff: number;
  readonly theorem3ChunkedVsPerLoopMaxAbsDiff: number;
  readonly theorem3ChunkedVsOneByOneMaxAbsDiff: number;
  /** Max abs logit diff vs. `per_loop` for the currently active `partitionMode`. */
  readonly activePartitionVsPerLoopMaxAbsDiff: number;
  /** Confirms `per_loop` and `last_loop` ran in isolated sessions with zero caller mutation (`D6`/`DR18`). */
  readonly isolatedDualSessionVerified: true;
  readonly hasLiveCausalMaskGChannel: true;
  readonly perPositionDivergence: readonly DoryKvPositionDivergencePoint[];
  /** Production `1 -> 256K` BF16 `per_loop` scaling curve at `kLoops`. */
  readonly productionScalingCurve: readonly DoryProductionKvPoint[];
  /** Production `1 -> 256K` BF16 `last_loop` scaling curve (`36` banks, `k`-invariant). */
  readonly productionLastLoopCurve: readonly DoryProductionKvPoint[];
}

/** View 6 (`self_spec`) Contract A §6.2 self-speculation position, SWA ring & residual descriptors. */
export type DorySpecSwaRingSlotStatus =
  | 'empty'
  | 'committed_past'
  | 'committed_new'
  | 'rolled_back'
  | 'evicted_from_window';

export interface DorySpecSwaRingSlotState {
  readonly slotIndex: number;
  readonly absPos: number | null;
  readonly status: DorySpecSwaRingSlotStatus;
}

export interface DorySpecPositionBarDescriptor {
  readonly verifyOffset: number;
  readonly absPos: number;
  readonly inputTokenId: number;
  readonly inputTokenText: string;
  readonly proposedTokenId: number | null;
  readonly proposedTokenText: string | null;
  readonly isBonusSlot: boolean;
  readonly reusedPreludeAndLoop1: boolean;
  readonly proposalProb: number | null;
  readonly targetProb: number | null;
  readonly uniformVariate: number | null;
  readonly accepted: boolean | null;
  readonly rejectionReason: string | null;
  readonly rolledBack: boolean;
}

export interface DorySpecResidualTopToken {
  readonly tokenId: number;
  readonly tokenText: string;
  readonly residualProb: number;
  readonly targetProb: number;
  readonly proposalProb: number;
}

export interface DorySelfSpecViewMetadata {
  readonly contractBadge: typeof DORY_SPEC_CONTRACT_A_BADGE;
  readonly totalBankCount: typeof DORY_SPEC_CONTRACT_A_BANK_COUNT;
  /** Exact Contract A 104-bank partition: `8 + 20 + 60 + 8 + 8 = 104` (`D3`, `DR23`). */
  readonly inputBankCount: 8;
  readonly loop1SharedBankCount: 20;
  readonly loops2To4VerifierBankCount: 60;
  readonly outputDeepBankCount: 8;
  readonly outputShallowBankCount: 8;
  /** Extra persistent KV payload bytes for Contract A's `8` `output_shallow` Coda banks (`20,480 B` = `+20 KiB`). */
  readonly shallowCodaExtraBytes: number;
  /** Total persistent KV payload bytes across all `104` banks (`266,240 B` = `260 KiB`). */
  readonly totalKvPayloadBytes: number;
  /** True when View 6 synthesized an isolated read-only preview cycle from a standard session. */
  readonly isSynthesizedPreview: boolean;
  readonly selectedVerifyOffset: number;
  readonly selectedAttentionLayerIdx: number;
  /** True iff 1-token Shallow Coda catch-up ran on `s_1^{(L+gamma')}` (`false` on rejection `a < gamma'`). */
  readonly shallowCodaCatchupExecuted: boolean;
  /** Division-free binary64 acceptance rule (`Audit §8.2`, Finding 3 — zero `1e-38` clamp). */
  readonly divisionFreeBinary64Rule: 'p[d] >= q[d] || r * q[d] < p[d]';
  readonly hasLiveCausalMaskGChannel: true;
  readonly cycleSummary: DorySpecCycleSummary;
  readonly positionBars: readonly DorySpecPositionBarDescriptor[];
  readonly residualTopTokens: readonly DorySpecResidualTopToken[];
  readonly swaRingSlots: readonly DorySpecSwaRingSlotState[];
  readonly swaRingCapacity: number;
  readonly slidingWindow: number;
  readonly rolledBackPositions: number;
}

/** View 7 (`bookends`) untied bookends, 4-format readouts & read-only TBPTL descriptor. */
export interface DoryBookendsLogitRow {
  readonly tokenId: number;
  readonly tokenText: string;
  readonly rawLogit: number;
  readonly rawLogitScale: number;
  readonly effectiveLogitScale: number;
  readonly scaledLogit: number;
  readonly prob: number;
}

export interface DoryBookendsViewMetadata {
  readonly readOnly: true;
  readonly hasGradients: false;
  readonly tbptlBadgeLabel: typeof DORY_TBPTL_READONLY_BADGE;
  readonly untiedVerified: true;
  readonly selectedTokenPos: number;
  readonly selectedTokenId: number;
  readonly selectedTokenText: string;
  readonly topLogitRows: readonly DoryBookendsLogitRow[];
  readonly checkpointFormats: readonly DoryCheckpointFormatSpec[];
  readonly scalingDisclosure: readonly DoryScalingRatioEntry[];
}

// ============================================================================
// UNIFIED 5-PASS WEBGL2 SPATIAL SCENE PAYLOAD & SESSION CONTEXT
// ============================================================================

export interface DorySpatialScene {
  readonly viewMode: DorySpatialViewMode;
  readonly title: string;
  readonly subtitle: string;
  readonly modelWeightId: string;
  readonly branchId: string;
  readonly recordId: string;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  readonly physicalLayerIdx: number | null;
  readonly loopIteration: number | null;
  /** Pass 1 (`SolidsPass`): 12 floats/instance. */
  readonly solidData: Float32Array;
  readonly solidCount: number;
  /** Pass 2 (`LinesPass`): 18 floats/instance. */
  readonly lineData: Float32Array;
  readonly lineCount: number;
  /** Pass 3 (`RibbonsPass`): 12 floats/instance. */
  readonly ribbonData: Float32Array;
  readonly ribbonCount: number;
  /** Pass 4 (`SheetsPass`): 20 floats/instance + 6 floats basis + 4 floats reveal. */
  readonly sheetData: Float32Array;
  readonly sheetBasisData: Float32Array;
  readonly sheetRevealData: Float32Array;
  readonly sheetCount: number;
  /** Pass 5 (`FlatFlowRibbonsPass`): 10 floats/vertex + Uint32 triangle indices. */
  readonly flowVertices: Float32Array;
  readonly flowIndices: Uint32Array;
  readonly pickBoxes: readonly PickBox[];
  readonly cameraPreset: DoryCameraPreset;
  readonly fitBounds: AABB;
  /** Strictly 2-channel `RG32F` (`R = scalar, G = mask`) `<= 512 KiB`, or `null`. */
  readonly valueAtlas: Rg32fValueAtlas | null;
  /** Deterministic content-addressed key for GPU `RG32F` upload deduplication. */
  readonly atlasKey: string | null;
  readonly atlasRegions: readonly DoryAtlasRegionBox[];
  readonly atlasAudit: DoryAtlasLayoutAudit | null;
  readonly provenanceBadges: readonly DoryProvenanceBadge[];
  readonly callouts: readonly DorySceneCallout[];
  readonly overlayAnchors: readonly DoryOverlayAnchor[];
  readonly landmarks: readonly LandmarkAnchor[];
  readonly kvDepthTicks: readonly KvDepthTickAnchor[];
  readonly calipers: readonly DimensionCaliper[];
  /** Focused sheets eligible for zoom-adaptive `<= 64`-node on-cell numeric overlays (`DORY6`). */
  readonly activeCellGrids: readonly ActiveSheetCellGrid[];
  /** Active hovered/selected cell popover spec (`DORY6` `spatialOverlay.ts`). */
  readonly activeCellPopover: CellPopoverSpec | null;
  /** Optional view-specific analytical metadata payloads consumed by insets & Gate 3/4/5 tests. */
  readonly macroMetadata?: DoryMacroViewMetadata | null;
  readonly hypersphereMetadata?: DoryHypersphereViewMetadata | null;
  readonly gqaRopeMetadata?: DoryGqaRopeViewMetadata | null;
  readonly swigluMetadata?: DorySwigluViewMetadata | null;
  readonly kvCompareMetadata?: DoryKvCompareViewMetadata | null;
  readonly selfSpecMetadata?: DorySelfSpecViewMetadata | null;
  readonly bookendsMetadata?: DoryBookendsViewMetadata | null;
}

/**
 * Strictly non-mutating resolved spatial context backed by `DorySession`'s immutable snapshots,
 * 0-ULP lazy `reconstructStepTrace`, and zero-KV-mutation `evaluateCounterfactualIntermediateCoda`.
 */
export interface DoryResolvedSessionContext {
  readonly params: DoryModelParameters;
  readonly ropeTables: DoryDualRopeTables;
  readonly session: DorySession;
  readonly branch: DorySessionBranchSnapshot;
  readonly record: DorySessionRecord;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  readonly isHistoricalCursor: boolean;
  readonly recordIndexInBranch: number;
  readonly branchRecordCount: number;
  /** Lazily reconstructs and caches the 0-ULP full step trace for this scene build. */
  readonly resolveStepTrace: () => DoryReconstructedStepResult;
  /** Extracts the harmonized `DoryForwardTrace` (`executedSublayers[0..31+40k]`) for standard or spec records. */
  readonly resolveForwardTrace: () => DoryForwardTrace;
  /**
   * Resolves the exact executed sublayer record for `(physicalLayerIdx, loopIteration)`
   * (`loopIteration = 0` for Prelude/Coda, `1..kLoops` for Recurrent core).
   */
  readonly resolveSublayerRecord: (
    physicalLayerIdx: number,
    loopIteration?: number,
  ) => DoryExecutedSublayerRecord;
  /** Default token offset inside `resolveForwardTrace()` corresponding to the final committed token. */
  readonly resolveDefaultTokenOffset: () => number;
  /** Read-only View 2 intermediate-loop Diagnostic Counterfactual Coda (`writeKv: false`, zero KV mutation). */
  readonly resolveCounterfactualCoda: (loopIteration: number) => DoryCounterfactualCodaResult;
  readonly provenanceBadges: readonly DoryProvenanceBadge[];
}

// ============================================================================
// VIEW-SPECIFIC BUILD OPTIONS & UNIFIED SPATIAL FOCUS STATE (DORY3–DORY6)
// ============================================================================

export interface DoryMacroSceneOptions {
  /** `'folded'` (72 physical sublayers + feedback arc) or `'unrolled'` (`32 + 40k` executed stack). */
  readonly displayMode?: DoryMacroDisplayMode;
  /** Highlighted physical sublayer `0..71`, or `null`. */
  readonly selectedPhysicalLayerIdx?: number | null;
  /** Highlighted recurrent loop pass `1..k` (or `0` for Prelude/Coda), or `null`. */
  readonly selectedLoopIteration?: number | null;
  readonly hoveredPartId?: string | null;
  readonly hoveredCell?: DoryPickIdentity | null;
  readonly recordId?: string;
}

export interface DoryHypersphereSceneOptions {
  /** Selected physical sublayer `0..71` for the NLERP cutaway (default `16` — planted `|alpha_hat| > 1` layer!). */
  readonly physicalLayerIdx?: number;
  /** Selected recurrent loop iteration `1..k` (ignored / `0` for Prelude/Coda). */
  readonly loopIteration?: number;
  /** Token offset inside the reconstructed trace (defaults to final committed token). */
  readonly tokenOffset?: number;
  /** Whether to evaluate read-only `Diagnostic Counterfactual` Coda at `counterfactualLoop`. */
  readonly includeCounterfactualCoda?: boolean;
  /** Intermediate loop iteration `1..k` to probe with read-only Deep Coda (`writeKv: false`). */
  readonly counterfactualLoop?: number;
  readonly hoveredCell?: DoryPickIdentity | null;
  readonly recordId?: string;
}

export interface DoryGqaRopeSceneOptions {
  /** Even physical attention layer `0, 2, ..., 70` (default `16` SWA or `22` Full). */
  readonly physicalLayerIdx?: number;
  /** Recurrent loop iteration `1..k` (or `0` for Prelude/Coda). */
  readonly loopIteration?: number;
  /** Focused query head `0..7`. */
  readonly selectedQHead?: number;
  /** Focused query row offset inside the reconstructed chunk trace. */
  readonly selectedQueryRow?: number;
  /** Focused key column index `0..kvLen-1` for dual-RoPE delta phasor & score inspection. */
  readonly selectedKeyCol?: number;
  /** Focused NeoX rotary plane `0..3` highlighted on the 4-plane dual-RoPE inset. */
  readonly selectedRotaryPlane?: number;
  readonly hoveredCell?: DoryPickIdentity | null;
  readonly recordId?: string;
}

export interface DorySwigluSceneOptions {
  /** Odd physical MLP layer `1, 3, ..., 71` (default `17` first Recurrent MLP). */
  readonly physicalLayerIdx?: number;
  /** Recurrent loop iteration `1..k` (or `0` for Prelude/Coda). */
  readonly loopIteration?: number;
  readonly selectedTokenRow?: number;
  /** Focused intermediate channel `0..23`. */
  readonly selectedChannel?: number;
  /** Focused `down_proj` output channel `0..7` for progressive accumulation reveal. */
  readonly selectedOutputDim?: number;
  /** Progressive accumulation reveal in `[0, 1]` (default `1.0` = fully revealed). */
  readonly accumReveal?: number;
  readonly hoveredCell?: DoryPickIdentity | null;
  readonly recordId?: string;
}

export interface DoryKvCompareSceneOptions {
  /** Prefill partition mode to compare against `per_loop` (`'single_shot' | 'chunked' | 'one_by_one'`). */
  readonly partitionMode?: DoryKvComparePartitionMode;
  /** Optional explicit prompt partition for `'chunked'` comparison (defaults to `[4, remaining]`). */
  readonly chunkPartition?: readonly number[];
  /** Focused physical attention layer `even 0..70` for KV bank & causal plate inspection (default `16`). */
  readonly physicalLayerIdx?: number;
  /** Focused recurrent loop pass `1..k` for `per_loop` vs. `last_loop` plate inspection (default `1`). */
  readonly loopIteration?: number;
  /** Optional loop-budget override for isolated dual-session comparison (defaults to active session `kLoops`). */
  readonly kLoopsOverride?: number;
  /** Optional explicit token sequence for isolated dual-session comparison (defaults to session committed tokens). */
  readonly comparisonTokens?: readonly number[];
  readonly hoveredCell?: DoryPickIdentity | null;
  readonly recordId?: string;
}

export interface DorySelfSpecSceneOptions {
  /** Focused verify position offset `0..verifiedPositions-1`. */
  readonly selectedVerifyOffset?: number;
  /** Focused even physical attention layer `0..70` for live `DR20` verify attention plate (default `16`). */
  readonly selectedAttentionLayerIdx?: number;
  /** Optional sampling override when synthesizing an isolated preview cycle from a standard session. */
  readonly previewSampling?: Partial<DorySamplingConfig>;
  readonly hoveredCell?: DoryPickIdentity | null;
  readonly recordId?: string;
}

export interface DoryBookendsSceneOptions {
  /** Focused sequence position for embedding / LM-head inspection. */
  readonly selectedTokenPos?: number;
  /** Selected production checkpoint row in the 4-format read-only table. */
  readonly selectedCheckpointFormat?: 'bf16' | 'mxfp8' | 'nvfp4' | 'q5_k_m';
  readonly hoveredCell?: DoryPickIdentity | null;
  readonly recordId?: string;
}

// ============================================================================
// FROZEN SCENE BUILDER SIGNATURES (DORY3 / DORY4 / DORY5 DISJOINT SLICES)
// ============================================================================
// Canonical module exports:
//   - View 1 (`src/dory/scene/macroScene.ts`):       `buildDoryMacroScene`
//   - View 2 (`src/dory/scene/hypersphereScene.ts`): `buildDoryHypersphereScene`
//   - View 3 (`src/dory/scene/gqaRopeScene.ts`):     `buildDoryGqaRopeScene`
//   - View 4 (`src/dory/scene/swigluScene.ts`):      `buildDorySwigluScene`
//   - View 5 (`src/dory/scene/kvCompareScene.ts`):   `buildDoryKvCompareScene`
//   - View 6 (`src/dory/scene/selfSpecScene.ts`):    `buildDorySelfSpecScene`
//   - View 7 (`src/dory/scene/bookendsScene.ts`):    `buildDoryBookendsScene`

export type DorySceneBuilder<TOptions> = (
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  options?: TOptions,
) => DorySpatialScene;

export type DoryMacroSceneBuilder = DorySceneBuilder<DoryMacroSceneOptions>;
export type DoryHypersphereSceneBuilder = DorySceneBuilder<DoryHypersphereSceneOptions>;
export type DoryGqaRopeSceneBuilder = DorySceneBuilder<DoryGqaRopeSceneOptions>;
export type DorySwigluSceneBuilder = DorySceneBuilder<DorySwigluSceneOptions>;
export type DoryKvCompareSceneBuilder = DorySceneBuilder<DoryKvCompareSceneOptions>;
export type DorySelfSpecSceneBuilder = DorySceneBuilder<DorySelfSpecSceneOptions>;
export type DoryBookendsSceneBuilder = DorySceneBuilder<DoryBookendsSceneOptions>;

/** Unified focus state passed to `buildDorySpatialScene` across all 7 Dory spatial views. */
export interface DorySpatialFocusState {
  readonly macroDisplayMode: DoryMacroDisplayMode;
  readonly selectedPhysicalLayerIdx: number;
  readonly selectedLoopIteration: number;
  readonly selectedTokenOffset: number | null;
  readonly includeCounterfactualCoda: boolean;
  readonly counterfactualLoop: number;
  readonly gqaLayerIdx: number;
  readonly selectedQHead: number;
  readonly selectedQueryRow: number | null;
  readonly selectedKeyCol: number | null;
  readonly selectedRotaryPlane: number;
  readonly swigluLayerIdx: number;
  readonly swigluChannel: number;
  readonly swigluOutputDim: number;
  readonly swigluAccumReveal: number;
  readonly kvComparePartitionMode: DoryKvComparePartitionMode;
  readonly kvCompareLayerIdx: number;
  readonly selectedVerifyOffset: number;
  readonly selectedCheckpointFormat: 'bf16' | 'mxfp8' | 'nvfp4' | 'q5_k_m';
  readonly recordId?: string;
  readonly hoveredPartId: string | null;
  readonly hoveredCell: DoryPickIdentity | null;
}

export const DEFAULT_DORY_SPATIAL_FOCUS_STATE: DorySpatialFocusState = Object.freeze({
  macroDisplayMode: 'folded',
  selectedPhysicalLayerIdx: 16,
  selectedLoopIteration: 1,
  selectedTokenOffset: null,
  includeCounterfactualCoda: true,
  counterfactualLoop: 1,
  gqaLayerIdx: 16,
  selectedQHead: 0,
  selectedQueryRow: null,
  selectedKeyCol: null,
  selectedRotaryPlane: 0,
  swigluLayerIdx: 17,
  swigluChannel: 0,
  swigluOutputDim: 0,
  swigluAccumReveal: 1.0,
  kvComparePartitionMode: 'chunked',
  kvCompareLayerIdx: 16,
  selectedVerifyOffset: 0,
  selectedCheckpointFormat: 'bf16',
  hoveredPartId: null,
  hoveredCell: null,
});
