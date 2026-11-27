/**
 * Shared Headless Spatial Scene, Focus, Pick, Atlas & RGB Preview Contracts
 * (`src/uni/scene/types.ts`)
 *
 * Pinned Step-0 contract shared across all 7 NLD-Uni spatial views:
 *   - UNI4 (Part 1): `macro` (View 1), `visual_input` (View 2), `joint_attn` (View 3)
 *   - UNI5 (Part 2): `ball_tree` (View 4), `refinement` (View 5), `output_edit` (View 6),
 *                    `opd_readonly` (View 7)
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import type { Vec3 } from '../../render/primitives';
import type { AABB } from '../../render/camera';
import type { PickBox } from '../../render/pick';
import type { Rg32fValueAtlas } from '../../scene/atlasPacker';
import type { DimensionCaliper, KvDepthTickAnchor, LandmarkAnchor } from '../../render/primitives';
import type { UniFullRuntimeBundle } from '../imageSampler';
import type { UniReconstructedHeadLogits, UniSession } from '../session';
import type {
  UniCfgBatchSize,
  UniEditGuidanceMode,
  UniElasticD5ComparisonReadout,
  UniElasticPrefillMode,
  UniExecutionViewId,
  UniFlowEditingOpdResult,
  UniGainPresetId,
  UniGroundingBox,
  UniLayerTrace,
  UniPrimeDepthFloorMismatchReport,
  UniRgbImage,
  UniScenePresetId,
  UniSessionBranch,
  UniSessionRecord,
  UniSourcePaddingVqMask,
  UniTrainerBugDiagnosticU28,
  UniUnderstandingOpdResult,
  UniViewArchReadout,
} from '../types';

// ============================================================================
// SEVEN ARCHITECTURE-SPECIFIC NLD-UNI SPATIAL VIEW MODES (§9)
// ============================================================================

export type UniSpatialViewMode =
  | 'macro'
  | 'visual_input'
  | 'joint_attn'
  | 'ball_tree'
  | 'refinement'
  | 'output_edit'
  | 'opd_readonly';

export interface UniSpatialViewDescriptor {
  readonly id: UniSpatialViewMode;
  readonly viewNumber: 1 | 2 | 3 | 4 | 5 | 6 | 7;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
  readonly atlasBudgetBytes: number;
}

export const UNI_SPATIAL_VIEW_MODES: readonly UniSpatialViewDescriptor[] = Object.freeze([
  Object.freeze({
    id: 'macro',
    viewNumber: 1,
    label: 'Elastic Macro Architecture',
    shortLabel: 'Macro 34L',
    description:
      'Twin 34-layer Understanding (U) and Generation (G) towers bridged by 18 joint-attention couplers (L0..17) and split at L18..33 with live D5 HF-Faithful vs. Elastic-Pruned KV waste toggle.',
    atlasBudgetBytes: 0,
  }),
  Object.freeze({
    id: 'visual_input',
    viewNumber: 2,
    label: 'Visual Input Pipeline',
    shortLabel: 'Visual Input',
    description:
      'Side-by-side Mini RADIO-H continuous ViT + RadioHProjector (U) and Mini Emu3 IBQ VQ-VAE + PackedLinearImageCodec (G) feeding multimodal sequence assembly.',
    atlasBudgetBytes: 131_072,
  }),
  Object.freeze({
    id: 'joint_attn',
    viewNumber: 3,
    label: 'Joint vs. Split Attention & Dual-SwiGLU',
    shortLabel: 'Joint / Split',
    description:
      'Routed U/G QKV projections, 1D dyadic YaRN RoPE + Llama-4 scaling, live RG32F G-channel joint-to-split mask transition (L17 -> L18), and routed 3.5x SwiGLU.',
    atlasBudgetBytes: 524_288,
  }),
  Object.freeze({
    id: 'ball_tree',
    viewNumber: 4,
    label: 'Codebook Geometry & Ball-Tree',
    shortLabel: 'Ball-Tree',
    description:
      '3D PCA geometry of the 256-code VQ codebook overlaid with depth-8 BalancedSemanticTree nested candidate balls and U26 production D=17 floor-mismatch comparison.',
    atlasBudgetBytes: 131_072,
  }),
  Object.freeze({
    id: 'refinement',
    viewNumber: 5,
    label: 'Image Refinement Canvas',
    shortLabel: 'Refinement',
    description:
      '4x6 marked VQ child plate coupled via 2x2 PackedLinearImageCodec brackets to 6 parent + 4 fixed-prefix tokens, stage/age polarity (U9), and additive mask noise (U10).',
    atlasBudgetBytes: 131_072,
  }),
  Object.freeze({
    id: 'output_edit',
    viewNumber: 6,
    label: 'Output, Grounding & Editing Pipeline',
    shortLabel: 'Output & Edit',
    description:
      'Task-reactive output stage: Dual-CFG editing & lock_source_padding_vq un-pad (T2I/Edit) or FLARE block-diffusion unmasking & <LOC_0..16> pixel_box grounding (U27).',
    atlasBudgetBytes: 131_072,
  }),
  Object.freeze({
    id: 'opd_readonly',
    viewNumber: 7,
    label: 'Cross-Paradigm OPD Explorer (Read-Only)',
    shortLabel: 'OPD Ref',
    description:
      'Forward-only (0 gradients) Transitive Understanding 129-bucket tail_kl, Continuous-Flow Editing matched_sigmas(0.9), and U28 reference-trainer diagnostics.',
    atlasBudgetBytes: 65_536,
  }),
]);

// ============================================================================
// PROVENANCE BADGES, ATLAS AUDIT, RGB PREVIEW & PICK IDENTITY CONTRACTS
// ============================================================================

export type UniBadgeTone =
  | 'u_indigo'
  | 'g_teal'
  | 'joint'
  | 'split'
  | 'warn'
  | 'muted'
  | 'gold'
  | 'readonly';

export interface UniProvenanceBadge {
  readonly label: string;
  readonly tone: UniBadgeTone;
}

export interface UniAtlasRegionBox {
  readonly id: string;
  readonly x0: number;
  readonly y0: number;
  readonly cols: number;
  readonly rows: number;
  readonly mode: 0 | 1;
  readonly isMaskOrScore: boolean;
}

export interface UniAtlasLayoutAudit {
  readonly width: number;
  readonly height: number;
  readonly totalTexels: number;
  readonly usedTexels: number;
  readonly utilizationPct: number;
  readonly byteLength: number;
  readonly maxBudgetBytes: number;
  readonly withinBudget: boolean;
  readonly regions: readonly UniAtlasRegionBox[];
  readonly hasOverlap: boolean;
  readonly overlapPair: readonly [string, string] | null;
  readonly outOfBoundsIds: readonly string[];
}

export type UniRgbPreviewKind =
  | 'source'
  | 'letterboxed'
  | 'decoded'
  | 'cropped'
  | 'grounding';

/**
 * Typed 3-channel RGB preview descriptor consumed by UNI6's 2D `<canvas>` `pixelCanvas.ts`.
 * Enforces D4: zero 3-channel RGB pixels are ever packed into 2-channel `RG32F` WebGL2 atlases.
 */
export interface UniRgbPreviewDescriptor {
  readonly id: string;
  readonly kind: UniRgbPreviewKind;
  readonly label: string;
  readonly caption: string;
  readonly width: number;
  readonly height: number;
  /** Row-major CHW `[3, height, width]` Float32 pixels in `[0, 1]`. */
  readonly rgbChw: Float32Array;
  /** Optional `4x4` VQ padding-lock overlay mask (`U18`, `1` = locked border cell). */
  readonly paddingLockMask?: UniSourcePaddingVqMask;
  /** Optional `<LOC_0..16>` bounding boxes (`U27`). */
  readonly groundingBoxes?: readonly UniGroundingBox[];
  readonly honestyBadge: string;
}

/**
 * Full NLD-Uni pick identity decoded from shared `PickHit` with ZERO edits to `src/render/pick.ts`.
 * `PickBox.groupIndex` carries KV head (`0..1`) or CFG row (`0..2`); `PickBox.tapIndex` carries
 * rotary plane (`0..3`) or semantic-tree depth (`0..8`).
 */
export interface UniPickIdentity {
  readonly viewMode: UniSpatialViewMode;
  readonly layerIndex: number | null;
  readonly isSplitLayer: boolean | null;
  readonly branch: 'u' | 'g' | 'joint' | null;
  readonly partId: string | null;
  readonly tensorKey: string | null;
  readonly qHeadIndex: number | null;
  readonly kvHeadIndex: number | null;
  readonly cfgBatchRow: number | null;
  readonly rotaryPlane: number | null;
  readonly treeDepth: number | null;
  readonly tokenIndex: number | null;
  readonly absPosition: number | null;
  readonly row: number | null;
  readonly col: number | null;
  readonly semanticTag: string | null;
}

export interface UniOverlayAnchor {
  readonly id: string;
  readonly label: string;
  readonly sublabel?: string;
  readonly worldPos: Vec3;
  readonly tone?: UniBadgeTone;
  readonly badge?: string;
}

export interface UniCameraPreset {
  readonly azimuthDeg: number;
  readonly elevationDeg: number;
  readonly fitBounds: AABB;
}

export interface UniSceneValidationResult {
  readonly valid: boolean;
  readonly errors: readonly string[];
}

// ============================================================================
// UNIFIED 5-PASS WEBGL2 SPATIAL SCENE PAYLOAD & SESSION CONTEXT
// ============================================================================

export interface UniSpatialScene {
  readonly viewMode: UniSpatialViewMode;
  readonly title: string;
  readonly subtitle: string;
  readonly layerIndex: number | null;
  readonly executionViewId: UniExecutionViewId;
  readonly prefillMode: UniElasticPrefillMode;
  readonly solidData: Float32Array;
  readonly solidCount: number;
  readonly lineData: Float32Array;
  readonly lineCount: number;
  readonly ribbonData: Float32Array;
  readonly ribbonCount: number;
  readonly sheetData: Float32Array;
  readonly sheetBasisData: Float32Array;
  readonly sheetRevealData: Float32Array;
  readonly sheetCount: number;
  readonly flowVertices: Float32Array;
  readonly flowIndices: Uint32Array;
  readonly pickBoxes: readonly PickBox[];
  readonly cameraPreset: UniCameraPreset;
  readonly fitBounds: AABB;
  /** Strictly 2-channel `RG32F` (`R = scalar, G = mask`) or `null` (View 1 Macro). */
  readonly valueAtlas: Rg32fValueAtlas | null;
  /** Content-addressed dual-hash key for GPU `RG32F` texture upload deduplication. */
  readonly atlasKey: string | null;
  readonly atlasRegions: readonly UniAtlasRegionBox[];
  readonly atlasAudit: UniAtlasLayoutAudit | null;
  /** Dedicated 3-channel RGB insets rendered by `UNI6`'s 2D `<canvas>` `pixelCanvas.ts`. */
  readonly rgbPreviews: readonly UniRgbPreviewDescriptor[];
  readonly provenanceBadges: readonly UniProvenanceBadge[];
  readonly overlayAnchors: readonly UniOverlayAnchor[];
  readonly landmarks: readonly LandmarkAnchor[];
  readonly kvDepthTicks: readonly KvDepthTickAnchor[];
  readonly calipers: readonly DimensionCaliper[];
  readonly d5Comparison: UniElasticD5ComparisonReadout | null;
  readonly viewReadout: UniViewArchReadout | null;
  /** Optional View 4 (`ball_tree`) U26 production D=17 vs miniature D=8 floor-mismatch diagnostic. */
  readonly u26Mismatch?: UniPrimeDepthFloorMismatchReport | null;
  /** Optional View 7 (`opd_readonly`) strictly forward-only, zero-gradient OPD reference summary. */
  readonly opdReadOnly?: UniOpdReadOnlyMetadata | null;
}

/** Strictly forward-only, zero-gradient diagnostic metadata exposed by View 7 (`opd_readonly`). */
export interface UniOpdReadOnlyMetadata {
  readonly readOnly: true;
  readonly hasGradients: false;
  readonly badgeLabel: 'READ-ONLY · NO GRADIENTS';
  readonly understandingOpd: UniUnderstandingOpdResult;
  readonly flowEditingOpd: UniFlowEditingOpdResult;
  readonly u28Diagnostics: UniTrainerBugDiagnosticU28;
}

export interface UniResolvedSessionContext {
  readonly bundle: UniFullRuntimeBundle;
  readonly session: UniSession;
  readonly branch: UniSessionBranch;
  readonly record: UniSessionRecord;
  readonly executionViewId: UniExecutionViewId;
  readonly prefillMode: UniElasticPrefillMode;
  readonly gainPresetId: UniGainPresetId;
  readonly cfgBatchSize: UniCfgBatchSize;
  readonly isHistoricalCursor: boolean;
  readonly recordIndexInBranch: number;
  readonly branchRecordCount: number;
  readonly resolveLayerTrace: (layerIndex: number, batchRow?: number) => UniLayerTrace;
  readonly resolveStepHeadLogits: () => UniReconstructedHeadLogits;
  readonly provenanceBadges: readonly UniProvenanceBadge[];
  readonly d5Comparison: UniElasticD5ComparisonReadout;
  readonly viewReadout: UniViewArchReadout;
}

// ============================================================================
// VIEW-SPECIFIC BUILD OPTIONS (UNI4 VIEWS 1–3)
// ============================================================================

export interface UniMacroSceneOptions {
  /** Explicit execution view override (defaults to active session branch's view). */
  readonly executionViewId?: UniExecutionViewId;
  /** Live D5 prefill mode override (`'hf_faithful' | 'elastic_pruned'`). */
  readonly prefillMode?: UniElasticPrefillMode;
  /** Highlighted zero-based layer index (`0..33`), or `null`. */
  readonly selectedLayerIndex?: number | null;
  readonly hoveredPartId?: string | null;
  /** Prefix token length used for D5 KV waste calculations (defaults to active prefix cache seqLen or 16). */
  readonly prefixTokens?: number;
  /** Optional explicit historical record ID. */
  readonly recordId?: string;
}

export interface UniJointAttnSceneOptions {
  /** Zero-based layer index (`0..33`; presets `0, 17, 18, 33`). Defaults to `17` or `18`. */
  readonly layerIndex?: number;
  /** CFG batch row (`0..cfgBatchSize-1`; defaults to highest conditioned row `cfgBatchSize - 1`). */
  readonly batchRow?: number;
  /** Live D5 comparison mode (`'hf_faithful' | 'elastic_pruned'`). */
  readonly prefillMode?: UniElasticPrefillMode;
  /** Focused query head (`0..7`). */
  readonly selectedQHead?: number;
  /** Focused query token row within active pass (`0..S-1`). */
  readonly selectedTokenRow?: number;
  /** Optional explicit historical record ID. */
  readonly recordId?: string;
  readonly hoveredCell?: UniPickIdentity | null;
}

export interface UniVisualInputSceneOptions {
  /** Optional explicit procedural RGB preset (`'sunlit_cat_16x16' | 'harbor_panorama_16x8' | 'geometric_blocks_16x16'`). */
  readonly scenePresetId?: UniScenePresetId;
  /** Optional explicit RGB image override. */
  readonly sourceImage?: UniRgbImage;
  /** Dual gain preset (`'mini_default'` `0.25` / `1.0/0.5` vs `'prod_exact'` `0.01` / prod). */
  readonly gainPresetId?: UniGainPresetId;
  /** Force discrete Emu3VQ + PackedLinearImageCodec stream visible even in understanding mode (defaults to task-reactive or true when preset shown). */
  readonly includeDiscreteStream?: boolean;
  /** Optional explicit historical record ID. */
  readonly recordId?: string;
  readonly hoveredCell?: UniPickIdentity | null;
}

// ============================================================================
// VIEW-SPECIFIC BUILD OPTIONS (UNI5 VIEWS 4–7)
// ============================================================================

export type UniRefinementStepPreset = 1 | 2 | 4 | 8;

export interface UniBallTreeSceneOptions {
  /** Target VQ codebook entry (`0..255`) whose nested candidate balls are highlighted. */
  readonly selectedCodeId?: number;
  /** Progressive self-refinement step count `N ∈ {1, 2, 4, 8}` (`N = 1` reduces to binary MDM). */
  readonly refinementSteps?: UniRefinementStepPreset;
  /** Active code stage (`-1` masked, `0..N` progressively refined; `N` = finalized leaf). */
  readonly stageIndex?: number;
  /** Emphasize the interactive `U26` production `D = 17` train-vs-inference floor-rounding callout. */
  readonly showProductionD17Comparison?: boolean;
  /** Optional explicit historical record ID (used when resolving defaults from a live `UniSession`). */
  readonly recordId?: string;
  readonly hoveredCell?: UniPickIdentity | null;
}

export interface UniRefinementSceneOptions {
  /** Optional explicit historical record ID for full trajectory scrubbing (`step 0 .. N_steps`). */
  readonly recordId?: string;
  /** Highlighted cell index inside the `4×6` marked VQ child grid (`0..23`), or `null`. */
  readonly selectedGridCell?: number | null;
  /** Toggle `stratified_random(4, seed=42)` quadtree unmasking-order overlay. */
  readonly showStratifiedOrder?: boolean;
  /** Focused CFG branch row (`0..cfgBatchSize-1`) for top-32 logit comparison emphasis. */
  readonly cfgBranchRow?: number;
  readonly hoveredCell?: UniPickIdentity | null;
}

export interface UniOutputEditSceneOptions {
  /** Optional explicit historical record ID. */
  readonly recordId?: string;
  /** Dual-CFG editing mixer override (`'legacy' | 'independent' | 'ip2p'`) for comparison slices. */
  readonly editGuidanceMode?: UniEditGuidanceMode;
  /** Whether to emit the center-cropped native-aspect RGB preview (e.g., `16×8` on panorama edits). */
  readonly cropToSource?: boolean;
  /** Toggle explicit `U18` `lock_source_padding_vq` border-lock visual overlay. */
  readonly showPaddingLockOverlay?: boolean;
  readonly hoveredCell?: UniPickIdentity | null;
}

export type UniOpdBayFocus = 'both' | 'understanding' | 'editing';

export interface UniOpdSceneOptions {
  /** Deterministic PRNG seed for `createDefaultOpdDemoInputs(seed)` (defaults to `0x4f504431`). */
  readonly seed?: number;
  /** Continuous-flow editing noising ratio `r ∈ (0, 1)` (default `0.9`). */
  readonly r?: number;
  /** Continuous-flow editing ODE schedule step count for `matched_sigmas` (default `40`). */
  readonly steps?: number;
  /** Sequence length for Qwen-Image dynamic time-shift `mu(seqLen)` (default `1024`). */
  readonly seqLen?: number;
  /** Optional focus emphasis on left bay (`understanding`), right bay (`editing`), or `both`. */
  readonly selectedBay?: UniOpdBayFocus;
  readonly hoveredCell?: UniPickIdentity | null;
}

/** Unified focus state passed to `buildUniSpatialScene` across all 7 NLD-Uni spatial views. */
export interface UniSpatialFocusState {
  readonly executionViewId?: UniExecutionViewId;
  readonly prefillMode: UniElasticPrefillMode;
  readonly selectedLayerIndex: number | null;
  readonly jointLayerIndex: number;
  readonly batchRow: number;
  readonly selectedQHead: number;
  readonly selectedTokenRow: number;
  readonly scenePresetId?: UniScenePresetId;
  readonly gainPresetId?: UniGainPresetId;
  readonly includeDiscreteStream?: boolean;
  readonly selectedCodeId: number;
  readonly refinementSteps: UniRefinementStepPreset;
  readonly stageIndex: number;
  readonly showProductionD17Comparison: boolean;
  readonly selectedGridCell: number | null;
  readonly showStratifiedOrder: boolean;
  readonly cfgBranchRow: number;
  readonly editGuidanceMode?: UniEditGuidanceMode;
  readonly cropToSource: boolean;
  readonly showPaddingLockOverlay: boolean;
  readonly opdSeed: number;
  readonly opdR: number;
  readonly opdSteps: number;
  readonly opdSeqLen: number;
  readonly opdSelectedBay: UniOpdBayFocus;
  readonly recordId?: string;
  readonly hoveredPartId: string | null;
  readonly hoveredCell: UniPickIdentity | null;
}

export const DEFAULT_UNI_SPATIAL_FOCUS_STATE: UniSpatialFocusState = Object.freeze({
  prefillMode: 'elastic_pruned',
  selectedLayerIndex: 17,
  jointLayerIndex: 17,
  batchRow: 0,
  selectedQHead: 0,
  selectedTokenRow: 0,
  selectedCodeId: 42,
  refinementSteps: 4,
  stageIndex: 2,
  showProductionD17Comparison: true,
  selectedGridCell: 0,
  showStratifiedOrder: true,
  cfgBranchRow: 0,
  cropToSource: true,
  showPaddingLockOverlay: true,
  opdSeed: 0x4f504431,
  opdR: 0.9,
  opdSteps: 40,
  opdSeqLen: 1024,
  opdSelectedBay: 'both',
  hoveredPartId: null,
  hoveredCell: null,
});
