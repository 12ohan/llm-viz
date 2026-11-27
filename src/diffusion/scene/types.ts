/**
 * Shared Headless Spatial Scene, Focus, Pick & Atlas Contracts (`src/diffusion/scene/types.ts`)
 *
 * Pinned Step-0 contract shared across all 7 DF3 spatial views (`macro`, `gqa_rope`,
 * `swiglu`, `mask_explorer`, `spec_verify`, `embedding`, `output`), browser shell
 * (`src/diffusion/app/*`), and spatial verification suites (`tests/diffusion-spatial-*.test.ts`).
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import type { AABB } from '../../render/camera';
import type { PickBox } from '../../render/pick';
import type { Rg32fValueAtlas } from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import type { LayerFormulaReadout, StepperStepDef } from '../../scene/sheetInspector';
import type { DimensionCaliper, KvDepthTickAnchor, LandmarkAnchor } from '../../render/primitives';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../shared/overlayLayout';
import type { DiffusionSession } from '../session';
import type {
  DiffusionExecutionMode,
  DiffusionExplorerMaskMode,
  DiffusionLayerTrace,
  DiffusionModelParameters,
  DiffusionSessionRecord,
  KvPositionSlab,
  PersistentKvSnapshot,
  YarnPlaneZone,
} from '../types';

// ============================================================================
// SEVEN ARCHITECTURE-SPECIFIC SPATIAL VIEW MODES
// ============================================================================

export type DiffusionSpatialViewMode =
  | 'macro'
  | 'gqa_rope'
  | 'swiglu'
  | 'mask_explorer'
  | 'spec_verify'
  | 'embedding'
  | 'output';

export interface DiffusionSpatialViewDescriptor {
  readonly id: DiffusionSpatialViewMode;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
}

export const DIFFUSION_SPATIAL_VIEW_MODES: readonly DiffusionSpatialViewDescriptor[] =
  Object.freeze([
    Object.freeze({
      id: 'macro',
      label: '40L Backbone Macro',
      shortLabel: 'Macro 40L',
      description:
        'Single physical 40-layer homogeneous dense decoder spine with 0..31 KV rail and detachable draft-only o_proj LoRA bridge.',
    }),
    Object.freeze({
      id: 'gqa_rope',
      label: 'RoPE + YaRN + GQA Cutaway',
      shortLabel: 'RoPE GQA',
      description:
        '32Q/8KV (4:1) grouped-query attention with 4 dyadic YaRN rotary planes, live Llama-4 query scale, and draft-only o_proj LoRA delta.',
    }),
    Object.freeze({
      id: 'swiglu',
      label: 'Dense SwiGLU Cutaway',
      shortLabel: 'SwiGLU',
      description:
        'Post-attention RMSNorm into parallel W_gate -> SiLU and W_up projections, Hadamard product, W_down, and residual add.',
    }),
    Object.freeze({
      id: 'mask_explorer',
      label: '5-Tab Attention Mask Explorer',
      shortLabel: 'Masks',
      description:
        'Drafting plate comparing Joint Training [x_t || x_0], AR Causal, Block-Diff Bidir, Causal Commit/Verify, and VLM Quadratic SS.',
    }),
    Object.freeze({
      id: 'spec_verify',
      label: 'Speculative Verify & KV Strip',
      shortLabel: 'Spec & KV',
      description:
        'Shifted draft-vs-causal-verify alignment connectors coupled to the 0..31 KV cache lifecycle strip and D19 dual-crop provenance.',
    }),
    Object.freeze({
      id: 'embedding',
      label: 'Input Embedding Cutaway',
      shortLabel: 'Embedding',
      description:
        'Untied V=36 encoder.embed_tokens lookup highlighting [MASK], <|im_end|>, <s>, and </think> (positional injection via per-layer RoPE).',
    }),
    Object.freeze({
      id: 'output',
      label: 'Output diffusion_head Cutaway',
      shortLabel: 'Output Head',
      description:
        'Final RMSNorm -> untied diffusion_head contrasting causal shifted (+1) vs. diffusion same-position (+0) logits and un-suppressed [MASK] softmax.',
    }),
  ]);

// ============================================================================
// PROVENANCE, ATLAS AUDIT & PICKING IDENTITY
// ============================================================================

export type DiffusionBadgeTone =
  | 'causal'
  | 'bidir'
  | 'lora'
  | 'warn'
  | 'muted'
  | 'gold'
  | 'emerald';

export interface DiffusionProvenanceBadge {
  readonly label: string;
  readonly tone: DiffusionBadgeTone;
}

export interface DiffusionAtlasRegionBox {
  readonly id: string;
  readonly x0: number;
  readonly y0: number;
  readonly cols: number;
  readonly rows: number;
  readonly mode: 0 | 1;
  readonly isMaskOrScore: boolean;
}

export interface DiffusionAtlasLayoutAudit {
  readonly width: number;
  readonly height: number;
  readonly totalTexels: number;
  readonly usedTexels: number;
  readonly utilizationPct: number;
  readonly byteLength: number;
  readonly regions: readonly DiffusionAtlasRegionBox[];
  readonly hasOverlap: boolean;
  readonly overlapPair: readonly [string, string] | null;
  readonly outOfBoundsIds: readonly string[];
}

/**
 * Full Diffusion pick identity resolved from shared `PickHit` WITHOUT modifying `src/render/pick.ts`.
 * Diffusion's 8 KV heads (`0..7`) are carried in `PickBox.groupIndex` and decoded here into `kvHeadIndex`.
 */
export interface DiffusionPickIdentity {
  readonly viewMode: DiffusionSpatialViewMode;
  readonly layerIndex: number | null;
  readonly partId: string | null;
  readonly tensorKey: string | null;
  readonly qHeadIndex: number | null;
  /** Logical KV head index (`0..7`), decoded from `PickBox.groupIndex`. */
  readonly kvHeadIndex: number | null;
  /** Rotary plane index (`0..3`) when picking rotary/phasor sheets. */
  readonly rotaryPlane: number | null;
  readonly tokenIndex: number | null;
  readonly absPosition: number | null;
  readonly row: number | null;
  readonly col: number | null;
  readonly semanticTag: string | null;
}

export interface DiffusionFocusState {
  readonly viewMode: DiffusionSpatialViewMode;
  /** Zero-based layer index (`0..39`). */
  readonly selectedLayerIndex: number;
  /** Active query head (`0..31`). */
  readonly selectedQHead: number;
  /** Active KV head (`0..7`, typically `Math.floor(selectedQHead / 4)`). */
  readonly selectedKvHead: number;
  /** Active rotary plane (`0..3`: `(0,4)..(3,7)`). */
  readonly selectedRotaryPlane: number;
  /** Active Mask Explorer tab. */
  readonly selectedMaskTab: DiffusionExplorerMaskMode;
  readonly selectedPartId: string | null;
  readonly hoveredPartId: string | null;
  readonly hoveredCell: DiffusionPickIdentity | null;
  readonly pinnedCell: DiffusionPickIdentity | null;
  readonly activeStepperStep: number;
  /** Progressive accumulation reveal in `[0..1]` (`1` = 100% revealed) for SwiGLU cutaway. */
  readonly swigluAccumReveal: number;
  /** Focused query token row index within the active pass (`0..S-1`). */
  readonly selectedTokenRow: number;
}

export const DEFAULT_DIFFUSION_FOCUS_STATE: DiffusionFocusState = Object.freeze({
  viewMode: 'macro',
  selectedLayerIndex: 0,
  selectedQHead: 0,
  selectedKvHead: 0,
  selectedRotaryPlane: 0,
  selectedMaskTab: 'joint_training',
  selectedPartId: null,
  hoveredPartId: null,
  hoveredCell: null,
  pinnedCell: null,
  activeStepperStep: 0,
  swigluAccumReveal: 1.0,
  selectedTokenRow: 0,
});

// ============================================================================
// 2D UNIT-CIRCLE ROPE PHASOR TELEMETRY (for `gqa_rope` SVG inset)
// ============================================================================

export interface DiffusionRopePhasorTelemetry {
  readonly layerIndex: number;
  readonly qHeadIndex: number;
  readonly kvHeadIndex: number;
  readonly rotaryPlane: number; // 0..3
  readonly invFreq: number;
  readonly yarnZone: YarnPlaneZone;
  readonly queryPosition: number;
  readonly keyPosition: number;
  readonly angleRad: number;
  readonly relAngleRad: number;
  readonly llama4Scale: number;
  readonly origMaxPos: number;
  readonly qPrePair: readonly [number, number];
  readonly qRotPair: readonly [number, number];
  readonly qScaledPair: readonly [number, number];
  readonly kPrePair: readonly [number, number];
  readonly kRotPair: readonly [number, number];
  readonly allPlaneInvFreqs: readonly [number, number, number, number];
  readonly allPlaneZones: readonly YarnPlaneZone[];
}

// ============================================================================
// UNIFIED 5-PASS WEBGL2 SPATIAL SCENE PAYLOAD & SESSION CONTEXT
// ============================================================================

export interface DiffusionSpatialScene {
  readonly viewMode: DiffusionSpatialViewMode;
  readonly title: string;
  readonly subtitle: string;
  readonly layerIndex: number | null;
  readonly solidData: Float32Array;
  readonly solidCount: number;
  readonly lineData: Float32Array;
  readonly lineCount: number;
  readonly ribbonData: Float32Array;
  readonly ribbonCount: number;
  readonly sheetData?: Float32Array;
  readonly sheetCount?: number;
  readonly sheetBasisData?: Float32Array;
  readonly sheetRevealData?: Float32Array;
  readonly flowVertices?: Float32Array;
  readonly flowIndices?: Uint32Array;
  readonly valueAtlas?: Rg32fValueAtlas;
  /** Content-addressed cache key for RG32F GPU texture upload deduplication. */
  readonly atlasKey?: string;
  readonly atlasAudit?: DiffusionAtlasLayoutAudit;
  readonly fitBounds: AABB;
  readonly defaultCameraAzimuthDeg: number;
  readonly defaultCameraElevationDeg: number;
  readonly landmarks: readonly LandmarkAnchor[];
  readonly kvDepthTicks: readonly KvDepthTickAnchor[];
  readonly calipers?: readonly DimensionCaliper[];
  readonly solidPickBoxes: readonly PickBox[];
  readonly fasciaPickBoxes: readonly PickBox[];
  readonly partCatalog: Readonly<Record<string, CutawayPartInfo>>;
  readonly activeOpPartIds?: ReadonlySet<string>;
  readonly activeCellGrids?: readonly ActiveSheetCellGrid[];
  readonly cellPopover?: CellPopoverSpec | null;
  readonly activeFormula?: LayerFormulaReadout | null;
  readonly stepperSteps?: readonly StepperStepDef[];
  readonly provenanceBadges: readonly DiffusionProvenanceBadge[];
  readonly epistemicNote: string;
  readonly warningBanner: string | null;
  readonly ropePhasor?: DiffusionRopePhasorTelemetry | null;
  readonly diagnostics?: Readonly<Record<string, unknown>>;
}

/**
 * Read-only session resolution snapshot derived strictly from the active or requested
 * `DiffusionSessionRecord`, providing 0-ULP lazy layer and cropped-suffix KV access.
 */
export interface DiffusionResolvedSessionContext {
  readonly model: DiffusionModelParameters;
  readonly session: DiffusionSession;
  readonly record: DiffusionSessionRecord;
  readonly parentRecord: DiffusionSessionRecord | null;
  readonly branchId: string;
  readonly recordIndex: number;
  readonly branchRecordCount: number;
  readonly isHistoricalCursor: boolean;
  readonly modeAtRecord: DiffusionExecutionMode;
  readonly causalContextAtRecord: boolean;
  /** True strictly when the active record executed with draft-only `o_proj` LoRA enabled. */
  readonly loraActiveOnDraft: boolean;
  readonly kvSnapshot: PersistentKvSnapshot;
  readonly kvIntegrityValid: boolean;
  readonly kvIntegrityError: string | null;
  resolveLayerTrace(layerIndex: number): DiffusionLayerTrace;
  resolveSpecCroppedSuffixSlabs(): readonly KvPositionSlab[];
  readonly provenanceBadges: readonly DiffusionProvenanceBadge[];
  readonly warningBanner: string | null;
}
