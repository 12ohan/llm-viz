import type { GqaSimResult } from '../math/gqaSim';
import type { MambaSimResult } from '../math/mambaSim';
import type { MoeSimResult } from '../math/moeSim';
import type { AABB } from '../render/camera';
import type { PickBox } from '../render/pick';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../ui/overlay';
import type { FocusedAccumulationResult } from './accumulation';
import type { Rg32fValueAtlas } from './atlasPacker';
import type { EmbeddingCellFormula } from './embeddingCutaway';
import type { OutputCellFormula } from './outputCutaway';
import type { LayerFormulaReadout } from './sheetInspector';
import type { DimensionCaliper, KvDepthTickAnchor, LandmarkAnchor } from './spine';

export {
  buildEmbeddingCutaway,
  fitSpatialEmbeddingCamera,
  pickHitToEmbeddingCell,
  EMBEDDING_STEPPER_STEPS,
  type HybridEmbeddingResult,
  type EmbeddingTensorKey,
  type EmbeddingCellRef,
  type EmbeddingSheetSpan,
  type EmbeddingCellFormula,
  type EmbeddingFormulaInput,
  type BuildEmbeddingCutawayOptions,
  type SpatialEmbeddingFitOptions,
} from './embeddingCutaway';
export {
  buildOutputCutaway,
  fitSpatialOutputCamera,
  pickHitToOutputCell,
  parseOutputCandidatePartId,
  OUTPUT_STEPPER_STEPS,
  type OutputTensorKey,
  type OutputAuxTensorKey,
  type OutputCellRef,
  type OutputSheetSpan,
  type OutputCellFormula,
  type OutputFormulaInput,
  type BuildOutputCutawayOptions,
  type SpatialOutputFitOptions,
} from './outputCutaway';
export type { ActiveSheetCellGrid, CellPopoverSpec } from '../ui/overlay';
export type { FocusedAccumulationResult } from './accumulation';

export {
  buildGqaCutaway,
  fitSpatialGqaCamera,
  parseQueryHeadPartId,
  formatQueryHeadId,
  GQA_FIT_BOUNDS,
  GQA_SPATIAL_FIT_BOUNDS,
} from './gqaCutaway';
export {
  buildMambaCutaway,
  fitSpatialMambaCamera,
  parseMambaHeadPartId,
  MAMBA_FIT_BOUNDS,
  MAMBA_SPATIAL_FIT_BOUNDS,
} from './mambaCutaway';
export { formatMambaHeadId, mambaGroupForHead } from '../math/mambaSim';
export {
  buildMoECutaway,
  fitSpatialMoeCamera,
  parseMoeExpertPartId,
  MOE_FIT_BOUNDS,
  MOE_SPATIAL_FIT_BOUNDS,
} from './moeCutaway';
export { activeRoutedExpertIndices } from '../math/moeSim';

export interface CutawayPartInfo {
  readonly id: string;
  readonly title: string;
  readonly badge: string;
  readonly summary: string;
  readonly specs: readonly string[];
}

export interface MambaCutawayStats {
  readonly bayId: 1 | 42;
  readonly isFeeder: boolean;
  readonly conv1dTapCount: number;
  readonly conv1dFp8: boolean;
  readonly ssmHeadCount: number;
  readonly ssmStateFp8: boolean;
  readonly inProjFp8: boolean;
  readonly outProjFp8: boolean;
  readonly activeHead?: number;
  readonly activeGroup?: number;
  readonly stateTokenIndex?: number;
  readonly miniatureHeadCount?: number;
  readonly miniatureGroupCount?: number;
  readonly usesSmoothRibbons?: boolean;
  readonly voxelArrowCount?: number;
  readonly ssmBytesAll23?: number;
  readonly conv1dBytesAll23?: number;
}

export interface MoECutawayStats {
  readonly bayId: 2 | 52;
  readonly routedTotal: number;
  readonly routedActiveCount: number;
  readonly activeExpertIds: readonly number[];
  readonly sharedExpertCount: number;
  readonly routedWidth: number;
  readonly sharedWidth: number;
  readonly matricesPerExpert: number;
  readonly hasGateProj: boolean;
  readonly cacheBytes: number;
  readonly weightsFp8: boolean;
  readonly routerFp8?: boolean;
  readonly focusToken?: number;
  readonly focusExpert?: number;
  readonly usesSmoothRibbons?: boolean;
  readonly voxelArrowCount?: number;
}

export interface GqaCutawayStats {
  readonly bayId: 6 | 43;
  readonly qHeadCount: number;
  readonly qGroupCount: number;
  readonly qHeadsPerGroup: number;
  readonly kvHeadCount: number;
  readonly gqaRatio: number;
  readonly kvTokenSliceCount: number; // 2 * T paired K+V slices across 2 KV heads
  readonly computeFp8: boolean;
  readonly kvCacheFp8: boolean;
  readonly usesSmoothRibbons?: boolean;
  readonly voxelArrowCount?: number;
  readonly selectedQueryHead?: number | null;
  readonly activeGroup0Head?: number;
  readonly activeGroup1Head?: number;
}

export interface CutawaySceneGeometry {
  readonly kind:
    | 'cutaway_embedding'
    | 'cutaway_mamba'
    | 'cutaway_moe'
    | 'cutaway_gqa'
    | 'cutaway_output';
  readonly bayId: number;
  readonly solidData: Float32Array;
  readonly solidCount: number;
  readonly lineData: Float32Array;
  readonly lineCount: number;
  readonly ribbonData: Float32Array;
  readonly ribbonCount: number;
  /** Thin-sheet instances for SheetsPass (20 floats/instance). Present on all rebuilt cutaways. */
  readonly sheetData?: Float32Array;
  readonly sheetCount?: number;
  /** Optional per-sheet 3D orthonormal basis buffer (6 floats/instance: axisU, axisN; axisV = U x N). */
  readonly sheetBasisData?: Float32Array;
  /** Optional parallel 4-float-per-sheet progressive reveal buffer for SheetsPass.upload(..., sheetRevealData). */
  readonly sheetRevealData?: Float32Array;
  /** Smooth 2D/3D vector flow ribbons for FlatFlowRibbonsPass. */
  readonly flowVertices?: Float32Array;
  readonly flowIndices?: Uint32Array;
  /** Live RG32F value atlas texture data for SheetsPass. */
  readonly valueAtlas?: Rg32fValueAtlas;
  /** Cached miniature GQA simulator output for live formula readout & stepper. */
  readonly gqaSim?: GqaSimResult;
  /** Cached miniature Mamba-2 simulator output for live formula readout & stepper. */
  readonly mambaSim?: MambaSimResult;
  /** Cached miniature MoE simulator output for live formula readout & stepper. */
  readonly moeSim?: MoeSimResult;
  /** Live formula readout for the currently inspected (hovered/pinned/stepped) cell. */
  readonly activeFormula?: LayerFormulaReadout | null;
  /** Live formula readout for the dedicated Embedding (NoPE) cutaway. */
  readonly embeddingFormula?: EmbeddingCellFormula | null;
  /** Live formula readout for the dedicated Output & Untied LM Head cutaway. */
  readonly outputFormula?: OutputCellFormula | null;
  /** Active-op part IDs for V1 secondary sheet label dimming in MacroOverlay. */
  readonly activeOpPartIds?: ReadonlySet<string>;
  /** In-canvas contextual cell formula popover + provenance pointers for V3 MacroOverlay. */
  readonly cellPopover?: CellPopoverSpec | null;
  /** Active/inspected thin sheets eligible for V3 zoom-adaptive cell value overlays. */
  readonly activeCellGrids?: readonly ActiveSheetCellGrid[];
  /** Optional focused-cell term-by-term accumulation readout for RC1 progressive reveal. */
  readonly focusedAccumulation?: FocusedAccumulationResult | null;
  readonly fitBounds: AABB;
  /** Per-operation camera focus AABBs keyed by stepper step index 0..5. */
  readonly opFocusBounds?: Readonly<Record<number, AABB>>;
  readonly landmarks: readonly LandmarkAnchor[];
  readonly kvDepthTicks: readonly KvDepthTickAnchor[];
  readonly calipers?: readonly DimensionCaliper[];
  readonly solidPickBoxes: readonly PickBox[];
  readonly partCatalog: Readonly<Record<string, CutawayPartInfo>>;
  readonly mambaStats?: MambaCutawayStats;
  readonly moeStats?: MoECutawayStats;
  readonly gqaStats?: GqaCutawayStats;
}
