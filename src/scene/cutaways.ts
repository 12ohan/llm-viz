import type { GqaSimResult } from '../math/gqaSim';
import type { MambaSimResult } from '../math/mambaSim';
import type { MoeSimResult } from '../math/moeSim';
import type { AABB } from '../render/camera';
import type { PickBox } from '../render/pick';
import type { Rg32fValueAtlas } from './atlasPacker';
import type { LayerFormulaReadout } from './sheetInspector';
import type { DimensionCaliper, KvDepthTickAnchor, LandmarkAnchor } from './spine';

export {
  buildGqaCutaway,
  fitSpatialGqaCamera,
  kvHeadForQueryHead,
  parseQueryHeadPartId,
  formatQueryHeadId,
  GQA_FIT_BOUNDS,
  GQA_SPATIAL_FIT_BOUNDS,
} from './gqaCutaway';
export { buildMambaCutaway, parseMambaHeadPartId, MAMBA_FIT_BOUNDS } from './mambaCutaway';
export { formatMambaHeadId, mambaGroupForHead } from '../math/mambaSim';
export { buildMoECutaway, parseMoeExpertPartId, MOE_FIT_BOUNDS } from './moeCutaway';
export { activeRoutedExpertIndices, formatExpertId } from '../math/moeSim';

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
  readonly kind: 'cutaway_mamba' | 'cutaway_moe' | 'cutaway_gqa';
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
  readonly fitBounds: AABB;
  readonly landmarks: readonly LandmarkAnchor[];
  readonly kvDepthTicks: readonly KvDepthTickAnchor[];
  readonly calipers?: readonly DimensionCaliper[];
  readonly solidPickBoxes: readonly PickBox[];
  readonly partCatalog: Readonly<Record<string, CutawayPartInfo>>;
  readonly mambaStats?: MambaCutawayStats;
  readonly moeStats?: MoECutawayStats;
  readonly gqaStats?: GqaCutawayStats;
}
