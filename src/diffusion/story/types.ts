/**
 * Shared Headless DF4 Contracts (`src/diffusion/story/types.ts`)
 *
 * Pinned Step-0 contract shared across:
 *   - Slice 4A (`src/diffusion/archReadouts.ts`, `tests/diffusion-arch-readouts.test.ts`)
 *   - Slice 4B (`src/diffusion/story/walkthrough.ts`, `src/diffusion/app/timelineDock.ts`,
 *               `src/diffusion/app/spatialApp.ts`, `src/diffusion/app/spatial.css`)
 *   - Slice 4C (`tests/diffusion-tour.test.ts`, `tests/diffusion-timeline-a11y.test.ts`)
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import type { CameraPose } from '../../model/state';
import type {
  DiffusionBadgeTone,
  DiffusionFocusState,
  DiffusionResolvedSessionContext,
  DiffusionSpatialViewMode,
} from '../scene/types';
import type { DiffusionSession } from '../session';
import type {
  DiffusionDiscrepancyEntry,
  DiffusionExecutionMode,
  DiffusionLifecyclePhase,
  DiffusionSessionMemoryReport,
  DiffusionSessionRecordKind,
  DiffusionVariantSpec,
  ForwardCounters,
  KvMaskProvenance,
} from '../types';

// ============================================================================
// SLICE 4A: ARCHITECTURE READOUTS, DUAL SI/IEC TABLES, SCALING & LEDGER
// ============================================================================

export interface DiffusionDualByteReadout {
  readonly bytes: number;
  /** Decimal SI formatting (`B`, `KB`, `MB`, `GB`). */
  readonly siLabel: string;
  /** Binary IEC formatting (`B`, `KiB`, `MiB`, `GiB`). */
  readonly iecLabel: string;
  /** Combined editorial display string, e.g. `"25.16 GiB (27.01 GB)"`. */
  readonly dualLabel: string;
}

export interface DiffusionComponentReadoutRow {
  readonly component: string;
  readonly tensorPattern: string;
  readonly shape: string;
  readonly tensorCount: number;
  readonly parameters: number;
  readonly parametersFormatted: string;
  readonly shareOfBackbonePct: string;
  readonly dtype: 'BF16' | 'F32';
  readonly payload: DiffusionDualByteReadout;
}

export interface DiffusionLfsTableRow {
  readonly repoId: string;
  readonly shortRepoLabel: string;
  readonly revision: string;
  readonly shortRevision: string;
  readonly file: string;
  readonly dtype: 'BF16' | 'F32';
  readonly isLora: boolean;
  readonly tensors: number;
  readonly parameters: number;
  readonly parametersFormatted: string;
  readonly headerBytes: number;
  readonly payload: DiffusionDualByteReadout;
  readonly total: DiffusionDualByteReadout;
  readonly hfLfsSizeBytes: number;
  readonly diffBytes: number;
  readonly lfsSha256Prefix: string;
}

export interface DiffusionProductionVariantRow {
  readonly id: DiffusionVariantSpec['id'];
  readonly repoId: string;
  readonly shortRevision: string;
  readonly isCanonical: boolean;
  readonly layersLabel: string;
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly gqaLabel: string;
  readonly ropeThetaLabel: string;
  readonly yarnLabel: string;
  readonly maxPositionEmbeddings: number;
  readonly eosSummary: string;
  readonly backboneParameters: number;
  readonly backboneParamsFormatted: string;
  readonly backbonePayload: DiffusionDualByteReadout;
  readonly loraParameters: number | null;
  readonly loraSummary: string;
  readonly loraPayload: DiffusionDualByteReadout | null;
}

export interface DiffusionContextKvScalingRow {
  readonly tokens: number;
  readonly tokenLabel: string;
  readonly tier: 1 | 2 | 3 | 4;
  readonly tierBadge: string;
  readonly tierProvenance: string;
  /** Production `14B` BF16 KV footprint (`163,840 B / tok`). */
  readonly prodBf16Kv: DiffusionDualByteReadout;
  /** Production `14B` BF16 KV plus temporary `B = 32` block buffer (`+5.00 MiB`). */
  readonly prodWithTempBlock32: DiffusionDualByteReadout;
  /** Miniature `40L` FP32 analytical KV footprint (`20,480 B / tok`). */
  readonly miniFp32Kv: DiffusionDualByteReadout;
  /** Llama-4 query scale at tail position (`pos = max(0, tokens - 1)`, `orig = 16,384`, `beta = 0.1`). */
  readonly llama4ScaleAtTail: number;
  readonly llama4ScaleLabel: string;
  readonly crossesOrigMaxPos: boolean;
  readonly transitionNote: string;
}

export type DiffusionMemoryBoundaryId =
  | 'backbone_weights'
  | 'lora_weights'
  | 'session_retained'
  | 'active_atlas';

export interface DiffusionMemoryBoundaryRow {
  readonly id: DiffusionMemoryBoundaryId;
  readonly label: string;
  readonly bytes: number;
  readonly formatted: DiffusionDualByteReadout;
  readonly capBytes: number | null;
  readonly capFormatted: string | null;
  readonly utilizationPct: number | null;
  readonly withinCap: boolean;
  readonly ownership: string;
  readonly note: string;
}

export type DiffusionDiscrepancyCategory =
  | 'all'
  | 'architecture_specs'
  | 'rope_masks'
  | 'execution_kv'
  | 'samplers_guards';

export interface DiffusionDiscrepancyFilterOptions {
  readonly query?: string;
  readonly category?: DiffusionDiscrepancyCategory;
  readonly ids?: readonly string[];
}

export interface DiffusionDiscrepancyViewRow extends DiffusionDiscrepancyEntry {
  readonly numericId: number;
  readonly category: Exclude<DiffusionDiscrepancyCategory, 'all'>;
  readonly categoryLabel: string;
}

export interface DiffusionDiscrepancyTableModel {
  readonly totalCount: number;
  readonly filteredCount: number;
  readonly query: string;
  readonly category: DiffusionDiscrepancyCategory;
  readonly rows: readonly DiffusionDiscrepancyViewRow[];
}

export interface DiffusionNfeTpfReadout {
  readonly mode: DiffusionExecutionMode;
  readonly modeLabel: string;
  readonly actualForwardPasses: number;
  readonly decodeForwardPasses: number;
  readonly referenceMethodNfe: number;
  readonly generatedTokens: number;
  /** Formatted tokens per actual forward pass, or `'—'` when denominator is `0`. */
  readonly tpfActualLabel: string;
  /** Formatted tokens per decode forward pass, or `'—'` when denominator is `0`. */
  readonly tpfDecodeLabel: string;
  /** Formatted tokens per reference method NFE, or `'—'` when denominator is `0`. */
  readonly tpfRefLabel: string;
  readonly actualRatioValue: number | null;
  readonly decodeRatioValue: number | null;
  readonly refRatioValue: number | null;
  /** Explains D12 upstream NFE counting convention for the active mode. */
  readonly methodNote: string;
  readonly d12ComparisonNote: string;
}

export interface DiffusionCanonical14BSummary {
  readonly repoId: string;
  readonly revision: string;
  readonly shortRevision: string;
  readonly paperArxivId: string;
  readonly githubRepo: string;
  readonly backboneTensors: number;
  readonly backboneParameters: number;
  readonly backboneParamsFormatted: string;
  readonly backbonePayload: DiffusionDualByteReadout;
  readonly loraTensors: number;
  readonly loraParameters: number;
  readonly loraParamsFormatted: string;
  readonly loraShareOfBackbonePct: string;
  readonly loraPayload: DiffusionDualByteReadout;
  readonly lfsFileCount: number;
  readonly lfsTotalDiffBytes: number;
  readonly lfsAllZeroDiff: boolean;
  readonly projectionReversalNote: string;
  readonly epistemicNote: string;
  readonly outOfScopeNote: string;
}

export interface DiffusionArchReadoutsInput {
  readonly mode: DiffusionExecutionMode;
  readonly causalContext: boolean;
  readonly counters: ForwardCounters;
  readonly generatedTokens: number;
  readonly memoryReport: DiffusionSessionMemoryReport;
  readonly activeAtlasBytes?: number;
  readonly activeAtlasWidth?: number;
  readonly activeAtlasHeight?: number;
  readonly activeAtlasName?: string;
  readonly discrepancyFilter?: DiffusionDiscrepancyFilterOptions;
}

export interface DiffusionArchReadoutsBundle {
  readonly canonical14B: DiffusionCanonical14BSummary;
  readonly components14B: readonly DiffusionComponentReadoutRow[];
  readonly familyVariants: readonly DiffusionProductionVariantRow[];
  readonly lfsInventory: readonly DiffusionLfsTableRow[];
  readonly contextKvScaling: readonly DiffusionContextKvScalingRow[];
  readonly memoryBoundaries: readonly DiffusionMemoryBoundaryRow[];
  readonly memoryTotalMiniBytes: number;
  readonly memoryTotalMiniFormatted: DiffusionDualByteReadout;
  readonly nfeTpf: DiffusionNfeTpfReadout;
  readonly discrepancyLedger: DiffusionDiscrepancyTableModel;
}

// ============================================================================
// SLICE 4B: GUIDED WALKTHROUGH (4 ACTS, ISOLATED SESSION)
// ============================================================================

export type DiffusionTourActId = 'act-1' | 'act-2' | 'act-3' | 'act-4';
export type DiffusionTourStatus = 'idle' | 'active' | 'exited';
export type DiffusionInspectorTabId = 'inspect' | 'architecture' | 'ledger';

export interface DiffusionTourLiveMetric {
  readonly label: string;
  readonly value: string;
  readonly tone: DiffusionBadgeTone;
}

export interface DiffusionTourActSummary {
  readonly actIndex: number;
  readonly id: DiffusionTourActId;
  readonly title: string;
  readonly shortTitle: string;
  readonly startBeatIndex: number;
  readonly beatCount: number;
  readonly isCurrent: boolean;
}

export interface DiffusionTourBeatSnapshot {
  readonly beatIndex: number;
  readonly totalBeats: number;
  readonly beatId: string;
  readonly actId: DiffusionTourActId;
  readonly actIndex: number;
  readonly actTitle: string;
  readonly title: string;
  readonly subtitle: string;
  readonly narrative: string;
  readonly takeaway: string;
  readonly viewMode: DiffusionSpatialViewMode;
  readonly focus: DiffusionFocusState;
  readonly recordId: string;
  readonly branchId: string;
  readonly mode: DiffusionExecutionMode;
  readonly causalContext: boolean;
  readonly loraActiveOnDraft: boolean;
  /** 100% live metrics derived strictly from the isolated tour session's real tensors/records. */
  readonly liveMetrics: readonly DiffusionTourLiveMetric[];
  readonly provenanceSummary: string;
  readonly epistemicNote: string;
  /** Optional RoPE position scrubber override for Act 2 Llama-4 boundary beats (`null` = live). */
  readonly ropeScrubPosition: number | null;
}

export interface DiffusionFreeExploreRestoreState {
  readonly activeRecordId: string | null;
  readonly activeBranchId: string;
  readonly focus: DiffusionFocusState;
  readonly cameraPose: CameraPose;
  readonly inspectorTab: DiffusionInspectorTabId;
  readonly inspectorOpen: boolean;
  readonly ropeScrubPosition: number | null;
}

export interface DiffusionGuidedTourController {
  enter(freeExploreSnapshot: DiffusionFreeExploreRestoreState): DiffusionTourBeatSnapshot;
  exit(): DiffusionFreeExploreRestoreState | null;
  next(): DiffusionTourBeatSnapshot | null;
  prev(): DiffusionTourBeatSnapshot | null;
  goToBeat(index: number): DiffusionTourBeatSnapshot | null;
  goToAct(actId: DiffusionTourActId): DiffusionTourBeatSnapshot | null;
  getStatus(): DiffusionTourStatus;
  isActive(): boolean;
  getBeatCount(): number;
  getCurrentBeatIndex(): number;
  getCurrentBeat(): DiffusionTourBeatSnapshot | null;
  getAllBeats(): readonly DiffusionTourBeatSnapshot[];
  getActs(): readonly DiffusionTourActSummary[];
  getTourSession(): DiffusionSession;
  getTourContext(): DiffusionResolvedSessionContext;
  getRestoreState(): DiffusionFreeExploreRestoreState | null;
}

// ============================================================================
// SLICE 4B: UNIFIED TRI-MODE EVENT TIMELINE DOCK CONTRACTS
// ============================================================================

export interface DiffusionTimelineNodeViewModel {
  readonly recordId: string;
  readonly parentRecordId: string | null;
  readonly branchId: string;
  readonly recordIndex: number;
  readonly kind: DiffusionSessionRecordKind;
  readonly kindBadge: string;
  readonly shortLabel: string;
  readonly mode: DiffusionExecutionMode;
  readonly modeLabel: string;
  readonly causalContext: boolean;
  readonly kvProvenance: KvMaskProvenance;
  readonly kvSeqLen: number;
  readonly kvCapacity: number;
  readonly emittedCount: number;
  readonly totalGeneratedCount: number;
  readonly lifecyclePhase: DiffusionLifecyclePhase;
  readonly counters: ForwardCounters;
  readonly hasGuardWarning: boolean;
  readonly guardSummary: string | null;
  readonly loraActiveOnDraft: boolean;
  readonly isActive: boolean;
  readonly isOnActiveBranch: boolean;
  readonly isForkSource: boolean;
}

export interface DiffusionTimelineBranchViewModel {
  readonly branchId: string;
  readonly parentBranchId: string | null;
  readonly forkRecordId: string | null;
  readonly isActive: boolean;
  readonly canDelete: boolean;
  readonly nodeCount: number;
  readonly tipRecordId: string | null;
  readonly nodes: readonly DiffusionTimelineNodeViewModel[];
}

export interface DiffusionTimelineDockState {
  readonly branches: readonly DiffusionTimelineBranchViewModel[];
  readonly activeBranchId: string;
  readonly activeRecordId: string | null;
  readonly branchCount: number;
  readonly maxBranches: number;
  readonly recordCount: number;
  readonly maxRecords: number;
  readonly memoryReport: DiffusionSessionMemoryReport;
  readonly memoryBarPct: number;
  readonly memoryLabel: string;
  readonly withinMemoryCap: boolean;
  readonly isReadOnly: boolean;
}
