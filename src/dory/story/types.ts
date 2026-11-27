/**
 * Frozen `DORY5` Step-0 Story, 4-Act Guided Walkthrough & Headless Readouts Contracts
 * (`src/dory/story/types.ts`)
 *
 * Shared Step-0 contract consumed across strictly disjoint parallel DORY5 slices & DORY6:
 *   - Slice 5A: `src/dory/archReadouts.ts`, `src/dory/scene/bookendsScene.ts`, `src/dory/scene/index.ts`
 *   - Slice 5B: `src/dory/story/walkthrough.ts` (4-Act guided tour over isolated `DorySession`)
 *   - Slice 5C: `tests/dory-readouts-tour.test.ts` (Gate 5 verification suite)
 *   - DORY6:    `src/dory/app/{spatialApp,spatialOverlay,timelineDock,charts}.ts`
 *
 * Locked Decisions & Invariants Enforced Here:
 *   - `D1` Physical `0..71` single-sublayer indexing (`16 + 40 + 16`) with paired `L{p >> 1}` badges.
 *   - `D3`/`DR23` Contract A `104`-bank self-speculation badged `Reconstructed from Paper §6.2`.
 *   - `D4` Architectural Boundary Isolation: ZERO imports from Gate 4b/4c Nano modules
 *     (`src/ui/*`, `src/scene/spine.ts`, `src/scene/sheetInspector.ts`, `src/model/*`,
 *     `src/math/*`, `src/story/*`, `src/main.ts`) or sibling apps (`twotower`, `diffusion`, `uni`).
 *   - `D7` 4-Act Guided Narrative (`Audit §12.2`) carrying the full immutable 8-tuple
 *     `DoryExecutionCoord` (`Audit §6.3`) on every beat and executing strictly against an
 *     isolated private `DorySession` (byte-for-byte zero mutation of any external user session).
 *   - `D10` Byte-budgeted `3.00 MiB` LRU sparse-checkpoint pool keeping 12-branch sessions
 *     `<= 12.00 MiB` at both `k = 4` (`<= 7.29 MiB`) and `k = 8` (`<= 9.60 MiB`).
 *   - View 7 & `archReadouts.ts` strictly read-only (`DORY_TBPTL_READONLY_BADGE = 'READ-ONLY · NO GRADIENTS'`,
 *     zero fake quantization toggles, zero training/RLVR loops, `0 B` diff 4-checkpoint constants).
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL execution, Node built-ins, or ambient RNG.
 */

import type { CameraPose } from '../../render/camera';
import type {
  DORY_TBPTL_READONLY_BADGE,
  DoryBadgeTone,
  DoryCheckpointFormatId,
  DoryProvenanceBadge,
  DoryResolvedSessionContext,
  DorySpatialFocusState,
  DorySpatialScene,
  DorySpatialSceneDispatcher,
  DorySpatialViewMode,
} from '../scene/types';
import type {
  DoryCacheRole,
  DoryCheckpointFormatSpec,
  DoryDesignDecisionEntry,
  DoryDesignDecisionId,
  DoryDiscrepancyCategory,
  DoryDiscrepancyEntry,
  DoryDiscrepancyId,
  DoryDualRopeTables,
  DoryExecutionCoord,
  DoryExecutionMode,
  DoryKvBankKind,
  DoryKvCacheMode,
  DoryModelParameters,
  DoryProductionKvPoint,
  DoryScalingRatioEntry,
  DorySession,
  DorySessionMemoryAudit,
} from '../types';

// Re-export canonical Audit §6.3 8-tuple execution coordinate types for story/tour callers
export type { DoryCacheOwnership, DoryExecutionCoord } from '../types';

// ============================================================================
// 4-ACT GUIDED WALKTHROUGH CONSTANTS & ACT DEFINITIONS (AUDIT §12.2)
// ============================================================================

/** Canonical act count for the Dory1.0-4B Guided Narrative (`Audit §12.2`). */
export const DORY_TOUR_TOTAL_ACTS = 4 as const;

/** Minimum beats per act ensuring substantive pedagogical pacing across all 7 views. */
export const DORY_TOUR_MIN_BEATS_PER_ACT = 3 as const;

/** Dedicated deterministic PRNG seed for the tour's strictly isolated internal `DorySession`. */
export const DORY_TOUR_SEED = 0xd0415005 as const;

/** Mandatory disclosure badge for paper-reported benchmarks (never claimed on synthetic mini weights). */
export const DORY_PAPER_RESULTS_BADGE =
  'PAPER RESULTS (arXiv:2611.03998) · NOT MEASURED ON SYNTHETIC MINIATURE' as const;

export type DoryTourActId = 'act1' | 'act2' | 'act3' | 'act4';
export type DoryTourStatus = 'idle' | 'active' | 'exited';
export type DoryInspectorTabId = 'inspect' | 'readouts' | 'ledger';

/**
 * Isolated deterministic branch scenarios staged inside the guided tour's private `DorySession`.
 * Every scenario uses non-aliased KV slabs (`DR18`/`DR19`) and remains well inside `<= 12.00 MiB`.
 */
export type DoryTourScenarioKey =
  | 'standard_k1_per_loop'
  | 'standard_k2_per_loop'
  | 'standard_k4_per_loop'
  | 'standard_k4_last_loop_chunked'
  | 'self_spec_contract_a';

export interface DoryTourActDescriptor {
  readonly id: DoryTourActId;
  readonly actNumber: 1 | 2 | 3 | 4;
  readonly actIndex: 0 | 1 | 2 | 3;
  readonly title: string;
  readonly shortTitle: string;
  readonly subtitle: string;
  readonly views: readonly DorySpatialViewMode[];
}

/**
 * Authoritative 4-Act structure directly matching `DORY0_ARCHITECTURE_AUDIT.md` §12.2:
 *   - Act I:   Hypersphere Foundations (Views 7, 1, 2)
 *   - Act II:  Inside the Normalized Mixers (Views 3, 4)
 *   - Act III: Shared-Weight Latent Recurrence & Dual-Axis KV (Views 1, 2, 5)
 *   - Act IV:  Elastic Depth, Paper §6.2 Self-Speculation & Production Context (Views 6, 7)
 */
export const DORY_TOUR_ACT_DESCRIPTORS: readonly DoryTourActDescriptor[] = Object.freeze([
  Object.freeze({
    id: 'act1',
    actNumber: 1,
    actIndex: 0,
    title: 'Act I — Hypersphere Foundations',
    shortTitle: 'Act I · Hypersphere',
    subtitle:
      'Untied unit-norm tokens on S^(d-1), physical 72-sublayer topology (16+40+16), and channel-wise nGPT NLERP + 5 scaling families.',
    views: Object.freeze(['bookends', 'macro', 'hypersphere'] as const),
  }),
  Object.freeze({
    id: 'act2',
    actNumber: 2,
    actIndex: 1,
    title: 'Act II — Inside the Normalized Mixers',
    shortTitle: 'Act II · Mixers',
    subtitle:
      'Gated GQA (8->64 mini vs. 1.60x prod, sqk^2, dual-RoPE SWA vs. Full, sigmoid gate, attn_sub_norm) and Normalized SwiGLU (signed suv, ffn_sub_norm).',
    views: Object.freeze(['gqa_rope', 'swiglu'] as const),
  }),
  Object.freeze({
    id: 'act3',
    actNumber: 3,
    actIndex: 2,
    title: 'Act III — Shared-Weight Latent Recurrence & Dual-Axis KV',
    shortTitle: 'Act III · Recurrence & KV',
    subtitle:
      'Unrolling k = 1 -> 4 with identical physical weights, anchor re-injection u_t = N(s_{t-1} + x), per_loop vs. last_loop partition divergence (DR17), and 1 -> 256K scaling.',
    views: Object.freeze(['macro', 'hypersphere', 'kv_compare'] as const),
  }),
  Object.freeze({
    id: 'act4',
    actNumber: 4,
    actIndex: 3,
    title: 'Act IV — Elastic Depth, Paper §6.2 Self-Speculation & Production Context',
    shortTitle: 'Act IV · Self-Spec & Prod',
    subtitle:
      'Reconstructed Contract A gamma=4 k=1->k=4 self-speculation (104 Dual-Coda banks, binary64 rejection, cap_swa=16 rollback), 0 B diff 4-checkpoint footprints, and READ-ONLY TBPTL.',
    views: Object.freeze(['self_spec', 'bookends'] as const),
  }),
]);

export interface DoryTourActSummary extends DoryTourActDescriptor {
  readonly startBeatIndex: number;
  readonly beatCount: number;
  readonly isCurrent: boolean;
}

// ============================================================================
// BEAT BLUEPRINT, 8-TUPLE COORDINATE TARGET & LIVE SNAPSHOT CONTRACTS
// ============================================================================

export interface DoryTourCameraOverride {
  readonly azimuthDeg: number;
  readonly elevationDeg: number;
}

export interface DoryTourLiveMetric {
  readonly label: string;
  readonly value: string;
  readonly tone: DoryBadgeTone;
  readonly discrepancyId?: DoryDiscrepancyId;
}

/**
 * Declarative architectural target resolved into a live full 8-tuple `DoryExecutionCoord`
 * (`modelWeightId`, `physicalLayerIdx`, `executedLayerIdx`, `loopIteration`, `absTokenPos`,
 * `forwardStepId`, `sessionBranchId`, `cacheOwnership`) against the isolated tour session.
 */
export interface DoryTourBeatCoordTarget {
  /** Canonical physical sublayer `0..71` (even = Attn, odd = MLP; `pairedLayerIdx = p >> 1`). */
  readonly physicalLayerIdx: number;
  /** Target loop iteration (`0` for Prelude/Coda; `1..k` for Recurrent passes). */
  readonly loopIteration: number;
  /** Token offset inside the target record (`'last'` = final committed position, or explicit integer). */
  readonly tokenOffset: 'last' | number;
  /** Optional explicit cache-ownership bank kind override (e.g. `'output_shallow'` for Contract A). */
  readonly expectedBankKind?: DoryKvBankKind;
  /** Optional explicit cache-ownership role override (e.g. `'spec_verify_reused'`, `'spec_draft'`). */
  readonly expectedCacheRole?: DoryCacheRole;
}

/** Static declarative specification of a single guided walkthrough beat. */
export interface DoryTourBeatSpec {
  /** Zero-based global beat index (`0 .. totalBeats - 1`). */
  readonly index: number;
  readonly id: string;
  readonly actId: DoryTourActId;
  readonly actNumber: 1 | 2 | 3 | 4;
  readonly actIndex: 0 | 1 | 2 | 3;
  /** One-based beat index within its act (`1 .. act.beatCount`). */
  readonly beatInAct: number;
  readonly title: string;
  readonly subtitle: string;
  readonly takeaway: string;
  readonly narrative: string;
  readonly equation?: string;
  /** Explicit epistemic governance disclosure (e.g. synthetic mini vs. paper results, DR23, anti-SLERP). */
  readonly epistemicNote: string;
  readonly discrepancyRefs: readonly DoryDiscrepancyId[];
  readonly designDecisionRefs: readonly DoryDesignDecisionId[];
  readonly viewMode: DorySpatialViewMode;
  readonly scenarioKey: DoryTourScenarioKey;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  /** Target record index within the scenario branch (`0` = prefill, `-1` = latest tip record). */
  readonly targetStepOffset: number;
  /** Declarative target resolved to a live 8-tuple `DoryExecutionCoord` in `DoryTourBeatSnapshot`. */
  readonly coordTarget: DoryTourBeatCoordTarget;
  /** Partial spatial focus merged over `DEFAULT_DORY_SPATIAL_FOCUS_STATE` when this beat activates. */
  readonly focusPatch: Partial<DorySpatialFocusState>;
  readonly highlightPartId?: string | null;
  readonly cameraOverride?: DoryTourCameraOverride | null;
}

/**
 * Live resolved guided walkthrough beat snapshot bound to the isolated tour `DorySession`.
 * Carries the exact immutable 8-tuple `DoryExecutionCoord` (`Audit §6.3 & §12.2`) and 100% live
 * numerical metrics computed strictly from real miniature tensors & session records.
 */
export interface DoryTourBeatSnapshot {
  readonly beatIndex: number;
  readonly totalBeats: number;
  readonly beatId: string;
  readonly actId: DoryTourActId;
  readonly actNumber: 1 | 2 | 3 | 4;
  readonly actIndex: 0 | 1 | 2 | 3;
  readonly beatInAct: number;
  readonly actTitle: string;
  readonly title: string;
  readonly subtitle: string;
  readonly takeaway: string;
  readonly narrative: string;
  readonly equation: string | null;
  readonly epistemicNote: string;
  readonly discrepancyRefs: readonly DoryDiscrepancyId[];
  readonly designDecisionRefs: readonly DoryDesignDecisionId[];
  readonly viewMode: DorySpatialViewMode;
  readonly scenarioKey: DoryTourScenarioKey;
  readonly branchId: string;
  readonly recordId: string;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  /** Full verified 8-tuple execution coordinate (`Audit §6.3`) bound to the isolated tour session. */
  readonly coord: DoryExecutionCoord;
  /** Fully resolved focus state merged over `DEFAULT_DORY_SPATIAL_FOCUS_STATE` with `recordId` set. */
  readonly focus: DorySpatialFocusState;
  readonly highlightPartId: string | null;
  readonly cameraOverride: DoryTourCameraOverride | null;
  /** 100% live scalar readouts computed from the isolated tour session's actual tensors & records. */
  readonly liveMetrics: readonly DoryTourLiveMetric[];
  readonly provenanceBadges: readonly DoryProvenanceBadge[];
}

// ============================================================================
// TOUR STATE, FREE-EXPLORE RESTORE & CONTROLLER CONTRACTS
// ============================================================================

/**
 * Caller Free-Explore UI & session cursor snapshot captured on `start()` and returned untouched
 * on `exit()`. The tour NEVER holds a reference to or mutates the caller's `DorySession`.
 */
export interface DoryFreeExploreRestoreState {
  readonly viewMode: DorySpatialViewMode;
  readonly focus: DorySpatialFocusState;
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
  readonly cameraPose?: CameraPose | null;
  readonly inspectorTab?: DoryInspectorTabId;
  readonly inspectorOpen?: boolean;
}

export interface DoryTourState {
  readonly status: DoryTourStatus;
  readonly active: boolean;
  readonly beatIndex: number;
  readonly beatCount: number;
  readonly actIndex: 0 | 1 | 2 | 3;
  readonly actCount: typeof DORY_TOUR_TOTAL_ACTS;
  readonly currentAct: DoryTourActSummary;
  readonly currentBeat: DoryTourBeatSnapshot;
  readonly acts: readonly DoryTourActSummary[];
  readonly activeBranchId: string;
  readonly activeRecordId: string;
  readonly viewMode: DorySpatialViewMode;
  readonly focus: DorySpatialFocusState;
  /** Shortcut to `currentBeat.coord` (full 8-tuple `DoryExecutionCoord`). */
  readonly coord: DoryExecutionCoord;
}

/**
 * Initialization options for `createDoryGuidedTour()`.
 * Deliberately accepts ONLY read-only shared singleton parameters/tables, an optional deterministic
 * seed, and an optional scene dispatcher — NEVER an external `DorySession`, guaranteeing
 * structural isolation from any caller Free-Explore session.
 */
export interface DoryTourOptions {
  readonly params?: DoryModelParameters;
  readonly ropeTables?: DoryDualRopeTables;
  readonly seed?: number;
  readonly sceneDispatcher?: DorySpatialSceneDispatcher;
}

export interface DoryGuidedTourController {
  readonly acts: readonly DoryTourActSummary[];
  readonly beatSpecs: readonly DoryTourBeatSpec[];
  readonly totalActs: typeof DORY_TOUR_TOTAL_ACTS;
  readonly totalBeats: number;
  /** Strictly private isolated `DorySession` owned exclusively by this guided tour instance. */
  readonly tourSession: DorySession;
  getStatus(): DoryTourStatus;
  isActive(): boolean;
  getState(): DoryTourState;
  getBeatCount(): number;
  getCurrentBeatIndex(): number;
  getCurrentBeat(): DoryTourBeatSnapshot;
  getBeat(beatIndex: number): DoryTourBeatSnapshot;
  getAllBeats(): readonly DoryTourBeatSnapshot[];
  getActs(): readonly DoryTourActSummary[];
  getRestoreState(): DoryFreeExploreRestoreState | null;
  /** Enter (or restart) the guided tour, storing `restoreState` untouched for `exit()`. */
  start(
    restoreState?: DoryFreeExploreRestoreState | null,
    initialBeatIndex?: number,
  ): DoryTourState;
  goToBeat(beatIndex: number): DoryTourState;
  nextBeat(): DoryTourState;
  prevBeat(): DoryTourState;
  goToAct(actId: DoryTourActId): DoryTourState;
  /** Build the live 5-pass WebGL2 `DorySpatialScene` for the current (or specified) beat. */
  buildBeatScene(beatIndex?: number): DorySpatialScene;
  /** Resolve a non-mutating `DoryResolvedSessionContext` for the current (or specified) beat. */
  getTourContext(beatIndex?: number): DoryResolvedSessionContext;
  /** Audit bounded `Set<ArrayBufferLike>` memory (`<= 12.00 MiB`) of the isolated tour session. */
  measureTourMemory(): DorySessionMemoryAudit;
  /** Exit the tour and return the exact saved Free-Explore restore snapshot (`null` if none). */
  exit(): DoryFreeExploreRestoreState | null;
}

export interface DoryExecutionCoordValidationResult {
  readonly valid: boolean;
  readonly errors: readonly string[];
}

/** Frozen module export contract implemented by `src/dory/story/walkthrough.ts` (Slice 5B). */
export interface DoryWalkthroughModuleContract {
  readonly DORY_TOUR_BEATS: readonly DoryTourBeatSpec[];
  validateDoryExecutionCoord(
    coord: DoryExecutionCoord,
    expectedModelWeightId?: string,
    kLoops?: number,
  ): DoryExecutionCoordValidationResult;
  createDoryGuidedTour(options?: DoryTourOptions): DoryGuidedTourController;
}

// ============================================================================
// HEADLESS ARCHITECTURE READOUTS & READ-ONLY TBPTL CONTRACTS (`archReadouts.ts`)
// ============================================================================

export interface DoryDualByteReadout {
  readonly bytes: number;
  /** Decimal SI formatting (`B`, `KB`, `MB`, `GB`). */
  readonly siLabel: string;
  /** Binary IEC formatting (`B`, `KiB`, `MiB`, `GiB`). */
  readonly iecLabel: string;
  /** Combined editorial display string, e.g. `"7.405 GiB (7.951 GB)"`. */
  readonly dualLabel: string;
}

/** Enriched 0-byte-diff 4-checkpoint production readout row (`Audit §3.2`, `DR14`–`DR16`). */
export interface DoryCheckpointFormatReadoutRow extends DoryCheckpointFormatSpec {
  readonly shortSha: string;
  readonly payloadFormatted: DoryDualByteReadout | null;
  readonly totalFormatted: DoryDualByteReadout;
  readonly unquantizedBf16Bytes: number | null;
  readonly quantizedPayloadBytes: number | null;
  readonly zeroDiffVerified: boolean;
  /** Verified container reconciliation formula check (`8 + headerJsonBytes + payloadBytes === totalFileBytes`). */
  readonly safetensorsContainerExact: boolean | null;
  readonly discrepancyRefs: readonly DoryDiscrepancyId[];
  readonly dgxSparkReadoutNote: string;
}

/** Exact BF16 stage & component parameter breakdown row (`Audit §3.2`, summing to `3,975,229,440`). */
export interface DoryStageParamBreakdownRow {
  readonly id: string;
  readonly stageOrComponent: string;
  readonly physicalRangeLabel: string;
  readonly tensorCount: number;
  readonly parameters: number;
  readonly parametersFormatted: string;
  readonly shareOfParamsPct: number;
  readonly bf16Payload: DoryDualByteReadout;
  readonly quantizedInMxfp8Nvfp4: boolean | 'partial' | 'n/a';
  readonly quantNote: string;
  readonly isSubtotal?: boolean;
}

/** Effective depth vs. linear compute & KV bank scaling row (`Audit §6.1`). */
export interface DoryDepthComputeScalingRow {
  readonly kLoops: number;
  readonly isTrained: boolean;
  readonly isDefault: boolean;
  readonly physicalSublayers: 72;
  readonly executedSublayers: number;
  readonly effectivePairedDepth: number;
  readonly effectiveLinearBodyParams: number;
  readonly effectiveLinearWithHeadParams: number;
  /** Exact body ratio `(4 + 5k) / 9` vs. `k = 1`. */
  readonly bodyComputeRatioVsK1: number;
  /** Single-token linear ratio including `k`-invariant untied LM head. */
  readonly linearWithHeadRatioVsK1: number;
  readonly perLoopTotalBanks: number;
  readonly perLoopSwaBanks: number;
  readonly perLoopFullBanks: number;
  readonly lastLoopTotalBanks: 36;
  readonly perLoopToLastLoopKvRatio: number;
}

/** Production `1 -> 256K` BF16 logical KV scaling row (`Audit §7.2`, `S`-invariant `(4 + 5k) / 9`). */
export interface DoryProductionKvScalingRow {
  readonly seqLen: number;
  readonly seqLenLabel: string;
  readonly isSwaKnee: boolean;
  readonly lastLoopPoint: DoryProductionKvPoint;
  readonly perLoopK2Point: DoryProductionKvPoint;
  readonly perLoopK4Point: DoryProductionKvPoint;
  readonly perLoopK8Point: DoryProductionKvPoint;
  readonly lastLoopFormatted: DoryDualByteReadout;
  readonly perLoopK2Formatted: DoryDualByteReadout;
  readonly perLoopK4Formatted: DoryDualByteReadout;
  readonly perLoopK8Formatted: DoryDualByteReadout;
  readonly swaSharePct: number;
  readonly marginalBytesPerTokLastLoop: number;
  readonly marginalBytesPerTokPerLoopK4: number;
}

/** Analytical conditional `D10` memory bound & live session boundary readout row (`Audit §11.4`). */
export interface DoryMemoryBoundReadoutRow {
  readonly id:
    | 'singleton_weights'
    | 'kv_k4_spec_104'
    | 'kv_k8_extrap_176'
    | 'd10_lru_pool'
    | 'session_bound_k4'
    | 'session_bound_k8'
    | 'rg32f_atlas_cap'
    | 'live_session_retained';
  readonly label: string;
  readonly bytes: number;
  readonly formatted: DoryDualByteReadout;
  readonly capBytes: number | null;
  readonly capFormatted: string | null;
  readonly withinCap: boolean;
  readonly note: string;
}

export type DoryDiscrepancyFilterCategory = 'all' | DoryDiscrepancyCategory;

export interface DoryDiscrepancyFilterOptions {
  readonly query?: string;
  readonly category?: DoryDiscrepancyFilterCategory;
  readonly onlyBrowserGuards?: boolean;
  readonly ids?: readonly DoryDiscrepancyId[];
}

export interface DoryDiscrepancyTableModel {
  readonly totalCount: number;
  readonly browserGuardCount: number;
  readonly filteredCount: number;
  readonly query: string;
  readonly category: DoryDiscrepancyFilterCategory;
  readonly onlyBrowserGuards: boolean;
  readonly rows: readonly DoryDiscrepancyEntry[];
}

/** Strictly read-only pretraining / post-training stage descriptor (`Audit §1.4 & §12.1 View 7`). */
export interface DoryPretrainingStageReadout {
  readonly stageNumber: 1 | 2 | 3 | 4 | 5;
  readonly id: 'pretrain_stage1' | 'pretrain_stage2' | 'pretrain_stage3' | 'posttrain_sft' | 'posttrain_rlvr';
  readonly title: string;
  readonly tokenBudgetLabel: string;
  readonly precisionAndDepthLabel: string;
  readonly maxContextLabel: string;
  readonly keyMechanism: string;
}

/**
 * Strictly read-only TBPTL depth-1 & training schematic data (`Audit §1.4 & §12.1 View 7`).
 * Guarantees zero gradients, zero training execution, and zero fake quantization toggles.
 */
export interface DoryTbptlSchematicReadout {
  readonly readOnly: true;
  readonly hasGradients: false;
  readonly zeroTrainingLoops: true;
  readonly zeroFakeQuantToggles: true;
  readonly badge: typeof DORY_TBPTL_READONLY_BADGE;
  readonly tbptlDepth: 1;
  readonly stopGradientBetweenLoops: true;
  readonly earlyExitLosses: 0;
  readonly preludeGradientPath: string;
  readonly elrRankOverride: 0.1;
  readonly rlvrDifficultyRatio: '4:1:1:1';
  readonly rlvrAlpha: 0.3;
  readonly stages: readonly DoryPretrainingStageReadout[];
  readonly epistemicCaveat: string;
}

/** Paper-reported empirical result entry explicitly separated from synthetic miniature behavior. */
export interface DoryPaperResultReadout {
  readonly id: string;
  readonly badge: typeof DORY_PAPER_RESULTS_BADGE;
  readonly metricLabel: string;
  readonly reportedValue: string;
  readonly provenance: string;
  readonly caveat: string;
  readonly discrepancyRefs: readonly DoryDiscrepancyId[];
}

export interface DoryArchReadoutsInput {
  readonly activeKLoops?: number;
  readonly activeKvCacheMode?: DoryKvCacheMode;
  readonly selectedCheckpointFormat?: DoryCheckpointFormatId;
  readonly liveSessionMemory?: DorySessionMemoryAudit | null;
  readonly activeAtlasBytes?: number | null;
  readonly discrepancyFilter?: DoryDiscrepancyFilterOptions;
}

export interface DoryArchReadoutsBundle {
  readonly readOnly: true;
  readonly zeroFakeQuantToggles: true;
  readonly activeKLoops: number;
  readonly activeKvCacheMode: DoryKvCacheMode;
  readonly selectedCheckpointFormat: DoryCheckpointFormatId;
  readonly checkpoints: readonly DoryCheckpointFormatReadoutRow[];
  /** True iff all 3 safetensors checkpoints reconcile to exact `0 B` diff against HF LFS sizes. */
  readonly allSafetensorsZeroByteDiff: true;
  readonly bf16StageBreakdown: readonly DoryStageParamBreakdownRow[];
  readonly scalingDisclosure: readonly DoryScalingRatioEntry[];
  readonly depthComputeScaling: readonly DoryDepthComputeScalingRow[];
  readonly productionKvScaling: readonly DoryProductionKvScalingRow[];
  /** Exact `S`-invariant ratio `(4 + 5k) / 9` for `activeKLoops` (`Audit §7.2`). */
  readonly activeSInvariantRatio: number;
  /** Exact `4x` marginal bytes/token slope drop across `S = 1,024` (`Audit §7.2`). */
  readonly swaKneeSlopeDropFactor: 4;
  readonly memoryBounds: readonly DoryMemoryBoundReadoutRow[];
  readonly discrepancyLedger: DoryDiscrepancyTableModel;
  readonly designDecisions: readonly DoryDesignDecisionEntry[];
  readonly tbptlSchematic: DoryTbptlSchematicReadout;
  readonly paperResults: readonly DoryPaperResultReadout[];
  readonly governanceScopeNote: string;
}

/** Frozen module export contract implemented by `src/dory/archReadouts.ts` (Slice 5A). */
export interface DoryArchReadoutsModuleContract {
  formatDoryDualBytes(bytes: number): DoryDualByteReadout;
  buildDoryCheckpointReadouts(): readonly DoryCheckpointFormatReadoutRow[];
  buildDoryBf16StageBreakdown(): readonly DoryStageParamBreakdownRow[];
  buildDoryDepthComputeScalingTable(): readonly DoryDepthComputeScalingRow[];
  buildDoryProductionKvScalingTable(): readonly DoryProductionKvScalingRow[];
  buildDoryMemoryBoundsTable(
    liveSessionMemory?: DorySessionMemoryAudit | null,
    activeAtlasBytes?: number | null,
  ): readonly DoryMemoryBoundReadoutRow[];
  filterDoryDiscrepancyLedger(options?: DoryDiscrepancyFilterOptions): DoryDiscrepancyTableModel;
  buildDoryTbptlSchematicReadout(): DoryTbptlSchematicReadout;
  buildDoryPaperResultsReadouts(): readonly DoryPaperResultReadout[];
  buildDoryArchReadouts(input?: DoryArchReadoutsInput): DoryArchReadoutsBundle;
}
