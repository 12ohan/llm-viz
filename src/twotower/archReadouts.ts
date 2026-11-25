/**
 * Stage TT4 Headless Architectural Readouts, Context-Length Memory Scaling,
 * Persistent Nested Generation Timeline, Config Scope, and Advanced Sampler View-Models
 * (`src/twotower/archReadouts.ts`)
 *
 * Pure headless TypeScript — ZERO DOM, ZERO WebGL, ZERO scene/UI imports, ZERO `Math.random()`.
 * All returned view-models are deeply immutable (`Object.freeze`) and strictly `Infinity`-free.
 *
 * Composes authoritative sources directly without duplicating numerical kernels or specs:
 *   - `src/twotower/spec.ts` (`TWOTOWER_SPEC`, `TWOTOWER_DISCREPANCY_LEDGER`, `twoTowerMemoryFor`, `TWOTOWER_MINI_CONFIG`)
 *   - `src/twotower/session.ts` (`getActiveBranch`, `getActiveRecord`, `getLatestContextExecution`,
 *     `getSessionCapabilities`, `measureTwoTowerSessionRetainedBytes`)
 *   - `src/twotower/sampler.ts` (`TWOTOWER_CONFIDENCE_PRESETS`, `assertPresentationSafeNoInfinity`)
 *   - `src/model/memory.ts`, `src/model/spec.ts`, `src/math/{gqaSim,mambaSim,moeSim}.ts`
 */

import { GQA_SIM_CONFIG } from '../math/gqaSim';
import { MAMBA_SIM_CONFIG } from '../math/mambaSim';
import { MOE_SIM_CONFIG } from '../math/moeSim';
import {
  formatBytesAdaptive,
  formatGiB,
  formatKiB,
  formatMiB,
  formatTokenCount,
  memoryCrossovers,
} from '../model/memory';
import { NEMOTRON_SPEC } from '../model/spec';
import {
  TWOTOWER_CONFIDENCE_PRESETS,
  assertPresentationSafeNoInfinity,
  type ConfidenceCommitReason,
  type ConstrainedMdlmCandidate,
  type RawLmHeadCandidate,
  type SamplerSelectionMode,
  type SamplerSlotDecision,
  type SamplerStepPresentation,
  type TwoTowerConfidencePreset,
  type TwoTowerConfidencePresetId,
} from './sampler';
import {
  getActiveBranch,
  getActiveRecord,
  getLatestContextExecution,
  getSessionCapabilities,
  measureTwoTowerSessionRetainedBytes,
  type TwoTowerActiveBlockConfig,
  type TwoTowerSession,
  type TwoTowerSessionConfig,
  type TwoTowerSessionDiffusionConfidenceCommitRecord,
  type TwoTowerSessionLifecyclePhase,
  type TwoTowerSessionMemoryBreakdown,
  type TwoTowerSessionRecord,
  type TwoTowerSessionTermination,
  type TwoTowerSessionTransitionKind,
  type TwoTowerVariateProvenance,
  type TwoTowerVariateProvenanceMode,
} from './session';
import {
  TWOTOWER_COMMUNITY_NVFP4_REFERENCE,
  TWOTOWER_DISCREPANCY_LEDGER,
  TWOTOWER_MINI_CONFIG,
  TWOTOWER_SPEC,
  twoTowerMemoryFor,
  type CommunityNvfp4ReferenceNote,
  type TwoTowerDiscrepancyEntry,
  type TwoTowerExecutionMode,
  type TwoTowerMemoryReadout,
} from './spec';

// ============================================================================
// CONSTANTS
// ============================================================================

/** Standard context-length sweep anchors from 1K up to 1M tokens. */
export const TWOTOWER_STANDARD_CONTEXT_LENGTHS: readonly number[] = Object.freeze([
  1_024,
  4_096,
  16_384,
  65_536,
  131_072,
  262_144,
  524_288,
  1_048_576,
]);

/** Model card operational context note (128K) distinct from config.json `max_position_embeddings` (256K). */
export const TWOTOWER_MODEL_CARD_OPERATIONAL_CONTEXT_TOKENS = 131_072 as const;

/** Mandatory sampler calibration disclaimer stating confidence is a relative heuristic, not semantic truth. */
export const TWOTOWER_SAMPLER_CALIBRATION_DISCLAIMER =
  'Unscaled MDLM softmax probability (x_theta) is a relative confidence heuristic used solely to rank masked block slots for unmasking order — NOT a calibrated probability of semantic correctness or factual accuracy. Browser execution runs deterministic synthetic miniature weights (d = 8, V = 34), not 63.19B checkpoint probabilities.' as const;

const PRESET_MATCH_EPS = 1e-6;

// ============================================================================
// 1. PRODUCTION ARCHITECTURAL READOUTS & CONTEXT-LENGTH MEMORY SCALING TYPES
// ============================================================================

export type ContextLengthScalingRegime = 'supported' | 'extrapolated_beyond_max_pos';

export interface TwoTowerTimeConditioningComponentEntry {
  readonly id: 't_embedder_mlp' | 't_block' | 'scale_shift_tables';
  readonly label: string;
  readonly params: number;
  readonly tensorCount: number;
  readonly payloadBytes: number;
}

export interface TwoTowerTowerInventorySection {
  readonly contextTower: {
    readonly label: string;
    readonly params: number;
    readonly payloadBytes: number;
    readonly payloadDisplay: string;
    readonly tensorCount: number;
    readonly fp32BiasTensorCount: number;
    readonly fp32BiasParams: number;
    readonly activeParamsExclEmbHead: number;
    readonly activeParamsInclHeadExclEmb: number;
    readonly activeParamsInclEmbHead: number;
  };
  readonly denoiserBackbone: {
    readonly label: string;
    readonly params: number;
    readonly payloadBytes: number;
    readonly payloadDisplay: string;
    readonly tensorCount: number;
    readonly fp32BiasTensorCount: number;
    readonly fp32BiasParams: number;
  };
  readonly denoiserTimeConditioning: {
    readonly label: string;
    readonly params: number;
    readonly payloadBytes: number;
    readonly payloadDisplay: string;
    readonly tensorCount: number;
    readonly components: readonly TwoTowerTimeConditioningComponentEntry[];
  };
  readonly denoiserTotal: {
    readonly label: string;
    readonly params: number;
    readonly payloadBytes: number;
    readonly payloadDisplay: string;
    readonly tensorCount: number;
  };
  readonly grandTotal: {
    readonly params: number;
    readonly payloadBytes: number;
    readonly payloadDisplay: string;
    readonly diskBytes: number;
    readonly headerBytes: number;
    readonly tensorCount: number;
    readonly shardCount: number;
    readonly payloadByteDiscrepancy: number;
    readonly bf16TensorCount: number;
    readonly bf16Params: number;
    readonly bf16PayloadBytes: number;
    readonly fp32BiasTensorCount: number;
    readonly fp32BiasParams: number;
    readonly fp32BiasPayloadBytes: number;
    readonly fp32BiasExtraBytesOverBf16: number;
    readonly auditSummary: string;
  };
  readonly layerMix: {
    readonly totalLayersPerTower: 52;
    readonly mambaLayers: number;
    readonly gqaLayers: number;
    readonly moeLayers: number;
    readonly alignedTransferBays: number;
    readonly statelessMoeBays: number;
    readonly mambaLayerParams: number;
    readonly mambaAllLayersParams: number;
    readonly gqaLayerParams: number;
    readonly gqaAllLayersParams: number;
    readonly moeLayerParams: number;
    readonly moeAllLayersParams: number;
  };
  readonly communityNvfp4Reference: CommunityNvfp4ReferenceNote;
}

export interface TwoTowerDiscrepancyDisplayEntry extends TwoTowerDiscrepancyEntry {
  readonly index: number;
  readonly shortSummary: string;
}

export interface TwoTowerWeightsVsCachesBreakdown {
  readonly headline: 'Two Resident Weight Sets ≠ Two Persistent Caches';
  readonly mode: TwoTowerExecutionMode;
  readonly prefixTokens: number;
  readonly blockSize: number;
  readonly memoryReadout: TwoTowerMemoryReadout;
  readonly residentWeights: {
    readonly activeTowerCount: 1 | 2;
    readonly totalCheckpointBytes: number;
    readonly totalCheckpointDisplay: string;
    readonly activeWeightBytes: number;
    readonly activeWeightDisplay: string;
    readonly contextTowerBytes: number;
    readonly contextTowerDisplay: string;
    readonly denoiserTowerBytes: number;
    readonly denoiserTowerDisplay: string;
    readonly note: string;
  };
  readonly persistentContextCache: {
    readonly owner: 'context_tower';
    readonly isUniquePersistentSequenceCache: true;
    readonly fixedMambaBytes: number;
    readonly fixedMambaDisplay: string;
    readonly gqaKvBytes: number;
    readonly gqaKvDisplay: string;
    readonly kvBytesPerToken: number;
    readonly kvPerTokenDisplay: string;
    readonly totalBytes: number;
    readonly totalDisplay: string;
    readonly note: string;
  };
  readonly transientDenoiserWorkspace: {
    readonly owner: 'denoiser_tower';
    readonly persistence: 'ephemeral_scratch';
    readonly workspaceKind: 'ephemeral_block_kv' | 'mock_ar_mamba_s2_snapshot' | 'none';
    readonly ephemeralBlockKvBytes: number;
    readonly ephemeralBlockKvDisplay: string;
    readonly mockArMambaS2SnapshotBytes: number;
    readonly mockArMambaS2SnapshotDisplay: string;
    readonly totalTransientBytes: number;
    readonly totalTransientDisplay: string;
    readonly discardedAfterStep: boolean;
    readonly note: string;
  };
  readonly crossGpuMirrorVsLogicalSharing: {
    readonly logicalPersistentCacheCount: 1;
    readonly logicalSharingNote: string;
    readonly crossGpuMirrorBytes: number;
    readonly crossGpuMirrorDisplay: string;
    readonly crossGpuMirrorNote: string;
  };
  readonly totalActiveLogicalCacheBytes: number;
  readonly totalActiveLogicalCacheDisplay: string;
}

export interface TwoTowerPrecisionComparisonRow {
  readonly id: string;
  readonly dimension: string;
  readonly productionValue: string;
  readonly miniatureValue: string;
  readonly structuralFidelity: 'exact_preserved' | 'ratio_preserved' | 'scaled_miniature';
  readonly note: string;
}

export interface TwoTowerPrecisionVsMiniatureMatrix {
  readonly epistemicBanner: string;
  readonly production: {
    readonly primaryWeightsAndActivations: 'BF16';
    readonly ssmScanPrecision: 'FP32';
    readonly moeBiasPrecision: 'FP32';
    readonly moeRouterPrecision: 'FP64';
    readonly d: number;
    readonly vocabSize: number;
    readonly gqaQueryHeads: number;
    readonly gqaKvHeads: number;
    readonly gqaHeadDim: number;
    readonly mambaHeads: number;
    readonly mambaGroups: number;
    readonly mambaHeadDim: number;
    readonly mambaStateDim: number;
    readonly mambaConvKernel: number;
    readonly moeRoutedExperts: number;
    readonly moeActiveTopK: number;
    readonly moeRoutedWidth: number;
    readonly moeSharedExperts: number;
    readonly moeSharedWidth: number;
    readonly maxPositionEmbeddings: number;
    readonly positionalEncoding: 'NoPE';
    readonly defaultBlockSize: number;
    readonly defaultStepsPerBlock: number;
    readonly timeEmbedderFreqDim: number;
  };
  readonly miniature: {
    readonly primaryWeightsAndActivations: 'Float32 (synthetic)';
    readonly ssmScanPrecision: 'Float32';
    readonly moeBiasPrecision: 'Float32';
    readonly moeRouterPrecision: 'Float64 (preserved)';
    readonly d: number;
    readonly vocabSize: number;
    readonly kvCapacity: number;
    readonly gqaQueryHeads: number;
    readonly gqaKvHeads: number;
    readonly gqaHeadDim: number;
    readonly mambaHeads: number;
    readonly mambaGroups: number;
    readonly mambaHeadDim: number;
    readonly mambaStateDim: number;
    readonly mambaSsmMatrixShape: readonly [number, number];
    readonly mambaConvKernel: number;
    readonly moeRoutedExperts: number;
    readonly moeActiveTopK: number;
    readonly moeRoutedWidth: number;
    readonly moeSharedExperts: number;
    readonly moeSharedWidth: number;
    readonly supportedBlockSizes: readonly [4, 8];
    readonly timeEmbedderFreqDim: number;
  };
  readonly rows: readonly TwoTowerPrecisionComparisonRow[];
}

export interface ContextLengthScalingPoint {
  readonly tokens: number;
  readonly tokenLabel: string;
  readonly isCustomPoint: boolean;
  readonly isSelectedPrefix: boolean;
  readonly regime: ContextLengthScalingRegime;
  readonly exceedsModelCardOperational128k: boolean;
  readonly readout: TwoTowerMemoryReadout;
  readonly persistentMambaBytes: number;
  readonly persistentKvBytes: number;
  readonly persistentTotalBytes: number;
  readonly persistentTotalDisplay: string;
  readonly ephemeralWorkspaceBytes: number;
  readonly ephemeralWorkspaceDisplay: string;
  readonly ephemeralDenoiserBlockKvBytes: number;
  readonly mockArMambaS2SnapshotBytes: number;
  readonly totalActiveCacheBytes: number;
  readonly denoiserReadOnlyMirrorBytes: number;
  readonly mambaSharePct: number;
  readonly kvSharePct: number;
  readonly reference52LayerGqaKvBytes: number;
  readonly hybridVsFullGqaSavingsRatio: number;
}

export interface TwoTowerContextLengthScalingSeries {
  readonly blockSize: number;
  readonly mode: TwoTowerExecutionMode;
  readonly maxPositionEmbeddings: number;
  readonly modelCardOperationalContextTokens: number;
  readonly fixedMambaBytes: number;
  readonly fixedMambaDisplay: string;
  readonly kvBytesPerToken: number;
  readonly kvPerTokenDisplay: string;
  /** First integer context length where growing GQA KV >= fixed 47.08 MiB Mamba-2 state (`8,035` tokens). */
  readonly crossoverTokens: number;
  /** Exact continuous crossover ratio `fixedMambaBytes / kvBytesPerToken` (`~8,034.67` tokens). */
  readonly crossoverTokensExact: number;
  /** Exact token count where hybrid total equals 52-layer GQA reference (`1,048` tokens in BF16). */
  readonly beatsFull52LayerGqaTokens: number;
  readonly points: readonly ContextLengthScalingPoint[];
  readonly extrapolationDisclaimer: string;
}

export interface TwoTowerArchitecturalReadout {
  readonly prefixTokens: number;
  readonly blockSize: number;
  readonly mode: TwoTowerExecutionMode;
  readonly provenance: {
    readonly hfRepo: string;
    readonly hfRevision: string;
    readonly revisionCheckedDate: string;
    readonly paperArxivId: string;
    readonly architectureClass: string;
    readonly hybridOverridePattern: string;
    readonly numTowers: 2;
    readonly numHiddenLayersPerTower: 52;
    readonly crossTowerWeightSharing: false;
    readonly tieWordEmbeddings: false;
    readonly positionalEncoding: 'NoPE';
    readonly maxPositionEmbeddings: number;
  };
  readonly inventory: TwoTowerTowerInventorySection;
  readonly discrepancyLedger: readonly TwoTowerDiscrepancyDisplayEntry[];
  readonly weightsVsCaches: TwoTowerWeightsVsCachesBreakdown;
  readonly precisionVsMiniature: TwoTowerPrecisionVsMiniatureMatrix;
  readonly scaling: TwoTowerContextLengthScalingSeries;
}

// ============================================================================
// 2. PERSISTENT NESTED GENERATION TIMELINE & CONFIG SCOPE TYPES
// ============================================================================

export type TimelineSegmentKind = 'prefill' | 'ar_step' | 'mock_ar_step' | 'diffusion_block';

export type TimelineBadgeTone = 'context' | 'denoiser' | 'gold' | 'warn' | 'stop' | 'info' | 'muted';

export interface TwoTowerTimelineBadge {
  readonly id: string;
  readonly label: string;
  readonly tone: TimelineBadgeTone;
}

export interface TwoTowerTimelineMaskTransition {
  readonly nMaskedBefore: number;
  readonly nMaskedAfter: number;
  readonly thresholdCount: number;
  readonly floorCount: number;
  readonly committedCount: number;
  readonly remaskedCount: number;
  readonly commitReason: ConfidenceCommitReason;
}

export interface TwoTowerTimelinePrefixTransition {
  readonly before: number;
  readonly after: number;
  readonly delta: number;
}

export interface TimelineNode {
  /** Zero-based index inside `activeBranch.records` (pass directly to `navigateTwoTowerSession`). */
  readonly cursorIndex: number;
  /** Monotonic logical record index from `record.identity.recordIndex` (may exceed `cursorIndex` after trimming). */
  readonly recordIndex: number;
  readonly recordId: string;
  readonly parentRecordId: string | null;
  readonly kind: TwoTowerSessionTransitionKind;
  readonly lifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly label: string;
  readonly shortLabel: string;
  readonly blockId: number | null;
  readonly refinementIndex: number | null;
  readonly isActiveCursor: boolean;
  /** True when this record lies strictly prior to the branch tail (including when inspecting a rewound cursor). */
  readonly isHistorical: boolean;
  readonly isBeforeCursor: boolean;
  readonly isAfterCursor: boolean;
  readonly isBranchTail: boolean;
  /** Exact executed timestep `t = nMasked / B` for denoiser/commit nodes, else `null`. */
  readonly executedTimestep: number | null;
  /** Current unresolved `[MASK]` count on diffusion nodes (`0` when fully resolved), `null` on AR/prefill nodes. */
  readonly unresolvedCount: number | null;
  readonly maskTransition: TwoTowerTimelineMaskTransition | null;
  readonly prefixTransition: TwoTowerTimelinePrefixTransition | null;
  /** Distinct milestone flag when `nMaskedAfter === 0 && !contextCommitted`. */
  readonly isCompletedUncommitted: boolean;
  readonly terminationMarker: 'eos_terminated' | 'capacity_exhausted' | null;
  readonly stateBadges: readonly TwoTowerTimelineBadge[];
  readonly contextPrefixLen: number;
  readonly kvCapacity: number;
  readonly absoluteTokenRange: readonly [number, number];
  readonly cacheDigest: string;
  readonly cacheDigestShort: string;
  readonly snapshotDigestShort: string;
  readonly modeAtStep: TwoTowerExecutionMode;
}

export interface TimelineSegment {
  readonly segmentId: string;
  readonly segmentIndex: number;
  readonly kind: TimelineSegmentKind;
  readonly label: string;
  readonly blockId: number | null;
  readonly blockSize: number | null;
  readonly stepsPerBlock: number | null;
  readonly absoluteTokenRange: readonly [number, number];
  readonly containsActiveCursor: boolean;
  readonly isHistorical: boolean;
  readonly completionState: 'initialized' | 'refining' | 'completed_uncommitted' | 'committed';
  readonly isCompletedUncommitted: boolean;
  readonly isCommittedToContext: boolean;
  readonly nfe: number;
  readonly earlyCompleted: boolean;
  readonly terminationMarker: 'eos_terminated' | 'capacity_exhausted' | null;
  readonly hasTrimmedHead: boolean;
  readonly nodes: readonly TimelineNode[];
}

export interface TwoTowerTimelineBranchEntry {
  readonly branchId: string;
  readonly label: string;
  readonly isActive: boolean;
  readonly recordCount: number;
  readonly cursorIndex: number;
  readonly isAtTail: boolean;
  readonly parentBranchId: string | null;
  readonly forkPointRecordId: string | null;
  readonly forkPointIndex: number | null;
  readonly activeLifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly tailLifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly committedPrefixLen: number;
  readonly isDeletable: boolean;
}

export interface TwoTowerTimelineRetainedHistory {
  readonly retainedRecordCount: number;
  readonly totalUniqueRecordCount: number;
  readonly maxRecords: number;
  readonly branchCount: number;
  readonly maxBranches: number;
  readonly prunedRecordCount: number;
  /** True when history trimming evicted all Context-executing records at/before cursor (`getLatestContextExecution === null`). */
  readonly hasPrunedContextTraces: boolean;
  /** True when any retained Context execution released eager bay traces (still 0-ULP lazily reconstructible). */
  readonly hasReleasedEagerTraces: boolean;
  readonly retainedBytes: number;
  readonly retainedBytesDisplay: string;
  readonly memoryBreakdown: TwoTowerSessionMemoryBreakdown;
}

export interface TwoTowerTimelineNavigationCapabilities {
  readonly canBack: boolean;
  readonly canForward: boolean;
  readonly canFork: boolean;
  readonly canRestart: true;
  /** Always `false` per session contract: the active branch cannot be deleted until switching away. */
  readonly canDeleteActiveBranch: false;
  readonly canDeleteAnyInactiveBranch: boolean;
  readonly canStepNext: boolean;
  readonly nextTransitionKind: TwoTowerSessionTransitionKind | null;
  readonly willAutoForkOnMutatingStep: boolean;
}

export interface TwoTowerTimelineModel {
  readonly sessionId: string;
  readonly activeBranchId: string;
  readonly activeBranchLabel: string;
  readonly activeCursorIndex: number;
  readonly activeRecordId: string;
  readonly activeRecordIndex: number;
  readonly activeLifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly isInspectingHistoricalCursor: boolean;
  readonly isTerminated: boolean;
  readonly termination: TwoTowerSessionTermination | null;
  readonly segments: readonly TimelineSegment[];
  readonly flatNodes: readonly TimelineNode[];
  readonly branches: readonly TwoTowerTimelineBranchEntry[];
  readonly retainedHistory: TwoTowerTimelineRetainedHistory;
  readonly navigation: TwoTowerTimelineNavigationCapabilities;
}

export type TwoTowerConfigApplicabilityScope =
  | 'next_refinement'
  | 'next_block'
  | 'next_step'
  | 'session_restart';

export type TwoTowerConfigParameterKey =
  | 'confidenceThreshold'
  | 'temperature'
  | 'samplingSeed'
  | 'blockSize'
  | 'stepsPerBlock'
  | 'mode'
  | 'eosTokenIds'
  | 'initialPromptTokens'
  | 'kvCapacity';

export interface TwoTowerConfigParameterScope {
  readonly key: TwoTowerConfigParameterKey;
  readonly label: string;
  readonly applicability: TwoTowerConfigApplicabilityScope;
  readonly historicalValue: number | string | readonly number[];
  readonly activeBlockValue: number | string | readonly number[] | null;
  readonly pendingValue: number | string | readonly number[];
  readonly differsFromHistorical: boolean;
  readonly differsFromActiveBlock: boolean;
  readonly scopeBadge: string;
  readonly scopeNote: string;
}

export interface TwoTowerConfigScopeDraftRestartOverrides {
  readonly initialPromptTokens?: readonly number[];
  readonly kvCapacity?: number;
}

export interface TwoTowerConfigScopeModel {
  readonly activeRecordIndex: number;
  readonly activeRecordKind: TwoTowerSessionTransitionKind;
  readonly lifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly isHistoricalCursor: boolean;
  readonly isTerminated: boolean;
  readonly historicalConfig: TwoTowerSessionConfig;
  readonly activeBlockConfig: TwoTowerActiveBlockConfig | null;
  readonly pendingConfig: TwoTowerSessionConfig;
  readonly sessionPromptTokens: readonly number[];
  readonly sessionKvCapacity: number;
  readonly hasPendingDiffFromHistorical: boolean;
  readonly hasPendingDiffFromActiveBlock: boolean;
  readonly diffKeysFromHistorical: readonly TwoTowerConfigParameterKey[];
  readonly diffKeysFromActiveBlock: readonly TwoTowerConfigParameterKey[];
  readonly badgeText: string;
  readonly epistemicNote: string;
  readonly parameters: readonly TwoTowerConfigParameterScope[];
  readonly parametersByKey: Readonly<Record<TwoTowerConfigParameterKey, TwoTowerConfigParameterScope>>;
}

// ============================================================================
// 3. ADVANCED SAMPLER INSPECTOR VIEW-MODEL TYPES
// ============================================================================

export type SamplerInspectorSourceKind = 'current_record' | 'prior_commit_record' | 'none';

export type SamplerInspectorSlotStatus =
  | 'locked_prior'
  | 'committed_threshold'
  | 'committed_budget_floor'
  | 'committed_final_step'
  | 'remasked';

export type SamplerInspectorSlotConfidence =
  | {
      readonly kind: 'locked_prior';
      readonly confidenceDisplay: 'Locked (prior step)';
      readonly numericConfidence: null;
    }
  | {
      readonly kind: 'unscaled_probability';
      readonly confidenceDisplay: string;
      readonly numericConfidence: number;
    };

export interface SamplerInspectorTokenInfo {
  readonly tokenId: number;
  readonly label: string;
  readonly isMask: boolean;
}

export interface SamplerInspectorRawLmHeadTop {
  readonly top1TokenId: number;
  readonly top1TokenLabel: string;
  readonly top1Logit: number;
  readonly top1Prob: number;
  readonly maskTokenProb: number;
  readonly topK: readonly RawLmHeadCandidate[];
}

export interface SamplerInspectorConstrainedMdlmTop {
  readonly maskTokenExcluded: true;
  readonly maskTokenProb: 0;
  readonly isCarryOverDelta: boolean;
  readonly top1TokenId: number;
  readonly top1TokenLabel: string;
  readonly top1LogProb: number;
  readonly top1Prob: number;
  readonly topK: readonly ConstrainedMdlmCandidate[];
}

export interface SamplerInspectorSlotRow {
  readonly blockPosition: number;
  readonly absolutePosition: number | null;
  readonly inputToken: SamplerInspectorTokenInfo;
  readonly outputToken: SamplerInspectorTokenInfo;
  readonly status: SamplerInspectorSlotStatus;
  /** Exact underlying `SamplerSlotDecision` from `sampler.ts` (preserves `committed_progress_floor` vs `committed_budget_floor`). */
  readonly rawDecision: SamplerSlotDecision;
  readonly rawLmHeadTop: SamplerInspectorRawLmHeadTop;
  readonly constrainedMdlmTop: SamplerInspectorConstrainedMdlmTop;
  readonly candidateMode: SamplerSelectionMode;
  readonly confidence: SamplerInspectorSlotConfidence;
  readonly unscaledConfidence: number | null;
  readonly confidenceDisplay: string;
  readonly passesStrictThreshold: boolean | null;
  readonly variateDigest: string | null;
}

export interface SamplerInspectorThresholdInfo {
  readonly value: number;
  readonly activeCursorThreshold: number;
  readonly matchedPresetId: TwoTowerConfidencePresetId | 'custom';
  readonly matchedPresetLabel: string;
  readonly presets: readonly TwoTowerConfidencePreset[];
}

export interface SamplerInspectorExecutedTimestepModel {
  readonly tModel: number;
  readonly blockSize: number;
  readonly stepIndex: number;
  readonly stepsPerBlock: number;
  readonly nMaskedBefore: number;
  readonly nMaskedAfter: number;
  readonly remainingSteps: number;
  readonly minCommitFloor: number;
  readonly commitReason: ConfidenceCommitReason;
  readonly earlyCompleted: boolean;
  readonly isBlockComplete: boolean;
}

export interface SamplerInspectorCommitmentDecomposition {
  readonly thresholdQualifiedCount: number;
  /** Additional masked slots committed purely to satisfy `minCommitFloor`, progress floor (`>= 1`), or final step. */
  readonly budgetFloorExtraCount: number;
  readonly progressFloorExtraCount: number;
  readonly budgetFloorOnlyExtraCount: number;
  readonly finalStepExtraCount: number;
  readonly totalCommittedThisStep: number;
  readonly remaskedCount: number;
  readonly priorLockedCount: number;
}

export interface TwoTowerAdvancedSamplerInspectorModel {
  readonly source: SamplerInspectorSourceKind;
  readonly hasSamplerCommit: boolean;
  readonly isCurrentRecord: boolean;
  readonly isPriorCommitRecord: boolean;
  readonly sourceRecordId: string | null;
  readonly sourceRecordIndex: number | null;
  readonly sourceCursorIndex: number | null;
  readonly sourceBlockId: number | null;
  readonly sourceRefinementIndex: number | null;
  readonly sourceNote: string;
  readonly samplerPresentation: SamplerStepPresentation | null;
  readonly threshold: SamplerInspectorThresholdInfo;
  readonly temperature: number;
  readonly variateMode: TwoTowerVariateProvenanceMode | null;
  readonly variateDigest: string | null;
  readonly samplingSeed: number | null;
  readonly derivationKey: string | null;
  readonly executedTModel: SamplerInspectorExecutedTimestepModel | null;
  readonly commitmentDecomposition: SamplerInspectorCommitmentDecomposition | null;
  readonly slots: readonly SamplerInspectorSlotRow[];
  readonly calibrationDisclaimer: string;
}

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

function validateNonNegativeInt(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a nonnegative integer, received ${value}`);
  }
}

function arrayEqualsNumber(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function matchConfidencePreset(gamma: number): {
  readonly id: TwoTowerConfidencePresetId | 'custom';
  readonly label: string;
} {
  for (const preset of TWOTOWER_CONFIDENCE_PRESETS) {
    if (Math.abs(gamma - preset.threshold) < PRESET_MATCH_EPS) {
      if (preset.id === 'mini_default') {
        return {
          id: 'mini_default',
          label: 'Miniature Educational Default (γ = 0.22)',
        };
      }
      if (preset.id === 'paper_default') {
        return {
          id: 'paper_default',
          label: 'Published Paper Operating Point (γ = 0.80)',
        };
      }
      return {
        id: 'reference_code_default',
        label: 'Reference-Code Default (γ = 0.90)',
      };
    }
  }
  return {
    id: 'custom',
    label: `Custom (γ = ${gamma.toFixed(2)})`,
  };
}

function mapSlotDecisionToStatus(decision: SamplerSlotDecision): SamplerInspectorSlotStatus {
  switch (decision) {
    case 'locked_prior':
      return 'locked_prior';
    case 'committed_threshold':
      return 'committed_threshold';
    case 'committed_progress_floor':
    case 'committed_budget_floor':
      return 'committed_budget_floor';
    case 'committed_final_step':
      return 'committed_final_step';
    case 'remasked':
      return 'remasked';
  }
}

function shortDigest(digest: string): string {
  return digest.slice(0, 8);
}

// ============================================================================
// 1. PRODUCTION ARCHITECTURAL READOUTS & CONTEXT-LENGTH MEMORY SCALING
// ============================================================================

/**
 * Compute the context-length memory scaling curve across standard lengths
 * `[1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]` (plus an optional custom
 * `prefixTokens` point inserted in ascending order without duplication).
 *
 * Strictly separates `supported` (`L <= 262,144`) from `extrapolated_beyond_max_pos`
 * (`L > 262,144`) relative to `config.json`'s `max_position_embeddings`, and computes the
 * exact GQA-KV-vs-fixed-Mamba-2 crossover (`8,035` tokens; continuous `~8,034.67`).
 */
export function buildContextLengthScalingSeries(
  blockSize: number = TWOTOWER_SPEC.productionDefaults.blockSize,
  mode: TwoTowerExecutionMode = 'mask_diffusion',
  customPrefixTokens?: number,
): TwoTowerContextLengthScalingSeries {
  validateNonNegativeInt(blockSize, 'blockSize');
  if (mode !== 'ar' && mode !== 'mock_ar' && mode !== 'mask_diffusion') {
    throw new RangeError(`Unsupported TwoTower execution mode: ${String(mode)}`);
  }
  if (customPrefixTokens !== undefined) {
    validateNonNegativeInt(customPrefixTokens, 'customPrefixTokens');
  }

  const maxPositionEmbeddings = TWOTOWER_SPEC.directlyInspectedMetadata.maxPositionEmbeddings;
  const standardSet = new Set<number>(TWOTOWER_STANDARD_CONTEXT_LENGTHS);
  const lengths = [...TWOTOWER_STANDARD_CONTEXT_LENGTHS];
  if (customPrefixTokens !== undefined && !standardSet.has(customPrefixTokens)) {
    lengths.push(customPrefixTokens);
    lengths.sort((a, b) => a - b);
  }

  const crossovers = memoryCrossovers(NEMOTRON_SPEC, 'bf16');
  const probe = twoTowerMemoryFor(1, blockSize, mode);
  const fixedMambaBytes = probe.persistentContextMambaBytes;
  const kvBytesPerToken = probe.kvBytesPerToken;
  const crossoverTokensExact = fixedMambaBytes / kvBytesPerToken;

  const points: ContextLengthScalingPoint[] = [];
  for (let i = 0; i < lengths.length; i++) {
    const tokens = lengths[i];
    const readout = twoTowerMemoryFor(tokens, blockSize, mode);
    const persistentMambaBytes = readout.persistentContextMambaBytes;
    const persistentKvBytes = readout.persistentContextKvBytes;
    const persistentTotalBytes = readout.persistentContextTotalBytes;
    const ephemeralWorkspaceBytes =
      readout.ephemeralDenoiserBlockKvBytes + readout.mockArMambaS2SnapshotBytes;
    const mambaSharePct =
      persistentTotalBytes > 0 ? (persistentMambaBytes / persistentTotalBytes) * 100 : 0;
    const kvSharePct =
      persistentTotalBytes > 0 ? (persistentKvBytes / persistentTotalBytes) * 100 : 0;
    const reference52LayerGqaKvBytes = readout.contextPrefixCache.referenceBytes;
    const hybridVsFullGqaSavingsRatio =
      persistentTotalBytes > 0 ? reference52LayerGqaKvBytes / persistentTotalBytes : 0;
    const regime: ContextLengthScalingRegime =
      tokens <= maxPositionEmbeddings ? 'supported' : 'extrapolated_beyond_max_pos';

    points.push(
      Object.freeze({
        tokens,
        tokenLabel: formatTokenCount(tokens),
        isCustomPoint: !standardSet.has(tokens),
        isSelectedPrefix: customPrefixTokens !== undefined && tokens === customPrefixTokens,
        regime,
        exceedsModelCardOperational128k:
          tokens > TWOTOWER_MODEL_CARD_OPERATIONAL_CONTEXT_TOKENS,
        readout,
        persistentMambaBytes,
        persistentKvBytes,
        persistentTotalBytes,
        persistentTotalDisplay: formatBytesAdaptive(persistentTotalBytes),
        ephemeralWorkspaceBytes,
        ephemeralWorkspaceDisplay: formatBytesAdaptive(ephemeralWorkspaceBytes),
        ephemeralDenoiserBlockKvBytes: readout.ephemeralDenoiserBlockKvBytes,
        mockArMambaS2SnapshotBytes: readout.mockArMambaS2SnapshotBytes,
        totalActiveCacheBytes: readout.totalActiveCacheBytes,
        denoiserReadOnlyMirrorBytes: readout.denoiserReadOnlyMirrorBytes,
        mambaSharePct,
        kvSharePct,
        reference52LayerGqaKvBytes,
        hybridVsFullGqaSavingsRatio,
      }),
    );
  }

  const series: TwoTowerContextLengthScalingSeries = Object.freeze({
    blockSize,
    mode,
    maxPositionEmbeddings,
    modelCardOperationalContextTokens: TWOTOWER_MODEL_CARD_OPERATIONAL_CONTEXT_TOKENS,
    fixedMambaBytes,
    fixedMambaDisplay: formatMiB(fixedMambaBytes, 2),
    kvBytesPerToken,
    kvPerTokenDisplay: formatKiB(kvBytesPerToken),
    crossoverTokens: crossovers.kvExceedsMambaTokens,
    crossoverTokensExact,
    beatsFull52LayerGqaTokens: crossovers.beatsGqaTokens,
    points: Object.freeze(points),
    extrapolationDisclaimer:
      'Points at 512K (524,288) and 1M (1,048,576) tokens extrapolate analytically beyond config.json max_position_embeddings = 262,144 (256K) under NoPE KV-cache arithmetic; the model card separately notes 128K operational context.',
  });

  assertPresentationSafeNoInfinity(series, 'contextLengthScalingSeries');
  return series;
}

/**
 * Build the complete headless production architectural readout for Nemotron-Labs-TwoTower:
 *   - Independent tower parameter & byte inventory verified to 0 bytes against the 24-shard header audit
 *   - Full paper-versus-checkpoint discrepancy ledger
 *   - Explicit "Two Resident Weight Sets ≠ Two Persistent Caches" breakdown
 *   - Production Precision vs Miniature Arithmetic comparison matrix verified against live sim configs
 *   - Context-length memory scaling series including `prefixTokens`
 */
export function buildTwoTowerArchitecturalReadout(
  prefixTokens: number,
  blockSize: number = TWOTOWER_SPEC.productionDefaults.blockSize,
  mode: TwoTowerExecutionMode = 'mask_diffusion',
): TwoTowerArchitecturalReadout {
  const mem = twoTowerMemoryFor(prefixTokens, blockSize, mode);
  const meta = TWOTOWER_SPEC.directlyInspectedMetadata;
  const inv = TWOTOWER_SPEC.derivedInventoryCalculations;
  const audit = meta.localCheckpointHeaderAudit;
  const align = TWOTOWER_SPEC.layerAlignment;
  const prod = TWOTOWER_SPEC.productionDefaults;
  const backbone = NEMOTRON_SPEC;

  const fp32BiasPerTowerTensors = meta.fp32ScoreCorrectionBiasTensorCount / 2;
  const fp32BiasPerTowerParams = meta.fp32ScoreCorrectionBiasParams / 2;

  const timeCondComponents: readonly TwoTowerTimeConditioningComponentEntry[] = Object.freeze([
    Object.freeze({
      id: 't_embedder_mlp' as const,
      label: 't_embedder.mlp (256 → 2,688 → 2,688)',
      params: inv.timeEmbedderMlpParams,
      tensorCount: 4,
      payloadBytes: inv.timeEmbedderMlpParams * 2,
    }),
    Object.freeze({
      id: 't_block' as const,
      label: 't_block (2,688 → 8,064 global shift/scale/gate)',
      params: inv.timeBlockParams,
      tensorCount: 2,
      payloadBytes: inv.timeBlockParams * 2,
    }),
    Object.freeze({
      id: 'scale_shift_tables' as const,
      label: '52 × scale_shift_table (3 × 2,688 per layer)',
      params: inv.scaleShiftTablesParams,
      tensorCount: 52,
      payloadBytes: inv.scaleShiftTablesParams * 2,
    }),
  ]);

  const inventory: TwoTowerTowerInventorySection = Object.freeze({
    contextTower: Object.freeze({
      label: 'Context Tower (Causal AR Backbone)',
      params: inv.singleTowerParams,
      payloadBytes: inv.contextTowerPayloadBytes,
      payloadDisplay: formatGiB(inv.contextTowerPayloadBytes, 2),
      tensorCount: meta.contextTowerTensorCount,
      fp32BiasTensorCount: fp32BiasPerTowerTensors,
      fp32BiasParams: fp32BiasPerTowerParams,
      activeParamsExclEmbHead: inv.singleTowerActiveParamsExclEmbHead,
      activeParamsInclHeadExclEmb: inv.singleTowerActiveParamsInclHeadExclEmb,
      activeParamsInclEmbHead: inv.singleTowerActiveParamsInclEmbHead,
    }),
    denoiserBackbone: Object.freeze({
      label: 'Denoiser Tower Backbone (Independent 52-Layer Spine)',
      params: inv.singleTowerParams,
      payloadBytes: inv.denoiserTowerBackbonePayloadBytes,
      payloadDisplay: formatGiB(inv.denoiserTowerBackbonePayloadBytes, 2),
      tensorCount: meta.denoiserTowerTensorCount,
      fp32BiasTensorCount: fp32BiasPerTowerTensors,
      fp32BiasParams: fp32BiasPerTowerParams,
    }),
    denoiserTimeConditioning: Object.freeze({
      label: 'Denoiser-Only adaLN-single Time Conditioning',
      params: inv.timeConditioningTotalParams,
      payloadBytes: inv.timeConditioningPayloadBytes,
      payloadDisplay: formatMiB(inv.timeConditioningPayloadBytes, 2),
      tensorCount: meta.timeConditioningTensorCount,
      components: timeCondComponents,
    }),
    denoiserTotal: Object.freeze({
      label: 'Denoiser Tower Total (Backbone + Time Conditioning)',
      params: inv.singleTowerParams + inv.timeConditioningTotalParams,
      payloadBytes: inv.denoiserTowerTotalPayloadBytes,
      payloadDisplay: formatGiB(inv.denoiserTowerTotalPayloadBytes, 2),
      tensorCount: meta.denoiserTowerTensorCount + meta.timeConditioningTensorCount,
    }),
    grandTotal: Object.freeze({
      params: inv.grandTotalParams,
      payloadBytes: inv.derivedPayloadBytes,
      payloadDisplay: formatGiB(inv.derivedPayloadBytes, 2),
      diskBytes: meta.safetensorsDiskBytes,
      headerBytes: audit.verifiedHeaderBytes,
      tensorCount: meta.safetensorsTensorCount,
      shardCount: meta.safetensorsShardCount,
      payloadByteDiscrepancy: inv.payloadByteDiscrepancy,
      bf16TensorCount: audit.bf16TensorCount,
      bf16Params: audit.bf16Params,
      bf16PayloadBytes: audit.bf16PayloadBytes,
      fp32BiasTensorCount: meta.fp32ScoreCorrectionBiasTensorCount,
      fp32BiasParams: meta.fp32ScoreCorrectionBiasParams,
      fp32BiasPayloadBytes: audit.fp32PayloadBytes,
      fp32BiasExtraBytesOverBf16: inv.fp32ScoreCorrectionBiasExtraBytes,
      auditSummary:
        '0-byte discrepancy vs 24-shard safetensors header audit (12,498 BF16 tensors + 46 FP32 e_score_correction_bias tensors = 12,544 tensors; 63,185,902,848 params; 126,371,817,472 B payload).',
    }),
    layerMix: Object.freeze({
      totalLayersPerTower: 52 as const,
      mambaLayers: align.mambaTransferBayCount,
      gqaLayers: align.gqaTransferBayCount,
      moeLayers: align.statelessMoeBayCount,
      alignedTransferBays: align.alignedTransferBayCount,
      statelessMoeBays: align.statelessMoeBayCount,
      mambaLayerParams: inv.mambaLayerParams,
      mambaAllLayersParams: inv.mambaAllLayersParams,
      gqaLayerParams: inv.gqaLayerParams,
      gqaAllLayersParams: inv.gqaAllLayersParams,
      moeLayerParams: inv.moeLayerParams,
      moeAllLayersParams: inv.moeAllLayersParams,
    }),
    communityNvfp4Reference: TWOTOWER_COMMUNITY_NVFP4_REFERENCE,
  });

  const discrepancyLedger: readonly TwoTowerDiscrepancyDisplayEntry[] = Object.freeze(
    TWOTOWER_DISCREPANCY_LEDGER.map((entry, idx) =>
      Object.freeze({
        ...entry,
        index: idx + 1,
        shortSummary: `${entry.topic}: ${entry.authoritativeResolution}`,
      }),
    ),
  );

  const totalTransientBytes = mem.ephemeralDenoiserBlockKvBytes + mem.mockArMambaS2SnapshotBytes;
  const workspaceKind =
    mode === 'mask_diffusion'
      ? ('ephemeral_block_kv' as const)
      : mode === 'mock_ar'
        ? ('mock_ar_mamba_s2_snapshot' as const)
        : ('none' as const);

  const transientNote =
    mode === 'mask_diffusion'
      ? `Ephemeral Denoiser block KV scratchpad (${blockSize} × 6 KiB = ${formatBytesAdaptive(mem.ephemeralDenoiserBlockKvBytes)}) concatenated read-only with Context prefix KV during a denoising step and discarded immediately afterward — 0 B persistent Denoiser sequence cache.`
      : mode === 'mock_ar'
        ? `Transient Mock-AR mamba_s2 boundary snapshot (${formatMiB(mem.mockArMambaS2SnapshotBytes, 2)}) holds Context's Mamba-2 state prior to the last token so Denoiser can re-run last_token; 0 B ephemeral block KV.`
        : 'AR mode executes strictly on Context Tower; Denoiser Tower is bypassed with 0 B scratch workspace.';

  const crossGpuMirrorNote =
    mode === 'mask_diffusion'
      ? `In a 2-GPU tower-split deployment, mirroring Context's read-only boundary (47.08 MiB S-1 Mamba-2 + ${prefixTokens.toLocaleString('en-US')} × 6 KiB prefix KV = ${formatBytesAdaptive(mem.denoiserReadOnlyMirrorBytes)}) to the Denoiser GPU is a read-only transport mirror — NOT a second independent writable persistent cache.`
      : mode === 'mock_ar'
        ? `In a 2-GPU tower-split deployment, Mock-AR mirrors mamba_s2 + KV[:-1] (${Math.max(0, prefixTokens - 1).toLocaleString('en-US')} prefix tokens = ${formatBytesAdaptive(mem.denoiserReadOnlyMirrorBytes)}) read-only to the Denoiser GPU.`
        : 'In AR mode, Denoiser is inactive so 0 B cross-GPU read-only mirror is transferred.';

  const weightsVsCaches: TwoTowerWeightsVsCachesBreakdown = Object.freeze({
    headline: 'Two Resident Weight Sets ≠ Two Persistent Caches' as const,
    mode,
    prefixTokens,
    blockSize,
    memoryReadout: mem,
    residentWeights: Object.freeze({
      activeTowerCount: mode === 'ar' ? (1 as const) : (2 as const),
      totalCheckpointBytes: mem.dualTowerBf16WeightBytes,
      totalCheckpointDisplay: formatGiB(mem.dualTowerBf16WeightBytes, 2),
      activeWeightBytes: mem.activeModelWeightBytes,
      activeWeightDisplay: formatGiB(mem.activeModelWeightBytes, 2),
      contextTowerBytes: mem.contextTowerBf16WeightBytes,
      contextTowerDisplay: formatGiB(mem.contextTowerBf16WeightBytes, 2),
      denoiserTowerBytes: mem.denoiserTowerBf16WeightBytes,
      denoiserTowerDisplay: formatGiB(mem.denoiserTowerBf16WeightBytes, 2),
      note:
        mode === 'ar'
          ? `AR mode keeps only the Context Tower weights active (${formatGiB(mem.contextTowerBf16WeightBytes, 2)}); the full dual-tower checkpoint remains ${formatGiB(mem.dualTowerBf16WeightBytes, 2)} across 2 untied towers.`
          : `Both independent towers reside in memory (${formatGiB(mem.contextTowerBf16WeightBytes, 2)} Context + ${formatGiB(mem.denoiserTowerBf16WeightBytes, 2)} Denoiser incl. 57.26 MiB adaLN = ${formatGiB(mem.dualTowerBf16WeightBytes, 2)}), yet only ONE persistent sequence cache exists.`,
    }),
    persistentContextCache: Object.freeze({
      owner: 'context_tower' as const,
      isUniquePersistentSequenceCache: true as const,
      fixedMambaBytes: mem.persistentContextMambaBytes,
      fixedMambaDisplay: formatMiB(mem.persistentContextMambaBytes, 2),
      gqaKvBytes: mem.persistentContextKvBytes,
      gqaKvDisplay: formatBytesAdaptive(mem.persistentContextKvBytes),
      kvBytesPerToken: mem.kvBytesPerToken,
      kvPerTokenDisplay: formatKiB(mem.kvBytesPerToken),
      totalBytes: mem.persistentContextTotalBytes,
      totalDisplay: formatBytesAdaptive(mem.persistentContextTotalBytes),
      note: `Single persistent sequence cache owned exclusively by Context Tower: ${formatMiB(mem.persistentContextMambaBytes, 2)} fixed Mamba-2 state (23 layers) + ${prefixTokens.toLocaleString('en-US')} × ${formatKiB(mem.kvBytesPerToken)} BF16 GQA KV (6 layers) = ${formatBytesAdaptive(mem.persistentContextTotalBytes)}.`,
    }),
    transientDenoiserWorkspace: Object.freeze({
      owner: 'denoiser_tower' as const,
      persistence: 'ephemeral_scratch' as const,
      workspaceKind,
      ephemeralBlockKvBytes: mem.ephemeralDenoiserBlockKvBytes,
      ephemeralBlockKvDisplay: formatBytesAdaptive(mem.ephemeralDenoiserBlockKvBytes),
      mockArMambaS2SnapshotBytes: mem.mockArMambaS2SnapshotBytes,
      mockArMambaS2SnapshotDisplay: formatBytesAdaptive(mem.mockArMambaS2SnapshotBytes),
      totalTransientBytes,
      totalTransientDisplay: formatBytesAdaptive(totalTransientBytes),
      discardedAfterStep: mode !== 'ar',
      note: transientNote,
    }),
    crossGpuMirrorVsLogicalSharing: Object.freeze({
      logicalPersistentCacheCount: 1 as const,
      logicalSharingNote:
        'Logical single-cache architecture: Denoiser consumes Context prefix KV[0..P-1] and S-1 Mamba-2 boundary strictly read-only without maintaining a second growing sequence cache.',
      crossGpuMirrorBytes: mem.denoiserReadOnlyMirrorBytes,
      crossGpuMirrorDisplay: formatBytesAdaptive(mem.denoiserReadOnlyMirrorBytes),
      crossGpuMirrorNote,
    }),
    totalActiveLogicalCacheBytes: mem.totalActiveCacheBytes,
    totalActiveLogicalCacheDisplay: formatBytesAdaptive(mem.totalActiveCacheBytes),
  });

  const precisionRows: readonly TwoTowerPrecisionComparisonRow[] = Object.freeze([
    Object.freeze({
      id: 'primary_weights_activations',
      dimension: 'Primary Weights & Activations',
      productionValue: 'BF16 (12,498 weight tensors, 117.69 GiB dual-tower payload)',
      miniatureValue: 'Float32 deterministic synthetic weights (zero checkpoint weights)',
      structuralFidelity: 'scaled_miniature' as const,
      note: 'Educational browser simulator runs deterministic IEEE-754 Float32 arithmetic instead of BF16 checkpoint weights.',
    }),
    Object.freeze({
      id: 'ssm_scan_and_moe_bias',
      dimension: 'Mamba-2 SSM Scan & MoE Bias',
      productionValue: 'FP32 SSM scan + 46 FP32 e_score_correction_bias tensors (5,888 params)',
      miniatureValue: 'Float32 SSM recurrence & Float32 e_score_correction_bias',
      structuralFidelity: 'exact_preserved' as const,
      note: 'Maintains FP32 stability for recurrent SSM state accumulation and MoE expert load-balancing bias.',
    }),
    Object.freeze({
      id: 'moe_router_compute',
      dimension: 'MoE Router Matmul & Sigmoid',
      productionValue: 'FP64 (Megatron-Core --moe-router-dtype fp64)',
      miniatureValue: 'FP64 (float64_reference policy preserved)',
      structuralFidelity: 'exact_preserved' as const,
      note: 'TwoTower miniature simulator preserves exact FP64 router projection, sigmoid, biased top-6 selection, and normalization.',
    }),
    Object.freeze({
      id: 'hidden_and_vocab',
      dimension: 'Hidden Width d & Untied Vocab V',
      productionValue: `d = ${backbone.hidden.toLocaleString('en-US')}, V = ${backbone.vocabSize.toLocaleString('en-US')} (mask=3 [INST], eos=[2, 11])`,
      miniatureValue: `d = ${TWOTOWER_MINI_CONFIG.d}, V = ${TWOTOWER_MINI_CONFIG.vocabSize} (mask=${TWOTOWER_MINI_CONFIG.maskTokenId} [MASK], eos=${TWOTOWER_MINI_CONFIG.eosTokenId} <eos>)`,
      structuralFidelity: 'scaled_miniature' as const,
      note: 'Untied embedding and LM-head matrices are preserved in both towers.',
    }),
    Object.freeze({
      id: 'gqa_geometry',
      dimension: 'GQA Attention Heads & Head Width',
      productionValue: `${backbone.gqa.q} Q / ${backbone.gqa.kv} KV heads (16:1), d_head = ${backbone.gqa.headDim}, 6 layers/tower, NoPE`,
      miniatureValue: `${GQA_SIM_CONFIG.nQ} Q / ${GQA_SIM_CONFIG.nKv} KV heads (16:1), d_head = ${GQA_SIM_CONFIG.dHead}, 6 layers/tower, NoPE`,
      structuralFidelity: 'ratio_preserved' as const,
      note: 'Exact 32 Q / 2 KV (16:1) grouping and NoPE are preserved directly in GQA_SIM_CONFIG; head width scales 128 → 4.',
    }),
    Object.freeze({
      id: 'mamba_geometry',
      dimension: 'Mamba-2 Heads, Groups & SSM State',
      productionValue: `${backbone.mamba.heads} heads / ${backbone.mamba.groups} groups, ${backbone.mamba.channels}×${backbone.mamba.state} per-head SSM (d_state=${backbone.mamba.state}), conv k=${backbone.mamba.kernel}`,
      miniatureValue: `${MAMBA_SIM_CONFIG.nHeads} heads / ${MAMBA_SIM_CONFIG.nGroups} groups, ${MAMBA_SIM_CONFIG.dHead}×${MAMBA_SIM_CONFIG.dState} per-head SSM (d_state=${MAMBA_SIM_CONFIG.dState}), conv k=${MAMBA_SIM_CONFIG.convKernel}`,
      structuralFidelity: 'ratio_preserved' as const,
      note: 'Exact 8 B/C groups, Conv1D kernel k=4, and norm_before_gate=False are preserved in MAMBA_SIM_CONFIG.',
    }),
    Object.freeze({
      id: 'moe_geometry',
      dimension: 'Granular MoE Experts & Scaling',
      productionValue: `${backbone.moe.routed} routed (top-${backbone.moe.active}, w=${backbone.moe.width.toLocaleString('en-US')}) + ${backbone.moe.shared} shared (w=${backbone.moe.sharedWidth.toLocaleString('en-US')}), 2.5× scale`,
      miniatureValue: `${MOE_SIM_CONFIG.nRoutedExperts} routed (top-${MOE_SIM_CONFIG.topK}, d_exp=${MOE_SIM_CONFIG.dExp}) + 1 shared (d_shared=${MOE_SIM_CONFIG.dShared}), ${MOE_SIM_CONFIG.routedScalingFactor}× scale`,
      structuralFidelity: 'ratio_preserved' as const,
      note: 'Exact 128 routed / top-6 (95.3% sparsity), 2× double-wide shared expert, ReLU² non-gated MLP, and 2.5× scaling preserved.',
    }),
    Object.freeze({
      id: 'context_and_block',
      dimension: 'Context Window & Block Diffusion',
      productionValue: `max_position_embeddings = ${meta.maxPositionEmbeddings.toLocaleString('en-US')} (256K config; 128K card), B=${prod.blockSize}, K=${prod.stepsPerBlock}, freq_dim=${meta.timeEmbedderFreqDim}`,
      miniatureValue: `kvCapacity = ${TWOTOWER_MINI_CONFIG.kvCapacity}, B ∈ {${TWOTOWER_MINI_CONFIG.supportedBlockSizes.join(', ')}}, independent K, freq_dim = ${TWOTOWER_MINI_CONFIG.freqDim}`,
      structuralFidelity: 'scaled_miniature' as const,
      note: 'Miniature kvCapacity=24 keeps interactive browser sessions fast while enforcing exact capacity boundaries and independent K vs B.',
    }),
  ]);

  const precisionVsMiniature: TwoTowerPrecisionVsMiniatureMatrix = Object.freeze({
    epistemicBanner: prod.epistemicNote,
    production: Object.freeze({
      primaryWeightsAndActivations: 'BF16' as const,
      ssmScanPrecision: 'FP32' as const,
      moeBiasPrecision: 'FP32' as const,
      moeRouterPrecision: 'FP64' as const,
      d: backbone.hidden,
      vocabSize: backbone.vocabSize,
      gqaQueryHeads: backbone.gqa.q,
      gqaKvHeads: backbone.gqa.kv,
      gqaHeadDim: backbone.gqa.headDim,
      mambaHeads: backbone.mamba.heads,
      mambaGroups: backbone.mamba.groups,
      mambaHeadDim: backbone.mamba.channels,
      mambaStateDim: backbone.mamba.state,
      mambaConvKernel: backbone.mamba.kernel,
      moeRoutedExperts: backbone.moe.routed,
      moeActiveTopK: backbone.moe.active,
      moeRoutedWidth: backbone.moe.width,
      moeSharedExperts: backbone.moe.shared,
      moeSharedWidth: backbone.moe.sharedWidth,
      maxPositionEmbeddings: meta.maxPositionEmbeddings,
      positionalEncoding: 'NoPE' as const,
      defaultBlockSize: prod.blockSize,
      defaultStepsPerBlock: prod.stepsPerBlock,
      timeEmbedderFreqDim: meta.timeEmbedderFreqDim,
    }),
    miniature: Object.freeze({
      primaryWeightsAndActivations: 'Float32 (synthetic)' as const,
      ssmScanPrecision: 'Float32' as const,
      moeBiasPrecision: 'Float32' as const,
      moeRouterPrecision: 'Float64 (preserved)' as const,
      d: TWOTOWER_MINI_CONFIG.d,
      vocabSize: TWOTOWER_MINI_CONFIG.vocabSize,
      kvCapacity: TWOTOWER_MINI_CONFIG.kvCapacity,
      gqaQueryHeads: GQA_SIM_CONFIG.nQ,
      gqaKvHeads: GQA_SIM_CONFIG.nKv,
      gqaHeadDim: GQA_SIM_CONFIG.dHead,
      mambaHeads: MAMBA_SIM_CONFIG.nHeads,
      mambaGroups: MAMBA_SIM_CONFIG.nGroups,
      mambaHeadDim: MAMBA_SIM_CONFIG.dHead,
      mambaStateDim: MAMBA_SIM_CONFIG.dState,
      mambaSsmMatrixShape: Object.freeze([MAMBA_SIM_CONFIG.dHead, MAMBA_SIM_CONFIG.dState] as const),
      mambaConvKernel: MAMBA_SIM_CONFIG.convKernel,
      moeRoutedExperts: MOE_SIM_CONFIG.nRoutedExperts,
      moeActiveTopK: MOE_SIM_CONFIG.topK,
      moeRoutedWidth: MOE_SIM_CONFIG.dExp,
      moeSharedExperts: 1,
      moeSharedWidth: MOE_SIM_CONFIG.dShared,
      supportedBlockSizes: TWOTOWER_MINI_CONFIG.supportedBlockSizes,
      timeEmbedderFreqDim: TWOTOWER_MINI_CONFIG.freqDim,
    }),
    rows: precisionRows,
  });

  const scaling = buildContextLengthScalingSeries(blockSize, mode, prefixTokens);

  const readout: TwoTowerArchitecturalReadout = Object.freeze({
    prefixTokens,
    blockSize,
    mode,
    provenance: Object.freeze({
      hfRepo: TWOTOWER_SPEC.provenance.hfRepo,
      hfRevision: TWOTOWER_SPEC.provenance.hfRevision,
      revisionCheckedDate: TWOTOWER_SPEC.provenance.revisionCheckedDate,
      paperArxivId: TWOTOWER_SPEC.provenance.paperArxivId,
      architectureClass: meta.architectureClass,
      hybridOverridePattern: meta.hybridOverridePattern,
      numTowers: meta.numTowers,
      numHiddenLayersPerTower: meta.numHiddenLayersPerTower,
      crossTowerWeightSharing: meta.crossTowerWeightSharing,
      tieWordEmbeddings: meta.tieWordEmbeddings,
      positionalEncoding: meta.positionalEncoding,
      maxPositionEmbeddings: meta.maxPositionEmbeddings,
    }),
    inventory,
    discrepancyLedger,
    weightsVsCaches,
    precisionVsMiniature,
    scaling,
  });

  assertPresentationSafeNoInfinity(readout, 'twoTowerArchitecturalReadout');
  return readout;
}

// ============================================================================
// 2. PERSISTENT NESTED GENERATION TIMELINE VIEW-MODEL
// ============================================================================

function buildTimelineNode(
  rec: TwoTowerSessionRecord,
  cursorIndex: number,
  activeCursorIndex: number,
  branchLength: number,
): TimelineNode {
  const isActiveCursor = cursorIndex === activeCursorIndex;
  const isBranchTail = cursorIndex === branchLength - 1;
  const isHistorical = !isBranchTail;
  const isBeforeCursor = cursorIndex < activeCursorIndex;
  const isAfterCursor = cursorIndex > activeCursorIndex;

  let executedTimestep: number | null = null;
  let unresolvedCount: number | null = null;
  let maskTransition: TwoTowerTimelineMaskTransition | null = null;
  let prefixTransition: TwoTowerTimelinePrefixTransition | null = null;
  let label: string;
  let shortLabel: string;

  const isCompletedUncommitted =
    rec.activeBlock !== null &&
    rec.activeBlock.nMasked === 0 &&
    !rec.activeBlock.contextCommitted;

  const badges: TwoTowerTimelineBadge[] = [];

  switch (rec.kind) {
    case 'context_prefill': {
      prefixTransition = Object.freeze({
        before: 0,
        after: rec.committedPrefixLen,
        delta: rec.committedPrefixLen,
      });
      label = `Context Prefill (0 → ${rec.committedPrefixLen} tokens)`;
      shortLabel = `Prefill P=${rec.committedPrefixLen}`;
      badges.push(Object.freeze({ id: 'context_prefill', label: 'Context Prefill', tone: 'context' as const }));
      break;
    }
    case 'ar_step': {
      const prevP = Math.max(0, rec.committedPrefixLen - 1);
      prefixTransition = Object.freeze({
        before: prevP,
        after: rec.committedPrefixLen,
        delta: 1,
      });
      label = `AR Step #${rec.identity.absoluteTokenStart} → "${rec.selectedTokenLabel}" (${prevP} → ${rec.committedPrefixLen})`;
      shortLabel = `AR "${rec.selectedTokenLabel}"`;
      badges.push(Object.freeze({ id: 'ar_step', label: 'AR (Context-only)', tone: 'context' as const }));
      break;
    }
    case 'mock_ar_step': {
      const prevP = Math.max(0, rec.committedPrefixLen - 1);
      prefixTransition = Object.freeze({
        before: prevP,
        after: rec.committedPrefixLen,
        delta: 1,
      });
      label = `Mock-AR Step #${rec.identity.absoluteTokenStart} → "${rec.selectedTokenLabel}" (${prevP} → ${rec.committedPrefixLen})`;
      shortLabel = `Mock-AR "${rec.selectedTokenLabel}"`;
      badges.push(Object.freeze({ id: 'mock_ar_step', label: 'Mock-AR (Unmodulated)', tone: 'denoiser' as const }));
      break;
    }
    case 'diffusion_block_init': {
      unresolvedCount = rec.activeBlock.nMasked;
      const B = rec.blockConfig.blockSize;
      label = `Block #${rec.blockConfig.blockId} Init (${unresolvedCount}/${B} [MASK], K=${rec.blockConfig.stepsPerBlock})`;
      shortLabel = `Init B${rec.blockConfig.blockId} (${unresolvedCount}/${B})`;
      badges.push(
        Object.freeze({
          id: 'block_init',
          label: `Masked Block Init (${unresolvedCount}/${B})`,
          tone: 'denoiser' as const,
        }),
      );
      break;
    }
    case 'diffusion_refinement': {
      executedTimestep = rec.tModel;
      unresolvedCount = rec.activeBlock.nMasked;
      label = `Denoiser Pass r=${rec.refinementIndex} (t = ${rec.tModel.toFixed(2)}, ${unresolvedCount}/${rec.blockConfig.blockSize} masked)`;
      shortLabel = `Denoise r=${rec.refinementIndex} (t=${rec.tModel.toFixed(2)})`;
      badges.push(
        Object.freeze({
          id: 'denoiser_eval',
          label: `t = ${rec.tModel.toFixed(2)}`,
          tone: 'gold' as const,
        }),
      );
      break;
    }
    case 'diffusion_confidence_commit': {
      const u = rec.unmaskStep;
      executedTimestep = u.tModel;
      unresolvedCount = u.nMaskedAfter;
      const thresholdCount = u.thresholdCommittedIndices.length;
      const floorCount = u.floorCommittedIndices.length;
      maskTransition = Object.freeze({
        nMaskedBefore: u.nMaskedBefore,
        nMaskedAfter: u.nMaskedAfter,
        thresholdCount,
        floorCount,
        committedCount: u.tokensCommittedCount,
        remaskedCount: u.remaskedIndices.length,
        commitReason: u.commitReason,
      });
      label = `Confidence Commit r=${rec.refinementIndex} (${u.nMaskedBefore} → ${u.nMaskedAfter} masked · ${thresholdCount} thr / ${floorCount} floor)`;
      shortLabel = `Commit r=${rec.refinementIndex} (${u.nMaskedBefore}→${u.nMaskedAfter})`;
      badges.push(
        Object.freeze({
          id: 'confidence_commit',
          label: `${u.nMaskedBefore}→${u.nMaskedAfter} masked`,
          tone: 'denoiser' as const,
        }),
      );
      if (rec.activeBlock.pendingEosInBlock) {
        badges.push(
          Object.freeze({
            id: 'pending_eos',
            label: 'Pending EOS (stops on Context commit)',
            tone: 'warn' as const,
          }),
        );
      }
      break;
    }
    case 'diffusion_block_commit': {
      unresolvedCount = 0;
      const B = rec.blockConfig.blockSize;
      const prevP = Math.max(0, rec.committedPrefixLen - B);
      prefixTransition = Object.freeze({
        before: prevP,
        after: rec.committedPrefixLen,
        delta: B,
      });
      label = `Context Commit Block #${rec.blockConfig.blockId} (${prevP} → ${rec.committedPrefixLen}, NFE=${rec.nfe})`;
      shortLabel = `Commit B${rec.blockConfig.blockId} (${prevP}→${rec.committedPrefixLen})`;
      badges.push(
        Object.freeze({
          id: 'context_commit',
          label: `Context P: ${prevP} → ${rec.committedPrefixLen}`,
          tone: 'context' as const,
        }),
      );
      if (rec.earlyCompleted) {
        badges.push(
          Object.freeze({
            id: 'early_completed',
            label: `Early Completed (${rec.nfe}/${rec.blockConfig.stepsPerBlock} steps)`,
            tone: 'info' as const,
          }),
        );
      }
      break;
    }
  }

  if (isCompletedUncommitted) {
    badges.push(
      Object.freeze({
        id: 'completed_uncommitted',
        label: 'Completed · Uncommitted to Context',
        tone: 'warn' as const,
      }),
    );
  }

  let terminationMarker: 'eos_terminated' | 'capacity_exhausted' | null = null;
  if (rec.termination?.reason === 'eos') {
    terminationMarker = 'eos_terminated';
    badges.push(
      Object.freeze({
        id: 'eos_terminated',
        label: `EOS Terminated (tokens: ${rec.termination.triggerTokenIds.join(', ')})`,
        tone: 'stop' as const,
      }),
    );
  } else if (rec.termination?.reason === 'capacity_exhausted') {
    terminationMarker = 'capacity_exhausted';
    badges.push(
      Object.freeze({
        id: 'capacity_exhausted',
        label: `KV Capacity Exhausted (${rec.committedPrefixLen}/${rec.kvCapacity})`,
        tone: 'stop' as const,
      }),
    );
  }

  return Object.freeze({
    cursorIndex,
    recordIndex: rec.identity.recordIndex,
    recordId: rec.identity.recordId,
    parentRecordId: rec.identity.parentRecordId,
    kind: rec.kind,
    lifecyclePhase: rec.lifecyclePhase,
    label,
    shortLabel,
    blockId: rec.identity.blockId,
    refinementIndex: rec.identity.refinementIndex,
    isActiveCursor,
    isHistorical,
    isBeforeCursor,
    isAfterCursor,
    isBranchTail,
    executedTimestep,
    unresolvedCount,
    maskTransition,
    prefixTransition,
    isCompletedUncommitted,
    terminationMarker,
    stateBadges: Object.freeze(badges),
    contextPrefixLen: rec.committedPrefixLen,
    kvCapacity: rec.kvCapacity,
    absoluteTokenRange: rec.identity.absoluteTokenRange,
    cacheDigest: rec.identity.resultCacheDigest,
    cacheDigestShort: shortDigest(rec.identity.resultCacheDigest),
    snapshotDigestShort: shortDigest(rec.identity.resultSnapshotDigest),
    modeAtStep: rec.configAtStep.mode,
  });
}

function detectReleasedEagerContextTraces(session: TwoTowerSession): boolean {
  for (let i = 0; i < session.branches.length; i++) {
    const recs = session.branches[i].records;
    for (let j = 0; j < recs.length; j++) {
      const exec = recs[j].contextExecution;
      if (!exec) continue;
      for (let s = 0; s < exec.segments.length; s++) {
        if (exec.segments[s].retainedTraceBayIds().length === 0) {
          return true;
        }
      }
    }
  }
  return false;
}

/**
 * Build the persistent nested generation timeline view-model for the active branch of `session`.
 *
 * Groups records into top-level `TimelineSegment`s (`prefill`, `ar_step`, `mock_ar_step`, or
 * `diffusion_block` containing nested child `TimelineNode`s), covering all 7 explicit lifecycle
 * milestones, `completed_uncommitted`, `eos_terminated`, `capacity_exhausted`, branch roster,
 * retained-history / pruning metadata, and navigation capabilities.
 */
export function buildTwoTowerTimelineModel(session: TwoTowerSession): TwoTowerTimelineModel {
  const activeBranch = getActiveBranch(session);
  const activeRecord = getActiveRecord(session);
  const caps = getSessionCapabilities(session);
  const memoryBreakdown = measureTwoTowerSessionRetainedBytes(session);
  const latestCtx = getLatestContextExecution(session);

  const records = activeBranch.records;
  const flatNodes: TimelineNode[] = new Array(records.length);
  for (let i = 0; i < records.length; i++) {
    flatNodes[i] = buildTimelineNode(records[i], i, activeBranch.cursorIndex, records.length);
  }

  // Group records into top-level TimelineSegments
  interface MutableSegmentAccumulator {
    kind: TimelineSegmentKind;
    blockId: number | null;
    blockSize: number | null;
    stepsPerBlock: number | null;
    records: TwoTowerSessionRecord[];
    nodes: TimelineNode[];
  }

  const rawSegments: MutableSegmentAccumulator[] = [];
  for (let i = 0; i < records.length; i++) {
    const rec = records[i];
    const node = flatNodes[i];

    if (
      rec.kind === 'diffusion_block_init' ||
      rec.kind === 'diffusion_refinement' ||
      rec.kind === 'diffusion_confidence_commit' ||
      rec.kind === 'diffusion_block_commit'
    ) {
      const blockId = rec.blockConfig.blockId;
      const lastSeg = rawSegments.length > 0 ? rawSegments[rawSegments.length - 1] : null;
      if (lastSeg && lastSeg.kind === 'diffusion_block' && lastSeg.blockId === blockId) {
        lastSeg.records.push(rec);
        lastSeg.nodes.push(node);
      } else {
        rawSegments.push({
          kind: 'diffusion_block',
          blockId,
          blockSize: rec.blockConfig.blockSize,
          stepsPerBlock: rec.blockConfig.stepsPerBlock,
          records: [rec],
          nodes: [node],
        });
      }
    } else if (rec.kind === 'context_prefill') {
      rawSegments.push({
        kind: 'prefill',
        blockId: null,
        blockSize: null,
        stepsPerBlock: null,
        records: [rec],
        nodes: [node],
      });
    } else if (rec.kind === 'ar_step') {
      rawSegments.push({
        kind: 'ar_step',
        blockId: null,
        blockSize: null,
        stepsPerBlock: null,
        records: [rec],
        nodes: [node],
      });
    } else {
      rawSegments.push({
        kind: 'mock_ar_step',
        blockId: null,
        blockSize: null,
        stepsPerBlock: null,
        records: [rec],
        nodes: [node],
      });
    }
  }

  const segments: TimelineSegment[] = rawSegments.map((acc, segIdx) => {
    const firstRec = acc.records[0];
    const lastRec = acc.records[acc.records.length - 1];
    const lastNode = acc.nodes[acc.nodes.length - 1];
    const containsActiveCursor = acc.nodes.some((n) => n.isActiveCursor);
    const isHistorical = acc.nodes.every((n) => n.isHistorical);
    const range = firstRec.identity.absoluteTokenRange;

    let segmentId: string;
    let label: string;
    let completionState: TimelineSegment['completionState'];
    let isCompletedUncommitted = false;
    let isCommittedToContext = false;
    let nfe = 0;
    let earlyCompleted = false;
    let hasTrimmedHead = false;

    if (acc.kind === 'prefill') {
      segmentId = `prefill-${firstRec.identity.recordIndex}`;
      label = `Context Prefill [${range[0]}..${range[1]}]`;
      completionState = 'committed';
      isCommittedToContext = true;
    } else if (acc.kind === 'ar_step') {
      segmentId = `ar-step-${firstRec.identity.recordIndex}`;
      label = `AR Step [${range[0]}]`;
      completionState = 'committed';
      isCommittedToContext = true;
    } else if (acc.kind === 'mock_ar_step') {
      segmentId = `mock-ar-step-${firstRec.identity.recordIndex}`;
      label = `Mock-AR Step [${range[0]}]`;
      completionState = 'committed';
      isCommittedToContext = true;
      nfe = 1;
    } else {
      segmentId = `diffusion-block-${acc.blockId}`;
      label = `Diffusion Block #${acc.blockId} [${range[0]}..${range[1]}] (B=${acc.blockSize}, K=${acc.stepsPerBlock})`;
      hasTrimmedHead = firstRec.kind !== 'diffusion_block_init';

      if (lastRec.kind === 'diffusion_block_commit') {
        completionState = 'committed';
        isCommittedToContext = true;
        nfe = lastRec.nfe;
        earlyCompleted = lastRec.earlyCompleted;
      } else if (lastNode.isCompletedUncommitted) {
        completionState = 'completed_uncommitted';
        isCompletedUncommitted = true;
        nfe = lastRec.activeBlock?.nfe ?? 0;
        earlyCompleted = lastRec.activeBlock?.earlyCompleted ?? false;
      } else if (lastRec.kind === 'diffusion_block_init') {
        completionState = 'initialized';
        nfe = 0;
      } else {
        completionState = 'refining';
        nfe = lastRec.activeBlock?.nfe ?? 0;
      }
    }

    return Object.freeze({
      segmentId,
      segmentIndex: segIdx,
      kind: acc.kind,
      label,
      blockId: acc.blockId,
      blockSize: acc.blockSize,
      stepsPerBlock: acc.stepsPerBlock,
      absoluteTokenRange: range,
      containsActiveCursor,
      isHistorical,
      completionState,
      isCompletedUncommitted,
      isCommittedToContext,
      nfe,
      earlyCompleted,
      terminationMarker: lastNode.terminationMarker,
      hasTrimmedHead,
      nodes: Object.freeze(acc.nodes),
    });
  });

  const branches: readonly TwoTowerTimelineBranchEntry[] = Object.freeze(
    session.branches.map((b) => {
      const curRec = b.records[b.cursorIndex];
      const tailRec = b.records[b.records.length - 1];
      const isActive = b.branchId === session.activeBranchId;
      return Object.freeze({
        branchId: b.branchId,
        label: b.label,
        isActive,
        recordCount: b.records.length,
        cursorIndex: b.cursorIndex,
        isAtTail: b.cursorIndex === b.records.length - 1,
        parentBranchId: b.parentBranchId,
        forkPointRecordId: b.forkPointRecordId,
        forkPointIndex: b.forkPointIndex,
        activeLifecyclePhase: curRec.lifecyclePhase,
        tailLifecyclePhase: tailRec.lifecyclePhase,
        committedPrefixLen: curRec.committedPrefixLen,
        isDeletable: !isActive && session.branches.length > 1,
      });
    }),
  );

  const prunedRecordCount = records.length > 0 ? records[0].identity.recordIndex : 0;
  const retainedHistory: TwoTowerTimelineRetainedHistory = Object.freeze({
    retainedRecordCount: records.length,
    totalUniqueRecordCount: memoryBreakdown.uniqueRecordCount,
    maxRecords: session.bounds.maxRecords,
    branchCount: session.branches.length,
    maxBranches: session.bounds.maxBranches,
    prunedRecordCount,
    hasPrunedContextTraces: latestCtx === null,
    hasReleasedEagerTraces: detectReleasedEagerContextTraces(session),
    retainedBytes: memoryBreakdown.totalRetainedBytes,
    retainedBytesDisplay: formatBytesAdaptive(memoryBreakdown.totalRetainedBytes),
    memoryBreakdown,
  });

  const navigation: TwoTowerTimelineNavigationCapabilities = Object.freeze({
    canBack: caps.canNavigateBack,
    canForward: caps.canNavigateForward,
    canFork: caps.canForkBranch,
    canRestart: true as const,
    canDeleteActiveBranch: false as const,
    canDeleteAnyInactiveBranch: caps.canDeleteAnyBranch,
    canStepNext: caps.canStepNext,
    nextTransitionKind: caps.nextTransitionKind,
    willAutoForkOnMutatingStep: caps.willAutoForkOnMutatingStep,
  });

  const model: TwoTowerTimelineModel = Object.freeze({
    sessionId: session.sessionId,
    activeBranchId: activeBranch.branchId,
    activeBranchLabel: activeBranch.label,
    activeCursorIndex: activeBranch.cursorIndex,
    activeRecordId: activeRecord.identity.recordId,
    activeRecordIndex: activeRecord.identity.recordIndex,
    activeLifecyclePhase: activeRecord.lifecyclePhase,
    isInspectingHistoricalCursor: activeBranch.cursorIndex < records.length - 1,
    isTerminated: activeRecord.termination !== null,
    termination: activeRecord.termination,
    segments: Object.freeze(segments),
    flatNodes: Object.freeze(flatNodes),
    branches,
    retainedHistory,
    navigation,
  });

  assertPresentationSafeNoInfinity(model, 'twoTowerTimelineModel');
  return model;
}

// ============================================================================
// 2B. CONFIG SCOPE & NON-RETROACTIVITY MODEL
// ============================================================================

/**
 * Compare historical `record.configAtStep`, frozen `activeBlock.blockConfig` (when a diffusion
 * block is in flight), and future `session.pendingConfig`, classifying every configurable
 * parameter with its exact session-contract applicability scope so UI never conflates future
 * controls with historical inspection.
 */
export function buildTwoTowerConfigScopeModel(
  session: TwoTowerSession,
  draftRestartOverrides?: TwoTowerConfigScopeDraftRestartOverrides,
): TwoTowerConfigScopeModel {
  const branch = getActiveBranch(session);
  const record = getActiveRecord(session);
  const historicalConfig = record.configAtStep;
  const activeBlock = record.activeBlock;
  const activeBlockConfig = activeBlock ? activeBlock.blockConfig : null;
  const pendingConfig = session.pendingConfig;
  const isHistoricalCursor = branch.cursorIndex < branch.records.length - 1;
  const isTerminated = record.termination !== null;
  const hasActiveUncommittedBlock = activeBlock !== null;

  const pendingPromptTokens = draftRestartOverrides?.initialPromptTokens
    ? Object.freeze([...draftRestartOverrides.initialPromptTokens])
    : session.promptTokenIds;
  const pendingKvCapacity = draftRestartOverrides?.kvCapacity ?? session.kvCapacity;

  const modeScope: TwoTowerConfigApplicabilityScope = hasActiveUncommittedBlock
    ? 'next_block'
    : 'next_step';
  const eosScope: TwoTowerConfigApplicabilityScope =
    hasActiveUncommittedBlock && activeBlock.status !== 'completed_uncommitted'
      ? 'next_block'
      : 'next_step';

  const makeScopeBadge = (scope: TwoTowerConfigApplicabilityScope): string => {
    switch (scope) {
      case 'next_refinement':
        return 'Next refinement';
      case 'next_block':
        return 'Next block';
      case 'next_step':
        return 'Next step';
      case 'session_restart':
        return 'Session restart';
    }
  };

  const parameters: TwoTowerConfigParameterScope[] = [
    Object.freeze({
      key: 'confidenceThreshold' as const,
      label: 'Confidence Threshold (γ)',
      applicability: 'next_refinement' as const,
      historicalValue: historicalConfig.confidenceThreshold,
      activeBlockValue: null,
      pendingValue: pendingConfig.confidenceThreshold,
      differsFromHistorical: !Object.is(
        pendingConfig.confidenceThreshold,
        historicalConfig.confidenceThreshold,
      ),
      differsFromActiveBlock: false,
      scopeBadge: makeScopeBadge('next_refinement'),
      scopeNote:
        hasActiveUncommittedBlock && activeBlock.status !== 'completed_uncommitted'
          ? 'Applies on the next confidence-unmasking step of the active block; never retroactively alters prior refinements.'
          : 'Applies on the next masked-diffusion confidence-unmasking step.',
    }),
    Object.freeze({
      key: 'temperature' as const,
      label: 'Sampling Temperature',
      applicability: 'next_refinement' as const,
      historicalValue: historicalConfig.temperature,
      activeBlockValue: null,
      pendingValue: pendingConfig.temperature,
      differsFromHistorical: !Object.is(pendingConfig.temperature, historicalConfig.temperature),
      differsFromActiveBlock: false,
      scopeBadge: makeScopeBadge('next_refinement'),
      scopeNote:
        'Applies on the next confidence-unmasking step (0 = deterministic greedy argmax; > 0 = seeded Gumbel-max using unscaled confidence).',
    }),
    Object.freeze({
      key: 'samplingSeed' as const,
      label: 'Deterministic Sampling Seed',
      applicability: 'next_refinement' as const,
      historicalValue: historicalConfig.samplingSeed,
      activeBlockValue: null,
      pendingValue: pendingConfig.samplingSeed,
      differsFromHistorical: pendingConfig.samplingSeed !== historicalConfig.samplingSeed,
      differsFromActiveBlock: false,
      scopeBadge: makeScopeBadge('next_refinement'),
      scopeNote:
        'Applies on the next stochastic (temperature > 0) refinement step via keyed xoshiro128** derivation; strictly deterministic with zero unseeded PRNG.',
    }),
    Object.freeze({
      key: 'blockSize' as const,
      label: 'Diffusion Block Size (B)',
      applicability: 'next_block' as const,
      historicalValue: historicalConfig.blockSize,
      activeBlockValue: activeBlockConfig ? activeBlockConfig.blockSize : null,
      pendingValue: pendingConfig.blockSize,
      differsFromHistorical: pendingConfig.blockSize !== historicalConfig.blockSize,
      differsFromActiveBlock:
        activeBlockConfig !== null && pendingConfig.blockSize !== activeBlockConfig.blockSize,
      scopeBadge: makeScopeBadge('next_block'),
      scopeNote:
        activeBlockConfig !== null
          ? `Frozen at B=${activeBlockConfig.blockSize} for in-flight Block #${activeBlockConfig.blockId}; pending B=${pendingConfig.blockSize} (supported: 4 or 8) applies strictly to the next diffusion block.`
          : `Supported miniature block sizes: 4 or 8 tokens (independent of stepsPerBlock). Applies on next diffusion block initialization.`,
    }),
    Object.freeze({
      key: 'stepsPerBlock' as const,
      label: 'Refinement Steps Per Block (K)',
      applicability: 'next_block' as const,
      historicalValue: historicalConfig.stepsPerBlock,
      activeBlockValue: activeBlockConfig ? activeBlockConfig.stepsPerBlock : null,
      pendingValue: pendingConfig.stepsPerBlock,
      differsFromHistorical: pendingConfig.stepsPerBlock !== historicalConfig.stepsPerBlock,
      differsFromActiveBlock:
        activeBlockConfig !== null &&
        pendingConfig.stepsPerBlock !== activeBlockConfig.stepsPerBlock,
      scopeBadge: makeScopeBadge('next_block'),
      scopeNote:
        activeBlockConfig !== null
          ? `Frozen at K=${activeBlockConfig.stepsPerBlock} for in-flight Block #${activeBlockConfig.blockId}; pending K=${pendingConfig.stepsPerBlock} (independently configurable from blockSize) applies strictly to the next block.`
          : `Independently configurable from blockSize (K < B, K = B, or K > B). Applies on next diffusion block initialization.`,
    }),
    Object.freeze({
      key: 'mode' as const,
      label: 'Execution Mode',
      applicability: modeScope,
      historicalValue: historicalConfig.mode,
      activeBlockValue: activeBlockConfig !== null ? 'mask_diffusion' : null,
      pendingValue: pendingConfig.mode,
      differsFromHistorical: pendingConfig.mode !== historicalConfig.mode,
      differsFromActiveBlock:
        activeBlockConfig !== null && pendingConfig.mode !== 'mask_diffusion',
      scopeBadge: makeScopeBadge(modeScope),
      scopeNote:
        activeBlockConfig !== null
          ? `Active Block #${activeBlockConfig.blockId} must finish and commit in mask_diffusion before switching to "${pendingConfig.mode}".`
          : isTerminated
            ? `Branch is terminated (${record.termination.reason}); rewind/fork or restart session to run in "${pendingConfig.mode}".`
            : `Takes effect immediately on the next step (${pendingConfig.mode}).`,
    }),
    Object.freeze({
      key: 'eosTokenIds' as const,
      label: 'EOS Stop Token IDs',
      applicability: eosScope,
      historicalValue: historicalConfig.eosTokenIds,
      activeBlockValue: null,
      pendingValue: pendingConfig.eosTokenIds,
      differsFromHistorical: !arrayEqualsNumber(
        pendingConfig.eosTokenIds,
        historicalConfig.eosTokenIds,
      ),
      differsFromActiveBlock: false,
      scopeBadge: makeScopeBadge(eosScope),
      scopeNote:
        'Checked post-block commit in mask_diffusion (full-block EOS semantics) and after token commit in AR / Mock-AR.',
    }),
    Object.freeze({
      key: 'initialPromptTokens' as const,
      label: 'Initial Prompt Tokens',
      applicability: 'session_restart' as const,
      historicalValue: session.promptTokenIds,
      activeBlockValue: null,
      pendingValue: pendingPromptTokens,
      differsFromHistorical: !arrayEqualsNumber(pendingPromptTokens, session.promptTokenIds),
      differsFromActiveBlock: false,
      scopeBadge: makeScopeBadge('session_restart'),
      scopeNote:
        'Fixed at Context prefill; modifying the initial prompt requires restarting the session.',
    }),
    Object.freeze({
      key: 'kvCapacity' as const,
      label: 'KV Cache Capacity',
      applicability: 'session_restart' as const,
      historicalValue: record.kvCapacity,
      activeBlockValue: null,
      pendingValue: pendingKvCapacity,
      differsFromHistorical: pendingKvCapacity !== record.kvCapacity,
      differsFromActiveBlock: false,
      scopeBadge: makeScopeBadge('session_restart'),
      scopeNote:
        'Allocated once at session creation; changing kvCapacity requires restarting the session.',
    }),
  ];

  const diffKeysFromHistorical: TwoTowerConfigParameterKey[] = [];
  const diffKeysFromActiveBlock: TwoTowerConfigParameterKey[] = [];
  const byKey = {} as Record<TwoTowerConfigParameterKey, TwoTowerConfigParameterScope>;

  for (let i = 0; i < parameters.length; i++) {
    const p = parameters[i];
    byKey[p.key] = p;
    if (p.differsFromHistorical) diffKeysFromHistorical.push(p.key);
    if (p.differsFromActiveBlock) diffKeysFromActiveBlock.push(p.key);
  }

  const hasPendingDiffFromHistorical = diffKeysFromHistorical.length > 0;
  const hasPendingDiffFromActiveBlock = diffKeysFromActiveBlock.length > 0;

  let badgeText: string;
  if (isHistoricalCursor && hasPendingDiffFromHistorical) {
    badgeText = `Historical Record #${record.identity.recordIndex} · Pending edits (${diffKeysFromHistorical.join(', ')}) apply only to future/forked steps`;
  } else if (isHistoricalCursor) {
    badgeText = `Historical Record #${record.identity.recordIndex} · Pending config matches historical step`;
  } else if (activeBlockConfig !== null && hasPendingDiffFromActiveBlock) {
    badgeText = `Active Block #${activeBlockConfig.blockId} frozen (B=${activeBlockConfig.blockSize}, K=${activeBlockConfig.stepsPerBlock}) · Pending edits (${diffKeysFromActiveBlock.join(', ')}) apply next block`;
  } else if (hasPendingDiffFromHistorical) {
    badgeText = `Pending edits (${diffKeysFromHistorical.join(', ')}) queued for future steps · Current record unchanged`;
  } else {
    badgeText = `Live Record #${record.identity.recordIndex} · Pending config matches active step`;
  }

  const model: TwoTowerConfigScopeModel = Object.freeze({
    activeRecordIndex: record.identity.recordIndex,
    activeRecordKind: record.kind,
    lifecyclePhase: record.lifecyclePhase,
    isHistoricalCursor,
    isTerminated,
    historicalConfig,
    activeBlockConfig,
    pendingConfig,
    sessionPromptTokens: session.promptTokenIds,
    sessionKvCapacity: session.kvCapacity,
    hasPendingDiffFromHistorical,
    hasPendingDiffFromActiveBlock,
    diffKeysFromHistorical: Object.freeze(diffKeysFromHistorical),
    diffKeysFromActiveBlock: Object.freeze(diffKeysFromActiveBlock),
    badgeText,
    epistemicNote:
      'Parameter edits update session.pendingConfig non-retroactively: historical records preserve their frozen configAtStep, and active diffusion blocks preserve their frozen blockSize and stepsPerBlock until committed.',
    parameters: Object.freeze(parameters),
    parametersByKey: Object.freeze(byKey),
  });

  assertPresentationSafeNoInfinity(model, 'twoTowerConfigScopeModel');
  return model;
}

// ============================================================================
// 3. ADVANCED SAMPLER INSPECTOR VIEW-MODEL
// ============================================================================

function findMatchingPriorConfidenceCommitRecord(
  records: readonly TwoTowerSessionRecord[],
  fromIndexExclusive: number,
  blockId: number,
  stepIndex?: number,
): {
  readonly record: TwoTowerSessionDiffusionConfidenceCommitRecord;
  readonly cursorIndex: number;
} | null {
  for (let idx = fromIndexExclusive - 1; idx >= 0; idx--) {
    const candidate = records[idx];
    if (
      candidate.kind === 'diffusion_confidence_commit' &&
      candidate.blockConfig.blockId === blockId &&
      (stepIndex === undefined || candidate.samplerPresentation.stepIndex === stepIndex)
    ) {
      return { record: candidate, cursorIndex: idx };
    }
  }
  return null;
}

/**
 * Build the strictly `Infinity`-free Advanced Sampler Inspector view-model for `session`.
 *
 * Inspects the active record's `SamplerStepPresentation` (or `activeBlock.latestSamplerPresentation`
 * / prior block commit when on a refinement or block-commit record), explicitly flagging whether
 * data belongs to `'current_record'`, `'prior_commit_record'`, or `'none'`.
 *
 * Guarantees zero `Infinity` / `-Infinity` / `NaN` values via `assertPresentationSafeNoInfinity`.
 */
export function buildAdvancedSamplerInspectorModel(
  session: TwoTowerSession,
): TwoTowerAdvancedSamplerInspectorModel {
  const branch = getActiveBranch(session);
  const record = getActiveRecord(session);
  const historicalConfig = record.configAtStep;

  let source: SamplerInspectorSourceKind = 'none';
  let presentation: SamplerStepPresentation | null = null;
  let variateProvenance: TwoTowerVariateProvenance | null = null;
  let sourceRecordId: string | null = null;
  let sourceRecordIndex: number | null = null;
  let sourceCursorIndex: number | null = null;
  let sourceBlockId: number | null = null;
  let sourceRefinementIndex: number | null = null;
  let sourceNote: string;

  if (record.kind === 'diffusion_confidence_commit') {
    source = 'current_record';
    presentation = record.samplerPresentation;
    variateProvenance = record.variateProvenance;
    sourceRecordId = record.identity.recordId;
    sourceRecordIndex = record.identity.recordIndex;
    sourceCursorIndex = branch.cursorIndex;
    sourceBlockId = record.blockConfig.blockId;
    sourceRefinementIndex = record.refinementIndex;
    sourceNote = `Active confidence-commit record #${record.identity.recordIndex} (Block #${record.blockConfig.blockId}, refinement r=${record.refinementIndex}).`;
  } else if (record.activeBlock?.latestSamplerPresentation != null) {
    source = 'prior_commit_record';
    presentation = record.activeBlock.latestSamplerPresentation;
    sourceBlockId = record.activeBlock.blockId;
    sourceRefinementIndex = presentation.stepIndex;
    const priorMatch = findMatchingPriorConfidenceCommitRecord(
      branch.records,
      branch.cursorIndex,
      record.activeBlock.blockId,
      presentation.stepIndex,
    );
    if (priorMatch) {
      variateProvenance = priorMatch.record.variateProvenance;
      sourceRecordId = priorMatch.record.identity.recordId;
      sourceRecordIndex = priorMatch.record.identity.recordIndex;
      sourceCursorIndex = priorMatch.cursorIndex;
      sourceNote = `No confidence commit on current ${record.kind} record #${record.identity.recordIndex}; showing prior confidence commit from record #${priorMatch.record.identity.recordIndex} (Block #${sourceBlockId}, r=${presentation.stepIndex}).`;
    } else {
      sourceNote = `No confidence commit on current ${record.kind} record #${record.identity.recordIndex}; showing activeBlock.latestSamplerPresentation (Block #${sourceBlockId}, r=${presentation.stepIndex}).`;
    }
  } else if (record.kind === 'diffusion_block_commit') {
    const priorMatch = findMatchingPriorConfidenceCommitRecord(
      branch.records,
      branch.cursorIndex,
      record.blockConfig.blockId,
    );
    if (priorMatch) {
      source = 'prior_commit_record';
      presentation = priorMatch.record.samplerPresentation;
      variateProvenance = priorMatch.record.variateProvenance;
      sourceRecordId = priorMatch.record.identity.recordId;
      sourceRecordIndex = priorMatch.record.identity.recordIndex;
      sourceCursorIndex = priorMatch.cursorIndex;
      sourceBlockId = priorMatch.record.blockConfig.blockId;
      sourceRefinementIndex = priorMatch.record.refinementIndex;
      sourceNote = `Block #${record.blockConfig.blockId} committed to Context at record #${record.identity.recordIndex}; showing its final prior confidence commit from record #${priorMatch.record.identity.recordIndex} (r=${priorMatch.record.refinementIndex}).`;
    } else {
      sourceNote = `Block #${record.blockConfig.blockId} committed to Context at record #${record.identity.recordIndex}; prior confidence-commit record is no longer in retained history.`;
    }
  } else {
    sourceNote =
      record.activeBlock !== null
        ? `No sampler confidence commit has occurred yet for active Block #${record.activeBlock.blockId} (current record: ${record.kind}).`
        : `No sampler confidence commit on current ${record.kind} record #${record.identity.recordIndex}.`;
  }

  const effectiveThreshold = presentation
    ? presentation.confidenceThreshold
    : historicalConfig.confidenceThreshold;
  const matchedPreset = matchConfidencePreset(effectiveThreshold);

  const threshold: SamplerInspectorThresholdInfo = Object.freeze({
    value: effectiveThreshold,
    activeCursorThreshold: historicalConfig.confidenceThreshold,
    matchedPresetId: matchedPreset.id,
    matchedPresetLabel: matchedPreset.label,
    presets: TWOTOWER_CONFIDENCE_PRESETS,
  });

  if (presentation === null) {
    const emptyModel: TwoTowerAdvancedSamplerInspectorModel = Object.freeze({
      source,
      hasSamplerCommit: false,
      isCurrentRecord: false,
      isPriorCommitRecord: false,
      sourceRecordId: null,
      sourceRecordIndex: null,
      sourceCursorIndex: null,
      sourceBlockId: record.activeBlock ? record.activeBlock.blockId : null,
      sourceRefinementIndex: null,
      sourceNote,
      samplerPresentation: null,
      threshold,
      temperature: historicalConfig.temperature,
      variateMode: null,
      variateDigest: null,
      samplingSeed: null,
      derivationKey: null,
      executedTModel: null,
      commitmentDecomposition: null,
      slots: Object.freeze([]),
      calibrationDisclaimer: TWOTOWER_SAMPLER_CALIBRATION_DISCLAIMER,
    });
    assertPresentationSafeNoInfinity(emptyModel, 'advancedSamplerInspectorModel');
    return emptyModel;
  }

  assertPresentationSafeNoInfinity(presentation, 'samplerStepPresentation');

  let priorLockedCount = 0;
  let thresholdQualifiedCount = 0;
  let progressFloorExtraCount = 0;
  let budgetFloorOnlyExtraCount = 0;
  let finalStepExtraCount = 0;
  let remaskedCount = 0;

  const maskTokenId = presentation.maskTokenId;
  const stepVariateDigest = variateProvenance?.variateDigest ?? null;

  const slots: SamplerInspectorSlotRow[] = presentation.slots.map((slot) => {
    const status = mapSlotDecisionToStatus(slot.decision);
    switch (slot.decision) {
      case 'locked_prior':
        priorLockedCount++;
        break;
      case 'committed_threshold':
        thresholdQualifiedCount++;
        break;
      case 'committed_progress_floor':
        progressFloorExtraCount++;
        break;
      case 'committed_budget_floor':
        budgetFloorOnlyExtraCount++;
        break;
      case 'committed_final_step':
        finalStepExtraCount++;
        break;
      case 'remasked':
        remaskedCount++;
        break;
    }

    const confidence: SamplerInspectorSlotConfidence =
      slot.confidenceDisplay.kind === 'locked_prior'
        ? Object.freeze({
            kind: 'locked_prior' as const,
            confidenceDisplay: 'Locked (prior step)' as const,
            numericConfidence: null,
          })
        : Object.freeze({
            kind: 'unscaled_probability' as const,
            confidenceDisplay: slot.confidenceDisplay.displayText,
            numericConfidence: slot.confidenceDisplay.unscaledConfidence,
          });

    const constrainedTop1 = slot.constrainedMdlm.topK[0];

    return Object.freeze({
      blockPosition: slot.slotIndex,
      absolutePosition: slot.absolutePosition,
      inputToken: Object.freeze({
        tokenId: slot.tokenIdBefore,
        label: slot.tokenLabelBefore,
        isMask: slot.tokenIdBefore === maskTokenId,
      }),
      outputToken: Object.freeze({
        tokenId: slot.tokenIdAfter,
        label: slot.tokenLabelAfter,
        isMask: slot.tokenIdAfter === maskTokenId,
      }),
      status,
      rawDecision: slot.decision,
      rawLmHeadTop: Object.freeze({
        top1TokenId: slot.rawLmHead.top1TokenId,
        top1TokenLabel: slot.rawLmHead.top1TokenLabel,
        top1Logit: slot.rawLmHead.top1Logit,
        top1Prob: slot.rawLmHead.top1Prob,
        maskTokenProb: slot.rawLmHead.maskTokenProb,
        topK: slot.rawLmHead.topK,
      }),
      constrainedMdlmTop: Object.freeze({
        maskTokenExcluded: true as const,
        maskTokenProb: 0 as const,
        isCarryOverDelta: slot.constrainedMdlm.isCarryOverDelta,
        top1TokenId: constrainedTop1.tokenId,
        top1TokenLabel: constrainedTop1.tokenLabel,
        top1LogProb: constrainedTop1.logProb,
        top1Prob: constrainedTop1.prob,
        topK: slot.constrainedMdlm.topK,
      }),
      candidateMode: slot.selection.samplingMode,
      confidence,
      unscaledConfidence: confidence.numericConfidence,
      confidenceDisplay: confidence.confidenceDisplay,
      passesStrictThreshold: slot.confidenceDisplay.exceedsThreshold,
      variateDigest: slot.selection.samplingMode === 'gumbel_max' ? stepVariateDigest : null,
    });
  });

  const budgetFloorExtraCount =
    progressFloorExtraCount + budgetFloorOnlyExtraCount + finalStepExtraCount;
  const totalCommittedThisStep = thresholdQualifiedCount + budgetFloorExtraCount;

  const executedTModel: SamplerInspectorExecutedTimestepModel = Object.freeze({
    tModel: presentation.tModel,
    blockSize: presentation.blockSize,
    stepIndex: presentation.stepIndex,
    stepsPerBlock: presentation.stepsPerBlock,
    nMaskedBefore: presentation.nMaskedBefore,
    nMaskedAfter: presentation.nMaskedAfter,
    remainingSteps: presentation.remainingSteps,
    minCommitFloor: presentation.minCommitFloor,
    commitReason: presentation.commitReason,
    earlyCompleted: presentation.earlyCompleted,
    isBlockComplete: presentation.isBlockComplete,
  });

  const commitmentDecomposition: SamplerInspectorCommitmentDecomposition = Object.freeze({
    thresholdQualifiedCount,
    budgetFloorExtraCount,
    progressFloorExtraCount,
    budgetFloorOnlyExtraCount,
    finalStepExtraCount,
    totalCommittedThisStep,
    remaskedCount,
    priorLockedCount,
  });

  const model: TwoTowerAdvancedSamplerInspectorModel = Object.freeze({
    source,
    hasSamplerCommit: true,
    isCurrentRecord: source === 'current_record',
    isPriorCommitRecord: source === 'prior_commit_record',
    sourceRecordId,
    sourceRecordIndex,
    sourceCursorIndex,
    sourceBlockId,
    sourceRefinementIndex,
    sourceNote,
    samplerPresentation: presentation,
    threshold,
    temperature: presentation.temperature,
    variateMode: variateProvenance?.mode ?? null,
    variateDigest: stepVariateDigest,
    samplingSeed: variateProvenance?.samplingSeed ?? null,
    derivationKey: variateProvenance?.derivationKey ?? null,
    executedTModel,
    commitmentDecomposition,
    slots: Object.freeze(slots),
    calibrationDisclaimer: TWOTOWER_SAMPLER_CALIBRATION_DISCLAIMER,
  });

  assertPresentationSafeNoInfinity(model, 'advancedSamplerInspectorModel');
  return model;
}
