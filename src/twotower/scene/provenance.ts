/**
 * TwoTower Spatial Provenance, Tower-Aware Identity & Shared Scene Contracts
 * (`src/twotower/scene/provenance.ts`)
 *
 * Pure headless TypeScript (zero DOM or WebGL2 imports) unifying all Stage TT3 spatial views:
 *   - Tower-aware selection identity (`TwoTowerSelectionIdentity`) preserving tower role, model ID,
 *     parameter fingerprint, execution ID, session/branch/record/block/refinement IDs, 1-based bay
 *     and 0-based reference layer index, mixer kind, head/group/expert, tensor role, absolute token
 *     position, trace origin, and historical `configAtStep`.
 *   - Historical session resolution (`resolveTwoTowerSessionViewState`) strictly reading the active
 *     historical record (`getActiveRecord`), never substituting `session.pendingConfig` or standalone
 *     Nano fixtures, and explicitly surfacing pruned historical Context executions without inventing
 *     synthetic replacement traces.
 *   - Fail-safe external-mutation integrity check (`validateContextSnapshot` guard).
 *   - Unified WebGL2 pass-compatible `TwoTowerSpatialScene` payload consumed directly by Nano's
 *     existing `SolidsPass`, `LinesPass`, `RibbonsPass`, `FlatFlowRibbonsPass`, and `SheetsPass`.
 */

import { GQA_SIM_CONFIG } from '../../math/gqaSim';
import { MAMBA_SIM_CONFIG } from '../../math/mambaSim';
import type { Kind } from '../../model/spec';
import type { AABB } from '../../render/camera';
import type { PickBox, PickHit } from '../../render/pick';
import type { FocusedAccumulationResult } from '../../scene/accumulation';
import type { Rg32fValueAtlas } from '../../scene/atlasPacker';
import type { CutawayPartInfo } from '../../scene/cutaways';
import type { LayerFormulaReadout, StepperStepDef } from '../../scene/sheetInspector';
import type { DimensionCaliper, KvDepthTickAnchor, LandmarkAnchor } from '../../scene/spine';
import type { ActiveSheetCellGrid, CellPopoverSpec } from '../../ui/overlay';
import {
  validateContextSnapshot,
  type TwoTowerContextConditioningState,
  type TwoTowerContextSnapshot,
  type TwoTowerTraceOrigin,
} from '../contracts';
import type {
  TwoTowerContextExecutionResult,
  TwoTowerContextLayerTrace,
  TwoTowerContextSegmentTraceHandle,
  TwoTowerDenoiserLayerTrace,
  TwoTowerDenoiserPassResult,
} from '../execution';
import type { TwoTowerModelParameters } from '../parameters';
import type { SamplerStepPresentation } from '../sampler';
import {
  getActiveBranch,
  getActiveRecord,
  getLatestContextExecution,
  type TwoTowerActiveDiffusionBlock,
  type TwoTowerLatestContextExecution,
  type TwoTowerRecordIdentity,
  type TwoTowerSession,
  type TwoTowerSessionConfig,
  type TwoTowerSessionLifecyclePhase,
  type TwoTowerSessionRecord,
  type TwoTowerSessionTermination,
} from '../session';
import { TWOTOWER_MINI_CONFIG, TWOTOWER_SPEC, type TwoTowerExecutionMode } from '../spec';

/**
 * Authoritative miniature TwoTower model dimensional inventory (derived strictly from
 * `GQA_SIM_CONFIG`, `MAMBA_SIM_CONFIG`, and `TWOTOWER_MINI_CONFIG`) — clearly separated from
 * selected per-view RG32F atlas slices so slice counts are never mislabeled as model head counts.
 */
export const TWOTOWER_MINI_MODEL_INVENTORY = Object.freeze({
  d: TWOTOWER_MINI_CONFIG.d,
  vocabSize: TWOTOWER_MINI_CONFIG.vocabSize,
  gqa: Object.freeze({
    queryHeads: GQA_SIM_CONFIG.nQ, // 32 miniature GQA query heads
    kvHeads: GQA_SIM_CONFIG.nKv, // 2 KV heads
    queriesPerKvGroup: (GQA_SIM_CONFIG.nQ / GQA_SIM_CONFIG.nKv) as 16,
    headDim: GQA_SIM_CONFIG.dHead, // head width 4
    qConcatDim: GQA_SIM_CONFIG.qConcatDim, // 128
    kvConcatDim: GQA_SIM_CONFIG.kvConcatDim, // 8
    atlasSelectedScoreHeadCount: 2 as const, // 1 selected Q head per KV group on Scores/Softmax plates
  }),
  mamba: Object.freeze({
    heads: MAMBA_SIM_CONFIG.nHeads, // 16 miniature Mamba heads
    groups: MAMBA_SIM_CONFIG.nGroups, // 8 Mamba groups
    headsPerGroup: MAMBA_SIM_CONFIG.headsPerGroup, // 2
    dHead: MAMBA_SIM_CONFIG.dHead, // 4
    dInner: MAMBA_SIM_CONFIG.dInner, // 64
    dState: MAMBA_SIM_CONFIG.dState, // 4
    ssmStateMatrixShape: Object.freeze([MAMBA_SIM_CONFIG.dHead, MAMBA_SIM_CONFIG.dState] as const), // 4×4
    convKernel: MAMBA_SIM_CONFIG.convKernel, // 4
    convDim: MAMBA_SIM_CONFIG.convDim, // 128
    inProjDim: MAMBA_SIM_CONFIG.inProjDim, // 208
    atlasSelectedDecompHeadCount: 1 as const, // 1 selected head for native 4×4 analytical decomposition plates
  }),
});

export type TwoTowerTowerRole = 'context' | 'denoiser' | 'shared';

export type TwoTowerSpatialViewMode =
  | 'macro'
  | 'paired_gqa'
  | 'paired_mamba'
  | 'timestep'
  | 'moe'
  | 'embedding'
  | 'output';

export type TwoTowerPairedFocus = 'paired' | 'context_solo' | 'denoiser_solo';

export type TwoTowerLinkerChannel = 'all' | 'gqa_kv' | 'mamba_conv' | 'mamba_ssm' | 'timestep';

export type TwoTowerMixerRole = Kind | 'embed' | 'output' | 'time_cond' | 'linker';

/**
 * Full 10-axis selection identity preserved across ray-picking, formula inspection, sidebars,
 * and cross-view transitions. Never drops tower, execution, block, refinement, or absolute position.
 */
export interface TwoTowerSelectionIdentity {
  readonly towerRole: TwoTowerTowerRole;
  readonly modelId: string;
  readonly parameterFingerprint: string;
  readonly executionId: string;
  readonly sessionId: string;
  readonly branchId: string;
  readonly recordId: string;
  readonly recordIndex: number;
  readonly blockId: number | null;
  readonly refinementIndex: number | null;
  /** One-based bay number (`1..52`) or `0` for embedding / `53` for output / `-1` for shared rail. */
  readonly bayId: number;
  /** Zero-based NVIDIA reference layer index (`0..51`), or `null` for bookends / shared rail. */
  readonly layerIndex: number | null;
  readonly mixerKind: TwoTowerMixerRole;
  readonly headIndex?: number;
  readonly kvHeadIndex?: 0 | 1;
  readonly groupIndex?: number;
  readonly expertId?: number;
  readonly tensorRole: string;
  readonly absoluteTokenPosition: number | null;
  readonly absoluteTokenRange: readonly [number, number];
  readonly traceOrigin: TwoTowerTraceOrigin;
  readonly historicalConfigAtStep: TwoTowerSessionConfig;
  readonly provenanceNote: string;
}

export interface TwoTowerContextTraceResolution {
  readonly status: 'available' | 'pruned_unavailable';
  readonly recordId: string | null;
  readonly recordIndex: number | null;
  readonly execution: TwoTowerContextExecutionResult | null;
  readonly reason: string;
}

export interface TwoTowerContextBayTraceLookup {
  readonly status: 'available' | 'pruned_unavailable';
  readonly bayId: number;
  readonly layerIndex: number;
  readonly segment: TwoTowerContextSegmentTraceHandle | null;
  readonly trace: TwoTowerContextLayerTrace | null;
  readonly traceOrigin: 'context_computed' | 'context_reconstructed' | 'persistent_cache';
  readonly sourceRecordId: string | null;
  readonly reason: string;
}

export interface TwoTowerDenoiserPassLookup {
  readonly status: 'active_on_record' | 'historical_prior_record' | 'none_yet';
  readonly pass: TwoTowerDenoiserPassResult | null;
  readonly conditioning: TwoTowerContextConditioningState | null;
  readonly sourceRecordId: string | null;
  readonly sourceRecordIndex: number | null;
  readonly sourceBlockId: number | null;
  readonly sourceRefinementIndex: number | null;
  /** True strictly when showing a prior Denoiser pass (`status === 'historical_prior_record'`). */
  readonly isHistoricalInspection: boolean;
  /** Actual Context prefix length (`conditioning.committedPrefixLen` / `pass.committedPrefixLen`) that the Denoiser pass read. */
  readonly sourceContextPrefixLen: number | null;
  /** Authoritative Context cache digest (`conditioning.sourceStateDigest` / `pass.sourceStateDigest`) that the Denoiser pass read. */
  readonly sourceContextCacheDigest: string | null;
  /** Authoritative frozen historical `configAtStep` of the record where the Denoiser pass executed. */
  readonly sourceRecordConfig: TwoTowerSessionConfig | null;
  /**
   * True ONLY when the Denoiser pass executed on the current record AND its conditioning cache
   * digest and prefix length match the current record's `contextSnapshot` in the same active
   * block/mode. Strictly `false` after block commit (`P -> P+B`), mode switch, or new block init.
   */
  readonly matchesCurrentContextBoundary: boolean;
  /**
   * Explicit non-null warning whenever `isHistoricalInspection` is true (especially when
   * `!matchesCurrentContextBoundary`) so prior KV/Mamba conditioning is never relabeled as
   * belonging to the current Context boundary.
   */
  readonly historicalProvenanceWarning: string | null;
  readonly traceOrigin: 'denoiser_scratch';
  readonly note: string;
}

export type TwoTowerBlockCompletionState =
  | 'no_active_block'
  | 'refining'
  | 'completed_uncommitted'
  | 'committed_to_context';

export type TwoTowerTimestepConditioningMode =
  | 'active_diffusion'
  | 'inactive_no_pass_yet'
  | 'historical_prior_pass'
  | 'bypassed_ar'
  | 'bypassed_mock_ar';

export interface TwoTowerExecutedPassTimestep {
  /** Model timestep scalar actually fed into sinusoidal embedding φ(t). */
  readonly tModel: number;
  /** Actual masked token count entering the executed Denoiser pass (`pass.tokenIds`). */
  readonly nMaskedInput: number;
  /** Block size B of the executed Denoiser pass (`pass.blockSize`). */
  readonly blockSize: number;
  /** Canonical empirical mask fraction `nMaskedInput / blockSize` (never `r / K`). */
  readonly empiricalMaskFraction: number;
  /** True when `tModel` differs from `empiricalMaskFraction` via a diagnostic `tOverride`. */
  readonly isDiagnosticOverride: boolean;
  /** Explicit diagnostic override scalar when `isDiagnosticOverride` is true, else `null`. */
  readonly overrideValue: number | null;
  readonly passExecutionId: string;
  readonly sourceRecordIndex: number;
  readonly sourceBlockId: number | null;
  readonly sourceRefinementIndex: number | null;
}

export interface TwoTowerCurrentBlockMaskStatus {
  /** Unresolved `[MASK]` token count on the current record (post-commit on `diffusion_confidence_commit`). */
  readonly currentUnresolvedCount: number;
  readonly blockSize: number;
  /** Current unresolved mask fraction `currentUnresolvedCount / blockSize`. */
  readonly currentMaskFraction: number;
  readonly isBlockComplete: boolean;
  readonly isCommittedToContext: boolean;
}

export interface TwoTowerNextEligibleStepTimestep {
  /** Canonical timestep `nextNMasked / blockSize` for the next eligible refinement pass. */
  readonly nextTModel: number;
  readonly nextNMasked: number;
  readonly blockSize: number;
  readonly remainingSteps: number;
}

/**
 * Frozen canonical timestep status (`t = nMasked / B`, NEVER `r / K`) strictly separating:
 *   1. `executedPass`: the executed Denoiser pass's actual input mask fraction (plus any diagnostic override).
 *   2. `currentBlockMask`: the current record's block mask state (post-commit if on a commit record).
 *   3. `nextEligibleStep`: populated ONLY when an incomplete active block can run another refinement (`nMasked > 0 && remainingSteps > 0`).
 *   4. `blockCompletionState`: explicit lifecycle completion state (confirming no further denoiser pass is needed when complete).
 *   5. `conditioningMode`: cleanly distinguishes active diffusion, inactive/no-pass-yet, historical prior pass, AR bypass, and Mock-AR bypass.
 */
export interface TwoTowerTimestepStatus {
  readonly executedPass: TwoTowerExecutedPassTimestep | null;
  readonly currentBlockMask: TwoTowerCurrentBlockMaskStatus | null;
  readonly nextEligibleStep: TwoTowerNextEligibleStepTimestep | null;
  readonly blockCompletionState: TwoTowerBlockCompletionState;
  readonly blockCompletionNote: string;
  readonly conditioningMode: TwoTowerTimestepConditioningMode;
  readonly summaryLabel: string;
}

export interface TwoTowerSnapshotIntegrity {
  readonly valid: boolean;
  readonly error: string | null;
}

/**
 * Resolved read-only view context derived strictly from a historical or active `TwoTowerSession`
 * cursor. Never mutates session buffers and never substitutes `session.pendingConfig`.
 */
export interface TwoTowerSessionViewState {
  readonly model: TwoTowerModelParameters;
  readonly sessionId: string;
  readonly branchId: string;
  readonly branchLabel: string;
  readonly cursorIndex: number;
  readonly branchRecordCount: number;
  readonly isHistoricalCursor: boolean;
  readonly record: TwoTowerSessionRecord;
  readonly identity: TwoTowerRecordIdentity;
  /** Strictly the historical `record.configAtStep` — NEVER `session.pendingConfig`. */
  readonly historicalConfig: TwoTowerSessionConfig;
  readonly modeAtRecord: TwoTowerExecutionMode;
  readonly lifecyclePhase: TwoTowerSessionLifecyclePhase;
  readonly committedPrefixLen: number;
  readonly kvCapacity: number;
  readonly contextSnapshot: TwoTowerContextSnapshot;
  readonly snapshotIntegrity: TwoTowerSnapshotIntegrity;
  readonly activeBlock: TwoTowerActiveDiffusionBlock | null;
  readonly termination: TwoTowerSessionTermination | null;
  readonly latestContextTrace: TwoTowerContextTraceResolution;
  readonly denoiserLookup: TwoTowerDenoiserPassLookup;
  /** Canonical stage-separated timestep status (`t = nMasked / B`, never `r / K`). */
  readonly timestepStatus: TwoTowerTimestepStatus;
  readonly latestSamplerPresentation: SamplerStepPresentation | null;
  /** Resolve an actual Context bay trace (`1..52`) from the latest non-pruned Context execution. */
  resolveContextBayTrace(bayId: number, preferredAbsPos?: number): TwoTowerContextBayTraceLookup;
  /** Resolve an actual Denoiser bay trace (`1..52`) from the active or latest Denoiser pass. */
  resolveDenoiserBayTrace(bayId: number): TwoTowerDenoiserLayerTrace | null;
}

export interface TwoTowerProvenanceBadge {
  readonly label: string;
  readonly tone: 'context' | 'denoiser' | 'shared' | 'warn' | 'muted' | 'gold';
  readonly origin?: TwoTowerTraceOrigin;
}

/**
 * Unified 3D spatial geometry payload for ALL Stage TT3 TwoTower views.
 * Directly uploadable to Nano's `SolidsPass`, `LinesPass`, `RibbonsPass`, `FlatFlowRibbonsPass`,
 * and `SheetsPass` without duplicating WebGL2 shaders or render passes.
 */
export interface TwoTowerSpatialScene {
  readonly viewMode: TwoTowerSpatialViewMode;
  readonly title: string;
  readonly subtitle: string;
  readonly bayId: number;
  readonly layerIndex: number | null;
  readonly activeTowerFocus: TwoTowerPairedFocus;
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
  /** Deterministic cache key for RG32F GPU texture upload deduplication. */
  readonly atlasKey?: string;
  readonly fitBounds: AABB;
  readonly opFocusBounds?: Readonly<Record<number, AABB>>;
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
  readonly focusedAccumulation?: FocusedAccumulationResult | null;
  readonly activeSelection: TwoTowerSelectionIdentity | null;
  readonly stepperSteps?: readonly StepperStepDef[];
  readonly provenanceBadges: readonly TwoTowerProvenanceBadge[];
  readonly epistemicNote: string;
  /** Explicit non-empty warning when a historical Context trace was pruned or snapshot tampered. */
  readonly warningBanner: string | null;
  /** Optional view-specific structured diagnostic metadata for side panels and unit tests. */
  readonly diagnostics?: Readonly<Record<string, unknown>>;
}

/**
 * Helper to look up a 1-based bay's mixer kind (`'M' | 'E' | '*'`) from the authoritative 52-layer
 * Nemotron-Labs-TwoTower pattern.
 */
export function getTwoTowerBayKind(bayId: number): Kind {
  const idx = bayId - 1;
  const pattern = TWOTOWER_SPEC.directlyInspectedMetadata.hybridOverridePattern;
  if (!Number.isInteger(bayId) || idx < 0 || idx >= pattern.length) {
    throw new RangeError(`TwoTower bayId must be an integer in [1..${pattern.length}], got ${bayId}`);
  }
  return pattern[idx] as Kind;
}

/**
 * Resolve the nearest valid bay ID for a desired mixer kind (`'M'`, `'E'`, or `'*'`).
 */
export function resolveNearestBayOfKind(requestedBayId: number, kind: Kind): number {
  const pattern = TWOTOWER_SPEC.directlyInspectedMetadata.hybridOverridePattern;
  const clamped = Math.max(1, Math.min(pattern.length, Math.trunc(requestedBayId)));
  if (pattern[clamped - 1] === kind) return clamped;
  let best = kind === '*' ? 6 : kind === 'E' ? 2 : 1;
  let bestDist = Infinity;
  for (let i = 0; i < pattern.length; i++) {
    if (pattern[i] === kind) {
      const id = i + 1;
      const d = Math.abs(id - clamped);
      if (d < bestDist) {
        bestDist = d;
        best = id;
      }
    }
  }
  return best;
}

function countMaskedTokenIds(tokenIds: ArrayLike<number>, maskTokenId: number): number {
  let count = 0;
  for (let i = 0; i < tokenIds.length; i++) {
    if (tokenIds[i] === maskTokenId) count++;
  }
  return count;
}

/**
 * Resolve the canonical, stage-separated `TwoTowerTimestepStatus` (`t = nMasked / B`, NEVER `r / K`)
 * for the active or historical cursor record without mutating any session buffers.
 */
export function resolveTwoTowerTimestepStatus(params: {
  readonly record: TwoTowerSessionRecord;
  readonly denoiserLookup: TwoTowerDenoiserPassLookup;
  readonly maskTokenId: number;
}): TwoTowerTimestepStatus {
  const { record, denoiserLookup, maskTokenId } = params;
  const modeAtRecord = record.configAtStep.mode;
  const activeBlock = record.activeBlock;

  // 1. Current record's block mask state & completion state
  let currentBlockMask: TwoTowerCurrentBlockMaskStatus | null = null;
  let blockCompletionState: TwoTowerBlockCompletionState = 'no_active_block';
  let blockCompletionNote = 'No active diffusion block on this record.';
  let nextEligibleStep: TwoTowerNextEligibleStepTimestep | null = null;

  if (activeBlock !== null) {
    const B = activeBlock.blockConfig.blockSize;
    const unresolved = activeBlock.nMasked;
    const isComplete =
      unresolved === 0 || activeBlock.status === 'completed_uncommitted';
    currentBlockMask = Object.freeze({
      currentUnresolvedCount: unresolved,
      blockSize: B,
      currentMaskFraction: B > 0 ? unresolved / B : 0,
      isBlockComplete: isComplete,
      isCommittedToContext: false,
    });

    if (isComplete) {
      blockCompletionState = 'completed_uncommitted';
      blockCompletionNote =
        `Block b${activeBlock.blockId} is fully unmasked (0/${B} [MASK] remaining) — no further denoiser evaluation is required; ready to commit block to Context Tower.`;
    } else {
      blockCompletionState = 'refining';
      const remainingSteps = Math.max(
        0,
        activeBlock.blockConfig.stepsPerBlock - activeBlock.completedRefinements,
      );
      if (
        record.termination === null &&
        (activeBlock.status === 'initialized' || activeBlock.status === 'refining') &&
        unresolved > 0 &&
        remainingSteps > 0
      ) {
        nextEligibleStep = Object.freeze({
          nextTModel: unresolved / B,
          nextNMasked: unresolved,
          blockSize: B,
          remainingSteps,
        });
        blockCompletionNote =
          `Block b${activeBlock.blockId} has ${unresolved}/${B} unresolved [MASK] slot${unresolved === 1 ? '' : 's'}; next eligible pass uses t = ${unresolved}/${B} = ${(unresolved / B).toFixed(3)} (${remainingSteps} step${remainingSteps === 1 ? '' : 's'} remaining).`;
      } else if (activeBlock.status === 'awaiting_confidence_commit') {
        blockCompletionNote =
          `Block b${activeBlock.blockId} refinement r=${activeBlock.pendingRefinementIndex ?? 0} denoised (${unresolved}/${B} pre-commit masks); commit confidence to reveal post-commit mask state.`;
      } else {
        blockCompletionNote =
          `Block b${activeBlock.blockId} has ${unresolved}/${B} masked slots remaining with 0 refinement steps left in budget.`;
      }
    }
  } else if (record.kind === 'diffusion_block_commit') {
    const B = record.blockConfig.blockSize;
    currentBlockMask = Object.freeze({
      currentUnresolvedCount: 0,
      blockSize: B,
      currentMaskFraction: 0,
      isBlockComplete: true,
      isCommittedToContext: true,
    });
    blockCompletionState = 'committed_to_context';
    blockCompletionNote =
      `Block b${record.blockConfig.blockId} (${B}/${B} tokens) committed to Context Tower (P=${record.committedPrefixLen}) — no further denoiser evaluation is required.`;
  }

  // 2. Conditioning mode & executed pass timestep (strictly derived from executed pass input tokens!)
  let conditioningMode: TwoTowerTimestepConditioningMode;
  let executedPass: TwoTowerExecutedPassTimestep | null = null;

  if (modeAtRecord === 'ar') {
    conditioningMode = 'bypassed_ar';
  } else if (modeAtRecord === 'mock_ar') {
    conditioningMode = 'bypassed_mock_ar';
  } else {
    const pass = denoiserLookup.pass;
    if (
      pass !== null &&
      pass.mode === 'mask_diffusion' &&
      pass.tModel !== null &&
      denoiserLookup.sourceRecordIndex !== null
    ) {
      const nMaskedInput = countMaskedTokenIds(pass.tokenIds, maskTokenId);
      const blockSize = pass.blockSize;
      const empiricalMaskFraction = blockSize > 0 ? nMaskedInput / blockSize : 0;
      const isDiagnosticOverride = Math.abs(pass.tModel - empiricalMaskFraction) > 1e-6;
      executedPass = Object.freeze({
        tModel: pass.tModel,
        nMaskedInput,
        blockSize,
        empiricalMaskFraction,
        isDiagnosticOverride,
        overrideValue: isDiagnosticOverride ? pass.tModel : null,
        passExecutionId: pass.passExecutionId,
        sourceRecordIndex: denoiserLookup.sourceRecordIndex,
        sourceBlockId: denoiserLookup.sourceBlockId,
        sourceRefinementIndex: denoiserLookup.sourceRefinementIndex,
      });
      conditioningMode =
        denoiserLookup.status === 'active_on_record'
          ? 'active_diffusion'
          : 'historical_prior_pass';
    } else {
      conditioningMode = 'inactive_no_pass_yet';
    }
  }

  // 3. Stage-consistent summary label (NEVER pairs an executed pass's tModel with a different stage's nMasked)
  let summaryLabel: string;
  if (conditioningMode === 'bypassed_ar') {
    summaryLabel = 'BYPASSED (t = null) · AR mode';
  } else if (conditioningMode === 'bypassed_mock_ar') {
    summaryLabel = 'BYPASSED (t = null) · Mock-AR mode';
  } else if (conditioningMode === 'inactive_no_pass_yet') {
    if (nextEligibleStep !== null) {
      summaryLabel =
        `INACTIVE (no pass yet) · Next t = ${nextEligibleStep.nextTModel.toFixed(2)} ` +
        `(${nextEligibleStep.nextNMasked}/${nextEligibleStep.blockSize} masked)`;
    } else if (blockCompletionState === 'completed_uncommitted') {
      summaryLabel = 'INACTIVE (no pass yet) · Block already complete (0 masked, no pass needed)';
    } else if (blockCompletionState === 'committed_to_context') {
      summaryLabel = 'INACTIVE (no pass yet) · Block committed to Context (no further pass needed)';
    } else {
      summaryLabel = 'INACTIVE · No diffusion pass yet (ready)';
    }
  } else if (conditioningMode === 'historical_prior_pass' && executedPass !== null) {
    const passDesc = executedPass.isDiagnosticOverride
      ? `t_override = ${executedPass.overrideValue!.toFixed(2)} (empirical ${executedPass.nMaskedInput}/${executedPass.blockSize} = ${executedPass.empiricalMaskFraction.toFixed(2)})`
      : `t = ${executedPass.tModel.toFixed(2)} (${executedPass.nMaskedInput}/${executedPass.blockSize} input masked)`;
    const locTag = `Rec #${executedPass.sourceRecordIndex} b${executedPass.sourceBlockId ?? '-'} r${executedPass.sourceRefinementIndex ?? '-'}`;
    if (nextEligibleStep !== null) {
      summaryLabel =
        `HISTORICAL (${locTag}: ${passDesc}) · Next block t = ${nextEligibleStep.nextTModel.toFixed(2)} ` +
        `(${nextEligibleStep.nextNMasked}/${nextEligibleStep.blockSize} masked)`;
    } else {
      summaryLabel = `HISTORICAL (${locTag}: ${passDesc}) · Block committed (no further pass needed)`;
    }
  } else if (executedPass !== null) {
    // active_diffusion
    const passDesc = executedPass.isDiagnosticOverride
      ? `t_override = ${executedPass.overrideValue!.toFixed(2)} (empirical ${executedPass.nMaskedInput}/${executedPass.blockSize} = ${executedPass.empiricalMaskFraction.toFixed(2)})`
      : `t = ${executedPass.tModel.toFixed(2)} (${executedPass.nMaskedInput}/${executedPass.blockSize} input masked)`;
    if (blockCompletionState === 'completed_uncommitted') {
      summaryLabel = `ACTIVE · Pass ${passDesc} · Block complete (0/${executedPass.blockSize} left, no further pass needed)`;
    } else if (nextEligibleStep !== null) {
      summaryLabel =
        `ACTIVE · Pass ${passDesc} · Block now ${nextEligibleStep.nextNMasked}/${nextEligibleStep.blockSize} masked ` +
        `(Next t = ${nextEligibleStep.nextTModel.toFixed(2)})`;
    } else {
      summaryLabel = `ACTIVE · ${passDesc}`;
    }
  } else {
    summaryLabel = 'INACTIVE · No diffusion pass yet';
  }

  return Object.freeze({
    executedPass,
    currentBlockMask,
    nextEligibleStep,
    blockCompletionState,
    blockCompletionNote,
    conditioningMode,
    summaryLabel,
  });
}

/**
 * Safely inspect a `TwoTowerSession` at its current historical or live cursor, enforcing:
 *   1. Historical `record.configAtStep` is authoritative (never `session.pendingConfig`).
 *   2. Fail-safe validation of `record.contextSnapshot` against `session.model.contextModel`.
 *   3. Explicit handling of missing historical Context execution after pruning (never invents traces).
 *   4. Clear distinction between active-record Denoiser passes and prior-record Denoiser passes,
 *      including explicit Context cache boundary provenance & mismatch warnings.
 *   5. Stage-separated canonical timestep semantics (`t = nMasked / B`, NEVER `r / K`).
 */
export function resolveTwoTowerSessionViewState(session: TwoTowerSession): TwoTowerSessionViewState {
  const branch = getActiveBranch(session);
  const record = getActiveRecord(session);
  const isHistoricalCursor = branch.cursorIndex < branch.records.length - 1;

  // Fail-safe snapshot integrity verification (detects unauthorized external buffer mutation)
  let snapshotIntegrity: TwoTowerSnapshotIntegrity;
  try {
    validateContextSnapshot(record.contextSnapshot, session.model.contextModel);
    snapshotIntegrity = Object.freeze({ valid: true, error: null });
  } catch (err) {
    snapshotIntegrity = Object.freeze({
      valid: false,
      error: err instanceof Error ? err.message : String(err),
    });
  }

  // Latest Context execution at or before cursor (explicitly handles pruned history)
  const latestCtx: TwoTowerLatestContextExecution | null = getLatestContextExecution(session);
  const latestContextTrace: TwoTowerContextTraceResolution = latestCtx
    ? Object.freeze({
        status: 'available' as const,
        recordId: latestCtx.record.identity.recordId,
        recordIndex: latestCtx.branchCursorIndex,
        execution: latestCtx.execution,
        reason:
          latestCtx.branchCursorIndex === branch.cursorIndex
            ? `Live Context ${latestCtx.execution.operation} execution at record #${latestCtx.branchCursorIndex} (${latestCtx.execution.segments.length} segment${latestCtx.execution.segments.length === 1 ? '' : 's'}).`
            : `Latest committed Context ${latestCtx.execution.operation} execution from record #${latestCtx.branchCursorIndex} (${latestCtx.execution.segments.length} segment${latestCtx.execution.segments.length === 1 ? '' : 's'}).`,
      })
    : Object.freeze({
        status: 'pruned_unavailable' as const,
        recordId: null,
        recordIndex: null,
        execution: null,
        reason:
          'Historical Context execution trace is unavailable after history pruning; persistent Context cache snapshot remains authoritative (no synthetic trace substituted).',
      });

  // Resolve Denoiser pass: active on current record vs latest prior record at or before cursor,
  // preserving explicit Context cache boundary provenance (`sourceContextPrefixLen`, `sourceContextCacheDigest`,
  // `sourceRecordConfig`, `matchesCurrentContextBoundary`, `historicalProvenanceWarning`).
  const currentCtxDigest = record.contextSnapshot.ctxState.stateDigest;
  const currentCtxPrefixLen = record.contextSnapshot.ctxState.processedTokens;

  let denoiserLookup: TwoTowerDenoiserPassLookup;
  if (record.denoiserPass !== null) {
    const pass = record.denoiserPass;
    const cond = record.conditioning;
    const sourceContextPrefixLen = cond?.committedPrefixLen ?? pass.committedPrefixLen;
    const sourceContextCacheDigest =
      cond?.sourceStateDigest ?? pass.sourceStateDigest ?? record.identity.sourceCacheDigest;
    const matchesCurrentContextBoundary =
      sourceContextCacheDigest === currentCtxDigest &&
      sourceContextPrefixLen === currentCtxPrefixLen &&
      pass.mode === record.configAtStep.mode &&
      (record.activeBlock === null || record.activeBlock.blockId === pass.blockId);

    denoiserLookup = Object.freeze({
      status: 'active_on_record' as const,
      pass,
      conditioning: cond,
      sourceRecordId: record.identity.recordId,
      sourceRecordIndex: branch.cursorIndex,
      sourceBlockId: record.identity.blockId,
      sourceRefinementIndex: record.identity.refinementIndex,
      isHistoricalInspection: false,
      sourceContextPrefixLen,
      sourceContextCacheDigest,
      sourceRecordConfig: record.configAtStep,
      matchesCurrentContextBoundary,
      historicalProvenanceWarning: null,
      traceOrigin: 'denoiser_scratch' as const,
      note: `Active Denoiser scratch pass (${pass.mode}, ${pass.passExecutionId}, conditioned on Context P=${sourceContextPrefixLen}, cache ${sourceContextCacheDigest.slice(0, 8)}).`,
    });
  } else {
    let priorPass: TwoTowerDenoiserPassResult | null = null;
    let priorCond: TwoTowerContextConditioningState | null = null;
    let priorRecordId: string | null = null;
    let priorRecordIndex: number | null = null;
    let priorBlockId: number | null = null;
    let priorRefinementIndex: number | null = null;
    let priorRecordConfig: TwoTowerSessionConfig | null = null;
    let priorRecordSourceDigest: string | null = null;

    for (let idx = branch.cursorIndex - 1; idx >= 0; idx--) {
      const candidate = branch.records[idx];
      if (candidate.denoiserPass !== null) {
        priorPass = candidate.denoiserPass;
        priorCond = candidate.conditioning;
        priorRecordId = candidate.identity.recordId;
        priorRecordIndex = idx;
        priorBlockId = candidate.identity.blockId;
        priorRefinementIndex = candidate.identity.refinementIndex;
        priorRecordConfig = candidate.configAtStep;
        priorRecordSourceDigest = candidate.identity.sourceCacheDigest;
        break;
      }
    }

    if (priorPass !== null) {
      const sourceContextPrefixLen = priorCond?.committedPrefixLen ?? priorPass.committedPrefixLen;
      const sourceContextCacheDigest =
        priorCond?.sourceStateDigest ?? priorPass.sourceStateDigest ?? priorRecordSourceDigest;
      // Strictly FALSE for historical prior passes after block commitment, mode switch, or new block init
      const currentActiveBlock: TwoTowerActiveDiffusionBlock | null = record.activeBlock;
      const matchesCurrentContextBoundary =
        sourceContextCacheDigest !== null &&
        sourceContextCacheDigest === currentCtxDigest &&
        sourceContextPrefixLen === currentCtxPrefixLen &&
        priorPass.mode === record.configAtStep.mode &&
        record.kind !== 'diffusion_block_commit' &&
        record.kind !== 'diffusion_block_init' &&
        currentActiveBlock !== null &&
        currentActiveBlock.blockId === priorPass.blockId;

      const locTag =
        priorBlockId !== null
          ? `Block b${priorBlockId}, r=${priorRefinementIndex ?? 0}`
          : `mode=${priorPass.mode}`;
      const shortSrcDigest = sourceContextCacheDigest ? sourceContextCacheDigest.slice(0, 8) : 'n/a';
      const shortCurDigest = currentCtxDigest.slice(0, 8);
      const blockSwitchNote =
        record.activeBlock !== null && record.activeBlock.blockId !== priorBlockId
          ? ` (active Block b${record.activeBlock.blockId} has not been denoised yet)`
          : '';
      const boundaryMismatchDetail = !matchesCurrentContextBoundary
        ? ` Current record #${branch.cursorIndex} (${record.kind}, mode=${record.configAtStep.mode}) is at Context P=${currentCtxPrefixLen} (cache ${shortCurDigest})${blockSwitchNote} — prior Denoiser KV/Mamba conditioning (P=${sourceContextPrefixLen}) does NOT belong to the current Context boundary.`
        : '';
      const historicalProvenanceWarning =
        `Historical Denoiser pass inspection: showing prior ${priorPass.mode} pass from Record #${priorRecordIndex} ` +
        `(${locTag}, ${priorPass.passExecutionId}) conditioned on Context P=${sourceContextPrefixLen} ` +
        `(cache ${shortSrcDigest}).${boundaryMismatchDetail}`;

      denoiserLookup = Object.freeze({
        status: 'historical_prior_record' as const,
        pass: priorPass,
        conditioning: priorCond,
        sourceRecordId: priorRecordId,
        sourceRecordIndex: priorRecordIndex,
        sourceBlockId: priorBlockId,
        sourceRefinementIndex: priorRefinementIndex,
        isHistoricalInspection: true,
        sourceContextPrefixLen,
        sourceContextCacheDigest,
        sourceRecordConfig: priorRecordConfig,
        matchesCurrentContextBoundary,
        historicalProvenanceWarning,
        traceOrigin: 'denoiser_scratch' as const,
        note: `No Denoiser pass executed on record #${branch.cursorIndex} (${record.kind}); showing prior Denoiser pass from record #${priorRecordIndex} (${locTag}, ${priorPass.passExecutionId}, conditioned on Context P=${sourceContextPrefixLen}, cache ${shortSrcDigest}).`,
      });
    } else {
      denoiserLookup = Object.freeze({
        status: 'none_yet' as const,
        pass: null,
        conditioning: null,
        sourceRecordId: null,
        sourceRecordIndex: null,
        sourceBlockId: null,
        sourceRefinementIndex: null,
        isHistoricalInspection: false,
        sourceContextPrefixLen: null,
        sourceContextCacheDigest: null,
        sourceRecordConfig: null,
        matchesCurrentContextBoundary: false,
        historicalProvenanceWarning: null,
        traceOrigin: 'denoiser_scratch' as const,
        note: 'No Denoiser forward pass has executed yet on this branch (Context-only state).',
      });
    }
  }

  // Canonical stage-separated timestep status (`t = nMasked / B`, NEVER `r / K`)
  const timestepStatus = resolveTwoTowerTimestepStatus({
    record,
    denoiserLookup,
    maskTokenId: session.model.maskTokenId,
  });

  // Latest sampler presentation
  const latestSamplerPresentation: SamplerStepPresentation | null =
    record.kind === 'diffusion_confidence_commit'
      ? record.samplerPresentation
      : record.activeBlock?.latestSamplerPresentation ?? null;

  const resolveContextBayTrace = (
    bayId: number,
    preferredAbsPos?: number,
  ): TwoTowerContextBayTraceLookup => {
    const layerIndex = bayId - 1;
    if (latestContextTrace.status !== 'available' || !latestContextTrace.execution) {
      return Object.freeze({
        status: 'pruned_unavailable' as const,
        bayId,
        layerIndex,
        segment: null,
        trace: null,
        traceOrigin: 'persistent_cache' as const,
        sourceRecordId: null,
        reason: latestContextTrace.reason,
      });
    }

    const exec = latestContextTrace.execution;
    let seg: TwoTowerContextSegmentTraceHandle;
    if (
      preferredAbsPos !== undefined &&
      preferredAbsPos >= exec.committedTokenRange[0] &&
      preferredAbsPos <= exec.committedTokenRange[1]
    ) {
      seg = exec.getSegmentForAbsolutePosition(preferredAbsPos);
    } else {
      // Default to the final segment (or the prefix segment if inspecting multi-token prefill context)
      seg = exec.segments[exec.segments.length - 1];
    }

    const trace = seg.getLayerTrace(bayId);
    return Object.freeze({
      status: 'available' as const,
      bayId,
      layerIndex,
      segment: seg,
      trace,
      traceOrigin: trace.traceOrigin,
      sourceRecordId: latestContextTrace.recordId,
      reason: `Context ${exec.operation} segment ${seg.segmentIndex + 1}/${seg.segmentCount} (${seg.segmentRole}, pos [${seg.tokenRange[0]}..${seg.tokenRange[1]}], origin=${trace.traceOrigin}).`,
    });
  };

  const resolveDenoiserBayTrace = (bayId: number): TwoTowerDenoiserLayerTrace | null => {
    if (!denoiserLookup.pass) return null;
    return denoiserLookup.pass.getLayerTrace(bayId);
  };

  return Object.freeze({
    model: session.model,
    sessionId: session.sessionId,
    branchId: branch.branchId,
    branchLabel: branch.label,
    cursorIndex: branch.cursorIndex,
    branchRecordCount: branch.records.length,
    isHistoricalCursor,
    record,
    identity: record.identity,
    historicalConfig: record.configAtStep,
    modeAtRecord: record.configAtStep.mode,
    lifecyclePhase: record.lifecyclePhase,
    committedPrefixLen: record.committedPrefixLen,
    kvCapacity: record.kvCapacity,
    contextSnapshot: record.contextSnapshot,
    snapshotIntegrity,
    activeBlock: record.activeBlock,
    termination: record.termination,
    latestContextTrace,
    denoiserLookup,
    timestepStatus,
    latestSamplerPresentation,
    resolveContextBayTrace,
    resolveDenoiserBayTrace,
  });
}

/**
 * Construct a complete `TwoTowerSelectionIdentity` from resolved session state and target coordinates.
 */
export function buildTwoTowerSelectionIdentity(
  view: TwoTowerSessionViewState,
  params: {
    readonly towerRole: TwoTowerTowerRole;
    readonly bayId: number;
    readonly mixerKind: TwoTowerMixerRole;
    readonly tensorRole: string;
    readonly traceOrigin: TwoTowerTraceOrigin;
    readonly executionId: string;
    readonly absoluteTokenPosition: number | null;
    readonly absoluteTokenRange?: readonly [number, number];
    readonly headIndex?: number;
    readonly kvHeadIndex?: 0 | 1;
    readonly groupIndex?: number;
    readonly expertId?: number;
    readonly provenanceNote: string;
  },
): TwoTowerSelectionIdentity {
  const { model, identity, historicalConfig } = view;
  const modelId =
    params.towerRole === 'context'
      ? model.contextModel.modelId
      : params.towerRole === 'denoiser'
        ? model.denoiserModel.modelId
        : model.modelId;
  const parameterFingerprint =
    params.towerRole === 'context'
      ? model.contextModel.parameterFingerprint
      : params.towerRole === 'denoiser'
        ? model.denoiserModel.parameterFingerprint
        : model.twoTowerFingerprint;
  const layerIndex = params.bayId >= 1 && params.bayId <= 52 ? params.bayId - 1 : null;

  return Object.freeze({
    towerRole: params.towerRole,
    modelId,
    parameterFingerprint,
    executionId: params.executionId,
    sessionId: identity.sessionId,
    branchId: identity.branchId,
    recordId: identity.recordId,
    recordIndex: identity.recordIndex,
    blockId: identity.blockId,
    refinementIndex: identity.refinementIndex,
    bayId: params.bayId,
    layerIndex,
    mixerKind: params.mixerKind,
    ...(params.headIndex !== undefined ? { headIndex: params.headIndex } : {}),
    ...(params.kvHeadIndex !== undefined ? { kvHeadIndex: params.kvHeadIndex } : {}),
    ...(params.groupIndex !== undefined ? { groupIndex: params.groupIndex } : {}),
    ...(params.expertId !== undefined ? { expertId: params.expertId } : {}),
    tensorRole: params.tensorRole,
    absoluteTokenPosition: params.absoluteTokenPosition,
    absoluteTokenRange: params.absoluteTokenRange ?? identity.absoluteTokenRange,
    traceOrigin: params.traceOrigin,
    historicalConfigAtStep: historicalConfig,
    provenanceNote: params.provenanceNote,
  });
}

/**
 * Resolve tower-aware selection metadata directly from any ray-cast `PickHit`.
 */
export function resolvePickHitTowerIdentity(
  view: TwoTowerSessionViewState,
  hit: PickHit,
): TwoTowerSelectionIdentity {
  const towerRole: TwoTowerTowerRole = hit.towerRole ?? 'shared';
  const bayId = hit.bayId;
  const mixerKind: TwoTowerMixerRole =
    bayId >= 1 && bayId <= 52
      ? getTwoTowerBayKind(bayId)
      : bayId === 0
        ? 'embed'
        : bayId === 53
          ? 'output'
          : hit.semanticTag === 'timestep'
            ? 'time_cond'
            : 'linker';
  const traceOrigin: TwoTowerTraceOrigin =
    hit.traceOrigin === 'context_computed' ||
    hit.traceOrigin === 'context_reconstructed' ||
    hit.traceOrigin === 'persistent_cache' ||
    hit.traceOrigin === 'denoiser_scratch'
      ? hit.traceOrigin
      : towerRole === 'denoiser'
        ? 'denoiser_scratch'
        : 'persistent_cache';
  const absPos =
    hit.tokenIndex !== undefined
      ? (hit.absoluteTokenOffset ?? 0) + hit.tokenIndex
      : hit.hitRow !== undefined && hit.absoluteTokenOffset !== undefined
        ? hit.absoluteTokenOffset + hit.hitRow
        : null;
  const executionId =
    hit.executionId ??
    (towerRole === 'denoiser'
      ? view.denoiserLookup.pass?.passExecutionId ?? view.identity.recordId
      : view.latestContextTrace.execution?.segments[
          view.latestContextTrace.execution.segments.length - 1
        ]?.stepExecutionId ?? view.identity.sourceCacheDigest);

  return buildTwoTowerSelectionIdentity(view, {
    towerRole,
    bayId,
    mixerKind,
    tensorRole: hit.tensorKey ?? hit.partId ?? hit.semanticTag ?? 'bay',
    traceOrigin,
    executionId,
    absoluteTokenPosition: absPos,
    headIndex: hit.headIndex,
    kvHeadIndex: hit.kvHeadIndex,
    groupIndex: hit.groupIndex,
    provenanceNote: `${towerRole.toUpperCase()} · ${bayId >= 1 && bayId <= 52 ? `Bay ${String(bayId).padStart(2, '0')} (layer ${bayId - 1})` : hit.partId ?? 'structure'} · origin=${traceOrigin}`,
  });
}
