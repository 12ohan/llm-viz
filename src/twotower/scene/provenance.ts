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
import { TWOTOWER_SPEC, type TwoTowerExecutionMode } from '../spec';

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
  readonly traceOrigin: 'denoiser_scratch';
  readonly note: string;
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

/**
 * Safely inspect a `TwoTowerSession` at its current historical or live cursor, enforcing:
 *   1. Historical `record.configAtStep` is authoritative (never `session.pendingConfig`).
 *   2. Fail-safe validation of `record.contextSnapshot` against `session.model.contextModel`.
 *   3. Explicit handling of missing historical Context execution after pruning (never invents traces).
 *   4. Clear distinction between active-record Denoiser passes and prior-record Denoiser passes.
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

  // Resolve Denoiser pass: active on current record vs latest prior record at or before cursor
  let denoiserLookup: TwoTowerDenoiserPassLookup;
  if (record.denoiserPass !== null) {
    denoiserLookup = Object.freeze({
      status: 'active_on_record' as const,
      pass: record.denoiserPass,
      conditioning: record.conditioning,
      sourceRecordId: record.identity.recordId,
      sourceRecordIndex: branch.cursorIndex,
      sourceBlockId: record.identity.blockId,
      sourceRefinementIndex: record.identity.refinementIndex,
      traceOrigin: 'denoiser_scratch' as const,
      note: `Active Denoiser scratch pass (${record.denoiserPass.mode}, ${record.denoiserPass.passExecutionId}).`,
    });
  } else {
    let priorPass: TwoTowerDenoiserPassResult | null = null;
    let priorCond: TwoTowerContextConditioningState | null = null;
    let priorRecordId: string | null = null;
    let priorRecordIndex: number | null = null;
    let priorBlockId: number | null = null;
    let priorRefinementIndex: number | null = null;

    for (let idx = branch.cursorIndex - 1; idx >= 0; idx--) {
      const candidate = branch.records[idx];
      if (candidate.denoiserPass !== null) {
        priorPass = candidate.denoiserPass;
        priorCond = candidate.conditioning;
        priorRecordId = candidate.identity.recordId;
        priorRecordIndex = idx;
        priorBlockId = candidate.identity.blockId;
        priorRefinementIndex = candidate.identity.refinementIndex;
        break;
      }
    }

    if (priorPass !== null) {
      denoiserLookup = Object.freeze({
        status: 'historical_prior_record' as const,
        pass: priorPass,
        conditioning: priorCond,
        sourceRecordId: priorRecordId,
        sourceRecordIndex: priorRecordIndex,
        sourceBlockId: priorBlockId,
        sourceRefinementIndex: priorRefinementIndex,
        traceOrigin: 'denoiser_scratch' as const,
        note: `No Denoiser pass executed on record #${branch.cursorIndex} (${record.kind}); showing most recent prior Denoiser pass from record #${priorRecordIndex} (${priorPass.passExecutionId}).`,
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
        traceOrigin: 'denoiser_scratch' as const,
        note: 'No Denoiser forward pass has executed yet on this branch (Context-only state).',
      });
    }
  }

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
