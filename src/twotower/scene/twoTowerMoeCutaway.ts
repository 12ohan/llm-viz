/**
 * TwoTower Granular MoE (128+1) Cutaway with FP64 6th-vs-7th Selection Margin &
 * Strict Nano v1.0 Partial-Denominator Preservation (`src/twotower/scene/twoTowerMoeCutaway.ts`)
 *
 * Pure headless TypeScript (zero DOM, WebGL2, or `Math.random()`).
 * Directly reuses Nano's verified `buildMoECutaway` over the ACTUAL Context or Denoiser tower's
 * `MoeSimResult` (never a standalone Nano fixture!) while adding:
 *   1. Exact Float64 6th-vs-7th biased-score selection margin (`computeMoeSelectionMargin`).
 *   2. Strict preservation of Nano v1.0's partial-denominator non-exposure contract across
 *      `activeFormula`, `cellPopover`, `activeCellGrids`, `landmarks`, `partCatalog`, and
 *      `diagnostics.sidebarWeights` during Step 2 (`stepperStep === 2`, `termProgress < 0.999`).
 *   3. Full tower-aware `PickBox` provenance (`towerRole`, `executionId`, `traceOrigin`,
 *      `absoluteTokenOffset`, actual MoE `bayId` across all 23 MoE layers).
 *   4. Explicit post-RMSNorm AdaLN modulation + outer residual gate status (`active` in Denoiser
 *      diffusion vs `bypassed` in Context/Mock-AR), `0 B` stateless cache verification, and clean
 *      `pruned_unavailable` handling when historical Context execution traces have been pruned.
 */

import {
  formatExpertId,
  MOE_SIM_CONFIG,
  MOE_STEPPER_STEPS,
  moeStepperCellForState,
  type MoeCellRef,
  type MoeSimResult,
  type RouterPrecisionPolicy,
} from '../../math/moeSim';
import type { GqaViewAngle } from '../../model/state';
import type { PickBox } from '../../render/pick';
import { SHEET_REVEAL_UNCOMPUTED } from '../../render/sheets';
import type { CutawayPartInfo } from '../../scene/cutaways';
import {
  buildMoECutaway,
  fitSpatialMoeCamera,
  MOE_FIT_BOUNDS,
  MOE_SPATIAL_FIT_BOUNDS,
  parseMoeExpertPartId,
} from '../../scene/moeCutaway';
import type { LayerFormulaReadout } from '../../scene/sheetInspector';
import type { LandmarkAnchor } from '../../scene/spine';
import type { ActiveSheetCellGrid } from '../../ui/overlay';
import type { TwoTowerTraceOrigin } from '../contracts';
import type { TwoTowerSession } from '../session';
import {
  buildTwoTowerSelectionIdentity,
  resolveNearestBayOfKind,
  resolveTwoTowerSessionViewState,
  type TwoTowerProvenanceBadge,
  type TwoTowerSpatialScene,
} from './provenance';

export { fitSpatialMoeCamera, MOE_FIT_BOUNDS, MOE_SPATIAL_FIT_BOUNDS, parseMoeExpertPartId };

export interface MoeTop8MarginEntry {
  readonly rank: number; // 1..8
  readonly expertId: number;
  readonly expertTag: string;
  readonly selected: boolean; // rank 1..6 true, rank 7..8 false
  readonly cutoffBoundaryAfter: boolean; // true only on rank 6
  readonly choiceScore: number;
  readonly unbiasedScore: number;
  readonly bias: number;
  readonly logit: number;
}

export interface MoeSelectionMarginInfo {
  readonly tokenIndex: number;
  readonly absoluteTokenPosition: number;
  readonly rank6ExpertId: number;
  readonly rank6ChoiceScore: number;
  readonly rank6UnbiasedScore: number;
  readonly rank6Bias: number;
  readonly rank7ExpertId: number;
  readonly rank7ChoiceScore: number;
  readonly rank7UnbiasedScore: number;
  readonly rank7Bias: number;
  /** Exact `rank6ChoiceScore - rank7ChoiceScore` (>= 0), computed in FP64 when `scores_for_choice_f64` exists. */
  readonly choiceScoreMargin: number;
  readonly unbiasedScoreDelta: number;
  readonly biasDelta: number;
  readonly routerPrecisionPolicy: RouterPrecisionPolicy;
  readonly isTightMargin: boolean;
  readonly top8Summary: readonly MoeTop8MarginEntry[];
}

export interface TwoTowerMoeSidebarWeightEntry {
  readonly rankSlot: number;
  readonly expertId: number;
  readonly expertTag: string;
  /** Masked `'—'` (future/pre-Step-2), `'…'` (active accumulating slot), or `w.toFixed(2)` when revealed. */
  readonly displayText: string;
  /** Revealed Float32 weight ONLY when unmasked/finalized; strictly `null` while masked/accumulating. */
  readonly weight: number | null;
  readonly state: 'uncomputed' | 'accumulating' | 'revealed';
}

export interface TwoTowerMoeDiagnostics {
  readonly bayId: number;
  readonly layerIndex: number;
  readonly towerRole: 'context' | 'denoiser';
  readonly contextTraceStatus: 'available' | 'pruned_unavailable' | 'not_applicable';
  readonly isHistoricalDenoiserInspection: boolean;
  readonly matchesCurrentContextBoundary: boolean;
  readonly denoiserSourceRecordIndex: number | null;
  readonly denoiserSourceBlockId: number | null;
  readonly denoiserSourceRefinementIndex: number | null;
  readonly denoiserSourceContextPrefixLen: number | null;
  readonly denoiserSourceContextCacheDigest: string | null;
  readonly historicalProvenanceWarning: string | null;
  readonly routerPrecisionPolicy: RouterPrecisionPolicy;
  readonly selectionMargin: MoeSelectionMarginInfo | null;
  readonly modulationOrder: 'norm_then_modulate';
  readonly modulationStatus: 'active' | 'bypassed';
  readonly residualGateApplied: boolean;
  readonly cacheBytes: 0;
  readonly stateless: true;
  readonly crossTowerCacheTransfer: false;
  readonly focusToken: number | null;
  readonly absoluteTokenPosition: number | null;
  readonly focusExpert: number | null;
  readonly partialDenominatorActive: boolean;
  readonly runningDenominator: number | null;
  readonly currentTermIndex: number | null;
  readonly sidebarWeights: readonly TwoTowerMoeSidebarWeightEntry[];
  readonly sidebarWeightTexts: readonly string[];
}

export interface BuildTwoTowerMoeOptions {
  /** Requested bay ID (snaps to nearest MoE `'E'` bay; defaults to Bay `02`). */
  readonly bayId?: number;
  /** Tower to inspect (`'denoiser'` when a Denoiser pass exists, else `'context'`). */
  readonly towerRole?: 'context' | 'denoiser';
  readonly focusToken?: number;
  readonly focusExpert?: number;
  readonly selectedPart?: string | null;
  readonly hoveredPart?: string | null;
  readonly selectedCell?: MoeCellRef | null;
  /** 0..5 MoE Math Stepper stage (`2` = Top-6 unbiased sigmoid normalization). */
  readonly stepperStep?: number | null;
  /** Explicit stepper active override (defaults to `stepperStep !== null`). */
  readonly stepperActive?: boolean;
  readonly reducedMotion?: boolean;
  /** Sub-step progress `[0..1)` controlling which Top-6 slot/cell is stepped. */
  readonly stepProgress?: number;
  /** Focused-cell term-by-term accumulation progress `[0..1]` (`< 0.999` triggers partial-denominator mode). */
  readonly termProgress?: number;
  readonly viewAngle?: GqaViewAngle;
}

/**
 * Compute exact 6th-vs-7th biased-score selection margin across all 128 routed experts for
 * `tokenIndex` using Float64 `scores_for_choice_f64` when available (falling back to Float32
 * `scores_for_choice` under `'float32_nano'`), preserving exact lower-index tie-break ordering.
 */
export function computeMoeSelectionMargin(
  sim: MoeSimResult,
  tokenIndex: number,
): MoeSelectionMarginInfo {
  const T = sim.tokens;
  const t = Math.max(0, Math.min(T - 1, Math.trunc(tokenIndex)));
  const nExperts = sim.nRoutedExperts;
  const choiceArr = sim.scores_for_choice_f64 ?? sim.scores_for_choice;
  const unbiasedArr = sim.router_scores_f64 ?? sim.scores;
  const logitsArr = sim.router_logits_f64 ?? sim.router_logits;

  const ranked = new Array<number>(nExperts);
  for (let e = 0; e < nExperts; e++) ranked[e] = e;
  ranked.sort((a, b) => {
    const sa = choiceArr[a * T + t];
    const sb = choiceArr[b * T + t];
    if (sb !== sa) return sb - sa;
    return a - b;
  });

  const rank6ExpertId = ranked[5];
  const rank7ExpertId = ranked[6];

  const rank6ChoiceScore = choiceArr[rank6ExpertId * T + t];
  const rank6UnbiasedScore = unbiasedArr[rank6ExpertId * T + t];
  const rank6Bias = sim.e_score_correction_bias[rank6ExpertId];

  const rank7ChoiceScore = choiceArr[rank7ExpertId * T + t];
  const rank7UnbiasedScore = unbiasedArr[rank7ExpertId * T + t];
  const rank7Bias = sim.e_score_correction_bias[rank7ExpertId];

  const choiceScoreMargin = rank6ChoiceScore - rank7ChoiceScore;
  const unbiasedScoreDelta = rank6UnbiasedScore - rank7UnbiasedScore;
  const biasDelta = rank6Bias - rank7Bias;

  const top8Count = Math.min(8, nExperts);
  const top8Summary: MoeTop8MarginEntry[] = [];
  for (let idx = 0; idx < top8Count; idx++) {
    const eid = ranked[idx];
    const rank = idx + 1;
    top8Summary.push(
      Object.freeze({
        rank,
        expertId: eid,
        expertTag: formatExpertId(eid),
        selected: rank <= sim.topK,
        cutoffBoundaryAfter: rank === sim.topK,
        choiceScore: choiceArr[eid * T + t],
        unbiasedScore: unbiasedArr[eid * T + t],
        bias: sim.e_score_correction_bias[eid],
        logit: logitsArr[eid * T + t],
      }),
    );
  }

  return Object.freeze({
    tokenIndex: t,
    absoluteTokenPosition: sim.startTokenOffset + t,
    rank6ExpertId,
    rank6ChoiceScore,
    rank6UnbiasedScore,
    rank6Bias,
    rank7ExpertId,
    rank7ChoiceScore,
    rank7UnbiasedScore,
    rank7Bias,
    choiceScoreMargin,
    unbiasedScoreDelta,
    biasDelta,
    routerPrecisionPolicy: sim.routerPrecisionPolicy,
    isTightMargin: choiceScoreMargin < 0.02,
    top8Summary: Object.freeze(top8Summary),
  });
}

function buildUnavailableMoeScene(params: {
  readonly view: ReturnType<typeof resolveTwoTowerSessionViewState>;
  readonly bayId: number;
  readonly towerRole: 'context' | 'denoiser';
  readonly contextTraceStatus: 'pruned_unavailable' | 'not_applicable';
  readonly warningBanner: string;
}): TwoTowerSpatialScene {
  const { view, bayId, towerRole, contextTraceStatus, warningBanner } = params;
  const layerIndex = bayId - 1;
  const bayStr = `Bay ${String(bayId).padStart(2, '0')} (L${layerIndex})`;
  const activeSelection = buildTwoTowerSelectionIdentity(view, {
    towerRole,
    bayId,
    mixerKind: 'E',
    tensorRole: 'moe_unavailable',
    traceOrigin: towerRole === 'denoiser' ? 'denoiser_scratch' : 'persistent_cache',
    executionId: view.identity.recordId,
    absoluteTokenPosition: null,
    provenanceNote: warningBanner,
  });

  const diagnostics: TwoTowerMoeDiagnostics = Object.freeze({
    bayId,
    layerIndex,
    towerRole,
    contextTraceStatus,
    isHistoricalDenoiserInspection: false,
    matchesCurrentContextBoundary: false,
    denoiserSourceRecordIndex: null,
    denoiserSourceBlockId: null,
    denoiserSourceRefinementIndex: null,
    denoiserSourceContextPrefixLen: null,
    denoiserSourceContextCacheDigest: null,
    historicalProvenanceWarning: null,
    routerPrecisionPolicy: view.model.config.routerPrecisionPolicy,
    selectionMargin: null,
    modulationOrder: 'norm_then_modulate' as const,
    modulationStatus: 'bypassed' as const,
    residualGateApplied: false,
    cacheBytes: 0 as const,
    stateless: true as const,
    crossTowerCacheTransfer: false as const,
    focusToken: null,
    absoluteTokenPosition: null,
    focusExpert: null,
    partialDenominatorActive: false,
    runningDenominator: null,
    currentTermIndex: null,
    sidebarWeights: Object.freeze([]),
    sidebarWeightTexts: Object.freeze([]),
  });

  return {
    viewMode: 'moe',
    title: `TwoTower Granular MoE (128+1) · ${bayStr}`,
    subtitle: `${towerRole === 'denoiser' ? 'Denoiser Tower' : 'Context Tower'} · Trace Unavailable · 0 B Stateless`,
    bayId,
    layerIndex,
    activeTowerFocus: towerRole === 'denoiser' ? 'denoiser_solo' : 'context_solo',
    solidData: new Float32Array(0),
    solidCount: 0,
    lineData: new Float32Array(0),
    lineCount: 0,
    ribbonData: new Float32Array(0),
    ribbonCount: 0,
    fitBounds: MOE_SPATIAL_FIT_BOUNDS,
    landmarks: [
      {
        id: 'moe-unavailable-notice',
        label: warningBanner,
        world: [0, 1.0, 0],
        partId: 'moe-unavailable',
        align: 'top',
        tier: 'primary',
        tone: 'muted',
      },
    ],
    kvDepthTicks: [],
    solidPickBoxes: [],
    fasciaPickBoxes: [],
    partCatalog: {
      'moe-unavailable': {
        id: 'moe-unavailable',
        title: `${bayStr} · ${towerRole === 'denoiser' ? 'Denoiser' : 'Context'} MoE Trace Unavailable`,
        badge: '0 B Stateless · No Synthetic Trace Substituted',
        summary: warningBanner,
        specs: [
          'MoE layers are strictly stateless (0 B sequence cache; zero cross-tower cache transfer).',
          'Historical Context traces pruned to bound memory are never replaced with synthetic fixtures.',
        ],
      },
    },
    activeSelection,
    stepperSteps: MOE_STEPPER_STEPS,
    provenanceBadges: [
      { label: towerRole === 'denoiser' ? 'DENOISER TOWER' : 'CONTEXT TOWER', tone: towerRole },
      { label: '0 B Stateless · No Cross-Tower Cache', tone: 'shared' },
      { label: 'Trace Unavailable (No Fake Substitute)', tone: 'warn' },
    ],
    epistemicNote:
      'Educational miniature TwoTower simulator (d=8, 128 routed + 1 shared expert, FP64 router precision). No standalone fixture is ever substituted for pruned or unexecuted traces.',
    warningBanner,
    diagnostics: diagnostics as unknown as Readonly<Record<string, unknown>>,
  };
}

/**
 * Build the TwoTower Granular MoE (128+1) 3D spatial cutaway directly from a `TwoTowerSession`,
 * reusing Nano's verified `buildMoECutaway` over the selected tower's REAL `MoeSimResult`,
 * computing exact FP64 6th-vs-7th selection margin, and strictly enforcing Nano v1.0's
 * partial-denominator non-exposure contract during Step 2 (`termProgress < 0.999`).
 */
export function buildTwoTowerMoeScene(
  session: TwoTowerSession,
  options?: BuildTwoTowerMoeOptions,
): TwoTowerSpatialScene {
  const view = resolveTwoTowerSessionViewState(session);
  const bayId = resolveNearestBayOfKind(options?.bayId ?? 2, 'E');
  const layerIndex = bayId - 1;
  const towerRole: 'context' | 'denoiser' =
    options?.towerRole ?? (view.denoiserLookup.pass !== null ? 'denoiser' : 'context');

  // Resolve REAL tower-specific MoeSimResult (never substitute a standalone Nano fixture!)
  let sim: MoeSimResult;
  let executionId: string;
  let traceOrigin: TwoTowerTraceOrigin;
  let modulationActive = false;
  let contextTraceStatus: 'available' | 'pruned_unavailable' | 'not_applicable' = 'not_applicable';

  if (towerRole === 'denoiser') {
    const denPass = view.denoiserLookup.pass;
    if (!denPass) {
      return buildUnavailableMoeScene({
        view,
        bayId,
        towerRole: 'denoiser',
        contextTraceStatus: 'not_applicable',
        warningBanner:
          'No Denoiser forward pass has executed yet on this branch; switch to Context Tower or run a Denoiser refinement step.',
      });
    }
    const denTrace = view.resolveDenoiserBayTrace(bayId);
    if (!denTrace || denTrace.kind !== 'E') {
      return buildUnavailableMoeScene({
        view,
        bayId,
        towerRole: 'denoiser',
        contextTraceStatus: 'not_applicable',
        warningBanner: `Denoiser trace for MoE Bay ${bayId} is unavailable.`,
      });
    }
    sim = denTrace.sim;
    executionId = denTrace.passExecutionId;
    traceOrigin = denTrace.traceOrigin;
    modulationActive = denTrace.modulation !== null;
  } else {
    const ctxLookup = view.resolveContextBayTrace(bayId);
    if (ctxLookup.status !== 'available' || !ctxLookup.trace || ctxLookup.trace.kind !== 'E') {
      return buildUnavailableMoeScene({
        view,
        bayId,
        towerRole: 'context',
        contextTraceStatus: 'pruned_unavailable',
        warningBanner:
          ctxLookup.reason ||
          'Historical Context MoE execution trace is unavailable after pruning; persistent Context snapshot remains authoritative (no synthetic MoE trace substituted).',
      });
    }
    contextTraceStatus = 'available';
    sim = ctxLookup.trace.sim;
    executionId = ctxLookup.segment!.stepExecutionId;
    traceOrigin = ctxLookup.traceOrigin;
    modulationActive = false;
  }

  const clampedT = sim.tokens;
  const focusToken = Math.max(
    0,
    Math.min(clampedT - 1, Math.trunc(options?.selectedCell?.tokenIndex ?? options?.focusToken ?? clampedT - 1)),
  );
  const defaultFocusExpert =
    options?.focusExpert !== undefined
      ? Math.max(0, Math.min(MOE_SIM_CONFIG.nRoutedExperts - 1, Math.trunc(options.focusExpert)))
      : sim.topk_indices[focusToken * sim.topK];

  const stepperStep =
    options?.stepperStep !== undefined && options?.stepperStep !== null
      ? Math.max(0, Math.min(5, Math.trunc(options.stepperStep)))
      : null;
  const stepperActive = options?.stepperActive ?? stepperStep !== null;
  const reducedMotion = options?.reducedMotion ?? false;
  const stepProgress = options?.stepProgress ?? 0;
  const termProgress = options?.termProgress;

  // Deterministic stepper cell when stepper is active and caller did not pin an explicit cell
  const effectiveCell: MoeCellRef | null =
    options?.selectedCell ??
    (stepperActive && stepperStep !== null
      ? moeStepperCellForState(sim, stepperStep, stepProgress, focusToken, defaultFocusExpert)
      : null);

  const viewAngle: GqaViewAngle = options?.viewAngle ?? 'three_quarter';
  const nanoBayId: 2 | 52 = bayId === 52 ? 52 : 2;

  // Directly reuse Nano's verified spatial MoE builder with our REAL TwoTower MoeSimResult!
  const nanoScene = buildMoECutaway({
    bayId: nanoBayId,
    tokens: clampedT,
    precision: 'bf16',
    selectedPart: options?.selectedPart ?? null,
    hoveredPart: options?.hoveredPart ?? null,
    focusToken,
    activeExpert: defaultFocusExpert,
    activeCell: effectiveCell,
    stepperStep,
    stepperActive,
    reducedMotion,
    stepperSubProgress: stepProgress,
    termProgress,
    viewAngle,
    simOverride: sim,
  });

  const resolvedFocusToken = nanoScene.moeStats?.focusToken ?? focusToken;
  const resolvedFocusExpert = nanoScene.moeStats?.focusExpert ?? defaultFocusExpert;
  const selectionMargin = computeMoeSelectionMargin(sim, resolvedFocusToken);

  // =========================================================================
  // STRICT NANO v1.0 PARTIAL-DENOMINATOR NON-EXPOSURE PRESERVATION
  // =========================================================================
  const accum = nanoScene.focusedAccumulation ?? null;
  const revealMaskActive = stepperActive && !reducedMotion && stepperStep !== null;
  const isPartialDenomStep =
    revealMaskActive &&
    stepperStep === 2 &&
    accum !== null &&
    accum.active &&
    !accum.isFinal;

  const isTopkStepRevealing =
    revealMaskActive &&
    stepperStep === 2 &&
    effectiveCell?.tensor === 'topk_weights';

  const activeSlot =
    effectiveCell?.tensor === 'topk_weights'
      ? Math.max(0, Math.min(sim.topK - 1, effectiveCell.rankSlot ?? effectiveCell.col))
      : 0;

  // 1. activeFormula: NEVER expose completed final weight or completed denominator when accumulating!
  let activeFormula: LayerFormulaReadout | null = nanoScene.activeFormula ?? null;
  if (activeFormula && accum !== null && accum.active) {
    activeFormula = {
      ...activeFormula,
      value: accum.partialSum,
      valueText: accum.valueBadgeText,
      equation: accum.partialEquation,
      subEquation: accum.partialSubEquation,
      headBadge: accum.badgeText,
    };
  }

  // 2. activeCellGrids: sanitize `topk_weights` raw values array during steps 0..2 so completed
  //    final normalized weights NEVER leak in `values` or overrides before accumulation finishes!
  const activeCellGrids: readonly ActiveSheetCellGrid[] | undefined = nanoScene.activeCellGrids?.map(
    (grid): ActiveSheetCellGrid => {
      if (grid.colDimLabel !== 'Top-6 (×2.5)' || !revealMaskActive || stepperStep === null || stepperStep > 2) {
        return grid;
      }
      if (stepperStep < 2) {
        return {
          ...grid,
          values: new Float32Array(grid.cols),
          revealMode: SHEET_REVEAL_UNCOMPUTED,
        };
      }
      // stepperStep === 2: sanitize activeSlot (when !isFinal) and all future slots (> activeSlot)
      const sanitizedValues = new Float32Array(grid.cols);
      const isFinalTerm = accum ? accum.isFinal : true;
      for (let j = 0; j < grid.cols; j++) {
        if (j < activeSlot) {
          sanitizedValues[j] = grid.values[j];
        } else if (j === activeSlot) {
          sanitizedValues[j] =
            accum && accum.active ? accum.partialSum : isFinalTerm ? grid.values[j] : 0;
        } else {
          sanitizedValues[j] = 0;
        }
      }
      return {
        ...grid,
        values: sanitizedValues,
      };
    },
  );

  // 3. diagnostics.sidebarWeights: exact Nano v1.0 non-spoiling sidebar weight masking
  const sidebarWeights: TwoTowerMoeSidebarWeightEntry[] = [];
  const sidebarWeightTexts: string[] = [];
  const isSlotFinal = accum ? accum.isFinal : true;

  for (let j = 0; j < sim.topK; j++) {
    const eid = sim.topk_indices[resolvedFocusToken * sim.topK + j];
    const w = sim.topk_weights[resolvedFocusToken * sim.topK + j];
    let displayText: string;
    let weight: number | null;
    let state: 'uncomputed' | 'accumulating' | 'revealed';

    if (!revealMaskActive || stepperStep === null || stepperStep > 2) {
      displayText = w.toFixed(2);
      weight = w;
      state = 'revealed';
    } else if (stepperStep < 2 || j > activeSlot) {
      displayText = '—';
      weight = null;
      state = 'uncomputed';
    } else if (j < activeSlot) {
      displayText = w.toFixed(2);
      weight = w;
      state = 'revealed';
    } else if (isSlotFinal) {
      displayText = w.toFixed(2);
      weight = w;
      state = 'revealed';
    } else {
      displayText = '…';
      weight = null;
      state = 'accumulating';
    }

    sidebarWeights.push(
      Object.freeze({
        rankSlot: j,
        expertId: eid,
        expertTag: formatExpertId(eid),
        displayText,
        weight,
        state,
      }),
    );
    sidebarWeightTexts.push(displayText);
  }

  // Propagate tower-aware PickBox provenance and actual 1-based bayId (2..52)
  const solidPickBoxes: PickBox[] = nanoScene.solidPickBoxes.map((box) => ({
    ...box,
    bayId,
    towerRole,
    executionId,
    traceOrigin,
    semanticTag: 'moe',
    absoluteTokenOffset: sim.startTokenOffset,
  }));

  // Augment partCatalog with TwoTower FP64 margin, post-norm AdaLN, historical provenance, and 0 B stateless metadata
  const dl = view.denoiserLookup;
  const isHistoricalDenoiser = towerRole === 'denoiser' && dl.isHistoricalInspection;
  const historicalTag =
    isHistoricalDenoiser && dl.pass !== null
      ? `Historical Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, ${dl.pass.passExecutionId}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'} vs P=${view.committedPrefixLen})`
      : null;

  const bayStr = `Bay ${String(bayId).padStart(2, '0')} (L${layerIndex})`;
  const towerLabel = towerRole === 'denoiser' ? 'Denoiser Tower' : 'Context Tower';
  const marginText =
    `6th ${formatExpertId(selectionMargin.rank6ExpertId)} (${selectionMargin.rank6ChoiceScore.toFixed(4)}) vs ` +
    `7th ${formatExpertId(selectionMargin.rank7ExpertId)} (${selectionMargin.rank7ChoiceScore.toFixed(4)}) · ` +
    `Δ = ${selectionMargin.choiceScoreMargin.toFixed(4)}`;

  const partCatalog: Record<string, CutawayPartInfo> = {
    ...nanoScene.partCatalog,
    'w-router': {
      ...nanoScene.partCatalog['w-router'],
      badge:
        sim.routerPrecisionPolicy === 'float64_reference'
          ? 'FP64 Reference Router (TwoTower)'
          : nanoScene.partCatalog['w-router'].badge,
      specs: [
        ...nanoScene.partCatalog['w-router'].specs,
        `Router precision policy: ${sim.routerPrecisionPolicy}`,
        `Selection margin (6th vs 7th): ${marginText}`,
        ...(historicalTag ? [`Historical Denoiser provenance: ${historicalTag}`] : []),
      ],
    },
    rmsnorm: {
      ...nanoScene.partCatalog.rmsnorm,
      badge: modulationActive
        ? 'Post-Norm AdaLN ACTIVE (norm_then_modulate)'
        : towerRole === 'context'
          ? 'Unmodulated Context Tower'
          : 'AdaLN BYPASSED (Mock-AR)',
      specs: [
        ...nanoScene.partCatalog.rmsnorm.specs,
        modulationActive
          ? 'TwoTower MoE ordering: RMSNorm(X) → U → U_mod = U·(1+scale)+shift (norm_then_modulate)'
          : towerRole === 'context'
            ? 'Context Tower never applies timestep modulation; U feeds router and experts unmodulated.'
            : 'Timestep modulation bypassed in Mock-AR; U feeds router and experts unmodulated.',
      ],
    },
    'residual-out': {
      ...nanoScene.partCatalog['residual-out'],
      badge: sim.residualGateApplied
        ? 'Outer Residual Gate ACTIVE · Y = X + gate·y_moe'
        : 'Residual Add · Y = X + y_moe',
      specs: [
        `${towerLabel} ${bayStr} · 0 B stateless (NO cross-tower cache transfer)`,
        sim.residualGateApplied
          ? 'Channel-wise outer AdaLN residual gate (d=8) scales y_moe before merging with untouched X.'
          : 'Unmodulated residual merge (Context Tower or Mock-AR).',
      ],
    },
  };

  // Augment landmarks with TwoTower margin + stateless + AdaLN badges (keeping existing IDs intact)
  const landmarks: LandmarkAnchor[] = [
    ...nanoScene.landmarks,
    {
      id: 'twotower-moe-margin',
      label: `FP64 Top-6 Cutoff: ${marginText}${selectionMargin.isTightMargin ? ' (TIGHT)' : ''}`,
      sublabel:
        `${towerLabel} ${bayStr} · ${modulationActive ? 'Post-Norm AdaLN + Outer Gate ACTIVE' : 'AdaLN BYPASSED'} · 0 B Stateless` +
        (historicalTag ? ` · ${historicalTag}` : ''),
      world: [0.0, 3.08, -2.55],
      partId: 'w-router',
      align: 'top',
      tier: 'primary',
      tone: isHistoricalDenoiser ? 'gold' : selectionMargin.isTightMargin ? 'gold' : 'cyan',
    },
  ];

  const provenanceBadges: TwoTowerProvenanceBadge[] = [
    {
      label: `${towerRole === 'denoiser' ? 'DENOISER' : 'CONTEXT'} · ${bayStr}`,
      tone: towerRole,
      origin: traceOrigin,
    },
    {
      label: `Router: ${sim.routerPrecisionPolicy === 'float64_reference' ? 'FP64 Reference' : 'FP32 Nano'}`,
      tone: 'shared',
    },
    {
      label: `6th-vs-7th Δ = ${selectionMargin.choiceScoreMargin.toFixed(4)}`,
      tone: selectionMargin.isTightMargin ? 'warn' : 'gold',
    },
    {
      label: modulationActive ? 'Post-Norm AdaLN + Gate: ACTIVE' : 'Post-Norm AdaLN + Gate: BYPASSED',
      tone: modulationActive ? 'denoiser' : 'muted',
    },
    {
      label: '0 B Stateless · No Cross-Tower Cache',
      tone: 'shared',
    },
  ];
  if (isHistoricalDenoiser) {
    provenanceBadges.push({
      label: `Historical Denoiser Pass · Rec #${dl.sourceRecordIndex} (b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'}, P_cond=${dl.sourceContextPrefixLen ?? 'n/a'} vs P=${view.committedPrefixLen})`,
      tone: 'warn',
      origin: 'denoiser_scratch',
    });
  }

  const activeSelection = buildTwoTowerSelectionIdentity(view, {
    towerRole,
    bayId,
    mixerKind: 'E',
    tensorRole: effectiveCell?.tensor ?? 'router_bank',
    traceOrigin,
    executionId,
    absoluteTokenPosition: sim.startTokenOffset + resolvedFocusToken,
    absoluteTokenRange: [sim.startTokenOffset, sim.startTokenOffset + sim.tokens - 1],
    expertId: resolvedFocusExpert,
    provenanceNote:
      `${towerLabel} ${bayStr} · ${marginText} · 0 B stateless` +
      (historicalTag ? ` · ${historicalTag}` : ''),
  });

  const diagnostics: TwoTowerMoeDiagnostics = Object.freeze({
    bayId,
    layerIndex,
    towerRole,
    contextTraceStatus,
    isHistoricalDenoiserInspection: isHistoricalDenoiser,
    matchesCurrentContextBoundary: towerRole === 'denoiser' ? dl.matchesCurrentContextBoundary : true,
    denoiserSourceRecordIndex: towerRole === 'denoiser' ? dl.sourceRecordIndex : null,
    denoiserSourceBlockId: towerRole === 'denoiser' ? dl.sourceBlockId : null,
    denoiserSourceRefinementIndex: towerRole === 'denoiser' ? dl.sourceRefinementIndex : null,
    denoiserSourceContextPrefixLen: towerRole === 'denoiser' ? dl.sourceContextPrefixLen : null,
    denoiserSourceContextCacheDigest: towerRole === 'denoiser' ? dl.sourceContextCacheDigest : null,
    historicalProvenanceWarning: isHistoricalDenoiser ? dl.historicalProvenanceWarning : null,
    routerPrecisionPolicy: sim.routerPrecisionPolicy,
    selectionMargin,
    modulationOrder: 'norm_then_modulate' as const,
    modulationStatus: modulationActive ? ('active' as const) : ('bypassed' as const),
    residualGateApplied: sim.residualGateApplied,
    cacheBytes: 0 as const,
    stateless: true as const,
    crossTowerCacheTransfer: false as const,
    focusToken: resolvedFocusToken,
    absoluteTokenPosition: sim.startTokenOffset + resolvedFocusToken,
    focusExpert: resolvedFocusExpert,
    partialDenominatorActive: isPartialDenomStep,
    runningDenominator: isPartialDenomStep && accum ? accum.partialSum : null,
    currentTermIndex: isTopkStepRevealing && accum ? accum.currentTermIndex : null,
    sidebarWeights: Object.freeze(sidebarWeights),
    sidebarWeightTexts: Object.freeze(sidebarWeightTexts),
  });

  const warningParts: string[] = [];
  if (!view.snapshotIntegrity.valid) {
    warningParts.push(`Context snapshot integrity error: ${view.snapshotIntegrity.error}`);
  }
  if (isHistoricalDenoiser && dl.historicalProvenanceWarning !== null) {
    warningParts.push(dl.historicalProvenanceWarning);
  }
  const warningBanner = warningParts.length > 0 ? warningParts.join(' | ') : null;

  const atlasKey =
    `tt-moe|${view.model.twoTowerFingerprint}|${towerRole}|${executionId}|b${bayId}|` +
    `tok${resolvedFocusToken}|exp${resolvedFocusExpert}`;

  return {
    viewMode: 'moe',
    title: `TwoTower Granular MoE (128+1) · ${towerLabel} ${bayStr}`,
    subtitle:
      (historicalTag ? `${historicalTag} · ` : '') +
      `${sim.routerPrecisionPolicy === 'float64_reference' ? 'FP64 Router' : 'FP32 Router'} · ` +
      `6th-vs-7th Δ=${selectionMargin.choiceScoreMargin.toFixed(4)} · ` +
      `${modulationActive ? 'Post-Norm AdaLN ACTIVE' : 'AdaLN BYPASSED'} · 0 B Stateless`,
    bayId,
    layerIndex,
    activeTowerFocus: towerRole === 'denoiser' ? 'denoiser_solo' : 'context_solo',
    solidData: nanoScene.solidData,
    solidCount: nanoScene.solidCount,
    lineData: nanoScene.lineData,
    lineCount: nanoScene.lineCount,
    ribbonData: nanoScene.ribbonData,
    ribbonCount: nanoScene.ribbonCount,
    sheetData: nanoScene.sheetData,
    sheetCount: nanoScene.sheetCount,
    sheetBasisData: nanoScene.sheetBasisData,
    sheetRevealData: nanoScene.sheetRevealData,
    flowVertices: nanoScene.flowVertices,
    flowIndices: nanoScene.flowIndices,
    valueAtlas: nanoScene.valueAtlas,
    atlasKey,
    fitBounds: nanoScene.fitBounds,
    opFocusBounds: nanoScene.opFocusBounds,
    landmarks,
    kvDepthTicks: nanoScene.kvDepthTicks,
    calipers: nanoScene.calipers,
    solidPickBoxes,
    fasciaPickBoxes: [],
    partCatalog,
    activeOpPartIds: nanoScene.activeOpPartIds,
    activeCellGrids,
    cellPopover: nanoScene.cellPopover,
    activeFormula,
    focusedAccumulation: accum,
    activeSelection,
    stepperSteps: MOE_STEPPER_STEPS,
    provenanceBadges,
    epistemicNote:
      'Educational miniature TwoTower Float32/Float64 MoE simulator (d=8, 128 routed + 1 shared expert, 0 B stateless) preserving exact FP64 router precision, 6th-vs-7th margin, and non-spoiling partial denominator reveal; NOT 63.19B checkpoint weights.',
    warningBanner,
    diagnostics: diagnostics as unknown as Readonly<Record<string, unknown>>,
  };
}
