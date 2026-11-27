import type { GqaSimResult } from '../math/gqaSim';
import type { HybridModelParameters } from '../math/hybridFlow';
import {
  createDecodeSession,
  getActiveDecodeBranch,
  getActiveDecodeStep,
  navigateDecodeSession,
  stepDecodeSession,
  stepDecodeSessionGreedy,
  switchDecodeSessionBranch,
  type HybridDecodeSession,
} from '../math/incrementalDecode';
import type { MambaSimResult } from '../math/mambaSim';
import { moeStepperCellForState, type MoeCellRef, type MoeSimResult } from '../math/moeSim';
import {
  getWalkthroughBeat,
  TOUR_PREFILL_TOKEN_IDS,
  TOUR_PREFILL_TOKENS,
} from '../story/walkthrough';
import type { PrecisionMode } from './spec';
import type { SpineWaveMode } from './state';

// ============================================================================
// RC1 · SUB-CELL FOCUSED ACCUMULATION & PLAYBACK PACING
// ============================================================================

/** Progress values at or above this threshold settle cleanly to completed (`1.0`). */
export const ACCUM_SETTLE_THRESHOLD = 0.999;

/** Target pacing (~400 ms per operand term) during cutaway Math Stepper Play. */
const PLAY_MS_PER_ACCUM_TERM = 400;

/** Default stage sweep duration (ms) for non-accumulating cutaway Math Stepper stages. */
const PLAY_BASE_STEP_DURATION_MS = 2200;

/** Fractional split inside MoE Step 4 (`4. Routed ReLU²`): `routed_act` -> `routed_sum`. */
const MOE_STEP3_SUM_SPLIT = 0.25;

/** Fractional split inside MoE Step 6 (`6. Merge + Res`): `moe_out` -> `residual`. */
const MOE_STEP5_RESIDUAL_SPLIT = 0.60;

/**
 * Extract the sub-cell fractional progress `∈ [0, 0.998]` within a uniform cell sweep
 * of `cellCount` cells for `subProgress ∈ [0, ACCUM_SETTLE_THRESHOLD)`.
 * Returns `1` (settled) once `subProgress >= ACCUM_SETTLE_THRESHOLD`.
 */
export function computeSubCellFraction(subProgress: number, cellCount: number): number {
  if (!Number.isFinite(subProgress) || subProgress >= ACCUM_SETTLE_THRESHOLD) {
    return 1;
  }
  const safeCells = Math.max(1, Math.trunc(cellCount));
  const p = Math.max(0, Math.min(0.9999, subProgress));
  const scaled = p * safeCells;
  const frac = scaled - Math.floor(scaled);
  return Math.max(0, Math.min(0.998, frac));
}

/**
 * Resolve GQA sub-cell `termProgress ∈ [0, 1]`.
 * Forces `1` (unrestricted inspection) whenever `!stepperActive`, `reducedMotion`,
 * `hasDirectCellOverride` (hover/pinned cell), non-accumulating steps, or settled `subProgress >= 0.999`.
 */
export function resolveGqaAccumTermProgress(
  sim: GqaSimResult,
  stepperStep: number,
  subProgress: number,
  stepperActive: boolean,
  reducedMotion: boolean,
  hasDirectCellOverride: boolean,
): number {
  if (!stepperActive || reducedMotion || hasDirectCellOverride) return 1;
  if (!Number.isFinite(subProgress) || subProgress >= ACCUM_SETTLE_THRESHOLD) return 1;

  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  if (step === 2) {
    // 3. QK^T/sqrt(d): sweeps K = sim.keyTokens cells along latest query row
    return computeSubCellFraction(subProgress, sim.keyTokens);
  }
  if (step === 4) {
    // 5. Weighted V: sweeps dHead = 4 channels in V_out bank
    return computeSubCellFraction(subProgress, sim.dHead);
  }
  return 1;
}

/**
 * Resolve Mamba-2 sub-cell `termProgress ∈ [0, 1]`.
 * Matches `mambaStepperCellForState` cell partitioning on Step 3 (Conv1D), Step 4 (SSM Update),
 * and Step 5 (Readout + D). Forces `1` for unrestricted hover/click or `reducedMotion`.
 */
export function resolveMambaAccumTermProgress(
  sim: MambaSimResult,
  stepperStep: number,
  subProgress: number,
  stepperActive: boolean,
  reducedMotion: boolean,
  hasDirectCellOverride: boolean,
): number {
  if (!stepperActive || reducedMotion || hasDirectCellOverride) return 1;
  if (!Number.isFinite(subProgress) || subProgress >= ACCUM_SETTLE_THRESHOLD) return 1;

  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  const p01 = Math.max(0, Math.min(0.9999, subProgress));
  if (step === 2) {
    // 3. Causal Conv1D: [0, 0.45) sweeps convKernel slots; [0.45, 1) sweeps dHead post-SiLU x channels
    if (p01 < 0.45) {
      return computeSubCellFraction(p01 / 0.45, sim.convKernel);
    }
    return computeSubCellFraction((p01 - 0.45) / 0.55, sim.dHead);
  }
  if (step === 3) {
    // 4. SSM Update: sweeps dHead * dState cells of H_t
    return computeSubCellFraction(p01, sim.dHead * sim.dState);
  }
  if (step === 4) {
    // 5. Readout + D: sweeps y cells along max(tokens, dHead)
    return computeSubCellFraction(p01, Math.max(sim.tokens, sim.dHead));
  }
  return 1;
}

/**
 * Resolve the active MoE stepper cell during playback/scrubbing, extending Step 4 (`Routed ReLU²`)
 * into `routed_sum` and Step 6 (`Merge + Res`) into `moe_out` so 6-branch weighted accumulation
 * is visibly swept end-to-end without altering `moeStepperCellForState`.
 */
export function resolveMoeStepperCellForAccum(
  sim: MoeSimResult,
  stepperStep: number,
  subProgress: number,
  focusToken: number,
  focusExpert: number,
): MoeCellRef {
  const t = Math.max(0, Math.min(sim.tokens - 1, Math.round(focusToken)));
  const p = Math.max(0, Math.min(0.9999, subProgress));
  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));

  if (step === 3) {
    if (p < MOE_STEP3_SUM_SPLIT) {
      const localP = p / MOE_STEP3_SUM_SPLIT;
      const hRow = Math.min(sim.dExp - 1, Math.floor(localP * sim.dExp));
      return {
        tensor: 'routed_act',
        row: hRow,
        col: 0,
        tokenIndex: t,
        expertId: focusExpert,
      };
    }
    const localP = (p - MOE_STEP3_SUM_SPLIT) / (1 - MOE_STEP3_SUM_SPLIT);
    const ch = Math.min(sim.d - 1, Math.floor(localP * sim.d));
    return {
      tensor: 'routed_sum',
      row: t,
      col: ch,
      tokenIndex: t,
      expertId: focusExpert,
    };
  }

  if (step === 5) {
    if (p < MOE_STEP5_RESIDUAL_SPLIT) {
      const localP = p / MOE_STEP5_RESIDUAL_SPLIT;
      const ch = Math.min(sim.d - 1, Math.floor(localP * sim.d));
      return {
        tensor: 'moe_out',
        row: t,
        col: ch,
        tokenIndex: t,
      };
    }
    const localP = (p - MOE_STEP5_RESIDUAL_SPLIT) / (1 - MOE_STEP5_RESIDUAL_SPLIT);
    const ch = Math.min(sim.d - 1, Math.floor(localP * sim.d));
    return {
      tensor: 'residual',
      row: t,
      col: ch,
      tokenIndex: t,
    };
  }

  return moeStepperCellForState(sim, step, p, t, focusExpert);
}

/**
 * Resolve MoE sub-cell `termProgress ∈ [0, 1]`.
 * Supports Step 3 (`Top-6 ×2.5` 6-term unbiased σ denominator), Step 4 (`routed_sum` 6 branches),
 * and Step 6 (`moe_out` 6 branches + shared). Forces `1` when idle, hovered/pinned, or `reducedMotion`.
 */
export function resolveMoeAccumTermProgress(
  sim: MoeSimResult,
  stepperStep: number,
  subProgress: number,
  stepperActive: boolean,
  reducedMotion: boolean,
  hasDirectCellOverride: boolean,
): number {
  if (!stepperActive || reducedMotion || hasDirectCellOverride) return 1;
  if (!Number.isFinite(subProgress) || subProgress >= ACCUM_SETTLE_THRESHOLD) return 1;

  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  const p = Math.max(0, Math.min(0.9999, subProgress));
  if (step === 2) {
    // 3. Top-6 (×2.5): sweeps topK = 6 slots
    return computeSubCellFraction(p, sim.topK);
  }
  if (step === 3 && p >= MOE_STEP3_SUM_SPLIT) {
    const localP = (p - MOE_STEP3_SUM_SPLIT) / (1 - MOE_STEP3_SUM_SPLIT);
    return computeSubCellFraction(localP, sim.d);
  }
  if (step === 5 && p < MOE_STEP5_RESIDUAL_SPLIT) {
    const localP = p / MOE_STEP5_RESIDUAL_SPLIT;
    return computeSubCellFraction(localP, sim.d);
  }
  return 1;
}

/** Play duration (seconds) for one GQA stepper stage, paced ~400 ms per operand term. */
export function gqaStepPlayDurationSec(sim: GqaSimResult, stepperStep: number): number {
  const secPerTerm = PLAY_MS_PER_ACCUM_TERM / 1000;
  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  if (step === 2) {
    return Math.max(PLAY_BASE_STEP_DURATION_MS / 1000, sim.keyTokens * sim.dHead * secPerTerm);
  }
  if (step === 4) {
    return Math.max(PLAY_BASE_STEP_DURATION_MS / 1000, sim.dHead * sim.keyTokens * secPerTerm);
  }
  return PLAY_BASE_STEP_DURATION_MS / 1000;
}

/** Play duration (seconds) for one Mamba-2 stepper stage, paced ~400 ms per operand term. */
export function mambaStepPlayDurationSec(sim: MambaSimResult, stepperStep: number): number {
  const secPerTerm = PLAY_MS_PER_ACCUM_TERM / 1000;
  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  if (step === 2) {
    return Math.max(
      PLAY_BASE_STEP_DURATION_MS / 1000,
      (sim.convKernel + sim.dHead) * sim.convKernel * secPerTerm,
    );
  }
  if (step === 3) {
    return Math.max(
      PLAY_BASE_STEP_DURATION_MS / 1000,
      sim.dHead * sim.dState * 2 * secPerTerm,
    );
  }
  if (step === 4) {
    return Math.max(
      PLAY_BASE_STEP_DURATION_MS / 1000,
      Math.max(sim.tokens, sim.dHead) * (sim.dState + 1) * secPerTerm,
    );
  }
  return PLAY_BASE_STEP_DURATION_MS / 1000;
}

/** Play duration (seconds) for one MoE stepper stage, paced ~400 ms per operand term. */
export function moeStepPlayDurationSec(sim: MoeSimResult, stepperStep: number): number {
  const secPerTerm = PLAY_MS_PER_ACCUM_TERM / 1000;
  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  if (step === 2) {
    return Math.max(PLAY_BASE_STEP_DURATION_MS / 1000, sim.topK * sim.topK * secPerTerm);
  }
  if (step === 3) {
    return Math.max(
      PLAY_BASE_STEP_DURATION_MS / 1000,
      (sim.d * sim.topK * secPerTerm) / (1 - MOE_STEP3_SUM_SPLIT),
    );
  }
  if (step === 5) {
    return Math.max(
      PLAY_BASE_STEP_DURATION_MS / 1000,
      (sim.d * sim.topK * secPerTerm) / MOE_STEP5_RESIDUAL_SPLIT,
    );
  }
  return PLAY_BASE_STEP_DURATION_MS / 1000;
}

/** Discrete operand-term index bucket used to trigger dirty geometry rebuilds on term ticks. */
export function gqaAccumTermBucket(
  sim: GqaSimResult,
  stepperStep: number,
  termProgress: number,
): number {
  if (!Number.isFinite(termProgress) || termProgress >= ACCUM_SETTLE_THRESHOLD) return -1;
  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  const totalTerms = step === 2 ? sim.dHead : step === 4 ? sim.keyTokens : 0;
  if (totalTerms <= 1) return -1;
  return Math.min(totalTerms - 1, Math.floor(Math.max(0, termProgress) * totalTerms));
}

export function mambaAccumTermBucket(
  sim: MambaSimResult,
  stepperStep: number,
  termProgress: number,
): number {
  if (!Number.isFinite(termProgress) || termProgress >= ACCUM_SETTLE_THRESHOLD) return -1;
  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  const totalTerms = step === 2 ? sim.convKernel : step === 3 ? 2 : step === 4 ? sim.dState + 1 : 0;
  if (totalTerms <= 1) return -1;
  return Math.min(totalTerms - 1, Math.floor(Math.max(0, termProgress) * totalTerms));
}

export function moeAccumTermBucket(
  sim: MoeSimResult,
  stepperStep: number,
  termProgress: number,
): number {
  if (!Number.isFinite(termProgress) || termProgress >= ACCUM_SETTLE_THRESHOLD) return -1;
  const step = Math.max(0, Math.min(5, Math.round(stepperStep)));
  const totalTerms = step === 2 || step === 3 || step === 5 ? sim.topK : 0;
  if (totalTerms <= 1) return -1;
  return Math.min(totalTerms - 1, Math.floor(Math.max(0, termProgress) * totalTerms));
}

// ============================================================================
// RC2 · ISOLATED TOUR VS. FREE-EXPLORE AR DECODE SESSION STORE
// ============================================================================

const FREE_EXPLORE_DEFAULT_PROMPT_IDS: readonly number[] = Object.freeze([0, 1, 2, 3]);

export interface DualDecodeSessionStore {
  readonly freeExplore: HybridDecodeSession;
  readonly tour: HybridDecodeSession | null;
}

export type DecodeSessionAction =
  | { readonly type: 'step_token'; readonly tokenId: number }
  | { readonly type: 'step_greedy' }
  | { readonly type: 'nav_step'; readonly stepIndex: number }
  | { readonly type: 'switch_branch'; readonly branchId: string }
  | { readonly type: 'reset' };

/** True when the Guided Tour is currently positioned on an Act III AR Decode beat (`7.2`–`7.4`). */
export function isTourArDecodeContext(tourActive: boolean, tourBeat: number): boolean {
  return tourActive && getWalkthroughBeat(tourBeat).executionMode === 'ar_decode';
}

export function createDualDecodeSessionStore(
  model: HybridModelParameters,
  precision: PrecisionMode = 'bf16',
): DualDecodeSessionStore {
  return {
    freeExplore: createDecodeSession(model, {
      promptTokenIds: FREE_EXPLORE_DEFAULT_PROMPT_IDS,
      precisionMetadata: precision,
    }),
    tour: null,
  };
}

/**
 * Idempotently ensure the canonical Guided Tour AR session is seeded with
 * `TOUR_PREFILL_TOKEN_IDS = [0..7]` and has at least one cached AR decode step (`8P + 1S = 9`),
 * preserving `freeExplore` untouched.
 */
export function ensureTourArDecodeSession(
  store: DualDecodeSessionStore,
  model: HybridModelParameters,
  precision: PrecisionMode = 'fp8_selective',
): DualDecodeSessionStore {
  let tour = store.tour;
  if (!tour) {
    const prefill = createDecodeSession(model, {
      promptTokenIds: TOUR_PREFILL_TOKEN_IDS,
      precisionMetadata: precision,
    });
    tour = stepDecodeSessionGreedy(prefill);
    return { freeExplore: store.freeExplore, tour };
  }

  const activeStep = getActiveDecodeStep(tour);
  if (activeStep.totalTokens <= TOUR_PREFILL_TOKENS) {
    const branch = getActiveDecodeBranch(tour);
    if (branch.steps.length > 1) {
      tour = navigateDecodeSession(tour, 1);
    } else {
      tour = stepDecodeSessionGreedy(tour);
    }
    return { freeExplore: store.freeExplore, tour };
  }

  return store;
}

/**
 * Return the active AR decode session (`tour` during Act III AR beats; `freeExplore` everywhere else).
 */
export function resolveActiveDecodeSession(
  store: DualDecodeSessionStore,
  tourActive: boolean,
  tourBeat: number,
  model: HybridModelParameters,
  precision: PrecisionMode = 'fp8_selective',
): {
  readonly store: DualDecodeSessionStore;
  readonly session: HybridDecodeSession;
  readonly isTour: boolean;
} {
  if (isTourArDecodeContext(tourActive, tourBeat)) {
    const ensured = store.tour ? store : ensureTourArDecodeSession(store, model, precision);
    return {
      store: ensured,
      session: ensured.tour!,
      isTour: true,
    };
  }
  return {
    store,
    session: store.freeExplore,
    isTour: false,
  };
}

/**
 * Apply an interactive AR decode mutation strictly to the active session (`tour` inside Act III
 * AR beats, `freeExplore` everywhere else), guaranteeing zero cross-session bleed.
 */
export function applyDecodeActionToActiveSession(
  store: DualDecodeSessionStore,
  tourActive: boolean,
  tourBeat: number,
  model: HybridModelParameters,
  precision: PrecisionMode,
  action: DecodeSessionAction,
): {
  readonly store: DualDecodeSessionStore;
  readonly session: HybridDecodeSession;
  readonly isTour: boolean;
} {
  const resolved = resolveActiveDecodeSession(store, tourActive, tourBeat, model, precision);
  let nextSession = resolved.session;

  switch (action.type) {
    case 'step_token':
      nextSession = stepDecodeSession(nextSession, action.tokenId);
      break;
    case 'step_greedy':
      nextSession = stepDecodeSessionGreedy(nextSession);
      break;
    case 'nav_step':
      nextSession = navigateDecodeSession(nextSession, action.stepIndex);
      break;
    case 'switch_branch':
      nextSession = switchDecodeSessionBranch(nextSession, action.branchId);
      break;
    case 'reset':
      nextSession = createDecodeSession(model, {
        promptTokenIds: resolved.isTour
          ? TOUR_PREFILL_TOKEN_IDS
          : FREE_EXPLORE_DEFAULT_PROMPT_IDS,
        precisionMetadata: precision,
      });
      break;
  }

  const nextStore: DualDecodeSessionStore = resolved.isTour
    ? { freeExplore: resolved.store.freeExplore, tour: nextSession }
    : { freeExplore: nextSession, tour: resolved.store.tour };

  return {
    store: nextStore,
    session: nextSession,
    isTour: resolved.isTour,
  };
}

/** Reset ONLY `tour` (e.g. Guided Tour Restart) while preserving `freeExplore` completely intact. */
export function resetTourOnlyDecodeSession(
  store: DualDecodeSessionStore,
): DualDecodeSessionStore {
  return {
    freeExplore: store.freeExplore,
    tour: null,
  };
}

// ============================================================================
// RC3 · FAST (0.9s) & INSTRUCTIONAL (5.5s) SPINE REPLAY CLOCK
// ============================================================================

export const SPINE_WAVE_FAST_DURATION_MS = 900;
export const SPINE_WAVE_INSTRUCTIONAL_DURATION_MS = 5500;

export function spineWaveDurationMs(mode: SpineWaveMode): number {
  return mode === 'instructional'
    ? SPINE_WAVE_INSTRUCTIONAL_DURATION_MS
    : SPINE_WAVE_FAST_DURATION_MS;
}

export interface SpineWavePlaybackState {
  readonly progress: number | null;
  readonly mode: SpineWaveMode;
  readonly paused: boolean;
  readonly elapsedMs: number;
}

export function createIdleSpineWaveState(mode: SpineWaveMode = 'fast'): SpineWavePlaybackState {
  return {
    progress: null,
    mode,
    paused: false,
    elapsedMs: 0,
  };
}

/**
 * Start a Tower spine propagation replay in `'fast'` (900 ms) or `'instructional'` (5500 ms) mode.
 * Under `reducedMotion`, settles immediately to a dismissible static summary at `progress = 1.0`.
 */
export function startSpineWavePlayback(
  mode: SpineWaveMode,
  reducedMotion: boolean,
): SpineWavePlaybackState {
  if (reducedMotion) {
    return {
      progress: 1.0,
      mode,
      paused: false,
      elapsedMs: spineWaveDurationMs(mode),
    };
  }
  return {
    progress: 0,
    mode,
    paused: false,
    elapsedMs: 0,
  };
}

export function toggleSpineWavePlaybackPause(
  wave: SpineWavePlaybackState,
  reducedMotion: boolean,
): SpineWavePlaybackState {
  if (reducedMotion || wave.progress === null) return wave;
  return {
    ...wave,
    paused: !wave.paused,
  };
}

export function advanceSpineWavePlayback(
  wave: SpineWavePlaybackState,
  dtMs: number,
  reducedMotion: boolean,
): {
  readonly next: SpineWavePlaybackState;
  readonly finished: boolean;
  readonly animating: boolean;
} {
  if (wave.progress === null) {
    return { next: wave, finished: false, animating: false };
  }
  if (reducedMotion) {
    return {
      next: { ...wave, progress: 1.0, paused: false },
      finished: false,
      animating: false,
    };
  }
  if (wave.paused) {
    return { next: wave, finished: false, animating: false };
  }

  const duration = spineWaveDurationMs(wave.mode);
  const clampedDt = Number.isFinite(dtMs) ? Math.max(0, Math.min(100, dtMs)) : 0;
  const elapsedMs = wave.elapsedMs + clampedDt;
  const progress = Math.max(0, Math.min(1, elapsedMs / duration));
  if (progress >= 1) {
    return {
      next: createIdleSpineWaveState(wave.mode),
      finished: true,
      animating: false,
    };
  }
  return {
    next: {
      progress,
      mode: wave.mode,
      paused: false,
      elapsedMs,
    },
    finished: false,
    animating: true,
  };
}
