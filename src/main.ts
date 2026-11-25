import './ui/theme.css';

import { NEMOTRON_SPEC, type BayDescriptor, type PrecisionMode } from './model/spec';
type BayKind = BayDescriptor['kind'];
import {
  CAMERA_FLY_DURATION_MS,
  cloneCameraPose,
  createInitialState,
  DEFAULT_FOV_DEG,
  DEFAULT_TOKENS,
  MACRO_DEFAULT_AZIMUTH_DEG,
  MACRO_DEFAULT_ELEVATION_DEG,
  MAX_TOKENS,
  MIN_TOKENS,
  type CameraPose,
  type ExecutionMode,
  type GqaViewAngle,
  type SceneState,
  type SpineWaveMode,
  type TocSectionId,
  type ViewMode,
} from './model/state';
import {
  buildCameraSnapshot,
  fitOpBoundsCamera,
  fitPerspectiveToBounds,
  interpolateCameraPose,
  makeViewportMetrics,
  orbitCameraByCssDelta,
  panCameraByCssDelta,
  zoomPerspectiveDistance,
  type AABB,
  type CameraSnapshot,
  type ViewportMetrics,
} from './render/camera';
import {
  GQA_STEPPER_STEPS,
  inferGqaCellHead,
  stepperCellForState,
  type GqaCellRef,
} from './math/gqaSim';
import {
  createDefaultHybridModel,
  defaultTokenIdsForLength,
  getOrRunHybridExecution,
  HybridExecutionCache,
  type HybridExecutionResult,
} from './math/hybridFlow';
import {
  getActiveDecodeStep,
  type HybridDecodeSession,
} from './math/incrementalDecode';
import {
  advanceSpineWavePlayback,
  applyDecodeActionToActiveSession,
  createDualDecodeSessionStore,
  createIdleSpineWaveState,
  ensureTourArDecodeSession,
  gqaAccumTermBucket,
  gqaStepPlayDurationSec,
  mambaAccumTermBucket,
  mambaStepPlayDurationSec,
  moeAccumTermBucket,
  moeStepPlayDurationSec,
  resetTourOnlyDecodeSession,
  resolveActiveDecodeSession,
  resolveGqaAccumTermProgress,
  resolveMambaAccumTermProgress,
  resolveMoeAccumTermProgress,
  resolveMoeStepperCellForAccum,
  startSpineWavePlayback,
  toggleSpineWavePlaybackPause,
  type DecodeSessionAction,
  type DualDecodeSessionStore,
  type SpineWavePlaybackState,
} from './model/rcPlayback';
import {
  inferMambaCellHead,
  MAMBA_STEPPER_STEPS,
  type MambaCellRef,
} from './math/mambaSim';
import {
  advanceMambaTemporalPlaybackTick,
  remapMambaCellToken,
  resolveActiveMambaStateToken,
  resolveEffectiveMambaCell,
  resolveMambaHoverDuringPlayback,
} from './model/mambaPlayback';
import {
  inferMoeCellExpert,
  MOE_STEPPER_STEPS,
  type MoeCellRef,
} from './math/moeSim';
import {
  advanceMoeTokenCycleTick,
  remapMoeCellToken,
  resolveActiveMoeFocusToken,
  resolveEffectiveMoeCell,
  resolveMoeHoverDuringCycling,
  stepMoeFocusToken,
} from './model/moePlayback';
import { RenderDevice } from './render/device';
import { LinesPass } from './render/lines';
import {
  clientPointToRay,
  pickHitToGqaCell,
  pickHitToMambaCell,
  pickHitToMoeCell,
  pickMacroBay,
  type PickHit,
} from './render/pick';
import { FlatFlowRibbonsPass, RibbonsPass } from './render/ribbons';
import { SheetsPass } from './render/sheets';
import { SolidsPass } from './render/solids';
import {
  buildEmbeddingCutaway,
  buildGqaCutaway,
  buildMambaCutaway,
  buildMoECutaway,
  buildOutputCutaway,
  EMBEDDING_STEPPER_STEPS,
  fitSpatialEmbeddingCamera,
  fitSpatialGqaCamera,
  fitSpatialMambaCamera,
  fitSpatialMoeCamera,
  fitSpatialOutputCamera,
  OUTPUT_STEPPER_STEPS,
  parseMambaHeadPartId,
  parseMoeExpertPartId,
  parseOutputCandidatePartId,
  parseQueryHeadPartId,
  pickHitToEmbeddingCell,
  pickHitToOutputCell,
  type CutawaySceneGeometry,
  type EmbeddingCellRef,
  type OutputCellRef,
} from './scene/cutaways';
import {
  boundsForBay,
  boundsForSection,
  buildMacroSpine,
  SPINE_STAGE_BAYS_END,
  SPINE_STAGE_EMBED_END,
  SPINE_STAGE_NORM_END,
  TOWER_FIT_BOUNDS,
  type MacroSceneGeometry,
} from './scene/spine';
import {
  MacroControls,
  queryControlsElements,
  shouldDeferGlobalShortcut,
  type CrossBayNeighbor,
} from './ui/controls';
export { shouldDeferGlobalShortcut };
import { MemoryScalingDrawer } from './ui/memoryChart';
import { MacroOverlay } from './ui/overlay';
import {
  getWalkthroughBeat,
  nextBeatIndex,
  prevBeatIndex,
} from './story/walkthrough';

const DRAG_THRESHOLD_PX = 5;

function measureHostMetrics(host: HTMLElement): ViewportMetrics | null {
  const rect = host.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return null;
  return makeViewportMetrics(
    rect.left,
    rect.top,
    rect.width,
    rect.height,
    window.devicePixelRatio || 1,
  );
}

function showFallbackCard(
  el: HTMLElement,
  heading: string,
  detail: string,
  onOpenMemoryChart: () => void,
  canRetry: boolean,
): void {
  el.innerHTML = '';

  const card = document.createElement('div');
  card.className = 'webgl-fallback-card';

  const title = document.createElement('h2');
  title.className = 'webgl-fallback-title';
  title.textContent = heading;

  const body = document.createElement('p');
  body.className = 'webgl-fallback-body';
  body.textContent = detail;

  const summary = document.createElement('div');
  summary.className = 'webgl-fallback-summary';
  summary.textContent =
    'Nemotron 3 Nano 30B-A3B · 52 layers (23 Mamba-2 · 23 MoE 128+1 · 6 GQA) · 3.2B active / 31.6B total · 47.1 MiB fixed SSM + 6 KiB/tok BF16 KV';

  const actions = document.createElement('div');
  actions.className = 'webgl-fallback-actions';

  const chartBtn = document.createElement('button');
  chartBtn.type = 'button';
  chartBtn.className = 'ed-cta-btn';
  chartBtn.textContent = 'Open Memory Scaling (1 → 1M) ↗';
  chartBtn.addEventListener('click', onOpenMemoryChart);
  actions.appendChild(chartBtn);

  if (canRetry) {
    const reloadBtn = document.createElement('button');
    reloadBtn.type = 'button';
    reloadBtn.className = 'ed-cta-btn ed-cta-secondary';
    reloadBtn.textContent = 'Reload Page';
    reloadBtn.addEventListener('click', () => window.location.reload());
    actions.appendChild(reloadBtn);
  }

  card.append(title, body, summary, actions);
  el.appendChild(card);
  el.hidden = false;
}

function hideFallbackMessage(el: HTMLElement): void {
  el.hidden = true;
}

function cameraPoseForSection(
  section: TocSectionId,
  metrics: ViewportMetrics,
): CameraPose {
  const bounds = boundsForSection(section);
  switch (section) {
    case 'overview':
      return fitPerspectiveToBounds(
        bounds,
        MACRO_DEFAULT_AZIMUTH_DEG,
        MACRO_DEFAULT_ELEVATION_DEG,
        metrics.innerWidth,
        metrics.innerHeight,
        DEFAULT_FOV_DEG,
        0.84,
      );
    case 'embedding':
      return fitPerspectiveToBounds(
        bounds, 26, 18, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.78,
      );
    case 'mamba':
      return fitPerspectiveToBounds(
        bounds, 22, 16, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.76,
      );
    case 'moe':
      return fitPerspectiveToBounds(
        bounds, 24, 24, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.78,
      );
    case 'gqa':
      // Part A #1: Clean 3/4 front-left perspective angle matching MoE (02) & Feeder+Attn (42-43)
      return fitPerspectiveToBounds(
        bounds, 24, 24, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.78,
      );
    case 'kv_fp8':
      return fitPerspectiveToBounds(
        bounds, 30, 20, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.80,
      );
    case 'output':
      return fitPerspectiveToBounds(
        bounds, 26, 16, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.78,
      );
  }
}

function sameGqaCell(a: GqaCellRef | null, b: GqaCellRef | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.tensor === b.tensor &&
    a.row === b.row &&
    a.col === b.col &&
    a.headIndex === b.headIndex &&
    a.kvHeadIndex === b.kvHeadIndex &&
    a.tokenIndex === b.tokenIndex
  );
}

function sameMambaCell(a: MambaCellRef | null, b: MambaCellRef | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.tensor === b.tensor &&
    a.row === b.row &&
    a.col === b.col &&
    a.headIndex === b.headIndex &&
    a.groupIndex === b.groupIndex &&
    a.tapIndex === b.tapIndex &&
    a.tokenIndex === b.tokenIndex
  );
}

function sameMoeCell(a: MoeCellRef | null, b: MoeCellRef | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.tensor === b.tensor &&
    a.row === b.row &&
    a.col === b.col &&
    a.tokenIndex === b.tokenIndex &&
    a.expertId === b.expertId &&
    a.rankSlot === b.rankSlot
  );
}

function sameEmbeddingCell(a: EmbeddingCellRef | null, b: EmbeddingCellRef | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.tensor === b.tensor &&
    a.row === b.row &&
    a.col === b.col &&
    a.tokenIndex === b.tokenIndex &&
    a.vocabId === b.vocabId
  );
}

function sameOutputCell(a: OutputCellRef | null, b: OutputCellRef | null): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.tensor === b.tensor &&
    a.row === b.row &&
    a.col === b.col &&
    a.tokenIndex === b.tokenIndex &&
    a.vocabId === b.vocabId
  );
}

type CutawayWithOpFocusBounds = CutawaySceneGeometry & {
  readonly opFocusBounds?: Readonly<Record<number, AABB>>;
};

function resolveOpFocusBounds(
  scene: CutawaySceneGeometry | null,
  stepIndex: number,
): AABB | null {
  if (!scene) return null;
  const withOp = scene as CutawayWithOpFocusBounds;
  const opBounds = withOp.opFocusBounds?.[stepIndex];
  return opBounds ?? scene.fitBounds;
}

function cameraPoseForOpStep(
  _mode: Exclude<ViewMode, 'tower'>,
  opBounds: AABB,
  metrics: ViewportMetrics,
  sheetViewAngle: GqaViewAngle = 'three_quarter',
  insets?: { readonly topInsetPx: number; readonly bottomInsetPx: number },
): CameraPose {
  const isStackedNarrow = window.innerWidth <= 900 || metrics.innerWidth < 760;
  let azimuthDeg: number;
  let elevationDeg: number;
  if (sheetViewAngle === 'near_plan') {
    azimuthDeg = 1.5;
    elevationDeg = 78;
  } else {
    azimuthDeg = isStackedNarrow ? 17 : 20;
    elevationDeg = isStackedNarrow ? 25 : 23.5;
  }
  return fitOpBoundsCamera(opBounds, metrics, {
    azimuthDeg,
    elevationDeg,
    fovDeg: DEFAULT_FOV_DEG,
    topInsetPx: insets?.topInsetPx ?? (isStackedNarrow ? 12 : 84),
    bottomInsetPx: insets?.bottomInsetPx ?? (isStackedNarrow ? 12 : 118),
    fillFraction: 0.80,
    isNarrow: isStackedNarrow,
  });
}

function cameraPoseForCutaway(
  mode: Exclude<ViewMode, 'tower'>,
  bounds: AABB,
  metrics: ViewportMetrics,
  sheetViewAngle: GqaViewAngle = 'three_quarter',
  insets?: { readonly topInsetPx: number; readonly bottomInsetPx: number },
): CameraPose {
  const isStackedNarrow = window.innerWidth <= 900 || metrics.innerWidth < 760;

  // Spatial Increment 1, 2 & V4: 3D multi-plane tensor cutaways with exact projective framing
  // centered in the unobstructed canvas band.
  if (mode === 'cutaway_embedding') {
    return fitSpatialEmbeddingCamera(metrics, bounds, {
      topInsetPx: insets?.topInsetPx ?? (isStackedNarrow ? 12 : 84),
      bottomInsetPx: insets?.bottomInsetPx ?? (isStackedNarrow ? 12 : 118),
      isNarrow: isStackedNarrow,
    });
  }
  if (mode === 'cutaway_output') {
    return fitSpatialOutputCamera(metrics, bounds, {
      topInsetPx: insets?.topInsetPx ?? (isStackedNarrow ? 12 : 84),
      bottomInsetPx: insets?.bottomInsetPx ?? (isStackedNarrow ? 12 : 118),
      isNarrow: isStackedNarrow,
    });
  }
  if (mode === 'cutaway_gqa' && sheetViewAngle === 'three_quarter') {
    return fitSpatialGqaCamera(bounds, metrics, {
      topInsetPx: insets?.topInsetPx ?? (isStackedNarrow ? 12 : 84),
      bottomInsetPx: insets?.bottomInsetPx ?? (isStackedNarrow ? 12 : 118),
      isNarrow: isStackedNarrow,
    });
  }
  if (mode === 'cutaway_mamba' && sheetViewAngle === 'three_quarter') {
    return fitSpatialMambaCamera(bounds, metrics, {
      topInsetPx: insets?.topInsetPx ?? (isStackedNarrow ? 12 : 84),
      bottomInsetPx: insets?.bottomInsetPx ?? (isStackedNarrow ? 12 : 118),
      isNarrow: isStackedNarrow,
    });
  }
  if (mode === 'cutaway_moe' && sheetViewAngle === 'three_quarter') {
    return fitSpatialMoeCamera(bounds, metrics, {
      topInsetPx: insets?.topInsetPx ?? (isStackedNarrow ? 12 : 84),
      bottomInsetPx: insets?.bottomInsetPx ?? (isStackedNarrow ? 12 : 118),
      isNarrow: isStackedNarrow,
    });
  }

  // Preserved Near-Plan View for GQA, and existing XZ drafting-table schematics for Mamba-2 & MoE.
  if (sheetViewAngle === 'near_plan') {
    const pose = fitPerspectiveToBounds(
      bounds,
      1.5,
      78,
      metrics.innerWidth,
      metrics.innerHeight,
      DEFAULT_FOV_DEG,
      isStackedNarrow ? 0.78 : 0.77,
    );
    return {
      ...pose,
      target: [pose.target[0], pose.target[1], pose.target[2] + (isStackedNarrow ? 0.1 : 0.75)],
    };
  }
  const pose = fitPerspectiveToBounds(
    bounds,
    9,
    50,
    metrics.innerWidth,
    metrics.innerHeight,
    DEFAULT_FOV_DEG,
    isStackedNarrow ? 0.80 : 0.87,
  );
  return {
    ...pose,
    target: [pose.target[0], pose.target[1], pose.target[2] + (isStackedNarrow ? 0.1 : 0.55)],
  };
}

interface FlyAnimation {
  readonly from: CameraPose;
  readonly to: CameraPose;
  readonly startMs: number;
  readonly durationMs: number;
}

export type CutawayPulseGateState = Pick<
  SceneState,
  | 'mode'
  | 'reducedMotion'
  | 'gqaStepperActive'
  | 'gqaStepperPlaying'
  | 'mambaStepperActive'
  | 'mambaStepperPlaying'
  | 'moeStepperActive'
  | 'moeStepperPlaying'
  | 'mambaTemporalPlaying'
  | 'moeTokenCycling'
>;

/**
 * PR-11 idle RAF gate for cutaway dataflow pulse ribbons:
 * continuous pulse animation runs ONLY while a cutaway playback loop is actively playing
 * (never for static pinned cells, steady hover, or paused steppers).
 */
export function shouldAnimateCutawayPulse(state: CutawayPulseGateState): boolean {
  if (state.reducedMotion) return false;
  if (state.mode === 'cutaway_gqa') {
    return state.gqaStepperActive && state.gqaStepperPlaying;
  }
  if (state.mode === 'cutaway_mamba') {
    return (
      (state.mambaStepperActive && state.mambaStepperPlaying) ||
      state.mambaTemporalPlaying
    );
  }
  if (state.mode === 'cutaway_moe') {
    return (
      (state.moeStepperActive && state.moeStepperPlaying) ||
      state.moeTokenCycling
    );
  }
  return false;
}

export interface RafContinuationInput {
  readonly state: CutawayPulseGateState;
  readonly flyAnimActive: boolean;
  readonly spineWaveActive: boolean;
  readonly spineWavePaused: boolean;
  readonly contextLost?: boolean;
}

/**
 * PR-11/PR-10 global continuous-RAF predicate: true ONLY when camera fly, active Tower
 * spine replay, or active cutaway playback is genuinely advancing frames (and WebGL is live).
 */
export function shouldContinueRafLoop(input: RafContinuationInput): boolean {
  if (input.contextLost) return false;
  if (input.flyAnimActive) return true;
  if (
    input.state.mode === 'tower' &&
    input.spineWaveActive &&
    !input.spineWavePaused &&
    !input.state.reducedMotion
  ) {
    return true;
  }
  return shouldAnimateCutawayPulse(input.state);
}

export interface ValueAtlasKeyInput {
  readonly mode: ViewMode;
  readonly bayId: number;
  readonly executionMode: ExecutionMode;
  readonly executionId: string;
  readonly tokenCount: number;
  readonly precision: PrecisionMode;
  readonly gqaGroup0Head: number;
  readonly gqaGroup1Head: number;
  readonly mambaActiveHead: number;
  readonly mambaStateToken: number;
  readonly moeFocusToken: number;
  readonly moeFocusExpert: number;
  readonly outputFocusToken: number;
  readonly stepperActive: boolean;
  readonly stepperStep: number;
  readonly stepperTermBucket: number;
  readonly reducedMotion: boolean;
  readonly atlasWidth: number;
  readonly atlasHeight: number;
  readonly atlasByteLength: number;
}

/**
 * PR-12 deterministic RG32F value-atlas cache key covering every input that affects atlas texels
 * or mask bits while remaining invariant across cell-hover moves within the same head/expert/token.
 */
export function computeValueAtlasContentKey(input: ValueAtlasKeyInput): string {
  const modeSpecific =
    input.mode === 'cutaway_gqa'
      ? `g0=${input.gqaGroup0Head}:g1=${input.gqaGroup1Head}`
      : input.mode === 'cutaway_mamba'
        ? `mh=${input.mambaActiveHead}:mt=${input.mambaStateToken}`
        : input.mode === 'cutaway_moe'
          ? `et=${input.moeFocusToken}:ee=${input.moeFocusExpert}`
          : input.mode === 'cutaway_output'
            ? `ot=${input.outputFocusToken}`
            : 'emb';
  return [
    input.mode,
    input.bayId,
    input.executionMode,
    input.executionId,
    input.tokenCount,
    input.precision,
    modeSpecific,
    input.stepperActive ? `s1:${input.stepperStep}:b${input.stepperTermBucket}` : 's0',
    input.reducedMotion ? 'rm1' : 'rm0',
    `${input.atlasWidth}x${input.atlasHeight}:${input.atlasByteLength}`,
  ].join('|');
}

/**
 * PR-14 discrete Tower spine-wave geometry key: changes only on stage/bay transitions
 * (Embedding -> Bays 01..52 -> Final RMSNorm -> Output -> idle; ~55 transitions total
 * across a 330-frame 5.5s instructional replay).
 */
export function computeSpineWaveGeometryKey(
  progress: number | null,
  mode: SpineWaveMode,
  bayCount = NEMOTRON_SPEC.bays.length,
): string {
  if (progress === null || !Number.isFinite(progress)) {
    return 'idle';
  }
  const p = Math.max(0, Math.min(1, progress));
  if (p < SPINE_STAGE_EMBED_END) {
    return `${mode}:embedding:none`;
  }
  if (p >= SPINE_STAGE_NORM_END) {
    return `${mode}:output:none`;
  }
  if (p >= SPINE_STAGE_BAYS_END) {
    return `${mode}:final_norm:none`;
  }
  const span = SPINE_STAGE_BAYS_END - SPINE_STAGE_EMBED_END;
  const bayIdx = Math.min(
    bayCount - 1,
    Math.max(0, Math.floor(((p - SPINE_STAGE_EMBED_END) / span) * bayCount)),
  );
  return `${mode}:bay:${bayIdx + 1}`;
}

function boot(): void {
  const host = document.getElementById('stage-host');
  const canvas = document.getElementById('gl-canvas') as HTMLCanvasElement | null;
  const overlaySvg = document.getElementById('overlay-svg') as SVGSVGElement | null;
  const overlayLabels = document.getElementById('overlay-labels') as HTMLDivElement | null;
  const fallbackEl = document.getElementById('webgl-fallback');
  const memoryChartHost = document.getElementById('memory-chart-drawer');

  if (!host || !canvas || !overlaySvg || !overlayLabels || !fallbackEl || !memoryChartHost) {
    throw new Error('Required stage DOM elements are missing');
  }

  const initialMetrics =
    measureHostMetrics(host) ?? makeViewportMetrics(0, 0, 960, 700, window.devicePixelRatio || 1);
  const initialPose = cameraPoseForSection('overview', initialMetrics);
  const reducedMotionMq = window.matchMedia?.('(prefers-reduced-motion: reduce)') ?? null;
  const reducedMotion = reducedMotionMq?.matches ?? false;
  const state: SceneState = createInitialState(initialPose, reducedMotion);

  let currentMetrics: ViewportMetrics = initialMetrics;
  let currentTowerScene: MacroSceneGeometry | null = null;
  let currentCutawayScene: CutawaySceneGeometry | null = null;
  let currentSnapshot: CameraSnapshot | null = null;
  let geometryDirty = true;
  let pendingRafId: number | null = null;
  let pendingHoverRafId: number | null = null;
  let latestHoverClientX = 0;
  let latestHoverClientY = 0;
  let lastUploadedAtlasKey = '';
  let lastOverlayRenderKey = '';
  let lastChromeKey = '';
  let sceneBuildVersion = 0;
  let lastSpineWaveGeoKey = 'idle';
  let cachedCutawayInsets: { readonly topInsetPx: number; readonly bottomInsetPx: number } | null =
    null;
  let flyAnim: FlyAnimation | null = null;
  let lastStepperTickMs: number | null = null;
  let lastMambaTemporalTickMs: number | null = null;
  let lastMoeCycleTickMs: number | null = null;
  let lastStepperCellKey = '';
  let manualCameraOverride = false;
  let lastAutoFollowedStepKey = '';
  let savedGqaSpatialPose: CameraPose | null = null;
  let savedGqaPlanPose: CameraPose | null = null;
  let savedMambaSpatialPose: CameraPose | null = null;
  let savedMambaPlanPose: CameraPose | null = null;
  let savedMoeSpatialPose: CameraPose | null = null;
  let savedMoePlanPose: CameraPose | null = null;
  let embeddingHoveredCell: EmbeddingCellRef | null = null;
  let outputHoveredCell: OutputCellRef | null = null;
  let bookendStepIndex: number | null = null;
  let spineWaveClock: SpineWavePlaybackState = createIdleSpineWaveState('fast');
  let spineWaveLastTickMs: number | null = null;

  const cancelPendingHover = () => {
    if (pendingHoverRafId !== null) {
      window.cancelAnimationFrame(pendingHoverRafId);
      pendingHoverRafId = null;
    }
  };

  const invalidateCutawayInsets = () => {
    cachedCutawayInsets = null;
    lastOverlayRenderKey = '';
  };

  const applySpineWaveStateToScene = (wave: SpineWavePlaybackState) => {
    spineWaveClock = wave;
    state.spineWaveProgress = wave.progress;
    state.spineWaveMode = wave.mode;
    state.spineWavePaused = wave.paused;
  };

  const clearSpineWave = (keepMode = true) => {
    spineWaveLastTickMs = null;
    lastSpineWaveGeoKey = 'idle';
    applySpineWaveStateToScene(createIdleSpineWaveState(keepMode ? state.spineWaveMode : 'fast'));
  };

  let device: RenderDevice | null = null;
  let solidsPass: SolidsPass | null = null;
  let sheetsPass: SheetsPass | null = null;
  let linesPass: LinesPass | null = null;
  let ribbonsPass: RibbonsPass | null = null;
  let flatFlowPass: FlatFlowRibbonsPass | null = null;

  const openMemoryChartFromFallback = () => {
    state.memoryChartOpen = true;
    state.shortcutsOpen = false;
    syncUi();
  };

  try {
    device = new RenderDevice(canvas);
    solidsPass = new SolidsPass(device.gl);
    sheetsPass = new SheetsPass(device.gl);
    linesPass = new LinesPass(device.gl);
    ribbonsPass = new RibbonsPass(device.gl);
    flatFlowPass = new FlatFlowRibbonsPass(device.gl);
  } catch (err) {
    const detail = err instanceof Error ? err.message : 'WebGL2 unavailable';
    showFallbackCard(
      fallbackEl,
      '3D WebGL2 Canvas Unavailable',
      `Your browser could not initialize a WebGL2 context (${detail}). The 22-beat architectural walkthrough, token scrubber, precision toggle, and interactive 1 → 1M memory scaling chart remain fully available.`,
      openMemoryChartFromFallback,
      true,
    );
  }

  const overlay = new MacroOverlay(overlaySvg, overlayLabels);
  const controlEls = queryControlsElements();

  const measureCutawayInsets = (_mode: Exclude<ViewMode, 'tower'>) => {
    if (cachedCutawayInsets !== null) {
      return cachedCutawayInsets;
    }
    const isMobileStack = window.innerWidth <= 900;
    const toolbarH = !isMobileStack ? controlEls.floatingToolbar.offsetHeight || 68 : 0;
    const topInsetPx = isMobileStack ? 12 : Math.max(64, toolbarH + 16);
    const usesMathDock = true;
    const dockH =
      usesMathDock && !controlEls.gqaMathDock.hidden
        ? controlEls.gqaMathDock.offsetHeight || 104
        : 0;
    const bottomInsetPx = isMobileStack
      ? 12
      : usesMathDock
        ? Math.max(106, dockH + 18)
        : 30;
    cachedCutawayInsets = { topInsetPx, bottomInsetPx };
    return cachedCutawayInsets;
  };

  const cancelFly = () => {
    flyAnim = null;
  };

  /** Non-destructive tour exploration marker (V5 Audit S4): never destroys tourActive. */
  const markTourExploring = (): boolean => {
    if (state.tourActive && !state.tourExploring) {
      state.tourExploring = true;
      return true;
    }
    return false;
  };

  const markManualCameraControl = () => {
    cancelFly();
    const enteredExploring = markTourExploring();
    if (!manualCameraOverride || enteredExploring) {
      manualCameraOverride = true;
      syncUi();
    }
  };

  const flyToPose = (targetPose: CameraPose) => {
    if (state.reducedMotion) {
      flyAnim = null;
      state.camera = cloneCameraPose(targetPose);
      scheduleDraw();
      return;
    }
    flyAnim = {
      from: cloneCameraPose(state.camera),
      to: cloneCameraPose(targetPose),
      startMs: performance.now(),
      durationMs: CAMERA_FLY_DURATION_MS,
    };
    scheduleDraw();
  };

  const applyHeadSelectionToGroupHeads = (partId: string | null) => {
    const gqaHead = parseQueryHeadPartId(partId);
    if (gqaHead !== null) {
      if (gqaHead < 16) state.gqaActiveGroup0Head = gqaHead;
      else state.gqaActiveGroup1Head = gqaHead;
      return;
    }
    const mambaHead = parseMambaHeadPartId(partId);
    if (mambaHead !== null) {
      state.mambaActiveHead = mambaHead;
      return;
    }
    const moeExp = parseMoeExpertPartId(partId);
    if (moeExp !== null) {
      state.moeActiveExpert = moeExp;
    }
  };

  const activeViewAngleForMode = (mode: Exclude<ViewMode, 'tower'>): GqaViewAngle =>
    mode === 'cutaway_embedding' || mode === 'cutaway_output'
      ? 'three_quarter'
      : mode === 'cutaway_mamba'
        ? state.mambaViewAngle
        : mode === 'cutaway_moe'
          ? state.moeViewAngle
          : state.gqaViewAngle;

  // Single authoritative unified 52-layer miniature model + bounded execution cache + isolated Tour/Free-Explore AR sessions
  const hybridModel = createDefaultHybridModel();
  const hybridCache = new HybridExecutionCache(16);
  let decodeSessions: DualDecodeSessionStore = createDualDecodeSessionStore(
    hybridModel,
    state.precision,
  );

  const activeDecodeSession = (): HybridDecodeSession => {
    const resolved = resolveActiveDecodeSession(
      decodeSessions,
      state.tourActive,
      state.tourBeat,
      hybridModel,
      state.precision,
    );
    decodeSessions = resolved.store;
    return resolved.session;
  };

  const activeDecodeSessionOrNull = (): HybridDecodeSession | null =>
    state.executionMode === 'ar_decode' ? activeDecodeSession() : null;

  const activeHybridExecution = (): HybridExecutionResult =>
    getOrRunHybridExecution(hybridModel, hybridCache, {
      tokenIds: defaultTokenIdsForLength(state.tokens),
    });

  const activeCutawayTokens = (): number =>
    state.executionMode === 'ar_decode'
      ? getActiveDecodeStep(activeDecodeSession()).tokens
      : state.tokens;

  const activeMambaSim = () =>
    state.executionMode === 'ar_decode'
      ? getActiveDecodeStep(activeDecodeSession()).getMambaTrace(state.mambaBayId)
      : activeHybridExecution().getMambaTrace(state.mambaBayId);

  const activeMoeSim = () =>
    state.executionMode === 'ar_decode'
      ? getActiveDecodeStep(activeDecodeSession()).getMoeTrace(state.moeBayId)
      : activeHybridExecution().getMoeTrace(state.moeBayId);

  const activeGqaSim = () =>
    state.executionMode === 'ar_decode'
      ? getActiveDecodeStep(activeDecodeSession()).getGqaTrace(state.gqaBayId)
      : activeHybridExecution().getGqaTrace(state.gqaBayId);

  const activeMoeFocusToken = (): number =>
    resolveActiveMoeFocusToken(state.moeFocusToken, activeCutawayTokens());

  const activeMoeFocusExpert = (sim = activeMoeSim(), tok = activeMoeFocusToken()): number => {
    if (state.moeActiveExpert !== null) {
      return Math.max(0, Math.min(sim.nRoutedExperts - 1, state.moeActiveExpert));
    }
    return sim.topk_indices[tok * sim.topK];
  };

  const activeStepperHead = (): number => {
    const sel = parseQueryHeadPartId(state.selectedPart);
    return sel !== null ? sel : state.gqaActiveGroup0Head;
  };

  const resolveActiveGqaCell = (): GqaCellRef | null => {
    if (state.mode !== 'cutaway_gqa') return null;
    if (state.gqaHoveredCell) return state.gqaHoveredCell;
    if (state.gqaSelectedCell) return state.gqaSelectedCell;
    if (state.gqaStepperActive) {
      const sim = activeGqaSim();
      return stepperCellForState(
        sim,
        activeStepperHead(),
        state.gqaStepperStep,
        state.gqaStepperSubProgress,
      );
    }
    return null;
  };

  const activeMambaStateToken = (): number =>
    resolveActiveMambaStateToken(state.mambaStateToken, activeCutawayTokens());

  const resolveActiveMambaCell = (): MambaCellRef | null => {
    if (state.mode !== 'cutaway_mamba') return null;
    const sim = activeMambaSim();
    return resolveEffectiveMambaCell({
      sim,
      tokens: sim.tokens,
      activeHead: state.mambaActiveHead,
      mambaStateToken: state.mambaStateToken,
      mambaTemporalPlaying: state.mambaTemporalPlaying,
      mambaHoveredCell: state.mambaHoveredCell,
      mambaSelectedCell: state.mambaSelectedCell,
      mambaStepperActive: state.mambaStepperActive,
      mambaStepperStep: state.mambaStepperStep,
      mambaStepperSubProgress: state.mambaStepperSubProgress,
    });
  };

  const resolveActiveMoeCell = (): MoeCellRef | null => {
    if (state.mode !== 'cutaway_moe') return null;
    const sim = activeMoeSim();
    if (state.moeStepperActive && !state.moeHoveredCell && !state.moeSelectedCell) {
      const tok = resolveActiveMoeFocusToken(state.moeFocusToken, sim.tokens);
      const exp = activeMoeFocusExpert(sim, tok);
      return resolveMoeStepperCellForAccum(
        sim,
        state.moeStepperStep,
        state.moeStepperSubProgress,
        tok,
        exp,
      );
    }
    return resolveEffectiveMoeCell({
      sim,
      moeFocusToken: state.moeFocusToken,
      moeActiveExpert: state.moeActiveExpert,
      moeTokenCycling: state.moeTokenCycling,
      moeHoveredCell: state.moeHoveredCell,
      moeSelectedCell: state.moeSelectedCell,
      moeStepperActive: state.moeStepperActive,
      moeStepperStep: state.moeStepperStep,
      moeStepperSubProgress: state.moeStepperSubProgress,
    });
  };

  const activeEmbeddingPayload = () => {
    if (state.executionMode === 'ar_decode') {
      const step = getActiveDecodeStep(activeDecodeSession());
      return {
        embedding: { X: step.X_embed },
        tokenIds: step.newTokenIds,
        tokenLabels: step.newTokenLabels,
        startTokenOffset: step.prefixTokens,
      };
    }
    const exec = activeHybridExecution();
    return {
      embedding: { X: exec.X_embed },
      tokenIds: exec.tokenIds,
      tokenLabels: exec.tokenLabels,
      startTokenOffset: 0,
    };
  };

  const activeOutputPayload = () => {
    if (state.executionMode === 'ar_decode') {
      const step = getActiveDecodeStep(activeDecodeSession());
      return {
        residualIn: step.residualStreams[step.residualStreams.length - 1],
        finalNorm: step.finalNorm,
        lmHead: step.lmHead,
        tokenLabels: step.newTokenLabels,
        startTokenOffset: step.prefixTokens,
        isArDecode: true,
      };
    }
    const exec = activeHybridExecution();
    return {
      residualIn: exec.residualStreams[exec.residualStreams.length - 1],
      finalNorm: exec.finalNorm,
      lmHead: exec.lmHead,
      tokenLabels: exec.tokenLabels,
      startTokenOffset: 0,
      isArDecode: false,
    };
  };

  const clampedEmbeddingFocusToken = (tokenCount: number): number =>
    Math.max(0, Math.min(Math.max(0, tokenCount - 1), state.embeddingFocusToken));

  const clampedOutputFocusToken = (tokenCount: number): number =>
    Math.max(0, Math.min(Math.max(0, tokenCount - 1), state.outputFocusToken));

  const buildActiveCutaway = (): CutawaySceneGeometry | null => {
    if (state.mode === 'cutaway_embedding') {
      const payload = activeEmbeddingPayload();
      const focusToken = clampedEmbeddingFocusToken(payload.tokenIds.length);
      return buildEmbeddingCutaway({
        model: hybridModel,
        embedding: payload.embedding,
        tokenIds: payload.tokenIds,
        tokenLabels: payload.tokenLabels,
        startTokenOffset: payload.startTokenOffset,
        focusToken,
        precision: state.precision,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
        activeCell: embeddingHoveredCell ?? state.embeddingActiveCell,
      });
    }
    if (state.mode === 'cutaway_output') {
      const payload = activeOutputPayload();
      const focusToken = clampedOutputFocusToken(payload.tokenLabels.length);
      return buildOutputCutaway({
        model: hybridModel,
        residualIn: payload.residualIn,
        finalNorm: payload.finalNorm,
        lmHead: payload.lmHead,
        tokenLabels: payload.tokenLabels,
        startTokenOffset: payload.startTokenOffset,
        focusToken,
        precision: state.precision,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
        activeCell: outputHoveredCell ?? state.outputActiveCell,
        isArDecode: payload.isArDecode,
        selectedCandidateTokenId: parseOutputCandidatePartId(state.selectedPart),
      });
    }
    if (state.mode === 'cutaway_mamba') {
      const sim = activeMambaSim();
      const termProgress = resolveMambaAccumTermProgress(
        sim,
        state.mambaStepperStep,
        state.mambaStepperSubProgress,
        state.mambaStepperActive,
        state.reducedMotion,
        state.mambaHoveredCell !== null || state.mambaSelectedCell !== null,
      );
      state.mambaAccumTermProgress = termProgress;
      return buildMambaCutaway({
        bayId: state.mambaBayId,
        tokens: sim.tokens,
        precision: state.precision,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
        activeHead: state.mambaActiveHead,
        activeCell: resolveActiveMambaCell(),
        stepperStep: state.mambaStepperActive ? state.mambaStepperStep : null,
        stepperActive: state.mambaStepperActive,
        stepperSubProgress: state.mambaStepperSubProgress,
        termProgress,
        reducedMotion: state.reducedMotion,
        viewAngle: state.mambaViewAngle,
        stateTokenIndex: state.mambaStateToken,
        simOverride: sim,
      });
    }
    if (state.mode === 'cutaway_moe') {
      const sim = activeMoeSim();
      const tok = activeMoeFocusToken();
      const termProgress = resolveMoeAccumTermProgress(
        sim,
        state.moeStepperStep,
        state.moeStepperSubProgress,
        state.moeStepperActive,
        state.reducedMotion,
        state.moeHoveredCell !== null || state.moeSelectedCell !== null,
      );
      state.moeAccumTermProgress = termProgress;
      return buildMoECutaway({
        bayId: state.moeBayId,
        tokens: sim.tokens,
        precision: state.precision,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
        focusToken: tok,
        activeExpert: state.moeActiveExpert ?? undefined,
        activeCell: resolveActiveMoeCell(),
        stepperStep: state.moeStepperActive ? state.moeStepperStep : null,
        stepperActive: state.moeStepperActive,
        stepperSubProgress: state.moeStepperSubProgress,
        termProgress,
        reducedMotion: state.reducedMotion,
        viewAngle: state.moeViewAngle,
        simOverride: sim,
      });
    }
    if (state.mode === 'cutaway_gqa') {
      const sim = activeGqaSim();
      const termProgress = resolveGqaAccumTermProgress(
        sim,
        state.gqaStepperStep,
        state.gqaStepperSubProgress,
        state.gqaStepperActive,
        state.reducedMotion,
        state.gqaHoveredCell !== null || state.gqaSelectedCell !== null,
      );
      state.gqaAccumTermProgress = termProgress;
      return buildGqaCutaway({
        bayId: state.gqaBayId,
        tokens: sim.tokens,
        precision: state.precision,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
        activeGroup0Head: state.gqaActiveGroup0Head,
        activeGroup1Head: state.gqaActiveGroup1Head,
        activeCell: resolveActiveGqaCell(),
        stepperStep: state.gqaStepperActive ? state.gqaStepperStep : null,
        stepperActive: state.gqaStepperActive,
        stepperPulsePhase: state.gqaStepperSubProgress,
        termProgress,
        reducedMotion: state.reducedMotion,
        viewAngle: state.gqaViewAngle,
        simOverride: sim,
      });
    }
    return null;
  };

  const currentExecutionIdentity = (): string =>
    state.executionMode === 'ar_decode'
      ? getActiveDecodeStep(activeDecodeSession()).stepExecutionId
      : activeHybridExecution().executionId;

  const currentAtlasKeyForCutaway = (cutaway: CutawaySceneGeometry): string => {
    const atlas = cutaway.valueAtlas;
    if (!atlas) return '';
    const bayId =
      state.mode === 'cutaway_mamba'
        ? state.mambaBayId
        : state.mode === 'cutaway_moe'
          ? state.moeBayId
          : state.mode === 'cutaway_gqa'
            ? state.gqaBayId
            : state.mode === 'cutaway_output'
              ? NEMOTRON_SPEC.bays.length + 1
              : 0;
    const tokenCount = activeCutawayTokens();
    const stepperActive =
      state.mode === 'cutaway_gqa'
        ? state.gqaStepperActive
        : state.mode === 'cutaway_mamba'
          ? state.mambaStepperActive
          : state.mode === 'cutaway_moe'
            ? state.moeStepperActive
            : false;
    const stepperStep =
      state.mode === 'cutaway_gqa'
        ? state.gqaStepperStep
        : state.mode === 'cutaway_mamba'
          ? state.mambaStepperStep
          : state.mode === 'cutaway_moe'
            ? state.moeStepperStep
            : 0;
    const stepperTermBucket =
      state.mode === 'cutaway_gqa' && cutaway.gqaSim
        ? gqaAccumTermBucket(cutaway.gqaSim, state.gqaStepperStep, state.gqaAccumTermProgress)
        : state.mode === 'cutaway_mamba' && cutaway.mambaSim
          ? mambaAccumTermBucket(cutaway.mambaSim, state.mambaStepperStep, state.mambaAccumTermProgress)
          : state.mode === 'cutaway_moe' && cutaway.moeSim
            ? moeAccumTermBucket(cutaway.moeSim, state.moeStepperStep, state.moeAccumTermProgress)
            : -1;
    return computeValueAtlasContentKey({
      mode: state.mode,
      bayId,
      executionMode: state.executionMode,
      executionId: currentExecutionIdentity(),
      tokenCount,
      precision: state.precision,
      gqaGroup0Head: cutaway.gqaStats?.activeGroup0Head ?? state.gqaActiveGroup0Head,
      gqaGroup1Head: cutaway.gqaStats?.activeGroup1Head ?? state.gqaActiveGroup1Head,
      mambaActiveHead: cutaway.mambaStats?.activeHead ?? state.mambaActiveHead,
      mambaStateToken: cutaway.mambaStats?.stateTokenIndex ?? activeMambaStateToken(),
      moeFocusToken: cutaway.moeStats?.focusToken ?? activeMoeFocusToken(),
      moeFocusExpert:
        cutaway.moeStats?.focusExpert ??
        (state.mode === 'cutaway_moe' ? activeMoeFocusExpert(cutaway.moeSim ?? activeMoeSim()) : 0),
      outputFocusToken: clampedOutputFocusToken(tokenCount),
      stepperActive,
      stepperStep,
      stepperTermBucket,
      reducedMotion: state.reducedMotion,
      atlasWidth: atlas.width,
      atlasHeight: atlas.height,
      atlasByteLength: atlas.data.byteLength,
    });
  };

  const rebuildSceneIfNeeded = () => {
    if (!geometryDirty && ((state.mode === 'tower' && currentTowerScene) || (state.mode !== 'tower' && currentCutawayScene))) {
      return;
    }
    if (state.mode === 'tower') {
      currentTowerScene = buildMacroSpine({
        spec: NEMOTRON_SPEC,
        tokens: state.tokens,
        precision: state.precision,
        selectedBay: state.selectedBay,
        hoveredBay: state.hoveredBay,
        propagationWave:
          state.spineWaveProgress !== null
            ? {
                progress: state.spineWaveProgress,
                activeTokenIndex: Math.max(0, state.tokens - 1),
                mode: state.spineWaveMode,
              }
            : null,
      });
      currentCutawayScene = null;
      lastUploadedAtlasKey = '';
      sceneBuildVersion++;
      if (device && !device.isLost && solidsPass && linesPass && ribbonsPass) {
        solidsPass.upload(currentTowerScene.solidData, currentTowerScene.solidCount);
        linesPass.upload(currentTowerScene.lineData, currentTowerScene.lineCount);
        ribbonsPass.upload(currentTowerScene.ribbonData, currentTowerScene.ribbonCount);
      }
    } else {
      currentCutawayScene = buildActiveCutaway();
      sceneBuildVersion++;
      if (
        currentCutawayScene &&
        device &&
        !device.isLost &&
        solidsPass &&
        sheetsPass &&
        linesPass &&
        ribbonsPass &&
        flatFlowPass
      ) {
        if (currentCutawayScene.sheetData && currentCutawayScene.sheetCount && currentCutawayScene.sheetCount > 0) {
          if (currentCutawayScene.valueAtlas) {
            const atlasKey = currentAtlasKeyForCutaway(currentCutawayScene);
            if (atlasKey !== lastUploadedAtlasKey) {
              sheetsPass.uploadValueAtlas(
                currentCutawayScene.valueAtlas.width,
                currentCutawayScene.valueAtlas.height,
                currentCutawayScene.valueAtlas.data,
              );
              lastUploadedAtlasKey = atlasKey;
            }
          }
          sheetsPass.upload(
            currentCutawayScene.sheetData,
            currentCutawayScene.sheetCount,
            currentCutawayScene.sheetBasisData,
            currentCutawayScene.sheetRevealData,
          );
          linesPass.upload(currentCutawayScene.lineData, currentCutawayScene.lineCount);
          flatFlowPass.upload(
            currentCutawayScene.flowVertices ?? new Float32Array(0),
            currentCutawayScene.flowIndices ?? new Uint32Array(0),
          );
        } else {
          lastUploadedAtlasKey = '';
          solidsPass.upload(currentCutawayScene.solidData, currentCutawayScene.solidCount);
          linesPass.upload(currentCutawayScene.lineData, currentCutawayScene.lineCount);
          ribbonsPass.upload(currentCutawayScene.ribbonData, currentCutawayScene.ribbonCount);
        }
      }
    }
    geometryDirty = false;
  };

  const startSpineWave = (mode: SpineWaveMode = 'fast') => {
    if (state.mode !== 'tower') return;
    const started = startSpineWavePlayback(mode, state.reducedMotion);
    applySpineWaveStateToScene(started);
    lastSpineWaveGeoKey = computeSpineWaveGeometryKey(started.progress, started.mode);
    spineWaveLastTickMs = state.reducedMotion ? null : performance.now();
    geometryDirty = true;
    rebuildSceneIfNeeded();
    syncUi();
    scheduleDraw();
  };

  const syncUi = () => {
    controls.sync(
      state,
      currentCutawayScene,
      manualCameraOverride,
      activeHybridExecution(),
      activeDecodeSessionOrNull(),
    );
    memoryDrawer.sync(state.memoryChartOpen, state.precision, state.memoryProbeTokens);
    if (state.mode !== 'tower') {
      invalidateCutawayInsets();
    }
  };

  const getActiveCutawayStepper = (): { active: boolean; playing: boolean; step: number } | null => {
    if (state.mode === 'cutaway_mamba') {
      return {
        active: state.mambaStepperActive,
        playing: state.mambaStepperPlaying,
        step: state.mambaStepperStep,
      };
    }
    if (state.mode === 'cutaway_moe') {
      return {
        active: state.moeStepperActive,
        playing: state.moeStepperPlaying,
        step: state.moeStepperStep,
      };
    }
    if (state.mode === 'cutaway_gqa') {
      return {
        active: state.gqaStepperActive,
        playing: state.gqaStepperPlaying,
        step: state.gqaStepperStep,
      };
    }
    return null;
  };

  const focusCameraOnActiveOpStep = (force = false) => {
    if (state.mode === 'tower' || !currentCutawayScene) return;
    if (!state.stepperAutoFollow) return;
    if (manualCameraOverride && !force) return;
    const stepper = getActiveCutawayStepper();
    if (!stepper || !stepper.active) return;

    const viewAngle = activeViewAngleForMode(state.mode);
    const stepKey = `${state.mode}:${viewAngle}:${stepper.step}`;
    if (!force && stepKey === lastAutoFollowedStepKey) return;

    const opBounds = resolveOpFocusBounds(currentCutawayScene, stepper.step);
    if (!opBounds) return;
    lastAutoFollowedStepKey = stepKey;
    flyToPose(
      cameraPoseForOpStep(
        state.mode,
        opBounds,
        currentMetrics,
        viewAngle,
        measureCutawayInsets(state.mode),
      ),
    );
  };

  const applyBookendStepFocus = (step: number) => {
    if (state.mode === 'cutaway_embedding') {
      const clamped = Math.max(0, Math.min(EMBEDDING_STEPPER_STEPS.length - 1, step));
      bookendStepIndex = clamped;
      const payload = activeEmbeddingPayload();
      const tok = clampedEmbeddingFocusToken(payload.tokenIds.length);
      const vid = payload.tokenIds[tok] ?? 0;
      embeddingHoveredCell = null;
      if (clamped === 0) {
        state.embeddingActiveCell = { tensor: 'token_strip', row: tok, col: 0, tokenIndex: tok, vocabId: vid };
      } else if (clamped === 1) {
        state.embeddingActiveCell = { tensor: 'w_embed', row: vid, col: 0, tokenIndex: tok, vocabId: vid };
      } else {
        state.embeddingActiveCell = { tensor: 'x_embed', row: 0, col: tok, tokenIndex: tok, vocabId: vid };
      }
      manualCameraOverride = false;
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      const opBounds = resolveOpFocusBounds(currentCutawayScene, clamped);
      if (opBounds && state.stepperAutoFollow) {
        flyToPose(
          cameraPoseForOpStep(
            'cutaway_embedding',
            opBounds,
            currentMetrics,
            'three_quarter',
            measureCutawayInsets('cutaway_embedding'),
          ),
        );
      }
      scheduleDraw();
      return;
    }
    if (state.mode === 'cutaway_output') {
      const clamped = Math.max(0, Math.min(OUTPUT_STEPPER_STEPS.length - 1, step));
      bookendStepIndex = clamped;
      const payload = activeOutputPayload();
      const tok = clampedOutputFocusToken(payload.tokenLabels.length);
      const topCand = payload.lmHead.topKByToken[tok]?.[0]?.tokenId ?? 0;
      outputHoveredCell = null;
      if (clamped === 0) {
        state.outputActiveCell = { tensor: 'residual_in', row: tok, col: 0, tokenIndex: tok };
      } else if (clamped === 1) {
        state.outputActiveCell = { tensor: 'final_norm', row: tok, col: 0, tokenIndex: tok };
      } else if (clamped === 2) {
        state.outputActiveCell = { tensor: 'w_lm_head', row: topCand, col: 0, vocabId: topCand };
      } else if (clamped === 3) {
        state.outputActiveCell = { tensor: 'logits', row: topCand, col: 0, tokenIndex: tok, vocabId: topCand };
      } else {
        state.outputActiveCell = { tensor: 'probs', row: topCand, col: 0, tokenIndex: tok, vocabId: topCand };
      }
      manualCameraOverride = false;
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      const opBounds = resolveOpFocusBounds(currentCutawayScene, clamped);
      if (opBounds && state.stepperAutoFollow) {
        flyToPose(
          cameraPoseForOpStep(
            'cutaway_output',
            opBounds,
            currentMetrics,
            'three_quarter',
            measureCutawayInsets('cutaway_output'),
          ),
        );
      }
      scheduleDraw();
    }
  };

  const toggleCutawayStepperPlay = (fromTourBar = false) => {
    if (state.mode === 'tower') return;
    if (!fromTourBar) markTourExploring();

    if (state.mode === 'cutaway_embedding') {
      const next = ((bookendStepIndex ?? -1) + 1) % EMBEDDING_STEPPER_STEPS.length;
      applyBookendStepFocus(next);
      return;
    }
    if (state.mode === 'cutaway_output') {
      const next = ((bookendStepIndex ?? -1) + 1) % OUTPUT_STEPPER_STEPS.length;
      applyBookendStepFocus(next);
      return;
    }

    if (state.mode === 'cutaway_mamba') {
      state.mambaSelectedCell = null;
      state.mambaHoveredCell = null;
      state.mambaTemporalPlaying = false;
      lastMambaTemporalTickMs = null;
      if (state.reducedMotion) {
        state.mambaStepperActive = true;
        state.mambaStepperPlaying = false;
        state.mambaStepperStep = (state.mambaStepperStep + 1) % MAMBA_STEPPER_STEPS.length;
        state.mambaStepperSubProgress = 0;
        manualCameraOverride = false;
        geometryDirty = true;
        rebuildSceneIfNeeded();
        syncUi();
        focusCameraOnActiveOpStep(true);
        scheduleDraw();
        return;
      }
      if (state.mambaStepperActive && state.mambaStepperPlaying) {
        state.mambaStepperPlaying = false;
        lastStepperTickMs = null;
        syncUi();
      } else {
        state.mambaStepperActive = true;
        state.mambaStepperPlaying = true;
        lastStepperTickMs = null;
        geometryDirty = true;
        rebuildSceneIfNeeded();
        syncUi();
        focusCameraOnActiveOpStep(false);
        scheduleDraw();
      }
      return;
    }

    if (state.mode === 'cutaway_moe') {
      state.moeSelectedCell = null;
      state.moeHoveredCell = null;
      state.moeTokenCycling = false;
      lastMoeCycleTickMs = null;
      if (state.reducedMotion) {
        state.moeStepperActive = true;
        state.moeStepperPlaying = false;
        state.moeStepperStep = (state.moeStepperStep + 1) % MOE_STEPPER_STEPS.length;
        state.moeStepperSubProgress = 0;
        manualCameraOverride = false;
        geometryDirty = true;
        rebuildSceneIfNeeded();
        syncUi();
        focusCameraOnActiveOpStep(true);
        scheduleDraw();
        return;
      }
      if (state.moeStepperActive && state.moeStepperPlaying) {
        state.moeStepperPlaying = false;
        lastStepperTickMs = null;
        syncUi();
      } else {
        state.moeStepperActive = true;
        state.moeStepperPlaying = true;
        lastStepperTickMs = null;
        geometryDirty = true;
        rebuildSceneIfNeeded();
        syncUi();
        focusCameraOnActiveOpStep(false);
        scheduleDraw();
      }
      return;
    }

    state.gqaSelectedCell = null;
    state.gqaHoveredCell = null;
    if (state.reducedMotion) {
      state.gqaStepperActive = true;
      state.gqaStepperPlaying = false;
      state.gqaStepperStep = (state.gqaStepperStep + 1) % GQA_STEPPER_STEPS.length;
      state.gqaStepperSubProgress = 0;
      manualCameraOverride = false;
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      focusCameraOnActiveOpStep(true);
      scheduleDraw();
      return;
    }
    if (state.gqaStepperActive && state.gqaStepperPlaying) {
      state.gqaStepperPlaying = false;
      lastStepperTickMs = null;
      syncUi();
    } else {
      state.gqaStepperActive = true;
      state.gqaStepperPlaying = true;
      lastStepperTickMs = null;
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      focusCameraOnActiveOpStep(false);
      scheduleDraw();
    }
  };

  const markDirtyAndSchedule = (rebuildGeometry: boolean) => {
    if (rebuildGeometry) {
      geometryDirty = true;
      rebuildSceneIfNeeded();
    }
    syncUi();
    scheduleDraw();
  };

  const flyCameraForBeat = (beatIndex: number) => {
    const beat = getWalkthroughBeat(beatIndex);
    manualCameraOverride = false;
    lastAutoFollowedStepKey = '';
    if (beat.mode === 'tower') {
      flyToPose(cameraPoseForSection(beat.section, currentMetrics));
      return;
    }
    if (!currentCutawayScene) return;
    if (beat.mode === 'cutaway_embedding' || beat.mode === 'cutaway_output') {
      if (beat.stepperStep !== undefined && beat.stepperStep > 0) {
        const opBounds = resolveOpFocusBounds(currentCutawayScene, beat.stepperStep);
        if (opBounds) {
          flyToPose(
            cameraPoseForOpStep(
              beat.mode,
              opBounds,
              currentMetrics,
              'three_quarter',
              measureCutawayInsets(beat.mode),
            ),
          );
          return;
        }
      }
      flyToPose(
        cameraPoseForCutaway(
          beat.mode,
          currentCutawayScene.fitBounds,
          currentMetrics,
          'three_quarter',
          measureCutawayInsets(beat.mode),
        ),
      );
      return;
    }
    if (beat.stepperStep !== undefined && state.stepperAutoFollow) {
      focusCameraOnActiveOpStep(true);
      return;
    }
    flyToPose(
      cameraPoseForCutaway(
        beat.mode,
        currentCutawayScene.fitBounds,
        currentMetrics,
        activeViewAngleForMode(beat.mode),
        measureCutawayInsets(beat.mode),
      ),
    );
  };

  const applyTourBeat = (beatIndex: number) => {
    const beat = getWalkthroughBeat(beatIndex);
    if (state.mode === 'tower' && beat.mode !== 'tower') {
      state.savedMacroCamera = cloneCameraPose(state.camera);
    }

    state.tourActive = true;
    state.tourBeat = beat.index;
    state.tourChapter = beat.chapterIndex;
    state.tourExploring = false;
    state.precision = beat.precision;
    state.section = beat.section;
    state.selectedBay = beat.selectedBay;
    state.hoveredBay = null;
    state.selectedPart = null;
    state.hoveredPart = null;

    // Clear stale pinned/hovered cells & playback timers before staging the new beat
    state.gqaHoveredCell = null;
    state.gqaSelectedCell = null;
    state.gqaStepperActive = false;
    state.gqaStepperPlaying = false;
    state.gqaAccumTermProgress = 1;
    state.mambaHoveredCell = null;
    state.mambaSelectedCell = null;
    state.mambaStepperActive = false;
    state.mambaStepperPlaying = false;
    state.mambaTemporalPlaying = false;
    state.mambaAccumTermProgress = 1;
    state.moeHoveredCell = null;
    state.moeSelectedCell = null;
    state.moeStepperActive = false;
    state.moeStepperPlaying = false;
    state.moeTokenCycling = false;
    state.moeAccumTermProgress = 1;
    state.embeddingActiveCell = null;
    state.outputActiveCell = null;
    embeddingHoveredCell = null;
    outputHoveredCell = null;
    bookendStepIndex = beat.stepperStep ?? null;
    lastStepperTickMs = null;
    lastStepperCellKey = '';
    lastMambaTemporalTickMs = null;
    lastMoeCycleTickMs = null;
    clearSpineWave(false);

    if (beat.mambaBayId) state.mambaBayId = beat.mambaBayId;
    if (beat.moeBayId) state.moeBayId = beat.moeBayId;
    if (beat.gqaBayId) state.gqaBayId = beat.gqaBayId;
    if (beat.gqaHead !== undefined) {
      if (beat.gqaHead < 16) state.gqaActiveGroup0Head = beat.gqaHead;
      else state.gqaActiveGroup1Head = beat.gqaHead;
    }

    state.executionMode = beat.executionMode;
    if (
      beat.executionMode === 'ar_decode' ||
      beat.arAction === 'step_once_with_wave' ||
      beat.arAction === 'inspect_cached_gqa' ||
      beat.arAction === 'invite_branch_and_chart'
    ) {
      decodeSessions = ensureTourArDecodeSession(decodeSessions, hybridModel, beat.precision);
      const activeStep = getActiveDecodeStep(decodeSessions.tour!);
      state.tokens = activeStep.totalTokens;
      state.embeddingFocusToken = Math.max(0, activeStep.tokens - 1);
      state.outputFocusToken = Math.max(0, activeStep.tokens - 1);
    } else {
      state.tokens = beat.tokens;
      state.embeddingFocusToken = Math.max(0, state.tokens - 1);
      state.outputFocusToken = Math.max(0, state.tokens - 1);
    }

    state.mambaStateToken = null;
    state.moeFocusToken = null;
    state.moeActiveExpert = null;
    state.mode = beat.mode;

    if (beat.stepperStep !== undefined) {
      if (beat.mode === 'cutaway_gqa') {
        state.gqaStepperActive = true;
        state.gqaStepperStep = beat.stepperStep;
        state.gqaStepperSubProgress = 0;
        state.gqaStepperPlaying = Boolean(beat.stepperAutoPlay && !state.reducedMotion);
      } else if (beat.mode === 'cutaway_mamba') {
        state.mambaStepperActive = true;
        state.mambaStepperStep = beat.stepperStep;
        state.mambaStepperSubProgress = 0;
        state.mambaStepperPlaying = Boolean(beat.stepperAutoPlay && !state.reducedMotion);
      } else if (beat.mode === 'cutaway_moe') {
        state.moeStepperActive = true;
        state.moeStepperStep = beat.stepperStep;
        state.moeStepperSubProgress = 0;
        state.moeStepperPlaying = Boolean(beat.stepperAutoPlay && !state.reducedMotion);
      }
    }

    geometryDirty = true;
    rebuildSceneIfNeeded();
    manualCameraOverride = false;
    lastAutoFollowedStepKey = '';
    syncUi();

    flyCameraForBeat(beat.index);

    if (beat.arAction === 'step_once_with_wave' && state.mode === 'tower') {
      startSpineWave('instructional');
    } else {
      scheduleDraw();
    }
  };

  const exitTourToFreeExplore = () => {
    if (!state.tourActive && !state.tourExploring) return;
    state.tourActive = false;
    state.tourExploring = false;
    if (state.executionMode === 'ar_decode') {
      const activeStep = getActiveDecodeStep(activeDecodeSession());
      state.tokens = activeStep.totalTokens;
      state.embeddingFocusToken = Math.max(0, activeStep.tokens - 1);
      state.outputFocusToken = Math.max(0, activeStep.tokens - 1);
      state.mambaStateToken = null;
      state.mambaSelectedCell = null;
      state.mambaHoveredCell = null;
      state.moeFocusToken = null;
      state.moeActiveExpert = null;
      state.moeSelectedCell = null;
      state.moeHoveredCell = null;
      state.gqaSelectedCell = null;
      state.gqaHoveredCell = null;
      state.embeddingActiveCell = null;
      state.outputActiveCell = null;
      embeddingHoveredCell = null;
      outputHoveredCell = null;
      geometryDirty = true;
      rebuildSceneIfNeeded();
    }
    syncUi();
    scheduleDraw();
  };

  const selectMode = (nextMode: ViewMode, customPose?: CameraPose, keepTour = false) => {
    if (!keepTour) markTourExploring();
    if (state.mode === 'tower' && nextMode !== 'tower') {
      state.savedMacroCamera = cloneCameraPose(state.camera);
    } else if (state.mode === 'cutaway_embedding' && nextMode !== 'cutaway_embedding' && manualCameraOverride) {
      state.savedEmbedPose = cloneCameraPose(state.camera);
    } else if (state.mode === 'cutaway_output' && nextMode !== 'cutaway_output' && manualCameraOverride) {
      state.savedOutputPose = cloneCameraPose(state.camera);
    }

    const modeChanged = state.mode !== nextMode;
    state.mode = nextMode;
    state.hoveredBay = null;
    state.hoveredPart = null;
    state.gqaHoveredCell = null;
    state.mambaHoveredCell = null;
    state.moeHoveredCell = null;
    embeddingHoveredCell = null;
    outputHoveredCell = null;
    if (modeChanged) {
      state.selectedPart = null;
      state.gqaSelectedCell = null;
      state.mambaSelectedCell = null;
      state.moeSelectedCell = null;
      state.embeddingActiveCell = null;
      state.outputActiveCell = null;
      bookendStepIndex = null;
      if (nextMode !== 'tower') {
        clearSpineWave(true);
      }
      if (nextMode !== 'cutaway_gqa') {
        state.gqaStepperPlaying = false;
      }
      if (nextMode !== 'cutaway_mamba') {
        state.mambaStepperPlaying = false;
        state.mambaTemporalPlaying = false;
        lastMambaTemporalTickMs = null;
      }
      if (nextMode !== 'cutaway_moe') {
        state.moeStepperPlaying = false;
        state.moeTokenCycling = false;
        lastMoeCycleTickMs = null;
      }
      lastStepperTickMs = null;
      lastStepperCellKey = '';
    }
    if (nextMode === 'cutaway_embedding') state.section = 'embedding';
    else if (nextMode === 'cutaway_mamba') state.section = 'mamba';
    else if (nextMode === 'cutaway_moe') state.section = 'moe';
    else if (nextMode === 'cutaway_gqa') state.section = 'gqa';
    else if (nextMode === 'cutaway_output') state.section = 'output';

    manualCameraOverride = false;
    lastAutoFollowedStepKey = '';
    geometryDirty = true;
    rebuildSceneIfNeeded();
    syncUi();

    if (customPose) {
      flyToPose(customPose);
    } else if (nextMode === 'tower') {
      flyToPose(state.savedMacroCamera);
    } else if (nextMode === 'cutaway_embedding' && state.savedEmbedPose) {
      flyToPose(cloneCameraPose(state.savedEmbedPose));
    } else if (nextMode === 'cutaway_output' && state.savedOutputPose) {
      flyToPose(cloneCameraPose(state.savedOutputPose));
    } else if (currentCutawayScene) {
      const stepper = getActiveCutawayStepper();
      if (stepper?.active && state.stepperAutoFollow) {
        focusCameraOnActiveOpStep(true);
      } else {
        flyToPose(
          cameraPoseForCutaway(
            nextMode,
            currentCutawayScene.fitBounds,
            currentMetrics,
            activeViewAngleForMode(nextMode),
            measureCutawayInsets(nextMode),
          ),
        );
      }
    }

    scheduleDraw();
  };

  const selectSection = (section: TocSectionId) => {
    markTourExploring();
    // If already in a cutaway and user clicks any cutaway section in TOC, switch cutaways cleanly!
    if (state.mode !== 'tower') {
      if (section === 'embedding') {
        selectMode('cutaway_embedding');
        return;
      }
      if (section === 'mamba') {
        selectMode('cutaway_mamba');
        return;
      }
      if (section === 'moe') {
        selectMode('cutaway_moe');
        return;
      }
      if (section === 'gqa') {
        selectMode('cutaway_gqa');
        return;
      }
      if (section === 'output') {
        selectMode('cutaway_output');
        return;
      }
    }

    state.section = section;
    if (section === 'mamba') state.selectedBay = 1;
    else if (section === 'moe') state.selectedBay = 2;
    else if (section === 'gqa') state.selectedBay = 6;
    else if (section === 'kv_fp8') state.selectedBay = 42;
    else if (section === 'overview' || section === 'embedding' || section === 'output') {
      state.selectedBay = null;
    }
    state.hoveredBay = null;
    state.selectedPart = null;
    state.hoveredPart = null;

    const targetPose = cameraPoseForSection(section, currentMetrics);
    if (state.mode !== 'tower') {
      selectMode('tower', targetPose);
      return;
    }
    flyToPose(targetPose);
    markDirtyAndSchedule(true);
  };

  const inspectInternals = (kind: BayKind, bayId?: number) => {
    if (kind === 'M') {
      const bay = bayId ? NEMOTRON_SPEC.bays[bayId - 1] : null;
      state.mambaBayId = bay?.feeder ? 42 : 1;
      selectMode('cutaway_mamba');
    } else if (kind === 'E') {
      state.moeBayId = bayId === 52 ? 52 : 2;
      selectMode('cutaway_moe');
    } else {
      state.gqaBayId = bayId === 43 ? 43 : 6;
      selectMode('cutaway_gqa');
    }
  };

  const zoomToBay = (bayId: number) => {
    const bayBounds = boundsForBay(bayId);
    const bay = NEMOTRON_SPEC.bays[bayId - 1];
    const az = bay?.kind === '*' ? 24 : 26;
    const el = bay?.kind === '*' || bay?.kind === 'E' ? 24 : 18;
    const targetPose = fitPerspectiveToBounds(
      bayBounds,
      az,
      el,
      currentMetrics.innerWidth,
      currentMetrics.innerHeight,
      DEFAULT_FOV_DEG,
      0.78,
    );
    flyToPose(targetPose);
  };

  const fitActiveView = () => {
    manualCameraOverride = false;
    lastAutoFollowedStepKey = '';
    if (state.mode === 'tower') {
      state.section = 'overview';
      flyToPose(cameraPoseForSection('overview', currentMetrics));
      syncUi();
    } else if (currentCutawayScene) {
      if (state.mode === 'cutaway_embedding') {
        state.savedEmbedPose = null;
      } else if (state.mode === 'cutaway_output') {
        state.savedOutputPose = null;
      } else if (state.mode === 'cutaway_gqa') {
        if (state.gqaViewAngle === 'three_quarter') savedGqaSpatialPose = null;
        else savedGqaPlanPose = null;
      } else if (state.mode === 'cutaway_mamba') {
        if (state.mambaViewAngle === 'three_quarter') savedMambaSpatialPose = null;
        else savedMambaPlanPose = null;
      } else if (state.mode === 'cutaway_moe') {
        if (state.moeViewAngle === 'three_quarter') savedMoeSpatialPose = null;
        else savedMoePlanPose = null;
      }
      flyToPose(
        cameraPoseForCutaway(
          state.mode,
          currentCutawayScene.fitBounds,
          currentMetrics,
          activeViewAngleForMode(state.mode),
          measureCutawayInsets(state.mode),
        ),
      );
      syncUi();
    }
  };

  const resetAll = () => {
    state.executionMode = 'full_sequence';
    decodeSessions = createDualDecodeSessionStore(hybridModel, 'bf16');
    state.tokens = DEFAULT_TOKENS;
    state.precision = 'bf16';
    state.selectedBay = null;
    state.hoveredBay = null;
    state.selectedPart = null;
    state.hoveredPart = null;
    state.mambaHoveredCell = null;
    state.mambaSelectedCell = null;
    state.mambaActiveHead = 0;
    state.mambaStateToken = null;
    state.mambaTemporalPlaying = false;
    state.mambaStepperActive = false;
    state.mambaStepperPlaying = false;
    state.mambaStepperStep = 0;
    state.mambaStepperSubProgress = 0;
    state.mambaAccumTermProgress = 1;
    state.gqaHoveredCell = null;
    state.gqaSelectedCell = null;
    state.gqaActiveGroup0Head = 0;
    state.gqaActiveGroup1Head = 16;
    state.gqaStepperActive = false;
    state.gqaStepperPlaying = false;
    state.gqaStepperStep = 0;
    state.gqaStepperSubProgress = 0;
    state.gqaAccumTermProgress = 1;
    state.moeFocusToken = null;
    state.moeActiveExpert = null;
    state.moeTokenCycling = false;
    state.moeHoveredCell = null;
    state.moeSelectedCell = null;
    state.moeStepperActive = false;
    state.moeStepperPlaying = false;
    state.moeStepperStep = 0;
    state.moeStepperSubProgress = 0;
    state.moeAccumTermProgress = 1;
    state.embeddingFocusToken = DEFAULT_TOKENS - 1;
    state.outputFocusToken = DEFAULT_TOKENS - 1;
    state.embeddingActiveCell = null;
    state.outputActiveCell = null;
    embeddingHoveredCell = null;
    outputHoveredCell = null;
    bookendStepIndex = null;
    clearSpineWave(false);
    state.savedEmbedPose = null;
    state.savedOutputPose = null;
    state.tourExploring = false;
    state.stepperAutoFollow = true;
    manualCameraOverride = false;
    lastAutoFollowedStepKey = '';
    lastStepperTickMs = null;
    lastStepperCellKey = '';
    lastMambaTemporalTickMs = null;
    lastMoeCycleTickMs = null;
    savedGqaSpatialPose = null;
    savedGqaPlanPose = null;
    savedMambaSpatialPose = null;
    savedMambaPlanPose = null;
    savedMoeSpatialPose = null;
    savedMoePlanPose = null;

    if (state.mode === 'tower') {
      state.section = 'overview';
      flyToPose(cameraPoseForSection('overview', currentMetrics));
      markDirtyAndSchedule(true);
    } else {
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      if (currentCutawayScene) {
        flyToPose(
          cameraPoseForCutaway(
            state.mode,
            currentCutawayScene.fitBounds,
            currentMetrics,
            activeViewAngleForMode(state.mode),
            measureCutawayInsets(state.mode),
          ),
        );
      }
      scheduleDraw();
    }
  };

  const stepBay = (delta: -1 | 1) => {
    if (state.mode !== 'tower') return;
    markTourExploring();
    if (state.selectedBay === null) {
      state.selectedBay = delta > 0 ? 1 : 52;
    } else {
      state.selectedBay = Math.max(1, Math.min(52, state.selectedBay + delta));
    }
    state.hoveredBay = null;
    markDirtyAndSchedule(true);
  };

  const memoryDrawer = new MemoryScalingDrawer(NEMOTRON_SPEC, memoryChartHost, {
    onClose: () => {
      state.memoryChartOpen = false;
      syncUi();
    },
    onPrecisionChange: (mode: PrecisionMode) => {
      if (mode === state.precision) return;
      state.precision = mode;
      markDirtyAndSchedule(true);
    },
    onProbeTokensChange: (tokens: number) => {
      state.memoryProbeTokens = tokens;
      syncUi();
    },
  });

  const applyActiveDecodeAction = (
    action: DecodeSessionAction,
    triggerFastWave = false,
  ) => {
    markTourExploring();
    const result = applyDecodeActionToActiveSession(
      decodeSessions,
      state.tourActive,
      state.tourBeat,
      hybridModel,
      state.precision,
      action,
    );
    decodeSessions = result.store;
    const activeStep = getActiveDecodeStep(result.session);
    state.tokens = activeStep.totalTokens;
    state.embeddingFocusToken = Math.max(0, activeStep.tokens - 1);
    state.outputFocusToken = Math.max(0, activeStep.tokens - 1);
    state.mambaStateToken = null;
    state.mambaTemporalPlaying = false;
    state.mambaSelectedCell = null;
    state.mambaHoveredCell = null;
    state.moeFocusToken = null;
    state.moeActiveExpert = null;
    state.moeTokenCycling = false;
    state.moeSelectedCell = null;
    state.moeHoveredCell = null;
    state.gqaSelectedCell = null;
    state.gqaHoveredCell = null;
    state.embeddingActiveCell = null;
    state.outputActiveCell = null;
    embeddingHoveredCell = null;
    outputHoveredCell = null;
    if (action.type === 'reset') {
      clearSpineWave(true);
    }
    if (triggerFastWave && state.mode === 'tower') {
      startSpineWave('fast');
    } else {
      markDirtyAndSchedule(true);
    }
  };

  const controls = new MacroControls(
    NEMOTRON_SPEC,
    {
      ...controlEls,
      spineWaveHud:
        controlEls.spineWaveHud ??
        (document.getElementById('spine-wave-hud') as HTMLDivElement | null),
    },
    {
    onSelectMode: (mode) => selectMode(mode),
    onSelectSection: selectSection,
    onSelectMambaBay: (bayId) => {
      markTourExploring();
      if (state.mambaBayId === bayId && state.mode === 'cutaway_mamba') return;
      state.mambaBayId = bayId;
      state.selectedPart = null;
      state.hoveredPart = null;
      state.mambaSelectedCell = null;
      state.mambaHoveredCell = null;
      markDirtyAndSchedule(true);
    },
    onSelectMambaView: (view: GqaViewAngle) => {
      markTourExploring();
      if (state.mambaViewAngle === view && state.mode === 'cutaway_mamba') return;
      if (
        state.mode === 'cutaway_mamba' &&
        !state.mambaStepperActive &&
        !flyAnim &&
        manualCameraOverride
      ) {
        if (state.mambaViewAngle === 'three_quarter') {
          savedMambaSpatialPose = cloneCameraPose(state.camera);
        } else {
          savedMambaPlanPose = cloneCameraPose(state.camera);
        }
      }
      state.mambaViewAngle = view;
      lastAutoFollowedStepKey = '';
      // Preserve active head, temporal token t, pinned cell, tokens, precision, and stepper state!
      if (state.mode === 'cutaway_mamba') {
        geometryDirty = true;
        rebuildSceneIfNeeded();
        syncUi();
        if (currentCutawayScene) {
          if (state.mambaStepperActive && state.stepperAutoFollow && !manualCameraOverride) {
            focusCameraOnActiveOpStep(true);
          } else {
            const savedPose = view === 'three_quarter' ? savedMambaSpatialPose : savedMambaPlanPose;
            const nextPose =
              savedPose !== null
                ? cloneCameraPose(savedPose)
                : cameraPoseForCutaway(
                    'cutaway_mamba',
                    currentCutawayScene.fitBounds,
                    currentMetrics,
                    view,
                    measureCutawayInsets('cutaway_mamba'),
                  );
            flyToPose(nextPose);
          }
        }
      } else {
        syncUi();
      }
      scheduleDraw();
    },
    onSelectMambaStateToken: (tokenIdx: number | null) => {
      markTourExploring();
      state.mambaTemporalPlaying = false;
      lastMambaTemporalTickMs = null;
      state.mambaHoveredCell = null;
      state.hoveredPart = null;
      const cutT = activeCutawayTokens();
      state.mambaStateToken =
        tokenIdx === null ? null : Math.max(0, Math.min(cutT - 1, tokenIdx));
      if (state.mambaSelectedCell !== null) {
        state.mambaSelectedCell = remapMambaCellToken(
          state.mambaSelectedCell,
          activeMambaStateToken(),
          cutT,
        );
      }
      markDirtyAndSchedule(true);
    },
    onStepMambaStateToken: (delta: -1 | 1) => {
      markTourExploring();
      state.mambaTemporalPlaying = false;
      lastMambaTemporalTickMs = null;
      state.mambaHoveredCell = null;
      state.hoveredPart = null;
      const cutT = activeCutawayTokens();
      const cur = activeMambaStateToken();
      const next = (cur + delta + cutT) % cutT;
      state.mambaStateToken = next;
      if (state.mambaSelectedCell !== null) {
        state.mambaSelectedCell = remapMambaCellToken(state.mambaSelectedCell, next, cutT);
      }
      markDirtyAndSchedule(true);
    },
    onToggleMambaTemporalPlay: () => {
      markTourExploring();
      state.mambaHoveredCell = null;
      state.hoveredPart = null;
      const cutT = activeCutawayTokens();
      if (state.reducedMotion) {
        const next = (activeMambaStateToken() + 1) % cutT;
        state.mambaStateToken = next;
        state.mambaTemporalPlaying = false;
        lastMambaTemporalTickMs = null;
        if (state.mambaSelectedCell !== null) {
          state.mambaSelectedCell = remapMambaCellToken(state.mambaSelectedCell, next, cutT);
        }
        markDirtyAndSchedule(true);
        return;
      }
      if (state.mambaTemporalPlaying) {
        state.mambaTemporalPlaying = false;
        lastMambaTemporalTickMs = null;
        syncUi();
        return;
      }
      state.mambaStepperPlaying = false;
      lastStepperTickMs = null;
      const cur = activeMambaStateToken();
      // When starting playback from latest token (or unpinned), begin cleanly from t = 0 so
      // users watch Conv1D zero-padding fill in and H_{t-1} -> H_t build up in place.
      const startTok = state.mambaStateToken === null || cur >= cutT - 1 ? 0 : cur;
      state.mambaStateToken = startTok;
      if (state.mambaSelectedCell !== null) {
        state.mambaSelectedCell = remapMambaCellToken(state.mambaSelectedCell, startTok, cutT);
      }
      state.mambaTemporalPlaying = true;
      lastMambaTemporalTickMs = null;
      markDirtyAndSchedule(true);
    },
    onSelectMoeBay: (bayId) => {
      markTourExploring();
      if (state.moeBayId === bayId && state.mode === 'cutaway_moe') return;
      state.moeBayId = bayId;
      state.selectedPart = null;
      state.hoveredPart = null;
      state.moeSelectedCell = null;
      state.moeHoveredCell = null;
      markDirtyAndSchedule(true);
    },
    onSelectMoeView: (view: GqaViewAngle) => {
      markTourExploring();
      if (state.moeViewAngle === view && state.mode === 'cutaway_moe') return;
      if (
        state.mode === 'cutaway_moe' &&
        !state.moeStepperActive &&
        !flyAnim &&
        manualCameraOverride
      ) {
        if (state.moeViewAngle === 'three_quarter') {
          savedMoeSpatialPose = cloneCameraPose(state.camera);
        } else {
          savedMoePlanPose = cloneCameraPose(state.camera);
        }
      }
      state.moeViewAngle = view;
      lastAutoFollowedStepKey = '';
      // Preserve focused token, active expert, pinned cell, tokens, precision, and stepper state!
      if (state.mode === 'cutaway_moe') {
        geometryDirty = true;
        rebuildSceneIfNeeded();
        syncUi();
        if (currentCutawayScene) {
          if (state.moeStepperActive && state.stepperAutoFollow && !manualCameraOverride) {
            focusCameraOnActiveOpStep(true);
          } else {
            const savedPose = view === 'three_quarter' ? savedMoeSpatialPose : savedMoePlanPose;
            const nextPose =
              savedPose !== null
                ? cloneCameraPose(savedPose)
                : cameraPoseForCutaway(
                    'cutaway_moe',
                    currentCutawayScene.fitBounds,
                    currentMetrics,
                    view,
                    measureCutawayInsets('cutaway_moe'),
                  );
            flyToPose(nextPose);
          }
        }
      } else {
        syncUi();
      }
      scheduleDraw();
    },
    onSelectMoeFocusToken: (tokenIdx: number) => {
      markTourExploring();
      state.moeTokenCycling = false;
      lastMoeCycleTickMs = null;
      const clamped = Math.max(0, Math.min(activeCutawayTokens() - 1, tokenIdx));
      state.moeFocusToken = clamped;
      state.moeActiveExpert = null; // follow rank-0 active expert for newly selected token
      state.moeSelectedCell = null;
      state.moeHoveredCell = null;
      state.selectedPart = null;
      state.hoveredPart = null;
      markDirtyAndSchedule(true);
    },
    onStepMoeFocusToken: (delta: -1 | 1) => {
      markTourExploring();
      state.moeTokenCycling = false;
      lastMoeCycleTickMs = null;
      const sim = currentCutawayScene?.moeSim ?? activeMoeSim();
      const stepped = stepMoeFocusToken(state.moeFocusToken, delta, sim, state.moeSelectedCell);
      state.moeFocusToken = stepped.nextToken;
      state.moeActiveExpert = null;
      state.moeSelectedCell = stepped.nextSelectedCell;
      state.moeHoveredCell = null;
      state.hoveredPart = null;
      if (state.moeSelectedCell === null && state.selectedPart?.startsWith('moe-expert-')) {
        state.selectedPart = null;
      }
      markDirtyAndSchedule(true);
    },
    onToggleMoeTokenCycle: () => {
      markTourExploring();
      state.moeHoveredCell = null;
      state.hoveredPart = null;
      if (state.reducedMotion) {
        const sim = currentCutawayScene?.moeSim ?? activeMoeSim();
        const stepped = stepMoeFocusToken(state.moeFocusToken, 1, sim, state.moeSelectedCell);
        state.moeStepperPlaying = false;
        lastStepperTickMs = null;
        state.moeFocusToken = stepped.nextToken;
        state.moeActiveExpert = null;
        state.moeSelectedCell = stepped.nextSelectedCell;
        if (state.moeSelectedCell === null && state.selectedPart?.startsWith('moe-expert-')) {
          state.selectedPart = null;
        }
        state.moeTokenCycling = false;
        lastMoeCycleTickMs = null;
        markDirtyAndSchedule(true);
        return;
      }
      if (state.moeTokenCycling) {
        state.moeTokenCycling = false;
        lastMoeCycleTickMs = null;
        syncUi();
        return;
      }
      state.moeStepperPlaying = false;
      lastStepperTickMs = null;
      const cur = activeMoeFocusToken();
      const startTok = state.moeFocusToken === null || cur >= activeCutawayTokens() - 1 ? 0 : cur;
      state.moeFocusToken = startTok;
      state.moeActiveExpert = null;
      if (state.moeSelectedCell !== null) {
        const sim = currentCutawayScene?.moeSim ?? activeMoeSim();
        state.moeSelectedCell = remapMoeCellToken(state.moeSelectedCell, startTok, sim);
      } else if (state.selectedPart?.startsWith('moe-expert-')) {
        state.selectedPart = null;
      }
      state.moeTokenCycling = true;
      lastMoeCycleTickMs = null;
      markDirtyAndSchedule(true);
    },
    onSelectMoeExpert: (expertId: number) => {
      markTourExploring();
      state.moeTokenCycling = false;
      lastMoeCycleTickMs = null;
      const clamped = Math.max(0, Math.min(127, expertId));
      const tok = activeMoeFocusToken();
      state.moeActiveExpert = clamped;
      state.selectedPart = `moe-expert-${clamped}`;
      state.hoveredPart = null;
      state.moeStepperPlaying = false;
      state.moeStepperActive = false;
      state.moeSelectedCell = {
        tensor: 'router_bank',
        row: Math.floor(clamped / 16),
        col: clamped % 16,
        tokenIndex: tok,
        expertId: clamped,
      };
      state.moeHoveredCell = null;
      markDirtyAndSchedule(true);
    },
    onSelectGqaBay: (bayId) => {
      markTourExploring();
      if (state.gqaBayId === bayId && state.mode === 'cutaway_gqa') return;
      state.gqaBayId = bayId;
      state.selectedPart = null;
      state.hoveredPart = null;
      state.gqaSelectedCell = null;
      state.gqaHoveredCell = null;
      markDirtyAndSchedule(true);
    },
    onSelectGqaView: (view: GqaViewAngle) => {
      markTourExploring();
      if (state.gqaViewAngle === view && state.mode === 'cutaway_gqa') return;
      if (
        state.mode === 'cutaway_gqa' &&
        !state.gqaStepperActive &&
        !flyAnim &&
        manualCameraOverride
      ) {
        if (state.gqaViewAngle === 'three_quarter') {
          savedGqaSpatialPose = cloneCameraPose(state.camera);
        } else {
          savedGqaPlanPose = cloneCameraPose(state.camera);
        }
      }
      state.gqaViewAngle = view;
      lastAutoFollowedStepKey = '';
      // Preserve selected head, pinned cell, tokens, precision, and stepper state across mode switches!
      if (state.mode === 'cutaway_gqa') {
        geometryDirty = true;
        rebuildSceneIfNeeded();
        syncUi();
        if (currentCutawayScene) {
          if (state.gqaStepperActive && state.stepperAutoFollow && !manualCameraOverride) {
            focusCameraOnActiveOpStep(true);
          } else {
            const savedPose = view === 'three_quarter' ? savedGqaSpatialPose : savedGqaPlanPose;
            const nextPose =
              savedPose !== null
                ? cloneCameraPose(savedPose)
                : cameraPoseForCutaway(
                    'cutaway_gqa',
                    currentCutawayScene.fitBounds,
                    currentMetrics,
                    view,
                    measureCutawayInsets('cutaway_gqa'),
                  );
            flyToPose(nextPose);
          }
        }
      } else {
        syncUi();
      }
      scheduleDraw();
    },
    onSelectPart: (partId: string | null) => {
      markTourExploring();
      state.selectedPart = partId;
      state.hoveredPart = null;
      applyHeadSelectionToGroupHeads(partId);
      if (partId === null) {
        state.gqaSelectedCell = null;
        state.mambaSelectedCell = null;
        state.moeSelectedCell = null;
        state.embeddingActiveCell = null;
        state.outputActiveCell = null;
      } else if (parseQueryHeadPartId(partId) !== null && state.gqaSelectedCell) {
        // Clear stale cell from a different head so the newly selected head's live Scores/Softmax stay in sync
        state.gqaSelectedCell = null;
      } else if (parseMambaHeadPartId(partId) !== null && state.mambaSelectedCell) {
        state.mambaSelectedCell = null;
      } else if (parseMoeExpertPartId(partId) !== null && state.moeSelectedCell) {
        state.moeSelectedCell = null;
      }
      markDirtyAndSchedule(true);
    },
    onInspectInternals: inspectInternals,
    onTokensChange: (nextTokens) => {
      const clamped = Math.max(MIN_TOKENS, Math.min(MAX_TOKENS, nextTokens));
      if (clamped === state.tokens) return;
      markTourExploring();
      state.tokens = clamped;
      if (state.embeddingFocusToken >= clamped) {
        state.embeddingFocusToken = clamped - 1;
      }
      if (state.outputFocusToken >= clamped) {
        state.outputFocusToken = clamped - 1;
      }
      if (state.embeddingActiveCell && (state.embeddingActiveCell.tokenIndex ?? 0) >= clamped) {
        state.embeddingActiveCell = null;
      }
      if (state.outputActiveCell && (state.outputActiveCell.tokenIndex ?? 0) >= clamped) {
        state.outputActiveCell = null;
      }
      if (state.mambaStateToken !== null && state.mambaStateToken >= clamped) {
        state.mambaStateToken = clamped - 1;
      }
      if (state.mambaSelectedCell && (state.mambaSelectedCell.tokenIndex ?? 0) >= clamped) {
        state.mambaSelectedCell = null;
      }
      if (state.moeFocusToken !== null && state.moeFocusToken >= clamped) {
        state.moeFocusToken = clamped - 1;
        state.moeActiveExpert = null;
      }
      if (state.moeSelectedCell && (state.moeSelectedCell.tokenIndex ?? 0) >= clamped) {
        state.moeSelectedCell = null;
      }
      markDirtyAndSchedule(true);
    },
    onPrecisionChange: (mode: PrecisionMode) => {
      if (mode === state.precision) return;
      markTourExploring();
      state.precision = mode;
      markDirtyAndSchedule(true);
    },
    onFitAll: fitActiveView,
    onZoomSelectedBay: () => {
      if (state.mode === 'tower' && state.selectedBay !== null) {
        zoomToBay(state.selectedBay);
      }
    },
    onReset: resetAll,
    onStepBay: stepBay,
    onClearSelection: () => {
      state.selectedBay = null;
      state.hoveredBay = null;
      state.selectedPart = null;
      state.hoveredPart = null;
      state.gqaSelectedCell = null;
      state.gqaHoveredCell = null;
      state.mambaSelectedCell = null;
      state.mambaHoveredCell = null;
      state.moeSelectedCell = null;
      state.moeHoveredCell = null;
      state.embeddingActiveCell = null;
      state.outputActiveCell = null;
      embeddingHoveredCell = null;
      outputHoveredCell = null;
      markDirtyAndSchedule(true);
    },
    onTourToggle: () => {
      if (state.tourActive) {
        exitTourToFreeExplore();
      } else {
        applyTourBeat(state.tourBeat);
      }
    },
    onTourPrev: () => {
      const prev = prevBeatIndex(state.tourBeat);
      if (prev !== null) applyTourBeat(prev);
    },
    onTourNext: () => {
      const next = nextBeatIndex(state.tourBeat);
      if (next !== null) applyTourBeat(next);
    },
    onTourBeatPrev: () => {
      const prev = prevBeatIndex(state.tourBeat);
      if (prev !== null) applyTourBeat(prev);
    },
    onTourBeatNext: () => {
      const next = nextBeatIndex(state.tourBeat);
      if (next !== null) applyTourBeat(next);
    },
    onTourJumpBeat: (beatIndex: number) => applyTourBeat(beatIndex),
    onTourResumeBeat: () => applyTourBeat(state.tourBeat),
    onTourRefitBeat: () => {
      if (!state.tourActive) {
        fitActiveView();
        return;
      }
      const beat = getWalkthroughBeat(state.tourBeat);
      if (state.mode !== beat.mode) {
        applyTourBeat(state.tourBeat);
        return;
      }
      flyCameraForBeat(state.tourBeat);
      syncUi();
      scheduleDraw();
    },
    onTourToggleBeatPlay: () => toggleCutawayStepperPlay(true),
    onTourRestart: () => {
      decodeSessions = resetTourOnlyDecodeSession(decodeSessions);
      applyTourBeat(0);
    },
    onTourExit: exitTourToFreeExplore,
    onSelectEmbeddingToken: (tokenIdx: number) => {
      markTourExploring();
      const payload = activeEmbeddingPayload();
      const clamped = Math.max(0, Math.min(payload.tokenIds.length - 1, tokenIdx));
      state.embeddingFocusToken = clamped;
      embeddingHoveredCell = null;
      state.embeddingActiveCell = {
        tensor: 'x_embed',
        row: 0,
        col: clamped,
        tokenIndex: clamped,
        vocabId: payload.tokenIds[clamped],
      };
      markDirtyAndSchedule(true);
    },
    onSelectOutputToken: (tokenIdx: number) => {
      markTourExploring();
      const payload = activeOutputPayload();
      const clamped = Math.max(0, Math.min(payload.tokenLabels.length - 1, tokenIdx));
      state.outputFocusToken = clamped;
      outputHoveredCell = null;
      const topCand = payload.lmHead.topKByToken[clamped]?.[0]?.tokenId ?? 0;
      state.outputActiveCell = {
        tensor: 'probs',
        row: topCand,
        col: 0,
        tokenIndex: clamped,
        vocabId: topCand,
      };
      markDirtyAndSchedule(true);
    },
    onNavigateAdjacentBay: (neighbor: CrossBayNeighbor) => {
      if (neighbor.mode === 'cutaway_mamba') {
        const bay = NEMOTRON_SPEC.bays[neighbor.bayId - 1];
        state.mambaBayId = neighbor.bayId === 42 || bay?.feeder ? 42 : 1;
      } else if (neighbor.mode === 'cutaway_moe') {
        state.moeBayId = neighbor.bayId === 52 ? 52 : 2;
      } else if (neighbor.mode === 'cutaway_gqa') {
        state.gqaBayId = neighbor.bayId === 43 ? 43 : 6;
      }
      selectMode(neighbor.mode);
    },
    onReplaySpineWave: () => {
      if (state.mode !== 'tower') {
        selectMode('tower', cameraPoseForSection('overview', currentMetrics), true);
      }
      startSpineWave('fast');
    },
    onReplaySpineWaveMode: (mode: SpineWaveMode) => {
      if (state.mode !== 'tower') {
        selectMode('tower', cameraPoseForSection('overview', currentMetrics), true);
      }
      startSpineWave(mode);
    },
    onToggleSpineWavePause: () => {
      if (state.mode !== 'tower' || spineWaveClock.progress === null || state.reducedMotion) return;
      const toggled = toggleSpineWavePlaybackPause(spineWaveClock, state.reducedMotion);
      applySpineWaveStateToScene(toggled);
      spineWaveLastTickMs = toggled.paused ? null : performance.now();
      controls.syncSpineWaveHud(state, activeHybridExecution(), activeDecodeSessionOrNull());
      scheduleDraw();
    },
    onDismissSpineWaveHud: () => {
      if (spineWaveClock.progress === null && state.spineWaveProgress === null) return;
      clearSpineWave(true);
      markDirtyAndSchedule(true);
    },
    onToggleMemoryChart: () => {
      state.memoryChartOpen = !state.memoryChartOpen;
      if (state.memoryChartOpen) state.shortcutsOpen = false;
      syncUi();
    },
    onToggleShortcuts: () => {
      state.shortcutsOpen = !state.shortcutsOpen;
      syncUi();
    },
    onCloseShortcuts: () => {
      if (!state.shortcutsOpen) return;
      state.shortcutsOpen = false;
      syncUi();
    },
    onGqaStepPrev: () => {
      markTourExploring();
      lastStepperTickMs = null;
      lastStepperCellKey = '';
      manualCameraOverride = false;
      if (state.mode === 'cutaway_embedding') {
        const cur = bookendStepIndex ?? 0;
        const prev = (cur - 1 + EMBEDDING_STEPPER_STEPS.length) % EMBEDDING_STEPPER_STEPS.length;
        applyBookendStepFocus(prev);
        return;
      }
      if (state.mode === 'cutaway_output') {
        const cur = bookendStepIndex ?? 0;
        const prev = (cur - 1 + OUTPUT_STEPPER_STEPS.length) % OUTPUT_STEPPER_STEPS.length;
        applyBookendStepFocus(prev);
        return;
      }
      if (state.mode === 'cutaway_mamba') {
        state.mambaSelectedCell = null;
        state.mambaHoveredCell = null;
        state.mambaStepperPlaying = false;
        if (!state.mambaStepperActive) {
          state.mambaStepperActive = true;
          state.mambaStepperStep = MAMBA_STEPPER_STEPS.length - 1;
        } else {
          state.mambaStepperStep =
            (state.mambaStepperStep - 1 + MAMBA_STEPPER_STEPS.length) % MAMBA_STEPPER_STEPS.length;
        }
        state.mambaStepperSubProgress = 0;
      } else if (state.mode === 'cutaway_moe') {
        state.moeSelectedCell = null;
        state.moeHoveredCell = null;
        state.moeStepperPlaying = false;
        if (!state.moeStepperActive) {
          state.moeStepperActive = true;
          state.moeStepperStep = MOE_STEPPER_STEPS.length - 1;
        } else {
          state.moeStepperStep =
            (state.moeStepperStep - 1 + MOE_STEPPER_STEPS.length) % MOE_STEPPER_STEPS.length;
        }
        state.moeStepperSubProgress = 0;
      } else {
        state.gqaSelectedCell = null;
        state.gqaHoveredCell = null;
        state.gqaStepperPlaying = false;
        if (!state.gqaStepperActive) {
          state.gqaStepperActive = true;
          state.gqaStepperStep = GQA_STEPPER_STEPS.length - 1;
        } else {
          state.gqaStepperStep =
            (state.gqaStepperStep - 1 + GQA_STEPPER_STEPS.length) % GQA_STEPPER_STEPS.length;
        }
        state.gqaStepperSubProgress = 0;
      }
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      focusCameraOnActiveOpStep(true);
      scheduleDraw();
    },
    onGqaStepNext: () => {
      markTourExploring();
      lastStepperTickMs = null;
      lastStepperCellKey = '';
      manualCameraOverride = false;
      if (state.mode === 'cutaway_embedding') {
        const cur = bookendStepIndex ?? -1;
        const next = (cur + 1) % EMBEDDING_STEPPER_STEPS.length;
        applyBookendStepFocus(next);
        return;
      }
      if (state.mode === 'cutaway_output') {
        const cur = bookendStepIndex ?? -1;
        const next = (cur + 1) % OUTPUT_STEPPER_STEPS.length;
        applyBookendStepFocus(next);
        return;
      }
      if (state.mode === 'cutaway_mamba') {
        state.mambaSelectedCell = null;
        state.mambaHoveredCell = null;
        state.mambaStepperPlaying = false;
        if (!state.mambaStepperActive) {
          state.mambaStepperActive = true;
          state.mambaStepperStep = 0;
        } else {
          state.mambaStepperStep = (state.mambaStepperStep + 1) % MAMBA_STEPPER_STEPS.length;
        }
        state.mambaStepperSubProgress = 0;
      } else if (state.mode === 'cutaway_moe') {
        state.moeSelectedCell = null;
        state.moeHoveredCell = null;
        state.moeStepperPlaying = false;
        if (!state.moeStepperActive) {
          state.moeStepperActive = true;
          state.moeStepperStep = 0;
        } else {
          state.moeStepperStep = (state.moeStepperStep + 1) % MOE_STEPPER_STEPS.length;
        }
        state.moeStepperSubProgress = 0;
      } else {
        state.gqaSelectedCell = null;
        state.gqaHoveredCell = null;
        state.gqaStepperPlaying = false;
        if (!state.gqaStepperActive) {
          state.gqaStepperActive = true;
          state.gqaStepperStep = 0;
        } else {
          state.gqaStepperStep = (state.gqaStepperStep + 1) % GQA_STEPPER_STEPS.length;
        }
        state.gqaStepperSubProgress = 0;
      }
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      focusCameraOnActiveOpStep(true);
      scheduleDraw();
    },
    onGqaStepTogglePlay: () => toggleCutawayStepperPlay(false),
    onGqaStepSelect: (step: number) => {
      markTourExploring();
      lastStepperTickMs = null;
      lastStepperCellKey = '';
      manualCameraOverride = false;
      if (state.mode === 'cutaway_embedding' || state.mode === 'cutaway_output') {
        applyBookendStepFocus(step);
        return;
      }
      if (state.mode === 'cutaway_mamba') {
        state.mambaSelectedCell = null;
        state.mambaHoveredCell = null;
        state.mambaStepperActive = true;
        state.mambaStepperPlaying = false;
        state.mambaStepperStep = Math.max(0, Math.min(MAMBA_STEPPER_STEPS.length - 1, step));
        state.mambaStepperSubProgress = 0;
      } else if (state.mode === 'cutaway_moe') {
        state.moeSelectedCell = null;
        state.moeHoveredCell = null;
        state.moeStepperActive = true;
        state.moeStepperPlaying = false;
        state.moeStepperStep = Math.max(0, Math.min(MOE_STEPPER_STEPS.length - 1, step));
        state.moeStepperSubProgress = 0;
      } else {
        state.gqaSelectedCell = null;
        state.gqaHoveredCell = null;
        state.gqaStepperActive = true;
        state.gqaStepperPlaying = false;
        state.gqaStepperStep = Math.max(0, Math.min(GQA_STEPPER_STEPS.length - 1, step));
        state.gqaStepperSubProgress = 0;
      }
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      focusCameraOnActiveOpStep(true);
      scheduleDraw();
    },
    onGqaStepScrub: (progress01: number) => {
      markTourExploring();
      lastStepperTickMs = null;
      lastStepperCellKey = '';
      const clampedProgress = Math.max(0, Math.min(1, progress01));
      if (state.mode === 'cutaway_embedding') {
        const step = Math.min(
          EMBEDDING_STEPPER_STEPS.length - 1,
          Math.floor(Math.max(0, Math.min(0.9999, clampedProgress)) * EMBEDDING_STEPPER_STEPS.length),
        );
        applyBookendStepFocus(step);
        return;
      }
      if (state.mode === 'cutaway_output') {
        const step = Math.min(
          OUTPUT_STEPPER_STEPS.length - 1,
          Math.floor(Math.max(0, Math.min(0.9999, clampedProgress)) * OUTPUT_STEPPER_STEPS.length),
        );
        applyBookendStepFocus(step);
        return;
      }
      if (state.mode === 'cutaway_mamba') {
        state.mambaSelectedCell = null;
        state.mambaHoveredCell = null;
        state.mambaStepperActive = true;
        state.mambaStepperPlaying = false;
        state.mambaStepperSubProgress = clampedProgress;
      } else if (state.mode === 'cutaway_moe') {
        state.moeSelectedCell = null;
        state.moeHoveredCell = null;
        state.moeStepperActive = true;
        state.moeStepperPlaying = false;
        state.moeStepperSubProgress = clampedProgress;
      } else {
        state.gqaSelectedCell = null;
        state.gqaHoveredCell = null;
        state.gqaStepperActive = true;
        state.gqaStepperPlaying = false;
        state.gqaStepperSubProgress = clampedProgress;
      }
      markDirtyAndSchedule(true);
    },
    onGqaClearCell: () => {
      const hadStepper =
        state.gqaStepperActive ||
        state.mambaStepperActive ||
        state.moeStepperActive ||
        bookendStepIndex !== null;
      state.gqaSelectedCell = null;
      state.gqaHoveredCell = null;
      state.gqaStepperActive = false;
      state.gqaStepperPlaying = false;
      state.gqaAccumTermProgress = 1;
      state.mambaSelectedCell = null;
      state.mambaHoveredCell = null;
      state.mambaStepperActive = false;
      state.mambaStepperPlaying = false;
      state.mambaTemporalPlaying = false;
      state.mambaAccumTermProgress = 1;
      lastMambaTemporalTickMs = null;
      state.moeSelectedCell = null;
      state.moeHoveredCell = null;
      state.moeStepperActive = false;
      state.moeStepperPlaying = false;
      state.moeTokenCycling = false;
      state.moeAccumTermProgress = 1;
      state.embeddingActiveCell = null;
      state.outputActiveCell = null;
      embeddingHoveredCell = null;
      outputHoveredCell = null;
      bookendStepIndex = null;
      lastStepperTickMs = null;
      lastStepperCellKey = '';
      lastMoeCycleTickMs = null;
      manualCameraOverride = false;
      lastAutoFollowedStepKey = '';
      geometryDirty = true;
      rebuildSceneIfNeeded();
      syncUi();
      if (state.mode !== 'tower' && (hadStepper || state.stepperAutoFollow)) {
        fitActiveView();
      }
      scheduleDraw();
    },
    onToggleAutoFollow: () => {
      const stepper = getActiveCutawayStepper();
      if (stepper?.active && state.stepperAutoFollow && manualCameraOverride) {
        manualCameraOverride = false;
        lastAutoFollowedStepKey = '';
        syncUi();
        focusCameraOnActiveOpStep(true);
        scheduleDraw();
        return;
      }
      state.stepperAutoFollow = !state.stepperAutoFollow;
      manualCameraOverride = false;
      lastAutoFollowedStepKey = '';
      syncUi();
      if (state.stepperAutoFollow && stepper?.active) {
        focusCameraOnActiveOpStep(true);
        scheduleDraw();
      }
    },
    onSelectExecutionMode: (mode: ExecutionMode) => {
      if (mode === state.executionMode) return;
      markTourExploring();
      state.executionMode = mode;
      state.mambaStateToken = null;
      state.mambaTemporalPlaying = false;
      state.mambaSelectedCell = null;
      state.mambaHoveredCell = null;
      state.moeFocusToken = null;
      state.moeActiveExpert = null;
      state.moeTokenCycling = false;
      state.moeSelectedCell = null;
      state.moeHoveredCell = null;
      state.gqaSelectedCell = null;
      state.gqaHoveredCell = null;
      state.embeddingActiveCell = null;
      state.outputActiveCell = null;
      embeddingHoveredCell = null;
      outputHoveredCell = null;
      lastMambaTemporalTickMs = null;
      lastMoeCycleTickMs = null;
      if (mode === 'ar_decode') {
        const activeStep = getActiveDecodeStep(activeDecodeSession());
        state.tokens = activeStep.totalTokens;
        state.embeddingFocusToken = Math.max(0, activeStep.tokens - 1);
        state.outputFocusToken = Math.max(0, activeStep.tokens - 1);
      } else {
        state.embeddingFocusToken = Math.max(0, state.tokens - 1);
        state.outputFocusToken = Math.max(0, state.tokens - 1);
      }
      markDirtyAndSchedule(true);
    },
    onDecodeStepToken: (tokenId: number) =>
      applyActiveDecodeAction({ type: 'step_token', tokenId }, true),
    onDecodeStepGreedy: () => applyActiveDecodeAction({ type: 'step_greedy' }, true),
    onDecodeNavStep: (stepIndex: number) =>
      applyActiveDecodeAction({ type: 'nav_step', stepIndex }, false),
    onDecodeSwitchBranch: (branchId: string) =>
      applyActiveDecodeAction({ type: 'switch_branch', branchId }, false),
    onDecodeReset: () => applyActiveDecodeAction({ type: 'reset' }, false),
  });

  const drawFrame = (nowMs: number) => {
    pendingRafId = null;

    const measured = measureHostMetrics(host);
    if (!measured) return;
    currentMetrics = measured;

    const canAnimate = device !== null && !device.isLost;
    let continueAnimating = false;
    let stepperChangedUi = false;
    let stepperStageAdvanced = false;
    let stepperProgressToSync: number | null = null;
    let spineWaveHudDirty = false;

    if (canAnimate) {
      if (flyAnim) {
        const elapsed = nowMs - flyAnim.startMs;
        const t = Math.max(0, Math.min(1, elapsed / flyAnim.durationMs));
        state.camera = interpolateCameraPose(flyAnim.from, flyAnim.to, t);
        if (t >= 1) {
          flyAnim = null;
        } else {
          continueAnimating = true;
        }
      }

      // Live Math Stepper playback loop (GQA, Mamba-2 & MoE cutaways, paced ~400 ms/term, respects reducedMotion)
      if (
        state.mode === 'cutaway_gqa' &&
        state.gqaStepperActive &&
        state.gqaStepperPlaying &&
        !state.reducedMotion
      ) {
        const dtSec =
          lastStepperTickMs === null ? 0 : Math.max(0, Math.min(0.1, (nowMs - lastStepperTickMs) / 1000));
        lastStepperTickMs = nowMs;
        const sim = activeGqaSim();
        const durationSec = gqaStepPlayDurationSec(sim, state.gqaStepperStep);
        const nextProgress = state.gqaStepperSubProgress + dtSec / durationSec;
        if (nextProgress >= 1) {
          state.gqaStepperSubProgress = nextProgress % 1;
          state.gqaStepperStep = (state.gqaStepperStep + 1) % GQA_STEPPER_STEPS.length;
          stepperStageAdvanced = true;
        } else {
          state.gqaStepperSubProgress = nextProgress;
        }
        const termP = resolveGqaAccumTermProgress(
          sim,
          state.gqaStepperStep,
          state.gqaStepperSubProgress,
          true,
          false,
          state.gqaHoveredCell !== null || state.gqaSelectedCell !== null,
        );
        state.gqaAccumTermProgress = termP;
        const termBucket = gqaAccumTermBucket(sim, state.gqaStepperStep, termP);
        const cell = resolveActiveGqaCell();
        const key = cell
          ? `gqa:${state.gqaStepperStep}:${cell.tensor}:${cell.row}:${cell.col}:${cell.headIndex ?? -1}:t${termBucket}`
          : '';
        if (key !== lastStepperCellKey || stepperStageAdvanced) {
          lastStepperCellKey = key;
          geometryDirty = true;
          stepperChangedUi = true;
        } else {
          stepperProgressToSync = state.gqaStepperSubProgress;
        }
        continueAnimating = true;
      } else if (
        state.mode === 'cutaway_mamba' &&
        state.mambaStepperActive &&
        state.mambaStepperPlaying &&
        !state.reducedMotion
      ) {
        const dtSec =
          lastStepperTickMs === null ? 0 : Math.max(0, Math.min(0.1, (nowMs - lastStepperTickMs) / 1000));
        lastStepperTickMs = nowMs;
        const sim = activeMambaSim();
        const durationSec = mambaStepPlayDurationSec(sim, state.mambaStepperStep);
        const nextProgress = state.mambaStepperSubProgress + dtSec / durationSec;
        if (nextProgress >= 1) {
          state.mambaStepperSubProgress = nextProgress % 1;
          state.mambaStepperStep = (state.mambaStepperStep + 1) % MAMBA_STEPPER_STEPS.length;
          stepperStageAdvanced = true;
        } else {
          state.mambaStepperSubProgress = nextProgress;
        }
        const termP = resolveMambaAccumTermProgress(
          sim,
          state.mambaStepperStep,
          state.mambaStepperSubProgress,
          true,
          false,
          state.mambaHoveredCell !== null || state.mambaSelectedCell !== null,
        );
        state.mambaAccumTermProgress = termP;
        const termBucket = mambaAccumTermBucket(sim, state.mambaStepperStep, termP);
        const cell = resolveActiveMambaCell();
        const key = cell
          ? `mamba:${state.mambaStepperStep}:${cell.tensor}:${cell.row}:${cell.col}:${cell.headIndex ?? -1}:t${termBucket}`
          : '';
        if (key !== lastStepperCellKey || stepperStageAdvanced) {
          lastStepperCellKey = key;
          geometryDirty = true;
          stepperChangedUi = true;
        } else {
          stepperProgressToSync = state.mambaStepperSubProgress;
        }
        continueAnimating = true;
      } else if (
        state.mode === 'cutaway_moe' &&
        state.moeStepperActive &&
        state.moeStepperPlaying &&
        !state.reducedMotion
      ) {
        const dtSec =
          lastStepperTickMs === null ? 0 : Math.max(0, Math.min(0.1, (nowMs - lastStepperTickMs) / 1000));
        lastStepperTickMs = nowMs;
        const sim = activeMoeSim();
        const durationSec = moeStepPlayDurationSec(sim, state.moeStepperStep);
        const nextProgress = state.moeStepperSubProgress + dtSec / durationSec;
        if (nextProgress >= 1) {
          state.moeStepperSubProgress = nextProgress % 1;
          state.moeStepperStep = (state.moeStepperStep + 1) % MOE_STEPPER_STEPS.length;
          stepperStageAdvanced = true;
        } else {
          state.moeStepperSubProgress = nextProgress;
        }
        const termP = resolveMoeAccumTermProgress(
          sim,
          state.moeStepperStep,
          state.moeStepperSubProgress,
          true,
          false,
          state.moeHoveredCell !== null || state.moeSelectedCell !== null,
        );
        state.moeAccumTermProgress = termP;
        const termBucket = moeAccumTermBucket(sim, state.moeStepperStep, termP);
        const cell = resolveActiveMoeCell();
        const key = cell
          ? `moe:${state.moeStepperStep}:${cell.tensor}:${cell.row}:${cell.col}:${cell.expertId ?? -1}:t${termBucket}`
          : '';
        if (key !== lastStepperCellKey || stepperStageAdvanced) {
          lastStepperCellKey = key;
          geometryDirty = true;
          stepperChangedUi = true;
        } else {
          stepperProgressToSync = state.moeStepperSubProgress;
        }
        continueAnimating = true;
      } else {
        lastStepperTickMs = null;
      }

      // Mamba-2 temporal recurrence playback loop (~950 ms per token t = 0 -> T-1, respects reducedMotion)
      if (
        state.mode === 'cutaway_mamba' &&
        state.mambaTemporalPlaying &&
        !state.reducedMotion
      ) {
        if (lastMambaTemporalTickMs === null) {
          lastMambaTemporalTickMs = nowMs;
        } else if (nowMs - lastMambaTemporalTickMs >= 950) {
          lastMambaTemporalTickMs = nowMs;
          const tick = advanceMambaTemporalPlaybackTick(
            state.mambaStateToken,
            activeCutawayTokens(),
            state.mambaSelectedCell,
            state.mambaHoveredCell,
          );
          state.mambaStateToken = tick.nextToken;
          state.mambaSelectedCell = tick.nextSelectedCell;
          state.mambaHoveredCell = tick.nextHoveredCell;
          geometryDirty = true;
          stepperChangedUi = true;
        }
        continueAnimating = true;
      } else {
        lastMambaTemporalTickMs = null;
      }

      // MoE per-token routing cycle loop (~1100 ms per token t = 0 -> T-1, respects reducedMotion)
      if (
        state.mode === 'cutaway_moe' &&
        state.moeTokenCycling &&
        !state.reducedMotion
      ) {
        if (lastMoeCycleTickMs === null) {
          lastMoeCycleTickMs = nowMs;
        } else if (nowMs - lastMoeCycleTickMs >= 1100) {
          lastMoeCycleTickMs = nowMs;
          const sim = currentCutawayScene?.moeSim ?? activeMoeSim();
          const tick = advanceMoeTokenCycleTick(
            state.moeFocusToken,
            sim,
            state.moeSelectedCell,
            state.moeHoveredCell,
          );
          state.moeFocusToken = tick.nextToken;
          state.moeActiveExpert = null;
          state.moeSelectedCell = tick.nextSelectedCell;
          state.moeHoveredCell = tick.nextHoveredCell;
          geometryDirty = true;
          stepperChangedUi = true;
        }
        continueAnimating = true;
      } else {
        lastMoeCycleTickMs = null;
      }

      // RC3 52-Layer Spine Propagation Wave (Fast 0.9s vs Slow Instructional 5.5s, Pause/Resume & static reducedMotion summary)
      if (spineWaveClock.progress !== null) {
        if (state.mode !== 'tower') {
          clearSpineWave(true);
          geometryDirty = true;
          stepperChangedUi = true;
        } else if (state.reducedMotion) {
          spineWaveLastTickMs = null;
          if (state.spineWaveProgress !== 1.0) {
            const settled = startSpineWavePlayback(state.spineWaveMode, true);
            applySpineWaveStateToScene(settled);
            lastSpineWaveGeoKey = computeSpineWaveGeometryKey(settled.progress, settled.mode);
            geometryDirty = true;
            spineWaveHudDirty = true;
          }
        } else if (spineWaveClock.paused) {
          spineWaveLastTickMs = null;
        } else {
          const dtMs =
            spineWaveLastTickMs === null
              ? 0
              : Math.max(0, Math.min(100, nowMs - spineWaveLastTickMs));
          spineWaveLastTickMs = nowMs;
          const stepped = advanceSpineWavePlayback(spineWaveClock, dtMs, false);
          applySpineWaveStateToScene(stepped.next);
          const nextWaveGeoKey = computeSpineWaveGeometryKey(
            stepped.next.progress,
            stepped.next.mode,
          );
          if (nextWaveGeoKey !== lastSpineWaveGeoKey) {
            lastSpineWaveGeoKey = nextWaveGeoKey;
            geometryDirty = true;
          }
          if (stepped.finished) {
            spineWaveLastTickMs = null;
            lastSpineWaveGeoKey = 'idle';
            geometryDirty = true;
            stepperChangedUi = true;
          } else if (stepped.animating) {
            spineWaveHudDirty = true;
            continueAnimating = true;
          }
        }
      } else {
        spineWaveLastTickMs = null;
      }
    } else {
      lastStepperTickMs = null;
      lastMambaTemporalTickMs = null;
      lastMoeCycleTickMs = null;
      spineWaveLastTickMs = null;
    }

    rebuildSceneIfNeeded();
    if (stepperChangedUi) {
      syncUi();
    } else {
      if (stepperProgressToSync !== null) {
        controls.syncStepperProgressOnly(stepperProgressToSync);
      }
      if (spineWaveHudDirty) {
        controls.syncSpineWaveHud(state, activeHybridExecution(), activeDecodeSessionOrNull());
      }
    }
    if (stepperStageAdvanced && state.stepperAutoFollow && !manualCameraOverride && canAnimate) {
      focusCameraOnActiveOpStep(false);
    }

    const activeBounds =
      state.mode === 'tower'
        ? currentTowerScene!.fitBounds
        : currentCutawayScene!.fitBounds;

    currentSnapshot = buildCameraSnapshot(state.camera, activeBounds, currentMetrics);

    if (
      canAnimate &&
      solidsPass &&
      sheetsPass &&
      linesPass &&
      ribbonsPass &&
      flatFlowPass
    ) {
      const fb = device!.ensureSize(currentMetrics);
      device!.beginFrame(fb);
      const useSheetPipeline =
        state.mode !== 'tower' &&
        currentCutawayScene !== null &&
        Boolean(currentCutawayScene.sheetCount && currentCutawayScene.sheetCount > 0);

      if (useSheetPipeline) {
        sheetsPass.draw(currentSnapshot.viewProj);
        linesPass.draw(
          currentSnapshot.viewProj,
          fb.vpWidth,
          fb.vpHeight,
          currentMetrics.dpr,
          currentSnapshot.eye,
        );
        const hasActivePulseRibbon = shouldAnimateCutawayPulse(state);
        const pulseTime = hasActivePulseRibbon ? nowMs * 0.001 : -1;
        flatFlowPass.draw(currentSnapshot.viewProj, pulseTime);
        if (hasActivePulseRibbon) {
          continueAnimating = true;
        }
      } else {
        solidsPass.draw(currentSnapshot.viewProj);
        linesPass.draw(
          currentSnapshot.viewProj,
          fb.vpWidth,
          fb.vpHeight,
          currentMetrics.dpr,
          currentSnapshot.eye,
        );
        ribbonsPass.draw(currentSnapshot.viewProj);
      }
      device!.endFrame(fb);
    }

    // Synchronous black-pill DOM/SVG overlay projection (skipped when camera, metrics, chrome, and scene are unchanged — PR-11)
    const cam = state.camera;
    const overlayKey =
      `${state.mode}|v${sceneBuildVersion}|${currentMetrics.hostWidth.toFixed(1)}x${currentMetrics.hostHeight.toFixed(1)}@${currentMetrics.dpr.toFixed(2)}|` +
      `${cam.target[0].toFixed(4)},${cam.target[1].toFixed(4)},${cam.target[2].toFixed(4)},${cam.distance.toFixed(4)},${cam.azimuthDeg.toFixed(3)},${cam.elevationDeg.toFixed(3)},${cam.fovDeg.toFixed(2)}|` +
      `${state.section}|${state.selectedBay ?? ''}|${state.hoveredBay ?? ''}|${state.selectedPart ?? ''}|${state.hoveredPart ?? ''}|t${state.tokens}|${lastChromeKey}`;

    if (overlayKey !== lastOverlayRenderKey) {
      lastOverlayRenderKey = overlayKey;
      if (state.mode === 'tower' && currentTowerScene) {
        overlay.update({
          spec: NEMOTRON_SPEC,
          scene: currentTowerScene,
          snapshot: currentSnapshot,
          section: state.section,
          tokens: state.tokens,
          selectedBay: state.selectedBay,
          hoveredBay: state.hoveredBay,
        });
      } else if (state.mode !== 'tower' && currentCutawayScene) {
        const { topInsetPx, bottomInsetPx } = measureCutawayInsets(state.mode);
        overlay.updateCutaway({
          scene: currentCutawayScene,
          snapshot: currentSnapshot,
          tokens: state.tokens,
          selectedPart: state.selectedPart,
          hoveredPart: state.hoveredPart,
          topInsetPx,
          bottomInsetPx,
          activeOpPartIds: currentCutawayScene.activeOpPartIds ?? null,
          cellPopover: currentCutawayScene.cellPopover ?? null,
          activeCellGrids: currentCutawayScene.activeCellGrids ?? null,
        });
      }
    }

    if (continueAnimating && canAnimate) {
      scheduleDraw();
    }
  };

  const scheduleDraw = () => {
    if (pendingRafId !== null) return;
    pendingRafId = window.requestAnimationFrame(drawFrame);
  };

  const drawNow = () => {
    if (pendingRafId !== null) {
      window.cancelAnimationFrame(pendingRafId);
      pendingRafId = null;
    }
    drawFrame(performance.now());
  };

  // --- ResizeObserver & DPR watcher ---
  let lastHostW = currentMetrics.hostWidth;
  let lastHostH = currentMetrics.hostHeight;
  let lastDpr = currentMetrics.dpr;

  const handleResizeOrDpr = () => {
    const next = measureHostMetrics(host);
    if (!next) return;
    if (
      Math.abs(next.hostWidth - lastHostW) < 0.5 &&
      Math.abs(next.hostHeight - lastHostH) < 0.5 &&
      Math.abs(next.dpr - lastDpr) < 1e-3
    ) {
      return;
    }
    invalidateCutawayInsets();
    currentMetrics = next;
    lastHostW = next.hostWidth;
    lastHostH = next.hostHeight;
    lastDpr = next.dpr;
    if (state.mode === 'tower' && state.section === 'overview' && !flyAnim && state.selectedBay === null) {
      state.camera = cameraPoseForSection('overview', next);
    } else if (state.mode !== 'tower' && currentCutawayScene && !manualCameraOverride) {
      flyAnim = null;
      savedGqaSpatialPose = null;
      savedGqaPlanPose = null;
      savedMambaSpatialPose = null;
      savedMambaPlanPose = null;
      savedMoeSpatialPose = null;
      savedMoePlanPose = null;
      state.savedEmbedPose = null;
      state.savedOutputPose = null;
      const stepper = getActiveCutawayStepper();
      const insets = measureCutawayInsets(state.mode);
      if (stepper?.active && state.stepperAutoFollow) {
        const opBounds = resolveOpFocusBounds(currentCutawayScene, stepper.step);
        state.camera = cameraPoseForOpStep(
          state.mode,
          opBounds ?? currentCutawayScene.fitBounds,
          next,
          activeViewAngleForMode(state.mode),
          insets,
        );
      } else {
        state.camera = cameraPoseForCutaway(
          state.mode,
          currentCutawayScene.fitBounds,
          next,
          activeViewAngleForMode(state.mode),
          insets,
        );
      }
    }
    drawNow();
  };

  const resizeObserver = new ResizeObserver(() => handleResizeOrDpr());
  resizeObserver.observe(host);

  // Dynamically re-measure cutaway insets whenever Math Dock (wrapped formulas) or floating toolbar resizes
  const chromeResizeObserver = new ResizeObserver(() => {
    const key = `${controlEls.floatingToolbar.offsetHeight}:${controlEls.gqaMathDock.offsetHeight}`;
    if (key === lastChromeKey) return;
    lastChromeKey = key;
    invalidateCutawayInsets();
    if (state.mode !== 'tower') {
      scheduleDraw();
    }
  });
  chromeResizeObserver.observe(controlEls.gqaMathDock);
  chromeResizeObserver.observe(controlEls.floatingToolbar);

  let dprMediaQuery: MediaQueryList | null = null;
  const watchDpr = () => {
    if (dprMediaQuery) dprMediaQuery.removeEventListener('change', onDprChange);
    dprMediaQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    dprMediaQuery.addEventListener('change', onDprChange);
  };
  const onDprChange = () => {
    watchDpr();
    handleResizeOrDpr();
  };
  watchDpr();

  // Live prefers-reduced-motion subscription (PR-18)
  const handleReducedMotionChange = (matches: boolean) => {
    if (state.reducedMotion === matches) return;
    state.reducedMotion = matches;
    if (matches) {
      if (flyAnim) {
        state.camera = cloneCameraPose(flyAnim.to);
        flyAnim = null;
      }
      state.gqaStepperPlaying = false;
      state.mambaStepperPlaying = false;
      state.moeStepperPlaying = false;
      state.mambaTemporalPlaying = false;
      state.moeTokenCycling = false;
      state.gqaAccumTermProgress = 1;
      state.mambaAccumTermProgress = 1;
      state.moeAccumTermProgress = 1;
      lastStepperTickMs = null;
      lastMambaTemporalTickMs = null;
      lastMoeCycleTickMs = null;
      if (spineWaveClock.progress !== null) {
        spineWaveLastTickMs = null;
        const settled = startSpineWavePlayback(state.spineWaveMode, true);
        applySpineWaveStateToScene(settled);
        lastSpineWaveGeoKey = computeSpineWaveGeometryKey(settled.progress, settled.mode);
      }
    }
    geometryDirty = true;
    rebuildSceneIfNeeded();
    syncUi();
    scheduleDraw();
  };
  if (reducedMotionMq) {
    const onMotionChange = (ev: MediaQueryListEvent) => handleReducedMotionChange(ev.matches);
    if (typeof reducedMotionMq.addEventListener === 'function') {
      reducedMotionMq.addEventListener('change', onMotionChange);
    } else if (typeof reducedMotionMq.addListener === 'function') {
      reducedMotionMq.addListener(onMotionChange);
    }
  }

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      overlay.clearMetricCache();
      lastOverlayRenderKey = '';
      scheduleDraw();
    });
  }

  // --- WebGL Context Lost / Restored (PR-10) ---
  canvas.addEventListener('webglcontextlost', (ev) => {
    ev.preventDefault();
    cancelFly();
    cancelPendingHover();
    if (pendingRafId !== null) {
      window.cancelAnimationFrame(pendingRafId);
      pendingRafId = null;
    }
    state.gqaStepperPlaying = false;
    state.mambaStepperPlaying = false;
    state.moeStepperPlaying = false;
    state.mambaTemporalPlaying = false;
    state.moeTokenCycling = false;
    lastStepperTickMs = null;
    lastMambaTemporalTickMs = null;
    lastMoeCycleTickMs = null;
    if (spineWaveClock.progress !== null) {
      applySpineWaveStateToScene({ ...spineWaveClock, paused: true });
    }
    spineWaveLastTickMs = null;
    lastUploadedAtlasKey = '';
    lastOverlayRenderKey = '';
    if (device) device.markLost(true);
    syncUi();
    showFallbackCard(
      fallbackEl,
      'WebGL2 Context Lost',
      'Waiting for the browser to restore the graphics context. The guided walkthrough, sidebar controls, and interactive 1 → 1M memory chart remain active.',
      openMemoryChartFromFallback,
      true,
    );
  });

  canvas.addEventListener('webglcontextrestored', () => {
    if (!device) return;
    try {
      const nextSolidsPass = new SolidsPass(device.gl);
      const nextSheetsPass = new SheetsPass(device.gl);
      const nextLinesPass = new LinesPass(device.gl);
      const nextRibbonsPass = new RibbonsPass(device.gl);
      const nextFlatFlowPass = new FlatFlowRibbonsPass(device.gl);
      solidsPass = nextSolidsPass;
      sheetsPass = nextSheetsPass;
      linesPass = nextLinesPass;
      ribbonsPass = nextRibbonsPass;
      flatFlowPass = nextFlatFlowPass;
      lastUploadedAtlasKey = '';
      lastOverlayRenderKey = '';
      device.markLost(false);
      hideFallbackMessage(fallbackEl);
      geometryDirty = true;
      drawNow();
    } catch {
      device.markLost(true);
      lastUploadedAtlasKey = '';
      showFallbackCard(
        fallbackEl,
        'WebGL2 Context Recovery Failed',
        'The 3D graphics context could not be restored. Sidebar controls, guided tour text, and the 1 → 1M memory chart remain active.',
        openMemoryChartFromFallback,
        true,
      );
    }
  });

  // --- Free 3D Perspective Orbit, Pan, Zoom & Ray Picking ---
  canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());

  const activePointers = new Map<number, { x: number; y: number }>();
  let primaryPointerId: number | null = null;
  let dragMode: 'orbit' | 'pan' = 'orbit';
  let downX = 0;
  let downY = 0;
  let lastX = 0;
  let lastY = 0;
  let isDragging = false;
  let pendingSelectionCancelled = false;

  let pinchStartDist = 0;
  let pinchStartCamDist = initialPose.distance;

  const pickAtClient = (clientX: number, clientY: number): PickHit | null => {
    const rect = host.getBoundingClientRect();
    const liveMetrics = makeViewportMetrics(
      rect.left,
      rect.top,
      rect.width,
      rect.height,
      window.devicePixelRatio || 1,
    );
    if (state.mode === 'tower') {
      if (!currentTowerScene) return null;
      const snap = buildCameraSnapshot(state.camera, currentTowerScene.fitBounds, liveMetrics);
      const ray = clientPointToRay(clientX, clientY, snap);
      return pickMacroBay(ray, currentTowerScene.solidPickBoxes, currentTowerScene.fasciaPickBoxes);
    }
    if (!currentCutawayScene) return null;
    const snap = buildCameraSnapshot(state.camera, currentCutawayScene.fitBounds, liveMetrics);
    const ray = clientPointToRay(clientX, clientY, snap);
    return pickMacroBay(ray, currentCutawayScene.solidPickBoxes, []);
  };

  canvas.addEventListener('pointerdown', (ev) => {
    cancelPendingHover();
    canvas.setPointerCapture(ev.pointerId);
    activePointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });

    if (activePointers.size === 1) {
      primaryPointerId = ev.pointerId;
      dragMode = ev.button === 2 || ev.button === 1 || ev.shiftKey ? 'pan' : 'orbit';
      downX = ev.clientX;
      downY = ev.clientY;
      lastX = ev.clientX;
      lastY = ev.clientY;
      isDragging = false;
      pendingSelectionCancelled = ev.button !== 0;
    } else if (activePointers.size === 2) {
      markManualCameraControl();
      pendingSelectionCancelled = true;
      isDragging = true;
      const pts = Array.from(activePointers.values());
      pinchStartDist = Math.max(1, Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y));
      pinchStartCamDist = state.camera.distance;
    }
  });

  const processHoverAt = (clientX: number, clientY: number) => {
    const hit = pickAtClient(clientX, clientY);
    canvas.style.cursor = hit !== null ? 'pointer' : '';
    if (state.mode === 'tower') {
      const hitBay = hit ? hit.bayId : null;
      if (hitBay !== state.hoveredBay) {
        state.hoveredBay = hitBay;
        markDirtyAndSchedule(true);
      }
    } else if (state.mode === 'cutaway_embedding') {
      const payload = activeEmbeddingPayload();
      const hitPart = hit?.partId ?? null;
      const hitCell = hit ? pickHitToEmbeddingCell(hit, payload.tokenIds) : null;
      if (hitPart !== state.hoveredPart || !sameEmbeddingCell(hitCell, embeddingHoveredCell)) {
        state.hoveredPart = hitPart;
        embeddingHoveredCell = hitCell;
        markDirtyAndSchedule(true);
      }
    } else if (state.mode === 'cutaway_output') {
      const payload = activeOutputPayload();
      const focusTok = clampedOutputFocusToken(payload.tokenLabels.length);
      const hitPart = hit?.partId ?? null;
      const hitCell = hit ? pickHitToOutputCell(hit, focusTok) : null;
      if (hitPart !== state.hoveredPart || !sameOutputCell(hitCell, outputHoveredCell)) {
        state.hoveredPart = hitPart;
        outputHoveredCell = hitCell;
        markDirtyAndSchedule(true);
      }
    } else if (state.mode === 'cutaway_gqa') {
      const hitPart =
        hit?.headIndex !== undefined ? `q-head-${hit.headIndex}` : hit?.partId ?? null;
      const hitCell = hit
        ? pickHitToGqaCell(hit, state.gqaActiveGroup0Head, state.gqaActiveGroup1Head)
        : null;
      if (hitPart !== state.hoveredPart || !sameGqaCell(hitCell, state.gqaHoveredCell)) {
        state.hoveredPart = hitPart;
        state.gqaHoveredCell = hitCell;
        markDirtyAndSchedule(true);
      }
    } else if (state.mode === 'cutaway_mamba') {
      const hitPart =
        hit?.partId === 'ssm-state'
          ? 'ssm-state'
          : hit?.headIndex !== undefined && hit.tensorKey === 'x'
            ? `mamba-head-${hit.headIndex}`
            : hit?.partId ?? null;
      const cutT = activeCutawayTokens();
      const rawHitCell = hit
        ? pickHitToMambaCell(hit, state.mambaActiveHead, cutT)
        : null;
      const resolvedHover = resolveMambaHoverDuringPlayback(
        rawHitCell,
        state.mambaTemporalPlaying,
        activeMambaStateToken(),
        cutT,
      );
      const hitCell = resolvedHover.hoveredCell;
      const pausePlayback = resolvedHover.shouldPauseTemporalPlayback;
      if (
        pausePlayback ||
        hitPart !== state.hoveredPart ||
        !sameMambaCell(hitCell, state.mambaHoveredCell)
      ) {
        if (pausePlayback) {
          state.mambaTemporalPlaying = false;
          lastMambaTemporalTickMs = null;
        }
        state.hoveredPart = hitPart;
        state.mambaHoveredCell = hitCell;
        markDirtyAndSchedule(true);
      }
    } else if (state.mode === 'cutaway_moe') {
      const sim = currentCutawayScene?.moeSim ?? activeMoeSim();
      const tok = activeMoeFocusToken();
      const exp = activeMoeFocusExpert(sim, tok);
      const pickTok =
        hit?.tokenIndex !== undefined
          ? Math.max(0, Math.min(sim.tokens - 1, hit.tokenIndex))
          : tok;
      const topkSlice = sim.topk_indices.subarray(pickTok * sim.topK, (pickTok + 1) * sim.topK);
      const rawHitCell = hit ? pickHitToMoeCell(hit, tok, exp, topkSlice) : null;
      const resolvedHover = resolveMoeHoverDuringCycling(
        rawHitCell,
        state.moeTokenCycling,
        tok,
        sim,
      );
      const hitCell = resolvedHover.hoveredCell;
      const pauseCycling = resolvedHover.shouldPauseTokenCycling;
      const hitPart =
        (hit?.tensorKey === 'router_bank' || hit?.partId === 'routed-branch') &&
        hit.headIndex !== undefined
          ? `moe-expert-${hit.headIndex}`
          : hit?.partId ?? null;
      if (
        pauseCycling ||
        hitPart !== state.hoveredPart ||
        !sameMoeCell(hitCell, state.moeHoveredCell)
      ) {
        if (pauseCycling) {
          state.moeTokenCycling = false;
          lastMoeCycleTickMs = null;
        }
        state.hoveredPart = hitPart;
        state.moeHoveredCell = hitCell;
        markDirtyAndSchedule(true);
      }
    } else {
      const hitPart = hit?.partId ?? null;
      if (hitPart !== state.hoveredPart) {
        state.hoveredPart = hitPart;
        markDirtyAndSchedule(true);
      }
    }
  };

  canvas.addEventListener('pointermove', (ev) => {
    if (activePointers.has(ev.pointerId)) {
      activePointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    }

    if (activePointers.size === 2) {
      cancelPendingHover();
      const pts = Array.from(activePointers.values());
      const dist = Math.max(1, Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y));
      const scale = pinchStartDist / dist;
      state.camera = zoomPerspectiveDistance(
        { ...state.camera, distance: pinchStartCamDist },
        scale,
      );
      scheduleDraw();
      return;
    }

    if (primaryPointerId === ev.pointerId && activePointers.size === 1) {
      cancelPendingHover();
      const totalDist = Math.hypot(ev.clientX - downX, ev.clientY - downY);
      if (!isDragging && totalDist >= DRAG_THRESHOLD_PX) {
        isDragging = true;
        canvas.style.cursor = 'grabbing';
        markManualCameraControl();
      }
      if (isDragging && currentSnapshot) {
        const dx = ev.clientX - lastX;
        const dy = ev.clientY - lastY;
        lastX = ev.clientX;
        lastY = ev.clientY;
        if (dragMode === 'pan' || ev.shiftKey) {
          state.camera = panCameraByCssDelta(state.camera, currentSnapshot, dx, dy);
        } else {
          state.camera = orbitCameraByCssDelta(state.camera, dx, dy);
        }
        scheduleDraw();
      }
      return;
    }

    if (activePointers.size === 0 && ev.pointerType !== 'touch') {
      latestHoverClientX = ev.clientX;
      latestHoverClientY = ev.clientY;
      if (pendingHoverRafId === null) {
        pendingHoverRafId = window.requestAnimationFrame(() => {
          pendingHoverRafId = null;
          if (activePointers.size > 0 || isDragging) return;
          processHoverAt(latestHoverClientX, latestHoverClientY);
        });
      }
    }
  });

  const endPointer = (ev: PointerEvent, cancelled: boolean) => {
    const wasPrimary = primaryPointerId === ev.pointerId;
    activePointers.delete(ev.pointerId);
    if (canvas.hasPointerCapture(ev.pointerId)) {
      canvas.releasePointerCapture(ev.pointerId);
    }

    if (wasPrimary && !cancelled && !isDragging && !pendingSelectionCancelled) {
      const hit = pickAtClient(ev.clientX, ev.clientY);
      if (state.mode === 'tower') {
        const hitBay = hit ? hit.bayId : null;
        if (hitBay !== null) markTourExploring();
        state.selectedBay = hitBay;
        state.hoveredBay = hitBay;
      } else if (state.mode === 'cutaway_embedding') {
        const payload = activeEmbeddingPayload();
        const hitPart = hit?.partId ?? null;
        const hitCell = hit ? pickHitToEmbeddingCell(hit, payload.tokenIds) : null;
        if (hitPart !== null || hitCell !== null) markTourExploring();
        if (hitCell?.tokenIndex !== undefined) {
          state.embeddingFocusToken = Math.max(0, Math.min(payload.tokenIds.length - 1, hitCell.tokenIndex));
        }
        state.selectedPart = hitPart;
        state.hoveredPart = hitPart;
        state.embeddingActiveCell = hitCell;
        embeddingHoveredCell = hitCell;
      } else if (state.mode === 'cutaway_output') {
        const payload = activeOutputPayload();
        const focusTok = clampedOutputFocusToken(payload.tokenLabels.length);
        const candId = parseOutputCandidatePartId(hit?.partId);
        if (candId !== null && state.executionMode === 'ar_decode') {
          markTourExploring();
          const result = applyDecodeActionToActiveSession(
            decodeSessions,
            state.tourActive,
            state.tourBeat,
            hybridModel,
            state.precision,
            { type: 'step_token', tokenId: candId },
          );
          decodeSessions = result.store;
          const activeStep = getActiveDecodeStep(result.session);
          state.tokens = activeStep.totalTokens;
          state.embeddingFocusToken = Math.max(0, activeStep.tokens - 1);
          state.outputFocusToken = Math.max(0, activeStep.tokens - 1);
          state.selectedPart = `output-cand-${candId}`;
          state.hoveredPart = state.selectedPart;
          state.outputActiveCell = null;
          outputHoveredCell = null;
        } else {
          const hitPart = hit?.partId ?? null;
          const hitCell = hit ? pickHitToOutputCell(hit, focusTok) : null;
          if (hitPart !== null || hitCell !== null) markTourExploring();
          if (hitCell?.tokenIndex !== undefined) {
            state.outputFocusToken = Math.max(0, Math.min(payload.tokenLabels.length - 1, hitCell.tokenIndex));
          }
          state.selectedPart = hitPart;
          state.hoveredPart = hitPart;
          state.outputActiveCell = hitCell;
          outputHoveredCell = hitCell;
        }
      } else if (state.mode === 'cutaway_gqa') {
        const hitPart =
          hit?.headIndex !== undefined ? `q-head-${hit.headIndex}` : hit?.partId ?? null;
        if (hitPart !== null) markTourExploring();
        applyHeadSelectionToGroupHeads(hitPart);
        const hitCell = hit
          ? pickHitToGqaCell(hit, state.gqaActiveGroup0Head, state.gqaActiveGroup1Head)
          : null;
        // Part 0 #1: Synchronize displayed KV group head to clicked cell's owning head (e.g. V_out/Concat Q19)
        const cellHead = inferGqaCellHead(hitCell);
        if (cellHead !== null) {
          if (cellHead < 16) state.gqaActiveGroup0Head = cellHead;
          else state.gqaActiveGroup1Head = cellHead;
        }
        state.selectedPart = hitPart;
        state.hoveredPart = hitPart;
        state.gqaSelectedCell = hitCell;
        state.gqaHoveredCell = hitCell;
        if (hitCell !== null) {
          state.gqaStepperPlaying = false;
          state.gqaStepperActive = false;
          lastStepperTickMs = null;
        }
      } else if (state.mode === 'cutaway_mamba') {
        const hitPart =
          hit?.partId === 'ssm-state' && hit.headIndex !== undefined
            ? `mamba-head-${hit.headIndex}`
            : hit?.headIndex !== undefined && hit.tensorKey === 'x'
              ? `mamba-head-${hit.headIndex}`
              : hit?.partId ?? null;
        if (hitPart !== null) markTourExploring();
        applyHeadSelectionToGroupHeads(hitPart);
        const hitCell = hit
          ? pickHitToMambaCell(hit, state.mambaActiveHead, activeCutawayTokens())
          : null;
        const cellHead = inferMambaCellHead(hitCell);
        if (cellHead !== null) {
          state.mambaActiveHead = cellHead;
        } else if (hit?.headIndex !== undefined) {
          state.mambaActiveHead = hit.headIndex;
        }
        state.selectedPart = hitPart;
        state.hoveredPart = hitPart;
        state.mambaSelectedCell = hitCell;
        state.mambaHoveredCell = hitCell;
        if (hitCell !== null) {
          state.mambaStepperPlaying = false;
          state.mambaStepperActive = false;
          state.mambaTemporalPlaying = false;
          lastStepperTickMs = null;
          lastMambaTemporalTickMs = null;
        }
      } else if (state.mode === 'cutaway_moe') {
        const sim = currentCutawayScene?.moeSim ?? activeMoeSim();
        const tok = activeMoeFocusToken();
        const exp = activeMoeFocusExpert(sim, tok);
        const pickTok =
          hit?.tokenIndex !== undefined
            ? Math.max(0, Math.min(sim.tokens - 1, hit.tokenIndex))
            : tok;
        const topkSlice = sim.topk_indices.subarray(pickTok * sim.topK, (pickTok + 1) * sim.topK);
        const hitCell = hit ? pickHitToMoeCell(hit, tok, exp, topkSlice) : null;
        const hitPart =
          (hit?.tensorKey === 'router_bank' || hit?.partId === 'routed-branch') &&
          hit.headIndex !== undefined
            ? `moe-expert-${hit.headIndex}`
            : hit?.partId ?? null;
        if (hitPart !== null) markTourExploring();
        const cellExp = inferMoeCellExpert(hitCell, sim, exp);
        if (cellExp !== null) {
          state.moeActiveExpert = cellExp;
        } else if (hit?.headIndex !== undefined) {
          state.moeActiveExpert = hit.headIndex;
        }
        if (hitCell?.tokenIndex !== undefined) {
          state.moeFocusToken = Math.max(0, Math.min(sim.tokens - 1, hitCell.tokenIndex));
        }
        state.selectedPart = hitPart;
        state.hoveredPart = hitPart;
        state.moeSelectedCell = hitCell;
        state.moeHoveredCell = hitCell;
        if (hitCell !== null || hitPart !== null) {
          state.moeTokenCycling = false;
          lastMoeCycleTickMs = null;
        }
        if (hitCell !== null) {
          state.moeStepperPlaying = false;
          state.moeStepperActive = false;
          lastStepperTickMs = null;
        }
      } else {
        const hitPart = hit?.partId ?? null;
        if (hitPart !== null) markTourExploring();
        state.selectedPart = hitPart;
        state.hoveredPart = hitPart;
      }
      markDirtyAndSchedule(true);
    }

    if (activePointers.size === 0) {
      primaryPointerId = null;
      isDragging = false;
      pendingSelectionCancelled = false;
      canvas.style.cursor = '';
    }
  };

  canvas.addEventListener('pointerup', (ev) => endPointer(ev, false));
  canvas.addEventListener('pointercancel', (ev) => endPointer(ev, true));

  canvas.addEventListener('dblclick', (ev) => {
    if (state.mode !== 'tower') return;
    const hit = pickAtClient(ev.clientX, ev.clientY);
    if (hit) {
      markTourExploring();
      state.selectedBay = hit.bayId;
      state.hoveredBay = hit.bayId;
      zoomToBay(hit.bayId);
      markDirtyAndSchedule(true);
    }
  });

  canvas.addEventListener('pointerleave', () => {
    cancelPendingHover();
    canvas.style.cursor = '';
    if (
      state.hoveredBay !== null ||
      state.hoveredPart !== null ||
      state.gqaHoveredCell !== null ||
      state.mambaHoveredCell !== null ||
      state.moeHoveredCell !== null ||
      embeddingHoveredCell !== null ||
      outputHoveredCell !== null
    ) {
      state.hoveredBay = null;
      state.hoveredPart = null;
      state.gqaHoveredCell = null;
      state.mambaHoveredCell = null;
      state.moeHoveredCell = null;
      embeddingHoveredCell = null;
      outputHoveredCell = null;
      markDirtyAndSchedule(true);
    }
  });

  window.addEventListener('pagehide', cancelPendingHover);

  canvas.addEventListener(
    'wheel',
    (ev) => {
      ev.preventDefault();
      markManualCameraControl();
      let deltaPx = ev.deltaY;
      if (ev.deltaMode === WheelEvent.DOM_DELTA_LINE) deltaPx *= 16;
      else if (ev.deltaMode === WheelEvent.DOM_DELTA_PAGE) deltaPx *= currentMetrics.innerHeight;
      const scale = Math.exp(0.0012 * deltaPx);
      state.camera = zoomPerspectiveDistance(state.camera, scale);
      scheduleDraw();
    },
    { passive: false },
  );

  // Keyboard shortcuts (ignored when focus is inside an input/textarea/select,
  // and deferred on Space/Enter for focused interactive controls — RB-3)
  window.addEventListener('keydown', (ev) => {
    const target = ev.target as HTMLElement | null;
    const key = ev.key;
    if (
      target &&
      (target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable)
    ) {
      if (key === 'Escape' && (state.shortcutsOpen || state.memoryChartOpen)) {
        ev.preventDefault();
        if (state.shortcutsOpen) state.shortcutsOpen = false;
        else state.memoryChartOpen = false;
        syncUi();
      }
      return;
    }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (shouldDeferGlobalShortcut(key, target)) return;

    if (key === '?') {
      ev.preventDefault();
      state.shortcutsOpen = !state.shortcutsOpen;
      syncUi();
      return;
    }

    if (key === ' ') {
      if (state.tourActive) {
        ev.preventDefault();
        const targetBeat = ev.shiftKey
          ? prevBeatIndex(state.tourBeat)
          : nextBeatIndex(state.tourBeat);
        if (targetBeat !== null) applyTourBeat(targetBeat);
        return;
      }
      if (
        state.mode === 'cutaway_gqa' ||
        state.mode === 'cutaway_mamba' ||
        state.mode === 'cutaway_moe' ||
        state.mode === 'cutaway_embedding' ||
        state.mode === 'cutaway_output'
      ) {
        ev.preventDefault();
        toggleCutawayStepperPlay(false);
        return;
      }
    }

    if (key === 'Escape') {
      if (controls.closeFormulaNotePopover()) {
        ev.preventDefault();
      } else if (state.shortcutsOpen) {
        ev.preventDefault();
        state.shortcutsOpen = false;
        syncUi();
      } else if (state.memoryChartOpen) {
        ev.preventDefault();
        state.memoryChartOpen = false;
        syncUi();
      } else if (
        state.gqaSelectedCell !== null ||
        state.gqaStepperActive ||
        state.mambaSelectedCell !== null ||
        state.mambaStepperActive ||
        state.mambaTemporalPlaying ||
        state.moeSelectedCell !== null ||
        state.moeStepperActive ||
        state.moeTokenCycling ||
        state.embeddingActiveCell !== null ||
        state.outputActiveCell !== null ||
        bookendStepIndex !== null
      ) {
        ev.preventDefault();
        const hadStepper =
          state.gqaStepperActive ||
          state.mambaStepperActive ||
          state.moeStepperActive ||
          bookendStepIndex !== null;
        state.gqaSelectedCell = null;
        state.gqaHoveredCell = null;
        state.gqaStepperActive = false;
        state.gqaStepperPlaying = false;
        state.gqaAccumTermProgress = 1;
        state.mambaSelectedCell = null;
        state.mambaHoveredCell = null;
        state.mambaStepperActive = false;
        state.mambaStepperPlaying = false;
        state.mambaTemporalPlaying = false;
        state.mambaAccumTermProgress = 1;
        lastMambaTemporalTickMs = null;
        state.moeSelectedCell = null;
        state.moeHoveredCell = null;
        state.moeStepperActive = false;
        state.moeStepperPlaying = false;
        state.moeTokenCycling = false;
        state.moeAccumTermProgress = 1;
        state.embeddingActiveCell = null;
        state.outputActiveCell = null;
        embeddingHoveredCell = null;
        outputHoveredCell = null;
        bookendStepIndex = null;
        lastStepperTickMs = null;
        lastStepperCellKey = '';
        lastMoeCycleTickMs = null;
        manualCameraOverride = false;
        lastAutoFollowedStepKey = '';
        geometryDirty = true;
        rebuildSceneIfNeeded();
        syncUi();
        if (state.mode !== 'tower' && (hadStepper || state.stepperAutoFollow)) {
          fitActiveView();
        }
        scheduleDraw();
      } else if (state.mode === 'tower' && state.spineWaveProgress !== null) {
        ev.preventDefault();
        clearSpineWave(true);
        markDirtyAndSchedule(true);
      } else if (state.selectedPart !== null || state.hoveredPart !== null) {
        ev.preventDefault();
        state.selectedPart = null;
        state.hoveredPart = null;
        markDirtyAndSchedule(true);
      } else if (state.mode !== 'tower') {
        ev.preventDefault();
        selectMode('tower');
      } else if (state.selectedBay !== null || state.hoveredBay !== null) {
        ev.preventDefault();
        state.selectedBay = null;
        state.hoveredBay = null;
        markDirtyAndSchedule(true);
      }
      return;
    }

    if (key === '1') {
      ev.preventDefault();
      selectMode('tower');
      return;
    }
    if (key === '2') {
      ev.preventDefault();
      selectMode('cutaway_mamba');
      return;
    }
    if (key === '3') {
      ev.preventDefault();
      selectMode('cutaway_moe');
      return;
    }
    if (key === '4') {
      ev.preventDefault();
      selectMode('cutaway_gqa');
      return;
    }
    if (key === '5') {
      ev.preventDefault();
      selectMode('cutaway_embedding');
      return;
    }
    if (key === '6') {
      ev.preventDefault();
      selectMode('cutaway_output');
      return;
    }
    if (key === 'p' || key === 'P') {
      ev.preventDefault();
      markTourExploring();
      state.precision = state.precision === 'bf16' ? 'fp8_selective' : 'bf16';
      markDirtyAndSchedule(true);
      return;
    }
    if (key === 'm' || key === 'M') {
      ev.preventDefault();
      state.memoryChartOpen = !state.memoryChartOpen;
      if (state.memoryChartOpen) state.shortcutsOpen = false;
      syncUi();
      return;
    }
    if (key === 'r' || key === 'R') {
      ev.preventDefault();
      resetAll();
      return;
    }

    if (key === 'ArrowRight') {
      if (state.tourActive) {
        ev.preventDefault();
        const next = nextBeatIndex(state.tourBeat);
        if (next !== null) applyTourBeat(next);
      } else if (state.mode === 'tower') {
        ev.preventDefault();
        stepBay(1);
      }
      return;
    }

    if (key === 'ArrowLeft') {
      if (state.tourActive) {
        ev.preventDefault();
        const prev = prevBeatIndex(state.tourBeat);
        if (prev !== null) applyTourBeat(prev);
      } else if (state.mode === 'tower') {
        ev.preventDefault();
        stepBay(-1);
      }
      return;
    }

    if (state.mode === 'tower' && !state.tourActive) {
      if (key === 'ArrowDown') {
        ev.preventDefault();
        stepBay(1);
      } else if (key === 'ArrowUp') {
        ev.preventDefault();
        stepBay(-1);
      } else if (key === 'Enter' && state.selectedBay !== null) {
        ev.preventDefault();
        zoomToBay(state.selectedBay);
      }
    }
  });

  void TOWER_FIT_BOUNDS;

  rebuildSceneIfNeeded();
  syncUi();
  drawNow();
}

if (typeof document !== 'undefined' && document.getElementById('stage-host')) {
  boot();
}
