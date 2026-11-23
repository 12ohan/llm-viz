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
  type GqaViewAngle,
  type SceneState,
  type TocSectionId,
  type ViewMode,
} from './model/state';
import {
  buildCameraSnapshot,
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
import { RenderDevice } from './render/device';
import { LinesPass } from './render/lines';
import { clientPointToRay, pickMacroBay, type PickHit } from './render/pick';
import { FlatFlowRibbonsPass, RibbonsPass } from './render/ribbons';
import { SheetsPass } from './render/sheets';
import { SolidsPass } from './render/solids';
import {
  buildGqaCutaway,
  buildMambaCutaway,
  buildMoECutaway,
  type CutawaySceneGeometry,
} from './scene/cutaways';
import {
  boundsForBay,
  boundsForSection,
  buildMacroSpine,
  TOWER_FIT_BOUNDS,
  type MacroSceneGeometry,
} from './scene/spine';
import { MacroControls, queryControlsElements } from './ui/controls';
import { MemoryScalingDrawer } from './ui/memoryChart';
import { MacroOverlay } from './ui/overlay';
import {
  getWalkthroughChapter,
  nextChapterIndex,
  prevChapterIndex,
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

function cameraPoseForCutaway(
  mode: Exclude<ViewMode, 'tower'>,
  bounds: AABB,
  metrics: ViewportMetrics,
  gqaViewAngle: GqaViewAngle = 'three_quarter',
): CameraPose {
  if (mode === 'cutaway_mamba') {
    return fitPerspectiveToBounds(
      bounds, 24, 20, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.88,
    );
  }
  if (mode === 'cutaway_moe') {
    return fitPerspectiveToBounds(
      bounds, 26, 28, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.88,
    );
  }
  // cutaway_gqa: XZ drafting-table sheet schematic with consistent top-to-bottom orientation
  if (gqaViewAngle === 'near_plan') {
    return fitPerspectiveToBounds(
      bounds, 1.5, 78, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.85,
    );
  }
  return fitPerspectiveToBounds(
    bounds, 9, 50, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.98,
  );
}

interface FlyAnimation {
  readonly from: CameraPose;
  readonly to: CameraPose;
  readonly startMs: number;
  readonly durationMs: number;
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
  const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  const state: SceneState = createInitialState(initialPose, reducedMotion);

  let currentMetrics: ViewportMetrics = initialMetrics;
  let currentTowerScene: MacroSceneGeometry | null = null;
  let currentCutawayScene: CutawaySceneGeometry | null = null;
  let currentSnapshot: CameraSnapshot | null = null;
  let geometryDirty = true;
  let pendingRafId: number | null = null;
  let flyAnim: FlyAnimation | null = null;

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
      `Your browser could not initialize a WebGL2 context (${detail}). The 7-chapter architectural walkthrough, token scrubber, precision toggle, and interactive 1 → 1M memory scaling chart remain fully available.`,
      openMemoryChartFromFallback,
      true,
    );
  }

  const overlay = new MacroOverlay(overlaySvg, overlayLabels);
  const controlEls = queryControlsElements();

  const cancelFly = () => {
    flyAnim = null;
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

  const buildActiveCutaway = (): CutawaySceneGeometry | null => {
    if (state.mode === 'cutaway_mamba') {
      return buildMambaCutaway({
        bayId: state.mambaBayId,
        tokens: state.tokens,
        precision: state.precision,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
      });
    }
    if (state.mode === 'cutaway_moe') {
      return buildMoECutaway({
        tokens: state.tokens,
        precision: state.precision,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
      });
    }
    if (state.mode === 'cutaway_gqa') {
      return buildGqaCutaway({
        bayId: state.gqaBayId,
        tokens: state.tokens,
        precision: state.precision,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
      });
    }
    return null;
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
      });
      currentCutawayScene = null;
      if (device && !device.isLost && solidsPass && linesPass && ribbonsPass) {
        solidsPass.upload(currentTowerScene.solidData, currentTowerScene.solidCount);
        linesPass.upload(currentTowerScene.lineData, currentTowerScene.lineCount);
        ribbonsPass.upload(currentTowerScene.ribbonData, currentTowerScene.ribbonCount);
      }
    } else {
      currentCutawayScene = buildActiveCutaway();
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
          sheetsPass.upload(currentCutawayScene.sheetData, currentCutawayScene.sheetCount);
          linesPass.upload(currentCutawayScene.lineData, currentCutawayScene.lineCount);
          flatFlowPass.upload(
            currentCutawayScene.flowVertices ?? new Float32Array(0),
            currentCutawayScene.flowIndices ?? new Uint32Array(0),
          );
        } else {
          solidsPass.upload(currentCutawayScene.solidData, currentCutawayScene.solidCount);
          linesPass.upload(currentCutawayScene.lineData, currentCutawayScene.lineCount);
          ribbonsPass.upload(currentCutawayScene.ribbonData, currentCutawayScene.ribbonCount);
        }
      }
    }
    geometryDirty = false;
  };

  const syncUi = () => {
    controls.sync(state, currentCutawayScene);
    memoryDrawer.sync(state.memoryChartOpen, state.precision, state.memoryProbeTokens);
  };

  const markDirtyAndSchedule = (rebuildGeometry: boolean) => {
    if (rebuildGeometry) {
      geometryDirty = true;
      rebuildSceneIfNeeded();
    }
    syncUi();
    scheduleDraw();
  };

  const applyTourChapter = (chapterIdx: number) => {
    const ch = getWalkthroughChapter(chapterIdx);
    if (state.mode === 'tower' && ch.mode !== 'tower') {
      state.savedMacroCamera = cloneCameraPose(state.camera);
    }
    state.tourActive = true;
    state.tourChapter = ch.index;
    state.precision = ch.precision;
    state.tokens = ch.tokens;
    state.section = ch.section;
    state.selectedBay = ch.selectedBay;
    state.hoveredBay = null;
    state.selectedPart = null;
    state.hoveredPart = null;
    if (ch.mambaBayId) state.mambaBayId = ch.mambaBayId;
    if (ch.gqaBayId) state.gqaBayId = ch.gqaBayId;
    state.mode = ch.mode;

    geometryDirty = true;
    rebuildSceneIfNeeded();

    if (ch.mode === 'tower') {
      flyToPose(cameraPoseForSection(ch.section, currentMetrics));
    } else if (currentCutawayScene) {
      flyToPose(
        cameraPoseForCutaway(ch.mode, currentCutawayScene.fitBounds, currentMetrics, state.gqaViewAngle),
      );
    }

    syncUi();
    scheduleDraw();
  };

  const exitTourToFreeExplore = () => {
    if (!state.tourActive) return;
    state.tourActive = false;
    syncUi();
  };

  const selectMode = (nextMode: ViewMode, customPose?: CameraPose, keepTour = false) => {
    if (!keepTour) state.tourActive = false;
    if (state.mode === 'tower' && nextMode !== 'tower') {
      state.savedMacroCamera = cloneCameraPose(state.camera);
    }
    const modeChanged = state.mode !== nextMode;
    state.mode = nextMode;
    state.hoveredBay = null;
    state.hoveredPart = null;
    if (modeChanged) {
      state.selectedPart = null;
    }
    if (nextMode === 'cutaway_mamba') state.section = 'mamba';
    else if (nextMode === 'cutaway_moe') state.section = 'moe';
    else if (nextMode === 'cutaway_gqa') state.section = 'gqa';

    geometryDirty = true;
    rebuildSceneIfNeeded();

    if (customPose) {
      flyToPose(customPose);
    } else if (nextMode === 'tower') {
      flyToPose(state.savedMacroCamera);
    } else if (currentCutawayScene) {
      flyToPose(
        cameraPoseForCutaway(nextMode, currentCutawayScene.fitBounds, currentMetrics, state.gqaViewAngle),
      );
    }

    syncUi();
    scheduleDraw();
  };

  const selectSection = (section: TocSectionId) => {
    state.tourActive = false;
    // If already in a cutaway and user clicks Mamba/MoE/GQA in TOC, switch cutaways cleanly!
    if (state.mode !== 'tower') {
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
    if (state.mode === 'tower') {
      state.section = 'overview';
      flyToPose(cameraPoseForSection('overview', currentMetrics));
      syncUi();
    } else if (currentCutawayScene) {
      flyToPose(
        cameraPoseForCutaway(state.mode, currentCutawayScene.fitBounds, currentMetrics, state.gqaViewAngle),
      );
    }
  };

  const resetAll = () => {
    state.tokens = DEFAULT_TOKENS;
    state.precision = 'bf16';
    state.selectedBay = null;
    state.hoveredBay = null;
    state.selectedPart = null;
    state.hoveredPart = null;

    if (state.mode === 'tower') {
      state.section = 'overview';
      flyToPose(cameraPoseForSection('overview', currentMetrics));
      markDirtyAndSchedule(true);
    } else {
      geometryDirty = true;
      rebuildSceneIfNeeded();
      if (currentCutawayScene) {
        flyToPose(
          cameraPoseForCutaway(state.mode, currentCutawayScene.fitBounds, currentMetrics, state.gqaViewAngle),
        );
      }
      syncUi();
      scheduleDraw();
    }
  };

  const stepBay = (delta: -1 | 1) => {
    if (state.mode !== 'tower') return;
    state.tourActive = false;
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

  const controls = new MacroControls(NEMOTRON_SPEC, controlEls, {
    onSelectMode: (mode) => selectMode(mode),
    onSelectSection: selectSection,
    onSelectMambaBay: (bayId) => {
      state.tourActive = false;
      if (state.mambaBayId === bayId && state.mode === 'cutaway_mamba') return;
      state.mambaBayId = bayId;
      state.selectedPart = null;
      state.hoveredPart = null;
      markDirtyAndSchedule(true);
    },
    onSelectGqaBay: (bayId) => {
      state.tourActive = false;
      if (state.gqaBayId === bayId && state.mode === 'cutaway_gqa') return;
      state.gqaBayId = bayId;
      state.selectedPart = null;
      state.hoveredPart = null;
      markDirtyAndSchedule(true);
    },
    onSelectGqaView: (view: GqaViewAngle) => {
      if (state.gqaViewAngle === view && state.mode === 'cutaway_gqa') return;
      state.gqaViewAngle = view;
      if (state.mode === 'cutaway_gqa' && currentCutawayScene) {
        flyToPose(
          cameraPoseForCutaway('cutaway_gqa', currentCutawayScene.fitBounds, currentMetrics, view),
        );
      }
      syncUi();
    },
    onSelectPart: (partId: string | null) => {
      state.tourActive = false;
      state.selectedPart = partId;
      state.hoveredPart = null;
      markDirtyAndSchedule(true);
    },
    onInspectInternals: inspectInternals,
    onTokensChange: (nextTokens) => {
      const clamped = Math.max(MIN_TOKENS, Math.min(MAX_TOKENS, nextTokens));
      if (clamped === state.tokens) return;
      state.tokens = clamped;
      markDirtyAndSchedule(true);
    },
    onPrecisionChange: (mode: PrecisionMode) => {
      if (mode === state.precision) return;
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
      markDirtyAndSchedule(true);
    },
    onTourToggle: () => {
      if (state.tourActive) {
        exitTourToFreeExplore();
      } else {
        applyTourChapter(state.tourChapter);
      }
    },
    onTourPrev: () => {
      const prev = prevChapterIndex(state.tourChapter);
      if (prev !== null) applyTourChapter(prev);
    },
    onTourNext: () => {
      const next = nextChapterIndex(state.tourChapter);
      if (next !== null) applyTourChapter(next);
    },
    onTourRestart: () => applyTourChapter(0),
    onTourExit: exitTourToFreeExplore,
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
  });

  const drawFrame = (nowMs: number) => {
    pendingRafId = null;

    const measured = measureHostMetrics(host);
    if (!measured) return;
    currentMetrics = measured;

    let continueAnimating = false;
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

    rebuildSceneIfNeeded();

    const activeBounds =
      state.mode === 'tower'
        ? currentTowerScene!.fitBounds
        : currentCutawayScene!.fitBounds;

    currentSnapshot = buildCameraSnapshot(state.camera, activeBounds, currentMetrics);

    if (
      device &&
      !device.isLost &&
      solidsPass &&
      sheetsPass &&
      linesPass &&
      ribbonsPass &&
      flatFlowPass
    ) {
      const fb = device.ensureSize(currentMetrics);
      device.beginFrame(fb);
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
        flatFlowPass.draw(currentSnapshot.viewProj);
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
      device.endFrame(fb);
    }

    // Synchronous black-pill DOM/SVG overlay projection in the same animation frame
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
    } else if (currentCutawayScene) {
      overlay.updateCutaway({
        scene: currentCutawayScene,
        snapshot: currentSnapshot,
        tokens: state.tokens,
        selectedPart: state.selectedPart,
        hoveredPart: state.hoveredPart,
      });
    }

    if (continueAnimating) {
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
    currentMetrics = next;
    lastHostW = next.hostWidth;
    lastHostH = next.hostHeight;
    lastDpr = next.dpr;
    if (state.mode === 'tower' && state.section === 'overview' && !flyAnim && state.selectedBay === null) {
      state.camera = cameraPoseForSection('overview', next);
    } else if (state.mode !== 'tower' && !flyAnim && currentCutawayScene) {
      state.camera = cameraPoseForCutaway(
        state.mode,
        currentCutawayScene.fitBounds,
        next,
        state.gqaViewAngle,
      );
    }
    drawNow();
  };

  const resizeObserver = new ResizeObserver(() => handleResizeOrDpr());
  resizeObserver.observe(host);

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

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      overlay.clearMetricCache();
      scheduleDraw();
    });
  }

  // --- WebGL Context Lost / Restored ---
  canvas.addEventListener('webglcontextlost', (ev) => {
    ev.preventDefault();
    cancelFly();
    if (pendingRafId !== null) {
      window.cancelAnimationFrame(pendingRafId);
      pendingRafId = null;
    }
    if (device) device.markLost(true);
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
      device.markLost(false);
      solidsPass = new SolidsPass(device.gl);
      sheetsPass = new SheetsPass(device.gl);
      linesPass = new LinesPass(device.gl);
      ribbonsPass = new RibbonsPass(device.gl);
      flatFlowPass = new FlatFlowRibbonsPass(device.gl);
      hideFallbackMessage(fallbackEl);
      geometryDirty = true;
      drawNow();
    } catch {
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
      cancelFly();
      pendingSelectionCancelled = true;
      isDragging = true;
      const pts = Array.from(activePointers.values());
      pinchStartDist = Math.max(1, Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y));
      pinchStartCamDist = state.camera.distance;
    }
  });

  canvas.addEventListener('pointermove', (ev) => {
    if (activePointers.has(ev.pointerId)) {
      activePointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    }

    if (activePointers.size === 2) {
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
      const totalDist = Math.hypot(ev.clientX - downX, ev.clientY - downY);
      if (!isDragging && totalDist >= DRAG_THRESHOLD_PX) {
        isDragging = true;
        cancelFly();
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
      const hit = pickAtClient(ev.clientX, ev.clientY);
      if (state.mode === 'tower') {
        const hitBay = hit ? hit.bayId : null;
        if (hitBay !== state.hoveredBay) {
          state.hoveredBay = hitBay;
          markDirtyAndSchedule(true);
        }
      } else {
        const hitPart =
          hit?.headIndex !== undefined ? `q-head-${hit.headIndex}` : hit?.partId ?? null;
        if (hitPart !== state.hoveredPart) {
          state.hoveredPart = hitPart;
          markDirtyAndSchedule(true);
        }
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
        if (hitBay !== null) state.tourActive = false;
        state.selectedBay = hitBay;
        state.hoveredBay = hitBay;
      } else {
        const hitPart =
          hit?.headIndex !== undefined ? `q-head-${hit.headIndex}` : hit?.partId ?? null;
        if (hitPart !== null) state.tourActive = false;
        state.selectedPart = hitPart;
        state.hoveredPart = hitPart;
      }
      markDirtyAndSchedule(true);
    }

    if (activePointers.size === 0) {
      primaryPointerId = null;
      isDragging = false;
      pendingSelectionCancelled = false;
    }
  };

  canvas.addEventListener('pointerup', (ev) => endPointer(ev, false));
  canvas.addEventListener('pointercancel', (ev) => endPointer(ev, true));

  canvas.addEventListener('dblclick', (ev) => {
    if (state.mode !== 'tower') return;
    const hit = pickAtClient(ev.clientX, ev.clientY);
    if (hit) {
      state.selectedBay = hit.bayId;
      state.hoveredBay = hit.bayId;
      zoomToBay(hit.bayId);
      markDirtyAndSchedule(true);
    }
  });

  canvas.addEventListener('pointerleave', () => {
    if (state.hoveredBay !== null || state.hoveredPart !== null) {
      state.hoveredBay = null;
      state.hoveredPart = null;
      markDirtyAndSchedule(true);
    }
  });

  canvas.addEventListener(
    'wheel',
    (ev) => {
      ev.preventDefault();
      cancelFly();
      let deltaPx = ev.deltaY;
      if (ev.deltaMode === WheelEvent.DOM_DELTA_LINE) deltaPx *= 16;
      else if (ev.deltaMode === WheelEvent.DOM_DELTA_PAGE) deltaPx *= currentMetrics.innerHeight;
      const scale = Math.exp(0.0012 * deltaPx);
      state.camera = zoomPerspectiveDistance(state.camera, scale);
      scheduleDraw();
    },
    { passive: false },
  );

  // Keyboard shortcuts (ignored when focus is inside an input/textarea/select)
  window.addEventListener('keydown', (ev) => {
    const target = ev.target as HTMLElement | null;
    if (
      target &&
      (target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable)
    ) {
      return;
    }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;

    const key = ev.key;

    if (key === '?') {
      ev.preventDefault();
      state.shortcutsOpen = !state.shortcutsOpen;
      syncUi();
      return;
    }

    if (key === 'Escape') {
      if (state.shortcutsOpen) {
        ev.preventDefault();
        state.shortcutsOpen = false;
        syncUi();
      } else if (state.memoryChartOpen) {
        ev.preventDefault();
        state.memoryChartOpen = false;
        syncUi();
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
    if (key === 'p' || key === 'P') {
      ev.preventDefault();
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
        const next = nextChapterIndex(state.tourChapter);
        if (next !== null) applyTourChapter(next);
      } else if (state.mode === 'tower') {
        ev.preventDefault();
        stepBay(1);
      }
      return;
    }

    if (key === 'ArrowLeft') {
      if (state.tourActive) {
        ev.preventDefault();
        const prev = prevChapterIndex(state.tourChapter);
        if (prev !== null) applyTourChapter(prev);
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

boot();
