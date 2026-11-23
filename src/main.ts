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
import { RibbonsPass } from './render/ribbons';
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
import { MacroOverlay } from './ui/overlay';

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

function showFallbackMessage(el: HTMLElement, message: string): void {
  el.textContent = message;
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
  // cutaway_gqa: 3/4 front-left perspective revealing 32 stacked Q plates, 16x thinner K/V, and twin +Z KV stacks
  return fitPerspectiveToBounds(
    bounds, 30, 22, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.88,
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

  if (!host || !canvas || !overlaySvg || !overlayLabels || !fallbackEl) {
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
  let linesPass: LinesPass | null = null;
  let ribbonsPass: RibbonsPass | null = null;

  try {
    device = new RenderDevice(canvas);
    solidsPass = new SolidsPass(device.gl);
    linesPass = new LinesPass(device.gl);
    ribbonsPass = new RibbonsPass(device.gl);
  } catch (err) {
    const detail = err instanceof Error ? err.message : 'WebGL2 unavailable';
    showFallbackMessage(
      fallbackEl,
      `3D canvas unavailable (${detail}). Use the left sidebar contents, token scrubber, and precision toggle to inspect all 52 layers.`,
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
      if (currentCutawayScene && device && !device.isLost && solidsPass && linesPass && ribbonsPass) {
        solidsPass.upload(currentCutawayScene.solidData, currentCutawayScene.solidCount);
        linesPass.upload(currentCutawayScene.lineData, currentCutawayScene.lineCount);
        ribbonsPass.upload(currentCutawayScene.ribbonData, currentCutawayScene.ribbonCount);
      }
    }
    geometryDirty = false;
  };

  const markDirtyAndSchedule = (rebuildGeometry: boolean) => {
    if (rebuildGeometry) {
      geometryDirty = true;
      rebuildSceneIfNeeded();
    }
    controls.sync(state, currentCutawayScene);
    scheduleDraw();
  };

  const selectMode = (nextMode: ViewMode, customPose?: CameraPose) => {
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
      flyToPose(cameraPoseForCutaway(nextMode, currentCutawayScene.fitBounds, currentMetrics));
    }

    controls.sync(state, currentCutawayScene);
    scheduleDraw();
  };

  const selectSection = (section: TocSectionId) => {
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
      controls.sync(state, currentCutawayScene);
    } else if (currentCutawayScene) {
      flyToPose(cameraPoseForCutaway(state.mode, currentCutawayScene.fitBounds, currentMetrics));
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
        flyToPose(cameraPoseForCutaway(state.mode, currentCutawayScene.fitBounds, currentMetrics));
      }
      controls.sync(state, currentCutawayScene);
      scheduleDraw();
    }
  };

  const stepBay = (delta: -1 | 1) => {
    if (state.mode !== 'tower') return;
    if (state.selectedBay === null) {
      state.selectedBay = delta > 0 ? 1 : 52;
    } else {
      state.selectedBay = Math.max(1, Math.min(52, state.selectedBay + delta));
    }
    state.hoveredBay = null;
    markDirtyAndSchedule(true);
  };

  const controls = new MacroControls(NEMOTRON_SPEC, controlEls, {
    onSelectMode: (mode) => selectMode(mode),
    onSelectSection: selectSection,
    onSelectMambaBay: (bayId) => {
      if (state.mambaBayId === bayId && state.mode === 'cutaway_mamba') return;
      state.mambaBayId = bayId;
      state.selectedPart = null;
      state.hoveredPart = null;
      markDirtyAndSchedule(true);
    },
    onSelectGqaBay: (bayId) => {
      if (state.gqaBayId === bayId && state.mode === 'cutaway_gqa') return;
      state.gqaBayId = bayId;
      state.selectedPart = null;
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

    if (device && !device.isLost && solidsPass && linesPass && ribbonsPass) {
      const fb = device.ensureSize(currentMetrics);
      device.beginFrame(fb);
      solidsPass.draw(currentSnapshot.viewProj);
      linesPass.draw(
        currentSnapshot.viewProj,
        fb.vpWidth,
        fb.vpHeight,
        currentMetrics.dpr,
        currentSnapshot.eye,
      );
      ribbonsPass.draw(currentSnapshot.viewProj);
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
      state.camera = cameraPoseForCutaway(state.mode, currentCutawayScene.fitBounds, next);
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
    showFallbackMessage(
      fallbackEl,
      'WebGL2 context lost. Waiting for restoration; sidebar controls remain active.',
    );
  });

  canvas.addEventListener('webglcontextrestored', () => {
    if (!device) return;
    try {
      device.markLost(false);
      solidsPass = new SolidsPass(device.gl);
      linesPass = new LinesPass(device.gl);
      ribbonsPass = new RibbonsPass(device.gl);
      hideFallbackMessage(fallbackEl);
      geometryDirty = true;
      drawNow();
    } catch {
      showFallbackMessage(
        fallbackEl,
        'WebGL2 context could not be restored. Sidebar controls and inspector remain active.',
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
        const hitPart = hit?.partId ?? null;
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
        state.selectedBay = hitBay;
        state.hoveredBay = hitBay;
      } else {
        const hitPart = hit?.partId ?? null;
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

  // Keyboard shortcuts
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

    if (state.mode === 'tower') {
      if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') {
        ev.preventDefault();
        stepBay(1);
      } else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') {
        ev.preventDefault();
        stepBay(-1);
      } else if (ev.key === 'Enter' && state.selectedBay !== null) {
        ev.preventDefault();
        zoomToBay(state.selectedBay);
      }
    }

    if (ev.key === 'Escape') {
      if (state.selectedPart !== null || state.hoveredPart !== null) {
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
    }
  });

  void TOWER_FIT_BOUNDS;

  rebuildSceneIfNeeded();
  controls.sync(state, currentCutawayScene);
  drawNow();
}

boot();
