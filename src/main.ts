import './ui/theme.css';

import { NEMOTRON_SPEC, type PrecisionMode } from './model/spec';
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
} from './model/state';
import {
  buildCameraSnapshot,
  fitPerspectiveToBounds,
  interpolateCameraPose,
  makeViewportMetrics,
  orbitCameraByCssDelta,
  panCameraByCssDelta,
  zoomPerspectiveDistance,
  type CameraSnapshot,
  type ViewportMetrics,
} from './render/camera';
import { RenderDevice } from './render/device';
import { LinesPass } from './render/lines';
import { clientPointToRay, pickMacroBay } from './render/pick';
import { RibbonsPass } from './render/ribbons';
import { SolidsPass } from './render/solids';
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
      return fitPerspectiveToBounds(
        bounds, 36, 20, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.80,
      );
    case 'kv_fp8':
      return fitPerspectiveToBounds(
        bounds, 34, 18, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.80,
      );
    case 'output':
      return fitPerspectiveToBounds(
        bounds, 26, 16, metrics.innerWidth, metrics.innerHeight, DEFAULT_FOV_DEG, 0.78,
      );
  }
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
  let currentScene: MacroSceneGeometry | null = null;
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

  const markDirtyAndSchedule = (rebuildGeometry: boolean) => {
    if (rebuildGeometry) geometryDirty = true;
    controls.sync(state);
    scheduleDraw();
  };

  const selectSection = (section: TocSectionId) => {
    state.section = section;
    if (section === 'mamba') state.selectedBay = 1;
    else if (section === 'moe') state.selectedBay = 2;
    else if (section === 'gqa') state.selectedBay = 6;
    else if (section === 'kv_fp8') state.selectedBay = 42;
    else if (section === 'overview' || section === 'embedding' || section === 'output') {
      state.selectedBay = null;
    }
    state.hoveredBay = null;
    flyToPose(cameraPoseForSection(section, currentMetrics));
    markDirtyAndSchedule(true);
  };

  const zoomToBay = (bayId: number) => {
    const bayBounds = boundsForBay(bayId);
    const bay = NEMOTRON_SPEC.bays[bayId - 1];
    const az = bay?.kind === '*' ? 36 : 26;
    const el = bay?.kind === 'E' ? 24 : 18;
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

  const resetAll = () => {
    state.section = 'overview';
    state.tokens = DEFAULT_TOKENS;
    state.precision = 'bf16';
    state.selectedBay = null;
    state.hoveredBay = null;
    state.selectedPart = null;
    flyToPose(cameraPoseForSection('overview', currentMetrics));
    markDirtyAndSchedule(true);
  };

  const stepBay = (delta: -1 | 1) => {
    if (state.selectedBay === null) {
      state.selectedBay = delta > 0 ? 1 : 52;
    } else {
      state.selectedBay = Math.max(1, Math.min(52, state.selectedBay + delta));
    }
    state.hoveredBay = null;
    markDirtyAndSchedule(true);
  };

  const controls = new MacroControls(NEMOTRON_SPEC, controlEls, {
    onSelectSection: selectSection,
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
    onFitAll: () => {
      state.section = 'overview';
      flyToPose(cameraPoseForSection('overview', currentMetrics));
      controls.sync(state);
    },
    onZoomSelectedBay: () => {
      if (state.selectedBay !== null) {
        zoomToBay(state.selectedBay);
      }
    },
    onReset: resetAll,
    onStepBay: stepBay,
    onClearSelection: () => {
      state.selectedBay = null;
      state.hoveredBay = null;
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

    if (geometryDirty || !currentScene) {
      currentScene = buildMacroSpine({
        spec: NEMOTRON_SPEC,
        tokens: state.tokens,
        precision: state.precision,
        selectedBay: state.selectedBay,
        hoveredBay: state.hoveredBay,
      });
      if (device && !device.isLost && solidsPass && linesPass && ribbonsPass) {
        solidsPass.upload(currentScene.solidData, currentScene.solidCount);
        linesPass.upload(currentScene.lineData, currentScene.lineCount);
        ribbonsPass.upload(currentScene.ribbonData, currentScene.ribbonCount);
      }
      geometryDirty = false;
    }

    currentSnapshot = buildCameraSnapshot(state.camera, currentScene.fitBounds, currentMetrics);

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
    overlay.update({
      spec: NEMOTRON_SPEC,
      scene: currentScene,
      snapshot: currentSnapshot,
      tokens: state.tokens,
      selectedBay: state.selectedBay,
      hoveredBay: state.hoveredBay,
    });

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
    // Refit cleanly if still on full overview and not manually panned/zoomed
    if (state.section === 'overview' && !flyAnim && state.selectedBay === null) {
      state.camera = cameraPoseForSection('overview', next);
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

  const pickAtClient = (clientX: number, clientY: number): number | null => {
    if (!currentScene) return null;
    const rect = host.getBoundingClientRect();
    const liveMetrics = makeViewportMetrics(
      rect.left,
      rect.top,
      rect.width,
      rect.height,
      window.devicePixelRatio || 1,
    );
    const snap = buildCameraSnapshot(state.camera, currentScene.fitBounds, liveMetrics);
    const ray = clientPointToRay(clientX, clientY, snap);
    const hit = pickMacroBay(ray, currentScene.solidPickBoxes, currentScene.fasciaPickBoxes);
    return hit ? hit.bayId : null;
  };

  canvas.addEventListener('pointerdown', (ev) => {
    canvas.setPointerCapture(ev.pointerId);
    activePointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });

    if (activePointers.size === 1) {
      primaryPointerId = ev.pointerId;
      // Right-drag, middle-drag, or Shift+left-drag pans; plain left-drag orbits!
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
      const hitBay = pickAtClient(ev.clientX, ev.clientY);
      if (hitBay !== state.hoveredBay) {
        state.hoveredBay = hitBay;
        markDirtyAndSchedule(true);
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
      const hitBay = pickAtClient(ev.clientX, ev.clientY);
      state.selectedBay = hitBay;
      state.hoveredBay = hitBay;
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
    const hitBay = pickAtClient(ev.clientX, ev.clientY);
    if (hitBay !== null) {
      state.selectedBay = hitBay;
      state.hoveredBay = hitBay;
      zoomToBay(hitBay);
      markDirtyAndSchedule(true);
    }
  });

  canvas.addEventListener('pointerleave', () => {
    if (state.hoveredBay !== null) {
      state.hoveredBay = null;
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

    if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') {
      ev.preventDefault();
      stepBay(1);
    } else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') {
      ev.preventDefault();
      stepBay(-1);
    } else if (ev.key === 'Enter' && state.selectedBay !== null) {
      ev.preventDefault();
      zoomToBay(state.selectedBay);
    } else if (ev.key === 'Escape') {
      if (state.selectedBay !== null || state.hoveredBay !== null) {
        ev.preventDefault();
        state.selectedBay = null;
        state.hoveredBay = null;
        markDirtyAndSchedule(true);
      }
    }
  });

  // Expose subtle bounds reference for tests / diagnostics (unused in UI)
  void TOWER_FIT_BOUNDS;

  controls.sync(state);
  drawNow();
}

boot();
