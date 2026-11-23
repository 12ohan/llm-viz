import './ui/theme.css';

import { NEMOTRON_SPEC, type PrecisionMode } from './model/spec';
import {
  cloneCameraPose,
  createInitialState,
  createMacroCameraPose,
  DEFAULT_TOKENS,
  MAX_TOKENS,
  MIN_TOKENS,
  type SceneState,
} from './model/state';
import {
  buildCameraSnapshot,
  fitCameraToBounds,
  makeViewportMetrics,
  panCameraByCssDelta,
  refitCameraOnResize,
  type CameraSnapshot,
  type ViewportMetrics,
} from './render/camera';
import { RenderDevice } from './render/device';
import { LinesPass } from './render/lines';
import { clientPointToRay, pickMacroBay, zoomCameraAtClientPoint } from './render/pick';
import { SolidsPass } from './render/solids';
import { buildMacroSpine, MACRO_FIT_BOUNDS, type MacroSceneGeometry } from './scene/spine';
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
    measureHostMetrics(host) ?? makeViewportMetrics(0, 0, 768, 600, window.devicePixelRatio || 1);
  const initialFit = fitCameraToBounds(
    MACRO_FIT_BOUNDS,
    24,
    27,
    initialMetrics.innerWidth,
    initialMetrics.innerHeight,
  );
  const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
  const state: SceneState = createInitialState(initialFit.target, reducedMotion);

  let currentMetrics: ViewportMetrics = initialMetrics;
  let currentScene: MacroSceneGeometry | null = null;
  let currentSnapshot: CameraSnapshot | null = null;
  let geometryDirty = true;
  let pendingRafId: number | null = null;

  // WebGL2 initialization with graceful fallback (§3)
  let device: RenderDevice | null = null;
  let solidsPass: SolidsPass | null = null;
  let linesPass: LinesPass | null = null;

  try {
    device = new RenderDevice(canvas);
    solidsPass = new SolidsPass(device.gl);
    linesPass = new LinesPass(device.gl);
  } catch (err) {
    const detail = err instanceof Error ? err.message : 'WebGL2 unavailable';
    showFallbackMessage(
      fallbackEl,
      `3D canvas unavailable (${detail}). Use the token scrubber, precision toggle, and ← / → bay controls below to inspect all 52 layers.`,
    );
  }

  const overlay = new MacroOverlay(overlaySvg, overlayLabels);
  const controlEls = queryControlsElements();

  const markDirtyAndSchedule = (rebuildGeometry: boolean) => {
    if (rebuildGeometry) geometryDirty = true;
    controls.sync(state);
    scheduleDraw();
  };

  const resetAll = () => {
    const fit = fitCameraToBounds(
      MACRO_FIT_BOUNDS,
      24,
      27,
      currentMetrics.innerWidth,
      currentMetrics.innerHeight,
    );
    state.tokens = DEFAULT_TOKENS;
    state.precision = 'bf16';
    state.selectedBay = null;
    state.hoveredBay = null;
    state.selectedPart = null;
    state.camera = createMacroCameraPose(fit.target);
    state.savedMacroCamera = cloneCameraPose(state.camera);
    markDirtyAndSchedule(true);
  };

  const stepBay = (delta: -1 | 1) => {
    if (state.selectedBay === null) {
      state.selectedBay = delta > 0 ? 1 : 52;
    } else {
      const next = state.selectedBay + delta;
      state.selectedBay = Math.max(1, Math.min(52, next));
    }
    state.hoveredBay = null;
    markDirtyAndSchedule(true);
  };

  const controls = new MacroControls(NEMOTRON_SPEC, controlEls, {
    onTokensChange: (nextTokens) => {
      const clamped = Math.max(MIN_TOKENS, Math.min(MAX_TOKENS, nextTokens));
      if (clamped === state.tokens) return;
      state.tokens = clamped;
      markDirtyAndSchedule(true);
    },
    onPrecisionChange: (mode: PrecisionMode) => {
      if (mode === state.precision) return;
      state.precision = mode;
      // Preserve T, selection, camera (§5)
      markDirtyAndSchedule(true);
    },
    onReset: resetAll,
    onStepBay: stepBay,
    onClearSelection: () => {
      state.selectedBay = null;
      state.hoveredBay = null;
      markDirtyAndSchedule(true);
    },
  });

  /** Single synchronous draw function (§3 Frame lifecycle). */
  const drawFrame = () => {
    pendingRafId = null;

    const measured = measureHostMetrics(host);
    if (!measured) return;
    currentMetrics = measured;

    if (geometryDirty || !currentScene) {
      currentScene = buildMacroSpine({
        spec: NEMOTRON_SPEC,
        tokens: state.tokens,
        precision: state.precision,
        selectedBay: state.selectedBay,
        hoveredBay: state.hoveredBay,
      });
      if (device && !device.isLost && solidsPass && linesPass) {
        solidsPass.upload(currentScene.solidData, currentScene.solidCount);
        linesPass.upload(currentScene.lineData, currentScene.lineCount);
      }
      geometryDirty = false;
    }

    currentSnapshot = buildCameraSnapshot(state.camera, currentScene.fitBounds, currentMetrics);

    if (device && !device.isLost && solidsPass && linesPass) {
      const fb = device.ensureSize(currentMetrics);
      device.beginFrame(fb);
      solidsPass.draw(currentSnapshot.viewProj);
      linesPass.draw(
        currentSnapshot.viewProj,
        fb.vpWidth,
        fb.vpHeight,
        currentMetrics.dpr,
        currentSnapshot.basis.eyeDir,
      );
      device.endFrame(fb);
    }

    // Project and update SVG/DOM overlay synchronously in the same draw call (§4)
    overlay.update({
      spec: NEMOTRON_SPEC,
      scene: currentScene,
      snapshot: currentSnapshot,
      tokens: state.tokens,
      selectedBay: state.selectedBay,
      hoveredBay: state.hoveredBay,
    });
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
    drawFrame();
  };

  // --- ResizeObserver + DPR media query (§3) ---
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
    state.camera = refitCameraOnResize(state.camera, MACRO_FIT_BOUNDS, currentMetrics, next);
    currentMetrics = next;
    lastHostW = next.hostWidth;
    lastHostH = next.hostHeight;
    lastDpr = next.dpr;
    drawNow();
  };

  const resizeObserver = new ResizeObserver(() => {
    handleResizeOrDpr();
  });
  resizeObserver.observe(host);

  let dprMediaQuery: MediaQueryList | null = null;
  const watchDpr = () => {
    if (dprMediaQuery) {
      dprMediaQuery.removeEventListener('change', onDprChange);
    }
    dprMediaQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    dprMediaQuery.addEventListener('change', onDprChange);
  };
  const onDprChange = () => {
    watchDpr();
    handleResizeOrDpr();
  };
  watchDpr();

  // Refresh metric cache after web fonts settle (§4)
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      overlay.clearMetricCache();
      scheduleDraw();
    });
  }

  // --- WebGL context lost / restored (§3) ---
  canvas.addEventListener('webglcontextlost', (ev) => {
    ev.preventDefault();
    if (pendingRafId !== null) {
      window.cancelAnimationFrame(pendingRafId);
      pendingRafId = null;
    }
    if (device) device.markLost(true);
    showFallbackMessage(
      fallbackEl,
      'WebGL2 context lost. Waiting for restoration; controls and readout remain active.',
    );
  });

  canvas.addEventListener('webglcontextrestored', () => {
    if (!device) return;
    try {
      device.markLost(false);
      solidsPass = new SolidsPass(device.gl);
      linesPass = new LinesPass(device.gl);
      hideFallbackMessage(fallbackEl);
      geometryDirty = true;
      drawNow();
    } catch {
      showFallbackMessage(
        fallbackEl,
        'WebGL2 context could not be restored. Controls and layer inspector remain active below.',
      );
    }
  });

  // --- Pointer gestures, picking, drag pan, wheel & pinch zoom (§4) ---
  const activePointers = new Map<number, { x: number; y: number }>();
  let primaryPointerId: number | null = null;
  let downX = 0;
  let downY = 0;
  let lastPanX = 0;
  let lastPanY = 0;
  let isDragging = false;
  let pendingSelectionCancelled = false;

  // Pinch state
  let pinchStartDist = 0;
  let pinchStartZoom = 1;
  let pinchLastMidX = 0;
  let pinchLastMidY = 0;

  const pickAtClient = (clientX: number, clientY: number): number | null => {
    if (!currentScene || !currentSnapshot) return null;
    // Refresh host origin from live rect in case page scrolled
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
    if (ev.button !== 0 && ev.pointerType === 'mouse') return;
    canvas.setPointerCapture(ev.pointerId);
    activePointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });

    if (activePointers.size === 1) {
      primaryPointerId = ev.pointerId;
      downX = ev.clientX;
      downY = ev.clientY;
      lastPanX = ev.clientX;
      lastPanY = ev.clientY;
      isDragging = false;
      pendingSelectionCancelled = false;
    } else if (activePointers.size === 2) {
      // Two-finger pinch cancels any pending selection (§4)
      pendingSelectionCancelled = true;
      isDragging = true;
      const pts = Array.from(activePointers.values());
      pinchStartDist = Math.max(1, Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y));
      pinchStartZoom = state.camera.zoom;
      pinchLastMidX = 0.5 * (pts[0].x + pts[1].x);
      pinchLastMidY = 0.5 * (pts[0].y + pts[1].y);
    }
  });

  canvas.addEventListener('pointermove', (ev) => {
    if (activePointers.has(ev.pointerId)) {
      activePointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    }

    if (activePointers.size === 2 && currentScene) {
      const pts = Array.from(activePointers.values());
      const dist = Math.max(1, Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y));
      const midX = 0.5 * (pts[0].x + pts[1].x);
      const midY = 0.5 * (pts[0].y + pts[1].y);
      const rect = host.getBoundingClientRect();
      const liveMetrics = makeViewportMetrics(
        rect.left,
        rect.top,
        rect.width,
        rect.height,
        window.devicePixelRatio || 1,
      );
      const nextZoom = pinchStartZoom * (dist / pinchStartDist);
      state.camera = zoomCameraAtClientPoint(
        state.camera,
        currentScene.fitBounds,
        liveMetrics,
        pinchLastMidX,
        pinchLastMidY,
        nextZoom,
        midX,
        midY,
      );
      pinchLastMidX = midX;
      pinchLastMidY = midY;
      scheduleDraw();
      return;
    }

    if (primaryPointerId === ev.pointerId && activePointers.size === 1) {
      const totalDist = Math.hypot(ev.clientX - downX, ev.clientY - downY);
      if (!isDragging && totalDist >= DRAG_THRESHOLD_PX) {
        isDragging = true;
      }
      if (isDragging && currentSnapshot) {
        const dx = ev.clientX - lastPanX;
        const dy = ev.clientY - lastPanY;
        lastPanX = ev.clientX;
        lastPanY = ev.clientY;
        state.camera = panCameraByCssDelta(state.camera, currentSnapshot, dx, dy);
        scheduleDraw();
      }
      return;
    }

    // Hover picking when no drag is active
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

  canvas.addEventListener('pointerleave', () => {
    if (state.hoveredBay !== null) {
      state.hoveredBay = null;
      markDirtyAndSchedule(true);
    }
  });

  // Wheel zoom anchored at cursor (§4)
  canvas.addEventListener(
    'wheel',
    (ev) => {
      if (!currentScene) return;
      ev.preventDefault();
      const rect = host.getBoundingClientRect();
      const liveMetrics = makeViewportMetrics(
        rect.left,
        rect.top,
        rect.width,
        rect.height,
        window.devicePixelRatio || 1,
      );
      let deltaPx = ev.deltaY;
      if (ev.deltaMode === WheelEvent.DOM_DELTA_LINE) {
        deltaPx *= 16;
      } else if (ev.deltaMode === WheelEvent.DOM_DELTA_PAGE) {
        deltaPx *= liveMetrics.innerHeight;
      }
      const factor = Math.exp(-0.0015 * deltaPx);
      const nextZoom = state.camera.zoom * factor;
      state.camera = zoomCameraAtClientPoint(
        state.camera,
        currentScene.fitBounds,
        liveMetrics,
        ev.clientX,
        ev.clientY,
        nextZoom,
      );
      scheduleDraw();
    },
    { passive: false },
  );

  // Keyboard shortcuts: Left/Right step 01–52, Esc clears selection/hover (§4)
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

    if (ev.key === 'ArrowRight') {
      ev.preventDefault();
      stepBay(1);
    } else if (ev.key === 'ArrowLeft') {
      ev.preventDefault();
      stepBay(-1);
    } else if (ev.key === 'Escape') {
      if (state.selectedBay !== null || state.hoveredBay !== null) {
        ev.preventDefault();
        state.selectedBay = null;
        state.hoveredBay = null;
        markDirtyAndSchedule(true);
      }
    }
  });

  // Initial sync and synchronous first paint
  controls.sync(state);
  drawNow();
}

boot();
