/**
 * Interactive WebGL2 Spatial App Shell for Nemotron-Labs-Diffusion-Uni
 * (`src/uni/app/spatialApp.ts`)
 *
 * Hosts all 5 unchanged WebGL2 passes (`RenderDevice`, `SolidsPass`, `LinesPass`,
 * `RibbonsPass`, `FlatFlowRibbonsPass`, `SheetsPass`) across all 7 NLD-Uni views
 * (`macro`, `visual_input`, `joint_attn`, `ball_tree`, `refinement`, `output_edit`,
 * `opd_readonly`), paired with:
 *   - Pure Node-testable `createIdleRafScheduler` (strict idle-RAF gating: `rafActive === false` at rest)
 *   - Pure Node-testable `createAtlasUploadTracker` (content-addressed `RG32F` dedup)
 *   - Dedicated 2D `<canvas>` RGB inset panel (`pixelCanvas.ts`, D4)
 *   - Pooled 3D-to-2D overlay (`spatialOverlay.ts`, `<= 64` cell-number node cap)
 *   - Accessible timeline & Elastic-MoT dock (`timelineDock.ts`)
 *   - Collapsible 3-tab inspector drawer (`Inspect` | `Architecture` | `Ledger U1–U30`)
 *   - 28-beat isolated Guided Pedagogical Tour HUD (`createUniGuidedTour`)
 *   - `window.__uniSpatialDebug` E2E & diagnostic handle (`UniSpatialDebugHandle`)
 *
 * Strictly ZERO imports from `src/ui/controls.ts` (D4), `src/twotower/`, or `src/diffusion/`.
 * Strictly ZERO `Math.random()`.
 */

import type { CameraPose } from '../../model/state';
import {
  buildCameraSnapshot,
  fitOpBoundsCamera,
  interpolateCameraPose,
  makeViewportMetrics,
  orbitCameraByCssDelta,
  panCameraByCssDelta,
  zoomPerspectiveDistance,
  type CameraSnapshot,
  type ViewportMetrics,
} from '../../render/camera';
import { RenderDevice } from '../../render/device';
import { LinesPass } from '../../render/lines';
import { clientPointToRay, pickMacroBay } from '../../render/pick';
import { FlatFlowRibbonsPass, RibbonsPass } from '../../render/ribbons';
import { SheetsPass } from '../../render/sheets';
import { SolidsPass } from '../../render/solids';
import {
  buildUniElasticD5Comparison,
  buildUniViewArchReadouts,
} from '../archReadouts';
import { createDefaultUniRuntimeBundle } from '../imageSampler';
import { buildUniSpatialScene } from '../scene/index';
import { pickHitToUniCell } from '../scene/provenance';
import {
  DEFAULT_UNI_SPATIAL_FOCUS_STATE,
  UNI_SPATIAL_VIEW_MODES,
  type UniOpdBayFocus,
  type UniPickIdentity,
  type UniRefinementStepPreset,
  type UniSpatialFocusState,
  type UniSpatialScene,
  type UniSpatialViewMode,
} from '../scene/types';
import { UniSession } from '../session';
import {
  formatSiIecBytes,
  UNI_CANONICAL_REFS,
  UNI_CONTEXT_TIERS,
  UNI_DISCREPANCY_LEDGER,
} from '../spec';
import type { UniFreeExploreRestoreState, UniTourState } from '../story/types';
import { createUniGuidedTour } from '../story/walkthrough';
import type {
  UniEditGuidanceMode,
  UniElasticPrefillMode,
  UniExecutionViewId,
  UniGainPresetId,
  UniScenePresetId,
  UniSessionTaskMode,
} from '../types';
import {
  createUniFocusTrap,
  handleUniTablistKeydown,
  isUniReducedMotionPreferred,
  shouldDeferGlobalShortcut,
  syncRovingTabindex,
} from './a11yHelpers';
import { createUniPixelCanvasPanel } from './pixelCanvas';
import { UniSpatialOverlay } from './spatialOverlay';
import {
  buildUniTimelineDockState,
  createUniTimelineDock,
  taskChoiceToSessionTaskMode,
} from './timelineDock';
import type {
  UniDockTaskChoice,
  UniInspectorTabId,
  UniOverlayRect,
  UniSpatialDebugHandle,
  UniSpatialDebugSnapshot,
} from './types';

// ============================================================================
// PURE NODE-TESTABLE STATE MACHINES (IDLE-RAF & RG32F ATLAS UPLOAD DEDUP)
// ============================================================================

export interface UniIdleRafSchedulerOptions {
  readonly onFrame: (nowMs: number, dtMs: number) => void;
  readonly requestFrameFn?: (callback: (nowMs: number) => void) => number;
  readonly cancelFrameFn?: (handle: number) => void;
}

export interface UniIdleRafScheduler {
  isActive(): boolean;
  getContinuousReasons(): readonly string[];
  requestFrame(): void;
  setContinuousReason(reason: string, active: boolean): void;
  dispose(): void;
}

/**
 * Strict idle-RAF scheduler guaranteeing `isActive() === false` at rest:
 * wakes only on `requestFrame()` or active `setContinuousReason(reason, true)`
 * (e.g., playback or camera fly), and immediately halts once settled.
 */
export function createIdleRafScheduler(
  options: UniIdleRafSchedulerOptions,
): UniIdleRafScheduler {
  const reqFn =
    options.requestFrameFn ??
    ((cb: (nowMs: number) => void) =>
      typeof requestAnimationFrame === 'function'
        ? requestAnimationFrame(cb)
        : (setTimeout(() => cb(Date.now()), 16) as unknown as number));

  const cancelFn =
    options.cancelFrameFn ??
    ((handle: number) => {
      if (typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(handle);
      } else {
        clearTimeout(handle);
      }
    });

  let rafActive = false;
  let rafHandle: number | null = null;
  let pendingOneShot = false;
  let lastTimestampMs: number | null = null;
  let disposed = false;
  const continuousReasons = new Set<string>();

  const scheduleNext = (): void => {
    if (disposed || rafHandle !== null) return;
    rafActive = true;
    rafHandle = reqFn(onTick);
  };

  const onTick = (nowMs: number): void => {
    rafHandle = null;
    if (disposed) {
      rafActive = false;
      lastTimestampMs = null;
      return;
    }

    pendingOneShot = false;
    const dtMs =
      lastTimestampMs === null ? 0 : Math.max(0, Math.min(100, nowMs - lastTimestampMs));
    lastTimestampMs = nowMs;

    options.onFrame(nowMs, dtMs);

    if (!disposed && (pendingOneShot || continuousReasons.size > 0)) {
      scheduleNext();
    } else {
      rafActive = false;
      lastTimestampMs = null;
    }
  };

  return {
    isActive(): boolean {
      return rafActive;
    },
    getContinuousReasons(): readonly string[] {
      return Array.from(continuousReasons);
    },
    requestFrame(): void {
      if (disposed) return;
      pendingOneShot = true;
      scheduleNext();
    },
    setContinuousReason(reason: string, active: boolean): void {
      if (disposed) return;
      if (active) {
        continuousReasons.add(reason);
        scheduleNext();
      } else {
        continuousReasons.delete(reason);
        if (continuousReasons.size === 0 && !pendingOneShot && rafHandle !== null) {
          cancelFn(rafHandle);
          rafHandle = null;
          rafActive = false;
          lastTimestampMs = null;
        }
      }
    },
    dispose(): void {
      if (disposed) return;
      disposed = true;
      continuousReasons.clear();
      pendingOneShot = false;
      if (rafHandle !== null) {
        cancelFn(rafHandle);
        rafHandle = null;
      }
      rafActive = false;
      lastTimestampMs = null;
    },
  };
}

export interface UniAtlasUploadTracker {
  getLastAtlasKey(): string | null;
  getUploadCount(): number;
  shouldUpload(atlasKey: string | null): boolean;
  reset(): void;
}

/**
 * Pure content-addressed `RG32F` GPU texture upload deduplicator:
 * increments `atlasUploadCount` and returns `true` ONLY when `atlasKey !== null`
 * and `atlasKey !== lastAtlasKey`. Revisiting an identical `atlasKey` or passing
 * `null` (e.g., View 1 Macro) triggers `0` redundant uploads.
 */
export function createAtlasUploadTracker(): UniAtlasUploadTracker {
  let lastAtlasKey: string | null = null;
  let atlasUploadCount = 0;

  return {
    getLastAtlasKey(): string | null {
      return lastAtlasKey;
    },
    getUploadCount(): number {
      return atlasUploadCount;
    },
    shouldUpload(atlasKey: string | null): boolean {
      if (atlasKey === null) {
        return false;
      }
      if (atlasKey === lastAtlasKey) {
        return false;
      }
      lastAtlasKey = atlasKey;
      atlasUploadCount += 1;
      return true;
    },
    reset(): void {
      lastAtlasKey = null;
    },
  };
}

// ============================================================================
// APP CONSTANTS & DOM HELPERS
// ============================================================================

const CAMERA_FLY_DURATION_MS = 420;
const PLAYBACK_STEP_INTERVAL_MS = 850;
const LAYER_PRESETS: readonly number[] = [0, 17, 18, 33];
const REFINEMENT_STEP_PRESETS: readonly UniRefinementStepPreset[] = [1, 2, 4, 8];
const EXECUTION_VIEW_CHOICES: readonly {
  readonly id: UniExecutionViewId;
  readonly label: string;
}[] = [
  { id: 'text_understanding', label: 'Text (34U)' },
  { id: 'visual_understanding', label: 'VLM (34U)' },
  { id: 'object_grounding', label: 'Ground (34U)' },
  { id: 'text_to_image', label: 'T2I (18U+34G)' },
  { id: 'image_edit', label: 'Edit (18U+34G)' },
  { id: 'interleaved_mixed', label: 'Mixed (U7)' },
];
const SCENE_PRESET_CHOICES: readonly {
  readonly id: UniScenePresetId;
  readonly label: string;
}[] = [
  { id: 'sunlit_cat_16x16', label: 'Sunlit Cat (16×16)' },
  { id: 'harbor_panorama_16x8', label: 'Harbor Panorama (16×8)' },
  { id: 'geometric_blocks_16x16', label: 'Geometric Blocks (16×16)' },
];
const INSPECTOR_TABS: readonly {
  readonly id: UniInspectorTabId;
  readonly label: string;
}[] = [
  { id: 'inspect', label: 'Inspect' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'ledger', label: 'Ledger U1–U30' },
];

type LedgerFilterCategory =
  | 'all'
  | 'browser_guard'
  | 'architecture_elastic'
  | 'multimodal_codec'
  | 'training_serving'
  | 'provenance_license';

const LEDGER_FILTERS: readonly {
  readonly id: LedgerFilterCategory;
  readonly label: string;
}[] = [
  { id: 'all', label: 'All (30)' },
  { id: 'browser_guard', label: 'Guards' },
  { id: 'architecture_elastic', label: 'Elastic' },
  { id: 'multimodal_codec', label: 'Multimodal' },
  { id: 'training_serving', label: 'Train/OPD' },
  { id: 'provenance_license', label: 'Provenance' },
];

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addKvRow(grid: HTMLElement, key: string, value: string): void {
  grid.append(el('span', 'uni-kv-key', key), el('span', 'uni-kv-val', value));
}

function formatCompactCount(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(2)}K`;
  return String(n);
}

// ============================================================================
// MAIN WEBGL2 SPATIAL APPLICATION SHELL
// ============================================================================

export interface UniSpatialAppInstance {
  getDebugSnapshot(): UniSpatialDebugSnapshot;
  getDebugHandle(): UniSpatialDebugHandle;
  dispose(): void;
}

interface GlPassesBundle {
  readonly device: RenderDevice;
  readonly solids: SolidsPass;
  readonly lines: LinesPass;
  readonly ribbons: RibbonsPass;
  readonly flatFlow: FlatFlowRibbonsPass;
  readonly sheets: SheetsPass;
}

interface CameraFlyState {
  readonly from: CameraPose;
  readonly to: CameraPose;
  readonly durationMs: number;
  elapsedMs: number;
}

export function createUniSpatialApp(root: HTMLElement): UniSpatialAppInstance {
  root.innerHTML = '';

  const bundle = createDefaultUniRuntimeBundle();
  const freeSession = new UniSession(bundle);
  const initialBranch = freeSession.startBranch({
    taskMode: 'text_to_image',
    label: 'T2I · Elastic-Pruned',
    prefillMode: 'elastic_pruned',
    gainPresetId: 'mini_default',
    activationSteps: 4,
    refinementSteps: 4,
  });
  freeSession.runBranchToCompletion(initialBranch.branchId, 4);

  const tour = createUniGuidedTour(bundle);
  const atlasTracker = createAtlasUploadTracker();

  let disposed = false;
  let viewMode: UniSpatialViewMode = 'macro';
  let inspectorTab: UniInspectorTabId = 'inspect';
  let inspectorCollapsed = false;
  let ledgerFilter: LedgerFilterCategory = 'all';
  let ledgerQuery = '';
  let isPlaying = false;
  let playbackAccumMs = 0;
  let glErrorCount = 0;
  let reducedMotion = isUniReducedMotionPreferred();

  let focus: UniSpatialFocusState = {
    ...DEFAULT_UNI_SPATIAL_FOCUS_STATE,
    executionViewId: 'text_to_image',
    prefillMode: 'elastic_pruned',
    selectedLayerIndex: 17,
    jointLayerIndex: 17,
    batchRow: 1,
    cfgBranchRow: 1,
  };

  let hoveredCell: UniPickIdentity | null = null;
  let selectedCell: UniPickIdentity | null = null;
  let selectedPartId: string | null = null;

  let currentScene: UniSpatialScene = buildUniSpatialScene(viewMode, freeSession, focus);
  let cameraPose: CameraPose | null = null;
  let cameraSnapshot: CameraSnapshot | null = null;
  let activeFly: CameraFlyState | null = null;
  let sceneDirtyForGl = true;

  // --------------------------------------------------------------------------
  // DOM SKELETON
  // --------------------------------------------------------------------------

  const shell = el('div', 'uni-app-shell');
  shell.dataset.testid = 'uni-app-shell';

  // Top Header Bar
  const topbar = el('header', 'uni-topbar');
  const brand = el('div', 'uni-brand');
  brand.append(
    el('span', 'uni-brand-mark', 'NLD-UNI'),
    el('span', 'uni-brand-title', 'Nemotron-Labs-Diffusion-Uni'),
    el('span', 'uni-brand-sub', '18.27B Elastic-MoT · 34L (18J + 16S)'),
  );

  const viewTablist = el('div', 'uni-view-tablist');
  viewTablist.setAttribute('role', 'tablist');
  viewTablist.setAttribute('aria-label', 'NLD-Uni spatial views');

  const viewTabButtons: HTMLButtonElement[] = [];
  for (let i = 0; i < UNI_SPATIAL_VIEW_MODES.length; i++) {
    const desc = UNI_SPATIAL_VIEW_MODES[i]!;
    const btn = el('button', 'uni-view-tab');
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.dataset.viewMode = desc.id;
    btn.dataset.testid = `uni-view-tab-${desc.id}`;
    btn.title = `${desc.viewNumber}. ${desc.label} — ${desc.description}`;
    btn.append(
      el('span', 'uni-view-tab-num', String(desc.viewNumber)),
      el('span', undefined, desc.shortLabel),
    );
    btn.addEventListener('click', () => {
      if (tour.isActive()) {
        exitTourInternal(false);
      }
      setViewInternal(desc.id, true);
    });
    btn.addEventListener('keydown', (ev) => {
      const curIdx = UNI_SPATIAL_VIEW_MODES.findIndex((v) => v.id === viewMode);
      handleUniTablistKeydown(ev, viewTabButtons, curIdx >= 0 ? curIdx : 0, (nextIdx) => {
        if (tour.isActive()) {
          exitTourInternal(false);
        }
        setViewInternal(UNI_SPATIAL_VIEW_MODES[nextIdx]!.id, true);
      });
    });
    viewTabButtons.push(btn);
    viewTablist.appendChild(btn);
  }
  syncRovingTabindex(viewTabButtons, 0);

  const topActions = el('div', 'uni-topbar-actions');

  const tourLaunchBtn = el('button', 'uni-action-btn uni-action-btn--tour', 'Guided Tour (28)');
  tourLaunchBtn.type = 'button';
  tourLaunchBtn.dataset.testid = 'uni-tour-launch-btn';
  tourLaunchBtn.addEventListener('click', () => {
    if (tour.isActive()) {
      exitTourInternal(true);
    } else {
      startTourInternal(0);
    }
  });

  const resetCamBtn = el('button', 'uni-action-btn', 'Reset View');
  resetCamBtn.type = 'button';
  resetCamBtn.dataset.testid = 'uni-reset-camera-btn';
  resetCamBtn.addEventListener('click', () => {
    flyToCurrentScenePreset(true);
  });

  const inspectorToggleBtn = el('button', 'uni-action-btn', 'Hide Inspector');
  inspectorToggleBtn.type = 'button';
  inspectorToggleBtn.dataset.testid = 'uni-inspector-toggle-btn';
  inspectorToggleBtn.setAttribute('aria-expanded', 'true');
  inspectorToggleBtn.addEventListener('click', () => {
    toggleInspectorCollapsed();
  });

  topActions.append(tourLaunchBtn, resetCamBtn, inspectorToggleBtn);
  topbar.append(brand, viewTablist, topActions);

  // Main Workspace
  const workspace = el('div', 'uni-workspace');
  const stageColumn = el('div', 'uni-stage-column');
  const viewportHost = el('div', 'uni-viewport-host');
  viewportHost.dataset.testid = 'uni-viewport-host';

  const canvas = el('canvas', 'uni-webgl-canvas');
  canvas.dataset.testid = 'uni-webgl-canvas';
  canvas.setAttribute('aria-label', 'Interactive 3D NLD-Uni WebGL2 drafting stage');
  viewportHost.appendChild(canvas);

  const glFallback = el('div', 'uni-webgl-fallback');
  glFallback.dataset.testid = 'uni-webgl-fallback';
  glFallback.hidden = true;
  viewportHost.appendChild(glFallback);

  // Floating Stage Title & Provenance Badges HUD
  const stageHud = el('div', 'uni-stage-hud');
  const stageHudRow = el('div', 'uni-stage-hud-row');
  const stageTitleCard = el('div', 'uni-stage-title-card');
  const stageTitleEl = el('h1', 'uni-stage-title');
  const stageSubtitleEl = el('p', 'uni-stage-subtitle');
  const stageBadgesEl = el('div', 'uni-badge-strip');
  stageTitleCard.append(stageTitleEl, stageSubtitleEl, stageBadgesEl);

  const opdReadOnlyBanner = el(
    'div',
    'uni-opd-readonly-banner',
    'READ-ONLY · NO GRADIENTS · FORWARD REFERENCE ONLY',
  );
  opdReadOnlyBanner.dataset.testid = 'uni-opd-readonly-banner';
  opdReadOnlyBanner.hidden = true;

  stageHudRow.append(stageTitleCard, opdReadOnlyBanner);
  stageHud.appendChild(stageHudRow);
  viewportHost.appendChild(stageHud);

  // Pooled 3D-to-2D Overlay
  const overlay = new UniSpatialOverlay(viewportHost);

  // Dedicated 2D RGB Inset Strip Host (D4)
  const rgbInsetHost = el('div', 'uni-rgb-inset-host');
  rgbInsetHost.dataset.testid = 'uni-rgb-inset-host';
  viewportHost.appendChild(rgbInsetHost);
  const pixelPanel = createUniPixelCanvasPanel(rgbInsetHost);

  // 28-Beat Guided Tour HUD Card
  const tourHud = el('aside', 'uni-tour-hud');
  tourHud.dataset.testid = 'uni-tour-hud';
  tourHud.setAttribute('role', 'dialog');
  tourHud.setAttribute('aria-label', 'Guided Pedagogical Tour');
  tourHud.tabIndex = -1;
  tourHud.hidden = true;

  const tourActsRow = el('div', 'uni-tour-acts');
  const tourActButtons: HTMLButtonElement[] = [];
  for (let i = 0; i < tour.acts.length; i++) {
    const act = tour.acts[i]!;
    const btn = el('button', 'uni-tour-act-btn', `Act ${act.actNumber}: ${act.shortTitle}`);
    btn.type = 'button';
    btn.dataset.testid = `uni-tour-act-${act.id}`;
    btn.addEventListener('click', () => {
      goToTourBeatInternal(act.startBeatIndex);
    });
    tourActButtons.push(btn);
    tourActsRow.appendChild(btn);
  }

  const tourHeader = el('div', 'uni-tour-header');
  const tourBeatCounter = el('span', 'uni-tour-beat-counter', 'Beat 1 / 28');
  const tourExitBtn = el('button', 'uni-action-btn', 'Exit Tour');
  tourExitBtn.type = 'button';
  tourExitBtn.dataset.testid = 'uni-tour-exit-btn';
  tourExitBtn.addEventListener('click', () => exitTourInternal(true));
  tourHeader.append(tourBeatCounter, tourExitBtn);

  const tourTitleEl = el('h2', 'uni-tour-title');
  const tourTakeawayEl = el('p', 'uni-tour-takeaway');
  const tourNarrativeEl = el('p', 'uni-tour-narrative');
  const tourEquationEl = el('div', 'uni-tour-equation');

  const tourFooter = el('div', 'uni-tour-footer');
  const tourRefsEl = el('div', 'uni-tour-refs');
  const tourNavEl = el('div', 'uni-tour-nav');
  const tourPrevBtn = el('button', 'uni-action-btn', '← Prev');
  tourPrevBtn.type = 'button';
  tourPrevBtn.dataset.testid = 'uni-tour-prev-btn';
  tourPrevBtn.addEventListener('click', () => {
    const st = tour.getState();
    goToTourBeatInternal(Math.max(0, st.beatIndex - 1));
  });
  const tourNextBtn = el('button', 'uni-action-btn uni-action-btn--tour', 'Next →');
  tourNextBtn.type = 'button';
  tourNextBtn.dataset.testid = 'uni-tour-next-btn';
  tourNextBtn.addEventListener('click', () => {
    const st = tour.getState();
    if (st.beatIndex >= st.beatCount - 1) {
      exitTourInternal(true);
    } else {
      goToTourBeatInternal(st.beatIndex + 1);
    }
  });
  tourNavEl.append(tourPrevBtn, tourNextBtn);
  tourFooter.append(tourRefsEl, tourNavEl);

  tourHud.append(
    tourActsRow,
    tourHeader,
    tourTitleEl,
    tourTakeawayEl,
    tourNarrativeEl,
    tourEquationEl,
    tourFooter,
  );
  viewportHost.appendChild(tourHud);

  const tourFocusTrap = createUniFocusTrap(tourHud, () => {
    exitTourInternal(true);
  });

  // Bottom Timeline Dock Host
  const dockHost = el('div', 'uni-dock-host');
  const timelineDock = createUniTimelineDock(dockHost, {
    onStepBack: () => stepCursorBy(-1),
    onStepForward: () => stepCursorBy(1),
    onScrubIndex: (idx) => scrubToRecordIndex(idx),
    onTogglePlay: () => togglePlayback(),
    onExecuteNextStep: () => executeNextStep(),
    onSelectTaskChoice: (choice) => selectDockTaskChoice(choice),
    onSelectPrefillMode: (mode) => setPrefillModeInternal(mode),
    onForkBranch: () => forkCurrentBranch(),
    onSwitchBranch: (branchId) => switchToBranch(branchId),
  });

  stageColumn.append(viewportHost, dockHost);

  // Collapsible Right 3-Tab Inspector Drawer
  const inspectorDrawer = el('aside', 'uni-inspector-drawer');
  inspectorDrawer.dataset.testid = 'uni-inspector-drawer';
  inspectorDrawer.setAttribute('aria-label', 'NLD-Uni architectural inspector');

  const inspectorTablist = el('div', 'uni-inspector-tablist');
  inspectorTablist.setAttribute('role', 'tablist');
  inspectorTablist.setAttribute('aria-label', 'Inspector panels');

  const inspectorTabButtons: HTMLButtonElement[] = [];
  for (let i = 0; i < INSPECTOR_TABS.length; i++) {
    const tab = INSPECTOR_TABS[i]!;
    const btn = el('button', 'uni-inspector-tab', tab.label);
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.dataset.testid = `uni-inspector-tab-${tab.id}`;
    btn.addEventListener('click', () => {
      inspectorTab = tab.id;
      syncInspectorTabs();
      renderInspectorBody();
    });
    btn.addEventListener('keydown', (ev) => {
      const curIdx = INSPECTOR_TABS.findIndex((t) => t.id === inspectorTab);
      handleUniTablistKeydown(ev, inspectorTabButtons, curIdx >= 0 ? curIdx : 0, (nextIdx) => {
        inspectorTab = INSPECTOR_TABS[nextIdx]!.id;
        syncInspectorTabs();
        renderInspectorBody();
      });
    });
    inspectorTabButtons.push(btn);
    inspectorTablist.appendChild(btn);
  }
  syncRovingTabindex(inspectorTabButtons, 0);

  const inspectorBody = el('div', 'uni-inspector-body');
  inspectorBody.dataset.testid = 'uni-inspector-body';

  inspectorDrawer.append(inspectorTablist, inspectorBody);
  workspace.append(stageColumn, inspectorDrawer);
  shell.append(topbar, workspace);
  root.appendChild(shell);

  // --------------------------------------------------------------------------
  // WEBGL2 PASS INITIALIZATION & CONTEXT LOSS RESILIENCE
  // --------------------------------------------------------------------------

  let glBundle: GlPassesBundle | null = null;

  function drainGlErrors(): void {
    if (!glBundle || glBundle.device.isLost) return;
    const gl = glBundle.device.gl;
    for (let guard = 0; guard < 16; guard++) {
      const err = gl.getError();
      if (err === gl.NO_ERROR) break;
      glErrorCount += 1;
    }
  }

  function initWebGl(): void {
    try {
      const device = new RenderDevice(canvas);
      const gl = device.gl;
      glBundle = {
        device,
        solids: new SolidsPass(gl),
        lines: new LinesPass(gl),
        ribbons: new RibbonsPass(gl),
        flatFlow: new FlatFlowRibbonsPass(gl),
        sheets: new SheetsPass(gl),
      };
      glFallback.hidden = true;
      drainGlErrors();
    } catch (err) {
      glBundle = null;
      glFallback.hidden = false;
      glFallback.textContent =
        err instanceof Error
          ? `WebGL2 renderer unavailable (${err.message}). 2D RGB insets, 28-beat tour, and architectural inspectors remain interactive.`
          : 'WebGL2 renderer unavailable.';
    }
  }

  initWebGl();

  const onContextLost = (ev: Event): void => {
    ev.preventDefault();
    if (glBundle) {
      glBundle.device.markLost(true);
    }
    stopPlayback();
    activeFly = null;
    scheduler.setContinuousReason('camera_fly', false);
    glFallback.hidden = false;
    glFallback.textContent = 'WebGL2 context lost — waiting for GPU restoration…';
  };

  const onContextRestored = (): void => {
    if (!glBundle) {
      initWebGl();
    } else {
      glBundle.device.markLost(false);
      try {
        glBundle.solids.init();
        glBundle.lines.init();
        glBundle.ribbons.init();
        glBundle.flatFlow.init();
        glBundle.sheets.init();
        atlasTracker.reset();
        sceneDirtyForGl = true;
        glFallback.hidden = true;
        drainGlErrors();
      } catch (err) {
        glFallback.hidden = false;
        glFallback.textContent =
          err instanceof Error ? `WebGL2 restore failed: ${err.message}` : 'WebGL2 restore failed.';
      }
    }
    scheduler.requestFrame();
  };

  canvas.addEventListener('webglcontextlost', onContextLost);
  canvas.addEventListener('webglcontextrestored', onContextRestored);

  // --------------------------------------------------------------------------
  // VIEWPORT METRICS, CAMERA FITTING & RENDER PIPELINE
  // --------------------------------------------------------------------------

  function readViewportMetrics(): ViewportMetrics {
    const rect = viewportHost.getBoundingClientRect();
    const w = Math.max(320, rect.width || viewportHost.clientWidth || 960);
    const h = Math.max(240, rect.height || viewportHost.clientHeight || 600);
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    return makeViewportMetrics(rect.left, rect.top, w, h, dpr, 0, 0);
  }

  function computePresetPose(scene: UniSpatialScene): CameraPose {
    const metrics = readViewportMetrics();
    const topInsetPx = 88;
    const bottomInsetPx = scene.rgbPreviews.length > 0 ? 180 : 40;
    return fitOpBoundsCamera(scene.cameraPreset.fitBounds, metrics, {
      azimuthDeg: scene.cameraPreset.azimuthDeg,
      elevationDeg: scene.cameraPreset.elevationDeg,
      topInsetPx,
      bottomInsetPx,
      fillFraction: 0.82,
    });
  }

  function flyToCurrentScenePreset(animate: boolean): void {
    const targetPose = computePresetPose(currentScene);
    reducedMotion = isUniReducedMotionPreferred();
    if (!animate || reducedMotion || !cameraPose) {
      cameraPose = targetPose;
      activeFly = null;
      scheduler.setContinuousReason('camera_fly', false);
      renderFrameNow();
      return;
    }
    activeFly = {
      from: cameraPose,
      to: targetPose,
      durationMs: CAMERA_FLY_DURATION_MS,
      elapsedMs: 0,
    };
    scheduler.setContinuousReason('camera_fly', true);
  }

  function uploadSceneToGpuIfNeeded(): void {
    if (!glBundle || glBundle.device.isLost || !sceneDirtyForGl) return;
    const scene = currentScene;

    glBundle.solids.upload(scene.solidData, scene.solidCount);
    glBundle.lines.upload(scene.lineData, scene.lineCount);
    glBundle.ribbons.upload(scene.ribbonData, scene.ribbonCount);
    glBundle.flatFlow.upload(scene.flowVertices, scene.flowIndices);
    glBundle.sheets.upload(
      scene.sheetData,
      scene.sheetCount,
      scene.sheetBasisData,
      scene.sheetRevealData,
    );

    if (scene.valueAtlas && atlasTracker.shouldUpload(scene.atlasKey)) {
      glBundle.sheets.uploadValueAtlas(
        scene.valueAtlas.width,
        scene.valueAtlas.height,
        scene.valueAtlas.data,
      );
    }

    sceneDirtyForGl = false;
    drainGlErrors();
  }

  function computeOverlayAvoidRects(metrics: ViewportMetrics): UniOverlayRect[] {
    const avoid: UniOverlayRect[] = [];
    const hostX = metrics.hostOriginX;
    const hostY = metrics.hostOriginY;

    const addElementRect = (node: HTMLElement | null): void => {
      if (!node || node.hidden) return;
      const r = node.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) return;
      avoid.push({
        x0: r.left - hostX,
        y0: r.top - hostY,
        x1: r.right - hostX,
        y1: r.bottom - hostY,
      });
    };

    addElementRect(stageTitleCard);
    if (!opdReadOnlyBanner.hidden) addElementRect(opdReadOnlyBanner);
    if (currentScene.rgbPreviews.length > 0) addElementRect(rgbInsetHost);
    if (!tourHud.hidden) addElementRect(tourHud);
    return avoid;
  }

  function renderFrameNow(): void {
    if (disposed) return;
    const metrics = readViewportMetrics();
    if (!cameraPose) {
      cameraPose = computePresetPose(currentScene);
    }

    const snapshot = buildCameraSnapshot(cameraPose, currentScene.fitBounds, metrics);
    cameraSnapshot = snapshot;

    if (glBundle && !glBundle.device.isLost) {
      uploadSceneToGpuIfNeeded();
      const fb = glBundle.device.ensureSize(metrics);
      glBundle.device.beginFrame(fb);
      glBundle.solids.draw(snapshot.viewProj);
      glBundle.sheets.draw(snapshot.viewProj);
      glBundle.lines.draw(snapshot.viewProj, fb.vpWidth, fb.vpHeight, metrics.dpr, snapshot.eye);
      glBundle.ribbons.draw(snapshot.viewProj);
      glBundle.flatFlow.draw(snapshot.viewProj, -1);
      glBundle.device.endFrame(fb);
      drainGlErrors();
    }

    overlay.update({
      scene: currentScene,
      snapshot,
      hoveredCell,
      selectedPartId,
      topInsetPx: 74,
      bottomInsetPx: currentScene.rgbPreviews.length > 0 ? 156 : 18,
      avoidRects: computeOverlayAvoidRects(metrics),
    });
  }

  // --------------------------------------------------------------------------
  // IDLE-RAF SCHEDULER
  // --------------------------------------------------------------------------

  const scheduler = createIdleRafScheduler({
    onFrame: (_nowMs, dtMs) => {
      if (activeFly) {
        activeFly.elapsedMs += Math.max(dtMs, 16);
        const t = Math.min(1, activeFly.elapsedMs / activeFly.durationMs);
        cameraPose = interpolateCameraPose(activeFly.from, activeFly.to, t);
        if (t >= 1) {
          cameraPose = activeFly.to;
          activeFly = null;
          scheduler.setContinuousReason('camera_fly', false);
        }
      }

      if (isPlaying) {
        playbackAccumMs += dtMs;
        if (playbackAccumMs >= PLAYBACK_STEP_INTERVAL_MS) {
          playbackAccumMs = 0;
          advancePlaybackStep();
        }
      }

      renderFrameNow();
    },
  });

  // --------------------------------------------------------------------------
  // SESSION & SCENE SYNCHRONIZATION
  // --------------------------------------------------------------------------

  function getActiveSession(): UniSession {
    return tour.isActive() ? tour.tourSession : freeSession;
  }

  function getActiveTaskMode(): UniSessionTaskMode {
    const sess = getActiveSession();
    const branchId = sess.getActiveBranchId();
    const branch = branchId ? sess.getBranch(branchId) : undefined;
    return branch?.taskMode ?? 'text_to_image';
  }

  function refreshRgbInset(): void {
    if (currentScene.rgbPreviews.length === 0) {
      pixelPanel.clear();
      return;
    }
    pixelPanel.renderDescriptors(currentScene.rgbPreviews, {
      scale: 8,
      showPaddingLock: focus.showPaddingLockOverlay,
      showGroundingBoxes: true,
    });
  }

  function refreshStageHud(): void {
    stageTitleEl.textContent = currentScene.title;
    stageSubtitleEl.textContent = currentScene.subtitle;

    stageBadgesEl.innerHTML = '';
    for (let i = 0; i < currentScene.provenanceBadges.length; i++) {
      const b = currentScene.provenanceBadges[i]!;
      stageBadgesEl.appendChild(el('span', `uni-badge uni-badge--${b.tone}`, b.label));
    }

    opdReadOnlyBanner.hidden = currentScene.viewMode !== 'opd_readonly';
  }

  function syncTopViewTabs(): void {
    const idx = UNI_SPATIAL_VIEW_MODES.findIndex((v) => v.id === viewMode);
    syncRovingTabindex(viewTabButtons, idx >= 0 ? idx : 0);
    for (let i = 0; i < viewTabButtons.length; i++) {
      viewTabButtons[i]!.classList.toggle('is-active', i === idx);
    }
    tourLaunchBtn.classList.toggle('is-active', tour.isActive());
    tourLaunchBtn.textContent = tour.isActive() ? 'Exit Tour' : 'Guided Tour (28)';
  }

  function syncInspectorTabs(): void {
    const idx = INSPECTOR_TABS.findIndex((t) => t.id === inspectorTab);
    syncRovingTabindex(inspectorTabButtons, idx >= 0 ? idx : 0);
    for (let i = 0; i < inspectorTabButtons.length; i++) {
      inspectorTabButtons[i]!.classList.toggle('is-active', i === idx);
    }
  }

  function refreshDock(): void {
    const sess = getActiveSession();
    const dockState = buildUniTimelineDockState(sess, {
      isPlaying,
      tourActive: tour.isActive(),
    });
    timelineDock.update(dockState);
  }

  function rebuildSceneAndRefresh(opts?: {
    readonly refitCamera?: boolean;
    readonly animateCamera?: boolean;
    readonly syncRender?: boolean;
  }): void {
    const activeSess = getActiveSession();
    if (tour.isActive()) {
      const tourState = tour.getState();
      viewMode = tourState.currentBeat.viewMode;
      focus = tourState.focus;
      selectedPartId = tourState.currentBeat.highlightPartId ?? null;
      currentScene = tour.buildBeatScene(tourState.beatIndex);
    } else {
      const activeRecId = activeSess.getActiveRecordId();
      const activeBranchId = activeSess.getActiveBranchId();
      const activeBranch = activeBranchId ? activeSess.getBranch(activeBranchId) : undefined;
      const clampedRow = Math.max(
        0,
        Math.min((activeBranch?.cfgBatchSize ?? 1) - 1, focus.batchRow),
      );
      const clampedCfgRow = Math.max(
        0,
        Math.min((activeBranch?.cfgBatchSize ?? 1) - 1, focus.cfgBranchRow),
      );
      const nextFocus: UniSpatialFocusState = {
        ...focus,
        batchRow: clampedRow,
        cfgBranchRow: clampedCfgRow,
        hoveredPartId: selectedPartId,
        hoveredCell,
        ...(activeRecId !== null ? { recordId: activeRecId } : {}),
      };
      focus = nextFocus;
      currentScene = buildUniSpatialScene(viewMode, activeSess, focus);
    }

    sceneDirtyForGl = true;
    syncTopViewTabs();
    refreshStageHud();
    refreshRgbInset();
    refreshDock();
    renderInspectorBody();

    if (opts?.refitCamera) {
      flyToCurrentScenePreset(opts.animateCamera === true);
    } else if (opts?.syncRender) {
      renderFrameNow();
    } else {
      scheduler.requestFrame();
    }
  }

  // --------------------------------------------------------------------------
  // VIEW, TASK, D5 PREFILL & TRAJECTORY ACTIONS
  // --------------------------------------------------------------------------

  function setViewInternal(nextView: UniSpatialViewMode, animateCamera: boolean): void {
    viewMode = nextView;
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    rebuildSceneAndRefresh({
      refitCamera: true,
      animateCamera,
      syncRender: !animateCamera,
    });
  }

  function ensureFreeBranchForTask(
    taskMode: UniSessionTaskMode,
    prefillMode: UniElasticPrefillMode,
  ): void {
    const isGenTask = taskMode === 'text_to_image' || taskMode === 'image_edit';
    const existing = freeSession
      .getAllBranches()
      .find((b) => b.taskMode === taskMode && (!isGenTask || b.prefillMode === prefillMode));
    if (existing && existing.recordIds.length > 0) {
      freeSession.setActiveRecord(existing.recordIds[existing.recordIds.length - 1]!);
      return;
    }

    const isEdit = taskMode === 'image_edit';
    const branch = freeSession.startBranch({
      taskMode,
      prefillMode,
      gainPresetId: focus.gainPresetId ?? 'mini_default',
      ...(isEdit
        ? {
            scenePresetId: 'harbor_panorama_16x8',
            guidanceMode: focus.editGuidanceMode ?? 'legacy',
            lockSourcePaddingVq: true,
            cropToSource: true,
          }
        : taskMode === 'visual_understanding' || taskMode === 'object_grounding'
          ? { scenePresetId: 'sunlit_cat_16x16' }
          : {}),
      activationSteps: 4,
      refinementSteps: 4,
    });
    const maxInitSteps =
      taskMode === 'object_grounding' ? 2 : taskMode === 'text_to_image' || isEdit ? 4 : 3;
    freeSession.runBranchToCompletion(branch.branchId, maxInitSteps);
  }

  function setTaskModeInternal(taskMode: UniSessionTaskMode, syncRender = false): void {
    stopPlayback();
    if (tour.isActive()) {
      exitTourInternal(false);
    }
    ensureFreeBranchForTask(taskMode, focus.prefillMode);
    const activeBranch = freeSession.getActiveBranchId()
      ? freeSession.getBranch(freeSession.getActiveBranchId()!)
      : undefined;
    const highestRow = Math.max(0, (activeBranch?.cfgBatchSize ?? 1) - 1);
    focus = {
      ...focus,
      executionViewId: taskMode,
      batchRow: highestRow,
      cfgBranchRow: highestRow,
      ...(taskMode === 'image_edit' ? { scenePresetId: 'harbor_panorama_16x8' } : {}),
    };
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    rebuildSceneAndRefresh({ refitCamera: false, syncRender });
  }

  function selectDockTaskChoice(choice: UniDockTaskChoice): void {
    setTaskModeInternal(taskChoiceToSessionTaskMode(choice), false);
  }

  function setPrefillModeInternal(mode: UniElasticPrefillMode, syncRender = false): void {
    focus = {
      ...focus,
      prefillMode: mode,
    };

    if (!tour.isActive()) {
      const curBranchId = freeSession.getActiveBranchId();
      const curBranch = curBranchId ? freeSession.getBranch(curBranchId) : undefined;
      if (
        curBranch &&
        curBranch.prefillMode !== mode &&
        (curBranch.taskMode === 'text_to_image' || curBranch.taskMode === 'image_edit')
      ) {
        const match = freeSession
          .getAllBranches()
          .find((b) => b.taskMode === curBranch.taskMode && b.prefillMode === mode);
        if (match && match.recordIds.length > 0) {
          freeSession.setActiveRecord(match.recordIds[match.recordIds.length - 1]!);
        } else {
          const curRecId = freeSession.getActiveRecordId();
          if (curRecId) {
            const forked = freeSession.forkBranch(curRecId, {
              label: `${curBranch.taskMode} · ${mode}`,
              prefillMode: mode,
            });
            freeSession.stepBranch(forked.branchId);
          }
        }
      }
    }

    rebuildSceneAndRefresh({ refitCamera: false, syncRender });
  }

  function stepCursorBy(delta: number): void {
    const sess = getActiveSession();
    const branchId = sess.getActiveBranchId();
    const branch = branchId ? sess.getBranch(branchId) : undefined;
    if (!branch || branch.recordIds.length === 0) return;
    const curId = sess.getActiveRecordId();
    const curIdx = curId ? branch.recordIds.indexOf(curId) : branch.recordIds.length - 1;
    const nextIdx = Math.max(0, Math.min(branch.recordIds.length - 1, curIdx + delta));
    const targetId = branch.recordIds[nextIdx];
    if (targetId && targetId !== curId) {
      sess.setActiveRecord(targetId);
      rebuildSceneAndRefresh();
    }
  }

  function scrubToRecordIndex(indexInBranch: number): void {
    const sess = getActiveSession();
    const branchId = sess.getActiveBranchId();
    const branch = branchId ? sess.getBranch(branchId) : undefined;
    if (!branch || branch.recordIds.length === 0) return;
    const clamped = Math.max(0, Math.min(branch.recordIds.length - 1, Math.trunc(indexInBranch)));
    const targetId = branch.recordIds[clamped];
    if (targetId) {
      sess.setActiveRecord(targetId);
      rebuildSceneAndRefresh();
    }
  }

  function executeNextStep(): void {
    if (tour.isActive()) return;
    const rec = freeSession.stepBranch();
    if (rec) {
      rebuildSceneAndRefresh();
    } else {
      refreshDock();
    }
  }

  function forkCurrentBranch(): void {
    if (tour.isActive()) return;
    const activeRecId = freeSession.getActiveRecordId();
    if (!activeRecId) return;
    const forked = freeSession.forkBranch(activeRecId, {
      prefillMode: focus.prefillMode,
    });
    freeSession.stepBranch(forked.branchId);
    rebuildSceneAndRefresh();
  }

  function switchToBranch(branchId: string): void {
    if (tour.isActive()) return;
    const branch = freeSession.getBranch(branchId);
    if (!branch || branch.recordIds.length === 0) return;
    freeSession.setActiveRecord(branch.recordIds[branch.recordIds.length - 1]!);
    focus = {
      ...focus,
      executionViewId: branch.taskMode,
      prefillMode: branch.prefillMode,
      batchRow: Math.max(0, branch.cfgBatchSize - 1),
      cfgBranchRow: Math.max(0, branch.cfgBatchSize - 1),
    };
    rebuildSceneAndRefresh();
  }

  function stopPlayback(): void {
    if (!isPlaying) return;
    isPlaying = false;
    playbackAccumMs = 0;
    scheduler.setContinuousReason('playback', false);
    refreshDock();
  }

  function togglePlayback(): void {
    if (isPlaying) {
      stopPlayback();
      return;
    }
    const sess = getActiveSession();
    const branchId = sess.getActiveBranchId();
    const branch = branchId ? sess.getBranch(branchId) : undefined;
    if (!branch || branch.recordIds.length <= 1) return;
    isPlaying = true;
    playbackAccumMs = 0;
    scheduler.setContinuousReason('playback', true);
    refreshDock();
  }

  function advancePlaybackStep(): void {
    const sess = getActiveSession();
    const branchId = sess.getActiveBranchId();
    const branch = branchId ? sess.getBranch(branchId) : undefined;
    if (!branch || branch.recordIds.length <= 1) {
      stopPlayback();
      return;
    }
    const curId = sess.getActiveRecordId();
    const curIdx = curId ? branch.recordIds.indexOf(curId) : 0;
    if (curIdx >= branch.recordIds.length - 1) {
      stopPlayback();
      return;
    }
    sess.setActiveRecord(branch.recordIds[curIdx + 1]!);
    rebuildSceneAndRefresh();
  }

  // --------------------------------------------------------------------------
  // 28-BEAT GUIDED TOUR HUD INTEGRATION (ISOLATED PRIVATE SESSION)
  // --------------------------------------------------------------------------

  function updateTourHudDom(state: UniTourState): void {
    tourHud.hidden = !state.active;
    if (!state.active) return;

    const beat = state.currentBeat;
    tourBeatCounter.textContent = `Act ${beat.actNumber} · Beat ${beat.index + 1} / ${state.beatCount}`;
    tourTitleEl.textContent = beat.title;
    tourTakeawayEl.textContent = beat.takeaway;
    tourNarrativeEl.textContent = beat.narrative;

    if (beat.equation) {
      tourEquationEl.hidden = false;
      tourEquationEl.textContent = beat.equation;
    } else {
      tourEquationEl.hidden = true;
      tourEquationEl.textContent = '';
    }

    for (let i = 0; i < tourActButtons.length; i++) {
      tourActButtons[i]!.classList.toggle('is-active', i === state.actIndex);
    }

    tourRefsEl.innerHTML = '';
    for (let i = 0; i < beat.ledgerRefs.length; i++) {
      const refId = beat.ledgerRefs[i]!;
      const chip = el('button', 'uni-tour-ref-chip', refId);
      chip.type = 'button';
      chip.title = `Inspect ${refId} in Ledger`;
      chip.addEventListener('click', () => {
        if (inspectorCollapsed) toggleInspectorCollapsed();
        inspectorTab = 'ledger';
        ledgerFilter = 'all';
        ledgerQuery = refId;
        syncInspectorTabs();
        renderInspectorBody();
      });
      tourRefsEl.appendChild(chip);
    }

    tourPrevBtn.disabled = state.beatIndex <= 0;
    tourNextBtn.textContent =
      state.beatIndex >= state.beatCount - 1 ? 'Finish Tour' : 'Next →';
  }

  function startTourInternal(beatIndex = 0, syncRender = false): void {
    stopPlayback();
    const restore: UniFreeExploreRestoreState = {
      viewMode,
      focus: { ...focus },
      activeBranchId: freeSession.getActiveBranchId(),
      activeRecordId: freeSession.getActiveRecordId(),
    };
    const state = tour.start(restore, beatIndex);
    hoveredCell = null;
    selectedCell = null;
    updateTourHudDom(state);
    tourFocusTrap.activate();
    rebuildSceneAndRefresh({
      refitCamera: true,
      animateCamera: !syncRender,
      syncRender,
    });
  }

  function goToTourBeatInternal(beatIndex: number, syncRender = false): void {
    if (!tour.isActive()) {
      startTourInternal(beatIndex, syncRender);
      return;
    }
    const state = tour.goToBeat(beatIndex);
    hoveredCell = null;
    selectedCell = null;
    updateTourHudDom(state);
    rebuildSceneAndRefresh({
      refitCamera: true,
      animateCamera: !syncRender,
      syncRender,
    });
  }

  function exitTourInternal(restoreFreeExplore: boolean, syncRender = false): void {
    if (!tour.isActive()) return;
    const restored = tour.exit();
    tourFocusTrap.deactivate();
    tourHud.hidden = true;
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;

    if (restoreFreeExplore && restored) {
      viewMode = restored.viewMode;
      focus = { ...restored.focus };
      if (restored.activeRecordId && freeSession.getRecord(restored.activeRecordId)) {
        freeSession.setActiveRecord(restored.activeRecordId);
      }
      rebuildSceneAndRefresh({
        refitCamera: true,
        animateCamera: !syncRender,
        syncRender,
      });
    } else {
      syncTopViewTabs();
      refreshDock();
    }
  }

  // --------------------------------------------------------------------------
  // COLLAPSIBLE 3-TAB INSPECTOR DRAWER RENDERING
  // --------------------------------------------------------------------------

  function toggleInspectorCollapsed(): void {
    inspectorCollapsed = !inspectorCollapsed;
    inspectorDrawer.classList.toggle('is-collapsed', inspectorCollapsed);
    inspectorToggleBtn.textContent = inspectorCollapsed ? 'Show Inspector' : 'Hide Inspector';
    inspectorToggleBtn.setAttribute('aria-expanded', inspectorCollapsed ? 'false' : 'true');
    scheduler.requestFrame();
  }

  function renderViewControlsSection(container: HTMLElement): void {
    const sec = el('section', 'uni-inspector-section');
    sec.appendChild(
      el('h3', 'uni-section-heading', `View Controls · ${currentScene.viewMode.replace('_', ' ')}`),
    );

    if (viewMode === 'macro' || viewMode === 'joint_attn') {
      // Live D5 Prefill Mode Toggle
      const d5Row = el('div', 'uni-control-row');
      d5Row.appendChild(el('div', 'uni-control-label', 'D5 Elastic Prefill Mode'));
      const d5Chips = el('div', 'uni-chip-group');

      const hfBtn = el('button', 'uni-chip-btn uni-chip-btn--amber', 'HF-Faithful (34L U)');
      hfBtn.type = 'button';
      hfBtn.dataset.testid = 'uni-inspector-d5-hf';
      hfBtn.classList.toggle('is-active', focus.prefillMode === 'hf_faithful');
      hfBtn.addEventListener('click', () => setPrefillModeInternal('hf_faithful'));

      const prunedBtn = el('button', 'uni-chip-btn uni-chip-btn--teal', 'Elastic-Pruned (18L U)');
      prunedBtn.type = 'button';
      prunedBtn.dataset.testid = 'uni-inspector-d5-pruned';
      prunedBtn.classList.toggle('is-active', focus.prefillMode === 'elastic_pruned');
      prunedBtn.addEventListener('click', () => setPrefillModeInternal('elastic_pruned'));

      d5Chips.append(hfBtn, prunedBtn);
      d5Row.appendChild(d5Chips);
      sec.appendChild(d5Row);

      // Layer Presets (L0, L17, L18, L33) + 0..33 Slider
      const activeLayer =
        viewMode === 'macro'
          ? (focus.selectedLayerIndex ?? 17)
          : focus.jointLayerIndex;
      const layerRow = el('div', 'uni-control-row');
      const layerLbl = el('div', 'uni-control-label');
      layerLbl.append(
        el('span', undefined, 'Layer Index (0..33)'),
        el(
          'span',
          'uni-control-value',
          `L${activeLayer} · ${activeLayer < 18 ? 'Joint' : 'Split U⊥G'}`,
        ),
      );
      const presetChips = el('div', 'uni-chip-group');
      for (const preset of LAYER_PRESETS) {
        const b = el('button', 'uni-chip-btn', `L${preset}`);
        b.type = 'button';
        b.classList.toggle('is-active', activeLayer === preset);
        b.addEventListener('click', () => {
          focus =
            viewMode === 'macro'
              ? { ...focus, selectedLayerIndex: preset }
              : { ...focus, jointLayerIndex: preset };
          rebuildSceneAndRefresh();
        });
        presetChips.appendChild(b);
      }
      const slider = el('input', 'uni-range-input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = '33';
      slider.value = String(activeLayer);
      slider.setAttribute('aria-label', 'Select transformer layer index');
      slider.addEventListener('input', () => {
        const val = Number.parseInt(slider.value, 10);
        if (!Number.isFinite(val)) return;
        focus =
          viewMode === 'macro'
            ? { ...focus, selectedLayerIndex: val }
            : { ...focus, jointLayerIndex: val };
        rebuildSceneAndRefresh();
      });
      layerRow.append(layerLbl, presetChips, slider);
      sec.appendChild(layerRow);
    }

    if (viewMode === 'macro') {
      const execRow = el('div', 'uni-control-row');
      execRow.appendChild(el('div', 'uni-control-label', 'Execution View (Tower Dimming)'));
      const chips = el('div', 'uni-chip-group');
      const curExec = focus.executionViewId ?? currentScene.executionViewId;
      for (const choice of EXECUTION_VIEW_CHOICES) {
        const b = el('button', 'uni-chip-btn', choice.label);
        b.type = 'button';
        b.classList.toggle('is-active', curExec === choice.id);
        b.addEventListener('click', () => {
          focus = { ...focus, executionViewId: choice.id };
          rebuildSceneAndRefresh();
        });
        chips.appendChild(b);
      }
      execRow.appendChild(chips);
      sec.appendChild(execRow);
    }

    if (viewMode === 'visual_input') {
      const presetRow = el('div', 'uni-control-row');
      presetRow.appendChild(el('div', 'uni-control-label', 'Procedural RGB Scene Preset'));
      const chips = el('div', 'uni-chip-group');
      const curPreset = focus.scenePresetId ?? 'sunlit_cat_16x16';
      for (const p of SCENE_PRESET_CHOICES) {
        const b = el('button', 'uni-chip-btn', p.label);
        b.type = 'button';
        b.classList.toggle('is-active', curPreset === p.id);
        b.addEventListener('click', () => {
          focus = { ...focus, scenePresetId: p.id };
          rebuildSceneAndRefresh();
        });
        chips.appendChild(b);
      }
      presetRow.appendChild(chips);
      sec.appendChild(presetRow);

      const gainRow = el('div', 'uni-control-row');
      gainRow.appendChild(el('div', 'uni-control-label', 'D3 Multimodal Gain Preset'));
      const gChips = el('div', 'uni-chip-group');
      const curGain: UniGainPresetId = focus.gainPresetId ?? 'mini_default';
      for (const g of [
        { id: 'mini_default' as const, label: 'Mini Default (0.25)' },
        { id: 'prod_exact' as const, label: 'Prod-Exact (0.01)' },
      ]) {
        const b = el('button', 'uni-chip-btn uni-chip-btn--teal', g.label);
        b.type = 'button';
        b.classList.toggle('is-active', curGain === g.id);
        b.addEventListener('click', () => {
          focus = { ...focus, gainPresetId: g.id };
          rebuildSceneAndRefresh();
        });
        gChips.appendChild(b);
      }
      gainRow.appendChild(gChips);
      sec.appendChild(gainRow);

      const discLabel = el('label', 'uni-checkbox-label');
      const discCb = el('input');
      discCb.type = 'checkbox';
      discCb.checked = focus.includeDiscreteStream !== false;
      discCb.addEventListener('change', () => {
        focus = { ...focus, includeDiscreteStream: discCb.checked };
        rebuildSceneAndRefresh();
      });
      discLabel.append(discCb, document.createTextNode('Show Emu3 IBQ + Packed Codec Stream (G)'));
      sec.appendChild(discLabel);
    }

    if (viewMode === 'joint_attn') {
      const sess = getActiveSession();
      const activeBranch = sess.getActiveBranchId()
        ? sess.getBranch(sess.getActiveBranchId()!)
        : undefined;
      const cfgSize = activeBranch?.cfgBatchSize ?? 1;

      const cfgRow = el('div', 'uni-control-row');
      cfgRow.appendChild(el('div', 'uni-control-label', `CFG Batch Row (B_cfg = ${cfgSize})`));
      const cfgChips = el('div', 'uni-chip-group');
      for (let r = 0; r < cfgSize; r++) {
        const label =
          cfgSize === 3
            ? r === 0
              ? 'Row 0 (Uncond)'
              : r === 1
                ? 'Row 1 (VQ-Drop)'
                : 'Row 2 (Full)'
            : cfgSize === 2
              ? r === 0
                ? 'Row 0 (Uncond)'
                : 'Row 1 (Cond)'
              : 'Row 0';
        const b = el('button', 'uni-chip-btn', label);
        b.type = 'button';
        b.classList.toggle('is-active', focus.batchRow === r);
        b.addEventListener('click', () => {
          focus = { ...focus, batchRow: r };
          rebuildSceneAndRefresh();
        });
        cfgChips.appendChild(b);
      }
      cfgRow.appendChild(cfgChips);
      sec.appendChild(cfgRow);

      const qRow = el('div', 'uni-control-row');
      const qLbl = el('div', 'uni-control-label');
      qLbl.append(
        el('span', undefined, 'Focused Q Head (4:1 GQA)'),
        el(
          'span',
          'uni-control-value',
          `Q${focus.selectedQHead} → KV${Math.floor(focus.selectedQHead / 4)}`,
        ),
      );
      const qChips = el('div', 'uni-chip-group');
      for (let h = 0; h < 8; h++) {
        const b = el('button', 'uni-chip-btn uni-chip-btn--teal', `Q${h}`);
        b.type = 'button';
        b.classList.toggle('is-active', focus.selectedQHead === h);
        b.addEventListener('click', () => {
          focus = { ...focus, selectedQHead: h };
          rebuildSceneAndRefresh();
        });
        qChips.appendChild(b);
      }
      qRow.append(qLbl, qChips);
      sec.appendChild(qRow);
    }

    if (viewMode === 'ball_tree') {
      const codeRow = el('div', 'uni-control-row');
      const codeLbl = el('div', 'uni-control-label');
      codeLbl.append(
        el('span', undefined, 'Selected VQ Code (0..255)'),
        el('span', 'uni-control-value', `#${focus.selectedCodeId}`),
      );
      const codeSlider = el('input', 'uni-range-input');
      codeSlider.type = 'range';
      codeSlider.min = '0';
      codeSlider.max = '255';
      codeSlider.value = String(focus.selectedCodeId);
      codeSlider.setAttribute('aria-label', 'Selected VQ codebook ID');
      codeSlider.addEventListener('input', () => {
        const val = Number.parseInt(codeSlider.value, 10);
        if (Number.isFinite(val)) {
          focus = { ...focus, selectedCodeId: val };
          rebuildSceneAndRefresh();
        }
      });
      codeRow.append(codeLbl, codeSlider);
      sec.appendChild(codeRow);

      const nRow = el('div', 'uni-control-row');
      nRow.appendChild(el('div', 'uni-control-label', 'Refinement Steps N (N=1 ≡ Binary MDM)'));
      const nChips = el('div', 'uni-chip-group');
      for (const n of REFINEMENT_STEP_PRESETS) {
        const b = el('button', 'uni-chip-btn uni-chip-btn--teal', `N = ${n}`);
        b.type = 'button';
        b.classList.toggle('is-active', focus.refinementSteps === n);
        b.addEventListener('click', () => {
          const clampedStage = Math.min(n, focus.stageIndex);
          focus = { ...focus, refinementSteps: n, stageIndex: clampedStage };
          rebuildSceneAndRefresh();
        });
        nChips.appendChild(b);
      }
      nRow.appendChild(nChips);
      sec.appendChild(nRow);

      const stageRow = el('div', 'uni-control-row');
      const stageLbl = el('div', 'uni-control-label');
      stageLbl.append(
        el('span', undefined, 'Active Stage (-1..N)'),
        el('span', 'uni-control-value', `stage = ${focus.stageIndex}`),
      );
      const stageSlider = el('input', 'uni-range-input');
      stageSlider.type = 'range';
      stageSlider.min = '-1';
      stageSlider.max = String(focus.refinementSteps);
      stageSlider.value = String(focus.stageIndex);
      stageSlider.setAttribute('aria-label', 'Ball-tree stage index');
      stageSlider.addEventListener('input', () => {
        const val = Number.parseInt(stageSlider.value, 10);
        if (Number.isFinite(val)) {
          focus = { ...focus, stageIndex: val };
          rebuildSceneAndRefresh();
        }
      });
      stageRow.append(stageLbl, stageSlider);
      sec.appendChild(stageRow);

      const u26Label = el('label', 'uni-checkbox-label');
      const u26Cb = el('input');
      u26Cb.type = 'checkbox';
      u26Cb.checked = focus.showProductionD17Comparison;
      u26Cb.addEventListener('change', () => {
        focus = { ...focus, showProductionD17Comparison: u26Cb.checked };
        rebuildSceneAndRefresh();
      });
      u26Label.append(
        u26Cb,
        document.createTextNode('Highlight U26 Production D=17 Floor-Rounding Mismatch'),
      );
      sec.appendChild(u26Label);
    }

    if (viewMode === 'refinement') {
      const stratLabel = el('label', 'uni-checkbox-label');
      const stratCb = el('input');
      stratCb.type = 'checkbox';
      stratCb.checked = focus.showStratifiedOrder;
      stratCb.addEventListener('change', () => {
        focus = { ...focus, showStratifiedOrder: stratCb.checked };
        rebuildSceneAndRefresh();
      });
      stratLabel.append(
        stratCb,
        document.createTextNode('Show stratified_random(4, 42) Quadtree Order'),
      );
      sec.appendChild(stratLabel);

      const cellRow = el('div', 'uni-control-row');
      const cellLbl = el('div', 'uni-control-label');
      cellLbl.append(
        el('span', undefined, 'Marked 4×6 Grid Cell (0..23)'),
        el('span', 'uni-control-value', `#${focus.selectedGridCell ?? 0}`),
      );
      const cellSlider = el('input', 'uni-range-input');
      cellSlider.type = 'range';
      cellSlider.min = '0';
      cellSlider.max = '23';
      cellSlider.value = String(focus.selectedGridCell ?? 0);
      cellSlider.setAttribute('aria-label', 'Select marked 4x6 grid cell');
      cellSlider.addEventListener('input', () => {
        const val = Number.parseInt(cellSlider.value, 10);
        if (Number.isFinite(val)) {
          focus = { ...focus, selectedGridCell: val };
          rebuildSceneAndRefresh();
        }
      });
      cellRow.append(cellLbl, cellSlider);
      sec.appendChild(cellRow);
    }

    if (viewMode === 'output_edit') {
      const modeRow = el('div', 'uni-control-row');
      modeRow.appendChild(el('div', 'uni-control-label', 'Dual-CFG Edit Guidance Mode (U16)'));
      const mChips = el('div', 'uni-chip-group');
      const curMode: UniEditGuidanceMode = focus.editGuidanceMode ?? 'legacy';
      for (const m of ['legacy', 'independent', 'ip2p'] as const) {
        const b = el('button', 'uni-chip-btn uni-chip-btn--teal', m);
        b.type = 'button';
        b.classList.toggle('is-active', curMode === m);
        b.addEventListener('click', () => {
          focus = { ...focus, editGuidanceMode: m };
          rebuildSceneAndRefresh();
        });
        mChips.appendChild(b);
      }
      modeRow.appendChild(mChips);
      sec.appendChild(modeRow);

      const cropLabel = el('label', 'uni-checkbox-label');
      const cropCb = el('input');
      cropCb.type = 'checkbox';
      cropCb.checked = focus.cropToSource;
      cropCb.addEventListener('change', () => {
        focus = { ...focus, cropToSource: cropCb.checked };
        rebuildSceneAndRefresh();
      });
      cropLabel.append(cropCb, document.createTextNode('Crop Decoded 16×16 Back to Native Source'));
      sec.appendChild(cropLabel);

      const lockLabel = el('label', 'uni-checkbox-label');
      const lockCb = el('input');
      lockCb.type = 'checkbox';
      lockCb.checked = focus.showPaddingLockOverlay;
      lockCb.addEventListener('change', () => {
        focus = { ...focus, showPaddingLockOverlay: lockCb.checked };
        rebuildSceneAndRefresh();
      });
      lockLabel.append(
        lockCb,
        document.createTextNode('Show U18 lock_source_padding_vq Hatch Overlay'),
      );
      sec.appendChild(lockLabel);
    }

    if (viewMode === 'opd_readonly') {
      sec.classList.add('uni-inspector-section--readonly');
      const bayRow = el('div', 'uni-control-row');
      bayRow.appendChild(el('div', 'uni-control-label', 'OPD Read-Only Bay Focus'));
      const bChips = el('div', 'uni-chip-group');
      for (const bay of [
        { id: 'both' as UniOpdBayFocus, label: 'Both Bays' },
        { id: 'understanding' as UniOpdBayFocus, label: 'Understanding (U)' },
        { id: 'editing' as UniOpdBayFocus, label: 'Flow Editing (G)' },
      ]) {
        const b = el('button', 'uni-chip-btn', bay.label);
        b.type = 'button';
        b.classList.toggle('is-active', focus.opdSelectedBay === bay.id);
        b.addEventListener('click', () => {
          focus = { ...focus, opdSelectedBay: bay.id };
          rebuildSceneAndRefresh();
        });
        bChips.appendChild(b);
      }
      bayRow.appendChild(bChips);
      sec.appendChild(bayRow);

      const rRow = el('div', 'uni-control-row');
      const rLbl = el('div', 'uni-control-label');
      rLbl.append(
        el('span', undefined, 'Flow Noising Ratio r'),
        el('span', 'uni-control-value', focus.opdR.toFixed(2)),
      );
      const rSlider = el('input', 'uni-range-input');
      rSlider.type = 'range';
      rSlider.min = '0.5';
      rSlider.max = '0.95';
      rSlider.step = '0.05';
      rSlider.value = String(focus.opdR);
      rSlider.setAttribute('aria-label', 'Continuous-flow OPD noising ratio r');
      rSlider.addEventListener('input', () => {
        const val = Number.parseFloat(rSlider.value);
        if (Number.isFinite(val)) {
          focus = { ...focus, opdR: val };
          rebuildSceneAndRefresh();
        }
      });
      rRow.append(rLbl, rSlider);
      sec.appendChild(rRow);
    }

    container.appendChild(sec);
  }

  function renderCellInspectSection(container: HTMLElement): void {
    const sec = el('section', 'uni-inspector-section');
    sec.appendChild(el('h3', 'uni-section-heading', 'Ray-Pick Cell & Tensor Inspector'));
    const cell = hoveredCell ?? selectedCell;
    if (!cell) {
      sec.appendChild(
        el(
          'p',
          'uni-note-text',
          'Hover or click any 3D tensor slab, attention plate, codebook node, or ball-membership row to inspect exact layer, branch, GQA head, rotary plane, or tree depth.',
        ),
      );
      container.appendChild(sec);
      return;
    }

    const grid = el('div', 'uni-kv-grid');
    if (cell.partId) addKvRow(grid, 'Part', cell.partId);
    if (cell.tensorKey) addKvRow(grid, 'Tensor', cell.tensorKey);
    if (cell.branch) addKvRow(grid, 'Tower Branch', cell.branch.toUpperCase());
    if (cell.layerIndex !== null) {
      addKvRow(
        grid,
        'Layer',
        `L${cell.layerIndex} (${cell.isSplitLayer ? 'Split U⊥G' : 'Joint'})`,
      );
    }
    if (cell.qHeadIndex !== null) addKvRow(grid, 'Q Head', `Q${cell.qHeadIndex}`);
    if (cell.kvHeadIndex !== null) addKvRow(grid, 'KV Head', `KV${cell.kvHeadIndex}`);
    if (cell.cfgBatchRow !== null) addKvRow(grid, 'CFG Row', `B[${cell.cfgBatchRow}]`);
    if (cell.rotaryPlane !== null) addKvRow(grid, 'YaRN Plane', `Plane ${cell.rotaryPlane} / 3`);

    // Explicit `ball_membership` per-row depth awareness (rows 0..8 map to tree depths 0..8)
    if (cell.tensorKey === 'ball_membership' && cell.row !== null) {
      if (cell.row >= 0 && cell.row <= 8) {
        const ballSize = 1 << (8 - cell.row);
        addKvRow(grid, 'Ball Depth (row)', `Depth ${cell.row} / 8 (|N| = ${ballSize} codes)`);
      } else {
        addKvRow(grid, 'Membership Row', `Row ${cell.row} (funnel / proximity / U26 strip)`);
      }
    } else if (cell.treeDepth !== null) {
      addKvRow(grid, 'Tree Depth', `Depth ${cell.treeDepth} / 8`);
    }

    if (cell.row !== null && cell.col !== null) {
      addKvRow(grid, 'Cell [row, col]', `[${cell.row}, ${cell.col}]`);
    } else if (cell.tokenIndex !== null) {
      addKvRow(grid, 'Token / Code', `#${cell.tokenIndex}`);
    }
    if (cell.absPosition !== null) addKvRow(grid, 'Abs Pos', String(cell.absPosition));
    if (cell.semanticTag) addKvRow(grid, 'Semantic Tag', cell.semanticTag);

    sec.appendChild(grid);
    container.appendChild(sec);
  }

  function renderViewDiagnosticsSection(container: HTMLElement): void {
    // View 4: U26 Production D=17 vs Mini D=8 Floor-Rounding Mismatch Panel
    if (viewMode === 'ball_tree' && currentScene.u26Mismatch) {
      const report = currentScene.u26Mismatch;
      const sec = el(
        'section',
        `uni-inspector-section${report.hasMismatch ? ' uni-inspector-section--amber' : ''}`,
      );
      sec.dataset.testid = 'uni-u26-mismatch-panel';
      sec.appendChild(
        el(
          'h3',
          'uni-section-heading',
          `U26 Prime-Depth D=${report.treeDepth} Floor-Rounding Mismatch (Mini D=8: 0 drift)`,
        ),
      );
      sec.appendChild(
        el(
          'p',
          'uni-note-text',
          report.hasMismatch
            ? `${report.mismatchedStageCount}/${report.totalInternalStages} internal stages diverge at N=${report.refinementSteps}: inference samples 2× wider subtrees than training.`
            : `N=${report.refinementSteps} has 0 internal stages (reduces directly to binary MDM).`,
        ),
      );

      if (report.rows.length > 0) {
        const table = el('table', 'uni-data-table');
        const thead = el('thead');
        const hr = el('tr');
        for (const h of ['Stage', 'Train L', 'Infer L', 'Train |N|', 'Infer |N|', 'Ratio']) {
          hr.appendChild(el('th', undefined, h));
        }
        thead.appendChild(hr);
        const tbody = el('tbody');
        for (const row of report.rows) {
          const tr = el('tr', row.levelDelta !== 0 ? 'is-mismatch' : undefined);
          tr.append(
            el('td', undefined, String(row.stageIndex)),
            el('td', undefined, String(row.trainLevel)),
            el('td', undefined, String(row.inferLevel)),
            el('td', undefined, String(row.trainBallSize)),
            el('td', undefined, String(row.inferBallSize)),
            el('td', undefined, `${row.ballSizeRatio}×`),
          );
          tbody.appendChild(tr);
        }
        table.append(thead, tbody);
        sec.appendChild(table);
      }
      container.appendChild(sec);
    }

    // View 7: Read-Only OPD & U28 Diagnostics
    if (viewMode === 'opd_readonly' && currentScene.opdReadOnly) {
      const opd = currentScene.opdReadOnly;
      const sec = el('section', 'uni-inspector-section uni-inspector-section--readonly');
      sec.dataset.testid = 'uni-opd-diagnostics-panel';
      sec.appendChild(el('h3', 'uni-section-heading', `${opd.badgeLabel} · Live Readouts`));

      const grid = el('div', 'uni-kv-grid');
      addKvRow(grid, 'Gradients', String(opd.hasGradients));
      addKvRow(
        grid,
        'Understanding OPD',
        `total=${opd.understandingOpd.totalUnderstandingOpdLoss.toFixed(4)} (AR=${opd.understandingOpd.arOpdLoss.toFixed(4)}, Self=${opd.understandingOpd.selfConsistencyLoss.toFixed(4)})`,
      );
      addKvRow(
        grid,
        '129-Bucket tail_kl',
        `${opd.understandingOpd.arTailKl.bucketCount} buckets · targetTail=${opd.understandingOpd.arTailKl.pTargetTailMass.toFixed(4)}`,
      );
      addKvRow(
        grid,
        'Flow matched_sigmas',
        `${opd.flowEditingOpd.matchedSigmas.sigmaStart.toFixed(2)} → ${opd.flowEditingOpd.matchedSigmas.sigmaEnd.toFixed(2)} (mu=${opd.flowEditingOpd.matchedSigmas.mu.toFixed(3)}, ${opd.flowEditingOpd.matchedSigmas.steps} steps)`,
      );
      addKvRow(
        grid,
        'Flow Hybrid CE',
        `total=${opd.flowEditingOpd.totalEditingOpdLoss.toFixed(4)} (exact=${opd.flowEditingOpd.exactCe.toFixed(4)} + 0.2·grp=${opd.flowEditingOpd.groupedCe.toFixed(4)})`,
      );
      sec.appendChild(grid);

      const u28 = opd.u28Diagnostics;
      const u28Sec = el('section', 'uni-inspector-section uni-inspector-section--amber');
      u28Sec.appendChild(
        el('h3', 'uni-section-heading', 'U28 Verified Reference-Trainer Bugs (minimal_training/)'),
      );
      const u28Grid = el('div', 'uni-kv-grid');
      addKvRow(
        u28Grid,
        '1. SFT Double-Norm',
        `${u28.sftDoubleNorm.expectedInflationFactor.toFixed(3)}× avg / ${u28.sftDoubleNorm.peakSpikeAtTMin.toFixed(0)}× peak @ t=${u28.sftDoubleNorm.tMin}`,
      );
      addKvRow(
        u28Grid,
        '2. <|im_end|> Leak',
        `id=${u28.accidentalImEndSupervision.eosTokenId} (${u28.accidentalImEndSupervision.affectedTurns})`,
      );
      addKvRow(
        u28Grid,
        '3. <LOC_*> Tail Leak',
        `${u28.opdLocTailLeak.leakedTailDenom.toLocaleString('en-US')} vs ${u28.opdLocTailLeak.trueTeacherTailDenom.toLocaleString('en-US')} (${u28.opdLocTailLeak.locExtensionTokens} loc tokens)`,
      );
      u28Sec.appendChild(u28Grid);
      container.append(sec, u28Sec);
    }

    // D5 Summary Card (Views 1 & 3)
    if ((viewMode === 'macro' || viewMode === 'joint_attn') && currentScene.d5Comparison) {
      const d5 = currentScene.d5Comparison;
      const isWaste = focus.prefillMode === 'hf_faithful' && d5.wastedPrefixKvFraction > 0;
      const sec = el(
        'section',
        `uni-inspector-section${isWaste ? ' uni-inspector-section--amber' : ''}`,
      );
      sec.appendChild(el('h3', 'uni-section-heading', 'D5 Elastic Prefill Telemetry'));
      const grid = el('div', 'uni-kv-grid');
      addKvRow(grid, 'Mode', focus.prefillMode);
      addKvRow(
        grid,
        'U Layers Executed',
        focus.prefillMode === 'elastic_pruned'
          ? `${d5.elasticPrunedLayersExecuted} / 34`
          : `${d5.hfFaithfulLayersExecuted} / 34`,
      );
      addKvRow(grid, 'Wasted Prefix KV', `${(d5.wastedPrefixKvFraction * 100).toFixed(1)}%`);
      addKvRow(
        grid,
        'Prod Prefix KV',
        formatSiIecBytes(
          focus.prefillMode === 'elastic_pruned'
            ? d5.prodElasticPrunedPrefixKvBytes
            : d5.prodHfFaithfulPrefixKvBytes,
        ),
      );
      addKvRow(grid, 'Stage-2 Parity', `${d5.stage2UlpDiff} ULP (exact match)`);
      sec.appendChild(grid);
      container.appendChild(sec);
    }
  }

  function renderArchitectureTab(container: HTMLElement): void {
    const readouts = buildUniViewArchReadouts();
    const activeViewId = currentScene.executionViewId;

    const viewsSec = el('section', 'uni-inspector-section');
    viewsSec.appendChild(el('h3', 'uni-section-heading', '6 Execution Views (Active vs Eager HF)'));

    const table = el('table', 'uni-data-table');
    const thead = el('thead');
    const hr = el('tr');
    for (const h of ['View', 'Active', 'Eager HF', 'KV/tok', 'Mini']) {
      hr.appendChild(el('th', undefined, h));
    }
    thead.appendChild(hr);
    const tbody = el('tbody');
    for (const r of readouts) {
      const tr = el('tr', r.viewId === activeViewId ? 'is-highlight' : undefined);
      tr.append(
        el('td', undefined, r.label),
        el('td', undefined, formatCompactCount(r.prodActiveParams)),
        el('td', undefined, formatCompactCount(r.prodEagerHfParams)),
        el('td', undefined, `${(r.prodKvBytesPerTokenBf16 / 1024).toFixed(0)} KiB`),
        el('td', undefined, formatCompactCount(r.miniActiveFloats)),
      );
      tbody.appendChild(tr);
    }
    table.append(thead, tbody);
    viewsSec.appendChild(table);
    container.appendChild(viewsSec);

    // Live D5 comparison
    const sess = getActiveSession();
    const branch = sess.getActiveBranchId() ? sess.getBranch(sess.getActiveBranchId()!) : undefined;
    const prefixLen = Math.max(1, branch?.prefixCache.seqLen ?? 16);
    const d5 = buildUniElasticD5Comparison(activeViewId, prefixLen);
    const d5Sec = el('section', 'uni-inspector-section');
    d5Sec.appendChild(el('h3', 'uni-section-heading', `D5 KV Footprint (${prefixLen} prefix tok)`));
    const d5Grid = el('div', 'uni-kv-grid');
    addKvRow(d5Grid, 'HF-Faithful (Prod)', formatSiIecBytes(d5.prodHfFaithfulPrefixKvBytes));
    addKvRow(d5Grid, 'Elastic-Pruned (Prod)', formatSiIecBytes(d5.prodElasticPrunedPrefixKvBytes));
    addKvRow(
      d5Grid,
      'Saved Prefix KV',
      `${formatSiIecBytes(d5.prodSavedPrefixKvBytes)} (${(d5.wastedPrefixKvFraction * 100).toFixed(1)}%)`,
    );
    addKvRow(d5Grid, 'Mini FP32 Pruned', formatSiIecBytes(d5.miniElasticPrunedPrefixKvBytes));
    d5Sec.appendChild(d5Grid);
    container.appendChild(d5Sec);

    // 4-Tier Context Honesty Table (U3)
    const ctxSec = el('section', 'uni-inspector-section');
    ctxSec.appendChild(el('h3', 'uni-section-heading', '4-Tier Context Honesty (U3)'));
    const ctxGrid = el('div', 'uni-kv-grid');
    for (const tier of UNI_CONTEXT_TIERS) {
      addKvRow(ctxGrid, `T${tier.tier} · ${tier.tokens.toLocaleString('en-US')}`, tier.label);
    }
    ctxSec.appendChild(ctxGrid);
    container.appendChild(ctxSec);

    // Canonical Provenance
    const refSec = el('section', 'uni-inspector-section');
    refSec.appendChild(el('h3', 'uni-section-heading', 'Canonical Provenance (U1 / U2)'));
    const refGrid = el('div', 'uni-kv-grid');
    addKvRow(refGrid, 'Paper', `${UNI_CANONICAL_REFS.primaryPaper.id} (${UNI_CANONICAL_REFS.primaryPaper.license})`);
    addKvRow(refGrid, 'Checkpoint', `${UNI_CANONICAL_REFS.checkpoint.repoId} @ ${UNI_CANONICAL_REFS.checkpoint.commit.slice(0, 10)}`);
    addKvRow(refGrid, 'Repository', `${UNI_CANONICAL_REFS.githubRepo.repoId} @ ${UNI_CANONICAL_REFS.githubRepo.commit.slice(0, 10)}`);
    refSec.appendChild(refGrid);
    container.appendChild(refSec);
  }

  function renderLedgerTab(container: HTMLElement): void {
    const filterSec = el('section', 'uni-inspector-section');
    filterSec.appendChild(el('h3', 'uni-section-heading', 'Discrepancy & Guard Ledger (U1–U30)'));

    const search = el('input', 'uni-ledger-search');
    search.type = 'search';
    search.placeholder = 'Filter by ID (U6, U18, U26) or keyword…';
    search.value = ledgerQuery;
    search.dataset.testid = 'uni-ledger-search';
    search.addEventListener('input', () => {
      ledgerQuery = search.value;
      renderLedgerCards(listWrap);
    });

    const filterRow = el('div', 'uni-chip-group');
    for (const f of LEDGER_FILTERS) {
      const btn = el('button', 'uni-chip-btn', f.label);
      btn.type = 'button';
      btn.classList.toggle('is-active', ledgerFilter === f.id);
      btn.addEventListener('click', () => {
        ledgerFilter = f.id;
        renderInspectorBody();
      });
      filterRow.appendChild(btn);
    }

    filterSec.append(search, filterRow);
    container.appendChild(filterSec);

    const listWrap = el('div', 'uni-ledger-list');
    renderLedgerCards(listWrap);
    container.appendChild(listWrap);
  }

  function renderLedgerCards(listWrap: HTMLElement): void {
    listWrap.innerHTML = '';
    const q = ledgerQuery.trim().toLowerCase();
    const activeTourRefs = new Set<string>(
      tour.isActive() ? tour.getState().currentBeat.ledgerRefs : [],
    );

    for (let i = 0; i < UNI_DISCREPANCY_LEDGER.length; i++) {
      const entry = UNI_DISCREPANCY_LEDGER[i]!;
      if (ledgerFilter === 'browser_guard' && !entry.isBrowserGuard) continue;
      if (
        ledgerFilter !== 'all' &&
        ledgerFilter !== 'browser_guard' &&
        entry.category !== ledgerFilter
      ) {
        continue;
      }
      if (q.length > 0) {
        const hay =
          `${entry.id} ${entry.topic} ${entry.paperOrConfigClaim} ${entry.verifiedGroundTruth} ${entry.browserResolution}`.toLowerCase();
        if (!hay.includes(q)) continue;
      }

      const card = el(
        'article',
        `uni-ledger-card${activeTourRefs.has(entry.id) ? ' is-referenced' : ''}`,
      );
      card.dataset.testid = `uni-ledger-card-${entry.id}`;

      const head = el('div', 'uni-ledger-card-head');
      head.appendChild(el('span', 'uni-ledger-id', entry.id));
      if (entry.isBrowserGuard) {
        head.appendChild(el('span', 'uni-ledger-badge--guard', 'BROWSER GUARD'));
      }
      head.appendChild(el('span', 'uni-ledger-topic', entry.topic));

      card.append(
        head,
        el('div', 'uni-ledger-detail', `Verified: ${entry.verifiedGroundTruth}`),
        el('div', 'uni-ledger-detail', `Resolution: ${entry.browserResolution}`),
      );
      listWrap.appendChild(card);
    }
  }

  function renderInspectorBody(): void {
    inspectorBody.innerHTML = '';
    if (inspectorTab === 'inspect') {
      renderViewControlsSection(inspectorBody);
      renderCellInspectSection(inspectorBody);
      renderViewDiagnosticsSection(inspectorBody);
    } else if (inspectorTab === 'architecture') {
      renderArchitectureTab(inspectorBody);
    } else {
      renderLedgerTab(inspectorBody);
    }
  }

  // --------------------------------------------------------------------------
  // POINTER ORBIT / PAN / ZOOM & RAY-PICKING
  // --------------------------------------------------------------------------

  let dragMode: 'none' | 'orbit' | 'pan' = 'none';
  let dragStartX = 0;
  let dragStartY = 0;
  let lastPtrX = 0;
  let lastPtrY = 0;
  let pointerMoved = false;

  function pickAtClientPoint(clientX: number, clientY: number): UniPickIdentity | null {
    if (!cameraSnapshot || currentScene.pickBoxes.length === 0) return null;
    const ray = clientPointToRay(clientX, clientY, cameraSnapshot);
    const hit = pickMacroBay(ray, currentScene.pickBoxes, []);
    return pickHitToUniCell(hit, currentScene.viewMode);
  }

  const onPointerDown = (ev: PointerEvent): void => {
    if (ev.button !== 0 && ev.button !== 2) return;
    activeFly = null;
    scheduler.setContinuousReason('camera_fly', false);
    dragMode = ev.button === 2 || ev.shiftKey ? 'pan' : 'orbit';
    dragStartX = ev.clientX;
    dragStartY = ev.clientY;
    lastPtrX = ev.clientX;
    lastPtrY = ev.clientY;
    pointerMoved = false;
    if (typeof canvas.setPointerCapture === 'function') {
      try {
        canvas.setPointerCapture(ev.pointerId);
      } catch {
        // Ignore synthetic events without active pointer
      }
    }
  };

  const onPointerMove = (ev: PointerEvent): void => {
    if (dragMode !== 'none' && cameraPose && cameraSnapshot) {
      const dx = ev.clientX - lastPtrX;
      const dy = ev.clientY - lastPtrY;
      lastPtrX = ev.clientX;
      lastPtrY = ev.clientY;
      if (Math.hypot(ev.clientX - dragStartX, ev.clientY - dragStartY) > 4) {
        pointerMoved = true;
      }
      if (dragMode === 'orbit') {
        cameraPose = orbitCameraByCssDelta(cameraPose, dx, dy);
      } else {
        cameraPose = panCameraByCssDelta(cameraPose, cameraSnapshot, dx, dy);
      }
      scheduler.requestFrame();
      return;
    }

    const nextHover = pickAtClientPoint(ev.clientX, ev.clientY);
    const prevKey = hoveredCell
      ? `${hoveredCell.partId}:${hoveredCell.tensorKey}:${hoveredCell.row}:${hoveredCell.col}`
      : 'none';
    const nextKey = nextHover
      ? `${nextHover.partId}:${nextHover.tensorKey}:${nextHover.row}:${nextHover.col}`
      : 'none';
    if (prevKey !== nextKey) {
      hoveredCell = nextHover;
      if (inspectorTab === 'inspect') {
        renderInspectorBody();
      }
      scheduler.requestFrame();
    }
  };

  const onPointerUp = (ev: PointerEvent): void => {
    const wasClick = dragMode !== 'none' && !pointerMoved;
    dragMode = 'none';
    if (typeof canvas.releasePointerCapture === 'function') {
      try {
        canvas.releasePointerCapture(ev.pointerId);
      } catch {
        // Ignore
      }
    }

    if (!wasClick) return;
    const picked = pickAtClientPoint(ev.clientX, ev.clientY);
    selectedCell = picked;
    selectedPartId = picked?.partId ?? null;

    if (picked && !tour.isActive()) {
      if (viewMode === 'macro' && picked.layerIndex !== null) {
        focus = { ...focus, selectedLayerIndex: picked.layerIndex };
        rebuildSceneAndRefresh();
        return;
      }
      if (viewMode === 'joint_attn' && picked.qHeadIndex !== null) {
        focus = { ...focus, selectedQHead: picked.qHeadIndex };
        rebuildSceneAndRefresh();
        return;
      }
      if (viewMode === 'ball_tree') {
        if (picked.tensorKey === 'codebook_pca_node' && picked.tokenIndex !== null) {
          focus = { ...focus, selectedCodeId: picked.tokenIndex };
          rebuildSceneAndRefresh();
          return;
        }
        if (picked.tensorKey === 'ball_membership' && picked.row !== null && picked.row <= 8) {
          if (inspectorTab === 'inspect') renderInspectorBody();
          scheduler.requestFrame();
          return;
        }
      }
      if (viewMode === 'refinement' && picked.tokenIndex !== null && picked.tokenIndex <= 23) {
        focus = { ...focus, selectedGridCell: picked.tokenIndex };
        rebuildSceneAndRefresh();
        return;
      }
    }

    if (inspectorTab === 'inspect') {
      renderInspectorBody();
    }
    scheduler.requestFrame();
  };

  const onPointerLeave = (): void => {
    if (dragMode !== 'none') return;
    if (hoveredCell !== null) {
      hoveredCell = null;
      if (inspectorTab === 'inspect') {
        renderInspectorBody();
      }
      scheduler.requestFrame();
    }
  };

  const onWheel = (ev: WheelEvent): void => {
    if (!cameraPose) return;
    ev.preventDefault();
    activeFly = null;
    scheduler.setContinuousReason('camera_fly', false);
    const scaleFactor = ev.deltaY > 0 ? 1.1 : 0.91;
    cameraPose = zoomPerspectiveDistance(cameraPose, scaleFactor);
    scheduler.requestFrame();
  };

  const onContextMenu = (ev: MouseEvent): void => {
    ev.preventDefault();
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', onContextMenu);

  // --------------------------------------------------------------------------
  // GLOBAL KEYBOARD SHORTCUTS & RESIZE OBSERVER
  // --------------------------------------------------------------------------

  const onWindowKeydown = (ev: KeyboardEvent): void => {
    if (tour.isActive() && tourFocusTrap.handleKeydown(ev)) {
      return;
    }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (shouldDeferGlobalShortcut(ev.target as HTMLElement | null)) return;

    if (tour.isActive()) {
      if (ev.key === 'ArrowRight') {
        ev.preventDefault();
        const st = tour.getState();
        if (st.beatIndex < st.beatCount - 1) {
          goToTourBeatInternal(st.beatIndex + 1);
        }
        return;
      }
      if (ev.key === 'ArrowLeft') {
        ev.preventDefault();
        const st = tour.getState();
        if (st.beatIndex > 0) {
          goToTourBeatInternal(st.beatIndex - 1);
        }
        return;
      }
      if (ev.key === 'Escape') {
        ev.preventDefault();
        exitTourInternal(true);
        return;
      }
    }

    if (ev.key >= '1' && ev.key <= '7') {
      const idx = Number.parseInt(ev.key, 10) - 1;
      const targetView = UNI_SPATIAL_VIEW_MODES[idx];
      if (targetView) {
        ev.preventDefault();
        if (tour.isActive()) exitTourInternal(false);
        setViewInternal(targetView.id, true);
      }
      return;
    }

    if (ev.key === 'r' || ev.key === 'R') {
      ev.preventDefault();
      flyToCurrentScenePreset(true);
    } else if (ev.key === 't' || ev.key === 'T') {
      ev.preventDefault();
      if (tour.isActive()) exitTourInternal(true);
      else startTourInternal(0);
    } else if (ev.key === 'i' || ev.key === 'I') {
      ev.preventDefault();
      toggleInspectorCollapsed();
    }
  };

  if (typeof window !== 'undefined') {
    window.addEventListener('keydown', onWindowKeydown);
  }

  let resizeObserver: ResizeObserver | null = null;
  if (typeof ResizeObserver === 'function') {
    resizeObserver = new ResizeObserver(() => {
      scheduler.requestFrame();
    });
    resizeObserver.observe(viewportHost);
  }

  // --------------------------------------------------------------------------
  // DEBUG SNAPSHOT & `window.__uniSpatialDebug` E2E HANDLE
  // --------------------------------------------------------------------------

  function getDebugSnapshot(): UniSpatialDebugSnapshot {
    const sess = getActiveSession();
    const audit = sess.auditMemory();
    const tourState = tour.getState();
    return Object.freeze({
      viewMode: currentScene.viewMode,
      executionViewId: currentScene.executionViewId,
      taskMode: getActiveTaskMode(),
      prefillMode: focus.prefillMode,
      rafActive: scheduler.isActive(),
      glErrorCount,
      atlasKey: currentScene.atlasKey,
      atlasUploadCount: atlasTracker.getUploadCount(),
      rgbPreviewCount: currentScene.rgbPreviews.length,
      overlayCellNumNodes: overlay.getActiveCellNumNodeCount(),
      activeRecordId: sess.getActiveRecordId(),
      branchCount: audit.branchCount,
      tourActive: tourState.active,
      tourBeatIndex: tourState.beatIndex,
      tourBeatCount: tourState.beatCount,
      memoryAudit: audit,
    });
  }

  const debugHandle: UniSpatialDebugHandle = {
    get viewMode() {
      return getDebugSnapshot().viewMode;
    },
    get executionViewId() {
      return getDebugSnapshot().executionViewId;
    },
    get taskMode() {
      return getDebugSnapshot().taskMode;
    },
    get prefillMode() {
      return getDebugSnapshot().prefillMode;
    },
    get rafActive() {
      return getDebugSnapshot().rafActive;
    },
    get glErrorCount() {
      return getDebugSnapshot().glErrorCount;
    },
    get atlasKey() {
      return getDebugSnapshot().atlasKey;
    },
    get atlasUploadCount() {
      return getDebugSnapshot().atlasUploadCount;
    },
    get rgbPreviewCount() {
      return getDebugSnapshot().rgbPreviewCount;
    },
    get overlayCellNumNodes() {
      return getDebugSnapshot().overlayCellNumNodes;
    },
    get activeRecordId() {
      return getDebugSnapshot().activeRecordId;
    },
    get branchCount() {
      return getDebugSnapshot().branchCount;
    },
    get tourActive() {
      return getDebugSnapshot().tourActive;
    },
    get tourBeatIndex() {
      return getDebugSnapshot().tourBeatIndex;
    },
    get tourBeatCount() {
      return getDebugSnapshot().tourBeatCount;
    },
    get memoryAudit() {
      return getDebugSnapshot().memoryAudit;
    },
    getSnapshot(): UniSpatialDebugSnapshot {
      return getDebugSnapshot();
    },
    setView(nextView: UniSpatialViewMode): UniSpatialDebugSnapshot {
      if (tour.isActive()) {
        exitTourInternal(false, true);
      }
      setViewInternal(nextView, false);
      return getDebugSnapshot();
    },
    setTaskMode(taskMode: UniSessionTaskMode): UniSpatialDebugSnapshot {
      setTaskModeInternal(taskMode, true);
      return getDebugSnapshot();
    },
    setPrefillMode(mode: UniElasticPrefillMode): UniSpatialDebugSnapshot {
      setPrefillModeInternal(mode, true);
      return getDebugSnapshot();
    },
    startTour(beatIndex = 0): UniSpatialDebugSnapshot {
      startTourInternal(beatIndex, true);
      return getDebugSnapshot();
    },
    goToBeat(beatIndex: number): UniSpatialDebugSnapshot {
      goToTourBeatInternal(beatIndex, true);
      return getDebugSnapshot();
    },
    exitTour(): UniSpatialDebugSnapshot {
      exitTourInternal(true, true);
      return getDebugSnapshot();
    },
  };

  if (typeof window !== 'undefined') {
    (window as unknown as { __uniSpatialDebug?: UniSpatialDebugHandle }).__uniSpatialDebug =
      debugHandle;
  }

  // Initial synchronous layout, upload, draw & idle settle (`rafActive === false` at rest!)
  syncTopViewTabs();
  syncInspectorTabs();
  refreshStageHud();
  refreshRgbInset();
  refreshDock();
  renderInspectorBody();
  cameraPose = computePresetPose(currentScene);
  renderFrameNow();

  // --------------------------------------------------------------------------
  // CLEAN DISPOSAL
  // --------------------------------------------------------------------------

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    scheduler.dispose();
    tourFocusTrap.deactivate();
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    if (typeof window !== 'undefined') {
      window.removeEventListener('keydown', onWindowKeydown);
      const win = window as unknown as { __uniSpatialDebug?: UniSpatialDebugHandle };
      if (win.__uniSpatialDebug === debugHandle) {
        delete win.__uniSpatialDebug;
      }
    }
    canvas.removeEventListener('webglcontextlost', onContextLost);
    canvas.removeEventListener('webglcontextrestored', onContextRestored);
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointerleave', onPointerLeave);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('contextmenu', onContextMenu);

    overlay.dispose();
    pixelPanel.dispose();
    timelineDock.dispose();

    if (glBundle) {
      glBundle.solids.dispose();
      glBundle.lines.dispose();
      glBundle.ribbons.dispose();
      glBundle.flatFlow.dispose();
      glBundle.sheets.dispose();
      glBundle.device.dispose();
      glBundle = null;
    }

    if (shell.parentNode === root) {
      root.removeChild(shell);
    }
  }

  return {
    getDebugSnapshot,
    getDebugHandle: () => debugHandle,
    dispose,
  };
}
