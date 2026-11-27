/**
 * Nemotron-Labs-Diffusion-14B Editorial WebGL2 Spatial Application
 * (`src/diffusion/app/spatialApp.ts`)
 *
 * Composes:
 *   - Shared 5-pass WebGL2 pipeline (`RenderDevice`, `SolidsPass`, `LinesPass`,
 *     `RibbonsPass`, `FlatFlowRibbonsPass`, `SheetsPass`) unchanged
 *   - Strict idle-RAF gating (ZERO continuous RAF spin when idle)
 *   - Content-addressed RG32F value-atlas deduplication (`scene.atlasKey`)
 *   - WebGL context loss/restore recovery + `gl.getError()` telemetry
 *   - All 7 DF3 spatial views routed via `buildDiffusionSpatialScene`
 *   - Live `DiffusionSession` mode/execution controls with graceful `KvProvenanceMismatchError`
 *     re-prefill confirmation flow
 *   - Interactive 2D Unit-Circle RoPE Phasor SVG inset for `gqa_rope` with dyadic YaRN
 *     plane selectors and `p = 16` Llama-4 query temperature scale boundary scrubber
 *   - Pooled DOM + SVG 3D-to-2D projected overlay (`DiffusionSpatialOverlay`)
 *   - DF4 Isolated 4-Act Guided Walkthrough HUD (`createDiffusionGuidedTour`)
 *   - DF4 Unified Tri-Mode Event Timeline Dock (`createDiffusionTimelineDock`)
 *   - DF4 Collapsible 3-Tab Editorial Inspector Drawer (`Inspect` / `Architecture` / `Ledger D1–D24`)
 *   - WAI-ARIA roving tabindex (`handleTablistKeydown`), global shortcuts, polite live region,
 *     and dynamic `prefers-reduced-motion` compliance
 *
 * Pure deterministic implementation with zero sibling-visualizer dependencies.
 */

import {
  buildCameraSnapshot,
  CAMERA_FLY_DURATION_MS,
  DEFAULT_FOV_DEG,
  fitPerspectiveToBounds,
  interpolateCameraPose,
  makeViewportMetrics,
  orbitCameraByCssDelta,
  panCameraByCssDelta,
  zoomPerspectiveDistance,
  type CameraPose,
  type CameraSnapshot,
  type ViewportMetrics,
} from '../../render/camera';
import { RenderDevice } from '../../render/device';
import { LinesPass } from '../../render/lines';
import { clientPointToRay, pickMacroBay } from '../../render/pick';
import { FlatFlowRibbonsPass, RibbonsPass } from '../../render/ribbons';
import { SheetsPass } from '../../render/sheets';
import { SolidsPass } from '../../render/solids';
import { handleTablistKeydown } from '../../shared/a11y';
import { mountModelSwitcher } from '../../shared/modelSwitcher';
import type { OverlayRect } from '../../shared/overlayLayout';
import {
  buildDiffusionArchReadouts,
  buildDiffusionNfeTpfReadout,
  filterDiffusionDiscrepancyLedger,
  formatDualSiIecBytes,
} from '../archReadouts';
import { KvProvenanceMismatchError } from '../kvCache';
import { createDefaultDiffusionModel } from '../parameters';
import { computeLlama4AttnScale } from '../rope';
import { buildDiffusionSpatialScene } from '../scene/index';
import {
  DIFFUSION_EPISTEMIC_NOTE,
  DIFFUSION_PROJECTION_REVERSAL_BADGE,
  ensureSessionHasActiveRecord,
  pickHitToDiffusionCell,
  resolveDiffusionSessionContext,
} from '../scene/provenance';
import {
  DEFAULT_DIFFUSION_FOCUS_STATE,
  DIFFUSION_SPATIAL_VIEW_MODES,
  type DiffusionFocusState,
  type DiffusionPickIdentity,
  type DiffusionResolvedSessionContext,
  type DiffusionRopePhasorTelemetry,
  type DiffusionSpatialScene,
  type DiffusionSpatialViewMode,
} from '../scene/types';
import { createDiffusionSession, type DiffusionSession } from '../session';
import type {
  DiffusionDiscrepancyCategory,
  DiffusionFreeExploreRestoreState,
  DiffusionInspectorTabId,
  DiffusionTourActId,
  DiffusionTourBeatSnapshot,
} from '../story/types';
import {
  createDiffusionGuidedTour,
  DIFFUSION_TOUR_ACT_DEFS,
} from '../story/walkthrough';
import {
  DIFFUSION_DYADIC_YARN_INV_FREQ,
  DIFFUSION_MINI_CONFIG,
  DIFFUSION_MINI_VOCAB,
  type DiffusionExecutionMode,
  type DiffusionExplorerMaskMode,
  type DiffusionSessionConfigPatch,
  type DiffusionSessionMemoryReport,
  type YarnPlaneZone,
} from '../types';
import { DiffusionSpatialOverlay } from './spatialOverlay';
import {
  buildDiffusionTimelineDockState,
  createDiffusionTimelineDock,
  isReducedMotionPreferred,
  resolveDiffusionGlobalKeyAction,
} from './timelineDock';

// ============================================================================
// PUBLIC DEBUG & HANDLE CONTRACTS
// ============================================================================

export interface DiffusionSpatialDebugHandle {
  readonly viewMode: DiffusionSpatialViewMode;
  readonly maskTab: DiffusionExplorerMaskMode;
  readonly layerIndex: number;
  readonly glErrorCount: number;
  readonly lastGlError: number;
  readonly frameCount: number;
  readonly atlasUploadCount: number;
  readonly atlasWidth: number;
  readonly atlasHeight: number;
  readonly atlasKey: string | null;
  readonly recordCount: number;
  readonly kvSeqLen: number;
  readonly mode: DiffusionExecutionMode;
  readonly loraActiveOnDraft: boolean;
  readonly rafActive: boolean;
  readonly tourActive: boolean;
  readonly tourBeatIndex: number | null;
  readonly tourTotalBeats: number;
  readonly inspectorTab: DiffusionInspectorTabId;
  readonly inspectorOpen: boolean;
  readonly branchCount: number;
  readonly activeBranchId: string;
  readonly activeRecordId: string | null;
  readonly retainedBytes: number;
}

export interface DiffusionSpatialAppHandle {
  setViewMode(mode: DiffusionSpatialViewMode): void;
  getDebugSnapshot(): DiffusionSpatialDebugHandle;
  dispose(): void;
}

interface DiffusionWindowHost {
  __diffusionSpatialDebug?: DiffusionSpatialDebugHandle;
}

// ============================================================================
// UI CONFIGURATION TABLES
// ============================================================================

interface ModePreset {
  readonly id: 'ar' | 'dlm_causal' | 'dlm_bidir' | 'linear_spec' | 'linear_spec_lora';
  readonly label: string;
  readonly title: string;
  readonly patch: DiffusionSessionConfigPatch;
}

const MODE_PRESETS: readonly ModePreset[] = Object.freeze([
  {
    id: 'ar',
    label: 'AR',
    title: 'Standard token-causal Autoregressive decoding (1 token/step, LoRA OFF)',
    patch: { mode: 'ar', causalContext: true },
  },
  {
    id: 'dlm_causal',
    label: 'DLM Causal',
    title: 'Block Diffusion (B=4) with causal KV context (token_causal provenance)',
    patch: { mode: 'dlm', causalContext: true },
  },
  {
    id: 'dlm_bidir',
    label: 'DLM Bidir',
    title: 'Block Diffusion (B=4) with bidirectional KV context (block_bidirectional provenance)',
    patch: { mode: 'dlm', causalContext: false },
  },
  {
    id: 'linear_spec',
    label: 'Linear Spec',
    title: 'Linear Speculative Drafting + Causal Verify + KV rollback (LoRA OFF)',
    patch: { mode: 'linear_spec', causalContext: true },
  },
  {
    id: 'linear_spec_lora',
    label: 'Spec + LoRA',
    title: 'Linear Speculative Drafting with draft-only o_proj LoRA ON; Verify LoRA OFF',
    patch: { mode: 'linear_spec_lora', causalContext: true },
  },
]);

interface MaskTabDef {
  readonly id: DiffusionExplorerMaskMode;
  readonly label: string;
  readonly title: string;
}

const MASK_TABS: readonly MaskTabDef[] = Object.freeze([
  Object.freeze({
    id: 'joint_training',
    label: 'Joint [x_t‖x_0]',
    title: 'BD3-LM Joint Training Block Mask (M_BD, M_OBC, M_BC)',
  }),
  Object.freeze({
    id: 'ar_causal',
    label: 'AR Causal',
    title: 'Lower-triangular token-causal autoregressive mask',
  }),
  Object.freeze({
    id: 'block_bidir',
    label: 'Block Bidir',
    title: 'Block-diagonal bidirectional diffusion mask',
  }),
  Object.freeze({
    id: 'causal_commit_verify',
    label: 'Commit / Verify',
    title: 'Committed prefix + shifted causal verify attention mask',
  }),
  Object.freeze({
    id: 'vlm_quadratic_ss',
    label: 'VLM Quadratic SS',
    title:
      'Read-only VLM-8B cached Quadratic Self-Speculation reference mask (k(k+1) interleaved queries, [draft_kv || clean_kv])',
  }),
]);

interface InspectorTabDef {
  readonly id: DiffusionInspectorTabId;
  readonly label: string;
  readonly title: string;
}

const INSPECTOR_TABS: readonly InspectorTabDef[] = Object.freeze([
  Object.freeze({
    id: 'inspect',
    label: 'Inspect',
    title: 'Live record provenance, 3-counter NFE/TPF, active formula, D19 crop & token stream',
  }),
  Object.freeze({
    id: 'architecture',
    label: 'Architecture',
    title: 'Verified 0 B LFS 14B specs, 7-variant family, 4-tier KV scaling & 4-way memory boundaries',
  }),
  Object.freeze({
    id: 'ledger',
    label: 'Ledger (D1–D24)',
    title: 'Searchable & filterable 24-entry verified discrepancy ledger (D1–D24)',
  }),
]);

const LEDGER_CATEGORIES: readonly {
  readonly id: DiffusionDiscrepancyCategory;
  readonly label: string;
}[] = Object.freeze([
  Object.freeze({ id: 'all', label: 'All (24)' }),
  Object.freeze({ id: 'architecture_specs', label: 'Architecture' }),
  Object.freeze({ id: 'rope_masks', label: 'RoPE & Masks' }),
  Object.freeze({ id: 'execution_kv', label: 'Exec & KV' }),
  Object.freeze({ id: 'samplers_guards', label: 'Samplers & Guards' }),
]);

const ROTARY_PLANE_LABELS: readonly string[] = Object.freeze(['(0,4)', '(1,5)', '(2,6)', '(3,7)']);
const DEFAULT_PLANE_ZONES: readonly YarnPlaneZone[] = Object.freeze([
  'original',
  'blended',
  'interpolated',
  'interpolated',
]);

const SVG_NS = 'http://www.w3.org/2000/svg';
const DRAG_THRESHOLD_PX = 5;

// ============================================================================
// DOM & MATH HELPERS
// ============================================================================

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  textContent?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (textContent !== undefined) node.textContent = textContent;
  return node;
}

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs?: Record<string, string | number>,
  textContent?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      node.setAttribute(k, String(v));
    }
  }
  if (textContent !== undefined) node.textContent = textContent;
  return node;
}

function samePickIdentity(
  a: DiffusionPickIdentity | null,
  b: DiffusionPickIdentity | null,
): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.viewMode === b.viewMode &&
    a.layerIndex === b.layerIndex &&
    a.partId === b.partId &&
    a.tensorKey === b.tensorKey &&
    a.qHeadIndex === b.qHeadIndex &&
    a.kvHeadIndex === b.kvHeadIndex &&
    a.rotaryPlane === b.rotaryPlane &&
    a.tokenIndex === b.tokenIndex &&
    a.row === b.row &&
    a.col === b.col
  );
}

function cloneFocusState(f: DiffusionFocusState): DiffusionFocusState {
  return {
    ...f,
    hoveredCell: f.hoveredCell ? { ...f.hoveredCell } : null,
    pinnedCell: f.pinnedCell ? { ...f.pinnedCell } : null,
  };
}

function cloneCameraPose(pose: CameraPose): CameraPose {
  return {
    target: [pose.target[0], pose.target[1], pose.target[2]],
    azimuthDeg: pose.azimuthDeg,
    elevationDeg: pose.elevationDeg,
    distance: pose.distance,
    fovDeg: pose.fovDeg,
  };
}

function formatLayerCode(layerIdx: number): string {
  return `L${String(layerIdx + 1).padStart(2, '0')}`;
}

function formatQHeadCode(qHead: number): string {
  return `Q${String(qHead).padStart(2, '0')}`;
}

function activeModePresetId(
  mode: DiffusionExecutionMode,
  causalContext: boolean,
): ModePreset['id'] {
  if (mode === 'ar') return 'ar';
  if (mode === 'dlm') return causalContext ? 'dlm_causal' : 'dlm_bidir';
  if (mode === 'linear_spec') return 'linear_spec';
  return 'linear_spec_lora';
}

function normalizePair(pair: readonly [number, number]): readonly [number, number] {
  const len = Math.hypot(pair[0], pair[1]);
  if (len < 1e-6) return [1, 0];
  return [pair[0] / len, pair[1] / len];
}

function rotatePair(pair: readonly [number, number], angleRad: number): readonly [number, number] {
  const c = Math.cos(angleRad);
  const s = Math.sin(angleRad);
  return [pair[0] * c - pair[1] * s, pair[1] * c + pair[0] * s];
}

// ============================================================================
// MAIN APPLICATION MOUNT
// ============================================================================

export function mountDiffusionSpatialApp(root: HTMLElement): DiffusionSpatialAppHandle {
  root.innerHTML = '';

  const model = createDefaultDiffusionModel();
  const session = createDiffusionSession(model);
  ensureSessionHasActiveRecord(session);

  // Isolated 4-Act Guided Tour (owns its own private DiffusionSession)
  const tour = createDiffusionGuidedTour(model);

  let focus: DiffusionFocusState = { ...DEFAULT_DIFFUSION_FOCUS_STATE };
  let resolvedCtx: DiffusionResolvedSessionContext = resolveDiffusionSessionContext(session);
  let scene: DiffusionSpatialScene = buildDiffusionSpatialScene(resolvedCtx, focus);

  // Scrubber override for 2D RoPE phasor position (`null` = follow live record queryPosition)
  let ropeScrubPosition: number | null = null;

  // Collapsible 3-tab Editorial Inspector state
  let inspectorOpen = false;
  let inspectorTab: DiffusionInspectorTabId = 'inspect';
  let ledgerQuery = '';
  let ledgerCategory: DiffusionDiscrepancyCategory = 'all';

  // Cached memory report for currently displayed session (refreshed on session/tour updates)
  let cachedMemoryReport: DiffusionSessionMemoryReport = session.measureMemory();

  // Pending mode switch awaiting explicit re-prefill confirmation after KvProvenanceMismatchError
  let pendingRePrefill: {
    readonly preset: ModePreset;
    readonly errorMessage: string;
  } | null = null;

  // Render & telemetry state
  let glErrorCount = 0;
  let lastGlError = 0;
  let frameCount = 0;
  let atlasUploadCount = 0;
  let lastUploadedAtlasKey: string | null = null;
  let atlasWidth = 0;
  let atlasHeight = 0;

  let rafId: number | null = null;
  let rafActive = false;
  let dirty = true;
  let sceneUploadPending = true;
  let disposed = false;

  // Camera state
  let cameraPose: CameraPose = {
    target: [0, 0, 0],
    azimuthDeg: scene.defaultCameraAzimuthDeg,
    elevationDeg: scene.defaultCameraElevationDeg,
    distance: 28,
    fovDeg: DEFAULT_FOV_DEG,
  };
  let cameraInitialized = false;
  let cameraAnim: {
    readonly from: CameraPose;
    readonly to: CameraPose;
    readonly startMs: number;
    readonly durationMs: number;
  } | null = null;
  let lastSnapshot: CameraSnapshot | null = null;

  // Pointer interaction state
  let pointerDown = false;
  let pointerMoved = false;
  let pointerMode: 'orbit' | 'pan' = 'orbit';
  let pointerStartX = 0;
  let pointerStartY = 0;
  let pointerLastX = 0;
  let pointerLastY = 0;

  function getDisplayedSession(): DiffusionSession {
    return tour.isActive() ? tour.getTourSession() : session;
  }

  function refreshMemoryReport(): DiffusionSessionMemoryReport {
    cachedMemoryReport = getDisplayedSession().measureMemory();
    return cachedMemoryReport;
  }

  // ==========================================================================
  // BUILD EDITORIAL DRAFTING-TABLE SHELL DOM
  // ==========================================================================

  const shell = el('div', 'df-shell');
  root.appendChild(shell);

  // Visually hidden polite ARIA live region for screen-reader announcements
  const liveRegion = el('div', 'df-sr-only');
  liveRegion.dataset.testid = 'diffusion-live-region';
  liveRegion.setAttribute('role', 'status');
  liveRegion.setAttribute('aria-live', 'polite');
  liveRegion.setAttribute('aria-atomic', 'true');
  shell.appendChild(liveRegion);

  function announce(message: string): void {
    liveRegion.textContent = '';
    liveRegion.textContent = message;
  }

  // 1. Top Navigation & View Bar
  const topbar = el('header', 'df-topbar');
  const brand = el('div', 'df-brand');
  const titleEl = el('h1', 'df-title', 'Nemotron-Labs-Diffusion-14B');
  const epistemicPill = el('span', 'df-epistemic-pill', DIFFUSION_EPISTEMIC_NOTE);
  epistemicPill.dataset.testid = 'diffusion-epistemic-note';
  epistemicPill.title = DIFFUSION_EPISTEMIC_NOTE;
  brand.append(titleEl, epistemicPill);

  const viewNav = el('nav', 'df-view-tabs');
  viewNav.setAttribute('role', 'tablist');
  viewNav.setAttribute('aria-label', 'Spatial views');
  const viewButtons = new Map<DiffusionSpatialViewMode, HTMLButtonElement>();
  const viewButtonList: HTMLButtonElement[] = [];
  for (const desc of DIFFUSION_SPATIAL_VIEW_MODES) {
    const btn = el('button', 'df-view-tab', desc.shortLabel);
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.title = `${desc.label} — ${desc.description}`;
    btn.dataset.testid = `diffusion-view-${desc.id}`;
    btn.addEventListener('click', () => setViewMode(desc.id));
    viewNav.appendChild(btn);
    viewButtons.set(desc.id, btn);
    viewButtonList.push(btn);
  }
  viewNav.addEventListener('keydown', (ev) => {
    handleTablistKeydown(viewButtonList, ev, (_tab, idx) => {
      const desc = DIFFUSION_SPATIAL_VIEW_MODES[idx];
      if (desc) setViewMode(desc.id);
    });
  });

  const topActions = el('div', 'df-top-actions');

  const tourToggleBtn = el('button', 'df-top-btn df-top-btn--tour', 'Guided Tour');
  tourToggleBtn.type = 'button';
  tourToggleBtn.dataset.testid = 'diffusion-tour-toggle';
  tourToggleBtn.setAttribute('aria-pressed', 'false');
  tourToggleBtn.title = 'Enter or exit the isolated 4-Act Guided Walkthrough (Free-Explore preserved)';
  tourToggleBtn.addEventListener('click', () => {
    if (tour.isActive()) {
      exitGuidedTour();
    } else {
      enterGuidedTour();
    }
  });

  const inspectorToggleBtn = el('button', 'df-top-btn df-top-btn--inspector', 'Inspector');
  inspectorToggleBtn.type = 'button';
  inspectorToggleBtn.dataset.testid = 'diffusion-inspector-toggle';
  inspectorToggleBtn.setAttribute('aria-expanded', 'false');
  inspectorToggleBtn.setAttribute('aria-controls', 'diffusion-inspector-drawer');
  inspectorToggleBtn.title = 'Toggle the 3-Tab Editorial Inspector (Inspect, Architecture, Ledger D1–D24)';
  inspectorToggleBtn.addEventListener('click', () => {
    setInspectorOpen(!inspectorOpen);
  });

  const resetCamBtn = el('button', 'df-top-btn', 'Fit View');
  resetCamBtn.type = 'button';
  resetCamBtn.title = 'Re-fit perspective camera to the active view bounds';
  resetCamBtn.dataset.testid = 'diffusion-camera-reset-btn';
  resetCamBtn.addEventListener('click', () => fitCameraToScene(true));

  const baseUrl = (import.meta.env.BASE_URL || '../').replace(/\/?$/, '/');

  topActions.append(
    tourToggleBtn,
    inspectorToggleBtn,
    resetCamBtn,
  );
  const modelSwitcher = mountModelSwitcher(topActions, {
    activeId: 'diffusion',
    baseUrl,
    theme: 'dark',
  });
  topbar.append(brand, viewNav, topActions);
  shell.appendChild(topbar);

  // 2. Session Control & Context-Sensitive View Controls Strip
  const controlStrip = el('section', 'df-control-strip');
  controlStrip.setAttribute('aria-label', 'Diffusion session and view controls');

  // 2a. Mode switcher
  const modeGroup = el('div', 'df-strip-group');
  modeGroup.appendChild(el('span', 'df-strip-label', 'Mode'));
  const modeSeg = el('div', 'df-segmented');
  const modeButtons = new Map<ModePreset['id'], HTMLButtonElement>();
  for (const preset of MODE_PRESETS) {
    const btn = el('button', 'df-seg-btn', preset.label);
    btn.type = 'button';
    btn.title = preset.title;
    btn.dataset.testid = `diffusion-mode-${preset.id}`;
    btn.addEventListener('click', () => handleModeSwitch(preset));
    modeSeg.appendChild(btn);
    modeButtons.set(preset.id, btn);
  }
  modeGroup.appendChild(modeSeg);

  // 2b. Execution controls
  const execGroup = el('div', 'df-strip-group');
  const stepBackBtn = el('button', 'df-btn', '◀ Back');
  stepBackBtn.type = 'button';
  stepBackBtn.title = 'Step backward to previous session record';
  stepBackBtn.dataset.testid = 'diffusion-step-back-btn';
  stepBackBtn.addEventListener('click', () => {
    if (tour.isActive()) return;
    const prev = session.stepBack();
    if (prev) {
      pendingRePrefill = null;
      ropeScrubPosition = null;
      rebuildFromSession(false);
      announce(`Stepped back to record #${prev.recordIndex}`);
    }
  });

  const stepFwdBtn = el('button', 'df-btn', 'Fwd ▶');
  stepFwdBtn.type = 'button';
  stepFwdBtn.title = 'Step forward along active branch';
  stepFwdBtn.dataset.testid = 'diffusion-step-forward-btn';
  stepFwdBtn.addEventListener('click', () => {
    if (tour.isActive()) return;
    const next = session.stepForward();
    if (next) {
      pendingRePrefill = null;
      ropeScrubPosition = null;
      rebuildFromSession(false);
      announce(`Stepped forward to record #${next.recordIndex}`);
    }
  });

  const stepBtn = el('button', 'df-btn df-btn--primary', 'Step');
  stepBtn.type = 'button';
  stepBtn.title = 'Execute one generation step / block / speculative cycle';
  stepBtn.dataset.testid = 'diffusion-step-btn';
  stepBtn.addEventListener('click', () => {
    if (tour.isActive()) return;
    pendingRePrefill = null;
    ropeScrubPosition = null;
    const rec = session.step();
    rebuildFromSession(false);
    announce(`Executed ${rec.kind}, KV length ${rec.kvSnapshot.seqLen}`);
  });

  const runBtn = el('button', 'df-btn df-btn--accent', 'Run');
  runBtn.type = 'button';
  runBtn.title = 'Run session deterministically to completion / EOS / capacity';
  runBtn.dataset.testid = 'diffusion-run-btn';
  runBtn.addEventListener('click', () => {
    if (tour.isActive()) return;
    pendingRePrefill = null;
    ropeScrubPosition = null;
    session.runToCompletion();
    rebuildFromSession(false);
    announce(`Run complete: ${resolvedCtx.record.lifecyclePhase}, KV length ${resolvedCtx.kvSnapshot.seqLen}`);
  });

  const resetBtn = el('button', 'df-btn', 'Reset');
  resetBtn.type = 'button';
  resetBtn.title = 'Reset session back to initial prompt prefill';
  resetBtn.dataset.testid = 'diffusion-reset-btn';
  resetBtn.addEventListener('click', () => {
    if (tour.isActive()) return;
    pendingRePrefill = null;
    ropeScrubPosition = null;
    session.reset();
    ensureSessionHasActiveRecord(session);
    rebuildFromSession(false);
    announce('Session reset to initial prompt prefill');
  });

  execGroup.append(stepBackBtn, stepFwdBtn, stepBtn, runBtn, resetBtn);

  // 2c. Context-sensitive view controls
  const contextGroup = el('div', 'df-strip-group df-context-controls');

  // Layer selector (L01..L40)
  const layerWrap = el('label', 'df-field');
  layerWrap.appendChild(el('span', 'df-field-label', 'Layer'));
  const layerSelect = el('select', 'df-select');
  layerSelect.dataset.testid = 'diffusion-layer-select';
  layerSelect.setAttribute('aria-label', 'Select transformer layer L01 to L40');
  for (let i = 0; i < DIFFUSION_MINI_CONFIG.numLayers; i++) {
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = formatLayerCode(i);
    layerSelect.appendChild(opt);
  }
  layerSelect.addEventListener('change', () => {
    const idx = Number.parseInt(layerSelect.value, 10);
    if (Number.isInteger(idx) && idx >= 0 && idx < DIFFUSION_MINI_CONFIG.numLayers) {
      focus = { ...focus, selectedLayerIndex: idx, pinnedCell: null };
      rebuildSceneOnly(false);
    }
  });
  layerWrap.appendChild(layerSelect);

  // Quick inspect-layer button for macro view
  const inspectLayerBtn = el('button', 'df-btn df-btn--subtle', 'Inspect RoPE GQA →');
  inspectLayerBtn.type = 'button';
  inspectLayerBtn.dataset.testid = 'diffusion-inspect-layer-btn';
  inspectLayerBtn.addEventListener('click', () => setViewMode('gqa_rope'));

  // Q Head selector (Q00..Q31)
  const qHeadWrap = el('label', 'df-field');
  qHeadWrap.appendChild(el('span', 'df-field-label', 'Q Head'));
  const qHeadSelect = el('select', 'df-select');
  qHeadSelect.dataset.testid = 'diffusion-qhead-select';
  qHeadSelect.setAttribute('aria-label', 'Select query head Q00 to Q31');
  for (let q = 0; q < DIFFUSION_MINI_CONFIG.numAttentionHeads; q++) {
    const kv = Math.floor(q / DIFFUSION_MINI_CONFIG.kvGroupSize);
    const opt = document.createElement('option');
    opt.value = String(q);
    opt.textContent = `${formatQHeadCode(q)} (KV${kv})`;
    qHeadSelect.appendChild(opt);
  }
  qHeadSelect.addEventListener('change', () => {
    const q = Number.parseInt(qHeadSelect.value, 10);
    if (Number.isInteger(q) && q >= 0 && q < DIFFUSION_MINI_CONFIG.numAttentionHeads) {
      const kv = Math.floor(q / DIFFUSION_MINI_CONFIG.kvGroupSize);
      focus = { ...focus, selectedQHead: q, selectedKvHead: kv, pinnedCell: null };
      rebuildSceneOnly(false);
    }
  });
  qHeadWrap.appendChild(qHeadSelect);

  // KV Head selector (KV0..KV7)
  const kvHeadWrap = el('label', 'df-field');
  kvHeadWrap.appendChild(el('span', 'df-field-label', 'KV Group'));
  const kvHeadSelect = el('select', 'df-select');
  kvHeadSelect.dataset.testid = 'diffusion-kvhead-select';
  kvHeadSelect.setAttribute('aria-label', 'Select KV head KV0 to KV7');
  for (let kv = 0; kv < DIFFUSION_MINI_CONFIG.numKeyValueHeads; kv++) {
    const qStart = kv * DIFFUSION_MINI_CONFIG.kvGroupSize;
    const qEnd = qStart + DIFFUSION_MINI_CONFIG.kvGroupSize - 1;
    const opt = document.createElement('option');
    opt.value = String(kv);
    opt.textContent = `KV${kv} (Q${String(qStart).padStart(2, '0')}–Q${String(qEnd).padStart(2, '0')})`;
    kvHeadSelect.appendChild(opt);
  }
  kvHeadSelect.addEventListener('change', () => {
    const kv = Number.parseInt(kvHeadSelect.value, 10);
    if (Number.isInteger(kv) && kv >= 0 && kv < DIFFUSION_MINI_CONFIG.numKeyValueHeads) {
      const currentKv = Math.floor(focus.selectedQHead / DIFFUSION_MINI_CONFIG.kvGroupSize);
      const nextQ = currentKv === kv ? focus.selectedQHead : kv * DIFFUSION_MINI_CONFIG.kvGroupSize;
      focus = { ...focus, selectedKvHead: kv, selectedQHead: nextQ, pinnedCell: null };
      rebuildSceneOnly(false);
    }
  });
  kvHeadWrap.appendChild(kvHeadSelect);

  // Mask Explorer 5-tab bar (WAI-ARIA roving tabindex + handleTablistKeydown)
  const maskTabWrap = el('div', 'df-segmented df-mask-tabs');
  maskTabWrap.setAttribute('role', 'tablist');
  maskTabWrap.setAttribute('aria-label', 'Attention mask explorer modes');
  const maskTabButtons = new Map<DiffusionExplorerMaskMode, HTMLButtonElement>();
  const maskTabButtonList: HTMLButtonElement[] = [];
  function selectMaskTab(tabId: DiffusionExplorerMaskMode): void {
    if (focus.selectedMaskTab === tabId) return;
    focus = { ...focus, selectedMaskTab: tabId, pinnedCell: null, selectedPartId: null };
    rebuildSceneOnly(false);
    const tabDef = MASK_TABS.find((t) => t.id === tabId);
    announce(`Mask Explorer: ${tabDef?.label ?? tabId}`);
  }
  for (const tab of MASK_TABS) {
    const btn = el('button', 'df-seg-btn', tab.label);
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.title = tab.title;
    btn.dataset.testid = `diffusion-mask-tab-${tab.id}`;
    btn.addEventListener('click', () => selectMaskTab(tab.id));
    maskTabWrap.appendChild(btn);
    maskTabButtons.set(tab.id, btn);
    maskTabButtonList.push(btn);
  }
  maskTabWrap.addEventListener('keydown', (ev) => {
    handleTablistKeydown(maskTabButtonList, ev, (_tab, idx) => {
      const def = MASK_TABS[idx];
      if (def) selectMaskTab(def.id);
    });
  });

  // SwiGLU accumulation reveal slider
  const swigluWrap = el('label', 'df-field df-slider-field');
  swigluWrap.appendChild(el('span', 'df-field-label', 'Accum Reveal'));
  const swigluSlider = el('input', 'df-range');
  swigluSlider.type = 'range';
  swigluSlider.min = '0';
  swigluSlider.max = '100';
  swigluSlider.step = '2';
  swigluSlider.value = String(Math.round(focus.swigluAccumReveal * 100));
  swigluSlider.dataset.testid = 'diffusion-swiglu-reveal';
  swigluSlider.setAttribute('aria-label', 'SwiGLU progressive accumulation reveal percentage');
  const swigluReadout = el('span', 'df-mono-val', '100%');
  swigluSlider.addEventListener('input', () => {
    const pct = Math.max(0, Math.min(100, Number.parseInt(swigluSlider.value, 10) || 0));
    swigluReadout.textContent = `${pct}%`;
    focus = { ...focus, swigluAccumReveal: pct / 100 };
    rebuildSceneOnly(false);
  });
  swigluWrap.append(swigluSlider, swigluReadout);

  // Stepper buttons container (populated dynamically from `scene.stepperSteps`)
  const stepperWrap = el('div', 'df-segmented df-stepper-bar');
  stepperWrap.dataset.testid = 'diffusion-stepper-bar';

  contextGroup.append(
    layerWrap,
    inspectLayerBtn,
    qHeadWrap,
    kvHeadWrap,
    maskTabWrap,
    swigluWrap,
    stepperWrap,
  );

  controlStrip.append(modeGroup, el('span', 'df-sep'), execGroup, el('span', 'df-sep'), contextGroup);
  shell.appendChild(controlStrip);

  // 3. Provenance Badge Strip & Graceful Re-Prefill Banner
  const provenanceBar = el('div', 'df-provenance-bar');
  const badgeContainer = el('div', 'df-badge-row');
  badgeContainer.dataset.testid = 'diffusion-provenance-badges';
  const sessionStatsEl = el('div', 'df-session-stats');
  sessionStatsEl.dataset.testid = 'diffusion-session-stats';
  provenanceBar.append(badgeContainer, sessionStatsEl);
  shell.appendChild(provenanceBar);

  const rePrefillBanner = el('div', 'df-reprefill-banner');
  rePrefillBanner.dataset.testid = 'diffusion-reprefill-banner';
  rePrefillBanner.hidden = true;
  const rePrefillText = el('span', 'df-reprefill-text');
  const rePrefillActions = el('div', 'df-reprefill-actions');
  const rePrefillConfirmBtn = el('button', 'df-btn df-btn--warn', 'Re-Prefill KV & Switch');
  rePrefillConfirmBtn.type = 'button';
  rePrefillConfirmBtn.dataset.testid = 'diffusion-reprefill-confirm-btn';
  rePrefillConfirmBtn.addEventListener('click', () => {
    if (tour.isActive() || !pendingRePrefill) return;
    const targetPatch = pendingRePrefill.preset.patch;
    const targetLabel = pendingRePrefill.preset.label;
    pendingRePrefill = null;
    ropeScrubPosition = null;
    session.updateConfig({ ...targetPatch, allowExplicitRePrefill: true });
    rebuildFromSession(false);
    announce(`Explicit re-prefill completed for ${targetLabel}`);
  });
  const rePrefillCancelBtn = el('button', 'df-btn', 'Cancel');
  rePrefillCancelBtn.type = 'button';
  rePrefillCancelBtn.dataset.testid = 'diffusion-reprefill-cancel-btn';
  rePrefillCancelBtn.addEventListener('click', () => {
    pendingRePrefill = null;
    syncChromeUi();
  });
  rePrefillActions.append(rePrefillConfirmBtn, rePrefillCancelBtn);
  rePrefillBanner.append(rePrefillText, rePrefillActions);
  shell.appendChild(rePrefillBanner);

  const warningStrip = el('div', 'df-warning-strip');
  warningStrip.dataset.testid = 'diffusion-warning-banner';
  warningStrip.hidden = true;
  shell.appendChild(warningStrip);

  // 4. Main WebGL2 Viewport Stage + Projected Overlay + RoPE Phasor + Tour HUD + Inspector Drawer
  const stage = el('main', 'df-stage');
  const canvas = el('canvas', 'df-canvas');
  canvas.id = 'diffusion-canvas';
  canvas.dataset.testid = 'diffusion-canvas';

  const overlaySvg = svgEl('svg', { class: 'df-overlay-svg', 'aria-hidden': 'true' });
  const overlayLabels = el('div', 'df-overlay-labels');
  overlayLabels.setAttribute('aria-hidden', 'true');

  // Subtle corner view title card
  const viewCaption = el('div', 'df-view-caption');
  const viewTitleEl = el('div', 'df-view-title');
  const viewSubtitleEl = el('div', 'df-view-subtitle');
  const viewDisclosureEl = el('div', 'df-view-disclosure', DIFFUSION_PROJECTION_REVERSAL_BADGE);
  viewCaption.append(viewTitleEl, viewSubtitleEl, viewDisclosureEl);

  // 4a. Guided Tour HUD Overlay inside stage
  const tourHud = el('section', 'df-tour-hud');
  tourHud.dataset.testid = 'diffusion-tour-hud';
  tourHud.setAttribute('aria-label', 'Guided 4-Act Walkthrough');
  tourHud.hidden = true;

  const tourActsRow = el('div', 'df-tour-acts');
  const tourActButtons = new Map<DiffusionTourActId, HTMLButtonElement>();
  for (const actDef of DIFFUSION_TOUR_ACT_DEFS) {
    const actNum = actDef.actIndex + 1;
    const btn = el('button', 'df-tour-act-btn', actDef.shortTitle);
    btn.type = 'button';
    btn.dataset.testid = `diffusion-tour-act-${actNum}`;
    btn.title = actDef.title;
    btn.addEventListener('click', () => {
      const beat = tour.goToAct(actDef.id);
      if (beat) applyTourBeat(beat);
    });
    tourActsRow.appendChild(btn);
    tourActButtons.set(actDef.id, btn);
  }

  const tourMetaRow = el('div', 'df-tour-meta-row');
  const tourCounterEl = el('span', 'df-tour-counter', 'Beat 1 / 26');
  const tourProvenanceEl = el('span', 'df-tour-provenance', '');
  tourMetaRow.append(tourCounterEl, tourProvenanceEl);

  const tourTitleEl = el('h2', 'df-tour-title', '');
  const tourSubtitleEl = el('div', 'df-tour-subtitle', '');
  const tourNarrativeEl = el('p', 'df-tour-narrative', '');
  const tourTakeawayEl = el('div', 'df-tour-takeaway', '');

  const tourMetricsEl = el('div', 'df-tour-metrics');
  tourMetricsEl.dataset.testid = 'diffusion-tour-metrics';

  const tourFooter = el('div', 'df-tour-footer');
  const tourNavBtns = el('div', 'df-tour-nav-btns');
  const tourPrevBtn = el('button', 'df-btn', '← Prev');
  tourPrevBtn.type = 'button';
  tourPrevBtn.dataset.testid = 'diffusion-tour-prev';
  tourPrevBtn.addEventListener('click', () => {
    const beat = tour.prev();
    if (beat) applyTourBeat(beat);
  });

  const tourNextBtn = el('button', 'df-btn df-btn--primary', 'Next →');
  tourNextBtn.type = 'button';
  tourNextBtn.dataset.testid = 'diffusion-tour-next';
  tourNextBtn.addEventListener('click', () => {
    const beat = tour.next();
    if (beat) applyTourBeat(beat);
  });

  const tourExitBtn = el('button', 'df-btn df-btn--subtle', 'Exit Tour');
  tourExitBtn.type = 'button';
  tourExitBtn.dataset.testid = 'diffusion-tour-exit';
  tourExitBtn.addEventListener('click', () => exitGuidedTour());

  tourNavBtns.append(tourPrevBtn, tourNextBtn);
  tourFooter.append(tourNavBtns, tourExitBtn);

  tourHud.append(
    tourActsRow,
    tourMetaRow,
    tourTitleEl,
    tourSubtitleEl,
    tourNarrativeEl,
    tourTakeawayEl,
    tourMetricsEl,
    tourFooter,
  );

  // 4b. Collapsible 3-Tab Editorial Inspector Drawer
  const inspectorDrawer = el('aside', 'df-inspector-drawer');
  inspectorDrawer.id = 'diffusion-inspector-drawer';
  inspectorDrawer.dataset.testid = 'diffusion-inspector-drawer';
  inspectorDrawer.setAttribute('aria-label', 'Diffusion editorial inspector');
  inspectorDrawer.hidden = true;

  const inspectorHeader = el('div', 'df-inspector-header');
  const inspectorTitle = el('span', 'df-inspector-title', 'Editorial Inspector');
  const inspectorCloseBtn = el('button', 'df-inspector-close', '×');
  inspectorCloseBtn.type = 'button';
  inspectorCloseBtn.setAttribute('aria-label', 'Close inspector drawer');
  inspectorCloseBtn.title = 'Close inspector (Escape)';
  inspectorCloseBtn.addEventListener('click', () => setInspectorOpen(false));
  inspectorHeader.append(inspectorTitle, inspectorCloseBtn);

  const inspectorTablist = el('div', 'df-inspector-tabs');
  inspectorTablist.setAttribute('role', 'tablist');
  inspectorTablist.setAttribute('aria-label', 'Diffusion inspector sections');
  const inspectorTabButtons = new Map<DiffusionInspectorTabId, HTMLButtonElement>();
  const inspectorTabButtonList: HTMLButtonElement[] = [];
  for (const tabDef of INSPECTOR_TABS) {
    const btn = el('button', 'df-inspector-tab', tabDef.label);
    btn.type = 'button';
    btn.id = `diffusion-inspector-tab-${tabDef.id}`;
    btn.dataset.testid = `diffusion-inspector-tab-${tabDef.id}`;
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-controls', 'diffusion-inspector-panel');
    btn.title = tabDef.title;
    btn.addEventListener('click', () => setInspectorTab(tabDef.id));
    inspectorTablist.appendChild(btn);
    inspectorTabButtons.set(tabDef.id, btn);
    inspectorTabButtonList.push(btn);
  }
  inspectorTablist.addEventListener('keydown', (ev) => {
    handleTablistKeydown(inspectorTabButtonList, ev, (_tab, idx) => {
      const def = INSPECTOR_TABS[idx];
      if (def) setInspectorTab(def.id);
    });
  });

  const inspectorPanel = el('div', 'df-inspector-panel');
  inspectorPanel.id = 'diffusion-inspector-panel';
  inspectorPanel.dataset.testid = 'diffusion-inspector-panel';
  inspectorPanel.setAttribute('role', 'tabpanel');
  inspectorPanel.setAttribute('aria-labelledby', 'diffusion-inspector-tab-inspect');

  // Persistent sub-containers inside inspectorPanel so typing in Ledger search never loses focus
  const inspectBody = el('div', 'df-inspector-body');
  const archBody = el('div', 'df-inspector-body');
  const ledgerBody = el('div', 'df-inspector-body');

  const ledgerControls = el('div', 'df-ledger-controls');
  const ledgerSearchInput = el('input', 'df-ledger-search');
  ledgerSearchInput.type = 'search';
  ledgerSearchInput.placeholder = 'Filter D1–D24 by ID, topic, or ground truth…';
  ledgerSearchInput.dataset.testid = 'diffusion-ledger-search';
  ledgerSearchInput.setAttribute('aria-label', 'Search discrepancy ledger D1 to D24');
  ledgerSearchInput.addEventListener('input', () => {
    ledgerQuery = ledgerSearchInput.value;
    renderLedgerList();
  });

  const ledgerCatBar = el('div', 'df-ledger-cats');
  const ledgerCatButtons = new Map<DiffusionDiscrepancyCategory, HTMLButtonElement>();
  for (const cat of LEDGER_CATEGORIES) {
    const btn = el('button', 'df-ledger-cat-btn', cat.label);
    btn.type = 'button';
    btn.dataset.testid = `diffusion-ledger-cat-${cat.id}`;
    btn.addEventListener('click', () => {
      ledgerCategory = cat.id;
      renderLedgerList();
    });
    ledgerCatBar.appendChild(btn);
    ledgerCatButtons.set(cat.id, btn);
  }
  const ledgerMatchCountEl = el('div', 'df-ledger-count', '24 / 24 discrepancies');
  ledgerControls.append(ledgerSearchInput, ledgerCatBar, ledgerMatchCountEl);

  const ledgerListWrap = el('div', 'df-ledger-list');
  ledgerBody.append(ledgerControls, ledgerListWrap);

  inspectorPanel.append(inspectBody, archBody, ledgerBody);
  inspectorDrawer.append(inspectorHeader, inspectorTablist, inspectorPanel);

  // 4c. Interactive 2D Unit-Circle RoPE Phasor SVG Inset (visible in `gqa_rope`)
  const phasorCard = el('aside', 'df-rope-phasor');
  phasorCard.dataset.testid = 'diffusion-rope-phasor';
  phasorCard.hidden = true;

  const phasorHead = el('div', 'df-phasor-head');
  const phasorTitle = el('span', 'df-phasor-title', '2D Unit-Circle RoPE Phasor');
  const phasorZoneBadge = el('span', 'df-phasor-zone-badge', 'original');
  phasorHead.append(phasorTitle, phasorZoneBadge);

  const phasorPlaneBar = el('div', 'df-phasor-planes');
  const planeButtons: HTMLButtonElement[] = [];
  for (let pIdx = 0; pIdx < 4; pIdx++) {
    const btn = el('button', 'df-plane-btn');
    btn.type = 'button';
    btn.dataset.testid = `diffusion-rope-plane-${pIdx}`;
    const labelSpan = el('span', 'df-plane-pair', ROTARY_PLANE_LABELS[pIdx]!);
    const freqSpan = el('span', 'df-plane-freq', String(DIFFUSION_DYADIC_YARN_INV_FREQ[pIdx]!));
    btn.append(labelSpan, freqSpan);
    btn.addEventListener('click', () => {
      if (focus.selectedRotaryPlane === pIdx) return;
      focus = { ...focus, selectedRotaryPlane: pIdx };
      rebuildSceneOnly(false);
    });
    phasorPlaneBar.appendChild(btn);
    planeButtons.push(btn);
  }

  const UNIT_R = 96;
  const LLAMA4_R = UNIT_R * (1 + 0.1 * Math.log(2));

  const phasorSvg = svgEl('svg', {
    class: 'df-phasor-svg',
    viewBox: '-136 -136 272 272',
    role: 'img',
    'aria-label': '2D RoPE Unit Circle and Llama-4 Query Scale Phasor Diagram',
  });

  phasorSvg.append(
    svgEl('line', { x1: -122, y1: 0, x2: 122, y2: 0, class: 'df-ph-axis' }),
    svgEl('line', { x1: 0, y1: -122, x2: 0, y2: 122, class: 'df-ph-axis' }),
    svgEl('circle', { cx: 0, cy: 0, r: UNIT_R.toFixed(2), class: 'df-ph-unit-ring' }),
    svgEl('circle', { cx: 0, cy: 0, r: LLAMA4_R.toFixed(2), class: 'df-ph-l4-ring' }),
    svgEl('text', { x: 4, y: -(UNIT_R - 11), class: 'df-ph-ring-label' }, 'r=1.0'),
    svgEl('text', { x: 4, y: -(LLAMA4_R + 4), class: 'df-ph-ring-label df-ph-ring-label--l4' }, 'r=1.069 (L4 p≥16)'),
    svgEl('text', { x: 112, y: -5, class: 'df-ph-axis-label' }, '+x₁'),
    svgEl('text', { x: 5, y: -116, class: 'df-ph-axis-label' }, '+x₂'),
  );

  const phAngleArc = svgEl('path', { class: 'df-ph-arc', d: '' });
  const phQPreLine = svgEl('line', { class: 'df-ph-vec df-ph-vec--pre', x1: 0, y1: 0, x2: UNIT_R, y2: 0 });
  const phKPreLine = svgEl('line', { class: 'df-ph-vec df-ph-vec--kpre', x1: 0, y1: 0, x2: UNIT_R, y2: 0 });
  const phKRotLine = svgEl('line', { class: 'df-ph-vec df-ph-vec--krot', x1: 0, y1: 0, x2: UNIT_R, y2: 0 });
  const phQScaledLine = svgEl('line', { class: 'df-ph-vec df-ph-vec--qscaled', x1: 0, y1: 0, x2: UNIT_R, y2: 0 });
  const phKRotDot = svgEl('circle', { class: 'df-ph-dot df-ph-dot--krot', cx: UNIT_R, cy: 0, r: 4 });
  const phQScaledDot = svgEl('circle', { class: 'df-ph-dot df-ph-dot--qscaled', cx: UNIT_R, cy: 0, r: 4.5 });
  const phOriginDot = svgEl('circle', { class: 'df-ph-origin', cx: 0, cy: 0, r: 2.5 });
  const phKLabel = svgEl('text', { class: 'df-ph-vec-label df-ph-vec-label--krot', x: 0, y: 0 }, 'k_rot');
  const phQLabel = svgEl('text', { class: 'df-ph-vec-label df-ph-vec-label--qscaled', x: 0, y: 0 }, 'q_scaled');

  phasorSvg.append(
    phAngleArc,
    phQPreLine,
    phKPreLine,
    phKRotLine,
    phQScaledLine,
    phKRotDot,
    phQScaledDot,
    phOriginDot,
    phKLabel,
    phQLabel,
  );

  const scrubberWrap = el('div', 'df-phasor-scrubber');
  const scrubberTop = el('div', 'df-phasor-scrubber-top');
  const scrubberPosLabel = el('span', 'df-phasor-pos-label', 'p = 0');
  const scrubberSyncBtn = el('button', 'df-mini-btn', 'Sync Live');
  scrubberSyncBtn.type = 'button';
  scrubberSyncBtn.dataset.testid = 'diffusion-rope-pos-sync';
  scrubberSyncBtn.addEventListener('click', () => {
    ropeScrubPosition = null;
    updateRopePhasorInset();
  });
  scrubberTop.append(scrubberPosLabel, scrubberSyncBtn);

  const scrubberTrack = el('div', 'df-scrubber-track');
  const posSlider = el('input', 'df-range df-rope-slider');
  posSlider.type = 'range';
  posSlider.min = '0';
  posSlider.max = '31';
  posSlider.step = '1';
  posSlider.value = '0';
  posSlider.dataset.testid = 'diffusion-rope-pos-scrubber';
  posSlider.setAttribute('aria-label', 'RoPE position index 0 to 31');
  posSlider.addEventListener('input', () => {
    const p = Number.parseInt(posSlider.value, 10);
    if (Number.isInteger(p) && p >= 0 && p <= 31) {
      ropeScrubPosition = p;
      updateRopePhasorInset();
    }
  });
  const boundaryTick = el('div', 'df-scrubber-boundary');
  boundaryTick.style.left = `${(16 / 31) * 100}%`;
  boundaryTick.title = 'p = 16 · Llama-4 query temperature scale boundary (1.0 → 1.0693)';
  const boundaryCaption = el(
    'div',
    'df-scrubber-ticks',
    '0 · · · · · · · p = 16 · Llama-4 scale boundary · · · · · · · 31',
  );
  scrubberTrack.append(posSlider, boundaryTick);
  scrubberWrap.append(scrubberTop, scrubberTrack, boundaryCaption);

  const phasorLegend = el('div', 'df-phasor-legend');
  phasorLegend.innerHTML =
    '<span class="df-leg-item"><i class="df-leg-swatch df-leg-swatch--pre"></i>q/k_pre</span>' +
    '<span class="df-leg-item"><i class="df-leg-swatch df-leg-swatch--k"></i>k_rot (r=1.0)</span>' +
    '<span class="df-leg-item"><i class="df-leg-swatch df-leg-swatch--q"></i>q_scaled</span>';

  const phasorTelemetryEl = el('div', 'df-phasor-telemetry');
  phasorCard.append(
    phasorHead,
    phasorPlaneBar,
    phasorSvg,
    scrubberWrap,
    phasorLegend,
    phasorTelemetryEl,
  );

  // WebGL error fallback notice (hidden when WebGL2 works)
  const webglFallbackEl = el('div', 'df-webgl-fallback');
  webglFallbackEl.hidden = true;

  stage.append(
    canvas,
    overlaySvg,
    overlayLabels,
    viewCaption,
    tourHud,
    phasorCard,
    inspectorDrawer,
    webglFallbackEl,
  );
  shell.appendChild(stage);

  // 5. Unified Tri-Mode Event Timeline Dock (mounted between stage and formulaDock)
  const timelineDock = createDiffusionTimelineDock(shell, {
    onSeekRecord(recordId: string) {
      if (tour.isActive()) return;
      try {
        const rec = session.seekRecord(recordId);
        pendingRePrefill = null;
        ropeScrubPosition = null;
        rebuildFromSession(false);
        announce(`Timeline seek: record #${rec.recordIndex} on ${rec.branchId}`);
      } catch {
        // Ignore stale record clicks
      }
    },
    onDeleteBranch(branchId: string) {
      if (tour.isActive()) return;
      if (session.deleteBranch(branchId)) {
        rebuildFromSession(false);
        announce(`Deleted inactive branch ${branchId}`);
      }
    },
  });

  // 6. Docked Formula Inspector Bar (`activeFormula` + selected part + token stream)
  const formulaDock = el('footer', 'df-formula-dock');
  formulaDock.dataset.testid = 'diffusion-formula-dock';
  const formulaLeft = el('div', 'df-formula-left');
  const formulaTensorEl = el('span', 'df-formula-tensor');
  const formulaCoordEl = el('span', 'df-formula-coord');
  const formulaBadgeEl = el('span', 'df-formula-badge');
  formulaLeft.append(formulaTensorEl, formulaCoordEl, formulaBadgeEl);

  const formulaCenter = el('div', 'df-formula-center');
  const formulaEqEl = el('div', 'df-formula-eq');
  const formulaSubEl = el('div', 'df-formula-sub');
  formulaCenter.append(formulaEqEl, formulaSubEl);

  const formulaRight = el('div', 'df-formula-right');
  const formulaValEl = el('span', 'df-formula-val');
  const tokenPreviewEl = el('span', 'df-token-preview');
  formulaRight.append(formulaValEl, tokenPreviewEl);

  formulaDock.append(formulaLeft, formulaCenter, formulaRight);
  shell.appendChild(formulaDock);

  // ==========================================================================
  // INITIALIZE 5-PASS WEBGL2 PIPELINE & POOLED OVERLAY
  // ==========================================================================

  const overlay = new DiffusionSpatialOverlay(overlaySvg, overlayLabels);

  let device: RenderDevice | null = null;
  let solidsPass: SolidsPass | null = null;
  let linesPass: LinesPass | null = null;
  let ribbonsPass: RibbonsPass | null = null;
  let flatFlowPass: FlatFlowRibbonsPass | null = null;
  let sheetsPass: SheetsPass | null = null;

  try {
    device = new RenderDevice(canvas);
    const gl = device.gl;
    solidsPass = new SolidsPass(gl);
    linesPass = new LinesPass(gl);
    ribbonsPass = new RibbonsPass(gl);
    flatFlowPass = new FlatFlowRibbonsPass(gl);
    sheetsPass = new SheetsPass(gl);
  } catch (err) {
    webglFallbackEl.hidden = false;
    webglFallbackEl.textContent =
      err instanceof Error ? `WebGL2 unavailable: ${err.message}` : 'WebGL2 unavailable.';
  }

  const onContextLost = (ev: Event) => {
    ev.preventDefault();
    if (device) {
      device.markLost(true);
    }
    lastUploadedAtlasKey = null;
    sceneUploadPending = true;
  };

  const onContextRestored = () => {
    if (!device) return;
    device.markLost(false);
    try {
      solidsPass?.init();
      linesPass?.init();
      ribbonsPass?.init();
      flatFlowPass?.init();
      sheetsPass?.init();
    } catch {
      glErrorCount++;
    }
    lastUploadedAtlasKey = null;
    sceneUploadPending = true;
    markDirtyAndRender();
  };

  canvas.addEventListener('webglcontextlost', onContextLost, false);
  canvas.addEventListener('webglcontextrestored', onContextRestored, false);

  // ==========================================================================
  // VIEWPORT METRICS, CAMERA FITTING & IDLE-GATED RAF LOOP
  // ==========================================================================

  function measureViewport(): ViewportMetrics {
    const rect = stage.getBoundingClientRect();
    const w = Math.max(1, rect.width || stage.clientWidth || 960);
    const h = Math.max(1, rect.height || stage.clientHeight || 540);
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    return makeViewportMetrics(rect.left, rect.top, w, h, dpr, 0, 0);
  }

  function fitCameraToScene(animate: boolean): void {
    const metrics = measureViewport();
    const targetPose = fitPerspectiveToBounds(
      scene.fitBounds,
      scene.defaultCameraAzimuthDeg,
      scene.defaultCameraElevationDeg,
      metrics.innerWidth,
      metrics.innerHeight,
      DEFAULT_FOV_DEG,
      0.84,
    );
    if (!cameraInitialized || !animate || isReducedMotionPreferred()) {
      cameraPose = targetPose;
      cameraAnim = null;
      cameraInitialized = true;
    } else {
      cameraAnim = {
        from: cloneCameraPose(cameraPose),
        to: targetPose,
        startMs: performance.now(),
        durationMs: CAMERA_FLY_DURATION_MS,
      };
    }
    markDirtyAndRender();
  }

  function uploadSceneBuffersToGpu(): void {
    if (!device || device.isLost || !solidsPass || !linesPass || !ribbonsPass || !flatFlowPass || !sheetsPass) {
      return;
    }

    solidsPass.upload(scene.solidData, scene.solidCount);
    linesPass.upload(scene.lineData, scene.lineCount);
    ribbonsPass.upload(scene.ribbonData, scene.ribbonCount);

    if (scene.flowVertices && scene.flowIndices && scene.flowIndices.length > 0) {
      flatFlowPass.upload(scene.flowVertices, scene.flowIndices);
    } else {
      flatFlowPass.upload(new Float32Array(0), new Uint32Array(0));
    }

    if (scene.sheetData && scene.sheetCount && scene.sheetCount > 0) {
      sheetsPass.upload(
        scene.sheetData,
        scene.sheetCount,
        scene.sheetBasisData,
        scene.sheetRevealData,
      );
    } else {
      sheetsPass.upload(new Float32Array(0), 0);
    }

    // Atlas Content-Key Deduplication: ONLY upload RG32F texture when `scene.atlasKey` changes
    if (scene.valueAtlas && scene.valueAtlas.width > 0 && scene.valueAtlas.height > 0) {
      const effectiveKey =
        scene.atlasKey ??
        `${scene.viewMode}:${scene.valueAtlas.width}x${scene.valueAtlas.height}`;
      if (effectiveKey !== lastUploadedAtlasKey) {
        sheetsPass.uploadValueAtlas(
          scene.valueAtlas.width,
          scene.valueAtlas.height,
          scene.valueAtlas.data,
        );
        lastUploadedAtlasKey = effectiveKey;
        atlasUploadCount++;
      }
      atlasWidth = scene.valueAtlas.width;
      atlasHeight = scene.valueAtlas.height;
    } else {
      atlasWidth = 0;
      atlasHeight = 0;
    }

    sceneUploadPending = false;
  }

  function pollGlErrors(): void {
    if (!device || device.isLost) return;
    const gl = device.gl;
    for (let i = 0; i < 16; i++) {
      const err = gl.getError();
      if (err === gl.NO_ERROR) break;
      glErrorCount++;
      lastGlError = err;
    }
  }

  function computeFloatingAvoidRects(): OverlayRect[] {
    const rects: OverlayRect[] = [];
    const stageRect = stage.getBoundingClientRect();
    const pushIfVisible = (node: HTMLElement, pad: number) => {
      if (node.hidden) return;
      const r = node.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) {
        rects.push({
          x0: r.left - stageRect.left - pad,
          y0: r.top - stageRect.top - pad,
          x1: r.right - stageRect.left + pad,
          y1: r.bottom - stageRect.top + pad,
        });
      }
    };
    pushIfVisible(phasorCard, 6);
    pushIfVisible(viewCaption, 4);
    pushIfVisible(tourHud, 6);
    pushIfVisible(inspectorDrawer, 6);
    return rects;
  }

  function renderFrame(nowMs: number): void {
    if (disposed) return;

    let keepAnimating = false;
    if (cameraAnim) {
      const elapsed = nowMs - cameraAnim.startMs;
      const t = cameraAnim.durationMs > 0 ? Math.min(1, elapsed / cameraAnim.durationMs) : 1;
      cameraPose = interpolateCameraPose(cameraAnim.from, cameraAnim.to, t);
      if (t >= 1) {
        cameraAnim = null;
      } else {
        keepAnimating = true;
      }
    }

    const metrics = measureViewport();
    const snapshot = buildCameraSnapshot(cameraPose, scene.fitBounds, metrics);
    lastSnapshot = snapshot;

    if (device && !device.isLost && solidsPass && linesPass && ribbonsPass && flatFlowPass && sheetsPass) {
      if (sceneUploadPending) {
        uploadSceneBuffersToGpu();
      }
      const fb = device.ensureSize(metrics);
      device.beginFrame(fb);
      solidsPass.draw(snapshot.viewProj);
      sheetsPass.draw(snapshot.viewProj);
      linesPass.draw(snapshot.viewProj, fb.vpWidth, fb.vpHeight, metrics.dpr, snapshot.eye);
      ribbonsPass.draw(snapshot.viewProj);
      flatFlowPass.draw(snapshot.viewProj, -1);
      device.endFrame(fb);
      pollGlErrors();
    }

    overlay.update({
      scene,
      snapshot,
      selectedPartId: focus.selectedPartId,
      hoveredPartId: focus.hoveredPartId,
      activeKvSeqLen: resolvedCtx.kvSnapshot.seqLen,
      topInsetPx: 12,
      bottomInsetPx: 14,
      popoverAvoidRects: computeFloatingAvoidRects(),
    });

    frameCount++;
    dirty = false;

    // Strict idle-RAF gating: ONLY schedule another RAF if animating or still dirty
    if (keepAnimating || dirty) {
      scheduleRaf();
    }
  }

  function scheduleRaf(): void {
    if (disposed || rafId !== null) return;
    rafActive = true;
    rafId = window.requestAnimationFrame((ts) => {
      rafId = null;
      rafActive = false;
      renderFrame(ts);
    });
  }

  function markDirtyAndRender(): void {
    dirty = true;
    scheduleRaf();
  }

  // ==========================================================================
  // 3-TAB EDITORIAL INSPECTOR RENDERING
  // ==========================================================================

  function makeCard(title: string, badge?: string): HTMLElement {
    const card = el('section', 'df-ins-card');
    const head = el('div', 'df-ins-card-head');
    head.appendChild(el('h3', 'df-ins-card-title', title));
    if (badge) head.appendChild(el('span', 'df-ins-card-badge', badge));
    card.appendChild(head);
    return card;
  }

  function renderInspectTab(): void {
    inspectBody.innerHTML = '';
    const rec = resolvedCtx.record;
    const nfe = buildDiffusionNfeTpfReadout(
      resolvedCtx.modeAtRecord,
      rec.counters,
      rec.generatedTokenIds.length,
      resolvedCtx.causalContextAtRecord,
    );

    // 1. Provenance & Record Summary
    const provCard = makeCard(
      'Active Record Provenance',
      tour.isActive() ? 'Tour (Isolated)' : `${resolvedCtx.branchId} · #${rec.recordIndex}`,
    );
    const provBadges = el('div', 'df-ins-badge-wrap');
    for (const b of scene.provenanceBadges) {
      provBadges.appendChild(el('span', `df-badge df-badge--${b.tone}`, b.label));
    }
    const provMeta = el('div', 'df-ins-kv-grid');
    const kvItems: [string, string][] = [
      ['Record', `${rec.recordId} (${rec.kind})`],
      ['Lifecycle', rec.lifecyclePhase],
      ['KV Cache', `${resolvedCtx.kvSnapshot.seqLen} / ${resolvedCtx.kvSnapshot.capacity} tok (${rec.kvMaskProvenance})`],
      ['Session RAM', formatDualSiIecBytes(cachedMemoryReport.totalRetainedBytes).dualLabel],
    ];
    for (const [k, v] of kvItems) {
      const row = el('div', 'df-ins-kv-row');
      row.append(el('span', 'df-ins-k', k), el('span', 'df-ins-v', v));
      provMeta.appendChild(row);
    }
    provCard.append(provBadges, provMeta);
    inspectBody.appendChild(provCard);

    // 2. Live 3-Counter NFE & TPF Readout (D12 Honest Accounting)
    const nfeCard = makeCard('Live 3-Counter NFE & TPF (D12)', nfe.modeLabel);
    const nfeGrid = el('div', 'df-ins-nfe-grid');
    const nfeCells: [string, string, string][] = [
      ['Actual Fwd', String(nfe.actualForwardPasses), `TPF ${nfe.tpfActualLabel}`],
      ['Decode Fwd', String(nfe.decodeForwardPasses), `TPF ${nfe.tpfDecodeLabel}`],
      ['Ref NFE (D12)', String(nfe.referenceMethodNfe), `TPF ${nfe.tpfRefLabel}`],
      ['Generated', `${nfe.generatedTokens} tok`, 'Live'],
    ];
    for (const [label, big, sub] of nfeCells) {
      const cell = el('div', 'df-ins-stat-box');
      cell.append(
        el('span', 'df-ins-stat-label', label),
        el('strong', 'df-ins-stat-val', big),
        el('span', 'df-ins-stat-sub', sub),
      );
      nfeGrid.appendChild(cell);
    }
    nfeCard.append(
      nfeGrid,
      el('p', 'df-ins-note', nfe.methodNote),
      el('p', 'df-ins-note df-ins-note--muted', nfe.d12ComparisonNote),
    );
    inspectBody.appendChild(nfeCard);

    // 3. Active Formula / Selected Part Details
    const formulaCard = makeCard('Active Formula / Part');
    if (scene.activeFormula) {
      const f = scene.activeFormula;
      formulaCard.append(
        el('div', 'df-ins-mono-strong', `${f.tensorLabel} · ${f.coordLabel} = ${f.valueText}`),
        el('div', 'df-ins-code', f.equation),
        el('div', 'df-ins-note df-ins-note--muted', f.subEquation),
      );
    } else if (focus.selectedPartId && scene.partCatalog[focus.selectedPartId]) {
      const p = scene.partCatalog[focus.selectedPartId]!;
      formulaCard.append(
        el('div', 'df-ins-mono-strong', `${p.title} (${p.badge})`),
        el('p', 'df-ins-note', p.summary),
      );
      if (p.specs.length > 0) {
        const ul = el('ul', 'df-ins-list');
        for (const s of p.specs) ul.appendChild(el('li', undefined, s));
        formulaCard.appendChild(ul);
      }
    } else {
      formulaCard.append(
        el('div', 'df-ins-mono-strong', scene.title),
        el('p', 'df-ins-note df-ins-note--muted', 'Click any 3D sheet cell or part to pin its live equation here.'),
      );
    }
    inspectBody.appendChild(formulaCard);

    // 4. Guards & D19 Dual-Crop Provenance
    if (rec.cropProvenance || rec.guardFlags.warnings.length > 0 || rec.guardFlags.draftGuardTriggered !== 'none') {
      const guardCard = makeCard('Guards & D19 Dual-Crop Provenance');
      if (rec.cropProvenance) {
        const cp = rec.cropProvenance;
        const cpGrid = el('div', 'df-ins-kv-grid');
        const cpRows: [string, string][] = [
          ['Before Verify KV', `${cp.cacheLenBeforeVerify} tok`],
          ['Pre-Crop KV', `${cp.preCropKvLen} tok`],
          ['Browser Post-EOS Crop', `${cp.browserPostEosCropLen} tok`],
          ['Upstream Pre-EOS Crop (D19)', `${cp.upstreamPreEosCropLen} tok`],
          ['Cropped Suffix', `${cp.croppedSuffixLen} slots`],
        ];
        for (const [k, v] of cpRows) {
          const row = el('div', 'df-ins-kv-row');
          row.append(el('span', 'df-ins-k', k), el('span', 'df-ins-v', v));
          cpGrid.appendChild(row);
        }
        guardCard.appendChild(cpGrid);
      }
      if (rec.guardFlags.warnings.length > 0) {
        const ul = el('ul', 'df-ins-list df-ins-list--warn');
        for (const w of rec.guardFlags.warnings) ul.appendChild(el('li', undefined, w));
        guardCard.appendChild(ul);
      }
      inspectBody.appendChild(guardCard);
    }

    // 5. Decoded V=36 Token Stream
    const tokCard = makeCard(
      'Committed V=36 Token Stream',
      `${resolvedCtx.kvSnapshot.committedTokenIds.length} tok`,
    );
    const tokWrap = el('div', 'df-ins-tokens');
    const committed = resolvedCtx.kvSnapshot.committedTokenIds;
    for (let i = 0; i < committed.length; i++) {
      const id = committed[i]!;
      const label = DIFFUSION_MINI_VOCAB[id] ?? `#${id}`;
      const chip = el('span', 'df-ins-tok-chip', `${i}:${label}`);
      if (id >= 32) chip.classList.add('is-special');
      if (i >= DIFFUSION_MINI_CONFIG.defaultPromptLen) chip.classList.add('is-gen');
      tokWrap.appendChild(chip);
    }
    tokCard.appendChild(tokWrap);
    inspectBody.appendChild(tokCard);
  }

  function renderArchitectureTab(): void {
    archBody.innerHTML = '';
    const rec = resolvedCtx.record;
    const bundle = buildDiffusionArchReadouts({
      mode: resolvedCtx.modeAtRecord,
      causalContext: resolvedCtx.causalContextAtRecord,
      counters: rec.counters,
      generatedTokens: rec.generatedTokenIds.length,
      memoryReport: cachedMemoryReport,
      activeAtlasBytes: scene.valueAtlas ? scene.valueAtlas.width * scene.valueAtlas.height * 8 : undefined,
      activeAtlasWidth: scene.valueAtlas?.width,
      activeAtlasHeight: scene.valueAtlas?.height,
      activeAtlasName: scene.viewMode,
    });

    // 1. Canonical 14B Instruct Summary
    const c14 = bundle.canonical14B;
    const summaryCard = makeCard(
      'Canonical 14B Instruct (Verified 0 B LFS)',
      c14.lfsAllZeroDiff ? `${c14.lfsFileCount} files · 0 B diff` : 'Verify',
    );
    const summaryGrid = el('div', 'df-ins-kv-grid');
    const sumRows: [string, string][] = [
      ['HF Repo', `${c14.repoId} @ ${c14.shortRevision}`],
      ['Paper / GitHub', `${c14.paperArxivId} · ${c14.githubRepo}`],
      ['Backbone', `${c14.backboneParamsFormatted} (${c14.backboneTensors} BF16) · ${c14.backbonePayload.dualLabel}`],
      ['o_proj LoRA', `${c14.loraParamsFormatted} (${c14.loraTensors} F32, ${c14.loraShareOfBackbonePct}) · ${c14.loraPayload.dualLabel}`],
    ];
    for (const [k, v] of sumRows) {
      const row = el('div', 'df-ins-kv-row');
      row.append(el('span', 'df-ins-k', k), el('span', 'df-ins-v', v));
      summaryGrid.appendChild(row);
    }
    summaryCard.append(
      summaryGrid,
      el('div', 'df-ins-callout df-ins-callout--amber', c14.projectionReversalNote),
      el('p', 'df-ins-note', c14.epistemicNote),
      el('p', 'df-ins-note df-ins-note--muted', c14.outOfScopeNote),
    );
    archBody.appendChild(summaryCard);

    // 2. 14B Component Breakdown Table
    const compCard = makeCard('14B Component Breakdown', `${bundle.components14B.length} groups`);
    const compTable = el('table', 'df-ins-table');
    compTable.innerHTML =
      '<thead><tr><th>Component</th><th>Shape</th><th>Params</th><th>Share</th><th>Payload</th></tr></thead>';
    const compTbody = el('tbody');
    for (const row of bundle.components14B) {
      const tr = el('tr');
      tr.append(
        el('td', undefined, row.component),
        el('td', 'df-mono-cell', row.shape),
        el('td', 'df-mono-cell', row.parametersFormatted),
        el('td', 'df-mono-cell', row.shareOfBackbonePct),
        el('td', 'df-mono-cell', row.payload.dualLabel),
      );
      compTbody.appendChild(tr);
    }
    compTable.appendChild(compTbody);
    const compScroll = el('div', 'df-ins-table-scroll');
    compScroll.appendChild(compTable);
    compCard.appendChild(compScroll);
    archBody.appendChild(compCard);

    // 3. 7-Variant Family Matrix
    const famCard = makeCard('7-Variant Family Matrix', 'HF Collection');
    const famTable = el('table', 'df-ins-table');
    famTable.innerHTML =
      '<thead><tr><th>ID</th><th>Layers</th><th>d / d_ff</th><th>YaRN / θ</th><th>Backbone</th><th>LoRA</th></tr></thead>';
    const famTbody = el('tbody');
    for (const v of bundle.familyVariants) {
      const tr = el('tr', v.isCanonical ? 'is-canonical' : undefined);
      tr.append(
        el('td', 'df-mono-cell', v.isCanonical ? `${v.id} ★` : v.id),
        el('td', 'df-mono-cell', v.layersLabel),
        el('td', 'df-mono-cell', `${v.hiddenSize}/${v.intermediateSize}`),
        el('td', 'df-mono-cell', `${v.yarnLabel} · ${v.ropeThetaLabel}`),
        el('td', 'df-mono-cell', v.backboneParamsFormatted),
        el('td', 'df-mono-cell', v.loraSummary),
      );
      famTbody.appendChild(tr);
    }
    famTable.appendChild(famTbody);
    const famScroll = el('div', 'df-ins-table-scroll');
    famScroll.appendChild(famTable);
    famCard.appendChild(famScroll);
    archBody.appendChild(famCard);

    // 4. 4-Tier Context KV Scaling Table (with Llama-4 16,384 transition highlight)
    const kvCard = makeCard('4-Tier Context KV Scaling & Llama-4 Boundary', '14B BF16');
    const kvTable = el('table', 'df-ins-table');
    kvTable.innerHTML =
      '<thead><tr><th>Context</th><th>Tier</th><th>14B KV</th><th>+B=32 Buf</th><th>Llama-4 s_L4</th></tr></thead>';
    const kvTbody = el('tbody');
    for (const row of bundle.contextKvScaling) {
      const tr = el('tr', row.crossesOrigMaxPos ? 'is-l4-cross' : undefined);
      tr.title = `${row.tierProvenance} — ${row.transitionNote}`;
      tr.append(
        el('td', 'df-mono-cell', row.tokenLabel),
        el('td', undefined, `T${row.tier} · ${row.tierBadge}`),
        el('td', 'df-mono-cell', row.prodBf16Kv.dualLabel),
        el('td', 'df-mono-cell', row.prodWithTempBlock32.dualLabel),
        el('td', 'df-mono-cell', row.llama4ScaleLabel),
      );
      kvTbody.appendChild(tr);
    }
    kvTable.appendChild(kvTbody);
    const kvScroll = el('div', 'df-ins-table-scroll');
    kvScroll.appendChild(kvTable);
    kvCard.appendChild(kvScroll);
    archBody.appendChild(kvCard);

    // 5. 4-Way Memory Boundary Table
    const memCard = makeCard('4-Way Browser Memory Boundaries', bundle.memoryTotalMiniFormatted.dualLabel);
    const memTable = el('table', 'df-ins-table');
    memTable.innerHTML =
      '<thead><tr><th>Boundary</th><th>Current</th><th>Cap</th><th>Status</th></tr></thead>';
    const memTbody = el('tbody');
    for (const row of bundle.memoryBoundaries) {
      const tr = el('tr');
      tr.title = `${row.ownership} — ${row.note}`;
      const statusText = row.utilizationPct !== null
        ? `${row.utilizationPct.toFixed(1)}% (${row.withinCap ? 'OK' : 'OVER'})`
        : 'Fixed';
      tr.append(
        el('td', undefined, row.label),
        el('td', 'df-mono-cell', row.formatted.dualLabel),
        el('td', 'df-mono-cell', row.capFormatted ?? '—'),
        el('td', 'df-mono-cell', statusText),
      );
      memTbody.appendChild(tr);
    }
    memTable.appendChild(memTbody);
    const memScroll = el('div', 'df-ins-table-scroll');
    memScroll.appendChild(memTable);
    memCard.appendChild(memScroll);
    archBody.appendChild(memCard);

    // 6. All 13 Verified 0 B LFS Files Table
    const lfsCard = makeCard('Verified 0 B LFS Inventory', `${bundle.lfsInventory.length} safetensors`);
    const lfsTable = el('table', 'df-ins-table');
    lfsTable.innerHTML =
      '<thead><tr><th>Repo / File</th><th>Dtype</th><th>Tensors</th><th>Total</th><th>Diff</th></tr></thead>';
    const lfsTbody = el('tbody');
    for (const row of bundle.lfsInventory) {
      const tr = el('tr');
      tr.title = `${row.repoId}@${row.shortRevision} · sha256:${row.lfsSha256Prefix}`;
      tr.append(
        el('td', 'df-mono-cell', `${row.shortRepoLabel}/${row.file}`),
        el('td', 'df-mono-cell', row.dtype),
        el('td', 'df-mono-cell', String(row.tensors)),
        el('td', 'df-mono-cell', row.total.dualLabel),
        el('td', 'df-mono-cell', `${row.diffBytes} B`),
      );
      lfsTbody.appendChild(tr);
    }
    lfsTable.appendChild(lfsTbody);
    const lfsScroll = el('div', 'df-ins-table-scroll');
    lfsScroll.appendChild(lfsTable);
    lfsCard.appendChild(lfsScroll);
    archBody.appendChild(lfsCard);
  }

  function renderLedgerList(): void {
    const model = filterDiffusionDiscrepancyLedger({
      query: ledgerQuery,
      category: ledgerCategory,
    });

    for (const [catId, btn] of ledgerCatButtons.entries()) {
      const active = catId === ledgerCategory;
      btn.classList.toggle('is-active', active);
      btn.setAttribute('aria-pressed', String(active));
    }
    ledgerMatchCountEl.textContent = `${model.filteredCount} / ${model.totalCount} discrepancies`;

    ledgerListWrap.innerHTML = '';
    if (model.rows.length === 0) {
      ledgerListWrap.appendChild(
        el('p', 'df-ins-note df-ins-note--muted', 'No discrepancy entries match the current filter.'),
      );
      return;
    }

    for (const entry of model.rows) {
      const item = el('article', 'df-ledger-item');
      item.dataset.testid = `diffusion-ledger-row-${entry.id}`;
      const head = el('div', 'df-ledger-item-head');
      head.append(
        el('span', 'df-ledger-id', entry.id),
        el('span', 'df-ledger-topic', entry.topic),
        el('span', 'df-ledger-cat-tag', entry.categoryLabel),
      );
      const claimed = el('div', 'df-ledger-field');
      claimed.append(el('strong', undefined, 'Claimed: '), document.createTextNode(entry.claimedOrExpected));
      const truth = el('div', 'df-ledger-field');
      truth.append(el('strong', undefined, 'Verified: '), document.createTextNode(entry.verifiedGroundTruth));
      const res = el('div', 'df-ledger-field df-ledger-res');
      res.append(el('strong', undefined, 'Resolution: '), document.createTextNode(entry.resolution));
      const src = el('div', 'df-ledger-src', `Source: ${entry.sourceRef}`);
      item.append(head, claimed, truth, res, src);
      ledgerListWrap.appendChild(item);
    }
  }

  function syncInspectorUi(): void {
    inspectorDrawer.hidden = !inspectorOpen;
    stage.classList.toggle('is-inspector-open', inspectorOpen);
    inspectorToggleBtn.setAttribute('aria-expanded', String(inspectorOpen));
    inspectorToggleBtn.classList.toggle('is-active', inspectorOpen);

    for (const [tabId, btn] of inspectorTabButtons.entries()) {
      const selected = tabId === inspectorTab;
      btn.classList.toggle('is-active', selected);
      btn.setAttribute('aria-selected', String(selected));
      btn.setAttribute('tabindex', selected ? '0' : '-1');
    }
    inspectorPanel.setAttribute('aria-labelledby', `diffusion-inspector-tab-${inspectorTab}`);

    if (!inspectorOpen) return;

    inspectBody.hidden = inspectorTab !== 'inspect';
    archBody.hidden = inspectorTab !== 'architecture';
    ledgerBody.hidden = inspectorTab !== 'ledger';

    if (inspectorTab === 'inspect') {
      renderInspectTab();
    } else if (inspectorTab === 'architecture') {
      renderArchitectureTab();
    } else {
      if (ledgerSearchInput.value !== ledgerQuery) {
        ledgerSearchInput.value = ledgerQuery;
      }
      renderLedgerList();
    }
  }

  function setInspectorOpen(open: boolean): void {
    if (inspectorOpen === open) return;
    inspectorOpen = open;
    syncInspectorUi();
    markDirtyAndRender();
    announce(open ? `Inspector opened (${inspectorTab})` : 'Inspector closed');
  }

  function setInspectorTab(tab: DiffusionInspectorTabId): void {
    if (inspectorTab === tab && inspectorOpen) return;
    inspectorTab = tab;
    if (!inspectorOpen) inspectorOpen = true;
    syncInspectorUi();
    markDirtyAndRender();
    const label = INSPECTOR_TABS.find((t) => t.id === tab)?.label ?? tab;
    announce(`Inspector tab: ${label}`);
  }

  // ==========================================================================
  // GUIDED TOUR LIFECYCLE (ISOLATED SESSION + EXACT FREE-EXPLORE RESTORE)
  // ==========================================================================

  function applyTourBeat(beat: DiffusionTourBeatSnapshot): void {
    const prevView = focus.viewMode;
    focus = cloneFocusState(beat.focus);
    ropeScrubPosition = beat.ropeScrubPosition;
    resolvedCtx = tour.getTourContext();
    refreshMemoryReport();
    scene = buildDiffusionSpatialScene(resolvedCtx, focus);
    sceneUploadPending = true;
    syncChromeUi();
    if (prevView !== focus.viewMode) {
      fitCameraToScene(true);
    } else {
      markDirtyAndRender();
    }
    announce(`Tour Beat ${beat.beatIndex + 1} of ${beat.totalBeats}: ${beat.title}`);
  }

  function enterGuidedTour(): void {
    if (tour.isActive()) return;
    const feState = session.getState();
    const snapshot: DiffusionFreeExploreRestoreState = {
      activeRecordId: feState.activeRecordId,
      activeBranchId: feState.activeBranchId,
      focus: cloneFocusState(focus),
      cameraPose: cloneCameraPose(cameraPose),
      inspectorTab,
      inspectorOpen,
      ropeScrubPosition,
    };
    pendingRePrefill = null;
    const firstBeat = tour.enter(snapshot);
    applyTourBeat(firstBeat);
  }

  function exitGuidedTour(): void {
    if (!tour.isActive()) return;
    const restored = tour.exit();
    pendingRePrefill = null;
    if (restored) {
      if (restored.activeRecordId) {
        try {
          session.seekRecord(restored.activeRecordId);
        } catch {
          ensureSessionHasActiveRecord(session);
        }
      }
      focus = cloneFocusState(restored.focus);
      inspectorTab = restored.inspectorTab;
      inspectorOpen = restored.inspectorOpen;
      ropeScrubPosition = restored.ropeScrubPosition;
      cameraAnim = null;
      cameraPose = cloneCameraPose(restored.cameraPose);
    }
    resolvedCtx = resolveDiffusionSessionContext(session);
    refreshMemoryReport();
    scene = buildDiffusionSpatialScene(resolvedCtx, focus);
    sceneUploadPending = true;
    syncChromeUi();
    markDirtyAndRender();
    announce('Exited Guided Tour and restored Free-Explore session');
  }

  function syncTourHud(): void {
    const active = tour.isActive();
    tourHud.hidden = !active;
    tourToggleBtn.setAttribute('aria-pressed', String(active));
    tourToggleBtn.classList.toggle('is-active', active);
    tourToggleBtn.textContent = active ? 'Exit Tour' : 'Guided Tour';

    if (!active) return;

    const beat = tour.getCurrentBeat();
    if (!beat) return;

    const acts = tour.getActs();
    for (const act of acts) {
      const btn = tourActButtons.get(act.id);
      if (btn) {
        btn.classList.toggle('is-active', act.isCurrent);
        if (act.isCurrent) {
          btn.setAttribute('aria-current', 'step');
        } else {
          btn.removeAttribute('aria-current');
        }
      }
    }

    tourCounterEl.textContent = `Beat ${beat.beatIndex + 1} / ${beat.totalBeats} · ${beat.actTitle}`;
    tourProvenanceEl.textContent = beat.provenanceSummary;
    tourTitleEl.textContent = beat.title;
    tourSubtitleEl.textContent = beat.subtitle;
    tourNarrativeEl.textContent = beat.narrative;
    tourTakeawayEl.textContent = `Key takeaway: ${beat.takeaway}`;

    tourMetricsEl.innerHTML = '';
    for (const metric of beat.liveMetrics) {
      const chip = el('span', `df-badge df-badge--${metric.tone} df-tour-metric-chip`);
      chip.append(
        el('span', 'df-tour-metric-label', `${metric.label}:`),
        el('strong', 'df-tour-metric-val', ` ${metric.value}`),
      );
      tourMetricsEl.appendChild(chip);
    }

    tourPrevBtn.disabled = beat.beatIndex <= 0;
    tourNextBtn.disabled = beat.beatIndex >= beat.totalBeats - 1;
  }

  // ==========================================================================
  // 2D UNIT-CIRCLE ROPE PHASOR SVG INSET UPDATER
  // ==========================================================================

  function updateRopePhasorInset(): void {
    const show = focus.viewMode === 'gqa_rope';
    phasorCard.hidden = !show;
    if (!show) return;

    const tel: DiffusionRopePhasorTelemetry | null | undefined = scene.ropePhasor;
    const plane = tel ? tel.rotaryPlane : focus.selectedRotaryPlane;
    const invFreqs = tel ? tel.allPlaneInvFreqs : DIFFUSION_DYADIC_YARN_INV_FREQ;
    const zones = tel ? tel.allPlaneZones : DEFAULT_PLANE_ZONES;
    const origMaxPos = tel ? tel.origMaxPos : DIFFUSION_MINI_CONFIG.rope.originalMaxPositionEmbeddings;
    const invFreq = invFreqs[plane] ?? DIFFUSION_DYADIC_YARN_INV_FREQ[0];
    const zone: YarnPlaneZone = zones[plane] ?? DEFAULT_PLANE_ZONES[plane] ?? 'original';

    const livePos = tel ? tel.queryPosition : 0;
    const activePos = ropeScrubPosition !== null ? ropeScrubPosition : livePos;
    posSlider.value = String(Math.max(0, Math.min(31, activePos)));
    scrubberPosLabel.textContent =
      ropeScrubPosition !== null
        ? `p = ${activePos} (scrubbed · live p=${livePos})`
        : `p = ${activePos} (live)`;
    scrubberSyncBtn.disabled = ropeScrubPosition === null;

    for (let i = 0; i < planeButtons.length; i++) {
      const btn = planeButtons[i]!;
      const z = zones[i] ?? DEFAULT_PLANE_ZONES[i] ?? 'original';
      btn.classList.toggle('is-active', i === plane);
      btn.setAttribute('aria-pressed', String(i === plane));
      btn.title = `Plane ${ROTARY_PLANE_LABELS[i]} · inv_freq=${invFreqs[i]} · YaRN zone: ${z}`;
      const freqNode = btn.querySelector('.df-plane-freq');
      if (freqNode) freqNode.textContent = String(invFreqs[i]);
    }

    phasorZoneBadge.textContent = `YaRN: ${zone}`;
    phasorZoneBadge.className = `df-phasor-zone-badge df-zone--${zone}`;

    const qPreNorm = normalizePair(tel ? tel.qPrePair : [1, 0]);
    const kPreNorm = normalizePair(tel ? tel.kPrePair : [0.85, 0.52]);

    let angleRad: number;
    let llama4Scale: number;
    let qRotNorm: readonly [number, number];
    let kRotNorm: readonly [number, number];

    if (ropeScrubPosition === null && tel) {
      angleRad = tel.angleRad;
      llama4Scale = tel.llama4Scale;
      qRotNorm = normalizePair(tel.qRotPair);
      kRotNorm = normalizePair(tel.kRotPair);
    } else {
      angleRad = activePos * invFreq;
      llama4Scale = computeLlama4AttnScale(
        activePos,
        DIFFUSION_MINI_CONFIG.rope.llama4Beta,
        origMaxPos,
      );
      qRotNorm = rotatePair(qPreNorm, angleRad);
      kRotNorm = rotatePair(kPreNorm, angleRad);
    }

    const qPreX = qPreNorm[0] * UNIT_R;
    const qPreY = -qPreNorm[1] * UNIT_R;
    const kPreX = kPreNorm[0] * UNIT_R;
    const kPreY = -kPreNorm[1] * UNIT_R;

    const kRotX = kRotNorm[0] * UNIT_R;
    const kRotY = -kRotNorm[1] * UNIT_R;

    const qScaleR = UNIT_R * llama4Scale;
    const qScaledX = qRotNorm[0] * qScaleR;
    const qScaledY = -qRotNorm[1] * qScaleR;

    phQPreLine.setAttribute('x2', qPreX.toFixed(1));
    phQPreLine.setAttribute('y2', qPreY.toFixed(1));
    phKPreLine.setAttribute('x2', kPreX.toFixed(1));
    phKPreLine.setAttribute('y2', kPreY.toFixed(1));

    phKRotLine.setAttribute('x2', kRotX.toFixed(1));
    phKRotLine.setAttribute('y2', kRotY.toFixed(1));
    phKRotDot.setAttribute('cx', kRotX.toFixed(1));
    phKRotDot.setAttribute('cy', kRotY.toFixed(1));

    phQScaledLine.setAttribute('x2', qScaledX.toFixed(1));
    phQScaledLine.setAttribute('y2', qScaledY.toFixed(1));
    phQScaledDot.setAttribute('cx', qScaledX.toFixed(1));
    phQScaledDot.setAttribute('cy', qScaledY.toFixed(1));
    phQScaledLine.classList.toggle('is-l4-boosted', llama4Scale > 1.0001);
    phQScaledDot.classList.toggle('is-l4-boosted', llama4Scale > 1.0001);

    const arcR = 34;
    const a0 = Math.atan2(qPreNorm[1], qPreNorm[0]);
    const sweep = Math.min(Math.PI * 1.85, Math.abs(angleRad));
    if (sweep > 0.04) {
      const a1 = a0 + (angleRad >= 0 ? sweep : -sweep);
      const sx = Math.cos(a0) * arcR;
      const sy = -Math.sin(a0) * arcR;
      const ex = Math.cos(a1) * arcR;
      const ey = -Math.sin(a1) * arcR;
      const largeArc = sweep > Math.PI ? 1 : 0;
      const sweepFlag = angleRad >= 0 ? 0 : 1;
      phAngleArc.setAttribute(
        'd',
        `M${sx.toFixed(1)},${sy.toFixed(1)}A${arcR},${arcR} 0 ${largeArc},${sweepFlag} ${ex.toFixed(1)},${ey.toFixed(1)}`,
      );
    } else {
      phAngleArc.setAttribute('d', '');
    }

    phKLabel.setAttribute('x', (kRotNorm[0] * (UNIT_R + 14) - 10).toFixed(1));
    phKLabel.setAttribute('y', (-kRotNorm[1] * (UNIT_R + 14) + 3).toFixed(1));
    phQLabel.setAttribute('x', (qRotNorm[0] * (qScaleR + 15) - 16).toFixed(1));
    phQLabel.setAttribute('y', (-qRotNorm[1] * (qScaleR + 15) + 3).toFixed(1));

    const scaleFlag = llama4Scale > 1.0001 ? 'BOOSTED (+6.93%)' : '1.0× (in-window)';
    phasorTelemetryEl.textContent =
      `${formatLayerCode(focus.selectedLayerIndex)} ${formatQHeadCode(focus.selectedQHead)}/KV${focus.selectedKvHead} · ` +
      `plane ${ROTARY_PLANE_LABELS[plane]} · inv_freq=${invFreq} · ` +
      `φ=${angleRad.toFixed(3)} rad · s_L4=${llama4Scale.toFixed(4)} (${scaleFlag})`;
  }

  // ==========================================================================
  // CHROME UI SYNCHRONIZATION
  // ==========================================================================

  function syncChromeUi(): void {
    const tourActive = tour.isActive();
    const activeSession = getDisplayedSession();
    const state = activeSession.getState();
    const rec = resolvedCtx.record;
    const branchRecs = activeSession.getBranchRecords(resolvedCtx.branchId);
    const idxInBranch = branchRecs.findIndex((r) => r.recordId === rec.recordId);

    // 1. View tabs (with WAI-ARIA roving tabindex)
    for (const [mode, btn] of viewButtons.entries()) {
      const isSelected = focus.viewMode === mode;
      btn.setAttribute('aria-selected', String(isSelected));
      btn.setAttribute('tabindex', isSelected ? '0' : '-1');
      btn.classList.toggle('is-active', isSelected);
    }

    // 2. Mode switcher (disabled during Tour so Free-Explore cannot drift)
    const curModeId = activeModePresetId(resolvedCtx.modeAtRecord, resolvedCtx.causalContextAtRecord);
    for (const [id, btn] of modeButtons.entries()) {
      const isActive = id === curModeId;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-pressed', String(isActive));
      btn.disabled = tourActive;
    }

    // 3. Execution buttons (disabled during Tour)
    const isTerminal =
      rec.lifecyclePhase === 'completed_eos' ||
      rec.lifecyclePhase === 'capacity_exhausted' ||
      (rec.lifecyclePhase === 'completed_max_tokens' &&
        rec.generatedTokenIds.length >= state.config.maxNewTokens);
    stepBackBtn.disabled = tourActive || idxInBranch <= 0;
    stepFwdBtn.disabled = tourActive || idxInBranch < 0 || idxInBranch >= branchRecs.length - 1;
    stepBtn.disabled = tourActive || isTerminal;
    runBtn.disabled = tourActive || isTerminal;
    resetBtn.disabled = tourActive;

    // 4. Context-sensitive view controls
    const showLayer =
      focus.viewMode === 'macro' ||
      focus.viewMode === 'gqa_rope' ||
      focus.viewMode === 'swiglu';
    const showGqa = focus.viewMode === 'gqa_rope';
    const showMasks = focus.viewMode === 'mask_explorer';
    const showSwiglu = focus.viewMode === 'swiglu';

    layerWrap.hidden = !showLayer;
    layerSelect.value = String(focus.selectedLayerIndex);
    inspectLayerBtn.hidden = focus.viewMode !== 'macro';
    inspectLayerBtn.textContent = `Inspect ${formatLayerCode(focus.selectedLayerIndex)} GQA →`;

    qHeadWrap.hidden = !showGqa;
    kvHeadWrap.hidden = !showGqa;
    qHeadSelect.value = String(focus.selectedQHead);
    kvHeadSelect.value = String(focus.selectedKvHead);

    // Mask Explorer tabs (with WAI-ARIA roving tabindex)
    maskTabWrap.hidden = !showMasks;
    for (const [tabId, btn] of maskTabButtons.entries()) {
      const isActive = focus.selectedMaskTab === tabId;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-selected', String(isActive));
      btn.setAttribute('tabindex', isActive ? '0' : '-1');
    }

    swigluWrap.hidden = !showSwiglu;
    swigluSlider.value = String(Math.round(focus.swigluAccumReveal * 100));
    swigluReadout.textContent = `${Math.round(focus.swigluAccumReveal * 100)}%`;

    // Stepper bar
    stepperWrap.innerHTML = '';
    if (scene.stepperSteps && scene.stepperSteps.length > 0) {
      stepperWrap.hidden = false;
      for (const st of scene.stepperSteps) {
        const btn = el('button', 'df-seg-btn', st.shortLabel);
        btn.type = 'button';
        btn.title = `${st.title}: ${st.summary}`;
        btn.classList.toggle('is-active', st.index === focus.activeStepperStep);
        btn.addEventListener('click', () => {
          if (focus.activeStepperStep === st.index) return;
          focus = { ...focus, activeStepperStep: st.index };
          rebuildSceneOnly(false);
        });
        stepperWrap.appendChild(btn);
      }
    } else {
      stepperWrap.hidden = true;
    }

    // 5. Provenance badges & session stats
    badgeContainer.innerHTML = '';
    if (tourActive) {
      badgeContainer.appendChild(
        el('span', 'df-badge df-badge--gold', 'Guided Tour (Isolated Session)'),
      );
    }
    for (const badge of scene.provenanceBadges) {
      const pill = el('span', `df-badge df-badge--${badge.tone}`, badge.label);
      badgeContainer.appendChild(pill);
    }

    const totalRecs = state.records.size;
    sessionStatsEl.textContent =
      `Rec #${rec.recordIndex}/${totalRecs} · KV ${resolvedCtx.kvSnapshot.seqLen}/${resolvedCtx.kvSnapshot.capacity} · ` +
      `Gen ${rec.generatedTokenIds.length} tok · ${rec.lifecyclePhase}`;

    // 6. Re-prefill confirmation banner & warning banner
    if (pendingRePrefill && !tourActive) {
      rePrefillBanner.hidden = false;
      rePrefillText.textContent =
        `KV provenance guard (${pendingRePrefill.preset.label}): ${pendingRePrefill.errorMessage}`;
    } else {
      rePrefillBanner.hidden = true;
    }

    if (scene.warningBanner) {
      warningStrip.hidden = false;
      warningStrip.textContent = scene.warningBanner;
    } else {
      warningStrip.hidden = true;
    }

    // 7. View title caption
    viewTitleEl.textContent = scene.title;
    viewSubtitleEl.textContent = scene.subtitle;
    viewDisclosureEl.hidden = focus.viewMode !== 'gqa_rope' && focus.viewMode !== 'macro';

    // 8. Tour HUD, RoPE Phasor inset, Inspector drawer & Unified Timeline Dock
    syncTourHud();
    updateRopePhasorInset();
    syncInspectorUi();
    timelineDock.update(
      buildDiffusionTimelineDockState(activeSession, { isReadOnly: tourActive }),
    );

    // 9. Docked Formula Inspector bar
    const f = scene.activeFormula;
    if (f) {
      formulaTensorEl.textContent = f.tensorLabel;
      formulaCoordEl.textContent = f.coordLabel;
      formulaBadgeEl.textContent = f.headBadge ?? '';
      formulaBadgeEl.hidden = !f.headBadge;
      formulaEqEl.textContent = f.equation;
      formulaSubEl.textContent = f.subEquation;
      formulaValEl.textContent = f.valueText;
      formulaValEl.classList.toggle('is-masked', f.isMasked);
    } else if (focus.selectedPartId && scene.partCatalog[focus.selectedPartId]) {
      const part = scene.partCatalog[focus.selectedPartId]!;
      formulaTensorEl.textContent = part.title;
      formulaCoordEl.textContent = part.id;
      formulaBadgeEl.textContent = part.badge;
      formulaBadgeEl.hidden = !part.badge;
      formulaEqEl.textContent = part.specs.join(' · ') || part.summary;
      formulaSubEl.textContent = part.summary;
      formulaValEl.textContent = '';
      formulaValEl.classList.remove('is-masked');
    } else {
      formulaTensorEl.textContent = scene.title;
      formulaCoordEl.textContent =
        scene.layerIndex !== null ? formatLayerCode(scene.layerIndex) : '40L Spine';
      formulaBadgeEl.textContent = resolvedCtx.loraActiveOnDraft ? 'LoRA Draft ON' : 'Base Backbone';
      formulaBadgeEl.hidden = false;
      formulaEqEl.textContent = scene.subtitle;
      formulaSubEl.textContent =
        'Drag to orbit · Shift+drag or right-drag to pan · Scroll to zoom · Click sheets to inspect';
      formulaValEl.textContent = '';
      formulaValEl.classList.remove('is-masked');
    }

    const toks = resolvedCtx.kvSnapshot.committedTokenIds;
    const previewCount = Math.min(toks.length, 10);
    const tail = toks.slice(toks.length - previewCount).map((id) => DIFFUSION_MINI_VOCAB[id] ?? `#${id}`);
    tokenPreviewEl.textContent =
      toks.length > previewCount ? `… ${tail.join(' ')}` : tail.join(' ');
    tokenPreviewEl.title = toks.map((id) => DIFFUSION_MINI_VOCAB[id] ?? `#${id}`).join(' ');
  }

  // ==========================================================================
  // SCENE REBUILD & MODE SWITCH HANDLERS
  // ==========================================================================

  function rebuildFromSession(refitCamera: boolean): void {
    resolvedCtx = tour.isActive()
      ? tour.getTourContext()
      : resolveDiffusionSessionContext(session);
    refreshMemoryReport();
    scene = buildDiffusionSpatialScene(resolvedCtx, focus);
    sceneUploadPending = true;
    syncChromeUi();
    if (refitCamera) {
      fitCameraToScene(true);
    } else {
      markDirtyAndRender();
    }
  }

  function rebuildSceneOnly(refitCamera: boolean): void {
    scene = buildDiffusionSpatialScene(resolvedCtx, focus);
    sceneUploadPending = true;
    syncChromeUi();
    if (refitCamera) {
      fitCameraToScene(true);
    } else {
      markDirtyAndRender();
    }
  }

  function setViewMode(mode: DiffusionSpatialViewMode): void {
    if (focus.viewMode === mode) {
      fitCameraToScene(true);
      return;
    }
    focus = {
      ...focus,
      viewMode: mode,
      selectedPartId: null,
      hoveredPartId: null,
      hoveredCell: null,
      pinnedCell: null,
      activeStepperStep: 0,
    };
    rebuildSceneOnly(true);
    const desc = DIFFUSION_SPATIAL_VIEW_MODES.find((d) => d.id === mode);
    announce(`Spatial view: ${desc?.label ?? mode}`);
  }

  function handleModeSwitch(preset: ModePreset): void {
    if (tour.isActive()) return;
    pendingRePrefill = null;
    try {
      session.updateConfig(preset.patch);
      ropeScrubPosition = null;
      rebuildFromSession(false);
      announce(`Execution mode switched to ${preset.label}`);
    } catch (err) {
      if (err instanceof KvProvenanceMismatchError) {
        pendingRePrefill = {
          preset,
          errorMessage: err.message,
        };
        syncChromeUi();
      } else {
        pendingRePrefill = {
          preset,
          errorMessage: err instanceof Error ? err.message : String(err),
        };
        syncChromeUi();
      }
    }
  }

  // ==========================================================================
  // GLOBAL KEYBOARD HANDLER (WAI-ARIA + TOUR + ESCAPE)
  // ==========================================================================

  const onWindowKeydown = (ev: KeyboardEvent) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const action = resolveDiffusionGlobalKeyAction(ev.key, {
      target: ev.target,
      tourActive: tour.isActive(),
      inspectorOpen,
      hasPinnedCell: focus.pinnedCell !== null,
    });
    switch (action) {
      case 'tour_next': {
        ev.preventDefault();
        const beat = tour.next();
        if (beat) applyTourBeat(beat);
        break;
      }
      case 'tour_prev': {
        ev.preventDefault();
        const beat = tour.prev();
        if (beat) applyTourBeat(beat);
        break;
      }
      case 'close_inspector': {
        ev.preventDefault();
        setInspectorOpen(false);
        break;
      }
      case 'clear_pin': {
        ev.preventDefault();
        focus = { ...focus, selectedPartId: null, pinnedCell: null };
        rebuildSceneOnly(false);
        announce('Cleared pinned inspection cell');
        break;
      }
      case 'exit_tour': {
        ev.preventDefault();
        exitGuidedTour();
        break;
      }
      case 'none':
        break;
    }
  };
  window.addEventListener('keydown', onWindowKeydown);

  // ==========================================================================
  // POINTER INTERACTIONS (ORBIT / PAN / ZOOM / HOVER / CLICK / DBLCLICK PICKING)
  // ==========================================================================

  function pickAtClient(clientX: number, clientY: number): DiffusionPickIdentity | null {
    if (!lastSnapshot) return null;
    const ray = clientPointToRay(clientX, clientY, lastSnapshot);
    const hit = pickMacroBay(ray, scene.solidPickBoxes, scene.fasciaPickBoxes);
    return pickHitToDiffusionCell(hit, focus.viewMode);
  }

  const onPointerDown = (ev: PointerEvent) => {
    if (ev.button !== 0 && ev.button !== 1 && ev.button !== 2) return;
    pointerDown = true;
    pointerMoved = false;
    pointerMode = ev.button === 2 || ev.button === 1 || ev.shiftKey ? 'pan' : 'orbit';
    pointerStartX = ev.clientX;
    pointerStartY = ev.clientY;
    pointerLastX = ev.clientX;
    pointerLastY = ev.clientY;
    cameraAnim = null;
    canvas.setPointerCapture?.(ev.pointerId);
    canvas.classList.add('is-dragging');
  };

  const onPointerMove = (ev: PointerEvent) => {
    if (pointerDown) {
      const dx = ev.clientX - pointerLastX;
      const dy = ev.clientY - pointerLastY;
      pointerLastX = ev.clientX;
      pointerLastY = ev.clientY;
      if (
        !pointerMoved &&
        Math.hypot(ev.clientX - pointerStartX, ev.clientY - pointerStartY) >= DRAG_THRESHOLD_PX
      ) {
        pointerMoved = true;
      }
      if (pointerMode === 'pan' && lastSnapshot) {
        cameraPose = panCameraByCssDelta(cameraPose, lastSnapshot, dx, dy);
      } else {
        cameraPose = orbitCameraByCssDelta(cameraPose, dx, dy);
      }
      markDirtyAndRender();
      return;
    }

    const picked = pickAtClient(ev.clientX, ev.clientY);
    const nextHoverPart = picked?.partId ?? null;
    if (nextHoverPart === focus.hoveredPartId && samePickIdentity(picked, focus.hoveredCell)) {
      return;
    }
    canvas.style.cursor = picked ? 'pointer' : 'grab';
    focus = {
      ...focus,
      hoveredPartId: nextHoverPart,
      hoveredCell: picked,
    };
    rebuildSceneOnly(false);
  };

  const onPointerUp = (ev: PointerEvent) => {
    if (!pointerDown) return;
    pointerDown = false;
    canvas.classList.remove('is-dragging');
    try {
      canvas.releasePointerCapture?.(ev.pointerId);
    } catch {
      // ignore if capture was already released
    }

    if (!pointerMoved && ev.button === 0) {
      const picked = pickAtClient(ev.clientX, ev.clientY);
      if (picked) {
        const nextLayer =
          picked.layerIndex !== null ? picked.layerIndex : focus.selectedLayerIndex;
        const nextQ =
          picked.qHeadIndex !== null ? picked.qHeadIndex : focus.selectedQHead;
        const nextKv =
          picked.kvHeadIndex !== null
            ? picked.kvHeadIndex
            : picked.qHeadIndex !== null
              ? Math.floor(picked.qHeadIndex / DIFFUSION_MINI_CONFIG.kvGroupSize)
              : focus.selectedKvHead;
        const nextPlane =
          picked.rotaryPlane !== null ? picked.rotaryPlane : focus.selectedRotaryPlane;
        const nextTokenRow =
          picked.tokenIndex !== null ? picked.tokenIndex : focus.selectedTokenRow;

        focus = {
          ...focus,
          selectedLayerIndex: nextLayer,
          selectedQHead: nextQ,
          selectedKvHead: nextKv,
          selectedRotaryPlane: nextPlane,
          selectedTokenRow: nextTokenRow,
          selectedPartId: picked.partId,
          pinnedCell: picked,
        };
      } else {
        focus = {
          ...focus,
          selectedPartId: null,
          pinnedCell: null,
        };
      }
      rebuildSceneOnly(false);
    }
  };

  const onPointerLeave = () => {
    if (pointerDown) return;
    if (focus.hoveredPartId !== null || focus.hoveredCell !== null) {
      focus = { ...focus, hoveredPartId: null, hoveredCell: null };
      rebuildSceneOnly(false);
    }
  };

  const onDoubleClick = (ev: MouseEvent) => {
    if (focus.viewMode !== 'macro') return;
    const picked = pickAtClient(ev.clientX, ev.clientY);
    if (picked && picked.layerIndex !== null) {
      focus = { ...focus, selectedLayerIndex: picked.layerIndex };
      setViewMode('gqa_rope');
    }
  };

  const onWheel = (ev: WheelEvent) => {
    ev.preventDefault();
    cameraAnim = null;
    const scaleFactor = ev.deltaY > 0 ? 1.09 : 1 / 1.09;
    cameraPose = zoomPerspectiveDistance(cameraPose, scaleFactor);
    markDirtyAndRender();
  };

  const onContextMenu = (ev: MouseEvent) => {
    ev.preventDefault();
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('pointerleave', onPointerLeave);
  canvas.addEventListener('dblclick', onDoubleClick);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', onContextMenu);

  let resizeObserver: ResizeObserver | null = null;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => {
      markDirtyAndRender();
    });
    resizeObserver.observe(stage);
  }
  const onWindowResize = () => markDirtyAndRender();
  window.addEventListener('resize', onWindowResize);

  // ==========================================================================
  // READ-ONLY WINDOW DIAGNOSTIC HANDLE
  // ==========================================================================

  function getDebugSnapshot(): DiffusionSpatialDebugHandle {
    const activeSession = getDisplayedSession();
    const state = activeSession.getState();
    const tourActive = tour.isActive();
    return {
      viewMode: focus.viewMode,
      maskTab: focus.selectedMaskTab,
      layerIndex: focus.selectedLayerIndex,
      glErrorCount,
      lastGlError,
      frameCount,
      atlasUploadCount,
      atlasWidth,
      atlasHeight,
      atlasKey: scene.atlasKey ?? null,
      recordCount: state.records.size,
      kvSeqLen: resolvedCtx.kvSnapshot.seqLen,
      mode: resolvedCtx.modeAtRecord,
      loraActiveOnDraft: resolvedCtx.loraActiveOnDraft,
      rafActive,
      tourActive,
      tourBeatIndex: tourActive ? tour.getCurrentBeatIndex() : null,
      tourTotalBeats: tour.getBeatCount(),
      inspectorTab,
      inspectorOpen,
      branchCount: state.branches.length,
      activeBranchId: state.activeBranchId,
      activeRecordId: state.activeRecordId,
      retainedBytes: cachedMemoryReport.totalRetainedBytes,
    };
  }

  const liveDebugHandle: DiffusionSpatialDebugHandle = {
    get viewMode() {
      return focus.viewMode;
    },
    get maskTab() {
      return focus.selectedMaskTab;
    },
    get layerIndex() {
      return focus.selectedLayerIndex;
    },
    get glErrorCount() {
      return glErrorCount;
    },
    get lastGlError() {
      return lastGlError;
    },
    get frameCount() {
      return frameCount;
    },
    get atlasUploadCount() {
      return atlasUploadCount;
    },
    get atlasWidth() {
      return atlasWidth;
    },
    get atlasHeight() {
      return atlasHeight;
    },
    get atlasKey() {
      return scene.atlasKey ?? null;
    },
    get recordCount() {
      return getDisplayedSession().getState().records.size;
    },
    get kvSeqLen() {
      return resolvedCtx.kvSnapshot.seqLen;
    },
    get mode() {
      return resolvedCtx.modeAtRecord;
    },
    get loraActiveOnDraft() {
      return resolvedCtx.loraActiveOnDraft;
    },
    get rafActive() {
      return rafActive;
    },
    get tourActive() {
      return tour.isActive();
    },
    get tourBeatIndex() {
      return tour.isActive() ? tour.getCurrentBeatIndex() : null;
    },
    get tourTotalBeats() {
      return tour.getBeatCount();
    },
    get inspectorTab() {
      return inspectorTab;
    },
    get inspectorOpen() {
      return inspectorOpen;
    },
    get branchCount() {
      return getDisplayedSession().getState().branches.length;
    },
    get activeBranchId() {
      return getDisplayedSession().getState().activeBranchId;
    },
    get activeRecordId() {
      return getDisplayedSession().getState().activeRecordId;
    },
    get retainedBytes() {
      return cachedMemoryReport.totalRetainedBytes;
    },
  };

  if (typeof window !== 'undefined') {
    (window as unknown as DiffusionWindowHost).__diffusionSpatialDebug = liveDebugHandle;
  }

  // Initial chrome sync, camera fit, and single on-demand render frame
  syncChromeUi();
  fitCameraToScene(false);

  return {
    setViewMode,
    getDebugSnapshot,
    dispose(): void {
      if (disposed) return;
      disposed = true;
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
        rafId = null;
        rafActive = false;
      }
      resizeObserver?.disconnect();
      window.removeEventListener('resize', onWindowResize);
      window.removeEventListener('keydown', onWindowKeydown);
      canvas.removeEventListener('webglcontextlost', onContextLost);
      canvas.removeEventListener('webglcontextrestored', onContextRestored);
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      canvas.removeEventListener('dblclick', onDoubleClick);
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('contextmenu', onContextMenu);
      timelineDock.dispose();
      modelSwitcher.destroy();
      solidsPass?.dispose();
      linesPass?.dispose();
      ribbonsPass?.dispose();
      flatFlowPass?.dispose();
      sheetsPass?.dispose();
      device?.dispose();
      if (typeof window !== 'undefined') {
        const host = window as unknown as DiffusionWindowHost;
        if (host.__diffusionSpatialDebug === liveDebugHandle) {
          delete host.__diffusionSpatialDebug;
        }
      }
    },
  };
}
