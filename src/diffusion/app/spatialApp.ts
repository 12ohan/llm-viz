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
 *
 * Pure deterministic implementation with zero sibling-visualizer dependencies.
 */

import { CAMERA_FLY_DURATION_MS, DEFAULT_FOV_DEG, type CameraPose } from '../../model/state';
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
} from '../../render/camera';
import { RenderDevice } from '../../render/device';
import { LinesPass } from '../../render/lines';
import { clientPointToRay, pickMacroBay } from '../../render/pick';
import { FlatFlowRibbonsPass, RibbonsPass } from '../../render/ribbons';
import { SheetsPass } from '../../render/sheets';
import { SolidsPass } from '../../render/solids';
import type { OverlayRect } from '../../ui/overlay';
import { KvProvenanceMismatchError } from '../kvCache';
import { createDefaultDiffusionModel } from '../parameters';
import { computeLlama4AttnScale } from '../rope';
import { createDiffusionSession } from '../session';
import {
  DIFFUSION_DYADIC_YARN_INV_FREQ,
  DIFFUSION_MINI_CONFIG,
  DIFFUSION_MINI_VOCAB,
  type DiffusionExecutionMode,
  type DiffusionExplorerMaskMode,
  type DiffusionSessionConfigPatch,
  type YarnPlaneZone,
} from '../types';
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
import { DiffusionSpatialOverlay } from './spatialOverlay';

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

  let focus: DiffusionFocusState = { ...DEFAULT_DIFFUSION_FOCUS_STATE };
  let resolvedCtx: DiffusionResolvedSessionContext = resolveDiffusionSessionContext(session);
  let scene: DiffusionSpatialScene = buildDiffusionSpatialScene(resolvedCtx, focus);

  // Scrubber override for 2D RoPE phasor position (`null` = follow live record queryPosition)
  let ropeScrubPosition: number | null = null;

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

  const prefersReducedMotion =
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

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

  // ==========================================================================
  // BUILD EDITORIAL DRAFTING-TABLE SHELL DOM
  // ==========================================================================

  const shell = el('div', 'df-shell');
  root.appendChild(shell);

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
  for (const desc of DIFFUSION_SPATIAL_VIEW_MODES) {
    const btn = el('button', 'df-view-tab', desc.shortLabel);
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.title = `${desc.label} — ${desc.description}`;
    btn.dataset.testid = `diffusion-view-${desc.id}`;
    btn.addEventListener('click', () => setViewMode(desc.id));
    viewNav.appendChild(btn);
    viewButtons.set(desc.id, btn);
  }

  const topActions = el('div', 'df-top-actions');
  const resetCamBtn = el('button', 'df-top-btn', 'Fit View');
  resetCamBtn.type = 'button';
  resetCamBtn.title = 'Re-fit perspective camera to the active view bounds';
  resetCamBtn.dataset.testid = 'diffusion-camera-reset-btn';
  resetCamBtn.addEventListener('click', () => fitCameraToScene(true));
  topActions.append(resetCamBtn);

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
    const prev = session.stepBack();
    if (prev) {
      pendingRePrefill = null;
      ropeScrubPosition = null;
      rebuildFromSession(false);
    }
  });

  const stepFwdBtn = el('button', 'df-btn', 'Fwd ▶');
  stepFwdBtn.type = 'button';
  stepFwdBtn.title = 'Step forward along active branch';
  stepFwdBtn.dataset.testid = 'diffusion-step-forward-btn';
  stepFwdBtn.addEventListener('click', () => {
    const next = session.stepForward();
    if (next) {
      pendingRePrefill = null;
      ropeScrubPosition = null;
      rebuildFromSession(false);
    }
  });

  const stepBtn = el('button', 'df-btn df-btn--primary', 'Step');
  stepBtn.type = 'button';
  stepBtn.title = 'Execute one generation step / block / speculative cycle';
  stepBtn.dataset.testid = 'diffusion-step-btn';
  stepBtn.addEventListener('click', () => {
    pendingRePrefill = null;
    ropeScrubPosition = null;
    session.step();
    rebuildFromSession(false);
  });

  const runBtn = el('button', 'df-btn df-btn--accent', 'Run');
  runBtn.type = 'button';
  runBtn.title = 'Run session deterministically to completion / EOS / capacity';
  runBtn.dataset.testid = 'diffusion-run-btn';
  runBtn.addEventListener('click', () => {
    pendingRePrefill = null;
    ropeScrubPosition = null;
    session.runToCompletion();
    rebuildFromSession(false);
  });

  const resetBtn = el('button', 'df-btn', 'Reset');
  resetBtn.type = 'button';
  resetBtn.title = 'Reset session back to initial prompt prefill';
  resetBtn.dataset.testid = 'diffusion-reset-btn';
  resetBtn.addEventListener('click', () => {
    pendingRePrefill = null;
    ropeScrubPosition = null;
    session.reset();
    ensureSessionHasActiveRecord(session);
    rebuildFromSession(false);
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

  // Mask Explorer 5-tab bar
  const maskTabWrap = el('div', 'df-segmented df-mask-tabs');
  maskTabWrap.setAttribute('role', 'tablist');
  maskTabWrap.setAttribute('aria-label', 'Attention mask explorer modes');
  const maskTabButtons = new Map<DiffusionExplorerMaskMode, HTMLButtonElement>();
  for (const tab of MASK_TABS) {
    const btn = el('button', 'df-seg-btn', tab.label);
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.title = tab.title;
    btn.dataset.testid = `diffusion-mask-tab-${tab.id}`;
    btn.addEventListener('click', () => {
      if (focus.selectedMaskTab === tab.id) return;
      focus = { ...focus, selectedMaskTab: tab.id, pinnedCell: null, selectedPartId: null };
      rebuildSceneOnly(false);
    });
    maskTabWrap.appendChild(btn);
    maskTabButtons.set(tab.id, btn);
  }

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
    if (!pendingRePrefill) return;
    const targetPatch = pendingRePrefill.preset.patch;
    pendingRePrefill = null;
    ropeScrubPosition = null;
    session.updateConfig({ ...targetPatch, allowExplicitRePrefill: true });
    rebuildFromSession(false);
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

  // 4. Main WebGL2 Viewport Stage + Projected Overlay + 2D Unit-Circle RoPE Phasor SVG Inset
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

  // Interactive 2D Unit-Circle RoPE Phasor SVG Inset (visible in `gqa_rope`)
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

  // Crisp SVG unit circle (`r = 1.0`) + outer `r = 1.069` dashed Llama-4 ring
  // Coordinate system: viewBox -140 -140 280 280; unit circle radius R = 96 px
  const UNIT_R = 96;
  const LLAMA4_R = UNIT_R * (1 + 0.1 * Math.log(2)); // ~102.65 px (r = 1.0693)

  const phasorSvg = svgEl('svg', {
    class: 'df-phasor-svg',
    viewBox: '-136 -136 272 272',
    role: 'img',
    'aria-label': '2D RoPE Unit Circle and Llama-4 Query Scale Phasor Diagram',
  });

  // Grid & axes
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

  // Dynamic vector elements
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

  // Position scrubber 0..31 with visible `p = 16 · Llama-4 scale boundary` tick
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

  stage.append(canvas, overlaySvg, overlayLabels, viewCaption, phasorCard, webglFallbackEl);
  shell.appendChild(stage);

  // 5. Docked Formula Inspector Bar (`activeFormula` + selected part + token stream)
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
    if (!cameraInitialized || !animate || prefersReducedMotion) {
      cameraPose = targetPose;
      cameraAnim = null;
      cameraInitialized = true;
    } else {
      cameraAnim = {
        from: {
          target: [cameraPose.target[0], cameraPose.target[1], cameraPose.target[2]],
          azimuthDeg: cameraPose.azimuthDeg,
          elevationDeg: cameraPose.elevationDeg,
          distance: cameraPose.distance,
          fovDeg: cameraPose.fovDeg,
        },
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
    if (!phasorCard.hidden) {
      const r = phasorCard.getBoundingClientRect();
      if (r.width > 0 && r.height > 0) {
        rects.push({
          x0: r.left - stageRect.left - 6,
          y0: r.top - stageRect.top - 6,
          x1: r.right - stageRect.left + 6,
          y1: r.bottom - stageRect.top + 6,
        });
      }
    }
    const capRect = viewCaption.getBoundingClientRect();
    if (capRect.width > 0 && capRect.height > 0) {
      rects.push({
        x0: capRect.left - stageRect.left - 4,
        y0: capRect.top - stageRect.top - 4,
        x1: capRect.right - stageRect.left + 4,
        y1: capRect.bottom - stageRect.top + 4,
      });
    }
    return rects;
  }

  function renderFrame(nowMs: number): void {
    if (disposed) return;

    // Advance camera animation if active
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

    // Synchronous pooled DOM + SVG overlay projection inside render pass
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

    // Update rotary plane selector buttons
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

    // Compute exact normalized vectors on the unit circle + Llama-4 outer ring
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

    // SVG Y axis points downward, so negate Y for standard mathematical unit circle
    const qPreX = qPreNorm[0] * UNIT_R;
    const qPreY = -qPreNorm[1] * UNIT_R;
    const kPreX = kPreNorm[0] * UNIT_R;
    const kPreY = -kPreNorm[1] * UNIT_R;

    const kRotX = kRotNorm[0] * UNIT_R;
    const kRotY = -kRotNorm[1] * UNIT_R;

    // Scaled query visibly pops out to r = llama4Scale * UNIT_R (~1.0693 * UNIT_R for p >= 16)
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

    // Subtle rotation arc between q_pre and q_rot
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
      const sweepFlag = angleRad >= 0 ? 0 : 1; // CCW in math = 0 in SVG
      phAngleArc.setAttribute(
        'd',
        `M${sx.toFixed(1)},${sy.toFixed(1)}A${arcR},${arcR} 0 ${largeArc},${sweepFlag} ${ex.toFixed(1)},${ey.toFixed(1)}`,
      );
    } else {
      phAngleArc.setAttribute('d', '');
    }

    // Position vector tip labels cleanly outward
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
    const state = session.getState();
    const rec = resolvedCtx.record;
    const branchRecs = session.getBranchRecords(resolvedCtx.branchId);
    const idxInBranch = branchRecs.findIndex((r) => r.recordId === rec.recordId);

    // 1. View tabs
    for (const [mode, btn] of viewButtons.entries()) {
      const isSelected = focus.viewMode === mode;
      btn.setAttribute('aria-selected', String(isSelected));
      btn.classList.toggle('is-active', isSelected);
    }

    // 2. Mode switcher
    const curModeId = activeModePresetId(state.config.mode, state.config.causalContext);
    for (const [id, btn] of modeButtons.entries()) {
      const isActive = id === curModeId;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-pressed', String(isActive));
    }

    // 3. Execution buttons
    const isTerminal =
      rec.lifecyclePhase === 'completed_eos' ||
      rec.lifecyclePhase === 'capacity_exhausted' ||
      (rec.lifecyclePhase === 'completed_max_tokens' &&
        rec.generatedTokenIds.length >= state.config.maxNewTokens);
    stepBackBtn.disabled = idxInBranch <= 0;
    stepFwdBtn.disabled = idxInBranch < 0 || idxInBranch >= branchRecs.length - 1;
    stepBtn.disabled = isTerminal;
    runBtn.disabled = isTerminal;

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

    maskTabWrap.hidden = !showMasks;
    for (const [tabId, btn] of maskTabButtons.entries()) {
      const isActive = focus.selectedMaskTab === tabId;
      btn.classList.toggle('is-active', isActive);
      btn.setAttribute('aria-selected', String(isActive));
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
    for (const badge of scene.provenanceBadges) {
      const pill = el('span', `df-badge df-badge--${badge.tone}`, badge.label);
      badgeContainer.appendChild(pill);
    }

    const totalRecs = state.records.size;
    sessionStatsEl.textContent =
      `Rec #${rec.recordIndex}/${totalRecs} · KV ${resolvedCtx.kvSnapshot.seqLen}/${resolvedCtx.kvSnapshot.capacity} · ` +
      `Gen ${rec.generatedTokenIds.length} tok · ${rec.lifecyclePhase}`;

    // 6. Re-prefill confirmation banner & warning banner
    if (pendingRePrefill) {
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

    // 8. 2D RoPE Phasor SVG inset
    updateRopePhasorInset();

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
      formulaSubEl.textContent = 'Drag to orbit · Shift+drag or right-drag to pan · Scroll to zoom · Click sheets to inspect';
      formulaValEl.textContent = '';
      formulaValEl.classList.remove('is-masked');
    }

    // Token stream preview
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
    resolvedCtx = resolveDiffusionSessionContext(session);
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
  }

  function handleModeSwitch(preset: ModePreset): void {
    pendingRePrefill = null;
    try {
      session.updateConfig(preset.patch);
      ropeScrubPosition = null;
      rebuildFromSession(false);
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

    // Hover ray picking with strict identity deduplication
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

  // Resize observer (only re-renders on actual container size change)
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
    const state = session.getState();
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
      return session.getState().records.size;
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
