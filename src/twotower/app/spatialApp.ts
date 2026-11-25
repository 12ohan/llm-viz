/**
 * Nemotron-Labs-TwoTower Stage TT3 Interactive WebGL2 Spatial Application
 * (`src/twotower/app/spatialApp.ts`)
 *
 * Directly composes:
 *   - Nano's 5 WebGL2 render passes (`RenderDevice`, `SolidsPass`, `SheetsPass`, `LinesPass`,
 *     `RibbonsPass`, `FlatFlowRibbonsPass`) on a studio-white `#F7F8FA` editorial drafting table
 *   - All 7 Stage TT3 headless TwoTower spatial scene builders (`src/twotower/scene/index.ts`)
 *   - Projected editorial overlay (`TwoTowerSpatialOverlay` reusing `src/ui/overlay.ts`)
 *   - Retained Developer Diagnostic Workbench inside a collapsible drawer synced bidirectionally
 *     with the live `TwoTowerSession`.
 *
 * ZERO runtime dependencies. ZERO Nano files modified. ZERO cross-app link back to Nano.
 */

import { inferGqaCellHead, type GqaCellRef } from '../../math/gqaSim';
import {
  formatMambaHeadId,
  inferMambaCellHead,
  type MambaCellRef,
} from '../../math/mambaSim';
import {
  formatExpertId,
  inferMoeCellExpert,
  MOE_SIM_CONFIG,
  type MoeCellRef,
} from '../../math/moeSim';
import { formatBayId } from '../../model/spec';
import {
  CAMERA_FLY_DURATION_MS,
  cloneCameraPose,
  DEFAULT_FOV_DEG,
  type CameraPose,
} from '../../model/state';
import {
  buildCameraSnapshot,
  fitOpBoundsCamera,
  interpolateCameraPose,
  makeViewportMetrics,
  orbitCameraByCssDelta,
  panCameraByCssDelta,
  projectWorldToClientCss,
  zoomPerspectiveDistance,
  type AABB,
  type CameraSnapshot,
  type ViewportMetrics,
} from '../../render/camera';
import { RenderDevice } from '../../render/device';
import { LinesPass } from '../../render/lines';
import {
  clientPointToRay,
  pickHitToGqaCell,
  pickHitToMambaCell,
  pickHitToMoeCell,
  pickMacroBay,
  type PickHit,
} from '../../render/pick';
import { FlatFlowRibbonsPass, RibbonsPass } from '../../render/ribbons';
import { SheetsPass } from '../../render/sheets';
import { SolidsPass } from '../../render/solids';
import type { EmbeddingCellRef } from '../../scene/embeddingCutaway';
import type { OutputCellRef } from '../../scene/outputCutaway';
import {
  commitTwoTowerActiveBlockToContext,
  commitTwoTowerRefinementConfidence,
  createDefaultTwoTowerModel,
  createTwoTowerSession,
  forkTwoTowerSessionBranch,
  getActiveBranch,
  getActiveRecord,
  getSessionCapabilities,
  initTwoTowerDiffusionBlock,
  measureTwoTowerSessionRetainedBytes,
  navigateTwoTowerSessionBack,
  navigateTwoTowerSessionForward,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  stepTwoTowerSessionNext,
  switchTwoTowerSessionBranch,
  TWOTOWER_MINI_CONFIG,
  TWOTOWER_SPEC,
  updateTwoTowerSessionConfig,
  type TwoTowerExecutionMode,
  type TwoTowerSession,
  type TwoTowerSessionCapabilities,
  type TwoTowerSessionRecord,
} from '../index';
import {
  buildTwoTowerEmbeddingScene,
  buildTwoTowerMacroScene,
  buildTwoTowerMoeScene,
  buildTwoTowerOutputScene,
  buildTwoTowerPairedGqaScene,
  buildTwoTowerPairedMambaScene,
  buildTwoTowerTimestepScene,
  contextBayFrontWorld,
  denoiserBayFrontWorld,
  fitPairedGqaCamera,
  fitPairedMambaCamera,
  fitSpatialEmbeddingCamera,
  fitSpatialMoeCamera,
  fitSpatialOutputCamera,
  fitTimestepCamera,
  fitTwoTowerMacroCamera,
  getTwoTowerBayKind,
  parseOutputCandidatePartId,
  pickHitToEmbeddingCell,
  pickHitToOutputCell,
  pickHitToTimestepCell,
  resolveTwoTowerSessionViewState,
  TWO_TOWER_MACRO_FIT_BOUNDS,
  type PairedGqaDiagnostics,
  type PairedMambaDiagnostics,
  type TimestepCellRef,
  type TimestepCutawayDiagnostics,
  type TwoTowerEmbeddingDiagnostics,
  type TwoTowerLinkerChannel,
  type TwoTowerMacroDiagnostics,
  type TwoTowerMoeDiagnostics,
  type TwoTowerOutputDiagnostics,
  type TwoTowerPairedFocus,
  type TwoTowerSelectionIdentity,
  type TwoTowerSpatialScene,
  type TwoTowerSpatialViewMode,
  type TwoTowerTowerRole,
} from '../scene/index';
import {
  mountTwoTowerDiagnosticWorkbench,
  type DiagnosticWorkbenchController,
} from './diagnosticUi';
import { TwoTowerSpatialOverlay } from './spatialOverlay';

// ============================================================================
// PUBLIC TELEMETRY & INSPECTION CONTRACTS
// ============================================================================

export interface TwoTowerTelemetrySnapshot {
  readonly viewMode: TwoTowerSpatialViewMode;
  readonly drawCallsLastFrame: number;
  readonly lastFrameMs: number;
  readonly avgFrameMs: number;
  readonly atlasUploads: number;
  readonly currentAtlasDims: readonly [number, number] | null;
  readonly maxTextureSize: number;
  readonly retainedSessionBytes: number;
  readonly sharedWeightBytes: number;
  readonly solidCount: number;
  readonly sheetCount: number;
  readonly lineCount: number;
  readonly ribbonCount: number;
  readonly flowTriangleCount: number;
  readonly lifecyclePhase: TwoTowerSessionRecord['lifecyclePhase'];
  readonly committedPrefixLen: number;
  readonly kvCapacity: number;
  readonly recordIndex: number;
  readonly branchId: string;
  readonly glErrorCount: number;
  readonly reducedMotion: boolean;
  readonly rafFramesRendered: number;
}

export interface TwoTowerAppHandle {
  getState(): {
    readonly viewMode: TwoTowerSpatialViewMode;
    readonly selectedBayId: number | null;
    readonly selectedTower: TwoTowerTowerRole;
    readonly towerFocus: TwoTowerPairedFocus;
    readonly linkerChannel: TwoTowerLinkerChannel;
    readonly stepperStep: number | null;
    readonly stepperPlaying: boolean;
    readonly stepperStepProgress: number;
    readonly stepperTermProgress: number;
    readonly reducedMotion: boolean;
    readonly workbenchOpen: boolean;
    readonly activeSelection: TwoTowerSelectionIdentity | null;
    readonly warningBanner: string | null;
    readonly diagnostics: Readonly<Record<string, unknown>> | undefined;
  };
  setViewMode(mode: TwoTowerSpatialViewMode): void;
  setStepperStep(step: number | null): void;
  setStepperTermProgress(progress: number): void;
  selectBay(bayId: number | null, tower?: TwoTowerTowerRole): void;
  getCanvasClickPointForBay(
    bayId: number,
    tower: 'context' | 'denoiser',
  ): { readonly x: number; readonly y: number; readonly visible: boolean } | null;
}

declare global {
  interface Window {
    __TWOTOWER_TELEMETRY__?: TwoTowerTelemetrySnapshot;
    __TWOTOWER_APP__?: TwoTowerAppHandle;
  }
}

// ============================================================================
// SMALL PURE HELPERS
// ============================================================================

const DRAG_THRESHOLD_PX = 5;

const VIEW_TABS: readonly {
  readonly mode: TwoTowerSpatialViewMode;
  readonly label: string;
  readonly title: string;
}[] = Object.freeze([
  { mode: 'macro', label: 'Macro', title: 'Dual-Tower 52×2 Macro Architecture' },
  { mode: 'paired_gqa', label: 'GQA', title: 'Paired 35/65 GQA Attention Cutaway (6 bays)' },
  { mode: 'paired_mamba', label: 'Mamba', title: 'Paired 35/65 Mamba-2 SSM & Conv1D Cutaway (23 bays)' },
  { mode: 'timestep', label: 'Timestep', title: 'Timestep Conditioning & Split AdaLN-Single' },
  { mode: 'moe', label: 'MoE', title: 'Granular MoE (128+1) FP64 Margin Cutaway (23 bays)' },
  { mode: 'embedding', label: 'Embed', title: 'Independent Untied NoPE Embeddings (V=34)' },
  { mode: 'output', label: 'Output', title: 'Unmodulated Final RMSNorm, Untied LM Head & MDLM Sampler' },
]);

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

function fmtBytesCompact(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function fmtL2(arr: Float32Array): string {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
  return Math.sqrt(s).toFixed(4);
}

function fmtVecPreview(arr: Float32Array, max = 4): string {
  const parts: string[] = [];
  for (let i = 0; i < Math.min(max, arr.length); i++) {
    const v = arr[i];
    parts.push(`${v >= 0 ? '+' : ''}${v.toFixed(2)}`);
  }
  return `[${parts.join(', ')}${arr.length > max ? ', …' : ''}]`;
}

function isInteractiveControl(target: EventTarget | null): boolean {
  if (!target || typeof target !== 'object') return false;
  const node = target as Partial<HTMLElement>;
  if (typeof node.closest === 'function') {
    return (
      node.closest(
        'button, input, select, textarea, a[href], [role="button"], [role="tab"], summary, [contenteditable="true"]',
      ) !== null
    );
  }
  const tag = (node.tagName ?? '').toUpperCase();
  return (
    tag === 'BUTTON' ||
    tag === 'INPUT' ||
    tag === 'SELECT' ||
    tag === 'TEXTAREA' ||
    tag === 'A'
  );
}

function isTextInputTarget(target: EventTarget | null): boolean {
  if (!target || typeof target !== 'object') return false;
  const node = target as Partial<HTMLElement>;
  if (typeof node.closest === 'function') {
    return node.closest('input, select, textarea, [contenteditable="true"]') !== null;
  }
  const tag = (node.tagName ?? '').toUpperCase();
  return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA';
}

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

function phaseBadgeClass(phase: TwoTowerSessionRecord['lifecyclePhase']): string {
  switch (phase) {
    case 'ready':
      return 'sp-badge sp-badge--emerald';
    case 'block_initialized':
    case 'block_refining':
      return 'sp-badge sp-badge--time';
    case 'refinement_denoised':
      return 'sp-badge sp-badge--den';
    case 'completed_uncommitted':
      return 'sp-badge sp-badge--ctx';
    case 'eos_terminated':
    case 'capacity_exhausted':
      return 'sp-badge sp-badge--warn';
  }
}

function towerBadgeClass(role: TwoTowerTowerRole): string {
  if (role === 'context') return 'sp-badge sp-badge--ctx';
  if (role === 'denoiser') return 'sp-badge sp-badge--den';
  return 'sp-badge sp-badge--teal';
}

function originBadgeClass(origin: string): string {
  if (origin === 'context_computed') return 'sp-badge sp-badge--emerald';
  if (origin === 'context_reconstructed') return 'sp-badge sp-badge--time';
  if (origin === 'persistent_cache') return 'sp-badge sp-badge--ctx';
  return 'sp-badge sp-badge--den';
}

function buildCapabilityTooltips(
  session: TwoTowerSession,
  record: TwoTowerSessionRecord,
  caps: TwoTowerSessionCapabilities,
) {
  const autoFork = caps.willAutoForkOnMutatingStep ? ' (rewound: will auto-fork branch)' : '';
  const term = caps.isTerminated
    ? `Disabled: branch terminated (${caps.terminationReason}). Rewind, fork, or open Workbench to restart.`
    : null;
  const maxBr =
    !caps.isAtBranchTail && session.bounds.maxBranches < 2
      ? 'Disabled: rewound cursor and maxBranches < 2.'
      : null;
  const blocked = term ?? maxBr;

  return {
    init: caps.canInitDiffusionBlock
      ? `Initialize [MASK]^${caps.pendingBlockSize} diffusion block at pos [${caps.committedPrefixLen}..${caps.committedPrefixLen + caps.pendingBlockSize - 1}]${autoFork}.`
      : blocked ??
        (caps.hasActiveBlock
          ? `Disabled: diffusion block b${record.activeBlock!.blockId} is already active (${caps.activeBlockStatus}).`
          : `Disabled: P + B (${caps.committedPrefixLen} + ${caps.pendingBlockSize}) exceeds KV capacity (${caps.kvCapacity}).`),
    refine: caps.canRunDiffusionRefinement
      ? `Run Denoiser forward pass r=${record.activeBlock!.completedRefinements} on block b${record.activeBlock!.blockId}${autoFork}.`
      : blocked ??
        (!caps.hasActiveBlock
          ? 'Disabled: initialize a diffusion block first.'
          : caps.activeBlockStatus === 'awaiting_confidence_commit'
            ? 'Disabled: pending denoised refinement is awaiting Commit Conf.'
            : caps.activeBlockStatus === 'completed_uncommitted'
              ? 'Disabled: block has 0 masks remaining — click Commit Block.'
              : 'Disabled: refinement budget K exhausted.'),
    conf: caps.canCommitRefinementConfidence
      ? `Apply MDLM [MASK]=0 constraints & confidence unmasking (γ=${session.pendingConfig.confidenceThreshold.toFixed(2)})${autoFork}.`
      : blocked ??
        (!caps.hasActiveBlock
          ? 'Disabled: initialize and refine a block first.'
          : `Disabled: run Refine first (status: ${caps.activeBlockStatus}).`),
    commit: caps.canCommitActiveBlockToContext
      ? `Commit completed block b${record.activeBlock!.blockId} into persistent Context cache${autoFork}.`
      : blocked ??
        (!caps.hasActiveBlock
          ? 'Disabled: no active diffusion block.'
          : `Disabled: ${record.activeBlock!.nMasked} masked slot(s) remain (status: ${caps.activeBlockStatus}).`),
    ar: caps.canStepAr
      ? `Run 1 causal Context AR step at pos ${caps.committedPrefixLen}${autoFork}.`
      : blocked ??
        (caps.hasActiveBlock
          ? `Disabled: diffusion block b${record.activeBlock!.blockId} is active.`
          : `Disabled: KV capacity full (${caps.committedPrefixLen}/${caps.kvCapacity}).`),
    mockAr: caps.canStepMockAr
      ? `Run 1 unmodulated Denoiser Mock-AR step (tModel=null, KV[:-1] + mambaS2)${autoFork}.`
      : blocked ??
        (caps.hasActiveBlock
          ? `Disabled: diffusion block b${record.activeBlock!.blockId} is active.`
          : `Disabled: KV capacity full (${caps.committedPrefixLen}/${caps.kvCapacity}).`),
    next: caps.canStepNext
      ? `Execute next legal step (${caps.nextTransitionKind}) for mode ${session.pendingConfig.mode}${autoFork}.`
      : blocked ?? 'Disabled: no legal next transition in current mode/capacity.',
  };
}

interface FlyAnimation {
  readonly from: CameraPose;
  readonly to: CameraPose;
  readonly startMs: number;
  readonly durationMs: number;
}

// ============================================================================
// MAIN SPATIAL APPLICATION MOUNT
// ============================================================================

export function mountTwoTowerSpatialApp(root: HTMLElement): TwoTowerAppHandle {
  const model = createDefaultTwoTowerModel();
  let session: TwoTowerSession = createTwoTowerSession(model);

  const reducedMotionMq =
    typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;
  let reducedMotion = reducedMotionMq?.matches ?? false;

  // Spatial UI state
  let viewMode: TwoTowerSpatialViewMode = 'macro';
  let selectedBayId: number | null = null;
  let selectedTower: TwoTowerTowerRole = 'shared';
  let hoveredBayId: number | null = null;
  let hoveredTower: TwoTowerTowerRole | null = null;
  let activeLinkerChannel: TwoTowerLinkerChannel = 'all';
  let macroReplayActive = false;
  let macroReplayPhase: number | null = null;
  let macroReplayLastMs: number | null = null;

  // Per-view inspection parameters
  let towerFocus: TwoTowerPairedFocus = 'paired';
  let gqaBayId = 6;
  let gqaGroup0Head = 0;
  let gqaGroup1Head = 16;
  let gqaSelectedCell: { readonly tower: 'context' | 'denoiser'; readonly cell: GqaCellRef } | null =
    null;
  let gqaHoveredCell: { readonly tower: 'context' | 'denoiser'; readonly cell: GqaCellRef } | null =
    null;

  let mambaBayId = 1;
  let mambaHead = 0;
  let mambaLocalPos = 0;
  let mambaSelectedCell: {
    readonly tower: 'context' | 'denoiser';
    readonly cell: MambaCellRef;
  } | null = null;
  let mambaHoveredCell: {
    readonly tower: 'context' | 'denoiser';
    readonly cell: MambaCellRef;
  } | null = null;

  let timestepBayId = 6;
  let timestepLocalPos = 0;
  let timestepSelectedPart: string | null = null;
  let timestepHoveredPart: string | null = null;
  let timestepSelectedCell: TimestepCellRef | null = null;

  let moeBayId = 2;
  let moeTowerRole: 'context' | 'denoiser' | undefined = undefined;
  let moeFocusToken: number | undefined = undefined;
  let moeFocusExpert: number | undefined = undefined;
  let moeSelectedPart: string | null = null;
  let moeHoveredPart: string | null = null;
  let moeSelectedCell: MoeCellRef | null = null;

  let embedTowerRole: 'context' | 'denoiser' | undefined = undefined;
  let embedFocusToken: number | undefined = undefined;
  let embedSelectedPart: string | null = null;
  let embedHoveredPart: string | null = null;
  let embedActiveCell: EmbeddingCellRef | null = null;

  let outputTowerRole: 'context' | 'denoiser' | undefined = undefined;
  let outputFocusToken: number | undefined = undefined;
  let outputSelectedPart: string | null = null;
  let outputHoveredPart: string | null = null;
  let outputActiveCell: OutputCellRef | null = null;
  let outputSelectedCandidateId: number | null = null;

  // Stepper & partial-denominator accumulation playback
  let stepperStep: number | null = null;
  let stepperPlaying = false;
  let stepperStepProgress = 0;
  let stepperTermProgress = 1;
  let stepperLastTickMs: number | null = null;

  // Drawer state
  let workbenchOpen = false;
  let workbenchCtrl: DiagnosticWorkbenchController | null = null;

  // Renderer, camera & telemetry state
  let currentScene: TwoTowerSpatialScene | null = null;
  let currentMetrics: ViewportMetrics = makeViewportMetrics(0, 0, 960, 680, window.devicePixelRatio || 1);
  let cameraPose: CameraPose = fitTwoTowerMacroCamera(TWO_TOWER_MACRO_FIT_BOUNDS, currentMetrics);
  let currentSnapshot: CameraSnapshot | null = null;
  let flyAnim: FlyAnimation | null = null;
  let geometryDirty = true;
  let sceneVersion = 0;
  let pendingRafId: number | null = null;
  let pendingHoverRafId: number | null = null;
  let hoverClientX = 0;
  let hoverClientY = 0;

  let lastAtlasKey: string | null = null;
  let atlasUploads = 0;
  let currentAtlasDims: readonly [number, number] | null = null;
  let maxTextureSize = 0;
  let drawCallsLastFrame = 0;
  let lastFrameMs = 0;
  let totalFrameMs = 0;
  let rafFramesRendered = 0;
  let glErrorCount = 0;
  let lastOverlayKey = '';

  // ==========================================================================
  // BUILD DOM SHELL
  // ==========================================================================
  root.textContent = '';
  const shell = el('div', 'sp-shell');

  // 1. Top Header / View Bar (~48px)
  const topbar = el('header', 'sp-topbar');
  const brand = el('div', 'sp-brand');
  const titleEl = el('h1', 'sp-title', 'Nemotron-Labs-TwoTower · Spatial TT3');
  const epistemicPill = el(
    'span',
    'sp-epistemic-pill',
    `d=${TWOTOWER_MINI_CONFIG.d} · V=${TWOTOWER_MINI_CONFIG.vocabSize} · 2×52L · Synthetic Mini`,
  );
  epistemicPill.dataset.testid = 'epistemic-pill';
  epistemicPill.title = `${TWOTOWER_SPEC.productionDefaults.epistemicNote} Pinned HF rev ${TWOTOWER_SPEC.provenance.hfRevision.slice(0, 8)}.`;
  brand.append(titleEl, epistemicPill);

  const tablist = el('div', 'sp-tablist');
  tablist.setAttribute('role', 'tablist');
  tablist.setAttribute('aria-label', 'TwoTower Spatial Views');
  const tabButtons: HTMLButtonElement[] = [];

  for (let i = 0; i < VIEW_TABS.length; i++) {
    const def = VIEW_TABS[i];
    const btn = el('button', 'sp-tab', def.label);
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', def.mode === viewMode ? 'true' : 'false');
    btn.setAttribute('tabindex', def.mode === viewMode ? '0' : '-1');
    btn.title = def.title;
    btn.dataset.viewMode = def.mode;
    btn.dataset.testid = `view-tab-${def.mode}`;
    btn.addEventListener('click', () => switchViewMode(def.mode));
    tablist.append(btn);
    tabButtons.push(btn);
  }

  tablist.addEventListener('keydown', (ev) => {
    if (
      ev.key !== 'ArrowLeft' &&
      ev.key !== 'ArrowRight' &&
      ev.key !== 'Home' &&
      ev.key !== 'End'
    ) {
      return;
    }
    const curIdx = tabButtons.findIndex((t) => t === ev.target || t.getAttribute('aria-selected') === 'true');
    const baseIdx = curIdx >= 0 ? curIdx : 0;
    let nextIdx = baseIdx;
    if (ev.key === 'ArrowRight') nextIdx = (baseIdx + 1) % tabButtons.length;
    else if (ev.key === 'ArrowLeft') nextIdx = (baseIdx - 1 + tabButtons.length) % tabButtons.length;
    else if (ev.key === 'Home') nextIdx = 0;
    else if (ev.key === 'End') nextIdx = tabButtons.length - 1;

    ev.preventDefault();
    ev.stopPropagation();
    const targetTab = tabButtons[nextIdx];
    const nextMode = targetTab.dataset.viewMode as TwoTowerSpatialViewMode;
    switchViewMode(nextMode);
    targetTab.focus();
  });

  const topActions = el('div', 'sp-top-actions');
  const btnResetCamera = el('button', 'sp-top-btn', 'Reset View');
  btnResetCamera.type = 'button';
  btnResetCamera.dataset.testid = 'btn-reset-camera';
  btnResetCamera.title = 'Fit camera to current view (R)';

  const btnToggleWorkbench = el('button', 'sp-top-btn', 'Workbench');
  btnToggleWorkbench.type = 'button';
  btnToggleWorkbench.dataset.testid = 'btn-toggle-workbench';
  btnToggleWorkbench.setAttribute('aria-expanded', 'false');
  btnToggleWorkbench.setAttribute('aria-controls', 'workbench-drawer');
  btnToggleWorkbench.title = 'Open collapsible Developer Diagnostic Workbench synced with this session';

  topActions.append(btnResetCamera, btnToggleWorkbench);
  topbar.append(brand, tablist, topActions);

  // 2. Compact Session & Context Control Strip (~42px)
  const sessionStrip = el('section', 'sp-session-strip');
  sessionStrip.setAttribute('aria-label', 'TwoTower session lifecycle and history controls');

  const leftStrip = el('div', 'sp-strip-group');
  const modeSelect = el('select', 'sp-select');
  modeSelect.dataset.testid = 'spatial-mode-select';
  modeSelect.setAttribute('aria-label', 'Execution mode');
  for (const [val, lbl] of [
    ['mask_diffusion', 'Mask-Diffusion'],
    ['ar', 'AR'],
    ['mock_ar', 'Mock-AR'],
  ] as const) {
    const opt = el('option', undefined, lbl);
    opt.value = val;
    modeSelect.append(opt);
  }

  const spBtnInit = el('button', 'sp-btn sp-btn--den', 'Init Block');
  spBtnInit.type = 'button';
  spBtnInit.dataset.testid = 'sp-btn-init';

  const spBtnRefine = el('button', 'sp-btn sp-btn--den', 'Refine');
  spBtnRefine.type = 'button';
  spBtnRefine.dataset.testid = 'sp-btn-refine';

  const spBtnConf = el('button', 'sp-btn sp-btn--den', 'Commit Conf');
  spBtnConf.type = 'button';
  spBtnConf.dataset.testid = 'sp-btn-conf';

  const spBtnCommit = el('button', 'sp-btn sp-btn--emerald', 'Commit Block');
  spBtnCommit.type = 'button';
  spBtnCommit.dataset.testid = 'sp-btn-commit';

  const spBtnAr = el('button', 'sp-btn sp-btn--ctx', 'Step AR');
  spBtnAr.type = 'button';
  spBtnAr.dataset.testid = 'sp-btn-ar';

  const spBtnMockAr = el('button', 'sp-btn sp-btn--ctx', 'Step Mock-AR');
  spBtnMockAr.type = 'button';
  spBtnMockAr.dataset.testid = 'sp-btn-mock-ar';

  const spBtnNext = el('button', 'sp-btn sp-btn--primary', 'Next Step');
  spBtnNext.type = 'button';
  spBtnNext.dataset.testid = 'sp-btn-next';

  leftStrip.append(
    modeSelect,
    el('span', 'sp-strip-sep'),
    spBtnInit,
    spBtnRefine,
    spBtnConf,
    spBtnCommit,
    el('span', 'sp-strip-sep'),
    spBtnAr,
    spBtnMockAr,
    spBtnNext,
  );

  const rightStrip = el('div', 'sp-strip-group');
  const spBtnBack = el('button', 'sp-btn', '← Back');
  spBtnBack.type = 'button';
  spBtnBack.dataset.testid = 'sp-btn-back';

  const spBtnFwd = el('button', 'sp-btn', 'Forward →');
  spBtnFwd.type = 'button';
  spBtnFwd.dataset.testid = 'sp-btn-fwd';

  const spBtnFork = el('button', 'sp-btn', 'Fork');
  spBtnFork.type = 'button';
  spBtnFork.dataset.testid = 'sp-btn-fork';

  const spBranchSelect = el('select', 'sp-select');
  spBranchSelect.dataset.testid = 'sp-branch-select';
  spBranchSelect.setAttribute('aria-label', 'Active session branch');

  const spPhaseBadge = el('span', 'sp-badge');
  spPhaseBadge.dataset.testid = 'sp-phase-badge';

  const spCapacityBadge = el('span', 'sp-badge sp-badge--ctx');
  spCapacityBadge.dataset.testid = 'sp-capacity-badge';

  rightStrip.append(
    spBtnBack,
    spBtnFwd,
    spBtnFork,
    spBranchSelect,
    el('span', 'sp-strip-sep'),
    spPhaseBadge,
    spCapacityBadge,
  );
  sessionStrip.append(leftStrip, rightStrip);

  // 3. Main Split Stage
  const stageSplit = el('div', 'sp-stage-split');

  // Left/Center WebGL2 Viewport + Overlay Host
  const viewportHost = el('main', 'sp-viewport-host');
  const canvas = el('canvas');
  canvas.id = 'twotower-canvas';
  canvas.dataset.testid = 'twotower-canvas';
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'TwoTower 3D WebGL2 Canvas');

  const overlaySvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  overlaySvg.id = 'twotower-overlay-svg';
  overlaySvg.setAttribute('class', 'tt-overlay-svg');
  overlaySvg.setAttribute('aria-hidden', 'true');

  const overlayLabels = el('div', 'tt-overlay-labels');
  overlayLabels.id = 'twotower-overlay-labels';
  overlayLabels.setAttribute('aria-hidden', 'true');

  const canvasHud = el('div', 'sp-canvas-hud');
  const canvasTitleBox = el('div', 'sp-canvas-title-box');
  const canvasTitleEl = el('span', 'sp-canvas-title');
  const canvasSubEl = el('span', 'sp-canvas-sub');
  canvasTitleBox.append(canvasTitleEl, canvasSubEl);
  canvasHud.append(canvasTitleBox);

  const bottomDock = el('div', 'sp-bottom-dock');
  bottomDock.dataset.testid = 'view-controls-dock';
  bottomDock.setAttribute('role', 'region');
  bottomDock.setAttribute('aria-label', 'Active view controls');

  const fallbackEl = el('div', 'tt-webgl-fallback');
  fallbackEl.id = 'webgl-fallback';
  fallbackEl.hidden = true;

  viewportHost.append(canvas, overlaySvg, overlayLabels, canvasHud, bottomDock, fallbackEl);

  // Right Editorial Inspector Rail
  const inspector = el('aside', 'sp-inspector');
  inspector.setAttribute('aria-label', 'TwoTower Spatial Inspector');

  const warningBannerEl = el('div', 'sp-warning-banner');
  warningBannerEl.dataset.testid = 'inspector-warning-banner';
  warningBannerEl.hidden = true;

  const historicalConfigCard = el('section', 'sp-card sp-card--muted');
  historicalConfigCard.dataset.testid = 'historical-config-card';

  const selectionIdentityCard = el('section', 'sp-card');
  selectionIdentityCard.dataset.testid = 'selection-identity-card';

  const formulaCard = el('section', 'sp-card');
  formulaCard.dataset.testid = 'formula-card';

  const viewDiagnosticCard = el('section', 'sp-card');
  viewDiagnosticCard.dataset.testid = 'view-diagnostic-card';

  const telemetryFooter = el('footer', 'sp-telemetry-footer');
  telemetryFooter.dataset.testid = 'inspector-telemetry-footer';

  inspector.append(
    warningBannerEl,
    historicalConfigCard,
    selectionIdentityCard,
    formulaCard,
    viewDiagnosticCard,
    telemetryFooter,
  );

  stageSplit.append(viewportHost, inspector);

  // 4. Collapsible Developer Diagnostic Workbench Drawer
  const workbenchDrawer = el('aside', 'tt-workbench-drawer');
  workbenchDrawer.id = 'workbench-drawer';
  workbenchDrawer.dataset.testid = 'workbench-drawer';
  workbenchDrawer.setAttribute('role', 'dialog');
  workbenchDrawer.setAttribute('aria-label', 'Developer Diagnostic Session Workbench');
  workbenchDrawer.hidden = true;

  const drawerBar = el('div', 'tt-drawer-bar');
  const drawerTitle = el(
    'span',
    'tt-drawer-title',
    'Developer Diagnostic Session Workbench (Bidirectionally Synced)',
  );
  const drawerCloseBtn = el('button', 'tt-drawer-close', 'Close (Esc)');
  drawerCloseBtn.type = 'button';
  drawerCloseBtn.dataset.testid = 'btn-close-workbench';
  drawerBar.append(drawerTitle, drawerCloseBtn);

  const drawerBody = el('div', 'tt-drawer-body');
  workbenchDrawer.append(drawerBar, drawerBody);

  shell.append(topbar, sessionStrip, stageSplit, workbenchDrawer);
  root.append(shell);

  // ==========================================================================
  // INITIALIZE WEBGL2 DEVICE, PASSES & OVERLAY
  // ==========================================================================
  let device: RenderDevice | null = null;
  let solidsPass: SolidsPass | null = null;
  let sheetsPass: SheetsPass | null = null;
  let linesPass: LinesPass | null = null;
  let ribbonsPass: RibbonsPass | null = null;
  let flatFlowPass: FlatFlowRibbonsPass | null = null;

  const showFallback = (heading: string, detail: string) => {
    fallbackEl.innerHTML = '';
    const card = el('div', 'tt-webgl-fallback-card');
    card.append(
      el('h2', undefined, heading),
      el('p', 'sp-note', detail),
      el(
        'p',
        'sp-note',
        'You can still inspect the session, lifecycle stepping, and full Developer Workbench drawer.',
      ),
    );
    fallbackEl.append(card);
    fallbackEl.hidden = false;
  };

  try {
    device = new RenderDevice(canvas);
    solidsPass = new SolidsPass(device.gl);
    sheetsPass = new SheetsPass(device.gl);
    linesPass = new LinesPass(device.gl);
    ribbonsPass = new RibbonsPass(device.gl);
    flatFlowPass = new FlatFlowRibbonsPass(device.gl);
    maxTextureSize = Number(device.gl.getParameter(device.gl.MAX_TEXTURE_SIZE)) || 0;
  } catch (err) {
    const detail = err instanceof Error ? err.message : 'WebGL2 unavailable';
    showFallback('WebGL2 Viewport Unavailable', `Could not initialize WebGL2 context (${detail}).`);
  }

  const overlay = new TwoTowerSpatialOverlay(overlaySvg, overlayLabels);

  // ==========================================================================
  // CAMERA FITTERS & SMOOTH FLY
  // ==========================================================================
  function getCanvasInsets(): { topInsetPx: number; bottomInsetPx: number; isNarrow: boolean } {
    const isNarrow = window.innerWidth < 1200 || currentMetrics.hostWidth < 760;
    const dockH = bottomDock.offsetHeight || 42;
    const hudH = canvasHud.offsetHeight || 28;
    return {
      topInsetPx: Math.max(36, hudH + 14),
      bottomInsetPx: Math.max(54, dockH + 18),
      isNarrow,
    };
  }

  function computeFitPoseForScene(scene: TwoTowerSpatialScene, bounds: AABB = scene.fitBounds): CameraPose {
    const { topInsetPx, bottomInsetPx, isNarrow } = getCanvasInsets();
    switch (scene.viewMode) {
      case 'macro':
        return fitTwoTowerMacroCamera(bounds, currentMetrics, {
          topInsetPx,
          bottomInsetPx,
          isNarrow,
        });
      case 'paired_gqa':
        return fitPairedGqaCamera(bounds, currentMetrics, {
          topInsetPx,
          bottomInsetPx,
          isNarrow,
        });
      case 'paired_mamba':
        return fitPairedMambaCamera(bounds, currentMetrics, {
          topInsetPx,
          bottomInsetPx,
          isNarrow,
        });
      case 'timestep':
        return fitTimestepCamera(currentMetrics, bounds, {
          topInsetPx,
          bottomInsetPx,
          isNarrow,
        });
      case 'moe':
        return fitSpatialMoeCamera(bounds, currentMetrics, {
          topInsetPx,
          bottomInsetPx,
          isNarrow,
        });
      case 'embedding':
        return fitSpatialEmbeddingCamera(currentMetrics, bounds, {
          topInsetPx,
          bottomInsetPx,
          isNarrow,
        });
      case 'output':
        return fitSpatialOutputCamera(currentMetrics, bounds, {
          topInsetPx,
          bottomInsetPx,
          isNarrow,
        });
    }
  }

  function flyToPose(target: CameraPose, instant = false): void {
    if (reducedMotion || instant) {
      flyAnim = null;
      cameraPose = cloneCameraPose(target);
      scheduleDraw();
      return;
    }
    flyAnim = {
      from: cloneCameraPose(cameraPose),
      to: cloneCameraPose(target),
      startMs: performance.now(),
      durationMs: CAMERA_FLY_DURATION_MS,
    };
    scheduleDraw();
  }

  function focusOpStepCamera(stepIdx: number): void {
    if (!currentScene) return;
    const opBounds = currentScene.opFocusBounds?.[stepIdx];
    if (!opBounds) return;
    const { topInsetPx, bottomInsetPx, isNarrow } = getCanvasInsets();
    const pose = fitOpBoundsCamera(opBounds, currentMetrics, {
      azimuthDeg: isNarrow ? 16 : 19,
      elevationDeg: isNarrow ? 25 : 23,
      fovDeg: DEFAULT_FOV_DEG,
      topInsetPx,
      bottomInsetPx,
      fillFraction: 0.80,
      isNarrow,
    });
    flyToPose(pose);
  }

  // ==========================================================================
  // SCENE BUILDING & GPU UPLOAD (WITH ATLAS DEDUPLICATION)
  // ==========================================================================
  function buildActiveScene(): TwoTowerSpatialScene {
    switch (viewMode) {
      case 'macro':
        return buildTwoTowerMacroScene(session, {
          selectedTower,
          selectedBayId,
          hoveredBayId,
          hoveredTower,
          activeLinkerChannel,
          instructionalReplayPhase: macroReplayPhase,
          reducedMotion,
        });
      case 'paired_gqa':
        return buildTwoTowerPairedGqaScene(session, {
          bayId: gqaBayId,
          towerFocus,
          group0Head: gqaGroup0Head,
          group1Head: gqaGroup1Head,
          selectedCell: gqaSelectedCell,
          hoveredCell: gqaHoveredCell,
          stepperStep,
          stepProgress: stepperStepProgress,
          termProgress: stepperTermProgress,
          reducedMotion,
        });
      case 'paired_mamba':
        return buildTwoTowerPairedMambaScene(session, {
          bayId: mambaBayId,
          towerFocus,
          activeHead: mambaHead,
          localTokenIndex: mambaLocalPos,
          selectedCell: mambaSelectedCell,
          hoveredCell: mambaHoveredCell,
          stepperStep,
          stepProgress: stepperStepProgress,
          termProgress: stepperTermProgress,
          reducedMotion,
        });
      case 'timestep':
        return buildTwoTowerTimestepScene(session, {
          bayId: timestepBayId,
          localTokenIndex: timestepLocalPos,
          selectedPartId: timestepSelectedPart,
          hoveredPartId: timestepHoveredPart,
          selectedCell: timestepSelectedCell,
          stepperStep,
        });
      case 'moe':
        return buildTwoTowerMoeScene(session, {
          bayId: moeBayId,
          towerRole: moeTowerRole,
          focusToken: moeFocusToken,
          focusExpert: moeFocusExpert,
          selectedPart: moeSelectedPart,
          hoveredPart: moeHoveredPart,
          selectedCell: moeSelectedCell,
          stepperStep,
          stepperActive: stepperStep !== null,
          reducedMotion,
          stepProgress: stepperStepProgress,
          termProgress: stepperTermProgress,
        });
      case 'embedding':
        return buildTwoTowerEmbeddingScene(session, {
          towerRole: embedTowerRole,
          focusToken: embedFocusToken,
          selectedPart: embedSelectedPart,
          hoveredPart: embedHoveredPart,
          activeCell: embedActiveCell,
        });
      case 'output':
        return buildTwoTowerOutputScene(session, {
          towerRole: outputTowerRole,
          focusToken: outputFocusToken,
          selectedPart: outputSelectedPart,
          hoveredPart: outputHoveredPart,
          activeCell: outputActiveCell,
          selectedCandidateTokenId: outputSelectedCandidateId,
        });
    }
  }

  function uploadSceneToGpu(scene: TwoTowerSpatialScene): void {
    if (!device || device.isLost || !solidsPass || !sheetsPass || !linesPass || !ribbonsPass || !flatFlowPass) {
      return;
    }
    const useSheets = Boolean(scene.sheetCount && scene.sheetCount > 0 && scene.viewMode !== 'macro');

    if (useSheets && scene.sheetData && scene.sheetCount) {
      sheetsPass.upload(
        scene.sheetData,
        scene.sheetCount,
        scene.sheetBasisData,
        scene.sheetRevealData,
      );
      if (scene.valueAtlas) {
        const key =
          scene.atlasKey ??
          `${scene.viewMode}:${scene.valueAtlas.width}x${scene.valueAtlas.height}:${scene.valueAtlas.data.byteLength}`;
        if (key !== lastAtlasKey) {
          sheetsPass.uploadValueAtlas(
            scene.valueAtlas.width,
            scene.valueAtlas.height,
            scene.valueAtlas.data,
          );
          lastAtlasKey = key;
          atlasUploads += 1;
          currentAtlasDims = [scene.valueAtlas.width, scene.valueAtlas.height];
        }
      } else {
        sheetsPass.setValueTexture(null);
        currentAtlasDims = null;
      }
      solidsPass.upload(new Float32Array(0), 0);
    } else {
      solidsPass.upload(scene.solidData, scene.solidCount);
      sheetsPass.upload(new Float32Array(0), 0);
      currentAtlasDims = null;
    }

    linesPass.upload(scene.lineData, scene.lineCount);
    ribbonsPass.upload(scene.ribbonData, scene.ribbonCount);
    if (scene.flowVertices && scene.flowIndices && scene.flowIndices.length > 0) {
      flatFlowPass.upload(scene.flowVertices, scene.flowIndices);
    } else {
      flatFlowPass.upload(new Float32Array(0), new Uint32Array(0));
    }
  }

  function rebuildSceneIfNeeded(): void {
    if (!geometryDirty && currentScene !== null) return;
    geometryDirty = false;
    currentScene = buildActiveScene();
    sceneVersion += 1;
    uploadSceneToGpu(currentScene);
    syncAllUi();
  }

  // ==========================================================================
  // UI SYNCHRONIZATION (HEADER, STRIP, DOCK, INSPECTOR, TELEMETRY)
  // ==========================================================================
  function syncHeaderAndSessionStrip(): void {
    for (const tab of tabButtons) {
      const active = tab.dataset.viewMode === viewMode;
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
      tab.setAttribute('tabindex', active ? '0' : '-1');
    }

    const record = getActiveRecord(session);
    const caps = getSessionCapabilities(session);
    const tips = buildCapabilityTooltips(session, record, caps);

    modeSelect.value = session.pendingConfig.mode;

    spBtnInit.disabled = !caps.canInitDiffusionBlock;
    spBtnInit.title = tips.init;
    spBtnInit.textContent = `Init Block`;

    spBtnRefine.disabled = !caps.canRunDiffusionRefinement;
    spBtnRefine.title = tips.refine;

    spBtnConf.disabled = !caps.canCommitRefinementConfidence;
    spBtnConf.title = tips.conf;

    spBtnCommit.disabled = !caps.canCommitActiveBlockToContext;
    spBtnCommit.title = tips.commit;

    spBtnAr.disabled = !caps.canStepAr;
    spBtnAr.title = tips.ar;

    spBtnMockAr.disabled = !caps.canStepMockAr;
    spBtnMockAr.title = tips.mockAr;

    spBtnNext.disabled = !caps.canStepNext;
    spBtnNext.title = tips.next;

    spBtnBack.disabled = !caps.canNavigateBack;
    spBtnBack.title = caps.canNavigateBack
      ? 'Step backward in active branch history'
      : 'Disabled: at root prefill record (#0)';

    spBtnFwd.disabled = !caps.canNavigateForward;
    spBtnFwd.title = caps.canNavigateForward
      ? 'Step forward in active branch history'
      : 'Disabled: at branch tail';

    spBtnFork.disabled = !caps.canForkBranch;
    spBtnFork.title = caps.canForkBranch
      ? 'Fork a new branch from the current record'
      : 'Disabled: maxBranches < 2';

    spBranchSelect.textContent = '';
    for (const br of session.branches) {
      const opt = el('option', undefined, `${br.branchId} (#${br.cursorIndex}/${br.records.length - 1})`);
      opt.value = br.branchId;
      spBranchSelect.append(opt);
    }
    spBranchSelect.value = session.activeBranchId;

    spPhaseBadge.className = phaseBadgeClass(record.lifecyclePhase);
    spPhaseBadge.textContent = record.lifecyclePhase;

    spCapacityBadge.textContent = `Context ${caps.committedPrefixLen} / ${caps.kvCapacity}`;
    spCapacityBadge.title = `${caps.remainingKvCapacity} free KV slots of ${caps.kvCapacity}`;

    btnToggleWorkbench.setAttribute('aria-expanded', workbenchOpen ? 'true' : 'false');
  }

  function renderStepperDockGroup(
    scene: TwoTowerSpatialScene,
    allowPlay: boolean,
  ): HTMLDivElement {
    const group = el('div', 'sp-dock-group');
    group.append(el('span', 'sp-dock-label', 'Op'));
    const steps = scene.stepperSteps ?? [];
    for (let i = 0; i < steps.length; i++) {
      const st = steps[i];
      const b = el('button', `sp-dock-pill${stepperStep === i ? ' is-active' : ''}`, `${i}`);
      b.type = 'button';
      b.dataset.testid = `stepper-step-${i}`;
      b.title = `${st.title}: ${st.summary}`;
      b.addEventListener('click', () => {
        stepperPlaying = false;
        stepperLastTickMs = null;
        stepperStep = i;
        stepperStepProgress = 0;
        stepperTermProgress = 1;
        clearDirectCellSelectionForView();
        geometryDirty = true;
        rebuildSceneIfNeeded();
        focusOpStepCamera(i);
        scheduleDraw();
      });
      group.append(b);
    }

    const offBtn = el('button', `sp-dock-pill${stepperStep === null ? ' is-active' : ''}`, 'Off');
    offBtn.type = 'button';
    offBtn.dataset.testid = 'stepper-off';
    offBtn.title = 'Exit operation stepper and show full view';
    offBtn.addEventListener('click', () => {
      stepperPlaying = false;
      stepperLastTickMs = null;
      stepperStep = null;
      stepperStepProgress = 0;
      stepperTermProgress = 1;
      geometryDirty = true;
      rebuildSceneIfNeeded();
      if (currentScene) flyToPose(computeFitPoseForScene(currentScene));
      scheduleDraw();
    });
    group.append(offBtn);

    if (allowPlay) {
      const playBtn = el(
        'button',
        `sp-dock-pill${stepperPlaying ? ' is-active' : ''}`,
        stepperPlaying ? 'Pause' : 'Play',
      );
      playBtn.type = 'button';
      playBtn.dataset.testid = 'stepper-play';
      playBtn.disabled = reducedMotion;
      playBtn.title = reducedMotion
        ? 'Disabled under prefers-reduced-motion: reduce (manual steps remain active)'
        : 'Animate operation steps and term-by-term accumulation';
      playBtn.addEventListener('click', () => {
        if (reducedMotion) return;
        if (stepperPlaying) {
          stepperPlaying = false;
          stepperLastTickMs = null;
        } else {
          if (stepperStep === null) stepperStep = 0;
          stepperPlaying = true;
          stepperLastTickMs = performance.now();
          stepperTermProgress = 0.05;
        }
        geometryDirty = true;
        scheduleDraw();
      });
      group.append(playBtn);
    }
    return group;
  }

  function renderBottomDock(scene: TwoTowerSpatialScene): void {
    bottomDock.textContent = '';

    if (scene.viewMode === 'macro') {
      const chGroup = el('div', 'sp-dock-group');
      chGroup.append(el('span', 'sp-dock-label', 'Channel'));
      const channels: readonly [TwoTowerLinkerChannel, string][] = [
        ['all', 'All'],
        ['gqa_kv', 'GQA KV'],
        ['mamba_conv', 'Conv1D'],
        ['mamba_ssm', 'SSM'],
        ['timestep', 'Timestep'],
      ];
      for (const [id, lbl] of channels) {
        const b = el(
          'button',
          `sp-dock-pill${activeLinkerChannel === id ? ' is-active' : ''}`,
          lbl,
        );
        b.type = 'button';
        b.dataset.testid = `linker-channel-${id}`;
        b.addEventListener('click', () => {
          activeLinkerChannel = id;
          geometryDirty = true;
          scheduleDraw();
        });
        chGroup.append(b);
      }

      const replayGroup = el('div', 'sp-dock-group');
      const replayBtn = el(
        'button',
        `sp-dock-pill${macroReplayActive ? ' is-active' : ''}`,
        macroReplayActive ? 'Stop Wave' : 'Replay Wave',
      );
      replayBtn.type = 'button';
      replayBtn.dataset.testid = 'btn-replay-wave';
      replayBtn.disabled = reducedMotion;
      replayBtn.title = reducedMotion
        ? 'Instructional wave replay disabled under prefers-reduced-motion: reduce'
        : 'Instructional visual propagation replay (not measured latency)';
      replayBtn.addEventListener('click', () => {
        if (reducedMotion) return;
        if (macroReplayActive) {
          macroReplayActive = false;
          macroReplayPhase = null;
          macroReplayLastMs = null;
        } else {
          macroReplayActive = true;
          macroReplayPhase = 0.02;
          macroReplayLastMs = performance.now();
        }
        geometryDirty = true;
        scheduleDraw();
      });
      replayGroup.append(replayBtn);

      bottomDock.append(chGroup, el('span', 'sp-strip-sep'), replayGroup);
      return;
    }

    if (scene.viewMode === 'paired_gqa') {
      const focusGroup = el('div', 'sp-dock-group');
      focusGroup.append(el('span', 'sp-dock-label', 'Focus'));
      for (const [val, lbl] of [
        ['paired', 'Paired 35/65'],
        ['context_solo', 'Context Solo'],
        ['denoiser_solo', 'Denoiser Solo'],
      ] as const) {
        const b = el('button', `sp-dock-pill${towerFocus === val ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `tower-focus-${val}`;
        b.addEventListener('click', () => {
          towerFocus = val;
          geometryDirty = true;
          rebuildSceneIfNeeded();
          if (currentScene) flyToPose(computeFitPoseForScene(currentScene));
          scheduleDraw();
        });
        focusGroup.append(b);
      }

      const baySelect = el('select', 'sp-dock-select');
      baySelect.dataset.testid = 'gqa-bay-select';
      baySelect.setAttribute('aria-label', 'GQA Bay');
      for (const id of TWOTOWER_SPEC.layerAlignment.gqaTransferBayIds) {
        const opt = el('option', undefined, `Bay ${formatBayId(id)} (L${id - 1})`);
        opt.value = String(id);
        baySelect.append(opt);
      }
      baySelect.value = String(gqaBayId);
      baySelect.addEventListener('change', () => {
        gqaBayId = Number(baySelect.value);
        gqaSelectedCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      const headSelect = el('select', 'sp-dock-select');
      headSelect.dataset.testid = 'gqa-head-select';
      headSelect.setAttribute('aria-label', 'GQA Query Head');
      for (let h = 0; h < 32; h++) {
        const g = h < 16 ? 0 : 1;
        const opt = el('option', undefined, `Q${String(h).padStart(2, '0')} (KV${g})`);
        opt.value = String(h);
        headSelect.append(opt);
      }
      headSelect.value = String(
        gqaSelectedCell?.cell.headIndex ?? gqaGroup0Head,
      );
      headSelect.addEventListener('change', () => {
        const h = Number(headSelect.value);
        if (h < 16) gqaGroup0Head = h;
        else gqaGroup1Head = h;
        gqaSelectedCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      const pickGroup = el('div', 'sp-dock-group');
      pickGroup.append(baySelect, headSelect);
      bottomDock.append(
        focusGroup,
        el('span', 'sp-strip-sep'),
        pickGroup,
        el('span', 'sp-strip-sep'),
        renderStepperDockGroup(scene, true),
      );
      return;
    }

    if (scene.viewMode === 'paired_mamba') {
      const diag = scene.diagnostics as PairedMambaDiagnostics | undefined;
      const blockLen = Math.max(1, diag?.blockLength ?? 1);
      if (mambaLocalPos >= blockLen) mambaLocalPos = blockLen - 1;

      const focusGroup = el('div', 'sp-dock-group');
      focusGroup.append(el('span', 'sp-dock-label', 'Focus'));
      for (const [val, lbl] of [
        ['paired', 'Paired 35/65'],
        ['context_solo', 'Context Solo'],
        ['denoiser_solo', 'Denoiser Solo'],
      ] as const) {
        const b = el('button', `sp-dock-pill${towerFocus === val ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `tower-focus-${val}`;
        b.addEventListener('click', () => {
          towerFocus = val;
          geometryDirty = true;
          rebuildSceneIfNeeded();
          if (currentScene) flyToPose(computeFitPoseForScene(currentScene));
          scheduleDraw();
        });
        focusGroup.append(b);
      }

      const baySelect = el('select', 'sp-dock-select');
      baySelect.dataset.testid = 'mamba-bay-select';
      baySelect.setAttribute('aria-label', 'Mamba-2 Bay');
      for (const id of TWOTOWER_SPEC.layerAlignment.mambaTransferBayIds) {
        const opt = el('option', undefined, `Bay ${formatBayId(id)} (L${id - 1})`);
        opt.value = String(id);
        baySelect.append(opt);
      }
      baySelect.value = String(mambaBayId);
      baySelect.addEventListener('change', () => {
        mambaBayId = Number(baySelect.value);
        mambaSelectedCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      const headSelect = el('select', 'sp-dock-select');
      headSelect.dataset.testid = 'mamba-head-select';
      headSelect.setAttribute('aria-label', 'Mamba-2 Head');
      for (let h = 0; h < 16; h++) {
        const opt = el('option', undefined, `${formatMambaHeadId(h)} (G${Math.floor(h / 2)})`);
        opt.value = String(h);
        headSelect.append(opt);
      }
      headSelect.value = String(mambaHead);
      headSelect.addEventListener('change', () => {
        mambaHead = Number(headSelect.value);
        mambaSelectedCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      const posSelect = el('select', 'sp-dock-select');
      posSelect.dataset.testid = 'mamba-pos-select';
      posSelect.setAttribute('aria-label', 'Block Position');
      for (let p = 0; p < blockLen; p++) {
        const opt = el('option', undefined, `pos ${p}/${blockLen - 1}`);
        opt.value = String(p);
        posSelect.append(opt);
      }
      posSelect.value = String(mambaLocalPos);
      posSelect.addEventListener('change', () => {
        mambaLocalPos = Number(posSelect.value);
        mambaSelectedCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      const pickGroup = el('div', 'sp-dock-group');
      pickGroup.append(baySelect, headSelect, posSelect);
      bottomDock.append(
        focusGroup,
        el('span', 'sp-strip-sep'),
        pickGroup,
        el('span', 'sp-strip-sep'),
        renderStepperDockGroup(scene, true),
      );
      return;
    }

    if (scene.viewMode === 'timestep') {
      const bayGroup = el('div', 'sp-dock-group');
      bayGroup.append(el('span', 'sp-dock-label', 'Layer'));
      const baySelect = el('select', 'sp-dock-select');
      baySelect.dataset.testid = 'timestep-bay-select';
      baySelect.setAttribute('aria-label', 'Timestep Conditioning Layer Bay');
      for (const entry of TWOTOWER_SPEC.layerAlignment.entries) {
        const orderTag = entry.kind === 'E' ? 'post-norm' : 'pre-norm';
        const opt = el(
          'option',
          undefined,
          `Bay ${formatBayId(entry.bayId)} [${entry.kind} · ${orderTag}]`,
        );
        opt.value = String(entry.bayId);
        baySelect.append(opt);
      }
      baySelect.value = String(timestepBayId);
      baySelect.addEventListener('change', () => {
        timestepBayId = Number(baySelect.value);
        timestepSelectedCell = null;
        geometryDirty = true;
        scheduleDraw();
      });
      bayGroup.append(baySelect);

      bottomDock.append(
        bayGroup,
        el('span', 'sp-strip-sep'),
        renderStepperDockGroup(scene, false),
      );
      return;
    }

    if (scene.viewMode === 'moe') {
      const diag = scene.diagnostics as TwoTowerMoeDiagnostics | undefined;
      const activeTower = diag?.towerRole ?? moeTowerRole ?? 'context';

      const towerGroup = el('div', 'sp-dock-group');
      towerGroup.append(el('span', 'sp-dock-label', 'Tower'));
      for (const [role, lbl] of [
        ['denoiser', 'Denoiser'],
        ['context', 'Context'],
      ] as const) {
        const b = el('button', `sp-dock-pill${activeTower === role ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `moe-tower-${role}`;
        b.addEventListener('click', () => {
          moeTowerRole = role;
          moeSelectedCell = null;
          moeFocusToken = undefined;
          moeFocusExpert = undefined;
          geometryDirty = true;
          scheduleDraw();
        });
        towerGroup.append(b);
      }

      const baySelect = el('select', 'sp-dock-select');
      baySelect.dataset.testid = 'moe-bay-select';
      baySelect.setAttribute('aria-label', 'MoE Bay');
      for (const id of TWOTOWER_SPEC.layerAlignment.statelessMoeBayIds) {
        const opt = el('option', undefined, `Bay ${formatBayId(id)} (L${id - 1})`);
        opt.value = String(id);
        baySelect.append(opt);
      }
      baySelect.value = String(moeBayId);
      baySelect.addEventListener('change', () => {
        moeBayId = Number(baySelect.value);
        moeSelectedCell = null;
        moeFocusExpert = undefined;
        geometryDirty = true;
        scheduleDraw();
      });

      const view = resolveTwoTowerSessionViewState(session);
      const moeTrace =
        activeTower === 'denoiser'
          ? view.resolveDenoiserBayTrace(moeBayId)
          : view.resolveContextBayTrace(moeBayId).trace;
      const tokenSelect = el('select', 'sp-dock-select');
      tokenSelect.dataset.testid = 'moe-token-select';
      tokenSelect.setAttribute('aria-label', 'MoE Focus Token');
      if (moeTrace && moeTrace.kind === 'E') {
        const sOff = moeTrace.sim.startTokenOffset;
        for (let t = 0; t < moeTrace.sim.tokens; t++) {
          const opt = el(
            'option',
            undefined,
            `t${sOff + t}: ${moeTrace.sim.tokenLabels[t] ?? `tok${t}`}`,
          );
          opt.value = String(t);
          tokenSelect.append(opt);
        }
        tokenSelect.value = String(diag?.focusToken ?? Math.max(0, moeTrace.sim.tokens - 1));
      } else {
        const opt = el('option', undefined, 't0');
        opt.value = '0';
        tokenSelect.append(opt);
      }
      tokenSelect.addEventListener('change', () => {
        moeFocusToken = Number(tokenSelect.value);
        moeFocusExpert = undefined;
        moeSelectedCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      const expertSelect = el('select', 'sp-dock-select');
      expertSelect.dataset.testid = 'moe-expert-select';
      expertSelect.setAttribute('aria-label', 'MoE Expert');
      const top8 = diag?.selectionMargin?.top8Summary ?? [];
      if (top8.length > 0) {
        for (const e of top8) {
          const opt = el(
            'option',
            undefined,
            `#${e.rank} ${e.expertTag}${e.selected ? ' ★' : ''}`,
          );
          opt.value = String(e.expertId);
          expertSelect.append(opt);
        }
      } else {
        for (let e = 0; e < 16; e++) {
          const opt = el('option', undefined, formatExpertId(e));
          opt.value = String(e);
          expertSelect.append(opt);
        }
      }
      if (diag?.focusExpert !== null && diag?.focusExpert !== undefined) {
        expertSelect.value = String(diag.focusExpert);
      }
      expertSelect.addEventListener('change', () => {
        moeFocusExpert = Number(expertSelect.value);
        moeSelectedCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      const pickGroup = el('div', 'sp-dock-group');
      pickGroup.append(baySelect, tokenSelect, expertSelect);
      bottomDock.append(
        towerGroup,
        el('span', 'sp-strip-sep'),
        pickGroup,
        el('span', 'sp-strip-sep'),
        renderStepperDockGroup(scene, true),
      );
      return;
    }

    if (scene.viewMode === 'embedding') {
      const diag = scene.diagnostics as TwoTowerEmbeddingDiagnostics | undefined;
      const activeTower = diag?.towerRole ?? embedTowerRole ?? 'context';
      const towerGroup = el('div', 'sp-dock-group');
      towerGroup.append(el('span', 'sp-dock-label', 'Tower'));
      for (const [role, lbl] of [
        ['denoiser', 'Denoiser'],
        ['context', 'Context'],
      ] as const) {
        const b = el('button', `sp-dock-pill${activeTower === role ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `embed-tower-${role}`;
        b.addEventListener('click', () => {
          embedTowerRole = role;
          embedFocusToken = undefined;
          embedActiveCell = null;
          geometryDirty = true;
          scheduleDraw();
        });
        towerGroup.append(b);
      }

      const tokSelect = el('select', 'sp-dock-select');
      tokSelect.dataset.testid = 'embed-token-select';
      tokSelect.setAttribute('aria-label', 'Embedding Focus Token');
      const labels = diag?.tokenLabels ?? [];
      const startOff = diag?.startTokenOffset ?? 0;
      for (let i = 0; i < labels.length; i++) {
        const opt = el('option', undefined, `t${startOff + i}: ${labels[i]}`);
        opt.value = String(i);
        tokSelect.append(opt);
      }
      tokSelect.value = String(embedFocusToken ?? Math.max(0, labels.length - 1));
      tokSelect.addEventListener('change', () => {
        embedFocusToken = Number(tokSelect.value);
        embedActiveCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      bottomDock.append(towerGroup, el('span', 'sp-strip-sep'), tokSelect);
      return;
    }

    if (scene.viewMode === 'output') {
      const diag = scene.diagnostics as TwoTowerOutputDiagnostics | undefined;
      const activeTower = diag?.towerRole ?? outputTowerRole ?? 'context';
      const towerGroup = el('div', 'sp-dock-group');
      towerGroup.append(el('span', 'sp-dock-label', 'Tower'));
      for (const [role, lbl] of [
        ['denoiser', 'Denoiser'],
        ['context', 'Context'],
      ] as const) {
        const b = el('button', `sp-dock-pill${activeTower === role ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `output-tower-${role}`;
        b.addEventListener('click', () => {
          outputTowerRole = role;
          outputFocusToken = undefined;
          outputActiveCell = null;
          geometryDirty = true;
          scheduleDraw();
        });
        towerGroup.append(b);
      }

      const tokSelect = el('select', 'sp-dock-select');
      tokSelect.dataset.testid = 'output-token-select';
      tokSelect.setAttribute('aria-label', 'Output Focus Token');
      const count = diag?.tokenCount ?? 0;
      const startOff = diag?.startTokenOffset ?? 0;
      for (let i = 0; i < count; i++) {
        const opt = el('option', undefined, `pos t${startOff + i} (local ${i})`);
        opt.value = String(i);
        tokSelect.append(opt);
      }
      tokSelect.value = String(diag?.focusToken ?? Math.max(0, count - 1));
      tokSelect.addEventListener('change', () => {
        outputFocusToken = Number(tokSelect.value);
        outputActiveCell = null;
        geometryDirty = true;
        scheduleDraw();
      });

      bottomDock.append(towerGroup, el('span', 'sp-strip-sep'), tokSelect);
    }
  }

  function addKvCell(dl: HTMLDListElement, label: string, value: string, full = false): void {
    const cell = el('div', `sp-kv-cell${full ? ' sp-kv-cell--full' : ''}`);
    cell.append(el('dt', undefined, label), el('dd', undefined, value));
    dl.append(cell);
  }

  function renderInspector(scene: TwoTowerSpatialScene): void {
    const view = resolveTwoTowerSessionViewState(session);

    // 1. Warning Banner
    if (scene.warningBanner) {
      warningBannerEl.hidden = false;
      warningBannerEl.textContent = scene.warningBanner;
    } else {
      warningBannerEl.hidden = true;
      warningBannerEl.textContent = '';
    }

    // 2. Historical Config Guard Badge
    historicalConfigCard.textContent = '';
    const cfg = view.historicalConfig;
    const hHead = el('div', 'sp-card-head');
    hHead.append(
      el(
        'h2',
        'sp-card-title',
        view.isHistoricalCursor
          ? `Historical Record #${view.cursorIndex}`
          : `Live Record #${view.cursorIndex} (configAtStep)`,
      ),
      el(
        'span',
        view.isHistoricalCursor ? 'sp-badge sp-badge--time' : 'sp-badge sp-badge--muted',
        view.isHistoricalCursor ? `Historical Record #${view.cursorIndex}` : view.record.kind,
      ),
    );
    const cfgPills = el('div', 'sp-pill-row');
    cfgPills.append(
      el('span', 'sp-badge sp-badge--ctx', `mode=${cfg.mode}`),
      el('span', 'sp-badge', `B=${cfg.blockSize}`),
      el('span', 'sp-badge', `K=${cfg.stepsPerBlock}`),
      el('span', 'sp-badge', `γ=${cfg.confidenceThreshold.toFixed(2)}`),
      el('span', 'sp-badge', `T=${cfg.temperature}`),
    );
    historicalConfigCard.append(hHead, cfgPills);
    if (view.isHistoricalCursor) {
      historicalConfigCard.append(
        el(
          'p',
          'sp-note',
          'Rewound cursor: strictly renders frozen historical configAtStep. Pending future settings never alter historical records.',
        ),
      );
    }

    // 3. Active Selection Identity Card (ALL 10 AXES!)
    selectionIdentityCard.textContent = '';
    const sel = scene.activeSelection;
    const idHead = el('div', 'sp-card-head');
    idHead.append(
      el('h2', 'sp-card-title', 'Selection Identity (10-Axis)'),
      sel
        ? el('span', originBadgeClass(sel.traceOrigin), sel.traceOrigin)
        : el('span', 'sp-badge sp-badge--muted', 'overview'),
    );
    selectionIdentityCard.append(idHead);

    const idDl = el('dl', 'sp-kv-grid');
    if (sel) {
      const bayDisplay =
        sel.bayId >= 1 && sel.bayId <= 52
          ? `Bay ${formatBayId(sel.bayId)} (L${sel.layerIndex})`
          : sel.bayId === 0
            ? 'Bay 00 (Embed)'
            : sel.bayId === 53
              ? 'Bay 53 (Output)'
              : 'Shared Timestep Rail';
      addKvCell(idDl, '1. towerRole', sel.towerRole);
      addKvCell(idDl, '2. modelId', sel.modelId);
      addKvCell(idDl, '3. paramFingerprint', sel.parameterFingerprint);
      addKvCell(idDl, '4. bayId / layerIdx', bayDisplay);
      addKvCell(idDl, '5. mixerKind', String(sel.mixerKind));
      addKvCell(idDl, '6. executionId', sel.executionId);
      addKvCell(idDl, '7. recordId', `${sel.recordId} (#${sel.recordIndex})`);
      addKvCell(
        idDl,
        '8. block / ref',
        `b=${sel.blockId ?? 'null'} · r=${sel.refinementIndex ?? 'null'}`,
      );
      addKvCell(
        idDl,
        '9. absTokenPos',
        `${sel.absoluteTokenPosition ?? 'null'} [${sel.absoluteTokenRange[0]}..${sel.absoluteTokenRange[1]}]`,
      );
      addKvCell(idDl, '10. traceOrigin', sel.traceOrigin);
      if (sel.headIndex !== undefined || sel.expertId !== undefined || sel.tensorRole) {
        const extra = [
          `tensor=${sel.tensorRole}`,
          sel.headIndex !== undefined ? `head=${sel.headIndex}` : null,
          sel.kvHeadIndex !== undefined ? `kv=${sel.kvHeadIndex}` : null,
          sel.expertId !== undefined ? `expert=${formatExpertId(sel.expertId)}` : null,
        ]
          .filter(Boolean)
          .join(' · ');
        addKvCell(idDl, 'Target Coordinate', extra, true);
      }
      selectionIdentityCard.append(idDl);
    } else {
      addKvCell(idDl, '1. towerRole', 'shared (overview)');
      addKvCell(idDl, '2. modelId', view.model.modelId);
      addKvCell(idDl, '3. paramFingerprint', view.model.twoTowerFingerprint);
      addKvCell(idDl, '4. bayId / layerIdx', 'Click any Context/Denoiser bay');
      addKvCell(idDl, '5. mixerKind', '23M · 23E · 6*');
      addKvCell(idDl, '6. executionId', view.identity.recordId);
      addKvCell(idDl, '7. recordId', `${view.identity.recordId} (#${view.cursorIndex})`);
      addKvCell(
        idDl,
        '8. block / ref',
        `b=${view.identity.blockId ?? 'null'} · r=${view.identity.refinementIndex ?? 'null'}`,
      );
      addKvCell(
        idDl,
        '9. absTokenRange',
        `[${view.identity.absoluteTokenRange[0]}..${view.identity.absoluteTokenRange[1]}]`,
      );
      addKvCell(idDl, '10. traceOrigin', view.contextSnapshot.stateOrigin);
      selectionIdentityCard.append(idDl);
    }

    // 4. Active Formula & Accumulation Card
    formulaCard.textContent = '';
    const fHead = el('div', 'sp-card-head');
    const fTitle = el('h2', 'sp-card-title', 'Active Formula & Accumulation');
    const accum = scene.focusedAccumulation;
    const fBadge = accum?.active
      ? el('span', 'sp-badge sp-badge--time', accum.badgeText)
      : scene.activeFormula
        ? el('span', 'sp-badge sp-badge--den', scene.activeFormula.valueText)
        : el('span', 'sp-badge sp-badge--muted', 'hover / step cell');
    fHead.append(fTitle, fBadge);
    formulaCard.append(fHead);

    if (scene.activeFormula) {
      const f = scene.activeFormula;
      const metaRow = el('div', 'sp-pill-row');
      metaRow.append(
        el('span', 'sp-badge sp-badge--ctx', f.tensorLabel),
        el('span', 'sp-badge', f.coordLabel),
      );
      formulaCard.append(
        metaRow,
        el('div', 'sp-formula-eq', `${f.equation} = ${f.valueText}`),
        el('div', 'sp-formula-sub', f.subEquation),
      );
    } else {
      formulaCard.append(
        el(
          'p',
          'sp-note',
          scene.viewMode === 'macro'
            ? 'Click any bay on either tower to inspect 10-axis provenance, or switch to a cutaway to inspect live Float32 cell equations.'
            : 'Hover or click any matrix sheet cell, or pick an Op step below, to inspect live Float32 cell equations.',
        ),
      );
    }

    // 5. View-Specific Diagnostic Card
    viewDiagnosticCard.textContent = '';
    renderViewSpecificDiagnostics(scene, viewDiagnosticCard);

    // 6. Retained Memory & Telemetry Footer (updated live after each WebGL2 draw too)
    refreshTelemetryFooter();
  }

  function refreshTelemetryFooter(): void {
    const mem = measureTwoTowerSessionRetainedBytes(session);
    telemetryFooter.textContent = '';
    telemetryFooter.append(
      el(
        'p',
        'sp-note',
        `Synthetic Mini (d=8, V=34, L=52, kvCap=24 · HF rev ${TWOTOWER_SPEC.provenance.hfRevision.slice(0, 8)}): not 63.19B checkpoint weights.`,
      ),
    );
    const telGrid = el('div', 'sp-tel-grid');
    telGrid.append(
      el('span', undefined, `Retained: ${fmtBytesCompact(mem.totalRetainedBytes)}`),
      el('span', undefined, `Weights: ${fmtBytesCompact(mem.sharedModelWeightBytes)}`),
      el('span', undefined, `Draws: ${drawCallsLastFrame} · Atlas: ${atlasUploads}`),
      el('span', undefined, `MaxTex: ${maxTextureSize} · ${lastFrameMs.toFixed(1)}ms`),
    );
    telemetryFooter.append(telGrid);
  }

  function renderViewSpecificDiagnostics(
    scene: TwoTowerSpatialScene,
    container: HTMLElement,
  ): void {
    if (scene.viewMode === 'macro') {
      const d = scene.diagnostics as TwoTowerMacroDiagnostics | undefined;
      const head = el('div', 'sp-card-head');
      head.append(
        el('h2', 'sp-card-title', 'Dual-Tower Linker & Loop State'),
        el('span', 'sp-badge sp-badge--teal', '29 Transfer Bays'),
      );
      container.append(head);

      const dl = el('dl', 'sp-kv-grid');
      addKvCell(dl, 'GQA KV Linkers', `${d?.alignedGqaLinkerCount ?? 6} bays (* · read-only)`);
      addKvCell(dl, 'Mamba Conv1D', `${d?.alignedMambaConvLinkerCount ?? 23} bays (upper lane)`);
      addKvCell(dl, 'Mamba SSM FP32', `${d?.alignedMambaSsmLinkerCount ?? 23} bays (lower lane)`);
      addKvCell(dl, 'MoE Linkers', `${d?.moeLinkerCount ?? 0} (0 B stateless)`);
      addKvCell(dl, 'Refinement Loop', String(d?.refinementLoopState ?? 'dormant'));
      addKvCell(dl, 'Commit Pathway', String(d?.blockCommitPathwayState ?? 'dormant'));
      addKvCell(dl, 'Timestep Rail', String(d?.timestepStatusLabel ?? 'BYPASSED'), true);
      container.append(dl);

      const inspectBtn = el(
        'button',
        'sp-btn sp-btn--primary',
        selectedBayId !== null
          ? `Inspect Bay ${selectedBayId >= 1 && selectedBayId <= 52 ? formatBayId(selectedBayId) : selectedBayId} Cutaway →`
          : 'Inspect Bay 06 GQA Cutaway →',
      );
      inspectBtn.type = 'button';
      inspectBtn.dataset.testid = 'btn-inspect-bay-cutaway';
      inspectBtn.addEventListener('click', () => {
        openCutawayForBay(selectedBayId ?? 6, selectedTower);
      });
      container.append(inspectBtn);
      return;
    }

    if (scene.viewMode === 'paired_gqa') {
      const d = scene.diagnostics as PairedGqaDiagnostics | undefined;
      const head = el('div', 'sp-card-head');
      head.append(
        el('h2', 'sp-card-title', 'Paired GQA 4-Length & Mask Audit'),
        el(
          'span',
          d?.attentionMaskMode === 'bidirectional_block'
            ? 'sp-badge sp-badge--den'
            : 'sp-badge sp-badge--ctx',
          d?.attentionMaskMode ?? 'causal',
        ),
      );
      container.append(head);

      const dl = el('dl', 'sp-kv-grid');
      addKvCell(dl, 'queryLen (Q)', String(d?.queryLen ?? 0));
      addKvCell(dl, 'contextPrefixKvLen', String(d?.contextPrefixKvLen ?? 0));
      addKvCell(dl, 'blockKvLen', String(d?.blockKvLen ?? 0));
      addKvCell(dl, 'allocatedKvCapacity', String(d?.allocatedKvCapacity ?? 24));
      addKvCell(
        dl,
        'RG32F Atlas',
        d ? `${d.atlasDimensions[0]}×${d.atlasDimensions[1]} (${d.atlasPrimaryMaskedCells} masked)` : '128×128',
      );
      addKvCell(
        dl,
        'AdaLN / Gate',
        `${d?.hasDenoiserPreNormMod ? 'pre-norm ON' : 'bypassed'} · ${d?.hasDenoiserResidualGate ? 'gate ON' : 'off'}`,
      );
      container.append(dl);
      container.append(
        el(
          'p',
          'sp-note sp-note--ctx',
          d?.conditioningNote ??
            'Logical read-only view of Context persistent prefix KV (zero physical inter-device copy required).',
        ),
      );
      return;
    }

    if (scene.viewMode === 'paired_mamba') {
      const d = scene.diagnostics as PairedMambaDiagnostics | undefined;
      const dec = d?.decomposition ?? null;
      const head = el('div', 'sp-card-head');
      head.append(
        el('h2', 'sp-card-title', 'Full-Block Mamba-2 Decomposition'),
        el(
          'span',
          'sp-badge sp-badge--teal',
          dec ? `${dec.headLabel} · pos ${dec.localPos}/${dec.blockLength - 1}` : 'boundary',
        ),
      );
      container.append(head);

      if (dec) {
        const cw = dec.convWindow;
        const convRow = el('div', 'sp-pill-row');
        convRow.append(
          el('span', 'sp-badge sp-badge--ctx', `Cache: ${cw.allocatedCacheColumns} cols`),
          el('span', 'sp-badge sp-badge--teal', `Inherited: ${cw.inheritedHistoryColumnCount}`),
          el('span', 'sp-badge sp-badge--den', `Block: ${cw.blockColumnCount}`),
        );
        container.append(convRow);
        container.append(el('p', 'sp-note', cw.cacheConventionNote));

        const tblWrap = el('div', 'sp-mini-table-wrap');
        const tbl = el('table', 'sp-mini-table');
        const tbody = el('tbody');
        const rows: readonly [string, string][] = [
          ['H_prev ‖·‖₂', fmtL2(dec.H_prev)],
          ['dA / Δt', `${dec.decayFactor.toFixed(4)} / ${dec.dtValue.toFixed(4)}`],
          ['write (Δt·x·B) ‖·‖₂', fmtL2(dec.inputWrite)],
          ['H_curr ‖·‖₂', dec.hCurrL2Norm.toFixed(4)],
          ['y_ssm', fmtVecPreview(dec.recurrentReadout)],
          ['D·x (skip)', fmtVecPreview(dec.skipContribution)],
          ['gated (y·SiLU(z))', fmtVecPreview(dec.gatedActivation)],
          ['rms_gate (G' + dec.groupIndex + ')', dec.groupRmsDenominator.toFixed(4)],
          ['y_norm', fmtVecPreview(dec.normalizedOutput)],
          ['Inherited Retention', `‖·‖₂=${dec.inheritedRetentionL2Norm.toFixed(4)} (Π dA=${dec.cumulativeDecay.toFixed(4)})`],
          ['Block Write', `‖·‖₂=${dec.accumulatedBlockWriteL2Norm.toFixed(4)}`],
          ['F32 Residual', `max |sum−H| = ${dec.roundingResidualMaxAbs.toExponential(2)}`],
        ];
        for (const [k, v] of rows) {
          const tr = el('tr');
          tr.append(el('th', undefined, k), el('td', undefined, v));
          tbody.append(tr);
        }
        tbl.append(tbody);
        tblWrap.append(tbl);
        container.append(tblWrap);
      }

      if (d?.causalAfterGqaNote) {
        container.append(el('p', 'sp-note sp-note--callout', d.causalAfterGqaNote));
      }
      if (d?.decaySemanticsDisclaimer) {
        container.append(el('p', 'sp-note', d.decaySemanticsDisclaimer));
      }
      return;
    }

    if (scene.viewMode === 'timestep') {
      const d = scene.diagnostics as TimestepCutawayDiagnostics | undefined;
      const head = el('div', 'sp-card-head');
      head.append(
        el('h2', 'sp-card-title', 'Split AdaLN-Single Conditioning'),
        el(
          'span',
          d?.bypassed ? 'sp-badge sp-badge--muted' : 'sp-badge sp-badge--time',
          d?.bypassed ? 'BYPASSED (t = null)' : `ACTIVE t = ${d?.tModel?.toFixed(3) ?? '—'}`,
        ),
      );
      container.append(head);

      const dl = el('dl', 'sp-kv-grid');
      addKvCell(dl, 'tModel', d?.tModel === null || d?.tModel === undefined ? 'null (BYPASSED)' : d.tModel.toFixed(4));
      addKvCell(dl, 'Ordering', d?.modulationOrder ?? 'modulate_then_norm');
      addKvCell(dl, 'Equation', d?.modulationOrderEquation ?? '—', true);
      addKvCell(
        dl,
        '‖shift‖ / ‖scale‖ / ‖gate‖',
        d ? `${d.shiftL2.toFixed(3)} / ${d.scaleL2.toFixed(3)} / ${d.gateL2.toFixed(3)}` : '—',
        true,
      );
      container.append(dl);
      container.append(
        el(
          'p',
          'sp-note sp-note--callout',
          d?.outerVsInternalGateNote ??
            'Outer residual gate (d=8) is distinct from Mamba internal SiLU(z) gate (d_inner=64).',
        ),
        el(
          'p',
          'sp-note sp-note--ctx',
          d?.finalNormNote ?? 'Final RMSNorm (norm_f) is strictly UNMODULATED in both towers.',
        ),
      );
      return;
    }

    if (scene.viewMode === 'moe') {
      const d = scene.diagnostics as TwoTowerMoeDiagnostics | undefined;
      const head = el('div', 'sp-card-head');
      head.append(
        el('h2', 'sp-card-title', 'Granular MoE FP64 Margin & Top-6'),
        el('span', 'sp-badge sp-badge--teal', '0 B Stateless'),
      );
      container.append(head);

      const pills = el('div', 'sp-pill-row');
      pills.append(
        el(
          'span',
          'sp-badge sp-badge--ctx',
          d?.routerPrecisionPolicy === 'float64_reference' ? 'FP64 Router' : 'FP32 Router',
        ),
        el(
          'span',
          d?.modulationStatus === 'active' ? 'sp-badge sp-badge--den' : 'sp-badge sp-badge--muted',
          `Post-Norm AdaLN: ${d?.modulationStatus ?? 'bypassed'}`,
        ),
      );
      if (d?.partialDenominatorActive && d.runningDenominator !== null) {
        pills.append(
          el(
            'span',
            'sp-badge sp-badge--time',
            `Running Σσ = ${d.runningDenominator.toFixed(4)} (partial)`,
          ),
        );
      }
      container.append(pills);

      // Partial-Denominator-Safe Top-6 Sidebar Weights (STRICTLY uses d.sidebarWeights!)
      if (d && d.sidebarWeights.length > 0) {
        const wWrap = el('div');
        wWrap.dataset.testid = 'moe-sidebar-weights';
        wWrap.append(
          el(
            'div',
            'sp-note',
            d.partialDenominatorActive
              ? 'Top-6 Normalized Weights (accumulating Σσ — final weights masked):'
              : 'Top-6 Normalized Weights (×2.5):',
          ),
        );
        const wGrid = el('div', 'sp-weight-grid');
        for (const entry of d.sidebarWeights) {
          const chip = el(
            'div',
            `sp-weight-chip${
              entry.state === 'accumulating'
                ? ' is-accumulating'
                : entry.state === 'uncomputed'
                  ? ' is-uncomputed'
                  : ''
            }`,
          );
          chip.append(el('span', undefined, entry.expertTag), el('span', undefined, entry.displayText));
          wGrid.append(chip);
        }
        wWrap.append(wGrid);
        container.append(wWrap);
      }

      // Exact FP64 6th-vs-7th Biased-Score Selection Margin
      const m = d?.selectionMargin;
      if (m) {
        const mDl = el('dl', 'sp-kv-grid');
        addKvCell(
          mDl,
          'Rank 6 (Selected)',
          `${formatExpertId(m.rank6ExpertId)} (${m.rank6ChoiceScore.toFixed(4)})`,
        );
        addKvCell(
          mDl,
          'Rank 7 (Cutoff)',
          `${formatExpertId(m.rank7ExpertId)} (${m.rank7ChoiceScore.toFixed(4)})`,
        );
        addKvCell(
          mDl,
          '6th-vs-7th Margin Δ',
          `${m.choiceScoreMargin.toFixed(5)}${m.isTightMargin ? ' (TIGHT)' : ''}`,
          true,
        );
        container.append(mDl);

        const tblWrap = el('div', 'sp-mini-table-wrap');
        const tbl = el('table', 'sp-mini-table');
        const thead = el('thead');
        const hr = el('tr');
        for (const h of ['#', 'Expert', 'Choice (FP64)', 'Unbiased', 'Bias']) {
          hr.append(el('th', undefined, h));
        }
        thead.append(hr);
        const tbody = el('tbody');
        for (const row of m.top8Summary) {
          const tr = el(
            'tr',
            `${row.cutoffBoundaryAfter ? 'is-cutoff' : ''}${!row.selected ? ' is-unselected' : ''}`.trim(),
          );
          tr.append(
            el('td', undefined, String(row.rank)),
            el('td', undefined, row.expertTag),
            el('td', undefined, row.choiceScore.toFixed(4)),
            el('td', undefined, row.unbiasedScore.toFixed(4)),
            el('td', undefined, `${row.bias >= 0 ? '+' : ''}${row.bias.toFixed(3)}`),
          );
          tbody.append(tr);
        }
        tbl.append(thead, tbody);
        tblWrap.append(tbl);
        container.append(tblWrap);
      }
      return;
    }

    if (scene.viewMode === 'embedding') {
      const d = scene.diagnostics as TwoTowerEmbeddingDiagnostics | undefined;
      const head = el('div', 'sp-card-head');
      head.append(
        el('h2', 'sp-card-title', 'Independent Untied Embeddings (NoPE)'),
        el('span', towerBadgeClass(d?.towerRole ?? 'context'), d?.weightTableName ?? 'W_embed'),
      );
      container.append(head);

      const dl = el('dl', 'sp-kv-grid');
      addKvCell(dl, 'Tower Fingerprint', d?.towerParameterFingerprint ?? '—');
      addKvCell(dl, 'Other Tower FP', d?.otherTowerParameterFingerprint ?? '—');
      addKvCell(dl, 'Vocab / Special', `V=${d?.vocabSize ?? 34} ([MASK]=#${d?.maskTokenId ?? 32}, <eos>=#${d?.eosTokenId ?? 33})`, true);
      addKvCell(
        dl,
        'Active Token Range',
        d ? `pos [${d.absoluteTokenRange[0]}..${d.absoluteTokenRange[1]}] (${d.tokenCount} tok)` : '—',
        true,
      );
      addKvCell(
        dl,
        '[MASK] Positions',
        d && d.maskTokenAbsolutePositions.length > 0
          ? d.maskTokenAbsolutePositions.map((p) => `t${p}`).join(', ')
          : 'none in slice',
      );
      addKvCell(
        dl,
        'Untied / NoPE',
        'Untied from W_lm_head · Zero RoPE',
      );
      container.append(dl);
      return;
    }

    if (scene.viewMode === 'output') {
      const d = scene.diagnostics as TwoTowerOutputDiagnostics | undefined;
      const head = el('div', 'sp-card-head');
      head.append(
        el('h2', 'sp-card-title', 'Output LM Head & MDLM Sampler'),
        el(
          'span',
          d?.predictionSemantics === 'unshifted_block_diffusion'
            ? 'sp-badge sp-badge--den'
            : 'sp-badge sp-badge--ctx',
          d?.predictionSemantics === 'unshifted_block_diffusion'
            ? 'Unshifted Block Logits'
            : 'Shifted Causal AR',
        ),
      );
      container.append(head);

      const dl = el('dl', 'sp-kv-grid');
      addKvCell(dl, 'Weight Table', d?.weightTableName ?? 'W_lm_head');
      addKvCell(dl, 'Final norm_f', 'Strictly UNMODULATED');
      addKvCell(dl, 'Semantics', d?.predictionSemanticsLabel ?? '—', true);
      container.append(dl);

      const pres = d?.samplerPresentation;
      if (pres) {
        const meta = el('div', 'sp-pill-row');
        meta.append(
          el('span', 'sp-badge sp-badge--time', `step ${pres.stepIndex + 1}/${pres.stepsPerBlock} · t=${pres.tModel.toFixed(3)}`),
          el('span', 'sp-badge sp-badge--emerald', `reason=${pres.commitReason}`),
        );
        container.append(meta);

        const tblWrap = el('div', 'sp-mini-table-wrap');
        tblWrap.dataset.testid = 'spatial-sampler-table';
        const tbl = el('table', 'sp-mini-table');
        const thead = el('thead');
        const hr = el('tr');
        for (const h of ['Pos', 'Raw Top-1', 'MDLM ([MASK]=0)', 'Selected', 'Confidence', 'Decision']) {
          hr.append(el('th', undefined, h));
        }
        thead.append(hr);
        const tbody = el('tbody');
        for (const slot of pres.slots) {
          const tr = el('tr');
          const topMdlm = slot.constrainedMdlm.topK[0];
          tr.append(
            el('td', undefined, `t${slot.absolutePosition ?? slot.slotIndex}`),
            el(
              'td',
              undefined,
              `${slot.rawLmHead.top1TokenLabel} (${(slot.rawLmHead.top1Prob * 100).toFixed(1)}%)`,
            ),
            el(
              'td',
              undefined,
              topMdlm
                ? `${topMdlm.tokenLabel} (${(topMdlm.prob * 100).toFixed(1)}%) · [MASK]=0.0000`
                : '[MASK]=0.0000',
            ),
            el('td', undefined, slot.selection.selectedTokenLabel),
            el('td', undefined, slot.confidenceDisplay.displayText),
            el('td', undefined, slot.decision),
          );
          tbody.append(tr);
        }
        tbl.append(thead, tbody);
        tblWrap.append(tbl);
        container.append(tblWrap);
      } else if (d && d.top5AtFocus.length > 0) {
        const tblWrap = el('div', 'sp-mini-table-wrap');
        const tbl = el('table', 'sp-mini-table');
        const thead = el('thead');
        const hr = el('tr');
        for (const h of ['Rank', 'Token', 'Logit', 'Prob']) hr.append(el('th', undefined, h));
        thead.append(hr);
        const tbody = el('tbody');
        d.top5AtFocus.forEach((pred, idx) => {
          const tr = el('tr');
          tr.append(
            el('td', undefined, `#${idx + 1}`),
            el('td', undefined, `${pred.label} (#${pred.tokenId})`),
            el('td', undefined, pred.logit.toFixed(3)),
            el('td', undefined, `${(pred.prob * 100).toFixed(2)}%`),
          );
          tbody.append(tr);
        });
        tbl.append(thead, tbody);
        tblWrap.append(tbl);
        container.append(tblWrap);
      }
    }
  }

  function publishTelemetry(): void {
    const record = getActiveRecord(session);
    const branch = getActiveBranch(session);
    const mem = measureTwoTowerSessionRetainedBytes(session);
    const scene = currentScene;

    window.__TWOTOWER_TELEMETRY__ = Object.freeze({
      viewMode,
      drawCallsLastFrame,
      lastFrameMs,
      avgFrameMs: rafFramesRendered > 0 ? totalFrameMs / rafFramesRendered : 0,
      atlasUploads,
      currentAtlasDims,
      maxTextureSize,
      retainedSessionBytes: mem.totalRetainedBytes,
      sharedWeightBytes: mem.sharedModelWeightBytes,
      solidCount: scene?.solidCount ?? 0,
      sheetCount: scene?.sheetCount ?? 0,
      lineCount: scene?.lineCount ?? 0,
      ribbonCount: scene?.ribbonCount ?? 0,
      flowTriangleCount: scene?.flowIndices ? Math.floor(scene.flowIndices.length / 3) : 0,
      lifecyclePhase: record.lifecyclePhase,
      committedPrefixLen: record.committedPrefixLen,
      kvCapacity: record.kvCapacity,
      recordIndex: branch.cursorIndex,
      branchId: branch.branchId,
      glErrorCount,
      reducedMotion,
      rafFramesRendered,
    });
  }

  function syncAllUi(): void {
    syncHeaderAndSessionStrip();
    if (currentScene) {
      canvasTitleEl.textContent = currentScene.title;
      canvasSubEl.textContent = currentScene.subtitle;
      renderBottomDock(currentScene);
      renderInspector(currentScene);
    }
    publishTelemetry();
  }

  // ==========================================================================
  // VIEW & SESSION TRANSITIONS
  // ==========================================================================
  function clearDirectCellSelectionForView(): void {
    gqaSelectedCell = null;
    gqaHoveredCell = null;
    mambaSelectedCell = null;
    mambaHoveredCell = null;
    timestepSelectedCell = null;
    moeSelectedCell = null;
    embedActiveCell = null;
    outputActiveCell = null;
  }

  function switchViewMode(nextMode: TwoTowerSpatialViewMode, instantFit = false): void {
    if (nextMode === viewMode && currentScene !== null) return;
    viewMode = nextMode;
    stepperPlaying = false;
    stepperLastTickMs = null;
    stepperStep = null;
    stepperStepProgress = 0;
    stepperTermProgress = 1;
    macroReplayActive = false;
    macroReplayPhase = null;
    macroReplayLastMs = null;
    hoveredBayId = null;
    hoveredTower = null;
    clearDirectCellSelectionForView();
    geometryDirty = true;
    rebuildSceneIfNeeded();
    if (currentScene) {
      flyToPose(computeFitPoseForScene(currentScene), instantFit);
    }
    scheduleDraw();
  }

  function openCutawayForBay(bayId: number, tower: TwoTowerTowerRole): void {
    if (bayId === -1) {
      switchViewMode('timestep');
      return;
    }
    if (bayId === 0) {
      if (tower === 'context' || tower === 'denoiser') embedTowerRole = tower;
      switchViewMode('embedding');
      return;
    }
    if (bayId === 53) {
      if (tower === 'context' || tower === 'denoiser') outputTowerRole = tower;
      switchViewMode('output');
      return;
    }
    const kind = getTwoTowerBayKind(bayId);
    if (kind === '*') {
      gqaBayId = bayId;
      switchViewMode('paired_gqa');
    } else if (kind === 'M') {
      mambaBayId = bayId;
      switchViewMode('paired_mamba');
    } else {
      moeBayId = bayId;
      if (tower === 'context' || tower === 'denoiser') moeTowerRole = tower;
      switchViewMode('moe');
    }
  }

  function applySessionUpdate(next: TwoTowerSession, fromWorkbench = false, note?: string): void {
    session = next;
    // Clamp token indices safely after session changes
    mambaLocalPos = 0;
    timestepLocalPos = 0;
    moeFocusToken = undefined;
    moeFocusExpert = undefined;
    embedFocusToken = undefined;
    outputFocusToken = undefined;
    clearDirectCellSelectionForView();
    if (!fromWorkbench && workbenchCtrl) {
      workbenchCtrl.syncSession(session, note);
    }
    geometryDirty = true;
    rebuildSceneIfNeeded();
    scheduleDraw();
  }

  function runSessionAction(fn: () => TwoTowerSession, note?: string): void {
    try {
      const next = fn();
      applySessionUpdate(next, false, note);
    } catch {
      // Capabilities prevent illegal calls; silently ignore defensive throw
      syncHeaderAndSessionStrip();
    }
  }

  // Wire Session Strip buttons
  modeSelect.addEventListener('change', () => {
    const mode = modeSelect.value as TwoTowerExecutionMode;
    runSessionAction(() => updateTwoTowerSessionConfig(session, { mode }), `Mode set to ${mode}.`);
  });
  spBtnInit.addEventListener('click', () =>
    runSessionAction(() => initTwoTowerDiffusionBlock(session), 'Initialized diffusion block.'),
  );
  spBtnRefine.addEventListener('click', () =>
    runSessionAction(() => runTwoTowerDiffusionRefinement(session), 'Ran Denoiser refinement pass.'),
  );
  spBtnConf.addEventListener('click', () =>
    runSessionAction(() => commitTwoTowerRefinementConfidence(session), 'Committed MDLM confidence step.'),
  );
  spBtnCommit.addEventListener('click', () =>
    runSessionAction(() => commitTwoTowerActiveBlockToContext(session), 'Committed completed block to Context.'),
  );
  spBtnAr.addEventListener('click', () =>
    runSessionAction(() => stepTwoTowerSessionAr(session), 'Executed Context AR step.'),
  );
  spBtnMockAr.addEventListener('click', () =>
    runSessionAction(() => stepTwoTowerSessionMockAr(session), 'Executed Mock-AR step.'),
  );
  spBtnNext.addEventListener('click', () =>
    runSessionAction(() => stepTwoTowerSessionNext(session), 'Executed next step.'),
  );
  spBtnBack.addEventListener('click', () =>
    runSessionAction(() => navigateTwoTowerSessionBack(session), 'Rewound history.'),
  );
  spBtnFwd.addEventListener('click', () =>
    runSessionAction(() => navigateTwoTowerSessionForward(session), 'Advanced history.'),
  );
  spBtnFork.addEventListener('click', () =>
    runSessionAction(() => forkTwoTowerSessionBranch(session), 'Forked branch.'),
  );
  spBranchSelect.addEventListener('change', () => {
    const id = spBranchSelect.value;
    runSessionAction(() => switchTwoTowerSessionBranch(session, id), `Switched to ${id}.`);
  });

  btnResetCamera.addEventListener('click', () => {
    if (currentScene) {
      flyToPose(computeFitPoseForScene(currentScene));
    }
  });

  function setWorkbenchOpen(open: boolean): void {
    workbenchOpen = open;
    workbenchDrawer.hidden = !open;
    btnToggleWorkbench.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
      if (!workbenchCtrl) {
        workbenchCtrl = mountTwoTowerDiagnosticWorkbench(drawerBody, {
          initialSession: session,
          onSessionChange: (nextSession) => {
            applySessionUpdate(nextSession, true);
          },
          onInspectedBayChange: (bayId) => {
            selectedBayId = bayId;
            if (viewMode === 'macro') {
              geometryDirty = true;
              scheduleDraw();
            }
          },
        });
      } else {
        workbenchCtrl.syncSession(session);
      }
      drawerCloseBtn.focus();
    } else {
      btnToggleWorkbench.focus();
    }
  }

  btnToggleWorkbench.addEventListener('click', () => setWorkbenchOpen(!workbenchOpen));
  drawerCloseBtn.addEventListener('click', () => setWorkbenchOpen(false));

  // ==========================================================================
  // RENDER LOOP (STRICT IDLE-RAF GATING & ZERO Z-FIGHTING)
  // ==========================================================================
  const scheduleDraw = () => {
    if (pendingRafId !== null) return;
    pendingRafId = window.requestAnimationFrame(drawFrame);
  };

  function drawFrame(nowMs: number): void {
    pendingRafId = null;
    const t0 = performance.now();

    const measured = measureHostMetrics(viewportHost);
    if (!measured) return;
    currentMetrics = measured;

    const canAnimate = device !== null && !device.isLost;
    let continueAnimating = false;

    if (canAnimate) {
      // 1. Camera fly animation
      if (flyAnim) {
        const t = Math.max(0, Math.min(1, (nowMs - flyAnim.startMs) / flyAnim.durationMs));
        cameraPose = interpolateCameraPose(flyAnim.from, flyAnim.to, t);
        if (t >= 1) flyAnim = null;
        else continueAnimating = true;
      }

      // 2. Macro instructional wave replay (disabled under reducedMotion)
      if (viewMode === 'macro' && macroReplayActive && !reducedMotion) {
        const dtMs =
          macroReplayLastMs === null ? 0 : Math.max(0, Math.min(80, nowMs - macroReplayLastMs));
        macroReplayLastMs = nowMs;
        const nextPhase = (macroReplayPhase ?? 0) + dtMs / 3200;
        if (nextPhase >= 1) {
          macroReplayActive = false;
          macroReplayPhase = null;
          macroReplayLastMs = null;
        } else {
          macroReplayPhase = nextPhase;
          continueAnimating = true;
        }
        geometryDirty = true;
      }

      // 3. Operation stepper playback (GQA / Mamba / MoE, disabled under reducedMotion)
      if (
        stepperPlaying &&
        !reducedMotion &&
        stepperStep !== null &&
        (viewMode === 'paired_gqa' || viewMode === 'paired_mamba' || viewMode === 'moe')
      ) {
        const dtSec =
          stepperLastTickMs === null ? 0 : Math.max(0, Math.min(0.1, (nowMs - stepperLastTickMs) / 1000));
        stepperLastTickMs = nowMs;
        // Advance termProgress across [0..1] over ~1.4s per cell, then advance step
        const nextTerm = stepperTermProgress + dtSec / 1.4;
        if (nextTerm >= 1) {
          stepperTermProgress = 0.05;
          const nextStepP = stepperStepProgress + 0.25;
          if (nextStepP >= 1) {
            stepperStepProgress = 0;
            stepperStep = (stepperStep + 1) % 6;
          } else {
            stepperStepProgress = nextStepP;
          }
        } else {
          stepperTermProgress = nextTerm;
        }
        geometryDirty = true;
        continueAnimating = true;
      }
    }

    rebuildSceneIfNeeded();
    if (!currentScene) return;

    currentSnapshot = buildCameraSnapshot(cameraPose, currentScene.fitBounds, currentMetrics);

    // WebGL2 draw in exact Nano pass order (NO SolidsPass on sheet cutaways -> zero z-fighting!)
    drawCallsLastFrame = 0;
    if (canAnimate && solidsPass && sheetsPass && linesPass && ribbonsPass && flatFlowPass) {
      const fb = device!.ensureSize(currentMetrics);
      device!.beginFrame(fb);

      const useSheets = Boolean(
        currentScene.sheetCount &&
          currentScene.sheetCount > 0 &&
          currentScene.viewMode !== 'macro',
      );
      if (useSheets) {
        sheetsPass.draw(currentSnapshot.viewProj);
        drawCallsLastFrame += 1;
      } else if (currentScene.solidCount > 0) {
        solidsPass.draw(currentSnapshot.viewProj);
        drawCallsLastFrame += 1;
      }

      if (currentScene.lineCount > 0) {
        linesPass.draw(
          currentSnapshot.viewProj,
          fb.vpWidth,
          fb.vpHeight,
          currentMetrics.dpr,
          currentSnapshot.eye,
        );
        drawCallsLastFrame += 1;
      }

      if (currentScene.ribbonCount > 0) {
        ribbonsPass.draw(currentSnapshot.viewProj);
        drawCallsLastFrame += 1;
      }

      if (currentScene.flowIndices && currentScene.flowIndices.length > 0) {
        const pulseActive =
          !reducedMotion &&
          ((viewMode === 'macro' && macroReplayActive) || stepperPlaying);
        flatFlowPass.draw(currentSnapshot.viewProj, pulseActive ? nowMs * 0.001 : -1);
        drawCallsLastFrame += 1;
      }

      device!.endFrame(fb);

      const glErr = device!.gl.getError();
      if (glErr !== device!.gl.NO_ERROR) {
        glErrorCount += 1;
      }
    }

    // Synchronous projected editorial overlay (deduplicated by camera + scene version)
    const { topInsetPx, bottomInsetPx } = getCanvasInsets();
    const overlayKey =
      `${viewMode}|v${sceneVersion}|${currentMetrics.hostWidth.toFixed(1)}x${currentMetrics.hostHeight.toFixed(1)}|` +
      `${cameraPose.target[0].toFixed(3)},${cameraPose.target[1].toFixed(3)},${cameraPose.target[2].toFixed(3)},` +
      `${cameraPose.distance.toFixed(3)},${cameraPose.azimuthDeg.toFixed(2)},${cameraPose.elevationDeg.toFixed(2)}|` +
      `${topInsetPx}:${bottomInsetPx}`;
    if (overlayKey !== lastOverlayKey) {
      lastOverlayKey = overlayKey;
      const activePart =
        viewMode === 'timestep'
          ? timestepHoveredPart ?? timestepSelectedPart
          : viewMode === 'moe'
            ? moeHoveredPart ?? moeSelectedPart
            : viewMode === 'embedding'
              ? embedHoveredPart ?? embedSelectedPart
              : viewMode === 'output'
                ? outputHoveredPart ?? outputSelectedPart
                : null;
      overlay.update({
        scene: currentScene,
        snapshot: currentSnapshot,
        selectedPartId: activePart,
        activeTokens: getActiveRecord(session).committedPrefixLen,
        topInsetPx,
        bottomInsetPx,
      });
    }

    lastFrameMs = performance.now() - t0;
    totalFrameMs += lastFrameMs;
    rafFramesRendered += 1;
    publishTelemetry();
    refreshTelemetryFooter();

    if (continueAnimating && canAnimate) {
      scheduleDraw();
    }
  }

  // ==========================================================================
  // POINTER INTERACTION: ORBIT, PAN, ZOOM & RAY PICKING
  // ==========================================================================
  const pickAtClient = (clientX: number, clientY: number): PickHit | null => {
    if (!currentScene) return null;
    const rect = viewportHost.getBoundingClientRect();
    const metrics = makeViewportMetrics(
      rect.left,
      rect.top,
      rect.width,
      rect.height,
      window.devicePixelRatio || 1,
    );
    const snap = buildCameraSnapshot(cameraPose, currentScene.fitBounds, metrics);
    const ray = clientPointToRay(clientX, clientY, snap);
    return pickMacroBay(ray, currentScene.solidPickBoxes, currentScene.fasciaPickBoxes);
  };

  const activePointers = new Map<number, { x: number; y: number }>();
  let primaryPointerId: number | null = null;
  let dragMode: 'orbit' | 'pan' = 'orbit';
  let downX = 0;
  let downY = 0;
  let lastX = 0;
  let lastY = 0;
  let isDragging = false;
  let cancelClick = false;
  let pinchStartDist = 0;
  let pinchStartCamDist = cameraPose.distance;

  canvas.addEventListener('contextmenu', (ev) => ev.preventDefault());

  canvas.addEventListener('pointerdown', (ev) => {
    if (pendingHoverRafId !== null) {
      window.cancelAnimationFrame(pendingHoverRafId);
      pendingHoverRafId = null;
    }
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
      cancelClick = ev.button !== 0;
    } else if (activePointers.size === 2) {
      flyAnim = null;
      cancelClick = true;
      isDragging = true;
      const pts = Array.from(activePointers.values());
      pinchStartDist = Math.max(1, Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y));
      pinchStartCamDist = cameraPose.distance;
    }
  });

  function resolveMoeTopKForPick(): Int32Array | undefined {
    const view = resolveTwoTowerSessionViewState(session);
    const activeTower =
      (currentScene?.diagnostics as TwoTowerMoeDiagnostics | undefined)?.towerRole ??
      moeTowerRole ??
      'context';
    const trace =
      activeTower === 'denoiser'
        ? view.resolveDenoiserBayTrace(moeBayId)
        : view.resolveContextBayTrace(moeBayId).trace;
    if (!trace || trace.kind !== 'E') return undefined;
    const tok = Math.max(
      0,
      Math.min(
        trace.sim.tokens - 1,
        (currentScene?.diagnostics as TwoTowerMoeDiagnostics | undefined)?.focusToken ??
          trace.sim.tokens - 1,
      ),
    );
    return trace.sim.topk_indices.subarray(tok * trace.sim.topK, (tok + 1) * trace.sim.topK);
  }

  function applyPickSelection(hit: PickHit | null, isClick: boolean): void {
    if (!currentScene) return;

    if (viewMode === 'macro') {
      if (isClick) {
        selectedBayId = hit ? hit.bayId : null;
        selectedTower = hit?.towerRole ?? 'shared';
        if (selectedBayId !== null && selectedBayId >= 1 && selectedBayId <= 52 && workbenchCtrl) {
          workbenchCtrl.setInspectedBay(selectedBayId);
        }
      } else {
        const nextBay = hit ? hit.bayId : null;
        const nextTower = hit?.towerRole ?? null;
        if (nextBay === hoveredBayId && nextTower === hoveredTower) return;
        hoveredBayId = nextBay;
        hoveredTower = nextTower;
      }
      geometryDirty = true;
      scheduleDraw();
      return;
    }

    if (viewMode === 'paired_gqa') {
      const cell = hit ? pickHitToGqaCell(hit, gqaGroup0Head, gqaGroup1Head) : null;
      const wrapped = cell
        ? { tower: (hit?.towerRole === 'context' ? 'context' : 'denoiser') as 'context' | 'denoiser', cell }
        : null;
      if (isClick) {
        const h = inferGqaCellHead(cell);
        if (h !== null) {
          if (h < 16) gqaGroup0Head = h;
          else gqaGroup1Head = h;
        }
        gqaSelectedCell = wrapped;
        gqaHoveredCell = wrapped;
        if (wrapped) {
          stepperPlaying = false;
          stepperStep = null;
        }
      } else {
        gqaHoveredCell = wrapped;
      }
      geometryDirty = true;
      scheduleDraw();
      return;
    }

    if (viewMode === 'paired_mamba') {
      const diag = currentScene.diagnostics as PairedMambaDiagnostics | undefined;
      const cell = hit ? pickHitToMambaCell(hit, mambaHead, diag?.blockLength ?? 4) : null;
      const wrapped = cell
        ? { tower: (hit?.towerRole === 'context' ? 'context' : 'denoiser') as 'context' | 'denoiser', cell }
        : null;
      if (isClick) {
        const h = inferMambaCellHead(cell);
        if (h !== null) mambaHead = h;
        mambaSelectedCell = wrapped;
        mambaHoveredCell = wrapped;
        if (wrapped) {
          stepperPlaying = false;
          stepperStep = null;
        }
      } else {
        mambaHoveredCell = wrapped;
      }
      geometryDirty = true;
      scheduleDraw();
      return;
    }

    if (viewMode === 'timestep') {
      const cell = hit ? pickHitToTimestepCell(hit) : null;
      if (isClick) {
        timestepSelectedPart = hit?.partId ?? null;
        timestepSelectedCell = cell;
        if (cell) stepperStep = null;
      } else {
        if ((hit?.partId ?? null) === timestepHoveredPart) return;
        timestepHoveredPart = hit?.partId ?? null;
      }
      geometryDirty = true;
      scheduleDraw();
      return;
    }

    if (viewMode === 'moe') {
      const diag = currentScene.diagnostics as TwoTowerMoeDiagnostics | undefined;
      const tok = diag?.focusToken ?? 0;
      const exp = diag?.focusExpert ?? 0;
      const topk = resolveMoeTopKForPick();
      const cell = hit ? pickHitToMoeCell(hit, tok, exp, topk) : null;
      if (isClick) {
        moeSelectedPart = hit?.partId ?? null;
        moeSelectedCell = cell;
        const view = resolveTwoTowerSessionViewState(session);
        const activeTower = diag?.towerRole ?? moeTowerRole ?? 'context';
        const trace =
          activeTower === 'denoiser'
            ? view.resolveDenoiserBayTrace(moeBayId)
            : view.resolveContextBayTrace(moeBayId).trace;
        const inferredExp =
          trace && trace.kind === 'E' ? inferMoeCellExpert(cell, trace.sim, exp) : null;
        if (inferredExp !== null) moeFocusExpert = inferredExp;
        else if (cell?.expertId !== undefined) moeFocusExpert = cell.expertId;
        else if (hit?.headIndex !== undefined && hit.headIndex < MOE_SIM_CONFIG.nRoutedExperts) {
          moeFocusExpert = hit.headIndex;
        }
        if (cell?.tokenIndex !== undefined) moeFocusToken = cell.tokenIndex;
        if (cell) {
          stepperPlaying = false;
          stepperStep = null;
        }
      } else {
        if ((hit?.partId ?? null) === moeHoveredPart) return;
        moeHoveredPart = hit?.partId ?? null;
      }
      geometryDirty = true;
      scheduleDraw();
      return;
    }

    if (viewMode === 'embedding') {
      const diag = currentScene.diagnostics as TwoTowerEmbeddingDiagnostics | undefined;
      const tokIds = Int32Array.from(diag?.tokenIds ?? []);
      const cell = hit ? pickHitToEmbeddingCell(hit, tokIds) : null;
      if (isClick) {
        embedSelectedPart = hit?.partId ?? null;
        embedActiveCell = cell;
        if (cell?.tokenIndex !== undefined) embedFocusToken = cell.tokenIndex;
      } else {
        if ((hit?.partId ?? null) === embedHoveredPart) return;
        embedHoveredPart = hit?.partId ?? null;
      }
      geometryDirty = true;
      scheduleDraw();
      return;
    }

    if (viewMode === 'output') {
      const diag = currentScene.diagnostics as TwoTowerOutputDiagnostics | undefined;
      const focusTok = diag?.focusToken ?? 0;
      const cell = hit ? pickHitToOutputCell(hit, focusTok) : null;
      if (isClick) {
        outputSelectedPart = hit?.partId ?? null;
        outputActiveCell = cell;
        outputSelectedCandidateId = parseOutputCandidatePartId(hit?.partId);
        if (cell?.tokenIndex !== undefined) outputFocusToken = cell.tokenIndex;
      } else {
        if ((hit?.partId ?? null) === outputHoveredPart) return;
        outputHoveredPart = hit?.partId ?? null;
      }
      geometryDirty = true;
      scheduleDraw();
    }
  }

  canvas.addEventListener('pointermove', (ev) => {
    if (activePointers.has(ev.pointerId)) {
      activePointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });
    }
    if (activePointers.size === 2) {
      const pts = Array.from(activePointers.values());
      const dist = Math.max(1, Math.hypot(pts[1].x - pts[0].x, pts[1].y - pts[0].y));
      cameraPose = zoomPerspectiveDistance(
        { ...cameraPose, distance: pinchStartCamDist },
        pinchStartDist / dist,
      );
      scheduleDraw();
      return;
    }
    if (primaryPointerId === ev.pointerId && activePointers.size === 1) {
      const total = Math.hypot(ev.clientX - downX, ev.clientY - downY);
      if (!isDragging && total >= DRAG_THRESHOLD_PX) {
        isDragging = true;
        flyAnim = null;
        canvas.style.cursor = 'grabbing';
      }
      if (isDragging && currentSnapshot) {
        const dx = ev.clientX - lastX;
        const dy = ev.clientY - lastY;
        lastX = ev.clientX;
        lastY = ev.clientY;
        cameraPose =
          dragMode === 'pan' || ev.shiftKey
            ? panCameraByCssDelta(cameraPose, currentSnapshot, dx, dy)
            : orbitCameraByCssDelta(cameraPose, dx, dy);
        scheduleDraw();
      }
      return;
    }

    if (activePointers.size === 0 && ev.pointerType !== 'touch') {
      hoverClientX = ev.clientX;
      hoverClientY = ev.clientY;
      if (pendingHoverRafId === null) {
        pendingHoverRafId = window.requestAnimationFrame(() => {
          pendingHoverRafId = null;
          if (activePointers.size > 0 || isDragging) return;
          const hit = pickAtClient(hoverClientX, hoverClientY);
          canvas.style.cursor = hit ? 'pointer' : 'grab';
          applyPickSelection(hit, false);
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
    if (wasPrimary && !cancelled && !isDragging && !cancelClick) {
      const hit = pickAtClient(ev.clientX, ev.clientY);
      applyPickSelection(hit, true);
    }
    if (activePointers.size === 0) {
      primaryPointerId = null;
      isDragging = false;
      cancelClick = false;
      canvas.style.cursor = 'grab';
    }
  };

  canvas.addEventListener('pointerup', (ev) => endPointer(ev, false));
  canvas.addEventListener('pointercancel', (ev) => endPointer(ev, true));
  canvas.addEventListener('pointerleave', () => {
    if (pendingHoverRafId !== null) {
      window.cancelAnimationFrame(pendingHoverRafId);
      pendingHoverRafId = null;
    }
    canvas.style.cursor = 'grab';
    if (
      hoveredBayId !== null ||
      gqaHoveredCell !== null ||
      mambaHoveredCell !== null ||
      timestepHoveredPart !== null ||
      moeHoveredPart !== null ||
      embedHoveredPart !== null ||
      outputHoveredPart !== null
    ) {
      hoveredBayId = null;
      hoveredTower = null;
      gqaHoveredCell = null;
      mambaHoveredCell = null;
      timestepHoveredPart = null;
      moeHoveredPart = null;
      embedHoveredPart = null;
      outputHoveredPart = null;
      geometryDirty = true;
      scheduleDraw();
    }
  });

  canvas.addEventListener('dblclick', (ev) => {
    if (viewMode !== 'macro') return;
    const hit = pickAtClient(ev.clientX, ev.clientY);
    if (hit) {
      openCutawayForBay(hit.bayId, hit.towerRole ?? 'shared');
    }
  });

  canvas.addEventListener(
    'wheel',
    (ev) => {
      ev.preventDefault();
      flyAnim = null;
      const factor = Math.exp(ev.deltaY * 0.0014);
      cameraPose = zoomPerspectiveDistance(cameraPose, factor);
      scheduleDraw();
    },
    { passive: false },
  );

  // ==========================================================================
  // KEYBOARD, REDUCED MOTION, RESIZE & WEBGL CONTEXT RECOVERY
  // ==========================================================================
  window.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && workbenchOpen) {
      ev.preventDefault();
      setWorkbenchOpen(false);
      return;
    }
    // NEVER hijack Space or Enter on any interactive control!
    if ((ev.key === ' ' || ev.key === 'Enter') && isInteractiveControl(ev.target)) {
      return;
    }
    if (isTextInputTarget(ev.target) || ev.metaKey || ev.ctrlKey || ev.altKey) {
      return;
    }
    if (ev.key === 'r' || ev.key === 'R') {
      if (currentScene) {
        ev.preventDefault();
        flyToPose(computeFitPoseForScene(currentScene));
      }
    }
  });

  const handleReducedMotionChange = (matches: boolean) => {
    if (reducedMotion === matches) return;
    reducedMotion = matches;
    if (matches) {
      if (flyAnim) {
        cameraPose = cloneCameraPose(flyAnim.to);
        flyAnim = null;
      }
      stepperPlaying = false;
      stepperLastTickMs = null;
      stepperTermProgress = 1;
      macroReplayActive = false;
      macroReplayPhase = null;
      macroReplayLastMs = null;
    }
    geometryDirty = true;
    rebuildSceneIfNeeded();
    scheduleDraw();
  };

  if (reducedMotionMq) {
    const onChange = (ev: MediaQueryListEvent) => handleReducedMotionChange(ev.matches);
    if (typeof reducedMotionMq.addEventListener === 'function') {
      reducedMotionMq.addEventListener('change', onChange);
    } else if (typeof reducedMotionMq.addListener === 'function') {
      reducedMotionMq.addListener(onChange);
    }
  }

  let lastResizeW = 0;
  let lastResizeH = 0;
  const resizeObserver = new ResizeObserver(() => {
    const next = measureHostMetrics(viewportHost);
    if (!next) return;
    if (Math.abs(next.hostWidth - lastResizeW) < 0.5 && Math.abs(next.hostHeight - lastResizeH) < 0.5) {
      return;
    }
    lastResizeW = next.hostWidth;
    lastResizeH = next.hostHeight;
    currentMetrics = next;
    lastOverlayKey = '';
    if (currentScene && !flyAnim) {
      cameraPose = computeFitPoseForScene(currentScene);
    }
    scheduleDraw();
  });
  resizeObserver.observe(viewportHost);

  canvas.addEventListener('webglcontextlost', (ev) => {
    ev.preventDefault();
    flyAnim = null;
    stepperPlaying = false;
    macroReplayActive = false;
    if (pendingRafId !== null) {
      window.cancelAnimationFrame(pendingRafId);
      pendingRafId = null;
    }
    lastAtlasKey = null;
    if (device) device.markLost(true);
    showFallback('WebGL2 Context Lost', 'Waiting for the browser to restore the graphics context.');
  });

  canvas.addEventListener('webglcontextrestored', () => {
    if (!device) return;
    try {
      solidsPass = new SolidsPass(device.gl);
      sheetsPass = new SheetsPass(device.gl);
      linesPass = new LinesPass(device.gl);
      ribbonsPass = new RibbonsPass(device.gl);
      flatFlowPass = new FlatFlowRibbonsPass(device.gl);
      maxTextureSize = Number(device.gl.getParameter(device.gl.MAX_TEXTURE_SIZE)) || maxTextureSize;
      lastAtlasKey = null;
      lastOverlayKey = '';
      device.markLost(false);
      fallbackEl.hidden = true;
      geometryDirty = true;
      scheduleDraw();
    } catch {
      device.markLost(true);
      showFallback('WebGL2 Context Recovery Failed', 'Could not reinitialize WebGL2 shaders.');
    }
  });

  // ==========================================================================
  // INITIAL BOOT & EXPOSED HANDLE
  // ==========================================================================
  const initialMetrics = measureHostMetrics(viewportHost);
  if (initialMetrics) {
    currentMetrics = initialMetrics;
    lastResizeW = initialMetrics.hostWidth;
    lastResizeH = initialMetrics.hostHeight;
  }
  geometryDirty = true;
  rebuildSceneIfNeeded();
  if (currentScene) {
    cameraPose = computeFitPoseForScene(currentScene);
  }
  drawFrame(performance.now());

  const handle: TwoTowerAppHandle = {
    getState() {
      return {
        viewMode,
        selectedBayId,
        selectedTower,
        towerFocus,
        linkerChannel: activeLinkerChannel,
        stepperStep,
        stepperPlaying,
        stepperStepProgress,
        stepperTermProgress,
        reducedMotion,
        workbenchOpen,
        activeSelection: currentScene?.activeSelection ?? null,
        warningBanner: currentScene?.warningBanner ?? null,
        diagnostics: currentScene?.diagnostics,
      };
    },
    setViewMode(mode: TwoTowerSpatialViewMode) {
      switchViewMode(mode, true);
    },
    setStepperStep(step: number | null) {
      stepperPlaying = false;
      stepperLastTickMs = null;
      stepperStep = step;
      stepperStepProgress = 0;
      clearDirectCellSelectionForView();
      geometryDirty = true;
      rebuildSceneIfNeeded();
      drawFrame(performance.now());
    },
    setStepperTermProgress(progress: number) {
      stepperPlaying = false;
      stepperLastTickMs = null;
      stepperTermProgress = Math.max(0, Math.min(1, progress));
      geometryDirty = true;
      rebuildSceneIfNeeded();
      drawFrame(performance.now());
    },
    selectBay(bayId: number | null, tower: TwoTowerTowerRole = 'shared') {
      selectedBayId = bayId;
      selectedTower = tower;
      geometryDirty = true;
      rebuildSceneIfNeeded();
      drawFrame(performance.now());
    },
    getCanvasClickPointForBay(bayId: number, tower: 'context' | 'denoiser') {
      if (!currentScene || currentScene.viewMode !== 'macro' || !currentSnapshot) return null;
      const world =
        tower === 'context' ? contextBayFrontWorld(bayId) : denoiserBayFrontWorld(bayId);
      const pt = projectWorldToClientCss(world, currentSnapshot);
      return { x: pt.x, y: pt.y, visible: pt.visible };
    },
  };

  window.__TWOTOWER_APP__ = handle;
  return handle;
}
