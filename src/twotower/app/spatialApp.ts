/**
 * Nemotron-Labs-TwoTower Stage TT4 Interactive WebGL2 Spatial Application
 * (`src/twotower/app/spatialApp.ts`)
 *
 * Directly composes:
 *   - Nano's 5 WebGL2 render passes (`RenderDevice`, `SolidsPass`, `SheetsPass`, `LinesPass`,
 *     `RibbonsPass`, `FlatFlowRibbonsPass`) on a studio-white `#F7F8FA` editorial drafting table
 *   - All 7 headless TwoTower spatial scene builders (`src/twotower/scene/index.ts`)
 *   - Projected editorial overlay (`TwoTowerSpatialOverlay` reusing `src/ui/overlay.ts`)
 *   - Stage TT4 Wave 1–4 features:
 *       * Wave 1: Canonical `t = nMasked / B` timestep status, unmistakable historical Denoiser
 *         provenance boundary warnings, `TWOTOWER_MINI_MODEL_INVENTORY` distinction, RB-2 partial-
 *         accumulation non-exposure preservation
 *       * Wave 2: Persistent Nested Generation Timeline dock, segmented editorial right sidebar
 *         (`Inspect` | `Sampler` | `Architecture`), Advanced MDLM Sampler Inspector with
 *         Future vs Historical Config Scope controls, and Production Architectural & Memory
 *         Scaling Readouts (with inline SVG 262,144 max_pos boundary & 8,035 crossover chart)
 *       * Wave 3: Isolated 4-Act / 28-Beat Guided Educational Walkthrough HUD
 *       * Wave 4: On-demand idle RAF sleep, discrete replay/stepper ticks (zero 60Hz full-scene
 *         rebuilds), hover deduplication, cached memory accounting, and rich WebGL2 telemetry on
 *         `window.__TWOTOWER_SPATIAL_DEBUG__`
 *   - Retained Developer Diagnostic Workbench inside a collapsible drawer synced bidirectionally
 *     with the live `TwoTowerSession`.
 *
 * ZERO runtime dependencies. ZERO Nano files modified. Quiet reciprocal navigation link back to Nano.
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
  buildCameraSnapshot,
  CAMERA_FLY_DURATION_MS,
  cloneCameraPose,
  DEFAULT_FOV_DEG,
  fitOpBoundsCamera,
  interpolateCameraPose,
  makeViewportMetrics,
  orbitCameraByCssDelta,
  panCameraByCssDelta,
  projectWorldToClientCss,
  zoomPerspectiveDistance,
  type AABB,
  type CameraPose,
  type CameraSnapshot,
  type ViewportMetrics,
} from '../../render/camera';
import { RenderDevice } from '../../render/device';
import { LinesPass } from '../../render/lines';
import {
  clientPointToRay,
  pickMacroBay,
  type PickHit,
} from '../../render/pick';
import { FlatFlowRibbonsPass, RibbonsPass } from '../../render/ribbons';
import { SheetsPass } from '../../render/sheets';
import { SolidsPass } from '../../render/solids';
import type { EmbeddingCellRef } from '../../scene/embeddingCutaway';
import {
  pickHitToGqaCell,
  pickHitToMambaCell,
  pickHitToMoeCell,
} from '../../scene/hybridPick';
import type { OutputCellRef } from '../../scene/outputCutaway';
import { handleTablistKeydown } from '../../shared/a11y';
import { mountModelSwitcher } from '../../shared/modelSwitcher';
import {
  buildAdvancedSamplerInspectorModel,
  buildTwoTowerArchitecturalReadout,
  buildTwoTowerConfigScopeModel,
  buildTwoTowerTimelineModel,
  commitTwoTowerActiveBlockToContext,
  commitTwoTowerRefinementConfidence,
  createDefaultTwoTowerModel,
  createTwoTowerSession,
  createTwoTowerTourController,
  deleteTwoTowerSessionBranch,
  forkTwoTowerSessionBranch,
  getActiveBranch,
  getActiveRecord,
  getSessionCapabilities,
  initTwoTowerDiffusionBlock,
  measureTwoTowerSessionRetainedBytes,
  navigateTwoTowerSession,
  navigateTwoTowerSessionBack,
  navigateTwoTowerSessionForward,
  restartTwoTowerSession,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  stepTwoTowerSessionNext,
  switchTwoTowerSessionBranch,
  TWOTOWER_CONFIDENCE_PRESETS,
  TWOTOWER_MINI_CONFIG,
  TWOTOWER_SPEC,
  TWOTOWER_TOUR_ACTS,
  updateTwoTowerSessionConfig,
  type SamplerInspectorSlotRow,
  type SamplerInspectorSlotStatus,
  type TimelineNode,
  type TimelineSegment,
  type TwoTowerAdvancedSamplerInspectorModel,
  type TwoTowerArchitecturalReadout,
  type TwoTowerConfigScopeModel,
  type TwoTowerContextLengthScalingSeries,
  type TwoTowerExecutionMode,
  type TwoTowerSession,
  type TwoTowerSessionCapabilities,
  type TwoTowerSessionMemoryBreakdown,
  type TwoTowerSessionRecord,
  type TwoTowerTimelineModel,
  type TwoTowerTourController,
  type TwoTowerTourSnapshot,
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
  resolveNearestBayOfKind,
  resolveTwoTowerSessionViewState,
  TWO_TOWER_MACRO_FIT_BOUNDS,
  TWOTOWER_MINI_MODEL_INVENTORY,
  type PairedGqaDiagnostics,
  type PairedMambaDiagnostics,
  type TimestepCellRef,
  type TimestepCutawayDiagnostics,
  type TwoTowerDenoiserPassLookup,
  type TwoTowerEmbeddingDiagnostics,
  type TwoTowerLinkerChannel,
  type TwoTowerMacroDiagnostics,
  type TwoTowerMoeDiagnostics,
  type TwoTowerOutputDiagnostics,
  type TwoTowerPairedFocus,
  type TwoTowerSelectionIdentity,
  type TwoTowerSessionViewState,
  type TwoTowerSpatialScene,
  type TwoTowerSpatialViewMode,
  type TwoTowerTimestepConditioningMode,
  type TwoTowerTimestepStatus,
  type TwoTowerTowerRole,
} from '../scene/index';
import {
  mountTwoTowerDiagnosticWorkbench,
  type DiagnosticWorkbenchController,
} from './diagnosticUi';
import { TwoTowerSpatialOverlay } from './spatialOverlay';

// ============================================================================
// PUBLIC TELEMETRY, DEBUG & INSPECTION CONTRACTS
// ============================================================================

export type TwoTowerSidebarTab = 'inspect' | 'sampler' | 'architecture';

export interface TwoTowerTelemetrySnapshot {
  readonly viewMode: TwoTowerSpatialViewMode;
  readonly drawCallsLastFrame: number;
  readonly lastFrameMs: number;
  readonly avgFrameMs: number;
  readonly medianFrameMs: number;
  readonly p95FrameMs: number;
  readonly frameSampleCount: number;
  readonly lastRafIntervalMs: number;
  readonly avgRafIntervalMs: number;
  readonly medianRafIntervalMs: number;
  readonly p95RafIntervalMs: number;
  readonly rafIntervalSampleCount: number;
  readonly geometryRebuilds: number;
  readonly atlasUploads: number;
  readonly rafIdlePauses: number;
  readonly rafActive: boolean;
  readonly layoutResizeCount: number;
  readonly glVendor: string;
  readonly glRenderer: string;
  readonly isSoftwareRenderer: boolean;
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
  readonly tourActive: boolean;
  readonly tourBeatIndex: number;
  readonly tourActIndex: number;
  readonly activeSidebarTab: TwoTowerSidebarTab;
}

export interface TwoTowerSpatialDebugHandle extends TwoTowerTelemetrySnapshot {
  resetPerfCounters(): void;
  startTour(): TwoTowerTourSnapshot;
  exitTour(): void;
  goToTourBeat(idx: number): TwoTowerTourSnapshot;
  goToTourAct(actIdx: number): TwoTowerTourSnapshot;
  setSidebarTab(tab: TwoTowerSidebarTab): void;
  jumpToRecord(idx: number): void;
  orbitDelta(dx: number, dy: number): void;
  triggerMacroReplay(on: boolean): void;
  triggerOpStepper(on: boolean): void;
  getTourSnapshot(): TwoTowerTourSnapshot | null;
  getSnapshot(): TwoTowerTelemetrySnapshot;
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
    readonly activeSidebarTab: TwoTowerSidebarTab;
    readonly tourActive: boolean;
    readonly tourBeatIndex: number;
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
  dispose(): void;
}

declare global {
  interface ImportMeta {
    readonly env: {
      readonly BASE_URL: string;
    };
  }
  interface Window {
    __TWOTOWER_TELEMETRY__?: TwoTowerTelemetrySnapshot;
    __TWOTOWER_APP__?: TwoTowerAppHandle;
    __TWOTOWER_SPATIAL_DEBUG__?: TwoTowerSpatialDebugHandle;
  }
}

// ============================================================================
// CONSTANTS & PURE HELPERS
// ============================================================================

const DRAG_THRESHOLD_PX = 5;
const SVG_NS = 'http://www.w3.org/2000/svg';
const MAX_FRAME_SAMPLES = 240;
const MACRO_REPLAY_STAGE_COUNT = 8;
const MACRO_REPLAY_DURATION_MS = 3200;
const STEPPER_SUBTERM_BUCKETS = 6;
const STEPPER_CELL_DURATION_SEC = 1.4;
const TOUR_AUTOPLAY_MS = 6500;
const TELEMETRY_FOOTER_THROTTLE_MS = 250;

const MINI_INVENTORY_SUMMARY =
  `${TWOTOWER_MINI_MODEL_INVENTORY.gqa.queryHeads} Q / ${TWOTOWER_MINI_MODEL_INVENTORY.gqa.kvHeads} KV, ` +
  `d_head=${TWOTOWER_MINI_MODEL_INVENTORY.gqa.headDim} · ` +
  `${TWOTOWER_MINI_MODEL_INVENTORY.mamba.heads} Mamba / ${TWOTOWER_MINI_MODEL_INVENTORY.mamba.groups} groups / ` +
  `${TWOTOWER_MINI_MODEL_INVENTORY.mamba.ssmStateMatrixShape[0]}×${TWOTOWER_MINI_MODEL_INVENTORY.mamba.ssmStateMatrixShape[1]} SSM`;

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

const SIDEBAR_TABS: readonly {
  readonly id: TwoTowerSidebarTab;
  readonly label: string;
  readonly title: string;
}[] = Object.freeze([
  { id: 'inspect', label: 'Inspect', title: 'Spatial Inspector: 10-Axis Provenance, Timestep Status & Formula' },
  { id: 'sampler', label: 'Sampler', title: 'Advanced MDLM Sampler Inspector & Future Config Scope' },
  { id: 'architecture', label: 'Architecture', title: 'Production 63.19B Inventory, Weights vs Caches & Scaling' },
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

function fmtBytesCompact(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${bytes} B`;
}

function fmtParamsBillions(params: number): string {
  if (params >= 1e9) return `${(params / 1e9).toFixed(2)}B`;
  if (params >= 1e6) return `${(params / 1e6).toFixed(2)}M`;
  if (params >= 1e3) return `${(params / 1e3).toFixed(1)}K`;
  return String(params);
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

function computeQuantiles(samples: readonly number[]): {
  readonly avg: number;
  readonly median: number;
  readonly p95: number;
} {
  const n = samples.length;
  if (n === 0) return { avg: 0, median: 0, p95: 0 };
  let sum = 0;
  for (let i = 0; i < n; i++) sum += samples[i];
  const sorted = [...samples].sort((a, b) => a - b);
  const mid = Math.floor(n / 2);
  const median = n % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) * 0.5;
  const p95Idx = Math.min(n - 1, Math.max(0, Math.ceil(n * 0.95) - 1));
  return {
    avg: sum / n,
    median,
    p95: sorted[p95Idx],
  };
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

const MODAL_FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function getModalFocusableElements(container: HTMLElement): HTMLElement[] {
  if (typeof container.querySelectorAll !== 'function') return [];
  const nodes = Array.from(container.querySelectorAll<HTMLElement>(MODAL_FOCUSABLE_SELECTOR));
  return nodes.filter((node) => {
    if (node.hidden) return false;
    if ('disabled' in node && (node as HTMLButtonElement).disabled) return false;
    let cur: HTMLElement | null = node.parentElement;
    while (cur && cur !== container) {
      if (cur.hidden) return false;
      cur = cur.parentElement;
    }
    return true;
  });
}

function handleModalTabTrap(container: HTMLElement, ev: KeyboardEvent): boolean {
  if (ev.key !== 'Tab') return false;
  const focusables = getModalFocusableElements(container);
  if (focusables.length === 0) {
    ev.preventDefault();
    container.focus();
    return true;
  }
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  const activeIdx = document.activeElement
    ? focusables.indexOf(document.activeElement as HTMLElement)
    : -1;
  if (ev.shiftKey) {
    if (activeIdx <= 0) {
      ev.preventDefault();
      last.focus();
      return true;
    }
  } else if (activeIdx === -1 || activeIdx >= focusables.length - 1) {
    ev.preventDefault();
    first.focus();
    return true;
  }
  return false;
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

function conditioningModeLabel(mode: TwoTowerTimestepConditioningMode): string {
  switch (mode) {
    case 'active_diffusion':
      return 'Active Diffusion';
    case 'inactive_no_pass_yet':
      return 'Inactive (No Pass Yet)';
    case 'historical_prior_pass':
      return 'Historical Prior Pass';
    case 'bypassed_ar':
      return 'Bypassed (AR)';
    case 'bypassed_mock_ar':
      return 'Bypassed (Mock-AR)';
  }
}

function conditioningModeBadgeClass(mode: TwoTowerTimestepConditioningMode): string {
  switch (mode) {
    case 'active_diffusion':
      return 'sp-badge sp-badge--time';
    case 'historical_prior_pass':
      return 'sp-badge sp-badge--warn';
    case 'inactive_no_pass_yet':
    case 'bypassed_ar':
    case 'bypassed_mock_ar':
      return 'sp-badge sp-badge--muted';
  }
}

function samplerDecisionChip(status: SamplerInspectorSlotStatus): {
  readonly label: string;
  readonly cls: string;
} {
  switch (status) {
    case 'committed_threshold':
      return { label: 'Threshold', cls: 'sp-badge sp-badge--emerald' };
    case 'committed_budget_floor':
      return { label: 'Budget Floor', cls: 'sp-badge sp-badge--time' };
    case 'committed_final_step':
      return { label: 'Final Step', cls: 'sp-badge sp-badge--ctx' };
    case 'remasked':
      return { label: 'Remasked', cls: 'sp-badge sp-badge--den' };
    case 'locked_prior':
      return { label: 'Locked', cls: 'sp-badge sp-badge--muted' };
  }
}

function tourToneBadgeClass(tone: string): string {
  switch (tone) {
    case 'context':
      return 'sp-badge sp-badge--ctx';
    case 'denoiser':
      return 'sp-badge sp-badge--den';
    case 'shared':
      return 'sp-badge sp-badge--teal';
    case 'gold':
      return 'sp-badge sp-badge--time';
    case 'warn':
      return 'sp-badge sp-badge--warn';
    default:
      return 'sp-badge sp-badge--muted';
  }
}

function formatCompactTimestepPill(ts: TwoTowerTimestepStatus): string {
  const modeLbl = conditioningModeLabel(ts.conditioningMode);
  if (ts.conditioningMode === 'active_diffusion' && ts.executedPass) {
    const ep = ts.executedPass;
    return `t=${ep.tModel.toFixed(2)} (${ep.nMaskedInput}/${ep.blockSize}) · ${modeLbl}`;
  }
  if (ts.conditioningMode === 'historical_prior_pass' && ts.executedPass) {
    const ep = ts.executedPass;
    return `Prior t=${ep.tModel.toFixed(2)} (Rec #${ep.sourceRecordIndex}) · ${modeLbl}`;
  }
  if (ts.conditioningMode === 'inactive_no_pass_yet') {
    if (ts.nextEligibleStep) {
      const ns = ts.nextEligibleStep;
      return `Next t=${ns.nextTModel.toFixed(2)} (${ns.nextNMasked}/${ns.blockSize}) · ${modeLbl}`;
    }
    return modeLbl;
  }
  return `t=null · ${modeLbl}`;
}

function sameGqaHover(
  a: { readonly tower: 'context' | 'denoiser'; readonly cell: GqaCellRef } | null,
  b: { readonly tower: 'context' | 'denoiser'; readonly cell: GqaCellRef } | null,
): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.tower === b.tower &&
    a.cell.tensor === b.cell.tensor &&
    a.cell.row === b.cell.row &&
    a.cell.col === b.cell.col &&
    a.cell.headIndex === b.cell.headIndex &&
    a.cell.kvHeadIndex === b.cell.kvHeadIndex
  );
}

function sameMambaHover(
  a: { readonly tower: 'context' | 'denoiser'; readonly cell: MambaCellRef } | null,
  b: { readonly tower: 'context' | 'denoiser'; readonly cell: MambaCellRef } | null,
): boolean {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    a.tower === b.tower &&
    a.cell.tensor === b.cell.tensor &&
    a.cell.row === b.cell.row &&
    a.cell.col === b.cell.col &&
    a.cell.headIndex === b.cell.headIndex &&
    a.cell.groupIndex === b.cell.groupIndex
  );
}

function buildCapabilityTooltips(
  session: TwoTowerSession,
  record: TwoTowerSessionRecord,
  caps: TwoTowerSessionCapabilities,
) {
  const autoFork = caps.willAutoForkOnMutatingStep ? ' (rewound: will auto-fork branch)' : '';
  const term = caps.isTerminated
    ? `Disabled: branch terminated (${caps.terminationReason}). Rewind, fork, or restart.`
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
              ? 'Disabled: block has 0 masks remaining — click Commit → Context.'
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

interface PreTourExplorationState {
  readonly viewMode: TwoTowerSpatialViewMode;
  readonly selectedBayId: number | null;
  readonly selectedTower: TwoTowerTowerRole;
  readonly towerFocus: TwoTowerPairedFocus;
  readonly activeLinkerChannel: TwoTowerLinkerChannel;
  readonly gqaBayId: number;
  readonly mambaBayId: number;
  readonly mambaLocalPos: number;
  readonly timestepBayId: number;
  readonly timestepLocalPos: number;
  readonly moeBayId: number;
  readonly moeTowerRole: 'context' | 'denoiser' | undefined;
  readonly embedTowerRole: 'context' | 'denoiser' | undefined;
  readonly outputTowerRole: 'context' | 'denoiser' | undefined;
  readonly outputFocusToken: number | undefined;
  readonly stepperStep: number | null;
  readonly activeSidebarTab: TwoTowerSidebarTab;
  readonly cameraPose: CameraPose;
}

// ============================================================================
// CONTEXT-LENGTH MEMORY SCALING INLINE SVG CHART BUILDER
// ============================================================================

function buildContextLengthScalingSvg(series: TwoTowerContextLengthScalingSeries): SVGSVGElement {
  const W = 316;
  const H = 192;
  const padL = 48;
  const padR = 10;
  const padT = 24;
  const padB = 30;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const minLogX = Math.log2(1_024); // 10
  const maxLogX = Math.log2(1_048_576); // 20
  const minLogY = Math.log10(1 * 1024 * 1024); // 1 MiB
  const maxLogY = Math.log10(100 * 1024 * 1024 * 1024); // 100 GiB

  const xOf = (tokens: number): number => {
    const clamped = Math.max(1_024, Math.min(1_048_576, tokens));
    const u = (Math.log2(clamped) - minLogX) / (maxLogX - minLogX);
    return padL + u * plotW;
  };

  const yOf = (bytes: number): number => {
    const safe = Math.max(1 * 1024 * 1024, Math.min(100 * 1024 * 1024 * 1024, bytes));
    const v = (Math.log10(safe) - minLogY) / (maxLogY - minLogY);
    return padT + (1 - v) * plotH;
  };

  const svg = svgEl('svg', {
    class: 'sp-scaling-svg',
    viewBox: `0 0 ${W} ${H}`,
    role: 'img',
    'aria-label':
      '3-regime context-length memory scaling chart from 1K to 1M tokens with 131,072 (128K) operational boundary, 262,144 (256K) configured positional limit boundary, and 8,035 Mamba/KV crossover in binary MiB/GiB',
  });

  const operationalX = xOf(series.modelCardOperationalMaxTokens); // 131,072 (128K)
  const maxPosX = xOf(series.maxPositionEmbeddings); // 262,144 (256K)
  const crossoverX = xOf(series.crossoverTokens); // 8,035
  const mambaY = yOf(series.fixedMambaBytes); // 47.08 MiB

  // 3 explicit shaded background regimes: Operational (≤128K), Configured Limit (≤256K), Extrapolated (>256K)
  svg.append(
    svgEl('rect', {
      x: padL,
      y: padT,
      width: Math.max(0, operationalX - padL),
      height: plotH,
      fill: '#F0FDF4',
      'data-testid': 'scaling-zone-operational',
    }),
    svgEl('rect', {
      x: operationalX,
      y: padT,
      width: Math.max(0, maxPosX - operationalX),
      height: plotH,
      fill: '#FFFBEB',
      'data-testid': 'scaling-zone-configured-limit',
    }),
    svgEl('rect', {
      x: maxPosX,
      y: padT,
      width: Math.max(0, padL + plotW - maxPosX),
      height: plotH,
      fill: '#FFF1F2',
      'data-testid': 'scaling-zone-extrapolated',
    }),
  );

  // Top regime labels (staggered vertically so all 3 regimes stay legible with zero overlap)
  svg.append(
    svgEl(
      'text',
      {
        x: (padL + (operationalX - padL) * 0.5).toFixed(1),
        y: (padT - 5).toFixed(1),
        'text-anchor': 'middle',
        class: 'sp-svg-regime sp-svg-regime--ok',
      },
      'Operational (≤128K)',
    ),
    svgEl(
      'text',
      {
        x: (operationalX + (maxPosX - operationalX) * 0.5).toFixed(1),
        y: (padT - 13).toFixed(1),
        'text-anchor': 'middle',
        class: 'sp-svg-regime sp-svg-regime--limit',
      },
      'Config Limit',
    ),
    svgEl(
      'text',
      {
        x: (padL + plotW).toFixed(1),
        y: (padT - 5).toFixed(1),
        'text-anchor': 'end',
        class: 'sp-svg-regime sp-svg-regime--extrap',
      },
      'Extrapolated',
    ),
  );

  // Horizontal log Y gridlines & explicit binary IEC labels (1 MiB..100 GiB)
  const yTicks: readonly [number, string][] = [
    [1 * 1024 * 1024, '1 MiB'],
    [10 * 1024 * 1024, '10 MiB'],
    [100 * 1024 * 1024, '100 MiB'],
    [1 * 1024 * 1024 * 1024, '1 GiB'],
    [10 * 1024 * 1024 * 1024, '10 GiB'],
    [100 * 1024 * 1024 * 1024, '100 GiB'],
  ];
  for (const [b, lbl] of yTicks) {
    const yy = yOf(b);
    svg.append(
      svgEl('line', {
        x1: padL,
        y1: yy.toFixed(1),
        x2: padL + plotW,
        y2: yy.toFixed(1),
        stroke: '#E2E8F0',
        'stroke-width': 0.8,
      }),
      svgEl(
        'text',
        {
          x: padL - 4,
          y: (yy + 2.8).toFixed(1),
          'text-anchor': 'end',
          class: 'sp-svg-axis',
        },
        lbl,
      ),
    );
  }

  // X ticks across standard context lengths (explicitly including 128K & 256K)
  const xTicks: readonly [number, string][] = [
    [1_024, '1K'],
    [4_096, '4K'],
    [16_384, '16K'],
    [65_536, '64K'],
    [131_072, '128K'],
    [262_144, '256K'],
    [1_048_576, '1M'],
  ];
  for (const [tok, lbl] of xTicks) {
    const xx = xOf(tok);
    svg.append(
      svgEl('line', {
        x1: xx.toFixed(1),
        y1: padT,
        x2: xx.toFixed(1),
        y2: padT + plotH,
        stroke: '#E2E8F0',
        'stroke-width': 0.6,
      }),
      svgEl(
        'text',
        {
          x: xx.toFixed(1),
          y: H - 14,
          'text-anchor': 'middle',
          class: 'sp-svg-axis',
        },
        lbl,
      ),
    );
  }

  // Standard points >= 1024 for curves
  const curvePts = series.points.filter((p) => p.tokens >= 1_024);
  const pathFor = (getter: (p: TwoTowerContextLengthScalingSeries['points'][number]) => number): string =>
    curvePts
      .map((p, idx) => `${idx === 0 ? 'M' : 'L'}${xOf(p.tokens).toFixed(1)},${yOf(getter(p)).toFixed(1)}`)
      .join(' ');

  // 1. 52-Layer Pure GQA reference (dashed muted slate)
  svg.append(
    svgEl('path', {
      d: pathFor((p) => p.reference52LayerGqaKvBytes),
      fill: 'none',
      stroke: '#94A3B8',
      'stroke-width': 1.25,
      'stroke-dasharray': '3 2.5',
    }),
  );

  // 2. Fixed 47.08 MiB Mamba-2 state (teal horizontal)
  svg.append(
    svgEl('line', {
      x1: padL,
      y1: mambaY.toFixed(1),
      x2: padL + plotW,
      y2: mambaY.toFixed(1),
      stroke: '#0F766E',
      'stroke-width': 1.6,
    }),
  );

  // 3. Growing GQA KV cache (cobalt blue)
  svg.append(
    svgEl('path', {
      d: pathFor((p) => p.persistentKvBytes),
      fill: 'none',
      stroke: '#2563EB',
      'stroke-width': 1.6,
    }),
  );

  // 4. Hybrid Total Persistent Context Cache (navy)
  svg.append(
    svgEl('path', {
      d: pathFor((p) => p.persistentTotalBytes),
      fill: 'none',
      stroke: '#1B2340',
      'stroke-width': 2.1,
    }),
  );

  // Vertical dashed boundary 1: 131,072 (128K model-card operational ceiling)
  svg.append(
    svgEl('line', {
      x1: operationalX.toFixed(1),
      y1: padT,
      x2: operationalX.toFixed(1),
      y2: padT + plotH,
      stroke: '#0F766E',
      'stroke-width': 1.35,
      'stroke-dasharray': '3 2.5',
      'data-testid': 'scaling-operational-128k-boundary',
    }),
    svgEl(
      'text',
      {
        x: (operationalX - 3).toFixed(1),
        y: (padT + 10).toFixed(1),
        'text-anchor': 'end',
        class: 'sp-svg-callout sp-svg-callout--teal sp-svg-callout--halo',
      },
      '128K operational',
    ),
  );

  // Vertical dashed boundary 2: 262,144 (256K config.json max_position_embeddings limit)
  svg.append(
    svgEl('line', {
      x1: maxPosX.toFixed(1),
      y1: padT,
      x2: maxPosX.toFixed(1),
      y2: padT + plotH,
      stroke: '#B45309',
      'stroke-width': 1.5,
      'stroke-dasharray': '4 2.5',
      'data-testid': 'scaling-max-pos-boundary',
    }),
    svgEl(
      'text',
      {
        x: (maxPosX - 3).toFixed(1),
        y: (padT + 21).toFixed(1),
        'text-anchor': 'end',
        class: 'sp-svg-callout sp-svg-callout--amber sp-svg-callout--halo',
      },
      '256K max_pos limit',
    ),
  );

  // Explicit 8,035-token Mamba/KV crossover marker
  svg.append(
    svgEl('line', {
      x1: crossoverX.toFixed(1),
      y1: mambaY.toFixed(1),
      x2: crossoverX.toFixed(1),
      y2: padT + plotH,
      stroke: '#DC4C34',
      'stroke-width': 1.1,
      'stroke-dasharray': '2 2',
    }),
    svgEl('circle', {
      cx: crossoverX.toFixed(1),
      cy: mambaY.toFixed(1),
      r: 3.4,
      fill: '#DC4C34',
      stroke: '#FFFFFF',
      'stroke-width': 1.2,
      'data-testid': 'scaling-crossover-marker',
    }),
    svgEl(
      'text',
      {
        x: (crossoverX + 4).toFixed(1),
        y: (mambaY + 11).toFixed(1),
        'text-anchor': 'start',
        class: 'sp-svg-callout sp-svg-callout--coral sp-svg-callout--halo',
      },
      `${series.crossoverTokens.toLocaleString('en-US')} crossover`,
    ),
  );

  // Outer plot border
  svg.append(
    svgEl('rect', {
      x: padL,
      y: padT,
      width: plotW,
      height: plotH,
      fill: 'none',
      stroke: '#CBD5E1',
      'stroke-width': 1,
    }),
  );

  // Bottom axis caption (explicit binary MiB/GiB disclosure)
  svg.append(
    svgEl(
      'text',
      {
        x: (padL + plotW * 0.5).toFixed(1),
        y: (H - 3).toFixed(1),
        'text-anchor': 'middle',
        class: 'sp-svg-axis-caption',
      },
      'Context Prefix (log₂ tokens) · Persistent Cache Size (binary MiB/GiB, log₁₀)',
    ),
  );

  return svg;
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
  let activeSidebarTab: TwoTowerSidebarTab = 'inspect';
  let mobileInspectorOpen = false;
  let selectedBayId: number | null = null;
  let selectedTower: TwoTowerTowerRole = 'shared';
  let hoveredBayId: number | null = null;
  let hoveredTower: TwoTowerTowerRole | null = null;
  let activeLinkerChannel: TwoTowerLinkerChannel = 'all';

  // Discrete macro replay state (avoids 60Hz full-scene rebuilds!)
  let macroReplayActive = false;
  let macroReplayElapsedMs = 0;
  let macroReplayStage = -1;
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

  // Stepper & partial-denominator accumulation playback (discrete sub-term ticks!)
  let stepperStep: number | null = null;
  let stepperPlaying = false;
  let stepperStepProgress = 0;
  let stepperTermProgress = 1;
  let stepperAccumSec = 0;
  let stepperLastTickMs: number | null = null;

  // Guided Tour state (strictly isolated session, never mutates user exploration `session`!)
  let tourController: TwoTowerTourController | null = null;
  let tourActive = false;
  let tourSnapshot: TwoTowerTourSnapshot | null = null;
  let preTourState: PreTourExplorationState | null = null;
  let tourAutoplayTimerId: number | null = null;

  // Drawer state
  let workbenchOpen = false;
  let workbenchCtrl: DiagnosticWorkbenchController | null = null;

  // Renderer, camera & rich Wave 4 performance telemetry state
  let currentScene: TwoTowerSpatialScene | null = null;
  let currentMetrics: ViewportMetrics = makeViewportMetrics(
    0,
    0,
    960,
    680,
    window.devicePixelRatio || 1,
  );
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
  let geometryRebuilds = 0;
  let atlasUploads = 0;
  let currentAtlasDims: readonly [number, number] | null = null;
  let maxTextureSize = 0;
  let glVendor = 'unavailable';
  let glRenderer = 'unavailable';
  let isSoftwareRenderer = false;
  let drawCallsLastFrame = 0;
  let lastFrameMs = 0;
  let totalFrameMs = 0;
  const frameTimesMs: number[] = [];
  let prevContinuousRafMs: number | null = null;
  let lastRafIntervalMs = 0;
  const rafIntervalsMs: number[] = [];
  let rafFramesRendered = 0;
  let rafIdlePauses = 0;
  let rafActive = false;
  let layoutResizeCount = 0;
  let glErrorCount = 0;
  let lastOverlayKey = '';
  let lastFooterRefreshMs = 0;
  let disposed = false;
  let cachedMemBreakdown: TwoTowerSessionMemoryBreakdown | null = null;
  let cachedMemSessionRef: TwoTowerSession | null = null;

  function getActiveDisplaySession(): TwoTowerSession {
    return tourActive && tourController ? tourController.getSession() : session;
  }

  function getCachedMemoryBreakdown(): TwoTowerSessionMemoryBreakdown {
    const active = getActiveDisplaySession();
    if (cachedMemBreakdown === null || cachedMemSessionRef !== active) {
      cachedMemSessionRef = active;
      cachedMemBreakdown = measureTwoTowerSessionRetainedBytes(active);
    }
    return cachedMemBreakdown;
  }

  // ==========================================================================
  // BUILD DOM SHELL
  // ==========================================================================
  root.textContent = '';
  const shell = el('div', 'sp-shell');

  // 1. Top Header / View Bar (~48px)
  const topbar = el('header', 'sp-topbar');
  const brand = el('div', 'sp-brand');
  const titleEl = el('h1', 'sp-title', 'Nemotron-Labs-TwoTower');
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
    handleTablistKeydown(tabButtons, ev, (targetTab) => {
      const nextMode = targetTab.dataset.viewMode as TwoTowerSpatialViewMode;
      switchViewMode(nextMode);
    });
  });

  const topActions = el('div', 'sp-top-actions');

  const btnTourToggle = el('button', 'sp-top-btn sp-top-btn--tour', 'Guided Tour');
  btnTourToggle.type = 'button';
  btnTourToggle.dataset.testid = 'tt-tour-toggle';
  btnTourToggle.setAttribute('aria-pressed', 'false');
  btnTourToggle.title = 'Launch isolated 4-Act / 28-Beat Guided Educational Walkthrough (T)';

  const btnResetCamera = el('button', 'sp-top-btn', 'Reset View');
  btnResetCamera.type = 'button';
  btnResetCamera.dataset.testid = 'btn-reset-camera';
  btnResetCamera.title = 'Fit camera to current view (R)';

  const btnToggleWorkbench = el('button', 'sp-top-btn', 'Workbench');
  btnToggleWorkbench.type = 'button';
  btnToggleWorkbench.dataset.testid = 'btn-toggle-workbench';
  btnToggleWorkbench.setAttribute('aria-expanded', 'false');
  btnToggleWorkbench.setAttribute('aria-controls', 'workbench-drawer');
  btnToggleWorkbench.title = 'Open collapsible Developer Diagnostic Workbench (W)';

  const btnToggleMobileInspector = el('button', 'sp-top-btn sp-mobile-inspector-btn', 'Inspector');
  btnToggleMobileInspector.type = 'button';
  btnToggleMobileInspector.dataset.testid = 'btn-toggle-inspector';
  btnToggleMobileInspector.setAttribute('aria-expanded', 'false');
  btnToggleMobileInspector.setAttribute('aria-controls', 'sp-inspector-drawer');
  btnToggleMobileInspector.title = 'Toggle spatial inspector drawer';

  const baseUrl = (import.meta.env.BASE_URL || '../').replace(/\/?$/, '/');

  topActions.append(
    btnTourToggle,
    btnResetCamera,
    btnToggleWorkbench,
    btnToggleMobileInspector,
  );
  const modelSwitcher = mountModelSwitcher(topActions, {
    activeId: 'twotower',
    baseUrl,
    theme: 'dark',
  });
  topbar.append(brand, tablist, topActions);

  // 2. Compact Session Execution & Instructional Playback Control Strip (~40px)
  const sessionStrip = el('section', 'sp-session-strip');
  sessionStrip.setAttribute('aria-label', 'TwoTower execution actions and status');

  // Left cluster: Execution Actions
  const execCluster = el('div', 'sp-strip-group sp-strip-cluster--exec');
  const execTag = el('span', 'sp-cluster-tag', 'Exec');
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

  const spBtnCommit = el('button', 'sp-btn sp-btn--emerald', 'Commit → Context');
  spBtnCommit.type = 'button';
  spBtnCommit.dataset.testid = 'sp-btn-commit';

  const spBtnAr = el('button', 'sp-btn sp-btn--ctx', 'Step AR');
  spBtnAr.type = 'button';
  spBtnAr.dataset.testid = 'sp-btn-ar';

  const spBtnMockAr = el('button', 'sp-btn sp-btn--ctx', 'Step Mock-AR');
  spBtnMockAr.type = 'button';
  spBtnMockAr.dataset.testid = 'sp-btn-mock-ar';

  const spBtnNext = el('button', 'sp-btn sp-btn--primary', 'Step');
  spBtnNext.type = 'button';
  spBtnNext.dataset.testid = 'sp-btn-next';

  execCluster.append(
    execTag,
    modeSelect,
    spBtnInit,
    spBtnRefine,
    spBtnConf,
    spBtnCommit,
    el('span', 'sp-strip-sep'),
    spBtnAr,
    spBtnMockAr,
    spBtnNext,
  );

  // Right cluster: Live Status Pills + Clearly Separated Instructional Playback Tag
  const statusAndPlaybackCluster = el('div', 'sp-strip-group sp-strip-cluster--right');

  const spPhaseBadge = el('span', 'sp-badge');
  spPhaseBadge.dataset.testid = 'sp-phase-badge';

  const spCapacityBadge = el('span', 'sp-badge sp-badge--ctx');
  spCapacityBadge.dataset.testid = 'sp-capacity-badge';

  const spTimestepBadge = el('span', 'sp-badge sp-badge--muted');
  spTimestepBadge.dataset.testid = 'sp-timestep-badge';

  const playbackCluster = el('div', 'sp-playback-cluster');
  playbackCluster.dataset.testid = 'instructional-playback-cluster';
  const playbackTag = el(
    'span',
    'sp-playback-tag',
    'Instructional replay ≠ measured compute',
  );
  const playbackQuickBtn = el('button', 'sp-btn sp-btn--subtle', 'Macro Flow Replay');
  playbackQuickBtn.type = 'button';
  playbackQuickBtn.dataset.testid = 'sp-btn-instructional-replay';
  playbackCluster.append(playbackTag, playbackQuickBtn);

  statusAndPlaybackCluster.append(
    spPhaseBadge,
    spCapacityBadge,
    spTimestepBadge,
    el('span', 'sp-strip-sep'),
    playbackCluster,
  );
  sessionStrip.append(execCluster, statusAndPlaybackCluster);

  // 3. Main Split Stage
  const stageSplit = el('div', 'sp-stage-split');

  // Left/Center WebGL2 Viewport + Overlay Host
  const viewportHost = el('main', 'sp-viewport-host');
  const canvas = el('canvas');
  canvas.id = 'twotower-canvas';
  canvas.dataset.testid = 'twotower-canvas';
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', 'TwoTower 3D WebGL2 Canvas');

  const overlaySvg = document.createElementNS(SVG_NS, 'svg');
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

  const canvasHistBadge = el('div', 'sp-canvas-hist-badge');
  canvasHistBadge.dataset.testid = 'canvas-historical-badge';
  canvasHistBadge.hidden = true;

  canvasHud.append(canvasTitleBox, canvasHistBadge);

  // Guided Tour Floating Editorial HUD Card (non-occluding bottom-left overlay)
  const tourCard = el('section', 'sp-tour-card');
  tourCard.dataset.testid = 'tt-tour-card';
  tourCard.setAttribute('role', 'region');
  tourCard.setAttribute('aria-label', 'Guided Educational Walkthrough');
  tourCard.hidden = true;

  const tourAriaLive = el('div', 'sr-only');
  tourAriaLive.dataset.testid = 'tour-aria-live';
  tourAriaLive.setAttribute('aria-live', 'polite');
  tourAriaLive.setAttribute('aria-atomic', 'true');

  const bottomDock = el('div', 'sp-bottom-dock');
  bottomDock.dataset.testid = 'view-controls-dock';
  bottomDock.setAttribute('role', 'region');
  bottomDock.setAttribute('aria-label', 'Active view controls');

  const fallbackEl = el('div', 'tt-webgl-fallback');
  fallbackEl.id = 'webgl-fallback';
  fallbackEl.hidden = true;

  viewportHost.append(
    canvas,
    overlaySvg,
    overlayLabels,
    canvasHud,
    tourCard,
    tourAriaLive,
    bottomDock,
    fallbackEl,
  );

  // Right Editorial Inspector Rail with 3 Segmented Tabs (Inspect | Sampler | Architecture)
  const inspector = el('aside', 'sp-inspector');
  inspector.id = 'sp-inspector-drawer';
  inspector.dataset.testid = 'sp-inspector';
  inspector.setAttribute('aria-label', 'TwoTower Spatial Inspector');

  const inspectorMobileBar = el('div', 'sp-inspector-mobile-bar');
  const inspectorMobileTitle = el('span', 'sp-inspector-mobile-title', 'Spatial Inspector');
  const btnCloseMobileInspector = el('button', 'sp-btn sp-btn--subtle', 'Close');
  btnCloseMobileInspector.type = 'button';
  btnCloseMobileInspector.dataset.testid = 'btn-close-mobile-inspector';
  btnCloseMobileInspector.setAttribute('aria-label', 'Close spatial inspector drawer');
  inspectorMobileBar.append(inspectorMobileTitle, btnCloseMobileInspector);

  const sidebarTablist = el('div', 'sp-sidebar-tabs');
  sidebarTablist.setAttribute('role', 'tablist');
  sidebarTablist.setAttribute('aria-label', 'Inspector Sidebar Tabs');
  const sidebarTabButtons: HTMLButtonElement[] = [];

  for (const st of SIDEBAR_TABS) {
    const b = el('button', 'sp-sidebar-tab', st.label);
    b.type = 'button';
    b.id = `sp-sidebar-tab-${st.id}`;
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', st.id === activeSidebarTab ? 'true' : 'false');
    b.setAttribute('aria-controls', `sp-sidebar-panel-${st.id}`);
    b.setAttribute('tabindex', st.id === activeSidebarTab ? '0' : '-1');
    b.title = st.title;
    b.dataset.sidebarTab = st.id;
    b.dataset.testid = `sidebar-tab-${st.id}`;
    b.addEventListener('click', () => setSidebarTab(st.id));
    sidebarTablist.append(b);
    sidebarTabButtons.push(b);
  }

  sidebarTablist.addEventListener('keydown', (ev) => {
    if (
      ev.key !== 'ArrowLeft' &&
      ev.key !== 'ArrowRight' &&
      ev.key !== 'Home' &&
      ev.key !== 'End'
    ) {
      return;
    }
    const curIdx = sidebarTabButtons.findIndex(
      (t) => t === ev.target || t.getAttribute('aria-selected') === 'true',
    );
    const baseIdx = curIdx >= 0 ? curIdx : 0;
    let nextIdx = baseIdx;
    if (ev.key === 'ArrowRight') nextIdx = (baseIdx + 1) % sidebarTabButtons.length;
    else if (ev.key === 'ArrowLeft')
      nextIdx = (baseIdx - 1 + sidebarTabButtons.length) % sidebarTabButtons.length;
    else if (ev.key === 'Home') nextIdx = 0;
    else if (ev.key === 'End') nextIdx = sidebarTabButtons.length - 1;

    ev.preventDefault();
    ev.stopPropagation();
    const targetBtn = sidebarTabButtons[nextIdx];
    setSidebarTab(targetBtn.dataset.sidebarTab as TwoTowerSidebarTab);
    targetBtn.focus();
  });

  const warningBannerEl = el('div', 'sp-warning-banner');
  warningBannerEl.dataset.testid = 'inspector-warning-banner';
  warningBannerEl.hidden = true;

  const historicalDenoiserBannerEl = el('div', 'sp-hist-denoiser-banner');
  historicalDenoiserBannerEl.dataset.testid = 'historical-denoiser-banner';
  historicalDenoiserBannerEl.hidden = true;

  const inspectorScroll = el('div', 'sp-inspector-scroll');

  // Panel 1: Inspect (default visible — preserves all existing E2E selectors!)
  const inspectPanel = el('div', 'sp-tab-panel');
  inspectPanel.id = 'sp-sidebar-panel-inspect';
  inspectPanel.setAttribute('role', 'tabpanel');
  inspectPanel.setAttribute('aria-labelledby', 'sp-sidebar-tab-inspect');
  inspectPanel.dataset.testid = 'sidebar-panel-inspect';

  const historicalConfigCard = el('section', 'sp-card sp-card--muted');
  historicalConfigCard.dataset.testid = 'historical-config-card';

  const timestepStatusCard = el('section', 'sp-card');
  timestepStatusCard.dataset.testid = 'timestep-status-card';

  const selectionIdentityCard = el('section', 'sp-card');
  selectionIdentityCard.dataset.testid = 'selection-identity-card';

  const formulaCard = el('section', 'sp-card');
  formulaCard.dataset.testid = 'formula-card';

  const viewDiagnosticCard = el('section', 'sp-card');
  viewDiagnosticCard.dataset.testid = 'view-diagnostic-card';

  inspectPanel.append(
    historicalConfigCard,
    timestepStatusCard,
    selectionIdentityCard,
    formulaCard,
    viewDiagnosticCard,
  );

  // Panel 2: Advanced Sampler Inspector & Future Config Scope
  const samplerPanel = el('div', 'sp-tab-panel');
  samplerPanel.id = 'sp-sidebar-panel-sampler';
  samplerPanel.setAttribute('role', 'tabpanel');
  samplerPanel.setAttribute('aria-labelledby', 'sp-sidebar-tab-sampler');
  samplerPanel.dataset.testid = 'sidebar-panel-sampler';
  samplerPanel.hidden = true;

  const samplerReadoutMount = el('div', 'sp-panel-stack');
  const samplerConfigMount = el('section', 'sp-card');
  samplerConfigMount.dataset.testid = 'sampler-config-card';
  samplerPanel.append(samplerReadoutMount, samplerConfigMount);

  // Stable interactive controls inside Sampler Config Card (so sliders/inputs never lose focus!)
  const cfgHead = el('div', 'sp-card-head');
  cfgHead.append(
    el('h2', 'sp-card-title', 'Sampler & Block Config Scope'),
    el('span', 'sp-badge sp-badge--ctx', 'Non-Retroactive'),
  );
  const cfgDiffBanner = el('div', 'sp-config-diff-banner');
  cfgDiffBanner.dataset.testid = 'config-scope-diff-banner';
  cfgDiffBanner.hidden = true;

  // Threshold γ controls
  const gammaGroup = el('div', 'sp-cfg-field');
  const gammaTop = el('div', 'sp-cfg-field-head');
  const gammaLabel = el('label', 'sp-cfg-label', 'Confidence Threshold γ');
  gammaLabel.htmlFor = 'sp-sampler-gamma-num';
  const gammaScopeBadge = el('span', 'sp-scope-badge', 'Applies: Next Refinement');
  gammaScopeBadge.dataset.testid = 'scope-badge-gamma';
  gammaTop.append(gammaLabel, gammaScopeBadge);

  const presetRow = el('div', 'sp-preset-row');
  const presetMiniBtn = el('button', 'sp-dock-pill', '0.22 Mini Default');
  presetMiniBtn.type = 'button';
  presetMiniBtn.dataset.testid = 'sampler-preset-mini';
  presetMiniBtn.title = TWOTOWER_CONFIDENCE_PRESETS[0].provenanceNote;
  const presetPaperBtn = el('button', 'sp-dock-pill', '0.80 Paper');
  presetPaperBtn.type = 'button';
  presetPaperBtn.dataset.testid = 'sampler-preset-paper';
  presetPaperBtn.title = TWOTOWER_CONFIDENCE_PRESETS[1].provenanceNote;
  const presetCodeBtn = el('button', 'sp-dock-pill', '0.90 Ref Code');
  presetCodeBtn.type = 'button';
  presetCodeBtn.dataset.testid = 'sampler-preset-code';
  presetCodeBtn.title = TWOTOWER_CONFIDENCE_PRESETS[2].provenanceNote;
  presetRow.append(presetMiniBtn, presetPaperBtn, presetCodeBtn);

  const gammaInputRow = el('div', 'sp-cfg-slider-row');
  const gammaRange = el('input', 'sp-range');
  gammaRange.type = 'range';
  gammaRange.min = '0';
  gammaRange.max = '1';
  gammaRange.step = '0.01';
  gammaRange.setAttribute('aria-label', 'Confidence threshold gamma slider');
  gammaRange.dataset.testid = 'sampler-threshold-slider';
  const gammaNum = el('input', 'sp-num-input');
  gammaNum.id = 'sp-sampler-gamma-num';
  gammaNum.type = 'number';
  gammaNum.min = '0';
  gammaNum.max = '1';
  gammaNum.step = '0.01';
  gammaNum.dataset.testid = 'sampler-threshold-input';
  gammaInputRow.append(gammaRange, gammaNum);
  const gammaDiffNote = el('div', 'sp-cfg-sub');
  gammaGroup.append(gammaTop, presetRow, gammaInputRow, gammaDiffNote);

  // Independent Block Width B (4 / 8)
  const blockBGroup = el('div', 'sp-cfg-field');
  const blockBTop = el('div', 'sp-cfg-field-head');
  const blockBLabel = el('span', 'sp-cfg-label', 'Block Width B');
  const blockBScopeBadge = el('span', 'sp-scope-badge', 'Applies: Next Block');
  blockBScopeBadge.dataset.testid = 'scope-badge-block-size';
  blockBTop.append(blockBLabel, blockBScopeBadge);

  const blockBRow = el('div', 'sp-preset-row');
  const btnB4 = el('button', 'sp-dock-pill', 'B = 4');
  btnB4.type = 'button';
  btnB4.dataset.testid = 'sampler-block-size-4';
  const btnB8 = el('button', 'sp-dock-pill', 'B = 8');
  btnB8.type = 'button';
  btnB8.dataset.testid = 'sampler-block-size-8';
  blockBRow.append(btnB4, btnB8);
  const blockBDiffNote = el('div', 'sp-cfg-sub');
  blockBGroup.append(blockBTop, blockBRow, blockBDiffNote);

  // Independent Refinement Budget K (1..12)
  const stepsKGroup = el('div', 'sp-cfg-field');
  const stepsKTop = el('div', 'sp-cfg-field-head');
  const stepsKLabel = el('label', 'sp-cfg-label', 'Refinement Budget K (1..12)');
  stepsKLabel.htmlFor = 'sp-sampler-k-num';
  const stepsKScopeBadge = el('span', 'sp-scope-badge', 'Applies: Next Block');
  stepsKScopeBadge.dataset.testid = 'scope-badge-steps';
  stepsKTop.append(stepsKLabel, stepsKScopeBadge);

  const stepsKRow = el('div', 'sp-cfg-slider-row');
  const stepsKRange = el('input', 'sp-range');
  stepsKRange.type = 'range';
  stepsKRange.min = '1';
  stepsKRange.max = '12';
  stepsKRange.step = '1';
  stepsKRange.setAttribute('aria-label', 'Refinement budget K slider');
  stepsKRange.dataset.testid = 'sampler-steps-slider';
  const stepsKNum = el('input', 'sp-num-input');
  stepsKNum.id = 'sp-sampler-k-num';
  stepsKNum.type = 'number';
  stepsKNum.min = '1';
  stepsKNum.max = '12';
  stepsKNum.step = '1';
  stepsKNum.dataset.testid = 'sampler-steps-input';
  stepsKRow.append(stepsKRange, stepsKNum);
  const stepsKDiffNote = el('div', 'sp-cfg-sub');
  stepsKGroup.append(stepsKTop, stepsKRow, stepsKDiffNote);

  // Temperature & Seed controls
  const tempSeedGroup = el('div', 'sp-cfg-field');
  const tempSeedTop = el('div', 'sp-cfg-field-head');
  const tempSeedLabel = el('span', 'sp-cfg-label', 'Temperature T & Seed');
  const tempSeedScopeBadge = el('span', 'sp-scope-badge', 'Applies: Next Refinement');
  tempSeedTop.append(tempSeedLabel, tempSeedScopeBadge);

  const tempSeedRow = el('div', 'sp-cfg-pair-row');
  const tempWrap = el('label', 'sp-cfg-inline');
  tempWrap.append(document.createTextNode('T '));
  const tempNum = el('input', 'sp-num-input');
  tempNum.type = 'number';
  tempNum.min = '0';
  tempNum.max = '2';
  tempNum.step = '0.1';
  tempNum.dataset.testid = 'sampler-temp-input';
  tempWrap.append(tempNum);

  const seedWrap = el('label', 'sp-cfg-inline');
  seedWrap.append(document.createTextNode('Seed '));
  const seedNum = el('input', 'sp-num-input sp-num-input--wide');
  seedNum.type = 'number';
  seedNum.step = '1';
  seedNum.dataset.testid = 'sampler-seed-input';
  seedWrap.append(seedNum);
  tempSeedRow.append(tempWrap, seedWrap);
  const tempSeedNote = el('div', 'sp-cfg-sub');
  tempSeedGroup.append(tempSeedTop, tempSeedRow, tempSeedNote);

  // Mode & Restart scope summary footer inside config card
  const scopeLegendRow = el('div', 'sp-pill-row');
  const modeScopeChip = el('span', 'sp-scope-badge', 'Mode · Applies: Next Step');
  const restartScopeChip = el('span', 'sp-scope-badge', 'Prompt/KV · Applies: Session Restart');
  scopeLegendRow.append(modeScopeChip, restartScopeChip);

  samplerConfigMount.append(
    cfgHead,
    cfgDiffBanner,
    gammaGroup,
    blockBGroup,
    stepsKGroup,
    tempSeedGroup,
    scopeLegendRow,
  );

  // Panel 3: Production Architectural & Scaling Readouts
  const archPanel = el('div', 'sp-tab-panel');
  archPanel.id = 'sp-sidebar-panel-architecture';
  archPanel.setAttribute('role', 'tabpanel');
  archPanel.setAttribute('aria-labelledby', 'sp-sidebar-tab-architecture');
  archPanel.dataset.testid = 'sidebar-panel-architecture';
  archPanel.hidden = true;

  inspectorScroll.append(inspectPanel, samplerPanel, archPanel);

  const telemetryFooter = el('footer', 'sp-telemetry-footer');
  telemetryFooter.dataset.testid = 'inspector-telemetry-footer';

  inspector.append(
    inspectorMobileBar,
    sidebarTablist,
    warningBannerEl,
    historicalDenoiserBannerEl,
    inspectorScroll,
    telemetryFooter,
  );

  stageSplit.append(viewportHost, inspector);

  // 4. Persistent Nested Generation Timeline Dock (Bottom Editorial Strip)
  const timelineDock = el('section', 'sp-timeline-dock');
  timelineDock.dataset.testid = 'tt-timeline-dock';
  timelineDock.setAttribute('aria-label', 'Persistent Nested Generation Timeline and Branch Controls');

  const tlControls = el('div', 'sp-tl-controls');
  const spBtnBack = el('button', 'sp-btn', '← Back');
  spBtnBack.type = 'button';
  spBtnBack.dataset.testid = 'sp-btn-back';

  const spBtnFwd = el('button', 'sp-btn', 'Forward →');
  spBtnFwd.type = 'button';
  spBtnFwd.dataset.testid = 'sp-btn-fwd';

  const spBranchSelect = el('select', 'sp-select');
  spBranchSelect.dataset.testid = 'sp-branch-select';
  spBranchSelect.setAttribute('aria-label', 'Active session branch');

  const spBtnFork = el('button', 'sp-btn', 'Fork');
  spBtnFork.type = 'button';
  spBtnFork.dataset.testid = 'sp-btn-fork';

  const spBtnDeleteBranch = el('button', 'sp-btn sp-btn--danger', 'Delete Branch');
  spBtnDeleteBranch.type = 'button';
  spBtnDeleteBranch.dataset.testid = 'sp-btn-delete-branch';

  const spBtnRestart = el('button', 'sp-btn', 'Restart');
  spBtnRestart.type = 'button';
  spBtnRestart.dataset.testid = 'sp-btn-restart';

  const tlRetentionBadge = el('span', 'sp-badge sp-badge--muted sp-tl-retention');
  tlRetentionBadge.dataset.testid = 'tt-timeline-retention';

  tlControls.append(
    spBtnBack,
    spBtnFwd,
    spBranchSelect,
    spBtnFork,
    spBtnDeleteBranch,
    spBtnRestart,
    tlRetentionBadge,
  );

  const tlTrack = el('div', 'sp-tl-track');
  tlTrack.dataset.testid = 'tt-timeline-track';
  tlTrack.setAttribute('role', 'list');
  tlTrack.setAttribute('aria-label', 'Generation milestones');

  timelineDock.append(tlControls, tlTrack);

  // 5. Collapsible Developer Diagnostic Workbench Drawer
  const workbenchDrawer = el('aside', 'tt-workbench-drawer');
  workbenchDrawer.id = 'workbench-drawer';
  workbenchDrawer.dataset.testid = 'workbench-drawer';
  workbenchDrawer.setAttribute('role', 'dialog');
  workbenchDrawer.setAttribute('aria-modal', 'true');
  workbenchDrawer.setAttribute('aria-label', 'Developer Diagnostic Session Workbench');
  workbenchDrawer.tabIndex = -1;
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

  shell.append(topbar, sessionStrip, stageSplit, timelineDock, workbenchDrawer);
  root.append(shell);

  // ==========================================================================
  // INITIALIZE WEBGL2 DEVICE, PASSES, VENDOR TELEMETRY & OVERLAY
  // ==========================================================================
  let device: RenderDevice | null = null;
  let solidsPass: SolidsPass | null = null;
  let sheetsPass: SheetsPass | null = null;
  let linesPass: LinesPass | null = null;
  let ribbonsPass: RibbonsPass | null = null;
  let flatFlowPass: FlatFlowRibbonsPass | null = null;

  const queryGlRendererInfo = (gl: WebGL2RenderingContext) => {
    const dbgExt = gl.getExtension('WEBGL_debug_renderer_info');
    glVendor = dbgExt
      ? String(gl.getParameter(dbgExt.UNMASKED_VENDOR_WEBGL) ?? gl.getParameter(gl.VENDOR) ?? 'unknown')
      : String(gl.getParameter(gl.VENDOR) ?? 'unknown');
    glRenderer = dbgExt
      ? String(gl.getParameter(dbgExt.UNMASKED_RENDERER_WEBGL) ?? gl.getParameter(gl.RENDERER) ?? 'unknown')
      : String(gl.getParameter(gl.RENDERER) ?? 'unknown');
    isSoftwareRenderer = /swiftshader|llvmpipe|softpipe|mesa offscreen|basic render/i.test(glRenderer);
    maxTextureSize = Number(gl.getParameter(gl.MAX_TEXTURE_SIZE)) || 0;
  };

  const showFallback = (heading: string, detail: string) => {
    fallbackEl.innerHTML = '';
    const card = el('div', 'tt-webgl-fallback-card');
    const reloadBtn = el('button', 'sp-btn sp-btn--primary', 'Reload Page');
    reloadBtn.type = 'button';
    reloadBtn.dataset.testid = 'webgl-fallback-reload';
    reloadBtn.addEventListener('click', () => window.location.reload());
    card.append(
      el('h2', undefined, heading),
      el('p', 'sp-note', detail),
      el(
        'p',
        'sp-note',
        'You can still inspect the session, persistent timeline, sampler inspector, architectural readouts, and Developer Workbench drawer.',
      ),
      reloadBtn,
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
    queryGlRendererInfo(device.gl);
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

  function computeFitPoseForScene(
    scene: TwoTowerSpatialScene,
    bounds: AABB = scene.fitBounds,
  ): CameraPose {
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

  function focusOpStepCamera(stepIdx: number, instant = false): void {
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
    flyToPose(pose, instant);
  }

  // ==========================================================================
  // SCENE BUILDING & GPU UPLOAD (WITH ATLAS DEDUPLICATION & REBUILD COUNTER)
  // ==========================================================================
  function buildActiveScene(): TwoTowerSpatialScene {
    const activeSession = getActiveDisplaySession();
    switch (viewMode) {
      case 'macro':
        return buildTwoTowerMacroScene(activeSession, {
          selectedTower,
          selectedBayId,
          hoveredBayId,
          hoveredTower,
          activeLinkerChannel,
          instructionalReplayPhase: macroReplayPhase,
          reducedMotion,
        });
      case 'paired_gqa':
        return buildTwoTowerPairedGqaScene(activeSession, {
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
        return buildTwoTowerPairedMambaScene(activeSession, {
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
        return buildTwoTowerTimestepScene(activeSession, {
          bayId: timestepBayId,
          localTokenIndex: timestepLocalPos,
          selectedPartId: timestepSelectedPart,
          hoveredPartId: timestepHoveredPart,
          selectedCell: timestepSelectedCell,
          stepperStep,
        });
      case 'moe':
        return buildTwoTowerMoeScene(activeSession, {
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
        return buildTwoTowerEmbeddingScene(activeSession, {
          towerRole: embedTowerRole,
          focusToken: embedFocusToken,
          selectedPart: embedSelectedPart,
          hoveredPart: embedHoveredPart,
          activeCell: embedActiveCell,
        });
      case 'output':
        return buildTwoTowerOutputScene(activeSession, {
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
        lastAtlasKey = null;
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
    geometryRebuilds += 1;
    sceneVersion += 1;
    uploadSceneToGpu(currentScene);
    syncAllUi();
  }

  // ==========================================================================
  // SIDEBAR TAB SWITCHING & MOBILE INSPECTOR SHEET
  // ==========================================================================
  function setSidebarTab(tab: TwoTowerSidebarTab): void {
    activeSidebarTab = tab;
    for (const btn of sidebarTabButtons) {
      const isAct = btn.dataset.sidebarTab === tab;
      btn.setAttribute('aria-selected', isAct ? 'true' : 'false');
      btn.setAttribute('tabindex', isAct ? '0' : '-1');
    }
    inspectPanel.hidden = tab !== 'inspect';
    samplerPanel.hidden = tab !== 'sampler';
    archPanel.hidden = tab !== 'architecture';

    // Re-render newly visible tab cleanly
    if (currentScene) {
      const view = resolveTwoTowerSessionViewState(getActiveDisplaySession());
      if (tab === 'sampler') {
        renderSamplerTab(getActiveDisplaySession());
      } else if (tab === 'architecture') {
        renderArchitectureTab(view);
      }
    }
    publishTelemetry();
  }

  function setMobileInspectorOpen(open: boolean, restoreFocus = false): void {
    mobileInspectorOpen = open;
    inspector.classList.toggle('is-mobile-open', open);
    btnToggleMobileInspector.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
      if (btnCloseMobileInspector.offsetParent !== null) {
        btnCloseMobileInspector.focus();
      } else {
        const activeTabBtn =
          sidebarTabButtons.find((b) => b.dataset.sidebarTab === activeSidebarTab) ??
          sidebarTabButtons[0];
        activeTabBtn?.focus();
      }
    } else if (restoreFocus && btnToggleMobileInspector.offsetParent !== null) {
      btnToggleMobileInspector.focus();
    }
  }

  btnToggleMobileInspector.addEventListener('click', () => {
    setMobileInspectorOpen(!mobileInspectorOpen, mobileInspectorOpen);
  });
  btnCloseMobileInspector.addEventListener('click', () => {
    setMobileInspectorOpen(false, true);
  });

  // ==========================================================================
  // UI SYNCHRONIZATION: HEADER, SESSION STRIP, TIMELINE, DOCK, INSPECTOR TABS
  // ==========================================================================
  function syncHeaderAndSessionStrip(view: TwoTowerSessionViewState): void {
    for (const tab of tabButtons) {
      const active = tab.dataset.viewMode === viewMode;
      tab.setAttribute('aria-selected', active ? 'true' : 'false');
      tab.setAttribute('tabindex', active ? '0' : '-1');
    }

    const activeSession = getActiveDisplaySession();
    const record = getActiveRecord(activeSession);
    const caps = getSessionCapabilities(activeSession);
    const tips = buildCapabilityTooltips(activeSession, record, caps);
    const lockedByTour = tourActive;

    modeSelect.value = activeSession.pendingConfig.mode;
    modeSelect.disabled = lockedByTour;

    spBtnInit.disabled = lockedByTour || !caps.canInitDiffusionBlock;
    spBtnInit.title = lockedByTour ? 'Exit Guided Tour to step exploration session' : tips.init;
    spBtnInit.textContent = 'Init Block';

    spBtnRefine.disabled = lockedByTour || !caps.canRunDiffusionRefinement;
    spBtnRefine.title = lockedByTour ? 'Exit Guided Tour to step exploration session' : tips.refine;

    spBtnConf.disabled = lockedByTour || !caps.canCommitRefinementConfidence;
    spBtnConf.title = lockedByTour ? 'Exit Guided Tour to step exploration session' : tips.conf;

    spBtnCommit.disabled = lockedByTour || !caps.canCommitActiveBlockToContext;
    spBtnCommit.title = lockedByTour ? 'Exit Guided Tour to step exploration session' : tips.commit;

    spBtnAr.disabled = lockedByTour || !caps.canStepAr;
    spBtnAr.title = lockedByTour ? 'Exit Guided Tour to step exploration session' : tips.ar;

    spBtnMockAr.disabled = lockedByTour || !caps.canStepMockAr;
    spBtnMockAr.title = lockedByTour ? 'Exit Guided Tour to step exploration session' : tips.mockAr;

    spBtnNext.disabled = lockedByTour || !caps.canStepNext;
    spBtnNext.title = lockedByTour ? 'Exit Guided Tour to step exploration session' : tips.next;

    spPhaseBadge.className = phaseBadgeClass(record.lifecyclePhase);
    spPhaseBadge.textContent = record.lifecyclePhase;

    spCapacityBadge.textContent = `Context ${caps.committedPrefixLen} / ${caps.kvCapacity}`;
    spCapacityBadge.title = `${caps.remainingKvCapacity} free KV slots of ${caps.kvCapacity}`;

    // Wave 1 Timestep & Conditioning Mode pill
    const ts = view.timestepStatus;
    spTimestepBadge.className = conditioningModeBadgeClass(ts.conditioningMode);
    spTimestepBadge.textContent = formatCompactTimestepPill(ts);
    spTimestepBadge.title = `${ts.summaryLabel} — ${ts.blockCompletionNote}`;

    // Instructional Playback quick trigger (visually separated from Exec actions)
    const isMacro = viewMode === 'macro';
    const supportsStepperPlay =
      viewMode === 'paired_gqa' || viewMode === 'paired_mamba' || viewMode === 'moe';
    if (isMacro) {
      playbackQuickBtn.disabled = reducedMotion;
      playbackQuickBtn.textContent = macroReplayActive ? 'Stop Flow Replay' : 'Macro Flow Replay';
      playbackQuickBtn.classList.toggle('is-active', macroReplayActive);
      playbackQuickBtn.setAttribute('aria-pressed', macroReplayActive ? 'true' : 'false');
      playbackQuickBtn.title = reducedMotion
        ? 'Disabled under prefers-reduced-motion: reduce'
        : 'Instructional macro dataflow wave replay (not measured compute)';
    } else if (supportsStepperPlay) {
      playbackQuickBtn.disabled = reducedMotion;
      playbackQuickBtn.textContent = stepperPlaying ? 'Pause Op Stepper' : 'Op Stepper';
      playbackQuickBtn.classList.toggle('is-active', stepperPlaying);
      playbackQuickBtn.setAttribute('aria-pressed', stepperPlaying ? 'true' : 'false');
      playbackQuickBtn.title = reducedMotion
        ? 'Disabled under prefers-reduced-motion: reduce (manual Op steps remain active)'
        : 'Instructional operation stepper playback (not measured compute)';
    } else {
      playbackQuickBtn.disabled = true;
      playbackQuickBtn.textContent = 'Op Stepper';
      playbackQuickBtn.classList.remove('is-active');
      playbackQuickBtn.setAttribute('aria-pressed', 'false');
      playbackQuickBtn.title = 'Switch to Macro, GQA, Mamba, or MoE view for instructional replay';
    }

    btnTourToggle.textContent = tourActive ? 'Exit Tour' : 'Guided Tour';
    btnTourToggle.setAttribute('aria-pressed', tourActive ? 'true' : 'false');
    btnTourToggle.classList.toggle('is-active', tourActive);

    btnToggleWorkbench.disabled = tourActive;
    btnToggleWorkbench.setAttribute('aria-expanded', workbenchOpen ? 'true' : 'false');
    btnToggleWorkbench.title = tourActive
      ? 'Exit Guided Tour to open Developer Diagnostic Workbench'
      : 'Open collapsible Developer Diagnostic Workbench (W)';
  }

  function renderTimelineDock(activeSession: TwoTowerSession): void {
    const timeline: TwoTowerTimelineModel = buildTwoTowerTimelineModel(activeSession);
    const nav = timeline.navigation;
    const rh = timeline.retainedHistory;
    const lockedByTour = tourActive;

    spBtnBack.disabled = lockedByTour || !nav.canBack;
    spBtnBack.title = lockedByTour
      ? 'Guided Tour active — use Tour Prev/Next or Exit Tour'
      : nav.canBack
        ? 'Step backward in active branch history'
        : 'Disabled: at root prefill record (#0)';

    spBtnFwd.disabled = lockedByTour || !nav.canForward;
    spBtnFwd.title = lockedByTour
      ? 'Guided Tour active — use Tour Prev/Next or Exit Tour'
      : nav.canForward
        ? 'Step forward in active branch history'
        : 'Disabled: at branch tail';

    spBtnFork.disabled = lockedByTour || !nav.canFork;
    spBtnFork.title = lockedByTour
      ? 'Guided Tour active — Exit Tour to fork exploration session'
      : nav.canFork
        ? 'Fork a new branch from the current record'
        : 'Disabled: maxBranches < 2';

    spBranchSelect.textContent = '';
    for (const br of timeline.branches) {
      const opt = el(
        'option',
        undefined,
        `${br.branchId} (#${br.cursorIndex}/${br.recordCount - 1})`,
      );
      opt.value = br.branchId;
      spBranchSelect.append(opt);
    }
    spBranchSelect.value = timeline.activeBranchId;
    spBranchSelect.disabled = lockedByTour;

    // Delete Branch: strictly disabled on active branch when no other branch exists!
    const canDeleteInactive = !lockedByTour && nav.canDeleteAnyInactiveBranch;
    spBtnDeleteBranch.disabled = !canDeleteInactive;
    if (canDeleteInactive) {
      const victim = findDeletableBranchId(activeSession);
      spBtnDeleteBranch.title = `Delete inactive branch ${victim ?? ''} (active branch is protected and cannot be deleted)`;
    } else {
      spBtnDeleteBranch.title = lockedByTour
        ? 'Guided Tour active'
        : 'Disabled: active branch cannot be deleted (fork or switch away first)';
    }

    spBtnRestart.disabled = lockedByTour;
    spBtnRestart.title = lockedByTour
      ? 'Exit Guided Tour to restart exploration session'
      : 'Restart session on default 8-token prompt';

    const prunedTag =
      rh.prunedRecordCount > 0 || rh.hasPrunedContextTraces
        ? ` · ${rh.prunedRecordCount} pruned`
        : '';
    tlRetentionBadge.textContent = `Rec ${rh.retainedRecordCount}/${rh.maxRecords} · ${rh.retainedBytesDisplay}${prunedTag}`;
    tlRetentionBadge.title =
      `Unique retained records: ${rh.totalUniqueRecordCount} across ${rh.branchCount}/${rh.maxBranches} branch(es)` +
      (rh.hasPrunedContextTraces
        ? ' · Historical Context execution trace pruned (persistent cache remains authoritative)'
        : '');

    // Render nested milestone nodes
    tlTrack.textContent = '';
    if (tourActive && tourSnapshot) {
      tlTrack.append(
        el(
          'span',
          'sp-badge sp-badge--time sp-tl-tour-pill',
          `Tour Trajectory (${tourSnapshot.provenance.branchId} · Beat ${tourSnapshot.beatIndex + 1}/${tourSnapshot.totalBeats})`,
        ),
      );
    }

    const makeNodeButton = (node: TimelineNode, compactLabel: string, extraCls = ''): HTMLButtonElement => {
      const clsParts = ['sp-tl-node'];
      if (extraCls) clsParts.push(extraCls);
      if (node.isActiveCursor) clsParts.push('is-active');
      else if (node.isAfterCursor) clsParts.push('is-after');
      if (node.isCompletedUncommitted) clsParts.push('is-uncommitted');
      if (node.kind === 'diffusion_block_commit') clsParts.push('is-committed');
      if (node.terminationMarker !== null) clsParts.push('is-stop');

      const btn = el('button', clsParts.join(' '));
      btn.type = 'button';
      btn.dataset.testid = `timeline-node-${node.cursorIndex}`;
      btn.dataset.recordKind = node.kind;
      if (node.isActiveCursor) btn.setAttribute('aria-current', 'step');
      btn.disabled = lockedByTour;
      btn.title = `#${node.recordIndex} ${node.label} · P=${node.contextPrefixLen} · cache ${node.cacheDigestShort}`;

      btn.append(el('span', 'sp-tl-node-label', compactLabel));
      if (node.isCompletedUncommitted) {
        btn.append(el('span', 'sp-tl-chip sp-tl-chip--gold', 'Uncommitted'));
      }
      if (node.terminationMarker === 'eos_terminated') {
        btn.append(el('span', 'sp-tl-chip sp-tl-chip--stop', 'EOS'));
      } else if (node.terminationMarker === 'capacity_exhausted') {
        btn.append(el('span', 'sp-tl-chip sp-tl-chip--stop', 'Cap Exhausted'));
      }

      btn.addEventListener('click', () => {
        if (!tourActive) {
          jumpToRecordIndex(node.cursorIndex);
        }
      });
      return btn;
    };

    for (const seg of timeline.segments) {
      const segEl = renderTimelineSegment(seg, makeNodeButton);
      tlTrack.append(segEl);
    }

    // Auto-scroll active milestone smoothly into view (horizontal only)
    const activeNodeEl = tlTrack.querySelector('.sp-tl-node.is-active') as HTMLElement | null;
    if (activeNodeEl && typeof activeNodeEl.scrollIntoView === 'function') {
      activeNodeEl.scrollIntoView({
        behavior: reducedMotion ? 'auto' : 'smooth',
        inline: 'nearest',
        block: 'nearest',
      });
    }
  }

  function renderTimelineSegment(
    seg: TimelineSegment,
    makeNodeButton: (node: TimelineNode, compactLabel: string, extraCls?: string) => HTMLButtonElement,
  ): HTMLElement {
    const item = el('div', 'sp-tl-seg');
    item.setAttribute('role', 'listitem');

    if (seg.kind !== 'diffusion_block') {
      const node = seg.nodes[0];
      let text = node.shortLabel;
      let cls = 'sp-tl-node--ctx';
      if (seg.kind === 'prefill') {
        text = `Prefill P=${node.contextPrefixLen}`;
        cls = 'sp-tl-node--ctx';
      } else if (seg.kind === 'ar_step') {
        text = `AR #${node.absoluteTokenRange[0]}`;
        cls = 'sp-tl-node--ctx';
      } else {
        text = `Mock-AR #${node.absoluteTokenRange[0]}`;
        cls = 'sp-tl-node--den';
      }
      item.append(makeNodeButton(node, text, cls));
      return item;
    }

    // Grouped Diffusion Block card containing nested child milestones
    const group = el(
      'div',
      `sp-tl-block-group${seg.containsActiveCursor ? ' has-active' : ''}${
        seg.isCompletedUncommitted ? ' is-uncommitted' : ''
      }${seg.isCommittedToContext ? ' is-committed' : ''}`,
    );
    group.dataset.testid = `timeline-block-${seg.blockId}`;

    const head = el('span', 'sp-tl-block-head', `Block #${seg.blockId}`);
    head.title = seg.label;
    group.append(head);

    for (const node of seg.nodes) {
      let compact = node.shortLabel;
      let extra = 'sp-tl-node--child';
      if (node.kind === 'diffusion_block_init') {
        compact = `Init`;
      } else if (node.kind === 'diffusion_refinement') {
        compact = `Eval r${node.refinementIndex ?? 0}`;
        extra += ' sp-tl-node--eval';
      } else if (node.kind === 'diffusion_confidence_commit') {
        compact = `Commit r${node.refinementIndex ?? 0}`;
        extra += ' sp-tl-node--conf';
      } else if (node.kind === 'diffusion_block_commit') {
        compact = `→ Context`;
        extra += ' sp-tl-node--ctx-commit';
      }
      group.append(makeNodeButton(node, compact, extra));
    }

    item.append(group);
    return item;
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
      const isAct = stepperStep === i;
      const b = el('button', `sp-dock-pill${isAct ? ' is-active' : ''}`, `${i}`);
      b.type = 'button';
      b.dataset.testid = `stepper-step-${i}`;
      b.setAttribute('aria-pressed', isAct ? 'true' : 'false');
      b.title = `${st.title}: ${st.summary}`;
      b.addEventListener('click', () => {
        stepperPlaying = false;
        stepperLastTickMs = null;
        stepperAccumSec = 0;
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
    offBtn.setAttribute('aria-pressed', stepperStep === null ? 'true' : 'false');
    offBtn.title = 'Exit operation stepper and show full view';
    offBtn.addEventListener('click', () => {
      stepperPlaying = false;
      stepperLastTickMs = null;
      stepperAccumSec = 0;
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
      playBtn.setAttribute('aria-pressed', stepperPlaying ? 'true' : 'false');
      playBtn.disabled = reducedMotion;
      playBtn.title = reducedMotion
        ? 'Disabled under prefers-reduced-motion: reduce (manual steps remain active)'
        : 'Animate operation steps and term-by-term accumulation (instructional replay ≠ measured compute)';
      playBtn.addEventListener('click', () => {
        toggleOpStepperPlayback(!stepperPlaying);
      });
      group.append(playBtn);
    }
    return group;
  }

  function toggleMacroReplay(on: boolean): void {
    if (reducedMotion && on) return;
    if (!on) {
      macroReplayActive = false;
      macroReplayElapsedMs = 0;
      macroReplayStage = -1;
      macroReplayPhase = null;
      macroReplayLastMs = null;
    } else {
      macroReplayActive = true;
      macroReplayElapsedMs = 0;
      macroReplayStage = 0;
      macroReplayPhase = 0.05;
      macroReplayLastMs = performance.now();
    }
    geometryDirty = true;
    scheduleDraw();
  }

  function toggleOpStepperPlayback(on: boolean): void {
    if (reducedMotion && on) return;
    if (!on) {
      stepperPlaying = false;
      stepperLastTickMs = null;
      stepperAccumSec = 0;
    } else {
      if (stepperStep === null) stepperStep = 0;
      stepperPlaying = true;
      stepperLastTickMs = performance.now();
      stepperAccumSec = 0;
      stepperTermProgress = 1 / STEPPER_SUBTERM_BUCKETS;
    }
    geometryDirty = true;
    scheduleDraw();
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
        const isAct = activeLinkerChannel === id;
        const b = el(
          'button',
          `sp-dock-pill${isAct ? ' is-active' : ''}`,
          lbl,
        );
        b.type = 'button';
        b.dataset.testid = `linker-channel-${id}`;
        b.setAttribute('aria-pressed', isAct ? 'true' : 'false');
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
      replayBtn.setAttribute('aria-pressed', macroReplayActive ? 'true' : 'false');
      replayBtn.disabled = reducedMotion;
      replayBtn.title = reducedMotion
        ? 'Instructional wave replay disabled under prefers-reduced-motion: reduce'
        : 'Instructional visual propagation replay (not measured latency)';
      replayBtn.addEventListener('click', () => {
        toggleMacroReplay(!macroReplayActive);
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
        const isAct = towerFocus === val;
        const b = el('button', `sp-dock-pill${isAct ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `tower-focus-${val}`;
        b.setAttribute('aria-pressed', isAct ? 'true' : 'false');
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
      for (let h = 0; h < TWOTOWER_MINI_MODEL_INVENTORY.gqa.queryHeads; h++) {
        const g = h < 16 ? 0 : 1;
        const opt = el('option', undefined, `Q${String(h).padStart(2, '0')} (KV${g})`);
        opt.value = String(h);
        headSelect.append(opt);
      }
      headSelect.value = String(gqaSelectedCell?.cell.headIndex ?? gqaGroup0Head);
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
        const isAct = towerFocus === val;
        const b = el('button', `sp-dock-pill${isAct ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `tower-focus-${val}`;
        b.setAttribute('aria-pressed', isAct ? 'true' : 'false');
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
      for (let h = 0; h < TWOTOWER_MINI_MODEL_INVENTORY.mamba.heads; h++) {
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
        const isAct = activeTower === role;
        const b = el('button', `sp-dock-pill${isAct ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `moe-tower-${role}`;
        b.setAttribute('aria-pressed', isAct ? 'true' : 'false');
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

      const view = resolveTwoTowerSessionViewState(getActiveDisplaySession());
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
        const isAct = activeTower === role;
        const b = el('button', `sp-dock-pill${isAct ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `embed-tower-${role}`;
        b.setAttribute('aria-pressed', isAct ? 'true' : 'false');
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
        const isAct = activeTower === role;
        const b = el('button', `sp-dock-pill${isAct ? ' is-active' : ''}`, lbl);
        b.type = 'button';
        b.dataset.testid = `output-tower-${role}`;
        b.setAttribute('aria-pressed', isAct ? 'true' : 'false');
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

  function renderHistoricalDenoiserBanner(view: TwoTowerSessionViewState): void {
    const dl: TwoTowerDenoiserPassLookup = view.denoiserLookup;
    const showBanner =
      dl.isHistoricalInspection || (dl.pass !== null && !dl.matchesCurrentContextBoundary);

    if (!showBanner) {
      historicalDenoiserBannerEl.hidden = true;
      historicalDenoiserBannerEl.textContent = '';
      canvasHistBadge.hidden = true;
      canvasHistBadge.textContent = '';
      return;
    }

    const mismatch = !dl.matchesCurrentContextBoundary;
    const srcDigest = dl.sourceContextCacheDigest ? dl.sourceContextCacheDigest.slice(0, 8) : 'n/a';
    const curDigest = view.contextSnapshot.ctxState.stateDigest.slice(0, 8);
    const badgeLabel = mismatch
      ? 'HISTORICAL DENOISER · BOUNDARY MISMATCH'
      : 'HISTORICAL PRIOR PASS';

    // Canvas HUD compact badge
    canvasHistBadge.hidden = false;
    canvasHistBadge.className = `sp-canvas-hist-badge${mismatch ? ' is-mismatch' : ''}`;
    canvasHistBadge.textContent =
      `${badgeLabel} · Rec #${dl.sourceRecordIndex ?? '-'} b${dl.sourceBlockId ?? '-'} r${dl.sourceRefinementIndex ?? '-'} · ` +
      `P=${dl.sourceContextPrefixLen ?? 'n/a'} (cur P=${view.committedPrefixLen}) · ${srcDigest}`;

    // Sidebar crisp compact provenance banner
    historicalDenoiserBannerEl.hidden = false;
    historicalDenoiserBannerEl.className = `sp-hist-denoiser-banner${mismatch ? ' is-mismatch' : ''}`;
    historicalDenoiserBannerEl.textContent = '';
    if (dl.historicalProvenanceWarning) {
      historicalDenoiserBannerEl.title = dl.historicalProvenanceWarning;
    } else {
      historicalDenoiserBannerEl.removeAttribute('title');
    }

    const head = el('div', 'sp-hist-denoiser-head');
    head.append(
      el('span', mismatch ? 'sp-badge sp-badge--warn' : 'sp-badge sp-badge--time', badgeLabel),
    );
    const meta = el('div', 'sp-pill-row');
    meta.append(
      el(
        'span',
        'sp-badge sp-badge--muted',
        `Rec #${dl.sourceRecordIndex ?? '-'} · Blk b${dl.sourceBlockId ?? '-'} · Ref r${dl.sourceRefinementIndex ?? '-'}`,
      ),
      el(
        'span',
        'sp-badge sp-badge--ctx',
        `Source P=${dl.sourceContextPrefixLen ?? 'n/a'} (${srcDigest}) → Cur P=${view.committedPrefixLen} (${curDigest})`,
      ),
    );

    const activeBlk = view.activeBlock;
    const conciseSummary =
      activeBlk !== null && dl.sourceBlockId !== activeBlk.blockId
        ? `Block b${activeBlk.blockId} has not been denoised yet; showing prior Denoiser pass from Rec #${dl.sourceRecordIndex ?? '-'} (P=${dl.sourceContextPrefixLen ?? 'n/a'} ≠ cur P=${view.committedPrefixLen}).`
        : mismatch
          ? `Prior Denoiser KV/Mamba conditioning (P=${dl.sourceContextPrefixLen ?? 'n/a'}, ${srcDigest}) does NOT belong to current Context boundary P=${view.committedPrefixLen} (${curDigest}).`
          : `Showing prior Denoiser pass from Rec #${dl.sourceRecordIndex ?? '-'} at matching Context boundary P=${view.committedPrefixLen}.`;
    const warnText = el('p', 'sp-note', conciseSummary);
    historicalDenoiserBannerEl.append(head, meta, warnText);
  }

  function renderTimestepStatusCard(ts: TwoTowerTimestepStatus): void {
    timestepStatusCard.textContent = '';
    const head = el('div', 'sp-card-head');
    const condBadge = el(
      'span',
      conditioningModeBadgeClass(ts.conditioningMode),
      conditioningModeLabel(ts.conditioningMode),
    );
    condBadge.dataset.testid = 'timestep-conditioning-mode';
    head.append(el('h2', 'sp-card-title', 'Timestep Status (t = nMasked / B)'), condBadge);
    timestepStatusCard.append(head);

    const dl = el('dl', 'sp-kv-grid');
    if (ts.executedPass) {
      const ep = ts.executedPass;
      const overrideNote = ep.isDiagnosticOverride
        ? ` [override; empirical ${ep.nMaskedInput}/${ep.blockSize}]`
        : '';
      addKvCell(
        dl,
        'Executed Pass t',
        `t = ${ep.tModel.toFixed(3)} (${ep.nMaskedInput}/${ep.blockSize} masked · Rec #${ep.sourceRecordIndex} b${ep.sourceBlockId ?? '-'} r${ep.sourceRefinementIndex ?? '-'})${overrideNote}`,
        true,
      );
      addKvCell(dl, 'Executed Pass ID', ep.passExecutionId, true);
    } else {
      addKvCell(dl, 'Executed Pass t', 'None executed', true);
    }

    if (ts.currentBlockMask) {
      const cm = ts.currentBlockMask;
      addKvCell(
        dl,
        'Current Unresolved',
        `${cm.currentUnresolvedCount} / ${cm.blockSize} [MASK] (${(cm.currentMaskFraction * 100).toFixed(0)}%)`,
      );
    } else {
      addKvCell(dl, 'Current Unresolved', 'No active block');
    }

    if (ts.nextEligibleStep) {
      const ns = ts.nextEligibleStep;
      addKvCell(
        dl,
        'Next Eligible t',
        `t = ${ns.nextTModel.toFixed(3)} (${ns.nextNMasked}/${ns.blockSize} · ${ns.remainingSteps} rem)`,
      );
    } else if (
      ts.blockCompletionState === 'completed_uncommitted' ||
      ts.blockCompletionState === 'committed_to_context'
    ) {
      addKvCell(dl, 'Next Eligible t', 'None (complete — no pass required)');
    } else {
      addKvCell(dl, 'Next Eligible t', '—');
    }

    addKvCell(dl, 'Block Completion', ts.blockCompletionState, true);
    timestepStatusCard.append(dl, el('p', 'sp-note', ts.blockCompletionNote));
  }

  function renderInspector(scene: TwoTowerSpatialScene, view: TwoTowerSessionViewState): void {
    const activeSession = getActiveDisplaySession();
    const configScope = buildTwoTowerConfigScopeModel(activeSession);

    // 1. Warning Banner & Unmistakable Historical Denoiser Provenance Banner (strictly deduplicated!)
    renderHistoricalDenoiserBanner(view);
    const historicalBannerShown = !historicalDenoiserBannerEl.hidden;
    let distinctWarningText: string | null = scene.warningBanner ?? null;
    if (distinctWarningText && historicalBannerShown) {
      const histWarn = view.denoiserLookup.historicalProvenanceWarning;
      const distinctParts = distinctWarningText
        .split(' | ')
        .map((part) => part.trim())
        .filter(
          (part) =>
            part.length > 0 &&
            part !== histWarn &&
            !part.startsWith('Historical Denoiser pass inspection:'),
        );
      distinctWarningText = distinctParts.length > 0 ? distinctParts.join(' | ') : null;
    }
    if (distinctWarningText) {
      warningBannerEl.hidden = false;
      warningBannerEl.textContent = distinctWarningText;
    } else {
      warningBannerEl.hidden = true;
      warningBannerEl.textContent = '';
    }

    // 2. Historical Config Guard Card (with Future Config Diff badge)
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
    if (configScope.hasPendingDiffFromHistorical) {
      cfgPills.append(
        el(
          'span',
          'sp-badge sp-badge--time',
          `Pending ≠ Hist (${configScope.diffKeysFromHistorical.join(', ')})`,
        ),
      );
    }
    historicalConfigCard.append(hHead, cfgPills);
    if (view.isHistoricalCursor || configScope.hasPendingDiffFromHistorical) {
      historicalConfigCard.append(el('p', 'sp-note', configScope.badgeText));
    }

    // 3. Wave 1 Canonical Timestep & Conditioning Status Card
    renderTimestepStatusCard(view.timestepStatus);

    // 4. Active Selection Identity Card (ALL 10 AXES!)
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

    // 5. Active Formula & Accumulation Card (RB-2 Partial Accumulation Contract Preserved!)
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
      const trimmedEq = f.equation.trimEnd();
      const trimmedVal = f.valueText.trim();
      const eqAlreadyHasResult =
        trimmedVal.length === 0 ||
        trimmedEq.endsWith(`= ${trimmedVal}`) ||
        trimmedEq.endsWith(`=${trimmedVal}`) ||
        accum?.active === true;
      const eqDisplay = eqAlreadyHasResult ? f.equation : `${f.equation} = ${f.valueText}`;
      formulaCard.append(
        metaRow,
        el('div', 'sp-formula-eq', eqDisplay),
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

    // 6. View-Specific Diagnostic Card (includes TWOTOWER_MINI_MODEL_INVENTORY distinction!)
    viewDiagnosticCard.textContent = '';
    renderViewSpecificDiagnostics(scene, viewDiagnosticCard);

    // 7. Render active secondary tab if currently visible
    if (activeSidebarTab === 'sampler') {
      renderSamplerTab(activeSession);
    } else if (activeSidebarTab === 'architecture') {
      renderArchitectureTab(view);
    } else {
      // Keep Sampler config controls synced silently
      syncSamplerConfigControls(configScope);
    }

    // 8. Refresh telemetry footer immediately on scene/state change
    refreshTelemetryFooter(true);
  }

  function renderSamplerTab(activeSession: TwoTowerSession): void {
    const sm: TwoTowerAdvancedSamplerInspectorModel = buildAdvancedSamplerInspectorModel(activeSession);
    const configScope: TwoTowerConfigScopeModel = buildTwoTowerConfigScopeModel(activeSession);

    samplerReadoutMount.textContent = '';

    // 1. Explicit calibration caveat banner
    const caveat = el('div', 'sp-calibration-banner');
    caveat.dataset.testid = 'sampler-calibration-banner';
    const caveatStrong = el(
      'strong',
      undefined,
      'Unscaled confidence is a heuristic, not calibrated semantic correctness. ',
    );
    caveat.append(caveatStrong, document.createTextNode(sm.calibrationDisclaimer));
    samplerReadoutMount.append(caveat);

    // 2. Summary Metrics Card
    const summaryCard = el('section', 'sp-card');
    summaryCard.dataset.testid = 'sampler-summary-card';
    const sHead = el('div', 'sp-card-head');
    const sourceBadge =
      sm.source === 'current_record'
        ? el('span', 'sp-badge sp-badge--emerald', `Active Commit Rec #${sm.sourceRecordIndex}`)
        : sm.source === 'prior_commit_record'
          ? el('span', 'sp-badge sp-badge--time', `Prior Commit Rec #${sm.sourceRecordIndex}`)
          : el('span', 'sp-badge sp-badge--muted', 'No Commit Yet');
    sHead.append(el('h2', 'sp-card-title', 'MDLM Confidence Unmasking Summary'), sourceBadge);
    summaryCard.append(sHead, el('p', 'sp-note', sm.sourceNote));

    if (sm.hasSamplerCommit && sm.executedTModel && sm.commitmentDecomposition) {
      const et = sm.executedTModel;
      const cd = sm.commitmentDecomposition;

      const dl = el('dl', 'sp-kv-grid');
      addKvCell(
        dl,
        'Executed t = nMasked/B',
        `${et.nMaskedBefore}/${et.blockSize} = ${et.tModel.toFixed(3)}`,
      );
      addKvCell(
        dl,
        'Remaining Steps',
        `${et.remainingSteps} / ${et.stepsPerBlock} (step ${et.stepIndex + 1})`,
      );
      addKvCell(
        dl,
        'Commitment Floor',
        `ceil(${et.nMaskedBefore}/${et.remainingSteps}) = ${et.minCommitFloor}`,
      );
      addKvCell(dl, 'Active γ Preset', sm.threshold.matchedPresetLabel);
      addKvCell(
        dl,
        'Qualified by Confidence',
        `${cd.thresholdQualifiedCount} slot${cd.thresholdQualifiedCount === 1 ? '' : 's'} (> γ=${sm.threshold.value.toFixed(2)})`,
      );
      addKvCell(
        dl,
        'Extra for Budget Floor',
        `${cd.budgetFloorExtraCount} (prog=${cd.progressFloorExtraCount} · bud=${cd.budgetFloorOnlyExtraCount} · fin=${cd.finalStepExtraCount})`,
      );
      addKvCell(dl, 'Remasked', `${cd.remaskedCount} slot${cd.remaskedCount === 1 ? '' : 's'}`);
      addKvCell(dl, 'Prior Locked', `${cd.priorLockedCount} slot${cd.priorLockedCount === 1 ? '' : 's'}`);
      addKvCell(
        dl,
        'Candidate Selection',
        sm.variateMode !== null && sm.variateMode !== 'greedy_none'
          ? `Gumbel-max (${sm.variateMode}, T=${sm.temperature}, seed=${sm.samplingSeed ?? '—'}, digest=${sm.variateDigest?.slice(0, 8) ?? '—'})`
          : `Deterministic Greedy (T=${sm.temperature})`,
        true,
      );
      summaryCard.append(dl);
    } else {
      summaryCard.append(
        el(
          'p',
          'sp-note sp-note--ctx',
          'Run Init Block → Refine → Commit Conf in Mask-Diffusion mode to inspect per-position Raw LM-head vs Constrained MDLM ([MASK]=0%) distributions and confidence thresholds.',
        ),
      );
    }
    samplerReadoutMount.append(summaryCard);

    // 3. Per-Position Interactive Cards (pos 0..B-1)
    if (sm.slots.length > 0) {
      const slotsCard = el('section', 'sp-card');
      slotsCard.dataset.testid = 'sampler-slots-list';
      const slHead = el('div', 'sp-card-head');
      slHead.append(
        el('h2', 'sp-card-title', `Per-Position Decisions (pos 0..${sm.slots.length - 1})`),
        el('span', 'sp-badge sp-badge--ctx', '[MASK] = 0% enforced'),
      );
      slotsCard.append(slHead);

      const list = el('div', 'sp-slot-list');
      for (const slot of sm.slots) {
        list.append(renderSamplerSlotCard(slot, sm.threshold.value));
      }
      slotsCard.append(list);
      samplerReadoutMount.append(slotsCard);
    }

    // 4. Sync Non-Retroactive Config Scope Controls
    syncSamplerConfigControls(configScope);
  }

  function renderSamplerSlotCard(slot: SamplerInspectorSlotRow, gamma: number): HTMLElement {
    const chip = samplerDecisionChip(slot.status);
    const card = el('button', `sp-slot-card sp-slot-card--${slot.status}`);
    card.type = 'button';
    card.dataset.testid = `sampler-slot-${slot.blockPosition}`;
    card.title = 'Click to focus this block position in Mamba/Output cutaways';
    card.addEventListener('click', () => {
      mambaLocalPos = slot.blockPosition;
      timestepLocalPos = slot.blockPosition;
      outputFocusToken = slot.blockPosition;
      geometryDirty = true;
      scheduleDraw();
    });

    const top = el('div', 'sp-slot-top');
    const posTag = el(
      'span',
      'sp-slot-pos',
      `pos ${slot.blockPosition} (t${slot.absolutePosition ?? slot.blockPosition})`,
    );
    const tokFlow = el(
      'span',
      'sp-slot-tok',
      `${slot.inputToken.label} → ${slot.outputToken.label}`,
    );
    const decBadge = el('span', chip.cls, chip.label);
    top.append(posTag, tokFlow, decBadge);

    const confRow = el('div', 'sp-slot-meta');
    const candLabel =
      slot.candidateMode === 'gumbel_max'
        ? `Gumbel${slot.variateDigest ? ` (${slot.variateDigest.slice(0, 8)})` : ''}`
        : 'Greedy';
    confRow.append(el('span', 'sp-badge sp-badge--muted', candLabel));

    if (slot.confidence.kind === 'locked_prior') {
      confRow.append(el('span', 'sp-badge sp-badge--muted', 'Locked (prior step)'));
    } else {
      confRow.append(
        el('span', 'sp-badge sp-badge--ctx', `conf=${slot.confidenceDisplay}`),
        el(
          'span',
          slot.passesStrictThreshold ? 'sp-badge sp-badge--emerald' : 'sp-badge sp-badge--den',
          slot.passesStrictThreshold
            ? `> γ (${gamma.toFixed(2)})`
            : `≤ γ (${gamma.toFixed(2)})`,
        ),
      );
    }

    const rawTop3 = slot.rawLmHeadTop.topK
      .slice(0, 3)
      .map((c) => `${c.tokenLabel} ${(c.prob * 100).toFixed(1)}%`)
      .join(' · ');
    const mdlmTop3 = slot.constrainedMdlmTop.topK
      .slice(0, 3)
      .map((c) => `${c.tokenLabel} ${(c.prob * 100).toFixed(1)}%`)
      .join(' · ');

    const distWrap = el('div', 'sp-slot-dists');
    distWrap.append(
      el(
        'div',
        'sp-slot-dist-line',
        `Raw LM Top-3: ${rawTop3} (raw [MASK]=${(slot.rawLmHeadTop.maskTokenProb * 100).toFixed(1)}%)`,
      ),
      el('div', 'sp-slot-dist-line', `MDLM Top-3 ([MASK]=0%): ${mdlmTop3}`),
    );

    card.append(top, confRow, distWrap);
    return card;
  }

  function syncSamplerConfigControls(scope: TwoTowerConfigScopeModel): void {
    const pending = scope.pendingConfig;
    const locked = tourActive;
    const activeEl = document.activeElement;

    if (scope.hasPendingDiffFromHistorical || scope.hasPendingDiffFromActiveBlock) {
      cfgDiffBanner.hidden = false;
      cfgDiffBanner.textContent = scope.badgeText;
    } else {
      cfgDiffBanner.hidden = true;
      cfgDiffBanner.textContent = '';
    }

    const pGamma = scope.parametersByKey.confidenceThreshold;
    gammaScopeBadge.textContent = `Applies: ${pGamma.scopeBadge}`;
    gammaScopeBadge.classList.toggle('is-diff', pGamma.differsFromHistorical);
    gammaDiffNote.textContent = pGamma.differsFromHistorical
      ? `Historical: ${Number(pGamma.historicalValue).toFixed(2)} → Future pending: ${pending.confidenceThreshold.toFixed(2)}`
      : pGamma.scopeNote;

    const isMiniPreset = Math.abs(pending.confidenceThreshold - 0.22) < 1e-6;
    const isPaperPreset = Math.abs(pending.confidenceThreshold - 0.8) < 1e-6;
    const isCodePreset = Math.abs(pending.confidenceThreshold - 0.9) < 1e-6;
    presetMiniBtn.classList.toggle('is-active', isMiniPreset);
    presetPaperBtn.classList.toggle('is-active', isPaperPreset);
    presetCodeBtn.classList.toggle('is-active', isCodePreset);
    presetMiniBtn.setAttribute('aria-pressed', isMiniPreset ? 'true' : 'false');
    presetPaperBtn.setAttribute('aria-pressed', isPaperPreset ? 'true' : 'false');
    presetCodeBtn.setAttribute('aria-pressed', isCodePreset ? 'true' : 'false');
    presetMiniBtn.disabled = locked;
    presetPaperBtn.disabled = locked;
    presetCodeBtn.disabled = locked;

    gammaRange.disabled = locked;
    gammaNum.disabled = locked;
    if (activeEl !== gammaRange) gammaRange.value = pending.confidenceThreshold.toFixed(2);
    if (activeEl !== gammaNum) gammaNum.value = pending.confidenceThreshold.toFixed(2);

    const pB = scope.parametersByKey.blockSize;
    blockBScopeBadge.textContent = `Applies: ${pB.scopeBadge}`;
    blockBScopeBadge.classList.toggle('is-diff', pB.differsFromHistorical || pB.differsFromActiveBlock);
    btnB4.classList.toggle('is-active', pending.blockSize === 4);
    btnB8.classList.toggle('is-active', pending.blockSize === 8);
    btnB4.setAttribute('aria-pressed', pending.blockSize === 4 ? 'true' : 'false');
    btnB8.setAttribute('aria-pressed', pending.blockSize === 8 ? 'true' : 'false');
    btnB4.disabled = locked;
    btnB8.disabled = locked;
    blockBDiffNote.textContent = pB.differsFromActiveBlock
      ? `Active block frozen at B=${pB.activeBlockValue} · Pending B=${pending.blockSize} applies next block`
      : pB.differsFromHistorical
        ? `Historical B=${pB.historicalValue} → Pending B=${pending.blockSize}`
        : pB.scopeNote;

    const pK = scope.parametersByKey.stepsPerBlock;
    stepsKScopeBadge.textContent = `Applies: ${pK.scopeBadge}`;
    stepsKScopeBadge.classList.toggle('is-diff', pK.differsFromHistorical || pK.differsFromActiveBlock);
    stepsKRange.disabled = locked;
    stepsKNum.disabled = locked;
    if (activeEl !== stepsKRange) stepsKRange.value = String(pending.stepsPerBlock);
    if (activeEl !== stepsKNum) stepsKNum.value = String(pending.stepsPerBlock);
    stepsKDiffNote.textContent = pK.differsFromActiveBlock
      ? `Active block frozen at K=${pK.activeBlockValue} · Pending K=${pending.stepsPerBlock} applies next block`
      : pK.differsFromHistorical
        ? `Historical K=${pK.historicalValue} → Pending K=${pending.stepsPerBlock}`
        : pK.scopeNote;

    const pT = scope.parametersByKey.temperature;
    const pSeed = scope.parametersByKey.samplingSeed;
    tempSeedScopeBadge.classList.toggle(
      'is-diff',
      pT.differsFromHistorical || pSeed.differsFromHistorical,
    );
    tempNum.disabled = locked;
    seedNum.disabled = locked;
    if (activeEl !== tempNum) tempNum.value = String(pending.temperature);
    if (activeEl !== seedNum) seedNum.value = String(pending.samplingSeed);
    tempSeedNote.textContent =
      pending.temperature === 0
        ? 'T = 0 runs deterministic greedy argmax (zero Gumbel noise).'
        : `T = ${pending.temperature} runs seeded Gumbel-max (seed=${pending.samplingSeed}) using unscaled confidence.`;

    const pMode = scope.parametersByKey.mode;
    modeScopeChip.textContent = `Mode (${pending.mode}) · Applies: ${pMode.scopeBadge}`;
    modeScopeChip.classList.toggle('is-diff', pMode.differsFromHistorical);
  }

  function renderArchitectureTab(view: TwoTowerSessionViewState): void {
    const activeBlockSize =
      view.activeBlock?.blockConfig.blockSize ?? view.historicalConfig.blockSize;
    const readout: TwoTowerArchitecturalReadout = buildTwoTowerArchitecturalReadout(
      view.committedPrefixLen,
      activeBlockSize,
      view.modeAtRecord,
    );
    const inv = readout.inventory;
    const wvc = readout.weightsVsCaches;
    const sc = readout.scaling;

    archPanel.textContent = '';

    // 1. Independent Tower Parameter Inventory Card (Dual Decimal GB/MB & Binary GiB/MiB + FP32 Bias Extra)
    const invCard = el('section', 'sp-card');
    invCard.dataset.testid = 'arch-inventory-card';
    const invHead = el('div', 'sp-card-head');
    invHead.append(
      el('h2', 'sp-card-title', 'Independent Dual-Tower Inventory'),
      el('span', 'sp-badge sp-badge--emerald', `${inv.grandTotal.payloadByteDiscrepancy} B audit diff`),
    );
    invCard.append(invHead);

    const invDl = el('dl', 'sp-kv-grid');
    addKvCell(
      invDl,
      'Context Tower',
      `${fmtParamsBillions(inv.contextTower.params)} · ${inv.contextTower.payloadDualDisplay} (${inv.contextTower.tensorCount.toLocaleString('en-US')} tensors)`,
    );
    addKvCell(
      invDl,
      'Denoiser + Time-Cond',
      `${fmtParamsBillions(inv.denoiserBackbone.params)} + ${fmtParamsBillions(inv.denoiserTimeConditioning.params)} · ${inv.denoiserTotal.payloadDualDisplay} (Time-Cond: ${inv.denoiserTimeConditioning.payloadDualDisplay})`,
    );
    addKvCell(
      invDl,
      'Grand Total (24 Shards)',
      `${fmtParamsBillions(inv.grandTotal.params)} · ${inv.grandTotal.payloadDualDisplay} payload · Headers ${inv.grandTotal.headerDualDisplay} · ${inv.grandTotal.payloadByteDiscrepancy} B discrepancy`,
      true,
    );
    const fp32BiasCell = el('div', 'sp-kv-cell sp-kv-cell--full');
    fp32BiasCell.dataset.testid = 'arch-fp32-bias-cell';
    fp32BiasCell.title = inv.grandTotal.fp32BiasSummary;
    fp32BiasCell.append(
      el('dt', undefined, 'FP32 Router Bias Extra'),
      el(
        'dd',
        undefined,
        `${inv.grandTotal.fp32BiasTensorCount} tensors · ${inv.grandTotal.fp32BiasParams.toLocaleString('en-US')} params · ${inv.grandTotal.fp32BiasPayloadBytes.toLocaleString('en-US')} B (+${inv.grandTotal.fp32BiasExtraBytesOverBf16.toLocaleString('en-US')} B vs BF16)`,
      ),
    );
    invDl.append(fp32BiasCell);
    addKvCell(invDl, 'Miniature Model Inventory', MINI_INVENTORY_SUMMARY, true);
    invCard.append(invDl);

    const tcList = el('div', 'sp-tc-list');
    for (const c of inv.denoiserTimeConditioning.components) {
      tcList.append(
        el(
          'div',
          'sp-tc-row',
          `${c.label}: ${fmtParamsBillions(c.params)} (${c.payloadDualDisplay} · ${c.tensorCount} tensors)`,
        ),
      );
    }
    const unitNote = el('p', 'sp-note sp-note--ctx', readout.unitConventionNote);
    unitNote.dataset.testid = 'arch-unit-convention-note';
    invCard.append(tcList, el('p', 'sp-note', inv.grandTotal.auditSummary), unitNote);
    archPanel.append(invCard);

    // 2. "Two Resident Weight Sets ≠ Two Persistent Caches" + 4-Way Memory Separation & Peak Exclusion
    const wvcCard = el('section', 'sp-card');
    wvcCard.dataset.testid = 'arch-weights-vs-caches';
    const wvcHead = el('div', 'sp-card-head');
    wvcHead.append(
      el('h2', 'sp-card-title', wvc.headline),
      el('span', 'sp-badge sp-badge--teal', '1 Persistent Cache'),
    );
    wvcCard.append(wvcHead);

    const schematicNote = el(
      'p',
      'sp-schematic-note',
      'Schematic (not linear to scale) — bar widths distinguish architectural buffer categories rather than linear byte ratios.',
    );
    schematicNote.dataset.testid = 'arch-bars-schematic-note';
    wvcCard.append(schematicNote);

    const makeCompareBar = (
      title: string,
      valueText: string,
      pctWidth: number,
      barCls: string,
      note: string,
      testId?: string,
    ): HTMLDivElement => {
      const box = el('div', 'sp-bar-item');
      if (testId) box.dataset.testid = testId;
      const top = el('div', 'sp-bar-head');
      top.append(el('span', 'sp-bar-title', title), el('span', 'sp-bar-val', valueText));
      const track = el('div', 'sp-bar-track');
      track.setAttribute('aria-hidden', 'true');
      const fill = el('div', `sp-bar-fill ${barCls}`);
      fill.style.width = `${Math.max(4, Math.min(100, pctWidth)).toFixed(1)}%`;
      track.append(fill);
      box.append(top, track, el('div', 'sp-bar-note', note));
      return box;
    };

    const opt = wvc.optionalImplementationSnapshotsAndMirrors;
    const transientKv = wvc.transientDenoiserBlockKv;
    const excl = wvc.totalRuntimeWorkspaceExclusion;

    const optBox = makeCompareBar(
      opt.categoryLabel,
      `mamba_s2: ${opt.mockArMambaS2SnapshotDualDisplay} · Mirror: ${opt.denoiserReadOnlyMirrorDualDisplay}`,
      26,
      'sp-bar-fill--amber',
      'Context-owned optional Mock-AR boundary snapshot and deployment-specific cross-GPU read-only mirror are tracked strictly separately from transient Denoiser block KV.',
      'arch-mem-optional-mirrors',
    );
    const optSubList = el('div', 'sp-mem-sublist');
    const s2Row = el('div', 'sp-mem-subrow');
    s2Row.append(
      el('strong', undefined, `Mock-AR mamba_s2 (Context owner): ${opt.mockArMambaS2SnapshotDualDisplay} — `),
      document.createTextNode(opt.mockArMambaS2Note),
    );
    const mirrorRow = el('div', 'sp-mem-subrow');
    mirrorRow.append(
      el(
        'strong',
        undefined,
        `Cross-GPU Read-Only Mirror (0 B extra single-device): ${opt.denoiserReadOnlyMirrorDualDisplay} — `,
      ),
      document.createTextNode(opt.crossGpuMirrorConditionNote),
    );
    optSubList.append(s2Row, mirrorRow);
    optBox.append(optSubList);

    const exclusionBanner = el('div', 'sp-exclusion-banner');
    exclusionBanner.dataset.testid = 'arch-peak-exclusion-banner';
    const exclHead = el('div', 'sp-exclusion-head');
    exclHead.append(
      el('span', 'sp-exclusion-title', `4. ${excl.headline}`),
      el('span', 'sp-badge sp-badge--warn', 'Excluded from Accounted Buffers'),
    );
    const exclList = el('ul', 'sp-exclusion-list');
    for (const comp of excl.excludedComponents) {
      const li = el('li');
      li.append(el('strong', undefined, `${comp.label}: `), document.createTextNode(comp.detail));
      exclList.append(li);
    }
    exclusionBanner.append(exclHead, el('p', 'sp-note', excl.summary), exclList);

    wvcCard.append(
      makeCompareBar(
        `Resident Weights (${wvc.residentWeights.activeTowerCount} active / 2 resident)`,
        `${wvc.residentWeights.contextTowerDualDisplay} + ${wvc.residentWeights.denoiserTowerDualDisplay}`,
        100,
        'sp-bar-fill--navy',
        wvc.residentWeights.note,
      ),
      makeCompareBar(
        wvc.persistentContextCache.categoryLabel,
        `${wvc.persistentContextCache.fixedMambaDualDisplay} Mamba + ${wvc.persistentContextCache.kvPerTokenDisplay}/tok KV = ${wvc.persistentContextCache.totalDualDisplay}`,
        42,
        'sp-bar-fill--ctx',
        wvc.persistentContextCache.note,
        'arch-mem-persistent-cache',
      ),
      optBox,
      makeCompareBar(
        `${transientKv.categoryLabel} (Ephemeral Denoiser Scratchpad)`,
        transientKv.ephemeralDenoiserBlockKvDualDisplay,
        12,
        'sp-bar-fill--den',
        transientKv.note,
        'arch-mem-transient-kv',
      ),
      exclusionBanner,
    );
    archPanel.append(wvcCard);

    // 3. 3-Regime Context-Length Memory Scaling Inline SVG Chart (1K..1M, 128K operational, 256K max_pos limit, 8,035 crossover)
    const scCard = el('section', 'sp-card');
    scCard.dataset.testid = 'arch-scaling-chart';
    const scHead = el('div', 'sp-card-head');
    scHead.append(
      el('h2', 'sp-card-title', 'Context-Length Memory Scaling (1K–1M)'),
      el(
        'span',
        'sp-badge sp-badge--den',
        `Crossover: ${sc.crossoverTokens.toLocaleString('en-US')} tok`,
      ),
    );
    scCard.append(scHead);
    scCard.append(buildContextLengthScalingSvg(sc));

    const regimeKey = el('div', 'sp-svg-regime-key');
    regimeKey.dataset.testid = 'scaling-regime-key';
    for (const [swatchCls, txt] of [
      ['sp-svg-swatch--ok', 'Operational (≤128K · 131,072)'],
      ['sp-svg-swatch--limit', 'Configured Limit (≤256K · 262,144 max_pos)'],
      ['sp-svg-swatch--extrap', 'Extrapolated (>256K)'],
    ] as const) {
      const it = el('span', 'sp-svg-legend-item');
      it.append(el('span', `sp-svg-swatch ${swatchCls}`), document.createTextNode(txt));
      regimeKey.append(it);
    }
    scCard.append(regimeKey);

    const legend = el('div', 'sp-svg-legend');
    for (const [dotCls, txt] of [
      ['is-navy', 'Hybrid Total'],
      ['is-teal', 'Mamba 47.08 MiB'],
      ['is-blue', 'GQA KV 6 KiB/tok'],
      ['is-slate', '52L Pure GQA'],
    ] as const) {
      const it = el('span', 'sp-svg-legend-item');
      it.append(el('span', `sp-svg-dot ${dotCls}`), document.createTextNode(txt));
      legend.append(it);
    }
    scCard.append(legend);

    // Compact scaling table for anchor lengths (strict nowrap so headers & regime labels NEVER split mid-word!)
    const scTblWrap = el('div', 'sp-mini-table-wrap');
    const scTbl = el('table', 'sp-mini-table sp-mini-table--scaling');
    scTbl.dataset.testid = 'arch-scaling-table';
    const scThead = el('thead');
    const scHr = el('tr');
    for (const h of ['Prefix', 'Regime', 'Persistent (SI / IEC)', 'vs 52L GQA']) {
      scHr.append(el('th', undefined, h));
    }
    scThead.append(scHr);
    const scTbody = el('tbody');
    for (const pt of sc.points) {
      if (pt.isCustomPoint && pt.tokens < 1_024) continue;
      const rowCls =
        pt.scalingRegime === 'extrapolated_beyond_max_pos'
          ? 'is-extrap'
          : pt.scalingRegime === 'configured_positional_limit'
            ? 'is-config-limit'
            : '';
      const tr = el('tr', rowCls);
      tr.title = pt.regimeLabel;
      tr.append(
        el('td', undefined, pt.tokenLabel),
        el('td', undefined, pt.regimeShortLabel),
        el('td', undefined, pt.persistentTotalDualDisplay),
        el('td', undefined, `${pt.hybridVsFullGqaSavingsRatio.toFixed(1)}×`),
      );
      scTbody.append(tr);
    }
    scTbl.append(scThead, scTbody);
    scTblWrap.append(scTbl);
    const regimeDisclaimerEl = el('p', 'sp-note sp-note--callout', sc.regimeDisclaimer);
    regimeDisclaimerEl.dataset.testid = 'scaling-regime-disclaimer';
    scCard.append(scTblWrap, regimeDisclaimerEl);
    archPanel.append(scCard);

    // 4. Production BF16/FP32/FP64 vs Miniature Float32 Comparison Table
    const precCard = el('section', 'sp-card');
    precCard.dataset.testid = 'arch-precision-table';
    const precHead = el('div', 'sp-card-head');
    precHead.append(
      el('h2', 'sp-card-title', 'Production BF16/FP32/FP64 vs Miniature Float32'),
      el('span', 'sp-badge sp-badge--ctx', 'Exact / Ratio / Mini'),
    );
    precCard.append(precHead);

    const pTblWrap = el('div', 'sp-mini-table-wrap');
    const pTbl = el('table', 'sp-mini-table sp-mini-table--prose');
    const pThead = el('thead');
    const pHr = el('tr');
    for (const h of ['Axis', 'Production', 'Miniature']) pHr.append(el('th', undefined, h));
    pThead.append(pHr);
    const pTbody = el('tbody');
    for (const row of readout.precisionVsMiniature.rows) {
      const tr = el('tr');
      tr.title = row.note;
      tr.append(
        el('td', undefined, row.dimension),
        el('td', undefined, row.productionValue),
        el('td', undefined, row.miniatureValue),
      );
      pTbody.append(tr);
    }
    pTbl.append(pThead, pTbody);
    pTblWrap.append(pTbl);
    precCard.append(pTblWrap);
    archPanel.append(precCard);

    // 5. Collapsible Paper vs Checkpoint Discrepancy Ledger
    const ledCard = el('section', 'sp-card');
    ledCard.dataset.testid = 'arch-discrepancy-ledger';
    const ledHead = el('div', 'sp-card-head');
    ledHead.append(
      el('h2', 'sp-card-title', 'Paper vs Checkpoint Discrepancy Ledger'),
      el('span', 'sp-badge sp-badge--time', `${readout.discrepancyLedger.length} audited`),
    );
    ledCard.append(ledHead);

    for (const entry of readout.discrepancyLedger) {
      const det = el('details', 'sp-discrepancy-item');
      const sum = el('summary', 'sp-discrepancy-summary', `${entry.index}. ${entry.topic}`);
      const body = el('div', 'sp-discrepancy-body');
      body.append(
        el('div', 'sp-note', `Paper: ${entry.paperClaim}`),
        el('div', 'sp-note', `Model Card: ${entry.modelCardClaim}`),
        el('div', 'sp-note', `Config / Checkpoint: ${entry.configOrIndexFact}`),
        el('div', 'sp-note', `Reference Code: ${entry.codeFact}`),
        el('div', 'sp-note sp-note--ctx', `Resolution: ${entry.authoritativeResolution}`),
      );
      det.append(sum, body);
      ledCard.append(det);
    }
    archPanel.append(ledCard);
  }

  function refreshTelemetryFooter(force = false): void {
    const now = performance.now();
    if (!force && now - lastFooterRefreshMs < TELEMETRY_FOOTER_THROTTLE_MS) {
      return;
    }
    lastFooterRefreshMs = now;
    const mem = getCachedMemoryBreakdown();
    const q = computeQuantiles(frameTimesMs);
    const rafQ = computeQuantiles(rafIntervalsMs);
    const rafDeltaLabel =
      rafIntervalsMs.length > 0 ? `${lastRafIntervalMs.toFixed(1)}ms` : 'idle';

    telemetryFooter.textContent = '';
    telemetryFooter.append(
      el(
        'p',
        'sp-note',
        `Synthetic Mini (d=8, V=34, L=52, kvCap=24 · HF rev ${TWOTOWER_SPEC.provenance.hfRevision.slice(0, 8)}): not 63.19B checkpoint weights.`,
      ),
    );
    const telGrid = el('div', 'sp-tel-grid');
    const timingSpan = el(
      'span',
      'sp-tel-timing',
      `JS cb: ${lastFrameMs.toFixed(1)}ms (p95 ${q.p95.toFixed(1)}ms) · RAF Δ: ${rafDeltaLabel} (${isSoftwareRenderer ? 'SW' : 'HW'})`,
    );
    timingSpan.dataset.testid = 'telemetry-timing-readout';
    timingSpan.title =
      `performance.now() measures main-thread JS/WebGL-submit callback duration ONLY (NOT full asynchronous GPU execution or compositor latency). ` +
      `Observed consecutive animating RAF interval (idle pauses excluded): ${
        rafIntervalsMs.length > 0
          ? `last ${lastRafIntervalMs.toFixed(1)}ms · avg ${rafQ.avg.toFixed(1)}ms · p95 ${rafQ.p95.toFixed(1)}ms (${rafIntervalsMs.length} samples)`
          : 'idle (no active continuous animation)'
      }. Renderer: ${glRenderer}.`;

    telGrid.append(
      el('span', undefined, `Retained: ${fmtBytesCompact(mem.totalRetainedBytes)}`),
      el('span', undefined, `Weights: ${fmtBytesCompact(mem.sharedModelWeightBytes)}`),
      el('span', undefined, `Draws: ${drawCallsLastFrame} · Rebuilds: ${geometryRebuilds} · Atlas: ${atlasUploads}`),
      timingSpan,
    );
    const attributionEl = el(
      'p',
      'sp-tel-attribution',
      'MIT · Inspired by bbycroft/llm-viz · arXiv:2606.26493v2 · Unofficial / zero weights shipped',
    );
    attributionEl.dataset.testid = 'twotower-attribution';
    telemetryFooter.append(telGrid, attributionEl);
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
      addKvCell(dl, 'Model Inventory (Full)', MINI_INVENTORY_SUMMARY, true);
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
        d
          ? `${d.atlasDimensions[0]}×${d.atlasDimensions[1]} (${d.atlasPrimaryMaskedCells} masked)`
          : '128×128',
      );
      addKvCell(
        dl,
        'AdaLN / Gate',
        `${d?.hasDenoiserPreNormMod ? 'pre-norm ON' : 'bypassed'} · ${d?.hasDenoiserResidualGate ? 'gate ON' : 'off'}`,
      );
      addKvCell(
        dl,
        'Model Inventory vs Atlas Slice',
        `Model: ${TWOTOWER_MINI_MODEL_INVENTORY.gqa.queryHeads} Q / ${TWOTOWER_MINI_MODEL_INVENTORY.gqa.kvHeads} KV (d_head=${TWOTOWER_MINI_MODEL_INVENTORY.gqa.headDim}) · Atlas Slice: ${TWOTOWER_MINI_MODEL_INVENTORY.gqa.atlasSelectedScoreHeadCount} selected Q heads (Q${gqaGroup0Head}, Q${gqaGroup1Head})`,
        true,
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

      container.append(
        el(
          'p',
          'sp-note',
          `Model Inventory: ${TWOTOWER_MINI_MODEL_INVENTORY.mamba.heads} Mamba / ${TWOTOWER_MINI_MODEL_INVENTORY.mamba.groups} groups / ${TWOTOWER_MINI_MODEL_INVENTORY.mamba.ssmStateMatrixShape[0]}×${TWOTOWER_MINI_MODEL_INVENTORY.mamba.ssmStateMatrixShape[1]} SSM · Atlas Slice: ${TWOTOWER_MINI_MODEL_INVENTORY.mamba.atlasSelectedDecompHeadCount} selected head (${formatMambaHeadId(mambaHead)})`,
        ),
      );

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
          [
            'Inherited Retention',
            `‖·‖₂=${dec.inheritedRetentionL2Norm.toFixed(4)} (Π dA=${dec.cumulativeDecay.toFixed(4)})`,
          ],
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
      addKvCell(
        dl,
        'tModel',
        d?.tModel === null || d?.tModel === undefined ? 'null (BYPASSED)' : d.tModel.toFixed(4),
      );
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

      // Partial-Denominator-Safe Top-6 Sidebar Weights (STRICTLY uses d.sidebarWeights — RB-2!)
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
      addKvCell(
        dl,
        'Vocab / Special',
        `V=${d?.vocabSize ?? 34} ([MASK]=#${d?.maskTokenId ?? 32}, <eos>=#${d?.eosTokenId ?? 33})`,
        true,
      );
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
      addKvCell(dl, 'Untied / NoPE', 'Untied from W_lm_head · Zero RoPE');
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
          el(
            'span',
            'sp-badge sp-badge--time',
            `step ${pres.stepIndex + 1}/${pres.stepsPerBlock} · t=${pres.tModel.toFixed(3)}`,
          ),
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

  // ==========================================================================
  // WAVE 3 — GUIDED EDUCATIONAL WALKTHROUGH HUD & STRICT ISOLATION
  // ==========================================================================
  function clearTourAutoplayTimer(): void {
    if (tourAutoplayTimerId !== null) {
      window.clearTimeout(tourAutoplayTimerId);
      tourAutoplayTimerId = null;
    }
  }

  function scheduleTourAutoplayIfPlaying(): void {
    clearTourAutoplayTimer();
    if (!tourActive || !tourSnapshot || !tourSnapshot.isPlaying || !tourSnapshot.canNext) {
      return;
    }
    tourAutoplayTimerId = window.setTimeout(() => {
      tourAutoplayTimerId = null;
      if (!tourActive || !tourController || !tourSnapshot?.isPlaying) return;
      const nextSnap = tourController.next();
      applyTourSnapshot(nextSnap);
    }, TOUR_AUTOPLAY_MS);
  }

  function ensureTourController(): TwoTowerTourController {
    if (!tourController) {
      tourController = createTwoTowerTourController(model);
    }
    return tourController;
  }

  function capturePreTourState(): PreTourExplorationState {
    return {
      viewMode,
      selectedBayId,
      selectedTower,
      towerFocus,
      activeLinkerChannel,
      gqaBayId,
      mambaBayId,
      mambaLocalPos,
      timestepBayId,
      timestepLocalPos,
      moeBayId,
      moeTowerRole,
      embedTowerRole,
      outputTowerRole,
      outputFocusToken,
      stepperStep,
      activeSidebarTab,
      cameraPose: cloneCameraPose(cameraPose),
    };
  }

  function applyTourSnapshot(snap: TwoTowerTourSnapshot, instantCamera = false): void {
    tourSnapshot = snap;
    tourActive = snap.isActive;
    cachedMemBreakdown = null;

    // Stop any exploration playback animations cleanly
    macroReplayActive = false;
    macroReplayPhase = null;
    macroReplayStage = -1;
    stepperPlaying = false;
    stepperLastTickMs = null;
    stepperTermProgress = 1;
    clearDirectCellSelectionForView();

    viewMode = snap.viewMode;
    towerFocus = snap.towerFocus;
    activeLinkerChannel = snap.linkerChannel;
    mambaLocalPos = snap.blockPos;
    timestepLocalPos = snap.blockPos;
    outputFocusToken = snap.blockPos;
    stepperStep = snap.opStep;

    if (snap.viewMode === 'macro') {
      selectedBayId = snap.bayId;
      selectedTower =
        snap.towerFocus === 'context_solo'
          ? 'context'
          : snap.towerFocus === 'denoiser_solo'
            ? 'denoiser'
            : 'shared';
    } else if (snap.viewMode === 'paired_gqa') {
      gqaBayId = resolveNearestBayOfKind(Math.max(1, Math.min(52, snap.bayId)), '*');
    } else if (snap.viewMode === 'paired_mamba') {
      mambaBayId = resolveNearestBayOfKind(Math.max(1, Math.min(52, snap.bayId)), 'M');
    } else if (snap.viewMode === 'timestep') {
      timestepBayId = Math.max(1, Math.min(52, snap.bayId));
    } else if (snap.viewMode === 'moe') {
      moeBayId = resolveNearestBayOfKind(Math.max(1, Math.min(52, snap.bayId)), 'E');
      moeTowerRole =
        snap.towerFocus === 'context_solo'
          ? 'context'
          : snap.towerFocus === 'denoiser_solo'
            ? 'denoiser'
            : undefined;
    } else if (snap.viewMode === 'embedding') {
      embedTowerRole =
        snap.towerFocus === 'context_solo'
          ? 'context'
          : snap.towerFocus === 'denoiser_solo'
            ? 'denoiser'
            : undefined;
    } else if (snap.viewMode === 'output') {
      outputTowerRole =
        snap.towerFocus === 'context_solo'
          ? 'context'
          : snap.towerFocus === 'denoiser_solo'
            ? 'denoiser'
            : undefined;
    }

    // Apply beat-directed sidebar panel
    const desiredTab: TwoTowerSidebarTab = snap.showSamplerPanel
      ? 'sampler'
      : snap.showArchPanel
        ? 'architecture'
        : 'inspect';
    activeSidebarTab = desiredTab;
    for (const btn of sidebarTabButtons) {
      const isAct = btn.dataset.sidebarTab === desiredTab;
      btn.setAttribute('aria-selected', isAct ? 'true' : 'false');
      btn.setAttribute('tabindex', isAct ? '0' : '-1');
    }
    inspectPanel.hidden = desiredTab !== 'inspect';
    samplerPanel.hidden = desiredTab !== 'sampler';
    archPanel.hidden = desiredTab !== 'architecture';

    geometryDirty = true;
    rebuildSceneIfNeeded();
    renderTourCard(snap);

    if (currentScene) {
      if (snap.opStep !== null && currentScene.opFocusBounds?.[snap.opStep]) {
        focusOpStepCamera(snap.opStep, reducedMotion || instantCamera);
      } else {
        flyToPose(computeFitPoseForScene(currentScene), reducedMotion || instantCamera);
      }
    }

    tourAriaLive.textContent = `Beat ${snap.beatIndex + 1} of ${snap.totalBeats}: ${snap.title}`;
    scheduleTourAutoplayIfPlaying();
    drawNow();
  }

  function startTour(): TwoTowerTourSnapshot {
    if (workbenchOpen) setWorkbenchOpen(false, false);
    if (!tourActive) {
      preTourState = capturePreTourState();
    }
    const ctrl = ensureTourController();
    const snap = ctrl.start();
    applyTourSnapshot(snap);
    return snap;
  }

  function goToTourBeat(idx: number): TwoTowerTourSnapshot {
    if (workbenchOpen) setWorkbenchOpen(false, false);
    if (!tourActive) {
      preTourState = capturePreTourState();
    }
    const ctrl = ensureTourController();
    const snap = ctrl.goToBeat(idx);
    applyTourSnapshot(snap, true);
    return snap;
  }

  function goToTourAct(actIdx: number): TwoTowerTourSnapshot {
    if (workbenchOpen) setWorkbenchOpen(false, false);
    if (!tourActive) {
      preTourState = capturePreTourState();
    }
    const ctrl = ensureTourController();
    const snap = ctrl.goToAct(actIdx);
    applyTourSnapshot(snap, true);
    return snap;
  }

  function exitTour(): void {
    clearTourAutoplayTimer();
    if (!tourActive) return;
    if (tourController) {
      tourSnapshot = tourController.exit();
    }
    tourActive = false;
    tourCard.hidden = true;
    cachedMemBreakdown = null;

    // Restore user's EXACT pre-tour exploration state untouched!
    if (preTourState) {
      const s = preTourState;
      viewMode = s.viewMode;
      selectedBayId = s.selectedBayId;
      selectedTower = s.selectedTower;
      towerFocus = s.towerFocus;
      activeLinkerChannel = s.activeLinkerChannel;
      gqaBayId = s.gqaBayId;
      mambaBayId = s.mambaBayId;
      mambaLocalPos = s.mambaLocalPos;
      timestepBayId = s.timestepBayId;
      timestepLocalPos = s.timestepLocalPos;
      moeBayId = s.moeBayId;
      moeTowerRole = s.moeTowerRole;
      embedTowerRole = s.embedTowerRole;
      outputTowerRole = s.outputTowerRole;
      outputFocusToken = s.outputFocusToken;
      stepperStep = s.stepperStep;
      activeSidebarTab = s.activeSidebarTab;
      flyAnim = null;
      cameraPose = cloneCameraPose(s.cameraPose);
      preTourState = null;

      for (const btn of sidebarTabButtons) {
        const isAct = btn.dataset.sidebarTab === activeSidebarTab;
        btn.setAttribute('aria-selected', isAct ? 'true' : 'false');
        btn.setAttribute('tabindex', isAct ? '0' : '-1');
      }
      inspectPanel.hidden = activeSidebarTab !== 'inspect';
      samplerPanel.hidden = activeSidebarTab !== 'sampler';
      archPanel.hidden = activeSidebarTab !== 'architecture';
    }

    geometryDirty = true;
    rebuildSceneIfNeeded();
    drawNow();
    btnTourToggle.focus();
  }

  function renderTourCard(snap: TwoTowerTourSnapshot): void {
    tourCard.hidden = !snap.isActive;
    if (!snap.isActive) return;

    tourCard.textContent = '';

    // 4 Act chapter-jump buttons
    const actsBar = el('div', 'sp-tour-acts');
    actsBar.setAttribute('role', 'group');
    actsBar.setAttribute('aria-label', 'Guided Tour Acts');
    const actNumerals = ['Act I', 'Act II', 'Act III', 'Act IV'] as const;
    for (let i = 0; i < TWOTOWER_TOUR_ACTS.length; i++) {
      const act = TWOTOWER_TOUR_ACTS[i];
      const b = el(
        'button',
        `sp-tour-act-btn${snap.actIndex === i ? ' is-active' : ''}`,
        actNumerals[i] ?? `Act ${i + 1}`,
      );
      b.type = 'button';
      b.dataset.testid = `tour-act-${i}`;
      b.title = act.title;
      b.setAttribute('aria-pressed', snap.actIndex === i ? 'true' : 'false');
      b.addEventListener('click', () => {
        if (!tourController) return;
        applyTourSnapshot(tourController.goToAct(i));
      });
      actsBar.append(b);
    }

    // Header: Beat counter + badge + progress bar
    const head = el('div', 'sp-tour-head');
    const counter = el(
      'span',
      'sp-tour-counter',
      `Beat ${snap.beatIndex + 1} / ${snap.totalBeats} · ${snap.beatId}`,
    );
    counter.dataset.testid = 'tour-beat-counter';
    const badge = el('span', 'sp-badge sp-badge--ctx', snap.badge);
    head.append(counter, badge);

    const progTrack = el('div', 'sp-tour-progress');
    const progFill = el('div', 'sp-tour-progress-fill');
    progFill.style.width = `${(snap.progressFraction * 100).toFixed(1)}%`;
    progTrack.append(progFill);

    // Title & subtitle
    const titleWrap = el('div', 'sp-tour-titles');
    const h3 = el('h3', 'sp-tour-title', snap.title);
    h3.dataset.testid = 'tour-beat-title';
    titleWrap.append(
      el('div', 'sp-tour-act-label', snap.actTitle),
      h3,
      el('div', 'sp-tour-subtitle', snap.subtitle),
    );

    // Pinned Top Header: Act tabs + Beat counter + progress bar + Title/Subtitle
    const headerZone = el('div', 'sp-tour-header');
    headerZone.append(actsBar, head, progTrack, titleWrap);

    // Scrollable Middle Body: Narrative + Takeaways + Live Metrics + Provenance + Epistemic Tag
    const body = el('div', 'sp-tour-body');
    for (const p of snap.paragraphs) {
      body.append(el('p', 'sp-tour-p', p));
    }
    if (snap.takeaways.length > 0) {
      const ul = el('ul', 'sp-tour-takeaways');
      for (const tk of snap.takeaways) {
        ul.append(el('li', undefined, tk));
      }
      body.append(ul);
    }

    // Live metrics chips computed from real tour record
    const metricsRow = el('div', 'sp-tour-metrics');
    metricsRow.dataset.testid = 'tour-live-metrics';
    for (const m of snap.liveMetrics) {
      const chip = el('span', tourToneBadgeClass(m.tone), `${m.label}: ${m.value}`);
      metricsRow.append(chip);
    }

    // Provenance footer
    const prov = snap.provenance;
    const provEl = el(
      'div',
      'sp-tour-prov',
      `Rec #${prov.recordIndex} · Blk ${prov.blockId ?? '—'} · Ref ${prov.refinementIndex ?? '—'} · ` +
        `Abs Pos [${prov.absoluteTokenRange[0]}..${prov.absoluteTokenRange[1]}] · ` +
        `Cache ${prov.sourceCacheDigest.slice(0, 8)}`,
    );
    provEl.dataset.testid = 'tour-provenance';

    // Epistemic tag badge
    const epiEl = el('div', 'sp-tour-epistemic', snap.epistemicTag);
    epiEl.dataset.testid = 'tour-epistemic-tag';

    body.append(metricsRow, provEl, epiEl);

    // Pinned Bottom Footer Controls: Prev, Play/Pause, Next, Restart Tour, Exit Tour
    const ctrlBar = el('div', 'sp-tour-controls');
    const prevBtn = el('button', 'sp-btn', '← Prev');
    prevBtn.type = 'button';
    prevBtn.dataset.testid = 'tour-btn-prev';
    prevBtn.disabled = !snap.canPrev;
    prevBtn.addEventListener('click', () => {
      if (!tourController) return;
      applyTourSnapshot(tourController.prev());
    });

    const playBtn = el(
      'button',
      `sp-btn ${snap.isPlaying ? 'sp-btn--den' : 'sp-btn--primary'}`,
      snap.isPlaying ? 'Pause' : 'Play',
    );
    playBtn.type = 'button';
    playBtn.dataset.testid = 'tour-btn-play';
    playBtn.addEventListener('click', () => {
      if (!tourController) return;
      const toggled = tourController.togglePlay();
      tourSnapshot = toggled;
      renderTourCard(toggled);
      scheduleTourAutoplayIfPlaying();
      publishTelemetry();
    });

    const nextBtn = el('button', 'sp-btn sp-btn--primary', 'Next →');
    nextBtn.type = 'button';
    nextBtn.dataset.testid = 'tour-btn-next';
    nextBtn.disabled = !snap.canNext;
    nextBtn.addEventListener('click', () => {
      if (!tourController) return;
      applyTourSnapshot(tourController.next());
    });

    const restartBtn = el('button', 'sp-btn', 'Restart Tour');
    restartBtn.type = 'button';
    restartBtn.dataset.testid = 'tour-btn-restart';
    restartBtn.addEventListener('click', () => {
      if (!tourController) return;
      applyTourSnapshot(tourController.restart());
    });

    const exitBtn = el('button', 'sp-btn sp-btn--danger', 'Exit Tour');
    exitBtn.type = 'button';
    exitBtn.dataset.testid = 'tour-btn-exit';
    exitBtn.addEventListener('click', () => exitTour());

    ctrlBar.append(prevBtn, playBtn, nextBtn, restartBtn, exitBtn);

    tourCard.append(headerZone, body, ctrlBar);
  }

  // ==========================================================================
  // TELEMETRY SNAPSHOT & WINDOW.__TWOTOWER_SPATIAL_DEBUG__
  // ==========================================================================
  function buildTelemetrySnapshot(): TwoTowerTelemetrySnapshot {
    const activeSession = getActiveDisplaySession();
    const record = getActiveRecord(activeSession);
    const branch = getActiveBranch(activeSession);
    const mem = getCachedMemoryBreakdown();
    const scene = currentScene;
    const q = computeQuantiles(frameTimesMs);
    const rafQ = computeQuantiles(rafIntervalsMs);

    return Object.freeze({
      viewMode,
      drawCallsLastFrame,
      lastFrameMs,
      avgFrameMs: frameTimesMs.length > 0 ? q.avg : rafFramesRendered > 0 ? totalFrameMs / rafFramesRendered : 0,
      medianFrameMs: q.median,
      p95FrameMs: q.p95,
      frameSampleCount: frameTimesMs.length,
      lastRafIntervalMs,
      avgRafIntervalMs: rafQ.avg,
      medianRafIntervalMs: rafQ.median,
      p95RafIntervalMs: rafQ.p95,
      rafIntervalSampleCount: rafIntervalsMs.length,
      geometryRebuilds,
      atlasUploads,
      rafIdlePauses,
      rafActive,
      layoutResizeCount,
      glVendor,
      glRenderer,
      isSoftwareRenderer,
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
      tourActive,
      tourBeatIndex: tourActive && tourSnapshot ? tourSnapshot.beatIndex : -1,
      tourActIndex: tourActive && tourSnapshot ? tourSnapshot.actIndex : -1,
      activeSidebarTab,
    });
  }

  function publishTelemetry(): void {
    window.__TWOTOWER_TELEMETRY__ = buildTelemetrySnapshot();
  }

  function syncAllUi(): void {
    const activeSession = getActiveDisplaySession();
    const view = resolveTwoTowerSessionViewState(activeSession);
    syncHeaderAndSessionStrip(view);
    renderTimelineDock(activeSession);
    if (currentScene) {
      canvasTitleEl.textContent = currentScene.title;
      canvasSubEl.textContent = currentScene.subtitle;
      renderBottomDock(currentScene);
      renderInspector(currentScene, view);
    }
    publishTelemetry();
  }

  // ==========================================================================
  // VIEW & SESSION TRANSITIONS
  // ==========================================================================
  function cancelPendingHover(): void {
    if (pendingHoverRafId !== null) {
      window.cancelAnimationFrame(pendingHoverRafId);
      pendingHoverRafId = null;
    }
  }

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

  function switchViewMode(nextMode: TwoTowerSpatialViewMode, instantFit = true): void {
    if (nextMode === viewMode && currentScene !== null) return;
    cancelPendingHover();
    viewMode = nextMode;
    stepperPlaying = false;
    stepperLastTickMs = null;
    stepperAccumSec = 0;
    stepperStep = null;
    stepperStepProgress = 0;
    stepperTermProgress = 1;
    macroReplayActive = false;
    macroReplayElapsedMs = 0;
    macroReplayStage = -1;
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

  function findDeletableBranchId(targetSession: TwoTowerSession): string | null {
    for (let i = targetSession.branches.length - 1; i >= 0; i--) {
      if (targetSession.branches[i].branchId !== targetSession.activeBranchId) {
        return targetSession.branches[i].branchId;
      }
    }
    return null;
  }

  function applySessionUpdate(next: TwoTowerSession, fromWorkbench = false, note?: string): void {
    session = next;
    cachedMemBreakdown = null;
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
    if (tourActive) return;
    try {
      const next = fn();
      applySessionUpdate(next, false, note);
    } catch {
      const view = resolveTwoTowerSessionViewState(getActiveDisplaySession());
      syncHeaderAndSessionStrip(view);
      renderTimelineDock(getActiveDisplaySession());
    }
  }

  function jumpToRecordIndex(cursorIndex: number): void {
    if (tourActive) return;
    runSessionAction(
      () => navigateTwoTowerSession(session, cursorIndex),
      `Navigated to record #${cursorIndex}.`,
    );
  }

  // Wire Execution Strip & Timeline Controls
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
    runSessionAction(
      () => commitTwoTowerActiveBlockToContext(session),
      'Committed completed block to Context.',
    ),
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
  playbackQuickBtn.addEventListener('click', () => {
    if (viewMode === 'macro') {
      toggleMacroReplay(!macroReplayActive);
    } else if (viewMode === 'paired_gqa' || viewMode === 'paired_mamba' || viewMode === 'moe') {
      toggleOpStepperPlayback(!stepperPlaying);
    }
  });

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
  spBtnDeleteBranch.addEventListener('click', () => {
    const victim = findDeletableBranchId(session);
    if (!victim) return;
    runSessionAction(
      () => deleteTwoTowerSessionBranch(session, victim),
      `Deleted inactive branch ${victim}.`,
    );
  });
  spBtnRestart.addEventListener('click', () =>
    runSessionAction(() => restartTwoTowerSession(session), 'Restarted session.'),
  );

  // Wire Sampler Tab Future Config Controls
  const applyThreshold = (val: number, label: string) => {
    if (!Number.isFinite(val)) return;
    const clamped = Math.max(0, Math.min(1, val));
    runSessionAction(
      () => updateTwoTowerSessionConfig(session, { confidenceThreshold: clamped }),
      `Confidence threshold set to ${clamped.toFixed(2)} (${label}).`,
    );
  };
  presetMiniBtn.addEventListener('click', () => applyThreshold(0.22, 'Mini Default'));
  presetPaperBtn.addEventListener('click', () => applyThreshold(0.8, 'Paper'));
  presetCodeBtn.addEventListener('click', () => applyThreshold(0.9, 'Ref Code'));
  gammaRange.addEventListener('input', () => {
    gammaNum.value = Number(gammaRange.value).toFixed(2);
  });
  gammaRange.addEventListener('change', () => applyThreshold(Number(gammaRange.value), 'Custom'));
  gammaNum.addEventListener('change', () => applyThreshold(Number(gammaNum.value), 'Custom'));

  const applyBlockWidth = (b: 4 | 8) => {
    runSessionAction(
      () => updateTwoTowerSessionConfig(session, { blockSize: b }),
      `Next-block width B set to ${b}.`,
    );
  };
  btnB4.addEventListener('click', () => applyBlockWidth(4));
  btnB8.addEventListener('click', () => applyBlockWidth(8));

  const applyStepsK = (raw: number) => {
    if (!Number.isFinite(raw)) return;
    const clamped = Math.max(1, Math.min(12, Math.round(raw)));
    runSessionAction(
      () => updateTwoTowerSessionConfig(session, { stepsPerBlock: clamped }),
      `Next-block refinement budget K set to ${clamped}.`,
    );
  };
  stepsKRange.addEventListener('input', () => {
    stepsKNum.value = stepsKRange.value;
  });
  stepsKRange.addEventListener('change', () => applyStepsK(Number(stepsKRange.value)));
  stepsKNum.addEventListener('change', () => applyStepsK(Number(stepsKNum.value)));

  tempNum.addEventListener('change', () => {
    const val = Number(tempNum.value);
    if (!Number.isFinite(val)) return;
    const clamped = Math.max(0, Math.min(2, val));
    runSessionAction(
      () => updateTwoTowerSessionConfig(session, { temperature: clamped }),
      `Sampling temperature set to ${clamped}.`,
    );
  });

  seedNum.addEventListener('change', () => {
    const val = Number(seedNum.value);
    if (!Number.isFinite(val)) return;
    const seed = Math.trunc(val);
    runSessionAction(
      () => updateTwoTowerSessionConfig(session, { samplingSeed: seed }),
      `Sampling seed set to ${seed}.`,
    );
  });

  // Wire Topbar Tour, Reset & Workbench buttons
  btnTourToggle.addEventListener('click', () => {
    if (tourActive) exitTour();
    else startTour();
  });

  btnResetCamera.addEventListener('click', () => {
    if (currentScene) {
      flyToPose(computeFitPoseForScene(currentScene));
    }
  });

  function setWorkbenchOpen(open: boolean, restoreFocus = true): void {
    if (open && tourActive) return;
    workbenchOpen = open;
    if (open) {
      shell.style.setProperty('--sp-topbar-h', `${topbar.offsetHeight || 48}px`);
    }
    workbenchDrawer.hidden = !open;
    btnToggleWorkbench.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
      if (!workbenchCtrl) {
        workbenchCtrl = mountTwoTowerDiagnosticWorkbench(drawerBody, {
          initialSession: session,
          onSessionChange: (nextSession) => {
            if (tourActive) return;
            applySessionUpdate(nextSession, true);
          },
          onInspectedBayChange: (bayId) => {
            if (tourActive) return;
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
    } else if (restoreFocus) {
      btnToggleWorkbench.focus();
    }
  }

  btnToggleWorkbench.addEventListener('click', () => setWorkbenchOpen(!workbenchOpen));
  drawerCloseBtn.addEventListener('click', () => setWorkbenchOpen(false));

  // ==========================================================================
  // RENDER LOOP (ON-DEMAND IDLE RAF SLEEP, DISCRETE REPLAY TICKS, ZERO Z-FIGHT)
  // ==========================================================================
  const scheduleDraw = () => {
    if (disposed || pendingRafId !== null) return;
    rafActive = true;
    pendingRafId = window.requestAnimationFrame((ts) => drawFrame(ts, true));
  };

  const drawNow = () => {
    if (disposed) return;
    if (pendingRafId !== null) {
      window.cancelAnimationFrame(pendingRafId);
      pendingRafId = null;
    }
    drawFrame(performance.now(), false);
  };

  function drawFrame(nowMs: number, fromRaf = true): void {
    pendingRafId = null;
    if (disposed) {
      rafActive = false;
      prevContinuousRafMs = null;
      return;
    }
    if (fromRaf && prevContinuousRafMs !== null) {
      const intervalMs = nowMs - prevContinuousRafMs;
      if (intervalMs > 0 && intervalMs < 2000) {
        lastRafIntervalMs = intervalMs;
        rafIntervalsMs.push(intervalMs);
        if (rafIntervalsMs.length > MAX_FRAME_SAMPLES) {
          rafIntervalsMs.shift();
        }
      }
    }
    const t0 = performance.now();

    const measured = measureHostMetrics(viewportHost);
    if (!measured) {
      rafActive = false;
      prevContinuousRafMs = null;
      return;
    }
    currentMetrics = measured;

    const canAnimate = device !== null && !device.isLost;
    let continueAnimating = false;

    if (canAnimate) {
      // 1. Camera fly animation (ZERO geometry rebuilds!)
      if (flyAnim) {
        const t = Math.max(0, Math.min(1, (nowMs - flyAnim.startMs) / flyAnim.durationMs));
        cameraPose = interpolateCameraPose(flyAnim.from, flyAnim.to, t);
        if (t >= 1) flyAnim = null;
        else continueAnimating = true;
      }

      // 2. Macro instructional wave replay: discrete ~2.5 Hz stages (8 stages over 3.2s)
      //    while FlatFlowRibbonsPass GPU shader animates smoothly at 60Hz!
      if (viewMode === 'macro' && macroReplayActive && !reducedMotion) {
        const dtMs =
          macroReplayLastMs === null ? 0 : Math.max(0, Math.min(100, nowMs - macroReplayLastMs));
        macroReplayLastMs = nowMs;
        macroReplayElapsedMs += dtMs;
        if (macroReplayElapsedMs >= MACRO_REPLAY_DURATION_MS) {
          macroReplayActive = false;
          macroReplayElapsedMs = 0;
          macroReplayStage = -1;
          macroReplayPhase = null;
          macroReplayLastMs = null;
          geometryDirty = true;
        } else {
          const nextStage = Math.min(
            MACRO_REPLAY_STAGE_COUNT - 1,
            Math.floor((macroReplayElapsedMs / MACRO_REPLAY_DURATION_MS) * MACRO_REPLAY_STAGE_COUNT),
          );
          if (nextStage !== macroReplayStage) {
            macroReplayStage = nextStage;
            macroReplayPhase = (nextStage + 0.5) / MACRO_REPLAY_STAGE_COUNT;
            geometryDirty = true;
          }
          continueAnimating = true;
        }
      }

      // 3. Operation stepper playback: discrete sub-term buckets (~4.3 Hz)
      //    instead of rebuilding geometry 60 times/sec!
      if (
        stepperPlaying &&
        !reducedMotion &&
        stepperStep !== null &&
        (viewMode === 'paired_gqa' || viewMode === 'paired_mamba' || viewMode === 'moe')
      ) {
        const dtSec =
          stepperLastTickMs === null
            ? 0
            : Math.max(0, Math.min(0.1, (nowMs - stepperLastTickMs) / 1000));
        stepperLastTickMs = nowMs;
        stepperAccumSec += dtSec;
        const bucketDuration = STEPPER_CELL_DURATION_SEC / STEPPER_SUBTERM_BUCKETS;
        if (stepperAccumSec >= bucketDuration) {
          const ticks = Math.max(1, Math.floor(stepperAccumSec / bucketDuration));
          stepperAccumSec -= ticks * bucketDuration;
          const prevBucket = Math.round(stepperTermProgress * STEPPER_SUBTERM_BUCKETS);
          const nextBucket = prevBucket + ticks;
          if (nextBucket > STEPPER_SUBTERM_BUCKETS) {
            stepperTermProgress = 1 / STEPPER_SUBTERM_BUCKETS;
            const nextStepP = stepperStepProgress + 0.25;
            if (nextStepP >= 1) {
              stepperStepProgress = 0;
              stepperStep = (stepperStep + 1) % 6;
            } else {
              stepperStepProgress = nextStepP;
            }
          } else {
            stepperTermProgress = Math.min(1, nextBucket / STEPPER_SUBTERM_BUCKETS);
          }
          geometryDirty = true;
        }
        continueAnimating = true;
      }
    }

    const didRebuild = geometryDirty || currentScene === null;
    rebuildSceneIfNeeded();
    if (!currentScene) {
      rafActive = false;
      prevContinuousRafMs = null;
      return;
    }

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
    viewportHost.style.setProperty(
      '--sp-canvas-hud-h',
      `${Math.max(48, (canvasHud.offsetHeight || 28) + 14)}px`,
    );
    const tourHighlightPart = tourActive ? tourSnapshot?.highlightPartId ?? null : null;
    const tourCardVisible = tourActive && !tourCard.hidden;
    const tourRectKey = tourCardVisible
      ? `${tourCard.offsetLeft},${tourCard.offsetTop},${tourCard.offsetWidth},${tourCard.offsetHeight}`
      : '';
    const overlayKey =
      `${viewMode}|v${sceneVersion}|${currentMetrics.hostWidth.toFixed(1)}x${currentMetrics.hostHeight.toFixed(1)}|` +
      `${cameraPose.target[0].toFixed(3)},${cameraPose.target[1].toFixed(3)},${cameraPose.target[2].toFixed(3)},` +
      `${cameraPose.distance.toFixed(3)},${cameraPose.azimuthDeg.toFixed(2)},${cameraPose.elevationDeg.toFixed(2)}|` +
      `${topInsetPx}:${bottomInsetPx}|${tourHighlightPart ?? ''}|${tourRectKey}`;
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
      const popoverAvoidRects =
        tourCardVisible && tourCard.offsetWidth > 0 && tourCard.offsetHeight > 0
          ? [
              {
                x0: Math.max(0, tourCard.offsetLeft - 8),
                y0: Math.max(0, tourCard.offsetTop - 8),
                x1: tourCard.offsetLeft + tourCard.offsetWidth + 8,
                y1: tourCard.offsetTop + tourCard.offsetHeight + 8,
              },
            ]
          : undefined;
      overlay.update({
        scene: currentScene,
        snapshot: currentSnapshot,
        selectedPartId: activePart,
        highlightPartId: tourHighlightPart,
        activeTokens: getActiveRecord(getActiveDisplaySession()).committedPrefixLen,
        topInsetPx,
        bottomInsetPx,
        popoverAvoidRects,
      });
    }

    lastFrameMs = performance.now() - t0;
    totalFrameMs += lastFrameMs;
    frameTimesMs.push(lastFrameMs);
    if (frameTimesMs.length > MAX_FRAME_SAMPLES) {
      frameTimesMs.shift();
    }
    rafFramesRendered += 1;

    if (continueAnimating && canAnimate) {
      prevContinuousRafMs = fromRaf ? nowMs : null;
      refreshTelemetryFooter(false);
      publishTelemetry();
      scheduleDraw();
    } else {
      // On-demand Idle RAF sleep!
      rafActive = false;
      prevContinuousRafMs = null;
      rafIdlePauses += 1;
      refreshTelemetryFooter(rafFramesRendered === 1 || didRebuild);
      publishTelemetry();
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
    const view = resolveTwoTowerSessionViewState(getActiveDisplaySession());
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
        ? {
            tower: (hit?.towerRole === 'context' ? 'context' : 'denoiser') as 'context' | 'denoiser',
            cell,
          }
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
        if (sameGqaHover(gqaHoveredCell, wrapped)) return;
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
        ? {
            tower: (hit?.towerRole === 'context' ? 'context' : 'denoiser') as 'context' | 'denoiser',
            cell,
          }
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
        if (sameMambaHover(mambaHoveredCell, wrapped)) return;
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
        const view = resolveTwoTowerSessionViewState(getActiveDisplaySession());
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
    cancelPendingHover();
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
  // KEYBOARD SHORTCUTS, REDUCED MOTION, RESIZE, DPR, FONTS & WEBGL RECOVERY
  // ==========================================================================
  const onWindowKeyDown = (ev: KeyboardEvent) => {
    // Modal Workbench Dialog: trap Tab/Shift+Tab, allow Escape & W to close, suppress background shortcuts!
    if (workbenchOpen) {
      if (ev.key === 'Tab') {
        handleModalTabTrap(workbenchDrawer, ev);
        return;
      }
      if (ev.key === 'Escape') {
        ev.preventDefault();
        setWorkbenchOpen(false);
        return;
      }
      if (
        (ev.key === 'w' || ev.key === 'W') &&
        !isTextInputTarget(ev.target) &&
        !ev.metaKey &&
        !ev.ctrlKey &&
        !ev.altKey
      ) {
        ev.preventDefault();
        setWorkbenchOpen(false);
        return;
      }
      return;
    }

    if (ev.key === 'Escape') {
      if (tourActive) {
        ev.preventDefault();
        exitTour();
        return;
      }
      if (mobileInspectorOpen) {
        ev.preventDefault();
        setMobileInspectorOpen(false, true);
        return;
      }
    }

    // NEVER hijack Space or Enter on any interactive control!
    if ((ev.key === ' ' || ev.key === 'Enter') && isInteractiveControl(ev.target)) {
      return;
    }
    if (isTextInputTarget(ev.target) || ev.metaKey || ev.ctrlKey || ev.altKey) {
      return;
    }

    if (ev.key === 't' || ev.key === 'T') {
      ev.preventDefault();
      if (tourActive) exitTour();
      else startTour();
      return;
    }

    if (ev.key === 'w' || ev.key === 'W') {
      if (tourActive) return;
      ev.preventDefault();
      setWorkbenchOpen(true);
      return;
    }

    if (ev.key === 'r' || ev.key === 'R') {
      if (currentScene) {
        ev.preventDefault();
        flyToPose(computeFitPoseForScene(currentScene));
      }
      return;
    }

    if (tourActive && tourController) {
      const inTablist =
        ev.target instanceof HTMLElement && ev.target.closest('[role="tablist"]') !== null;
      if (!inTablist && ev.key === 'ArrowRight') {
        ev.preventDefault();
        applyTourSnapshot(tourController.next());
        return;
      }
      if (!inTablist && ev.key === 'ArrowLeft') {
        ev.preventDefault();
        applyTourSnapshot(tourController.prev());
        return;
      }
      if (ev.key === ' ') {
        ev.preventDefault();
        const toggled = tourController.togglePlay();
        tourSnapshot = toggled;
        renderTourCard(toggled);
        scheduleTourAutoplayIfPlaying();
        publishTelemetry();
        return;
      }
    }
  };
  window.addEventListener('keydown', onWindowKeyDown);

  const onPageHide = () => {
    cancelPendingHover();
    activePointers.clear();
    primaryPointerId = null;
    isDragging = false;
    cancelClick = false;
  };
  window.addEventListener('pagehide', onPageHide);

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
      stepperAccumSec = 0;
      stepperTermProgress = 1;
      macroReplayActive = false;
      macroReplayElapsedMs = 0;
      macroReplayStage = -1;
      macroReplayPhase = null;
      macroReplayLastMs = null;
    }
    geometryDirty = true;
    rebuildSceneIfNeeded();
    scheduleDraw();
  };

  const onMotionChange = (ev: MediaQueryListEvent) => handleReducedMotionChange(ev.matches);
  if (reducedMotionMq) {
    if (typeof reducedMotionMq.addEventListener === 'function') {
      reducedMotionMq.addEventListener('change', onMotionChange);
    } else if (typeof reducedMotionMq.addListener === 'function') {
      reducedMotionMq.addListener(onMotionChange);
    }
  }

  let lastResizeW = 0;
  let lastResizeH = 0;
  let lastResizeDpr = window.devicePixelRatio || 1;
  const handleResizeOrDpr = () => {
    if (disposed) return;
    const next = measureHostMetrics(viewportHost);
    if (!next) return;
    if (workbenchOpen) {
      shell.style.setProperty('--sp-topbar-h', `${topbar.offsetHeight || 48}px`);
    }
    if (
      Math.abs(next.hostWidth - lastResizeW) < 0.5 &&
      Math.abs(next.hostHeight - lastResizeH) < 0.5 &&
      Math.abs(next.dpr - lastResizeDpr) < 1e-3
    ) {
      return;
    }
    lastResizeW = next.hostWidth;
    lastResizeH = next.hostHeight;
    lastResizeDpr = next.dpr;
    layoutResizeCount += 1;
    currentMetrics = next;
    lastOverlayKey = '';
    if (currentScene && !flyAnim) {
      cameraPose = computeFitPoseForScene(currentScene);
    }
    scheduleDraw();
  };

  const resizeObserver = new ResizeObserver(() => handleResizeOrDpr());
  resizeObserver.observe(viewportHost);

  let dprMediaQuery: MediaQueryList | null = null;
  const onDprChange = () => {
    watchDpr();
    handleResizeOrDpr();
  };
  const watchDpr = () => {
    if (dprMediaQuery) {
      if (typeof dprMediaQuery.removeEventListener === 'function') {
        dprMediaQuery.removeEventListener('change', onDprChange);
      } else if (typeof dprMediaQuery.removeListener === 'function') {
        dprMediaQuery.removeListener(onDprChange);
      }
    }
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      dprMediaQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
      if (typeof dprMediaQuery.addEventListener === 'function') {
        dprMediaQuery.addEventListener('change', onDprChange);
      } else if (typeof dprMediaQuery.addListener === 'function') {
        dprMediaQuery.addListener(onDprChange);
      }
    }
  };
  watchDpr();

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(() => {
      if (disposed) return;
      overlay.clearMetricCache();
      lastOverlayKey = '';
      scheduleDraw();
    });
  }

  canvas.addEventListener('webglcontextlost', (ev) => {
    ev.preventDefault();
    flyAnim = null;
    stepperPlaying = false;
    stepperLastTickMs = null;
    stepperAccumSec = 0;
    macroReplayActive = false;
    macroReplayLastMs = null;
    clearTourAutoplayTimer();
    cancelPendingHover();
    if (pendingRafId !== null) {
      window.cancelAnimationFrame(pendingRafId);
      pendingRafId = null;
    }
    rafActive = false;
    prevContinuousRafMs = null;
    lastAtlasKey = null;
    lastOverlayKey = '';
    if (device) device.markLost(true);
    syncAllUi();
    showFallback('WebGL2 Context Lost', 'Waiting for the browser to restore the graphics context.');
  });

  canvas.addEventListener('webglcontextrestored', () => {
    if (!device || disposed) return;
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
      queryGlRendererInfo(device.gl);
      lastAtlasKey = null;
      lastOverlayKey = '';
      prevContinuousRafMs = null;
      device.markLost(false);
      fallbackEl.hidden = true;
      geometryDirty = true;
      drawNow();
    } catch {
      device.markLost(true);
      lastAtlasKey = null;
      syncAllUi();
      showFallback('WebGL2 Context Recovery Failed', 'Could not reinitialize WebGL2 shaders.');
    }
  });

  // ==========================================================================
  // INITIAL BOOT & EXPOSED HANDLES (`__TWOTOWER_APP__`, `__TWOTOWER_SPATIAL_DEBUG__`)
  // ==========================================================================
  const initialMetrics = measureHostMetrics(viewportHost);
  if (initialMetrics) {
    currentMetrics = initialMetrics;
    lastResizeW = initialMetrics.hostWidth;
    lastResizeH = initialMetrics.hostHeight;
    lastResizeDpr = initialMetrics.dpr;
  }
  geometryDirty = true;
  rebuildSceneIfNeeded();
  if (currentScene) {
    cameraPose = computeFitPoseForScene(currentScene);
  }
  drawNow();

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
        activeSidebarTab,
        tourActive,
        tourBeatIndex: tourActive && tourSnapshot ? tourSnapshot.beatIndex : -1,
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
      stepperAccumSec = 0;
      stepperStep = step;
      stepperStepProgress = 0;
      clearDirectCellSelectionForView();
      geometryDirty = true;
      rebuildSceneIfNeeded();
      drawNow();
    },
    setStepperTermProgress(progress: number) {
      stepperPlaying = false;
      stepperLastTickMs = null;
      stepperAccumSec = 0;
      stepperTermProgress = Math.max(0, Math.min(1, progress));
      geometryDirty = true;
      rebuildSceneIfNeeded();
      drawNow();
    },
    selectBay(bayId: number | null, tower: TwoTowerTowerRole = 'shared') {
      selectedBayId = bayId;
      selectedTower = tower;
      geometryDirty = true;
      rebuildSceneIfNeeded();
      drawNow();
    },
    getCanvasClickPointForBay(bayId: number, tower: 'context' | 'denoiser') {
      if (!currentScene || currentScene.viewMode !== 'macro' || !currentSnapshot) return null;
      const world =
        tower === 'context' ? contextBayFrontWorld(bayId) : denoiserBayFrontWorld(bayId);
      const pt = projectWorldToClientCss(world, currentSnapshot);
      return { x: pt.x, y: pt.y, visible: pt.visible };
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      clearTourAutoplayTimer();
      cancelPendingHover();
      if (pendingRafId !== null) {
        window.cancelAnimationFrame(pendingRafId);
        pendingRafId = null;
      }
      rafActive = false;
      prevContinuousRafMs = null;
      resizeObserver.disconnect();
      modelSwitcher.destroy();
      window.removeEventListener('keydown', onWindowKeyDown);
      window.removeEventListener('pagehide', onPageHide);
      if (reducedMotionMq) {
        if (typeof reducedMotionMq.removeEventListener === 'function') {
          reducedMotionMq.removeEventListener('change', onMotionChange);
        } else if (typeof reducedMotionMq.removeListener === 'function') {
          reducedMotionMq.removeListener(onMotionChange);
        }
      }
      if (dprMediaQuery) {
        if (typeof dprMediaQuery.removeEventListener === 'function') {
          dprMediaQuery.removeEventListener('change', onDprChange);
        } else if (typeof dprMediaQuery.removeListener === 'function') {
          dprMediaQuery.removeListener(onDprChange);
        }
        dprMediaQuery = null;
      }
      solidsPass?.dispose();
      sheetsPass?.dispose();
      linesPass?.dispose();
      ribbonsPass?.dispose();
      flatFlowPass?.dispose();
      device?.dispose();
      solidsPass = null;
      sheetsPass = null;
      linesPass = null;
      ribbonsPass = null;
      flatFlowPass = null;
      device = null;
      if (window.__TWOTOWER_APP__ === handle) delete window.__TWOTOWER_APP__;
      if (window.__TWOTOWER_SPATIAL_DEBUG__ === debugHandle) {
        delete window.__TWOTOWER_SPATIAL_DEBUG__;
      }
    },
  };

  // Expose live `window.__TWOTOWER_SPATIAL_DEBUG__` with all telemetry getters & helper methods
  const debugMethods = {
    resetPerfCounters(): void {
      frameTimesMs.length = 0;
      lastFrameMs = 0;
      totalFrameMs = 0;
      prevContinuousRafMs = null;
      lastRafIntervalMs = 0;
      rafIntervalsMs.length = 0;
      geometryRebuilds = 0;
      atlasUploads = 0;
      rafIdlePauses = 0;
      rafFramesRendered = 0;
      layoutResizeCount = 0;
      publishTelemetry();
    },
    startTour(): TwoTowerTourSnapshot {
      return startTour();
    },
    exitTour(): void {
      exitTour();
    },
    goToTourBeat(idx: number): TwoTowerTourSnapshot {
      return goToTourBeat(idx);
    },
    goToTourAct(actIdx: number): TwoTowerTourSnapshot {
      return goToTourAct(actIdx);
    },
    setSidebarTab(tab: TwoTowerSidebarTab): void {
      setSidebarTab(tab);
    },
    jumpToRecord(idx: number): void {
      jumpToRecordIndex(idx);
    },
    orbitDelta(dx: number, dy: number): void {
      flyAnim = null;
      cameraPose = orbitCameraByCssDelta(cameraPose, dx, dy);
      drawNow();
    },
    triggerMacroReplay(on: boolean): void {
      if (viewMode !== 'macro' && on) {
        switchViewMode('macro', true);
      }
      toggleMacroReplay(on);
    },
    triggerOpStepper(on: boolean): void {
      toggleOpStepperPlayback(on);
    },
    getTourSnapshot(): TwoTowerTourSnapshot | null {
      return tourActive ? tourSnapshot : null;
    },
    getSnapshot(): TwoTowerTelemetrySnapshot {
      return buildTelemetrySnapshot();
    },
  };

  const debugHandle = Object.create(null) as TwoTowerSpatialDebugHandle;
  const telemetryKeys: readonly (keyof TwoTowerTelemetrySnapshot)[] = [
    'viewMode',
    'drawCallsLastFrame',
    'lastFrameMs',
    'avgFrameMs',
    'medianFrameMs',
    'p95FrameMs',
    'frameSampleCount',
    'lastRafIntervalMs',
    'avgRafIntervalMs',
    'medianRafIntervalMs',
    'p95RafIntervalMs',
    'rafIntervalSampleCount',
    'geometryRebuilds',
    'atlasUploads',
    'rafIdlePauses',
    'rafActive',
    'layoutResizeCount',
    'glVendor',
    'glRenderer',
    'isSoftwareRenderer',
    'currentAtlasDims',
    'maxTextureSize',
    'retainedSessionBytes',
    'sharedWeightBytes',
    'solidCount',
    'sheetCount',
    'lineCount',
    'ribbonCount',
    'flowTriangleCount',
    'lifecyclePhase',
    'committedPrefixLen',
    'kvCapacity',
    'recordIndex',
    'branchId',
    'glErrorCount',
    'reducedMotion',
    'rafFramesRendered',
    'tourActive',
    'tourBeatIndex',
    'tourActIndex',
    'activeSidebarTab',
  ];
  for (const key of telemetryKeys) {
    Object.defineProperty(debugHandle, key, {
      enumerable: true,
      configurable: false,
      get: () => buildTelemetrySnapshot()[key],
    });
  }
  Object.assign(debugHandle, debugMethods);

  window.__TWOTOWER_APP__ = handle;
  window.__TWOTOWER_SPATIAL_DEBUG__ = debugHandle;
  return handle;
}
