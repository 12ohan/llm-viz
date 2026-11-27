/**
 * Frozen `DORY6` Step-0 Browser App Shell, Overlay, Analytical Charts, Timeline Dock &
 * `window.__dorySpatialDebug` Contracts (`src/dory/app/types.ts`).
 *
 * Shared Step-0 contract enabling strictly disjoint parallel implementation across `DORY6`:
 *   - Slice 6A: `src/dory/app/{spatialOverlay,charts}.ts`, `src/dory/app/spatial.css`
 *   - Slice 6B: `src/dory/app/{timelineDock,spatialApp,main}.ts`
 *   - Slice 6C: `tests/dory-app-contracts.test.ts` + dev-only Rollup/WebGL2 dry-run
 *
 * Locked Design Decisions & Guards Enforced Here:
 *   - `D1` Physical `0..71` single-sublayer indexing (`16 + 40 + 16`) with paired `L{p >> 1}` badges.
 *   - `D3`/`DR23` Contract A `104`-bank self-speculation badged `Reconstructed from Paper §6.2`.
 *   - `D4` Architectural Boundary Isolation: ZERO imports from Gate 4b/4c Nano modules
 *     (`src/ui/{controls,overlay,memoryChart}.ts`, `src/scene/{spine,sheetInspector}.ts`,
 *     `src/model/{spec,memory,state,rcPlayback}.ts`, `src/math/{gqaSim,mambaSim,moeSim,hybridFlow,incrementalDecode}.ts`,
 *     `src/story/walkthrough.ts`, `src/main.ts`) or sibling apps (`twotower`, `diffusion`, `uni`).
 *     Reuses ONLY `src/shared/{a11y,format,modelRegistry,modelSwitcher,overlayLayout}.ts` & `src/render/*`.
 *   - `D6`/`DR18` Mid-session depth (`kLoops`), KV policy (`per_loop <-> last_loop`), or execution mode
 *     (`standard <-> self_spec`) switches NEVER mutate existing KV slabs — always fork/re-prefill cleanly.
 *   - `D7` 7 focused editorial WebGL2 views (`#F7F8FA` light drafting-table theme), pooled overlay
 *     (`<= 64` cell numbers), strict `idle-RAF` gating (`rafActive === false` at rest), content-addressed
 *     `RG32F` atlas upload deduplication, and strictly `READ-ONLY · NO GRADIENTS` View 7 / readouts
 *     (zero fake quantization toggles, zero training/RLVR loops).
 *   - `D8` `vite.config.ts` and `src/shared/modelRegistry.ts` stay untouched throughout `DORY6`
 *     (the 5-way MPA registration entry `'dory'` is appended strictly in `DORY7`).
 *
 * 100% clean-room original TypeScript — zero runtime npm deps, zero ambient RNG.
 */

import type { CameraPose, CameraSnapshot } from '../../render/camera';
import type { ModelEntryId, ModelEntryMeta } from '../../shared/modelRegistry';
import type { MAX_OVERLAY_CELL_NUM_NODES, OverlayRect } from '../../shared/overlayLayout';
import type {
  DoryCheckpointFormatId,
  DoryGqaRopeViewMetadata,
  DoryHypersphereViewMetadata,
  DoryKvComparePartitionMode,
  DoryMacroDisplayMode,
  DoryPickIdentity,
  DorySelfSpecViewMetadata,
  DorySpanningSubspaceDiagramData,
  DorySpatialFocusState,
  DorySpatialScene,
  DorySpatialViewMode,
} from '../scene/types';
import type {
  DoryArchReadoutsBundle,
  DoryDiscrepancyFilterOptions,
  DoryInspectorTabId,
  DoryProductionKvScalingRow,
  DoryTbptlSchematicReadout,
  DoryTourActId,
  DoryTourState,
} from '../story/types';
import type {
  DoryExecutionCoord,
  DoryExecutionMode,
  DoryHaltReason,
  DoryKvCacheMode,
  DorySamplingConfig,
  DorySessionForkReason,
  DorySessionMemoryAudit,
  DorySessionRecordKind,
  DorySupportedLoopCount,
} from '../types';

// Re-export key domain UI types for concise single-source imports across Slice 6A / 6B / 6C
export type { DorySpatialFocusState, DorySpatialViewMode } from '../scene/types';
export type { DoryInspectorTabId, DoryTourActId } from '../story/types';

// ============================================================================
// 1. APP CONSTANTS, THEME, PRE-DORY7 SWITCHER METADATA & PROMPT PRESETS
// ============================================================================

/** Hard DOM pool ceiling for on-cell numeric readouts (matches `src/shared/overlayLayout.ts`). */
export const DORY_MAX_OVERLAY_CELL_NUM_NODES: typeof MAX_OVERLAY_CELL_NUM_NODES = 64;

/** Hard DOM pool ceiling for exterior sheet row/column axis header pills. */
export const DORY_MAX_OVERLAY_AXIS_HEADER_NODES = 24 as const;

/** Editorial drafting-table canvas background & switcher theme (`Audit §12.1`). */
export const DORY_APP_THEME = 'light' as const;
export const DORY_DRAFTING_BG_HEX = '#F7F8FA' as const;

/**
 * Forward-compatible switcher entry ID supporting `DORY6` standalone preview before `'dory'` is
 * appended to `src/shared/modelRegistry.ts` in `DORY7` (`D8`).
 */
export type DoryModelSwitcherEntryId = ModelEntryId | 'dory';

/**
 * Pre-`DORY7` standalone metadata entry passed to `mountModelSwitcher` (via `entries` override)
 * during `DORY6` dev-only preview without modifying `src/shared/modelRegistry.ts`.
 */
export interface DoryStandaloneModelMeta extends Omit<ModelEntryMeta, 'id' | 'testId'> {
  readonly id: DoryModelSwitcherEntryId;
  readonly testId: `nav-to-${DoryModelSwitcherEntryId}`;
}

export const DORY_STANDALONE_MODEL_META: DoryStandaloneModelMeta = Object.freeze({
  id: 'dory',
  shortLabel: 'Dory1.0-4B',
  title: 'NVIDIA Nemotron-Labs-Dory1.0-4B',
  subtitle: '72L Looped nGPT · SWA/Full + Self-Spec',
  pathSuffix: 'dory/',
  testId: 'nav-to-dory',
});

export type DoryPromptPresetId =
  | 'reasoning_12'
  | 'tool_call_10'
  | 'long_eviction_16'
  | 'short_probe_6';

export interface DoryPromptPreset {
  readonly id: DoryPromptPresetId;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
  /** Non-EOS deterministic token IDs from `DORY_MINI_VOCAB` (`0..63`, `<= S_max = 32`). */
  readonly tokens: readonly number[];
  /** Suggested chunk partition for `DR17`/`DR24` `last_loop` partition exploration. */
  readonly defaultChunkPartition: readonly number[];
  /** True when `tokens.length > W_mini = 8`, immediately exercising SWA eviction. */
  readonly crossesSwaWindow: boolean;
}

export const DORY_DEFAULT_PROMPT_PRESET_ID: DoryPromptPresetId = 'reasoning_12';

/**
 * Canonical deterministic prompt presets built strictly from `DORY_MINI_VOCAB` (`0..63`, zero EOS).
 * Default `'reasoning_12'` (`12` tokens > `W = 8`) matches the Audit §7.4 `[4, 8]` partition fixture
 * while leaving `20` slots for incremental decode & `gamma = 4` Contract A self-speculation cycles.
 */
export const DORY_PROMPT_PRESETS: readonly DoryPromptPreset[] = Object.freeze([
  Object.freeze({
    id: 'reasoning_12',
    label: 'Looped Reasoning (12 tok · > W=8)',
    shortLabel: 'Reasoning 12',
    description:
      '12-token reasoning prompt crossing SWA W=8 eviction (pos 8..11) with [4, 8] last_loop partition support.',
    // ['<s>', '<|im_start|>', 'user', '\n', 'solve', 'loop', 'depth', '4', 'sphere', 'anchor', 'cache', 'verify']
    tokens: Object.freeze([1, 10, 4, 18, 40, 25, 26, 57, 29, 30, 31, 37] as const),
    defaultChunkPartition: Object.freeze([4, 8] as const),
    crossesSwaWindow: true,
  }),
  Object.freeze({
    id: 'tool_call_10',
    label: 'Think + Tool Call (10 tok · > W=8)',
    shortLabel: 'Tool Call 10',
    description:
      '10-token chat prompt exercising production-aligned <think> (12) and <tool_call> (14) special tokens.',
    // ['<s>', '<|im_start|>', 'assistant', '\n', '<think>', 'check', 'ratio', '<tool_call>', 'query', 'cache']
    tokens: Object.freeze([1, 10, 5, 18, 12, 41, 45, 14, 51, 31] as const),
    defaultChunkPartition: Object.freeze([4, 6] as const),
    crossesSwaWindow: true,
  }),
  Object.freeze({
    id: 'long_eviction_16',
    label: 'Deep SWA Eviction (16 tok · 2x W)',
    shortLabel: 'Long 16',
    description:
      '16-token prompt filling the entire cap_swa=16 ring (8 evicted from W=8) while leaving 16 slots to S_max=32.',
    // ['<s>', '<|im_start|>', 'user', '\n', 'fn', 'let', 'state', '=', 'norm', 'sphere', '+', 'anchor', 'loop', 'depth', '4', 'verify']
    tokens: Object.freeze([1, 10, 4, 18, 48, 49, 27, 59, 28, 29, 60, 30, 25, 26, 57, 37] as const),
    defaultChunkPartition: Object.freeze([4, 6, 6] as const),
    crossesSwaWindow: true,
  }),
  Object.freeze({
    id: 'short_probe_6',
    label: 'Short Pre-Eviction Probe (6 tok · < W)',
    shortLabel: 'Short 6',
    description:
      '6-token prompt strictly inside W=8 so SWA and Full attention share identical causal visibility prior to eviction.',
    // ['<s>', 'solve', 'loop', 'sphere', 'anchor', 'proof']
    tokens: Object.freeze([1, 40, 25, 29, 30, 42] as const),
    defaultChunkPartition: Object.freeze([3, 3] as const),
    crossesSwaWindow: false,
  }),
]);

// ============================================================================
// 2. POOLED 3D-TO-2D OVERLAY CONTRACTS (`src/dory/app/spatialOverlay.ts`)
// ============================================================================

/** Screen-space exclusion / occupied rectangle compatible with `src/shared/overlayLayout.ts`. */
export type DoryOverlayRect = OverlayRect;

export interface DorySpatialOverlayOptions {
  /** Hard node cap for on-cell numbers (defaults to `64`, clamped strictly into `[0, 64]`). */
  readonly maxCellNumNodes?: number;
  /** Hard node cap for exterior axis headers (defaults to `24`, clamped strictly into `[0, 24]`). */
  readonly maxAxisHeaderNodes?: number;
  /** Default popover collision footprint width in CSS px (default `296`). */
  readonly popoverWidthPx?: number;
  /** Default popover collision footprint height in CSS px (default `124`). */
  readonly popoverHeightPx?: number;
}

export interface DorySpatialOverlayUpdateInput {
  readonly scene: DorySpatialScene;
  readonly snapshot: CameraSnapshot;
  readonly hoveredCell?: DoryPickIdentity | null;
  readonly selectedPartId?: string | null;
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly avoidRects?: readonly DoryOverlayRect[];
  /** Allows toggling zoom-adaptive on-cell numbers (default `true`; still capped `<= 64`). */
  readonly showCellNumbers?: boolean;
}

export interface DorySpatialOverlayStats {
  readonly renderedCellNumNodes: number;
  readonly renderedAxisHeaderNodes: number;
  readonly renderedAnchorPills: number;
  readonly renderedCalipers: number;
  readonly popoverVisible: boolean;
  readonly poolCellNumCapacity: number;
}

export interface DorySpatialOverlayHandle {
  /**
   * Re-projects pooled DOM/SVG overlay nodes for the current frame.
   * MUST execute ZERO `document.createElement` / `document.createElementNS` calls after mount!
   */
  update(input: DorySpatialOverlayUpdateInput): DorySpatialOverlayStats;
  getStats(): DorySpatialOverlayStats;
  clear(): void;
  dispose(): void;
}

/** Frozen module export contract implemented by `src/dory/app/spatialOverlay.ts` (Slice 6A). */
export interface DorySpatialOverlayModuleContract {
  readonly DORY_MAX_OVERLAY_CELL_NUM_NODES: typeof DORY_MAX_OVERLAY_CELL_NUM_NODES;
  computeDoryOverlayCellNumBudget(requestedCount: number, maxNodes?: number): number;
  createDorySpatialOverlay(
    host: HTMLElement,
    options?: DorySpatialOverlayOptions,
  ): DorySpatialOverlayHandle;
}

// ============================================================================
// 3. 2D ANALYTICAL `<svg>` CHART CONTRACTS (`src/dory/app/charts.ts`)
// ============================================================================

/**
 * Six editorial 2D `<svg>` analytical insets paired with the 7 WebGL2 spatial views (`Audit §12.1`):
 *   - View 2 (`hypersphere`): `'spanning_subspace'` (Exact 2D `span(a,b)`) + `'exploratory_pca'` (8D-metric PCA)
 *   - View 3 (`gqa_rope`):    `'dual_rope_phasors'` (4-plane NeoX SWA `theta_S=16` vs. Full `theta_F=1024`)
 *   - View 5 (`kv_compare`):  `'kv_scaling_256k'`   (`1 -> 256K` BF16 SWA/Full + `4x` knee + `(4+5k)/9`)
 *   - View 6 (`self_spec`):   `'self_spec_probs'`   (Contract A `(q_j, p_j)` acceptance & binary64 residual)
 *   - View 7 (`bookends`):    `'tbptl_schematic'`   (`READ-ONLY · NO GRADIENTS` TBPTL depth-1 & 5 stages)
 */
export type DorySvgChartKind =
  | 'spanning_subspace'
  | 'exploratory_pca'
  | 'dual_rope_phasors'
  | 'kv_scaling_256k'
  | 'self_spec_probs'
  | 'tbptl_schematic';

export interface DorySvgChartDimensions {
  readonly width?: number;
  readonly height?: number;
}

/** Headless pure-string SVG chart payload (100% Node-testable in Slice 6C without DOM). */
export interface DorySvgChartRenderResult {
  readonly kind: DorySvgChartKind;
  readonly title: string;
  readonly badge: string | null;
  readonly ariaLabel: string;
  readonly width: number;
  readonly height: number;
  /** Well-formed standalone `<svg ...>...</svg>` markup with finite viewBox coordinates. */
  readonly svgMarkup: string;
  /** Mandatory epistemic disclosures rendered alongside the diagram (anti-SLERP, PCA, DR13/DR23, TBPTL). */
  readonly epistemicCaveats: readonly string[];
}

export interface DoryKvScalingChartInput {
  readonly rows: readonly DoryProductionKvScalingRow[];
  readonly activeKLoops: number;
  readonly activeKvCacheMode: DoryKvCacheMode;
  readonly selectedSeqLen?: number;
}

export interface DoryAnalyticalChartsPanelHandle {
  readonly root: HTMLElement;
  renderForScene(
    scene: DorySpatialScene,
    readouts: DoryArchReadoutsBundle,
  ): readonly DorySvgChartKind[];
  getRenderedChartKinds(): readonly DorySvgChartKind[];
  clear(): void;
  dispose(): void;
}

/** Frozen module export contract implemented by `src/dory/app/charts.ts` (Slice 6A). */
export interface DoryChartsModuleContract {
  renderSpanningSubspaceSvg(
    data: DorySpanningSubspaceDiagramData,
    dims?: DorySvgChartDimensions,
  ): DorySvgChartRenderResult;
  renderExploratoryPcaSvg(
    metadata: DoryHypersphereViewMetadata,
    dims?: DorySvgChartDimensions,
  ): DorySvgChartRenderResult;
  renderDualRopePhasorsSvg(
    metadata: DoryGqaRopeViewMetadata,
    dims?: DorySvgChartDimensions,
  ): DorySvgChartRenderResult;
  renderProductionKvScalingSvg(
    input: DoryKvScalingChartInput,
    dims?: DorySvgChartDimensions,
  ): DorySvgChartRenderResult;
  renderSelfSpecProbsSvg(
    metadata: DorySelfSpecViewMetadata,
    dims?: DorySvgChartDimensions,
  ): DorySvgChartRenderResult;
  renderTbptlSchematicSvg(
    schematic: DoryTbptlSchematicReadout,
    dims?: DorySvgChartDimensions,
  ): DorySvgChartRenderResult;
  buildChartsForDoryScene(
    scene: DorySpatialScene,
    readouts: DoryArchReadoutsBundle,
  ): readonly DorySvgChartRenderResult[];
  createDoryChartsPanel(host: HTMLElement): DoryAnalyticalChartsPanelHandle;
}

// ============================================================================
// 4. TIMELINE, ELASTIC DEPTH & BRANCH DOCK CONTRACTS (`src/dory/app/timelineDock.ts`)
// ============================================================================

export interface DoryTimelineBranchSummary {
  readonly branchId: string;
  readonly label: string;
  readonly forkReason: DorySessionForkReason;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  readonly committedLen: number;
  readonly recordCount: number;
  readonly kvBankCount: number;
  readonly kvBytes: number;
  readonly haltReason: DoryHaltReason | null;
  readonly isActive: boolean;
}

export interface DoryTimelineDockState {
  readonly promptPresetId: DoryPromptPresetId;
  readonly kLoops: DorySupportedLoopCount;
  /** Exposes optional base-model extrapolation loop budgets `k in {6, 8}` (`D2`, `D10`). */
  readonly allowExtrapolationDepths: boolean;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  readonly sampling: DorySamplingConfig;
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
  readonly activeRecordKind: DorySessionRecordKind | null;
  readonly currentStepIndex: number;
  readonly totalRecordsInBranch: number;
  readonly committedLen: number;
  readonly maxSeqLen: number;
  readonly slidingWindow: number;
  readonly canStepBack: boolean;
  readonly canStepForward: boolean;
  readonly canExecuteStep: boolean;
  readonly haltReason: DoryHaltReason | null;
  readonly isHistoricalCursor: boolean;
  readonly isPlaying: boolean;
  readonly tourActive: boolean;
  /** Concise telemetry badge e.g. `"112L (56 paired) · 56 banks"` or `"192L · 104 Contract A banks"`. */
  readonly executedDepthLabel: string;
  /** Optional `D6`/`DR18` notice surfaced when changing `k`/policy/mode forks a clean branch. */
  readonly dr18ForkNotice: string | null;
  readonly branches: readonly DoryTimelineBranchSummary[];
  readonly memoryAudit: DorySessionMemoryAudit;
}

export interface DoryTimelineDockCallbacks {
  readonly onStepBack: () => void;
  readonly onStepForward: () => void;
  readonly onScrubIndex: (indexInBranch: number) => void;
  readonly onTogglePlay: () => void;
  readonly onExecuteNextStep: () => void;
  /** Changing `kLoops` on a populated session triggers `D6`/`DR18` clean fork/re-prefill. */
  readonly onSelectKLoops: (kLoops: DorySupportedLoopCount) => void;
  readonly onToggleExtrapolationDepths: (enabled: boolean) => void;
  /** Changing `per_loop <-> last_loop` triggers `D6`/`DR18` clean fork/re-prefill. */
  readonly onSelectKvCacheMode: (mode: DoryKvCacheMode) => void;
  /** Switching `standard <-> self_spec` (Contract A `k=4, per_loop`) triggers clean branch fork (`DR23`). */
  readonly onSelectExecutionMode: (mode: DoryExecutionMode) => void;
  readonly onSelectPromptPreset: (presetId: DoryPromptPresetId) => void;
  readonly onUpdateSampling: (patch: Partial<DorySamplingConfig>) => void;
  readonly onForkBranch: () => void;
  readonly onSwitchBranch: (branchId: string) => void;
  readonly onResetSession: () => void;
}

export interface DoryTimelineDockHandle {
  readonly root: HTMLElement;
  update(state: DoryTimelineDockState): void;
  getRenderedBranchSignature(): string;
  dispose(): void;
}

// ============================================================================
// 5. IDLE-RAF SCHEDULER, ATLAS UPLOAD TRACKER & APP CONTROLLER (`spatialApp.ts`)
// ============================================================================

export interface DoryIdleRafSchedulerOptions {
  readonly onFrame: (nowMs: number, dtMs: number) => void;
  readonly requestFrameFn?: (callback: (nowMs: number) => void) => number;
  readonly cancelFrameFn?: (handle: number) => void;
}

/**
 * Pure Node-testable idle-RAF scheduler guaranteeing `isActive() === false` at rest (`D7`).
 * Wakes strictly on `requestFrame()` or active `setContinuousReason(reason, true)`
 * (e.g., playback or camera fly animation) and halts immediately once settled.
 */
export interface DoryIdleRafScheduler {
  isActive(): boolean;
  getContinuousReasons(): readonly string[];
  requestFrame(): void;
  setContinuousReason(reason: string, active: boolean): void;
  dispose(): void;
}

/**
 * Pure Node-testable content-addressed `RG32F` GPU texture upload deduplicator:
 * increments `uploadCount` and returns `true` ONLY when `atlasKey !== null` and
 * `atlasKey !== lastAtlasKey`.
 */
export interface DoryAtlasUploadTracker {
  getLastAtlasKey(): string | null;
  getUploadCount(): number;
  shouldUpload(atlasKey: string | null): boolean;
  reset(): void;
}

export interface DoryAppMountOptions {
  readonly host: HTMLElement;
  /** Normalized Vite `import.meta.env.BASE_URL` (e.g. `'/'` or `'/llm-viz/'`). */
  readonly baseUrl?: string;
  readonly initialViewMode?: DorySpatialViewMode;
  readonly initialPromptPresetId?: DoryPromptPresetId;
  readonly initialKLoops?: DorySupportedLoopCount;
  readonly initialKvCacheMode?: DoryKvCacheMode;
  readonly initialExecutionMode?: DoryExecutionMode;
  /** Attach `window.__dorySpatialDebug` when `window` is defined (default `true`). */
  readonly attachDebugHandle?: boolean;
}

export interface DoryAppControllerState {
  readonly viewMode: DorySpatialViewMode;
  readonly promptPresetId: DoryPromptPresetId;
  readonly kLoops: DorySupportedLoopCount;
  readonly allowExtrapolationDepths: boolean;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  readonly sampling: DorySamplingConfig;
  readonly inspectorTab: DoryInspectorTabId;
  readonly inspectorCollapsed: boolean;
  readonly chartsPanelCollapsed: boolean;
  readonly shortcutsModalOpen: boolean;
  readonly discrepancyFilter: DoryDiscrepancyFilterOptions;
  readonly focus: DorySpatialFocusState;
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
  readonly tourActive: boolean;
  readonly tourBeatIndex: number;
  readonly isPlaying: boolean;
  readonly reducedMotion: boolean;
  readonly dr18ForkNotice: string | null;
}

export interface DorySpatialAppHandle {
  readonly root: HTMLElement;
  readonly debug: DorySpatialDebugHandle;
  getState(): DoryAppControllerState;
  getCurrentScene(): DorySpatialScene;
  getTourState(): DoryTourState;
  getCameraPose(): CameraPose | null;
  dispose(): void;
}

// ============================================================================
// 6. `window.__dorySpatialDebug` E2E & DIAGNOSTIC CONTRACT (`DORY6` / `DORY7`)
// ============================================================================

/**
 * Clean JSON-serializable state & telemetry snapshot returned by `window.__dorySpatialDebug`.
 * Directly drives Gate 6 dev-only WebGL2 smoke and Gate 7 Playwright E2E assertions
 * (idle-RAF `rafActive === false`, `glErrorCount === 0`, atlas dedup, `<= 64` overlay nodes,
 * `<= 12.00 MiB` memory audit, 7 views, `D6`/`DR18` fork transitions, and 4-act / 14-beat tour).
 */
export interface DorySpatialDebugSnapshot {
  readonly modelId: 'dory';
  readonly viewMode: DorySpatialViewMode;
  readonly macroDisplayMode: DoryMacroDisplayMode;
  readonly kLoops: number;
  readonly kvCacheMode: DoryKvCacheMode;
  readonly executionMode: DoryExecutionMode;
  /** Executed physical sublayer depth `32 + 40 * kLoops` (`72..352`). */
  readonly executedDepth: number;
  /** Logical persistent KV bank count (`36` `last_loop`, `16+20k` `per_loop`, `104` Contract A). */
  readonly kvBankCount: number;
  /** Active `requestAnimationFrame` flag — MUST be `false` at rest (`idle-RAF` contract). */
  readonly rafActive: boolean;
  /** Cumulative WebGL2 `gl.getError()` non-zero error count — MUST be `0`. */
  readonly glErrorCount: number;
  /** Content-addressed `RG32F` atlas key of the active scene (for GPU upload dedup verification). */
  readonly atlasKey: string | null;
  /** Cumulative GPU `RG32F` texture upload count (must not increment on identical `atlasKey`). */
  readonly atlasUploadCount: number;
  /** Active `RG32F` atlas byte footprint (`<= 524,288 B` = `512.00 KiB`). */
  readonly atlasByteLength: number;
  /** Currently visible on-cell numeric readout DOM nodes (`<= 64`). */
  readonly overlayCellNumNodes: number;
  /** Active 2D `<svg>` analytical inset chart kinds currently rendered for `viewMode`. */
  readonly renderedChartKinds: readonly DorySvgChartKind[];
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
  readonly branchCount: number;
  readonly recordCount: number;
  readonly committedLen: number;
  readonly inspectorTab: DoryInspectorTabId;
  readonly inspectorCollapsed: boolean;
  readonly tourActive: boolean;
  readonly tourBeatIndex: number;
  readonly tourBeatCount: number;
  readonly tourActIndex: number;
  /** Live 8-tuple `DoryExecutionCoord` of the active guided tour beat (`null` in Free-Explore). */
  readonly activeTourCoord: DoryExecutionCoord | null;
  readonly selectedCheckpointFormat: DoryCheckpointFormatId;
  /** View 7 & readout structural guarantees (strictly read-only, zero fake quant toggles). */
  readonly readOnlyTbptlVerified: true;
  readonly zeroFakeQuantToggles: true;
  readonly memoryAudit: DorySessionMemoryAudit;
}

/**
 * Interactive E2E & diagnostic handle attached to `window.__dorySpatialDebug` by `spatialApp.ts`.
 * Note: `spatialApp.ts` MUST instantiate `createDoryGuidedTour({ sceneDispatcher: buildDorySpatialScene })`
 * so all 7 views (including View 7 `bookends`) dispatch cleanly throughout the 14-beat tour.
 */
export interface DorySpatialDebugHandle extends DorySpatialDebugSnapshot {
  getSnapshot(): DorySpatialDebugSnapshot;
  setView(viewMode: DorySpatialViewMode): DorySpatialDebugSnapshot;
  updateFocus(patch: Partial<DorySpatialFocusState>): DorySpatialDebugSnapshot;
  setKLoops(kLoops: DorySupportedLoopCount): DorySpatialDebugSnapshot;
  setKvCacheMode(mode: DoryKvCacheMode): DorySpatialDebugSnapshot;
  setExecutionMode(mode: DoryExecutionMode): DorySpatialDebugSnapshot;
  setKvComparePartitionMode(mode: DoryKvComparePartitionMode): DorySpatialDebugSnapshot;
  setMacroDisplayMode(mode: DoryMacroDisplayMode): DorySpatialDebugSnapshot;
  stepSession(): DorySpatialDebugSnapshot;
  forkBranch(): DorySpatialDebugSnapshot;
  switchBranch(branchId: string): DorySpatialDebugSnapshot;
  setInspectorTab(tab: DoryInspectorTabId): DorySpatialDebugSnapshot;
  setInspectorCollapsed(collapsed: boolean): DorySpatialDebugSnapshot;
  startTour(beatIndex?: number): DorySpatialDebugSnapshot;
  goToBeat(beatIndex: number): DorySpatialDebugSnapshot;
  goToAct(actId: DoryTourActId): DorySpatialDebugSnapshot;
  nextBeat(): DorySpatialDebugSnapshot;
  prevBeat(): DorySpatialDebugSnapshot;
  exitTour(): DorySpatialDebugSnapshot;
  resetCamera(): DorySpatialDebugSnapshot;
}

declare global {
  interface Window {
    __dorySpatialDebug?: DorySpatialDebugHandle;
  }
}
