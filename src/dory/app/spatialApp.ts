/**
 * Interactive 5-Pass WebGL2 Spatial App Shell for NVIDIA Nemotron-Labs-Dory1.0-4B
 * (`src/dory/app/spatialApp.ts`).
 *
 * Hosts all 5 unchanged shared WebGL2 passes (`RenderDevice`, `SolidsPass`, `LinesPass`,
 * `RibbonsPass`, `FlatFlowRibbonsPass`, `SheetsPass`) across all 7 `D7` editorial views:
 *   1. `macro`       — Folded 72L vs. Unrolled `32 + 40k` + Parallel KV-Bank Rail
 *   2. `hypersphere` — Channel-Wise nGPT NLERP (`DR7` anti-SLERP), Exact 2D `span(a,b)`,
 *                      Exploratory 2D PCA (`8D`-only metrics), local `120 deg` anchor-map
 *                      caveat & read-only `Diagnostic Counterfactual` Coda
 *   3. `gqa_rope`    — Gated 4:1 GQA (`SWA W=8` vs. `Full`), `sqk^2`, 4-plane Dual-RoPE,
 *                      live `RG32F` `G`-channel masks (`DR20`), sigmoid gate & `attn_sub_norm`
 *   4. `swiglu`      — Normalized SwiGLU (`signed suv`, `ffn_sub_norm`, `Accum Reveal`)
 *   5. `kv_compare`  — Dual-Session `per_loop` vs. `last_loop`, `DR17` partition probe &
 *                      `1 -> 256K` BF16 scaling (`S`-invariant `(4 + 5k) / 9`)
 *   6. `self_spec`   — Reconstructed Paper §6.2 Contract A (`104` Dual-Coda banks, `DR23`),
 *                      binary64 rejection/residual & `cap_swa = 16` rollback (`DR19`)
 *   7. `bookends`    — Untied unit-norm bookends, `0 B` diff 4-checkpoint readouts (zero
 *                      fake quant toggles) & `READ-ONLY · NO GRADIENTS` TBPTL depth-1
 *
 * Key Guarantees:
 *   - `D4` Architectural Boundary Isolation: ZERO imports from Gate 4b/4c Nano modules
 *     (`src/ui/*`, `src/scene/{spine,sheetInspector}.ts`, `src/model/*`, `src/math/*`,
 *     `src/story/*`, `src/main.ts`) or sibling apps (`twotower`, `diffusion`, `uni`).
 *     Reuses ONLY `src/render/*` and `src/shared/{a11y,format,modelRegistry,modelSwitcher}.ts`
 *     with `theme: 'light'`.
 *   - Strict `idle-RAF` gating (`createIdleRafScheduler`: `rafActive === false` at rest).
 *   - Dual-hash content-addressed `RG32F` atlas upload deduplication (`createAtlasUploadTracker`).
 *   - Strict in-place DOM identity for sliders & controls across drags and canvas hover.
 *   - `D6`/`DR18` immutable branch fork/re-prefill on any depth, KV-policy, or mode switch.
 *   - Isolated 4-Act, 14-beat Guided Tour (`createDoryGuidedTour({ sceneDispatcher: buildDorySpatialScene })`)
 *     preserving Free-Explore session byte-for-byte.
 *   - Complete `window.__dorySpatialDebug` (`DorySpatialDebugHandle`) E2E & diagnostic contract.
 *
 * 100% clean-room original TypeScript — zero runtime npm deps, zero ambient RNG.
 */

import {
  buildCameraSnapshot,
  cloneCameraPose,
  fitOpBoundsCamera,
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
import {
  createFocusTrap,
  handleIndexedTablistKeydown,
  isEditableShortcutTarget,
  isReducedMotionPreferred,
  shouldDeferActivationKey,
  syncRovingTabindex,
} from '../../shared/a11y';
import { fmtSigned, formatBytesAdaptive, formatKiB } from '../../shared/format';
import {
  MODEL_REGISTRY,
  type ModelEntryId,
  type ModelEntryMeta,
} from '../../shared/modelRegistry';
import { mountModelSwitcher, type ModelSwitcherHandle } from '../../shared/modelSwitcher';
import { buildDoryArchReadouts } from '../archReadouts';
import { getDefaultDoryModelParameters } from '../parameters';
import { buildDoryDualRopeTables } from '../rope';
import { buildDorySpatialScene } from '../scene/index';
import {
  computeDoryAtlasContentKey,
  DORY_ANCHOR_MAP_CAVEAT_BADGE,
  DORY_ANTI_SLERP_PLAQUE,
  DORY_EXPLORATORY_PCA_BADGE,
  DORY_TBPTL_READONLY_BADGE,
  pickHitToDoryCell,
} from '../scene/provenance';
import {
  DEFAULT_DORY_SPATIAL_FOCUS_STATE,
  DORY_SPATIAL_VIEW_MODES,
  type DoryCheckpointFormatId,
  type DoryKvComparePartitionMode,
  type DoryMacroDisplayMode,
  type DoryPickIdentity,
  type DorySpatialFocusState,
  type DorySpatialScene,
  type DorySpatialViewMode,
} from '../scene/types';
import { createDorySession } from '../session';
import {
  computeExecutedDepth,
  computeKvBankCounts,
  decodeDoryToken,
  getDoryPhysicalLayerSpec,
} from '../spec';
import {
  DORY_PAPER_RESULTS_BADGE,
  type DoryArchReadoutsBundle,
  type DoryDiscrepancyFilterCategory,
  type DoryDiscrepancyFilterOptions,
  type DoryFreeExploreRestoreState,
  type DoryInspectorTabId,
  type DoryTourActId,
  type DoryTourState,
} from '../story/types';
import { createDoryGuidedTour } from '../story/guidedTour';
import {
  DORY_DEFAULT_GREEDY_SAMPLING,
  DORY_MINI_CONFIG,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_VERIFY_LOOPS,
  type DoryExecutionMode,
  type DoryKvCacheMode,
  type DorySamplingConfig,
  type DorySession,
  type DorySupportedLoopCount,
} from '../types';
import { createDoryChartsPanel } from './charts';
import { createDorySpatialOverlay } from './spatialOverlay';
import {
  buildDoryTimelineDockState,
  createDoryTimelineDock,
  getDoryPromptPreset,
} from './timelineDock';
import {
  DORY_APP_THEME,
  DORY_DEFAULT_PROMPT_PRESET_ID,
  DORY_STANDALONE_MODEL_META,
  type DoryAnalyticalChartsPanelHandle,
  type DoryAppControllerState,
  type DoryAppMountOptions,
  type DoryAtlasUploadTracker,
  type DoryIdleRafScheduler,
  type DoryIdleRafSchedulerOptions,
  type DoryOverlayRect,
  type DoryPromptPresetId,
  type DorySpatialAppHandle,
  type DorySpatialDebugHandle,
  type DorySpatialDebugSnapshot,
  type DorySpatialOverlayHandle,
  type DorySvgChartKind,
} from './types';

// ============================================================================
// PURE NODE-TESTABLE STATE MACHINES & HELPERS
// ============================================================================

/**
 * Strict idle-RAF scheduler guaranteeing `isActive() === false` at rest (`D7`):
 * wakes strictly on `requestFrame()` or active `setContinuousReason(reason, true)`
 * (e.g., playback or camera fly animation) and halts immediately once settled.
 */
export function createIdleRafScheduler(
  options: DoryIdleRafSchedulerOptions,
): DoryIdleRafScheduler {
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

/**
 * Pure dual-hash content-addressed `RG32F` GPU texture upload deduplicator:
 * increments `uploadCount` and returns `true` ONLY when `atlasKey !== null` and
 * `atlasKey !== lastAtlasKey`. Revisiting an identical dual-hash `atlasKey` or
 * passing `null` triggers `0` redundant GPU uploads.
 */
export function createAtlasUploadTracker(): DoryAtlasUploadTracker {
  let lastAtlasKey: string | null = null;
  let uploadCount = 0;

  return {
    getLastAtlasKey(): string | null {
      return lastAtlasKey;
    },
    getUploadCount(): number {
      return uploadCount;
    },
    shouldUpload(atlasKey: string | null): boolean {
      if (atlasKey === null) {
        return false;
      }
      if (atlasKey === lastAtlasKey) {
        return false;
      }
      lastAtlasKey = atlasKey;
      uploadCount += 1;
      return true;
    },
    reset(): void {
      lastAtlasKey = null;
    },
  };
}

/**
 * Builds the top-bar model-switcher entries list for `mountModelSwitcher`:
 * in `DORY6` (before `'dory'` is registered in `src/shared/modelRegistry.ts` in `DORY7`),
 * appends `DORY_STANDALONE_MODEL_META` without mutating `MODEL_REGISTRY`; once `DORY7`
 * registers `'dory'`, uses `MODEL_REGISTRY` directly with zero duplicates.
 */
export function buildDorySwitcherEntries(
  registry: readonly ModelEntryMeta[] = MODEL_REGISTRY,
): readonly ModelEntryMeta[] {
  for (let i = 0; i < registry.length; i += 1) {
    if ((registry[i]!.id as string) === 'dory') {
      return registry;
    }
  }
  return Object.freeze([
    ...registry,
    DORY_STANDALONE_MODEL_META as unknown as ModelEntryMeta,
  ]);
}

export type DoryGlobalKeyAction =
  | { readonly type: 'none' }
  | { readonly type: 'select_view'; readonly viewMode: DorySpatialViewMode }
  | { readonly type: 'toggle_tour' }
  | { readonly type: 'tour_next' }
  | { readonly type: 'tour_prev' }
  | { readonly type: 'exit_tour' }
  | { readonly type: 'reset_camera' }
  | { readonly type: 'toggle_inspector' }
  | { readonly type: 'toggle_charts' }
  | { readonly type: 'open_shortcuts' }
  | { readonly type: 'close_shortcuts' };

/**
 * Pure Node-testable global keyboard shortcut resolver reusing `src/shared/a11y.ts`
 * (`shouldDeferActivationKey` and `isEditableShortcutTarget`) so editable inputs
 * (e.g., ledger search) retain native `Escape`, arrow, and typing behavior.
 */
export function resolveDoryGlobalKeyAction(
  key: string,
  opts: {
    readonly target: EventTarget | null;
    readonly ctrlKey?: boolean;
    readonly metaKey?: boolean;
    readonly altKey?: boolean;
    readonly tourActive: boolean;
    readonly shortcutsModalOpen: boolean;
  },
): DoryGlobalKeyAction {
  if (opts.ctrlKey || opts.metaKey || opts.altKey) {
    return { type: 'none' };
  }

  if (opts.shortcutsModalOpen) {
    if (key === 'Escape') {
      return { type: 'close_shortcuts' };
    }
    return { type: 'none' };
  }

  if (
    isEditableShortcutTarget(opts.target as Parameters<typeof isEditableShortcutTarget>[0]) ||
    shouldDeferActivationKey(key, opts.target)
  ) {
    return { type: 'none' };
  }

  if (opts.tourActive) {
    if (key === 'ArrowRight') return { type: 'tour_next' };
    if (key === 'ArrowLeft') return { type: 'tour_prev' };
    if (key === 'Escape') return { type: 'exit_tour' };
  }

  if (key >= '1' && key <= '7') {
    const idx = Number.parseInt(key, 10) - 1;
    const desc = DORY_SPATIAL_VIEW_MODES[idx];
    if (desc) {
      return { type: 'select_view', viewMode: desc.id };
    }
  }

  if (key === 't' || key === 'T') return { type: 'toggle_tour' };
  if (key === 'r' || key === 'R') return { type: 'reset_camera' };
  if (key === 'i' || key === 'I') return { type: 'toggle_inspector' };
  if (key === 'c' || key === 'C') return { type: 'toggle_charts' };
  if (key === '?') return { type: 'open_shortcuts' };

  return { type: 'none' };
}

/**
 * Pure Node-testable helper translating a ray-picked `DoryPickIdentity` click in any of the
 * 7 spatial views into a clean partial `DorySpatialFocusState` update.
 */
export function resolveDoryPickInteraction(
  viewMode: DorySpatialViewMode,
  picked: DoryPickIdentity,
  kLoops: number,
): Partial<DorySpatialFocusState> | null {
  const clampLoop = (loop: number | null): number | undefined => {
    if (loop === null || loop < 1) return undefined;
    return Math.max(1, Math.min(kLoops, Math.trunc(loop)));
  };

  if (viewMode === 'macro') {
    const patch: {
      selectedPhysicalLayerIdx?: number;
      selectedLoopIteration?: number;
    } = {};
    if (picked.physicalLayerIdx !== null) {
      patch.selectedPhysicalLayerIdx = picked.physicalLayerIdx;
    }
    const loop = clampLoop(picked.loopIteration);
    if (loop !== undefined) {
      patch.selectedLoopIteration = loop;
    }
    return Object.keys(patch).length > 0 ? patch : null;
  }

  if (viewMode === 'hypersphere') {
    const patch: {
      selectedPhysicalLayerIdx?: number;
      selectedLoopIteration?: number;
    } = {};
    if (picked.physicalLayerIdx !== null) {
      patch.selectedPhysicalLayerIdx = picked.physicalLayerIdx;
    }
    const loop = clampLoop(picked.loopIteration);
    if (loop !== undefined) {
      patch.selectedLoopIteration = loop;
    }
    return Object.keys(patch).length > 0 ? patch : null;
  }

  if (viewMode === 'gqa_rope') {
    const patch: {
      gqaLayerIdx?: number;
      selectedQHead?: number;
      selectedRotaryPlane?: number;
      selectedQueryRow?: number;
      selectedKeyCol?: number;
    } = {};
    if (picked.physicalLayerIdx !== null && picked.physicalLayerIdx % 2 === 0) {
      patch.gqaLayerIdx = picked.physicalLayerIdx;
    }
    if (picked.qHeadIndex !== null) {
      patch.selectedQHead = picked.qHeadIndex;
    }
    if (picked.rotaryPlane !== null) {
      patch.selectedRotaryPlane = picked.rotaryPlane;
    }
    if (
      picked.row !== null &&
      (picked.tensorKey === 'attn_scores' || picked.tensorKey === 'attn_probs')
    ) {
      patch.selectedQueryRow = picked.row;
    }
    if (
      picked.col !== null &&
      (picked.tensorKey === 'attn_scores' || picked.tensorKey === 'attn_probs')
    ) {
      patch.selectedKeyCol = picked.col;
    }
    return Object.keys(patch).length > 0 ? patch : null;
  }

  if (viewMode === 'swiglu') {
    const patch: {
      swigluLayerIdx?: number;
      swigluChannel?: number;
      swigluOutputDim?: number;
    } = {};
    if (picked.physicalLayerIdx !== null && picked.physicalLayerIdx % 2 === 1) {
      patch.swigluLayerIdx = picked.physicalLayerIdx;
    }
    if (
      picked.col !== null &&
      picked.col >= 0 &&
      picked.col < DORY_MINI_CONFIG.intermediateSize &&
      (picked.tensorKey === 'swiglu_gate' ||
        picked.tensorKey === 'swiglu_up' ||
        picked.tensorKey === 'swiglu_act' ||
        picked.tensorKey === 'ffn_sub_norm')
    ) {
      patch.swigluChannel = picked.col;
    } else if (
      picked.col !== null &&
      picked.col >= 0 &&
      picked.col < DORY_MINI_CONFIG.hiddenSize &&
      picked.tensorKey === 'mlp_out'
    ) {
      patch.swigluOutputDim = picked.col;
    }
    return Object.keys(patch).length > 0 ? patch : null;
  }

  if (viewMode === 'kv_compare') {
    const patch: {
      kvCompareLayerIdx?: number;
      selectedLoopIteration?: number;
    } = {};
    if (picked.physicalLayerIdx !== null && picked.physicalLayerIdx % 2 === 0) {
      patch.kvCompareLayerIdx = picked.physicalLayerIdx;
    }
    const loop = clampLoop(picked.loopIteration);
    if (loop !== undefined) {
      patch.selectedLoopIteration = loop;
    }
    return Object.keys(patch).length > 0 ? patch : null;
  }

  if (viewMode === 'self_spec') {
    const patch: {
      selectedVerifyOffset?: number;
      selfSpecAttentionLayerIdx?: number;
    } = {};
    if (picked.specOffset !== null && picked.specOffset >= 0) {
      patch.selectedVerifyOffset = picked.specOffset;
    }
    if (picked.physicalLayerIdx !== null && picked.physicalLayerIdx % 2 === 0) {
      patch.selfSpecAttentionLayerIdx = picked.physicalLayerIdx;
    }
    return Object.keys(patch).length > 0 ? patch : null;
  }

  if (viewMode === 'bookends') {
    const part = picked.partId ?? '';
    if (part === 'bookend_ckpt_bf16') return { selectedCheckpointFormat: 'bf16' };
    if (part === 'bookend_ckpt_mxfp8') return { selectedCheckpointFormat: 'mxfp8' };
    if (part === 'bookend_ckpt_nvfp4') return { selectedCheckpointFormat: 'nvfp4' };
    if (part === 'bookend_ckpt_q5_k_m') return { selectedCheckpointFormat: 'q5_k_m' };
    if (
      picked.tokenIndex !== null &&
      picked.tokenIndex >= 0 &&
      picked.tokenIndex < DORY_MINI_CONFIG.vocabSize
    ) {
      return { selectedVocabTokenId: picked.tokenIndex };
    }
  }

  return null;
}

// ============================================================================
// APP UI CONSTANTS & DOM HELPERS
// ============================================================================

const CAMERA_FLY_DURATION_MS = 420;
const PLAYBACK_STEP_INTERVAL_MS = 850;

const INSPECTOR_TABS: readonly {
  readonly id: DoryInspectorTabId;
  readonly label: string;
}[] = Object.freeze([
  { id: 'inspect', label: 'Inspect' },
  { id: 'readouts', label: 'Readouts' },
  { id: 'ledger', label: 'Ledger DR1–DR24' },
]);

const LEDGER_CATEGORY_FILTERS: readonly {
  readonly id: 'all' | 'browser_guard' | DoryDiscrepancyFilterCategory;
  readonly label: string;
}[] = Object.freeze([
  { id: 'all', label: 'All (24)' },
  { id: 'browser_guard', label: 'Guards' },
  { id: 'spec_config', label: 'Spec' },
  { id: 'architecture_math', label: 'nGPT Math' },
  { id: 'runtime_precision', label: 'Precision' },
  { id: 'kv_cache', label: 'KV Cache' },
  { id: 'speculative', label: 'Self-Spec' },
  { id: 'provenance_license', label: 'Provenance' },
]);

const CHECKPOINT_FORMAT_CHOICES: readonly {
  readonly id: DoryCheckpointFormatId;
  readonly label: string;
}[] = Object.freeze([
  { id: 'bf16', label: 'BF16 (7.95 GB)' },
  { id: 'mxfp8', label: 'MXFP8 (4.75 GB)' },
  { id: 'nvfp4', label: 'NVFP4 (3.20 GB)' },
  { id: 'q5_k_m', label: 'Q5_K_M (3.67 GB)' },
]);

const PARTITION_MODE_CHOICES: readonly {
  readonly id: DoryKvComparePartitionMode;
  readonly label: string;
}[] = Object.freeze([
  { id: 'single_shot', label: 'Single-Shot (Thm 1)' },
  { id: 'chunked', label: 'Chunked [4, rem] (Thm 3)' },
  { id: 'one_by_one', label: '1-by-1 Decode (Thm 3)' },
]);

const MACRO_LAYER_PRESETS: readonly { readonly idx: number; readonly label: string }[] =
  Object.freeze([
    { idx: 0, label: 'P0 Prelude SWA' },
    { idx: 16, label: 'P16 Rec SWA (>1 α)' },
    { idx: 22, label: 'P22 Rec Full' },
    { idx: 56, label: 'P56 Coda SWA' },
    { idx: 71, label: 'P71 Coda MLP' },
  ]);

const GQA_LAYER_PRESETS: readonly { readonly idx: number; readonly label: string }[] = Object.freeze([
  { idx: 0, label: 'P0 SWA' },
  { idx: 6, label: 'P6 Full' },
  { idx: 16, label: 'P16 Rec SWA' },
  { idx: 22, label: 'P22 Rec Full' },
  { idx: 56, label: 'P56 Coda SWA' },
  { idx: 70, label: 'P70 Coda Full' },
]);

const SWIGLU_LAYER_PRESETS: readonly { readonly idx: number; readonly label: string }[] =
  Object.freeze([
    { idx: 1, label: 'P1 Prelude' },
    { idx: 17, label: 'P17 Rec MLP' },
    { idx: 55, label: 'P55 Rec Exit' },
    { idx: 71, label: 'P71 Coda Exit' },
  ]);

const SHORTCUT_ROWS: readonly { readonly key: string; readonly desc: string }[] = Object.freeze([
  { key: '1 – 7', desc: 'Switch across all 7 Dory1.0-4B editorial spatial views' },
  { key: 'T', desc: 'Start or exit the isolated 4-Act (14-beat) Guided Tour' },
  { key: '← / →', desc: 'Navigate previous / next beat while Guided Tour is active' },
  { key: 'R', desc: 'Reset 3D camera to the active view preset bounds' },
  { key: 'I', desc: 'Show or hide the right 3-tab Inspector drawer' },
  { key: 'C', desc: 'Show or hide the 2D analytical <svg> charts inset' },
  { key: '?', desc: 'Open this keyboard shortcuts dialog' },
  { key: 'Esc', desc: 'Close dialog or exit active Guided Tour (deferred in inputs)' },
  { key: 'Drag / Shift+Drag / Wheel', desc: 'Orbit, pan, and zoom the 3D WebGL2 drafting stage' },
]);

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
  grid.append(el('span', 'dory-kv-key', key), el('span', 'dory-kv-val', value));
}

function safeFiniteFixed(v: number, digits: number): string {
  if (!Number.isFinite(v)) return 'n/a';
  return v.toFixed(digits);
}

function safeFiniteSigned(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return 'n/a';
  return fmtSigned(v, digits);
}

function safeFiniteExp(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return 'n/a';
  if (v === 0) return '0.00e+0';
  return v.toExponential(digits);
}

function toSupportedLoopCount(k: number): DorySupportedLoopCount {
  if (k === 1 || k === 2 || k === 3 || k === 4 || k === 6 || k === 8) {
    return k;
  }
  return DORY_MINI_CONFIG.defaultRecurrentLoops;
}

function sameSampling(a: DorySamplingConfig, b: DorySamplingConfig): boolean {
  return (
    Math.abs(a.temperature - b.temperature) < 1e-9 &&
    a.topK === b.topK &&
    Math.abs(a.topP - b.topP) < 1e-9
  );
}

// ============================================================================
// MAIN WEBGL2 SPATIAL APPLICATION SHELL
// ============================================================================

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

/**
 * Mounts the full standalone isolated NVIDIA Nemotron-Labs-Dory1.0-4B WebGL2 spatial app
 * onto `options.host` (or bare `HTMLElement`), attaching `window.__dorySpatialDebug` by default.
 */
export function createDorySpatialApp(
  optionsOrHost: DoryAppMountOptions | HTMLElement,
): DorySpatialAppHandle {
  const options: DoryAppMountOptions =
    'host' in optionsOrHost ? optionsOrHost : { host: optionsOrHost };
  const host = options.host;
  host.innerHTML = '';

  // Shared read-only singleton weights & dual-RoPE tables (zero weight cloning across loops/sessions)
  const params = getDefaultDoryModelParameters();
  const ropeTables = buildDoryDualRopeTables(params.config);

  let promptPresetId: DoryPromptPresetId =
    options.initialPromptPresetId ?? DORY_DEFAULT_PROMPT_PRESET_ID;
  const initPreset = getDoryPromptPreset(promptPresetId);
  const initExecMode: DoryExecutionMode = options.initialExecutionMode ?? 'standard';
  const initKLoops: DorySupportedLoopCount =
    initExecMode === 'self_spec'
      ? DORY_SPEC_VERIFY_LOOPS
      : toSupportedLoopCount(options.initialKLoops ?? DORY_MINI_CONFIG.defaultRecurrentLoops);
  const initKvMode: DoryKvCacheMode =
    initExecMode === 'self_spec'
      ? 'per_loop'
      : (options.initialKvCacheMode ?? DORY_MINI_CONFIG.defaultKvCacheMode);

  const freeSession: DorySession = createDorySession({
    params,
    ropeTables,
    initialBranch: {
      promptTokens: initPreset.tokens,
      kLoops: initKLoops,
      kvCacheMode: initKvMode,
      executionMode: initExecMode,
      sampling: DORY_DEFAULT_GREEDY_SAMPLING,
    },
  });
  // Advance 1 initial deterministic step so timeline scrubbing & step-back are immediately active
  try {
    freeSession.step();
  } catch {
    // Keep prefill state cleanly if prompt already fills capacity
  }

  // Strictly isolated 4-Act / 14-beat Guided Tour wired to the full 7-view dispatcher (`types.ts:547`)
  const tour = createDoryGuidedTour({
    params,
    ropeTables,
    sceneDispatcher: buildDorySpatialScene,
  });

  const atlasTracker = createAtlasUploadTracker();

  let disposed = false;
  let viewMode: DorySpatialViewMode = options.initialViewMode ?? 'macro';
  let allowExtrapolationDepths = initKLoops > 4;
  let sampling: DorySamplingConfig = DORY_DEFAULT_GREEDY_SAMPLING;
  let inspectorTab: DoryInspectorTabId = 'inspect';
  let inspectorCollapsed = false;
  let chartsPanelCollapsed = false;
  let shortcutsModalOpen = false;
  let ledgerFilterMode: 'all' | 'browser_guard' | DoryDiscrepancyFilterCategory = 'all';
  let ledgerQuery = '';
  let isPlaying = false;
  let playbackAccumMs = 0;
  let glErrorCount = 0;
  let reducedMotion = isReducedMotionPreferred();
  let dr18ForkNotice: string | null = null;

  let focus: DorySpatialFocusState = {
    ...DEFAULT_DORY_SPATIAL_FOCUS_STATE,
    selectedLoopIteration: Math.min(initKLoops, DEFAULT_DORY_SPATIAL_FOCUS_STATE.selectedLoopIteration),
  };

  let hoveredCell: DoryPickIdentity | null = null;
  let selectedCell: DoryPickIdentity | null = null;
  let selectedPartId: string | null = null;

  let currentScene: DorySpatialScene = buildDorySpatialScene(freeSession, viewMode, focus);
  let currentReadouts: DoryArchReadoutsBundle = buildDoryArchReadouts({
    activeKLoops: currentScene.kLoops,
    activeKvCacheMode: currentScene.kvCacheMode,
    selectedCheckpointFormat: focus.selectedCheckpointFormat,
    liveSessionMemory: freeSession.measureMemory(),
    activeAtlasBytes: currentScene.valueAtlas?.data.byteLength ?? 0,
  });

  let cameraPose: CameraPose | null = null;
  let cameraSnapshot: CameraSnapshot | null = null;
  let activeFly: CameraFlyState | null = null;
  let sceneDirtyForGl = true;

  // --------------------------------------------------------------------------
  // DOM SKELETON (EDITORIAL #F7F8FA LIGHT DRAFTING-TABLE SHELL)
  // --------------------------------------------------------------------------

  const shell = el('div', 'dory-app-shell');
  shell.dataset.testid = 'dory-app-shell';

  // Polite screen-reader live region
  const liveRegion = el('div', 'dory-sr-only');
  liveRegion.setAttribute('role', 'status');
  liveRegion.setAttribute('aria-live', 'polite');
  liveRegion.setAttribute('aria-atomic', 'true');
  shell.appendChild(liveRegion);

  function announce(message: string): void {
    liveRegion.textContent = '';
    liveRegion.textContent = message;
  }

  // Top Header Bar
  const topbar = el('header', 'dory-topbar');
  const brand = el('div', 'dory-brand');
  brand.append(
    el('span', 'dory-brand-mark', 'DORY'),
    el('h1', 'dory-brand-title', 'Nemotron-Labs-Dory1.0-4B'),
    el('span', 'dory-brand-sub', '3.975B · 72L Looped nGPT · 3:1 SWA/Full'),
  );

  // 7-View Navigation Tablist (`D7`)
  const viewTablist = el('div', 'dory-view-tablist');
  viewTablist.setAttribute('role', 'tablist');
  viewTablist.setAttribute('aria-label', 'Dory1.0-4B spatial architecture views');

  const viewTabButtons: HTMLButtonElement[] = [];
  for (let i = 0; i < DORY_SPATIAL_VIEW_MODES.length; i += 1) {
    const desc = DORY_SPATIAL_VIEW_MODES[i]!;
    const btn = el('button', 'dory-view-tab');
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.dataset.viewMode = desc.id;
    btn.dataset.testid = `dory-view-tab-${desc.id}`;
    btn.title = `${desc.viewNumber}. ${desc.label} — ${desc.description}`;
    btn.append(
      el('span', 'dory-view-tab-num', String(desc.viewNumber)),
      el('span', undefined, desc.shortLabel),
    );
    btn.addEventListener('click', () => {
      if (tour.isActive()) {
        exitTourInternal(false);
      }
      setViewInternal(desc.id, true);
    });
    btn.addEventListener('keydown', (ev) => {
      const curIdx = DORY_SPATIAL_VIEW_MODES.findIndex((v) => v.id === viewMode);
      handleIndexedTablistKeydown(ev, viewTabButtons, curIdx >= 0 ? curIdx : 0, (nextIdx) => {
        if (tour.isActive()) {
          exitTourInternal(false);
        }
        setViewInternal(DORY_SPATIAL_VIEW_MODES[nextIdx]!.id, true);
      });
    });
    viewTabButtons.push(btn);
    viewTablist.appendChild(btn);
  }
  syncRovingTabindex(viewTabButtons, 0);

  const topActions = el('div', 'dory-topbar-actions');

  const tourLaunchBtn = el(
    'button',
    'dory-action-btn dory-action-btn--tour',
    `Guided Tour (${tour.totalBeats})`,
  );
  tourLaunchBtn.type = 'button';
  tourLaunchBtn.dataset.testid = 'dory-tour-launch-btn';
  tourLaunchBtn.addEventListener('click', () => {
    if (tour.isActive()) {
      exitTourInternal(true);
    } else {
      startTourInternal(0);
    }
  });

  const resetCamBtn = el('button', 'dory-action-btn', 'Reset View');
  resetCamBtn.type = 'button';
  resetCamBtn.dataset.testid = 'dory-reset-camera-btn';
  resetCamBtn.addEventListener('click', () => {
    flyToCurrentScenePreset(true);
  });

  const chartsToggleBtn = el('button', 'dory-action-btn', 'Hide Charts');
  chartsToggleBtn.type = 'button';
  chartsToggleBtn.dataset.testid = 'dory-charts-toggle-btn';
  chartsToggleBtn.setAttribute('aria-expanded', 'true');
  chartsToggleBtn.addEventListener('click', () => {
    toggleChartsPanelCollapsed();
  });

  const inspectorToggleBtn = el('button', 'dory-action-btn', 'Hide Inspector');
  inspectorToggleBtn.type = 'button';
  inspectorToggleBtn.dataset.testid = 'dory-inspector-toggle-btn';
  inspectorToggleBtn.setAttribute('aria-expanded', 'true');
  inspectorToggleBtn.addEventListener('click', () => {
    toggleInspectorCollapsed();
  });

  const shortcutsBtn = el('button', 'dory-action-btn dory-action-btn--icon', '?');
  shortcutsBtn.type = 'button';
  shortcutsBtn.dataset.testid = 'dory-shortcuts-btn';
  shortcutsBtn.setAttribute('aria-label', 'Keyboard shortcuts');
  shortcutsBtn.title = 'Keyboard shortcuts (?)';
  shortcutsBtn.addEventListener('click', () => {
    openShortcutsModal();
  });

  const resolvedBaseUrl = (
    options.baseUrl ??
    import.meta.env?.BASE_URL ??
    '../'
  ).replace(/\/?$/, '/');

  topActions.append(
    tourLaunchBtn,
    resetCamBtn,
    chartsToggleBtn,
    inspectorToggleBtn,
    shortcutsBtn,
  );

  // Shared Multi-Model Top-Bar Switcher with `theme: 'light'` (`D4`, `D7`, `D8`)
  const modelSwitcher: ModelSwitcherHandle = mountModelSwitcher(topActions, {
    activeId: 'dory' as ModelEntryId,
    baseUrl: resolvedBaseUrl,
    theme: DORY_APP_THEME,
    entries: buildDorySwitcherEntries(),
  });

  topbar.append(brand, viewTablist, topActions);

  // Main Workspace
  const workspace = el('div', 'dory-workspace');
  const stageColumn = el('div', 'dory-stage-column');
  const viewportHost = el('div', 'dory-viewport-host');
  viewportHost.dataset.testid = 'dory-viewport-host';

  const canvas = el('canvas', 'dory-webgl-canvas');
  canvas.dataset.testid = 'dory-webgl-canvas';
  canvas.setAttribute('aria-label', 'Interactive 3D Dory1.0-4B WebGL2 drafting stage');
  viewportHost.appendChild(canvas);

  const glFallback = el('div', 'dory-webgl-fallback');
  glFallback.dataset.testid = 'dory-webgl-fallback';
  glFallback.hidden = true;
  viewportHost.appendChild(glFallback);

  // Floating Stage Title & Provenance Badges HUD
  const stageHud = el('div', 'dory-stage-hud');
  const stageHudRow = el('div', 'dory-stage-hud-row');
  const stageTitleCard = el('div', 'dory-stage-title-card');
  const stageTitleEl = el('h2', 'dory-stage-title');
  const stageSubtitleEl = el('p', 'dory-stage-subtitle');
  const stageBadgesEl = el('div', 'dory-badge-strip');
  stageTitleCard.append(stageTitleEl, stageSubtitleEl, stageBadgesEl);

  const selfSpecBanner = el('div', 'dory-selfspec-banner', DORY_SPEC_CONTRACT_A_BADGE);
  selfSpecBanner.dataset.testid = 'dory-selfspec-banner';
  selfSpecBanner.hidden = true;

  const readOnlyBanner = el(
    'div',
    'dory-readonly-banner',
    `${DORY_TBPTL_READONLY_BADGE} · ZERO FAKE QUANT TOGGLES`,
  );
  readOnlyBanner.dataset.testid = 'dory-readonly-banner';
  readOnlyBanner.hidden = true;

  stageHudRow.append(stageTitleCard, selfSpecBanner, readOnlyBanner);
  stageHud.appendChild(stageHudRow);
  viewportHost.appendChild(stageHud);

  // Pooled 3D-to-2D Overlay (`src/dory/app/spatialOverlay.ts`, <= 64 cell numbers)
  const overlay: DorySpatialOverlayHandle = createDorySpatialOverlay(viewportHost);

  // Dedicated 2D `<svg>` Analytical Charts Inset (`src/dory/app/charts.ts`)
  const chartsInsetHost = el('div', 'dory-charts-inset-host');
  chartsInsetHost.dataset.testid = 'dory-charts-inset-host';
  viewportHost.appendChild(chartsInsetHost);
  const chartsPanel: DoryAnalyticalChartsPanelHandle = createDoryChartsPanel(chartsInsetHost);

  // 4-Act / 14-Beat Guided Tour HUD Card (non-modal region anchored top-right)
  const tourHud = el('aside', 'dory-tour-hud');
  tourHud.dataset.testid = 'dory-tour-hud';
  tourHud.setAttribute('role', 'region');
  tourHud.setAttribute('aria-label', 'Dory1.0-4B 4-Act Guided Pedagogical Tour');
  tourHud.tabIndex = -1;
  tourHud.hidden = true;

  const tourActsRow = el('div', 'dory-tour-acts');
  const tourActButtons: HTMLButtonElement[] = [];
  for (let i = 0; i < tour.acts.length; i += 1) {
    const act = tour.acts[i]!;
    const btn = el('button', 'dory-tour-act-btn', act.shortTitle);
    btn.type = 'button';
    btn.dataset.testid = `dory-tour-act-${act.id}`;
    btn.title = act.title;
    btn.addEventListener('click', () => {
      goToTourBeatInternal(act.startBeatIndex);
    });
    tourActButtons.push(btn);
    tourActsRow.appendChild(btn);
  }

  const tourHeader = el('div', 'dory-tour-header');
  const tourBeatCounter = el(
    'span',
    'dory-tour-beat-counter',
    `Act 1 · Beat 1 / ${tour.totalBeats}`,
  );
  const tourExitBtn = el('button', 'dory-action-btn', 'Exit Tour');
  tourExitBtn.type = 'button';
  tourExitBtn.dataset.testid = 'dory-tour-exit-btn';
  tourExitBtn.addEventListener('click', () => exitTourInternal(true));
  tourHeader.append(tourBeatCounter, tourExitBtn);

  const tourTitleEl = el('h2', 'dory-tour-title');
  const tourTakeawayEl = el('p', 'dory-tour-takeaway');
  const tourNarrativeEl = el('p', 'dory-tour-narrative');
  const tourEquationEl = el('div', 'dory-tour-equation');
  const tourMetricsEl = el('div', 'dory-tour-metrics');
  const tourCoordEl = el('div', 'dory-tour-coord');
  tourCoordEl.dataset.testid = 'dory-tour-coord';
  const tourEpistemicEl = el('p', 'dory-tour-epistemic');

  const tourFooter = el('div', 'dory-tour-footer');
  const tourRefsEl = el('div', 'dory-tour-refs');
  const tourNavEl = el('div', 'dory-tour-nav');
  const tourPrevBtn = el('button', 'dory-action-btn', '← Prev');
  tourPrevBtn.type = 'button';
  tourPrevBtn.dataset.testid = 'dory-tour-prev-btn';
  tourPrevBtn.addEventListener('click', () => {
    const st = tour.getState();
    goToTourBeatInternal(Math.max(0, st.beatIndex - 1));
  });
  const tourNextBtn = el('button', 'dory-action-btn dory-action-btn--tour', 'Next →');
  tourNextBtn.type = 'button';
  tourNextBtn.dataset.testid = 'dory-tour-next-btn';
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
    tourMetricsEl,
    tourCoordEl,
    tourEpistemicEl,
    tourFooter,
  );
  viewportHost.appendChild(tourHud);

  // Bottom Timeline, Elastic Depth & Contract A Dock
  const dockHost = el('div', 'dory-dock-host');
  const timelineDock = createDoryTimelineDock(dockHost, {
    onStepBack: () => stepCursorBy(-1),
    onStepForward: () => stepCursorBy(1),
    onScrubIndex: (idx) => scrubToRecordIndex(idx),
    onTogglePlay: () => togglePlayback(),
    onExecuteNextStep: () => executeNextStep(),
    onSelectKLoops: (k) => selectKLoopsInternal(k),
    onToggleExtrapolationDepths: (enabled) => toggleExtrapolationDepthsInternal(enabled),
    onSelectKvCacheMode: (mode) => selectKvCacheModeInternal(mode),
    onSelectExecutionMode: (mode) => selectExecutionModeInternal(mode),
    onSelectPromptPreset: (presetId) => selectPromptPresetInternal(presetId),
    onUpdateSampling: (patch) => updateSamplingInternal(patch),
    onForkBranch: () => forkCurrentBranchInternal(),
    onSwitchBranch: (branchId) => switchToBranchInternal(branchId),
    onResetSession: () => resetSessionInternal(),
  });

  stageColumn.append(viewportHost, dockHost);

  // Collapsible Right 3-Tab Inspector Drawer (`Inspect | Readouts | Ledger DR1–DR24`)
  const inspectorDrawer = el('aside', 'dory-inspector-drawer');
  inspectorDrawer.dataset.testid = 'dory-inspector-drawer';
  inspectorDrawer.setAttribute('aria-label', 'Dory1.0-4B architectural inspector');

  const inspectorTablist = el('div', 'dory-inspector-tablist');
  inspectorTablist.setAttribute('role', 'tablist');
  inspectorTablist.setAttribute('aria-label', 'Inspector panels');

  const inspectorTabButtons: HTMLButtonElement[] = [];
  for (let i = 0; i < INSPECTOR_TABS.length; i += 1) {
    const tab = INSPECTOR_TABS[i]!;
    const btn = el('button', 'dory-inspector-tab', tab.label);
    btn.type = 'button';
    btn.setAttribute('role', 'tab');
    btn.dataset.testid = `dory-inspector-tab-${tab.id}`;
    btn.addEventListener('click', () => {
      setInspectorTabInternal(tab.id);
    });
    btn.addEventListener('keydown', (ev) => {
      const curIdx = INSPECTOR_TABS.findIndex((t) => t.id === inspectorTab);
      handleIndexedTablistKeydown(
        ev,
        inspectorTabButtons,
        curIdx >= 0 ? curIdx : 0,
        (nextIdx) => {
          setInspectorTabInternal(INSPECTOR_TABS[nextIdx]!.id);
        },
      );
    });
    inspectorTabButtons.push(btn);
    inspectorTablist.appendChild(btn);
  }
  syncRovingTabindex(inspectorTabButtons, 0);

  const inspectorBody = el('div', 'dory-inspector-body');
  inspectorBody.dataset.testid = 'dory-inspector-body';

  inspectorDrawer.append(inspectorTablist, inspectorBody);
  workspace.append(stageColumn, inspectorDrawer);

  // Accessible Keyboard Shortcuts Modal (`role="dialog"`, `aria-modal="true"`, focus-trapped)
  const shortcutsBackdrop = el('div', 'dory-modal-backdrop');
  shortcutsBackdrop.dataset.testid = 'dory-shortcuts-backdrop';
  shortcutsBackdrop.hidden = true;

  const shortcutsDialog = el('div', 'dory-modal-dialog');
  shortcutsDialog.dataset.testid = 'dory-shortcuts-modal';
  shortcutsDialog.setAttribute('role', 'dialog');
  shortcutsDialog.setAttribute('aria-modal', 'true');
  shortcutsDialog.setAttribute('aria-label', 'Dory1.0-4B keyboard shortcuts');
  shortcutsDialog.tabIndex = -1;

  const modalHeader = el('div', 'dory-modal-header');
  modalHeader.appendChild(el('h2', 'dory-modal-title', 'Keyboard Shortcuts'));
  const modalCloseBtn = el('button', 'dory-action-btn', 'Close');
  modalCloseBtn.type = 'button';
  modalCloseBtn.dataset.testid = 'dory-shortcuts-close-btn';
  modalCloseBtn.addEventListener('click', () => closeShortcutsModal());
  modalHeader.appendChild(modalCloseBtn);

  const modalList = el('dl', 'dory-shortcuts-list');
  for (let i = 0; i < SHORTCUT_ROWS.length; i += 1) {
    const row = SHORTCUT_ROWS[i]!;
    modalList.append(
      el('dt', 'dory-shortcut-key', row.key),
      el('dd', 'dory-shortcut-desc', row.desc),
    );
  }

  shortcutsDialog.append(modalHeader, modalList);
  shortcutsBackdrop.appendChild(shortcutsDialog);
  shortcutsBackdrop.addEventListener('click', (ev) => {
    if (ev.target === shortcutsBackdrop) {
      closeShortcutsModal();
    }
  });

  const shortcutsFocusTrap = createFocusTrap(shortcutsDialog, () => closeShortcutsModal());

  shell.append(topbar, workspace, shortcutsBackdrop);
  host.appendChild(shell);

  // --------------------------------------------------------------------------
  // WEBGL2 5-PASS INITIALIZATION & CONTEXT LOSS RECOVERY
  // --------------------------------------------------------------------------

  let glBundle: GlPassesBundle | null = null;

  function drainGlErrors(): void {
    if (!glBundle || glBundle.device.isLost) return;
    const gl = glBundle.device.gl;
    for (let guard = 0; guard < 16; guard += 1) {
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
          ? `WebGL2 renderer unavailable (${err.message}). 2D analytical <svg> charts, 4-Act Guided Tour, and readouts remain interactive.`
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
          err instanceof Error
            ? `WebGL2 restore failed: ${err.message}`
            : 'WebGL2 restore failed.';
      }
    }
    scheduler.requestFrame();
  };

  canvas.addEventListener('webglcontextlost', onContextLost);
  canvas.addEventListener('webglcontextrestored', onContextRestored);

  // --------------------------------------------------------------------------
  // VIEWPORT METRICS, CAMERA FITTING, GPU UPLOAD DEDUP & 5-PASS RENDER
  // --------------------------------------------------------------------------

  function readViewportMetrics(): ViewportMetrics {
    const rect = viewportHost.getBoundingClientRect();
    const w = Math.max(320, rect.width || viewportHost.clientWidth || 960);
    const h = Math.max(240, rect.height || viewportHost.clientHeight || 600);
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    return makeViewportMetrics(rect.left, rect.top, w, h, dpr, 0, 0);
  }

  function hasVisibleCharts(): boolean {
    return !chartsPanelCollapsed && chartsPanel.getRenderedChartKinds().length > 0;
  }

  function computePresetPose(scene: DorySpatialScene): CameraPose {
    const metrics = readViewportMetrics();
    const isTour = tour.isActive();
    const camOverride = isTour ? tour.getState().currentBeat.cameraOverride : null;
    const topInsetPx = isTour ? 116 : 86;
    const bottomInsetPx = hasVisibleCharts() ? 178 : 38;
    return fitOpBoundsCamera(scene.cameraPreset.fitBounds, metrics, {
      azimuthDeg: camOverride?.azimuthDeg ?? scene.cameraPreset.azimuthDeg,
      elevationDeg: camOverride?.elevationDeg ?? scene.cameraPreset.elevationDeg,
      topInsetPx,
      bottomInsetPx,
      fillFraction: 0.82,
    });
  }

  function flyToCurrentScenePreset(animate: boolean): void {
    const targetPose = computePresetPose(currentScene);
    reducedMotion = isReducedMotionPreferred();
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

  function resolveEffectiveAtlasKey(scene: DorySpatialScene): string | null {
    if (!scene.valueAtlas) return null;
    if (scene.atlasKey !== null && scene.atlasKey.length > 0) {
      return scene.atlasKey;
    }
    return computeDoryAtlasContentKey(scene.viewMode, scene.valueAtlas);
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

    const effectiveKey = resolveEffectiveAtlasKey(scene);
    if (scene.valueAtlas && atlasTracker.shouldUpload(effectiveKey)) {
      glBundle.sheets.uploadValueAtlas(
        scene.valueAtlas.width,
        scene.valueAtlas.height,
        scene.valueAtlas.data,
      );
    }

    sceneDirtyForGl = false;
    drainGlErrors();
  }

  function computeOverlayAvoidRects(metrics: ViewportMetrics): DoryOverlayRect[] {
    const avoid: DoryOverlayRect[] = [];
    const hostX = metrics.hostOriginX;
    const hostY = metrics.hostOriginY;

    const addRect = (node: HTMLElement | null): void => {
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

    addRect(stageTitleCard);
    if (!selfSpecBanner.hidden) addRect(selfSpecBanner);
    if (!readOnlyBanner.hidden) addRect(readOnlyBanner);
    if (!chartsInsetHost.hidden) addRect(chartsInsetHost);
    if (!tourHud.hidden) addRect(tourHud);
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
      bottomInsetPx: hasVisibleCharts() ? 158 : 20,
      avoidRects: computeOverlayAvoidRects(metrics),
      showCellNumbers: true,
    });
  }

  // --------------------------------------------------------------------------
  // IDLE-RAF SCHEDULER (`rafActive === false` AT REST)
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
  // SESSION, READOUTS, CHARTS & HUD SYNCHRONIZATION
  // --------------------------------------------------------------------------

  function getActiveSession(): DorySession {
    return tour.isActive() ? tour.tourSession : freeSession;
  }

  function buildCurrentDiscrepancyFilter(): DoryDiscrepancyFilterOptions {
    return {
      query: ledgerQuery,
      category:
        ledgerFilterMode === 'all' || ledgerFilterMode === 'browser_guard'
          ? 'all'
          : ledgerFilterMode,
      onlyBrowserGuards: ledgerFilterMode === 'browser_guard',
    };
  }

  function refreshReadoutsAndCharts(): void {
    const activeSess = getActiveSession();
    currentReadouts = buildDoryArchReadouts({
      activeKLoops: currentScene.kLoops,
      activeKvCacheMode: currentScene.kvCacheMode,
      selectedCheckpointFormat: focus.selectedCheckpointFormat,
      liveSessionMemory: activeSess.measureMemory(),
      activeAtlasBytes: currentScene.valueAtlas?.data.byteLength ?? 0,
      discrepancyFilter: buildCurrentDiscrepancyFilter(),
    });

    if (chartsPanelCollapsed) {
      chartsPanel.clear();
      chartsInsetHost.hidden = true;
    } else {
      const renderedKinds: readonly DorySvgChartKind[] = chartsPanel.renderForScene(
        currentScene,
        currentReadouts,
      );
      chartsInsetHost.hidden = renderedKinds.length === 0;
    }
    chartsInsetHost.classList.toggle('is-tour-active', tour.isActive());
  }

  function refreshStageHud(): void {
    stageTitleEl.textContent = currentScene.title;
    stageSubtitleEl.textContent = currentScene.subtitle;

    stageBadgesEl.innerHTML = '';
    for (let i = 0; i < currentScene.provenanceBadges.length; i += 1) {
      const b = currentScene.provenanceBadges[i]!;
      stageBadgesEl.appendChild(el('span', `dory-badge dory-badge--${b.tone}`, b.label));
    }

    const showSpecBanner =
      currentScene.viewMode === 'self_spec' || currentScene.executionMode === 'self_spec';
    const showReadOnlyBanner = currentScene.viewMode === 'bookends';

    selfSpecBanner.hidden = !showSpecBanner;
    readOnlyBanner.hidden = !showReadOnlyBanner;
    tourHud.classList.toggle('has-banner', showSpecBanner || showReadOnlyBanner);
    stageTitleCard.classList.toggle('is-tour-active', tour.isActive());
  }

  function syncTopViewTabs(): void {
    const idx = DORY_SPATIAL_VIEW_MODES.findIndex((v) => v.id === viewMode);
    syncRovingTabindex(viewTabButtons, idx >= 0 ? idx : 0);
    for (let i = 0; i < viewTabButtons.length; i += 1) {
      viewTabButtons[i]!.classList.toggle('is-active', i === idx);
    }
    tourLaunchBtn.classList.toggle('is-active', tour.isActive());
    tourLaunchBtn.textContent = tour.isActive()
      ? 'Exit Tour'
      : `Guided Tour (${tour.totalBeats})`;
  }

  function syncInspectorTabs(): void {
    const idx = INSPECTOR_TABS.findIndex((t) => t.id === inspectorTab);
    syncRovingTabindex(inspectorTabButtons, idx >= 0 ? idx : 0);
    for (let i = 0; i < inspectorTabButtons.length; i += 1) {
      inspectorTabButtons[i]!.classList.toggle('is-active', i === idx);
    }
  }

  function refreshDock(): void {
    const activeSess = getActiveSession();
    const dockState = buildDoryTimelineDockState(activeSess, {
      promptPresetId,
      allowExtrapolationDepths,
      isPlaying,
      tourActive: tour.isActive(),
      dr18ForkNotice: tour.isActive() ? null : dr18ForkNotice,
    });
    timelineDock.update(dockState);
  }

  function rebuildSceneAndRefresh(opts?: {
    readonly refitCamera?: boolean;
    readonly animateCamera?: boolean;
    readonly syncRender?: boolean;
    readonly preserveViewControlsDom?: boolean;
  }): void {
    const activeSess = getActiveSession();
    if (tour.isActive()) {
      const tourState = tour.getState();
      viewMode = tourState.currentBeat.viewMode;
      focus = tourState.focus;
      selectedPartId = tourState.currentBeat.highlightPartId ?? null;
      currentScene = tour.buildBeatScene(tourState.beatIndex);
    } else {
      const activeRec = activeSess.getActiveRecord();
      const activeK = activeRec?.kLoops ?? DORY_MINI_CONFIG.defaultRecurrentLoops;
      const clampedLoop = Math.max(1, Math.min(activeK, focus.selectedLoopIteration));
      const clampedCfLoop = Math.max(1, Math.min(activeK, focus.counterfactualLoop));
      const nextFocus: DorySpatialFocusState = {
        ...focus,
        selectedLoopIteration: clampedLoop,
        counterfactualLoop: clampedCfLoop,
        selfSpecPreviewSampling: sampling,
        hoveredPartId: selectedPartId,
        hoveredCell,
        ...(activeRec !== null ? { recordId: activeRec.recordId } : {}),
      };
      focus = nextFocus;
      currentScene = buildDorySpatialScene(activeSess, viewMode, focus);
    }

    sceneDirtyForGl = true;
    syncTopViewTabs();
    refreshStageHud();
    refreshReadoutsAndCharts();
    refreshDock();
    renderInspectorBody({
      preserveViewControlsDom: opts?.preserveViewControlsDom === true,
    });

    if (opts?.refitCamera) {
      flyToCurrentScenePreset(opts.animateCamera === true);
    } else if (opts?.syncRender) {
      renderFrameNow();
    } else {
      scheduler.requestFrame();
    }
  }

  // --------------------------------------------------------------------------
  // VIEW, DR18 DEPTH/POLICY/MODE FORK & TRAJECTORY ACTIONS
  // --------------------------------------------------------------------------

  function setViewInternal(nextView: DorySpatialViewMode, animateCamera: boolean): void {
    viewMode = nextView;
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    rebuildSceneAndRefresh({
      refitCamera: true,
      animateCamera,
      syncRender: !animateCamera,
    });
    const desc = DORY_SPATIAL_VIEW_MODES.find((d) => d.id === nextView);
    if (desc) {
      announce(`Switched to View ${desc.viewNumber}: ${desc.label}`);
    }
  }

  function updateFocusInternal(
    patch: Partial<DorySpatialFocusState>,
    syncRender = false,
  ): void {
    if (tour.isActive()) return;
    focus = {
      ...focus,
      ...patch,
    };
    rebuildSceneAndRefresh({
      refitCamera: false,
      syncRender,
      preserveViewControlsDom: true,
    });
  }

  /**
   * `D6` / `DR18` immutable depth switch: NEVER mutates populated KV slabs — always forks and
   * re-prefills a clean branch from prompt tokens & seed.
   */
  function selectKLoopsInternal(nextK: DorySupportedLoopCount, syncRender = false): void {
    if (tour.isActive()) return;
    stopPlayback();
    if (nextK > 4) {
      allowExtrapolationDepths = true;
    }

    const curBranch = freeSession.getActiveBranch();
    if (
      curBranch &&
      curBranch.config.kLoops === nextK &&
      (curBranch.config.executionMode === 'standard' || nextK === DORY_SPEC_VERIFY_LOOPS)
    ) {
      return;
    }

    const prevBranchId = curBranch?.branchId ?? 'root';
    const nextExecMode: DoryExecutionMode =
      curBranch?.config.executionMode === 'self_spec' && nextK !== DORY_SPEC_VERIFY_LOOPS
        ? 'standard'
        : (curBranch?.config.executionMode ?? 'standard');

    const newRec = freeSession.switchDepthOrPolicy({
      kLoops: nextK,
      executionMode: nextExecMode,
      sampling,
    });
    try {
      freeSession.step();
    } catch {
      // Keep prefill cleanly if halted
    }

    const extraNote =
      curBranch?.config.executionMode === 'self_spec' && nextExecMode === 'standard'
        ? ' (exited Contract A self-spec per DR23)'
        : '';
    dr18ForkNotice =
      `DR18 Immutable Fork: k=${nextK} (${computeExecutedDepth(nextK)}L exec) spawned clean branch ` +
      `${newRec.branchId}${extraNote} — prior branch ${prevBranchId} preserved untouched.`;
    focus = {
      ...focus,
      selectedLoopIteration: Math.min(nextK, Math.max(1, focus.selectedLoopIteration)),
      counterfactualLoop: Math.min(nextK, Math.max(1, focus.counterfactualLoop)),
    };
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    announce(dr18ForkNotice);
    rebuildSceneAndRefresh({
      refitCamera: viewMode === 'macro',
      animateCamera: !syncRender,
      syncRender,
    });
  }

  function toggleExtrapolationDepthsInternal(enabled: boolean): void {
    if (tour.isActive()) return;
    allowExtrapolationDepths = enabled;
    refreshDock();
  }

  /**
   * `D6` / `DR18` immutable KV-policy switch (`per_loop <-> last_loop`): NEVER mutates
   * existing KV slabs — always forks and re-prefills cleanly (preserving chunk schedule under
   * `last_loop` so `DR17`/`DR24` partition effects are immediately inspectable!).
   */
  function selectKvCacheModeInternal(mode: DoryKvCacheMode, syncRender = false): void {
    if (tour.isActive()) return;
    stopPlayback();
    const curBranch = freeSession.getActiveBranch();
    if (curBranch && curBranch.config.kvCacheMode === mode) {
      return;
    }

    const prevBranchId = curBranch?.branchId ?? 'root';
    const nextExecMode: DoryExecutionMode =
      mode === 'last_loop' ? 'standard' : (curBranch?.config.executionMode ?? 'standard');
    const preset = getDoryPromptPreset(promptPresetId);
    const usePartition =
      mode === 'last_loop' &&
      nextExecMode === 'standard' &&
      curBranch !== null &&
      curBranch.promptTokens.length === preset.tokens.length;

    const newRec = freeSession.switchDepthOrPolicy({
      kvCacheMode: mode,
      executionMode: nextExecMode,
      sampling,
      ...(usePartition ? { prefillChunkSizes: preset.defaultChunkPartition } : {}),
    });
    try {
      freeSession.step();
    } catch {
      // Keep prefill cleanly if halted
    }

    const extraNote =
      curBranch?.config.executionMode === 'self_spec' && mode === 'last_loop'
        ? ' (exited Contract A per DR23: last_loop incompatible with self_spec)'
        : '';
    dr18ForkNotice =
      `DR18 Immutable Fork: KV policy '${mode}' spawned clean branch ${newRec.branchId}` +
      `${extraNote} — prior branch ${prevBranchId} preserved untouched.`;
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    announce(dr18ForkNotice);
    rebuildSceneAndRefresh({
      refitCamera: false,
      syncRender,
    });
  }

  /**
   * `D3` / `D6` / `DR18` / `DR23` execution-mode switch (`standard <-> self_spec`):
   * Entering Contract A `self_spec` forks a clean `104`-bank `k = 4, per_loop` branch and
   * executes 1 live `gamma = 4` cycle so View 6 immediately showcases real reuse & rejection data!
   */
  function selectExecutionModeInternal(mode: DoryExecutionMode, syncRender = false): void {
    if (tour.isActive()) return;
    stopPlayback();
    const curBranch = freeSession.getActiveBranch();
    if (curBranch && curBranch.config.executionMode === mode) {
      return;
    }

    const prevBranchId = curBranch?.branchId ?? 'root';
    const newRec =
      mode === 'self_spec'
        ? freeSession.switchDepthOrPolicy({
            executionMode: 'self_spec',
            kvCacheMode: 'per_loop',
            kLoops: DORY_SPEC_VERIFY_LOOPS,
            sampling,
          })
        : freeSession.switchDepthOrPolicy({
            executionMode: 'standard',
            sampling,
          });
    try {
      freeSession.step();
    } catch {
      // Keep prefill cleanly if halted
    }

    dr18ForkNotice =
      mode === 'self_spec'
        ? `DR18 + DR23 Immutable Fork: Contract A Self-Spec (k=1->4, per_loop, 104 banks) ` +
          `spawned clean branch ${newRec.branchId} — prior branch ${prevBranchId} preserved.`
        : `DR18 Immutable Fork: Standard decode spawned clean branch ${newRec.branchId} ` +
          `— prior branch ${prevBranchId} preserved.`;
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    announce(dr18ForkNotice);
    rebuildSceneAndRefresh({
      refitCamera: false,
      syncRender,
    });
  }

  function selectPromptPresetInternal(nextPresetId: DoryPromptPresetId, syncRender = false): void {
    if (tour.isActive()) return;
    stopPlayback();
    promptPresetId = nextPresetId;
    const preset = getDoryPromptPreset(nextPresetId);
    const curBranch = freeSession.getActiveBranch();
    const k = curBranch?.config.kLoops ?? DORY_MINI_CONFIG.defaultRecurrentLoops;
    const kvMode = curBranch?.config.kvCacheMode ?? DORY_MINI_CONFIG.defaultKvCacheMode;
    const execMode = curBranch?.config.executionMode ?? 'standard';

    const newRec = freeSession.startBranch({
      promptTokens: preset.tokens,
      kLoops: execMode === 'self_spec' ? DORY_SPEC_VERIFY_LOOPS : k,
      kvCacheMode: execMode === 'self_spec' ? 'per_loop' : kvMode,
      executionMode: execMode,
      sampling,
      ...(execMode === 'standard' && kvMode === 'last_loop'
        ? { prefillChunkSizes: preset.defaultChunkPartition }
        : {}),
    });
    try {
      freeSession.step();
    } catch {
      // Keep prefill cleanly if halted
    }

    dr18ForkNotice =
      `Prompt preset '${preset.shortLabel}' (${preset.tokens.length} tok) started branch ${newRec.branchId}.`;
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    rebuildSceneAndRefresh({
      refitCamera: true,
      animateCamera: !syncRender,
      syncRender,
    });
  }

  function updateSamplingInternal(patch: Partial<DorySamplingConfig>): void {
    if (tour.isActive()) return;
    const temp =
      patch.temperature !== undefined && Number.isFinite(patch.temperature)
        ? Math.max(0, Math.min(1.5, patch.temperature))
        : sampling.temperature;
    const topK =
      patch.topK !== undefined && Number.isInteger(patch.topK) && patch.topK >= 0
        ? Math.min(DORY_MINI_CONFIG.vocabSize, patch.topK)
        : sampling.topK;
    const topP =
      patch.topP !== undefined && Number.isFinite(patch.topP)
        ? Math.max(0.05, Math.min(1.0, patch.topP))
        : sampling.topP;
    sampling = Object.freeze({ temperature: temp, topK, topP });

    // Preserve slider DOM identity strictly in-place; refresh preview sampling in View 6 without branch spam
    if (viewMode === 'self_spec') {
      focus = { ...focus, selfSpecPreviewSampling: sampling };
      rebuildSceneAndRefresh({ preserveViewControlsDom: true });
    } else {
      refreshDock();
    }
  }

  function stepCursorBy(delta: number): void {
    if (tour.isActive()) return;
    const branch = freeSession.getActiveBranch();
    if (!branch || branch.recordIds.length === 0) return;
    const curRec = freeSession.getActiveRecord();
    const curIdx = curRec ? branch.recordIds.indexOf(curRec.recordId) : branch.recordIds.length - 1;
    const nextIdx = Math.max(0, Math.min(branch.recordIds.length - 1, curIdx + delta));
    const targetId = branch.recordIds[nextIdx];
    if (targetId && targetId !== curRec?.recordId) {
      freeSession.seekRecord(targetId);
      rebuildSceneAndRefresh({ preserveViewControlsDom: true });
    }
  }

  function scrubToRecordIndex(indexInBranch: number): void {
    if (tour.isActive()) return;
    const branch = freeSession.getActiveBranch();
    if (!branch || branch.recordIds.length === 0) return;
    const clamped = Math.max(0, Math.min(branch.recordIds.length - 1, Math.trunc(indexInBranch)));
    const targetId = branch.recordIds[clamped];
    if (targetId) {
      freeSession.seekRecord(targetId);
      rebuildSceneAndRefresh({ preserveViewControlsDom: true });
    }
  }

  function executeNextStep(syncRender = false): void {
    if (tour.isActive()) return;
    const curBranch = freeSession.getActiveBranch();
    const curRec = freeSession.getActiveRecord();
    if (!curBranch || !curRec) return;

    // If user adjusted sampling temperature since branch creation, fork cleanly with updated sampling
    if (!sameSampling(curBranch.config.sampling, sampling)) {
      const forked = freeSession.forkBranch(curRec.recordId, {
        sampling,
      });
      dr18ForkNotice =
        `Sampling T=${sampling.temperature.toFixed(2)} forked non-aliased branch ${forked.branchId}.`;
    }

    try {
      freeSession.step();
    } catch {
      // Branch already halted (eos / capacity_exhausted)
      refreshDock();
      return;
    }
    rebuildSceneAndRefresh({
      refitCamera: false,
      syncRender,
      preserveViewControlsDom: true,
    });
  }

  function forkCurrentBranchInternal(syncRender = false): void {
    if (tour.isActive()) return;
    const curRec = freeSession.getActiveRecord();
    if (!curRec) return;
    const forked = freeSession.forkBranch(curRec.recordId, { sampling });
    dr18ForkNotice =
      `Immutable CoW Fork: created branch ${forked.branchId} from ${curRec.recordId} (non-aliased slabs).`;
    announce(dr18ForkNotice);
    rebuildSceneAndRefresh({
      refitCamera: false,
      syncRender,
      preserveViewControlsDom: true,
    });
  }

  function switchToBranchInternal(branchId: string, syncRender = false): void {
    if (tour.isActive()) return;
    stopPlayback();
    const target = freeSession.getBranch(branchId);
    if (!target) return;
    freeSession.setActiveBranch(branchId);
    sampling = target.config.sampling;
    if (target.config.kLoops > 4) {
      allowExtrapolationDepths = true;
    }
    dr18ForkNotice = null;
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    rebuildSceneAndRefresh({
      refitCamera: false,
      syncRender,
    });
  }

  function resetSessionInternal(syncRender = false): void {
    if (tour.isActive()) return;
    stopPlayback();
    const preset = getDoryPromptPreset(promptPresetId);
    const curBranch = freeSession.getActiveBranch();
    const k = curBranch?.config.kLoops ?? DORY_MINI_CONFIG.defaultRecurrentLoops;
    const kvMode = curBranch?.config.kvCacheMode ?? DORY_MINI_CONFIG.defaultKvCacheMode;
    const execMode = curBranch?.config.executionMode ?? 'standard';
    freeSession.reset({
      promptTokens: preset.tokens,
      kLoops: execMode === 'self_spec' ? DORY_SPEC_VERIFY_LOOPS : k,
      kvCacheMode: execMode === 'self_spec' ? 'per_loop' : kvMode,
      executionMode: execMode,
      sampling,
    });
    try {
      freeSession.step();
    } catch {
      // Keep prefill cleanly
    }
    dr18ForkNotice = null;
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;
    rebuildSceneAndRefresh({
      refitCamera: true,
      animateCamera: !syncRender,
      syncRender,
    });
  }

  function stopPlayback(): void {
    if (!isPlaying) return;
    isPlaying = false;
    playbackAccumMs = 0;
    scheduler.setContinuousReason('playback', false);
    refreshDock();
  }

  function togglePlayback(): void {
    if (tour.isActive()) return;
    if (isPlaying) {
      stopPlayback();
      return;
    }
    const branch = freeSession.getActiveBranch();
    if (!branch || branch.recordIds.length <= 1) return;
    isPlaying = true;
    playbackAccumMs = 0;
    scheduler.setContinuousReason('playback', true);
    refreshDock();
  }

  function advancePlaybackStep(): void {
    const branch = freeSession.getActiveBranch();
    if (!branch || branch.recordIds.length <= 1) {
      stopPlayback();
      return;
    }
    const curRec = freeSession.getActiveRecord();
    const curIdx = curRec ? branch.recordIds.indexOf(curRec.recordId) : 0;
    if (curIdx >= branch.recordIds.length - 1) {
      stopPlayback();
      return;
    }
    freeSession.seekRecord(branch.recordIds[curIdx + 1]!);
    rebuildSceneAndRefresh({ preserveViewControlsDom: true });
  }

  // --------------------------------------------------------------------------
  // 4-ACT / 14-BEAT GUIDED TOUR HUD INTEGRATION (ISOLATED PRIVATE SESSION)
  // --------------------------------------------------------------------------

  function updateTourHudDom(state: DoryTourState): void {
    tourHud.hidden = !state.active;
    if (!state.active) return;

    const beat = state.currentBeat;
    tourBeatCounter.textContent =
      `Act ${beat.actNumber} · Beat ${beat.beatIndex + 1} / ${state.beatCount}`;
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

    // Live tensor metrics pills
    tourMetricsEl.innerHTML = '';
    for (let i = 0; i < beat.liveMetrics.length; i += 1) {
      const m = beat.liveMetrics[i]!;
      const pill = el('span', `dory-tour-metric dory-badge--${m.tone}`);
      pill.textContent = `${m.label}: ${m.value}`;
      tourMetricsEl.appendChild(pill);
    }

    // Full immutable 8-tuple DoryExecutionCoord provenance strip (Audit §6.3)
    const c = beat.coord;
    tourCoordEl.textContent =
      `Coord · p=${c.physicalLayerIdx} (Pair L${c.physicalLayerIdx >> 1}) · e=${c.executedLayerIdx} · ` +
      `t=${c.loopIteration}/${beat.kLoops} · pos=${c.absTokenPos} · step=${c.forwardStepId} · ` +
      `${c.sessionBranchId} · ${c.cacheOwnership.policy}/${c.cacheOwnership.bankKind}[${c.cacheOwnership.bankSlotId}] ` +
      `writer=t${c.cacheOwnership.writerLoop} (${c.cacheOwnership.role})`;

    tourEpistemicEl.textContent = beat.epistemicNote;

    for (let i = 0; i < tourActButtons.length; i += 1) {
      tourActButtons[i]!.classList.toggle('is-active', i === state.actIndex);
    }

    tourRefsEl.innerHTML = '';
    for (let i = 0; i < beat.discrepancyRefs.length; i += 1) {
      const refId = beat.discrepancyRefs[i]!;
      const chip = el('button', 'dory-tour-ref-chip', refId);
      chip.type = 'button';
      chip.title = `Inspect ${refId} in Discrepancy & Guard Ledger`;
      chip.addEventListener('click', () => {
        if (inspectorCollapsed) {
          setInspectorCollapsedInternal(false);
        }
        inspectorTab = 'ledger';
        ledgerFilterMode = 'all';
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
    const restore: DoryFreeExploreRestoreState = {
      viewMode,
      focus: { ...focus },
      activeBranchId: freeSession.getActiveBranch()?.branchId ?? null,
      activeRecordId: freeSession.getActiveRecord()?.recordId ?? null,
      cameraPose: cameraPose ? cloneCameraPose(cameraPose) : null,
      inspectorTab,
      inspectorOpen: !inspectorCollapsed,
    };
    const state = tour.start(restore, beatIndex);
    hoveredCell = null;
    selectedCell = null;
    updateTourHudDom(state);
    announce(`Started Guided Tour: Act ${state.currentBeat.actNumber}, Beat ${state.beatIndex + 1}: ${state.currentBeat.title}`);
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
    announce(`Beat ${state.beatIndex + 1} of ${state.beatCount}: ${state.currentBeat.title}`);
    rebuildSceneAndRefresh({
      refitCamera: true,
      animateCamera: !syncRender,
      syncRender,
    });
  }

  function goToTourActInternal(actId: DoryTourActId, syncRender = false): void {
    if (!tour.isActive()) {
      const targetAct = tour.acts.find((a) => a.id === actId);
      startTourInternal(targetAct?.startBeatIndex ?? 0, syncRender);
      return;
    }
    const state = tour.goToAct(actId);
    hoveredCell = null;
    selectedCell = null;
    updateTourHudDom(state);
    announce(`${state.currentAct.title} — Beat ${state.beatIndex + 1}: ${state.currentBeat.title}`);
    rebuildSceneAndRefresh({
      refitCamera: true,
      animateCamera: !syncRender,
      syncRender,
    });
  }

  function exitTourInternal(restoreFreeExplore: boolean, syncRender = false): void {
    if (!tour.isActive()) return;
    const focusWasInTour =
      typeof document !== 'undefined' &&
      document.activeElement !== null &&
      tourHud.contains(document.activeElement);
    const restored = tour.exit();
    tourHud.hidden = true;
    tourHud.classList.remove('has-banner');
    stageTitleCard.classList.remove('is-tour-active');
    chartsInsetHost.classList.remove('is-tour-active');
    if (focusWasInTour && typeof tourLaunchBtn.focus === 'function') {
      tourLaunchBtn.focus();
    }
    hoveredCell = null;
    selectedCell = null;
    selectedPartId = null;

    if (restoreFreeExplore && restored) {
      viewMode = restored.viewMode;
      focus = { ...restored.focus };
      if (restored.activeRecordId && freeSession.getRecord(restored.activeRecordId)) {
        freeSession.seekRecord(restored.activeRecordId);
      } else if (restored.activeBranchId && freeSession.getBranch(restored.activeBranchId)) {
        freeSession.setActiveBranch(restored.activeBranchId);
      }
      if (restored.inspectorTab) {
        inspectorTab = restored.inspectorTab;
        syncInspectorTabs();
      }
      if (restored.inspectorOpen !== undefined) {
        applyInspectorCollapsedDom(!restored.inspectorOpen);
      }
      if (restored.cameraPose) {
        cameraPose = cloneCameraPose(restored.cameraPose);
      }
      announce('Exited Guided Tour and restored Free-Explore session.');
      rebuildSceneAndRefresh({
        refitCamera: !restored.cameraPose,
        animateCamera: !syncRender,
        syncRender,
      });
    } else {
      syncTopViewTabs();
      refreshDock();
    }
  }

  // --------------------------------------------------------------------------
  // MODAL, CHARTS & INSPECTOR DRAWER TOGGLES
  // --------------------------------------------------------------------------

  function openShortcutsModal(): void {
    if (shortcutsModalOpen) return;
    shortcutsModalOpen = true;
    shortcutsBackdrop.hidden = false;
    shortcutsFocusTrap.activate();
  }

  function closeShortcutsModal(): void {
    if (!shortcutsModalOpen) return;
    shortcutsModalOpen = false;
    shortcutsBackdrop.hidden = true;
    shortcutsFocusTrap.deactivate();
  }

  function setInspectorTabInternal(tab: DoryInspectorTabId): void {
    inspectorTab = tab;
    syncInspectorTabs();
    renderInspectorBody();
  }

  function applyInspectorCollapsedDom(collapsed: boolean): void {
    inspectorCollapsed = collapsed;
    inspectorDrawer.classList.toggle('is-collapsed', inspectorCollapsed);
    inspectorToggleBtn.textContent = inspectorCollapsed ? 'Show Inspector' : 'Hide Inspector';
    inspectorToggleBtn.setAttribute('aria-expanded', inspectorCollapsed ? 'false' : 'true');
  }

  function setInspectorCollapsedInternal(collapsed: boolean, syncRender = false): void {
    applyInspectorCollapsedDom(collapsed);
    if (syncRender) {
      renderFrameNow();
    } else {
      scheduler.requestFrame();
    }
  }

  function toggleInspectorCollapsed(): void {
    setInspectorCollapsedInternal(!inspectorCollapsed);
  }

  function toggleChartsPanelCollapsed(): void {
    chartsPanelCollapsed = !chartsPanelCollapsed;
    chartsToggleBtn.textContent = chartsPanelCollapsed ? 'Show Charts' : 'Hide Charts';
    chartsToggleBtn.setAttribute('aria-expanded', chartsPanelCollapsed ? 'false' : 'true');
    refreshReadoutsAndCharts();
    scheduler.requestFrame();
  }

  // --------------------------------------------------------------------------
  // COLLAPSIBLE 3-TAB INSPECTOR DRAWER RENDERING (IN-PLACE CONTROLS SLOT!)
  // --------------------------------------------------------------------------

  function syncSliderValue(slider: HTMLInputElement, nextVal: string): void {
    if (typeof document !== 'undefined' && document.activeElement === slider) return;
    if (slider.value !== nextVal) slider.value = nextVal;
  }

  function renderViewControlsSection(container: HTMLElement, activeKLoops: number): () => void {
    const tourLocked = tour.isActive();
    const sec = el('section', 'dory-inspector-section');
    const viewDesc = DORY_SPATIAL_VIEW_MODES.find((d) => d.id === viewMode);
    sec.appendChild(
      el(
        'h3',
        'dory-section-heading',
        `View ${viewDesc?.viewNumber ?? ''} Controls · ${viewDesc?.shortLabel ?? viewMode}`,
      ),
    );

    if (tourLocked) {
      const lockNote = el(
        'p',
        'dory-tour-lock-note',
        'Guided Tour Active — Exit Tour to customize view controls',
      );
      lockNote.dataset.testid = 'dory-tour-lock-note';
      sec.appendChild(lockNote);
    }

    const syncFns: (() => void)[] = [];

    const addLoopPassSelector = (labelText: string): void => {
      const loopRow = el('div', 'dory-control-row');
      const loopLbl = el('div', 'dory-control-label');
      const loopValEl = el('span', 'dory-control-value');
      loopLbl.append(el('span', undefined, labelText), loopValEl);
      const chips = el('div', 'dory-chip-group');
      const btns: { readonly loop: number; readonly btn: HTMLButtonElement }[] = [];
      for (let t = 1; t <= activeKLoops; t += 1) {
        const b = el('button', 'dory-chip-btn dory-chip-btn--teal', `t=${t}`);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, selectedLoopIteration: t };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        btns.push({ loop: t, btn: b });
        chips.appendChild(b);
      }
      syncFns.push(() => {
        loopValEl.textContent = `t=${focus.selectedLoopIteration} / ${activeKLoops}`;
        for (const item of btns) {
          item.btn.classList.toggle('is-active', focus.selectedLoopIteration === item.loop);
        }
      });
      loopRow.append(loopLbl, chips);
      sec.appendChild(loopRow);
    };

    // View 1: Folded / Unrolled Macro Architecture
    if (viewMode === 'macro') {
      const modeRow = el('div', 'dory-control-row');
      modeRow.appendChild(el('div', 'dory-control-label', 'Macro Topology Display'));
      const modeChips = el('div', 'dory-chip-group');
      const foldedBtn = el('button', 'dory-chip-btn dory-chip-btn--indigo', 'Folded (72L Phys)');
      foldedBtn.type = 'button';
      foldedBtn.dataset.testid = 'dory-macro-folded-btn';
      foldedBtn.disabled = tourLocked;
      foldedBtn.addEventListener('click', () => {
        if (tour.isActive()) return;
        focus = { ...focus, macroDisplayMode: 'folded' };
        rebuildSceneAndRefresh({ refitCamera: true, animateCamera: true, preserveViewControlsDom: true });
      });

      const unrolledBtn = el(
        'button',
        'dory-chip-btn dory-chip-btn--teal',
        `Unrolled (${computeExecutedDepth(activeKLoops)}L Exec)`,
      );
      unrolledBtn.type = 'button';
      unrolledBtn.dataset.testid = 'dory-macro-unrolled-btn';
      unrolledBtn.disabled = tourLocked;
      unrolledBtn.addEventListener('click', () => {
        if (tour.isActive()) return;
        focus = { ...focus, macroDisplayMode: 'unrolled' };
        rebuildSceneAndRefresh({ refitCamera: true, animateCamera: true, preserveViewControlsDom: true });
      });

      syncFns.push(() => {
        foldedBtn.classList.toggle('is-active', focus.macroDisplayMode === 'folded');
        unrolledBtn.classList.toggle('is-active', focus.macroDisplayMode === 'unrolled');
      });
      modeChips.append(foldedBtn, unrolledBtn);
      modeRow.appendChild(modeChips);
      sec.appendChild(modeRow);

      // Physical Layer Slider (0..71) — stable DOM node preserved mid-drag!
      const layerRow = el('div', 'dory-control-row');
      const layerLbl = el('div', 'dory-control-label');
      const layerValEl = el('span', 'dory-control-value');
      layerLbl.append(el('span', undefined, 'Physical Sublayer (0..71)'), layerValEl);

      const presetChips = el('div', 'dory-chip-group');
      const presetBtns: { readonly idx: number; readonly btn: HTMLButtonElement }[] = [];
      for (const preset of MACRO_LAYER_PRESETS) {
        const b = el('button', 'dory-chip-btn', preset.label);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, selectedPhysicalLayerIdx: preset.idx };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        presetBtns.push({ idx: preset.idx, btn: b });
        presetChips.appendChild(b);
      }

      const slider = el('input', 'dory-range-input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = '71';
      slider.disabled = tourLocked;
      slider.dataset.testid = 'dory-layer-slider';
      slider.setAttribute('aria-label', 'Select physical sublayer 0 to 71');
      const syncLayer = (): void => {
        const p = focus.selectedPhysicalLayerIdx;
        const spec = getDoryPhysicalLayerSpec(p);
        layerValEl.textContent = `p=${p} (${spec.transformersPairBadge})`;
        for (const item of presetBtns) {
          item.btn.classList.toggle('is-active', item.idx === p);
        }
        syncSliderValue(slider, String(p));
      };
      slider.addEventListener('input', () => {
        if (tour.isActive()) return;
        const val = Number.parseInt(slider.value, 10);
        if (!Number.isFinite(val)) return;
        focus = { ...focus, selectedPhysicalLayerIdx: Math.max(0, Math.min(71, val)) };
        syncLayer();
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
      });
      syncFns.push(syncLayer);
      layerRow.append(layerLbl, presetChips, slider);
      sec.appendChild(layerRow);

      addLoopPassSelector('Highlighted Recurrent Loop Pass');
    }

    // View 2: nGPT Hyperspherical Residual & Latent Trajectory Inspector
    if (viewMode === 'hypersphere') {
      const layerRow = el('div', 'dory-control-row');
      const layerLbl = el('div', 'dory-control-label');
      const layerValEl = el('span', 'dory-control-value');
      layerLbl.append(el('span', undefined, 'NLERP Sublayer (0..71)'), layerValEl);

      const quickChips = el('div', 'dory-chip-group');
      const quickPresets = [
        { idx: 16, label: 'P16 Planted >1 α' },
        { idx: 17, label: 'P17 Rec MLP' },
        { idx: 55, label: 'P55 Core Exit' },
        { idx: 0, label: 'P0 Prelude' },
      ];
      const qBtns: { readonly idx: number; readonly btn: HTMLButtonElement }[] = [];
      for (const p of quickPresets) {
        const b = el('button', 'dory-chip-btn', p.label);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, selectedPhysicalLayerIdx: p.idx };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        qBtns.push({ idx: p.idx, btn: b });
        quickChips.appendChild(b);
      }

      const slider = el('input', 'dory-range-input');
      slider.type = 'range';
      slider.min = '0';
      slider.max = '71';
      slider.disabled = tourLocked;
      slider.dataset.testid = 'dory-layer-slider';
      slider.setAttribute('aria-label', 'Select NLERP physical sublayer 0 to 71');
      const syncLayer = (): void => {
        const p = focus.selectedPhysicalLayerIdx;
        const spec = getDoryPhysicalLayerSpec(p);
        layerValEl.textContent = `p=${p} (${spec.transformersPairBadge})`;
        for (const item of qBtns) {
          item.btn.classList.toggle('is-active', item.idx === p);
        }
        syncSliderValue(slider, String(p));
      };
      slider.addEventListener('input', () => {
        if (tour.isActive()) return;
        const val = Number.parseInt(slider.value, 10);
        if (!Number.isFinite(val)) return;
        focus = { ...focus, selectedPhysicalLayerIdx: Math.max(0, Math.min(71, val)) };
        syncLayer();
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
      });
      syncFns.push(syncLayer);
      layerRow.append(layerLbl, quickChips, slider);
      sec.appendChild(layerRow);

      addLoopPassSelector('Recurrent Loop Pass');

      // Read-only Diagnostic Counterfactual Coda Probe (`writeKv = false`)
      const cfLabel = el('label', `dory-checkbox-label${tourLocked ? ' is-disabled' : ''}`);
      const cfCb = el('input');
      cfCb.type = 'checkbox';
      cfCb.dataset.testid = 'dory-counterfactual-toggle';
      cfCb.disabled = tourLocked;
      cfCb.addEventListener('change', () => {
        if (tour.isActive()) return;
        focus = { ...focus, includeCounterfactualCoda: cfCb.checked };
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
      });
      syncFns.push(() => {
        cfCb.checked = focus.includeCounterfactualCoda;
      });
      cfLabel.append(
        cfCb,
        document.createTextNode('Read-Only Diagnostic Counterfactual Coda (0 KV mutation)'),
      );
      sec.appendChild(cfLabel);

      const cfLoopRow = el('div', 'dory-control-row');
      const cfLoopLbl = el('div', 'dory-control-label');
      const cfLoopVal = el('span', 'dory-control-value');
      cfLoopLbl.append(el('span', undefined, 'Counterfactual Probe Loop'), cfLoopVal);
      const cfChips = el('div', 'dory-chip-group');
      const cfBtns: { readonly loop: number; readonly btn: HTMLButtonElement }[] = [];
      for (let t = 1; t <= activeKLoops; t += 1) {
        const b = el('button', 'dory-chip-btn dory-chip-btn--amber', `s_${t}`);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, counterfactualLoop: t, includeCounterfactualCoda: true };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        cfBtns.push({ loop: t, btn: b });
        cfChips.appendChild(b);
      }
      syncFns.push(() => {
        cfLoopVal.textContent = `s_${focus.counterfactualLoop}`;
        for (const item of cfBtns) {
          item.btn.classList.toggle('is-active', focus.counterfactualLoop === item.loop);
        }
      });
      cfLoopRow.append(cfLoopLbl, cfChips);
      sec.appendChild(cfLoopRow);
    }

    // View 3: Gated GQA & Dual-RoPE Cutaway (`SWA W=8` vs. `Full`)
    if (viewMode === 'gqa_rope') {
      const layerRow = el('div', 'dory-control-row');
      const layerLbl = el('div', 'dory-control-label');
      const layerValEl = el('span', 'dory-control-value');
      layerLbl.append(el('span', undefined, 'Attention Sublayer (even 0..70)'), layerValEl);

      const gqaChips = el('div', 'dory-chip-group');
      const gqaBtns: { readonly idx: number; readonly btn: HTMLButtonElement }[] = [];
      for (const p of GQA_LAYER_PRESETS) {
        const b = el('button', 'dory-chip-btn', p.label);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, gqaLayerIdx: p.idx };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        gqaBtns.push({ idx: p.idx, btn: b });
        gqaChips.appendChild(b);
      }

      const gqaSlider = el('input', 'dory-range-input');
      gqaSlider.type = 'range';
      gqaSlider.min = '0';
      gqaSlider.max = '70';
      gqaSlider.step = '2';
      gqaSlider.disabled = tourLocked;
      gqaSlider.dataset.testid = 'dory-gqa-layer-slider';
      gqaSlider.setAttribute('aria-label', 'Select even attention physical layer 0 to 70');
      const syncGqaLayer = (): void => {
        const p = focus.gqaLayerIdx & ~1;
        const spec = getDoryPhysicalLayerSpec(p);
        layerValEl.textContent = `p=${p} (${spec.transformersPairBadge})`;
        for (const item of gqaBtns) {
          item.btn.classList.toggle('is-active', item.idx === p);
        }
        syncSliderValue(gqaSlider, String(p));
      };
      gqaSlider.addEventListener('input', () => {
        if (tour.isActive()) return;
        const raw = Number.parseInt(gqaSlider.value, 10);
        if (!Number.isFinite(raw)) return;
        const even = Math.max(0, Math.min(70, raw & ~1));
        focus = { ...focus, gqaLayerIdx: even };
        syncGqaLayer();
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
      });
      syncFns.push(syncGqaLayer);
      layerRow.append(layerLbl, gqaChips, gqaSlider);
      sec.appendChild(layerRow);

      addLoopPassSelector('Recurrent Loop Pass');

      // 4:1 GQA Query Head Selector (`Q0..Q7 -> KV0/KV1`)
      const qRow = el('div', 'dory-control-row');
      const qLbl = el('div', 'dory-control-label');
      const qValEl = el('span', 'dory-control-value');
      qLbl.append(el('span', undefined, 'Query Head (4:1 GQA)'), qValEl);
      const qChips = el('div', 'dory-chip-group');
      const qBtns: { readonly h: number; readonly btn: HTMLButtonElement }[] = [];
      for (let h = 0; h < DORY_MINI_CONFIG.numAttentionHeads; h += 1) {
        const b = el('button', 'dory-chip-btn dory-chip-btn--teal', `Q${h}`);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, selectedQHead: h };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        qBtns.push({ h, btn: b });
        qChips.appendChild(b);
      }
      syncFns.push(() => {
        qValEl.textContent = `Q${focus.selectedQHead} -> KV${focus.selectedQHead >> 2}`;
        for (const item of qBtns) {
          item.btn.classList.toggle('is-active', focus.selectedQHead === item.h);
        }
      });
      qRow.append(qLbl, qChips);
      sec.appendChild(qRow);

      // 4-Plane NeoX Rotary Plane Selector (`0..3`)
      const planeRow = el('div', 'dory-control-row');
      const planeLbl = el('div', 'dory-control-label');
      const planeValEl = el('span', 'dory-control-value');
      planeLbl.append(el('span', undefined, 'Dual-RoPE Plane (0..3)'), planeValEl);
      const pChips = el('div', 'dory-chip-group');
      const pBtns: { readonly plane: number; readonly btn: HTMLButtonElement }[] = [];
      for (let j = 0; j < DORY_MINI_CONFIG.rotaryPlanes; j += 1) {
        const b = el('button', 'dory-chip-btn', `Plane ${j}`);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, selectedRotaryPlane: j };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        pBtns.push({ plane: j, btn: b });
        pChips.appendChild(b);
      }
      syncFns.push(() => {
        planeValEl.textContent = `Plane ${focus.selectedRotaryPlane} / 3`;
        for (const item of pBtns) {
          item.btn.classList.toggle('is-active', focus.selectedRotaryPlane === item.plane);
        }
      });
      planeRow.append(planeLbl, pChips);
      sec.appendChild(planeRow);
    }

    // View 4: Normalized SwiGLU Mixer Cutaway
    if (viewMode === 'swiglu') {
      const layerRow = el('div', 'dory-control-row');
      const layerLbl = el('div', 'dory-control-label');
      const layerValEl = el('span', 'dory-control-value');
      layerLbl.append(el('span', undefined, 'SwiGLU Sublayer (odd 1..71)'), layerValEl);

      const mlpChips = el('div', 'dory-chip-group');
      const mlpBtns: { readonly idx: number; readonly btn: HTMLButtonElement }[] = [];
      for (const p of SWIGLU_LAYER_PRESETS) {
        const b = el('button', 'dory-chip-btn', p.label);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, swigluLayerIdx: p.idx };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        mlpBtns.push({ idx: p.idx, btn: b });
        mlpChips.appendChild(b);
      }

      const mlpSlider = el('input', 'dory-range-input');
      mlpSlider.type = 'range';
      mlpSlider.min = '1';
      mlpSlider.max = '71';
      mlpSlider.step = '2';
      mlpSlider.disabled = tourLocked;
      mlpSlider.dataset.testid = 'dory-swiglu-layer-slider';
      mlpSlider.setAttribute('aria-label', 'Select odd MLP physical layer 1 to 71');
      const syncMlpLayer = (): void => {
        const p = focus.swigluLayerIdx | 1;
        const spec = getDoryPhysicalLayerSpec(p);
        layerValEl.textContent = `p=${p} (${spec.transformersPairBadge})`;
        for (const item of mlpBtns) {
          item.btn.classList.toggle('is-active', item.idx === p);
        }
        syncSliderValue(mlpSlider, String(p));
      };
      mlpSlider.addEventListener('input', () => {
        if (tour.isActive()) return;
        const raw = Number.parseInt(mlpSlider.value, 10);
        if (!Number.isFinite(raw)) return;
        const odd = Math.max(1, Math.min(71, raw | 1));
        focus = { ...focus, swigluLayerIdx: odd };
        syncMlpLayer();
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
      });
      syncFns.push(syncMlpLayer);
      layerRow.append(layerLbl, mlpChips, mlpSlider);
      sec.appendChild(layerRow);

      addLoopPassSelector('Recurrent Loop Pass');

      // Intermediate Channel Slider (0..23)
      const chRow = el('div', 'dory-control-row');
      const chLbl = el('div', 'dory-control-label');
      const chValEl = el('span', 'dory-control-value');
      chLbl.append(el('span', undefined, 'Intermediate Channel (0..23)'), chValEl);
      const chSlider = el('input', 'dory-range-input');
      chSlider.type = 'range';
      chSlider.min = '0';
      chSlider.max = String(DORY_MINI_CONFIG.intermediateSize - 1);
      chSlider.disabled = tourLocked;
      chSlider.dataset.testid = 'dory-swiglu-channel-slider';
      chSlider.setAttribute('aria-label', 'Select SwiGLU intermediate channel 0 to 23');
      const syncCh = (): void => {
        chValEl.textContent = `ch=${focus.swigluChannel}`;
        syncSliderValue(chSlider, String(focus.swigluChannel));
      };
      chSlider.addEventListener('input', () => {
        if (tour.isActive()) return;
        const val = Number.parseInt(chSlider.value, 10);
        if (!Number.isFinite(val)) return;
        focus = { ...focus, swigluChannel: Math.max(0, Math.min(23, val)) };
        syncCh();
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
      });
      syncFns.push(syncCh);
      chRow.append(chLbl, chSlider);
      sec.appendChild(chRow);

      // Progressive Accum Reveal Slider (0.00..1.00)
      const revRow = el('div', 'dory-control-row');
      const revLbl = el('div', 'dory-control-label');
      const revValEl = el('span', 'dory-control-value');
      revLbl.append(el('span', undefined, 'down_proj Accum Reveal'), revValEl);
      const revSlider = el('input', 'dory-range-input');
      revSlider.type = 'range';
      revSlider.min = '0';
      revSlider.max = '1';
      revSlider.step = '0.05';
      revSlider.disabled = tourLocked;
      revSlider.dataset.testid = 'dory-swiglu-reveal-slider';
      revSlider.setAttribute('aria-label', 'SwiGLU down_proj progressive accumulation reveal');
      const syncRev = (): void => {
        const pct = Math.round(focus.swigluAccumReveal * 100);
        revValEl.textContent = `${pct}%`;
        syncSliderValue(revSlider, String(focus.swigluAccumReveal));
      };
      revSlider.addEventListener('input', () => {
        if (tour.isActive()) return;
        const val = Number.parseFloat(revSlider.value);
        if (!Number.isFinite(val)) return;
        focus = { ...focus, swigluAccumReveal: Math.max(0, Math.min(1, val)) };
        syncRev();
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
      });
      syncFns.push(syncRev);
      revRow.append(revLbl, revSlider);
      sec.appendChild(revRow);
    }

    // View 5: Dual-Axis KV & `last_loop` Divergence Inspector
    if (viewMode === 'kv_compare') {
      const partRow = el('div', 'dory-control-row');
      partRow.appendChild(
        el('div', 'dory-control-label', 'Prefill Partition Probe (DR17 Theorems 1–3)'),
      );
      const pChips = el('div', 'dory-chip-group');
      const pBtns: { readonly id: DoryKvComparePartitionMode; readonly btn: HTMLButtonElement }[] =
        [];
      for (const choice of PARTITION_MODE_CHOICES) {
        const b = el('button', 'dory-chip-btn dory-chip-btn--amber', choice.label);
        b.type = 'button';
        b.dataset.testid = `dory-kv-part-${choice.id}`;
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, kvComparePartitionMode: choice.id };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        pBtns.push({ id: choice.id, btn: b });
        pChips.appendChild(b);
      }
      syncFns.push(() => {
        for (const item of pBtns) {
          item.btn.classList.toggle('is-active', focus.kvComparePartitionMode === item.id);
        }
      });
      partRow.appendChild(pChips);
      sec.appendChild(partRow);

      const kvLayerRow = el('div', 'dory-control-row');
      kvLayerRow.appendChild(el('div', 'dory-control-label', 'Compared Attention Bank Layer'));
      const lChips = el('div', 'dory-chip-group');
      const lBtns: { readonly idx: number; readonly btn: HTMLButtonElement }[] = [];
      for (const p of [
        { idx: 16, label: 'P16 Rec SWA' },
        { idx: 22, label: 'P22 Rec Full' },
        { idx: 0, label: 'P0 Input SWA' },
        { idx: 56, label: 'P56 Coda SWA' },
      ]) {
        const b = el('button', 'dory-chip-btn', p.label);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, kvCompareLayerIdx: p.idx };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        lBtns.push({ idx: p.idx, btn: b });
        lChips.appendChild(b);
      }
      syncFns.push(() => {
        for (const item of lBtns) {
          item.btn.classList.toggle('is-active', focus.kvCompareLayerIdx === item.idx);
        }
      });
      kvLayerRow.appendChild(lChips);
      sec.appendChild(kvLayerRow);

      addLoopPassSelector('Inspected Loop Pass');
    }

    // View 6: Contract A Paper §6.2 Self-Speculative Execution Inspector
    if (viewMode === 'self_spec') {
      const offsetRow = el('div', 'dory-control-row');
      const offsetLbl = el('div', 'dory-control-label');
      const offsetVal = el('span', 'dory-control-value');
      offsetLbl.append(el('span', undefined, 'Verify Position Offset (0..4)'), offsetVal);
      const oChips = el('div', 'dory-chip-group');
      const oBtns: { readonly off: number; readonly btn: HTMLButtonElement }[] = [];
      for (let j = 0; j < 5; j += 1) {
        const b = el('button', 'dory-chip-btn dory-chip-btn--oxide', j < 4 ? `d_${j + 1}` : 'Bonus');
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, selectedVerifyOffset: j };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        oBtns.push({ off: j, btn: b });
        oChips.appendChild(b);
      }
      syncFns.push(() => {
        offsetVal.textContent = `offset ${focus.selectedVerifyOffset}`;
        for (const item of oBtns) {
          item.btn.classList.toggle('is-active', focus.selectedVerifyOffset === item.off);
        }
      });
      offsetRow.append(offsetLbl, oChips);
      sec.appendChild(offsetRow);

      const attRow = el('div', 'dory-control-row');
      attRow.appendChild(el('div', 'dory-control-label', 'Verify Attention Plate Layer'));
      const aChips = el('div', 'dory-chip-group');
      const aBtns: { readonly idx: number; readonly btn: HTMLButtonElement }[] = [];
      for (const p of [
        { idx: 16, label: 'P16 Loop-1 SWA' },
        { idx: 22, label: 'P22 Loop-1 Full' },
        { idx: 56, label: 'P56 Deep Coda SWA' },
      ]) {
        const b = el('button', 'dory-chip-btn', p.label);
        b.type = 'button';
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, selfSpecAttentionLayerIdx: p.idx };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        aBtns.push({ idx: p.idx, btn: b });
        aChips.appendChild(b);
      }
      syncFns.push(() => {
        for (const item of aBtns) {
          item.btn.classList.toggle('is-active', focus.selfSpecAttentionLayerIdx === item.idx);
        }
      });
      attRow.appendChild(aChips);
      sec.appendChild(attRow);

      if (!tourLocked && currentScene.executionMode !== 'self_spec') {
        const activateSpecBtn = el(
          'button',
          'dory-action-btn dory-action-btn--spec',
          'Fork Live Contract A Session (k=1→4 · 104 Banks)',
        );
        activateSpecBtn.type = 'button';
        activateSpecBtn.dataset.testid = 'dory-activate-selfspec-btn';
        activateSpecBtn.addEventListener('click', () => {
          selectExecutionModeInternal('self_spec');
        });
        sec.appendChild(activateSpecBtn);
      }
    }

    // View 7: Untied Bookends, 4-Format Readouts & READ-ONLY TBPTL
    if (viewMode === 'bookends') {
      sec.classList.add('dory-inspector-section--readonly');
      const fmtRow = el('div', 'dory-control-row');
      fmtRow.appendChild(
        el(
          'div',
          'dory-control-label',
          'Inspect Read-Only Checkpoint Row (Zero Quantization Toggles)',
        ),
      );
      const fChips = el('div', 'dory-chip-group');
      const fBtns: { readonly id: DoryCheckpointFormatId; readonly btn: HTMLButtonElement }[] = [];
      for (const fmt of CHECKPOINT_FORMAT_CHOICES) {
        const b = el('button', 'dory-chip-btn', fmt.label);
        b.type = 'button';
        b.dataset.testid = `dory-bookends-ckpt-${fmt.id}`;
        b.disabled = tourLocked;
        b.addEventListener('click', () => {
          if (tour.isActive()) return;
          focus = { ...focus, selectedCheckpointFormat: fmt.id };
          rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        });
        fBtns.push({ id: fmt.id, btn: b });
        fChips.appendChild(b);
      }
      syncFns.push(() => {
        for (const item of fBtns) {
          item.btn.classList.toggle('is-active', focus.selectedCheckpointFormat === item.id);
        }
      });
      fmtRow.appendChild(fChips);
      sec.appendChild(fmtRow);

      // Vocab Token ID Slider (0..63) — inspects untied embeddings vs lm_head & logit_scale
      const vocRow = el('div', 'dory-control-row');
      const vocLbl = el('div', 'dory-control-label');
      const vocVal = el('span', 'dory-control-value');
      vocLbl.append(el('span', undefined, 'Inspect Vocab Token ID (0..63)'), vocVal);
      const vocSlider = el('input', 'dory-range-input');
      vocSlider.type = 'range';
      vocSlider.min = '0';
      vocSlider.max = String(DORY_MINI_CONFIG.vocabSize - 1);
      vocSlider.disabled = tourLocked;
      vocSlider.dataset.testid = 'dory-vocab-slider';
      vocSlider.setAttribute('aria-label', 'Inspect untied vocabulary token ID 0 to 63');
      const syncVocab = (): void => {
        const tid =
          focus.selectedVocabTokenId ?? currentScene.bookendsMetadata?.selectedVocabTokenId ?? 0;
        vocVal.textContent = `#${tid} "${decodeDoryToken(tid)}"`;
        syncSliderValue(vocSlider, String(tid));
      };
      vocSlider.addEventListener('input', () => {
        if (tour.isActive()) return;
        const val = Number.parseInt(vocSlider.value, 10);
        if (!Number.isFinite(val)) return;
        focus = { ...focus, selectedVocabTokenId: Math.max(0, Math.min(63, val)) };
        syncVocab();
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
      });
      syncFns.push(syncVocab);
      vocRow.append(vocLbl, vocSlider);
      sec.appendChild(vocRow);
    }

    container.appendChild(sec);
    const syncAll = (): void => {
      for (let i = 0; i < syncFns.length; i += 1) {
        syncFns[i]!();
      }
    };
    syncAll();
    return syncAll;
  }

  function renderCellInspectSection(container: HTMLElement): void {
    const sec = el('section', 'dory-inspector-section');
    sec.appendChild(el('h3', 'dory-section-heading', 'Ray-Pick Cell & Provenance Inspector'));
    const cell = hoveredCell ?? selectedCell;
    if (!cell) {
      sec.appendChild(
        el(
          'p',
          'dory-note-text',
          'Hover or click any 3D sublayer block, NLERP sheet, SWA/Full attention plate, KV bank rail, or bookend matrix to inspect its physical/paired layer, loop pass, KV head, bank slot, and provenance.',
        ),
      );
      container.appendChild(sec);
      return;
    }

    const grid = el('div', 'dory-kv-grid');
    if (cell.partId) addKvRow(grid, 'Part', cell.partId);
    if (cell.tensorKey) addKvRow(grid, 'Tensor', cell.tensorKey);
    if (cell.physicalLayerIdx !== null) {
      const pairBadge = cell.pairedLayerIdx !== null ? ` (Pair L${cell.pairedLayerIdx})` : '';
      addKvRow(grid, 'Physical Layer', `p=${cell.physicalLayerIdx}${pairBadge}`);
    }
    if (cell.executedLayerIdx !== null) {
      addKvRow(grid, 'Executed Layer', `e=${cell.executedLayerIdx}`);
    }
    if (cell.stage) addKvRow(grid, 'Stage', cell.stage);
    if (cell.sublayerKind) addKvRow(grid, 'Sublayer Kind', cell.sublayerKind);
    if (cell.loopIteration !== null) {
      addKvRow(
        grid,
        'Loop Iteration',
        cell.loopIteration === 0 ? 't=0 (Prelude/Coda)' : `t=${cell.loopIteration}`,
      );
    }
    if (cell.qHeadIndex !== null) addKvRow(grid, 'Q Head', `Q${cell.qHeadIndex}`);
    if (cell.kvHeadIndex !== null) addKvRow(grid, 'KV Head', `KV${cell.kvHeadIndex}`);
    if (cell.rotaryPlane !== null) addKvRow(grid, 'Dual-RoPE Plane', `Plane ${cell.rotaryPlane}/3`);
    if (cell.kvBankSlotId !== null) {
      const kind = cell.kvBankKind ? ` (${cell.kvBankKind})` : '';
      addKvRow(grid, 'KV Bank Slot', `#${cell.kvBankSlotId}${kind}`);
    }
    if (cell.kvPolicy) addKvRow(grid, 'KV Policy', cell.kvPolicy);
    if (cell.cacheRole) addKvRow(grid, 'Cache Role', cell.cacheRole);
    if (cell.specOffset !== null) addKvRow(grid, 'Spec Offset', `verify[${cell.specOffset}]`);
    if (cell.row !== null && cell.col !== null) {
      addKvRow(grid, 'Cell [row, col]', `[${cell.row}, ${cell.col}]`);
    } else if (cell.tokenIndex !== null) {
      addKvRow(grid, 'Token', `#${cell.tokenIndex} "${decodeDoryToken(cell.tokenIndex)}"`);
    }
    if (cell.absTokenPos !== null) addKvRow(grid, 'Abs Pos', String(cell.absTokenPos));
    if (cell.modelWeightId) addKvRow(grid, 'Weight SHA-256', cell.modelWeightId.slice(0, 12));
    sec.appendChild(grid);
    container.appendChild(sec);
  }

  function renderViewDiagnosticsSection(container: HTMLElement): void {
    // View 1: Macro Shared-Weight & KV Rail Diagnostics
    if (viewMode === 'macro' && currentScene.macroMetadata) {
      const m = currentScene.macroMetadata;
      const sec = el('section', 'dory-inspector-section');
      sec.appendChild(el('h3', 'dory-section-heading', 'Physical 72L vs. Executed Depth Telemetry'));
      const grid = el('div', 'dory-kv-grid');
      addKvRow(grid, 'Display Mode', m.displayMode);
      addKvRow(grid, 'Physical / Executed', `72L phys -> ${m.executedSublayerCount}L exec (${m.effectivePairedDepth} paired)`);
      addKvRow(grid, 'Shared Weight ID', m.modelWeightId.slice(0, 14));
      addKvRow(grid, 'KV Bank Rail', `${m.kvRail.totalBanks} total (${m.kvRail.totalSwaBanks} SWA + ${m.kvRail.totalFullBanks} Full)`);
      addKvRow(
        grid,
        'Recurrent Passes',
        `${m.sharedRecurrentWeightBindings.length} passes bound to phys 16..55`,
      );
      sec.appendChild(grid);
      container.appendChild(sec);
    }

    // View 2: Anti-SLERP NLERP, 8D Trajectory Metrics & Read-Only Counterfactual Coda
    if (viewMode === 'hypersphere' && currentScene.hypersphereMetadata) {
      const h = currentScene.hypersphereMetadata;
      const sec = el('section', 'dory-inspector-section');
      sec.dataset.testid = 'dory-hypersphere-diagnostics';
      sec.appendChild(el('h3', 'dory-section-heading', DORY_ANTI_SLERP_PLAQUE));
      const grid = el('div', 'dory-kv-grid');
      addKvRow(grid, 'Sublayer', `p=${h.selectedPhysicalLayerIdx} (Pair L${h.selectedPairedLayerIdx}) · t=${h.selectedLoopIteration}`);
      addKvRow(grid, 'Token', `pos=${h.selectedAbsTokenPos} "${h.selectedTokenText}"`);
      addKvRow(
        grid,
        'Effective |α_hat|',
        `min=${safeFiniteFixed(h.spanningSubspace.minEffectiveAlpha, 3)} · mean=${safeFiniteFixed(h.spanningSubspace.meanEffectiveAlpha, 3)} · max=${safeFiniteFixed(h.spanningSubspace.maxEffectiveAlpha, 3)}`,
      );
      addKvRow(
        grid,
        'Extrapolated (> 1)',
        h.spanningSubspace.extrapolatedChannels.length > 0
          ? `ch [${h.spanningSubspace.extrapolatedChannels.join(', ')}] (unclamped DR7)`
          : 'none on this sublayer (see p=16 ch [6, 7])',
      );
      addKvRow(
        grid,
        'span(a, b) Angle',
        `${safeFiniteFixed(h.spanningSubspace.omegaDegrees, 2)} deg · out-of-plane=${safeFiniteFixed(h.spanningSubspace.outOfPlaneNorm, 4)}`,
      );
      sec.appendChild(grid);

      // 8D trajectory table
      if (h.loopMetrics8d.length > 0) {
        const table = el('table', 'dory-data-table');
        const thead = el('thead');
        const hr = el('tr');
        for (const col of ['Loop', '||s_t||', 'cos(s,prev)', '||Δs||', 'cos(s,x)', 'θ(deg)', 'J_N']) {
          hr.appendChild(el('th', undefined, col));
        }
        thead.appendChild(hr);
        const tbody = el('tbody');
        for (const row of h.loopMetrics8d) {
          const tr = el('tr');
          tr.append(
            el('td', undefined, `t=${row.loopIteration}`),
            el('td', undefined, safeFiniteFixed(row.sNorm, 4)),
            el('td', undefined, safeFiniteSigned(row.consecutiveCosine, 3)),
            el('td', undefined, safeFiniteFixed(row.consecutiveDisplacement, 3)),
            el('td', undefined, safeFiniteSigned(row.anchorCosine, 3)),
            el('td', undefined, safeFiniteFixed(row.entryAnchorJacobian.angleDegrees, 1)),
            el('td', undefined, safeFiniteFixed(row.entryAnchorJacobian.epsStabilizedJacobianNorm, 3)),
          );
          tbody.appendChild(tr);
        }
        table.append(thead, tbody);
        sec.appendChild(table);
      }

      sec.appendChild(el('p', 'dory-note-text', DORY_EXPLORATORY_PCA_BADGE));
      sec.appendChild(el('p', 'dory-note-text', DORY_ANCHOR_MAP_CAVEAT_BADGE));

      if (h.counterfactualCoda) {
        const cf = h.counterfactualCoda;
        const cfSec = el('section', 'dory-inspector-section dory-inspector-section--amber');
        cfSec.dataset.testid = 'dory-counterfactual-panel';
        cfSec.appendChild(el('h3', 'dory-section-heading', cf.badge));
        const cfGrid = el('div', 'dory-kv-grid');
        addKvRow(cfGrid, 'Probed Loop', `s_${cf.loopIteration} / k=${cf.kLoops} (0 KV mutation)`);
        addKvRow(cfGrid, 'KL(p_final || p_cf)', `${safeFiniteFixed(cf.klFromFinalLoopNats, 4)} nats`);
        addKvRow(
          cfGrid,
          'Top-1 Token',
          `"${decodeDoryToken(cf.argMaxTokenId)}" vs final "${decodeDoryToken(cf.finalLoopArgMaxTokenId)}"${cf.argMaxChanged ? ' (SHIFTED)' : ' (same)'}`,
        );
        const top3 = cf.topTokens
          .slice(0, 3)
          .map((t) => `"${t.tokenText}" (${safeFiniteFixed(t.prob * 100, 1)}%)`)
          .join(', ');
        addKvRow(cfGrid, 'Top-3 Readout', top3);
        cfSec.appendChild(cfGrid);
        container.append(sec, cfSec);
      } else {
        container.appendChild(sec);
      }
    }

    // View 3: Gated GQA & Dual-RoPE Diagnostics
    if (viewMode === 'gqa_rope' && currentScene.gqaRopeMetadata) {
      const g = currentScene.gqaRopeMetadata;
      const sec = el('section', 'dory-inspector-section');
      sec.dataset.testid = 'dory-gqa-diagnostics';
      sec.appendChild(
        el(
          'h3',
          'dory-section-heading',
          `${g.attentionKind === 'sliding_attention' ? `SWA (W=${g.slidingWindowMini} / prod ${g.slidingWindowProd})` : 'Full Causal Attention'} · Dual-RoPE`,
        ),
      );
      const grid = el('div', 'dory-kv-grid');
      addKvRow(grid, 'Layer / Loop', `p=${g.physicalLayerIdx} (Pair L${g.pairedLayerIdx}) · t=${g.loopIteration}`);
      addKvRow(grid, 'Dual-RoPE Base', `θ=${g.ropeThetaMini} mini / ${g.ropeThetaProd.toLocaleString('en-US')} prod`);
      addKvRow(grid, 'DR6 Norm Order', 'RoPE -> per-head Q/K L2Norm (Q-only sqk^2)');
      addKvRow(grid, 'sqk^2 Range', `[${safeFiniteFixed(g.sqkScaleRange[0], 2)}, ${safeFiniteFixed(g.sqkScaleRange[1], 2)}]`);
      addKvRow(grid, 's_gate^2 Range', `[${safeFiniteFixed(g.sGateScaleRange[0], 2)}, ${safeFiniteFixed(g.sGateScaleRange[1], 2)}]`);
      addKvRow(grid, 'attn_sub_norm', `||Y||_2=${safeFiniteFixed(g.attnSubNormL2Norm, 4)} over full ${g.attnSubNormAxisDim}-dim concat`);
      addKvRow(grid, 'Live Mask (DR20)', `visible=${g.visibleCellCount}, masked=${g.maskedCellCount}, writeKv=${String(g.writeKv)}`);
      sec.appendChild(grid);
      sec.appendChild(el('p', 'dory-note-text', g.qExpansionDisclosure));
      container.appendChild(sec);
    }

    // View 4: Normalized SwiGLU Diagnostics
    if (viewMode === 'swiglu' && currentScene.swigluMetadata) {
      const sw = currentScene.swigluMetadata;
      const sec = el('section', 'dory-inspector-section');
      sec.dataset.testid = 'dory-swiglu-diagnostics';
      sec.appendChild(el('h3', 'dory-section-heading', 'Normalized SwiGLU & Signed suv Scales'));
      const grid = el('div', 'dory-kv-grid');
      addKvRow(grid, 'Layer / Loop', `p=${sw.physicalLayerIdx} (Pair L${sw.pairedLayerIdx}) · t=${sw.loopIteration}`);
      addKvRow(grid, 'Family 4 Scale', `raw*sqrt(d) = x${safeFiniteFixed(sw.suvMultiplierMini, 3)} mini (no sigma_0, signed)`);
      addKvRow(grid, 'Signed Negative suv', `gate=${sw.negativeSuvGateCount}, up=${sw.negativeSuvUpCount} (no .abs())`);
      addKvRow(grid, 'ffn_sub_norm', `||M||_2=${safeFiniteFixed(sw.ffnSubNormL2Norm, 4)} over d_ff=${sw.ffnSubNormAxisDim}`);
      addKvRow(grid, 'Accum Reveal', `${sw.revealedIntermediateCount} / ${sw.ffnSubNormAxisDim} channels`);
      sec.appendChild(grid);
      sec.appendChild(el('p', 'dory-note-text', sw.mlpExpansionDisclosure));
      container.appendChild(sec);
    }

    // View 5: Dual-Axis KV & `last_loop` Divergence Diagnostics
    if (viewMode === 'kv_compare' && currentScene.kvCompareMetadata) {
      const kv = currentScene.kvCompareMetadata;
      const sec = el('section', 'dory-inspector-section dory-inspector-section--amber');
      sec.dataset.testid = 'dory-kv-compare-diagnostics';
      sec.appendChild(el('h3', 'dory-section-heading', 'Dual-Axis KV & DR17 Partition Probe'));
      const grid = el('div', 'dory-kv-grid');
      addKvRow(grid, 'Bank Count', `per_loop=${kv.perLoopBankCount} vs. last_loop=${kv.lastLoopBankCount}`);
      addKvRow(grid, 'S-Invariant Ratio', `(4+5k)/9 = ${safeFiniteFixed(kv.sInvariantRatio, 4)}x (strictly S-invariant)`);
      addKvRow(grid, 'Thm 1 Single-Shot', `max_abs = ${safeFiniteExp(kv.theorem1SingleShotMaxAbsDiff)}`);
      addKvRow(grid, 'Thm 3 Chunked vs PL', `max_abs = ${safeFiniteExp(kv.theorem3ChunkedVsPerLoopMaxAbsDiff)}`);
      addKvRow(grid, 'Thm 3 Chunked vs 1x1', `max_abs = ${safeFiniteExp(kv.theorem3ChunkedVsOneByOneMaxAbsDiff)}`);
      addKvRow(grid, 'Active Partition', `${kv.partitionMode} [${kv.chunkPartition.join(', ')}]`);
      addKvRow(grid, 'SWA Knee Drop', `${kv.swaKneeSlopeDropFactor}x marginal slope drop past S=${kv.swaWindowProd}`);
      sec.appendChild(grid);
      container.appendChild(sec);
    }

    // View 6: Contract A Paper §6.2 Self-Speculation Diagnostics
    if (viewMode === 'self_spec' && currentScene.selfSpecMetadata) {
      const sp = currentScene.selfSpecMetadata;
      const c = sp.cycleSummary;
      const sec = el('section', 'dory-inspector-section dory-inspector-section--spec');
      sec.dataset.testid = 'dory-selfspec-diagnostics';
      sec.appendChild(el('h3', 'dory-section-heading', sp.contractBadge));
      const grid = el('div', 'dory-kv-grid');
      addKvRow(grid, 'Contract A Banks', `${sp.totalBankCount} (${sp.inputBankCount} In + ${sp.loop1SharedBankCount} L1 + ${sp.loops2To4VerifierBankCount} L2..4 + ${sp.outputDeepBankCount} Deep + ${sp.outputShallowBankCount} Shallow)`);
      addKvRow(grid, 'Shallow Coda Delta', `+${formatKiB(sp.shallowCodaExtraBytes)} (total ${formatKiB(sp.totalKvPayloadBytes)})`);
      addKvRow(grid, 'Cycle Outcome', `${c.outcome} · accepted ${c.acceptedDraftCount}/${c.draftCount}`);
      addKvRow(grid, 'Prelude+L1 Reuse', `${c.reusedActivationPositions}/${c.verifiedPositions} positions reused`);
      addKvRow(grid, 'Shallow Catch-Up', sp.shallowCodaCatchupExecuted ? '1-token write (full accept)' : '0 extra work (rejection)');
      addKvRow(grid, 'cap_swa=16 Rollback', `${sp.rolledBackPositions} speculative positions rewound`);
      addKvRow(grid, 'Binary64 Rule', sp.divisionFreeBinary64Rule);
      if (sp.isSynthesizedPreview) {
        addKvRow(grid, 'Source', 'Isolated read-only preview cycle (standard session untouched)');
      }
      sec.appendChild(grid);
      container.appendChild(sec);
    }

    // View 7: Untied Bookends & READ-ONLY TBPTL Diagnostics
    if (viewMode === 'bookends' && currentScene.bookendsMetadata) {
      const bk = currentScene.bookendsMetadata;
      const sec = el('section', 'dory-inspector-section dory-inspector-section--readonly');
      sec.dataset.testid = 'dory-bookends-diagnostics';
      sec.appendChild(el('h3', 'dory-section-heading', `${bk.tbptlBadgeLabel} · Bookends & TBPTL`));
      const grid = el('div', 'dory-kv-grid');
      addKvRow(grid, 'Gradients / Quant Toggles', `hasGradients=${String(bk.hasGradients)} · zeroFakeQuant=${String(bk.zeroFakeQuantToggles)}`);
      addKvRow(grid, 'Untied Matrices', `max|E - W_lm| = ${safeFiniteFixed(bk.embeddingVsLmHeadMaxAbsDiff, 4)} (> 0 verified)`);
      addKvRow(grid, 'Row L2 Norms', `||E[v]||=${safeFiniteFixed(bk.embeddingRowL2Norm, 4)}, ||W_lm[v]||=${safeFiniteFixed(bk.lmHeadRowL2Norm, 4)}`);
      addKvRow(grid, 'logit_scale Mult', `1 / 0.0177 = ${safeFiniteFixed(bk.logitScaleMultiplier, 3)} (FP32)`);
      addKvRow(grid, 'ArgMax Token', `#${bk.argMaxTokenId} "${bk.argMaxTokenText}"`);
      addKvRow(grid, 'TBPTL Depth', `depth-${bk.tbptlDepth} stop-grad · ${bk.tbptlEarlyExitLosses} early-exit losses`);
      sec.appendChild(grid);
      container.appendChild(sec);
    }
  }

  function renderReadoutsTab(container: HTMLElement): void {
    const r = currentReadouts;

    // 1. Governance Scope Note
    const scopeSec = el('section', 'dory-inspector-section dory-inspector-section--readonly');
    scopeSec.dataset.testid = 'dory-readouts-scope';
    scopeSec.appendChild(
      el('h3', 'dory-section-heading', `${DORY_TBPTL_READONLY_BADGE} · Epistemic Scope`),
    );
    scopeSec.appendChild(el('p', 'dory-note-text', r.governanceScopeNote));
    container.appendChild(scopeSec);

    // 2. 0 B Diff 4-Checkpoint Production Table
    const ckptSec = el('section', 'dory-inspector-section');
    ckptSec.dataset.testid = 'dory-readouts-checkpoints';
    ckptSec.appendChild(
      el('h3', 'dory-section-heading', '0 B Diff 4-Checkpoint Production Footprints (OpenMDW-1.1)'),
    );
    const ckptTable = el('table', 'dory-data-table');
    const cHead = el('thead');
    const cHr = el('tr');
    for (const h of ['Format', 'Tensors', 'Total Size', 'Diff', 'SHA']) {
      cHr.appendChild(el('th', undefined, h));
    }
    cHead.appendChild(cHr);
    const cBody = el('tbody');
    for (const row of r.checkpoints) {
      const tr = el('tr', row.id === r.selectedCheckpointFormat ? 'is-highlight' : undefined);
      tr.append(
        el('td', undefined, row.label),
        el('td', undefined, row.tensorCount !== null ? String(row.tensorCount) : 'GGUF'),
        el('td', undefined, row.totalFormatted.dualLabel),
        el('td', undefined, row.diffBytes !== null ? `${row.diffBytes} B` : 'container'),
        el('td', undefined, row.shortSha),
      );
      cBody.appendChild(tr);
    }
    ckptTable.append(cHead, cBody);
    ckptSec.appendChild(ckptTable);
    container.appendChild(ckptSec);

    // 3. Depth vs. Compute & KV Bank Scaling Table (k = 1..8)
    const depthSec = el('section', 'dory-inspector-section');
    depthSec.appendChild(el('h3', 'dory-section-heading', 'Elastic Depth & (4+5k)/9 Scaling'));
    const dTable = el('table', 'dory-data-table');
    const dHead = el('thead');
    const dHr = el('tr');
    for (const h of ['k', 'Exec L', 'Paired', 'Body Ratio', 'PL Banks', 'LL Banks']) {
      dHr.appendChild(el('th', undefined, h));
    }
    dHead.appendChild(dHr);
    const dBody = el('tbody');
    for (const row of r.depthComputeScaling) {
      const tr = el('tr', row.kLoops === r.activeKLoops ? 'is-highlight' : undefined);
      tr.append(
        el('td', undefined, `${row.kLoops}${row.isTrained ? '' : '*'}`),
        el('td', undefined, `${row.executedSublayers}L`),
        el('td', undefined, `${row.effectivePairedDepth}L`),
        el('td', undefined, `${safeFiniteFixed(row.bodyComputeRatioVsK1, 3)}x`),
        el('td', undefined, String(row.perLoopTotalBanks)),
        el('td', undefined, String(row.lastLoopTotalBanks)),
      );
      dBody.appendChild(tr);
    }
    dTable.append(dHead, dBody);
    depthSec.appendChild(dTable);
    container.appendChild(depthSec);

    // 4. Production 1 -> 256K BF16 Logical KV Scaling Table
    const kvSec = el('section', 'dory-inspector-section');
    kvSec.appendChild(
      el('h3', 'dory-section-heading', '1 -> 256K BF16 KV Scaling (4x Knee @ S=1,024)'),
    );
    const kTable = el('table', 'dory-data-table');
    const kHead = el('thead');
    const kHr = el('tr');
    for (const h of ['Seq S', 'last_loop', 'k=2 PL', 'k=4 PL', 'SWA %']) {
      kHr.appendChild(el('th', undefined, h));
    }
    kHead.appendChild(kHr);
    const kBody = el('tbody');
    for (const row of r.productionKvScaling) {
      const tr = el('tr', row.isSwaKnee ? 'is-highlight' : undefined);
      tr.append(
        el('td', undefined, row.isSwaKnee ? `${row.seqLen} (W)` : String(row.seqLen)),
        el('td', undefined, row.lastLoopFormatted.iecLabel),
        el('td', undefined, row.perLoopK2Formatted.iecLabel),
        el('td', undefined, row.perLoopK4Formatted.iecLabel),
        el('td', undefined, `${safeFiniteFixed(row.swaSharePct, 1)}%`),
      );
      kBody.appendChild(tr);
    }
    kTable.append(kHead, kBody);
    kvSec.appendChild(kTable);
    container.appendChild(kvSec);

    // 5. D10 Memory Bounds & Live Session Audit
    const memSec = el('section', 'dory-inspector-section');
    memSec.appendChild(
      el('h3', 'dory-section-heading', 'D10 Bounded <= 12.00 MiB Memory & Live Audit'),
    );
    const memGrid = el('div', 'dory-kv-grid');
    for (const row of r.memoryBounds) {
      const capStr = row.capFormatted ? ` / ${row.capFormatted}` : '';
      addKvRow(memGrid, row.label, `${formatBytesAdaptive(row.bytes)}${capStr}`);
    }
    memSec.appendChild(memGrid);
    container.appendChild(memSec);

    // 6. Paper Results (Strictly Separated from Synthetic Miniature)
    const paperSec = el('section', 'dory-inspector-section');
    paperSec.appendChild(el('h3', 'dory-section-heading', DORY_PAPER_RESULTS_BADGE));
    for (const item of r.paperResults) {
      const card = el('article', 'dory-paper-card');
      card.append(
        el('div', 'dory-paper-metric', item.metricLabel),
        el('div', 'dory-paper-val', item.reportedValue),
        el('div', 'dory-paper-caveat', item.caveat),
      );
      paperSec.appendChild(card);
    }
    container.appendChild(paperSec);
  }

  let ledgerListWrap: HTMLElement | null = null;
  let ledgerSearchInput: HTMLInputElement | null = null;
  let ledgerFilterButtons: {
    readonly id: 'all' | 'browser_guard' | DoryDiscrepancyFilterCategory;
    readonly btn: HTMLButtonElement;
  }[] = [];

  function renderLedgerCardsOnly(): void {
    if (!ledgerListWrap) return;
    ledgerListWrap.innerHTML = '';

    const activeSess = getActiveSession();
    const filtered = buildDoryArchReadouts({
      activeKLoops: currentScene.kLoops,
      activeKvCacheMode: currentScene.kvCacheMode,
      selectedCheckpointFormat: focus.selectedCheckpointFormat,
      liveSessionMemory: activeSess.measureMemory(),
      activeAtlasBytes: currentScene.valueAtlas?.data.byteLength ?? 0,
      discrepancyFilter: buildCurrentDiscrepancyFilter(),
    }).discrepancyLedger;

    const activeTourRefs = new Set<string>(
      tour.isActive() ? tour.getState().currentBeat.discrepancyRefs : [],
    );

    if (filtered.rows.length === 0) {
      ledgerListWrap.appendChild(
        el('p', 'dory-note-text', 'No DR1–DR24 entries match the current filter.'),
      );
      return;
    }

    for (let i = 0; i < filtered.rows.length; i += 1) {
      const entry = filtered.rows[i]!;
      const card = el(
        'article',
        `dory-ledger-card${activeTourRefs.has(entry.id) ? ' is-referenced' : ''}`,
      );
      card.dataset.testid = `dory-ledger-card-${entry.id}`;

      const head = el('div', 'dory-ledger-card-head');
      head.appendChild(el('span', 'dory-ledger-id', entry.id));
      head.appendChild(el('span', 'dory-ledger-tag', entry.primaryTag));
      if (entry.isBrowserGuard) {
        head.appendChild(el('span', 'dory-ledger-badge--guard', 'BROWSER GUARD'));
      }
      head.appendChild(el('span', 'dory-ledger-title', entry.title));

      card.append(
        head,
        el('div', 'dory-ledger-detail', entry.summary),
        el('div', 'dory-ledger-detail', `Resolution: ${entry.resolution}`),
        el('div', 'dory-ledger-prov', `Provenance: ${entry.provenance}`),
      );
      ledgerListWrap.appendChild(card);
    }
  }

  function renderLedgerTab(container: HTMLElement): void {
    const filterSec = el('section', 'dory-inspector-section');
    filterSec.appendChild(
      el('h3', 'dory-section-heading', 'Discrepancy & Browser-Guard Ledger (DR1–DR24)'),
    );

    const search = el('input', 'dory-ledger-search');
    search.type = 'search';
    search.placeholder = 'Filter by ID (DR7, DR17, DR18, DR23) or keyword…';
    search.value = ledgerQuery;
    search.dataset.testid = 'dory-ledger-search';
    search.setAttribute('aria-label', 'Filter DR1 to DR24 discrepancy ledger');
    search.addEventListener('input', () => {
      ledgerQuery = search.value;
      renderLedgerCardsOnly();
    });
    ledgerSearchInput = search;

    const filterRow = el('div', 'dory-chip-group');
    ledgerFilterButtons = [];
    for (const f of LEDGER_CATEGORY_FILTERS) {
      const btn = el('button', 'dory-chip-btn', f.label);
      btn.type = 'button';
      btn.classList.toggle('is-active', ledgerFilterMode === f.id);
      btn.addEventListener('click', () => {
        ledgerFilterMode = f.id;
        for (const item of ledgerFilterButtons) {
          item.btn.classList.toggle('is-active', item.id === ledgerFilterMode);
        }
        renderLedgerCardsOnly();
      });
      ledgerFilterButtons.push({ id: f.id, btn });
      filterRow.appendChild(btn);
    }

    filterSec.append(search, filterRow);
    container.appendChild(filterSec);

    ledgerListWrap = el('div', 'dory-ledger-list');
    renderLedgerCardsOnly();
    container.appendChild(ledgerListWrap);
  }

  let mountedInspectorTab: DoryInspectorTabId | null = null;
  let mountedControlsKey: string | null = null;
  let viewControlsSlot: HTMLElement | null = null;
  let cellInspectSlot: HTMLElement | null = null;
  let diagnosticsSlot: HTMLElement | null = null;
  let syncViewControlsInPlace: (() => void) | null = null;

  function renderCellInspectOnly(): void {
    if (inspectorTab !== 'inspect' || !cellInspectSlot) return;
    cellInspectSlot.innerHTML = '';
    renderCellInspectSection(cellInspectSlot);
  }

  function renderInspectorBody(opts?: { readonly preserveViewControlsDom?: boolean }): void {
    if (inspectorTab === 'inspect') {
      if (
        mountedInspectorTab !== 'inspect' ||
        !viewControlsSlot ||
        !cellInspectSlot ||
        !diagnosticsSlot
      ) {
        inspectorBody.innerHTML = '';
        viewControlsSlot = el('div', 'dory-inspector-slot');
        viewControlsSlot.dataset.testid = 'dory-inspector-controls-slot';
        cellInspectSlot = el('div', 'dory-inspector-slot');
        cellInspectSlot.dataset.testid = 'dory-inspector-cell-slot';
        diagnosticsSlot = el('div', 'dory-inspector-slot');
        diagnosticsSlot.dataset.testid = 'dory-inspector-diagnostics-slot';
        inspectorBody.append(viewControlsSlot, cellInspectSlot, diagnosticsSlot);
        mountedInspectorTab = 'inspect';
        mountedControlsKey = null;
        syncViewControlsInPlace = null;
        ledgerListWrap = null;
        ledgerSearchInput = null;
      }

      const activeK = currentScene.kLoops;
      const nextControlsKey = `${viewMode}|${tour.isActive() ? '1' : '0'}|${activeK}|${currentScene.executionMode}`;
      if (
        mountedControlsKey === null ||
        (!opts?.preserveViewControlsDom && mountedControlsKey !== nextControlsKey)
      ) {
        viewControlsSlot.innerHTML = '';
        syncViewControlsInPlace = renderViewControlsSection(viewControlsSlot, activeK);
        mountedControlsKey = nextControlsKey;
      } else if (syncViewControlsInPlace) {
        syncViewControlsInPlace();
      }

      cellInspectSlot.innerHTML = '';
      renderCellInspectSection(cellInspectSlot);

      diagnosticsSlot.innerHTML = '';
      renderViewDiagnosticsSection(diagnosticsSlot);
      return;
    }

    if (inspectorTab === 'ledger') {
      if (mountedInspectorTab === 'ledger' && ledgerSearchInput && ledgerListWrap) {
        if (
          typeof document === 'undefined' ||
          document.activeElement !== ledgerSearchInput
        ) {
          ledgerSearchInput.value = ledgerQuery;
        }
        for (const item of ledgerFilterButtons) {
          item.btn.classList.toggle('is-active', item.id === ledgerFilterMode);
        }
        renderLedgerCardsOnly();
        return;
      }
      mountedInspectorTab = 'ledger';
      mountedControlsKey = null;
      viewControlsSlot = null;
      cellInspectSlot = null;
      diagnosticsSlot = null;
      syncViewControlsInPlace = null;
      inspectorBody.innerHTML = '';
      renderLedgerTab(inspectorBody);
      return;
    }

    // Readouts tab
    mountedInspectorTab = 'readouts';
    mountedControlsKey = null;
    viewControlsSlot = null;
    cellInspectSlot = null;
    diagnosticsSlot = null;
    syncViewControlsInPlace = null;
    ledgerListWrap = null;
    ledgerSearchInput = null;
    inspectorBody.innerHTML = '';
    renderReadoutsTab(inspectorBody);
  }

  // --------------------------------------------------------------------------
  // POINTER ORBIT / PAN / ZOOM & DR21 RAY-PICKING
  // --------------------------------------------------------------------------

  let dragMode: 'none' | 'orbit' | 'pan' = 'none';
  let dragStartX = 0;
  let dragStartY = 0;
  let lastPtrX = 0;
  let lastPtrY = 0;
  let pointerMoved = false;

  function pickAtClientPoint(clientX: number, clientY: number): DoryPickIdentity | null {
    if (!cameraSnapshot || currentScene.pickBoxes.length === 0) return null;
    const ray = clientPointToRay(clientX, clientY, cameraSnapshot);
    const hit = pickMacroBay(ray, currentScene.pickBoxes, []);
    return pickHitToDoryCell(hit, currentScene.viewMode);
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
        renderCellInspectOnly();
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
      const patch = resolveDoryPickInteraction(viewMode, picked, currentScene.kLoops);
      if (patch) {
        focus = { ...focus, ...patch };
        rebuildSceneAndRefresh({ preserveViewControlsDom: true });
        return;
      }
    }

    if (inspectorTab === 'inspect') {
      renderCellInspectOnly();
    }
    scheduler.requestFrame();
  };

  const onPointerLeave = (): void => {
    if (dragMode !== 'none') return;
    if (hoveredCell !== null) {
      hoveredCell = null;
      if (inspectorTab === 'inspect') {
        renderCellInspectOnly();
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
    if (shortcutsModalOpen) {
      if (shortcutsFocusTrap.handleKeydown(ev)) {
        return;
      }
    }

    const action = resolveDoryGlobalKeyAction(ev.key, {
      target: ev.target,
      ctrlKey: ev.ctrlKey,
      metaKey: ev.metaKey,
      altKey: ev.altKey,
      tourActive: tour.isActive(),
      shortcutsModalOpen,
    });

    switch (action.type) {
      case 'none':
        return;
      case 'close_shortcuts':
        ev.preventDefault();
        closeShortcutsModal();
        return;
      case 'open_shortcuts':
        ev.preventDefault();
        openShortcutsModal();
        return;
      case 'select_view':
        ev.preventDefault();
        if (tour.isActive()) exitTourInternal(false);
        setViewInternal(action.viewMode, true);
        return;
      case 'toggle_tour':
        ev.preventDefault();
        if (tour.isActive()) exitTourInternal(true);
        else startTourInternal(0);
        return;
      case 'tour_next': {
        ev.preventDefault();
        const st = tour.getState();
        if (st.beatIndex < st.beatCount - 1) {
          goToTourBeatInternal(st.beatIndex + 1);
        }
        return;
      }
      case 'tour_prev': {
        ev.preventDefault();
        const st = tour.getState();
        if (st.beatIndex > 0) {
          goToTourBeatInternal(st.beatIndex - 1);
        }
        return;
      }
      case 'exit_tour':
        ev.preventDefault();
        exitTourInternal(true);
        return;
      case 'reset_camera':
        ev.preventDefault();
        flyToCurrentScenePreset(true);
        return;
      case 'toggle_inspector':
        ev.preventDefault();
        toggleInspectorCollapsed();
        return;
      case 'toggle_charts':
        ev.preventDefault();
        toggleChartsPanelCollapsed();
        return;
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
  // `window.__dorySpatialDebug` E2E & DIAGNOSTIC HANDLE (`types.ts §6`)
  // --------------------------------------------------------------------------

  function getDebugSnapshot(): DorySpatialDebugSnapshot {
    const activeSess = getActiveSession();
    const activeBranch = activeSess.getActiveBranch();
    const activeRec = activeSess.getActiveRecord();
    const audit = activeSess.measureMemory();
    const tourState = tour.getState();
    const kLoops = currentScene.kLoops;
    const kvMode = currentScene.kvCacheMode;
    const execMode = currentScene.executionMode;
    const banks = computeKvBankCounts(kLoops, kvMode, execMode === 'self_spec');

    return Object.freeze({
      modelId: 'dory' as const,
      viewMode: currentScene.viewMode,
      macroDisplayMode: focus.macroDisplayMode,
      kLoops,
      kvCacheMode: kvMode,
      executionMode: execMode,
      executedDepth: computeExecutedDepth(kLoops, params.config),
      kvBankCount: banks.totalBanks,
      rafActive: scheduler.isActive(),
      glErrorCount,
      atlasKey: resolveEffectiveAtlasKey(currentScene),
      atlasUploadCount: atlasTracker.getUploadCount(),
      atlasByteLength: currentScene.valueAtlas?.data.byteLength ?? 0,
      overlayCellNumNodes: overlay.getStats().renderedCellNumNodes,
      renderedChartKinds: chartsPanel.getRenderedChartKinds(),
      activeBranchId: activeBranch?.branchId ?? null,
      activeRecordId: activeRec?.recordId ?? null,
      branchCount: audit.branchCount,
      recordCount: audit.recordCount,
      committedLen: activeRec?.committedLen ?? activeBranch?.committedLen ?? 0,
      inspectorTab,
      inspectorCollapsed,
      tourActive: tourState.active,
      tourBeatIndex: tourState.beatIndex,
      tourBeatCount: tourState.beatCount,
      tourActIndex: tourState.actIndex,
      activeTourCoord: tourState.active ? tourState.coord : null,
      selectedCheckpointFormat: focus.selectedCheckpointFormat,
      readOnlyTbptlVerified: true as const,
      zeroFakeQuantToggles: true as const,
      memoryAudit: audit,
    });
  }

  const debugHandle: DorySpatialDebugHandle = {
    get modelId() {
      return 'dory' as const;
    },
    get viewMode() {
      return getDebugSnapshot().viewMode;
    },
    get macroDisplayMode() {
      return getDebugSnapshot().macroDisplayMode;
    },
    get kLoops() {
      return getDebugSnapshot().kLoops;
    },
    get kvCacheMode() {
      return getDebugSnapshot().kvCacheMode;
    },
    get executionMode() {
      return getDebugSnapshot().executionMode;
    },
    get executedDepth() {
      return getDebugSnapshot().executedDepth;
    },
    get kvBankCount() {
      return getDebugSnapshot().kvBankCount;
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
    get atlasByteLength() {
      return getDebugSnapshot().atlasByteLength;
    },
    get overlayCellNumNodes() {
      return getDebugSnapshot().overlayCellNumNodes;
    },
    get renderedChartKinds() {
      return getDebugSnapshot().renderedChartKinds;
    },
    get activeBranchId() {
      return getDebugSnapshot().activeBranchId;
    },
    get activeRecordId() {
      return getDebugSnapshot().activeRecordId;
    },
    get branchCount() {
      return getDebugSnapshot().branchCount;
    },
    get recordCount() {
      return getDebugSnapshot().recordCount;
    },
    get committedLen() {
      return getDebugSnapshot().committedLen;
    },
    get inspectorTab() {
      return getDebugSnapshot().inspectorTab;
    },
    get inspectorCollapsed() {
      return getDebugSnapshot().inspectorCollapsed;
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
    get tourActIndex() {
      return getDebugSnapshot().tourActIndex;
    },
    get activeTourCoord() {
      return getDebugSnapshot().activeTourCoord;
    },
    get selectedCheckpointFormat() {
      return getDebugSnapshot().selectedCheckpointFormat;
    },
    get readOnlyTbptlVerified() {
      return true as const;
    },
    get zeroFakeQuantToggles() {
      return true as const;
    },
    get memoryAudit() {
      return getDebugSnapshot().memoryAudit;
    },
    getSnapshot(): DorySpatialDebugSnapshot {
      return getDebugSnapshot();
    },
    setView(nextView: DorySpatialViewMode): DorySpatialDebugSnapshot {
      if (tour.isActive()) {
        exitTourInternal(false, true);
      }
      setViewInternal(nextView, false);
      return getDebugSnapshot();
    },
    updateFocus(patch: Partial<DorySpatialFocusState>): DorySpatialDebugSnapshot {
      updateFocusInternal(patch, true);
      return getDebugSnapshot();
    },
    setKLoops(kLoops: DorySupportedLoopCount): DorySpatialDebugSnapshot {
      if (tour.isActive()) {
        exitTourInternal(false, true);
      }
      selectKLoopsInternal(kLoops, true);
      return getDebugSnapshot();
    },
    setKvCacheMode(mode: DoryKvCacheMode): DorySpatialDebugSnapshot {
      if (tour.isActive()) {
        exitTourInternal(false, true);
      }
      selectKvCacheModeInternal(mode, true);
      return getDebugSnapshot();
    },
    setExecutionMode(mode: DoryExecutionMode): DorySpatialDebugSnapshot {
      if (tour.isActive()) {
        exitTourInternal(false, true);
      }
      selectExecutionModeInternal(mode, true);
      return getDebugSnapshot();
    },
    setKvComparePartitionMode(mode: DoryKvComparePartitionMode): DorySpatialDebugSnapshot {
      updateFocusInternal({ kvComparePartitionMode: mode }, true);
      return getDebugSnapshot();
    },
    setMacroDisplayMode(mode: DoryMacroDisplayMode): DorySpatialDebugSnapshot {
      if (!tour.isActive()) {
        focus = { ...focus, macroDisplayMode: mode };
        rebuildSceneAndRefresh({
          refitCamera: viewMode === 'macro',
          animateCamera: false,
          syncRender: true,
          preserveViewControlsDom: true,
        });
      }
      return getDebugSnapshot();
    },
    stepSession(): DorySpatialDebugSnapshot {
      executeNextStep(true);
      return getDebugSnapshot();
    },
    forkBranch(): DorySpatialDebugSnapshot {
      forkCurrentBranchInternal(true);
      return getDebugSnapshot();
    },
    switchBranch(branchId: string): DorySpatialDebugSnapshot {
      switchToBranchInternal(branchId, true);
      return getDebugSnapshot();
    },
    setInspectorTab(tab: DoryInspectorTabId): DorySpatialDebugSnapshot {
      setInspectorTabInternal(tab);
      return getDebugSnapshot();
    },
    setInspectorCollapsed(collapsed: boolean): DorySpatialDebugSnapshot {
      setInspectorCollapsedInternal(collapsed, true);
      return getDebugSnapshot();
    },
    startTour(beatIndex = 0): DorySpatialDebugSnapshot {
      startTourInternal(beatIndex, true);
      return getDebugSnapshot();
    },
    goToBeat(beatIndex: number): DorySpatialDebugSnapshot {
      goToTourBeatInternal(beatIndex, true);
      return getDebugSnapshot();
    },
    goToAct(actId: DoryTourActId): DorySpatialDebugSnapshot {
      goToTourActInternal(actId, true);
      return getDebugSnapshot();
    },
    nextBeat(): DorySpatialDebugSnapshot {
      const st = tour.getState();
      if (!st.active) {
        startTourInternal(0, true);
      } else if (st.beatIndex < st.beatCount - 1) {
        goToTourBeatInternal(st.beatIndex + 1, true);
      }
      return getDebugSnapshot();
    },
    prevBeat(): DorySpatialDebugSnapshot {
      const st = tour.getState();
      if (st.active && st.beatIndex > 0) {
        goToTourBeatInternal(st.beatIndex - 1, true);
      }
      return getDebugSnapshot();
    },
    exitTour(): DorySpatialDebugSnapshot {
      exitTourInternal(true, true);
      return getDebugSnapshot();
    },
    resetCamera(): DorySpatialDebugSnapshot {
      flyToCurrentScenePreset(false);
      return getDebugSnapshot();
    },
  };

  const shouldAttachDebug = options.attachDebugHandle ?? true;
  if (shouldAttachDebug && typeof window !== 'undefined') {
    window.__dorySpatialDebug = debugHandle;
  }

  // Initial synchronous layout, upload, draw & idle settle (`rafActive === false` at rest!)
  syncTopViewTabs();
  syncInspectorTabs();
  refreshStageHud();
  refreshReadoutsAndCharts();
  refreshDock();
  renderInspectorBody();
  cameraPose = computePresetPose(currentScene);
  renderFrameNow();

  // --------------------------------------------------------------------------
  // CONTROLLER STATE SNAPSHOT & CLEAN DISPOSAL
  // --------------------------------------------------------------------------

  function getState(): DoryAppControllerState {
    const activeSess = getActiveSession();
    const activeBranch = activeSess.getActiveBranch();
    const activeRec = activeSess.getActiveRecord();
    const tourState = tour.getState();
    return Object.freeze({
      viewMode,
      promptPresetId,
      kLoops: toSupportedLoopCount(currentScene.kLoops),
      allowExtrapolationDepths,
      kvCacheMode: currentScene.kvCacheMode,
      executionMode: currentScene.executionMode,
      sampling,
      inspectorTab,
      inspectorCollapsed,
      chartsPanelCollapsed,
      shortcutsModalOpen,
      discrepancyFilter: buildCurrentDiscrepancyFilter(),
      focus,
      activeBranchId: activeBranch?.branchId ?? null,
      activeRecordId: activeRec?.recordId ?? null,
      tourActive: tourState.active,
      tourBeatIndex: tourState.beatIndex,
      isPlaying,
      reducedMotion,
      dr18ForkNotice,
    });
  }

  function dispose(): void {
    if (disposed) return;
    disposed = true;
    scheduler.dispose();
    shortcutsFocusTrap.deactivate();
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    if (typeof window !== 'undefined') {
      window.removeEventListener('keydown', onWindowKeydown);
      if (window.__dorySpatialDebug === debugHandle) {
        delete window.__dorySpatialDebug;
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
    chartsPanel.dispose();
    timelineDock.dispose();
    modelSwitcher.destroy();

    if (glBundle) {
      glBundle.solids.dispose();
      glBundle.lines.dispose();
      glBundle.ribbons.dispose();
      glBundle.flatFlow.dispose();
      glBundle.sheets.dispose();
      glBundle.device.dispose();
      glBundle = null;
    }

    if (shell.parentNode === host) {
      host.removeChild(shell);
    }
  }

  return {
    root: shell,
    debug: debugHandle,
    getState,
    getCurrentScene: () => currentScene,
    getTourState: () => tour.getState(),
    getCameraPose: () => (cameraPose ? cloneCameraPose(cameraPose) : null),
    dispose,
  };
}

/** Alias matching `mountDiffusionSpatialApp` / `mountDorySpatialApp` ergonomics. */
export const mountDorySpatialApp = createDorySpatialApp;
