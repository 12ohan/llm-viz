/**
 * Shared UNI6 Interactive Browser App, RGB Canvas, Overlay & Timeline Dock Contracts
 * (`src/uni/app/types.ts`)
 *
 * Pinned Step-0 contract shared across:
 *   - Slice 6B (`pixelCanvas.ts`, `a11yHelpers.ts`, `spatialOverlay.ts`, `timelineDock.ts`)
 *   - Slice 6C (`spatial.css`, `spatialApp.ts`, `main.ts`, `tests/uni-app-contracts.test.ts`)
 *
 * Strictly zero imports from `src/ui/controls.ts` (D4), `src/twotower/`, or `src/diffusion/`.
 */

import type { CameraSnapshot } from '../../render/camera';
import type {
  UniPickIdentity,
  UniRgbPreviewDescriptor,
  UniSpatialFocusState,
  UniSpatialScene,
  UniSpatialViewMode,
} from '../scene/types';
import type {
  UniEditGuidanceMode,
  UniElasticPrefillMode,
  UniExecutionViewId,
  UniGainPresetId,
  UniScenePresetId,
  UniSessionMemoryAudit,
  UniSessionTaskMode,
} from '../types';

export {
  DEFAULT_UNI_SPATIAL_FOCUS_STATE,
  type UniSpatialFocusState,
} from '../scene/types';

export type UniInspectorTabId = 'inspect' | 'architecture' | 'ledger';

/** High-level 4-task UI selector mapped onto `UniSessionTaskMode`. */
export type UniDockTaskChoice =
  | 'understanding'
  | 'grounding'
  | 't2i'
  | 'image_edit';

export interface UniOverlayRect {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

export interface UniPixelCanvasRgbaBuffer {
  readonly width: number;
  readonly height: number;
  /** Row-major `[height * width * 4]` RGBA Uint8ClampedArray (`0..255`). */
  readonly rgba: Uint8ClampedArray;
  /** Number of pixels overlaid with `U18` border-lock hatching. */
  readonly hatchedPixelCount: number;
  /** Number of pixels drawn by `<LOC_0..16>` grounding box borders. */
  readonly boxBorderPixelCount: number;
}

export interface UniPixelCanvasMountOptions {
  /** Integer display scale factor for crisp pixelated canvas upscaling (default `8`). */
  readonly scale?: number;
  /** Whether to draw hatched `U18` `paddingLockMask` borders when present (default `true`). */
  readonly showPaddingLock?: boolean;
  /** Whether to draw `<LOC_0..16>` `groundingBoxes` overlays when present (default `true`). */
  readonly showGroundingBoxes?: boolean;
}

export interface UniSpatialOverlayOptions {
  /** Hard node cap enforced strictly per §8.3C / §10.2 (defaults to `64`, clamped `<= 64`). */
  readonly maxCellNumNodes?: number;
}

export interface UniSpatialOverlayUpdateInput {
  readonly scene: UniSpatialScene;
  readonly snapshot: CameraSnapshot;
  readonly hoveredCell?: UniPickIdentity | null;
  readonly selectedPartId?: string | null;
  readonly topInsetPx?: number;
  readonly bottomInsetPx?: number;
  readonly avoidRects?: readonly UniOverlayRect[];
}

export interface UniTimelineBranchSummary {
  readonly branchId: string;
  readonly label: string;
  readonly taskMode: UniSessionTaskMode;
  readonly prefillMode: UniElasticPrefillMode;
  readonly recordCount: number;
  readonly isActive: boolean;
}

export interface UniTimelineDockState {
  readonly taskChoice: UniDockTaskChoice;
  readonly taskMode: UniSessionTaskMode;
  readonly executionViewId: UniExecutionViewId;
  readonly prefillMode: UniElasticPrefillMode;
  readonly gainPresetId: UniGainPresetId;
  readonly scenePresetId: UniScenePresetId;
  readonly editGuidanceMode: UniEditGuidanceMode;
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
  readonly currentStepIndex: number;
  readonly totalRecordsInBranch: number;
  readonly canStepBack: boolean;
  readonly canStepForward: boolean;
  readonly canExecuteStep: boolean;
  readonly isPlaying: boolean;
  readonly tourActive: boolean;
  readonly branches: readonly UniTimelineBranchSummary[];
  readonly memoryAudit: UniSessionMemoryAudit;
  readonly layerSummary: string;
}

export interface UniTimelineDockCallbacks {
  readonly onStepBack: () => void;
  readonly onStepForward: () => void;
  readonly onScrubIndex: (indexInBranch: number) => void;
  readonly onTogglePlay: () => void;
  readonly onExecuteNextStep: () => void;
  readonly onSelectTaskChoice: (choice: UniDockTaskChoice) => void;
  readonly onSelectPrefillMode: (mode: UniElasticPrefillMode) => void;
  readonly onForkBranch: () => void;
  readonly onSwitchBranch: (branchId: string) => void;
}

export interface UniAppControllerState {
  readonly viewMode: UniSpatialViewMode;
  readonly taskMode: UniSessionTaskMode;
  readonly executionViewId: UniExecutionViewId;
  readonly prefillMode: UniElasticPrefillMode;
  readonly inspectorTab: UniInspectorTabId;
  readonly inspectorCollapsed: boolean;
  readonly focus: UniSpatialFocusState;
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
  readonly tourActive: boolean;
  readonly tourBeatIndex: number;
  readonly isPlaying: boolean;
  readonly reducedMotion: boolean;
}

/** Clean JSON-serializable snapshot returned by `window.__uniSpatialDebug`. */
export interface UniSpatialDebugSnapshot {
  readonly viewMode: UniSpatialViewMode;
  readonly executionViewId: UniExecutionViewId;
  readonly taskMode: UniSessionTaskMode;
  readonly prefillMode: UniElasticPrefillMode;
  readonly rafActive: boolean;
  readonly glErrorCount: number;
  readonly atlasKey: string | null;
  readonly atlasUploadCount: number;
  readonly rgbPreviewCount: number;
  readonly overlayCellNumNodes: number;
  readonly activeRecordId: string | null;
  readonly branchCount: number;
  readonly tourActive: boolean;
  readonly tourBeatIndex: number;
  readonly tourBeatCount: number;
  readonly memoryAudit: UniSessionMemoryAudit;
}

/** E2E & diagnostic debug handle attached to `window.__uniSpatialDebug` in UNI6/UNI7. */
export interface UniSpatialDebugHandle extends UniSpatialDebugSnapshot {
  getSnapshot(): UniSpatialDebugSnapshot;
  setView(viewMode: UniSpatialViewMode): UniSpatialDebugSnapshot;
  setTaskMode(taskMode: UniSessionTaskMode): UniSpatialDebugSnapshot;
  setPrefillMode(mode: UniElasticPrefillMode): UniSpatialDebugSnapshot;
  startTour(beatIndex?: number): UniSpatialDebugSnapshot;
  goToBeat(beatIndex: number): UniSpatialDebugSnapshot;
  exitTour(): UniSpatialDebugSnapshot;
}

export interface UniRgbPreviewRendererApi {
  renderDescriptors(
    descriptors: readonly UniRgbPreviewDescriptor[],
    options?: UniPixelCanvasMountOptions,
  ): void;
  clear(): void;
  dispose(): void;
}
