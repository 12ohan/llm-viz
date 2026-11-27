/**
 * Unified Spatial Scene Barrel & 7-View Dispatcher for NVIDIA Nemotron-Labs-Dory1.0-4B
 * (`src/dory/scene/index.ts`).
 *
 * Implements `DorySceneIndexModuleContract` (`DORY5` Slice 5A) dispatching cleanly across all
 * 7 locked `D7` Editorial WebGL2 spatial views:
 *   1. `'macro'`       -> `buildDoryMacroScene`       (`src/dory/scene/macroScene.ts`)
 *   2. `'hypersphere'` -> `buildDoryHypersphereScene` (`src/dory/scene/hypersphereScene.ts`)
 *   3. `'gqa_rope'`    -> `buildDoryGqaRopeScene`     (`src/dory/scene/gqaRopeScene.ts`)
 *   4. `'swiglu'`      -> `buildDorySwigluScene`      (`src/dory/scene/swigluScene.ts`)
 *   5. `'kv_compare'`  -> `buildDoryKvCompareScene`   (`src/dory/scene/kvCompareScene.ts`)
 *   6. `'self_spec'`   -> `buildDorySelfSpecScene`    (`src/dory/scene/selfSpecScene.ts`)
 *   7. `'bookends'`    -> `buildDoryBookendsScene`    (`src/dory/scene/bookendsScene.ts`)
 *
 * Strictly `D4`-compliant headless barrel: ZERO DOM, WebGL calls, Node built-ins,
 * ambient RNG (`Math.random`), Gate 4b/4c Nano modules, or sibling-app imports.
 */

import type { DorySession } from '../types';
import { buildDoryBookendsScene } from './bookendsScene';
import { buildDoryGqaRopeScene } from './gqaRopeScene';
import { buildDoryHypersphereScene } from './hypersphereScene';
import { buildDoryKvCompareScene } from './kvCompareScene';
import { buildDoryMacroScene } from './macroScene';
import { ensureDorySessionContext } from './provenance';
import { buildDorySelfSpecScene } from './selfSpecScene';
import { buildDorySwigluScene } from './swigluScene';
import {
  DEFAULT_DORY_SPATIAL_FOCUS_STATE,
  DORY_SPATIAL_VIEW_MODES,
  type DoryBookendsSceneOptions,
  type DoryGqaRopeSceneOptions,
  type DoryHypersphereSceneOptions,
  type DoryKvCompareSceneOptions,
  type DoryMacroSceneOptions,
  type DoryResolvedSessionContext,
  type DorySceneBuilderRegistry,
  type DorySceneIndexModuleContract,
  type DorySceneOptionsByView,
  type DorySelfSpecSceneOptions,
  type DorySpatialFocusState,
  type DorySpatialScene,
  type DorySpatialViewDescriptor,
  type DorySpatialViewMode,
  type DorySwigluSceneOptions,
} from './types';

// ============================================================================
// UNIFIED BARREL RE-EXPORTS (TYPES, PROVENANCE & ALL 7 SCENE BUILDERS)
// ============================================================================

export * from './types';
export {
  DORY_FLOW_COLORS,
  DoryAtlasPacker,
  DoryDr20CausalMaskError,
  DoryFlatFlowBufferBuilder,
  DoryLineBufferBuilder,
  DoryRibbonBufferBuilder,
  DorySheetBufferBuilder,
  DorySolidBufferBuilder,
  LINE_STYLE_ACTIVE,
  LINE_STYLE_BF16_GOLD,
  LINE_STYLE_GUIDE,
  LINE_STYLE_NEUTRAL,
  auditAtlasRegions,
  computeBoundsFromPoints,
  computeDoryAtlasContentKey,
  encodeDoryPickMetadata,
  ensureDorySessionContext,
  mergeAabbs,
  packPickBoxFromAabb,
  packPickBoxFromFrame,
  pickHitToDoryCell,
  resolveDorySessionContext,
  validateSceneGeometryFinite,
  type DoryAtlasPackMatrixOptions,
  type DorySheetPushParams,
} from './provenance';

export * from './macroScene';
export * from './hypersphereScene';
export * from './gqaRopeScene';
export * from './swigluScene';
export * from './kvCompareScene';
export * from './selfSpecScene';
export * from './bookendsScene';

// ============================================================================
// EXHAUSTIVE 7-VIEW BUILDER REGISTRY & VIEW DESCRIPTOR LOOKUP
// ============================================================================

/** Exhaustive 7-view scene builder registry matching `DORY_SPATIAL_VIEW_MODES`. */
export const DORY_SCENE_BUILDERS: DorySceneBuilderRegistry = Object.freeze({
  macro: buildDoryMacroScene,
  hypersphere: buildDoryHypersphereScene,
  gqa_rope: buildDoryGqaRopeScene,
  swiglu: buildDorySwigluScene,
  kv_compare: buildDoryKvCompareScene,
  self_spec: buildDorySelfSpecScene,
  bookends: buildDoryBookendsScene,
});

const VIEW_DESCRIPTOR_BY_ID: ReadonlyMap<DorySpatialViewMode, DorySpatialViewDescriptor> =
  new Map(DORY_SPATIAL_VIEW_MODES.map((desc) => [desc.id, desc]));

/**
 * Returns the canonical frozen `DorySpatialViewDescriptor` for any of the 7 `D7` views,
 * throwing `RangeError` for unrecognized view IDs.
 */
export function getDorySpatialViewDescriptor(
  viewMode: DorySpatialViewMode,
): DorySpatialViewDescriptor {
  const desc = VIEW_DESCRIPTOR_BY_ID.get(viewMode);
  if (desc === undefined) {
    throw new RangeError(`Unknown DorySpatialViewMode: '${String(viewMode)}'.`);
  }
  return desc;
}

// ============================================================================
// UNIFIED FOCUS RESOLUTION & PER-VIEW OPTION PROJECTION
// ============================================================================

function clampUnit(val: number, fallback: number): number {
  if (!Number.isFinite(val)) return fallback;
  return Math.max(0, Math.min(1, val));
}

/**
 * Normalizes a partial `DorySpatialFocusState` over `DEFAULT_DORY_SPATIAL_FOCUS_STATE`,
 * preserving explicit `null` values on nullable focus cursors while safely ignoring `undefined`
 * keys and clamping continuous ratios (`swigluAccumReveal in [0, 1]`).
 */
export function resolveDorySpatialFocus(
  focus?: Partial<DorySpatialFocusState>,
): DorySpatialFocusState {
  if (focus === undefined) {
    return DEFAULT_DORY_SPATIAL_FOCUS_STATE;
  }

  const d = DEFAULT_DORY_SPATIAL_FOCUS_STATE;
  return Object.freeze({
    macroDisplayMode: focus.macroDisplayMode ?? d.macroDisplayMode,
    selectedPhysicalLayerIdx: focus.selectedPhysicalLayerIdx ?? d.selectedPhysicalLayerIdx,
    selectedLoopIteration: focus.selectedLoopIteration ?? d.selectedLoopIteration,
    selectedTokenOffset:
      focus.selectedTokenOffset !== undefined ? focus.selectedTokenOffset : d.selectedTokenOffset,
    includeCounterfactualCoda: focus.includeCounterfactualCoda ?? d.includeCounterfactualCoda,
    counterfactualLoop: focus.counterfactualLoop ?? d.counterfactualLoop,
    gqaLayerIdx: focus.gqaLayerIdx ?? d.gqaLayerIdx,
    selectedQHead: focus.selectedQHead ?? d.selectedQHead,
    selectedQueryRow:
      focus.selectedQueryRow !== undefined ? focus.selectedQueryRow : d.selectedQueryRow,
    selectedKeyCol:
      focus.selectedKeyCol !== undefined ? focus.selectedKeyCol : d.selectedKeyCol,
    selectedRotaryPlane: focus.selectedRotaryPlane ?? d.selectedRotaryPlane,
    swigluLayerIdx: focus.swigluLayerIdx ?? d.swigluLayerIdx,
    swigluChannel: focus.swigluChannel ?? d.swigluChannel,
    swigluOutputDim: focus.swigluOutputDim ?? d.swigluOutputDim,
    swigluAccumReveal: clampUnit(
      focus.swigluAccumReveal ?? d.swigluAccumReveal,
      d.swigluAccumReveal,
    ),
    kvComparePartitionMode: focus.kvComparePartitionMode ?? d.kvComparePartitionMode,
    kvCompareLayerIdx: focus.kvCompareLayerIdx ?? d.kvCompareLayerIdx,
    kvCompareChunkPartition:
      focus.kvCompareChunkPartition !== undefined
        ? focus.kvCompareChunkPartition
        : d.kvCompareChunkPartition,
    kvCompareKLoopsOverride:
      focus.kvCompareKLoopsOverride !== undefined
        ? focus.kvCompareKLoopsOverride
        : d.kvCompareKLoopsOverride,
    selectedVerifyOffset: focus.selectedVerifyOffset ?? d.selectedVerifyOffset,
    selfSpecAttentionLayerIdx: focus.selfSpecAttentionLayerIdx ?? d.selfSpecAttentionLayerIdx,
    selfSpecPreviewSampling:
      focus.selfSpecPreviewSampling !== undefined
        ? focus.selfSpecPreviewSampling
        : d.selfSpecPreviewSampling,
    selectedCheckpointFormat: focus.selectedCheckpointFormat ?? d.selectedCheckpointFormat,
    selectedVocabTokenId:
      focus.selectedVocabTokenId !== undefined
        ? focus.selectedVocabTokenId
        : d.selectedVocabTokenId,
    ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
    hoveredPartId: focus.hoveredPartId !== undefined ? focus.hoveredPartId : d.hoveredPartId,
    hoveredCell: focus.hoveredCell !== undefined ? focus.hoveredCell : d.hoveredCell,
  });
}

/**
 * Pure type-safe projection from unified `DorySpatialFocusState` to the exact options object
 * accepted by view `V`'s scene builder.
 */
export function focusStateToViewOptions<V extends DorySpatialViewMode>(
  viewMode: V,
  focus: DorySpatialFocusState,
): DorySceneOptionsByView[V] {
  switch (viewMode) {
    case 'macro': {
      const opts: DoryMacroSceneOptions = {
        displayMode: focus.macroDisplayMode,
        selectedPhysicalLayerIdx: focus.selectedPhysicalLayerIdx,
        selectedLoopIteration: focus.selectedLoopIteration,
        hoveredPartId: focus.hoveredPartId,
        hoveredCell: focus.hoveredCell,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
      };
      return opts as DorySceneOptionsByView[V];
    }
    case 'hypersphere': {
      const opts: DoryHypersphereSceneOptions = {
        physicalLayerIdx: focus.selectedPhysicalLayerIdx,
        loopIteration: focus.selectedLoopIteration,
        ...(focus.selectedTokenOffset !== null ? { tokenOffset: focus.selectedTokenOffset } : {}),
        includeCounterfactualCoda: focus.includeCounterfactualCoda,
        counterfactualLoop: focus.counterfactualLoop,
        hoveredCell: focus.hoveredCell,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
      };
      return opts as DorySceneOptionsByView[V];
    }
    case 'gqa_rope': {
      const opts: DoryGqaRopeSceneOptions = {
        physicalLayerIdx: focus.gqaLayerIdx,
        loopIteration: focus.selectedLoopIteration,
        selectedQHead: focus.selectedQHead,
        ...(focus.selectedQueryRow !== null ? { selectedQueryRow: focus.selectedQueryRow } : {}),
        ...(focus.selectedKeyCol !== null ? { selectedKeyCol: focus.selectedKeyCol } : {}),
        selectedRotaryPlane: focus.selectedRotaryPlane,
        hoveredCell: focus.hoveredCell,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
      };
      return opts as DorySceneOptionsByView[V];
    }
    case 'swiglu': {
      const opts: DorySwigluSceneOptions = {
        physicalLayerIdx: focus.swigluLayerIdx,
        loopIteration: focus.selectedLoopIteration,
        ...(focus.selectedTokenOffset !== null
          ? { selectedTokenRow: focus.selectedTokenOffset }
          : {}),
        selectedChannel: focus.swigluChannel,
        selectedOutputDim: focus.swigluOutputDim,
        accumReveal: focus.swigluAccumReveal,
        hoveredCell: focus.hoveredCell,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
      };
      return opts as DorySceneOptionsByView[V];
    }
    case 'kv_compare': {
      const opts: DoryKvCompareSceneOptions = {
        partitionMode: focus.kvComparePartitionMode,
        ...(focus.kvCompareChunkPartition != null
          ? { chunkPartition: focus.kvCompareChunkPartition }
          : {}),
        physicalLayerIdx: focus.kvCompareLayerIdx,
        loopIteration: focus.selectedLoopIteration,
        ...(focus.kvCompareKLoopsOverride != null
          ? { kLoopsOverride: focus.kvCompareKLoopsOverride }
          : {}),
        hoveredCell: focus.hoveredCell,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
      };
      return opts as DorySceneOptionsByView[V];
    }
    case 'self_spec': {
      const opts: DorySelfSpecSceneOptions = {
        selectedVerifyOffset: focus.selectedVerifyOffset,
        selectedAttentionLayerIdx: focus.selfSpecAttentionLayerIdx,
        ...(focus.selfSpecPreviewSampling != null
          ? { previewSampling: focus.selfSpecPreviewSampling }
          : {}),
        hoveredCell: focus.hoveredCell,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
      };
      return opts as DorySceneOptionsByView[V];
    }
    case 'bookends': {
      const opts: DoryBookendsSceneOptions = {
        ...(focus.selectedTokenOffset !== null
          ? { selectedTokenOffset: focus.selectedTokenOffset }
          : {}),
        selectedVocabTokenId: focus.selectedVocabTokenId ?? null,
        selectedCheckpointFormat: focus.selectedCheckpointFormat,
        hoveredCell: focus.hoveredCell,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
      };
      return opts as DorySceneOptionsByView[V];
    }
    default: {
      const exhaustive: never = viewMode;
      throw new RangeError(`Unknown DorySpatialViewMode: '${String(exhaustive)}'.`);
    }
  }
}

// ============================================================================
// UNIFIED 7-VIEW SPATIAL SCENE DISPATCHER
// ============================================================================

/**
 * Dispatches cleanly to any of the 7 `D7` Dory1.0-4B spatial scene builders (`macro`,
 * `hypersphere`, `gqa_rope`, `swiglu`, `kv_compare`, `self_spec`, `bookends`) over a strictly
 * non-mutating resolved `DorySession` context.
 */
export function buildDorySpatialScene(
  sessionOrCtx: DorySession | DoryResolvedSessionContext,
  viewMode: DorySpatialViewMode,
  focus?: Partial<DorySpatialFocusState>,
): DorySpatialScene {
  // Validate viewMode explicitly before resolving session context
  getDorySpatialViewDescriptor(viewMode);

  const resolvedFocus = resolveDorySpatialFocus(focus);
  const ctx = ensureDorySessionContext(sessionOrCtx, resolvedFocus.recordId);

  switch (viewMode) {
    case 'macro':
      return buildDoryMacroScene(ctx, focusStateToViewOptions('macro', resolvedFocus));
    case 'hypersphere':
      return buildDoryHypersphereScene(ctx, focusStateToViewOptions('hypersphere', resolvedFocus));
    case 'gqa_rope':
      return buildDoryGqaRopeScene(ctx, focusStateToViewOptions('gqa_rope', resolvedFocus));
    case 'swiglu':
      return buildDorySwigluScene(ctx, focusStateToViewOptions('swiglu', resolvedFocus));
    case 'kv_compare':
      return buildDoryKvCompareScene(ctx, focusStateToViewOptions('kv_compare', resolvedFocus));
    case 'self_spec':
      return buildDorySelfSpecScene(ctx, focusStateToViewOptions('self_spec', resolvedFocus));
    case 'bookends':
      return buildDoryBookendsScene(ctx, focusStateToViewOptions('bookends', resolvedFocus));
    default: {
      const exhaustive: never = viewMode;
      throw new RangeError(`Unknown DorySpatialViewMode: '${String(exhaustive)}'.`);
    }
  }
}

// Compile-time verification against frozen Step-0 contract (`src/dory/scene/types.ts`)
const _SCENE_INDEX_CONTRACT_CHECK: DorySceneIndexModuleContract = {
  DORY_SCENE_BUILDERS,
  getDorySpatialViewDescriptor,
  resolveDorySpatialFocus,
  focusStateToViewOptions,
  buildDorySpatialScene,
};
void _SCENE_INDEX_CONTRACT_CHECK;
