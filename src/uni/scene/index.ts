/**
 * Unified Spatial Scene Barrel for Nemotron-Labs-Diffusion-Uni (`src/uni/scene/index.ts`)
 *
 * Re-exports pinned contracts, provenance/pick/atlas/5-pass buffer helpers, and all
 * 7 headless 3D spatial scene builders (`UNI4` Views 1–3 & `UNI5` Views 4–7):
 *   - View 1 (`macro`): `macroScene.ts` (`0 B` RG32F)
 *   - View 2 (`visual_input`): `visualInputScene.ts` (`128×128` RG32F `128.00 KiB`)
 *   - View 3 (`joint_attn`): `jointAttnAtlas.ts` + `jointAttnCutaway.ts` (`256×256` RG32F `512.00 KiB`)
 *   - View 4 (`ball_tree`): `ballTreeScene.ts` (`128×128` RG32F `128.00 KiB`)
 *   - View 5 (`refinement`): `refinementCanvasScene.ts` (`128×128` RG32F `128.00 KiB`)
 *   - View 6 (`output_edit`): `outputEditScene.ts` (`128×128` RG32F `128.00 KiB`)
 *   - View 7 (`opd_readonly`): `opdExplorerScene.ts` (`128×64` RG32F `64.00 KiB`)
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

export * from './types';
export * from './provenance';
export * from './macroScene';
export {
  UNI_VISUAL_INPUT_ATLAS_BUDGET_BYTES,
  UNI_VISUAL_INPUT_ATLAS_SIZE,
  UNI_VISUAL_INPUT_DEFAULT_PRESET,
  UNI_VISUAL_INPUT_IBQ_SLICE_COLS,
  UNI_VISUAL_INPUT_REGION_COUNT,
  buildUniVisualInputScene,
  packVisualInputAtlas,
  runVisualInputPipelines,
  type UniVisualInputAtlasPackResult,
  type UniVisualInputAtlasSliceEntry,
  type UniVisualInputPipelineRun,
  type UniVisualInputStageId,
} from './visualInputScene';
export {
  UNI_JOINT_ATTN_ATLAS_HEIGHT,
  UNI_JOINT_ATTN_ATLAS_WIDTH,
  UNI_JOINT_ATTN_MAX_KEY,
  UNI_JOINT_ATTN_MAX_QUERY,
  UNI_JOINT_ATTN_MAX_REGIONS,
  packUniJointAttnAtlas,
  readAtlasTexel,
  type UniJointAttnAtlasInput,
  type UniJointAttnAtlasPack,
  type UniJointAttnAtlasSlices,
} from './jointAttnAtlas';
export * from './jointAttnCutaway';
export * from './ballTreeScene';
export * from './refinementCanvasScene';
export * from './outputEditScene';
export * from './opdExplorerScene';

import type { UniSession } from '../session';
import { buildUniBallTreeScene } from './ballTreeScene';
import { buildUniJointAttnScene } from './jointAttnCutaway';
import { buildUniMacroScene } from './macroScene';
import { buildUniOpdExplorerScene } from './opdExplorerScene';
import { buildUniOutputEditScene } from './outputEditScene';
import { buildUniRefinementCanvasScene } from './refinementCanvasScene';
import {
  DEFAULT_UNI_SPATIAL_FOCUS_STATE,
  type UniSpatialFocusState,
  type UniSpatialScene,
  type UniSpatialViewMode,
} from './types';
import { buildUniVisualInputScene } from './visualInputScene';

/**
 * Unified typed scene dispatcher routing all 7 NLD-Uni spatial views (`macro`, `visual_input`,
 * `joint_attn`, `ball_tree`, `refinement`, `output_edit`, `opd_readonly`) from a shared focus state.
 *
 * Views `macro`, `ball_tree`, and `opd_readonly` support `session === null`; session-backed views
 * (`visual_input`, `joint_attn`, `refinement`, `output_edit`) require an active `UniSession`.
 */
export function buildUniSpatialScene(
  viewMode: UniSpatialViewMode,
  session: UniSession | null,
  focusPatch?: Partial<UniSpatialFocusState>,
): UniSpatialScene {
  const focus: UniSpatialFocusState = focusPatch
    ? { ...DEFAULT_UNI_SPATIAL_FOCUS_STATE, ...focusPatch }
    : DEFAULT_UNI_SPATIAL_FOCUS_STATE;

  switch (viewMode) {
    case 'macro':
      return buildUniMacroScene(session, {
        ...(focus.executionViewId !== undefined ? { executionViewId: focus.executionViewId } : {}),
        prefillMode: focus.prefillMode,
        selectedLayerIndex: focus.selectedLayerIndex,
        hoveredPartId: focus.hoveredPartId,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
      });

    case 'visual_input': {
      if (!session) {
        throw new RangeError('buildUniSpatialScene("visual_input") requires a live UniSession');
      }
      return buildUniVisualInputScene(session, {
        ...(focus.scenePresetId !== undefined ? { scenePresetId: focus.scenePresetId } : {}),
        ...(focus.gainPresetId !== undefined ? { gainPresetId: focus.gainPresetId } : {}),
        ...(focus.includeDiscreteStream !== undefined
          ? { includeDiscreteStream: focus.includeDiscreteStream }
          : {}),
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
        hoveredCell: focus.hoveredCell,
      });
    }

    case 'joint_attn': {
      if (!session) {
        throw new RangeError('buildUniSpatialScene("joint_attn") requires a live UniSession');
      }
      return buildUniJointAttnScene(session, {
        layerIndex: focus.jointLayerIndex,
        batchRow: focus.batchRow,
        prefillMode: focus.prefillMode,
        selectedQHead: focus.selectedQHead,
        selectedTokenRow: focus.selectedTokenRow,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
        hoveredCell: focus.hoveredCell,
      });
    }

    case 'ball_tree':
      return buildUniBallTreeScene(session, {
        selectedCodeId: focus.selectedCodeId,
        refinementSteps: focus.refinementSteps,
        stageIndex: focus.stageIndex,
        showProductionD17Comparison: focus.showProductionD17Comparison,
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
        hoveredCell: focus.hoveredCell,
      });

    case 'refinement': {
      if (!session) {
        throw new RangeError('buildUniSpatialScene("refinement") requires a live UniSession');
      }
      return buildUniRefinementCanvasScene(session, {
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
        selectedGridCell: focus.selectedGridCell,
        showStratifiedOrder: focus.showStratifiedOrder,
        cfgBranchRow: focus.cfgBranchRow,
        hoveredCell: focus.hoveredCell,
      });
    }

    case 'output_edit': {
      if (!session) {
        throw new RangeError('buildUniSpatialScene("output_edit") requires a live UniSession');
      }
      return buildUniOutputEditScene(session, {
        ...(focus.recordId !== undefined ? { recordId: focus.recordId } : {}),
        ...(focus.editGuidanceMode !== undefined
          ? { editGuidanceMode: focus.editGuidanceMode }
          : {}),
        cropToSource: focus.cropToSource,
        showPaddingLockOverlay: focus.showPaddingLockOverlay,
        hoveredCell: focus.hoveredCell,
      });
    }

    case 'opd_readonly':
      return buildUniOpdExplorerScene(session, {
        seed: focus.opdSeed,
        r: focus.opdR,
        steps: focus.opdSteps,
        seqLen: focus.opdSeqLen,
        selectedBay: focus.opdSelectedBay,
        hoveredCell: focus.hoveredCell,
      });
  }
}
