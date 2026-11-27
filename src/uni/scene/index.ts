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
