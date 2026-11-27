/**
 * Headless Barrel for Nemotron-Labs-Diffusion-Uni (`src/uni/index.ts`)
 *
 * Pure headless TypeScript entry point re-exporting all UNI1, UNI2, UNI3, and UNI4
 * simulation, multimodal, diffusion, OPD reference, telemetry, bounded interactive
 * session, and headless spatial scene builder modules with zero DOM, WebGL execution,
 * or sibling-app imports.
 */

export * from './types';
export * from './spec';
export * from './prng';
export * from './rope';
export * from './masks';
export * from './attention';
export * from './swiglu';
export * from './parameters';
export * from './forward';
export * from './vision';
export * from './vqvae';
export * from './codec';
export * from './kvCache';
export * from './textSampler';
export * from './tree';
export * from './opdRef';
export * from './archReadouts';
export * from './imageSampler';
export * from './session';
export * from './scene/types';
export * from './scene/provenance';
export * from './scene/macroScene';
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
} from './scene/jointAttnAtlas';
export * from './scene/jointAttnCutaway';
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
} from './scene/visualInputScene';
