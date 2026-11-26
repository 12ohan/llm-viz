/**
 * Unified Diffusion Spatial Scene Barrel & 7-View Router (`src/diffusion/scene/index.ts`)
 *
 * Pure headless TypeScript — zero DOM, WebGL execution, `Math.random()`, or `src/twotower/` imports.
 */

import { buildDiffusionEmbeddingCutaway, buildDiffusionOutputCutaway } from './bookendCutaways';
import { buildDiffusionGqaRopeCutaway } from './gqaRopeCutaway';
import { buildDiffusionMacroScene } from './macroScene';
import { buildDiffusionMaskExplorerScene } from './maskExplorerScene';
import { buildDiffusionSpecVerifyScene } from './specVerifyScene';
import { buildDiffusionSwigluCutaway } from './swigluCutaway';
import type {
  DiffusionFocusState,
  DiffusionResolvedSessionContext,
  DiffusionSpatialScene,
} from './types';

export * from './types';
export * from './provenance';
export * from './macroScene';
export * from './gqaRopeAtlas';
export * from './gqaRopeCutaway';
export * from './swigluAtlas';
export * from './swigluCutaway';
export * from './maskExplorerScene';
export * from './specVerifyScene';
export * from './bookendCutaways';

/** Unified router building any of the 7 DF3 architecture-specific WebGL2 spatial views. */
export function buildDiffusionSpatialScene(
  ctx: DiffusionResolvedSessionContext,
  focus: DiffusionFocusState,
): DiffusionSpatialScene {
  switch (focus.viewMode) {
    case 'macro':
      return buildDiffusionMacroScene(ctx, focus);
    case 'gqa_rope':
      return buildDiffusionGqaRopeCutaway(ctx, focus);
    case 'swiglu':
      return buildDiffusionSwigluCutaway(ctx, focus);
    case 'mask_explorer':
      return buildDiffusionMaskExplorerScene(ctx, focus);
    case 'spec_verify':
      return buildDiffusionSpecVerifyScene(ctx, focus);
    case 'embedding':
      return buildDiffusionEmbeddingCutaway(ctx, focus);
    case 'output':
      return buildDiffusionOutputCutaway(ctx, focus);
  }
}
