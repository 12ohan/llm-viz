/**
 * Diffusion Spatial Scene Barrel (`src/diffusion/scene/index.ts`)
 *
 * Exports shared spatial contracts, provenance/audit builders, and Slice 3B spatial views
 * (`swigluAtlas`, `swigluCutaway`, `maskExplorerScene`, `specVerifyScene`, `bookendCutaways`).
 * Slice 3A exports (`macroScene`, `gqaRopeAtlas`, `gqaRopeCutaway`) and the unified router
 * will be added upon merging 3A + 3B.
 */

export * from './types';
export * from './provenance';
export * from './swigluAtlas';
export * from './swigluCutaway';
export * from './maskExplorerScene';
export * from './specVerifyScene';
export * from './bookendCutaways';
