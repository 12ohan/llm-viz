/**
 * Headless Barrel for Nemotron-Labs-Diffusion-Uni (`src/uni/index.ts`)
 *
 * Pure headless TypeScript entry point re-exporting all UNI1, UNI2, and UNI3
 * simulation, multimodal, diffusion, OPD reference, telemetry, and bounded
 * interactive session modules with zero DOM, WebGL, or sibling-app imports.
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
