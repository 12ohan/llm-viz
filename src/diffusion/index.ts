/**
 * Public Barrel for DF1 Headless Diffusion Foundation (`src/diffusion/index.ts`)
 *
 * Re-exports shared contracts, specification tables, deterministic PRNGs,
 * 5-mode masks, and core headless math kernels (YaRN RoPE, GQA + Draft-Only LoRA,
 * Dense SwiGLU, Deterministic Parameters, 40L Forward Pass).
 */

export * from './types';
export * from './spec';
export * from './prng';
export * from './masks';
export * from './rope';
export * from './attention';
export * from './swiglu';
export * from './parameters';
export * from './forward';
