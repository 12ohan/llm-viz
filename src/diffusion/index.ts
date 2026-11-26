/**
 * Public Barrel for DF1 Slice 1B Core Diffusion Kernels (`src/diffusion/index.ts`)
 *
 * Re-exports shared contracts and core headless math modules (YaRN RoPE,
 * GQA + Draft-Only LoRA, Dense SwiGLU, Deterministic Parameters, 40L Forward Pass).
 */

export * from './types';
export * from './rope';
export * from './attention';
export * from './swiglu';
export * from './parameters';
export * from './forward';
