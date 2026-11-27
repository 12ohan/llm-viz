/**
 * Headless Barrel for NVIDIA Nemotron-Labs-Dory1.0-4B (`src/dory/index.ts`)
 *
 * Pure headless TypeScript entry point re-exporting all DORY1 specification,
 * discrepancy ledger, deterministic PRNG, 507-tensor parameter builder, Full-nGPT
 * math kernels, dual-RoPE tables, dual-axis KV cache, Gated GQA attention,
 * Normalized SwiGLU MLP, and physical-72 depth-looped forward pass modules.
 *
 * Strictly zero DOM, WebGL, Node built-ins, ambient RNG, or sibling-app imports (`D4`).
 */

export * from './types';
export * from './spec';
export { DORY_MINI_CONFIG } from './types';
export * from './ledger';
export * from './prng';
export * from './parameters';
export * from './ngptMath';
export * from './rope';
export * from './kvCache';
export * from './attention';
export * from './swiglu';
export * from './forward';
