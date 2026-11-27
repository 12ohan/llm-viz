/**
 * Dense SwiGLU Feed-Forward Network (`src/diffusion/swiglu.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching `Ministral3MLP`:
 *   `output = W_down(SiLU(W_gate(x)) ⊙ W_up(x))`
 *
 * Pure headless TypeScript — deterministic, self-contained math kernel.
 */

import { fmtSigned } from '../shared/format';
import { linearProject } from './attention';
import type {
  DiffusionMiniConfig,
  DiffusionSwigluTrace,
  DiffusionSwigluWeights,
} from './types';

/**
 * Float32 Sigmoid Linear Unit activation: `SiLU(x) = x / (1 + exp(-x))`.
 */
export function silu(x: number): number {
  const denom = 1.0 + Math.exp(-x);
  if (!Number.isFinite(denom)) return 0.0;
  return Math.fround(x / denom);
}

/**
 * Execute a single dense SwiGLU feed-forward block and capture every intermediate
 * tensor for cell-level Formula Inspection and 64x64 texture atlas packing.
 */
export function runDiffusionSwiglu(
  input: Float32Array,
  seqLen: number,
  weights: DiffusionSwigluWeights,
  config: DiffusionMiniConfig,
): DiffusionSwigluTrace {
  const d = config.hiddenSize;
  const dFf = config.intermediateSize;

  const gatePre = linearProject(input, seqLen, d, weights.gateProj, dFf);
  const gateSilu = new Float32Array(seqLen * dFf);
  for (let i = 0; i < seqLen * dFf; i++) {
    gateSilu[i] = silu(gatePre[i]);
  }

  const upVal = linearProject(input, seqLen, d, weights.upProj, dFf);
  const gated = new Float32Array(seqLen * dFf);
  for (let i = 0; i < seqLen * dFf; i++) {
    gated[i] = Math.fround(gateSilu[i] * upVal[i]);
  }

  const output = linearProject(gated, seqLen, dFf, weights.downProj, d);

  return {
    gatePre,
    gateSilu,
    upVal,
    gated,
    output,
    seqLen,
    d,
    dFf,
  };
}

export interface DiffusionSwigluCellFormulaParams {
  readonly stage: 'gate_pre' | 'gate_silu' | 'up' | 'gated' | 'output';
  readonly tokenIndex: number;
  readonly channelIndex: number;
  readonly value: number;
  readonly gatePreVal?: number;
  readonly gateSiluVal?: number;
  readonly upVal?: number;
  readonly dFf?: number;
}

/**
 * Format a concise human-readable equation for a single SwiGLU cell.
 */
export function formatDiffusionSwigluCellFormula(
  params: DiffusionSwigluCellFormulaParams,
): string {
  const {
    stage,
    tokenIndex,
    channelIndex,
    value,
    gatePreVal,
    gateSiluVal,
    upVal,
    dFf = 24,
  } = params;

  switch (stage) {
    case 'gate_pre':
      return `gate_pre[t=${tokenIndex}, f=${channelIndex}] = (W_gate·u)_${channelIndex} = ${fmtSigned(value)}`;
    case 'gate_silu':
      return `gate_silu[t=${tokenIndex}, f=${channelIndex}] = SiLU(${fmtSigned(gatePreVal ?? 0)}) = ${fmtSigned(value)}`;
    case 'up':
      return `up[t=${tokenIndex}, f=${channelIndex}] = (W_up·u)_${channelIndex} = ${fmtSigned(value)}`;
    case 'gated':
      return `gated[t=${tokenIndex}, f=${channelIndex}] = ${fmtSigned(gateSiluVal ?? 0)} ⊙ ${fmtSigned(upVal ?? 0)} = ${fmtSigned(value)}`;
    case 'output':
      return `mlp_out[t=${tokenIndex}, d=${channelIndex}] = Σ_{f=0..${dFf - 1}} W_down[${channelIndex},f]·gated[f] = ${fmtSigned(value)}`;
  }
}
