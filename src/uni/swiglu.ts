/**
 * Routed 3.5x SwiGLU Feed-Forward Network (`src/uni/swiglu.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching Ministral 3 /
 * NLD-Uni branch-routed SwiGLU MLP (`d=8 -> d_ff=28 -> d=8`):
 *   `output = W_down(SiLU(W_gate(x)) ⊙ W_up(x))`
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG, or forbidden imports.
 */

import { fmtSigned } from '../shared/format';
import { linearProject, routedLinearProject } from './attention';
import {
  UNI_MINI_CONFIG,
  type UniBranchSwigluWeights,
  type UniMiniConfig,
  type UniSwigluTrace,
} from './types';

/**
 * Float32 Sigmoid Linear Unit activation: `SiLU(x) = x / (1 + exp(-x))`,
 * returning `0.0` when `Math.exp(-x)` overflows.
 */
export function silu(x: number): number {
  const denom = 1.0 + Math.exp(-x);
  if (!Number.isFinite(denom)) return 0.0;
  return Math.fround(x / denom);
}

/**
 * Execute a single-branch dense SwiGLU feed-forward block and capture intermediates.
 */
export function runUniBranchSwiglu(
  input: Float32Array,
  seqLen: number,
  weights: UniBranchSwigluWeights,
  config: UniMiniConfig,
): UniSwigluTrace {
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

/**
 * Execute a per-token U/G routed SwiGLU feed-forward block.
 * When `skipU` is true, `U` rows (`routing[t] === 0`) are left zeroed and
 * `weightsU` is never read.
 */
export function runUniRoutedSwiglu(
  input: Float32Array,
  seqLen: number,
  routing: Uint8Array,
  weightsU: UniBranchSwigluWeights,
  weightsG: UniBranchSwigluWeights,
  config: UniMiniConfig,
  skipU = false,
): UniSwigluTrace {
  const d = config.hiddenSize;
  const dFf = config.intermediateSize;

  const gatePre = routedLinearProject(
    input,
    seqLen,
    d,
    routing,
    weightsU.gateProj,
    weightsG.gateProj,
    dFf,
    skipU,
  );
  const upVal = routedLinearProject(
    input,
    seqLen,
    d,
    routing,
    weightsU.upProj,
    weightsG.upProj,
    dFf,
    skipU,
  );

  const gateSilu = new Float32Array(seqLen * dFf);
  const gated = new Float32Array(seqLen * dFf);

  for (let t = 0; t < seqLen; t++) {
    if (skipU && routing[t] === 0) continue;
    const base = t * dFf;
    for (let f = 0; f < dFf; f++) {
      const idx = base + f;
      const s = silu(gatePre[idx]);
      gateSilu[idx] = s;
      gated[idx] = Math.fround(s * upVal[idx]);
    }
  }

  const output = routedLinearProject(
    gated,
    seqLen,
    dFf,
    routing,
    weightsU.downProj,
    weightsG.downProj,
    d,
    skipU,
  );

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

export interface UniSwigluCellFormulaParams {
  readonly stage: 'gate_pre' | 'gate_silu' | 'up' | 'gated' | 'output';
  readonly tokenIndex: number;
  readonly channelIndex: number;
  readonly value: number;
  readonly branch?: 'U' | 'G';
  readonly gatePreVal?: number;
  readonly gateSiluVal?: number;
  readonly upVal?: number;
  readonly dFf?: number;
}

/**
 * Format a concise human-readable equation for a single SwiGLU cell.
 */
export function formatUniSwigluCellFormula(
  params: UniSwigluCellFormulaParams,
): string {
  const {
    stage,
    tokenIndex,
    channelIndex,
    value,
    branch = 'U',
    gatePreVal,
    gateSiluVal,
    upVal,
    dFf = UNI_MINI_CONFIG.intermediateSize,
  } = params;

  const tag = branch === 'G' ? 'G' : 'U';

  switch (stage) {
    case 'gate_pre':
      return `gate_pre[${tag}:t=${tokenIndex}, f=${channelIndex}] = (W_gate^${tag}·u)_${channelIndex} = ${fmtSigned(value)}`;
    case 'gate_silu':
      return `gate_silu[${tag}:t=${tokenIndex}, f=${channelIndex}] = SiLU(${fmtSigned(gatePreVal ?? 0)}) = ${fmtSigned(value)}`;
    case 'up':
      return `up[${tag}:t=${tokenIndex}, f=${channelIndex}] = (W_up^${tag}·u)_${channelIndex} = ${fmtSigned(value)}`;
    case 'gated':
      return `gated[${tag}:t=${tokenIndex}, f=${channelIndex}] = ${fmtSigned(gateSiluVal ?? 0)} ⊙ ${fmtSigned(upVal ?? 0)} = ${fmtSigned(value)}`;
    case 'output':
      return `mlp_out[${tag}:t=${tokenIndex}, d=${channelIndex}] = Σ_{f=0..${dFf - 1}} W_down^${tag}[${channelIndex},f]·gated[f] = ${fmtSigned(value)}`;
  }
}
