/**
 * 64×64 RG32F SwiGLU Cutaway Texture Atlas (`src/diffusion/scene/swigluAtlas.ts`)
 *
 * Packs all 11 live SwiGLU tensors + post-attention RMSNorm gamma into a single
 * 64×64 RG32F WebGL2 atlas (`32 KiB`, `0` overlaps, verified via `auditAtlasRegions`):
 *   - Weights: `w_gate [8×24]`, `w_up [8×24]`, `w_down [24×8]`, `norm2_gamma [1×8]`
 *   - Sequence tensors (`S <= 16`): `post_attn_resid [S×8]`, `norm2 [S×8]`,
 *     `gate_pre [S×24]`, `gate_silu [S×24]`, `up_val [S×24]`, `gated [S×24]`,
 *     `swiglu_out [S×8]`, `layer_out [S×8]`
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, or `src/twotower/` imports.
 */

import {
  computeMaxAbs,
  createRg32fAtlas,
  type AtlasSlice,
  type Rg32fValueAtlas,
} from '../../scene/atlasPacker';
import type {
  DiffusionLayerTrace,
  DiffusionLayerWeights,
  DiffusionMiniConfig,
} from '../types';
import { auditAtlasRegions } from './provenance';
import type {
  DiffusionAtlasLayoutAudit,
  DiffusionAtlasRegionBox,
} from './types';

export const DIFFUSION_SWIGLU_ATLAS_SIZE = 64 as const;
export const DIFFUSION_SWIGLU_MAX_QUERY = 16 as const;

export interface DiffusionSwigluAtlasResult {
  readonly atlas: Rg32fValueAtlas;
  readonly audit: DiffusionAtlasLayoutAudit;
  readonly slices: Readonly<Record<string, AtlasSlice>>;
  readonly queryTokens: number;
  readonly atlasKey: string;
}

function packRowMajorMatrix(
  writer: ReturnType<typeof createRg32fAtlas>,
  regions: DiffusionAtlasRegionBox[],
  slices: Record<string, AtlasSlice>,
  id: string,
  x0: number,
  y0: number,
  cols: number,
  rows: number,
  src: Float32Array,
  srcCols: number,
  mode: 0 | 1 = 0,
  transpose = false,
): void {
  let maxAbs = 0.25;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = transpose ? src[c * srcCols + r]! : src[r * srcCols + c]!;
      writer.setTexel(x0 + c, y0 + r, val, false);
      if (Number.isFinite(val)) {
        const a = Math.abs(val);
        if (a > maxAbs) maxAbs = a;
      }
    }
  }
  regions.push({
    id,
    x0,
    y0,
    cols,
    rows,
    mode,
    isMaskOrScore: false,
  });
  slices[id] = {
    rect: writer.makeRect(x0, y0, cols, rows),
    scale: mode === 1 ? 1.0 : 1.0 / maxAbs,
    mode,
  };
}

/**
 * Pack a single layer's dense SwiGLU weights and activations into a non-overlapping
 * `64×64` RG32F value atlas (`32 KiB`).
 *
 * Layout map (`S <= 16`, `d = 8`, `dFf = 24`):
 *   - `w_gate`          `[cols=24, rows=8]`  at `(0,  0)`  (transposed display `[d × dFf]` matching `norm2 · W_gate^T`)
 *   - `w_up`            `[cols=24, rows=8]`  at `(24, 0)`  (transposed display `[d × dFf]` matching `norm2 · W_up^T`)
 *   - `w_down`          `[cols=8,  rows=24]` at `(48, 0)`  (transposed display `[dFf × d]` matching `gated · W_down^T`)
 *   - `gate_pre`        `[cols=24, rows=S]`  at `(0,  8)`
 *   - `gate_silu`       `[cols=24, rows=S]`  at `(24, 8)`
 *   - `up_val`          `[cols=24, rows=S]`  at `(0,  24)`
 *   - `gated`           `[cols=24, rows=S]`  at `(24, 24)`
 *   - `post_attn_resid` `[cols=8,  rows=S]`  at `(56, 0)`
 *   - `norm2`           `[cols=8,  rows=S]`  at `(56, 16)`
 *   - `swiglu_out`      `[cols=8,  rows=S]`  at `(56, 32)`
 *   - `layer_out`       `[cols=8,  rows=S]`  at `(56, 48)`
 *   - `norm2_gamma`     `[cols=8,  rows=1]`  at `(0,  40)`
 */
export function buildDiffusionSwigluAtlas(
  trace: DiffusionLayerTrace,
  layerWeights: DiffusionLayerWeights,
  config: DiffusionMiniConfig,
  options: { readonly layerIndex: number; readonly snapshotDigest: string },
): DiffusionSwigluAtlasResult {
  const size = DIFFUSION_SWIGLU_ATLAS_SIZE;
  const d = config.hiddenSize; // 8
  const dFf = config.intermediateSize; // 24
  const queryTokens = Math.max(
    1,
    Math.min(trace.swiglu.seqLen, DIFFUSION_SWIGLU_MAX_QUERY),
  );

  const writer = createRg32fAtlas(size, size);
  const regions: DiffusionAtlasRegionBox[] = [];
  const slices: Record<string, AtlasSlice> = {};

  // Weights: stored [outDim, inDim]; packed transposed [inDim rows × outDim cols] to align with matmul axes
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'w_gate',
    0,
    0,
    dFf,
    d,
    layerWeights.mlp.gateProj,
    d,
    0,
    true,
  );
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'w_up',
    24,
    0,
    dFf,
    d,
    layerWeights.mlp.upProj,
    d,
    0,
    true,
  );
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'w_down',
    48,
    0,
    d,
    dFf,
    layerWeights.mlp.downProj,
    dFf,
    0,
    true,
  );

  // Intermediate SwiGLU activations [S × dFf]
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'gate_pre',
    0,
    8,
    dFf,
    queryTokens,
    trace.swiglu.gatePre,
    dFf,
    0,
    false,
  );
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'gate_silu',
    24,
    8,
    dFf,
    queryTokens,
    trace.swiglu.gateSilu,
    dFf,
    0,
    false,
  );
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'up_val',
    0,
    24,
    dFf,
    queryTokens,
    trace.swiglu.upVal,
    dFf,
    0,
    false,
  );
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'gated',
    24,
    24,
    dFf,
    queryTokens,
    trace.swiglu.gated,
    dFf,
    0,
    false,
  );

  // Residual stream & hidden-dim activations [S × d]
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'post_attn_resid',
    56,
    0,
    d,
    queryTokens,
    trace.postAttentionResidual,
    d,
    0,
    false,
  );
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'norm2',
    56,
    16,
    d,
    queryTokens,
    trace.postAttentionNorm,
    d,
    0,
    false,
  );
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'swiglu_out',
    56,
    32,
    d,
    queryTokens,
    trace.swiglu.output,
    d,
    0,
    false,
  );
  packRowMajorMatrix(
    writer,
    regions,
    slices,
    'layer_out',
    56,
    48,
    d,
    queryTokens,
    trace.output,
    d,
    0,
    false,
  );

  // RMSNorm weight vector gamma [1 × d]
  const gammaScale = 1.0 / computeMaxAbs(layerWeights.postAttentionLayerNorm, 0.25);
  for (let c = 0; c < d; c++) {
    writer.setTexel(c, 40, layerWeights.postAttentionLayerNorm[c]!, false);
  }
  regions.push({
    id: 'norm2_gamma',
    x0: 0,
    y0: 40,
    cols: d,
    rows: 1,
    mode: 0,
    isMaskOrScore: false,
  });
  slices.norm2_gamma = {
    rect: writer.makeRect(0, 40, d, 1),
    scale: gammaScale,
    mode: 0,
  };

  const audit = auditAtlasRegions(size, size, regions);
  if (audit.hasOverlap || audit.outOfBoundsIds.length > 0) {
    throw new Error(
      `SwiGLU 64×64 atlas layout error: overlap=${String(audit.overlapPair)}, oob=${audit.outOfBoundsIds.join(',')}`,
    );
  }

  return {
    atlas: {
      width: size,
      height: size,
      data: writer.data,
    },
    audit,
    slices: Object.freeze(slices),
    queryTokens,
    atlasKey: `swiglu|L=${options.layerIndex}|S=${queryTokens}|snap=${options.snapshotDigest}`,
  };
}
