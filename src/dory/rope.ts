/**
 * Dual-Regime NeoX `rotate_half` Rotary Position Embeddings (`src/dory/rope.ts`).
 *
 * Implements `DoryRopeModuleContract` (`D2`, `D5`, `DR6`):
 *   - Sliding-Window Attention (`sliding`): `theta_S = 16` (`inv_freq = [1, 1/2, 1/4, 1/8]`)
 *   - Full Attention (`full`): `theta_F = 1,024`
 *   - 100% rotary (`partialRotaryFactor = 1.0`, `headDim = 8`, `4` rotary planes)
 *   - Canonical NeoX `rotate_half` pairing `(j, j + headDim / 2)`:
 *       `y[j]            = x[j] * cos - x[j + halfDim] * sin`
 *       `y[j + halfDim]  = x[j + halfDim] * cos + x[j] * sin`
 *   - Executed strictly BEFORE per-head Q/K L2Norm (`DR6`, `vLLM`/`llama.cpp`/TeX contract).
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL, Node built-ins, or sibling imports.
 */

import {
  DORY_MINI_CONFIG,
  type DoryDualRopeTables,
  type DoryMiniConfig,
  type DoryRopeRegime,
  type DoryRopeTable,
} from './types';

export function computeDoryRopeInvFreq(headDim: number, theta: number): Float32Array {
  if (!Number.isInteger(headDim) || headDim <= 0 || headDim % 2 !== 0) {
    throw new RangeError(
      `Invalid RoPE headDim ${String(headDim)}: expected positive even integer.`,
    );
  }
  if (!Number.isFinite(theta) || theta <= 0) {
    throw new RangeError(`Invalid RoPE theta ${String(theta)}: expected finite positive number.`);
  }

  const halfDim = headDim >> 1;
  const invFreq = new Float32Array(halfDim);
  for (let j = 0; j < halfDim; j++) {
    invFreq[j] = Math.pow(theta, -(2 * j) / headDim);
  }
  return invFreq;
}

export function buildDoryRopeTable(
  maxPositions: number,
  headDim: number,
  theta: number,
  regime: DoryRopeRegime,
): DoryRopeTable {
  if (!Number.isInteger(maxPositions) || maxPositions < 1) {
    throw new RangeError(
      `Invalid RoPE maxPositions ${String(maxPositions)}: expected integer >= 1.`,
    );
  }

  const invFreq = computeDoryRopeInvFreq(headDim, theta);
  const halfDim = headDim >> 1;
  const cos = new Float32Array(maxPositions * halfDim);
  const sin = new Float32Array(maxPositions * halfDim);

  for (let pos = 0; pos < maxPositions; pos++) {
    const rowBase = pos * halfDim;
    for (let j = 0; j < halfDim; j++) {
      const angle = pos * invFreq[j];
      cos[rowBase + j] = Math.cos(angle);
      sin[rowBase + j] = Math.sin(angle);
    }
  }

  return Object.freeze({
    regime,
    theta,
    headDim,
    halfDim,
    maxPositions,
    invFreq,
    cos,
    sin,
  });
}

export function buildDoryDualRopeTables(
  config: Pick<
    DoryMiniConfig,
    'maxPositionEmbeddings' | 'headDim' | 'ropeThetaSliding' | 'ropeThetaFull'
  > = DORY_MINI_CONFIG,
): DoryDualRopeTables {
  const sliding = buildDoryRopeTable(
    config.maxPositionEmbeddings,
    config.headDim,
    config.ropeThetaSliding,
    'sliding',
  );
  const full = buildDoryRopeTable(
    config.maxPositionEmbeddings,
    config.headDim,
    config.ropeThetaFull,
    'full',
  );
  return Object.freeze({ sliding, full });
}

function assertValidRopeCall(
  headDim: number,
  pos: number,
  table: DoryRopeTable,
): void {
  if (headDim !== table.headDim) {
    throw new RangeError(
      `RoPE headDim mismatch: tensor headDim=${String(headDim)}, table headDim=${String(table.headDim)}.`,
    );
  }
  if (!Number.isInteger(pos) || pos < 0 || pos >= table.maxPositions) {
    throw new RangeError(
      `RoPE position ${String(pos)} out of table bounds [0, ${table.maxPositions - 1}].`,
    );
  }
}

/**
 * Applies 100% NeoX `rotate_half` RoPE from `src` into `dst` for contiguous positions
 * `[startPos .. startPos + seqLen - 1]`. Safe for in-place execution (`src === dst`).
 */
export function applyNeoXRotateHalfInto(
  src: Float32Array,
  dst: Float32Array,
  seqLen: number,
  numHeads: number,
  headDim: number,
  startPos: number,
  table: DoryRopeTable,
): void {
  if (seqLen < 0 || !Number.isInteger(seqLen)) {
    throw new RangeError(`Invalid RoPE seqLen ${String(seqLen)}: expected integer >= 0.`);
  }
  if (seqLen === 0) {
    return;
  }
  assertValidRopeCall(headDim, startPos, table);
  assertValidRopeCall(headDim, startPos + seqLen - 1, table);

  const halfDim = table.halfDim;
  const { cos, sin } = table;

  for (let t = 0; t < seqLen; t++) {
    const pos = startPos + t;
    const tableBase = pos * halfDim;
    for (let h = 0; h < numHeads; h++) {
      const vecBase = (t * numHeads + h) * headDim;
      for (let j = 0; j < halfDim; j++) {
        const x0 = src[vecBase + j];
        const x1 = src[vecBase + j + halfDim];
        const c = cos[tableBase + j];
        const s = sin[tableBase + j];
        dst[vecBase + j] = x0 * c - x1 * s;
        dst[vecBase + j + halfDim] = x1 * c + x0 * s;
      }
    }
  }
}

export function applyNeoXRotateHalf(
  input: Float32Array,
  seqLen: number,
  numHeads: number,
  headDim: number,
  startPos: number,
  table: DoryRopeTable,
): Float32Array {
  const dst = new Float32Array(seqLen * numHeads * headDim);
  applyNeoXRotateHalfInto(input, dst, seqLen, numHeads, headDim, startPos, table);
  return dst;
}

export function applyNeoXRotateHalfAtPositions(
  input: Float32Array,
  positionIds: ArrayLike<number>,
  numHeads: number,
  headDim: number,
  table: DoryRopeTable,
): Float32Array {
  const seqLen = positionIds.length;
  const dst = new Float32Array(seqLen * numHeads * headDim);
  if (seqLen === 0) {
    return dst;
  }

  const halfDim = table.halfDim;
  const { cos, sin } = table;

  for (let t = 0; t < seqLen; t++) {
    const pos = positionIds[t];
    assertValidRopeCall(headDim, pos, table);
    const tableBase = pos * halfDim;
    for (let h = 0; h < numHeads; h++) {
      const vecBase = (t * numHeads + h) * headDim;
      for (let j = 0; j < halfDim; j++) {
        const x0 = input[vecBase + j];
        const x1 = input[vecBase + j + halfDim];
        const c = cos[tableBase + j];
        const s = sin[tableBase + j];
        dst[vecBase + j] = x0 * c - x1 * s;
        dst[vecBase + j + halfDim] = x1 * c + x0 * s;
      }
    }
  }

  return dst;
}
