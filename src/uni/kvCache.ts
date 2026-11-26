/**
 * Immutable Per-Position KV Slabs & Elastic-MoT Waste Telemetry (`src/uni/kvCache.ts`)
 *
 * Clean-room headless TypeScript KV-cache layer for Nemotron-Labs-Diffusion-Uni:
 *   - Supports CFG batch sizes `B_cfg ∈ {1, 2, 3}` (Understanding/Grounding = 1,
 *     T2I = 2, Image Editing = 3).
 *   - Strictly immutable per-position `UniKvPositionSlab` entries across all 34 layers
 *     (`L0..17` joint, `L18..33` split-tower which may be `null` under `elastic_pruned` T2I).
 *   - Appending or cropping returns a new `UniKvCacheState` reusing existing immutable
 *     slab references with zero duplication for history branching / rewind.
 *   - Exact D5 waste telemetry quantifying `L18..33` Understanding prefix KV bytes
 *     (`16 / 34` wasted fraction under `hf_faithful` T2I vs `0` under `elastic_pruned`
 *     or Understanding).
 *
 * Pure headless TypeScript — zero DOM, WebGL, `Math.random()`, `src/twotower/`,
 * `src/diffusion/`, or `src/ui/controls.ts` imports.
 */

import {
  UNI_MINI_CONFIG,
  UNI_ROUTE_G,
  UNI_ROUTE_U,
  type UniCfgBatchSize,
  type UniElasticPrefillMode,
  type UniKvCacheState,
  type UniKvPositionSlab,
  type UniKvWasteTelemetry,
  type UniLayerBatchKv,
  type UniLayerKvSlice,
} from './types';

const FLOAT32_BYTES = 4;
const KV_PAIR_COUNT = 2; // K and V

function assertValidBatchSize(batchSize: number): asserts batchSize is UniCfgBatchSize {
  if (batchSize !== 1 && batchSize !== 2 && batchSize !== 3) {
    throw new RangeError(`batchSize must be 1, 2, or 3, got ${String(batchSize)}`);
  }
}

function assertValidCapacity(capacity: number): void {
  if (!Number.isInteger(capacity) || capacity < 1) {
    throw new RangeError(`capacity must be a positive integer, got ${String(capacity)}`);
  }
}

function assertValidRoutingValue(value: number, label: string): void {
  if (value !== UNI_ROUTE_U && value !== UNI_ROUTE_G) {
    throw new RangeError(`${label} must be 0 (U) or 1 (G), got ${String(value)}`);
  }
}

/**
 * Compute D5 KV allocation & split-tower waste telemetry over immutable position slabs.
 *
 * Exact byte accounting:
 *   - Allocated bytes = sum of each slab's `byteLength`
 *     (`34 * 2 * B_cfg * 16 * 4 = 4,352 * B_cfg B/tok` for 34-layer slabs;
 *      `18 / 34` of that when `L18..33` are `null`).
 *   - When `downstreamTarget === 'g' && prefillMode === 'hf_faithful'`:
 *     for each slab with any `U` row (`routing[b] === 0`), every allocated split-tower
 *     layer `L18..33` on those `U` rows is causally unreachable by downstream `G` queries
 *     (`16 * 2 * 16 * 4 = 2,048 B` per wasted `U` row per position, yielding exact
 *     `16 / 34` wasted fraction for `u_only` T2I across all `B_cfg`).
 *   - When `downstreamTarget === 'u'` or `prefillMode === 'elastic_pruned'`:
 *     `wastedSplitUBytes === 0`, `wastedSplitULayerCount === 0`, `wastedFraction === 0`.
 */
export function computeUniKvWasteTelemetry(
  slabs: readonly UniKvPositionSlab[],
  batchSize: UniCfgBatchSize,
  prefillMode: UniElasticPrefillMode,
  downstreamTarget: 'u' | 'g',
): UniKvWasteTelemetry {
  assertValidBatchSize(batchSize);

  const dualTowerLayers = UNI_MINI_CONFIG.dualTowerLayers;
  const numLayers = UNI_MINI_CONFIG.numLayers;
  const bytesPerLayerPerRow = KV_PAIR_COUNT * UNI_MINI_CONFIG.kvProjOutDim * FLOAT32_BYTES; // 128 B

  let allocatedBytes = 0;
  for (let i = 0; i < slabs.length; i++) {
    allocatedBytes += slabs[i].byteLength;
  }

  let wastedSplitUBytes = 0;
  let wastedSplitULayerCount = 0;

  if (downstreamTarget === 'g' && prefillMode === 'hf_faithful' && slabs.length > 0) {
    const wastedLayerSeen = new Uint8Array(numLayers);
    for (let i = 0; i < slabs.length; i++) {
      const slab = slabs[i];
      let uRowCount = 0;
      for (let b = 0; b < slab.batchSize; b++) {
        if (slab.routing[b] === UNI_ROUTE_U) {
          uRowCount += 1;
        }
      }
      if (uRowCount === 0) continue;

      let slabWastedSplitLayers = 0;
      for (let l = dualTowerLayers; l < numLayers; l++) {
        if (slab.layers[l] !== null) {
          slabWastedSplitLayers += 1;
          wastedLayerSeen[l] = 1;
        }
      }
      wastedSplitUBytes += slabWastedSplitLayers * uRowCount * bytesPerLayerPerRow;
    }

    for (let l = dualTowerLayers; l < numLayers; l++) {
      wastedSplitULayerCount += wastedLayerSeen[l];
    }
  }

  const activeBytes = allocatedBytes - wastedSplitUBytes;
  const wastedFraction = allocatedBytes > 0 ? wastedSplitUBytes / allocatedBytes : 0;

  return Object.freeze({
    prefillMode,
    downstreamTarget,
    batchSize,
    seqLen: slabs.length,
    allocatedBytes,
    activeBytes,
    wastedSplitUBytes,
    wastedSplitULayerCount,
    wastedFraction,
  });
}

export interface CreateEmptyUniKvCacheOptions {
  readonly batchSize: UniCfgBatchSize;
  readonly capacity?: number;
  readonly prefillMode?: UniElasticPrefillMode;
  readonly downstreamTarget?: 'u' | 'g';
}

/**
 * Create an empty immutable `UniKvCacheState` with validated CFG batch size and capacity.
 */
export function createEmptyUniKvCache(
  options: CreateEmptyUniKvCacheOptions,
): UniKvCacheState {
  assertValidBatchSize(options.batchSize);
  const capacity = options.capacity ?? UNI_MINI_CONFIG.kvCapacity;
  assertValidCapacity(capacity);

  const prefillMode: UniElasticPrefillMode = options.prefillMode ?? 'hf_faithful';
  const downstreamTarget: 'u' | 'g' = options.downstreamTarget ?? 'u';
  const slabs: readonly UniKvPositionSlab[] = Object.freeze([]);
  const wasteTelemetry = computeUniKvWasteTelemetry(
    slabs,
    options.batchSize,
    prefillMode,
    downstreamTarget,
  );

  return Object.freeze({
    batchSize: options.batchSize,
    capacity,
    seqLen: 0,
    slabs,
    dualTowerHasGeneration: false,
    prefillMode,
    downstreamTarget,
    wasteTelemetry,
  });
}

/**
 * Append new per-row K/V slices and routing to a `UniKvCacheState` without mutating
 * the input cache or any existing `UniKvPositionSlab`.
 *
 * Existing slab references are shared directly so history branching incurs zero copying
 * of prior sequence positions. Throws `RangeError` if `cache.seqLen + appendLen > cache.capacity`.
 */
export function appendBatchKvsToCache(
  cache: UniKvCacheState,
  perRowNewKvs: readonly (readonly (UniLayerKvSlice | null)[])[],
  perRowRouting: readonly Uint8Array[],
): UniKvCacheState {
  const { batchSize, capacity, seqLen } = cache;
  const numLayers = UNI_MINI_CONFIG.numLayers;
  const kvDim = UNI_MINI_CONFIG.kvProjOutDim;

  if (perRowNewKvs.length !== batchSize) {
    throw new RangeError(
      `perRowNewKvs length (${perRowNewKvs.length}) must equal cache.batchSize (${batchSize})`,
    );
  }
  if (perRowRouting.length !== batchSize) {
    throw new RangeError(
      `perRowRouting length (${perRowRouting.length}) must equal cache.batchSize (${batchSize})`,
    );
  }

  const appendLen = perRowRouting[0].length;
  for (let b = 0; b < batchSize; b++) {
    if (perRowRouting[b].length !== appendLen) {
      throw new RangeError(
        `perRowRouting[${b}] length (${perRowRouting[b].length}) does not match row 0 length (${appendLen})`,
      );
    }
    if (perRowNewKvs[b].length !== numLayers) {
      throw new RangeError(
        `perRowNewKvs[${b}] length (${perRowNewKvs[b].length}) must equal numLayers (${numLayers})`,
      );
    }
    for (let t = 0; t < appendLen; t++) {
      assertValidRoutingValue(perRowRouting[b][t], `perRowRouting[${b}][${t}]`);
    }
  }

  if (appendLen === 0) {
    return cache;
  }

  if (seqLen + appendLen > capacity) {
    throw new RangeError(
      `KV cache capacity exceeded: seqLen (${seqLen}) + appendLen (${appendLen}) > capacity (${capacity})`,
    );
  }

  // Track whether each CFG row contains any G token in the appended window.
  // Under `elastic_pruned` 3-branch Editing (B_cfg = 3), rows 0 & 1 (`uncond`, `vq_dropped`)
  // are U-only (`newKvs[18..33] === null`) while row 2 (`full`) is mixed (`newKvs[18..33] !== null`).
  const dualTowerLayers = UNI_MINI_CONFIG.dualTowerLayers;
  const rowHasGen = new Uint8Array(batchSize);
  for (let b = 0; b < batchSize; b++) {
    for (let t = 0; t < appendLen; t++) {
      if (perRowRouting[b][t] === UNI_ROUTE_G) {
        rowHasGen[b] = 1;
        break;
      }
    }
  }

  // Validate per-layer nullability rules and buffer sizes across all CFG rows:
  //   - Joint layers (l < 18): null is strictly forbidden on every row.
  //   - Split layers (l >= 18): null is permitted iff that row has zero G tokens;
  //     if all rows are null, store `layers[l] = null`; if some rows are non-null,
  //     allocate `layers[l]` leaving null U-only rows zero-filled (skipped at 0 ULP).
  const layerAllocated = new Uint8Array(numLayers);
  for (let l = 0; l < numLayers; l++) {
    const isSplit = l >= dualTowerLayers;
    let anyNonNull = false;
    for (let b = 0; b < batchSize; b++) {
      const slice = perRowNewKvs[b][l];
      if (slice === null) {
        if (!isSplit) {
          throw new RangeError(
            `Null KV slice is forbidden on joint layer ${l} (CFG row ${b})`,
          );
        }
        if (rowHasGen[b] !== 0) {
          throw new RangeError(
            `Null KV slice on split layer ${l} is forbidden for CFG row ${b} containing G tokens`,
          );
        }
        continue;
      }
      anyNonNull = true;
      if (slice.seqLen !== appendLen) {
        throw new RangeError(
          `perRowNewKvs[${b}][${l}].seqLen (${slice.seqLen}) does not match appendLen (${appendLen})`,
        );
      }
      const reqFloats = appendLen * kvDim;
      if (slice.k.length < reqFloats || slice.v.length < reqFloats) {
        throw new RangeError(
          `perRowNewKvs[${b}][${l}] buffer too small for appendLen=${appendLen} * kvDim=${kvDim}`,
        );
      }
    }
    layerAllocated[l] = anyNonNull ? 1 : 0;
  }

  const newSlabs: UniKvPositionSlab[] = new Array<UniKvPositionSlab>(appendLen);
  let newHasGen = false;

  for (let t = 0; t < appendLen; t++) {
    const routing = new Uint8Array(batchSize);
    let hasGeneration = false;
    for (let b = 0; b < batchSize; b++) {
      const r = perRowRouting[b][t];
      routing[b] = r;
      if (r === UNI_ROUTE_G) {
        hasGeneration = true;
      }
    }
    if (hasGeneration) {
      newHasGen = true;
    }

    const layers: (UniLayerBatchKv | null)[] = new Array<UniLayerBatchKv | null>(numLayers).fill(
      null,
    );
    let allocatedLayerCount = 0;

    for (let l = 0; l < numLayers; l++) {
      if (layerAllocated[l] === 0) continue;

      const k = new Float32Array(batchSize * kvDim);
      const v = new Float32Array(batchSize * kvDim);
      const srcOffset = t * kvDim;

      for (let b = 0; b < batchSize; b++) {
        const slice = perRowNewKvs[b][l];
        if (slice === null) continue; // Null U-only split row remains zero-filled (0-ULP skipped)
        const dstOffset = b * kvDim;
        for (let c = 0; c < kvDim; c++) {
          k[dstOffset + c] = slice.k[srcOffset + c];
          v[dstOffset + c] = slice.v[srcOffset + c];
        }
      }

      layers[l] = Object.freeze({
        batchSize,
        k,
        v,
      });
      allocatedLayerCount += 1;
    }

    const byteLength = allocatedLayerCount * KV_PAIR_COUNT * batchSize * kvDim * FLOAT32_BYTES;

    newSlabs[t] = Object.freeze({
      positionIndex: seqLen + t,
      batchSize,
      routing,
      hasGeneration,
      prefillMode: cache.prefillMode,
      layers: Object.freeze(layers),
      allocatedLayerCount,
      byteLength,
    });
  }

  const combinedSlabs: readonly UniKvPositionSlab[] = Object.freeze([
    ...cache.slabs,
    ...newSlabs,
  ]);
  const dualTowerHasGeneration = cache.dualTowerHasGeneration || newHasGen;
  const wasteTelemetry = computeUniKvWasteTelemetry(
    combinedSlabs,
    batchSize,
    cache.prefillMode,
    cache.downstreamTarget,
  );

  return Object.freeze({
    batchSize,
    capacity,
    seqLen: combinedSlabs.length,
    slabs: combinedSlabs,
    dualTowerHasGeneration,
    prefillMode: cache.prefillMode,
    downstreamTarget: cache.downstreamTarget,
    wasteTelemetry,
  });
}

/**
 * Convenience wrapper over `appendBatchKvsToCache` for single-row caches (`batchSize === 1`).
 */
export function appendSingleRowKvsToCache(
  cache: UniKvCacheState,
  newKvs: readonly (UniLayerKvSlice | null)[],
  routing: Uint8Array,
): UniKvCacheState {
  if (cache.batchSize !== 1) {
    throw new RangeError(
      `appendSingleRowKvsToCache requires cache.batchSize === 1, got ${cache.batchSize}`,
    );
  }
  return appendBatchKvsToCache(cache, [newKvs], [routing]);
}

/**
 * Materialize contiguous `[seqLen, 16]` per-layer K/V slices and `[seqLen]` routing
 * for a single CFG batch row so they can be passed directly into `runUniForwardPass`.
 *
 * Returns `null` for any layer whose slabs are `null` (e.g. `L18..33` under `elastic_pruned` T2I).
 */
export function extractRowPastKvs(
  cache: UniKvCacheState,
  batchRow: number,
): { pastKvs: (UniLayerKvSlice | null)[]; pastRouting: Uint8Array } {
  if (!Number.isInteger(batchRow) || batchRow < 0 || batchRow >= cache.batchSize) {
    throw new RangeError(
      `batchRow (${String(batchRow)}) out of range for batchSize=${cache.batchSize}`,
    );
  }

  const numLayers = UNI_MINI_CONFIG.numLayers;
  const kvDim = UNI_MINI_CONFIG.kvProjOutDim;
  const seqLen = cache.seqLen;

  const pastRouting = new Uint8Array(seqLen);
  for (let t = 0; t < seqLen; t++) {
    pastRouting[t] = cache.slabs[t].routing[batchRow];
  }

  const pastKvs: (UniLayerKvSlice | null)[] = new Array<UniLayerKvSlice | null>(numLayers).fill(
    null,
  );
  if (seqLen === 0) {
    return { pastKvs, pastRouting };
  }

  const rowOffset = batchRow * kvDim;

  for (let l = 0; l < numLayers; l++) {
    let anyAllocated = false;
    for (let t = 0; t < seqLen; t++) {
      if (cache.slabs[t].layers[l] !== null) {
        anyAllocated = true;
        break;
      }
    }
    if (!anyAllocated) {
      pastKvs[l] = null;
      continue;
    }

    const k = new Float32Array(seqLen * kvDim);
    const v = new Float32Array(seqLen * kvDim);

    for (let t = 0; t < seqLen; t++) {
      const layerKv = cache.slabs[t].layers[l];
      if (layerKv === null) {
        if (cache.slabs[t].routing[batchRow] === UNI_ROUTE_G) {
          throw new RangeError(
            `Missing KV slice at layer ${l}, pos ${t}, batchRow ${batchRow} for G token`,
          );
        }
        // Pruned U position in split layer: masked out by split-tower routing, left zeroed.
        continue;
      }
      const dstOffset = t * kvDim;
      for (let c = 0; c < kvDim; c++) {
        k[dstOffset + c] = layerKv.k[rowOffset + c];
        v[dstOffset + c] = layerKv.v[rowOffset + c];
      }
    }

    pastKvs[l] = { k, v, seqLen };
  }

  return { pastKvs, pastRouting };
}

/**
 * Non-mutating crop of a `UniKvCacheState` to `[0 .. newSeqLen - 1]`, sharing existing
 * immutable slab references and recomputing `dualTowerHasGeneration` and `wasteTelemetry`.
 */
export function cropUniKvCache(
  cache: UniKvCacheState,
  newSeqLen: number,
): UniKvCacheState {
  if (!Number.isInteger(newSeqLen) || newSeqLen < 0 || newSeqLen > cache.seqLen) {
    throw new RangeError(
      `newSeqLen (${String(newSeqLen)}) must be an integer in [0, ${cache.seqLen}]`,
    );
  }
  if (newSeqLen === cache.seqLen) {
    return cache;
  }

  const croppedSlabs: readonly UniKvPositionSlab[] = Object.freeze(
    cache.slabs.slice(0, newSeqLen),
  );
  let dualTowerHasGeneration = false;
  for (let i = 0; i < croppedSlabs.length; i++) {
    if (croppedSlabs[i].hasGeneration) {
      dualTowerHasGeneration = true;
      break;
    }
  }

  const wasteTelemetry = computeUniKvWasteTelemetry(
    croppedSlabs,
    cache.batchSize,
    cache.prefillMode,
    cache.downstreamTarget,
  );

  return Object.freeze({
    batchSize: cache.batchSize,
    capacity: cache.capacity,
    seqLen: croppedSlabs.length,
    slabs: croppedSlabs,
    dualTowerHasGeneration,
    prefillMode: cache.prefillMode,
    downstreamTarget: cache.downstreamTarget,
    wasteTelemetry,
  });
}
