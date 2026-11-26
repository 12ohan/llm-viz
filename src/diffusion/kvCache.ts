/**
 * Shared-Slab Persistent KV Cache with Provenance & Tamper Detection
 * (`src/diffusion/kvCache.ts`)
 *
 * Implements immutable per-position KV slabs (`20.00 KiB / tok` across 40 layers
 * × 2 tensors × 64 floats) with structural prefix sharing (`0`-ULP prefix
 * immutability across append, crop, and speculative rollback), strict
 * `kvMaskProvenance` isolation (`'token_causal'` vs `'block_bidirectional'`),
 * and deterministic `dualHashHex` integrity verification over metadata and raw
 * IEEE-754 little-endian float bytes.
 *
 * Pure headless TypeScript — deterministic, self-contained kernel.
 */

import { dualHashHex } from './prng';
import {
  DIFFUSION_MINI_CONFIG,
  type DiffusionLayerKvSlice,
  type KvMaskProvenance,
  type KvPositionSlab,
  type PersistentKvSnapshot,
} from './types';

const UTF8_ENCODER = new TextEncoder();

// ============================================================================
// ERRORS
// ============================================================================

export class KvProvenanceMismatchError extends Error {
  readonly actualProvenance?: KvMaskProvenance;
  readonly requiredProvenance?: KvMaskProvenance;

  constructor(
    message: string,
    actualProvenance?: KvMaskProvenance,
    requiredProvenance?: KvMaskProvenance,
  ) {
    super(message);
    this.name = 'KvProvenanceMismatchError';
    this.actualProvenance = actualProvenance;
    this.requiredProvenance = requiredProvenance;
    Object.setPrototypeOf(this, KvProvenanceMismatchError.prototype);
  }
}

export class KvIntegrityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'KvIntegrityError';
    Object.setPrototypeOf(this, KvIntegrityError.prototype);
  }
}

// ============================================================================
// TAMPER-DETECTING DIGEST HELPERS (IEEE-754 LITTLE-ENDIAN)
// ============================================================================

function computeSlabDigest(
  position: number,
  tokenId: number,
  maskProvenance: KvMaskProvenance,
  numLayers: number,
  kvProjOutDim: number,
  k: Float32Array,
  v: Float32Array,
): string {
  const headerBytes = UTF8_ENCODER.encode(
    `kvslab|p=${position}|t=${tokenId}|m=${maskProvenance}|L=${numLayers}|D=${kvProjOutDim}|`,
  );
  const kByteLen = k.length * 4;
  const vByteLen = v.length * 4;
  const payload = new Uint8Array(headerBytes.length + kByteLen + vByteLen);
  payload.set(headerBytes, 0);

  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  let offset = headerBytes.length;
  for (let i = 0; i < k.length; i++) {
    view.setFloat32(offset, k[i]!, true);
    offset += 4;
  }
  for (let i = 0; i < v.length; i++) {
    view.setFloat32(offset, v[i]!, true);
    offset += 4;
  }

  return dualHashHex(payload);
}

function computeSnapshotDigest(
  seqLen: number,
  capacity: number,
  kvMaskProvenance: KvMaskProvenance,
  committedTokenIds: readonly number[],
  slabs: readonly KvPositionSlab[],
): string {
  const parts: string[] = [
    `kvsnap|len=${seqLen}|cap=${capacity}|prov=${kvMaskProvenance}|toks=${committedTokenIds.join(',')}|slabs=`,
  ];
  for (let i = 0; i < slabs.length; i++) {
    const slab = slabs[i]!;
    if (i > 0) parts.push(';');
    parts.push(`${slab.position}:${slab.tokenId}:${slab.maskProvenance}:${slab.slabDigest}`);
  }
  return dualHashHex(parts.join(''));
}

// ============================================================================
// SLAB & SNAPSHOT OPERATIONS
// ============================================================================

/**
 * Extracts a single token position `[tokenOffsetInSlice * kvProjOutDim .. (tokenOffsetInSlice+1) * kvProjOutDim)`
 * from every layer in `newKvs` into contiguous `k` and `v` Float32Arrays
 * (`2560` + `2560` floats = `20.00 KiB` for 40 layers × 64 dims) with a
 * tamper-detecting `dualHashHex` digest.
 */
export function createKvPositionSlab(
  position: number,
  tokenId: number,
  maskProvenance: KvMaskProvenance,
  newKvs: readonly DiffusionLayerKvSlice[],
  tokenOffsetInSlice: number,
): KvPositionSlab {
  if (!Number.isInteger(position) || position < 0) {
    throw new RangeError(`position must be a non-negative integer, got ${position}`);
  }
  if (!Number.isInteger(tokenId) || tokenId < 0) {
    throw new RangeError(`tokenId must be a non-negative integer, got ${tokenId}`);
  }
  const numLayers = newKvs.length;
  if (numLayers <= 0) {
    throw new RangeError('newKvs must contain at least one layer slice');
  }

  const firstSlice = newKvs[0]!;
  if (!Number.isInteger(firstSlice.seqLen) || firstSlice.seqLen <= 0) {
    throw new RangeError(`newKvs[0].seqLen must be a positive integer, got ${firstSlice.seqLen}`);
  }
  if (
    !Number.isInteger(tokenOffsetInSlice) ||
    tokenOffsetInSlice < 0 ||
    tokenOffsetInSlice >= firstSlice.seqLen
  ) {
    throw new RangeError(
      `tokenOffsetInSlice (${tokenOffsetInSlice}) out of bounds [0, ${firstSlice.seqLen - 1}]`,
    );
  }

  const kvProjOutDim = Math.trunc(firstSlice.k.length / firstSlice.seqLen);
  if (kvProjOutDim <= 0 || kvProjOutDim * firstSlice.seqLen !== firstSlice.k.length) {
    throw new RangeError(
      `newKvs[0].k length (${firstSlice.k.length}) is not a positive multiple of seqLen (${firstSlice.seqLen})`,
    );
  }

  const totalFloats = numLayers * kvProjOutDim;
  const k = new Float32Array(totalFloats);
  const v = new Float32Array(totalFloats);
  const srcStart = tokenOffsetInSlice * kvProjOutDim;
  const srcEnd = srcStart + kvProjOutDim;

  for (let l = 0; l < numLayers; l++) {
    const slice = newKvs[l]!;
    if (slice.seqLen !== firstSlice.seqLen) {
      throw new RangeError(
        `Inconsistent newKvs seqLen at layer ${l}: expected ${firstSlice.seqLen}, got ${slice.seqLen}`,
      );
    }
    if (slice.k.length < srcEnd || slice.v.length < srcEnd) {
      throw new RangeError(
        `Layer ${l} KV slice buffers too small for tokenOffsetInSlice=${tokenOffsetInSlice} * kvProjOutDim=${kvProjOutDim}`,
      );
    }
    const dstOffset = l * kvProjOutDim;
    k.set(slice.k.subarray(srcStart, srcEnd), dstOffset);
    v.set(slice.v.subarray(srcStart, srcEnd), dstOffset);
  }

  const slabDigest = computeSlabDigest(
    position,
    tokenId,
    maskProvenance,
    numLayers,
    kvProjOutDim,
    k,
    v,
  );

  return Object.freeze({
    position,
    tokenId,
    maskProvenance,
    numLayers,
    kvProjOutDim,
    k,
    v,
    slabDigest,
  });
}

/** Creates an empty persistent KV snapshot tagged with immutable `kvMaskProvenance`. */
export function createEmptyKvSnapshot(
  kvMaskProvenance: KvMaskProvenance,
  capacity: number = DIFFUSION_MINI_CONFIG.kvCapacity,
): PersistentKvSnapshot {
  if (!Number.isInteger(capacity) || capacity < 0) {
    throw new RangeError(`capacity must be a non-negative integer, got ${capacity}`);
  }
  const committedTokenIds: readonly number[] = Object.freeze([]);
  const slabs: readonly KvPositionSlab[] = Object.freeze([]);
  const snapshotDigest = computeSnapshotDigest(
    0,
    capacity,
    kvMaskProvenance,
    committedTokenIds,
    slabs,
  );
  return Object.freeze({
    seqLen: 0,
    capacity,
    kvMaskProvenance,
    committedTokenIds,
    slabs,
    snapshotDigest,
  });
}

/**
 * Structurally appends newly computed per-layer KV slices onto `snapshot`,
 * preserving exact object and `ArrayBuffer` identity for all existing prefix slabs.
 * Throws `KvProvenanceMismatchError` if `snapshot.kvMaskProvenance !== appendProvenance`.
 */
export function appendNewKvsToSnapshot(
  snapshot: PersistentKvSnapshot,
  newKvs: readonly DiffusionLayerKvSlice[],
  newTokenIds: ArrayLike<number>,
  appendProvenance: KvMaskProvenance,
): PersistentKvSnapshot {
  if (snapshot.kvMaskProvenance !== appendProvenance) {
    throw new KvProvenanceMismatchError(
      `Cannot append '${appendProvenance}' KVs to snapshot with provenance '${snapshot.kvMaskProvenance}'`,
      snapshot.kvMaskProvenance,
      appendProvenance,
    );
  }

  const appendLen = newTokenIds.length;
  if (appendLen === 0) {
    return snapshot;
  }

  const nextSeqLen = snapshot.seqLen + appendLen;
  if (nextSeqLen > snapshot.capacity) {
    throw new RangeError(
      `KV capacity overflow: seqLen (${snapshot.seqLen}) + appendLen (${appendLen}) > capacity (${snapshot.capacity})`,
    );
  }
  if (newKvs.length === 0) {
    throw new RangeError('newKvs must contain layer slices when appending non-empty tokens');
  }
  if (newKvs[0]!.seqLen !== appendLen) {
    throw new RangeError(
      `newKvs[0].seqLen (${newKvs[0]!.seqLen}) must match newTokenIds.length (${appendLen})`,
    );
  }

  // Structurally share all existing prefix slabs (0-ULP prefix immutability)
  const nextSlabs: KvPositionSlab[] = snapshot.slabs.slice();
  const nextTokenIds: number[] = snapshot.committedTokenIds.slice();

  for (let i = 0; i < appendLen; i++) {
    const pos = snapshot.seqLen + i;
    const tokenId = newTokenIds[i]!;
    const slab = createKvPositionSlab(pos, tokenId, appendProvenance, newKvs, i);
    nextSlabs.push(slab);
    nextTokenIds.push(tokenId);
  }

  const frozenSlabs = Object.freeze(nextSlabs);
  const frozenTokenIds = Object.freeze(nextTokenIds);
  const snapshotDigest = computeSnapshotDigest(
    nextSeqLen,
    snapshot.capacity,
    snapshot.kvMaskProvenance,
    frozenTokenIds,
    frozenSlabs,
  );

  return Object.freeze({
    seqLen: nextSeqLen,
    capacity: snapshot.capacity,
    kvMaskProvenance: snapshot.kvMaskProvenance,
    committedTokenIds: frozenTokenIds,
    slabs: frozenSlabs,
    snapshotDigest,
  });
}

/**
 * Structurally crops `snapshot` to `targetLen` (`0 <= targetLen <= snapshot.seqLen`),
 * reusing prefix `KvPositionSlab` references verbatim (`0`-ULP prefix immutability).
 */
export function cropKvSnapshot(
  snapshot: PersistentKvSnapshot,
  targetLen: number,
): PersistentKvSnapshot {
  if (!Number.isInteger(targetLen) || targetLen < 0 || targetLen > snapshot.seqLen) {
    throw new RangeError(
      `targetLen (${targetLen}) out of bounds [0, ${snapshot.seqLen}]`,
    );
  }
  if (targetLen === snapshot.seqLen) {
    return snapshot;
  }

  const croppedSlabs = Object.freeze(snapshot.slabs.slice(0, targetLen));
  const croppedTokenIds = Object.freeze(snapshot.committedTokenIds.slice(0, targetLen));
  const snapshotDigest = computeSnapshotDigest(
    targetLen,
    snapshot.capacity,
    snapshot.kvMaskProvenance,
    croppedTokenIds,
    croppedSlabs,
  );

  return Object.freeze({
    seqLen: targetLen,
    capacity: snapshot.capacity,
    kvMaskProvenance: snapshot.kvMaskProvenance,
    committedTokenIds: croppedTokenIds,
    slabs: croppedSlabs,
    snapshotDigest,
  });
}

/**
 * Returns the cropped/rejected suffix slabs (`snapshotBeforeCrop.slabs.slice(croppedLen)`).
 */
export function extractCroppedSuffixSlabs(
  snapshotBeforeCrop: PersistentKvSnapshot,
  croppedLen: number,
): readonly KvPositionSlab[] {
  if (
    !Number.isInteger(croppedLen) ||
    croppedLen < 0 ||
    croppedLen > snapshotBeforeCrop.seqLen
  ) {
    throw new RangeError(
      `croppedLen (${croppedLen}) out of bounds [0, ${snapshotBeforeCrop.seqLen}]`,
    );
  }
  return Object.freeze(snapshotBeforeCrop.slabs.slice(croppedLen));
}

/**
 * Materializes all per-layer `[seqLen, kvProjOutDim]` KV slices from `snapshot`.
 * Returns `null` when `snapshot.seqLen === 0`.
 */
export function materializePastKvs(
  snapshot: PersistentKvSnapshot,
  numLayers: number = DIFFUSION_MINI_CONFIG.numLayers,
  kvProjOutDim: number = DIFFUSION_MINI_CONFIG.kvProjOutDim,
): readonly DiffusionLayerKvSlice[] | null {
  const seqLen = snapshot.seqLen;
  if (seqLen === 0) {
    return null;
  }

  const firstSlab = snapshot.slabs[0]!;
  if (firstSlab.numLayers !== numLayers || firstSlab.kvProjOutDim !== kvProjOutDim) {
    throw new RangeError(
      `Snapshot slab geometry (${firstSlab.numLayers}x${firstSlab.kvProjOutDim}) does not match requested (${numLayers}x${kvProjOutDim})`,
    );
  }

  const slices: DiffusionLayerKvSlice[] = new Array<DiffusionLayerKvSlice>(numLayers);
  for (let l = 0; l < numLayers; l++) {
    const k = new Float32Array(seqLen * kvProjOutDim);
    const v = new Float32Array(seqLen * kvProjOutDim);
    const srcStart = l * kvProjOutDim;
    const srcEnd = srcStart + kvProjOutDim;

    for (let t = 0; t < seqLen; t++) {
      const slab = snapshot.slabs[t]!;
      const dstOffset = t * kvProjOutDim;
      k.set(slab.k.subarray(srcStart, srcEnd), dstOffset);
      v.set(slab.v.subarray(srcStart, srcEnd), dstOffset);
    }

    slices[l] = { k, v, seqLen };
  }

  return Object.freeze(slices);
}

/**
 * Materializes a single layer's `[seqLen, kvProjOutDim]` KV slice from `snapshot`.
 * Returns `null` when `snapshot.seqLen === 0`.
 */
export function materializeLayerPastKv(
  snapshot: PersistentKvSnapshot,
  layerIndex: number,
): DiffusionLayerKvSlice | null {
  const seqLen = snapshot.seqLen;
  if (seqLen === 0) {
    return null;
  }

  const firstSlab = snapshot.slabs[0]!;
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= firstSlab.numLayers) {
    throw new RangeError(
      `layerIndex (${layerIndex}) out of bounds [0, ${firstSlab.numLayers - 1}]`,
    );
  }

  const kvProjOutDim = firstSlab.kvProjOutDim;
  const k = new Float32Array(seqLen * kvProjOutDim);
  const v = new Float32Array(seqLen * kvProjOutDim);
  const srcStart = layerIndex * kvProjOutDim;
  const srcEnd = srcStart + kvProjOutDim;

  for (let t = 0; t < seqLen; t++) {
    const slab = snapshot.slabs[t]!;
    const dstOffset = t * kvProjOutDim;
    k.set(slab.k.subarray(srcStart, srcEnd), dstOffset);
    v.set(slab.v.subarray(srcStart, srcEnd), dstOffset);
  }

  return { k, v, seqLen };
}

/**
 * Guards against silently reusing a `'block_bidirectional'` KV snapshot in a
 * `'token_causal'` path (or vice versa) whenever `snapshot.seqLen > 0`.
 */
export function assertKvProvenanceCompatible(
  snapshot: PersistentKvSnapshot,
  requiredProvenance: KvMaskProvenance,
  contextLabel: string,
): void {
  if (snapshot.seqLen > 0 && snapshot.kvMaskProvenance !== requiredProvenance) {
    throw new KvProvenanceMismatchError(
      `[${contextLabel}] KV provenance mismatch: snapshot holds ${snapshot.seqLen} '${snapshot.kvMaskProvenance}' token(s), but '${requiredProvenance}' is required`,
      snapshot.kvMaskProvenance,
      requiredProvenance,
    );
  }
}

/**
 * Recomputes every `slabDigest` and the top-level `snapshotDigest` to detect
 * any in-place mutation of `k`/`v` floats, `tokenId`, `position`, or `kvMaskProvenance`.
 */
export function verifyKvSnapshotIntegrity(
  snapshot: PersistentKvSnapshot,
): { readonly valid: boolean; readonly reason: string | null } {
  if (!Number.isInteger(snapshot.seqLen) || snapshot.seqLen < 0) {
    return { valid: false, reason: `Invalid snapshot seqLen: ${snapshot.seqLen}` };
  }
  if (!Number.isInteger(snapshot.capacity) || snapshot.seqLen > snapshot.capacity) {
    return {
      valid: false,
      reason: `Snapshot seqLen (${snapshot.seqLen}) exceeds capacity (${snapshot.capacity})`,
    };
  }
  if (snapshot.slabs.length !== snapshot.seqLen) {
    return {
      valid: false,
      reason: `Snapshot slabs.length (${snapshot.slabs.length}) !== seqLen (${snapshot.seqLen})`,
    };
  }
  if (snapshot.committedTokenIds.length !== snapshot.seqLen) {
    return {
      valid: false,
      reason: `Snapshot committedTokenIds.length (${snapshot.committedTokenIds.length}) !== seqLen (${snapshot.seqLen})`,
    };
  }

  for (let i = 0; i < snapshot.seqLen; i++) {
    const slab = snapshot.slabs[i]!;
    if (slab.position !== i) {
      return {
        valid: false,
        reason: `Slab ${i} position mismatch: expected ${i}, got ${slab.position}`,
      };
    }
    if (slab.tokenId !== snapshot.committedTokenIds[i]) {
      return {
        valid: false,
        reason: `Slab ${i} tokenId (${slab.tokenId}) !== committedTokenIds[${i}] (${snapshot.committedTokenIds[i]})`,
      };
    }
    if (slab.maskProvenance !== snapshot.kvMaskProvenance) {
      return {
        valid: false,
        reason: `Slab ${i} maskProvenance ('${slab.maskProvenance}') !== snapshot kvMaskProvenance ('${snapshot.kvMaskProvenance}')`,
      };
    }
    const expectedLen = slab.numLayers * slab.kvProjOutDim;
    if (slab.k.length !== expectedLen || slab.v.length !== expectedLen) {
      return {
        valid: false,
        reason: `Slab ${i} KV length mismatch: expected ${expectedLen}, got k=${slab.k.length}, v=${slab.v.length}`,
      };
    }
    const recomputedSlabDigest = computeSlabDigest(
      slab.position,
      slab.tokenId,
      slab.maskProvenance,
      slab.numLayers,
      slab.kvProjOutDim,
      slab.k,
      slab.v,
    );
    if (recomputedSlabDigest !== slab.slabDigest) {
      return {
        valid: false,
        reason: `Slab ${i} digest mismatch (tampered k/v or metadata): expected ${slab.slabDigest}, got ${recomputedSlabDigest}`,
      };
    }
  }

  const recomputedSnapshotDigest = computeSnapshotDigest(
    snapshot.seqLen,
    snapshot.capacity,
    snapshot.kvMaskProvenance,
    snapshot.committedTokenIds,
    snapshot.slabs,
  );
  if (recomputedSnapshotDigest !== snapshot.snapshotDigest) {
    return {
      valid: false,
      reason: `Snapshot digest mismatch: expected ${snapshot.snapshotDigest}, got ${recomputedSnapshotDigest}`,
    };
  }

  return { valid: true, reason: null };
}

/** Throws `KvIntegrityError` if `verifyKvSnapshotIntegrity(snapshot)` fails. */
export function assertKvSnapshotIntegrity(snapshot: PersistentKvSnapshot): void {
  const result = verifyKvSnapshotIntegrity(snapshot);
  if (!result.valid) {
    throw new KvIntegrityError(result.reason ?? 'KV snapshot integrity verification failed');
  }
}
