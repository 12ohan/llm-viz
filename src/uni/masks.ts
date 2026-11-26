/**
 * Elastic-MoT Routing Canonicalization, Split-Tower & SBD/OPD Attention Masks
 * (`src/uni/masks.ts`)
 *
 * Clean-room first-principles TypeScript implementation matching NLD-Uni:
 *   - Token-level U/G routing canonicalization (`0` = U, `1` = G)
 *   - Causal (`kCol <= pastLen + qRow`) and Bidirectional masks
 *   - Asymmetric SBD image block mask (`U15`: excludes trailing tokens from image queries)
 *   - Read-only OPD dual-stream `[noisy || clean]` block diffusion mask (`[2L * 2L]`)
 *   - Joint (`L0..17`) vs. Elastic Split-Tower (`L18..33`, `r_q === r_kv`) masking
 *
 * Pure headless TypeScript — zero DOM, WebGL, ambient RNG, or forbidden imports.
 */

import {
  UNI_MINI_CONFIG,
  type UniAttentionMaskMatrix,
  type UniCanonicalRouting,
  type UniMaskExplorerMode,
  type UniTokenRoutingPlan,
} from './types';

function assertIntAtLeast(value: number, min: number, name: string): void {
  if (!Number.isInteger(value) || value < min) {
    throw new RangeError(`${name} must be an integer >= ${min}, got ${value}`);
  }
}

/**
 * Validate per-token U/G routing (`0` = Understanding, `1` = Generation) and
 * partition sequence indices into `understandingIndices` and `generationIndices`.
 */
export function canonicalizeTokenRouting(
  routing: ArrayLike<number>,
): UniTokenRoutingPlan {
  const len = routing.length;
  const copy = new Uint8Array(len);
  let uCount = 0;
  let gCount = 0;

  for (let i = 0; i < len; i++) {
    const val = routing[i];
    if (val !== 0 && val !== 1) {
      throw new RangeError(
        `routing[${i}] must be 0 (U) or 1 (G), received ${String(val)}`,
      );
    }
    copy[i] = val;
    if (val === 0) {
      uCount++;
    } else {
      gCount++;
    }
  }

  const understandingIndices = new Int32Array(uCount);
  const generationIndices = new Int32Array(gCount);
  let uPtr = 0;
  let gPtr = 0;
  for (let i = 0; i < len; i++) {
    if (copy[i] === 0) {
      understandingIndices[uPtr++] = i;
    } else {
      generationIndices[gPtr++] = i;
    }
  }

  const hasUnderstanding = uCount > 0;
  const hasGeneration = gCount > 0;

  let canonical: UniCanonicalRouting;
  if (hasUnderstanding && hasGeneration) {
    canonical = 'mixed';
  } else if (hasGeneration) {
    canonical = 'g_only';
  } else {
    canonical = 'u_only';
  }

  return {
    routing: copy,
    canonical,
    understandingIndices,
    generationIndices,
    hasUnderstanding,
    hasGeneration,
  };
}

/**
 * Build row-major `[seqLen * (pastLen + seqLen)]` binary causal mask (`1` iff `kCol <= pastLen + qRow`).
 */
export function buildCausalMask(seqLen: number, pastLen = 0): Uint8Array {
  assertIntAtLeast(seqLen, 1, 'seqLen');
  assertIntAtLeast(pastLen, 0, 'pastLen');

  const kvLen = pastLen + seqLen;
  const mask = new Uint8Array(seqLen * kvLen);
  for (let qRow = 0; qRow < seqLen; qRow++) {
    const maxK = pastLen + qRow;
    const rowBase = qRow * kvLen;
    for (let kCol = 0; kCol <= maxK; kCol++) {
      mask[rowBase + kCol] = 1;
    }
  }
  return mask;
}

/**
 * Build row-major `[seqLen * (pastLen + seqLen)]` all-ones bidirectional mask.
 */
export function buildBidirectionalMask(seqLen: number, pastLen = 0): Uint8Array {
  assertIntAtLeast(seqLen, 1, 'seqLen');
  assertIntAtLeast(pastLen, 0, 'pastLen');

  const kvLen = pastLen + seqLen;
  const mask = new Uint8Array(seqLen * kvLen);
  mask.fill(1);
  return mask;
}

/**
 * Build unified `[totalLen * totalLen]` asymmetric SBD image mask (`U15`):
 *   - `q < imageBlockStart`: causal (`k <= q`)
 *   - `imageBlockStart <= q < end`: attends to prefix (`k < imageBlockStart`) AND
 *     full image block (`imageBlockStart <= k < end`), excluding trailing (`k >= end`)
 *   - `q >= end`: causal (`k <= q`)
 */
export function buildSbdImageMask(
  totalLen: number,
  imageBlockStart: number,
  imageBlockEnd?: number,
): Uint8Array {
  assertIntAtLeast(totalLen, 1, 'totalLen');
  assertIntAtLeast(imageBlockStart, 0, 'imageBlockStart');
  const end = imageBlockEnd ?? totalLen;
  assertIntAtLeast(end, imageBlockStart, 'imageBlockEnd');
  if (imageBlockStart > totalLen || end > totalLen) {
    throw new RangeError(
      `imageBlockStart (${imageBlockStart}) and imageBlockEnd (${end}) must be <= totalLen (${totalLen})`,
    );
  }

  const mask = new Uint8Array(totalLen * totalLen);
  for (let q = 0; q < totalLen; q++) {
    const rowBase = q * totalLen;
    if (q >= imageBlockStart && q < end) {
      for (let k = 0; k < end; k++) {
        mask[rowBase + k] = 1;
      }
    } else {
      for (let k = 0; k <= q; k++) {
        mask[rowBase + k] = 1;
      }
    }
  }
  return mask;
}

/**
 * Build read-only OPD dual-stream `[2L * 2L]` block diffusion mask over
 * concatenated `[noisy (0..L-1), clean (L..2L-1)]` sequence:
 *   - Noisy block `b = floor(q / blockSize)` (`q < L`) attends to same noisy block
 *     (`floor(k / blockSize) === b`, `k < L`) plus strictly earlier clean blocks
 *     (`floor((k - L) / blockSize) < b`, `k >= L`).
 *   - Clean token (`q >= L`) attends causally to clean tokens (`L <= k <= q`).
 */
export function buildDualStreamBlockDiffusionMask(
  cleanLen: number,
  blockSize: number,
): Uint8Array {
  assertIntAtLeast(cleanLen, 1, 'cleanLen');
  assertIntAtLeast(blockSize, 1, 'blockSize');

  const totalLen = 2 * cleanLen;
  const mask = new Uint8Array(totalLen * totalLen);

  for (let q = 0; q < totalLen; q++) {
    const rowBase = q * totalLen;
    if (q < cleanLen) {
      const bq = Math.floor(q / blockSize);
      for (let k = 0; k < totalLen; k++) {
        if (k < cleanLen) {
          if (Math.floor(k / blockSize) === bq) {
            mask[rowBase + k] = 1;
          }
        } else {
          const bk = Math.floor((k - cleanLen) / blockSize);
          if (bk < bq) {
            mask[rowBase + k] = 1;
          }
        }
      }
    } else {
      for (let k = cleanLen; k <= q; k++) {
        mask[rowBase + k] = 1;
      }
    }
  }

  return mask;
}

/**
 * Apply Elastic Split-Tower routing isolation on top of `baseMask`:
 *   - `layerIndex < dualTowerLayers` (`0..17`): returns a fresh copy of `baseMask`.
 *   - `layerIndex >= dualTowerLayers` (`18..33`): restricts visibility by
 *     `baseMask[q, k] & (queryRouting[q] === kvRouting[k])`.
 */
export function applyElasticSplitTowerMask(
  baseMask: Uint8Array,
  queryRouting: Uint8Array,
  kvRouting: Uint8Array,
  layerIndex: number,
  dualTowerLayers: number = UNI_MINI_CONFIG.dualTowerLayers,
): Uint8Array {
  assertIntAtLeast(layerIndex, 0, 'layerIndex');
  assertIntAtLeast(dualTowerLayers, 0, 'dualTowerLayers');

  const qLen = queryRouting.length;
  const kvLen = kvRouting.length;
  if (baseMask.length !== qLen * kvLen) {
    throw new RangeError(
      `baseMask length (${baseMask.length}) must equal queryRouting.length * kvRouting.length (${qLen * kvLen})`,
    );
  }

  if (layerIndex < dualTowerLayers) {
    const copy = new Uint8Array(baseMask.length);
    for (let i = 0; i < baseMask.length; i++) {
      copy[i] = baseMask[i] !== 0 ? 1 : 0;
    }
    return copy;
  }

  const out = new Uint8Array(qLen * kvLen);
  for (let q = 0; q < qLen; q++) {
    const qr = queryRouting[q];
    if (qr !== 0 && qr !== 1) {
      throw new RangeError(`queryRouting[${q}] must be 0 or 1, got ${qr}`);
    }
    const rowBase = q * kvLen;
    for (let k = 0; k < kvLen; k++) {
      if (baseMask[rowBase + k] !== 0 && qr === kvRouting[k]) {
        out[rowBase + k] = 1;
      }
    }
  }
  return out;
}

/**
 * Build an inspector-ready `UniAttentionMaskMatrix` across all 5 Mask Explorer modes.
 */
export function buildUniAttentionMaskMatrix(options: {
  mode: UniMaskExplorerMode;
  seqLen: number;
  pastLen?: number;
  queryRouting?: Uint8Array;
  kvRouting?: Uint8Array;
  layerIndex?: number;
  dualTowerLayers?: number;
  imageBlockStart?: number;
  imageBlockEnd?: number;
  blockSize?: number;
}): UniAttentionMaskMatrix {
  const {
    mode,
    seqLen,
    pastLen = 0,
    queryRouting,
    kvRouting,
    layerIndex,
    dualTowerLayers = UNI_MINI_CONFIG.dualTowerLayers,
    imageBlockStart,
    imageBlockEnd,
    blockSize,
  } = options;

  assertIntAtLeast(seqLen, 1, 'seqLen');
  assertIntAtLeast(pastLen, 0, 'pastLen');

  if (mode === 'opd_dual_stream') {
    const bSize = blockSize ?? Math.max(1, Math.min(4, seqLen));
    const visible = buildDualStreamBlockDiffusionMask(seqLen, bSize);
    const totalLen = 2 * seqLen;
    return {
      mode,
      qLen: totalLen,
      kvLen: totalLen,
      visible,
      splitTowerApplied: false,
      layerIndex: null,
      isReadOnlyReference: true,
      badge: 'Read-Only OPD Dual-Stream Mask (No Gradients)',
    };
  }

  let base: Uint8Array;
  let qLen = seqLen;
  let kvLen = pastLen + seqLen;
  let baseBadge = '';

  if (mode === 'causal') {
    base = buildCausalMask(seqLen, pastLen);
    baseBadge = pastLen > 0 ? 'AR Causal (Cached)' : 'AR Causal';
  } else if (mode === 'bidirectional') {
    base = buildBidirectionalMask(seqLen, pastLen);
    baseBadge = 'Bidirectional Self-Attention';
  } else if (mode === 'sbd_image') {
    const start = imageBlockStart ?? Math.floor(seqLen / 2);
    const end = imageBlockEnd ?? seqLen;
    kvLen = seqLen;
    base = buildSbdImageMask(seqLen, start, end);
    baseBadge = 'SBD Image Block (Trailing Excluded)';
  } else {
    // split_tower_mixed
    if (imageBlockStart !== undefined) {
      kvLen = seqLen;
      base = buildSbdImageMask(seqLen, imageBlockStart, imageBlockEnd ?? seqLen);
    } else {
      base = buildCausalMask(seqLen, pastLen);
    }
    baseBadge = 'Mixed U/G Routing';
  }

  const resolvedLayer =
    layerIndex ?? (mode === 'split_tower_mixed' ? dualTowerLayers : null);

  let effectiveQRouting = queryRouting;
  let effectiveKvRouting = kvRouting;
  if (mode === 'split_tower_mixed') {
    if (!effectiveQRouting) {
      effectiveQRouting = new Uint8Array(qLen);
      const splitAt = Math.floor(qLen / 2);
      for (let i = splitAt; i < qLen; i++) effectiveQRouting[i] = 1;
    }
    if (!effectiveKvRouting) {
      if (pastLen === 0 && effectiveQRouting.length === kvLen) {
        effectiveKvRouting = effectiveQRouting;
      } else {
        effectiveKvRouting = new Uint8Array(kvLen);
        const splitAt = Math.floor(kvLen / 2);
        for (let i = splitAt; i < kvLen; i++) effectiveKvRouting[i] = 1;
      }
    }
  }

  const canSplit =
    resolvedLayer !== null &&
    effectiveQRouting !== undefined &&
    effectiveKvRouting !== undefined;

  if (canSplit) {
    const splitTowerApplied = resolvedLayer >= dualTowerLayers;
    const visible = applyElasticSplitTowerMask(
      base,
      effectiveQRouting!,
      effectiveKvRouting!,
      resolvedLayer,
      dualTowerLayers,
    );
    const badge = splitTowerApplied
      ? `${baseBadge} · Split-Tower L${resolvedLayer} (U⊥G)`
      : `${baseBadge} · Joint L${resolvedLayer} (U↔G)`;
    return {
      mode,
      qLen,
      kvLen,
      visible,
      splitTowerApplied,
      layerIndex: resolvedLayer,
      isReadOnlyReference: false,
      badge,
    };
  }

  return {
    mode,
    qLen,
    kvLen,
    visible: base,
    splitTowerApplied: false,
    layerIndex: resolvedLayer,
    isReadOnlyReference: false,
    badge: baseBadge,
  };
}
