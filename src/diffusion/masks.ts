/**
 * Five-Mode Attention Masks & Three-Coordinate Positional Layout (`src/diffusion/masks.ts`).
 *
 * Implements:
 *   1. Joint Training `[x_t || x_0]` (Read-Only Analytical Training Mask, `M_BD`, `M_OBC`, `M_BC`)
 *   2. AR Causal (`ar_generate` prefill & decode)
 *   3. Block-Diffusion Bidirectional (`generate` denoise / `linear_spec` draft)
 *   4. Causal Commit / Speculative Verify (`generate` commit / `linear_spec` verify)
 *   5. VLM-8B Cached Quadratic SS (`Quadratic SS · VLM/Paper Ref`, read-only reference)
 *
 * Pure headless TypeScript — zero DOM, WebGL, Math.random(), or `src/twotower/` imports.
 */

import type { DiffusionMaskMatrix, JointTrainingMaskRegion } from './types';

function assertIntAtLeast(value: number, min: number, name: string): void {
  if (!Number.isInteger(value) || value < min) {
    throw new RangeError(`${name} must be an integer >= ${min}, received ${value}`);
  }
}

function contiguousRange(start: number, length: number): Int32Array {
  const arr = new Int32Array(length);
  for (let i = 0; i < length; i++) {
    arr[i] = start + i;
  }
  return arr;
}

/**
 * Mode 1: Joint Training `[x_t(0..L-1) || x_0(0..L-1)]` (`2L x 2L`).
 * Enforces strict `M_OBC` (`b_q > b_k`, guaranteeing `Δp >= 1` and zero `Δp = 0` twin leakage)
 * and surfaces the D23 divergence between chunk-shared `ropePositionIds` (`[0..L-1, 0..L-1]`)
 * and `cachePositions` (`[0..2L-1]`).
 */
export function buildJointTrainingMask(seqLenL: number, blockSizeB: number): DiffusionMaskMatrix {
  assertIntAtLeast(seqLenL, 1, 'seqLenL');
  assertIntAtLeast(blockSizeB, 1, 'blockSizeB');

  const qLen = 2 * seqLenL;
  const kvLen = 2 * seqLenL;
  const visible = new Uint8Array(qLen * kvLen);
  const regions: JointTrainingMaskRegion[] = new Array(qLen * kvLen);

  const seqIdx = contiguousRange(0, qLen);
  const ropePositionIds = new Int32Array(qLen);
  const cachePositions = contiguousRange(0, qLen);
  for (let i = 0; i < seqLenL; i++) {
    ropePositionIds[i] = i;
    ropePositionIds[seqLenL + i] = i;
  }

  for (let q = 0; q < qLen; q++) {
    const fq = q >= seqLenL ? 1 : 0;
    const bq = Math.floor((q - seqLenL * fq) / blockSizeB);
    for (let k = 0; k < kvLen; k++) {
      const fk = k >= seqLenL ? 1 : 0;
      const bk = Math.floor((k - seqLenL * fk) / blockSizeB);
      const idx = q * kvLen + k;

      let region: JointTrainingMaskRegion = 'none';
      if (bq === bk && fk === 0 && fq === 0) {
        region = 'M_BD';
      } else if (bq > bk && fk === 1 && fq === 0) {
        region = 'M_OBC';
      } else if (q >= k && fk === 1 && fq === 1) {
        region = 'M_BC';
      }

      regions[idx] = region;
      visible[idx] = region === 'none' ? 0 : 1;
    }
  }

  return {
    mode: 'joint_training',
    qLen,
    kvLen,
    visible,
    queryCoords: {
      seqIdx,
      ropePositionIds,
      cachePositions,
    },
    kvRopePositionIds: new Int32Array(ropePositionIds),
    kvCachePositions: new Int32Array(cachePositions),
    isReadOnlyReference: true,
    badge: 'Read-Only Analytical Training Mask (No Gradients)',
    regions: Object.freeze(regions),
  };
}

export interface MobcStrictOffsetCheck {
  readonly valid: boolean;
  readonly minDeltaP: number | null;
  readonly zeroTwinViolations: number;
}

/**
 * Verifies that every visible `M_OBC` cell satisfies `ropePos(q) - ropePos(k) >= 1`
 * and that zero noisy-to-clean cells leak at `Δp = 0`.
 */
export function verifyMobcStrictOffset(mask: DiffusionMaskMatrix): MobcStrictOffsetCheck {
  if (mask.mode !== 'joint_training' || !mask.regions || mask.qLen % 2 !== 0 || mask.kvLen % 2 !== 0) {
    return { valid: false, minDeltaP: null, zeroTwinViolations: 0 };
  }

  const seqLenL = mask.qLen / 2;
  let minDeltaP: number | null = null;
  let zeroTwinViolations = 0;
  let valid = true;

  for (let q = 0; q < mask.qLen; q++) {
    const qPos = mask.queryCoords.ropePositionIds[q]!;
    for (let k = 0; k < mask.kvLen; k++) {
      const idx = q * mask.kvLen + k;
      const isVisible = mask.visible[idx] === 1;
      if (!isVisible) continue;

      const region = mask.regions[idx];
      const kPos = mask.kvRopePositionIds[k]!;
      const deltaP = qPos - kPos;

      if (q < seqLenL && k >= seqLenL && deltaP === 0) {
        zeroTwinViolations++;
        valid = false;
      }

      if (region === 'M_OBC') {
        if (minDeltaP === null || deltaP < minDeltaP) {
          minDeltaP = deltaP;
        }
        if (deltaP < 1) {
          valid = false;
        }
      }
    }
  }

  return { valid, minDeltaP, zeroTwinViolations };
}

/** Mode 2: AR Causal (`ar_generate` prefill when `prefixLenP = 0` or decode when `queryLenQ = 1`). */
export function buildArCausalMask(prefixLenP: number, queryLenQ: number): DiffusionMaskMatrix {
  assertIntAtLeast(prefixLenP, 0, 'prefixLenP');
  assertIntAtLeast(queryLenQ, 1, 'queryLenQ');

  const qLen = queryLenQ;
  const kvLen = prefixLenP + queryLenQ;
  const visible = new Uint8Array(qLen * kvLen);

  for (let r = 0; r < qLen; r++) {
    const maxK = prefixLenP + r;
    for (let k = 0; k <= maxK; k++) {
      visible[r * kvLen + k] = 1;
    }
  }

  const seqIdx = contiguousRange(0, qLen);
  const ropePositionIds = contiguousRange(prefixLenP, qLen);
  const cachePositions = contiguousRange(prefixLenP, qLen);
  const kvRopePositionIds = contiguousRange(0, kvLen);
  const kvCachePositions = contiguousRange(0, kvLen);

  return {
    mode: 'ar_causal',
    qLen,
    kvLen,
    visible,
    queryCoords: {
      seqIdx,
      ropePositionIds,
      cachePositions,
    },
    kvRopePositionIds,
    kvCachePositions,
    isReadOnlyReference: false,
    badge: prefixLenP === 0 ? 'AR Causal Prefill' : 'AR Causal Decode',
  };
}

/** Mode 3: Block-Diffusion Bidirectional (`generate` denoise / `linear_spec` draft, `use_cache=False`). */
export function buildBlockBidirMask(prefixLenP: number, blockSizeB: number): DiffusionMaskMatrix {
  assertIntAtLeast(prefixLenP, 0, 'prefixLenP');
  assertIntAtLeast(blockSizeB, 1, 'blockSizeB');

  const qLen = blockSizeB;
  const kvLen = prefixLenP + blockSizeB;
  const visible = new Uint8Array(qLen * kvLen);
  visible.fill(1);

  return {
    mode: 'block_bidir',
    qLen,
    kvLen,
    visible,
    queryCoords: {
      seqIdx: contiguousRange(0, qLen),
      ropePositionIds: contiguousRange(prefixLenP, qLen),
      cachePositions: contiguousRange(prefixLenP, qLen),
    },
    kvRopePositionIds: contiguousRange(0, kvLen),
    kvCachePositions: contiguousRange(0, kvLen),
    isReadOnlyReference: false,
    badge: 'Block-Diff Bidir (use_cache=False)',
  };
}

/** Mode 4: Causal Commit / Speculative Verify (`generate` commit / `linear_spec` verify, `use_cache=True`). */
export function buildCausalCommitVerifyMask(
  prefixLenP: number,
  blockSizeB: number,
): DiffusionMaskMatrix {
  assertIntAtLeast(prefixLenP, 0, 'prefixLenP');
  assertIntAtLeast(blockSizeB, 1, 'blockSizeB');

  const qLen = blockSizeB;
  const kvLen = prefixLenP + blockSizeB;
  const visible = new Uint8Array(qLen * kvLen);

  for (let r = 0; r < qLen; r++) {
    const maxK = prefixLenP + r;
    for (let k = 0; k <= maxK; k++) {
      visible[r * kvLen + k] = 1;
    }
  }

  return {
    mode: 'causal_commit_verify',
    qLen,
    kvLen,
    visible,
    queryCoords: {
      seqIdx: contiguousRange(0, qLen),
      ropePositionIds: contiguousRange(prefixLenP, qLen),
      cachePositions: contiguousRange(prefixLenP, qLen),
    },
    kvRopePositionIds: contiguousRange(0, kvLen),
    kvCachePositions: contiguousRange(0, kvLen),
    isReadOnlyReference: false,
    badge: 'Causal Commit / Spec Verify (use_cache=True)',
  };
}

/**
 * Mode 5 (Read-Only Reference): VLM-8B Cached Quadratic Self-Speculation Mask.
 * Exact predicate from `modeling_nemotron_labs_diffusion_vlm.py` (DF0 §6.3 Fixture 5):
 *   - Cached KVs reordered `[draft_kv(0..draftLen-1) || clean_kv(0..P-1)]`
 *   - `cleanPart = kv >= draftLen`
 *   - `firstClean = (kv % (k + 1) === 0) && (kv < draftLen) && (q >= kv)`
 *   - `sameBlockExceptFirst = (floor(q / (k + 1)) === floor(kv / (k + 1))) && (q < draftLen) && (q % (k + 1) !== 0)`
 */
export function buildVlmQuadraticSsMask(prefixLenP: number, blockK: number): DiffusionMaskMatrix {
  assertIntAtLeast(prefixLenP, 0, 'prefixLenP');
  assertIntAtLeast(blockK, 1, 'blockK');

  const stride = blockK + 1;
  const draftLen = blockK * stride;
  const qLen = draftLen;
  const kvLen = draftLen + prefixLenP;
  const visible = new Uint8Array(qLen * kvLen);

  const seqIdx = contiguousRange(0, qLen);
  const ropePositionIds = new Int32Array(qLen);
  for (let b = 0; b < blockK; b++) {
    for (let j = 0; j <= blockK; j++) {
      ropePositionIds[b * stride + j] = prefixLenP + b + j;
    }
  }
  const cachePositions = contiguousRange(prefixLenP, qLen);

  const kvRopePositionIds = new Int32Array(kvLen);
  const kvCachePositions = new Int32Array(kvLen);
  kvRopePositionIds.set(ropePositionIds, 0);
  kvCachePositions.set(cachePositions, 0);
  for (let p = 0; p < prefixLenP; p++) {
    kvRopePositionIds[draftLen + p] = p;
    kvCachePositions[draftLen + p] = p;
  }

  for (let q = 0; q < qLen; q++) {
    const qBlock = Math.floor(q / stride);
    const qInBlock = q % stride;
    for (let kv = 0; kv < kvLen; kv++) {
      const cleanPart = kv >= draftLen;
      const firstClean = kv % stride === 0 && kv < draftLen && q >= kv;
      const sameBlockExceptFirst =
        qBlock === Math.floor(kv / stride) && q < draftLen && qInBlock !== 0;
      visible[q * kvLen + kv] = cleanPart || firstClean || sameBlockExceptFirst ? 1 : 0;
    }
  }

  return {
    mode: 'vlm_quadratic_ss',
    qLen,
    kvLen,
    visible,
    queryCoords: {
      seqIdx,
      ropePositionIds,
      cachePositions,
    },
    kvRopePositionIds,
    kvCachePositions,
    isReadOnlyReference: true,
    badge: 'Quadratic SS · VLM/Paper Ref',
  };
}

/** Prebuilt canonical fixtures matching DF0 §6.3 Fixtures 1, 2A, 2B, 3, 4, and 5. */
export const DIFFUSION_MASK_FIXTURES: {
  readonly jointTraining: DiffusionMaskMatrix;
  readonly arPrefill: DiffusionMaskMatrix;
  readonly arDecode: DiffusionMaskMatrix;
  readonly blockBidir: DiffusionMaskMatrix;
  readonly causalCommitVerify: DiffusionMaskMatrix;
  readonly vlmQuadraticSs: DiffusionMaskMatrix;
} = Object.freeze({
  jointTraining: buildJointTrainingMask(6, 2),
  arPrefill: buildArCausalMask(0, 4),
  arDecode: buildArCausalMask(4, 1),
  blockBidir: buildBlockBidirMask(4, 4),
  causalCommitVerify: buildCausalCommitVerifyMask(4, 4),
  vlmQuadraticSs: buildVlmQuadraticSsMask(2, 2),
});
