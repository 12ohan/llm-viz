/**
 * Deterministic Xoshiro128** PRNG & SplitMix32 Seed Derivation (`src/dory/prng.ts`).
 *
 * Implements `DoryPrngModuleContract` with zero ambient randomness (strictly seeded RNG).
 * Uses SplitMix32 for 32-bit seed derivation and 4-word Xoshiro128** state initialization,
 * providing 128-bit state, full 53-bit binary64 uniforms `[0, 1)`, centered 52-bit strictly
 * open uniforms `(0, 1)` for safe Box-Muller logarithms, and non-aliasing exact state cloning.
 *
 * 100% clean-room original TypeScript — zero DOM, WebGL, Node built-ins, or sibling imports.
 */

import type { DoryPrng } from './types';

const INV_2_POW_53 = 1 / 9_007_199_254_740_992; // 2^-53
const INV_2_POW_52 = 1 / 4_503_599_627_370_496; // 2^-52
const SHIFT_26_MULT = 67_108_864; // 2^26
const TWO_PI = 2 * Math.PI;

function rotl32(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/**
 * Pure 32-bit SplitMix32 mixing step returning an unsigned 32-bit integer in `[0, 2^32 - 1]`.
 */
export function splitMix32(seed: number): number {
  let z = ((seed >>> 0) + 0x9e3779b9) >>> 0;
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
  return (z ^ (z >>> 16)) >>> 0;
}

/**
 * Deterministically derives an uncorrelated 32-bit unsigned child seed from `(baseSeed, salt)`.
 */
export function deriveDorySeed(baseSeed: number, salt: number): number {
  const mixed = ((baseSeed >>> 0) ^ Math.imul(salt >>> 0, 0x9e3779b1)) >>> 0;
  return splitMix32(mixed);
}

function createXoshiro128StarStarFromState(
  initS0: number,
  initS1: number,
  initS2: number,
  initS3: number,
  initHasSpare: boolean,
  initSpare: number,
): DoryPrng {
  let s0 = initS0 >>> 0;
  let s1 = initS1 >>> 0;
  let s2 = initS2 >>> 0;
  let s3 = initS3 >>> 0;
  let hasSpare = initHasSpare;
  let spare = initSpare;

  if ((s0 | s1 | s2 | s3) === 0) {
    s0 = 0x9e3779b9;
  }

  function nextUint32(): number {
    const result = Math.imul(rotl32(Math.imul(s1, 5) >>> 0, 7), 9) >>> 0;
    const t = (s1 << 9) >>> 0;

    s2 = (s2 ^ s0) >>> 0;
    s3 = (s3 ^ s1) >>> 0;
    s1 = (s1 ^ s2) >>> 0;
    s0 = (s0 ^ s3) >>> 0;
    s2 = (s2 ^ t) >>> 0;
    s3 = rotl32(s3, 11);

    return result;
  }

  function nextFloat64(): number {
    const hi = nextUint32() >>> 5; // 27 bits
    const lo = nextUint32() >>> 6; // 26 bits
    return (hi * SHIFT_26_MULT + lo) * INV_2_POW_53;
  }

  function nextOpenUnitFloat64(): number {
    // 52-bit integer k in [0, 2^52 - 1] -> (k + 0.5) / 2^52 is exact in binary64
    // and strictly bounded in [2^-53, 1 - 2^-53] subset (0, 1).
    const hi = nextUint32() >>> 6; // 26 bits
    const lo = nextUint32() >>> 6; // 26 bits
    return (hi * SHIFT_26_MULT + lo + 0.5) * INV_2_POW_52;
  }

  function nextGaussian(): number {
    if (hasSpare) {
      hasSpare = false;
      return spare;
    }
    const u1 = nextOpenUnitFloat64();
    const u2 = nextFloat64();
    const radius = Math.sqrt(-2.0 * Math.log(u1));
    const angle = TWO_PI * u2;
    spare = radius * Math.sin(angle);
    hasSpare = true;
    return radius * Math.cos(angle);
  }

  function nextRange(min: number, max: number): number {
    if (!Number.isFinite(min) || !Number.isFinite(max) || max < min) {
      throw new RangeError(
        `Invalid PRNG range [${String(min)}, ${String(max)}): expected finite min <= max.`,
      );
    }
    if (max === min) {
      return min;
    }
    return min + (max - min) * nextFloat64();
  }

  function getState(): number {
    return (s0 ^ rotl32(s1, 7) ^ rotl32(s2, 13) ^ rotl32(s3, 23)) >>> 0;
  }

  function clone(): DoryPrng {
    return createXoshiro128StarStarFromState(s0, s1, s2, s3, hasSpare, spare);
  }

  return {
    nextUint32,
    nextFloat64,
    nextOpenUnitFloat64,
    nextGaussian,
    nextRange,
    getState,
    clone,
  };
}

/**
 * Creates a deterministic Xoshiro128** PRNG seeded from a 32-bit integer via 4 SplitMix32 steps.
 */
export function createDoryPrng(seed: number): DoryPrng {
  let state = seed >>> 0;
  const nextWord = (): number => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };

  const s0 = nextWord();
  const s1 = nextWord();
  const s2 = nextWord();
  const s3 = nextWord();
  return createXoshiro128StarStarFromState(s0, s1, s2, s3, false, 0);
}
