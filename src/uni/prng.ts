/**
 * Deterministic Clean-Room PRNG & Hashing Primitives for Nemotron-Labs-Diffusion-Uni (`src/uni/prng.ts`)
 *
 * Pure headless TypeScript:
 *   - ZERO `Math.random()` (enforces `U23` explicit seeded randomness)
 *   - ZERO DOM, WebGL, `src/twotower/`, `src/diffusion/`, or `src/ui/controls.ts` imports
 *   - Open-unit uniform sampling strictly clamped to `[1e-15, 1 - 1e-15]` to prevent
 *     `log(0)` / `-log(1)` infinities in log-domain Gumbel sampling (`U22`).
 */

import type { UniXoshiro128ssGenerator, UniXoshiro128ssState } from './types';

const OPEN_UNIT_MIN = 1e-15;
const OPEN_UNIT_MAX = 1 - 1e-15;
const INV_2_POW_24 = 1 / 16_777_216; // 2^-24
const POW_2_26 = 67_108_864; // 2^26
const INV_2_POW_53 = 1 / 9_007_199_254_740_992; // 2^-53
const INV_2_POW_32 = 1 / 4_294_967_296; // 2^-32

const FNV1A_OFFSET_32 = 0x811c9dc5;
const FNV1A_OFFSET_ALT_32 = 0x84222325;
const FNV1A_PRIME_32 = 0x01000193;

const UTF8_ENCODER = new TextEncoder();

function rotl32(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

function fmix32(h: number): number {
  let x = h >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b) >>> 0;
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

function toBytes(input: string | Uint8Array): Uint8Array {
  return typeof input === 'string' ? UTF8_ENCODER.encode(input) : input;
}

/**
 * 32-bit SplitMix mixing function returning an unsigned 32-bit integer in `[0, 2^32 - 1]`.
 */
export function splitmix32(seed: number): number {
  let z = ((seed >>> 0) + 0x9e3779b9) >>> 0;
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
  return (z ^ (z >>> 16)) >>> 0;
}

function createXoshiro128ssFromState(
  init0: number,
  init1: number,
  init2: number,
  init3: number,
): UniXoshiro128ssGenerator {
  let s0 = init0 >>> 0;
  let s1 = init1 >>> 0;
  let s2 = init2 >>> 0;
  let s3 = init3 >>> 0;

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

  function nextFloat32(): number {
    return (nextUint32() >>> 8) * INV_2_POW_24;
  }

  function nextFloat64(): number {
    const hi = nextUint32() >>> 5; // 27 bits
    const lo = nextUint32() >>> 6; // 26 bits
    return (hi * POW_2_26 + lo) * INV_2_POW_53;
  }

  function nextUniformOpenUnit(): number {
    const u = nextFloat64();
    if (u < OPEN_UNIT_MIN) return OPEN_UNIT_MIN;
    if (u > OPEN_UNIT_MAX) return OPEN_UNIT_MAX;
    return u;
  }

  function getState(): UniXoshiro128ssState {
    return Object.freeze([s0, s1, s2, s3] as const);
  }

  function clone(): UniXoshiro128ssGenerator {
    return createXoshiro128ssFromState(s0, s1, s2, s3);
  }

  return {
    nextUint32,
    nextFloat32,
    nextFloat64,
    nextUniformOpenUnit,
    getState,
    clone,
  };
}

/**
 * Create a seeded `xoshiro128**` PRNG initialized via 4 sequential `splitmix32` words.
 */
export function createXoshiro128ss(seed: number): UniXoshiro128ssGenerator {
  const s0 = splitmix32(seed >>> 0);
  const s1 = splitmix32(s0);
  const s2 = splitmix32(s1);
  const s3 = splitmix32(s2);
  return createXoshiro128ssFromState(s0, s1, s2, s3);
}

/**
 * Standard 32-bit FNV-1a hash over UTF-8 string bytes or a `Uint8Array`.
 */
export function fnv1a32(input: string | Uint8Array): number {
  const bytes = toBytes(input);
  let hash = FNV1A_OFFSET_32;
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i]!;
    hash = Math.imul(hash, FNV1A_PRIME_32) >>> 0;
  }
  return hash >>> 0;
}

/**
 * Deterministic, order- and length-sensitive 32-bit seed mixer.
 */
export function mixSeed32(...parts: number[]): number {
  let acc = FNV1A_OFFSET_32;
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]! >>> 0;
    const laneSalt = Math.imul(i + 1, 0x9e3779b1) >>> 0;
    acc = Math.imul(acc ^ (part + laneSalt), 0x85ebca6b) >>> 0;
    acc = splitmix32(acc);
  }
  acc = Math.imul(acc ^ (parts.length >>> 0), 0xc2b2ae35) >>> 0;
  return fmix32(acc);
}

/**
 * Deterministic 64-bit fingerprint formatted as 16 lowercase hexadecimal characters,
 * combining dual independent 32-bit FNV-1a passes finalized with Murmur3 `fmix32`.
 */
export function dualHashHex(input: string | Uint8Array): string {
  const bytes = toBytes(input);
  const len = bytes.length;

  let h1 = FNV1A_OFFSET_32;
  let h2 = FNV1A_OFFSET_ALT_32 ^ (len >>> 0);

  for (let i = 0; i < len; i++) {
    const b = bytes[i]!;
    h1 ^= b;
    h1 = Math.imul(h1, FNV1A_PRIME_32) >>> 0;

    const rev = bytes[len - 1 - i]!;
    h2 ^= (rev + ((i * 31) & 0xff)) & 0xff;
    h2 = Math.imul(h2, FNV1A_PRIME_32) >>> 0;
  }

  const hi = fmix32((h1 ^ Math.imul(len, 0x9e3779b9)) >>> 0);
  const lo = fmix32((h2 ^ hi) >>> 0);

  return hi.toString(16).padStart(8, '0') + lo.toString(16).padStart(8, '0');
}

/**
 * Classic 32-bit Mulberry32 PRNG returning deterministic floats in `[0, 1)`.
 */
export function createMulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) * INV_2_POW_32;
  };
}

/**
 * Standard normal `N(0, 1)` sample via Box-Muller transform.
 * Clamps the first uniform variate strictly to `[1e-15, 1 - 1e-15]` to guarantee finite outputs.
 */
export function nextGaussianBoxMuller(nextUniform: () => number): number {
  let u1 = nextUniform();
  if (!Number.isFinite(u1) || u1 < OPEN_UNIT_MIN) {
    u1 = OPEN_UNIT_MIN;
  } else if (u1 > OPEN_UNIT_MAX) {
    u1 = OPEN_UNIT_MAX;
  }
  const u2 = nextUniform();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}
