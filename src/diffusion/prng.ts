/**
 * Deterministic PRNGs, Seed Mixing, Dual Hasher & Box-Muller Gaussian Sampling
 * (`src/diffusion/prng.ts`).
 *
 * Pure headless TypeScript — zero `Math.random()`, zero DOM/WebGL, zero `src/twotower/` imports.
 */

import type { Xoshiro128ssGenerator, Xoshiro128ssState } from './types';

const OPEN_UNIT_MIN = 1e-15;
const OPEN_UNIT_MAX = 1 - 1e-15;
const TWO_POW_24 = 16_777_216;
const TWO_POW_26 = 67_108_864;
const TWO_POW_32 = 4_294_967_296;
const TWO_POW_53 = 9_007_199_254_740_992;

const UTF8_ENCODER = new TextEncoder();

function toBytes(input: string | Uint8Array): Uint8Array {
  return typeof input === 'string' ? UTF8_ENCODER.encode(input) : input;
}

function rotl32(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

function fmix32(h: number): number {
  let z = h >>> 0;
  z ^= z >>> 16;
  z = Math.imul(z, 0x85ebca6b) >>> 0;
  z ^= z >>> 13;
  z = Math.imul(z, 0xc2b2ae35) >>> 0;
  return (z ^ (z >>> 16)) >>> 0;
}

/** Single-step SplitMix32 integer mixer returning an unsigned 32-bit integer. */
export function splitmix32(seed: number): number {
  let z = (seed + 0x9e3779b9) >>> 0;
  z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
  z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
  return (z ^ (z >>> 16)) >>> 0;
}

function makeXoshiroFromState(
  init0: number,
  init1: number,
  init2: number,
  init3: number,
): Xoshiro128ssGenerator {
  let s0 = init0 >>> 0;
  let s1 = init1 >>> 0;
  let s2 = init2 >>> 0;
  let s3 = init3 >>> 0;

  if ((s0 | s1 | s2 | s3) === 0) {
    s0 = 0x9e3779b9;
  }

  const nextUint32 = (): number => {
    const result = Math.imul(rotl32(Math.imul(s1, 5) >>> 0, 7), 9) >>> 0;
    const t = (s1 << 9) >>> 0;

    s2 = (s2 ^ s0) >>> 0;
    s3 = (s3 ^ s1) >>> 0;
    s1 = (s1 ^ s2) >>> 0;
    s0 = (s0 ^ s3) >>> 0;
    s2 = (s2 ^ t) >>> 0;
    s3 = rotl32(s3, 11);

    return result;
  };

  const nextFloat32 = (): number => {
    return (nextUint32() >>> 8) / TWO_POW_24;
  };

  const nextFloat64 = (): number => {
    const hi = nextUint32() >>> 5;
    const lo = nextUint32() >>> 6;
    return (hi * TWO_POW_26 + lo) / TWO_POW_53;
  };

  const nextUniformOpenUnit = (): number => {
    const u = nextFloat64();
    if (u < OPEN_UNIT_MIN) return OPEN_UNIT_MIN;
    if (u > OPEN_UNIT_MAX) return OPEN_UNIT_MAX;
    return u;
  };

  const getState = (): Xoshiro128ssState => {
    return Object.freeze([s0, s1, s2, s3] as const);
  };

  const clone = (): Xoshiro128ssGenerator => {
    return makeXoshiroFromState(s0, s1, s2, s3);
  };

  return {
    nextUint32,
    nextFloat32,
    nextFloat64,
    nextUniformOpenUnit,
    getState,
    clone,
  };
}

/** Creates a deterministic `xoshiro128**` PRNG seeded via 4 chained SplitMix32 steps. */
export function createXoshiro128ss(seed: number): Xoshiro128ssGenerator {
  const s0 = splitmix32(seed >>> 0);
  const s1 = splitmix32(s0);
  const s2 = splitmix32(s1);
  const s3 = splitmix32(s2);
  return makeXoshiroFromState(s0, s1, s2, s3);
}

/** 32-bit FNV-1a hash over UTF-8 string or raw bytes. */
export function fnv1a32(input: string | Uint8Array): number {
  const bytes = toBytes(input);
  let hash = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i]!;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** Order-sensitive 32-bit seed mixer using chained SplitMix32 steps. */
export function mixSeed32(...parts: number[]): number {
  let acc = 0x9e3779b9;
  for (let i = 0; i < parts.length; i++) {
    acc = splitmix32((acc ^ (parts[i]! >>> 0)) >>> 0);
  }
  return acc >>> 0;
}

/**
 * Deterministic 16-hex-char dual 32-bit FNV-1a + Murmur3-fmix32 digest.
 * Combines standard FNV-1a with a position/length-salted second lane.
 */
export function dualHashHex(input: string | Uint8Array): string {
  const bytes = toBytes(input);
  let h1 = 0x811c9dc5;
  let h2 = 0x1f351f35 ^ bytes.length;
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i]!;
    h1 ^= b;
    h1 = Math.imul(h1, 0x01000193) >>> 0;

    h2 ^= (b + ((i * 0x9e) & 0xff)) & 0xff;
    h2 = Math.imul(h2, 0x01000193) >>> 0;
    h2 = rotl32(h2, 5);
  }
  const hi = fmix32(h1 ^ bytes.length);
  const lo = fmix32(h2 ^ hi);
  return hi.toString(16).padStart(8, '0') + lo.toString(16).padStart(8, '0');
}

/** Classic Mulberry32 generator returning uniform floats in `[0, 1)`. */
export function createMulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / TWO_POW_32;
  };
}

/** Box-Muller standard normal `N(0, 1)` sample driven by the supplied uniform variate source. */
export function nextGaussianBoxMuller(nextUniform: () => number): number {
  const u1 = Math.min(Math.max(nextUniform(), OPEN_UNIT_MIN), OPEN_UNIT_MAX);
  const u2 = nextUniform();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}
