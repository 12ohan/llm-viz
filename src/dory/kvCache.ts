/**
 * Dual-Axis KV Cache & Long-Chunk Transient Attention Workspace (`src/dory/kvCache.ts`)
 *
 * Implements the locked dual-axis KV cache contract for NVIDIA Nemotron-Labs-Dory1.0-4B:
 *   - Policy axis: `per_loop` (`16 + 20k` persistent banks; `104` banks with Contract A
 *     separate persistent Shallow Coda KV) vs. `last_loop` (`36` persistent banks at any `k`,
 *     reading committed final-loop history without mutating the bank on non-final loops `t < k`,
 *     matching `DoryPastReader` semantics — `DR17`).
 *   - Attention axis: Sliding-Window Attention (`W = 8`, `cap_swa = 16 >= W + gamma + 1 = 13`
 *     power-of-2 ring buffer with per-slot `absPosTag` & `writerLoopTag`, `DR19`) vs.
 *     Full Attention (`capacity = S_max = 32`).
 *   - Long-chunk workspace (`DR24`): `buildTransientKvWorkspace()` constructs a read-only
 *     `[past_visible (<= W - 1) || current_chunk (L)]` buffer (up to `39` positions for `L = 32`)
 *     so chunks exceeding `cap_swa = 16` execute exact SWA attention with zero ring overwrite;
 *     `commitChunkToKvBank()` then commits only the newest `min(endPos, cap_swa)` slots.
 *   - Non-aliased cloning (`cloneDoryKvCache`) for immutable branch forks & checkpoints (`DR18`),
 *     exact `chunkSizes[]` schedule tracking for `last_loop` lazy replay (`DR24`), and
 *     transaction-local `O(1)` length rewind (`rewindDoryKvCache`) with stale-tag invalidation.
 *
 * 100% clean-room original TypeScript — zero ambient RNG, zero DOM/WebGL/Node/sibling imports.
 */

import {
  getDoryPhysicalLayerSpec,
  validateDoryRecurrentLoops,
} from './spec';
import {
  DORY_MINI_CONFIG,
  type DoryAttentionKind,
  type DoryCodaBankKind,
  type DoryKvBank,
  type DoryKvBankKind,
  type DoryKvCache,
  type DoryKvCacheByteUsage,
  type DoryKvCacheCreateOptions,
  type DoryLayerStage,
  type DoryMiniConfig,
  type DoryPhysicalLayerSpec,
  type DoryTransientKvWorkspace,
} from './types';

function slotIndexForPos(bank: DoryKvBank, absPos: number): number {
  if (bank.attentionKind === 'sliding_attention') {
    const cap = bank.capacity;
    if ((cap & (cap - 1)) === 0) {
      return absPos & (cap - 1);
    }
    return absPos % cap;
  }
  return absPos;
}

function createBank(
  bankSlotId: number,
  layerSpec: DoryPhysicalLayerSpec,
  bankKind: DoryKvBankKind,
  loopSlot: number,
  config: DoryMiniConfig,
): DoryKvBank {
  const attentionKind: DoryAttentionKind | null = layerSpec.attentionKind;
  if (attentionKind === null) {
    throw new RangeError(
      `Physical layer ${layerSpec.physicalLayerIdx} is an MLP sublayer and cannot own a KV bank.`,
    );
  }
  const isSliding = attentionKind === 'sliding_attention';
  const capacity = isSliding ? config.swaRingCapacity : config.maxPositionEmbeddings;
  const windowSize = isSliding ? config.slidingWindow : null;
  const kvDim = config.kvProjOutDim;

  const absPosTag = new Int32Array(capacity);
  absPosTag.fill(-1);
  const writerLoopTag = new Int32Array(capacity);
  writerLoopTag.fill(-1);

  return {
    bankSlotId,
    physicalLayerIdx: layerSpec.physicalLayerIdx,
    pairedLayerIdx: layerSpec.pairedLayerIdx,
    stage: layerSpec.stage as DoryLayerStage,
    bankKind,
    attentionKind,
    loopSlot,
    capacity,
    windowSize,
    kvDim,
    k: new Float32Array(capacity * kvDim),
    v: new Float32Array(capacity * kvDim),
    absPosTag,
    writerLoopTag,
    logicalLen: 0,
  };
}

function cloneBank(bank: DoryKvBank): DoryKvBank {
  return {
    bankSlotId: bank.bankSlotId,
    physicalLayerIdx: bank.physicalLayerIdx,
    pairedLayerIdx: bank.pairedLayerIdx,
    stage: bank.stage,
    bankKind: bank.bankKind,
    attentionKind: bank.attentionKind,
    loopSlot: bank.loopSlot,
    capacity: bank.capacity,
    windowSize: bank.windowSize,
    kvDim: bank.kvDim,
    k: new Float32Array(bank.k),
    v: new Float32Array(bank.v),
    absPosTag: new Int32Array(bank.absPosTag),
    writerLoopTag: new Int32Array(bank.writerLoopTag),
    logicalLen: bank.logicalLen,
  };
}

/**
 * Allocates a fresh dual-axis `DoryKvCache` with all slots tagged `-1` (`empty`).
 *
 * Bank ordering (`bankSlotId` ascending):
 *   1. `input` (`0..7`): physical layers `{0, 2, 4, 6, 8, 10, 12, 14}`
 *   2. `recurrent`: `k` sets in `per_loop` (`loopSlot = 1..k`) or `1` shared set in `last_loop`
 *      (`loopSlot = 1`), physical layers `{16, 18, ..., 54}` (`20` banks per set)
 *   3. `output_deep`: physical layers `{56, 58, 60, 62, 64, 66, 68, 70}` (`8` banks)
 *   4. `output_shallow` (iff `includeShallowCoda === true`, Contract A `D3`/`DR23`): `8` banks
 */
export function createDoryKvCache(options: DoryKvCacheCreateOptions): DoryKvCache {
  const { mode } = options;
  if (mode !== 'per_loop' && mode !== 'last_loop') {
    throw new RangeError(`Unsupported Dory KV cache mode: ${String(mode)}`);
  }
  const config = options.config ?? DORY_MINI_CONFIG;
  const kLoops = validateDoryRecurrentLoops(options.kLoops);
  const includeShallowCoda = options.includeShallowCoda ?? false;

  const inputSpecs: DoryPhysicalLayerSpec[] = [];
  const recurrentSpecs: DoryPhysicalLayerSpec[] = [];
  const outputSpecs: DoryPhysicalLayerSpec[] = [];

  for (let p = 0; p < config.numHiddenLayers; p += 1) {
    const spec = getDoryPhysicalLayerSpec(p);
    if (spec.attentionKind === null) {
      continue;
    }
    if (spec.stage === 'input') {
      inputSpecs.push(spec);
    } else if (spec.stage === 'recurrent') {
      recurrentSpecs.push(spec);
    } else {
      outputSpecs.push(spec);
    }
  }

  const allBanks: DoryKvBank[] = [];
  let nextSlotId = 0;

  // 1. Input Prelude banks (8)
  const inputBanks = new Map<number, DoryKvBank>();
  for (let i = 0; i < inputSpecs.length; i += 1) {
    const spec = inputSpecs[i];
    const bank = createBank(nextSlotId, spec, 'input', 0, config);
    nextSlotId += 1;
    inputBanks.set(spec.physicalLayerIdx, bank);
    allBanks.push(bank);
  }

  // 2. Recurrent Core banks (20 * k in per_loop; 20 in last_loop)
  const recurrentSetCount = mode === 'per_loop' ? kLoops : 1;
  const recurrentBanks: Map<number, DoryKvBank>[] = [];
  for (let loopIdx = 0; loopIdx < recurrentSetCount; loopIdx += 1) {
    const loopSlot = loopIdx + 1;
    const loopMap = new Map<number, DoryKvBank>();
    for (let i = 0; i < recurrentSpecs.length; i += 1) {
      const spec = recurrentSpecs[i];
      const bank = createBank(nextSlotId, spec, 'recurrent', loopSlot, config);
      nextSlotId += 1;
      loopMap.set(spec.physicalLayerIdx, bank);
      allBanks.push(bank);
    }
    recurrentBanks.push(loopMap);
  }

  // 3. Output Deep Coda banks (8)
  const outputDeepBanks = new Map<number, DoryKvBank>();
  for (let i = 0; i < outputSpecs.length; i += 1) {
    const spec = outputSpecs[i];
    const bank = createBank(nextSlotId, spec, 'output_deep', 0, config);
    nextSlotId += 1;
    outputDeepBanks.set(spec.physicalLayerIdx, bank);
    allBanks.push(bank);
  }

  // 4. Optional Output Shallow Coda banks (8, Contract A self-speculation)
  let outputShallowBanks: Map<number, DoryKvBank> | null = null;
  if (includeShallowCoda) {
    outputShallowBanks = new Map<number, DoryKvBank>();
    for (let i = 0; i < outputSpecs.length; i += 1) {
      const spec = outputSpecs[i];
      const bank = createBank(nextSlotId, spec, 'output_shallow', 0, config);
      nextSlotId += 1;
      outputShallowBanks.set(spec.physicalLayerIdx, bank);
      allBanks.push(bank);
    }
  }

  return {
    mode,
    kLoops,
    includeShallowCoda,
    config,
    inputBanks,
    recurrentBanks,
    outputDeepBanks,
    outputShallowBanks,
    allBanks,
    committedLen: 0,
    chunkSizes: [],
  };
}

/**
 * Produces a strictly non-aliased deep clone of `cache` with independent `ArrayBuffer`s
 * for every bank (`k`, `v`, `absPosTag`, `writerLoopTag`) and a copied `chunkSizes` schedule.
 * Guarantees immutable branch forks & sparse checkpoints (`DR18`, `DR19`).
 */
export function cloneDoryKvCache(cache: DoryKvCache): DoryKvCache {
  const allBanks: DoryKvBank[] = new Array<DoryKvBank>(cache.allBanks.length);

  const inputBanks = new Map<number, DoryKvBank>();
  cache.inputBanks.forEach((bank, physicalLayerIdx) => {
    const copy = cloneBank(bank);
    inputBanks.set(physicalLayerIdx, copy);
    allBanks[copy.bankSlotId] = copy;
  });

  const recurrentBanks: Map<number, DoryKvBank>[] = cache.recurrentBanks.map((loopMap) => {
    const clonedMap = new Map<number, DoryKvBank>();
    loopMap.forEach((bank, physicalLayerIdx) => {
      const copy = cloneBank(bank);
      clonedMap.set(physicalLayerIdx, copy);
      allBanks[copy.bankSlotId] = copy;
    });
    return clonedMap;
  });

  const outputDeepBanks = new Map<number, DoryKvBank>();
  cache.outputDeepBanks.forEach((bank, physicalLayerIdx) => {
    const copy = cloneBank(bank);
    outputDeepBanks.set(physicalLayerIdx, copy);
    allBanks[copy.bankSlotId] = copy;
  });

  let outputShallowBanks: Map<number, DoryKvBank> | null = null;
  if (cache.outputShallowBanks !== null) {
    outputShallowBanks = new Map<number, DoryKvBank>();
    cache.outputShallowBanks.forEach((bank, physicalLayerIdx) => {
      const copy = cloneBank(bank);
      outputShallowBanks!.set(physicalLayerIdx, copy);
      allBanks[copy.bankSlotId] = copy;
    });
  }

  return {
    mode: cache.mode,
    kLoops: cache.kLoops,
    includeShallowCoda: cache.includeShallowCoda,
    config: cache.config,
    inputBanks,
    recurrentBanks,
    outputDeepBanks,
    outputShallowBanks,
    allBanks,
    committedLen: cache.committedLen,
    chunkSizes: [...cache.chunkSizes],
  };
}

export function getDoryInputBank(cache: DoryKvCache, physicalLayerIdx: number): DoryKvBank {
  const bank = cache.inputBanks.get(physicalLayerIdx);
  if (!bank) {
    throw new RangeError(
      `No input KV bank exists for physicalLayerIdx=${physicalLayerIdx} (expected even Prelude index in 0..14).`,
    );
  }
  return bank;
}

export function getDoryRecurrentBank(
  cache: DoryKvCache,
  physicalLayerIdx: number,
  loopIteration: number,
): DoryKvBank {
  if (!Number.isInteger(loopIteration) || loopIteration < 1 || loopIteration > cache.kLoops) {
    throw new RangeError(
      `Invalid recurrent loopIteration=${loopIteration} for cache with kLoops=${cache.kLoops}.`,
    );
  }
  const loopIndex = cache.mode === 'per_loop' ? loopIteration - 1 : 0;
  const loopMap = cache.recurrentBanks[loopIndex];
  const bank = loopMap?.get(physicalLayerIdx);
  if (!bank) {
    throw new RangeError(
      `No recurrent KV bank exists for physicalLayerIdx=${physicalLayerIdx}, loopIteration=${loopIteration} (mode=${cache.mode}).`,
    );
  }
  return bank;
}

export function getDoryOutputBank(
  cache: DoryKvCache,
  physicalLayerIdx: number,
  codaKind: DoryCodaBankKind = 'output_deep',
): DoryKvBank {
  if (codaKind === 'output_shallow') {
    if (cache.outputShallowBanks === null) {
      throw new Error(
        `Requested 'output_shallow' KV bank for physicalLayerIdx=${physicalLayerIdx}, but cache was created with includeShallowCoda=false.`,
      );
    }
    const bank = cache.outputShallowBanks.get(physicalLayerIdx);
    if (!bank) {
      throw new RangeError(
        `No output_shallow KV bank exists for physicalLayerIdx=${physicalLayerIdx} (expected even Coda index in 56..70).`,
      );
    }
    return bank;
  }
  if (codaKind !== 'output_deep') {
    throw new RangeError(`Unsupported codaKind=${String(codaKind)}.`);
  }
  const bank = cache.outputDeepBanks.get(physicalLayerIdx);
  if (!bank) {
    throw new RangeError(
      `No output_deep KV bank exists for physicalLayerIdx=${physicalLayerIdx} (expected even Coda index in 56..70).`,
    );
  }
  return bank;
}

/**
 * Builds a strictly read-only transient KV workspace `[past_visible || current_chunk]` (`DR24`):
 *   - For SWA banks (`W = 8`, `cap_swa = 16`): gathers `[max(0, startPos - (W - 1)) .. startPos - 1]`
 *     (`<= W - 1 = 7` past tokens) verified via `absPosTag[slot] === absPos`, followed by all
 *     `chunkLen` current positions `[startPos .. startPos + chunkLen - 1]` (supporting `L > 16`
 *     up to `S_max = 32` without overwriting persistent ring entries during attention!).
 *   - For Full banks (`capacity = 32`): gathers `[0 .. startPos - 1]` verified via `absPosTag`,
 *     followed by all `chunkLen` current positions.
 *
 * NEVER mutates `bank` — making non-final `last_loop` iterations (`t < k`) and read-only
 * counterfactual Coda probes completely side-effect-free (`DR17`).
 */
export function buildTransientKvWorkspace(
  bank: DoryKvBank,
  startPos: number,
  currentK: Float32Array,
  currentV: Float32Array,
  chunkLen: number,
  currentWriterLoop: number,
): DoryTransientKvWorkspace {
  if (!Number.isInteger(chunkLen) || chunkLen < 1) {
    throw new RangeError(`chunkLen must be an integer >= 1; got ${chunkLen}.`);
  }
  if (!Number.isInteger(startPos) || startPos < 0) {
    throw new RangeError(`startPos must be an integer >= 0; got ${startPos}.`);
  }
  if (startPos > bank.logicalLen) {
    throw new RangeError(
      `Non-contiguous KV read on bankSlotId=${bank.bankSlotId} (p=${bank.physicalLayerIdx}): startPos=${startPos} exceeds bank.logicalLen=${bank.logicalLen}.`,
    );
  }
  const kvDim = bank.kvDim;
  const expectedElems = chunkLen * kvDim;
  if (currentK.length !== expectedElems || currentV.length !== expectedElems) {
    throw new RangeError(
      `currentK/currentV length mismatch for chunkLen=${chunkLen}, kvDim=${kvDim}: got K=${currentK.length}, V=${currentV.length}, expected ${expectedElems}.`,
    );
  }
  if (bank.attentionKind === 'full_attention' && startPos + chunkLen > bank.capacity) {
    throw new RangeError(
      `Full-attention KV overflow on bankSlotId=${bank.bankSlotId} (p=${bank.physicalLayerIdx}): startPos(${startPos}) + chunkLen(${chunkLen}) > capacity(${bank.capacity}).`,
    );
  }

  let pastStart = 0;
  if (bank.attentionKind === 'sliding_attention') {
    const windowSize = bank.windowSize;
    if (windowSize === null || windowSize < 1) {
      throw new Error(
        `SWA bankSlotId=${bank.bankSlotId} has invalid windowSize=${String(windowSize)}.`,
      );
    }
    pastStart = Math.max(0, startPos - (windowSize - 1));
  }

  const pastLen = startPos - pastStart;
  const totalLen = pastLen + chunkLen;
  const k = new Float32Array(totalLen * kvDim);
  const v = new Float32Array(totalLen * kvDim);
  const absPositions = new Int32Array(totalLen);
  const writerLoops = new Int32Array(totalLen);

  // 1. Gather strictly-past visible committed entries, enforcing per-slot absPosTag (DR19)
  for (let i = 0; i < pastLen; i += 1) {
    const absPos = pastStart + i;
    const slot = slotIndexForPos(bank, absPos);
    const taggedPos = bank.absPosTag[slot];
    if (taggedPos !== absPos) {
      throw new Error(
        `Stale or evicted KV slot read on bankSlotId=${bank.bankSlotId} (p=${bank.physicalLayerIdx}, ${bank.attentionKind}): expected absPos=${absPos} at ring slot ${slot}, found absPosTag=${taggedPos} (logicalLen=${bank.logicalLen}).`,
      );
    }
    const writer = bank.writerLoopTag[slot];
    if (writer < 0) {
      throw new Error(
        `Corrupted writerLoopTag=${writer} at ring slot ${slot} (absPos=${absPos}) on bankSlotId=${bank.bankSlotId}.`,
      );
    }
    const srcOff = slot * kvDim;
    const dstOff = i * kvDim;
    for (let c = 0; c < kvDim; c += 1) {
      k[dstOff + c] = bank.k[srcOff + c];
      v[dstOff + c] = bank.v[srcOff + c];
    }
    absPositions[i] = absPos;
    writerLoops[i] = writer;
  }

  // 2. Append current chunk entries [startPos .. startPos + chunkLen - 1]
  for (let j = 0; j < chunkLen; j += 1) {
    const srcOff = j * kvDim;
    const dstIdx = pastLen + j;
    const dstOff = dstIdx * kvDim;
    for (let c = 0; c < kvDim; c += 1) {
      k[dstOff + c] = currentK[srcOff + c];
      v[dstOff + c] = currentV[srcOff + c];
    }
    absPositions[dstIdx] = startPos + j;
    writerLoops[dstIdx] = currentWriterLoop;
  }

  return {
    pastLen,
    chunkLen,
    totalLen,
    kvDim,
    k,
    v,
    absPositions,
    writerLoops,
  };
}

/**
 * Commits `[startPos .. startPos + chunkLen - 1]` into `bank` AFTER chunk attention finishes:
 *   - Enforces contiguous commit (`startPos === bank.logicalLen`).
 *   - For Full attention: writes all `chunkLen` positions into slots `[startPos .. endPos - 1]`.
 *   - For SWA (`cap_swa = 16`, `DR24`): writes only the newest `min(endPos, capacity)` positions
 *     falling within the current chunk (`[max(startPos, endPos - capacity) .. endPos - 1]`),
 *     updating `absPosTag[slot]` and `writerLoopTag[slot]` atomically with `bank.logicalLen`.
 */
export function commitChunkToKvBank(
  bank: DoryKvBank,
  startPos: number,
  currentK: Float32Array,
  currentV: Float32Array,
  chunkLen: number,
  writerLoop: number,
): void {
  if (!Number.isInteger(chunkLen) || chunkLen < 1) {
    throw new RangeError(`chunkLen must be an integer >= 1; got ${chunkLen}.`);
  }
  if (!Number.isInteger(startPos) || startPos < 0) {
    throw new RangeError(`startPos must be an integer >= 0; got ${startPos}.`);
  }
  if (startPos !== bank.logicalLen) {
    throw new RangeError(
      `Non-contiguous KV commit on bankSlotId=${bank.bankSlotId} (p=${bank.physicalLayerIdx}): startPos=${startPos} !== bank.logicalLen=${bank.logicalLen}.`,
    );
  }
  if (!Number.isInteger(writerLoop) || writerLoop < 0) {
    throw new RangeError(`writerLoop must be a non-negative integer; got ${writerLoop}.`);
  }
  const kvDim = bank.kvDim;
  const expectedElems = chunkLen * kvDim;
  if (currentK.length !== expectedElems || currentV.length !== expectedElems) {
    throw new RangeError(
      `currentK/currentV length mismatch for chunkLen=${chunkLen}, kvDim=${kvDim}: got K=${currentK.length}, V=${currentV.length}, expected ${expectedElems}.`,
    );
  }

  const endPos = startPos + chunkLen;
  if (bank.attentionKind === 'full_attention') {
    if (endPos > bank.capacity) {
      throw new RangeError(
        `Full-attention KV commit overflow on bankSlotId=${bank.bankSlotId} (p=${bank.physicalLayerIdx}): endPos=${endPos} > capacity=${bank.capacity}.`,
      );
    }
    for (let j = 0; j < chunkLen; j += 1) {
      const absPos = startPos + j;
      const srcOff = j * kvDim;
      const dstOff = absPos * kvDim;
      for (let c = 0; c < kvDim; c += 1) {
        bank.k[dstOff + c] = currentK[srcOff + c];
        bank.v[dstOff + c] = currentV[srcOff + c];
      }
      bank.absPosTag[absPos] = absPos;
      bank.writerLoopTag[absPos] = writerLoop;
    }
    bank.logicalLen = endPos;
    return;
  }

  // Sliding-window ring buffer: commit only the newest positions surviving in `bank.capacity` (DR24)
  const commitStart = Math.max(startPos, endPos - bank.capacity);
  for (let absPos = commitStart; absPos < endPos; absPos += 1) {
    const j = absPos - startPos;
    const slot = slotIndexForPos(bank, absPos);
    const srcOff = j * kvDim;
    const dstOff = slot * kvDim;
    for (let c = 0; c < kvDim; c += 1) {
      bank.k[dstOff + c] = currentK[srcOff + c];
      bank.v[dstOff + c] = currentV[srcOff + c];
    }
    bank.absPosTag[slot] = absPos;
    bank.writerLoopTag[slot] = writerLoop;
  }
  bank.logicalLen = endPos;
}

/**
 * Transaction-local `O(1)`-per-bank logical length rewind across every bank in `cache` (`DR19`):
 *   - Verifies no bank is rewound forward past its written `logicalLen`.
 *   - Verifies every SWA bank still retains all required past-visible positions
 *     `[max(0, targetLen - (W - 1)) .. targetLen - 1]` via `absPosTag` (throws if history was
 *     already evicted beyond the `cap_swa = 16` retention horizon).
 *   - Invalidates every physical slot carrying `absPosTag[slot] >= targetLen` (`-1`, zeroing K/V),
 *     sets `bank.logicalLen = targetLen` and `cache.committedLen = targetLen`, and synchronizes
 *     `cache.chunkSizes` so `sum(chunkSizes) === targetLen`.
 */
export function rewindDoryKvCache(cache: DoryKvCache, targetLen: number): void {
  if (!Number.isInteger(targetLen) || targetLen < 0) {
    throw new RangeError(`targetLen must be an integer >= 0; got ${targetLen}.`);
  }

  // Phase 1: Validate all banks before mutating anything (atomic guard)
  for (let b = 0; b < cache.allBanks.length; b += 1) {
    const bank = cache.allBanks[b];
    if (targetLen > bank.logicalLen) {
      throw new RangeError(
        `Cannot rewind bankSlotId=${bank.bankSlotId} (p=${bank.physicalLayerIdx}) forward to targetLen=${targetLen} beyond logicalLen=${bank.logicalLen}.`,
      );
    }
    if (bank.attentionKind === 'sliding_attention' && targetLen > 0) {
      const windowSize = bank.windowSize ?? cache.config.slidingWindow;
      const reqStart = Math.max(0, targetLen - (windowSize - 1));
      for (let absPos = reqStart; absPos < targetLen; absPos += 1) {
        const slot = slotIndexForPos(bank, absPos);
        if (bank.absPosTag[slot] !== absPos) {
          throw new Error(
            `Cannot rewind SWA bankSlotId=${bank.bankSlotId} (p=${bank.physicalLayerIdx}) to targetLen=${targetLen}: required history absPos=${absPos} was evicted from ring slot ${slot} (holds absPosTag=${bank.absPosTag[slot]}).`,
          );
        }
      }
    }
  }

  // Phase 2: Apply rewind and invalidate rolled-back tags cleanly
  for (let b = 0; b < cache.allBanks.length; b += 1) {
    const bank = cache.allBanks[b];
    const kvDim = bank.kvDim;
    for (let slot = 0; slot < bank.capacity; slot += 1) {
      if (bank.absPosTag[slot] >= targetLen) {
        bank.absPosTag[slot] = -1;
        bank.writerLoopTag[slot] = -1;
        const off = slot * kvDim;
        for (let c = 0; c < kvDim; c += 1) {
          bank.k[off + c] = 0;
          bank.v[off + c] = 0;
        }
      }
    }
    bank.logicalLen = targetLen;
  }

  cache.committedLen = targetLen;

  // Synchronize recorded chunkSizes schedule so sum(chunkSizes) === targetLen
  if (targetLen === 0) {
    cache.chunkSizes.length = 0;
    return;
  }
  let acc = 0;
  let keepCount = 0;
  for (let i = 0; i < cache.chunkSizes.length; i += 1) {
    const sz = cache.chunkSizes[i];
    if (acc + sz <= targetLen) {
      acc += sz;
      keepCount = i + 1;
    } else {
      const remainder = targetLen - acc;
      if (remainder > 0) {
        cache.chunkSizes[i] = remainder;
        acc = targetLen;
        keepCount = i + 1;
      }
      break;
    }
  }
  cache.chunkSizes.length = keepCount;
  if (acc < targetLen) {
    cache.chunkSizes.push(targetLen - acc);
  }
}

/**
 * Reports exact allocated byte usage across all KV banks in `cache`, counting unique
 * underlying `ArrayBuffer`s to verify memory ceilings and non-aliasing invariants (`D10`).
 */
export function measureDoryKvCacheBytes(cache: DoryKvCache): DoryKvCacheByteUsage {
  let kvPayloadBytes = 0;
  let tagMetadataBytes = 0;
  const uniqueBuffers = new Set<ArrayBufferLike>();

  for (let b = 0; b < cache.allBanks.length; b += 1) {
    const bank = cache.allBanks[b];
    kvPayloadBytes += bank.k.byteLength + bank.v.byteLength;
    tagMetadataBytes += bank.absPosTag.byteLength + bank.writerLoopTag.byteLength;
    uniqueBuffers.add(bank.k.buffer);
    uniqueBuffers.add(bank.v.buffer);
    uniqueBuffers.add(bank.absPosTag.buffer);
    uniqueBuffers.add(bank.writerLoopTag.buffer);
  }

  let totalUniqueBufferBytes = 0;
  uniqueBuffers.forEach((buf) => {
    totalUniqueBufferBytes += buf.byteLength;
  });

  return {
    bankCount: cache.allBanks.length,
    kvPayloadBytes,
    tagMetadataBytes,
    totalUniqueBufferBytes,
  };
}
