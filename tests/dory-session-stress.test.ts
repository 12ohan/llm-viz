/**
 * Bounded <= 12.00 MiB `DorySession` Stress, `DR18` Immutability, `D10` LRU Pool &
 * `0-ULP` Lazy Reconstruction (`last_loop` `chunkSizes[]` Replay) Suite
 * (`tests/dory-session-stress.test.ts`, DORY2 Slice 2C).
 *
 * Enforces every `DORY2` Gate 2 session invariant (`D6`, `D10`, `DR18`, `DR19`, `DR23`, `DR24`):
 *   1. Worst-case 12-branch / 128-record stress at BOTH:
 *      - `k = 4` (Contract A `104`-bank `self_spec` heaviest `k = 4` regime) `<= 12.00 MiB`, `< 2,000 ms`,
 *      - `k = 8` (`176`-bank `per_loop` extrapolation regime) `<= 12.00 MiB`, `< 2,000 ms`,
 *      verifying deduplicated `Set<ArrayBufferLike>` bytes, branch/record cap eviction, `D10`
 *      `3.00 MiB` LRU pool ceiling + eviction firing, and strictly non-aliased KV slabs.
 *   2. `DR18` / `D6` mid-session depth (`kLoops`) & KV-policy (`per_loop <-> last_loop`)
 *      immutable fork/switch guard + `DR23` `self_spec` guard + rewound copy-on-step isolation.
 *   3. `D10` `3.00 MiB` (`3,145,728 B`) byte-budgeted LRU sparse-checkpoint pool behavior
 *      (stride insertion, LRU touch/hit promotion, oldest-unused eviction, miss + cold replay).
 *   4. `0-ULP` lazy step-trace reconstruction (`reconstructStepTrace` & `reconstructKvCacheAtRecord`)
 *      INCLUDING verbatim `last_loop` `chunkSizes[]` replay (`DR24`) with a negative proof that
 *      naive coalescing diverges `> 1e-2`, plus `0-ULP` Contract A `spec_prefill`/`spec_cycle`
 *      reconstruction across both LRU hit and LRU eviction!
 *   5. View 2 read-only `Diagnostic Counterfactual` intermediate-loop Coda (`writeKv: false`)
 *      with zero KV / session / LRU mutation.
 */

import { describe, expect, it } from 'vitest';

import {
  assertBitExactFiniteArray,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';
import {
  DORY_CHECKPOINT_LRU_POOL_BYTES,
  DORY_COUNTERFACTUAL_CODA_BADGE,
  DORY_DEFAULT_SESSION_BOUNDS,
  DORY_MAX_SESSION_RETAINED_BYTES,
  DORY_MINI_CONFIG,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  DorySessionImmutableConfigError,
  buildDoryDualRopeTables,
  createDoryKvCache,
  createDorySession,
  getDefaultDoryModelParameters,
  measureDoryKvCacheBytes,
  measureDorySessionMemory,
  runDoryForwardChunk,
  type DoryKvCache,
} from '../src/dory';

const params = getDefaultDoryModelParameters();
const ropeTables = buildDoryDualRopeTables(params.config);

function snapshotKvBankBuffers(kv: DoryKvCache): {
  readonly kCopies: Float32Array[];
  readonly vCopies: Float32Array[];
  readonly posCopies: Int32Array[];
  readonly loopCopies: Int32Array[];
} {
  return {
    kCopies: kv.allBanks.map((b) => new Float32Array(b.k)),
    vCopies: kv.allBanks.map((b) => new Float32Array(b.v)),
    posCopies: kv.allBanks.map((b) => new Int32Array(b.absPosTag)),
    loopCopies: kv.allBanks.map((b) => new Int32Array(b.writerLoopTag)),
  };
}

function assertKvCachesBitExact(actual: DoryKvCache, expected: DoryKvCache, label: string): void {
  expect(actual.mode).toBe(expected.mode);
  expect(actual.kLoops).toBe(expected.kLoops);
  expect(actual.includeShallowCoda).toBe(expected.includeShallowCoda);
  expect(actual.committedLen).toBe(expected.committedLen);
  expect(actual.chunkSizes).toEqual(expected.chunkSizes);
  expect(actual.allBanks.length).toBe(expected.allBanks.length);
  for (let b = 0; b < expected.allBanks.length; b += 1) {
    const ba = actual.allBanks[b];
    const be = expected.allBanks[b];
    expect(ba.logicalLen).toBe(be.logicalLen);
    // Also prove buffers are non-aliased!
    expect(ba.k.buffer).not.toBe(be.k.buffer);
    expect(ba.v.buffer).not.toBe(be.v.buffer);
    assertBitExactFiniteArray(ba.k, be.k, `${label}.bank[${b}].k`);
    assertBitExactFiniteArray(ba.v, be.v, `${label}.bank[${b}].v`);
    assertBitExactFiniteArray(ba.absPosTag, be.absPosTag, `${label}.bank[${b}].absPosTag`);
    assertBitExactFiniteArray(ba.writerLoopTag, be.writerLoopTag, `${label}.bank[${b}].writerLoopTag`);
  }
}

describe('Dory1.0-4B DorySession — Bounded <= 12 MiB Stress (k=4 & k=8), DR18, D10 LRU & 0-ULP Replay', () => {
  it('survives worst-case 12-branch / 128-record Contract A k = 4 (104-bank) stress <= 12.00 MiB and < 2,000 ms', () => {
    const t0 = performance.now();
    const session = createDorySession({
      params,
      ropeTables,
      autoStartInitialBranch: false,
    });

    // Spawn 14 Contract A self_spec branches (each owns the heaviest k = 4 KV: 104 banks = 260 KiB
    // payload + tags) so branch eviction fires cleanly (14 -> 12 surviving branches).
    // Mix full gamma = 4 cycles and gamma = 1 cycles (all 104 banks!) so the 12 surviving branches
    // saturate >= 140 records -> capped strictly at 128, while firing D10 3.00 MiB LRU evictions!
    const createdBranchIds: string[] = [];

    for (let b = 0; b < 14; b += 1) {
      const prompt = [1, 10, 4 + (b % 5)];
      const useFullGamma = b < 4;
      const prefillRec = session.startBranch({
        label: `spec-k4-b${b}`,
        promptTokens: prompt,
        executionMode: 'self_spec',
        kvCacheMode: 'per_loop',
        kLoops: 4,
        speculativeGamma: useFullGamma ? 4 : 1,
        sampling: { temperature: 0.9, topK: 0, topP: 1.0 },
        seed: 0xd0414000 + b,
      });
      createdBranchIds.push(prefillRec.branchId);

      // Advance each branch cleanly without exceeding S_max = 32 or stepping past EOS
      const targetSteps = useFullGamma ? 6 : 13;
      for (let s = 0; s < targetSteps; s += 1) {
        const curBranch = session.getBranch(prefillRec.branchId);
        if (
          curBranch === null ||
          curBranch.haltReason !== null ||
          curBranch.committedLen >= DORY_MINI_CONFIG.maxPositionEmbeddings
        ) {
          break;
        }
        session.step({ branchId: prefillRec.branchId });
      }
    }

    // Top up surviving branches if any halted early on random EOS so record cap (128) is strictly hit
    const liveBranches = session.listBranches();
    let safetyIters = 0;
    while (session.getSnapshot().recordCount < DORY_DEFAULT_SESSION_BOUNDS.maxRecords && safetyIters < 80) {
      safetyIters += 1;
      let steppedAny = false;
      for (const br of liveBranches) {
        const freshBr = session.getBranch(br.branchId);
        if (
          freshBr !== null &&
          freshBr.haltReason === null &&
          freshBr.committedLen < DORY_MINI_CONFIG.maxPositionEmbeddings
        ) {
          session.step({ branchId: br.branchId });
          steppedAny = true;
          if (session.getSnapshot().recordCount >= DORY_DEFAULT_SESSION_BOUNDS.maxRecords) {
            break;
          }
        }
      }
      if (!steppedAny) {
        break;
      }
    }

    const elapsedMs = performance.now() - t0;
    expect(elapsedMs).toBeLessThan(2_000);

    // Oldest 2 branches (b0, b1) must have been evicted cleanly
    expect(session.getBranch(createdBranchIds[0])).toBeNull();
    expect(session.getBranch(createdBranchIds[1])).toBeNull();

    const snap = session.getSnapshot();
    const audit = session.measureMemory();
    expect(measureDorySessionMemory(session)).toEqual(audit);

    expect(snap.branches.length).toBe(12);
    expect(audit.branchCount).toBe(12);
    expect(audit.recordCount).toBe(128);
    expect(snap.recordCount).toBe(128);

    // Every surviving branch owns exact 104 Contract A banks
    for (const br of snap.branches) {
      expect(br.kvBankCount).toBe(DORY_SPEC_CONTRACT_A_BANK_COUNT);
    }

    // D10 3.00 MiB LRU checkpoint pool cap + eviction verification
    expect(audit.maxCheckpointPoolBytes).toBe(DORY_CHECKPOINT_LRU_POOL_BYTES);
    expect(audit.checkpointPoolBytes).toBeLessThanOrEqual(DORY_CHECKPOINT_LRU_POOL_BYTES);
    expect(audit.withinCheckpointPoolBudget).toBe(true);
    expect(snap.checkpointStats.usedBytes).toBe(audit.checkpointPoolBytes);
    expect(snap.checkpointStats.evictionCount).toBeGreaterThan(0);

    // Strict <= 12.00 MiB total budget (both session-only AND including singleton weights)
    expect(audit.maxSessionRetainedBytes).toBe(DORY_MAX_SESSION_RETAINED_BYTES);
    expect(audit.totalSessionRetainedBytes).toBeLessThanOrEqual(DORY_MAX_SESSION_RETAINED_BYTES);
    expect(audit.totalWithWeightsBytes).toBeLessThanOrEqual(DORY_MAX_SESSION_RETAINED_BYTES);
    expect(audit.withinSessionBudget).toBe(true);
    expect(audit.withinTotalBudgetWithWeights).toBe(true);
    expect(audit.nonAliasedSlabsVerified).toBe(true);
  });

  it('survives worst-case 12-branch / 128-record k = 8 extrapolation (176-bank) stress <= 12.00 MiB and < 2,000 ms', () => {
    const t0 = performance.now();
    const session = createDorySession({
      params,
      ropeTables,
      autoStartInitialBranch: false,
    });

    // Spawn 14 per_loop k = 8 branches (176 banks = 440 KiB payload + 27.5 KiB tags per cache!),
    // saturating BOTH branch eviction (14 -> 12), record eviction (> 140 -> 128), and heavy D10
    // 3.00 MiB LRU checkpoint eviction (3.00 MiB holds at most 6 k = 8 checkpoints!).
    const createdBranchIds: string[] = [];

    for (let b = 0; b < 14; b += 1) {
      const prompt = [1, 10, 4 + (b % 6)];
      const prefillRec = session.startBranch({
        label: `extrap-k8-b${b}`,
        promptTokens: prompt,
        executionMode: 'standard',
        kvCacheMode: 'per_loop',
        kLoops: 8,
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
        seed: 0xd0418000 + b,
      });
      createdBranchIds.push(prefillRec.branchId);

      // 11 decode steps + 1 prefill = 12 records/branch -> 12 * 12 = 144 > 128 records!
      for (let s = 0; s < 11; s += 1) {
        const cur = session.getBranch(prefillRec.branchId);
        if (
          cur === null ||
          cur.haltReason !== null ||
          cur.committedLen >= DORY_MINI_CONFIG.maxPositionEmbeddings
        ) {
          break;
        }
        session.step({ branchId: prefillRec.branchId });
      }
    }

    const elapsedMs = performance.now() - t0;
    expect(elapsedMs).toBeLessThan(2_000);

    // Oldest 2 branches evicted
    expect(session.getBranch(createdBranchIds[0])).toBeNull();
    expect(session.getBranch(createdBranchIds[1])).toBeNull();

    const snap = session.getSnapshot();
    const audit = session.measureMemory();
    expect(measureDorySessionMemory(session)).toEqual(audit);

    expect(snap.branches.length).toBe(12);
    expect(audit.branchCount).toBe(12);
    expect(audit.recordCount).toBe(128);

    // Every surviving k = 8 per_loop branch owns exact 176 banks (16 + 20 * 8)
    for (const br of snap.branches) {
      expect(br.kvBankCount).toBe(176);
    }

    // D10 3.00 MiB LRU cap holds <= 6 k = 8 checkpoints and fired evictions
    expect(audit.checkpointCount).toBeLessThanOrEqual(6);
    expect(audit.checkpointPoolBytes).toBeLessThanOrEqual(DORY_CHECKPOINT_LRU_POOL_BYTES);
    expect(audit.withinCheckpointPoolBudget).toBe(true);
    expect(snap.checkpointStats.evictionCount).toBeGreaterThan(0);

    // Strict <= 12.00 MiB ceiling even at k = 8 (including singleton weights!)
    expect(audit.totalSessionRetainedBytes).toBeLessThanOrEqual(DORY_MAX_SESSION_RETAINED_BYTES);
    expect(audit.totalWithWeightsBytes).toBeLessThanOrEqual(DORY_MAX_SESSION_RETAINED_BYTES);
    expect(audit.withinSessionBudget).toBe(true);
    expect(audit.withinTotalBudgetWithWeights).toBe(true);
    expect(audit.nonAliasedSlabsVerified).toBe(true);
  });

  it('enforces D10 3.00 MiB LRU checkpoint pool hit promotion, least-recently-used eviction, and cold replay', () => {
    // Use checkpointStride = 1 and a tight pool cap equal to exactly 2 k = 2 per_loop caches
    // so we can deterministically prove LRU promotion (touching an older entry spares it from eviction!).
    const singleK2Kv = createDoryKvCache({
      mode: 'per_loop',
      kLoops: 2,
      includeShallowCoda: false,
      config: params.config,
    });
    const k2CacheBytes = measureDoryKvCacheBytes(singleK2Kv).totalUniqueBufferBytes;

    const session = createDorySession({
      params,
      ropeTables,
      bounds: {
        maxBranches: 12,
        maxRecords: 128,
        maxCheckpointPoolBytes: k2CacheBytes * 2,
        maxSessionRetainedBytes: DORY_MAX_SESSION_RETAINED_BYTES,
        checkpointStride: 1,
      },
      initialBranch: {
        promptTokens: [1, 10, 4, 18],
        kLoops: 2,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });

    const r0 = session.getActiveRecord()!;
    const r1 = session.step();
    // Pool now holds [r0, r1] (2 entries = exact pool budget)
    let stats = session.getCheckpointLruStats();
    expect(stats.entryCount).toBe(2);
    expect(stats.usedBytes).toBe(k2CacheBytes * 2);
    expect(stats.evictionCount).toBe(0);

    // Touch r0 via reconstructKvCacheAtRecord -> promotes r0 above r1 in LRU order!
    const kvR0Hit = session.reconstructKvCacheAtRecord(r0.recordId);
    expect(kvR0Hit.committedLen).toBe(r0.committedLen);
    stats = session.getCheckpointLruStats();
    expect(stats.hitCount).toBe(1);

    // Step r2 -> pool must evict r1 (least recently used) and KEEP promoted r0!
    const r2 = session.step();
    stats = session.getCheckpointLruStats();
    expect(stats.entryCount).toBe(2);
    expect(stats.evictionCount).toBe(1);
    expect(stats.usedBytes).toBeLessThanOrEqual(k2CacheBytes * 2);
    const remainingKeys = stats.entries.map((e) => e.recordId);
    expect(remainingKeys).toContain(r0.recordId);
    expect(remainingKeys).toContain(r2.recordId);
    expect(remainingKeys).not.toContain(r1.recordId);

    // Reconstructing evicted r1 hits r0 as prefix (1 chunk replayed) at 0-ULP!
    const reconR1 = session.reconstructStepTrace(r1.recordId);
    expect(reconR1.usedCheckpoint).toBe(true);
    expect(reconR1.checkpointHitRecordId).toBe(r0.recordId);
    expect(reconR1.forwardResult).not.toBeNull();
    assertBitExactFiniteArray(
      reconR1.forwardResult!.scaledLogits,
      r1.lastTokenScaledLogits,
      'd10.lru.reconR1.scaledLogits',
    );

    // Verify cold miss replay when maxCheckpointPoolBytes = 0
    const coldSession = createDorySession({
      params,
      ropeTables,
      bounds: { maxCheckpointPoolBytes: 0 },
      initialBranch: {
        promptTokens: [1, 10, 4, 18],
        kLoops: 2,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
      },
    });
    coldSession.step();
    const coldR2 = coldSession.step();
    const coldRecon = coldSession.reconstructStepTrace(coldR2.recordId);
    expect(coldRecon.usedCheckpoint).toBe(false);
    expect(coldRecon.checkpointHitRecordId).toBeNull();
    expect(coldSession.getCheckpointLruStats().missCount).toBe(1);
    assertBitExactFiniteArray(
      coldRecon.forwardResult!.scaledLogits,
      coldR2.lastTokenScaledLogits,
      'd10.cold.reconR2.scaledLogits',
    );
  });

  it('enforces DR18 immutable branch/fork/switch guards, DR23 self_spec guards, and non-aliased CoW copy-on-step', () => {
    const session = createDorySession({
      params,
      ropeTables,
      initialBranch: {
        label: 'root-k2-per-loop',
        promptTokens: [1, 10, 4, 18, 40, 25],
        kLoops: 2,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
        seed: 0xd0411801,
      },
    });

    const r0 = session.getActiveRecord()!;
    const r1 = session.step();
    const r2 = session.step();
    const rootBranchId = r0.branchId;

    // Capture exact pre-switch snapshot & KV contents of root branch
    const rootSnapBefore = session.getBranch(rootBranchId)!;
    const rootKvBefore = session.reconstructKvCacheAtRecord(r2.recordId);
    const rootKvDump = snapshotKvBankBuffers(rootKvBefore);

    // 1. assertCompatibleBranchConfig throws DR18_INCOMPATIBLE_BRANCH_CONFIG on any mismatch
    expect(() => session.assertCompatibleBranchConfig(rootBranchId, { kLoops: 4 })).toThrow(
      DorySessionImmutableConfigError,
    );
    expect(() =>
      session.assertCompatibleBranchConfig(rootBranchId, { kvCacheMode: 'last_loop' }),
    ).toThrow(DorySessionImmutableConfigError);
    expect(() =>
      session.assertCompatibleBranchConfig(rootBranchId, { executionMode: 'self_spec' }),
    ).toThrow(DorySessionImmutableConfigError);
    try {
      session.assertCompatibleBranchConfig(rootBranchId, { kLoops: 4 });
    } catch (err) {
      expect((err as DorySessionImmutableConfigError).code).toBe('DR18_INCOMPATIBLE_BRANCH_CONFIG');
    }

    // 2. forkBranch across incompatible config throws DR18_REPREFILL_REQUIRED unless allowRePrefill: true
    expect(() => session.forkBranch(r2.recordId, { kLoops: 4 })).toThrow(
      DorySessionImmutableConfigError,
    );
    try {
      session.forkBranch(r2.recordId, { kvCacheMode: 'last_loop' });
    } catch (err) {
      expect((err as DorySessionImmutableConfigError).code).toBe('DR18_REPREFILL_REQUIRED');
    }

    // 3. switchDepthOrPolicy (k=2 -> k=4, per_loop -> last_loop, standard -> self_spec) spawns
    //    clean re-prefilled branches without mutating the source branch!
    const switchedK4 = session.switchDepthOrPolicy({
      branchId: rootBranchId,
      kLoops: 4,
      kvCacheMode: 'last_loop',
    });
    expect(switchedK4.branchId).not.toBe(rootBranchId);
    expect(switchedK4.kLoops).toBe(4);
    expect(switchedK4.kvCacheMode).toBe('last_loop');
    expect(session.getBranch(switchedK4.branchId)!.forkReason).toBe('dr18_re_prefill');
    expect(session.getBranch(switchedK4.branchId)!.parentBranchId).toBe(rootBranchId);
    expect(Array.from(switchedK4.committedSequence)).toEqual([1, 10, 4, 18, 40, 25]);

    const switchedSpec = session.switchDepthOrPolicy({
      branchId: rootBranchId,
      executionMode: 'self_spec',
    });
    expect(switchedSpec.executionMode).toBe('self_spec');
    expect(switchedSpec.kLoops).toBe(4);
    expect(switchedSpec.kvCacheMode).toBe('per_loop');
    expect(session.getBranch(switchedSpec.branchId)!.kvBankCount).toBe(
      DORY_SPEC_CONTRACT_A_BANK_COUNT,
    );
    session.step({ branchId: switchedSpec.branchId });

    // 4. Compatible forkBranch + rewound copy-on-step fork leave source branch completely untouched!
    const compatFork = session.forkBranch(r1.recordId, { label: 'compat-fork' });
    expect(compatFork.branchId).not.toBe(rootBranchId);
    expect(session.getBranch(compatFork.branchId)!.forkReason).toBe('cursor_fork');
    session.step({ branchId: compatFork.branchId });

    // Seek to historical record r1 on rootBranch and call step() -> triggers automatic copy-on-step fork!
    session.seekRecord(r1.recordId);
    const cosRec = session.step();
    expect(cosRec.branchId).not.toBe(rootBranchId);
    expect(cosRec.parentRecordId).toBe(r1.recordId);
    expect(session.getBranch(cosRec.branchId)!.forkReason).toBe('cursor_fork');

    // Verify root branch is 100% immutable (snapshot + KV buffers bit-for-bit identical!)
    const rootSnapAfter = session.getBranch(rootBranchId)!;
    expect(rootSnapAfter.committedLen).toBe(rootSnapBefore.committedLen);
    expect(rootSnapAfter.headRecordId).toBe(rootSnapBefore.headRecordId);
    expect(Array.from(rootSnapAfter.committedTokens)).toEqual(
      Array.from(rootSnapBefore.committedTokens),
    );
    expect(rootSnapAfter.chunkSizes).toEqual(rootSnapBefore.chunkSizes);
    const rootKvAfter = session.reconstructKvCacheAtRecord(r2.recordId);
    for (let b = 0; b < rootKvAfter.allBanks.length; b += 1) {
      assertBitExactFiniteArray(rootKvAfter.allBanks[b].k, rootKvDump.kCopies[b], `dr18.root.k[${b}]`);
      assertBitExactFiniteArray(rootKvAfter.allBanks[b].v, rootKvDump.vCopies[b], `dr18.root.v[${b}]`);
      assertBitExactFiniteArray(
        rootKvAfter.allBanks[b].absPosTag,
        rootKvDump.posCopies[b],
        `dr18.root.pos[${b}]`,
      );
      assertBitExactFiniteArray(
        rootKvAfter.allBanks[b].writerLoopTag,
        rootKvDump.loopCopies[b],
        `dr18.root.loop[${b}]`,
      );
    }

    // Zero aliased ArrayBuffers across all branches, checkpoints, records, and weights!
    expect(session.measureMemory().nonAliasedSlabsVerified).toBe(true);

    // 5. DR23 Guards on DorySession
    expect(() =>
      session.startBranch({ executionMode: 'self_spec', kvCacheMode: 'last_loop', kLoops: 4 }),
    ).toThrow(DorySessionImmutableConfigError);
    expect(() =>
      session.startBranch({ executionMode: 'self_spec', kvCacheMode: 'per_loop', kLoops: 2 }),
    ).toThrow(DorySessionImmutableConfigError);
    expect(() =>
      session.appendChunk([20, 21], { branchId: switchedSpec.branchId }),
    ).toThrow(DorySessionImmutableConfigError);
  });

  it('performs 0-ULP lazy step-trace reconstruction INCLUDING verbatim last_loop chunkSizes[] replay (DR24) & Contract A spec cycles', () => {
    // ------------------------------------------------------------------------
    // Part A: `last_loop` k = 2 & k = 4 with non-uniform chunked prefill [4, 3, 5]
    //         + appendChunk([30, 31]) + decode steps [1, 1, 1] -> chunkSizes = [4, 3, 5, 2, 1, 1, 1]
    //         Tested under BOTH checkpoint-hit AND cold-replay (maxCheckpointPoolBytes = 0)!
    // ------------------------------------------------------------------------
    const prompt12 = [1, 10, 4, 18, 40, 25, 26, 57, 20, 21, 22, 23];

    for (const kLoops of [2, 4] as const) {
      for (const poolBytes of [DORY_CHECKPOINT_LRU_POOL_BYTES, 0] as const) {
        const session = createDorySession({
          params,
          ropeTables,
          bounds: {
            maxCheckpointPoolBytes: poolBytes,
            checkpointStride: 3,
          },
          initialBranch: {
            promptTokens: prompt12,
            prefillChunkSizes: [4, 3, 5],
            kLoops,
            kvCacheMode: 'last_loop',
            executionMode: 'standard',
            sampling: { temperature: 0, topK: 0, topP: 1.0 },
          },
        });

        const branchId = session.getActiveBranch()!.branchId;
        const chunkRec = session.appendChunk([30, 31], { branchId });
        const stepRec1 = session.step({ branchId });
        const stepRec2 = session.step({ branchId });
        const stepRec3 = session.step({ branchId });

        expect(stepRec3.chunkSizesBefore).toEqual([4, 3, 5, 2, 1, 1]);
        expect(stepRec3.chunkSizesAfter).toEqual([4, 3, 5, 2, 1, 1, 1]);

        // Reconstruct BOTH the multi-token appendChunk record AND the final decode_step record!
        for (const rec of [chunkRec, stepRec1, stepRec2, stepRec3]) {
          const recon = session.reconstructStepTrace(rec.recordId);
          expect(recon.forwardResult).not.toBeNull();
          expect(recon.forwardResult!.trace).not.toBeNull();
          expect(recon.postStepKvCache.committedLen).toBe(rec.committedLen);
          expect(recon.postStepKvCache.chunkSizes).toEqual(rec.chunkSizesAfter);

          if (poolBytes === 0) {
            expect(recon.usedCheckpoint).toBe(false);
            // Cold replay MUST replay exact chunkSizesBefore verbatim (NEVER coalesced!)
            expect(recon.replayedChunkSizes).toEqual(rec.chunkSizesBefore);
          }

          const V = params.config.vocabSize;
          const d = params.config.hiddenSize;
          const lastOffLogit = (rec.inputTokens.length - 1) * V;
          const lastOffVec = (rec.inputTokens.length - 1) * d;

          // 0-ULP bit-exact match on scaled logits, xAnchor, and ALL k loop entry/exit states!
          assertBitExactFiniteArray(
            recon.forwardResult!.scaledLogits.subarray(lastOffLogit, lastOffLogit + V),
            rec.lastTokenScaledLogits,
            `dr24.k${kLoops}.pool${poolBytes}.${rec.recordId}.scaledLogits`,
          );
          assertBitExactFiniteArray(
            recon.forwardResult!.xAnchor.subarray(lastOffVec, lastOffVec + d),
            rec.lastTokenXAnchor,
            `dr24.k${kLoops}.pool${poolBytes}.${rec.recordId}.xAnchor`,
          );
          for (let l = 0; l < kLoops; l += 1) {
            assertBitExactFiniteArray(
              recon.forwardResult!.loopEntryStates[l].subarray(lastOffVec, lastOffVec + d),
              rec.lastTokenLoopEntryStates[l],
              `dr24.k${kLoops}.pool${poolBytes}.${rec.recordId}.u[${l}]`,
            );
            assertBitExactFiniteArray(
              recon.forwardResult!.loopExitStates[l].subarray(lastOffVec, lastOffVec + d),
              rec.lastTokenLoopExitStates[l],
              `dr24.k${kLoops}.pool${poolBytes}.${rec.recordId}.s[${l}]`,
            );
          }
        }

        // Direct 0-ULP comparison against independent manual chunked replay AND negative proof
        // that naive single-shot coalescing of [4, 3, 5, 2, 1, 1] -> [16] WOULD diverge > 1e-2!
        const manualKv = createDoryKvCache({
          mode: 'last_loop',
          kLoops,
          config: params.config,
        });
        let offset = 0;
        for (const sz of stepRec3.chunkSizesBefore) {
          runDoryForwardChunk({
            tokens: stepRec3.committedSequence.subarray(offset, offset + sz),
            params,
            kvCache: manualKv,
            startPos: offset,
            kLoops,
            kvCacheMode: 'last_loop',
            ropeTables,
          });
          offset += sz;
        }
        const manualStep3 = runDoryForwardChunk({
          tokens: stepRec3.inputTokens,
          params,
          kvCache: manualKv,
          startPos: stepRec3.startPos,
          kLoops,
          kvCacheMode: 'last_loop',
          ropeTables,
        });
        const reconStep3 = session.reconstructStepTrace(stepRec3.recordId);
        assertBitExactFiniteArray(
          reconStep3.forwardResult!.scaledLogits,
          manualStep3.scaledLogits,
          `dr24.k${kLoops}.pool${poolBytes}.vsManualVerbatimReplay`,
        );
        assertKvCachesBitExact(
          reconStep3.postStepKvCache,
          manualKv,
          `dr24.k${kLoops}.pool${poolBytes}.postKv`,
        );

        // Coalesced single-shot prefix negative check: proves chunkSizes[] replay is essential!
        const coalescedKv = createDoryKvCache({
          mode: 'last_loop',
          kLoops,
          config: params.config,
        });
        runDoryForwardChunk({
          tokens: stepRec3.committedSequence.subarray(0, stepRec3.startPos),
          params,
          kvCache: coalescedKv,
          startPos: 0,
          kLoops,
          kvCacheMode: 'last_loop',
          ropeTables,
        });
        const coalescedStep3 = runDoryForwardChunk({
          tokens: stepRec3.inputTokens,
          params,
          kvCache: coalescedKv,
          startPos: stepRec3.startPos,
          kLoops,
          kvCacheMode: 'last_loop',
          ropeTables,
        });
        const coalesceDivergence = computeStrictMaxAbsDiff(
          coalescedStep3.scaledLogits,
          stepRec3.lastTokenScaledLogits,
          `dr24.k${kLoops}.coalesceDivergence`,
        );
        expect(coalesceDivergence).toBeGreaterThan(1e-2);
      }
    }

    // ------------------------------------------------------------------------
    // Part B: 0-ULP lazy reconstruction of Contract A `spec_prefill` & `spec_cycle`
    //         under BOTH checkpoint-hit and cold-replay (`maxCheckpointPoolBytes = 0`)
    // ------------------------------------------------------------------------
    for (const poolBytes of [DORY_CHECKPOINT_LRU_POOL_BYTES, 0] as const) {
      const specSession = createDorySession({
        params,
        ropeTables,
        bounds: { maxCheckpointPoolBytes: poolBytes, checkpointStride: 2 },
        initialBranch: {
          promptTokens: [1, 10, 4, 18, 40, 25, 26, 57],
          executionMode: 'self_spec',
          kvCacheMode: 'per_loop',
          kLoops: 4,
          speculativeGamma: 4,
          sampling: { temperature: 1.0, topK: 0, topP: 1.0 },
          seed: 0xd0415555,
        },
      });

      const specPrefillRec = specSession.getActiveRecord()!;
      const specCycle1 = specSession.step();
      const specCycle2 = specSession.step();
      const specCycle3 = specSession.step();

      // Reconstruct spec_prefill at 0-ULP
      const reconPrefill = specSession.reconstructStepTrace(specPrefillRec.recordId);
      expect(reconPrefill.specPrefillResult).not.toBeNull();
      expect(reconPrefill.specPrefillResult!.deepTrace).not.toBeNull();
      expect(reconPrefill.specPrefillResult!.shallowCodaTrace).not.toBeNull();
      assertBitExactFiniteArray(
        reconPrefill.specPrefillResult!.lastDeepScaledLogits,
        specPrefillRec.lastTokenScaledLogits,
        `spec.recon.pool${poolBytes}.prefill.logits`,
      );

      // Reconstruct every spec_cycle at 0-ULP (including trace, emitted tokens, logits, states!)
      for (const rec of [specCycle1, specCycle2, specCycle3]) {
        const recon = specSession.reconstructStepTrace(rec.recordId);
        expect(recon.specCycleResult).not.toBeNull();
        expect(recon.specCycleResult!.verifyTrace).not.toBeNull();
        expect(recon.specCycleResult!.verifyTrace!.executedSublayerCount).toBe(192);
        expect(recon.postStepKvCache.allBanks.length).toBe(DORY_SPEC_CONTRACT_A_BANK_COUNT);
        expect(recon.postStepKvCache.committedLen).toBe(rec.committedLen);
        expect(recon.specCycleResult!.outcome).toBe(rec.specSummary!.outcome);
        expect(recon.specCycleResult!.acceptedDraftCount).toBe(rec.specSummary!.acceptedDraftCount);
        expect(Array.from(recon.specCycleResult!.emittedTokens)).toEqual(
          Array.from(rec.emittedTokens),
        );
        assertBitExactFiniteArray(
          recon.specCycleResult!.lastCommittedDeepScaledLogits,
          rec.lastTokenScaledLogits,
          `spec.recon.pool${poolBytes}.${rec.recordId}.logits`,
        );
        assertBitExactFiniteArray(
          recon.specCycleResult!.lastCommittedXAnchor,
          rec.lastTokenXAnchor,
          `spec.recon.pool${poolBytes}.${rec.recordId}.xAnchor`,
        );
        for (let l = 0; l < 4; l += 1) {
          assertBitExactFiniteArray(
            recon.specCycleResult!.lastCommittedLoopExitStates[l],
            rec.lastTokenLoopExitStates[l],
            `spec.recon.pool${poolBytes}.${rec.recordId}.s[${l}]`,
          );
        }
      }
    }
  });

  it('evaluates View 2 read-only Diagnostic Counterfactual intermediate Coda with 0 KV/session/LRU mutation', () => {
    const session = createDorySession({
      params,
      ropeTables,
      initialBranch: {
        promptTokens: [1, 10, 4, 18, 40, 25, 26, 57],
        kLoops: 4,
        kvCacheMode: 'per_loop',
        executionMode: 'standard',
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
      },
    });
    const rec = session.step();

    const snapBefore = session.getSnapshot();
    const auditBefore = session.measureMemory();
    const lruBefore = session.getCheckpointLruStats();
    const kvBefore = session.reconstructKvCacheAtRecord(rec.recordId);
    const kvBeforeDump = snapshotKvBankBuffers(kvBefore);
    const lruAfterBaselineRead = session.getCheckpointLruStats();

    // Evaluate all intermediate loops t = 1..4
    const cf1 = session.evaluateCounterfactualIntermediateCoda(rec.recordId, 1);
    const cf2 = session.evaluateCounterfactualIntermediateCoda(rec.recordId, 2);
    const cf3 = session.evaluateCounterfactualIntermediateCoda(rec.recordId, 3);
    const cf4 = session.evaluateCounterfactualIntermediateCoda(rec.recordId, 4);

    for (const [idx, cf] of [cf1, cf2, cf3, cf4].entries()) {
      expect(cf.badge).toBe(DORY_COUNTERFACTUAL_CODA_BADGE);
      expect(cf.recordId).toBe(rec.recordId);
      expect(cf.loopIteration).toBe(idx + 1);
      expect(cf.kLoops).toBe(4);
      expect(cf.absTokenPos).toBe(rec.committedLen - 1);
    }

    // Final loop t = 4 MUST match the record's actual Deep Coda logits at 0-ULP!
    assertBitExactFiniteArray(cf4.scaledLogits, rec.lastTokenScaledLogits, 'counterfactual.t4.logits');

    // Intermediate loop t = 1 differs materially from final loop t = 4
    expect(computeStrictMaxAbsDiff(cf1.scaledLogits, cf4.scaledLogits, 'counterfactual.t1VsT4')).toBeGreaterThan(
      1e-3,
    );

    // Invalid loopIteration throws RangeError
    expect(() => session.evaluateCounterfactualIntermediateCoda(rec.recordId, 0)).toThrow(RangeError);
    expect(() => session.evaluateCounterfactualIntermediateCoda(rec.recordId, 5)).toThrow(RangeError);

    // Zero mutation to session memory, LRU counters, or branch KV buffers!
    expect(session.measureMemory()).toEqual(auditBefore);
    expect(session.getCheckpointLruStats()).toEqual(lruAfterBaselineRead);
    expect(session.getSnapshot().activeRecordId).toBe(snapBefore.activeRecordId);
    expect(lruBefore.evictionCount).toBe(session.getCheckpointLruStats().evictionCount);

    const kvAfter = session.reconstructKvCacheAtRecord(rec.recordId);
    for (let b = 0; b < kvAfter.allBanks.length; b += 1) {
      assertBitExactFiniteArray(kvAfter.allBanks[b].k, kvBeforeDump.kCopies[b], `cf.kv.k[${b}]`);
      assertBitExactFiniteArray(kvAfter.allBanks[b].v, kvBeforeDump.vCopies[b], `cf.kv.v[${b}]`);
    }
  });

  it('enforces DR19 maxSafeGamma = cap_swa - W - 1 (7) at branch creation and leaves zero zombie branches on throwing rewound specExplicitVariates', () => {
    const session = createDorySession({
      params,
      ropeTables,
      autoStartInitialBranch: false,
    });

    // Cleanup 1: gamma = 8 violates DR19 (W=8 + gamma=8 + 1 = 17 > cap_swa=16) and MUST fail at startBranch()
    expect(() =>
      session.startBranch({
        executionMode: 'self_spec',
        kvCacheMode: 'per_loop',
        kLoops: 4,
        speculativeGamma: 8,
      }),
    ).toThrow(RangeError);
    expect(session.listBranches().length).toBe(0);

    // gamma = 7 satisfies DR19 (8 + 7 + 1 = 16 <= 16) and executes cleanly
    const maxSafeRec = session.startBranch({
      promptTokens: [1, 10, 4, 18],
      executionMode: 'self_spec',
      kvCacheMode: 'per_loop',
      kLoops: 4,
      speculativeGamma: 7,
      sampling: { temperature: 1.0, topK: 0, topP: 1.0 },
      seed: 0xd0419901,
    });
    const steppedSafe = session.step({ branchId: maxSafeRec.branchId });
    expect(steppedSafe.specSummary).not.toBeNull();

    // Cleanup 2: Rewind to maxSafeRec and pass invalid specExplicitVariates -> throws RangeError
    // and leaves ZERO zombie recordless branch or leaked KV memory in branchesById!
    session.seekRecord(maxSafeRec.recordId);
    const branchesBefore = session.listBranches().length;
    const memBefore = session.measureMemory();

    expect(() =>
      session.step({
        specExplicitVariates: {
          draftSampleUniforms: [1.5], // invalid uniform >= 1
        },
      }),
    ).toThrow(RangeError);

    expect(() =>
      session.step({
        specExplicitVariates: {
          acceptUniforms: [-0.01], // invalid uniform < 0
        },
      }),
    ).toThrow(RangeError);

    expect(() =>
      session.step({
        specExplicitVariates: {
          correctionOrBonusUniform: Number.NaN,
        },
      }),
    ).toThrow(RangeError);

    expect(session.listBranches().length).toBe(branchesBefore);
    for (const br of session.listBranches()) {
      expect(br.headRecordId).not.toBe('');
      expect(br.recordIds.length).toBeGreaterThan(0);
    }
    expect(session.measureMemory()).toEqual(memBefore);
  });
});
