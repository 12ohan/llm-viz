import { describe, expect, it } from 'vitest';

import {
  assertKvSnapshotIntegrity,
  createDefaultDiffusionModel,
  createDiffusionSession,
  DIFFUSION_MINI_CONFIG,
  KvIntegrityError,
  KvProvenanceMismatchError,
  runDiffusionExecution,
  runDiffusionForwardPass,
  verifyKvSnapshotIntegrity,
  type DiffusionExecutionMode,
} from '../src/diffusion';
import { assertBitExactFiniteArray } from './helpers/parityAssert';

describe('DiffusionSession — Step-by-Step Parity, Copy-on-Step Branching, Provenance & 0-ULP Reconstruction', () => {
  const model = createDefaultDiffusionModel();
  const prompt = [0, 1, 2, 3, 4, 5, 6, 7];

  it('matches runDiffusionExecution at 0-ULP across all 4 modes + dlm causalContext=false', () => {
    const cases: Array<{
      readonly label: string;
      readonly mode: DiffusionExecutionMode;
      readonly causalContext: boolean;
      readonly temperature: number;
      readonly samplerAThreshold: number | null;
      readonly maxNewTokens: number;
    }> = [
      { label: 'ar-t0', mode: 'ar', causalContext: true, temperature: 0, samplerAThreshold: null, maxNewTokens: 8 },
      { label: 'dlm-causal-t0', mode: 'dlm', causalContext: true, temperature: 0, samplerAThreshold: 0.22, maxNewTokens: 8 },
      { label: 'dlm-bidir-t0', mode: 'dlm', causalContext: false, temperature: 0, samplerAThreshold: null, maxNewTokens: 8 },
      { label: 'spec-t0', mode: 'linear_spec', causalContext: true, temperature: 0, samplerAThreshold: null, maxNewTokens: 8 },
      { label: 'spec-lora-t07', mode: 'linear_spec_lora', causalContext: true, temperature: 0.7, samplerAThreshold: null, maxNewTokens: 8 },
    ];

    for (const c of cases) {
      const exec = runDiffusionExecution(model, {
        mode: c.mode,
        causalContext: c.causalContext,
        promptTokenIds: prompt,
        blockLength: 4,
        maxNewTokens: c.maxNewTokens,
        temperature: c.temperature,
        samplerAThreshold: c.samplerAThreshold,
        samplerBThreshold: 0.0,
        samplingSeed: 0x4e4c4453,
      });

      const session = createDiffusionSession(model, {
        promptTokenIds: prompt,
        config: {
          mode: c.mode,
          causalContext: c.causalContext,
          blockLength: 4,
          maxNewTokens: c.maxNewTokens,
          temperature: c.temperature,
          samplerAThreshold: c.samplerAThreshold,
          samplerBThreshold: 0.0,
          samplingSeed: 0x4e4c4453,
        },
      });
      const records = session.runToCompletion();
      const tip = records[records.length - 1]!;

      expect(Array.from(tip.generatedTokenIds), `${c.label}.tokens`).toEqual(
        Array.from(exec.generatedTokenIds),
      );
      expect(tip.lifecyclePhase, `${c.label}.phase`).toBe(exec.lifecyclePhase);
      expect(tip.kvMaskProvenance, `${c.label}.prov`).toBe(exec.kvMaskProvenance);
      expect(tip.counters, `${c.label}.counters`).toEqual(exec.counters);
      expect(tip.kvSnapshot.seqLen, `${c.label}.kvLen`).toBe(exec.finalKvSnapshot.seqLen);

      for (let pos = 0; pos < exec.finalKvSnapshot.seqLen; pos++) {
        assertBitExactFiniteArray(
          tip.kvSnapshot.slabs[pos]!.k,
          exec.finalKvSnapshot.slabs[pos]!.k,
          `${c.label}.slab${pos}.k`,
        );
        assertBitExactFiniteArray(
          tip.kvSnapshot.slabs[pos]!.v,
          exec.finalKvSnapshot.slabs[pos]!.v,
          `${c.label}.slab${pos}.v`,
        );
      }
    }
  });

  it('supports copy-on-step branching, non-retroactive updateConfig, seekRecord, stepBack, stepForward, deleteBranch, and reset', () => {
    const session = createDiffusionSession(model, {
      promptTokenIds: prompt,
      config: { mode: 'ar', maxNewTokens: 10, temperature: 0 },
    });

    const r0 = session.step(); // prefill (1 token)
    const r1 = session.step(); // ar_step (2 tokens)
    const r2 = session.step(); // ar_step (3 tokens)
    const r3 = session.step(); // ar_step (4 tokens)
    expect(session.getBranches().length).toBe(1);
    const branch1Id = session.getState().activeBranchId;

    // Seek back to r1 and verify stepBack / stepForward navigation
    const back = session.stepBack();
    expect(back?.recordId).toBe(r2.recordId);
    const fwd = session.stepForward();
    expect(fwd?.recordId).toBe(r3.recordId);
    expect(session.stepForward()).toBeNull();

    const rewound = session.seekRecord(r1.recordId);
    expect(rewound.recordId).toBe(r1.recordId);
    expect(session.stepBack()?.recordId).toBe(r0.recordId);
    session.seekRecord(r1.recordId);

    // Non-retroactive updateConfig and copy-on-step fork into linear_spec_lora
    session.updateConfig({ mode: 'linear_spec_lora', blockLength: 4, maxNewTokens: 12 });
    const forkedStep = session.step();
    expect(session.getBranches().length).toBe(2);
    expect(forkedStep.branchId).not.toBe(branch1Id);
    expect(forkedStep.parentRecordId).toBe(r1.recordId);
    expect(forkedStep.kind).toBe('spec_cycle');

    // Original branch records & configAtStep remain untouched (non-retroactive)
    expect(session.getRecord(r2.recordId)!.configAtStep.mode).toBe('ar');
    expect(session.getRecord(r3.recordId)!.configAtStep.mode).toBe('ar');

    // Structural KV prefix sharing: prefix slabs [0..r1.kvSnapshot.seqLen-1] share identical ArrayBuffers
    for (let pos = 0; pos < r1.kvSnapshot.seqLen; pos++) {
      expect(forkedStep.kvSnapshot.slabs[pos]).toBe(r1.kvSnapshot.slabs[pos]);
      expect(forkedStep.kvSnapshot.slabs[pos]!.k.buffer).toBe(r1.kvSnapshot.slabs[pos]!.k.buffer);
    }

    // deleteBranch cannot delete the last remaining branch; deletes branch1 cleanly
    expect(session.deleteBranch(branch1Id)).toBe(true);
    expect(session.getBranches().length).toBe(1);
    expect(session.getRecord(r2.recordId)).toBeNull();
    expect(session.getRecord(r3.recordId)).toBeNull();
    expect(session.deleteBranch(session.getState().activeBranchId)).toBe(false);

    // reset clears tree and starts fresh
    session.reset([1, 2, 3, 4], { mode: 'dlm', causalContext: true, blockLength: 4 });
    expect(session.getActiveRecord()).toBeNull();
    expect(session.getBranches().length).toBe(1);
    expect(session.getState().records.size).toBe(0);
    const freshPrefill = session.step();
    expect(freshPrefill.kvSnapshot.seqLen).toBe(4);
  });

  it('enforces kvMaskProvenance on cross-mode switch unless allowExplicitRePrefill=true', () => {
    const session = createDiffusionSession(model, {
      promptTokenIds: prompt,
      config: { mode: 'dlm', causalContext: false, blockLength: 4, maxNewTokens: 8 },
    });

    const prefill = session.step();
    expect(prefill.kvMaskProvenance).toBe('block_bidirectional');
    const blk0 = session.step();
    expect(blk0.kvMaskProvenance).toBe('block_bidirectional');
    expect(blk0.kvSnapshot.seqLen).toBe(12);

    // Switching directly from block_bidirectional -> token_causal throws KvProvenanceMismatchError
    expect(() => session.updateConfig({ mode: 'ar' })).toThrow(KvProvenanceMismatchError);
    expect(() => session.updateConfig({ mode: 'linear_spec_lora' })).toThrow(
      KvProvenanceMismatchError,
    );
    expect(() => session.updateConfig({ causalContext: true })).toThrow(
      KvProvenanceMismatchError,
    );

    // Explicit re-prefill succeeds and cleanly re-prefills all 12 committed tokens under causal attention
    session.updateConfig({ mode: 'ar', allowExplicitRePrefill: true, maxNewTokens: 12 });
    const rePrefill = session.getActiveRecord()!;
    expect(rePrefill.kind).toBe('prefill');
    expect(rePrefill.kvMaskProvenance).toBe('token_causal');
    expect(rePrefill.kvSnapshot.seqLen).toBe(12);
    expect(rePrefill.guardFlags.warnings.some((w) => w.includes('Explicit re-prefill'))).toBe(true);

    // Subsequent AR step succeeds cleanly with causal KV
    const nextAr = session.step();
    expect(nextAr.kind).toBe('ar_step');
    expect(nextAr.kvMaskProvenance).toBe('token_causal');
  });

  it('detects single-float KV slab tampering via verifyKvSnapshotIntegrity & assertKvSnapshotIntegrity', () => {
    const session = createDiffusionSession(model, {
      promptTokenIds: prompt,
      config: { mode: 'ar', maxNewTokens: 4, temperature: 0 },
    });
    const rec = session.step();
    expect(verifyKvSnapshotIntegrity(rec.kvSnapshot)).toEqual({ valid: true, reason: null });
    assertKvSnapshotIntegrity(rec.kvSnapshot);

    // Tamper a single float in a cloned slab buffer and verify rejection
    const tamperedK = rec.kvSnapshot.slabs[3]!.k.slice();
    tamperedK[17] = tamperedK[17]! + 1e-4;
    const tamperedSlab = { ...rec.kvSnapshot.slabs[3]!, k: tamperedK };
    const tamperedSlabs = rec.kvSnapshot.slabs.slice();
    tamperedSlabs[3] = tamperedSlab;
    const tamperedSnapshot = { ...rec.kvSnapshot, slabs: tamperedSlabs };

    const check = verifyKvSnapshotIntegrity(tamperedSnapshot);
    expect(check.valid).toBe(false);
    expect(check.reason).toContain('Slab 3 digest mismatch');
    expect(() => assertKvSnapshotIntegrity(tamperedSnapshot)).toThrow(KvIntegrityError);

    // Also detect in-place mutation of v on an isolated slab clone
    const tamperedV = rec.kvSnapshot.slabs[0]!.v.slice();
    tamperedV[0] = -tamperedV[0]!;
    const tamperedSnapshotV = {
      ...rec.kvSnapshot,
      slabs: [{ ...rec.kvSnapshot.slabs[0]!, v: tamperedV }, ...rec.kvSnapshot.slabs.slice(1)],
    };
    expect(() => assertKvSnapshotIntegrity(tamperedSnapshotV)).toThrow(KvIntegrityError);
  });

  it('reconstructs non-eager layer traces (1..18, 20..38) and historical spec cropped suffix KVs at 0-ULP', () => {
    const session = createDiffusionSession(model, {
      promptTokenIds: prompt,
      config: { mode: 'linear_spec_lora', blockLength: 4, maxNewTokens: 10, temperature: 0 },
    });

    const prefill = session.step(); // seqLen = 8
    const cycle0 = session.step();  // spec_cycle 0 (croppedSuffixLen > 0)
    expect(cycle0.kind).toBe('spec_cycle');
    expect(cycle0.cropProvenance!.croppedSuffixLen).toBeGreaterThan(0);

    // While cycle0 is active, eagerLayerTraces holds [0, 19, 39] and activeCroppedSuffixSlabs is populated
    expect(Array.from(cycle0.eagerLayerTraces.keys()).sort((a, b) => a - b)).toEqual([0, 19, 39]);
    expect(cycle0.activeCroppedSuffixSlabs).not.toBeNull();
    const expectedSuffixSlabs = cycle0.activeCroppedSuffixSlabs!.map((s) => ({
      position: s.position,
      tokenId: s.tokenId,
      k: s.k.slice(),
      v: s.v.slice(),
      slabDigest: s.slabDigest,
    }));

    // Step to cycle1 -> cycle0 eager traces & activeCroppedSuffixSlabs must be evicted!
    const cycle1 = session.step();
    expect(cycle1.kind).toBe('spec_cycle');
    const evictedCycle0 = session.getRecord(cycle0.recordId)!;
    expect(evictedCycle0.eagerLayerTraces.size).toBe(0);
    expect(evictedCycle0.activeCroppedSuffixSlabs).toBeNull();

    // 0-ULP lazy layer trace reconstruction for non-eager layers [1, 18, 20, 38] vs full eager forward pass
    const fullVerifyEager = runDiffusionForwardPass(model, {
      tokenIds: cycle0.specMeta!.draftedBlockTokens,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
      pastKvs: prefill.kvSnapshot.slabs.length > 0
        ? Array.from({ length: DIFFUSION_MINI_CONFIG.numLayers }, (_, l) => ({
            k: Float32Array.from(
              prefill.kvSnapshot.slabs.flatMap((slab) => Array.from(slab.k.subarray(l * 64, (l + 1) * 64))),
            ),
            v: Float32Array.from(
              prefill.kvSnapshot.slabs.flatMap((slab) => Array.from(slab.v.subarray(l * 64, (l + 1) * 64))),
            ),
            seqLen: prefill.kvSnapshot.seqLen,
          }))
        : null,
    });

    for (const layerIdx of [1, 18, 20, 38]) {
      const reconstructed = session.reconstructLayerTrace(cycle0.recordId, layerIdx);
      const eager = fullVerifyEager.layerTraces.get(layerIdx)!;
      assertBitExactFiniteArray(reconstructed.inputNorm, eager.inputNorm, `lazyL${layerIdx}.inputNorm`);
      assertBitExactFiniteArray(reconstructed.attention.qScaled, eager.attention.qScaled, `lazyL${layerIdx}.qScaled`);
      assertBitExactFiniteArray(reconstructed.attention.probs, eager.attention.probs, `lazyL${layerIdx}.probs`);
      assertBitExactFiniteArray(reconstructed.attention.output, eager.attention.output, `lazyL${layerIdx}.attnOut`);
      assertBitExactFiniteArray(reconstructed.swiglu.output, eager.swiglu.output, `lazyL${layerIdx}.swigluOut`);
      assertBitExactFiniteArray(reconstructed.output, eager.output, `lazyL${layerIdx}.output`);
    }

    // 0-ULP lazy cropped-suffix reconstruction on historical cycle0 after eviction
    const rebuiltSuffix = session.reconstructSpecCroppedSuffixKvs(cycle0.recordId);
    expect(rebuiltSuffix.length).toBe(expectedSuffixSlabs.length);
    for (let i = 0; i < rebuiltSuffix.length; i++) {
      expect(rebuiltSuffix[i]!.position).toBe(expectedSuffixSlabs[i]!.position);
      expect(rebuiltSuffix[i]!.tokenId).toBe(expectedSuffixSlabs[i]!.tokenId);
      expect(rebuiltSuffix[i]!.slabDigest).toBe(expectedSuffixSlabs[i]!.slabDigest);
      assertBitExactFiniteArray(rebuiltSuffix[i]!.k, expectedSuffixSlabs[i]!.k, `lazySuffix[${i}].k`);
      assertBitExactFiniteArray(rebuiltSuffix[i]!.v, expectedSuffixSlabs[i]!.v, `lazySuffix[${i}].v`);
    }

    // Seeking back to cycle0 rehydrates activeCroppedSuffixSlabs & eagerLayerTraces
    const rehydrated = session.seekRecord(cycle0.recordId);
    expect(rehydrated.eagerLayerTraces.size).toBe(3);
    expect(rehydrated.activeCroppedSuffixSlabs?.length).toBe(expectedSuffixSlabs.length);
    expect(session.getRecord(cycle1.recordId)!.eagerLayerTraces.size).toBe(0);
    expect(session.getRecord(cycle1.recordId)!.activeCroppedSuffixSlabs).toBeNull();
  });
});
