import { describe, expect, it } from 'vitest';

import {
  appendNewKvsToSnapshot,
  createDefaultDiffusionModel,
  createDiffusionSession,
  createEmptyKvSnapshot,
  DIFFUSION_MINI_CONFIG,
  DIFFUSION_MINI_EOS_TOKEN_ID,
  DIFFUSION_MINI_MASK_TOKEN_ID,
  measureDiffusionSessionMemory,
  runDiffusionExecution,
  runDiffusionForwardPass,
  runDlmBlockCycle,
  runLinearSpecCycle,
  type DiffusionForwardResult,
  type DiffusionModelParameters,
} from '../src/diffusion';

function makeBiasedModel(
  baseModel: DiffusionModelParameters,
  targetTokenId: number,
  promptTokens: readonly number[],
): DiffusionModelParameters {
  const d = baseModel.config.hiddenSize;
  const fwd = runDiffusionForwardPass(baseModel, {
    tokenIds: promptTokens,
    maskMode: 'causal',
    loraEnabled: false,
    useCache: false,
  });
  const lastNorm = fwd.finalNorm.subarray((promptTokens.length - 1) * d, promptTokens.length * d);
  const head = new Float32Array(baseModel.diffusionHead.length);
  for (let i = 0; i < d; i++) {
    head[targetTokenId * d + i] = lastNorm[i]! * 100.0;
  }
  return {
    ...baseModel,
    diffusionHead: head,
  };
}

function makeOverrideForward(
  block: Int32Array,
  logits: Float32Array,
): DiffusionForwardResult {
  const seqLen = block.length;
  const vocabSize = DIFFUSION_MINI_CONFIG.vocabSize;
  return {
    tokenIds: new Int32Array(block),
    seqLen,
    pastLen: 8,
    kvLen: 8 + seqLen,
    maskMode: 'bidirectional',
    loraEnabled: false,
    useCache: false,
    positionIds: Int32Array.from({ length: seqLen }, (_, i) => 8 + i),
    cachePositions: Int32Array.from({ length: seqLen }, (_, i) => 8 + i),
    embeddings: new Float32Array(seqLen * DIFFUSION_MINI_CONFIG.hiddenSize),
    residualCheckpoints: [],
    layerTraces: new Map(),
    newKvs: [],
    finalNorm: new Float32Array(seqLen * DIFFUSION_MINI_CONFIG.hiddenSize),
    logits,
    probs: new Float32Array(seqLen * vocabSize),
  };
}

describe('Diffusion Guards, 3-Counter NFE & Adversarial 12-Branch / 128-Record Memory Cap (Category 12)', () => {
  const model = createDefaultDiffusionModel();
  const vocabSize = DIFFUSION_MINI_CONFIG.vocabSize;
  const maskId = DIFFUSION_MINI_MASK_TOKEN_ID;
  const eosId = DIFFUSION_MINI_EOS_TOKEN_ID;
  const prompt = [0, 1, 2, 3, 4, 5, 6, 7];

  it('D18 Guard: halts immediately on greedy [MASK] non-progress and caps at maxDraftPassesPerCycle=B when T>0', () => {
    const prefill8 = runDiffusionForwardPass(model, {
      tokenIds: prompt,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    const kv8 = appendNewKvsToSnapshot(
      createEmptyKvSnapshot('token_causal', 32),
      prefill8.newKvs,
      prompt,
      'token_causal',
    );

    const makeAllMaskLogits = (seqLen: number): Float32Array => {
      const logits = new Float32Array(seqLen * vocabSize);
      for (let t = 0; t < seqLen; t++) {
        logits[t * vocabSize + maskId] = 12.0;
      }
      return logits;
    };

    // 1. T = 0.0, threshold = 0.5 -> immediate break with non_progress_mask_loop
    const greedyNonProgress = runLinearSpecCycle(model, {
      kvSnapshot: kv8,
      seedToken: 20,
      blockLength: 4,
      loraEnabled: false,
      samplerBConfig: { temperature: 0.0, threshold: 0.5, maxDraftPassesPerCycle: 4 },
      verifyTemperature: 0.0,
      overrideDraftForward: (blk) => makeOverrideForward(blk, makeAllMaskLogits(blk.length)),
    });
    expect(greedyNonProgress.draftPassCount).toBe(1);
    expect(greedyNonProgress.guardFlags.draftGuardTriggered).toBe('non_progress_mask_loop');
    expect(greedyNonProgress.draftPasses[0]!.noProgress).toBe(true);
    expect(greedyNonProgress.verifyForward).not.toBeNull();

    // 2. T = 0.5, threshold = 0.99 -> caps at maxDraftPassesPerCycle = B (test both B=4 and B=8)
    for (const blockLen of [4, 8] as const) {
      const draftUniforms = Array.from({ length: blockLen }, () =>
        Float64Array.from({ length: blockLen }, () => 0.99),
      );
      const capped = runLinearSpecCycle(model, {
        kvSnapshot: kv8,
        seedToken: 20,
        blockLength: blockLen,
        loraEnabled: true,
        samplerBConfig: {
          temperature: 0.5,
          threshold: 0.99,
          maxDraftPassesPerCycle: blockLen,
        },
        verifyTemperature: 0.0,
        explicitDraftUniforms: draftUniforms,
        overrideDraftForward: (blk) => {
          // Force logits where softmax(logits / 0.5) stays < 0.99 and u=0.99 samples MASK so masks remain all B passes
          const logits = new Float32Array(blk.length * vocabSize);
          logits.fill(-10.0);
          for (let t = 0; t < blk.length; t++) {
            logits[t * vocabSize + 0] = 0.4;
            logits[t * vocabSize + maskId] = 0.5;
          }
          return makeOverrideForward(blk, logits);
        },
      });
      expect(capped.draftPassCount).toBe(blockLen);
      expect(capped.guardFlags.draftGuardTriggered).toBe('max_draft_passes');
      expect(capped.guardFlags.warnings.some((w) => w.includes('maxDraftPassesPerCycle'))).toBe(true);
    }
  });

  it('flags residualMasksCommittedCount and trailingMasksAfterEosCount without clamping [MASK] logits', () => {
    const prefill8 = runDiffusionForwardPass(model, {
      tokenIds: prompt,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    const kv8 = appendNewKvsToSnapshot(
      createEmptyKvSnapshot('token_causal', 32),
      prefill8.newKvs,
      prompt,
      'token_causal',
    );

    // Residual [MASK] commit in DLM when model predicts [MASK] without silent clamping
    // (When diffusionHead only weights [MASK] along its norm direction or has zeroed weights for non-MASK,
    // wait: if residualMasksCommittedCount > 0, no silent clamping occurs)
    const maskHead = new Float32Array(model.diffusionHead.length);
    const dlmDenoiseFwd = runDiffusionForwardPass(model, {
      tokenIds: [20, maskId, maskId, maskId],
      maskMode: 'bidirectional',
      loraEnabled: false,
      useCache: false,
      pastKvs: prefill8.newKvs,
    });
    const d = DIFFUSION_MINI_CONFIG.hiddenSize;
    for (let i = 0; i < d; i++) {
      // Align MASK row with slot 1's finalNorm vector
      maskHead[maskId * d + i] = dlmDenoiseFwd.finalNorm[d + i]! * 100.0;
    }
    const maskBiasedModel: DiffusionModelParameters = { ...model, diffusionHead: maskHead };
    const dlmResidualMask = runDlmBlockCycle(maskBiasedModel, {
      kvSnapshot: kv8,
      blockIndex: 0,
      blockLength: 4,
      causalContext: true,
      initialSeedToken: 20,
      temperature: 0.0,
      samplerAThreshold: null,
    });
    expect(dlmResidualMask.guardFlags.residualMasksCommittedCount).toBeGreaterThan(0);
    expect(
      dlmResidualMask.guardFlags.warnings.some((w) => w.includes('Residual [MASK] committed')),
    ).toBe(true);
    expect(dlmResidualMask.finalBlockTokens.slice(1)).toContain(maskId);

    // Intra-block EOS leaving trailing [MASK] tokens in DLM:
    // On pass 0, threshold=null (quota=1) unmasks 1 slot to EOS -> no MASK before first EOS -> breaks early!
    // Remaining slots stay MASK (trailingMasksAfterEosCount > 0), committed to KV, emitted sliced to EOS.
    const eosHead = new Float32Array(model.diffusionHead.length);
    for (let i = 0; i < d; i++) {
      eosHead[eosId * d + i] = dlmDenoiseFwd.finalNorm[d + i]! * 100.0;
    }
    const eosBiasedModel: DiffusionModelParameters = { ...model, diffusionHead: eosHead };
    const dlmIntraEos = runDlmBlockCycle(eosBiasedModel, {
      kvSnapshot: kv8,
      blockIndex: 0,
      blockLength: 4,
      causalContext: true,
      initialSeedToken: 20,
      temperature: 0.0,
      samplerAThreshold: null,
    });
    expect(dlmIntraEos.eosHit).toBe(true);
    expect(dlmIntraEos.firstEosOffset).not.toBeNull();
    expect(dlmIntraEos.guardFlags.trailingMasksAfterEosCount).toBeGreaterThan(0);
    expect(dlmIntraEos.emittedTokens.length).toBe(dlmIntraEos.firstEosOffset! + 1);
    expect(dlmIntraEos.commitKvSnapshot.seqLen).toBe(12);
    expect(
      dlmIntraEos.guardFlags.warnings.some((w) => w.includes('Trailing [MASK] tokens after intra-block EOS')),
    ).toBe(true);
  });

  it('D19 EOS Crop Provenance: enforces browserPostEosCropLen == cache_len + first_eos + 1 and upstreamPreEosCropLen == cache_len + accepted', () => {
    const prefill8 = runDiffusionForwardPass(model, {
      tokenIds: prompt,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    const kv8 = appendNewKvsToSnapshot(
      createEmptyKvSnapshot('token_causal', 32),
      prefill8.newKvs,
      prompt,
      'token_causal',
    );

    // Use verifyTemperature = 2.0 + CDF midpoints on drafted block [20, 9, EOS=33, 5] so all 3 draft tokens match!
    const drafted = [20, 9, eosId, 5];
    const verifyFwd = runDiffusionForwardPass(model, {
      tokenIds: drafted,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
      pastKvs: prefill8.newKvs,
    });
    const temp = 2.0;
    const midpointUniform = (rowIdx: number, targetTok: number): number => {
      const row = verifyFwd.logits.subarray(rowIdx * vocabSize, (rowIdx + 1) * vocabSize);
      const exps = Array.from(row, (x) => Math.exp(x / temp));
      const sum = exps.reduce((a, b) => a + b, 0);
      let cdf = 0;
      for (let v = 0; v < vocabSize; v++) {
        const p = exps[v]! / sum;
        if (v === targetTok) return (cdf + 0.5 * p);
        cdf += p;
      }
      return 0.5;
    };

    const cycle = runLinearSpecCycle(model, {
      kvSnapshot: kv8,
      seedToken: 20,
      blockLength: 4,
      loraEnabled: false,
      samplerBConfig: { temperature: 0, threshold: 0.0 },
      verifyTemperature: temp,
      explicitVerifyUniforms: Float64Array.from([
        midpointUniform(0, drafted[1]!),
        midpointUniform(1, drafted[2]!),
        midpointUniform(2, drafted[3]!),
        0.5,
      ]),
      overrideDraftForward: (blk) => {
        const logits = new Float32Array(blk.length * vocabSize);
        for (let i = 0; i < blk.length; i++) {
          logits[i * vocabSize + drafted[i]!] = 10.0;
        }
        return makeOverrideForward(blk, logits);
      },
    });

    expect(cycle.acceptedCount).toBe(4);
    expect(cycle.firstEosOffset).toBe(2);
    expect(cycle.cropProvenance.cacheLenBeforeVerify).toBe(8);
    expect(cycle.cropProvenance.browserPostEosCropLen).toBe(8 + 2 + 1); // 11
    expect(cycle.cropProvenance.upstreamPreEosCropLen).toBe(8 + 4);     // 12
    expect(cycle.postCropKvSnapshot.seqLen).toBe(cycle.cropProvenance.browserPostEosCropLen);
  });

  it('Clarification #2 (kvCapacity=32 tail-block boundary) and D22 zero-token validation', () => {
    // D22: maxNewTokens <= 0 throws cleanly across all 4 modes + DiffusionSession
    for (const mode of ['ar', 'dlm', 'linear_spec', 'linear_spec_lora'] as const) {
      expect(() =>
        runDiffusionExecution(model, {
          mode,
          promptTokenIds: prompt,
          maxNewTokens: 0,
        }),
      ).toThrow(RangeError);
      expect(() =>
        runDiffusionExecution(model, {
          mode,
          promptTokenIds: prompt,
          maxNewTokens: -2,
        }),
      ).toThrow(RangeError);
    }
    expect(() =>
      createDiffusionSession(model, { promptTokenIds: prompt, config: { maxNewTokens: 0 } }),
    ).toThrow(RangeError);

    // Clarification #2: DLM B=8 from P=6 -> 6 -> 14 -> 22 -> 30 (2 slots left < 8) halts cleanly, never clamps B=1
    const prompt6 = [0, 1, 2, 3, 4, 5];
    const dlmExhausted = runDiffusionExecution(model, {
      mode: 'dlm',
      promptTokenIds: prompt6,
      blockLength: 8,
      maxNewTokens: 40,
      kvCapacity: 32,
      temperature: 0,
    });
    expect(dlmExhausted.lifecyclePhase).toBe('capacity_exhausted');
    expect(dlmExhausted.guardFlags.capacityExhausted).toBe(true);
    expect(dlmExhausted.finalKvSnapshot.seqLen).toBe(30); // 6 + 3*8 = 30, NEVER clamped to 31/32

    // Clarification #2: linear_spec B=8 from P=8 -> halts cleanly whenever remaining slots < 8
    const specExhausted = runDiffusionExecution(model, {
      mode: 'linear_spec',
      promptTokenIds: prompt,
      blockLength: 8,
      maxNewTokens: 40,
      kvCapacity: 32,
      temperature: 0,
    });
    expect(specExhausted.lifecyclePhase).toBe('capacity_exhausted');
    expect(specExhausted.guardFlags.capacityExhausted).toBe(true);
    expect(specExhausted.finalKvSnapshot.seqLen).toBeGreaterThan(24);
    expect(specExhausted.finalKvSnapshot.seqLen).toBeLessThanOrEqual(32);

    // AR halts cleanly at kvCapacity = 32 (< 1 slot left)
    const arExhausted = runDiffusionExecution(model, {
      mode: 'ar',
      promptTokenIds: prompt,
      maxNewTokens: 40,
      kvCapacity: 32,
      temperature: 0,
    });
    expect(arExhausted.lifecyclePhase).toBe('capacity_exhausted');
    expect(arExhausted.guardFlags.capacityExhausted).toBe(true);
    expect(arExhausted.finalKvSnapshot.seqLen).toBe(32);
    expect(arExhausted.generatedTokenIds.length).toBe(25); // 1 prefill + 24 decode = 25 (kv=8+24=32)
  });

  it('verifies 3-counter NFE exactness (D12 & D21) across normal and prefill-EOS scenarios', () => {
    // Normal trajectories
    const arNorm = runDiffusionExecution(model, {
      mode: 'ar',
      promptTokenIds: prompt,
      maxNewTokens: 6,
      temperature: 0,
    });
    expect(arNorm.counters).toEqual({
      actualForwardPasses: 6,
      decodeForwardPasses: 5,
      referenceMethodNfe: 6,
    });

    const dlmNorm = runDiffusionExecution(model, {
      mode: 'dlm',
      promptTokenIds: prompt,
      blockLength: 4,
      maxNewTokens: 8,
      causalContext: true,
      samplerAThreshold: 0.22,
      temperature: 0,
    });
    const sumDlmDenoise = dlmNorm.dlmBlocks.reduce((acc, b) => acc + b.denoisePassCount, 0);
    const numBlocks = dlmNorm.dlmBlocks.length;
    expect(dlmNorm.counters).toEqual({
      actualForwardPasses: 1 + sumDlmDenoise + numBlocks,
      decodeForwardPasses: sumDlmDenoise + numBlocks,
      referenceMethodNfe: sumDlmDenoise + numBlocks,
    });

    const specNorm = runDiffusionExecution(model, {
      mode: 'linear_spec_lora',
      promptTokenIds: prompt,
      blockLength: 4,
      maxNewTokens: 8,
      temperature: 0,
    });
    const sumDraft = specNorm.specCycles.reduce((acc, c) => acc + c.draftPassCount, 0);
    const numCycles = specNorm.specCycles.length;
    expect(specNorm.counters).toEqual({
      actualForwardPasses: 1 + sumDraft + numCycles,
      decodeForwardPasses: sumDraft + numCycles,
      referenceMethodNfe: 1 + sumDraft + numCycles,
    });

    // Prefill-EOS scenarios (D21): ar = 1/0/1, linear_spec* = 1/0/1, dlm = 3/2/2
    const eosModel = makeBiasedModel(model, eosId, prompt);
    const arPrefillEos = runDiffusionExecution(eosModel, {
      mode: 'ar',
      promptTokenIds: prompt,
      maxNewTokens: 8,
      temperature: 0,
    });
    expect(Array.from(arPrefillEos.generatedTokenIds)).toEqual([eosId]);
    expect(arPrefillEos.counters).toEqual({
      actualForwardPasses: 1,
      decodeForwardPasses: 0,
      referenceMethodNfe: 1,
    });

    const specPrefillEos = runDiffusionExecution(eosModel, {
      mode: 'linear_spec',
      promptTokenIds: prompt,
      blockLength: 4,
      maxNewTokens: 8,
      temperature: 0,
    });
    expect(Array.from(specPrefillEos.generatedTokenIds)).toEqual([eosId]);
    expect(specPrefillEos.counters).toEqual({
      actualForwardPasses: 1,
      decodeForwardPasses: 0,
      referenceMethodNfe: 1,
    });

    const dlmPrefillEos = runDiffusionExecution(eosModel, {
      mode: 'dlm',
      promptTokenIds: prompt,
      blockLength: 4,
      maxNewTokens: 8,
      causalContext: true,
      temperature: 0,
    });
    expect(Array.from(dlmPrefillEos.generatedTokenIds)).toEqual([eosId]);
    expect(dlmPrefillEos.dlmBlocks[0]!.denoisePassCount).toBe(1);
    expect(dlmPrefillEos.counters).toEqual({
      actualForwardPasses: 3,
      decodeForwardPasses: 2,
      referenceMethodNfe: 2,
    });
  });

  it('Adversarial Stress Test: 12 branches / 128 records with repeated B=8 spec_cycle rejections stays <= 12.00 MiB', () => {
    const shortPrompt = [0, 1, 2, 3];
    const session = createDiffusionSession(model, {
      promptTokenIds: shortPrompt,
      config: {
        mode: 'linear_spec_lora',
        blockLength: 8,
        maxNewTokens: 40,
        temperature: 0.7,
        samplerBThreshold: 0.0,
        samplingSeed: 0x4e4c4453,
      },
    });

    // Record 0: prefill (seqLen = 4)
    const rootPrefill = session.step();
    expect(rootPrefill.kvSnapshot.seqLen).toBe(4);

    // Create 12 branches from rootPrefill (1 + 11 forks) and step each tip with B=8 spec_cycles
    for (let b = 0; b < 12; b++) {
      if (b > 0) {
        session.seekRecord(rootPrefill.recordId);
        session.updateConfig({
          mode: b % 2 === 0 ? 'linear_spec_lora' : 'linear_spec',
          blockLength: 8,
          maxNewTokens: 40,
          temperature: 0.7,
          samplingSeed: 0x1000 + b,
        });
      }
      session.step();
    }

    // Round-robin step ONLY at branch tips (never creating a 13th branch) until 128 total records
    let cursorIdx = 0;
    let safety = 0;
    while (session.getState().records.size < 128 && safety < 400) {
      safety++;
      const branches = session.getBranches();
      const targetBranch = branches[cursorIdx % branches.length]!;
      const tipId = targetBranch.recordIds[targetBranch.recordIds.length - 1]!;
      const tipRec = session.getRecord(tipId)!;
      cursorIdx++;

      if (
        tipRec.lifecyclePhase === 'running' &&
        32 - tipRec.kvSnapshot.seqLen >= 8
      ) {
        session.seekRecord(tipId);
        session.step();
      }
    }

    const state = session.getState();
    expect(state.branches.length).toBe(12);
    expect(state.records.size).toBe(128);

    // Verify all non-active records evicted eagerLayerTraces & activeCroppedSuffixSlabs
    let specRecordCount = 0;
    for (const [id, rec] of state.records.entries()) {
      if (rec.kind === 'spec_cycle') specRecordCount++;
      if (id !== state.activeRecordId) {
        expect(rec.eagerLayerTraces.size).toBe(0);
        expect(rec.activeCroppedSuffixSlabs).toBeNull();
      }
    }
    expect(specRecordCount).toBeGreaterThan(50);

    const memReport = measureDiffusionSessionMemory(session);
    const retainedMiB = memReport.totalRetainedBytes / (1024 * 1024);
    console.log(
      `[Diffusion Adversarial 12-Branch / 128-Record B=8 Stress] ` +
        `branches=${memReport.branchCount}, records=${memReport.recordCount}, ` +
        `uniqueSlabs=${memReport.uniqueSlabCount}, uniqueBuffers=${memReport.uniqueBufferCount}, ` +
        `totalRetainedBytes=${memReport.totalRetainedBytes} (${retainedMiB.toFixed(4)} MiB <= 12.00 MiB)`,
    );

    expect(memReport.withinBudget).toBe(true);
    expect(memReport.totalRetainedBytes).toBeLessThanOrEqual(12_582_912);
    expect(memReport.maxRetainedBytes).toBe(12_582_912);
  });
});
