import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';

import {
  appendNewKvsToSnapshot,
  assertKvProvenanceCompatible,
  computeLlama4AttnScale,
  createDefaultDiffusionModel,
  createEmptyKvSnapshot,
  DIFFUSION_MINI_CONFIG,
  KvProvenanceMismatchError,
  materializeLayerPastKv,
  runDiffusionExecution,
  runDiffusionForwardPass,
  runDlmBlockCycle,
  runLinearSpecCycle,
  runSamplerATransferStep,
  runSamplerBDraftStep,
  type DiffusionForwardResult,
} from '../src/diffusion';
import {
  assertBitExactFiniteArray,
  assertFiniteArrayParity,
  assertFiniteNumber,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';

// ============================================================================
// FIXTURE TYPES & LOADING (CATEGORIES 7-11)
// ============================================================================

interface LayerKvFixture {
  readonly k: number[];
  readonly v: number[];
}

interface DlmDenoiseStepFixture {
  readonly stepIndex: number;
  readonly quota: number;
  readonly blockAfter: number[];
  readonly candidateTokens: number[];
  readonly confidences: Array<number | null>;
  readonly transferIndex: number[];
  readonly remainingMaskCount: number;
}

interface DlmRunFixture {
  readonly temperature?: number;
  readonly threshold?: number;
  readonly generatedTokenIds: number[];
  readonly kvMaskProvenance: 'token_causal' | 'block_bidirectional';
  readonly counters: {
    readonly actualForwardPasses: number;
    readonly decodeForwardPasses: number;
    readonly referenceMethodNfe: number;
  };
  readonly blocks: Array<{
    readonly denoisePassCount: number;
    readonly denoiseSteps: DlmDenoiseStepFixture[];
    readonly finalBlockTokens: number[];
    readonly emittedTokens: number[];
    readonly commitLastLogits: number[];
  }>;
  readonly explicitVariates: {
    readonly prefillUniform: number | null;
    readonly dlmBlockGumbelUniforms: number[][][];
    readonly dlmCommitUniforms: number[];
  };
}

interface SpecRunFixture {
  readonly temperature?: number;
  readonly threshold?: number;
  readonly generatedTokenIds: number[];
  readonly counters: {
    readonly actualForwardPasses: number;
    readonly decodeForwardPasses: number;
    readonly referenceMethodNfe: number;
  };
  readonly explicitVariates: {
    readonly prefillUniform: number | null;
    readonly specCycleDraftUniforms: number[][][];
    readonly specCycleVerifyUniforms: number[][];
  };
  readonly cycles: Array<{
    readonly cacheLenBeforeVerify: number;
    readonly draftedBlockTokens: number[];
    readonly draftPassCount: number;
    readonly verifyArTokens: number[];
    readonly matchedDraftCount: number;
    readonly acceptedCount: number;
    readonly emittedTokens: number[];
  }>;
}

interface TrajectoriesParityFixture {
  readonly metadata: {
    readonly generator: string;
    readonly pythonVersion: string;
    readonly torchVersion: string;
    readonly pinnedTorchVersion: string;
    readonly weightsSha256: string;
    readonly backboneSha256: string;
    readonly loraSha256: string;
    readonly tolerances: {
      readonly kvCacheProbsMaxAbs: number;
      readonly logitsMaxAbs: number;
      readonly tsInternalMaxAbs: number;
    };
  };
  readonly cat7IncrementalVsFullCausal: {
    readonly promptTokenIds: number[];
    readonly maxNewTokens: number;
    readonly generatedTokenIds: number[];
    readonly fullSequence18: number[];
    readonly prefillLastLogits: number[];
    readonly prefillLastProbs: number[];
    readonly stepLastLogits: number[][];
    readonly stepLastProbs: number[][];
    readonly full18Logits: number[];
    readonly full18Probs: number[];
    readonly full18LayerKvs: Record<string, LayerKvFixture>;
    readonly counters: {
      readonly actualForwardPasses: number;
      readonly decodeForwardPasses: number;
      readonly referenceMethodNfe: number;
    };
  };
  readonly cat8DualSamplersAndCausalContext: {
    readonly promptTokenIds: number[];
    readonly blockLength: number;
    readonly maxNewTokens: number;
    readonly dlmRuns: {
      readonly causalThresholdNull: DlmRunFixture;
      readonly causalThreshold022: DlmRunFixture;
      readonly causalThreshold090: DlmRunFixture;
      readonly causalStochasticT07: DlmRunFixture;
      readonly bidirThresholdNull: DlmRunFixture;
      readonly bidirThreshold022: DlmRunFixture;
    };
    readonly samplerAUnitCases: {
      readonly unscaledVsScaledConfidence: {
        readonly logits: number[];
        readonly blockTokens: number[];
        readonly temperature: number;
        readonly threshold: number;
        readonly explicitGumbelUniforms: number[];
        readonly candidateTokens: number[];
        readonly unscaledConfidences: Array<number | null>;
        readonly scaledConfidencesForContrast: Array<number | null>;
        readonly transferIndex: number[];
        readonly updatedBlock: number[];
      };
      readonly k0UnconditionalFloor: {
        readonly logits: number[];
        readonly blockTokens: number[];
        readonly temperature: number;
        readonly threshold: number;
        readonly candidateTokens: number[];
        readonly confidences: Array<number | null>;
        readonly transferIndex: number[];
        readonly updatedBlock: number[];
        readonly selectedCount: number;
      };
      readonly nonStrictThresholdGe: {
        readonly logits: number[];
        readonly blockTokens: number[];
        readonly temperature: number;
        readonly threshold: number;
        readonly confidences: Array<number | null>;
        readonly transferIndex: number[];
        readonly updatedBlock: number[];
      };
      readonly rawUnsuppressedMaskLogits: {
        readonly logits: number[];
        readonly blockTokens: number[];
        readonly temperature: number;
        readonly threshold: number;
        readonly candidateTokens: number[];
        readonly confidences: Array<number | null>;
        readonly transferIndex: number[];
        readonly updatedBlock: number[];
        readonly unmaskedMaskTokenCount: number;
      };
    };
    readonly samplerBRuns: {
      readonly linearSpecThreshold0T0: SpecRunFixture;
      readonly linearSpecThreshold022T0: SpecRunFixture;
      readonly linearSpecThreshold090T0: SpecRunFixture;
      readonly linearSpecThreshold022T07: SpecRunFixture;
    };
    readonly samplerBUnitCases: {
      readonly threshold0SinglePass: {
        readonly draftLogits: number[];
        readonly blockTokens: number[];
        readonly temperature: number;
        readonly threshold: number;
        readonly draftTokens: number[];
        readonly confidences: number[];
        readonly unmaskMask: number[];
        readonly updatedBlock: number[];
        readonly singlePassCommitAll: boolean;
      };
      readonly stochasticTempScaledConfAndMaskRaw: {
        readonly draftLogits: number[];
        readonly blockTokens: number[];
        readonly temperature: number;
        readonly threshold: number;
        readonly explicitUniforms: number[];
        readonly draftTokens: number[];
        readonly confidences: number[];
        readonly unmaskMask: number[];
        readonly updatedBlock: number[];
      };
      readonly argmaxFallback: {
        readonly draftLogits: number[];
        readonly blockTokens: number[];
        readonly temperature: number;
        readonly threshold: number;
        readonly draftTokens: number[];
        readonly confidences: number[];
        readonly unmaskMask: number[];
        readonly updatedBlock: number[];
      };
    };
  };
  readonly cat9CacheLifecycleAndD19EosCrop: {
    readonly promptTokenIds: number[];
    readonly specCycleRollback: {
      readonly blockLength: number;
      readonly cycle0DraftedBlock: number[];
      readonly cycle0AcceptedCount: number;
      readonly cycle0PreCropLen: number;
      readonly cycle0PostCropLen: number;
      readonly acceptedPrefixTokens: number[];
      readonly croppedLayerKvs: Record<string, LayerKvFixture>;
      readonly pureArLayerKvs: Record<string, LayerKvFixture>;
    };
    readonly d19MidAcceptedPrefixEosCrop: {
      readonly promptTokenIds: number[];
      readonly cacheLenBeforeVerify: number;
      readonly draftedBlockTokens: number[];
      readonly verifyTemperature: number;
      readonly explicitVerifyUniforms: number[];
      readonly verifyArTokens: number[];
      readonly matchedDraftCount: number;
      readonly acceptedCount: number;
      readonly firstEosOffset: number;
      readonly browserPostEosCropLen: number;
      readonly upstreamPreEosCropLen: number;
      readonly emittedTokens: number[];
      readonly postEosArPrefixTokens: number[];
      readonly croppedLayerKvs: Record<string, LayerKvFixture>;
      readonly pureArLayerKvs: Record<string, LayerKvFixture>;
    };
  };
  readonly cat10LinearSpecGreedyAndLora: {
    readonly promptTokenIds: number[];
    readonly maxNewTokens: number;
    readonly blockLength: number;
    readonly arTokens: number[];
    readonly linearSpecTokens: number[];
    readonly linearSpecLoraTokens: number[];
    readonly cycle0DraftLogitsNoLora: number[];
    readonly cycle0DraftLogitsWithLora: number[];
    readonly cycle0VerifyLogitsNoLora: number[];
    readonly cycle0VerifyLogitsWithLora: number[];
    readonly draftLogitMaxDiffLoraOnVsOff: number;
    readonly verifyLogitMaxDiffLoraOnVsOff: number;
    readonly arCounters: {
      readonly actualForwardPasses: number;
      readonly decodeForwardPasses: number;
      readonly referenceMethodNfe: number;
    };
    readonly linearSpecCounters: {
      readonly actualForwardPasses: number;
      readonly decodeForwardPasses: number;
      readonly referenceMethodNfe: number;
    };
    readonly linearSpecLoraCounters: {
      readonly actualForwardPasses: number;
      readonly decodeForwardPasses: number;
      readonly referenceMethodNfe: number;
    };
  };
  readonly cat11StochasticExplicitVariates: {
    readonly promptTokenIds: number[];
    readonly temperature: number;
    readonly blockLength: number;
    readonly maxNewTokens: number;
    readonly ar: {
      readonly explicitVariates: { readonly prefillUniform: number; readonly arStepUniforms: number[] };
      readonly generatedTokenIds: number[];
      readonly lastStepLogits: number[];
      readonly counters: {
        readonly actualForwardPasses: number;
        readonly decodeForwardPasses: number;
        readonly referenceMethodNfe: number;
      };
    };
    readonly dlmCausal: {
      readonly threshold: number;
      readonly explicitVariates: {
        readonly prefillUniform: number;
        readonly dlmBlockGumbelUniforms: number[][][];
        readonly dlmCommitUniforms: number[];
      };
      readonly generatedTokenIds: number[];
      readonly lastBlockCommitLogits: number[];
      readonly counters: {
        readonly actualForwardPasses: number;
        readonly decodeForwardPasses: number;
        readonly referenceMethodNfe: number;
      };
    };
    readonly linearSpec: {
      readonly threshold: number;
      readonly explicitVariates: {
        readonly prefillUniform: number;
        readonly specCycleDraftUniforms: number[][][];
        readonly specCycleVerifyUniforms: number[][];
      };
      readonly generatedTokenIds: number[];
      readonly lastCycleVerifyLogits: number[];
      readonly counters: {
        readonly actualForwardPasses: number;
        readonly decodeForwardPasses: number;
        readonly referenceMethodNfe: number;
      };
    };
    readonly linearSpecLora: {
      readonly threshold: number;
      readonly explicitVariates: {
        readonly prefillUniform: number;
        readonly specCycleDraftUniforms: number[][][];
        readonly specCycleVerifyUniforms: number[][];
      };
      readonly generatedTokenIds: number[];
      readonly lastCycleVerifyLogits: number[];
      readonly counters: {
        readonly actualForwardPasses: number;
        readonly decodeForwardPasses: number;
        readonly referenceMethodNfe: number;
      };
    };
  };
}

const FX: TrajectoriesParityFixture = JSON.parse(
  readFileSync(resolve(__dirname, 'fixtures/pytorch-diffusion-trajectories-parity.json'), 'utf-8'),
);

function makeSyntheticOverrideForward(
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

function toTsExplicitVariates(v: {
  readonly prefillUniform?: number | null;
  readonly arStepUniforms?: readonly number[];
  readonly dlmBlockGumbelUniforms?: readonly (readonly (readonly number[])[])[];
  readonly dlmCommitUniforms?: readonly number[];
  readonly specCycleDraftUniforms?: readonly (readonly (readonly number[])[])[];
  readonly specCycleVerifyUniforms?: readonly (readonly number[])[];
}) {
  return {
    ...(v.prefillUniform !== null && v.prefillUniform !== undefined
      ? { prefillUniform: v.prefillUniform }
      : {}),
    ...(v.arStepUniforms ? { arStepUniforms: v.arStepUniforms } : {}),
    ...(v.dlmBlockGumbelUniforms
      ? {
          dlmBlockGumbelUniforms: v.dlmBlockGumbelUniforms.map((blk) =>
            blk.map((step) => Float64Array.from(step)),
          ),
        }
      : {}),
    ...(v.dlmCommitUniforms ? { dlmCommitUniforms: v.dlmCommitUniforms } : {}),
    ...(v.specCycleDraftUniforms
      ? {
          specCycleDraftUniforms: v.specCycleDraftUniforms.map((cyc) =>
            cyc.map((pass) => Float64Array.from(pass)),
          ),
        }
      : {}),
    ...(v.specCycleVerifyUniforms
      ? {
          specCycleVerifyUniforms: v.specCycleVerifyUniforms.map((cyc) => Float64Array.from(cyc)),
        }
      : {}),
  };
}

// ============================================================================
// TEST SUITE (CATEGORIES 7-11)
// ============================================================================

describe('Independent PyTorch 2.15.0 Trajectory Parity — Nemotron-Labs-Diffusion (Categories 7–11)', () => {
  const model = createDefaultDiffusionModel();
  const vocabSize = DIFFUSION_MINI_CONFIG.vocabSize;
  const observedMaxErrByCategory: Record<string, number> = {};

  function recordErr(category: string, err: number): void {
    assertFiniteNumber(err, `observedErr[${category}]`);
    const prev = observedMaxErrByCategory[category];
    if (prev === undefined || err > prev) observedMaxErrByCategory[category] = err;
  }

  afterAll(() => {
    const lines = Object.entries(observedMaxErrByCategory).map(
      ([k, v]) => `  ${k}: maxAbsErr = ${v.toExponential(3)}`,
    );
    console.log(`[Diffusion PyTorch 2.15.0 Parity Summary — Cats 7–11]\n${lines.join('\n')}`);
  });

  // --------------------------------------------------------------------------
  // Category 7: Incremental vs Full-Context Causal AR crossing pos = 16
  // --------------------------------------------------------------------------
  it('Category 7: verifies incremental AR vs full 18-token causal pass crossing pos = 16 (Llama-4 jump)', () => {
    expect(FX.metadata.torchVersion.startsWith(FX.metadata.pinnedTorchVersion)).toBe(true);
    const fx = FX.cat7IncrementalVsFullCausal;
    const tol = FX.metadata.tolerances;

    const arRes = runDiffusionExecution(model, {
      mode: 'ar',
      promptTokenIds: fx.promptTokenIds,
      maxNewTokens: fx.maxNewTokens,
      temperature: 0,
    });

    expect(Array.from(arRes.generatedTokenIds)).toEqual(fx.generatedTokenIds);
    expect(arRes.counters).toEqual(fx.counters);
    expect(arRes.finalKvSnapshot.seqLen).toBe(18);
    expect(arRes.arSteps.length).toBe(10);

    // Llama-4 scale jump at pos = 16 (origMaxPos = 16)
    const scalePos15 = computeLlama4AttnScale(15, DIFFUSION_MINI_CONFIG.rope.llama4Beta, 16);
    const scalePos16 = computeLlama4AttnScale(16, DIFFUSION_MINI_CONFIG.rope.llama4Beta, 16);
    expect(scalePos15).toBe(1.0);
    expect(scalePos16).toBeCloseTo(1.06931472, 6);
    expect(arRes.arSteps[7]!.layerTraces.get(0)!.attention.llama4Scales[0]).toBe(1.0);
    expect(arRes.arSteps[8]!.layerTraces.get(0)!.attention.llama4Scales[0]).toBeCloseTo(1.06931472, 6);

    // PyTorch parity on prefill + 10 AR steps
    const prefillLastLogits = arRes.prefillForward.logits.subarray(7 * vocabSize, 8 * vocabSize);
    const prefillLastProbs = arRes.prefillForward.probs.subarray(7 * vocabSize, 8 * vocabSize);
    recordErr(
      'cat7_logits_vs_pytorch',
      assertFiniteArrayParity(prefillLastLogits, fx.prefillLastLogits, tol.logitsMaxAbs, 'cat7.prefillLogits'),
    );
    recordErr(
      'cat7_probs_kv_vs_pytorch',
      assertFiniteArrayParity(prefillLastProbs, fx.prefillLastProbs, tol.kvCacheProbsMaxAbs, 'cat7.prefillProbs'),
    );

    for (let s = 0; s < arRes.arSteps.length; s++) {
      const step = arRes.arSteps[s]!;
      recordErr(
        'cat7_logits_vs_pytorch',
        assertFiniteArrayParity(step.logits, fx.stepLastLogits[s]!, tol.logitsMaxAbs, `cat7.step${s}.logits`),
      );
      recordErr(
        'cat7_probs_kv_vs_pytorch',
        assertFiniteArrayParity(step.probs, fx.stepLastProbs[s]!, tol.kvCacheProbsMaxAbs, `cat7.step${s}.probs`),
      );
    }

    // Single full 18-token causal forward pass over prompt + generated[:10]
    const full18 = runDiffusionForwardPass(model, {
      tokenIds: fx.fullSequence18,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    recordErr(
      'cat7_logits_vs_pytorch',
      assertFiniteArrayParity(full18.logits, fx.full18Logits, tol.logitsMaxAbs, 'cat7.full18Logits'),
    );
    recordErr(
      'cat7_probs_kv_vs_pytorch',
      assertFiniteArrayParity(full18.probs, fx.full18Probs, tol.kvCacheProbsMaxAbs, 'cat7.full18Probs'),
    );

    // Verify incremental KV snapshot vs PyTorch full-18 KVs (layers 0, 19, 39) & TS full18 (all 40 layers <= 1e-6)
    for (const layerIdx of [0, 19, 39]) {
      const incSlice = materializeLayerPastKv(arRes.finalKvSnapshot, layerIdx)!;
      const refKv = fx.full18LayerKvs[String(layerIdx)]!;
      recordErr(
        'cat7_probs_kv_vs_pytorch',
        assertFiniteArrayParity(incSlice.k, refKv.k, tol.kvCacheProbsMaxAbs, `cat7.L${layerIdx}.kvK`),
      );
      recordErr(
        'cat7_probs_kv_vs_pytorch',
        assertFiniteArrayParity(incSlice.v, refKv.v, tol.kvCacheProbsMaxAbs, `cat7.L${layerIdx}.kvV`),
      );
    }

    for (let l = 0; l < DIFFUSION_MINI_CONFIG.numLayers; l++) {
      const incSlice = materializeLayerPastKv(arRes.finalKvSnapshot, l)!;
      const fullSlice = full18.newKvs[l]!;
      recordErr(
        'cat7_ts_incremental_vs_full',
        assertFiniteArrayParity(incSlice.k, fullSlice.k, tol.tsInternalMaxAbs, `cat7.tsInternal.L${l}.k`),
      );
      recordErr(
        'cat7_ts_incremental_vs_full',
        assertFiniteArrayParity(incSlice.v, fullSlice.v, tol.tsInternalMaxAbs, `cat7.tsInternal.L${l}.v`),
      );
    }

    // TS incremental logits/probs (pos 7..17) vs TS full18 logits/probs (pos 7..17) <= 1e-6
    recordErr(
      'cat7_ts_incremental_vs_full',
      assertFiniteArrayParity(
        prefillLastLogits,
        full18.logits.subarray(7 * vocabSize, 8 * vocabSize),
        tol.tsInternalMaxAbs,
        'cat7.tsInternal.pos7.logits',
      ),
    );
    for (let s = 0; s < arRes.arSteps.length; s++) {
      const pos = 8 + s;
      recordErr(
        'cat7_ts_incremental_vs_full',
        assertFiniteArrayParity(
          arRes.arSteps[s]!.logits,
          full18.logits.subarray(pos * vocabSize, (pos + 1) * vocabSize),
          tol.tsInternalMaxAbs,
          `cat7.tsInternal.pos${pos}.logits`,
        ),
      );
      recordErr(
        'cat7_ts_incremental_vs_full',
        assertFiniteArrayParity(
          arRes.arSteps[s]!.probs,
          full18.probs.subarray(pos * vocabSize, (pos + 1) * vocabSize),
          tol.tsInternalMaxAbs,
          `cat7.tsInternal.pos${pos}.probs`,
        ),
      );
    }
  });

  // --------------------------------------------------------------------------
  // Category 8: Dual Samplers & causal_context = True | False
  // --------------------------------------------------------------------------
  it('Category 8: verifies Sampler A (dlm causal/bidir, static quotas, thresholds, T=0.7) and Sampler B (linear_spec)', () => {
    const fx = FX.cat8DualSamplersAndCausalContext;
    const tol = FX.metadata.tolerances;

    const assertDlmParity = (
      label: string,
      causalContext: boolean,
      temperature: number,
      threshold: number | null,
      ref: DlmRunFixture,
    ) => {
      const res = runDiffusionExecution(model, {
        mode: 'dlm',
        promptTokenIds: fx.promptTokenIds,
        blockLength: fx.blockLength,
        maxNewTokens: fx.maxNewTokens,
        causalContext,
        temperature,
        samplerAThreshold: threshold,
        explicitVariates: toTsExplicitVariates(ref.explicitVariates),
      });

      expect(Array.from(res.generatedTokenIds), `${label}.tokens`).toEqual(ref.generatedTokenIds);
      expect(res.kvMaskProvenance, `${label}.prov`).toBe(ref.kvMaskProvenance);
      expect(res.counters, `${label}.counters`).toEqual(ref.counters);
      expect(res.prefillForward.maskMode).toBe(causalContext ? 'causal' : 'bidirectional');
      expect(res.dlmBlocks.length).toBe(ref.blocks.length);

      for (let b = 0; b < ref.blocks.length; b++) {
        const blk = res.dlmBlocks[b]!;
        const refBlk = ref.blocks[b]!;
        expect(blk.denoisePassCount).toBe(refBlk.denoisePassCount);
        expect(Array.from(blk.finalBlockTokens)).toEqual(refBlk.finalBlockTokens);
        expect(Array.from(blk.emittedTokens)).toEqual(refBlk.emittedTokens);
        expect(blk.commitForward.maskMode).toBe(causalContext ? 'causal' : 'bidirectional');

        for (let s = 0; s < refBlk.denoiseSteps.length; s++) {
          const step = blk.denoiseSteps[s]!;
          const refStep = refBlk.denoiseSteps[s]!;
          expect(step.quota).toBe(refStep.quota);
          expect(Array.from(step.blockAfter)).toEqual(refStep.blockAfter);
          expect(Array.from(step.candidateTokens)).toEqual(refStep.candidateTokens);
          expect(Array.from(step.transferIndex)).toEqual(refStep.transferIndex);
          expect(step.remainingMaskCount).toBe(refStep.remainingMaskCount);
          for (let t = 0; t < step.confidences.length; t++) {
            const refC = refStep.confidences[t];
            if (refC === null) {
              expect(step.confidences[t]).toBe(-Infinity);
            } else {
              const err = Math.abs(step.confidences[t]! - refC);
              expect(err).toBeLessThanOrEqual(tol.kvCacheProbsMaxAbs);
              recordErr('cat8_probs_conf', err);
            }
          }
        }
      }
      return res;
    };

    const dlmCausalNull = assertDlmParity('dlm.causal.null', true, 0, null, fx.dlmRuns.causalThresholdNull);
    expect(dlmCausalNull.dlmBlocks[0]!.denoiseSteps.map((s) => s.quota)).toEqual([1, 1, 1]);
    assertDlmParity('dlm.causal.022', true, 0, 0.22, fx.dlmRuns.causalThreshold022);
    assertDlmParity('dlm.causal.090', true, 0, 0.9, fx.dlmRuns.causalThreshold090);
    assertDlmParity('dlm.causal.t07', true, 0.7, 0.22, fx.dlmRuns.causalStochasticT07);

    const dlmBidirNull = assertDlmParity('dlm.bidir.null', false, 0, null, fx.dlmRuns.bidirThresholdNull);
    expect(dlmBidirNull.dlmBlocks[0]!.denoiseSteps.map((s) => s.quota)).toEqual([1, 1, 1, 1]);
    assertDlmParity('dlm.bidir.022', false, 0, 0.22, fx.dlmRuns.bidirThreshold022);

    // Sampler A direct edge-case unit fixtures
    const saUnscaledRef = fx.samplerAUnitCases.unscaledVsScaledConfidence;
    const saUnscaledRes = runSamplerATransferStep({
      logits: Float32Array.from(saUnscaledRef.logits),
      blockTokens: Int32Array.from(saUnscaledRef.blockTokens),
      seqLen: 4,
      vocabSize,
      numTransferTokens: 3,
      config: { temperature: saUnscaledRef.temperature, threshold: saUnscaledRef.threshold },
      explicitGumbelUniforms: Float64Array.from(saUnscaledRef.explicitGumbelUniforms),
    });
    expect(Array.from(saUnscaledRes.candidateTokens)).toEqual(saUnscaledRef.candidateTokens);
    expect(Array.from(saUnscaledRes.transferIndex)).toEqual(saUnscaledRef.transferIndex);
    expect(Array.from(saUnscaledRes.updatedBlock)).toEqual(saUnscaledRef.updatedBlock);
    // Slot 2 has unscaled conf < 0.22 (stays masked) while scaled conf >= 0.22!
    expect(saUnscaledRes.confidences[2]).toBeLessThan(0.22);
    expect(saUnscaledRef.scaledConfidencesForContrast[2]!).toBeGreaterThanOrEqual(0.22);
    expect(saUnscaledRes.transferIndex[2]).toBe(0);

    const saK0Ref = fx.samplerAUnitCases.k0UnconditionalFloor;
    const saK0Res = runSamplerATransferStep({
      logits: Float32Array.from(saK0Ref.logits),
      blockTokens: Int32Array.from(saK0Ref.blockTokens),
      seqLen: 4,
      vocabSize,
      numTransferTokens: 3,
      config: { temperature: saK0Ref.temperature, threshold: saK0Ref.threshold },
    });
    expect(saK0Res.selectedCount).toBe(1);
    expect(Array.from(saK0Res.transferIndex)).toEqual(saK0Ref.transferIndex);
    expect(Array.from(saK0Res.updatedBlock)).toEqual(saK0Ref.updatedBlock);

    const saGeRef = fx.samplerAUnitCases.nonStrictThresholdGe;
    const saGeRes = runSamplerATransferStep({
      logits: Float32Array.from(saGeRef.logits),
      blockTokens: Int32Array.from(saGeRef.blockTokens),
      seqLen: 4,
      vocabSize,
      numTransferTokens: 3,
      config: { temperature: saGeRef.temperature, threshold: saGeRef.threshold },
    });
    expect(Array.from(saGeRes.transferIndex)).toEqual(saGeRef.transferIndex);
    expect(Array.from(saGeRes.updatedBlock)).toEqual(saGeRef.updatedBlock);

    const saMaskRef = fx.samplerAUnitCases.rawUnsuppressedMaskLogits;
    const saMaskRes = runSamplerATransferStep({
      logits: Float32Array.from(saMaskRef.logits),
      blockTokens: Int32Array.from(saMaskRef.blockTokens),
      seqLen: 4,
      vocabSize,
      numTransferTokens: 3,
      config: { temperature: saMaskRef.temperature, threshold: saMaskRef.threshold },
    });
    expect(saMaskRes.unmaskedMaskTokenCount).toBe(saMaskRef.unmaskedMaskTokenCount);
    expect(Array.from(saMaskRes.updatedBlock)).toEqual(saMaskRef.updatedBlock);

    // Sampler B trajectories
    const assertSpecParity = (
      label: string,
      temperature: number,
      threshold: number,
      ref: SpecRunFixture,
    ) => {
      const res = runDiffusionExecution(model, {
        mode: 'linear_spec',
        promptTokenIds: fx.promptTokenIds,
        blockLength: fx.blockLength,
        maxNewTokens: fx.maxNewTokens,
        temperature,
        samplerBThreshold: threshold,
        explicitVariates: toTsExplicitVariates(ref.explicitVariates),
      });
      expect(Array.from(res.generatedTokenIds), `${label}.tokens`).toEqual(ref.generatedTokenIds);
      expect(res.counters, `${label}.counters`).toEqual(ref.counters);
      expect(res.specCycles.length).toBe(ref.cycles.length);
      for (let c = 0; c < ref.cycles.length; c++) {
        const cyc = res.specCycles[c]!;
        const refCyc = ref.cycles[c]!;
        expect(Array.from(cyc.draftedBlockTokens)).toEqual(refCyc.draftedBlockTokens);
        expect(cyc.draftPassCount).toBe(refCyc.draftPassCount);
        expect(Array.from(cyc.verifyArTokens)).toEqual(refCyc.verifyArTokens);
        expect(cyc.acceptedCount).toBe(refCyc.acceptedCount);
        expect(Array.from(cyc.emittedTokens)).toEqual(refCyc.emittedTokens);
      }
    };

    assertSpecParity('spec.t0.0', 0, 0.0, fx.samplerBRuns.linearSpecThreshold0T0);
    assertSpecParity('spec.t0.022', 0, 0.22, fx.samplerBRuns.linearSpecThreshold022T0);
    assertSpecParity('spec.t0.090', 0, 0.9, fx.samplerBRuns.linearSpecThreshold090T0);
    assertSpecParity('spec.t07.022', 0.7, 0.22, fx.samplerBRuns.linearSpecThreshold022T07);

    // Sampler B unit cases
    const sb0Ref = fx.samplerBUnitCases.threshold0SinglePass;
    const sb0Res = runSamplerBDraftStep({
      draftLogits: Float32Array.from(sb0Ref.draftLogits),
      blockTokens: Int32Array.from(sb0Ref.blockTokens),
      seqLen: 4,
      vocabSize,
      config: { temperature: sb0Ref.temperature, threshold: sb0Ref.threshold },
    });
    expect(sb0Res.singlePassCommitAll).toBe(true);
    expect(Array.from(sb0Res.draftTokens)).toEqual(sb0Ref.draftTokens);
    expect(Array.from(sb0Res.unmaskMask)).toEqual(sb0Ref.unmaskMask);
    expect(Array.from(sb0Res.updatedBlock)).toEqual(sb0Ref.updatedBlock);

    const sbT07Ref = fx.samplerBUnitCases.stochasticTempScaledConfAndMaskRaw;
    const sbT07Res = runSamplerBDraftStep({
      draftLogits: Float32Array.from(sbT07Ref.draftLogits),
      blockTokens: Int32Array.from(sbT07Ref.blockTokens),
      seqLen: 4,
      vocabSize,
      config: { temperature: sbT07Ref.temperature, threshold: sbT07Ref.threshold },
      explicitUniforms: Float64Array.from(sbT07Ref.explicitUniforms),
    });
    expect(Array.from(sbT07Res.draftTokens)).toEqual(sbT07Ref.draftTokens);
    expect(Array.from(sbT07Res.unmaskMask)).toEqual(sbT07Ref.unmaskMask);
    expect(Array.from(sbT07Res.updatedBlock)).toEqual(sbT07Ref.updatedBlock);
    recordErr(
      'cat8_probs_conf',
      assertFiniteArrayParity(sbT07Res.confidences, sbT07Ref.confidences, tol.kvCacheProbsMaxAbs, 'cat8.sbT07.conf'),
    );

    const sbFallbackRef = fx.samplerBUnitCases.argmaxFallback;
    const sbFallbackRes = runSamplerBDraftStep({
      draftLogits: Float32Array.from(sbFallbackRef.draftLogits),
      blockTokens: Int32Array.from(sbFallbackRef.blockTokens),
      seqLen: 4,
      vocabSize,
      config: { temperature: sbFallbackRef.temperature, threshold: sbFallbackRef.threshold },
    });
    expect(Array.from(sbFallbackRes.unmaskMask)).toEqual(sbFallbackRef.unmaskMask);
    expect(Array.from(sbFallbackRes.updatedBlock)).toEqual(sbFallbackRef.updatedBlock);
  });

  // --------------------------------------------------------------------------
  // Category 9: Cache Lifecycle, kvMaskProvenance & Rollback / D19 EOS Crop
  // --------------------------------------------------------------------------
  it('Category 9: verifies 0-ULP prefix immutability, kvMaskProvenance isolation, rollback KV parity, and D19 EOS crop', () => {
    const fx = FX.cat9CacheLifecycleAndD19EosCrop;
    const tol = FX.metadata.tolerances;

    // 1. Causal prefill snapshot P=8
    const prefill8 = runDiffusionForwardPass(model, {
      tokenIds: fx.promptTokenIds,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    const prefixSnap = appendNewKvsToSnapshot(
      createEmptyKvSnapshot('token_causal', 32),
      prefill8.newKvs,
      fx.promptTokenIds,
      'token_causal',
    );
    const slab0KCopy = prefixSnap.slabs[0]!.k.slice();
    const slab7VCopy = prefixSnap.slabs[7]!.v.slice();

    // 2. 0-ULP prefix immutability during useCache=false DLM denoise & Spec draft
    const dlmCycle = runDlmBlockCycle(model, {
      kvSnapshot: prefixSnap,
      blockIndex: 0,
      blockLength: 4,
      causalContext: true,
      initialSeedToken: 20,
      temperature: 0,
      samplerAThreshold: null,
    });
    for (let i = 0; i < prefixSnap.seqLen; i++) {
      expect(dlmCycle.commitKvSnapshot.slabs[i]).toBe(prefixSnap.slabs[i]);
    }
    assertBitExactFiniteArray(prefixSnap.slabs[0]!.k, slab0KCopy, 'cat9.dlmPrefixImmutableK');
    assertBitExactFiniteArray(prefixSnap.slabs[7]!.v, slab7VCopy, 'cat9.dlmPrefixImmutableV');

    // 3. kvMaskProvenance cross-mode reuse rejection
    const bidirSnap = appendNewKvsToSnapshot(
      createEmptyKvSnapshot('block_bidirectional', 32),
      runDiffusionForwardPass(model, {
        tokenIds: fx.promptTokenIds,
        maskMode: 'bidirectional',
        loraEnabled: false,
        useCache: true,
      }).newKvs,
      fx.promptTokenIds,
      'block_bidirectional',
    );
    expect(() => assertKvProvenanceCompatible(bidirSnap, 'token_causal', 'test')).toThrow(
      KvProvenanceMismatchError,
    );
    expect(() =>
      runLinearSpecCycle(model, {
        kvSnapshot: bidirSnap,
        seedToken: 20,
        blockLength: 4,
        loraEnabled: false,
        samplerBConfig: { temperature: 0, threshold: 0.0 },
        verifyTemperature: 0,
      }),
    ).toThrow(KvProvenanceMismatchError);
    expect(() =>
      runDlmBlockCycle(model, {
        kvSnapshot: prefixSnap,
        blockIndex: 0,
        blockLength: 4,
        causalContext: false,
        initialSeedToken: null,
        temperature: 0,
        samplerAThreshold: null,
      }),
    ).toThrow(KvProvenanceMismatchError);

    // 4. Speculative KV rollback vs pure causal AR KV (0-ULP TS, <= 2e-5 PyTorch)
    const rb = fx.specCycleRollback;
    const specCycle0 = runLinearSpecCycle(model, {
      kvSnapshot: prefixSnap,
      seedToken: rb.cycle0DraftedBlock[0]!,
      blockLength: rb.blockLength,
      loraEnabled: false,
      samplerBConfig: { temperature: 0, threshold: 0.0 },
      verifyTemperature: 0,
    });
    expect(Array.from(specCycle0.draftedBlockTokens)).toEqual(rb.cycle0DraftedBlock);
    expect(specCycle0.acceptedCount).toBe(rb.cycle0AcceptedCount);
    expect(specCycle0.cropProvenance.preCropKvLen).toBe(rb.cycle0PreCropLen);
    expect(specCycle0.postCropKvSnapshot.seqLen).toBe(rb.cycle0PostCropLen);
    expect(specCycle0.croppedSuffixSlabs.length).toBe(rb.cycle0PreCropLen - rb.cycle0PostCropLen);

    const pureArPrefix = runDiffusionForwardPass(model, {
      tokenIds: rb.acceptedPrefixTokens,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    for (let l = 0; l < DIFFUSION_MINI_CONFIG.numLayers; l++) {
      const croppedSlice = materializeLayerPastKv(specCycle0.postCropKvSnapshot, l)!;
      assertBitExactFiniteArray(croppedSlice.k, pureArPrefix.newKvs[l]!.k, `cat9.tsCroppedVsAr.L${l}.k`);
      assertBitExactFiniteArray(croppedSlice.v, pureArPrefix.newKvs[l]!.v, `cat9.tsCroppedVsAr.L${l}.v`);
    }
    for (const layerIdx of [0, 19, 39]) {
      const croppedSlice = materializeLayerPastKv(specCycle0.postCropKvSnapshot, layerIdx)!;
      recordErr(
        'cat9_cropped_kv_vs_pytorch',
        assertFiniteArrayParity(
          croppedSlice.k,
          rb.croppedLayerKvs[String(layerIdx)]!.k,
          tol.kvCacheProbsMaxAbs,
          `cat9.L${layerIdx}.croppedK`,
        ),
      );
      recordErr(
        'cat9_cropped_kv_vs_pytorch',
        assertFiniteArrayParity(
          croppedSlice.v,
          rb.croppedLayerKvs[String(layerIdx)]!.v,
          tol.kvCacheProbsMaxAbs,
          `cat9.L${layerIdx}.croppedV`,
        ),
      );
      recordErr(
        'cat9_cropped_kv_vs_pytorch',
        assertFiniteArrayParity(
          pureArPrefix.newKvs[layerIdx]!.k,
          rb.pureArLayerKvs[String(layerIdx)]!.k,
          tol.kvCacheProbsMaxAbs,
          `cat9.L${layerIdx}.pureArK`,
        ),
      );
    }

    // 5. Explicit D19 mid-accepted-prefix EOS crop parity
    const d19 = fx.d19MidAcceptedPrefixEosCrop;
    const d19Cycle = runLinearSpecCycle(model, {
      kvSnapshot: prefixSnap,
      seedToken: d19.draftedBlockTokens[0]!,
      blockLength: d19.draftedBlockTokens.length,
      loraEnabled: false,
      samplerBConfig: { temperature: 0, threshold: 0.0 },
      verifyTemperature: d19.verifyTemperature,
      explicitVerifyUniforms: Float64Array.from(d19.explicitVerifyUniforms),
      overrideDraftForward: (blk) => {
        const logits = new Float32Array(blk.length * vocabSize);
        for (let i = 0; i < blk.length; i++) {
          logits[i * vocabSize + d19.draftedBlockTokens[i]!] = 10.0;
        }
        return makeSyntheticOverrideForward(blk, logits);
      },
    });
    expect(Array.from(d19Cycle.draftedBlockTokens)).toEqual(d19.draftedBlockTokens);
    expect(Array.from(d19Cycle.verifyArTokens)).toEqual(d19.verifyArTokens);
    expect(d19Cycle.acceptedCount).toBe(d19.acceptedCount);
    expect(d19Cycle.firstEosOffset).toBe(d19.firstEosOffset);
    expect(d19Cycle.cropProvenance.browserPostEosCropLen).toBe(d19.browserPostEosCropLen);
    expect(d19Cycle.cropProvenance.upstreamPreEosCropLen).toBe(d19.upstreamPreEosCropLen);
    expect(d19Cycle.postCropKvSnapshot.seqLen).toBe(d19.browserPostEosCropLen);
    expect(Array.from(d19Cycle.emittedTokens)).toEqual(d19.emittedTokens);

    const d19PureAr = runDiffusionForwardPass(model, {
      tokenIds: d19.postEosArPrefixTokens,
      maskMode: 'causal',
      loraEnabled: false,
      useCache: true,
    });
    for (let l = 0; l < DIFFUSION_MINI_CONFIG.numLayers; l++) {
      const s = materializeLayerPastKv(d19Cycle.postCropKvSnapshot, l)!;
      assertBitExactFiniteArray(s.k, d19PureAr.newKvs[l]!.k, `cat9.d19.tsVsAr.L${l}.k`);
      assertBitExactFiniteArray(s.v, d19PureAr.newKvs[l]!.v, `cat9.d19.tsVsAr.L${l}.v`);
    }
    for (const layerIdx of [0, 39]) {
      const s = materializeLayerPastKv(d19Cycle.postCropKvSnapshot, layerIdx)!;
      recordErr(
        'cat9_cropped_kv_vs_pytorch',
        assertFiniteArrayParity(
          s.k,
          d19.croppedLayerKvs[String(layerIdx)]!.k,
          tol.kvCacheProbsMaxAbs,
          `cat9.d19.L${layerIdx}.k`,
        ),
      );
      recordErr(
        'cat9_cropped_kv_vs_pytorch',
        assertFiniteArrayParity(
          s.v,
          d19.croppedLayerKvs[String(layerIdx)]!.v,
          tol.kvCacheProbsMaxAbs,
          `cat9.d19.L${layerIdx}.v`,
        ),
      );
    }
  });

  // --------------------------------------------------------------------------
  // Category 10: Linear Self-Speculation (±LoRA), Greedy T = 0
  // --------------------------------------------------------------------------
  it('Category 10: verifies greedy T=0 AR == LinearSpec == LinearSpecLoRA == PyTorch tokens and 0-ULP verify invariance', () => {
    const fx = FX.cat10LinearSpecGreedyAndLora;
    const tol = FX.metadata.tolerances;

    const arRes = runDiffusionExecution(model, {
      mode: 'ar',
      promptTokenIds: fx.promptTokenIds,
      maxNewTokens: fx.maxNewTokens,
      temperature: 0,
    });
    const specRes = runDiffusionExecution(model, {
      mode: 'linear_spec',
      promptTokenIds: fx.promptTokenIds,
      blockLength: fx.blockLength,
      maxNewTokens: fx.maxNewTokens,
      temperature: 0,
      samplerBThreshold: 0.0,
    });
    const specLoraRes = runDiffusionExecution(model, {
      mode: 'linear_spec_lora',
      promptTokenIds: fx.promptTokenIds,
      blockLength: fx.blockLength,
      maxNewTokens: fx.maxNewTokens,
      temperature: 0,
      samplerBThreshold: 0.0,
    });

    // Exact greedy discrete token parity (zero NFE speedup assertions!)
    expect(Array.from(arRes.generatedTokenIds)).toEqual(fx.arTokens);
    expect(Array.from(specRes.generatedTokenIds)).toEqual(fx.linearSpecTokens);
    expect(Array.from(specLoraRes.generatedTokenIds)).toEqual(fx.linearSpecLoraTokens);
    expect(Array.from(arRes.generatedTokenIds)).toEqual(Array.from(specRes.generatedTokenIds));
    expect(Array.from(specRes.generatedTokenIds)).toEqual(Array.from(specLoraRes.generatedTokenIds));

    expect(arRes.counters).toEqual(fx.arCounters);
    expect(specRes.counters).toEqual(fx.linearSpecCounters);
    expect(specLoraRes.counters).toEqual(fx.linearSpecLoraCounters);

    // 0-ULP verify invariance (loraEnabled = false on verify) on cycle 0 (identical drafted block)
    const c0NoLora = specRes.specCycles[0]!;
    const c0WithLora = specLoraRes.specCycles[0]!;
    expect(c0NoLora.verifyForward!.loraEnabled).toBe(false);
    expect(c0WithLora.verifyForward!.loraEnabled).toBe(false);
    expect(Array.from(c0NoLora.draftedBlockTokens)).toEqual(Array.from(c0WithLora.draftedBlockTokens));
    assertBitExactFiniteArray(
      c0NoLora.verifyForward!.logits,
      c0WithLora.verifyForward!.logits,
      'cat10.verifyLoraOffInvariance',
    );
    recordErr(
      'cat10_verify_logits_vs_pytorch',
      assertFiniteArrayParity(
        c0NoLora.verifyForward!.logits,
        fx.cycle0VerifyLogitsNoLora,
        tol.logitsMaxAbs,
        'cat10.c0VerifyLogits',
      ),
    );

    // Draft logits DO differ between LoRA OFF and LoRA ON
    const prefill8 = arRes.prefillForward.newKvs;
    const draftOff = runDiffusionForwardPass(model, {
      tokenIds: [arRes.generatedTokenIds[0]!, 32, 32, 32],
      maskMode: 'bidirectional',
      loraEnabled: false,
      useCache: false,
      pastKvs: prefill8,
    });
    const draftOn = runDiffusionForwardPass(model, {
      tokenIds: [arRes.generatedTokenIds[0]!, 32, 32, 32],
      maskMode: 'bidirectional',
      loraEnabled: true,
      useCache: false,
      pastKvs: prefill8,
    });
    const draftDiff = computeStrictMaxAbsDiff(draftOn.logits, draftOff.logits, 'cat10.draftLoraDelta');
    expect(draftDiff).toBeGreaterThan(1e-4);
    recordErr(
      'cat10_draft_logits_vs_pytorch',
      assertFiniteArrayParity(draftOff.logits, fx.cycle0DraftLogitsNoLora, tol.logitsMaxAbs, 'cat10.draftOff'),
    );
    recordErr(
      'cat10_draft_logits_vs_pytorch',
      assertFiniteArrayParity(draftOn.logits, fx.cycle0DraftLogitsWithLora, tol.logitsMaxAbs, 'cat10.draftOn'),
    );
  });

  // --------------------------------------------------------------------------
  // Category 11: Stochastic T = 0.7 Explicit-Variate Replay
  // --------------------------------------------------------------------------
  it('Category 11: verifies stochastic T=0.7 explicit-variate replay matches PyTorch per mode and 0-ULP TS replay determinism', () => {
    const fx = FX.cat11StochasticExplicitVariates;
    const tol = FX.metadata.tolerances;

    const ar1 = runDiffusionExecution(model, {
      mode: 'ar',
      promptTokenIds: fx.promptTokenIds,
      maxNewTokens: fx.maxNewTokens,
      temperature: fx.temperature,
      explicitVariates: toTsExplicitVariates(fx.ar.explicitVariates),
    });
    const ar2 = runDiffusionExecution(model, {
      mode: 'ar',
      promptTokenIds: fx.promptTokenIds,
      maxNewTokens: fx.maxNewTokens,
      temperature: fx.temperature,
      explicitVariates: toTsExplicitVariates(fx.ar.explicitVariates),
    });
    expect(Array.from(ar1.generatedTokenIds)).toEqual(fx.ar.generatedTokenIds);
    expect(ar1.counters).toEqual(fx.ar.counters);
    assertBitExactFiniteArray(
      ar1.arSteps[ar1.arSteps.length - 1]!.logits,
      ar2.arSteps[ar2.arSteps.length - 1]!.logits,
      'cat11.ar.tsReplay0Ulp',
    );
    recordErr(
      'cat11_stochastic_logits_vs_pytorch',
      assertFiniteArrayParity(
        ar1.arSteps[ar1.arSteps.length - 1]!.logits,
        fx.ar.lastStepLogits,
        tol.logitsMaxAbs,
        'cat11.ar.lastLogits',
      ),
    );

    const dlm1 = runDiffusionExecution(model, {
      mode: 'dlm',
      promptTokenIds: fx.promptTokenIds,
      blockLength: fx.blockLength,
      maxNewTokens: fx.maxNewTokens,
      causalContext: true,
      temperature: fx.temperature,
      samplerAThreshold: fx.dlmCausal.threshold,
      explicitVariates: toTsExplicitVariates(fx.dlmCausal.explicitVariates),
    });
    expect(Array.from(dlm1.generatedTokenIds)).toEqual(fx.dlmCausal.generatedTokenIds);
    expect(dlm1.counters).toEqual(fx.dlmCausal.counters);
    const dlmLastCommit = dlm1.dlmBlocks[dlm1.dlmBlocks.length - 1]!.commitForward;
    recordErr(
      'cat11_stochastic_logits_vs_pytorch',
      assertFiniteArrayParity(
        dlmLastCommit.logits.subarray((dlmLastCommit.seqLen - 1) * vocabSize),
        fx.dlmCausal.lastBlockCommitLogits,
        tol.logitsMaxAbs,
        'cat11.dlm.commitLogits',
      ),
    );

    const spec1 = runDiffusionExecution(model, {
      mode: 'linear_spec',
      promptTokenIds: fx.promptTokenIds,
      blockLength: fx.blockLength,
      maxNewTokens: fx.maxNewTokens,
      temperature: fx.temperature,
      samplerBThreshold: fx.linearSpec.threshold,
      explicitVariates: toTsExplicitVariates(fx.linearSpec.explicitVariates),
    });
    expect(Array.from(spec1.generatedTokenIds)).toEqual(fx.linearSpec.generatedTokenIds);
    expect(spec1.counters).toEqual(fx.linearSpec.counters);
    recordErr(
      'cat11_stochastic_logits_vs_pytorch',
      assertFiniteArrayParity(
        spec1.specCycles[spec1.specCycles.length - 1]!.verifyForward!.logits,
        fx.linearSpec.lastCycleVerifyLogits,
        tol.logitsMaxAbs,
        'cat11.spec.verifyLogits',
      ),
    );

    const specLora1 = runDiffusionExecution(model, {
      mode: 'linear_spec_lora',
      promptTokenIds: fx.promptTokenIds,
      blockLength: fx.blockLength,
      maxNewTokens: fx.maxNewTokens,
      temperature: fx.temperature,
      samplerBThreshold: fx.linearSpecLora.threshold,
      explicitVariates: toTsExplicitVariates(fx.linearSpecLora.explicitVariates),
    });
    expect(Array.from(specLora1.generatedTokenIds)).toEqual(fx.linearSpecLora.generatedTokenIds);
    expect(specLora1.counters).toEqual(fx.linearSpecLora.counters);
    recordErr(
      'cat11_stochastic_logits_vs_pytorch',
      assertFiniteArrayParity(
        specLora1.specCycles[specLora1.specCycles.length - 1]!.verifyForward!.logits,
        fx.linearSpecLora.lastCycleVerifyLogits,
        tol.logitsMaxAbs,
        'cat11.specLora.verifyLogits',
      ),
    );

    // Same-seed PRNG replay 0-ULP determinism (zero Math.random())
    const prngA = runDiffusionExecution(model, {
      mode: 'linear_spec_lora',
      promptTokenIds: fx.promptTokenIds,
      blockLength: fx.blockLength,
      maxNewTokens: fx.maxNewTokens,
      temperature: fx.temperature,
      samplingSeed: 0x12345678,
    });
    const prngB = runDiffusionExecution(model, {
      mode: 'linear_spec_lora',
      promptTokenIds: fx.promptTokenIds,
      blockLength: fx.blockLength,
      maxNewTokens: fx.maxNewTokens,
      temperature: fx.temperature,
      samplingSeed: 0x12345678,
    });
    expect(Array.from(prngA.generatedTokenIds)).toEqual(Array.from(prngB.generatedTokenIds));
    assertBitExactFiniteArray(
      prngA.specCycles[0]!.verifyForward!.logits,
      prngB.specCycles[0]!.verifyForward!.logits,
      'cat11.prngReplay0Ulp',
    );
  });
});
