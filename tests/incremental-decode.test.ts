import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  createDefaultHybridModel,
  createHybridModelParameters,
  runHybridExecution,
  type GqaBayWeights,
  type MambaBayWeights,
} from '../src/math/hybridFlow';
import { runGqaSim } from '../src/math/gqaSim';
import {
  describeMambaCellFormula,
  runMambaSim,
} from '../src/math/mambaSim';
import { describeCellFormula } from '../src/math/gqaSim';
import {
  createDecodeSession,
  createExternalDecodeCacheState,
  createZeroDecodeCacheState,
  getActiveDecodeStep,
  measureDecodeCacheBytes,
  measureDecodeSessionRetainedBytes,
  navigateDecodeSession,
  runIncrementalDecodeStep,
  stepDecodeSession,
  stepDecodeSessionGreedy,
  switchDecodeSessionBranch,
  validateDecodeCacheState,
} from '../src/math/incrementalDecode';
import type { HybridDecodeCacheState } from '../src/math/cacheContracts';
import { buildMambaValueAtlas } from '../src/scene/mambaAtlas';
import { buildGqaValueAtlas } from '../src/scene/gqaAtlas';
import { computeModelWeightsSha256 } from './helpers/modelWeightDigest';
import {
  assertBitExactFiniteArray,
  assertCausalScoresParity,
  assertFiniteArrayParity,
} from './helpers/parityAssert';

interface IncrementalParityFixture {
  readonly metadata: {
    readonly pinnedTorchVersion: string;
    readonly torchVersion: string;
    readonly tolerances: {
      readonly residualCheckpointMaxAbs: number;
      readonly lmHeadLogitsMaxAbs: number;
      readonly lmHeadProbsMaxAbs: number;
      readonly cacheStateMaxAbs: number;
      readonly gqaScoresMaxAbs: number;
      readonly gqaSoftmaxMaxAbs: number;
    };
  };
  readonly modelId: string;
  readonly parameterFingerprint: string;
  readonly weightsSha256: string;
  readonly totalWeightFloats: number;
  readonly pattern: string;
  readonly steps: readonly {
    readonly stepIndex: number;
    readonly prefixTokens: number;
    readonly newTokenIds: number[];
    readonly residualStreams: number[][];
    readonly logits: number[];
    readonly probs: number[];
    readonly top5TokenIdsByToken: number[][];
    readonly bay01Trace: {
      readonly convState: number[];
      readonly HPrev: number[];
      readonly H: number[];
      readonly Y: number[];
    };
    readonly bay43Trace: {
      readonly K: number[];
      readonly V: number[];
      readonly Scores: Array<number | null>;
      readonly Softmax: number[];
      readonly Y: number[];
    };
  }[];
  readonly finalCacheCheckpoints: {
    readonly mamba: Record<string, { readonly convState: number[]; readonly ssmState: number[] }>;
    readonly gqa: Record<string, { readonly K: number[]; readonly V: number[] }>;
  };
  readonly externalSeededSingleSteps: {
    readonly bay01Mamba: {
      readonly X: number[];
      readonly convStateIn: number[];
      readonly ssmStateIn: number[];
      readonly convStateOut: number[];
      readonly HPrev: number[];
      readonly HOut: number[];
      readonly Y: number[];
    };
    readonly bay43Gqa: {
      readonly prefixTokens: number;
      readonly X: number[];
      readonly KIn: number[];
      readonly VIn: number[];
      readonly KOut: number[];
      readonly VOut: number[];
      readonly Scores: number[];
      readonly Softmax: number[];
      readonly Y: number[];
    };
  };
}

const INC_PARITY_PATH = resolve(__dirname, 'fixtures/pytorch-incremental-52layer-parity.json');
const INC_FIXTURE: IncrementalParityFixture = JSON.parse(readFileSync(INC_PARITY_PATH, 'utf-8'));

/** Slice columns `[start..start+len-1]` out of a `[rows, totalT]` channel-major matrix into `[rows, len]`. */
function sliceCols(mat: Float32Array, rows: number, totalT: number, start: number, len: number): Float32Array {
  const out = new Float32Array(rows * len);
  for (let r = 0; r < rows; r++) {
    for (let t = 0; t < len; t++) {
      out[r * len + t] = mat[r * totalT + (start + t)];
    }
  }
  return out;
}

/** Snapshot all populated bytes of a `HybridDecodeCacheState` so we can prove zero mutation later. */
function copyAllCacheArrays(state: HybridDecodeCacheState): {
  mamba: Map<number, { conv: Float32Array; ssm: Float32Array }>;
  gqa: Map<number, { K: Float32Array; V: Float32Array }>;
} {
  const mamba = new Map<number, { conv: Float32Array; ssm: Float32Array }>();
  for (const [id, m] of state.mambaStates.entries()) {
    mamba.set(id, { conv: new Float32Array(m.convState), ssm: new Float32Array(m.ssmState) });
  }
  const gqa = new Map<number, { K: Float32Array; V: Float32Array }>();
  for (const [id, g] of state.gqaCaches.entries()) {
    gqa.set(id, { K: new Float32Array(g.K), V: new Float32Array(g.V) });
  }
  return { mamba, gqa };
}

describe('Gate A · Cached Incremental AR Decoding (52-Layer Miniature Engine)', () => {
  it('1. enforces exact 75 KiB (76,800 B) cache accounting and exhaustive cache validation', () => {
    const model = createDefaultHybridModel();
    const zeroState = createZeroDecodeCacheState(model);
    expect(() => validateDecodeCacheState(model, zeroState)).not.toThrow();

    // Exact default miniature snapshot accounting:
    //   23 Mamba * (512 + 256) * 4 = 70,656 B (69.00 KiB)
    //    6 GQA   * 2 * (8 * 16) * 4 =  6,144 B ( 6.00 KiB)
    //   23 MoE   * 0                =      0 B
    //   Total allocated             = 76,800 B (75.00 KiB exactly)
    const acct0 = measureDecodeCacheBytes(zeroState);
    expect(acct0.mambaLayerCount).toBe(23);
    expect(acct0.gqaLayerCount).toBe(6);
    expect(acct0.moeLayerCount).toBe(23);
    expect(acct0.mambaFixedBytes).toBe(70_656);
    expect(acct0.gqaAllocatedBytes).toBe(6_144);
    expect(acct0.moeBytes).toBe(0);
    expect(acct0.totalAllocatedBytes).toBe(76_800);
    expect(acct0.totalAllocatedBytes / 1024).toBe(75);
    expect(acct0.bytesPerTokenGqaPopulated).toBe(384);
    expect(acct0.gqaPopulatedBytes).toBe(0);
    expect(acct0.totalPopulatedBytes).toBe(70_656);

    // Populated GQA bytes grow linearly (+384 B/tok) while Mamba stays strictly fixed (70,656 B) and MoE = 0 B
    const step1 = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1, 2, 3],
      prevState: zeroState,
    });
    expect(step1.cacheAccounting.mambaFixedBytes).toBe(70_656);
    expect(step1.cacheAccounting.gqaAllocatedBytes).toBe(6_144);
    expect(step1.cacheAccounting.gqaPopulatedBytes).toBe(4 * 384);
    expect(step1.cacheAccounting.totalAllocatedBytes).toBe(76_800);
    expect(step1.cacheAccounting.moeBytes).toBe(0);

    // Validation rejections:
    // (a) Differently parameterized model with DUPLICATED display modelId must still be rejected!
    const dupIdOtherModel = createHybridModelParameters({
      modelId: model.modelId,
      seed: model.config.seed + 1,
    });
    expect(dupIdOtherModel.modelId).toBe(model.modelId);
    expect(dupIdOtherModel.parameterFingerprint).not.toBe(model.parameterFingerprint);
    expect(() => validateDecodeCacheState(dupIdOtherModel, step1.nextState)).toThrow(
      /parameterFingerprint mismatch/,
    );
    expect(() =>
      runIncrementalDecodeStep(dupIdOtherModel, {
        newTokenIds: [4],
        prevState: step1.nextState,
      }),
    ).toThrow(/parameterFingerprint mismatch/);

    // (b) Missing Mamba bay
    const missingMamba = new Map(step1.nextState.mambaStates);
    missingMamba.delete(1);
    expect(() =>
      validateDecodeCacheState(model, { ...step1.nextState, mambaStates: missingMamba }),
    ).toThrow(/mambaStates count/);

    // (c) Missing GQA bay
    const missingGqa = new Map(step1.nextState.gqaCaches);
    missingGqa.delete(43);
    expect(() =>
      validateDecodeCacheState(model, { ...step1.nextState, gqaCaches: missingGqa }),
    ).toThrow(/gqaCaches count/);

    // (d) Non-finite (NaN / Infinity) inside Mamba state and populated GQA KV cache
    const nanMambaMap = new Map(step1.nextState.mambaStates);
    const badMamba = step1.nextState.mambaStates.get(1)!;
    const badSsm = new Float32Array(badMamba.ssmState);
    badSsm[17] = Number.NaN;
    nanMambaMap.set(1, { ...badMamba, ssmState: badSsm });
    expect(() =>
      validateDecodeCacheState(model, { ...step1.nextState, mambaStates: nanMambaMap }),
    ).toThrow(/Non-finite/);

    const infGqaMap = new Map(step1.nextState.gqaCaches);
    const badGqa = step1.nextState.gqaCaches.get(6)!;
    const badK = new Float32Array(badGqa.K);
    badK[2] = Number.POSITIVE_INFINITY; // populated slot k=2 < seqLen=4
    infGqaMap.set(6, { ...badGqa, K: badK });
    expect(() =>
      validateDecodeCacheState(model, { ...step1.nextState, gqaCaches: infGqaMap }),
    ).toThrow(/Non-finite/);

    // (e) Wrong dimension / seqLen mismatch / capacity overflow
    const wrongLenMamba = new Map(step1.nextState.mambaStates);
    wrongLenMamba.set(1, { ...badMamba, convState: new Float32Array(128) });
    expect(() =>
      validateDecodeCacheState(model, { ...step1.nextState, mambaStates: wrongLenMamba }),
    ).toThrow(/convState length/);

    const wrongSeqGqa = new Map(step1.nextState.gqaCaches);
    wrongSeqGqa.set(6, { ...badGqa, seqLen: 3 });
    expect(() =>
      validateDecodeCacheState(model, { ...step1.nextState, gqaCaches: wrongSeqGqa }),
    ).toThrow(/seqLen/);

    // (f) Capacity overflow on step
    expect(() =>
      runIncrementalDecodeStep(model, {
        newTokenIds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        prevState: step1.nextState, // 4 + 13 = 17 > maxTokens (16)
      }),
    ).toThrow(/exceeds cache capacity/);
  });

  it('2. achieves bit-exact (0 ulp) equivalence across full-sequence, prefill+AR 1-by-1, and arbitrary chunk partitions', () => {
    const model = createDefaultHybridModel();
    const fullTokens = [2, 5, 2, 11, 0, 19, 7, 14];
    const fullExec = runHybridExecution(model, { tokenIds: fullTokens });

    // Partition A: Prefill 3 + single-token AR decode 1+1+1+1+1
    const partitions: readonly (readonly number[])[] = [
      [3, 1, 1, 1, 1, 1],
      [4, 4],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [2, 3, 1, 2],
    ];

    const finalDigests: string[] = [];

    for (const chunkSizes of partitions) {
      let state: HybridDecodeCacheState | null = null;
      let cursor = 0;
      let lastStep: ReturnType<typeof runIncrementalDecodeStep> | null = null;

      for (let sIdx = 0; sIdx < chunkSizes.length; sIdx++) {
        const len = chunkSizes[sIdx];
        const slice = fullTokens.slice(cursor, cursor + len);
        const step = runIncrementalDecodeStep(model, {
          newTokenIds: slice,
          prevState: state,
          stepIndex: sIdx,
          parentStepExecutionId: lastStep?.stepExecutionId ?? null,
          initialStateDigest: lastStep?.provenance.initialStateDigest,
        });

        // Verify ALL 53 residual checkpoints match the full-sequence pass BIT-EXACT (0 ulp!)
        for (let k = 0; k <= 52; k++) {
          const refStream = sliceCols(fullExec.residualStreams[k], 8, fullTokens.length, cursor, len);
          assertBitExactFiniteArray(
            step.residualStreams[k],
            refStream,
            `chunk[${chunkSizes.join('+')}].step${sIdx}.residual[${k}]`,
          );
        }

        // Verify Final RMSNorm, Untied LM Head logits & probabilities BIT-EXACT
        assertBitExactFiniteArray(
          step.logits,
          sliceCols(fullExec.lmHead.logits, 32, fullTokens.length, cursor, len),
          `chunk[${chunkSizes.join('+')}].step${sIdx}.logits`,
        );
        assertBitExactFiniteArray(
          step.probs,
          sliceCols(fullExec.lmHead.probs, 32, fullTokens.length, cursor, len),
          `chunk[${chunkSizes.join('+')}].step${sIdx}.probs`,
        );

        // Verify ALL 23 MoE routers pick bit-identical Top-6 expert indices
        for (let b = 0; b < 52; b++) {
          const sum = step.baySummaries[b];
          if (sum.kind === 'E') {
            const fullIdx = fullExec.baySummaries[b].moeTopkIndices!;
            const expectedIdx = fullIdx.subarray(cursor * 6, (cursor + len) * 6);
            expect(Array.from(sum.moeTopkIndices!)).toEqual(Array.from(expectedIdx));
          }
        }

        state = step.nextState;
        lastStep = step;
        cursor += len;
      }

      expect(cursor).toBe(fullTokens.length);
      finalDigests.push(state!.stateDigest);

      // Verify final Mamba states (all 23 bays) and final GQA KV caches (all 6 bays) match full execution traces bit-exact
      for (const bayId of [1, 12, 33, 51]) {
        const fullM = fullExec.getMambaTrace(bayId);
        const cachedM = state!.mambaStates.get(bayId)!;
        assertBitExactFiniteArray(cachedM.convState, fullM.conv_buf, `final.mamba[${bayId}].convState`);
        assertBitExactFiniteArray(cachedM.ssmState, fullM.H, `final.mamba[${bayId}].ssmState`);
      }
      for (const bayId of [6, 13, 20, 27, 34, 43]) {
        const fullG = fullExec.getGqaTrace(bayId);
        const cachedG = state!.gqaCaches.get(bayId)!;
        expect(cachedG.seqLen).toBe(fullTokens.length);
        for (let row = 0; row < 8; row++) {
          for (let k = 0; k < fullTokens.length; k++) {
            expect(cachedG.K[row * cachedG.maxTokens + k]).toBe(fullG.K[row * fullTokens.length + k]);
            expect(cachedG.V[row * cachedG.maxTokens + k]).toBe(fullG.V[row * fullTokens.length + k]);
          }
        }
      }
    }

    // Every chunking strategy reaches the EXACT SAME 64-bit boundary state digest!
    expect(new Set(finalDigests).size).toBe(1);
  });

  it('3. supports externally supplied boundary states (midpoint resume + arbitrary synthetic non-zero seeds + defensive isolation)', () => {
    const model = createDefaultHybridModel();
    const fullTokens = [0, 1, 2, 3, 4, 5];
    const fullExec = runHybridExecution(model, { tokenIds: fullTokens });

    // (a) Midpoint resume via createExternalDecodeCacheState WITHOUT replaying prefix tokens
    const prefixStep = runIncrementalDecodeStep(model, {
      newTokenIds: [0, 1, 2, 3],
      prevState: null,
    });

    const rawExternalMamba = new Map<number, { convState: Float32Array; ssmState: Float32Array }>();
    for (const [id, m] of prefixStep.nextState.mambaStates.entries()) {
      rawExternalMamba.set(id, {
        convState: new Float32Array(m.convState),
        ssmState: new Float32Array(m.ssmState),
      });
    }
    const rawExternalGqa = new Map<number, { K: Float32Array; V: Float32Array; maxTokens: number }>();
    for (const [id, g] of prefixStep.nextState.gqaCaches.entries()) {
      rawExternalGqa.set(id, {
        K: new Float32Array(g.K),
        V: new Float32Array(g.V),
        maxTokens: g.maxTokens,
      });
    }

    const externalSnapshot = createExternalDecodeCacheState(model, {
      processedTokens: 4,
      maxTokens: 16,
      prefixTokenIds: [0, 1, 2, 3],
      gqaCaches: rawExternalGqa,
      mambaStates: rawExternalMamba,
    });

    // Mutate caller's raw input buffers AFTER snapshot creation -> MUST NOT affect externalSnapshot!
    rawExternalMamba.get(1)!.ssmState.fill(999.0);
    rawExternalMamba.get(1)!.convState.fill(-999.0);
    rawExternalGqa.get(43)!.K.fill(777.0);
    rawExternalGqa.get(43)!.V.fill(-777.0);

    const resumed = runIncrementalDecodeStep(model, {
      newTokenIds: [4, 5],
      prevState: externalSnapshot,
    });
    assertBitExactFiniteArray(
      resumed.logits,
      sliceCols(fullExec.lmHead.logits, 32, 6, 4, 2),
      'externalMidpoint.logits',
    );
    assertBitExactFiniteArray(
      resumed.residualStreams[52],
      sliceCols(fullExec.residualStreams[52], 8, 6, 4, 2),
      'externalMidpoint.residual52',
    );

    // (b) Arbitrary synthetic non-zero boundary state executes cleanly with truthful seeded H_prev, conv_buf & rectangular GQA
    const synthMamba = new Map<number, { convState: Float32Array; ssmState: Float32Array }>();
    const synthGqa = new Map<number, { K: Float32Array; V: Float32Array; maxTokens: number }>();
    for (let i = 0; i < model.config.pattern.length; i++) {
      const bayId = i + 1;
      const kind = model.config.pattern[i];
      if (kind === 'M') {
        const conv = new Float32Array(128 * 4);
        const ssm = new Float32Array(16 * 4 * 4);
        for (let j = 0; j < conv.length; j++) conv[j] = Math.fround(0.25 * Math.sin(bayId + j * 0.17));
        for (let j = 0; j < ssm.length; j++) ssm[j] = Math.fround(0.35 * Math.cos(bayId + j * 0.23));
        synthMamba.set(bayId, { convState: conv, ssmState: ssm });
      } else if (kind === '*') {
        const K = new Float32Array(8 * 16);
        const V = new Float32Array(8 * 16);
        for (let row = 0; row < 8; row++) {
          for (let k = 0; k < 3; k++) {
            K[row * 16 + k] = Math.fround(0.4 * Math.sin(bayId + row * 0.5 + k));
            V[row * 16 + k] = Math.fround(0.4 * Math.cos(bayId + row * 0.5 - k));
          }
        }
        synthGqa.set(bayId, { K, V, maxTokens: 16 });
      }
    }

    const synthSeed = createExternalDecodeCacheState(model, {
      processedTokens: 3,
      maxTokens: 16,
      prefixTokenLabels: ['ext0', 'ext1', 'ext2'],
      gqaCaches: synthGqa,
      mambaStates: synthMamba,
    });
    const synthStep = runIncrementalDecodeStep(model, {
      newTokenIds: [9],
      prevState: synthSeed,
    });

    expect(synthStep.provenance.stepKind).toBe('decode');
    expect(synthStep.provenance.tokenRange).toEqual([3, 3]);

    // Verify Mamba Bay 01 trace truthfully exposes seeded H_prev and seeded conv_buf taps (never falsely 0!)
    const m1 = synthStep.getMambaTrace(1);
    expect(m1.tokens).toBe(1);
    expect(m1.startTokenOffset).toBe(3);
    expect(m1.hasInitialState).toBe(true);
    assertBitExactFiniteArray(m1.H_prev, synthSeed.mambaStates.get(1)!.ssmState, 'synth.bay01.H_prev');
    const hPrevFormula = describeMambaCellFormula(m1, { tensor: 'h_prev', row: 0, col: 0, headIndex: 0, tokenIndex: 0 });
    expect(hPrevFormula.value).toBe(synthSeed.mambaStates.get(1)!.ssmState[0]);
    expect(hPrevFormula.equation).toContain('carried FP32 state');
    const convSeedFormula = describeMambaCellFormula(m1, { tensor: 'conv_buf', row: 2, col: 5, tapIndex: 2, tokenIndex: 0 });
    expect(convSeedFormula.value).toBe(synthSeed.mambaStates.get(1)!.convState[5 * 4 + 3]);
    expect(convSeedFormula.equation).toContain('carried conv_state');
    expect(convSeedFormula.upstreamSpans).toHaveLength(0);
    const mAtlas = buildMambaValueAtlas(m1, 0, 0);
    // H_prev texel (210, 0) in atlas matches seeded ssmState[0]
    expect(mAtlas.data[(0 * 256 + 210) * 2]).toBe(synthSeed.mambaStates.get(1)!.ssmState[0]);

    // Verify GQA Bay 43 trace truthfully represents rectangular [32, 1, 4] attention & distinguishes reused vs new KV
    const g43 = synthStep.getGqaTrace(43);
    expect(g43.tokens).toBe(1);
    expect(g43.prefixTokens).toBe(3);
    expect(g43.keyTokens).toBe(4);
    expect(g43.Scores.length).toBe(32 * 1 * 4);
    expect(g43.Softmax.length).toBe(32 * 1 * 4);
    const reusedKvFormula = describeCellFormula(g43, { tensor: 'kv_k', kvHeadIndex: 0, tokenIndex: 1, row: 0, col: 0 });
    expect(reusedKvFormula.tensorLabel).toContain('Reused');
    expect(reusedKvFormula.upstreamSpans).toHaveLength(0);
    const newKvFormula = describeCellFormula(g43, { tensor: 'kv_k', kvHeadIndex: 0, tokenIndex: 3, row: 0, col: 0 });
    expect(newKvFormula.tensorLabel).toContain('New');
    expect(newKvFormula.upstreamSpans).toHaveLength(2);
    const gAtlas = buildGqaValueAtlas(g43, 0, 16);
    expect(gAtlas.kvK[0]).toHaveLength(4);
    expect(gAtlas.scores[0].rect[2]).toBeCloseTo(4 / 128, 6); // width = 4 keys
    expect(gAtlas.scores[0].rect[3]).toBeCloseTo(1 / 64, 6);  // height = 1 query row
  });

  it('4. preserves snapshot immutability, branch divergence isolation, and trace provenance/origin distinction', () => {
    const model = createDefaultHybridModel();
    const session0 = createDecodeSession(model, { promptTokenIds: [0, 1, 2, 3] });
    const rootStep = getActiveDecodeStep(session0);
    const rootCacheCopy = copyAllCacheArrays(rootStep.nextState);

    // Step along branch-0 using greedy candidate
    const session1 = stepDecodeSessionGreedy(session0);
    const stepA = getActiveDecodeStep(session1);
    const stepACacheCopy = copyAllCacheArrays(stepA.nextState);

    // Navigate back to step 0 and fork branch-1 with an alternative Top-5 candidate (rank #2)
    const altCandidate = rootStep.predictedNextTokens[1];
    expect(altCandidate.tokenId).not.toBe(rootStep.greedyNextToken.tokenId);
    const rewound = navigateDecodeSession(session1, 0);
    const session2 = stepDecodeSession(rewound, altCandidate.tokenId);
    const stepB = getActiveDecodeStep(session2);

    // 1. Branch provenance is distinct and unambiguous
    expect(stepA.provenance.branchId).toBe('branch-0');
    expect(stepB.provenance.branchId).toBe('branch-1');
    expect(stepA.provenance.parentStepExecutionId).toBe(rootStep.stepExecutionId);
    expect(stepB.provenance.parentStepExecutionId).toBe(rootStep.stepExecutionId);
    expect(stepA.provenance.initialStateDigest).toBe(rootStep.provenance.initialStateDigest);
    expect(stepB.provenance.initialStateDigest).toBe(rootStep.provenance.initialStateDigest);
    expect(stepA.stepExecutionId).not.toBe(stepB.stepExecutionId);
    expect(stepA.nextState.stateDigest).not.toBe(stepB.nextState.stateDigest);
    expect(stepA.provenance.tokenRange).toEqual([4, 4]);
    expect(stepB.provenance.tokenRange).toEqual([4, 4]);

    // 2. Prior snapshots (rootStep and stepA) are BIT-FOR-BIT UNCHANGED after stepping & branching!
    for (const [id, m] of rootStep.nextState.mambaStates.entries()) {
      assertBitExactFiniteArray(m.convState, rootCacheCopy.mamba.get(id)!.conv, `root.immut.mamba[${id}].conv`);
      assertBitExactFiniteArray(m.ssmState, rootCacheCopy.mamba.get(id)!.ssm, `root.immut.mamba[${id}].ssm`);
    }
    for (const [id, g] of rootStep.nextState.gqaCaches.entries()) {
      assertBitExactFiniteArray(g.K, rootCacheCopy.gqa.get(id)!.K, `root.immut.gqa[${id}].K`);
      assertBitExactFiniteArray(g.V, rootCacheCopy.gqa.get(id)!.V, `root.immut.gqa[${id}].V`);
    }
    const branch0Again = switchDecodeSessionBranch(session2, 'branch-0');
    const stepAAgain = getActiveDecodeStep(branch0Again);
    expect(stepAAgain.stepExecutionId).toBe(stepA.stepExecutionId);
    for (const [id, g] of stepAAgain.nextState.gqaCaches.entries()) {
      assertBitExactFiniteArray(g.K, stepACacheCopy.gqa.get(id)!.K, `branch0.immut.gqa[${id}].K`);
    }

    // 3. Trace origin distinguishes eagerly computed step traces vs. lazily reconstructed traces
    const eagerTrace = stepA.getLayerTrace(43);
    expect(eagerTrace.traceOrigin).toBe('computed_this_step');
    expect(stepA.getLayerTraceOrigin(43)).toBe('computed_this_step');
    expect(stepA.getLayerTraceOrigin(20)).toBeNull(); // Bay 20 (GQA) not in default eager set

    const rebuiltTrace = stepA.getLayerTrace(20);
    expect(rebuiltTrace.traceOrigin).toBe('reconstructed_from_checkpoint');
    expect(stepA.getLayerTraceOrigin(20)).toBe('reconstructed_from_checkpoint');
    // Rebuilt trace matches full sequence bit-exact!
    const fullA = runHybridExecution(model, { tokenIds: Array.from(stepA.sequenceTokenIds) });
    assertBitExactFiniteArray(
      rebuiltTrace.sim.Y,
      sliceCols(fullA.getGqaTrace(20).Y, 8, 5, 4, 1),
      'rebuilt.bay20.Y',
    );

    // Session memory accounting is finite and deduplicated
    const stats = measureDecodeSessionRetainedBytes(session2);
    expect(stats.branchCount).toBe(2);
    expect(stats.uniqueSteps).toBe(3); // shared root + stepA + stepB
    expect(stats.totalRetainedBytes).toBeGreaterThan(3 * 76_800);
  });

  it('5. matches pinned PyTorch 2.15.0 across 52-layer cached prefill+AR decode and externally seeded single steps', () => {
    const model = createDefaultHybridModel();
    const digest = computeModelWeightsSha256(model);

    expect(INC_FIXTURE.metadata.pinnedTorchVersion).toBe('2.15.0');
    expect(model.modelId).toBe(INC_FIXTURE.modelId);
    expect(model.parameterFingerprint).toBe(INC_FIXTURE.parameterFingerprint);
    expect(digest.sha256).toBe(INC_FIXTURE.weightsSha256);

    const tol = INC_FIXTURE.metadata.tolerances;
    let state: HybridDecodeCacheState | null = null;

    for (const refStep of INC_FIXTURE.steps) {
      const step = runIncrementalDecodeStep(model, {
        newTokenIds: refStep.newTokenIds,
        prevState: state,
        stepIndex: refStep.stepIndex,
      });

      expect(step.prefixTokens).toBe(refStep.prefixTokens);
      expect(step.residualStreams).toHaveLength(53);

      // All 53 step residual checkpoints match PyTorch 2.15.0
      assertBitExactFiniteArray(step.residualStreams[0], refStep.residualStreams[0], `pt.step${refStep.stepIndex}.embed`);
      for (let k = 1; k <= 52; k++) {
        assertFiniteArrayParity(
          step.residualStreams[k],
          refStep.residualStreams[k],
          tol.residualCheckpointMaxAbs,
          `pt.step${refStep.stepIndex}.residual[${k}]`,
        );
      }

      // Logits, probs & Top-5 token rankings match PyTorch 2.15.0
      assertFiniteArrayParity(step.logits, refStep.logits, tol.lmHeadLogitsMaxAbs, `pt.step${refStep.stepIndex}.logits`);
      assertFiniteArrayParity(step.probs, refStep.probs, tol.lmHeadProbsMaxAbs, `pt.step${refStep.stepIndex}.probs`);
      for (let t = 0; t < refStep.newTokenIds.length; t++) {
        const actualTop5 = step.lmHead.topKByToken[t].map((p) => p.tokenId);
        expect(actualTop5).toEqual(refStep.top5TokenIdsByToken[t]);
      }

      // Bay 01 Mamba-2 & Bay 43 rectangular GQA traces match PyTorch 2.15.0
      const m1 = step.getMambaTrace(1);
      assertFiniteArrayParity(m1.conv_buf, refStep.bay01Trace.convState, tol.cacheStateMaxAbs, `pt.step${refStep.stepIndex}.bay01.conv`);
      assertFiniteArrayParity(m1.H_prev, refStep.bay01Trace.HPrev, tol.cacheStateMaxAbs, `pt.step${refStep.stepIndex}.bay01.HPrev`);
      assertFiniteArrayParity(m1.H, refStep.bay01Trace.H, tol.cacheStateMaxAbs, `pt.step${refStep.stepIndex}.bay01.H`);
      assertFiniteArrayParity(m1.Y, refStep.bay01Trace.Y, tol.residualCheckpointMaxAbs, `pt.step${refStep.stepIndex}.bay01.Y`);

      const g43 = step.getGqaTrace(43);
      assertFiniteArrayParity(g43.K, refStep.bay43Trace.K, tol.cacheStateMaxAbs, `pt.step${refStep.stepIndex}.bay43.K`);
      assertFiniteArrayParity(g43.V, refStep.bay43Trace.V, tol.cacheStateMaxAbs, `pt.step${refStep.stepIndex}.bay43.V`);
      assertCausalScoresParity(g43.Scores, refStep.bay43Trace.Scores, tol.gqaScoresMaxAbs, `pt.step${refStep.stepIndex}.bay43.Scores`);
      assertFiniteArrayParity(g43.Softmax, refStep.bay43Trace.Softmax, tol.gqaSoftmaxMaxAbs, `pt.step${refStep.stepIndex}.bay43.Softmax`);
      assertFiniteArrayParity(g43.Y, refStep.bay43Trace.Y, tol.residualCheckpointMaxAbs, `pt.step${refStep.stepIndex}.bay43.Y`);

      state = step.nextState;
    }

    // Final cached Mamba & GQA states match PyTorch 2.15.0 across bays
    for (const [bayKey, refM] of Object.entries(INC_FIXTURE.finalCacheCheckpoints.mamba)) {
      const m = state!.mambaStates.get(Number(bayKey))!;
      assertFiniteArrayParity(m.convState, refM.convState, tol.cacheStateMaxAbs, `pt.final.mamba[${bayKey}].conv`);
      assertFiniteArrayParity(m.ssmState, refM.ssmState, tol.cacheStateMaxAbs, `pt.final.mamba[${bayKey}].ssm`);
    }
    for (const [bayKey, refG] of Object.entries(INC_FIXTURE.finalCacheCheckpoints.gqa)) {
      const g = state!.gqaCaches.get(Number(bayKey))!;
      const flatK = new Float32Array(8 * 6);
      const flatV = new Float32Array(8 * 6);
      for (let row = 0; row < 8; row++) {
        for (let k = 0; k < 6; k++) {
          flatK[row * 6 + k] = g.K[row * g.maxTokens + k];
          flatV[row * 6 + k] = g.V[row * g.maxTokens + k];
        }
      }
      assertFiniteArrayParity(flatK, refG.K, tol.cacheStateMaxAbs, `pt.final.gqa[${bayKey}].K`);
      assertFiniteArrayParity(flatV, refG.V, tol.cacheStateMaxAbs, `pt.final.gqa[${bayKey}].V`);
    }

    // Externally seeded single-step parity against PyTorch 2.15.0 (Bay 01 Mamba + Bay 43 GQA)
    const extM = INC_FIXTURE.externalSeededSingleSteps.bay01Mamba;
    const w1 = model.getBayWeights(1) as MambaBayWeights;
    const mExtSim = runMambaSim(1, {
      X: new Float32Array(extM.X),
      gamma_in: w1.gamma_in,
      W_in: w1.W_in,
      w_conv: w1.w_conv,
      b_conv: w1.b_conv,
      A_log: w1.A_log,
      dt_bias: w1.dt_bias,
      D: w1.D,
      gamma_gate: w1.gamma_gate,
      W_out: w1.W_out,
      eps: model.config.rmsNormEps,
      initialConvState: new Float32Array(extM.convStateIn),
      initialSsmState: new Float32Array(extM.ssmStateIn),
      startTokenOffset: 5,
    });
    assertFiniteArrayParity(mExtSim.conv_buf, extM.convStateOut, tol.cacheStateMaxAbs, 'pt.ext.mamba.convOut');
    assertBitExactFiniteArray(mExtSim.H_prev, extM.HPrev, 'pt.ext.mamba.HPrev');
    assertFiniteArrayParity(mExtSim.H, extM.HOut, tol.cacheStateMaxAbs, 'pt.ext.mamba.HOut');
    assertFiniteArrayParity(mExtSim.Y, extM.Y, tol.residualCheckpointMaxAbs, 'pt.ext.mamba.Y');

    const extG = INC_FIXTURE.externalSeededSingleSteps.bay43Gqa;
    const w43 = model.getBayWeights(43) as GqaBayWeights;
    const gExtSim = runGqaSim(1, {
      X: new Float32Array(extG.X),
      gamma: w43.gamma,
      W_Q: w43.W_Q,
      W_K: w43.W_K,
      W_V: w43.W_V,
      W_O: w43.W_O,
      rmsEps: model.config.rmsNormEps,
      kvCacheIn: {
        seqLen: extG.prefixTokens,
        maxTokens: extG.prefixTokens + 1,
        K: (() => {
          const buf = new Float32Array(8 * (extG.prefixTokens + 1));
          for (let r = 0; r < 8; r++) {
            for (let k = 0; k < extG.prefixTokens; k++) {
              buf[r * (extG.prefixTokens + 1) + k] = extG.KIn[r * extG.prefixTokens + k];
            }
          }
          return buf;
        })(),
        V: (() => {
          const buf = new Float32Array(8 * (extG.prefixTokens + 1));
          for (let r = 0; r < 8; r++) {
            for (let k = 0; k < extG.prefixTokens; k++) {
              buf[r * (extG.prefixTokens + 1) + k] = extG.VIn[r * extG.prefixTokens + k];
            }
          }
          return buf;
        })(),
      },
    });
    assertFiniteArrayParity(gExtSim.K, extG.KOut, tol.cacheStateMaxAbs, 'pt.ext.gqa.KOut');
    assertFiniteArrayParity(gExtSim.V, extG.VOut, tol.cacheStateMaxAbs, 'pt.ext.gqa.VOut');
    assertFiniteArrayParity(gExtSim.Scores, extG.Scores, tol.gqaScoresMaxAbs, 'pt.ext.gqa.Scores');
    assertFiniteArrayParity(gExtSim.Softmax, extG.Softmax, tol.gqaSoftmaxMaxAbs, 'pt.ext.gqa.Softmax');
    assertFiniteArrayParity(gExtSim.Y, extG.Y, tol.residualCheckpointMaxAbs, 'pt.ext.gqa.Y');
  });
});
