import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';

import type { GqaLayerKvCache } from '../src/math/cacheContracts';
import { runGqaSim } from '../src/math/gqaSim';
import { applyRmsNorm } from '../src/math/hybridFlow';
import { runMambaSim } from '../src/math/mambaSim';
import { runMoeSim } from '../src/math/moeSim';
import {
  applyAdaLnModulate,
  blockContainsEos,
  buildContextConditioningState,
  collectTwoTowerCanonicalWeights,
  computeGlobalTimestepProjection,
  computeLayerModulationParams,
  computeSinusoidalTimestepEmbedding,
  createDefaultTwoTowerModel,
  createTwoTowerModelParameters,
  generateTwoTowerDiffusion,
  mdlmConstrainLogProbs,
  modulationOrderForBayKind,
  runConfidenceUnmaskStep,
  runTwoTowerArStep,
  runTwoTowerContextPrefill,
  runTwoTowerDenoiserPass,
  runTwoTowerMockArStep,
  type TwoTowerModelParameters,
} from '../src/twotower';
import {
  assertBitExactFiniteArray,
  assertCausalScoresParity,
  assertFiniteArrayParity,
  assertFiniteNumber,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';

// ============================================================================
// FIXTURE TYPES & LOADING
// ============================================================================

interface BlocksParityFixture {
  readonly metadata: {
    readonly generator: string;
    readonly pythonVersion: string;
    readonly torchVersion: string;
    readonly pinnedTorchVersion: string;
    readonly referenceArchitecture: string;
    readonly precisionNote: string;
    readonly tolerances: {
      readonly modulationMaxAbs: number;
      readonly gqaMaxAbs: number;
      readonly mamba2MaxAbs: number;
      readonly mamba2HighMagRel: number;
      readonly moeMaxAbs: number;
    };
  };
  readonly modelId: string;
  readonly twoTowerFingerprint: string;
  readonly contextFingerprint: string;
  readonly denoiserFingerprint: string;
  readonly timeConditioningFingerprint: string;
  readonly weightsSha256: string;
  readonly totalWeightFloats: number;
  readonly cat1TimestepAndModulation: {
    readonly timesteps: Array<{
      readonly t: number;
      readonly phi: number[];
      readonly tRepr: number[];
      readonly tEmb: number[];
      readonly layers: Array<{
        readonly layerIndex: number;
        readonly bayId: number;
        readonly kind: 'M' | 'E' | '*';
        readonly shift: number[];
        readonly scale: number[];
        readonly gate: number[];
      }>;
    }>;
    readonly orderInversionSensitivity: {
      readonly t: number;
      readonly tokens: number;
      readonly X: number[];
      readonly bay01Mamba: {
        readonly bayId: number;
        readonly layerIndex: number;
        readonly kind: 'M';
        readonly correctOrder: 'modulate_then_norm';
        readonly U_correct: number[];
        readonly U_inverted: number[];
        readonly Y_correct: number[];
        readonly Y_inverted: number[];
        readonly maxAbsDeltaU: number;
        readonly maxAbsDeltaY: number;
      };
      readonly bay06Gqa: {
        readonly bayId: number;
        readonly layerIndex: number;
        readonly kind: '*';
        readonly correctOrder: 'modulate_then_norm';
        readonly U_correct: number[];
        readonly U_inverted: number[];
        readonly Y_correct: number[];
        readonly Y_inverted: number[];
        readonly maxAbsDeltaU: number;
        readonly maxAbsDeltaY: number;
      };
      readonly bay02Moe: {
        readonly bayId: number;
        readonly layerIndex: number;
        readonly kind: 'E';
        readonly correctOrder: 'norm_then_modulate';
        readonly U_correct: number[];
        readonly U_inverted: number[];
        readonly Y_correct: number[];
        readonly Y_inverted: number[];
        readonly maxAbsDeltaU: number;
        readonly maxAbsDeltaY: number;
      };
    };
  };
  readonly cat2BidirectionalGqa: {
    readonly bayId: number;
    readonly layerIndex: number;
    readonly t: number;
    readonly prefixLen: number;
    readonly blockSize: number;
    readonly kvCapacity: number;
    readonly eps: number;
    readonly shift: number[];
    readonly scale: number[];
    readonly gate: number[];
    readonly KPrefix: number[];
    readonly VPrefix: number[];
    readonly X: number[];
    readonly bidirectional: {
      readonly X_mod: number[];
      readonly U: number[];
      readonly Q: number[];
      readonly K: number[];
      readonly V: number[];
      readonly Scores: number[];
      readonly Softmax: number[];
      readonly V_out: number[];
      readonly Attn_out: number[];
      readonly Y: number[];
    };
    readonly futureTokenInfluence: {
      readonly perturbedSlot: number;
      readonly observedEarlierSlot: number;
      readonly X_a: number[];
      readonly X_b: number[];
      readonly Y_bidir_a: number[];
      readonly Y_bidir_b: number[];
      readonly Y_causal_a: number[];
      readonly Y_causal_b: number[];
      readonly causalScoresA: Array<number | null>;
      readonly bidirToken0MaxDiff: number;
      readonly causalToken0MaxDiff: number;
    };
  };
  readonly cat3SeededMamba2: {
    readonly bayId: number;
    readonly layerIndex: number;
    readonly t: number;
    readonly eps: number;
    readonly shift: number[];
    readonly scale: number[];
    readonly gate: number[];
    readonly modulatedSeeded: {
      readonly tokens: number;
      readonly X: number[];
      readonly convStateIn: number[];
      readonly ssmStateIn: number[];
      readonly X_mod: number[];
      readonly U: number[];
      readonly conv_out: number[];
      readonly convStateOut: number[];
      readonly dt: number[];
      readonly dA: number[];
      readonly H_prev: number[];
      readonly H: number[];
      readonly y: number[];
      readonly y_norm: number[];
      readonly Mamba_out: number[];
      readonly Y: number[];
    };
    readonly prefixStitching: {
      readonly prefixLen: number;
      readonly blockSize: number;
      readonly X_full: number[];
      readonly Y_full_tail: number[];
      readonly Y_stitched: number[];
      readonly convStateFull: number[];
      readonly convStateStitched: number[];
      readonly HFull: number[];
      readonly HStitched: number[];
      readonly pytorchStitchMaxDiff: number;
    };
    readonly highMagnitudeStability: {
      readonly tokens: number;
      readonly X: number[];
      readonly convStateIn: number[];
      readonly ssmStateIn: number[];
      readonly conv_out: number[];
      readonly dt: number[];
      readonly dA: number[];
      readonly H: number[];
      readonly y: number[];
      readonly y_norm: number[];
      readonly Mamba_out: number[];
      readonly Y: number[];
    };
  };
  readonly cat4Fp64MoeRouting: {
    readonly bayId: number;
    readonly layerIndex: number;
    readonly t: number;
    readonly eps: number;
    readonly shift: number[];
    readonly scale: number[];
    readonly gate: number[];
    readonly modulatedFp64: {
      readonly tokens: number;
      readonly X: number[];
      readonly U: number[];
      readonly U_mod: number[];
      readonly scores: number[];
      readonly scores_for_choice: number[];
      readonly topk_indices: number[];
      readonly topk_raw_scores: number[];
      readonly topk_weights: number[];
      readonly y_shared: number[];
      readonly y_routed: number[];
      readonly y_moe: number[];
      readonly Y: number[];
    };
    readonly nearTieRouter: {
      readonly tokens: number;
      readonly expertWeightsBayId: number;
      readonly X: number[];
      readonly gamma_in: number[];
      readonly W_router: number[];
      readonly e_score_correction_bias: number[];
      readonly float64MarginRank6VsRank7: number;
      readonly float64TopkIndices: number[];
      readonly float64TopkWeights: number[];
      readonly float64Y: number[];
      readonly float32TopkIndices: number[];
      readonly float32TopkWeights: number[];
      readonly float32Y: number[];
    };
    readonly exactTieRouter: {
      readonly tokens: number;
      readonly expertWeightsBayId: number;
      readonly X: number[];
      readonly gamma_in: number[];
      readonly W_router: number[];
      readonly e_score_correction_bias: number[];
      readonly tiedExperts: number[];
      readonly expectedTopkIndices: number[];
      readonly expectedTopkWeights: number[];
      readonly expectedY: number[];
    };
  };
}

interface HandoffParityFixture {
  readonly metadata: {
    readonly generator: string;
    readonly pythonVersion: string;
    readonly torchVersion: string;
    readonly pinnedTorchVersion: string;
    readonly referenceArchitecture: string;
    readonly tolerances: {
      readonly embeddingMaxAbs: number;
      readonly modulationMaxAbs: number;
      readonly cacheStateMaxAbs: number;
      readonly residualCheckpointMaxAbs: number;
      readonly traceMaxAbs: number;
      readonly finalNormMaxAbs: number;
      readonly lmHeadLogitsMaxAbs: number;
      readonly lmHeadProbsMaxAbs: number;
    };
  };
  readonly modelId: string;
  readonly twoTowerFingerprint: string;
  readonly contextFingerprint: string;
  readonly denoiserFingerprint: string;
  readonly timeConditioningFingerprint: string;
  readonly weightsSha256: string;
  readonly totalWeightFloats: number;
  readonly promptTokenIds: number[];
  readonly kvCapacity: number;
  readonly blockTokenIds: number[];
  readonly blockId: number;
  readonly refinementIndex: number;
  readonly tModel: number;
  readonly contextPrefill: {
    readonly prefixLen: number;
    readonly arLogits: number[];
    readonly arProbs: number[];
    readonly mambaS1: Record<string, { readonly convState: number[]; readonly ssmState: number[] }>;
    readonly mambaS2: Record<string, { readonly convState: number[]; readonly ssmState: number[] }>;
    readonly gqaKv: Record<string, { readonly K: number[]; readonly V: number[] }>;
  };
  readonly denoiserPass: {
    readonly timestepProjection: {
      readonly phi: number[];
      readonly tRepr: number[];
      readonly tEmb: number[];
    };
    readonly residualStreams: number[][];
    readonly moeTopkByBay: Record<string, number[]>;
    readonly representativeTraces: {
      readonly bay01Mamba: {
        readonly bayId: number;
        readonly shift: number[];
        readonly scale: number[];
        readonly gate: number[];
        readonly U: number[];
        readonly conv_out: number[];
        readonly convState: number[];
        readonly H_prev: number[];
        readonly H: number[];
        readonly y_norm: number[];
        readonly Mamba_out: number[];
        readonly Y: number[];
      };
      readonly bay02Moe: {
        readonly bayId: number;
        readonly shift: number[];
        readonly scale: number[];
        readonly gate: number[];
        readonly U: number[];
        readonly U_mod: number[];
        readonly topk_indices: number[];
        readonly topk_weights: number[];
        readonly y_shared: number[];
        readonly y_routed: number[];
        readonly y_moe: number[];
        readonly Y: number[];
      };
      readonly bay43Gqa: {
        readonly bayId: number;
        readonly shift: number[];
        readonly scale: number[];
        readonly gate: number[];
        readonly U: number[];
        readonly Q: number[];
        readonly K: number[];
        readonly V: number[];
        readonly Scores: number[];
        readonly Softmax: number[];
        readonly V_out: number[];
        readonly Attn_out: number[];
        readonly Y: number[];
      };
    };
    readonly finalNorm: {
      readonly rms: number[];
      readonly X_norm: number[];
      readonly U_final: number[];
    };
    readonly logits: number[];
    readonly probs: number[];
    readonly top5ByPosition: number[][];
  };
}

interface SamplerStepFixture {
  readonly confidenceThreshold?: number;
  readonly temperature?: number;
  readonly uniformVariates?: number[];
  readonly greedyPredictedForComparison?: number[];
  readonly xtBefore: number[];
  readonly logits: number[];
  readonly stepIndex: number;
  readonly stepsPerBlock: number;
  readonly tModel: number;
  readonly predicted: number[];
  readonly confidence: Array<number | null>;
  readonly logXTheta: number[];
  readonly unscaledProbs: number[];
  readonly numAboveThreshold: number;
  readonly minCommitFloor: number;
  readonly tokensCommittedCount: number;
  readonly commitReason: string;
  readonly committedIndices: number[];
  readonly thresholdCommittedIndices: number[];
  readonly floorCommittedIndices: number[];
  readonly remaskedIndices: number[];
  readonly xtNext: number[];
  readonly nMaskedBefore: number;
  readonly nMaskedAfter: number;
  readonly earlyCompleted: boolean;
  readonly isBlockComplete: boolean;
  readonly containsEos: boolean;
}

interface ContextCheckpointFixture {
  readonly prefixLen: number;
  readonly prefixTokenIds: number[];
  readonly arLogits: number[];
  readonly arProbs: number[];
  readonly mambaS1: Record<string, { readonly convState: number[]; readonly ssmState: number[] }>;
  readonly mambaS2Bay01: { readonly convState: number[]; readonly ssmState: number[] };
  readonly gqaKv: Record<string, { readonly K: number[]; readonly V: number[] }>;
}

interface ArMockArVariantFixture {
  readonly arSteps: Array<{
    readonly stepIndex: number;
    readonly tokenId: number;
    readonly logits: number[];
    readonly probs: number[];
    readonly postCommitArLogits: number[];
    readonly postCommitBay01Ssm: number[];
    readonly postCommitBay43K: number[];
    readonly postCommitBay43V: number[];
  }>;
  readonly mockArSteps: Array<{
    readonly stepIndex: number;
    readonly queryTokenId: number;
    readonly committedPrefixLen: number;
    readonly tokenId: number;
    readonly denoiserLogits: number[];
    readonly denoiserProbs: number[];
    readonly residualStreams: number[][];
    readonly postCommitArLogits: number[];
    readonly postCommitBay01Ssm: number[];
    readonly postCommitBay43K: number[];
    readonly postCommitBay43V: number[];
  }>;
}

interface TrajectoryParityFixture {
  readonly metadata: {
    readonly generator: string;
    readonly pythonVersion: string;
    readonly torchVersion: string;
    readonly pinnedTorchVersion: string;
    readonly referenceArchitecture: string;
    readonly tolerances: {
      readonly cacheStateMaxAbs: number;
      readonly residualCheckpointMaxAbs: number;
      readonly lmHeadLogitsMaxAbs: number;
      readonly lmHeadProbsMaxAbs: number;
      readonly samplerLogProbMaxAbs: number;
      readonly samplerProbMaxAbs: number;
    };
  };
  readonly defaultModel: {
    readonly modelId: string;
    readonly twoTowerFingerprint: string;
    readonly weightsSha256: string;
    readonly totalWeightFloats: number;
  };
  readonly identityModel: {
    readonly modelId: string;
    readonly twoTowerFingerprint: string;
    readonly weightsSha256: string;
    readonly totalWeightFloats: number;
  };
  readonly cat6TwoBlockTrajectory: {
    readonly promptTokenIds: number[];
    readonly numBlocks: number;
    readonly blockSize: number;
    readonly stepsPerBlock: number;
    readonly kvCapacity: number;
    readonly temperature: number;
    readonly confidenceThreshold: number;
    readonly eosTokenIds: number[];
    readonly blocks: Array<{
      readonly blockId: number;
      readonly blockSize: number;
      readonly stepsPerBlock: number;
      readonly nfe: number;
      readonly earlyCompleted: boolean;
      readonly finalBlockTokens: number[];
      readonly stoppedByEos: boolean;
      readonly refinements: Array<{
        readonly refinementIndex: number;
        readonly tModel: number;
        readonly xtBefore: number[];
        readonly logits: number[];
        readonly confidence: Array<number | null>;
        readonly predicted: number[];
        readonly committedIndices: number[];
        readonly thresholdCommittedIndices: number[];
        readonly floorCommittedIndices: number[];
        readonly remaskedIndices: number[];
        readonly commitReason: string;
        readonly nMaskedBefore: number;
        readonly nMaskedAfter: number;
        readonly xtNext: number[];
        readonly isBlockComplete: boolean;
        readonly earlyCompleted: boolean;
      }>;
      readonly postCommitContext: ContextCheckpointFixture;
    }>;
    readonly generatedTokenIds: number[];
    readonly fullSequenceTokenIds: number[];
    readonly totalNfe: number;
    readonly stoppedByEos: boolean;
    readonly completedBlocks: number;
  };
  readonly cat7ArAndMockArAlignment: {
    readonly promptTokenIds: number[];
    readonly kvCapacity: number;
    readonly numSteps: number;
    readonly defaultPerturbed: ArMockArVariantFixture;
    readonly identityZeroPerturbation: ArMockArVariantFixture;
  };
  readonly cat8SamplerEdgeCases: {
    readonly vocabSize: number;
    readonly maskTokenId: number;
    readonly eosTokenId: number;
    readonly maskExclusionAndCarryOver: SamplerStepFixture;
    readonly strictThresholdAndBudgetFloor: SamplerStepFixture;
    readonly progressFloorExactEquality: SamplerStepFixture;
    readonly earlyCompletionNoOp: SamplerStepFixture;
    readonly earlyCompletionMidBudget: SamplerStepFixture;
    readonly postBlockEosHalt: {
      readonly confidenceThreshold: number;
      readonly temperature: number;
      readonly step0PartialWithEos: SamplerStepFixture;
      readonly step1CompletedWithEos: SamplerStepFixture;
    };
    readonly deterministicGumbelMax: SamplerStepFixture;
  };
}

const BLOCKS_FX: BlocksParityFixture = JSON.parse(
  readFileSync(resolve(__dirname, 'fixtures/pytorch-twotower-blocks-parity.json'), 'utf-8'),
);
const HANDOFF_FX: HandoffParityFixture = JSON.parse(
  readFileSync(resolve(__dirname, 'fixtures/pytorch-twotower-handoff-parity.json'), 'utf-8'),
);
const TRAJ_FX: TrajectoryParityFixture = JSON.parse(
  readFileSync(resolve(__dirname, 'fixtures/pytorch-twotower-trajectory-parity.json'), 'utf-8'),
);

// ============================================================================
// HELPERS: LIVE WEIGHT SHA-256, KV SLICING, CONFIDENCE & ERROR TRACKING
// ============================================================================

function computeTwoTowerWeightsSha256(model: TwoTowerModelParameters): {
  readonly sha256: string;
  readonly tensorCount: number;
  readonly totalFloats: number;
} {
  const hash = createHash('sha256');
  const tensors = collectTwoTowerCanonicalWeights(model);
  let totalFloats = 0;
  for (const t of tensors) {
    const buf = Buffer.allocUnsafe(t.data.length * 4);
    for (let i = 0; i < t.data.length; i++) {
      const val = t.data[i];
      assertFiniteNumber(val, `weight:${t.name}[${i}]`);
      buf.writeFloatLE(val, i * 4);
    }
    hash.update(buf);
    totalFloats += t.data.length;
  }
  return {
    sha256: hash.digest('hex'),
    tensorCount: tensors.length,
    totalFloats,
  };
}

function padKvRowsToCapacity(
  compact: readonly number[],
  rows: number,
  seqLen: number,
  maxTokens: number,
): Float32Array {
  const out = new Float32Array(rows * maxTokens);
  for (let r = 0; r < rows; r++) {
    for (let k = 0; k < seqLen; k++) {
      out[r * maxTokens + k] = compact[r * seqLen + k];
    }
  }
  return out;
}

function extractCompactKv(cache: GqaLayerKvCache): {
  readonly K: Float32Array;
  readonly V: Float32Array;
} {
  const rows = cache.nKv * cache.dHead;
  const K = new Float32Array(rows * cache.seqLen);
  const V = new Float32Array(rows * cache.seqLen);
  for (let r = 0; r < rows; r++) {
    const srcOff = r * cache.maxTokens;
    const dstOff = r * cache.seqLen;
    for (let k = 0; k < cache.seqLen; k++) {
      K[dstOff + k] = cache.K[srcOff + k];
      V[dstOff + k] = cache.V[srcOff + k];
    }
    for (let k = cache.seqLen; k < cache.maxTokens; k++) {
      expect(cache.K[srcOff + k], `trailing zero K[bay=${cache.bayId},r=${r},k=${k}]`).toBe(0);
      expect(cache.V[srcOff + k], `trailing zero V[bay=${cache.bayId},r=${r},k=${k}]`).toBe(0);
    }
  }
  return { K, V };
}

function assertConfidenceParity(
  actual: ArrayLike<number>,
  expectedNullable: readonly (number | null)[],
  tolerance: number,
  label: string,
): number {
  assertFiniteNumber(tolerance, `${label}.tolerance`);
  if (actual.length !== expectedNullable.length) {
    throw new Error(
      `Length mismatch for ${label}: actual=${actual.length}, expected=${expectedNullable.length}`,
    );
  }
  let maxErr = 0;
  for (let i = 0; i < expectedNullable.length; i++) {
    const a = actual[i];
    const ref = expectedNullable[i];
    if (ref === null) {
      if (a !== Infinity) {
        throw new Error(
          `Expected locked unmasked +Infinity confidence at ${label}[${i}], got ${String(a)}`,
        );
      }
      continue;
    }
    if (!Number.isFinite(a)) {
      throw new Error(`Unexpected non-finite masked confidence at ${label}[${i}]: ${String(a)}`);
    }
    if (!Number.isFinite(ref)) {
      throw new Error(
        `Unexpected non-finite expected confidence at ${label}[${i}]: ${String(ref)}`,
      );
    }
    const err = Math.abs(a - ref);
    if (err > tolerance) {
      throw new Error(
        `Confidence parity exceeded at ${label}[${i}]: |${a} - ${ref}| = ${err} > ${tolerance}`,
      );
    }
    if (err > maxErr) maxErr = err;
  }
  return maxErr;
}

function assertFiniteRelativeOrAbsParity(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  atol: number,
  rtol: number,
  label: string,
): { readonly maxAbs: number; readonly maxRel: number } {
  if (actual.length !== expected.length) {
    throw new Error(`Length mismatch for ${label}: actual=${actual.length}, expected=${expected.length}`);
  }
  let maxAbs = 0;
  let maxRel = 0;
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const b = expected[i];
    if (!Number.isFinite(a)) {
      throw new Error(`Non-finite actual value at ${label}[${i}]: ${String(a)}`);
    }
    if (!Number.isFinite(b)) {
      throw new Error(`Non-finite expected value at ${label}[${i}]: ${String(b)}`);
    }
    const err = Math.abs(a - b);
    const bound = atol + rtol * Math.abs(b);
    if (err > bound) {
      throw new Error(
        `Relative/abs parity exceeded for ${label}[${i}]: |${a} - ${b}| = ${err} > ${bound} (atol=${atol}, rtol=${rtol})`,
      );
    }
    if (err > maxAbs) maxAbs = err;
    const rel = err / Math.max(1.0, Math.abs(b));
    if (rel > maxRel) maxRel = rel;
  }
  return { maxAbs, maxRel };
}

// ============================================================================
// TEST SUITE (ALL 8 MANDATORY CATEGORIES)
// ============================================================================

describe('Independent PyTorch 2.15.0 Reference Parity — Nemotron-Labs-TwoTower (All 8 Categories)', () => {
  const defaultModel = createDefaultTwoTowerModel();
  const identityModel = createTwoTowerModelParameters({ perturbationScale: 0 });
  const observedMaxErrByCategory: Record<string, number> = {};

  function recordErr(category: string, err: number): void {
    assertFiniteNumber(err, `observedErr[${category}]`);
    const prev = observedMaxErrByCategory[category] ?? 0;
    if (err > prev) observedMaxErrByCategory[category] = err;
  }

  afterAll(() => {
    const lines = Object.entries(observedMaxErrByCategory).map(
      ([k, v]) => `  ${k}: maxAbsErr = ${v.toExponential(3)}`,
    );
    console.log(
      `[TwoTower PyTorch 2.15.0 Parity Summary]\n${lines.join('\n')}`,
    );
  });

  it('0. cryptographically verifies weightsSha256 & fingerprints for BOTH default and identity models', () => {
    const defDigest = computeTwoTowerWeightsSha256(defaultModel);
    const idDigest = computeTwoTowerWeightsSha256(identityModel);

    expect(defDigest.tensorCount).toBe(860);
    expect(idDigest.tensorCount).toBe(860);

    // Pinned PyTorch 2.15.0 provenance across all 3 fixtures
    expect(BLOCKS_FX.metadata.pinnedTorchVersion).toBe('2.15.0');
    expect(HANDOFF_FX.metadata.pinnedTorchVersion).toBe('2.15.0');
    expect(TRAJ_FX.metadata.pinnedTorchVersion).toBe('2.15.0');
    expect(BLOCKS_FX.metadata.torchVersion.startsWith('2.15.0')).toBe(true);
    expect(HANDOFF_FX.metadata.torchVersion.startsWith('2.15.0')).toBe(true);
    expect(TRAJ_FX.metadata.torchVersion.startsWith('2.15.0')).toBe(true);

    // Default model SHA-256 & fingerprints across all 3 fixtures
    for (const fx of [BLOCKS_FX, HANDOFF_FX]) {
      expect(defaultModel.modelId).toBe(fx.modelId);
      expect(defaultModel.twoTowerFingerprint).toBe(fx.twoTowerFingerprint);
      expect(defaultModel.contextModel.parameterFingerprint).toBe(fx.contextFingerprint);
      expect(defaultModel.denoiserModel.parameterFingerprint).toBe(fx.denoiserFingerprint);
      expect(defaultModel.timeConditioning.fingerprint).toBe(fx.timeConditioningFingerprint);
      expect(defDigest.totalFloats).toBe(fx.totalWeightFloats);
      expect(defDigest.sha256).toBe(fx.weightsSha256);
    }

    expect(defaultModel.modelId).toBe(TRAJ_FX.defaultModel.modelId);
    expect(defaultModel.twoTowerFingerprint).toBe(TRAJ_FX.defaultModel.twoTowerFingerprint);
    expect(defDigest.totalFloats).toBe(TRAJ_FX.defaultModel.totalWeightFloats);
    expect(defDigest.sha256).toBe(TRAJ_FX.defaultModel.weightsSha256);

    // Identity-weight model (perturbationScale = 0) SHA-256 & fingerprints
    expect(identityModel.modelId).toBe(TRAJ_FX.identityModel.modelId);
    expect(identityModel.twoTowerFingerprint).toBe(TRAJ_FX.identityModel.twoTowerFingerprint);
    expect(idDigest.totalFloats).toBe(TRAJ_FX.identityModel.totalWeightFloats);
    expect(idDigest.sha256).toBe(TRAJ_FX.identityModel.weightsSha256);
    expect(defDigest.sha256).not.toBe(idDigest.sha256);
  });

  it('Cat 1: matches PyTorch 2.15.0 timestep embedding, adaLN-single modulation, and order-inversion sensitivity (> 1e-2)', () => {
    const fx = BLOCKS_FX.cat1TimestepAndModulation;
    const tolMod = BLOCKS_FX.metadata.tolerances.modulationMaxAbs;

    for (const rec of fx.timesteps) {
      const phiDirect = computeSinusoidalTimestepEmbedding(rec.t, defaultModel.config.freqDim);
      const proj = computeGlobalTimestepProjection(rec.t, defaultModel.timeConditioning);
      assertBitExactFiniteArray(phiDirect, proj.phi, `cat1.phiDirectVsProj(t=${rec.t})`);

      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(proj.phi, rec.phi, tolMod, `cat1.t${rec.t}.phi`),
      );
      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(proj.tRepr, rec.tRepr, tolMod, `cat1.t${rec.t}.tRepr`),
      );
      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(proj.tEmb, rec.tEmb, tolMod, `cat1.t${rec.t}.tEmb`),
      );

      for (const layerRef of rec.layers) {
        const mod = computeLayerModulationParams(
          proj.tEmb,
          defaultModel.timeConditioning,
          layerRef.layerIndex,
        );
        recordErr(
          'Cat 1 (Timestep & Modulation)',
          assertFiniteArrayParity(mod.shift, layerRef.shift, tolMod, `cat1.t${rec.t}.l${layerRef.layerIndex}.shift`),
        );
        recordErr(
          'Cat 1 (Timestep & Modulation)',
          assertFiniteArrayParity(mod.scale, layerRef.scale, tolMod, `cat1.t${rec.t}.l${layerRef.layerIndex}.scale`),
        );
        recordErr(
          'Cat 1 (Timestep & Modulation)',
          assertFiniteArrayParity(mod.gate, layerRef.gate, tolMod, `cat1.t${rec.t}.l${layerRef.layerIndex}.gate`),
        );
      }
    }

    // Verify asymmetric modulation-vs-norm order sensitivity (> 1e-2!) on M, *, and E bays
    const sens = fx.orderInversionSensitivity;
    const X = new Float32Array(sens.X);
    const eps = defaultModel.denoiserModel.config.rmsNormEps;
    const proj = computeGlobalTimestepProjection(sens.t, defaultModel.timeConditioning);

    // Bay 01 Mamba (M): modulate -> RMSNorm
    expect(modulationOrderForBayKind('M')).toBe('modulate_then_norm');
    const mod0 = computeLayerModulationParams(proj.tEmb, defaultModel.timeConditioning, 0);
    const w1 = defaultModel.denoiserModel.getBayWeights(1);
    expect(w1.kind).toBe('M');
    if (w1.kind === 'M') {
      const mSim = runMambaSim(sens.tokens, {
        X,
        gamma_in: w1.gamma_in,
        W_in: w1.W_in,
        w_conv: w1.w_conv,
        b_conv: w1.b_conv,
        A_log: w1.A_log,
        dt_bias: w1.dt_bias,
        D: w1.D,
        gamma_gate: w1.gamma_gate,
        W_out: w1.W_out,
        eps,
        preNormMod: { shift: mod0.shift, scale: mod0.scale },
        residualGate: mod0.gate,
      });
      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(mSim.U, sens.bay01Mamba.U_correct, BLOCKS_FX.metadata.tolerances.mamba2MaxAbs, 'cat1.sens.mamba.U'),
      );
      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(mSim.Y, sens.bay01Mamba.Y_correct, BLOCKS_FX.metadata.tolerances.mamba2MaxAbs, 'cat1.sens.mamba.Y'),
      );
      const uInv = applyAdaLnModulate(
        applyRmsNorm(X, w1.gamma_in, 8, sens.tokens, eps).U,
        8,
        sens.tokens,
        mod0.shift,
        mod0.scale,
      );
      expect(computeStrictMaxAbsDiff(mSim.U, uInv, 'cat1.sens.mamba.U_vs_inv')).toBeGreaterThan(1e-2);
      expect(computeStrictMaxAbsDiff(mSim.Y, sens.bay01Mamba.Y_inverted, 'cat1.sens.mamba.Y_vs_inv')).toBeGreaterThan(1e-2);
    }

    // Bay 06 GQA (*): modulate -> RMSNorm
    expect(modulationOrderForBayKind('*')).toBe('modulate_then_norm');
    const mod5 = computeLayerModulationParams(proj.tEmb, defaultModel.timeConditioning, 5);
    const w6 = defaultModel.denoiserModel.getBayWeights(6);
    expect(w6.kind).toBe('*');
    if (w6.kind === '*') {
      const gSim = runGqaSim(sens.tokens, {
        X,
        gamma: w6.gamma,
        W_Q: w6.W_Q,
        W_K: w6.W_K,
        W_V: w6.W_V,
        W_O: w6.W_O,
        rmsEps: eps,
        attentionMaskMode: 'bidirectional_block',
        preNormMod: { shift: mod5.shift, scale: mod5.scale },
        residualGate: mod5.gate,
      });
      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(gSim.U, sens.bay06Gqa.U_correct, BLOCKS_FX.metadata.tolerances.gqaMaxAbs, 'cat1.sens.gqa.U'),
      );
      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(gSim.Y, sens.bay06Gqa.Y_correct, BLOCKS_FX.metadata.tolerances.gqaMaxAbs, 'cat1.sens.gqa.Y'),
      );
      expect(computeStrictMaxAbsDiff(gSim.U, sens.bay06Gqa.U_inverted, 'cat1.sens.gqa.U_vs_inv')).toBeGreaterThan(1e-2);
      expect(computeStrictMaxAbsDiff(gSim.Y, sens.bay06Gqa.Y_inverted, 'cat1.sens.gqa.Y_vs_inv')).toBeGreaterThan(1e-2);
    }

    // Bay 02 MoE (E): RMSNorm -> modulate
    expect(modulationOrderForBayKind('E')).toBe('norm_then_modulate');
    const mod1 = computeLayerModulationParams(proj.tEmb, defaultModel.timeConditioning, 1);
    const w2 = defaultModel.denoiserModel.getBayWeights(2);
    expect(w2.kind).toBe('E');
    if (w2.kind === 'E') {
      const eSim = runMoeSim(sens.tokens, {
        X,
        gamma_in: w2.gamma_in,
        W_router: w2.W_router,
        e_score_correction_bias: w2.e_score_correction_bias,
        W_up: w2.W_up,
        W_down: w2.W_down,
        W_up_s: w2.W_up_s,
        W_down_s: w2.W_down_s,
        rmsNormEps: eps,
        routerPrecisionPolicy: 'float64_reference',
        postNormMod: { shift: mod1.shift, scale: mod1.scale },
        residualGate: mod1.gate,
      });
      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(eSim.U_mod!, sens.bay02Moe.U_correct, BLOCKS_FX.metadata.tolerances.moeMaxAbs, 'cat1.sens.moe.U_mod'),
      );
      recordErr(
        'Cat 1 (Timestep & Modulation)',
        assertFiniteArrayParity(eSim.Y, sens.bay02Moe.Y_correct, BLOCKS_FX.metadata.tolerances.moeMaxAbs, 'cat1.sens.moe.Y'),
      );
      expect(computeStrictMaxAbsDiff(eSim.U_mod!, sens.bay02Moe.U_inverted, 'cat1.sens.moe.U_vs_inv')).toBeGreaterThan(1e-2);
      expect(computeStrictMaxAbsDiff(eSim.Y, sens.bay02Moe.Y_inverted, 'cat1.sens.moe.Y_vs_inv')).toBeGreaterThan(1e-2);
    }
  });

  it('Cat 2: matches PyTorch 2.15.0 bidirectional rectangular [4x12] GQA (zero -Inf) and future-token influence vs causal invariance', () => {
    const fx = BLOCKS_FX.cat2BidirectionalGqa;
    const tol = BLOCKS_FX.metadata.tolerances.gqaMaxAbs;
    const w43 = defaultModel.denoiserModel.getBayWeights(fx.bayId);
    expect(w43.kind).toBe('*');
    if (w43.kind !== '*') return;

    const kPadded = padKvRowsToCapacity(fx.KPrefix, 8, fx.prefixLen, fx.kvCapacity);
    const vPadded = padKvRowsToCapacity(fx.VPrefix, 8, fx.prefixLen, fx.kvCapacity);
    const preNormMod = {
      shift: new Float32Array(fx.shift),
      scale: new Float32Array(fx.scale),
    };
    const residualGate = new Float32Array(fx.gate);

    const commonFixture = {
      gamma: w43.gamma,
      W_Q: w43.W_Q,
      W_K: w43.W_K,
      W_V: w43.W_V,
      W_O: w43.W_O,
      rmsEps: fx.eps,
      kvCacheIn: {
        seqLen: fx.prefixLen,
        maxTokens: fx.kvCapacity,
        K: kPadded,
        V: vPadded,
      },
      preNormMod,
      residualGate,
    };

    const bidirA = runGqaSim(fx.blockSize, {
      ...commonFixture,
      X: new Float32Array(fx.X),
      attentionMaskMode: 'bidirectional_block',
    });

    // Every score in bidirectional mode MUST be strictly finite (ZERO -Infinity!)
    for (let i = 0; i < bidirA.Scores.length; i++) {
      expect(Number.isFinite(bidirA.Scores[i]), `bidir score[${i}] finite`).toBe(true);
    }

    const bRef = fx.bidirectional;
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.X_mod!, bRef.X_mod, tol, 'cat2.X_mod'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.U, bRef.U, tol, 'cat2.U'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.Q, bRef.Q, tol, 'cat2.Q'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.K, bRef.K, tol, 'cat2.K'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.V, bRef.V, tol, 'cat2.V'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.Scores, bRef.Scores, tol, 'cat2.Scores'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.Softmax, bRef.Softmax, tol, 'cat2.Softmax'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.V_out, bRef.V_out, tol, 'cat2.V_out'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.Attn_out, bRef.Attn_out, tol, 'cat2.Attn_out'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirA.Y, bRef.Y, tol, 'cat2.Y'));

    // Nondegenerate future-token influence: perturbing ONLY slot j=3 changes slot i=0 in bidirectional mode
    // while leaving slot i=0 BIT-EXACT UNCHANGED in causal mode!
    const inf = fx.futureTokenInfluence;
    const bidirB = runGqaSim(fx.blockSize, {
      ...commonFixture,
      X: new Float32Array(inf.X_b),
      attentionMaskMode: 'bidirectional_block',
    });
    const causalA = runGqaSim(fx.blockSize, {
      ...commonFixture,
      X: new Float32Array(inf.X_a),
      attentionMaskMode: 'causal',
    });
    const causalB = runGqaSim(fx.blockSize, {
      ...commonFixture,
      X: new Float32Array(inf.X_b),
      attentionMaskMode: 'causal',
    });

    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(bidirB.Y, inf.Y_bidir_b, tol, 'cat2.inf.Y_bidir_b'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(causalA.Y, inf.Y_causal_a, tol, 'cat2.inf.Y_causal_a'));
    recordErr('Cat 2 (Bidirectional GQA)', assertFiniteArrayParity(causalB.Y, inf.Y_causal_b, tol, 'cat2.inf.Y_causal_b'));
    recordErr(
      'Cat 2 (Bidirectional GQA)',
      assertCausalScoresParity(causalA.Scores, inf.causalScoresA, tol, 'cat2.inf.causalScoresA'),
    );

    const extractSlot0 = (Y: Float32Array) => {
      const col = new Float32Array(8);
      for (let i = 0; i < 8; i++) col[i] = Y[i * fx.blockSize + 0];
      return col;
    };
    const tsBidirTok0Diff = computeStrictMaxAbsDiff(
      extractSlot0(bidirA.Y),
      extractSlot0(bidirB.Y),
      'cat2.inf.bidirTok0',
    );
    expect(tsBidirTok0Diff).toBeGreaterThan(1e-3);
    expect(Math.abs(tsBidirTok0Diff - inf.bidirToken0MaxDiff)).toBeLessThan(tol);

    // Causal mode: earlier token i=0 MUST be bit-exact unchanged when future token j=3 changes!
    assertBitExactFiniteArray(extractSlot0(causalA.Y), extractSlot0(causalB.Y), 'cat2.inf.causalTok0');
  });

  it('Cat 3: matches PyTorch 2.15.0 seeded modulated Mamba-2, prefix-stitching parity, and O(1e3) FP32 SSM stability', () => {
    const fx = BLOCKS_FX.cat3SeededMamba2;
    const tol = BLOCKS_FX.metadata.tolerances.mamba2MaxAbs;
    const w1 = defaultModel.denoiserModel.getBayWeights(fx.bayId);
    expect(w1.kind).toBe('M');
    if (w1.kind !== 'M') return;

    const baseWeights = {
      gamma_in: w1.gamma_in,
      W_in: w1.W_in,
      w_conv: w1.w_conv,
      b_conv: w1.b_conv,
      A_log: w1.A_log,
      dt_bias: w1.dt_bias,
      D: w1.D,
      gamma_gate: w1.gamma_gate,
      W_out: w1.W_out,
      eps: fx.eps,
    };
    const preNormMod = {
      shift: new Float32Array(fx.shift),
      scale: new Float32Array(fx.scale),
    };
    const residualGate = new Float32Array(fx.gate);

    // 3A: Seeded & modulated Bay 01 Mamba-2
    const sRef = fx.modulatedSeeded;
    const mSeeded = runMambaSim(sRef.tokens, {
      ...baseWeights,
      X: new Float32Array(sRef.X),
      initialConvState: new Float32Array(sRef.convStateIn),
      initialSsmState: new Float32Array(sRef.ssmStateIn),
      preNormMod,
      residualGate,
    });
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.X_mod!, sRef.X_mod, tol, 'cat3.seed.X_mod'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.U, sRef.U, tol, 'cat3.seed.U'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.conv_out, sRef.conv_out, tol, 'cat3.seed.conv_out'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.conv_buf, sRef.convStateOut, tol, 'cat3.seed.conv_buf'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.dt, sRef.dt, tol, 'cat3.seed.dt'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.dA, sRef.dA, tol, 'cat3.seed.dA'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.H_prev, sRef.H_prev, tol, 'cat3.seed.H_prev'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.H, sRef.H, tol, 'cat3.seed.H'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.y, sRef.y, tol, 'cat3.seed.y'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.y_norm, sRef.y_norm, tol, 'cat3.seed.y_norm'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.Mamba_out, sRef.Mamba_out, tol, 'cat3.seed.Mamba_out'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mSeeded.Y, sRef.Y, tol, 'cat3.seed.Y'));

    // 3B: Prefix-stitching parity ([0..11] 1-pass vs [0..7] then seeded [8..11])
    const stRef = fx.prefixStitching;
    const totalLen = stRef.prefixLen + stRef.blockSize;
    const X_full = new Float32Array(stRef.X_full);
    const X_pref = new Float32Array(8 * stRef.prefixLen);
    const X_tail = new Float32Array(8 * stRef.blockSize);
    for (let i = 0; i < 8; i++) {
      for (let t = 0; t < stRef.prefixLen; t++) {
        X_pref[i * stRef.prefixLen + t] = X_full[i * totalLen + t];
      }
      for (let t = 0; t < stRef.blockSize; t++) {
        X_tail[i * stRef.blockSize + t] = X_full[i * totalLen + stRef.prefixLen + t];
      }
    }
    const mFull = runMambaSim(totalLen, { ...baseWeights, X: X_full });
    const mPref = runMambaSim(stRef.prefixLen, { ...baseWeights, X: X_pref });
    const mTail = runMambaSim(stRef.blockSize, {
      ...baseWeights,
      X: X_tail,
      initialConvState: mPref.conv_buf,
      initialSsmState: mPref.H,
    });

    const fullTailY = new Float32Array(8 * stRef.blockSize);
    for (let i = 0; i < 8; i++) {
      for (let t = 0; t < stRef.blockSize; t++) {
        fullTailY[i * stRef.blockSize + t] = mFull.Y[i * totalLen + stRef.prefixLen + t];
      }
    }
    // TypeScript 1-pass vs stitched 2-pass is bit-exact!
    assertBitExactFiniteArray(fullTailY, mTail.Y, 'cat3.stitch.tsFullVsStitched.Y');
    assertBitExactFiniteArray(mFull.conv_buf, mTail.conv_buf, 'cat3.stitch.tsFullVsStitched.conv');
    assertBitExactFiniteArray(mFull.H, mTail.H, 'cat3.stitch.tsFullVsStitched.H');

    // Both match PyTorch 2.15.0 reference!
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mTail.Y, stRef.Y_stitched, tol, 'cat3.stitch.Y_vs_pt'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mTail.conv_buf, stRef.convStateStitched, tol, 'cat3.stitch.conv_vs_pt'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mTail.H, stRef.HStitched, tol, 'cat3.stitch.H_vs_pt'));

    // 3C: High-magnitude O(1e3) FP32 SSM stability (zero NaN/Inf!)
    const lgRef = fx.highMagnitudeStability;
    const mLarge = runMambaSim(lgRef.tokens, {
      ...baseWeights,
      X: new Float32Array(lgRef.X),
      initialConvState: new Float32Array(lgRef.convStateIn),
      initialSsmState: new Float32Array(lgRef.ssmStateIn),
      preNormMod,
      residualGate,
    });
    const relTol = BLOCKS_FX.metadata.tolerances.mamba2HighMagRel;
    assertFiniteRelativeOrAbsParity(mLarge.conv_out, lgRef.conv_out, tol, relTol, 'cat3.large.conv_out');
    assertFiniteArrayParity(mLarge.dt, lgRef.dt, tol, 'cat3.large.dt');
    assertFiniteArrayParity(mLarge.dA, lgRef.dA, tol, 'cat3.large.dA');
    assertFiniteRelativeOrAbsParity(mLarge.H, lgRef.H, tol, relTol, 'cat3.large.H');
    assertFiniteRelativeOrAbsParity(mLarge.y, lgRef.y, tol, relTol, 'cat3.large.y');
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mLarge.y_norm, lgRef.y_norm, tol, 'cat3.large.y_norm'));
    recordErr('Cat 3 (Seeded Mamba-2)', assertFiniteArrayParity(mLarge.Mamba_out, lgRef.Mamba_out, tol, 'cat3.large.Mamba_out'));
    assertFiniteRelativeOrAbsParity(mLarge.Y, lgRef.Y, tol, relTol, 'cat3.large.Y');
  });

  it('Cat 4: matches PyTorch 2.15.0 FP64 MoE routing, gated residuals, near-tie FP64 vs FP32 flip, and exact-tie lower-index break', () => {
    const fx = BLOCKS_FX.cat4Fp64MoeRouting;
    const tol = BLOCKS_FX.metadata.tolerances.moeMaxAbs;
    const w2 = defaultModel.denoiserModel.getBayWeights(fx.bayId);
    expect(w2.kind).toBe('E');
    if (w2.kind !== 'E') return;

    // 4A: Bay 02 modulated FP64 MoE
    const mRef = fx.modulatedFp64;
    const moeMod = runMoeSim(mRef.tokens, {
      X: new Float32Array(mRef.X),
      gamma_in: w2.gamma_in,
      W_router: w2.W_router,
      e_score_correction_bias: w2.e_score_correction_bias,
      W_up: w2.W_up,
      W_down: w2.W_down,
      W_up_s: w2.W_up_s,
      W_down_s: w2.W_down_s,
      rmsNormEps: fx.eps,
      routerPrecisionPolicy: 'float64_reference',
      postNormMod: {
        shift: new Float32Array(fx.shift),
        scale: new Float32Array(fx.scale),
      },
      residualGate: new Float32Array(fx.gate),
    });

    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.U, mRef.U, tol, 'cat4.mod.U'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.U_mod!, mRef.U_mod, tol, 'cat4.mod.U_mod'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.scores, mRef.scores, tol, 'cat4.mod.scores'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.scores_for_choice, mRef.scores_for_choice, tol, 'cat4.mod.choice'));
    expect(Array.from(moeMod.topk_indices)).toEqual(mRef.topk_indices);
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.topk_raw_scores, mRef.topk_raw_scores, tol, 'cat4.mod.topk_raw'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.topk_weights, mRef.topk_weights, tol, 'cat4.mod.topk_w'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.y_shared, mRef.y_shared, tol, 'cat4.mod.y_shared'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.y_routed, mRef.y_routed, tol, 'cat4.mod.y_routed'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.y_moe, mRef.y_moe, tol, 'cat4.mod.y_moe'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(moeMod.Y, mRef.Y, tol, 'cat4.mod.Y'));

    // 4B: Near-tie router (~2e-8 Float64 margin): float64_reference matches PyTorch FP64, while float32_nano flips!
    const near = fx.nearTieRouter;
    const nearBase = {
      X: new Float32Array(near.X),
      gamma_in: new Float32Array(near.gamma_in),
      W_router: new Float32Array(near.W_router),
      e_score_correction_bias: new Float32Array(near.e_score_correction_bias),
      W_up: w2.W_up,
      W_down: w2.W_down,
      W_up_s: w2.W_up_s,
      W_down_s: w2.W_down_s,
      rmsNormEps: fx.eps,
    };
    const nearF64 = runMoeSim(near.tokens, { ...nearBase, routerPrecisionPolicy: 'float64_reference' });
    const nearF32 = runMoeSim(near.tokens, { ...nearBase, routerPrecisionPolicy: 'float32_nano' });

    expect(Array.from(nearF64.topk_indices)).toEqual(near.float64TopkIndices);
    expect(Array.from(nearF32.topk_indices)).toEqual(near.float32TopkIndices);
    expect(Array.from(nearF64.topk_indices)).not.toEqual(Array.from(nearF32.topk_indices));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(nearF64.topk_weights, near.float64TopkWeights, tol, 'cat4.near.f64.w'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(nearF64.Y, near.float64Y, tol, 'cat4.near.f64.Y'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(nearF32.topk_weights, near.float32TopkWeights, tol, 'cat4.near.f32.w'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(nearF32.Y, near.float32Y, tol, 'cat4.near.f32.Y'));

    // 4C: Exact-tie router (deterministic lower-index tie-break)
    const exact = fx.exactTieRouter;
    const exactF64 = runMoeSim(exact.tokens, {
      X: new Float32Array(exact.X),
      gamma_in: new Float32Array(exact.gamma_in),
      W_router: new Float32Array(exact.W_router),
      e_score_correction_bias: new Float32Array(exact.e_score_correction_bias),
      W_up: w2.W_up,
      W_down: w2.W_down,
      W_up_s: w2.W_up_s,
      W_down_s: w2.W_down_s,
      rmsNormEps: fx.eps,
      routerPrecisionPolicy: 'float64_reference',
    });
    expect(Array.from(exactF64.topk_indices)).toEqual(exact.expectedTopkIndices);
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(exactF64.topk_weights, exact.expectedTopkWeights, tol, 'cat4.exact.w'));
    recordErr('Cat 4 (FP64 MoE Routing)', assertFiniteArrayParity(exactF64.Y, exact.expectedY, tol, 'cat4.exact.Y'));
  });

  it('Cat 5: matches PyTorch 2.15.0 complete 52-layer Context-to-Denoiser handoff with 0-ULP Context immutability', () => {
    const tol = HANDOFF_FX.metadata.tolerances;

    // 1. Context prefill on canonical 8-token prompt [0..7]
    const ctxSnap = runTwoTowerContextPrefill(defaultModel, HANDOFF_FX.promptTokenIds, {
      kvCapacity: HANDOFF_FX.kvCapacity,
    });
    expect(ctxSnap.ctxState.processedTokens).toBe(HANDOFF_FX.contextPrefill.prefixLen);

    // Clone pre-handoff Context tensors for strict 0-ULP immutability verification
    const preSnapshotDigest = ctxSnap.snapshotDigest;
    const preCtxStateDigest = ctxSnap.ctxState.stateDigest;
    const preMambaS1Copies = new Map<number, { conv: Float32Array; ssm: Float32Array }>();
    for (const [b, m] of ctxSnap.ctxState.mambaStates.entries()) {
      preMambaS1Copies.set(b, { conv: new Float32Array(m.convState), ssm: new Float32Array(m.ssmState) });
    }
    const preMambaS2Copies = new Map<number, { conv: Float32Array; ssm: Float32Array }>();
    for (const [b, m] of ctxSnap.mambaS2.entries()) {
      preMambaS2Copies.set(b, { conv: new Float32Array(m.convState), ssm: new Float32Array(m.ssmState) });
    }
    const preGqaCopies = new Map<number, { K: Float32Array; V: Float32Array }>();
    for (const [b, g] of ctxSnap.ctxState.gqaCaches.entries()) {
      preGqaCopies.set(b, { K: new Float32Array(g.K), V: new Float32Array(g.V) });
    }

    // Verify Context prefill states & AR logits/probs against PyTorch 2.15.0
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(ctxSnap.arLogits, HANDOFF_FX.contextPrefill.arLogits, tol.lmHeadLogitsMaxAbs, 'cat5.ctx.arLogits'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(ctxSnap.arProbs, HANDOFF_FX.contextPrefill.arProbs, tol.lmHeadProbsMaxAbs, 'cat5.ctx.arProbs'));
    for (const [bayKey, refM] of Object.entries(HANDOFF_FX.contextPrefill.mambaS1)) {
      const m = ctxSnap.ctxState.mambaStates.get(Number(bayKey))!;
      recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(m.convState, refM.convState, tol.cacheStateMaxAbs, `cat5.ctx.m${bayKey}.conv`));
      recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(m.ssmState, refM.ssmState, tol.cacheStateMaxAbs, `cat5.ctx.m${bayKey}.ssm`));
    }
    for (const [bayKey, refM] of Object.entries(HANDOFF_FX.contextPrefill.mambaS2)) {
      const m = ctxSnap.mambaS2.get(Number(bayKey))!;
      recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(m.convState, refM.convState, tol.cacheStateMaxAbs, `cat5.s2.m${bayKey}.conv`));
      recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(m.ssmState, refM.ssmState, tol.cacheStateMaxAbs, `cat5.s2.m${bayKey}.ssm`));
    }
    for (const [bayKey, refG] of Object.entries(HANDOFF_FX.contextPrefill.gqaKv)) {
      const g = extractCompactKv(ctxSnap.ctxState.gqaCaches.get(Number(bayKey))!);
      recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(g.K, refG.K, tol.cacheStateMaxAbs, `cat5.ctx.g${bayKey}.K`));
      recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(g.V, refG.V, tol.cacheStateMaxAbs, `cat5.ctx.g${bayKey}.V`));
    }

    // 2. Build read-only conditioning & execute complete 52-layer Denoiser pass
    const cond = buildContextConditioningState(defaultModel, ctxSnap, {
      mode: 'mask_diffusion',
      blockId: HANDOFF_FX.blockId,
      refinementIndex: HANDOFF_FX.refinementIndex,
    });
    const denPass = runTwoTowerDenoiserPass(defaultModel, cond, HANDOFF_FX.blockTokenIds, {
      retainedTraceBays: [1, 2, 43],
    });
    expect(denPass.tModel).toBe(HANDOFF_FX.tModel);

    const denRef = HANDOFF_FX.denoiserPass;
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(denPass.timestepProjection!.phi, denRef.timestepProjection.phi, tol.modulationMaxAbs, 'cat5.den.phi'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(denPass.timestepProjection!.tRepr, denRef.timestepProjection.tRepr, tol.modulationMaxAbs, 'cat5.den.tRepr'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(denPass.timestepProjection!.tEmb, denRef.timestepProjection.tEmb, tol.modulationMaxAbs, 'cat5.den.tEmb'));

    // All 53 residual checkpoints (X_0 bit-exact embedding, X_1..X_52 <= 2e-5)
    expect(denPass.residualStreams).toHaveLength(53);
    assertBitExactFiniteArray(denPass.residualStreams[0], denRef.residualStreams[0], 'cat5.den.X_embed');
    for (let k = 1; k <= 52; k++) {
      recordErr(
        'Cat 5 (52L Handoff)',
        assertFiniteArrayParity(
          denPass.residualStreams[k],
          denRef.residualStreams[k],
          tol.residualCheckpointMaxAbs,
          `cat5.den.residual[${k}]`,
        ),
      );
    }

    // All 23 MoE layers Top-6 routed expert indices match PyTorch FP64 exactly
    for (const [bayKey, refTopk] of Object.entries(denRef.moeTopkByBay)) {
      const moeTrace = denPass.getMoeTrace(Number(bayKey));
      expect(Array.from(moeTrace.topk_indices), `cat5.moeTopk[bay=${bayKey}]`).toEqual(refTopk);
    }

    // Representative traces: Bay 01 Mamba, Bay 02 MoE, Bay 43 Bidirectional GQA
    const trM = denPass.getMambaTrace(1);
    const refM1 = denRef.representativeTraces.bay01Mamba;
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trM.U, refM1.U, tol.traceMaxAbs, 'cat5.trM1.U'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trM.conv_out, refM1.conv_out, tol.traceMaxAbs, 'cat5.trM1.conv_out'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trM.conv_buf, refM1.convState, tol.traceMaxAbs, 'cat5.trM1.convState'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trM.H_prev, refM1.H_prev, tol.traceMaxAbs, 'cat5.trM1.H_prev'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trM.H, refM1.H, tol.traceMaxAbs, 'cat5.trM1.H'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trM.y_norm, refM1.y_norm, tol.traceMaxAbs, 'cat5.trM1.y_norm'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trM.Mamba_out, refM1.Mamba_out, tol.traceMaxAbs, 'cat5.trM1.Mamba_out'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trM.Y, refM1.Y, tol.traceMaxAbs, 'cat5.trM1.Y'));

    const trE = denPass.getMoeTrace(2);
    const refE2 = denRef.representativeTraces.bay02Moe;
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trE.U, refE2.U, tol.traceMaxAbs, 'cat5.trE2.U'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trE.U_mod!, refE2.U_mod, tol.traceMaxAbs, 'cat5.trE2.U_mod'));
    expect(Array.from(trE.topk_indices)).toEqual(refE2.topk_indices);
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trE.topk_weights, refE2.topk_weights, tol.traceMaxAbs, 'cat5.trE2.topk_w'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trE.y_shared, refE2.y_shared, tol.traceMaxAbs, 'cat5.trE2.y_shared'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trE.y_routed, refE2.y_routed, tol.traceMaxAbs, 'cat5.trE2.y_routed'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trE.y_moe, refE2.y_moe, tol.traceMaxAbs, 'cat5.trE2.y_moe'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trE.Y, refE2.Y, tol.traceMaxAbs, 'cat5.trE2.Y'));

    const trG = denPass.getGqaTrace(43);
    const refG43 = denRef.representativeTraces.bay43Gqa;
    expect(trG.attentionMaskMode).toBe('bidirectional_block');
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.U, refG43.U, tol.traceMaxAbs, 'cat5.trG43.U'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.Q, refG43.Q, tol.traceMaxAbs, 'cat5.trG43.Q'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.K, refG43.K, tol.traceMaxAbs, 'cat5.trG43.K'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.V, refG43.V, tol.traceMaxAbs, 'cat5.trG43.V'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.Scores, refG43.Scores, tol.traceMaxAbs, 'cat5.trG43.Scores'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.Softmax, refG43.Softmax, tol.traceMaxAbs, 'cat5.trG43.Softmax'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.V_out, refG43.V_out, tol.traceMaxAbs, 'cat5.trG43.V_out'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.Attn_out, refG43.Attn_out, tol.traceMaxAbs, 'cat5.trG43.Attn_out'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(trG.Y, refG43.Y, tol.traceMaxAbs, 'cat5.trG43.Y'));

    // Unmodulated final RMSNorm & unshifted [4, 34] logits & probs
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(denPass.finalNorm.rms, denRef.finalNorm.rms, tol.finalNormMaxAbs, 'cat5.fn.rms'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(denPass.finalNorm.X_norm, denRef.finalNorm.X_norm, tol.finalNormMaxAbs, 'cat5.fn.X_norm'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(denPass.finalNorm.U_final, denRef.finalNorm.U_final, tol.finalNormMaxAbs, 'cat5.fn.U_final'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(denPass.logits, denRef.logits, tol.lmHeadLogitsMaxAbs, 'cat5.den.logits'));
    recordErr('Cat 5 (52L Handoff)', assertFiniteArrayParity(denPass.probs, denRef.probs, tol.lmHeadProbsMaxAbs, 'cat5.den.probs'));
    for (let j = 0; j < HANDOFF_FX.blockTokenIds.length; j++) {
      const actualTop5 = denPass.topKByPosition[j].map((p) => p.tokenId);
      expect(actualTop5, `cat5.top5[j=${j}]`).toEqual(denRef.top5ByPosition[j]);
    }

    // Strict 0-ULP Context cache immutability check after Denoiser handoff!
    expect(ctxSnap.snapshotDigest).toBe(preSnapshotDigest);
    expect(ctxSnap.ctxState.stateDigest).toBe(preCtxStateDigest);
    for (const [b, prev] of preMambaS1Copies.entries()) {
      const cur = ctxSnap.ctxState.mambaStates.get(b)!;
      assertBitExactFiniteArray(cur.convState, prev.conv, `cat5.immut.mS1[${b}].conv`);
      assertBitExactFiniteArray(cur.ssmState, prev.ssm, `cat5.immut.mS1[${b}].ssm`);
    }
    for (const [b, prev] of preMambaS2Copies.entries()) {
      const cur = ctxSnap.mambaS2.get(b)!;
      assertBitExactFiniteArray(cur.convState, prev.conv, `cat5.immut.mS2[${b}].conv`);
      assertBitExactFiniteArray(cur.ssmState, prev.ssm, `cat5.immut.mS2[${b}].ssm`);
    }
    for (const [b, prev] of preGqaCopies.entries()) {
      const cur = ctxSnap.ctxState.gqaCaches.get(b)!;
      assertBitExactFiniteArray(cur.K, prev.K, `cat5.immut.gqa[${b}].K`);
      assertBitExactFiniteArray(cur.V, prev.V, `cat5.immut.gqa[${b}].V`);
    }
  });

  it('Cat 6: matches PyTorch 2.15.0 full two-block diffusion trajectory (P=8 -> 12 -> 16) with actual Context commits', () => {
    const fx = TRAJ_FX.cat6TwoBlockTrajectory;
    const tol = TRAJ_FX.metadata.tolerances;

    const genRes = generateTwoTowerDiffusion(defaultModel, fx.promptTokenIds, {
      numBlocks: fx.numBlocks,
      blockSize: fx.blockSize,
      stepsPerBlock: fx.stepsPerBlock,
      kvCapacity: fx.kvCapacity,
      confidenceThreshold: fx.confidenceThreshold,
      temperature: fx.temperature,
      eosTokenIds: fx.eosTokenIds,
    });

    expect(genRes.completedBlocks).toBe(fx.completedBlocks);
    expect(genRes.totalNfe).toBe(fx.totalNfe);
    expect(genRes.stoppedByEos).toBe(fx.stoppedByEos);
    expect(Array.from(genRes.generatedTokenIds)).toEqual(fx.generatedTokenIds);
    expect(Array.from(genRes.fullSequenceTokenIds)).toEqual(fx.fullSequenceTokenIds);

    for (let bIdx = 0; bIdx < fx.blocks.length; bIdx++) {
      const blkActual = genRes.blocks[bIdx];
      const blkRef = fx.blocks[bIdx];
      expect(blkActual.blockId).toBe(blkRef.blockId);
      expect(blkActual.nfe).toBe(blkRef.nfe);
      expect(blkActual.earlyCompleted).toBe(blkRef.earlyCompleted);
      expect(blkActual.stoppedByEos).toBe(blkRef.stoppedByEos);
      expect(Array.from(blkActual.finalBlockTokens)).toEqual(blkRef.finalBlockTokens);
      expect(blkActual.contextCommitted).toBe(true);

      for (let rIdx = 0; rIdx < blkRef.refinements.length; rIdx++) {
        const refActual = blkActual.refinements[rIdx];
        const refExpected = blkRef.refinements[rIdx];
        expect(refActual.refinementIndex).toBe(refExpected.refinementIndex);
        expect(refActual.denoiserPass.tModel).toBe(refExpected.tModel);
        expect(Array.from(refActual.unmaskStep.xtBefore)).toEqual(refExpected.xtBefore);
        expect(Array.from(refActual.unmaskStep.predicted)).toEqual(refExpected.predicted);
        expect(Array.from(refActual.unmaskStep.committedIndices)).toEqual(refExpected.committedIndices);
        expect(Array.from(refActual.unmaskStep.thresholdCommittedIndices)).toEqual(refExpected.thresholdCommittedIndices);
        expect(Array.from(refActual.unmaskStep.floorCommittedIndices)).toEqual(refExpected.floorCommittedIndices);
        expect(Array.from(refActual.unmaskStep.remaskedIndices)).toEqual(refExpected.remaskedIndices);
        expect(refActual.unmaskStep.commitReason).toBe(refExpected.commitReason);
        expect(Array.from(refActual.unmaskStep.xtNext)).toEqual(refExpected.xtNext);
        expect(refActual.unmaskStep.isBlockComplete).toBe(refExpected.isBlockComplete);
        expect(refActual.unmaskStep.earlyCompleted).toBe(refExpected.earlyCompleted);

        recordErr(
          'Cat 6 (2-Block Trajectory)',
          assertFiniteArrayParity(
            refActual.denoiserPass.logits,
            refExpected.logits,
            tol.lmHeadLogitsMaxAbs,
            `cat6.b${bIdx}.r${rIdx}.logits`,
          ),
        );
        recordErr(
          'Cat 6 (2-Block Trajectory)',
          assertConfidenceParity(
            refActual.unmaskStep.confidence,
            refExpected.confidence,
            tol.lmHeadProbsMaxAbs,
            `cat6.b${bIdx}.r${rIdx}.conf`,
          ),
        );
      }

      // Post-commit Context snapshot parity (Mamba S-1, mambaS2, all 6 GQA KV caches, AR logits/probs)
      const postCtx = blkActual.nextSnapshot;
      const postRef = blkRef.postCommitContext;
      expect(postCtx.ctxState.processedTokens).toBe(postRef.prefixLen);
      expect(Array.from(postCtx.ctxState.prefixTokenIds)).toEqual(postRef.prefixTokenIds);
      recordErr('Cat 6 (2-Block Trajectory)', assertFiniteArrayParity(postCtx.arLogits, postRef.arLogits, tol.lmHeadLogitsMaxAbs, `cat6.b${bIdx}.postCtx.arLogits`));
      recordErr('Cat 6 (2-Block Trajectory)', assertFiniteArrayParity(postCtx.arProbs, postRef.arProbs, tol.lmHeadProbsMaxAbs, `cat6.b${bIdx}.postCtx.arProbs`));
      for (const [bayKey, mRef] of Object.entries(postRef.mambaS1)) {
        const m = postCtx.ctxState.mambaStates.get(Number(bayKey))!;
        recordErr('Cat 6 (2-Block Trajectory)', assertFiniteArrayParity(m.convState, mRef.convState, tol.cacheStateMaxAbs, `cat6.b${bIdx}.postCtx.m${bayKey}.conv`));
        recordErr('Cat 6 (2-Block Trajectory)', assertFiniteArrayParity(m.ssmState, mRef.ssmState, tol.cacheStateMaxAbs, `cat6.b${bIdx}.postCtx.m${bayKey}.ssm`));
      }
      const mS2 = postCtx.mambaS2.get(1)!;
      recordErr('Cat 6 (2-Block Trajectory)', assertFiniteArrayParity(mS2.convState, postRef.mambaS2Bay01.convState, tol.cacheStateMaxAbs, `cat6.b${bIdx}.postCtx.s2m1.conv`));
      recordErr('Cat 6 (2-Block Trajectory)', assertFiniteArrayParity(mS2.ssmState, postRef.mambaS2Bay01.ssmState, tol.cacheStateMaxAbs, `cat6.b${bIdx}.postCtx.s2m1.ssm`));
      for (const [bayKey, gRef] of Object.entries(postRef.gqaKv)) {
        const g = extractCompactKv(postCtx.ctxState.gqaCaches.get(Number(bayKey))!);
        recordErr('Cat 6 (2-Block Trajectory)', assertFiniteArrayParity(g.K, gRef.K, tol.cacheStateMaxAbs, `cat6.b${bIdx}.postCtx.g${bayKey}.K`));
        recordErr('Cat 6 (2-Block Trajectory)', assertFiniteArrayParity(g.V, gRef.V, tol.cacheStateMaxAbs, `cat6.b${bIdx}.postCtx.g${bayKey}.V`));
      }
    }
  });

  it('Cat 7: matches PyTorch 2.15.0 AR & Mock-AR (perturbed + identity perturbationScale=0 proving Mock-AR === Context AR)', () => {
    const fx = TRAJ_FX.cat7ArAndMockArAlignment;
    const tol = TRAJ_FX.metadata.tolerances;

    const verifyVariant = (
      model: TwoTowerModelParameters,
      variantRef: ArMockArVariantFixture,
      label: string,
      expectBitExactMockEqualsAr: boolean,
    ) => {
      // 1. AR (2 steps)
      let snapAr = runTwoTowerContextPrefill(model, fx.promptTokenIds, {
        kvCapacity: fx.kvCapacity,
      });
      const arResults = [];
      for (let s = 0; s < fx.numSteps; s++) {
        const stepRes = runTwoTowerArStep(model, snapAr);
        arResults.push(stepRes);
        const refS = variantRef.arSteps[s];
        expect(stepRes.tokenId, `${label}.ar[${s}].tokenId`).toBe(refS.tokenId);
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(stepRes.logits, refS.logits, tol.lmHeadLogitsMaxAbs, `${label}.ar[${s}].logits`));
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(stepRes.probs, refS.probs, tol.lmHeadProbsMaxAbs, `${label}.ar[${s}].probs`));
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(stepRes.nextSnapshot.arLogits, refS.postCommitArLogits, tol.lmHeadLogitsMaxAbs, `${label}.ar[${s}].postArLogits`));
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(stepRes.nextSnapshot.ctxState.mambaStates.get(1)!.ssmState, refS.postCommitBay01Ssm, tol.cacheStateMaxAbs, `${label}.ar[${s}].ssm`));
        const g43 = extractCompactKv(stepRes.nextSnapshot.ctxState.gqaCaches.get(43)!);
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(g43.K, refS.postCommitBay43K, tol.cacheStateMaxAbs, `${label}.ar[${s}].g43.K`));
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(g43.V, refS.postCommitBay43V, tol.cacheStateMaxAbs, `${label}.ar[${s}].g43.V`));
        snapAr = stepRes.nextSnapshot;
      }

      // 2. Mock-AR (2 steps)
      let snapMock = runTwoTowerContextPrefill(model, fx.promptTokenIds, {
        kvCapacity: fx.kvCapacity,
      });
      const mockResults = [];
      for (let s = 0; s < fx.numSteps; s++) {
        const stepRes = runTwoTowerMockArStep(model, snapMock, { blockId: s, refinementIndex: 0 });
        mockResults.push(stepRes);
        const refS = variantRef.mockArSteps[s];
        expect(stepRes.denoiserPass.tModel).toBeNull();
        expect(stepRes.denoiserPass.tEmb).toBeNull();
        expect(stepRes.conditioning.committedPrefixLen).toBe(refS.committedPrefixLen);
        expect(stepRes.denoiserPass.tokenIds[0]).toBe(refS.queryTokenId);
        expect(stepRes.tokenId, `${label}.mock[${s}].tokenId`).toBe(refS.tokenId);
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(stepRes.denoiserPass.logits, refS.denoiserLogits, tol.lmHeadLogitsMaxAbs, `${label}.mock[${s}].logits`));
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(stepRes.denoiserPass.probs, refS.denoiserProbs, tol.lmHeadProbsMaxAbs, `${label}.mock[${s}].probs`));
        for (let k = 0; k <= 52; k++) {
          recordErr(
            'Cat 7 (AR & Mock-AR)',
            assertFiniteArrayParity(
              stepRes.denoiserPass.residualStreams[k],
              refS.residualStreams[k],
              tol.residualCheckpointMaxAbs,
              `${label}.mock[${s}].res[${k}]`,
            ),
          );
        }
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(stepRes.nextSnapshot.arLogits, refS.postCommitArLogits, tol.lmHeadLogitsMaxAbs, `${label}.mock[${s}].postArLogits`));
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(stepRes.nextSnapshot.ctxState.mambaStates.get(1)!.ssmState, refS.postCommitBay01Ssm, tol.cacheStateMaxAbs, `${label}.mock[${s}].ssm`));
        const g43 = extractCompactKv(stepRes.nextSnapshot.ctxState.gqaCaches.get(43)!);
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(g43.K, refS.postCommitBay43K, tol.cacheStateMaxAbs, `${label}.mock[${s}].g43.K`));
        recordErr('Cat 7 (AR & Mock-AR)', assertFiniteArrayParity(g43.V, refS.postCommitBay43V, tol.cacheStateMaxAbs, `${label}.mock[${s}].g43.V`));
        snapMock = stepRes.nextSnapshot;
      }

      if (expectBitExactMockEqualsAr) {
        // Under identical V=34, float64_reference, perturbationScale=0: Mock-AR === Context AR (0 ULP!)
        for (let s = 0; s < fx.numSteps; s++) {
          expect(mockResults[s].tokenId).toBe(arResults[s].tokenId);
          assertBitExactFiniteArray(mockResults[s].denoiserPass.logits, arResults[s].logits, `${label}.mockVsAr[${s}].logits`);
          assertBitExactFiniteArray(mockResults[s].denoiserPass.probs, arResults[s].probs, `${label}.mockVsAr[${s}].probs`);
          expect(mockResults[s].nextSnapshot.snapshotDigest).toBe(arResults[s].nextSnapshot.snapshotDigest);
        }
      }
    };

    verifyVariant(defaultModel, fx.defaultPerturbed, 'cat7.perturbed', false);
    verifyVariant(identityModel, fx.identityZeroPerturbation, 'cat7.identity', true);
  });

  it('Cat 8: matches PyTorch 2.15.0 sampler edge cases (mask exclusion, strict > & floor, early exit, EOS, deterministic Gumbel-max)', () => {
    const fx = TRAJ_FX.cat8SamplerEdgeCases;
    const tolLp = TRAJ_FX.metadata.tolerances.samplerLogProbMaxAbs;
    const tolP = TRAJ_FX.metadata.tolerances.samplerProbMaxAbs;

    const assertSamplerStepMatches = (
      stepFx: SamplerStepFixture,
      threshold: number,
      temperature: number,
      label: string,
    ) => {
      const logits = new Float32Array(stepFx.logits);
      const constrained = mdlmConstrainLogProbs(logits, stepFx.xtBefore, fx.vocabSize, fx.maskTokenId);
      recordErr('Cat 8 (Sampler Edge Cases)', assertFiniteArrayParity(constrained.logXTheta, stepFx.logXTheta, tolLp, `${label}.mdlm.logXTheta`));
      recordErr('Cat 8 (Sampler Edge Cases)', assertFiniteArrayParity(constrained.xTheta, stepFx.unscaledProbs, tolP, `${label}.mdlm.xTheta`));

      const res = runConfidenceUnmaskStep({
        xt: stepFx.xtBefore,
        logits,
        vocabSize: fx.vocabSize,
        maskTokenId: fx.maskTokenId,
        stepIndex: stepFx.stepIndex,
        stepsPerBlock: stepFx.stepsPerBlock,
        confidenceThreshold: threshold,
        temperature,
        uniformVariates: stepFx.uniformVariates,
      });

      expect(res.tModel, `${label}.tModel`).toBe(stepFx.tModel);
      expect(Array.from(res.predicted), `${label}.predicted`).toEqual(stepFx.predicted);
      recordErr('Cat 8 (Sampler Edge Cases)', assertConfidenceParity(res.confidence, stepFx.confidence, tolP, `${label}.conf`));
      recordErr('Cat 8 (Sampler Edge Cases)', assertFiniteArrayParity(res.logXTheta, stepFx.logXTheta, tolLp, `${label}.logXTheta`));
      recordErr('Cat 8 (Sampler Edge Cases)', assertFiniteArrayParity(res.unscaledProbs, stepFx.unscaledProbs, tolP, `${label}.xTheta`));
      expect(res.numAboveThreshold, `${label}.numAboveThreshold`).toBe(stepFx.numAboveThreshold);
      expect(res.minCommitFloor, `${label}.minCommitFloor`).toBe(stepFx.minCommitFloor);
      expect(res.tokensCommittedCount, `${label}.tokensCommittedCount`).toBe(stepFx.tokensCommittedCount);
      expect(res.commitReason, `${label}.commitReason`).toBe(stepFx.commitReason);
      expect(Array.from(res.committedIndices), `${label}.committedIndices`).toEqual(stepFx.committedIndices);
      expect(Array.from(res.thresholdCommittedIndices), `${label}.thresholdCommittedIndices`).toEqual(stepFx.thresholdCommittedIndices);
      expect(Array.from(res.floorCommittedIndices), `${label}.floorCommittedIndices`).toEqual(stepFx.floorCommittedIndices);
      expect(Array.from(res.remaskedIndices), `${label}.remaskedIndices`).toEqual(stepFx.remaskedIndices);
      expect(Array.from(res.xtNext), `${label}.xtNext`).toEqual(stepFx.xtNext);
      expect(res.nMaskedBefore, `${label}.nMaskedBefore`).toBe(stepFx.nMaskedBefore);
      expect(res.nMaskedAfter, `${label}.nMaskedAfter`).toBe(stepFx.nMaskedAfter);
      expect(res.earlyCompleted, `${label}.earlyCompleted`).toBe(stepFx.earlyCompleted);
      expect(res.isBlockComplete, `${label}.isBlockComplete`).toBe(stepFx.isBlockComplete);
      expect(res.containsEos(fx.eosTokenId), `${label}.containsEos`).toBe(stepFx.containsEos);
      return res;
    };

    // 8A: Mask exclusion (prob[32] === 0 even with +50 logit!) & carry-over delta
    const maskRes = assertSamplerStepMatches(
      fx.maskExclusionAndCarryOver,
      fx.maskExclusionAndCarryOver.confidenceThreshold!,
      fx.maskExclusionAndCarryOver.temperature!,
      'cat8.maskExclusion',
    );
    for (let j = 0; j < 4; j++) {
      expect(maskRes.unscaledProbs[j * fx.vocabSize + fx.maskTokenId]).toBe(0);
    }
    expect(maskRes.predicted[0]).toBe(5);
    expect(maskRes.predicted[3]).toBe(12);
    expect(maskRes.predicted[1]).not.toBe(fx.maskTokenId);
    expect(maskRes.predicted[2]).not.toBe(fx.maskTokenId);

    // 8B: Strict '>' threshold (exact 0.5 excluded!) + budget floor (K=2 < B=4 -> ceil(4/2) = 2)
    const budgetRes = assertSamplerStepMatches(
      fx.strictThresholdAndBudgetFloor,
      fx.strictThresholdAndBudgetFloor.confidenceThreshold!,
      fx.strictThresholdAndBudgetFloor.temperature!,
      'cat8.budgetFloor',
    );
    expect(budgetRes.confidence[0]).toBe(0.5);
    expect(budgetRes.numAboveThreshold).toBe(1);
    expect(budgetRes.commitReason).toBe('budget_floor');

    // 8B-ii: Progress floor with exact threshold equality
    const progRes = assertSamplerStepMatches(
      fx.progressFloorExactEquality,
      fx.progressFloorExactEquality.confidenceThreshold!,
      fx.progressFloorExactEquality.temperature!,
      'cat8.progressFloor',
    );
    expect(progRes.numAboveThreshold).toBe(0);
    expect(progRes.commitReason).toBe('progress_floor');

    // 8C: Early completion (no-op on 0 masked + mid-budget completion)
    assertSamplerStepMatches(
      fx.earlyCompletionNoOp,
      fx.earlyCompletionNoOp.confidenceThreshold!,
      fx.earlyCompletionNoOp.temperature!,
      'cat8.earlyNoOp',
    );
    assertSamplerStepMatches(
      fx.earlyCompletionMidBudget,
      fx.earlyCompletionMidBudget.confidenceThreshold!,
      fx.earlyCompletionMidBudget.temperature!,
      'cat8.earlyMid',
    );

    // 8D: Post-block EOS halt (partial step 0 with EOS still continues; step 1 completes and halts!)
    const eos0 = assertSamplerStepMatches(
      fx.postBlockEosHalt.step0PartialWithEos,
      fx.postBlockEosHalt.confidenceThreshold,
      fx.postBlockEosHalt.temperature,
      'cat8.eos0',
    );
    expect(eos0.isBlockComplete).toBe(false);
    const eos1 = assertSamplerStepMatches(
      fx.postBlockEosHalt.step1CompletedWithEos,
      fx.postBlockEosHalt.confidenceThreshold,
      fx.postBlockEosHalt.temperature,
      'cat8.eos1',
    );
    expect(eos1.isBlockComplete).toBe(true);
    expect(blockContainsEos(eos1.xtNext, fx.eosTokenId)).toBe(true);
    expect(blockContainsEos(eos1.xtNext, [2, fx.eosTokenId])).toBe(true);
    expect(blockContainsEos([10, 12, 4, 8], fx.eosTokenId)).toBe(false);

    // 8E: Deterministic Gumbel-max at temperature = 0.7 with explicit shared uniform variates u
    const gumbelFx = fx.deterministicGumbelMax;
    const gumbelRes = assertSamplerStepMatches(
      gumbelFx,
      gumbelFx.confidenceThreshold!,
      gumbelFx.temperature!,
      'cat8.gumbelMax',
    );
    expect(gumbelRes.predicted[0]).not.toBe(gumbelFx.greedyPredictedForComparison![0]);
    expect(gumbelRes.predicted[1]).toBe(7);
    expect(gumbelRes.predicted).not.toContain(fx.maskTokenId);
    // Confidence is gathered from UNSCALED xTheta even when temperature = 0.7
    for (const j of [0, 2, 3]) {
      const expectedConf = gumbelRes.unscaledProbs[j * fx.vocabSize + gumbelRes.predicted[j]];
      expect(gumbelRes.confidence[j]).toBe(expectedConf);
    }
  });
});
