/**
 * Independent PyTorch 2.15.0 Progressive Self-Refining Diffusion, Dual-CFG
 * Image Editing & Read-Only OPD Verification Suite (`tests/uni-diffusion-parity.test.ts`)
 *
 * Gate G3 Verification Suite covering:
 *   - Cryptographic SHA-256 over ALL 766 tensors (`618` core + `148` multimodal =
 *     `191,235` floats = `764,940 B`) and pinned `torch==2.15.0`
 *   - Category 1: `BalancedSemanticTree` (`leafToToken`/`tokenToLeaf` 0-ULP, `heapRadii`
 *     & `pca3d` `<= 2e-5`), `U26` `D=17` vs `D=8` floor-rounding mismatch (`K ∈ {1,4,8,16}`),
 *     `semanticTreeLevelsForAges` (`N ∈ {1,2,4,8}` incl. `N=1` binary MDM),
 *     `sampleRepresentatives` (0-ULP), `buildSemanticTreeEmbeddings` (`U10`, `<= 2e-5`),
 *     `agesAtTime` (`Eq. 16`, 0-ULP), `shift=5.0` quotas + `N-1` flush rounds (0-ULP),
 *     and `stratifiedRandom(4, 42)` (0-ULP)
 *   - Category 2: Full T2I trajectories (`steps=8, N=4` -> `11` rounds + `1` prefill = `12` passes,
 *     `B_cfg=2, w=3.0, T=0.8` with shared explicit variates `U23`), `N=1` binary MDM limit,
 *     `guidanceScale=0` `U30` `B_cfg=1` short-circuit, finalization immutability, AND `D5`
 *     `0`-ULP `elastic_pruned` vs `hf_faithful` Stage-2 equivalence
 *   - Category 3: Full 3-branch Dual-CFG Image Editing on non-square `harbor_panorama_16x8 -> 16x16`
 *     (`B_cfg=3`, `legacy` `w_txt=1.6, w_img=2.0`, `lockSourcePaddingVq=true` 0-ULP invariance at
 *     every round, `cropToSource=true`, 1-step `independent` & `ip2p` mixer checks `<= 2e-5`,
 *     AND `D5` `0`-ULP `elastic_pruned` vs `hf_faithful` across all 3 CFG rows)
 *   - Category 4: Read-only OPD reference parity (`129`-bucket `tail_kl`, `matched_sigmas(0.9)`,
 *     flow `x^r` + `exact_ce + 0.2 * grouped_ce`, `<= 2e-5`, `0` gradients)
 *   - Unit & Hygiene Checks: `applyTopKTopPFilter`, `U22` extreme-logit Gumbel safety, fixed
 *     `16 * 256` draw count, zero `Math.random()`, zero forbidden sibling/controls imports.
 */

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  assertBitExactFiniteArray,
  assertFiniteArrayParity,
} from './helpers/parityAssert';
import { collectUniCodecWeights } from '../src/uni/codec';
import {
  applyEditDualCfgLogits,
  applyT2iCfgLogits,
  applyTopKTopPFilter,
  createDefaultUniRuntimeBundle,
  runImageEditTrajectory,
  runT2iTrajectory,
  sampleVqProposals,
} from '../src/uni/imageSampler';
import {
  computeMatchedSigmas,
  createDefaultOpdDemoInputs,
  evaluateFlowEditingOpdRef,
  evaluateUnderstandingOpdRef,
} from '../src/uni/opdRef';
import { collectUniBackboneWeights } from '../src/uni/parameters';
import { createXoshiro128ss } from '../src/uni/prng';
import {
  agesAtTime,
  analyzePrimeDepthFloorMismatch,
  buildSemanticTreeEmbeddings,
  contentIndexToMarkedGridIndex,
  projectCodebookPca3d,
  sampleRepresentatives,
  semanticTreeActivationCounts,
  semanticTreeLevelsForAges,
  stratifiedRandom,
} from '../src/uni/tree';
import {
  UNI_MINI_CORE_FLOAT_COUNT,
  UNI_MINI_FULL_FLOAT_COUNT,
  UNI_MINI_FULL_TENSOR_COUNT,
  UNI_MINI_FULL_WEIGHT_BYTES,
  UNI_MINI_TREE_BUFFER_BYTES,
  type UniImageSamplingVariates,
} from '../src/uni/types';
import {
  collectUniVisionWeights,
  createUniScenePreset,
} from '../src/uni/vision';
import { collectUniVqVaeWeights } from '../src/uni/vqvae';

// ============================================================================
// FIXTURE SCHEMA & RUNTIME INITIALIZATION
// ============================================================================

interface TrajectoryStepFixture {
  roundIndex: number;
  isFlushRound: boolean;
  activatedContentIndices: number[];
  proposals: number[];
  stagesAfter: number[];
  representativesAfter: number[];
  levelsAfter: number[];
  noiseWeightsAfter: number[];
  top4TokenIds: number[];
  top4Probs: number[];
  perBranchFinalNorm: number[][];
}

interface TrajectoryFixture {
  taskKind: 'text_to_image' | 'image_edit';
  cfgBatchSize: 1 | 2 | 3;
  guidanceMode: 'legacy' | 'independent' | 'ip2p' | null;
  guidanceScaleText: number;
  guidanceScaleImage: number;
  activationSteps: number;
  refinementSteps: number;
  temperature: number;
  elasticPrunedVerifiedZeroUlp: boolean;
  passCount: number;
  branchForwardCount: number;
  round0GuidedLogits: number[];
  finalRoundGuidedLogits: number[] | null;
  round0DualCfgAltModes?: {
    independent: number[];
    ip2p: number[];
  };
  steps: TrajectoryStepFixture[];
  finalMarkedCodes: number[];
  finalContentCodes: number[];
  decodedPaddedImage: number[];
  outputImage: number[];
  outputWidth: number;
  outputHeight: number;
}

interface U26MismatchCaseFixture {
  treeDepth: number;
  refinementSteps: number;
  hasMismatch: boolean;
  mismatchedStageCount: number;
  rows: Array<{
    stageIndex: number;
    trainAge: number;
    trainLevel: number;
    inferLevel: number;
    levelDelta: number;
    trainBallSize: number;
    inferBallSize: number;
  }>;
}

interface UniDiffusionParityFixture {
  schemaVersion: string;
  modelId: string;
  backboneSeed: number;
  multimodalSeed: number;
  backboneFingerprint: string;
  counts: {
    totalTensors: number;
    totalFloats: number;
    totalBytes: number;
  };
  weightsSha256: string;
  coreSha256: string;
  multimodalSha256: string;
  torchVersion: string;
  pythonVersion: string;
  semanticTree: {
    tree: {
      vocabSize: number;
      depth: number;
      latentDim: number;
      leafToToken: number[];
      tokenToLeaf: number[];
      heapRadii: number[];
      pca3d: number[];
    };
    u26FloorMismatch: {
      d17: Record<string, U26MismatchCaseFixture>;
      d8: Record<string, U26MismatchCaseFixture>;
    };
    levelsByRefinementSteps: Record<
      string,
      {
        stages: number[];
        levels: number[];
        noiseWeights: number[];
        ballSizes: number[];
      }
    >;
    sampleRepresentatives: {
      proposals: number[];
      levels: number[];
      randBits: number[];
      representatives: number[];
    };
    buildEmbeddings: {
      representatives: number[];
      noiseWeights: number[];
      embeddings: number[];
    };
    agesAtTimeEq16: {
      tau: number[];
      cases: Array<{
        t: number;
        epsilon: number;
        K: number;
        ages: number[];
      }>;
    };
    quotas: Record<
      string,
      {
        contentCount: number;
        activationSteps: number;
        refinementSteps: number;
        flushRounds: number;
        totalRounds: number;
        activationCounts: number[];
        cumulativeActivations: number[];
      }
    >;
    stratifiedOrder4Seed42: number[];
  };
  sharedExplicitVariates: {
    initialRootCodes: number[];
    roundRepRandBits: number[][];
    roundProposalUniforms: number[][];
  };
  t2i: {
    promptTokenIds: number[];
    defaultSteps8N4: TrajectoryFixture;
    binaryMdmN1: TrajectoryFixture;
    guidanceZeroU30: TrajectoryFixture;
  };
  imageEdit: {
    scenePresetId: 'harbor_panorama_16x8';
    promptTokenIds: number[];
    sourceVqCodes: number[];
    sourcePaddingMask: number[];
    harborLegacy: TrajectoryFixture;
  };
  opdRef: {
    seed: number;
    inputs: {
      responseLen: number;
      teacherArProbs: number[];
      studentArProbs: number[];
      studentDiffProbs: number[];
      x0: number[];
      noiseDelta: number[];
      exactLogits: number[];
      targetCode: number;
      groupLogits: number[];
      targetGroup: number;
    };
    understanding: {
      expectedMaskDenominator: number;
      arTailKl: {
        unionIndices: number[];
        bucketCount: number;
        pTargetBuckets: number[];
        pStudentBuckets: number[];
        pTargetTailMass: number;
        pStudentTailMass: number;
        reverseKl: number;
        forwardKl: number;
      };
      consistencyTailKl: {
        unionIndices: number[];
        bucketCount: number;
        pTargetBuckets: number[];
        pStudentBuckets: number[];
        pTargetTailMass: number;
        pStudentTailMass: number;
        reverseKl: number;
        forwardKl: number;
      };
      arOpdLoss: number;
      selfConsistencyLoss: number;
      totalUnderstandingOpdLoss: number;
    };
    flowEditing: {
      r: number;
      matchedSigmas: {
        r: number;
        seqLen: number;
        steps: number;
        mu: number;
        sigmaStart: number;
        sigmaEnd: number;
        sigmas: number[];
      };
      xNoisyR: number[];
      exactCe: number;
      groupedCe: number;
      groupedCeWeight: number;
      totalEditingOpdLoss: number;
    };
  };
}

const FIXTURE_PATH = resolve(process.cwd(), 'tests/fixtures/pytorch-uni-diffusion.json');
const RAW_FIXTURE_TEXT = readFileSync(FIXTURE_PATH, 'utf-8');
const FIXTURE = JSON.parse(RAW_FIXTURE_TEXT) as UniDiffusionParityFixture;
const BUNDLE = createDefaultUniRuntimeBundle();

const SHARED_VARIATES: UniImageSamplingVariates = {
  initialRootCodes: FIXTURE.sharedExplicitVariates.initialRootCodes,
  roundRepRandBits: FIXTURE.sharedExplicitVariates.roundRepRandBits,
  roundProposalUniforms: FIXTURE.sharedExplicitVariates.roundProposalUniforms,
};

function computeRuntime766TensorDigests(): {
  tensorCount: number;
  floatCount: number;
  byteCount: number;
  weightsSha256: string;
  coreSha256: string;
  multimodalSha256: string;
} {
  const coreTensors = collectUniBackboneWeights(BUNDLE.core);
  const visionTensors = collectUniVisionWeights(BUNDLE.vision);
  const codecTensors = collectUniCodecWeights(BUNDLE.codec);
  const vqTensors = collectUniVqVaeWeights(BUNDLE.vqvae);

  const allData: Float32Array[] = [
    ...coreTensors.map((t) => t.data),
    ...visionTensors.map((t) => t.data),
    ...codecTensors.map((t) => t.data),
    ...vqTensors.map((t) => t.data),
  ];

  let floatCount = 0;
  for (const arr of allData) floatCount += arr.length;
  const binBuf = Buffer.allocUnsafe(floatCount * 4);
  let off = 0;
  for (const arr of allData) {
    for (let i = 0; i < arr.length; i++) {
      binBuf.writeFloatLE(arr[i], off);
      off += 4;
    }
  }

  const coreBytes = UNI_MINI_CORE_FLOAT_COUNT * 4;
  return {
    tensorCount: allData.length,
    floatCount,
    byteCount: binBuf.byteLength,
    weightsSha256: createHash('sha256').update(binBuf).digest('hex'),
    coreSha256: createHash('sha256').update(binBuf.subarray(0, coreBytes)).digest('hex'),
    multimodalSha256: createHash('sha256').update(binBuf.subarray(coreBytes)).digest('hex'),
  };
}

// ============================================================================
// SUITE 0: CRYPTOGRAPHIC 766-TENSOR PROVENANCE & PYTORCH 2.15.0 PIN
// ============================================================================

describe('UNI3 Gate G3 — 766-Tensor SHA-256 Provenance & Pinned PyTorch 2.15.0', () => {
  it('matches all 766 runtime tensor digests and enforces pinned torch==2.15.0 & compact JSON', () => {
    expect(FIXTURE.schemaVersion).toBe('uni-diffusion-parity-v1');
    expect(FIXTURE.torchVersion.startsWith('2.15.0')).toBe(true);
    expect(RAW_FIXTURE_TEXT.includes('\n  ')).toBe(false); // compact serialization

    const digests = computeRuntime766TensorDigests();
    expect(digests.tensorCount).toBe(UNI_MINI_FULL_TENSOR_COUNT);
    expect(digests.floatCount).toBe(UNI_MINI_FULL_FLOAT_COUNT);
    expect(digests.byteCount).toBe(UNI_MINI_FULL_WEIGHT_BYTES);
    expect(digests.weightsSha256).toBe(FIXTURE.weightsSha256);
    expect(digests.coreSha256).toBe(FIXTURE.coreSha256);
    expect(digests.multimodalSha256).toBe(FIXTURE.multimodalSha256);
    expect(BUNDLE.tree.byteLength).toBe(UNI_MINI_TREE_BUFFER_BYTES);
  });
});

// ============================================================================
// CATEGORY 1: BALANCED SEMANTIC TREE, REFINEMENT MATH & U26
// ============================================================================

describe('UNI3 Gate G3 — Category 1: BalancedSemanticTree, Refinement & U26 Parity', () => {
  it('matches PyTorch 2.15.0 BalancedSemanticTree permutations (0 ULP), heapRadii & pca3d (<= 2e-5)', () => {
    const ref = FIXTURE.semanticTree.tree;
    assertBitExactFiniteArray(BUNDLE.tree.leafToToken, ref.leafToToken, 'tree.leafToToken');
    assertBitExactFiniteArray(BUNDLE.tree.tokenToLeaf, ref.tokenToLeaf, 'tree.tokenToLeaf');
    assertFiniteArrayParity(BUNDLE.tree.heapRadii, ref.heapRadii, 2e-5, 'tree.heapRadii');

    const pca3d = projectCodebookPca3d(BUNDLE.vqvae.codebook, 256, 8);
    assertFiniteArrayParity(pca3d, ref.pca3d, 2e-5, 'tree.pca3d');
  });

  it('matches U26 D=17 prime-depth floor-rounding mismatch vs D=8 exact divisibility to 0 ULP', () => {
    for (const k of [1, 4, 8, 16] as const) {
      const actual = analyzePrimeDepthFloorMismatch(17, k);
      const ref = FIXTURE.semanticTree.u26FloorMismatch.d17[String(k)];
      expect(actual.hasMismatch).toBe(ref.hasMismatch);
      expect(actual.mismatchedStageCount).toBe(ref.mismatchedStageCount);
      expect(actual.rows.length).toBe(ref.rows.length);
      for (let i = 0; i < ref.rows.length; i++) {
        expect(actual.rows[i].trainLevel).toBe(ref.rows[i].trainLevel);
        expect(actual.rows[i].inferLevel).toBe(ref.rows[i].inferLevel);
        expect(actual.rows[i].levelDelta).toBe(ref.rows[i].levelDelta);
        expect(actual.rows[i].trainBallSize).toBe(ref.rows[i].trainBallSize);
        expect(actual.rows[i].inferBallSize).toBe(ref.rows[i].inferBallSize);
      }
    }

    for (const k of [1, 2, 4, 8] as const) {
      const actual = analyzePrimeDepthFloorMismatch(8, k);
      const ref = FIXTURE.semanticTree.u26FloorMismatch.d8[String(k)];
      expect(actual.hasMismatch).toBe(false);
      expect(actual.mismatchedStageCount).toBe(0);
      expect(ref.mismatchedStageCount).toBe(0);
    }
  });

  it('matches semanticTreeLevelsForAges (N in {1,2,4,8}), sampleRepresentatives, U10 embeddings, Eq. 16 ages, shift=5.0 quotas & stratifiedRandom(4,42)', () => {
    for (const nRef of [1, 2, 4, 8] as const) {
      const ref = FIXTURE.semanticTree.levelsByRefinementSteps[String(nRef)];
      const actual = semanticTreeLevelsForAges(ref.stages, 8, nRef);
      assertBitExactFiniteArray(actual.levels, ref.levels, `levels.N${nRef}`);
      assertBitExactFiniteArray(actual.ballSizes, ref.ballSizes, `ballSizes.N${nRef}`);
      assertFiniteArrayParity(actual.noiseWeights, ref.noiseWeights, 1e-6, `noiseWeights.N${nRef}`);
    }

    // Bitwise O(1) representative sampling (0 ULP)
    const repRef = FIXTURE.semanticTree.sampleRepresentatives;
    const actualReps = sampleRepresentatives(
      BUNDLE.tree,
      repRef.proposals,
      repRef.levels,
      repRef.randBits,
    );
    assertBitExactFiniteArray(actualReps, repRef.representatives, 'sampleRepresentatives');

    // Additive continuous mask embeddings (U10, <= 2e-5)
    const embRef = FIXTURE.semanticTree.buildEmbeddings;
    const actualEmbs = buildSemanticTreeEmbeddings(
      embRef.representatives,
      embRef.noiseWeights,
      BUNDLE.core.genEmbedding,
    );
    assertFiniteArrayParity(actualEmbs, embRef.embeddings, 2e-5, 'buildSemanticTreeEmbeddings');

    // Paper Eq. 16 continuous-time agesAtTime (0 ULP integer ages)
    const eq16 = FIXTURE.semanticTree.agesAtTimeEq16;
    for (const c of eq16.cases) {
      const actualAges = agesAtTime(c.t, eq16.tau, c.epsilon, c.K);
      assertBitExactFiniteArray(actualAges, c.ages, `agesAtTime(t=${c.t})`);
    }

    // shift=5.0 activation quotas + N-1 flush rounds (0 ULP)
    const qChecks: Array<[string, number, number]> = [
      ['steps8_N4', 8, 4],
      ['steps8_N1', 8, 1],
      ['steps4_N2', 4, 2],
    ];
    for (const [key, steps, nRef] of qChecks) {
      const ref = FIXTURE.semanticTree.quotas[key];
      const actual = semanticTreeActivationCounts(16, steps, nRef, 5.0, 0);
      expect(actual.flushRounds).toBe(ref.flushRounds);
      expect(actual.totalRounds).toBe(ref.totalRounds);
      assertBitExactFiniteArray(actual.activationCounts, ref.activationCounts, `quotas.${key}.counts`);
      assertBitExactFiniteArray(
        actual.cumulativeActivations,
        ref.cumulativeActivations,
        `quotas.${key}.cum`,
      );
    }

    // stratifiedRandom(4, 42) exact 0-ULP permutation
    const order = stratifiedRandom(4, 42);
    assertBitExactFiniteArray(order, FIXTURE.semanticTree.stratifiedOrder4Seed42, 'stratifiedRandom(4,42)');
  });
});

// ============================================================================
// CATEGORY 2: T2I TRAJECTORIES (B=2, N=1 MDM, U30 B=1, D5 0-ULP PRUNING)
// ============================================================================

describe('UNI3 Gate G3 — Category 2: T2I Trajectories, N=1 MDM, U30 Short-Circuit & D5 0-ULP', () => {
  it('matches full PyTorch 2.15.0 T2I trajectory (steps=8, N=4, B_cfg=2, w=3.0, T=0.8) & proves D5 0-ULP elastic_pruned == hf_faithful', () => {
    const ref = FIXTURE.t2i.defaultSteps8N4;
    expect(ref.elasticPrunedVerifiedZeroUlp).toBe(true);

    const commonOpts = {
      promptTokenIds: FIXTURE.t2i.promptTokenIds,
      activationSteps: 8 as const,
      refinementSteps: 4 as const,
      shift: 5.0,
      guidanceScale: 3.0,
      temperature: 0.8,
      topK: 0,
      topP: 1.0,
      gainPresetId: 'mini_default' as const,
      explicitVariates: SHARED_VARIATES,
      retainFullLogitsOnSteps: true,
    };

    const hf = runT2iTrajectory(BUNDLE, { ...commonOpts, prefillMode: 'hf_faithful' });
    const pruned = runT2iTrajectory(BUNDLE, { ...commonOpts, prefillMode: 'elastic_pruned' });

    expect(hf.cfgBatchSize).toBe(2);
    expect(hf.cfgShortCircuitApplied).toBe(false);
    expect(hf.passCount).toBe(12);
    expect(hf.branchForwardCount).toBe(24);
    expect(hf.steps.length).toBe(11);

    // D5 KV waste telemetry
    expect(hf.prefixCache.wasteTelemetry.wastedFraction).toBeCloseTo(16 / 34, 12);
    expect(pruned.prefixCache.wasteTelemetry.wastedSplitUBytes).toBe(0);
    expect(pruned.prefixCache.wasteTelemetry.wastedFraction).toBe(0);

    // Track finalization immutability across all 11 rounds
    const committedValues = new Map<number, number>();

    for (let r = 0; r < ref.steps.length; r++) {
      const sHf = hf.steps[r];
      const sEp = pruned.steps[r];
      const sRef = ref.steps[r];

      expect(sHf.isFlushRound).toBe(sRef.isFlushRound);
      assertBitExactFiniteArray(sHf.activatedContentIndices, sRef.activatedContentIndices, `t2i.r${r}.act`);
      assertBitExactFiniteArray(sHf.proposals, sRef.proposals, `t2i.r${r}.proposals`);
      assertBitExactFiniteArray(sHf.stagesAfter, sRef.stagesAfter, `t2i.r${r}.stages`);
      assertBitExactFiniteArray(sHf.representativesAfter, sRef.representativesAfter, `t2i.r${r}.reps`);
      assertBitExactFiniteArray(sHf.levelsAfter, sRef.levelsAfter, `t2i.r${r}.levels`);
      assertBitExactFiniteArray(sHf.top4Guided.tokenIds, sRef.top4TokenIds, `t2i.r${r}.top4Ids`);
      assertFiniteArrayParity(sHf.noiseWeightsAfter, sRef.noiseWeightsAfter, 1e-6, `t2i.r${r}.noise`);
      assertFiniteArrayParity(sHf.top4Guided.probs, sRef.top4Probs, 2e-5, `t2i.r${r}.top4Probs`);

      for (let b = 0; b < 2; b++) {
        assertFiniteArrayParity(sHf.perBranchFinalNorm[b], sRef.perBranchFinalNorm[b], 2e-5, `t2i.r${r}.b${b}.fn`);
        // D5 0-ULP bit-exact elastic_pruned == hf_faithful
        assertBitExactFiniteArray(sEp.perBranchFinalNorm[b], sHf.perBranchFinalNorm[b], `t2i.d5.r${r}.b${b}.fn`);
      }
      assertBitExactFiniteArray(sEp.guidedLogits!, sHf.guidedLogits!, `t2i.d5.r${r}.guidedLogits`);
      assertBitExactFiniteArray(sEp.representativesAfter, sHf.representativesAfter, `t2i.d5.r${r}.reps`);

      // Finalization immutability check: committed cells (stage === 4) never change
      for (let g = 0; g < sHf.stagesAfter.length; g++) {
        if (committedValues.has(g)) {
          expect(sHf.representativesAfter[g]).toBe(committedValues.get(g));
          expect(sHf.stagesAfter[g]).toBe(4);
        } else if (sHf.stagesAfter[g] >= 4) {
          committedValues.set(g, sHf.representativesAfter[g]);
        }
      }
    }

    assertFiniteArrayParity(hf.steps[0].guidedLogits!, ref.round0GuidedLogits, 2e-5, 't2i.r0.guidedLogits');
    assertFiniteArrayParity(
      hf.steps[hf.steps.length - 1].guidedLogits!,
      ref.finalRoundGuidedLogits!,
      2e-5,
      't2i.rLast.guidedLogits',
    );
    assertBitExactFiniteArray(hf.finalMarkedCodes, ref.finalMarkedCodes, 't2i.finalMarkedCodes');
    assertBitExactFiniteArray(hf.finalContentCodes, ref.finalContentCodes, 't2i.finalContentCodes');
    assertFiniteArrayParity(hf.outputImage.data, ref.outputImage, 2e-5, 't2i.outputImage');
    assertBitExactFiniteArray(pruned.outputImage.data, hf.outputImage.data, 't2i.d5.outputImage');
  });

  it('matches PyTorch 2.15.0 T2I N=1 binary MDM limit (0 flush rounds, 8 denoise passes)', () => {
    const ref = FIXTURE.t2i.binaryMdmN1;
    const res = runT2iTrajectory(BUNDLE, {
      promptTokenIds: FIXTURE.t2i.promptTokenIds,
      activationSteps: 8,
      refinementSteps: 1,
      shift: 5.0,
      guidanceScale: 3.0,
      temperature: 0.8,
      topK: 0,
      topP: 1.0,
      gainPresetId: 'mini_default',
      prefillMode: 'hf_faithful',
      explicitVariates: {
        initialRootCodes: SHARED_VARIATES.initialRootCodes,
        roundRepRandBits: SHARED_VARIATES.roundRepRandBits!.slice(0, 8),
        roundProposalUniforms: SHARED_VARIATES.roundProposalUniforms!.slice(0, 8),
      },
      retainFullLogitsOnSteps: true,
    });

    expect(res.schedule.flushRounds).toBe(0);
    expect(res.schedule.totalRounds).toBe(8);
    expect(res.passCount).toBe(9);
    for (let r = 0; r < 8; r++) {
      expect(res.steps[r].isFlushRound).toBe(false);
      assertBitExactFiniteArray(res.steps[r].stagesAfter, ref.steps[r].stagesAfter, `mdmN1.r${r}.stages`);
      assertBitExactFiniteArray(
        res.steps[r].representativesAfter,
        ref.steps[r].representativesAfter,
        `mdmN1.r${r}.reps`,
      );
      // Every non-unactivated cell is strictly committed (stage === 1, noise === 0)
      for (let g = 0; g < res.steps[r].stagesAfter.length; g++) {
        const st = res.steps[r].stagesAfter[g];
        expect(st === -1 || st === 1).toBe(true);
        expect(res.steps[r].noiseWeightsAfter[g]).toBe(st === 1 ? 0.0 : 1.0);
      }
    }
    assertFiniteArrayParity(res.steps[0].guidedLogits!, ref.round0GuidedLogits, 2e-5, 'mdmN1.r0.guidedLogits');
    assertBitExactFiniteArray(res.finalContentCodes, ref.finalContentCodes, 'mdmN1.finalContentCodes');
    assertFiniteArrayParity(res.outputImage.data, ref.outputImage, 2e-5, 'mdmN1.outputImage');
  });

  it('enforces U30 guidanceScale=0 B_cfg=1 short-circuit and matches PyTorch 2.15.0', () => {
    const ref = FIXTURE.t2i.guidanceZeroU30;
    const res = runT2iTrajectory(BUNDLE, {
      promptTokenIds: FIXTURE.t2i.promptTokenIds,
      activationSteps: 8,
      refinementSteps: 4,
      shift: 5.0,
      guidanceScale: 0,
      temperature: 0.8,
      topK: 0,
      topP: 1.0,
      gainPresetId: 'mini_default',
      prefillMode: 'elastic_pruned',
      explicitVariates: SHARED_VARIATES,
      retainFullLogitsOnSteps: true,
    });

    expect(res.cfgBatchSize).toBe(1);
    expect(res.cfgShortCircuitApplied).toBe(true);
    expect(res.hfGuidanceZeroDriftNote).toContain('U30');
    expect(res.passCount).toBe(12);
    expect(res.branchForwardCount).toBe(12);

    assertFiniteArrayParity(res.steps[0].guidedLogits!, ref.round0GuidedLogits, 2e-5, 'u30.r0.guidedLogits');
    assertBitExactFiniteArray(res.finalContentCodes, ref.finalContentCodes, 'u30.finalContentCodes');
    assertFiniteArrayParity(res.outputImage.data, ref.outputImage, 2e-5, 'u30.outputImage');
  });
});

// ============================================================================
// CATEGORY 3: 3-BRANCH DUAL-CFG IMAGE EDITING (`harbor_panorama_16x8`)
// ============================================================================

describe('UNI3 Gate G3 — Category 3: 3-Branch Dual-CFG Editing, U18 Lock & D5 0-ULP', () => {
  it('matches PyTorch 2.15.0 harbor_panorama_16x8 legacy editing, independent/ip2p mixers, U18 lock & D5 0-ULP', () => {
    const editRef = FIXTURE.imageEdit;
    const ref = editRef.harborLegacy;
    expect(ref.elasticPrunedVerifiedZeroUlp).toBe(true);

    const harbor = createUniScenePreset('harbor_panorama_16x8');
    const commonOpts = {
      sourceImage: harbor,
      promptTokenIds: editRef.promptTokenIds,
      guidanceMode: 'legacy' as const,
      guidanceScaleText: 1.6,
      guidanceScaleImage: 2.0,
      lockSourcePaddingVq: true,
      cropToSource: true,
      activationSteps: 8 as const,
      refinementSteps: 4 as const,
      shift: 5.0,
      temperature: 0.8,
      topK: 0,
      topP: 1.0,
      gainPresetId: 'mini_default' as const,
      explicitVariates: SHARED_VARIATES,
      retainFullLogitsOnSteps: true,
    };

    const hf = runImageEditTrajectory(BUNDLE, { ...commonOpts, prefillMode: 'hf_faithful' });
    const pruned = runImageEditTrajectory(BUNDLE, { ...commonOpts, prefillMode: 'elastic_pruned' });

    expect(hf.cfgBatchSize).toBe(3);
    expect(hf.passCount).toBe(12);
    expect(hf.branchForwardCount).toBe(36);
    expect(hf.sourcePaddingMask?.lockedCellCount).toBe(8);
    expect(hf.sourcePaddingMask?.editableCellCount).toBe(8);
    expect(hf.outputImage.width).toBe(16);
    expect(hf.outputImage.height).toBe(8);
    expect(hf.decodedPaddedImage.width).toBe(16);
    expect(hf.decodedPaddedImage.height).toBe(16);

    // D5 KV waste telemetry on 3-branch mixed editing prefix
    expect(hf.prefixCache.wasteTelemetry.wastedSplitUBytes).toBeGreaterThan(0);
    expect(pruned.prefixCache.wasteTelemetry.wastedSplitUBytes).toBe(0);
    expect(pruned.prefixCache.wasteTelemetry.wastedFraction).toBe(0);

    // Source VQ & padding lock mask (0 ULP)
    assertBitExactFiniteArray(hf.sourceVqCodes!, editRef.sourceVqCodes, 'edit.sourceVqCodes');
    assertBitExactFiniteArray(
      hf.sourcePaddingMask!.lockMask,
      editRef.sourcePaddingMask,
      'edit.sourcePaddingMask',
    );

    for (let r = 0; r < ref.steps.length; r++) {
      const sHf = hf.steps[r];
      const sEp = pruned.steps[r];
      const sRef = ref.steps[r];

      expect(sHf.lockedCellCount).toBe(8);
      assertBitExactFiniteArray(sHf.activatedContentIndices, sRef.activatedContentIndices, `edit.r${r}.act`);
      assertBitExactFiniteArray(sHf.proposals, sRef.proposals, `edit.r${r}.proposals`);
      assertBitExactFiniteArray(sHf.stagesAfter, sRef.stagesAfter, `edit.r${r}.stages`);
      assertBitExactFiniteArray(sHf.representativesAfter, sRef.representativesAfter, `edit.r${r}.reps`);
      assertBitExactFiniteArray(sHf.levelsAfter, sRef.levelsAfter, `edit.r${r}.levels`);
      assertBitExactFiniteArray(sHf.top4Guided.tokenIds, sRef.top4TokenIds, `edit.r${r}.top4Ids`);
      assertFiniteArrayParity(sHf.top4Guided.probs, sRef.top4Probs, 2e-5, `edit.r${r}.top4Probs`);

      // U18: 0-ULP locked padding cell immutability at EVERY round
      for (let c = 0; c < 16; c++) {
        if (editRef.sourcePaddingMask[c] === 1) {
          const g = contentIndexToMarkedGridIndex(c);
          expect(sHf.representativesAfter[g]).toBe(editRef.sourceVqCodes[c]);
          expect(sHf.stagesAfter[g]).toBe(4);
          expect(sHf.noiseWeightsAfter[g]).toBe(0.0);
        }
      }

      // Per-branch finalNorm parity + D5 0-ULP invariance across all 3 CFG branches
      for (let b = 0; b < 3; b++) {
        assertFiniteArrayParity(sHf.perBranchFinalNorm[b], sRef.perBranchFinalNorm[b], 2e-5, `edit.r${r}.b${b}.fn`);
        assertBitExactFiniteArray(sEp.perBranchFinalNorm[b], sHf.perBranchFinalNorm[b], `edit.d5.r${r}.b${b}.fn`);
      }
      assertBitExactFiniteArray(sEp.guidedLogits!, sHf.guidedLogits!, `edit.d5.r${r}.guidedLogits`);
      assertBitExactFiniteArray(sEp.representativesAfter, sHf.representativesAfter, `edit.d5.r${r}.reps`);
    }

    // Round 0 & final round guided logits + 1-step Dual-CFG mixer checks for independent & ip2p
    assertFiniteArrayParity(hf.steps[0].guidedLogits!, ref.round0GuidedLogits, 2e-5, 'edit.r0.legacyLogits');
    assertFiniteArrayParity(
      hf.steps[hf.steps.length - 1].guidedLogits!,
      ref.finalRoundGuidedLogits!,
      2e-5,
      'edit.rLast.legacyLogits',
    );

    const r0Branches = hf.steps[0].perBranchFullLogits!;
    const r0Indep = applyEditDualCfgLogits(r0Branches[0], r0Branches[1], r0Branches[2], 'independent', 1.6, 2.0);
    const r0Ip2p = applyEditDualCfgLogits(r0Branches[0], r0Branches[1], r0Branches[2], 'ip2p', 1.6, 2.0);
    assertFiniteArrayParity(r0Indep, ref.round0DualCfgAltModes!.independent, 2e-5, 'edit.r0.independentLogits');
    assertFiniteArrayParity(r0Ip2p, ref.round0DualCfgAltModes!.ip2p, 2e-5, 'edit.r0.ip2pLogits');

    // Final codes, padded 16x16 decode, and cropped 16x8 output
    assertBitExactFiniteArray(hf.finalMarkedCodes, ref.finalMarkedCodes, 'edit.finalMarkedCodes');
    assertBitExactFiniteArray(hf.finalContentCodes, ref.finalContentCodes, 'edit.finalContentCodes');
    assertFiniteArrayParity(hf.decodedPaddedImage.data, ref.decodedPaddedImage, 2e-5, 'edit.decodedPaddedImage');
    assertFiniteArrayParity(hf.outputImage.data, ref.outputImage, 2e-5, 'edit.outputImage');
    assertBitExactFiniteArray(pruned.outputImage.data, hf.outputImage.data, 'edit.d5.outputImage');
  });
});

// ============================================================================
// CATEGORY 4: READ-ONLY OPD REFERENCE PARITY (`U24`, `§5.3`)
// ============================================================================

describe('UNI3 Gate G3 — Category 4: Read-Only OPD Reference Parity (tail_kl & matched_sigmas)', () => {
  it('matches PyTorch 2.15.0 129-bucket tail_kl, matched_sigmas(0.9) and flow editing CE (<= 2e-5)', () => {
    const opdRef = FIXTURE.opdRef;
    const demo = createDefaultOpdDemoInputs(opdRef.seed);

    // Deterministic demo input reproduction parity
    assertFiniteArrayParity(demo.teacherArProbs, opdRef.inputs.teacherArProbs, 1e-6, 'opd.teacherArProbs');
    assertFiniteArrayParity(demo.studentArProbs, opdRef.inputs.studentArProbs, 1e-6, 'opd.studentArProbs');
    assertFiniteArrayParity(demo.studentDiffProbs, opdRef.inputs.studentDiffProbs, 1e-6, 'opd.studentDiffProbs');
    assertFiniteArrayParity(demo.x0, opdRef.inputs.x0, 1e-6, 'opd.x0');
    assertFiniteArrayParity(demo.noiseDelta, opdRef.inputs.noiseDelta, 1e-6, 'opd.noiseDelta');
    expect(demo.targetCode).toBe(opdRef.inputs.targetCode);
    expect(demo.targetGroup).toBe(opdRef.inputs.targetGroup);

    // Understanding OPD: 129-bucket tail_kl (reverseKl, forwardKl, arOpdLoss, selfConsistencyLoss)
    const und = evaluateUnderstandingOpdRef(
      Float32Array.from(opdRef.inputs.teacherArProbs),
      Float32Array.from(opdRef.inputs.studentArProbs),
      Float32Array.from(opdRef.inputs.studentDiffProbs),
      { responseLen: opdRef.inputs.responseLen },
    );
    expect(und.readOnly).toBe(true);
    expect(und.hasGradients).toBe(false);
    expect(und.arTailKl.bucketCount).toBe(129);
    expect(und.consistencyTailKl.bucketCount).toBe(129);

    assertBitExactFiniteArray(und.arTailKl.unionIndices, opdRef.understanding.arTailKl.unionIndices, 'opd.ar.union');
    assertFiniteArrayParity(und.arTailKl.pTargetBuckets, opdRef.understanding.arTailKl.pTargetBuckets, 2e-5, 'opd.ar.pT');
    assertFiniteArrayParity(und.arTailKl.pStudentBuckets, opdRef.understanding.arTailKl.pStudentBuckets, 2e-5, 'opd.ar.pS');
    assertBitExactFiniteArray(
      und.consistencyTailKl.unionIndices,
      opdRef.understanding.consistencyTailKl.unionIndices,
      'opd.cons.union',
    );
    assertFiniteArrayParity(
      und.consistencyTailKl.pTargetBuckets,
      opdRef.understanding.consistencyTailKl.pTargetBuckets,
      2e-5,
      'opd.cons.pT',
    );
    assertFiniteArrayParity(
      und.consistencyTailKl.pStudentBuckets,
      opdRef.understanding.consistencyTailKl.pStudentBuckets,
      2e-5,
      'opd.cons.pS',
    );

    expect(Math.abs(und.arTailKl.reverseKl - opdRef.understanding.arTailKl.reverseKl)).toBeLessThanOrEqual(2e-5);
    expect(Math.abs(und.arTailKl.forwardKl - opdRef.understanding.arTailKl.forwardKl)).toBeLessThanOrEqual(2e-5);
    expect(Math.abs(und.arOpdLoss - opdRef.understanding.arOpdLoss)).toBeLessThanOrEqual(2e-5);
    expect(Math.abs(und.selfConsistencyLoss - opdRef.understanding.selfConsistencyLoss)).toBeLessThanOrEqual(2e-5);
    expect(Math.abs(und.totalUnderstandingOpdLoss - opdRef.understanding.totalUnderstandingOpdLoss)).toBeLessThanOrEqual(
      2e-5,
    );

    // Continuous-Flow Editing OPD: matched_sigmas(0.9, 1024, 40) + x^r + exact_ce + 0.2 * grouped_ce
    const sigmas = computeMatchedSigmas(0.9, 1024, 40, 0.02);
    assertFiniteArrayParity(sigmas.sigmas, opdRef.flowEditing.matchedSigmas.sigmas, 2e-5, 'opd.matchedSigmas');
    expect(Math.abs(sigmas.mu - opdRef.flowEditing.matchedSigmas.mu)).toBeLessThanOrEqual(1e-12);

    const flow = evaluateFlowEditingOpdRef(
      Float32Array.from(opdRef.inputs.x0),
      Float32Array.from(opdRef.inputs.noiseDelta),
      Float32Array.from(opdRef.inputs.exactLogits),
      opdRef.inputs.targetCode,
      Float32Array.from(opdRef.inputs.groupLogits),
      opdRef.inputs.targetGroup,
      { r: 0.9, seqLen: 1024, steps: 40, sigmaEnd: 0.02, groupedCeWeight: 0.2 },
    );
    expect(flow.readOnly).toBe(true);
    expect(flow.hasGradients).toBe(false);
    assertFiniteArrayParity(flow.xNoisyR, opdRef.flowEditing.xNoisyR, 2e-5, 'opd.xNoisyR');
    expect(Math.abs(flow.exactCe - opdRef.flowEditing.exactCe)).toBeLessThanOrEqual(2e-5);
    expect(Math.abs(flow.groupedCe - opdRef.flowEditing.groupedCe)).toBeLessThanOrEqual(2e-5);
    expect(Math.abs(flow.totalEditingOpdLoss - opdRef.flowEditing.totalEditingOpdLoss)).toBeLessThanOrEqual(2e-5);
  });
});

// ============================================================================
// SUITE 5: SAMPLER GUARDS, TOP-K/TOP-P, U22 SAFETY & CLEAN-ROOM ISOLATION
// ============================================================================

describe('UNI3 Gate G3 — Sampler Guards, U22 Extreme-Logit Safety & Isolation', () => {
  it('enforces topK/topP filtering, fixed 16*256 RNG draw count, and U22 overflow-safe log-domain Gumbel', () => {
    const row = new Float32Array([1.0, 5.0, 3.0, 2.0, 4.0, 0.0]);
    const top2 = applyTopKTopPFilter(row, 2, 1.0);
    expect(Number.isFinite(top2[1])).toBe(true);
    expect(Number.isFinite(top2[4])).toBe(true);
    expect(top2[0]).toBe(-Infinity);
    expect(top2[2]).toBe(-Infinity);

    const topP = applyTopKTopPFilter(row, 0, 0.75);
    expect(Number.isFinite(topP[1])).toBe(true);
    expect(Number.isFinite(topP[4])).toBe(true);

    // U22 extreme-logit safety (> 800 would overflow exp(logits) in naive Gumbel)
    const extreme = new Float32Array(24 * 256);
    extreme.fill(-1000.0);
    const isMarker = new Uint8Array(24);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const g = r * 6 + c;
        extreme[g * 256 + ((g * 7) % 256)] = 950.0;
      }
      isMarker[r * 6 + 4] = 1;
      isMarker[r * 6 + 5] = 1;
    }

    const rng = createXoshiro128ss(12345);
    let drawCount = 0;
    const proposals = sampleVqProposals(
      extreme,
      24,
      256,
      0.8,
      16,
      0.95,
      () => {
        drawCount++;
        return rng.nextUniformOpenUnit();
      },
      isMarker,
    );
    expect(drawCount).toBe(16 * 256);
    for (let r = 0; r < 4; r++) {
      expect(proposals[r * 6 + 4]).toBe(0);
      expect(proposals[r * 6 + 5]).toBe(0);
      for (let c = 0; c < 4; c++) {
        const g = r * 6 + c;
        expect(proposals[g]).toBe((g * 7) % 256);
      }
    }

    // T2I CFG linear check
    const u = new Float32Array([1.0, -2.0]);
    const c = new Float32Array([3.0, 0.5]);
    const guided = applyT2iCfgLogits(u, c, 3.0);
    expect(guided[0]).toBeCloseTo(9.0, 6);
    expect(guided[1]).toBeCloseTo(8.0, 6);
  });

  it('maintains strict clean-room isolation (zero Math.random, zero sibling/controls imports)', () => {
    const src = readFileSync(resolve(process.cwd(), 'src/uni/imageSampler.ts'), 'utf-8');
    const codeOnly = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
    expect(codeOnly).not.toMatch(/\bMath\.random\b/);
    expect(codeOnly).not.toMatch(/src\/twotower|twotower\//);
    expect(codeOnly).not.toMatch(/src\/diffusion|diffusion\//);
    expect(codeOnly).not.toMatch(/ui\/controls/);
  });
});
