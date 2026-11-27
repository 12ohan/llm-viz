/**
 * Independent PyTorch 2.15.0 Backbone Math Parity & Invariant Suite for
 * NVIDIA Nemotron-Labs-Dory1.0-4B (`tests/dory-math-parity.test.ts`, DORY1 Slice 1C).
 *
 * Verifies every Gate 1 & DORY0 §13 verification contract against the clean-room
 * PyTorch 2.15.0 fixture `tests/fixtures/pytorch-dory-backbone.json`:
 *   1. Cryptographic SHA-256 weight match (507 tensors / 93,248 floats = 372,992 B),
 *      `DORY-MINI-F32-V1\n` framed `modelWeightId`, spec/ledger completeness (`DR1–DR24`,
 *      `D1–D10`), 0 `Math.random()`, and `D4` architectural boundary isolation.
 *   2. Single-op Full-nGPT kernels (`<= 1e-6`): L2Norm (all axes + zero/near-zero + N(2x)),
 *      all 5 scale families (incl. planted `|alpha_hat| > 1`, custom `g_0`, signed `suv`),
 *      channel-wise NLERP (Anti-SLERP `DR7`), Dual-RoPE (`theta_S=16` vs `theta_F=1024`),
 *      and local anchor-map Jacobian (`Finding 9`).
 *   3. Single sublayers (`<= 1e-5`): SWA (`p=0`), Full (`p=6`), Recurrent SWA post-eviction
 *      with planted `> 1` alpha (`p=16`), and SwiGLU MLPs (`p=1`, `p=17`).
 *   4. Full 72-sublayer forward (`<= 2e-5`) for `k in {1, 2, 3, 4, 6}` + shared physical
 *      parameter identity + distinct per-loop activations (`s_1 != s_2 != s_4`).
 *   5. `per_loop` chunked/incremental/eviction parity & `DR24` long-chunk transient workspace
 *      (`L = 24, 32 > cap_swa = 16`, `<= 2e-5`) + `DR18` guard & `cap_swa=16` rewind.
 *   6. `last_loop` Theorems 1–3 (`DR17`) + constructive `max_abs > 1e-2` partition divergence
 *      fixture (TS vs PyTorch `<= 2e-5` on each partition) + `W_v = 0` counterexample.
 *   7. Layer A/C upstream differential checks (`DR5/DR6/DR8/DR9`): vLLM FP32 vs transformers
 *      BF16, RoPE<->L2Norm order, unit vs non-unit Layer-0 pre-norm, and literal `N(2x)`.
 */

import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  assertBitExactFiniteArray,
  assertCausalScoresParity,
  assertFiniteArrayParity,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';
import {
  DORY_BACKBONE_SEED,
  DORY_DESIGN_DECISIONS,
  DORY_DISCREPANCY_LEDGER,
  DORY_MINI_CONFIG,
  DORY_MINI_FLOAT_COUNT,
  DORY_MINI_ROPE_SLIDING_INV_FREQ,
  DORY_MINI_TENSOR_COUNT,
  DORY_MINI_WEIGHT_BYTES,
  DORY_NUMERICAL_TOLERANCES,
  DORY_PLANTED_EXTRAPOLATION_SPEC,
  DORY_WEIGHT_DIGEST_MAGIC,
  anchorMapJacobianNorm,
  applyNeoXRotateHalf,
  applyNeoXRotateHalfAtPositions,
  buildDoryDualRopeTables,
  cloneDoryKvCache,
  computeDoryWeightDigestHex,
  computeExecutedDepth,
  countDoryParameterStats,
  createDoryKvCache,
  effectiveLogitScale,
  effectiveResidualAlpha,
  effectiveSGateSquared,
  effectiveSqkSquared,
  effectiveSuvScale,
  getDefaultDoryModelParameters,
  getDoryInputBank,
  getDoryRecurrentBank,
  l2NormHeadsInto,
  l2NormVector,
  listDoryCanonicalTensors,
  loopEntryAnchorInject,
  loopExitNormalize,
  ngptNlerpResidualRows,
  ngptNlerpResidualRowsDetailed,
  rewindDoryKvCache,
  runDoryAttentionSublayer,
  runDoryForwardChunk,
  runDoryMlpSublayer,
  type DoryAttentionSublayerWeights,
  type DoryKvCacheMode,
  type DoryMlpSublayerWeights,
  type DoryModelParameters,
  type DorySublayerWeights,
} from '../src/dory';

// ============================================================================
// FIXTURE SCHEMA TYPES & LOADING
// ============================================================================

interface NlerpFixtureBlock {
  aNorm: number[];
  bNorm: number[];
  delta: number[];
  effectiveAlpha: number[];
  blendedPreNorm: number[];
  output: number[];
}

interface AttnSublayerFixture {
  physicalLayerIdx: number;
  attentionKind: 'sliding_attention' | 'full_attention';
  seqLen: number;
  startPos: number;
  pastLen: number;
  kvLen: number;
  kvAbsPositions: number[];
  kvWriterLoops: number[];
  qPre: number[];
  kPre: number[];
  vNew: number[];
  gPre: number[];
  qRot: number[];
  kRot: number[];
  qNorm: number[];
  kCache: number[];
  effectiveSqkSq: number[];
  qScaled: number[];
  maskVisible: number[];
  scores: (number | null)[];
  probs: number[];
  attnContext: number[];
  effectiveSGateSq: number[];
  gateValues: number[];
  gatedContext: number[];
  attnSubNorm: number[];
  oProjOut: number[];
  nlerp: NlerpFixtureBlock;
  output: number[];
}

interface MlpSublayerFixture {
  physicalLayerIdx: number;
  seqLen: number;
  gatePre: number[];
  upPre: number[];
  effectiveSuvGate: number[];
  effectiveSuvUp: number[];
  gateScaled: number[];
  upScaled: number[];
  gateSilu: number[];
  swigluProduct: number[];
  ffnSubNorm: number[];
  downProjOut: number[];
  nlerp: NlerpFixtureBlock;
  output: number[];
}

interface PartitionOutputFixture {
  codaHidden: number[];
  scaledLogits: number[];
}

interface LastLoopKFixture {
  kLoops: number;
  theorem1SingleShotVsPerLoopMaxAbs: number;
  chunk4_8Prefix0_4VsPerLoopMaxAbs: number;
  chunk4_8Suffix4_12VsPerLoopMaxAbs: number;
  chunk4_8VsOneByOnePos1_12MaxAbs: number;
  singleShotVsOneByOnePos1_12MaxAbs: number;
  partitions: {
    perLoopSingleShot: PartitionOutputFixture;
    lastLoopSingleShot: PartitionOutputFixture;
    lastLoopChunk4_8: PartitionOutputFixture;
    lastLoopOneByOne: PartitionOutputFixture;
  };
}

interface DoryBackboneFixture {
  schemaVersion: string;
  format: string;
  modelWeightId: string;
  weightsSha256: string;
  backboneSeed: number;
  tensorCount: number;
  totalFloats: number;
  totalBytes: number;
  torchVersion: string;
  pythonVersion: string;
  spotTensorSha256: Record<string, string>;
  singleOps: {
    l2Norm: {
      zeroIn: number[];
      zeroOut: number[];
      nearZeroIn: number[];
      nearZeroOut: number[];
      stdHiddenIn: number[];
      stdHiddenOut: number[];
      attnSubNormIn: number[];
      attnSubNormOut: number[];
      ffnSubNormIn: number[];
      ffnSubNormOut: number[];
      unitX: number[];
      unitS: number[];
      nX: number[];
      n2X: number[];
      nSPlusX: number[];
      nAntipodal: number[];
      n2xVsNxMaxAbsDiff: number;
    };
    scaleTransforms: {
      p0AttnAlphaEff: number[];
      p16AttnAlphaEff: number[];
      p1MlpAlphaEff: number[];
      p0SqkSqEff: number[];
      p16SqkSqEff: number[];
      p0SGateSqDefaultEff: number[];
      p0SGateSqCustomG0Value: number;
      p0SGateSqCustomG0Eff: number[];
      p1SuvGateEff: number[];
      p1SuvUpEff: number[];
      lmLogitScaleEff: number[];
    };
    nlerp: {
      seqLen: number;
      residualRows: number[];
      sublayerRows: number[];
      p0: NlerpFixtureBlock;
      p16PlantedExtrapolation: NlerpFixtureBlock;
      p16NlerpVsTrigSlerpMaxAbsDiff: number;
    };
    dualRope: {
      slidingInvFreq: number[];
      slidingCos: number[];
      slidingSin: number[];
      fullInvFreq: number[];
      fullCos: number[];
      fullSin: number[];
      sampleQIn: number[];
      sampleKIn: number[];
      contigStartPos: number;
      nonContigPosIds: number[];
      qRotSlidingContig: number[];
      kRotSlidingContig: number[];
      qRotFullContig: number[];
      kRotFullContig: number[];
      qRotFullNonContig: number[];
    };
    anchorJacobian: {
      angleDegrees: number;
      angleRadians: number;
      epsStabilizedJacobianNorm: number;
      finiteDiffOrthogonalNorm: number;
      isLocalContractiveRegime: boolean;
    }[];
  };
  sublayers: {
    tokens12: number[];
    hidden12: number[];
    hidden6: number[];
    attnP0Swa: AttnSublayerFixture;
    attnP6Full: AttnSublayerFixture;
    attnP16RecurrentChunkAfterEviction: AttnSublayerFixture;
    mlpP1Prelude: MlpSublayerFixture;
    mlpP17Recurrent: MlpSublayerFixture;
  };
  full72L: {
    tokenIds: number[];
    byK: Record<
      string,
      {
        kLoops: number;
        executedSublayerCount: number;
        xAnchor: number[];
        loopEntryStates: number[][];
        loopExitStates: number[][];
        codaHidden: number[];
        rawLogits: number[];
        scaledLogits: number[];
        lastTokenProbs: number[];
      }
    >;
  };
  perLoopAndLongChunk: {
    tokens12: number[];
    chunkPartition: number[];
    s12ByK: Record<
      string,
      {
        kLoops: number;
        codaHidden: number[];
        scaledLogits: number[];
        pytorchChunkVsFullMaxAbsDiff: number;
        pytorchOneByOneVsFullMaxAbsDiff: number;
      }
    >;
    longChunkDr24: {
      kLoops: number;
      tokens28: number[];
      l24Plus4: {
        chunkSizes: number[];
        refChunkSizes: number[];
        codaHidden: number[];
        scaledLogits: number[];
        pytorchVsRefMaxAbsDiff: number;
      };
      tokens32: number[];
      l32SingleShot: {
        chunkSizes: number[];
        refChunkSizes: number[];
        codaHidden: number[];
        scaledLogits: number[];
        pytorchVsRefMaxAbsDiff: number;
      };
    };
  };
  lastLoopTheorems: {
    tokens12: number[];
    theorem2K1: {
      lastLoopSingleShotVsPerLoopMaxAbs: number;
      lastLoopChunk4_8VsPerLoopMaxAbs: number;
      lastLoopOneByOneVsPerLoopMaxAbs: number;
      scaledLogits: number[];
    };
    byK: Record<string, LastLoopKFixture>;
    degenerateZeroVProjCounterexample: {
      kLoops: number;
      coreNonIdentityMaxAbsDelta: number;
      lastLoopFullVsChunk4_8MaxAbs: number;
      lastLoopChunk4_8VsPerLoopChunk4_8MaxAbs: number;
    };
  };
  upstreamDifferential: {
    tokenIds: number[];
    dr5Layer0PreNorm: {
      unitEmbeddingsLayer0MaxAbsDiff: number;
      unitEmbeddingsFull72LMaxAbsDiff: number;
      nonUnit3xScaleFactor: number;
      nonUnitEmbeddingsLayer0MaxAbsDiff: number;
      nonUnitEmbeddingsFull72LMaxAbsDiff: number;
      canonicalUnitLayer0Out: number[];
      canonicalNonUnitLayer0Out: number[];
      preNormNonUnitLayer0Out: number[];
    };
    dr6RopeNormOrder: {
      p0SwaFp32MaxAbsDiff: number;
      p6FullFp32MaxAbsDiff: number;
      p0QueryBf16MaxAbsDiff: number;
    };
    dr8LiteralLoopBoundaries: {
      literalVsElidedU1MaxAbsDiff: number;
      literalVsElidedFull72LLogitsMaxAbsDiff: number;
      canonicalU1: number[];
    };
    dr9Fp32VsTransformersBf16: {
      p0AttnBf16VsFp32MaxAbsDiff: number;
      p1MlpBf16VsFp32MaxAbsDiff: number;
      full72LK2TransformersBf16VsVllmFp32MaxAbsDiff: number;
      full72LK2TransformersBf16VsVllmFp32MeanAbsDiff: number;
      vllmFp32CanonicalScaledLogits: number[];
    };
  };
}

const FIXTURE_PATH = resolve(process.cwd(), 'tests/fixtures/pytorch-dory-backbone.json');
const fixture: DoryBackboneFixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8'));
const params = getDefaultDoryModelParameters();
const ropeTables = buildDoryDualRopeTables(params.config);

// ============================================================================
// TEST HELPERS
// ============================================================================

function toF32(arr: readonly number[]): Float32Array {
  return Float32Array.from(arr);
}

function getAttnWeights(p: number): DoryAttentionSublayerWeights {
  const layer = params.layers[p];
  if (layer.kind !== 'attention') {
    throw new Error(`Expected attention weights at physical layer ${p}`);
  }
  return layer;
}

function getMlpWeights(p: number): DoryMlpSublayerWeights {
  const layer = params.layers[p];
  if (layer.kind !== 'mlp') {
    throw new Error(`Expected MLP weights at physical layer ${p}`);
  }
  return layer;
}

function runPartitionedTs(
  tokenIds: readonly number[],
  chunkSizes: readonly number[],
  kLoops: number,
  mode: DoryKvCacheMode,
  modelParams: DoryModelParameters = params,
) {
  const kvCache = createDoryKvCache({ mode, kLoops, config: modelParams.config });
  const d = modelParams.config.hiddenSize;
  const v = modelParams.config.vocabSize;
  const totalLen = tokenIds.length;

  const codaHidden = new Float32Array(totalLen * d);
  const rawLogits = new Float32Array(totalLen * v);
  const scaledLogits = new Float32Array(totalLen * v);

  let offset = 0;
  for (const sz of chunkSizes) {
    const chunk = tokenIds.slice(offset, offset + sz);
    const res = runDoryForwardChunk({
      tokens: chunk,
      params: modelParams,
      kvCache,
      kLoops,
      kvCacheMode: mode,
      ropeTables,
    });
    codaHidden.set(res.codaHidden, offset * d);
    rawLogits.set(res.rawLogits, offset * v);
    scaledLogits.set(res.scaledLogits, offset * v);
    offset += sz;
  }

  return { kvCache, codaHidden, rawLogits, scaledLogits };
}

function collectTsFilesRecursively(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectTsFilesRecursively(full));
    } else if (entry.isFile() && entry.name.endsWith('.ts')) {
      files.push(full);
    }
  }
  return files;
}

function assertNlerpMatches(
  actual: {
    aNorm: Float32Array;
    bNorm: Float32Array;
    delta: Float32Array;
    effectiveAlpha: Float32Array;
    blendedPreNorm: Float32Array;
    output: Float32Array;
  },
  expected: NlerpFixtureBlock,
  tol: number,
  label: string,
): void {
  assertFiniteArrayParity(actual.aNorm, expected.aNorm, tol, `${label}.aNorm`);
  assertFiniteArrayParity(actual.bNorm, expected.bNorm, tol, `${label}.bNorm`);
  assertFiniteArrayParity(actual.delta, expected.delta, tol, `${label}.delta`);
  assertFiniteArrayParity(actual.effectiveAlpha, expected.effectiveAlpha, tol, `${label}.effectiveAlpha`);
  assertFiniteArrayParity(actual.blendedPreNorm, expected.blendedPreNorm, tol, `${label}.blendedPreNorm`);
  assertFiniteArrayParity(actual.output, expected.output, tol, `${label}.output`);
}

function assertAttnSublayerTraceMatches(
  actualTrace: NonNullable<ReturnType<typeof runDoryAttentionSublayer>['trace']>,
  actualOutput: Float32Array,
  expected: AttnSublayerFixture,
  tol: number,
  label: string,
): void {
  expect(actualTrace.physicalLayerIdx).toBe(expected.physicalLayerIdx);
  expect(actualTrace.attentionKind).toBe(expected.attentionKind);
  expect(actualTrace.seqLen).toBe(expected.seqLen);
  expect(actualTrace.startPos).toBe(expected.startPos);
  expect(actualTrace.pastLen).toBe(expected.pastLen);
  expect(actualTrace.kvLen).toBe(expected.kvLen);
  expect(Array.from(actualTrace.kvAbsPositions)).toEqual(expected.kvAbsPositions);
  expect(Array.from(actualTrace.kvWriterLoops)).toEqual(expected.kvWriterLoops);
  expect(Array.from(actualTrace.maskVisible)).toEqual(expected.maskVisible);

  assertFiniteArrayParity(actualTrace.qPre, expected.qPre, tol, `${label}.qPre`);
  assertFiniteArrayParity(actualTrace.kPre, expected.kPre, tol, `${label}.kPre`);
  assertFiniteArrayParity(actualTrace.vNew, expected.vNew, tol, `${label}.vNew`);
  assertFiniteArrayParity(actualTrace.gPre, expected.gPre, tol, `${label}.gPre`);
  assertFiniteArrayParity(actualTrace.qRot, expected.qRot, tol, `${label}.qRot`);
  assertFiniteArrayParity(actualTrace.kRot, expected.kRot, tol, `${label}.kRot`);
  assertFiniteArrayParity(actualTrace.qNorm, expected.qNorm, tol, `${label}.qNorm`);
  assertFiniteArrayParity(actualTrace.kCache, expected.kCache, tol, `${label}.kCache`);
  assertFiniteArrayParity(actualTrace.effectiveSqkSq, expected.effectiveSqkSq, tol, `${label}.effectiveSqkSq`);
  assertFiniteArrayParity(actualTrace.qScaled, expected.qScaled, tol, `${label}.qScaled`);
  assertCausalScoresParity(actualTrace.scores, expected.scores, tol, `${label}.scores`);
  assertFiniteArrayParity(actualTrace.probs, expected.probs, tol, `${label}.probs`);
  assertFiniteArrayParity(actualTrace.attnContext, expected.attnContext, tol, `${label}.attnContext`);
  assertFiniteArrayParity(actualTrace.effectiveSGateSq, expected.effectiveSGateSq, tol, `${label}.effectiveSGateSq`);
  assertFiniteArrayParity(actualTrace.gateValues, expected.gateValues, tol, `${label}.gateValues`);
  assertFiniteArrayParity(actualTrace.gatedContext, expected.gatedContext, tol, `${label}.gatedContext`);
  assertFiniteArrayParity(actualTrace.attnSubNorm, expected.attnSubNorm, tol, `${label}.attnSubNorm`);
  assertFiniteArrayParity(actualTrace.oProjOut, expected.oProjOut, tol, `${label}.oProjOut`);
  assertNlerpMatches(actualTrace.nlerp, expected.nlerp, tol, `${label}.nlerp`);
  assertFiniteArrayParity(actualOutput, expected.output, tol, `${label}.output`);
}

function assertMlpSublayerTraceMatches(
  actualTrace: NonNullable<ReturnType<typeof runDoryMlpSublayer>['trace']>,
  actualOutput: Float32Array,
  expected: MlpSublayerFixture,
  tol: number,
  label: string,
): void {
  expect(actualTrace.physicalLayerIdx).toBe(expected.physicalLayerIdx);
  expect(actualTrace.seqLen).toBe(expected.seqLen);
  assertFiniteArrayParity(actualTrace.gatePre, expected.gatePre, tol, `${label}.gatePre`);
  assertFiniteArrayParity(actualTrace.upPre, expected.upPre, tol, `${label}.upPre`);
  assertFiniteArrayParity(actualTrace.effectiveSuvGate, expected.effectiveSuvGate, tol, `${label}.effectiveSuvGate`);
  assertFiniteArrayParity(actualTrace.effectiveSuvUp, expected.effectiveSuvUp, tol, `${label}.effectiveSuvUp`);
  assertFiniteArrayParity(actualTrace.gateScaled, expected.gateScaled, tol, `${label}.gateScaled`);
  assertFiniteArrayParity(actualTrace.upScaled, expected.upScaled, tol, `${label}.upScaled`);
  assertFiniteArrayParity(actualTrace.gateSilu, expected.gateSilu, tol, `${label}.gateSilu`);
  assertFiniteArrayParity(actualTrace.swigluProduct, expected.swigluProduct, tol, `${label}.swigluProduct`);
  assertFiniteArrayParity(actualTrace.ffnSubNorm, expected.ffnSubNorm, tol, `${label}.ffnSubNorm`);
  assertFiniteArrayParity(actualTrace.downProjOut, expected.downProjOut, tol, `${label}.downProjOut`);
  assertNlerpMatches(actualTrace.nlerp, expected.nlerp, tol, `${label}.nlerp`);
  assertFiniteArrayParity(actualOutput, expected.output, tol, `${label}.output`);
}

// ============================================================================
// TEST SUITE
// ============================================================================

describe('Dory1.0-4B Backbone — Independent PyTorch 2.15.0 Parity & DORY1 Gate 1', () => {
  it('cryptographically verifies all 507 tensors (93,248 floats = 372,992 B), pinned torch==2.15.0, ledger, and D4/zero-Math.random isolation', () => {
    expect(fixture.schemaVersion).toBe('dory1-backbone-pytorch-parity-v1');
    expect(fixture.format).toBe(DORY_WEIGHT_DIGEST_MAGIC);
    expect(fixture.torchVersion.startsWith('2.15.0')).toBe(true);
    expect(fixture.backboneSeed).toBe(DORY_BACKBONE_SEED);
    expect(fixture.tensorCount).toBe(DORY_MINI_TENSOR_COUNT);
    expect(fixture.totalFloats).toBe(DORY_MINI_FLOAT_COUNT);
    expect(fixture.totalBytes).toBe(DORY_MINI_WEIGHT_BYTES);

    // Singleton identity + stats verification
    expect(getDefaultDoryModelParameters()).toBe(params);
    expect(params.config).toBe(DORY_MINI_CONFIG);
    const stats = countDoryParameterStats(params);
    expect(stats.totalTensors).toBe(507);
    expect(stats.totalFloats).toBe(93_248);
    expect(stats.totalBytes).toBe(372_992);
    expect(stats.linearWeightTensors).toBe(288);
    expect(stats.ngptScaleTensors).toBe(217);
    expect(stats.bookendMatrixTensors).toBe(2);
    expect(stats.normWeightTensors).toBe(0);
    expect(stats.biasTensors).toBe(0);
    expect(stats.loopAdapterTensors).toBe(0);
    expect(stats.inputStageTensors).toBe(112);
    expect(stats.recurrentStageTensors).toBe(280);
    expect(stats.outputStageTensors).toBe(112);
    expect(stats.bookendTensors).toBe(3);

    // Cryptographic SHA-256 verification (both canonical framed modelWeightId AND raw .f32 SHA-256)
    const canonicalTensors = listDoryCanonicalTensors(params);
    expect(canonicalTensors.length).toBe(507);

    const rawBuf = Buffer.allocUnsafe(DORY_MINI_WEIGHT_BYTES);
    const nodeFramed = createHash('sha256');
    nodeFramed.update(Buffer.from(DORY_WEIGHT_DIGEST_MAGIC, 'utf-8'));

    let offsetFloats = 0;
    const shaByName = new Map<string, string>();
    for (const t of canonicalTensors) {
      const tBuf = Buffer.allocUnsafe(t.data.length * 4);
      for (let i = 0; i < t.data.length; i += 1) {
        const val = t.data[i];
        expect(Number.isFinite(val)).toBe(true);
        tBuf.writeFloatLE(val, i * 4);
      }
      tBuf.copy(rawBuf, offsetFloats * 4);
      offsetFloats += t.data.length;

      nodeFramed.update(Buffer.from(t.name, 'utf-8'));
      nodeFramed.update(Buffer.from([0x00]));
      const ndimBuf = Buffer.allocUnsafe(4);
      ndimBuf.writeUInt32LE(t.shape.length, 0);
      nodeFramed.update(ndimBuf);
      for (const dim of t.shape) {
        const dimBuf = Buffer.allocUnsafe(4);
        dimBuf.writeUInt32LE(dim, 0);
        nodeFramed.update(dimBuf);
      }
      nodeFramed.update(tBuf);
      shaByName.set(t.name, createHash('sha256').update(tBuf).digest('hex'));
    }

    expect(offsetFloats).toBe(DORY_MINI_FLOAT_COUNT);
    const actualRawSha = createHash('sha256').update(rawBuf).digest('hex');
    const actualFramedSha = nodeFramed.digest('hex');

    expect(actualRawSha).toBe(fixture.weightsSha256);
    expect(actualFramedSha).toBe(fixture.modelWeightId);
    expect(params.modelWeightId).toBe(fixture.modelWeightId);
    expect(computeDoryWeightDigestHex(params)).toBe(fixture.modelWeightId);

    for (const [tensorName, expectedSha] of Object.entries(fixture.spotTensorSha256)) {
      expect(shaByName.get(tensorName)).toBe(expectedSha);
    }

    // Discrepancy & Design Decision ledgers complete (DR1–DR24, D1–D10)
    expect(DORY_DISCREPANCY_LEDGER.length).toBe(24);
    expect(DORY_DESIGN_DECISIONS.length).toBe(10);

    // Static scan under src/dory/: 0 Math.random() and 0 Gate 4b/4c or sibling imports (D4)
    const doryRoot = resolve(process.cwd(), 'src/dory');
    const doryFiles = collectTsFilesRecursively(doryRoot);
    expect(doryFiles.length).toBeGreaterThanOrEqual(11);

    const forbiddenImportPatterns = [
      /from\s+['"][^'"]*(?:ui\/(?:controls|overlay|memoryChart)|scene\/(?:spine|sheetInspector)|model\/(?:spec|memory|state|rcPlayback)|math\/(?:gqaSim|mambaSim|moeSim|hybridFlow|incrementalDecode)|story\/walkthrough|src\/main|\.\.\/main)['"]/,
      /from\s+['"][^'"]*(?:twotower|diffusion|uni)(?:\/|['"])/,
    ];
    for (const filePath of doryFiles) {
      const src = readFileSync(filePath, 'utf-8');
      expect(src, `Math.random forbidden in ${filePath}`).not.toMatch(/Math\.random\s*\(/);
      for (const re of forbiddenImportPatterns) {
        expect(src, `D4 forbidden import in ${filePath}`).not.toMatch(re);
      }
    }
  });

  it('Category 1 — Single-op Full-nGPT kernels match PyTorch 2.15.0 within <= 1e-6 (L2Norm, 5 scale families, NLERP, Dual-RoPE, Anchor Jacobian)', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.singleOpMaxAbs; // 1e-6
    const s = fixture.singleOps;

    // 1A. Unit-sphere L2Norm (all axes + zero + near-zero + literal N(2x))
    const zeroOut = l2NormVector(toF32(s.l2Norm.zeroIn));
    assertBitExactFiniteArray(zeroOut, s.l2Norm.zeroOut, 'l2Norm.zero');
    assertFiniteArrayParity(l2NormVector(toF32(s.l2Norm.nearZeroIn)), s.l2Norm.nearZeroOut, tol, 'l2Norm.nearZero');
    assertFiniteArrayParity(l2NormVector(toF32(s.l2Norm.stdHiddenIn)), s.l2Norm.stdHiddenOut, tol, 'l2Norm.stdHidden');
    assertFiniteArrayParity(l2NormVector(toF32(s.l2Norm.attnSubNormIn)), s.l2Norm.attnSubNormOut, tol, 'l2Norm.attnSubNorm');
    assertFiniteArrayParity(l2NormVector(toF32(s.l2Norm.ffnSubNormIn)), s.l2Norm.ffnSubNormOut, tol, 'l2Norm.ffnSubNorm');

    // Head-level L2Norm matches vector L2Norm per 8D head
    const headsOut = new Float32Array(64);
    l2NormHeadsInto(toF32(s.l2Norm.attnSubNormIn), headsOut, 1, 8, 8);
    for (let h = 0; h < 8; h += 1) {
      const slice = toF32(s.l2Norm.attnSubNormIn.slice(h * 8, (h + 1) * 8));
      assertBitExactFiniteArray(headsOut.subarray(h * 8, (h + 1) * 8), l2NormVector(slice), `l2NormHeads.h${h}`);
    }

    // Literal loop boundary u_1 = N(2x) vs N(x) (DR8)
    const unitX = toF32(s.l2Norm.unitX);
    const unitS = toF32(s.l2Norm.unitS);
    const nX = l2NormVector(unitX);
    const n2X = loopEntryAnchorInject(unitX, unitX, 1, 8);
    const nSPlusX = loopEntryAnchorInject(unitS, unitX, 1, 8);
    const negX = Float32Array.from(unitX, (v) => -v);
    const nAntipodal = loopEntryAnchorInject(unitX, negX, 1, 8);

    assertFiniteArrayParity(nX, s.l2Norm.nX, tol, 'l2Norm.nX');
    assertFiniteArrayParity(n2X, s.l2Norm.n2X, tol, 'l2Norm.n2X');
    assertFiniteArrayParity(nSPlusX, s.l2Norm.nSPlusX, tol, 'l2Norm.nSPlusX');
    assertBitExactFiniteArray(nAntipodal, s.l2Norm.nAntipodal, 'l2Norm.antipodal');
    expect(computeStrictMaxAbsDiff(n2X, nX, 'DR8.n2x_vs_nx')).toBeGreaterThan(1e-8);
    assertFiniteArrayParity(loopExitNormalize(toF32(s.l2Norm.stdHiddenIn), 1, 8), s.l2Norm.stdHiddenOut, tol, 'loopExitNormalize');

    // 1B. All 5 learned nGPT scale families (DR3, DR7)
    const p0Attn = getAttnWeights(0);
    const p16Attn = getAttnWeights(16);
    const p1Mlp = getMlpWeights(1);

    const effP0Alpha = effectiveResidualAlpha(p0Attn.attnAlpha, params.config);
    const effP16Alpha = effectiveResidualAlpha(p16Attn.attnAlpha, params.config);
    const effP1Alpha = effectiveResidualAlpha(p1Mlp.mlpAlpha, params.config);
    assertFiniteArrayParity(effP0Alpha, s.scaleTransforms.p0AttnAlphaEff, tol, 'scale.p0Alpha');
    assertFiniteArrayParity(effP16Alpha, s.scaleTransforms.p16AttnAlphaEff, tol, 'scale.p16Alpha');
    assertFiniteArrayParity(effP1Alpha, s.scaleTransforms.p1MlpAlphaEff, tol, 'scale.p1MlpAlpha');

    // Planted unclamped > 1 extrapolation channels on p=16 ch 6, 7 (and raw ch 7 negative -> tests .abs())
    const [ch0, ch1] = DORY_PLANTED_EXTRAPOLATION_SPEC.channelIndices;
    expect(p16Attn.attnAlpha[ch1]).toBeLessThan(0);
    expect(effP16Alpha[ch0]).toBeGreaterThan(DORY_PLANTED_EXTRAPOLATION_SPEC.minEffectiveAlpha);
    expect(effP16Alpha[ch0]).toBeLessThan(DORY_PLANTED_EXTRAPOLATION_SPEC.maxEffectiveAlpha);
    expect(effP16Alpha[ch1]).toBeGreaterThan(DORY_PLANTED_EXTRAPOLATION_SPEC.minEffectiveAlpha);
    expect(effP16Alpha[ch1]).toBeLessThan(DORY_PLANTED_EXTRAPOLATION_SPEC.maxEffectiveAlpha);

    assertFiniteArrayParity(effectiveSqkSquared(p0Attn.sqk, params.config), s.scaleTransforms.p0SqkSqEff, tol, 'scale.p0Sqk');
    assertFiniteArrayParity(effectiveSqkSquared(p16Attn.sqk, params.config), s.scaleTransforms.p16SqkSqEff, tol, 'scale.p16Sqk');
    assertFiniteArrayParity(effectiveSGateSquared(p0Attn.sGate, params.config), s.scaleTransforms.p0SGateSqDefaultEff, tol, 'scale.p0SGateDef');
    // DR3: honor custom attentionOutputGateLogitScaleInit = 4.0
    const customG0Config = {
      ...params.config,
      attentionOutputGateLogitScaleInit: s.scaleTransforms.p0SGateSqCustomG0Value,
    };
    assertFiniteArrayParity(
      effectiveSGateSquared(p0Attn.sGate, customG0Config),
      s.scaleTransforms.p0SGateSqCustomG0Eff,
      tol,
      'scale.p0SGateCustomG0',
    );
    const effSuvGate = effectiveSuvScale(p1Mlp.suvGate, params.config);
    const effSuvUp = effectiveSuvScale(p1Mlp.suvUp, params.config);
    assertFiniteArrayParity(effSuvGate, s.scaleTransforms.p1SuvGateEff, tol, 'scale.p1SuvGate');
    assertFiniteArrayParity(effSuvUp, s.scaleTransforms.p1SuvUpEff, tol, 'scale.p1SuvUp');
    expect(Array.from(effSuvGate).some((v) => v < 0)).toBe(true);
    expect(Array.from(effSuvUp).some((v) => v < 0)).toBe(true);
    assertFiniteArrayParity(
      effectiveLogitScale(params.logitScale, params.config),
      s.scaleTransforms.lmLogitScaleEff,
      tol,
      'scale.lmLogitScale',
    );

    // 1C. Channel-wise nGPT NLERP (Anti-SLERP DR7)
    const resRows = toF32(s.nlerp.residualRows);
    const subRows = toF32(s.nlerp.sublayerRows);
    const nlerpP0 = ngptNlerpResidualRowsDetailed(resRows, subRows, p0Attn.attnAlpha, s.nlerp.seqLen, 8, params.config);
    const nlerpP0Fast = ngptNlerpResidualRows(resRows, subRows, p0Attn.attnAlpha, s.nlerp.seqLen, 8, params.config);
    assertBitExactFiniteArray(nlerpP0Fast, nlerpP0.output, 'nlerp.p0.fastVsDetailed');
    assertNlerpMatches(nlerpP0, s.nlerp.p0, tol, 'nlerp.p0');
    expect(nlerpP0.extrapolatedChannelCount).toBe(0);

    const nlerpP16 = ngptNlerpResidualRowsDetailed(resRows, subRows, p16Attn.attnAlpha, s.nlerp.seqLen, 8, params.config);
    const nlerpP16Fast = ngptNlerpResidualRows(resRows, subRows, p16Attn.attnAlpha, s.nlerp.seqLen, 8, params.config);
    assertBitExactFiniteArray(nlerpP16Fast, nlerpP16.output, 'nlerp.p16.fastVsDetailed');
    assertNlerpMatches(nlerpP16, s.nlerp.p16PlantedExtrapolation, tol, 'nlerp.p16');
    expect(nlerpP16.extrapolatedChannelCount).toBe(2);
    expect(nlerpP16.maxEffectiveAlpha).toBeGreaterThan(1.2);
    expect(s.nlerp.p16NlerpVsTrigSlerpMaxAbsDiff).toBeGreaterThan(1e-2);

    // 1D. Dual-RoPE (theta_S=16 dyadic vs theta_F=1024, NeoX rotate_half)
    assertBitExactFiniteArray(ropeTables.sliding.invFreq, DORY_MINI_ROPE_SLIDING_INV_FREQ, 'rope.sliding.dyadic');
    assertBitExactFiniteArray(ropeTables.sliding.invFreq, s.dualRope.slidingInvFreq, 'rope.sliding.invFreq');
    assertFiniteArrayParity(ropeTables.sliding.cos, s.dualRope.slidingCos, tol, 'rope.sliding.cos');
    assertFiniteArrayParity(ropeTables.sliding.sin, s.dualRope.slidingSin, tol, 'rope.sliding.sin');
    assertFiniteArrayParity(ropeTables.full.invFreq, s.dualRope.fullInvFreq, tol, 'rope.full.invFreq');
    assertFiniteArrayParity(ropeTables.full.cos, s.dualRope.fullCos, tol, 'rope.full.cos');
    assertFiniteArrayParity(ropeTables.full.sin, s.dualRope.fullSin, tol, 'rope.full.sin');

    const sampleQ = toF32(s.dualRope.sampleQIn);
    const sampleK = toF32(s.dualRope.sampleKIn);
    assertFiniteArrayParity(
      applyNeoXRotateHalf(sampleQ, 5, 8, 8, s.dualRope.contigStartPos, ropeTables.sliding),
      s.dualRope.qRotSlidingContig,
      tol,
      'rope.qRotSlidingContig',
    );
    assertFiniteArrayParity(
      applyNeoXRotateHalf(sampleK, 5, 2, 8, s.dualRope.contigStartPos, ropeTables.sliding),
      s.dualRope.kRotSlidingContig,
      tol,
      'rope.kRotSlidingContig',
    );
    assertFiniteArrayParity(
      applyNeoXRotateHalf(sampleQ, 5, 8, 8, s.dualRope.contigStartPos, ropeTables.full),
      s.dualRope.qRotFullContig,
      tol,
      'rope.qRotFullContig',
    );
    assertFiniteArrayParity(
      applyNeoXRotateHalf(sampleK, 5, 2, 8, s.dualRope.contigStartPos, ropeTables.full),
      s.dualRope.kRotFullContig,
      tol,
      'rope.kRotFullContig',
    );
    assertFiniteArrayParity(
      applyNeoXRotateHalfAtPositions(sampleQ, s.dualRope.nonContigPosIds, 8, 8, ropeTables.full),
      s.dualRope.qRotFullNonContig,
      tol,
      'rope.qRotFullNonContig',
    );

    // 1E. Anchor-map Jacobian sensitivity metrics (Finding 9)
    for (const check of s.anchorJacobian) {
      const m = anchorMapJacobianNorm(check.angleRadians, params.config.l2NormEps);
      expect(Math.abs(m.epsStabilizedJacobianNorm - check.epsStabilizedJacobianNorm)).toBeLessThanOrEqual(tol);
      expect(Math.abs(m.epsStabilizedJacobianNorm - check.finiteDiffOrthogonalNorm)).toBeLessThanOrEqual(1e-3);
      expect(m.isLocalContractiveRegime).toBe(check.isLocalContractiveRegime);
    }
  });

  it('Category 2 — Single sublayers (SWA p=0, Full p=6, Recurrent post-eviction p=16, SwiGLU p=1 & p=17) match PyTorch 2.15.0 within <= 1e-5', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.sublayerMaxAbs; // 1e-5
    const h12 = toF32(fixture.sublayers.hidden12);
    const h6 = toF32(fixture.sublayers.hidden6);

    // p=0 SWA (W=8, S=12 crossing window)
    const cacheP0Traced = createDoryKvCache({ mode: 'per_loop', kLoops: 1, config: params.config });
    const resP0Traced = runDoryAttentionSublayer({
      hiddenStates: h12,
      seqLen: 12,
      startPos: 0,
      physicalLayerIdx: 0,
      loopIteration: 0,
      weights: getAttnWeights(0),
      ropeTables,
      kvBank: getDoryInputBank(cacheP0Traced, 0),
      writeKv: true,
      config: params.config,
      captureTrace: true,
    });
    const cacheP0Fast = createDoryKvCache({ mode: 'per_loop', kLoops: 1, config: params.config });
    const resP0Fast = runDoryAttentionSublayer({
      hiddenStates: h12,
      seqLen: 12,
      startPos: 0,
      physicalLayerIdx: 0,
      loopIteration: 0,
      weights: getAttnWeights(0),
      ropeTables,
      kvBank: getDoryInputBank(cacheP0Fast, 0),
      writeKv: true,
      config: params.config,
      captureTrace: false,
    });
    assertBitExactFiniteArray(resP0Fast.output, resP0Traced.output, 'attn.p0.fastVsTraced');
    expect(resP0Traced.trace).not.toBeNull();
    assertAttnSublayerTraceMatches(resP0Traced.trace!, resP0Traced.output, fixture.sublayers.attnP0Swa, tol, 'attn.p0.swa');

    // p=6 Full Attention (theta_F=1024, S=12)
    const cacheP6 = createDoryKvCache({ mode: 'per_loop', kLoops: 1, config: params.config });
    const resP6 = runDoryAttentionSublayer({
      hiddenStates: h12,
      seqLen: 12,
      startPos: 0,
      physicalLayerIdx: 6,
      loopIteration: 0,
      weights: getAttnWeights(6),
      ropeTables,
      kvBank: getDoryInputBank(cacheP6, 6),
      writeKv: true,
      config: params.config,
      captureTrace: true,
    });
    expect(resP6.trace).not.toBeNull();
    assertAttnSublayerTraceMatches(resP6.trace!, resP6.output, fixture.sublayers.attnP6Full, tol, 'attn.p6.full');

    // p=16 Recurrent SWA (planted > 1 alpha) after past=9 SWA eviction -> chunk=3 at startPos=9
    const cacheP16 = createDoryKvCache({ mode: 'per_loop', kLoops: 2, config: params.config });
    const bankP16 = getDoryRecurrentBank(cacheP16, 16, 1);
    runDoryAttentionSublayer({
      hiddenStates: h12.subarray(0, 9 * 8),
      seqLen: 9,
      startPos: 0,
      physicalLayerIdx: 16,
      loopIteration: 1,
      weights: getAttnWeights(16),
      ropeTables,
      kvBank: bankP16,
      writeKv: true,
      config: params.config,
      captureTrace: false,
    });
    const resP16Chunk = runDoryAttentionSublayer({
      hiddenStates: h12.subarray(9 * 8, 12 * 8),
      seqLen: 3,
      startPos: 9,
      physicalLayerIdx: 16,
      loopIteration: 1,
      weights: getAttnWeights(16),
      ropeTables,
      kvBank: bankP16,
      writeKv: true,
      config: params.config,
      captureTrace: true,
    });
    expect(resP16Chunk.trace).not.toBeNull();
    expect(resP16Chunk.trace!.nlerp.extrapolatedChannelCount).toBe(2);
    assertAttnSublayerTraceMatches(
      resP16Chunk.trace!,
      resP16Chunk.output,
      fixture.sublayers.attnP16RecurrentChunkAfterEviction,
      tol,
      'attn.p16.postEviction',
    );

    // p=1 Prelude SwiGLU & p=17 Recurrent SwiGLU
    const resP1Traced = runDoryMlpSublayer({
      hiddenStates: h6,
      seqLen: 6,
      physicalLayerIdx: 1,
      loopIteration: 0,
      weights: getMlpWeights(1),
      config: params.config,
      captureTrace: true,
    });
    const resP1Fast = runDoryMlpSublayer({
      hiddenStates: h6,
      seqLen: 6,
      physicalLayerIdx: 1,
      loopIteration: 0,
      weights: getMlpWeights(1),
      config: params.config,
      captureTrace: false,
    });
    assertBitExactFiniteArray(resP1Fast.output, resP1Traced.output, 'mlp.p1.fastVsTraced');
    assertMlpSublayerTraceMatches(resP1Traced.trace!, resP1Traced.output, fixture.sublayers.mlpP1Prelude, tol, 'mlp.p1');

    const resP17 = runDoryMlpSublayer({
      hiddenStates: h6,
      seqLen: 6,
      physicalLayerIdx: 17,
      loopIteration: 1,
      weights: getMlpWeights(17),
      config: params.config,
      captureTrace: true,
    });
    assertMlpSublayerTraceMatches(resP17.trace!, resP17.output, fixture.sublayers.mlpP17Recurrent, tol, 'mlp.p17');
  });

  it('Category 3 — Full 72-sublayer forward matches PyTorch 2.15.0 within <= 2e-5 for k in {1, 2, 3, 4, 6} with shared weight identity & distinct per-loop states', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.fullForwardMaxAbs; // 2e-5
    const tokens = fixture.full72L.tokenIds;

    for (const k of [1, 2, 3, 4, 6] as const) {
      const ref = fixture.full72L.byK[String(k)];
      expect(ref).toBeDefined();
      const captureTrace = k === 2;
      const res = runDoryForwardChunk({
        tokens,
        params,
        kLoops: k,
        kvCacheMode: 'per_loop',
        ropeTables,
        captureTrace,
      });

      expect(res.executedSublayerCount).toBe(computeExecutedDepth(k));
      expect(res.executedSublayerCount).toBe(ref.executedSublayerCount);
      assertFiniteArrayParity(res.xAnchor, ref.xAnchor, tol, `full72.k${k}.xAnchor`);
      expect(res.loopEntryStates.length).toBe(k);
      expect(res.loopExitStates.length).toBe(k);

      for (let t = 0; t < k; t += 1) {
        assertFiniteArrayParity(res.loopEntryStates[t], ref.loopEntryStates[t], tol, `full72.k${k}.u${t + 1}`);
        assertFiniteArrayParity(res.loopExitStates[t], ref.loopExitStates[t], tol, `full72.k${k}.s${t + 1}`);
      }
      assertFiniteArrayParity(res.codaHidden, ref.codaHidden, tol, `full72.k${k}.codaHidden`);
      assertFiniteArrayParity(res.rawLogits, ref.rawLogits, tol, `full72.k${k}.rawLogits`);
      assertFiniteArrayParity(res.scaledLogits, ref.scaledLogits, tol, `full72.k${k}.scaledLogits`);
      assertFiniteArrayParity(res.lastTokenProbs, ref.lastTokenProbs, tol, `full72.k${k}.lastTokenProbs`);

      if (k === 2) {
        // Verify 0-ULP equivalence between captureTrace=true and captureTrace=false, plus 8-tuple coords
        const fastRes = runDoryForwardChunk({
          tokens,
          params,
          kLoops: 2,
          kvCacheMode: 'per_loop',
          ropeTables,
          captureTrace: false,
        });
        assertBitExactFiniteArray(fastRes.scaledLogits, res.scaledLogits, 'full72.k2.fastVsTraced');
        expect(res.trace).not.toBeNull();
        expect(res.trace!.executedSublayers.length).toBe(112);
        const recPass1 = res.trace!.executedSublayers.find(
          (r) => r.coord.physicalLayerIdx === 16 && r.coord.loopIteration === 1,
        );
        const recPass2 = res.trace!.executedSublayers.find(
          (r) => r.coord.physicalLayerIdx === 16 && r.coord.loopIteration === 2,
        );
        expect(recPass1).toBeDefined();
        expect(recPass2).toBeDefined();
        expect(recPass1!.coord.modelWeightId).toBe(params.modelWeightId);
        expect(recPass2!.coord.modelWeightId).toBe(params.modelWeightId);
        expect(recPass1!.coord.executedLayerIdx).toBe(16);
        expect(recPass2!.coord.executedLayerIdx).toBe(56);
        expect(recPass1!.coord.cacheOwnership.bankSlotId).not.toBe(
          recPass2!.coord.cacheOwnership.bankSlotId,
        );
      }

      if (k === 4) {
        // Distinct per-loop activations s_1 != s_2 != s_4 using identical physical weights
        expect(computeStrictMaxAbsDiff(res.loopExitStates[0], res.loopExitStates[1], 'k4.s1_vs_s2')).toBeGreaterThan(1e-2);
        expect(computeStrictMaxAbsDiff(res.loopExitStates[1], res.loopExitStates[3], 'k4.s2_vs_s4')).toBeGreaterThan(1e-2);
      }
    }
  });

  it('Category 4 — per_loop chunked/incremental/eviction parity & DR24 long-chunk workspace (L=24, 32 > cap_swa=16) match PyTorch 2.15.0 within <= 2e-5', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.fullForwardMaxAbs; // 2e-5
    const idTol = DORY_NUMERICAL_TOLERANCES.algebraicIdentityMaxAbs; // 1e-6
    const { tokens12, chunkPartition, s12ByK, longChunkDr24 } = fixture.perLoopAndLongChunk;

    for (const k of [2, 4] as const) {
      const ref = s12ByK[String(k)];
      const fullTs = runPartitionedTs(tokens12, [12], k, 'per_loop');
      const chunkTs = runPartitionedTs(tokens12, chunkPartition, k, 'per_loop');
      const incTs = runPartitionedTs(tokens12, Array(12).fill(1), k, 'per_loop');

      assertFiniteArrayParity(fullTs.codaHidden, ref.codaHidden, tol, `perLoop.k${k}.full.coda`);
      assertFiniteArrayParity(fullTs.scaledLogits, ref.scaledLogits, tol, `perLoop.k${k}.full.logits`);
      assertFiniteArrayParity(chunkTs.scaledLogits, ref.scaledLogits, tol, `perLoop.k${k}.chunk.logits`);
      assertFiniteArrayParity(incTs.scaledLogits, ref.scaledLogits, tol, `perLoop.k${k}.inc.logits`);

      assertFiniteArrayParity(chunkTs.scaledLogits, fullTs.scaledLogits, idTol, `perLoop.k${k}.chunkVsFull`);
      assertFiniteArrayParity(incTs.scaledLogits, fullTs.scaledLogits, idTol, `perLoop.k${k}.incVsFull`);
      expect(chunkTs.kvCache.chunkSizes).toEqual(chunkPartition);
      expect(incTs.kvCache.chunkSizes.length).toBe(12);
    }

    // DR24 Long-Chunk Workspace: L = 24 (> cap_swa = 16) + 4-token continuation = 28 tokens
    const l24p4 = runPartitionedTs(
      longChunkDr24.tokens28,
      longChunkDr24.l24Plus4.chunkSizes,
      longChunkDr24.kLoops,
      'per_loop',
    );
    const l24Ref = runPartitionedTs(
      longChunkDr24.tokens28,
      longChunkDr24.l24Plus4.refChunkSizes,
      longChunkDr24.kLoops,
      'per_loop',
    );
    assertFiniteArrayParity(l24p4.codaHidden, longChunkDr24.l24Plus4.codaHidden, tol, 'dr24.l24p4.coda');
    assertFiniteArrayParity(l24p4.scaledLogits, longChunkDr24.l24Plus4.scaledLogits, tol, 'dr24.l24p4.logits');
    assertFiniteArrayParity(l24p4.scaledLogits, l24Ref.scaledLogits, idTol, 'dr24.l24p4_vs_8x3_4');

    // Verify persistent SWA ring retained newest 16 slots [12..27] with exact absPosTag
    const swaBank28 = getDoryInputBank(l24p4.kvCache, 0);
    expect(swaBank28.logicalLen).toBe(28);
    for (let pos = 12; pos < 28; pos += 1) {
      expect(swaBank28.absPosTag[pos & 15]).toBe(pos);
    }

    // DR24 Long-Chunk Workspace: L = 32 (full S_max = 32 > cap_swa = 16) single-shot
    const l32Single = runPartitionedTs(
      longChunkDr24.tokens32,
      longChunkDr24.l32SingleShot.chunkSizes,
      longChunkDr24.kLoops,
      'per_loop',
    );
    const l32Split16 = runPartitionedTs(
      longChunkDr24.tokens32,
      longChunkDr24.l32SingleShot.refChunkSizes,
      longChunkDr24.kLoops,
      'per_loop',
    );
    assertFiniteArrayParity(l32Single.codaHidden, longChunkDr24.l32SingleShot.codaHidden, tol, 'dr24.l32.coda');
    assertFiniteArrayParity(l32Single.scaledLogits, longChunkDr24.l32SingleShot.scaledLogits, tol, 'dr24.l32.logits');
    assertFiniteArrayParity(l32Single.scaledLogits, l32Split16.scaledLogits, idTol, 'dr24.l32_vs_16_16');

    const swaBank32 = getDoryInputBank(l32Single.kvCache, 0);
    expect(swaBank32.logicalLen).toBe(32);
    for (let pos = 16; pos < 32; pos += 1) {
      expect(swaBank32.absPosTag[pos & 15]).toBe(pos);
    }

    // DR18 mid-session depth/policy guard + non-aliased CoW clone + transaction-local rewind
    const liveCache = createDoryKvCache({ mode: 'per_loop', kLoops: 2, config: params.config });
    runDoryForwardChunk({ tokens: tokens12.slice(0, 10), params, kvCache: liveCache, ropeTables });
    expect(() =>
      runDoryForwardChunk({ tokens: [30], params, kvCache: liveCache, kLoops: 4, ropeTables }),
    ).toThrow(/DR18/);
    expect(() =>
      runDoryForwardChunk({ tokens: [30], params, kvCache: liveCache, kvCacheMode: 'last_loop', ropeTables }),
    ).toThrow(/DR18/);

    const clonedCache = cloneDoryKvCache(liveCache);
    expect(getDoryInputBank(clonedCache, 0).k.buffer).not.toBe(getDoryInputBank(liveCache, 0).k.buffer);
    rewindDoryKvCache(clonedCache, 8);
    expect(clonedCache.committedLen).toBe(8);
    expect(liveCache.committedLen).toBe(10); // original strictly untouched
    expect(getDoryInputBank(clonedCache, 0).absPosTag[8]).toBe(-1);
    expect(getDoryInputBank(clonedCache, 0).absPosTag[9]).toBe(-1);
  });

  it('Category 5 — last_loop Theorems 1–3, constructive > 1e-2 divergence fixture (TS vs PyTorch <= 2e-5 per partition) & W_v = 0 counterexample', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.fullForwardMaxAbs; // 2e-5
    const idTol = DORY_NUMERICAL_TOLERANCES.algebraicIdentityMaxAbs; // 1e-6
    const divMin = DORY_NUMERICAL_TOLERANCES.lastLoopDivergenceMinAbs; // 1e-2
    const { tokens12, theorem2K1, byK, degenerateZeroVProjCounterexample } = fixture.lastLoopTheorems;
    const vocabSize = params.config.vocabSize;

    // Theorem 2: k = 1 universal equivalence (last_loop === per_loop on all partitions)
    const k1PlFull = runPartitionedTs(tokens12, [12], 1, 'per_loop');
    const k1LlFull = runPartitionedTs(tokens12, [12], 1, 'last_loop');
    const k1Ll48 = runPartitionedTs(tokens12, [4, 8], 1, 'last_loop');
    const k1Ll1x12 = runPartitionedTs(tokens12, Array(12).fill(1), 1, 'last_loop');
    assertFiniteArrayParity(k1LlFull.scaledLogits, k1PlFull.scaledLogits, idTol, 'th2.k1.full');
    assertFiniteArrayParity(k1Ll48.scaledLogits, k1PlFull.scaledLogits, idTol, 'th2.k1.chunk4_8');
    assertFiniteArrayParity(k1Ll1x12.scaledLogits, k1PlFull.scaledLogits, idTol, 'th2.k1.1by1');
    assertFiniteArrayParity(k1LlFull.scaledLogits, theorem2K1.scaledLogits, tol, 'th2.k1.vsPyTorch');

    // Theorems 1 & 3 for k = 2 and k = 4
    for (const k of [2, 4] as const) {
      const refK = byK[String(k)];
      const plFull = runPartitionedTs(tokens12, [12], k, 'per_loop');
      const llFull = runPartitionedTs(tokens12, [12], k, 'last_loop');
      const ll48 = runPartitionedTs(tokens12, [4, 8], k, 'last_loop');
      const ll1x12 = runPartitionedTs(tokens12, Array(12).fill(1), k, 'last_loop');

      // TS vs PyTorch <= 2e-5 on EACH partition independently
      assertFiniteArrayParity(plFull.codaHidden, refK.partitions.perLoopSingleShot.codaHidden, tol, `ll.k${k}.pl.coda`);
      assertFiniteArrayParity(plFull.scaledLogits, refK.partitions.perLoopSingleShot.scaledLogits, tol, `ll.k${k}.pl.logits`);
      assertFiniteArrayParity(llFull.codaHidden, refK.partitions.lastLoopSingleShot.codaHidden, tol, `ll.k${k}.ll12.coda`);
      assertFiniteArrayParity(llFull.scaledLogits, refK.partitions.lastLoopSingleShot.scaledLogits, tol, `ll.k${k}.ll12.logits`);
      assertFiniteArrayParity(ll48.codaHidden, refK.partitions.lastLoopChunk4_8.codaHidden, tol, `ll.k${k}.ll48.coda`);
      assertFiniteArrayParity(ll48.scaledLogits, refK.partitions.lastLoopChunk4_8.scaledLogits, tol, `ll.k${k}.ll48.logits`);
      assertFiniteArrayParity(ll1x12.codaHidden, refK.partitions.lastLoopOneByOne.codaHidden, tol, `ll.k${k}.ll1x12.coda`);
      assertFiniteArrayParity(ll1x12.scaledLogits, refK.partitions.lastLoopOneByOne.scaledLogits, tol, `ll.k${k}.ll1x12.logits`);

      // Theorem 1: empty-cache single-shot last_loop === per_loop (<= 1e-6)
      assertFiniteArrayParity(llFull.scaledLogits, plFull.scaledLogits, idTol, `th1.k${k}.singleShot`);

      // Theorem 3: prefix [0..4] matches per_loop <= 1e-6; suffix [4..12] diverges > 1e-2!
      assertFiniteArrayParity(
        ll48.scaledLogits.subarray(0, 4 * vocabSize),
        plFull.scaledLogits.subarray(0, 4 * vocabSize),
        idTol,
        `th3.k${k}.prefix0_4`,
      );
      const tsSuffixDiv = computeStrictMaxAbsDiff(
        ll48.scaledLogits.subarray(4 * vocabSize),
        plFull.scaledLogits.subarray(4 * vocabSize),
        `th3.k${k}.suffix4_12_vs_perLoop`,
      );
      const tsVs1by1Div = computeStrictMaxAbsDiff(
        ll48.scaledLogits.subarray(1 * vocabSize),
        ll1x12.scaledLogits.subarray(1 * vocabSize),
        `th3.k${k}.chunk4_8_vs_1by1`,
      );
      expect(tsSuffixDiv).toBeGreaterThan(divMin);
      expect(tsVs1by1Div).toBeGreaterThan(divMin);
      expect(refK.chunk4_8Suffix4_12VsPerLoopMaxAbs).toBeGreaterThan(divMin);
      expect(refK.chunk4_8VsOneByOnePos1_12MaxAbs).toBeGreaterThan(divMin);

      // Verify non-final loops never advanced ring pointers or overwrote writerLoopTag
      expect(ll48.kvCache.allBanks.length).toBe(36);
      const recBank = getDoryRecurrentBank(ll48.kvCache, 16, 1);
      expect(recBank.logicalLen).toBe(12);
      for (let pos = 0; pos < 12; pos += 1) {
        expect(recBank.absPosTag[pos & 15]).toBe(pos);
        expect(recBank.writerLoopTag[pos & 15]).toBe(k);
      }
    }

    // Finding 4: Degenerate W_v = 0 counterexample (non-identity F_theta, zero last_loop partition divergence)
    const zeroVLayers: DorySublayerWeights[] = params.layers.map((layer) => {
      if (layer.kind === 'attention' && layer.stage === 'recurrent') {
        return { ...layer, vProj: new Float32Array(layer.vProj.length) };
      }
      return layer;
    });
    const zeroVParams: DoryModelParameters = { ...params, layers: zeroVLayers };
    const zvCheck = runDoryForwardChunk({
      tokens: tokens12,
      params: zeroVParams,
      kLoops: 2,
      kvCacheMode: 'last_loop',
      ropeTables,
    });
    const tsCoreDelta = computeStrictMaxAbsDiff(zvCheck.loopExitStates[0], zvCheck.loopEntryStates[0], 'zeroV.coreDelta');
    expect(tsCoreDelta).toBeGreaterThan(0.05);
    expect(degenerateZeroVProjCounterexample.coreNonIdentityMaxAbsDelta).toBeGreaterThan(0.05);

    const zvFull = runPartitionedTs(tokens12, [12], 2, 'last_loop', zeroVParams);
    const zv48 = runPartitionedTs(tokens12, [4, 8], 2, 'last_loop', zeroVParams);
    const zvPl48 = runPartitionedTs(tokens12, [4, 8], 2, 'per_loop', zeroVParams);
    assertFiniteArrayParity(zvFull.scaledLogits, zv48.scaledLogits, idTol, 'zeroV.ll12_vs_ll48');
    assertBitExactFiniteArray(zv48.scaledLogits, zvPl48.scaledLogits, 'zeroV.ll48_vs_pl48');
    expect(degenerateZeroVProjCounterexample.lastLoopFullVsChunk4_8MaxAbs).toBeLessThanOrEqual(idTol);
    expect(degenerateZeroVProjCounterexample.lastLoopChunk4_8VsPerLoopChunk4_8MaxAbs).toBe(0);
  });

  it('Category 6 — Layer A/C upstream differential audit (DR5 Layer-0 pre-norm, DR6 RoPE<->L2Norm order, DR8 literal N(2x), DR9 vLLM FP32 vs transformers BF16)', () => {
    const diff = fixture.upstreamDifferential;
    const subTol = DORY_NUMERICAL_TOLERANCES.sublayerMaxAbs; // 1e-5
    const fullTol = DORY_NUMERICAL_TOLERANCES.fullForwardMaxAbs; // 2e-5
    const tokens = diff.tokenIds;

    // DR5: Unit embeddings -> Layer-0 pre-norm agrees <= 1e-6; non-unit 3x embeddings -> diverges > 1e-3!
    expect(diff.dr5Layer0PreNorm.unitEmbeddingsLayer0MaxAbsDiff).toBeLessThanOrEqual(1e-6);
    expect(diff.dr5Layer0PreNorm.nonUnitEmbeddingsLayer0MaxAbsDiff).toBeGreaterThan(1e-3);
    expect(diff.dr5Layer0PreNorm.nonUnitEmbeddingsFull72LMaxAbsDiff).toBeGreaterThan(1e-2);

    // Cross-check TS Layer 0 on unit & 3x non-unit embeddings (verifies NO Layer-0 pre-norm in TS!)
    const hUnit = new Float32Array(tokens.length * 8);
    const hNonUnit = new Float32Array(tokens.length * 8);
    for (let t = 0; t < tokens.length; t += 1) {
      const base = tokens[t] * 8;
      for (let c = 0; c < 8; c += 1) {
        const v = params.embeddings[base + c];
        hUnit[t * 8 + c] = v;
        hNonUnit[t * 8 + c] = v * diff.dr5Layer0PreNorm.nonUnit3xScaleFactor;
      }
    }
    const tsL0Unit = runDoryAttentionSublayer({
      hiddenStates: hUnit,
      seqLen: tokens.length,
      startPos: 0,
      physicalLayerIdx: 0,
      loopIteration: 0,
      weights: getAttnWeights(0),
      ropeTables,
      kvBank: getDoryInputBank(createDoryKvCache({ mode: 'per_loop', kLoops: 1 }), 0),
      writeKv: true,
      config: params.config,
    });
    const tsL0NonUnit = runDoryAttentionSublayer({
      hiddenStates: hNonUnit,
      seqLen: tokens.length,
      startPos: 0,
      physicalLayerIdx: 0,
      loopIteration: 0,
      weights: getAttnWeights(0),
      ropeTables,
      kvBank: getDoryInputBank(createDoryKvCache({ mode: 'per_loop', kLoops: 1 }), 0),
      writeKv: true,
      config: params.config,
    });
    assertFiniteArrayParity(tsL0Unit.output, diff.dr5Layer0PreNorm.canonicalUnitLayer0Out, subTol, 'dr5.tsUnitL0');
    assertFiniteArrayParity(tsL0NonUnit.output, diff.dr5Layer0PreNorm.canonicalNonUnitLayer0Out, subTol, 'dr5.tsNonUnitL0');
    expect(
      computeStrictMaxAbsDiff(tsL0NonUnit.output, diff.dr5Layer0PreNorm.preNormNonUnitLayer0Out, 'dr5.tsVsPreNormNonUnit'),
    ).toBeGreaterThan(1e-3);

    // DR6: FP32 RoPE <-> L2Norm orthogonal equivalence <= 1e-6; BF16 drift documented (> 1e-4)
    expect(diff.dr6RopeNormOrder.p0SwaFp32MaxAbsDiff).toBeLessThanOrEqual(1e-6);
    expect(diff.dr6RopeNormOrder.p6FullFp32MaxAbsDiff).toBeLessThanOrEqual(1e-6);
    expect(diff.dr6RopeNormOrder.p0QueryBf16MaxAbsDiff).toBeGreaterThan(1e-4);

    // DR8: Literal u_1 = N(2x) shift strictly > 0 and TS matches canonical u_1
    expect(diff.dr8LiteralLoopBoundaries.literalVsElidedU1MaxAbsDiff).toBeGreaterThan(1e-8);
    expect(diff.dr8LiteralLoopBoundaries.literalVsElidedU1MaxAbsDiff).toBeLessThan(1e-5);

    // DR9: vLLM FP32 vs transformers native BF16 drift (> 1e-3) while TS D5 matches PyTorch vLLM FP32 <= 2e-5
    expect(diff.dr9Fp32VsTransformersBf16.p0AttnBf16VsFp32MaxAbsDiff).toBeGreaterThan(1e-4);
    expect(diff.dr9Fp32VsTransformersBf16.p1MlpBf16VsFp32MaxAbsDiff).toBeGreaterThan(1e-4);
    expect(diff.dr9Fp32VsTransformersBf16.full72LK2TransformersBf16VsVllmFp32MaxAbsDiff).toBeGreaterThan(1e-3);

    const tsFullK2 = runDoryForwardChunk({
      tokens,
      params,
      kLoops: 2,
      kvCacheMode: 'per_loop',
      ropeTables,
    });
    assertFiniteArrayParity(tsFullK2.loopEntryStates[0], diff.dr8LiteralLoopBoundaries.canonicalU1, subTol, 'dr8.tsCanonU1');
    const tsVsVllmFp32Err = assertFiniteArrayParity(
      tsFullK2.scaledLogits,
      diff.dr9Fp32VsTransformersBf16.vllmFp32CanonicalScaledLogits,
      fullTol,
      'dr9.tsVsVllmFp32',
    );
    expect(tsVsVllmFp32Err).toBeLessThan(
      diff.dr9Fp32VsTransformersBf16.full72LK2TransformersBf16VsVllmFp32MaxAbsDiff / 100,
    );
  });
});
