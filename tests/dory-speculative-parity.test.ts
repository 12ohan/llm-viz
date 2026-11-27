/**
 * Contract A §6.2 Self-Speculative Decoding — 6-Layer Parity & Verification Suite
 * (`tests/dory-speculative-parity.test.ts`, DORY2 Slice 2C).
 *
 * Enforces ALL 6 speculative verification layers from `DORY0_ARCHITECTURE_AUDIT.md` §8.4 & §13
 * and `DORY0_REVIEW_RESPONSE.md` Findings 1–3 against the clean-room PyTorch 2.15.0 reference
 * fixture `tests/fixtures/pytorch-dory-spec.json`:
 *   - Provenance & Isolation: cryptographic SHA-256 weight match, pinned `torch==2.15.0`,
 *     Contract A `104`-bank badge (`DR23`), `0` `Math.random()`, and `D4` isolation on
 *     `src/dory/{speculative,session}.ts`.
 *   - Layer 1: Binary64 algebraic marginal identity `min(p_v, q_v) + max(p_v - q_v, 0) === p_v`
 *     and `enumerateSpecSingleStepExact` `maxAbsError <= 1e-15`.
 *   - Layer 2: Exact small-vocab (`V = 4`) 1-step & 2-step joint analytical enumeration
 *     `<= 1e-15` (zero Monte Carlo variance) matching independent Python binary64 reference.
 *   - Layer 3: Explicit-variate deterministic boundary tests (`0.0`, `p/q - 1e-7`, `p/q`,
 *     `p/q + 1e-7`, `1 - 1e-12`, unconditional `p >= q`, residual CDF bins, and
 *     `(temperature, topK, topP)` transform consistency).
 *   - Layer 4: Adversarial edge-case suite (`p === q`, disjoint support, partial support &
 *     `NON_POSITIVE_PROPOSAL_PROB` throw, sub-`1e-38` tail `q = 1e-40 / p = 5e-41`,
 *     near-identical `Delta = 4e-13` & `ZERO_RESIDUAL_MASS` throw, concentrated & one-hot).
 *   - Layer 5: Multi-token Contract A (`104` banks) execution & PyTorch 2.15.0 parity (`<= 2e-5`):
 *     * Reused `k = 4` verify vs. fresh uncached `k = 4` (`<= 2e-5`) across `S > W = 8` rollback,
 *     * Greedy (`T = 0`) token-for-token `===` standard `k = 4` AR,
 *     * Contract A multi-cycle shallow draft (`output_shallow`) vs. standalone `k = 1` Dory (`<= 2e-5`),
 *     * All 5 outcomes (`zero_accept_reject`, `partial_accept_reject`, `full_accept_bonus` w/
 *       1-token Shallow Coda catch-up, `full_accept_eos` w/o bonus, `final_slot_deep_decode`),
 *     * `DR23` guard & transaction-local `cap_swa = 16` `absPosTag` invalidation.
 *   - Layer 6: Pearson chi-squared (`N = 20,000, alpha = 0.001, df = 63, < 103.51`) PLUS
 *     2 mandatory failing negative controls (resampling from `p` instead of residual;
 *     buggy `1e-38` denominator clamp) + bonus `Z <= 1e-12` fallback negative control.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import {
  assertBitExactFiniteArray,
  assertFiniteArrayParity,
  computeStrictMaxAbsDiff,
} from './helpers/parityAssert';
import {
  DORY_BACKBONE_SEED,
  DORY_MINI_CONFIG,
  DORY_MINI_FLOAT_COUNT,
  DORY_MINI_TENSOR_COUNT,
  DORY_MINI_WEIGHT_BYTES,
  DORY_NUMERICAL_TOLERANCES,
  DORY_SPEC_CONTRACT_A_BADGE,
  DORY_SPEC_CONTRACT_A_BANK_COUNT,
  DORY_SPEC_DEFAULT_GAMMA,
  DORY_SPEC_DRAFT_LOOPS,
  DORY_SPEC_PEARSON_CHI2_DF63_ALPHA_001,
  DORY_SPEC_VERIFY_LOOPS,
  DORY_WEIGHT_DIGEST_MAGIC,
  DorySessionImmutableConfigError,
  DorySpecSupportInvariantError,
  argmaxIndex,
  buildDoryDualRopeTables,
  computeDoryWeightDigestHex,
  computePearsonChiSquared,
  computeRejectionResidualFloat64,
  createDoryKvCache,
  createDoryPrng,
  enumerateSpecSingleStepExact,
  enumerateSpecTwoStepExact,
  evaluateRejectionAcceptance,
  getDefaultDoryModelParameters,
  logitsToSamplingProbsFloat64,
  measureDoryKvCacheBytes,
  normalizeSamplingConfig,
  prefillDorySelfSpec,
  runDoryForwardChunk,
  runDorySelfSpecCycle,
  sampleCategoricalFloat64,
  sampleSpecSingleStep,
  summarizeSelfSpecCycle,
  type DoryKvCache,
  type DorySpecExplicitVariates,
  type DorySpecOutcomeKind,
} from '../src/dory';

// ============================================================================
// FIXTURE SCHEMA TYPES & LOADING
// ============================================================================

interface OneStepEnumFixture {
  name: string;
  targetProbs: number[];
  proposalProbs: number[];
  marginalProbs: number[];
  acceptanceRate: number;
  rejectionRate: number;
  residualMassZ: number;
  residualProbs: number[] | null;
  maxAbsError: number;
}

interface DraftStepFixture {
  draftOffset: number;
  absPos: number;
  inputTokenId: number;
  proposedTokenId: number;
  proposedIsEos: boolean;
  shallowScaledLogits: number[];
}

interface VerifyPosFixture {
  verifyOffset: number;
  absPos: number;
  inputTokenId: number;
  reusedPreludeAndLoop1: boolean;
  deepScaledLogits: number[];
  accepted: boolean | null;
  reason?: string | null;
}

interface CycleFixture {
  startCommittedLen: number;
  endCommittedLen: number;
  inputPendingToken: number;
  nextPendingToken: number | null;
  gammaEffective: number;
  draftCount: number;
  draftEndedWithEos: boolean;
  acceptedDraftCount: number;
  emittedTokens: number[];
  outcome: DorySpecOutcomeKind;
  haltReason: string | null;
  reusedActivationPositions: number;
  verifiedPositions: number;
  shallowCodaCatchupExecuted: boolean;
  rolledBackPositions: number;
  residualMassZ?: number | null;
  correctionOrBonusToken: number | null;
  draftSteps: DraftStepFixture[];
  verifyPositions: VerifyPosFixture[];
  reusedVsFreshK4MaxAbs: number;
  shallowDraftVsStandaloneK1MaxAbs: number;
  committedDeltaTokens: number[];
}

interface DorySpecFixture {
  schemaVersion: string;
  format: string;
  contractBadge: string;
  contractABankCount: number;
  modelWeightId: string;
  weightsSha256: string;
  backboneSeed: number;
  tensorCount: number;
  totalFloats: number;
  totalBytes: number;
  torchVersion: string;
  pythonVersion: string;
  layers1And2ExactEnum: {
    oneStepCases: OneStepEnumFixture[];
    twoStepV4: {
      vocabSize: number;
      targetP1: number[];
      proposalQ1: number[];
      targetP2ByToken1: number[][];
      proposalQ2ByToken1: number[][];
      jointProbs: number[];
      targetJointProbs: number[];
      maxAbsError: number;
    };
  };
  layers3And4BoundariesAndEdges: {
    boundaryTargetProbs: number[];
    boundaryProposalProbs: number[];
    boundaryResidualProbs: number[];
    boundaryResidualMassZ: number;
    boundaryChecks: {
      label: string;
      draftUniform: number;
      acceptUniform: number;
      residualUniform: number;
      expectedDraftToken: number;
      expectedAccepted: boolean;
      expectedReason: string;
      expectedEmittedToken: number;
    }[];
    filteredSamplingCases: {
      label: string;
      temperature: number;
      topK: number;
      topP: number;
      logits: number[];
      probs: number[];
    }[];
    edgeCases: {
      sub1e38: {
        targetProbs: number[];
        proposalProbs: number[];
        exactAcceptProbAtD0: number;
        buggyClamp1e38AcceptProbAtD0: number;
      };
      delta4e13: {
        targetProbs: number[];
        proposalProbs: number[];
        exactResidualProbs: number[];
        exactResidualMassZ: number;
        buggyFallback1e12ResidualProbs: number[];
      };
      concentrated: { targetProbs: number[]; proposalProbs: number[] };
      oneHotSame: { targetProbs: number[]; proposalProbs: number[] };
      oneHotDifferent: { targetProbs: number[]; proposalProbs: number[] };
    };
  };
  layer5ContractAExecution: {
    prompt8: number[];
    greedyMultiCycle: {
      prefill: {
        committedLen: number;
        pendingToken: number;
        lastDeepScaledLogits: number[];
        lastShallowScaledLogits: number[];
      };
      cycles: CycleFixture[];
      emittedSequence: number[];
      standaloneK4ArSequence: number[];
    };
    stochasticMultiCycle: {
      initialSampleUniform: number;
      prefill: {
        committedLen: number;
        pendingToken: number;
        lastDeepScaledLogits: number[];
        lastShallowScaledLogits: number[];
      };
      cycles: {
        explicitVariates: {
          draftSampleUniforms: number[];
          acceptUniforms: number[];
          correctionOrBonusUniform: number | null;
        };
        result: CycleFixture;
      }[];
    };
    draftEosEarlyStop: {
      initialSampleUniform: number;
      explicitVariates: {
        draftSampleUniforms: number[];
        acceptUniforms: number[];
        correctionOrBonusUniform: number | null;
      };
      result: CycleFixture;
    };
    finalSlotDeepDecode: {
      prompt31: number[];
      prefillPendingToken: number;
      result: CycleFixture;
    };
  };
  layer6Chi2AndNegativeControls: {
    sampleCount: number;
    alpha: number;
    degreesOfFreedom: number;
    criticalValueDf63Alpha001: number;
    modelPair: {
      prompt8: number[];
      k4ScaledLogits: number[];
      k1ScaledLogits: number[];
      targetProbs: number[];
      proposalProbs: number[];
      minExpectedCount: number;
      exactAcceptanceRate: number;
      exactMaxAbsError: number;
      negControl1ResampleFromPTvd: number;
    };
    highDivergencePair: {
      targetProbs: number[];
      proposalProbs: number[];
      minExpectedCount: number;
      exactAcceptanceRate: number;
      exactMaxAbsError: number;
      negControl1ResampleFromPMarginal: number[];
      negControl1ResampleFromPTvd: number;
      negControl1ResampleFromPMaxAbsErr: number;
      negControl1ExpectedChi2Noncentrality: number;
    };
    negControl2BuggyClamp1e38: {
      qDraft: number;
      pDraft: number;
      exactAcceptRate: number;
      buggyClampAcceptRate: number;
      criticalValueDf1Alpha001: number;
    };
  };
}

const FIXTURE_PATH = resolve(process.cwd(), 'tests/fixtures/pytorch-dory-spec.json');
const fixture: DorySpecFixture = JSON.parse(readFileSync(FIXTURE_PATH, 'utf-8'));
const params = getDefaultDoryModelParameters();
const ropeTables = buildDoryDualRopeTables(params.config);

// ============================================================================
// HELPER UTILITIES
// ============================================================================

function toF64(arr: readonly number[]): Float64Array {
  return Float64Array.from(arr);
}

function runFreshTsForward(tokenIds: readonly number[], kLoops: number): Float32Array {
  const kv = createDoryKvCache({
    mode: 'per_loop',
    kLoops,
    includeShallowCoda: false,
    config: params.config,
  });
  return runDoryForwardChunk({
    tokens: tokenIds,
    params,
    kvCache: kv,
    startPos: 0,
    kLoops,
    kvCacheMode: 'per_loop',
    ropeTables,
  }).scaledLogits;
}

function assertAll104BanksSyncedAndRolledBackClean(kvCache: DoryKvCache, expectedLen: number): void {
  expect(kvCache.allBanks.length).toBe(DORY_SPEC_CONTRACT_A_BANK_COUNT);
  expect(kvCache.committedLen).toBe(expectedLen);
  for (const bank of kvCache.allBanks) {
    expect(bank.logicalLen).toBe(expectedLen);
    // Verify no stale slot carries absPosTag >= expectedLen after transaction-local rewind (DR19)
    for (let slot = 0; slot < bank.capacity; slot += 1) {
      expect(bank.absPosTag[slot]).toBeLessThan(expectedLen);
    }
    if (bank.attentionKind === 'sliding_attention' && expectedLen > 0) {
      const reqStart = Math.max(0, expectedLen - (DORY_MINI_CONFIG.slidingWindow - 1));
      for (let pos = reqStart; pos < expectedLen; pos += 1) {
        const slot = pos & (bank.capacity - 1);
        expect(bank.absPosTag[slot]).toBe(pos);
      }
    }
  }
}

function toTsExplicitVariates(v: {
  draftSampleUniforms: number[];
  acceptUniforms: number[];
  correctionOrBonusUniform: number | null;
}): DorySpecExplicitVariates {
  return {
    draftSampleUniforms: v.draftSampleUniforms,
    acceptUniforms: v.acceptUniforms,
    correctionOrBonusUniform:
      v.correctionOrBonusUniform === null ? undefined : v.correctionOrBonusUniform,
  };
}

// ============================================================================
// GATE 2 — 6-LAYER SPECULATIVE VERIFICATION SUITE
// ============================================================================

describe('Dory1.0-4B Contract A §6.2 Self-Speculation — 6-Layer Parity & Gate 2 Suite', () => {
  it('verifies cryptographic PyTorch 2.15.0 fixture provenance, 104-bank Contract A metadata, and D4/zero-Math.random isolation', () => {
    expect(fixture.schemaVersion).toBe('dory2-spec-pytorch-parity-v1');
    expect(fixture.format).toBe(DORY_WEIGHT_DIGEST_MAGIC);
    expect(fixture.torchVersion.startsWith('2.15.0')).toBe(true);
    expect(fixture.contractBadge).toBe(DORY_SPEC_CONTRACT_A_BADGE);
    expect(fixture.contractABankCount).toBe(DORY_SPEC_CONTRACT_A_BANK_COUNT);
    expect(fixture.backboneSeed).toBe(DORY_BACKBONE_SEED);
    expect(fixture.tensorCount).toBe(DORY_MINI_TENSOR_COUNT);
    expect(fixture.totalFloats).toBe(DORY_MINI_FLOAT_COUNT);
    expect(fixture.totalBytes).toBe(DORY_MINI_WEIGHT_BYTES);
    expect(params.modelWeightId).toBe(fixture.modelWeightId);
    expect(computeDoryWeightDigestHex(params)).toBe(fixture.modelWeightId);

    // Contract A 104-bank payload sizing: 96 Deep (240 KiB) + 8 Shallow Coda (+20 KiB) = 260 KiB
    const specKv = createDoryKvCache({
      mode: 'per_loop',
      kLoops: DORY_SPEC_VERIFY_LOOPS,
      includeShallowCoda: true,
      config: DORY_MINI_CONFIG,
    });
    const deepOnlyKv = createDoryKvCache({
      mode: 'per_loop',
      kLoops: DORY_SPEC_VERIFY_LOOPS,
      includeShallowCoda: false,
      config: DORY_MINI_CONFIG,
    });
    const specBytes = measureDoryKvCacheBytes(specKv);
    const deepBytes = measureDoryKvCacheBytes(deepOnlyKv);
    expect(specBytes.bankCount).toBe(104);
    expect(deepBytes.bankCount).toBe(96);
    expect(specBytes.kvPayloadBytes).toBe(260 * 1024);
    expect(specBytes.kvPayloadBytes - deepBytes.kvPayloadBytes).toBe(20 * 1024);

    // Static D4 & zero-Math.random audit on DORY2 implementation files
    const forbiddenImportPatterns = [
      /from\s+['"][^'"]*(?:ui\/(?:controls|overlay|memoryChart)|scene\/(?:spine|sheetInspector)|model\/(?:spec|memory|state|rcPlayback)|math\/(?:gqaSim|mambaSim|moeSim|hybridFlow|incrementalDecode)|story\/walkthrough|src\/main|\.\.\/main)['"]/,
      /from\s+['"][^'"]*(?:twotower|diffusion|uni)(?:\/|['"])/,
    ];
    for (const relPath of ['src/dory/speculative.ts', 'src/dory/session.ts', 'src/dory/index.ts']) {
      const src = readFileSync(resolve(process.cwd(), relPath), 'utf-8');
      expect(src, `Math.random forbidden in ${relPath}`).not.toMatch(/Math\.random\s*\(/);
      for (const re of forbiddenImportPatterns) {
        expect(src, `D4 forbidden import in ${relPath}`).not.toMatch(re);
      }
    }
  });

  it('Layer 1 — Binary64 algebraic marginal identity <= 1e-15 across diverse & model-derived (p, q) pairs', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.binary64SpecEnumMaxAbs; // 1e-15

    // Verify pointwise identity min(p_v, q_v) + max(p_v - q_v, 0) === p_v AND exact 1-step marginal
    for (const c of fixture.layers1And2ExactEnum.oneStepCases) {
      const p = toF64(c.targetProbs);
      const q = toF64(c.proposalProbs);
      for (let v = 0; v < p.length; v += 1) {
        const rhs = Math.min(p[v], q[v]) + Math.max(p[v] - q[v], 0.0);
        expect(Math.abs(rhs - p[v]), `algebraic identity on ${c.name}[${v}]`).toBeLessThanOrEqual(
          tol,
        );
      }
      const res = enumerateSpecSingleStepExact(p, q);
      expect(res.maxAbsError, `enum1.maxAbsError on ${c.name}`).toBeLessThanOrEqual(tol);
      expect(Math.abs(res.acceptanceRate - c.acceptanceRate)).toBeLessThanOrEqual(tol);
      assertFiniteArrayParity(res.marginalProbs, c.marginalProbs, tol, `layer1.${c.name}.marginal`);
      assertFiniteArrayParity(res.marginalProbs, p, tol, `layer1.${c.name}.vsTarget`);

      if (c.residualProbs !== null) {
        const residual = computeRejectionResidualFloat64(p, q);
        expect(Math.abs(residual.massZ - c.residualMassZ)).toBeLessThanOrEqual(tol);
        assertFiniteArrayParity(residual.probs, c.residualProbs, tol, `layer1.${c.name}.residual`);
      }
    }

    // Also verify on actual V = 64 Dory k=4 vs k=1 model probabilities AND filtered top-k/top-p probs
    const modelP = toF64(fixture.layer6Chi2AndNegativeControls.modelPair.targetProbs);
    const modelQ = toF64(fixture.layer6Chi2AndNegativeControls.modelPair.proposalProbs);
    const modelEnum = enumerateSpecSingleStepExact(modelP, modelQ);
    expect(modelEnum.maxAbsError).toBeLessThanOrEqual(tol);
    assertFiniteArrayParity(modelEnum.marginalProbs, modelP, tol, 'layer1.modelV64.marginal');

    const filtP = logitsToSamplingProbsFloat64(
      fixture.layer6Chi2AndNegativeControls.modelPair.k4ScaledLogits,
      0,
      64,
      { temperature: 0.85, topK: 16, topP: 0.9 },
    );
    const filtQ = logitsToSamplingProbsFloat64(
      fixture.layer6Chi2AndNegativeControls.modelPair.k1ScaledLogits,
      0,
      64,
      { temperature: 0.85, topK: 16, topP: 0.9 },
    );
    const filtEnum = enumerateSpecSingleStepExact(filtP, filtQ);
    expect(filtEnum.maxAbsError).toBeLessThanOrEqual(tol);
  });

  it('Layer 2 — Exact V = 4 analytical 1-step & 2-step joint speculative enumeration <= 1e-15', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.binary64SpecEnumMaxAbs; // 1e-15
    const f2 = fixture.layers1And2ExactEnum.twoStepV4;
    expect(f2.vocabSize).toBe(4);
    expect(f2.maxAbsError).toBeLessThanOrEqual(tol);

    const p1 = toF64(f2.targetP1);
    const q1 = toF64(f2.proposalQ1);
    const p2Rows = f2.targetP2ByToken1.map((row) => toF64(row));
    const q2Rows = f2.proposalQ2ByToken1.map((row) => toF64(row));

    const tsTwoStep = enumerateSpecTwoStepExact(p1, q1, p2Rows, q2Rows);
    expect(tsTwoStep.vocabSize).toBe(4);
    expect(tsTwoStep.maxAbsError).toBeLessThanOrEqual(tol);
    assertFiniteArrayParity(tsTwoStep.jointProbs, tsTwoStep.targetJointProbs, tol, 'layer2.v4.jointVsTarget');
    assertFiniteArrayParity(tsTwoStep.jointProbs, f2.jointProbs, tol, 'layer2.v4.tsVsPythonJoint');
    assertFiniteArrayParity(
      tsTwoStep.targetJointProbs,
      f2.targetJointProbs,
      tol,
      'layer2.v4.tsVsPythonTarget',
    );
  });

  it('Layer 3 — Explicit-variate deterministic boundaries & (temperature, topK, topP) transform consistency', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.binary64SpecEnumMaxAbs; // 1e-15
    const f3 = fixture.layers3And4BoundariesAndEdges;
    const p = toF64(f3.boundaryTargetProbs);
    const q = toF64(f3.boundaryProposalProbs);

    const residual = computeRejectionResidualFloat64(p, q);
    expect(Math.abs(residual.massZ - f3.boundaryResidualMassZ)).toBeLessThanOrEqual(tol);
    assertFiniteArrayParity(residual.probs, f3.boundaryResidualProbs, tol, 'layer3.residual');

    for (const check of f3.boundaryChecks) {
      const step = sampleSpecSingleStep(p, q, undefined, {
        draftUniform: check.draftUniform,
        acceptUniform: check.acceptUniform,
        residualUniform: check.residualUniform,
      });
      expect(step.draftTokenId, `${check.label}.draftTokenId`).toBe(check.expectedDraftToken);
      expect(step.decision.accepted, `${check.label}.accepted`).toBe(check.expectedAccepted);
      expect(step.decision.reason, `${check.label}.reason`).toBe(check.expectedReason);
      expect(step.emittedTokenId, `${check.label}.emittedTokenId`).toBe(check.expectedEmittedToken);
      if (check.expectedReason === 'unconditional_p_ge_q') {
        expect(step.decision.uniformVariate).toBeNull();
      }
    }

    // Exact division-free boundary test at p[0]/q[0] = 0.15 / 0.50 = 0.30
    expect(evaluateRejectionAcceptance(0, p, q, 0.3 - 1e-12, false).accepted).toBe(true);
    expect(evaluateRejectionAcceptance(0, p, q, 0.3, false).accepted).toBe(false);
    expect(evaluateRejectionAcceptance(0, p, q, 0.3 + 1e-12, false).accepted).toBe(false);

    // Transform consistency for (temperature, topK, topP) matching Python binary64 reference <= 1e-15
    for (const fc of f3.filteredSamplingCases) {
      const cfg = normalizeSamplingConfig({
        temperature: fc.temperature,
        topK: fc.topK,
        topP: fc.topP,
      });
      const tsProbs = logitsToSamplingProbsFloat64(fc.logits, 0, fc.logits.length, cfg);
      assertFiniteArrayParity(tsProbs, fc.probs, tol, `layer3.filtered.${fc.label}`);
      // Verify exact normalization sum == 1 within 1e-15
      let sum = 0;
      for (let i = 0; i < tsProbs.length; i += 1) {
        sum += tsProbs[i];
      }
      expect(Math.abs(sum - 1.0)).toBeLessThanOrEqual(tol);
    }

    // Invalid uniform variate bounds throw RangeError
    expect(() => sampleCategoricalFloat64(p, -1e-9)).toThrow(RangeError);
    expect(() => sampleCategoricalFloat64(p, 1.0)).toThrow(RangeError);
    expect(() => sampleCategoricalFloat64(p, Number.NaN)).toThrow(RangeError);
  });

  it('Layer 4 — Adversarial edge suite (p===q, disjoint, partial support, q=1e-40, Delta=4e-13, concentrated, one-hot)', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.binary64SpecEnumMaxAbs; // 1e-15
    const edges = fixture.layers3And4BoundariesAndEdges.edgeCases;

    // 4A. Identical distributions (p === q): 0 rejections across 1,000 draws; residual throws ZERO_RESIDUAL_MASS
    const pEq = Float64Array.of(0.1, 0.2, 0.3, 0.4);
    const prngEq = createDoryPrng(0xd041a001);
    for (let i = 0; i < 1_000; i += 1) {
      const res = sampleSpecSingleStep(pEq, pEq, prngEq);
      expect(res.decision.accepted).toBe(true);
      expect(res.decision.reason).toBe('unconditional_p_ge_q');
      expect(res.decision.uniformVariate).toBeNull();
      expect(res.residual).toBeNull();
    }
    expect(() => computeRejectionResidualFloat64(pEq, pEq)).toThrow(DorySpecSupportInvariantError);
    try {
      computeRejectionResidualFloat64(pEq, pEq);
    } catch (err) {
      expect((err as DorySpecSupportInvariantError).code).toBe('ZERO_RESIDUAL_MASS');
    }

    // 4B. Disjoint support (p _|_ q): 100% rejection & exact p recovery
    const pDisj = Float64Array.of(0.0, 0.0, 0.65, 0.35);
    const qDisj = Float64Array.of(0.7, 0.3, 0.0, 0.0);
    const resDisj = computeRejectionResidualFloat64(pDisj, qDisj);
    expect(Math.abs(resDisj.massZ - 1.0)).toBeLessThanOrEqual(tol);
    assertFiniteArrayParity(resDisj.probs, pDisj, tol, 'layer4.disjoint.residual');
    const prngDisj = createDoryPrng(0xd041a002);
    for (let i = 0; i < 200; i += 1) {
      const draw = sampleSpecSingleStep(pDisj, qDisj, prngDisj);
      expect(draw.decision.accepted).toBe(false);
      expect(draw.emittedTokenId === 2 || draw.emittedTokenId === 3).toBe(true);
    }

    // 4C. Partial support (q[v] === 0 where p[v] > 0): never proposes zero-support token;
    //     direct call with q[d] === 0 MUST throw NON_POSITIVE_PROPOSAL_PROB (zero 1e-38 clamp!)
    const pPart = Float64Array.of(0.35, 0.25, 0.25, 0.15);
    const qPart = Float64Array.of(0.6, 0.4, 0.0, 0.0);
    expect(() => evaluateRejectionAcceptance(2, pPart, qPart, 0.5, false)).toThrow(
      DorySpecSupportInvariantError,
    );
    try {
      evaluateRejectionAcceptance(2, pPart, qPart, 0.5, false);
    } catch (err) {
      expect((err as DorySpecSupportInvariantError).code).toBe('NON_POSITIVE_PROPOSAL_PROB');
    }
    expect(enumerateSpecSingleStepExact(pPart, qPart).maxAbsError).toBeLessThanOrEqual(tol);

    // 4D. Sub-1e-38 tail: q[0] = 1e-40, p[0] = 5e-41 -> exact division-free 0.500 threshold (zero 1e-38 clamp!)
    const pSub = toF64(edges.sub1e38.targetProbs);
    const qSub = toF64(edges.sub1e38.proposalProbs);
    expect(evaluateRejectionAcceptance(0, pSub, qSub, 0.0, false).accepted).toBe(true);
    expect(evaluateRejectionAcceptance(0, pSub, qSub, 0.25, false).accepted).toBe(true);
    expect(evaluateRejectionAcceptance(0, pSub, qSub, 0.499999999999, false).accepted).toBe(true);
    expect(evaluateRejectionAcceptance(0, pSub, qSub, 0.5, false).accepted).toBe(false);
    expect(evaluateRejectionAcceptance(0, pSub, qSub, 0.500000000001, false).accepted).toBe(false);
    expect(edges.sub1e38.buggyClamp1e38AcceptProbAtD0).toBeCloseTo(0.005, 12);

    // 4E. Near-identical Delta = 4e-13: exact residual [1.0, 0.0], Z = 4e-13 (zero 1e-12 fallback to p!)
    const pDelta = toF64(edges.delta4e13.targetProbs);
    const qDelta = toF64(edges.delta4e13.proposalProbs);
    const resDelta = computeRejectionResidualFloat64(pDelta, qDelta);
    expect(resDelta.massZ).toBe(edges.delta4e13.exactResidualMassZ);
    expect(Math.abs(resDelta.massZ - 4e-13)).toBeLessThanOrEqual(tol);
    assertFiniteArrayParity(resDelta.probs, edges.delta4e13.exactResidualProbs, tol, 'layer4.delta4e13');
    expect(resDelta.probs[0]).toBe(1.0);
    expect(resDelta.probs[1]).toBe(0.0);

    // 4F. Concentrated (p_0 = 1 - 1e-9) & exact one-hot distributions
    const pConc = toF64(edges.concentrated.targetProbs);
    const qConc = toF64(edges.concentrated.proposalProbs);
    expect(enumerateSpecSingleStepExact(pConc, qConc).maxAbsError).toBeLessThanOrEqual(tol);

    const pOneSame = toF64(edges.oneHotSame.targetProbs);
    const qOneSame = toF64(edges.oneHotSame.proposalProbs);
    const drawOneSame = sampleSpecSingleStep(pOneSame, qOneSame, undefined, {
      draftUniform: 0.999999999,
    });
    expect(drawOneSame.draftTokenId).toBe(0);
    expect(drawOneSame.emittedTokenId).toBe(0);
    expect(drawOneSame.decision.accepted).toBe(true);

    const pOneDiff = toF64(edges.oneHotDifferent.targetProbs);
    const qOneDiff = toF64(edges.oneHotDifferent.proposalProbs);
    const drawOneDiff = sampleSpecSingleStep(pOneDiff, qOneDiff, undefined, {
      draftUniform: 0.5,
      acceptUniform: 0.0,
      residualUniform: 0.5,
    });
    expect(drawOneDiff.draftTokenId).toBe(0);
    expect(drawOneDiff.decision.accepted).toBe(false);
    expect(drawOneDiff.emittedTokenId).toBe(1);
    expect(enumerateSpecSingleStepExact(pOneDiff, qOneDiff).maxAbsError).toBeLessThanOrEqual(tol);
  });

  it('Layer 5A — Greedy multi-cycle Contract A matches PyTorch 2.15.0 (<= 2e-5), fresh k=4, standalone k=1, and k=4 AR token-for-token across S > 8', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.fullForwardMaxAbs; // 2e-5
    const f5 = fixture.layer5ContractAExecution;
    const prompt8 = f5.prompt8;
    const greedyRef = f5.greedyMultiCycle;

    const prefill = prefillDorySelfSpec({
      promptTokens: prompt8,
      params,
      sampling: { temperature: 0, topK: 0, topP: 1.0 },
      ropeTables,
      captureTrace: true,
    });
    expect(prefill.contractBadge).toBe(DORY_SPEC_CONTRACT_A_BADGE);
    expect(prefill.committedLen).toBe(greedyRef.prefill.committedLen);
    expect(prefill.pendingToken).toBe(greedyRef.prefill.pendingToken);
    expect(prefill.deepTrace).not.toBeNull();
    expect(prefill.shallowCodaTrace).not.toBeNull();
    assertAll104BanksSyncedAndRolledBackClean(prefill.kvCache, prompt8.length);
    assertFiniteArrayParity(
      prefill.lastDeepScaledLogits,
      greedyRef.prefill.lastDeepScaledLogits,
      tol,
      'layer5A.prefill.deepLogits',
    );
    assertFiniteArrayParity(
      prefill.lastShallowScaledLogits,
      greedyRef.prefill.lastShallowScaledLogits,
      tol,
      'layer5A.prefill.shallowLogits',
    );

    const committedPrefix = [...prompt8];
    const specEmitted = [...prompt8, prefill.pendingToken];
    let pending: number | null = prefill.pendingToken;

    for (let cIdx = 0; cIdx < greedyRef.cycles.length; cIdx += 1) {
      const refCyc = greedyRef.cycles[cIdx];
      expect(pending).not.toBeNull();
      const cyc = runDorySelfSpecCycle({
        kvCache: prefill.kvCache,
        pendingToken: pending!,
        params,
        gamma: DORY_SPEC_DEFAULT_GAMMA,
        sampling: { temperature: 0, topK: 0, topP: 1.0 },
        ropeTables,
        captureTrace: cIdx === 0,
      });

      expect(cyc.contractBadge).toBe(DORY_SPEC_CONTRACT_A_BADGE);
      expect(cyc.startCommittedLen).toBe(refCyc.startCommittedLen);
      expect(cyc.endCommittedLen).toBe(refCyc.endCommittedLen);
      expect(cyc.inputPendingToken).toBe(refCyc.inputPendingToken);
      expect(cyc.nextPendingToken).toBe(refCyc.nextPendingToken);
      expect(cyc.draftCount).toBe(refCyc.draftCount);
      expect(cyc.acceptedDraftCount).toBe(refCyc.acceptedDraftCount);
      expect(cyc.outcome).toBe(refCyc.outcome);
      expect(cyc.shallowCodaCatchupExecuted).toBe(refCyc.shallowCodaCatchupExecuted);
      expect(cyc.rolledBackPositions).toBe(refCyc.rolledBackPositions);
      expect(Array.from(cyc.emittedTokens)).toEqual(refCyc.emittedTokens);
      assertAll104BanksSyncedAndRolledBackClean(prefill.kvCache, cyc.endCommittedLen);

      // Summary helper preserves scalar metrics without retaining [V] arrays
      const summary = summarizeSelfSpecCycle(cyc);
      expect(summary.outcome).toBe(cyc.outcome);
      expect(summary.acceptedDraftCount).toBe(cyc.acceptedDraftCount);
      expect(Array.from(summary.emittedTokens)).toEqual(refCyc.emittedTokens);

      // 1. Compare TS reused k=4 verify logits vs PyTorch AND vs fresh uncached TS k=4 forward!
      const verifyInputTokens = cyc.verifyPositions.map((v) => v.inputTokenId);
      const freshK4Logits = runFreshTsForward(
        [...committedPrefix, ...verifyInputTokens],
        DORY_SPEC_VERIFY_LOOPS,
      );
      const V = params.config.vocabSize;
      for (let j = 0; j < cyc.verifyPositions.length; j += 1) {
        const vp = cyc.verifyPositions[j];
        const refVp = refCyc.verifyPositions[j];
        expect(vp.absPos).toBe(refVp.absPos);
        expect(vp.inputTokenId).toBe(refVp.inputTokenId);
        expect(vp.reusedPreludeAndLoop1).toBe(refVp.reusedPreludeAndLoop1);
        assertFiniteArrayParity(
          vp.deepScaledLogits,
          refVp.deepScaledLogits,
          tol,
          `layer5A.c${cIdx}.verify[${j}].vsPyTorch`,
        );
        const freshSlice = freshK4Logits.subarray(vp.absPos * V, (vp.absPos + 1) * V);
        assertFiniteArrayParity(
          vp.deepScaledLogits,
          freshSlice,
          tol,
          `layer5A.c${cIdx}.verify[${j}].reusedVsFreshK4`,
        );
      }

      // 2. Compare TS Contract A shallow draft logits vs PyTorch AND vs standalone TS k=1 Dory!
      const draftInputTokens = cyc.draftSteps.map((d) => d.inputTokenId);
      const standaloneK1Logits = runFreshTsForward(
        [...committedPrefix, ...draftInputTokens],
        DORY_SPEC_DRAFT_LOOPS,
      );
      for (let j = 0; j < cyc.draftSteps.length; j += 1) {
        const ds = cyc.draftSteps[j];
        const refDs = refCyc.draftSteps[j];
        expect(ds.absPos).toBe(refDs.absPos);
        expect(ds.inputTokenId).toBe(refDs.inputTokenId);
        expect(ds.proposedTokenId).toBe(refDs.proposedTokenId);
        assertFiniteArrayParity(
          ds.shallowScaledLogits,
          refDs.shallowScaledLogits,
          tol,
          `layer5A.c${cIdx}.draft[${j}].vsPyTorch`,
        );
        const k1Slice = standaloneK1Logits.subarray(ds.absPos * V, (ds.absPos + 1) * V);
        assertFiniteArrayParity(
          ds.shallowScaledLogits,
          k1Slice,
          tol,
          `layer5A.c${cIdx}.draft[${j}].vsStandaloneK1`,
        );
      }

      committedPrefix.push(...refCyc.committedDeltaTokens);
      specEmitted.push(...cyc.emittedTokens);
      pending = cyc.nextPendingToken;
    }

    // 3. Greedy Contract A token sequence === standard TS k=4 AR === PyTorch k=4 AR!
    expect(specEmitted).toEqual(greedyRef.emittedSequence);
    expect(specEmitted).toEqual(greedyRef.standaloneK4ArSequence);

    const tsArKv = createDoryKvCache({
      mode: 'per_loop',
      kLoops: DORY_SPEC_VERIFY_LOOPS,
      config: params.config,
    });
    const tsArPrefill = runDoryForwardChunk({
      tokens: prompt8,
      params,
      kvCache: tsArKv,
      kLoops: DORY_SPEC_VERIFY_LOOPS,
      kvCacheMode: 'per_loop',
      ropeTables,
    });
    const tsArSeq = [...prompt8];
    const V = params.config.vocabSize;
    tsArSeq.push(
      argmaxIndex(tsArPrefill.scaledLogits, (prompt8.length - 1) * V, V),
    );
    while (tsArSeq.length < specEmitted.length) {
      const nextIn = tsArSeq[tsArSeq.length - 1];
      const stepRes = runDoryForwardChunk({
        tokens: [nextIn],
        params,
        kvCache: tsArKv,
        kLoops: DORY_SPEC_VERIFY_LOOPS,
        kvCacheMode: 'per_loop',
        ropeTables,
      });
      tsArSeq.push(argmaxIndex(stepRes.scaledLogits, 0, V));
    }
    expect(specEmitted).toEqual(tsArSeq);
  });

  it('Layer 5B — Multi-cycle Contract A rollback, Shallow Coda catch-up, draft EOS early stop, final slot & DR23 guards match PyTorch 2.15.0 (<= 2e-5)', () => {
    const tol = DORY_NUMERICAL_TOLERANCES.fullForwardMaxAbs; // 2e-5
    const f5 = fixture.layer5ContractAExecution;
    const V = params.config.vocabSize;

    // --- 5B-1: Stochastic 4-cycle suite exercising partial_accept_reject -> full_accept_bonus
    //           (+ 1-token Shallow Coda catch-up) -> zero_accept_reject -> partial_accept_reject
    //           crossing BOTH W = 8 eviction AND cap_swa = 16 ring wraparound! ---
    const stochRef = f5.stochasticMultiCycle;
    const prefillS = prefillDorySelfSpec({
      promptTokens: f5.prompt8,
      params,
      sampling: { temperature: 1.0, topK: 0, topP: 1.0 },
      initialSampleUniform: stochRef.initialSampleUniform,
      ropeTables,
    });
    expect(prefillS.pendingToken).toBe(stochRef.prefill.pendingToken);
    assertAll104BanksSyncedAndRolledBackClean(prefillS.kvCache, f5.prompt8.length);

    const prefixS = [...f5.prompt8];
    let pendingS: number | null = prefillS.pendingToken;

    const observedOutcomes: DorySpecOutcomeKind[] = [];
    for (let cIdx = 0; cIdx < stochRef.cycles.length; cIdx += 1) {
      const { explicitVariates, result: refCyc } = stochRef.cycles[cIdx];
      const cyc = runDorySelfSpecCycle({
        kvCache: prefillS.kvCache,
        pendingToken: pendingS!,
        params,
        gamma: DORY_SPEC_DEFAULT_GAMMA,
        sampling: { temperature: 1.0, topK: 0, topP: 1.0 },
        explicitVariates: toTsExplicitVariates(explicitVariates),
        ropeTables,
        captureTrace: true,
      });

      observedOutcomes.push(cyc.outcome);
      expect(cyc.outcome).toBe(refCyc.outcome);
      expect(cyc.startCommittedLen).toBe(refCyc.startCommittedLen);
      expect(cyc.endCommittedLen).toBe(refCyc.endCommittedLen);
      expect(cyc.acceptedDraftCount).toBe(refCyc.acceptedDraftCount);
      expect(cyc.rolledBackPositions).toBe(refCyc.rolledBackPositions);
      expect(cyc.shallowCodaCatchupExecuted).toBe(refCyc.shallowCodaCatchupExecuted);
      expect(cyc.nextPendingToken).toBe(refCyc.nextPendingToken);
      expect(Array.from(cyc.emittedTokens)).toEqual(refCyc.emittedTokens);
      expect(cyc.verifyTrace).not.toBeNull();
      expect(cyc.verifyTrace!.executedSublayerCount).toBe(192);
      assertAll104BanksSyncedAndRolledBackClean(prefillS.kvCache, cyc.endCommittedLen);

      // Reused k=4 verify logits vs PyTorch AND vs fresh uncached TS k=4 across S > 8 rollback!
      const verifyInputTokens = cyc.verifyPositions.map((v) => v.inputTokenId);
      const freshK4 = runFreshTsForward([...prefixS, ...verifyInputTokens], DORY_SPEC_VERIFY_LOOPS);
      for (let j = 0; j < cyc.verifyPositions.length; j += 1) {
        const vp = cyc.verifyPositions[j];
        const refVp = refCyc.verifyPositions[j];
        assertFiniteArrayParity(
          vp.deepScaledLogits,
          refVp.deepScaledLogits,
          tol,
          `layer5B.c${cIdx}.verify[${j}].vsPyTorch`,
        );
        assertFiniteArrayParity(
          vp.deepScaledLogits,
          freshK4.subarray(vp.absPos * V, (vp.absPos + 1) * V),
          tol,
          `layer5B.c${cIdx}.verify[${j}].vsFreshK4`,
        );
      }

      // Contract A multi-cycle shallow draft vs PyTorch AND vs standalone TS k=1 Dory (proving
      // BOTH 0-extra-work rollback AND 1-token Shallow Coda catch-up stay exact across cycles!)
      const draftInputTokens = cyc.draftSteps.map((d) => d.inputTokenId);
      const standaloneK1 = runFreshTsForward(
        [...prefixS, ...draftInputTokens],
        DORY_SPEC_DRAFT_LOOPS,
      );
      for (let j = 0; j < cyc.draftSteps.length; j += 1) {
        const ds = cyc.draftSteps[j];
        const refDs = refCyc.draftSteps[j];
        expect(ds.proposedTokenId).toBe(refDs.proposedTokenId);
        assertFiniteArrayParity(
          ds.shallowScaledLogits,
          refDs.shallowScaledLogits,
          tol,
          `layer5B.c${cIdx}.draft[${j}].vsPyTorch`,
        );
        assertFiniteArrayParity(
          ds.shallowScaledLogits,
          standaloneK1.subarray(ds.absPos * V, (ds.absPos + 1) * V),
          tol,
          `layer5B.c${cIdx}.draft[${j}].vsStandaloneK1`,
        );
      }

      prefixS.push(...refCyc.committedDeltaTokens);
      pendingS = cyc.nextPendingToken;
    }

    expect(observedOutcomes).toEqual([
      'partial_accept_reject',
      'full_accept_bonus',
      'zero_accept_reject',
      'partial_accept_reject',
    ]);

    // --- 5B-2: Draft EOS Early Stop (full_accept_eos, no bonus token, clean halt) ---
    const eosRef = f5.draftEosEarlyStop;
    const prefillEos = prefillDorySelfSpec({
      promptTokens: f5.prompt8,
      params,
      sampling: { temperature: 1.0, topK: 0, topP: 1.0 },
      initialSampleUniform: eosRef.initialSampleUniform,
      ropeTables,
    });
    const eosCyc = runDorySelfSpecCycle({
      kvCache: prefillEos.kvCache,
      pendingToken: prefillEos.pendingToken,
      params,
      gamma: DORY_SPEC_DEFAULT_GAMMA,
      sampling: { temperature: 1.0, topK: 0, topP: 1.0 },
      explicitVariates: toTsExplicitVariates(eosRef.explicitVariates),
      ropeTables,
    });
    expect(eosCyc.outcome).toBe('full_accept_eos');
    expect(eosCyc.draftEndedWithEos).toBe(true);
    expect(eosCyc.draftCount).toBe(eosRef.result.draftCount);
    expect(eosCyc.acceptedDraftCount).toBe(eosRef.result.acceptedDraftCount);
    expect(eosCyc.nextPendingToken).toBeNull();
    expect(eosCyc.haltReason).toBe('eos');
    expect(eosCyc.shallowCodaCatchupExecuted).toBe(false);
    expect(eosCyc.correctionOrBonusToken).toBeNull();
    expect(Array.from(eosCyc.emittedTokens)).toEqual(eosRef.result.emittedTokens);
    assertAll104BanksSyncedAndRolledBackClean(prefillEos.kvCache, eosCyc.endCommittedLen);

    // --- 5B-3: Final remaining slot (L = S_max - 1 = 31, gammaEffective = 0) ---
    const finalRef = f5.finalSlotDeepDecode;
    const prefillFinal = prefillDorySelfSpec({
      promptTokens: finalRef.prompt31,
      params,
      sampling: { temperature: 0, topK: 0, topP: 1.0 },
      ropeTables,
    });
    expect(prefillFinal.pendingToken).toBe(finalRef.prefillPendingToken);
    const finalCyc = runDorySelfSpecCycle({
      kvCache: prefillFinal.kvCache,
      pendingToken: prefillFinal.pendingToken,
      params,
      gamma: DORY_SPEC_DEFAULT_GAMMA,
      sampling: { temperature: 0, topK: 0, topP: 1.0 },
      ropeTables,
    });
    expect(finalCyc.outcome).toBe('final_slot_deep_decode');
    expect(finalCyc.gammaEffective).toBe(0);
    expect(finalCyc.draftCount).toBe(0);
    expect(finalCyc.endCommittedLen).toBe(DORY_MINI_CONFIG.maxPositionEmbeddings);
    expect(finalCyc.shallowCodaCatchupExecuted).toBe(true);
    expect(finalCyc.haltReason).toBe(finalRef.result.haltReason);
    expect(Array.from(finalCyc.emittedTokens)).toEqual(finalRef.result.emittedTokens);
    assertAll104BanksSyncedAndRolledBackClean(prefillFinal.kvCache, 32);
    assertFiniteArrayParity(
      finalCyc.lastCommittedDeepScaledLogits,
      finalRef.result.verifyPositions[0].deepScaledLogits,
      tol,
      'layer5B.finalSlot.vsPyTorch',
    );
    const freshFinalK4 = runFreshTsForward(
      [...finalRef.prompt31, prefillFinal.pendingToken],
      DORY_SPEC_VERIFY_LOOPS,
    );
    assertFiniteArrayParity(
      finalCyc.lastCommittedDeepScaledLogits,
      freshFinalK4.subarray(31 * V, 32 * V),
      tol,
      'layer5B.finalSlot.vsFreshK4',
    );

    // --- 5B-4: DR23 Hard Guards (forbid last_loop, kLoops !== 4, or missing output_shallow) ---
    const badLastLoop = createDoryKvCache({
      mode: 'last_loop',
      kLoops: 4,
      includeShallowCoda: true,
      config: params.config,
    });
    const badK2 = createDoryKvCache({
      mode: 'per_loop',
      kLoops: 2,
      includeShallowCoda: true,
      config: params.config,
    });
    const badNoShallow = createDoryKvCache({
      mode: 'per_loop',
      kLoops: 4,
      includeShallowCoda: false,
      config: params.config,
    });
    for (const badKv of [badLastLoop, badK2, badNoShallow]) {
      expect(() =>
        runDorySelfSpecCycle({ kvCache: badKv, pendingToken: 4, params, ropeTables }),
      ).toThrow(DorySessionImmutableConfigError);
    }
  });

  it('Layer 6 — Pearson chi^2 < 103.51 (N=20,000, alpha=0.001, df=63) PLUS 2 mandatory failing negative controls', () => {
    const f6 = fixture.layer6Chi2AndNegativeControls;
    const N = f6.sampleCount; // 20,000
    const crit63 = DORY_SPEC_PEARSON_CHI2_DF63_ALPHA_001; // 103.51
    expect(N).toBe(20_000);
    expect(f6.degreesOfFreedom).toBe(63);
    expect(f6.criticalValueDf63Alpha001).toBe(crit63);

    // 6A. Valid division-free binary64 sampler PASSES Pearson chi^2 < 103.51 on BOTH:
    //     (i) Real Dory V=64 k=4 vs k=1 model distribution, AND
    //     (ii) High-divergence V=64 distribution (used for Negative Control 1)
    const pairs = [
      {
        label: 'modelPair',
        p: toF64(f6.modelPair.targetProbs),
        q: toF64(f6.modelPair.proposalProbs),
        seed: 0xd0416001,
      },
      {
        label: 'highDivergencePair',
        p: toF64(f6.highDivergencePair.targetProbs),
        q: toF64(f6.highDivergencePair.proposalProbs),
        seed: 0xd0416002,
      },
    ];

    for (const pair of pairs) {
      const counts = new Int32Array(pair.p.length);
      const prng = createDoryPrng(pair.seed);
      for (let i = 0; i < N; i += 1) {
        const res = sampleSpecSingleStep(pair.p, pair.q, prng);
        counts[res.emittedTokenId] += 1;
      }
      const chi2 = computePearsonChiSquared(counts, pair.p, N);
      expect(chi2.degreesOfFreedom).toBe(63);
      expect(chi2.sampleCount).toBe(N);
      expect(chi2.minExpectedCount).toBeGreaterThanOrEqual(10);
      expect(chi2.statistic, `valid sampler ${pair.label} chi^2`).toBeLessThan(crit63);
    }

    // ------------------------------------------------------------------------
    // MANDATORY NEGATIVE CONTROL 1: Buggy sampler resampling from `p` instead of
    // `norm(max(0, p - q))` MUST FAIL BOTH exact enumeration AND Pearson chi^2!
    // ------------------------------------------------------------------------
    const hdP = toF64(f6.highDivergencePair.targetProbs);
    const hdQ = toF64(f6.highDivergencePair.proposalProbs);

    // Exact analytical check on Negative Control 1 (proving exact enumeration rejects it!)
    let alphaHd = 0.0;
    for (let v = 0; v < hdP.length; v += 1) {
      alphaHd += Math.min(hdP[v], hdQ[v]);
    }
    const buggyMarginal = new Float64Array(hdP.length);
    for (let v = 0; v < hdP.length; v += 1) {
      buggyMarginal[v] = Math.min(hdP[v], hdQ[v]) + (1.0 - alphaHd) * hdP[v];
    }
    assertFiniteArrayParity(
      buggyMarginal,
      f6.highDivergencePair.negControl1ResampleFromPMarginal,
      1e-15,
      'layer6.negControl1.analyticalMarginal',
    );
    const neg1MaxAbsErr = computeStrictMaxAbsDiff(buggyMarginal, hdP, 'layer6.negControl1.err');
    expect(neg1MaxAbsErr).toBeGreaterThan(1e-3); // Violates 1e-15 exact gate by > 12 orders of magnitude!

    // Monte Carlo Pearson chi^2 check on Negative Control 1 (N = 20,000, df = 63, crit = 103.51)
    const buggyCounts = new Int32Array(hdP.length);
    const neg1Prng = createDoryPrng(0xd0416002);
    for (let i = 0; i < N; i += 1) {
      const d = sampleCategoricalFloat64(hdQ, neg1Prng.nextFloat64());
      const pd = hdP[d];
      const qd = hdQ[d];
      const accepted = pd >= qd || neg1Prng.nextFloat64() * qd < pd;
      // DELIBERATELY BROKEN MUTATION: resample from `p` instead of `norm(max(0, p - q))`!
      const emitted = accepted ? d : sampleCategoricalFloat64(hdP, neg1Prng.nextFloat64());
      buggyCounts[emitted] += 1;
    }
    const neg1Chi2 = computePearsonChiSquared(buggyCounts, hdP, N);
    expect(neg1Chi2.degreesOfFreedom).toBe(63);
    expect(neg1Chi2.statistic, 'Negative Control 1 (resample from p) MUST FAIL chi^2 > 103.51').toBeGreaterThan(
      crit63,
    );
    // Expected non-centrality is ~1,795 -> observed statistic exceeds 500 cleanly!
    expect(neg1Chi2.statistic).toBeGreaterThan(500);

    // ------------------------------------------------------------------------
    // MANDATORY NEGATIVE CONTROL 2: Buggy `1e-38` denominator clamp
    // `r < p[d] / Math.max(q[d], 1e-38)` MUST FAIL on `q = 1e-40, p = 5e-41`!
    // ------------------------------------------------------------------------
    const neg2 = f6.negControl2BuggyClamp1e38;
    const subP = Float64Array.of(neg2.pDraft, 1.0 - neg2.pDraft);
    const subQ = Float64Array.of(neg2.qDraft, 1.0 - neg2.qDraft);
    const critDf1 = neg2.criticalValueDf1Alpha001; // 10.828

    // Deterministic boundary failure: r = 0.25 accepted by valid division-free code, rejected by clamp!
    expect(evaluateRejectionAcceptance(0, subP, subQ, 0.25, false).accepted).toBe(true);
    const buggyClampThreshold = neg2.pDraft / Math.max(neg2.qDraft, 1e-38);
    expect(buggyClampThreshold).toBeCloseTo(0.005, 12);
    expect(0.25 < buggyClampThreshold).toBe(false); // 100x corrupted acceptance threshold!

    // Conditional Monte Carlo Pearson chi^2 (N = 20,000 draws conditioned on d = 0, df = 1):
    // True acceptance rate is 0.500 -> valid division-free PASSES (< 10.828); buggy clamp FAILS (>> 10.828)!
    const expectedAcceptPair = Float64Array.of(neg2.exactAcceptRate, 1.0 - neg2.exactAcceptRate);
    const validAcceptCounts = new Int32Array(2);
    const buggyClampCounts = new Int32Array(2);
    const neg2Prng = createDoryPrng(0xd0416003);
    for (let i = 0; i < N; i += 1) {
      const r = neg2Prng.nextFloat64();
      const validAccepted = evaluateRejectionAcceptance(0, subP, subQ, r, false).accepted;
      validAcceptCounts[validAccepted ? 0 : 1] += 1;

      // DELIBERATELY BROKEN MUTATION: buggy 1e-38 denominator floor clamp!
      const buggyAccepted = r < neg2.pDraft / Math.max(neg2.qDraft, 1e-38);
      buggyClampCounts[buggyAccepted ? 0 : 1] += 1;
    }
    const validNeg2Chi2 = computePearsonChiSquared(validAcceptCounts, expectedAcceptPair, N);
    const buggyNeg2Chi2 = computePearsonChiSquared(buggyClampCounts, expectedAcceptPair, N);
    expect(validNeg2Chi2.degreesOfFreedom).toBe(1);
    expect(validNeg2Chi2.statistic).toBeLessThan(critDf1);
    expect(
      buggyNeg2Chi2.statistic,
      'Negative Control 2 (buggy 1e-38 clamp) MUST FAIL chi^2 > 10.828',
    ).toBeGreaterThan(critDf1);
    expect(buggyNeg2Chi2.statistic).toBeGreaterThan(10_000);

    // Bonus Negative Control 3: Buggy `Z <= 1e-12` residual fallback to `p` fails on Delta = 4e-13
    const deltaP = toF64(fixture.layers3And4BoundariesAndEdges.edgeCases.delta4e13.targetProbs);
    const deltaQ = toF64(fixture.layers3And4BoundariesAndEdges.edgeCases.delta4e13.proposalProbs);
    const validDeltaRes = computeRejectionResidualFloat64(deltaP, deltaQ);
    assertBitExactFiniteArray(validDeltaRes.probs, [1.0, 0.0], 'layer6.negControl3.validDelta');
    const buggyDeltaFallback = validDeltaRes.massZ <= 1e-12 ? deltaP : validDeltaRes.probs;
    expect(computeStrictMaxAbsDiff(buggyDeltaFallback, [1.0, 0.0], 'layer6.negControl3.buggy')).toBeGreaterThan(
      0.49,
    );
  });
});
