import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { memoryFor } from '../src/model/memory';
import { NEMOTRON_SPEC } from '../src/model/spec';
import {
  applyAdaLnModulate,
  collectTimeConditioningTensors,
  computeGlobalTimestepProjection,
  computeLayerModulationParams,
  computeSinusoidalTimestepEmbedding,
  computeTimeConditioningFingerprint,
  createTimeConditioningWeights,
  DEFAULT_TIMESTEP_MAX_PERIOD,
  DEFAULT_TIMESTEP_MAX_PERIOD_SCALE,
  modulationOrderForBayKind,
} from '../src/twotower/modulation';
import {
  blockContainsEos,
  GUMBEL_UNIFORM_MIN,
  MDLM_MASK_SUPPRESS_LOGIT,
  mdlmConstrainLogProbs,
  runConfidenceUnmaskStep,
  sampleBlockPredictions,
} from '../src/twotower/sampler';
import {
  TWOTOWER_COMMUNITY_NVFP4_REFERENCE,
  TWOTOWER_DISCREPANCY_LEDGER,
  TWOTOWER_HF_REPO,
  TWOTOWER_HF_REVISION,
  TWOTOWER_MINI_CONFIG,
  TWOTOWER_PAPER_ARXIV_ID,
  TWOTOWER_REVISION_CHECKED_DATE,
  TWOTOWER_SPEC,
  twoTowerMemoryFor,
} from '../src/twotower/spec';

describe('TwoTower Production Spec, Discrepancy Ledger & Memory Accounting (src/twotower/spec.ts)', () => {
  it('pins exact HF revision provenance, paper ID, and composes NEMOTRON_SPEC unchanged', () => {
    expect(TWOTOWER_SPEC.backbone).toBe(NEMOTRON_SPEC);
    expect(TWOTOWER_SPEC.provenance.hfRepo).toBe(TWOTOWER_HF_REPO);
    expect(TWOTOWER_SPEC.provenance.hfRepo).toBe('nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16');
    expect(TWOTOWER_SPEC.provenance.hfRevision).toBe(TWOTOWER_HF_REVISION);
    expect(TWOTOWER_SPEC.provenance.hfRevision).toBe('35b6498508be8856d380cdfe4a900a79b918d1ea');
    expect(TWOTOWER_SPEC.provenance.revisionCheckedDate).toBe(TWOTOWER_REVISION_CHECKED_DATE);
    expect(TWOTOWER_SPEC.provenance.revisionCheckedDate).toBe('2026-11-25');
    expect(TWOTOWER_SPEC.provenance.revisionPinDecision).toContain(
      '35b6498508be8856d380cdfe4a900a79b918d1ea',
    );
    expect(TWOTOWER_SPEC.provenance.paperArxivId).toBe(TWOTOWER_PAPER_ARXIV_ID);
    expect(TWOTOWER_SPEC.provenance.paperArxivId).toBe('2606.26493v2');
  });

  it('proves exact 0-byte parameter-to-payload accounting and separates inspected vs. derived inventory', () => {
    const inspected = TWOTOWER_SPEC.directlyInspectedMetadata;
    const derived = TWOTOWER_SPEC.derivedInventoryCalculations;

    // Directly inspected HF metadata & safetensors index header
    expect(inspected.safetensorsTotalPayloadBytes).toBe(126_371_817_472);
    expect(inspected.safetensorsTensorCount).toBe(12_544);
    expect(inspected.safetensorsShardCount).toBe(24);
    expect(inspected.contextTowerTensorCount).toBe(6_243);
    expect(inspected.denoiserTowerTensorCount).toBe(6_243);
    expect(inspected.timeConditioningTensorCount).toBe(58);
    expect(
      inspected.contextTowerTensorCount +
        inspected.denoiserTowerTensorCount +
        inspected.timeConditioningTensorCount,
    ).toBe(inspected.safetensorsTensorCount);
    expect(inspected.fp32ScoreCorrectionBiasParams).toBe(2 * 23 * 128);
    expect(inspected.fp32ScoreCorrectionBiasParams).toBe(5_888);

    // Analytically derived parameter breakdown
    expect(derived.singleTowerEmbeddingsParams).toBe(352_321_536);
    expect(derived.singleTowerLmHeadParams).toBe(352_321_536);
    expect(derived.singleTowerFinalNormParams).toBe(2_688);
    expect(derived.mambaLayerParams).toBe(38_744_896);
    expect(derived.mambaAllLayersParams).toBe(891_132_608);
    expect(derived.gqaLayerParams).toBe(23_399_040);
    expect(derived.gqaAllLayersParams).toBe(140_394_240);
    expect(derived.moeLayerParams).toBe(1_297_468_160);
    expect(derived.moeAllLayersParams).toBe(29_841_767_680);
    expect(derived.singleTowerParams).toBe(31_577_940_288);
    expect(derived.dualTowerBackboneParams).toBe(63_155_880_576);

    // Denoiser-only adaLN-single time conditioning
    expect(derived.timeEmbedderMlp0Params).toBe(690_816);
    expect(derived.timeEmbedderMlp2Params).toBe(7_228_032);
    expect(derived.timeEmbedderMlpParams).toBe(7_918_848);
    expect(derived.timeBlockParams).toBe(21_684_096);
    expect(derived.scaleShiftTablesParams).toBe(419_328);
    expect(derived.timeConditioningTotalParams).toBe(30_022_272);

    // Grand total & exact 0-byte safetensors payload identity
    expect(derived.grandTotalParams).toBe(63_185_902_848);
    expect(derived.grandTotalParams * 2 + inspected.fp32ScoreCorrectionBiasParams * 2).toBe(
      126_371_817_472,
    );
    expect(derived.derivedPayloadBytes).toBe(inspected.safetensorsTotalPayloadBytes);
    expect(derived.payloadByteDiscrepancy).toBe(0);
    expect(derived.contextTowerPayloadBytes + derived.denoiserTowerTotalPayloadBytes).toBe(
      inspected.safetensorsTotalPayloadBytes,
    );

    // Active per-token counts for single tower
    expect(derived.singleTowerActiveParamsExclEmbHead).toBe(2_875_433_280);
    expect(derived.singleTowerActiveParamsInclHeadExclEmb).toBe(3_227_754_816);
    expect(derived.singleTowerActiveParamsInclEmbHead).toBe(3_580_076_352);
  });

  it('maps exact 29/52 aligned transfer bays (6 GQA + 23 Mamba-2) and 23 stateless MoE bays', () => {
    const align = TWOTOWER_SPEC.layerAlignment;
    expect(align.totalLayers).toBe(52);
    expect(align.alignedTransferBayCount).toBe(29);
    expect(align.gqaTransferBayCount).toBe(6);
    expect(align.mambaTransferBayCount).toBe(23);
    expect(align.statelessMoeBayCount).toBe(23);
    expect(align.gqaTransferBayIds).toEqual([6, 13, 20, 27, 34, 43]);
    expect(align.gqaTransferLayerIndices).toEqual([5, 12, 19, 26, 33, 42]);
    expect(align.mambaTransferBayIds).toHaveLength(23);
    expect(align.statelessMoeBayIds).toHaveLength(23);
    expect(align.entries).toHaveLength(52);

    for (const entry of align.entries) {
      expect(entry.bayId).toBe(entry.layerIndex + 1);
      expect(entry.denoiserModulationOrder).toBe(modulationOrderForBayKind(entry.kind));
      if (entry.kind === '*') {
        expect(entry.transferKind).toBe('gqa_kv');
        expect(entry.transfersState).toBe(true);
      } else if (entry.kind === 'M') {
        expect(entry.transferKind).toBe('mamba_state');
        expect(entry.transfersState).toBe(true);
      } else {
        expect(entry.transferKind).toBe('none');
        expect(entry.transfersState).toBe(false);
      }
    }
  });

  it('records verified production defaults, discrepancy ledger entries, and miniature config', () => {
    const defaults = TWOTOWER_SPEC.productionDefaults;
    expect(defaults.blockSize).toBe(16);
    expect(defaults.stepsPerBlock).toBe(16);
    expect(defaults.maskTokenId).toBe(3);
    expect(defaults.eosTokenIds).toEqual([2, 11]);
    expect(defaults.codeConfidenceThreshold).toBe(0.9);
    expect(defaults.paperConfidenceThreshold).toBe(0.8);
    expect(defaults.codeTemperature).toBe(0.0);
    expect(defaults.cardTemperature).toBe(0.1);
    expect(defaults.primaryPrecision).toBe('bf16');
    expect(defaults.mambaScanPrecision).toBe('fp32');
    expect(defaults.moeRouterPrecision).toBe('fp64');
    expect(defaults.epistemicNote).toMatch(/BF16/i);
    expect(defaults.epistemicNote).toMatch(/miniature Float32\/Float64/i);

    const ledgerIds = new Set(TWOTOWER_DISCREPANCY_LEDGER.map((e) => e.id));
    expect(ledgerIds.has('adaln_param_count')).toBe(true);
    expect(ledgerIds.has('shared_experts_count')).toBe(true);
    expect(ledgerIds.has('total_parameter_count')).toBe(true);
    expect(ledgerIds.has('max_context_length')).toBe(true);
    expect(ledgerIds.has('confidence_threshold_and_temperature')).toBe(true);
    expect(ledgerIds.has('adaln_rmsnorm_ordering')).toBe(true);
    expect(ledgerIds.has('moe_router_precision')).toBe(true);
    expect(TWOTOWER_SPEC.discrepancyLedger).toBe(TWOTOWER_DISCREPANCY_LEDGER);

    expect(TWOTOWER_MINI_CONFIG).toEqual({
      d: 8,
      vocabSize: 34,
      maskTokenId: 32,
      eosTokenId: 33,
      freqDim: 16,
      kvCapacity: 24,
      defaultPromptLen: 8,
      defaultBlockSize: 4,
      supportedBlockSizes: [4, 8],
      defaultStepsPerBlock: 4,
      provisionalDemoThreshold: 0.22,
      presetThresholds: { miniDemo: 0.22, paper: 0.8, referenceCode: 0.9 },
      gqaAtlasWidth: 128,
      gqaAtlasHeight: 128,
    });
  });

  it('computes exact TwoTower production memory across mask_diffusion, mock_ar, and ar modes', () => {
    const nanoRef = memoryFor(NEMOTRON_SPEC, 1024, 'bf16');

    const diff = twoTowerMemoryFor(1024, 16, 'mask_diffusion');
    expect(diff.persistentContextMambaBytes).toBe(nanoRef.fixedMambaBytes); // 49,364,992 B (47.08 MiB)
    expect(diff.persistentContextKvBytes).toBe(nanoRef.kvBytes); // 1024 * 6,144 B
    expect(diff.persistentContextTotalBytes).toBe(nanoRef.hybridTotalBytes);
    expect(diff.kvBytesPerToken).toBe(6_144);
    expect(diff.ephemeralDenoiserBlockKvBytes).toBe(16 * 6_144); // 96 KiB ephemeral scratchpad
    expect(diff.mockArMambaS2SnapshotBytes).toBe(0);
    expect(diff.totalActiveCacheBytes).toBe(nanoRef.hybridTotalBytes + 16 * 6_144);
    expect(diff.denoiserReadOnlyMirrorBytes).toBe(nanoRef.hybridTotalBytes);
    expect(diff.dualTowerBf16WeightBytes).toBe(126_371_817_472);
    expect(diff.dualTowerBf16WeightBytes / 1024 ** 3).toBeCloseTo(117.69, 2);
    expect(diff.averagePerTowerBf16WeightBytes / 1024 ** 3).toBeCloseTo(58.85, 2);
    expect(diff.activeModelWeightBytes).toBe(126_371_817_472);
    expect(diff.communityNvfp4Reference).toBe(TWOTOWER_COMMUNITY_NVFP4_REFERENCE);
    expect(diff.communityNvfp4Reference.isOfficialCheckpoint).toBe(false);
    expect(diff.communityNvfp4Reference.quantizedScope).toBe('routed_moe_experts_only');

    const mockAr = twoTowerMemoryFor(1024, 16, 'mock_ar');
    expect(mockAr.persistentContextTotalBytes).toBe(nanoRef.hybridTotalBytes);
    expect(mockAr.ephemeralDenoiserBlockKvBytes).toBe(0);
    expect(mockAr.mockArMambaS2SnapshotBytes).toBe(49_364_992); // extra 47.08 MiB mamba_s2 snapshot
    expect(mockAr.totalActiveCacheBytes).toBe(nanoRef.hybridTotalBytes + 49_364_992);
    expect(mockAr.denoiserReadOnlyMirrorBytes).toBe(49_364_992 + 1023 * 6_144);
    expect(mockAr.activeModelWeightBytes).toBe(126_371_817_472);

    const ar = twoTowerMemoryFor(1024, 16, 'ar');
    expect(ar.ephemeralDenoiserBlockKvBytes).toBe(0);
    expect(ar.mockArMambaS2SnapshotBytes).toBe(0);
    expect(ar.denoiserReadOnlyMirrorBytes).toBe(0);
    expect(ar.totalActiveCacheBytes).toBe(nanoRef.hybridTotalBytes);
    expect(ar.activeModelWeightBytes).toBe(TWOTOWER_SPEC.derivedInventoryCalculations.contextTowerPayloadBytes);

    // Reject invalid inputs
    expect(() => twoTowerMemoryFor(-1, 16, 'mask_diffusion')).toThrow(RangeError);
    expect(() => twoTowerMemoryFor(8.5, 16, 'mask_diffusion')).toThrow(RangeError);
    expect(() => twoTowerMemoryFor(8, -4, 'mask_diffusion')).toThrow(RangeError);
    expect(() => twoTowerMemoryFor(8, 4, 'invalid' as 'ar')).toThrow(RangeError);
  });
});

describe('TwoTower Timestep Embedding & PixArt-alpha adaLN-single (src/twotower/modulation.ts)', () => {
  it('computes cos-first [cos, sin] sinusoidal embedding with t*1000 scaling and exact t=0 boundary', () => {
    const phi0 = computeSinusoidalTimestepEmbedding(0, 16);
    expect(phi0).toHaveLength(16);
    // COS FIRST [0..7] === 1.0, SIN SECOND [8..15] === 0.0
    for (let k = 0; k < 8; k++) {
      expect(phi0[k]).toBe(1.0);
      expect(phi0[8 + k]).toBe(0.0);
    }

    // Verify exact t * 1000 scaling and FP32 exp(-ln(10000) * k / half) progression at t = 0.75
    const t = 0.75;
    const freqDim = 16;
    const half = freqDim / 2;
    const phi = computeSinusoidalTimestepEmbedding(
      t,
      freqDim,
      DEFAULT_TIMESTEP_MAX_PERIOD_SCALE,
      DEFAULT_TIMESTEP_MAX_PERIOD,
    );
    const tScaledF32 = Math.fround(Math.fround(t) * Math.fround(1000));
    const negLogMaxPeriodF32 = Math.fround(-Math.log(10000));
    for (let k = 0; k < half; k++) {
      const exponentF32 = Math.fround(Math.fround(negLogMaxPeriodF32 * k) / half);
      const omega = Math.fround(Math.exp(exponentF32));
      const arg = Math.fround(tScaledF32 * omega);
      expect(phi[k]).toBe(Math.fround(Math.cos(arg)));
      expect(phi[half + k]).toBe(Math.fround(Math.sin(arg)));
    }

    // Also verify production freqDim = 256
    const phi256 = computeSinusoidalTimestepEmbedding(1.0, 256);
    expect(phi256).toHaveLength(256);
    expect(phi256[0]).toBe(Math.fround(Math.cos(1000)));
    expect(phi256[128]).toBe(Math.fround(Math.sin(1000)));

    // Non-finite t and invalid freqDim rejected
    expect(() => computeSinusoidalTimestepEmbedding(Number.NaN, 16)).toThrow(RangeError);
    expect(() => computeSinusoidalTimestepEmbedding(Infinity, 16)).toThrow(RangeError);
    expect(() => computeSinusoidalTimestepEmbedding(-Infinity, 16)).toThrow(RangeError);
    expect(() => computeSinusoidalTimestepEmbedding(0.5, 15)).toThrow(RangeError);
    expect(() => computeSinusoidalTimestepEmbedding(0.5, 0)).toThrow(RangeError);
    expect(() => computeSinusoidalTimestepEmbedding(0.5, -4)).toThrow(RangeError);
  });

  it('generates deterministic, well-conditioned weights with canonical tensor export & fingerprint', () => {
    const cfg = { d: 8, freqDim: 16, numLayers: 52, seed: 0x54543031 };
    const w1 = createTimeConditioningWeights(cfg);
    const w2 = createTimeConditioningWeights(cfg);
    const wDiff = createTimeConditioningWeights({ ...cfg, seed: 0x54543032 });

    expect(w1.fingerprint).toBe(w2.fingerprint);
    expect(w1.fingerprint).toHaveLength(16);
    expect(w1.fingerprint).not.toBe(wDiff.fingerprint);

    const tensors = collectTimeConditioningTensors(w1);
    expect(tensors).toHaveLength(58); // 6 MLP/block tensors + 52 layer tables
    const totalFloats = tensors.reduce((acc, item) => acc + item.data.length, 0);
    expect(totalFloats).toBe(1_672); // 128 + 8 + 64 + 8 + 192 + 24 + 52*24

    // Bit-identical weight tensors across calls
    const tensors2 = collectTimeConditioningTensors(w2);
    for (let i = 0; i < tensors.length; i++) {
      expect(tensors[i].name).toBe(tensors2[i].name);
      expect(Array.from(tensors[i].data)).toEqual(Array.from(tensors2[i].data));
    }

    // Mutating a single Float32 value must change fingerprint
    const perturbedBias = new Float32Array(w1.tBlockBias);
    perturbedBias[0] = Math.fround(perturbedBias[0] + 0.01);
    const perturbedFp = computeTimeConditioningFingerprint(cfg, {
      ...w1,
      tBlockBias: perturbedBias,
    });
    expect(perturbedFp).not.toBe(w1.fingerprint);

    // Verify scaleShiftTables have non-degenerate, finite magnitudes (~randn / sqrt(d))
    let sumSq = 0;
    let count = 0;
    for (const tbl of w1.scaleShiftTables) {
      expect(tbl).toHaveLength(24);
      for (let i = 0; i < tbl.length; i++) {
        expect(Number.isFinite(tbl[i])).toBe(true);
        sumSq += tbl[i] * tbl[i];
        count++;
      }
    }
    const rms = Math.sqrt(sumSq / count);
    // Expected RMS ~ 1 / sqrt(8) = 0.3536
    expect(rms).toBeGreaterThan(0.2);
    expect(rms).toBeLessThan(0.55);
  });

  it('evaluates global timestep MLP, per-layer (shift, scale, gate), bay modulation order, and applyAdaLnModulate', () => {
    expect(modulationOrderForBayKind('M')).toBe('modulate_then_norm');
    expect(modulationOrderForBayKind('*')).toBe('modulate_then_norm');
    expect(modulationOrderForBayKind('E')).toBe('norm_then_modulate');

    const w = createTimeConditioningWeights({ d: 8, freqDim: 16, numLayers: 52, seed: 42 });
    const proj1 = computeGlobalTimestepProjection(1.0, w);
    const projQuarter = computeGlobalTimestepProjection(0.25, w);

    expect(proj1.phi).toHaveLength(16);
    expect(proj1.tRepr).toHaveLength(8);
    expect(proj1.tEmb).toHaveLength(24);
    expect(Array.from(proj1.tEmb)).not.toEqual(Array.from(projQuarter.tEmb));

    // Independently verify full global MLP + tBlock projection equations element-by-element
    const silu = (v: number) => Math.fround(v / (1 + Math.exp(-v)));
    const h0Act = new Float32Array(8);
    for (let r = 0; r < 8; r++) {
      let acc = 0;
      for (let c = 0; c < 16; c++) acc += w.mlp0Weight[r * 16 + c] * proj1.phi[c];
      h0Act[r] = silu(Math.fround(acc + w.mlp0Bias[r]));
    }
    const expectedTRepr = new Float32Array(8);
    for (let r = 0; r < 8; r++) {
      let acc = 0;
      for (let c = 0; c < 8; c++) acc += w.mlp2Weight[r * 8 + c] * h0Act[c];
      expectedTRepr[r] = Math.fround(acc + w.mlp2Bias[r]);
    }
    expect(Array.from(proj1.tRepr)).toEqual(Array.from(expectedTRepr));

    const expectedTEmb = new Float32Array(24);
    for (let r = 0; r < 24; r++) {
      let acc = 0;
      for (let c = 0; c < 8; c++) acc += w.tBlockWeight[r * 8 + c] * silu(expectedTRepr[c]);
      expectedTEmb[r] = Math.fround(acc + w.tBlockBias[r]);
    }
    expect(Array.from(proj1.tEmb)).toEqual(Array.from(expectedTEmb));

    // Independently verify exact equations for layer 5
    const layer5 = computeLayerModulationParams(proj1.tEmb, w, 5);
    const layer6 = computeLayerModulationParams(proj1.tEmb, w, 6);
    expect(layer5.layerIndex).toBe(5);
    expect(layer5.shift).toHaveLength(8);
    expect(layer5.scale).toHaveLength(8);
    expect(layer5.gate).toHaveLength(8);
    for (let i = 0; i < 8; i++) {
      expect(layer5.shift[i]).toBe(Math.fround(w.scaleShiftTables[5][i] + proj1.tEmb[i]));
      expect(layer5.scale[i]).toBe(Math.fround(w.scaleShiftTables[5][8 + i] + proj1.tEmb[8 + i]));
      expect(layer5.gate[i]).toBe(Math.fround(w.scaleShiftTables[5][16 + i] + proj1.tEmb[16 + i]));
    }
    expect(Array.from(layer5.shift)).not.toEqual(Array.from(layer6.shift));

    // Verify applyAdaLnModulate exact formula on channel-major [d=3, T=2]
    const x = new Float32Array([
      1.0, -2.0, // channel 0
      0.5, 4.0, // channel 1
      -1.5, 3.0, // channel 2
    ]);
    const shift = new Float32Array([0.25, -0.5, 1.0]);
    const scale = new Float32Array([0.5, -0.25, 0.0]);
    const modulated = applyAdaLnModulate(x, 3, 2, shift, scale);
    expect(Array.from(modulated)).toEqual([
      Math.fround(1.0 * 1.5 + 0.25),
      Math.fround(-2.0 * 1.5 + 0.25),
      Math.fround(0.5 * 0.75 - 0.5),
      Math.fround(4.0 * 0.75 - 0.5),
      Math.fround(-1.5 * 1.0 + 1.0),
      Math.fround(3.0 * 1.0 + 1.0),
    ]);

    // Identity under zero shift & scale
    const zero = new Float32Array(3);
    expect(Array.from(applyAdaLnModulate(x, 3, 2, zero, zero))).toEqual(Array.from(x));

    // Non-finite inputs and bad shapes rejected
    expect(() =>
      applyAdaLnModulate(new Float32Array([Number.NaN, 0, 0, 0, 0, 0]), 3, 2, shift, scale),
    ).toThrow(RangeError);
    expect(() =>
      applyAdaLnModulate(x, 3, 2, new Float32Array([Infinity, 0, 0]), scale),
    ).toThrow(RangeError);
    expect(() =>
      applyAdaLnModulate(x, 3, 2, shift, new Float32Array([0, -Infinity, 0])),
    ).toThrow(RangeError);
    expect(() => applyAdaLnModulate(x, 3, 3, shift, scale)).toThrow(RangeError);
    expect(() => computeLayerModulationParams(proj1.tEmb, w, -1)).toThrow(RangeError);
    expect(() => computeLayerModulationParams(proj1.tEmb, w, 52)).toThrow(RangeError);
  });
});

describe('TwoTower Released Confidence-Unmasking Sampler (src/twotower/sampler.ts)', () => {
  const V = 6;
  const MASK = 4;
  const EOS = 5;

  it('enforces mask-token exclusion (prob === 0) and exact locked carry-over delta under adversarial logits', () => {
    // Position 0: masked with huge adversarial logit on MASK token!
    // Position 1: locked at token 2 with huge adversarial logit on token 0!
    const xt = [MASK, 2];
    const logits = new Float32Array([
      1.0, 2.0, 0.5, 0.0, 1e6, -1.0, // pos 0 (masked): MASK logit = +1e6 must be suppressed!
      1e6, 1e6, -50.0, 1e6, 1e6, 1e6, // pos 1 (locked=2): must stay 1.0 on token 2!
    ]);

    const { logXTheta, xTheta } = mdlmConstrainLogProbs(logits, xt, V, MASK);

    // Position 0: mask probability strictly 0, non-mask probabilities sum to 1
    expect(xTheta[0 * V + MASK]).toBe(0);
    expect(logXTheta[0 * V + MASK]).toBeLessThan(-1e11);
    let sum0 = 0;
    for (let v = 0; v < V; v++) sum0 += xTheta[0 * V + v];
    expect(sum0).toBeCloseTo(1.0, 5);

    // Position 1: exact carry-over delta on token 2
    for (let v = 0; v < V; v++) {
      if (v === 2) {
        expect(logXTheta[1 * V + v]).toBe(0);
        expect(xTheta[1 * V + v]).toBe(1);
      } else {
        expect(logXTheta[1 * V + v]).toBe(Math.fround(MDLM_MASK_SUPPRESS_LOGIT));
        expect(xTheta[1 * V + v]).toBe(0);
      }
    }

    // Greedy prediction picks token 1 for pos 0 and keeps locked token 2 for pos 1
    const greedy = sampleBlockPredictions(logits, xt, V, MASK, 0.0);
    expect(greedy.predicted).toEqual([1, 2]);

    // Even under high-temperature Gumbel-max with extreme variates, locked pos 1 NEVER changes
    const uniforms = new Float32Array([
      0.5, 0.5, 0.5, 0.5, 0.999999, 0.5,
      0.999999, 0.999999, 0.0, 0.999999, 0.999999, 0.999999,
    ]);
    const sampled = sampleBlockPredictions(logits, xt, V, MASK, 1.5, uniforms);
    expect(sampled.predicted[0]).not.toBe(MASK);
    expect(sampled.predicted[1]).toBe(2);

    // Non-finite logits and invalid uniform variates rejected
    expect(() =>
      mdlmConstrainLogProbs(new Float32Array([Number.NaN, 0, 0, 0, 0, 0]), [MASK], V, MASK),
    ).toThrow(RangeError);
    expect(() => sampleBlockPredictions(logits, xt, V, MASK, 0.5)).toThrow(RangeError);
    expect(() =>
      sampleBlockPredictions(logits, xt, V, MASK, 0.5, new Float32Array(12).fill(1.0)),
    ).toThrow(RangeError);
    expect(() =>
      sampleBlockPredictions(logits, xt, V, MASK, 0.5, new Float32Array(12).fill(-0.01)),
    ).toThrow(RangeError);
  });

  it('scores confidence strictly from UNSCALED xTheta even when temperature > 0 Gumbel-max picks another token', () => {
    const xt = [MASK, 3];
    // Position 0: token 0 has high logit (3.0), token 1 has low logit (0.0).
    // With temperature = 2.0 and huge Gumbel noise on token 1 (u = 0.99999), Gumbel-max picks token 1,
    // yet confidence[0] MUST equal unscaled xTheta[0, 1] (NOT the temperature-scaled probability)!
    const logits = new Float32Array([
      3.0, 0.0, -2.0, -2.0, 50.0, -2.0,
      1e5, 1e5, 1e5, -10.0, 1e5, 1e5,
    ]);
    const uniforms = new Float32Array([
      0.001, 0.99999, 0.001, 0.001, 0.99999, 0.001,
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ]);

    const res = runConfidenceUnmaskStep({
      xt,
      logits,
      vocabSize: V,
      maskTokenId: MASK,
      stepIndex: 0,
      stepsPerBlock: 2,
      confidenceThreshold: 0.9,
      temperature: 2.0,
      uniformVariates: uniforms,
    });

    expect(res.predicted[0]).toBe(1);
    expect(res.predicted[1]).toBe(3);

    // Independently verify unscaled xTheta[0, 1] vs temperature-scaled prob
    const expUnscaled = [Math.exp(3), Math.exp(0), Math.exp(-2), Math.exp(-2), 0, Math.exp(-2)];
    const sumUnscaled = expUnscaled.reduce((a, b) => a + b, 0);
    const expectedUnscaledProb = Math.fround(expUnscaled[1] / sumUnscaled);

    const expScaled = [Math.exp(1.5), Math.exp(0), Math.exp(-1), Math.exp(-1), 0, Math.exp(-1)];
    const sumScaled = expScaled.reduce((a, b) => a + b, 0);
    const scaledProb = expScaled[1] / sumScaled;

    expect(res.confidence[0]).toBeCloseTo(expectedUnscaledProb, 6);
    expect(Math.abs(res.confidence[0] - scaledProb)).toBeGreaterThan(0.05);
    expect(res.confidence[1]).toBe(Infinity);
    expect(res.xtNext[1]).toBe(3);
    expect(GUMBEL_UNIFORM_MIN).toBe(1e-10);
  });

  it('enforces strict ">" threshold comparison (conf === threshold does NOT count as above)', () => {
    // 5 tokens total (4 non-mask + 1 mask): equal logits across the 4 non-mask tokens =>
    // exact Float32 probability 0.25 for every non-mask token!
    const vocab = 5;
    const maskId = 4;
    const xt = [maskId, maskId, maskId, maskId];
    const logits = new Float32Array(4 * vocab).fill(0);

    // With threshold === 0.25 exact: strict '>' yields numAboveThreshold === 0 => progress floor commits 1!
    const eqRes = runConfidenceUnmaskStep({
      xt,
      logits,
      vocabSize: vocab,
      maskTokenId: maskId,
      stepIndex: 0,
      stepsPerBlock: 4,
      confidenceThreshold: 0.25,
      temperature: 0.0,
    });
    expect(eqRes.confidence[0]).toBe(0.25);
    expect(eqRes.numAboveThreshold).toBe(0);
    expect(eqRes.tokensCommittedCount).toBe(1);
    expect(eqRes.commitReason).toBe('progress_floor');
    expect(eqRes.thresholdCommittedIndices).toEqual([]);
    expect(eqRes.floorCommittedIndices).toHaveLength(1);
    expect(eqRes.remaskedIndices).toHaveLength(3);

    // With threshold slightly below 0.25: all 4 strictly exceed threshold => commits all 4 early!
    const belowRes = runConfidenceUnmaskStep({
      xt,
      logits,
      vocabSize: vocab,
      maskTokenId: maskId,
      stepIndex: 0,
      stepsPerBlock: 4,
      confidenceThreshold: 0.249999,
      temperature: 0.0,
    });
    expect(belowRes.numAboveThreshold).toBe(4);
    expect(belowRes.tokensCommittedCount).toBe(4);
    expect(belowRes.commitReason).toBe('threshold');
    expect(belowRes.thresholdCommittedIndices).toEqual([0, 1, 2, 3]);
    expect(belowRes.floorCommittedIndices).toEqual([]);
    expect(belowRes.remaskedIndices).toEqual([]);
    expect(belowRes.earlyCompleted).toBe(true);
    expect(belowRes.isBlockComplete).toBe(true);
  });

  it('enforces remaining-budget floor ceil(nMasked / remainingSteps) for both K < B and K > B', () => {
    // Build logits where every slot has distinct low confidence (< 0.99 threshold) so floor governs!
    const makeBlockLogits = (B: number) => {
      const logits = new Float32Array(B * V);
      for (let j = 0; j < B; j++) {
        // Increasing margin with j so higher slot index has strictly higher confidence
        logits[j * V + (j % 4)] = 0.5 + 0.15 * j;
      }
      return logits;
    };

    // Case 1: K < B (B = 8, K = 3) -> must commit ceil(8/3)=3, ceil(5/2)=3, final 2
    let xt: readonly number[] = new Array(8).fill(MASK);
    const logits8 = makeBlockLogits(8);

    const s0 = runConfidenceUnmaskStep({
      xt,
      logits: logits8,
      vocabSize: V,
      maskTokenId: MASK,
      stepIndex: 0,
      stepsPerBlock: 3,
      confidenceThreshold: 0.99,
      temperature: 0.0,
    });
    expect(s0.tModel).toBe(1.0);
    expect(s0.minCommitFloor).toBe(3);
    expect(s0.tokensCommittedCount).toBe(3);
    expect(s0.commitReason).toBe('budget_floor');
    expect(s0.committedIndices).toEqual([5, 6, 7]); // top-3 highest-confidence slots
    expect(s0.remaskedIndices).toEqual([0, 1, 2, 3, 4]);
    expect(s0.nMaskedAfter).toBe(5);
    xt = s0.xtNext;

    const s1 = runConfidenceUnmaskStep({
      xt,
      logits: logits8,
      vocabSize: V,
      maskTokenId: MASK,
      stepIndex: 1,
      stepsPerBlock: 3,
      confidenceThreshold: 0.99,
      temperature: 0.0,
    });
    expect(s1.tModel).toBe(5 / 8);
    expect(s1.minCommitFloor).toBe(3); // ceil(5 / 2) = 3
    expect(s1.tokensCommittedCount).toBe(3);
    expect(s1.commitReason).toBe('budget_floor');
    expect(s1.committedIndices).toEqual([2, 3, 4]);
    expect(s1.nMaskedAfter).toBe(2);
    // Previously committed slots [5, 6, 7] stay untouched
    expect(s1.xtNext.slice(5)).toEqual(s0.xtNext.slice(5));
    xt = s1.xtNext;

    const s2 = runConfidenceUnmaskStep({
      xt,
      logits: logits8,
      vocabSize: V,
      maskTokenId: MASK,
      stepIndex: 2,
      stepsPerBlock: 3,
      confidenceThreshold: 0.99,
      temperature: 0.0,
    });
    expect(s2.commitReason).toBe('final_step');
    expect(s2.tokensCommittedCount).toBe(2);
    expect(s2.committedIndices).toEqual([0, 1]);
    expect(s2.nMaskedAfter).toBe(0);
    expect(s2.isBlockComplete).toBe(true);
    expect(s2.earlyCompleted).toBe(false); // finished on final step, not early

    // Case 2: K > B (B = 4, K = 6) -> ceil(4/6)=1, progress floor advances 1/step and finishes early at step 3!
    let xt4: readonly number[] = [MASK, MASK, MASK, MASK];
    const logits4 = makeBlockLogits(4);
    for (let step = 0; step < 4; step++) {
      const res = runConfidenceUnmaskStep({
        xt: xt4,
        logits: logits4,
        vocabSize: V,
        maskTokenId: MASK,
        stepIndex: step,
        stepsPerBlock: 6,
        confidenceThreshold: 0.99,
        temperature: 0.0,
      });
      expect(res.minCommitFloor).toBe(1);
      expect(res.tokensCommittedCount).toBe(1);
      expect(res.commitReason).toBe('progress_floor');
      expect(res.earlyCompleted).toBe(step === 3);
      xt4 = res.xtNext;
    }

    // Calling step 4 when nMasked === 0 returns early-exit no-op without mutating xt4
    const noop = runConfidenceUnmaskStep({
      xt: xt4,
      logits: logits4,
      vocabSize: V,
      maskTokenId: MASK,
      stepIndex: 4,
      stepsPerBlock: 6,
      confidenceThreshold: 0.99,
      temperature: 0.5, // even without uniformVariates, early-exit no-op succeeds cleanly
    });
    expect(noop.earlyCompleted).toBe(true);
    expect(noop.isBlockComplete).toBe(true);
    expect(noop.commitReason).toBe('early_completed');
    expect(noop.tModel).toBe(0);
    expect(noop.nMaskedBefore).toBe(0);
    expect(noop.nMaskedAfter).toBe(0);
    expect(noop.tokensCommittedCount).toBe(0);
    expect(noop.xtNext).toEqual(xt4);
    expect(Array.from(noop.confidence)).toEqual([Infinity, Infinity, Infinity, Infinity]);
  });

  it('detects EOS across full committed block for both scalar and array EOS token IDs', () => {
    expect(blockContainsEos([1, 2, 3, 5], EOS)).toBe(true);
    expect(blockContainsEos([5, 0, 1, 2], EOS)).toBe(true); // EOS at slot 0, full block still inspected
    expect(blockContainsEos([0, 1, 2, 3], EOS)).toBe(false);
    expect(blockContainsEos([0, 1, 11, 3], [2, 11])).toBe(true);
    expect(blockContainsEos([0, 1, 2, 3], [2, 11])).toBe(true);
    expect(blockContainsEos([0, 1, 4, 3], [2, 11])).toBe(false);
    expect(blockContainsEos([0, 1, 2, 3], [])).toBe(false);

    // Check result.containsEos convenience helper after committing an EOS token in slot 1 (unshifted!)
    const logits = new Float32Array(4 * V);
    logits[0 * V + 1] = 8.0; // slot 0 -> token 1
    logits[1 * V + EOS] = 8.0; // slot 1 -> EOS (unshifted!)
    logits[2 * V + 2] = 8.0; // slot 2 -> token 2
    logits[3 * V + 3] = 8.0; // slot 3 -> token 3

    const res = runConfidenceUnmaskStep({
      xt: [MASK, MASK, MASK, MASK],
      logits,
      vocabSize: V,
      maskTokenId: MASK,
      stepIndex: 0,
      stepsPerBlock: 4,
      confidenceThreshold: 0.8,
      temperature: 0.0,
    });
    expect(res.xtNext).toEqual([1, EOS, 2, 3]);
    expect(res.containsEos(EOS)).toBe(true);
    expect(res.containsEos([0, EOS])).toBe(true);
    expect(res.containsEos(0)).toBe(false);
  });

  it('uses deterministic lower-index tie-breaking in greedy argmax and rejects temperature overflow', () => {
    // Exact logit tie between token 1 and token 3 at pos 0 => lower index (1) wins deterministically
    const logits = new Float32Array([0.0, 2.0, 0.0, 2.0, 5.0, 0.0]);
    const greedy = sampleBlockPredictions(logits, [MASK], V, MASK, 0.0);
    expect(greedy.predicted).toEqual([1]);

    // Extreme subnormal temperature causing logit / T overflow to Infinity must throw RangeError
    expect(() =>
      sampleBlockPredictions(
        new Float32Array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        [MASK],
        V,
        MASK,
        1e-320,
        new Float32Array(V).fill(0.5),
      ),
    ).toThrow(RangeError);
  });
});

describe('TwoTower Headless & Determinism Guardrails (src/twotower/{spec,modulation,sampler}.ts)', () => {
  it('keeps all Stage TT1b-1 modules strictly headless with zero DOM/WebGL/scene/render imports or Math.random', () => {
    const files = ['src/twotower/spec.ts', 'src/twotower/modulation.ts', 'src/twotower/sampler.ts'];
    for (const rel of files) {
      const src = readFileSync(resolve(__dirname, '..', rel), 'utf-8');
      const codeOnly = src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
      expect(codeOnly, `${rel} must not import scene or render modules`).not.toMatch(
        /from\s+['"][^'"]*(?:\/scene\/|\/render\/)/,
      );
      expect(codeOnly, `${rel} must not reference DOM or WebGL APIs`).not.toMatch(
        /\b(?:window|document|HTMLElement|WebGL2RenderingContext|HTMLCanvasElement)\b/,
      );
      expect(codeOnly, `${rel} must not call Math.random()`).not.toMatch(/Math\.random\s*\(/);
    }
  });
});
