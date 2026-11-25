import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { MINI_VOCAB } from '../src/math/hybridFlow';
import {
  DEFAULT_TWOTOWER_PERTURBATION,
  TWOTOWER_MINI_EOS_TOKEN_ID,
  TWOTOWER_MINI_MASK_TOKEN_ID,
  TWOTOWER_MINI_VOCAB,
  TWOTOWER_SPEC,
  assertNonAliasedWithContextSnapshot,
  buildContextConditioningState,
  collectTwoTowerCanonicalWeights,
  commitTwoTowerBlockToContext,
  computeConditioningDigest,
  createDefaultTwoTowerModel,
  createTwoTowerModelParameters,
  generateTwoTowerDiffusion,
  runTwoTowerArStep,
  runTwoTowerContextPrefill,
  runTwoTowerDenoiserPass,
  runTwoTowerDiffusionBlock,
  runTwoTowerMockArStep,
  validateContextConditioningState,
  validateContextSnapshot,
  type TwoTowerContextConditioningState,
  type TwoTowerContextSnapshot,
} from '../src/twotower';
import { assertBitExactFiniteArray } from './helpers/parityAssert';

interface SnapshotDeepCopy {
  readonly stateDigest: string;
  readonly snapshotDigest: string;
  readonly processedTokens: number;
  readonly prefixTokenIds: Int32Array;
  readonly arLogits: Float32Array;
  readonly arProbs: Float32Array;
  readonly mambaConv: Map<number, Float32Array>;
  readonly mambaSsm: Map<number, Float32Array>;
  readonly mambaS2Conv: Map<number, Float32Array>;
  readonly mambaS2Ssm: Map<number, Float32Array>;
  readonly gqaK: Map<number, Float32Array>;
  readonly gqaV: Map<number, Float32Array>;
}

function captureSnapshotDeepCopy(snap: TwoTowerContextSnapshot): SnapshotDeepCopy {
  const mambaConv = new Map<number, Float32Array>();
  const mambaSsm = new Map<number, Float32Array>();
  for (const [bayId, m] of snap.ctxState.mambaStates.entries()) {
    mambaConv.set(bayId, new Float32Array(m.convState));
    mambaSsm.set(bayId, new Float32Array(m.ssmState));
  }
  const mambaS2Conv = new Map<number, Float32Array>();
  const mambaS2Ssm = new Map<number, Float32Array>();
  for (const [bayId, m] of snap.mambaS2.entries()) {
    mambaS2Conv.set(bayId, new Float32Array(m.convState));
    mambaS2Ssm.set(bayId, new Float32Array(m.ssmState));
  }
  const gqaK = new Map<number, Float32Array>();
  const gqaV = new Map<number, Float32Array>();
  for (const [bayId, g] of snap.ctxState.gqaCaches.entries()) {
    gqaK.set(bayId, new Float32Array(g.K));
    gqaV.set(bayId, new Float32Array(g.V));
  }
  return {
    stateDigest: snap.ctxState.stateDigest,
    snapshotDigest: snap.snapshotDigest,
    processedTokens: snap.ctxState.processedTokens,
    prefixTokenIds: new Int32Array(snap.ctxState.prefixTokenIds),
    arLogits: new Float32Array(snap.arLogits),
    arProbs: new Float32Array(snap.arProbs),
    mambaConv,
    mambaSsm,
    mambaS2Conv,
    mambaS2Ssm,
    gqaK,
    gqaV,
  };
}

function assertSnapshotBitExact(
  actual: TwoTowerContextSnapshot,
  expected: SnapshotDeepCopy,
  label: string,
): void {
  expect(actual.ctxState.stateDigest, `${label}.stateDigest`).toBe(expected.stateDigest);
  expect(actual.snapshotDigest, `${label}.snapshotDigest`).toBe(expected.snapshotDigest);
  expect(actual.ctxState.processedTokens, `${label}.processedTokens`).toBe(expected.processedTokens);
  expect(Array.from(actual.ctxState.prefixTokenIds)).toEqual(Array.from(expected.prefixTokenIds));
  assertBitExactFiniteArray(actual.arLogits, expected.arLogits, `${label}.arLogits`);
  assertBitExactFiniteArray(actual.arProbs, expected.arProbs, `${label}.arProbs`);

  for (const [bayId, expConv] of expected.mambaConv.entries()) {
    const m = actual.ctxState.mambaStates.get(bayId)!;
    assertBitExactFiniteArray(m.convState, expConv, `${label}.mamba[${bayId}].convState`);
    assertBitExactFiniteArray(m.ssmState, expected.mambaSsm.get(bayId)!, `${label}.mamba[${bayId}].ssmState`);
  }
  for (const [bayId, expConv] of expected.mambaS2Conv.entries()) {
    const m = actual.mambaS2.get(bayId)!;
    assertBitExactFiniteArray(m.convState, expConv, `${label}.mambaS2[${bayId}].convState`);
    assertBitExactFiniteArray(m.ssmState, expected.mambaS2Ssm.get(bayId)!, `${label}.mambaS2[${bayId}].ssmState`);
  }
  for (const [bayId, expK] of expected.gqaK.entries()) {
    const g = actual.ctxState.gqaCaches.get(bayId)!;
    assertBitExactFiniteArray(g.K, expK, `${label}.gqa[${bayId}].K`);
    assertBitExactFiniteArray(g.V, expected.gqaV.get(bayId)!, `${label}.gqa[${bayId}].V`);
  }
}

describe('Stage TT1 Wave 2: TwoTower Parameters, Read-Only Conditioning Contract & Headless Engine', () => {
  describe('1. Independent Miniature Parameters, Non-Aliasing, Fingerprints & Canonical Weights', () => {
    it('constructs strictly non-aliased Context, Denoiser & TimeConditioning weights with exact canonical inventory', () => {
      expect(TWOTOWER_MINI_VOCAB).toHaveLength(34);
      expect(TWOTOWER_MINI_VOCAB.slice(0, 32)).toEqual(MINI_VOCAB);
      expect(TWOTOWER_MINI_VOCAB[TWOTOWER_MINI_MASK_TOKEN_ID]).toBe('[MASK]');
      expect(TWOTOWER_MINI_VOCAB[TWOTOWER_MINI_EOS_TOKEN_ID]).toBe('<eos>');
      expect(DEFAULT_TWOTOWER_PERTURBATION).toEqual({ seed: 20260701, scale: 0.15 });

      const model = createDefaultTwoTowerModel();
      expect(model.vocab).toBe(TWOTOWER_MINI_VOCAB);
      expect(model.maskTokenId).toBe(32);
      expect(model.eosTokenId).toBe(33);
      expect(model.contextModel.config.vocabSize).toBe(34);
      expect(model.denoiserModel.config.vocabSize).toBe(34);
      expect(model.contextModel.config.routerPrecisionPolicy).toBe('float64_reference');
      expect(model.denoiserModel.config.routerPrecisionPolicy).toBe('float64_reference');
      expect(model.contextModel.config.generationPolicy).toBe('twotower-context-v1');
      expect(model.denoiserModel.config.generationPolicy).toBe('twotower-denoiser-v1');
      expect(model.denoiserModel.config.perturbation).toEqual({
        seed: 20260701,
        scale: 0.15,
        sourceFingerprint: model.contextModel.parameterFingerprint,
      });

      // Context and Denoiser fingerprints and model IDs must be strictly distinct
      expect(model.contextModel.parameterFingerprint).not.toBe(
        model.denoiserModel.parameterFingerprint,
      );
      expect(model.contextModel.modelId).not.toBe(model.denoiserModel.modelId);

      // Canonical weight collection: 401 context + 401 denoiser + 58 time_cond = 860 tensors
      const canonical = collectTwoTowerCanonicalWeights(model);
      expect(canonical).toHaveLength(860);
      expect(canonical[0].name).toBe('context.W_embed');
      expect(canonical[1].name).toBe('context.finalNormGamma');
      expect(canonical[2].name).toBe('context.W_lm_head');
      expect(canonical[3].name).toBe('context.bay_01.gamma_in');
      expect(canonical[401].name).toBe('denoiser.W_embed');
      expect(canonical[802].name).toBe('time_cond.t_embedder.mlp.0.weight');
      expect(canonical[859].name).toBe('time_cond.scale_shift_tables.51');

      // Every single tensor across Context, Denoiser & TimeConditioning must own a unique ArrayBuffer
      const seenBuffers = new Set<ArrayBufferLike>();
      const seenNames = new Set<string>();
      let canonicalBytes = 0;
      for (const t of canonical) {
        expect(seenNames.has(t.name), `Duplicate canonical tensor name ${t.name}`).toBe(false);
        seenNames.add(t.name);
        expect(seenBuffers.has(t.data.buffer), `Buffer aliasing detected at ${t.name}`).toBe(false);
        seenBuffers.add(t.data.buffer);
        canonicalBytes += t.data.byteLength;
      }

      // All 52 bays materialized -> retainedWeightBytes() matches exact canonical byte sum
      expect(model.retainedWeightBytes()).toBe(canonicalBytes);
      expect(canonicalBytes).toBeGreaterThan(2_200_000);

      // Default perturbationScale = 0.15 produces genuinely distinct Denoiser weights vs Context
      let maxEmbedDiff = 0;
      for (let i = 0; i < model.contextModel.W_embed.length; i++) {
        maxEmbedDiff = Math.max(
          maxEmbedDiff,
          Math.abs(model.contextModel.W_embed[i] - model.denoiserModel.W_embed[i]),
        );
      }
      expect(maxEmbedDiff).toBeGreaterThan(0.05);
      expect(maxEmbedDiff).toBeLessThanOrEqual(0.150001);

      // Identity-weight configuration (perturbationScale = 0) keeps strictly distinct buffers
      // with bit-identical weight values AND still distinct tower parameterFingerprints!
      const idModel = createTwoTowerModelParameters({ perturbationScale: 0 });
      expect(idModel.denoiserModel.config.perturbation).toBeUndefined();
      expect(idModel.contextModel.W_embed).not.toBe(idModel.denoiserModel.W_embed);
      expect(idModel.contextModel.W_embed.buffer).not.toBe(idModel.denoiserModel.W_embed.buffer);
      assertBitExactFiniteArray(
        idModel.denoiserModel.W_embed,
        idModel.contextModel.W_embed,
        'identity W_embed',
      );
      expect(idModel.contextModel.parameterFingerprint).not.toBe(
        idModel.denoiserModel.parameterFingerprint,
      );

      // Verify local 24-shard safetensors header audit summary recorded in spec metadata (zero weight data)
      const audit = TWOTOWER_SPEC.directlyInspectedMetadata.localCheckpointHeaderAudit;
      expect(audit.verifiedShardCount).toBe(24);
      expect(audit.verifiedTensorCount).toBe(12_544);
      expect(audit.verifiedTotalParams).toBe(63_185_902_848);
      expect(audit.verifiedTensorPayloadBytes + audit.verifiedHeaderBytes).toBe(
        audit.verifiedDiskBytes,
      );
      expect(audit.bf16TensorCount + audit.fp32TensorCount).toBe(audit.verifiedTensorCount);
      expect(audit.fp32TensorCount).toBe(46);
      expect(audit.fp32TensorScope).toBe('e_score_correction_bias_only');
      expect(audit.integrityErrorCount).toBe(0);
    });

    it('varies twoTowerFingerprint deterministically on every parameter and perturbation knob', () => {
      const mBase = createTwoTowerModelParameters();
      const mRepeat = createTwoTowerModelParameters();
      expect(mBase.twoTowerFingerprint).toBe(mRepeat.twoTowerFingerprint);
      expect(mBase.twoTowerFingerprint).toHaveLength(16);

      const variants = [
        createTwoTowerModelParameters({ seed: 0x12345678 }),
        createTwoTowerModelParameters({ perturbationSeed: 20260702 }),
        createTwoTowerModelParameters({ perturbationScale: 0.25 }),
        createTwoTowerModelParameters({ perturbationScale: 0 }),
        createTwoTowerModelParameters({ timeConditioningSeed: 0x54543032 }),
        createTwoTowerModelParameters({ maskTokenId: 31, eosTokenId: 33 }),
        createTwoTowerModelParameters({ denoiserGenerationPolicy: 'twotower-denoiser-v2' }),
      ];

      const fps = new Set([mBase.twoTowerFingerprint, ...variants.map((m) => m.twoTowerFingerprint)]);
      expect(fps.size).toBe(1 + variants.length);

      // Invalid parameter options rejected
      expect(() => createTwoTowerModelParameters({ perturbationScale: -0.1 })).toThrow(RangeError);
      expect(() => createTwoTowerModelParameters({ maskTokenId: 33, eosTokenId: 33 })).toThrow(
        RangeError,
      );
      expect(() =>
        createTwoTowerModelParameters({
          contextGenerationPolicy: 'same-policy',
          denoiserGenerationPolicy: 'same-policy',
        }),
      ).toThrow(RangeError);
    });
  });

  describe('2. Read-Only Cross-Tower Conditioning Contract Integrity & Security Checks', () => {
    it('enforces ownership roles, 29-layer alignment, non-aliasing, digest integrity, capacity, and queryLen limits', () => {
      const model = createDefaultTwoTowerModel();
      const otherModel = createTwoTowerModelParameters({ perturbationScale: 0.25 });
      const snap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });

      const cond = buildContextConditioningState(model, snap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 0,
      });

      expect(cond.ownerRole).toBe('context_tower');
      expect(cond.consumerRole).toBe('denoiser_tower');
      expect(cond.mode).toBe('mask_diffusion');
      expect(cond.committedPrefixLen).toBe(8);
      expect(cond.sourceFullPrefixLen).toBe(8);
      expect(cond.absoluteTokenStart).toBe(8);
      expect(cond.allocatedKvCapacity).toBe(24);
      expect(cond.layerAlignment).toHaveLength(29);
      expect(cond.gqaPrefixCaches.size).toBe(6);
      expect(cond.mambaSeedStates.size).toBe(23);
      expect(cond.layerAlignment.some((e) => (e.kind as string) === 'E')).toBe(false);

      expect(() => validateContextConditioningState(cond, model, 4)).not.toThrow();
      expect(() => assertNonAliasedWithContextSnapshot(cond, snap)).not.toThrow();

      // Trailing KV slots [8..23] strictly zeroed
      const g6 = cond.gqaPrefixCaches.get(6)!;
      for (let row = 0; row < g6.nKv * g6.dHead; row++) {
        for (let k = 8; k < 24; k++) {
          expect(g6.K[row * 24 + k]).toBe(0);
          expect(g6.V[row * 24 + k]).toBe(0);
        }
      }

      // Reject relabeling Context-owned cache as Denoiser-owned
      const relabeled = {
        ...cond,
        ownerRole: 'denoiser_tower' as unknown as 'context_tower',
      };
      expect(() => validateContextConditioningState(relabeled, model, 4)).toThrow(TypeError);

      // Reject wrong model fingerprint (different denoiser perturbation)
      expect(() => validateContextConditioningState(cond, otherModel, 4)).toThrow(TypeError);

      // Reject tampered conditioningDigest
      const badDigest: TwoTowerContextConditioningState = {
        ...cond,
        conditioningDigest: '0000000000000000',
      };
      expect(() => validateContextConditioningState(badDigest, model, 4)).toThrow();

      // Reject tampered sourceStateDigest
      const badSourceDigest: TwoTowerContextConditioningState = {
        ...cond,
        sourceStateDigest: 'deadbeefdeadbeef',
      };
      expect(() => validateContextConditioningState(badSourceDigest, model, 4)).toThrow();

      // Reject silent tensor mutation (caught by digest AND trailing-zero check)
      const condCopy = buildContextConditioningState(model, snap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 1,
      });
      condCopy.gqaPrefixCaches.get(6)!.K[0] += 0.125;
      expect(() => validateContextConditioningState(condCopy, model, 4)).toThrow();

      // Reject unzeroed trailing KV slot even if digest is recomputed!
      const condTrail = buildContextConditioningState(model, snap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 2,
      });
      condTrail.gqaPrefixCaches.get(6)!.K[8] = 0.5;
      const condTrailRehashed: TwoTowerContextConditioningState = {
        ...condTrail,
        conditioningDigest: computeConditioningDigest(condTrail),
      };
      expect(() => validateContextConditioningState(condTrailRehashed, model, 4)).toThrow(
        RangeError,
      );

      // Reject non-finite tensor value even if digest is recomputed
      const condNan = buildContextConditioningState(model, snap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 3,
      });
      condNan.mambaSeedStates.get(1)!.ssmState[0] = Number.NaN;
      const condNanRehashed: TwoTowerContextConditioningState = {
        ...condNan,
        conditioningDigest: computeConditioningDigest(condNan),
      };
      expect(() => validateContextConditioningState(condNanRehashed, model, 4)).toThrow(RangeError);

      // Reject cross-tower buffer aliasing explicitly
      const aliasedGqa = new Map(cond.gqaPrefixCaches);
      const liveGqa6 = snap.ctxState.gqaCaches.get(6)!;
      aliasedGqa.set(6, {
        ...cond.gqaPrefixCaches.get(6)!,
        K: liveGqa6.K,
      });
      const aliasedCond: TwoTowerContextConditioningState = {
        ...cond,
        gqaPrefixCaches: aliasedGqa,
      };
      expect(() => assertNonAliasedWithContextSnapshot(aliasedCond, snap)).toThrow(TypeError);

      // Reject capacity overflow (P + B > kvCapacity)
      const tightSnap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], {
        kvCapacity: 10,
      });
      const tightCond = buildContextConditioningState(model, tightSnap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 0,
      });
      expect(() => validateContextConditioningState(tightCond, model, 2)).not.toThrow();
      expect(() => validateContextConditioningState(tightCond, model, 3)).toThrow(RangeError);
      expect(() =>
        runTwoTowerDenoiserPass(model, tightCond, [32, 32, 32]),
      ).toThrow(RangeError);

      // Reject queryLen > 16 even when kvCapacity is large (e.g. 32)
      const wideSnap = runTwoTowerContextPrefill(model, [0, 1, 2, 3], { kvCapacity: 32 });
      const wideCond = buildContextConditioningState(model, wideSnap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 0,
      });
      expect(() => validateContextConditioningState(wideCond, model, 17)).toThrow(RangeError);
      expect(() =>
        runTwoTowerDenoiserPass(model, wideCond, new Array(17).fill(TWOTOWER_MINI_MASK_TOKEN_ID)),
      ).toThrow(RangeError);

      // Reject mock_ar queryLen !== 1 or tOverride in mock_ar
      const mockCond = buildContextConditioningState(model, snap, {
        mode: 'mock_ar',
        blockId: 0,
        refinementIndex: 0,
      });
      expect(mockCond.committedPrefixLen).toBe(7);
      expect(mockCond.absoluteTokenStart).toBe(7);
      expect(() => validateContextConditioningState(mockCond, model, 2)).toThrow(RangeError);
      expect(() =>
        runTwoTowerDenoiserPass(model, mockCond, [snap.lastTokenId], { tOverride: 0.5 }),
      ).toThrow(RangeError);
    });
  });

  describe('3. Zero-ULP Context Immutability & Scratchpad Isolation Across Refinements', () => {
    it('preserves Context snapshot at 0-ULP across 4 denoising refinements and isolates external buffer mutations', () => {
      const model = createDefaultTwoTowerModel();
      const snap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });
      const baseline = captureSnapshotDeepCopy(snap);

      // Force all 4 refinements (high threshold -> 1 token committed per step via progress floor)
      const block = runTwoTowerDiffusionBlock(model, snap, {
        blockId: 0,
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        temperature: 0.0,
        commitToContext: true,
      });

      expect(block.nfe).toBe(4);
      expect(block.refinements).toHaveLength(4);
      expect(block.contextCommitted).toBe(true);
      expect(block.nextSnapshot.ctxState.processedTokens).toBe(12);

      // 1. Original Context snapshot is 0-ULP identical after all 4 refinements AND block commit!
      assertSnapshotBitExact(snap, baseline, 'contextSnapshot after 4 refinements + commit');
      expect(() => validateContextSnapshot(snap, model.contextModel)).not.toThrow();

      // 2. Capture clean lazy trace of non-eager bay 15 & bay 20 from refinement 0 BEFORE mutating exposed buffers
      const ref0CleanMamba15 = new Float32Array(
        block.refinements[0].denoiserPass.getMambaTrace(15).Y,
      );
      const ref0CleanGqa20 = new Float32Array(
        block.refinements[0].denoiserPass.getGqaTrace(20).Y,
      );

      // Run a second fresh pass for refinement 0, mutate its exposed conditioning & residual buffers
      // BEFORE querying lazy non-eager bays -> lazy reconstruction MUST remain 0-ULP identical!
      const freshCond0 = buildContextConditioningState(model, snap, {
        mode: 'mask_diffusion',
        blockId: 0,
        refinementIndex: 0,
      });
      const freshPass0 = runTwoTowerDenoiserPass(
        model,
        freshCond0,
        [32, 32, 32, 32],
        { retainedTraceBays: [1, 6] },
      );

      // Mutate exposed conditioning buffers, exposed residualStreams, and eager trace output
      freshCond0.gqaPrefixCaches.get(20)!.K.fill(9999);
      freshCond0.mambaSeedStates.get(15)!.ssmState.fill(-9999);
      freshPass0.residualStreams[14].fill(7777);
      freshPass0.residualStreams[19].fill(-7777);
      freshPass0.getMambaTrace(1).Y.fill(5555);

      // Context snapshot STILL 0-ULP identical!
      assertSnapshotBitExact(snap, baseline, 'contextSnapshot after external mutations');
      // Lazy reconstruction of bay 15 and bay 20 STILL 0-ULP identical to clean pass!
      assertBitExactFiniteArray(
        freshPass0.getMambaTrace(15).Y,
        ref0CleanMamba15,
        'isolated lazy Mamba bay 15',
      );
      assertBitExactFiniteArray(
        freshPass0.getGqaTrace(20).Y,
        ref0CleanGqa20,
        'isolated lazy GQA bay 20',
      );
    });

    it('seeds every refinement r = 0..3 freshly from committed Context boundary (never from r - 1 scratchpad) and enforces split adaLN ordering', () => {
      const model = createDefaultTwoTowerModel();
      const snap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });

      const block = runTwoTowerDiffusionBlock(model, snap, {
        blockId: 0,
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        temperature: 0.0,
      });

      expect(block.refinements).toHaveLength(4);
      const ctxMamba1 = snap.ctxState.mambaStates.get(1)!;
      const ctxGqa6 = snap.ctxState.gqaCaches.get(6)!;

      for (let r = 0; r < 4; r++) {
        const rec = block.refinements[r];
        const pass = rec.denoiserPass;

        expect(rec.conditioning.sourceStateDigest).toBe(snap.ctxState.stateDigest);
        expect(rec.conditioning.committedPrefixLen).toBe(8);
        expect(rec.conditioning.refinementIndex).toBe(r);
        expect(pass.tModel).toBe((4 - r) / 4);
        expect(pass.tEmb).not.toBeNull();

        // Check Mamba bay 1 seeds strictly from Context S-1 (not r-1's scratchpad H!)
        const mTrace = pass.getMambaTrace(1);
        expect(mTrace.hasInitialState).toBe(true);
        expect(mTrace.startTokenOffset).toBe(8);
        expect(mTrace.X_mod).toBeDefined(); // M: modulate -> RMSNorm
        expect(mTrace.residualGateApplied).toBe(true);
        assertBitExactFiniteArray(
          mTrace.initialConvState!,
          ctxMamba1.convState,
          `r=${r} Mamba bay 1 initialConvState === Context`,
        );
        assertBitExactFiniteArray(
          mTrace.initialSsmState!,
          ctxMamba1.ssmState,
          `r=${r} Mamba bay 1 initialSsmState === Context`,
        );

        // Scratchpad H after processing 4 block tokens genuinely differs from Context S-1
        let maxScratchDiff = 0;
        for (let i = 0; i < mTrace.H.length; i++) {
          maxScratchDiff = Math.max(maxScratchDiff, Math.abs(mTrace.H[i] - ctxMamba1.ssmState[i]));
        }
        expect(maxScratchDiff).toBeGreaterThan(1e-4);

        // If r > 0, prove r's seed !== r-1's scratchpad H
        if (r > 0) {
          const prevScratchH = block.refinements[r - 1].denoiserPass.getMambaTrace(1).H;
          let diffFromPrevScratch = 0;
          for (let i = 0; i < prevScratchH.length; i++) {
            diffFromPrevScratch = Math.max(
              diffFromPrevScratch,
              Math.abs(mTrace.initialSsmState![i] - prevScratchH[i]),
            );
          }
          expect(diffFromPrevScratch).toBeGreaterThan(1e-4);
        }

        // Check GQA bay 6: bidirectional [4 x (8+4)], prefix KV 0-ULP matches Context [0..7]
        const gTrace = pass.getGqaTrace(6);
        expect(gTrace.attentionMaskMode).toBe('bidirectional_block');
        expect(gTrace.prefixTokens).toBe(8);
        expect(gTrace.keyTokens).toBe(12);
        expect(gTrace.X_mod).toBeDefined(); // *: modulate -> RMSNorm
        expect(gTrace.residualGateApplied).toBe(true);
        // Zero -Infinity masked scores in bidirectional block attention
        for (let i = 0; i < gTrace.Scores.length; i++) {
          expect(Number.isFinite(gTrace.Scores[i])).toBe(true);
        }
        const kvRows = ctxGqa6.nKv * ctxGqa6.dHead;
        for (let row = 0; row < kvRows; row++) {
          for (let k = 0; k < 8; k++) {
            expect(gTrace.K[row * 12 + k]).toBe(ctxGqa6.K[row * 24 + k]);
            expect(gTrace.V[row * 12 + k]).toBe(ctxGqa6.V[row * 24 + k]);
          }
        }

        // Check MoE bay 2: RMSNorm -> modulate (U_mod defined) + float64_reference router
        const eTrace = pass.getMoeTrace(2);
        expect(eTrace.routerPrecisionPolicy).toBe('float64_reference');
        expect(eTrace.U_mod).toBeDefined(); // E: RMSNorm -> modulate
        expect(eTrace.residualGateApplied).toBe(true);
      }
    });
  });

  describe('4. S = 1 Prompt Prefill & Identity-Weight AR vs. Mock-AR (0-ULP Equivalence)', () => {
    it('handles S = 1 prompt cleanly across prefill, AR, Mock-AR, and diffusion without crashing', () => {
      const model = createDefaultTwoTowerModel();
      const snap1 = runTwoTowerContextPrefill(model, [7], { kvCapacity: 24 });
      expect(snap1.ctxState.processedTokens).toBe(1);
      expect(snap1.lastTokenId).toBe(7);
      expect(snap1.lastTokenLabel).toBe(TWOTOWER_MINI_VOCAB[7]);

      // mambaS2 must have all 23 bays with processedTokens === 0 and all-zero tensors
      expect(snap1.mambaS2.size).toBe(23);
      for (const m of snap1.mambaS2.values()) {
        expect(m.processedTokens).toBe(0);
        for (let i = 0; i < m.convState.length; i++) expect(m.convState[i]).toBe(0);
        for (let i = 0; i < m.ssmState.length; i++) expect(m.ssmState[i]).toBe(0);
      }

      const arStep = runTwoTowerArStep(model, snap1);
      expect(arStep.nextSnapshot.ctxState.processedTokens).toBe(2);

      const mockStep = runTwoTowerMockArStep(model, snap1);
      expect(mockStep.conditioning.committedPrefixLen).toBe(0);
      expect(mockStep.conditioning.absoluteTokenStart).toBe(0);
      expect(mockStep.denoiserPass.tModel).toBeNull();
      expect(mockStep.denoiserPass.tEmb).toBeNull();
      expect(mockStep.nextSnapshot.ctxState.processedTokens).toBe(2);

      const diffBlock = runTwoTowerDiffusionBlock(model, snap1, {
        blockId: 0,
        blockSize: 4,
        stepsPerBlock: 4,
      });
      expect(diffBlock.nextSnapshot.ctxState.processedTokens).toBe(5);
      expect(diffBlock.finalBlockTokens).toHaveLength(4);
      expect(diffBlock.finalBlockTokens.includes(TWOTOWER_MINI_MASK_TOKEN_ID)).toBe(false);
    });

    it('proves 0-ULP exact equivalence between Context AR and Mock-AR under identity weights (perturbationScale = 0, V = 34, float64_reference)', () => {
      const idModel = createTwoTowerModelParameters({
        perturbationScale: 0,
        routerPrecisionPolicy: 'float64_reference',
      });

      // Verify S = 1, S = 4, and S = 8 prompts plus 3-step rollout
      for (const prompt of [[5], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]]) {
        let arSnap = runTwoTowerContextPrefill(idModel, prompt, { kvCapacity: 24 });
        let mockSnap = runTwoTowerContextPrefill(idModel, prompt, { kvCapacity: 24 });

        for (let step = 0; step < 3; step++) {
          const arRes = runTwoTowerArStep(idModel, arSnap);
          const mockRes = runTwoTowerMockArStep(idModel, mockSnap);

          // Mock-AR bypasses all adaLN modulation and uses causal GQA
          expect(mockRes.denoiserPass.tModel).toBeNull();
          expect(mockRes.denoiserPass.tEmb).toBeNull();
          expect(mockRes.denoiserPass.getLayerModulation(1)).toBeNull();
          expect(mockRes.denoiserPass.getGqaTrace(6).attentionMaskMode).toBe('causal');
          expect(mockRes.denoiserPass.getGqaTrace(6).X_mod).toBeUndefined();
          expect(mockRes.denoiserPass.getGqaTrace(6).residualGateApplied).toBe(false);
          expect(mockRes.denoiserPass.getMambaTrace(1).X_mod).toBeUndefined();
          expect(mockRes.denoiserPass.getMambaTrace(1).residualGateApplied).toBe(false);
          expect(mockRes.denoiserPass.getMoeTrace(2).U_mod).toBeUndefined();
          expect(mockRes.denoiserPass.getMoeTrace(2).residualGateApplied).toBe(false);

          // 0-ULP bit-exact logits, probs, greedy token, and next Context snapshot!
          assertBitExactFiniteArray(
            mockRes.denoiserPass.logits,
            arRes.logits,
            `promptLen=${prompt.length} step=${step} Mock-AR vs AR logits`,
          );
          assertBitExactFiniteArray(
            mockRes.denoiserPass.probs,
            arRes.probs,
            `promptLen=${prompt.length} step=${step} Mock-AR vs AR probs`,
          );
          expect(mockRes.tokenId).toBe(arRes.tokenId);
          expect(mockRes.nextSnapshot.ctxState.stateDigest).toBe(
            arRes.nextSnapshot.ctxState.stateDigest,
          );
          expect(mockRes.nextSnapshot.snapshotDigest).toBe(arRes.nextSnapshot.snapshotDigest);

          arSnap = arRes.nextSnapshot;
          mockSnap = mockRes.nextSnapshot;
        }
      }

      // With default perturbationScale = 0.15, Mock-AR logits genuinely diverge from Context AR
      const perturbedModel = createDefaultTwoTowerModel();
      const snap = runTwoTowerContextPrefill(perturbedModel, [0, 1, 2, 3, 4, 5, 6, 7], {
        kvCapacity: 24,
      });
      const arRes = runTwoTowerArStep(perturbedModel, snap);
      const mockRes = runTwoTowerMockArStep(perturbedModel, snap);
      let maxDiff = 0;
      for (let v = 0; v < 34; v++) {
        maxDiff = Math.max(maxDiff, Math.abs(mockRes.denoiserPass.logits[v] - arRes.logits[v]));
      }
      expect(maxDiff).toBeGreaterThan(1e-3);
    });
  });

  describe('5. Block-Commit (B = 4) vs. 4 Sequential Single-Token Context AR Steps (0-ULP Exact)', () => {
    it('matches 4 sequential single-token Context commits bit-for-bit across all caches, mambaS2, logits, and subsequent Mock-AR', () => {
      const model = createDefaultTwoTowerModel();
      const snap0 = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });
      const blockTokens = [16, 17, 18, 19];

      // Path A: single B = 4 block commit
      const blockCommitted = commitTwoTowerBlockToContext(model, snap0, blockTokens);

      // Path B: 4 sequential single-token AR steps with identical tokens
      let seqSnap = snap0;
      for (const tok of blockTokens) {
        seqSnap = runTwoTowerArStep(model, seqSnap, { nextTokenId: tok }).nextSnapshot;
      }

      const seqCopy = captureSnapshotDeepCopy(seqSnap);
      assertSnapshotBitExact(blockCommitted, seqCopy, 'B=4 block commit vs 4x single-token AR');

      // Verify mambaS2 after block commit is also 0-ULP exact via a follow-up Mock-AR step!
      const mockAfterBlock = runTwoTowerMockArStep(model, blockCommitted);
      const mockAfterSeq = runTwoTowerMockArStep(model, seqSnap);
      assertBitExactFiniteArray(
        mockAfterBlock.denoiserPass.logits,
        mockAfterSeq.denoiserPass.logits,
        'Mock-AR logits after B=4 commit vs 4x single-token AR',
      );
      expect(mockAfterBlock.tokenId).toBe(mockAfterSeq.tokenId);
    });
  });

  describe('6. Multi-Block Diffusion Trajectories, Block Widths, Independent K != B, Early Exit, EOS & Provenance', () => {
    it('executes a full 2-block diffusion trajectory (P_0 = 8 -> 12 -> 16, kvCapacity = 24) and supports opt-in skipTerminalBlockCommit', () => {
      const model = createDefaultTwoTowerModel();
      const prompt = [0, 1, 2, 3, 4, 5, 6, 7];

      const gen = generateTwoTowerDiffusion(model, prompt, {
        numBlocks: 2,
        blockSize: 4,
        stepsPerBlock: 4,
        kvCapacity: 24,
        confidenceThreshold: 0.22,
        temperature: 0.0,
        eosTokenIds: [], // ensure both blocks run cleanly
      });

      expect(gen.completedBlocks).toBe(2);
      expect(gen.stoppedByEos).toBe(false);
      expect(gen.initialSnapshot.ctxState.processedTokens).toBe(8);
      expect(gen.blocks[0].contextCommitted).toBe(true);
      expect(gen.blocks[0].nextSnapshot.ctxState.processedTokens).toBe(12);
      expect(gen.blocks[1].contextCommitted).toBe(true);
      expect(gen.finalSnapshot.ctxState.processedTokens).toBe(16);
      expect(gen.generatedTokenIds).toHaveLength(8);
      expect(gen.fullSequenceTokenIds).toHaveLength(16);
      expect(gen.generatedTokenIds.includes(TWOTOWER_MINI_MASK_TOKEN_ID)).toBe(false);
      expect(gen.totalNfe).toBe(gen.blocks[0].nfe + gen.blocks[1].nfe);
      expect(gen.totalNfe).toBeGreaterThanOrEqual(2);
      expect(gen.totalNfe).toBeLessThanOrEqual(8);

      // Opt-in skipTerminalBlockCommit skips ONLY the final scheduled block's Context commit
      const genSkip = generateTwoTowerDiffusion(model, prompt, {
        numBlocks: 2,
        blockSize: 4,
        stepsPerBlock: 4,
        kvCapacity: 24,
        confidenceThreshold: 0.22,
        temperature: 0.0,
        eosTokenIds: [],
        skipTerminalBlockCommit: true,
      });
      expect(genSkip.blocks[0].contextCommitted).toBe(true);
      expect(genSkip.blocks[1].contextCommitted).toBe(false);
      expect(genSkip.finalSnapshot.ctxState.processedTokens).toBe(12);
      expect(genSkip.generatedTokenIds).toEqual(gen.generatedTokenIds);

      // Deterministic repeatability & distinct trace provenance across blocks/refinements
      const genRepeat = generateTwoTowerDiffusion(model, prompt, {
        numBlocks: 2,
        blockSize: 4,
        stepsPerBlock: 4,
        kvCapacity: 24,
        confidenceThreshold: 0.22,
        temperature: 0.0,
        eosTokenIds: [],
      });
      expect(genRepeat.generatedTokenIds).toEqual(gen.generatedTokenIds);
      expect(genRepeat.blocks[0].refinements[0].denoiserPass.passExecutionId).toBe(
        gen.blocks[0].refinements[0].denoiserPass.passExecutionId,
      );
      expect(gen.blocks[0].refinements[0].denoiserPass.passExecutionId).not.toBe(
        gen.blocks[1].refinements[0].denoiserPass.passExecutionId,
      );
      expect(gen.blocks[0].refinements[0].conditioning.conditioningDigest).not.toBe(
        gen.blocks[1].refinements[0].conditioning.conditioningDigest,
      );
      expect(gen.blocks[0].refinements[0].denoiserPass.blockId).toBe(0);
      expect(gen.blocks[1].refinements[0].denoiserPass.blockId).toBe(1);
    });

    it('supports B = 8 block width and independent stepsPerBlock !== blockSize (K < B and K > B with early completion)', () => {
      const model = createDefaultTwoTowerModel();
      const snap = runTwoTowerContextPrefill(model, [0, 1, 2, 3, 4, 5, 6, 7], { kvCapacity: 24 });

      // B = 8 wide block (P_0 = 8 -> 16)
      const wideBlock = runTwoTowerDiffusionBlock(model, snap, {
        blockId: 0,
        blockSize: 8,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
      });
      expect(wideBlock.finalBlockTokens).toHaveLength(8);
      expect(wideBlock.finalBlockTokens.includes(TWOTOWER_MINI_MASK_TOKEN_ID)).toBe(false);
      expect(wideBlock.nfe).toBe(4);
      expect(wideBlock.nextSnapshot.ctxState.processedTokens).toBe(16);
      const wideGqa = wideBlock.refinements[0].denoiserPass.getGqaTrace(6);
      expect(wideGqa.tokens).toBe(8);
      expect(wideGqa.prefixTokens).toBe(8);
      expect(wideGqa.keyTokens).toBe(16);
      expect(wideGqa.attentionMaskMode).toBe('bidirectional_block');

      // K < B: B = 4, K = 2 -> budget floor forces >= 2 tokens per step, completes in 2 NFE
      const shortBudget = runTwoTowerDiffusionBlock(model, snap, {
        blockId: 0,
        blockSize: 4,
        stepsPerBlock: 2,
        confidenceThreshold: 0.9999,
      });
      expect(shortBudget.nfe).toBe(2);
      expect(shortBudget.earlyCompleted).toBe(false);
      expect(shortBudget.finalBlockTokens.includes(TWOTOWER_MINI_MASK_TOKEN_ID)).toBe(false);
      expect(shortBudget.refinements[0].unmaskStep.tokensCommittedCount).toBe(2);
      expect(shortBudget.refinements[1].unmaskStep.tokensCommittedCount).toBe(2);

      // K > B: B = 4, K = 6 with strict high threshold -> 1 token/step, early-completes at step 3 (4 NFE < 6!)
      const extraBudget = runTwoTowerDiffusionBlock(model, snap, {
        blockId: 0,
        blockSize: 4,
        stepsPerBlock: 6,
        confidenceThreshold: 0.9999,
      });
      expect(extraBudget.nfe).toBe(4);
      expect(extraBudget.earlyCompleted).toBe(true);
      expect(extraBudget.finalBlockTokens.includes(TWOTOWER_MINI_MASK_TOKEN_ID)).toBe(false);

      // Very low threshold -> completes entire block in 1 NFE with earlyCompleted === true
      const instant = runTwoTowerDiffusionBlock(model, snap, {
        blockId: 0,
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.0,
      });
      expect(instant.nfe).toBe(1);
      expect(instant.earlyCompleted).toBe(true);
      expect(instant.finalBlockTokens.includes(TWOTOWER_MINI_MASK_TOKEN_ID)).toBe(false);
    });

    it('halts strictly after full block completion on EOS and supports deterministic Gumbel sampling with explicit variates', () => {
      const model = createDefaultTwoTowerModel();
      const prompt = [0, 1, 2, 3, 4, 5, 6, 7];

      // Inspect block 0's deterministic output, then designate slot 1's token as an EOS sentinel
      const probe = generateTwoTowerDiffusion(model, prompt, {
        numBlocks: 1,
        blockSize: 4,
        stepsPerBlock: 4,
        kvCapacity: 24,
        eosTokenIds: [],
      });
      const triggerEosToken = probe.blocks[0].finalBlockTokens[1];

      const eosHalted = generateTwoTowerDiffusion(model, prompt, {
        numBlocks: 3,
        blockSize: 4,
        stepsPerBlock: 4,
        kvCapacity: 24,
        eosTokenIds: [TWOTOWER_MINI_EOS_TOKEN_ID, triggerEosToken],
      });

      // Halts after block 0 completes all 4 tokens and commits to Context (never starts block 1)
      expect(eosHalted.stoppedByEos).toBe(true);
      expect(eosHalted.completedBlocks).toBe(1);
      expect(eosHalted.blocks[0].stoppedByEos).toBe(true);
      expect(eosHalted.blocks[0].contextCommitted).toBe(true);
      expect(eosHalted.blocks[0].finalBlockTokens).toHaveLength(4);
      expect(eosHalted.finalSnapshot.ctxState.processedTokens).toBe(12);

      // Deterministic Gumbel-max sampling when temperature > 0 using explicit variates
      const makeVariates = (seedOffset: number) => {
        const arr = new Float32Array(4 * 34);
        for (let i = 0; i < arr.length; i++) {
          arr[i] = ((i * 73 + seedOffset * 19 + 7) % 997) / 997;
        }
        return arr;
      };
      const variates = [[makeVariates(1), makeVariates(2), makeVariates(3), makeVariates(4)]];

      const gumbelA = generateTwoTowerDiffusion(model, prompt, {
        numBlocks: 1,
        blockSize: 4,
        stepsPerBlock: 4,
        temperature: 0.7,
        uniformVariatesByBlock: variates,
        eosTokenIds: [],
      });
      const gumbelB = generateTwoTowerDiffusion(model, prompt, {
        numBlocks: 1,
        blockSize: 4,
        stepsPerBlock: 4,
        temperature: 0.7,
        uniformVariatesByBlock: variates,
        eosTokenIds: [],
      });
      expect(gumbelA.generatedTokenIds).toEqual(gumbelB.generatedTokenIds);

      // Missing uniformVariates when temperature > 0 must throw (never falls back to Math.random)
      expect(() =>
        generateTwoTowerDiffusion(model, prompt, {
          numBlocks: 1,
          blockSize: 4,
          stepsPerBlock: 4,
          temperature: 0.5,
        }),
      ).toThrow(RangeError);
    });
  });

  describe('7. Strict Headless & Zero-Math.random Guardrails (All src/twotower/*.ts)', () => {
    it('contains zero DOM, WebGL, scene, render imports, or Math.random() calls across all TwoTower modules', () => {
      const files = [
        'src/twotower/spec.ts',
        'src/twotower/modulation.ts',
        'src/twotower/sampler.ts',
        'src/twotower/parameters.ts',
        'src/twotower/contracts.ts',
        'src/twotower/execution.ts',
        'src/twotower/session.ts',
        'src/twotower/index.ts',
      ];
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
});
