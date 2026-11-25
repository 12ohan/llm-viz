import { describe, expect, it } from 'vitest';
import {
  DEFAULT_TWOTOWER_PROMPT_TOKENS,
  DEFAULT_TWOTOWER_SESSION_MAX_BRANCHES,
  DEFAULT_TWOTOWER_SESSION_MAX_RECORDS,
  TWOTOWER_MINI_EOS_TOKEN_ID,
  TWOTOWER_MINI_MASK_TOKEN_ID,
  assertPresentationSafeNoInfinity,
  commitTwoTowerActiveBlockToContext,
  commitTwoTowerRefinementConfidence,
  createDefaultTwoTowerModel,
  createTwoTowerModelParameters,
  createTwoTowerSession,
  deleteTwoTowerSessionBranch,
  deriveRefinementUniformVariates,
  forkTwoTowerSessionBranch,
  generateTwoTowerDiffusion,
  getActiveBranch,
  getActiveRecord,
  getLatestContextExecution,
  getSessionCapabilities,
  initTwoTowerDiffusionBlock,
  measureConditioningTypedArrayBytes,
  measureContextSnapshotTypedArrayBytes,
  measureTwoTowerSessionRetainedBytes,
  navigateTwoTowerSession,
  navigateTwoTowerSessionBack,
  navigateTwoTowerSessionForward,
  pruneInactiveSessionTraces,
  restartTwoTowerSession,
  runTwoTowerArStep,
  runTwoTowerContextPrefill,
  runTwoTowerDiffusionRefinement,
  runTwoTowerMockArStep,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  stepTwoTowerSessionNext,
  switchTwoTowerSessionBranch,
  updateTwoTowerSessionConfig,
  validateContextConditioningState,
  validateContextSnapshot,
  type TwoTowerSession,
  type TwoTowerSessionDiffusionBlockCommitRecord,
  type TwoTowerSessionDiffusionConfidenceCommitRecord,
  type TwoTowerSessionDiffusionRefinementRecord,
  type TwoTowerSessionRecord,
} from '../src/twotower';
import { assertBitExactFiniteArray } from './helpers/parityAssert';

function expectRecordKind<K extends TwoTowerSessionRecord['kind']>(
  record: TwoTowerSessionRecord,
  kind: K,
): Extract<TwoTowerSessionRecord, { readonly kind: K }> {
  expect(record.kind).toBe(kind);
  if (record.kind !== kind) {
    throw new Error(`Expected record kind "${kind}", received "${record.kind}"`);
  }
  return record as Extract<TwoTowerSessionRecord, { readonly kind: K }>;
}

/** Advance one full diffusion block via explicit step-by-step lifecycle transitions. */
function runOneSessionDiffusionBlock(
  session: TwoTowerSession,
  options?: {
    readonly uniformVariatesPerStep?: readonly (Float32Array | readonly number[])[];
  },
): {
  readonly session: TwoTowerSession;
  readonly refinementRecords: readonly TwoTowerSessionDiffusionRefinementRecord[];
  readonly confidenceRecords: readonly TwoTowerSessionDiffusionConfidenceCommitRecord[];
  readonly commitRecord: TwoTowerSessionDiffusionBlockCommitRecord;
} {
  let s = initTwoTowerDiffusionBlock(session);
  const refinementRecords: TwoTowerSessionDiffusionRefinementRecord[] = [];
  const confidenceRecords: TwoTowerSessionDiffusionConfidenceCommitRecord[] = [];

  let stepIdx = 0;
  while (getActiveRecord(s).lifecyclePhase !== 'completed_uncommitted') {
    s = runTwoTowerDiffusionRefinement(s);
    const refRec = getActiveRecord(s);
    if (refRec.kind !== 'diffusion_refinement') {
      throw new Error(`Expected diffusion_refinement, got ${refRec.kind}`);
    }
    refinementRecords.push(refRec);

    s = commitTwoTowerRefinementConfidence(s, {
      uniformVariates: options?.uniformVariatesPerStep?.[stepIdx],
    });
    const confRec = getActiveRecord(s);
    if (confRec.kind !== 'diffusion_confidence_commit') {
      throw new Error(`Expected diffusion_confidence_commit, got ${confRec.kind}`);
    }
    confidenceRecords.push(confRec);
    stepIdx++;
  }

  s = commitTwoTowerActiveBlockToContext(s);
  const commitRec = getActiveRecord(s);
  if (commitRec.kind !== 'diffusion_block_commit') {
    throw new Error(`Expected diffusion_block_commit, got ${commitRec.kind}`);
  }

  return {
    session: s,
    refinementRecords,
    confidenceRecords,
    commitRecord: commitRec,
  };
}

function makeDeterministicVariates(B: number, V: number, seedOffset: number): Float32Array {
  const arr = new Float32Array(B * V);
  for (let i = 0; i < arr.length; i++) {
    arr[i] = ((i * 73 + seedOffset * 19 + 7) % 997) / 997;
  }
  return arr;
}

describe('Stage TT2 Wave 2: Immutable-by-Contract Interactive TwoTowerSession Architecture', () => {
  describe('1. Explicit Step-by-Step Diffusion Lifecycle & 0-ULP Equivalence vs. TT1 generateTwoTowerDiffusion', () => {
    it('steps cleanly through prefill -> init -> refinement -> confidence_commit -> completed_uncommitted -> block_commit', () => {
      const model = createDefaultTwoTowerModel();
      const s0 = createTwoTowerSession(model, {
        confidenceThreshold: 0.9999, // force 1 token/step -> 4 refinements for B=4, K=4
        eosTokenIds: [],
      });

      expect(s0.promptTokenIds).toEqual(DEFAULT_TWOTOWER_PROMPT_TOKENS);
      expect(s0.kvCapacity).toBe(24);
      expect(s0.bounds).toEqual({
        maxBranches: DEFAULT_TWOTOWER_SESSION_MAX_BRANCHES,
        maxRecords: DEFAULT_TWOTOWER_SESSION_MAX_RECORDS,
      });

      const rec0 = expectRecordKind(getActiveRecord(s0), 'context_prefill');
      expect(rec0.lifecyclePhase).toBe('ready');
      expect(rec0.committedPrefixLen).toBe(8);
      expect(rec0.activeBlock).toBeNull();
      expect(rec0.contextExecution.segments).toHaveLength(2);
      expect(rec0.contextExecution.segments[0].tokenRange).toEqual([0, 6]);
      expect(rec0.contextExecution.segments[1].tokenRange).toEqual([7, 7]);

      // Step 1: diffusion_block_init
      const s1 = initTwoTowerDiffusionBlock(s0);
      const rec1 = expectRecordKind(getActiveRecord(s1), 'diffusion_block_init');
      expect(rec1.lifecyclePhase).toBe('block_initialized');
      expect(rec1.committedPrefixLen).toBe(8);
      expect(rec1.contextSnapshot).toBe(rec0.contextSnapshot);
      expect(rec1.activeBlock.status).toBe('initialized');
      expect(rec1.activeBlock.blockId).toBe(0);
      expect(rec1.activeBlock.blockConfig).toEqual({ blockId: 0, blockSize: 4, stepsPerBlock: 4 });
      expect(rec1.activeBlock.currentTokens).toEqual([32, 32, 32, 32]);
      expect(rec1.activeBlock.nMasked).toBe(4);
      expect(rec1.activeBlock.contextCommitted).toBe(false);
      const sharedBase = rec1.conditioningBase;
      expect(sharedBase.byteLength).toBe(79_872);

      // Cannot commit confidence or commit block before running Denoiser refinement
      expect(() => commitTwoTowerRefinementConfidence(s1)).toThrow(RangeError);
      expect(() => commitTwoTowerActiveBlockToContext(s1)).toThrow(RangeError);
      // Cannot run AR or Mock-AR while diffusion block is active
      expect(() => stepTwoTowerSessionAr(s1)).toThrow(RangeError);
      expect(() => stepTwoTowerSessionMockAr(s1)).toThrow(RangeError);

      // Run 4 explicit refinement -> confidence_commit pairs
      let cur = s1;
      for (let r = 0; r < 4; r++) {
        const beforeTokens = getActiveRecord(cur).activeBlock!.currentTokens;
        cur = runTwoTowerDiffusionRefinement(cur);
        const refRec = expectRecordKind(getActiveRecord(cur), 'diffusion_refinement');
        expect(refRec.lifecyclePhase).toBe('refinement_denoised');
        expect(refRec.activeBlock.status).toBe('awaiting_confidence_commit');
        expect(refRec.refinementIndex).toBe(r);
        // xt MUST remain unchanged after diffusion_refinement so Denoiser pass is inspectable prior to unmasking
        expect(refRec.activeBlock.currentTokens).toEqual(beforeTokens);
        expect(refRec.committedPrefixLen).toBe(8);
        // Shared ~78 KiB conditioning maps reused without duplication!
        expect(refRec.conditioningBase).toBe(sharedBase);
        expect(refRec.conditioning.gqaPrefixCaches).toBe(sharedBase.baseConditioning.gqaPrefixCaches);
        expect(refRec.conditioning.mambaSeedStates).toBe(sharedBase.baseConditioning.mambaSeedStates);
        expect(() => validateContextConditioningState(refRec.conditioning, model, 4)).not.toThrow();
        // Cannot run another refinement before committing confidence
        expect(() => runTwoTowerDiffusionRefinement(cur)).toThrow(RangeError);

        cur = commitTwoTowerRefinementConfidence(cur);
        const confRec = expectRecordKind(getActiveRecord(cur), 'diffusion_confidence_commit');
        expect(confRec.refinementIndex).toBe(r);
        expect(confRec.committedPrefixLen).toBe(8);
        expect(confRec.contextSnapshot).toBe(rec0.contextSnapshot);
        expect(confRec.variateProvenance).toEqual({
          mode: 'greedy_none',
          samplingSeed: null,
          derivationKey: null,
          variateDigest: null,
          variates: null,
        });
        expect(() => assertPresentationSafeNoInfinity(confRec.samplerPresentation)).not.toThrow();

        if (r < 3) {
          expect(confRec.lifecyclePhase).toBe('block_refining');
          expect(confRec.activeBlock.status).toBe('refining');
          expect(confRec.activeBlock.nMasked).toBe(3 - r);
          expect(() => commitTwoTowerActiveBlockToContext(cur)).toThrow(RangeError);
        } else {
          // Completed-but-uncommitted pause state!
          expect(confRec.lifecyclePhase).toBe('completed_uncommitted');
          expect(confRec.activeBlock.status).toBe('completed_uncommitted');
          expect(confRec.activeBlock.nMasked).toBe(0);
          expect(confRec.activeBlock.contextCommitted).toBe(false);
          // Context MUST STILL be at P = 8 until explicit block commit!
          expect(confRec.committedPrefixLen).toBe(8);
          expect(confRec.contextSnapshot.ctxState.processedTokens).toBe(8);
          expect(confRec.activeBlock.currentTokens.includes(TWOTOWER_MINI_MASK_TOKEN_ID)).toBe(false);
          expect(confRec.activeBlock.pendingEosInBlock).toBe(false);
          // Cannot run more refinements on a completed block
          expect(() => runTwoTowerDiffusionRefinement(cur)).toThrow(RangeError);
          expect(() => commitTwoTowerRefinementConfidence(cur)).toThrow(RangeError);
        }
      }

      // Explicit block commit advances Context P = 8 -> 12 with BOTH split segments [8..10] + [11..11]
      const sCommit = commitTwoTowerActiveBlockToContext(cur);
      const commitRec = expectRecordKind(getActiveRecord(sCommit), 'diffusion_block_commit');
      expect(commitRec.lifecyclePhase).toBe('ready');
      expect(commitRec.activeBlock).toBeNull();
      expect(commitRec.committedPrefixLen).toBe(12);
      expect(commitRec.nextBlockId).toBe(1);
      expect(commitRec.contextExecution.operation).toBe('commit_block');
      expect(commitRec.contextExecution.committedTokenRange).toEqual([8, 11]);
      expect(commitRec.contextExecution.segments).toHaveLength(2);
      expect(commitRec.contextExecution.segments[0].segmentRole).toBe('split_prefix_for_mamba_s2');
      expect(commitRec.contextExecution.segments[0].tokenRange).toEqual([8, 10]);
      expect(commitRec.contextExecution.segments[1].segmentRole).toBe('split_final_token');
      expect(commitRec.contextExecution.segments[1].tokenRange).toEqual([11, 11]);
      expect(commitRec.nfe).toBe(4);
      expect(commitRec.earlyCompleted).toBe(false);
      expect(commitRec.stoppedByEos).toBe(false);
    });

    it('matches TT1 generateTwoTowerDiffusion at 0-ULP across multi-block B=4, B=8, K=2, K=6, greedy and stochastic (T=0.7)', () => {
      const model = createDefaultTwoTowerModel();
      const prompt = [0, 1, 2, 3, 4, 5, 6, 7];

      const configs: readonly {
        readonly label: string;
        readonly numBlocks: number;
        readonly blockSize: number;
        readonly stepsPerBlock: number;
        readonly confidenceThreshold: number;
        readonly temperature: number;
      }[] = [
        {
          label: '2-block B=4 K=4 greedy (γ=0.22)',
          numBlocks: 2,
          blockSize: 4,
          stepsPerBlock: 4,
          confidenceThreshold: 0.22,
          temperature: 0.0,
        },
        {
          label: '2-block B=8 K=4 greedy (γ=0.9999)',
          numBlocks: 2,
          blockSize: 8,
          stepsPerBlock: 4,
          confidenceThreshold: 0.9999,
          temperature: 0.0,
        },
        {
          label: '2-block B=4 K=2 budget-floor (K < B)',
          numBlocks: 2,
          blockSize: 4,
          stepsPerBlock: 2,
          confidenceThreshold: 0.9999,
          temperature: 0.0,
        },
        {
          label: '2-block B=4 K=6 early-exit (K > B)',
          numBlocks: 2,
          blockSize: 4,
          stepsPerBlock: 6,
          confidenceThreshold: 0.9999,
          temperature: 0.0,
        },
        {
          label: '2-block B=4 K=4 stochastic (T=0.7)',
          numBlocks: 2,
          blockSize: 4,
          stepsPerBlock: 4,
          confidenceThreshold: 0.22,
          temperature: 0.7,
        },
      ];

      for (const cfg of configs) {
        let session = createTwoTowerSession(model, {
          sessionId: `parity-${cfg.label}`,
          promptTokenIds: prompt,
          kvCapacity: 24,
          blockSize: cfg.blockSize,
          stepsPerBlock: cfg.stepsPerBlock,
          confidenceThreshold: cfg.confidenceThreshold,
          temperature: cfg.temperature,
          eosTokenIds: [],
          samplingSeed: 4242,
        });

        const variatesByBlock: Float32Array[][] = [];
        const sessionBlocks: {
          readonly commitRecord: TwoTowerSessionDiffusionBlockCommitRecord;
          readonly refRecords: readonly TwoTowerSessionDiffusionRefinementRecord[];
          readonly confRecords: readonly TwoTowerSessionDiffusionConfidenceCommitRecord[];
        }[] = [];

        for (let b = 0; b < cfg.numBlocks; b++) {
          const ran = runOneSessionDiffusionBlock(session);
          session = ran.session;
          sessionBlocks.push({
            commitRecord: ran.commitRecord,
            refRecords: ran.refinementRecords,
            confRecords: ran.confidenceRecords,
          });
          variatesByBlock.push(
            ran.confidenceRecords.map((c) => c.variateProvenance.variates ?? new Float32Array(0)),
          );
        }

        const tt1 = generateTwoTowerDiffusion(model, prompt, {
          numBlocks: cfg.numBlocks,
          blockSize: cfg.blockSize,
          stepsPerBlock: cfg.stepsPerBlock,
          kvCapacity: 24,
          confidenceThreshold: cfg.confidenceThreshold,
          temperature: cfg.temperature,
          eosTokenIds: [],
          ...(cfg.temperature > 0 ? { uniformVariatesByBlock: variatesByBlock } : {}),
        });

        const finalRec = getActiveRecord(session);
        expect(finalRec.contextSnapshot.ctxState.stateDigest, cfg.label).toBe(
          tt1.finalSnapshot.ctxState.stateDigest,
        );
        expect(finalRec.contextSnapshot.snapshotDigest, cfg.label).toBe(
          tt1.finalSnapshot.snapshotDigest,
        );
        expect(Array.from(finalRec.contextSnapshot.ctxState.prefixTokenIds), cfg.label).toEqual(
          tt1.fullSequenceTokenIds,
        );
        assertBitExactFiniteArray(
          finalRec.contextSnapshot.arLogits,
          tt1.finalSnapshot.arLogits,
          `${cfg.label} final arLogits`,
        );

        for (let b = 0; b < cfg.numBlocks; b++) {
          const sb = sessionBlocks[b];
          const tb = tt1.blocks[b];
          expect(sb.commitRecord.committedBlockTokens, `${cfg.label} b=${b} tokens`).toEqual(
            tb.finalBlockTokens,
          );
          expect(sb.commitRecord.nfe, `${cfg.label} b=${b} nfe`).toBe(tb.nfe);
          expect(sb.commitRecord.earlyCompleted, `${cfg.label} b=${b} earlyCompleted`).toBe(
            tb.earlyCompleted,
          );
          expect(sb.refRecords).toHaveLength(tb.refinements.length);
          for (let r = 0; r < tb.refinements.length; r++) {
            assertBitExactFiniteArray(
              sb.refRecords[r].denoiserPass.logits,
              tb.refinements[r].denoiserPass.logits,
              `${cfg.label} b=${b} r=${r} logits`,
            );
            expect(sb.confRecords[r].unmaskStep.xtNext).toEqual(tb.refinements[r].unmaskStep.xtNext);
            expect(sb.confRecords[r].unmaskStep.committedIndices).toEqual(
              tb.refinements[r].unmaskStep.committedIndices,
            );
          }
        }
      }
    });

    it('advances through every legal transition automatically via stepTwoTowerSessionNext', () => {
      const model = createDefaultTwoTowerModel();
      let s = createTwoTowerSession(model, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.22,
        eosTokenIds: [],
      });

      const kinds: string[] = [getActiveRecord(s).kind];
      while (getActiveRecord(s).committedPrefixLen < 12) {
        s = stepTwoTowerSessionNext(s);
        kinds.push(getActiveRecord(s).kind);
      }

      expect(kinds[0]).toBe('context_prefill');
      expect(kinds[1]).toBe('diffusion_block_init');
      expect(kinds[2]).toBe('diffusion_refinement');
      expect(kinds[3]).toBe('diffusion_confidence_commit');
      expect(kinds[kinds.length - 1]).toBe('diffusion_block_commit');
      expect(getActiveRecord(s).committedPrefixLen).toBe(12);
    });
  });

  describe('2. Mode Switching (mask_diffusion <-> ar <-> mock_ar) & Identity-Weight 0-ULP Parity', () => {
    it('switches cleanly across mask_diffusion, ar, and mock_ar within a single session', () => {
      const model = createDefaultTwoTowerModel();
      let s = createTwoTowerSession(model, {
        mode: 'ar',
        eosTokenIds: [],
      });

      // Step 1: AR step via stepTwoTowerSessionNext
      const snapBeforeAr = getActiveRecord(s).contextSnapshot;
      s = stepTwoTowerSessionNext(s);
      const arRec = getActiveRecord(s);
      expect(arRec.kind).toBe('ar_step');
      if (arRec.kind !== 'ar_step') throw new Error('Expected ar_step');
      expect(arRec.committedPrefixLen).toBe(9);
      expect(arRec.selectionLogits.length).toBe(34);
      expect(arRec.selectionProbs.length).toBe(34);
      expect(arRec.wasExplicitTokenOverride).toBe(false);
      expect(arRec.contextExecution.segments).toHaveLength(1);
      expect(arRec.contextExecution.committedTokenRange).toEqual([8, 8]);

      const standaloneAr = runTwoTowerArStep(model, snapBeforeAr);
      expect(arRec.selectedTokenId).toBe(standaloneAr.tokenId);
      expect(arRec.selectedTokenLabel).toBe(standaloneAr.tokenLabel);
      assertBitExactFiniteArray(arRec.selectionLogits, standaloneAr.logits, 'AR selectionLogits');
      assertBitExactFiniteArray(arRec.selectionProbs, standaloneAr.probs, 'AR selectionProbs');
      expect(arRec.contextSnapshot.snapshotDigest).toBe(standaloneAr.nextSnapshot.snapshotDigest);

      // Step 2: Explicit token override in AR
      s = stepTwoTowerSessionAr(s, { nextTokenId: 12 });
      const arOverrideRec = getActiveRecord(s);
      expect(arOverrideRec.kind).toBe('ar_step');
      if (arOverrideRec.kind !== 'ar_step') throw new Error('Expected ar_step');
      expect(arOverrideRec.selectedTokenId).toBe(12);
      expect(arOverrideRec.wasExplicitTokenOverride).toBe(true);
      expect(arOverrideRec.committedPrefixLen).toBe(10);

      // Step 3: Switch to Mock-AR
      s = updateTwoTowerSessionConfig(s, { mode: 'mock_ar' });
      const snapBeforeMock = getActiveRecord(s).contextSnapshot;
      s = stepTwoTowerSessionNext(s);
      const mockRec = getActiveRecord(s);
      expect(mockRec.kind).toBe('mock_ar_step');
      if (mockRec.kind !== 'mock_ar_step') throw new Error('Expected mock_ar_step');
      expect(mockRec.committedPrefixLen).toBe(11);
      expect(mockRec.conditioning.mode).toBe('mock_ar');
      expect(mockRec.conditioning.committedPrefixLen).toBe(9); // KV[:-1]
      expect(mockRec.denoiserPass.tModel).toBeNull();
      expect(mockRec.denoiserPass.tEmb).toBeNull();
      expect(mockRec.denoiserPass.getLayerModulation(1)).toBeNull();
      expect(mockRec.contextExecution.committedTokenRange).toEqual([10, 10]);

      const standaloneMock = runTwoTowerMockArStep(model, snapBeforeMock);
      expect(mockRec.selectedTokenId).toBe(standaloneMock.tokenId);
      assertBitExactFiniteArray(
        mockRec.denoiserPass.logits,
        standaloneMock.denoiserPass.logits,
        'Mock-AR denoiserPass logits',
      );
      expect(mockRec.contextSnapshot.snapshotDigest).toBe(
        standaloneMock.nextSnapshot.snapshotDigest,
      );

      // Step 4: Switch to mask_diffusion block (P = 11 -> 15)
      s = updateTwoTowerSessionConfig(s, { mode: 'mask_diffusion', blockSize: 4, stepsPerBlock: 4 });
      const diffRan = runOneSessionDiffusionBlock(s);
      s = diffRan.session;
      expect(getActiveRecord(s).committedPrefixLen).toBe(15);

      // Step 5: Switch back to Mock-AR after block commit (verifying mambaS2 continuity!)
      const snapAfterDiff = getActiveRecord(s).contextSnapshot;
      s = stepTwoTowerSessionMockAr(s);
      const mockAfterDiff = getActiveRecord(s);
      expect(mockAfterDiff.kind).toBe('mock_ar_step');
      expect(mockAfterDiff.committedPrefixLen).toBe(16);
      const standaloneMockAfterDiff = runTwoTowerMockArStep(model, snapAfterDiff);
      expect(mockAfterDiff.contextSnapshot.snapshotDigest).toBe(
        standaloneMockAfterDiff.nextSnapshot.snapshotDigest,
      );
    });

    it('proves 0-ULP identity-weight Mock-AR === AR parity through the session API', () => {
      const idModel = createTwoTowerModelParameters({
        perturbationScale: 0,
        routerPrecisionPolicy: 'float64_reference',
      });

      let arSession = createTwoTowerSession(idModel, {
        mode: 'ar',
        eosTokenIds: [],
      });
      let mockSession = createTwoTowerSession(idModel, {
        mode: 'mock_ar',
        eosTokenIds: [],
      });

      for (let step = 0; step < 3; step++) {
        arSession = stepTwoTowerSessionNext(arSession);
        mockSession = stepTwoTowerSessionNext(mockSession);

        const arRec = getActiveRecord(arSession);
        const mockRec = getActiveRecord(mockSession);
        if (arRec.kind !== 'ar_step' || mockRec.kind !== 'mock_ar_step') {
          throw new Error('Unexpected record kinds');
        }
        assertBitExactFiniteArray(
          mockRec.denoiserPass.logits,
          arRec.selectionLogits,
          `step=${step} identity Mock-AR vs AR logits`,
        );
        assertBitExactFiniteArray(
          mockRec.denoiserPass.probs,
          arRec.selectionProbs,
          `step=${step} identity Mock-AR vs AR probs`,
        );
        expect(mockRec.selectedTokenId).toBe(arRec.selectedTokenId);
        expect(mockRec.contextSnapshot.ctxState.stateDigest).toBe(
          arRec.contextSnapshot.ctxState.stateDigest,
        );
        expect(mockRec.contextSnapshot.snapshotDigest).toBe(arRec.contextSnapshot.snapshotDigest);
      }
    });
  });

  describe('3. Deterministic Stochastic Replay (T > 0), Branching Seeds & Explicit Variates', () => {
    it('derives bit-exact identical variates, tokens & confidences on replay, differentiates branches/seeds, and supports explicit variates', () => {
      const model = createDefaultTwoTowerModel();

      const runStochasticBlock = (samplingSeed: number, sessionId: string) => {
        const s0 = createTwoTowerSession(model, {
          sessionId,
          blockSize: 4,
          stepsPerBlock: 4,
          confidenceThreshold: 0.22,
          temperature: 0.7,
          samplingSeed,
          eosTokenIds: [],
        });
        return runOneSessionDiffusionBlock(s0);
      };

      const runA = runStochasticBlock(777, 'stoch-session');
      const runB = runStochasticBlock(777, 'stoch-session');
      const runDiffSeed = runStochasticBlock(778, 'stoch-session');

      // Same seed + session + branch + block + refinement -> 0-ULP bit-exact identical!
      expect(runA.commitRecord.committedBlockTokens).toEqual(runB.commitRecord.committedBlockTokens);
      expect(runA.confidenceRecords).toHaveLength(runB.confidenceRecords.length);
      for (let r = 0; r < runA.confidenceRecords.length; r++) {
        const ca = runA.confidenceRecords[r];
        const cb = runB.confidenceRecords[r];
        expect(ca.variateProvenance.mode).toBe('derived_deterministic');
        expect(ca.variateProvenance.samplingSeed).toBe(777);
        expect(ca.variateProvenance.derivationKey).toBe(cb.variateProvenance.derivationKey);
        expect(ca.variateProvenance.variateDigest).toBe(cb.variateProvenance.variateDigest);
        assertBitExactFiniteArray(
          ca.variateProvenance.variates!,
          cb.variateProvenance.variates!,
          `r=${r} derived variates`,
        );
        expect(ca.unmaskStep.confidence.length).toBe(cb.unmaskStep.confidence.length);
        for (let j = 0; j < ca.unmaskStep.confidence.length; j++) {
          expect(Object.is(ca.unmaskStep.confidence[j], cb.unmaskStep.confidence[j])).toBe(true);
        }
        expect(ca.samplerPresentation).toEqual(cb.samplerPresentation);
        expect(() => assertPresentationSafeNoInfinity(ca.samplerPresentation)).not.toThrow();

        // Matches standalone deriveRefinementUniformVariates bit-for-bit
        const standalone = deriveRefinementUniformVariates({
          samplingSeed: 777,
          sessionId: 'stoch-session',
          branchId: 'branch-0',
          blockId: 0,
          refinementIndex: r,
          blockSize: 4,
          vocabSize: 34,
        });
        expect(ca.variateProvenance.variateDigest).toBe(standalone.variateDigest);
        assertBitExactFiniteArray(
          ca.variateProvenance.variates!,
          standalone.variates,
          `r=${r} standalone variate match`,
        );
      }

      // Restarting stochastic session with same seed reproduces bit-exact identical variates & tokens!
      const restartedRun = runOneSessionDiffusionBlock(restartTwoTowerSession(runA.session));
      expect(restartedRun.commitRecord.committedBlockTokens).toEqual(
        runA.commitRecord.committedBlockTokens,
      );
      expect(restartedRun.confidenceRecords[0].variateProvenance.variateDigest).toBe(
        runA.confidenceRecords[0].variateProvenance.variateDigest,
      );

      // Different seed -> distinct variateDigest
      expect(runA.confidenceRecords[0].variateProvenance.variateDigest).not.toBe(
        runDiffSeed.confidenceRecords[0].variateProvenance.variateDigest,
      );

      // Rewind to diffusion_refinement r=0 and fork two branches with same vs different seeds
      const ref0CursorIndex = 2; // 0: prefill, 1: init, 2: ref0
      let forkedSame = forkTwoTowerSessionBranch(runA.session, {
        fromCursorIndex: ref0CursorIndex,
        label: 'Fork Same Seed',
      });
      forkedSame = commitTwoTowerRefinementConfidence(forkedSame);
      const forkedSameConf = getActiveRecord(forkedSame);
      if (forkedSameConf.kind !== 'diffusion_confidence_commit') {
        throw new Error('Expected diffusion_confidence_commit');
      }
      // Forked branch has distinct branchId ('branch-1') -> distinct derivationKey & variateDigest from 'branch-0'
      expect(forkedSameConf.identity.branchId).toBe('branch-1');
      expect(forkedSameConf.variateProvenance.derivationKey).toContain('branch=branch-1');
      expect(forkedSameConf.variateProvenance.variateDigest).not.toBe(
        runA.confidenceRecords[0].variateProvenance.variateDigest,
      );

      // Forking from runB at identical point produces bit-exact identical 'branch-1' variates!
      let forkedSameReplay = forkTwoTowerSessionBranch(runB.session, {
        fromCursorIndex: ref0CursorIndex,
        label: 'Fork Same Seed Replay',
      });
      forkedSameReplay = commitTwoTowerRefinementConfidence(forkedSameReplay);
      const forkedSameReplayConf = getActiveRecord(forkedSameReplay);
      if (forkedSameReplayConf.kind !== 'diffusion_confidence_commit') {
        throw new Error('Expected diffusion_confidence_commit');
      }
      expect(forkedSameReplayConf.variateProvenance.variateDigest).toBe(
        forkedSameConf.variateProvenance.variateDigest,
      );
      expect(forkedSameReplayConf.unmaskStep.xtNext).toEqual(forkedSameConf.unmaskStep.xtNext);

      // Fork with different seed
      let forkedDiff = forkTwoTowerSessionBranch(runA.session, {
        fromCursorIndex: ref0CursorIndex,
        configPatch: { samplingSeed: 9999 },
      });
      forkedDiff = commitTwoTowerRefinementConfidence(forkedDiff);
      const forkedDiffConf = getActiveRecord(forkedDiff);
      if (forkedDiffConf.kind !== 'diffusion_confidence_commit') {
        throw new Error('Expected diffusion_confidence_commit');
      }
      expect(forkedDiffConf.variateProvenance.samplingSeed).toBe(9999);
      expect(forkedDiffConf.variateProvenance.variateDigest).not.toBe(
        forkedSameConf.variateProvenance.variateDigest,
      );

      // Caller explicit variates override (and mutation isolation of caller's input array!)
      const explicitArr = makeDeterministicVariates(4, 34, 5);
      const explicitCopy = new Float32Array(explicitArr);
      let explicitBranch = forkTwoTowerSessionBranch(runA.session, {
        fromCursorIndex: ref0CursorIndex,
        label: 'Explicit Variates',
      });
      explicitBranch = commitTwoTowerRefinementConfidence(explicitBranch, {
        uniformVariates: explicitArr,
      });
      // Mutate caller's array after the call -> session record's stored variates MUST be isolated!
      explicitArr.fill(0.5);
      const explicitRec = getActiveRecord(explicitBranch);
      if (explicitRec.kind !== 'diffusion_confidence_commit') {
        throw new Error('Expected diffusion_confidence_commit');
      }
      expect(explicitRec.variateProvenance.mode).toBe('caller_explicit');
      expect(explicitRec.variateProvenance.derivationKey).toBe('caller_explicit');
      expect(explicitRec.variateProvenance.samplingSeed).toBeNull();
      expect(explicitRec.variateProvenance.variateDigest).toHaveLength(16);
      assertBitExactFiniteArray(
        explicitRec.variateProvenance.variates!,
        explicitCopy,
        'explicit variates isolated from caller mutation',
      );
    });
  });

  describe('4. Non-Retroactive Parameter Changes', () => {
    it('never mutates historical records and never resizes an in-flight diffusion block', () => {
      const model = createDefaultTwoTowerModel();
      const s0 = createTwoTowerSession(model, {
        mode: 'mask_diffusion',
        blockSize: 4,
        stepsPerBlock: 4,
        confidenceThreshold: 0.9999,
        temperature: 0.0,
        samplingSeed: 101,
        eosTokenIds: [],
      });

      const sInit = initTwoTowerDiffusionBlock(s0);
      const sRef0 = runTwoTowerDiffusionRefinement(sInit);
      const sConf0 = commitTwoTowerRefinementConfidence(sRef0);

      // Snapshot historical record values before parameter updates
      const origInitRec = getActiveRecord(sInit);
      const origConf0Rec = getActiveRecord(sConf0);
      if (origConf0Rec.kind !== 'diffusion_confidence_commit') {
        throw new Error('Expected diffusion_confidence_commit');
      }
      const origConf0Tokens = [...origConf0Rec.unmaskStep.xtNext];
      const origConf0Threshold = origConf0Rec.samplerPresentation.confidenceThreshold;
      const origConf0Temp = origConf0Rec.samplerPresentation.temperature;
      const origConf0Config = { ...origConf0Rec.configAtStep };

      // Update EVERY parameter mid-block!
      const sUpdated = updateTwoTowerSessionConfig(sConf0, {
        mode: 'ar',
        blockSize: 8,
        stepsPerBlock: 2,
        confidenceThreshold: 0.05,
        temperature: 0.65,
        samplingSeed: 909,
        eosTokenIds: [TWOTOWER_MINI_EOS_TOKEN_ID],
      });

      // Prior session object is completely unchanged (immutable-by-contract)
      expect(sConf0.pendingConfig.blockSize).toBe(4);
      expect(sConf0.pendingConfig.mode).toBe('mask_diffusion');
      expect(sUpdated.pendingConfig.blockSize).toBe(8);
      expect(sUpdated.pendingConfig.stepsPerBlock).toBe(2);
      expect(sUpdated.pendingConfig.confidenceThreshold).toBe(0.05);
      expect(sUpdated.pendingConfig.temperature).toBe(0.65);
      expect(sUpdated.pendingConfig.mode).toBe('ar');

      // Historical records MUST remain 100% untouched
      const branch = getActiveBranch(sUpdated);
      expect(branch.records[1]).toBe(origInitRec);
      expect(branch.records[3]).toBe(origConf0Rec);
      expect(origInitRec.configAtStep.blockSize).toBe(4);
      expect(origConf0Rec.configAtStep).toEqual(origConf0Config);
      expect(origConf0Rec.unmaskStep.xtNext).toEqual(origConf0Tokens);
      expect(origConf0Rec.samplerPresentation.confidenceThreshold).toBe(origConf0Threshold);
      expect(origConf0Rec.samplerPresentation.temperature).toBe(origConf0Temp);

      // In-flight block STILL keeps frozen blockConfig { blockSize: 4, stepsPerBlock: 4 }!
      const sRef1 = runTwoTowerDiffusionRefinement(sUpdated);
      const ref1Rec = getActiveRecord(sRef1);
      if (ref1Rec.kind !== 'diffusion_refinement') {
        throw new Error('Expected diffusion_refinement');
      }
      expect(ref1Rec.blockConfig).toEqual({ blockId: 0, blockSize: 4, stepsPerBlock: 4 });
      expect(ref1Rec.xtInput).toHaveLength(4);
      expect(ref1Rec.configAtStep.blockSize).toBe(4);
      expect(ref1Rec.configAtStep.stepsPerBlock).toBe(4);

      // Updated confidenceThreshold (0.05) and temperature (0.65) DO apply to future confidence commit r=1!
      const sConf1 = commitTwoTowerRefinementConfidence(sRef1);
      const conf1Rec = getActiveRecord(sConf1);
      if (conf1Rec.kind !== 'diffusion_confidence_commit') {
        throw new Error('Expected diffusion_confidence_commit');
      }
      expect(conf1Rec.blockConfig).toEqual({ blockId: 0, blockSize: 4, stepsPerBlock: 4 });
      expect(conf1Rec.configAtStep.confidenceThreshold).toBe(0.05);
      expect(conf1Rec.configAtStep.temperature).toBe(0.65);
      expect(conf1Rec.configAtStep.samplingSeed).toBe(909);
      expect(conf1Rec.variateProvenance.mode).toBe('derived_deterministic');
      expect(conf1Rec.variateProvenance.samplingSeed).toBe(909);

      // Finish block 0 and commit (still B = 4 -> P = 12)
      let sEndBlock0 = sConf1;
      while (getActiveRecord(sEndBlock0).lifecyclePhase !== 'completed_uncommitted') {
        sEndBlock0 = stepTwoTowerSessionNext(sEndBlock0);
      }
      sEndBlock0 = commitTwoTowerActiveBlockToContext(sEndBlock0);
      expect(getActiveRecord(sEndBlock0).committedPrefixLen).toBe(12);

      // Now that block 0 committed, pendingConfig.mode = 'ar' governs next step!
      expect(getSessionCapabilities(sEndBlock0).nextTransitionKind).toBe('ar_step');
      const sAr = stepTwoTowerSessionNext(sEndBlock0);
      expect(getActiveRecord(sAr).kind).toBe('ar_step');
      expect(getActiveRecord(sAr).committedPrefixLen).toBe(13);

      // Switching back to mask_diffusion uses the updated blockSize = 8, stepsPerBlock = 2 for block 1!
      const sDiff1 = initTwoTowerDiffusionBlock(
        updateTwoTowerSessionConfig(sAr, { mode: 'mask_diffusion' }),
      );
      const diff1Rec = getActiveRecord(sDiff1);
      if (diff1Rec.kind !== 'diffusion_block_init') {
        throw new Error('Expected diffusion_block_init');
      }
      expect(diff1Rec.blockConfig).toEqual({ blockId: 1, blockSize: 8, stepsPerBlock: 2 });
      expect(diff1Rec.initialBlockTokens).toHaveLength(8);

      // Invalid config patches rejected cleanly
      expect(() => updateTwoTowerSessionConfig(s0, { blockSize: 0 })).toThrow(RangeError);
      expect(() => updateTwoTowerSessionConfig(s0, { blockSize: 17 })).toThrow(RangeError);
      expect(() => updateTwoTowerSessionConfig(s0, { stepsPerBlock: 0 })).toThrow(RangeError);
      expect(() => updateTwoTowerSessionConfig(s0, { temperature: -0.1 })).toThrow(RangeError);
      expect(() => updateTwoTowerSessionConfig(s0, { confidenceThreshold: Number.NaN })).toThrow(
        RangeError,
      );
      expect(() => updateTwoTowerSessionConfig(s0, { samplingSeed: 1.5 })).toThrow(RangeError);
      expect(() => updateTwoTowerSessionConfig(s0, { eosTokenIds: [99] })).toThrow(RangeError);
    });
  });

  describe('5. Bounded History, Navigation, Explicit & Auto-Branching, Safe Deletion, Pruning & Restart', () => {
    it('supports back/forward navigation, non-destructive auto-branching from rewound cursor, switching, safe deletion, and restart', () => {
      const model = createDefaultTwoTowerModel();
      let s = createTwoTowerSession(model, { mode: 'ar', eosTokenIds: [] });

      s = stepTwoTowerSessionAr(s, { nextTokenId: 10 }); // idx 1
      s = stepTwoTowerSessionAr(s, { nextTokenId: 11 }); // idx 2
      s = stepTwoTowerSessionAr(s, { nextTokenId: 12 }); // idx 3
      const originalBranch0Tail = getActiveBranch(s).records.map((r) => r.identity.recordId);
      expect(originalBranch0Tail).toHaveLength(4);

      // Navigation back/forward & bounds checking
      expect(() => navigateTwoTowerSessionForward(s)).toThrow(RangeError);
      const sBack1 = navigateTwoTowerSessionBack(s);
      expect(getActiveBranch(sBack1).cursorIndex).toBe(2);
      const sBack2 = navigateTwoTowerSessionBack(sBack1);
      expect(getActiveBranch(sBack2).cursorIndex).toBe(1);
      const sFwd = navigateTwoTowerSessionForward(sBack2);
      expect(getActiveBranch(sFwd).cursorIndex).toBe(2);
      const sAt0 = navigateTwoTowerSession(sFwd, 0);
      expect(getActiveBranch(sAt0).cursorIndex).toBe(0);
      expect(() => navigateTwoTowerSessionBack(sAt0)).toThrow(RangeError);
      expect(() => navigateTwoTowerSession(sAt0, -1)).toThrow(RangeError);
      expect(() => navigateTwoTowerSession(sAt0, 99)).toThrow(RangeError);

      // Non-destructive auto-branching when stepping from rewound cursor index 1!
      expect(getSessionCapabilities(sBack2).willAutoForkOnMutatingStep).toBe(true);
      const sAutoForked = stepTwoTowerSessionAr(sBack2, { nextTokenId: 20 });
      expect(sAutoForked.branches).toHaveLength(2);
      expect(sAutoForked.activeBranchId).toBe('branch-1');

      const origBranch = sAutoForked.branches.find((b) => b.branchId === 'branch-0')!;
      const autoBranch = sAutoForked.branches.find((b) => b.branchId === 'branch-1')!;
      // Original branch tail is 100% preserved and uncorrupted!
      expect(origBranch.records.map((r) => r.identity.recordId)).toEqual(originalBranch0Tail);
      expect(autoBranch.parentBranchId).toBe('branch-0');
      expect(autoBranch.forkPointIndex).toBe(1);
      expect(autoBranch.forkPointRecordId).toBe(origBranch.records[1].identity.recordId);
      expect(autoBranch.records).toHaveLength(3);
      // Shared immutable prefix records
      expect(autoBranch.records[0]).toBe(origBranch.records[0]);
      expect(autoBranch.records[1]).toBe(origBranch.records[1]);
      expect(autoBranch.records[2].identity.parentRecordId).toBe(
        origBranch.records[1].identity.recordId,
      );
      expect(autoBranch.records[2].identity.branchId).toBe('branch-1');

      // Explicit fork & switch & active-branch-safe deletion
      let sWithFork2 = forkTwoTowerSessionBranch(sAutoForked, {
        fromCursorIndex: 1,
        label: 'Explicit Fork',
      });
      expect(sWithFork2.branches).toHaveLength(3);
      expect(sWithFork2.activeBranchId).toBe('branch-2');

      // Cannot delete active branch or unknown branch
      expect(() => deleteTwoTowerSessionBranch(sWithFork2, 'branch-2')).toThrow(RangeError);
      expect(() => deleteTwoTowerSessionBranch(sWithFork2, 'nonexistent')).toThrow(RangeError);

      // Delete inactive branch-1 safely
      sWithFork2 = deleteTwoTowerSessionBranch(sWithFork2, 'branch-1');
      expect(sWithFork2.branches.map((b) => b.branchId)).toEqual(['branch-0', 'branch-2']);

      // Switch back to branch-0 and delete branch-2; cannot delete last remaining branch
      let sSwitched = switchTwoTowerSessionBranch(sWithFork2, 'branch-0');
      expect(sSwitched.activeBranchId).toBe('branch-0');
      sSwitched = deleteTwoTowerSessionBranch(sSwitched, 'branch-2');
      expect(sSwitched.branches).toHaveLength(1);
      expect(() => deleteTwoTowerSessionBranch(sSwitched, 'branch-0')).toThrow(RangeError);

      // Restart preserves model/sessionId/pendingConfig cleanly on a fresh branch-0 without mutating old session
      const sRestarted = restartTwoTowerSession(sAutoForked, {
        configPatch: { mode: 'mask_diffusion', blockSize: 4 },
      });
      expect(sRestarted.sessionId).toBe(sAutoForked.sessionId);
      expect(sRestarted.branches).toHaveLength(1);
      expect(sRestarted.activeBranchId).toBe('branch-0');
      expect(getActiveBranch(sRestarted).records).toHaveLength(1);
      expect(getActiveRecord(sRestarted).kind).toBe('context_prefill');
      expect(sRestarted.pendingConfig.mode).toBe('mask_diffusion');
      // Prior session untouched
      expect(sAutoForked.branches).toHaveLength(2);
    });

    it('enforces maxBranches and maxRecords bounds deterministically without corrupting active branch', () => {
      const model = createDefaultTwoTowerModel();

      // A. maxBranches = 3: creating 5 forks evicts oldest inactive branches while keeping active branch safe
      let s = createTwoTowerSession(model, {
        mode: 'ar',
        eosTokenIds: [],
        maxBranches: 3,
        maxRecords: 128,
      });
      s = stepTwoTowerSessionAr(s, { nextTokenId: 1 });
      for (let i = 0; i < 4; i++) {
        s = forkTwoTowerSessionBranch(s, { fromCursorIndex: 0, label: `Fork ${i + 1}` });
        s = stepTwoTowerSessionAr(s, { nextTokenId: 2 + i });
        expect(s.branches.length).toBeLessThanOrEqual(3);
        expect(s.branches.some((b) => b.branchId === s.activeBranchId)).toBe(true);
      }
      expect(s.branches).toHaveLength(3);
      // Monotonic branch IDs never reused
      expect(s.activeBranchId).toBe('branch-4');

      // B. maxRecords = 5 on single branch -> sliding-window eviction bounds uniqueRecordCount <= 5
      let sSmall = createTwoTowerSession(model, {
        mode: 'ar',
        eosTokenIds: [],
        maxBranches: 4,
        maxRecords: 5,
      });
      for (let step = 0; step < 8; step++) {
        sSmall = stepTwoTowerSessionAr(sSmall, { nextTokenId: (step % 20) + 1 });
        const mem = measureTwoTowerSessionRetainedBytes(sSmall);
        expect(mem.uniqueRecordCount).toBeLessThanOrEqual(5);
        expect(getActiveRecord(sSmall).committedPrefixLen).toBe(8 + step + 1);
      }
      expect(getActiveBranch(sSmall).records).toHaveLength(5);
      // Record indices remain strictly monotonic even after front trimming
      expect(getActiveRecord(sSmall).identity.recordIndex).toBe(8);

      // C. maxRecords across multiple branches evicts oldest inactive branch before trimming active branch
      let sMulti = createTwoTowerSession(model, {
        mode: 'ar',
        eosTokenIds: [],
        maxBranches: 4,
        maxRecords: 6,
      });
      sMulti = stepTwoTowerSessionAr(sMulti, { nextTokenId: 1 }); // rec 1 (2 unique)
      sMulti = stepTwoTowerSessionAr(sMulti, { nextTokenId: 2 }); // rec 2 (3 unique)
      sMulti = stepTwoTowerSessionAr(sMulti, { nextTokenId: 3 }); // rec 3 (4 unique)
      // Fork at idx 1 and add 3 records -> total would be 7 > 6, so branch-0 gets evicted cleanly!
      sMulti = forkTwoTowerSessionBranch(sMulti, { fromCursorIndex: 1 });
      sMulti = stepTwoTowerSessionAr(sMulti, { nextTokenId: 10 }); // 5 unique
      sMulti = stepTwoTowerSessionAr(sMulti, { nextTokenId: 11 }); // 6 unique
      expect(sMulti.branches).toHaveLength(2);
      sMulti = stepTwoTowerSessionAr(sMulti, { nextTokenId: 12 }); // would be 7 -> evicts inactive branch-0
      expect(sMulti.branches).toHaveLength(1);
      expect(sMulti.activeBranchId).toBe('branch-1');
      expect(measureTwoTowerSessionRetainedBytes(sMulti).uniqueRecordCount).toBe(5);

      // D. maxBranches = 1 refuses explicit fork and rewound-cursor auto-fork without corrupting branch-0
      let sSingleBranch = createTwoTowerSession(model, {
        mode: 'ar',
        eosTokenIds: [],
        maxBranches: 1,
      });
      sSingleBranch = stepTwoTowerSessionAr(sSingleBranch, { nextTokenId: 3 });
      expect(getSessionCapabilities(sSingleBranch).canForkBranch).toBe(false);
      expect(() => forkTwoTowerSessionBranch(sSingleBranch)).toThrow(RangeError);
      const sRewoundSingle = navigateTwoTowerSessionBack(sSingleBranch);
      expect(getSessionCapabilities(sRewoundSingle).canStepAr).toBe(false);
      expect(() => stepTwoTowerSessionAr(sRewoundSingle)).toThrow(RangeError);
      expect(getActiveBranch(sRewoundSingle).records).toHaveLength(2);
    });
  });

  describe('6. Full Explicit Identity Taxonomy on Every Record', () => {
    it('populates every model, cache, session, branch, record, block, refinement, and token-range field', () => {
      const model = createDefaultTwoTowerModel();
      const s0 = createTwoTowerSession(model, {
        sessionId: 'id-taxonomy-test',
        eosTokenIds: [],
      });
      const r0 = expectRecordKind(getActiveRecord(s0), 'context_prefill');
      expect(r0.identity).toEqual({
        modelId: model.modelId,
        twoTowerFingerprint: model.twoTowerFingerprint,
        contextParameterFingerprint: model.contextModel.parameterFingerprint,
        denoiserParameterFingerprint: model.denoiserModel.parameterFingerprint,
        sourceCacheDigest: r0.contextExecution.prevStateDigest,
        sourceSnapshotDigest: null,
        resultCacheDigest: r0.contextSnapshot.ctxState.stateDigest,
        resultSnapshotDigest: r0.contextSnapshot.snapshotDigest,
        sessionId: 'id-taxonomy-test',
        branchId: 'branch-0',
        recordId: 'id-taxonomy-test:branch-0:rec-0:context_prefill',
        parentRecordId: null,
        recordIndex: 0,
        blockId: null,
        refinementIndex: null,
        absoluteTokenStart: 0,
        absoluteTokenRange: [0, 7],
      });

      const s1 = initTwoTowerDiffusionBlock(s0);
      const r1 = getActiveRecord(s1);
      expect(r1.identity.parentRecordId).toBe(r0.identity.recordId);
      expect(r1.identity.recordIndex).toBe(1);
      expect(r1.identity.blockId).toBe(0);
      expect(r1.identity.refinementIndex).toBeNull();
      expect(r1.identity.sourceCacheDigest).toBe(r0.identity.resultCacheDigest);
      expect(r1.identity.sourceSnapshotDigest).toBe(r0.identity.resultSnapshotDigest);
      expect(r1.identity.resultCacheDigest).toBe(r0.identity.resultCacheDigest);
      expect(r1.identity.resultSnapshotDigest).toBe(r0.identity.resultSnapshotDigest);
      expect(r1.identity.absoluteTokenStart).toBe(8);
      expect(r1.identity.absoluteTokenRange).toEqual([8, 11]);

      const s2 = runTwoTowerDiffusionRefinement(s1);
      const r2 = getActiveRecord(s2);
      expect(r2.identity.parentRecordId).toBe(r1.identity.recordId);
      expect(r2.identity.recordIndex).toBe(2);
      expect(r2.identity.blockId).toBe(0);
      expect(r2.identity.refinementIndex).toBe(0);
      expect(r2.identity.absoluteTokenRange).toEqual([8, 11]);

      const s3 = commitTwoTowerRefinementConfidence(s2);
      const r3 = getActiveRecord(s3);
      expect(r3.identity.parentRecordId).toBe(r2.identity.recordId);
      expect(r3.identity.recordIndex).toBe(3);
      expect(r3.identity.blockId).toBe(0);
      expect(r3.identity.refinementIndex).toBe(0);
      expect(r3.identity.absoluteTokenRange).toEqual([8, 11]);

      let sDone = s3;
      while (getActiveRecord(sDone).lifecyclePhase !== 'completed_uncommitted') {
        sDone = stepTwoTowerSessionNext(sDone);
      }
      const prevUncommitted = getActiveRecord(sDone);
      const sCommit = commitTwoTowerActiveBlockToContext(sDone);
      const rCommit = getActiveRecord(sCommit);
      expect(rCommit.identity.parentRecordId).toBe(prevUncommitted.identity.recordId);
      expect(rCommit.identity.blockId).toBe(0);
      expect(rCommit.identity.refinementIndex).toBeNull();
      expect(rCommit.identity.sourceCacheDigest).toBe(r0.identity.resultCacheDigest);
      expect(rCommit.identity.resultCacheDigest).not.toBe(r0.identity.resultCacheDigest);
      expect(rCommit.identity.resultCacheDigest).toBe(rCommit.contextSnapshot.ctxState.stateDigest);
      expect(rCommit.identity.resultSnapshotDigest).toBe(rCommit.contextSnapshot.snapshotDigest);
      expect(rCommit.identity.absoluteTokenStart).toBe(8);
      expect(rCommit.identity.absoluteTokenRange).toEqual([8, 11]);

      // Verify AR & Mock-AR records carry null blockId/refinementIndex and [P, P] token range
      const sAr = stepTwoTowerSessionAr(sCommit);
      const rAr = getActiveRecord(sAr);
      expect(rAr.identity.blockId).toBeNull();
      expect(rAr.identity.refinementIndex).toBeNull();
      expect(rAr.identity.absoluteTokenStart).toBe(12);
      expect(rAr.identity.absoluteTokenRange).toEqual([12, 12]);

      const sMock = stepTwoTowerSessionMockAr(sAr);
      const rMock = getActiveRecord(sMock);
      expect(rMock.identity.blockId).toBeNull();
      expect(rMock.identity.refinementIndex).toBeNull();
      expect(rMock.identity.absoluteTokenStart).toBe(13);
      expect(rMock.identity.absoluteTokenRange).toEqual([13, 13]);
    });
  });

  describe('7. Strict Mutation Isolation', () => {
    it('prevents mutations on exposed snapshots, residuals, modulation, sim.Y, or samplerPresentation from corrupting earlier records, other branches, or lazy reconstruction', () => {
      const model = createDefaultTwoTowerModel();
      const s0 = createTwoTowerSession(model, {
        eosTokenIds: [],
        confidenceThreshold: 0.9999,
      });
      const rec0 = expectRecordKind(getActiveRecord(s0), 'context_prefill');
      const baselinePrefillK6 = new Float32Array(rec0.contextSnapshot.ctxState.gqaCaches.get(6)!.K);
      const baselinePrefillMamba15Y = new Float32Array(
        rec0.contextExecution.segments[0].getMambaTrace(15).Y,
      );

      // Branch A: advance full diffusion block + commit
      const ranA = runOneSessionDiffusionBlock(s0);
      const ref0Rec = ranA.refinementRecords[0];
      const conf0Rec = ranA.confidenceRecords[0];
      const commitRecA = ranA.commitRecord;

      // Branch B: fork from prefill (idx 0) and run AR step
      let sForked = forkTwoTowerSessionBranch(ranA.session, { fromCursorIndex: 0 });
      sForked = stepTwoTowerSessionAr(sForked, { nextTokenId: 7 });
      const recB = getActiveRecord(sForked);
      const baselineBranchBK6 = new Float32Array(recB.contextSnapshot.ctxState.gqaCaches.get(6)!.K);

      // Capture clean lazy reference outputs BEFORE adversarial mutation
      const cleanDenoiserMamba15Y = new Float32Array(ref0Rec.denoiserPass.getMambaTrace(15).Y);
      const cleanDenoiserMod15Shift = new Float32Array(
        ref0Rec.denoiserPass.getLayerModulation(15)!.shift,
      );
      const cleanCommitSeg0Gqa20Y = new Float32Array(
        commitRecA.contextExecution.segments[0].getGqaTrace(20).Y,
      );
      const cleanCommitSeg1Mamba15Y = new Float32Array(
        commitRecA.contextExecution.segments[1].getMambaTrace(15).Y,
      );

      // Adversarially mutate exposed buffers on Branch A's commit & refinement records!
      commitRecA.contextSnapshot.ctxState.gqaCaches.get(6)!.K.fill(9999);
      commitRecA.contextSnapshot.ctxState.mambaStates.get(1)!.ssmState.fill(-9999);
      ref0Rec.denoiserPass.residualStreams[14].fill(7777);
      const ref0Trace15 = ref0Rec.denoiserPass.getLayerTrace(15);
      ref0Trace15.modulation!.shift.fill(8888);
      ref0Trace15.sim.Y.fill(-8888);
      // Exposed samplerPresentation is deeply frozen (mutation throws TypeError in strict mode)
      expect(Object.isFrozen(conf0Rec.samplerPresentation)).toBe(true);
      expect(Object.isFrozen(conf0Rec.samplerPresentation.slots)).toBe(true);
      expect(Object.isFrozen(conf0Rec.samplerPresentation.slots[0])).toBe(true);
      expect(() => {
        (conf0Rec.samplerPresentation as { confidenceThreshold: number }).confidenceThreshold = -5;
      }).toThrow(TypeError);

      // 1. Earlier prefill snapshot MUST be 0-ULP intact and pass strict validation!
      expect(() => validateContextSnapshot(rec0.contextSnapshot, model.contextModel)).not.toThrow();
      assertBitExactFiniteArray(
        rec0.contextSnapshot.ctxState.gqaCaches.get(6)!.K,
        baselinePrefillK6,
        'earlier prefill GQA K intact',
      );
      rec0.contextExecution.releaseEagerTraces();
      assertBitExactFiniteArray(
        rec0.contextExecution.segments[0].getMambaTrace(15).Y,
        baselinePrefillMamba15Y,
        'earlier prefill lazy Mamba 15 intact',
      );

      // 2. Diverged Branch B snapshot MUST be 0-ULP intact and pass strict validation!
      expect(() => validateContextSnapshot(recB.contextSnapshot, model.contextModel)).not.toThrow();
      assertBitExactFiniteArray(
        recB.contextSnapshot.ctxState.gqaCaches.get(6)!.K,
        baselineBranchBK6,
        'other branch GQA K intact',
      );

      // 3. Lazy Denoiser reconstruction & modulation MUST be 0-ULP intact!
      ref0Rec.denoiserPass.releaseEagerTraces();
      assertBitExactFiniteArray(
        ref0Rec.denoiserPass.getMambaTrace(15).Y,
        cleanDenoiserMamba15Y,
        'lazy Denoiser Mamba 15 Y intact',
      );
      assertBitExactFiniteArray(
        ref0Rec.denoiserPass.getLayerModulation(15)!.shift,
        cleanDenoiserMod15Shift,
        'Denoiser getLayerModulation(15).shift intact',
      );

      // 4. Lazy Context commit segment reconstruction MUST be 0-ULP intact even after mutating commitRecA.contextSnapshot!
      commitRecA.contextExecution.releaseEagerTraces();
      assertBitExactFiniteArray(
        commitRecA.contextExecution.segments[0].getGqaTrace(20).Y,
        cleanCommitSeg0Gqa20Y,
        'lazy Context commit seg0 GQA 20 Y intact',
      );
      assertBitExactFiniteArray(
        commitRecA.contextExecution.segments[1].getMambaTrace(15).Y,
        cleanCommitSeg1Mamba15Y,
        'lazy Context commit seg1 Mamba 15 Y intact',
      );

      // 5. Mutating exposed denoiserPass.logits BEFORE commitTwoTowerRefinementConfidence cannot corrupt sampler!
      const sFresh = runTwoTowerDiffusionRefinement(initTwoTowerDiffusionBlock(s0));
      const cleanConf = commitTwoTowerRefinementConfidence(sFresh);
      const sTampered = runTwoTowerDiffusionRefinement(initTwoTowerDiffusionBlock(s0));
      const tamperedRefRec = getActiveRecord(sTampered);
      if (tamperedRefRec.kind !== 'diffusion_refinement') throw new Error('Expected refinement');
      tamperedRefRec.denoiserPass.logits.fill(-500);
      const isolatedConf = commitTwoTowerRefinementConfidence(sTampered);
      const cleanConfRec = getActiveRecord(cleanConf);
      const isolatedConfRec = getActiveRecord(isolatedConf);
      if (
        cleanConfRec.kind !== 'diffusion_confidence_commit' ||
        isolatedConfRec.kind !== 'diffusion_confidence_commit'
      ) {
        throw new Error('Expected diffusion_confidence_commit');
      }
      expect(isolatedConfRec.unmaskStep.xtNext).toEqual(cleanConfRec.unmaskStep.xtNext);
      assertBitExactFiniteArray(
        isolatedConfRec.unmaskStep.confidence,
        cleanConfRec.unmaskStep.confidence,
        'sampler isolated from pre-commit denoiserPass.logits mutation',
      );
    });
  });

  describe('8. Split-Segment Context Trace Inspection & Unified Trace-Origin Taxonomy', () => {
    it('verifies split segments for prefill & commit, single segments for S=1/B=1, and all 4 trace origins', () => {
      const model = createDefaultTwoTowerModel();

      // Prefill S = 8 (2 segments) -> active record retains eager traces ('context_computed')
      const s0 = createTwoTowerSession(model, {
        promptTokenIds: [0, 1, 2, 3, 4, 5, 6, 7],
        eosTokenIds: [],
      });
      const latest0 = getLatestContextExecution(s0)!;
      expect(latest0.record.kind).toBe('context_prefill');
      expect(latest0.branchCursorIndex).toBe(0);
      expect(latest0.record.contextSnapshot.stateOrigin).toBe('persistent_cache');
      expect(latest0.execution.segments).toHaveLength(2);
      expect(latest0.execution.segments[0].tokenRange).toEqual([0, 6]);
      expect(latest0.execution.segments[1].tokenRange).toEqual([7, 7]);
      expect(latest0.execution.segments[0].getLayerTrace(1).traceOrigin).toBe('context_computed');
      expect(latest0.execution.segments[1].getLayerTrace(6).traceOrigin).toBe('context_computed');
      // Non-eager bay lazily reconstructs with 'context_reconstructed'
      expect(latest0.execution.segments[0].getLayerTrace(13).traceOrigin).toBe(
        'context_reconstructed',
      );

      // Step into diffusion_refinement -> Denoiser trace is 'denoiser_scratch', and getLatestContextExecution
      // STILL returns prefill (now lazily reconstructed because retainEagerTracesOnHistory defaults to false!)
      const sRef = runTwoTowerDiffusionRefinement(initTwoTowerDiffusionBlock(s0));
      const refRec = getActiveRecord(sRef);
      if (refRec.kind !== 'diffusion_refinement') throw new Error('Expected refinement');
      expect(refRec.denoiserPass.getLayerTrace(1).traceOrigin).toBe('denoiser_scratch');
      expect(refRec.denoiserPass.getLayerTrace(20).traceOrigin).toBe('denoiser_scratch');

      const latestDuringRef = getLatestContextExecution(sRef)!;
      expect(latestDuringRef.record.kind).toBe('context_prefill');
      expect(latestDuringRef.execution.segments[0].getLayerTrace(1).traceOrigin).toBe(
        'context_reconstructed',
      );

      // Complete and commit B = 4 block -> active commit record retains eager 'context_computed' across BOTH split segments!
      let sDone = sRef;
      while (getActiveRecord(sDone).lifecyclePhase !== 'completed_uncommitted') {
        sDone = stepTwoTowerSessionNext(sDone);
      }
      const sCommit = commitTwoTowerActiveBlockToContext(sDone);
      const latestCommit = getLatestContextExecution(sCommit)!;
      expect(latestCommit.record.kind).toBe('diffusion_block_commit');
      expect(latestCommit.execution.segments).toHaveLength(2);
      expect(latestCommit.execution.segments[0].tokenRange).toEqual([8, 10]);
      expect(latestCommit.execution.segments[1].tokenRange).toEqual([11, 11]);
      expect(latestCommit.execution.segments[0].getLayerTrace(1).traceOrigin).toBe(
        'context_computed',
      );
      expect(latestCommit.execution.segments[1].getLayerTrace(6).traceOrigin).toBe(
        'context_computed',
      );

      // Single-segment S = 1 prefill and B = 1 block commit
      const sSingle = createTwoTowerSession(model, {
        promptTokenIds: [5],
        blockSize: 1,
        stepsPerBlock: 1,
        eosTokenIds: [],
      });
      expect(getLatestContextExecution(sSingle)!.execution.segments).toHaveLength(1);
      expect(getLatestContextExecution(sSingle)!.execution.segments[0].segmentRole).toBe(
        'single_segment',
      );
      const ranSingle = runOneSessionDiffusionBlock(sSingle);
      expect(ranSingle.commitRecord.contextExecution.segments).toHaveLength(1);
      expect(ranSingle.commitRecord.contextExecution.segments[0].segmentRole).toBe(
        'single_segment',
      );
      expect(ranSingle.commitRecord.contextExecution.segments[0].tokenRange).toEqual([1, 1]);
    });
  });

  describe('9. Exact Capacity Exhaustion & Full-Block EOS Termination Semantics', () => {
    it('fills P=8 to kvCapacity=24 cleanly, rejects P+B > kvCapacity while permitting AR/Mock-AR/B=4, and blocks terminated branches', () => {
      const model = createDefaultTwoTowerModel();
      let s = createTwoTowerSession(model, {
        promptTokenIds: [0, 1, 2, 3, 4, 5, 6, 7],
        kvCapacity: 24,
        blockSize: 4,
        stepsPerBlock: 2,
        eosTokenIds: [],
      });

      // Fill 3 blocks of B = 4: 8 -> 12 -> 16 -> 20
      for (let b = 0; b < 3; b++) {
        s = runOneSessionDiffusionBlock(s).session;
      }
      expect(getActiveRecord(s).committedPrefixLen).toBe(20);
      expect(getActiveRecord(s).lifecyclePhase).toBe('ready');

      // At P = 20, kvCapacity = 24: B = 8 MUST be rejected while AR, Mock-AR, and B = 4 stay legal!
      const sWide = updateTwoTowerSessionConfig(s, { blockSize: 8 });
      const capsWide = getSessionCapabilities(sWide);
      expect(capsWide.remainingKvCapacity).toBe(4);
      expect(capsWide.canInitDiffusionBlock).toBe(false);
      expect(capsWide.canStepNext).toBe(false);
      expect(capsWide.canStepAr).toBe(true);
      expect(capsWide.canStepMockAr).toBe(true);
      expect(() => initTwoTowerDiffusionBlock(sWide)).toThrow(RangeError);
      expect(() => stepTwoTowerSessionNext(sWide)).toThrow(RangeError);

      // Forking from P = 20 allows AR and Mock-AR cleanly
      const sArBranch = stepTwoTowerSessionAr(forkTwoTowerSessionBranch(sWide));
      expect(getActiveRecord(sArBranch).committedPrefixLen).toBe(21);
      const sMockBranch = stepTwoTowerSessionMockAr(forkTwoTowerSessionBranch(sWide));
      expect(getActiveRecord(sMockBranch).committedPrefixLen).toBe(21);

      // Switching back to B = 4 fills cleanly from 20 -> 24 and enters capacity_exhausted!
      const sB4 = updateTwoTowerSessionConfig(sWide, { blockSize: 4 });
      expect(getSessionCapabilities(sB4).canInitDiffusionBlock).toBe(true);
      const sFull = runOneSessionDiffusionBlock(sB4).session;
      const fullRec = getActiveRecord(sFull);
      expect(fullRec.committedPrefixLen).toBe(24);
      expect(fullRec.lifecyclePhase).toBe('capacity_exhausted');
      expect(fullRec.termination).toEqual({
        reason: 'capacity_exhausted',
        committedPrefixLen: 24,
        kvCapacity: 24,
      });

      const fullCaps = getSessionCapabilities(sFull);
      expect(fullCaps.isTerminated).toBe(true);
      expect(fullCaps.canStepAr).toBe(false);
      expect(fullCaps.canStepMockAr).toBe(false);
      expect(fullCaps.canInitDiffusionBlock).toBe(false);
      expect(fullCaps.canStepNext).toBe(false);
      expect(() => stepTwoTowerSessionAr(sFull)).toThrow(RangeError);
      expect(() => stepTwoTowerSessionMockAr(sFull)).toThrow(RangeError);
      expect(() => initTwoTowerDiffusionBlock(sFull)).toThrow(RangeError);
      expect(() => stepTwoTowerSessionNext(sFull)).toThrow(RangeError);

      // Rewinding from capacity_exhausted to earlier non-terminal record STILL works and can branch!
      const sRewound = navigateTwoTowerSession(sFull, 0);
      expect(getSessionCapabilities(sRewound).isTerminated).toBe(false);
      const sResumed = stepTwoTowerSessionAr(sRewound);
      expect(getActiveRecord(sResumed).committedPrefixLen).toBe(9);

      // Creating a session with prompt length === kvCapacity enters capacity_exhausted immediately
      const sAtCap = createTwoTowerSession(model, {
        promptTokenIds: [0, 1, 2, 3],
        kvCapacity: 4,
      });
      expect(getActiveRecord(sAtCap).lifecyclePhase).toBe('capacity_exhausted');
      expect(getSessionCapabilities(sAtCap).canStepAr).toBe(false);
      expect(() => stepTwoTowerSessionAr(sAtCap)).toThrow(RangeError);

      // Initializing a block with already-unmasked custom tokens enters completed_uncommitted directly
      const sPreUnmasked = initTwoTowerDiffusionBlock(
        createTwoTowerSession(model, { eosTokenIds: [] }),
        { blockSize: 4, initialBlockTokens: [1, 2, 3, 4] },
      );
      expect(getActiveRecord(sPreUnmasked).lifecyclePhase).toBe('completed_uncommitted');
      expect(getSessionCapabilities(sPreUnmasked).nextTransitionKind).toBe('diffusion_block_commit');
      const sPreCommitted = stepTwoTowerSessionNext(sPreUnmasked);
      expect(getActiveRecord(sPreCommitted).committedPrefixLen).toBe(12);
    });

    it('enforces full-block EOS termination (and AR/Mock-AR EOS) while permitting rewind & branching from pre-EOS history', () => {
      const model = createDefaultTwoTowerModel();

      // Probe block 0's deterministic tokens first
      const probe = runOneSessionDiffusionBlock(
        createTwoTowerSession(model, { eosTokenIds: [] }),
      );
      const eosTrigger = probe.commitRecord.committedBlockTokens[1];

      let s = createTwoTowerSession(model, {
        eosTokenIds: [TWOTOWER_MINI_EOS_TOKEN_ID, eosTrigger],
      });
      s = initTwoTowerDiffusionBlock(s);
      while (getActiveRecord(s).lifecyclePhase !== 'completed_uncommitted') {
        s = stepTwoTowerSessionNext(s);
      }

      // At completed_uncommitted: pendingEosInBlock is inspectable BEFORE Context commit,
      // and session has NOT yet terminated until explicit block commit!
      const uncommittedRec = getActiveRecord(s);
      expect(uncommittedRec.lifecyclePhase).toBe('completed_uncommitted');
      expect(uncommittedRec.termination).toBeNull();
      expect(uncommittedRec.activeBlock!.pendingEosInBlock).toBe(true);
      expect(uncommittedRec.activeBlock!.pendingEosTokenIds).toContain(eosTrigger);

      // Explicit block commit triggers eos_terminated
      const sEos = commitTwoTowerActiveBlockToContext(s);
      const eosRec = getActiveRecord(sEos);
      expect(eosRec.lifecyclePhase).toBe('eos_terminated');
      expect(eosRec.termination).toEqual({
        reason: 'eos',
        mode: 'mask_diffusion',
        committedPrefixLen: 12,
        blockId: 0,
        triggerTokenIds: [eosTrigger],
      });

      // Terminated branch refuses further generation
      expect(() => initTwoTowerDiffusionBlock(sEos)).toThrow(RangeError);
      expect(() => stepTwoTowerSessionAr(sEos)).toThrow(RangeError);
      expect(() => stepTwoTowerSessionMockAr(sEos)).toThrow(RangeError);
      expect(() => stepTwoTowerSessionNext(sEos)).toThrow(RangeError);

      // Rewinding to pre-EOS history (idx 0) and branching with different eosTokenIds continues cleanly!
      let sBranchPreEos = forkTwoTowerSessionBranch(sEos, {
        fromCursorIndex: 0,
        configPatch: { eosTokenIds: [TWOTOWER_MINI_EOS_TOKEN_ID] },
      });
      sBranchPreEos = runOneSessionDiffusionBlock(sBranchPreEos).session;
      expect(getActiveRecord(sBranchPreEos).lifecyclePhase).toBe('ready');

      // Verify AR and Mock-AR EOS termination too
      const sArEos = stepTwoTowerSessionAr(
        createTwoTowerSession(model, { mode: 'ar', eosTokenIds: [TWOTOWER_MINI_EOS_TOKEN_ID] }),
        { nextTokenId: TWOTOWER_MINI_EOS_TOKEN_ID },
      );
      expect(getActiveRecord(sArEos).lifecyclePhase).toBe('eos_terminated');
      expect(getActiveRecord(sArEos).termination).toEqual({
        reason: 'eos',
        mode: 'ar',
        committedPrefixLen: 9,
        blockId: null,
        triggerTokenIds: [TWOTOWER_MINI_EOS_TOKEN_ID],
      });

      const sMockEos = stepTwoTowerSessionMockAr(
        createTwoTowerSession(model, {
          mode: 'mock_ar',
          eosTokenIds: [TWOTOWER_MINI_EOS_TOKEN_ID],
        }),
        { nextTokenId: TWOTOWER_MINI_EOS_TOKEN_ID },
      );
      expect(getActiveRecord(sMockEos).lifecyclePhase).toBe('eos_terminated');
      expect(getActiveRecord(sMockEos).termination).toEqual({
        reason: 'eos',
        mode: 'mock_ar',
        committedPrefixLen: 9,
        blockId: null,
        triggerTokenIds: [TWOTOWER_MINI_EOS_TOKEN_ID],
      });
    });
  });

  describe('10. Deduplicated Memory Accounting & Inactive Trace Pruning', () => {
    it('proves 4 refinements share 1 conditioning base (~78 KiB) and pruning inactive eager traces strictly lowers trace bytes with 0-ULP reconstruction', () => {
      const model = createDefaultTwoTowerModel();
      const snapBaseline = runTwoTowerContextPrefill(model, DEFAULT_TWOTOWER_PROMPT_TOKENS, {
        kvCapacity: 24,
      });
      const expectedSingleSnapshotBytes = measureContextSnapshotTypedArrayBytes(snapBaseline);

      // Retain eager traces on history so we can verify pruning drops them cleanly!
      const s0 = createTwoTowerSession(model, {
        confidenceThreshold: 0.9999, // force all 4 refinements
        eosTokenIds: [],
        retainEagerTracesOnHistory: true,
      });

      const ran = runOneSessionDiffusionBlock(s0);
      const sCommitted = ran.session;
      expect(ran.refinementRecords).toHaveLength(4);

      const memBeforePrune = measureTwoTowerSessionRetainedBytes(sCommitted);

      // 1. Exact accounting identity (session-retained excludes read-only shared model weights)
      expect(memBeforePrune.totalRetainedBytes).toBe(
        memBeforePrune.contextSnapshotBytes +
          memBeforePrune.blockConditioningBytes +
          memBeforePrune.contextTraceBytes +
          memBeforePrune.denoiserTraceBytes +
          memBeforePrune.samplerAndSelectionBytes,
      );
      expect(memBeforePrune.sharedModelWeightBytes).toBe(model.retainedWeightBytes());
      expect(memBeforePrune.totalIncludingModelWeightsBytes).toBe(
        memBeforePrune.totalRetainedBytes + memBeforePrune.sharedModelWeightBytes,
      );

      // 2. Across ALL 4 refinements of Block 0: uniqueBlockConditioningCount === 1 (~78 KiB = 79,872 B, NOT 4 * 79,872 B!)
      expect(memBeforePrune.uniqueBlockConditioningCount).toBe(1);
      expect(memBeforePrune.blockConditioningBytes).toBe(79_872);
      expect(memBeforePrune.blockConditioningBytes).toBe(
        measureConditioningTypedArrayBytes(ran.refinementRecords[0].conditioning),
      );

      // 3. Only 2 distinct Context snapshots (prefill P=8 + block commit P=12), NOT 10!
      expect(memBeforePrune.uniqueContextSnapshotCount).toBe(2);
      expect(memBeforePrune.branchCount).toBe(1);
      // prefill (P=8 -> 150,832 B) + commit (P=12 -> 150,832 + 16 B for prefixTokenIds)
      expect(memBeforePrune.contextSnapshotBytes).toBe(expectedSingleSnapshotBytes * 2 + 16);
      // Records: 1 prefill + 1 init + 4 ref + 4 conf + 1 commit = 11 unique records
      expect(memBeforePrune.uniqueRecordCount).toBe(11);

      // Capture pristine historical traces before pruning
      const prefillRec = getActiveBranch(sCommitted).records[0];
      if (prefillRec.kind !== 'context_prefill') throw new Error('Expected prefill');
      const ref0CleanY = new Float32Array(ran.refinementRecords[0].denoiserPass.getMambaTrace(1).Y);
      const prefillCleanY = new Float32Array(
        prefillRec.contextExecution.segments[0].getMambaTrace(1).Y,
      );

      // Prune inactive historical eager traces (leaving active commit record's eager traces intact)
      const sPruned = pruneInactiveSessionTraces(sCommitted);
      const memAfterPrune = measureTwoTowerSessionRetainedBytes(sPruned);

      expect(memAfterPrune.contextTraceBytes).toBeLessThan(memBeforePrune.contextTraceBytes);
      expect(memAfterPrune.denoiserTraceBytes).toBeLessThan(memBeforePrune.denoiserTraceBytes);
      expect(memAfterPrune.totalRetainedBytes).toBeLessThan(memBeforePrune.totalRetainedBytes);

      // Active commit record STILL keeps its eager traces ('context_computed')!
      const activeCommit = getActiveRecord(sPruned);
      if (activeCommit.kind !== 'diffusion_block_commit') throw new Error('Expected commit');
      expect(activeCommit.contextExecution.segments[0].getLayerTraceOrigin(1)).toBe(
        'context_computed',
      );
      // Historical records had eager traces cleared
      expect(prefillRec.contextExecution.segments[0].retainedTraceBayIds()).toEqual([]);
      expect(ran.refinementRecords[0].denoiserPass.retainedTraceBayIds()).toEqual([]);

      // Lazy reconstruction after pruning is STILL 0-ULP bit-exact!
      const rebuiltPrefill = prefillRec.contextExecution.segments[0].getLayerTrace(1);
      expect(rebuiltPrefill.traceOrigin).toBe('context_reconstructed');
      if (rebuiltPrefill.kind !== 'M') throw new Error('Expected Mamba');
      assertBitExactFiniteArray(rebuiltPrefill.sim.Y, prefillCleanY, 'pruned prefill lazy Mamba 1');

      const rebuiltRef0 = ran.refinementRecords[0].denoiserPass.getLayerTrace(1);
      expect(rebuiltRef0.traceOrigin).toBe('denoiser_scratch');
      if (rebuiltRef0.kind !== 'M') throw new Error('Expected Mamba');
      assertBitExactFiniteArray(rebuiltRef0.sim.Y, ref0CleanY, 'pruned ref0 lazy Mamba 1');

      // Verify structural sharing across forked branches does NOT duplicate snapshot/conditioning bytes!
      const sFork = forkTwoTowerSessionBranch(sPruned, { fromCursorIndex: 5 });
      const memAfterFork = measureTwoTowerSessionRetainedBytes(sFork);
      expect(memAfterFork.branchCount).toBe(2);
      expect(memAfterFork.uniqueRecordCount).toBe(memAfterPrune.uniqueRecordCount);
      expect(memAfterFork.contextSnapshotBytes).toBe(memAfterPrune.contextSnapshotBytes);
      expect(memAfterFork.blockConditioningBytes).toBe(memAfterPrune.blockConditioningBytes);
    });
  });
});
