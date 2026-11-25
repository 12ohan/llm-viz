import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { GQA_SIM_CONFIG } from '../src/math/gqaSim';
import { MAMBA_SIM_CONFIG } from '../src/math/mambaSim';
import { MOE_SIM_CONFIG } from '../src/math/moeSim';
import { memoryCrossovers } from '../src/model/memory';
import { NEMOTRON_SPEC } from '../src/model/spec';
import {
  TWOTOWER_MODEL_CARD_OPERATIONAL_CONTEXT_TOKENS,
  TWOTOWER_SAMPLER_CALIBRATION_DISCLAIMER,
  TWOTOWER_STANDARD_CONTEXT_LENGTHS,
  buildAdvancedSamplerInspectorModel,
  buildContextLengthScalingSeries,
  buildTwoTowerArchitecturalReadout,
  buildTwoTowerConfigScopeModel,
  buildTwoTowerTimelineModel,
} from '../src/twotower/archReadouts';
import {
  TWOTOWER_CONFIDENCE_PRESETS,
  assertPresentationSafeNoInfinity,
} from '../src/twotower/sampler';
import {
  TWOTOWER_MINI_EOS_TOKEN_ID,
  TWOTOWER_MINI_MASK_TOKEN_ID,
  createDefaultTwoTowerModel,
} from '../src/twotower/parameters';
import {
  commitTwoTowerActiveBlockToContext,
  commitTwoTowerRefinementConfidence,
  createTwoTowerSession,
  deleteTwoTowerSessionBranch,
  forkTwoTowerSessionBranch,
  getActiveRecord,
  initTwoTowerDiffusionBlock,
  measureTwoTowerSessionRetainedBytes,
  navigateTwoTowerSession,
  navigateTwoTowerSessionBack,
  navigateTwoTowerSessionForward,
  pruneInactiveSessionTraces,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  stepTwoTowerSessionMockAr,
  switchTwoTowerSessionBranch,
  updateTwoTowerSessionConfig,
  type TwoTowerSession,
} from '../src/twotower/session';
import {
  TWOTOWER_DISCREPANCY_LEDGER,
  TWOTOWER_MINI_CONFIG,
  TWOTOWER_SPEC,
  twoTowerMemoryFor,
} from '../src/twotower/spec';

const SHARED_MODEL = createDefaultTwoTowerModel();

function completeActiveDiffusionBlock(
  session: TwoTowerSession,
  commitToContext = true,
): TwoTowerSession {
  let s = session;
  while (getActiveRecord(s).lifecyclePhase !== 'completed_uncommitted') {
    s = runTwoTowerDiffusionRefinement(s);
    s = commitTwoTowerRefinementConfidence(s);
  }
  if (commitToContext) {
    s = commitTwoTowerActiveBlockToContext(s);
  }
  return s;
}

describe('1. Production Architectural Readouts & Context-Length Scaling (Wave 4)', () => {
  it('verifies exact independent tower parameter & byte inventory and 0-byte safetensors audit', () => {
    const readout = buildTwoTowerArchitecturalReadout(8_192, 16, 'mask_diffusion');
    const inv = readout.inventory;
    const specInv = TWOTOWER_SPEC.derivedInventoryCalculations;
    const specMeta = TWOTOWER_SPEC.directlyInspectedMetadata;

    expect(inv.contextTower.params).toBe(specInv.singleTowerParams);
    expect(inv.contextTower.params).toBe(31_577_940_288);
    expect(inv.contextTower.payloadBytes).toBe(specInv.contextTowerPayloadBytes);
    expect(inv.contextTower.payloadBytes).toBe(63_155_886_464);
    expect(inv.contextTower.tensorCount).toBe(6_243);
    expect(inv.contextTower.fp32BiasTensorCount).toBe(23);
    expect(inv.contextTower.fp32BiasParams).toBe(2_944);

    expect(inv.denoiserBackbone.params).toBe(specInv.singleTowerParams);
    expect(inv.denoiserBackbone.payloadBytes).toBe(specInv.denoiserTowerBackbonePayloadBytes);
    expect(inv.denoiserBackbone.tensorCount).toBe(6_243);

    expect(inv.denoiserTimeConditioning.params).toBe(specInv.timeConditioningTotalParams);
    expect(inv.denoiserTimeConditioning.params).toBe(30_022_272);
    expect(inv.denoiserTimeConditioning.payloadBytes).toBe(specInv.timeConditioningPayloadBytes);
    expect(inv.denoiserTimeConditioning.payloadBytes).toBe(60_044_544);
    expect(inv.denoiserTimeConditioning.tensorCount).toBe(58);
    expect(inv.denoiserTimeConditioning.components).toHaveLength(3);
    const compParams = inv.denoiserTimeConditioning.components.reduce((s, c) => s + c.params, 0);
    const compTensors = inv.denoiserTimeConditioning.components.reduce(
      (s, c) => s + c.tensorCount,
      0,
    );
    expect(compParams).toBe(30_022_272);
    expect(compTensors).toBe(58);

    expect(inv.grandTotal.params).toBe(63_185_902_848);
    expect(inv.grandTotal.payloadBytes).toBe(126_371_817_472);
    expect(inv.grandTotal.payloadByteDiscrepancy).toBe(0);
    expect(inv.grandTotal.shardCount).toBe(24);
    expect(inv.grandTotal.tensorCount).toBe(12_544);
    expect(inv.grandTotal.fp32BiasTensorCount).toBe(46);
    expect(inv.grandTotal.fp32BiasParams).toBe(5_888);
    expect(inv.grandTotal.fp32BiasPayloadBytes).toBe(23_552);
    expect(inv.grandTotal.bf16TensorCount + inv.grandTotal.fp32BiasTensorCount).toBe(
      specMeta.safetensorsTensorCount,
    );

    // Preserves entire paper-vs-checkpoint discrepancy ledger cleanly
    expect(readout.discrepancyLedger).toHaveLength(TWOTOWER_DISCREPANCY_LEDGER.length);
    for (let i = 0; i < TWOTOWER_DISCREPANCY_LEDGER.length; i++) {
      expect(readout.discrepancyLedger[i].id).toBe(TWOTOWER_DISCREPANCY_LEDGER[i].id);
      expect(readout.discrepancyLedger[i].authoritativeResolution).toBe(
        TWOTOWER_DISCREPANCY_LEDGER[i].authoritativeResolution,
      );
      expect(readout.discrepancyLedger[i].index).toBe(i + 1);
    }
    expect(readout.discrepancyLedger.some((e) => e.id === 'adaln_param_count')).toBe(true);
    expect(readout.discrepancyLedger.some((e) => e.id === 'max_context_length')).toBe(true);
  });

  it('distinguishes Two Resident Weight Sets != Two Persistent Caches across all 3 execution modes', () => {
    const P = 16_384;
    const B = 16;

    const diff = buildTwoTowerArchitecturalReadout(P, B, 'mask_diffusion').weightsVsCaches;
    const mockAr = buildTwoTowerArchitecturalReadout(P, B, 'mock_ar').weightsVsCaches;
    const ar = buildTwoTowerArchitecturalReadout(P, B, 'ar').weightsVsCaches;

    expect(diff.headline).toBe('Two Resident Weight Sets ≠ Two Persistent Caches');

    // Persistent Context cache is IDENTICAL across all 3 modes (single persistent cache!)
    const expectedMamba = 49_364_992; // 47.08 MiB
    const expectedKv = P * 6_144; // P * 6 KiB
    for (const wvc of [diff, mockAr, ar]) {
      expect(wvc.persistentContextCache.owner).toBe('context_tower');
      expect(wvc.persistentContextCache.isUniquePersistentSequenceCache).toBe(true);
      expect(wvc.persistentContextCache.fixedMambaBytes).toBe(expectedMamba);
      expect(wvc.persistentContextCache.fixedMambaDisplay).toBe('47.08 MiB');
      expect(wvc.persistentContextCache.gqaKvBytes).toBe(expectedKv);
      expect(wvc.persistentContextCache.kvBytesPerToken).toBe(6_144);
      expect(wvc.persistentContextCache.totalBytes).toBe(expectedMamba + expectedKv);
      expect(wvc.crossGpuMirrorVsLogicalSharing.logicalPersistentCacheCount).toBe(1);
    }

    // Transient Denoiser workspace differs strictly by mode
    expect(diff.residentWeights.activeTowerCount).toBe(2);
    expect(diff.transientDenoiserWorkspace.workspaceKind).toBe('ephemeral_block_kv');
    expect(diff.transientDenoiserWorkspace.ephemeralBlockKvBytes).toBe(B * 6_144);
    expect(diff.transientDenoiserWorkspace.mockArMambaS2SnapshotBytes).toBe(0);
    expect(diff.transientDenoiserWorkspace.totalTransientBytes).toBe(B * 6_144);
    expect(diff.transientDenoiserWorkspace.discardedAfterStep).toBe(true);
    expect(diff.crossGpuMirrorVsLogicalSharing.crossGpuMirrorBytes).toBe(
      expectedMamba + expectedKv,
    );

    expect(mockAr.residentWeights.activeTowerCount).toBe(2);
    expect(mockAr.transientDenoiserWorkspace.workspaceKind).toBe('mock_ar_mamba_s2_snapshot');
    expect(mockAr.transientDenoiserWorkspace.ephemeralBlockKvBytes).toBe(0);
    expect(mockAr.transientDenoiserWorkspace.mockArMambaS2SnapshotBytes).toBe(expectedMamba);
    expect(mockAr.transientDenoiserWorkspace.totalTransientBytes).toBe(expectedMamba);
    expect(mockAr.crossGpuMirrorVsLogicalSharing.crossGpuMirrorBytes).toBe(
      expectedMamba + (P - 1) * 6_144,
    );

    expect(ar.residentWeights.activeTowerCount).toBe(1);
    expect(ar.transientDenoiserWorkspace.workspaceKind).toBe('none');
    expect(ar.transientDenoiserWorkspace.totalTransientBytes).toBe(0);
    expect(ar.transientDenoiserWorkspace.discardedAfterStep).toBe(false);
    expect(ar.crossGpuMirrorVsLogicalSharing.crossGpuMirrorBytes).toBe(0);
  });

  it('verifies Production Precision vs Miniature Arithmetic directly against GQA_SIM_CONFIG & MAMBA_SIM_CONFIG', () => {
    const matrix = buildTwoTowerArchitecturalReadout(4_096, 16, 'mask_diffusion').precisionVsMiniature;

    // Production facts
    expect(matrix.production.primaryWeightsAndActivations).toBe('BF16');
    expect(matrix.production.ssmScanPrecision).toBe('FP32');
    expect(matrix.production.moeBiasPrecision).toBe('FP32');
    expect(matrix.production.moeRouterPrecision).toBe('FP64');
    expect(matrix.production.d).toBe(2_688);
    expect(matrix.production.vocabSize).toBe(131_072);
    expect(matrix.production.gqaQueryHeads).toBe(32);
    expect(matrix.production.gqaKvHeads).toBe(2);
    expect(matrix.production.gqaHeadDim).toBe(128);
    expect(matrix.production.mambaHeads).toBe(64);
    expect(matrix.production.mambaGroups).toBe(8);
    expect(matrix.production.mambaStateDim).toBe(128);
    expect(matrix.production.maxPositionEmbeddings).toBe(262_144);

    // Miniature facts strictly match live simulator configs
    expect(matrix.miniature.d).toBe(TWOTOWER_MINI_CONFIG.d);
    expect(matrix.miniature.d).toBe(8);
    expect(matrix.miniature.vocabSize).toBe(TWOTOWER_MINI_CONFIG.vocabSize);
    expect(matrix.miniature.vocabSize).toBe(34);
    expect(matrix.miniature.kvCapacity).toBe(TWOTOWER_MINI_CONFIG.kvCapacity);
    expect(matrix.miniature.kvCapacity).toBe(24);
    expect(matrix.miniature.gqaQueryHeads).toBe(GQA_SIM_CONFIG.nQ);
    expect(matrix.miniature.gqaQueryHeads).toBe(32);
    expect(matrix.miniature.gqaKvHeads).toBe(GQA_SIM_CONFIG.nKv);
    expect(matrix.miniature.gqaKvHeads).toBe(2);
    expect(matrix.miniature.gqaHeadDim).toBe(GQA_SIM_CONFIG.dHead);
    expect(matrix.miniature.gqaHeadDim).toBe(4);
    expect(matrix.miniature.mambaHeads).toBe(MAMBA_SIM_CONFIG.nHeads);
    expect(matrix.miniature.mambaHeads).toBe(16);
    expect(matrix.miniature.mambaGroups).toBe(MAMBA_SIM_CONFIG.nGroups);
    expect(matrix.miniature.mambaGroups).toBe(8);
    expect(matrix.miniature.mambaHeadDim).toBe(MAMBA_SIM_CONFIG.dHead);
    expect(matrix.miniature.mambaStateDim).toBe(MAMBA_SIM_CONFIG.dState);
    expect(matrix.miniature.mambaSsmMatrixShape).toEqual([4, 4]);
    expect(matrix.miniature.moeRoutedExperts).toBe(MOE_SIM_CONFIG.nRoutedExperts);
    expect(matrix.miniature.moeActiveTopK).toBe(MOE_SIM_CONFIG.topK);
    expect(matrix.miniature.moeRouterPrecision).toBe('Float64 (preserved)');
    expect(matrix.rows.length).toBeGreaterThanOrEqual(8);
  });

  it('builds exact context-length scaling series, marks supported vs extrapolated regimes, and handles custom prefixTokens', () => {
    const standard = buildContextLengthScalingSeries(16, 'mask_diffusion');
    const expectedCross = memoryCrossovers(NEMOTRON_SPEC, 'bf16');

    expect(standard.points.map((p) => p.tokens)).toEqual([...TWOTOWER_STANDARD_CONTEXT_LENGTHS]);
    expect(standard.maxPositionEmbeddings).toBe(262_144);
    expect(standard.modelCardOperationalContextTokens).toBe(
      TWOTOWER_MODEL_CARD_OPERATIONAL_CONTEXT_TOKENS,
    );
    expect(standard.crossoverTokens).toBe(expectedCross.kvExceedsMambaTokens);
    expect(standard.crossoverTokens).toBe(8_035);
    expect(standard.crossoverTokensExact).toBeCloseTo(49_364_992 / 6_144, 10);
    expect(standard.beatsFull52LayerGqaTokens).toBe(1_048);

    for (const pt of standard.points) {
      const direct = twoTowerMemoryFor(pt.tokens, 16, 'mask_diffusion');
      expect(pt.persistentTotalBytes).toBe(direct.persistentContextTotalBytes);
      expect(pt.ephemeralWorkspaceBytes).toBe(direct.ephemeralDenoiserBlockKvBytes);
      expect(pt.mambaSharePct + pt.kvSharePct).toBeCloseTo(100, 10);
      if (pt.tokens <= 262_144) {
        expect(pt.regime).toBe('supported');
      } else {
        expect(pt.regime).toBe('extrapolated_beyond_max_pos');
      }
      expect(pt.isCustomPoint).toBe(false);
    }

    // Verify crossover flip between 4K (< 8,035) and 16K (> 8,035)
    const pt4k = standard.points.find((p) => p.tokens === 4_096)!;
    const pt16k = standard.points.find((p) => p.tokens === 16_384)!;
    expect(pt4k.mambaSharePct).toBeGreaterThan(50);
    expect(pt16k.kvSharePct).toBeGreaterThan(50);

    // Custom prefixTokens inserted in ascending order without duplicating standard lengths
    const withCustom = buildContextLengthScalingSeries(16, 'mask_diffusion', 12_345);
    expect(withCustom.points).toHaveLength(TWOTOWER_STANDARD_CONTEXT_LENGTHS.length + 1);
    const customPt = withCustom.points.find((p) => p.tokens === 12_345)!;
    expect(customPt.isCustomPoint).toBe(true);
    expect(customPt.isSelectedPrefix).toBe(true);
    expect(customPt.regime).toBe('supported');
    for (let i = 1; i < withCustom.points.length; i++) {
      expect(withCustom.points[i].tokens).toBeGreaterThan(withCustom.points[i - 1].tokens);
    }

    // Standard prefixTokens marks selected without duplicating
    const withStd = buildContextLengthScalingSeries(16, 'mask_diffusion', 262_144);
    expect(withStd.points).toHaveLength(TWOTOWER_STANDARD_CONTEXT_LENGTHS.length);
    const pt256k = withStd.points.find((p) => p.tokens === 262_144)!;
    expect(pt256k.isCustomPoint).toBe(false);
    expect(pt256k.isSelectedPrefix).toBe(true);
    expect(pt256k.regime).toBe('supported');

    // Edge cases: L = 0 and invalid inputs
    const zeroReadout = buildTwoTowerArchitecturalReadout(0, 0, 'ar');
    expect(zeroReadout.weightsVsCaches.persistentContextCache.gqaKvBytes).toBe(0);
    expect(() => buildTwoTowerArchitecturalReadout(-1, 16, 'mask_diffusion')).toThrow(RangeError);
    expect(() => buildContextLengthScalingSeries(-4, 'mask_diffusion')).toThrow(RangeError);
    expect(() => buildContextLengthScalingSeries(16, 'mask_diffusion', -1)).toThrow(RangeError);
  });
});

describe('2. Persistent Nested Generation Timeline & Config Scope View-Models (Wave 2)', () => {
  it('groups all 7 lifecycle milestones, completed_uncommitted, AR, Mock-AR, subsequent blocks, and termination markers', () => {
    let s = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      kvCapacity: 14,
      blockSize: 4,
      stepsPerBlock: 2,
      confidenceThreshold: 0.22,
      mode: 'mask_diffusion',
    });

    // Milestone 1: context_prefill
    const tlPrefill = buildTwoTowerTimelineModel(s);
    expect(tlPrefill.segments).toHaveLength(1);
    expect(tlPrefill.segments[0].kind).toBe('prefill');
    expect(tlPrefill.segments[0].nodes[0].kind).toBe('context_prefill');
    expect(tlPrefill.segments[0].nodes[0].isActiveCursor).toBe(true);
    expect(tlPrefill.segments[0].nodes[0].isHistorical).toBe(false);
    expect(tlPrefill.segments[0].nodes[0].executedTimestep).toBeNull();
    expect(tlPrefill.segments[0].nodes[0].contextPrefixLen).toBe(4);
    expect(tlPrefill.navigation.canBack).toBe(false);
    expect(tlPrefill.navigation.canForward).toBe(false);
    expect(tlPrefill.navigation.canDeleteActiveBranch).toBe(false);
    expect(tlPrefill.navigation.canRestart).toBe(true);

    // Milestone 2: diffusion_block_init
    s = initTwoTowerDiffusionBlock(s);
    const tlInit = buildTwoTowerTimelineModel(s);
    expect(tlInit.segments).toHaveLength(2);
    expect(tlInit.segments[1].kind).toBe('diffusion_block');
    expect(tlInit.segments[1].blockId).toBe(0);
    expect(tlInit.segments[1].nodes).toHaveLength(1);
    expect(tlInit.segments[1].nodes[0].kind).toBe('diffusion_block_init');
    expect(tlInit.segments[1].nodes[0].unresolvedCount).toBe(4);
    expect(tlInit.segments[1].nodes[0].executedTimestep).toBeNull();

    // Milestone 3: diffusion_refinement (carries exact executed t = nMasked / B = 4/4 = 1.0)
    s = runTwoTowerDiffusionRefinement(s);
    const tlRef0 = buildTwoTowerTimelineModel(s);
    const ref0Node = tlRef0.segments[1].nodes[1];
    expect(ref0Node.kind).toBe('diffusion_refinement');
    expect(ref0Node.refinementIndex).toBe(0);
    expect(ref0Node.executedTimestep).toBe(1);
    expect(ref0Node.unresolvedCount).toBe(4);

    // Milestone 4: diffusion_confidence_commit (carries nMaskedBefore -> nMaskedAfter, thresholdCount, floorCount)
    s = commitTwoTowerRefinementConfidence(s);
    const tlCommit0 = buildTwoTowerTimelineModel(s);
    const conf0Node = tlCommit0.segments[1].nodes[2];
    expect(conf0Node.kind).toBe('diffusion_confidence_commit');
    expect(conf0Node.maskTransition).not.toBeNull();
    expect(conf0Node.maskTransition!.nMaskedBefore).toBe(4);
    expect(conf0Node.maskTransition!.nMaskedAfter).toBe(conf0Node.unresolvedCount);
    expect(
      conf0Node.maskTransition!.thresholdCount + conf0Node.maskTransition!.floorCount,
    ).toBe(conf0Node.maskTransition!.committedCount);

    // Complete remaining refinements until Milestone 5: completed_uncommitted
    s = completeActiveDiffusionBlock(s, false);
    const tlUncommitted = buildTwoTowerTimelineModel(s);
    const block0Seg = tlUncommitted.segments[1];
    expect(block0Seg.completionState).toBe('completed_uncommitted');
    expect(block0Seg.isCompletedUncommitted).toBe(true);
    expect(block0Seg.isCommittedToContext).toBe(false);
    const lastBlock0Node = block0Seg.nodes[block0Seg.nodes.length - 1];
    expect(lastBlock0Node.isCompletedUncommitted).toBe(true);
    expect(lastBlock0Node.unresolvedCount).toBe(0);
    expect(lastBlock0Node.contextPrefixLen).toBe(4); // Context NOT advanced yet
    expect(lastBlock0Node.stateBadges.some((b) => b.id === 'completed_uncommitted')).toBe(true);

    // Milestone 6: diffusion_block_commit (Context advances P=4 -> P=8)
    s = commitTwoTowerActiveBlockToContext(s);
    const tlCommitted = buildTwoTowerTimelineModel(s);
    expect(tlCommitted.segments[1].completionState).toBe('committed');
    expect(tlCommitted.segments[1].isCommittedToContext).toBe(true);
    const commitBlockNode =
      tlCommitted.segments[1].nodes[tlCommitted.segments[1].nodes.length - 1];
    expect(commitBlockNode.kind).toBe('diffusion_block_commit');
    expect(commitBlockNode.prefixTransition).toEqual({ before: 4, after: 8, delta: 4 });
    expect(commitBlockNode.contextPrefixLen).toBe(8);
    expect(commitBlockNode.cacheDigestShort).toHaveLength(8);

    // Milestone 7: AR step, Mock-AR step, and subsequent block reaching exact kvCapacity (14)
    s = stepTwoTowerSessionAr(s, { nextTokenId: 5 }); // P = 9
    s = stepTwoTowerSessionMockAr(s, { nextTokenId: 6 }); // P = 10
    s = initTwoTowerDiffusionBlock(s, { blockSize: 4, stepsPerBlock: 1 });
    s = completeActiveDiffusionBlock(s, true); // P = 14 === kvCapacity -> capacity_exhausted

    const tlCap = buildTwoTowerTimelineModel(s);
    expect(tlCap.segments.map((seg) => seg.kind)).toEqual([
      'prefill',
      'diffusion_block',
      'ar_step',
      'mock_ar_step',
      'diffusion_block',
    ]);
    expect(tlCap.segments[4].blockId).toBe(1);
    expect(tlCap.segments[4].terminationMarker).toBe('capacity_exhausted');
    expect(tlCap.isTerminated).toBe(true);
    expect(tlCap.activeLifecyclePhase).toBe('capacity_exhausted');
    expect(
      tlCap.flatNodes[tlCap.flatNodes.length - 1].stateBadges.some(
        (b) => b.id === 'capacity_exhausted',
      ),
    ).toBe(true);

    // Separate branch verifying eos_terminated marker
    let eosSession = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      kvCapacity: 16,
    });
    eosSession = stepTwoTowerSessionAr(eosSession, { nextTokenId: TWOTOWER_MINI_EOS_TOKEN_ID });
    const tlEos = buildTwoTowerTimelineModel(eosSession);
    expect(tlEos.isTerminated).toBe(true);
    expect(tlEos.activeLifecyclePhase).toBe('eos_terminated');
    expect(tlEos.segments[1].terminationMarker).toBe('eos_terminated');
    expect(tlEos.segments[1].nodes[0].stateBadges.some((b) => b.id === 'eos_terminated')).toBe(
      true,
    );
  });

  it('tracks branch roster, rewind historical flags, history trimming (hasPrunedContextTraces), and eager trace pruning', () => {
    let s = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      kvCapacity: 24,
      blockSize: 4,
      stepsPerBlock: 2,
      retainEagerTracesOnHistory: true,
    });
    s = initTwoTowerDiffusionBlock(s);
    s = runTwoTowerDiffusionRefinement(s);
    s = commitTwoTowerRefinementConfidence(s);

    // Rewind 2 steps back
    s = navigateTwoTowerSessionBack(s);
    s = navigateTwoTowerSessionBack(s);
    const tlRewound = buildTwoTowerTimelineModel(s);
    expect(tlRewound.isInspectingHistoricalCursor).toBe(true);
    expect(tlRewound.navigation.canBack).toBe(true);
    expect(tlRewound.navigation.canForward).toBe(true);
    expect(tlRewound.navigation.willAutoForkOnMutatingStep).toBe(true);
    expect(tlRewound.flatNodes[tlRewound.activeCursorIndex].isActiveCursor).toBe(true);
    expect(tlRewound.flatNodes[tlRewound.activeCursorIndex].isHistorical).toBe(true);
    expect(tlRewound.flatNodes[tlRewound.flatNodes.length - 1].isAfterCursor).toBe(true);
    expect(tlRewound.flatNodes[tlRewound.flatNodes.length - 1].isHistorical).toBe(false);

    // Navigate forward once and fork an explicit branch
    s = navigateTwoTowerSessionForward(s);
    s = forkTwoTowerSessionBranch(s, { label: 'Alt Hypothesis' });
    const tlForked = buildTwoTowerTimelineModel(s);
    expect(tlForked.branches).toHaveLength(2);
    const mainEntry = tlForked.branches.find((b) => b.branchId === 'branch-0')!;
    const altEntry = tlForked.branches.find((b) => b.branchId === 'branch-1')!;
    expect(mainEntry.isActive).toBe(false);
    expect(mainEntry.isDeletable).toBe(true);
    expect(altEntry.isActive).toBe(true);
    expect(altEntry.isDeletable).toBe(false);
    expect(tlForked.navigation.canDeleteActiveBranch).toBe(false);
    expect(tlForked.navigation.canDeleteAnyInactiveBranch).toBe(true);

    // Switch back and delete inactive branch
    s = switchTwoTowerSessionBranch(s, 'branch-0');
    s = deleteTwoTowerSessionBranch(s, 'branch-1');
    expect(buildTwoTowerTimelineModel(s).branches).toHaveLength(1);

    // Prune inactive eager traces and verify retainedBytes matches measureTwoTowerSessionRetainedBytes
    s = navigateTwoTowerSession(s, 3);
    const beforePrune = buildTwoTowerTimelineModel(s).retainedHistory;
    expect(beforePrune.retainedBytes).toBe(measureTwoTowerSessionRetainedBytes(s).totalRetainedBytes);
    s = pruneInactiveSessionTraces(s);
    const afterPrune = buildTwoTowerTimelineModel(s).retainedHistory;
    expect(afterPrune.retainedBytes).toBe(measureTwoTowerSessionRetainedBytes(s).totalRetainedBytes);
    expect(afterPrune.retainedBytes).toBeLessThanOrEqual(beforePrune.retainedBytes);
    expect(afterPrune.hasReleasedEagerTraces).toBe(true);
    expect(afterPrune.hasPrunedContextTraces).toBe(false);

    // Hard maxRecords trimming evicts context_prefill so hasPrunedContextTraces === true
    let trimmed = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      kvCapacity: 24,
      blockSize: 4,
      stepsPerBlock: 4,
      maxRecords: 3,
    });
    trimmed = initTwoTowerDiffusionBlock(trimmed); // rec 1
    trimmed = runTwoTowerDiffusionRefinement(trimmed); // rec 2
    trimmed = commitTwoTowerRefinementConfidence(trimmed); // rec 3 -> trims rec 0 (context_prefill)!
    const tlTrimmed = buildTwoTowerTimelineModel(trimmed);
    expect(tlTrimmed.retainedHistory.retainedRecordCount).toBe(3);
    expect(tlTrimmed.retainedHistory.maxRecords).toBe(3);
    expect(tlTrimmed.retainedHistory.prunedRecordCount).toBe(1);
    expect(tlTrimmed.retainedHistory.hasPrunedContextTraces).toBe(true);
    expect(tlTrimmed.flatNodes[0].cursorIndex).toBe(0);
    expect(tlTrimmed.flatNodes[0].recordIndex).toBe(1);
  });

  it('classifies config scope non-retroactively across ready, mid-block, and historical cursors', () => {
    let s = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      kvCapacity: 24,
      blockSize: 4,
      stepsPerBlock: 4,
      confidenceThreshold: 0.22,
      temperature: 0.0,
      mode: 'mask_diffusion',
    });

    // At ready state: mode applies to 'next_step'; blockSize/stepsPerBlock -> 'next_block'; sampler -> 'next_refinement'; prompt/kv -> 'session_restart'
    const scopeReady = buildTwoTowerConfigScopeModel(s);
    expect(scopeReady.hasPendingDiffFromHistorical).toBe(false);
    expect(scopeReady.activeBlockConfig).toBeNull();
    expect(scopeReady.parametersByKey.confidenceThreshold.applicability).toBe('next_refinement');
    expect(scopeReady.parametersByKey.temperature.applicability).toBe('next_refinement');
    expect(scopeReady.parametersByKey.samplingSeed.applicability).toBe('next_refinement');
    expect(scopeReady.parametersByKey.blockSize.applicability).toBe('next_block');
    expect(scopeReady.parametersByKey.stepsPerBlock.applicability).toBe('next_block');
    expect(scopeReady.parametersByKey.mode.applicability).toBe('next_step');
    expect(scopeReady.parametersByKey.initialPromptTokens.applicability).toBe('session_restart');
    expect(scopeReady.parametersByKey.kvCapacity.applicability).toBe('session_restart');

    // Initialize a diffusion block (frozen B=4, K=4) then edit pendingConfig mid-block
    s = initTwoTowerDiffusionBlock(s);
    s = runTwoTowerDiffusionRefinement(s);
    s = updateTwoTowerSessionConfig(s, {
      blockSize: 8,
      stepsPerBlock: 2,
      mode: 'ar',
      confidenceThreshold: 0.8,
      temperature: 0.35,
      samplingSeed: 99_123,
    });

    const scopeMidBlock = buildTwoTowerConfigScopeModel(s);
    expect(scopeMidBlock.activeBlockConfig).toEqual({ blockId: 0, blockSize: 4, stepsPerBlock: 4 });
    expect(scopeMidBlock.historicalConfig.blockSize).toBe(4);
    expect(scopeMidBlock.historicalConfig.stepsPerBlock).toBe(4);
    expect(scopeMidBlock.historicalConfig.confidenceThreshold).toBe(0.22);
    expect(scopeMidBlock.pendingConfig.blockSize).toBe(8);
    expect(scopeMidBlock.pendingConfig.stepsPerBlock).toBe(2);
    expect(scopeMidBlock.pendingConfig.mode).toBe('ar');
    expect(scopeMidBlock.parametersByKey.mode.applicability).toBe('next_block');
    expect(scopeMidBlock.hasPendingDiffFromHistorical).toBe(true);
    expect(scopeMidBlock.hasPendingDiffFromActiveBlock).toBe(true);
    expect(scopeMidBlock.diffKeysFromActiveBlock).toEqual(['blockSize', 'stepsPerBlock', 'mode']);
    expect(scopeMidBlock.badgeText).toContain('Active Block #0 frozen');

    // Rewinding to prefill highlights historical vs pending diff clearly
    const rewound = navigateTwoTowerSession(s, 0);
    const scopeHistorical = buildTwoTowerConfigScopeModel(rewound);
    expect(scopeHistorical.isHistoricalCursor).toBe(true);
    expect(scopeHistorical.hasPendingDiffFromHistorical).toBe(true);
    expect(scopeHistorical.badgeText).toContain('Historical Record #0');
    expect(scopeHistorical.badgeText).toContain('future/forked steps');

    // Draft restart overrides highlight session_restart parameter diffs
    const scopeRestartDraft = buildTwoTowerConfigScopeModel(s, {
      initialPromptTokens: [0, 1],
      kvCapacity: 32,
    });
    expect(scopeRestartDraft.parametersByKey.initialPromptTokens.differsFromHistorical).toBe(true);
    expect(scopeRestartDraft.parametersByKey.kvCapacity.differsFromHistorical).toBe(true);
    expect(scopeRestartDraft.diffKeysFromHistorical).toContain('initialPromptTokens');
    expect(scopeRestartDraft.diffKeysFromHistorical).toContain('kvCapacity');
  });
});

describe('3. Advanced Sampler Inspector View-Model (Wave 2)', () => {
  it('returns null presentation cleanly before any sampler commit and preserves threshold presets + calibration disclaimer', () => {
    let s = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      confidenceThreshold: 0.22,
    });

    const atPrefill = buildAdvancedSamplerInspectorModel(s);
    expect(atPrefill.source).toBe('none');
    expect(atPrefill.hasSamplerCommit).toBe(false);
    expect(atPrefill.samplerPresentation).toBeNull();
    expect(atPrefill.executedTModel).toBeNull();
    expect(atPrefill.commitmentDecomposition).toBeNull();
    expect(atPrefill.slots).toHaveLength(0);
    expect(atPrefill.threshold.value).toBe(0.22);
    expect(atPrefill.threshold.matchedPresetId).toBe('mini_default');
    expect(atPrefill.threshold.matchedPresetLabel).toContain('0.22');
    expect(atPrefill.threshold.presets).toEqual(TWOTOWER_CONFIDENCE_PRESETS);
    expect(atPrefill.calibrationDisclaimer).toBe(TWOTOWER_SAMPLER_CALIBRATION_DISCLAIMER);
    expect(atPrefill.calibrationDisclaimer.toLowerCase()).toContain('not a calibrated');

    // Still null on block init and refinement r=0 (before confidence commit)
    s = initTwoTowerDiffusionBlock(s);
    expect(buildAdvancedSamplerInspectorModel(s).samplerPresentation).toBeNull();
    s = runTwoTowerDiffusionRefinement(s);
    expect(buildAdvancedSamplerInspectorModel(s).samplerPresentation).toBeNull();
  });

  it('decomposes threshold vs budget-floor/final-step commits, maps locked slots without Infinity, and preserves variate digest', () => {
    // High threshold (0.9999) + B=8, K=2 forces minCommitFloor = ceil(8/2) = 4 on step 0,
    // then 4 locked_prior + 4 committed_final_step on step 1!
    let s = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      kvCapacity: 24,
      blockSize: 8,
      stepsPerBlock: 2,
      confidenceThreshold: 0.9999,
      temperature: 0.4,
      samplingSeed: 777,
    });

    s = initTwoTowerDiffusionBlock(s);
    s = runTwoTowerDiffusionRefinement(s);
    s = commitTwoTowerRefinementConfidence(s);

    const step0 = buildAdvancedSamplerInspectorModel(s);
    assertPresentationSafeNoInfinity(step0, 'step0Inspector');
    expect(step0.source).toBe('current_record');
    expect(step0.isCurrentRecord).toBe(true);
    expect(step0.isPriorCommitRecord).toBe(false);
    expect(step0.threshold.matchedPresetId).toBe('custom');
    expect(step0.executedTModel).not.toBeNull();
    expect(step0.executedTModel!.tModel).toBe(1);
    expect(step0.executedTModel!.nMaskedBefore).toBe(8);
    expect(step0.executedTModel!.nMaskedAfter).toBe(4);
    expect(step0.executedTModel!.remainingSteps).toBe(2);
    expect(step0.executedTModel!.minCommitFloor).toBe(4);

    // Exact commitment decomposition conservation identities
    const dec0 = step0.commitmentDecomposition!;
    expect(dec0.priorLockedCount).toBe(0);
    expect(dec0.thresholdQualifiedCount).toBe(0);
    expect(dec0.budgetFloorExtraCount).toBe(4);
    expect(dec0.totalCommittedThisStep).toBe(4);
    expect(dec0.remaskedCount).toBe(4);
    expect(dec0.thresholdQualifiedCount + dec0.budgetFloorExtraCount).toBe(
      dec0.totalCommittedThisStep,
    );
    expect(dec0.totalCommittedThisStep + dec0.remaskedCount).toBe(8);

    // Stochastic variate digest recorded for gumbel_max slots
    expect(step0.variateMode).toBe('derived_deterministic');
    expect(step0.variateDigest).toMatch(/^[0-9a-f]{16}$/);
    expect(step0.samplingSeed).toBe(777);
    expect(step0.slots).toHaveLength(8);
    for (const slot of step0.slots) {
      expect(slot.constrainedMdlmTop.maskTokenProb).toBe(0);
      expect(slot.constrainedMdlmTop.maskTokenExcluded).toBe(true);
      expect(slot.rawLmHeadTop.maskTokenProb).toBeGreaterThanOrEqual(0);
      expect(slot.candidateMode).toBe('gumbel_max');
      expect(slot.variateDigest).toBe(step0.variateDigest);
      expect(slot.unscaledConfidence).not.toBeNull();
      expect(Number.isFinite(slot.unscaledConfidence!)).toBe(true);
      expect(slot.confidenceDisplay).not.toContain('Infinity');
    }
    expect(step0.slots.filter((sl) => sl.status === 'committed_budget_floor')).toHaveLength(4);
    expect(step0.slots.filter((sl) => sl.status === 'remasked')).toHaveLength(4);

    // Advance to refinement r=1 (denoised, before commit r=1): inspector flags prior_commit_record!
    s = runTwoTowerDiffusionRefinement(s);
    const duringRef1 = buildAdvancedSamplerInspectorModel(s);
    expect(duringRef1.source).toBe('prior_commit_record');
    expect(duringRef1.isCurrentRecord).toBe(false);
    expect(duringRef1.isPriorCommitRecord).toBe(true);
    expect(duringRef1.sourceRefinementIndex).toBe(0);
    expect(duringRef1.variateDigest).toBe(step0.variateDigest);

    // Commit refinement r=1 (final step of K=2): 4 locked_prior slots must have ZERO Infinity!
    s = commitTwoTowerRefinementConfidence(s);
    const step1 = buildAdvancedSamplerInspectorModel(s);
    assertPresentationSafeNoInfinity(step1, 'step1Inspector');
    expect(step1.source).toBe('current_record');
    expect(step1.executedTModel!.tModel).toBe(0.5);
    expect(step1.executedTModel!.nMaskedBefore).toBe(4);
    expect(step1.executedTModel!.nMaskedAfter).toBe(0);
    expect(step1.executedTModel!.remainingSteps).toBe(1);
    expect(step1.executedTModel!.minCommitFloor).toBe(4);

    const dec1 = step1.commitmentDecomposition!;
    expect(dec1.priorLockedCount).toBe(4);
    expect(dec1.thresholdQualifiedCount).toBe(0);
    expect(dec1.finalStepExtraCount).toBe(4);
    expect(dec1.budgetFloorExtraCount).toBe(4);
    expect(dec1.totalCommittedThisStep).toBe(4);
    expect(dec1.remaskedCount).toBe(0);

    const lockedSlots = step1.slots.filter((sl) => sl.status === 'locked_prior');
    const finalStepSlots = step1.slots.filter((sl) => sl.status === 'committed_final_step');
    expect(lockedSlots).toHaveLength(4);
    expect(finalStepSlots).toHaveLength(4);

    for (const locked of lockedSlots) {
      expect(locked.candidateMode).toBe('locked_carry_over');
      expect(locked.confidence).toEqual({
        kind: 'locked_prior',
        confidenceDisplay: 'Locked (prior step)',
        numericConfidence: null,
      });
      expect(locked.unscaledConfidence).toBeNull();
      expect(locked.confidenceDisplay).toBe('Locked (prior step)');
      expect(locked.passesStrictThreshold).toBeNull();
      expect(locked.variateDigest).toBeNull();
      expect(locked.inputToken.isMask).toBe(false);
      expect(locked.outputToken.tokenId).toBe(locked.inputToken.tokenId);
    }

    for (const committed of finalStepSlots) {
      expect(committed.inputToken.tokenId).toBe(TWOTOWER_MINI_MASK_TOKEN_ID);
      expect(committed.inputToken.isMask).toBe(true);
      expect(committed.outputToken.isMask).toBe(false);
    }

    // Also test JSON serialization contains zero "Infinity" or "NaN"
    const serialized = JSON.stringify(step1);
    expect(serialized).not.toContain('Infinity');
    expect(serialized).not.toContain('NaN');

    // Commit block to Context: inspector shows block's final commit explicitly tagged as prior_commit_record
    s = commitTwoTowerActiveBlockToContext(s);
    const afterBlockCommit = buildAdvancedSamplerInspectorModel(s);
    expect(afterBlockCommit.source).toBe('prior_commit_record');
    expect(afterBlockCommit.isPriorCommitRecord).toBe(true);
    expect(afterBlockCommit.sourceBlockId).toBe(0);
    expect(afterBlockCommit.sourceRefinementIndex).toBe(1);

    // Initializing Block #1 resets activeBlock.latestSamplerPresentation to null (never leaks Block #0!)
    s = initTwoTowerDiffusionBlock(s, { blockSize: 4, stepsPerBlock: 2 });
    const block1Init = buildAdvancedSamplerInspectorModel(s);
    expect(block1Init.source).toBe('none');
    expect(block1Init.samplerPresentation).toBeNull();
  });

  it('recognizes all three official confidence threshold presets and threshold-qualified commitments', () => {
    for (const [gamma, expectedId] of [
      [0.22, 'mini_default'],
      [0.8, 'paper_default'],
      [0.9, 'reference_code_default'],
    ] as const) {
      let s = createTwoTowerSession(SHARED_MODEL, {
        promptTokenIds: [0, 1, 2, 3],
        confidenceThreshold: gamma,
        blockSize: 4,
        stepsPerBlock: 4,
      });
      s = initTwoTowerDiffusionBlock(s);
      s = runTwoTowerDiffusionRefinement(s);
      s = commitTwoTowerRefinementConfidence(s);
      const model = buildAdvancedSamplerInspectorModel(s);
      expect(model.threshold.value).toBe(gamma);
      expect(model.threshold.matchedPresetId).toBe(expectedId);
    }

    // Threshold 0.0 causes ALL 4 masked slots to qualify strictly via conf > 0
    let greedy = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      confidenceThreshold: 0.0,
      temperature: 0.0,
      blockSize: 4,
      stepsPerBlock: 4,
    });
    greedy = initTwoTowerDiffusionBlock(greedy);
    greedy = runTwoTowerDiffusionRefinement(greedy);
    greedy = commitTwoTowerRefinementConfidence(greedy);
    const greedyModel = buildAdvancedSamplerInspectorModel(greedy);
    expect(greedyModel.variateMode).toBe('greedy_none');
    expect(greedyModel.variateDigest).toBeNull();
    expect(greedyModel.commitmentDecomposition!.thresholdQualifiedCount).toBe(4);
    expect(greedyModel.commitmentDecomposition!.budgetFloorExtraCount).toBe(0);
    expect(greedyModel.slots.every((sl) => sl.status === 'committed_threshold')).toBe(true);
    expect(greedyModel.slots.every((sl) => sl.candidateMode === 'greedy_argmax')).toBe(true);
    expect(greedyModel.slots.every((sl) => sl.passesStrictThreshold === true)).toBe(true);

    // Progress-floor (B=4, K=4, high threshold) & caller_explicit variate provenance
    let prog = createTwoTowerSession(SHARED_MODEL, {
      promptTokenIds: [0, 1, 2, 3],
      confidenceThreshold: 0.9999,
      temperature: 0.5,
      blockSize: 4,
      stepsPerBlock: 4,
    });
    prog = initTwoTowerDiffusionBlock(prog);
    prog = runTwoTowerDiffusionRefinement(prog);
    const explicitVariates = new Float32Array(4 * SHARED_MODEL.vocab.length).fill(0.42);
    prog = commitTwoTowerRefinementConfidence(prog, { uniformVariates: explicitVariates });

    const progModel = buildAdvancedSamplerInspectorModel(prog);
    expect(progModel.variateMode).toBe('caller_explicit');
    expect(progModel.variateDigest).toMatch(/^[0-9a-f]{16}$/);
    expect(progModel.commitmentDecomposition!.thresholdQualifiedCount).toBe(0);
    expect(progModel.commitmentDecomposition!.progressFloorExtraCount).toBe(1);
    expect(progModel.commitmentDecomposition!.budgetFloorExtraCount).toBe(1);
    expect(progModel.commitmentDecomposition!.totalCommittedThisStep).toBe(1);
    expect(progModel.commitmentDecomposition!.remaskedCount).toBe(3);
    const floorSlot = progModel.slots.find((sl) => sl.rawDecision === 'committed_progress_floor')!;
    expect(floorSlot.status).toBe('committed_budget_floor');
    expect(Object.isFrozen(progModel)).toBe(true);
    expect(Object.isFrozen(progModel.slots)).toBe(true);
  });
});

describe('4. Headless Purity & Zero-Math.random Guardrail', () => {
  it('enforces zero DOM, zero WebGL, zero scene/UI/app imports, and zero Math.random in src/twotower/archReadouts.ts', () => {
    const absPath = resolve(process.cwd(), 'src/twotower/archReadouts.ts');
    const src = readFileSync(absPath, 'utf8');
    // Strip block comments and line comments before scanning code tokens
    const codeOnly = src
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\/\/.*$/gm, '');

    expect(codeOnly).not.toMatch(/\b(document|window|HTMLElement|HTMLCanvasElement|WebGL2RenderingContext)\b/);
    expect(codeOnly).not.toMatch(/from\s+['"][^'"]*(scene|render|ui|app)['"/]/);
    expect(codeOnly).not.toMatch(/Math\.random\s*\(/);
  });
});
