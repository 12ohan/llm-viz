import { describe, expect, it } from 'vitest';

import {
  createDefaultHybridModel,
  defaultTokenIdsForLength,
  getOrRunHybridExecution,
  HybridExecutionCache,
} from '../src/math/hybridFlow';
import {
  getActiveDecodeBranch,
  getActiveDecodeStep,
} from '../src/math/incrementalDecode';
import { stepperCellForState } from '../src/math/gqaSim';
import { mambaStepperCellForState } from '../src/math/mambaSim';
import {
  ACCUM_SETTLE_THRESHOLD,
  advanceSpineWavePlayback,
  applyDecodeActionToActiveSession,
  computeSubCellFraction,
  createDualDecodeSessionStore,
  ensureTourArDecodeSession,
  gqaAccumTermBucket,
  gqaStepPlayDurationSec,
  isTourArDecodeContext,
  mambaAccumTermBucket,
  mambaStepPlayDurationSec,
  moeAccumTermBucket,
  moeStepPlayDurationSec,
  resetTourOnlyDecodeSession,
  resolveActiveDecodeSession,
  resolveGqaAccumTermProgress,
  resolveMambaAccumTermProgress,
  resolveMoeAccumTermProgress,
  resolveMoeStepperCellForAccum,
  SPINE_WAVE_FAST_DURATION_MS,
  SPINE_WAVE_INSTRUCTIONAL_DURATION_MS,
  spineWaveDurationMs,
  startSpineWavePlayback,
  toggleSpineWavePlaybackPause,
} from '../src/model/rcPlayback';
import {
  buildGqaCutaway,
  buildMambaCutaway,
  buildMoECutaway,
} from '../src/scene/cutaways';
import {
  TOUR_PREFILL_TOKEN_IDS,
  TOUR_PREFILL_TOKENS,
  WALKTHROUGH_BEATS,
  getWalkthroughBeat,
} from '../src/story/walkthrough';

const BEAT_7_1_IDX = WALKTHROUGH_BEATS.findIndex((b) => b.id === '7.1');
const BEAT_7_2_IDX = WALKTHROUGH_BEATS.findIndex((b) => b.id === '7.2');
const BEAT_7_3_IDX = WALKTHROUGH_BEATS.findIndex((b) => b.id === '7.3');
const BEAT_7_4_IDX = WALKTHROUGH_BEATS.findIndex((b) => b.id === '7.4');

describe('RC2 · Beat 7.1 Output Head -> Beat 7.2 Cached AR Decode bit-exact parity', () => {
  it('matches Beat 7.1 greedy #1 token on TOUR_PREFILL_TOKEN_IDS === Beat 7.2 cached AR t8 token bit-exact', () => {
    const model = createDefaultHybridModel();
    const cache = new HybridExecutionCache(8);

    const beat71 = getWalkthroughBeat(BEAT_7_1_IDX);
    const beat72 = getWalkthroughBeat(BEAT_7_2_IDX);
    expect(beat71.id).toBe('7.1');
    expect(beat71.executionMode).toBe('full_sequence');
    expect(beat71.tokens).toBe(TOUR_PREFILL_TOKENS);
    expect(beat72.id).toBe('7.2');
    expect(beat72.executionMode).toBe('ar_decode');

    // Beat 7.1 evaluates full-sequence Output Head on defaultTokenIdsForLength(8) === TOUR_PREFILL_TOKEN_IDS
    const beat71TokenIds = defaultTokenIdsForLength(beat71.tokens);
    expect(Array.from(beat71TokenIds)).toEqual(Array.from(TOUR_PREFILL_TOKEN_IDS));

    const exec71 = getOrRunHybridExecution(model, cache, { tokenIds: beat71TokenIds });
    const top5AtT7 = exec71.lmHead.topKByToken[TOUR_PREFILL_TOKENS - 1];
    expect(top5AtT7.length).toBeGreaterThanOrEqual(5);
    const greedy71 = top5AtT7[0];

    // Beat 7.2 initializes isolated tourDecodeSession from TOUR_PREFILL_TOKEN_IDS + 1 greedy step (8P + 1S = 9)
    let store = createDualDecodeSessionStore(model, 'bf16');
    store = ensureTourArDecodeSession(store, model, beat72.precision);
    expect(store.tour).not.toBeNull();

    const step72 = getActiveDecodeStep(store.tour!);
    expect(step72.prefixTokens).toBe(TOUR_PREFILL_TOKENS);
    expect(step72.tokens).toBe(1);
    expect(step72.totalTokens).toBe(TOUR_PREFILL_TOKENS + 1);
    expect(step72.newTokenIds[0]).toBe(greedy71.tokenId);
    expect(step72.newTokenLabels[0]).toBe(greedy71.label);

    // Verify bit-exact Bay 43 cached AR GQA trace shape [1 x 9] and 0-ulp parity against full-seq T=9
    const fullSeq9 = getOrRunHybridExecution(model, cache, {
      tokenIds: Array.from(step72.sequenceTokenIds),
    });
    const gqaCached43 = step72.getGqaTrace(43);
    const gqaFull43 = fullSeq9.getGqaTrace(43);
    expect(gqaCached43.tokens).toBe(1);
    expect(gqaCached43.prefixTokens).toBe(8);
    expect(gqaCached43.keyTokens).toBe(9);
    // Final residual output bit-exact parity at t8 (0 ulp)
    const d = gqaCached43.d;
    const cachedRes = step72.residualStreams[step72.residualStreams.length - 1];
    const fullRes = fullSeq9.residualStreams[fullSeq9.residualStreams.length - 1];
    for (let i = 0; i < d; i++) {
      expect(cachedRes[i]).toBe(fullRes[i * 9 + 8]);
    }
    expect(gqaCached43.Y.length).toBe(d);
    for (let i = 0; i < d; i++) {
      expect(gqaCached43.Y[i]).toBe(gqaFull43.Y[i * 9 + 8]);
    }
  });
});

describe('RC2 · Isolated tourDecodeSession vs. freeExploreDecodeSession continuity', () => {
  it('isolates free-explore branching from direct Beat 7.3 jump, backward 7.3 -> 7.1, restart, and resume', () => {
    const model = createDefaultHybridModel();
    let store = createDualDecodeSessionStore(model, 'bf16');

    // 1. User builds an independent Free Explore AR Decode session with a forked branch-1
    let res = applyDecodeActionToActiveSession(store, false, 0, model, 'bf16', {
      type: 'step_greedy',
    });
    store = res.store;
    expect(res.isTour).toBe(false);
    // Navigate back to step 0 and fork branch-1 with an alternative candidate (#2)
    res = applyDecodeActionToActiveSession(store, false, 0, model, 'bf16', {
      type: 'nav_step',
      stepIndex: 0,
    });
    store = res.store;
    const altCand = getActiveDecodeStep(store.freeExplore).predictedNextTokens[1].tokenId;
    res = applyDecodeActionToActiveSession(store, false, 0, model, 'bf16', {
      type: 'step_token',
      tokenId: altCand,
    });
    store = res.store;
    res = applyDecodeActionToActiveSession(store, false, 0, model, 'bf16', {
      type: 'step_greedy',
    });
    store = res.store;

    const savedFreeBranchId = store.freeExplore.activeBranchId;
    const savedFreeStepIdx = store.freeExplore.activeStepIndex;
    const savedFreeDigest = getActiveDecodeStep(store.freeExplore).provenance.nextStateDigest;
    const savedFreeTotalTokens = getActiveDecodeStep(store.freeExplore).totalTokens;
    expect(savedFreeBranchId).toBe('branch-1');
    expect(savedFreeTotalTokens).toBe(6); // 4 prefill + 2 steps

    // 2. Direct jump to Beat 7.3 (via 22-pip rail): idempotently initializes tourDecodeSession at 8P+1S = 9
    expect(isTourArDecodeContext(true, BEAT_7_3_IDX)).toBe(true);
    store = ensureTourArDecodeSession(store, model, 'fp8_selective');
    const resolved73 = resolveActiveDecodeSession(store, true, BEAT_7_3_IDX, model, 'fp8_selective');
    store = resolved73.store;
    expect(resolved73.isTour).toBe(true);
    expect(getActiveDecodeStep(resolved73.session).prefixTokens).toBe(8);
    expect(getActiveDecodeStep(resolved73.session).tokens).toBe(1);
    expect(getActiveDecodeStep(resolved73.session).totalTokens).toBe(9);
    // Calling ensureTourArDecodeSession again is strictly idempotent
    const digest73 = getActiveDecodeStep(store.tour!).provenance.nextStateDigest;
    store = ensureTourArDecodeSession(store, model, 'fp8_selective');
    expect(getActiveDecodeStep(store.tour!).provenance.nextStateDigest).toBe(digest73);

    // 3. User steps / forks inside Act III (Beat 7.4): mutates ONLY tourDecodeSession
    res = applyDecodeActionToActiveSession(store, true, BEAT_7_4_IDX, model, 'fp8_selective', {
      type: 'nav_step',
      stepIndex: 0,
    });
    store = res.store;
    expect(res.isTour).toBe(true);
    const tourAltCand = getActiveDecodeStep(store.tour!).predictedNextTokens[1].tokenId;
    res = applyDecodeActionToActiveSession(store, true, BEAT_7_4_IDX, model, 'fp8_selective', {
      type: 'step_token',
      tokenId: tourAltCand,
    });
    store = res.store;
    expect(store.tour!.activeBranchId).toBe('branch-1');
    expect(getActiveDecodeStep(store.tour!).totalTokens).toBe(9);

    // Free Explore session is 100% untouched!
    expect(store.freeExplore.activeBranchId).toBe(savedFreeBranchId);
    expect(store.freeExplore.activeStepIndex).toBe(savedFreeStepIdx);
    expect(getActiveDecodeStep(store.freeExplore).provenance.nextStateDigest).toBe(savedFreeDigest);

    // 4. Backward navigation (7.3 -> 7.1): Beat 7.1 is Full Seq (not Tour AR context), both sessions stay intact
    expect(isTourArDecodeContext(true, BEAT_7_1_IDX)).toBe(false);
    const resolved71 = resolveActiveDecodeSession(store, true, BEAT_7_1_IDX, model, 'fp8_selective');
    expect(resolved71.isTour).toBe(false);
    expect(store.tour).not.toBeNull();
    expect(store.tour!.activeBranchId).toBe('branch-1');

    // Resume Beat 7.3 restores the tourDecodeSession cleanly
    const resumed73 = resolveActiveDecodeSession(store, true, BEAT_7_3_IDX, model, 'fp8_selective');
    expect(resumed73.isTour).toBe(true);
    expect(resumed73.session.activeBranchId).toBe('branch-1');

    // If user stepped back to prefill step 0 inside tour, ensureTourArDecodeSession navigates to existing step 1 without creating a spurious branch
    res = applyDecodeActionToActiveSession(store, true, BEAT_7_2_IDX, model, 'fp8_selective', {
      type: 'nav_step',
      stepIndex: 0,
    });
    store = res.store;
    const branchCountBeforeEnsure = store.tour!.branches.size;
    expect(getActiveDecodeStep(store.tour!).totalTokens).toBe(8);
    store = ensureTourArDecodeSession(store, model, 'fp8_selective');
    expect(getActiveDecodeStep(store.tour!).totalTokens).toBe(9);
    expect(store.tour!.branches.size).toBe(branchCountBeforeEnsure);

    // 5. Tour Restart resets ONLY tourDecodeSession to null, preserving freeExploreDecodeSession completely
    store = resetTourOnlyDecodeSession(store);
    expect(store.tour).toBeNull();
    expect(store.freeExplore.activeBranchId).toBe(savedFreeBranchId);
    expect(store.freeExplore.activeStepIndex).toBe(savedFreeStepIdx);
    expect(getActiveDecodeStep(store.freeExplore).provenance.nextStateDigest).toBe(savedFreeDigest);

    // Next pass into Beat 7.2 re-seeds cleanly at canonical branch-0 8P + 1S = 9
    store = ensureTourArDecodeSession(store, model, 'fp8_selective');
    expect(store.tour!.activeBranchId).toBe('branch-0');
    expect(getActiveDecodeBranch(store.tour!).steps.length).toBe(2);
    expect(getActiveDecodeStep(store.tour!).totalTokens).toBe(9);

    // Exiting tour immediately returns the untouched freeExploreDecodeSession
    const exited = resolveActiveDecodeSession(store, false, BEAT_7_2_IDX, model, 'fp8_selective');
    expect(exited.isTour).toBe(false);
    expect(exited.session.activeBranchId).toBe(savedFreeBranchId);
    expect(getActiveDecodeStep(exited.session).provenance.nextStateDigest).toBe(savedFreeDigest);
  });
});

describe('RC1 · Sub-cell termProgress mapping & cutaway end-to-end integration', () => {
  it('cleanly separates intermediate partial terms (< 0.999) from completed/unrestricted states (1.0)', () => {
    expect(computeSubCellFraction(0.20, 8)).toBeCloseTo(0.60, 5);
    expect(computeSubCellFraction(0.35, 9)).toBeCloseTo(0.15, 5);
    expect(computeSubCellFraction(0.0, 4)).toBe(0);
    expect(computeSubCellFraction(ACCUM_SETTLE_THRESHOLD, 8)).toBe(1);
    expect(computeSubCellFraction(1.0, 8)).toBe(1);

    const model = createDefaultHybridModel();
    const cache = new HybridExecutionCache(4);
    const exec = getOrRunHybridExecution(model, cache, {
      tokenIds: defaultTokenIdsForLength(8),
    });
    const gqaSim = exec.getGqaTrace(6);
    const mambaSim = exec.getMambaTrace(1);
    const moeSim = exec.getMoeTrace(2);

    // --- GQA Step 3 (QK^T/sqrt(d)) & Step 5 (Weighted V) ---
    const gqaMid2 = resolveGqaAccumTermProgress(gqaSim, 2, 0.20, true, false, false);
    expect(gqaMid2).toBeLessThan(ACCUM_SETTLE_THRESHOLD);
    expect(gqaAccumTermBucket(gqaSim, 2, gqaMid2)).toBe(2); // Term 3/4
    expect(resolveGqaAccumTermProgress(gqaSim, 2, 1.0, true, false, false)).toBe(1);
    expect(resolveGqaAccumTermProgress(gqaSim, 2, 0.20, false, false, false)).toBe(1); // idle/ordinary
    expect(resolveGqaAccumTermProgress(gqaSim, 2, 0.20, true, true, false)).toBe(1); // reducedMotion
    expect(resolveGqaAccumTermProgress(gqaSim, 2, 0.20, true, false, true)).toBe(1); // hover/pin override
    expect(resolveGqaAccumTermProgress(gqaSim, 0, 0.20, true, false, false)).toBe(1); // non-accumulating step
    expect(gqaStepPlayDurationSec(gqaSim, 2)).toBeGreaterThan(10); // ~400ms * 32 terms

    const gqaCell2 = stepperCellForState(gqaSim, 0, 2, 0.20);
    const gqaSceneMid = buildGqaCutaway({
      bayId: 6,
      tokens: gqaSim.tokens,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeCell: gqaCell2,
      stepperStep: 2,
      stepperActive: true,
      stepperPulsePhase: 0.20,
      termProgress: gqaMid2,
      reducedMotion: false,
      simOverride: gqaSim,
    });
    expect(gqaSceneMid.focusedAccumulation?.active).toBe(true);
    expect(gqaSceneMid.focusedAccumulation?.currentTermIndex).toBe(2);
    expect(gqaSceneMid.cellPopover?.isAccumulating).toBe(true);

    const gqaMid4 = resolveGqaAccumTermProgress(gqaSim, 4, 0.35, true, false, false);
    expect(gqaMid4).toBeLessThan(ACCUM_SETTLE_THRESHOLD);
    expect(gqaAccumTermBucket(gqaSim, 4, gqaMid4)).toBe(3); // Term 4/8

    // --- Mamba-2 Step 3 (Conv1D), Step 4 (SSM Update), Step 5 (Readout + D) ---
    const mambaConvMid = resolveMambaAccumTermProgress(mambaSim, 2, 0.35, true, false, false);
    expect(mambaConvMid).toBeLessThan(ACCUM_SETTLE_THRESHOLD);
    expect(mambaAccumTermBucket(mambaSim, 2, mambaConvMid)).toBe(0); // Term 1/4
    const mambaSsmMid = resolveMambaAccumTermProgress(mambaSim, 3, 0.20, true, false, false);
    expect(mambaSsmMid).toBeLessThan(ACCUM_SETTLE_THRESHOLD);
    expect(mambaAccumTermBucket(mambaSim, 3, mambaSsmMid)).toBe(0); // Term 1/2 (dA * H_{t-1})
    expect(resolveMambaAccumTermProgress(mambaSim, 3, 1.0, true, false, false)).toBe(1);
    expect(resolveMambaAccumTermProgress(mambaSim, 3, 0.20, true, true, false)).toBe(1);
    expect(mambaStepPlayDurationSec(mambaSim, 3)).toBeGreaterThan(10);

    const mambaCell3 = mambaStepperCellForState(mambaSim, 0, 3, 0.20);
    const mambaSceneMid = buildMambaCutaway({
      bayId: 1,
      tokens: mambaSim.tokens,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      activeCell: mambaCell3,
      stepperStep: 3,
      stepperActive: true,
      stepperSubProgress: 0.20,
      termProgress: mambaSsmMid,
      reducedMotion: false,
      simOverride: mambaSim,
    });
    expect(mambaSceneMid.focusedAccumulation?.active).toBe(true);
    expect(mambaSceneMid.focusedAccumulation?.totalTerms).toBe(2);
    expect(mambaSceneMid.cellPopover?.isAccumulating).toBe(true);

    // --- MoE Step 3 (Top-6), Step 4 (Routed Sum), Step 6 (MoE Out) ---
    const moeTopkMid = resolveMoeAccumTermProgress(moeSim, 2, 0.20, true, false, false);
    expect(moeTopkMid).toBeLessThan(ACCUM_SETTLE_THRESHOLD);
    expect(moeAccumTermBucket(moeSim, 2, moeTopkMid)).toBe(1); // Term 2/6
    expect(resolveMoeAccumTermProgress(moeSim, 2, 1.0, true, false, false)).toBe(1);
    expect(moeStepPlayDurationSec(moeSim, 2)).toBeGreaterThan(10);

    const moeRoutedCell = resolveMoeStepperCellForAccum(moeSim, 3, 0.35, 7, moeSim.topk_indices[7 * 6]);
    expect(moeRoutedCell.tensor).toBe('routed_sum');
    const moeRoutedMid = resolveMoeAccumTermProgress(moeSim, 3, 0.35, true, false, false);
    expect(moeRoutedMid).toBeLessThan(ACCUM_SETTLE_THRESHOLD);

    const moeSceneMid = buildMoECutaway({
      bayId: 2,
      tokens: moeSim.tokens,
      precision: 'bf16',
      selectedPart: null,
      hoveredPart: null,
      focusToken: 7,
      activeCell: moeRoutedCell,
      stepperStep: 3,
      stepperActive: true,
      stepperSubProgress: 0.35,
      termProgress: moeRoutedMid,
      reducedMotion: false,
      simOverride: moeSim,
    });
    expect(moeSceneMid.focusedAccumulation?.active).toBe(true);
    expect(moeSceneMid.focusedAccumulation?.totalTerms).toBe(6);
    expect(moeSceneMid.cellPopover?.isAccumulating).toBe(true);
  });
});

describe('RC3 · Fast (0.9s) vs Instructional (5.5s) spine wave replay clock', () => {
  it('paces fast/instructional modes, supports pause/resume, and settles static summary under reducedMotion', () => {
    expect(spineWaveDurationMs('fast')).toBe(SPINE_WAVE_FAST_DURATION_MS);
    expect(spineWaveDurationMs('instructional')).toBe(SPINE_WAVE_INSTRUCTIONAL_DURATION_MS);

    // Instructional mode advances smoothly over 5500 ms and freezes cleanly while paused
    let wave = startSpineWavePlayback('instructional', false);
    expect(wave.progress).toBe(0);
    expect(wave.mode).toBe('instructional');
    expect(wave.paused).toBe(false);

    // Advance 25 ticks of 100 ms = 2500 ms -> progress = 2500 / 5500
    for (let i = 0; i < 25; i++) {
      const step = advanceSpineWavePlayback(wave, 100, false);
      expect(step.finished).toBe(false);
      expect(step.animating).toBe(true);
      wave = step.next;
    }
    expect(wave.progress).toBeCloseTo(2500 / 5500, 5);

    // Pause freezes progress across ticks
    wave = toggleSpineWavePlaybackPause(wave, false);
    expect(wave.paused).toBe(true);
    const frozen = advanceSpineWavePlayback(wave, 100, false);
    expect(frozen.animating).toBe(false);
    expect(frozen.next.progress).toBeCloseTo(2500 / 5500, 5);

    // Resume and advance remaining 3000 ms finishes cleanly
    wave = toggleSpineWavePlaybackPause(wave, false);
    expect(wave.paused).toBe(false);
    let finished = false;
    for (let i = 0; i < 31; i++) {
      const step = advanceSpineWavePlayback(wave, 100, false);
      wave = step.next;
      if (step.finished) {
        finished = true;
        break;
      }
    }
    expect(finished).toBe(true);
    expect(wave.progress).toBeNull();

    // Reduced motion settles immediately at progress = 1.0 static summary (no animation)
    const reduced = startSpineWavePlayback('instructional', true);
    expect(reduced.progress).toBe(1.0);
    expect(reduced.paused).toBe(false);
    const reducedTick = advanceSpineWavePlayback(reduced, 100, true);
    expect(reducedTick.animating).toBe(false);
    expect(reducedTick.finished).toBe(false);
    expect(reducedTick.next.progress).toBe(1.0);
  });
});
