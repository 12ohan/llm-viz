import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { formatExpertId } from '../src/math/moeSim';
import {
  createDefaultTwoTowerModel,
  createTwoTowerModelParameters,
  createTwoTowerSession,
  forkTwoTowerSessionBranch,
  getActiveBranch,
  getActiveRecord,
  initTwoTowerDiffusionBlock,
  runTwoTowerDiffusionRefinement,
  stepTwoTowerSessionAr,
  type TwoTowerSession,
} from '../src/twotower/index';
import { assertPresentationSafeNoInfinity } from '../src/twotower/sampler';
import {
  buildTwoTowerEmbeddingScene,
  buildTwoTowerMacroScene,
  buildTwoTowerMoeScene,
  buildTwoTowerOutputScene,
  buildTwoTowerPairedGqaScene,
  buildTwoTowerPairedMambaScene,
  buildTwoTowerTimestepScene,
  computeMoeSelectionMargin,
  extractMambaFullBlockPositionState,
  getTwoTowerBayKind,
  type TwoTowerSpatialScene,
} from '../src/twotower/scene/index';
import {
  createTwoTowerTourController,
  TWOTOWER_TOUR_ACT_COUNT,
  TWOTOWER_TOUR_ACTS,
  TWOTOWER_TOUR_BEAT_COUNT,
  TWOTOWER_TOUR_CONFIG,
  TWOTOWER_TOUR_SESSION_ID,
  verifyTourIsolation,
  type TwoTowerTourSnapshot,
} from '../src/twotower/story/walkthrough';

const VALID_VIEW_MODES = new Set([
  'macro',
  'paired_gqa',
  'paired_mamba',
  'timestep',
  'moe',
  'embedding',
  'output',
]);
const VALID_TOWER_FOCUS = new Set(['paired', 'context_solo', 'denoiser_solo']);
const VALID_LINKER_CHANNELS = new Set(['all', 'gqa_kv', 'mamba_conv', 'mamba_ssm', 'timestep']);
const VALID_METRIC_TONES = new Set(['context', 'denoiser', 'shared', 'warn', 'muted', 'gold']);

function buildSceneForTourBeat(
  session: TwoTowerSession,
  snap: TwoTowerTourSnapshot,
): TwoTowerSpatialScene {
  switch (snap.viewMode) {
    case 'macro':
      return buildTwoTowerMacroScene(session, {
        selectedTower:
          snap.towerFocus === 'context_solo'
            ? 'context'
            : snap.towerFocus === 'denoiser_solo'
              ? 'denoiser'
              : 'shared',
        selectedBayId: snap.bayId,
        activeLinkerChannel: snap.linkerChannel,
      });
    case 'paired_gqa':
      return buildTwoTowerPairedGqaScene(session, {
        bayId: snap.bayId,
        towerFocus: snap.towerFocus,
        stepperStep: snap.opStep,
      });
    case 'paired_mamba':
      return buildTwoTowerPairedMambaScene(session, {
        bayId: snap.bayId,
        towerFocus: snap.towerFocus,
        activeHead: 0,
        localTokenIndex: snap.blockPos,
        stepperStep: snap.opStep,
      });
    case 'timestep':
      return buildTwoTowerTimestepScene(session, {
        bayId: snap.bayId,
        localTokenIndex: snap.blockPos,
        selectedPartId: snap.highlightPartId,
        stepperStep: snap.opStep,
      });
    case 'moe':
      return buildTwoTowerMoeScene(session, {
        bayId: snap.bayId,
        towerRole: snap.towerFocus === 'context_solo' ? 'context' : 'denoiser',
        focusToken: snap.blockPos,
        selectedPart: snap.highlightPartId,
        stepperStep: snap.opStep,
      });
    case 'embedding':
      return buildTwoTowerEmbeddingScene(session, {
        towerRole: snap.towerFocus === 'context_solo' ? 'context' : 'denoiser',
        focusToken: snap.blockPos,
        selectedPart: snap.highlightPartId,
      });
    case 'output':
      return buildTwoTowerOutputScene(session, {
        towerRole: snap.towerFocus === 'context_solo' ? 'context' : 'denoiser',
        focusToken: snap.blockPos,
        selectedPart: snap.highlightPartId,
      });
  }
}

function assertAllFiniteFloat32(arr: Float32Array | undefined, label: string): void {
  if (!arr) return;
  for (let i = 0; i < arr.length; i++) {
    if (!Number.isFinite(arr[i])) {
      throw new Error(`Non-finite Float32 at ${label}[${i}]: ${arr[i]}`);
    }
  }
}

describe('Stage TT4 Wave 3: Headless 4-Act Guided TwoTower Walkthrough Engine', () => {
  const model = createDefaultTwoTowerModel();

  it('defines 4 contiguous acts totaling 28 beats (6 + 10 + 7 + 5) covering all 7 views', () => {
    expect(TWOTOWER_TOUR_ACT_COUNT).toBe(4);
    expect(TWOTOWER_TOUR_ACTS).toHaveLength(4);
    expect(TWOTOWER_TOUR_BEAT_COUNT).toBe(28);
    expect(TWOTOWER_TOUR_BEAT_COUNT).toBeGreaterThanOrEqual(26);
    expect(TWOTOWER_TOUR_BEAT_COUNT).toBeLessThanOrEqual(30);

    expect(TWOTOWER_TOUR_ACTS.map((a) => a.id)).toEqual(['act-1', 'act-2', 'act-3', 'act-4']);
    expect(TWOTOWER_TOUR_ACTS.map((a) => a.beatCount)).toEqual([6, 10, 7, 5]);
    expect(TWOTOWER_TOUR_ACTS.map((a) => a.startBeatIndex)).toEqual([0, 6, 16, 23]);

    const totalFromActs = TWOTOWER_TOUR_ACTS.reduce((acc, a) => acc + a.beatCount, 0);
    expect(totalFromActs).toBe(TWOTOWER_TOUR_BEAT_COUNT);

    const ctrl = createTwoTowerTourController(model);
    const seenViewModes = new Set<string>();
    const seenBeatIds = new Set<string>();

    for (let i = 0; i < TWOTOWER_TOUR_BEAT_COUNT; i++) {
      const snap = ctrl.goToBeat(i);
      expect(snap.beatIndex).toBe(i);
      expect(seenBeatIds.has(snap.beatId)).toBe(false);
      seenBeatIds.add(snap.beatId);
      seenViewModes.add(snap.viewMode);

      expect(VALID_VIEW_MODES.has(snap.viewMode)).toBe(true);
      expect(VALID_TOWER_FOCUS.has(snap.towerFocus)).toBe(true);
      expect(VALID_LINKER_CHANNELS.has(snap.linkerChannel)).toBe(true);
      expect(Number.isInteger(snap.blockPos)).toBe(true);
      expect(snap.blockPos).toBeGreaterThanOrEqual(0);
      expect(snap.blockPos).toBeLessThanOrEqual(3);
      expect(typeof snap.showSamplerPanel).toBe('boolean');
      expect(typeof snap.showArchPanel).toBe('boolean');
      expect(snap.paragraphs.length).toBeGreaterThanOrEqual(2);
      expect(snap.takeaways.length).toBeGreaterThanOrEqual(2);
      expect(snap.liveMetrics.length).toBeGreaterThanOrEqual(4);
      expect(snap.epistemicTag.length).toBeGreaterThan(20);
      expect(snap.epistemicTag.toLowerCase()).toMatch(/miniature|synthetic/);
      expect(snap.epistemicTag.toLowerCase()).toMatch(/canonical|released|production|mdlm|mamba|d=8/);

      // Validate bayId for each viewMode
      if (snap.viewMode === 'paired_gqa') {
        expect(getTwoTowerBayKind(snap.bayId)).toBe('*');
      } else if (snap.viewMode === 'paired_mamba') {
        expect(getTwoTowerBayKind(snap.bayId)).toBe('M');
      } else if (snap.viewMode === 'moe') {
        expect(getTwoTowerBayKind(snap.bayId)).toBe('E');
      } else if (snap.viewMode === 'timestep') {
        expect(snap.bayId).toBeGreaterThanOrEqual(1);
        expect(snap.bayId).toBeLessThanOrEqual(52);
      } else if (snap.viewMode === 'embedding') {
        expect(snap.bayId).toBe(0);
      } else if (snap.viewMode === 'output') {
        expect(snap.bayId).toBe(53);
      } else if (snap.viewMode === 'macro') {
        expect(snap.bayId === -1 || snap.bayId === 0 || snap.bayId === 53 || (snap.bayId >= 1 && snap.bayId <= 52)).toBe(true);
      }

      // Validate opStep range
      if (snap.viewMode === 'macro') {
        expect(snap.opStep).toBeNull();
      } else if (snap.opStep !== null) {
        const maxOp = snap.viewMode === 'embedding' ? 3 : 5;
        expect(snap.opStep).toBeGreaterThanOrEqual(0);
        expect(snap.opStep).toBeLessThanOrEqual(maxOp);
      }

      // Every live metric has valid label, value, tone, and zero Infinity/NaN text
      for (const m of snap.liveMetrics) {
        expect(m.label.length).toBeGreaterThan(0);
        expect(m.value.length).toBeGreaterThan(0);
        expect(VALID_METRIC_TONES.has(m.tone)).toBe(true);
        expect(m.value).not.toMatch(/Infinity|NaN/i);
      }
    }

    // All 7 spatial views are showcased across the 28-beat walkthrough
    expect(seenViewModes.size).toBe(7);
  });

  it('guarantees zero Infinity/NaN, exact provenance alignment, and real t = nMasked / B on every beat', () => {
    const ctrl = createTwoTowerTourController(model);
    ctrl.start();

    for (let i = 0; i < TWOTOWER_TOUR_BEAT_COUNT; i++) {
      const snap = ctrl.goToBeat(i);
      // Strict presentation-safe no-Infinity / no-NaN assertion across entire snapshot object graph
      expect(() => assertPresentationSafeNoInfinity(snap, `beat[${i}]`)).not.toThrow();

      // Check rendered narrative text never leaks Infinity or NaN
      const allText = [
        snap.title,
        snap.subtitle,
        snap.badge,
        ...snap.paragraphs,
        ...snap.takeaways,
        ...snap.liveMetrics.map((m) => `${m.label}: ${m.value}`),
        snap.interactivePrompt,
        snap.epistemicTag,
      ].join('\n');
      expect(allText).not.toContain('Infinity');
      expect(allText).not.toContain('∞');
      expect(allText).not.toContain('NaN');

      // Tour session cursor is strictly synchronized to beat provenance
      const tourSession = ctrl.getSession();
      const activeBranch = getActiveBranch(tourSession);
      const activeRecord = getActiveRecord(tourSession);
      expect(tourSession.sessionId).toBe(TWOTOWER_TOUR_SESSION_ID);
      expect(activeBranch.branchId).toBe(snap.provenance.branchId);
      expect(activeBranch.cursorIndex).toBe(snap.provenance.recordIndex);
      expect(activeRecord.identity.recordId).toBe(snap.provenance.recordId);
      expect(activeRecord.identity.blockId).toBe(snap.provenance.blockId);
      expect(activeRecord.identity.refinementIndex).toBe(snap.provenance.refinementIndex);
      expect(activeRecord.identity.absoluteTokenRange).toEqual(snap.provenance.absoluteTokenRange);
      expect(activeRecord.identity.sourceCacheDigest).toBe(snap.provenance.sourceCacheDigest);
      expect(activeRecord.configAtStep.mode).toBe(snap.provenance.mode);

      // Exact t = nMasked / B consistency
      if (activeRecord.kind === 'diffusion_refinement') {
        const expectedT = activeRecord.activeBlock.nMasked / activeRecord.blockConfig.blockSize;
        expect(snap.provenance.executedTModel).toBeCloseTo(expectedT, 7);
        expect(snap.provenance.executedTModel).toBe(activeRecord.tModel);
        expect(snap.provenance.currentUnresolvedCount).toBe(activeRecord.activeBlock.nMasked);
      } else if (activeRecord.kind === 'diffusion_confidence_commit') {
        const expectedT =
          activeRecord.unmaskStep.nMaskedBefore / activeRecord.blockConfig.blockSize;
        expect(snap.provenance.executedTModel).toBeCloseTo(expectedT, 7);
        expect(snap.provenance.executedTModel).toBe(activeRecord.unmaskStep.tModel);
        expect(snap.provenance.currentUnresolvedCount).toBe(activeRecord.unmaskStep.nMaskedAfter);
      } else {
        expect(snap.provenance.executedTModel).toBeNull();
        expect(snap.provenance.currentUnresolvedCount).toBe(
          activeRecord.activeBlock ? activeRecord.activeBlock.nMasked : 0,
        );
      }
    }
  });

  it('computes real live numerical callouts directly from underlying tensors, Mamba decomposition, MoE margin, and Sampler presentation', () => {
    const ctrl = createTwoTowerTourController(model);

    // Beat 0 (1.1): Independent Context & Denoiser fingerprints
    const b0 = ctrl.goToBeat(0);
    expect(b0.liveMetrics.some((m) => m.value.includes(model.contextModel.parameterFingerprint.slice(0, 8)))).toBe(true);
    expect(b0.liveMetrics.some((m) => m.value.includes(model.denoiserModel.parameterFingerprint.slice(0, 8)))).toBe(true);

    // Beat 3 (1.4 AR) & Beat 4 (1.5 Mock-AR): Isolated side branches, both from P=8 prefill, unmodulated t=null
    const bAr = ctrl.goToBeat(3);
    const bMockAr = ctrl.goToBeat(4);
    expect(bAr.provenance.mode).toBe('ar');
    expect(bAr.provenance.executedTModel).toBeNull();
    expect(bAr.provenance.branchId).not.toBe(b0.provenance.branchId);
    expect(bMockAr.provenance.mode).toBe('mock_ar');
    expect(bMockAr.provenance.executedTModel).toBeNull();
    expect(bMockAr.provenance.branchId).not.toBe(b0.provenance.branchId);
    expect(bMockAr.provenance.branchId).not.toBe(bAr.provenance.branchId);
    // Both AR and Mock-AR branch from the exact same P=8 Context prefill cache digest
    expect(bAr.provenance.sourceCacheDigest).toBe(bMockAr.provenance.sourceCacheDigest);

    // Beat 8 (2.3): Split adaLN ordering (M/* modulate_then_norm vs E norm_then_modulate, unmodulated norm_f)
    const bSplit = ctrl.goToBeat(8);
    expect(bSplit.viewMode).toBe('timestep');
    expect(bSplit.bayId).toBe(2);
    expect(bSplit.liveMetrics.some((m) => m.value === 'modulate_then_norm')).toBe(true);
    expect(bSplit.liveMetrics.some((m) => m.value === 'norm_then_modulate')).toBe(true);

    // Beat 11 (2.6): Paired Mamba Conv1D boundary replacement (3->2->1->0) matches live decomposition
    const bConv = ctrl.goToBeat(11);
    const recConv = getActiveRecord(ctrl.getSession());
    const m1 = recConv.denoiserPass!.getMambaTrace(1);
    const expectedInhSeq = [0, 1, 2, 3]
      .map((p) => extractMambaFullBlockPositionState(m1, 0, p, { bayId: 1 }).convWindow.inheritedHistoryColumnCount)
      .join('→');
    expect(expectedInhSeq).toBe('3→2→1→0');
    expect(bConv.liveMetrics.some((m) => m.value === expectedInhSeq)).toBe(true);

    // Beat 12 (2.7): Bay 08 full-block inspection after bidirectional GQA (Bay 06)
    const bPostGqa = ctrl.goToBeat(12);
    expect(bPostGqa.bayId).toBe(8);
    expect(bPostGqa.blockPos).toBe(2);
    const m8Decomp = extractMambaFullBlockPositionState(
      getActiveRecord(ctrl.getSession()).denoiserPass!.getMambaTrace(8),
      0,
      2,
      { bayId: 8 },
    );
    expect(m8Decomp.isAfterFirstBidirectionalGqa).toBe(true);
    expect(bPostGqa.liveMetrics.some((m) => m.value.includes('afterBay06=true'))).toBe(true);

    // Beat 13 (2.8): Bay 42 p=3 analytical decay vs block write decomposition + epistemic norm disclaimer
    const bDecay = ctrl.goToBeat(13);
    expect(bDecay.bayId).toBe(42);
    expect(bDecay.blockPos).toBe(3);
    const m42Decomp = extractMambaFullBlockPositionState(
      getActiveRecord(ctrl.getSession()).denoiserPass!.getMambaTrace(42),
      0,
      3,
      { bayId: 42 },
    );
    expect(bDecay.liveMetrics.some((m) => m.value === m42Decomp.cumulativeDecay.toFixed(4))).toBe(true);
    expect(bDecay.liveMetrics.some((m) => m.value === m42Decomp.inheritedRetentionL2Norm.toFixed(4))).toBe(true);
    expect(bDecay.liveMetrics.some((m) => m.value === m42Decomp.accumulatedBlockWriteL2Norm.toFixed(4))).toBe(true);
    expect(bDecay.paragraphs.join(' ')).toMatch(/NOT measure retained semantic information/i);

    // Beat 15 (2.10): Stateless MoE FP64 6th-vs-7th cutoff margin matches live computeMoeSelectionMargin
    const bMoe = ctrl.goToBeat(15);
    expect(bMoe.viewMode).toBe('moe');
    expect(bMoe.bayId).toBe(2);
    const moeTrace = getActiveRecord(ctrl.getSession()).denoiserPass!.getMoeTrace(2);
    const margin = computeMoeSelectionMargin(moeTrace, bMoe.blockPos);
    expect(bMoe.liveMetrics.some((m) => m.value.includes(formatExpertId(margin.rank6ExpertId)))).toBe(true);
    expect(bMoe.liveMetrics.some((m) => m.value.includes(formatExpertId(margin.rank7ExpertId)))).toBe(true);
    expect(bMoe.liveMetrics.some((m) => m.value.includes(margin.choiceScoreMargin.toFixed(6)))).toBe(true);

    // Beat 17 (3.2) & Beat 18 (3.3): MDLM maskTokenProb === 0 & γ = 0.22 calibration boundary
    const bMdlm = ctrl.goToBeat(17);
    expect(bMdlm.showSamplerPanel).toBe(true);
    expect(bMdlm.liveMetrics.some((m) => m.value.includes('0.00%'))).toBe(true);

    const bThresh = ctrl.goToBeat(18);
    expect(bThresh.paragraphs.join(' ')).toMatch(/NOT calibrated semantic correctness/i);
    expect(bThresh.liveMetrics.some((m) => m.value.includes('0.22'))).toBe(true);

    // Beat 20 (3.5): Successive refinement r = 1 proves live t = nMasked / B = 0.500 !== r / K (0.250)
    const bRef1 = ctrl.goToBeat(20);
    expect(bRef1.provenance.refinementIndex).toBe(1);
    expect(bRef1.provenance.executedTModel).toBe(0.5);
    expect(bRef1.provenance.currentUnresolvedCount).toBe(2);
    expect(bRef1.provenance.executedTModel).not.toBe(1 / TWOTOWER_TOUR_CONFIG.stepsPerBlock);

    // Beat 21 (3.6): Locked prior positions display "Locked (prior step)" with zero Infinity
    const bConf1 = ctrl.goToBeat(21);
    const recConf1 = getActiveRecord(ctrl.getSession());
    expect(recConf1.kind).toBe('diffusion_confidence_commit');
    expect(bConf1.liveMetrics.some((m) => m.value.includes('Locked (prior step)'))).toBe(true);
    expect(bConf1.provenance.executedTModel).toBe(0.5);
    expect(bConf1.provenance.currentUnresolvedCount).toBe(1);

    // Beat 22 (3.7): Completed-but-uncommitted state (nMasked = 0, contextCommitted = false, P = 8)
    const bCompletedUncommitted = ctrl.goToBeat(22);
    const recUncommitted = getActiveRecord(ctrl.getSession());
    expect(recUncommitted.lifecyclePhase).toBe('completed_uncommitted');
    expect(recUncommitted.activeBlock?.nMasked).toBe(0);
    expect(recUncommitted.activeBlock?.contextCommitted).toBe(false);
    expect(recUncommitted.committedPrefixLen).toBe(8);
    expect(bCompletedUncommitted.provenance.currentUnresolvedCount).toBe(0);

    // Beat 23 (4.1): Explicit Context block commitment (P = 8 -> 12, 2 split segments)
    const bCommit = ctrl.goToBeat(23);
    const recCommit = getActiveRecord(ctrl.getSession());
    expect(recCommit.kind).toBe('diffusion_block_commit');
    expect(recCommit.committedPrefixLen).toBe(12);
    expect(recCommit.contextExecution!.segments).toHaveLength(2);
    expect(recCommit.contextExecution!.segments[0].segmentRole).toBe('split_prefix_for_mamba_s2');
    expect(recCommit.contextExecution!.segments[1].segmentRole).toBe('split_final_token');
    expect(bCommit.provenance.mode).toBe('mask_diffusion');

    // Beat 24 (4.2): Next block (Blk 1) conditioned on NEW P = 12 Context boundary (never conflating P = 8!)
    const bBlk1 = ctrl.goToBeat(24);
    const recBlk1 = getActiveRecord(ctrl.getSession());
    expect(bBlk1.provenance.blockId).toBe(1);
    expect(bBlk1.provenance.absoluteTokenRange).toEqual([12, 15]);
    expect(recBlk1.committedPrefixLen).toBe(12);
    expect(bBlk1.provenance.sourceCacheDigest).toBe(recCommit.identity.resultCacheDigest);
    expect(bBlk1.provenance.sourceCacheDigest).not.toBe(bRef1.provenance.sourceCacheDigest);
  });

  it('integrates cleanly with all 7 Stage TT3 spatial scene builders at every single beat (0..27)', () => {
    const ctrl = createTwoTowerTourController(model);
    ctrl.start();

    for (let i = 0; i < TWOTOWER_TOUR_BEAT_COUNT; i++) {
      const snap = ctrl.goToBeat(i);
      const scene = buildSceneForTourBeat(ctrl.getSession(), snap);

      expect(scene.viewMode).toBe(snap.viewMode);
      expect(scene.bayId).toBe(snap.bayId);
      expect(scene.provenanceBadges.length).toBeGreaterThan(0);
      expect(scene.fitBounds.min).toHaveLength(3);
      expect(scene.fitBounds.max).toHaveLength(3);
      assertAllFiniteFloat32(scene.solidData, `beat[${i}].solidData`);
      assertAllFiniteFloat32(scene.lineData, `beat[${i}].lineData`);
      assertAllFiniteFloat32(scene.ribbonData, `beat[${i}].ribbonData`);
      assertAllFiniteFloat32(scene.sheetData, `beat[${i}].sheetData`);
    }
  });

  it('supports full playback & navigation controls (start, pause, resume, togglePlay, next, prev, goToBeat, goToAct, restart, exit)', () => {
    const ctrl = createTwoTowerTourController(model);

    // Initial idle state
    const idleSnap = ctrl.getSnapshot();
    expect(idleSnap.status).toBe('idle');
    expect(idleSnap.isActive).toBe(false);
    expect(idleSnap.isPlaying).toBe(false);
    expect(idleSnap.beatIndex).toBe(0);
    expect(idleSnap.canPrev).toBe(false);
    expect(idleSnap.canNext).toBe(true);

    // start()
    const started = ctrl.start();
    expect(started.status).toBe('playing');
    expect(started.isActive).toBe(true);
    expect(started.isPlaying).toBe(true);
    expect(started.beatIndex).toBe(0);

    // next() and prev() preserve playing state
    const s1 = ctrl.next();
    expect(s1.beatIndex).toBe(1);
    expect(s1.isPlaying).toBe(true);
    expect(s1.canPrev).toBe(true);

    const s0 = ctrl.prev();
    expect(s0.beatIndex).toBe(0);
    expect(s0.isPlaying).toBe(true);

    // prev() clamps cleanly at 0
    expect(ctrl.prev().beatIndex).toBe(0);

    // pause() / resume() / togglePlay()
    const paused = ctrl.pause();
    expect(paused.status).toBe('paused');
    expect(paused.isActive).toBe(true);
    expect(paused.isPlaying).toBe(false);

    const resumed = ctrl.resume();
    expect(resumed.status).toBe('playing');
    expect(resumed.isPlaying).toBe(true);

    const toggledOff = ctrl.togglePlay();
    expect(toggledOff.status).toBe('paused');
    expect(toggledOff.isPlaying).toBe(false);

    const toggledOn = ctrl.togglePlay();
    expect(toggledOn.status).toBe('playing');
    expect(toggledOn.isPlaying).toBe(true);

    // goToAct() jumps to exact startBeatIndex of every act (0..3) and clamps out of bounds
    for (let actIdx = 0; actIdx < TWOTOWER_TOUR_ACT_COUNT; actIdx++) {
      const actSnap = ctrl.goToAct(actIdx);
      expect(actSnap.actIndex).toBe(actIdx);
      expect(actSnap.beatIndex).toBe(TWOTOWER_TOUR_ACTS[actIdx].startBeatIndex);
      expect(actSnap.actBeatIndex).toBe(0);
      expect(actSnap.acts[actIdx].isCurrent).toBe(true);
      expect(actSnap.acts.filter((a) => a.isCurrent)).toHaveLength(1);
    }
    expect(ctrl.goToAct(-5).actIndex).toBe(0);
    expect(ctrl.goToAct(99).actIndex).toBe(3);

    // goToBeat() clamps finite bounds and rejects NaN / Infinity
    expect(ctrl.goToBeat(-10).beatIndex).toBe(0);
    const last = ctrl.goToBeat(999);
    expect(last.beatIndex).toBe(TWOTOWER_TOUR_BEAT_COUNT - 1);
    expect(last.canNext).toBe(false);
    expect(last.progressFraction).toBe(1);
    expect(() => ctrl.goToBeat(Number.NaN)).toThrow(RangeError);
    expect(() => ctrl.goToBeat(Number.POSITIVE_INFINITY)).toThrow(RangeError);
    expect(() => ctrl.goToAct(Number.NaN)).toThrow(RangeError);

    // next() at final beat stays at final beat and pauses playback cleanly
    const afterEnd = ctrl.next();
    expect(afterEnd.beatIndex).toBe(TWOTOWER_TOUR_BEAT_COUNT - 1);
    expect(afterEnd.status).toBe('paused');
    expect(afterEnd.isPlaying).toBe(false);

    // restart() returns to beat 0 playing
    const restarted = ctrl.restart();
    expect(restarted.beatIndex).toBe(0);
    expect(restarted.status).toBe('playing');
    expect(restarted.isActive).toBe(true);
    expect(restarted.isPlaying).toBe(true);

    // exit() exits cleanly while keeping getSnapshot() valid and finite
    ctrl.goToBeat(12);
    const exited = ctrl.exit();
    expect(exited.status).toBe('exited');
    expect(exited.isActive).toBe(false);
    expect(exited.isPlaying).toBe(false);
    expect(exited.beatIndex).toBe(12);
    expect(ctrl.getSnapshot().status).toBe('exited');

    // togglePlay() from exited restarts cleanly at beat 0
    const restartedFromExit = ctrl.togglePlay();
    expect(restartedFromExit.status).toBe('playing');
    expect(restartedFromExit.beatIndex).toBe(0);
  });

  it('is 100% deterministic across independent controller instances and works on alternate model seeds', () => {
    const c1 = createTwoTowerTourController(model);
    const c2 = createTwoTowerTourController(model);

    for (let i = 0; i < TWOTOWER_TOUR_BEAT_COUNT; i++) {
      expect(c1.goToBeat(i)).toEqual(c2.goToBeat(i));
    }

    // Also verify alternate synthetic model seed completes all 28 beats cleanly with zero non-finite values
    const altModel = createTwoTowerModelParameters({ seed: 0x12345678, perturbationScale: 0.1 });
    const altCtrl = createTwoTowerTourController(altModel);
    altCtrl.start();
    for (let i = 0; i < TWOTOWER_TOUR_BEAT_COUNT; i++) {
      const snap = altCtrl.goToBeat(i);
      expect(() => assertPresentationSafeNoInfinity(snap, `altBeat[${i}]`)).not.toThrow();
      expect(snap.liveMetrics.length).toBeGreaterThanOrEqual(4);
    }
  });

  it('strictly isolates external user exploration sessions (verifyTourIsolation positive & negative tests)', () => {
    // Build a non-trivial external user exploration session with custom config, branch, and in-flight block
    let userSession = createTwoTowerSession(model, {
      sessionId: 'user-exploration-session',
      promptTokenIds: [2, 3, 4, 5],
      mode: 'mask_diffusion',
      blockSize: 8,
      stepsPerBlock: 6,
      confidenceThreshold: 0.8,
      temperature: 0.1,
    });
    userSession = forkTwoTowerSessionBranch(userSession, {
      label: 'User AR Experiment',
      configPatch: { mode: 'ar' },
      switchToNewBranch: false,
    });
    userSession = initTwoTowerDiffusionBlock(userSession);
    userSession = runTwoTowerDiffusionRefinement(userSession);
    const userSessionBefore = userSession;

    // Exercise the entire tour controller across all 28 beats, acts, pause/resume, restart, and exit
    const tourCtrl = createTwoTowerTourController(model);
    tourCtrl.start();
    for (let i = 0; i < TWOTOWER_TOUR_BEAT_COUNT; i++) {
      tourCtrl.goToBeat(i);
    }
    tourCtrl.goToAct(1);
    tourCtrl.pause();
    tourCtrl.next();
    tourCtrl.prev();
    tourCtrl.resume();
    tourCtrl.restart();
    tourCtrl.exit();

    // Positive isolation verification
    const report = verifyTourIsolation(userSessionBefore, tourCtrl, userSession);
    expect(report.isolated).toBe(true);
    expect(report.sameSessionReference).toBe(true);
    expect(report.sameSessionId).toBe(true);
    expect(report.sameActiveBranchId).toBe(true);
    expect(report.sameCursorIndex).toBe(true);
    expect(report.sameMutationSeq).toBe(true);
    expect(report.sameBranchCount).toBe(true);
    expect(report.sameActiveRecordId).toBe(true);
    expect(report.sameContextSnapshotDigest).toBe(true);
    expect(report.sameContextCacheDigest).toBe(true);
    expect(report.samePendingConfig).toBe(true);
    expect(report.distinctFromTourSessionId).toBe(true);
    expect(report.distinctFromTourSessionReference).toBe(true);
    expect(report.snapshotIntegrityValid).toBe(true);
    expect(report.sharedArrayBufferCount).toBe(0);
    expect(report.reasons).toEqual([]);

    // Negative test 1: detects session advancement / mutation
    const userSessionAdvanced = stepTwoTowerSessionAr(
      createTwoTowerSession(model, { sessionId: 'user-ar-check', mode: 'ar' }),
    );
    const diffReport = verifyTourIsolation(userSessionBefore, tourCtrl, userSessionAdvanced);
    expect(diffReport.isolated).toBe(false);
    expect(diffReport.reasons.length).toBeGreaterThan(0);

    // Negative test 2: detects silent in-place Float32Array tampering on user snapshot
    const tamperCandidate = createTwoTowerSession(model, {
      sessionId: 'user-tamper-check',
      promptTokenIds: [0, 1, 2, 3],
    });
    const activeRec = getActiveRecord(tamperCandidate);
    const firstGqa = activeRec.contextSnapshot.ctxState.gqaCaches.values().next().value!;
    firstGqa.K[0] += 1.0;
    const tamperReport = verifyTourIsolation(tamperCandidate, tourCtrl, tamperCandidate);
    expect(tamperReport.isolated).toBe(false);
    expect(tamperReport.snapshotIntegrityValid).toBe(false);
    expect(tamperReport.reasons.some((r) => r.includes('integrity failed'))).toBe(true);
  });

  it('maintains strict headless purity (zero DOM, WebGL, app imports, or Math.random)', () => {
    const raw = readFileSync(
      resolve(process.cwd(), 'src/twotower/story/walkthrough.ts'),
      'utf8',
    );
    // Strip block and line comments before checking executable TypeScript code
    const codeOnly = raw
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\/\/[^\n]*/g, '');
    expect(codeOnly).not.toContain('Math.random');
    expect(codeOnly).not.toMatch(/\bdocument\./);
    expect(codeOnly).not.toMatch(/\bwindow\./);
    expect(codeOnly).not.toMatch(/\bWebGL/);
    expect(codeOnly).not.toMatch(/from\s+['"][^'"]*\/app\//);
  });
});
