import { describe, expect, it } from 'vitest';

import {
  computeLlama4AttnScale,
  createDefaultDiffusionModel,
  createDiffusionGuidedTour,
  createDiffusionSession,
  DIFFUSION_TOUR_ACT_DEFS,
  findInvalidUiNumericStrings,
  type DiffusionExecutionMode,
  type DiffusionExplorerMaskMode,
  type DiffusionFreeExploreRestoreState,
  type DiffusionTourActId,
} from '../src/diffusion';
import {
  buildDiffusionSpatialScene,
  DEFAULT_DIFFUSION_FOCUS_STATE,
  DIFFUSION_SPATIAL_VIEW_MODES,
  validateSceneGeometryFinite,
  type DiffusionSpatialViewMode,
} from '../src/diffusion/scene';

const ALL_VIEW_MODES: readonly DiffusionSpatialViewMode[] = DIFFUSION_SPATIAL_VIEW_MODES.map(
  (v) => v.id,
);

const ALL_EXPLORER_MASK_MODES: readonly DiffusionExplorerMaskMode[] = [
  'joint_training',
  'ar_causal',
  'block_bidir',
  'causal_commit_verify',
  'vlm_quadratic_ss',
];

const FORBIDDEN_SPEEDUP_PATTERNS: readonly RegExp[] = [
  /\b\d+(?:\.\d+)?\s*[x×]\s*(?:faster|speed\s*-?\s*up)\b/i,
  /\b(?:3x|5x)\s+(?:faster|speedup)\b/i,
  /\bfaster\s+than\b/i,
];

describe('DF4 Guided Tour — Structure, Coverage, Isolation & Live Scene Verification', () => {
  const model = createDefaultDiffusionModel();

  it('produces 4 acts and 26 contiguous beats covering all 7 views, 5 mask tabs, and 4 execution modes within bounds', () => {
    const tour = createDiffusionGuidedTour(model);

    expect(tour.getStatus()).toBe('idle');
    expect(tour.isActive()).toBe(false);
    expect(tour.getCurrentBeatIndex()).toBe(-1);
    expect(tour.getCurrentBeat()).toBeNull();
    expect(tour.getRestoreState()).toBeNull();
    expect(tour.next()).toBeNull();
    expect(tour.prev()).toBeNull();
    expect(tour.goToBeat(0)).toBeNull();
    expect(tour.goToAct('act-1')).toBeNull();
    expect(tour.exit()).toBeNull();

    const beatCount = tour.getBeatCount();
    expect(beatCount).toBe(26);
    expect(beatCount).toBeGreaterThanOrEqual(24);
    expect(beatCount).toBeLessThanOrEqual(28);

    const beats = tour.getAllBeats();
    expect(beats).toHaveLength(beatCount);

    const acts = tour.getActs();
    expect(acts).toHaveLength(4);
    expect(acts.map((a) => a.id)).toEqual(['act-1', 'act-2', 'act-3', 'act-4']);
    expect(acts.map((a) => a.id)).toEqual(DIFFUSION_TOUR_ACT_DEFS.map((d) => d.id));
    expect(acts.every((a) => !a.isCurrent)).toBe(true);

    // Contiguous beat indices and contiguous act spans covering every beat
    let expectedStart = 0;
    for (let aIdx = 0; aIdx < acts.length; aIdx++) {
      const act = acts[aIdx]!;
      expect(act.actIndex).toBe(aIdx);
      expect(act.beatCount).toBeGreaterThan(0);
      expect(act.startBeatIndex).toBe(expectedStart);
      for (let bIdx = act.startBeatIndex; bIdx < act.startBeatIndex + act.beatCount; bIdx++) {
        const beat = beats[bIdx]!;
        expect(beat.beatIndex).toBe(bIdx);
        expect(beat.totalBeats).toBe(beatCount);
        expect(beat.actId).toBe(act.id);
        expect(beat.actIndex).toBe(act.actIndex);
      }
      expectedStart += act.beatCount;
    }
    expect(expectedStart).toBe(beatCount);

    // All 7 spatial view modes covered
    const coveredViews = new Set<DiffusionSpatialViewMode>(beats.map((b) => b.viewMode));
    for (const viewMode of ALL_VIEW_MODES) {
      expect(coveredViews.has(viewMode), `missing viewMode: ${viewMode}`).toBe(true);
    }

    // All 5 Mask Explorer modes covered on mask_explorer beats
    const coveredMaskTabs = new Set<DiffusionExplorerMaskMode>(
      beats.filter((b) => b.viewMode === 'mask_explorer').map((b) => b.focus.selectedMaskTab),
    );
    for (const maskTab of ALL_EXPLORER_MASK_MODES) {
      expect(coveredMaskTabs.has(maskTab), `missing maskTab: ${maskTab}`).toBe(true);
    }

    // All 4 execution modes + DLM causalContext true/false covered
    const coveredModes = new Set<DiffusionExecutionMode>(beats.map((b) => b.mode));
    expect(coveredModes.has('ar')).toBe(true);
    expect(coveredModes.has('dlm')).toBe(true);
    expect(coveredModes.has('linear_spec')).toBe(true);
    expect(coveredModes.has('linear_spec_lora')).toBe(true);
    expect(beats.some((b) => b.mode === 'dlm' && b.causalContext === true)).toBe(true);
    expect(beats.some((b) => b.mode === 'dlm' && b.causalContext === false)).toBe(true);

    // Tour session stays strictly within memory and branch/record bounds
    const tourSession = tour.getTourSession();
    const state = tourSession.getState();
    const mem = tourSession.measureMemory();
    expect(state.branches.length).toBeLessThanOrEqual(12);
    expect(state.records.size).toBeLessThanOrEqual(128);
    expect(mem.withinBudget).toBe(true);
    expect(mem.totalRetainedBytes).toBeLessThanOrEqual(12_582_912);
  });

  it('isolates Tour from Free-Explore session and restores exact pre-tour state on exit despite caller mutation', () => {
    // 1. Build a multi-branch Free-Explore session (step + rewind + fork)
    const freeExplore = createDiffusionSession(model, {
      config: { mode: 'ar', causalContext: true, blockLength: 4, maxNewTokens: 12, temperature: 0 },
    });
    const r0 = freeExplore.step(); // prefill
    freeExplore.step(); // ar_step 1
    freeExplore.step(); // ar_step 2
    freeExplore.seekRecord(r0.recordId);
    freeExplore.updateConfig({ mode: 'linear_spec', blockLength: 4, maxNewTokens: 12, temperature: 0 });
    const forkedRec = freeExplore.step(); // fork branch-2

    const preState = freeExplore.getState();
    const preActive = freeExplore.getActiveRecord()!;
    const preDigest = preActive.kvSnapshot.snapshotDigest;
    const preCounters = { ...preActive.counters };
    const preRecordsSize = preState.records.size;
    const preBranchesLength = preState.branches.length;
    const preActiveRecordId = preState.activeRecordId;
    const preActiveBranchId = preState.activeBranchId;

    expect(preBranchesLength).toBe(2);
    expect(preActiveRecordId).toBe(forkedRec.recordId);

    const restoreSnapshot: DiffusionFreeExploreRestoreState = {
      activeRecordId: preActiveRecordId,
      activeBranchId: preActiveBranchId,
      focus: {
        ...DEFAULT_DIFFUSION_FOCUS_STATE,
        viewMode: 'gqa_rope',
        selectedLayerIndex: 19,
        selectedQHead: 12,
        selectedKvHead: 3,
        selectedRotaryPlane: 2,
        selectedMaskTab: 'causal_commit_verify',
        selectedPartId: 'gqa-q12',
        hoveredPartId: 'gqa-kv3',
        hoveredCell: {
          viewMode: 'gqa_rope',
          layerIndex: 19,
          partId: 'gqa-q12',
          tensorKey: 'attn_probs',
          qHeadIndex: 12,
          kvHeadIndex: 3,
          rotaryPlane: 2,
          tokenIndex: 3,
          absPosition: 10,
          row: 3,
          col: 2,
          semanticTag: 'hover-test',
        },
        pinnedCell: {
          viewMode: 'gqa_rope',
          layerIndex: 19,
          partId: 'gqa-kv3',
          tensorKey: 'q_scaled',
          qHeadIndex: 12,
          kvHeadIndex: 3,
          rotaryPlane: 2,
          tokenIndex: 1,
          absPosition: 8,
          row: 1,
          col: 4,
          semanticTag: 'pin-test',
        },
        activeStepperStep: 3,
        swigluAccumReveal: 0.75,
        selectedTokenRow: 2,
      },
      cameraPose: {
        target: [1.5, 2.5, -3.5],
        azimuthDeg: 42,
        elevationDeg: 18,
        distance: 28,
        fovDeg: 38,
      },
      inspectorTab: 'ledger',
      inspectorOpen: true,
      ropeScrubPosition: 17,
    };

    const expectedRestored = structuredClone(restoreSnapshot);

    // 2. Enter tour, mutate caller's original restoreSnapshot, navigate all acts & beats
    const tour = createDiffusionGuidedTour(model);
    expect(tour.getTourSession()).not.toBe(freeExplore);

    const entered = tour.enter(restoreSnapshot);
    expect(entered.beatIndex).toBe(0);
    expect(tour.isActive()).toBe(true);
    expect(tour.getStatus()).toBe('active');

    // Mutate caller's original snapshot to prove deep-copy defense
    (restoreSnapshot.cameraPose.target as unknown as number[])[0] = 9999;
    (restoreSnapshot.focus as { selectedLayerIndex: number }).selectedLayerIndex = 0;

    // Boundary checks at beat 0
    expect(tour.prev()).toBeNull();
    expect(tour.goToBeat(-1)).toBeNull();
    expect(tour.goToBeat(tour.getBeatCount())).toBeNull();

    // Navigate via next(), prev(), goToAct('act-1'..'act-4'), and goToBeat(0..N-1)
    const next1 = tour.next()!;
    expect(next1.beatIndex).toBe(1);
    const back0 = tour.prev()!;
    expect(back0.beatIndex).toBe(0);

    const actIds: readonly DiffusionTourActId[] = ['act-1', 'act-2', 'act-3', 'act-4'];
    for (const actId of actIds) {
      const actBeat = tour.goToAct(actId)!;
      expect(actBeat.actId).toBe(actId);
      const actSummary = tour.getActs().find((a) => a.id === actId)!;
      expect(actSummary.isCurrent).toBe(true);
      expect(actBeat.beatIndex).toBe(actSummary.startBeatIndex);
    }

    for (let i = 0; i < tour.getBeatCount(); i++) {
      const beat = tour.goToBeat(i)!;
      expect(beat.beatIndex).toBe(i);
      expect(tour.getCurrentBeatIndex()).toBe(i);
      expect(tour.getCurrentBeat()?.beatId).toBe(beat.beatId);
    }
    expect(tour.next()).toBeNull(); // at final beat

    // 3. Assert Free-Explore session was NEVER mutated during Tour navigation
    const postState = freeExplore.getState();
    const postActive = freeExplore.getActiveRecord()!;
    expect(postState.activeRecordId).toBe(preActiveRecordId);
    expect(postState.activeBranchId).toBe(preActiveBranchId);
    expect(postState.records.size).toBe(preRecordsSize);
    expect(postState.branches.length).toBe(preBranchesLength);
    expect(postActive.kvSnapshot.snapshotDigest).toBe(preDigest);
    expect(postActive.counters).toEqual(preCounters);

    // 4. Exit tour and verify exact deep-copied restoration
    const restored = tour.exit();
    expect(tour.isActive()).toBe(false);
    expect(tour.getStatus()).toBe('exited');
    expect(tour.getCurrentBeatIndex()).toBe(-1);
    expect(tour.getCurrentBeat()).toBeNull();
    expect(tour.getRestoreState()).toBeNull();
    expect(tour.exit()).toBeNull();

    expect(restored).toEqual(expectedRestored);
    expect(restored?.cameraPose.target[0]).toBe(1.5);
  });

  it('builds finite WebGL2 scenes, passes atlas audit, avoids invalid strings, and matches live tensors across all 26 beats', () => {
    const tour = createDiffusionGuidedTour(model);
    tour.enter({
      activeRecordId: null,
      activeBranchId: 'branch-1',
      focus: { ...DEFAULT_DIFFUSION_FOCUS_STATE },
      cameraPose: {
        target: [0, 0, 0],
        azimuthDeg: 35,
        elevationDeg: 22,
        distance: 24,
        fovDeg: 40,
      },
      inspectorTab: 'inspect',
      inspectorOpen: false,
      ropeScrubPosition: null,
    });

    const beatCount = tour.getBeatCount();
    let sawCrossingLlama4Beat = false;
    let sawInRangeLlama4Beat = false;
    let sawSpecLoraBeat = false;
    let sawSpecBaseBeat = false;
    let sawDlmCausalBeat = false;
    let sawDlmBidirBeat = false;

    for (let i = 0; i < beatCount; i++) {
      const beat = tour.goToBeat(i)!;
      const ctx = tour.getTourContext();

      expect(ctx.record.recordId).toBe(beat.recordId);
      expect(ctx.branchId).toBe(beat.branchId);
      expect(ctx.modeAtRecord).toBe(beat.mode);
      expect(ctx.causalContextAtRecord).toBe(beat.causalContext);
      expect(ctx.kvIntegrityValid).toBe(true);

      // Build actual WebGL2 spatial scene and verify finite geometry & atlas non-overlap
      const scene = buildDiffusionSpatialScene(ctx, beat.focus);
      const geomCheck = validateSceneGeometryFinite(scene);
      expect(geomCheck.valid, `beat ${beat.beatId}: ${geomCheck.errors.join('; ')}`).toBe(true);

      if (scene.atlasAudit) {
        expect(scene.atlasAudit.hasOverlap, `beat ${beat.beatId} atlas overlap`).toBe(false);
        expect(
          scene.atlasAudit.outOfBoundsIds,
          `beat ${beat.beatId} atlas OOB`,
        ).toEqual([]);
      }

      // Zero NaN / Infinity / undefined / stringified null anywhere in beat payload
      expect(findInvalidUiNumericStrings(beat), `beat ${beat.beatId} invalid strings`).toEqual([]);
      expect(beat.liveMetrics.length).toBeGreaterThan(0);

      // Zero fabricated speedup claims
      const copyTexts = [
        beat.title,
        beat.subtitle,
        beat.narrative,
        beat.takeaway,
        beat.provenanceSummary,
        beat.epistemicNote,
        ...beat.liveMetrics.map((m) => `${m.label}: ${m.value}`),
      ];
      for (const text of copyTexts) {
        for (const pattern of FORBIDDEN_SPEEDUP_PATTERNS) {
          expect(pattern.test(text), `fabricated speedup text in ${beat.beatId}: ${text}`).toBe(
            false,
          );
        }
      }

      // Live tensor & KV seqLen consistency checks
      const kvMetric = beat.liveMetrics.find(
        (m) =>
          m.label === 'KV Seq Len' ||
          m.label === 'KV Prefix' ||
          m.label === 'Prefill KV' ||
          m.label === 'Active KV' ||
          m.label === 'KV Capacity',
      );
      if (kvMetric) {
        expect(kvMetric.value).toContain(`${ctx.kvSnapshot.seqLen}`);
      }

      if (beat.beatId === 'act2-llama4-in-range') {
        sawInRangeLlama4Beat = true;
        expect(beat.ropeScrubPosition).toBe(15);
        const sMetric = beat.liveMetrics.find((m) => m.label === 's_L4(15)');
        expect(sMetric?.value).toBe('1.0000×');
      }

      if (beat.beatId === 'act2-llama4-crossing') {
        sawCrossingLlama4Beat = true;
        const pos = ctx.record.primaryPassCheckpoint.cachePositions[0] ?? 0;
        expect(pos).toBeGreaterThanOrEqual(16);
        const expectedScale = computeLlama4AttnScale(
          pos,
          ctx.model.config.rope.llama4Beta,
          ctx.model.config.rope.originalMaxPositionEmbeddings,
        );
        expect(expectedScale.toFixed(4)).toBe('1.0693');
        const scaleMetric = beat.liveMetrics.find((m) => m.label === 'Live s_L4');
        expect(scaleMetric?.value).toBe('1.0693×');
      }

      if (beat.mode === 'linear_spec_lora') {
        sawSpecLoraBeat = true;
        expect(ctx.record.kind).toBe('spec_cycle');
        expect(beat.loraActiveOnDraft).toBe(true);
        expect(ctx.loraActiveOnDraft).toBe(true);
        expect(ctx.record.specMeta?.loraEnabled).toBe(true);
        // Verify pass checkpoint itself runs with LoRA OFF (D17)
        const l0 = ctx.resolveLayerTrace(0);
        expect(l0.attention.loraEnabled).toBe(false);
        if (beat.beatId === 'act4-spec-kv-rollback') {
          const lazySlabsMetric = beat.liveMetrics.find((m) => m.label === 'Lazy Slabs');
          expect(lazySlabsMetric?.value).toBe(`${ctx.resolveSpecCroppedSuffixSlabs().length}`);
        }
      }

      if (beat.mode === 'linear_spec') {
        sawSpecBaseBeat = true;
        expect(ctx.record.kind).toBe('spec_cycle');
        expect(beat.loraActiveOnDraft).toBe(false);
        expect(ctx.loraActiveOnDraft).toBe(false);
      }

      if (beat.mode === 'dlm' && beat.causalContext) {
        sawDlmCausalBeat = true;
        expect(ctx.record.kvMaskProvenance).toBe('token_causal');
      }

      if (beat.mode === 'dlm' && !beat.causalContext) {
        sawDlmBidirBeat = true;
        expect(ctx.record.kvMaskProvenance).toBe('block_bidirectional');
      }
    }

    expect(sawInRangeLlama4Beat).toBe(true);
    expect(sawCrossingLlama4Beat).toBe(true);
    expect(sawSpecLoraBeat).toBe(true);
    expect(sawSpecBaseBeat).toBe(true);
    expect(sawDlmCausalBeat).toBe(true);
    expect(sawDlmBidirBeat).toBe(true);

    tour.exit();
  });
});
