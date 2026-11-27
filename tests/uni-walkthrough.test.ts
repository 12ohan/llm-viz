/**
 * Milestone UNI6 — Slice 6A Unit & Isolation Suite (`tests/uni-walkthrough.test.ts`)
 *
 * Verifies:
 *   1. Exact 5-Act, 28-Beat catalog integrity (`0..27` contiguous indices, all 7
 *      `UniSpatialViewMode` views & all 7 `UniTourDemoKey` demos covered).
 *   2. Live 3D scene construction across ALL 28 beats (`tour.buildBeatScene(0..27)`),
 *      asserting `validateSceneGeometryFinite(scene).valid === true`, `0` atlas overlaps,
 *      `0` OOB rects, `withinBudget: true`, and strict `U25` live-`RG32F` compliance.
 *   3. Pedagogical milestones across Acts I–V (D5 waste/pruning, U18 16×8 border lock,
 *      L17 joint → L18 split, U26 D=17 mismatch, U9/U10 refinement canvas, Dual-CFG
 *      editing + 16×8 crop-back, FLARE understanding, U27 grounding, Read-Only OPD).
 *   4. Navigation & clamping (`start`, `goToBeat`, `nextBeat`, `prevBeat`, `goToAct`, `exit`).
 *   5. Byte-for-byte isolation proof: an external Free-Explore `UniSession` is 100%
 *      untouched after entering the tour, stepping all 28 beats, and exiting.
 *   6. Guardrail checks: zero `Math.random(`, zero `src/ui/controls.ts`, `src/twotower/`,
 *      or `src/diffusion/` imports in `src/uni/story/walkthrough.ts`.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
} from '../src/render/sheets';
import { createDefaultUniRuntimeBundle } from '../src/uni/imageSampler';
import { dualHashHex } from '../src/uni/prng';
import { validateSceneGeometryFinite } from '../src/uni/scene/provenance';
import {
  DEFAULT_UNI_SPATIAL_FOCUS_STATE,
  UNI_SPATIAL_VIEW_MODES,
  type UniSpatialViewMode,
} from '../src/uni/scene/types';
import { UniSession } from '../src/uni/session';
import {
  UNI_TOUR_TOTAL_ACTS,
  UNI_TOUR_TOTAL_BEATS,
  type UniFreeExploreRestoreState,
  type UniTourActId,
  type UniTourDemoKey,
} from '../src/uni/story/types';
import {
  createUniGuidedTour,
  UNI_TOUR_ACTS,
  UNI_TOUR_BEATS,
} from '../src/uni/story/walkthrough';

const SHARED_BUNDLE = createDefaultUniRuntimeBundle();

function hashTypedArray(arr: ArrayBufferView | null | undefined): string {
  if (!arr) return 'null';
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  return `${bytes.byteLength}:${dualHashHex(bytes)}`;
}

interface SessionByteDigest {
  readonly activeBranchId: string | null;
  readonly activeRecordId: string | null;
  readonly branchCount: number;
  readonly recordCount: number;
  readonly memoryAudit: ReturnType<UniSession['auditMemory']>;
  readonly branchDigests: readonly {
    readonly branchId: string;
    readonly taskMode: string;
    readonly prefillMode: string;
    readonly recordIds: readonly string[];
    readonly prefixCacheSeqLen: number;
    readonly sourceVqCodesHash: string;
  }[];
  readonly recordDigests: readonly {
    readonly recordId: string;
    readonly branchId: string;
    readonly stepIndex: number;
    readonly routingHash: string;
    readonly positionIdsHash: string;
    readonly imageStagesHash: string;
    readonly imageRepsHash: string;
    readonly imageLevelsHash: string;
    readonly imageNoiseHash: string;
    readonly textTokensHash: string;
    readonly residualHashes: readonly (readonly string[])[];
  }[];
}

function captureSessionByteDigest(session: UniSession): SessionByteDigest {
  const branches = session.getAllBranches();
  const records = session.getAllRecords();
  return {
    activeBranchId: session.getActiveBranchId(),
    activeRecordId: session.getActiveRecordId(),
    branchCount: branches.length,
    recordCount: records.length,
    memoryAudit: session.auditMemory(),
    branchDigests: branches.map((b) => ({
      branchId: b.branchId,
      taskMode: b.taskMode,
      prefillMode: b.prefillMode,
      recordIds: [...b.recordIds],
      prefixCacheSeqLen: b.prefixCache.seqLen,
      sourceVqCodesHash: hashTypedArray(b.sourceVqCodes),
    })),
    recordDigests: records.map((r) => ({
      recordId: r.recordId,
      branchId: r.branchId,
      stepIndex: r.stepIndex,
      routingHash: hashTypedArray(r.routing),
      positionIdsHash: hashTypedArray(r.positionIds),
      imageStagesHash: hashTypedArray(r.imageState?.stages),
      imageRepsHash: hashTypedArray(r.imageState?.representatives),
      imageLevelsHash: hashTypedArray(r.imageState?.levels),
      imageNoiseHash: hashTypedArray(r.imageState?.noiseWeights),
      textTokensHash: hashTypedArray(r.textState?.tokenIds),
      residualHashes: r.perBranchResiduals.map((row) =>
        row.map((cp) => hashTypedArray(cp)),
      ),
    })),
  };
}

describe('UNI6 Slice 6A — 5-Act, 28-Beat Guided Pedagogical Tour', () => {
  it('defines exact 5 Acts and 28 contiguous Beats covering all 7 views and all 7 demo keys', () => {
    expect(UNI_TOUR_ACTS.length).toBe(UNI_TOUR_TOTAL_ACTS);
    expect(UNI_TOUR_ACTS.length).toBe(5);
    expect(UNI_TOUR_BEATS.length).toBe(UNI_TOUR_TOTAL_BEATS);
    expect(UNI_TOUR_BEATS.length).toBe(28);

    // Verify exact 5 + 5 + 6 + 7 + 5 Act partition
    const expectedActCounts = [5, 5, 6, 7, 5];
    let cursor = 0;
    for (let a = 0; a < UNI_TOUR_ACTS.length; a++) {
      const act = UNI_TOUR_ACTS[a]!;
      expect(act.actNumber).toBe(a + 1);
      expect(act.id).toBe(`act${a + 1}`);
      expect(act.startBeatIndex).toBe(cursor);
      expect(act.beatCount).toBe(expectedActCounts[a]);
      cursor += act.beatCount;
    }
    expect(cursor).toBe(28);

    const seenBeatIds = new Set<string>();
    const seenViews = new Set<UniSpatialViewMode>();
    const seenDemoKeys = new Set<UniTourDemoKey>();

    for (let i = 0; i < UNI_TOUR_BEATS.length; i++) {
      const beat = UNI_TOUR_BEATS[i]!;
      expect(beat.index).toBe(i);
      expect(seenBeatIds.has(beat.id)).toBe(false);
      seenBeatIds.add(beat.id);

      const act = UNI_TOUR_ACTS[beat.actNumber - 1]!;
      expect(beat.actId).toBe(act.id);
      expect(beat.beatInAct).toBe(i - act.startBeatIndex + 1);
      expect(beat.beatInAct).toBeGreaterThanOrEqual(1);
      expect(beat.beatInAct).toBeLessThanOrEqual(act.beatCount);

      expect(beat.title.length).toBeGreaterThan(5);
      expect(beat.takeaway.length).toBeGreaterThan(10);
      expect(beat.narrative.length).toBeGreaterThan(30);
      expect(beat.ledgerRefs.length).toBeGreaterThan(0);

      seenViews.add(beat.viewMode);
      seenDemoKeys.add(beat.demoKey);
    }

    // Every UniSpatialViewMode (7/7) is covered
    for (const viewDesc of UNI_SPATIAL_VIEW_MODES) {
      expect(seenViews.has(viewDesc.id)).toBe(true);
    }

    // Every UniTourDemoKey (7/7) is covered
    const allDemoKeys: readonly UniTourDemoKey[] = [
      't2i_elastic',
      't2i_hf_faithful',
      'edit_panorama_elastic',
      'edit_panorama_hf',
      'visual_understanding',
      'text_understanding',
      'object_grounding',
    ];
    for (const key of allDemoKeys) {
      expect(seenDemoKeys.has(key)).toBe(true);
    }
  });

  it('builds valid, finite, 0-overlap, U25-compliant live 3D scenes across all 28 beats within memory & latency budgets', () => {
    const startMs = performance.now();
    const tour = createUniGuidedTour(SHARED_BUNDLE);

    const scenes = [];
    for (let i = 0; i < UNI_TOUR_TOTAL_BEATS; i++) {
      const beat = UNI_TOUR_BEATS[i]!;
      const scene = tour.buildBeatScene(i);
      scenes.push(scene);

      expect(scene.viewMode).toBe(beat.viewMode);

      // Finite geometry + buffer stride + U25 + RGB preview validation
      const validation = validateSceneGeometryFinite(scene);
      expect(validation.errors).toEqual([]);
      expect(validation.valid).toBe(true);

      // Atlas audit: 0 overlaps, 0 OOB, withinBudget === true
      if (scene.atlasAudit !== null) {
        expect(scene.atlasAudit.hasOverlap).toBe(false);
        expect(scene.atlasAudit.overlapPair).toBeNull();
        expect(scene.atlasAudit.outOfBoundsIds).toEqual([]);
        expect(scene.atlasAudit.withinBudget).toBe(true);
      }

      // Explicit U25 check on every causal score/softmax sheet
      for (let s = 0; s < scene.sheetCount; s++) {
        const base = s * SHEET_INSTANCE_FLOATS;
        const sheetKind = Math.round(scene.sheetData[base + 8]!);
        const uWidth = scene.sheetData[base + 15]!;
        const vHeight = scene.sheetData[base + 16]!;
        if (
          sheetKind === SHEET_KIND_CAUSAL_SCORES ||
          sheetKind === SHEET_KIND_CAUSAL_SOFTMAX
        ) {
          expect(scene.valueAtlas).not.toBeNull();
          expect(uWidth).toBeGreaterThan(0);
          expect(vHeight).toBeGreaterThan(0);
        }
      }
    }

    const elapsedMs = performance.now() - startMs;
    expect(elapsedMs).toBeLessThan(3000);

    // Verify private tourSession stays well inside bounded caps (7 branches <= 12, <= 128 records, <= 12 MiB)
    const audit = tour.tourSession.auditMemory();
    expect(audit.branchCount).toBe(7);
    expect(audit.branchCount).toBeLessThanOrEqual(audit.maxBranches);
    expect(audit.recordCount).toBeLessThanOrEqual(audit.maxRecords);
    expect(audit.withinBudget).toBe(true);
    expect(audit.nonActiveRecordsWithFullLogits).toBe(0);

    // Spot-check key pedagogical landmarks across all 5 Acts:
    // Act I: D5 HF-Faithful (beat 2) vs Elastic-Pruned (beat 3)
    expect(scenes[2]!.prefillMode).toBe('hf_faithful');
    expect(scenes[2]!.d5Comparison?.wastedPrefixKvFraction).toBeCloseTo(16 / 34, 6);
    expect(scenes[2]!.d5Comparison?.stage2UlpDiff).toBe(0);
    expect(scenes[3]!.prefillMode).toBe('elastic_pruned');

    // Act II: 16×8 panorama letterbox & U18 8/16 border-lock mask (beats 5–6)
    const beat6Source = scenes[6]!.rgbPreviews.find((p) => p.kind === 'source');
    const beat6Letterboxed = scenes[6]!.rgbPreviews.find((p) => p.kind === 'letterboxed');
    expect(beat6Source?.width).toBe(16);
    expect(beat6Source?.height).toBe(8);
    expect(beat6Letterboxed?.width).toBe(16);
    expect(beat6Letterboxed?.height).toBe(16);
    expect(beat6Letterboxed?.paddingLockMask?.lockedCellCount).toBe(8);

    // Act III: L17 joint (beat 11) -> L18 split (beat 12)
    expect(scenes[11]!.layerIndex).toBe(17);
    expect(scenes[12]!.layerIndex).toBe(18);
    expect(scenes[11]!.atlasKey).not.toBe(scenes[12]!.atlasKey);

    // Act IV: Ball-tree U26 D=17 comparison (beat 18), refinement progression (19 -> 20), panorama crop-back (22)
    expect(scenes[18]!.u26Mismatch).not.toBeNull();
    expect(scenes[18]!.u26Mismatch?.hasMismatch).toBe(true);
    expect(scenes[18]!.u26Mismatch?.mismatchedStageCount).toBe(3);
    expect(scenes[19]!.atlasKey).not.toBe(scenes[20]!.atlasKey);
    const beat22Cropped = scenes[22]!.rgbPreviews.find((p) => p.kind === 'cropped');
    expect(beat22Cropped?.width).toBe(16);
    expect(beat22Cropped?.height).toBe(8);

    // Act V: Grounding <LOC_0..16> & U27 badge (beat 24), Read-Only OPD (beats 25–27)
    const beat24Grounding = scenes[24]!.rgbPreviews.find((p) => p.kind === 'grounding');
    expect(beat24Grounding?.groundingBoxes?.length).toBeGreaterThan(0);
    expect(
      scenes[24]!.provenanceBadges.some((b) => b.label.includes('U27')),
    ).toBe(true);
    for (const idx of [25, 26, 27]) {
      expect(scenes[idx]!.opdReadOnly?.readOnly).toBe(true);
      expect(scenes[idx]!.opdReadOnly?.hasGradients).toBe(false);
      expect(scenes[idx]!.opdReadOnly?.badgeLabel).toBe('READ-ONLY · NO GRADIENTS');
    }
  });

  it('supports start, goToBeat, nextBeat, prevBeat, goToAct, clamping at 0 and 27, and exit', () => {
    const tour = createUniGuidedTour(SHARED_BUNDLE);

    expect(tour.totalBeats).toBe(28);
    expect(tour.isActive()).toBe(false);
    expect(tour.getState().active).toBe(false);
    expect(tour.getState().beatIndex).toBe(0);

    const savedRestore: UniFreeExploreRestoreState = Object.freeze({
      viewMode: 'refinement',
      focus: Object.freeze({
        ...DEFAULT_UNI_SPATIAL_FOCUS_STATE,
        prefillMode: 'hf_faithful',
        jointLayerIndex: 33,
        selectedGridCell: 7,
      }),
      activeBranchId: 'branch-user-1',
      activeRecordId: 'rec-user-5',
    });

    // Start at beat 0 & clamp prevBeat at 0
    const s0 = tour.start(savedRestore, 0);
    expect(tour.isActive()).toBe(true);
    expect(s0.active).toBe(true);
    expect(s0.beatIndex).toBe(0);
    expect(s0.actIndex).toBe(0);
    expect(s0.currentAct.id).toBe('act1');
    expect(s0.activeRecordId).not.toBeNull();

    expect(tour.prevBeat().beatIndex).toBe(0);
    expect(tour.goToBeat(-10).beatIndex).toBe(0);

    // Step forward with nextBeat
    const s1 = tour.nextBeat();
    expect(s1.beatIndex).toBe(1);
    expect(s1.currentBeat.id).toBe(UNI_TOUR_BEATS[1]!.id);

    // Jump directly to each Act start index
    const expectedActStarts: readonly [UniTourActId, number][] = [
      ['act1', 0],
      ['act2', 5],
      ['act3', 10],
      ['act4', 16],
      ['act5', 23],
    ];
    for (const [actId, startIdx] of expectedActStarts) {
      const actState = tour.goToAct(actId);
      expect(actState.beatIndex).toBe(startIdx);
      expect(actState.currentAct.id).toBe(actId);
      expect(actState.currentBeat.beatInAct).toBe(1);
    }

    // Clamp at 27
    const sLast = tour.goToBeat(999);
    expect(sLast.beatIndex).toBe(27);
    expect(sLast.actIndex).toBe(4);
    expect(tour.nextBeat().beatIndex).toBe(27);

    // Step backward from 27
    expect(tour.prevBeat().beatIndex).toBe(26);

    // Invalid actId throws
    expect(() => tour.goToAct('act99' as UniTourActId)).toThrow(RangeError);

    // Exit returns saved Free-Explore state untouched & clears restore slot
    const restored = tour.exit();
    expect(tour.isActive()).toBe(false);
    expect(tour.getState().active).toBe(false);
    expect(restored).toBe(savedRestore);
    expect(restored).toEqual(savedRestore);
    expect(tour.exit()).toBeNull();
  });

  it('leaves an external Free-Explore UniSession byte-for-byte untouched across all 28 tour beats', () => {
    // Build a populated Free-Explore UniSession with two active task histories
    const freeExploreSession = new UniSession(SHARED_BUNDLE);
    const editBranch = freeExploreSession.startBranch({
      taskMode: 'image_edit',
      label: 'User Free-Explore Edit',
      scenePresetId: 'harbor_panorama_16x8',
      prefillMode: 'hf_faithful',
      activationSteps: 4,
      refinementSteps: 4,
      seed: 0x12345678,
    });
    freeExploreSession.stepBranch(editBranch.branchId);
    freeExploreSession.stepBranch(editBranch.branchId);

    const vlmBranch = freeExploreSession.startBranch({
      taskMode: 'visual_understanding',
      label: 'User Free-Explore VLM',
      scenePresetId: 'sunlit_cat_16x16',
      maxNewTokens: 4,
      seed: 0x87654321,
    });
    freeExploreSession.stepBranch(vlmBranch.branchId);

    // Park Free-Explore cursor on a historical edit record
    const targetUserRecordId = freeExploreSession.getBranch(editBranch.branchId)!.recordIds[1]!;
    freeExploreSession.setActiveRecord(targetUserRecordId);

    const beforeDigest = captureSessionByteDigest(freeExploreSession);

    const restoreState: UniFreeExploreRestoreState = Object.freeze({
      viewMode: 'joint_attn',
      focus: Object.freeze({
        ...DEFAULT_UNI_SPATIAL_FOCUS_STATE,
        jointLayerIndex: 18,
        batchRow: 2,
        prefillMode: 'hf_faithful',
        recordId: targetUserRecordId,
      }),
      activeBranchId: freeExploreSession.getActiveBranchId(),
      activeRecordId: freeExploreSession.getActiveRecordId(),
    });

    // Enter tour, traverse every beat 0..27, build every beat scene, and exit
    const tour = createUniGuidedTour(SHARED_BUNDLE);
    expect(tour.tourSession).not.toBe(freeExploreSession);

    tour.start(restoreState, 0);
    for (let i = 0; i < UNI_TOUR_TOTAL_BEATS; i++) {
      const state = i === 0 ? tour.getState() : tour.nextBeat();
      expect(state.beatIndex).toBe(i);
      const scene = tour.buildBeatScene();
      expect(scene.viewMode).toBe(UNI_TOUR_BEATS[i]!.viewMode);
    }
    const returnedRestore = tour.exit();

    // Restore state is returned completely untouched
    expect(returnedRestore).toEqual(restoreState);

    // External Free-Explore UniSession is byte-for-byte identical
    const afterDigest = captureSessionByteDigest(freeExploreSession);
    expect(afterDigest).toEqual(beforeDigest);
  });

  it('enforces hard guardrails: zero Math.random(), zero src/ui/controls.ts, src/twotower/, or src/diffusion/ imports', () => {
    const src = readFileSync(
      resolve(__dirname, '../src/uni/story/walkthrough.ts'),
      'utf8',
    );
    expect(src).not.toMatch(/Math\.random\s*\(/);
    expect(src).not.toMatch(/ui\/controls/);
    expect(src).not.toMatch(/twotower/);
    expect(src).not.toMatch(/['"][^'"]*\/diffusion\//);
  });
});
