/**
 * UNI3 Slice 3C Verification Suite (`tests/uni-session-stress.test.ts`):
 *   1. Worst-case 12-Branch / 128-Record 3-Branch Image Editing Stress Test:
 *      - Strict `<= 12.00 MiB` (`12,582,912 B`) retained-memory cap via `Set<ArrayBufferLike>`
 *      - Branch eviction (`<= 12`) & record eviction (`<= 128`)
 *      - `nonActiveRecordsWithFullLogits === 0`
 *      - Wall-clock runtime strictly `< 2,000 ms`
 *   2. Immutable `prefixCache` slab sharing on `forkBranch` + live `D5` toggle
 *      (`hf_faithful` <-> `elastic_pruned`) with 0-ULP output equivalence
 *   3. `0`-ULP Lazy Reconstruction (`reconstructLayerTrace` across joint `L0`/`L17`,
 *      split `L18`/`L33`, and non-eager intermediate `L7`/`L25`, plus
 *      `reconstructStepHeadLogits` on both T2I `B=2` and Editing `B=3`)
 *   4. Active-record-only eager inspection eviction (`[0, 17, 18, 33]`) and all 5
 *      interactive task modes (`text_understanding`, `visual_understanding`,
 *      `object_grounding`, `text_to_image`, `image_edit`)
 *   5. Headless barrel (`src/uni/index.ts`), zero `Math.random()`, and zero
 *      forbidden sibling-app imports.
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { assertBitExactFiniteArray } from './helpers/parityAssert';
import * as uni from '../src/uni/index';
import {
  UNI_MAX_SESSION_RETAINED_BYTES,
  UNI_MINI_CONFIG,
  UNI_ROUTE_G,
  UNI_SESSION_EAGER_TRACE_LAYERS,
  UNI_SESSION_MAX_BRANCHES,
  UNI_SESSION_MAX_RECORDS,
  buildSemanticTreeEmbeddings,
  compressPackedImageEmbeddings,
  createDefaultUniRuntimeBundle,
  createUniSession,
  embedFixedPrefixTokens,
  extractRowPastKvs,
  measureSessionRetainedBytes,
  runSingleImageDenoiseRound,
  runUniForwardPass,
  type UniEditGuidanceMode,
  type UniElasticPrefillMode,
  type UniScenePresetId,
  type UniSemanticTreeState,
} from '../src/uni/index';

function assertBitExactScoresIncludingMasks(
  actual: Float32Array,
  expected: Float32Array,
  label: string,
): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    const a = actual[i];
    const b = expected[i];
    if (b === -Infinity) {
      if (a !== -Infinity) {
        throw new Error(`Expected -Infinity mask at ${label}[${i}], got ${String(a)}`);
      }
      continue;
    }
    if (!Number.isFinite(a) || !Number.isFinite(b) || a !== b) {
      throw new Error(
        `Bit-exact score mismatch at ${label}[${i}]: actual=${String(a)}, expected=${String(b)}`,
      );
    }
  }
}

function buildImageDenoiseWindowEmbeds(
  bundle: uni.UniFullRuntimeBundle,
  treeState: UniSemanticTreeState,
  gainPresetId: uni.UniGainPresetId,
): Float32Array {
  const d = bundle.core.config.hiddenSize;
  const fixedEmbeds = embedFixedPrefixTokens(bundle.core.genEmbedding, d);
  const childEmbeds = buildSemanticTreeEmbeddings(
    treeState.representatives,
    treeState.noiseWeights,
    bundle.core.genEmbedding,
    d,
  );
  const compressed = compressPackedImageEmbeddings(
    childEmbeds,
    UNI_MINI_CONFIG.vqGridSize,
    UNI_MINI_CONFIG.vqMarkedGridCols,
    bundle.codec,
    { isSourceEdit: false, gainPresetId },
  );
  const windowEmbeds = new Float32Array(fixedEmbeds.length + compressed.output.length);
  windowEmbeds.set(fixedEmbeds, 0);
  windowEmbeds.set(compressed.output, fixedEmbeds.length);
  return windowEmbeds;
}

describe('UNI3 Slice 3C — Bounded 12 MiB UniSession, 0-ULP Lazy Reconstruction & Headless Barrel', () => {
  const sharedBundle = createDefaultUniRuntimeBundle();

  it('survives worst-case 12-branch / 128-record 3-branch Image Editing stress under 12.00 MiB and 2,000 ms', () => {
    const t0 = performance.now();
    const session = createUniSession(sharedBundle);

    const scenes: readonly UniScenePresetId[] = [
      'harbor_panorama_16x8',
      'sunlit_cat_16x16',
      'geometric_blocks_16x16',
    ];
    const guidanceModes: readonly UniEditGuidanceMode[] = ['legacy', 'independent', 'ip2p'];
    const prefillModes: readonly UniElasticPrefillMode[] = ['hf_faithful', 'elastic_pruned'];

    // Create 13 total 3-branch Image Editing branches (1 oldest branch evicted so
    // branchCount caps strictly at 12), and advance >= 128 steps across the 12
    // surviving branches so BOTH branch eviction (<= 12) and record eviction (<= 128)
    // trigger simultaneously at full 12-branch / 128-record worst-case saturation.
    let totalStepInvocations = 0;

    // Branch 0 (will be evicted once branch 12 is created)
    const evictedSeedBranch = session.startBranch({
      taskMode: 'image_edit',
      label: 'edit-evicted-0',
      scenePresetId: 'sunlit_cat_16x16',
      prefillMode: 'hf_faithful',
      guidanceMode: 'legacy',
      guidanceScaleText: 1.6,
      guidanceScaleImage: 2.0,
      activationSteps: 8,
      refinementSteps: 4,
      seed: 900,
    });
    expect(session.stepBranch(evictedSeedBranch.branchId)).not.toBeNull();
    totalStepInvocations += 1;

    // 7 surviving root branches (8 * 11 = 88 records incl. 7 prefills + 77 steps)
    for (let i = 0; i < 7; i++) {
      const branch = session.startBranch({
        taskMode: 'image_edit',
        label: `edit-root-${i}`,
        scenePresetId: scenes[i % scenes.length],
        prefillMode: prefillModes[i % prefillModes.length],
        guidanceMode: guidanceModes[i % guidanceModes.length],
        guidanceScaleText: 1.6,
        guidanceScaleImage: 2.0,
        activationSteps: 8,
        refinementSteps: 4,
        seed: 1000 + i,
      });
      // Advance all 11 denoise rounds (8 activation + 3 flush = 11 steps + 1 prefill = 12 records/branch)
      for (let s = 0; s < 11; s++) {
        const rec = session.stepBranch(branch.branchId);
        expect(rec).not.toBeNull();
        totalStepInvocations += 1;
      }
    }

    // 5 surviving forked branches (7 + 5 = 12 live branches, 13 created total -> branch 0 evicted!)
    // Each forks from prefill or round 0 and runs 10-11 rounds -> 52+ more records -> total > 136 -> capped at 128!
    for (let f = 0; f < 5; f++) {
      const allBranches = session.getAllBranches();
      const sourceBranch = allBranches[allBranches.length - 1 - (f % 4)];
      const forkFromId = sourceBranch.recordIds[0];

      const forked = session.forkBranch(forkFromId, {
        label: `edit-fork-${f}`,
        // Alternate shared-prefix forks and live D5 prefillMode switches
        prefillMode:
          f % 2 === 0
            ? sourceBranch.prefillMode
            : sourceBranch.prefillMode === 'hf_faithful'
              ? 'elastic_pruned'
              : 'hf_faithful',
        guidanceMode: guidanceModes[(f + 1) % guidanceModes.length],
        seed: 5000 + f,
      });

      for (let s = 0; s < 11; s++) {
        const rec = session.stepBranch(forked.branchId);
        if (rec === null) break;
        totalStepInvocations += 1;
      }
    }

    // Confirm branch 0 was evicted cleanly
    expect(session.getBranch(evictedSeedBranch.branchId)).toBeUndefined();

    const elapsedMs = performance.now() - t0;

    expect(totalStepInvocations).toBeGreaterThanOrEqual(128);
    expect(elapsedMs).toBeLessThan(2_000);

    const audit = session.auditMemory();
    const standaloneAudit = measureSessionRetainedBytes(session);
    expect(standaloneAudit).toEqual(audit);

    expect(audit.maxBranches).toBe(UNI_SESSION_MAX_BRANCHES);
    expect(audit.maxRecords).toBe(UNI_SESSION_MAX_RECORDS);
    expect(audit.maxAllowedBytes).toBe(UNI_MAX_SESSION_RETAINED_BYTES);
    expect(audit.branchCount).toBe(UNI_SESSION_MAX_BRANCHES);
    expect(audit.recordCount).toBe(UNI_SESSION_MAX_RECORDS);
    expect(audit.nonActiveRecordsWithFullLogits).toBe(0);
    expect(audit.uniqueBufferCount).toBeGreaterThan(0);
    expect(audit.withinBudget).toBe(true);
    expect(audit.totalRetainedBytes).toBeLessThanOrEqual(UNI_MAX_SESSION_RETAINED_BYTES);

    // Every remaining branch must still have valid prefixCache and records
    const snap = session.getSnapshot();
    expect(snap.branches.length).toBe(UNI_SESSION_MAX_BRANCHES);
    expect(snap.activeRecordId).not.toBeNull();
    expect(snap.activeInspection?.recordId).toBe(snap.activeRecordId);
    expect(snap.memoryAudit.withinBudget).toBe(true);
  });

  it('shares immutable prefixCache slabs on forkBranch and preserves 0-ULP equivalence across live D5 prefillMode toggle', () => {
    const session = createUniSession(sharedBundle);

    const base = session.startBranch({
      taskMode: 'image_edit',
      label: 'base-hf-faithful',
      scenePresetId: 'harbor_panorama_16x8',
      prefillMode: 'hf_faithful',
      guidanceMode: 'legacy',
      guidanceScaleText: 1.6,
      guidanceScaleImage: 2.0,
      activationSteps: 8,
      refinementSteps: 4,
      seed: 777,
    });
    const rec0 = session.stepBranch(base.branchId)!;
    expect(rec0.kind).toBe('image_denoise');
    expect(base.prefixCache.wasteTelemetry.wastedSplitUBytes).toBeGreaterThan(0);

    // Fork with same prefillMode -> exact shared prefixCache reference (zero slab duplication)
    const beforeSharedAudit = session.auditMemory();
    const sharedFork = session.forkBranch(rec0.recordId, {
      label: 'shared-slab-fork',
      seed: 999,
    });
    expect(sharedFork.prefixCache).toBe(base.prefixCache);
    expect(sharedFork.prefixCache.slabs).toBe(base.prefixCache.slabs);
    const afterSharedAudit = session.auditMemory();
    // Adding a shared-prefix fork should not allocate any new KV slabs (< 4 KiB overhead)
    expect(afterSharedAudit.totalRetainedBytes - beforeSharedAudit.totalRetainedBytes).toBeLessThan(
      4_096,
    );

    // Fork with live D5 toggle from hf_faithful -> elastic_pruned with SAME seed (999)
    const prunedFork = session.forkBranch(rec0.recordId, {
      label: 'd5-elastic-pruned-fork',
      prefillMode: 'elastic_pruned',
      seed: 999,
    });
    expect(prunedFork.prefillMode).toBe('elastic_pruned');
    expect(prunedFork.prefixCache.wasteTelemetry.wastedSplitUBytes).toBe(0);

    const sharedNext = session.stepBranch(sharedFork.branchId)!;
    const prunedNext = session.stepBranch(prunedFork.branchId)!;

    // 0-ULP equivalence between hf_faithful and elastic_pruned downstream G denoising!
    expect(Array.from(prunedNext.imageState!.representatives)).toEqual(
      Array.from(sharedNext.imageState!.representatives),
    );
    expect(Array.from(prunedNext.imageState!.proposals)).toEqual(
      Array.from(sharedNext.imageState!.proposals),
    );

    const sharedHead = session.reconstructStepHeadLogits(sharedNext.recordId);
    const prunedHead = session.reconstructStepHeadLogits(prunedNext.recordId);
    assertBitExactFiniteArray(
      prunedHead.guidedGenLogits!,
      sharedHead.guidedGenLogits!,
      'd5.fork.guidedGenLogits',
    );
  });

  it('reconstructs layer traces at 0 ULP across joint (L0, L17), split (L18, L33), and non-eager (L7, L25) layers on active and historical records', () => {
    const session = createUniSession(sharedBundle);

    // 1. Image Editing (B=3, harbor_panorama_16x8, elastic_pruned)
    const editBranch = session.startBranch({
      taskMode: 'image_edit',
      scenePresetId: 'harbor_panorama_16x8',
      prefillMode: 'elastic_pruned',
      guidanceMode: 'legacy',
      activationSteps: 8,
      refinementSteps: 4,
      seed: 2026,
    });
    const prefillRecId = editBranch.recordIds[0];
    const round0 = session.stepBranch(editBranch.branchId)!;
    const round1 = session.stepBranch(editBranch.branchId)!;
    const round2 = session.stepBranch(editBranch.branchId)!;

    // Active record is currently round2; verify BOTH non-active historical round1 AND active round2!
    expect(session.getActiveRecordId()).toBe(round2.recordId);

    const targetLayers = [0, 7, 17, 18, 25, 33] as const;

    for (const [label, prevRec, curRec] of [
      ['historical-round1', round0, round1],
      ['active-round2', round1, round2],
    ] as const) {
      const prevTreeState: UniSemanticTreeState = {
        stages: prevRec.imageState!.stages,
        representatives: prevRec.imageState!.representatives,
        levels: prevRec.imageState!.levels,
        noiseWeights: prevRec.imageState!.noiseWeights,
        isMarker: new Uint8Array([
          0, 0, 0, 0, 1, 1,
          0, 0, 0, 0, 1, 1,
          0, 0, 0, 0, 1, 1,
          0, 0, 0, 0, 1, 1,
        ]),
        refinementSteps: editBranch.refinementSteps,
      };
      const windowEmbeds = buildImageDenoiseWindowEmbeds(
        sharedBundle,
        prevTreeState,
        editBranch.gainPresetId,
      );
      const allG = new Uint8Array(10).fill(UNI_ROUTE_G);

      for (let b = 0; b < editBranch.cfgBatchSize; b++) {
        const { pastKvs, pastRouting } = extractRowPastKvs(editBranch.prefixCache, b);
        const liveForward = runUniForwardPass(sharedBundle.core, {
          inputsEmbeds: windowEmbeds,
          routing: allG,
          maskMode: 'bidirectional',
          useCache: false,
          pastKvs,
          pastRouting,
          positionIds: curRec.positionIds,
          cachePositions: curRec.positionIds,
          computeTextHead: false,
          computeGenHead: false,
        });

        for (const layerIdx of targetLayers) {
          const liveTrace = liveForward.layerTraces.get(layerIdx)!;
          const reconTrace = session.reconstructLayerTrace(curRec.recordId, layerIdx, b);

          const tag = `${label}.b${b}.L${layerIdx}`;
          assertBitExactFiniteArray(reconTrace.inputNorm, liveTrace.inputNorm, `${tag}.inputNorm`);
          assertBitExactFiniteArray(
            reconTrace.attention.qRot,
            liveTrace.attention.qRot,
            `${tag}.qRot`,
          );
          assertBitExactFiniteArray(
            reconTrace.attention.kRot,
            liveTrace.attention.kRot,
            `${tag}.kRot`,
          );
          assertBitExactScoresIncludingMasks(
            reconTrace.attention.scores,
            liveTrace.attention.scores,
            `${tag}.scores`,
          );
          assertBitExactFiniteArray(
            reconTrace.attention.probs,
            liveTrace.attention.probs,
            `${tag}.probs`,
          );
          assertBitExactFiniteArray(
            reconTrace.attention.output,
            liveTrace.attention.output,
            `${tag}.attnOutput`,
          );
          assertBitExactFiniteArray(
            reconTrace.swiglu.output,
            liveTrace.swiglu.output,
            `${tag}.swigluOutput`,
          );
          assertBitExactFiniteArray(reconTrace.output, liveTrace.output, `${tag}.output`);
        }
      }
    }

    // Verify prefill record reconstruction:
    // Row 0 (uncond, u_only) under elastic_pruned executed L0..L17 and skipped L18..L33
    const prefillL0 = session.reconstructLayerTrace(prefillRecId, 0, 0);
    expect(prefillL0.layerIndex).toBe(0);
    expect(() => session.reconstructLayerTrace(prefillRecId, 18, 0)).toThrow(RangeError);
    // Row 2 (full, mixed U+G) executed all L0..L33 with U pruned at split layers
    const prefillRow2L18 = session.reconstructLayerTrace(prefillRecId, 18, 2);
    expect(prefillRow2L18.understandingPruned).toBe(true);
    const prefillRow2L33 = session.reconstructLayerTrace(prefillRecId, 33, 2);
    expect(prefillRow2L33.understandingPruned).toBe(true);
  });

  it('reconstructs head logits at 0 ULP matching live runSingleImageDenoiseRound(..., retainFullLogits: true) on T2I (B=2) and Editing (B=3)', () => {
    const session = createUniSession(sharedBundle);

    // Case A: T2I (B = 2)
    const t2iBranch = session.startBranch({
      taskMode: 'text_to_image',
      prefillMode: 'hf_faithful',
      guidanceScaleText: 3.0,
      activationSteps: 8,
      refinementSteps: 4,
      seed: 31415,
    });
    const t2iRound0 = session.stepBranch(t2iBranch.branchId)!;
    const t2iRound1 = session.stepBranch(t2iBranch.branchId)!;

    const t2iPrevState: UniSemanticTreeState = {
      stages: t2iRound0.imageState!.stages,
      representatives: t2iRound0.imageState!.representatives,
      levels: t2iRound0.imageState!.levels,
      noiseWeights: t2iRound0.imageState!.noiseWeights,
      isMarker: new Uint8Array([
        0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 1, 1,
      ]),
      refinementSteps: t2iBranch.refinementSteps,
    };
    const dummyRng = uni.createXoshiro128ss(1);
    const liveT2i = runSingleImageDenoiseRound(sharedBundle, {
      prefixCache: t2iBranch.prefixCache,
      treeState: t2iPrevState,
      schedule: t2iBranch.schedule!,
      unmaskOrder: t2iBranch.unmaskOrder!,
      roundIndex: 1,
      taskKind: 'text_to_image',
      guidanceScaleText: t2iBranch.guidanceScaleText,
      guidanceScaleImage: 0,
      temperature: 0.8,
      topK: 0,
      topP: 1.0,
      gainPresetId: t2iBranch.gainPresetId,
      nextUniformOpenUnit: () => dummyRng.nextUniformOpenUnit(),
      nextRandBits: () => dummyRng.nextUint32(),
      retainFullLogits: true,
    });

    const reconT2i = session.reconstructStepHeadLogits(t2iRound1.recordId);
    for (let b = 0; b < 2; b++) {
      assertBitExactFiniteArray(
        reconT2i.perBranchFinalNorm[b]!,
        liveT2i.stepRecord.perBranchFinalNorm[b],
        `t2i.b${b}.finalNorm`,
      );
      assertBitExactFiniteArray(
        reconT2i.perBranchGenLogits[b]!,
        liveT2i.stepRecord.perBranchFullLogits![b],
        `t2i.b${b}.genLogits`,
      );
    }
    assertBitExactFiniteArray(
      reconT2i.guidedGenLogits!,
      liveT2i.stepRecord.guidedLogits!,
      't2i.guidedGenLogits',
    );

    // Case B: Image Editing (B = 3) across all 3 Dual-CFG guidance modes (`legacy`, `independent`, `ip2p`)
    for (const guidanceMode of ['legacy', 'independent', 'ip2p'] as const) {
      const editBranch = session.startBranch({
        taskMode: 'image_edit',
        scenePresetId: 'harbor_panorama_16x8',
        prefillMode: 'elastic_pruned',
        guidanceMode,
        guidanceScaleText: 1.6,
        guidanceScaleImage: 2.0,
        activationSteps: 8,
        refinementSteps: 4,
        seed: 27182,
      });
      const editRound0 = session.stepBranch(editBranch.branchId)!;
      const editRound1 = session.stepBranch(editBranch.branchId)!;

      const editPrevState: UniSemanticTreeState = {
        stages: editRound0.imageState!.stages,
        representatives: editRound0.imageState!.representatives,
        levels: editRound0.imageState!.levels,
        noiseWeights: editRound0.imageState!.noiseWeights,
        isMarker: new Uint8Array([
          0, 0, 0, 0, 1, 1,
          0, 0, 0, 0, 1, 1,
          0, 0, 0, 0, 1, 1,
          0, 0, 0, 0, 1, 1,
        ]),
        refinementSteps: editBranch.refinementSteps,
      };
      const liveEdit = runSingleImageDenoiseRound(sharedBundle, {
        prefixCache: editBranch.prefixCache,
        treeState: editPrevState,
        schedule: editBranch.schedule!,
        unmaskOrder: editBranch.unmaskOrder!,
        roundIndex: 1,
        taskKind: 'image_edit',
        guidanceMode,
        guidanceScaleText: editBranch.guidanceScaleText,
        guidanceScaleImage: editBranch.guidanceScaleImage,
        temperature: 0.8,
        topK: 0,
        topP: 1.0,
        gainPresetId: editBranch.gainPresetId,
        sourcePaddingMask: editBranch.sourcePaddingMask,
        sourceVqCodes: editBranch.sourceVqCodes,
        nextUniformOpenUnit: () => dummyRng.nextUniformOpenUnit(),
        nextRandBits: () => dummyRng.nextUint32(),
        retainFullLogits: true,
      });

      const reconEdit = session.reconstructStepHeadLogits(editRound1.recordId);
      for (let b = 0; b < 3; b++) {
        assertBitExactFiniteArray(
          reconEdit.perBranchFinalNorm[b]!,
          liveEdit.stepRecord.perBranchFinalNorm[b],
          `edit.${guidanceMode}.b${b}.finalNorm`,
        );
        assertBitExactFiniteArray(
          reconEdit.perBranchGenLogits[b]!,
          liveEdit.stepRecord.perBranchFullLogits![b],
          `edit.${guidanceMode}.b${b}.genLogits`,
        );
      }
      assertBitExactFiniteArray(
        reconEdit.guidedGenLogits!,
        liveEdit.stepRecord.guidedLogits!,
        `edit.${guidanceMode}.guidedGenLogits`,
      );
    }
  });

  it('supports all 5 interactive task modes, active-record inspection eviction, and lazy image decoding', () => {
    const session = createUniSession(sharedBundle);

    // 1. Text Understanding
    const textBranch = session.startBranch({
      taskMode: 'text_understanding',
      maxNewTokens: 4,
      blockLength: 4,
      temperature: 0,
    });
    const textSteps = session.runBranchToCompletion(textBranch.branchId);
    expect(textSteps.length).toBeGreaterThanOrEqual(1);
    expect(textSteps[0].kind).toBe('text_block_step');
    const textHead = session.reconstructStepHeadLogits(textSteps[0].recordId);
    expect(textHead.perBranchTextLogits[0]).not.toBeNull();
    expect(textHead.guidedGenLogits).toBeNull();
    const textL18 = session.reconstructLayerTrace(textSteps[0].recordId, 18, 0);
    expect(textL18.layerIndex).toBe(18);

    // 2. Visual Understanding
    const visBranch = session.startBranch({
      taskMode: 'visual_understanding',
      scenePresetId: 'sunlit_cat_16x16',
      maxNewTokens: 4,
      blockLength: 4,
      temperature: 0,
    });
    const visSteps = session.runBranchToCompletion(visBranch.branchId);
    expect(visSteps.length).toBeGreaterThanOrEqual(1);
    expect(session.reconstructLayerTrace(visSteps[0].recordId, 17, 0).layerIndex).toBe(17);

    // 3. Object Grounding
    const groundBranch = session.startBranch({
      taskMode: 'object_grounding',
      scenePresetId: 'sunlit_cat_16x16',
      groundingTags: ['cat'],
      groundingSteps: 2,
    });
    const groundSteps = session.runBranchToCompletion(groundBranch.branchId);
    expect(groundSteps.length).toBe(2);
    expect(groundSteps[0].kind).toBe('grounding_step');
    const groundHead = session.reconstructStepHeadLogits(groundSteps[1].recordId);
    expect(groundHead.perBranchTextLogits[0]).not.toBeNull();

    // 4. Text-to-Image (including U30 guidanceScale=0 B=1 short-circuit)
    const t2iBranch = session.startBranch({
      taskMode: 'text_to_image',
      guidanceScaleText: 0,
      activationSteps: 4,
      refinementSteps: 2,
      seed: 42,
    });
    expect(t2iBranch.cfgBatchSize).toBe(1);
    const t2iSteps = session.runBranchToCompletion(t2iBranch.branchId);
    expect(t2iSteps.length).toBe(5); // 4 activation + (2 - 1) flush = 5 rounds
    const decodedT2i = session.decodeBranchImage(t2iBranch.branchId);
    expect(decodedT2i.outputImage.width).toBe(16);
    expect(decodedT2i.outputImage.height).toBe(16);

    // 5. Image Editing with cropToSource on non-square 16x8 panorama
    const editBranch = session.startBranch({
      taskMode: 'image_edit',
      scenePresetId: 'harbor_panorama_16x8',
      activationSteps: 4,
      refinementSteps: 1,
      cropToSource: true,
      seed: 99,
    });
    const editSteps = session.runBranchToCompletion(editBranch.branchId);
    expect(editSteps.length).toBe(4);
    const decodedEdit = session.decodeBranchImage(editBranch.branchId);
    expect(decodedEdit.decodedPaddedImage.width).toBe(16);
    expect(decodedEdit.decodedPaddedImage.height).toBe(16);
    expect(decodedEdit.outputImage.width).toBe(16);
    expect(decodedEdit.outputImage.height).toBe(8);

    // Active inspection switches cleanly and populates ONLY [0, 17, 18, 33]
    const inspectA = session.setActiveRecord(t2iSteps[0].recordId);
    expect(inspectA.recordId).toBe(t2iSteps[0].recordId);
    expect(Array.from(inspectA.eagerTracesByBranch[0].keys())).toEqual([
      ...UNI_SESSION_EAGER_TRACE_LAYERS,
    ]);
    expect(inspectA.guidedGenLogits).not.toBeNull();

    const inspectB = session.setActiveRecord(editSteps[editSteps.length - 1].recordId);
    expect(inspectB.recordId).toBe(editSteps[editSteps.length - 1].recordId);
    expect(session.getActiveInspection()).toBe(inspectB);
    expect(inspectB.eagerTracesByBranch.length).toBe(3);
    for (let b = 0; b < 3; b++) {
      expect(Array.from(inspectB.eagerTracesByBranch[b].keys())).toEqual([
        ...UNI_SESSION_EAGER_TRACE_LAYERS,
      ]);
    }

    const audit = session.auditMemory();
    expect(audit.nonActiveRecordsWithFullLogits).toBe(0);
    expect(audit.withinBudget).toBe(true);
  });

  it('reconstructs exact mid-trajectory fork state for text_understanding and object_grounding without leaking later parent steps, and tallies causalContext pre-seed guard interventions', () => {
    // 1. Mid-trajectory fork on text_understanding (run 2 full blocks = 8 tokens, then fork from
    //    prefill AND from block-0's commit step; verify zero leakage from parent's block-1!)
    const session = createUniSession(sharedBundle);
    const parentText = session.startBranch({
      taskMode: 'text_understanding',
      label: 'parent-text',
      blockLength: 4,
      stepsPerBlock: 2,
      maxNewTokens: 8,
      causalContext: true,
      temperature: 0,
      seed: 4242,
    });
    const parentTextSteps = session.runBranchToCompletion(parentText.branchId, 20);
    expect(parentTextSteps.length).toBeGreaterThanOrEqual(4); // 2 steps/block * 2 blocks

    const prefillId = session.getBranch(parentText.branchId)!.recordIds[0];
    // Fork from prefill AFTER parent already completed both blocks (parentInternal has block-1 tail!)
    const forkFromPrefill = session.forkBranch(prefillId, { label: 'fork-text-prefill', seed: 4242 });
    const forkPrefillSteps = session.runBranchToCompletion(forkFromPrefill.branchId, 20);
    expect(forkPrefillSteps.length).toBe(parentTextSteps.length);
    for (let i = 0; i < parentTextSteps.length; i++) {
      assertBitExactFiniteArray(
        forkPrefillSteps[i].textState!.tokenIds,
        parentTextSteps[i].textState!.tokenIds,
        `fork_text_prefill.step[${i}].tokens`,
      );
    }

    // Fork from end of block 0 (step index 1, after parent already ran block 1!)
    const block0EndRec = parentTextSteps[1];
    const forkFromBlock0 = session.forkBranch(block0EndRec.recordId, {
      label: 'fork-text-block0',
      seed: 4242,
    });
    const forkBlock0Steps = session.runBranchToCompletion(forkFromBlock0.branchId, 20);
    expect(forkBlock0Steps.length).toBe(parentTextSteps.length - 2);
    for (let i = 0; i < forkBlock0Steps.length; i++) {
      assertBitExactFiniteArray(
        forkBlock0Steps[i].textState!.tokenIds,
        parentTextSteps[i + 2].textState!.tokenIds,
        `fork_text_block0.step[${i}].tokens`,
      );
    }

    // 2. Mid-trajectory fork on object_grounding (2-step grounding: fork from prefill AND from step 1
    //    after parent already finished step 2; verify zero leakage from parent's final suffix!)
    const parentGround = session.startBranch({
      taskMode: 'object_grounding',
      label: 'parent-ground',
      scenePresetId: 'harbor_panorama_16x8',
      groundingTags: ['house', 'water'],
      groundingSteps: 2,
      enforceLocDomainGuard: true,
      seed: 777,
    });
    const groundSteps = session.runBranchToCompletion(parentGround.branchId, 8);
    expect(groundSteps.length).toBe(2);

    const groundPrefillId = session.getBranch(parentGround.branchId)!.recordIds[0];
    const forkGroundPrefill = session.forkBranch(groundPrefillId, { label: 'fork-ground-prefill' });
    const forkGroundPrefillSteps = session.runBranchToCompletion(forkGroundPrefill.branchId, 8);
    expect(forkGroundPrefillSteps.length).toBe(2);
    assertBitExactFiniteArray(
      forkGroundPrefillSteps[0].textState!.tokenIds,
      groundSteps[0].textState!.tokenIds,
      'fork_ground_prefill.step0',
    );
    assertBitExactFiniteArray(
      forkGroundPrefillSteps[1].textState!.tokenIds,
      groundSteps[1].textState!.tokenIds,
      'fork_ground_prefill.step1',
    );

    const forkGroundMid = session.forkBranch(groundSteps[0].recordId, { label: 'fork-ground-mid' });
    const forkGroundMidSteps = session.runBranchToCompletion(forkGroundMid.branchId, 8);
    expect(forkGroundMidSteps.length).toBe(1);
    assertBitExactFiniteArray(
      forkGroundMidSteps[0].textState!.tokenIds,
      groundSteps[1].textState!.tokenIds,
      'fork_ground_mid.step1',
    );

    // 3. Preserve causalContext slot-0 pre-seed U21 guardInterventions on both B=1 (AR) and B=4
    const d = sharedBundle.core.config.hiddenSize;
    const probePrompt = [1, 10, 26, 27];
    const probeFwd = runUniForwardPass(sharedBundle.core, {
      tokenIds: probePrompt,
      maskMode: 'causal',
      useCache: false,
      downstreamTarget: 'u',
    });
    const lastNorm = probeFwd.finalNorm!.subarray(3 * d, 4 * d);
    const tweakedHead = new Float32Array(sharedBundle.core.diffusionHead.length);
    for (let c = 0; c < d; c++) {
      tweakedHead[25 * d + c] = lastNorm[c] * 25.0; // [MASK] raw top-1
      tweakedHead[6 * d + c] = lastNorm[c] * 5.0;   // 'cat' fallback after U21 suppression
    }
    const guardBundle: uni.UniFullRuntimeBundle = {
      ...sharedBundle,
      core: { ...sharedBundle.core, diffusionHead: tweakedHead },
    };
    const guardSession = createUniSession(guardBundle);

    const arBranch = guardSession.startBranch({
      taskMode: 'text_understanding',
      promptTokenIds: probePrompt,
      blockLength: 1,
      stepsPerBlock: 1,
      maxNewTokens: 1,
      causalContext: true,
      suppressMaskLogit: true,
      temperature: 0,
    });
    const arRec = guardSession.stepBranch(arBranch.branchId);
    expect(arRec).not.toBeNull();
    expect(arRec!.textState!.tokenIds[0]).toBe(6);
    expect(arRec!.textState!.guardInterventions).toBeGreaterThanOrEqual(1);

    const blockBranch = guardSession.startBranch({
      taskMode: 'text_understanding',
      promptTokenIds: probePrompt,
      blockLength: 4,
      stepsPerBlock: 2,
      maxNewTokens: 4,
      causalContext: true,
      suppressMaskLogit: true,
      temperature: 0,
    });
    const blockRec0 = guardSession.stepBranch(blockBranch.branchId);
    expect(blockRec0).not.toBeNull();
    expect(blockRec0!.textState!.tokenIds[0]).toBe(6);
    expect(blockRec0!.textState!.guardInterventions).toBeGreaterThanOrEqual(1);
  });

  it('exports all headless modules cleanly from src/uni/index.ts with zero Math.random() or forbidden imports', () => {
    // Verify key exports across all 19 UNI modules via the headless barrel
    expect(typeof uni.verifyUniLfsReconciliation).toBe('function');
    expect(typeof uni.createXoshiro128ss).toBe('function');
    expect(typeof uni.computeUniYarnInvFreq).toBe('function');
    expect(typeof uni.buildUniAttentionMaskMatrix).toBe('function');
    expect(typeof uni.runUniElasticAttention).toBe('function');
    expect(typeof uni.runUniRoutedSwiglu).toBe('function');
    expect(typeof uni.createDefaultUniCoreModel).toBe('function');
    expect(typeof uni.runUniForwardPass).toBe('function');
    expect(typeof uni.runMiniVisionPipeline).toBe('function');
    expect(typeof uni.encodeMiniEmu3Vq).toBe('function');
    expect(typeof uni.compressPackedImageEmbeddings).toBe('function');
    expect(typeof uni.createEmptyUniKvCache).toBe('function');
    expect(typeof uni.generateWithPrefixCacheBlockDiff).toBe('function');
    expect(typeof uni.buildBalancedSemanticTree).toBe('function');
    expect(typeof uni.evaluateUnderstandingOpdRef).toBe('function');
    expect(typeof uni.buildUniViewArchReadouts).toBe('function');
    expect(typeof uni.createDefaultUniRuntimeBundle).toBe('function');
    expect(typeof uni.runT2iTrajectory).toBe('function');
    expect(typeof uni.runImageEditTrajectory).toBe('function');
    expect(typeof uni.createUniSession).toBe('function');
    expect(typeof uni.measureSessionRetainedBytes).toBe('function');

    // Static isolation & determinism check on our owned files
    const sessionSrc = readFileSync(resolve(process.cwd(), 'src/uni/session.ts'), 'utf8');
    const indexSrc = readFileSync(resolve(process.cwd(), 'src/uni/index.ts'), 'utf8');
    const combined = `${sessionSrc}\n${indexSrc}`;

    expect(combined).not.toMatch(/\bMath\.random\s*\(/);
    expect(combined).not.toMatch(/from\s+['"][^'"]*twotower/);
    expect(combined).not.toMatch(/from\s+['"][^'"]*diffusion/);
    expect(combined).not.toMatch(/from\s+['"][^'"]*ui\/controls/);
  });
});
