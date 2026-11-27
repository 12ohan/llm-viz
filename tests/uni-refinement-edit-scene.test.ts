import { describe, expect, it } from 'vitest';
import { pickMacroBay, type Ray } from '../src/render/pick';
import { localToWorldOnFrame } from '../src/render/sheetFrame';
import {
  SHEET_HL_ACTIVE,
  SHEET_HL_DIMMED,
  SHEET_INSTANCE_FLOATS,
  SHEET_KIND_CAUSAL_SCORES,
  SHEET_KIND_CAUSAL_SOFTMAX,
} from '../src/render/sheets';
import { runUniForwardPass } from '../src/uni/forward';
import { createDefaultUniRuntimeBundle } from '../src/uni/imageSampler';
import { cropUniKvCache, extractRowPastKvs } from '../src/uni/kvCache';
import {
  buildUniOutputEditScene,
  UNI_OUTPUT_EDIT_ATLAS_BUDGET_BYTES,
  UNI_OUTPUT_EDIT_ATLAS_SIZE,
} from '../src/uni/scene/outputEditScene';
import {
  pickHitToUniCell,
  UNI_SCENE_VISUAL_HONESTY_BADGE,
  validateSceneGeometryFinite,
} from '../src/uni/scene/provenance';
import {
  buildUniRefinementCanvasScene,
  UNI_REFINEMENT_ATLAS_BUDGET_BYTES,
  UNI_REFINEMENT_ATLAS_SIZE,
} from '../src/uni/scene/refinementCanvasScene';
import type { UniAtlasRegionBox, UniSpatialScene } from '../src/uni/scene/types';
import { UniSession } from '../src/uni/session';
import { UNI_MINI_LOC_END_ID, UNI_MINI_LOC_START_ID, UNI_MINI_MASK_TOKEN_ID } from '../src/uni/types';

const sharedBundle = createDefaultUniRuntimeBundle();

function createCompletedSession(
  options: Parameters<UniSession['startBranch']>[0],
): UniSession {
  const session = new UniSession(sharedBundle);
  const branch = session.startBranch(options);
  session.runBranchToCompletion(branch.branchId);
  return session;
}

function readRegionTexels(
  scene: UniSpatialScene,
  regionId: string,
): { region: UniAtlasRegionBox; rValues: Float32Array; gValues: Float32Array } {
  const atlas = scene.valueAtlas;
  expect(atlas).not.toBeNull();
  const region = scene.atlasRegions.find((r) => r.id === regionId);
  expect(region).toBeDefined();

  const w = atlas!.width;
  const rows = region!.rows;
  const cols = region!.cols;
  const rValues = new Float32Array(rows * cols);
  const gValues = new Float32Array(rows * cols);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const texIdx = ((region!.y0 + r) * w + (region!.x0 + c)) * 2;
      rValues[r * cols + c] = atlas!.data[texIdx]!;
      gValues[r * cols + c] = atlas!.data[texIdx + 1]!;
    }
  }
  return { region: region!, rValues, gValues };
}

function assertCleanAtlasAndU25(
  scene: UniSpatialScene,
  expectedViewMode: 'refinement' | 'output_edit',
  expectedSize: number,
  expectedBudget: number,
): void {
  expect(scene.viewMode).toBe(expectedViewMode);
  expect(scene.valueAtlas).not.toBeNull();
  expect(scene.valueAtlas!.width).toBe(expectedSize);
  expect(scene.valueAtlas!.height).toBe(expectedSize);
  expect(scene.valueAtlas!.data.byteLength).toBe(expectedBudget);
  expect(scene.atlasKey).toMatch(new RegExp(`^${expectedViewMode}:128x128:`));

  const audit = scene.atlasAudit;
  expect(audit).not.toBeNull();
  expect(audit!.width).toBe(expectedSize);
  expect(audit!.height).toBe(expectedSize);
  expect(audit!.byteLength).toBe(expectedBudget);
  expect(audit!.withinBudget).toBe(true);
  expect(audit!.regions.length).toBeGreaterThanOrEqual(5);
  expect(audit!.regions.length).toBeLessThanOrEqual(14);
  expect(audit!.hasOverlap).toBe(false);
  expect(audit!.overlapPair).toBeNull();
  expect(audit!.outOfBoundsIds.length).toBe(0);

  // U25 compliance: every sheetKind in {1, 2} has positive TexRect
  for (let i = 0; i < scene.sheetCount; i++) {
    const base = i * SHEET_INSTANCE_FLOATS;
    const sheetKind = Math.round(scene.sheetData[base + 8]!);
    if (sheetKind === SHEET_KIND_CAUSAL_SCORES || sheetKind === SHEET_KIND_CAUSAL_SOFTMAX) {
      expect(scene.sheetData[base + 15]!).toBeGreaterThan(0);
      expect(scene.sheetData[base + 16]!).toBeGreaterThan(0);
    }
  }

  expect(validateSceneGeometryFinite(scene).valid).toBe(true);
}

function assertValidRgbPreviews(
  scene: UniSpatialScene,
  minCount: number,
): void {
  expect(scene.rgbPreviews.length).toBeGreaterThanOrEqual(minCount);
  for (const preview of scene.rgbPreviews) {
    expect(preview.rgbChw.length).toBe(3 * preview.width * preview.height);
    expect(preview.honestyBadge).toBe(UNI_SCENE_VISUAL_HONESTY_BADGE);
    for (let i = 0; i < preview.rgbChw.length; i++) {
      const v = preview.rgbChw[i]!;
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  }
}

function assertAllGZero(gValues: Float32Array): void {
  for (let i = 0; i < gValues.length; i++) {
    expect(gValues[i]).toBe(0.0);
  }
}

describe('UNI5 Views 5 (refinement) & 6 (output_edit)', () => {
  describe('View 5 — Image Refinement Canvas (refinementCanvasScene.ts)', () => {
    it('builds T2I refinement scene with 128x128 RG32F, 4x6 plates, U9/U10/U12, 2-way CFG slices, decoded 16x16 RGB & PickBox round-trip', () => {
      const session = createCompletedSession({
        taskMode: 'text_to_image',
        activationSteps: 8,
        refinementSteps: 4,
      });

      const scene = buildUniRefinementCanvasScene(session);
      assertCleanAtlasAndU25(
        scene,
        'refinement',
        UNI_REFINEMENT_ATLAS_SIZE,
        UNI_REFINEMENT_ATLAS_BUDGET_BYTES,
      );
      assertValidRgbPreviews(scene, 1);

      // Synced 16x16 RGB decoded preview (0 RGB in RG32F)
      expect(scene.rgbPreviews.length).toBe(1);
      expect(scene.rgbPreviews[0]!.kind).toBe('decoded');
      expect(scene.rgbPreviews[0]!.width).toBe(16);
      expect(scene.rgbPreviews[0]!.height).toBe(16);

      // Verify 4x6 plates, 4x4 stratified order, 6x8 parents, 4x8 fixed prefix, and CFG slices
      expect(readRegionTexels(scene, 'vq_code_stage').region).toMatchObject({ rows: 4, cols: 6 });
      expect(readRegionTexels(scene, 'vq_paper_age').region).toMatchObject({ rows: 4, cols: 6 });
      expect(readRegionTexels(scene, 'vq_tree_level').region).toMatchObject({ rows: 4, cols: 6 });
      expect(readRegionTexels(scene, 'vq_mask_noise').region).toMatchObject({ rows: 4, cols: 6 });
      expect(readRegionTexels(scene, 'stratified_order').region).toMatchObject({ rows: 4, cols: 4 });
      expect(readRegionTexels(scene, 'vq_tree_embeddings').region).toMatchObject({ rows: 24, cols: 8 });
      expect(readRegionTexels(scene, 'parent_tokens').region).toMatchObject({ rows: 6, cols: 8 });
      expect(readRegionTexels(scene, 'fixed_prefix_tokens').region).toMatchObject({ rows: 4, cols: 8 });

      // G = 0.0 on all active scalar, age, level, noise, stratified, embedding, parent, prefix, and logit plates
      assertAllGZero(readRegionTexels(scene, 'vq_paper_age').gValues);
      assertAllGZero(readRegionTexels(scene, 'vq_tree_level').gValues);
      assertAllGZero(readRegionTexels(scene, 'vq_mask_noise').gValues);
      assertAllGZero(readRegionTexels(scene, 'stratified_order').gValues);
      assertAllGZero(readRegionTexels(scene, 'vq_tree_embeddings').gValues);
      assertAllGZero(readRegionTexels(scene, 'parent_tokens').gValues);
      assertAllGZero(readRegionTexels(scene, 'fixed_prefix_tokens').gValues);
      assertAllGZero(readRegionTexels(scene, 'cfg_branch_0_logits').gValues);
      assertAllGZero(readRegionTexels(scene, 'cfg_branch_1_logits').gValues);
      assertAllGZero(readRegionTexels(scene, 'cfg_branch_2_logits').gValues);
      assertAllGZero(readRegionTexels(scene, 'cfg_guided_logits').gValues);
      // Completed T2I run has 0 unactivated stage === -1 cells -> G = 0.0 everywhere on vq_code_stage
      assertAllGZero(readRegionTexels(scene, 'vq_code_stage').gValues);

      // PickBox round-trip on CFG plates (cfgBatchRow) and 4x6 grid cells (row, col)
      const cfg1Box = scene.pickBoxes.find((p) => p.tensorKey === 'cfg_branch_1_logits')!;
      expect(cfg1Box).toBeDefined();
      const targetWorld = localToWorldOnFrame(
        cfg1Box.frame!,
        (10 + 0.5) / cfg1Box.gridCols! - 0.5,
        0.5,
        (7 + 0.5) / cfg1Box.gridRows! - 0.5,
      );
      const ray: Ray = {
        origin: [targetWorld[0], targetWorld[1] + 4.0, targetWorld[2]],
        direction: [0, -1, 0],
      };
      const hit = pickMacroBay(ray, scene.pickBoxes, []);
      expect(hit).not.toBeNull();
      const cell = pickHitToUniCell(hit, 'refinement');
      expect(cell).not.toBeNull();
      expect(cell!.cfgBatchRow).toBe(1);
      expect(cell!.row).toBe(7);
      expect(cell!.col).toBe(10);
      expect(cell!.branch).toBe('g');
    });

    it('supports 3-branch Editing (harbor_panorama_16x8), historical recordId scrubbing, selectedGridCell, and showStratifiedOrder', () => {
      const session = createCompletedSession({
        taskMode: 'image_edit',
        scenePresetId: 'harbor_panorama_16x8',
        activationSteps: 8,
        refinementSteps: 4,
      });

      const branch = session.getBranch(session.getActiveBranchId()!)!;
      expect(branch.cfgBatchSize).toBe(3);
      expect(branch.recordIds.length).toBe(12); // prefill + 11 denoise rounds

      // Scrub across step 0 (prefill), step 2 (early denoise), and final step 11
      const prefillScene = buildUniRefinementCanvasScene(session, {
        recordId: branch.recordIds[0]!,
      });
      const earlyScene = buildUniRefinementCanvasScene(session, {
        recordId: branch.recordIds[2]!,
        selectedGridCell: 5,
        showStratifiedOrder: true,
        cfgBranchRow: 2,
      });
      const finalScene = buildUniRefinementCanvasScene(session, {
        recordId: branch.recordIds[11]!,
        selectedGridCell: 5,
        showStratifiedOrder: false,
      });

      assertCleanAtlasAndU25(prefillScene, 'refinement', 128, 131_072);
      assertCleanAtlasAndU25(earlyScene, 'refinement', 128, 131_072);
      assertCleanAtlasAndU25(finalScene, 'refinement', 128, 131_072);
      assertValidRgbPreviews(earlyScene, 1);
      assertValidRgbPreviews(finalScene, 1);

      expect(prefillScene.atlasKey).not.toBe(earlyScene.atlasKey);
      expect(earlyScene.atlasKey).not.toBe(finalScene.atlasKey);

      // In prefill (step 0) of harbor_panorama_16x8, 8 border cells are locked (stage=4, G=0)
      // and 8 interior content cells are unactivated (stage=-1, G=1.0), + 8 markers (G=0)
      const prefillCodeStage = readRegionTexels(prefillScene, 'vq_code_stage');
      const maskedCountPrefill = Array.from(prefillCodeStage.gValues).filter((g) => g > 0.5).length;
      expect(maskedCountPrefill).toBe(8);

      // In final step, 0 cells have stage === -1
      const finalCodeStage = readRegionTexels(finalScene, 'vq_code_stage');
      assertAllGZero(finalCodeStage.gValues);

      // selectedGridCell adds readout anchor and flow highlight geometry
      expect(
        earlyScene.overlayAnchors.some((a) => a.id === 'selected_cell_readout'),
      ).toBe(true);

      // showStratifiedOrder toggles stratified_order sheet highlight & flow geometry
      const stratIdx = earlyScene.pickBoxes.findIndex((p) => p.tensorKey === 'stratified_order');
      expect(Math.round(earlyScene.sheetData[stratIdx * SHEET_INSTANCE_FLOATS + 11]!)).not.toBe(
        SHEET_HL_DIMMED,
      );
      expect(Math.round(finalScene.sheetData[stratIdx * SHEET_INSTANCE_FLOATS + 11]!)).toBe(
        SHEET_HL_DIMMED,
      );

      // PickBox on cfg_branch_2_logits decodes cfgBatchRow = 2
      const cfg2Box = earlyScene.pickBoxes.find((p) => p.tensorKey === 'cfg_branch_2_logits')!;
      const hit = pickMacroBay(
        {
          origin: [cfg2Box.frame!.center[0], cfg2Box.frame!.center[1] + 4.0, cfg2Box.frame!.center[2]],
          direction: [0, -1, 0],
        },
        earlyScene.pickBoxes,
        [],
      );
      expect(pickHitToUniCell(hit, 'refinement')!.cfgBatchRow).toBe(2);
    });

    it('gracefully falls back on Text Understanding and Object Grounding sessions without throwing', () => {
      const textSession = createCompletedSession({
        taskMode: 'text_understanding',
        maxNewTokens: 4,
        blockLength: 4,
      });
      const groundSession = createCompletedSession({
        taskMode: 'object_grounding',
        scenePresetId: 'sunlit_cat_16x16',
        groundingSteps: 2,
      });

      const textFallback = buildUniRefinementCanvasScene(textSession);
      const groundFallback = buildUniRefinementCanvasScene(groundSession);

      assertCleanAtlasAndU25(textFallback, 'refinement', 128, 131_072);
      assertCleanAtlasAndU25(groundFallback, 'refinement', 128, 131_072);
      assertValidRgbPreviews(textFallback, 1);
      assertValidRgbPreviews(groundFallback, 1);

      expect(
        textFallback.provenanceBadges.some(
          (b) => b.tone === 'muted' && b.label.includes('Refinement Preview Fallback'),
        ),
      ).toBe(true);
      expect(
        groundFallback.provenanceBadges.some(
          (b) => b.tone === 'muted' && b.label.includes('Refinement Preview Fallback'),
        ),
      ).toBe(true);
    });
  });

  describe('View 6 — Output, Grounding & Editing Pipeline (outputEditScene.ts)', () => {
    it('builds T2I output_edit scene with expand (24x8), gen_predictor, stripped content (16x32), 16x16 luma (G=0), and decoded RGB', () => {
      const session = createCompletedSession({
        taskMode: 'text_to_image',
        activationSteps: 8,
        refinementSteps: 4,
      });

      const scene = buildUniOutputEditScene(session);
      assertCleanAtlasAndU25(
        scene,
        'output_edit',
        UNI_OUTPUT_EDIT_ATLAS_SIZE,
        UNI_OUTPUT_EDIT_ATLAS_BUDGET_BYTES,
      );
      assertValidRgbPreviews(scene, 1);

      expect(scene.rgbPreviews.length).toBe(1);
      expect(scene.rgbPreviews[0]!.kind).toBe('decoded');
      expect(scene.rgbPreviews[0]!.width).toBe(16);
      expect(scene.rgbPreviews[0]!.height).toBe(16);

      expect(readRegionTexels(scene, 'codec_expand_children').region).toMatchObject({
        rows: 24,
        cols: 8,
      });
      expect(readRegionTexels(scene, 'gen_predictor_top32').region).toMatchObject({
        rows: 24,
        cols: 32,
      });
      expect(readRegionTexels(scene, 'stripped_content_logits').region).toMatchObject({
        rows: 16,
        cols: 32,
      });
      expect(readRegionTexels(scene, 'decoded_luma_16x16').region).toMatchObject({
        rows: 16,
        cols: 16,
      });

      assertAllGZero(readRegionTexels(scene, 'codec_expand_children').gValues);
      assertAllGZero(readRegionTexels(scene, 'gen_predictor_top32').gValues);
      assertAllGZero(readRegionTexels(scene, 'stripped_content_logits').gValues);
      assertAllGZero(readRegionTexels(scene, 'decoded_luma_16x16').gValues);
      assertAllGZero(readRegionTexels(scene, 'u18_padding_lock_mask').gValues);
    });

    it('builds 3-branch Editing (harbor_panorama_16x8) scene with source, letterboxed (8/16 U18 locked), decoded, cropped (16x8), and legacy/independent/ip2p Dual-CFG slices', () => {
      const session = createCompletedSession({
        taskMode: 'image_edit',
        scenePresetId: 'harbor_panorama_16x8',
        activationSteps: 8,
        refinementSteps: 4,
        guidanceMode: 'legacy',
      });

      const scene = buildUniOutputEditScene(session, {
        editGuidanceMode: 'ip2p',
        cropToSource: true,
      });
      assertCleanAtlasAndU25(scene, 'output_edit', 128, 131_072);
      assertValidRgbPreviews(scene, 4);

      const kinds = scene.rgbPreviews.map((p) => p.kind);
      expect(kinds).toEqual(['source', 'letterboxed', 'decoded', 'cropped']);

      const src = scene.rgbPreviews.find((p) => p.kind === 'source')!;
      const lb = scene.rgbPreviews.find((p) => p.kind === 'letterboxed')!;
      const dec = scene.rgbPreviews.find((p) => p.kind === 'decoded')!;
      const cropped = scene.rgbPreviews.find((p) => p.kind === 'cropped')!;

      expect(src.width).toBe(16);
      expect(src.height).toBe(8);
      expect(lb.width).toBe(16);
      expect(lb.height).toBe(16);
      expect(lb.paddingLockMask).toBeDefined();
      expect(lb.paddingLockMask!.lockedCellCount).toBe(8);
      expect(lb.paddingLockMask!.editableCellCount).toBe(8);
      expect(dec.width).toBe(16);
      expect(dec.height).toBe(16);
      expect(cropped.width).toBe(16);
      expect(cropped.height).toBe(8);

      // U18 paddingLockMask RG32F plate has G = 1.0 ONLY on the 8 locked border cells!
      const lockPlate = readRegionTexels(scene, 'u18_padding_lock_mask');
      expect(lockPlate.region).toMatchObject({ rows: 4, cols: 4 });
      let gLocked = 0;
      for (let i = 0; i < 16; i++) {
        const expectedG = lb.paddingLockMask!.lockMask[i] === 1 ? 1.0 : 0.0;
        expect(lockPlate.gValues[i]).toBe(expectedG);
        if (expectedG === 1.0) gLocked += 1;
      }
      expect(gLocked).toBe(8);

      // Scalar luma 16x16 plate and all 3 Dual-CFG mixer plates keep G = 0.0
      assertAllGZero(readRegionTexels(scene, 'decoded_luma_16x16').gValues);
      const legacyPlate = readRegionTexels(scene, 'cfg_mixer_legacy');
      const indepPlate = readRegionTexels(scene, 'cfg_mixer_independent');
      const ip2pPlate = readRegionTexels(scene, 'cfg_mixer_ip2p');
      assertAllGZero(legacyPlate.gValues);
      assertAllGZero(indepPlate.gValues);
      assertAllGZero(ip2pPlate.gValues);

      // Live Dual-CFG mixer formulas produce distinct logit slices across legacy/independent/ip2p
      let maxLegacyIndepDiff = 0;
      let maxLegacyIp2pDiff = 0;
      for (let i = 0; i < legacyPlate.rValues.length; i++) {
        maxLegacyIndepDiff = Math.max(
          maxLegacyIndepDiff,
          Math.abs(legacyPlate.rValues[i]! - indepPlate.rValues[i]!),
        );
        maxLegacyIp2pDiff = Math.max(
          maxLegacyIp2pDiff,
          Math.abs(legacyPlate.rValues[i]! - ip2pPlate.rValues[i]!),
        );
      }
      expect(maxLegacyIndepDiff).toBeGreaterThan(1e-4);
      expect(maxLegacyIp2pDiff).toBeGreaterThan(1e-4);

      // editGuidanceMode: 'ip2p' highlights cfg_mixer_ip2p with SHEET_HL_ACTIVE
      const ip2pIdx = scene.pickBoxes.findIndex((p) => p.tensorKey === 'cfg_mixer_ip2p');
      expect(Math.round(scene.sheetData[ip2pIdx * SHEET_INSTANCE_FLOATS + 11]!)).toBe(
        SHEET_HL_ACTIVE,
      );

      // cropToSource: false omits cropped preview
      const uncropped = buildUniOutputEditScene(session, { cropToSource: false });
      expect(uncropped.rgbPreviews.map((p) => p.kind)).toEqual([
        'source',
        'letterboxed',
        'decoded',
      ]);

      // PickBox round-trip on Dual-CFG mixer slices decodes cfgBatchRow, row, col
      const cfgIp2pBox = scene.pickBoxes[ip2pIdx]!;
      const hitPoint = localToWorldOnFrame(
        cfgIp2pBox.frame!,
        (14 + 0.5) / 32 - 0.5,
        0.5,
        (9 + 0.5) / 24 - 0.5,
      );
      const hit = pickMacroBay(
        {
          origin: [hitPoint[0], hitPoint[1] + 4.0, hitPoint[2]],
          direction: [0, -1, 0],
        },
        scene.pickBoxes,
        [],
      );
      const cell = pickHitToUniCell(hit, 'output_edit');
      expect(cell).not.toBeNull();
      expect(cell!.cfgBatchRow).toBe(2);
      expect(cell!.row).toBe(9);
      expect(cell!.col).toBe(14);
      expect(cell!.tensorKey).toBe('cfg_mixer_ip2p');
    });

    it('builds Text Understanding output_edit scene (B=4, shift_logits=true) with G=1.0 ONLY on still-masked [MASK] positions', () => {
      const session = new UniSession(sharedBundle);
      const branch = session.startBranch({
        taskMode: 'text_understanding',
        blockLength: 4,
        stepsPerBlock: 4,
        maxNewTokens: 4,
        confidenceThreshold: 1.0, // force 1 slot per step so intermediate step still has [MASK]
      });
      // Step 1 leaves 3 or 2 [MASK] slots remaining
      const step1 = session.stepBranch(branch.branchId)!;
      expect(step1.kind).toBe('text_block_step');

      const scene = buildUniOutputEditScene(session, { recordId: step1.recordId });
      assertCleanAtlasAndU25(scene, 'output_edit', 128, 131_072);
      expect(scene.rgbPreviews).toEqual([]);

      const logitsPlate = readRegionTexels(scene, 'shifted_diffusion_logits');
      const probsPlate = readRegionTexels(scene, 'shifted_diffusion_probs');
      const statePlate = readRegionTexels(scene, 'unmask_mask_state');
      const stripPlate = readRegionTexels(scene, 'token_slot_mask_strip');

      expect(logitsPlate.region).toMatchObject({ rows: 4, cols: 64 });
      expect(probsPlate.region).toMatchObject({ rows: 4, cols: 64 });
      assertAllGZero(logitsPlate.gValues);
      assertAllGZero(probsPlate.gValues);

      // G = 1.0 ONLY on still-masked [MASK] positions!
      const tokens = step1.textState!.tokenIds;
      let maskedRows = 0;
      for (let r = 0; r < 4; r++) {
        const isMasked = tokens[r] === UNI_MINI_MASK_TOKEN_ID;
        if (isMasked) maskedRows += 1;
        const expectedG = isMasked ? 1.0 : 0.0;
        expect(stripPlate.gValues[r]).toBe(expectedG);
        for (let c = 0; c < 4; c++) {
          expect(statePlate.gValues[r * 4 + c]).toBe(expectedG);
        }
      }
      expect(maskedRows).toBeGreaterThan(0);

      // PickBox round-trip on shifted_diffusion_logits
      const logitsBox = scene.pickBoxes.find((p) => p.tensorKey === 'shifted_diffusion_logits')!;
      const pt = localToWorldOnFrame(
        logitsBox.frame!,
        (21 + 0.5) / 64 - 0.5,
        0.5,
        (2 + 0.5) / 4 - 0.5,
      );
      const hit = pickMacroBay(
        { origin: [pt[0], pt[1] + 4.0, pt[2]], direction: [0, -1, 0] },
        scene.pickBoxes,
        [],
      );
      const cell = pickHitToUniCell(hit, 'output_edit');
      expect(cell!.row).toBe(2);
      expect(cell!.col).toBe(21);
      expect(cell!.branch).toBe('u');
    });

    it('resolves multi-block Understanding slot-0 shifted logits from preceding committed block causal tail (not prefill)', () => {
      const session = new UniSession(sharedBundle);
      const branch = session.startBranch({
        taskMode: 'text_understanding',
        blockLength: 4,
        stepsPerBlock: 4,
        maxNewTokens: 8,
        confidenceThreshold: 1.0,
      });

      const prefillId = branch.recordIds[0]!;
      // Step until Block #1 commits and Block #2 begins (with causalContext=true, slot 0 is
      // pre-seeded so Block #1 commits after 3 steps; step until positionIds[0] advances)
      const firstBlockStep = session.stepBranch(branch.branchId)!;
      const block1StartPos = firstBlockStep.positionIds[0]!;
      let prevRec = firstBlockStep;
      let block2Step1 = firstBlockStep;
      for (let i = 0; i < 8; i++) {
        const nextRec = session.stepBranch(branch.branchId)!;
        if (nextRec.positionIds[0]! > block1StartPos) {
          block2Step1 = nextRec;
          break;
        }
        prevRec = nextRec;
      }
      const block1Final = prevRec;
      expect(block2Step1.kind).toBe('text_block_step');
      expect(block2Step1.positionIds[0]!).toBeGreaterThan(block1Final.positionIds[0]!);

      // Compute independent ground-truth causal tail logits from Block #1 commit vs prefill tail
      const vocabSize = sharedBundle.core.config.vocabSize;
      const prefillHead = session.reconstructStepHeadLogits(prefillId).perBranchTextLogits[0]!;
      const prefillTail = prefillHead.subarray(prefillHead.length - vocabSize, prefillHead.length);

      const updatedBranch = session.getBranch(branch.branchId)!;
      const { pastKvs, pastRouting } = extractRowPastKvs(
        cropUniKvCache(updatedBranch.prefixCache, block1StartPos),
        0,
      );
      const block1Commit = runUniForwardPass(sharedBundle.core, {
        tokenIds: block1Final.textState!.tokenIds,
        maskMode: 'causal',
        useCache: false,
        prefillMode: 'hf_faithful',
        downstreamTarget: 'u',
        pastKvs,
        pastRouting,
        computeTextHead: true,
        computeGenHead: false,
        retainedTraceLayers: new Set(),
      });
      const block1Tail = block1Commit.textLogits!.subarray(
        block1Commit.textLogits!.length - vocabSize,
        block1Commit.textLogits!.length,
      );

      const scene = buildUniOutputEditScene(session, { recordId: block2Step1.recordId });
      assertCleanAtlasAndU25(scene, 'output_edit', 128, 131_072);
      const logitsPlate = readRegionTexels(scene, 'shifted_diffusion_logits');
      const slot0Row = logitsPlate.rValues.subarray(0, vocabSize);

      let maxErrVsBlock1Tail = 0;
      let maxErrVsPrefillTail = 0;
      for (let v = 0; v < vocabSize; v++) {
        maxErrVsBlock1Tail = Math.max(maxErrVsBlock1Tail, Math.abs(slot0Row[v]! - block1Tail[v]!));
        maxErrVsPrefillTail = Math.max(
          maxErrVsPrefillTail,
          Math.abs(slot0Row[v]! - prefillTail[v]!),
        );
      }
      expect(maxErrVsBlock1Tail).toBe(0);
      expect(maxErrVsPrefillTail).toBeGreaterThan(1e-3);
    });

    it('builds Object Grounding & Visual Understanding output_edit scenes with groundingBoxes (<LOC_0..16>) and explicit U27 badge', () => {
      const groundSession = createCompletedSession({
        taskMode: 'object_grounding',
        scenePresetId: 'harbor_panorama_16x8',
        groundingTags: ['sun', 'water'],
        groundingSteps: 2,
      });
      const visualSession = createCompletedSession({
        taskMode: 'visual_understanding',
        scenePresetId: 'sunlit_cat_16x16',
        maxNewTokens: 4,
      });

      const groundScene = buildUniOutputEditScene(groundSession);
      const visualScene = buildUniOutputEditScene(visualSession);

      for (const s of [groundScene, visualScene]) {
        assertCleanAtlasAndU25(s, 'output_edit', 128, 131_072);
        assertValidRgbPreviews(s, 1);

        const gPreview = s.rgbPreviews.find((p) => p.kind === 'grounding');
        expect(gPreview).toBeDefined();
        expect(gPreview!.groundingBoxes).toBeDefined();
        expect(gPreview!.groundingBoxes!.length).toBeGreaterThanOrEqual(1);

        for (const box of gPreview!.groundingBoxes!) {
          for (let k = 0; k < 4; k++) {
            expect(box.locTokenIds[k]!).toBeGreaterThanOrEqual(UNI_MINI_LOC_START_ID);
            expect(box.locTokenIds[k]!).toBeLessThanOrEqual(UNI_MINI_LOC_END_ID);
            expect(box.locTokenStrings[k]!).toMatch(/^<LOC_\d+>$/);
            expect(box.normalizedBins[k]!).toBeGreaterThanOrEqual(0);
            expect(box.normalizedBins[k]!).toBeLessThanOrEqual(16);
          }
          expect(box.pixelBox[0]).toBeGreaterThanOrEqual(0);
          expect(box.pixelBox[2]).toBeLessThanOrEqual(gPreview!.width);
          expect(box.pixelBox[1]).toBeGreaterThanOrEqual(0);
          expect(box.pixelBox[3]).toBeLessThanOrEqual(gPreview!.height);
        }

        // Explicit U27 provenance badge noting skip_special_tokens=True strips <LOC_*> tokens
        expect(
          s.provenanceBadges.some(
            (b) => b.label.includes('U27') && b.label.includes('skip_special_tokens=True'),
          ),
        ).toBe(true);
      }

      // Harbor panorama grounding carries 2 boxes mapped into 16x8 pixel space
      const gBoxes = groundScene.rgbPreviews[0]!.groundingBoxes!;
      expect(gBoxes.length).toBe(2);
      expect(groundScene.rgbPreviews[0]!.width).toBe(16);
      expect(groundScene.rgbPreviews[0]!.height).toBe(8);
    });

    it('is deterministic, non-mutating, and round-trips every PickBox on both View 5 and View 6', () => {
      const editSession = createCompletedSession({
        taskMode: 'image_edit',
        scenePresetId: 'harbor_panorama_16x8',
        activationSteps: 8,
        refinementSteps: 4,
      });

      const activeBefore = editSession.getActiveRecordId();
      const countBefore = editSession.getAllRecords().length;

      const v5a = buildUniRefinementCanvasScene(editSession);
      const v5b = buildUniRefinementCanvasScene(editSession);
      const v6a = buildUniOutputEditScene(editSession);
      const v6b = buildUniOutputEditScene(editSession);

      expect(editSession.getActiveRecordId()).toBe(activeBefore);
      expect(editSession.getAllRecords().length).toBe(countBefore);

      expect(v5a.atlasKey).toBe(v5b.atlasKey);
      expect(v5a.valueAtlas!.data).toEqual(v5b.valueAtlas!.data);
      expect(v6a.atlasKey).toBe(v6b.atlasKey);
      expect(v6a.valueAtlas!.data).toEqual(v6b.valueAtlas!.data);

      for (const [viewMode, scene] of [
        ['refinement', v5a],
        ['output_edit', v6a],
      ] as const) {
        for (const box of scene.pickBoxes) {
          expect(box.frame).toBeDefined();
          const c = box.frame!.center;
          const hit = pickMacroBay(
            { origin: [c[0], c[1] + 5.0, c[2]], direction: [0, -1, 0] },
            scene.pickBoxes,
            [],
          );
          expect(hit).not.toBeNull();
          expect(hit!.tensorKey).toBe(box.tensorKey);
          const id = pickHitToUniCell(hit, viewMode);
          expect(id).not.toBeNull();
          expect(id!.viewMode).toBe(viewMode);
          expect(id!.row).toBeGreaterThanOrEqual(0);
          expect(id!.row!).toBeLessThan(box.gridRows!);
          expect(id!.col).toBeGreaterThanOrEqual(0);
          expect(id!.col!).toBeLessThan(box.gridCols!);
        }
      }
    });
  });
});
